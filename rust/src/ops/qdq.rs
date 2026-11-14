//! Private Q/DQ matrix-product recognition. Quantized bytes are device-kernel inputs;
//! dequantized tensors remain available to other consumers and graph outputs.

use std::collections::HashMap;

use super::float8::Format;
use super::matmul::fused::{self, KernelKind, Routing};
use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::mlx::metal::{self, Kernel};
use crate::registry::{CompileShapeSafety, NodeView};
use crate::sys::{mlx, ort};

const F32: mlx::mlx_dtype = mlx::mlx_dtype__MLX_FLOAT32;
const FLOAT: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
const EAGER_REASON: &str =
    "QDQ products validate live geometry and specialize Metal kernel layouts";

#[derive(Clone)]
pub(crate) struct QuantizedMatMul {
    x: Option<NodeDesc>,
    w: NodeDesc,
}

fn dq_signature(
    domain: &str,
    op: &str,
    version: i32,
    inputs: usize,
    outputs: usize,
    dtype: ort::ONNXTensorElementDataType,
) -> bool {
    domain.is_empty()
        && op == "DequantizeLinear"
        && version >= 10
        && (2..=3).contains(&inputs)
        && outputs == 1
        && dtype == FLOAT
}

fn dq_candidate(node: &NodeDesc) -> bool {
    node.outputs.first().is_some_and(|out| {
        dq_signature(
            &node.domain,
            &node.op_type,
            node.since_version,
            node.inputs.len(),
            node.outputs.len(),
            out.otype,
        )
    })
}

fn product_kind(domain: &str, op: &str, version: i32) -> Option<bool> {
    if !domain.is_empty() {
        return None;
    }
    match op {
        "MatMul" | "Gemm" => Some(false),
        "GroupedMatMul" if version == 29 => Some(true),
        _ => None,
    }
}

/// Above this many activation rows, decoding the FP8 weight once beats per-tile decode.
const FLOAT8_TILED_ROWS: i32 = 8;

/// A conservative typed subset of the runtime recognizer, after ordinary claim validation.
/// Returns whether the activation is batched (rank > 2), which only FP8 MatMul weights fuse.
pub(crate) fn partition_product_candidate(node: &NodeView) -> Option<bool> {
    let grouped = product_kind(&node.domain(), &node.op_type(), node.since_version())?;
    let rank = node.input_info(0).filter(|x| x.dtype == FLOAT)?.shape.len();
    let batched = rank > 2 && node.op_type() == "MatMul";
    (node.num_outputs() == 1
        && node.output_info(0).is_some_and(|out| out.dtype == FLOAT)
        && (rank == 2 || batched)
        && node.input_info(1).is_some_and(|input| {
            input.dtype == FLOAT && input.shape.len() == if grouped { 3 } else { 2 }
        }))
    .then_some(batched)
}

pub(crate) fn partition_float8(node: &NodeView) -> bool {
    node.input_info(0)
        .is_some_and(|data| Format::from_onnx(data.dtype).is_some())
}

pub(crate) fn partition_dq_candidate(node: &NodeView, weight: bool) -> Option<bool> {
    let data = node.input_info(0)?;
    let scale = node.input_info(1)?;
    let output = node.output_info(0)?;
    if !dq_signature(
        &node.domain(),
        &node.op_type(),
        node.since_version(),
        node.num_inputs(),
        node.num_outputs(),
        output.dtype,
    ) || scale.dtype != FLOAT
        || !(data.shape.len() == 2 || weight && data.shape.len() == 3)
    {
        return None;
    }
    // Packed and FP8 payloads have no MLX tensor dtype, so their transport requires eager regions.
    let packed = crate::engine::packed4_signed(data.dtype).is_some()
        || Format::from_onnx(data.dtype).is_some();
    (packed
        || data.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
        || data.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
        .then_some(packed)
}

pub(crate) fn recognize(nodes: &mut [NodeDesc]) {
    if std::env::var_os("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION")
        .is_some_and(|v| v != "0" && !v.is_empty())
    {
        return;
    }
    let mut producers: HashMap<String, NodeDesc> = HashMap::new();
    for node in nodes {
        if product_kind(&node.domain, &node.op_type, node.since_version).is_some()
            && node.inputs.len() >= 2
            && node.outputs.len() == 1
            && node.outputs[0].otype == FLOAT
            && let Some(w) = producers.get(&node.inputs[1].name)
        {
            node.qdq = Some(Box::new(QuantizedMatMul {
                x: producers.get(&node.inputs[0].name).cloned(),
                w: w.clone(),
            }));
            node.compile_shape_safety = CompileShapeSafety::EagerOnly {
                reason: EAGER_REASON,
            };
        }
        if dq_candidate(node) {
            producers.insert(node.outputs[0].name.clone(), node.clone());
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Parameters {
    strides: [i32; 3],
    divisors: [i32; 3],
}

impl Parameters {
    fn scalar() -> Self {
        Self {
            strides: [0; 3],
            divisors: [1; 3],
        }
    }
}

fn parameter_layout(
    dq: &NodeDesc,
    data: &[i32],
    scale: &[i32],
    weight: bool,
) -> Option<Parameters> {
    let block = dq.ints.get("block_size").copied().unwrap_or(0);
    if block == 0 && scale.iter().try_fold(1i32, |n, &d| n.checked_mul(d))? == 1 {
        return Some(Parameters::scalar());
    }
    let rank = data.len();
    let raw_axis = dq.ints.get("axis").copied().unwrap_or(1);
    let axis = if raw_axis < 0 {
        raw_axis + rank as i64
    } else {
        raw_axis
    };
    if !(0..rank as i64).contains(&axis) {
        return None;
    }
    let axis = axis as usize;
    let offset = usize::from(weight && rank == 2);
    let mut result = Parameters::scalar();
    if block == 0 {
        if dq.since_version < 13 || scale.len() != 1 || scale[0] != data[axis] {
            return None;
        }
        result.strides[offset + axis] = 1;
    } else {
        let block = i32::try_from(block).ok().filter(|&b| b > 0)?;
        if dq.since_version < 21 || scale.len() != rank {
            return None;
        }
        let mut expected = data.to_vec();
        expected[axis] = data[axis] / block + i32::from(data[axis] % block != 0);
        if expected != scale {
            return None;
        }
        let mut stride = 1i32;
        for dim in (0..rank).rev() {
            result.strides[offset + dim] = stride;
            stride = stride.checked_mul(scale[dim])?;
        }
        result.divisors[offset + axis] = block;
    }
    Some(result)
}

struct Operand {
    data: mlx::mlx_array,
    scale: mlx::mlx_array,
    zero: mlx::mlx_array,
    shape: Vec<i32>,
    parameters: Parameters,
    packed: i32,
    float8: Option<Format>,
}

fn integer8(dtype: mlx::mlx_dtype) -> bool {
    matches!(dtype, mlx::mlx_dtype__MLX_INT8 | mlx::mlx_dtype__MLX_UINT8)
}

fn operand(
    ctx: &mut TranslationContext,
    dq: &NodeDesc,
    weight: bool,
) -> Result<Option<Operand>, MlxError> {
    super::quant::validate_packed_dq(ctx, dq)?;
    let packed = ctx.resolve_packed4(&dq.inputs[0])?;
    let float8 = if packed.is_none() {
        ctx.resolve_float8(&dq.inputs[0])?
    } else {
        None
    };
    let (data, shape, packing) = if let Some(packed) = packed {
        (packed.data, packed.shape, if packed.signed { 2 } else { 1 })
    } else if let Some((bits, _)) = float8 {
        (bits, ctx.shape_of(bits), 0)
    } else {
        let data = ctx.resolve(&dq.inputs[0])?;
        (data, ctx.shape_of(data), 0)
    };
    let scale = ctx.resolve(&dq.inputs[1])?;
    let float8 = float8.map(|(_, format)| format);
    if (packing == 0 && float8.is_none() && !integer8(ctx.dtype_of(data)))
        || ctx.dtype_of(scale) != F32
        || !(shape.len() == 2 || (weight && shape.len() == 3))
    {
        return Ok(None);
    }
    let scale_shape = ctx.shape_of(scale);
    let Some(parameters) = parameter_layout(dq, &shape, &scale_shape, weight) else {
        return Ok(None);
    };
    let zero = if dq.inputs.get(2).is_some_and(|r| r.source != Src::Absent) {
        if packing != 0 {
            let Some(zero) = ctx.resolve_packed4(&dq.inputs[2])? else {
                return Err("QDQ MatMul: packed zero point must match data type".into());
            };
            if zero.signed != (packing == 2) || zero.shape != scale_shape {
                return Err(
                    "QDQ MatMul: packed zero point must match data type and scale shape".into(),
                );
            }
            zero.data
        } else if let Some(format) = float8 {
            match ctx.resolve_float8(&dq.inputs[2])? {
                Some((zero, zero_format))
                    if zero_format == format && ctx.shape_of(zero) == scale_shape =>
                {
                    zero
                }
                _ => return Ok(None),
            }
        } else {
            let zero = ctx.resolve(&dq.inputs[2])?;
            if ctx.dtype_of(zero) != ctx.dtype_of(data) || ctx.shape_of(zero) != scale_shape {
                return Ok(None);
            }
            zero
        }
    } else {
        ctx.scalar_i32(0)
    };
    Ok(Some(Operand {
        data: ctx.contiguous(data)?,
        scale: ctx.contiguous(scale)?,
        zero: ctx.contiguous(zero)?,
        shape,
        parameters,
        packed: packing,
        float8,
    }))
}

fn binary_scales(dq: &NodeDesc) -> bool {
    let scale = &dq.inputs[1];
    if scale.source != Src::Initializer {
        return false;
    }
    let Some(init) = &scale.init else {
        return false;
    };
    let Some(bytes) = &init.owned else {
        return false;
    };
    if init.dtype != FLOAT || init.count == 0 || init.count.checked_mul(4) != Some(bytes.len()) {
        return false;
    }
    bytes.as_chunks::<4>().0.iter().all(|b| {
        let bits = u32::from_ne_bytes(*b);
        bits & 0x807fffff == 0 && (87..=167).contains(&((bits >> 23) & 0xff))
    })
}

fn integer_exact(
    pattern: &QuantizedMatMul,
    x: &Operand,
    w: &Operand,
    trans_a: bool,
    trans_b: bool,
) -> bool {
    // Each centered int8/uint8 value is in [-255,255]. For K<=256 every partial sum's
    // absolute value is <=16,646,400<2^24, hence exact in int32 AND float32. Binary
    // scales in [2^-40,2^40] keep every scaled product/partial normal and finite.
    (0..=256).contains(&x.shape[usize::from(!trans_a)])
        && x.parameters.strides[usize::from(!trans_a)] == 0
        && w.parameters.strides[1 + usize::from(trans_b)] == 0
        && pattern.x.as_ref().is_some_and(binary_scales)
        && binary_scales(&pattern.w)
}

const HEADER: &str = r#"
inline float2 mlx_qdq_two_sum(float a, float b) {
    float hi = precise::fma(1.0f, a, b);
    float bv = precise::fma(-1.0f, a, hi);
    float av = precise::fma(-1.0f, bv, hi);
    float br = precise::fma(-1.0f, bv, b);
    float ar = precise::fma(-1.0f, av, a);
    return float2(hi, precise::fma(1.0f, ar, br));
}
"#;

const SOURCE: &str = r#"
    size_t out = thread_position_in_grid.x / 32;
    uint lane = thread_position_in_grid.x % 32;
    if (out >= size_t(M) * S * N) { return; }
    uint col = out % N;
    size_t route = out / N;
    uint row = route / S;
    uint expert = GROUPED ? uint(indices[route]) : 0;
    float sum = 0.0f;
    float hi = 0.0f;
    float lo = 0.0f;
    bool compensated = true;
    int isum = 0;
    for (uint k = lane; k < K; k += 32) {
        uint xr = TRANS_A ? k : row;
        uint xc = TRANS_A ? row : k;
        uint wr = TRANS_B ? col : k;
        uint wc = TRANS_B ? k : col;
        size_t xp = size_t(xr / XD0) * XS0 + size_t(xc / XD1) * XS1;
        size_t wp = size_t(expert / WD0) * WS0 + size_t(wr / WD1) * WS1
                  + size_t(wc / WD2) * WS2;
        size_t xi = TRANS_A ? size_t(k) * M + row : size_t(row) * K + k;
        size_t wi = TRANS_B ? (size_t(expert) * N + col) * K + k
                           : (size_t(expert) * K + k) * N + col;
        int aq = 0;
        int az = 0;
        if (X_QUANT) {
            aq = X_PACKED ? ((int(x[xi / 2]) >> (4 * (xi % 2))) & 15) : int(x[xi]);
            aq = X_PACKED == 2 ? (aq ^ 8) - 8 : aq;
            if (X_ZERO) {
                az = X_PACKED ? ((int(xzero[xp / 2]) >> (4 * (xp % 2))) & 15) : int(xzero[xp]);
                az = X_PACKED == 2 ? (az ^ 8) - 8 : az;
            }
        }
        int bq = W_PACKED ? ((int(w[wi / 2]) >> (4 * (wi % 2))) & 15) : int(w[wi]);
        bq = W_PACKED == 2 ? (bq ^ 8) - 8 : bq;
        int bz = 0;
        if (W_ZERO) {
            bz = W_PACKED ? ((int(wzero[wp / 2]) >> (4 * (wp % 2))) & 15) : int(wzero[wp]);
            bz = W_PACKED == 2 ? (bz ^ 8) - 8 : bz;
        }
        int a = aq - az;
        int b = bq - bz;
        if (INTEGER) {
            isum += a * b;
        } else {
            // Separate, precise FP32 DQ rounding occurs before the dot, never outside it.
            float ax = X_QUANT ? precise::fma(float(a), xscale[xp], 0.0f)
                               : float(x[xi]);
            float bw = precise::fma(float(b), wscale[wp], 0.0f);
            sum = precise::fma(ax, bw, sum);
            if ((X_PACKED || W_PACKED) && compensated) {
                float product = precise::fma(ax, bw, 0.0f);
                float residual = precise::fma(ax, bw, -product);
                float2 pair = mlx_qdq_two_sum(hi, product);
                float correction = precise::fma(1.0f, lo,
                    precise::fma(1.0f, pair.y, residual));
                compensated = isfinite(ax) && isfinite(bw) && isfinite(product)
                    && isfinite(residual) && isfinite(pair.x) && isfinite(correction);
                hi = pair.x;
                lo = correction;
            }
        }
    }
    isum = simd_sum(isum);
    sum = simd_sum(sum);
    if ((X_PACKED || W_PACKED) && !INTEGER) {
        // Retain the original IEEE path if any compensated or original partial overflows.
        bool valid = simd_all(compensated) && isfinite(sum);
        for (uint offset = 16; offset != 0; offset /= 2) {
            float other_hi = simd_shuffle_down(hi, offset);
            float other_lo = simd_shuffle_down(lo, offset);
            if (valid && lane < offset) {
                float2 pair = mlx_qdq_two_sum(hi, other_hi);
                float correction = precise::fma(1.0f, lo,
                    precise::fma(1.0f, other_lo, pair.y));
                valid = isfinite(pair.x) && isfinite(correction);
                hi = pair.x;
                lo = correction;
            }
            valid = simd_all(valid);
        }
        float result = precise::fma(1.0f, hi, lo);
        valid = simd_all(valid && isfinite(result));
        if (valid) { sum = result; }
    }
    if (INTEGER) {
        size_t xp = TRANS_A ? size_t(row / XD1) * XS1 : size_t(row / XD0) * XS0;
        size_t wp = size_t(expert / WD0) * WS0
                  + (TRANS_B ? size_t(col / WD1) * WS1 : size_t(col / WD2) * WS2);
        sum = (float(isum) * xscale[xp]) * wscale[wp];
    }
    if (lane == 0) { y[out] = sum; }
"#;

struct Product<'a> {
    pattern: &'a QuantizedMatMul,
    x: &'a Operand,
    w: &'a Operand,
    m: i32,
    k: i32,
    n: i32,
    trans_a: bool,
    trans_b: bool,
    routing: Option<&'a Routing>,
}

fn has_zero(dq: &NodeDesc) -> bool {
    dq.inputs.get(2).is_some_and(|r| r.source != Src::Absent)
}

impl Product<'_> {
    fn operands(&self) -> [mlx::mlx_array; 6] {
        [
            self.x.data,
            self.w.data,
            self.x.scale,
            self.w.scale,
            self.x.zero,
            self.w.zero,
        ]
    }

    fn weight_only_header(&self) -> String {
        let Parameters {
            strides: [ws0, ws1, ws2],
            divisors: [wd0, wd1, wd2],
        } = self.w.parameters;
        let (m, k, n) = (self.m, self.k, self.n);
        let (ta, tb) = (i32::from(self.trans_a), i32::from(self.trans_b));
        let wz = i32::from(has_zero(&self.pattern.w));
        let (decoder, value, zero) = match self.w.float8 {
            Some(format) => (
                format.metal(),
                "mlx_fp8(uint(w[i]))",
                "mlx_fp8(uint(zero[p]))",
            ),
            None => (String::new(), "int(w[i])", "int(zero[p])"),
        };
        let neutral = if self.w.float8.is_some() { "0.0f" } else { "0" };
        format!(
            r#"
#include <metal_simdgroup_matrix>
{decoder}
template <typename X, typename S, typename Z>
inline float mlx_weight_x(X x, S, Z, uint row, uint k) {{
    size_t i = {ta} ? size_t(k) * {m} + row : size_t(row) * {k} + k;
    return float(x[i]);
}}
template <typename W, typename S, typename Z>
inline float mlx_weight_w(W w, S scale, Z zero, uint expert, uint k, uint col) {{
    uint wr = {tb} ? col : k, wc = {tb} ? k : col;
    size_t p = size_t(expert / {wd0}) * {ws0} + size_t(wr / {wd1}) * {ws1}
             + size_t(wc / {wd2}) * {ws2};
    size_t i = {tb} ? (size_t(expert) * {n} + col) * {k} + k
                   : (size_t(expert) * {k} + k) * {n} + col;
    return precise::fma(float({value} - ({wz} ? {zero} : {neutral})), scale[p], -0.0f);
}}
"#
        )
    }

    fn simd_templates(&self, exact: bool) -> Vec<(&'static str, i32)> {
        vec![
            ("M", self.m),
            ("K", self.k),
            ("N", self.n),
            ("S", self.routing.map_or(1, |r| r.selections)),
            ("GROUPED", i32::from(self.routing.is_some())),
            ("TRANS_A", i32::from(self.trans_a)),
            ("TRANS_B", i32::from(self.trans_b)),
            ("X_QUANT", i32::from(self.pattern.x.is_some())),
            ("X_PACKED", self.x.packed),
            ("W_PACKED", self.w.packed),
            ("INTEGER", i32::from(exact)),
            (
                "X_ZERO",
                i32::from(self.pattern.x.as_ref().is_some_and(has_zero)),
            ),
            ("W_ZERO", i32::from(has_zero(&self.pattern.w))),
            ("XS0", self.x.parameters.strides[0]),
            ("XS1", self.x.parameters.strides[1]),
            ("XD0", self.x.parameters.divisors[0]),
            ("XD1", self.x.parameters.divisors[1]),
            ("WS0", self.w.parameters.strides[0]),
            ("WS1", self.w.parameters.strides[1]),
            ("WS2", self.w.parameters.strides[2]),
            ("WD0", self.w.parameters.divisors[0]),
            ("WD1", self.w.parameters.divisors[1]),
            ("WD2", self.w.parameters.divisors[2]),
        ]
    }

    /// Vectorized FP8 decode GEMV for few activation rows; `None` defers to the tiled reader.
    fn float8_gemv(
        &self,
        ctx: &mut TranslationContext,
        shape: &[i32],
    ) -> Result<Option<(mlx::mlx_array, &'static str)>, MlxError> {
        let Some(format) = self.w.float8 else {
            return Ok(None);
        };
        let grouped = self.routing.is_some();
        let selections = self.routing.map_or(1, |r| r.selections);
        let routes = self.m * selections;
        let aligned = if self.trans_b {
            self.k % 8 == 0
        } else {
            self.n % 4 == 0
        };
        if self.trans_a || !aligned || routes > if grouped { 16 } else { 8 } {
            return Ok(None);
        }
        let Parameters {
            strides: [ws0, ws1, ws2],
            divisors: [wd0, wd1, wd2],
        } = self.w.parameters;
        // Four rows per threadgroup keeps the K-contiguous accumulators in registers.
        let (z, tile_m) = match (grouped, self.trans_b) {
            (true, _) => (routes, 1),
            (false, true) => ((self.m - 1) / 4 + 1, self.m.min(4)),
            (false, false) => (1, self.m),
        };
        let lanes = if self.trans_b { 8 } else { 4 };
        let mut templates = vec![
            ("M", self.m),
            ("K", self.k),
            ("N", self.n),
            ("S", selections),
            ("GROUPED", i32::from(grouped)),
            ("TILE_M", tile_m),
            ("W_ZERO", i32::from(has_zero(&self.pattern.w))),
            ("CHUNK_SCALE", i32::from(ws2 == 0 || wd2 % lanes == 0)),
            ("WS0", ws0),
            ("WS1", ws1),
            ("WS2", ws2),
            ("WD0", wd0),
            ("WD1", wd1),
            ("WD2", wd2),
        ];
        let header = format.metal();
        let indices = self
            .routing
            .map_or_else(|| ctx.scalar_i32(0), |r| r.indices);
        let inputs = [
            ("x", self.x.data),
            ("w", self.w.data),
            ("wscale", self.w.scale),
            ("wzero", self.w.zero),
            ("indices", indices),
        ];
        if self.trans_b {
            const ROWS: i32 = 4;
            const SIMDGROUPS: i32 = 2;
            templates.extend([("ROWS", ROWS), ("SIMDGROUPS", SIMDGROUPS)]);
            let groups = (self.n - 1) / (ROWS * SIMDGROUPS) + 1;
            let y = Kernel {
                name: c"mlx_fp8_gemv_k",
                source: include_str!("matmul/fp8_gemv_k.metal"),
                header: &header,
                templates: &templates,
                shape,
                dtype: F32,
                grid: [groups * 32 * SIMDGROUPS, 1, z],
                group: [32 * SIMDGROUPS, 1, 1],
            }
            .run(ctx, &inputs)?;
            return Ok(Some((y, "mlx_fp8_gemv_k")));
        }
        const SIMDGROUPS: i32 = 8;
        let groups = (self.n - 1) / 128 + 1;
        // Split K across threadgroups until the GPU has enough independent row streams.
        let split = (256 / (groups * z)).clamp(1, (self.k / (SIMDGROUPS * 16)).max(1));
        templates.extend([("SIMDGROUPS", SIMDGROUPS), ("SPLIT", split)]);
        let partial_shape = [split, routes, self.n];
        let y = Kernel {
            name: c"mlx_fp8_gemv_n",
            source: include_str!("matmul/fp8_gemv_n.metal"),
            header: &header,
            templates: &templates,
            shape: if split == 1 { shape } else { &partial_shape },
            dtype: F32,
            grid: [groups * 32, split * SIMDGROUPS, z],
            group: [32, SIMDGROUPS, 1],
        }
        .run(ctx, &inputs)?;
        let y = if split == 1 {
            y
        } else {
            let sum = ctx.emit(|res, s| unsafe { mlx::mlx_sum_axis(res, y, 0, false, s) })?;
            ctx.reshape(sum, shape)?
        };
        Ok(Some((y, "mlx_fp8_gemv_n")))
    }

    /// Prefill: decode the FP8 weight once per run on GPU, then use MLX's dense GEMM.
    fn float8_decoded_matmul(
        &self,
        ctx: &mut TranslationContext,
    ) -> Result<Option<mlx::mlx_array>, MlxError> {
        let Some(format) = self.w.float8 else {
            return Ok(None);
        };
        if self.routing.is_some() {
            return Ok(None);
        }
        let Parameters {
            strides: [ws0, ws1, ws2],
            divisors: [wd0, wd1, wd2],
        } = self.w.parameters;
        let (rows, cols) = (self.w.shape[0], self.w.shape[1]);
        let vec = if cols % 4 == 0 { 4 } else { 1 };
        let header = format.metal();
        let weight = Kernel {
            name: c"mlx_fp8_dequant",
            source: include_str!("matmul/fp8_dequant.metal"),
            header: &header,
            templates: &[
                ("R1", rows),
                ("C", cols),
                ("VEC", vec),
                ("W_ZERO", i32::from(has_zero(&self.pattern.w))),
                ("WS0", ws0),
                ("WS1", ws1),
                ("WS2", ws2),
                ("WD0", wd0),
                ("WD1", wd1),
                ("WD2", wd2),
            ],
            shape: &self.w.shape,
            dtype: F32,
            grid: [cols / vec, rows, 1],
            group: [64.min(cols / vec), 1, 1],
        }
        .run(
            ctx,
            &[
                ("w", self.w.data),
                ("wscale", self.w.scale),
                ("wzero", self.w.zero),
            ],
        )?;
        let weight = if self.trans_b {
            ctx.transpose(weight, &[1, 0])?
        } else {
            weight
        };
        let x = if self.trans_a {
            ctx.transpose(self.x.data, &[1, 0])?
        } else {
            self.x.data
        };
        Ok(Some(ctx.emit(|res, s| unsafe {
            mlx::mlx_matmul(res, x, weight, s)
        })?))
    }

    fn compute(
        &self,
        ctx: &mut TranslationContext,
        shape: &[i32],
        total: i32,
    ) -> Result<Option<(mlx::mlx_array, Option<&'static str>)>, MlxError> {
        if total == 0 || self.k == 0 {
            return Ok(Some((ctx.zeros(shape, F32)?, None)));
        }
        if self.w.float8.is_some() {
            if self.pattern.x.is_some() {
                return Ok(None);
            }
            if let Some((y, path)) = self.float8_gemv(ctx, shape)? {
                return Ok(Some((y, Some(path))));
            }
            if self.m > FLOAT8_TILED_ROWS
                && let Some(y) = self.float8_decoded_matmul(ctx)?
            {
                return Ok(Some((y, Some("mlx_fp8_dequant_matmul"))));
            }
        }
        let exact = integer_exact(self.pattern, self.x, self.w, self.trans_a, self.trans_b);
        // Retain the proven-integer/compensated paths for doubly quantized and packed4 inputs.
        if self.pattern.x.is_none() && self.w.packed == 0 {
            let header = self.weight_only_header();
            if let Some((y, kind)) = (fused::Matmul {
                operands: self.operands(),
                m: self.m,
                k: self.k,
                n: self.n,
                routing: self.routing,
                k_contiguous: self.trans_b,
                header: &header,
                shape,
                dtype: F32,
            })
            .run(ctx)?
            {
                return Ok(Some((
                    y,
                    Some(match (kind, self.w.float8.is_some()) {
                        (KernelKind::Matvec, false) => "mlx_qdq_matvec",
                        (KernelKind::Tiled, false) => "mlx_qdq_tiled_matmul",
                        (KernelKind::Matvec, true) => "mlx_fp8_matvec",
                        (KernelKind::Tiled, true) => "mlx_fp8_tiled_matmul",
                    }),
                )));
            }
        }
        if self.w.float8.is_some() {
            return Ok(None);
        }
        let Some(threads) = total.checked_mul(32) else {
            return Ok(None);
        };
        let indices = self
            .routing
            .map_or_else(|| ctx.scalar_i32(0), |r| r.indices);
        let y = Kernel {
            name: c"mlx_qdq_matmul",
            source: SOURCE,
            header: HEADER,
            templates: &self.simd_templates(exact),
            shape,
            dtype: F32,
            grid: [threads, 1, 1],
            group: [threads.min(256), 1, 1],
        }
        .run(
            ctx,
            &[
                ("x", self.x.data),
                ("w", self.w.data),
                ("xscale", self.x.scale),
                ("wscale", self.w.scale),
                ("xzero", self.x.zero),
                ("wzero", self.w.zero),
                ("indices", indices),
            ],
        )?;
        let packed = self.x.packed != 0 || self.w.packed != 0;
        let path = match (packed, exact) {
            (true, true) => "mlx_qdq_packed4_integer_matmul",
            (true, false) => "mlx_qdq_packed4_matmul",
            (false, true) => "mlx_qdq_integer_matmul",
            (false, false) => "mlx_qdq_matmul",
        };
        Ok(Some((y, Some(path))))
    }
}

pub(super) fn execute(ctx: &mut TranslationContext, node: &NodeDesc) -> Result<bool, MlxError> {
    let Some(pattern) = &node.qdq else {
        return Ok(false);
    };
    if !metal::is_gpu(ctx)? {
        return Ok(false);
    }
    let Some(w) = operand(ctx, &pattern.w, true)? else {
        return Ok(false);
    };
    let mut leading = None;
    let x = if let Some(dq) = &pattern.x {
        let Some(x) = operand(ctx, dq, false)? else {
            return Ok(false);
        };
        x
    } else {
        let mut data = ctx.resolve(&node.inputs[0])?;
        let mut shape = ctx.shape_of(data);
        if ctx.dtype_of(data) != F32 {
            return Ok(false);
        }
        if shape.len() > 2 && node.op_type == "MatMul" && w.float8.is_some() {
            let k = shape[shape.len() - 1];
            let Some(m) = shape[..shape.len() - 1]
                .iter()
                .try_fold(1i32, |acc, &d| acc.checked_mul(d))
            else {
                return Ok(false);
            };
            leading = Some(shape[..shape.len() - 1].to_vec());
            shape = vec![m, k];
            data = ctx.reshape(data, &shape)?;
        }
        if shape.len() != 2 {
            return Ok(false);
        }
        Operand {
            data: ctx.contiguous(data)?,
            scale: ctx.scalar_f32(1.0),
            zero: ctx.scalar_i32(0),
            shape,
            parameters: Parameters::scalar(),
            packed: 0,
            float8: None,
        }
    };
    let grouped = node.op_type == "GroupedMatMul";
    if w.shape.len() != if grouped { 3 } else { 2 } {
        return Ok(false);
    }
    let gemm = node.op_type == "Gemm";
    let trans_a = gemm && node.ints.get("transA").copied().unwrap_or(0) != 0;
    let trans_b = (grouped || gemm) && node.ints.get("transB").copied().unwrap_or(0) != 0;
    let (m, k) = (
        x.shape[usize::from(trans_a)],
        x.shape[usize::from(!trans_a)],
    );
    let rank = w.shape.len();
    let n = w.shape[rank - 1 - usize::from(trans_b)];
    let wk = w.shape[rank - 2 + usize::from(trans_b)];
    let bias = if grouped && node.inputs.get(3).is_some_and(|r| r.source != Src::Absent) {
        let b = ctx.resolve(&node.inputs[3])?;
        if ctx.dtype_of(b) != F32 {
            return Err("QDQ GroupedMatMul: bias must be float32".into());
        }
        Some(b)
    } else {
        None
    };
    let routing = if grouped {
        let indices = ctx.resolve(&node.inputs[2])?;
        let shape = ctx.shape_of(indices);
        let wide = |s: &[i32]| s.iter().copied().map(i64::from).collect::<Vec<_>>();
        let canonical_w = if trans_b {
            vec![w.shape[0], w.shape[2], w.shape[1]]
        } else {
            w.shape.clone()
        };
        super::matmul::grouped_geometry(
            &wide(&x.shape),
            &wide(&canonical_w),
            &wide(&shape),
            bias.map(|b| wide(&ctx.shape_of(b))).as_deref(),
            None,
        )?;
        Some(fused::read_routing(ctx, &node.inputs[2], w.shape[0])?)
    } else {
        if k != wk {
            return Err(format!(
                "QDQ {}: activation reduction dimension {k} differs from weight dimension {wk}; check transA/transB and the original DequantizeLinear layout",
                node.op_type
            ));
        }
        None
    };
    let selections = routing.as_ref().map_or(1, |r| r.selections);
    let shape = if grouped {
        vec![m, selections, n]
    } else {
        vec![m, n]
    };
    let Some(total) = m.checked_mul(selections).and_then(|v| v.checked_mul(n)) else {
        return Ok(false);
    };
    let Some((mut y, path)) = (Product {
        pattern,
        x: &x,
        w: &w,
        m,
        k,
        n,
        trans_a,
        trans_b,
        routing: routing.as_ref(),
    })
    .compute(ctx, &shape, total)?
    else {
        return Ok(false);
    };
    if let Some(b) = bias {
        let indices = routing.as_ref().expect("grouped routing").indices;
        let selected = ctx.emit(|res, s| unsafe { mlx::mlx_take_axis(res, b, indices, 0, s) })?;
        y = ctx.binary(mlx::mlx_add, y, selected)?;
    }
    if gemm {
        y = super::matmul::gemm_epilogue(ctx, node, y)?;
    }
    if let Some(mut leading) = leading {
        leading.push(n);
        y = ctx.reshape(y, &leading)?;
    }
    if let Some(path) = path {
        ctx.mark_fast(path);
    }
    ctx.bind(&node.outputs[0], y);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{InitData, OutRef, Plan, TensorRef};
    use crate::mlx::{Array, Stream};
    use std::sync::Arc;

    fn reference(name: &str) -> TensorRef {
        TensorRef {
            name: name.into(),
            source: Src::Intermediate,
            ..TensorRef::absent()
        }
    }

    fn output(name: &str) -> OutRef {
        OutRef {
            name: name.into(),
            external: false,
            ctx_index: 0,
            otype: FLOAT,
        }
    }

    fn dq(prefix: &str) -> NodeDesc {
        let mut node = NodeDesc::new(
            "DequantizeLinear".into(),
            String::new(),
            21,
            CompileShapeSafety::Shapeless,
        );
        node.inputs = vec![
            reference(prefix),
            reference(&format!("{prefix}s")),
            reference(&format!("{prefix}z")),
        ];
        node.outputs = vec![output(&format!("{prefix}f"))];
        node
    }

    fn scalar_scale(node: &mut NodeDesc, value: f32) {
        let owned = Arc::new(value.to_ne_bytes().to_vec());
        node.inputs[1].source = Src::Initializer;
        node.inputs[1].init = Some(InitData {
            data: owned.as_ptr().cast(),
            shape: vec![],
            dtype: FLOAT,
            count: 1,
            owned: Some(owned),
        });
    }

    fn quantized_reference(
        name: &str,
        values: &[i32],
        shape: &[i64],
        dtype: ort::ONNXTensorElementDataType,
    ) -> TensorRef {
        let bytes = if crate::engine::packed4_signed(dtype).is_some() {
            let mut bytes = vec![0u8; values.len().div_ceil(2)];
            for (index, &value) in values.iter().enumerate() {
                bytes[index / 2] |= ((value & 15) as u8) << ((index % 2) * 4);
            }
            if !values.len().is_multiple_of(2) {
                *bytes.last_mut().unwrap() |= 0xa0;
            }
            bytes
        } else {
            values.iter().map(|&v| v as u8).collect()
        };
        let owned = Arc::new(bytes);
        let mut input = reference(name);
        input.source = Src::Initializer;
        input.init = Some(InitData {
            data: owned.as_ptr().cast(),
            shape: shape.to_vec(),
            dtype,
            count: values.len(),
            owned: Some(owned),
        });
        input
    }

    #[test]
    fn packed4_transport_keeps_logical_metadata_and_checks_bounds() {
        let stream = Stream::new_default_gpu();
        let mut plan = Plan::new(vec![]);
        let mut ctx = TranslationContext::new(
            &mut plan,
            std::ptr::null(),
            std::ptr::null_mut(),
            stream.as_raw(),
        );
        let values: Vec<i32> = (-8..7).collect();
        for dtype in [
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4,
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4,
        ] {
            let input = quantized_reference("packed", &values, &[3, 5], dtype);
            let packed = ctx.resolve_packed4(&input).unwrap().unwrap();
            assert_eq!(packed.shape, [3, 5]);
            assert_eq!(ctx.shape_of(packed.data), [8]);
            assert_eq!(Some(packed.signed), crate::engine::packed4_signed(dtype));
            assert_eq!(ctx.tensor_shape(&input).unwrap(), [3, 5]);
            assert!(ctx.resolve(&input).is_err());
            assert!(!crate::registry::is_movable(dtype));
            let mut bad = input.clone();
            bad.init.as_mut().unwrap().count = 14;
            assert!(ctx.resolve_packed4(&bad).is_err());
            bad.init.as_mut().unwrap().shape = vec![2, 7];
            assert!(ctx.resolve_packed4(&bad).is_err());
            bad.init.as_mut().unwrap().shape = vec![-1, 14];
            assert!(ctx.resolve_packed4(&bad).is_err());
        }
    }

    #[test]
    fn qdq_packed_direct_gpu_products() {
        use ort::*;
        let types = [
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
        ];
        for (grouped, transposed) in [(false, false), (false, true), (true, false), (true, true)] {
            for &at in &types {
                for &wt in &types {
                    if crate::engine::packed4_signed(at).is_none()
                        && crate::engine::packed4_signed(wt).is_none()
                    {
                        continue;
                    }
                    for k in [0, 1, 33, 256, 1025] {
                        let stream = Stream::new_default_gpu();
                        let mut plan = Plan::new(vec![]);
                        let mut ctx = TranslationContext::new(
                            &mut plan,
                            std::ptr::null(),
                            std::ptr::null_mut(),
                            stream.as_raw(),
                        );
                        let (m, n, groups, selections) = (3, 5, if grouped { 3 } else { 1 }, 1);
                        let signed = |dtype| dtype == types[0] || dtype == types[2];
                        let av: Vec<i32> = (0..m * k)
                            .map(|i| i % 16 - if signed(at) { 8 } else { 0 })
                            .collect();
                        let wv: Vec<i32> = (0..groups * k * n)
                            .map(|i| (i * 3) % 16 - if signed(wt) { 8 } else { 0 })
                            .collect();
                        let az = if signed(at) { -3 } else { 5 };
                        let wz = if signed(wt) { -2 } else { 7 };
                        let mut a = dq("a");
                        let mut w = dq("w");
                        a.inputs[0] = quantized_reference("a", &av, &[m.into(), k.into()], at);
                        w.inputs[0] = quantized_reference(
                            "w",
                            &wv,
                            &if grouped && transposed {
                                vec![groups.into(), n.into(), k.into()]
                            } else if grouped {
                                vec![groups.into(), k.into(), n.into()]
                            } else if transposed {
                                vec![n.into(), k.into()]
                            } else {
                                vec![k.into(), n.into()]
                            },
                            wt,
                        );
                        a.inputs[2] = quantized_reference("az", &[az], &[], at);
                        w.inputs[2] = quantized_reference("wz", &[wz], &[], wt);
                        scalar_scale(&mut a, 0.25);
                        scalar_scale(&mut w, 0.125);
                        let mut node = NodeDesc::new(
                            if grouped {
                                "GroupedMatMul"
                            } else if transposed {
                                "Gemm"
                            } else {
                                "MatMul"
                            }
                            .into(),
                            String::new(),
                            if grouped { 29 } else { 13 },
                            CompileShapeSafety::EagerOnly {
                                reason: EAGER_REASON,
                            },
                        );
                        node.inputs = vec![reference("af"), reference("wf")];
                        node.outputs = vec![output("y")];
                        node.ints.insert("transB".into(), i64::from(transposed));
                        node.qdq = Some(Box::new(QuantizedMatMul { x: Some(a), w }));
                        let routes = [2i32, 0, 1];
                        if grouped {
                            let indices = ctx.keep(Array::from_data(
                                routes.as_ptr().cast(),
                                &[m, selections],
                                mlx::mlx_dtype__MLX_INT32,
                            ));
                            ctx.bind(&output("indices"), indices);
                            node.inputs.push(reference("indices"));
                        }
                        assert!(execute(&mut ctx, &node).unwrap());
                        let y = ctx.resolve(&reference("y")).unwrap();
                        let y = ctx.contiguous_eval(y).unwrap();
                        assert_eq!(
                            ctx.shape_of(y),
                            if grouped {
                                vec![m, selections, n]
                            } else {
                                vec![m, n]
                            }
                        );
                        if k != 0 {
                            let actual = unsafe {
                                std::slice::from_raw_parts(
                                    mlx::mlx_array_data_float32(y),
                                    (m * n) as usize,
                                )
                            };
                            for row in 0..m {
                                let group = if grouped { routes[row as usize] } else { 0 };
                                for col in 0..n {
                                    let expected: i32 = (0..k)
                                        .map(|kk| {
                                            let wi = if transposed {
                                                (group * n + col) * k + kk
                                            } else {
                                                (group * k + kk) * n + col
                                            };
                                            (av[(row * k + kk) as usize] - az)
                                                * (wv[wi as usize] - wz)
                                        })
                                        .sum();
                                    assert_eq!(
                                        actual[(row * n + col) as usize],
                                        expected as f32 / 32.0
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn qdq_packed_general_rounds_dot_before_grouped_bias() {
        let stream = Stream::new_default_gpu();
        let mut plan = Plan::new(vec![]);
        let mut ctx = TranslationContext::new(
            &mut plan,
            std::ptr::null(),
            std::ptr::null_mut(),
            stream.as_raw(),
        );
        let x = [1.0f32, 2.0f32.powi(-24)];
        let indices = [2i32, 0];
        let bias = [-1.0f32; 3];
        for (name, values, shape, dtype) in [
            ("x", x.as_ptr().cast(), vec![1, 2], F32),
            (
                "indices",
                indices.as_ptr().cast(),
                vec![1, 2],
                mlx::mlx_dtype__MLX_INT32,
            ),
            ("bias", bias.as_ptr().cast(), vec![3, 1], F32),
        ] {
            let array = ctx.keep(Array::from_data(values, &shape, dtype));
            ctx.bind(&output(name), array);
        }
        let mut w = dq("w");
        w.inputs[0] = quantized_reference(
            "w",
            &[1; 6],
            &[3, 2, 1],
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4,
        );
        scalar_scale(&mut w, 1.0);
        w.inputs[2] = TensorRef::absent();
        let mut node = NodeDesc::new(
            "GroupedMatMul".into(),
            String::new(),
            29,
            CompileShapeSafety::EagerOnly {
                reason: EAGER_REASON,
            },
        );
        node.inputs = vec![
            reference("x"),
            reference("wf"),
            reference("indices"),
            reference("bias"),
        ];
        node.outputs = vec![output("y")];
        node.qdq = Some(Box::new(QuantizedMatMul { x: None, w }));
        assert!(execute(&mut ctx, &node).unwrap());
        let y = ctx.resolve(&reference("y")).unwrap();
        let y = ctx.contiguous_eval(y).unwrap();
        assert_eq!(ctx.shape_of(y), [1, 2, 1]);
        let actual = unsafe { std::slice::from_raw_parts(mlx::mlx_array_data_float32(y), 2) };
        // RNE(1 + 2^-24) - 1 = 0; folding bias into the dot would instead give 2^-24.
        assert_eq!(actual, [0.0, 0.0]);
    }

    #[test]
    fn qdq_parameter_indices() {
        let mut node = dq("w");
        node.ints.insert("axis".into(), -2);
        node.ints.insert("block_size".into(), 32);
        let layout = parameter_layout(&node, &[3, 65, 7], &[3, 3, 7], true).unwrap();
        for g in 0..3 {
            for k in 0..65 {
                for n in 0..7 {
                    let coordinates = [g, k, n];
                    let actual: i32 = (0..3)
                        .map(|i| coordinates[i] / layout.divisors[i] * layout.strides[i])
                        .sum();
                    assert_eq!(actual, (g * 3 + k / 32) * 7 + n);
                }
            }
        }
        assert!(parameter_layout(&node, &[3, 65, 7], &[3, 2, 7], true).is_none());
        node.ints.clear();
        node.ints.insert("axis".into(), -1);
        assert_eq!(
            parameter_layout(&node, &[65, 7], &[7], true)
                .unwrap()
                .strides,
            [0, 0, 1],
        );
        assert!(parameter_layout(&node, &[65, 7], &[65], true).is_none());
        node.ints.insert("axis".into(), 5);
        assert!(parameter_layout(&node, &[65, 7], &[7], true).is_none());
        assert_eq!(
            parameter_layout(&node, &[65, 7], &[], true),
            Some(Parameters::scalar()),
        );
    }

    #[test]
    fn qdq_recognition_preserves_shared_results_and_domains() {
        let a = dq("a");
        let w = dq("w");
        let mut mm = NodeDesc::new(
            "MatMul".into(),
            String::new(),
            13,
            CompileShapeSafety::Shapeless,
        );
        mm.inputs = vec![reference("af"), reference("wf")];
        mm.outputs = vec![output("y")];
        let mut nodes = vec![a, w, mm];
        nodes[1].outputs[0].external = true;
        recognize(&mut nodes);
        assert!(nodes[2].qdq.as_ref().unwrap().x.is_some());
        assert_eq!(nodes[1].op_type, "DequantizeLinear");
        assert!(nodes[1].outputs[0].external);
        assert_eq!(nodes[2].inputs[1].name, "wf");
        nodes[2].qdq = None;
        nodes[1].domain = "com.microsoft".into();
        recognize(&mut nodes);
        assert!(nodes[2].qdq.is_none());
        nodes[1].domain.clear();
        nodes[2].op_type = "GroupedMatMul".into();
        for version in [28, 30] {
            nodes[2].since_version = version;
            recognize(&mut nodes);
            assert!(nodes[2].qdq.is_none());
        }
    }

    #[test]
    fn qdq_integer_bound_and_scale_guards() {
        for a in [-255_i32, -1, 0, 1, 255] {
            for b in [-255_i32, -1, 0, 1, 255] {
                for ea in [-40, 0, 40] {
                    for eb in [-40, 0, 40] {
                        let sa = 2.0f32.powi(ea);
                        let sb = 2.0f32.powi(eb);
                        let mut integer = 0i32;
                        let mut floating = 0.0f32;
                        for _ in 0..256 {
                            integer += a * b;
                            assert!(i64::from(integer).abs() < 1 << 24);
                            floating = (a as f32 * sa).mul_add(b as f32 * sb, floating);
                            assert_eq!((integer as f32 * sa) * sb, floating);
                            assert!(floating == 0.0 || floating.is_normal());
                        }
                    }
                }
            }
        }
        for exponent in [-40, 0, 40] {
            let mut node = dq("a");
            scalar_scale(&mut node, 2.0f32.powi(exponent));
            assert!(binary_scales(&node));
            for centered in [-255, -1, 0, 1, 255] {
                let value = centered as f32 * 2.0f32.powi(exponent);
                assert!(value == 0.0 || value.is_normal());
            }
        }
        for value in [
            0.0,
            -1.0,
            0.3,
            f32::NAN,
            f32::INFINITY,
            2.0f32.powi(-41),
            2.0f32.powi(41),
        ] {
            let mut node = dq("a");
            scalar_scale(&mut node, value);
            assert!(!binary_scales(&node));
        }
    }

    #[test]
    fn qdq_direct_gpu_products() {
        // Direct handler tests do not assert ORT can load GroupedMatMul-29.
        for (grouped, transposed) in [(false, false), (false, true), (true, false), (true, true)] {
            for signed_a in [false, true] {
                for signed_w in [false, true] {
                    for k in [0, 33, 256, 4096] {
                        let stream = Stream::new_default_gpu();
                        let mut plan = Plan::new(vec![]);
                        let mut ctx = TranslationContext::new(
                            &mut plan,
                            std::ptr::null(),
                            std::ptr::null_mut(),
                            stream.as_raw(),
                        );
                        let (m, n, groups, selections) = (
                            2,
                            5,
                            if grouped { 3 } else { 1 },
                            if grouped { 3 } else { 1 },
                        );
                        let mut a = dq("a");
                        let mut w = dq("w");
                        scalar_scale(&mut a, 0.25);
                        scalar_scale(&mut w, 0.125);
                        let mut node = NodeDesc::new(
                            if grouped {
                                "GroupedMatMul"
                            } else if transposed {
                                "Gemm"
                            } else {
                                "MatMul"
                            }
                            .into(),
                            String::new(),
                            if grouped { 29 } else { 13 },
                            CompileShapeSafety::EagerOnly {
                                reason: EAGER_REASON,
                            },
                        );
                        node.inputs = vec![reference("af"), reference("wf")];
                        node.outputs = vec![output("y")];
                        node.ints.insert("transB".into(), i64::from(transposed));
                        node.qdq = Some(Box::new(QuantizedMatMul { x: Some(a), w }));
                        let byte = |q: i32, signed: bool| {
                            if signed {
                                (q - 128) as i8 as u8
                            } else {
                                q as u8
                            }
                        };
                        let av: Vec<_> = (0..m * k).map(|i| byte(i % 7, signed_a)).collect();
                        let wv: Vec<_> = (0..groups * k * n)
                            .map(|i| byte(i % 11, signed_w))
                            .collect();
                        for (name, shape, data, signed) in [
                            ("a", vec![m, k], av, signed_a),
                            (
                                "w",
                                if grouped && transposed {
                                    vec![groups, n, k]
                                } else if grouped {
                                    vec![groups, k, n]
                                } else if transposed {
                                    vec![n, k]
                                } else {
                                    vec![k, n]
                                },
                                wv,
                                signed_w,
                            ),
                            ("az", vec![], vec![byte(0, signed_a)], signed_a),
                            ("wz", vec![], vec![byte(0, signed_w)], signed_w),
                        ] {
                            let dtype = if signed {
                                mlx::mlx_dtype__MLX_INT8
                            } else {
                                mlx::mlx_dtype__MLX_UINT8
                            };
                            let array =
                                ctx.keep(Array::from_data(data.as_ptr().cast(), &shape, dtype));
                            ctx.bind(&output(name), array);
                        }
                        let routes = [2i64, 0, 2, 1, 1, 0];
                        if grouped {
                            let array = ctx.keep(Array::from_data(
                                routes.as_ptr().cast(),
                                &[m, selections],
                                mlx::mlx_dtype__MLX_INT64,
                            ));
                            ctx.bind(&output("indices"), array);
                            let bias = [0.0625f32; 15];
                            let array =
                                ctx.keep(Array::from_data(bias.as_ptr().cast(), &[groups, n], F32));
                            ctx.bind(&output("bias"), array);
                            node.inputs
                                .extend([reference("indices"), reference("bias")]);
                        }
                        assert!(execute(&mut ctx, &node).unwrap());
                        let y = ctx.resolve(&reference("y")).unwrap();
                        let y = ctx.contiguous_eval(y).unwrap();
                        assert_eq!(
                            ctx.shape_of(y),
                            if grouped {
                                vec![m, selections, n]
                            } else {
                                vec![m, n]
                            }
                        );
                        let ptr = unsafe { mlx::mlx_array_data_float32(y) };
                        assert!(!ptr.is_null());
                        let result = unsafe {
                            std::slice::from_raw_parts(ptr, (m * selections * n) as usize)
                        };
                        for row in 0..m {
                            for selection in 0..selections {
                                let g = if grouped {
                                    routes[(row * selections + selection) as usize] as i32
                                } else {
                                    0
                                };
                                for col in 0..n {
                                    let expected: i64 = (0..k)
                                        .map(|kk| {
                                            i64::from((row * k + kk) % 7)
                                                * i64::from(
                                                    (if transposed {
                                                        (g * n + col) * k + kk
                                                    } else {
                                                        (g * k + kk) * n + col
                                                    }) % 11,
                                                )
                                        })
                                        .sum();
                                    assert_eq!(
                                        result[((row * selections + selection) * n + col) as usize],
                                        expected as f32 / 32.0 + if grouped { 0.0625 } else { 0.0 },
                                    );
                                }
                            }
                        }
                        if grouped {
                            for invalid in [-1i64, 3, i64::MAX] {
                                let bad = [invalid; 6];
                                let array = ctx.keep(Array::from_data(
                                    bad.as_ptr().cast(),
                                    &[m, selections],
                                    mlx::mlx_dtype__MLX_INT64,
                                ));
                                ctx.bind(&output("indices"), array);
                                assert!(execute(&mut ctx, &node).is_err());
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn qdq_grouped_blocked_gpu_product() {
        for packed_type in [
            None,
            Some(ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4),
            Some(ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4),
        ] {
            let stream = Stream::new_default_gpu();
            let mut plan = Plan::new(vec![]);
            let mut ctx = TranslationContext::new(
                &mut plan,
                std::ptr::null(),
                std::ptr::null_mut(),
                stream.as_raw(),
            );
            let mut a = dq("a");
            a.ints.insert("axis".into(), 0);
            let mut w = dq("w");
            w.ints
                .extend([("axis".into(), -2), ("block_size".into(), 32)]);
            let mut node = NodeDesc::new(
                "GroupedMatMul".into(),
                String::new(),
                29,
                CompileShapeSafety::EagerOnly {
                    reason: EAGER_REASON,
                },
            );
            node.qdq = Some(Box::new(QuantizedMatMul { x: Some(a), w }));
            node.inputs = ["af", "wf", "indices"].map(reference).to_vec();
            node.outputs = vec![output("y")];
            let (m, k, n, groups, selections, blocks) = (2, 65, 5, 3, 2, 3);
            let unsigned = packed_type
                .is_some_and(|dtype| crate::engine::packed4_signed(dtype) == Some(false));
            let av: Vec<i8> = (0..m * k)
                .map(|i| {
                    (i % if packed_type.is_some() { 16 } else { 17 } - if unsigned { 0 } else { 8 })
                        as i8
                })
                .collect();
            let wv: Vec<i8> = (0..groups * k * n)
                .map(|i| (i % 13 - if unsigned { 0 } else { 6 }) as i8)
                .collect();
            let az = [if unsigned { 2i8 } else { -2 }, 3];
            let wz: Vec<i8> = (0..groups * blocks * n)
                .map(|i| (i % 5 - if unsigned { 0 } else { 2 }) as i8)
                .collect();
            let ascale = [0.25f32, 0.125];
            let wscale: Vec<f32> = (0..groups * blocks * n)
                .map(|i| 2.0f32.powi(i % 4 - 5))
                .collect();
            for (name, shape, data) in [
                ("a", vec![m, k], av.as_slice()),
                ("w", vec![groups, k, n], wv.as_slice()),
                ("az", vec![m], az.as_slice()),
                ("wz", vec![groups, blocks, n], wz.as_slice()),
            ] {
                if let Some(dtype) = packed_type {
                    let input = quantized_reference(
                        name,
                        &data.iter().copied().map(i32::from).collect::<Vec<_>>(),
                        &shape.iter().copied().map(i64::from).collect::<Vec<_>>(),
                        dtype,
                    );
                    let pattern = node.qdq.as_mut().unwrap();
                    match name {
                        "a" => pattern.x.as_mut().unwrap().inputs[0] = input,
                        "az" => pattern.x.as_mut().unwrap().inputs[2] = input,
                        "w" => pattern.w.inputs[0] = input,
                        "wz" => pattern.w.inputs[2] = input,
                        _ => unreachable!(),
                    }
                    continue;
                }
                let array = ctx.keep(Array::from_data(
                    data.as_ptr().cast(),
                    &shape,
                    mlx::mlx_dtype__MLX_INT8,
                ));
                ctx.bind(&output(name), array);
            }
            for (name, shape, data) in [
                ("as", vec![m], ascale.as_slice()),
                ("ws", vec![groups, blocks, n], wscale.as_slice()),
            ] {
                let array = ctx.keep(Array::from_data(data.as_ptr().cast(), &shape, F32));
                ctx.bind(&output(name), array);
            }
            let routes = [2i32, 0, 2, 1];
            let array = ctx.keep(Array::from_data(
                routes.as_ptr().cast(),
                &[m, selections],
                mlx::mlx_dtype__MLX_INT32,
            ));
            ctx.bind(&output("indices"), array);
            assert!(execute(&mut ctx, &node).unwrap());
            let y = ctx.resolve(&reference("y")).unwrap();
            let y = ctx.contiguous_eval(y).unwrap();
            let actual = unsafe {
                std::slice::from_raw_parts(
                    mlx::mlx_array_data_float32(y),
                    (m * selections * n) as usize,
                )
            };
            for row in 0..m as usize {
                for selection in 0..selections as usize {
                    let g = routes[row * selections as usize + selection] as usize;
                    for col in 0..n as usize {
                        let mut expected = 0.0f64;
                        for kk in 0..k as usize {
                            let p = (g * blocks as usize + kk / 32) * n as usize + col;
                            let ax = f32::from(av[row * k as usize + kk]) - f32::from(az[row]);
                            let bw = f32::from(wv[(g * k as usize + kk) * n as usize + col])
                                - f32::from(wz[p]);
                            expected += f64::from(ax * ascale[row]) * f64::from(bw * wscale[p]);
                        }
                        assert_eq!(
                            actual[(row * selections as usize + selection) * n as usize + col],
                            expected as f32
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn qdq_simd_reduction_gpu() {
        for k in [32, 33, 64, 65, 256, 257, 4096] {
            for static_scales in [false, true] {
                let stream = Stream::new_default_gpu();
                let mut plan = Plan::new(vec![]);
                let mut ctx = TranslationContext::new(
                    &mut plan,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    stream.as_raw(),
                );
                let mut a = dq("a");
                let mut w = dq("w");
                if static_scales {
                    scalar_scale(&mut a, 1.0);
                    scalar_scale(&mut w, 1.0);
                }
                let mut node = NodeDesc::new(
                    "MatMul".into(),
                    String::new(),
                    13,
                    CompileShapeSafety::EagerOnly {
                        reason: EAGER_REASON,
                    },
                );
                node.inputs = vec![reference("af"), reference("wf")];
                node.outputs = vec![output("y")];
                node.qdq = Some(Box::new(QuantizedMatMul { x: Some(a), w }));
                let ones = vec![1u8; k as usize];
                for (name, shape, values) in [
                    ("a", vec![1, k], ones.as_slice()),
                    ("w", vec![k, 1], ones.as_slice()),
                    ("az", vec![], &[0u8]),
                    ("wz", vec![], &[0u8]),
                ] {
                    let array = ctx.keep(Array::from_data(
                        values.as_ptr().cast(),
                        &shape,
                        mlx::mlx_dtype__MLX_UINT8,
                    ));
                    ctx.bind(&output(name), array);
                }
                if !static_scales {
                    for name in ["as", "ws"] {
                        let value = ctx.scalar_f32(1.0);
                        ctx.bind(&output(name), value);
                    }
                }
                let pattern = node.qdq.as_ref().unwrap();
                let x = operand(&mut ctx, pattern.x.as_ref().unwrap(), false)
                    .unwrap()
                    .unwrap();
                let w = operand(&mut ctx, &pattern.w, true).unwrap().unwrap();
                assert_eq!(
                    integer_exact(pattern, &x, &w, false, false),
                    static_scales && k <= 256
                );
                assert!(execute(&mut ctx, &node).unwrap());
                let y = ctx.resolve(&reference("y")).unwrap();
                let y = ctx.contiguous_eval(y).unwrap();
                assert_eq!(unsafe { *mlx::mlx_array_data_float32(y) }, k as f32);
            }
        }
    }
}
