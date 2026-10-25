//! Private Q/DQ matrix-product recognition. Quantized bytes are device-kernel inputs;
//! dequantized tensors remain available to other consumers and graph outputs.

use std::collections::HashMap;
use std::ffi::CString;

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::mlx::VectorArray;
use crate::registry::CompileShapeSafety;
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

fn dq_candidate(node: &NodeDesc) -> bool {
    node.domain.is_empty()
        && node.op_type == "DequantizeLinear"
        && node.since_version >= 10
        && (2..=3).contains(&node.inputs.len())
        && node.outputs.len() == 1
        && node.outputs[0].otype == FLOAT
}

pub(crate) fn recognize(nodes: &mut [NodeDesc]) {
    if std::env::var_os("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION")
        .is_some_and(|v| v != "0" && !v.is_empty())
    {
        return;
    }
    let mut producers: HashMap<String, NodeDesc> = HashMap::new();
    for node in nodes {
        if node.domain.is_empty()
            && (node.op_type == "MatMul"
                || (node.op_type == "GroupedMatMul" && node.since_version == 29))
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
    let (data, shape, packing) = if let Some(packed) = packed {
        (packed.data, packed.shape, if packed.signed { 2 } else { 1 })
    } else {
        let data = ctx.resolve(&dq.inputs[0])?;
        (data, ctx.shape_of(data), 0)
    };
    let scale = ctx.resolve(&dq.inputs[1])?;
    if (packing == 0 && !integer8(ctx.dtype_of(data)))
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

fn integer_exact(pattern: &QuantizedMatMul, x: &Operand, w: &Operand) -> bool {
    // Each centered int8/uint8 value is in [-255,255]. For K<=256 every partial sum's
    // absolute value is <=16,646,400<2^24, hence exact in int32 AND float32. Binary
    // scales in [2^-40,2^40] keep every scaled product/partial normal and finite.
    (0..=256).contains(&x.shape[1])
        && x.parameters.strides[1] == 0
        && w.parameters.strides[1] == 0
        && pattern.x.as_ref().is_some_and(binary_scales)
        && binary_scales(&pattern.w)
}

const SOURCE: &str = r#"
    size_t out = thread_position_in_grid.x / 32;
    uint lane = thread_position_in_grid.x % 32;
    if (out >= size_t(M) * S * N) { return; }
    uint col = out % N;
    size_t route = out / N;
    uint row = route / S;
    uint expert = GROUPED ? uint(indices[route]) : 0;
    float sum = 0.0f;
    int isum = 0;
    for (uint k = lane; k < K; k += 32) {
        size_t xp = size_t(row / XD0) * XS0 + size_t(k / XD1) * XS1;
        size_t wp = size_t(expert / WD0) * WS0 + size_t(k / WD1) * WS1
                  + size_t(col / WD2) * WS2;
        size_t xi = size_t(row) * K + k;
        size_t wi = (size_t(expert) * K + k) * N + col;
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
                               : float(x[size_t(row) * K + k]);
            float bw = precise::fma(float(b), wscale[wp], 0.0f);
            sum = precise::fma(ax, bw, sum);
        }
    }
    isum = simd_sum(isum);
    sum = simd_sum(sum);
    if (INTEGER) {
        size_t xp = size_t(row / XD0) * XS0;
        size_t wp = size_t(expert / WD0) * WS0 + size_t(col / WD2) * WS2;
        sum = (float(isum) * xscale[xp]) * wscale[wp];
    }
    if (lane == 0) { y[out] = sum; }
"#;

pub(super) fn execute(ctx: &mut TranslationContext, node: &NodeDesc) -> Result<bool, MlxError> {
    let Some(pattern) = &node.qdq else {
        return Ok(false);
    };
    let mut device = unsafe { mlx::mlx_device_new() };
    let mut kind = mlx::mlx_device_type__MLX_CPU;
    let mut rc = unsafe { mlx::mlx_stream_get_device(&mut device, ctx.stream()) };
    if rc == 0 {
        rc = unsafe { mlx::mlx_device_get_type(&mut kind, device) };
    }
    unsafe { mlx::mlx_device_free(device) };
    if rc != 0 {
        return Err("QDQ MatMul: cannot inspect MLX stream device".into());
    }
    if kind != mlx::mlx_device_type__MLX_GPU {
        return Ok(false);
    }
    let Some(w) = operand(ctx, &pattern.w, true)? else {
        return Ok(false);
    };
    let x = if let Some(dq) = &pattern.x {
        let Some(x) = operand(ctx, dq, false)? else {
            return Ok(false);
        };
        x
    } else {
        let data = ctx.resolve(&node.inputs[0])?;
        let shape = ctx.shape_of(data);
        if shape.len() != 2 || ctx.dtype_of(data) != F32 {
            return Ok(false);
        }
        Operand {
            data: ctx.contiguous(data)?,
            scale: ctx.scalar_f32(1.0),
            zero: ctx.scalar_i32(0),
            shape,
            parameters: Parameters::scalar(),
            packed: 0,
        }
    };
    let grouped = node.op_type == "GroupedMatMul";
    if w.shape.len() != if grouped { 3 } else { 2 } {
        return Ok(false);
    }
    let (m, k) = (x.shape[0], x.shape[1]);
    let n = *w.shape.last().expect("rank checked");
    let mut indices = ctx.scalar_i32(0);
    let mut selections = 1;
    let bias = if grouped && node.inputs.get(3).is_some_and(|r| r.source != Src::Absent) {
        let b = ctx.resolve(&node.inputs[3])?;
        if ctx.dtype_of(b) != F32 {
            return Err("QDQ GroupedMatMul: bias must be float32".into());
        }
        Some(b)
    } else {
        None
    };
    if grouped {
        indices = ctx.resolve(&node.inputs[2])?;
        let shape = ctx.shape_of(indices);
        let wide = |s: &[i32]| s.iter().copied().map(i64::from).collect::<Vec<_>>();
        super::matmul::grouped_geometry(
            &wide(&x.shape),
            &wide(&w.shape),
            &wide(&shape),
            bias.map(|b| wide(&ctx.shape_of(b))).as_deref(),
            None,
        )?;
        if !matches!(
            ctx.dtype_of(indices),
            mlx::mlx_dtype__MLX_INT32 | mlx::mlx_dtype__MLX_INT64
        ) {
            return Err("QDQ GroupedMatMul: indices must be int32 or int64".into());
        }
        let values = ctx.read_ints_eval(&node.inputs[2])?;
        if values.len() != (i64::from(shape[0]) * i64::from(shape[1])) as usize
            || values.iter().any(|&i| i < 0 || i >= i64::from(w.shape[0]))
        {
            return Err("QDQ GroupedMatMul: group_indices values must be in [0,G)".into());
        }
        indices = ctx.contiguous(indices)?;
        selections = shape[1];
    } else if k != w.shape[0] {
        return Err("QDQ MatMul: reduction dimensions differ".into());
    }
    let shape = if grouped {
        vec![m, selections, n]
    } else {
        vec![m, n]
    };
    let total = m
        .checked_mul(selections)
        .and_then(|v| v.checked_mul(n))
        .ok_or("QDQ MatMul: output exceeds int32 indexing")?;
    let Some(threads) = total.checked_mul(32) else {
        return Ok(false);
    };
    let exact = integer_exact(pattern, &x, &w);
    let mut y = if total == 0 || k == 0 {
        ctx.zeros(&shape, F32)?
    } else {
        let mut inputs = VectorArray::new();
        for a in [x.data, w.data, x.scale, w.scale, x.zero, w.zero, indices] {
            // MLX emits scalar references for rank-0 arguments; this shader uses pointers.
            let a = if ctx.ndim(a) == 0 {
                ctx.reshape(a, &[1])?
            } else {
                a
            };
            inputs.append(a);
        }
        let names = super::selective_scan::VecString::new(&[
            "x", "w", "xscale", "wscale", "xzero", "wzero", "indices",
        ])?;
        let outputs = super::selective_scan::VecString::new(&["y"])?;
        let source = CString::new(SOURCE).expect("static shader has no NUL");
        let kernel = unsafe {
            mlx::mlx_fast_metal_kernel_new(
                c"mlx_qdq_matmul".as_ptr(),
                names.as_raw(),
                outputs.as_raw(),
                source.as_ptr(),
                c"".as_ptr(),
                true,
                false,
            )
        };
        if kernel.ctx.is_null() {
            return Err("QDQ MatMul: cannot create Metal kernel".into());
        }
        let config = unsafe { mlx::mlx_fast_metal_kernel_config_new() };
        if config.ctx.is_null() {
            unsafe { mlx::mlx_fast_metal_kernel_free(kernel) };
            return Err("QDQ MatMul: cannot create Metal kernel configuration".into());
        }
        let mut rc = 0;
        for (name, value) in [
            ("M", m),
            ("K", k),
            ("N", n),
            ("S", selections),
            ("GROUPED", i32::from(grouped)),
            ("X_QUANT", i32::from(pattern.x.is_some())),
            ("X_PACKED", x.packed),
            ("W_PACKED", w.packed),
            ("INTEGER", i32::from(exact)),
            (
                "X_ZERO",
                i32::from(
                    pattern
                        .x
                        .as_ref()
                        .is_some_and(|d| d.inputs.get(2).is_some_and(|r| r.source != Src::Absent)),
                ),
            ),
            (
                "W_ZERO",
                i32::from(
                    pattern
                        .w
                        .inputs
                        .get(2)
                        .is_some_and(|r| r.source != Src::Absent),
                ),
            ),
            ("XS0", x.parameters.strides[0]),
            ("XS1", x.parameters.strides[1]),
            ("XD0", x.parameters.divisors[0]),
            ("XD1", x.parameters.divisors[1]),
            ("WS0", w.parameters.strides[0]),
            ("WS1", w.parameters.strides[1]),
            ("WS2", w.parameters.strides[2]),
            ("WD0", w.parameters.divisors[0]),
            ("WD1", w.parameters.divisors[1]),
            ("WD2", w.parameters.divisors[2]),
        ] {
            let name = CString::new(name).expect("template name has no NUL");
            rc |= unsafe {
                mlx::mlx_fast_metal_kernel_config_add_template_arg_int(config, name.as_ptr(), value)
            };
        }
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_output_arg(
                config,
                shape.as_ptr(),
                shape.len(),
                F32,
            )
        };
        rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_grid(config, threads, 1, 1) };
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_set_thread_group(config, threads.min(256), 1, 1)
        };
        let mut result = VectorArray::new();
        if rc == 0 {
            rc = unsafe {
                mlx::mlx_fast_metal_kernel_apply(
                    result.as_mut_ptr(),
                    kernel,
                    inputs.as_raw(),
                    config,
                    ctx.stream(),
                )
            };
        }
        unsafe {
            mlx::mlx_fast_metal_kernel_config_free(config);
            mlx::mlx_fast_metal_kernel_free(kernel);
        }
        if rc != 0 || result.size() != 1 {
            return Err("QDQ MatMul: Metal kernel dispatch failed".into());
        }
        ctx.keep(result.get(0))
    };
    if let Some(b) = bias {
        let selected = ctx.emit(|res, s| unsafe { mlx::mlx_take_axis(res, b, indices, 0, s) })?;
        y = ctx.binary(mlx::mlx_add, y, selected)?;
    }
    if total != 0 && k != 0 {
        ctx.mark_fast(if (x.packed != 0 || w.packed != 0) && exact {
            "mlx_qdq_packed4_integer_matmul"
        } else if x.packed != 0 || w.packed != 0 {
            "mlx_qdq_packed4_matmul"
        } else if exact {
            "mlx_qdq_integer_matmul"
        } else {
            "mlx_qdq_matmul"
        });
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
        for grouped in [false, true] {
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
                            &if grouped {
                                vec![groups.into(), k.into(), n.into()]
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
                            if grouped { "GroupedMatMul" } else { "MatMul" }.into(),
                            String::new(),
                            if grouped { 29 } else { 13 },
                            CompileShapeSafety::EagerOnly {
                                reason: EAGER_REASON,
                            },
                        );
                        node.inputs = vec![reference("af"), reference("wf")];
                        node.outputs = vec![output("y")];
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
                                            (av[(row * k + kk) as usize] - az)
                                                * (wv[((group * k + kk) * n + col) as usize] - wz)
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
        for grouped in [false, true] {
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
                            if grouped { "GroupedMatMul" } else { "MatMul" }.into(),
                            String::new(),
                            if grouped { 29 } else { 13 },
                            CompileShapeSafety::EagerOnly {
                                reason: EAGER_REASON,
                            },
                        );
                        node.inputs = vec![reference("af"), reference("wf")];
                        node.outputs = vec![output("y")];
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
                                if grouped {
                                    vec![groups, k, n]
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
                                                * i64::from(((g * k + kk) * n + col) % 11)
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
                assert_eq!(integer_exact(pattern, &x, &w), static_scales && k <= 256);
                assert!(execute(&mut ctx, &node).unwrap());
                let y = ctx.resolve(&reference("y")).unwrap();
                let y = ctx.contiguous_eval(y).unwrap();
                assert_eq!(unsafe { *mlx::mlx_array_data_float32(y) }, k as f32);
            }
        }
    }
}
