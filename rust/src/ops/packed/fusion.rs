//! Private packed-weight consumers. Original decode/view nodes remain live for shared outputs.

use std::collections::HashMap;
use std::fmt::Write;

use super::super::matmul::fused::{self, KernelKind, Routing};
use super::{BFLOAT16, FLOAT, FLOAT16, INT64, PackedFormat, dequantize_input, gpu};
use crate::engine::{MlxError, NodeDesc, Src, TensorRef, TranslationContext, mlx_dtype_from_onnx};
use crate::registry::{CompileShapeSafety, NodeView};
use crate::sys::mlx;

#[derive(Clone)]
struct Permutation {
    dimensions: Vec<i32>,
    source_strides: Vec<usize>,
}

#[derive(Clone)]
pub(crate) struct FusedWeight {
    decode: NodeDesc,
    format: PackedFormat,
    shape: Vec<i32>,
    permutations: Vec<Permutation>,
}

fn strides(shape: &[i32]) -> Option<Vec<usize>> {
    let mut result = vec![1; shape.len()];
    let mut size = 1usize;
    for (index, &dim) in shape.iter().enumerate().rev() {
        result[index] = size;
        size = size.checked_mul(usize::try_from(dim).ok()?)?;
    }
    Some(result)
}

fn literal_shape(value: &TensorRef, producers: &HashMap<&str, &NodeDesc>) -> Option<Vec<i64>> {
    if let Some(init) = &value.init
        && (value.source == Src::Initializer || value.constant)
        && init.dtype == INT64
        && init.shape.len() == 1
        && init.count <= 16
        && !init.data.is_null()
    {
        return Some(
            (0..init.count)
                .map(|index| unsafe {
                    std::ptr::read_unaligned(init.data.cast::<i64>().add(index))
                })
                .collect(),
        );
    }
    let node = producers.get(value.name.as_str())?;
    if !node.domain.is_empty() || node.op_type != "Constant" {
        return None;
    }
    let tensor = node.tensors.get("value")?;
    if tensor.dtype != INT64
        || tensor.shape.len() != 1
        || tensor.count > 16
        || tensor.data.len() != tensor.count * 8
    {
        return None;
    }
    Some(
        tensor
            .data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|bytes| i64::from_ne_bytes(*bytes))
            .collect(),
    )
}

fn reshape(shape: &[i32], requested: &[i64], allowzero: bool) -> Option<Vec<i32>> {
    let size = shape
        .iter()
        .try_fold(1i64, |size, &dim| size.checked_mul(i64::from(dim)))?;
    let mut inferred = None;
    let mut output = Vec::with_capacity(requested.len());
    let mut known = 1i64;
    for (index, &dim) in requested.iter().enumerate() {
        let dim = match dim {
            -1 if inferred.is_none() => {
                inferred = Some(index);
                1
            }
            0 if !allowzero => i64::from(*shape.get(index)?),
            dim if dim >= 0 => dim,
            _ => return None,
        };
        known = known.checked_mul(dim)?;
        output.push(i32::try_from(dim).ok()?);
    }
    if let Some(index) = inferred {
        if known == 0 || size % known != 0 {
            return None;
        }
        output[index] = i32::try_from(size / known).ok()?;
    } else if known != size {
        return None;
    }
    Some(output)
}

fn trace_weight(
    value: &TensorRef,
    producers: &HashMap<&str, &NodeDesc>,
    remaining: usize,
) -> Option<FusedWeight> {
    let node = *producers.get(value.name.as_str())?;
    if remaining == 0 || !node.domain.is_empty() || node.outputs.len() != 1 {
        return None;
    }
    if node.op_type == "DequantizePacked"
        && node.since_version == 29
        && node.inputs.len() == 1
        && matches!(node.outputs[0].otype, FLOAT | FLOAT16 | BFLOAT16)
    {
        let format = PackedFormat::parse(node.strings.get("format")?).ok()?;
        format.gpu_id()?;
        let shape = node
            .int_arrays
            .get("shape")?
            .iter()
            .map(|&dim| i32::try_from(dim).ok().filter(|&dim| dim >= 0))
            .collect::<Option<Vec<_>>>()?;
        if shape.is_empty()
            || *shape.last()? % format.geometry().0 as i32 != 0
            || strides(&shape).is_none()
        {
            return None;
        }
        return Some(FusedWeight {
            decode: node.clone(),
            format,
            shape,
            permutations: Vec::new(),
        });
    }
    let input = node.inputs.first()?;
    let mut weight = trace_weight(input, producers, remaining - 1)?;
    match node.op_type.as_str() {
        "Reshape" if node.inputs.len() == 2 => {
            weight.shape = reshape(
                &weight.shape,
                &literal_shape(&node.inputs[1], producers)?,
                node.ints.get("allowzero").copied().unwrap_or(0) != 0,
            )?;
        }
        "Transpose" if node.inputs.len() == 1 => {
            let rank = weight.shape.len();
            let default = (0..rank as i64).rev().collect::<Vec<_>>();
            let axes = node.int_arrays.get("perm").unwrap_or(&default);
            if axes.len() != rank {
                return None;
            }
            let mut sorted = axes.clone();
            sorted.sort_unstable();
            if sorted.iter().copied().ne(0..rank as i64) || weight.shape.contains(&0) {
                return None;
            }
            let source = strides(&weight.shape)?;
            let shape = axes
                .iter()
                .map(|&axis| weight.shape[axis as usize])
                .collect();
            weight.permutations.push(Permutation {
                dimensions: shape,
                source_strides: axes.iter().map(|&axis| source[axis as usize]).collect(),
            });
            weight.shape = weight.permutations.last()?.dimensions.clone();
        }
        _ => return None,
    }
    Some(weight)
}

fn recognize_consumer(
    node: &NodeDesc,
    producers: &HashMap<&str, &NodeDesc>,
) -> Option<FusedWeight> {
    if !node.domain.is_empty()
        || !matches!(
            node.op_type.as_str(),
            "MatMul" | "Gemm" | "GroupedMatMul" | "Gather" | "Conv"
        )
        || (node.op_type == "GroupedMatMul" && node.since_version != 29)
        || node.outputs.len() != 1
    {
        return None;
    }
    let slot = usize::from(node.op_type != "Gather");
    let weight = trace_weight(node.inputs.get(slot)?, producers, producers.len() + 1)?;
    let shape_matches = if node.op_type == "Conv" {
        let shape = weight
            .shape
            .iter()
            .copied()
            .map(i64::from)
            .collect::<Vec<_>>();
        super::super::conv::packed_pointwise_profile(
            &shape,
            node.ints.get("group").copied().unwrap_or(1),
            node.int_arrays.get("strides").map(Vec::as_slice),
            node.int_arrays.get("pads").map(Vec::as_slice),
        )
    } else {
        weight.shape.len()
            == if node.op_type == "GroupedMatMul" {
                3
            } else {
                2
            }
    };
    (shape_matches
        && node.outputs[0].otype == weight.decode.outputs[0].otype
        && (node.op_type != "Gather"
            || matches!(node.ints.get("axis").copied().unwrap_or(0), 0 | -2)))
    .then_some(weight)
}

fn literal_shape_source(
    node: &NodeView,
    views: &[NodeView],
    supported: &[bool],
    producers: &HashMap<String, usize>,
) -> Option<Option<usize>> {
    let info = node.input_info(1)?;
    if info.dtype != INT64 || info.shape.len() != 1 || !(0..=16).contains(&info.shape[0]) {
        return None;
    }
    if node.is_const_int64(1) {
        return Some(None);
    }
    let source = *producers.get(node.input_names().get(1)?)?;
    let view = &views[source];
    (supported[source]
        && view.domain().is_empty()
        && view.op_type() == "Constant"
        && view.attr_tensor_dtype("value") == Some(INT64))
    .then_some(Some(source))
}

fn partition_region(
    consumer: usize,
    views: &[NodeView],
    supported: &[bool],
    producers: &HashMap<String, usize>,
) -> Option<Vec<usize>> {
    let node = &views[consumer];
    let op = node.op_type();
    if !supported[consumer]
        || !node.domain().is_empty()
        || node.num_outputs() != 1
        || !matches!(
            op.as_str(),
            "MatMul" | "Gemm" | "GroupedMatMul" | "Gather" | "Conv"
        )
        || (op == "GroupedMatMul" && node.since_version() != 29)
        || (op == "Gather" && !matches!(node.int_attr("axis", 0), 0 | -2))
    {
        return None;
    }
    let slot = usize::from(op != "Gather");
    let info = node.input_info(slot)?;
    let shape_matches = if op == "Conv" {
        let (have_strides, strides) = node.ints_attr("strides");
        let (have_pads, pads) = node.ints_attr("pads");
        super::super::conv::packed_pointwise_profile(
            &info.shape,
            node.int_attr("group", 1),
            have_strides.then_some(strides.as_slice()),
            have_pads.then_some(pads.as_slice()),
        )
    } else {
        info.shape.len() == if op == "GroupedMatMul" { 3 } else { 2 }
    };
    if !shape_matches
        || !matches!(info.dtype, FLOAT | FLOAT16 | BFLOAT16)
        || node.output_info(0)?.dtype != info.dtype
    {
        return None;
    }
    let mut value = node.input_names().get(slot)?.clone();
    let mut region = vec![consumer];
    for _ in 0..views.len() {
        let producer = *producers.get(&value)?;
        let view = &views[producer];
        if !supported[producer] || !view.domain().is_empty() || view.num_outputs() != 1 {
            return None;
        }
        region.push(producer);
        match view.op_type().as_str() {
            "DequantizePacked" if view.since_version() == 29 && gpu::has_decoder(view) => {
                if let Some(source) = view.input_names().first().and_then(|v| producers.get(v))
                    && supported[*source]
                    && views[*source].op_type() == "Constant"
                    && views[*source].domain().is_empty()
                {
                    region.push(*source);
                }
                region.sort_unstable();
                region.dedup();
                return Some(region);
            }
            "Transpose" if view.num_inputs() == 1 => {}
            "Reshape" if view.num_inputs() == 2 => {
                region.extend(literal_shape_source(view, views, supported, producers)?);
            }
            _ => return None,
        }
        value = view.input_names().first()?.clone();
    }
    None
}

pub(crate) fn partition_regions(views: &[NodeView], supported: &[bool]) -> Vec<Vec<usize>> {
    if std::env::var_os("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION")
        .is_some_and(|v| v != "0" && !v.is_empty())
        || !views.iter().any(|v| v.op_type() == "DequantizePacked")
    {
        return Vec::new();
    }
    let producers = views
        .iter()
        .enumerate()
        .flat_map(|(index, view)| {
            view.output_names()
                .into_iter()
                .map(move |name| (name, index))
        })
        .collect();
    (0..views.len())
        .filter_map(|consumer| partition_region(consumer, views, supported, &producers))
        .collect()
}

pub(crate) fn recognize(nodes: &mut [NodeDesc]) {
    if std::env::var_os("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION")
        .is_some_and(|value| value != "0" && !value.is_empty())
    {
        return;
    }
    let found = {
        let producers = nodes
            .iter()
            .flat_map(|node| {
                node.outputs
                    .iter()
                    .map(move |out| (out.name.as_str(), node))
            })
            .collect();
        nodes
            .iter()
            .map(|node| recognize_consumer(node, &producers))
            .collect::<Vec<_>>()
    };
    for (node, weight) in nodes.iter_mut().zip(found) {
        if let Some(weight) = weight {
            node.packed_weight = Some(Box::new(weight));
            node.compile_shape_safety = CompileShapeSafety::EagerOnly {
                reason: "packed consumers validate live routing and specialize decoded weight tiles",
            };
        }
    }
}

impl FusedWeight {
    pub(crate) fn shape(&self) -> &[i32] {
        &self.shape
    }

    pub(crate) fn payload(&self, ctx: &mut TranslationContext) -> Result<mlx::mlx_array, MlxError> {
        let shape = self
            .decode
            .int_arrays
            .get("shape")
            .expect("recognized shape");
        dequantize_input(ctx, &self.decode, self.format, shape).map(|(packed, _, _)| packed)
    }

    pub(crate) fn matrix(
        &self,
        ctx: &mut TranslationContext,
        inputs: [mlx::mlx_array; 2],
        geometry: [i32; 3],
        shape: &[i32],
        trans_b: bool,
        routing: Option<&Routing>,
    ) -> Result<Option<mlx::mlx_array>, MlxError> {
        if !gpu::is_gpu(ctx)? {
            return Ok(None);
        }
        let [m, k, n] = geometry;
        let dtype = mlx_dtype_from_onnx(self.decode.outputs[0].otype);
        if shape.contains(&0) || k == 0 {
            return ctx.zeros(shape, dtype).map(Some);
        }
        let x = ctx.contiguous(inputs[0])?;
        if routing.is_none() && m > 4 {
            if let Some(transpose) = self.qmm_layout(k, n, trans_b)
                && let Some(y) = super::qmm::execute(
                    ctx,
                    &self.decode,
                    self.format,
                    [x, inputs[1]],
                    [m, k, n],
                    transpose,
                )?
            {
                return ctx.reshape(y, shape).map(Some);
            }
            // Dense tiled prefill lost to native GEMM in A/B measurements. Retain
            // the original GPU path until an equally accurate fused route wins.
            return Ok(None);
        }
        let one = ctx.scalar_f32(1.0);
        let zero = ctx.scalar_i32(0);
        let header = self.matmul_header(k, n, trans_b);
        let Some((y, kind)) = (fused::Matmul {
            operands: [x, inputs[1], one, one, zero, zero],
            m,
            k,
            n,
            routing,
            k_contiguous: self.k_contiguous(k, n, trans_b),
            header: &header,
            shape,
            dtype,
        })
        .run(ctx)?
        else {
            return Ok(None);
        };
        ctx.mark_fast(match kind {
            KernelKind::Matvec => "mlx_packed_matvec",
            KernelKind::Tiled => "mlx_packed_tiled_matmul",
        });
        Ok(Some(y))
    }

    fn header(&self) -> String {
        let mut header = format!(
            "#include <metal_simdgroup_matrix>\n{}\ninline size_t mlx_packed_source_index(size_t i) {{\n",
            gpu::DECODE
        );
        for permutation in self.permutations.iter().rev() {
            let output = strides(&permutation.dimensions).expect("validated dimensions");
            header.push_str("{ size_t mapped = 0;\n");
            for ((&dim, stride), &source) in permutation
                .dimensions
                .iter()
                .zip(output)
                .zip(&permutation.source_strides)
            {
                writeln!(
                    header,
                    "mapped += ((i / {stride}ul) % {dim}ul) * {source}ul;"
                )
                .expect("String writes cannot fail");
            }
            header.push_str("i = mapped; }\n");
        }
        header.push_str("return i;\n}\n");
        header
    }

    fn matmul_header(&self, k: i32, n: i32, trans_b: bool) -> String {
        let mut header = self.header();
        let logical = if trans_b {
            format!("(size_t(expert) * {n} + col) * {k} + k")
        } else {
            format!("(size_t(expert) * {k} + k) * {n} + col")
        };
        let rank = self.shape.len();
        let matrix_shape = (2..=3).contains(&rank)
            && self.shape[rank - 2 + usize::from(trans_b)] == k
            && self.shape[rank - 1 - usize::from(trans_b)] == n;
        // Final matrix transposes have affine coordinates, avoiding per-read uint64 division.
        let index = if matrix_shape
            && let [permutation] = self.permutations.as_slice()
            && permutation.dimensions == self.shape
        {
            let strides = &permutation.source_strides;
            let expert = if rank == 3 { strides[0] } else { 0 };
            let reduction = strides[rank - 2 + usize::from(trans_b)];
            let column = strides[rank - 1 - usize::from(trans_b)];
            format!(
                "size_t(expert) * {expert}ul + size_t(k) * {reduction}ul + size_t(col) * {column}ul"
            )
        } else {
            format!("mlx_packed_source_index({logical})")
        };
        let (block, bytes) = self.format.geometry();
        let format = self.format.gpu_id().expect("recognized format");
        write!(
            header,
            r#"
template <typename X, typename S, typename Z>
inline float mlx_weight_x(X x, S, Z, uint row, uint k) {{
    return float(x[size_t(row) * {k} + k]);
}}
template <typename W, typename S, typename Z>
inline float mlx_weight_w(W w, S, Z, uint expert, uint k, uint col) {{
    return mlx_packed_value<{format}, {block}, {bytes}>(w, {index});
}}
"#
        )
        .expect("String writes cannot fail");
        header
    }

    fn k_contiguous(&self, k: i32, n: i32, trans_b: bool) -> bool {
        if k <= 1 {
            return trans_b;
        }
        let mut i = if trans_b { 1 } else { n as usize };
        for permutation in self.permutations.iter().rev() {
            let output = strides(&permutation.dimensions).expect("validated dimensions");
            i = permutation
                .dimensions
                .iter()
                .zip(output)
                .zip(&permutation.source_strides)
                .map(|((&dim, stride), &source)| ((i / stride) % dim as usize) * source)
                .sum();
        }
        i == 1
    }

    fn qmm_layout(&self, k: i32, n: i32, trans_b: bool) -> Option<bool> {
        if self.permutations.is_empty() {
            return Some(trans_b);
        }
        let [permutation] = self.permutations.as_slice() else {
            return None;
        };
        if self.shape.len() != 2 || permutation.dimensions != self.shape {
            return None;
        }
        let reduction = permutation.source_strides[usize::from(trans_b)];
        let column = permutation.source_strides[usize::from(!trans_b)];
        if reduction == 1 && column == k as usize {
            Some(true)
        } else if reduction == n as usize && column == 1 {
            Some(false)
        } else {
            None
        }
    }
}

fn routing(
    ctx: &mut TranslationContext,
    node: &NodeDesc,
    xs: &[i32],
    ws: &[i32],
    bias: Option<mlx::mlx_array>,
) -> Result<Routing, MlxError> {
    let indices = ctx.resolve(&node.inputs[2])?;
    let shape = ctx.shape_of(indices);
    let wide = |shape: &[i32]| shape.iter().copied().map(i64::from).collect::<Vec<_>>();
    super::super::matmul::grouped_geometry(
        &wide(xs),
        &wide(ws),
        &wide(&shape),
        bias.map(|value| wide(&ctx.shape_of(value))).as_deref(),
        None,
    )?;
    fused::read_routing(ctx, &node.inputs[2], ws[0])
}

fn gather(
    ctx: &mut TranslationContext,
    node: &NodeDesc,
    pattern: &FusedWeight,
    packed: mlx::mlx_array,
) -> Result<bool, MlxError> {
    let indices = ctx.resolve(&node.inputs[1])?;
    if !matches!(
        ctx.dtype_of(indices),
        mlx::mlx_dtype__MLX_INT32 | mlx::mlx_dtype__MLX_INT64
    ) {
        return Err("Packed Gather: use int32 or int64 row indices".into());
    }
    let rows = i64::from(pattern.shape[0]);
    for (position, value) in ctx.read_ints_eval(&node.inputs[1])?.into_iter().enumerate() {
        if value < -rows || value >= rows {
            return Err(format!(
                "Packed Gather: indices[{position}]={value} is outside [-{rows},{rows}); select a row from the decoded table"
            ));
        }
    }
    let mut shape = ctx.shape_of(indices);
    let width = pattern.shape[1];
    shape.push(width);
    let Some(size) = shape
        .iter()
        .try_fold(1i32, |size, &dim| size.checked_mul(dim))
    else {
        return Ok(false);
    };
    let dtype = mlx_dtype_from_onnx(node.outputs[0].otype);
    let y = if size == 0 {
        ctx.zeros(&shape, dtype)?
    } else {
        let indices = ctx.contiguous(indices)?;
        let (block, bytes) = pattern.format.geometry();
        gpu::Kernel {
            name: c"mlx_packed_gather",
            source: include_str!("gather.metal"),
            header: &pattern.header(),
            templates: &[
                (
                    "FORMAT",
                    pattern.format.gpu_id().expect("recognized format"),
                ),
                ("BLOCK", block as i32),
                ("BYTES", bytes as i32),
                ("ROWS", pattern.shape[0]),
                ("WIDTH", width),
                ("SIZE", size),
            ],
            shape: &shape,
            dtype,
            grid: [size, 1, 1],
            group: [size.min(256), 1, 1],
        }
        .run(ctx, &[("packed", packed), ("indices", indices)])?
    };
    ctx.mark_fast("mlx_packed_gather");
    ctx.bind(&node.outputs[0], y);
    Ok(true)
}

fn product(
    ctx: &mut TranslationContext,
    node: &NodeDesc,
    pattern: &FusedWeight,
    packed: mlx::mlx_array,
) -> Result<bool, MlxError> {
    let grouped = node.op_type == "GroupedMatMul";
    let gemm = node.op_type == "Gemm";
    let trans_b = (grouped || gemm) && node.ints.get("transB").copied().unwrap_or(0) != 0;
    let mut x = ctx.resolve(&node.inputs[0])?;
    let dtype = mlx_dtype_from_onnx(node.outputs[0].otype);
    if ctx.dtype_of(x) != dtype || ctx.ndim(x) == 0 {
        return Ok(false);
    }
    if gemm && ctx.ndim(x) >= 2 && node.ints.get("transA").copied().unwrap_or(0) != 0 {
        let mut axes = (0..ctx.ndim(x) as i32).collect::<Vec<_>>();
        let rank = axes.len();
        axes.swap(rank - 2, rank - 1);
        x = ctx.transpose(x, &axes)?;
    }
    let xs = ctx.shape_of(x);
    let rank = pattern.shape.len();
    let n = pattern.shape[rank - 1 - usize::from(trans_b)];
    let k = pattern.shape[rank - 2 + usize::from(trans_b)];
    if xs.last() != Some(&k) {
        return Err(format!(
            "Packed {}: X shape {xs:?} does not match reduction dimension {k}; check the decoded shape and transA/transB",
            node.op_type
        ));
    }
    let Some(m) = xs[..xs.len() - 1]
        .iter()
        .try_fold(1i32, |m, &d| m.checked_mul(d))
    else {
        return Ok(false);
    };
    let bias = if grouped
        && node
            .inputs
            .get(3)
            .is_some_and(|value| value.source != Src::Absent)
    {
        let bias = ctx.resolve(&node.inputs[3])?;
        if ctx.dtype_of(bias) != dtype {
            return Err(
                "Packed GroupedMatMul: bias dtype must match X and the decoded weights".into(),
            );
        }
        Some(bias)
    } else {
        None
    };
    let routing = if grouped {
        Some(routing(ctx, node, &xs, &[pattern.shape[0], k, n], bias)?)
    } else {
        None
    };
    let selections = routing.as_ref().map_or(1, |routing| routing.selections);
    let Some(routes) = m.checked_mul(selections) else {
        return Ok(false);
    };
    let Some(total) = routes.checked_mul(n) else {
        return Ok(false);
    };
    let mut shape = xs[..xs.len() - 1].to_vec();
    if grouped {
        shape.push(selections);
    }
    shape.push(n);
    let Some(mut y) = pattern.matrix(
        ctx,
        [x, packed],
        [m, k, n],
        &shape,
        trans_b,
        routing.as_ref(),
    )?
    else {
        return Ok(false);
    };
    if grouped
        && total != 0
        && let Some(bias) = bias
    {
        let indices = routing.as_ref().expect("grouped routing").indices;
        let selected =
            ctx.emit(|out, stream| unsafe { mlx::mlx_take_axis(out, bias, indices, 0, stream) })?;
        y = ctx.binary(mlx::mlx_add, y, selected)?;
    }
    if gemm {
        y = super::super::matmul::gemm_epilogue(ctx, node, y)?;
    }
    ctx.bind(&node.outputs[0], y);
    Ok(true)
}

pub(crate) fn execute(ctx: &mut TranslationContext, node: &NodeDesc) -> Result<bool, MlxError> {
    let Some(pattern) = &node.packed_weight else {
        return Ok(false);
    };
    if !gpu::is_gpu(ctx)? {
        return Ok(false);
    }
    let packed = pattern.payload(ctx)?;
    if node.op_type == "Gather" {
        gather(ctx, node, pattern, packed)
    } else {
        product(ctx, node, pattern, packed)
    }
}
