//! Dense linear algebra and eager com.microsoft::CDist pairwise distances.
//!
//! Matrix products use MLX's dense GEMM (`mlx_matmul`) with numpy/ONNX broadcasting.
//! FusedMatMul widens scaled fp16/bf16 products to avoid premature output rounding. An empty product is
//! re-materialised as a clean, correctly-shaped zeros array (mlx_matmul leaves an empty result with
//! no backing buffer, which the boundary CopyOut cannot memcpy).
//! CDist uses provider-owned eager host arithmetic with no M*K*N temporary.

use super::host_scan::{layout, read_live};
use crate::engine::{MlxError, NodeDesc, Src, TranslationContext, dim_i32};
use crate::registry::{
    ClaimResult, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry, is_mlx_cpu_float,
};
use crate::sys::{mlx, ort};
use crate::{deny, require};

fn present(n: &NodeDesc, i: usize) -> bool {
    i < n.inputs.len() && n.inputs[i].source != Src::Absent
}

/// A dtype-matched scalar (float value cast to `dt`) so alpha/beta scaling keeps the GEMM dtype.
fn scalar_like(
    ctx: &mut TranslationContext,
    value: f32,
    dt: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    let s = ctx.scalar_f32(value);
    ctx.astype(s, dt)
}

const CDIST_EAGER_REASON: &str =
    "CDist uses live typed host reduction without a pairwise feature workspace";

fn cdist_geometry(a: &[i64], b: &[i64], output: Option<&[i64]>) -> Result<(), MlxError> {
    if a.len() != 2 || b.len() != 2 {
        return Err("CDist: A and B must have rank 2".into());
    }
    if a[1] >= 0 && b[1] >= 0 && a[1] != b[1] {
        return Err("CDist: A and B feature dimensions must match".into());
    }
    if let Some(y) = output
        && (y.len() != 2
            || (y[0] >= 0 && a[0] >= 0 && y[0] != a[0])
            || (y[1] >= 0 && b[0] >= 0 && y[1] != b[0]))
    {
        return Err("CDist: output must have shape [M,K]".into());
    }
    Ok(())
}

fn cdist_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2
            && node.num_outputs() == 1
            && node.input_present(0)
            && node.input_present(1)
            && node.output_present(0),
        "CDist requires exactly two present inputs and one present output"
    );
    let (a, b, y) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(y)) => (a, b, y),
        _ => deny!("CDist requires tensor type/shape information"),
    };
    require!(
        matches!(
            a.dtype,
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                | ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
        ) && a.dtype == b.dtype
            && a.dtype == y.dtype,
        "CDist A, B and C must share float32 or float64 dtype"
    );
    cdist_geometry(&a.shape, &b.shape, Some(&y.shape))?;
    require!(
        !node.has_attr("metric")
            || node.attr_type("metric") == ort::OrtOpAttrType_ORT_OP_ATTR_STRING,
        "CDist metric must be STRING"
    );
    let metric = if node.has_attr("metric") {
        node.string_attr("metric", "")
    } else {
        "sqeuclidean".into()
    };
    require!(
        matches!(metric.as_str(), "sqeuclidean" | "euclidean"),
        "CDist supports only sqeuclidean and euclidean"
    );
    Ok(())
}

fn cdist_values<T>(
    a: &[T],
    b: &[T],
    m: usize,
    k: usize,
    features: usize,
    finish: impl Fn(T) -> T,
) -> Vec<T>
where
    T: Copy
        + Default
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>,
{
    let mut result = Vec::with_capacity(m * k);
    for i in 0..m {
        for j in 0..k {
            let mut sum = T::default();
            for n in 0..features {
                let difference = a[i * features + n] - b[j * features + n];
                sum = sum + difference * difference;
            }
            result.push(finish(sum));
        }
    }
    result
}

fn cdist_op(ctx: &mut TranslationContext, node: &NodeDesc) -> Result<(), MlxError> {
    let (a, a_shape, dtype) = {
        let a = read_live(ctx, &node.inputs[0], "CDist")?;
        (a.data.to_vec(), a.shape, a.dtype)
    };
    let b = read_live(ctx, &node.inputs[1], "CDist")?;
    cdist_geometry(&a_shape, &b.shape, None)?;
    if dtype != b.dtype
        || !matches!(
            dtype,
            mlx::mlx_dtype__MLX_FLOAT32 | mlx::mlx_dtype__MLX_FLOAT64
        )
    {
        return Err("CDist: live inputs must share float32 or float64 dtype".into());
    }
    let shape = [a_shape[0], b.shape[0]];
    layout(&shape, dtype, "CDist")?;
    let (m, k, features) = (
        a_shape[0] as usize,
        b.shape[0] as usize,
        a_shape[1] as usize,
    );
    let metric = node
        .strings
        .get("metric")
        .map(String::as_str)
        .unwrap_or("sqeuclidean");
    if !matches!(metric, "sqeuclidean" | "euclidean") {
        return Err(format!("CDist: unsupported metric {metric}"));
    }
    let euclidean = metric == "euclidean";
    let bytes: Vec<u8> = if dtype == mlx::mlx_dtype__MLX_FLOAT64 {
        let decode = |data: &[u8]| {
            data.as_chunks::<8>()
                .0
                .iter()
                .map(|v| f64::from_ne_bytes(*v))
                .collect::<Vec<_>>()
        };
        cdist_values(&decode(&a), &decode(b.data), m, k, features, |sum| {
            if euclidean { sum.sqrt() } else { sum }
        })
        .into_iter()
        .flat_map(f64::to_ne_bytes)
        .collect()
    } else {
        let decode = |data: &[u8]| {
            data.as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_ne_bytes(*v))
                .collect::<Vec<_>>()
        };
        cdist_values(&decode(&a), &decode(b.data), m, k, features, |sum| {
            if euclidean { sum.sqrt() } else { sum }
        })
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect()
    };
    let shape = [dim_i32(shape[0])?, dim_i32(shape[1])?];
    let output = ctx.from_host(bytes.as_ptr().cast(), &shape, dtype);
    ctx.bind(&node.outputs[0], output);
    Ok(())
}

fn matmul_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    if super::qdq::execute(ctx, n)? {
        return Ok(());
    }
    let a = ctx.resolve(&n.inputs[0])?;
    let b = ctx.resolve(&n.inputs[1])?;
    let y = ctx.binary(mlx::mlx_matmul, a, b)?;
    ctx.mark_fast("mlx_matmul");
    if ctx.size_of(y) == 0 {
        let shp = ctx.shape_of(y);
        let dt = ctx.dtype_of(y);
        let z = ctx.zeros(&shp, dt)?;
        ctx.bind(&n.outputs[0], z);
        return Ok(());
    }
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

const GROUPED_EAGER_REASON: &str =
    "GroupedMatMul validates live int32/int64 expert indices before device dispatch";

pub(super) fn grouped_geometry(
    x: &[i64],
    w: &[i64],
    indices: &[i64],
    bias: Option<&[i64]>,
    output: Option<&[i64]>,
) -> Result<(), MlxError> {
    if x.len() != 2 || w.len() != 3 || indices.len() != 2 {
        return Err("GroupedMatMul: expected ranks 2, 3, 2 for input, weights, indices".into());
    }
    let compatible = |a: i64, b: i64| a < 0 || b < 0 || a == b;
    if !compatible(x[0], indices[0]) || !compatible(x[1], w[1]) {
        return Err("GroupedMatMul: token or reduction dimensions differ".into());
    }
    if let Some(b) = bias
        && (b.len() != 2 || !compatible(b[0], w[0]) || !compatible(b[1], w[2]))
    {
        return Err("GroupedMatMul: bias must have shape [G,N]".into());
    }
    if let Some(y) = output
        && (y.len() != 3
            || !compatible(y[0], x[0])
            || !compatible(y[1], indices[1])
            || !compatible(y[2], w[2]))
    {
        return Err("GroupedMatMul: output must have shape [M,S,N]".into());
    }
    Ok(())
}

fn grouped_claim(node: &NodeView) -> ClaimResult {
    require!(
        (3..=4).contains(&node.num_inputs())
            && node.num_outputs() == 1
            && (0..3).all(|i| node.input_present(i))
            && node.output_present(0),
        "GroupedMatMul requires three inputs, optional bias, and one output"
    );
    let (x, w, indices, y) = match (
        node.input_info(0),
        node.input_info(1),
        node.input_info(2),
        node.output_info(0),
    ) {
        (Some(x), Some(w), Some(i), Some(y)) => (x, w, i, y),
        _ => deny!("GroupedMatMul requires tensor type/shape information"),
    };
    require!(
        is_mlx_cpu_float(x.dtype) && w.dtype == x.dtype && y.dtype == x.dtype,
        "GroupedMatMul input, weights, output must share float16/bfloat16/float32/float64 dtype"
    );
    require!(
        crate::registry::is_int_index(indices.dtype),
        "GroupedMatMul indices must be int32 or int64"
    );
    let bias = if node.input_present(3) {
        let Some(b) = node.input_info(3) else {
            deny!("GroupedMatMul requires bias tensor type/shape information")
        };
        require!(
            b.dtype == x.dtype,
            "GroupedMatMul bias dtype must match input"
        );
        Some(b)
    } else {
        None
    };
    grouped_geometry(
        &x.shape,
        &w.shape,
        &indices.shape,
        bias.as_ref().map(|b| b.shape.as_slice()),
        Some(&y.shape),
    )?;
    Ok(())
}

pub(super) fn grouped_matmul_op(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
) -> Result<(), MlxError> {
    if super::qdq::execute(ctx, n)? {
        return Ok(());
    }
    let x = ctx.resolve(&n.inputs[0])?;
    let xs = ctx.shape_of(x);
    let w = ctx.resolve(&n.inputs[1])?;
    let ws = ctx.shape_of(w);
    let indices = ctx.resolve(&n.inputs[2])?;
    let ish = ctx.shape_of(indices);
    let bias = if present(n, 3) {
        Some(ctx.resolve(&n.inputs[3])?)
    } else {
        None
    };
    let to_i64 = |shape: &[i32]| shape.iter().copied().map(i64::from).collect::<Vec<_>>();
    grouped_geometry(
        &to_i64(&xs),
        &to_i64(&ws),
        &to_i64(&ish),
        bias.map(|b| to_i64(&ctx.shape_of(b))).as_deref(),
        None,
    )?;
    let dtype = ctx.dtype_of(x);
    if !matches!(
        dtype,
        mlx::mlx_dtype__MLX_FLOAT16
            | mlx::mlx_dtype__MLX_BFLOAT16
            | mlx::mlx_dtype__MLX_FLOAT32
            | mlx::mlx_dtype__MLX_FLOAT64
    ) || ctx.dtype_of(w) != dtype
        || bias.is_some_and(|b| ctx.dtype_of(b) != dtype)
        || !matches!(
            ctx.dtype_of(indices),
            mlx::mlx_dtype__MLX_INT32 | mlx::mlx_dtype__MLX_INT64
        )
    {
        return Err("GroupedMatMul: invalid live data or indices dtype".into());
    }
    // Check in int64 before the gather API converts to uint32: truncation could route to a
    // different expert, and negative indices are invalid here (unlike ONNX Gather).
    let values = ctx.read_ints_eval(&n.inputs[2])?;
    if values.len() != (i64::from(ish[0]) * i64::from(ish[1])) as usize {
        return Err("GroupedMatMul: cannot read complete group_indices".into());
    }
    if values.iter().any(|&i| i < 0 || i >= i64::from(ws[0])) {
        return Err("GroupedMatMul: group_indices values must be in [0,G)".into());
    }
    let (m, selections, width) = (xs[0], ish[1], ws[2]);
    let shape = [m, selections, width];
    if m == 0 || selections == 0 || width == 0 {
        let y = ctx.zeros(&shape, dtype)?;
        ctx.bind(&n.outputs[0], y);
        return Ok(());
    }
    let mut y = if xs[1] == 0 {
        ctx.zeros(&shape, dtype)?
    } else {
        let x = ctx.reshape(x, &[m, 1, xs[1]])?;
        let row_indices = ctx.arange(0.0, f64::from(m), 1.0, mlx::mlx_dtype__MLX_INT32)?;
        let row_indices = ctx.reshape(row_indices, &[m, 1])?;
        {
            if ctx.dtype_of(w) != dtype {
                return Err("GroupedMatMul: live weights dtype must match input".into());
            }
            let mut device = unsafe { mlx::mlx_device_new() };
            let mut kind = mlx::mlx_device_type__MLX_CPU;
            let rc = unsafe { mlx::mlx_stream_get_device(&mut device, ctx.stream()) };
            let rc = if rc == 0 {
                unsafe { mlx::mlx_device_get_type(&mut kind, device) }
            } else {
                rc
            };
            unsafe { mlx::mlx_device_free(device) };
            if rc != 0 {
                return Err("GroupedMatMul: cannot inspect MLX stream device".into());
            }
            if kind == mlx::mlx_device_type__MLX_CPU {
                // Pinned GatherMM CPU only supports float32. Ordinary CPU Matmul uses typed
                // GEMM, including cblas_dgemm for double, without narrowing weights or bias.
                let selected =
                    ctx.emit(|res, s| unsafe { mlx::mlx_take_axis(res, w, indices, 0, s) })?;
                let x = ctx.reshape(x, &[m, 1, 1, xs[1]])?;
                let result = ctx.binary(mlx::mlx_matmul, x, selected)?;
                ctx.mark_composed("typed CPU gather + matmul (GatherMM CPU is float32-only)");
                ctx.reshape(result, &shape)?
            } else {
                let result = ctx.emit(|res, s| unsafe {
                    mlx::mlx_gather_mm(res, x, w, row_indices, indices, false, s)
                })?;
                ctx.mark_fast("mlx_gather_mm");
                ctx.reshape(result, &shape)?
            }
        }
    };
    if let Some(b) = bias {
        let selected = ctx.emit(|res, s| unsafe { mlx::mlx_take_axis(res, b, indices, 0, s) })?;
        y = ctx.binary(mlx::mlx_add, y, selected)?;
    }
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn fused_matmul_axes(rank: usize, trans: bool, trans_batch: bool) -> Vec<i32> {
    let mut axes: Vec<i32> = (0..rank as i32).collect();
    if rank >= 2 {
        if trans_batch {
            axes[..rank - 1].rotate_left(1);
        }
        if trans {
            axes.swap(rank - 2, rank - 1);
        }
    }
    axes
}

fn fused_matmul_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let mut a = ctx.resolve(&n.inputs[0])?;
    let mut b = ctx.resolve(&n.inputs[1])?;
    let dtype = ctx.dtype_of(a);
    let flag = |name: &str| n.ints.get(name).copied().unwrap_or(0) != 0;
    let (trans_batch_a, trans_batch_b) = (flag("transBatchA"), flag("transBatchB"));
    let (rank_a, rank_b) = (ctx.ndim(a), ctx.ndim(b));
    if (trans_batch_a || trans_batch_b) && (rank_a < 3 || rank_a != rank_b) {
        return Err("FusedMatMul batch transpose requires equal input ranks >= 3".into());
    }
    for (value, rank, trans, trans_batch) in [
        (&mut a, rank_a, flag("transA"), trans_batch_a),
        (&mut b, rank_b, flag("transB"), trans_batch_b),
    ] {
        if rank >= 2 && (trans || trans_batch) {
            *value = ctx.transpose(*value, &fused_matmul_axes(rank, trans, trans_batch))?;
        }
    }

    if ctx.dim(a, rank_a as i32 - 1) == 0 {
        let product = ctx.binary(mlx::mlx_matmul, a, b)?;
        let y = ctx.zeros(&ctx.shape_of(product), dtype)?;
        ctx.mark_fast("mlx_matmul");
        ctx.bind(&n.outputs[0], y);
        return Ok(());
    }

    let alpha = n.floats.get("alpha").copied().unwrap_or(1.0);
    let narrow = dtype == mlx::mlx_dtype__MLX_FLOAT16 || dtype == mlx::mlx_dtype__MLX_BFLOAT16;
    // ORT CUDA uses a T-rounded alpha, but scales the FP32 accumulator before rounding Y to T.
    // Widening avoids premature output overflow/underflow and double rounding in a T matmul.
    if narrow && alpha != 1.0 {
        a = ctx.astype(a, mlx::mlx_dtype__MLX_FLOAT32)?;
        b = ctx.astype(b, mlx::mlx_dtype__MLX_FLOAT32)?;
    }
    let mut y = ctx.binary(mlx::mlx_matmul, a, b)?;
    if alpha != 1.0 {
        let mut scale = scalar_like(ctx, alpha, dtype)?;
        if narrow {
            scale = ctx.astype(scale, mlx::mlx_dtype__MLX_FLOAT32)?;
        }
        y = ctx.binary(mlx::mlx_multiply, y, scale)?;
        if narrow {
            y = ctx.astype(y, dtype)?;
        }
    }
    if ctx.size_of(y) == 0 {
        y = ctx.zeros(&ctx.shape_of(y), dtype)?;
    }
    ctx.mark_fast("mlx_matmul");
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn gemm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let mut a = ctx.resolve(&n.inputs[0])?;
    let mut b = ctx.resolve(&n.inputs[1])?;
    let dt = ctx.dtype_of(a);

    let trans_a = n.ints.get("transA").copied().unwrap_or(0) != 0;
    let trans_b = n.ints.get("transB").copied().unwrap_or(0) != 0;
    if trans_a {
        a = ctx.transpose(a, &[1, 0])?;
    }
    if trans_b {
        b = ctx.transpose(b, &[1, 0])?;
    }

    let alpha = n.floats.get("alpha").copied().unwrap_or(1.0);
    let beta = n.floats.get("beta").copied().unwrap_or(1.0);

    let mm = ctx.binary(mlx::mlx_matmul, a, b)?;
    let mut y = if alpha != 1.0 {
        let s = scalar_like(ctx, alpha, dt)?;
        ctx.binary(mlx::mlx_multiply, mm, s)?
    } else {
        mm
    };
    if present(n, 2) {
        let mut c = ctx.resolve(&n.inputs[2])?;
        if beta != 1.0 {
            let s = scalar_like(ctx, beta, dt)?;
            c = ctx.binary(mlx::mlx_multiply, c, s)?;
        }
        y = ctx.binary(mlx::mlx_add, y, c)?;
    }
    ctx.mark_fast("mlx_matmul");
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

// ---- claim predicates --------------------------------------------------------------------------

fn matmul_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() >= 1,
        "expects 2 inputs and 1+ outputs, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        is_mlx_cpu_float(a.dtype) && b.dtype == a.dtype && out.dtype == a.dtype,
        "inputs/output must share one float dtype (fp32/fp16/bf16/fp64), got {}, {} -> {}",
        crate::registry::ort_dtype_name(a.dtype),
        crate::registry::ort_dtype_name(b.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        !a.shape.is_empty() && !b.shape.is_empty(),
        "both inputs must have rank >= 1 (got ranks {} and {})",
        a.shape.len(),
        b.shape.len()
    );
    Ok(())
}

fn gemm_claim(node: &NodeView) -> ClaimResult {
    let nin = node.num_inputs();
    require!(
        (nin == 2 || nin == 3) && node.num_outputs() >= 1,
        "expects 2 or 3 inputs and 1+ outputs, got {}in/{}out",
        nin,
        node.num_outputs()
    );
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        is_mlx_cpu_float(a.dtype) && b.dtype == a.dtype && out.dtype == a.dtype,
        "A/B/output must share one float dtype (fp32/fp16/bf16/fp64), got {}, {} -> {}",
        crate::registry::ort_dtype_name(a.dtype),
        crate::registry::ort_dtype_name(b.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        a.shape.len() == 2 && b.shape.len() == 2,
        "A and B must both have rank 2 (got ranks {} and {})",
        a.shape.len(),
        b.shape.len()
    );
    if nin == 3 && node.input_present(2) {
        match node.input_info(2) {
            Some(c) if c.dtype == a.dtype && c.shape.len() <= 2 => {}
            Some(c) => deny!(
                "C must match dtype {} and have rank <= 2 (got dtype {}, rank {})",
                crate::registry::ort_dtype_name(a.dtype),
                crate::registry::ort_dtype_name(c.dtype),
                c.shape.len()
            ),
            None => deny!("C input has no tensor type/shape info"),
        }
    }
    Ok(())
}

fn fused_matmul_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "FusedMatMul expects exactly 2 inputs and 1 output"
    );
    matmul_claim(node)?;
    if node.int_attr("transBatchA", 0) != 0 || node.int_attr("transBatchB", 0) != 0 {
        let a = node.input_info(0).expect("matmul_claim checked input A");
        let b = node.input_info(1).expect("matmul_claim checked input B");
        require!(
            a.shape.len() >= 3 && a.shape.len() == b.shape.len(),
            "FusedMatMul batch transpose requires equal input ranks >= 3, got {} and {}",
            a.shape.len(),
            b.shape.len()
        );
    }
    Ok(())
}

pub fn register(registry: &mut OpRegistry) {
    registry.register_eager(
        OpRegistration {
            domain: "",
            op_type: "GroupedMatMul",
            min_opset: 29,
            max_opset: 29,
            handler: grouped_matmul_op,
            claim: grouped_claim,
        },
        GROUPED_EAGER_REASON,
    );
    registry.register_eager(
        OpRegistration {
            domain: "com.microsoft",
            op_type: "CDist",
            min_opset: 1,
            max_opset: 1,
            handler: cdist_op,
            claim: cdist_claim,
        },
        CDIST_EAGER_REASON,
    );
    registry.register_shape_keyed(
        OpRegistration {
            domain: "com.microsoft",
            op_type: "FusedMatMul",
            min_opset: 1,
            max_opset: 1,
            handler: fused_matmul_op,
            claim: fused_matmul_claim,
        },
        "FusedMatMul conservatively specializes empty-result materialization on live output extents",
    );
    registry.register_shapeless(OpRegistration {
        domain: "",
        op_type: "MatMul",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: matmul_op,
        claim: matmul_claim,
    });
    registry.register_shapeless(OpRegistration {
        domain: "",
        op_type: "Gemm",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: gemm_op,
        claim: gemm_claim,
    });
}

#[cfg(test)]
mod tests {
    use super::{CDIST_EAGER_REASON, cdist_geometry, cdist_values, fused_matmul_axes};

    #[test]
    fn grouped_matmul_geometry_and_version() {
        use super::grouped_geometry;
        assert!(
            grouped_geometry(
                &[2, 3],
                &[4, 3, 5],
                &[2, 2],
                Some(&[4, 5]),
                Some(&[2, 2, 5])
            )
            .is_ok()
        );
        assert!(
            grouped_geometry(
                &[-1, -1],
                &[-1, -1, -1],
                &[-1, -1],
                None,
                Some(&[-1, -1, -1])
            )
            .is_ok()
        );
        for (x, w, i, b, y) in [
            (vec![2], vec![4, 3, 5], vec![2, 2], None, None),
            (vec![2, 3], vec![4, 6, 5], vec![2, 2], None, None),
            (vec![2, 3], vec![4, 3, 5], vec![3, 2], None, None),
            (vec![2, 3], vec![4, 3, 5], vec![2, 2], Some(vec![5]), None),
            (
                vec![2, 3],
                vec![4, 3, 5],
                vec![2, 2],
                None,
                Some(vec![2, 5]),
            ),
        ] {
            assert!(grouped_geometry(&x, &w, &i, b.as_deref(), y.as_deref()).is_err());
        }
    }

    #[test]
    fn grouped_matmul_typed_dense_handlers() {
        use crate::engine::{NodeDesc, OutRef, Plan, Src, TensorRef, TranslationContext};
        use crate::mlx::{Array, Stream};
        use crate::registry::CompileShapeSafety;
        use crate::sys::{mlx, ort};
        let reference = |name: &str| TensorRef {
            name: name.into(),
            source: Src::Intermediate,
            ..TensorRef::absent()
        };
        let output = |name: &str| OutRef {
            name: name.into(),
            external: false,
            ctx_index: 0,
            otype: ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        };
        for cpu in [false, true] {
            for dtype in [
                mlx::mlx_dtype__MLX_FLOAT32,
                mlx::mlx_dtype__MLX_FLOAT16,
                mlx::mlx_dtype__MLX_BFLOAT16,
                mlx::mlx_dtype__MLX_FLOAT64,
            ] {
                if !cpu && dtype == mlx::mlx_dtype__MLX_FLOAT64 {
                    continue;
                }
                let stream = if cpu {
                    Stream::new_default_cpu()
                } else {
                    Stream::new_default_gpu()
                };
                let mut plan = Plan::new(vec![]);
                let mut ctx = TranslationContext::new(
                    &mut plan,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    stream.as_raw(),
                );
                let mut node = NodeDesc::new(
                    "GroupedMatMul".into(),
                    String::new(),
                    29,
                    CompileShapeSafety::EagerOnly {
                        reason: super::GROUPED_EAGER_REASON,
                    },
                );
                node.inputs = ["x", "w", "indices", "bias"].map(reference).to_vec();
                node.outputs = vec![output("y")];
                for (name, shape, values) in [
                    ("x", vec![2, 2], vec![1.0f64, 2.0, 3.0, 4.0]),
                    (
                        "w",
                        vec![3, 2, 2],
                        vec![
                            1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                        ],
                    ),
                    ("bias", vec![3, 2], vec![0.5f64, 0.25, 1.5, 1.25, 2.5, 2.25]),
                ] {
                    let array = if dtype == mlx::mlx_dtype__MLX_FLOAT64 {
                        ctx.keep(Array::from_data(values.as_ptr().cast(), &shape, dtype))
                    } else {
                        let values: Vec<_> = values.into_iter().map(|v| v as f32).collect();
                        ctx.keep(Array::from_data(
                            values.as_ptr().cast(),
                            &shape,
                            mlx::mlx_dtype__MLX_FLOAT32,
                        ))
                    };
                    let array = ctx.astype(array, dtype).unwrap();
                    ctx.bind(&output(name), array);
                }
                for index_dtype in [mlx::mlx_dtype__MLX_INT32, mlx::mlx_dtype__MLX_INT64] {
                    let routes = [2i64, 0, 2, 1];
                    let array = ctx.keep(Array::from_data(
                        routes.as_ptr().cast(),
                        &[2, 2],
                        mlx::mlx_dtype__MLX_INT64,
                    ));
                    let array = ctx.astype(array, index_dtype).unwrap();
                    ctx.bind(&output("indices"), array);
                    super::grouped_matmul_op(&mut ctx, &node).unwrap();
                    let y = ctx.resolve(&reference("y")).unwrap();
                    assert_eq!(ctx.dtype_of(y), dtype);
                    assert_eq!(ctx.shape_of(y), [2, 2, 2]);
                    let y = ctx.astype(y, mlx::mlx_dtype__MLX_FLOAT32).unwrap();
                    let y = ctx.contiguous_eval(y).unwrap();
                    let actual =
                        unsafe { std::slice::from_raw_parts(mlx::mlx_array_data_float32(y), 8) };
                    let mut expected = [33.5f32, 36.25, 7.5, 10.25, 73.5, 80.25, 44.5, 51.25];
                    if dtype == mlx::mlx_dtype__MLX_BFLOAT16 {
                        // BF16 rounds the exact 80.25 tie to even 80.0.
                        expected[5] = 80.0;
                    }
                    assert_eq!(
                        actual, expected,
                        "cpu={cpu}, dtype={dtype}, index_dtype={index_dtype}"
                    );
                }
                for (m, k, s, n, g) in [
                    (0, 3, 2, 4, 2),
                    (2, 3, 0, 4, 2),
                    (2, 0, 2, 4, 2),
                    (2, 3, 2, 0, 2),
                    (0, 3, 2, 4, 0),
                ] {
                    for (name, shape, dt) in [
                        ("x", vec![m, k], dtype),
                        ("w", vec![g, k, n], dtype),
                        ("indices", vec![m, s], mlx::mlx_dtype__MLX_INT64),
                        ("bias", vec![g, n], dtype),
                    ] {
                        let array = ctx.zeros(&shape, dt).unwrap();
                        ctx.bind(&output(name), array);
                    }
                    super::grouped_matmul_op(&mut ctx, &node).unwrap();
                    let y = ctx.resolve(&reference("y")).unwrap();
                    assert_eq!(ctx.shape_of(y), [m, s, n]);
                    ctx.contiguous_eval(y).unwrap();
                }
            }
        }
    }

    #[test]
    fn grouped_matmul_double_keeps_low_bits() {
        use crate::engine::{NodeDesc, OutRef, Plan, Src, TensorRef, TranslationContext};
        use crate::mlx::{Array, Stream};
        use crate::registry::CompileShapeSafety;
        use crate::sys::{mlx, ort};
        let stream = Stream::new_default_cpu();
        let mut plan = Plan::new(vec![]);
        let mut ctx = TranslationContext::new(
            &mut plan,
            std::ptr::null(),
            std::ptr::null_mut(),
            stream.as_raw(),
        );
        let mut node = NodeDesc::new(
            "GroupedMatMul".into(),
            String::new(),
            29,
            CompileShapeSafety::EagerOnly {
                reason: super::GROUPED_EAGER_REASON,
            },
        );
        let output = |name: &str| OutRef {
            name: name.into(),
            external: false,
            ctx_index: 0,
            otype: ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        };
        for (name, shape, values) in [
            ("x", vec![1, 1], vec![1.0f64]),
            ("w", vec![1, 1, 1], vec![2f64.powi(40)]),
            ("bias", vec![1, 1], vec![0.125]),
        ] {
            let a = ctx.keep(Array::from_data(
                values.as_ptr().cast(),
                &shape,
                mlx::mlx_dtype__MLX_FLOAT64,
            ));
            ctx.bind(&output(name), a);
        }
        let i = ctx.zeros(&[1, 1], mlx::mlx_dtype__MLX_INT64).unwrap();
        ctx.bind(&output("indices"), i);
        node.inputs = ["x", "w", "indices", "bias"]
            .map(|name| TensorRef {
                name: name.into(),
                source: Src::Intermediate,
                ..TensorRef::absent()
            })
            .to_vec();
        node.outputs = vec![output("y")];
        super::grouped_matmul_op(&mut ctx, &node).unwrap();
        let y = ctx
            .resolve(&TensorRef {
                name: "y".into(),
                source: Src::Intermediate,
                ..TensorRef::absent()
            })
            .unwrap();
        let y = ctx.contiguous_eval(y).unwrap();
        assert_eq!(
            unsafe { *mlx::mlx_array_data_float64(y) },
            2f64.powi(40) + 0.125
        );
    }

    #[test]
    fn cdist_geometry_and_zero_features() {
        assert!(cdist_geometry(&[2, 3], &[4, 3], Some(&[2, 4])).is_ok());
        assert!(cdist_geometry(&[-1, -1], &[-1, -1], Some(&[-1, -1])).is_ok());
        assert!(cdist_geometry(&[2, 3], &[4, 2], None).is_err());
        assert!(cdist_geometry(&[3], &[4, 3], None).is_err());
        for output in [vec![2], vec![7, 4], vec![2, 7]] {
            assert!(cdist_geometry(&[2, 3], &[4, 3], Some(&output)).is_err());
        }
        assert_eq!(cdist_values::<f64>(&[], &[], 2, 3, 0, |x| x), vec![0.0; 6]);
        assert!(cdist_values::<f32>(&[], &[], 0, 3, 0, |x| x).is_empty());
    }

    #[test]
    fn cdist_direct_difference_preserves_double_and_overflow() {
        let large = 2f64.powi(40);
        assert_eq!(
            cdist_values(&[large], &[large + 0.125], 1, 1, 1, |x| x),
            [0.015625]
        );
        assert_eq!(
            cdist_values(&[0.0f32, 0.0], &[3.0, 4.0], 1, 1, 2, f32::sqrt),
            [5.0]
        );
        assert!(cdist_values(&[f32::MAX], &[0.0], 1, 1, 1, |x| x)[0].is_infinite());
        assert!(cdist_values(&[f64::INFINITY], &[f64::INFINITY], 1, 1, 1, |x| x)[0].is_nan());
    }

    #[test]
    fn cdist_eager_descriptors_block_nested_capture() {
        use crate::engine::{NodeDesc, SubgraphDesc};
        use crate::registry::CompileShapeSafety;
        let node = NodeDesc::new(
            "CDist".into(),
            "com.microsoft".into(),
            1,
            CompileShapeSafety::EagerOnly {
                reason: CDIST_EAGER_REASON,
            },
        );
        assert!(!crate::compiled::decode_enabled(
            false,
            std::slice::from_ref(&node)
        ));
        assert!(!crate::compiled::general_enabled(
            false,
            std::slice::from_ref(&node)
        ));
        let mut attention = node.clone();
        attention.op_type = "GroupQueryAttention".into();
        attention.compile_shape_safety = CompileShapeSafety::Shapeless;
        assert!(!crate::compiled::prefill_enabled(
            false,
            &[attention, node.clone()]
        ));
        let mut parent = node.clone();
        parent.op_type = "If".into();
        parent.compile_shape_safety = CompileShapeSafety::Shapeless;
        parent.subgraphs.push(SubgraphDesc {
            attr_name: "then_branch".into(),
            input_names: vec![],
            output_names: vec![],
            nodes: vec![node],
        });
        assert!(!crate::compiled::decode_enabled(
            false,
            std::slice::from_ref(&parent)
        ));
        assert!(!crate::compiled::general_enabled(false, &[parent]));
    }

    #[test]
    fn fused_transpose_order() {
        assert_eq!(fused_matmul_axes(1, true, false), [0]);
        assert_eq!(fused_matmul_axes(2, true, false), [1, 0]);
        assert_eq!(fused_matmul_axes(3, false, true), [1, 0, 2]);
        assert_eq!(fused_matmul_axes(3, true, true), [1, 2, 0]);
        assert_eq!(fused_matmul_axes(5, false, true), [1, 2, 3, 0, 4]);
        assert_eq!(fused_matmul_axes(5, true, true), [1, 2, 3, 4, 0]);
    }
}
