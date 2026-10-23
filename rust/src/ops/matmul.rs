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
            data.chunks_exact(8)
                .map(|v| f64::from_ne_bytes(v.try_into().unwrap()))
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
            data.chunks_exact(4)
                .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
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
