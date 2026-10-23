//! Dense linear-algebra op handlers (MatMul, Gemm, com.microsoft::FusedMatMul).
//!
//! These use MLX's dense GEMM (`mlx_matmul`) with numpy/ONNX batch-dim broadcasting.
//! FusedMatMul widens scaled fp16/bf16 products to avoid premature output rounding. An empty product is
//! re-materialised as a clean, correctly-shaped zeros array (mlx_matmul leaves an empty result with
//! no backing buffer, which the boundary CopyOut cannot memcpy).

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::registry::{
    ClaimResult, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry, is_mlx_cpu_float,
};
use crate::sys::mlx;
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
    use super::fused_matmul_axes;

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
