//! Remaining C++-EP compatibility operators.

use super::reduction::normalize_axes;
use crate::engine::{MlxError, NodeDesc, TranslationContext, dim_i32, mlx_dtype_from_onnx};
use crate::registry::{
    ClaimResult, CompileShapeSafety, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry,
    is_mlx_cpu_float, is_mlx_supported,
};
use crate::sys::{mlx, ort};
use crate::{deny, require};

fn mean_axes(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    axes: &[i32],
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_mean_axes(res, a, axes.as_ptr(), axes.len(), true, s) })
}

fn mean_variance_axes(raw: Option<&[i64]>, rank: usize) -> Result<Vec<i32>, MlxError> {
    let rank = i32::try_from(rank).map_err(|_| "MVN rank exceeds native axis range".to_string())?;
    let raw = raw.unwrap_or(&[0, 2, 3]);
    if raw.is_empty() {
        Ok((0..rank).collect())
    } else {
        normalize_axes(raw, rank)
    }
}

fn dropout_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    ctx.bind(&n.outputs[0], x);
    if n.outputs.len() > 1 && !n.outputs[1].name.is_empty() {
        let shape = ctx.shape_of(x);
        let mask = ctx.emit(|res, s| unsafe {
            mlx::mlx_ones(
                res,
                shape.as_ptr(),
                shape.len(),
                mlx::mlx_dtype__MLX_BOOL,
                s,
            )
        })?;
        ctx.bind(&n.outputs[1], mask);
    }
    Ok(())
}

fn mean_variance_normalization_op(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
) -> Result<(), MlxError> {
    let shape = ctx.tensor_shape(&n.inputs[0])?;
    let axes = mean_variance_axes(n.int_arrays.get("axes").map(Vec::as_slice), shape.len())?;
    let dtype = mlx_dtype_from_onnx(n.outputs[0].otype);
    if shape.contains(&0) {
        let shape = shape
            .into_iter()
            .map(dim_i32)
            .collect::<Result<Vec<_>, _>>()?;
        let output = ctx.zeros(&shape, dtype)?;
        ctx.bind(&n.outputs[0], output);
        return Ok(());
    }
    let x = ctx.resolve(&n.inputs[0])?;
    let working_dtype = if dtype == mlx::mlx_dtype__MLX_FLOAT64 {
        dtype
    } else {
        mlx::mlx_dtype__MLX_FLOAT32
    };
    let x = ctx.astype(x, working_dtype)?;
    let mean = mean_axes(ctx, x, &axes)?;
    let squared = ctx.mul(x, x)?;
    let mean_sq = mean_axes(ctx, squared, &axes)?;
    let mean_squared = ctx.mul(mean, mean)?;
    let variance = ctx.sub(mean_sq, mean_squared)?;
    let std = ctx.emit(|res, s| unsafe { mlx::mlx_sqrt(res, variance, s) })?;
    let scalar = ctx.scalar_f32(1e-9);
    let epsilon = ctx.astype(scalar, working_dtype)?;
    let denom = ctx.add(std, epsilon)?;
    let centered = ctx.sub(x, mean)?;
    let out = ctx.binary(mlx::mlx_divide, centered, denom)?;
    let out = ctx.astype(out, dtype)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn dropout_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() >= 1 && node.num_outputs() >= 1 && node.num_outputs() <= 2,
        "expects at least 1 input and 1-2 outputs, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, y) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(y)) => (x, y),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        is_mlx_supported(x.dtype) && y.dtype == x.dtype,
        "input/output must share one MLX-supported dtype, got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(y.dtype)
    );
    if node.output_present(1) {
        match node.output_info(1) {
            Some(mask)
                if mask.dtype
                    == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
                    && mask.shape == x.shape => {}
            Some(mask) => deny!(
                "mask must be bool with input shape {:?}, got dtype {} shape {:?}",
                x.shape,
                crate::registry::ort_dtype_name(mask.dtype),
                mask.shape
            ),
            None => deny!("missing tensor type/shape info on mask output"),
        }
    }
    Ok(())
}

fn mean_variance_normalization_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, y) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(y)) => (x, y),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        is_mlx_cpu_float(x.dtype) && y.dtype == x.dtype,
        "input/output must share one float dtype, got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(y.dtype)
    );
    require!(
        node.since_version() >= 13
            || x.dtype != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
        "bfloat16 requires MeanVarianceNormalization opset 13"
    );
    require!(
        x.shape.iter().all(|&dim| dim <= i32::MAX as i64),
        "input dimensions exceed native int32 geometry"
    );
    let (has_axes, axes) = node.ints_attr("axes");
    if node.input_rank_known(0) == Some(true) {
        mean_variance_axes(has_axes.then_some(axes.as_slice()), x.shape.len())?;
        require!(
            y.shape.is_empty()
                || (x.shape.len() == y.shape.len()
                    && x.shape
                        .iter()
                        .zip(&y.shape)
                        .all(|(&a, &b)| a < 0 || b < 0 || a == b)),
            "output must have the input shape"
        );
    }
    Ok(())
}

fn mean_variance_compile_shape_safety(node: &NodeView) -> CompileShapeSafety {
    if node.input_rank_known(0) == Some(true)
        && node
            .input_info(0)
            .is_some_and(|info| info.shape.iter().all(|&dim| dim > 0))
    {
        CompileShapeSafety::Shapeless
    } else {
        CompileShapeSafety::ShapeKeyedOnly {
            reason: "MeanVarianceNormalization validates live rank/axes and handles empty geometry",
        }
    }
}

pub fn register(registry: &mut OpRegistry) {
    registry.register_shapeless(OpRegistration {
        domain: "",
        op_type: "Dropout",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: dropout_op,
        claim: dropout_claim,
    });
    registry.register_shape_classified(
        OpRegistration {
            domain: "",
            op_type: "MeanVarianceNormalization",
            min_opset: 9,
            max_opset: K_ANY_OPSET,
            handler: mean_variance_normalization_op,
            claim: mean_variance_normalization_claim,
        },
        mean_variance_compile_shape_safety,
    );
}

#[cfg(test)]
mod tests {
    use super::mean_variance_axes;

    #[test]
    fn mean_variance_axes_follow_reduce_mean_contracts() {
        assert_eq!(mean_variance_axes(None, 4).unwrap(), [0, 2, 3]);
        assert_eq!(mean_variance_axes(Some(&[]), 3).unwrap(), [0, 1, 2]);
        assert_eq!(mean_variance_axes(Some(&[]), 0).unwrap(), []);
        assert_eq!(mean_variance_axes(Some(&[-1, 0]), 3).unwrap(), [2, 0]);
        assert!(mean_variance_axes(None, 3).is_err());
        assert!(mean_variance_axes(Some(&[0, -3]), 3).is_err());
        assert!(mean_variance_axes(Some(&[i64::MIN]), 3).is_err());
        assert!(mean_variance_axes(Some(&[i64::MAX]), 3).is_err());
    }
}
