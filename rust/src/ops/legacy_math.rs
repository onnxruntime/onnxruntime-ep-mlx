//! Deprecated schema-1 activations, evaluated eagerly on the provider-owned CPU path.

use crate::engine::{MlxError, NodeDesc, TranslationContext};
use crate::mlx::VectorArray;
use crate::registry::{ClaimResult, NodeView, OpRegistration, OpRegistry, is_mlx_cpu_float};
use crate::sys::{mlx, ort};
use crate::{deny, require};

fn map_values<T: Copy>(
    ctx: &mut TranslationContext,
    array: mlx::mlx_array,
    dtype: mlx::mlx_dtype,
    mut transform: impl FnMut(usize, T) -> T,
) -> Result<mlx::mlx_array, MlxError> {
    let count = ctx.size_of(array);
    let data = ctx.host_ptr(array).cast::<T>();
    let values = if count == 0 {
        &[]
    } else {
        if data.is_null() {
            return Err("MLX legacy activation: evaluated input data is unavailable".into());
        }
        unsafe { std::slice::from_raw_parts(data, count) }
    };
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|error| format!("MLX legacy activation: output allocation failed: {error}"))?;
    output.extend(values.iter().enumerate().map(|(i, &x)| transform(i, x)));
    let shape = ctx.shape_of(array);
    Ok(ctx.from_host(output.as_ptr().cast(), &shape, dtype))
}

fn evaluate(
    ctx: &mut TranslationContext,
    node: &NodeDesc,
    float: impl FnMut(usize, f32) -> f32,
    double: impl FnMut(usize, f64) -> f64,
) -> Result<(), MlxError> {
    if ctx.shapeless_compile() || ctx.shape_keyed_compile() {
        return Err("MLX legacy activation: host arithmetic requires eager execution".into());
    }
    let input = ctx.resolve(&node.inputs[0])?;
    let boundary_dtype = ctx.dtype_of(input);
    let working_dtype = if boundary_dtype == mlx::mlx_dtype__MLX_FLOAT64 {
        boundary_dtype
    } else {
        mlx::mlx_dtype__MLX_FLOAT32
    };
    // Keep half conversion on CPU too: Metal can flush a subnormal before its multiplier rescues it.
    let cpu = crate::mlx::Stream::new_default_cpu();
    let stream = cpu.as_raw();
    let input = ctx.emit(|out, _| unsafe { mlx::mlx_astype(out, input, working_dtype, stream) })?;
    let input = ctx.emit(|out, _| unsafe { mlx::mlx_contiguous(out, input, false, stream) })?;
    let mut arrays = VectorArray::new();
    arrays.append(input);
    crate::mlx::eval(&arrays)?;
    let result = if working_dtype == mlx::mlx_dtype__MLX_FLOAT64 {
        map_values(ctx, input, working_dtype, double)?
    } else {
        map_values(ctx, input, working_dtype, float)?
    };
    let result =
        ctx.emit(|out, _| unsafe { mlx::mlx_astype(out, result, boundary_dtype, stream) })?;
    ctx.bind(&node.outputs[0], result);
    Ok(())
}

fn softplus_float(x: f32, alpha: f32, beta: f32) -> f32 {
    let z = beta * x;
    if z == f32::INFINITY && x.is_finite() && beta.is_finite() && alpha.is_finite() {
        // FLOAT attributes have at most 24 significant bits: their product is exact in f64.
        // In this positive tail log1p(exp(-z)) rounds to zero, even before multiplication by alpha.
        return ((f64::from(alpha) * f64::from(beta)) * f64::from(x)) as f32;
    }
    if z < -80.0 && alpha.is_finite() {
        // Avoid rounding exp(z) to a coarse subnormal or zero before alpha can rescue it.
        let half = (z * 0.5).exp();
        return (alpha * half) * half;
    }
    alpha
        * if z > 0.0 {
            z + (-z).exp().ln_1p()
        } else {
            z.exp().ln_1p()
        }
}

fn softplus_double(x: f64, alpha: f32, beta: f32) -> f64 {
    let z = f64::from(beta) * x;
    if z == f64::INFINITY && x.is_finite() && beta.is_finite() && alpha.is_finite() {
        return (f64::from(alpha) * f64::from(beta)) * x;
    }
    if z < -700.0 && alpha.is_finite() {
        let half = (z * 0.5).exp();
        return (f64::from(alpha) * half) * half;
    }
    f64::from(alpha)
        * if z > 0.0 {
            z + (-z).exp().ln_1p()
        } else {
            z.exp().ln_1p()
        }
}

fn parametric_softplus_op(ctx: &mut TranslationContext, node: &NodeDesc) -> Result<(), MlxError> {
    let alpha = *node
        .floats
        .get("alpha")
        .ok_or("ParametricSoftplus requires alpha; its schema supplies no default")?;
    let beta = *node
        .floats
        .get("beta")
        .ok_or("ParametricSoftplus requires beta; its schema supplies no default")?;
    evaluate(
        ctx,
        node,
        |_, x| softplus_float(x, alpha, beta),
        |_, x| softplus_double(x, alpha, beta),
    )
}

fn image_geometry(shape: &[i32], bias_len: usize) -> Result<usize, MlxError> {
    let [_, channels, height, width] = shape else {
        return Err("MLX ImageScaler: input must have rank four [N,C,H,W]".into());
    };
    if *channels < 0 || *height < 0 || *width < 0 || *channels as usize != bias_len {
        return Err("MLX ImageScaler: bias must have one entry per runtime channel".into());
    }
    (*height as usize)
        .checked_mul(*width as usize)
        .ok_or_else(|| "MLX ImageScaler: spatial size overflow".into())
}

fn image_scaler_op(ctx: &mut TranslationContext, node: &NodeDesc) -> Result<(), MlxError> {
    let bias = node
        .float_arrays
        .get("bias")
        .ok_or("ImageScaler requires bias; its schema supplies no default")?;
    let input = ctx.resolve(&node.inputs[0])?;
    let plane = image_geometry(&ctx.shape_of(input), bias.len())?;
    let scale = node.floats.get("scale").copied().unwrap_or(1.0);
    evaluate(
        ctx,
        node,
        |i, x| scale * x + bias[(i / plane) % bias.len()],
        |i, x| f64::from(scale) * x + f64::from(bias[(i / plane) % bias.len()]),
    )
}

fn legacy_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects exactly one input and output"
    );
    let (Some(input), Some(output)) = (node.input_info(0), node.output_info(0)) else {
        deny!("missing input/output tensor information");
    };
    require!(
        is_mlx_cpu_float(input.dtype)
            && input.dtype != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
            && input.dtype == output.dtype,
        "requires matching FLOAT16, FLOAT, or DOUBLE tensors"
    );
    if node.op_type() == "ParametricSoftplus" {
        require!(
            node.float_attr_opt("alpha").is_some() && node.float_attr_opt("beta").is_some(),
            "alpha and beta must be FLOAT attributes; neither has a schema default"
        );
    } else {
        let Some(bias) = node.floats_attr("bias") else {
            deny!("bias must be an explicit FLOATS attribute");
        };
        require!(
            !node.has_attr("scale") || node.float_attr_opt("scale").is_some(),
            "scale must be FLOAT"
        );
        if node.input_rank_known(0) == Some(true) {
            require!(
                input.shape.len() == 4,
                "input must have rank four [N,C,H,W]"
            );
            let channels = input.shape[1];
            require!(
                channels < 0 || usize::try_from(channels).ok() == Some(bias.len()),
                "bias must contain exactly one value per channel"
            );
        }
    }
    Ok(())
}

pub(crate) fn requires_eager_boundary(node: &NodeView) -> bool {
    ((node.domain().is_empty() || node.domain() == "ai.onnx")
        && matches!(
            node.op_type().as_str(),
            "ParametricSoftplus" | "ImageScaler"
        ))
        || node
            .subgraphs()
            .iter()
            .any(|(_, body)| body.nodes().iter().any(requires_eager_boundary))
}

pub fn register(registry: &mut OpRegistry) {
    for (op_type, handler) in [
        (
            "ParametricSoftplus",
            parametric_softplus_op as crate::registry::OpHandler,
        ),
        ("ImageScaler", image_scaler_op as crate::registry::OpHandler),
    ] {
        registry.register_eager(
            OpRegistration {
                domain: "",
                op_type,
                min_opset: 1,
                max_opset: 9,
                handler,
                claim: legacy_claim,
            },
            "legacy activation uses live provider-owned CPU arithmetic",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_channels_and_empty_spatial_geometry() {
        assert_eq!(image_geometry(&[2, 3, 4, 5], 3).unwrap(), 20);
        assert_eq!(image_geometry(&[2, 0, 4, 5], 0).unwrap(), 20);
        assert_eq!(image_geometry(&[2, 3, 0, 5], 3).unwrap(), 0);
        assert!(image_geometry(&[2, 3, 4], 3).is_err());
        assert!(image_geometry(&[2, 3, 4, 5], 2).is_err());
    }

    #[test]
    fn finite_softplus_tail_does_not_overflow_before_alpha() {
        let alpha = 2.0f32.powi(-120);
        let beta = 2.0f32.powi(120);
        assert_eq!(
            softplus_float(2.0f32.powi(20), alpha, beta),
            2.0f32.powi(20)
        );
        assert_eq!(
            softplus_double(2.0f64.powi(1000), alpha, beta),
            2.0f64.powi(1000)
        );
        assert_eq!(softplus_float(2.0f32.powi(20), 0.0, beta), 0.0);
        assert!(softplus_float(f32::INFINITY, 0.0, beta).is_nan());
        assert_eq!(softplus_double(f64::NEG_INFINITY, 1.0, 1.0), 0.0);
        assert_eq!(softplus_double(f64::MAX, 0.5, 2.0), f64::MAX);
        assert_eq!(softplus_double(-f64::MAX, 0.5, -2.0), f64::MAX);
    }
}
