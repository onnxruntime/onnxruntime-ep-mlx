//! Remaining C++-EP compatibility operators.

use crate::engine::{MlxError, NodeDesc, Src, TensorRef, TranslationContext, mlx_dtype_from_onnx};
use crate::registry::{
    ClaimResult, CompileShapeSafety, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry,
    is_mlx_cpu_float,
};
use crate::sys::{mlx, ort};
use crate::{deny, require};
use half::{bf16, f16};

fn dropout_present(n: &NodeDesc, index: usize) -> bool {
    n.inputs
        .get(index)
        .is_some_and(|input| input.source != Src::Absent)
}

fn dropout_scalar_shape(ctx: &mut TranslationContext, input: &TensorRef) -> Result<(), MlxError> {
    if !ctx.tensor_shape(input)?.is_empty() {
        return Err("MLX: Dropout controls must be scalar tensors".to_string());
    }
    Ok(())
}

fn dropout_training(ctx: &mut TranslationContext, input: &TensorRef) -> Result<bool, MlxError> {
    dropout_scalar_shape(ctx, input)?;
    if matches!(input.source, Src::Initializer | Src::CtxInput) {
        let host = ctx.raw_host(input)?;
        if host.dtype != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
            || host.count != 1
            || host.data.is_null()
        {
            return Err("MLX: Dropout training_mode must be a bool scalar".to_string());
        }
        return Ok(unsafe { *host.data.cast::<u8>() != 0 });
    }
    let value = ctx.resolve(input)?;
    if ctx.dtype_of(value) != mlx::mlx_dtype__MLX_BOOL {
        return Err("MLX: Dropout training_mode must be a bool scalar".to_string());
    }
    let value = ctx.contiguous_eval(value)?;
    let data = unsafe { mlx::mlx_array_data_bool(value) }.cast::<u8>();
    if data.is_null() {
        return Err("MLX: Dropout training_mode has no scalar data".to_string());
    }
    Ok(unsafe { *data != 0 })
}

fn dropout_ratio(ctx: &mut TranslationContext, input: &TensorRef) -> Result<f64, MlxError> {
    dropout_scalar_shape(ctx, input)?;
    if matches!(input.source, Src::Initializer | Src::CtxInput) {
        let host = ctx.raw_host(input)?;
        if host.count != 1 || host.data.is_null() {
            return Err("MLX: Dropout ratio has no scalar data".to_string());
        }
        return unsafe {
            match host.dtype {
                ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
                    Ok(host.data.cast::<f32>().read_unaligned() as f64)
                }
                ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => {
                    Ok(host.data.cast::<f64>().read_unaligned())
                }
                ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => {
                    Ok(f16::from_bits(host.data.cast::<u16>().read_unaligned()).to_f64())
                }
                ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => {
                    Ok(bf16::from_bits(host.data.cast::<u16>().read_unaligned()).to_f64())
                }
                _ => Err("MLX: Dropout ratio must be a floating scalar".to_string()),
            }
        };
    }
    let value = ctx.resolve(input)?;
    let dtype = ctx.dtype_of(value);
    if !matches!(
        dtype,
        mlx::mlx_dtype__MLX_FLOAT16
            | mlx::mlx_dtype__MLX_BFLOAT16
            | mlx::mlx_dtype__MLX_FLOAT32
            | mlx::mlx_dtype__MLX_FLOAT64
    ) {
        return Err("MLX: Dropout ratio must be a floating scalar".to_string());
    }
    let dtype = if dtype == mlx::mlx_dtype__MLX_FLOAT64 {
        dtype
    } else {
        mlx::mlx_dtype__MLX_FLOAT32
    };
    let value = ctx.astype(value, dtype)?;
    let value = ctx.contiguous_eval(value)?;
    if dtype == mlx::mlx_dtype__MLX_FLOAT64 {
        let data = unsafe { mlx::mlx_array_data_float64(value) };
        if data.is_null() {
            return Err("MLX: Dropout ratio has no scalar data".to_string());
        }
        Ok(unsafe { *data })
    } else {
        let data = unsafe { mlx::mlx_array_data_float32(value) };
        if data.is_null() {
            return Err("MLX: Dropout ratio has no scalar data".to_string());
        }
        Ok(unsafe { *data } as f64)
    }
}

fn dropout_threshold(ratio: f64) -> f32 {
    let threshold = ratio as f32;
    if (threshold as f64) < ratio {
        threshold.next_up()
    } else {
        threshold
    }
}

fn dropout_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let legacy_training = n.since_version < 7 && n.ints.get("is_test").copied().unwrap_or(0) == 0;
    let has_training = n.since_version >= 12 && dropout_present(n, 2);
    if (legacy_training || has_training) && (ctx.shapeless_compile() || ctx.shape_keyed_compile()) {
        return Err(
            "MLX: Dropout training and live training_mode require eager execution".to_string(),
        );
    }
    let training = if has_training {
        dropout_training(ctx, &n.inputs[2])?
    } else {
        legacy_training
    };
    let ratio = if n.since_version >= 12 && dropout_present(n, 1) {
        dropout_scalar_shape(ctx, &n.inputs[1])?;
        if training {
            dropout_ratio(ctx, &n.inputs[1])?
        } else {
            0.5
        }
    } else {
        n.floats.get("ratio").copied().unwrap_or(0.5) as f64
    };
    if training && (!ratio.is_finite() || !(0.0..1.0).contains(&ratio)) {
        return Err("MLX: Dropout training ratio must be finite and in [0, 1)".to_string());
    }
    let x = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(x);
    let mask_slot = n.outputs.get(1).filter(|output| !output.name.is_empty());
    if !training || ratio == 0.0 || shape.contains(&0) {
        ctx.bind(&n.outputs[0], x);
        if let Some(output) = mask_slot {
            let dtype = mlx_dtype_from_onnx(output.otype);
            let mask = ctx.emit(|res, s| unsafe {
                mlx::mlx_ones(res, shape.as_ptr(), shape.len(), dtype, s)
            })?;
            ctx.bind(output, mask);
        }
        return Ok(());
    }
    let key = if let Some(&seed) = n.ints.get("seed") {
        ctx.emit(|res, _| unsafe { mlx::mlx_random_key(res, seed as u64) })?
    } else {
        ctx.keep(crate::mlx::Array::new())
    };
    let zero = ctx.scalar_f32(0.0);
    let one = ctx.scalar_f32(1.0);
    let draws = ctx.emit(|res, s| unsafe {
        mlx::mlx_random_uniform(
            res,
            zero,
            one,
            shape.as_ptr(),
            shape.len(),
            mlx::mlx_dtype__MLX_FLOAT32,
            key,
            s,
        )
    })?;
    // Ceiling the threshold preserves comparisons of float32 draws against a float64 ratio.
    let threshold = ctx.scalar_f32(dropout_threshold(ratio));
    let mask = ctx.binary(mlx::mlx_greater_equal, draws, threshold)?;
    let dtype = ctx.dtype_of(x);
    let working = if dtype == mlx::mlx_dtype__MLX_FLOAT64 {
        dtype
    } else {
        mlx::mlx_dtype__MLX_FLOAT32
    };
    let x = ctx.astype(x, working)?;
    let numeric_mask = ctx.astype(mask, working)?;
    let masked = ctx.mul(x, numeric_mask)?;
    let scale = if working == mlx::mlx_dtype__MLX_FLOAT64 {
        ctx.scalar_f64(1.0 / (1.0 - ratio))
    } else {
        ctx.scalar_f32((1.0 / (1.0 - ratio)) as f32)
    };
    let output = ctx.mul(masked, scale)?;
    let output = ctx.astype(output, dtype)?;
    ctx.bind(&n.outputs[0], output);
    if let Some(output) = mask_slot {
        let mask = ctx.astype(mask, mlx_dtype_from_onnx(output.otype))?;
        ctx.bind(output, mask);
    }
    Ok(())
}

fn dropout_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() >= 1
            && node.num_inputs() <= (if node.since_version() >= 12 { 3 } else { 1 })
            && node.num_outputs() >= 1
            && node.num_outputs() <= 2
            && node.input_present(0)
            && node.output_present(0),
        "expects data, versioned optional controls and 1-2 outputs, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, y) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(y)) => (x, y),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        is_mlx_cpu_float(x.dtype) && y.dtype == x.dtype,
        "input/output must share a supported floating dtype, got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(y.dtype)
    );
    require!(
        node.since_version() >= 13
            || x.dtype != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
        "bfloat16 data requires Dropout opset 13"
    );
    require!(
        x.shape.iter().all(|&dim| dim <= i32::MAX as i64),
        "input dimensions exceed native int32 geometry"
    );
    require!(
        x.shape.is_empty()
            || y.shape.is_empty()
            || (x.shape.len() == y.shape.len()
                && x.shape
                    .iter()
                    .zip(&y.shape)
                    .all(|(&a, &b)| a < 0 || b < 0 || a == b)),
        "output must have the input shape"
    );
    if node.input_present(1) {
        let ratio = match node.input_info(1) {
            Some(info) => info,
            None => deny!("missing ratio tensor metadata"),
        };
        require!(
            is_mlx_cpu_float(ratio.dtype)
                && (node.since_version() >= 22
                    || ratio.dtype
                        != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
            "ratio must have a version-supported floating dtype"
        );
        require!(
            node.input_rank_known(1) != Some(true) || ratio.shape.is_empty(),
            "ratio must be a scalar"
        );
    }
    if node.input_present(2) {
        let training = match node.input_info(2) {
            Some(info) => info,
            None => deny!("missing training_mode tensor metadata"),
        };
        require!(
            training.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
                && (node.input_rank_known(2) != Some(true) || training.shape.is_empty()),
            "training_mode must be a bool scalar"
        );
    }
    if node.output_present(1) {
        let dtype = if node.since_version() < 10 {
            x.dtype
        } else {
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
        };
        match node.output_info(1) {
            Some(mask)
                if mask.dtype == dtype
                    && (mask.shape.is_empty()
                        || x.shape.is_empty()
                        || (mask.shape.len() == x.shape.len()
                            && mask
                                .shape
                                .iter()
                                .zip(&x.shape)
                                .all(|(&a, &b)| a < 0 || b < 0 || a == b))) => {}
            Some(mask) => deny!(
                "mask must have version-supported dtype {} and input shape {:?}, got dtype {} shape {:?}",
                crate::registry::ort_dtype_name(dtype),
                x.shape,
                crate::registry::ort_dtype_name(mask.dtype),
                mask.shape
            ),
            None => deny!("missing tensor type/shape info on mask output"),
        }
    }
    Ok(())
}

fn dropout_compile_shape_safety(node: &NodeView) -> CompileShapeSafety {
    if (node.since_version() < 7 && node.int_attr("is_test", 0) == 0)
        || (node.since_version() >= 12 && node.input_present(2))
    {
        CompileShapeSafety::EagerOnly {
            reason: "Dropout training requires fresh draws and live scalar controls",
        }
    } else {
        CompileShapeSafety::ShapeKeyedOnly {
            reason: "Dropout inference mask follows the live input geometry",
        }
    }
}

pub fn register(registry: &mut OpRegistry) {
    registry.register_shape_classified(
        OpRegistration {
            domain: "",
            op_type: "Dropout",
            min_opset: 1,
            max_opset: K_ANY_OPSET,
            handler: dropout_op,
            claim: dropout_claim,
        },
        dropout_compile_shape_safety,
    );
}

#[cfg(test)]
mod tests {
    use super::dropout_threshold;

    #[test]
    fn dropout_threshold_preserves_double_comparisons() {
        for ratio in [
            0.0,
            f64::from_bits(1),
            1e-100,
            f32::from_bits(1) as f64,
            f32::MIN_POSITIVE as f64,
            0.1,
            0.5 - f64::EPSILON,
            0.5,
            0.5 + f64::EPSILON,
            1.0 - f64::EPSILON,
            1.0 - f64::EPSILON / 2.0,
        ] {
            let nearest = ratio as f32;
            let threshold = dropout_threshold(ratio);
            for draw in [
                0.0,
                f32::from_bits(1),
                f32::MIN_POSITIVE,
                nearest.next_down(),
                nearest,
                nearest.next_up(),
                1.0,
            ] {
                assert_eq!((draw as f64) >= ratio, draw >= threshold);
            }
        }
    }
}
