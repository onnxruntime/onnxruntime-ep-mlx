//! Shape-keyed execution of the portable onnax broadcast-adjoint contract.
use crate::engine::{MlxError, NodeDesc, TranslationContext};
use crate::registry::{ClaimResult, NodeView, OpRegistration, OpRegistry};
use crate::require;
use crate::sys::{mlx, ort};

fn claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "SumToShape requires a gradient and target shape vector"
    );
    let gradient = node.input_info(0);
    let reference = node.input_info(1);
    require!(
        gradient.is_some() && reference.is_some(),
        "SumToShape requires tensor inputs"
    );
    let gradient = gradient.unwrap();
    let reference = reference.unwrap();
    require!(
        matches!(
            gradient.dtype,
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                | ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
                | ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
        ),
        "SumToShape supports FP32, FP16 and BF16 gradients"
    );
    require!(
        reference.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
            && reference.shape.len() == 1,
        "SumToShape requires an INT64 target shape vector"
    );
    require!(
        reference.shape[0] < 0 || reference.shape[0] as usize <= gradient.shape.len(),
        "SumToShape target rank exceeds gradient rank"
    );
    Ok(())
}

fn translate(ctx: &mut TranslationContext, node: &NodeDesc) -> Result<(), MlxError> {
    let gradient = ctx.resolve(&node.inputs[0])?;
    let source = ctx.shape_of(gradient);
    let target = {
        ctx.read_ints_eval(&node.inputs[1])?
            .into_iter()
            .map(|dimension| {
                i32::try_from(dimension)
                    .ok()
                    .filter(|&value| value >= 0)
                    .ok_or_else(|| {
                        "SumToShape target dimensions must be nonnegative int32".to_string()
                    })
            })
            .collect::<Result<Vec<_>, MlxError>>()?
    };
    if source.len() < target.len() {
        return Err("SumToShape reference rank exceeds gradient rank".into());
    }
    let leading = source.len() - target.len();
    if target
        .iter()
        .enumerate()
        .any(|(i, &t)| t != 1 && t != source[leading + i])
    {
        return Err("SumToShape incompatible live broadcast extents".into());
    }
    let value = if source.contains(&0) {
        ctx.zeros(&target, ctx.dtype_of(gradient))?
    } else {
        let axes: Vec<i32> = (0..source.len())
            .filter(|&i| i < leading || target[i - leading] == 1)
            .map(|i| i as i32)
            .collect();
        let reduced = if axes.is_empty() {
            gradient
        } else {
            ctx.emit(|res, stream| unsafe {
                mlx::mlx_sum_axes(res, gradient, axes.as_ptr(), axes.len(), true, stream)
            })?
        };
        ctx.reshape(reduced, &target)?
    };
    ctx.bind(&node.outputs[0], value);
    Ok(())
}

pub fn register(registry: &mut OpRegistry) {
    registry.register_shape_keyed(
        OpRegistration {
            domain: "pkg.onnax",
            op_type: "SumToShape",
            min_opset: 1,
            max_opset: 1,
            handler: translate,
            claim,
        },
        "host-selected reduction axes are fixed by the trace, beyond Primitive::output_shapes",
    );
}
