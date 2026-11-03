//! Shape-keyed execution of the portable onnax broadcast-adjoint contract.
use crate::engine::{MlxError, NodeDesc, TranslationContext};
use crate::registry::{ClaimResult, NodeView, OpRegistration, OpRegistry};
use crate::require;
use crate::sys::{mlx, ort};

fn claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "SumToShape requires gradient and reference tensors"
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
        gradient.shape.len() >= reference.shape.len(),
        "SumToShape reference rank exceeds gradient rank"
    );
    let leading = gradient.shape.len() - reference.shape.len();
    require!(
        reference.shape.iter().enumerate().all(|(i, &t)| {
            let g = gradient.shape[leading + i];
            t < 0 || g < 0 || t == 1 || t == g
        }),
        "SumToShape incompatible broadcast extents"
    );
    Ok(())
}

fn translate(ctx: &mut TranslationContext, node: &NodeDesc) -> Result<(), MlxError> {
    let gradient = ctx.resolve(&node.inputs[0])?;
    let reference = ctx.resolve(&node.inputs[1])?;
    let source = ctx.shape_of(gradient);
    let target = ctx.shape_of(reference);
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
            op_type: "SumToShape_v1",
            min_opset: 1,
            max_opset: 1,
            handler: translate,
            claim,
        },
        "reduction axes specialize on both live tensor shapes",
    );
}
