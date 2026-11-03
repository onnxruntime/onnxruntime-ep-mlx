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
    if node.op_type() == "SumToShape_v2" {
        require!(
            reference.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                && reference.shape.len() == 1,
            "SumToShape_v2 requires an INT64 target shape vector"
        );
        require!(
            reference.shape[0] < 0 || reference.shape[0] as usize <= gradient.shape.len(),
            "SumToShape reference rank exceeds gradient rank"
        );
        return Ok(());
    }
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
    let source = ctx.shape_of(gradient);
    let target = if node.op_type == "SumToShape_v2" {
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
    } else {
        let reference = ctx.resolve(&node.inputs[1])?;
        ctx.shape_of(reference)
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
    for op_type in ["SumToShape_v1", "SumToShape_v2"] {
        registry.register_shape_keyed(
            OpRegistration {
                domain: "pkg.onnax",
                op_type,
                min_opset: 1,
                max_opset: 1,
                handler: translate,
                claim,
            },
            "host-selected reduction axes are fixed by the trace, beyond Primitive::output_shapes",
        );
    }
}
