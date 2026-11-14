pub(super) use crate::engine::{NodeDesc, Plan, TensorRef, TranslationContext};
use crate::engine::{OutRef, Src};
pub(super) use crate::mlx::{Array, Stream};
pub(super) use crate::registry::{CompileShapeSafety, MLX_SCAN_SHAPE_REASON};
pub(super) use crate::sys::{mlx, ort};

pub(super) fn reference(name: &str) -> TensorRef {
    TensorRef {
        name: name.into(),
        source: Src::Intermediate,
        ..TensorRef::absent()
    }
}

pub(super) fn output(name: &str) -> OutRef {
    OutRef {
        name: name.into(),
        external: false,
        ctx_index: 0,
        otype: ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    }
}

pub(super) fn bind_float(ctx: &mut TranslationContext, name: &str, shape: &[i32], data: &[f32]) {
    let value = ctx.keep(Array::from_data(
        data.as_ptr().cast(),
        shape,
        mlx::mlx_dtype__MLX_FLOAT32,
    ));
    ctx.bind(&output(name), value);
}

pub(super) fn bind_indices(ctx: &mut TranslationContext, shape: &[i32], data: &[i64], int32: bool) {
    let value = ctx.keep(Array::from_data(
        data.as_ptr().cast(),
        shape,
        mlx::mlx_dtype__MLX_INT64,
    ));
    let value = if int32 {
        ctx.astype(value, mlx::mlx_dtype__MLX_INT32).unwrap()
    } else {
        value
    };
    ctx.bind(&output("indices"), value);
}

pub(super) fn assert_output(
    ctx: &mut TranslationContext,
    name: &str,
    shape: &[i32],
    expected: &[f32],
) {
    let actual = ctx.resolve(&reference(name)).unwrap();
    assert_eq!(ctx.shape_of(actual), shape);
    let actual = ctx.contiguous_eval(actual).unwrap();
    let actual =
        unsafe { std::slice::from_raw_parts(mlx::mlx_array_data_float32(actual), expected.len()) };
    for (&a, &e) in actual.iter().zip(expected) {
        if e.is_nan() {
            assert!(a.is_nan());
        } else if e.is_infinite() {
            assert_eq!(a, e);
        } else {
            assert!((a - e).abs() <= 2e-5 * e.abs().max(1.0), "{a} != {e}");
        }
    }
}

pub(super) fn node(op: &str, inputs: &[&str], since_version: i32) -> NodeDesc {
    let mut node = NodeDesc::new(
        op.into(),
        "".into(),
        since_version,
        CompileShapeSafety::EagerOnly {
            reason: "direct native handler regression",
        },
    );
    node.inputs = inputs.iter().map(|name| reference(name)).collect();
    node.outputs = vec![output("y")];
    node
}
