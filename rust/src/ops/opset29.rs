//! Stable scalar transforms and cumulative log-sum-exp introduced in ONNX opset 29.

use crate::engine::{MlxError, NodeDesc, TranslationContext};
use crate::registry::{
    ClaimResult, K_ANY_OPSET, MLX_SCAN_SHAPE_REASON, NodeView, OpRegistration, OpRegistry,
    is_mlx_cpu_float,
};
use crate::sys::mlx;
use crate::{deny, require};

fn float_unary_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "{} requires one input and one output",
        node.op_type()
    );
    let (Some(x), Some(y)) = (node.input_info(0), node.output_info(0)) else {
        deny!(
            "{} requires input/output tensor type information",
            node.op_type()
        );
    };
    require!(
        is_mlx_cpu_float(x.dtype) && y.dtype == x.dtype,
        "{} requires matching float16/bfloat16/float32/float64 input and output",
        node.op_type()
    );
    Ok(())
}

fn expm1_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let y = ctx.unary(mlx::mlx_expm1, x)?;
    let y = preserve_zero(ctx, x, y)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn log1p_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let y = ctx.unary(mlx::mlx_log1p, x)?;
    let y = preserve_zero(ctx, x, y)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn preserve_zero(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    y: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    // The Accelerate Expm1 implementation canonicalizes negative zero.
    let zero = ctx.scalar_f32(0.0);
    let zero = ctx.astype(zero, ctx.dtype_of(x))?;
    let is_zero = ctx.binary(mlx::mlx_equal, x, zero)?;
    ctx.where_(is_zero, x, y)
}

fn scan_axis(raw: i64, rank: usize) -> Result<i32, MlxError> {
    let rank = rank as i64;
    if rank == 0 || raw < -rank || raw >= rank {
        return Err(format!(
            "LogCumSumExp: axis {raw} is invalid for rank {rank}; use a rank >= 1 tensor and an axis in [-rank, rank-1]"
        ));
    }
    Ok(if raw < 0 { raw + rank } else { raw } as i32)
}

fn logcumsumexp_claim(node: &NodeView) -> ClaimResult {
    float_unary_claim(node)?;
    if node.input_rank_known(0) == Some(true) {
        let x = node.input_info(0).expect("checked unary tensor");
        scan_axis(node.int_attr("axis", 0), x.shape.len())?;
    }
    Ok(())
}

fn logcumsumexp_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let axis = scan_axis(n.ints.get("axis").copied().unwrap_or(0), ctx.ndim(x))?;
    let dtype = ctx.dtype_of(x);
    if ctx.size_of(x) == 0 {
        let y = ctx.zeros(&ctx.shape_of(x), dtype)?;
        ctx.bind(&n.outputs[0], y);
        return Ok(());
    }
    let work = if dtype == mlx::mlx_dtype__MLX_FLOAT64 {
        x
    } else {
        ctx.astype(x, mlx::mlx_dtype__MLX_FLOAT32)?
    };
    let y =
        ctx.emit(|res, s| unsafe { mlx::mlx_logcumsumexp_axis(res, work, axis, false, true, s) })?;
    let y = ctx.astype(y, dtype)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

pub fn register(registry: &mut OpRegistry) {
    crate::ops::packed::register(registry);
    for (op_type, handler) in [
        ("Expm1", expm1_op as crate::registry::OpHandler),
        ("Log1p", log1p_op as crate::registry::OpHandler),
    ] {
        registry.register_shapeless(OpRegistration {
            domain: "",
            op_type,
            min_opset: 29,
            max_opset: K_ANY_OPSET,
            handler,
            claim: float_unary_claim,
        });
    }
    registry.register_shape_keyed(
        OpRegistration {
            domain: "",
            op_type: "LogCumSumExp",
            min_opset: 29,
            max_opset: K_ANY_OPSET,
            handler: logcumsumexp_op,
            claim: logcumsumexp_claim,
        },
        MLX_SCAN_SHAPE_REASON,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{OutRef, Plan, Src, TensorRef};
    use crate::mlx::{Array, Stream};
    use crate::registry::CompileShapeSafety;
    use crate::sys::ort;

    fn reference(name: &str) -> TensorRef {
        TensorRef {
            name: name.into(),
            source: Src::Intermediate,
            ..TensorRef::absent()
        }
    }

    fn output(name: &str) -> OutRef {
        OutRef {
            name: name.into(),
            external: false,
            ctx_index: 0,
            otype: ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        }
    }

    #[test]
    fn stable_unary_tiny_values_signed_zero_and_domains() {
        for cpu in [false, true] {
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
            let values = [
                0.0f32,
                -0.0,
                1e-8,
                -1e-8,
                -1.0,
                -2.0,
                f32::NEG_INFINITY,
                f32::INFINITY,
                f32::NAN,
            ];
            let x = ctx.keep(Array::from_data(
                values.as_ptr().cast(),
                &[values.len() as i32],
                mlx::mlx_dtype__MLX_FLOAT32,
            ));
            ctx.bind(&output("x"), x);
            for (op, handler, expected) in [
                (
                    "Expm1",
                    expm1_op as crate::registry::OpHandler,
                    values.map(f32::exp_m1),
                ),
                (
                    "Log1p",
                    log1p_op as crate::registry::OpHandler,
                    values.map(f32::ln_1p),
                ),
            ] {
                let mut n = NodeDesc::new(op.into(), "".into(), 29, CompileShapeSafety::Shapeless);
                n.inputs = vec![reference("x")];
                n.outputs = vec![output("y")];
                handler(&mut ctx, &n).unwrap();
                let y = ctx.resolve(&reference("y")).unwrap();
                let y = ctx.contiguous_eval(y).unwrap();
                let actual = unsafe {
                    std::slice::from_raw_parts(mlx::mlx_array_data_float32(y), values.len())
                };
                for (&a, e) in actual.iter().zip(expected) {
                    if e.is_nan() {
                        assert!(a.is_nan(), "{op}, cpu={cpu}: {a} should be NaN");
                    } else if e == 0.0 || e.is_infinite() {
                        assert_eq!(a.to_bits(), e.to_bits(), "{op}, cpu={cpu}");
                    } else {
                        assert!(
                            (a - e).abs() <= 2e-6 * e.abs(),
                            "{op}, cpu={cpu}: {a} != {e}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn integer_gemm29_exact_values_transposes_broadcast_and_empty_geometry() {
        use crate::ops::host_scan::read_live;
        use crate::sys::ort;
        for (dtype, otype) in [
            (
                mlx::mlx_dtype__MLX_INT32,
                ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            ),
            (
                mlx::mlx_dtype__MLX_UINT32,
                ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
            ),
            (
                mlx::mlx_dtype__MLX_INT64,
                ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            ),
            (
                mlx::mlx_dtype__MLX_UINT64,
                ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
            ),
        ] {
            let stream = Stream::new_default_gpu();
            let mut plan = Plan::new(vec![]);
            let mut ctx = TranslationContext::new(
                &mut plan,
                std::ptr::null(),
                std::ptr::null_mut(),
                stream.as_raw(),
            );
            let encode = |values: &[i128]| -> Vec<u8> {
                values
                    .iter()
                    .flat_map(|&value| match dtype {
                        mlx::mlx_dtype__MLX_INT32 => (value as i32).to_ne_bytes().to_vec(),
                        mlx::mlx_dtype__MLX_UINT32 => (value as u32).to_ne_bytes().to_vec(),
                        mlx::mlx_dtype__MLX_INT64 => (value as i64).to_ne_bytes().to_vec(),
                        _ => (value as u64).to_ne_bytes().to_vec(),
                    })
                    .collect()
            };
            let mut n = NodeDesc::new(
                "Gemm".into(),
                "".into(),
                29,
                CompileShapeSafety::EagerOnly {
                    reason: "checked integer arithmetic",
                },
            );
            n.inputs = vec![reference("a"), reference("b"), reference("c")];
            n.outputs = vec![output("y")];
            n.outputs[0].otype = otype;
            let result_ref = reference("y");
            n.ints.insert("transA".into(), 1);
            n.floats.insert("alpha".into(), 2.0);
            n.floats.insert("beta".into(), -1.0);
            for trans_b in [0, 1] {
                n.ints.insert("transB".into(), trans_b);
                for (name, shape, values) in [
                    ("a", vec![2, 2, 3], (1..=12).collect::<Vec<i128>>()),
                    (
                        "b",
                        vec![2, 2],
                        if trans_b == 0 {
                            vec![1, 2, 3, 4]
                        } else {
                            vec![1, 3, 2, 4]
                        },
                    ),
                    ("c", vec![2], vec![1, 2]),
                ] {
                    let bytes = encode(&values);
                    let array = ctx.from_host(bytes.as_ptr().cast(), &shape, dtype);
                    ctx.bind(&output(name), array);
                }
                crate::registry::translate(&mut ctx, &n).unwrap();
                let result = read_live(&mut ctx, &result_ref, "test").unwrap();
                assert_eq!(result.shape, [2, 3, 2]);
                assert_eq!(
                    result.data,
                    encode(&[25, 34, 33, 46, 41, 58, 73, 106, 81, 118, 89, 130])
                );
            }
            n.ints.insert("transB".into(), 0);
            n.floats.insert("alpha".into(), 1.0);
            n.floats.insert("beta".into(), 0.0);
            let large = if matches!(
                dtype,
                mlx::mlx_dtype__MLX_INT64 | mlx::mlx_dtype__MLX_UINT64
            ) {
                (1i128 << 53) + 1
            } else {
                (1i128 << 25) + 1
            };
            for (name, shape, values) in [
                ("a", vec![2], vec![large, 2]),
                ("b", vec![2, 1], vec![1, 1]),
                ("c", vec![1], vec![1]),
            ] {
                let bytes = encode(&values);
                let array = ctx.from_host(bytes.as_ptr().cast(), &shape, dtype);
                ctx.bind(&output(name), array);
            }
            crate::registry::translate(&mut ctx, &n).unwrap();
            let result = read_live(&mut ctx, &result_ref, "test").unwrap();
            assert_eq!(result.shape, [1]);
            assert_eq!(result.data, encode(&[large + 2]));
            n.floats.insert("alpha".into(), 0.5);
            assert!(
                crate::registry::translate(&mut ctx, &n)
                    .unwrap_err()
                    .contains("finite integral")
            );
            n.floats.insert("alpha".into(), 2f32.powi(100));
            assert!(
                crate::registry::translate(&mut ctx, &n)
                    .unwrap_err()
                    .contains("exceeds")
            );
            n.ints.insert("transA".into(), 0);
            n.floats.insert("alpha".into(), 1.0);
            n.inputs.truncate(2);
            for (a_shape, b_shape, expected_shape) in [
                (vec![0, 2], vec![2, 3], vec![0, 3]),
                (vec![2, 0], vec![0, 3], vec![2, 3]),
                (vec![4, 2], vec![2, 0], vec![4, 0]),
            ] {
                for (name, shape) in [("a", a_shape), ("b", b_shape)] {
                    let array = ctx.zeros(&shape, dtype).unwrap();
                    ctx.bind(&output(name), array);
                }
                crate::registry::translate(&mut ctx, &n).unwrap();
                let result = read_live(&mut ctx, &result_ref, "test").unwrap();
                assert_eq!(result.shape, expected_shape);
                assert!(result.data.iter().all(|&value| value == 0));
            }
        }
    }

    #[test]
    fn logcumsumexp_stable_prefixes_and_nonfinite_order() {
        for cpu in [false, true] {
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
            let mut n = NodeDesc::new(
                "LogCumSumExp".into(),
                "".into(),
                29,
                CompileShapeSafety::ShapeKeyedOnly {
                    reason: MLX_SCAN_SHAPE_REASON,
                },
            );
            n.ints.insert("axis".into(), -1);
            n.inputs = vec![reference("x")];
            n.outputs = vec![output("y")];
            for values in [
                vec![-1000.0f32, -1001.0, 1000.0, 1001.0],
                vec![
                    f32::NEG_INFINITY,
                    f32::NEG_INFINITY,
                    2.0,
                    f32::INFINITY,
                    3.0,
                    f32::NAN,
                ],
                vec![f32::NAN, f32::INFINITY, 0.0],
            ] {
                let x = ctx.keep(Array::from_data(
                    values.as_ptr().cast(),
                    &[1, values.len() as i32],
                    mlx::mlx_dtype__MLX_FLOAT32,
                ));
                ctx.bind(&output("x"), x);
                logcumsumexp_op(&mut ctx, &n).unwrap();
                let y = ctx.resolve(&reference("y")).unwrap();
                let y = ctx.contiguous_eval(y).unwrap();
                let actual = unsafe {
                    std::slice::from_raw_parts(mlx::mlx_array_data_float32(y), values.len())
                };
                let mut sum = f64::NEG_INFINITY;
                for (&value, &a) in values.iter().zip(actual) {
                    let value = f64::from(value);
                    sum = if sum.is_nan() || value.is_nan() {
                        f64::NAN
                    } else if sum == f64::NEG_INFINITY {
                        value
                    } else if value == f64::NEG_INFINITY {
                        sum
                    } else if sum == f64::INFINITY || value == f64::INFINITY {
                        f64::INFINITY
                    } else {
                        sum.max(value) + (-(sum - value).abs()).exp().ln_1p()
                    };
                    if sum.is_nan() {
                        assert!(a.is_nan());
                    } else if sum.is_infinite() {
                        assert_eq!(f64::from(a), sum);
                    } else {
                        assert!((f64::from(a) - sum).abs() < 2e-4, "{a} != {sum}");
                    }
                }
            }
            for shape in [vec![0, 2], vec![2, 0]] {
                let x = ctx.zeros(&shape, mlx::mlx_dtype__MLX_FLOAT32).unwrap();
                ctx.bind(&output("x"), x);
                logcumsumexp_op(&mut ctx, &n).unwrap();
                let y = ctx.resolve(&reference("y")).unwrap();
                assert_eq!(ctx.shape_of(y), shape);
            }
        }
        assert!(scan_axis(0, 0).is_err());
        assert!(scan_axis(i64::MIN, 2).is_err());
        assert!(scan_axis(2, 2).is_err());
    }

    fn bind_float(ctx: &mut TranslationContext, name: &str, shape: &[i32], data: &[f32]) {
        let value = ctx.keep(Array::from_data(
            data.as_ptr().cast(),
            shape,
            mlx::mlx_dtype__MLX_FLOAT32,
        ));
        ctx.bind(&output(name), value);
    }

    fn bind_indices(ctx: &mut TranslationContext, shape: &[i32], data: &[i64], int32: bool) {
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

    fn assert_output(ctx: &mut TranslationContext, name: &str, shape: &[i32], expected: &[f32]) {
        let actual = ctx.resolve(&reference(name)).unwrap();
        assert_eq!(ctx.shape_of(actual), shape);
        let actual = ctx.contiguous_eval(actual).unwrap();
        let actual = unsafe {
            std::slice::from_raw_parts(mlx::mlx_array_data_float32(actual), expected.len())
        };
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

    fn node(op: &str, inputs: &[&str]) -> NodeDesc {
        let mut node = NodeDesc::new(
            op.into(),
            "".into(),
            29,
            CompileShapeSafety::EagerOnly {
                reason: "direct native handler regression",
            },
        );
        node.inputs = inputs.iter().map(|name| reference(name)).collect();
        node.outputs = vec![output("y")];
        node
    }

    #[test]
    fn version29_gemm_grouped_mvn_and_dilated_causal_contracts() {
        for cpu in [false, true] {
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
            bind_float(
                &mut ctx,
                "x",
                &[2, 2, 3],
                &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            );
            bind_float(&mut ctx, "w", &[2, 2], &[1., 2., 3., 4.]);
            bind_float(&mut ctx, "bias", &[2], &[0.5, -0.5]);
            let mut gemm = node("Gemm", &["x", "w", "bias"]);
            gemm.ints.insert("transA".into(), 1);
            crate::registry::translate(&mut ctx, &gemm).unwrap();
            assert_output(
                &mut ctx,
                "y",
                &[2, 3, 2],
                &[
                    13.5, 17.5, 17.5, 23.5, 21.5, 29.5, 37.5, 53.5, 41.5, 59.5, 45.5, 65.5,
                ],
            );
            bind_float(&mut ctx, "x", &[2], &[1., 2.]);
            bind_float(&mut ctx, "bias", &[2], &[f32::NAN, f32::INFINITY]);
            gemm.floats.insert("beta".into(), 0.0);
            crate::registry::translate(&mut ctx, &gemm).unwrap();
            assert_output(&mut ctx, "y", &[2], &[7., 10.]);

            bind_float(&mut ctx, "x", &[2, 3], &[1., 2., 3., 4., 5., 6.]);
            bind_float(
                &mut ctx,
                "w",
                &[2, 2, 3],
                &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            );
            bind_float(&mut ctx, "bias", &[2, 2], &[0.5, -0.5, 1., -1.]);
            bind_indices(&mut ctx, &[2, 2], &[1, 0, 0, 1], true);
            let mut grouped = node("GroupedMatMul", &["x", "w", "indices", "bias"]);
            grouped.ints.insert("transB".into(), -1);
            crate::registry::translate(&mut ctx, &grouped).unwrap();
            assert_output(
                &mut ctx,
                "y",
                &[2, 2, 2],
                &[51., 67., 14.5, 31.5, 32.5, 76.5, 123., 166.],
            );

            bind_float(&mut ctx, "x", &[2], &[10000., 10002.]);
            let mut mvn = node("MeanVarianceNormalization", &["x"]);
            mvn.int_arrays.insert("axes".into(), vec![0]);
            mvn.floats.insert("epsilon".into(), 0.5);
            crate::registry::translate(&mut ctx, &mvn).unwrap();
            assert_output(&mut ctx, "y", &[2], &[-2. / 3., 2. / 3.]);

            bind_float(&mut ctx, "x", &[1, 1, 3], &[1., 2., 3.]);
            bind_float(&mut ctx, "w", &[1, 1, 2], &[1., 10.]);
            bind_float(&mut ctx, "past", &[1, 1, 2], &[4., 5.]);
            let mut causal = node("CausalConvWithState", &["x", "w", "", "past"]);
            causal.inputs[2] = TensorRef::absent();
            causal.ints.insert("dilation".into(), 2);
            causal.outputs.push(output("state"));
            crate::registry::translate(&mut ctx, &causal).unwrap();
            assert_output(&mut ctx, "y", &[1, 1, 3], &[14., 25., 31.]);
            assert_output(&mut ctx, "state", &[1, 1, 2], &[2., 3.]);
        }
    }

    #[test]
    fn version29_gather_and_scatter_mean_live_indices() {
        for cpu in [false, true] {
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
            for int32 in [false, true] {
                bind_float(&mut ctx, "x", &[2, 2], &[1., 2., 3., 4.]);
                bind_indices(&mut ctx, &[2, 1], &[0, -1], int32);
                let gather = node("GatherND", &["x", "indices"]);
                crate::registry::translate(&mut ctx, &gather).unwrap();
                assert_output(&mut ctx, "y", &[2, 2], &[1., 2., 3., 4.]);
                bind_indices(&mut ctx, &[2, 1], &[0, 2], int32);
                assert!(
                    crate::registry::translate(&mut ctx, &gather)
                        .unwrap_err()
                        .contains("outside")
                );

                bind_indices(&mut ctx, &[2, 1], &[0, 0], int32);
                bind_float(&mut ctx, "updates", &[2, 2], &[3., 4., 5., 6.]);
                let mut scatter = node("ScatterND", &["x", "indices", "updates"]);
                scatter.strings.insert("reduction".into(), "mean".into());
                crate::registry::translate(&mut ctx, &scatter).unwrap();
                assert_output(&mut ctx, "y", &[2, 2], &[3., 4., 3., 4.]);
                bind_indices(&mut ctx, &[2, 0], &[], int32);
                bind_float(
                    &mut ctx,
                    "updates",
                    &[2, 2, 2],
                    &[3., 4., 5., 6., 5., 6., 7., 8.],
                );
                crate::registry::translate(&mut ctx, &scatter).unwrap();
                assert_output(&mut ctx, "y", &[2, 2], &[3., 4., 5., 6.]);

                bind_float(&mut ctx, "x", &[2, 3], &[1., 2., 3., 4., 5., 6.]);
                bind_indices(&mut ctx, &[1, 4], &[0, -1, 0, 1], int32);
                bind_float(&mut ctx, "updates", &[1, 4], &[3., 7., 5., 6.]);
                let mut elements = node("ScatterElements", &["x", "indices", "updates"]);
                elements.ints.insert("axis".into(), -1);
                elements.strings.insert("reduction".into(), "mean".into());
                crate::registry::translate(&mut ctx, &elements).unwrap();
                assert_output(&mut ctx, "y", &[2, 3], &[3., 4., 5., 4., 5., 6.]);
                bind_indices(&mut ctx, &[1, 4], &[0, 3, 0, 1], int32);
                assert!(
                    crate::registry::translate(&mut ctx, &elements)
                        .unwrap_err()
                        .contains("outside")
                );
            }
            bind_float(&mut ctx, "x", &[1], &[f32::MAX]);
            bind_float(&mut ctx, "updates", &[2], &[f32::MAX, f32::MAX]);
            bind_indices(&mut ctx, &[2], &[0, 0], true);
            let mut scatter = node("ScatterElements", &["x", "indices", "updates"]);
            scatter.strings.insert("reduction".into(), "mean".into());
            crate::registry::translate(&mut ctx, &scatter).unwrap();
            assert_output(&mut ctx, "y", &[1], &[f32::MAX]);
        }
    }

    #[test]
    fn version29_integer_scatter_mean_is_exact_and_floors_negative_results() {
        let stream = Stream::new_default_gpu();
        let mut plan = Plan::new(vec![]);
        let mut ctx = TranslationContext::new(
            &mut plan,
            std::ptr::null(),
            std::ptr::null_mut(),
            stream.as_raw(),
        );
        for signed in [false, true] {
            let (dtype, values, updates, expected) = if signed {
                (
                    mlx::mlx_dtype__MLX_INT64,
                    [i64::MAX as u64, (-2i64) as u64],
                    [i64::MAX as u64, (-3i64) as u64],
                    [i64::MAX as u64, (-3i64) as u64],
                )
            } else {
                (
                    mlx::mlx_dtype__MLX_UINT64,
                    [u64::MAX, 2],
                    [u64::MAX, 3],
                    [u64::MAX, 2],
                )
            };
            for (name, values) in [("x", values), ("updates", updates)] {
                let value = ctx.keep(Array::from_data(values.as_ptr().cast(), &[2], dtype));
                ctx.bind(&output(name), value);
            }
            bind_indices(&mut ctx, &[2], &[0, 1], true);
            let mut scatter = node("ScatterElements", &["x", "indices", "updates"]);
            scatter.strings.insert("reduction".into(), "mean".into());
            crate::registry::translate(&mut ctx, &scatter).unwrap();
            let y = ctx.resolve(&reference("y")).unwrap();
            assert_eq!(ctx.dtype_of(y), dtype);
            let y = ctx.contiguous_eval(y).unwrap();
            let actual = unsafe { std::slice::from_raw_parts(ctx.host_ptr(y).cast::<u64>(), 2) };
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn version29_double_mean_and_whole_tensor_scatter() {
        let stream = Stream::new_default_cpu();
        let mut plan = Plan::new(vec![]);
        let mut ctx = TranslationContext::new(
            &mut plan,
            std::ptr::null(),
            std::ptr::null_mut(),
            stream.as_raw(),
        );
        let large = 2f64.powi(40);
        for (name, values) in [("x", [large, -1.]), ("updates", [large + 0.125, -2.])] {
            let value = ctx.keep(Array::from_data(
                values.as_ptr().cast(),
                &[2],
                mlx::mlx_dtype__MLX_FLOAT64,
            ));
            ctx.bind(&output(name), value);
        }
        bind_indices(&mut ctx, &[2], &[0, 1], true);
        let mut scatter = node("ScatterElements", &["x", "indices", "updates"]);
        scatter.strings.insert("reduction".into(), "mean".into());
        crate::registry::translate(&mut ctx, &scatter).unwrap();
        let y = ctx.resolve(&reference("y")).unwrap();
        let y = ctx.contiguous_eval(y).unwrap();
        let actual = unsafe { std::slice::from_raw_parts(mlx::mlx_array_data_float64(y), 2) };
        assert_eq!(actual, [large + 0.0625, -1.5]);

        bind_float(&mut ctx, "x", &[2], &[1., 2.]);
        bind_float(&mut ctx, "updates", &[1, 2], &[3., 4.]);
        bind_indices(&mut ctx, &[1, 0], &[], true);
        let mut scatter = node("ScatterND", &["x", "indices", "updates"]);
        crate::registry::translate(&mut ctx, &scatter).unwrap();
        assert_output(&mut ctx, "y", &[2], &[3., 4.]);
        scatter.strings.insert("reduction".into(), "add".into());
        crate::registry::translate(&mut ctx, &scatter).unwrap();
        assert_output(&mut ctx, "y", &[2], &[4., 6.]);
    }

    #[test]
    fn logcumsumexp_low_precision_accumulates_wide_and_double_keeps_low_bits() {
        for (cpu, dtype) in [
            (false, mlx::mlx_dtype__MLX_FLOAT16),
            (false, mlx::mlx_dtype__MLX_BFLOAT16),
            (true, mlx::mlx_dtype__MLX_FLOAT64),
        ] {
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
            let mut scan = node("LogCumSumExp", &["x"]);
            scan.ints.insert("axis".into(), 0);
            if dtype == mlx::mlx_dtype__MLX_FLOAT64 {
                let values = [2f64.powi(40), 2f64.powi(40)];
                let x = ctx.keep(Array::from_data(values.as_ptr().cast(), &[2], dtype));
                ctx.bind(&output("x"), x);
                logcumsumexp_op(&mut ctx, &scan).unwrap();
                let y = ctx.resolve(&reference("y")).unwrap();
                let y = ctx.contiguous_eval(y).unwrap();
                let actual =
                    unsafe { std::slice::from_raw_parts(mlx::mlx_array_data_float64(y), 2) };
                assert_eq!(actual, [values[0], values[0] + std::f64::consts::LN_2]);
            } else {
                let x = ctx.zeros(&[2048], dtype).unwrap();
                ctx.bind(&output("x"), x);
                logcumsumexp_op(&mut ctx, &scan).unwrap();
                let y = ctx.resolve(&reference("y")).unwrap();
                assert_eq!(ctx.dtype_of(y), dtype);
                let y = ctx.astype(y, mlx::mlx_dtype__MLX_FLOAT32).unwrap();
                let y = ctx.contiguous_eval(y).unwrap();
                let actual =
                    unsafe { std::slice::from_raw_parts(mlx::mlx_array_data_float32(y), 2048) };
                let expected = 2048f32.ln();
                assert!((actual[2047] - expected).abs() < 0.02);
            }
        }
    }

    #[test]
    fn stable_unary_preserves_narrow_and_double_storage() {
        for (cpu, dtype) in [
            (false, mlx::mlx_dtype__MLX_FLOAT16),
            (false, mlx::mlx_dtype__MLX_BFLOAT16),
            (true, mlx::mlx_dtype__MLX_FLOAT16),
            (true, mlx::mlx_dtype__MLX_BFLOAT16),
            (true, mlx::mlx_dtype__MLX_FLOAT64),
        ] {
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
            let values = [-0.0f64, 1e-4, -1e-4, 1e-20];
            let x = if cpu {
                ctx.keep(Array::from_data(
                    values.as_ptr().cast(),
                    &[4],
                    mlx::mlx_dtype__MLX_FLOAT64,
                ))
            } else {
                let values = values.map(|v| v as f32);
                ctx.keep(Array::from_data(
                    values.as_ptr().cast(),
                    &[4],
                    mlx::mlx_dtype__MLX_FLOAT32,
                ))
            };
            let x = ctx.astype(x, dtype).unwrap();
            ctx.bind(&output("x"), x);
            for op in ["Expm1", "Log1p"] {
                crate::registry::translate(&mut ctx, &node(op, &["x"])).unwrap();
                let y = ctx.resolve(&reference("y")).unwrap();
                assert_eq!(ctx.dtype_of(y), dtype);
                let read_dtype = if cpu {
                    mlx::mlx_dtype__MLX_FLOAT64
                } else {
                    mlx::mlx_dtype__MLX_FLOAT32
                };
                let y = ctx.astype(y, read_dtype).unwrap();
                let y = ctx.contiguous_eval(y).unwrap();
                let actual: Vec<f64> = if cpu {
                    unsafe { std::slice::from_raw_parts(mlx::mlx_array_data_float64(y), 4) }
                        .to_vec()
                } else {
                    unsafe { std::slice::from_raw_parts(mlx::mlx_array_data_float32(y), 4) }
                        .iter()
                        .map(|&v| f64::from(v))
                        .collect()
                };
                assert_eq!(actual[0].to_bits(), (-0.0f64).to_bits());
                for (i, &raw) in values.iter().enumerate().skip(1) {
                    let convert = |v| {
                        if dtype == mlx::mlx_dtype__MLX_FLOAT16 {
                            half::f16::from_f64(v).to_f64()
                        } else if dtype == mlx::mlx_dtype__MLX_BFLOAT16 {
                            half::bf16::from_f64(v).to_f64()
                        } else {
                            v
                        }
                    };
                    let value = convert(raw);
                    let expected = convert(if op == "Expm1" {
                        value.exp_m1()
                    } else {
                        value.ln_1p()
                    });
                    assert!(
                        (actual[i] - expected).abs() <= expected.abs() * 1e-12,
                        "{op}, dtype={dtype}, cpu={cpu}: {} != {expected}",
                        actual[i]
                    );
                }
            }
        }
    }
}
