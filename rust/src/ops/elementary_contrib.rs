//! Exact eager elementary com.microsoft v1 operators; see docs/ELEMENTARY_CONTRIB.md.

use half::{bf16, f16};

use super::host_scan::{layout, read_live};
use crate::engine::{MlxError, NodeDesc, TranslationContext, dim_i32};
use crate::registry::{ClaimResult, NodeView, OpRegistration, OpRegistry, SlotInfo};
use crate::sys::{mlx, ort};
use crate::{deny, require};

const EAGER_REASON: &str = "provider-owned live host arithmetic preserves explicit typed rounding";

pub(crate) fn is_elementary(op: &str) -> bool {
    matches!(op, "QuickGelu" | "BiasAdd" | "IsAllFinite" | "ScaledSiLU")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Float {
    F16,
    BF16,
    F32,
    F64,
}

impl Float {
    fn from_mlx(dtype: mlx::mlx_dtype) -> Result<Self, MlxError> {
        match dtype {
            mlx::mlx_dtype__MLX_FLOAT16 => Ok(Self::F16),
            mlx::mlx_dtype__MLX_BFLOAT16 => Ok(Self::BF16),
            mlx::mlx_dtype__MLX_FLOAT32 => Ok(Self::F32),
            mlx::mlx_dtype__MLX_FLOAT64 => Ok(Self::F64),
            _ => Err("MLX elementary contrib: expected floating-point input".into()),
        }
    }

    fn from_onnx(dtype: ort::ONNXTensorElementDataType) -> Option<Self> {
        match dtype {
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => Some(Self::F16),
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => {
                Some(Self::BF16)
            }
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => Some(Self::F32),
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => Some(Self::F64),
            _ => None,
        }
    }

    fn round(self, x: f64) -> f64 {
        match self {
            Self::F16 => f16::from_f32(x as f32).to_f64(),
            Self::BF16 => bf16::from_f32(x as f32).to_f64(),
            Self::F32 => (x as f32) as f64,
            Self::F64 => x,
        }
    }

    fn add(self, x: f64, y: f64) -> f64 {
        self.round(if self == Self::F64 {
            x + y
        } else {
            ((x as f32) + (y as f32)) as f64
        })
    }

    fn mul(self, x: f64, y: f64) -> f64 {
        self.round(if self == Self::F64 {
            x * y
        } else {
            ((x as f32) * (y as f32)) as f64
        })
    }

    fn sigmoid(self, x: f64) -> f64 {
        let value = if self == Self::F64 {
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                e / (1.0 + e)
            }
        } else {
            let x = x as f32;
            (if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                e / (1.0 + e)
            }) as f64
        };
        self.round(value)
    }

    fn decode(self, bytes: &[u8]) -> Vec<f64> {
        let width = match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        };
        bytes
            .chunks_exact(width)
            .map(|value| match self {
                Self::F16 => f16::from_bits(u16::from_ne_bytes(value.try_into().unwrap())).to_f64(),
                Self::BF16 => {
                    bf16::from_bits(u16::from_ne_bytes(value.try_into().unwrap())).to_f64()
                }
                Self::F32 => f32::from_ne_bytes(value.try_into().unwrap()) as f64,
                Self::F64 => f64::from_ne_bytes(value.try_into().unwrap()),
            })
            .collect()
    }

    fn encode(self, values: &[f64]) -> Vec<u8> {
        match self {
            Self::F16 => values
                .iter()
                .flat_map(|&x| f16::from_f32(x as f32).to_bits().to_ne_bytes())
                .collect(),
            Self::BF16 => values
                .iter()
                .flat_map(|&x| bf16::from_f32(x as f32).to_bits().to_ne_bytes())
                .collect(),
            Self::F32 => values
                .iter()
                .flat_map(|&x| (x as f32).to_ne_bytes())
                .collect(),
            Self::F64 => values.iter().flat_map(|&x| x.to_ne_bytes()).collect(),
        }
    }
}

fn compatible(a: &[i64], b: &[i64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&a, &b)| a < 0 || b < 0 || a == b)
}

fn geometry(op: &str, shapes: &[Vec<i64>]) -> Result<(), String> {
    let x = &shapes[0];
    match op {
        "BiasAdd" => {
            if x.len() != 3
                || shapes[1].len() != 1
                || !compatible(&x[2..], &shapes[1])
                || !compatible(x, &shapes[2])
            {
                return Err("BiasAdd requires X[N,S,C], bias[C], skip[N,S,C]".into());
            }
        }
        "ScaledSiLU" if shapes.len() == 2 && !shapes[1].is_empty() => {
            return Err("ScaledSiLU: scale must be a scalar (rank 0)".into());
        }
        _ => {}
    }
    Ok(())
}

fn claim(node: &NodeView) -> ClaimResult {
    let op = node.op_type();
    let count = node.num_inputs();
    let arity = match op.as_str() {
        "ScaledSiLU" => (1..=2).contains(&count),
        "BiasAdd" => count == 3,
        "QuickGelu" => count == 1,
        "IsAllFinite" => count >= 1,
        _ => false,
    };
    require!(arity, "{op}: invalid input arity");
    require!(
        node.num_outputs() == 1 && node.output_present(0),
        "{op}: requires exactly one present output"
    );
    let mut inputs: Vec<SlotInfo> = Vec::new();
    for i in 0..count {
        if i == 1 && op == "ScaledSiLU" && !node.input_present(i) {
            continue;
        }
        let Some(input) = node.input_info(i) else {
            deny!("{op}: input {i} must be a present floating tensor");
        };
        let Some(kind) = Float::from_onnx(input.dtype) else {
            deny!("{op}: input {i} must be a floating tensor");
        };
        require!(
            op != "BiasAdd" || matches!(kind, Float::F16 | Float::F32),
            "BiasAdd: only float16 and float32 are allowed"
        );
        require!(
            op != "ScaledSiLU" || kind != Float::F64,
            "ScaledSiLU: only float16, bfloat16 and float32 are allowed"
        );
        if let Some(first) = inputs.first() {
            require!(
                op == "ScaledSiLU" || input.dtype == first.dtype,
                "{op}: input types must be homogeneous"
            );
        }
        inputs.push(input);
    }
    let Some(output) = node.output_info(0) else {
        deny!("{op}: output tensor information is required");
    };
    if op == "IsAllFinite" {
        require!(
            output.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
                && output.shape.is_empty(),
            "IsAllFinite: output must be a scalar bool"
        );
        for name in ["isnan_only", "isinf_only"] {
            require!(
                !node.has_attr(name) || node.attr_type(name) == ort::OrtOpAttrType_ORT_OP_ATTR_INT,
                "IsAllFinite: {name} must be INT"
            );
        }
        require!(
            node.int_attr("isnan_only", 0) == 0 || node.int_attr("isinf_only", 0) == 0,
            "IsAllFinite: both flags cannot be nonzero"
        );
    } else {
        require!(
            output.dtype == inputs[0].dtype && compatible(&output.shape, &inputs[0].shape),
            "{op}: output must have X type and shape"
        );
        if matches!(op.as_str(), "QuickGelu" | "ScaledSiLU") {
            require!(
                !node.has_attr("alpha")
                    || node.attr_type("alpha") == ort::OrtOpAttrType_ORT_OP_ATTR_FLOAT,
                "{op}: alpha must be FLOAT"
            );
        }
    }
    geometry(
        &op,
        &inputs
            .iter()
            .map(|input| input.shape.clone())
            .collect::<Vec<_>>(),
    )
    .map_err(Into::into)
}

fn scaled_silu(kind: Float, x: f64, scale: f32) -> f64 {
    // Scale is decoded independently of T; never round scale to T before the product.
    let z = kind.round(((x as f32) * scale) as f64);
    let gate = kind.sigmoid(z);
    kind.mul(z, gate)
}

fn execute(ctx: &mut TranslationContext, node: &NodeDesc) -> Result<(), MlxError> {
    if ctx.shapeless_compile() || ctx.shape_keyed_compile() {
        return Err(format!("MLX {}: requires eager host arithmetic", node.op_type));
    }
    let mut values = Vec::new();
    let mut shapes = Vec::new();
    let mut kinds = Vec::new();
    let mut output_dtype = None;
    for (i, input) in node.inputs.iter().enumerate() {
        if i == 1 && node.op_type == "ScaledSiLU" && input.name.is_empty() {
            continue;
        }
        let host = read_live(ctx, input, &node.op_type)?;
        let kind = Float::from_mlx(host.dtype)?;
        output_dtype.get_or_insert(host.dtype);
        values.push(kind.decode(host.data));
        shapes.push(host.shape);
        kinds.push(kind);
    }
    if values.is_empty() {
        return Err(format!("MLX {}: missing input", node.op_type));
    }
    geometry(&node.op_type, &shapes)?;
    let kind = kinds[0];
    if node.op_type == "IsAllFinite" {
        let nan_only = node.ints.get("isnan_only").copied().unwrap_or(0) != 0;
        let inf_only = node.ints.get("isinf_only").copied().unwrap_or(0) != 0;
        if nan_only && inf_only {
            return Err("MLX IsAllFinite: both flags cannot be nonzero".into());
        }
        let finite = values.iter().flatten().all(|x| {
            if nan_only {
                !x.is_nan()
            } else if inf_only {
                !x.is_infinite()
            } else {
                x.is_finite()
            }
        });
        let output = ctx.scalar_bool(finite);
        ctx.bind(&node.outputs[0], output);
        return Ok(());
    }
    let alpha = node
        .floats
        .get("alpha")
        .copied()
        .unwrap_or(if node.op_type == "QuickGelu" {
            1.702
        } else {
            1.0
        });
    let scale = if node.op_type == "ScaledSiLU" && values.len() == 2 {
        values[1][0] as f32
    } else {
        alpha
    };
    let output: Vec<_> = values[0]
        .iter()
        .enumerate()
        .map(|(i, &x)| match node.op_type.as_str() {
            "QuickGelu" => {
                let z = kind.mul(kind.round(alpha as f64), x);
                kind.mul(x, kind.sigmoid(z))
            }
            "BiasAdd" => {
                let biased = kind.add(x, values[1][i % shapes[0][2] as usize]);
                kind.add(biased, values[2][i])
            }
            "ScaledSiLU" => scaled_silu(kind, x, scale),
            _ => unreachable!("registered elementary contrib operator"),
        })
        .collect();
    let bytes = kind.encode(&output);
    let shape = shapes[0]
        .iter()
        .map(|&d| dim_i32(d))
        .collect::<Result<Vec<_>, _>>()?;
    let dtype = output_dtype.unwrap();
    let (expected, _) = layout(&shapes[0], dtype, &node.op_type)?;
    if bytes.len() != expected {
        return Err(format!("MLX {}: output byte count mismatch", node.op_type));
    }
    let output = ctx.from_host(bytes.as_ptr().cast(), &shape, dtype);
    ctx.bind(&node.outputs[0], output);
    Ok(())
}

pub fn register(registry: &mut OpRegistry) {
    for op_type in ["QuickGelu", "BiasAdd", "IsAllFinite", "ScaledSiLU"] {
        registry.register_eager(
            OpRegistration {
                domain: "com.microsoft",
                op_type,
                min_opset: 1,
                max_opset: 1,
                handler: execute,
                claim,
            },
            EAGER_REASON,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_is_not_generic_bias_add_or_scale_broadcast() {
        assert!(geometry("BiasAdd", &[vec![2, 3, 4], vec![4], vec![2, 3, 4]]).is_ok());
        assert!(geometry("BiasAdd", &[vec![2, 3, 4], vec![1], vec![1, 3, 4]]).is_err());
        assert!(geometry("ScaledSiLU", &[vec![2, 3], vec![]]).is_ok());
        assert!(geometry("ScaledSiLU", &[vec![2, 3], vec![1]]).is_err());
    }

    #[test]
    fn scaled_silu_typed_stages_and_independent_scale() {
        // alpha=2 at x=1 gives SiLU(2), not Swish(1, alpha=2).
        assert_eq!(scaled_silu(Float::F32, 1.0, 2.0), 1.7615940570831299);
        assert_eq!(scaled_silu(Float::F16, 1.0, 2.0), 1.76171875);
        assert_eq!(scaled_silu(Float::BF16, 1.0, 2.0), 1.7578125);
        for kind in [Float::F16, Float::BF16, Float::F32] {
            assert_eq!(scaled_silu(kind, 3.0, 0.0), 0.0);
            assert!(scaled_silu(kind, f64::NAN, 1.0).is_nan());
        }
    }

    #[test]
    fn every_route_is_eager_including_nested_descriptors() {
        use crate::registry::CompileShapeSafety;
        for op in ["QuickGelu", "BiasAdd", "IsAllFinite", "ScaledSiLU"] {
            let node = NodeDesc::new(
                op.into(),
                "com.microsoft".into(),
                1,
                CompileShapeSafety::EagerOnly {
                    reason: EAGER_REASON,
                },
            );
            assert!(!crate::compiled::decode_enabled(false, &[node.clone()]));
            assert!(!crate::compiled::general_enabled(false, &[node.clone()]));
            let mut attention = node.clone();
            attention.op_type = "GroupQueryAttention".into();
            attention.compile_shape_safety = CompileShapeSafety::Shapeless;
            assert!(!crate::compiled::prefill_enabled(
                false,
                &[attention, node.clone()]
            ));
            let mut parent = node.clone();
            parent.compile_shape_safety = CompileShapeSafety::Shapeless;
            parent.subgraphs.push(crate::engine::SubgraphDesc {
                attr_name: "body".into(),
                input_names: vec![],
                output_names: vec![],
                nodes: vec![node],
            });
            assert!(!crate::compiled::decode_enabled(false, &[parent.clone()]));
            assert!(!crate::compiled::general_enabled(false, &[parent]));
        }
    }
}
