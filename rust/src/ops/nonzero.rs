//! Exact provider-owned host scanning for data-dependent ai.onnx NonZero.

use super::host_scan::{layout as host_layout, read_live};
use crate::engine::{MlxError, NodeDesc, TranslationContext, dim_i32};
use crate::registry::{ClaimResult, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry, is_movable};
use crate::sys::{mlx, ort};
use crate::{deny, require};

fn layout(shape: &[i64], dtype: mlx::mlx_dtype) -> Result<(usize, usize), MlxError> {
    host_layout(shape, dtype, "NonZero")
}

fn coordinates(
    shape: &[i64],
    data: &[u8],
    dtype: mlx::mlx_dtype,
) -> Result<(Vec<i64>, [i32; 2]), MlxError> {
    let (bytes, width) = layout(shape, dtype)?;
    let rank = dim_i32(shape.len() as i64)?;
    if data.len() != bytes {
        return Err("MLX NonZero: input byte count differs from runtime shape".into());
    }
    let mut selected = Vec::new();
    for (index, value) in data.chunks_exact(width).enumerate() {
        // Compare representations, not converted floats: preserve subnormals, NaNs, full-width
        // integers and both signed zeros, including fp16/bf16 without arithmetic or widening.
        let nonzero = match dtype {
            mlx::mlx_dtype__MLX_FLOAT16 | mlx::mlx_dtype__MLX_BFLOAT16 => {
                u16::from_ne_bytes(value.try_into().unwrap()) & 0x7fff != 0
            }
            mlx::mlx_dtype__MLX_FLOAT32 => {
                u32::from_ne_bytes(value.try_into().unwrap()) & 0x7fff_ffff != 0
            }
            mlx::mlx_dtype__MLX_FLOAT64 => {
                u64::from_ne_bytes(value.try_into().unwrap()) & 0x7fff_ffff_ffff_ffff != 0
            }
            _ => value.iter().any(|&byte| byte != 0),
        };
        if nonzero {
            selected
                .try_reserve(1)
                .map_err(|error| format!("MLX NonZero: index allocation failed: {error}"))?;
            selected.push(index);
        }
    }
    let columns = dim_i32(selected.len() as i64)?;
    let output_count = shape
        .len()
        .checked_mul(selected.len())
        .filter(|&count| count <= i32::MAX as usize)
        .ok_or_else(|| "MLX NonZero: output element count exceeds int32 indexing".to_string())?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_count)
        .map_err(|error| format!("MLX NonZero: output allocation failed: {error}"))?;
    output.resize(output_count, 0);
    for (column, &index) in selected.iter().enumerate() {
        let mut remainder = index;
        for axis in (0..shape.len()).rev() {
            let extent = shape[axis] as usize;
            output[axis * selected.len() + column] = (remainder % extent) as i64;
            remainder /= extent;
        }
    }
    Ok((output, [rank, columns]))
}

fn nonzero_op(ctx: &mut TranslationContext, node: &NodeDesc) -> Result<(), MlxError> {
    let host = read_live(ctx, &node.inputs[0], "NonZero")?;
    let (indices, shape) = coordinates(&host.shape, host.data, host.dtype)?;
    let output = ctx.from_host_i64(&indices, &shape);
    ctx.bind(&node.outputs[0], output);
    Ok(())
}

fn nonzero_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "NonZero expects one input and one output"
    );
    let (input, output) = match (node.input_info(0), node.output_info(0)) {
        (Some(input), Some(output)) => (input, output),
        _ => deny!("NonZero requires tensor input/output type information"),
    };
    require!(
        is_movable(input.dtype),
        "NonZero: complex/string forms remain unimplemented; expected real numeric or bool input"
    );
    require!(
        node.since_version() >= 13
            || input.dtype != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
        "NonZero: bfloat16 requires schema 13"
    );
    require!(
        output.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        "NonZero requires int64 output"
    );
    Ok(())
}

pub fn register(registry: &mut OpRegistry) {
    registry.register_eager(
        OpRegistration {
            domain: "",
            op_type: "NonZero",
            min_opset: 9,
            max_opset: K_ANY_OPSET,
            handler: nonzero_op,
            claim: nonzero_claim,
        },
        "host scan with data-dependent cardinality",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_bits_row_major_and_scalar_contract() {
        let dtype = mlx::mlx_dtype__MLX_FLOAT64;
        let values = [
            0u64,
            1 << 63,
            1,
            0x7ff8_0000_0000_0001,
            0x7ff0_0000_0000_0000,
            0,
        ];
        let data: Vec<_> = values.into_iter().flat_map(u64::to_ne_bytes).collect();
        assert_eq!(
            coordinates(&[2, 3], &data, dtype).unwrap(),
            (vec![0, 1, 1, 2, 0, 1], [2, 3])
        );
        assert_eq!(
            coordinates(&[], &0u64.to_ne_bytes(), dtype).unwrap(),
            (vec![], [0, 0])
        );
        assert_eq!(
            coordinates(&[], &1u64.to_ne_bytes(), dtype).unwrap(),
            (vec![], [0, 1])
        );
        for dtype in [mlx::mlx_dtype__MLX_FLOAT16, mlx::mlx_dtype__MLX_BFLOAT16] {
            let data: Vec<_> = [0u16, 0x8000, 1, 0x7fff]
                .into_iter()
                .flat_map(u16::to_ne_bytes)
                .collect();
            assert_eq!(
                coordinates(&[4], &data, dtype).unwrap(),
                (vec![2, 3], [1, 2])
            );
        }
        let dtype = mlx::mlx_dtype__MLX_UINT64;
        let data: Vec<_> = [0u64, 1 << 63, u64::MAX]
            .into_iter()
            .flat_map(u64::to_ne_bytes)
            .collect();
        assert_eq!(
            coordinates(&[3], &data, dtype).unwrap(),
            (vec![1, 2], [1, 2])
        );
    }

    #[test]
    fn empty_inputs_and_invalid_geometry() {
        let dtype = mlx::mlx_dtype__MLX_INT64;
        for shape in [vec![0], vec![2, 0, 3]] {
            assert_eq!(
                coordinates(&shape, &[], dtype).unwrap(),
                (vec![], [shape.len() as i32, 0])
            );
        }
        for shape in [
            vec![-1],
            vec![i64::MAX],
            vec![i32::MAX as i64, 2],
            vec![0, i64::MAX],
        ] {
            assert!(coordinates(&shape, &[], dtype).is_err());
        }
        assert!(coordinates(&[2], &0u64.to_ne_bytes(), dtype).is_err());
        assert!(coordinates(&[0], &[], mlx::mlx_dtype__MLX_COMPLEX64).is_err());
    }

    #[test]
    fn nonzero_excludes_every_compiled_route() {
        use crate::registry::CompileShapeSafety;
        let mut node = NodeDesc::new(
            "NonZero".into(),
            "".into(),
            13,
            CompileShapeSafety::EagerOnly {
                reason: "host scan with data-dependent cardinality",
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
        node.subgraphs.push(crate::engine::SubgraphDesc {
            attr_name: "body".into(),
            input_names: vec![],
            output_names: vec![],
            nodes: vec![node.clone()],
        });
        node.compile_shape_safety = CompileShapeSafety::Shapeless;
        assert!(!crate::compiled::decode_enabled(false, &[node.clone()]));
        assert!(!crate::compiled::general_enabled(false, &[node]));
    }
}
