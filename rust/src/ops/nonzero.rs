//! Exact provider-owned host scanning for data-dependent ai.onnx NonZero.

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext, dim_i32, mlx_dtype_from_onnx};
use crate::registry::{ClaimResult, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry, is_movable};
use crate::sys::{mlx, ort};
use crate::{deny, require};

fn layout(shape: &[i64], dtype: mlx::mlx_dtype) -> Result<(usize, usize), MlxError> {
    let width = match dtype {
        mlx::mlx_dtype__MLX_BOOL | mlx::mlx_dtype__MLX_INT8 | mlx::mlx_dtype__MLX_UINT8 => 1,
        mlx::mlx_dtype__MLX_INT16
        | mlx::mlx_dtype__MLX_UINT16
        | mlx::mlx_dtype__MLX_FLOAT16
        | mlx::mlx_dtype__MLX_BFLOAT16 => 2,
        mlx::mlx_dtype__MLX_INT32 | mlx::mlx_dtype__MLX_UINT32 | mlx::mlx_dtype__MLX_FLOAT32 => 4,
        mlx::mlx_dtype__MLX_INT64 | mlx::mlx_dtype__MLX_UINT64 | mlx::mlx_dtype__MLX_FLOAT64 => 8,
        _ => return Err("MLX NonZero: expected a real numeric or bool tensor".into()),
    };
    for &extent in shape {
        if extent < 0 {
            return Err("MLX NonZero: negative runtime extent".into());
        }
        dim_i32(extent)?;
    }
    let count = if shape.contains(&0) {
        0
    } else {
        shape.iter().try_fold(1usize, |count, &extent| {
            count
                .checked_mul(extent as usize)
                .filter(|&count| count <= i32::MAX as usize)
                .ok_or_else(|| {
                    "MLX NonZero: input element count exceeds int32 indexing".to_string()
                })
        })?
    };
    let bytes = count
        .checked_mul(width)
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| "MLX NonZero: input byte count overflow".to_string())?;
    Ok((bytes, width))
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
    if ctx.shapeless_compile() || ctx.shape_keyed_compile() {
        return Err("MLX NonZero: data-dependent cardinality requires eager execution".into());
    }
    let input = &node.inputs[0];
    let (data, shape, dtype) =
        if input.source == Src::Intermediate || ctx.env_get(&input.name).is_some() {
            let array = ctx.resolve(input)?;
            let array = ctx.contiguous_eval(array)?;
            (
                ctx.host_ptr(array).cast::<std::ffi::c_void>(),
                ctx.shape_of(array).into_iter().map(i64::from).collect(),
                ctx.dtype_of(array),
            )
        } else {
            let host = ctx.raw_host(input)?;
            if !is_movable(host.dtype) {
                return Err("MLX NonZero: live input is not real numeric or bool".into());
            }
            (host.data, host.shape, mlx_dtype_from_onnx(host.dtype))
        };
    let (bytes, _) = layout(&shape, dtype)?;
    let payload = if bytes == 0 {
        &[]
    } else {
        if data.is_null() {
            return Err("MLX NonZero: nonempty input has null data".into());
        }
        unsafe { std::slice::from_raw_parts(data.cast::<u8>(), bytes) }
    };
    let (indices, shape) = coordinates(&shape, payload, dtype)?;
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
