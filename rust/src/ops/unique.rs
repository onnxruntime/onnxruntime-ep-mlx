//! Exact eager ai.onnx Unique: host sorting/grouping, never ORT CPU fallback.

use std::cmp::Ordering;

use super::controlflow::normalized_axis;
use super::host_scan::{layout, read_live};
use crate::engine::{MlxError, NodeDesc, TranslationContext, dim_i32};
use crate::registry::{ClaimResult, NodeView, OpRegistration, OpRegistry, is_movable};
use crate::sys::{mlx, ort};
use crate::{deny, require};

// Integers never pass through floating point. Floating comparison widens losslessly;
// output payloads are copied directly from their first occurrence, not reconstructed.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
enum Value {
    Signed(i64),
    Unsigned(u64),
    Float(f64),
}

impl Value {
    fn is_nan(self) -> bool {
        matches!(self, Self::Float(value) if value.is_nan())
    }

    fn compare(self, other: Self) -> Ordering {
        match (self.is_nan(), other.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => self.partial_cmp(&other).expect("same typed non-NaN values"),
        }
    }
}

fn values(data: &[u8], dtype: mlx::mlx_dtype, width: usize) -> Vec<Value> {
    data.chunks_exact(width)
        .map(|bytes| match dtype {
            mlx::mlx_dtype__MLX_BOOL => Value::Unsigned(u64::from(bytes[0] != 0)),
            mlx::mlx_dtype__MLX_INT8 => Value::Signed(i64::from(bytes[0] as i8)),
            mlx::mlx_dtype__MLX_INT16 => {
                Value::Signed(i64::from(i16::from_ne_bytes(bytes.try_into().unwrap())))
            }
            mlx::mlx_dtype__MLX_INT32 => {
                Value::Signed(i64::from(i32::from_ne_bytes(bytes.try_into().unwrap())))
            }
            mlx::mlx_dtype__MLX_INT64 => {
                Value::Signed(i64::from_ne_bytes(bytes.try_into().unwrap()))
            }
            mlx::mlx_dtype__MLX_UINT8 => Value::Unsigned(u64::from(bytes[0])),
            mlx::mlx_dtype__MLX_UINT16 => {
                Value::Unsigned(u64::from(u16::from_ne_bytes(bytes.try_into().unwrap())))
            }
            mlx::mlx_dtype__MLX_UINT32 => {
                Value::Unsigned(u64::from(u32::from_ne_bytes(bytes.try_into().unwrap())))
            }
            mlx::mlx_dtype__MLX_UINT64 => {
                Value::Unsigned(u64::from_ne_bytes(bytes.try_into().unwrap()))
            }
            mlx::mlx_dtype__MLX_FLOAT16 => Value::Float(
                half::f16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f64(),
            ),
            mlx::mlx_dtype__MLX_BFLOAT16 => Value::Float(
                half::bf16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f64(),
            ),
            mlx::mlx_dtype__MLX_FLOAT32 => {
                Value::Float(f64::from(f32::from_ne_bytes(bytes.try_into().unwrap())))
            }
            mlx::mlx_dtype__MLX_FLOAT64 => {
                Value::Float(f64::from_ne_bytes(bytes.try_into().unwrap()))
            }
            _ => unreachable!("layout validated dtype"),
        })
        .collect()
}

struct Unique {
    data: Vec<u8>,
    shape: Vec<i32>,
    first: Vec<i64>,
    inverse: Vec<i64>,
    counts: Vec<i64>,
}

fn unique(
    data: &[u8],
    shape: &[i64],
    dtype: mlx::mlx_dtype,
    axis: Option<i64>,
    sorted: i64,
) -> Result<Unique, MlxError> {
    if !matches!(sorted, 0 | 1) {
        return Err("MLX Unique: sorted must be 0 or 1".into());
    }
    let (bytes, width) = layout(shape, dtype, "Unique")?;
    if bytes != data.len() {
        return Err("MLX Unique: payload size differs from runtime shape".into());
    }
    let axis = axis
        .map(|axis| {
            normalized_axis(axis, shape.len()).ok_or_else(|| {
                format!(
                    "MLX Unique: axis {axis} out of range for rank {}",
                    shape.len()
                )
            })
        })
        .transpose()?;
    let elements = bytes / width;
    let (items, rows, columns) = match axis {
        None => (elements, 1, 1),
        Some(axis) => {
            let items = shape[axis] as usize;
            // Zero-sized slices compare equal, but a zero-length axis contains no slices.
            // Avoid products of enormous nonzero dimensions in a zero-element tensor.
            if elements == 0 {
                (items, 0, 0)
            } else {
                let columns = shape[axis + 1..].iter().map(|&dim| dim as usize).product();
                (items, elements / items / columns, columns)
            }
        }
    };
    dim_i32(items as i64)?;
    let values = values(data, dtype, width);
    let offset = |item: usize, element: usize| {
        (element / columns * items + item) * columns + element % columns
    };
    let compare = |a: usize, b: usize| {
        for element in 0..rows * columns {
            let order = values[offset(a, element)].compare(values[offset(b, element)]);
            if order != Ordering::Equal {
                return order;
            }
        }
        Ordering::Equal
    };
    let mut order: Vec<_> = (0..items).collect();
    order.sort_by(|&a, &b| compare(a, b));
    let mut first: Vec<usize> = Vec::new();
    let mut counts: Vec<i64> = Vec::new();
    let mut inverse = vec![0i64; items];
    for item in order {
        let same = first.last().is_some_and(|&previous| {
            compare(previous, item) == Ordering::Equal
                // ONNX Reference/NumPy groups flat NaNs, but never NaN-containing axis slices.
                && (axis.is_none() || (0..rows * columns)
                    .all(|element| !values[offset(item, element)].is_nan()))
        });
        if !same {
            first.push(item);
            counts.push(0);
        }
        let group = first.len() - 1;
        counts[group] += 1;
        inverse[item] = group as i64;
    }
    let mut groups: Vec<_> = (0..first.len()).collect();
    if sorted == 0 {
        groups.sort_by_key(|&group| first[group]);
    }
    let mut remap = vec![0i64; groups.len()];
    for (rank, &group) in groups.iter().enumerate() {
        remap[group] = rank as i64;
    }
    for group in &mut inverse {
        *group = remap[*group as usize];
    }
    let count = dim_i32(groups.len() as i64)?;
    let output_shape = match axis {
        None => vec![count],
        Some(axis) => {
            let mut output_shape = shape
                .iter()
                .map(|&dim| dim_i32(dim))
                .collect::<Result<Vec<_>, _>>()?;
            output_shape[axis] = count;
            output_shape
        }
    };
    let mut output = Vec::with_capacity(groups.len() * rows * columns * width);
    for row in 0..rows {
        for &group in &groups {
            let start = (row * items + first[group]) * columns * width;
            output.extend_from_slice(&data[start..start + columns * width]);
        }
    }
    Ok(Unique {
        data: output,
        shape: output_shape,
        first: groups.iter().map(|&group| first[group] as i64).collect(),
        inverse,
        counts: groups.iter().map(|&group| counts[group]).collect(),
    })
}

fn unique_op(ctx: &mut TranslationContext, node: &NodeDesc) -> Result<(), MlxError> {
    let host = read_live(ctx, &node.inputs[0], "Unique")?;
    let dtype = host.dtype;
    let result = unique(
        host.data,
        &host.shape,
        dtype,
        node.ints.get("axis").copied(),
        node.ints.get("sorted").copied().unwrap_or(1),
    )?;
    let output = ctx.from_host(result.data.as_ptr().cast(), &result.shape, dtype);
    ctx.bind(&node.outputs[0], output);
    for (slot, data) in [
        (1, &result.first),
        (2, &result.inverse),
        (3, &result.counts),
    ] {
        if let Some(output) = node
            .outputs
            .get(slot)
            .filter(|output| !output.name.is_empty())
        {
            let length = dim_i32(data.len() as i64)?;
            let array = ctx.from_host_i64(data, &[length]);
            ctx.bind(output, array);
        }
    }
    Ok(())
}

fn unique_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && (1..=4).contains(&node.num_outputs()),
        "Unique expects one input and one to four outputs"
    );
    let input = match node.input_info(0) {
        Some(input) => input,
        None => deny!("Unique requires tensor input type information"),
    };
    require!(
        is_movable(input.dtype),
        "Unique: complex/string/encoded payload transport is not implemented"
    );
    require!(
        input.dtype != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
        "Unique: bfloat16 requires schema 28, beyond the supported ORT schema set"
    );
    require!(
        matches!(node.int_attr("sorted", 1), 0 | 1),
        "Unique: sorted must be 0 or 1"
    );
    for slot in 0..node.num_outputs() {
        if let Some(output) = node.output_info(slot) {
            require!(
                output.dtype
                    == if slot == 0 {
                        input.dtype
                    } else {
                        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                    },
                "Unique: Y must match input dtype and metadata outputs must be int64"
            );
        }
    }
    Ok(())
}

pub fn register(registry: &mut OpRegistry) {
    registry.register_eager(
        OpRegistration {
            domain: "",
            op_type: "Unique",
            min_opset: 11,
            max_opset: 28,
            handler: unique_op,
            claim: unique_claim,
        },
        "exact host sort/group with data-dependent cardinality",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn i64_unique(values: &[i64], shape: &[i64], axis: Option<i64>, sorted: i64) -> Unique {
        let bytes: Vec<_> = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        unique(&bytes, shape, mlx::mlx_dtype__MLX_INT64, axis, sorted).unwrap()
    }

    fn i64_values(result: &Unique) -> Vec<i64> {
        result
            .data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|bytes| i64::from_ne_bytes(*bytes))
            .collect()
    }

    #[test]
    fn exact_full_width_integers_and_first_occurrences() {
        let input = [i64::MAX, 1, i64::MIN, i64::MAX, 9_007_199_254_740_993, 1];
        let result = i64_unique(&input, &[2, 3], None, 1);
        assert_eq!(
            i64_values(&result),
            [i64::MIN, 1, 9_007_199_254_740_993, i64::MAX]
        );
        assert_eq!(result.first, [2, 1, 4, 0]);
        assert_eq!(result.inverse, [3, 1, 0, 3, 2, 1]);
        assert_eq!(result.counts, [1, 2, 1, 2]);
        let result = i64_unique(&input, &[2, 3], None, 0);
        assert_eq!(
            i64_values(&result),
            [i64::MAX, 1, i64::MIN, 9_007_199_254_740_993]
        );
        assert_eq!(result.first, [0, 1, 2, 4]);
        assert_eq!(result.inverse, [0, 1, 2, 0, 3, 1]);
        let data: Vec<_> = [u64::MAX, 1 << 63, 1 << 63, (1 << 53) + 1]
            .into_iter()
            .flat_map(u64::to_ne_bytes)
            .collect();
        let result = unique(&data, &[4], mlx::mlx_dtype__MLX_UINT64, None, 1).unwrap();
        assert_eq!(result.first, [3, 1, 0]);
        assert_eq!(result.counts, [1, 2, 1]);
        assert_eq!(
            result.data,
            [
                ((1u64 << 53) + 1).to_ne_bytes(),
                (1u64 << 63).to_ne_bytes(),
                u64::MAX.to_ne_bytes()
            ]
            .concat()
        );
    }

    #[test]
    fn nonleading_axis_uses_whole_slices_and_preserves_layout() {
        let input = [3, 4, 1, 2, 3, 4, 5, 6, 2, 7, 5, 6];
        let result = i64_unique(&input, &[2, 3, 2], Some(-2), 1);
        assert_eq!(result.shape, [2, 2, 2]);
        assert_eq!(i64_values(&result), [1, 2, 3, 4, 2, 7, 5, 6]);
        assert_eq!(result.first, [1, 0]);
        assert_eq!(result.inverse, [1, 0, 1]);
        assert_eq!(result.counts, [1, 2]);
        let result = i64_unique(&input, &[2, 3, 2], Some(1), 0);
        assert_eq!(i64_values(&result), [3, 4, 1, 2, 5, 6, 2, 7]);
    }

    #[test]
    fn zeros_and_nans_follow_reference_equality_without_payload_conversion() {
        let bits = [
            0x8000_0000u32,
            0,
            0x7fc0_0011,
            0xffc0_0022,
            0x7f80_0000,
            0xff80_0000,
            1,
        ];
        let bytes: Vec<_> = bits.into_iter().flat_map(u32::to_ne_bytes).collect();
        let result = unique(&bytes, &[7], mlx::mlx_dtype__MLX_FLOAT32, None, 1).unwrap();
        assert_eq!(result.first, [5, 0, 6, 4, 2]);
        assert_eq!(result.counts, [1, 2, 1, 1, 2]);
        assert_eq!(result.inverse, [1, 1, 4, 4, 3, 0, 2]);
        assert_eq!(&result.data[4..8], &bits[0].to_ne_bytes());
        assert_eq!(&result.data[16..20], &bits[2].to_ne_bytes());
        let bytes = [0x7fc0_0011u32.to_ne_bytes(); 2].concat();
        let result = unique(&bytes, &[2, 1], mlx::mlx_dtype__MLX_FLOAT32, Some(0), 1).unwrap();
        assert_eq!(result.counts, [1, 1]);
        assert_eq!(result.first, [0, 1]);
    }

    #[test]
    fn low_precision_bool_and_double_comparisons() {
        for dtype in [mlx::mlx_dtype__MLX_FLOAT16, mlx::mlx_dtype__MLX_BFLOAT16] {
            let data = [
                0x8000u16.to_ne_bytes(),
                0u16.to_ne_bytes(),
                1u16.to_ne_bytes(),
            ]
            .concat();
            let result = unique(&data, &[3], dtype, None, 1).unwrap();
            assert_eq!(result.first, [0, 2]);
            assert_eq!(result.counts, [2, 1]);
            assert_eq!(&result.data[..2], &0x8000u16.to_ne_bytes());
        }
        let result = unique(&[1, 0, 1], &[3], mlx::mlx_dtype__MLX_BOOL, None, 1).unwrap();
        assert_eq!(result.data, [0, 1]);
        let data = [1.0f64, 1.0 + f64::EPSILON, 1.0]
            .into_iter()
            .flat_map(f64::to_ne_bytes)
            .collect::<Vec<_>>();
        let result = unique(&data, &[3], mlx::mlx_dtype__MLX_FLOAT64, None, 1).unwrap();
        assert_eq!(result.first, [0, 1]);
        assert_eq!(result.counts, [2, 1]);
    }

    #[test]
    fn float16_axis_nans_are_legal_and_sort_last_without_merging_slices() {
        let bits = [0x8000u16, 0, 1, 0x7c00, 0xfc00, 0x7e11, 0xfe22, 0x7e11];
        let bytes: Vec<_> = bits.into_iter().flat_map(u16::to_ne_bytes).collect();
        let result = unique(&bytes, &[8, 1], mlx::mlx_dtype__MLX_FLOAT16, Some(0), 1).unwrap();
        assert_eq!(result.first, [4, 0, 2, 3, 5, 6, 7]);
        assert_eq!(result.inverse, [1, 1, 2, 3, 0, 4, 5, 6]);
        assert_eq!(result.counts, [1, 2, 1, 1, 1, 1, 1]);
        let expected: Vec<_> = result
            .first
            .iter()
            .flat_map(|&index| bits[index as usize].to_ne_bytes())
            .collect();
        assert_eq!(result.data, expected);
        let result = unique(&bytes, &[8, 1], mlx::mlx_dtype__MLX_FLOAT16, Some(0), 0).unwrap();
        assert_eq!(result.first, [0, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn empty_slices_scalar_and_checked_geometry() {
        for (shape, axis, output, inverse, counts) in [
            (vec![0], None, vec![0], vec![], vec![]),
            (vec![0, 3], Some(0), vec![0, 3], vec![], vec![]),
            (vec![0, 3], Some(1), vec![0, 1], vec![0, 0, 0], vec![3]),
            (vec![2, 0, 3], Some(0), vec![1, 0, 3], vec![0, 0], vec![2]),
            (vec![2, 0, 3], Some(1), vec![2, 0, 3], vec![], vec![]),
            (
                vec![2, 0, 3],
                Some(2),
                vec![2, 0, 1],
                vec![0, 0, 0],
                vec![3],
            ),
        ] {
            let result = i64_unique(&[], &shape, axis, 1);
            assert_eq!(result.shape, output);
            assert_eq!(result.inverse, inverse);
            assert_eq!(result.counts, counts);
            assert!(result.data.is_empty());
        }
        let result = i64_unique(&[42], &[], None, 0);
        assert_eq!(result.shape, [1]);
        assert_eq!(result.first, [0]);
        assert_eq!(result.inverse, [0]);
        assert_eq!(result.counts, [1]);
        for axis in [i64::MIN, -3, 2, i64::MAX] {
            assert!(unique(&[], &[0, 3], mlx::mlx_dtype__MLX_INT64, Some(axis), 1).is_err());
        }
        assert!(
            unique(
                &42i64.to_ne_bytes(),
                &[],
                mlx::mlx_dtype__MLX_INT64,
                Some(0),
                1
            )
            .is_err()
        );
        for shape in [vec![-1], vec![0, i64::MAX], vec![i32::MAX as i64, 2]] {
            assert!(unique(&[], &shape, mlx::mlx_dtype__MLX_INT64, None, 1).is_err());
        }
        assert!(unique(&[], &[0], mlx::mlx_dtype__MLX_INT64, None, 2).is_err());
    }

    #[test]
    fn unique_never_enters_any_capture_route_even_nested() {
        use crate::registry::CompileShapeSafety;
        let node = NodeDesc::new(
            "Unique".into(),
            "".into(),
            11,
            CompileShapeSafety::EagerOnly {
                reason: "data-dependent cardinality",
            },
        );
        assert!(!crate::compiled::decode_enabled(
            false,
            std::slice::from_ref(&node)
        ));
        assert!(!crate::compiled::general_enabled(
            false,
            std::slice::from_ref(&node)
        ));
        let mut attention = node.clone();
        attention.op_type = "GroupQueryAttention".into();
        attention.compile_shape_safety = CompileShapeSafety::Shapeless;
        assert!(!crate::compiled::prefill_enabled(
            false,
            &[attention, node.clone()]
        ));
        let mut parent = NodeDesc::new("If".into(), "".into(), 13, CompileShapeSafety::Shapeless);
        parent.subgraphs.push(crate::engine::SubgraphDesc {
            attr_name: "then_branch".into(),
            input_names: vec![],
            output_names: vec![],
            nodes: vec![node],
        });
        assert!(!crate::compiled::decode_enabled(false, &[parent.clone()]));
        assert!(!crate::compiled::general_enabled(false, &[parent]));
    }
}
