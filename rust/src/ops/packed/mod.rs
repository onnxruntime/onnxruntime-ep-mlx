//! ONNX opset-29 Pack, Unpack, and DequantizePacked.
//!
//! Vectorized MLX and tiled Metal paths preserve the normative little-endian block layouts.
//! Exact host paths retain the full signed-INT64 grouped-radix contract and all 27 fixed
//! GGUF profiles, including the pinned normative IQ codebooks.

use std::ffi::c_void;

use half::{bf16, f16};

use super::host_scan::read_live;
use crate::engine::{MlxError, NodeDesc, TranslationContext, dim_i32, mlx_dtype_from_onnx};
use crate::registry::{ClaimResult, CompileShapeSafety, NodeView, OpRegistration, OpRegistry};
use crate::sys::{mlx, ort};
use crate::{deny, require};

mod decode;
mod fusion;
mod gpu;
mod iq_tables;
mod qmm;

pub(super) use fusion::execute;
pub(crate) use fusion::{FusedWeight, partition_regions, recognize};

const UINT8: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
const UINT32: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
const INT64: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
const FLOAT: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
const FLOAT16: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
const BFLOAT16: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
const DOUBLE: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackSpec {
    radix: u64,
    group_size: usize,
    group_width: usize,
}

fn pack_spec(
    bits: Option<i64>,
    radix: Option<i64>,
    group_size: Option<i64>,
    op: &str,
) -> Result<PackSpec, MlxError> {
    match (bits, radix, group_size) {
        (Some(bits), None, None) if (1..=8).contains(&bits) => Ok(PackSpec {
            radix: 1u64 << bits,
            group_size: 1,
            group_width: bits as usize,
        }),
        (Some(bits), None, None) => Err(format!("MLX {op}: bits must be in [1, 8], got {bits}")),
        (None, Some(radix), Some(group_size)) if (2..=256).contains(&radix) && group_size > 0 => {
            let radix = radix as u64;
            let group_size = usize::try_from(group_size)
                .map_err(|_| format!("MLX {op}: group_size does not fit usize"))?;
            let mut range = 1u64;
            for _ in 0..group_size {
                range = range.checked_mul(radix).ok_or_else(|| {
                    format!("MLX {op}: radix^{group_size} exceeds the positive INT64 contract")
                })?;
                if range > i64::MAX as u64 {
                    return Err(format!(
                        "MLX {op}: radix^{group_size} exceeds the positive INT64 contract"
                    ));
                }
            }
            let group_width = (u64::BITS - (range - 1).leading_zeros()) as usize;
            Ok(PackSpec {
                radix,
                group_size,
                group_width,
            })
        }
        (None, Some(radix), Some(group_size)) => Err(format!(
            "MLX {op}: radix must be in [2, 256], group_size must be positive, and radix^group_size must fit positive INT64 (got radix={radix}, group_size={group_size})"
        )),
        (None, None, None) => Err(format!(
            "MLX {op}: specify bits or both radix and group_size"
        )),
        _ => Err(format!(
            "MLX {op}: bits is mutually exclusive with radix/group_size, and radix requires group_size"
        )),
    }
}

fn normalize_axis(axis: i64, rank: usize, op: &str) -> Result<usize, MlxError> {
    if rank == 0 {
        return Err(format!("MLX {op}: input rank must be at least one"));
    }
    let axis = if axis < 0 { axis + rank as i64 } else { axis };
    if !(0..rank as i64).contains(&axis) {
        return Err(format!(
            "MLX {op}: axis {axis} is outside [-{rank}, {}]",
            rank - 1
        ));
    }
    Ok(axis as usize)
}

fn element_count(shape: &[i64], op: &str) -> Result<usize, MlxError> {
    if shape.iter().any(|&dim| dim < 0) {
        return Err(format!(
            "MLX {op}: runtime shape contains a negative dimension"
        ));
    }
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim as usize).ok_or_else(|| {
            format!("MLX {op}: tensor element count overflows; reduce the logical shape")
        })
    })
}

fn checked_count(shape: &[i64], op: &str) -> Result<usize, MlxError> {
    let count = element_count(shape, op)?;
    if count > i32::MAX as usize {
        return Err(format!(
            "MLX {op}: tensor element count exceeds int32 indexing"
        ));
    }
    Ok(count)
}

fn packed_axis_count(count: usize, spec: PackSpec, word_bits: usize) -> Result<usize, MlxError> {
    let groups = count
        .checked_add(spec.group_size - 1)
        .ok_or_else(|| "MLX Pack: group count overflow".to_string())?
        / spec.group_size;
    groups
        .checked_mul(spec.group_width)
        .and_then(|bits| bits.checked_add(word_bits - 1))
        .map(|bits| bits / word_bits)
        .ok_or_else(|| "MLX Pack: packed axis size overflow".to_string())
}

fn write_bits(
    words: &mut [u64],
    word_bits: usize,
    mut offset: usize,
    mut width: usize,
    mut value: u64,
) {
    while width != 0 {
        let word = offset / word_bits;
        let shift = offset % word_bits;
        let take = width.min(word_bits - shift);
        let mask = if take == 64 {
            u64::MAX
        } else {
            (1u64 << take) - 1
        };
        words[word] |= (value & mask) << shift;
        value >>= take;
        offset += take;
        width -= take;
    }
}

fn read_bits(words: &[u64], word_bits: usize, mut offset: usize, mut width: usize) -> u64 {
    let mut value = 0u64;
    let mut written = 0usize;
    while width != 0 {
        let word = offset / word_bits;
        let shift = offset % word_bits;
        let take = width.min(word_bits - shift);
        let mask = (1u64 << take) - 1;
        value |= ((words[word] >> shift) & mask) << written;
        offset += take;
        written += take;
        width -= take;
    }
    value
}

fn pack_codes(
    input: &[u8],
    shape: &[i64],
    axis: i64,
    spec: PackSpec,
    word_bits: usize,
) -> Result<(Vec<u64>, Vec<i64>), MlxError> {
    let count = checked_count(shape, "Pack")?;
    if input.len() != count {
        return Err(format!(
            "MLX Pack: input has {} bytes but runtime shape requires {count}",
            input.len()
        ));
    }
    if word_bits != 8 && word_bits != 32 {
        return Err(format!(
            "MLX Pack: unsupported storage word width {word_bits}"
        ));
    }
    let axis = normalize_axis(axis, shape.len(), "Pack")?;
    let logical = shape[axis] as usize;
    let packed = packed_axis_count(logical, spec, word_bits)?;
    let mut output_shape = shape.to_vec();
    output_shape[axis] = packed as i64;
    let output_count = checked_count(&output_shape, "Pack")?;
    if output_count == 0 {
        return Ok((Vec::new(), output_shape));
    }
    let outer = checked_count(&shape[..axis], "Pack")?;
    let inner = checked_count(&shape[axis + 1..], "Pack")?;
    let mut output = vec![0u64; output_count];
    let groups = logical.div_ceil(spec.group_size);

    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let row_base = (outer_index * packed) * inner + inner_index;
            let mut row = vec![0u64; packed];
            for group in 0..groups {
                let mut value = 0u64;
                let mut factor = 1u64;
                for lane in 0..spec.group_size {
                    let axis_index = group * spec.group_size + lane;
                    let digit = if axis_index < logical {
                        input[(outer_index * logical + axis_index) * inner + inner_index] as u64
                    } else {
                        0
                    };
                    if digit >= spec.radix {
                        return Err(format!(
                            "MLX Pack: code {digit} at axis index {axis_index} is not below radix {}",
                            spec.radix
                        ));
                    }
                    value = value
                        .checked_add(digit.checked_mul(factor).ok_or_else(|| {
                            "MLX Pack: grouped-radix multiplication overflow".to_string()
                        })?)
                        .ok_or_else(|| "MLX Pack: grouped-radix addition overflow".to_string())?;
                    factor = factor
                        .checked_mul(spec.radix)
                        .ok_or_else(|| "MLX Pack: grouped-radix factor overflow".to_string())?;
                }
                write_bits(
                    &mut row,
                    word_bits,
                    group * spec.group_width,
                    spec.group_width,
                    value,
                );
            }
            for (word, value) in row.into_iter().enumerate() {
                output[row_base + word * inner] = value;
            }
        }
    }
    Ok((output, output_shape))
}

fn unpack_codes(
    input: &[u64],
    shape: &[i64],
    axis: i64,
    count: i64,
    spec: PackSpec,
    word_bits: usize,
) -> Result<(Vec<u8>, Vec<i64>), MlxError> {
    if count < 0 {
        return Err(format!(
            "MLX Unpack: count must be nonnegative, got {count}"
        ));
    }
    let input_count = checked_count(shape, "Unpack")?;
    if input.len() != input_count {
        return Err(format!(
            "MLX Unpack: input has {} words but runtime shape requires {input_count}",
            input.len()
        ));
    }
    if word_bits != 8 && word_bits != 32 {
        return Err(format!(
            "MLX Unpack: unsupported storage word width {word_bits}"
        ));
    }
    let axis = normalize_axis(axis, shape.len(), "Unpack")?;
    let logical =
        usize::try_from(count).map_err(|_| "MLX Unpack: count does not fit usize".to_string())?;
    let expected_packed = packed_axis_count(logical, spec, word_bits)?;
    if shape[axis] as usize != expected_packed {
        return Err(format!(
            "MLX Unpack: packed axis has {} words; count {count} requires {expected_packed}",
            shape[axis]
        ));
    }
    let mut output_shape = shape.to_vec();
    output_shape[axis] = count;
    let output_count = checked_count(&output_shape, "Unpack")?;
    if output_count == 0 {
        return Ok((Vec::new(), output_shape));
    }
    let outer = checked_count(&shape[..axis], "Unpack")?;
    let inner = checked_count(&shape[axis + 1..], "Unpack")?;
    let mut output = vec![0u8; output_count];
    let groups = logical.div_ceil(spec.group_size);
    let range = (0..spec.group_size).try_fold(1u64, |value, _| {
        value
            .checked_mul(spec.radix)
            .ok_or_else(|| "MLX Unpack: grouped-radix range overflow".to_string())
    })?;

    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let row: Vec<_> = (0..expected_packed)
                .map(|word| input[(outer_index * expected_packed + word) * inner + inner_index])
                .collect();
            for group in 0..groups {
                let mut value =
                    read_bits(&row, word_bits, group * spec.group_width, spec.group_width);
                if value >= range {
                    return Err(format!(
                        "MLX Unpack: group index {value} is outside radix^group_size range {range}"
                    ));
                }
                for lane in 0..spec.group_size {
                    let axis_index = group * spec.group_size + lane;
                    if axis_index >= logical {
                        break;
                    }
                    output[(outer_index * logical + axis_index) * inner + inner_index] =
                        (value % spec.radix) as u8;
                    value /= spec.radix;
                }
            }
        }
    }
    Ok((output, output_shape))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PackedFormat {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    Iq1M,
    Tq1_0,
    Tq2_0,
    Mxfp4,
    Nvfp4,
    Q1_0,
    Q2_0,
}

impl PackedFormat {
    fn parse(name: &str) -> Result<Self, MlxError> {
        match name {
            "gguf:q4_0" => Ok(Self::Q4_0),
            "gguf:q4_1" => Ok(Self::Q4_1),
            "gguf:q5_0" => Ok(Self::Q5_0),
            "gguf:q5_1" => Ok(Self::Q5_1),
            "gguf:q8_0" => Ok(Self::Q8_0),
            "gguf:q8_1" => Ok(Self::Q8_1),
            "gguf:q2_k" => Ok(Self::Q2K),
            "gguf:q3_k" => Ok(Self::Q3K),
            "gguf:q4_k" => Ok(Self::Q4K),
            "gguf:q5_k" => Ok(Self::Q5K),
            "gguf:q6_k" => Ok(Self::Q6K),
            "gguf:q8_k" => Ok(Self::Q8K),
            "gguf:iq2_xxs" => Ok(Self::Iq2Xxs),
            "gguf:iq2_xs" => Ok(Self::Iq2Xs),
            "gguf:iq3_xxs" => Ok(Self::Iq3Xxs),
            "gguf:iq1_s" => Ok(Self::Iq1S),
            "gguf:iq4_nl" => Ok(Self::Iq4Nl),
            "gguf:iq3_s" => Ok(Self::Iq3S),
            "gguf:iq2_s" => Ok(Self::Iq2S),
            "gguf:iq4_xs" => Ok(Self::Iq4Xs),
            "gguf:iq1_m" => Ok(Self::Iq1M),
            "gguf:tq1_0" => Ok(Self::Tq1_0),
            "gguf:tq2_0" => Ok(Self::Tq2_0),
            "gguf:mxfp4" => Ok(Self::Mxfp4),
            "gguf:nvfp4" => Ok(Self::Nvfp4),
            "gguf:q1_0" => Ok(Self::Q1_0),
            "gguf:q2_0" => Ok(Self::Q2_0),
            _ => Err(format!(
                "MLX DequantizePacked: unsupported format identifier {name:?}"
            )),
        }
    }

    fn geometry(self) -> (usize, usize) {
        match self {
            Self::Q4_0 | Self::Iq4Nl => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q8K => (256, 292),
            Self::Iq2Xxs => (256, 66),
            Self::Iq2Xs => (256, 74),
            Self::Iq3Xxs => (256, 98),
            Self::Iq1S => (256, 50),
            Self::Iq3S => (256, 110),
            Self::Iq2S => (256, 82),
            Self::Iq4Xs => (256, 136),
            Self::Iq1M => (256, 56),
            Self::Tq1_0 => (256, 54),
            Self::Tq2_0 => (256, 66),
            Self::Mxfp4 => (32, 17),
            Self::Nvfp4 => (64, 36),
            Self::Q1_0 => (128, 18),
            Self::Q2_0 => (64, 18),
        }
    }
}

fn decode_block(format: PackedFormat, block: &[u8]) -> Vec<f32> {
    decode::block(format, block)
}

fn decode_packed(input: &[u8], format: PackedFormat, shape: &[i64]) -> Result<Vec<f32>, MlxError> {
    if shape.is_empty() || shape.iter().any(|&dim| dim < 0) {
        return Err(
            "MLX DequantizePacked: shape must have rank at least one and nonnegative dimensions"
                .into(),
        );
    }
    let (block_elements, block_bytes) = format.geometry();
    if shape.last().copied().unwrap() % block_elements as i64 != 0 {
        return Err(format!(
            "MLX DequantizePacked: last dimension {} is not a multiple of block size {block_elements}",
            shape.last().unwrap()
        ));
    }
    let count = element_count(shape, "DequantizePacked")?;
    let expected = count
        .checked_div(block_elements)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(|| "MLX DequantizePacked: packed byte count overflow".to_string())?;
    if input.len() != expected {
        return Err(format!(
            "MLX DequantizePacked: input has {} bytes; format and shape require {expected}",
            input.len()
        ));
    }
    let mut output = Vec::with_capacity(count);
    for block in input.chunks_exact(block_bytes) {
        output.extend(decode_block(format, block));
    }
    Ok(output)
}

fn output_bytes(
    values: &[f32],
    dtype: ort::ONNXTensorElementDataType,
) -> Result<(Vec<u8>, mlx::mlx_dtype), MlxError> {
    let (bytes, dtype) = match dtype {
        FLOAT => (
            values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect(),
            mlx::mlx_dtype__MLX_FLOAT32,
        ),
        FLOAT16 => (
            values
                .iter()
                .flat_map(|&value| f16::from_f32(value).to_bits().to_ne_bytes())
                .collect(),
            mlx::mlx_dtype__MLX_FLOAT16,
        ),
        BFLOAT16 => (
            values
                .iter()
                .flat_map(|&value| bf16::from_f32(value).to_bits().to_ne_bytes())
                .collect(),
            mlx::mlx_dtype__MLX_BFLOAT16,
        ),
        DOUBLE => (
            values
                .iter()
                .flat_map(|&value| (value as f64).to_ne_bytes())
                .collect(),
            mlx::mlx_dtype__MLX_FLOAT64,
        ),
        _ => {
            return Err(
                "MLX DequantizePacked: output must be float, float16, bfloat16, or double".into(),
            );
        }
    };
    Ok((bytes, dtype))
}

struct AxisLast {
    value: mlx::mlx_array,
    shape: Vec<i32>,
    inverse: Vec<i32>,
}

fn move_axis_last(
    ctx: &mut TranslationContext,
    value: mlx::mlx_array,
    axis: usize,
) -> Result<AxisLast, MlxError> {
    let shape = ctx.shape_of(value);
    let rank = shape.len();
    let mut permutation: Vec<i32> = (0..rank)
        .filter(|&index| index != axis)
        .map(|index| index as i32)
        .collect();
    permutation.push(axis as i32);
    let mut moved_shape: Vec<i32> = permutation
        .iter()
        .map(|&index| shape[index as usize])
        .collect();
    let mut inverse = vec![0i32; rank];
    for (index, &original) in permutation.iter().enumerate() {
        inverse[original as usize] = index as i32;
    }
    let value = if axis + 1 == rank {
        value
    } else {
        ctx.transpose(value, &permutation)?
    };
    if moved_shape.is_empty() {
        moved_shape.push(1);
    }
    Ok(AxisLast {
        value,
        shape: moved_shape,
        inverse,
    })
}

fn restore_axis(
    ctx: &mut TranslationContext,
    value: mlx::mlx_array,
    moved_shape: &[i32],
    inverse: &[i32],
) -> Result<mlx::mlx_array, MlxError> {
    let value = ctx.reshape(value, moved_shape)?;
    if inverse
        .iter()
        .enumerate()
        .all(|(index, &axis)| index as i32 == axis)
    {
        Ok(value)
    } else {
        ctx.transpose(value, inverse)
    }
}

fn matrix_rows(shape: &[i32], op: &str) -> Result<i32, MlxError> {
    shape[..shape.len() - 1]
        .iter()
        .try_fold(1i32, |rows, &extent| {
            rows.checked_mul(extent)
                .ok_or_else(|| format!("MLX {op}: row count exceeds int32 indexing"))
        })
}

fn scalar_u32(ctx: &mut TranslationContext, value: u32) -> mlx::mlx_array {
    ctx.scalar_from_bytes(&value.to_ne_bytes(), mlx::mlx_dtype__MLX_UINT32)
}

fn take_matrix_column(
    ctx: &mut TranslationContext,
    matrix: mlx::mlx_array,
    index: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let index = ctx.scalar_i32(index);
    ctx.emit(|out, stream| unsafe { mlx::mlx_take_axis(out, matrix, index, 1, stream) })
}

fn shifted_piece(
    ctx: &mut TranslationContext,
    value: mlx::mlx_array,
    right: usize,
    width: usize,
    left: usize,
) -> Result<mlx::mlx_array, MlxError> {
    let mut value = value;
    if right != 0 {
        let shift = scalar_u32(ctx, right as u32);
        value = ctx.binary(mlx::mlx_right_shift, value, shift)?;
    }
    let mask = scalar_u32(ctx, (1u32 << width) - 1);
    value = ctx.binary(mlx::mlx_bitwise_and, value, mask)?;
    if left != 0 {
        let shift = scalar_u32(ctx, left as u32);
        value = ctx.binary(mlx::mlx_left_shift, value, shift)?;
    }
    Ok(value)
}

fn pack_bits_mlx(
    ctx: &mut TranslationContext,
    node: &NodeDesc,
    bits: usize,
    axis: i64,
    word_bits: usize,
) -> Result<mlx::mlx_array, MlxError> {
    let input = ctx.resolve(&node.inputs[0])?;
    if ctx.dtype_of(input) != mlx::mlx_dtype__MLX_UINT8 {
        return Err("MLX Pack: input must be UINT8".into());
    }
    let rank = ctx.ndim(input);
    let axis = normalize_axis(axis, rank, "Pack")?;
    let input_shape = ctx.shape_of(input);
    let logical = input_shape[axis] as usize;
    let packed = logical
        .checked_mul(bits)
        .and_then(|value| value.checked_add(word_bits - 1))
        .map(|value| value / word_bits)
        .ok_or_else(|| "MLX Pack: packed axis size overflow".to_string())?;
    let packed_dim = i32::try_from(packed)
        .map_err(|_| "MLX Pack: packed axis exceeds int32; reduce the input axis length")?;
    if input_shape.contains(&0) {
        let mut output_shape = input_shape;
        output_shape[axis] = packed_dim;
        let dtype = if word_bits == 8 {
            mlx::mlx_dtype__MLX_UINT8
        } else {
            mlx::mlx_dtype__MLX_UINT32
        };
        ctx.mark_fast("mlx_vectorized_pack_bits");
        return ctx.zeros(&output_shape, dtype);
    }
    if bits < 8 {
        if ctx.shapeless_compile() || ctx.shape_keyed_compile() {
            return Err(
                "MLX Pack: live code validation requires eager execution; disable compiled capture for this node"
                    .into(),
            );
        }
        let maximum = ctx
            .emit(|out, stream| unsafe { mlx::mlx_max(out, input, false, stream) })
            .and_then(|value| ctx.read_scalar_i64(value))
            .map_err(|error| {
                format!(
                    "MLX Pack: unable to validate input codes for bits={bits}: {error}; check MLX backend logs and retry"
                )
            })?;
        let limit = (1i64 << bits) - 1;
        if maximum > limit {
            return Err(format!(
                "MLX Pack: maximum input code {maximum} exceeds {limit} for bits={bits}; provide UINT8 codes in [0, {limit}]"
            ));
        }
    }
    let moved = move_axis_last(ctx, input, axis)?;
    let rows = matrix_rows(&moved.shape, "Pack")?;
    let matrix = ctx.reshape(moved.value, &[rows, logical as i32])?;
    let matrix = ctx.astype(matrix, mlx::mlx_dtype__MLX_UINT32)?;
    let mut columns = Vec::with_capacity(packed);
    for word in 0..packed {
        let word_start = word * word_bits;
        let word_end = word_start + word_bits;
        let first_code = word_start / bits;
        let last_code = logical.min(word_end.div_ceil(bits));
        let mut output = ctx.zeros(&[rows], mlx::mlx_dtype__MLX_UINT32)?;
        for code in first_code..last_code {
            let code_start = code * bits;
            let start = word_start.max(code_start);
            let end = word_end.min(code_start + bits);
            let value = take_matrix_column(ctx, matrix, code as i32)?;
            let value = shifted_piece(
                ctx,
                value,
                start - code_start,
                end - start,
                start - word_start,
            )?;
            output = ctx.binary(mlx::mlx_bitwise_or, output, value)?;
        }
        columns.push(output);
    }
    let packed_matrix = if columns.is_empty() {
        ctx.zeros(&[rows, 0], mlx::mlx_dtype__MLX_UINT32)?
    } else {
        ctx.stack(&columns, 1)?
    };
    let packed_matrix = if word_bits == 8 {
        ctx.astype(packed_matrix, mlx::mlx_dtype__MLX_UINT8)?
    } else {
        packed_matrix
    };
    let mut output_shape = moved.shape;
    *output_shape.last_mut().unwrap() = packed_dim;
    ctx.mark_fast("mlx_vectorized_pack_bits");
    restore_axis(ctx, packed_matrix, &output_shape, &moved.inverse)
}

fn unpack_bits_mlx(
    ctx: &mut TranslationContext,
    node: &NodeDesc,
    bits: usize,
    axis: i64,
    count: usize,
) -> Result<mlx::mlx_array, MlxError> {
    let input = ctx.resolve(&node.inputs[0])?;
    if !matches!(
        ctx.dtype_of(input),
        mlx::mlx_dtype__MLX_UINT8 | mlx::mlx_dtype__MLX_UINT32
    ) {
        return Err("MLX Unpack: input must be UINT8 or UINT32".into());
    }
    let rank = ctx.ndim(input);
    let axis = normalize_axis(axis, rank, "Unpack")?;
    let word_bits = if ctx.dtype_of(input) == mlx::mlx_dtype__MLX_UINT8 {
        8
    } else {
        32
    };
    let input_shape = ctx.shape_of(input);
    let packed = input_shape[axis] as usize;
    let expected = count
        .checked_mul(bits)
        .and_then(|value| value.checked_add(word_bits - 1))
        .map(|value| value / word_bits)
        .ok_or_else(|| "MLX Unpack: packed axis size overflow".to_string())?;
    if packed != expected {
        return Err(format!(
            "MLX Unpack: packed axis has {packed} words; count {count} requires {expected}"
        ));
    }
    let count_dim = i32::try_from(count).map_err(|_| {
        format!("MLX Unpack: count {count} exceeds the int32 dimension limit; reduce count")
    })?;
    if input_shape.contains(&0) {
        let mut output_shape = input_shape;
        output_shape[axis] = count_dim;
        ctx.mark_fast("mlx_vectorized_unpack_bits");
        return ctx.zeros(&output_shape, mlx::mlx_dtype__MLX_UINT8);
    }
    let moved = move_axis_last(ctx, input, axis)?;
    let rows = matrix_rows(&moved.shape, "Unpack")?;
    let matrix = ctx.reshape(moved.value, &[rows, packed as i32])?;
    let matrix = ctx.astype(matrix, mlx::mlx_dtype__MLX_UINT32)?;
    let mut columns = Vec::with_capacity(count);
    for code in 0..count {
        let code_start = code * bits;
        let code_end = code_start + bits;
        let first_word = code_start / word_bits;
        let last_word = code_end.div_ceil(word_bits);
        let mut output = ctx.zeros(&[rows], mlx::mlx_dtype__MLX_UINT32)?;
        for word in first_word..last_word {
            let word_start = word * word_bits;
            let start = code_start.max(word_start);
            let end = code_end.min(word_start + word_bits);
            let value = take_matrix_column(ctx, matrix, word as i32)?;
            let value = shifted_piece(
                ctx,
                value,
                start - word_start,
                end - start,
                start - code_start,
            )?;
            output = ctx.binary(mlx::mlx_bitwise_or, output, value)?;
        }
        columns.push(output);
    }
    let unpacked = if columns.is_empty() {
        ctx.zeros(&[rows, 0], mlx::mlx_dtype__MLX_UINT8)?
    } else {
        let unpacked = ctx.stack(&columns, 1)?;
        ctx.astype(unpacked, mlx::mlx_dtype__MLX_UINT8)?
    };
    let mut output_shape = moved.shape;
    *output_shape.last_mut().unwrap() = count_dim;
    ctx.mark_fast("mlx_vectorized_unpack_bits");
    restore_axis(ctx, unpacked, &output_shape, &moved.inverse)
}

fn slice_matrix(
    ctx: &mut TranslationContext,
    matrix: mlx::mlx_array,
    rows: i32,
    start: i32,
    stop: i32,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.slice(matrix, &[0, start], &[rows, stop])
}

fn view_dtype(
    ctx: &mut TranslationContext,
    value: mlx::mlx_array,
    dtype: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    let value = ctx.contiguous(value)?;
    ctx.emit(|out, stream| unsafe { mlx::mlx_view(out, value, dtype, stream) })
}

fn dequantize_input(
    ctx: &mut TranslationContext,
    node: &NodeDesc,
    format: PackedFormat,
    shape: &[i64],
) -> Result<(mlx::mlx_array, Vec<i32>, usize), MlxError> {
    let input = ctx.resolve(&node.inputs[0])?;
    if ctx.dtype_of(input) != mlx::mlx_dtype__MLX_UINT8 || ctx.ndim(input) != 1 {
        return Err("MLX DequantizePacked: input must be rank-1 UINT8".into());
    }
    let (block_elements, block_bytes) = format.geometry();
    let count = element_count(shape, "DequantizePacked")?;
    if shape.is_empty() || shape.last().unwrap() % block_elements as i64 != 0 {
        return Err(format!(
            "MLX DequantizePacked: last dimension must be a multiple of {block_elements}"
        ));
    }
    let expected_bytes = (count / block_elements)
        .checked_mul(block_bytes)
        .ok_or("MLX DequantizePacked: packed byte count overflow; reduce the logical shape")?;
    if ctx.size_of(input) != expected_bytes {
        return Err(format!(
            "MLX DequantizePacked: input has {} bytes; format and shape require {expected_bytes}",
            ctx.size_of(input)
        ));
    }
    let shape = shape
        .iter()
        .copied()
        .map(dim_i32)
        .collect::<Result<_, _>>()?;
    Ok((input, shape, count))
}

fn dequantize_common_mlx(
    ctx: &mut TranslationContext,
    node: &NodeDesc,
    format: PackedFormat,
    shape: &[i64],
) -> Result<mlx::mlx_array, MlxError> {
    let (input, output_shape, count) = dequantize_input(ctx, node, format, shape)?;
    let (block_elements, block_bytes) = format.geometry();
    let blocks = count / block_elements;
    let blocks = dim_i32(blocks as i64)?;
    let matrix = ctx.reshape(input, &[blocks, block_bytes as i32])?;
    let scale = slice_matrix(ctx, matrix, blocks, 0, 2)?;
    let scale = view_dtype(ctx, scale, mlx::mlx_dtype__MLX_FLOAT16)?;
    let scale = ctx.astype(scale, mlx::mlx_dtype__MLX_FLOAT32)?;
    let values = match format {
        PackedFormat::Q4_0 | PackedFormat::Q4_1 => {
            let offset = if format == PackedFormat::Q4_0 { 2 } else { 4 };
            let q = slice_matrix(ctx, matrix, blocks, offset, offset + 16)?;
            let q = ctx.astype(q, mlx::mlx_dtype__MLX_UINT32)?;
            let mask = scalar_u32(ctx, 15);
            let low = ctx.binary(mlx::mlx_bitwise_and, q, mask)?;
            let shift = scalar_u32(ctx, 4);
            let high = ctx.binary(mlx::mlx_right_shift, q, shift)?;
            let q = ctx.concat2(low, high, 1)?;
            let q = ctx.astype(q, mlx::mlx_dtype__MLX_FLOAT32)?;
            if format == PackedFormat::Q4_0 {
                let eight = ctx.scalar_f32(8.0);
                let q = ctx.sub(q, eight)?;
                ctx.mul(q, scale)?
            } else {
                let minimum = slice_matrix(ctx, matrix, blocks, 2, 4)?;
                let minimum = view_dtype(ctx, minimum, mlx::mlx_dtype__MLX_FLOAT16)?;
                let minimum = ctx.astype(minimum, mlx::mlx_dtype__MLX_FLOAT32)?;
                let scaled = ctx.mul(q, scale)?;
                ctx.add(scaled, minimum)?
            }
        }
        PackedFormat::Q8_0 => {
            let q = slice_matrix(ctx, matrix, blocks, 2, 34)?;
            let q = view_dtype(ctx, q, mlx::mlx_dtype__MLX_INT8)?;
            let q = ctx.astype(q, mlx::mlx_dtype__MLX_FLOAT32)?;
            ctx.mul(q, scale)?
        }
        _ => return Err("MLX DequantizePacked: format has no common MLX fast path".into()),
    };
    let values = ctx.reshape(values, &output_shape)?;
    let values = ctx.astype(values, mlx_dtype_from_onnx(node.outputs[0].otype))?;
    ctx.mark_fast(match format {
        PackedFormat::Q4_0 => "mlx_vectorized_dequantize_q4_0",
        PackedFormat::Q4_1 => "mlx_vectorized_dequantize_q4_1",
        PackedFormat::Q8_0 => "mlx_vectorized_dequantize_q8_0",
        _ => unreachable!(),
    });
    Ok(values)
}

fn pack_op(ctx: &mut TranslationContext, node: &NodeDesc) -> Result<(), MlxError> {
    let spec = pack_spec(
        node.ints.get("bits").copied(),
        node.ints.get("radix").copied(),
        node.ints.get("group_size").copied(),
        "Pack",
    )?;
    let axis = node.ints.get("axis").copied().unwrap_or(-1);
    let output_type = node
        .ints
        .get("output_type")
        .copied()
        .unwrap_or(UINT8 as i64);
    let word_bits = if output_type == UINT8 as i64 { 8 } else { 32 };
    if let Some(bits) = node.ints.get("bits").copied() {
        let output = pack_bits_mlx(ctx, node, bits as usize, axis, word_bits)?;
        ctx.bind(&node.outputs[0], output);
        return Ok(());
    }
    let host = read_live(ctx, &node.inputs[0], "Pack")?;
    if host.dtype != mlx::mlx_dtype__MLX_UINT8 {
        return Err("MLX Pack: input must be UINT8".into());
    }
    let (words, shape) = pack_codes(host.data, &host.shape, axis, spec, word_bits)?;
    let shape: Vec<_> = shape.into_iter().map(dim_i32).collect::<Result<_, _>>()?;
    let output = if word_bits == 8 {
        let data: Vec<u8> = words.into_iter().map(|word| word as u8).collect();
        ctx.from_host(
            data.as_ptr().cast::<c_void>(),
            &shape,
            mlx::mlx_dtype__MLX_UINT8,
        )
    } else {
        let data: Vec<u32> = words.into_iter().map(|word| word as u32).collect();
        ctx.from_host(
            data.as_ptr().cast::<c_void>(),
            &shape,
            mlx::mlx_dtype__MLX_UINT32,
        )
    };
    ctx.bind(&node.outputs[0], output);
    Ok(())
}

fn unpack_op(ctx: &mut TranslationContext, node: &NodeDesc) -> Result<(), MlxError> {
    if let Some(bits) = node.ints.get("bits").copied() {
        let values = ctx.read_ints_eval(&node.inputs[1])?;
        if values.len() != 1 || values[0] < 0 {
            return Err("MLX Unpack: count must be a nonnegative scalar INT64 tensor".into());
        }
        let output = unpack_bits_mlx(
            ctx,
            node,
            bits as usize,
            node.ints.get("axis").copied().unwrap_or(-1),
            usize::try_from(values[0])
                .map_err(|_| "MLX Unpack: count does not fit usize".to_string())?,
        )?;
        ctx.bind(&node.outputs[0], output);
        return Ok(());
    }
    let count = {
        let host = read_live(ctx, &node.inputs[1], "Unpack count")?;
        if host.dtype != mlx::mlx_dtype__MLX_INT64 || !host.shape.is_empty() || host.data.len() != 8
        {
            return Err("MLX Unpack: count must be a scalar INT64 tensor".into());
        }
        i64::from_ne_bytes(host.data.try_into().unwrap())
    };
    let spec = pack_spec(
        node.ints.get("bits").copied(),
        node.ints.get("radix").copied(),
        node.ints.get("group_size").copied(),
        "Unpack",
    )?;
    let axis = node.ints.get("axis").copied().unwrap_or(-1);
    let host = read_live(ctx, &node.inputs[0], "Unpack")?;
    let (word_bits, words): (usize, Vec<u64>) = match host.dtype {
        mlx::mlx_dtype__MLX_UINT8 => (8, host.data.iter().map(|&value| value as u64).collect()),
        mlx::mlx_dtype__MLX_UINT32 => (
            32,
            host.data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&bytes| u32::from_ne_bytes(bytes) as u64)
                .collect(),
        ),
        _ => return Err("MLX Unpack: input must be UINT8 or UINT32".into()),
    };
    let (values, shape) = unpack_codes(&words, &host.shape, axis, count, spec, word_bits)?;
    let shape: Vec<_> = shape.into_iter().map(dim_i32).collect::<Result<_, _>>()?;
    let output = ctx.from_host(
        values.as_ptr().cast::<c_void>(),
        &shape,
        mlx::mlx_dtype__MLX_UINT8,
    );
    ctx.bind(&node.outputs[0], output);
    Ok(())
}

fn dequantize_packed_op(ctx: &mut TranslationContext, node: &NodeDesc) -> Result<(), MlxError> {
    let format_name = node
        .strings
        .get("format")
        .ok_or("MLX DequantizePacked: required format attribute is missing")?;
    let format = PackedFormat::parse(format_name)?;
    let shape = node
        .int_arrays
        .get("shape")
        .ok_or("MLX DequantizePacked: required shape attribute is missing")?;
    if let Some(output) = gpu::dequantize(ctx, node, format, shape)? {
        ctx.bind(&node.outputs[0], output);
        return Ok(());
    }
    if matches!(
        format,
        PackedFormat::Q4_0 | PackedFormat::Q4_1 | PackedFormat::Q8_0
    ) && node.outputs[0].otype != DOUBLE
    {
        let output = dequantize_common_mlx(ctx, node, format, shape)?;
        ctx.bind(&node.outputs[0], output);
        return Ok(());
    }
    let host = read_live(ctx, &node.inputs[0], "DequantizePacked")?;
    if host.dtype != mlx::mlx_dtype__MLX_UINT8 || host.shape.len() != 1 {
        return Err("MLX DequantizePacked: input must be rank-1 UINT8".into());
    }
    let values = decode_packed(host.data, format, shape)?;
    let (bytes, dtype) = output_bytes(&values, node.outputs[0].otype)?;
    let shape: Vec<_> = shape
        .iter()
        .copied()
        .map(dim_i32)
        .collect::<Result<_, _>>()?;
    let output = ctx.from_host(bytes.as_ptr().cast::<c_void>(), &shape, dtype);
    ctx.bind(&node.outputs[0], output);
    Ok(())
}

fn claim_pack_spec(node: &NodeView, op: &str) -> Result<PackSpec, String> {
    let sentinel = i64::MIN;
    pack_spec(
        (node.int_attr("bits", sentinel) != sentinel).then(|| node.int_attr("bits", sentinel)),
        (node.int_attr("radix", sentinel) != sentinel).then(|| node.int_attr("radix", sentinel)),
        (node.int_attr("group_size", sentinel) != sentinel)
            .then(|| node.int_attr("group_size", sentinel)),
        op,
    )
}

fn pack_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "Pack expects one input and one output"
    );
    let (input, output) = match (node.input_info(0), node.output_info(0)) {
        (Some(input), Some(output)) => (input, output),
        _ => deny!("Pack requires tensor input/output type information"),
    };
    require!(input.dtype == UINT8, "Pack input must be UINT8");
    require!(
        output.dtype == UINT8 || output.dtype == UINT32,
        "Pack output must be UINT8 or UINT32"
    );
    require!(
        !input.shape.is_empty(),
        "Pack input rank must be at least one"
    );
    require!(
        output.shape.len() == input.shape.len(),
        "Pack output rank must match input rank"
    );
    let axis = normalize_axis(node.int_attr("axis", -1), input.shape.len(), "Pack")?;
    let spec = claim_pack_spec(node, "Pack")?;
    let output_type = node.int_attr("output_type", UINT8 as i64);
    require!(
        output_type == output.dtype as i64,
        "Pack output_type {output_type} does not match output tensor dtype {}",
        output.dtype
    );
    for index in 0..input.shape.len() {
        if index != axis && input.shape[index] >= 0 && output.shape[index] >= 0 {
            require!(
                input.shape[index] == output.shape[index],
                "Pack output dimension {index} is {}, expected {}",
                output.shape[index],
                input.shape[index]
            );
        }
    }
    if input.shape[axis] >= 0 {
        let word_bits = if output.dtype == UINT8 { 8 } else { 32 };
        let expected = packed_axis_count(input.shape[axis] as usize, spec, word_bits)?;
        require!(
            output.shape[axis] < 0 || output.shape[axis] == expected as i64,
            "Pack output axis is {}, expected {expected}",
            output.shape[axis]
        );
    }
    Ok(())
}

fn unpack_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "Unpack expects two inputs and one output"
    );
    let (input, count, output) = match (node.input_info(0), node.input_info(1), node.output_info(0))
    {
        (Some(input), Some(count), Some(output)) => (input, count, output),
        _ => deny!("Unpack requires tensor input/output type information"),
    };
    require!(
        input.dtype == UINT8 || input.dtype == UINT32,
        "Unpack input must be UINT8 or UINT32"
    );
    require!(
        count.dtype == INT64 && count.shape.is_empty(),
        "Unpack count must be scalar INT64"
    );
    require!(output.dtype == UINT8, "Unpack output must be UINT8");
    require!(
        !input.shape.is_empty(),
        "Unpack input rank must be at least one"
    );
    require!(
        output.shape.len() == input.shape.len(),
        "Unpack output rank must match input rank"
    );
    let axis = normalize_axis(node.int_attr("axis", -1), input.shape.len(), "Unpack")?;
    for index in 0..input.shape.len() {
        if index != axis && input.shape[index] >= 0 && output.shape[index] >= 0 {
            require!(
                input.shape[index] == output.shape[index],
                "Unpack output dimension {index} is {}, expected {}",
                output.shape[index],
                input.shape[index]
            );
        }
    }
    claim_pack_spec(node, "Unpack")?;
    Ok(())
}

fn dequantize_packed_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "DequantizePacked expects one input and one output"
    );
    let (input, output) = match (node.input_info(0), node.output_info(0)) {
        (Some(input), Some(output)) => (input, output),
        _ => deny!("DequantizePacked requires tensor input/output type information"),
    };
    require!(
        input.dtype == UINT8 && input.shape.len() == 1,
        "DequantizePacked input must be rank-1 UINT8"
    );
    require!(
        input.shape[0] <= i64::from(i32::MAX),
        "DequantizePacked encoded payload exceeds MLX's int32 dimension limit; split the expert bank or reduce its channel dimensions"
    );
    require!(
        matches!(output.dtype, FLOAT | FLOAT16 | BFLOAT16 | DOUBLE),
        "DequantizePacked output must be float, float16, bfloat16, or double"
    );
    let format_name = node.string_attr("format", "");
    require!(!format_name.is_empty(), "DequantizePacked requires format");
    let format = PackedFormat::parse(&format_name)?;
    let (present, shape) = node.ints_attr("shape");
    require!(
        present
            && !shape.is_empty()
            && shape
                .iter()
                .all(|&dim| (0..=i64::from(i32::MAX)).contains(&dim)),
        "DequantizePacked shape must be nonempty and contain nonnegative MLX int32 dimensions; split dimensions larger than 2147483647"
    );
    let output_dtype = node.int_attr("output_dtype", FLOAT as i64);
    require!(
        output_dtype == output.dtype as i64,
        "DequantizePacked output_dtype {output_dtype} does not match output tensor dtype {}",
        output.dtype
    );
    require!(
        output.shape.len() == shape.len(),
        "DequantizePacked output rank must match shape attribute"
    );
    for (index, (&expected, &actual)) in shape.iter().zip(&output.shape).enumerate() {
        require!(
            actual < 0 || actual == expected,
            "DequantizePacked output dimension {index} is {actual}, expected {expected}"
        );
    }
    let (block_elements, block_bytes) = format.geometry();
    require!(
        shape.last().unwrap() % block_elements as i64 == 0,
        "DequantizePacked last dimension must be a multiple of {block_elements}"
    );
    if shape.iter().all(|&dim| dim >= 0) {
        let count = element_count(&shape, "DequantizePacked")?;
        let expected = (count / block_elements)
            .checked_mul(block_bytes)
            .ok_or("DequantizePacked byte count overflows; reduce the logical shape")?;
        require!(
            input.shape[0] < 0 || input.shape[0] == expected as i64,
            "DequantizePacked input has {} bytes, expected {expected}",
            input.shape[0]
        );
    }
    Ok(())
}

fn packed_compile_shape_safety(node: &NodeView) -> CompileShapeSafety {
    const FAST_REASON: &str =
        "vectorized packed lowering depends on concrete row and packed-axis geometry";
    const HOST_REASON: &str =
        "exact grouped-radix or complex GGUF decoding requires eager host payload access";
    match node.op_type().as_str() {
        "Pack" if node.int_attr("bits", 0) == 8 => CompileShapeSafety::ShapeKeyedOnly {
            reason: FAST_REASON,
        },
        "Pack" if (1..8).contains(&node.int_attr("bits", 0)) => CompileShapeSafety::EagerOnly {
            reason: "Pack validates live codes against the bit width on every invocation",
        },
        "Unpack" if node.int_attr("bits", 0) != 0 && node.const_scalar_i64(1).is_some() => {
            CompileShapeSafety::ShapeKeyedOnly {
                reason: FAST_REASON,
            }
        }
        "DequantizePacked" if gpu::has_decoder(node) => CompileShapeSafety::ShapeKeyedOnly {
            reason: FAST_REASON,
        },
        _ => CompileShapeSafety::EagerOnly {
            reason: HOST_REASON,
        },
    }
}

pub fn register(registry: &mut OpRegistry) {
    for (op_type, handler, claim) in [
        (
            "Pack",
            pack_op as crate::registry::OpHandler,
            pack_claim as crate::registry::ClaimPredicate,
        ),
        (
            "Unpack",
            unpack_op as crate::registry::OpHandler,
            unpack_claim as crate::registry::ClaimPredicate,
        ),
        (
            "DequantizePacked",
            dequantize_packed_op as crate::registry::OpHandler,
            dequantize_packed_claim as crate::registry::ClaimPredicate,
        ),
    ] {
        registry.register_shape_classified(
            OpRegistration {
                domain: "",
                op_type,
                min_opset: 29,
                max_opset: 29,
                handler,
                claim,
            },
            packed_compile_shape_safety,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{OutRef, Plan, Src, TensorRef};
    use crate::mlx::{Array, Stream, VectorArray, eval};
    use crate::registry::CompileShapeSafety;
    use sha2::{Digest, Sha256};

    fn reference(name: &str) -> TensorRef {
        TensorRef {
            name: name.into(),
            source: Src::Intermediate,
            ..TensorRef::absent()
        }
    }

    fn output(name: &str, dtype: ort::ONNXTensorElementDataType) -> OutRef {
        OutRef {
            name: name.into(),
            external: false,
            ctx_index: 0,
            otype: dtype,
        }
    }

    fn evaluate(array: mlx::mlx_array) {
        let mut arrays = VectorArray::new();
        arrays.append(array);
        eval(&arrays).unwrap();
    }

    fn write_reference_half(bytes: &mut [u8], offset: usize, value: f32) {
        bytes[offset..offset + 2].copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
    }

    fn reference_payload(name: &str, size: usize) -> Vec<u8> {
        let mut bytes: Vec<_> = (0..size)
            .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
            .collect();
        match name {
            "gguf:q4_0" | "gguf:q5_0" | "gguf:q8_0" | "gguf:iq2_xxs" | "gguf:iq2_xs"
            | "gguf:iq3_xxs" | "gguf:iq1_s" | "gguf:iq4_nl" | "gguf:iq3_s" | "gguf:iq2_s"
            | "gguf:q1_0" | "gguf:q2_0" => write_reference_half(&mut bytes, 0, 0.75),
            "gguf:q4_1" | "gguf:q5_1" | "gguf:q8_1" => {
                write_reference_half(&mut bytes, 0, 0.75);
                write_reference_half(&mut bytes, 2, -0.5);
            }
            "gguf:q2_k" => {
                write_reference_half(&mut bytes, 80, 0.75);
                write_reference_half(&mut bytes, 82, 0.25);
            }
            "gguf:q3_k" => write_reference_half(&mut bytes, 108, 0.75),
            "gguf:q4_k" | "gguf:q5_k" => {
                write_reference_half(&mut bytes, 0, 0.75);
                write_reference_half(&mut bytes, 2, 0.25);
            }
            "gguf:q6_k" => write_reference_half(&mut bytes, 208, 0.75),
            "gguf:q8_k" => bytes[..4].copy_from_slice(&0.75f32.to_le_bytes()),
            "gguf:iq4_xs" => write_reference_half(&mut bytes, 0, 0.75),
            "gguf:iq1_m" => {
                let bits = f16::from_f32(0.75).to_bits();
                for index in 0..4 {
                    let offset = 48 + 2 * index;
                    let low = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]) & 0x0fff;
                    let word = low | (((bits >> (4 * index)) & 15) << 12);
                    bytes[offset..offset + 2].copy_from_slice(&word.to_le_bytes());
                }
            }
            "gguf:tq1_0" => write_reference_half(&mut bytes, 52, 0.75),
            "gguf:tq2_0" => write_reference_half(&mut bytes, 64, 0.75),
            "gguf:mxfp4" => bytes[0] = 126,
            "gguf:nvfp4" => bytes[..4].copy_from_slice(&[0x38, 0x21, 0x01, 0x7f]),
            _ => panic!("missing reference payload for {name}"),
        }
        bytes
    }

    fn iq_edge_payload(name: &str, block_size: usize) -> Vec<u8> {
        let mut state = 0x6d2b_79f5u32;
        let mut bytes: Vec<_> = (0..3 * block_size)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state % 255 + 1) as u8
            })
            .collect();
        for (block, scale) in [0.5, -1.0, 65504.0].into_iter().enumerate() {
            let base = block * block_size;
            if name == "gguf:iq1_m" {
                let bits = f16::from_f32(scale).to_bits();
                for (index, low) in [0x000, 0x249, 0xdb6, 0xfff].into_iter().enumerate() {
                    let offset = base + 48 + 2 * index;
                    let word = low | (((bits >> (4 * index)) & 15) << 12);
                    bytes[offset..offset + 2].copy_from_slice(&word.to_le_bytes());
                }
                for (index, byte) in bytes[base + 32..base + 48].iter_mut().enumerate() {
                    *byte = if index % 2 == 0 { 0x08 } else { 0xf7 };
                }
            } else {
                write_reference_half(&mut bytes, base, scale);
            }
            match name {
                "gguf:iq2_xxs" => {
                    bytes[base + 6..base + 10].fill(0xff);
                    bytes[base + 14..base + 18].fill(0);
                }
                "gguf:iq2_xs" => {
                    bytes[base + 2..base + 4].copy_from_slice(&0xfe00u16.to_le_bytes());
                    bytes[base + 4..base + 6].fill(0);
                    bytes[base + 66..base + 68].copy_from_slice(&[0xf0, 0x0f]);
                }
                "gguf:iq2_s" => {
                    bytes[base + 34..base + 36].copy_from_slice(&[0, 0xff]);
                    bytes[base + 66] = 0xff;
                    bytes[base + 74..base + 76].copy_from_slice(&[0xf0, 0x0f]);
                }
                "gguf:iq3_xxs" => {
                    bytes[base + 66..base + 70].fill(0xff);
                    bytes[base + 70..base + 74].fill(0);
                }
                "gguf:iq3_s" => {
                    bytes[base + 66] = 0xff;
                    bytes[base + 74..base + 76].copy_from_slice(&[0, 0xff]);
                    bytes[base + 106..base + 108].copy_from_slice(&[0xf0, 0x0f]);
                }
                "gguf:iq1_s" => {
                    bytes[base + 34..base + 36].copy_from_slice(&0x8000u16.to_le_bytes());
                    bytes[base + 36..base + 38].copy_from_slice(&0x7fffu16.to_le_bytes());
                }
                "gguf:iq1_m" => {}
                _ => panic!("missing IQ edge payload for {name}"),
            }
        }
        bytes
    }

    fn spec_bits(bits: i64) -> PackSpec {
        pack_spec(Some(bits), None, None, "test").unwrap()
    }

    fn spec_group(radix: i64, group_size: i64) -> PackSpec {
        pack_spec(None, Some(radix), Some(group_size), "test").unwrap()
    }

    #[test]
    fn pack_examples_and_uint32_storage() {
        let (words, shape) =
            pack_codes(&[0, 1, 2, 3, 4, 5, 6, 7], &[8], -1, spec_bits(3), 8).unwrap();
        assert_eq!(words, [136, 198, 250]);
        assert_eq!(shape, [3]);

        let (words, shape) = pack_codes(&[1, 2, 3], &[3], 0, spec_bits(5), 32).unwrap();
        assert_eq!(words, [3137]);
        assert_eq!(shape, [1]);

        let (words, _) = pack_codes(&[1, 2, 3, 31, 0, 16], &[2, 3], 1, spec_bits(5), 8).unwrap();
        assert_eq!(words, [65, 12, 31, 64]);
    }

    #[test]
    fn grouped_radix_and_arbitrary_axis_round_trip() {
        let spec = spec_group(3, 5);
        let (words, shape) =
            pack_codes(&[0, 1, 2, 0, 1, 2, 2, 2, 2, 2], &[2, 5], 0, spec, 8).unwrap();
        assert_eq!(shape, [1, 5]);
        let (decoded, decoded_shape) = unpack_codes(&words, &shape, 0, 2, spec, 8).unwrap();
        assert_eq!(decoded_shape, [2, 5]);
        assert_eq!(decoded, [0, 1, 2, 0, 1, 2, 2, 2, 2, 2]);

        let (words, _) = pack_codes(&[0, 1, 2, 0, 1], &[5], -1, spec, 8).unwrap();
        assert_eq!(words, [102]);
        let (words, _) = pack_codes(&[1, 2], &[2], -1, spec, 8).unwrap();
        assert_eq!(words, [7]);
    }

    #[test]
    fn grouped_empty_rows_do_not_overflow_independent_dimension_products() {
        let spec = spec_group(3, 5);
        let shape = [50_000, 50_000, 5, 0];
        let (packed, packed_shape) = pack_codes(&[], &shape, 2, spec, 8).unwrap();
        assert!(packed.is_empty());
        assert_eq!(packed_shape, [50_000, 50_000, 1, 0]);
        let (unpacked, unpacked_shape) = unpack_codes(&[], &packed_shape, 2, 5, spec, 8).unwrap();
        assert!(unpacked.is_empty());
        assert_eq!(unpacked_shape, shape);
        assert!(unpack_codes(&[], &packed_shape, 2, 0, spec, 8).is_err());
    }

    #[test]
    fn pack_unpack_match_onnx_reference_axis_vectors() {
        type Case<'a> = (&'a [u8], &'a [i64], i64, PackSpec, usize, &'a [u64]);
        let cases: [Case<'_>; 4] = [
            (
                &[1, 7, 3, 2, 0, 4, 5, 6, 1, 7, 2, 0],
                &[4, 3],
                0,
                spec_bits(3),
                8,
                &[81, 135, 99, 15, 5, 0],
            ),
            (
                &[1, 2, 3, 4, 5, 31, 0, 16, 7, 9],
                &[2, 5],
                1,
                spec_bits(5),
                32,
                &[5_377_089, 9_682_975],
            ),
            (
                &[0, 2, 1, 1, 1, 2, 2, 0, 0, 0, 2, 1, 1, 1, 2, 2, 0, 0],
                &[6, 3],
                0,
                spec_group(3, 5),
                8,
                &[102, 140, 196, 2, 0, 0],
            ),
            (
                &[0, 1, 2, 0, 1, 2, 2, 2, 2, 2, 2, 1],
                &[2, 6],
                1,
                spec_group(3, 5),
                32,
                &[614, 498],
            ),
        ];
        for (input, shape, axis, spec, word_bits, expected) in cases {
            let (packed, packed_shape) = pack_codes(input, shape, axis, spec, word_bits).unwrap();
            assert_eq!(packed, expected);
            let (unpacked, unpacked_shape) = unpack_codes(
                &packed,
                &packed_shape,
                axis,
                shape[axis.rem_euclid(shape.len() as i64) as usize],
                spec,
                word_bits,
            )
            .unwrap();
            assert_eq!(unpacked_shape, shape);
            assert_eq!(unpacked, input);
        }
    }

    #[test]
    fn grouped_radix_honors_positive_int64_bound() {
        assert_eq!(spec_group(2, 62).group_width, 62);
        assert!(pack_spec(None, Some(2), Some(63), "Pack").is_err());
        assert!(pack_spec(None, Some(256), Some(8), "Pack").is_err());
        assert!(pack_spec(Some(4), Some(16), Some(1), "Pack").is_err());
        assert!(pack_spec(None, Some(3), None, "Pack").is_err());
    }

    #[test]
    fn grouped_62_bit_index_crosses_uint32_words_exactly() {
        let input: Vec<u8> = (0..62).map(|index| (index % 3 == 0) as u8).collect();
        let spec = spec_group(2, 62);
        let (words, shape) = pack_codes(&input, &[62], 0, spec, 32).unwrap();
        assert_eq!(shape, [2]);
        let (decoded, decoded_shape) = unpack_codes(&words, &shape, 0, 62, spec, 32).unwrap();
        assert_eq!(decoded_shape, [62]);
        assert_eq!(decoded, input);
    }

    #[test]
    fn unpack_rejects_bad_geometry_and_group_indices() {
        let spec = spec_group(3, 5);
        assert!(unpack_codes(&[255], &[1], 0, 5, spec, 8).is_err());
        assert!(unpack_codes(&[102], &[1], 0, -1, spec, 8).is_err());
        assert!(unpack_codes(&[102, 0], &[2], 0, 5, spec, 8).is_err());
    }

    #[test]
    fn every_format_decodes_zero_block() {
        let formats = [
            "gguf:q4_0",
            "gguf:q4_1",
            "gguf:q5_0",
            "gguf:q5_1",
            "gguf:q8_0",
            "gguf:q8_1",
            "gguf:q2_k",
            "gguf:q3_k",
            "gguf:q4_k",
            "gguf:q5_k",
            "gguf:q6_k",
            "gguf:q8_k",
            "gguf:iq2_xxs",
            "gguf:iq2_xs",
            "gguf:iq3_xxs",
            "gguf:iq1_s",
            "gguf:iq4_nl",
            "gguf:iq3_s",
            "gguf:iq2_s",
            "gguf:iq4_xs",
            "gguf:iq1_m",
            "gguf:tq1_0",
            "gguf:tq2_0",
            "gguf:mxfp4",
            "gguf:nvfp4",
            "gguf:q1_0",
            "gguf:q2_0",
        ];
        assert_eq!(formats.len(), 27);
        for name in formats {
            let format = PackedFormat::parse(name).unwrap();
            let (elements, bytes) = format.geometry();
            let output = decode_packed(&vec![0; bytes], format, &[elements as i64]).unwrap();
            assert_eq!(output.len(), elements, "{name}");
            assert!(output.iter().all(|value| value.is_finite()), "{name}");
        }
    }

    #[test]
    fn supported_formats_match_onnx_reference_golden_hashes() {
        let expected = [
            (
                "gguf:q4_0",
                "a90720e173d085dc9909350e325b5aedcde971c741dddaffd79be99bf2440e95",
            ),
            (
                "gguf:q4_1",
                "3477c46e645e6efc46e1f4db0a3a9da86f67bc42b4590b07d208f247a35b43b8",
            ),
            (
                "gguf:q5_0",
                "83de917dca8f6fc5b443c1df9ca0f56d5e4beee705be983a5b57ebe1b85df0d0",
            ),
            (
                "gguf:q5_1",
                "22901be3fbd743e1cb8033f1fa8dd2df41ef750eac1d66b90b08fdfab0ab0d7b",
            ),
            (
                "gguf:q8_0",
                "81c210fbb597956e5bb5886548765bcb89b4543c859ada164db6ec4108480a61",
            ),
            (
                "gguf:q8_1",
                "6a88aa6f38aae39f127b4d409d132df3c72270a963b7ddeb689c316d75ac04a2",
            ),
            (
                "gguf:q2_k",
                "23eeb776edeb5dd8108599db95ad3e4a9da2df69c25e9babb69e373048961ce8",
            ),
            (
                "gguf:q3_k",
                "4ca578e80d4c1667d452dddbda42202bb5ac50d7550a7e9c07c590b620d0d476",
            ),
            (
                "gguf:q4_k",
                "0b7d60960bbe06d8729b29efd26faa5596d34e3d3bd6b4e9429deeb9bd74ccfa",
            ),
            (
                "gguf:q5_k",
                "bace8af8173213f13e200bc12909fcc01d91fcde7b0f194aa912c28f72c007db",
            ),
            (
                "gguf:q6_k",
                "2fa306b0594050ec51f42446383458c63c9e35d6ca49c0edab12e683d52c17c8",
            ),
            (
                "gguf:q8_k",
                "8dfa3856e7e2d6a013d7972667db95a0ef582090fd06aa181d0b6bf17cc6b816",
            ),
            (
                "gguf:iq2_xxs",
                "2d847c629a20c0680fd803ff4a3e6b9e52c1c0370827c1f2e298460232a9d4ac",
            ),
            (
                "gguf:iq2_xs",
                "3794aef862517c04fe35779d80807c6e11a0ac68192c5642f665214c0441f114",
            ),
            (
                "gguf:iq3_xxs",
                "0837edfdefd1d16019a57cd13b9c1e3da84b2b4111f0ac533bdfad8e5562eae1",
            ),
            (
                "gguf:iq1_s",
                "d684261b0f868b731aa93073192579ffe72cb620d8687c698fdbeed1b3485189",
            ),
            (
                "gguf:iq4_nl",
                "f6a109ad08842a7434e5ede115bb231181fb91f3d074ab73fe7d0b80b543f34c",
            ),
            (
                "gguf:iq3_s",
                "a1eb566ccfbfc5cb89a10a48c2d91d897508160a4f23ee7b85f4b639fb9e40a4",
            ),
            (
                "gguf:iq2_s",
                "6fab6c67ea45a243db184a07bb39fea51fcb2085cc89a512bd88e5751496e802",
            ),
            (
                "gguf:iq4_xs",
                "71290fdeb45737ab8df10e7c517805451e222c5321efc5f4274b6208d8ce382b",
            ),
            (
                "gguf:iq1_m",
                "78a282a22eb9b4f25a773fcaaf6d2d11555620e31031e04fb8b7c6a9f377f228",
            ),
            (
                "gguf:tq1_0",
                "828722236022425372a33c139a735a5df1e2b0531a37e843ed999b443e51f644",
            ),
            (
                "gguf:tq2_0",
                "4d47fb49ce1830cb43d7e7aa1671a68c6c8881388b6af66a674b7b9fb9ed1cf6",
            ),
            (
                "gguf:mxfp4",
                "225c2bd8325c321b0f614e4223e7c5926faa1aae8f9147d5b46261151b4a971c",
            ),
            (
                "gguf:nvfp4",
                "7786b973f254aeba6c190e61c7205102f8816b034e5fdded7490fe67ca1c4305",
            ),
            (
                "gguf:q1_0",
                "aa4b746ef275e19003de06231cfce05503506bb9e3578e3e0f6fe2265c35380d",
            ),
            (
                "gguf:q2_0",
                "a737a64244923a3ed27285517e505c26992139cd2300f0269d4e85048abed865",
            ),
        ];
        for (name, expected_hash) in expected {
            let format = PackedFormat::parse(name).unwrap();
            let (elements, bytes) = format.geometry();
            let output =
                decode_packed(&reference_payload(name, bytes), format, &[elements as i64]).unwrap();
            let encoded: Vec<_> = output
                .into_iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            assert_eq!(
                format!("{:x}", Sha256::digest(encoded)),
                expected_hash,
                "{name}"
            );
        }
    }

    #[test]
    fn iq_multiblock_edges_match_onnx_reference_golden_hashes() {
        let expected = [
            (
                "gguf:iq2_xxs",
                "9dae46c6ae7941e2eda7453fb20b8982176eea4f39ee83f11ad5d8eb3efeb2da",
            ),
            (
                "gguf:iq2_xs",
                "0e88985fd051c2ceb6e0445c29bb7310f62995166b3add6e4323b128d319ae6a",
            ),
            (
                "gguf:iq2_s",
                "bbe22508be22eb4de5426c94a7a23e319db5444871772ff4ec0db41c334aa537",
            ),
            (
                "gguf:iq3_xxs",
                "60ca09a43f9f8a511ec6574ad2fa91b494f19af494a857548c45d5f57032d3d0",
            ),
            (
                "gguf:iq3_s",
                "25529fa9c30b113bd83e14155b4a2e5b25377f28e8f92e08c43b060e6f502e4f",
            ),
            (
                "gguf:iq1_s",
                "aa41c544e32edfa8c6e6ba342d8680504107498a183efba6246d8b64bc5a4e64",
            ),
            (
                "gguf:iq1_m",
                "868b328bfa9e3547645fde1deae4f97a91fc4f4d655181d795bba6351c023699",
            ),
        ];
        for (name, expected_hash) in expected {
            let format = PackedFormat::parse(name).unwrap();
            let (elements, bytes) = format.geometry();
            let output = decode_packed(
                &iq_edge_payload(name, bytes),
                format,
                &[3 * elements as i64],
            )
            .unwrap();
            assert!(output.iter().all(|value| value.is_finite()), "{name}");
            let encoded: Vec<_> = output
                .into_iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            assert_eq!(
                format!("{:x}", Sha256::digest(encoded)),
                expected_hash,
                "{name}"
            );
        }
    }

    #[test]
    fn conventional_decoders_follow_lane_order_and_affine_rules() {
        let one = f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut q4 = vec![0u8; 18];
        q4[..2].copy_from_slice(&one);
        q4[2] = 0x1f;
        let output = decode_block(PackedFormat::Q4_0, &q4);
        assert_eq!(output[0], 7.0);
        assert_eq!(output[1], -8.0);
        assert_eq!(output[16], -7.0);

        let mut q4_1 = vec![0u8; 20];
        q4_1[..2].copy_from_slice(&one);
        q4_1[2..4].copy_from_slice(&f16::from_f32(2.0).to_bits().to_le_bytes());
        q4_1[4] = 0x31;
        let output = decode_block(PackedFormat::Q4_1, &q4_1);
        assert_eq!(output[0], 3.0);
        assert_eq!(output[16], 5.0);
    }

    #[test]
    fn decode_validates_exact_payload_and_block_alignment() {
        let format = PackedFormat::Q4_0;
        assert!(decode_packed(&[], format, &[]).is_err());
        assert!(decode_packed(&[], format, &[31]).is_err());
        assert!(decode_packed(&[0; 17], format, &[32]).is_err());
        assert!(decode_packed(&[], format, &[0, 32]).unwrap().is_empty());
    }

    #[test]
    fn mlx_fast_paths_pack_unpack_and_common_dequantize() {
        let stream = Stream::new_default_gpu();
        let mut plan = Plan::new(vec![]);
        let mut ctx = TranslationContext::new(
            &mut plan,
            std::ptr::null(),
            std::ptr::null_mut(),
            stream.as_raw(),
        );

        let input = [1u8, 2, 3, 31, 0, 16];
        let x = ctx.keep(Array::from_data(
            input.as_ptr().cast(),
            &[2, 3],
            mlx::mlx_dtype__MLX_UINT8,
        ));
        ctx.bind(&output("x", UINT8), x);
        let mut pack = NodeDesc::new(
            "Pack".into(),
            "".into(),
            29,
            CompileShapeSafety::ShapeKeyedOnly {
                reason: "packed test",
            },
        );
        pack.inputs = vec![reference("x")];
        pack.outputs = vec![output("packed", UINT8)];
        let packed = pack_bits_mlx(&mut ctx, &pack, 5, 1, 8).unwrap();
        let packed = ctx.contiguous(packed).unwrap();
        evaluate(packed);
        let packed_values =
            unsafe { std::slice::from_raw_parts(mlx::mlx_array_data_uint8(packed), 4) };
        assert_eq!(packed_values, [65, 12, 31, 64]);

        ctx.bind(&output("packed", UINT8), packed);
        let mut unpack = NodeDesc::new(
            "Unpack".into(),
            "".into(),
            29,
            CompileShapeSafety::ShapeKeyedOnly {
                reason: "packed test",
            },
        );
        unpack.inputs = vec![reference("packed"), reference("count")];
        unpack.outputs = vec![output("unpacked", UINT8)];
        let unpacked = unpack_bits_mlx(&mut ctx, &unpack, 5, 1, 3).unwrap();
        let unpacked = ctx.contiguous(unpacked).unwrap();
        evaluate(unpacked);
        let unpacked_values =
            unsafe { std::slice::from_raw_parts(mlx::mlx_array_data_uint8(unpacked), 6) };
        assert_eq!(unpacked_values, input);

        let axis_input = [1u8, 31, 2, 0, 3, 16];
        let axis_x = ctx.keep(Array::from_data(
            axis_input.as_ptr().cast(),
            &[3, 2],
            mlx::mlx_dtype__MLX_UINT8,
        ));
        ctx.bind(&output("axis_x", UINT8), axis_x);
        pack.inputs = vec![reference("axis_x")];
        pack.outputs = vec![output("axis_packed", UINT32)];
        let axis_packed = pack_bits_mlx(&mut ctx, &pack, 5, 0, 32).unwrap();
        let axis_packed = ctx.contiguous(axis_packed).unwrap();
        evaluate(axis_packed);
        let axis_words =
            unsafe { std::slice::from_raw_parts(mlx::mlx_array_data_uint32(axis_packed), 2) };
        assert_eq!(axis_words, [3137, 16415]);
        ctx.bind(&output("axis_packed", UINT32), axis_packed);
        unpack.inputs[0] = reference("axis_packed");
        let axis_unpacked = unpack_bits_mlx(&mut ctx, &unpack, 5, 0, 3).unwrap();
        let axis_unpacked = ctx.contiguous(axis_unpacked).unwrap();
        evaluate(axis_unpacked);
        let axis_values =
            unsafe { std::slice::from_raw_parts(mlx::mlx_array_data_uint8(axis_unpacked), 6) };
        assert_eq!(axis_values, axis_input);

        let mut payload = [0u8; 18];
        payload[..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        payload[2] = 0x1f;
        let q = ctx.keep(Array::from_data(
            payload.as_ptr().cast(),
            &[18],
            mlx::mlx_dtype__MLX_UINT8,
        ));
        ctx.bind(&output("q", UINT8), q);
        let mut dequantize = NodeDesc::new(
            "DequantizePacked".into(),
            "".into(),
            29,
            CompileShapeSafety::ShapeKeyedOnly {
                reason: "packed test",
            },
        );
        dequantize.inputs = vec![reference("q")];
        dequantize.outputs = vec![output("dequantized", FLOAT)];
        let dequantized =
            dequantize_common_mlx(&mut ctx, &dequantize, PackedFormat::Q4_0, &[32]).unwrap();
        let dequantized = ctx.contiguous(dequantized).unwrap();
        evaluate(dequantized);
        let values =
            unsafe { std::slice::from_raw_parts(mlx::mlx_array_data_float32(dequantized), 32) };
        assert_eq!(values[0], 7.0);
        assert_eq!(values[1], -8.0);
        assert_eq!(values[16], -7.0);

        let mut q4_1 = [0u8; 20];
        q4_1[..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        q4_1[2..4].copy_from_slice(&f16::from_f32(2.0).to_bits().to_le_bytes());
        q4_1[4] = 0x31;
        let q = ctx.keep(Array::from_data(
            q4_1.as_ptr().cast(),
            &[20],
            mlx::mlx_dtype__MLX_UINT8,
        ));
        ctx.bind(&output("q4_1", UINT8), q);
        dequantize.inputs = vec![reference("q4_1")];
        let values =
            dequantize_common_mlx(&mut ctx, &dequantize, PackedFormat::Q4_1, &[32]).unwrap();
        let values = ctx.contiguous(values).unwrap();
        evaluate(values);
        let values = unsafe { std::slice::from_raw_parts(mlx::mlx_array_data_float32(values), 32) };
        assert_eq!(values[0], 3.0);
        assert_eq!(values[16], 5.0);

        let mut q8 = [0u8; 34];
        q8[..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        q8[2] = (-2i8) as u8;
        q8[3] = 4;
        let q = ctx.keep(Array::from_data(
            q8.as_ptr().cast(),
            &[34],
            mlx::mlx_dtype__MLX_UINT8,
        ));
        ctx.bind(&output("q8", UINT8), q);
        dequantize.inputs = vec![reference("q8")];
        let values =
            dequantize_common_mlx(&mut ctx, &dequantize, PackedFormat::Q8_0, &[32]).unwrap();
        let values = ctx.contiguous(values).unwrap();
        evaluate(values);
        let values = unsafe { std::slice::from_raw_parts(mlx::mlx_array_data_float32(values), 32) };
        assert_eq!(values[0], -1.0);
        assert_eq!(values[1], 2.0);
    }

    #[test]
    fn mlx_bit_fast_paths_match_host_for_every_width() {
        let shape = [2i64, 17, 3];
        for bits in 1..=8 {
            let radix = 1usize << bits;
            let input: Vec<u8> = (0..102)
                .map(|index| ((index * 7 + 3) % radix) as u8)
                .collect();
            for word_bits in [8usize, 32] {
                let (expected, packed_shape) =
                    pack_codes(&input, &shape, 1, spec_bits(bits as i64), word_bits).unwrap();
                let stream = Stream::new_default_gpu();
                let mut plan = Plan::new(vec![]);
                let mut ctx = TranslationContext::new(
                    &mut plan,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    stream.as_raw(),
                );
                let x = ctx.keep(Array::from_data(
                    input.as_ptr().cast(),
                    &[2, 17, 3],
                    mlx::mlx_dtype__MLX_UINT8,
                ));
                ctx.bind(&output("x", UINT8), x);
                let mut node = NodeDesc::new(
                    "Pack".into(),
                    "".into(),
                    29,
                    CompileShapeSafety::ShapeKeyedOnly {
                        reason: "packed width test",
                    },
                );
                node.inputs = vec![reference("x")];
                node.outputs = vec![output(
                    "packed",
                    if word_bits == 8 { UINT8 } else { UINT32 },
                )];
                let packed = pack_bits_mlx(&mut ctx, &node, bits, 1, word_bits).unwrap();
                let packed = ctx.contiguous(packed).unwrap();
                evaluate(packed);
                if word_bits == 8 {
                    let actual = unsafe {
                        std::slice::from_raw_parts(
                            mlx::mlx_array_data_uint8(packed),
                            expected.len(),
                        )
                    };
                    assert_eq!(
                        actual,
                        expected.iter().map(|&word| word as u8).collect::<Vec<_>>(),
                        "bits={bits}, word_bits={word_bits}"
                    );
                } else {
                    let actual = unsafe {
                        std::slice::from_raw_parts(
                            mlx::mlx_array_data_uint32(packed),
                            expected.len(),
                        )
                    };
                    assert_eq!(
                        actual,
                        expected.iter().map(|&word| word as u32).collect::<Vec<_>>(),
                        "bits={bits}, word_bits={word_bits}"
                    );
                }
                ctx.bind(
                    &output("packed", if word_bits == 8 { UINT8 } else { UINT32 }),
                    packed,
                );
                node.inputs[0] = reference("packed");
                let unpacked = unpack_bits_mlx(&mut ctx, &node, bits, 1, 17).unwrap();
                let unpacked = ctx.contiguous(unpacked).unwrap();
                evaluate(unpacked);
                let actual = unsafe {
                    std::slice::from_raw_parts(mlx::mlx_array_data_uint8(unpacked), input.len())
                };
                assert_eq!(
                    actual, input,
                    "unpack bits={bits}, word_bits={word_bits}, shape={packed_shape:?}"
                );
            }
        }
    }

    #[test]
    fn metal_dequantize_matches_onnx_reference_vectors() {
        for (name, format) in [
            ("gguf:q4_0", PackedFormat::Q4_0),
            ("gguf:q4_1", PackedFormat::Q4_1),
            ("gguf:q5_0", PackedFormat::Q5_0),
            ("gguf:q5_1", PackedFormat::Q5_1),
            ("gguf:q8_0", PackedFormat::Q8_0),
            ("gguf:q8_1", PackedFormat::Q8_1),
            ("gguf:q2_k", PackedFormat::Q2K),
            ("gguf:q3_k", PackedFormat::Q3K),
            ("gguf:q4_k", PackedFormat::Q4K),
            ("gguf:q5_k", PackedFormat::Q5K),
            ("gguf:q6_k", PackedFormat::Q6K),
        ] {
            let (elements, bytes) = format.geometry();
            let block = reference_payload(name, bytes);
            let payload: Vec<_> = block.iter().copied().chain(block.iter().copied()).collect();
            let expected = decode_packed(&payload, format, &[2, elements as i64]).unwrap();
            let stream = Stream::new_default_gpu();
            let mut plan = Plan::new(vec![]);
            let mut ctx = TranslationContext::new(
                &mut plan,
                std::ptr::null(),
                std::ptr::null_mut(),
                stream.as_raw(),
            );
            let x = ctx.keep(Array::from_data(
                payload.as_ptr().cast(),
                &[payload.len() as i32],
                mlx::mlx_dtype__MLX_UINT8,
            ));
            ctx.bind(&output("x", UINT8), x);
            let mut node = NodeDesc::new(
                "DequantizePacked".into(),
                "".into(),
                29,
                CompileShapeSafety::ShapeKeyedOnly {
                    reason: "packed dequantize test",
                },
            );
            node.inputs = vec![reference("x")];
            node.outputs = vec![output("y", FLOAT)];
            let actual = gpu::dequantize(&mut ctx, &node, format, &[2, elements as i64])
                .unwrap()
                .unwrap();
            let actual = ctx.contiguous(actual).unwrap();
            evaluate(actual);
            let actual = unsafe {
                std::slice::from_raw_parts(mlx::mlx_array_data_float32(actual), expected.len())
            };
            assert_eq!(actual, expected, "{name}");
        }
    }
}
