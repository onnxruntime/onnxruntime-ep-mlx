//! Integer attention-cache storage, pinned to ORT v1.31 MLAS GQA and CUDA PagedAttention.

use crate::engine::{MlxError, NodeDesc, TranslationContext};
use crate::registry::{ClaimResult, NodeView};
use crate::sys::{mlx, ort};
use crate::{deny, require};

pub(super) fn integer_cache(dtype: ort::ONNXTensorElementDataType) -> bool {
    matches!(
        dtype,
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
            | ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
    )
}

pub(super) fn claim(
    node: &NodeView,
    paged: bool,
    activation: ort::ONNXTensorElementDataType,
    heads: i64,
    head: i64,
) -> ClaimResult {
    let first_scale = if paged { 14 } else { 12 };
    let cache_type = node.input_info(3).map(|v| v.dtype).unwrap_or(activation);
    let quantized = integer_cache(cache_type);
    for (cache, scale, side) in [(3, first_scale, "k"), (4, first_scale + 1, "v")] {
        let dtype = node
            .input_info(cache)
            .map(|v| v.dtype)
            .unwrap_or(activation);
        require!(
            dtype == cache_type,
            "key/value caches must share one storage dtype"
        );
        let mode = node.string_attr(&format!("{side}_quant_type"), "NONE");
        if quantized {
            require!(
                matches!(mode.as_str(), "PER_TENSOR" | "PER_CHANNEL"),
                "integer {side} cache requires PER_TENSOR or PER_CHANNEL quantization"
            );
            let Some(info) = node.input_info(scale) else {
                deny!("quantized {side} cache requires a scale tensor");
            };
            require!(
                info.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "{side}_scale must be float32"
            );
            let count = info
                .shape
                .iter()
                .try_fold(1_i64, |n, &d| if d < 0 { None } else { n.checked_mul(d) });
            let expected = if mode == "PER_TENSOR" {
                1
            } else {
                heads * head
            };
            require!(
                count.is_none_or(|count| count == expected),
                "{side}_scale must contain {expected} elements"
            );
            if paged && mode == "PER_CHANNEL" {
                require!(
                    info.shape.last().is_some_and(|&d| d < 0 || d == head),
                    "PagedAttention per-channel scales must end in head_size"
                );
            }
        } else {
            require!(
                dtype == activation && mode == "NONE" && !node.input_present(scale),
                "floating {side} cache requires activation dtype, NONE quantization and no scale"
            );
        }
        if paged {
            let logical = node.string_attr(&format!("{side}_cache_dtype"), "");
            let expected = match dtype {
                ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => "int8",
                ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => "int4",
                ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => "float16",
                ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => "bfloat16",
                _ => "",
            };
            require!(
                logical == expected || (logical.is_empty() && expected != "int4"),
                "{side}_cache_dtype must agree with cache storage; uint8 requires explicit int4"
            );
        }
        if node.output_present(cache - 2) {
            require!(
                node.output_info(cache - 2)
                    .is_some_and(|v| v.dtype == dtype),
                "present {side} cache must retain its input storage dtype"
            );
        }
    }
    if !paged {
        let bits = node.int_attr("kv_cache_bit_width", 0);
        let expected = match cache_type {
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => 8,
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => 4,
            _ => 0,
        };
        require!(
            bits == expected,
            "kv_cache_bit_width must agree with cache storage"
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct QuantizedCache {
    scale: mlx::mlx_array,
    pub width: i32,
    packed: bool,
    paged: bool,
}

impl QuantizedCache {
    pub fn resolve(
        ctx: &mut TranslationContext,
        n: &NodeDesc,
        side: usize,
        paged: bool,
        heads: i32,
        head: i32,
    ) -> Result<Option<Self>, MlxError> {
        let cache = ctx.resolve(&n.inputs[3 + side])?;
        let dtype = ctx.dtype_of(cache);
        if !matches!(dtype, mlx::mlx_dtype__MLX_INT8 | mlx::mlx_dtype__MLX_UINT8) {
            return Ok(None);
        }
        let scale = ctx.resolve(&n.inputs[if paged { 14 } else { 12 } + side])?;
        let shape = ctx.shape_of(scale);
        let mode = n.strings.get(if side == 0 {
            "k_quant_type"
        } else {
            "v_quant_type"
        });
        let per_channel = mode.is_some_and(|v| v == "PER_CHANNEL");
        let count: i64 = shape.iter().map(|&v| i64::from(v)).product();
        if count
            != if per_channel {
                i64::from(heads) * i64::from(head)
            } else {
                1
            }
            || (paged && per_channel && shape.last() != Some(&head))
        {
            return Err("quantized KV cache: invalid live scale shape".into());
        }
        let scale = if per_channel {
            let shape = if paged {
                [1, 1, heads, head]
            } else {
                [1, heads, 1, head]
            };
            ctx.reshape(scale, &shape)?
        } else {
            ctx.reshape(scale, &[])?
        };
        let packed = dtype == mlx::mlx_dtype__MLX_UINT8;
        Ok(Some(Self {
            scale,
            width: if packed { (head + 1) / 2 } else { head },
            packed,
            paged,
        }))
    }

    pub fn encode(
        self,
        ctx: &mut TranslationContext,
        input: mlx::mlx_array,
    ) -> Result<mlx::mlx_array, MlxError> {
        let input = ctx.astype(input, mlx::mlx_dtype__MLX_FLOAT32)?;
        let zero = ctx.scalar_f32(0.0);
        let one = ctx.scalar_f32(1.0);
        let is_zero = ctx.binary(mlx::mlx_equal, self.scale, zero)?;
        let divisor = ctx.where_(is_zero, one, self.scale)?;
        // MLAS GQA multiplies by an FP32 reciprocal (zero -> 1); Paged CUDA divides
        // (zero -> 0). Keeping these distinct matters at rounding boundaries.
        let scaled = if self.paged {
            let scaled = ctx.binary(mlx::mlx_divide, input, divisor)?;
            ctx.where_(is_zero, zero, scaled)?
        } else {
            let reciprocal = ctx.binary(mlx::mlx_divide, one, divisor)?;
            ctx.mul(input, reciprocal)?
        };
        let lo = ctx.scalar_f32(if self.packed { -8.0 } else { -128.0 });
        let hi = ctx.scalar_f32(if self.packed { 7.0 } else { 127.0 });
        let clipped = ctx.emit(|r, s| unsafe { mlx::mlx_clip(r, scaled, lo, hi, s) })?;
        let rounded = ctx.emit(|r, s| unsafe { mlx::mlx_round(r, clipped, 0, s) })?;
        if !self.packed {
            return ctx.astype(rounded, mlx::mlx_dtype__MLX_INT8);
        }
        let bias = ctx.scalar_f32(8.0);
        let biased = ctx.add(rounded, bias)?;
        let mut shape = ctx.shape_of(biased);
        let head = shape[3];
        let biased = if head % 2 != 0 {
            let mut padding_shape = shape.clone();
            padding_shape[3] = 1;
            let padding = ctx.zeros(&padding_shape, mlx::mlx_dtype__MLX_FLOAT32)?;
            let padding = ctx.add(padding, bias)?;
            ctx.concat2(biased, padding, 3)?
        } else {
            biased
        };
        shape[3] = (head + 1) / 2;
        shape.push(2);
        let pairs = ctx.reshape(biased, &shape)?;
        let low = ctx.scalar_i32(0);
        let high = ctx.scalar_i32(1);
        let low = ctx.emit(|r, s| unsafe { mlx::mlx_take_axis(r, pairs, low, 4, s) })?;
        let high = ctx.emit(|r, s| unsafe { mlx::mlx_take_axis(r, pairs, high, 4, s) })?;
        let sixteen = ctx.scalar_f32(16.0);
        let high = ctx.mul(high, sixteen)?;
        let packed = ctx.add(low, high)?;
        ctx.astype(packed, mlx::mlx_dtype__MLX_UINT8)
    }

    pub fn decode(
        self,
        ctx: &mut TranslationContext,
        input: mlx::mlx_array,
        head: i32,
    ) -> Result<mlx::mlx_array, MlxError> {
        let values = if self.packed {
            let input = ctx.astype(input, mlx::mlx_dtype__MLX_INT32)?;
            let mask = ctx.scalar_i32(15);
            let four = ctx.scalar_i32(4);
            let low = ctx.binary(mlx::mlx_bitwise_and, input, mask)?;
            let high = ctx.binary(mlx::mlx_right_shift, input, four)?;
            let mut shape = ctx.shape_of(input);
            shape.push(1);
            let low = ctx.reshape(low, &shape)?;
            let high = ctx.reshape(high, &shape)?;
            let values = ctx.concat2(low, high, 4)?;
            shape.pop();
            shape[3] *= 2;
            let values = ctx.reshape(values, &shape)?;
            let values = if shape[3] != head {
                let start = vec![0; 4];
                shape[3] = head;
                ctx.slice(values, &start, &shape)?
            } else {
                values
            };
            let values = ctx.astype(values, mlx::mlx_dtype__MLX_FLOAT32)?;
            let bias = ctx.scalar_f32(8.0);
            ctx.binary(mlx::mlx_subtract, values, bias)?
        } else {
            ctx.astype(input, mlx::mlx_dtype__MLX_FLOAT32)?
        };
        ctx.mul(values, self.scale)
    }
}
