use super::{DOUBLE, PackedFormat, dequantize_input};
use crate::engine::{MlxError, NodeDesc, TranslationContext, mlx_dtype_from_onnx};
use crate::registry::NodeView;
use crate::sys::mlx;

pub(super) use crate::mlx::metal::{Kernel, is_gpu};
pub(super) const DECODE: &str = include_str!("decode.metal");

pub(super) fn has_decoder(node: &NodeView) -> bool {
    node.output_info(0).is_some_and(|output| {
        matches!(
            output.dtype,
            super::FLOAT | super::FLOAT16 | super::BFLOAT16
        )
    }) && PackedFormat::parse(&node.string_attr("format", ""))
        .is_ok_and(|format| format.gpu_id().is_some())
}

impl PackedFormat {
    pub(super) fn gpu_id(self) -> Option<i32> {
        Some(match self {
            Self::Q4_0 => 0,
            Self::Q4_1 => 1,
            Self::Q5_0 => 2,
            Self::Q5_1 => 3,
            Self::Q8_0 => 4,
            Self::Q8_1 => 5,
            Self::Q2K => 6,
            Self::Q3K => 7,
            Self::Q4K => 8,
            Self::Q5K => 9,
            Self::Q6K => 10,
            _ => return None,
        })
    }
}

pub(super) fn dequantize(
    ctx: &mut TranslationContext,
    node: &NodeDesc,
    format: PackedFormat,
    shape: &[i64],
) -> Result<Option<mlx::mlx_array>, MlxError> {
    let Some(format_id) = format.gpu_id() else {
        return Ok(None);
    };
    if node.outputs[0].otype == DOUBLE || !is_gpu(ctx)? {
        return Ok(None);
    }
    let (input, shape, count) = dequantize_input(ctx, node, format, shape)?;
    let dtype = mlx_dtype_from_onnx(node.outputs[0].otype);
    if count == 0 {
        return ctx.zeros(&shape, dtype).map(Some);
    }
    let width = count.min(65536) as i32;
    let Ok(height) = i32::try_from(count.div_ceil(width as usize)) else {
        return Ok(None);
    };
    let (block, bytes) = format.geometry();
    let result = Kernel {
        name: c"mlx_packed_dequantize",
        source: include_str!("dequantize.metal"),
        header: DECODE,
        templates: &[
            ("FORMAT", format_id),
            ("BLOCK", block as i32),
            ("BYTES", bytes as i32),
            ("GRID_X", width),
        ],
        shape: &shape,
        dtype,
        grid: [width, height, 1],
        group: [256.min(width), 1, 1],
    }
    .run(ctx, &[("packed", input)])?;
    ctx.mark_fast("mlx_packed_dequantize");
    Ok(Some(result))
}
