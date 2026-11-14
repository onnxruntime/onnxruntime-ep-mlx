//! Lossless code/parameter repacking for constant Float32 affine-GGUF prefill.

use super::{PackedFormat, gpu};
use crate::engine::{MlxError, NodeDesc, Src, TranslationContext, mlx_dtype_from_onnx};
use crate::mlx::Array;
use crate::sys::mlx;

fn profile(format: PackedFormat) -> Option<(i32, i32)> {
    Some(match format {
        PackedFormat::Q4_0 => (4, 8),
        PackedFormat::Q4_1 | PackedFormat::Q4K => (4, 0),
        PackedFormat::Q5_0 => (5, 16),
        PackedFormat::Q5_1 | PackedFormat::Q5K => (5, 0),
        PackedFormat::Q8_0 | PackedFormat::Q8_1 => (8, 128),
        _ => return None,
    })
}

fn finite_parameters(
    ctx: &TranslationContext,
    node: &NodeDesc,
    format: PackedFormat,
) -> Result<bool, MlxError> {
    let input = ctx.raw_host(&node.inputs[0])?;
    let bytes = unsafe { std::slice::from_raw_parts(input.data.cast::<u8>(), input.count) };
    let affine = matches!(
        format,
        PackedFormat::Q4_1 | PackedFormat::Q5_1 | PackedFormat::Q4K | PackedFormat::Q5K
    );
    Ok(bytes.chunks_exact(format.geometry().1).all(|block| {
        [0, 2]
            .into_iter()
            .take(if affine { 2 } else { 1 })
            .all(|offset| u16::from_le_bytes([block[offset], block[offset + 1]]) & 0x7c00 != 0x7c00)
    }))
}

pub(super) fn execute(
    ctx: &mut TranslationContext,
    node: &NodeDesc,
    format: PackedFormat,
    inputs: [mlx::mlx_array; 2],
    geometry: [i32; 3],
    transpose: bool,
) -> Result<Option<mlx::mlx_array>, MlxError> {
    let Some((bits, center)) = profile(format) else {
        return Ok(None);
    };
    let [m, k, n] = geometry;
    let (rows, width) = if transpose { (n, k) } else { (k, n) };
    let input = &node.inputs[0];
    // Pinned MLX's largest QMV cutoff is 33. Its matrix readers decode weights
    // before multiplication; vector paths may instead scale partial dot products.
    if m < 33
        || width == 0
        || width % 32 != 0
        || !(input.source == Src::Initializer || input.source == Src::CtxInput && input.constant)
        || ctx.dtype_of(inputs[0]) != mlx::mlx_dtype__MLX_FLOAT32
        || mlx_dtype_from_onnx(node.outputs[0].otype) != mlx::mlx_dtype__MLX_FLOAT32
    {
        return Ok(None);
    }
    let groups = width / 32;
    let Some(count) = rows.checked_mul(groups) else {
        return Ok(None);
    };
    let Some(words) = count.checked_mul(bits) else {
        return Ok(None);
    };
    let prefix = format!("{}#gguf-qmm:{format:?}:{rows}:{width}", input.cache_key());
    if ctx.cache_get(&format!("{prefix}:nonfinite")).is_some() {
        return Ok(None);
    }
    let keys = [
        format!("{prefix}:words"),
        format!("{prefix}:scale"),
        format!("{prefix}:bias"),
    ];
    let cached = keys.each_ref().map(|key| ctx.cache_get(key));
    let [w, scale, bias] = if let [Some(w), Some(scale), Some(bias)] = cached {
        [w, scale, bias]
    } else {
        if !finite_parameters(ctx, node, format)? {
            let marker = [0i32];
            ctx.cache_put(
                format!("{prefix}:nonfinite"),
                Array::from_data(marker.as_ptr().cast(), &[1], mlx::mlx_dtype__MLX_INT32),
            );
            return Ok(None);
        }
        let (block, bytes) = format.geometry();
        let templates = [
            ("FORMAT", format.gpu_id().expect("affine GPU format")),
            ("BLOCK", block as i32),
            ("BYTES", bytes as i32),
            ("BITS", bits),
            ("CENTER", center),
            ("WORDS", words),
            ("GROUPS", count),
        ];
        let w = gpu::Kernel {
            name: c"mlx_packed_repack_words",
            source: include_str!("repack_words.metal"),
            header: gpu::DECODE,
            templates: &templates,
            shape: &[rows, groups * bits],
            dtype: mlx::mlx_dtype__MLX_UINT32,
            grid: [words, 1, 1],
            group: [words.min(256), 1, 1],
        }
        .run(ctx, &[("packed", inputs[1])])?;
        let params = gpu::Kernel {
            name: c"mlx_packed_repack_parameters",
            source: include_str!("repack_parameters.metal"),
            header: gpu::DECODE,
            templates: &templates,
            shape: &[2, rows, groups],
            dtype: mlx::mlx_dtype__MLX_FLOAT32,
            grid: [count, 1, 1],
            group: [count.min(256), 1, 1],
        }
        .run(ctx, &[("packed", inputs[1])])?;
        let scale = ctx.slice(params, &[0, 0, 0], &[1, rows, groups])?;
        let bias = ctx.slice(params, &[1, 0, 0], &[2, rows, groups])?;
        let scale = ctx.reshape(scale, &[rows, groups])?;
        let bias = ctx.reshape(bias, &[rows, groups])?;
        [
            ctx.cache_contiguous(keys[0].clone(), w)?,
            ctx.cache_contiguous(keys[1].clone(), scale)?,
            ctx.cache_contiguous(keys[2].clone(), bias)?,
        ]
    };
    // Half scales times <=63 subscales and <=31 codes need at most 22 significant
    // bits. The native Float32 tile reader reconstructs the same rounded weights.
    let x = ctx.reshape(inputs[0], &[m, k])?;
    let y = super::super::quant::quantized_matmul(
        ctx,
        x,
        w,
        scale,
        bias,
        [32, i64::from(bits)],
        transpose,
    )?;
    ctx.mark_fast("mlx_packed_quantized_matmul");
    Ok(Some(y))
}
