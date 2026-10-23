//! Checked exact live payload access for eager, provider-owned host scans.

use crate::engine::{MlxError, Src, TensorRef, TranslationContext, dim_i32, mlx_dtype_from_onnx};
use crate::registry::is_movable;
use crate::sys::mlx;

pub(super) fn layout(
    shape: &[i64],
    dtype: mlx::mlx_dtype,
    op: &str,
) -> Result<(usize, usize), MlxError> {
    let width = match dtype {
        mlx::mlx_dtype__MLX_BOOL | mlx::mlx_dtype__MLX_INT8 | mlx::mlx_dtype__MLX_UINT8 => 1,
        mlx::mlx_dtype__MLX_INT16
        | mlx::mlx_dtype__MLX_UINT16
        | mlx::mlx_dtype__MLX_FLOAT16
        | mlx::mlx_dtype__MLX_BFLOAT16 => 2,
        mlx::mlx_dtype__MLX_INT32 | mlx::mlx_dtype__MLX_UINT32 | mlx::mlx_dtype__MLX_FLOAT32 => 4,
        mlx::mlx_dtype__MLX_INT64 | mlx::mlx_dtype__MLX_UINT64 | mlx::mlx_dtype__MLX_FLOAT64 => 8,
        _ => return Err(format!("MLX {op}: expected a real numeric or bool tensor")),
    };
    for &extent in shape {
        if extent < 0 {
            return Err(format!("MLX {op}: negative runtime extent"));
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
                .ok_or_else(|| format!("MLX {op}: input element count exceeds int32 indexing"))
        })?
    };
    let bytes = count
        .checked_mul(width)
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| format!("MLX {op}: input byte count overflow"))?;
    Ok((bytes, width))
}

pub(super) struct HostTensor<'a> {
    pub data: &'a [u8],
    pub shape: Vec<i64>,
    pub dtype: mlx::mlx_dtype,
}

pub(super) fn read_live<'a>(
    ctx: &'a mut TranslationContext,
    input: &'a TensorRef,
    op: &str,
) -> Result<HostTensor<'a>, MlxError> {
    if ctx.shapeless_compile() || ctx.shape_keyed_compile() {
        return Err(format!(
            "MLX {op}: data-dependent cardinality requires eager execution"
        ));
    }
    let (data, shape, dtype) =
        if input.source == Src::Intermediate || ctx.env_get(&input.name).is_some() {
            let array = ctx.resolve(input)?;
            let shape: Vec<_> = ctx.shape_of(array).into_iter().map(i64::from).collect();
            layout(&shape, ctx.dtype_of(array), op)?;
            let array = ctx.contiguous_eval(array)?;
            (ctx.host_ptr(array), shape, ctx.dtype_of(array))
        } else {
            // Validate before raw_host computes its element count or MLX materializes the input.
            let shape = ctx.tensor_shape(input)?;
            layout(&shape, mlx::mlx_dtype__MLX_UINT8, op)?;
            let host = ctx.raw_host(input)?;
            if !is_movable(host.dtype) {
                return Err(format!("MLX {op}: live input is not real numeric or bool"));
            }
            let dtype = mlx_dtype_from_onnx(host.dtype);
            let (bytes, width) = layout(&host.shape, dtype, op)?;
            if host.count != bytes / width {
                return Err(format!(
                    "MLX {op}: element count differs from runtime shape"
                ));
            }
            (host.data.cast::<u8>(), host.shape, dtype)
        };
    let (bytes, _) = layout(&shape, dtype, op)?;
    let data = if bytes == 0 {
        &[]
    } else {
        if data.is_null() {
            return Err(format!("MLX {op}: nonempty input has null data"));
        }
        // ORT buffers live through Compute; evaluated arrays are retained in ctx's arena.
        unsafe { std::slice::from_raw_parts(data, bytes) }
    };
    Ok(HostTensor { data, shape, dtype })
}
