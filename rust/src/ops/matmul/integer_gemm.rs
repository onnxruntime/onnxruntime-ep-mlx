use super::present;
use crate::engine::{MlxError, NodeDesc, TranslationContext, dim_i32};
use crate::ops::host_scan::{layout, read_live};
use crate::sys::{mlx, ort};

pub(super) fn integer_gemm_dtype(dtype: ort::ONNXTensorElementDataType) -> bool {
    matches!(
        dtype,
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
            | ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
            | ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
            | ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
    )
}

pub(super) fn integer_gemm_scale(value: f32, name: &str) -> Result<i128, MlxError> {
    let limit = 2f32.powi(127);
    if !value.is_finite() || value.fract() != 0.0 || value < -limit || value >= limit {
        return Err(format!(
            "Gemm integer {name}={value} is unsupported; use a finite integral coefficient in [-2^127,2^127), or cast the operands to floating point for fractional scaling"
        ));
    }
    Ok(value as i128)
}

fn read_input(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    index: usize,
) -> Result<(Vec<i64>, mlx::mlx_dtype, Vec<i128>), MlxError> {
    let input = read_live(ctx, &n.inputs[index], "integer Gemm")?;
    let width = match input.dtype {
        mlx::mlx_dtype__MLX_INT32 | mlx::mlx_dtype__MLX_UINT32 => 4,
        mlx::mlx_dtype__MLX_INT64 | mlx::mlx_dtype__MLX_UINT64 => 8,
        _ => {
            return Err(format!(
                "Gemm input {index} must be int32/uint32/int64/uint64; supply matching integer tensors"
            ));
        }
    };
    let values = input
        .data
        .chunks_exact(width)
        .map(|value| match input.dtype {
            mlx::mlx_dtype__MLX_INT32 => i128::from(i32::from_ne_bytes(value.try_into().unwrap())),
            mlx::mlx_dtype__MLX_UINT32 => i128::from(u32::from_ne_bytes(value.try_into().unwrap())),
            mlx::mlx_dtype__MLX_INT64 => i128::from(i64::from_ne_bytes(value.try_into().unwrap())),
            _ => i128::from(u64::from_ne_bytes(value.try_into().unwrap())),
        })
        .collect();
    Ok((input.shape, input.dtype, values))
}

struct Geometry {
    shape: Vec<i64>,
    row_strides: Vec<usize>,
    reduction_stride: usize,
    rows: usize,
    k: usize,
    width: usize,
    trans_b: bool,
    bytes: usize,
}

fn geometry(
    a: &[i64],
    b: &[i64],
    c: Option<&[i64]>,
    trans_a: bool,
    trans_b: bool,
    dtype: mlx::mlx_dtype,
) -> Result<Geometry, MlxError> {
    if a.is_empty() || b.len() != 2 {
        return Err("Gemm integer inputs require rank >=1 A and rank-2 B with the same dtype; reshape or cast the operands".into());
    }
    let mut shape = a.to_vec();
    let mut strides = vec![1usize; shape.len()];
    for dim in (0..shape.len().saturating_sub(1)).rev() {
        strides[dim] = if a.contains(&0) {
            0
        } else {
            strides[dim + 1]
                .checked_mul(shape[dim + 1] as usize)
                .ok_or(
                    "Gemm integer strides exceed addressable memory; reduce the input dimensions",
                )?
        };
    }
    if trans_a && shape.len() >= 2 {
        let rank = shape.len();
        shape.swap(rank - 2, rank - 1);
        strides.swap(rank - 2, rank - 1);
    }
    let (k, width) = if trans_b { (b[1], b[0]) } else { (b[0], b[1]) };
    if shape.last() != Some(&k) {
        return Err(format!(
            "Gemm integer contraction dimensions differ: A {a:?}, B {b:?}; correct transA/transB or reshape the operands"
        ));
    }
    let reduction_stride = strides.pop().expect("checked non-scalar A");
    *shape.last_mut().expect("checked non-scalar A") = width;
    let (bytes, _) = layout(&shape, dtype, "integer Gemm")?;
    if let Some(c) = c
        && (c.len() > shape.len()
            || !c
                .iter()
                .rev()
                .zip(shape.iter().rev())
                .all(|(&c, &y)| c == 1 || c == y))
    {
        return Err(format!(
            "Gemm integer C shape {c:?} cannot broadcast to {shape:?}; reshape C to a suffix compatible with Y"
        ));
    }
    let rows = if shape.contains(&0) {
        0
    } else {
        shape[..shape.len() - 1]
            .iter()
            .try_fold(1usize, |total, &dim| total.checked_mul(dim as usize))
            .ok_or("Gemm integer output size overflow; reduce the input dimensions")?
    };
    Ok(Geometry {
        shape,
        row_strides: strides,
        reduction_stride,
        rows,
        k: k as usize,
        width: width as usize,
        trans_b,
        bytes,
    })
}

fn row_offset(row: usize, shape: &[i64], strides: &[usize]) -> usize {
    let mut coordinate = row;
    let mut offset = 0;
    for dim in (0..strides.len()).rev() {
        offset += coordinate % shape[dim] as usize * strides[dim];
        coordinate /= shape[dim] as usize;
    }
    offset
}

fn bias_offset(mut index: usize, output: &[i64], bias: &[i64]) -> usize {
    let mut offset = 0;
    let mut stride = 1;
    for dim in (0..output.len()).rev() {
        let coordinate = index % output[dim] as usize;
        index /= output[dim] as usize;
        if dim >= output.len() - bias.len() {
            let size = bias[dim - (output.len() - bias.len())];
            if size != 1 {
                offset += coordinate * stride;
            }
            stride *= size as usize;
        }
    }
    offset
}

fn overflow() -> MlxError {
    "Gemm integer arithmetic exceeds the checked i128 accumulator or output dtype; reduce operand magnitudes/scales or use a wider floating-point representation".into()
}

fn append_value(result: &mut Vec<u8>, value: i128, dtype: mlx::mlx_dtype) -> Result<(), MlxError> {
    match dtype {
        mlx::mlx_dtype__MLX_INT32 => {
            result.extend_from_slice(&i32::try_from(value).map_err(|_| overflow())?.to_ne_bytes())
        }
        mlx::mlx_dtype__MLX_UINT32 => {
            result.extend_from_slice(&u32::try_from(value).map_err(|_| overflow())?.to_ne_bytes())
        }
        mlx::mlx_dtype__MLX_INT64 => {
            result.extend_from_slice(&i64::try_from(value).map_err(|_| overflow())?.to_ne_bytes())
        }
        _ => result.extend_from_slice(&u64::try_from(value).map_err(|_| overflow())?.to_ne_bytes()),
    }
    Ok(())
}

fn product(
    g: &Geometry,
    a: &[i128],
    b: &[i128],
    c: Option<(&[i128], &[i64])>,
    coefficients: (i128, i128),
    dtype: mlx::mlx_dtype,
) -> Result<Vec<u8>, MlxError> {
    let (alpha, beta) = coefficients;
    let mut result = Vec::with_capacity(g.bytes);
    for row in 0..g.rows {
        let a_offset = row_offset(row, &g.shape, &g.row_strides);
        for column in 0..g.width {
            let mut sum = 0i128;
            if alpha != 0 {
                for j in 0..g.k {
                    let b_offset = if g.trans_b {
                        column * g.k + j
                    } else {
                        j * g.width + column
                    };
                    sum = a[a_offset + j * g.reduction_stride]
                        .checked_mul(b[b_offset])
                        .and_then(|value| sum.checked_add(value))
                        .ok_or_else(overflow)?;
                }
                sum = sum.checked_mul(alpha).ok_or_else(overflow)?;
            }
            if let Some((c, c_shape)) = c {
                let offset = bias_offset(row * g.width + column, &g.shape, c_shape);
                sum = c[offset]
                    .checked_mul(beta)
                    .and_then(|value| sum.checked_add(value))
                    .ok_or_else(overflow)?;
            }
            append_value(&mut result, sum, dtype)?;
        }
    }
    Ok(result)
}

pub(super) fn integer_gemm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    if n.since_version < 29 {
        return Err("Integer Gemm requires the version-29 handler; use floating-point Gemm for older schemas".into());
    }
    let alpha = integer_gemm_scale(n.floats.get("alpha").copied().unwrap_or(1.0), "alpha")?;
    let beta = integer_gemm_scale(n.floats.get("beta").copied().unwrap_or(1.0), "beta")?;
    let (a_shape, dtype, a) = read_input(ctx, n, 0)?;
    let (b_shape, b_dtype, b) = read_input(ctx, n, 1)?;
    if b_dtype != dtype {
        return Err("Gemm integer inputs require rank >=1 A and rank-2 B with the same dtype; reshape or cast the operands".into());
    }
    let c_shape = if present(n, 2) {
        Some(ctx.tensor_shape(&n.inputs[2])?)
    } else {
        None
    };
    let g = geometry(
        &a_shape,
        &b_shape,
        c_shape.as_deref(),
        n.ints.get("transA").copied().unwrap_or(0) != 0,
        n.ints.get("transB").copied().unwrap_or(0) != 0,
        dtype,
    )?;
    let output_shape = g
        .shape
        .iter()
        .copied()
        .map(dim_i32)
        .collect::<Result<Vec<_>, _>>()?;
    if g.rows == 0 {
        let y = ctx.zeros(&output_shape, dtype)?;
        ctx.bind(&n.outputs[0], y);
        return Ok(());
    }
    let c = if present(n, 2) && beta != 0 {
        let (_, c_dtype, values) = read_input(ctx, n, 2)?;
        if c_dtype != dtype {
            return Err(
                "Gemm integer C dtype differs from A/B; cast all operands to the same integer type"
                    .into(),
            );
        }
        Some(values)
    } else {
        None
    };
    let result = product(
        &g,
        &a,
        &b,
        c.as_deref().zip(c_shape.as_deref()),
        (alpha, beta),
        dtype,
    )?;
    let y = ctx.from_host(result.as_ptr().cast(), &output_shape, dtype);
    ctx.mark_composed("checked exact host integer Gemm (MLX has no integer GEMM primitive)");
    ctx.bind(&n.outputs[0], y);
    Ok(())
}
