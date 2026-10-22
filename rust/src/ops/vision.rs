//! Vision / spatial-transform op handlers. Faithful port of the C++ `ops/vision2.cc`
//! (GridSample, AffineGrid, Col2Im) and `ops/vision2b.cc` (RoiAlign, MaxRoiPool, MaxUnpool).
//!
//! Every op reduces to the same primitives the shape handlers already use — host-computed
//! gather/scatter index maps + `take`/`take_along_axis`/`scatter_add` + on-device coordinate
//! arithmetic — so they translate exactly to MLX with no bespoke kernel. ROI pooling and MaxUnpool
//! accept symbolic extents; host-read MaxUnpool controls stay eager unless constant. Unsupported
//! forms (cubic GridSample, 5-D volumetric, adaptive RoiAlign `sampling_ratio==0`, non-float
//! payloads) are left to CPU.
//! NonMaxSuppression is host-computed because MLX has no data-dependent compacting primitive.

use std::cmp::Ordering;
use std::os::raw::c_void;

use crate::engine::{MlxError, NodeDesc, TensorRef, TranslationContext, mlx_dtype_from_onnx};
use crate::mlx::{Array, VectorArray};
use crate::registry::{
    ClaimResult, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry, is_int_index, is_mlx_cpu_float,
    is_mlx_float,
};
use crate::sys::mlx;
use crate::sys::ort;
use crate::{deny, require};

// ---- translate-time helpers ---------------------------------------------------------------------

fn astype(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    t: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.astype(a, t)
}
fn reshape(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    shape: &[i32],
) -> Result<mlx::mlx_array, MlxError> {
    ctx.reshape(a, shape)
}
fn contiguous(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.contiguous(a)
}
fn add(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_add, a, b)
}
fn sub(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_subtract, a, b)
}
fn mul(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_multiply, a, b)
}
fn div(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_divide, a, b)
}
fn maximum(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_maximum, a, b)
}
fn floor(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.unary(mlx::mlx_floor, a)
}
fn ceil(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.unary(mlx::mlx_ceil, a)
}

fn input_present(n: &NodeDesc, index: usize) -> bool {
    n.inputs
        .get(index)
        .is_some_and(|input| !input.name.is_empty())
}

fn eval_f32(ctx: &mut TranslationContext, input: &TensorRef) -> Result<mlx::mlx_array, MlxError> {
    let value = ctx.resolve(input)?;
    ctx.contiguous_eval(value)
}

fn scalar_f32_input(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    index: usize,
    default: f32,
) -> Result<f32, MlxError> {
    if !input_present(n, index) {
        return Ok(default);
    }
    let value = eval_f32(ctx, &n.inputs[index])?;
    if ctx.size_of(value) != 1 {
        return Err(format!("input {index} must be a scalar"));
    }
    Ok(unsafe { *mlx::mlx_array_data_float32(value) })
}

fn scalar_i64_input(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    index: usize,
    default: i64,
) -> Result<i64, MlxError> {
    if !input_present(n, index) {
        return Ok(default);
    }
    let value = ctx.resolve(&n.inputs[index])?;
    let value = ctx.contiguous_eval(value)?;
    if ctx.size_of(value) != 1 {
        return Err(format!("input {index} must be a scalar"));
    }
    Ok(unsafe { *mlx::mlx_array_data_int64(value) })
}
fn round0(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_round(res, a, 0, s) })
}
fn sign(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.unary(mlx::mlx_sign, a)
}
fn abs_(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.unary(mlx::mlx_abs, a)
}

fn scalar_f(ctx: &mut TranslationContext, v: f32) -> mlx::mlx_array {
    ctx.scalar_f32(v)
}
fn scalar_i(ctx: &mut TranslationContext, v: i32) -> mlx::mlx_array {
    ctx.scalar_i32(v)
}

fn host_f32(ctx: &mut TranslationContext, data: &[f32], shape: &[i32]) -> mlx::mlx_array {
    ctx.keep(Array::from_data(
        data.as_ptr() as *const c_void,
        shape,
        mlx::mlx_dtype__MLX_FLOAT32,
    ))
}
fn host_i32(ctx: &mut TranslationContext, data: &[i32], shape: &[i32]) -> mlx::mlx_array {
    ctx.keep(Array::from_data(
        data.as_ptr() as *const c_void,
        shape,
        mlx::mlx_dtype__MLX_INT32,
    ))
}

/// Clamp `a` to [lo, hi] (scalar bounds).
fn clip(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    lo: f32,
    hi: f32,
) -> Result<mlx::mlx_array, MlxError> {
    let los = scalar_f(ctx, lo);
    let his = scalar_f(ctx, hi);
    ctx.emit(|res, s| unsafe { mlx::mlx_clip(res, a, los, his, s) })
}

/// Round half away from zero: sign(a)*floor(|a|+0.5) (matches std::round the ORT CPU kernels use).
fn round_away(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    let sgn = sign(ctx, a)?;
    let mag = abs_(ctx, a)?;
    let half = scalar_f(ctx, 0.5);
    let shifted = add(ctx, mag, half)?;
    let fl = floor(ctx, shifted)?;
    mul(ctx, sgn, fl)
}

/// 1.0 where lo <= a <= hi else 0.0 (float32) — the zeros-padding validity mask for a coordinate.
fn in_range_mask(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    lo: f32,
    hi: f32,
) -> Result<mlx::mlx_array, MlxError> {
    let los = scalar_f(ctx, lo);
    let his = scalar_f(ctx, hi);
    let ge = ctx.binary(mlx::mlx_greater_equal, a, los)?;
    let le = ctx.binary(mlx::mlx_less_equal, a, his)?;
    let both = ctx.binary(mlx::mlx_logical_and, ge, le)?;
    astype(ctx, both, mlx::mlx_dtype__MLX_FLOAT32)
}

fn broadcast_to(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    shape: &[i32],
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_broadcast_to(res, a, shape.as_ptr(), shape.len(), s) })
}
fn take_along_axis(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    idx: mlx::mlx_array,
    axis: i32,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_take_along_axis(res, a, idx, axis, s) })
}
fn take_axis(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    idx: mlx::mlx_array,
    axis: i32,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_take_axis(res, a, idx, axis, s) })
}
fn matmul(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_matmul(res, a, b, s) })
}
fn slice(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    start: &[i32],
    stop: &[i32],
) -> Result<mlx::mlx_array, MlxError> {
    let stride = vec![1i32; start.len()];
    ctx.emit(|res, s| unsafe {
        mlx::mlx_slice(
            res,
            a,
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            stride.as_ptr(),
            stride.len(),
            s,
        )
    })
}
fn max_axis(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    axis: i32,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_max_axis(res, a, axis, false, s) })
}
fn sum_axis(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    axis: i32,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_sum_axis(res, a, axis, false, s) })
}
fn where_(
    ctx: &mut TranslationContext,
    c: mlx::mlx_array,
    x: mlx::mlx_array,
    y: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_where(res, c, x, y, s) })
}
fn arange(ctx: &mut TranslationContext, nval: i32) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe {
        mlx::mlx_arange(res, 0.0, nval as f64, 1.0, mlx::mlx_dtype__MLX_FLOAT32, s)
    })
}

/// `mlx_scatter_add(a, [idx], updates, axes={0})` — RAII vector-array wrapper.
fn scatter_add(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    idx: mlx::mlx_array,
    updates: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    let mut vec = VectorArray::new();
    vec.append(idx);
    let vraw = vec.as_raw();
    let axes0 = [0i32];
    ctx.emit(|res, s| unsafe { mlx::mlx_scatter_add(res, a, vraw, updates, axes0.as_ptr(), 1, s) })
}

fn zeros_1d(ctx: &mut TranslationContext, n: i32) -> Result<mlx::mlx_array, MlxError> {
    let shape = [n];
    ctx.emit(|res, s| unsafe {
        mlx::mlx_zeros(res, shape.as_ptr(), 1, mlx::mlx_dtype__MLX_FLOAT32, s)
    })
}

/// Read a constant int64/int32 parameter INPUT (image_shape / block_shape / size) at translate time.
fn read_const_ints(ctx: &TranslationContext, r: &TensorRef) -> Result<Vec<i64>, MlxError> {
    let h = ctx.raw_host(r)?;
    if h.data.is_null() {
        return Ok(Vec::new());
    }
    if h.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 {
        let p = h.data as *const i64;
        Ok(unsafe { std::slice::from_raw_parts(p, h.count) }.to_vec())
    } else {
        let p = h.data as *const i32;
        Ok(unsafe { std::slice::from_raw_parts(p, h.count) }
            .iter()
            .map(|&v| v as i64)
            .collect())
    }
}

fn out_dtype(n: &NodeDesc) -> mlx::mlx_dtype {
    mlx_dtype_from_onnx(n.outputs[0].otype)
}

// =================================================================================================
// GridSample (2D)
// =================================================================================================

#[allow(clippy::too_many_arguments)]
fn gs_sample(
    ctx: &mut TranslationContext,
    xf: mlx::mlx_array,
    xf_coord: mlx::mlx_array,
    yf_coord: mlx::mlx_array,
    w: mlx::mlx_array,
    zeros_pad: bool,
    dims: (i32, i32, i32, i32, i32), // N, C, P, W, H
) -> Result<mlx::mlx_array, MlxError> {
    let (n, c, p, w_dim, h_dim) = dims;
    let mut weight = w;
    if zeros_pad {
        let mx = in_range_mask(ctx, xf_coord, 0.0, w_dim as f32 - 1.0)?;
        weight = mul(ctx, weight, mx)?;
        let my = in_range_mask(ctx, yf_coord, 0.0, h_dim as f32 - 1.0)?;
        weight = mul(ctx, weight, my)?;
    }
    let xc = clip(ctx, xf_coord, 0.0, w_dim as f32 - 1.0)?;
    let xi = astype(ctx, xc, mlx::mlx_dtype__MLX_INT32)?;
    let yc = clip(ctx, yf_coord, 0.0, h_dim as f32 - 1.0)?;
    let yi = astype(ctx, yc, mlx::mlx_dtype__MLX_INT32)?;
    let wsc = scalar_i(ctx, w_dim);
    let ymul = mul(ctx, yi, wsc)?;
    let flat = add(ctx, ymul, xi)?; // [N,P] int32
    let flat3 = reshape(ctx, flat, &[n, 1, p])?;
    let idx = broadcast_to(ctx, flat3, &[n, c, p])?;
    let g = take_along_axis(ctx, xf, idx, 2)?; // [N,C,P]
    let w3 = reshape(ctx, weight, &[n, 1, p])?;
    mul(ctx, g, w3)
}

fn grid_sample_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x0 = ctx.resolve(&n.inputs[0])?;
    let xc = contiguous(ctx, x0)?;
    let x = astype(ctx, xc, mlx::mlx_dtype__MLX_FLOAT32)?;
    let xs = ctx.shape_of(x);
    let (nn, cc, hh, ww) = (xs[0], xs[1], xs[2], xs[3]);

    let g0 = ctx.resolve(&n.inputs[1])?;
    let grid = astype(ctx, g0, mlx::mlx_dtype__MLX_FLOAT32)?;
    let gs = ctx.shape_of(grid);
    let (hout, wout) = (gs[1], gs[2]);
    let p = hout * wout;

    let mode = n
        .strings
        .get("mode")
        .map(String::as_str)
        .unwrap_or("linear");
    let padding = n
        .strings
        .get("padding_mode")
        .map(String::as_str)
        .unwrap_or("zeros");
    let align = n.ints.get("align_corners").copied().unwrap_or(0) != 0;
    let nearest = mode == "nearest";
    let zeros = padding == "zeros";

    let ax = if align {
        (ww - 1) as f32 / 2.0
    } else {
        ww as f32 / 2.0
    };
    let bx = (ww - 1) as f32 / 2.0;
    let ay = if align {
        (hh - 1) as f32 / 2.0
    } else {
        hh as f32 / 2.0
    };
    let by = (hh - 1) as f32 / 2.0;

    let xf = reshape(ctx, x, &[nn, cc, hh * ww])?;
    let gflat = reshape(ctx, grid, &[nn, p, 2])?;
    let gx_s = slice(ctx, gflat, &[0, 0, 0], &[nn, p, 1])?;
    let gx = reshape(ctx, gx_s, &[nn, p])?;
    let gy_s = slice(ctx, gflat, &[0, 0, 1], &[nn, p, 2])?;
    let gy = reshape(ctx, gy_s, &[nn, p])?;

    let axs = scalar_f(ctx, ax);
    let gxa = mul(ctx, gx, axs)?;
    let bxs = scalar_f(ctx, bx);
    let ix = add(ctx, gxa, bxs)?; // [N,P]
    let ays = scalar_f(ctx, ay);
    let gya = mul(ctx, gy, ays)?;
    let bys = scalar_f(ctx, by);
    let iy = add(ctx, gya, bys)?; // [N,P]

    let dims = (nn, cc, p, ww, hh);
    let acc = if nearest {
        let one = scalar_f(ctx, 1.0);
        let w1 = broadcast_to(ctx, one, &[nn, p])?;
        let rx = round0(ctx, ix)?;
        let ry = round0(ctx, iy)?;
        gs_sample(ctx, xf, rx, ry, w1, zeros, dims)?
    } else {
        let x0f = floor(ctx, ix)?;
        let y0f = floor(ctx, iy)?;
        let one = scalar_f(ctx, 1.0);
        let x1f = add(ctx, x0f, one)?;
        let y1f = add(ctx, y0f, one)?;
        let wx1 = sub(ctx, ix, x0f)?;
        let one2 = scalar_f(ctx, 1.0);
        let wx0 = sub(ctx, one2, wx1)?;
        let wy1 = sub(ctx, iy, y0f)?;
        let one3 = scalar_f(ctx, 1.0);
        let wy0 = sub(ctx, one3, wy1)?;

        let w00 = mul(ctx, wx0, wy0)?;
        let mut acc = gs_sample(ctx, xf, x0f, y0f, w00, zeros, dims)?;
        let w10 = mul(ctx, wx1, wy0)?;
        let c10 = gs_sample(ctx, xf, x1f, y0f, w10, zeros, dims)?;
        acc = add(ctx, acc, c10)?;
        let w01 = mul(ctx, wx0, wy1)?;
        let c01 = gs_sample(ctx, xf, x0f, y1f, w01, zeros, dims)?;
        acc = add(ctx, acc, c01)?;
        let w11 = mul(ctx, wx1, wy1)?;
        let c11 = gs_sample(ctx, xf, x1f, y1f, w11, zeros, dims)?;
        add(ctx, acc, c11)?
    };

    let out = reshape(ctx, acc, &[nn, cc, hout, wout])?;
    let out = astype(ctx, out, out_dtype(n))?;
    let y = contiguous(ctx, out)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn grid_sample_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, g, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(g), Some(o)) => (x, g, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        is_mlx_float(x.dtype) && out.dtype == x.dtype && is_mlx_float(g.dtype),
        "input/output must share one float dtype and grid must be float, got {} / {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(g.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() == 4 && g.shape.len() == 4 && g.shape.last() == Some(&2),
        "input must be rank 4 and grid must have shape [N,H,W,2], got {:?} and {:?}",
        x.shape,
        g.shape
    );
    require!(
        x.shape.iter().all(|&d| d >= 0) && g.shape.iter().all(|&d| d >= 0),
        "input and grid shapes must be static, got {:?} and {:?}",
        x.shape,
        g.shape
    );
    let mode = node.string_attr("mode", "linear");
    require!(
        mode == "linear" || mode == "bilinear" || mode == "nearest",
        "mode must be \"linear\", \"bilinear\", or \"nearest\", got {mode:?}"
    );
    let padding = node.string_attr("padding_mode", "zeros");
    require!(
        padding == "zeros" || padding == "border",
        "padding_mode must be \"zeros\" or \"border\", got {padding:?}"
    );
    Ok(())
}

// =================================================================================================
// AffineGrid (2D / 3D)
// =================================================================================================

fn affine_grid_shape(theta: &[i32], size: &[i64]) -> Result<Vec<i32>, MlxError> {
    if !matches!(size.len(), 4 | 5) {
        return Err("AffineGrid: size must contain 4 or 5 dimensions".into());
    }
    let rank = size.len() - 2;
    if theta.len() != 3
        || theta[1] != rank as i32
        || theta[2] != rank as i32 + 1
        || i64::from(theta[0]) != size[0]
    {
        return Err(
            "AffineGrid: theta must have shape [size[0], spatial_rank, spatial_rank+1]".into(),
        );
    }
    if size
        .iter()
        .enumerate()
        .any(|(axis, &d)| d < 0 || (axis != 1 && d > i64::from(i32::MAX)))
        || size[2..].contains(&0)
    {
        return Err("AffineGrid: size requires nonnegative N/C and positive spatial dimensions within int32".into());
    }
    let spatial = size[2..].iter().try_fold(1i64, |p, &d| p.checked_mul(d));
    let base_count = spatial.and_then(|p| p.checked_mul(rank as i64 + 1));
    let out_count = spatial
        .and_then(|p| p.checked_mul(size[0]))
        .and_then(|p| p.checked_mul(rank as i64));
    if base_count.is_none_or(|p| p > i64::from(i32::MAX))
        || out_count.is_none_or(|p| p > i64::from(i32::MAX))
    {
        return Err("AffineGrid: grid element count exceeds int32 indexing".into());
    }
    let mut shape = vec![theta[0]];
    shape.extend(size[2..].iter().map(|&d| d as i32));
    shape.push(rank as i32);
    Ok(shape)
}

fn affine_grid_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let size = ctx.read_ints_eval(&n.inputs[1])?;
    let t0 = ctx.resolve(&n.inputs[0])?;
    let out_shape = affine_grid_shape(&ctx.shape_of(t0), &size)?;
    let rank = size.len() - 2;
    let align = n.ints.get("align_corners").copied().unwrap_or(0) != 0;
    if align && size[2..].contains(&1) {
        return Err("AffineGrid: align_corners=1 singleton axes are not supported".into());
    }
    if size[0] == 0 {
        let y = ctx.emit(|res, s| unsafe {
            mlx::mlx_zeros(res, out_shape.as_ptr(), out_shape.len(), out_dtype(n), s)
        })?;
        ctx.bind(&n.outputs[0], y);
        return Ok(());
    }
    let coord = |i: i64, l: i64| -> f64 {
        if align {
            -1.0 + i as f64 * (2.0 / (l - 1) as f64)
        } else {
            (2.0 * i as f64 + 1.0) / l as f64 - 1.0
        }
    };
    let points: i64 = size[2..].iter().product();
    let mut base = Vec::new();
    base.try_reserve_exact(points as usize * (rank + 1))
        .map_err(|e| format!("AffineGrid: cannot allocate homogeneous grid: {e}"))?;
    for point in 0..points {
        let mut remaining = point;
        for &dim in size[2..].iter().rev() {
            base.push(coord(remaining % dim, dim));
            remaining /= dim;
        }
        base.push(1.0);
    }
    let base_shape = [1, points as i32, rank as i32 + 1];
    let double = out_dtype(n) == mlx::mlx_dtype__MLX_FLOAT64;
    let base_arr = if double {
        ctx.from_host(
            base.as_ptr().cast(),
            &base_shape,
            mlx::mlx_dtype__MLX_FLOAT64,
        )
    } else {
        let base: Vec<f32> = base.iter().map(|&v| v as f32).collect();
        host_f32(ctx, &base, &base_shape)
    };

    let theta = astype(
        ctx,
        t0,
        if double {
            mlx::mlx_dtype__MLX_FLOAT64
        } else {
            mlx::mlx_dtype__MLX_FLOAT32
        },
    )?;
    let theta_t = ctx.transpose(theta, &[0, 2, 1])?;
    let grid = matmul(ctx, base_arr, theta_t)?;
    let grid = reshape(ctx, grid, &out_shape)?;
    let grid = astype(ctx, grid, out_dtype(n))?;
    let y = contiguous(ctx, grid)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn affine_grid_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (t, size_info, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(t), Some(s), Some(o)) => (t, s, o),
        _ => deny!("missing tensor type/shape info on theta, size, or output"),
    };
    require!(
        is_mlx_cpu_float(t.dtype) && out.dtype == t.dtype,
        "theta and output must share one supported float dtype, got {} -> {}",
        crate::registry::ort_dtype_name(t.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        size_info.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
            && size_info.shape.len() == 1
            && matches!(size_info.shape[0], 4 | 5),
        "size must be an int64 vector of length 4 or 5"
    );
    let rank = size_info.shape[0] - 2;
    require!(
        t.shape.len() == 3
            && (t.shape[1] < 0 || t.shape[1] == rank)
            && (t.shape[2] < 0 || t.shape[2] == rank + 1),
        "theta must have shape [N,{rank},{}], got {:?}",
        rank + 1,
        t.shape
    );
    require!(
        out.shape.len() == rank as usize + 2
            && (out.shape[0] < 0 || t.shape[0] < 0 || out.shape[0] == t.shape[0])
            && (out.shape.last().is_some_and(|&d| d < 0 || d == rank)),
        "output must have shape [N,spatial...,{rank}], got {:?}",
        out.shape
    );
    let align = node.int_attr("align_corners", 0);
    require!(matches!(align, 0 | 1), "align_corners must be 0 or 1");
    if let Some(size) = node.read_const_ints_any(1) {
        let mut theta = vec![size.first().copied().unwrap_or(-1), rank, rank + 1];
        if t.shape[0] >= 0 {
            theta[0] = t.shape[0];
        }
        let theta: Option<Vec<i32>> = theta.iter().map(|&d| i32::try_from(d).ok()).collect();
        let shape = match theta.and_then(|t| affine_grid_shape(&t, &size).ok()) {
            Some(shape) => shape,
            None => deny!("invalid or oversized AffineGrid theta/size geometry"),
        };
        require!(
            out.shape
                .iter()
                .zip(&shape)
                .all(|(&a, &b)| a < 0 || a == i64::from(b)),
            "output shape disagrees with size"
        );
        require!(
            align == 0 || !size[2..].contains(&1),
            "align_corners=1 singleton axes have conflicting ONNX function/reference/ORT semantics"
        );
    } else {
        require!(
            align == 0,
            "live size with align_corners=1 may select unresolved singleton-axis semantics"
        );
    }
    Ok(())
}

// =================================================================================================
// Col2Im
// =================================================================================================

fn col2im_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let in0 = ctx.resolve(&n.inputs[0])?;
    let input = contiguous(ctx, in0)?;
    let is = ctx.shape_of(input);
    let nn = is[0];
    let l = is[2];

    let image_shape = read_const_ints(ctx, &n.inputs[1])?;
    let block_shape = read_const_ints(ctx, &n.inputs[2])?;
    let r = image_shape.len();

    let dil = n
        .int_arrays
        .get("dilations")
        .cloned()
        .unwrap_or_else(|| vec![1; r]);
    let stride = n
        .int_arrays
        .get("strides")
        .cloned()
        .unwrap_or_else(|| vec![1; r]);
    let pads = n
        .int_arrays
        .get("pads")
        .cloned()
        .unwrap_or_else(|| vec![0; 2 * r]);

    let mut kk: i64 = 1;
    for &block in &block_shape {
        kk *= block;
    }
    let c = is[1] / kk as i32;
    let mut ss: i64 = 1;
    for &image_dim in &image_shape {
        ss *= image_dim;
    }

    let mut npos = vec![0i64; r];
    let mut img_stride = vec![1i64; r];
    // Indexing aligned shape vectors keeps each spatial dimension's values together.
    #[allow(clippy::needless_range_loop)]
    for d in 0..r {
        npos[d] = (image_shape[d] + pads[d] + pads[r + d] - dil[d] * (block_shape[d] - 1) - 1)
            / stride[d]
            + 1;
    }
    for d in (0..r.saturating_sub(1)).rev() {
        img_stride[d] = img_stride[d + 1] * image_shape[d + 1];
    }

    // Host-compute the (column, output-spatial) map, dropping out-of-range positions.
    let mut col_of: Vec<i32> = Vec::new();
    let mut spatial: Vec<i32> = Vec::new();
    let mut kc = vec![0i64; r];
    let mut lc = vec![0i64; r];
    for k in 0..kk {
        let mut rem = k;
        for d in (0..r).rev() {
            kc[d] = rem % block_shape[d];
            rem /= block_shape[d];
        }
        for lv in 0..(l as i64) {
            let mut r2 = lv;
            for d in (0..r).rev() {
                lc[d] = r2 % npos[d];
                r2 /= npos[d];
            }
            let mut pos: i64 = 0;
            let mut ok = true;
            for d in 0..r {
                let op = lc[d] * stride[d] + kc[d] * dil[d] - pads[d];
                if op < 0 || op >= image_shape[d] {
                    ok = false;
                    break;
                }
                pos += op * img_stride[d];
            }
            if ok {
                col_of.push((k * l as i64 + lv) as i32);
                spatial.push(pos as i32);
            }
        }
    }
    let m = col_of.len() as i32;

    let total = nn * c * ss as i32;
    let mut out_acc = zeros_1d(ctx, total)?;

    if m > 0 {
        let in_reshaped = reshape(ctx, input, &[nn, c, kk as i32 * l])?;
        let in_cols = astype(ctx, in_reshaped, mlx::mlx_dtype__MLX_FLOAT32)?;
        let col_idx = host_i32(ctx, &col_of, &[m]);
        let gathered = take_axis(ctx, in_cols, col_idx, 2)?; // [N,C,M]

        // Flat scatter index [N*C*M]: element (nn,c,j) -> (nn*C+c)*S + spatial[j].
        let mut scatter_idx = vec![0i32; (nn as i64 * c as i64 * m as i64) as usize];
        let mut w = 0usize;
        for nv in 0..nn {
            for cv in 0..c {
                let base = ((nv as i64 * c as i64 + cv as i64) * ss) as i32;
                for &spatial_index in &spatial {
                    scatter_idx[w] = base + spatial_index;
                    w += 1;
                }
            }
        }
        let ilen = scatter_idx.len() as i32;
        let idx = host_i32(ctx, &scatter_idx, &[ilen]);
        let updates = reshape(ctx, gathered, &[ilen, 1])?;
        out_acc = scatter_add(ctx, out_acc, idx, updates)?;
    }

    let mut out_shape = vec![nn, c];
    out_shape.extend(image_shape.iter().map(|&d| d as i32));
    let out = reshape(ctx, out_acc, &out_shape)?;
    let out = astype(ctx, out, out_dtype(n))?;
    let y = contiguous(ctx, out)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn col2im_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 3 && node.num_outputs() == 1,
        "Col2Im expects 3 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (i, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => (i, o),
        _ => deny!("missing tensor type/shape info on input[0] or output"),
    };
    require!(
        is_mlx_float(i.dtype) && out.dtype == i.dtype,
        "Col2Im input/output must be the same float dtype (fp32/fp16/bf16), got {} -> {}",
        crate::registry::ort_dtype_name(i.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        i.shape.len() == 3 && i.shape.iter().all(|&d| d >= 0),
        "Col2Im input[0] must be a static rank-3 [N,C*prod(block),L] tensor, got shape {:?}",
        i.shape
    );
    require!(
        node.is_const_int_tensor(1) && node.is_const_int_tensor(2),
        "Col2Im `image_shape` (input 1) and `block_shape` (input 2) must be constant integer \
         initializers; runtime values stay on CPU"
    );
    let image_shape = match node.read_const_ints_any(1) {
        Some(v) => v,
        None => deny!("Col2Im `image_shape` (input 1) is not a readable constant int tensor"),
    };
    let block_shape = match node.read_const_ints_any(2) {
        Some(v) => v,
        None => deny!("Col2Im `block_shape` (input 2) is not a readable constant int tensor"),
    };
    let r = image_shape.len();
    require!(
        (1..=3).contains(&r) && block_shape.len() == r,
        "Col2Im supports 1-3 spatial dims with matching image_shape/block_shape ranks \
         (got image rank {r}, block rank {})",
        block_shape.len()
    );
    require!(
        image_shape.iter().all(|&d| d >= 1),
        "Col2Im `image_shape` dims must all be >= 1, got {image_shape:?}"
    );
    let mut kk: i64 = 1;
    for &b in &block_shape {
        require!(
            b >= 1,
            "Col2Im `block_shape` dims must all be >= 1, got {block_shape:?}"
        );
        kk *= b;
    }
    require!(
        kk != 0 && i.shape[1] % kk == 0 && i.shape[1] / kk >= 1,
        "Col2Im input channels {} must be a positive multiple of prod(block_shape)={kk}",
        i.shape[1]
    );
    let ints_ok = |name: &str, want: usize| -> bool {
        let (present, v) = node.ints_attr(name);
        !present || v.len() == want
    };
    require!(
        ints_ok("dilations", r) && ints_ok("strides", r) && ints_ok("pads", 2 * r),
        "Col2Im `dilations`/`strides` must have {r} entries and `pads` {} entries (per spatial dim)",
        2 * r
    );
    Ok(())
}

// =================================================================================================
// RoiAlign (2D)
// =================================================================================================

/// A single [R,1] ROI column reshaped to [R].
fn roi_column(
    ctx: &mut TranslationContext,
    rois: mlx::mlx_array,
    r: i32,
    col: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let s = slice(ctx, rois, &[0, col], &[r, col + 1])?;
    reshape(ctx, s, &[r])
}

fn roi_align_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x0 = ctx.resolve(&n.inputs[0])?;
    let xc = contiguous(ctx, x0)?;
    let xin = astype(ctx, xc, mlx::mlx_dtype__MLX_FLOAT32)?;
    let xs = ctx.shape_of(xin);
    let (cc, hh, ww) = (xs[1], xs[2], xs[3]);

    let r0 = ctx.resolve(&n.inputs[1])?;
    let rois = astype(ctx, r0, mlx::mlx_dtype__MLX_FLOAT32)?; // [R,4]
    let rr = ctx.shape_of(rois)[0];
    let b0 = ctx.resolve(&n.inputs[2])?;
    if hh <= 0 || ww <= 0 || ctx.shape_of(b0) != [rr] {
        return Err(
            "RoiAlign: positive spatial dimensions and one batch index per ROI are required".into(),
        );
    }
    let batch_idx = astype(ctx, b0, mlx::mlx_dtype__MLX_INT32)?; // [R]

    let mode = n.strings.get("mode").map(String::as_str).unwrap_or("avg");
    let ctm = n
        .strings
        .get("coordinate_transformation_mode")
        .map(String::as_str)
        .unwrap_or(if n.since_version < 16 {
            "output_half_pixel"
        } else {
            "half_pixel"
        });
    let half_pixel = ctm == "half_pixel";
    let is_max = mode == "max";
    let oh = n.ints.get("output_height").copied().unwrap_or(1) as i32;
    let ow = n.ints.get("output_width").copied().unwrap_or(1) as i32;
    let sr = n.ints.get("sampling_ratio").copied().unwrap_or(0) as i32;
    let scale = n.floats.get("spatial_scale").copied().unwrap_or(1.0);
    let (gh, gw) = (sr, sr);
    let offset = if half_pixel { 0.5 } else { 0.0 };
    let count = (gh * gw).max(1) as f32;

    // Per-ROI start / bin size on-device: coord = rois[:,k]*scale - offset.
    let denorm = |ctx: &mut TranslationContext, col: i32| -> Result<mlx::mlx_array, MlxError> {
        let c = roi_column(ctx, rois, rr, col)?;
        let sc = scalar_f(ctx, scale);
        let m = mul(ctx, c, sc)?;
        let off = scalar_f(ctx, offset);
        sub(ctx, m, off)
    };
    let rsw = denorm(ctx, 0)?;
    let rsh = denorm(ctx, 1)?;
    let rew = denorm(ctx, 2)?;
    let reh = denorm(ctx, 3)?;
    let mut roi_w = sub(ctx, rew, rsw)?;
    let mut roi_h = sub(ctx, reh, rsh)?;
    if !half_pixel {
        let one = scalar_f(ctx, 1.0);
        roi_w = maximum(ctx, roi_w, one)?;
        let one2 = scalar_f(ctx, 1.0);
        roi_h = maximum(ctx, roi_h, one2)?;
    }
    let owf = scalar_f(ctx, ow as f32);
    let binw = div(ctx, roi_w, owf)?;
    let ohf = scalar_f(ctx, oh as f32);
    let binh = div(ctx, roi_h, ohf)?;

    let rsh5 = reshape(ctx, rsh, &[rr, 1, 1, 1, 1])?;
    let rsw5 = reshape(ctx, rsw, &[rr, 1, 1, 1, 1])?;
    let binh5 = reshape(ctx, binh, &[rr, 1, 1, 1, 1])?;
    let binw5 = reshape(ctx, binw, &[rr, 1, 1, 1, 1])?;
    let ah = arange(ctx, oh)?;
    let ph_idx = reshape(ctx, ah, &[1, oh, 1, 1, 1])?;
    let aw = arange(ctx, ow)?;
    let pw_idx = reshape(ctx, aw, &[1, 1, ow, 1, 1])?;
    let agh = arange(ctx, gh)?;
    let iy_idx = reshape(ctx, agh, &[1, 1, 1, gh, 1])?;
    let agw = arange(ctx, gw)?;
    let ix_idx = reshape(ctx, agw, &[1, 1, 1, 1, gw])?;

    // y = rsh + ph*binh + (iy+0.5)*binh/gh ; x = rsw + pw*binw + (ix+0.5)*binw/gw
    let half = scalar_f(ctx, 0.5);
    let iy_h = add(ctx, iy_idx, half)?;
    let ghf = scalar_f(ctx, gh as f32);
    let iy_frac = div(ctx, iy_h, ghf)?;
    let iy_term = mul(ctx, iy_frac, binh5)?;
    let ph_term = mul(ctx, ph_idx, binh5)?;
    let yc0 = add(ctx, rsh5, ph_term)?;
    let yc = add(ctx, yc0, iy_term)?;

    let half2 = scalar_f(ctx, 0.5);
    let ix_h = add(ctx, ix_idx, half2)?;
    let gwf = scalar_f(ctx, gw as f32);
    let ix_frac = div(ctx, ix_h, gwf)?;
    let ix_term = mul(ctx, ix_frac, binw5)?;
    let pw_term = mul(ctx, pw_idx, binw5)?;
    let xc0 = add(ctx, rsw5, pw_term)?;
    let xc = add(ctx, xc0, ix_term)?;

    let ps = oh * ow * gh * gw;
    let yc_b = broadcast_to(ctx, yc, &[rr, oh, ow, gh, gw])?;
    let yc = reshape(ctx, yc_b, &[rr, ps])?;
    let xc_b = broadcast_to(ctx, xc, &[rr, oh, ow, gh, gw])?;
    let xc = reshape(ctx, xc_b, &[rr, ps])?;

    let vy = in_range_mask(ctx, yc, -1.0, hh as f32)?;
    let vx = in_range_mask(ctx, xc, -1.0, ww as f32)?;
    let valid = mul(ctx, vy, vx)?;
    let cy = clip(ctx, yc, 0.0, (hh - 1) as f32)?;
    let cx = clip(ctx, xc, 0.0, (ww - 1) as f32)?;
    let y0 = floor(ctx, cy)?;
    let x0 = floor(ctx, cx)?;
    let one = scalar_f(ctx, 1.0);
    let y0p1 = add(ctx, y0, one)?;
    let y1 = clip(ctx, y0p1, 0.0, (hh - 1) as f32)?;
    let one2 = scalar_f(ctx, 1.0);
    let x0p1 = add(ctx, x0, one2)?;
    let x1 = clip(ctx, x0p1, 0.0, (ww - 1) as f32)?;
    let ly = sub(ctx, cy, y0)?;
    let lx = sub(ctx, cx, x0)?;
    let one3 = scalar_f(ctx, 1.0);
    let hy = sub(ctx, one3, ly)?;
    let one4 = scalar_f(ctx, 1.0);
    let hx = sub(ctx, one4, lx)?;
    let hyhx = mul(ctx, hy, hx)?;
    let w1 = mul(ctx, hyhx, valid)?;
    let hylx = mul(ctx, hy, lx)?;
    let w2 = mul(ctx, hylx, valid)?;
    let lyhx = mul(ctx, ly, hx)?;
    let w3 = mul(ctx, lyhx, valid)?;
    let lylx = mul(ctx, ly, lx)?;
    let w4 = mul(ctx, lylx, valid)?;

    let xb = take_axis(ctx, xin, batch_idx, 0)?; // [R,C,H,W]
    let xbf = reshape(ctx, xb, &[rr, cc, hh * ww])?;

    // corner(yy, xx, w): gather bilinear neighbour, weighted.
    let corner = |ctx: &mut TranslationContext,
                  yy: mlx::mlx_array,
                  xx: mlx::mlx_array,
                  w: mlx::mlx_array|
     -> Result<mlx::mlx_array, MlxError> {
        let yi = astype(ctx, yy, mlx::mlx_dtype__MLX_INT32)?;
        let wsc = scalar_i(ctx, ww);
        let ymul = mul(ctx, yi, wsc)?;
        let xi = astype(ctx, xx, mlx::mlx_dtype__MLX_INT32)?;
        let flat = add(ctx, ymul, xi)?;
        let flat3 = reshape(ctx, flat, &[rr, 1, ps])?;
        let idx = broadcast_to(ctx, flat3, &[rr, cc, ps])?;
        let g = take_along_axis(ctx, xbf, idx, 2)?; // [R,C,PS]
        let w3d = reshape(ctx, w, &[rr, 1, ps])?;
        mul(ctx, g, w3d)
    };
    let c1 = corner(ctx, y0, x0, w1)?;
    let c2 = corner(ctx, y0, x1, w2)?;
    let c3 = corner(ctx, y1, x0, w3)?;
    let c4 = corner(ctx, y1, x1, w4)?;

    let out = if is_max {
        let m12 = maximum(ctx, c1, c2)?;
        let m34 = maximum(ctx, c3, c4)?;
        let v = maximum(ctx, m12, m34)?; // [R,C,PS]
        let vr = reshape(ctx, v, &[rr, cc, oh * ow, gh * gw])?;
        max_axis(ctx, vr, 3)?
    } else {
        let s12 = add(ctx, c1, c2)?;
        let s34 = add(ctx, c3, c4)?;
        let s = add(ctx, s12, s34)?;
        let sr2 = reshape(ctx, s, &[rr, cc, oh * ow, gh * gw])?;
        let summed = sum_axis(ctx, sr2, 3)?;
        let cnt = scalar_f(ctx, count);
        div(ctx, summed, cnt)?
    };
    let out = reshape(ctx, out, &[rr, cc, oh, ow])?;
    let out = astype(ctx, out, out_dtype(n))?;
    let y = contiguous(ctx, out)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn roi_align_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 3 && node.num_outputs() == 1,
        "expects 3 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, r, b, out) = match (
        node.input_info(0),
        node.input_info(1),
        node.input_info(2),
        node.output_info(0),
    ) {
        (Some(x), Some(r), Some(b), Some(o)) => (x, r, b, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        is_mlx_float(x.dtype)
            && out.dtype == x.dtype
            && is_mlx_float(r.dtype)
            && is_int_index(b.dtype),
        "input/output must share one float dtype, rois must be float, and batch_indices int32/int64; got {} / {} / {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(r.dtype),
        crate::registry::ort_dtype_name(b.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() == 4,
        "input must have rank-4 shape, got {:?}",
        x.shape
    );
    require!(
        r.shape.len() == 2 && r.shape[1] == 4,
        "rois must have shape [R,4], got {:?}",
        r.shape
    );
    require!(
        b.shape.len() == 1,
        "batch_indices must have rank-1 shape, got {:?}",
        b.shape
    );
    let sampling_ratio = node.int_attr("sampling_ratio", 0);
    require!(
        sampling_ratio > 0,
        "sampling_ratio must be positive, got {sampling_ratio}"
    );
    let mode = node.string_attr("mode", "avg");
    require!(
        mode == "avg" || mode == "max",
        "mode must be \"avg\" or \"max\", got {mode:?}"
    );
    require!(
        x.shape[2] != 0
            && x.shape[3] != 0
            && (b.shape[0] < 0 || r.shape[0] < 0 || b.shape[0] == r.shape[0]),
        "spatial dimensions must be positive and batch_indices must match the ROI count"
    );
    require!(
        node.int_attr("output_height", 1) > 0 && node.int_attr("output_width", 1) > 0,
        "output_height and output_width must be positive"
    );
    require!(
        [
            node.int_attr("output_height", 1),
            node.int_attr("output_width", 1),
            sampling_ratio,
            sampling_ratio,
        ]
        .iter()
        .try_fold(1i64, |size, &dim| size.checked_mul(dim))
        .is_some_and(|size| size <= i32::MAX as i64),
        "output grid and sampling ratio product must fit int32"
    );
    let ctm = node.string_attr(
        "coordinate_transformation_mode",
        if node.since_version() < 16 {
            "output_half_pixel"
        } else {
            "half_pixel"
        },
    );
    require!(
        ctm == "half_pixel" || ctm == "output_half_pixel",
        "coordinate_transformation_mode must be \"half_pixel\" or \"output_half_pixel\", got {ctm:?}"
    );
    Ok(())
}

// =================================================================================================
// MaxRoiPool
// =================================================================================================

fn max_roi_pool_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x0 = ctx.resolve(&n.inputs[0])?;
    let xc = contiguous(ctx, x0)?;
    let xin = astype(ctx, xc, mlx::mlx_dtype__MLX_FLOAT32)?;
    let xs = ctx.shape_of(xin);
    let (cc, hh, ww) = (xs[1], xs[2], xs[3]);
    if hh <= 0 || ww <= 0 {
        return Err("MaxRoiPool: spatial dimensions must be positive".into());
    }

    let r0 = ctx.resolve(&n.inputs[1])?;
    let rois = astype(ctx, r0, mlx::mlx_dtype__MLX_FLOAT32)?; // [R,5]
    let rr = ctx.shape_of(rois)[0];

    let pooled = n
        .int_arrays
        .get("pooled_shape")
        .cloned()
        .unwrap_or_default();
    let ph = pooled[0] as i32;
    let pw = pooled[1] as i32;
    let scale = n.floats.get("spatial_scale").copied().unwrap_or(1.0);

    let bc = roi_column(ctx, rois, rr, 0)?;
    let batch_idx = astype(ctx, bc, mlx::mlx_dtype__MLX_INT32)?; // [R]

    let rounded = |ctx: &mut TranslationContext, col: i32| -> Result<mlx::mlx_array, MlxError> {
        let c = roi_column(ctx, rois, rr, col)?;
        let sc = scalar_f(ctx, scale);
        let m = mul(ctx, c, sc)?;
        round_away(ctx, m)
    };
    let rsw = rounded(ctx, 1)?;
    let rsh = rounded(ctx, 2)?;
    let rew = rounded(ctx, 3)?;
    let reh = rounded(ctx, 4)?;
    let one = scalar_f(ctx, 1.0);
    let dw = sub(ctx, rew, rsw)?;
    let dw1 = add(ctx, dw, one)?;
    let one_a = scalar_f(ctx, 1.0);
    let roi_w = maximum(ctx, dw1, one_a)?;
    let one_b = scalar_f(ctx, 1.0);
    let dh = sub(ctx, reh, rsh)?;
    let dh1 = add(ctx, dh, one_b)?;
    let one_c = scalar_f(ctx, 1.0);
    let roi_h = maximum(ctx, dh1, one_c)?;
    let pwf = scalar_f(ctx, pw as f32);
    let binw = div(ctx, roi_w, pwf)?;
    let phf = scalar_f(ctx, ph as f32);
    let binh = div(ctx, roi_h, phf)?;

    // Integer window bounds per (ROI, output bin), clamped to [0,limit]. lo/hi shapes [R,P].
    let bounds = |ctx: &mut TranslationContext,
                  p: i32,
                  bin: mlx::mlx_array,
                  start: mlx::mlx_array,
                  limit: i32|
     -> Result<(mlx::mlx_array, mlx::mlx_array), MlxError> {
        let ap = arange(ctx, p)?;
        let p_idx = reshape(ctx, ap, &[1, p])?;
        let bin2 = reshape(ctx, bin, &[rr, 1])?;
        let start2 = reshape(ctx, start, &[rr, 1])?;
        let lo_m = mul(ctx, p_idx, bin2)?;
        let lo_f = floor(ctx, lo_m)?;
        let lo_a = add(ctx, lo_f, start2)?;
        let lo = clip(ctx, lo_a, 0.0, limit as f32)?;
        let one_h = scalar_f(ctx, 1.0);
        let p1 = add(ctx, p_idx, one_h)?;
        let hi_m = mul(ctx, p1, bin2)?;
        let hi_c = ceil(ctx, hi_m)?;
        let hi_a = add(ctx, hi_c, start2)?;
        let hi = clip(ctx, hi_a, 0.0, limit as f32)?;
        Ok((lo, hi))
    };
    let (hstart, hend) = bounds(ctx, ph, binh, rsh, hh)?; // [R,PH]
    let (wstart, wend) = bounds(ctx, pw, binw, rsw, ww)?; // [R,PW]

    let ahh = arange(ctx, hh)?;
    let h_idx = reshape(ctx, ahh, &[1, 1, hh])?;
    let aww = arange(ctx, ww)?;
    let w_idx = reshape(ctx, aww, &[1, 1, ww])?;
    let hstart3 = reshape(ctx, hstart, &[rr, ph, 1])?;
    let ge_h = ctx.binary(mlx::mlx_greater_equal, h_idx, hstart3)?;
    let hend3 = reshape(ctx, hend, &[rr, ph, 1])?;
    let lt_h = ctx.binary(mlx::mlx_less, h_idx, hend3)?;
    let mask_h = ctx.binary(mlx::mlx_logical_and, ge_h, lt_h)?; // [R,PH,H]
    let wstart3 = reshape(ctx, wstart, &[rr, pw, 1])?;
    let ge_w = ctx.binary(mlx::mlx_greater_equal, w_idx, wstart3)?;
    let wend3 = reshape(ctx, wend, &[rr, pw, 1])?;
    let lt_w = ctx.binary(mlx::mlx_less, w_idx, wend3)?;
    let mask_w = ctx.binary(mlx::mlx_logical_and, ge_w, lt_w)?; // [R,PW,W]

    let neg = scalar_f(ctx, f32::MIN);
    let xb = take_axis(ctx, xin, batch_idx, 0)?; // [R,C,H,W]

    // Masked max over H: [R,PH,C,H,W] -> [R,PH,C,W].
    let mask_h5 = reshape(ctx, mask_h, &[rr, ph, 1, hh, 1])?;
    let xb5 = reshape(ctx, xb, &[rr, 1, cc, hh, ww])?;
    let sel_h = where_(ctx, mask_h5, xb5, neg)?;
    let maxh = max_axis(ctx, sel_h, 3)?; // [R,PH,C,W]

    // Masked max over W: [R,PH,PW,C,W] -> [R,PH,PW,C].
    let mask_w5 = reshape(ctx, mask_w, &[rr, 1, pw, 1, ww])?;
    let maxh5 = reshape(ctx, maxh, &[rr, ph, 1, cc, ww])?;
    let sel_w = where_(ctx, mask_w5, maxh5, neg)?;
    let mut out = max_axis(ctx, sel_w, 4)?; // [R,PH,PW,C]

    // Empty windows (hend<=hstart or wend<=wstart) emit 0.
    let le_h = ctx.binary(mlx::mlx_less_equal, hend, hstart)?;
    let le_h3 = reshape(ctx, le_h, &[rr, ph, 1])?;
    let le_w = ctx.binary(mlx::mlx_less_equal, wend, wstart)?;
    let le_w3 = reshape(ctx, le_w, &[rr, 1, pw])?;
    let empty = ctx.binary(mlx::mlx_logical_or, le_h3, le_w3)?;
    let empty4 = reshape(ctx, empty, &[rr, ph, pw, 1])?;
    let zero = scalar_f(ctx, 0.0);
    out = where_(ctx, empty4, zero, out)?;

    let out_t = ctx.transpose(out, &[0, 3, 1, 2])?;
    let out_t = astype(ctx, out_t, out_dtype(n))?;
    let y = contiguous(ctx, out_t)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn max_roi_pool_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, r, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(r), Some(o)) => (x, r, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        is_mlx_float(x.dtype) && out.dtype == x.dtype && is_mlx_float(r.dtype),
        "input/output must share one float dtype and rois must be float, got {} / {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(r.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() == 4,
        "input must have rank-4 shape, got {:?}",
        x.shape
    );
    require!(
        r.shape.len() == 2 && r.shape[1] == 5,
        "rois must have shape [R,5], got {:?}",
        r.shape
    );
    require!(
        x.shape[2] != 0 && x.shape[3] != 0,
        "spatial dimensions must be positive"
    );
    let (present, pooled) = node.ints_attr("pooled_shape");
    require!(
        present && pooled.len() == 2,
        "pooled_shape must contain exactly 2 values, got {:?}",
        pooled
    );
    require!(
        pooled.iter().all(|&v| v >= 1 && v <= i32::MAX as i64),
        "pooled_shape dimensions must be positive int32 values, got {:?}",
        pooled
    );
    Ok(())
}

// =================================================================================================
// MaxUnpool
// =================================================================================================

fn max_unpool_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x0 = ctx.resolve(&n.inputs[0])?;
    let xin = contiguous(ctx, x0)?;
    let xs = ctx.shape_of(xin);
    let nn = xs[0];
    let cc = xs[1];
    let r = xs.len() - 2;

    let kernel = n
        .int_arrays
        .get("kernel_shape")
        .cloned()
        .unwrap_or_default();
    let strides = n
        .int_arrays
        .get("strides")
        .cloned()
        .unwrap_or_else(|| vec![1; r]);
    let pads = n
        .int_arrays
        .get("pads")
        .cloned()
        .unwrap_or_else(|| vec![0; 2 * r]);

    let mut out_shape = vec![nn, cc];
    for d in 0..r {
        let od = (xs[2 + d] as i64 - 1) * strides[d] - (pads[d] + pads[r + d]) + kernel[d];
        out_shape.push(i32::try_from(od).map_err(|_| "MaxUnpool: output extent exceeds int32")?);
    }
    if input_present(n, 2) {
        let shape = ctx.resolve(&n.inputs[2])?;
        let shape = ctx.contiguous_eval(shape)?;
        if ctx.shape_of(shape) != [xs.len() as i32] {
            return Err(
                "MaxUnpool: output_shape must be a vector with one entry per input axis".into(),
            );
        }
        let values =
            unsafe { std::slice::from_raw_parts(mlx::mlx_array_data_int64(shape), xs.len()) };
        out_shape = values
            .iter()
            .map(|&v| {
                i32::try_from(v).map_err(|_| "MaxUnpool: output_shape extent exceeds int32".into())
            })
            .collect::<Result<Vec<_>, MlxError>>()?;
        if out_shape[..2] != xs[..2] {
            return Err("MaxUnpool: output_shape batch and channel dimensions must match X".into());
        }
    }
    if out_shape[2..].iter().any(|&d| d <= 0) {
        return Err("MaxUnpool: output spatial dimensions must be positive".into());
    }
    let total =
        i32::try_from(ctx.size_of(xin)).map_err(|_| "MaxUnpool: input size exceeds int32")?;
    let total_out = out_shape
        .iter()
        .try_fold(1i32, |size, &dim| size.checked_mul(dim))
        .ok_or("MaxUnpool: output size exceeds int32")?;
    let out_acc = zeros_1d(ctx, total_out)?;
    let out_acc = astype(ctx, out_acc, ctx.dtype_of(xin))?;

    let idx0 = ctx.resolve(&n.inputs[1])?;
    if ctx.shape_of(idx0) != xs {
        return Err("MaxUnpool: indices shape must match X".into());
    }
    if total == 0 {
        let output = reshape(ctx, out_acc, &out_shape)?;
        ctx.bind(&n.outputs[0], output);
        return Ok(());
    }
    let idx0 = ctx.contiguous_eval(idx0)?;
    let indices =
        unsafe { std::slice::from_raw_parts(mlx::mlx_array_data_int64(idx0), total as usize) };
    if indices.iter().any(|&i| i < 0 || i >= i64::from(total_out)) {
        return Err("MaxUnpool: indices must be within the flattened output".into());
    }
    let idx_i = astype(ctx, idx0, mlx::mlx_dtype__MLX_INT32)?;
    let idx = reshape(ctx, idx_i, &[total])?;
    let updates = reshape(ctx, xin, &[total, 1])?;
    let mut indices = VectorArray::new();
    indices.append(idx);
    let scattered = ctx.emit(|res, s| unsafe {
        mlx::mlx_scatter(res, out_acc, indices.as_raw(), updates, [0].as_ptr(), 1, s)
    })?;
    let out = reshape(ctx, scattered, &out_shape)?;
    let out = astype(ctx, out, out_dtype(n))?;
    let y = contiguous(ctx, out)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn max_unpool_claim(node: &NodeView) -> ClaimResult {
    require!(
        (2..=3).contains(&node.num_inputs()) && node.num_outputs() == 1,
        "expects 2 or 3 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, i, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(i), Some(o)) => (x, i, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        is_mlx_float(x.dtype)
            && out.dtype == x.dtype
            && i.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        "input/output must share one float dtype and indices must be int64, got {} / {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(i.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() >= 3 && x.shape.len() <= 5,
        "input must have rank 3-5 shape, got {:?}",
        x.shape
    );
    require!(
        i.shape.len() == x.shape.len()
            && i.shape
                .iter()
                .zip(&x.shape)
                .all(|(&i, &x)| i < 0 || x < 0 || i == x),
        "indices shape must match input shape, got {:?} vs {:?}",
        i.shape,
        x.shape
    );
    let r = x.shape.len() - 2;
    let (kp, kernel) = node.ints_attr("kernel_shape");
    require!(
        kp && kernel.len() == r,
        "kernel_shape must contain {r} values, got {:?}",
        kernel
    );
    let (hs, strides) = node.ints_attr("strides");
    require!(
        !hs || strides.len() == r,
        "strides must contain {r} values, got {:?}",
        strides
    );
    let (hp, pads) = node.ints_attr("pads");
    require!(
        !hp || pads.len() == 2 * r,
        "pads must contain {} values, got {:?}",
        2 * r,
        pads
    );
    require!(
        kernel.iter().all(|&v| v > 0 && v <= i32::MAX as i64)
            && strides.iter().all(|&v| v > 0 && v <= i32::MAX as i64)
            && pads.iter().all(|&v| v >= 0 && v <= i32::MAX as i64),
        "kernel and strides must be positive int32 values and pads non-negative int32 values"
    );
    if node.input_present(2) {
        let shape = match node.input_info(2) {
            Some(shape) => shape,
            None => deny!("missing output_shape tensor info"),
        };
        require!(
            shape.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                && shape.shape == [x.shape.len() as i64],
            "output_shape must be an int64 vector with one entry per input axis"
        );
    }
    Ok(())
}

// ---- NonMaxSuppression --------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct NmsBox {
    y1: f32,
    x1: f32,
    y2: f32,
    x2: f32,
}

fn nms_box(raw: &[f32], center_point_box: bool) -> NmsBox {
    if center_point_box {
        let x_half = raw[2] * 0.5;
        let y_half = raw[3] * 0.5;
        NmsBox {
            y1: raw[1] - y_half,
            x1: raw[0] - x_half,
            y2: raw[1] + y_half,
            x2: raw[0] + x_half,
        }
    } else {
        NmsBox {
            y1: raw[0].min(raw[2]),
            x1: raw[1].min(raw[3]),
            y2: raw[0].max(raw[2]),
            x2: raw[1].max(raw[3]),
        }
    }
}

fn nms_iou(a: NmsBox, b: NmsBox) -> f32 {
    let intersection_h = (a.y2.min(b.y2) - a.y1.max(b.y1)).max(0.0);
    let intersection_w = (a.x2.min(b.x2) - a.x1.max(b.x1)).max(0.0);
    let intersection = intersection_h * intersection_w;
    let area_a = (a.y2 - a.y1).max(0.0) * (a.x2 - a.x1).max(0.0);
    let area_b = (b.y2 - b.y1).max(0.0) * (b.x2 - b.x1).max(0.0);
    let union = area_a + area_b - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn non_max_suppression_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let boxes = eval_f32(ctx, &n.inputs[0])?;
    let scores = eval_f32(ctx, &n.inputs[1])?;
    let bshape = ctx.shape_of(boxes);
    let sshape = ctx.shape_of(scores);
    let batches = bshape[0] as usize;
    let box_count = bshape[1] as usize;
    let classes = sshape[1] as usize;
    let max_output = scalar_i64_input(ctx, n, 2, 0)?.max(0) as usize;
    let iou_threshold = scalar_f32_input(ctx, n, 3, 0.0)?;
    let score_threshold = if input_present(n, 4) {
        Some(scalar_f32_input(ctx, n, 4, 0.0)?)
    } else {
        None
    };
    let center_point_box = n.ints.get("center_point_box").copied().unwrap_or(0) != 0;

    let boxes_data = unsafe {
        std::slice::from_raw_parts(mlx::mlx_array_data_float32(boxes), batches * box_count * 4)
    };
    let scores_data = unsafe {
        std::slice::from_raw_parts(
            mlx::mlx_array_data_float32(scores),
            batches * classes * box_count,
        )
    };
    let mut selected = Vec::<i64>::new();
    if max_output > 0 {
        for batch in 0..batches {
            for class in 0..classes {
                let score_base = (batch * classes + class) * box_count;
                let mut candidates: Vec<usize> = (0..box_count)
                    .filter(|&index| {
                        score_threshold
                            .is_none_or(|threshold| scores_data[score_base + index] > threshold)
                    })
                    .collect();
                candidates.sort_by(|&left, &right| {
                    scores_data[score_base + right]
                        .partial_cmp(&scores_data[score_base + left])
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| left.cmp(&right))
                });

                let mut kept = Vec::<usize>::new();
                for candidate in candidates {
                    let candidate_offset = (batch * box_count + candidate) * 4;
                    let candidate_box = nms_box(
                        &boxes_data[candidate_offset..candidate_offset + 4],
                        center_point_box,
                    );
                    let suppressed = kept.iter().any(|&prior| {
                        let prior_offset = (batch * box_count + prior) * 4;
                        let prior_box = nms_box(
                            &boxes_data[prior_offset..prior_offset + 4],
                            center_point_box,
                        );
                        nms_iou(candidate_box, prior_box) > iou_threshold
                    });
                    if !suppressed {
                        kept.push(candidate);
                        selected.extend([batch as i64, class as i64, candidate as i64]);
                        if kept.len() == max_output {
                            break;
                        }
                    }
                }
            }
        }
    }

    let rows = (selected.len() / 3) as i32;
    let out = if selected.is_empty() {
        ctx.zeros(&[0, 3], mlx::mlx_dtype__MLX_INT64)?
    } else {
        ctx.from_host_i64(&selected, &[rows, 3])
    };
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn non_max_suppression_claim(node: &NodeView) -> ClaimResult {
    require!(
        (2..=5).contains(&node.num_inputs()) && node.num_outputs() == 1,
        "expects 2-5 inputs and 1 output"
    );
    let (boxes, scores, output) =
        match (node.input_info(0), node.input_info(1), node.output_info(0)) {
            (Some(boxes), Some(scores), Some(output)) => (boxes, scores, output),
            _ => deny!("missing tensor type/shape info"),
        };
    require!(
        boxes.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            && scores.dtype == boxes.dtype,
        "boxes and scores must be float32"
    );
    require!(
        boxes.shape.len() == 3 && boxes.shape[2] == 4 && boxes.shape[0] > 0 && boxes.shape[1] >= 0,
        "boxes must have static shape [batch, boxes, 4]"
    );
    require!(
        scores.shape.len() == 3
            && scores.shape[0] == boxes.shape[0]
            && scores.shape[2] == boxes.shape[1]
            && scores.shape[1] >= 0,
        "scores must have static shape [batch, classes, boxes]"
    );
    require!(
        output.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        "selected_indices output must be int64"
    );
    if node.input_present(2) {
        let max_output = match node.input_info(2) {
            Some(info) => info,
            None => deny!("max_output_boxes_per_class lacks tensor type info"),
        };
        require!(
            max_output.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                && max_output.shape.is_empty(),
            "max_output_boxes_per_class must be an int64 scalar"
        );
    }
    for index in 3..=4 {
        if node.input_present(index) {
            let threshold = match node.input_info(index) {
                Some(info) => info,
                None => deny!("threshold input {index} lacks tensor type info"),
            };
            require!(
                threshold.dtype == boxes.dtype && threshold.shape.is_empty(),
                "threshold input {index} must be a float32 scalar"
            );
        }
    }
    require!(
        matches!(node.int_attr("center_point_box", 0), 0 | 1),
        "center_point_box must be 0 or 1"
    );
    Ok(())
}

// ---- registration -------------------------------------------------------------------------------

fn shapeless(
    registry: &mut OpRegistry,
    op_type: &'static str,
    handler: crate::registry::OpHandler,
    claim: crate::registry::ClaimPredicate,
) {
    registry.register_shapeless(OpRegistration {
        domain: "",
        op_type,
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler,
        claim,
    });
}

fn shape_keyed(
    registry: &mut OpRegistry,
    op_type: &'static str,
    handler: crate::registry::OpHandler,
    claim: crate::registry::ClaimPredicate,
    reason: &'static str,
) {
    registry.register_shape_keyed(
        OpRegistration {
            domain: "",
            op_type,
            min_opset: K_ANY_OPSET,
            max_opset: K_ANY_OPSET,
            handler,
            claim,
        },
        reason,
    );
}

pub fn register_vision(registry: &mut OpRegistry) {
    shape_keyed(
        registry,
        "GridSample",
        grid_sample_op,
        grid_sample_claim,
        crate::registry::MLX_SLICE_SHAPE_REASON,
    );
    registry.register_shape_keyed(
        OpRegistration {
            domain: "",
            op_type: "AffineGrid",
            min_opset: 20,
            max_opset: K_ANY_OPSET,
            handler: affine_grid_op,
            claim: affine_grid_claim,
        },
        "host-generated homogeneous grid depends on size values",
    );
    shape_keyed(
        registry,
        "Col2Im",
        col2im_op,
        col2im_claim,
        crate::registry::MLX_SCATTER_SHAPE_REASON,
    );
    shape_keyed(
        registry,
        "RoiAlign",
        roi_align_op,
        roi_align_claim,
        crate::registry::MLX_SLICE_SHAPE_REASON,
    );
    shape_keyed(
        registry,
        "MaxRoiPool",
        max_roi_pool_op,
        max_roi_pool_claim,
        crate::registry::MLX_SLICE_SHAPE_REASON,
    );
    shape_keyed(
        registry,
        "MaxUnpool",
        max_unpool_op,
        max_unpool_claim,
        crate::registry::MLX_SCATTER_SHAPE_REASON,
    );
    shapeless(
        registry,
        "NonMaxSuppression",
        non_max_suppression_op,
        non_max_suppression_claim,
    );
}

#[cfg(test)]
mod tests {
    use super::affine_grid_shape;

    #[test]
    fn affine_grid_geometry_is_checked_before_indexing_or_allocation() {
        assert_eq!(
            affine_grid_shape(&[2, 3, 4], &[2, 0, 1, 2, 3]).unwrap(),
            [2, 1, 2, 3, 3]
        );
        assert_eq!(
            affine_grid_shape(&[0, 2, 3], &[0, 0, 2, 3]).unwrap(),
            [0, 2, 3, 2]
        );
        assert_eq!(
            affine_grid_shape(&[2, 2, 3], &[2, 1 << 40, 2, 3]).unwrap(),
            [2, 2, 3, 2]
        );
        for (theta, size) in [
            (vec![2, 2, 3], vec![2, 1, 2]),
            (vec![2, 2], vec![2, 1, 2, 3]),
            (vec![2, 3, 4], vec![2, 1, 2, 3]),
            (vec![2, 2, 3], vec![3, 1, 2, 3]),
            (vec![2, 2, 3], vec![2, -1, 2, 3]),
            (vec![2, 2, 3], vec![2, 1, 0, 3]),
            (vec![2, 2, 3], vec![2, 1, 1 << 32, 3]),
            (vec![2, 2, 3], vec![2, 1, 1 << 30, 3]),
            (vec![2, 3, 4], vec![2, 1, i64::MAX, i64::MAX, i64::MAX]),
        ] {
            assert!(
                affine_grid_shape(&theta, &size).is_err(),
                "{theta:?} / {size:?}"
            );
        }
    }
}
