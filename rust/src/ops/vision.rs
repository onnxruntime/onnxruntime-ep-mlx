//! Vision / spatial-transform op handlers. Faithful port of the C++ `ops/vision2.cc`
//! (GridSample, AffineGrid, Col2Im) and `ops/vision2b.cc` (RoiAlign, MaxRoiPool, MaxUnpool).
//!
//! Every op reduces to the same primitives the shape handlers already use — host-computed
//! gather/scatter index maps + `take`/`take_along_axis`/`scatter_add` + on-device coordinate
//! arithmetic — so they translate exactly to MLX with no bespoke kernel. ROI pooling and MaxUnpool
//! accept symbolic extents; host-read MaxUnpool/Col2Im controls stay eager unless constant.
//! Col2Im preserves integer/bool payloads with an inverse gather map and same-dtype add/OR;
//! float64 uses that map on the existing MLX CPU-stream route without narrowing.
//! GridSample uses tensor-product gathers for small ranks and separable interpolation for larger
//! ranks. Nearest sampling preserves numeric/bool payloads without floating-point conversion.
//! Adaptive RoiAlign reads live ROI geometry eagerly; fixed grids retain shape-keyed compilation.
//! NonMaxSuppression is host-computed because MLX has no data-dependent compacting primitive.

use std::cmp::Ordering;
use std::os::raw::c_void;

use crate::engine::{MlxError, NodeDesc, Src, TensorRef, TranslationContext, mlx_dtype_from_onnx};
use crate::mlx::{Array, VectorArray};
use crate::registry::{
    ClaimResult, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry, is_int_index, is_mlx_cpu_float,
    is_mlx_float, is_movable,
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
fn out_dtype(n: &NodeDesc) -> mlx::mlx_dtype {
    mlx_dtype_from_onnx(n.outputs[0].otype)
}

// =================================================================================================
// GridSample
// =================================================================================================

fn gs_product(dims: &[i32]) -> Result<i32, MlxError> {
    if dims.iter().any(|&d| d < 0) {
        return Err("GridSample: dimensions must be nonnegative".into());
    }
    if dims.contains(&0) {
        return Ok(0);
    }
    dims.iter().try_fold(1i32, |a, &b| {
        a.checked_mul(b)
            .ok_or_else(|| "GridSample: element count exceeds int32 indexing".into())
    })
}

fn gs_shape(x: &[i32], grid: &[i32]) -> Result<Vec<i32>, MlxError> {
    if x.len() < 3 || grid.len() != x.len() || grid.last() != Some(&((x.len() - 2) as i32)) {
        return Err(
            "GridSample: expected X[N,C,spatial...] and grid[N,output...,spatial_rank]".into(),
        );
    }
    if x[0] != grid[0] {
        return Err("GridSample: grid and input batch sizes must match".into());
    }
    if x[2..].iter().any(|&d| d <= 0) {
        return Err("GridSample: input spatial dimensions must be positive".into());
    }
    let mut out = vec![x[0], x[1]];
    out.extend_from_slice(&grid[1..grid.len() - 1]);
    let count = gs_product(&out)?;
    out.iter().rev().try_fold(1i64, |stride, &dim| {
        stride
            .checked_mul(i64::from(dim))
            .ok_or("GridSample: output strides exceed int64")
    })?;
    if count != 0 {
        gs_product(x)?;
        gs_product(grid)?;
    }
    Ok(out)
}

fn gs_scalar(ctx: &mut TranslationContext, value: f64, double: bool) -> mlx::mlx_array {
    if double {
        ctx.scalar_f64(value)
    } else {
        ctx.scalar_f32(value as f32)
    }
}

fn gs_clip(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    lo: f64,
    hi: f64,
    double: bool,
) -> Result<mlx::mlx_array, MlxError> {
    let lo = gs_scalar(ctx, lo, double);
    let hi = gs_scalar(ctx, hi, double);
    ctx.emit(|res, s| unsafe { mlx::mlx_clip(res, x, lo, hi, s) })
}

fn gs_reflect(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    dim: i32,
    align: bool,
    double: bool,
) -> Result<mlx::mlx_array, MlxError> {
    if dim == 1 {
        let zero = gs_scalar(ctx, 0.0, double);
        return mul(ctx, x, zero);
    }
    let lo = gs_scalar(ctx, if align { 0.0 } else { -0.5 }, double);
    let span = gs_scalar(ctx, f64::from(if align { dim - 1 } else { dim }), double);
    let two_span = gs_scalar(
        ctx,
        2.0 * f64::from(if align { dim - 1 } else { dim }),
        double,
    );
    // Reduce before adding the half-pixel border: at large magnitudes x + 0.5 can round back
    // to x, changing which reflected integer pixel is selected.
    let reduced = if align {
        x
    } else {
        ctx.binary(mlx::mlx_remainder, x, two_span)?
    };
    let shifted = sub(ctx, reduced, lo)?;
    // A floating remainder avoids an overflowing integer quotient for far-away grids.
    let phase = ctx.binary(mlx::mlx_remainder, shifted, two_span)?;
    let dist = sub(ctx, phase, span)?;
    let dist = abs_(ctx, dist)?;
    let reflected = sub(ctx, span, dist)?;
    add(ctx, reflected, lo)
}

struct GsTap {
    index: mlx::mlx_array,
    weight: mlx::mlx_array,
    valid: mlx::mlx_array,
}

fn gs_cubic_weight(
    ctx: &mut TranslationContext,
    distance: mlx::mlx_array,
    outer: bool,
    double: bool,
) -> Result<mlx::mlx_array, MlxError> {
    let coefficients = if outer {
        [-0.75, 3.75, -6.0, 3.0]
    } else {
        [1.25, -2.25, 0.0, 1.0]
    };
    let mut weight = gs_scalar(ctx, coefficients[0], double);
    for coefficient in &coefficients[1..] {
        weight = mul(ctx, weight, distance)?;
        let coefficient = gs_scalar(ctx, *coefficient, double);
        weight = add(ctx, weight, coefficient)?;
    }
    Ok(weight)
}

/// Float32 cannot represent every pixel in a long axis. Decode its exact binary rational so
/// index rounding and bounds stay integer operations; only interpolation weights narrow to f32.
fn gs_large_axis_taps(
    ctx: &mut TranslationContext,
    coord: mlx::mlx_array,
    dim: i32,
    mode: &str,
    padding: &str,
    align: bool,
) -> Result<Vec<GsTap>, MlxError> {
    let bits =
        ctx.emit(|res, s| unsafe { mlx::mlx_view(res, coord, mlx::mlx_dtype__MLX_UINT32, s) })?;
    let bits = astype(ctx, bits, mlx::mlx_dtype__MLX_INT64)?;
    let zero = ctx.scalar_i64(0);
    let one = ctx.scalar_i64(1);
    let magnitude_mask = ctx.scalar_i64(0x7fff_ffff);
    let magnitude = ctx.binary(mlx::mlx_bitwise_and, bits, magnitude_mask)?;
    let infinity = ctx.scalar_i64(0x7f80_0000);
    let nonfinite = ctx.binary(mlx::mlx_greater_equal, magnitude, infinity)?;
    let minus_one = ctx.scalar_i64(0xbf80_0000);
    let bits = where_(ctx, nonfinite, minus_one, bits)?;
    let bits = if padding == "reflection" {
        bits
    } else {
        let magnitude = ctx.binary(mlx::mlx_bitwise_and, bits, magnitude_mask)?;
        let eight = ctx.scalar_i64(0x4100_0000);
        let magnitude = ctx.binary(mlx::mlx_minimum, magnitude, eight)?;
        let sign_mask = ctx.scalar_i64(0x8000_0000);
        let sign = ctx.binary(mlx::mlx_bitwise_and, bits, sign_mask)?;
        ctx.binary(mlx::mlx_bitwise_or, magnitude, sign)?
    };
    let shift = ctx.scalar_i64(23);
    let exponent = ctx.binary(mlx::mlx_right_shift, bits, shift)?;
    let mask = ctx.scalar_i64(255);
    let exponent = ctx.binary(mlx::mlx_bitwise_and, exponent, mask)?;
    let normal = ctx.binary(mlx::mlx_not_equal, exponent, zero)?;
    let hidden = ctx.scalar_i64(1 << 23);
    let hidden = where_(ctx, normal, hidden, zero)?;
    let mask = ctx.scalar_i64((1 << 23) - 1);
    let mantissa = ctx.binary(mlx::mlx_bitwise_and, bits, mask)?;
    let mantissa = add(ctx, mantissa, hidden)?;
    let shift = ctx.scalar_i64(31);
    let sign = ctx.binary(mlx::mlx_right_shift, bits, shift)?;
    let negative = ctx.binary(mlx::mlx_not_equal, sign, zero)?;
    let negated = sub(ctx, zero, mantissa)?;
    let mantissa = where_(ctx, negative, negated, mantissa)?;
    let bias = ctx.scalar_i64(150);
    let power = sub(ctx, bias, exponent)?;
    let subnormal_power = ctx.scalar_i64(149);
    let power = where_(ctx, normal, power, subnormal_power)?;
    let (mantissa, power) = if padding == "reflection" {
        // Reduce the rational modulo four without a floating operation: Metal's denormal
        // flushing must not erase the sign of a subnormal grid next to a nearest half-tie.
        let four_exponent = ctx.scalar_i64(129);
        let far = ctx.binary(mlx::mlx_greater_equal, exponent, four_exponent)?;
        let nonnegative_power = ctx.binary(mlx::mlx_maximum, power, zero)?;
        let cap = ctx.scalar_i64(21);
        let capped_power = ctx.binary(mlx::mlx_minimum, nonnegative_power, cap)?;
        let four = ctx.scalar_i64(4);
        let period = ctx.binary(mlx::mlx_left_shift, four, capped_power)?;
        let negative_power = sub(ctx, zero, power)?;
        let shift = ctx.binary(mlx::mlx_maximum, negative_power, zero)?;
        let shift = ctx.binary(mlx::mlx_minimum, shift, one)?;
        let magnitude = abs_(ctx, mantissa)?;
        let shifted = ctx.binary(mlx::mlx_left_shift, magnitude, shift)?;
        let reduced = ctx.binary(mlx::mlx_remainder, shifted, period)?;
        let too_large = ctx.scalar_i64(152);
        let too_large = ctx.binary(mlx::mlx_greater_equal, exponent, too_large)?;
        let reduced = where_(ctx, too_large, zero, reduced)?;
        // Restore the sign after magnitude reduction.
        let negated = sub(ctx, zero, reduced)?;
        let reduced = where_(ctx, negative, negated, reduced)?;
        (
            where_(ctx, far, reduced, mantissa)?,
            where_(ctx, far, nonnegative_power, power)?,
        )
    } else {
        (mantissa, power)
    };
    let scale = ctx.scalar_i64(i64::from(if align { dim - 1 } else { dim }));
    let product = mul(ctx, mantissa, scale)?;
    let limit = ctx.scalar_i64(55);
    let tiny = ctx.binary(mlx::mlx_greater, power, limit)?;
    let capped = ctx.binary(mlx::mlx_minimum, power, limit)?;
    let half_denominator = ctx.binary(mlx::mlx_left_shift, one, capped)?;
    let denominator = add(ctx, half_denominator, half_denominator)?;
    let center = i64::from(dim - 1) / 2;
    let center_array = ctx.scalar_i64(center);
    let numerator = if (dim - 1) % 2 != 0 {
        add(ctx, product, half_denominator)?
    } else {
        product
    };
    let quotient = ctx.binary(mlx::mlx_floor_divide, numerator, denominator)?;
    let multiple = mul(ctx, quotient, denominator)?;
    let remainder = sub(ctx, numerator, multiple)?;
    let base = add(ctx, center_array, quotient)?;
    let greater_half = ctx.binary(mlx::mlx_greater, remainder, half_denominator)?;
    let equal_half = ctx.binary(mlx::mlx_equal, remainder, half_denominator)?;
    let odd = ctx.binary(mlx::mlx_bitwise_and, base, one)?;
    let odd = ctx.binary(mlx::mlx_not_equal, odd, zero)?;
    let odd_tie = ctx.binary(mlx::mlx_logical_and, equal_half, odd)?;
    let increment = ctx.binary(mlx::mlx_logical_or, greater_half, odd_tie)?;
    let increment = astype(ctx, increment, mlx::mlx_dtype__MLX_INT64)?;
    let nearest = add(ctx, base, increment)?;
    let remainder_f = astype(ctx, remainder, mlx::mlx_dtype__MLX_FLOAT32)?;
    let denominator_f = astype(ctx, denominator, mlx::mlx_dtype__MLX_FLOAT32)?;
    let fraction = div(ctx, remainder_f, denominator_f)?;

    // For very small grids the rational denominator need not fit int64. The perturbation
    // is strictly less than half a pixel; its sign still decides a half-center nearest tie.
    let product_negative = ctx.binary(mlx::mlx_less, product, zero)?;
    let product_positive = ctx.binary(mlx::mlx_greater, product, zero)?;
    let product_zero = ctx.binary(mlx::mlx_equal, product, zero)?;
    let tiny_floor_offset = if (dim - 1) % 2 == 0 {
        astype(ctx, product_negative, mlx::mlx_dtype__MLX_INT64)?
    } else {
        zero
    };
    let tiny_base = sub(ctx, center_array, tiny_floor_offset)?;
    let tiny_increment = if (dim - 1) % 2 != 0 {
        let tie = if center % 2 != 0 {
            product_zero
        } else {
            ctx.scalar_bool(false)
        };
        let increment = ctx.binary(mlx::mlx_logical_or, product_positive, tie)?;
        astype(ctx, increment, mlx::mlx_dtype__MLX_INT64)?
    } else {
        zero
    };
    let tiny_nearest = add(ctx, center_array, tiny_increment)?;
    let scale_f = ctx.scalar_f32(if align { (dim - 1) as f32 } else { dim as f32 } * 0.5);
    let delta = mul(ctx, coord, scale_f)?;
    let half_center = ctx.scalar_f32(if (dim - 1) % 2 != 0 { 0.5 } else { 0.0 });
    let tiny_fraction = add(ctx, delta, half_center)?;
    let floor_offset_f = astype(ctx, tiny_floor_offset, mlx::mlx_dtype__MLX_FLOAT32)?;
    let tiny_fraction = add(ctx, tiny_fraction, floor_offset_f)?;
    let base = where_(ctx, tiny, tiny_base, base)?;
    let nearest = where_(ctx, tiny, tiny_nearest, nearest)?;
    let fraction = where_(ctx, tiny, tiny_fraction, fraction)?;
    let base = if mode == "nearest" { nearest } else { base };
    let count = if mode == "nearest" {
        1
    } else if mode == "cubic" {
        4
    } else {
        2
    };
    let last = ctx.scalar_i64(i64::from(dim - 1));
    let period = ctx.scalar_i64(2 * i64::from(if align { dim - 1 } else { dim }));
    let reflected_last =
        ctx.scalar_i64(2 * i64::from(if align { dim - 1 } else { dim }) - i64::from(!align));
    let mut taps = Vec::with_capacity(count);
    for tap in 0..count {
        let offset = tap as i64 - i64::from(count == 4);
        let offset_i = ctx.scalar_i64(offset);
        let position = add(ctx, base, offset_i)?;
        let ge = ctx.binary(mlx::mlx_greater_equal, position, zero)?;
        let le = ctx.binary(mlx::mlx_less_equal, position, last)?;
        let valid = ctx.binary(mlx::mlx_logical_and, ge, le)?;
        let index = if padding == "reflection" {
            let phase = ctx.binary(mlx::mlx_remainder, position, period)?;
            let other = sub(ctx, reflected_last, phase)?;
            ctx.binary(mlx::mlx_minimum, phase, other)?
        } else {
            position
        };
        let index = ctx.binary(mlx::mlx_maximum, index, zero)?;
        let index = ctx.binary(mlx::mlx_minimum, index, last)?;
        let index = astype(ctx, index, mlx::mlx_dtype__MLX_INT32)?;
        let offset_f = ctx.scalar_f32(offset as f32);
        let distance = sub(ctx, fraction, offset_f)?;
        let distance = abs_(ctx, distance)?;
        let one_f = ctx.scalar_f32(1.0);
        let weight = match count {
            1 => one_f,
            4 => gs_cubic_weight(ctx, distance, tap == 0 || tap == 3, false)?,
            _ => sub(ctx, one_f, distance)?,
        };
        taps.push(GsTap {
            index,
            weight,
            valid,
        });
    }
    Ok(taps)
}

#[allow(clippy::too_many_arguments)]
fn gs_taps(
    ctx: &mut TranslationContext,
    coord: mlx::mlx_array,
    dim: i32,
    mode: &str,
    padding: &str,
    align: bool,
    double: bool,
) -> Result<Vec<GsTap>, MlxError> {
    if !double && dim > (1 << 24) {
        return gs_large_axis_taps(ctx, coord, dim, mode, padding, align);
    }
    let finite = ctx.unary(mlx::mlx_isfinite, coord)?;
    let minus_one = gs_scalar(ctx, -1.0, double);
    let coord = where_(ctx, finite, coord, minus_one)?;
    let coord = if padding == "reflection" {
        // Signed reduction preserves small negative nearest-threshold coordinates exactly.
        // The period is four for either alignment, independent of the input spatial extent.
        let magnitude = abs_(ctx, coord)?;
        let period = gs_scalar(ctx, 4.0, double);
        let reduced = ctx.binary(mlx::mlx_remainder, magnitude, period)?;
        let sign = sign(ctx, coord)?;
        mul(ctx, reduced, sign)?
    } else {
        // Outside this normalized interval every cubic/linear/nearest tap is outside the image
        // (except aligned singleton axes, which always map to zero). This preserves padding
        // while avoiding overflow in denormalization, even for the largest finite grid value.
        gs_clip(ctx, coord, -8.0, 8.0, double)?
    };
    let one = gs_scalar(ctx, 1.0, double);
    let shifted = add(ctx, coord, one)?;
    let scale = gs_scalar(ctx, f64::from(if align { dim - 1 } else { dim }), double);
    let scaled = mul(ctx, shifted, scale)?;
    let scaled = if align {
        scaled
    } else {
        sub(ctx, scaled, one)?
    };
    let half = gs_scalar(ctx, 0.5, double);
    let coord = mul(ctx, scaled, half)?;
    // For interpolation, reducing the continuous coordinate is equivalent to padding every tap,
    // and keeps the unit offsets representable even when a finite grid is very far outside.
    let coord = if mode != "nearest" && padding == "reflection" {
        gs_reflect(ctx, coord, dim, align, double)?
    } else if padding == "border" {
        // Cubic can overshoot near the edge: only collapse coordinates whose entire support
        // lies outside. Clamping to the image itself would silently turn it into edge replication.
        gs_clip(ctx, coord, -2.0, f64::from(dim) + 1.0, double)?
    } else {
        coord
    };
    let base = if mode == "nearest" {
        round0(ctx, coord)?
    } else {
        floor(ctx, coord)?
    };
    let fraction = sub(ctx, coord, base)?;
    let count = if mode == "nearest" {
        1
    } else if mode == "cubic" {
        4
    } else {
        2
    };
    let mut taps = Vec::with_capacity(count);
    for tap in 0..count {
        let offset = gs_scalar(ctx, tap as f64 - if count == 4 { 1.0 } else { 0.0 }, double);
        let position = add(ctx, base, offset)?;
        let weight = if count == 1 {
            one
        } else {
            let distance = sub(ctx, fraction, offset)?;
            let distance = abs_(ctx, distance)?;
            if count == 4 {
                gs_cubic_weight(ctx, distance, tap == 0 || tap == 3, double)?
            } else {
                sub(ctx, one, distance)?
            }
        };
        let zero = gs_scalar(ctx, 0.0, double);
        let last = gs_scalar(ctx, f64::from(dim - 1), double);
        let ge = ctx.binary(mlx::mlx_greater_equal, position, zero)?;
        let le = ctx.binary(mlx::mlx_less_equal, position, last)?;
        let valid = ctx.binary(mlx::mlx_logical_and, ge, le)?;
        let position = if padding == "reflection" {
            gs_reflect(ctx, position, dim, align, double)?
        } else {
            position
        };
        let position = gs_clip(ctx, position, 0.0, f64::from(dim - 1), double)?;
        let index = astype(ctx, position, mlx::mlx_dtype__MLX_INT32)?;
        taps.push(GsTap {
            index,
            weight,
            valid,
        });
    }
    Ok(taps)
}

fn gs_gather(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    indices: mlx::mlx_array,
    valid: mlx::mlx_array,
    zeros: bool,
    shape: &[i32],
    axis: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let indices = broadcast_to(ctx, indices, shape)?;
    let value = take_along_axis(ctx, x, indices, axis)?;
    if zeros {
        let zero = scalar_i(ctx, 0);
        let dtype = ctx.dtype_of(x);
        let zero = astype(ctx, zero, dtype)?;
        where_(ctx, valid, value, zero)
    } else {
        Ok(value)
    }
}

fn grid_sample_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let mut shapes = Vec::with_capacity(2);
    for input in &n.inputs {
        let shape = ctx.tensor_shape(input)?;
        let shape: Vec<i32> = shape
            .into_iter()
            .map(|d| i32::try_from(d).map_err(|_| "GridSample: dimension exceeds int32"))
            .collect::<Result<_, _>>()?;
        shapes.push(shape);
    }
    let (xs, gs) = (&shapes[0], &shapes[1]);
    let out_shape = gs_shape(&xs, &gs)?;
    let rank = xs.len() - 2;
    let (nn, cc) = (xs[0], xs[1]);
    if gs_product(&out_shape)? == 0 {
        let out = ctx.emit(|res, s| unsafe {
            mlx::mlx_zeros(res, out_shape.as_ptr(), out_shape.len(), out_dtype(n), s)
        })?;
        ctx.bind(&n.outputs[0], out);
        return Ok(());
    }
    let p = gs_product(&gs[1..gs.len() - 1])?;
    let x = ctx.resolve(&n.inputs[0])?;
    let grid = ctx.resolve(&n.inputs[1])?;
    let mode = n
        .strings
        .get("mode")
        .map(String::as_str)
        .unwrap_or("linear");
    let mode = match mode {
        "bilinear" => "linear",
        "bicubic" => "cubic",
        other => other,
    };
    let padding = n
        .strings
        .get("padding_mode")
        .map(String::as_str)
        .unwrap_or("zeros");
    let align = n.ints.get("align_corners").copied().unwrap_or(0) == 1;
    let nearest = mode == "nearest";
    let double = ctx.dtype_of(x) == mlx::mlx_dtype__MLX_FLOAT64
        || ctx.dtype_of(grid) == mlx::mlx_dtype__MLX_FLOAT64;
    let calc_dtype = if double {
        mlx::mlx_dtype__MLX_FLOAT64
    } else {
        mlx::mlx_dtype__MLX_FLOAT32
    };
    let grid = astype(ctx, grid, calc_dtype)?;
    let grid = reshape(ctx, grid, &[nn, p, rank as i32])?;
    let mut taps = Vec::with_capacity(rank);
    for axis in 0..rank {
        let component = (rank - axis - 1) as i32;
        let coord = slice(ctx, grid, &[0, 0, component], &[nn, p, component + 1])?;
        let coord = reshape(ctx, coord, &[nn, p])?;
        taps.push(gs_taps(
            ctx,
            coord,
            xs[axis + 2],
            mode,
            padding,
            align,
            double,
        )?);
    }
    let x = if nearest {
        x
    } else {
        astype(ctx, x, calc_dtype)?
    };
    let acc = if nearest || rank <= 3 {
        // Keep the usual 2D/3D forms proportional to the number of output points, not input rows.
        let spatial = gs_product(&xs[2..])?;
        let x = reshape(ctx, x, &[nn, cc, spatial])?;
        let branches = taps.iter().map(Vec::len).product::<usize>();
        let mut acc = None;
        for branch in 0..branches {
            let mut remaining = branch;
            let mut index = scalar_i(ctx, 0);
            let mut weight = gs_scalar(ctx, 1.0, double);
            let mut valid = ctx.scalar_bool(true);
            for (axis, axis_taps) in taps.iter().enumerate() {
                let tap = &axis_taps[remaining % axis_taps.len()];
                remaining /= axis_taps.len();
                let dim = scalar_i(ctx, xs[axis + 2]);
                index = mul(ctx, index, dim)?;
                index = add(ctx, index, tap.index)?;
                if !nearest {
                    weight = mul(ctx, weight, tap.weight)?;
                }
                valid = ctx.binary(mlx::mlx_logical_and, valid, tap.valid)?;
            }
            let index = reshape(ctx, index, &[nn, 1, p])?;
            let valid = reshape(ctx, valid, &[nn, 1, p])?;
            let mut value = gs_gather(ctx, x, index, valid, padding == "zeros", &[nn, cc, p], 2)?;
            if !nearest {
                let weight = reshape(ctx, weight, &[nn, 1, p])?;
                value = mul(ctx, value, weight)?;
            }
            acc = Some(match acc {
                None => value,
                Some(previous) => add(ctx, previous, value)?,
            });
        }
        acc.ok_or("GridSample: no sampling branches")?
    } else {
        // Separable contraction avoids 2^rank/4^rank graph growth. Contract the largest input
        // axes first to minimize the live [N,C,P,remaining spatial...] intermediate.
        let mut order: Vec<usize> = (0..rank).collect();
        order.sort_by_key(|&axis| std::cmp::Reverse(xs[axis + 2]));
        let mut perm = vec![0, 1];
        perm.extend(order.iter().map(|&axis| axis as i32 + 2));
        let mut value = ctx.transpose(x, &perm)?;
        let mut remaining = gs_product(&xs[2..])?;
        for (step, &axis) in order.iter().enumerate() {
            let dim = xs[axis + 2];
            remaining /= dim;
            let gather_shape = [nn, cc, p, 1, remaining];
            gs_product(&gather_shape)?;
            let source = if step == 0 {
                let source = reshape(ctx, value, &[nn, cc, 1, dim, remaining])?;
                broadcast_to(ctx, source, &[nn, cc, p, dim, remaining])?
            } else {
                reshape(ctx, value, &[nn, cc, p, dim, remaining])?
            };
            let mut acc = None;
            for tap in &taps[axis] {
                let index = reshape(ctx, tap.index, &[nn, 1, p, 1, 1])?;
                let valid = reshape(ctx, tap.valid, &[nn, 1, p, 1, 1])?;
                let sample = gs_gather(
                    ctx,
                    source,
                    index,
                    valid,
                    padding == "zeros",
                    &gather_shape,
                    3,
                )?;
                let weight = reshape(ctx, tap.weight, &[nn, 1, p, 1, 1])?;
                let sample = mul(ctx, sample, weight)?;
                acc = Some(match acc {
                    None => sample,
                    Some(previous) => add(ctx, previous, sample)?,
                });
            }
            value = acc.ok_or("GridSample: no separable sampling taps")?;
        }
        value
    };
    let out = reshape(ctx, acc, &out_shape)?;
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
        is_movable(x.dtype) && out.dtype == x.dtype && is_mlx_cpu_float(g.dtype),
        "input/output must share one numeric/bool dtype and grid must be floating, got {} / {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(g.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() >= 3
            && g.shape.len() == x.shape.len()
            && g.shape
                .last()
                .is_some_and(|&d| d < 0 || d == x.shape.len() as i64 - 2)
            && out.shape.len() == x.shape.len(),
        "input/grid/output ranks or grid coordinate width disagree: {:?} and {:?}",
        x.shape,
        g.shape
    );
    require!(
        node.since_version() >= 20 || x.shape.len() == 4,
        "opset 16 GridSample only defines 2D sampling"
    );
    let legacy = node.since_version() < 20;
    let mode = node.string_attr("mode", if legacy { "bilinear" } else { "linear" });
    require!(
        mode == "nearest"
            || if legacy {
                mode == "bilinear" || mode == "bicubic"
            } else {
                mode == "linear" || mode == "cubic"
            },
        "invalid GridSample mode {mode:?} for schema {}",
        node.since_version()
    );
    require!(
        mode == "nearest" || is_mlx_cpu_float(x.dtype),
        "non-floating linear/cubic interpolation has no agreed intermediate/cast contract"
    );
    require!(
        node.since_version() >= 22
            || (x.dtype != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
                && g.dtype
                    != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
        "bfloat16 GridSample requires opset 22"
    );
    let padding = node.string_attr("padding_mode", "zeros");
    require!(
        matches!(padding.as_str(), "zeros" | "border" | "reflection"),
        "invalid GridSample padding_mode {padding:?}"
    );
    require!(
        matches!(node.int_attr("align_corners", 0), 0 | 1),
        "align_corners must be 0 or 1"
    );
    let compatible = |a: i64, b: i64| a < 0 || b < 0 || a == b;
    require!(
        compatible(x.shape[0], g.shape[0])
            && compatible(out.shape[0], x.shape[0])
            && compatible(out.shape[1], x.shape[1])
            && out.shape[2..]
                .iter()
                .zip(&g.shape[1..g.shape.len() - 1])
                .all(|(&a, &b)| compatible(a, b)),
        "GridSample input/grid/output shapes disagree"
    );
    require!(
        x.shape[2..].iter().all(|&d| d != 0),
        "input spatial dimensions must be positive"
    );
    for shape in [&x.shape, &g.shape, &out.shape] {
        require!(
            shape.iter().all(|&d| d <= i64::from(i32::MAX)),
            "GridSample dimension exceeds int32"
        );
    }
    if x.shape.iter().chain(&g.shape).all(|&d| d >= 0) {
        let x: Vec<i32> = x.shape.iter().map(|&d| d as i32).collect();
        let g: Vec<i32> = g.shape.iter().map(|&d| d as i32).collect();
        require!(
            gs_shape(&x, &g).is_ok(),
            "invalid or oversized GridSample geometry"
        );
    }
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

struct Col2ImGeometry {
    block_size: i64,
    spatial_size: i64,
    channels: i32,
    npos: Vec<i64>,
    image_strides: Vec<i64>,
    output_shape: Vec<i32>,
}

impl Col2ImGeometry {
    fn new(
        input: &[i64],
        image: &[i64],
        block: &[i64],
        dilations: &[i64],
        strides: &[i64],
        pads: &[i64],
    ) -> Result<Self, MlxError> {
        let rank = image.len();
        if !(1..=3).contains(&rank)
            || input.len() != 3
            || block.len() != rank
            || dilations.len() != rank
            || strides.len() != rank
            || pads.len() != 2 * rank
            || input.iter().any(|&v| v < 0 || v > i32::MAX as i64)
            || image
                .iter()
                .chain(block)
                .chain(dilations)
                .chain(strides)
                .any(|&v| v <= 0 || v > i32::MAX as i64)
            || pads.iter().any(|&v| v < 0 || v > i32::MAX as i64)
        {
            return Err("Col2Im: invalid dimensions or window attributes".into());
        }
        let product = |values: &[i64]| -> Result<i64, MlxError> {
            values.iter().try_fold(1i64, |p, &v| {
                p.checked_mul(v)
                    .filter(|&p| p <= i32::MAX as i64)
                    .ok_or_else(|| "Col2Im: flattened size exceeds int32".into())
            })
        };
        let block_size = product(block)?;
        let spatial_size = product(image)?;
        if input[1] == 0 || input[1] % block_size != 0 {
            return Err("Col2Im: channels must be a positive multiple of block size".into());
        }
        let channels = input[1] / block_size;
        let mut npos = Vec::with_capacity(rank);
        for d in 0..rank {
            let effective = dilations[d] * (block[d] - 1) + 1;
            let padded = image[d] + pads[d] + pads[d + rank];
            let count = (padded - effective).div_euclid(strides[d]) + 1;
            if count < 0 {
                return Err("Col2Im: padded image is smaller than effective block".into());
            }
            npos.push(count);
        }
        if input[2] != product(&npos)? {
            return Err("Col2Im: input length must equal product of sliding block counts".into());
        }
        product(input)?;
        product(&[input[0], channels, spatial_size])?;
        product(&[block_size, input[2]])?;
        // The scatter map has at most one entry per input element.
        let mut image_strides = vec![1; rank];
        for d in (0..rank - 1).rev() {
            image_strides[d] = image_strides[d + 1] * image[d + 1];
        }
        let mut output_shape = vec![input[0] as i32, channels as i32];
        output_shape.extend(image.iter().map(|&v| v as i32));
        Ok(Self {
            block_size,
            spatial_size,
            channels: channels as i32,
            npos,
            image_strides,
            output_shape,
        })
    }
}

// Metal scatter reductions reject 64-bit integers. Inverting the map gives each output one
// contribution per kernel position, so ordinary same-dtype add/OR preserves integer and bool bits.
// The same map avoids fp32 accumulation for float64 on the MLX CPU stream.
fn col2im_exact(
    ctx: &mut TranslationContext,
    input: mlx::mlx_array,
    geometry: &Col2ImGeometry,
    image: &[i64],
    block: &[i64],
    dilations: &[i64],
    strides: &[i64],
    pads: &[i64],
) -> Result<mlx::mlx_array, MlxError> {
    let shape = ctx.shape_of(input);
    let dtype = ctx.dtype_of(input);
    let size = geometry.spatial_size;
    let flat = ctx.reshape(
        input,
        &[
            shape[0],
            geometry.channels,
            geometry.block_size as i32 * shape[2],
        ],
    )?;
    let out_shape = [shape[0], geometry.channels, size as i32];
    let zero = ctx.emit(|res, s| unsafe {
        mlx::mlx_zeros(res, out_shape.as_ptr(), out_shape.len(), dtype, s)
    })?;
    let mut output = zero;
    for kernel in 0..geometry.block_size {
        let mut kc = vec![0; image.len()];
        let mut remainder = kernel;
        for d in (0..image.len()).rev() {
            kc[d] = remainder % block[d];
            remainder /= block[d];
        }
        let mut indices = Vec::with_capacity(size as usize);
        let mut valid = Vec::with_capacity(size as usize);
        for spatial in 0..size {
            let mut index = 0;
            let mut ok = true;
            for d in 0..image.len() {
                let coordinate = (spatial / geometry.image_strides[d]) % image[d];
                let position = coordinate + pads[d] - kc[d] * dilations[d];
                if position < 0
                    || position % strides[d] != 0
                    || position / strides[d] >= geometry.npos[d]
                {
                    ok = false;
                    break;
                }
                index = index * geometry.npos[d] + position / strides[d];
            }
            indices.push(if ok {
                (kernel * i64::from(shape[2]) + index) as i32
            } else {
                0
            });
            valid.push(u8::from(ok));
        }
        let index = host_i32(ctx, &indices, &[size as i32]);
        let mask = ctx.from_host(
            valid.as_ptr() as *const c_void,
            &[size as i32],
            mlx::mlx_dtype__MLX_BOOL,
        );
        let values = take_axis(ctx, flat, index, 2)?;
        let values = where_(ctx, mask, values, zero)?;
        output = ctx.binary(
            if dtype == mlx::mlx_dtype__MLX_BOOL {
                mlx::mlx_logical_or
            } else {
                mlx::mlx_add
            },
            output,
            values,
        )?;
    }
    ctx.reshape(output, &geometry.output_shape)
}

fn col2im_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let in0 = ctx.resolve(&n.inputs[0])?;
    let input = contiguous(ctx, in0)?;
    let is = ctx.shape_of(input);
    let nn = is[0];
    let l = is[2];

    let image_shape = ctx.read_ints_eval(&n.inputs[1])?;
    let block_shape = ctx.read_ints_eval(&n.inputs[2])?;
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

    let geometry = Col2ImGeometry::new(
        &is.iter().map(|&v| i64::from(v)).collect::<Vec<_>>(),
        &image_shape,
        &block_shape,
        &dil,
        &stride,
        &pads,
    )?;
    for index in [1, 2] {
        let control = ctx.resolve(&n.inputs[index])?;
        if ctx.shape_of(control) != [r as i32] {
            return Err(
                "Col2Im: image_shape and block_shape must be matching rank-1 vectors".into(),
            );
        }
    }
    let (kk, ss, c) = (
        geometry.block_size,
        geometry.spatial_size,
        geometry.channels,
    );
    let npos = &geometry.npos;
    let img_stride = &geometry.image_strides;
    if nn == 0 || l == 0 {
        let dtype = ctx.dtype_of(input);
        let shape = &geometry.output_shape;
        let output = ctx
            .emit(|res, s| unsafe { mlx::mlx_zeros(res, shape.as_ptr(), shape.len(), dtype, s) })?;
        ctx.bind(&n.outputs[0], output);
        return Ok(());
    }
    if !matches!(
        ctx.dtype_of(input),
        mlx::mlx_dtype__MLX_FLOAT32 | mlx::mlx_dtype__MLX_FLOAT16 | mlx::mlx_dtype__MLX_BFLOAT16
    ) {
        let out = col2im_exact(
            ctx,
            input,
            &geometry,
            &image_shape,
            &block_shape,
            &dil,
            &stride,
            &pads,
        )?;
        let out = contiguous(ctx, out)?;
        ctx.bind(&n.outputs[0], out);
        return Ok(());
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

    let out = reshape(ctx, out_acc, &geometry.output_shape)?;
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
        (is_mlx_float(i.dtype) || matches!(i.dtype, 2..=7 | 9..=13)) && out.dtype == i.dtype,
        "Col2Im input/output must share a supported float, integer or bool dtype, got {} -> {}",
        crate::registry::ort_dtype_name(i.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        i.shape.len() == 3 && i.shape.iter().all(|&d| d <= i32::MAX as i64),
        "Col2Im input[0] must be rank-3 [N,C*prod(block),L] with MLX-representable dims, got shape {:?}",
        i.shape
    );
    let (image_info, block_info) = match (node.input_info(1), node.input_info(2)) {
        (Some(image), Some(block)) => (image, block),
        _ => deny!("Col2Im: missing shape control type/shape info"),
    };
    require!(
        image_info.dtype == 7
            && block_info.dtype == 7
            && image_info.shape.len() == 1
            && block_info.shape.len() == 1
            && (1..=3).contains(&image_info.shape[0])
            && block_info.shape == image_info.shape,
        "Col2Im: image_shape/block_shape must be int64 vectors of the same known length 1-3"
    );
    let r = image_info.shape[0] as usize;
    require!(
        out.shape.len() == r + 2
            && (out.shape[0] < 0 || i.shape[0] < 0 || out.shape[0] == i.shape[0]),
        "Col2Im: output rank and batch must match input geometry"
    );
    let attribute = |name: &str, count: usize, default: i64| -> Option<Vec<i64>> {
        let (present, values) = node.ints_attr(name);
        let values = if present {
            values
        } else {
            vec![default; count]
        };
        (values.len() == count && values.iter().all(|&v| v >= default && v <= i32::MAX as i64))
            .then_some(values)
    };
    let (dilations, strides, pads) = match (
        attribute("dilations", r, 1),
        attribute("strides", r, 1),
        attribute("pads", 2 * r, 0),
    ) {
        (Some(d), Some(s), Some(p)) => (d, s, p),
        _ => deny!("Col2Im: invalid strides, dilations or pads"),
    };
    let image = node.read_const_ints_any(1);
    let block = node.read_const_ints_any(2);
    for shape in [&image, &block].into_iter().flatten() {
        require!(
            shape.len() == r && shape.iter().all(|&v| v > 0 && v <= i32::MAX as i64),
            "Col2Im: image/block dimensions must be positive int32"
        );
    }
    if let (Some(image), Some(block)) = (image, block)
        && i.shape.iter().all(|&v| v >= 0)
    {
        let geometry =
            match Col2ImGeometry::new(&i.shape, &image, &block, &dilations, &strides, &pads) {
                Ok(geometry) => geometry,
                Err(reason) => deny!("{reason}"),
            };
        require!(
            out.shape
                .iter()
                .zip(&geometry.output_shape)
                .all(|(&a, &b)| a < 0 || a == i64::from(b)),
            "Col2Im: output shape does not match derived geometry"
        );
    }
    Ok(())
}

// =================================================================================================
// RoiAlign (2D)
// =================================================================================================

#[derive(Clone, Copy)]
struct RoiAlignParams {
    height: i32,
    width: i32,
    ratio: i32,
    scale: f32,
    half_pixel: bool,
    max: bool,
}

fn roi_product(dims: &[i32]) -> Result<i32, MlxError> {
    if dims.iter().any(|&d| d < 0) {
        return Err("RoiAlign: dimensions must be nonnegative".into());
    }
    if dims.contains(&0) {
        return Ok(0);
    }
    dims.iter().try_fold(1i32, |a, &b| {
        a.checked_mul(b)
            .ok_or_else(|| "RoiAlign: grid or tensor element count exceeds int32 indexing".into())
    })
}

impl RoiAlignParams {
    fn new(n: &NodeDesc) -> Result<Self, MlxError> {
        let dimension = |name, default, minimum| {
            let value = n.ints.get(name).copied().unwrap_or(default);
            i32::try_from(value)
                .ok()
                .filter(|&v| v >= minimum)
                .ok_or_else(|| format!("RoiAlign: {name} must be in [{minimum}, int32::MAX]"))
        };
        let params = Self {
            height: dimension("output_height", 1, 1)?,
            width: dimension("output_width", 1, 1)?,
            ratio: dimension("sampling_ratio", 0, 0)?,
            scale: n.floats.get("spatial_scale").copied().unwrap_or(1.0),
            half_pixel: n
                .strings
                .get("coordinate_transformation_mode")
                .map(String::as_str)
                .unwrap_or(if n.since_version < 16 {
                    "output_half_pixel"
                } else {
                    "half_pixel"
                })
                == "half_pixel",
            max: n.strings.get("mode").is_some_and(|mode| mode == "max"),
        };
        if !params.scale.is_finite() {
            return Err("RoiAlign: spatial_scale must be finite".into());
        }
        roi_product(&[
            params.height,
            params.width,
            params.ratio.max(1),
            params.ratio.max(1),
        ])?;
        Ok(params)
    }

    fn grid(self, roi: &[f64], double: bool) -> Result<[i32; 2], MlxError> {
        let extent = |start: f64, end: f64, bins: i32| -> Result<f64, MlxError> {
            let offset = if self.half_pixel { 0.5 } else { 0.0 };
            let size = if double {
                (end * f64::from(self.scale) - offset) - (start * f64::from(self.scale) - offset)
            } else {
                f64::from(
                    (end as f32 * self.scale - offset as f32)
                        - (start as f32 * self.scale - offset as f32),
                )
            };
            if !size.is_finite() {
                return Err("RoiAlign: scaled ROI extents must be finite".into());
            }
            let size = if self.half_pixel { size } else { size.max(1.0) };
            let count = if self.ratio > 0 {
                f64::from(self.ratio)
            } else if double {
                (size / f64::from(bins)).ceil()
            } else {
                f64::from((size as f32 / bins as f32).ceil())
            };
            Ok(count)
        };
        let counts = [
            extent(roi[1], roi[3], self.height)?,
            extent(roi[0], roi[2], self.width)?,
        ];
        // An empty axis makes the entire grid empty, even if the other count cannot be indexed.
        if counts.iter().any(|&count| count <= 0.0) {
            return Ok([0; 2]);
        }
        if counts.iter().any(|&count| count > f64::from(i32::MAX)) {
            return Err("RoiAlign: adaptive grid exceeds int32 indexing".into());
        }
        let grid = counts.map(|count| count as i32);
        roi_product(&[self.height, self.width, grid[0], grid[1]])?;
        Ok(grid)
    }
}

fn roi_values(ctx: &mut TranslationContext, input: &TensorRef) -> Result<Vec<f64>, MlxError> {
    let value = ctx.resolve(input)?;
    let double = ctx.dtype_of(value) == mlx::mlx_dtype__MLX_FLOAT64;
    let value = if double {
        value
    } else {
        ctx.astype(value, mlx::mlx_dtype__MLX_FLOAT32)?
    };
    let value = ctx.contiguous_eval(value)?;
    let count = ctx.size_of(value);
    if count == 0 {
        return Ok(Vec::new());
    }
    if double {
        let ptr = unsafe { mlx::mlx_array_data_float64(value) };
        if ptr.is_null() {
            return Err("RoiAlign: evaluated ROI data is unavailable".into());
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) }.to_vec())
    } else {
        let ptr = unsafe { mlx::mlx_array_data_float32(value) };
        if ptr.is_null() {
            return Err("RoiAlign: evaluated ROI data is unavailable".into());
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) }
            .iter()
            .map(|&v| f64::from(v))
            .collect())
    }
}

fn validate_roi_align(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    params: RoiAlignParams,
) -> Result<Vec<[i32; 2]>, MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let rois = ctx.resolve(&n.inputs[1])?;
    let batch = ctx.resolve(&n.inputs[2])?;
    let xs = ctx.shape_of(x);
    let rs = ctx.shape_of(rois);
    if xs.len() != 4
        || rs.len() != 2
        || rs[1] != 4
        || ctx.shape_of(batch) != [rs[0]]
        || xs[2] <= 0
        || xs[3] <= 0
    {
        return Err("RoiAlign: expected X[N,C,H,W], rois[R,4], batch_indices[R], and positive spatial dimensions".into());
    }
    roi_product(&xs)?;
    roi_product(&[rs[0], xs[1], params.height, params.width])?;
    let batches = ctx.read_ints_eval(&n.inputs[2])?;
    if batches.len() != rs[0] as usize || batches.iter().any(|&b| b < 0 || b >= i64::from(xs[0])) {
        return Err("RoiAlign: batch_indices must be within the input batch".into());
    }
    let values = roi_values(ctx, &n.inputs[1])?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err("RoiAlign: all ROI coordinates must be finite".into());
    }
    values
        .chunks_exact(4)
        .map(|roi| {
            let grid = params.grid(roi, n.outputs[0].otype == 11)?;
            roi_product(&[xs[1], params.height, params.width, grid[0], grid[1]])?;
            Ok(grid)
        })
        .collect()
}

pub(crate) fn needs_roi_validation(n: &NodeDesc) -> bool {
    (n.domain.is_empty() && n.op_type == "RoiAlign")
        || n.subgraphs
            .iter()
            .any(|g| g.nodes.iter().any(needs_roi_validation))
}

pub(crate) fn validate_compiled_rois(
    ctx: &mut TranslationContext,
    nodes: &[NodeDesc],
) -> Result<bool, MlxError> {
    for n in nodes {
        if n.subgraphs
            .iter()
            .any(|g| g.nodes.iter().any(needs_roi_validation))
        {
            return Ok(false);
        }
        if n.domain.is_empty() && n.op_type == "RoiAlign" {
            if n.inputs
                .iter()
                .any(|input| input.source == Src::Intermediate)
            {
                return Ok(false);
            }
            validate_roi_align(ctx, n, RoiAlignParams::new(n)?)?;
        }
    }
    Ok(true)
}

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
    let params = RoiAlignParams::new(n)?;
    let tracing = ctx.shape_keyed_compile() || ctx.shapeless_compile();
    let grids = if tracing {
        Vec::new()
    } else {
        validate_roi_align(ctx, n, params)?
    };
    if n.outputs[0].otype == 11 {
        return roi_align_double(ctx, n, params, &grids);
    }
    let x0 = ctx.resolve(&n.inputs[0])?;
    let xc = contiguous(ctx, x0)?;
    let xin = astype(ctx, xc, mlx::mlx_dtype__MLX_FLOAT32)?;
    let rois = ctx.resolve(&n.inputs[1])?;
    let rois = astype(ctx, rois, mlx::mlx_dtype__MLX_FLOAT32)?;
    let batch = ctx.resolve(&n.inputs[2])?;
    let batch = astype(ctx, batch, mlx::mlx_dtype__MLX_INT32)?;
    let rr = ctx.shape_of(rois)[0];
    let cc = ctx.shape_of(xin)[1];
    let output_shape = [rr, cc, params.height, params.width];
    roi_product(&output_shape)?;
    let out = if rr == 0 || cc == 0 {
        roi_zeros(ctx, &output_shape)?
    } else if params.ratio > 0 {
        roi_align_fixed(ctx, xin, rois, batch, params, [params.ratio; 2])?
    } else {
        if tracing {
            return Err("RoiAlign: adaptive sampling requires live eager ROI geometry".into());
        }
        let mut outputs = VectorArray::new();
        for (i, &grid) in grids.iter().enumerate() {
            let i = i as i32;
            let roi = slice(ctx, rois, &[i, 0], &[i + 1, 4])?;
            let batch = slice(ctx, batch, &[i], &[i + 1])?;
            let output = if grid.contains(&0) {
                roi_zeros(ctx, &[1, cc, params.height, params.width])?
            } else {
                roi_align_fixed(ctx, xin, roi, batch, params, grid)?
            };
            outputs.append(output);
        }
        ctx.emit(|res, s| unsafe { mlx::mlx_concatenate_axis(res, outputs.as_raw(), 0, s) })?
    };
    let out = astype(ctx, out, out_dtype(n))?;
    let y = contiguous(ctx, out)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn roi_zeros(ctx: &mut TranslationContext, shape: &[i32]) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe {
        mlx::mlx_zeros(
            res,
            shape.as_ptr(),
            shape.len(),
            mlx::mlx_dtype__MLX_FLOAT32,
            s,
        )
    })
}

fn roi_align_fixed(
    ctx: &mut TranslationContext,
    xin: mlx::mlx_array,
    rois: mlx::mlx_array,
    batch_idx: mlx::mlx_array,
    params: RoiAlignParams,
    grid: [i32; 2],
) -> Result<mlx::mlx_array, MlxError> {
    let xs = ctx.shape_of(xin);
    let (cc, hh, ww) = (xs[1], xs[2], xs[3]);
    let rr = ctx.shape_of(rois)[0];
    let RoiAlignParams {
        height: oh,
        width: ow,
        scale,
        half_pixel,
        max: is_max,
        ..
    } = params;
    let [gh, gw] = grid;
    let ps = roi_product(&[oh, ow, gh, gw])?;
    roi_product(&[rr, cc, ps])?;
    let spatial = roi_product(&[hh, ww])?;
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
    let iy_scaled = mul(ctx, iy_h, binh5)?;
    let iy_term = div(ctx, iy_scaled, ghf)?;
    let ph_term = mul(ctx, ph_idx, binh5)?;
    let yc0 = add(ctx, rsh5, ph_term)?;
    let yc = add(ctx, yc0, iy_term)?;

    let half2 = scalar_f(ctx, 0.5);
    let ix_h = add(ctx, ix_idx, half2)?;
    let gwf = scalar_f(ctx, gw as f32);
    let ix_scaled = mul(ctx, ix_h, binw5)?;
    let ix_term = div(ctx, ix_scaled, gwf)?;
    let pw_term = mul(ctx, pw_idx, binw5)?;
    let xc0 = add(ctx, rsw5, pw_term)?;
    let xc = add(ctx, xc0, ix_term)?;

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
    let xbf = reshape(ctx, xb, &[rr, cc, spatial])?;

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
    reshape(ctx, out, &[rr, cc, oh, ow])
}

fn roi_align_double(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    params: RoiAlignParams,
    grids: &[[i32; 2]],
) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(x);
    let [_, channels, height, width] =
        <[i32; 4]>::try_from(shape).map_err(|_| "RoiAlign: X must have rank four")?;
    let values = roi_values(ctx, &n.inputs[0])?;
    let rois = roi_values(ctx, &n.inputs[1])?;
    let batches = ctx.read_ints_eval(&n.inputs[2])?;
    let output_shape = [grids.len() as i32, channels, params.height, params.width];
    let size = roi_product(&output_shape)? as usize;
    let mut output = Vec::with_capacity(size);
    let scale = f64::from(params.scale);
    let offset = if params.half_pixel { 0.5 } else { 0.0 };
    for ((roi, &batch), &[gh, gw]) in rois.chunks_exact(4).zip(&batches).zip(grids) {
        let start_x = roi[0] * scale - offset;
        let start_y = roi[1] * scale - offset;
        let mut extent_x = roi[2] * scale - offset - start_x;
        let mut extent_y = roi[3] * scale - offset - start_y;
        if !params.half_pixel {
            extent_x = extent_x.max(1.0);
            extent_y = extent_y.max(1.0);
        }
        let bin_x = extent_x / f64::from(params.width);
        let bin_y = extent_y / f64::from(params.height);
        for channel in 0..channels {
            let base = ((batch as usize * channels as usize + channel as usize) * height as usize)
                * width as usize;
            for ph in 0..params.height {
                for pw in 0..params.width {
                    let mut result = 0.0;
                    let mut first = true;
                    for iy in 0..gh {
                        let y = start_y
                            + f64::from(ph) * bin_y
                            + (f64::from(iy) + 0.5) * bin_y / f64::from(gh);
                        for ix in 0..gw {
                            let x = start_x
                                + f64::from(pw) * bin_x
                                + (f64::from(ix) + 0.5) * bin_x / f64::from(gw);
                            let (positions, weights) = roi_double_taps(y, x, height, width);
                            let mut terms = [0.0; 4];
                            for k in 0..4 {
                                terms[k] = weights[k] * values[base + positions[k]];
                            }
                            let sample = if params.max {
                                terms
                                    .into_iter()
                                    .reduce(|a, b| if a < b { b } else { a })
                                    .unwrap()
                            } else {
                                terms.into_iter().sum()
                            };
                            if params.max {
                                if first || result < sample {
                                    result = sample;
                                }
                                first = false;
                            } else {
                                result += sample;
                            }
                        }
                    }
                    if !params.max {
                        result /= f64::from((gh * gw).max(1));
                    }
                    output.push(result);
                }
            }
        }
    }
    let result = ctx.from_host(
        output.as_ptr().cast(),
        &output_shape,
        mlx::mlx_dtype__MLX_FLOAT64,
    );
    ctx.bind(&n.outputs[0], result);
    Ok(())
}

fn roi_double_taps(y: f64, x: f64, height: i32, width: i32) -> ([usize; 4], [f64; 4]) {
    if y < -1.0 || y > f64::from(height) || x < -1.0 || x > f64::from(width) {
        return ([0; 4], [0.0; 4]);
    }
    let y = y.clamp(0.0, f64::from(height - 1));
    let x = x.clamp(0.0, f64::from(width - 1));
    let y0 = y as usize;
    let x0 = x as usize;
    let y1 = (y0 + 1).min(height as usize - 1);
    let x1 = (x0 + 1).min(width as usize - 1);
    let ly = y - y0 as f64;
    let lx = x - x0 as f64;
    let hy = 1.0 - ly;
    let hx = 1.0 - lx;
    (
        [
            y0 * width as usize + x0,
            y0 * width as usize + x1,
            y1 * width as usize + x0,
            y1 * width as usize + x1,
        ],
        [hy * hx, hy * lx, ly * hx, ly * lx],
    )
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
        is_mlx_cpu_float(x.dtype)
            && out.dtype == x.dtype
            && r.dtype == x.dtype
            && (x.dtype != 16 || node.since_version() >= 22)
            && b.dtype == 7,
        "X/rois/output must share float16/float32/float64 (bfloat16 since opset 22), batch_indices must be int64; got {} / {} / {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(r.dtype),
        crate::registry::ort_dtype_name(b.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() == 4
            && x.shape
                .iter()
                .chain(&r.shape)
                .chain(&b.shape)
                .chain(&out.shape)
                .all(|&d| d <= i64::from(i32::MAX)),
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
        (0..=i64::from(i32::MAX)).contains(&sampling_ratio),
        "sampling_ratio must be a nonnegative int32, got {sampling_ratio}"
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
        [
            node.int_attr("output_height", 1),
            node.int_attr("output_width", 1)
        ]
        .iter()
        .all(|&d| (1..=i64::from(i32::MAX)).contains(&d)),
        "output_height and output_width must be positive int32"
    );
    require!(
        [
            node.int_attr("output_height", 1),
            node.int_attr("output_width", 1),
            sampling_ratio.max(1),
            sampling_ratio.max(1),
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
        (ctm == "half_pixel" || ctm == "output_half_pixel")
            && (node.since_version() >= 16 || ctm == "output_half_pixel"),
        "coordinate_transformation_mode must be \"half_pixel\" or \"output_half_pixel\", got {ctm:?}"
    );
    require!(
        node.float_attr("spatial_scale", 1.0).is_finite(),
        "spatial_scale must be finite"
    );
    let expected = [
        r.shape[0],
        x.shape[1],
        node.int_attr("output_height", 1),
        node.int_attr("output_width", 1),
    ];
    require!(
        out.shape.len() == 4
            && out
                .shape
                .iter()
                .zip(expected)
                .all(|(&a, b)| a < 0 || b < 0 || a == b),
        "output must match [R,C,output_height,output_width]"
    );
    Ok(())
}

fn roi_align_compile_safety(node: &NodeView) -> crate::registry::CompileShapeSafety {
    use crate::registry::CompileShapeSafety;
    if node.int_attr("sampling_ratio", 0) == 0 {
        CompileShapeSafety::EagerOnly {
            reason: "adaptive grids depend on live ROI values",
        }
    } else if node.input_info(0).is_some_and(|x| x.dtype == 11) {
        CompileShapeSafety::EagerOnly {
            reason: "float64 RoiAlign uses exact host arithmetic on the CPU route",
        }
    } else {
        CompileShapeSafety::ShapeKeyedOnly {
            reason: crate::registry::MLX_SLICE_SHAPE_REASON,
        }
    }
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
    registry.register_shape_keyed(
        OpRegistration {
            domain: "",
            op_type: "GridSample",
            min_opset: 16,
            max_opset: K_ANY_OPSET,
            handler: grid_sample_op,
            claim: grid_sample_claim,
        },
        "conservatively specialize checked sampling strides and separable geometry to input shapes",
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
        "conservatively specialize host-generated homogeneous grid to size values",
    );
    shape_keyed(
        registry,
        "Col2Im",
        col2im_op,
        col2im_claim,
        crate::registry::MLX_SCATTER_SHAPE_REASON,
    );
    registry.register_shape_classified(
        OpRegistration {
            domain: "",
            op_type: "RoiAlign",
            min_opset: 10,
            max_opset: K_ANY_OPSET,
            handler: roi_align_op,
            claim: roi_align_claim,
        },
        roi_align_compile_safety,
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
mod roi_align_tests {
    use super::{RoiAlignParams, roi_double_taps, roi_product};
    use crate::engine::NodeDesc;
    use crate::registry::CompileShapeSafety;

    fn params() -> RoiAlignParams {
        RoiAlignParams {
            height: 2,
            width: 3,
            ratio: 0,
            scale: 1.0,
            half_pixel: true,
            max: false,
        }
    }

    #[test]
    fn adaptive_counts_and_empty_geometry() {
        let mut p = params();
        assert_eq!(p.grid(&[0.0, 0.0, 6.0, 4.0], false).unwrap(), [2, 2]);
        assert_eq!(p.grid(&[0.0, 0.0, 6.01, 4.01], false).unwrap(), [3, 3]);
        assert_eq!(p.grid(&[0.0, 0.0, 0.0, 4.0], false).unwrap(), [0, 0]);
        assert_eq!(p.grid(&[1.0, 1.0, 0.0, 0.0], false).unwrap(), [0, 0]);
        p.half_pixel = false;
        assert_eq!(p.grid(&[1.0, 1.0, 0.0, 0.0], false).unwrap(), [1, 1]);
        p.ratio = 3;
        assert_eq!(p.grid(&[0.0; 4], false).unwrap(), [3, 3]);
    }

    #[test]
    fn geometry_overflow_and_f32_ceil_boundary() {
        let mut p = params();
        p.width = 1;
        let just_over_one = 1.0 + 2f64.powi(-30);
        assert_eq!(
            p.grid(&[0.0, 0.0, just_over_one, 2.0], false).unwrap(),
            [1, 1]
        );
        assert_eq!(
            p.grid(&[0.0, 0.0, just_over_one, 2.0], true).unwrap(),
            [1, 2]
        );
        assert!(p.grid(&[0.0, 0.0, 1e20, 2.0], false).is_err());
        assert!(p.grid(&[0.0, 0.0, f64::MAX, f64::MAX], true).is_err());
        assert!(roi_product(&[i32::MAX, 2]).is_err());
        assert!(roi_product(&[-1, 0]).is_err());
        assert_eq!(roi_product(&[i32::MAX, i32::MAX, 0]).unwrap(), 0);
    }

    #[test]
    fn attributes_are_checked_before_i32_narrowing() {
        let node = || {
            NodeDesc::new(
                "RoiAlign".into(),
                "".into(),
                22,
                CompileShapeSafety::EagerOnly {
                    reason: "unit test",
                },
            )
        };
        for (name, values) in [
            ("sampling_ratio", vec![-1, i64::MAX, 1i64 << 32]),
            ("output_height", vec![0, -1, i64::MAX, 1i64 << 32]),
            ("output_width", vec![0, -1, i64::MAX, 1i64 << 32]),
        ] {
            for value in values {
                let mut n = node();
                n.ints.insert(name.into(), value);
                assert!(RoiAlignParams::new(&n).is_err(), "{name}={value}");
            }
        }
        let mut n = node();
        n.ints.insert("sampling_ratio".into(), i64::from(i32::MAX));
        assert!(RoiAlignParams::new(&n).is_err());
        for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut n = node();
            n.floats.insert("spatial_scale".into(), scale);
            assert!(RoiAlignParams::new(&n).is_err());
        }
    }

    #[test]
    fn exact_double_corner_weights_and_boundary() {
        assert_eq!(
            roi_double_taps(-1.0, -1.0, 3, 3),
            ([0, 1, 3, 4], [1.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(
            roi_double_taps(3.0, 3.0, 3, 3),
            ([8; 4], [1.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(roi_double_taps(3.0001, 0.0, 3, 3), ([0; 4], [0.0; 4]));
        assert_eq!(
            roi_double_taps(0.25, 0.5, 3, 3),
            ([0, 1, 3, 4], [0.375, 0.375, 0.125, 0.125])
        );
    }
}

#[cfg(test)]
mod col2im_tests {
    use super::Col2ImGeometry;

    #[test]
    fn checked_geometry_and_empty_batch() {
        let geometry = |input: &[i64], image: &[i64], block: &[i64]| {
            Col2ImGeometry::new(
                input,
                image,
                block,
                &vec![1; image.len()],
                &vec![1; image.len()],
                &vec![0; 2 * image.len()],
            )
        };
        let g = geometry(&[2, 12, 9], &[4, 5], &[2, 3]).unwrap();
        assert_eq!(g.output_shape, [2, 2, 4, 5]);
        assert_eq!(g.npos, [3, 3]);
        assert!(geometry(&[0, 12, 9], &[4, 5], &[2, 3]).is_ok());
        for (input, image, block) in [
            (vec![1, 2, 2], vec![4], vec![2]),
            (vec![1, 3, 3], vec![4], vec![2]),
            (vec![1, 2, 3], vec![0], vec![2]),
            (vec![1, 2, 3], vec![4], vec![0]),
            (vec![1, 2, 3], vec![4], vec![5]),
            (vec![1, 2, 3], vec![i64::MAX], vec![2]),
            (vec![1, 2, 3], vec![i32::MAX as i64; 3], vec![2; 3]),
            (vec![i32::MAX as i64, 2, 3], vec![4], vec![2]),
        ] {
            assert!(geometry(&input, &image, &block).is_err());
        }
        for (dilations, strides, pads) in [
            (vec![0], vec![1], vec![0, 0]),
            (vec![1], vec![0], vec![0, 0]),
            (vec![1], vec![1], vec![-1, 0]),
            (vec![1, 1], vec![1], vec![0, 0]),
        ] {
            assert!(
                Col2ImGeometry::new(&[1, 2, 3], &[4], &[2], &dilations, &strides, &pads).is_err()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{affine_grid_shape, gs_product, gs_shape};

    #[test]
    fn grid_sample_geometry_is_checked_before_indexing_or_allocation() {
        assert_eq!(
            gs_shape(&[2, 3, 4, 5], &[2, 6, 7, 2]).unwrap(),
            [2, 3, 6, 7]
        );
        assert_eq!(gs_shape(&[0, 3, 1], &[0, 7, 1]).unwrap(), [0, 3, 7]);
        assert_eq!(
            gs_shape(&[2, 0, 1, 2, 3], &[2, 0, 7, 8, 3]).unwrap(),
            [2, 0, 0, 7, 8]
        );
        assert_eq!(
            gs_shape(&[0, 1, 2, 2], &[0, 65536, 65536, 2]).unwrap(),
            [0, 1, 65536, 65536]
        );
        assert_eq!(
            gs_shape(&[1, 1, 2, 2, 2], &[1, 65536, 65536, 0, 3]).unwrap(),
            [1, 1, 65536, 65536, 0]
        );
        assert_eq!(gs_product(&[i32::MAX, i32::MAX, 0]).unwrap(), 0);
        assert!(gs_product(&[2, i32::MAX]).is_err());
        assert!(gs_product(&[-1, 2]).is_err());
        assert!(gs_product(&[0, -1]).is_err());
        for (x, grid) in [
            (vec![1, 2], vec![1, 2]),
            (vec![1, 2, 3], vec![1, 2, 2]),
            (vec![1, 2, 3, 4], vec![1, 2, 2]),
            (vec![2, 1, 3, 4], vec![1, 2, 3, 2]),
            (vec![1, 1, 0, 4], vec![1, 2, 3, 2]),
            (vec![1, 1, -1, 4], vec![1, 2, 3, 2]),
            (vec![1, 1, 3, 4], vec![1, -1, 3, 2]),
            (vec![1, 2, i32::MAX, 2], vec![1, 1, 1, 2]),
            (vec![1, 2, 3, 4], vec![1, i32::MAX, 2, 2]),
            (
                vec![0, 1, 1, 1, 1, 1],
                vec![0, i32::MAX, i32::MAX, i32::MAX, 1, 4],
            ),
        ] {
            assert!(gs_shape(&x, &grid).is_err(), "{x:?} / {grid:?}");
        }
    }

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
