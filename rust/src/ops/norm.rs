//! Normalization op handlers. Faithful port of the C++ `ops/norm.cc` + `ops/norm_ext.cc`:
//!
//!   * RMSNormalization (ai.onnx opset 23+)             — fused or composed suffix normalization
//!   * LayerNormalization (ai.onnx opset 17+)           — fused or composed, optional statistics
//!   * SimplifiedLayerNormalization (ORT extension)    — RMS normalization, optional inverse std
//!   * SkipLayerNormalization (com.microsoft)           — residual add + mlx_fast_layer_norm
//!   * SkipSimplifiedLayerNormalization (com.microsoft) — residual add + mlx_fast_rms_norm
//!   * GroupNormalization (ai.onnx opset 18+)           — versioned affine and stash precision
//!   * InstanceNormalization (ai.onnx)                   — composed spatial mean/var/affine
//!   * LpNormalization (ai.onnx)                        — composed abs/sum or square/sum/sqrt
//!   * BatchNormalization (ai.onnx)                     — versioned affine and population statistics
//!   * LRN (ai.onnx, across-channel)                    — composed square/window-sum/power/divide
//!
//! Standard norms honor stash precision and broadcast affine parameters. Their ordinary last-axis
//! vector-affine forms retain the fused kernels; float64 tensors/statistics use an isolated CPU stream.

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::registry::{
    ClaimResult, CompileShapeSafety, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry,
    is_mlx_cpu_float, is_mlx_float,
};
use crate::sys::mlx;
use crate::sys::ort;
use crate::{deny, require};
use std::os::raw::c_char;

// ---- small local MLX helpers -------------------------------------------------------------------

/// A null-`ctx` `mlx_array` — the mlx-c "empty/absent" sentinel (e.g. an omitted layer-norm bias).
#[inline]
fn empty_array() -> mlx::mlx_array {
    mlx::mlx_array_ {
        ctx: std::ptr::null_mut(),
    }
}

fn mul(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_multiply, a, b)
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
fn divide(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_divide, a, b)
}
fn sqrt(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_sqrt(res, a, s) })
}
fn abs(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_abs(res, a, s) })
}
fn sum_axis(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    axis: i32,
    keepdims: bool,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_sum_axis(res, a, axis, keepdims, s) })
}
/// A 0-d scalar of dtype `dt` holding `v` (eps constant), matching the compute dtype so no unwanted
/// upcast occurs.
fn scalar_like(
    ctx: &mut TranslationContext,
    v: f32,
    dt: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    let s = ctx.scalar_f32(v);
    if dt == mlx::mlx_dtype__MLX_FLOAT32 {
        Ok(s)
    } else {
        ctx.astype(s, dt)
    }
}

/// Reshape a per-channel vector `[C]` to `[1, C, 1, ..., 1]` so it broadcasts over N and spatial.
fn channel_broadcast(
    ctx: &mut TranslationContext,
    v: mlx::mlx_array,
    rank: usize,
    channels: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let mut shape = vec![1i32; rank];
    if rank >= 2 {
        shape[1] = channels;
    }
    ctx.reshape(v, &shape)
}

fn epsilon(n: &NodeDesc, default: f32) -> f32 {
    n.floats.get("epsilon").copied().unwrap_or(default)
}

fn present(n: &NodeDesc, i: usize) -> bool {
    i < n.inputs.len() && n.inputs[i].source != Src::Absent
}

fn normalization_shape(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    minimum_rank: usize,
) -> Result<Vec<i32>, MlxError> {
    let shape = ctx.tensor_shape(&n.inputs[0])?;
    if shape.len() < minimum_rank {
        return Err(format!(
            "{}: input rank must be at least {minimum_rank}, got {}",
            n.op_type,
            shape.len()
        ));
    }
    shape
        .into_iter()
        .map(|dim| {
            i32::try_from(dim)
                .ok()
                .filter(|&dim| dim >= 0)
                .ok_or_else(|| format!("{}: dimension {dim} exceeds native bounds", n.op_type))
        })
        .collect()
}

fn normalization_compute_dtype(dtype: mlx::mlx_dtype) -> mlx::mlx_dtype {
    if dtype == mlx::mlx_dtype__MLX_FLOAT64 {
        dtype
    } else {
        mlx::mlx_dtype__MLX_FLOAT32
    }
}

// ---- handlers ----------------------------------------------------------------------------------

fn instance_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let shape = normalization_shape(ctx, n, 3)?;
    for input in &n.inputs[1..3] {
        let affine_shape = ctx.tensor_shape(input)?;
        if affine_shape != [i64::from(shape[1])] {
            return Err(format!(
                "InstanceNormalization: affine shape {affine_shape:?} must be [{}]",
                shape[1]
            ));
        }
    }
    let dtype = crate::engine::mlx_dtype_from_onnx(n.outputs[0].otype);
    if shape.contains(&0) {
        let out = ctx.zeros(&shape, dtype)?;
        ctx.bind(&n.outputs[0], out);
        return Ok(());
    }
    let x = ctx.resolve(&n.inputs[0])?;
    let scale = ctx.resolve(&n.inputs[1])?;
    let bias = ctx.resolve(&n.inputs[2])?;
    let compute_dtype = normalization_compute_dtype(dtype);
    let x = ctx.astype(x, compute_dtype)?;
    let scale = ctx.astype(scale, compute_dtype)?;
    let bias = ctx.astype(bias, compute_dtype)?;
    let axes: Vec<i32> = (2..shape.len() as i32).collect();
    let mean = norm_mean_axes(ctx, x, &axes, compute_dtype)?;
    let centered = sub(ctx, x, mean)?;
    let residual_mean = norm_mean_axes(ctx, centered, &axes, compute_dtype)?;
    let centered = sub(ctx, centered, residual_mean)?;
    let square = mul(ctx, centered, centered)?;
    let variance = norm_mean_axes(ctx, square, &axes, compute_dtype)?;
    let eps = scalar_like(ctx, epsilon(n, 1e-5), compute_dtype)?;
    let variance_eps = add(ctx, variance, eps)?;
    let stddev = sqrt(ctx, variance_eps)?;
    let normalized = divide(ctx, centered, stddev)?;
    let scale = channel_broadcast(ctx, scale, shape.len(), shape[1])?;
    let bias = channel_broadcast(ctx, bias, shape.len(), shape[1])?;
    let scaled = mul(ctx, normalized, scale)?;
    let out = add(ctx, scaled, bias)?;
    let out = ctx.astype(out, dtype)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn rms_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    standard_norm_op(ctx, n, false)
}

/// ORT SimplifiedLayerNormalization shares RMS statistics and adds an optional inverse std output.
fn simplified_layer_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    standard_norm_op(ctx, n, false)
}

fn layer_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    standard_norm_op(ctx, n, true)
}

fn affine_broadcasts_to(x: &[i64], affine: &[i64]) -> bool {
    affine.len() <= x.len()
        && x.iter()
            .rev()
            .zip(affine.iter().rev())
            .all(|(&x, &a)| a == 1 || a == x || a < 0 || x < 0)
}

fn norm_mean_axes(
    ctx: &mut TranslationContext,
    input: mlx::mlx_array,
    axes: &[i32],
    dtype: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    // Avoid narrow-type accumulation error, but round each statistic to the requested precision.
    let accumulation_dtype = normalization_compute_dtype(dtype);
    let input = ctx.astype(input, accumulation_dtype)?;
    let mean = ctx.emit(|res, stream| unsafe {
        mlx::mlx_mean_axes(res, input, axes.as_ptr(), axes.len(), true, stream)
    })?;
    ctx.astype(mean, dtype)
}

fn standard_norm_op(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    centered: bool,
) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let scale = ctx.resolve(&n.inputs[1])?;
    let bias = if centered && present(n, 2) {
        Some(ctx.resolve(&n.inputs[2])?)
    } else {
        None
    };
    let shape = ctx.shape_of(x);
    let rank = shape.len() as i64;
    let axis = n.ints.get("axis").copied().unwrap_or(-1);
    if rank == 0 || axis < -rank || axis >= rank {
        return Err(format!(
            "{}: axis {axis} is invalid for rank {rank}",
            n.op_type
        ));
    }
    let axis = if axis < 0 { axis + rank } else { axis } as i32;
    let input_shape: Vec<i64> = shape.iter().map(|&d| i64::from(d)).collect();
    for affine in [Some(scale), bias].into_iter().flatten() {
        let affine_shape: Vec<i64> = ctx.shape_of(affine).iter().map(|&d| i64::from(d)).collect();
        if !affine_broadcasts_to(&input_shape, &affine_shape) {
            return Err(format!(
                "{}: affine shape {affine_shape:?} cannot broadcast to {shape:?}",
                n.op_type
            ));
        }
    }
    let stash = n.ints.get("stash_type").copied().unwrap_or(1);
    let input_dtype = ctx.dtype_of(x);
    let eps = epsilon(n, 1e-5);
    let vector_shape = [shape[shape.len() - 1]];
    if axis == rank as i32 - 1
        && stash == 1
        && n.outputs.len() == 1
        && ctx.size_of(x) != 0
        && input_dtype != mlx::mlx_dtype__MLX_FLOAT64
        && ctx.dtype_of(scale) == input_dtype
        && ctx.shape_of(scale) == vector_shape
        && bias.is_none_or(|b| ctx.dtype_of(b) == input_dtype && ctx.shape_of(b) == vector_shape)
    {
        let r = if centered {
            ctx.mark_fast("mlx_fast_layer_norm");
            ctx.emit(|res, s| unsafe {
                mlx::mlx_fast_layer_norm(res, x, scale, bias.unwrap_or_else(empty_array), eps, s)
            })?
        } else {
            ctx.mark_fast("mlx_fast_rms_norm");
            ctx.emit(|res, s| unsafe { mlx::mlx_fast_rms_norm(res, x, scale, eps, s) })?
        };
        ctx.bind(&n.outputs[0], r);
        return Ok(());
    }
    let stash_dtype = crate::engine::mlx_dtype_from_onnx(stash as ort::ONNXTensorElementDataType);
    let compute = ctx.astype(x, stash_dtype)?;
    let axes: Vec<i32> = (axis..rank as i32).collect();
    let mean = if centered {
        Some(norm_mean_axes(ctx, compute, &axes, stash_dtype)?)
    } else {
        None
    };
    let values = match mean {
        Some(mean) => sub(ctx, compute, mean)?,
        None => compute,
    };
    let squared = mul(ctx, values, values)?;
    let variance = norm_mean_axes(ctx, squared, &axes, stash_dtype)?;
    let epsilon = scalar_like(ctx, eps, stash_dtype)?;
    let denominator = add(ctx, variance, epsilon)?;
    let stddev = sqrt(ctx, denominator)?;
    let inv_std = ctx.unary(mlx::mlx_reciprocal, stddev)?;
    let normalized = if centered {
        mul(ctx, values, inv_std)?
    } else {
        divide(ctx, values, stddev)?
    };
    let normalized = ctx.astype(normalized, input_dtype)?;
    let scaled = mul(ctx, normalized, scale)?;
    let output = match bias {
        Some(bias) => add(ctx, scaled, bias)?,
        None => scaled,
    };
    let output = ctx.astype(
        output,
        crate::engine::mlx_dtype_from_onnx(n.outputs[0].otype),
    )?;
    ctx.bind(&n.outputs[0], output);
    if centered {
        if let Some(mean) = mean
            && n.outputs.get(1).is_some_and(|o| !o.name.is_empty())
        {
            ctx.bind(&n.outputs[1], mean);
        }
        if n.outputs.get(2).is_some_and(|o| !o.name.is_empty()) {
            ctx.bind(&n.outputs[2], inv_std);
        }
    } else if n.op_type == "SimplifiedLayerNormalization"
        && n.outputs.get(1).is_some_and(|o| !o.name.is_empty())
    {
        ctx.bind(&n.outputs[1], inv_std);
    }
    ctx.mark_composed("normalization with explicit stash precision and broadcast affine");
    Ok(())
}

/// SkipLayerNormalization (com.microsoft): residual = input + skip (+ bias);
/// Y = layer_norm(residual, gamma, beta, eps); optional mean, inverse and residual outputs.
fn skip_layer_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    skip_norm_op(ctx, n, true)
}

/// SkipSimplifiedLayerNormalization: residual, normalized Y and optional inverse statistic.
fn skip_rms_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    skip_norm_op(ctx, n, false)
}

fn skip_shapes_compatible(input: &[i64], skip: &[i64]) -> bool {
    matches!(input.len(), 2 | 3)
        && (2..=input.len()).contains(&skip.len())
        && input[input.len() - skip.len()..]
            .iter()
            .zip(skip)
            .enumerate()
            .all(|(i, (&a, &b))| {
                a < 0
                    || b < 0
                    || a == b
                    || (input.len() == 3 && skip.len() == 3 && i == 0 && b == 1)
            })
}

fn skip_norm_op(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    centered: bool,
) -> Result<(), MlxError> {
    if !(3..=if centered { 5 } else { 4 }).contains(&n.inputs.len()) {
        return Err(format!(
            "{}: invalid input count {}",
            n.op_type,
            n.inputs.len()
        ));
    }
    let input = ctx.resolve(&n.inputs[0])?;
    let skip = ctx.resolve(&n.inputs[1])?;
    let shape: Vec<i64> = ctx.shape_of(input).iter().map(|&d| i64::from(d)).collect();
    let skip_shape: Vec<i64> = ctx.shape_of(skip).iter().map(|&d| i64::from(d)).collect();
    if !skip_shapes_compatible(&shape, &skip_shape) {
        return Err(format!(
            "{}: incompatible input/skip shapes {shape:?}/{skip_shape:?}",
            n.op_type
        ));
    }
    let dtype = ctx.dtype_of(input);
    let compute_dtype = mlx::mlx_dtype__MLX_FLOAT32;
    let input = ctx.astype(input, compute_dtype)?;
    let skip = ctx.astype(skip, compute_dtype)?;
    let mut affine = [None; 3];
    for i in 2..n.inputs.len() {
        if present(n, i) {
            let value = ctx.resolve(&n.inputs[i])?;
            if ctx.shape_of(value) != [shape[shape.len() - 1] as i32] {
                return Err(format!(
                    "{}: affine input {i} must have hidden-size vector shape",
                    n.op_type
                ));
            }
            affine[i - 2] = Some(ctx.astype(value, compute_dtype)?);
        }
    }
    let gamma = affine[0].ok_or_else(|| format!("{}: missing gamma input", n.op_type))?;
    let beta = if centered { affine[1] } else { None };
    let bias = affine[if centered { 2 } else { 1 }];
    let mut residual = add(ctx, input, skip)?;
    if let Some(bias) = bias {
        residual = add(ctx, residual, bias)?;
    }
    let has_mean = centered && n.outputs.get(1).is_some_and(|o| !o.name.is_empty());
    let has_inverse = n.outputs.get(2).is_some_and(|o| !o.name.is_empty());
    let output = if has_mean || has_inverse {
        let axes = [(shape.len() - 1) as i32];
        let mean = if centered {
            Some(norm_mean_axes(ctx, residual, &axes, compute_dtype)?)
        } else {
            None
        };
        let values = match mean {
            Some(mean) => sub(ctx, residual, mean)?,
            None => residual,
        };
        let squared = mul(ctx, values, values)?;
        let variance = norm_mean_axes(ctx, squared, &axes, compute_dtype)?;
        let eps = scalar_like(ctx, epsilon(n, 1e-12), compute_dtype)?;
        let denominator = add(ctx, variance, eps)?;
        let stddev = sqrt(ctx, denominator)?;
        let inverse = ctx.unary(mlx::mlx_reciprocal, stddev)?;
        let normalized = mul(ctx, values, inverse)?;
        let scaled = mul(ctx, normalized, gamma)?;
        let output = match beta {
            Some(beta) => add(ctx, scaled, beta)?,
            None => scaled,
        };
        if has_mean && let Some(mean) = mean {
            ctx.bind(&n.outputs[1], mean);
        }
        if has_inverse {
            ctx.bind(&n.outputs[2], inverse);
        }
        ctx.mark_composed("skip normalization with float32 saved statistics");
        output
    } else if ctx.size_of(residual) == 0 {
        ctx.mark_composed("empty skip normalization");
        residual
    } else if centered {
        ctx.mark_fast("mlx_fast_layer_norm");
        ctx.emit(|res, s| unsafe {
            mlx::mlx_fast_layer_norm(
                res,
                residual,
                gamma,
                beta.unwrap_or_else(empty_array),
                epsilon(n, 1e-12),
                s,
            )
        })?
    } else {
        ctx.mark_fast("mlx_fast_rms_norm");
        ctx.emit(|res, s| unsafe {
            mlx::mlx_fast_rms_norm(res, residual, gamma, epsilon(n, 1e-12), s)
        })?
    };
    let output = ctx.astype(output, dtype)?;
    ctx.bind(&n.outputs[0], output);
    if n.outputs.get(3).is_some_and(|o| !o.name.is_empty()) {
        let residual = ctx.astype(residual, dtype)?;
        ctx.bind(&n.outputs[3], residual);
    }
    Ok(())
}

/// GroupNormalization: opset 18 uses per-group affine parameters, opset 21 per-channel.
fn group_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let scale = ctx.resolve(&n.inputs[1])?;
    let bias = ctx.resolve(&n.inputs[2])?;
    let shape = ctx.shape_of(x);
    let rank = shape.len();
    if rank < 2 {
        return Err("GroupNormalization requires rank >= 2".to_string());
    }
    let c = shape[1];
    let groups = i32::try_from(n.ints.get("num_groups").copied().unwrap_or(0))
        .map_err(|_| "GroupNormalization num_groups exceeds MLX dimension range".to_string())?;
    if groups <= 0 || c % groups != 0 {
        return Err(format!(
            "GroupNormalization num_groups {groups} must be positive and divide channel count {c}"
        ));
    }
    let per_channel = n.since_version >= 21;
    let affine_size = if per_channel { c } else { groups };
    for affine in [scale, bias] {
        if ctx.shape_of(affine) != [affine_size] {
            return Err(format!(
                "GroupNormalization affine parameters must have shape [{affine_size}]"
            ));
        }
    }
    let per_group = if shape[1..].contains(&0) {
        0
    } else {
        shape[2..].iter().try_fold(c / groups, |size, &dim| {
            size.checked_mul(dim).ok_or_else(|| {
                "GroupNormalization group extent exceeds MLX dimension range".to_string()
            })
        })?
    };
    let input_dtype = ctx.dtype_of(x);
    let stash_dtype = if per_channel {
        crate::engine::mlx_dtype_from_onnx(
            n.ints.get("stash_type").copied().unwrap_or(1) as ort::ONNXTensorElementDataType
        )
    } else {
        input_dtype
    };
    let compute = ctx.astype(x, stash_dtype)?;
    let grp = ctx.reshape(compute, &[shape[0], groups, per_group])?;
    let mean = norm_mean_axes(ctx, grp, &[2], stash_dtype)?;
    let centered = sub(ctx, grp, mean)?;
    let squared = mul(ctx, grp, grp)?;
    let mean_squared = norm_mean_axes(ctx, squared, &[2], stash_dtype)?;
    let squared_mean = mul(ctx, mean, mean)?;
    let var = sub(ctx, mean_squared, squared_mean)?;
    let eps_s = scalar_like(ctx, epsilon(n, 1e-5), stash_dtype)?;
    let var_eps = add(ctx, var, eps_s)?;
    let stddev = sqrt(ctx, var_eps)?;
    let normed = divide(ctx, centered, stddev)?;
    let normed = ctx.astype(normed, input_dtype)?;
    let (normed, sb, bb) = if per_channel {
        (
            ctx.reshape(normed, &shape)?,
            channel_broadcast(ctx, scale, rank, c)?,
            channel_broadcast(ctx, bias, rank, c)?,
        )
    } else {
        (
            normed,
            ctx.reshape(scale, &[1, groups, 1])?,
            ctx.reshape(bias, &[1, groups, 1])?,
        )
    };
    let scaled = mul(ctx, normed, sb)?;
    let out = add(ctx, scaled, bb)?;
    let out = ctx.reshape(out, &shape)?;
    ctx.mark_composed("GroupNormalization with versioned affine parameters and stash precision");
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

/// LpNormalization (ai.onnx): Y = X / ||X||_p along `axis` (p in {1,2}, default 2; axis default -1).
fn lp_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let shape = normalization_shape(ctx, n, 1)?;
    let rank = shape.len() as i64;
    let axis = n.ints.get("axis").copied().unwrap_or(-1);
    if axis < -rank || axis >= rank {
        return Err(format!(
            "LpNormalization: axis {axis} is invalid for rank {rank}"
        ));
    }
    let axis = if axis < 0 { axis + rank } else { axis } as i32;
    let p = n.ints.get("p").copied().unwrap_or(2);
    if p != 1 && p != 2 {
        return Err(format!("LpNormalization: p must be 1 or 2, got {p}"));
    }
    let dtype = crate::engine::mlx_dtype_from_onnx(n.outputs[0].otype);
    if shape.contains(&0) {
        let out = ctx.zeros(&shape, dtype)?;
        ctx.bind(&n.outputs[0], out);
        return Ok(());
    }
    let x = ctx.resolve(&n.inputs[0])?;
    let x = ctx.astype(x, normalization_compute_dtype(dtype))?;

    let norm = if p == 1 {
        let a = abs(ctx, x)?;
        sum_axis(ctx, a, axis, true)?
    } else {
        let sq = mul(ctx, x, x)?;
        let s = sum_axis(ctx, sq, axis, true)?;
        sqrt(ctx, s)?
    };
    let quot = divide(ctx, x, norm)?;
    // ONNX LpNormalization: where the norm is 0, emit 0 rather than NaN (0/0). Matches the ONNX
    // reference `np.where(norm == 0, 0, x / norm)` and ORT's CPU kernel.
    let zero = scalar_like(ctx, 0.0, ctx.dtype_of(x))?;
    let is_zero = ctx.binary(mlx::mlx_equal, norm, zero)?;
    let out = ctx.where_(is_zero, zero, quot)?;
    let out = ctx.astype(out, dtype)?;
    ctx.mark_composed("LpNormalization composed (abs/sum/sqrt/divide) — no fused norm kernel");
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

/// Legacy and modern BN modes share graph-valued population statistics and affine arithmetic.
fn batch_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let shape = normalization_shape(ctx, n, if n.since_version >= 9 { 1 } else { 2 })?;
    let rank = shape.len();
    if n.since_version == 1 && rank != 4 {
        return Err("BatchNormalization1: input must have rank four".into());
    }
    let spatial = n.since_version >= 9 || n.ints.get("spatial").copied().unwrap_or(1) != 0;
    let training = if n.since_version < 7 {
        if n.ints.get("is_test").copied().unwrap_or(0) == 0 {
            return Err("BatchNormalization1/6: in-place training is unsupported".into());
        }
        false
    } else if n.since_version < 14 {
        n.outputs.len() > 1
    } else {
        n.ints.get("training_mode").copied().unwrap_or(0) != 0
    };
    let parameter_shape = if spatial {
        vec![if rank == 1 { 1 } else { shape[1] }]
    } else {
        shape[1..].to_vec()
    };
    for input in &n.inputs[1..5] {
        let actual = ctx.tensor_shape(input)?;
        if actual
            .iter()
            .copied()
            .ne(parameter_shape.iter().copied().map(i64::from))
        {
            return Err(format!(
                "BatchNormalization: parameter shape {actual:?} must be {parameter_shape:?}"
            ));
        }
    }
    let output_dtype = crate::engine::mlx_dtype_from_onnx(n.outputs[0].otype);
    if shape.contains(&0) && !training {
        let output = ctx.zeros(&shape, output_dtype)?;
        ctx.bind(&n.outputs[0], output);
        return Ok(());
    }
    let inputs = n.inputs[..5]
        .iter()
        .map(|input| ctx.resolve(input))
        .collect::<Result<Vec<_>, _>>()?;
    let compute_dtype = if inputs
        .iter()
        .any(|&input| ctx.dtype_of(input) == mlx::mlx_dtype__MLX_FLOAT64)
    {
        mlx::mlx_dtype__MLX_FLOAT64
    } else {
        mlx::mlx_dtype__MLX_FLOAT32
    };
    let inputs = inputs
        .into_iter()
        .map(|input| ctx.astype(input, compute_dtype))
        .collect::<Result<Vec<_>, _>>()?;
    let [x, scale, bias, input_mean, input_var] = inputs[..] else {
        unreachable!("five required BatchNormalization inputs were validated");
    };
    let mut broadcast_shape = vec![1; rank];
    if spatial {
        if rank > 1 {
            broadcast_shape[1] = shape[1];
        }
    } else {
        broadcast_shape[1..].copy_from_slice(&shape[1..]);
    }
    let (centered, mean, variance) = if training {
        if shape.contains(&0) {
            let nan = scalar_like(ctx, f32::NAN, compute_dtype)?;
            let nan = ctx.emit(|res, stream| unsafe {
                mlx::mlx_broadcast_to(
                    res,
                    nan,
                    parameter_shape.as_ptr(),
                    parameter_shape.len(),
                    stream,
                )
            })?;
            (x, nan, nan)
        } else {
            let axes: Vec<i32> = (0..rank as i32)
                .filter(|&axis| !spatial || rank == 1 || axis != 1)
                .filter(|&axis| spatial || axis == 0)
                .collect();
            let mean = norm_mean_axes(ctx, x, &axes, compute_dtype)?;
            let centered = sub(ctx, x, mean)?;
            let residual = norm_mean_axes(ctx, centered, &axes, compute_dtype)?;
            let centered = sub(ctx, centered, residual)?;
            let mean = add(ctx, mean, residual)?;
            let squared = mul(ctx, centered, centered)?;
            let variance = norm_mean_axes(ctx, squared, &axes, compute_dtype)?;
            (
                centered,
                ctx.reshape(mean, &parameter_shape)?,
                ctx.reshape(variance, &parameter_shape)?,
            )
        }
    } else {
        let mean = ctx.reshape(input_mean, &broadcast_shape)?;
        (sub(ctx, x, mean)?, input_mean, input_var)
    };
    let epsilon = scalar_like(ctx, epsilon(n, 1e-5), compute_dtype)?;
    let variance_eps = add(ctx, variance, epsilon)?;
    let stddev = sqrt(ctx, variance_eps)?;
    let denominator = ctx.reshape(stddev, &broadcast_shape)?;
    let normalized = divide(ctx, centered, denominator)?;
    let scale = ctx.reshape(scale, &broadcast_shape)?;
    let bias = ctx.reshape(bias, &broadcast_shape)?;
    let scaled = mul(ctx, normalized, scale)?;
    let output = add(ctx, scaled, bias)?;
    let output = ctx.astype(output, output_dtype)?;
    ctx.bind(&n.outputs[0], output);
    if training {
        let momentum = f64::from(n.floats.get("momentum").copied().unwrap_or(0.9));
        let (momentum, complement) = if compute_dtype == mlx::mlx_dtype__MLX_FLOAT64 {
            (ctx.scalar_f64(momentum), ctx.scalar_f64(1.0 - momentum))
        } else {
            (
                ctx.scalar_f32(momentum as f32),
                ctx.scalar_f32((1.0 - momentum) as f32),
            )
        };
        for (index, previous, current) in [(1, input_mean, mean), (2, input_var, variance)] {
            if n.outputs
                .get(index)
                .is_some_and(|output| !output.name.is_empty())
            {
                let previous = mul(ctx, previous, momentum)?;
                let current = mul(ctx, current, complement)?;
                let updated = add(ctx, previous, current)?;
                let dtype = crate::engine::mlx_dtype_from_onnx(n.outputs[index].otype);
                let updated = ctx.astype(updated, dtype)?;
                ctx.bind(&n.outputs[index], updated);
            }
        }
        if n.outputs
            .get(3)
            .is_some_and(|output| !output.name.is_empty())
        {
            let saved = ctx.astype(mean, crate::engine::mlx_dtype_from_onnx(n.outputs[3].otype))?;
            ctx.bind(&n.outputs[3], saved);
        }
        if n.outputs
            .get(4)
            .is_some_and(|output| !output.name.is_empty())
        {
            let inverse = ctx.unary(mlx::mlx_reciprocal, stddev)?;
            let inverse = ctx.astype(
                inverse,
                crate::engine::mlx_dtype_from_onnx(n.outputs[4].otype),
            )?;
            ctx.bind(&n.outputs[4], inverse);
        }
    }
    ctx.mark_composed("BatchNormalization with versioned modes and population statistics");
    Ok(())
}

// ---- LRN helpers -------------------------------------------------------------------------------

/// Pad `a` along the channel axis (axis 1) with `low`/`high` copies of `value`.
fn pad_channel(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    low: i32,
    high: i32,
    value: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    if low == 0 && high == 0 {
        return Ok(a);
    }
    let axes = [1i32];
    let lo = [low];
    let hi = [high];
    let mode = b"constant\0";
    let out = ctx.emit(|res, s| unsafe {
        mlx::mlx_pad(
            res,
            a,
            axes.as_ptr(),
            1,
            lo.as_ptr(),
            1,
            hi.as_ptr(),
            1,
            value,
            mode.as_ptr() as *const c_char,
            s,
        )
    })?;
    ctx.contiguous(out)
}

/// Slice `a` along the channel axis (axis 1) to `[lo, hi)`, keeping all other axes intact.
fn slice_channel(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    lo: i32,
    hi: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let shape = ctx.shape_of(a);
    let rank = shape.len();
    let mut start = vec![0i32; rank];
    let mut stop = shape;
    let stride = vec![1i32; rank];
    start[1] = lo;
    stop[1] = hi;
    ctx.emit(|res, s| unsafe {
        mlx::mlx_slice(
            res,
            a,
            start.as_ptr(),
            rank,
            stop.as_ptr(),
            rank,
            stride.as_ptr(),
            rank,
            s,
        )
    })
}

/// LRN (ai.onnx, across-channel): for input X=[N,C,*S],
///   square_sum[n,c,*] = sum over the `size`-wide channel window centered at c (clamped to [0,C-1]),
///   Y[n,c,*] = X[n,c,*] / (bias + (alpha/size) * square_sum[n,c,*])^beta.
/// Only offsets intersecting the channel axis contribute, even when `size` exceeds native bounds.
fn lrn_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let shape = normalization_shape(ctx, n, 2)?;
    let c = shape[1];
    let size = n.ints.get("size").copied().unwrap_or(0);
    if size < 1 {
        return Err(format!("LRN: size must be positive, got {size}"));
    }
    let dtype = crate::engine::mlx_dtype_from_onnx(n.outputs[0].otype);
    if shape.contains(&0) {
        let out = ctx.zeros(&shape, dtype)?;
        ctx.bind(&n.outputs[0], out);
        return Ok(());
    }
    let x = ctx.resolve(&n.inputs[0])?;
    let dt = normalization_compute_dtype(dtype);
    let x = ctx.astype(x, dt)?;
    let alpha = n.floats.get("alpha").copied().unwrap_or(1e-4);
    let beta = n.floats.get("beta").copied().unwrap_or(0.75);
    let bias = n.floats.get("bias").copied().unwrap_or(1.0);

    let x2 = mul(ctx, x, x)?;
    let before = ((size - 1) / 2).min(i64::from(c - 1)) as i32;
    let after = (size - 1 - (size - 1) / 2).min(i64::from(c - 1)) as i32;
    let square_sum = if before == c - 1 && after == c - 1 {
        sum_axis(ctx, x2, 1, true)?
    } else {
        let zero = scalar_like(ctx, 0.0, dt)?;
        let mut sum = x2;
        for offset in -before..=after {
            if offset == 0 {
                continue;
            }
            let slice = slice_channel(ctx, x2, offset.max(0), c + offset.min(0))?;
            let shifted = pad_channel(ctx, slice, (-offset).max(0), offset.max(0), zero)?;
            sum = add(ctx, sum, shifted)?;
        }
        sum
    };
    let scale = if dt == mlx::mlx_dtype__MLX_FLOAT64 {
        ctx.scalar_f64(f64::from(alpha) / size as f64)
    } else {
        ctx.scalar_f32(alpha / size as f32)
    };
    let bias_s = scalar_like(ctx, bias, dt)?;
    let beta_s = scalar_like(ctx, beta, dt)?;
    let scaled = mul(ctx, square_sum, scale)?;
    let base = add(ctx, scaled, bias_s)?;
    let denom = ctx.binary(mlx::mlx_power, base, beta_s)?;
    let out = divide(ctx, x, denom)?;
    let out = ctx.astype(out, dtype)?;
    ctx.mark_composed("LRN composed (square/window-sum/power/divide) — no fused LRN kernel");
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

// ---- claim predicates --------------------------------------------------------------------------

/// RMSNormalization (ai.onnx): floating input and independently typed broadcast scale.
fn rms_norm_claim(node: &NodeView) -> ClaimResult {
    standard_norm_claim(node, false)
}

/// ORT SimplifiedLayerNormalization: RMS normalization with optional inverse std.
fn simplified_layer_norm_claim(node: &NodeView) -> ClaimResult {
    standard_norm_claim(node, false)
}

fn layer_norm_claim(node: &NodeView) -> ClaimResult {
    standard_norm_claim(node, true)
}

fn standard_norm_claim(node: &NodeView, centered: bool) -> ClaimResult {
    let nin = node.num_inputs();
    let max_outputs = if centered {
        3
    } else if node.op_type() == "SimplifiedLayerNormalization" {
        2
    } else {
        1
    };
    require!(
        (2..=if centered { 3 } else { 2 }).contains(&nin)
            && (1..=max_outputs).contains(&node.num_outputs()),
        "invalid normalization input/output count: {}in/{}out",
        nin,
        node.num_outputs()
    );
    let (x, scale, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(scale), Some(o)) => (x, scale, o),
        _ => deny!("missing tensor type/shape info on X, scale, or output"),
    };
    require!(
        is_mlx_cpu_float(x.dtype)
            && is_mlx_cpu_float(scale.dtype)
            && out.dtype == scale.dtype
            && (!centered || scale.dtype == x.dtype),
        "normalization requires supported floating-point inputs and matching affine/output types"
    );
    require!(
        affine_broadcasts_to(&x.shape, &scale.shape),
        "scale shape {:?} cannot broadcast to input {:?}",
        scale.shape,
        x.shape
    );
    if nin == 3 && node.input_present(2) {
        let bias = match node.input_info(2) {
            Some(bias) => bias,
            None => deny!("missing tensor type/shape info on bias"),
        };
        require!(
            bias.dtype == x.dtype && affine_broadcasts_to(&x.shape, &bias.shape),
            "bias must match X dtype {}, got {}",
            crate::registry::ort_dtype_name(x.dtype),
            crate::registry::ort_dtype_name(bias.dtype)
        );
    }
    let rank = x.shape.len() as i64;
    require!(rank > 0, "input must have rank >= 1 (got a scalar)");
    let axis = node.int_attr("axis", -1);
    require!(
        axis >= -rank && axis < rank,
        "axis {axis} is out of range for rank {rank}"
    );
    let stash = node.int_attr("stash_type", 1);
    require!(
        matches!(stash, 1 | 10 | 11 | 16),
        "stash_type must be float32, float16, float64, or bfloat16"
    );
    for i in 1..node.num_outputs() {
        if node.output_present(i) {
            require!(
                node.output_info(i)
                    .is_some_and(|o| i64::from(o.dtype) == stash),
                "statistic output {i} must match stash_type {stash}"
            );
        }
    }
    Ok(())
}

fn standard_norm_compile_shape_safety(node: &NodeView) -> CompileShapeSafety {
    let fast_geometry = (|| {
        let x = node.input_info(0)?;
        let scale = node.input_info(1)?;
        let dim = *x.shape.last()?;
        let axis = node.int_attr("axis", -1);
        Some(
            dim > 0
                && (axis == -1 || axis == x.shape.len() as i64 - 1)
                && node.int_attr("stash_type", 1) == 1
                && node.num_outputs() == 1
                && is_mlx_float(x.dtype)
                && scale.dtype == x.dtype
                && scale.shape == [dim]
                && (!node.input_present(2)
                    || node
                        .input_info(2)
                        .is_some_and(|b| b.dtype == x.dtype && b.shape == [dim])),
        )
    })()
    .unwrap_or(false);
    if fast_geometry {
        CompileShapeSafety::Shapeless
    } else {
        CompileShapeSafety::ShapeKeyedOnly {
            reason: "conservatively shape-keyed: composed normalization axes and affine dispatch depend on concrete shapes",
        }
    }
}

fn skip_rms_norm_claim(node: &NodeView) -> ClaimResult {
    skip_norm_claim(node, false)
}

fn skip_layer_norm_claim(node: &NodeView) -> ClaimResult {
    skip_norm_claim(node, true)
}

fn skip_norm_claim(node: &NodeView, centered: bool) -> ClaimResult {
    let nin = node.num_inputs();
    require!(
        (3..=if centered { 5 } else { 4 }).contains(&nin) && (1..=4).contains(&node.num_outputs()),
        "invalid skip normalization input/output count: {}in/{}out",
        nin,
        node.num_outputs()
    );
    let (x, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(o)) => (x, o),
        _ => deny!("missing tensor type/shape info on X or the first output"),
    };
    require!(
        is_mlx_float(x.dtype) && out.dtype == x.dtype,
        "X and output must share one float dtype (fp32/fp16/bf16), got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    let skip = match node.input_info(1) {
        Some(skip) => skip,
        None => deny!("missing skip tensor"),
    };
    require!(
        skip_shapes_compatible(&x.shape, &skip.shape),
        "incompatible input/skip shapes {:?}/{:?}",
        x.shape,
        skip.shape
    );
    let hidden = x.shape[x.shape.len() - 1];
    for i in 1..nin {
        if i >= 3 && !node.input_present(i) {
            continue;
        }
        let info = match node.input_info(i) {
            Some(info) => info,
            None => deny!("missing tensor type/shape info on input {i}"),
        };
        require!(
            info.dtype == x.dtype,
            "input {i} must match X dtype {}, got {}",
            crate::registry::ort_dtype_name(x.dtype),
            crate::registry::ort_dtype_name(info.dtype)
        );
        if i >= 2 {
            require!(
                info.shape.len() == 1
                    && (info.shape[0] < 0 || hidden < 0 || info.shape[0] == hidden),
                "affine input {i} must have hidden-size vector shape"
            );
        }
    }
    for i in 1..=2 {
        let used = node.output_is_used(i).map_err(std::borrow::Cow::Owned)?;
        require!(
            centered || i != 1 || !used,
            "SkipSimplifiedLayerNormalization: the live mean output has no defined RMS statistic contract"
        );
        if node.output_present(i) {
            require!(
                node.output_info(i).is_some_and(|o| o.dtype
                    == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
                "saved statistic output {i} must be float32"
            );
        }
    }
    if node.output_present(3) {
        require!(
            node.output_info(3).is_some_and(|o| o.dtype == x.dtype),
            "residual output must match input dtype"
        );
    }
    require!(
        node.float_attr("epsilon", 1e-12) >= 0.0,
        "epsilon must be nonnegative"
    );
    Ok(())
}

fn skip_norm_compile_shape_safety(node: &NodeView) -> CompileShapeSafety {
    let statistics = node.output_present(2)
        || (node.op_type() == "SkipLayerNormalization" && node.output_present(1));
    if !statistics
        && node
            .input_info(0)
            .is_some_and(|x| x.shape.iter().all(|&d| d > 0))
    {
        CompileShapeSafety::Shapeless
    } else {
        CompileShapeSafety::ShapeKeyedOnly {
            reason: "conservatively shape-keyed: saved statistics and empty versus fused skip normalization depend on concrete extents",
        }
    }
}

/// GroupNormalization: versioned affine shape and independently selected stash precision.
fn group_norm_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 3 && node.num_outputs() == 1,
        "expects 3 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, scale, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(scale), Some(o)) => (x, scale, o),
        _ => deny!("missing tensor type/shape info on X, scale, or output"),
    };
    let bias = match node.input_info(2) {
        Some(b) => b,
        None => deny!("missing tensor type/shape info on bias"),
    };
    require!(
        is_mlx_cpu_float(x.dtype)
            && scale.dtype == x.dtype
            && bias.dtype == x.dtype
            && out.dtype == x.dtype,
        "X, scale, bias, and output must share one supported float dtype, got {}, {}, {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(scale.dtype),
        crate::registry::ort_dtype_name(bias.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() >= 2,
        "input must have rank >= 2 (got rank {})",
        x.shape.len()
    );
    let c = x.shape[1];
    let groups = node.int_attr("num_groups", 0);
    require!(
        groups > 0 && groups <= i64::from(i32::MAX) && (c < 0 || c % groups == 0),
        "num_groups must be positive and divide channel count {c} (got {groups})"
    );
    let affine_size = if node.since_version() >= 21 {
        c
    } else {
        groups
    };
    for (name, affine) in [("scale", scale), ("bias", bias)] {
        require!(
            affine.shape.len() == 1
                && (affine.shape[0] < 0 || affine_size < 0 || affine.shape[0] == affine_size),
            "{name} must have shape [{affine_size}] (got {:?})",
            affine.shape
        );
    }
    if node.since_version() >= 21 {
        require!(
            matches!(node.int_attr("stash_type", 1), 1 | 10 | 11 | 16),
            "stash_type must be float32, float16, float64, or bfloat16"
        );
    }
    Ok(())
}

/// LpNormalization: single float input/output, p in {1,2}.
fn lp_norm_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(o)) => (x, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        is_mlx_cpu_float(x.dtype) && out.dtype == x.dtype,
        "input/output must share one supported float dtype, got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        node.input_rank_known(0) != Some(true) || !x.shape.is_empty(),
        "input must have rank >= 1 (got a scalar)"
    );
    if node.input_rank_known(0) == Some(true) {
        let rank = x.shape.len() as i64;
        let axis = node.int_attr("axis", -1);
        require!(
            axis >= -rank && axis < rank,
            "axis {axis} is invalid for rank {rank}"
        );
    }
    let p = node.int_attr("p", 2);
    require!(p == 1 || p == 2, "only p=1 or p=2 is supported (got {p})");
    Ok(())
}

/// BatchNormalization: versioned affine/statistic types, modes, and feature geometry.
fn batch_norm_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 5,
        "expects five required inputs, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let version = node.since_version();
    let spatial = version >= 9 || node.int_attr("spatial", 1) != 0;
    if version < 9 {
        require!(
            matches!(node.int_attr("spatial", 1), 0 | 1),
            "spatial must be zero or one"
        );
    }
    let training = if version < 7 {
        require!(
            node.int_attr("is_test", 0) != 0,
            "legacy training requires in-place running-statistic semantics"
        );
        false
    } else if version < 14 {
        node.num_outputs() > 1
    } else {
        let training = node.int_attr("training_mode", 0);
        require!(
            matches!(training, 0 | 1),
            "training_mode must be zero or one"
        );
        training == 1
    };
    let outputs = if !training {
        1
    } else if version < 14 {
        5
    } else {
        3
    };
    require!(
        node.num_outputs() == outputs,
        "this BatchNormalization mode expects {outputs} output slots"
    );
    let (x, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(o)) => (x, o),
        _ => deny!("missing tensor type/shape info on X or output"),
    };
    require!(
        is_mlx_cpu_float(x.dtype) && out.dtype == x.dtype,
        "X/output must share one supported float dtype, got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    let rank_known = node.input_rank_known(0) == Some(true);
    if rank_known {
        require!(
            if version == 1 {
                x.shape.len() == 4
            } else {
                x.shape.len() >= if version >= 9 { 1 } else { 2 }
            },
            "invalid input rank {} for BatchNormalization version {version}",
            x.shape.len()
        );
    }
    let scale_type = match node.input_info(1) {
        Some(info) => info.dtype,
        None => deny!("missing scale type"),
    };
    let statistics_type = match node.input_info(3) {
        Some(info) => info.dtype,
        None => deny!("missing statistics type"),
    };
    for i in 1..5 {
        let info = match node.input_info(i) {
            Some(info) => info,
            None => deny!("missing tensor type/shape info on input {i}"),
        };
        let dtype = if i < 3 { scale_type } else { statistics_type };
        require!(
            is_mlx_cpu_float(info.dtype)
                && info.dtype == dtype
                && (version >= 15 || i >= 3 && version >= 14 || info.dtype == x.dtype),
            "input {i} violates the versioned affine/statistic dtype groups"
        );
        if node.input_rank_known(i) == Some(true) {
            let expected = if spatial {
                Some(vec![if rank_known && x.shape.len() > 1 {
                    x.shape[1]
                } else if rank_known {
                    1
                } else {
                    -1
                }])
            } else if rank_known {
                Some(x.shape[1..].to_vec())
            } else {
                None
            };
            if let Some(expected) = expected {
                require!(
                    info.shape.len() == expected.len()
                        && info.shape.iter().zip(expected.iter()).all(
                            |(&actual, &expected)| actual < 0 || expected < 0 || actual == expected
                        ),
                    "input {i} shape {:?} must match {expected:?}",
                    info.shape
                );
            }
        }
    }
    for i in 1..outputs {
        if node.output_present(i) {
            require!(
                node.output_info(i)
                    .is_some_and(|info| info.dtype == statistics_type),
                "statistic output {i} must match the input statistic dtype"
            );
        }
    }
    Ok(())
}

fn instance_norm_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 3 && node.num_outputs() == 1,
        "expects 3 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, scale, bias, out) = match (
        node.input_info(0),
        node.input_info(1),
        node.input_info(2),
        node.output_info(0),
    ) {
        (Some(x), Some(scale), Some(bias), Some(out)) => (x, scale, bias, out),
        _ => deny!("missing tensor type/shape info"),
    };
    require!(
        is_mlx_cpu_float(x.dtype)
            && scale.dtype == x.dtype
            && bias.dtype == x.dtype
            && out.dtype == x.dtype,
        "input, scale, bias, and output must share one MLX float dtype"
    );
    require!(
        node.input_rank_known(0) != Some(true) || x.shape.len() >= 3,
        "input must have rank at least 3"
    );
    for (index, affine) in [(1, scale), (2, bias)] {
        require!(
            node.input_rank_known(index) != Some(true) || affine.shape.len() == 1,
            "scale and bias must be rank-one channel vectors"
        );
        if node.input_rank_known(0) == Some(true) && node.input_rank_known(index) == Some(true) {
            require!(
                x.shape[1] < 0 || affine.shape[0] < 0 || affine.shape[0] == x.shape[1],
                "scale and bias must have length C={}",
                x.shape[1]
            );
        }
    }
    Ok(())
}

/// LRN (ai.onnx, across-channel): floating tensors with a channel axis and positive window size.
fn lrn_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(o)) => (x, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        is_mlx_cpu_float(x.dtype) && out.dtype == x.dtype,
        "input/output must share one supported float dtype, got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        node.input_rank_known(0) != Some(true) || x.shape.len() >= 2,
        "input must have rank >= 2 (got rank {})",
        x.shape.len()
    );
    let size = node.int_attr("size", 0);
    require!(size >= 1, "window size must be >= 1 (got {size})");
    Ok(())
}

fn spatial_norm_compile_shape_safety(node: &NodeView) -> CompileShapeSafety {
    if (0..node.num_inputs()).all(|i| {
        node.input_rank_known(i) == Some(true)
            && node
                .input_info(i)
                .is_some_and(|info| info.shape.iter().all(|&dim| dim > 0))
    }) {
        CompileShapeSafety::Shapeless
    } else {
        CompileShapeSafety::ShapeKeyedOnly {
            reason: "normalization rank, empty dispatch, and channel geometry depend on concrete shapes",
        }
    }
}

// ---- registration ------------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn shape_keyed(
    registry: &mut OpRegistry,
    domain: &'static str,
    op_type: &'static str,
    min_opset: i32,
    handler: crate::registry::OpHandler,
    claim: crate::registry::ClaimPredicate,
    reason: &'static str,
) {
    registry.register_shape_keyed(
        OpRegistration {
            domain,
            op_type,
            min_opset,
            max_opset: K_ANY_OPSET,
            handler,
            claim,
        },
        reason,
    );
}

pub fn register_norm(registry: &mut OpRegistry) {
    registry.register_shape_classified(
        OpRegistration {
            domain: "",
            op_type: "InstanceNormalization",
            min_opset: 1,
            max_opset: K_ANY_OPSET,
            handler: instance_norm_op,
            claim: instance_norm_claim,
        },
        spatial_norm_compile_shape_safety,
    );
    // RMSNormalization entered ai.onnx at opset 23.
    registry.register_shape_classified(
        OpRegistration {
            domain: "",
            op_type: "RMSNormalization",
            min_opset: 23,
            max_opset: K_ANY_OPSET,
            handler: rms_norm_op,
            claim: rms_norm_claim,
        },
        standard_norm_compile_shape_safety,
    );
    // LayerNormalization entered ai.onnx at opset 17.
    registry.register_shape_classified(
        OpRegistration {
            domain: "",
            op_type: "LayerNormalization",
            min_opset: 17,
            max_opset: K_ANY_OPSET,
            handler: layer_norm_op,
            claim: layer_norm_claim,
        },
        standard_norm_compile_shape_safety,
    );
    shape_keyed(
        registry,
        "",
        "GroupNormalization",
        18,
        group_norm_op,
        group_norm_claim,
        "conservatively shape-keyed: group reshape depends on concrete batch, channel, and spatial extents",
    );
    registry.register_shape_classified(
        OpRegistration {
            domain: "",
            op_type: "LpNormalization",
            min_opset: 1,
            max_opset: K_ANY_OPSET,
            handler: lp_norm_op,
            claim: lp_norm_claim,
        },
        spatial_norm_compile_shape_safety,
    );
    registry.register_shape_classified(
        OpRegistration {
            domain: "",
            op_type: "BatchNormalization",
            min_opset: 1,
            max_opset: K_ANY_OPSET,
            handler: batch_norm_op,
            claim: batch_norm_claim,
        },
        spatial_norm_compile_shape_safety,
    );
    shape_keyed(
        registry,
        "",
        "LRN",
        K_ANY_OPSET,
        lrn_op,
        lrn_claim,
        "emits MLX Pad and Slice, which lack Primitive::output_shapes in MLX 0.32.2",
    );
    // ORT exposes this extension in the default domain. Retain the exporter compatibility alias.
    for domain in ["", "com.microsoft"] {
        registry.register_shape_classified(
            OpRegistration {
                domain,
                op_type: "SimplifiedLayerNormalization",
                min_opset: 1,
                max_opset: K_ANY_OPSET,
                handler: simplified_layer_norm_op,
                claim: simplified_layer_norm_claim,
            },
            standard_norm_compile_shape_safety,
        );
    }
    registry.register_shape_classified(
        OpRegistration {
            domain: "com.microsoft",
            op_type: "SkipLayerNormalization",
            min_opset: 1,
            max_opset: K_ANY_OPSET,
            handler: skip_layer_norm_op,
            claim: skip_layer_norm_claim,
        },
        skip_norm_compile_shape_safety,
    );
    registry.register_shape_classified(
        OpRegistration {
            domain: "com.microsoft",
            op_type: "SkipSimplifiedLayerNormalization",
            min_opset: 1,
            max_opset: K_ANY_OPSET,
            handler: skip_rms_norm_op,
            claim: skip_rms_norm_claim,
        },
        skip_norm_compile_shape_safety,
    );
}
