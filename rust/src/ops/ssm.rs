//! State-space / KV-cache "misc" op handlers. Faithful port of the C++ `ops/ssm_misc.cc`:
//!   * TensorScatter (ai.onnx) — typed per-batch linear/circular KV-cache updates.
//!   * CausalConvWithState — dilated causal depthwise conv1d, channel layouts, and carry windows.
//!   * LinearAttention — stable recurrent linear attention (4 update rules, GQA), with a
//!     column-parallel Metal prefill scan and an unrolled fallback.
//!     Only statically translatable, MLX-supported forms are claimed; the rest fall to ORT CPU.
//!
//! Live TensorScatter offsets are validated on every eager application; initializer/no-index
//! updates can compile shape-keyed. Causal convolution supports ndim=1 with static channel/kernel
//! geometry. Linear attention always uses the recurrence: neither decay strength nor normalized
//! keys follows from shape alone, so the chunk_size tuning hint cannot justify a WY fast path.
//! GPU prefill uses the original typed scan for d_k <= 256 and FP32 working state for 257..=512.
//! Wider/CPU forms retain the unroll; output and state storage keep their declared types.
//! `ONNXRUNTIME_EP_MLX_NO_LINEAR_ATTENTION_SCAN=1` selects the unroll for numerical/perf A/B.

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext, mlx_dtype_from_onnx};
use crate::registry::{
    ClaimPredicate, ClaimResult, K_ANY_OPSET, NodeView, OpHandler, OpRegistration, OpRegistry,
    is_mlx_float, is_movable,
};
use crate::sys::mlx;
use crate::sys::ort;
use crate::{deny, require};
use std::ffi::CString;

const T_INT64: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;

// ---- shared helpers -----------------------------------------------------------------------------

fn present(n: &NodeDesc, i: usize) -> bool {
    i < n.inputs.len() && n.inputs[i].source != Src::Absent && !n.inputs[i].name.is_empty()
}

fn str_attr(n: &NodeDesc, name: &str, dflt: &str) -> String {
    n.strings
        .get(name)
        .cloned()
        .unwrap_or_else(|| dflt.to_string())
}

fn norm_axis(axis: i64, rank: i32) -> i32 {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    a as i32
}

fn compatible_shape(a: &[i64], b: &[i64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&a, &b)| a < 0 || b < 0 || a == b)
}

// =============================================================================================
// TensorScatter (ai.onnx) — per-batch linear/circular cache writes, with exact native copies.
// =============================================================================================
fn tensor_scatter_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let past = ctx.resolve(&n.inputs[0])?;
    let update = ctx.resolve(&n.inputs[1])?;
    let rank = ctx.ndim(past) as i32;
    let axis = norm_axis(n.ints.get("axis").copied().unwrap_or(-2), rank);

    let shape = ctx.shape_of(past);
    let update_shape = ctx.shape_of(update);
    if shape.len() < 2
        || axis <= 0
        || axis >= rank
        || update_shape.len() != shape.len()
        || shape
            .iter()
            .zip(&update_shape)
            .enumerate()
            .any(|(i, (a, b))| i != axis as usize && a != b)
        || update_shape[axis as usize] > shape[axis as usize]
    {
        return Err("TensorScatter: incompatible cache/update geometry".to_string());
    }
    let capacity = i64::from(shape[axis as usize]);
    let length = i64::from(update_shape[axis as usize]);
    let circular = str_attr(n, "mode", "linear") == "circular";
    if !present(n, 2) {
        let output = if length == 0 {
            past
        } else {
            ctx.slice_update(past, update, &vec![0; shape.len()], &update_shape)?
        };
        let output = ctx.contiguous(output)?;
        ctx.bind(&n.outputs[0], output);
        return Ok(());
    }
    let wi = ctx.resolve(&n.inputs[2])?;
    if ctx.shape_of(wi) != [shape[0]] {
        return Err("TensorScatter: write_indices must have one entry per batch".to_string());
    }
    let indices = ctx.read_ints_eval(&n.inputs[2])?;
    if circular && capacity == 0 && length > 0 {
        return Err("TensorScatter: circular cache capacity must be positive".to_string());
    }
    let mut output = past;
    for (batch, &index) in indices.iter().enumerate() {
        if !circular && (index < 0 || index > capacity - length) {
            return Err(
                "TensorScatter: linear write index plus update length must fit in cache"
                    .to_string(),
            );
        }
        if length == 0 {
            continue;
        }
        // Normalize before adding length, so even INT64_MIN/MAX cannot overflow.
        let start = if circular {
            index.rem_euclid(capacity)
        } else {
            index
        };
        let first = length.min(capacity - start);
        for (source, destination, count) in [(0, start, first), (first, 0, length - first)] {
            if count == 0 {
                continue;
            }
            let mut lo = vec![0; shape.len()];
            let mut hi = update_shape.clone();
            lo[0] = batch as i32;
            hi[0] = batch as i32 + 1;
            lo[axis as usize] = source as i32;
            hi[axis as usize] = (source + count) as i32;
            let slab = ctx.slice(update, &lo, &hi)?;
            lo[axis as usize] = destination as i32;
            hi[axis as usize] = (destination + count) as i32;
            output = ctx.slice_update(output, slab, &lo, &hi)?;
        }
    }

    let cont = ctx.contiguous(output)?;
    ctx.bind(&n.outputs[0], cont);
    Ok(())
}

fn tensor_scatter_claim(node: &NodeView) -> ClaimResult {
    let ni = node.num_inputs();
    require!(
        (ni == 2 || ni == 3) && node.num_outputs() == 1,
        "expects 2-3 inputs and 1 output, got {}in/{}out",
        ni,
        node.num_outputs()
    );
    let mode = node.string_attr("mode", "linear");
    require!(
        mode == "linear" || mode == "circular",
        "mode must be \"linear\" or \"circular\", got {mode:?}"
    );
    let (past, update, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        is_movable(past.dtype) && update.dtype == past.dtype && out.dtype == past.dtype,
        "past/update/output must share one MLX copy dtype, got {} / {} -> {}",
        crate::registry::ort_dtype_name(past.dtype),
        crate::registry::ort_dtype_name(update.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        past.shape.len() >= 2
            && update.shape.len() == past.shape.len()
            && compatible_shape(&out.shape, &past.shape),
        "past_cache/output must have the same rank >=2 and shape; update must have the same rank"
    );
    let rank = past.shape.len() as i64;
    let axis = node.int_attr("axis", -2);
    require!(
        axis > -rank && axis < rank && axis != 0,
        "axis must name a non-batch dimension in [-{rank}, {}], got {axis}",
        rank - 1
    );
    let axis = norm_axis(axis, rank as i32) as usize;
    for dim in 0..past.shape.len() {
        if dim != axis && past.shape[dim] >= 0 && update.shape[dim] >= 0 {
            require!(
                past.shape[dim] == update.shape[dim],
                "update dimension {dim} must equal past_cache ({} != {})",
                update.shape[dim],
                past.shape[dim]
            );
        }
    }
    if past.shape[axis] >= 0 && update.shape[axis] >= 0 {
        require!(
            update.shape[axis] <= past.shape[axis],
            "update sequence length {} cannot exceed cache length {}",
            update.shape[axis],
            past.shape[axis]
        );
    }
    if ni == 3 && node.input_present(2) {
        let wi = match node.input_info(2) {
            Some(wi) if wi.dtype == T_INT64 => wi,
            Some(wi) => deny!(
                "write_indices must be int64, got {}",
                crate::registry::ort_dtype_name(wi.dtype)
            ),
            None => deny!("missing tensor type/shape info on write_indices"),
        };
        require!(
            wi.shape.len() == 1 && compatible_shape(&wi.shape, &past.shape[..1]),
            "write_indices must be rank-1 with one entry per batch"
        );
        if mode == "linear"
            && let Some(indices) = node.read_const_int64(2)
        {
            require!(
                indices.iter().all(|&index| index >= 0
                    && (past.shape[axis] < 0
                        || update.shape[axis] < 0
                        || index <= past.shape[axis] - update.shape[axis])),
                "linear TensorScatter write index plus update length must fit in the cache"
            );
        }
    }
    Ok(())
}

// =============================================================================================
// CausalConvWithState (com.microsoft) — stateful causal depthwise conv1d.
// =============================================================================================
fn causal_channels_first(
    ctx: &mut TranslationContext,
    array: mlx::mlx_array,
    channels: i32,
    window: bool,
) -> Result<mlx::mlx_array, MlxError> {
    let shape = ctx.shape_of(array);
    if window {
        let flat = ctx.reshape(array, &[shape[0], shape[1], shape[2], channels])?;
        ctx.transpose(flat, &[0, 1, 3, 2])
    } else {
        let flat = ctx.reshape(array, &[shape[0], shape[1], channels])?;
        ctx.transpose(flat, &[0, 2, 1])
    }
}

fn causal_channels_last(
    ctx: &mut TranslationContext,
    array: mlx::mlx_array,
    shape: &[i32],
    window: bool,
) -> Result<mlx::mlx_array, MlxError> {
    let transposed = ctx.transpose(array, if window { &[0, 1, 3, 2] } else { &[0, 2, 1] })?;
    ctx.reshape(transposed, shape)
}

fn causal_conv_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let mut x = ctx.resolve(&n.inputs[0])?;
    let input_shape = ctx.shape_of(x);
    let weight = ctx.resolve(&n.inputs[1])?; // (C, 1, k)
    let channels_last = n.ints.get("channels_last").copied().unwrap_or(0) == 1;
    let c = ctx.dim(weight, 0);
    if channels_last {
        x = causal_channels_first(ctx, x, c, false)?;
    }
    let dt = ctx.dtype_of(x);
    let b = ctx.dim(x, 0);
    let l = ctx.dim(x, 2);
    let k = ctx.dim(weight, 2);
    let dilation = n.ints.get("dilation").copied().unwrap_or(1) as i32;
    let carry = (k - 1)
        .checked_mul(dilation)
        .filter(|&v| v >= 0)
        .ok_or("CausalConvWithState: invalid receptive field")?;
    if c <= 0 || k <= 0 || ctx.dim(x, 1) != c || ctx.dim(weight, 1) != 1 {
        return Err("CausalConvWithState: incompatible input/weight geometry".to_string());
    }
    let state_window = *n.ints.get("state_window").unwrap_or(&0) as i32;

    let has_bias = present(n, 2);
    let has_state = present(n, 3);
    let mut state_shape = input_shape.clone();
    state_shape[if channels_last { 1 } else { 2 }] = carry;
    if state_window > 0 {
        state_shape.insert(0, state_window);
    }
    let past = if has_state {
        let past = ctx.resolve(&n.inputs[3])?;
        if ctx.shape_of(past) != state_shape {
            return Err("CausalConvWithState: incompatible past_state geometry".to_string());
        }
        Some(if channels_last {
            causal_channels_first(ctx, past, c, state_window > 0)?
        } else {
            past
        })
    } else {
        None
    };
    if has_bias {
        let bias = ctx.resolve(&n.inputs[2])?;
        if ctx.shape_of(bias) != [c] {
            return Err("CausalConvWithState: bias must have shape [channels]".to_string());
        }
    }

    // Carry is the full dilated receptive field, not just the number of preceding taps.
    let mut x_pad = x;
    if carry > 0 {
        let state = if let Some(state) = past {
            if state_window > 0 {
                let state = ctx.slice(
                    state,
                    &[state_window - 1, 0, 0, 0],
                    &[state_window, b, c, carry],
                )?;
                ctx.reshape(state, &[b, c, carry])?
            } else {
                state
            }
        } else {
            ctx.zeros(&[b, c, carry], dt)?
        };
        x_pad = ctx.concat2(state, x, 2)?;
    }

    if n.outputs.len() >= 2 && !n.outputs[1].name.is_empty() {
        let ps = if state_window > 0 {
            if carry > 0 && l > 0 {
                let kept = l.min(state_window);
                let leading = state_window - kept;
                let mut window = if leading > 0 {
                    Some(ctx.zeros(&[leading, b, c, carry], dt)?)
                } else {
                    None
                };
                for token in (l - kept)..l {
                    let step = ctx.slice(x_pad, &[0, 0, token + 1], &[b, c, token + 1 + carry])?;
                    let step = ctx.expand_dims(step, 0)?;
                    window = Some(match window {
                        Some(previous) => ctx.concat2(previous, step, 0)?,
                        None => step,
                    });
                }
                window.ok_or("MLX CausalConv state_window produced no states")?
            } else if l == 0
                && let Some(past) = past
            {
                past
            } else {
                ctx.zeros(&[state_window, b, c, carry], dt)?
            }
        } else if carry > 0 {
            let padded = ctx.dim(x_pad, 2);
            ctx.slice(x_pad, &[0, 0, padded - carry], &[b, c, padded])?
        } else {
            ctx.zeros(&[b, c, 0], dt)?
        };
        let ps = if channels_last {
            causal_channels_last(ctx, ps, &state_shape, state_window > 0)?
        } else {
            ps
        };
        let ps = ctx.contiguous(ps)?;
        ctx.bind(&n.outputs[1], ps);
    }
    if l == 0 || b == 0 {
        let y = ctx.zeros(&input_shape, dt)?;
        ctx.bind(&n.outputs[0], y);
        return Ok(());
    }

    // Depthwise conv1d: MLX uses NLC data and (C_out, kernel, C_in/groups) weights.
    let x_t = ctx.transpose(x_pad, &[0, 2, 1])?;
    let x_nlc = ctx.contiguous(x_t)?; // (B, k-1+L, C)
    let w_t = ctx.transpose(weight, &[0, 2, 1])?;
    let w_ckc = ctx.contiguous(w_t)?; // (C, k, 1)
    let y_nlc =
        ctx.emit(|res, s| unsafe { mlx::mlx_conv1d(res, x_nlc, w_ckc, 1, 0, dilation, c, s) })?;
    let y_t = ctx.transpose(y_nlc, &[0, 2, 1])?;
    let mut y = ctx.contiguous(y_t)?; // (B, C, L)

    if has_bias {
        let bias = ctx.resolve(&n.inputs[2])?; // (C,)
        let b3 = ctx.reshape(bias, &[1, c, 1])?;
        y = ctx.add(y, b3)?;
    }

    let activation = str_attr(n, "activation", "none");
    if activation == "silu" || activation == "swish" {
        let sig = ctx.emit(|res, s| unsafe { mlx::mlx_sigmoid(res, y, s) })?;
        y = ctx.mul(y, sig)?;
    }
    if channels_last {
        y = causal_channels_last(ctx, y, &input_shape, false)?;
    }
    let y = ctx.contiguous(y)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn causal_conv_claim(node: &NodeView) -> ClaimResult {
    let ni = node.num_inputs();
    require!((2..=4).contains(&ni), "expects 2-4 inputs, got {ni}");
    let no = node.num_outputs();
    require!(no == 2, "expects two outputs, got {no}");
    let standard = node.domain().is_empty();
    let ndim = node.int_attr("ndim", 1);
    let channels_last = node.int_attr("channels_last", 0);
    let dilation = node.int_attr("dilation", 1);
    let state_window = node.int_attr("state_window", 0);
    require!(
        ndim == 1,
        "only one-dimensional causal convolution is supported"
    );
    require!(
        (0..=1).contains(&channels_last),
        "channels_last must be 0 or 1"
    );
    require!(
        dilation > 0 && dilation <= i32::MAX as i64,
        "dilation must be a positive int32"
    );
    require!(
        !standard || (channels_last == 0 && dilation == 1 && state_window == 0),
        "standard CausalConvWithState does not define contrib layout/dilation/window attributes"
    );
    let (input, weight) = match (node.input_info(0), node.input_info(1)) {
        (Some(a), Some(b)) => (a, b),
        _ => deny!("missing tensor type/shape info on input or weight"),
    };
    require!(
        is_mlx_float(input.dtype) && weight.dtype == input.dtype,
        "input/weight must share one float dtype, got {} / {}",
        crate::registry::ort_dtype_name(input.dtype),
        crate::registry::ort_dtype_name(weight.dtype)
    );
    require!(
        (if channels_last == 1 {
            input.shape.len() >= 3
        } else {
            input.shape.len() == 3
        }) && weight.shape.len() == 3,
        "input layout and weight rank must match one-dimensional convolution, got ranks {} and {}",
        input.shape.len(),
        weight.shape.len()
    );
    let channels = if channels_last == 1 {
        input.shape[2..]
            .iter()
            .try_fold(1i64, |a, &b| if b > 0 { a.checked_mul(b) } else { None })
    } else {
        Some(input.shape[1])
    };
    require!(
        channels.is_some_and(|c| c > 0 && c <= i32::MAX as i64 && weight.shape[0] == c)
            && weight.shape[1] == 1
            && weight.shape[2] > 0,
        "positive static channel/kernel geometry and weight [channels,1,kernel] required"
    );
    let carry = (weight.shape[2] - 1).checked_mul(dilation);
    require!(
        carry.is_some_and(|c| c < i32::MAX as i64),
        "receptive field exceeds int32 geometry"
    );
    let carry = carry.unwrap();
    require!(
        (0..=8).contains(&state_window),
        "state_window must be in [0,8], got {state_window}"
    );
    let mut expected_state_shape = input.shape.clone();
    expected_state_shape[if channels_last == 1 { 1 } else { 2 }] = carry;
    if state_window > 0 {
        expected_state_shape.insert(0, state_window);
    }
    require!(
        matches!(node.output_info(0), Some(out) if out.dtype == input.dtype && compatible_shape(&out.shape, &input.shape)),
        "output must have input shape and dtype"
    );
    if node.input_present(2) {
        match node.input_info(2) {
            Some(b) if b.dtype == input.dtype && b.shape == [channels.unwrap()] => {}
            Some(b) => deny!(
                "bias must have input dtype {} and shape [{}], got {} {:?}",
                crate::registry::ort_dtype_name(input.dtype),
                channels.unwrap(),
                crate::registry::ort_dtype_name(b.dtype),
                b.shape
            ),
            None => deny!("missing tensor type/shape info on bias"),
        }
    }
    if node.input_present(3) {
        match node.input_info(3) {
            Some(p)
                if p.dtype == input.dtype && compatible_shape(&p.shape, &expected_state_shape) => {}
            Some(p) => deny!(
                "past_state must have dtype {} and shape {:?}, got {} {:?}",
                crate::registry::ort_dtype_name(input.dtype),
                expected_state_shape,
                crate::registry::ort_dtype_name(p.dtype),
                p.shape
            ),
            None => deny!("missing tensor type/shape info on past_state"),
        }
    }
    if let Some(present) = node.output_info(1) {
        require!(
            present.dtype == input.dtype && compatible_shape(&present.shape, &expected_state_shape),
            "present_state must have input dtype and shape {:?}",
            expected_state_shape
        );
    }
    let activation = node.string_attr("activation", "none");
    require!(
        activation == "none" || activation == "silu" || activation == "swish",
        "activation must be \"none\", \"silu\", or \"swish\", got {activation:?}"
    );
    Ok(())
}

// =============================================================================================
// LinearAttention (com.microsoft) — stable delta-rule recurrence and fused prefill.
// =============================================================================================
fn rule_uses_decay(rule: &str) -> bool {
    rule == "gated" || rule == "gated_delta"
}
fn rule_uses_beta(rule: &str) -> bool {
    rule == "delta" || rule == "gated_delta"
}
fn is_known_rule(rule: &str) -> bool {
    matches!(rule, "linear" | "gated" | "delta" | "gated_delta")
}

fn la_scalar(
    ctx: &mut TranslationContext,
    value: f32,
    dt: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    let s = ctx.scalar_f32(value);
    if dt == mlx::mlx_dtype__MLX_FLOAT32 {
        Ok(s)
    } else {
        ctx.astype(s, dt)
    }
}

/// From a (B, H, T, X) tensor pick time-step `t` as a (B, H, X) slab.
fn time_slab(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    t: i32,
    b: i32,
    h: i32,
    x: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let s = ctx.slice(a, &[0, 0, t, 0], &[b, h, t + 1, x])?;
    ctx.reshape(s, &[b, h, x])
}

/// From a (B, H, T) tensor pick time-step `t` as a (B, H) slab.
fn time_slab2(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    t: i32,
    b: i32,
    h: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let s = ctx.slice(a, &[0, 0, t], &[b, h, t + 1])?;
    ctx.reshape(s, &[b, h])
}

fn repeat_axis(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    repeats: i32,
    axis: i32,
) -> Result<mlx::mlx_array, MlxError> {
    if repeats == 1 {
        return Ok(a);
    }
    ctx.emit(|res, s| unsafe { mlx::mlx_repeat_axis(res, a, repeats, axis, s) })
}

// Each SIMD group owns one state column. Time remains sequential, avoiding the unstable WY
// inversion, but state lives in registers and independent columns/heads run in parallel.
const LINEAR_ATTENTION_SCAN_SRC: &str = r#"
    size_t column = thread_position_in_grid.x / 32;
    if (column >= size_t(B) * H * DV) { return; }
    uint lane = thread_position_in_grid.x % 32;
    uint vi = column % DV;
    size_t bh = column / DV;
    uint bi = bh / H;
    uint hi = bh % H;
    S state[(DK + 31) / 32];
    for (uint j = 0; j < (DK + 31) / 32; ++j) {
        uint ki = lane + j * 32;
        if (ki < DK) {
            state[j] = initial[(bh * DK + ki) * DV + vi];
            for (int w = 0; w < WINDOW - TIME; ++w) {
                present[((size_t(w) * B * H + bh) * DK + ki) * DV + vi] = S(0);
            }
        }
    }
    for (uint t = 0; t < TIME; ++t) {
        size_t kt = (bh * TIME + t) * DK;
        float retrieval = 0.0f;
        for (uint j = 0; j < (DK + 31) / 32; ++j) {
            uint ki = lane + j * 32;
            if (ki < DK) {
                if (DECAY) {
                    size_t gi = SCALAR_DECAY ? bh * TIME + t : kt + ki;
                    S gate = S(precise::exp(float(decay[gi])));
                    state[j] = S(float(state[j]) * float(gate));
                }
                retrieval += float(key[kt + ki]) * float(state[j]);
            }
        }
        S delta = value[(bh * TIME + t) * DV + vi];
        if (BETA) {
            S retrieved = S(simd_sum(retrieval));
            S difference = S(float(delta) - float(retrieved));
            delta = S(float(beta[bh * TIME + t]) * float(difference));
        }
        for (uint j = 0; j < (DK + 31) / 32; ++j) {
            uint ki = lane + j * 32;
            if (ki < DK) {
                S outer = S(float(key[kt + ki]) * float(delta));
                state[j] = S(float(state[j]) + float(outer));
                int w = int(t) - (TIME - WINDOW);
                if (WINDOW > 0 && w >= 0) {
                    present[((size_t(w) * B * H + bh) * DK + ki) * DV + vi] = state[j];
                }
            }
        }
        for (uint q = 0; q < HQ / H; ++q) {
            uint qh = hi * (HQ / H) + q;
            size_t qt = ((size_t(bi) * HQ + qh) * TIME + t) * DK;
            float readout = 0.0f;
            for (uint j = 0; j < (DK + 31) / 32; ++j) {
                uint ki = lane + j * 32;
                if (ki < DK) {
                    readout += float(query[qt + ki]) * float(state[j]);
                }
            }
            S result = S(simd_sum(readout));
            if (lane == 0) {
                output[((size_t(bi) * TIME + t) * HQ + qh) * DV + vi] = result;
            }
        }
    }
    if (WINDOW == 0) {
        for (uint j = 0; j < (DK + 31) / 32; ++j) {
            uint ki = lane + j * 32;
            if (ki < DK) {
                present[(bh * DK + ki) * DV + vi] = state[j];
            }
        }
    }
"#;

#[allow(clippy::too_many_arguments)]
fn linear_attention_scan(
    ctx: &mut TranslationContext,
    state: mlx::mlx_array,
    query: mlx::mlx_array,
    key: mlx::mlx_array,
    value: mlx::mlx_array,
    decay: Option<mlx::mlx_array>,
    beta: Option<mlx::mlx_array>,
    scalar_decay: bool,
    window: i32,
) -> Result<Option<(mlx::mlx_array, mlx::mlx_array)>, MlxError> {
    let disabled = std::env::var_os("ONNXRUNTIME_EP_MLX_NO_LINEAR_ATTENTION_SCAN")
        .is_some_and(|value| value != "0" && !value.is_empty());
    let shape = ctx.shape_of(value);
    let (b, h, time, dv) = (shape[0], shape[1], shape[2], shape[3]);
    let qshape = ctx.shape_of(query);
    let (hq, dk) = (qshape[1], qshape[3]);
    // Bound per-lane register state, not legal operator inputs. Wider shapes retain the unroll.
    if disabled || time <= 1 || b <= 0 || dk <= 0 || dk > 256 {
        return Ok(None);
    }
    let Some(threads) = b
        .checked_mul(h)
        .and_then(|x| x.checked_mul(dv))
        .and_then(|x| x.checked_mul(32))
    else {
        return Ok(None);
    };
    let mut device = unsafe { mlx::mlx_device_new() };
    let mut device_type = mlx::mlx_device_type__MLX_CPU;
    let mut rc = unsafe { mlx::mlx_stream_get_device(&mut device, ctx.stream()) };
    if rc == 0 {
        rc = unsafe { mlx::mlx_device_get_type(&mut device_type, device) };
    }
    unsafe { mlx::mlx_device_free(device) };
    if rc != 0 {
        return Err("LinearAttention: failed to inspect recurrence stream".to_string());
    }
    if device_type != mlx::mlx_device_type__MLX_GPU {
        return Ok(None);
    }

    let inputs = super::selective_scan::VecString::new(&[
        "initial", "query", "key", "value", "decay", "beta",
    ])?;
    let outputs = super::selective_scan::VecString::new(&["output", "present"])?;
    let source = CString::new(LINEAR_ATTENTION_SCAN_SRC).unwrap();
    let kernel = unsafe {
        mlx::mlx_fast_metal_kernel_new(
            c"onnxrt_mlx_linear_attention_scan".as_ptr(),
            inputs.as_raw(),
            outputs.as_raw(),
            source.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        )
    };
    if kernel.ctx.is_null() {
        return Err("LinearAttention: failed to construct fused recurrence".to_string());
    }
    let config = unsafe { mlx::mlx_fast_metal_kernel_config_new() };
    let dtype = ctx.dtype_of(state);
    rc = unsafe {
        mlx::mlx_fast_metal_kernel_config_add_template_arg_dtype(config, c"S".as_ptr(), dtype)
    };
    for (name, number) in [
        ("B", b),
        ("H", h),
        ("HQ", hq),
        ("TIME", time),
        ("DK", dk),
        ("DV", dv),
        ("WINDOW", window),
        ("DECAY", decay.is_some() as i32),
        ("BETA", beta.is_some() as i32),
        ("SCALAR_DECAY", scalar_decay as i32),
    ] {
        let name = CString::new(name).unwrap();
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_template_arg_int(config, name.as_ptr(), number)
        };
    }
    let output_shape = [b, time, hq * dv];
    let mut state_shape = vec![b, h, dk, dv];
    if window > 0 {
        state_shape.insert(0, window);
    }
    rc |= unsafe {
        mlx::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            output_shape.as_ptr(),
            output_shape.len(),
            dtype,
        )
    };
    rc |= unsafe {
        mlx::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            state_shape.as_ptr(),
            state_shape.len(),
            dtype,
        )
    };
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_grid(config, threads, 1, 1) };
    rc |= unsafe {
        mlx::mlx_fast_metal_kernel_config_set_thread_group(config, threads.min(128), 1, 1)
    };
    let mut arguments = crate::mlx::VectorArray::new();
    for array in [
        state,
        query,
        key,
        value,
        decay.unwrap_or(state),
        beta.unwrap_or(state),
    ] {
        arguments.append(array);
    }
    let mut results = crate::mlx::VectorArray::new();
    if rc == 0 {
        rc = unsafe {
            mlx::mlx_fast_metal_kernel_apply(
                results.as_mut_ptr(),
                kernel,
                arguments.as_raw(),
                config,
                ctx.stream(),
            )
        };
    }
    unsafe {
        mlx::mlx_fast_metal_kernel_config_free(config);
        mlx::mlx_fast_metal_kernel_free(kernel);
    }
    if rc != 0 || results.size() != 2 {
        return Err("LinearAttention: fused recurrence kernel failed".to_string());
    }
    Ok(Some((ctx.keep(results.get(0)), ctx.keep(results.get(1)))))
}

// The wide scan rounds only IO stores, not intermediate updates or inter-token working state.
const LINEAR_ATTENTION_SCAN_WIDE_SRC: &str = r#"
    size_t column = thread_position_in_grid.x / 32;
    if (column >= size_t(B) * H * DV) { return; }
    uint lane = thread_position_in_grid.x % 32;
    uint vi = column % DV;
    size_t bh = column / DV;
    uint bi = bh / H;
    uint hi = bh % H;
    float state[(DK + 31) / 32];
    for (uint j = 0; j < (DK + 31) / 32; ++j) {
        uint ki = lane + j * 32;
        if (ki < DK) {
            state[j] = float(initial[(bh * DK + ki) * DV + vi]);
            for (int w = 0; w < WINDOW - TIME; ++w) {
                present[((size_t(w) * B * H + bh) * DK + ki) * DV + vi] = S(0);
            }
        }
    }
    for (uint t = 0; t < TIME; ++t) {
        size_t kt = (bh * TIME + t) * DK;
        float retrieval = 0.0f;
        for (uint j = 0; j < (DK + 31) / 32; ++j) {
            uint ki = lane + j * 32;
            if (ki < DK) {
                if (DECAY) {
                    size_t gi = SCALAR_DECAY ? bh * TIME + t : kt + ki;
                    float gate = precise::exp(float(decay[gi]));
                    state[j] *= gate;
                }
                retrieval += key[kt + ki] * state[j];
            }
        }
        float delta = value[(bh * TIME + t) * DV + vi];
        if (BETA) {
            delta = beta[bh * TIME + t] * (delta - simd_sum(retrieval));
        }
        for (uint j = 0; j < (DK + 31) / 32; ++j) {
            uint ki = lane + j * 32;
            if (ki < DK) {
                float outer = key[kt + ki] * delta;
                state[j] += outer;
                int w = int(t) - (TIME - WINDOW);
                if (WINDOW > 0 && w >= 0) {
                    present[((size_t(w) * B * H + bh) * DK + ki) * DV + vi] = S(state[j]);
                }
            }
        }
        for (uint q = 0; q < HQ / H; ++q) {
            uint qh = hi * (HQ / H) + q;
            size_t qt = ((size_t(bi) * HQ + qh) * TIME + t) * DK;
            float readout = 0.0f;
            for (uint j = 0; j < (DK + 31) / 32; ++j) {
                uint ki = lane + j * 32;
                if (ki < DK) {
                    readout += query[qt + ki] * state[j];
                }
            }
            O result = O(simd_sum(readout) * scale);
            if (lane == 0) {
                output[((size_t(bi) * TIME + t) * HQ + qh) * DV + vi] = result;
            }
        }
    }
    if (WINDOW == 0) {
        for (uint j = 0; j < (DK + 31) / 32; ++j) {
            uint ki = lane + j * 32;
            if (ki < DK) {
                present[(bh * DK + ki) * DV + vi] = S(state[j]);
            }
        }
    }
"#;

fn linear_attention_scan_wide(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    state: mlx::mlx_array,
    scale: f32,
) -> Result<Option<(mlx::mlx_array, mlx::mlx_array)>, MlxError> {
    let disabled = std::env::var_os("ONNXRUNTIME_EP_MLX_NO_LINEAR_ATTENTION_SCAN")
        .is_some_and(|value| value != "0" && !value.is_empty());
    let shape = ctx.shape_of(state);
    let (b, h, dk, dv) = (shape[0], shape[1], shape[2], shape[3]);
    if disabled || b <= 0 || !(257..=512).contains(&dk) {
        return Ok(None);
    }
    let Some(threads) = b
        .checked_mul(h)
        .and_then(|x| x.checked_mul(dv))
        .and_then(|x| x.checked_mul(32))
    else {
        return Ok(None);
    };
    let mut device = unsafe { mlx::mlx_device_new() };
    let mut device_type = mlx::mlx_device_type__MLX_CPU;
    let mut rc = unsafe { mlx::mlx_stream_get_device(&mut device, ctx.stream()) };
    if rc == 0 {
        rc = unsafe { mlx::mlx_device_get_type(&mut device_type, device) };
    }
    unsafe { mlx::mlx_device_free(device) };
    if rc != 0 {
        return Err("LinearAttention: failed to inspect wide recurrence stream".to_string());
    }
    if device_type != mlx::mlx_device_type__MLX_GPU {
        return Ok(None);
    }

    // Re-resolve live inputs: the common preparation has already rounded them to state dtype.
    let query = ctx.resolve(&n.inputs[0])?;
    let key = ctx.resolve(&n.inputs[1])?;
    let value = ctx.resolve(&n.inputs[2])?;
    let qshape = ctx.shape_of(query);
    let time = qshape[1];
    if time <= 1 {
        return Ok(None);
    }
    let hq = qshape[2] / dk;
    let hk = ctx.shape_of(key)[2] / dk;
    let output_dtype = ctx.dtype_of(query);
    let state_dtype = ctx.dtype_of(state);
    let working_dtype = mlx::mlx_dtype__MLX_FLOAT32;
    let to_heads = |ctx: &mut TranslationContext,
                    a: mlx::mlx_array,
                    heads: i32,
                    last: i32|
     -> Result<mlx::mlx_array, MlxError> {
        let a = ctx.astype(a, working_dtype)?;
        let a = ctx.reshape(a, &[b, time, heads, last])?;
        ctx.transpose(a, &[0, 2, 1, 3])
    };
    let query = to_heads(ctx, query, hq, dk)?;
    let query = if hq < h {
        repeat_axis(ctx, query, h / hq, 1)?
    } else {
        query
    };
    let hq = hq.max(h);
    let key = to_heads(ctx, key, hk, dk)?;
    let key = repeat_axis(ctx, key, h / hk, 1)?;
    let value = to_heads(ctx, value, h, dv)?;
    let scale = la_scalar(ctx, scale, working_dtype)?;
    let rule = str_attr(n, "update_rule", "gated_delta");
    let mut scalar_decay = false;
    let decay = if rule_uses_decay(&rule) {
        let decay = ctx.resolve(&n.inputs[4])?;
        scalar_decay = ctx.shape_of(decay)[2] == h;
        Some(if scalar_decay {
            let decay = ctx.astype(decay, working_dtype)?;
            ctx.transpose(decay, &[0, 2, 1])?
        } else {
            to_heads(ctx, decay, h, dk)?
        })
    } else {
        None
    };
    let beta = if rule_uses_beta(&rule) {
        let beta = ctx.resolve(&n.inputs[5])?;
        let beta = ctx.astype(beta, working_dtype)?;
        let beta = ctx.transpose(beta, &[0, 2, 1])?;
        Some(if ctx.shape_of(beta)[1] == 1 && h > 1 {
            repeat_axis(ctx, beta, h, 1)?
        } else {
            beta
        })
    } else {
        None
    };
    let window = *n.ints.get("state_window").unwrap_or(&0) as i32;
    let output_width = hq
        .checked_mul(dv)
        .ok_or("LinearAttention: wide recurrence output width exceeds native index range")?;
    let inputs = super::selective_scan::VecString::new(&[
        "initial", "query", "key", "value", "decay", "beta", "scale",
    ])?;
    let outputs = super::selective_scan::VecString::new(&["output", "present"])?;
    let source = CString::new(LINEAR_ATTENTION_SCAN_WIDE_SRC).unwrap();
    let kernel = unsafe {
        mlx::mlx_fast_metal_kernel_new(
            c"onnxrt_mlx_linear_attention_scan_wide".as_ptr(),
            inputs.as_raw(),
            outputs.as_raw(),
            source.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        )
    };
    if kernel.ctx.is_null() {
        return Err("LinearAttention: failed to construct wide recurrence".to_string());
    }
    let config = unsafe { mlx::mlx_fast_metal_kernel_config_new() };
    rc = unsafe {
        mlx::mlx_fast_metal_kernel_config_add_template_arg_dtype(config, c"S".as_ptr(), state_dtype)
    };
    rc |= unsafe {
        mlx::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"O".as_ptr(),
            output_dtype,
        )
    };
    for (name, number) in [
        ("B", b),
        ("H", h),
        ("HQ", hq),
        ("TIME", time),
        ("DK", dk),
        ("DV", dv),
        ("WINDOW", window),
        ("DECAY", decay.is_some() as i32),
        ("BETA", beta.is_some() as i32),
        ("SCALAR_DECAY", scalar_decay as i32),
    ] {
        let name = CString::new(name).unwrap();
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_template_arg_int(config, name.as_ptr(), number)
        };
    }
    let output_shape = [b, time, output_width];
    let mut state_shape = vec![b, h, dk, dv];
    if window > 0 {
        state_shape.insert(0, window);
    }
    rc |= unsafe {
        mlx::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            output_shape.as_ptr(),
            output_shape.len(),
            output_dtype,
        )
    };
    rc |= unsafe {
        mlx::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            state_shape.as_ptr(),
            state_shape.len(),
            state_dtype,
        )
    };
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_grid(config, threads, 1, 1) };
    rc |= unsafe {
        mlx::mlx_fast_metal_kernel_config_set_thread_group(config, threads.min(128), 1, 1)
    };
    let mut arguments = crate::mlx::VectorArray::new();
    for array in [
        state,
        query,
        key,
        value,
        decay.unwrap_or(state),
        beta.unwrap_or(state),
        scale,
    ] {
        arguments.append(array);
    }
    let mut results = crate::mlx::VectorArray::new();
    if rc == 0 {
        rc = unsafe {
            mlx::mlx_fast_metal_kernel_apply(
                results.as_mut_ptr(),
                kernel,
                arguments.as_raw(),
                config,
                ctx.stream(),
            )
        };
    }
    unsafe {
        mlx::mlx_fast_metal_kernel_config_free(config);
        mlx::mlx_fast_metal_kernel_free(kernel);
    }
    if rc != 0 || results.size() != 2 {
        return Err("LinearAttention: wide recurrence kernel failed".to_string());
    }
    Ok(Some((ctx.keep(results.get(0)), ctx.keep(results.get(1)))))
}

fn linear_attention_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let rule = str_attr(n, "update_rule", "gated_delta");
    let uses_decay = rule_uses_decay(&rule);
    let uses_beta = rule_uses_beta(&rule);
    let hq = *n
        .ints
        .get("q_num_heads")
        .ok_or("MLX LinearAttention: q_num_heads missing")? as i32;
    let h = *n
        .ints
        .get("kv_num_heads")
        .ok_or("MLX LinearAttention: kv_num_heads missing")? as i32;
    let state_window = *n.ints.get("state_window").unwrap_or(&0) as i32;

    let query = ctx.resolve(&n.inputs[0])?; // (B, T, Hq*d_k)
    let key = ctx.resolve(&n.inputs[1])?; // (B, T, Hk*d_k)
    let value = ctx.resolve(&n.inputs[2])?; // (B, T, H*d_v)
    let dt = ctx.dtype_of(query);
    let state_dt = if present(n, 3) {
        let past = ctx.resolve(&n.inputs[3])?;
        ctx.dtype_of(past)
    } else if n.outputs.len() >= 2 && !n.outputs[1].name.is_empty() {
        mlx_dtype_from_onnx(n.outputs[1].otype)
    } else {
        dt
    };

    let qsh = ctx.shape_of(query);
    let ksh = ctx.shape_of(key);
    let vsh = ctx.shape_of(value);
    if qsh.len() != 3
        || ksh.len() != 3
        || vsh.len() != 3
        || qsh[..2] != ksh[..2]
        || qsh[..2] != vsh[..2]
        || qsh[2] <= 0
        || qsh[2] % hq != 0
        || vsh[2] <= 0
        || vsh[2] % h != 0
    {
        return Err("LinearAttention: incompatible query/key/value geometry".to_string());
    }
    let b = qsh[0];
    let t_len = qsh[1];
    // A dynamic (symbolic) time extent only survives into a SHAPELESS trace; the static-length
    // recurrence needs a concrete T. Fail the trace so the plan falls back to the eager translator
    // (or the shape-keyed general trace), where `shape_of` reports the real extent. The claim admits
    // dynamic-time nodes precisely because this guard keeps them correct.
    if t_len < 0 {
        return Err(
            "MLX LinearAttention: dynamic time extent in a shapeless trace — falls back to eager"
                .to_string(),
        );
    }
    let d_k = qsh[2] / hq;
    let d_v = vsh[2] / h;
    let n_k_heads = ksh[2] / d_k;
    if ksh[2] % d_k != 0 || n_k_heads <= 0 || h % n_k_heads != 0 {
        return Err("MLX LinearAttention: kv_num_heads must be divisible by key heads".to_string());
    }
    let output_heads = hq.max(h);
    let mut expected_state = vec![b, h, d_k, d_v];
    if state_window > 0 {
        expected_state.insert(0, state_window);
    }
    if present(n, 3) {
        let past = ctx.resolve(&n.inputs[3])?;
        if ctx.shape_of(past) != expected_state {
            return Err("LinearAttention: incompatible past_state geometry".to_string());
        }
    }
    for (index, widths) in [(4, [h, h * d_k]), (5, [1, h])] {
        if present(n, index) {
            let a = ctx.resolve(&n.inputs[index])?;
            let shape = ctx.shape_of(a);
            if shape.len() != 3 || shape[..2] != qsh[..2] || !widths.contains(&shape[2]) {
                return Err(format!(
                    "LinearAttention: incompatible input {index} geometry"
                ));
            }
        }
    }

    let scale_attr = n.floats.get("scale").copied().unwrap_or(0.0);
    let scale = if scale_attr != 0.0 {
        scale_attr
    } else {
        1.0 / (d_k as f32).sqrt()
    };

    let has_past = present(n, 3);
    let mut state = if has_past {
        let past = ctx.resolve(&n.inputs[3])?;
        if state_window > 0 {
            let past = ctx.slice(
                past,
                &[state_window - 1, 0, 0, 0, 0],
                &[state_window, b, h, d_k, d_v],
            )?;
            ctx.reshape(past, &[b, h, d_k, d_v])?
        } else {
            past
        }
    } else {
        ctx.zeros(&[b, h, d_k, d_v], state_dt)?
    };

    // Zero-length time axis: no steps run. output empty; present_state == state.
    if t_len == 0 {
        if !n.outputs.is_empty() && !n.outputs[0].name.is_empty() {
            let z = ctx.zeros(&[b, 0, output_heads * d_v], dt)?;
            ctx.bind(&n.outputs[0], z);
        }
        if n.outputs.len() >= 2 && !n.outputs[1].name.is_empty() {
            let ps = if state_window > 0 {
                if has_past {
                    ctx.resolve(&n.inputs[3])?
                } else {
                    ctx.zeros(&[state_window, b, h, d_k, d_v], state_dt)?
                }
            } else {
                state
            };
            let ps = ctx.contiguous(ps)?;
            ctx.bind(&n.outputs[1], ps);
        }
        return Ok(());
    }

    // 3D -> 4D (B, H, T, ·): reshape by head count, transpose heads before T, tile Q/K for GQA.
    let to_heads = |ctx: &mut TranslationContext,
                    a: mlx::mlx_array,
                    heads: i32,
                    last: i32|
     -> Result<mlx::mlx_array, MlxError> {
        let r = ctx.reshape(a, &[b, t_len, heads, last])?;
        ctx.transpose(r, &[0, 2, 1, 3])
    };
    let query = ctx.astype(query, state_dt)?;
    let key = ctx.astype(key, state_dt)?;
    let value = ctx.astype(value, state_dt)?;
    let q_heads = to_heads(ctx, query, hq, d_k)?;
    let q4 = if hq < h {
        repeat_axis(ctx, q_heads, h / hq, 1)? // inverse GQA: one Q head serves multiple KV states
    } else {
        q_heads
    };
    let k_heads = to_heads(ctx, key, n_k_heads, d_k)?;
    let k4 = repeat_axis(ctx, k_heads, h / n_k_heads, 1)?; // each K head serves one or more KV states
    let v4 = to_heads(ctx, value, h, d_v)?; // (B, H, T, d_v)
    let scale_s = la_scalar(ctx, scale, state_dt)?;
    let q4 = ctx.mul(q4, scale_s)?; // scaled query

    // Decay may be per-head-per-key-dim `[B,T,H*d_k]` (the op-test / ORT reference form) OR a
    // per-head SCALAR `[B,T,H]` (Qwen3.5 GatedDeltaNet exports one gate per value head). Detect it
    // from the input's last dim; the scalar form broadcasts across (d_k, d_v).
    let decay_per_head_scalar = uses_decay && {
        let d = ctx.resolve(&n.inputs[4])?;
        let ds = ctx.shape_of(d);
        ds.len() == 3 && ds[2] == h
    };
    let decay4 = if !uses_decay {
        None
    } else if decay_per_head_scalar {
        // [B,T,H] -> [B,H,T]
        let d = ctx.resolve(&n.inputs[4])?;
        let d = ctx.astype(d, state_dt)?;
        Some(ctx.transpose(d, &[0, 2, 1])?)
    } else {
        let d = ctx.resolve(&n.inputs[4])?;
        let d = ctx.astype(d, state_dt)?;
        Some(to_heads(ctx, d, h, d_k)?)
    };
    let beta3 = if uses_beta {
        let bta = ctx.resolve(&n.inputs[5])?;
        let bta = ctx.astype(bta, state_dt)?;
        let beta = ctx.transpose(bta, &[0, 2, 1])?;
        Some(if ctx.shape_of(beta)[1] == 1 && h > 1 {
            repeat_axis(ctx, beta, h, 1)?
        } else {
            beta
        })
    } else {
        None
    };

    if t_len > 1 {
        if let Some((output, present)) = linear_attention_scan(
            ctx,
            state,
            q4,
            k4,
            v4,
            decay4,
            beta3,
            decay_per_head_scalar,
            state_window,
        )? {
            if !n.outputs[0].name.is_empty() {
                let output = ctx.astype(output, dt)?;
                ctx.bind(&n.outputs[0], output);
            }
            if !n.outputs[1].name.is_empty() {
                ctx.bind(&n.outputs[1], present);
            }
            ctx.mark_fast("mlx_linear_attention_scan");
            return Ok(());
        }
        let bf16_wide = (257..=512).contains(&d_k)
            && (dt == mlx::mlx_dtype__MLX_BFLOAT16 || state_dt == mlx::mlx_dtype__MLX_BFLOAT16);
        if (257..=512).contains(&d_k) && !bf16_wide {
            // Unused common preparation is not an evaluation root, but its arena handles can
            // remain in compiled trace_transient until plan teardown. Keep that ownership cost.
            if let Some((output, present)) = linear_attention_scan_wide(ctx, n, state, scale)? {
                if !n.outputs[0].name.is_empty() {
                    ctx.bind(&n.outputs[0], output);
                }
                if !n.outputs[1].name.is_empty() {
                    ctx.bind(&n.outputs[1], present);
                }
                ctx.mark_fast("mlx_linear_attention_scan");
                return Ok(());
            }
        }
        ctx.mark_composed(if bf16_wide {
            "stable unroll: wide BF16 preserves state-dtype primitives"
        } else {
            "stable unroll: scan disabled, CPU/empty geometry, d_k > 512, or grid limit"
        });
    }

    // Zero/weak decay and unnormalized keys remain legal on every route.
    let mut outs: Vec<mlx::mlx_array> = Vec::with_capacity(t_len as usize);
    let mut state_history: Vec<mlx::mlx_array> = if state_window > 0 {
        Vec::with_capacity(state_window as usize)
    } else {
        Vec::new()
    };
    for t in 0..t_len {
        if let Some(decay4) = decay4 {
            let g = if decay_per_head_scalar {
                let slab = time_slab2(ctx, decay4, t, b, h)?; // (B, H)
                let g = ctx.emit(|res, s| unsafe { mlx::mlx_exp(res, slab, s) })?;
                let g = ctx.expand_dims(g, 2)?; // (B,H,1)
                ctx.expand_dims(g, 3)? // (B,H,1,1) — broadcasts over (d_k,d_v)
            } else {
                let slab = time_slab(ctx, decay4, t, b, h, d_k)?; // (B, H, d_k)
                let g = ctx.emit(|res, s| unsafe { mlx::mlx_exp(res, slab, s) })?;
                ctx.expand_dims(g, 3)? // (B,H,d_k,1)
            };
            state = ctx.mul(state, g)?;
        }
        let k_t = time_slab(ctx, k4, t, b, h, d_k)?; // (B, H, d_k)
        // retrieval = squeeze(k_row @ state)
        let k_row = ctx.expand_dims(k_t, 2)?; // (B,H,1,d_k)
        let retrieval_m = ctx.matmul(k_row, state)?; // (B,H,1,d_v)
        let retrieval = ctx.squeeze(retrieval_m, 2)?; // (B,H,d_v)

        let v_t = time_slab(ctx, v4, t, b, h, d_v)?; // (B, H, d_v)
        let delta = if let Some(beta3) = beta3 {
            let beta_t = time_slab2(ctx, beta3, t, b, h)?; // (B, H)
            let diff = ctx.sub(v_t, retrieval)?;
            let beta_e = ctx.expand_dims(beta_t, 2)?; // (B,H,1)
            ctx.mul(diff, beta_e)?
        } else {
            v_t
        };
        // outer = k_col @ delta_row: (B,H,d_k,1) @ (B,H,1,d_v) -> (B,H,d_k,d_v)
        let k_col = ctx.expand_dims(k_t, 3)?;
        let delta_row = ctx.expand_dims(delta, 2)?;
        let outer = ctx.matmul(k_col, delta_row)?;
        state = ctx.add(state, outer)?;
        if state_window > 0 {
            if state_history.len() == state_window as usize {
                state_history.remove(0);
            }
            state_history.push(state);
        }

        let readout_state = if hq > h {
            repeat_axis(ctx, state, hq / h, 1)?
        } else {
            state
        };
        let q_t = time_slab(ctx, q4, t, b, output_heads, d_k)?;
        let q_row = ctx.expand_dims(q_t, 2)?; // (B,H,1,d_k)
        let out_m = ctx.matmul(q_row, readout_state)?; // (B,Hout,1,d_v)
        let out_t = ctx.squeeze(out_m, 2)?; // (B,H,d_v)
        outs.push(out_t);
    }

    if !n.outputs.is_empty() && !n.outputs[0].name.is_empty() {
        // Assemble output (B, T, H*d_v): each step's (B, H, d_v) reshapes to (B, 1, H*d_v).
        let mut out = ctx.reshape(outs[0], &[b, 1, output_heads * d_v])?;
        for &step_out in outs.iter().skip(1) {
            let slab = ctx.reshape(step_out, &[b, 1, output_heads * d_v])?;
            out = ctx.concat2(out, slab, 1)?;
        }
        let out = ctx.contiguous(out)?;
        let out = ctx.astype(out, dt)?;
        ctx.bind(&n.outputs[0], out);
    }
    if n.outputs.len() >= 2 && !n.outputs[1].name.is_empty() {
        let ps = if state_window > 0 {
            let kept = t_len.min(state_window);
            let leading = state_window - kept;
            let mut window = if leading > 0 {
                Some(ctx.zeros(&[leading, b, h, d_k, d_v], state_dt)?)
            } else {
                None
            };
            for &step_state in &state_history {
                let slot = ctx.expand_dims(step_state, 0)?;
                window = Some(match window {
                    Some(previous) => ctx.concat2(previous, slot, 0)?,
                    None => slot,
                });
            }
            window
                .ok_or_else(|| "MLX LinearAttention state_window produced no states".to_string())?
        } else {
            state
        };
        let ps = ctx.contiguous(ps)?;
        ctx.bind(&n.outputs[1], ps);
    }
    Ok(())
}

fn linear_attention_common_claim(node: &NodeView, allow_mixed_state: bool) -> ClaimResult {
    require!(
        (3..=6).contains(&node.num_inputs()) && node.num_outputs() == 2,
        "expects 3-6 inputs and two outputs, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let state_window = node.int_attr("state_window", 0);
    require!(
        (0..=8).contains(&state_window),
        "state_window must be in [0,8], got {state_window}"
    );
    let rule = node.string_attr("update_rule", "gated_delta");
    require!(is_known_rule(&rule), "unsupported update_rule {rule:?}");
    let hq = node.int_attr("q_num_heads", 0);
    let h = node.int_attr("kv_num_heads", 0);
    require!(
        hq > 0
            && h > 0
            && hq <= i32::MAX as i64
            && h <= i32::MAX as i64
            && (hq % h == 0 || h % hq == 0),
        "q_num_heads and kv_num_heads must be positive and one must divide the other, got q={hq}, kv={h}"
    );
    let (q, k, v) = match (node.input_info(0), node.input_info(1), node.input_info(2)) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => deny!("missing tensor type/shape info on query, key, or value"),
    };
    require!(
        is_mlx_float(q.dtype) && k.dtype == q.dtype && v.dtype == q.dtype,
        "query/key/value must share one float dtype, got {} / {} / {}",
        crate::registry::ort_dtype_name(q.dtype),
        crate::registry::ort_dtype_name(k.dtype),
        crate::registry::ort_dtype_name(v.dtype)
    );
    require!(
        q.shape.len() == 3,
        "query must be rank 3, got shape {:?}",
        q.shape
    );
    require!(
        q.shape[2] > 0 && q.shape[2] % hq == 0,
        "query hidden size must be a positive multiple of q_num_heads"
    );
    let d_k = q.shape[2] / hq;
    require!(
        k.shape.len() == 3
            && k.shape[2] > 0
            && k.shape[2] % d_k == 0
            && h % (k.shape[2] / d_k) == 0,
        "key heads inferred from key hidden size must divide kv_num_heads"
    );
    require!(
        v.shape.len() == 3 && v.shape[2] > 0 && v.shape[2] % h == 0,
        "value hidden size must be a positive multiple of kv_num_heads"
    );
    let d_v = v.shape[2] / h;
    require!(
        compatible_shape(&q.shape[..2], &k.shape[..2])
            && compatible_shape(&q.shape[..2], &v.shape[..2]),
        "query/key/value batch and time dimensions must match"
    );
    require!(
        matches!(node.output_info(0), Some(out) if out.dtype == q.dtype && compatible_shape(&out.shape, &[q.shape[0], q.shape[1], hq.max(h) * d_v])),
        "output must have query dtype and shape [B,T,max(q_num_heads,kv_num_heads)*d_v]"
    );
    let expected_state_shape = if state_window > 0 {
        vec![state_window, q.shape[0], h, d_k, d_v]
    } else {
        vec![q.shape[0], h, d_k, d_v]
    };
    if let Some(past) = node.input_info(3) {
        require!(
            compatible_shape(&past.shape, &expected_state_shape),
            "past_state shape {:?} must be {:?}",
            past.shape,
            expected_state_shape
        );
    }
    if let Some(present_state) = node.output_info(1) {
        require!(
            compatible_shape(&present_state.shape, &expected_state_shape)
                && (if allow_mixed_state {
                    is_mlx_float(present_state.dtype)
                } else {
                    present_state.dtype == q.dtype
                }),
            "present_state shape {:?} must be {:?}",
            present_state.shape,
            expected_state_shape
        );
    }
    // The time dimension may be DYNAMIC (symbolic `[-1]`, as in real Qwen3.5/Qwen3-Next decoder
    // exports): the shape-keyed general trace resolves the concrete extent at trace time, and the
    // handler specializes the recurrence (falling back to eager if it is ever traced shapeless).
    let float_ok = |i: usize| -> bool {
        if !node.input_present(i) {
            return true;
        }
        matches!(node.input_info(i), Some(info) if info.dtype == q.dtype)
    };
    if allow_mixed_state {
        if node.input_present(3) {
            require!(
                matches!(node.input_info(3), Some(info) if is_mlx_float(info.dtype)),
                "past_state must have an MLX float dtype"
            );
        }
    } else {
        require!(
            float_ok(3),
            "past_state dtype must match query dtype {}",
            crate::registry::ort_dtype_name(q.dtype)
        );
    }
    require!(
        float_ok(4),
        "decay dtype must match query dtype {}",
        crate::registry::ort_dtype_name(q.dtype)
    );
    require!(
        float_ok(5),
        "beta dtype must match query dtype {}",
        crate::registry::ort_dtype_name(q.dtype)
    );
    require!(
        !rule_uses_decay(&rule) || node.input_present(4),
        "update_rule {rule:?} requires the decay input"
    );
    require!(
        !rule_uses_beta(&rule) || node.input_present(5),
        "update_rule {rule:?} requires the beta input"
    );
    if let Some(decay) = node.input_info(4) {
        require!(
            decay.shape.len() == 3
                && compatible_shape(&decay.shape[..2], &q.shape[..2])
                && (decay.shape[2] == h || decay.shape[2] == h * d_k),
            "decay must have shape [B,T,kv_num_heads] or [B,T,kv_num_heads*d_k]"
        );
    }
    if node.input_present(5) {
        let beta = node.input_info(5).unwrap();
        require!(
            beta.shape.len() == 3
                && compatible_shape(&beta.shape[..2], &q.shape[..2])
                && (beta.shape[2] == 1 || beta.shape[2] == h),
            "beta must have shape [B,T,1] or [B,T,kv_num_heads]"
        );
    }
    Ok(())
}

fn linear_attention_claim(node: &NodeView) -> ClaimResult {
    linear_attention_common_claim(node, false)
}

fn linear_attention_standard_claim(node: &NodeView) -> ClaimResult {
    linear_attention_common_claim(node, true)?;
    require!(
        node.int_attr("state_window", 0) == 0,
        "standard LinearAttention does not define state_window"
    );
    let q = node.input_info(0).unwrap();
    let k = node.input_info(1).unwrap();
    let v = node.input_info(2).unwrap();
    let out = node.output_info(0).unwrap();
    let hq = node.int_attr("q_num_heads", 0);
    let h = node.int_attr("kv_num_heads", 0);
    require!(
        hq % h == 0,
        "ai.onnx LinearAttention requires q_num_heads divisible by kv_num_heads"
    );
    let d_k = q.shape[2] / hq;
    let d_v = v.shape[2] / h;
    require!(
        k.shape[2] == h * d_k,
        "ai.onnx LinearAttention key must use kv_num_heads heads"
    );
    require!(
        out.shape.len() == 3 && out.shape[2] == hq * d_v,
        "ai.onnx LinearAttention output must have hidden size q_num_heads*d_v"
    );
    if let Some(state_out) = node.output_info(1) {
        require!(
            is_mlx_float(state_out.dtype),
            "ai.onnx LinearAttention present_state must have an MLX float dtype"
        );
        if let Some(state_in) = node.input_info(3) {
            require!(
                state_out.dtype == state_in.dtype,
                "present_state dtype must match past_state dtype"
            );
        }
    }
    Ok(())
}

// ---- ORT 1.29 fused gates ----------------------------------------------------------------------

fn gated_add_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let y = ctx.resolve(&n.inputs[1])?;
    let gate = ctx.resolve(&n.inputs[2])?;
    let scaled = ctx.mul(y, gate)?;
    let out = ctx.add(x, scaled)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn gated_add_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 3 && node.num_outputs() == 1,
        "GatedAdd expects 3 inputs and 1 output"
    );
    let (x, y, gate, out) = match (
        node.input_info(0),
        node.input_info(1),
        node.input_info(2),
        node.output_info(0),
    ) {
        (Some(x), Some(y), Some(gate), Some(out)) => (x, y, gate, out),
        _ => deny!("GatedAdd requires typed input/output tensors"),
    };
    require!(
        is_mlx_float(x.dtype)
            && y.dtype == x.dtype
            && gate.dtype == x.dtype
            && out.dtype == x.dtype,
        "X, Y, gate, and output must share one MLX float dtype"
    );
    require!(
        !x.shape.is_empty() && x.shape.last().is_some_and(|&d| d > 0),
        "X must have rank >= 1 and a positive static last dimension"
    );
    require!(
        y.shape == x.shape && out.shape == x.shape,
        "Y and output must have the same shape as X"
    );
    let mut gate_shape = x.shape.clone();
    *gate_shape.last_mut().unwrap() = 1;
    require!(
        gate.shape == gate_shape,
        "gate shape must equal X shape with the last dimension replaced by 1"
    );
    Ok(())
}

fn linear_attention_gate_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(a);
    let out_dt = ctx.dtype_of(a);
    let a32 = ctx.astype(a, mlx::mlx_dtype__MLX_FLOAT32)?;
    let bias = ctx.resolve(&n.inputs[1])?;
    let scale = ctx.resolve(&n.inputs[2])?;
    if shape.len() != 3 || ctx.shape_of(bias) != [shape[2]] || ctx.shape_of(scale) != [shape[2]] {
        return Err("LinearAttentionGate: parameters must have shape [H]".to_string());
    }
    let shifted = ctx.add(a32, bias)?;
    let zero = ctx.zeros_like(shifted)?;
    let softplus = ctx.binary(mlx::mlx_logaddexp, zero, shifted)?;
    let decay32 = ctx.mul(softplus, scale)?;
    let decay = ctx.astype(decay32, out_dt)?;
    ctx.bind(&n.outputs[0], decay);

    if n.outputs.len() > 1 && !n.outputs[1].name.is_empty() {
        let b = ctx.resolve(&n.inputs[3])?;
        if ctx.shape_of(b) != shape {
            return Err("LinearAttentionGate: b must have the same shape as a".to_string());
        }
        let b32 = ctx.astype(b, mlx::mlx_dtype__MLX_FLOAT32)?;
        let beta32 = ctx.unary(mlx::mlx_sigmoid, b32)?;
        let beta = ctx.astype(beta32, out_dt)?;
        ctx.bind(&n.outputs[1], beta);
    }
    Ok(())
}

fn linear_attention_gate_claim(node: &NodeView) -> ClaimResult {
    require!(
        (3..=4).contains(&node.num_inputs()) && (1..=2).contains(&node.num_outputs()),
        "LinearAttentionGate expects 3-4 input slots and 1-2 outputs"
    );
    let a = match node.input_info(0) {
        Some(a) if is_mlx_float(a.dtype) => a,
        Some(_) => deny!("a must have an MLX float dtype"),
        None => deny!("a lacks tensor type/shape info"),
    };
    require!(a.shape.len() == 3, "a must be a rank-3 [B,T,H] tensor");
    for idx in [1usize, 2] {
        let param = match node.input_info(idx) {
            Some(param) => param,
            None => deny!("parameter input {idx} lacks tensor type/shape info"),
        };
        require!(
            param.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                && compatible_shape(&param.shape, &[a.shape[2]]),
            "dt_bias and decay_scale must be float32 [H]"
        );
    }
    let decay = match node.output_info(0) {
        Some(decay) => decay,
        None => deny!("decay output lacks tensor type/shape info"),
    };
    require!(
        decay.dtype == a.dtype && compatible_shape(&decay.shape, &a.shape),
        "decay output must match a"
    );
    if node.output_present(1) {
        require!(
            node.input_present(3),
            "b is required when beta output is requested"
        );
        let b = match node.input_info(3) {
            Some(b) => b,
            None => deny!("b lacks tensor type/shape info"),
        };
        let beta = match node.output_info(1) {
            Some(beta) => beta,
            None => deny!("beta output lacks tensor type/shape info"),
        };
        require!(
            b.dtype == a.dtype
                && compatible_shape(&b.shape, &a.shape)
                && beta.dtype == a.dtype
                && compatible_shape(&beta.shape, &a.shape),
            "b and beta must match a"
        );
    }
    Ok(())
}

fn gated_rms_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let scale = ctx.resolve(&n.inputs[1])?;
    let gate = ctx.resolve(&n.inputs[2])?;
    let out_dt = ctx.dtype_of(x);
    let x_shape = ctx.shape_of(x);
    let c = ctx.dim(scale, 0);
    let rows = ctx.size_of(x) as i32 / c;

    let x32 = ctx.astype(x, mlx::mlx_dtype__MLX_FLOAT32)?;
    let x2 = ctx.reshape(x32, &[rows, c])?;
    let squared = ctx.mul(x2, x2)?;
    let mean = ctx.emit(|res, s| unsafe { mlx::mlx_mean_axis(res, squared, 1, true, s) })?;
    let eps = ctx.scalar_f32(n.floats.get("epsilon").copied().unwrap_or(1e-5));
    let denom = ctx.add(mean, eps)?;
    let inv = ctx.unary(mlx::mlx_rsqrt, denom)?;
    let normalized = ctx.mul(x2, inv)?;
    let scale32 = ctx.astype(scale, mlx::mlx_dtype__MLX_FLOAT32)?;
    let normalized = ctx.mul(normalized, scale32)?;

    let gate32 = ctx.astype(gate, mlx::mlx_dtype__MLX_FLOAT32)?;
    let gate2 = ctx.reshape(gate32, &[rows, c])?;
    let sigmoid = ctx.unary(mlx::mlx_sigmoid, gate2)?;
    let activation = match str_attr(n, "activation", "silu").as_str() {
        "silu" | "swish" => ctx.mul(gate2, sigmoid)?,
        "sigmoid" => sigmoid,
        activation => {
            return Err(format!(
                "GatedRMSNorm: unsupported activation {activation:?}"
            ));
        }
    };
    let out32 = ctx.mul(normalized, activation)?;
    let out = ctx.astype(out32, out_dt)?;
    let out = ctx.reshape(out, &x_shape)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn gated_rms_norm_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 3 && node.num_outputs() == 1,
        "GatedRMSNorm expects 3 inputs and 1 output"
    );
    let activation = node.string_attr("activation", "silu");
    require!(
        matches!(activation.as_str(), "silu" | "swish" | "sigmoid"),
        "GatedRMSNorm activation must be silu, swish, or sigmoid"
    );
    let (x, scale, gate, out) = match (
        node.input_info(0),
        node.input_info(1),
        node.input_info(2),
        node.output_info(0),
    ) {
        (Some(x), Some(scale), Some(gate), Some(out)) => (x, scale, gate, out),
        _ => deny!("GatedRMSNorm requires typed input/output tensors"),
    };
    require!(
        is_mlx_float(x.dtype)
            && scale.dtype == x.dtype
            && gate.dtype == x.dtype
            && out.dtype == x.dtype,
        "X, scale, gate, and Y must share one MLX float dtype"
    );
    require!(
        !x.shape.is_empty() && x.shape.iter().all(|&d| d > 0),
        "X must have a static positive shape"
    );
    require!(
        gate.shape == x.shape && out.shape == x.shape,
        "gate and Y must have the same shape as X"
    );
    require!(
        scale.shape.len() == 1
            && scale.shape[0] > 0
            && x.shape.last().unwrap() % scale.shape[0] == 0,
        "scale must be [C] and X's last dimension must be a multiple of C"
    );
    Ok(())
}

// ---- registration -------------------------------------------------------------------------------

pub fn register(registry: &mut OpRegistry) {
    registry.register_shape_keyed(
        OpRegistration {
            domain: "",
            op_type: "TensorScatter",
            min_opset: 24,
            max_opset: K_ANY_OPSET,
            handler: tensor_scatter_op as OpHandler,
            claim: tensor_scatter_claim as ClaimPredicate,
        },
        crate::registry::MLX_SLICE_SHAPE_REASON,
    );
    registry.register_shape_keyed(
        OpRegistration {
            domain: "com.microsoft",
            op_type: "CausalConvWithState",
            min_opset: K_ANY_OPSET,
            max_opset: K_ANY_OPSET,
            handler: causal_conv_op as OpHandler,
            claim: causal_conv_claim as ClaimPredicate,
        },
        crate::registry::MLX_SLICE_SHAPE_REASON,
    );
    registry.register_shape_keyed(
        OpRegistration {
            domain: "",
            op_type: "CausalConvWithState",
            min_opset: 27,
            max_opset: K_ANY_OPSET,
            handler: causal_conv_op as OpHandler,
            claim: causal_conv_claim as ClaimPredicate,
        },
        crate::registry::MLX_SLICE_SHAPE_REASON,
    );
    registry.register_shape_keyed(
        OpRegistration {
            domain: "com.microsoft",
            op_type: "LinearAttention",
            min_opset: K_ANY_OPSET,
            max_opset: K_ANY_OPSET,
            handler: linear_attention_op as OpHandler,
            claim: linear_attention_claim as ClaimPredicate,
        },
        "emits MLX Slice or CustomKernel without Primitive::output_shapes",
    );
    registry.register_shape_keyed(
        OpRegistration {
            domain: "",
            op_type: "LinearAttention",
            min_opset: 27,
            max_opset: K_ANY_OPSET,
            handler: linear_attention_op as OpHandler,
            claim: linear_attention_standard_claim as ClaimPredicate,
        },
        "emits MLX Slice or CustomKernel without Primitive::output_shapes",
    );
    registry.register_shape_keyed(
        OpRegistration {
            domain: "com.microsoft",
            op_type: "LinearAttentionGate",
            min_opset: K_ANY_OPSET,
            max_opset: K_ANY_OPSET,
            handler: linear_attention_gate_op as OpHandler,
            claim: linear_attention_gate_claim as ClaimPredicate,
        },
        "conservatively validates concrete gate/parameter geometry",
    );
    for (op_type, handler, claim) in [
        (
            "GatedAdd",
            gated_add_op as OpHandler,
            gated_add_claim as ClaimPredicate,
        ),
        (
            "GatedRMSNorm",
            gated_rms_norm_op as OpHandler,
            gated_rms_norm_claim as ClaimPredicate,
        ),
    ] {
        registry.register_shapeless(OpRegistration {
            domain: "com.microsoft",
            op_type,
            min_opset: K_ANY_OPSET,
            max_opset: K_ANY_OPSET,
            handler,
            claim,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{engine::Plan, mlx::Stream};

    #[test]
    fn linear_attention_scan_declines_cpu_stream() {
        let stream = Stream::new_default_cpu();
        let mut plan = Plan::new(Vec::new());
        let mut ctx = TranslationContext::new(
            &mut plan,
            std::ptr::null(),
            std::ptr::null_mut(),
            stream.as_raw(),
        );
        let state = ctx
            .zeros(&[1, 1, 3, 2], mlx::mlx_dtype__MLX_FLOAT32)
            .unwrap();
        let query = ctx
            .zeros(&[1, 1, 5, 3], mlx::mlx_dtype__MLX_FLOAT32)
            .unwrap();
        let value = ctx
            .zeros(&[1, 1, 5, 2], mlx::mlx_dtype__MLX_FLOAT32)
            .unwrap();
        assert!(
            linear_attention_scan(&mut ctx, state, query, query, value, None, None, true, 0)
                .unwrap()
                .is_none()
        );
    }
}
