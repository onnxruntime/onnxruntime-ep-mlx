//! Attention op handlers: the scaled-dot-product-attention family mapping onto the MLX fast SDPA
//! primitive. Faithful port of the C++ `ops/attention.cc` + `ops/attention_ext.cc`:
//!
//!   * GroupQueryAttention (com.microsoft) — separate Q/K/V, in-op RoPE + KV-cache append + causal
//!     SDPA, softcap/sinks and optional before/after-softmax diagnostics. Decode-critical.
//!   * Attention (ai.onnx opset 23-28) — MHA / GQA / MQA, 3D (B,S,H*hd) or 4D (B,H,S,hd),
//!     optional attn_mask, nonpad lengths, causal/window masks, logit soft-cap, QK outputs, and
//!     in-op past/present KV concat.
//!   * Attention (com.microsoft) — fused QKV projection followed by MHA.
//!   * MultiHeadAttention (com.microsoft)  — separate Q/K/V with optional projection bias,
//!     unidirectional (causal), custom scale.
//!   * RotaryEmbedding (ai.onnx opset 23 & com.microsoft) — standalone RoPE (rotate-half or
//!     interleaved, partial rotation), consuming exact supplied cos/sin cache values.
//!
//! Every op honors the resolved input dtype (fp32/fp16/bf16). GQA head broadcast (q_num_heads a
//! multiple of kv_num_heads) is handled inside MLX SDPA, so K/V are passed with their own head count.
//! Compiled decoder RoPE receives exact cache rows as data; ragged/local/bidirectional GQA and
//! PagedAttention read live sequence geometry on the eager path.

use std::os::raw::{c_char, c_void};

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::mlx::{Array, VectorArray};
use crate::registry::{
    ClaimResult, CompileShapeSafety, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry,
    is_mlx_float,
};
use crate::sys::mlx;
use crate::sys::ort;
use crate::{deny, require};

// ---- small local MLX helpers -------------------------------------------------------------------

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

fn capped_attention_logits(
    ctx: &mut TranslationContext,
    q: mlx::mlx_array,
    k: mlx::mlx_array,
    scale: f32,
    softcap: f32,
) -> Result<mlx::mlx_array, MlxError> {
    let kt = ctx.transpose(k, &[0, 1, 3, 2])?;
    let scores = ctx.matmul(q, kt)?;
    let scale = ctx.scalar_f32(scale);
    let scores = ctx.mul(scores, scale)?;
    if softcap <= 0.0 {
        return Ok(scores);
    }
    let cap = ctx.scalar_f32(softcap);
    let scores = ctx.binary(mlx::mlx_divide, scores, cap)?;
    let scores = ctx.unary(mlx::mlx_tanh, scores)?;
    ctx.mul(scores, cap)
}

/// A sink contributes only to the denominator, never to the returned key probabilities.
fn attention_probs_with_sink(
    ctx: &mut TranslationContext,
    scores: mlx::mlx_array,
    sink: Option<mlx::mlx_array>,
) -> Result<mlx::mlx_array, MlxError> {
    let Some(sink) = sink else {
        return ctx.softmax_axis(scores, 3);
    };
    let shape = ctx.shape_of(scores);
    let sink = ctx.astype(sink, ctx.dtype_of(scores))?;
    let sink = ctx.reshape(sink, &[1, shape[1], 1, 1])?;
    let target = [shape[0], shape[1], shape[2], 1];
    let sink = ctx.emit(|res, st| unsafe {
        mlx::mlx_broadcast_to(res, sink, target.as_ptr(), target.len(), st)
    })?;
    let logits = concat2(ctx, scores, sink, 3)?;
    let probs = ctx.softmax_axis(logits, 3)?;
    slice(ctx, probs, &[0, 0, 0, 0], &shape)
}

fn gqa_composed(n: &NodeDesc) -> bool {
    n.floats.get("softcap").copied().unwrap_or(0.0) > 0.0
        || attr_int(n, "qk_output", 0) != 0
        || attr_int(n, "smooth_softmax", 0) == 1
        || present(n, 11)
}

/// A row-major slice [start, stop) with unit stride over all axes.
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

fn take_1d_range(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    start: i32,
    stop: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let idx = ctx.arange(start as f64, stop as f64, 1.0, mlx::mlx_dtype__MLX_INT32)?;
    ctx.emit(|res, s| unsafe { mlx::mlx_take_axis(res, a, idx, 0, s) })
}

/// Concatenate two arrays along `axis`.
fn concat2(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
    axis: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let mut vec = VectorArray::new();
    vec.append(a);
    vec.append(b);
    ctx.emit(|res, s| unsafe { mlx::mlx_concatenate_axis(res, vec.as_raw(), axis, s) })
}

/// Run MLX fast SDPA. `mask_mode` is a NUL-terminated byte string ("", "causal", or "array");
/// `mask` is used only for "array".
fn sdpa(
    ctx: &mut TranslationContext,
    q: mlx::mlx_array,
    k: mlx::mlx_array,
    v: mlx::mlx_array,
    scale: f32,
    mask_mode: &[u8],
    mask: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    let mask = if mask_mode == b"array\0" && ctx.shapeless_compile() {
        // MLX SDPA otherwise inserts a one-input Broadcast whose target captures the first KV
        // length. Materialize the full [B,H,Q,K] mask through a two-input broadcast so shapeless
        // decode can infer the changed K extent from live inputs on every call.
        let q_zeros = ctx.zeros_like(q)?;
        let head_template =
            ctx.emit(|res, s| unsafe { mlx::mlx_sum_axis(res, q_zeros, 3, true, s) })?;
        if ctx.dtype_of(mask) == mlx::mlx_dtype__MLX_BOOL {
            let head_template = ctx.astype(head_template, mlx::mlx_dtype__MLX_BOOL)?;
            ctx.binary(mlx::mlx_logical_or, mask, head_template)?
        } else {
            ctx.add(mask, head_template)?
        }
    } else {
        mask
    };
    let mode = mask_mode.as_ptr() as *const c_char;
    ctx.mark_fast("mlx_fast_scaled_dot_product_attention");
    ctx.emit(|res, s| unsafe {
        mlx::mlx_fast_scaled_dot_product_attention(
            res,
            q,
            k,
            v,
            scale,
            mode,
            mask,
            empty_array(),
            false,
            s,
        )
    })
}

/// Dispatch SDPA over the (mutually exclusive) causal / array-mask / no-mask cases.
// The dispatch needs all kernel inputs to preserve the separate mask cases.
#[allow(clippy::too_many_arguments)]
fn sdpa_dispatch(
    ctx: &mut TranslationContext,
    q: mlx::mlx_array,
    k: mlx::mlx_array,
    v: mlx::mlx_array,
    scale: f32,
    causal: bool,
    mask: Option<mlx::mlx_array>,
    compute_dtype: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    if causal {
        return sdpa(ctx, q, k, v, scale, b"causal\0", empty_array());
    }
    if let Some(mut m) = mask {
        // Bool masks stay bool (True = attend); float additive masks cast to the compute dtype.
        if ctx.dtype_of(m) != mlx::mlx_dtype__MLX_BOOL {
            m = ctx.astype(m, compute_dtype)?;
        }
        return sdpa(ctx, q, k, v, scale, b"array\0", m);
    }
    sdpa(ctx, q, k, v, scale, b"\0", empty_array())
}

/// Build an ONNX-semantics causal additive mask `[q_len, k_len]` for SDPA.
///
/// ONNX `Attention` `is_causal` masks query `i` against key `j` iff `j > past_seq + i`
/// (`onnx/reference/ops/op_attention.py::_apply_causal`), i.e. an upper-left alignment where the
/// first `past_seq` keys are always visible and the causal triangle starts at the new-key region.
/// MLX's built-in `"causal"` mode instead uses a lower-right alignment (`j <= i + (k_len - q_len)`),
/// which only matches ONNX when `past_seq == k_len - q_len`. For the non-square no-past case
/// (`k_len > q_len`, `past_seq == 0`) the two disagree, so we build the additive mask explicitly.
fn causal_mask_topleft(
    ctx: &mut TranslationContext,
    q_len: i32,
    k_len: i32,
    past_seq: i32,
    dt: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    let i32t = mlx::mlx_dtype__MLX_INT32;
    let key_pos = ctx.arange(0.0, k_len as f64, 1.0, i32t)?; // [k_len]
    let key_pos = ctx.reshape(key_pos, &[1, k_len])?; // [1,k_len]
    let q_pos = ctx.arange(past_seq as f64, (past_seq + q_len) as f64, 1.0, i32t)?; // [q_len]
    let q_pos = ctx.reshape(q_pos, &[q_len, 1])?; // [q_len,1]
    let allow = ctx.less_equal(key_pos, q_pos)?; // [q_len,k_len] bool
    let zero = ctx.scalar_f32(0.0);
    let zero = ctx.astype(zero, dt)?;
    let neg = ctx.scalar_f32(f32::NEG_INFINITY);
    let neg = ctx.astype(neg, dt)?;
    ctx.where_(allow, zero, neg)
}

fn causal_window_mask(
    ctx: &mut TranslationContext,
    q_len: i32,
    k_len: i32,
    past_seq: i32,
    window: i32,
    dt: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    let i32t = mlx::mlx_dtype__MLX_INT32;
    let key_pos = ctx.arange(0.0, k_len as f64, 1.0, i32t)?;
    let key_pos = ctx.reshape(key_pos, &[1, k_len])?;
    let q_pos = ctx.arange(past_seq as f64, (past_seq + q_len) as f64, 1.0, i32t)?;
    let q_pos = ctx.reshape(q_pos, &[q_len, 1])?;
    let upper = ctx.less_equal(key_pos, q_pos)?;
    let lower_pos = ctx.arange(
        (past_seq - window + 1) as f64,
        (past_seq + q_len - window + 1) as f64,
        1.0,
        i32t,
    )?;
    // Right-padded GQA query rows use the last live key as their window endpoint.
    let last_lower = ctx.scalar_i32(k_len - window);
    let lower_pos = ctx.binary(mlx::mlx_minimum, lower_pos, last_lower)?;
    let lower_pos = ctx.reshape(lower_pos, &[q_len, 1])?;
    let lower = ctx.less_equal(lower_pos, key_pos)?;
    let allow = ctx.binary(mlx::mlx_logical_and, upper, lower)?;
    let zero = ctx.scalar_f32(0.0);
    let zero = ctx.astype(zero, dt)?;
    let neg = ctx.scalar_f32(f32::NEG_INFINITY);
    let neg = ctx.astype(neg, dt)?;
    ctx.where_(allow, zero, neg)
}

/// The ONNX causal mask, with any supplied `attn_mask` folded into it.
///
/// ONNX `Attention` applies `is_causal` *and* `attn_mask` when both are given,
/// and both are expressible as one additive array, so they are combined rather
/// than sent through two dispatches:
///
/// * a float mask is additive already, so the two are added — `-inf` from either
///   side wins, and `0 + 0` leaves an attended position attended;
/// * a bool mask means "true = attend", so it becomes `0`/`-inf` first.
///
/// Neither side ever contributes `+inf`, so no addition can produce `NaN`.
fn combined_causal_mask(
    ctx: &mut TranslationContext,
    q_len: i32,
    k_len: i32,
    past_seq: i32,
    dt: mlx::mlx_dtype,
    mask: Option<mlx::mlx_array>,
) -> Result<mlx::mlx_array, MlxError> {
    let causal = causal_mask_topleft(ctx, q_len, k_len, past_seq, dt)?;
    let Some(mask) = mask else {
        return Ok(causal);
    };
    let additive = if ctx.dtype_of(mask) == mlx::mlx_dtype__MLX_BOOL {
        let zero = ctx.scalar_f32(0.0);
        let zero = ctx.astype(zero, dt)?;
        let neg = ctx.scalar_f32(f32::NEG_INFINITY);
        let neg = ctx.astype(neg, dt)?;
        ctx.where_(mask, zero, neg)?
    } else {
        ctx.astype(mask, dt)?
    };
    // In token-at-a-time cached self-attention the sole query follows every past key, so the
    // causal mask is all zeros. Returning the external mask directly also avoids MLX baking the
    // current KV extent into an internal Broadcast during shapeless compilation.
    if q_len == 1 && past_seq == k_len - 1 {
        return Ok(additive);
    }
    ctx.add(causal, additive)
}

/// GQA eager SDPA over the attended keys `[0, k_len)`, with the queries at positions
/// `[valid_past, valid_past+S)`. Without an `attention_bias` this is plain causal SDPA (bit-for-bit
/// the legacy behavior). With the 11-input Gemma3n `attention_bias` (input 10, `[B,1,S,total]`
/// additive mask encoding the causal + sliding-window mask), we fold it into an array mask:
/// `mask = causal_topleft[q_len,k_len] + attention_bias[..,:k_len]`. Adding the causal triangle is
/// idempotent where the bias already masks (−inf + finite = −inf) and supplies causality where it
/// doesn't; the sliding window is carried entirely by the bias.
#[allow(clippy::too_many_arguments)]
fn gqa_eager_sdpa(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    qh: mlx::mlx_array,
    ak: mlx::mlx_array,
    av: mlx::mlx_array,
    scale: f32,
    q_len: i32,
    valid_past: i32,
    k_len: i32,
) -> Result<mlx::mlx_array, MlxError> {
    if present(n, 10) {
        let dt = ctx.dtype_of(qh);
        let bias = ctx.resolve(&n.inputs[10])?; // [B,1,S,total]
        let bs = ctx.shape_of(bias);
        let bias = slice(ctx, bias, &[0, 0, 0, 0], &[bs[0], bs[1], q_len, k_len])?;
        let causal = causal_mask_topleft(ctx, q_len, k_len, valid_past, dt)?; // [q_len,k_len]
        let mask = add(ctx, causal, bias)?; // -> [B,1,S,k_len]
        return sdpa(ctx, qh, ak, av, scale, b"array\0", mask);
    }
    sdpa(ctx, qh, ak, av, scale, b"causal\0", empty_array())
}

/// [B,S,H*hd] -> [B,H,S,hd] (head-major split then transpose).
fn split_heads(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    b: i32,
    s: i32,
    h: i32,
    hd: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let r = ctx.reshape(x, &[b, s, h, hd])?;
    ctx.transpose(r, &[0, 2, 1, 3])
}

fn present(n: &NodeDesc, i: usize) -> bool {
    i < n.inputs.len() && n.inputs[i].source != Src::Absent
}

fn attr_int(n: &NodeDesc, name: &str, default: i64) -> i64 {
    n.ints.get(name).copied().unwrap_or(default)
}

fn attr_scale(n: &NodeDesc, hd: i32) -> f32 {
    match n.floats.get("scale") {
        Some(&s) if s != 0.0 => s,
        _ => 1.0 / (hd as f32).sqrt(),
    }
}

// ---- GroupQueryAttention composed RoPE ---------------------------------------------------------

/// Rotate-half (non-interleaved) or interleaved rotary over the first 2*half head dims of x [B,H,S,hd].
/// cos/sin are the exact per-position rows [S, half], broadcast over B and the head axis.
fn gqa_rope(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    cos: mlx::mlx_array,
    sin: mlx::mlx_array,
    half: i32,
    interleaved: bool,
) -> Result<mlx::mlx_array, MlxError> {
    let xs = ctx.shape_of(x); // [B,H,S,hd]
    let (b, h, s, hd) = (xs[0], xs[1], xs[2], xs[3]);
    let rot = 2 * half;
    let rotated = if !interleaved {
        let x1 = slice(ctx, x, &[0, 0, 0, 0], &[b, h, s, half])?;
        let x2 = slice(ctx, x, &[0, 0, 0, half], &[b, h, s, rot])?;
        let x1c = mul(ctx, x1, cos)?;
        let x2s = mul(ctx, x2, sin)?;
        let o1 = sub(ctx, x1c, x2s)?;
        let x2c = mul(ctx, x2, cos)?;
        let x1s = mul(ctx, x1, sin)?;
        let o2 = add(ctx, x2c, x1s)?;
        concat2(ctx, o1, o2, 3)?
    } else {
        let sl = slice(ctx, x, &[0, 0, 0, 0], &[b, h, s, rot])?;
        let xr = ctx.reshape(sl, &[b, h, s, half, 2])?;
        let xe = slice(ctx, xr, &[0, 0, 0, 0, 0], &[b, h, s, half, 1])?;
        let xo = slice(ctx, xr, &[0, 0, 0, 0, 1], &[b, h, s, half, 2])?;
        let c = ctx.reshape(cos, &[b, 1, s, half, 1])?;
        let sn = ctx.reshape(sin, &[b, 1, s, half, 1])?;
        let xe = ctx.reshape(xe, &[b, h, s, half, 1])?;
        let xo = ctx.reshape(xo, &[b, h, s, half, 1])?;
        let xec = mul(ctx, xe, c)?;
        let xosn = mul(ctx, xo, sn)?;
        let oe = sub(ctx, xec, xosn)?;
        let xoc = mul(ctx, xo, c)?;
        let xesn = mul(ctx, xe, sn)?;
        let oo = add(ctx, xoc, xesn)?;
        let cat = concat2(ctx, oe, oo, 4)?;
        ctx.reshape(cat, &[b, h, s, rot])?
    };
    if rot == hd {
        return Ok(rotated);
    }
    let tail = slice(ctx, x, &[0, 0, 0, rot], &[b, h, s, hd])?;
    concat2(ctx, rotated, tail, 3)
}

/// Slice the cos/sin cache rows for positions [past, past+S) -> [S, half] (eager static slice).
fn cos_sin_row(
    ctx: &mut TranslationContext,
    cache: mlx::mlx_array,
    past: i32,
    seq: i32,
    half: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let shape = ctx.shape_of(cache);
    if shape.len() != 2 || past < 0 || seq < 0 || past > shape[0] - seq || half > shape[1] {
        return Err("RotaryEmbedding: requested positions exceed the supplied cache".to_string());
    }
    slice(ctx, cache, &[past, 0], &[past + seq, half])
}

#[allow(clippy::too_many_arguments)]
fn gqa_batched(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    q: mlx::mlx_array,
    k: mlx::mlx_array,
    v: mlx::mlx_array,
    past_k: mlx::mlx_array,
    past_v: mlx::mlx_array,
    scale: f32,
) -> Result<(), MlxError> {
    let qs = ctx.shape_of(q);
    let ks = ctx.shape_of(k);
    let (batch, heads, seq, head) = (qs[0], qs[1], qs[2], qs[3]);
    let kv_heads = ks[1];
    let cap = ctx.shape_of(past_k)[2];
    let lengths = read_int_input(ctx, &n.inputs[5])?;
    let total = read_int_input(ctx, &n.inputs[6])?;
    if lengths.len() != batch as usize
        || total.len() != 1
        || total[0] <= 0
        || total[0] > i32::MAX as i64
        || lengths.iter().any(|&n| n < 0 || n >= total[0])
        || lengths.iter().max().copied() != Some(total[0] - 1)
    {
        return Err(
            "GroupQueryAttention: invalid per-batch or maximum sequence lengths".to_string(),
        );
    }
    let capacity = cap.max(total[0] as i32);
    let causal = attr_int(n, "causal", 1) != 0;
    let window = attr_int(n, "local_window_size", -1) as i32;
    let rotary = if attr_int(n, "do_rotary", 0) != 0 {
        Some((ctx.resolve(&n.inputs[7])?, ctx.resolve(&n.inputs[8])?))
    } else {
        None
    };
    let bias = if present(n, 10) {
        let bias = ctx.resolve(&n.inputs[10])?;
        let bs = ctx.shape_of(bias);
        if bs.len() != 4
            || (bs[0] != 1 && bs[0] != batch)
            || (bs[1] != 1 && bs[1] != heads)
            || bs[2] != seq
            || bs[3] != total[0] as i32
        {
            return Err(
                "GroupQueryAttention: attention_bias must broadcast to [batch,heads,sequence,total]"
                    .to_string(),
            );
        }
        Some(bias)
    } else {
        None
    };
    let mut outputs = VectorArray::new();
    let mut keys = VectorArray::new();
    let mut values = VectorArray::new();
    let mut diagnostics = VectorArray::new();
    for b in 0..batch {
        let length = lengths[b as usize] as i32 + 1;
        let past = (length - seq).max(0);
        let new_rows = length - past;
        if past > cap || (!present(n, 3) && past != 0) {
            return Err(
                "GroupQueryAttention: sequence length exceeds available past cache".to_string(),
            );
        }
        let mut qb = slice(ctx, q, &[b, 0, 0, 0], &[b + 1, heads, seq, head])?;
        let mut kb = slice(ctx, k, &[b, 0, 0, 0], &[b + 1, kv_heads, seq, head])?;
        let vb = slice(ctx, v, &[b, 0, 0, 0], &[b + 1, kv_heads, new_rows, head])?;
        if let Some((cos, sin)) = rotary {
            let half = ctx.shape_of(cos)[1];
            let cr = cos_sin_row(ctx, cos, past, seq, half)?;
            let sr = cos_sin_row(ctx, sin, past, seq, half)?;
            let interleaved = attr_int(n, "rotary_interleaved", 0) != 0;
            qb = gqa_rope(ctx, qb, cr, sr, half, interleaved)?;
            kb = gqa_rope(ctx, kb, cr, sr, half, interleaved)?;
        }
        let kb = slice(ctx, kb, &[0, 0, 0, 0], &[1, kv_heads, new_rows, head])?;
        let pk = slice(ctx, past_k, &[b, 0, 0, 0], &[b + 1, kv_heads, cap, head])?;
        let pv = slice(ctx, past_v, &[b, 0, 0, 0], &[b + 1, kv_heads, cap, head])?;
        let old_k = slice(ctx, pk, &[0, 0, 0, 0], &[1, kv_heads, past, head])?;
        let old_v = slice(ctx, pv, &[0, 0, 0, 0], &[1, kv_heads, past, head])?;
        let ak = concat2(ctx, old_k, kb, 2)?;
        let av = concat2(ctx, old_v, vb, 2)?;
        let dt = ctx.dtype_of(qb);
        let mut mask = if window > 0 {
            causal_window_mask(ctx, seq, length, past, window, dt)?
        } else if causal {
            causal_mask_topleft(ctx, seq, length, past, dt)?
        } else {
            ctx.zeros(&[seq, length], dt)?
        };
        let visibility = mask;
        let mut logits = if gqa_composed(n) {
            let q32 = ctx.astype(qb, mlx::mlx_dtype__MLX_FLOAT32)?;
            let k32 = ctx.astype(ak, mlx::mlx_dtype__MLX_FLOAT32)?;
            let k32 = gqa_repeat_heads(ctx, k32, kv_heads, heads / kv_heads, length, head)?;
            Some(capped_attention_logits(
                ctx,
                q32,
                k32,
                scale,
                n.floats.get("softcap").copied().unwrap_or(0.0),
            )?)
        } else {
            None
        };
        if let Some(bias) = bias {
            let bs = ctx.shape_of(bias);
            let bi = if bs[0] == 1 { 0 } else { b };
            let bias = slice(ctx, bias, &[bi, 0, 0, 0], &[bi + 1, bs[1], seq, length])?;
            if let Some(scores) = logits {
                let bias = ctx.astype(bias, mlx::mlx_dtype__MLX_FLOAT32)?;
                logits = Some(add(ctx, scores, bias)?);
            } else {
                mask = add(ctx, mask, bias)?;
            }
        }
        let out = if let Some(logits) = logits {
            let mask = ctx.astype(visibility, mlx::mlx_dtype__MLX_FLOAT32)?;
            let zero = ctx.scalar_f32(0.0);
            let visible = ctx.binary(mlx::mlx_equal, mask, zero)?;
            let neg = ctx.scalar_f32(f32::NEG_INFINITY);
            let scores = ctx.where_(visible, logits, neg)?;
            let sink = if present(n, 11) {
                Some(ctx.resolve(&n.inputs[11])?)
            } else if attr_int(n, "smooth_softmax", 0) == 1 {
                Some(ctx.zeros(&[heads], mlx::mlx_dtype__MLX_FLOAT32)?)
            } else {
                None
            };
            let probs = attention_probs_with_sink(ctx, scores, sink)?;
            let mode = attr_int(n, "qk_output", 0);
            if mode != 0 {
                let diagnostic = if mode == 1 {
                    ctx.where_(visible, logits, zero)?
                } else {
                    probs
                };
                let diagnostic = ctx.astype(diagnostic, dt)?;
                let padding = total[0] as i32 - length;
                let diagnostic = if padding > 0 {
                    let zero = ctx.zeros(&[1, heads, seq, padding], dt)?;
                    concat2(ctx, diagnostic, zero, 3)?
                } else {
                    diagnostic
                };
                diagnostics.append(diagnostic);
            }
            let v32 = ctx.astype(av, mlx::mlx_dtype__MLX_FLOAT32)?;
            let v32 = gqa_repeat_heads(ctx, v32, kv_heads, heads / kv_heads, length, head)?;
            let out = ctx.matmul(probs, v32)?;
            ctx.astype(out, dt)?
        } else {
            sdpa(ctx, qb, ak, av, scale, b"array\0", mask)?
        };
        outputs.append(out);
        let tail_len = capacity - length;
        let tail_k = if length < cap {
            slice(ctx, pk, &[0, 0, length, 0], &[1, kv_heads, cap, head])?
        } else {
            ctx.zeros(&[1, kv_heads, tail_len, head], dt)?
        };
        let tail_v = if length < cap {
            slice(ctx, pv, &[0, 0, length, 0], &[1, kv_heads, cap, head])?
        } else {
            ctx.zeros(&[1, kv_heads, tail_len, head], dt)?
        };
        let pk = concat2(ctx, ak, tail_k, 2)?;
        let pv = concat2(ctx, av, tail_v, 2)?;
        // A growing ragged cache can require padding beyond its original capacity.
        let pad = capacity - ctx.shape_of(pk)[2];
        if pad > 0 {
            let zeros = ctx.zeros(&[1, kv_heads, pad, head], dt)?;
            keys.append(concat2(ctx, pk, zeros, 2)?);
            values.append(concat2(ctx, pv, zeros, 2)?);
        } else {
            keys.append(pk);
            values.append(pv);
        }
    }
    let out =
        ctx.emit(|res, st| unsafe { mlx::mlx_concatenate_axis(res, outputs.as_raw(), 0, st) })?;
    let out = ctx.transpose(out, &[0, 2, 1, 3])?;
    let out = ctx.reshape(out, &[batch, seq, heads * head])?;
    ctx.bind(&n.outputs[0], out);
    if attr_int(n, "qk_output", 0) != 0 {
        let out = ctx.emit(|res, st| unsafe {
            mlx::mlx_concatenate_axis(res, diagnostics.as_raw(), 0, st)
        })?;
        ctx.bind(&n.outputs[3], out);
    }
    for (index, arrays) in [(1, keys), (2, values)] {
        if n.outputs.get(index).is_some_and(|o| !o.name.is_empty()) {
            let out = ctx.emit(|res, st| unsafe {
                mlx::mlx_concatenate_axis(res, arrays.as_raw(), 0, st)
            })?;
            ctx.bind(&n.outputs[index], out);
        }
    }
    Ok(())
}

/// Compiled-decode RoPE: rotate-half via a `[hd,hd]` matmul (no Slice) with FULL-width cos/sin rows
/// (`[1,1,S,hd]`, each half duplicated) fed as closure inputs. `out = x*cos + (x @ M)*sin`, matching
/// standard non-interleaved RoPE. Eligibility guarantees `rot == hd == 2*half`, so there is no
/// pass-through tail. Everything is cast to `x`'s dtype so the result dtype matches the eager path.
fn gqa_rope_matmul(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    cos_full: mlx::mlx_array,
    sin_full: mlx::mlx_array,
    hd: i32,
    half: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let dt = ctx.dtype_of(x);
    let m = ctx.rotate_half_matrix(hd, half);
    let m = ctx.astype(m, dt)?;
    let cos_c = ctx.astype(cos_full, dt)?;
    let sin_c = ctx.astype(sin_full, dt)?;
    let xrot = ctx.matmul(x, m)?; // rotate_half(x)
    let xc = mul(ctx, x, cos_c)?;
    let xs = mul(ctx, xrot, sin_c)?;
    add(ctx, xc, xs)
}

/// Compiled-decode shared-buffer KV update + masked SDPA. `valid_past` is DATA (recovered from the
/// in-graph `total_sequence_length` = input[6], minus S), so the new K/V are written in place with a
/// dynamic slice-update at that offset and attention runs over the whole `cap` buffer under a
/// static-shape additive mask (buffer tail beyond `valid_past+S` masked to -inf; causal within the
/// valid prefix). Every op is statically shaped, so the shapeless compiled closure can carry it.
/// Returns `(present_k, present_v, attn)` with present at the full `[B,kv,cap,hd]` capacity.
#[allow(clippy::too_many_arguments)]
fn gqa_shared_compiled(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    past_k: mlx::mlx_array,
    past_v: mlx::mlx_array,
    kh: mlx::mlx_array,
    vh: mlx::mlx_array,
    qh: mlx::mlx_array,
    s: i32,
    cap: i32,
    scale: f32,
) -> Result<(mlx::mlx_array, mlx::mlx_array, mlx::mlx_array), MlxError> {
    let i32t = mlx::mlx_dtype__MLX_INT32;
    // valid_past is fed as a live [1] int32 closure input (appended after the synth RoPE rows) so it
    // advances every decode step. Resolving the in-graph total_sequence_length here instead would let
    // shapeless compile freeze it at trace time (baking Shape(attention_mask) as a constant), which
    // pins the write offset + mask and corrupts generation. Fall back defensively if unavailable.
    let vp = match ctx.shared_valid_past() {
        Some(a) => a,
        None => {
            let ts = ctx.resolve(&n.inputs[6])?;
            let ts = ctx.astype(ts, i32t)?;
            let ts = ctx.reshape(ts, &[1])?;
            let s_scalar = ctx.scalar_i32(s);
            let s_scalar = ctx.reshape(s_scalar, &[1])?;
            ctx.sub(ts, s_scalar)?
        }
    }; // [1] valid_past

    // In-place write of the S new rows at axis-2 offset valid_past; present keeps the full capacity.
    let present_k = ctx.slice_update_dynamic(past_k, kh, vp, &[2])?;
    let present_v = ctx.slice_update_dynamic(past_v, vh, vp, &[2])?;

    // Additive causal mask [1,1,S,cap]: key j attends query i iff j <= valid_past + i.
    let key_pos = ctx.arange(0.0, cap as f64, 1.0, i32t)?; // [cap]
    let key_pos = ctx.reshape(key_pos, &[1, cap])?; // [1,cap]
    let q_off = ctx.arange(0.0, s as f64, 1.0, i32t)?; // [S]
    let q_pos = ctx.add(vp, q_off)?; // [S] (valid_past + i, broadcast [1]+[S])
    let q_pos = ctx.reshape(q_pos, &[s, 1])?; // [S,1]
    let allow = ctx.less_equal(key_pos, q_pos)?; // [S,cap] bool
    let dt = ctx.dtype_of(qh);
    let zero = ctx.scalar_f32(0.0);
    let zero = ctx.astype(zero, dt)?;
    let neg = ctx.scalar_f32(f32::NEG_INFINITY);
    let neg = ctx.astype(neg, dt)?;
    let mask = ctx.where_(allow, zero, neg)?; // [S,cap]
    let mask = ctx.reshape(mask, &[1, 1, s, cap])?;

    let attn = sdpa(ctx, qh, present_k, present_v, scale, b"array\0", mask)?;
    Ok((present_k, present_v, attn))
}

fn group_query_attention_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let num_heads = attr_int(n, "num_heads", 0) as i32;
    let kv_heads = attr_int(n, "kv_num_heads", 0) as i32;
    let interleaved = attr_int(n, "rotary_interleaved", 0) != 0;
    let do_rotary = attr_int(n, "do_rotary", 0) != 0;
    let sliding = attr_int(n, "sliding_window_cache", 0) != 0;
    let local_window = attr_int(n, "local_window_size", -1) as i32;

    let packed_qkv = n.inputs[1].source == Src::Absent && n.inputs[2].source == Src::Absent;
    let (q, k, v) = if packed_qkv {
        let packed = ctx.resolve(&n.inputs[0])?; // [B,S,(num+2*kv)*hd]
        let shape = ctx.shape_of(packed);
        let (b, s, hidden) = (shape[0], shape[1], shape[2]);
        let total_heads = num_heads + 2 * kv_heads;
        if total_heads <= 0 || hidden % total_heads != 0 {
            return Err(
                "GroupQueryAttention: packed QKV hidden size is not divisible by total heads"
                    .to_string(),
            );
        }
        let head = hidden / total_heads;
        let grouped = ctx.reshape(packed, &[b, s, total_heads, head])?;
        let q_indices = ctx.arange(0.0, num_heads as f64, 1.0, mlx::mlx_dtype__MLX_INT32)?;
        let k_indices = ctx.arange(
            num_heads as f64,
            (num_heads + kv_heads) as f64,
            1.0,
            mlx::mlx_dtype__MLX_INT32,
        )?;
        let v_indices = ctx.arange(
            (num_heads + kv_heads) as f64,
            total_heads as f64,
            1.0,
            mlx::mlx_dtype__MLX_INT32,
        )?;
        let q = ctx.emit(|res, stream| unsafe {
            mlx::mlx_take_axis(res, grouped, q_indices, 2, stream)
        })?;
        let k = ctx.emit(|res, stream| unsafe {
            mlx::mlx_take_axis(res, grouped, k_indices, 2, stream)
        })?;
        let v = ctx.emit(|res, stream| unsafe {
            mlx::mlx_take_axis(res, grouped, v_indices, 2, stream)
        })?;
        (
            ctx.reshape(q, &[b, s, num_heads * head])?,
            ctx.reshape(k, &[b, s, kv_heads * head])?,
            ctx.reshape(v, &[b, s, kv_heads * head])?,
        )
    } else {
        (
            ctx.resolve(&n.inputs[0])?, // [B,S,num*hd]
            ctx.resolve(&n.inputs[1])?, // [B,S,kv*hd]
            ctx.resolve(&n.inputs[2])?, // [B,S,kv*hd]
        )
    };
    let qs = ctx.shape_of(q);
    let (b, s) = (qs[0], qs[1]);
    let head = qs[2] / num_heads;
    let (past_k, past_v) = if present(n, 3) {
        (ctx.resolve(&n.inputs[3])?, ctx.resolve(&n.inputs[4])?)
    } else {
        let dt = ctx.dtype_of(q);
        let empty = ctx.zeros(&[b, kv_heads, 0, head], dt)?;
        (empty, empty)
    };
    // Sequence capacity of the past-KV cache (its axis-2 length). In the growing (ZeroCopyRebind)
    // contract this equals the number of valid past keys; in the fixed-capacity shared-buffer
    // contract it is the runtime-owned max length and exceeds the valid past.
    let cap = ctx.shape_of(past_k)[2];

    // Recover how many past keys are actually valid and whether the runtime handed us a max-length
    // shared buffer to write in place. `total_sequence_length` (input[6], int32 scalar) counts the
    // valid keys after this step (= valid_past + S), so `valid_past = total_sequence_length[0] - S`.
    // Shared-buffer mode is exactly `cap > valid_past` (past_k is a max-length buffer). The compiled
    // decode trace cannot eval mid-graph, so it takes the mode from the once-detected plan flag
    // (`ctx.shared_kv()`) and drives the in-place write with a DATA offset instead (see below).
    let (shared_buffer, valid_past) = if sliding {
        (true, 0)
    } else if ctx.rope_dynamic() {
        // valid_past (int) is unused on the shapeless compiled DECODE path: RoPE uses the pre-sliced
        // synth rows and the KV write/mask use a data offset derived from total_sequence_length. On
        // the shape-keyed compiled PREFILL path valid_past IS static (known per shape key), so use it
        // to narrow the attention to the valid prefix below.
        if ctx.compiled_shape_keyed() {
            (ctx.shared_kv(), ctx.compiled_valid_past())
        } else {
            (ctx.shared_kv(), cap)
        }
    } else {
        let ts = ctx.resolve(&n.inputs[6])?;
        let total = ctx.read_scalar_i64(ts)?;
        if total < 0 || total > i32::MAX as i64 {
            return Err(
                "GroupQueryAttention: total_sequence_length is outside int32 range".to_string(),
            );
        }
        let total = total as i32;
        let vp = total - s;
        if b == 1 && present(n, 3) {
            let lengths = read_int_input(ctx, &n.inputs[5])?;
            if lengths != [total as i64 - 1] || vp < 0 || vp > cap {
                return Err(
                    "GroupQueryAttention: inconsistent sequence lengths or cache capacity"
                        .to_string(),
                );
            }
        }
        (cap > vp, vp)
    };
    // RoPE positions for the new Q/K rows run [valid_past, valid_past+S). In growing mode this is the
    // past length exactly as before (bit-for-bit); in shared-buffer mode it must be valid_past, NOT
    // the buffer capacity.
    let past = valid_past;

    let scale = attr_scale(n, head);

    let mut qh = split_heads(ctx, q, b, s, num_heads, head)?;
    let mut kh = split_heads(ctx, k, b, s, kv_heads, head)?;
    let vh = split_heads(ctx, v, b, s, kv_heads, head)?;

    if present(n, 14) {
        let q_weight = ctx.resolve(&n.inputs[14])?;
        let k_weight = ctx.resolve(&n.inputs[15])?;
        let eps = n.floats.get("qk_norm_epsilon").copied().unwrap_or(1e-6);
        qh = ctx.emit(|res, st| unsafe { mlx::mlx_fast_rms_norm(res, qh, q_weight, eps, st) })?;
        kh = ctx.emit(|res, st| unsafe { mlx::mlx_fast_rms_norm(res, kh, k_weight, eps, st) })?;
    }

    if !sliding
        && (b != 1
            || !present(n, 3)
            || attr_int(n, "causal", 1) == 0
            || local_window >= 0
            || gqa_composed(n))
    {
        return gqa_batched(ctx, n, qh, kh, vh, past_k, past_v, scale);
    }

    if do_rotary && !sliding {
        let cos = ctx.resolve(&n.inputs[7])?; // [max_seq, rot/2]
        let sin = ctx.resolve(&n.inputs[8])?;
        let half = ctx.shape_of(cos)[1]; // rot/2
        if ctx.rope_dynamic() {
            // Compiled-decode trace: the position offset is DATA (the cos/sin rows arrive as
            // pre-sliced closure inputs), and rotate-half is a [hd,hd] matmul so the shapeless graph
            // carries no Slice. Eligibility (checked before compiling) guarantees rot == head_dim.
            // The matmul rotate-half matrix is non-interleaved only; an interleaved model errors here
            // so the trace fails and the plan falls back to the eager path (never miscomputed).
            if interleaved {
                return Err(
                    "MLX: compiled-decode RoPE does not support rotary_interleaved (falls back to eager)"
                        .to_string(),
                );
            }
            let cos_full = ctx.rope_row_full(&n.inputs[7].name, s, half)?; // [1,1,S,2*half]
            let sin_full = ctx.rope_row_full(&n.inputs[8].name, s, half)?;
            qh = gqa_rope_matmul(ctx, qh, cos_full, sin_full, head, half)?;
            kh = gqa_rope_matmul(ctx, kh, cos_full, sin_full, head, half)?;
        } else {
            let cr = cos_sin_row(ctx, cos, past, s, half)?;
            let sr = cos_sin_row(ctx, sin, past, s, half)?;
            qh = gqa_rope(ctx, qh, cr, sr, half, interleaved)?;
            kh = gqa_rope(ctx, kh, cr, sr, half, interleaved)?;
            ctx.mark_composed(
                "GroupQueryAttention RoPE consumes the supplied cache values exactly",
            );
        }
    }

    // Append the new K/V to the cache. Three contracts:
    //   * Shared-buffer, EAGER: write the S new rows in place at [valid_past, valid_past+S) of the
    //     [B,kv,cap,hd] buffer via slice_update, emit present at the FULL cap shape (matches ORT's
    //     pre-bound shared output), and attend over the valid prefix [0, valid_past+S) so causal
    //     alignment places the S queries at their true positions [valid_past, valid_past+S).
    //   * Shared-buffer, COMPILED PREFILL (shape-keyed): valid_past + S are static for the shape key,
    //     so narrow exactly like the eager path — write the S new rows in place at
    //     [valid_past, valid_past+S), then attend over only the valid prefix [0, valid_past+S) under a
    //     causal SDPA (STATIC slices, no full-cap mask). This drops the compute-bound full-capacity
    //     attention that made prefill-compile a TTFT regression while staying byte-identical to eager.
    //   * Shared-buffer, COMPILED DECODE (shapeless): valid_past is DATA (total_sequence_length - S),
    //     so the write is a slice_update_dynamic at that offset and attention uses a static-shape
    //     additive mask over the whole cap buffer (the tail beyond valid_past+S masked to -inf).
    //     Keeping every op statically shaped lets the shapeless compiled closure express it.
    //   * Growing: concat past+new along the sequence axis (unchanged legacy behavior).
    let (present_k, present_v, attn) = if sliding {
        let seqlens = read_int_input(ctx, &n.inputs[5])?;
        if seqlens.len() != b as usize {
            return Err("GroupQueryAttention: seqlens_k must have one value per batch".to_string());
        }
        if local_window <= 0 || cap < local_window {
            return Err(
                "GroupQueryAttention: sliding cache requires 0 < local_window_size <= capacity"
                    .to_string(),
            );
        }
        let gap = cap - local_window + 1;
        let cache_end = |length: i64| -> i32 {
            if length <= cap as i64 {
                length as i32
            } else {
                let overflow = length - cap as i64;
                let reclaimed = gap as i64 * ((overflow + gap as i64 - 1) / gap as i64);
                (length - reclaimed) as i32
            }
        };
        let mut geometry = Vec::with_capacity(b as usize);
        let mut use_staging = false;
        for &seqlen in &seqlens {
            let p = (seqlen + 1 - s as i64).max(0);
            let before = cache_end(p);
            let after = cache_end(p + s as i64);
            let kept = after - s;
            let required = p.min((local_window - 1) as i64) as i32;
            use_staging |= kept < required;
            geometry.push((p as i32, before, after, before - kept));
        }

        let rope_data = if do_rotary {
            let cos = ctx.resolve(&n.inputs[7])?;
            let sin = ctx.resolve(&n.inputs[8])?;
            let half = ctx.shape_of(cos)[1];
            Some((cos, sin, half))
        } else {
            None
        };

        let mut present_keys = VectorArray::new();
        let mut present_values = VectorArray::new();
        let mut attn_batches = VectorArray::new();
        for batch_index in 0..b {
            let (absolute_past, end_before, end_after, seed_offset) =
                geometry[batch_index as usize];
            let mut qb = slice(
                ctx,
                qh,
                &[batch_index, 0, 0, 0],
                &[batch_index + 1, num_heads, s, head],
            )?;
            let mut kb = slice(
                ctx,
                kh,
                &[batch_index, 0, 0, 0],
                &[batch_index + 1, kv_heads, s, head],
            )?;
            let vb = slice(
                ctx,
                vh,
                &[batch_index, 0, 0, 0],
                &[batch_index + 1, kv_heads, s, head],
            )?;
            if let Some((cos, sin, half)) = rope_data {
                let cr = cos_sin_row(ctx, cos, absolute_past, s, half)?;
                let sr = cos_sin_row(ctx, sin, absolute_past, s, half)?;
                qb = gqa_rope(ctx, qb, cr, sr, half, interleaved)?;
                kb = gqa_rope(ctx, kb, cr, sr, half, interleaved)?;
            }

            let past_kb = slice(
                ctx,
                past_k,
                &[batch_index, 0, 0, 0],
                &[batch_index + 1, kv_heads, cap, head],
            )?;
            let past_vb = slice(
                ctx,
                past_v,
                &[batch_index, 0, 0, 0],
                &[batch_index + 1, kv_heads, cap, head],
            )?;
            let seed_start = if use_staging { 0 } else { seed_offset };
            let seed_k = slice(
                ctx,
                past_kb,
                &[0, 0, seed_start, 0],
                &[1, kv_heads, end_before, head],
            )?;
            let seed_v = slice(
                ctx,
                past_vb,
                &[0, 0, seed_start, 0],
                &[1, kv_heads, end_before, head],
            )?;
            let work_k = concat2(ctx, seed_k, kb, 2)?;
            let work_v = concat2(ctx, seed_v, vb, 2)?;
            let seed_rows = end_before - seed_start;
            let k_len = seed_rows + s;
            let mask =
                causal_window_mask(ctx, s, k_len, seed_rows, local_window, ctx.dtype_of(qb))?;
            let mask = ctx.reshape(mask, &[1, 1, s, k_len])?;
            let ab = if gqa_composed(n) {
                let sink = if present(n, 11) {
                    Some(ctx.resolve(&n.inputs[11])?)
                } else if attr_int(n, "smooth_softmax", 0) == 1 {
                    Some(ctx.zeros(&[num_heads], mlx::mlx_dtype__MLX_FLOAT32)?)
                } else {
                    None
                };
                let dt = ctx.dtype_of(qb);
                let qb = ctx.astype(qb, mlx::mlx_dtype__MLX_FLOAT32)?;
                let k32 = ctx.astype(work_k, mlx::mlx_dtype__MLX_FLOAT32)?;
                let v32 = ctx.astype(work_v, mlx::mlx_dtype__MLX_FLOAT32)?;
                let k32 = gqa_repeat_heads(ctx, k32, kv_heads, num_heads / kv_heads, k_len, head)?;
                let v32 = gqa_repeat_heads(ctx, v32, kv_heads, num_heads / kv_heads, k_len, head)?;
                let mask = ctx.astype(mask, mlx::mlx_dtype__MLX_FLOAT32)?;
                let out = paged_sdpa(
                    ctx,
                    qb,
                    k32,
                    v32,
                    scale,
                    mask,
                    sink,
                    n.floats.get("softcap").copied().unwrap_or(0.0),
                )?;
                ctx.astype(out, dt)?
            } else {
                sdpa(ctx, qb, work_k, work_v, scale, b"array\0", mask)?
            };

            let write_start = if use_staging { seed_offset } else { 0 };
            let write_k = slice(
                ctx,
                work_k,
                &[0, 0, write_start, 0],
                &[1, kv_heads, k_len, head],
            )?;
            let write_v = slice(
                ctx,
                work_v,
                &[0, 0, write_start, 0],
                &[1, kv_heads, k_len, head],
            )?;
            let (pk, pv) = if end_after < cap {
                let kdt = ctx.dtype_of(past_kb);
                let vdt = ctx.dtype_of(past_vb);
                let tail_k = ctx.zeros(&[1, kv_heads, cap - end_after, head], kdt)?;
                let tail_v = ctx.zeros(&[1, kv_heads, cap - end_after, head], vdt)?;
                (
                    concat2(ctx, write_k, tail_k, 2)?,
                    concat2(ctx, write_v, tail_v, 2)?,
                )
            } else {
                (write_k, write_v)
            };
            present_keys.append(pk);
            present_values.append(pv);
            attn_batches.append(ab);
        }
        let pk = ctx.emit(|res, st| unsafe {
            mlx::mlx_concatenate_axis(res, present_keys.as_raw(), 0, st)
        })?;
        let pv = ctx.emit(|res, st| unsafe {
            mlx::mlx_concatenate_axis(res, present_values.as_raw(), 0, st)
        })?;
        let a = ctx.emit(|res, st| unsafe {
            mlx::mlx_concatenate_axis(res, attn_batches.as_raw(), 0, st)
        })?;
        (pk, pv, a)
    } else if shared_buffer && ctx.rope_dynamic() && ctx.compiled_shape_keyed() {
        let start = [0, 0, valid_past, 0];
        let stop = [b, kv_heads, valid_past + s, head];
        let pk = ctx.slice_update(past_k, kh, &start, &stop)?;
        let pv = ctx.slice_update(past_v, vh, &start, &stop)?;
        let vp1 = valid_past + s;
        let ak = slice(ctx, pk, &[0, 0, 0, 0], &[b, kv_heads, vp1, head])?;
        let av = slice(ctx, pv, &[0, 0, 0, 0], &[b, kv_heads, vp1, head])?;
        let attn = sdpa(ctx, qh, ak, av, scale, b"causal\0", empty_array())?;
        (pk, pv, attn)
    } else if shared_buffer && ctx.rope_dynamic() {
        gqa_shared_compiled(ctx, n, past_k, past_v, kh, vh, qh, s, cap, scale)?
    } else if shared_buffer {
        let start = [0, 0, valid_past, 0];
        let stop = [b, kv_heads, valid_past + s, head];
        let pk = ctx.slice_update(past_k, kh, &start, &stop)?;
        let pv = ctx.slice_update(past_v, vh, &start, &stop)?;
        let vp1 = valid_past + s;
        let ak = slice(ctx, pk, &[0, 0, 0, 0], &[b, kv_heads, vp1, head])?;
        let av = slice(ctx, pv, &[0, 0, 0, 0], &[b, kv_heads, vp1, head])?;
        let attn = gqa_eager_sdpa(ctx, n, qh, ak, av, scale, s, valid_past, vp1)?;
        (pk, pv, attn)
    } else {
        let pk = concat2(ctx, past_k, kh, 2)?;
        let pv = concat2(ctx, past_v, vh, 2)?;
        let attn = gqa_eager_sdpa(ctx, n, qh, pk, pv, scale, s, valid_past, valid_past + s)?;
        (pk, pv, attn)
    };

    // [B,H,S,hd] -> [B,S,H*hd].
    let t = ctx.transpose(attn, &[0, 2, 1, 3])?;
    let out = ctx.reshape(t, &[b, s, num_heads * head])?;
    // Prefill feeds this tensor into multiple large projections. Materialize the head-major
    // transpose once so each downstream quantized GEMM does not independently traverse the
    // strided SDPA view.
    let out = if ctx.shape_keyed_compile() {
        ctx.contiguous(out)?
    } else {
        out
    };

    ctx.bind(&n.outputs[0], out);
    if n.outputs.len() >= 2 {
        ctx.bind(&n.outputs[1], present_k);
    }
    if n.outputs.len() >= 3 {
        ctx.bind(&n.outputs[2], present_v);
    }
    // Shared-buffer `present` aliases `past` in ORT memory, so copy-out only needs the S new rows
    // written this step at axis-2 offset `valid_past`. Register the delta so the boundary copy-out
    // is O(new-tokens) instead of O(capacity). (No-op for the growing path, which never sets
    // `shared_buffer`, keeping its full copy-out bit-for-bit unchanged.)
    if shared_buffer && !sliding {
        if n.outputs.len() >= 2 {
            ctx.record_kv_present(
                &n.outputs[1].name,
                valid_past as i64,
                s as i64,
                n.inputs[3].ctx_index,
            );
        }
        if n.outputs.len() >= 3 {
            ctx.record_kv_present(
                &n.outputs[2].name,
                valid_past as i64,
                s as i64,
                n.inputs[4].ctx_index,
            );
        }
    }
    Ok(())
}

/// Convert an ONNX boolean attention mask (`true = attend`) to the additive representation used
/// by the composed Attention path.
fn mask_as_additive(
    ctx: &mut TranslationContext,
    mask: mlx::mlx_array,
    dt: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    if ctx.dtype_of(mask) != mlx::mlx_dtype__MLX_BOOL {
        return ctx.astype(mask, dt);
    }
    let zero = ctx.scalar_f32(0.0);
    let zero = ctx.astype(zero, dt)?;
    let neg = ctx.scalar_f32(f32::NEG_INFINITY);
    let neg = ctx.astype(neg, dt)?;
    ctx.where_(mask, zero, neg)
}

/// Materialize a schema-broadcastable attention mask to score shape. ONNX permits a short final
/// key axis and defines its missing suffix as `-inf`; MLX broadcast alone cannot express that pad.
fn attention_mask_to_score_shape(
    ctx: &mut TranslationContext,
    mask: mlx::mlx_array,
    b: i32,
    h: i32,
    q_len: i32,
    k_len: i32,
    dt: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    let mask = mask_as_additive(ctx, mask, dt)?;
    let mask_shape = ctx.shape_of(mask);
    let width = match mask_shape.last().copied() {
        None | Some(1) => k_len,
        Some(width) => width,
    };
    if width > k_len {
        return Err(format!(
            "Attention: attn_mask key dimension {width} exceeds total key sequence length {k_len}"
        ));
    }
    let shape = [b, h, q_len, width];
    let mask = ctx.emit(|res, st| unsafe {
        mlx::mlx_broadcast_to(res, mask, shape.as_ptr(), shape.len(), st)
    })?;
    if width == k_len {
        return Ok(mask);
    }
    let tail = ctx.zeros(&[b, h, q_len, k_len - width], dt)?;
    let neg = ctx.scalar_f32(f32::NEG_INFINITY);
    let neg = ctx.astype(neg, dt)?;
    let tail = add(ctx, tail, neg)?;
    concat2(ctx, mask, tail, 3)
}

/// Repeat grouped K/V heads for the composed Attention path. The fast SDPA path keeps native GQA;
/// only softcap/QK-output forms need this materialization.
fn gqa_repeat_heads_batched(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    b: i32,
    kv: i32,
    groups: i32,
    s: i32,
    hd: i32,
) -> Result<mlx::mlx_array, MlxError> {
    if groups == 1 {
        return Ok(x);
    }
    let r = ctx.reshape(x, &[b, kv, 1, s, hd])?;
    let shape = [b, kv, groups, s, hd];
    let expanded = ctx.emit(|res, st| unsafe {
        mlx::mlx_broadcast_to(res, r, shape.as_ptr(), shape.len(), st)
    })?;
    ctx.reshape(expanded, &[b, kv * groups, s, hd])
}

/// Build the v25 Attention additive overlay for causal/window/nonpad contracts. `None` means
/// neither the op nor its caller needs a mask, preserving the fused SDPA route.
#[allow(clippy::too_many_arguments)]
fn attention_overlay_mask(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    b: i32,
    q_len: i32,
    k_len: i32,
    past_len: i32,
    dt: mlx::mlx_dtype,
    mask: Option<mlx::mlx_array>,
) -> Result<Option<mlx::mlx_array>, MlxError> {
    let causal = attr_int(n, "is_causal", 0) != 0;
    let left = attr_int(n, "left_window_size", -1) as i32;
    let right = attr_int(n, "right_window_size", -1) as i32;
    let nonpad = present(n, 6);
    if !causal && left < 0 && right < 0 && !nonpad {
        return mask.map(|m| mask_as_additive(ctx, m, dt)).transpose();
    }

    let i32t = mlx::mlx_dtype__MLX_INT32;
    let key = ctx.arange(0.0, k_len as f64, 1.0, i32t)?;
    let key = ctx.reshape(key, &[1, 1, 1, k_len])?;
    let query = ctx.arange(0.0, q_len as f64, 1.0, i32t)?;
    let query = ctx.reshape(query, &[1, 1, q_len, 1])?;
    let query = if nonpad {
        let lengths = ctx.resolve(&n.inputs[6])?;
        let lengths = ctx.astype(lengths, i32t)?;
        let q_len = ctx.scalar_f32(q_len as f32);
        let q_len = ctx.astype(q_len, i32t)?;
        let offset = sub(ctx, lengths, q_len)?;
        let offset = ctx.reshape(offset, &[b, 1, 1, 1])?;
        add(ctx, query, offset)?
    } else if past_len != 0 {
        let past_len = ctx.scalar_f32(past_len as f32);
        let past_len = ctx.astype(past_len, i32t)?;
        add(ctx, query, past_len)?
    } else {
        query
    };

    let mut allowed = if causal {
        ctx.less_equal(key, query)?
    } else {
        ctx.binary(mlx::mlx_equal, key, key)?
    };
    if left >= 0 {
        let left = ctx.scalar_f32(left as f32);
        let left = ctx.astype(left, i32t)?;
        let left_limit = sub(ctx, query, left)?;
        let within_left = ctx.less_equal(left_limit, key)?;
        allowed = ctx.binary(mlx::mlx_logical_and, allowed, within_left)?;
    }
    if right >= 0 {
        let right = ctx.scalar_f32(right as f32);
        let right = ctx.astype(right, i32t)?;
        let right_limit = add(ctx, query, right)?;
        let within_right = ctx.less_equal(key, right_limit)?;
        allowed = ctx.binary(mlx::mlx_logical_and, allowed, within_right)?;
    }
    if nonpad {
        let lengths = ctx.resolve(&n.inputs[6])?;
        let lengths = ctx.astype(lengths, i32t)?;
        let lengths = ctx.reshape(lengths, &[b, 1, 1, 1])?;
        let valid_key = ctx.binary(mlx::mlx_less, key, lengths)?;
        allowed = ctx.binary(mlx::mlx_logical_and, allowed, valid_key)?;
    }
    let zero = ctx.scalar_f32(0.0);
    let zero = ctx.astype(zero, dt)?;
    let neg = ctx.scalar_f32(f32::NEG_INFINITY);
    let neg = ctx.astype(neg, dt)?;
    let mut overlay = ctx.where_(allowed, zero, neg)?;
    if let Some(mask) = mask {
        let mask = mask_as_additive(ctx, mask, dt)?;
        overlay = add(ctx, overlay, mask)?;
    }
    Ok(Some(overlay))
}

fn attention_softmax(
    ctx: &mut TranslationContext,
    scores: mlx::mlx_array,
    precision: mlx::mlx_dtype,
    output_dt: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    let scores = ctx.astype(scores, precision)?;
    let probs = ctx.softmax_axis(scores, 3)?;
    ctx.astype(probs, output_dt)
}

fn attention_zero_fully_masked_rows(
    ctx: &mut TranslationContext,
    values: mlx::mlx_array,
    additive_mask: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    let row_max = if ctx.shape_of(additive_mask).is_empty() {
        additive_mask
    } else {
        ctx.emit(|res, st| unsafe { mlx::mlx_max_axis(res, additive_mask, -1, true, st) })?
    };
    let neg = ctx.scalar_f32(f32::NEG_INFINITY);
    let neg = ctx.astype(neg, ctx.dtype_of(additive_mask))?;
    let all_masked = ctx.binary(mlx::mlx_equal, row_max, neg)?;
    let zero = ctx.scalar_f32(0.0);
    let zero = ctx.astype(zero, ctx.dtype_of(values))?;
    // Decide on the bias, not on possibly NaN logits/output. Select, rather than multiply,
    // so empty rows become zero without clearing NaNs from rows with attendable keys.
    ctx.where_(all_masked, zero, values)
}

fn attention_qk_output_op(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    q: mlx::mlx_array,
    k: mlx::mlx_array,
    v: mlx::mlx_array,
    b: i32,
    qh: i32,
    kvh: i32,
    q_len: i32,
    k_len: i32,
    q_hd: i32,
    v_hd: i32,
    past_len: i32,
    dt: mlx::mlx_dtype,
    mask: Option<mlx::mlx_array>,
) -> Result<mlx::mlx_array, MlxError> {
    let groups = qh / kvh;
    let k = gqa_repeat_heads_batched(ctx, k, b, kvh, groups, k_len, q_hd)?;
    let kt = ctx.transpose(k, &[0, 1, 3, 2])?;
    let scale = ctx.scalar_f32(attr_scale(n, q_hd).sqrt());
    let scale = ctx.astype(scale, dt)?;
    let q = mul(ctx, q, scale)?;
    let kt = mul(ctx, kt, scale)?;
    let raw = ctx.matmul(q, kt)?;
    let softcap = n.floats.get("softcap").copied().unwrap_or(0.0);
    let capped = if softcap == 0.0 {
        raw
    } else {
        let cap = ctx.scalar_f32(softcap);
        let cap = ctx.astype(cap, dt)?;
        let scaled = ctx.binary(mlx::mlx_divide, raw, cap)?;
        let tanh = ctx.unary(mlx::mlx_tanh, scaled)?;
        mul(ctx, tanh, cap)?
    };
    let mask = mask
        .map(|mask| attention_mask_to_score_shape(ctx, mask, b, qh, q_len, k_len, dt))
        .transpose()?;
    let overlay = attention_overlay_mask(ctx, n, b, q_len, k_len, past_len, dt, mask)?;
    let biased = match overlay {
        Some(overlay) => add(ctx, capped, overlay)?,
        None => capped,
    };
    let precision = mlx_softmax_precision(n.ints.get("softmax_precision").copied()).unwrap_or(dt);
    let mut probs = attention_softmax(ctx, biased, precision, dt)?;
    if let Some(overlay) = overlay {
        probs = attention_zero_fully_masked_rows(ctx, probs, overlay)?;
    }

    if n.outputs.len() > 3 && !n.outputs[3].name.is_empty() {
        let mode = attr_int(n, "qk_matmul_output_mode", 0);
        let output = match mode {
            1 => capped,
            2 => biased,
            3 => probs,
            _ => raw,
        };
        ctx.bind(&n.outputs[3], output);
    }
    let v = gqa_repeat_heads_batched(ctx, v, b, kvh, groups, k_len, v_hd)?;
    let probs_for_v = ctx.astype(probs, ctx.dtype_of(v))?;
    let out = ctx.matmul(probs_for_v, v)?;
    let out = ctx.astype(out, dt)?;
    match overlay {
        // Zero probabilities alone do not guarantee zero Y when V contains NaN or infinity.
        Some(overlay) => attention_zero_fully_masked_rows(ctx, out, overlay),
        None => Ok(out),
    }
}

// ---- Attention (ai.onnx) -----------------------------------------------------------------------

fn attention_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let q_in = ctx.resolve(&n.inputs[0])?;
    let k_in = ctx.resolve(&n.inputs[1])?;
    let v_in = ctx.resolve(&n.inputs[2])?;
    let dt = ctx.dtype_of(q_in);

    let qs = ctx.shape_of(q_in);
    let is3d = qs.len() == 3;
    let b = qs[0];

    let (qh, kvh, s, hd_q, hd_v, qh4, kh4, vh4) = if is3d {
        let qh = attr_int(n, "q_num_heads", 0) as i32;
        let kvh = attr_int(n, "kv_num_heads", 0) as i32;
        let s = qs[1];
        let ks = ctx.shape_of(k_in);
        let vs = ctx.shape_of(v_in);
        let hd_q = qs[2] / qh;
        let hd_k = ks[2] / kvh;
        let hd_v = vs[2] / kvh;
        let qh4 = split_heads(ctx, q_in, b, s, qh, hd_q)?;
        let kh4 = split_heads(ctx, k_in, b, ks[1], kvh, hd_k)?;
        let vh4 = split_heads(ctx, v_in, b, vs[1], kvh, hd_v)?;
        (qh, kvh, s, hd_q, hd_v, qh4, kh4, vh4)
    } else {
        let qh = qs[1];
        let kvh = ctx.shape_of(k_in)[1];
        let s = qs[2];
        let hd_q = qs[3];
        let hd_v = ctx.shape_of(v_in)[3];
        (qh, kvh, s, hd_q, hd_v, q_in, k_in, v_in)
    };

    let has_past = present(n, 4) && present(n, 5);
    let (present_k, present_v) = if has_past {
        let pk = ctx.resolve(&n.inputs[4])?;
        let pv = ctx.resolve(&n.inputs[5])?;
        (concat2(ctx, pk, kh4, 2)?, concat2(ctx, pv, vh4, 2)?)
    } else {
        (kh4, vh4)
    };

    let scale = attr_scale(n, hd_q);
    let causal = attr_int(n, "is_causal", 0) != 0;
    let mask = if present(n, 3) {
        Some(ctx.resolve(&n.inputs[3])?)
    } else {
        None
    };

    let k_len = ctx.shape_of(present_k)[2];
    let past_len = k_len - ctx.shape_of(kh4)[2];
    let special_form = present(n, 6)
        || n.since_version >= 25
        || n.floats.get("softcap").copied().unwrap_or(0.0) != 0.0
        || (n.outputs.len() > 3 && !n.outputs[3].name.is_empty())
        || attr_int(n, "left_window_size", -1) >= 0
        || attr_int(n, "right_window_size", -1) >= 0
        || n.ints.contains_key("softmax_precision")
        || ctx.dtype_of(vh4) != dt;
    let attn = if special_form {
        attention_qk_output_op(
            ctx, n, qh4, present_k, present_v, b, qh, kvh, s, k_len, hd_q, hd_v, past_len, dt, mask,
        )?
    } else {
        let mask = if causal {
            // MLX's built-in causal mode is lower-right aligned; ONNX aligns to the actual past.
            Some(combined_causal_mask(ctx, s, k_len, past_len, dt, mask)?)
        } else {
            mask.map(|mask| mask_as_additive(ctx, mask, dt))
                .transpose()?
        };
        let out = sdpa_dispatch(ctx, qh4, present_k, present_v, scale, false, mask, dt)?;
        match mask {
            // MLX SDPA does not implement ONNX's fully-masked-row zero-output contract.
            Some(mask) => attention_zero_fully_masked_rows(ctx, out, mask)?,
            None => out,
        }
    };

    if is3d {
        // [B,qh,S,hd_v] -> [B,S,qh*hd_v].
        let t = ctx.transpose(attn, &[0, 2, 1, 3])?;
        let out = ctx.reshape(t, &[b, s, qh * hd_v])?;
        ctx.bind(&n.outputs[0], out);
    } else {
        ctx.bind(&n.outputs[0], attn); // already [B,qh,S,hd_v]
    }
    if n.outputs.len() >= 2 && !n.outputs[1].name.is_empty() {
        ctx.bind(&n.outputs[1], present_k);
    }
    if n.outputs.len() >= 3 && !n.outputs[2].name.is_empty() {
        ctx.bind(&n.outputs[2], present_v);
    }
    Ok(())
}

// ---- MultiHeadAttention (com.microsoft) --------------------------------------------------------

fn ms_attention_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let h = attr_int(n, "num_heads", 0) as i32;
    let x = ctx.resolve(&n.inputs[0])?;
    let w = ctx.resolve(&n.inputs[1])?;
    let bias = ctx.resolve(&n.inputs[2])?;
    let xs = ctx.shape_of(x);
    let (b, s, d) = (xs[0], xs[1], xs[2]);

    let qkv = ctx.matmul(x, w)?;
    let qkv = add(ctx, qkv, bias)?;
    let q = slice(ctx, qkv, &[0, 0, 0], &[b, s, d])?;
    let k = slice(ctx, qkv, &[0, 0, d], &[b, s, 2 * d])?;
    let v = slice(ctx, qkv, &[0, 0, 2 * d], &[b, s, 3 * d])?;
    let hd = d / h;
    let q4 = split_heads(ctx, q, b, s, h, hd)?;
    let k4 = split_heads(ctx, k, b, s, h, hd)?;
    let v4 = split_heads(ctx, v, b, s, h, hd)?;
    let scale = attr_scale(n, hd);
    let attn = sdpa_dispatch(ctx, q4, k4, v4, scale, false, None, ctx.dtype_of(q4))?;
    let t = ctx.transpose(attn, &[0, 2, 1, 3])?;
    let out = ctx.reshape(t, &[b, s, d])?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn multihead_attention_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let h = attr_int(n, "num_heads", 0) as i32;
    let mut q_in = ctx.resolve(&n.inputs[0])?;
    let mut k_in = ctx.resolve(&n.inputs[1])?;
    let mut v_in = ctx.resolve(&n.inputs[2])?;

    let qs = ctx.shape_of(q_in);
    let ks = ctx.shape_of(k_in);
    let vs = ctx.shape_of(v_in);
    let (b, s, dq) = (qs[0], qs[1], qs[2]);
    let (lk, dk, hd_k) = if ks.len() == 3 {
        (ks[1], ks[2], ks[2] / h)
    } else {
        (ks[2], ks[1] * ks[3], ks[3])
    };
    let (lv, dv, hd_v) = if vs.len() == 3 {
        (vs[1], vs[2], vs[2] / h)
    } else {
        (vs[2], vs[1] * vs[3], vs[3])
    };
    let hd_q = dq / h;

    if present(n, 3) {
        let bias = ctx.resolve(&n.inputs[3])?; // 1D [Dq+Dk+Dv]
        // `mlx_compile(..., shapeless=true)` cannot infer Slice output shapes, even for a constant
        // 1-D bias. Take with a fixed-size index array preserves the same result and is traceable.
        let qb = take_1d_range(ctx, bias, 0, dq)?;
        let qb = ctx.reshape(qb, &[1, 1, dq])?;
        q_in = add(ctx, q_in, qb)?;
        if ks.len() == 3 {
            let kb = take_1d_range(ctx, bias, dq, dq + dk)?;
            let kb = ctx.reshape(kb, &[1, 1, dk])?;
            k_in = add(ctx, k_in, kb)?;
            let vb = take_1d_range(ctx, bias, dq + dk, dq + dk + dv)?;
            let vb = ctx.reshape(vb, &[1, 1, dv])?;
            v_in = add(ctx, v_in, vb)?;
        }
    }

    let qh4 = split_heads(ctx, q_in, b, s, h, hd_q)?;
    let kh4 = if ks.len() == 3 {
        split_heads(ctx, k_in, b, lk, h, hd_k)?
    } else {
        k_in
    };
    let vh4 = if vs.len() == 3 {
        split_heads(ctx, v_in, b, lv, h, hd_v)?
    } else {
        v_in
    };

    let has_past = present(n, 6) && present(n, 7);
    let (present_k, present_v) = if has_past {
        let pk = ctx.resolve(&n.inputs[6])?;
        let pv = ctx.resolve(&n.inputs[7])?;
        (concat2(ctx, pk, kh4, 2)?, concat2(ctx, pv, vh4, 2)?)
    } else {
        (kh4, vh4)
    };

    let scale = attr_scale(n, hd_q);
    let causal = attr_int(n, "unidirectional", 0) != 0;
    let dt = ctx.dtype_of(qh4);

    let emit_qk = n.outputs.len() > 3 && !n.outputs[3].name.is_empty();
    let attn = if emit_qk {
        let kt = ctx.transpose(present_k, &[0, 1, 3, 2])?;
        let scores = ctx.matmul(qh4, kt)?;
        let scale_v = ctx.scalar_f32(scale);
        let scale_v = ctx.astype(scale_v, dt)?;
        let scores = mul(ctx, scores, scale_v)?;
        if emit_qk {
            ctx.bind(&n.outputs[3], scores);
        }
        let probs = ctx.softmax_axis(scores, 3)?;
        ctx.matmul(probs, present_v)?
    } else {
        sdpa_dispatch(ctx, qh4, present_k, present_v, scale, causal, None, dt)?
    };

    // [B,H,S,hd_v] -> [B,S,H*hd_v].
    let t = ctx.transpose(attn, &[0, 2, 1, 3])?;
    let out = ctx.reshape(t, &[b, s, h * hd_v])?;
    ctx.bind(&n.outputs[0], out);
    if n.outputs.len() > 1 && !n.outputs[1].name.is_empty() {
        ctx.bind(&n.outputs[1], present_k);
    }
    if n.outputs.len() > 2 && !n.outputs[2].name.is_empty() {
        ctx.bind(&n.outputs[2], present_v);
    }
    Ok(())
}

// ---- PackedMultiHeadAttention (com.microsoft) --------------------------------------------------

fn packed_multihead_attention_op(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
) -> Result<(), MlxError> {
    let num_heads = attr_int(n, "num_heads", 0) as i32;
    let q = ctx.resolve(&n.inputs[0])?;
    let k = ctx.resolve(&n.inputs[1])?;
    let v = ctx.resolve(&n.inputs[2])?;
    let qs = ctx.shape_of(q);
    let vs = ctx.shape_of(v);
    let token_count = qs[0];
    let hidden = qs[1];
    let v_hidden = vs[1];
    let head = hidden / num_heads;
    let v_head = v_hidden / num_heads;
    let scale = attr_scale(n, head);
    let out_dt = ctx.dtype_of(q);

    // The packed rows are already in output order. token_offset describes their positions in the
    // padded source tensor, but no restore-padding step is part of this op's output contract.
    let cumulative = read_int_input(ctx, &n.inputs[5])?;
    if cumulative.len() < 2 || cumulative[0] != 0 {
        return Err(
            "PackedMultiHeadAttention: cumulative_sequence_length must start at zero".to_string(),
        );
    }
    if cumulative.last().copied() != Some(token_count as i64) {
        return Err(
            "PackedMultiHeadAttention: cumulative_sequence_length must end at token_count"
                .to_string(),
        );
    }

    let mut outputs = Vec::with_capacity(cumulative.len() - 1);
    for pair in cumulative.windows(2) {
        let start = pair[0] as i32;
        let end = pair[1] as i32;
        let seq = end - start;
        if seq <= 0 {
            continue;
        }
        let qb = slice(ctx, q, &[start, 0], &[end, hidden])?;
        let kb = slice(ctx, k, &[start, 0], &[end, hidden])?;
        let vb = slice(ctx, v, &[start, 0], &[end, v_hidden])?;
        let qb = ctx.reshape(qb, &[1, seq, hidden])?;
        let kb = ctx.reshape(kb, &[1, seq, hidden])?;
        let vb = ctx.reshape(vb, &[1, seq, v_hidden])?;
        let qh = split_heads(ctx, qb, 1, seq, num_heads, head)?;
        let kh = split_heads(ctx, kb, 1, seq, num_heads, head)?;
        let vh = split_heads(ctx, vb, 1, seq, num_heads, v_head)?;
        let out = sdpa_dispatch(ctx, qh, kh, vh, scale, false, None, out_dt)?;
        let out = ctx.transpose(out, &[0, 2, 1, 3])?;
        outputs.push(ctx.reshape(out, &[seq, v_hidden])?);
    }

    let output = match outputs.len() {
        0 => slice(ctx, v, &[0, 0], &[0, v_hidden])?,
        1 => outputs[0],
        _ => {
            let mut arrays = VectorArray::new();
            for output in &outputs {
                arrays.append(*output);
            }
            ctx.emit(|res, stream| unsafe {
                mlx::mlx_concatenate_axis(res, arrays.as_raw(), 0, stream)
            })?
        }
    };
    ctx.bind(&n.outputs[0], output);
    Ok(())
}

// ---- RotaryEmbedding (ai.onnx opset 23 / com.microsoft) ----------------------------------------

/// Apply RoPE over the first rot (= 2*half) head dims of x4 [B,N,S,hd]; cos4/sin4 are [Bc,1,S,half]
/// (Bc == B for per-position gather, 1 for the offset/absent forms) and broadcast over the head axis.
fn rope_apply(
    ctx: &mut TranslationContext,
    x4: mlx::mlx_array,
    cos4: mlx::mlx_array,
    sin4: mlx::mlx_array,
    half: i32,
    interleaved: bool,
) -> Result<mlx::mlx_array, MlxError> {
    let xs = ctx.shape_of(x4); // [B,N,S,hd]
    let (b, nh, s, hd) = (xs[0], xs[1], xs[2], xs[3]);
    let rot = 2 * half;
    if ctx.rope_dynamic() && !interleaved && rot == hd {
        let cos_full = concat2(ctx, cos4, cos4, 3)?;
        let sin_full = concat2(ctx, sin4, sin4, 3)?;
        return gqa_rope_matmul(ctx, x4, cos_full, sin_full, hd, half);
    }
    let rotated = if !interleaved {
        let x1 = slice(ctx, x4, &[0, 0, 0, 0], &[b, nh, s, half])?;
        let x2 = slice(ctx, x4, &[0, 0, 0, half], &[b, nh, s, rot])?;
        let x1c = mul(ctx, x1, cos4)?;
        let x2s = mul(ctx, x2, sin4)?;
        let o1 = sub(ctx, x1c, x2s)?;
        let x1s = mul(ctx, x1, sin4)?;
        let x2c = mul(ctx, x2, cos4)?;
        let o2 = add(ctx, x1s, x2c)?;
        concat2(ctx, o1, o2, 3)?
    } else {
        let bc = ctx.shape_of(cos4)[0];
        let sl = slice(ctx, x4, &[0, 0, 0, 0], &[b, nh, s, rot])?;
        let xr = ctx.reshape(sl, &[b, nh, s, half, 2])?;
        let xe = slice(ctx, xr, &[0, 0, 0, 0, 0], &[b, nh, s, half, 1])?; // even lanes (x1)
        let xo = slice(ctx, xr, &[0, 0, 0, 0, 1], &[b, nh, s, half, 2])?; // odd lanes (x2)
        let c = ctx.reshape(cos4, &[bc, 1, s, half, 1])?;
        let sn = ctx.reshape(sin4, &[bc, 1, s, half, 1])?;
        let xec = mul(ctx, xe, c)?;
        let xosn = mul(ctx, xo, sn)?;
        let oe = sub(ctx, xec, xosn)?; // real -> even lanes
        let xesn = mul(ctx, xe, sn)?;
        let xoc = mul(ctx, xo, c)?;
        let oo = add(ctx, xesn, xoc)?; // imag -> odd lanes
        let cat = concat2(ctx, oe, oo, 4)?;
        ctx.reshape(cat, &[b, nh, s, rot])?
    };
    if rot == hd {
        return Ok(rotated);
    }
    let tail = slice(ctx, x4, &[0, 0, 0, rot], &[b, nh, s, hd])?;
    concat2(ctx, rotated, tail, 3)
}

/// Build a [Bc,1,S,half] cos/sin tensor (broadcastable over heads) from a cos/sin cache. pos_rank < 0
/// = absent position_ids (cache is already [B,S,half]); pos_rank == 2 = per-position gather
/// (position_ids [B,S], Bc == B); otherwise the offset form (position_ids [1], positions offset+[0,S),
/// Bc == 1).
fn gather_cache(
    ctx: &mut TranslationContext,
    cache: mlx::mlx_array,
    pos: Option<mlx::mlx_array>,
    pos_rank: i32,
    s: i32,
) -> Result<mlx::mlx_array, MlxError> {
    if pos_rank < 0 {
        let cs = ctx.shape_of(cache); // [B,S,half]
        return ctx.reshape(cache, &[cs[0], 1, cs[1], cs[2]]);
    }
    let half = ctx.shape_of(cache)[1]; // cache: [max_seq, half]
    let pos = pos.expect("position_ids present when pos_rank >= 0");
    let (idx, bc) = if pos_rank == 2 {
        let idx = ctx.astype(pos, mlx::mlx_dtype__MLX_INT32)?; // [B,S]
        let bc = ctx.shape_of(pos)[0];
        (idx, bc)
    } else {
        let off = ctx.astype(pos, mlx::mlx_dtype__MLX_INT32)?; // [1]
        let ar = ctx.emit(|res, st| unsafe {
            mlx::mlx_arange(res, 0.0, s as f64, 1.0, mlx::mlx_dtype__MLX_INT32, st)
        })?; // [S]
        let idx = add(ctx, off, ar)?; // [S] = offset + [0,S)
        (idx, 1)
    };
    let g = ctx.emit(|res, st| unsafe { mlx::mlx_take_axis(res, cache, idx, 0, st) })?; // [B,S,half] or [S,half]
    ctx.reshape(g, &[bc, 1, s, half])
}

fn rotary_position_slots(n: &NodeDesc) -> Option<(usize, usize)> {
    match (n.domain.as_str(), n.op_type.as_str()) {
        ("com.microsoft", "RotaryEmbedding" | "MRotaryEmbedding") => Some((1, 2)),
        ("" | "ai.onnx", "RotaryEmbedding") => Some((3, 1)),
        _ => None,
    }
}

pub(crate) fn needs_rotary_position_validation(n: &NodeDesc) -> bool {
    rotary_position_slots(n).is_some_and(|(pi, _)| present(n, pi))
        || n.subgraphs
            .iter()
            .any(|g| g.nodes.iter().any(needs_rotary_position_validation))
}

fn validate_rotary_positions(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let Some((pi, ci)) = rotary_position_slots(n) else {
        return Ok(());
    };
    if !present(n, pi) {
        return Ok(());
    }
    let cache = ctx.resolve(&n.inputs[ci])?;
    let rows = ctx.shape_of(cache)[0] as i64;
    let pos = ctx.resolve(&n.inputs[pi])?;
    let offset = ctx.ndim(pos) == 1;
    let span = if offset {
        let x = ctx.resolve(&n.inputs[0])?;
        let shape = ctx.shape_of(x);
        shape[if shape.len() == 4 { 2 } else { 1 }] as i64
    } else {
        1
    };
    let values = read_int_input(ctx, &n.inputs[pi])?;
    if values.iter().any(|&value| value < 0 || value > rows - span) {
        return Err("RotaryEmbedding: position_ids exceed the supplied cache".to_string());
    }
    Ok(())
}

/// Boundary positions are checked before every compiled apply, not only on the first trace.
pub(crate) fn validate_compiled_rotary_positions(
    ctx: &mut TranslationContext,
    nodes: &[NodeDesc],
) -> Result<bool, MlxError> {
    for n in nodes {
        if n.subgraphs
            .iter()
            .any(|g| g.nodes.iter().any(needs_rotary_position_validation))
        {
            return Ok(false);
        }
        let Some((pi, ci)) = rotary_position_slots(n) else {
            continue;
        };
        if !present(n, pi) {
            continue;
        }
        if [pi, ci]
            .iter()
            .any(|&i| n.inputs[i].source == Src::Intermediate)
        {
            return Ok(false);
        }
        let pos = ctx.resolve(&n.inputs[pi])?;
        if ctx.ndim(pos) == 1 && n.inputs[0].source == Src::Intermediate {
            return Ok(false);
        }
        validate_rotary_positions(ctx, n)?;
    }
    Ok(true)
}

fn rotary_embedding_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    if !ctx.shape_keyed_compile() && !ctx.shapeless_compile() {
        validate_rotary_positions(ctx, n)?;
    }
    let ms = n.domain == "com.microsoft";
    let ci = if ms { 2 } else { 1 };
    let si = if ms { 3 } else { 2 };
    let pi = if ms { 1 } else { 3 };
    let interleaved = attr_int(n, "interleaved", 0) != 0;

    let x = ctx.resolve(&n.inputs[0])?;
    let xs = ctx.shape_of(x);
    let rank = xs.len();

    let cos_cache = ctx.resolve(&n.inputs[ci])?;
    let sin_cache = ctx.resolve(&n.inputs[si])?;

    let (b, nh, s, hd, x4) = if rank == 4 {
        (xs[0], xs[1], xs[2], xs[3], x)
    } else {
        let b = xs[0];
        let s = xs[1];
        let nh_attr = attr_int(n, "num_heads", 0) as i32;
        // num_heads=0 (Gemma3n form): infer the head size from the cos cache. With
        // rotary_embedding_dim=0 (full-head rotary) head_size = 2 * cos_cache_last_dim, so
        // num_heads = hidden / head_size.
        let hd = if nh_attr > 0 {
            xs[2] / nh_attr
        } else {
            let cos_shape = ctx.shape_of(cos_cache);
            2 * cos_shape[cos_shape.len() - 1]
        };
        let nh = xs[2] / hd;
        let r = ctx.reshape(x, &[b, s, nh, hd])?;
        let x4 = ctx.transpose(r, &[0, 2, 1, 3])?; // [B,N,S,hd]
        (b, nh, s, hd, x4)
    };

    let has_pos = present(n, pi);
    let pos_rank = if has_pos {
        let p = ctx.resolve(&n.inputs[pi])?;
        ctx.ndim(p) as i32
    } else {
        -1
    };
    let pos = if has_pos {
        Some(ctx.resolve(&n.inputs[pi])?)
    } else {
        None
    };
    let cos4 = gather_cache(ctx, cos_cache, pos, pos_rank, s)?;
    let sin4 = gather_cache(ctx, sin_cache, pos, pos_rank, s)?;
    let half = ctx.shape_of(cos4)[3];
    let out4 = rope_apply(ctx, x4, cos4, sin4, half, interleaved)?;
    ctx.mark_composed("RotaryEmbedding gathers the supplied cache values exactly");

    if rank == 4 {
        ctx.bind(&n.outputs[0], out4); // already [B,N,S,hd]
    } else {
        let t = ctx.transpose(out4, &[0, 2, 1, 3])?;
        let out = ctx.reshape(t, &[b, s, nh * hd])?;
        ctx.bind(&n.outputs[0], out);
    }
    Ok(())
}

fn mrotary_embedding_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    if !ctx.shape_keyed_compile() && !ctx.shapeless_compile() {
        validate_rotary_positions(ctx, n)?;
    }
    let x = ctx.resolve(&n.inputs[0])?;
    let pos = ctx.resolve(&n.inputs[1])?;
    let cos_cache = ctx.resolve(&n.inputs[2])?;
    let sin_cache = ctx.resolve(&n.inputs[3])?;
    let xs = ctx.shape_of(x);
    let rank = xs.len();
    let (b, nh, s, hd, x4) = if rank == 4 {
        (xs[0], xs[1], xs[2], xs[3], x)
    } else {
        let nh = attr_int(n, "num_heads", 0) as i32;
        let hd = xs[2] / nh;
        let reshaped = ctx.reshape(x, &[xs[0], xs[1], nh, hd])?;
        (
            xs[0],
            nh,
            xs[1],
            hd,
            ctx.transpose(reshaped, &[0, 2, 1, 3])?,
        )
    };
    let half = ctx.shape_of(cos_cache)[1];
    let sections = n
        .int_arrays
        .get("mrope_section")
        .ok_or("MRotaryEmbedding: mrope_section is required")?;
    let layout = attr_int(n, "mrope_layout", 0);

    let mut cos_streams = Vec::with_capacity(3);
    let mut sin_streams = Vec::with_capacity(3);
    for stream in 0..3 {
        let ids = slice(ctx, pos, &[stream, 0, 0], &[stream + 1, b, s])?;
        let ids = ctx.squeeze(ids, 0)?;
        let ids = ctx.astype(ids, mlx::mlx_dtype__MLX_INT32)?;
        cos_streams
            .push(ctx.emit(|res, st| unsafe { mlx::mlx_take_axis(res, cos_cache, ids, 0, st) })?);
        sin_streams
            .push(ctx.emit(|res, st| unsafe { mlx::mlx_take_axis(res, sin_cache, ids, 0, st) })?);
    }

    let mut assignment = vec![0usize; half as usize];
    if layout == 0 {
        let mut col = 0usize;
        for (stream, &width) in sections.iter().enumerate() {
            for _ in 0..width {
                assignment[col] = stream;
                col += 1;
            }
        }
    } else {
        for stream in 1..3 {
            let stop = sections[stream] * 3;
            let mut col = stream as i64;
            while col < stop && col < half as i64 {
                assignment[col as usize] = stream;
                col += 3;
            }
        }
    }

    let combine = |ctx: &mut TranslationContext,
                   streams: &[mlx::mlx_array]|
     -> Result<mlx::mlx_array, MlxError> {
        let mut out = slice(ctx, streams[assignment[0]], &[0, 0, 0], &[b, s, 1])?;
        for col in 1..half {
            let selected = slice(
                ctx,
                streams[assignment[col as usize]],
                &[0, 0, col],
                &[b, s, col + 1],
            )?;
            out = concat2(ctx, out, selected, 2)?;
        }
        let scale = n.floats.get("scale").copied().unwrap_or(1.0);
        if scale != 1.0 {
            let dtype = ctx.dtype_of(out);
            out = ctx.astype(out, mlx::mlx_dtype__MLX_FLOAT32)?;
            let scalar = ctx.scalar_f32(scale);
            out = ctx.mul(out, scalar)?;
            out = ctx.astype(out, dtype)?;
        }
        ctx.reshape(out, &[b, 1, s, half])
    };
    let cos4 = combine(ctx, &cos_streams)?;
    let sin4 = combine(ctx, &sin_streams)?;
    let interleaved = attr_int(n, "interleaved", 0) != 0;
    let out4 = rope_apply(ctx, x4, cos4, sin4, half, interleaved)?;
    if rank == 4 {
        ctx.bind(&n.outputs[0], out4);
    } else {
        let out = ctx.transpose(out4, &[0, 2, 1, 3])?;
        let out = ctx.reshape(out, &[b, s, nh * hd])?;
        ctx.bind(&n.outputs[0], out);
    }
    Ok(())
}

// ---- claim predicates --------------------------------------------------------------------------

fn dtype_of(node: &NodeView, i: usize) -> Option<ort::ONNXTensorElementDataType> {
    node.input_info(i).map(|s| s.dtype)
}

fn is_int32(t: ort::ONNXTensorElementDataType) -> bool {
    t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
}

/// GroupQueryAttention (com.microsoft): separate-QKV decode/prefill layout. Three accepted layouts:
///   * 7-input (q, k, v, past_k, past_v, seqlens_k, total_seq) — external/no RoPE decoder.
///   * 9-input (q, k, v, past_k, past_v, seqlens_k, total_seq, cos, sin) — in-op RoPE decoder.
///   * 11-input (…, seqlens_k, total_seq, cos, sin, position_ids, attention_bias) — the Gemma3n
///     variant with `do_rotary=0` (cos/sin absent, rotary applied by external RotaryEmbedding nodes)
///     and an additive `attention_bias` mask (input 10). `position_ids` (input 9) is ignored.
///     All floating inputs/outputs share one dtype; seqlens_k / total_sequence_length are int32.
fn group_query_attention_claim(node: &NodeView) -> ClaimResult {
    require!(
        (1..=4).contains(&node.num_outputs()),
        "requires 1 to 4 outputs"
    );
    require!(
        node.output_present(1) == node.output_present(2),
        "present_key and present_value must both be provided or both omitted"
    );
    let sliding = node.int_attr("sliding_window_cache", 0) != 0;
    require!(
        matches!(node.int_attr("causal", 1), 0 | 1),
        "causal must be 0 or 1"
    );
    let window = node.int_attr("local_window_size", -1);
    require!(
        window == -1 || window > 0,
        "local_window_size must be -1 or positive"
    );
    require!(
        node.int_attr("causal", 1) != 0 || window == -1,
        "bidirectional GroupQueryAttention requires local_window_size=-1"
    );
    require!(
        node.input_present(3) == node.input_present(4),
        "past_key and past_value must be provided together"
    );
    let out_type = match node.output_info(0) {
        Some(o) if is_mlx_float(o.dtype) => o.dtype,
        Some(o) => deny!(
            "output must have an MLX float dtype, got {}",
            crate::registry::ort_dtype_name(o.dtype)
        ),
        None => deny!("output lacks tensor type/shape info"),
    };
    let ninputs = node.num_inputs();
    require!(
        (7..=16).contains(&ninputs),
        "expects 7 required inputs plus optional rotary/bias/QK-norm inputs, got {}",
        ninputs
    );
    // The 11-input Gemma3n variant only maps to MLX when rotary is external (do_rotary=0): cos/sin at
    // 7,8 are absent, so we must not require their dtype and must not resolve them in the handler.
    let has_bias = node.input_present(10);
    let do_rotary = node.int_attr("do_rotary", 0) != 0;
    require!(
        ninputs != 7 || !do_rotary,
        "7-input GroupQueryAttention requires do_rotary=0 because cos/sin caches are absent"
    );
    if sliding {
        require!(
            node.int_attr("local_window_size", -1) > 0,
            "sliding_window_cache=1 requires local_window_size > 0"
        );
        require!(
            !has_bias,
            "sliding_window_cache is incompatible with attention_bias"
        );
        require!(
            node.output_present(1) && node.output_present(2),
            "sliding_window_cache requires present key/value outputs"
        );
        let window = node.int_attr("local_window_size", -1);
        require!(
            node.input_info(3)
                .is_some_and(|v| v.shape.len() == 4 && v.shape[2] >= window),
            "sliding-window key cache capacity must be at least local_window_size"
        );
        require!(
            node.input_info(4)
                .is_some_and(|v| v.shape.len() == 4 && v.shape[2] >= window),
            "sliding-window value cache capacity must be at least local_window_size"
        );
    }
    let packed_qkv = node.input_present(0) && !node.input_present(1) && !node.input_present(2);
    require!(
        packed_qkv || (node.input_present(0) && node.input_present(1) && node.input_present(2)),
        "Q/K/V must be either all separate or packed in input 0 with inputs 1 and 2 absent"
    );
    // Float inputs that must match the output dtype. cos/sin (7,8) exist only in the 9-input form
    // WITH in-op rotary (do_rotary=1); with external rotary (do_rotary=0) they are absent, as is the
    // whole 11-input Gemma3n variant.
    let mut float_idx = if packed_qkv {
        vec![0usize]
    } else {
        vec![0usize, 1, 2]
    };
    if node.input_present(3) {
        float_idx.extend([3, 4]);
    }
    if do_rotary {
        float_idx.extend([7, 8]);
    }
    if has_bias {
        float_idx.push(10);
    }
    if node.input_present(14) {
        float_idx.extend([14, 15]);
    }
    for idx in float_idx {
        let dtype = match dtype_of(node, idx) {
            Some(dtype) => dtype,
            None => deny!("input {} lacks tensor type/shape info", idx),
        };
        require!(
            dtype == out_type,
            "input {} must have output dtype {}, got {}",
            idx,
            crate::registry::ort_dtype_name(out_type),
            crate::registry::ort_dtype_name(dtype)
        );
    }
    for idx in 1..node.num_outputs().min(3) {
        if !node.output_present(idx) {
            continue;
        }
        let dtype = match node.output_info(idx) {
            Some(o) => o.dtype,
            None => deny!("output {} lacks tensor type/shape info", idx),
        };
        require!(
            dtype == out_type,
            "output {} must have dtype {}, got {}",
            idx,
            crate::registry::ort_dtype_name(out_type),
            crate::registry::ort_dtype_name(dtype)
        );
    }
    for idx in [5, 6] {
        let dtype = match dtype_of(node, idx) {
            Some(dtype) => dtype,
            None => deny!("input {} lacks tensor type/shape info", idx),
        };
        require!(
            is_int32(dtype),
            "input {} must have int32 dtype, got {}",
            idx,
            crate::registry::ort_dtype_name(dtype)
        );
    }
    // 11-input variant: attention_bias (input 10) is an additive [B,1,S,total] mask folded into the
    // SDPA scores. Require it present, sharing the output float dtype, and rank 4. position_ids
    // (input 9) is ignored when do_rotary=0, so it may be present or absent.
    if has_bias {
        let bias = match node.input_info(10) {
            Some(info) => info,
            None => deny!("attention_bias (input 10) lacks tensor type/shape info"),
        };
        require!(
            bias.dtype == out_type,
            "attention_bias (input 10) must have output dtype {}, got {}",
            crate::registry::ort_dtype_name(out_type),
            crate::registry::ort_dtype_name(bias.dtype)
        );
        require!(
            bias.shape.len() == 4,
            "attention_bias (input 10) must have rank 4, got rank {}",
            bias.shape.len()
        );
    }
    let nh = node.int_attr("num_heads", 0);
    let kvh = node.int_attr("kv_num_heads", 0);
    require!(
        nh > 0 && kvh > 0 && nh % kvh == 0,
        "num_heads ({}) must be a positive multiple of kv_num_heads ({})",
        nh,
        kvh
    );
    let q = node.input_info(0).unwrap();
    require!(
        q.shape.len() == 3 && q.shape[2] > 0,
        "query must have rank 3 and a positive static hidden size"
    );
    let total_heads = if packed_qkv { nh + 2 * kvh } else { nh };
    require!(
        q.shape[2] % total_heads == 0,
        "query hidden size must match head count"
    );
    let head = q.shape[2] / total_heads;
    if has_bias {
        let bias = node.input_info(10).unwrap();
        require!(
            (bias.shape[0] < 0
                || bias.shape[0] == 1
                || q.shape[0] < 0
                || bias.shape[0] == q.shape[0])
                && (bias.shape[1] < 0 || bias.shape[1] == 1 || bias.shape[1] == nh)
                && (bias.shape[2] < 0 || q.shape[1] < 0 || bias.shape[2] == q.shape[1]),
            "attention_bias must broadcast to [batch,num_heads,sequence_length,total_sequence_length]"
        );
    }
    if !packed_qkv {
        for idx in [1, 2] {
            let kv = node.input_info(idx).unwrap();
            require!(
                kv.shape.len() == 3
                    && kv.shape[2] == kvh * head
                    && (kv.shape[0] < 0 || q.shape[0] < 0 || kv.shape[0] == q.shape[0])
                    && (kv.shape[1] < 0 || q.shape[1] < 0 || kv.shape[1] == q.shape[1]),
                "key/value must match query batch, sequence length and KV head geometry"
            );
        }
    }
    for idx in [3, 4] {
        if let Some(cache) = node.input_info(idx) {
            require!(
                cache.shape.len() == 4
                    && cache.shape[1] == kvh
                    && cache.shape[3] == head
                    && (cache.shape[0] < 0 || q.shape[0] < 0 || cache.shape[0] == q.shape[0]),
                "past cache must be [batch,kv_num_heads,capacity,head_size]"
            );
        }
    }
    if node.input_present(3) {
        require!(
            node.input_info(3).unwrap().shape == node.input_info(4).unwrap().shape,
            "past key/value cache shapes must match"
        );
    }
    require!(
        node.input_info(5).is_some_and(|v| v.shape.len() == 1
            && (v.shape[0] < 0 || q.shape[0] < 0 || v.shape[0] == q.shape[0])),
        "seqlens_k must be [batch]"
    );
    require!(
        node.input_info(6)
            .is_some_and(|v| v.shape.is_empty() || v.shape == [1]),
        "total_sequence_length must be scalar or [1]"
    );
    if do_rotary {
        let cos = node.input_info(7).unwrap();
        let sin = node.input_info(8).unwrap();
        require!(
            cos.shape.len() == 2
                && cos.shape == sin.shape
                && cos.shape[1] > 0
                && 2 * cos.shape[1] <= head,
            "rotary caches must have matching [positions,rotary_dim/2] shapes within the head"
        );
    }
    if packed_qkv {
        let packed = match node.input_info(0) {
            Some(info) => info,
            None => deny!("packed QKV input lacks tensor type/shape info"),
        };
        require!(
            packed.shape.len() == 3,
            "packed QKV input must have rank 3, got rank {}",
            packed.shape.len()
        );
        let hidden = packed.shape[2];
        let total_heads = nh + 2 * kvh;
        require!(
            hidden > 0 && hidden % total_heads == 0,
            "packed QKV hidden size ({}) must be divisible by total heads ({})",
            hidden,
            total_heads
        );
    }
    require!(
        !node.input_present(9) || !do_rotary,
        "custom position_ids with in-op rotary are unsupported"
    );
    if node.input_present(11) {
        require!(
            node.input_info(11)
                .is_some_and(|v| v.dtype == out_type && v.shape == [nh]),
            "head_sink must be [num_heads] in activation dtype"
        );
    }
    require!(
        !node.input_present(12) && !node.input_present(13),
        "quantized KV cache scales are unsupported"
    );
    require!(
        node.string_attr("k_quant_type", "NONE") == "NONE"
            && node.string_attr("v_quant_type", "NONE") == "NONE"
            && node.int_attr("kv_cache_bit_width", 0) == 0,
        "quantized KV cache is unsupported"
    );
    require!(
        node.input_present(14) == node.input_present(15),
        "q_norm_weight and k_norm_weight must be provided together"
    );
    if node.input_present(14) {
        let q = node.input_info(0).unwrap();
        let head = if packed_qkv {
            q.shape[2] / (nh + 2 * kvh)
        } else {
            require!(
                q.shape.len() == 3 && q.shape[2] > 0 && q.shape[2] % nh == 0,
                "query hidden size must be divisible by num_heads"
            );
            q.shape[2] / nh
        };
        for idx in [14usize, 15] {
            let weight = node.input_info(idx).unwrap();
            require!(
                weight.shape == [head],
                "Q/K norm weight {} must have shape [head_size={head}]",
                idx
            );
        }
    }
    require!(
        matches!(node.int_attr("smooth_softmax", 0), -1..=1),
        "smooth_softmax must be -1 (schema default), 0 or 1"
    );
    let qk = node.int_attr("qk_output", 0);
    require!(
        matches!(qk, 0..=2) && (qk != 0) == node.output_present(3),
        "qk_output must be 0, 1 or 2 and enabled exactly when output_qk is present"
    );
    require!(
        !sliding || qk == 0,
        "sliding_window_cache with QK diagnostics is unsupported"
    );
    if qk != 0 {
        require!(
            node.output_info(3).is_some_and(|v| v.dtype == out_type
                && v.shape.len() == 4
                && (v.shape[0] < 0 || q.shape[0] < 0 || v.shape[0] == q.shape[0])
                && (v.shape[1] < 0 || v.shape[1] == nh)
                && (v.shape[2] < 0 || q.shape[1] < 0 || v.shape[2] == q.shape[1])),
            "output_qk must be [batch,num_heads,sequence_length,total_sequence_length] in activation dtype"
        );
    }
    require!(
        node.float_attr("softcap", 0.0).is_finite() && node.float_attr("softcap", 0.0) >= 0.0,
        "softcap must be finite and non-negative"
    );
    Ok(())
}

const GQA_UNSUPPORTED_COMPILE_REASON: &str =
    "GroupQueryAttention form is not accepted by the lowering claim predicate";
const GQA_SLIDING_COMPILE_REASON: &str =
    "sliding-window GroupQueryAttention emits MLX Slice and is not compile-eligible";
const GQA_BIAS_COMPILE_REASON: &str =
    "attention-bias GroupQueryAttention emits MLX Slice and its bias semantics are eager-only";
const ATTENTION_COMPOSED_COMPILE_REASON: &str =
    "Attention v24/v25 logits or dynamic mask form uses composed MLX operations and is eager-only";

fn group_query_attention_compile_shape_safety_for(
    claimable: bool,
    sliding: bool,
    has_bias: bool,
) -> CompileShapeSafety {
    if !claimable {
        CompileShapeSafety::EagerOnly {
            reason: GQA_UNSUPPORTED_COMPILE_REASON,
        }
    } else if sliding {
        CompileShapeSafety::EagerOnly {
            reason: GQA_SLIDING_COMPILE_REASON,
        }
    } else if has_bias {
        CompileShapeSafety::EagerOnly {
            reason: GQA_BIAS_COMPILE_REASON,
        }
    } else {
        CompileShapeSafety::Shapeless
    }
}

fn group_query_attention_compile_shape_safety(node: &NodeView) -> CompileShapeSafety {
    if node.float_attr("softcap", 0.0) > 0.0
        || node.int_attr("qk_output", 0) != 0
        || node.int_attr("smooth_softmax", 0) == 1
        || node.input_present(11)
    {
        return CompileShapeSafety::EagerOnly {
            reason: "composed GQA logits, diagnostics and sinks use live eager geometry",
        };
    }
    if node
        .input_info(0)
        .is_none_or(|q| q.shape.first() != Some(&1))
        || !node.input_present(3)
        || node.int_attr("causal", 1) == 0
        || (node.int_attr("local_window_size", -1) > 0
            && node.int_attr("sliding_window_cache", 0) == 0)
    {
        return CompileShapeSafety::EagerOnly {
            reason: "per-batch sequence lengths and local/bidirectional GQA use live eager geometry",
        };
    }
    group_query_attention_compile_shape_safety_for(
        group_query_attention_claim(node).is_ok(),
        node.int_attr("sliding_window_cache", 0) != 0,
        node.input_present(10),
    )
}

fn attention_requires_composed_path(node: &NodeView) -> bool {
    node.since_version() >= 25
        || node.input_present(6)
        || node.output_present(3)
        || node.float_attr("softcap", 0.0) != 0.0
        || node.int_attr("left_window_size", -1) >= 0
        || node.int_attr("right_window_size", -1) >= 0
        || node.has_attr("softmax_precision")
        || matches!((dtype_of(node, 0), dtype_of(node, 2)), (Some(q), Some(v)) if q != v)
}

fn attention_compile_shape_safety_for(claimable: bool, composed: bool) -> CompileShapeSafety {
    if !claimable || composed {
        CompileShapeSafety::EagerOnly {
            reason: ATTENTION_COMPOSED_COMPILE_REASON,
        }
    } else {
        CompileShapeSafety::Shapeless
    }
}

fn attention_compile_shape_safety(node: &NodeView) -> CompileShapeSafety {
    attention_compile_shape_safety_for(
        attention_claim(node).is_ok(),
        attention_requires_composed_path(node),
    )
}

/// Q/K and output 0 use Attention's T1; V is T2 and may be a different MLX float dtype.
fn check_qkv_float(node: &NodeView) -> Option<ort::ONNXTensorElementDataType> {
    if node.num_inputs() < 3 || node.num_outputs() == 0 {
        return None;
    }
    let qd = dtype_of(node, 0)?;
    if !node.input_present(1) || !node.input_present(2) {
        return None;
    }
    let kd = dtype_of(node, 1)?;
    let vd = dtype_of(node, 2)?;
    let od = node.output_info(0)?.dtype;
    if is_mlx_float(qd) && kd == qd && is_mlx_float(vd) && od == qd {
        Some(qd)
    } else {
        None
    }
}

/// A past/present pair must be used together and preserve Attention's separate T1/T2 dtype rules.
fn check_kv_cache(
    node: &NodeView,
    past_k: usize,
    past_v: usize,
    qd: ort::ONNXTensorElementDataType,
    vd: ort::ONNXTensorElementDataType,
) -> bool {
    let pk = node.input_present(past_k);
    let pv = node.input_present(past_v);
    if pk != pv {
        return false;
    }
    if pk {
        match (dtype_of(node, past_k), dtype_of(node, past_v)) {
            (Some(a), Some(b)) if a == qd && b == vd => {}
            _ => return false,
        }
    }
    if node.output_present(1) != node.output_present(2) {
        return false;
    }
    if node.output_present(1)
        && !matches!(
            (node.output_info(1), node.output_info(2)),
            (Some(k), Some(v)) if k.dtype == qd && v.dtype == vd
        )
    {
        return false;
    }
    true
}

/// An attn/attention_bias mask must be bool or an MLX-supported float dtype.
///
/// A mask alongside `is_causal` is accepted: ONNX applies both, and both reduce
/// to one additive array, so the two are folded together before dispatch (see
/// [`combined_causal_mask`]). This is the same trick `gqa_eager_sdpa` already
/// uses for Gemma3n's `attention_bias`.
///
/// Declining the combination instead is expensive out of proportion to the op:
/// exported decoders routinely emit `is_causal` *and* a padding mask, so every
/// attention node in the model falls back to CPU and the graph is cut into a
/// fused subgraph per layer.
fn check_mask(node: &NodeView, mask_idx: usize) -> bool {
    if !node.input_present(mask_idx) {
        return true;
    }
    match dtype_of(node, mask_idx) {
        Some(md) => {
            md == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
                || is_mlx_float(md)
        }
        None => false,
    }
}

fn attention_mask_is_broadcastable(
    node: &NodeView,
    mask_idx: usize,
    b: i64,
    h: i64,
    q: i64,
    k: i64,
) -> bool {
    let Some(mask) = node.input_info(mask_idx) else {
        return !node.input_present(mask_idx);
    };
    if mask.shape.len() > 4 {
        return false;
    }
    let target = [b, h, q, k];
    let skip = target.len() - mask.shape.len();
    mask.shape.iter().enumerate().all(|(i, &dim)| {
        attention_mask_dimension_is_compatible(
            dim,
            target[skip + i],
            node.since_version() >= 25 && skip + i == 3,
        )
    })
}

fn attention_mask_dimension_is_compatible(dim: i64, target: i64, key_axis: bool) -> bool {
    dim < 0 || dim == 1 || target < 0 || dim == target || (key_axis && dim >= 0 && dim < target)
}

fn supported_softmax_precision(node: &NodeView) -> bool {
    !node.has_attr("softmax_precision")
        || mlx_softmax_precision(Some(node.int_attr("softmax_precision", 0))).is_some()
}

fn mlx_softmax_precision(precision: Option<i64>) -> Option<mlx::mlx_dtype> {
    match precision {
        None => None,
        Some(1) => Some(mlx::mlx_dtype__MLX_FLOAT32),
        Some(10) => Some(mlx::mlx_dtype__MLX_FLOAT16),
        Some(16) => Some(mlx::mlx_dtype__MLX_BFLOAT16),
        _ => None,
    }
}

/// Attention (ai.onnx, opset 23-28). Fast SDPA is retained for the legacy form; v24/v25 features
/// requiring logits, nonpad/window masks, or a requested QK output take an explicit composed path.
fn attention_claim(node: &NodeView) -> ClaimResult {
    let qd = match check_qkv_float(node) {
        Some(qd) => qd,
        None => deny!("Q/K/output must use one MLX float T1 and V must use an MLX float T2"),
    };
    let vd = match dtype_of(node, 2) {
        Some(vd) => vd,
        None => deny!("V lacks tensor type/shape info"),
    };
    let qshape = match node.input_info(0) {
        Some(i) => i.shape,
        None => deny!("Q lacks tensor type/shape info"),
    };
    let kshape = match node.input_info(1) {
        Some(i) => i.shape,
        None => deny!("K lacks tensor type/shape info"),
    };
    let vshape = match node.input_info(2) {
        Some(i) => i.shape,
        None => deny!("V lacks tensor type/shape info"),
    };
    let rank = qshape.len();
    require!(
        rank == 3 || rank == 4,
        "Q must have rank 3 or 4, got rank {}",
        rank
    );
    require!(
        kshape.len() == rank && vshape.len() == rank,
        "Q, K, and V must have equal rank, got {}, {}, {}",
        rank,
        kshape.len(),
        vshape.len()
    );

    let (qh, kvh, b, q_len, k_len) = if rank == 3 {
        let qh = node.int_attr("q_num_heads", 0);
        let kvh = node.int_attr("kv_num_heads", 0);
        require!(
            qh > 0 && kvh > 0,
            "q_num_heads and kv_num_heads must be positive, got {} and {}",
            qh,
            kvh
        );
        require!(
            qshape[2] > 0 && qshape[0] != 0,
            "Q hidden dimension must be static and positive (batch/seq may be dynamic), got shape {:?}",
            qshape
        );
        require!(
            kshape[2] > 0 && vshape[2] > 0,
            "K and V hidden dimensions must be static and positive, got {:?} and {:?}",
            kshape,
            vshape
        );
        require!(
            qshape[2] % qh == 0 && kshape[2] % kvh == 0 && vshape[2] % kvh == 0,
            "hidden dimensions Q/K/V ({}/{}/{}) must divide evenly by head counts {}/{}",
            qshape[2],
            kshape[2],
            vshape[2],
            qh,
            kvh
        );
        require!(
            qshape[0] < 0 || kshape[0] < 0 || qshape[0] == kshape[0],
            "Q and K batch dimensions must match"
        );
        require!(
            qshape[0] < 0 || vshape[0] < 0 || qshape[0] == vshape[0],
            "Q and V batch dimensions must match"
        );
        require!(
            qshape[2] / qh == kshape[2] / kvh,
            "Q and K head sizes must match"
        );
        (qh, kvh, qshape[0], qshape[1], kshape[1])
    } else {
        let qh = qshape[1];
        let kvh = kshape[1];
        require!(
            qh > 0 && kvh > 0,
            "Q and K head dimensions must be positive, got {} and {}",
            qh,
            kvh
        );
        require!(
            qshape[0] < 0 || kshape[0] < 0 || qshape[0] == kshape[0],
            "Q and K batch dimensions must match"
        );
        require!(
            qshape[0] < 0 || vshape[0] < 0 || qshape[0] == vshape[0],
            "Q and V batch dimensions must match"
        );
        require!(
            qshape[3] > 0 && kshape[3] > 0 && qshape[3] == kshape[3],
            "Q and K head sizes must match and be positive"
        );
        require!(
            kshape[1] == vshape[1] && kshape[2] == vshape[2],
            "K and V head and sequence dimensions must match"
        );
        require!(
            node.since_version() < 25
                || (!node.has_attr("q_num_heads") && !node.has_attr("kv_num_heads")),
            "q_num_heads and kv_num_heads must be absent for rank-4 inputs"
        );
        (qh, kvh, qshape[0], qshape[2], kshape[2])
    };
    require!(
        qh % kvh == 0,
        "Q head count {} must be a multiple of KV head count {}",
        qh,
        kvh
    );
    require!(
        check_mask(node, 3),
        "attention mask must be bool or an MLX float dtype"
    );
    let total_k_len = if node.input_present(4) {
        match node.input_info(4) {
            Some(past) if past.shape.len() == 4 && past.shape[2] >= 0 && k_len >= 0 => {
                past.shape[2] + k_len
            }
            _ => -1,
        }
    } else {
        k_len
    };
    require!(
        attention_mask_is_broadcastable(node, 3, b, qh, q_len, total_k_len),
        "attention mask must be broadcastable to [batch,q_heads,q_sequence,total_kv_sequence] \
         (short final dimensions are not representable without MLX Pad)"
    );
    require!(
        check_kv_cache(node, 4, 5, qd, vd),
        "past/present K/V must be paired and use their respective T1/T2 dtypes"
    );
    require!(
        !node.input_present(6) || (!node.input_present(4) && !node.output_present(1)),
        "nonpad_kv_seqlen cannot be combined with past or present cache tensors"
    );
    if node.input_present(6) {
        let nonpad = match node.input_info(6) {
            Some(info) => info,
            None => deny!("nonpad_kv_seqlen lacks tensor type/shape info"),
        };
        require!(
            nonpad.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                && nonpad.shape.len() == 1
                && (b < 0 || nonpad.shape[0] < 0 || nonpad.shape[0] == b),
            "nonpad_kv_seqlen must be int64 [batch]"
        );
    }
    require!(
        supported_softmax_precision(node),
        "softmax_precision must be an MLX GPU float type (FLOAT=1, FLOAT16=10, or BFLOAT16=16); DOUBLE=11 is unsupported"
    );
    for name in ["left_window_size", "right_window_size"] {
        require!(
            node.int_attr(name, -1) >= -1,
            "{name} must be -1 or nonnegative"
        );
    }
    if node.output_present(3) {
        let qk = match node.output_info(3) {
            Some(info) => info,
            None => deny!("qk_matmul_output lacks tensor type/shape info"),
        };
        require!(
            qk.dtype == qd
                && qk.shape.len() == 4
                && (b < 0 || qk.shape[0] < 0 || qk.shape[0] == b)
                && (qk.shape[1] < 0 || qk.shape[1] == qh)
                && (qk.shape[2] < 0 || q_len < 0 || qk.shape[2] == q_len)
                && (qk.shape[3] < 0 || total_k_len < 0 || qk.shape[3] == total_k_len),
            "qk_matmul_output must be T1 [batch,q_heads,q_sequence,total_kv_sequence]"
        );
    }
    Ok(())
}

/// MultiHeadAttention (com.microsoft). Separate 3D Q/K/V + optional projection bias.
fn ms_attention_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() >= 3 && node.num_outputs() >= 1,
        "expects at least 3 inputs and 1 output"
    );
    let (x, w, bias, out) = match (
        node.input_info(0),
        node.input_info(1),
        node.input_info(2),
        node.output_info(0),
    ) {
        (Some(x), Some(w), Some(b), Some(o)) => (x, w, b, o),
        _ => deny!("input, QKV weight/bias, or output lacks tensor type/shape info"),
    };
    require!(
        is_mlx_float(x.dtype)
            && x.dtype == w.dtype
            && w.dtype == bias.dtype
            && bias.dtype == out.dtype,
        "input, QKV weight/bias, and output must share one MLX float dtype"
    );
    require!(
        x.shape.len() == 3 && x.shape[2] > 0,
        "input must be rank 3 with a static positive hidden dimension, got {:?}",
        x.shape
    );
    let d = x.shape[2];
    require!(
        w.shape == [d, 3 * d] && bias.shape == [3 * d],
        "QKV weight/bias must have shapes [{d},{}] and [{}], got {:?} and {:?}",
        3 * d,
        3 * d,
        w.shape,
        bias.shape
    );
    let h = node.int_attr("num_heads", 0);
    require!(
        h > 0 && d % h == 0,
        "hidden dimension must divide num_heads"
    );
    require!(
        node.int_attr("unidirectional", 0) == 0,
        "unidirectional attention is unsupported"
    );
    require!(
        node.int_attr("do_rotary", 0) == 0,
        "fused rotary attention is unsupported"
    );
    for i in 3..node.num_inputs() {
        require!(!node.input_present(i), "optional input {i} is unsupported");
    }
    for i in 1..node.num_outputs() {
        require!(
            !node.output_present(i),
            "optional output {i} is unsupported"
        );
    }
    Ok(())
}

/// MultiHeadAttention (com.microsoft). Separate 3D Q/K/V + optional projection bias.
fn multihead_attention_claim(node: &NodeView) -> ClaimResult {
    let h = node.int_attr("num_heads", 0);
    require!(h > 0, "num_heads must be positive, got {}", h);
    let qd = match check_qkv_float(node) {
        Some(qd) => qd,
        None => deny!("Q, K, V, and output 0 must be present and share one MLX float dtype"),
    };
    let qshape = match node.input_info(0) {
        Some(i) => i.shape,
        None => deny!("Q lacks tensor type/shape info"),
    };
    let kshape = match node.input_info(1) {
        Some(i) => i.shape,
        None => deny!("K lacks tensor type/shape info"),
    };
    let vshape = match node.input_info(2) {
        Some(i) => i.shape,
        None => deny!("V lacks tensor type/shape info"),
    };
    require!(
        qshape.len() == 3
            && matches!(kshape.len(), 3 | 4)
            && matches!(vshape.len(), 3 | 4)
            && kshape.len() == vshape.len(),
        "Q must be rank 3 and K/V matching rank 3 or 4, got ranks {}, {}, {}",
        qshape.len(),
        kshape.len(),
        vshape.len()
    );
    require!(
        qshape[2] > 0,
        "Q hidden dimension must be static and positive"
    );
    let dk = if kshape.len() == 3 {
        kshape[2]
    } else {
        require!(
            kshape[1] == h && kshape[3] > 0,
            "rank-4 K must be [B,num_heads,L,head_size], got {:?}",
            kshape
        );
        h * kshape[3]
    };
    let dv = if vshape.len() == 3 {
        vshape[2]
    } else {
        require!(
            vshape[1] == h && vshape[3] > 0,
            "rank-4 V must be [B,num_heads,L,head_size], got {:?}",
            vshape
        );
        h * vshape[3]
    };
    require!(
        qshape[2] % h == 0 && dk % h == 0 && dv % h == 0,
        "Q/K/V hidden dimensions ({}/{}/{}) must divide evenly by num_heads {}",
        qshape[2],
        dk,
        dv,
        h
    );
    if node.int_attr("unidirectional", 0) != 0 && !node.input_present(6) {
        let k_len = if kshape.len() == 3 {
            kshape[1]
        } else {
            kshape[2]
        };
        let v_len = if vshape.len() == 3 {
            vshape[1]
        } else {
            vshape[2]
        };
        require!(
            qshape[1] > 0 && qshape[1] == k_len && qshape[1] == v_len,
            "unidirectional MHA without past requires equal static Q/K/V sequence lengths"
        );
    }
    if node.input_present(3) {
        let (bd, bshape) = match node.input_info(3) {
            Some(i) => (i.dtype, i.shape),
            None => deny!("projection bias lacks tensor type/shape info"),
        };
        require!(
            bd == qd && bshape.len() == 1 && bshape[0] == qshape[2] + dk + dv,
            "projection bias must be a 1D {} tensor of length {}, got {} shape {:?}",
            crate::registry::ort_dtype_name(qd),
            qshape[2] + dk + dv,
            crate::registry::ort_dtype_name(bd),
            bshape
        );
    }
    for i in [4usize, 5, 8, 9] {
        require!(
            !node.input_present(i),
            "optional input {} is unsupported",
            i
        );
    }
    require!(
        node.input_present(6) == node.input_present(7),
        "past key/value must be paired"
    );
    if node.input_present(6) {
        let (pk, pv) = match (node.input_info(6), node.input_info(7)) {
            (Some(pk), Some(pv)) => (pk, pv),
            _ => deny!("past key/value lack tensor type/shape info"),
        };
        require!(
            pk.dtype == qd
                && pv.dtype == qd
                && pk.shape.len() == 4
                && pv.shape.len() == 4
                && pk.shape[1] == h
                && pv.shape[1] == h
                && pk.shape[3] == dk / h
                && pv.shape[3] == dv / h,
            "past key/value must be rank-4 cache tensors matching Q/K/V dtype and heads"
        );
    }
    require!(
        node.output_present(1) == node.output_present(2),
        "present key/value outputs must be paired"
    );
    require!(
        !(node.output_present(3) && node.int_attr("unidirectional", 0) != 0),
        "qk output with unidirectional attention is unsupported"
    );
    Ok(())
}

/// PackedMultiHeadAttention (com.microsoft). Separate packed Q/K/V without projection or attention
/// bias. Runtime cumulative sequence lengths partition the token-major rows into independent
/// sequences; output remains in the same packed order.
fn packed_multihead_attention_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() >= 6 && node.num_outputs() >= 1,
        "expects at least 6 inputs and 1 output"
    );
    let (q, k, v, token_offset, cumulative, output) = match (
        node.input_info(0),
        node.input_info(1),
        node.input_info(2),
        node.input_info(4),
        node.input_info(5),
        node.output_info(0),
    ) {
        (Some(q), Some(k), Some(v), Some(t), Some(c), Some(o)) => (q, k, v, t, c, o),
        _ => deny!("Q/K/V, packing metadata, or output lacks tensor type/shape info"),
    };
    require!(
        is_mlx_float(q.dtype)
            && q.dtype == k.dtype
            && k.dtype == v.dtype
            && v.dtype == output.dtype,
        "Q/K/V and output must share one MLX float dtype"
    );
    require!(
        q.shape.len() == 2 && k.shape == q.shape && v.shape.len() == 2,
        "separate Q/K/V must be token-major rank-2 tensors"
    );
    require!(
        q.shape[1] > 0 && v.shape[1] > 0,
        "Q/K/V hidden dimensions must be static and positive"
    );
    let num_heads = node.int_attr("num_heads", 0);
    require!(
        num_heads > 0 && q.shape[1] % num_heads == 0 && v.shape[1] % num_heads == 0,
        "Q/K/V hidden dimensions must divide num_heads"
    );
    require!(
        token_offset.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
            && token_offset.shape.len() == 2,
        "token_offset must be rank-2 int32"
    );
    require!(
        cumulative.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
            && cumulative.shape.len() == 1,
        "cumulative_sequence_length must be rank-1 int32"
    );
    require!(
        !node.input_present(3),
        "QKV projection bias is not supported"
    );
    require!(
        node.num_inputs() < 7 || !node.input_present(6),
        "attention_bias is not supported"
    );
    Ok(())
}

/// RotaryEmbedding (ai.onnx opset 23 / com.microsoft). Float 3D (B,S,H*hd)+num_heads or 4D input;
/// [B,S] gather, [1] offset (com.microsoft), or (ai.onnx only) absent pos with [B,S,half] cache.
fn rotary_embedding_claim(node: &NodeView) -> ClaimResult {
    require!(node.num_outputs() > 0, "requires at least 1 output");
    let ms = node.domain() == "com.microsoft";
    let ci = if ms { 2 } else { 1 };
    let si = if ms { 3 } else { 2 };
    let pi = if ms { 1 } else { 3 };
    let min_inputs = if ms { 4 } else { 3 };
    require!(
        node.num_inputs() >= min_inputs,
        "expects at least {} inputs, got {}",
        min_inputs,
        node.num_inputs()
    );
    let (xd, xshape) = match node.input_info(0) {
        Some(i) => (i.dtype, i.shape),
        None => deny!("input lacks tensor type/shape info"),
    };
    let (cd, cshape) = match node.input_info(ci) {
        Some(i) => (i.dtype, i.shape),
        None => deny!("cos cache lacks tensor type/shape info"),
    };
    require!(node.input_present(si), "sin cache input is required");
    let sd = match dtype_of(node, si) {
        Some(t) => t,
        None => deny!("sin cache lacks tensor type/shape info"),
    };
    let od = match node.output_info(0) {
        Some(o) => o.dtype,
        None => deny!("output lacks tensor type/shape info"),
    };
    require!(
        is_mlx_float(xd) && cd == xd && sd == xd && od == xd,
        "input, cos cache, sin cache, and output must share one MLX float dtype, got {}, {}, {}, {}",
        crate::registry::ort_dtype_name(xd),
        crate::registry::ort_dtype_name(cd),
        crate::registry::ort_dtype_name(sd),
        crate::registry::ort_dtype_name(od)
    );
    let rank = xshape.len();
    if rank == 3 {
        let nh = node.int_attr("num_heads", 0);
        require!(
            ms || nh > 0,
            "standard rank-3 RotaryEmbedding requires num_heads"
        );
        if nh > 0 {
            require!(
                xshape[2] > 0 && xshape[2] % nh == 0,
                "rank-3 input hidden dimension {} must divide evenly by positive num_heads {}",
                xshape[2],
                nh
            );
        } else {
            // num_heads=0 (Gemma3n form): infer head size from the cos cache. Only the full-head
            // rotary case (rotary_embedding_dim=0) lets us derive head_size = 2 * cos_last_dim.
            let red = node.int_attr("rotary_embedding_dim", 0);
            require!(
                red == 0,
                "rank-3 RotaryEmbedding with num_heads=0 needs rotary_embedding_dim=0 (full-head \
                 rotary) to infer the head size; got rotary_embedding_dim={red}"
            );
            let half = cshape.last().copied().unwrap_or(0);
            let head = 2 * half;
            require!(
                head > 0 && xshape[2] > 0 && xshape[2] % head == 0,
                "rank-3 input hidden dimension {} must divide evenly by the inferred head size {} \
                 (2 × cos-cache last dim); num_heads=0",
                xshape[2],
                head
            );
        }
    } else {
        require!(rank == 4, "input must have rank 3 or 4, got rank {}", rank);
    }
    require!(
        node.input_info(si).is_some_and(|s| s.shape == cshape),
        "cos/sin cache shapes must match"
    );
    let half = cshape.last().copied().unwrap_or(0);
    let nh = node.int_attr("num_heads", 0);
    let head = if rank == 4 {
        xshape[3]
    } else if nh > 0 {
        xshape[2] / nh
    } else {
        2 * half
    };
    let rotary_dim = node.int_attr("rotary_embedding_dim", 0);
    require!(
        half > 0
            && (head < 0 || (head % 2 == 0 && 2 * half <= head))
            && (rotary_dim == 0 || rotary_dim == 2 * half),
        "rotary cache width must match an even rotary dimension within the head"
    );
    let has_pos = node.input_present(pi);
    require!(
        !ms || has_pos,
        "com.microsoft RotaryEmbedding requires position_ids"
    );
    if has_pos {
        let (pd, pshape) = match node.input_info(pi) {
            Some(i) => (i.dtype, i.shape),
            None => deny!("position_ids lacks tensor type/shape info"),
        };
        require!(
            pd == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                || (ms && pd == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
            "position_ids must have int32 or int64 dtype, got {}",
            crate::registry::ort_dtype_name(pd)
        );
        let gather = pshape.len() == 2;
        let offset = ms && pshape.len() == 1 && pshape[0] == 1;
        require!(
            gather || offset,
            "position_ids must be [B,S] or, for com.microsoft, [1], got shape {:?}",
            pshape
        );
        require!(
            cshape.len() == 2,
            "gather/offset cos cache must have rank 2, got shape {:?}",
            cshape
        );
        if gather {
            let seq_axis = if rank == 4 { 2 } else { 1 };
            require!(
                (pshape[0] < 0 || xshape[0] < 0 || pshape[0] == xshape[0])
                    && (pshape[1] < 0 || xshape[seq_axis] < 0 || pshape[1] == xshape[seq_axis]),
                "position_ids must match input batch and sequence dimensions"
            );
        }
    } else {
        require!(
            cshape.len() == 3,
            "absent-position cos cache must have rank 3, got shape {:?}",
            cshape
        );
        let seq_axis = if rank == 4 { 2 } else { 1 };
        require!(
            (cshape[0] < 0 || xshape[0] < 0 || cshape[0] == xshape[0])
                && (cshape[1] < 0 || xshape[seq_axis] < 0 || cshape[1] == xshape[seq_axis]),
            "per-position caches must match input batch and sequence dimensions"
        );
    }
    Ok(())
}

fn mrotary_embedding_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 4 && node.num_outputs() == 1,
        "MRotaryEmbedding expects 4 inputs and 1 output"
    );
    let (x, pos, cos, sin, out) = match (
        node.input_info(0),
        node.input_info(1),
        node.input_info(2),
        node.input_info(3),
        node.output_info(0),
    ) {
        (Some(x), Some(pos), Some(cos), Some(sin), Some(out)) => (x, pos, cos, sin, out),
        _ => deny!("MRotaryEmbedding requires typed input/output tensors"),
    };
    require!(
        is_mlx_float(x.dtype)
            && cos.dtype == x.dtype
            && sin.dtype == x.dtype
            && out.dtype == x.dtype,
        "input, caches, and output must share one MLX float dtype"
    );
    require!(
        (x.shape.len() == 3 || x.shape.len() == 4)
            && x.shape.iter().all(|&d| d > 0)
            && out.shape == x.shape,
        "input/output must have equal static positive rank-3/rank-4 shapes"
    );
    let (b, s, hd) = if x.shape.len() == 4 {
        (x.shape[0], x.shape[2], x.shape[3])
    } else {
        let nh = node.int_attr("num_heads", 0);
        require!(
            nh > 0 && x.shape[2] % nh == 0,
            "rank-3 input requires positive num_heads dividing hidden_size"
        );
        (x.shape[0], x.shape[1], x.shape[2] / nh)
    };
    require!(
        pos.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
            && pos.shape == [3, b, s],
        "position_ids must be int64 [3,B,S]"
    );
    require!(
        cos.shape.len() == 2 && sin.shape == cos.shape && cos.shape[1] > 0,
        "cos/sin caches must have equal static rank-2 shapes"
    );
    let rot = node.int_attr("rotary_embedding_dim", 0);
    require!(
        rot == 0 || node.int_attr("num_heads", 0) > 0,
        "num_heads must be positive when rotary_embedding_dim is specified"
    );
    let effective_rot = if rot == 0 { hd } else { rot };
    require!(
        effective_rot > 0
            && effective_rot % 2 == 0
            && effective_rot <= hd
            && cos.shape[1] == effective_rot / 2,
        "rotary dimension and cache width are inconsistent with head_size"
    );
    let (present, sections) = node.ints_attr("mrope_section");
    require!(
        present
            && sections.len() == 3
            && sections.iter().all(|&v| v >= 0)
            && sections.iter().sum::<i64>() == effective_rot / 2,
        "mrope_section must contain 3 non-negative values summing to rotary_dim/2"
    );
    require!(
        matches!(node.int_attr("mrope_layout", 0), 0 | 1),
        "mrope_layout must be 0 or 1"
    );
    Ok(())
}

fn shapeless(
    registry: &mut OpRegistry,
    domain: &'static str,
    op_type: &'static str,
    min_opset: i32,
    max_opset: i32,
    handler: crate::registry::OpHandler,
    claim: crate::registry::ClaimPredicate,
) {
    registry.register_shapeless(OpRegistration {
        domain,
        op_type,
        min_opset,
        max_opset,
        handler,
        claim,
    });
}

#[allow(clippy::too_many_arguments)]
fn shape_keyed(
    registry: &mut OpRegistry,
    domain: &'static str,
    op_type: &'static str,
    min_opset: i32,
    max_opset: i32,
    handler: crate::registry::OpHandler,
    claim: crate::registry::ClaimPredicate,
    reason: &'static str,
) {
    registry.register_shape_keyed(
        OpRegistration {
            domain,
            op_type,
            min_opset,
            max_opset,
            handler,
            claim,
        },
        reason,
    );
}

// ---- PagedAttention (com.microsoft) ------------------------------------------------------------
// Block-paged KV-cache attention with packed var-length batches (vLLM-style continuous batching).
// ORT ships this CUDA-only (no CPU kernel), so the MLX EP claiming it is the only way such a model
// runs on Apple Silicon. This is an EAGER, data-dependent op: the packing boundaries
// (cumulative_sequence_length), per-sequence past lengths (past_seqlens) and the physical block map
// (block_table) are runtime INT32 tensors, read host-side each forward. For each sequence we gather
// its cached K/V from the scattered physical blocks, append the new (optionally RoPE'd) K/V, run a
// GQA SDPA in fp32, and pack the outputs back. Causality is optional; RoPE uses exact cache rows.
// SEPARATE packed QKV, softcap, and left windows compose on this eager path.

/// Read an integer input tensor (int32 or int64) to a host `Vec<i64>`, normalizing dtype through
/// MLX (astype + eval) — unlike `read_ints`, which reinterprets the raw bytes as i64 and so mangles
/// an int32 tensor. PagedAttention's control tensors (cum_seqlens/past_seqlens/block_table) are int32.
fn read_int_input(
    ctx: &mut TranslationContext,
    r: &crate::engine::TensorRef,
) -> Result<Vec<i64>, MlxError> {
    let a = ctx.resolve(r)?;
    let a = ctx.astype(a, mlx::mlx_dtype__MLX_INT64)?;
    let a = ctx.contiguous_eval(a)?;
    let sh = ctx.shape_of(a);
    let n: usize = sh.iter().map(|&d| d as usize).product();
    if n == 0 {
        return Ok(Vec::new());
    }
    let p = unsafe { mlx::mlx_array_data_int64(a) };
    if p.is_null() {
        return Err("Attention: evaluated integer metadata has no host data".to_string());
    }
    Ok(unsafe { std::slice::from_raw_parts(p, n) }.to_vec())
}

/// Build a 1-D int32 MLX array from concrete host indices (physical block ids for a `take`).
fn i32_host_array(ctx: &mut TranslationContext, data: &[i32]) -> mlx::mlx_array {
    ctx.keep(Array::from_data(
        data.as_ptr() as *const c_void,
        &[data.len() as i32],
        mlx::mlx_dtype__MLX_INT32,
    ))
}

/// Repeat each of `kv` heads `g` times along the head axis of a `[1, kv, s, hd]` tensor, yielding
/// `[1, kv*g, s, hd]` — the GQA broadcast (output head e reads kv head e/g), matching np.repeat.
fn gqa_repeat_heads(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    kv: i32,
    g: i32,
    s: i32,
    hd: i32,
) -> Result<mlx::mlx_array, MlxError> {
    if g == 1 {
        return Ok(x);
    }
    let r = ctx.reshape(x, &[1, kv, 1, s, hd])?;
    let bshape = [1, kv, g, s, hd];
    let b = ctx.emit(|res, st| unsafe {
        mlx::mlx_broadcast_to(res, r, bshape.as_ptr(), bshape.len(), st)
    })?;
    ctx.reshape(b, &[1, kv * g, s, hd])
}

#[allow(clippy::too_many_arguments)]
fn paged_sdpa(
    ctx: &mut TranslationContext,
    q: mlx::mlx_array,
    k: mlx::mlx_array,
    v: mlx::mlx_array,
    scale: f32,
    mask: mlx::mlx_array,
    head_sink: Option<mlx::mlx_array>,
    softcap: f32,
) -> Result<mlx::mlx_array, MlxError> {
    if head_sink.is_none() && softcap == 0.0 {
        return sdpa(ctx, q, k, v, scale, b"array\0", mask);
    }
    let scores = capped_attention_logits(ctx, q, k, scale, softcap)?;
    let scores = ctx.add(scores, mask)?;
    let probs = attention_probs_with_sink(ctx, scores, head_sink)?;
    ctx.matmul(probs, v)
}

#[allow(clippy::too_many_arguments)]
fn rope_with_offset(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    head_size: i32,
    rotary_dim: i32,
    rotary_offset: i32,
    interleaved: bool,
    position_offset: i32,
    cos: mlx::mlx_array,
    sin: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    let seq = ctx.shape_of(x)[2];
    let half = rotary_dim / 2;
    let cos = cos_sin_row(ctx, cos, position_offset, seq, half)?;
    let sin = cos_sin_row(ctx, sin, position_offset, seq, half)?;
    if rotary_offset == 0 {
        return gqa_rope(ctx, x, cos, sin, half, interleaved);
    }
    let shape = ctx.shape_of(x);
    let prefix = slice(
        ctx,
        x,
        &[0, 0, 0, 0],
        &[shape[0], shape[1], shape[2], rotary_offset],
    )?;
    let middle = slice(
        ctx,
        x,
        &[0, 0, 0, rotary_offset],
        &[shape[0], shape[1], shape[2], rotary_offset + rotary_dim],
    )?;
    let middle = gqa_rope(ctx, middle, cos, sin, half, interleaved)?;
    let mut out = concat2(ctx, prefix, middle, 3)?;
    if rotary_offset + rotary_dim < head_size {
        let suffix = slice(
            ctx,
            x,
            &[0, 0, 0, rotary_offset + rotary_dim],
            &[shape[0], shape[1], shape[2], head_size],
        )?;
        out = concat2(ctx, out, suffix, 3)?;
    }
    Ok(out)
}

fn paged_attention_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let num_heads = attr_int(n, "num_heads", 0) as i32;
    let kv_heads = attr_int(n, "kv_num_heads", 0) as i32;
    let do_rotary = attr_int(n, "do_rotary", 0) != 0;
    let interleaved = attr_int(n, "rotary_interleaved", 0) != 0;
    let latent = n
        .strings
        .get("kv_cache_layout")
        .map(String::as_str)
        .unwrap_or("SEPARATE")
        == "LATENT";
    if num_heads <= 0 || kv_heads <= 0 || num_heads % kv_heads != 0 {
        return Err("PagedAttention: bad num_heads/kv_num_heads".to_string());
    }
    let f32t = mlx::mlx_dtype__MLX_FLOAT32;

    let input_q = ctx.resolve(&n.inputs[0])?;
    let packed = !present(n, 1) && !present(n, 2);
    let input_shape = ctx.shape_of(input_q);
    let packed_head = input_shape[1] / (num_heads + 2 * kv_heads);
    let (q, key, value) = if packed {
        let q_end = num_heads * packed_head;
        let k_end = q_end + kv_heads * packed_head;
        (
            slice(ctx, input_q, &[0, 0], &[input_shape[0], q_end])?,
            slice(ctx, input_q, &[0, q_end], &[input_shape[0], k_end])?,
            Some(slice(ctx, input_q, &[0, k_end], &input_shape)?),
        )
    } else {
        (
            input_q,
            ctx.resolve(&n.inputs[1])?,
            if latent {
                None
            } else {
                Some(ctx.resolve(&n.inputs[2])?)
            },
        )
    };
    let out_dt = ctx.dtype_of(q);
    let qs = ctx.shape_of(q); // [num_tokens, hidden]
    let hidden = qs[1];
    let head = hidden / num_heads;
    let v_head = match attr_int(n, "v_head_size", 0) as i32 {
        0 => head,
        value => value,
    };
    let kv_hidden = kv_heads * head;
    let v_hidden = kv_heads * v_head;
    let out_hidden = num_heads * v_head;
    let scale = attr_scale(n, head);
    let g = num_heads / kv_heads;

    let mut kcache = ctx.resolve(&n.inputs[3])?;
    let mut vcache = if latent {
        None
    } else {
        Some(ctx.resolve(&n.inputs[4])?)
    };
    let cshape = ctx.shape_of(kcache); // [num_blocks, block_size, kv, head]
    let vcshape = vcache.map(|cache| ctx.shape_of(cache));
    let block_size = cshape[1];
    if cshape[0] <= 0 || block_size <= 0 || vcshape.as_ref().is_some_and(|s| s[..3] != cshape[..3])
    {
        return Err("PagedAttention: caches require matching positive block geometry".to_string());
    }

    // Data-dependent control tensors, read host-side (this op is forced onto the eager path).
    let cum = read_int_input(ctx, &n.inputs[5])?; // [batch+1]
    let past = read_int_input(ctx, &n.inputs[6])?; // [batch]
    let bt = read_int_input(ctx, &n.inputs[7])?; // [batch, max_blocks] flattened
    if cum.len() < 2 {
        return Err(
            "PagedAttention: cumulative_sequence_length must have length batch+1".to_string(),
        );
    }
    let batch = cum.len() - 1;
    let block_table = ctx.resolve(&n.inputs[7])?;
    let bt_shape = ctx.shape_of(block_table);
    if past.len() != batch
        || bt_shape.len() != 2
        || bt_shape[0] != batch as i32
        || cum[0] != 0
        || cum[batch] != qs[0] as i64
        || cum.windows(2).any(|v| v[0] < 0 || v[0] > v[1])
        || past.iter().any(|&v| v < 0 || v > i32::MAX as i64)
    {
        return Err(
            "PagedAttention: inconsistent cumulative lengths, past lengths or block table"
                .to_string(),
        );
    }
    let max_blocks = bt.len() / batch;
    let slot_mapping = if present(n, 10) {
        Some(read_int_input(ctx, &n.inputs[10])?)
    } else {
        None
    };
    if slot_mapping.as_ref().is_some_and(|slots| {
        slots.len() != qs[0] as usize
            || slots
                .iter()
                .any(|&slot| slot < -1 || slot >= cshape[0] as i64 * block_size as i64)
    }) {
        return Err(
            "PagedAttention: slot_mapping must contain -1 or valid physical cache slots"
                .to_string(),
        );
    }
    for b in 0..batch {
        let total = past[b] + cum[b + 1] - cum[b];
        if total > i32::MAX as i64 || total > max_blocks as i64 * block_size as i64 {
            return Err("PagedAttention: sequence exceeds block table capacity".to_string());
        }
        let needed = (total + block_size as i64 - 1) / block_size as i64;
        if bt[b * max_blocks..b * max_blocks + needed as usize]
            .iter()
            .any(|&block| block < 0 || block >= cshape[0] as i64)
        {
            return Err(
                "PagedAttention: block table contains an invalid physical block".to_string(),
            );
        }
    }
    let head_sink = if present(n, 11) {
        Some(ctx.resolve(&n.inputs[11])?)
    } else {
        None
    };
    let q_norm = if present(n, 12) {
        Some((
            ctx.resolve(&n.inputs[12])?,
            ctx.resolve(&n.inputs[13])?,
            n.floats.get("qk_norm_epsilon").copied().unwrap_or(1e-6),
        ))
    } else {
        None
    };
    let mut cache_updated = false;

    // The cache is [max_seq, rotary_dim/2]; consume its exact values, including reduced precision.
    let rope = if do_rotary {
        let cos = ctx.resolve(&n.inputs[8])?;
        let sin = ctx.resolve(&n.inputs[9])?;
        let half = ctx.shape_of(cos)[1];
        Some((cos, sin, 2 * half))
    } else {
        None
    };

    let mut outs: Vec<mlx::mlx_array> = Vec::new();
    for b in 0..batch {
        let s0 = cum[b] as i32;
        let s1 = cum[b + 1] as i32;
        let nq = s1 - s0;
        if nq <= 0 {
            continue;
        }
        let pastb = past[b] as i32;
        let total = pastb + nq;

        // Slice this sequence's packed rows and split into [1, heads, nq, head].
        let qb = slice(ctx, q, &[s0, 0], &[s1, hidden])?;
        let qb = ctx.reshape(qb, &[1, nq, hidden])?;
        let mut qh = split_heads(ctx, qb, 1, nq, num_heads, head)?; // [1,H,nq,head]
        let kb = slice(ctx, key, &[s0, 0], &[s1, kv_hidden])?;
        let kb = ctx.reshape(kb, &[1, nq, kv_hidden])?;
        let mut kh_new = split_heads(ctx, kb, 1, nq, kv_heads, head)?; // [1,kv,nq,head]
        let separate_vh_new = if let Some(value) = value {
            let vb = slice(ctx, value, &[s0, 0], &[s1, v_hidden])?;
            let vb = ctx.reshape(vb, &[1, nq, v_hidden])?;
            Some(split_heads(ctx, vb, 1, nq, kv_heads, v_head)?)
        } else {
            None
        };

        if let Some((qw, kw, eps)) = q_norm {
            qh = ctx.emit(|res, st| unsafe { mlx::mlx_fast_rms_norm(res, qh, qw, eps, st) })?;
            kh_new =
                ctx.emit(|res, st| unsafe { mlx::mlx_fast_rms_norm(res, kh_new, kw, eps, st) })?;
        }

        // RoPE the new Q and K at absolute positions [past, total).
        if let Some((cos, sin, rot)) = rope {
            let offset = attr_int(n, "rotary_offset", 0) as i32;
            qh = rope_with_offset(ctx, qh, head, rot, offset, interleaved, pastb, cos, sin)?;
            kh_new =
                rope_with_offset(ctx, kh_new, head, rot, offset, interleaved, pastb, cos, sin)?;
        }
        let vh_new = if latent {
            slice(ctx, kh_new, &[0, 0, 0, 0], &[1, kv_heads, nq, v_head])?
        } else {
            separate_vh_new.unwrap()
        };

        let mut write_slots = Vec::with_capacity(nq as usize);
        let mut write_rows = Vec::with_capacity(nq as usize);
        for j in 0..nq as usize {
            let token = s0 as usize + j;
            let slot = if let Some(ref mapping) = slot_mapping {
                mapping[token]
            } else {
                let logical = pastb as usize + j;
                let physical_block = bt[b * max_blocks + logical / block_size as usize];
                physical_block * block_size as i64 + (logical % block_size as usize) as i64
            };
            if slot >= 0 {
                write_slots.push(slot as i32);
                write_rows.push(j as i32);
            }
        }
        if !write_slots.is_empty() {
            let row_idx = i32_host_array(ctx, &write_rows);
            let slot_idx = i32_host_array(ctx, &write_slots);
            let knew = ctx.transpose(kh_new, &[0, 2, 1, 3])?;
            let knew = ctx.reshape(knew, &[nq, kv_heads, head])?;
            let knew =
                ctx.emit(|res, st| unsafe { mlx::mlx_take_axis(res, knew, row_idx, 0, st) })?;
            let knew = ctx.reshape(knew, &[write_slots.len() as i32, 1, kv_heads, head])?;
            let flat_slots = cshape[0] * block_size;
            let kc = ctx.reshape(kcache, &[flat_slots, kv_heads, head])?;
            let kc = ctx.emit(|res, st| unsafe {
                mlx::mlx_scatter_single(res, kc, slot_idx, knew, 0, st)
            })?;
            kcache = ctx.reshape(kc, &cshape)?;
            if let Some(current_vcache) = vcache {
                let vnew = ctx.transpose(vh_new, &[0, 2, 1, 3])?;
                let vnew = ctx.reshape(vnew, &[nq, kv_heads, v_head])?;
                let vnew =
                    ctx.emit(|res, st| unsafe { mlx::mlx_take_axis(res, vnew, row_idx, 0, st) })?;
                let vnew = ctx.reshape(vnew, &[write_slots.len() as i32, 1, kv_heads, v_head])?;
                let vc = ctx.reshape(current_vcache, &[flat_slots, kv_heads, v_head])?;
                let vc = ctx.emit(|res, st| unsafe {
                    mlx::mlx_scatter_single(res, vc, slot_idx, vnew, 0, st)
                })?;
                vcache = Some(ctx.reshape(vc, vcshape.as_ref().unwrap())?);
            }
            cache_updated = true;
        }

        // Gather the cached prefix [0, past) from the physical blocks and append the new K/V.
        let (kh, vh) = if pastb > 0 {
            let nblk = ((pastb + block_size - 1) / block_size) as usize;
            let idx: Vec<i32> = (0..nblk).map(|j| bt[b * max_blocks + j] as i32).collect();
            let idx_arr = i32_host_array(ctx, &idx);
            let gk =
                ctx.emit(|res, s| unsafe { mlx::mlx_take_axis(res, kcache, idx_arr, 0, s) })?;
            let gk = ctx.reshape(gk, &[1, (nblk as i32) * block_size, kv_hidden])?;
            let gk = split_heads(ctx, gk, 1, (nblk as i32) * block_size, kv_heads, head)?; // [1,kv,N,head]
            let gk = slice(ctx, gk, &[0, 0, 0, 0], &[1, kv_heads, pastb, head])?;
            let gv = if latent {
                slice(ctx, gk, &[0, 0, 0, 0], &[1, kv_heads, pastb, v_head])?
            } else {
                let gv = ctx.emit(|res, s| unsafe {
                    mlx::mlx_take_axis(res, vcache.unwrap(), idx_arr, 0, s)
                })?;
                let gv = ctx.reshape(gv, &[1, (nblk as i32) * block_size, v_hidden])?;
                let gv = split_heads(ctx, gv, 1, (nblk as i32) * block_size, kv_heads, v_head)?;
                slice(ctx, gv, &[0, 0, 0, 0], &[1, kv_heads, pastb, v_head])?
            };
            (concat2(ctx, gk, kh_new, 2)?, concat2(ctx, gv, vh_new, 2)?) // [1,kv,total,head]
        } else {
            (kh_new, vh_new)
        };

        // GQA broadcast to full head count, then causal SDPA (query i attends keys [0, past+i]).
        let kh = gqa_repeat_heads(ctx, kh, kv_heads, g, total, head)?; // [1,H,total,head]
        let vh = gqa_repeat_heads(ctx, vh, kv_heads, g, total, v_head)?;
        let qh = ctx.astype(qh, f32t)?;
        let kh = ctx.astype(kh, f32t)?;
        let vh = ctx.astype(vh, f32t)?;
        let mut mask = if attr_int(n, "is_causal", 1) != 0 {
            causal_mask_topleft(ctx, nq, total, pastb, f32t)?
        } else {
            ctx.zeros(&[nq, total], f32t)?
        };
        let window = attr_int(n, "local_window_size", -1) as i32;
        if window > 0 {
            let key_pos = ctx.arange(0.0, total as f64, 1.0, mlx::mlx_dtype__MLX_INT32)?;
            let key_pos = ctx.reshape(key_pos, &[1, total])?;
            let left = ctx.arange(
                (pastb - window + 1) as f64,
                (pastb + nq - window + 1) as f64,
                1.0,
                mlx::mlx_dtype__MLX_INT32,
            )?;
            let left = ctx.reshape(left, &[nq, 1])?;
            let allowed = ctx.less_equal(left, key_pos)?;
            let neg = ctx.scalar_f32(f32::NEG_INFINITY);
            mask = ctx.where_(allowed, mask, neg)?;
        }
        let mask = ctx.reshape(mask, &[1, 1, nq, total])?;
        let o = paged_sdpa(
            ctx,
            qh,
            kh,
            vh,
            scale,
            mask,
            head_sink,
            n.floats.get("softcap").copied().unwrap_or(0.0),
        )?; // [1,H,nq,head]
        let o = ctx.transpose(o, &[0, 2, 1, 3])?; // [1,nq,H,head]
        let o = ctx.reshape(o, &[nq, out_hidden])?;
        outs.push(ctx.astype(o, out_dt)?);
    }

    let output = match outs.len() {
        0 => {
            // No tokens (all sequences empty): bind an empty output tensor.
            ctx.zeros(&[0, out_hidden], out_dt)?
        }
        1 => outs[0],
        _ => {
            let mut vec = VectorArray::new();
            for o in &outs {
                vec.append(*o);
            }
            ctx.emit(|res, s| unsafe { mlx::mlx_concatenate_axis(res, vec.as_raw(), 0, s) })?
        }
    };
    if cache_updated {
        kcache = ctx.write_back_input(&n.inputs[3], kcache)?;
        if let Some(current_vcache) = vcache {
            vcache = Some(ctx.write_back_input(&n.inputs[4], current_vcache)?);
        }
    }
    ctx.bind(&n.outputs[0], output);
    if n.outputs.len() > 1 && !n.outputs[1].name.is_empty() {
        ctx.bind(&n.outputs[1], kcache);
    }
    if n.outputs.len() > 2 && !n.outputs[2].name.is_empty() {
        if let Some(vcache) = vcache {
            ctx.bind(&n.outputs[2], vcache);
        }
    }
    Ok(())
}

fn paged_attention_claim(node: &NodeView) -> ClaimResult {
    require!(
        matches!(node.int_attr("is_causal", 1), 0 | 1),
        "is_causal must be 0 or 1"
    );
    require!(
        node.num_outputs() >= 1,
        "PagedAttention requires at least 1 output"
    );
    require!(
        node.int_attr("v_head_size", 0) >= 0,
        "v_head_size must be non-negative"
    );
    require!(
        node.int_attr("rotary_offset", 0) >= 0,
        "rotary_offset must be non-negative"
    );
    for attr in ["k_cache_dtype", "v_cache_dtype"] {
        let dtype = node.string_attr(attr, "");
        require!(
            !matches!(dtype.as_str(), "int4" | "float4e2m1"),
            "{attr}={dtype:?} sub-byte packing is unsupported"
        );
    }
    // T = fp16/bf16 only; output dtype must match the query.
    let out_dt = match node.output_info(0) {
        Some(o) if is_mlx_float(o.dtype) => o.dtype,
        Some(o) => deny!(
            "output must be an MLX float dtype, got {}",
            crate::registry::ort_dtype_name(o.dtype)
        ),
        None => deny!("output lacks tensor type/shape info"),
    };
    require!(
        out_dt == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
            || out_dt == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
        "PagedAttention is only supported for fp16/bf16 (T)"
    );
    let ninputs = node.num_inputs();
    require!(
        (8..=17).contains(&ninputs),
        "expects 8 required inputs plus optional rotary, scheduling, normalization, and scale inputs, got {}",
        ninputs
    );
    let layout = node.string_attr("kv_cache_layout", "SEPARATE");
    require!(
        matches!(layout.as_str(), "SEPARATE" | "LATENT"),
        "kv_cache_layout must be SEPARATE or LATENT"
    );
    let latent = layout == "LATENT";
    let packed = !latent && !node.input_present(1) && !node.input_present(2);
    for idx in if packed {
        vec![0usize, 3]
    } else {
        vec![0usize, 1, 3]
    } {
        match dtype_of(node, idx) {
            Some(dt) => require!(
                dt == out_dt,
                "input {} must share the output float dtype",
                idx
            ),
            None => deny!(
                "input {} (Q/K/V/cache) must be present with a float dtype",
                idx
            ),
        }
    }
    if latent {
        require!(
            !node.input_present(2) && !node.input_present(4),
            "LATENT layout aliases value to key and requires absent value/value_cache"
        );
    } else {
        for idx in if packed {
            vec![4usize]
        } else {
            vec![2usize, 4]
        } {
            require!(
                dtype_of(node, idx) == Some(out_dt),
                "input {idx} must be present and share the output float dtype"
            );
        }
    }
    // Positional inputs are int32.
    for idx in [5usize, 6, 7] {
        match dtype_of(node, idx) {
            Some(dt) => require!(is_int32(dt), "input {} must be int32", idx),
            None => deny!("input {} (positional) must be present", idx),
        }
    }
    let num_heads = node.int_attr("num_heads", 0);
    let kv_heads = node.int_attr("kv_num_heads", 0);
    require!(
        num_heads > 0 && kv_heads > 0 && num_heads % kv_heads == 0,
        "num_heads ({num_heads}) must be a positive multiple of kv_num_heads ({kv_heads})"
    );
    require!(
        !latent || kv_heads == 1,
        "LATENT layout requires kv_num_heads=1"
    );
    let query = node.input_info(0).unwrap();
    let query_heads = if packed {
        num_heads + 2 * kv_heads
    } else {
        num_heads
    };
    require!(
        query.shape.len() == 2 && query.shape[1] > 0 && query.shape[1] % query_heads == 0,
        "query must have rank 2 and a hidden size divisible by its separate or packed head count"
    );
    let head = query.shape[1] / query_heads;
    let v_head = match node.int_attr("v_head_size", 0) {
        0 => head,
        value => value,
    };
    require!(v_head > 0, "effective v_head_size must be positive");
    if !packed {
        require!(
            node.input_info(1)
                .is_some_and(|v| v.shape.len() == 2 && v.shape[1] == kv_heads * head),
            "key must be [num_tokens,kv_num_heads*head_size]"
        );
        if !latent {
            require!(
                node.input_info(2)
                    .is_some_and(|v| v.shape.len() == 2 && v.shape[1] == kv_heads * v_head),
                "value must be [num_tokens,kv_num_heads*v_head_size]"
            );
        }
    }
    if !latent {
        require!(
            v_head == head,
            "SEPARATE layout requires equal key and value head sizes"
        );
    } else {
        require!(
            v_head <= head,
            "LATENT v_head_size must not exceed head_size"
        );
        if v_head != head {
            require!(
                node.has_attr("scale"),
                "LATENT layout with asymmetric value heads requires an explicit scale"
            );
        }
    }
    require!(
        node.input_info(3).is_some_and(|v| {
            v.shape.len() == 4 && v.shape[2] == kv_heads && v.shape[3] == head
        }),
        "key_cache must be [num_blocks,block_size,kv_num_heads,head_size]"
    );
    if !latent {
        require!(
            node.input_info(4).is_some_and(|v| {
                v.shape.len() == 4 && v.shape[2] == kv_heads && v.shape[3] == v_head
            }),
            "value_cache must be [num_blocks,block_size,kv_num_heads,v_head_size]"
        );
    }
    require!(
        node.output_info(0).is_some_and(|v| {
            v.shape.len() == 2 && (v.shape[1] < 0 || v.shape[1] == num_heads * v_head)
        }),
        "output must be [num_tokens,num_heads*v_head_size]"
    );
    require!(
        node.int_attr("local_window_size", -1) == -1 || node.int_attr("local_window_size", -1) > 0,
        "local_window_size must be -1 or positive"
    );
    require!(
        node.float_attr("softcap", 0.0).is_finite() && node.float_attr("softcap", 0.0) >= 0.0,
        "softcap must be finite and non-negative"
    );
    // do_rotary=1 needs the cos/sin caches present (inputs 8, 9).
    if node.int_attr("do_rotary", 0) != 0 {
        require!(
            node.input_present(8) && node.input_present(9),
            "do_rotary=1 requires cos_cache and sin_cache inputs"
        );
        match (dtype_of(node, 8), dtype_of(node, 9)) {
            (Some(c), Some(s)) => require!(
                c == out_dt && s == out_dt,
                "cos_cache/sin_cache must share the output float dtype"
            ),
            _ => deny!("do_rotary=1 requires typed cos_cache/sin_cache"),
        }
        let cos = node.input_info(8).unwrap();
        let sin = node.input_info(9).unwrap();
        require!(
            cos.shape.len() == 2 && cos.shape == sin.shape && cos.shape[1] > 0,
            "cos/sin caches must have matching [positions,rotary_dim/2] shapes"
        );
        let rotary_dim = cos.shape[1] * 2;
        require!(
            node.int_attr("rotary_offset", 0) + rotary_dim <= head,
            "rotary_offset + rotary_dim must not exceed head_size"
        );
        require!(
            node.int_attr("rotary_offset", 0) % 8 == 0,
            "rotary_offset must be a multiple of 8"
        );
    }
    require!(
        node.string_attr("k_quant_type", "NONE") == "NONE"
            && node.string_attr("v_quant_type", "NONE") == "NONE"
            && !node.input_present(14)
            && !node.input_present(15),
        "quantized KV cache is unsupported"
    );
    if node.input_present(10) {
        require!(
            node.input_info(10).is_some_and(|v| is_int32(v.dtype)
                && v.shape.len() == 1
                && (v.shape[0] < 0 || query.shape[0] < 0 || v.shape[0] == query.shape[0])),
            "slot_mapping must be int32 [num_tokens]"
        );
    }
    if node.input_present(11) {
        require!(
            node.input_info(11)
                .is_some_and(|v| v.dtype == out_dt && v.shape == [num_heads]),
            "head_sink must be [num_heads] in activation dtype"
        );
    }
    require!(
        node.input_present(12) == node.input_present(13),
        "q_norm_weight and k_norm_weight must be provided together"
    );
    if node.input_present(12) {
        for idx in [12usize, 13] {
            require!(
                node.input_info(idx)
                    .is_some_and(|v| v.dtype == out_dt && v.shape == [head]),
                "Q/K norm weight {idx} must be [head_size] in activation dtype"
            );
        }
    }
    if node.input_present(16) {
        require!(
            node.input_info(16)
                .is_some_and(|v| is_int32(v.dtype) && (v.shape == [2] || v.shape == [3])),
            "attention_metadata must be int32 [2] or [3]"
        );
    }
    if latent {
        require!(
            !node.output_present(2),
            "LATENT layout has no separate value cache output"
        );
    }
    for idx in 1..node.num_outputs().min(3) {
        if node.output_present(idx) {
            require!(
                node.output_info(idx).is_some_and(|v| v.dtype == out_dt),
                "cache output {idx} must share activation dtype"
            );
        }
    }
    Ok(())
}

pub fn register_attention(registry: &mut OpRegistry) {
    shape_keyed(
        registry,
        "com.microsoft",
        "PagedAttention",
        K_ANY_OPSET,
        K_ANY_OPSET,
        paged_attention_op,
        paged_attention_claim,
        "emits MLX Slice and Scatter, which lack Primitive::output_shapes in MLX 0.32.2",
    );
    // Non-sliding, no-bias GQA uses the slice-free compiled decode route. Sliding-window and
    // attention-bias forms emit Slice and remain eager; unsupported forms are conservative too.
    registry.register_shape_classified(
        OpRegistration {
            domain: "com.microsoft",
            op_type: "GroupQueryAttention",
            min_opset: K_ANY_OPSET,
            max_opset: K_ANY_OPSET,
            handler: group_query_attention_op,
            claim: group_query_attention_claim,
        },
        group_query_attention_compile_shape_safety,
    );
    // Attention entered ai.onnx at opset 23; v24 adds nonpad lengths and v25 adds logits,
    // windows, softcap, and softmax precision. Legacy SDPA remains shapeless; composed forms are
    // deliberately eager so dynamic decode never captures a first-call shape.
    registry.register_shape_classified(
        OpRegistration {
            domain: "",
            op_type: "Attention",
            min_opset: 23,
            max_opset: K_ANY_OPSET,
            handler: attention_op,
            claim: attention_claim,
        },
        attention_compile_shape_safety,
    );
    shape_keyed(
        registry,
        "com.microsoft",
        "Attention",
        K_ANY_OPSET,
        K_ANY_OPSET,
        ms_attention_op,
        ms_attention_claim,
        crate::registry::MLX_SLICE_SHAPE_REASON,
    );
    shapeless(
        registry,
        "com.microsoft",
        "MultiHeadAttention",
        K_ANY_OPSET,
        K_ANY_OPSET,
        multihead_attention_op,
        multihead_attention_claim,
    );
    shape_keyed(
        registry,
        "com.microsoft",
        "PackedMultiHeadAttention",
        K_ANY_OPSET,
        K_ANY_OPSET,
        packed_multihead_attention_op,
        packed_multihead_attention_claim,
        crate::registry::MLX_SLICE_SHAPE_REASON,
    );
    // RotaryEmbedding: ai.onnx entered at opset 23; com.microsoft is version-insensitive.
    shape_keyed(
        registry,
        "",
        "RotaryEmbedding",
        23,
        K_ANY_OPSET,
        rotary_embedding_op,
        rotary_embedding_claim,
        crate::registry::MLX_SLICE_SHAPE_REASON,
    );
    shape_keyed(
        registry,
        "com.microsoft",
        "RotaryEmbedding",
        K_ANY_OPSET,
        K_ANY_OPSET,
        rotary_embedding_op,
        rotary_embedding_claim,
        crate::registry::MLX_SLICE_SHAPE_REASON,
    );
    shape_keyed(
        registry,
        "com.microsoft",
        "MRotaryEmbedding",
        K_ANY_OPSET,
        K_ANY_OPSET,
        mrotary_embedding_op,
        mrotary_embedding_claim,
        crate::registry::MLX_SLICE_SHAPE_REASON,
    );
}

#[cfg(test)]
mod tests {
    use super::{
        ATTENTION_COMPOSED_COMPILE_REASON, GQA_BIAS_COMPILE_REASON, GQA_SLIDING_COMPILE_REASON,
        GQA_UNSUPPORTED_COMPILE_REASON, attention_compile_shape_safety_for,
        attention_mask_dimension_is_compatible, group_query_attention_compile_shape_safety_for,
        mlx_softmax_precision,
    };
    use crate::registry::CompileShapeSafety;
    use crate::sys::mlx;

    #[test]
    fn group_query_attention_compile_shape_safety_matrix() {
        assert_eq!(
            group_query_attention_compile_shape_safety_for(true, false, false),
            CompileShapeSafety::Shapeless,
            "claimable slice-free 7-input GQA must remain shapeless"
        );
        assert_eq!(
            group_query_attention_compile_shape_safety_for(true, false, false),
            CompileShapeSafety::Shapeless,
            "claimable slice-free 9-input GQA must remain shapeless"
        );
        assert_eq!(
            group_query_attention_compile_shape_safety_for(true, false, true),
            CompileShapeSafety::EagerOnly {
                reason: GQA_BIAS_COMPILE_REASON
            },
            "11-input attention-bias GQA must never regain shapeless classification"
        );
        assert_eq!(
            group_query_attention_compile_shape_safety_for(true, true, false),
            CompileShapeSafety::EagerOnly {
                reason: GQA_SLIDING_COMPILE_REASON
            },
            "sliding-window GQA must never regain shapeless classification"
        );
        assert_eq!(
            group_query_attention_compile_shape_safety_for(false, false, false),
            CompileShapeSafety::EagerOnly {
                reason: GQA_UNSUPPORTED_COMPILE_REASON
            },
            "invalid or unsupported GQA forms must never be classified shapeless"
        );
    }

    #[test]
    fn attention_v25_composed_forms_stay_eager_and_legacy_decode_stays_shapeless() {
        assert_eq!(
            attention_compile_shape_safety_for(true, false),
            CompileShapeSafety::Shapeless,
            "the pre-v25 fast-SDPA form must retain dynamic decode replay"
        );
        assert_eq!(
            attention_compile_shape_safety_for(true, true),
            CompileShapeSafety::EagerOnly {
                reason: ATTENTION_COMPOSED_COMPILE_REASON
            },
            "nonpad, window, softcap, QK-output, or mixed-T1/T2 forms must not capture a first-call shape"
        );
    }

    #[test]
    fn attention_v25_softmax_precision_rejects_double_without_claiming_it() {
        assert_eq!(
            mlx_softmax_precision(Some(1)),
            Some(mlx::mlx_dtype__MLX_FLOAT32)
        );
        assert_eq!(
            mlx_softmax_precision(Some(10)),
            Some(mlx::mlx_dtype__MLX_FLOAT16)
        );
        assert_eq!(
            mlx_softmax_precision(Some(16)),
            Some(mlx::mlx_dtype__MLX_BFLOAT16)
        );
        assert_eq!(
            mlx_softmax_precision(Some(11)),
            None,
            "MLX EP has no GPU fp64 Attention path, so the v25 DOUBLE request must be declined"
        );
    }

    #[test]
    fn attention_v25_short_key_mask_is_accepted_but_other_axes_must_broadcast() {
        // ORT 1.29 cannot execute the v25 schema, so this directly pins the schema-only claim
        // rule rather than pretending the v24 kernel implements the newer short-mask contract.
        assert!(attention_mask_dimension_is_compatible(3, 4, true));
        assert!(!attention_mask_dimension_is_compatible(3, 4, false));
        assert!(!attention_mask_dimension_is_compatible(5, 4, true));
    }
}
