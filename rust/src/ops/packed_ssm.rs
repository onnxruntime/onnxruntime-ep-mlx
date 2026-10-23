//! Packed state-space operators. Only sequence boundaries are read on the host; activation,
//! convolution, gate, and recurrent-state arithmetic stays in MLX.

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::registry::{
    ClaimResult, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry, is_mlx_float,
};
use crate::sys::{mlx, ort};
use crate::{deny, require};

const F32: mlx::mlx_dtype = mlx::mlx_dtype__MLX_FLOAT32;
const T_F32: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
const T_I32: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;

fn present(n: &NodeDesc, index: usize) -> bool {
    n.inputs
        .get(index)
        .is_some_and(|input| input.source != Src::Absent && !input.name.is_empty())
}

fn compatible(actual: &[i64], expected: &[i64]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(&a, &b)| a < 0 || b < 0 || a == b)
}

fn input_matches(
    node: &NodeView,
    index: usize,
    dtype: ort::ONNXTensorElementDataType,
    shape: &[i64],
) -> bool {
    node.input_info(index)
        .is_some_and(|info| info.dtype == dtype && compatible(&info.shape, shape))
}

fn output_matches(
    node: &NodeView,
    index: usize,
    dtype: ort::ONNXTensorElementDataType,
    shape: &[i64],
) -> bool {
    node.output_info(index)
        .is_some_and(|info| info.dtype == dtype && compatible(&info.shape, shape))
}

fn validate_ranges(
    offsets: &[i32],
    total: i32,
    allow_empty: bool,
) -> Result<Vec<(i32, i32)>, MlxError> {
    if offsets.len() < 2
        || offsets.first() != Some(&0)
        || offsets.last() != Some(&total)
        || offsets.windows(2).any(|pair| {
            pair[0] < 0
                || pair[1] > total
                || pair[0] > pair[1]
                || (!allow_empty && pair[0] == pair[1])
        })
    {
        return Err(
            "packed SSM: cumulative_sequence_length must partition all tokens in sequence order"
                .to_string(),
        );
    }
    Ok(offsets.windows(2).map(|pair| (pair[0], pair[1])).collect())
}

fn packed_ranges(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    index: usize,
    total: i32,
    allow_empty: bool,
) -> Result<Vec<(i32, i32)>, MlxError> {
    let offsets = ctx.resolve(&n.inputs[index])?;
    if ctx.shape_of(offsets).len() != 1 {
        return Err("packed SSM: cumulative_sequence_length must be rank 1".to_string());
    }
    let offsets = ctx.contiguous_eval(offsets)?;
    let count = ctx.size_of(offsets);
    let data = unsafe { mlx::mlx_array_data_int32(offsets) };
    if count < 2 || data.is_null() {
        return Err(
            "packed SSM: cumulative_sequence_length needs at least two offsets".to_string(),
        );
    }
    let values = unsafe { std::slice::from_raw_parts(data, count) };
    validate_ranges(values, total, allow_empty)
}

fn concat(
    ctx: &mut TranslationContext,
    arrays: &[mlx::mlx_array],
) -> Result<mlx::mlx_array, MlxError> {
    let mut result = *arrays
        .first()
        .ok_or("packed SSM: cannot assemble an empty batch")?;
    for &array in &arrays[1..] {
        result = ctx.concat2(result, array, 0)?;
    }
    ctx.contiguous(result)
}

fn varlen_causal_conv_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let weight = ctx.resolve(&n.inputs[1])?;
    let initial = ctx.resolve(&n.inputs[4])?;
    let xs = ctx.shape_of(x);
    let ws = ctx.shape_of(weight);
    let dilation = n.ints.get("dilation").copied().unwrap_or(1) as i32;
    if xs.len() != 2 || ws.len() != 3 || xs[0] <= 0 || xs[1] <= 0 || ws[2] <= 0 {
        return Err("VarlenCausalConvWithState: invalid input or weight shape".to_string());
    }
    let (total, channels, kernel) = (xs[0], xs[1], ws[2]);
    let pad = (kernel - 1)
        .checked_mul(dilation)
        .ok_or("VarlenCausalConvWithState: state length overflow")?;
    let ranges = packed_ranges(ctx, n, 2, total, false)?;
    let batch = ranges.len() as i32;
    if ws != [channels, 1, kernel] || ctx.shape_of(initial) != [batch, channels, pad] {
        return Err(
            "VarlenCausalConvWithState: weight or initial_state shape mismatch".to_string(),
        );
    }
    let dtype = ctx.dtype_of(x);
    let weight = ctx.astype(weight, F32)?;
    let weight = ctx.transpose(weight, &[0, 2, 1])?;
    let weight = ctx.contiguous(weight)?;
    let bias = if present(n, 3) {
        let bias = ctx.resolve(&n.inputs[3])?;
        if ctx.shape_of(bias) != [channels] {
            return Err("VarlenCausalConvWithState: bias must match channels".to_string());
        }
        Some(ctx.astype(bias, F32)?)
    } else {
        None
    };
    let activation = n
        .strings
        .get("activation")
        .map(String::as_str)
        .unwrap_or("none");
    let mut outputs = Vec::with_capacity(ranges.len());
    let mut states = Vec::with_capacity(ranges.len());
    for (sequence, &(start, end)) in ranges.iter().enumerate() {
        let length = end - start;
        let tokens = ctx.slice(x, &[start, 0], &[end, channels])?;
        let tokens = ctx.reshape(tokens, &[1, length, channels])?;
        let state = ctx.slice(
            initial,
            &[sequence as i32, 0, 0],
            &[sequence as i32 + 1, channels, pad],
        )?;
        let state = ctx.transpose(state, &[0, 2, 1])?;
        let raw = ctx.concat2(state, tokens, 1)?;
        let tail = ctx.slice(raw, &[0, length, 0], &[1, length + pad, channels])?;
        states.push(ctx.transpose(tail, &[0, 2, 1])?);

        let input = ctx.astype(raw, F32)?;
        let input = ctx.contiguous(input)?;
        let mut output = ctx.emit(|res, stream| unsafe {
            mlx::mlx_conv1d(res, input, weight, 1, 0, dilation, channels, stream)
        })?;
        if let Some(bias) = bias {
            output = ctx.add(output, bias)?;
        }
        if activation == "silu" || activation == "swish" {
            let sigmoid = ctx.unary(mlx::mlx_sigmoid, output)?;
            output = ctx.mul(output, sigmoid)?;
        }
        let output = ctx.astype(output, dtype)?;
        outputs.push(ctx.reshape(output, &[length, channels])?);
    }
    let output = concat(ctx, &outputs)?;
    let state = concat(ctx, &states)?;
    ctx.bind(&n.outputs[0], output);
    ctx.bind(&n.outputs[1], state);
    if n.outputs.get(2).is_some_and(|out| !out.name.is_empty()) {
        let empty = ctx.zeros(&[batch, 0, channels], dtype)?;
        ctx.bind(&n.outputs[2], empty);
    }
    Ok(())
}

fn varlen_causal_conv_claim(node: &NodeView) -> ClaimResult {
    require!(
        (5..=6).contains(&node.num_inputs()) && (2..=3).contains(&node.num_outputs()),
        "VarlenCausalConvWithState expects 5 inputs and 2 outputs (optional capture slots allowed)"
    );
    require!(
        node.int_attr("state_update_capacity", 0) == 0 && !node.input_present(5),
        "VarlenCausalConvWithState compact state capture is not supported"
    );
    let (x, weight, cu) = match (node.input_info(0), node.input_info(1), node.input_info(2)) {
        (Some(x), Some(weight), Some(cu)) => (x, weight, cu),
        _ => deny!("VarlenCausalConvWithState requires typed input, weight, and offsets"),
    };
    require!(
        is_mlx_float(x.dtype)
            && x.shape.len() == 2
            && x.shape[0] != 0
            && x.shape[1] > 0
            && weight.dtype == x.dtype
            && weight.shape.len() == 3
            && weight.shape[0] == x.shape[1]
            && weight.shape[1] == 1
            && weight.shape[2] > 0
            && weight.shape.iter().all(|&dim| dim <= i32::MAX as i64),
        "VarlenCausalConvWithState requires float [tokens,C] and static [C,1,K] weights"
    );
    require!(
        cu.dtype == T_I32 && cu.shape.len() == 1 && (cu.shape[0] < 0 || cu.shape[0] >= 2),
        "cumulative_sequence_length must be int32 [batch+1]"
    );
    let dilation = node.int_attr("dilation", 1);
    let pad = (weight.shape[2] - 1).checked_mul(dilation);
    require!(
        dilation > 0
            && dilation <= i32::MAX as i64
            && pad.is_some_and(|pad| (0..=i32::MAX as i64).contains(&pad)),
        "dilation and convolution state length must fit int32"
    );
    let batch = if cu.shape[0] < 0 { -1 } else { cu.shape[0] - 1 };
    let channels = x.shape[1];
    let state_shape = [batch, channels, pad.unwrap()];
    require!(
        input_matches(node, 4, x.dtype, &state_shape)
            && output_matches(node, 0, x.dtype, &x.shape)
            && output_matches(node, 1, x.dtype, &state_shape),
        "VarlenCausalConvWithState input/output state shapes or dtypes do not match"
    );
    require!(
        !node.input_present(3) || input_matches(node, 3, x.dtype, &[channels]),
        "bias must match input dtype and channels"
    );
    require!(
        !node.output_present(2) || output_matches(node, 2, x.dtype, &[batch, 0, channels]),
        "without capture, state_update must be [batch,0,C]"
    );
    require!(
        matches!(
            node.string_attr("activation", "none").as_str(),
            "none" | "silu" | "swish"
        ),
        "VarlenCausalConvWithState activation must be none, silu, or swish"
    );
    Ok(())
}

fn l2_normalize(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    let squared = ctx.mul(x, x)?;
    let norm =
        ctx.emit(|res, stream| unsafe { mlx::mlx_sum_axis(res, squared, -1, true, stream) })?;
    let epsilon = ctx.scalar_f32(1e-12);
    let norm = ctx.add(norm, epsilon)?;
    let inverse = ctx.unary(mlx::mlx_rsqrt, norm)?;
    ctx.mul(x, inverse)
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_decode(
    ctx: &mut TranslationContext,
    q: mlx::mlx_array,
    k: mlx::mlx_array,
    v: mlx::mlx_array,
    g: mlx::mlx_array,
    beta: mlx::mlx_array,
    state: mlx::mlx_array,
) -> Result<(mlx::mlx_array, mlx::mlx_array), MlxError> {
    let g = ctx.expand_dims(g, -1)?;
    let g = ctx.unary(mlx::mlx_exp, g)?;
    let state = ctx.mul(state, g)?;
    let retrieved = ctx.matmul(k, state)?;
    let delta = ctx.sub(v, retrieved)?;
    let beta = ctx.expand_dims(beta, -1)?;
    let delta = ctx.mul(delta, beta)?;
    let kt = ctx.transpose(k, &[0, 1, 3, 2])?;
    let update = ctx.matmul(kt, delta)?;
    let state = ctx.add(state, update)?;
    let output = ctx.matmul(q, state)?;
    let output = ctx.transpose(output, &[0, 2, 1, 3])?;
    Ok((output, state))
}

fn gated_delta_net_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let query = ctx.resolve(&n.inputs[0])?;
    let key = ctx.resolve(&n.inputs[1])?;
    let value = ctx.resolve(&n.inputs[2])?;
    let qs = ctx.shape_of(query);
    let ks = ctx.shape_of(key);
    let vs = ctx.shape_of(value);
    let rank = qs.len();
    if !(rank == 3 || rank == 4)
        || ks.len() != rank
        || vs.len() != rank
        || qs.iter().any(|&dim| dim <= 0)
        || ks[..rank - 2] != qs[..rank - 2]
        || vs[..rank - 2] != qs[..rank - 2]
    {
        return Err("GatedDeltaNet: Q/K/V leading shapes must match and be positive".to_string());
    }
    let total = qs[..rank - 2]
        .iter()
        .try_fold(1i32, |n, &dim| n.checked_mul(dim))
        .ok_or("GatedDeltaNet: token count overflow")?;
    let (hq, dk, hv, dv) = (qs[rank - 2], qs[rank - 1], vs[rank - 2], vs[rank - 1]);
    if ks[rank - 2..] != [hq, dk] || hv <= 0 || dv <= 0 || hv % hq != 0 {
        return Err("GatedDeltaNet: invalid Q/K/V head geometry".to_string());
    }
    let initial = if present(n, 6) {
        Some(ctx.resolve(&n.inputs[6])?)
    } else {
        None
    };
    let ranges = if present(n, 3) {
        packed_ranges(ctx, n, 3, total, true)?
    } else {
        let batch = if rank == 4 {
            qs[0]
        } else {
            let state =
                initial.ok_or("GatedDeltaNet: uniform rank-3 packing requires initial_state")?;
            *ctx.shape_of(state)
                .first()
                .ok_or("GatedDeltaNet: invalid initial_state")?
        };
        if batch <= 0 || total % batch != 0 {
            return Err("GatedDeltaNet: uniform batch must divide total_tokens".to_string());
        }
        let length = total / batch;
        (0..batch).map(|b| (b * length, (b + 1) * length)).collect()
    };
    let batch = ranges.len() as i32;
    if initial.is_some_and(|state| ctx.shape_of(state) != [batch, hv, dv, dk]) {
        return Err("GatedDeltaNet: initial_state must be [batch,Hv,V,K]".to_string());
    }
    let dtype = ctx.dtype_of(query);
    let query = ctx.astype(query, F32)?;
    let key = ctx.astype(key, F32)?;
    let value = ctx.astype(value, F32)?;
    let mut query = ctx.reshape(query, &[total, hq, dk])?;
    let mut key = ctx.reshape(key, &[total, hq, dk])?;
    let value = ctx.reshape(value, &[total, hv, dv])?;
    if n.ints.get("qk_l2_norm").copied().unwrap_or(0) != 0 {
        query = l2_normalize(ctx, query)?;
        key = l2_normalize(ctx, key)?;
    }
    let scale = match n.floats.get("scale").copied().unwrap_or(0.0) {
        0.0 => 1.0 / (dk as f32).sqrt(),
        scale => scale,
    };
    let scale = ctx.scalar_f32(scale);
    query = ctx.mul(query, scale)?;
    query =
        ctx.emit(|res, stream| unsafe { mlx::mlx_repeat_axis(res, query, hv / hq, 1, stream) })?;
    key = ctx.emit(|res, stream| unsafe { mlx::mlx_repeat_axis(res, key, hv / hq, 1, stream) })?;
    let mut gates = ctx.resolve(&n.inputs[4])?;
    let mut beta = ctx.resolve(&n.inputs[5])?;
    let mut gate_shape = qs[..rank - 2].to_vec();
    gate_shape.push(hv);
    if ctx.shape_of(gates) != gate_shape || ctx.shape_of(beta) != gate_shape {
        return Err(
            "GatedDeltaNet: decay and beta must match token axes and value heads".to_string(),
        );
    }
    gates = ctx.reshape(gates, &[total, hv])?;
    beta = ctx.reshape(beta, &[total, hv])?;
    if n.strings
        .get("gate_activation")
        .is_some_and(|value| value == "qwen")
    {
        let a_log = ctx.resolve(&n.inputs[7])?;
        let bias = ctx.resolve(&n.inputs[8])?;
        if ctx.shape_of(a_log) != [hv] || ctx.shape_of(bias) != [hv] {
            return Err("GatedDeltaNet: a_log and dt_bias must be [Hv]".to_string());
        }
        let raw = ctx.add(gates, bias)?;
        let zero = ctx.scalar_f32(0.0);
        let softplus = ctx.binary(mlx::mlx_logaddexp, raw, zero)?;
        let a = ctx.unary(mlx::mlx_exp, a_log)?;
        let a = ctx.unary(mlx::mlx_negative, a)?;
        gates = ctx.mul(a, softplus)?;
    }
    if n.strings
        .get("beta_activation")
        .is_some_and(|value| value == "sigmoid")
    {
        beta = ctx.unary(mlx::mlx_sigmoid, beta)?;
    }
    let chunk_size = if n.ints.get("chunk_size").copied().unwrap_or(64) == 32 {
        32
    } else {
        64
    };
    ctx.mark_composed(
        "chunked gated-delta prefill with FP32 state and recurrent single-token decode",
    );
    let mut outputs = Vec::with_capacity(ranges.len());
    let mut states = Vec::with_capacity(ranges.len());
    for (sequence, &(start, end)) in ranges.iter().enumerate() {
        let length = end - start;
        // The ONNX boundary is V-major; the shared chunk core multiplies K-major state.
        let state = if let Some(initial) = initial {
            let state = ctx.slice(
                initial,
                &[sequence as i32, 0, 0, 0],
                &[sequence as i32 + 1, hv, dv, dk],
            )?;
            ctx.transpose(state, &[0, 1, 3, 2])?
        } else {
            ctx.zeros(&[1, hv, dk, dv], F32)?
        };
        if length == 0 {
            states.push(ctx.transpose(state, &[0, 1, 3, 2])?);
            continue;
        }
        let to_heads = |ctx: &mut TranslationContext, x, heads, width| {
            let x = ctx.slice(x, &[start, 0, 0], &[end, heads, width])?;
            let x = ctx.transpose(x, &[1, 0, 2])?;
            ctx.expand_dims(x, 0)
        };
        let q = to_heads(ctx, query, hv, dk)?;
        let k = to_heads(ctx, key, hv, dk)?;
        let v = to_heads(ctx, value, hv, dv)?;
        let g = ctx.slice(gates, &[start, 0], &[end, hv])?;
        let g = ctx.transpose(g, &[1, 0])?;
        let g = ctx.expand_dims(g, 0)?;
        let b = ctx.slice(beta, &[start, 0], &[end, hv])?;
        let b = ctx.transpose(b, &[1, 0])?;
        let b = ctx.expand_dims(b, 0)?;
        let (output, state) = if length == 1 {
            gated_delta_decode(ctx, q, k, v, g, b, state)?
        } else {
            super::ssm::linear_attention_chunked(
                ctx, q, k, v, g, b, state, 1, hv, dk, dv, length, F32, chunk_size,
            )?
        };
        let output = ctx.reshape(output, &[length, hv, dv])?;
        outputs.push(ctx.astype(output, dtype)?);
        states.push(ctx.transpose(state, &[0, 1, 3, 2])?);
    }
    let output = concat(ctx, &outputs)?;
    let mut output_shape = qs[..rank - 2].to_vec();
    output_shape.extend([hv, dv]);
    let output = ctx.reshape(output, &output_shape)?;
    ctx.bind(&n.outputs[0], output);
    if n.outputs.get(1).is_some_and(|out| !out.name.is_empty()) {
        let state = concat(ctx, &states)?;
        ctx.bind(&n.outputs[1], state);
    }
    if n.outputs.get(2).is_some_and(|out| !out.name.is_empty()) {
        let empty = ctx.zeros(&[batch, 0], F32)?;
        ctx.bind(&n.outputs[2], empty);
    }
    Ok(())
}

fn gated_delta_net_claim(node: &NodeView) -> ClaimResult {
    require!(
        (6..=11).contains(&node.num_inputs()) && (1..=3).contains(&node.num_outputs()),
        "GatedDeltaNet expects Q/K/V, optional offsets, decay, and beta"
    );
    require!(
        node.string_attr("update_rule", "gated_delta") == "gated_delta",
        "GatedDeltaNet currently supports update_rule=gated_delta only"
    );
    require!(
        node.int_attr("state_update_capacity", 0) == 0
            && !node.input_present(9)
            && !node.input_present(10),
        "GatedDeltaNet compact state capture is not supported"
    );
    require!(
        node.int_attr("chunk_size", 64) > 0 && node.float_attr("scale", 0.0).is_finite(),
        "GatedDeltaNet requires a positive chunk_size and finite scale"
    );
    let (q, k, v) = match (node.input_info(0), node.input_info(1), node.input_info(2)) {
        (Some(q), Some(k), Some(v)) => (q, k, v),
        _ => deny!("GatedDeltaNet requires typed Q/K/V"),
    };
    let rank = q.shape.len();
    require!(
        (rank == 3 || rank == 4)
            && k.shape.len() == rank
            && v.shape.len() == rank
            && is_mlx_float(q.dtype)
            && k.dtype == q.dtype
            && v.dtype == q.dtype
            && q.shape[..rank - 2].iter().all(|&dim| dim != 0)
            && compatible(&q.shape[..rank - 2], &k.shape[..rank - 2])
            && compatible(&q.shape[..rank - 2], &v.shape[..rank - 2]),
        "GatedDeltaNet requires matching float rank-3/rank-4 token-major Q/K/V"
    );
    let (hq, dk, hv, dv) = (
        q.shape[rank - 2],
        q.shape[rank - 1],
        v.shape[rank - 2],
        v.shape[rank - 1],
    );
    require!(
        hq > 0
            && dk > 0
            && hv > 0
            && dv > 0
            && hv % hq == 0
            && k.shape[rank - 2..] == [hq, dk]
            && [hq, dk, hv, dv].iter().all(|&dim| dim <= i32::MAX as i64),
        "GatedDeltaNet requires static Q/K heads and Hv a positive multiple of Hq"
    );
    let mut gate_shape = q.shape[..rank - 2].to_vec();
    gate_shape.push(hv);
    require!(
        input_matches(node, 4, T_F32, &gate_shape) && input_matches(node, 5, T_F32, &gate_shape),
        "GatedDeltaNet supports FP32 per-head scalar decay and beta only"
    );
    let gate = node.string_attr("gate_activation", "none");
    require!(
        gate == "none" || gate == "qwen",
        "GatedDeltaNet gate_activation must be none or qwen"
    );
    require!(
        matches!(
            node.string_attr("beta_activation", "none").as_str(),
            "none" | "sigmoid"
        ),
        "GatedDeltaNet beta_activation must be none or sigmoid"
    );
    if gate == "qwen" {
        require!(
            input_matches(node, 7, T_F32, &[hv]) && input_matches(node, 8, T_F32, &[hv]),
            "Qwen gate activation requires FP32 a_log and dt_bias [Hv]"
        );
    } else {
        require!(
            !node.input_present(7) && !node.input_present(8),
            "a_log and dt_bias require gate_activation=qwen"
        );
    }
    let batch = if node.input_present(3) {
        let cu = match node.input_info(3) {
            Some(cu) => cu,
            None => deny!("GatedDeltaNet requires typed sequence offsets"),
        };
        require!(
            rank == 3
                && cu.dtype == T_I32
                && cu.shape.len() == 1
                && (cu.shape[0] < 0 || cu.shape[0] >= 2),
            "ragged GatedDeltaNet requires rank-3 Q/K/V and int32 [batch+1] offsets"
        );
        if cu.shape[0] < 0 { -1 } else { cu.shape[0] - 1 }
    } else if rank == 4 {
        q.shape[0]
    } else {
        let state = match node.input_info(6) {
            Some(state) if state.shape.len() == 4 => state,
            _ => deny!("uniform rank-3 GatedDeltaNet requires initial_state"),
        };
        state.shape[0]
    };
    let state_shape = [batch, hv, dv, dk];
    let mut output_shape = q.shape[..rank - 2].to_vec();
    output_shape.extend([hv, dv]);
    require!(
        (!node.input_present(6) || input_matches(node, 6, T_F32, &state_shape))
            && output_matches(node, 0, q.dtype, &output_shape)
            && (!node.output_present(1) || output_matches(node, 1, T_F32, &state_shape))
            && (!node.output_present(2) || output_matches(node, 2, T_F32, &[batch, 0])),
        "GatedDeltaNet output or FP32 V-major recurrent state geometry is incompatible"
    );
    Ok(())
}

pub fn register(registry: &mut OpRegistry) {
    for (op_type, handler, claim) in [
        (
            "VarlenCausalConvWithState",
            varlen_causal_conv_op as crate::registry::OpHandler,
            varlen_causal_conv_claim as crate::registry::ClaimPredicate,
        ),
        (
            "GatedDeltaNet",
            gated_delta_net_op as crate::registry::OpHandler,
            gated_delta_net_claim as crate::registry::ClaimPredicate,
        ),
    ] {
        registry.register_eager(
            OpRegistration {
                domain: "com.microsoft",
                op_type,
                min_opset: K_ANY_OPSET,
                max_opset: K_ANY_OPSET,
                handler,
                claim,
            },
            "reads packed sequence boundaries on the host and validates runtime state geometry",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::validate_ranges;

    #[test]
    fn packed_ranges_reject_incomplete_or_out_of_order_offsets() {
        for offsets in [vec![1, 3], vec![0, 2], vec![0, 3, 2, 3], vec![0, -1, 3]] {
            assert!(validate_ranges(&offsets, 3, true).is_err());
        }
        assert!(validate_ranges(&[0, 0, 3], 3, false).is_err());
        assert_eq!(
            validate_ranges(&[0, 0, 3], 3, true).unwrap(),
            [(0, 0), (0, 3)]
        );
    }
}
