//! Device-resident packed SSM kernels. Metadata stays on the GPU across decode calls.
//! Contracts follow ORT CUDA's `gated_delta_net_impl.cu` and
//! `varlen_causal_conv_with_state_impl.cu`; compact transition capture is not supported.

use std::ffi::CString;

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::registry::{
    ClaimResult, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry, is_mlx_float,
};
use crate::sys::{mlx, ort};
use crate::{deny, require};

const FLOAT: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
const INT32: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;

fn compatible(actual: &[i64], expected: &[i64]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(&a, &b)| a < 0 || b < 0 || a == b)
}

const CONV: &str = r#"
    uint c = thread_position_in_grid.x % C;
    uint b = thread_position_in_grid.x / C;
    if (b >= B) return;
    int start = offsets[b], end = offsets[b + 1];
    if (offsets[0] != 0 || offsets[B] != N ||
        start < 0 || end < start || end > N) return;
    for (int t = start; t < end; ++t) {
        float sum = BIAS ? float(bias[c]) : 0.0f;
        for (int j = 0; j < K; ++j) {
            int local = t - start - (K - 1 - j) * D;
            float x = local >= 0 ? float(input[(start + local) * C + c])
                                : float(initial[(b * C + c) * L + L + local]);
            sum += x * float(weight[c * K + j]);
        }
        if (SILU) sum = sum / (1.0f + precise::exp(-sum));
        output[t * C + c] = T(sum);
    }
    for (int j = 0; j < L; ++j) {
        int local = end - start - L + j;
        final[(b * C + c) * L + j] = local >= 0
            ? input[(start + local) * C + c]
            : initial[(b * C + c) * L + L + local];
    }
"#;

const DELTA: &str = r#"
    size_t column = thread_position_in_grid.x / 32;
    uint lane = thread_position_in_grid.x % 32;
    if (column >= size_t(B) * H * V) return;
    uint vi = column % V, h = (column / V) % H, b = column / (V * H);
    uint kh = h / (H / HQ);
    int start = offsets[b], end = offsets[b + 1];
    if (offsets[0] != 0 || offsets[B] != N ||
        start < 0 || end < start || end > N) return;
    float state[(K + 31) / 32];
    for (uint j = 0; j < (K + 31) / 32; ++j) {
        uint ki = lane + 32 * j;
        state[j] = ki < K ? initial[column * K + ki] : 0.0f;
    }
    for (int t = start; t < end; ++t) {
        float q[(K + 31) / 32], k[(K + 31) / 32];
        float nq = 0.0f, nk = 0.0f;
        for (uint j = 0; j < (K + 31) / 32; ++j) {
            uint ki = lane + 32 * j;
            q[j] = ki < K ? float(qkv[size_t(t) * WIDTH + kh * K + ki]) : 0.0f;
            k[j] = ki < K ? float(qkv[size_t(t) * WIDTH + HQ * K + kh * K + ki]) : 0.0f;
            nq += q[j] * q[j];
            nk += k[j] * k[j];
        }
        float rq = NORM ? rsqrt(simd_sum(nq) + 1e-12f) : 1.0f;
        float rk = NORM ? rsqrt(simd_sum(nk) + 1e-12f) : 1.0f;
        float g = 1.0f;
        if (DECAY) {
            float x = decay[t * H + h];
            if (SOFTPLUS) {
                x += dt_bias[h];
                x = -precise::exp(a_log[h]) *
                    (max(x, 0.0f) + precise::log(1.0f + precise::exp(-abs(x))));
            }
            g = precise::exp(x);
        }
        float retrieval = 0.0f;
        for (uint j = 0; j < (K + 31) / 32; ++j) {
            k[j] *= rk;
            state[j] *= g;
            retrieval += k[j] * state[j];
        }
        float delta = float(qkv[size_t(t) * WIDTH + 2 * HQ * K + h * V + vi]);
        if (RETRIEVE) delta -= simd_sum(retrieval);
        if (BETA) {
            float rate = beta[t * H + h];
            if (SIGMOID) rate = 1.0f / (1.0f + precise::exp(-rate));
            delta *= rate;
        }
        float readout = 0.0f;
        for (uint j = 0; j < (K + 31) / 32; ++j) {
            uint ki = lane + 32 * j;
            if (ki < K) {
                state[j] += k[j] * delta;
                readout += q[j] * rq * state[j];
            }
        }
        float result = simd_sum(readout);
        if (lane == 0) output[(size_t(t) * H + h) * V + vi] =
            T(result * as_type<float>(uint(SCALE_BITS)));
    }
    for (uint j = 0; j < (K + 31) / 32; ++j) {
        uint ki = lane + 32 * j;
        if (ki < K) final[column * K + ki] = state[j];
    }
"#;

fn present(n: &NodeDesc, index: usize) -> bool {
    n.inputs
        .get(index)
        .is_some_and(|input| input.source != Src::Absent && !input.name.is_empty())
}

#[allow(clippy::too_many_arguments)]
fn kernel(
    ctx: &mut TranslationContext,
    name: &str,
    source: &str,
    names: &[&str],
    arguments: &[mlx::mlx_array],
    ints: &[(&str, i32)],
    scale: Option<f32>,
    shapes: &[Vec<i32>],
    dtypes: &[mlx::mlx_dtype],
    threads: i32,
) -> Result<Vec<mlx::mlx_array>, MlxError> {
    let names = super::selective_scan::VecString::new(names)?;
    let outputs = super::selective_scan::VecString::new(&["output", "final"])?;
    let name = CString::new(name).unwrap();
    let source = CString::new(source).unwrap();
    let k = unsafe {
        mlx::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            names.as_raw(),
            outputs.as_raw(),
            source.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        )
    };
    if k.ctx.is_null() {
        return Err("Packed SSM: cannot construct Metal kernel; check MLX Metal support".into());
    }
    let config = unsafe { mlx::mlx_fast_metal_kernel_config_new() };
    let mut rc = unsafe {
        mlx::mlx_fast_metal_kernel_config_add_template_arg_dtype(config, c"T".as_ptr(), dtypes[0])
    };
    for &(key, value) in ints {
        let key = CString::new(key).unwrap();
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_template_arg_int(config, key.as_ptr(), value)
        };
    }
    if let Some(scale) = scale {
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_template_arg_int(
                config,
                c"SCALE_BITS".as_ptr(),
                scale.to_bits() as i32,
            )
        };
    }
    for (shape, &dtype) in shapes.iter().zip(dtypes) {
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_output_arg(
                config,
                shape.as_ptr(),
                shape.len(),
                dtype,
            )
        };
    }
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_grid(config, threads, 1, 1) };
    rc |= unsafe {
        mlx::mlx_fast_metal_kernel_config_set_thread_group(config, threads.min(128), 1, 1)
    };
    let mut args = crate::mlx::VectorArray::new();
    for &array in arguments {
        args.append(array);
    }
    let mut results = crate::mlx::VectorArray::new();
    if rc == 0 {
        rc = unsafe {
            mlx::mlx_fast_metal_kernel_apply(
                results.as_mut_ptr(),
                k,
                args.as_raw(),
                config,
                ctx.stream(),
            )
        };
    }
    unsafe {
        mlx::mlx_fast_metal_kernel_config_free(config);
        mlx::mlx_fast_metal_kernel_free(k);
    }
    if rc != 0 || results.size() != 2 {
        return Err(
            "Packed SSM: Metal dispatch failed; verify input geometry and GPU availability".into(),
        );
    }
    Ok((0..2).map(|i| ctx.keep(results.get(i))).collect())
}

fn conv(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let w = ctx.resolve(&n.inputs[1])?;
    let offsets = ctx.resolve(&n.inputs[2])?;
    let state = ctx.resolve(&n.inputs[4])?;
    let xs = ctx.shape_of(x);
    let ws = ctx.shape_of(w);
    let ss = ctx.shape_of(state);
    let dilation = n.ints.get("dilation").copied().unwrap_or(1) as i32;
    let length = (ws[2] - 1)
        .checked_mul(dilation)
        .ok_or("Packed convolution: receptive field overflow")?;
    if xs.len() != 2
        || ss.len() != 3
        || xs[1] != ws[0]
        || ss[1..] != [xs[1], length]
        || ctx.shape_of(offsets) != [ss[0] + 1]
    {
        return Err(
            "VarlenCausalConvWithState: input, state and offsets have incompatible shapes".into(),
        );
    }
    let bias = if present(n, 3) {
        ctx.resolve(&n.inputs[3])?
    } else {
        w
    };
    let activation = n
        .strings
        .get("activation")
        .map(String::as_str)
        .unwrap_or("none");
    let threads = ss[0]
        .checked_mul(xs[1])
        .filter(|&v| v > 0)
        .ok_or("VarlenCausalConvWithState: empty or oversized dispatch")?;
    let dtype = ctx.dtype_of(x);
    let out = kernel(
        ctx,
        "onnxrt_mlx_varlen_causal_conv",
        CONV,
        &["input", "weight", "offsets", "initial", "bias"],
        &[x, w, offsets, state, bias],
        &[
            ("B", ss[0]),
            ("N", xs[0]),
            ("C", xs[1]),
            ("K", ws[2]),
            ("L", length),
            ("D", dilation),
            ("BIAS", present(n, 3) as i32),
            (
                "SILU",
                (activation == "silu" || activation == "swish") as i32,
            ),
        ],
        None,
        &[xs, ss],
        &[dtype, dtype],
        threads,
    )?;
    ctx.bind(&n.outputs[0], out[0]);
    ctx.bind(&n.outputs[1], out[1]);
    ctx.mark_fast("mlx_varlen_causal_conv");
    Ok(())
}

fn delta(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let qkv = ctx.resolve(&n.inputs[0])?;
    let offsets = ctx.resolve(&n.inputs[3])?;
    let state = ctx.resolve(&n.inputs[6])?;
    let qs = ctx.shape_of(qkv);
    let ss = ctx.shape_of(state);
    if qs.len() != 2 || ss.len() != 4 || ctx.shape_of(offsets) != [ss[0] + 1] {
        return Err(
            "GatedDeltaNet: packed QKV, offsets and state have incompatible ranks/shapes".into(),
        );
    }
    let (b, h, v, k) = (ss[0], ss[1], ss[2], ss[3]);
    let qwidth = qs[1] - h * v;
    if b <= 0
        || h <= 0
        || v <= 0
        || k <= 0
        || k > 256
        || qwidth <= 0
        || qwidth % (2 * k) != 0
        || h % (qwidth / (2 * k)) != 0
    {
        return Err(
            "GatedDeltaNet: require positive packed head geometry with key width <= 256".into(),
        );
    }
    let hq = qwidth / (2 * k);
    let attr = |name: &str, default| n.strings.get(name).map(String::as_str).unwrap_or(default);
    let rule = attr("update_rule", "gated_delta");
    let decay = present(n, 4) && (rule == "gated" || rule == "gated_delta");
    let retrieve = rule == "delta" || rule == "gated_delta";
    let beta = present(n, 5);
    let softplus = matches!(attr("gate_activation", "none"), "softplus" | "qwen");
    let mut inputs = vec![qkv, offsets, state];
    for (index, needed, shape) in [
        (4, decay, vec![qs[0], h]),
        (5, beta, vec![qs[0], h]),
        (7, decay && softplus, vec![h]),
        (8, decay && softplus, vec![h]),
    ] {
        let array = if needed {
            let a = ctx.resolve(&n.inputs[index])?;
            if ctx.shape_of(a) != shape {
                return Err(format!(
                    "GatedDeltaNet: input {index} must have shape {shape:?}"
                ));
            }
            ctx.astype(a, mlx::mlx_dtype__MLX_FLOAT32)?
        } else {
            state
        };
        inputs.push(array);
    }
    let scale = n.floats.get("scale").copied().unwrap_or(0.0);
    let scale = if scale == 0.0 {
        1.0 / (k as f32).sqrt()
    } else {
        scale
    };
    let threads = b
        .checked_mul(h)
        .and_then(|x| x.checked_mul(v))
        .and_then(|x| x.checked_mul(32))
        .ok_or("GatedDeltaNet: dispatch size exceeds int32")?;
    let dtype = ctx.dtype_of(qkv);
    let out = kernel(
        ctx,
        "onnxrt_mlx_packed_gated_delta",
        DELTA,
        &[
            "qkv", "offsets", "initial", "decay", "beta", "a_log", "dt_bias",
        ],
        &inputs,
        &[
            ("B", b),
            ("N", qs[0]),
            ("H", h),
            ("HQ", hq),
            ("K", k),
            ("V", v),
            ("WIDTH", qs[1]),
            ("DECAY", decay as i32),
            ("BETA", beta as i32),
            ("RETRIEVE", retrieve as i32),
            ("SOFTPLUS", softplus as i32),
            (
                "SIGMOID",
                (attr("beta_activation", "none") == "sigmoid") as i32,
            ),
            (
                "NORM",
                n.ints.get("qk_l2_norm").copied().unwrap_or(0) as i32,
            ),
        ],
        Some(scale),
        &[vec![qs[0], h, v], ss],
        &[dtype, mlx::mlx_dtype__MLX_FLOAT32],
        threads,
    )?;
    ctx.bind(&n.outputs[0], out[0]);
    if n.outputs.len() > 1 && !n.outputs[1].name.is_empty() {
        ctx.bind(&n.outputs[1], out[1]);
    }
    ctx.mark_fast("mlx_packed_gated_delta");
    Ok(())
}

fn common_claim(n: &NodeView, offsets: usize) -> ClaimResult {
    require!(
        n.int_attr("state_update_capacity", 0) == 0,
        "compact transition capture is not implemented; use state_update_capacity=0"
    );
    let x = n.input_info(0).ok_or("missing input type/shape")?;
    require!(
        is_mlx_float(x.dtype) && x.shape.len() == 2,
        "requires packed rank-2 float input"
    );
    let cu = n
        .input_info(offsets)
        .ok_or("missing cumulative sequence lengths")?;
    require!(
        cu.dtype == INT32 && cu.shape.len() == 1,
        "cumulative sequence lengths must be rank-1 int32"
    );
    require!(
        (1..=i32::MAX as i64).contains(&x.shape[1]),
        "packed channel width must be a static positive int32"
    );
    Ok(())
}

fn conv_claim(n: &NodeView) -> ClaimResult {
    common_claim(n, 2)?;
    require!(
        n.num_inputs() == 5 && n.num_outputs() == 2,
        "expects input, weight, offsets, optional bias and initial state; two outputs"
    );
    let x = n.input_info(0).unwrap();
    let w = n.input_info(1).ok_or("missing convolution weight")?;
    let s = n.input_info(4).ok_or("missing convolution state")?;
    let d = n.int_attr("dilation", 1);
    require!(
        d > 0 && d <= i32::MAX as i64,
        "dilation must be positive int32"
    );
    require!(
        w.dtype == x.dtype
            && w.shape.len() == 3
            && w.shape[0] == x.shape[1]
            && w.shape[1] == 1
            && w.shape[2] > 0,
        "weight must have activation dtype and shape [channels,1,kernel]"
    );
    let l = (w.shape[2] - 1)
        .checked_mul(d)
        .filter(|&v| v < i32::MAX as i64)
        .ok_or("convolution receptive field exceeds int32")?;
    require!(
        s.dtype == x.dtype && s.shape.len() == 3 && s.shape[1..] == [x.shape[1], l],
        "state must have shape [batch,channels,receptive_field]"
    );
    let cu = n.input_info(2).unwrap();
    require!(
        s.shape[0] < 0 || cu.shape[0] < 0 || cu.shape[0] == s.shape[0] + 1,
        "offsets must have batch+1 entries"
    );
    for (index, shape) in [(0, &x.shape), (1, &s.shape)] {
        require!(
            n.output_info(index)
                .is_some_and(|o| o.dtype == x.dtype && compatible(&o.shape, shape)),
            "convolution outputs must preserve input/state shape and activation dtype"
        );
    }
    if let Some(bias) = n.input_info(3) {
        require!(
            bias.dtype == x.dtype && bias.shape == [x.shape[1]],
            "bias must have activation dtype and shape [channels]"
        );
    }
    require!(
        matches!(
            n.string_attr("activation", "none").as_str(),
            "none" | "silu" | "swish"
        ),
        "activation must be none, silu or swish"
    );
    Ok(())
}

fn delta_claim(n: &NodeView) -> ClaimResult {
    common_claim(n, 3)?;
    require!(
        (7..=9).contains(&n.num_inputs()) && (1..=2).contains(&n.num_outputs()),
        "expects packed QKV, offsets, gates and FP32 state; compact updates are unsupported"
    );
    require!(
        !n.input_present(1) && !n.input_present(2),
        "currently supports packed QKV; separate query/key/value is not implemented"
    );
    let s = n.input_info(6).ok_or("missing FP32 recurrent state")?;
    require!(
        s.dtype == FLOAT && s.shape.len() == 4 && s.shape[1..].iter().all(|&d| d > 0),
        "state must be FP32 [batch,value_heads,value_width,key_width] with static head geometry"
    );
    require!(
        s.shape[3] <= 256,
        "packed recurrence supports key width up to 256"
    );
    let x = n.input_info(0).unwrap();
    let value_width = s.shape[1]
        .checked_mul(s.shape[2])
        .filter(|&v| v < x.shape[1])
        .ok_or("state value width must fit packed input width")?;
    let remaining = x.shape[1] - value_width;
    require!(
        remaining > 0
            && remaining % (2 * s.shape[3]) == 0
            && s.shape[1] % (remaining / (2 * s.shape[3])) == 0,
        "packed QKV width must match state and inverse grouped heads"
    );
    let rule = n.string_attr("update_rule", "gated_delta");
    require!(
        matches!(rule.as_str(), "linear" | "gated" | "delta" | "gated_delta"),
        "update_rule must be linear, gated, delta or gated_delta"
    );
    let gate = n.string_attr("gate_activation", "none");
    require!(
        matches!(gate.as_str(), "none" | "qwen" | "softplus"),
        "unsupported gate activation"
    );
    require!(
        matches!(
            n.string_attr("beta_activation", "none").as_str(),
            "none" | "sigmoid"
        ),
        "beta activation must be none or sigmoid"
    );
    require!(
        (0..=1).contains(&n.int_attr("qk_l2_norm", 0)),
        "qk_l2_norm must be 0 or 1"
    );
    let cu = n.input_info(3).unwrap();
    require!(
        s.shape[0] < 0 || cu.shape[0] < 0 || cu.shape[0] == s.shape[0] + 1,
        "offsets must have batch+1 entries"
    );
    require!(
        n.output_info(0)
            .is_some_and(|o| o.dtype == x.dtype
                && compatible(&o.shape, &[x.shape[0], s.shape[1], s.shape[2]])),
        "output must have activation dtype and shape [tokens,value_heads,value_width]"
    );
    if let Some(output) = n.output_info(1) {
        require!(
            output.dtype == FLOAT && compatible(&output.shape, &s.shape),
            "final state must preserve the FP32 V-major initial-state shape"
        );
    }
    for (index, needed) in [
        (4, false),
        (5, false),
        (7, gate != "none" && n.input_present(4)),
        (8, gate != "none" && n.input_present(4)),
    ] {
        if needed && !n.input_present(index) {
            deny!("missing required gate input {index}");
        }
        if let Some(t) = n.input_info(index) {
            require!(
                if index >= 7 {
                    t.dtype == FLOAT
                } else {
                    t.dtype == FLOAT || t.dtype == x.dtype
                },
                "gate inputs must be FP32 or activation dtype; gate parameters must be FP32"
            );
            let expected = if index >= 7 {
                vec![s.shape[1]]
            } else {
                vec![x.shape[0], s.shape[1]]
            };
            require!(
                t.shape.len() == expected.len()
                    && t.shape
                        .iter()
                        .zip(expected)
                        .all(|(&a, b)| a < 0 || b < 0 || a == b),
                "only scalar-per-head gates are implemented"
            );
        }
    }
    Ok(())
}

pub fn register(registry: &mut OpRegistry) {
    for (op_type, handler, claim) in [
        (
            "VarlenCausalConvWithState",
            conv as crate::registry::OpHandler,
            conv_claim as crate::registry::ClaimPredicate,
        ),
        (
            "GatedDeltaNet",
            delta as crate::registry::OpHandler,
            delta_claim as crate::registry::ClaimPredicate,
        ),
    ] {
        registry.register_shape_keyed(
            OpRegistration {
                domain: "com.microsoft",
                op_type,
                min_opset: 1,
                max_opset: K_ANY_OPSET,
                handler,
                claim,
            },
            "emits Metal CustomKernel without Primitive::output_shapes",
        );
    }
}
