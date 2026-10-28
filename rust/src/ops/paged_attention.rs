//! Direct block-table GQA with online FP32 softmax. Cache mutation is deferred to the ORT boundary.

use crate::engine::{MlxError, TranslationContext};
use crate::mlx::VectorArray;
use crate::sys::mlx;

const SOURCE: &std::ffi::CStr = c"
    uint group = thread_position_in_grid.x / 32;
    uint lane = thread_position_in_grid.x % 32;
    uint token = group / H;
    uint head = group % H;
    uint kv_head = head / (H / KV);
    int past = metadata[0];
    int updates = metadata[1];
    float query[(D + 31) / 32];
    float acc[(VD + 31) / 32];
    for (uint i = 0; i < (D + 31) / 32; ++i) {
        uint d = lane + 32 * i;
        query[i] = d < D ? float(q[(head * NQ + token) * D + d]) : 0.0f;
    }
    for (uint i = 0; i < (VD + 31) / 32; ++i) { acc[i] = 0.0f; }
    float maximum = HAS_SINK ? float(sink[head]) : -INFINITY;
    float denominator = HAS_SINK ? 1.0f : 0.0f;
    for (int position = 0; position <= past + int(token); ++position) {
        int slot = 0;
        int update = -1;
        bool cached = position < past;
        if (cached) {
            slot = blocks[position / BLOCK] * BLOCK + position % BLOCK;
            for (int row = 0; row < updates; ++row) {
                if (slots[row] == slot) { update = row; }
            }
        }
        float score = 0.0f;
        for (uint i = 0; i < (D + 31) / 32; ++i) {
            uint d = lane + 32 * i;
            if (d < D) {
                float key = cached
                    ? (update >= 0 ? float(uk[(update * KV + kv_head) * D + d])
                                   : float(kc[(slot * KV + kv_head) * D + d]))
                    : float(k[(kv_head * NQ + position - past) * D + d]);
                score += query[i] * key;
            }
        }
        score = simd_sum(score) * scale;
        float next_maximum = max(maximum, score);
        float correction = exp(maximum - next_maximum);
        float probability = exp(score - next_maximum);
        denominator = denominator * correction + probability;
        for (uint i = 0; i < (VD + 31) / 32; ++i) {
            uint d = lane + 32 * i;
            if (d < VD) {
                float value = cached
                    ? (update >= 0 ? float(uv[(update * KV + kv_head) * VD + d])
                                   : float(vc[(slot * KV + kv_head) * VD + d]))
                    : float(v[(kv_head * NQ + position - past) * VD + d]);
                acc[i] = acc[i] * correction + probability * value;
            }
        }
        maximum = next_maximum;
    }
    for (uint i = 0; i < (VD + 31) / 32; ++i) {
        uint d = lane + 32 * i;
        if (d < VD) { out[(token * H + head) * VD + d] = acc[i] / denominator; }
    }
";

#[allow(clippy::too_many_arguments)]
pub fn emit(
    ctx: &mut TranslationContext,
    q: mlx::mlx_array,
    k: mlx::mlx_array,
    v: mlx::mlx_array,
    kc: mlx::mlx_array,
    vc: mlx::mlx_array,
    blocks: mlx::mlx_array,
    slots: mlx::mlx_array,
    update_keys: mlx::mlx_array,
    update_values: mlx::mlx_array,
    metadata: mlx::mlx_array,
    scale: f32,
    sink: Option<mlx::mlx_array>,
) -> Result<mlx::mlx_array, MlxError> {
    let qs = ctx.shape_of(q);
    let vs = ctx.shape_of(v);
    let (heads, tokens, width) = (qs[1], qs[2], qs[3]);
    let kv_heads = vs[1];
    let value_width = vs[3];
    let block_size = ctx.shape_of(kc)[1];
    let has_sink = sink.is_some();
    let sink = match sink {
        Some(sink) => sink,
        None => ctx.zeros(&[heads], mlx::mlx_dtype__MLX_FLOAT32)?,
    };
    let scale = ctx.scalar_f32(scale);
    let threads = tokens
        .checked_mul(heads)
        .and_then(|size| size.checked_mul(32))
        .ok_or("PagedAttention: direct GQA grid size overflow")?;
    let mut names = [
        c"q".as_ptr(),
        c"k".as_ptr(),
        c"v".as_ptr(),
        c"kc".as_ptr(),
        c"vc".as_ptr(),
        c"blocks".as_ptr(),
        c"slots".as_ptr(),
        c"uk".as_ptr(),
        c"uv".as_ptr(),
        c"metadata".as_ptr(),
        c"scale".as_ptr(),
        c"sink".as_ptr(),
    ];
    let mut outputs_name = [c"out".as_ptr()];
    let input_names = unsafe { mlx::mlx_vector_string_new_data(names.as_mut_ptr(), names.len()) };
    let output_names =
        unsafe { mlx::mlx_vector_string_new_data(outputs_name.as_mut_ptr(), outputs_name.len()) };
    let kernel = unsafe {
        mlx::mlx_fast_metal_kernel_new(
            c"onnxrt_paged_gqa".as_ptr(),
            input_names,
            output_names,
            SOURCE.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        )
    };
    unsafe {
        mlx::mlx_vector_string_free(input_names);
        mlx::mlx_vector_string_free(output_names);
    }
    if kernel.ctx.is_null() {
        return Err("PagedAttention: creating direct paged GQA kernel failed".into());
    }
    let config = unsafe { mlx::mlx_fast_metal_kernel_config_new() };
    let mut rc = 0;
    for (name, value) in [
        (c"H", heads),
        (c"KV", kv_heads),
        (c"D", width),
        (c"VD", value_width),
        (c"NQ", tokens),
        (c"BLOCK", block_size),
        (c"HAS_SINK", i32::from(has_sink)),
    ] {
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_template_arg_int(config, name.as_ptr(), value)
        };
    }
    let shape = [tokens, heads * value_width];
    rc |= unsafe {
        mlx::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            shape.as_ptr(),
            shape.len(),
            mlx::mlx_dtype__MLX_FLOAT32,
        )
    };
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_grid(config, threads, 1, 1) };
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1) };
    let mut inputs = VectorArray::new();
    for array in [
        q,
        k,
        v,
        kc,
        vc,
        blocks,
        slots,
        update_keys,
        update_values,
        metadata,
        scale,
        sink,
    ] {
        inputs.append(array);
    }
    let mut outputs = VectorArray::new();
    if rc == 0 {
        rc = unsafe {
            mlx::mlx_fast_metal_kernel_apply(
                outputs.as_mut_ptr(),
                kernel,
                inputs.as_raw(),
                config,
                ctx.stream(),
            )
        };
    }
    unsafe {
        mlx::mlx_fast_metal_kernel_config_free(config);
        mlx::mlx_fast_metal_kernel_free(kernel);
    }
    if rc != 0 || outputs.size() != 1 {
        return Err("PagedAttention: emitting direct paged GQA kernel failed".into());
    }
    ctx.mark_fast("mlx_paged_gqa");
    Ok(ctx.keep(outputs.get(0)))
}
