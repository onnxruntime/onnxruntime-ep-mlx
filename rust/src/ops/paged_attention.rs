//! Direct paged-cache vector attention, preserving MLX's FP32 reduction order.
// Adapted from MLX v0.32.4, mlx/backend/metal/kernels/sdpa_vector.h.
//
// Copyright (c) 2024 Apple Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use std::ffi::CString;

use crate::engine::{MlxError, TranslationContext};
use crate::mlx::VectorArray;
use crate::sys::mlx;

const VECTOR: &str = r#"
    constexpr int BN = 32, BD = 32, E = D / BD;
    uint h = threadgroup_position_in_grid.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint kh = h / (H / KV);
    int past = position[0], total = past + 1;
    float q[E], o[E];
    threadgroup float outputs[BN * BD];
    threadgroup float maxima[BN], sums[BN];
    for (int j = 0; j < E; ++j) {
        q[j] = as_type<float>(SCALE_BITS) * float(query[h * D + lane * E + j]);
        o[j] = 0.0f;
    }
    float maximum = -FLT_MAX, sum = 0.0f;
    for (int i = sg; i < total; i += BN) {
        size_t offset = 0;
        if (i < past) {
            int physical = table[B * MAX_BLOCKS + i / BLOCK];
            offset = ((size_t(physical) * BLOCK + i % BLOCK) * KV + kh) * D;
        }
        bool patched = i < past && position[1] >= 0
            && int(offset / (KV * D)) == position[1];
        float score = 0.0f;
        for (int j = 0; j < E; ++j) {
            uint d = lane * E + j;
            float k = i < past && !patched ? float(keys[offset + d]) : float(new_key[kh * D + d]);
            score += q[j] * k;
        }
        score = simd_sum(score);
        float next = max(maximum, score);
        float factor = fast::exp(maximum - next), weight = fast::exp(score - next);
        maximum = next;
        sum = sum * factor + weight;
        for (int j = 0; j < E; ++j) {
            uint d = lane * E + j;
            float v = i < past && !patched ? float(values[offset + d]) : float(new_value[kh * D + d]);
            o[j] = o[j] * factor + weight * v;
        }
    }
    if (lane == 0) {
        maxima[sg] = maximum;
        sums[sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = maxima[lane];
    float next = simd_max(maximum);
    float factor = fast::exp(maximum - next);
    sum = simd_sum(sums[lane] * factor);
    for (int j = 0; j < E; ++j) {
        outputs[lane * BD + sg] = o[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[j] = simd_sum(outputs[sg * BD + lane] * factor);
        o[j] = sum == 0.0f ? o[j] : o[j] / sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0)
        for (int j = 0; j < E; ++j) output[h * D + sg * E + j] = T(o[j]);
"#;

pub(super) fn gpu_stream(ctx: &TranslationContext) -> Result<bool, MlxError> {
    let mut device = unsafe { mlx::mlx_device_new() };
    let mut kind = mlx::mlx_device_type__MLX_CPU;
    let mut rc = unsafe { mlx::mlx_stream_get_device(&mut device, ctx.stream()) };
    if rc == 0 {
        rc = unsafe { mlx::mlx_device_get_type(&mut kind, device) };
    }
    unsafe { mlx::mlx_device_free(device) };
    if rc != 0 {
        return Err(format!(
            "PagedAttention: cannot inspect MLX stream device (status {rc}); verify MLX runtime and stream initialization"
        ));
    }
    Ok(kind == mlx::mlx_device_type__MLX_GPU)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn vector(
    ctx: &mut TranslationContext,
    arrays: &[mlx::mlx_array; 7],
    heads: i32,
    kv_heads: i32,
    head: i32,
    block: i32,
    max_blocks: i32,
    batch: i32,
    scale: f32,
) -> Result<mlx::mlx_array, MlxError> {
    let names = super::selective_scan::VecString::new(&[
        "query",
        "keys",
        "values",
        "table",
        "new_key",
        "new_value",
        "position",
    ])?;
    let outputs = super::selective_scan::VecString::new(&["output"])?;
    let source = CString::new(VECTOR).unwrap();
    let kernel = unsafe {
        mlx::mlx_fast_metal_kernel_new(
            c"onnxrt_mlx_paged_vector".as_ptr(),
            names.as_raw(),
            outputs.as_raw(),
            source.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        )
    };
    if kernel.ctx.is_null() {
        return Err(
            "PagedAttention: cannot construct direct Metal kernel; verify Metal support".into(),
        );
    }
    let config = unsafe { mlx::mlx_fast_metal_kernel_config_new() };
    let mut rc = unsafe {
        mlx::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"T".as_ptr(),
            mlx::mlx_dtype__MLX_FLOAT16,
        )
    };
    for (name, value) in [
        (c"H", heads),
        (c"KV", kv_heads),
        (c"D", head),
        (c"BLOCK", block),
        (c"MAX_BLOCKS", max_blocks),
        (c"B", batch),
        (c"SCALE_BITS", scale.to_bits() as i32),
    ] {
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_template_arg_int(config, name.as_ptr(), value)
        };
    }
    let shape = [1, heads, 1, head];
    rc |= unsafe {
        mlx::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            shape.as_ptr(),
            shape.len(),
            mlx::mlx_dtype__MLX_FLOAT16,
        )
    };
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_grid(config, heads * 1024, 1, 1) };
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_thread_group(config, 1024, 1, 1) };
    let mut input = VectorArray::new();
    for &array in arrays {
        input.append(array);
    }
    let mut result = VectorArray::new();
    if rc == 0 {
        rc = unsafe {
            mlx::mlx_fast_metal_kernel_apply(
                result.as_mut_ptr(),
                kernel,
                input.as_raw(),
                config,
                ctx.stream(),
            )
        };
    }
    unsafe {
        mlx::mlx_fast_metal_kernel_config_free(config);
        mlx::mlx_fast_metal_kernel_free(kernel);
    }
    if rc != 0 || result.size() != 1 {
        return Err(format!(
            "PagedAttention: direct Metal dispatch failed (status {rc}); verify GPU limits and cache geometry"
        ));
    }
    Ok(ctx.keep(result.get(0)))
}
