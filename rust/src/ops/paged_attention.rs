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

use std::ffi::{CStr, CString};
use std::sync::OnceLock;

use crate::engine::{MlxError, TranslationContext};
use crate::mlx::VectorArray;
use crate::sys::mlx;

const VECTOR_BODY: &str = r#"
    constexpr int BN = 32, BD = 32, E = D / BD;
    uint h = threadgroup_position_in_grid.x;
    uint row = TWO_PASS ? thread_position_in_threadgroup.y : threadgroup_position_in_grid.y;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint partition = TWO_PASS ? threadgroup_position_in_grid.z : sg;
    uint kh = h / (H / KV);
    int past = position[0], total = past + (CAUSAL ? int(row) + 1 : NQ);
    float q[E], o[E];
    for (int j = 0; j < E; ++j) {
        q[j] = as_type<float>(SCALE_BITS) * float(query[(h * NQ + row) * D + lane * E + j]);
        o[j] = 0.0f;
    }
    float maximum = -FLT_MAX, sum = 0.0f;
    for (size_t logical = partition; logical < size_t(total); logical += PARTITIONS) {
        int i = int(logical);
        size_t offset = 0;
        if (i < past) {
            int physical = table[B * MAX_BLOCKS + i / BLOCK];
            offset = ((size_t(physical) * BLOCK + i % BLOCK) * KV + kh) * D;
        }
        int source_row = i - past;
        if (i < past)
            for (int j = 0; j < NQ; ++j)
                if (position[j + 1] >= 0 && int(offset / (KV * D)) == position[j + 1])
                    source_row = j;
        bool patched = source_row >= 0;
        float score = 0.0f;
        for (int j = 0; j < E; ++j) {
            uint d = lane * E + j;
            float k = !patched ? float(keys[offset + d]) : float(new_key[(kh * NQ + source_row) * D + d]);
            score += q[j] * k;
        }
        score = simd_sum(score);
        float next = max(maximum, score);
        float factor = fast::exp(maximum - next), weight = fast::exp(score - next);
        maximum = next;
        sum = sum * factor + weight;
        for (int j = 0; j < E; ++j) {
            uint d = lane * E + j;
            float v = !patched ? float(values[offset + d]) : float(new_value[(kh * NQ + source_row) * D + d]);
            o[j] = o[j] * factor + weight * v;
        }
    }
 "#;

const VECTOR_REDUCE: &str = r#"
    threadgroup float outputs[BN * BD];
    threadgroup float maxima[BN], sums[BN];
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
        for (int j = 0; j < E; ++j) output[(h * NQ + row) * D + sg * E + j] = T(o[j]);
"#;

const VECTOR_PARTIALS: &str = r#"
    size_t index = (h * NQ + row) * PARTITIONS + partition;
    if (lane == 0) {
        partial_sums[index] = sum;
        partial_maxima[index] = maximum;
    }
    for (int j = 0; j < E; ++j)
        output[index * D + lane * E + j] = o[j];
"#;

const PARTIALS_REDUCE: &str = r#"
    constexpr int BN = 32, BD = 32, E = D / BD;
    uint h = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    size_t base = (h * NQ + row) * PARTITIONS;
    float o[E] = {0};
    threadgroup float outputs[BN * BD];
    float sum = 0.0f, maximum = -FLT_MAX;
    for (int b = 0; b < PARTITIONS / BN; ++b)
        maximum = max(maximum, maxima[base + lane + BN * b]);
    maximum = simd_max(maximum);
    for (int b = 0; b < PARTITIONS / BN; ++b) {
        float factor = fast::exp(maxima[base + lane + BN * b] - maximum);
        sum += factor * sums[base + lane + BN * b];
    }
    sum = simd_sum(sum);
    for (int b = 0; b < PARTITIONS / BN; ++b) {
        float factor = fast::exp(maxima[base + sg + BN * b] - maximum);
        for (int j = 0; j < E; ++j)
            o[j] += factor * partials[(base + sg + BN * b) * D + lane * E + j];
    }
    for (int j = 0; j < E; ++j) {
        outputs[lane * BD + sg] = o[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[j] = simd_sum(outputs[sg * BD + lane]);
        o[j] = sum == 0.0f ? o[j] : o[j] / sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0)
        for (int j = 0; j < E; ++j)
            output[(h * NQ + row) * D + sg * E + j] = T(o[j]);
"#;

fn architecture(ctx: &TranslationContext) -> Result<u8, MlxError> {
    static ARCHITECTURE: OnceLock<Result<u8, MlxError>> = OnceLock::new();
    ARCHITECTURE
        .get_or_init(|| {
            let mut device = unsafe { mlx::mlx_device_new() };
            let mut info = unsafe { mlx::mlx_device_info_new() };
            let mut value = std::ptr::null();
            let mut rc = unsafe { mlx::mlx_stream_get_device(&mut device, ctx.stream()) };
            if rc == 0 {
                rc = unsafe { mlx::mlx_device_info_get(&mut info, device) };
            }
            if rc == 0 {
                rc = unsafe {
                    mlx::mlx_device_info_get_string(&mut value, info, c"architecture".as_ptr())
                };
            }
            let result = if rc == 0 && !value.is_null() {
                unsafe { CStr::from_ptr(value) }
                    .to_bytes()
                    .last()
                    .copied()
                    .ok_or_else(|| "PagedAttention: empty GPU architecture; verify MLX device support".into())
            } else {
                Err(format!(
                    "PagedAttention: cannot read GPU architecture (status {rc}); verify MLX device support"
                ))
            };
            unsafe {
                mlx::mlx_device_info_free(info);
                mlx::mlx_device_free(device);
            }
            result
        })
        .clone()
}

fn partitions(architecture: u8, length: i32, queries: i32) -> i32 {
    // Match the FP32 dense fallback, whose KV heads have already been repeated.
    match architecture {
        b's' if length >= 1024 => {
            if length <= 1024 || queries <= 4 {
                64
            } else if length <= 8192 {
                128
            } else if length <= 32768 {
                256
            } else if length <= 65536 {
                512
            } else {
                1024
            }
        }
        b'd' if length >= 1024 => {
            if queries <= 2 && length > 8192 {
                256
            } else if queries >= 6 && (16384..65536).contains(&length) {
                512
            } else if queries >= 6 && length >= 65536 {
                1024
            } else {
                128
            }
        }
        _ => 0,
    }
}

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
    queries: i32,
    length: i32,
    causal: bool,
    scale: f32,
) -> Result<mlx::mlx_array, MlxError> {
    let partitions = if length < 1024 {
        0
    } else {
        partitions(architecture(ctx)?, length, queries)
    };
    let two_pass = partitions > 0;
    let templates = [
        (c"H", heads),
        (c"KV", kv_heads),
        (c"D", head),
        (c"BLOCK", block),
        (c"MAX_BLOCKS", max_blocks),
        (c"B", batch),
        (c"NQ", queries),
        (c"CAUSAL", i32::from(causal)),
        (c"TWO_PASS", i32::from(two_pass)),
        (c"PARTITIONS", if two_pass { partitions } else { 32 }),
        (c"SCALE_BITS", scale.to_bits() as i32),
    ];
    let names = [
        "query",
        "keys",
        "values",
        "table",
        "new_key",
        "new_value",
        "position",
    ];
    let shape = vec![1, heads, queries, head];
    let mut specs = vec![("output", shape.clone(), mlx::mlx_dtype__MLX_FLOAT16)];
    if two_pass {
        specs = vec![
            (
                "output",
                vec![heads, queries, partitions, head],
                mlx::mlx_dtype__MLX_FLOAT32,
            ),
            (
                "partial_sums",
                vec![heads, queries, partitions],
                mlx::mlx_dtype__MLX_FLOAT32,
            ),
            (
                "partial_maxima",
                vec![heads, queries, partitions],
                mlx::mlx_dtype__MLX_FLOAT32,
            ),
        ];
    }
    let source = format!(
        "{VECTOR_BODY}{}",
        if two_pass {
            VECTOR_PARTIALS
        } else {
            VECTOR_REDUCE
        }
    );
    let results = dispatch(
        ctx,
        c"onnxrt_mlx_paged_vector",
        &source,
        &names,
        arrays,
        &templates,
        &specs,
        [
            heads
                .checked_mul(if two_pass { 32 } else { 1024 })
                .ok_or("PagedAttention: vector launch grid exceeds int32")?,
            queries,
            if two_pass { partitions } else { 1 },
        ],
        [
            if two_pass { 32 } else { 1024 },
            if two_pass { queries } else { 1 },
            1,
        ],
    )?;
    ctx.mark_fast("mlx_paged_vector");
    if !two_pass {
        return Ok(results[0]);
    }
    let reduced = dispatch(
        ctx,
        c"onnxrt_mlx_paged_vector_reduce",
        PARTIALS_REDUCE,
        &["partials", "sums", "maxima"],
        &results,
        &templates,
        &[("output", shape, mlx::mlx_dtype__MLX_FLOAT16)],
        [
            heads
                .checked_mul(1024)
                .ok_or("PagedAttention: reduction launch grid exceeds int32")?,
            queries,
            1,
        ],
        [1024, 1, 1],
    )?;
    Ok(reduced[0])
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    ctx: &mut TranslationContext,
    name: &CStr,
    source: &str,
    input_names: &[&str],
    arrays: &[mlx::mlx_array],
    templates: &[(&CStr, i32)],
    specs: &[(&str, Vec<i32>, mlx::mlx_dtype)],
    grid: [i32; 3],
    group: [i32; 3],
) -> Result<Vec<mlx::mlx_array>, MlxError> {
    let names = super::selective_scan::VecString::new(input_names)?;
    let outputs = super::selective_scan::VecString::new(
        &specs.iter().map(|spec| spec.0).collect::<Vec<_>>(),
    )?;
    let source = CString::new(source).unwrap();
    let kernel = unsafe {
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
    if kernel.ctx.is_null() {
        return Err(
            "PagedAttention: cannot construct direct Metal kernel; verify Metal support".into(),
        );
    }
    let config = unsafe { mlx::mlx_fast_metal_kernel_config_new() };
    let mut rc = unsafe {
        mlx::mlx_fast_metal_kernel_config_add_template_arg_dtype(config, c"T".as_ptr(), specs[0].2)
    };
    for &(name, value) in templates {
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_template_arg_int(config, name.as_ptr(), value)
        };
    }
    for (_, shape, dtype) in specs {
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_output_arg(
                config,
                shape.as_ptr(),
                shape.len(),
                *dtype,
            )
        };
    }
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_grid(config, grid[0], grid[1], grid[2]) };
    rc |= unsafe {
        mlx::mlx_fast_metal_kernel_config_set_thread_group(config, group[0], group[1], group[2])
    };
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
    if rc != 0 || result.size() != specs.len() {
        return Err(format!(
            "PagedAttention: direct Metal dispatch failed (status {rc}); verify GPU limits and cache geometry"
        ));
    }
    Ok((0..specs.len()).map(|i| ctx.keep(result.get(i))).collect())
}

#[cfg(test)]
mod tests {
    use super::partitions;

    #[test]
    fn fp32_dense_partition_boundaries() {
        for (architecture, length, queries, expected) in [
            (b'd', 1023, 8, 0),
            (b'd', 1024, 1, 128),
            (b'd', 8192, 1, 128),
            (b'd', 8193, 1, 256),
            (b'd', 16384, 5, 128),
            (b'd', 16384, 8, 512),
            (b'd', 65536, 8, 1024),
            (b's', 1024, 8, 64),
            (b's', 1025, 5, 128),
            (b's', 8193, 5, 256),
            (b's', 32769, 5, 512),
            (b's', 65537, 5, 1024),
            (b's', 65537, 4, 64),
            (b'g', 16384, 8, 0),
        ] {
            assert_eq!(partitions(architecture, length, queries), expected);
        }
    }
}
