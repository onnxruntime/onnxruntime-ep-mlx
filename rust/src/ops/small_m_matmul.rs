//! Opt-in INT4 projection for short verification blocks; accumulation remains FP32.

use crate::engine::{MlxError, TranslationContext};
use crate::mlx::VectorArray;
use crate::sys::mlx;

const SOURCE: &std::ffi::CStr = c"
    constexpr uint KT = 64;
    constexpr uint NT = 32;
    constexpr uint AS = KT + 8;
    constexpr uint BS = KT + 8;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint col_base = threadgroup_position_in_grid.x * NT;
    threadgroup half atile[8 * AS];
    threadgroup half btile[NT * BS];
    threadgroup float result[8 * NT];
    simdgroup_matrix<float, 8, 8> accum = make_filled_simdgroup_matrix<float, 8, 8>(0);
    for (uint base = 0; base < K; base += KT) {
        for (uint i = tid; i < 8 * (KT / 8); i += 128) {
            uint row = i / (KT / 8);
            uint column = (i % (KT / 8)) * 8;
            threadgroup half4* dst =
                reinterpret_cast<threadgroup half4*>(atile + row * AS + column);
            if (row < M) {
                uint start = row * K + base + column;
                dst[0] = half4(a[start], a[start + 1], a[start + 2], a[start + 3]);
                dst[1] = half4(a[start + 4], a[start + 5], a[start + 6], a[start + 7]);
            } else {
                dst[0] = half4(0);
                dst[1] = half4(0);
            }
        }
        for (uint i = tid; i < NT * (KT / 32); i += 128) {
            uint column = i / (KT / 32);
            uint start = (i % (KT / 32)) * 32;
            uint n = col_base + column;
            half scale = scales[n * (K / 32) + (base + start) / 32];
            device const uint4* src =
                reinterpret_cast<device const uint4*>(w + n * (K / 8) + (base + start) / 8);
            uint4 packed = src[0];
            threadgroup half4* dst =
                reinterpret_cast<threadgroup half4*>(btile + column * BS + start);
            for (uint j = 0; j < 4; ++j) {
                uint bits = packed[j];
                dst[j * 2] = (half4(bits & 15, (bits >> 4) & 15,
                                    (bits >> 8) & 15, (bits >> 12) & 15) - half4(8)) * scale;
                dst[j * 2 + 1] = (half4((bits >> 16) & 15, (bits >> 20) & 15,
                                        (bits >> 24) & 15, (bits >> 28) & 15) - half4(8)) * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < KT; kk += 8) {
            simdgroup_matrix<half, 8, 8> aa, bb;
            simdgroup_load(aa, atile + kk, AS);
            simdgroup_load(bb, btile + (sg * 8) * BS + kk, BS, ulong2(0), true);
            simdgroup_multiply_accumulate(accum, aa, bb, accum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(accum, result + sg * 8, NT);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 8 * NT; i += 128) {
        uint row = i / NT;
        if (row < M) { out[row * N + col_base + i % NT] = result[i]; }
    }
";

pub(super) fn try_emit(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    w: mlx::mlx_array,
    scales: mlx::mlx_array,
    m: i32,
    n: i32,
    k: i32,
) -> Result<Option<mlx::mlx_array>, MlxError> {
    if !std::env::var("ONNXRUNTIME_EP_MLX_SMALL_M_MMA").is_ok_and(|value| value == "1")
        || !matches!(m, 7 | 8)
        || n <= 0
        || n % 32 != 0
        || k <= 0
        || k % 64 != 0
        || n.checked_mul(k / 8).is_none()
        || m.checked_mul(k).is_none()
        || m.checked_mul(n).is_none()
        || ctx.is_capturing()
        || ctx.dtype_of(a) != mlx::mlx_dtype__MLX_FLOAT16
        || ctx.dtype_of(scales) != mlx::mlx_dtype__MLX_FLOAT16
        || ctx.dtype_of(w) != mlx::mlx_dtype__MLX_UINT32
        || !super::paged_attention::gpu_stream(ctx)?
    {
        return Ok(None);
    }
    let Some(threads) = n.checked_mul(4) else {
        return Ok(None);
    };
    let mut names = [c"a".as_ptr(), c"w".as_ptr(), c"scales".as_ptr()];
    let mut output_names = [c"out".as_ptr()];
    let names = unsafe { mlx::mlx_vector_string_new_data(names.as_mut_ptr(), names.len()) };
    let output_names =
        unsafe { mlx::mlx_vector_string_new_data(output_names.as_mut_ptr(), output_names.len()) };
    let kernel = unsafe {
        mlx::mlx_fast_metal_kernel_new(
            c"onnxrt_int4_small_m_mma".as_ptr(),
            names,
            output_names,
            SOURCE.as_ptr(),
            c"#include <metal_simdgroup_matrix>\n".as_ptr(),
            true,
            false,
        )
    };
    unsafe {
        mlx::mlx_vector_string_free(names);
        mlx::mlx_vector_string_free(output_names);
    }
    if kernel.ctx.is_null() {
        return Err("MatMulNBits: creating small-row MMA kernel failed".into());
    }
    let config = unsafe { mlx::mlx_fast_metal_kernel_config_new() };
    let mut rc = 0;
    for (name, value) in [(c"M", m), (c"N", n), (c"K", k)] {
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_template_arg_int(config, name.as_ptr(), value)
        };
    }
    let shape = [m, n];
    rc |= unsafe {
        mlx::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            shape.as_ptr(),
            shape.len(),
            mlx::mlx_dtype__MLX_FLOAT16,
        )
    };
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_grid(config, threads, 1, 1) };
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1) };
    let mut inputs = VectorArray::new();
    for array in [a, w, scales] {
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
        return Err("MatMulNBits: emitting small-row MMA kernel failed".into());
    }
    ctx.mark_fast("mlx_int4_small_m_mma_fp32");
    Ok(Some(ctx.keep(outputs.get(0))))
}
