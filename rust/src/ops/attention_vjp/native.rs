//! Recomputed attention pullback through MLX's fused SDPA VJP.
use super::{Config, F32};
use crate::engine::{MlxError, TranslationContext};
use crate::mlx::{Array, Stream, VectorArray};
use crate::sys::mlx;
use std::ffi::{c_int, c_void};

type Raw = mlx::mlx_array;
struct Payload {
    scale: f32,
    stream: Stream,
    width: i32,
    query_length: i32,
}
struct Function(mlx::mlx_closure);
impl Drop for Function {
    fn drop(&mut self) {
        unsafe { mlx::mlx_closure_free(self.0) };
    }
}
fn emit(f: impl FnOnce(*mut Raw) -> i32) -> Result<Array, MlxError> {
    let mut raw = unsafe { mlx::mlx_array_new() };
    let rc = f(&mut raw);
    let a = Array::from_raw(raw);
    if rc != 0 {
        return Err("recomputed SDPA operation failed".into());
    }
    Ok(a)
}
unsafe extern "C" fn release(payload: *mut c_void) {
    drop(unsafe { Box::from_raw(payload.cast::<Payload>()) });
}
fn forward(inputs: mlx::mlx_vector_array, p: &Payload) -> Result<VectorArray, MlxError> {
    let mut owned = Vec::new();
    for i in 0..3 {
        let a = emit(|out| unsafe { mlx::mlx_vector_array_get(out, inputs, i) })?;
        let rows = if i == 0 {
            (16 - p.query_length).max(0)
        } else {
            0
        };
        let a = if p.width < 64 || rows > 0 {
            let zero = Array::from_raw(unsafe { mlx::mlx_array_new_float32(0.) });
            emit(|out| unsafe {
                mlx::mlx_pad(
                    out,
                    a.as_raw(),
                    [2, 3].as_ptr(),
                    2,
                    [0, 0].as_ptr(),
                    2,
                    [rows, 64 - p.width].as_ptr(),
                    2,
                    zero.as_raw(),
                    c"constant".as_ptr(),
                    p.stream.as_raw(),
                )
            })?
        } else {
            a
        };
        owned.push(a);
    }
    let empty = mlx::mlx_array_ {
        ctx: std::ptr::null_mut(),
    };
    let y = emit(|out| unsafe {
        mlx::mlx_fast_scaled_dot_product_attention(
            out,
            owned[0].as_raw(),
            owned[1].as_raw(),
            owned[2].as_raw(),
            p.scale,
            c"".as_ptr(),
            empty,
            empty,
            false,
            p.stream.as_raw(),
        )
    })?;
    let y = if p.width < 64 || p.query_length < 16 {
        let mut stop: Vec<i32> = y.shape().into_iter().map(|d| d as i32).collect();
        stop[2] = p.query_length;
        stop[3] = p.width;
        emit(|out| unsafe {
            mlx::mlx_slice(
                out,
                y.as_raw(),
                [0; 4].as_ptr(),
                4,
                stop.as_ptr(),
                4,
                [1; 4].as_ptr(),
                4,
                p.stream.as_raw(),
            )
        })?
    } else {
        y
    };
    let mut outputs = VectorArray::new();
    outputs.append(y.as_raw());
    Ok(outputs)
}
unsafe extern "C" fn thunk(
    out: *mut mlx::mlx_vector_array,
    inputs: mlx::mlx_vector_array,
    payload: *mut c_void,
) -> c_int {
    // Never unwind across the C callback boundary.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        forward(inputs, unsafe { &*payload.cast::<Payload>() })
    }));
    match result {
        Ok(Ok(values)) => unsafe { mlx::mlx_vector_array_set(out, values.as_raw()) },
        _ => 1,
    }
}

pub(super) fn execute(
    ctx: &mut TranslationContext,
    inputs: &[Raw],
    c: &Config,
) -> Result<Vec<Raw>, MlxError> {
    let [q, k, v, g, _mask] = inputs else {
        return Err("recomputed SDPA interface mismatch".into());
    };
    let qs = ctx.shape_of(*q);
    let ks = ctx.shape_of(*k);
    if ctx.shape_of(*g) != qs || ctx.shape_of(*v) != ks || qs[0] != ks[0] {
        return Err("recomputed SDPA live shape mismatch".into());
    }
    if qs[0] == 0 || qs[2] == 0 || ks[2] == 0 {
        return [qs, ks.clone(), ks]
            .iter()
            .map(|shape| ctx.zeros(shape, F32))
            .collect();
    }
    let payload = Box::into_raw(Box::new(Payload {
        scale: c.scale,
        stream: Stream::retained(ctx.stream())?,
        width: c.d,
        query_length: qs[2],
    }));
    let function = Function(unsafe {
        mlx::mlx_closure_new_func_payload(Some(thunk), payload.cast(), Some(release))
    });
    if function.0.ctx.is_null() {
        unsafe { release(payload.cast()) };
        return Err("recomputed SDPA closure creation failed".into());
    }
    let mut primals = VectorArray::new();
    for &a in [*q, *k, *v].iter() {
        primals.append(a);
    }
    let mut seeds = VectorArray::new();
    seeds.append(*g);
    let mut values = VectorArray::new();
    let mut gradients = VectorArray::new();
    if unsafe {
        mlx::mlx_vjp(
            values.as_mut_ptr(),
            gradients.as_mut_ptr(),
            function.0,
            primals.as_raw(),
            seeds.as_raw(),
        )
    } != 0
        || gradients.size() != 3
    {
        return Err("MLX fused SDPA VJP failed".into());
    }
    ctx.mark_fast("mlx_fast_sdpa_recomputed_vjp");
    Ok((0..3).map(|i| ctx.keep(gradients.get(i))).collect())
}
