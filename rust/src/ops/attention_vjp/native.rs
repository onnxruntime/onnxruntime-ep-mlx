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
    key_length: i32,
    causal: bool,
    kind: i32,
    mask: Option<Array>,
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
fn pad(
    a: Raw,
    axes: &[i32],
    high: &[i32],
    fill: Raw,
    stream: mlx::mlx_stream,
) -> Result<Array, MlxError> {
    let low = vec![0; axes.len()];
    emit(|out| unsafe {
        mlx::mlx_pad(
            out,
            a,
            axes.as_ptr(),
            axes.len(),
            low.as_ptr(),
            low.len(),
            high.as_ptr(),
            high.len(),
            fill,
            c"constant".as_ptr(),
            stream,
        )
    })
}

fn forward(inputs: mlx::mlx_vector_array, p: &Payload) -> Result<VectorArray, MlxError> {
    let stream = p.stream.as_raw();
    let width = [64, 96, 128, 256]
        .into_iter()
        .find(|&d| d >= p.width)
        .unwrap();
    let query_rows = p.query_length.max(16);
    let zero = Array::from_raw(unsafe { mlx::mlx_array_new_float32(0.) });
    let negative = Array::from_raw(unsafe { mlx::mlx_array_new_float32(f32::NEG_INFINITY) });
    let false_ = Array::from_raw(unsafe { mlx::mlx_array_new_bool(false) });
    let mut owned = Vec::new();
    for i in 0..3 {
        let a = emit(|out| unsafe { mlx::mlx_vector_array_get(out, inputs, i) })?;
        let rows = if i == 0 {
            query_rows - p.query_length
        } else {
            0
        };
        let a = if width != p.width || rows > 0 {
            pad(
                a.as_raw(),
                &[2, 3],
                &[rows, width - p.width],
                zero.as_raw(),
                stream,
            )?
        } else {
            a
        };
        owned.push(a);
    }
    let mut mask = match &p.mask {
        Some(a) => Some(a.retained()?),
        None => None,
    };
    if let Some(a) = mask.take() {
        let shape = a.shape();
        let mut axes = Vec::new();
        let mut high = Vec::new();
        let last = shape.len() - 1;
        if shape[last] != 1 && shape[last] < i64::from(p.key_length) {
            axes.push(last as i32);
            high.push(p.key_length - shape[last] as i32);
        }
        if last > 0 && shape[last - 1] != 1 && query_rows != p.query_length {
            axes.push(last as i32 - 1);
            high.push(query_rows - p.query_length);
        }
        mask = Some(if axes.is_empty() {
            a
        } else {
            pad(
                a.as_raw(),
                &axes,
                &high,
                if p.kind == 1 {
                    false_.as_raw()
                } else {
                    negative.as_raw()
                },
                stream,
            )?
        });
    }
    // ONNX aligns causal masks at the top left. MLX's causal mode is bottom-right.
    let causal_mode = p.causal && mask.is_none() && query_rows == p.key_length;
    if p.causal && !causal_mode {
        let qi = emit(|out| unsafe {
            mlx::mlx_arange(
                out,
                0.,
                f64::from(query_rows),
                1.,
                mlx::mlx_dtype__MLX_INT32,
                stream,
            )
        })?;
        let qi = emit(|out| unsafe {
            mlx::mlx_reshape(out, qi.as_raw(), [query_rows, 1].as_ptr(), 2, stream)
        })?;
        let ki = emit(|out| unsafe {
            mlx::mlx_arange(
                out,
                0.,
                f64::from(p.key_length),
                1.,
                mlx::mlx_dtype__MLX_INT32,
                stream,
            )
        })?;
        let causal =
            emit(|out| unsafe { mlx::mlx_less_equal(out, ki.as_raw(), qi.as_raw(), stream) })?;
        mask = Some(match mask {
            None => causal,
            Some(a) if p.kind == 1 => emit(|out| unsafe {
                mlx::mlx_logical_and(out, a.as_raw(), causal.as_raw(), stream)
            })?,
            Some(a) => emit(|out| unsafe {
                mlx::mlx_where(out, causal.as_raw(), a.as_raw(), negative.as_raw(), stream)
            })?,
        });
    }
    let valid = if let Some(a) = &mask {
        let allowed = if p.kind == 2 {
            emit(|out| unsafe { mlx::mlx_not_equal(out, a.as_raw(), negative.as_raw(), stream) })?
        } else {
            a.retained()?
        };
        Some(emit(|out| unsafe {
            mlx::mlx_any_axis(out, allowed.as_raw(), -1, true, stream)
        })?)
    } else {
        None
    };
    if let (Some(valid), Some(a)) = (valid.as_ref(), mask.take()) {
        // Give invalid rows a finite softmax before zeroing their output. This
        // prevents 0 * NaN in additive-mask backward kernels.
        let true_ = Array::from_raw(unsafe { mlx::mlx_array_new_bool(true) });
        let fill = if p.kind == 2 {
            zero.as_raw()
        } else {
            true_.as_raw()
        };
        mask = Some(emit(|out| unsafe {
            mlx::mlx_where(out, valid.as_raw(), a.as_raw(), fill, stream)
        })?);
    }
    let empty = mlx::mlx_array_ {
        ctx: std::ptr::null_mut(),
    };
    let mode = if mask.is_some() {
        c"array"
    } else if causal_mode {
        c"causal"
    } else {
        c""
    };
    let y = emit(|out| unsafe {
        mlx::mlx_fast_scaled_dot_product_attention(
            out,
            owned[0].as_raw(),
            owned[1].as_raw(),
            owned[2].as_raw(),
            p.scale,
            mode.as_ptr(),
            mask.as_ref().map_or(empty, |a| a.as_raw()),
            empty,
            false,
            stream,
        )
    })?;
    // Fully masked rows have zero output and zero upstream gradient into native SDPA.
    let y = match valid {
        Some(valid) => emit(|out| unsafe {
            mlx::mlx_where(out, valid.as_raw(), y.as_raw(), zero.as_raw(), stream)
        })?,
        None => y,
    };
    let y = if width != p.width || query_rows != p.query_length {
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
                stream,
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

fn shapes(ctx: &TranslationContext, inputs: &[Raw]) -> Result<(Vec<i32>, Vec<i32>), MlxError> {
    let [q, k, v, g, _] = inputs else {
        return Err("recomputed SDPA interface mismatch".into());
    };
    let qs = ctx.shape_of(*q);
    let ks = ctx.shape_of(*k);
    if ctx.shape_of(*g) != qs || ctx.shape_of(*v) != ks || qs[0] != ks[0] {
        return Err("recomputed SDPA live shape mismatch".into());
    }
    Ok((qs, ks))
}

fn payload(
    ctx: &TranslationContext,
    inputs: &[Raw],
    c: &Config,
    qs: &[i32],
    ks: &[i32],
) -> Result<Payload, MlxError> {
    if c.mask != 0 {
        let shape = ctx.shape_of(inputs[4]);
        let target = [qs[0], c.hq, qs[2], ks[2]];
        for (i, &dim) in shape.iter().enumerate() {
            let limit = target[4 - shape.len() + i];
            if dim != 1 && dim != limit && !(i + 1 == shape.len() && dim < limit) {
                return Err("recomputed SDPA mask does not broadcast to live scores".into());
            }
        }
    }
    let captured_mask = if c.mask == 0 {
        None
    } else {
        Some(emit(|out| unsafe { mlx::mlx_array_set(out, inputs[4]) })?)
    };
    Ok(Payload {
        scale: c.scale,
        stream: Stream::retained(ctx.stream())?,
        width: c.d,
        query_length: qs[2],
        key_length: ks[2],
        causal: c.causal != 0,
        kind: c.mask,
        mask: captured_mask,
    })
}

pub(super) fn primal(
    ctx: &mut TranslationContext,
    inputs: &[Raw],
    c: &Config,
) -> Result<Raw, MlxError> {
    let (qs, ks) = shapes(ctx, inputs)?;
    if qs[0] == 0 || qs[2] == 0 || ks[2] == 0 {
        return ctx.zeros(&qs, F32);
    }
    let captured = payload(ctx, inputs, c, &qs, &ks)?;
    let mut primals = VectorArray::new();
    for &value in &inputs[..3] {
        primals.append(value);
    }
    let values = forward(primals.as_raw(), &captured)?;
    if values.size() != 1 {
        return Err("recomputed SDPA forward output mismatch".into());
    }
    Ok(ctx.keep(values.get(0)))
}

pub(super) fn execute(
    ctx: &mut TranslationContext,
    inputs: &[Raw],
    c: &Config,
) -> Result<Vec<Raw>, MlxError> {
    let (qs, ks) = shapes(ctx, inputs)?;
    if qs[0] == 0 || qs[2] == 0 || ks[2] == 0 {
        return [qs, ks.clone(), ks]
            .iter()
            .map(|shape| ctx.zeros(shape, F32))
            .collect();
    }
    let payload = Box::into_raw(Box::new(payload(ctx, inputs, c, &qs, &ks)?));
    let function = Function(unsafe {
        mlx::mlx_closure_new_func_payload(Some(thunk), payload.cast(), Some(release))
    });
    if function.0.ctx.is_null() {
        unsafe { release(payload.cast()) };
        return Err("recomputed SDPA closure creation failed".into());
    }
    let mut primals = VectorArray::new();
    for &a in &inputs[..3] {
        primals.append(a);
    }
    let mut seeds = VectorArray::new();
    seeds.append(inputs[3]);
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
