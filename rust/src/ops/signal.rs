//! Signal / FFT op handlers (ai.onnx opset-17+): DFT, STFT, Hann/Hamming/Blackman windows and
//! MelWeightMatrix. FFTs compute in float32/complex64 and explicitly restore the ONNX dtype.
//! Host-readable scalar controls are validated before shape arithmetic or allocation.

use std::f64::consts::PI;

use crate::engine::{MlxError, NodeDesc, TensorRef, TranslationContext, mlx_dtype_from_onnx};
use crate::registry::{
    ClaimPredicate, ClaimResult, K_ANY_OPSET, NodeView, OpHandler, OpRegistration, OpRegistry,
    is_float64, is_mlx_float, is_mlx_numeric,
};
use crate::sys::mlx;
use crate::sys::ort;
use crate::{deny, require};

const NORM_BACKWARD: mlx::mlx_fft_norm = mlx::mlx_fft_norm__MLX_FFT_NORM_BACKWARD;

// ---- translate-time helpers ---------------------------------------------------------------------

/// Read a constant scalar integer input (int32/int64) at translate time.
fn read_scalar_int(ctx: &TranslationContext, r: &TensorRef) -> Result<i64, MlxError> {
    let h = ctx.raw_host(r)?;
    if h.data.is_null() || h.count != 1 || !is_scalar_shape(&h.shape) {
        return Err("MLX signal: expected a scalar int".to_string());
    }
    match h.dtype {
        t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
            Ok(unsafe { *(h.data as *const i64) })
        }
        t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => {
            Ok(unsafe { *(h.data as *const i32) } as i64)
        }
        _ => Err("MLX signal: scalar int input has an unsupported dtype".to_string()),
    }
}

/// Read a constant scalar float32 input at translate time.
fn read_scalar_float(ctx: &TranslationContext, r: &TensorRef) -> Result<f64, MlxError> {
    let h = ctx.raw_host(r)?;
    if h.data.is_null() || h.count != 1 || !is_scalar_shape(&h.shape) {
        return Err("MLX signal: expected a scalar float".to_string());
    }
    let value = match h.dtype {
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
            (unsafe { *(h.data as *const f32) }) as f64
        }
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => {
            half::f16::from_bits(unsafe { *(h.data as *const u16) }).to_f64()
        }
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => {
            half::bf16::from_bits(unsafe { *(h.data as *const u16) }).to_f64()
        }
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => unsafe {
            *(h.data as *const f64)
        },
        _ => return Err("MLX signal: scalar float input has an unsupported dtype".to_string()),
    };
    if !value.is_finite() {
        return Err("MLX signal: scalar float must be finite".to_string());
    }
    Ok(value)
}

fn positive_length(value: i64) -> Result<i32, MlxError> {
    i32::try_from(value)
        .ok()
        .filter(|&v| v > 0)
        .ok_or_else(|| "MLX signal: length/rate must be in 1..=i32::MAX".to_string())
}

fn bind_output(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    value: mlx::mlx_array,
) -> Result<(), MlxError> {
    let value = ctx.astype(value, mlx_dtype_from_onnx(n.outputs[0].otype))?;
    ctx.bind(&n.outputs[0], value);
    Ok(())
}

fn present(n: &NodeDesc, i: usize) -> bool {
    i < n.inputs.len() && n.inputs[i].source != crate::engine::Src::Absent
}

/// x[..., idx] dropping the trailing components axis (rank shrinks by 1).
fn take_last_index(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    idx: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let shape = ctx.shape_of(x);
    if shape.len() < 2 || !matches!(shape.last(), Some(1 | 2)) {
        return Err(
            "MLX DFT: expected signal dimensions followed by 1 or 2 components".to_string(),
        );
    }
    let rank = shape.len();
    let mut start = vec![0i32; rank];
    let stop = shape.clone();
    let mut stop2 = stop;
    start[rank - 1] = idx;
    stop2[rank - 1] = idx + 1;
    let sliced = ctx.slice(x, &start, &stop2)?;
    let squeezed: Vec<i32> = shape[..rank - 1].to_vec();
    ctx.reshape(sliced, &squeezed)
}

/// Stack real+imag of a complex array into a new trailing axis of size 2 (ONNX (real, imag) form).
fn stack_real_imag(
    ctx: &mut TranslationContext,
    cx: mlx::mlx_array,
    append_axis: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let re = ctx.emit(|res, s| unsafe { mlx::mlx_real(res, cx, s) })?;
    let im = ctx.emit(|res, s| unsafe { mlx::mlx_imag(res, cx, s) })?;
    ctx.stack(&[re, im], append_axis)
}

// ---- DFT ----------------------------------------------------------------------------------------

fn dft_axis(ctx: &TranslationContext, n: &NodeDesc, rank: i32) -> Result<i32, MlxError> {
    let mut axis: i64 = if n.since_version >= 20 {
        if present(n, 2) {
            read_scalar_int(ctx, &n.inputs[2])?
        } else {
            -2
        }
    } else {
        *n.ints.get("axis").unwrap_or(&1)
    };
    if axis < 0 {
        axis += rank as i64;
    }
    if axis < 0 || axis >= rank as i64 - 1 {
        return Err("MLX DFT: axis out of range".to_string());
    }
    Ok(axis as i32)
}

fn dft_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(x);
    let rank = shape.len() as i32;
    let last = shape[shape.len() - 1];
    let axis = dft_axis(ctx, n, rank)?;

    let inverse = n.ints.get("inverse").copied().unwrap_or(0) != 0;
    let onesided = n.ints.get("onesided").copied().unwrap_or(0) != 0;

    if onesided && last != if inverse { 2 } else { 1 } {
        return Err("MLX DFT: onesided requires real forward or complex inverse input".to_string());
    }
    let dft_length = positive_length(if present(n, 1) {
        read_scalar_int(ctx, &n.inputs[1])?
    } else if inverse && onesided {
        2 * (i64::from(shape[axis as usize]) - 1)
    } else {
        i64::from(shape[axis as usize])
    })?;

    let x = ctx.astype(x, mlx::mlx_dtype__MLX_FLOAT32)?;
    let one_c = ctx.scalar_complex(1.0, 0.0);
    let real = take_last_index(ctx, x, 0)?;
    let mut signal = ctx.mul(real, one_c)?;
    if last == 2 {
        let i_unit = ctx.scalar_complex(0.0, 1.0);
        let imag = take_last_index(ctx, x, 1)?;
        let imag_c = ctx.mul(imag, i_unit)?;
        signal = ctx.add(signal, imag_c)?;
    }

    if inverse && onesided {
        let real = ctx.emit(|res, s| unsafe {
            mlx::mlx_fft_irfft(res, signal, dft_length, axis, NORM_BACKWARD, s)
        })?;
        let mut out_shape = ctx.shape_of(real);
        out_shape.push(1);
        let out = ctx.reshape(real, &out_shape)?;
        return bind_output(ctx, n, out);
    }
    let mut spectrum = if inverse {
        ctx.emit(|res, s| unsafe {
            mlx::mlx_fft_ifft(res, signal, dft_length, axis, NORM_BACKWARD, s)
        })?
    } else {
        ctx.emit(|res, s| unsafe {
            mlx::mlx_fft_fft(res, signal, dft_length, axis, NORM_BACKWARD, s)
        })?
    };

    if onesided && !inverse {
        let res_shape = ctx.shape_of(spectrum);
        let start = vec![0i32; res_shape.len()];
        let mut stop = res_shape;
        stop[axis as usize] = dft_length / 2 + 1;
        spectrum = ctx.slice(spectrum, &start, &stop)?;
    }

    let out = stack_real_imag(ctx, spectrum, rank - 1)?;
    bind_output(ctx, n, out)
}

fn is_int_type(t: ort::ONNXTensorElementDataType) -> bool {
    t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
}

fn is_scalar_shape(shape: &[i64]) -> bool {
    shape.is_empty()
}

fn scalar_int(node: &NodeView, i: usize) -> bool {
    match node.input_info(i) {
        Some(info) => {
            is_int_type(info.dtype)
                && is_scalar_shape(&info.shape)
                && node.input_is_host_readable(i)
        }
        None => false,
    }
}

fn scalar_float(node: &NodeView, i: usize) -> bool {
    match node.input_info(i) {
        Some(info) => {
            matches!(
                info.dtype,
                ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                    | ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
                    | ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
                    | ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
            ) && is_scalar_shape(&info.shape)
                && node.input_is_host_readable(i)
        }
        None => false,
    }
}

fn const_int(node: &NodeView, i: usize) -> Option<i64> {
    let values = node.read_const_ints_any(i)?;
    match values.as_slice() {
        [value] => Some(*value),
        _ => None,
    }
}

fn length_claim(node: &NodeView, i: usize) -> ClaimResult {
    require!(
        scalar_int(node, i),
        "input {i} must be a host-readable scalar int32/int64"
    );
    if node.is_constant_initializer(i) {
        require!(
            const_int(node, i).is_some_and(|v| positive_length(v).is_ok()),
            "input {i} must be in 1..=i32::MAX"
        );
    }
    Ok(())
}

fn output_datatype_ok(node: &NodeView) -> bool {
    let dt = node.int_attr("output_datatype", 1);
    node.output_info(0).is_some_and(|o| {
        i64::from(o.dtype) == dt && (is_mlx_numeric(o.dtype) || is_float64(o.dtype))
    })
}

fn dft_claim(node: &NodeView) -> ClaimResult {
    let ni = node.num_inputs();
    require!(
        (1..=3).contains(&ni) && node.num_outputs() == 1,
        "expects 1-3 inputs and 1 output, got {}in/{}out",
        ni,
        node.num_outputs()
    );
    let (in_info, out_info) = match (node.input_info(0), node.output_info(0)) {
        (Some(a), Some(b)) => (a, b),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        is_mlx_float(in_info.dtype) && out_info.dtype == in_info.dtype,
        "input/output must be matching fp32/fp16/bf16, got {} -> {}",
        crate::registry::ort_dtype_name(in_info.dtype),
        crate::registry::ort_dtype_name(out_info.dtype)
    );
    let rank = in_info.shape.len();
    require!(rank >= 2, "input rank must be at least 2, got {rank}");
    let last = in_info.shape[rank - 1];
    require!(
        last < 0 || last == 1 || last == 2,
        "input trailing real/imag dimension must be 1 or 2, got {last}"
    );
    let since = node.since_version();
    let inverse = node.int_attr("inverse", 0);
    let onesided = node.int_attr("onesided", 0);
    require!(
        (inverse == 0 || inverse == 1) && (onesided == 0 || onesided == 1),
        "inverse and onesided must each be 0 or 1, got inverse={inverse}, onesided={onesided}"
    );
    require!(
        onesided == 0 || last < 0 || last == if inverse == 1 { 2 } else { 1 },
        "onesided requires real forward or complex inverse input"
    );
    let mut axis: i64 = if since >= 20 {
        if node.input_present(2) {
            require!(
                scalar_int(node, 2)
                    && node.input_info(2).is_some_and(|i| i.dtype
                        == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
                "axis must be a host-readable scalar int64"
            );
            if !node.is_constant_initializer(2) {
                if node.input_present(1) {
                    length_claim(node, 1)?;
                }
                return Ok(());
            }
            match const_int(node, 2) {
                Some(v) => v,
                None => deny!("axis must contain exactly one integer"),
            }
        } else {
            -2
        }
    } else {
        node.int_attr("axis", 1)
    };
    if axis < 0 {
        axis += rank as i64;
    }
    require!(
        axis >= 0 && axis < rank as i64 - 1,
        "axis is out of range for rank {rank} or selects the trailing real/imag dimension"
    );
    if node.input_present(1) {
        length_claim(node, 1)?;
    } else if in_info.shape[axis as usize] >= 0 {
        let size = in_info.shape[axis as usize];
        let size = if inverse == 1 && onesided == 1 {
            size.checked_sub(1).and_then(|v| v.checked_mul(2))
        } else {
            Some(size)
        };
        require!(
            size.is_some_and(|v| positive_length(v).is_ok()),
            "default DFT length must be in 1..=i32::MAX"
        );
    }
    Ok(())
}

// ---- STFT ---------------------------------------------------------------------------------------

fn stft_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let signal = ctx.resolve(&n.inputs[0])?;
    let in_shape = ctx.shape_of(signal);
    if in_shape.len() != 3 || !matches!(in_shape[2], 1 | 2) {
        return Err("MLX STFT: signal must have shape [batch, length, 1 or 2]".to_string());
    }
    let batch = in_shape[0];
    let signal_len = in_shape[1];
    let frame_step = positive_length(read_scalar_int(ctx, &n.inputs[1])?)?;

    let has_window = present(n, 2);
    let window = if has_window {
        Some(ctx.resolve(&n.inputs[2])?)
    } else {
        None
    };
    let frame_length = if present(n, 3) {
        positive_length(read_scalar_int(ctx, &n.inputs[3])?)?
    } else if let Some(w) = window {
        let shape = ctx.shape_of(w);
        if shape.len() != 1 {
            return Err("MLX STFT: window must have rank 1".to_string());
        }
        positive_length(i64::from(shape[0]))?
    } else {
        positive_length(i64::from(signal_len))?
    };
    if let Some(w) = window
        && ctx.shape_of(w) != [frame_length]
    {
        return Err("MLX STFT: window length must equal frame_length".to_string());
    }
    if signal_len < frame_length {
        return Err("MLX STFT: frame_length must not exceed signal length".to_string());
    }
    let onesided = n.ints.get("onesided").copied().unwrap_or(1) != 0;
    if onesided && in_shape[2] != 1 {
        return Err("MLX STFT: onesided requires real input".to_string());
    }
    let n_frames = 1 + (signal_len - frame_length) / frame_step;

    let signal = ctx.astype(signal, mlx::mlx_dtype__MLX_FLOAT32)?;
    let real = take_last_index(ctx, signal, 0)?;
    let flat = if in_shape[2] == 2 {
        let imag = take_last_index(ctx, signal, 1)?;
        let i_unit = ctx.scalar_complex(0.0, 1.0);
        let imag = ctx.mul(imag, i_unit)?;
        ctx.add(real, imag)?
    } else {
        real
    };
    let flat = ctx.contiguous(flat)?;
    let frame_shape = [batch, n_frames, frame_length];
    let strides: [i64; 3] = [signal_len as i64, frame_step as i64, 1];
    let mut frames = ctx.emit(|res, s| unsafe {
        mlx::mlx_as_strided(
            res,
            flat,
            frame_shape.as_ptr(),
            frame_shape.len(),
            strides.as_ptr(),
            strides.len(),
            0,
            s,
        )
    })?;
    if let Some(w) = window {
        let w = ctx.astype(w, mlx::mlx_dtype__MLX_FLOAT32)?;
        frames = ctx.mul(frames, w)?;
    }

    let spectrum = if onesided {
        ctx.emit(|res, s| unsafe {
            mlx::mlx_fft_rfft(res, frames, frame_length, 2, NORM_BACKWARD, s)
        })?
    } else {
        ctx.emit(|res, s| unsafe {
            mlx::mlx_fft_fft(res, frames, frame_length, 2, NORM_BACKWARD, s)
        })?
    };

    let out = stack_real_imag(ctx, spectrum, 3)?;
    bind_output(ctx, n, out)
}

fn stft_claim(node: &NodeView) -> ClaimResult {
    let ni = node.num_inputs();
    require!(
        (2..=4).contains(&ni) && node.num_outputs() == 1,
        "expects 2-4 inputs and 1 output, got {}in/{}out",
        ni,
        node.num_outputs()
    );
    let (sig, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(a), Some(b)) => (a, b),
        _ => deny!("missing tensor type/shape info on signal or output"),
    };
    require!(
        is_mlx_float(sig.dtype) && out.dtype == sig.dtype,
        "signal/output must be matching fp32/fp16/bf16, got {} -> {}",
        crate::registry::ort_dtype_name(sig.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        sig.shape.len() == 3 && (sig.shape[2] < 0 || matches!(sig.shape[2], 1 | 2)),
        "signal must have shape [batch, length, 1 or 2], got {:?}",
        sig.shape
    );
    length_claim(node, 1)?;
    let has_window = node.input_present(2);
    let has_frame_length = node.input_present(3);
    if has_window {
        match node.input_info(2) {
            Some(w) if w.dtype == sig.dtype && w.shape.len() == 1 && w.shape[0] != 0 => {}
            Some(w) => deny!(
                "window must be a nonempty rank-1 tensor matching signal dtype, got dtype {} shape {:?}",
                crate::registry::ort_dtype_name(w.dtype),
                w.shape
            ),
            None => deny!("missing tensor type/shape info on window"),
        }
    }
    if has_frame_length {
        length_claim(node, 3)?;
    }
    let window_len = node.input_info(2).and_then(|w| w.shape.first().copied());
    let frame_len = if has_frame_length {
        const_int(node, 3)
    } else if has_window {
        window_len
    } else {
        Some(sig.shape[1])
    };
    if let Some(len) = frame_len.filter(|&v| v >= 0) {
        require!(
            positive_length(len).is_ok(),
            "frame length must be positive"
        );
        require!(
            sig.shape[1] < 0 || len <= sig.shape[1],
            "frame_length must not exceed signal length"
        );
        require!(
            window_len.is_none_or(|w| w < 0 || w == len),
            "window length must equal frame_length"
        );
    }
    let onesided = node.int_attr("onesided", 1);
    require!(
        onesided == 0 || onesided == 1,
        "onesided must be 0 or 1, got {onesided}"
    );
    require!(
        onesided == 0 || sig.shape[2] < 0 || sig.shape[2] == 1,
        "onesided requires real signal"
    );
    Ok(())
}

// ---- Cosine-sum windows -------------------------------------------------------------------------

fn cosine_window(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    a0: f64,
    a1: f64,
    a2: f64,
) -> Result<(), MlxError> {
    let size = i64::from(positive_length(read_scalar_int(ctx, &n.inputs[0])?)?);
    let periodic = n.ints.get("periodic").copied().unwrap_or(1) != 0;
    let denom = if periodic {
        size as f64
    } else {
        (size - 1) as f64
    };
    if denom == 0.0 && !is_mlx_float(n.outputs[0].otype) && !is_float64(n.outputs[0].otype) {
        return Err(
            "MLX window: symmetric singleton is NaN and requires a floating output".to_string(),
        );
    }
    // Double output needs host cos; integer truncation must not follow rounded fp32 peaks.
    if !is_mlx_float(n.outputs[0].otype) {
        let mut values = Vec::new();
        values
            .try_reserve_exact(size as usize)
            .map_err(|e| format!("MLX window: allocation failed: {e}"))?;
        for i in 0..size {
            let arg = 2.0 * PI * i as f64 / denom;
            values.push((-a1 * arg.cos() + a2 * (2.0 * arg).cos()) + a0);
        }
        let output = if is_float64(n.outputs[0].otype) {
            ctx.from_host(
                values.as_ptr().cast(),
                &[size as i32],
                mlx::mlx_dtype__MLX_FLOAT64,
            )
        } else {
            let integers: Vec<i64> = values.iter().map(|&v| v as i64).collect();
            ctx.from_host(
                integers.as_ptr().cast(),
                &[size as i32],
                mlx::mlx_dtype__MLX_INT64,
            )
        };
        return bind_output(ctx, n, output);
    }
    let idx = ctx.emit(|res, s| unsafe {
        mlx::mlx_arange(res, 0.0, size as f64, 1.0, mlx::mlx_dtype__MLX_FLOAT32, s)
    })?;
    let two_pi = ctx.scalar_f32((2.0 * PI / denom) as f32);
    let arg = ctx.mul(idx, two_pi)?;
    let cos1 = ctx.emit(|res, s| unsafe { mlx::mlx_cos(res, arg, s) })?;
    let a1s = ctx.scalar_f32(a1 as f32);
    let cos1a1 = ctx.mul(cos1, a1s)?;
    let a0s = ctx.scalar_f32(a0 as f32);
    let mut y = ctx.sub(a0s, cos1a1)?;
    if a2 != 0.0 {
        let two = ctx.scalar_f32(2.0);
        let arg2 = ctx.mul(arg, two)?;
        let cos2 = ctx.emit(|res, s| unsafe { mlx::mlx_cos(res, arg2, s) })?;
        let a2s = ctx.scalar_f32(a2 as f32);
        let cos2a2 = ctx.mul(cos2, a2s)?;
        y = ctx.add(y, cos2a2)?;
    }
    bind_output(ctx, n, y)
}

fn hann_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    cosine_window(ctx, n, 0.5, 0.5, 0.0)
}

fn hamming_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    cosine_window(ctx, n, 25.0 / 46.0, 21.0 / 46.0, 0.0)
}

fn blackman_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    cosine_window(ctx, n, 0.42, 0.5, 0.08)
}

fn window_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    length_claim(node, 0)?;
    match node.output_info(0) {
        Some(o) if is_mlx_numeric(o.dtype) || is_float64(o.dtype) => {}
        Some(o) => deny!(
            "output dtype {} is not a supported numeric type",
            crate::registry::ort_dtype_name(o.dtype)
        ),
        None => deny!("missing tensor type/shape info on output"),
    }
    require!(
        output_datatype_ok(node),
        "output_datatype {} must be numeric and match the output tensor dtype",
        crate::registry::ort_dtype_name(
            node.int_attr("output_datatype", 1) as ort::ONNXTensorElementDataType
        )
    );
    let periodic = node.int_attr("periodic", 1);
    require!(
        periodic == 0 || periodic == 1,
        "periodic must be 0 or 1, got {periodic}"
    );
    Ok(())
}

// ---- MelWeightMatrix ----------------------------------------------------------------------------

fn mel_weight_matrix_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let num_mel_bins = positive_length(read_scalar_int(ctx, &n.inputs[0])?)?;
    let dft_length = positive_length(read_scalar_int(ctx, &n.inputs[1])?)?;
    let sample_rate = positive_length(read_scalar_int(ctx, &n.inputs[2])?)?;
    let lower_hz = read_scalar_float(ctx, &n.inputs[3])?;
    let upper_hz = read_scalar_float(ctx, &n.inputs[4])?;

    let num_spectrogram_bins = dft_length / 2 + 1;
    let num_edges = num_mel_bins
        .checked_add(2)
        .ok_or_else(|| "MLX MelWeightMatrix: edge count overflow".to_string())?;
    if lower_hz < 0.0 || upper_hz <= lower_hz || upper_hz > f64::from(sample_rate) / 2.0 {
        return Err("MLX MelWeightMatrix: require 0 <= lower < upper <= sample_rate/2".to_string());
    }

    let ncols = num_mel_bins as usize;
    let count = (num_spectrogram_bins as usize)
        .checked_mul(ncols)
        .filter(|&v| v <= i32::MAX as usize)
        .ok_or_else(|| "MLX MelWeightMatrix: output element count exceeds i32::MAX".to_string())?;
    let low_mel = 2595.0 * (1.0 + lower_hz / 700.0).log10();
    let high_mel = 2595.0 * (1.0 + upper_hz / 700.0).log10();
    let mel_step = (high_mel - low_mel) / num_edges as f64;

    let mut bins = Vec::new();
    bins.try_reserve_exact(num_edges as usize)
        .map_err(|e| format!("MLX MelWeightMatrix: edge allocation failed: {e}"))?;
    bins.resize(num_edges as usize, 0i32);
    for i in 0..num_edges {
        let mel = i as f64 * mel_step + low_mel;
        let hz = 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
        bins[i as usize] =
            ((f64::from(dft_length) + 1.0) * hz / f64::from(sample_rate)).floor() as i32;
    }

    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|e| format!("MLX MelWeightMatrix: allocation failed: {e}"))?;
    out.resize(count, 0.0f64);
    let mut put = |row: i32, col: i32, value: f64| {
        if row >= 0 && row < num_spectrogram_bins {
            out[row as usize * ncols + col as usize] = value;
        }
    };
    for i in 0..num_mel_bins {
        let lower_bin = bins[i as usize];
        let center_bin = bins[i as usize + 1];
        let higher_bin = bins[i as usize + 2];
        let low_to_center = center_bin - lower_bin;
        if low_to_center == 0 {
            put(center_bin, i, 1.0);
        } else {
            for j in lower_bin..=center_bin {
                put(j, i, f64::from(j - lower_bin) / f64::from(low_to_center));
            }
        }
        let center_to_high = higher_bin - center_bin;
        if center_to_high > 0 {
            for j in center_bin..higher_bin {
                put(j, i, f64::from(higher_bin - j) / f64::from(center_to_high));
            }
        }
    }

    let mat_shape = [num_spectrogram_bins, num_mel_bins];
    let arr = if is_float64(n.outputs[0].otype) {
        ctx.from_host(out.as_ptr().cast(), &mat_shape, mlx::mlx_dtype__MLX_FLOAT64)
    } else if is_mlx_float(n.outputs[0].otype) {
        let values: Vec<f32> = out.iter().map(|&v| v as f32).collect();
        ctx.from_host(
            values.as_ptr().cast(),
            &mat_shape,
            mlx::mlx_dtype__MLX_FLOAT32,
        )
    } else {
        let values: Vec<i64> = out.iter().map(|&v| v as i64).collect();
        ctx.from_host(
            values.as_ptr().cast(),
            &mat_shape,
            mlx::mlx_dtype__MLX_INT64,
        )
    };
    bind_output(ctx, n, arr)
}

fn mel_weight_matrix_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 5 && node.num_outputs() == 1,
        "expects 5 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    for i in 0..3 {
        length_claim(node, i)?;
    }
    if let Some(bins) = const_int(node, 0) {
        require!(
            bins <= i64::from(i32::MAX) - 2,
            "num_mel_bins + 2 exceeds i32::MAX"
        );
        if let Some(length) = const_int(node, 1) {
            require!(
                bins.checked_mul(length / 2 + 1)
                    .is_some_and(|v| v <= i64::from(i32::MAX)),
                "output element count exceeds i32::MAX"
            );
        }
    }
    require!(
        scalar_float(node, 3),
        "lower_edge_hertz must be a host-readable scalar float"
    );
    require!(
        scalar_float(node, 4),
        "upper_edge_hertz must be a host-readable scalar float"
    );
    match node.output_info(0) {
        Some(o) if is_mlx_numeric(o.dtype) || is_float64(o.dtype) => {}
        Some(o) => deny!(
            "output dtype {} is not a supported numeric type",
            crate::registry::ort_dtype_name(o.dtype)
        ),
        None => deny!("missing tensor type/shape info on output"),
    }
    require!(
        output_datatype_ok(node),
        "output_datatype {} must be numeric and match the output tensor dtype",
        crate::registry::ort_dtype_name(
            node.int_attr("output_datatype", 1) as ort::ONNXTensorElementDataType
        )
    );
    for i in 3..5 {
        if node.is_constant_initializer(i) {
            require!(
                node.read_const_scalar_f64(i)
                    .is_some_and(|v| v.is_finite() && v >= 0.0),
                "frequency bounds must be finite and nonnegative"
            );
        }
    }
    if let (Some(lower), Some(upper)) =
        (node.read_const_scalar_f64(3), node.read_const_scalar_f64(4))
    {
        require!(
            lower < upper,
            "lower frequency must be less than upper frequency"
        );
        if let Some(rate) = const_int(node, 2) {
            require!(
                upper <= rate as f64 / 2.0,
                "upper frequency must not exceed Nyquist"
            );
        }
    }
    Ok(())
}

// ---- registration -------------------------------------------------------------------------------

fn shape_keyed(
    registry: &mut OpRegistry,
    op_type: &'static str,
    handler: OpHandler,
    claim: ClaimPredicate,
    reason: &'static str,
) {
    registry.register_shape_keyed(
        OpRegistration {
            domain: "",
            op_type,
            min_opset: 17,
            max_opset: K_ANY_OPSET,
            handler,
            claim,
        },
        reason,
    );
}

pub fn register(registry: &mut OpRegistry) {
    shape_keyed(
        registry,
        "DFT",
        dft_op,
        dft_claim,
        "emits MLX FFT and Slice, which lack Primitive::output_shapes",
    );
    shape_keyed(
        registry,
        "STFT",
        stft_op,
        stft_claim,
        "emits MLX AsStrided and FFT, which lack Primitive::output_shapes",
    );
    shape_keyed(
        registry,
        "HannWindow",
        hann_op,
        window_claim,
        "conservatively shape-keyed: host-read size controls output shape",
    );
    shape_keyed(
        registry,
        "HammingWindow",
        hamming_op,
        window_claim,
        "conservatively shape-keyed: host-read size controls output shape",
    );
    shape_keyed(
        registry,
        "BlackmanWindow",
        blackman_op,
        window_claim,
        "conservatively shape-keyed: host-read size controls output shape",
    );
    shape_keyed(
        registry,
        "MelWeightMatrix",
        mel_weight_matrix_op,
        mel_weight_matrix_claim,
        "conservatively shape-keyed: host-read parameters control output shape and values",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_lengths_are_exact_and_bounded_before_narrowing() {
        for invalid in [
            i64::MIN,
            -1,
            0,
            i64::from(i32::MAX) + 1,
            (1 << 32) + 7,
            i64::MAX,
        ] {
            assert!(positive_length(invalid).is_err());
        }
        assert_eq!(positive_length(1).unwrap(), 1);
        assert_eq!(positive_length(i64::from(i32::MAX)).unwrap(), i32::MAX);
        assert!(is_scalar_shape(&[]));
        assert!(!is_scalar_shape(&[1]));
        assert!(!is_scalar_shape(&[1, 1]));
        assert!(!is_scalar_shape(&[0]));
    }
}
