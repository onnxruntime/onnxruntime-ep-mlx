//! Convolution and pooling op handlers. Faithful port of the C++ `ops/conv.cc` (Conv, ConvTranspose,
//! AveragePool, MaxPool, GlobalAveragePool, GlobalMaxPool) plus the pooling extras from
//! `ops/normpool.cc` (LpPool, GlobalLpPool).
//!
//! ONNX conv/pool tensors are NCHW (channels-first); MLX's `mlx_conv*` and the strided-window pooling
//! path expect NHWC (channels-last). Every handler therefore transposes the input to channels-last
//! (`to_channels_last`), runs the MLX op, and transposes the result back (`from_channels_last`);
//! conv weights are repacked from ONNX `[O, I/g, k...]` to MLX `[O, k..., I/g]`. Conv and
//! ConvTranspose support 1-3 spatial dimensions (MLX restricts groups to 1D/2D). Transpose
//! convolution uses an uncropped convolution followed by output cropping, supporting asymmetric
//! pads without slicing the undilated input. Window pooling uses dilated strided views for
//! 1-3 spatial dimensions, including ceil-mode tails and original-coordinate MaxPool indices.

use std::os::raw::{c_char, c_void};

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::registry::{
    ClaimResult, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry, is_mlx_float,
};
use crate::sys::mlx;
use crate::{deny, require};

// ---- small arithmetic/movement helpers ----------------------------------------------------------

fn add(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_add, a, b)
}
fn abs_(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.unary(mlx::mlx_abs, a)
}
fn power(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_power, a, b)
}

/// A 0-d scalar of dtype `dt` holding `v` (kept), so no unwanted upcast happens in the compute path.
fn scalar_for_dtype(
    ctx: &mut TranslationContext,
    v: f32,
    dt: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    let s = ctx.scalar_f32(v);
    ctx.astype(s, dt)
}

/// ONNX NCHW -> MLX NHWC (channels last), forced contiguous.
fn to_channels_last(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    spatial_rank: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let mut axes = vec![0];
    axes.extend(2..spatial_rank + 2);
    axes.push(1);
    let t = ctx.transpose(x, &axes)?;
    ctx.contiguous(t)
}

/// MLX NHWC -> ONNX NCHW, forced contiguous.
fn from_channels_last(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    spatial_rank: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let mut axes = vec![0, spatial_rank + 1];
    axes.extend(1..spatial_rank + 1);
    let t = ctx.transpose(x, &axes)?;
    ctx.contiguous(t)
}

/// ONNX conv weight `[O, I/g, k...]` -> MLX `[O, k..., I/g]`, forced contiguous.
fn conv_weight_to_mlx(
    ctx: &mut TranslationContext,
    w: mlx::mlx_array,
    spatial_rank: i32,
) -> Result<mlx::mlx_array, MlxError> {
    to_channels_last(ctx, w, spatial_rank)
}

/// `int_arrays[name]` or a default vector of `value` repeated `size` times.
fn attr_or(n: &NodeDesc, name: &str, size: usize, value: i64) -> Vec<i64> {
    n.int_arrays
        .get(name)
        .cloned()
        .unwrap_or_else(|| vec![value; size])
}

fn present(n: &NodeDesc, i: usize) -> bool {
    i < n.inputs.len() && n.inputs[i].source != Src::Absent
}

// ---- Conv ---------------------------------------------------------------------------------------

fn conv_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x0 = ctx.resolve(&n.inputs[0])?;
    let spatial_rank = ctx.ndim(x0) as i32 - 2;
    // The claim permits dynamic (symbolic) spatial dims; the general compiled path is shape-keyed so
    // the concrete extent is known here. Guard against a genuinely-unresolvable (<=0) spatial dim so
    // the EP declines gracefully to CPU rather than emitting a mis-sized conv.
    {
        let xs = ctx.shape_of(x0);
        if (0..spatial_rank as usize).any(|i| xs[i + 2] <= 0) {
            return Err(format!(
                "Conv: non-positive spatial dim at trace time: {xs:?}"
            ));
        }
    }
    let strides = attr_or(n, "strides", spatial_rank as usize, 1);
    let dilations = attr_or(n, "dilations", spatial_rank as usize, 1);
    let group = n.ints.get("group").copied().unwrap_or(1) as i32;

    let auto_pad = n
        .strings
        .get("auto_pad")
        .map(String::as_str)
        .unwrap_or("NOTSET");
    let x = to_channels_last(ctx, x0, spatial_rank)?;
    let w0 = ctx.resolve(&n.inputs[1])?;
    let xs = ctx.shape_of(x0);
    let ws = ctx.shape_of(w0);
    if i64::from(xs[1]) != i64::from(ws[1]) * i64::from(group) {
        return Err(format!(
            "Conv: input channels {} must equal W channels {} times group {group}",
            xs[1], ws[1]
        ));
    }
    let pads: Vec<i64> = if auto_pad == "NOTSET" {
        attr_or(n, "pads", 2 * spatial_rank as usize, 0)
    } else {
        let sr = spatial_rank as usize;
        let xs = ctx.shape_of(x0);
        let ws = ctx.shape_of(w0);
        let in_sp: Vec<i64> = (0..sr).map(|i| xs[i + 2] as i64).collect();
        let kernel: Vec<i64> = (0..sr).map(|i| ws[i + 2] as i64).collect();
        auto_pad_pads(auto_pad, &in_sp, &kernel, &strides, &dilations)
            .ok_or_else(|| format!("Conv: unsupported auto_pad '{auto_pad}'"))?
    };
    let weight = conv_weight_to_mlx(ctx, w0, spatial_rank)?;

    // Symmetric pads (`pads[i] == pads[i + spatial_rank]`) use the fast symmetric conv1d/conv2d
    // primitives. ONNX also permits per-dim asymmetric pads (begin != end); MLX cannot express those
    // through conv1d/conv2d, so route them through `mlx_conv_general`, which takes separate
    // `padding_lo`/`padding_hi` vectors.
    let symmetric = (0..spatial_rank as usize).all(|i| pads[i] == pads[i + spatial_rank as usize]);

    let mut out = if symmetric && spatial_rank == 1 {
        let (st, pa, di) = (strides[0] as i32, pads[0] as i32, dilations[0] as i32);
        ctx.emit(|res, s| unsafe { mlx::mlx_conv1d(res, x, weight, st, pa, di, group, s) })?
    } else if symmetric && spatial_rank == 2 {
        let (s0, s1) = (strides[0] as i32, strides[1] as i32);
        let (p0, p1) = (pads[0] as i32, pads[1] as i32);
        let (d0, d1) = (dilations[0] as i32, dilations[1] as i32);
        ctx.emit(|res, s| unsafe {
            mlx::mlx_conv2d(res, x, weight, s0, s1, p0, p1, d0, d1, group, s)
        })?
    } else {
        let sr = spatial_rank as usize;
        let stride_i: Vec<i32> = strides.iter().map(|&v| v as i32).collect();
        let pad_lo: Vec<i32> = (0..sr).map(|i| pads[i] as i32).collect();
        let pad_hi: Vec<i32> = (0..sr).map(|i| pads[i + sr] as i32).collect();
        let dil_i: Vec<i32> = dilations.iter().map(|&v| v as i32).collect();
        let input_dil: Vec<i32> = vec![1; sr];
        ctx.emit(|res, s| unsafe {
            mlx::mlx_conv_general(
                res,
                x,
                weight,
                stride_i.as_ptr(),
                stride_i.len(),
                pad_lo.as_ptr(),
                pad_lo.len(),
                pad_hi.as_ptr(),
                pad_hi.len(),
                dil_i.as_ptr(),
                dil_i.len(),
                input_dil.as_ptr(),
                input_dil.len(),
                group,
                false,
                s,
            )
        })?
    };

    if present(n, 2) {
        let bias = ctx.resolve(&n.inputs[2])?;
        out = add(ctx, out, bias)?;
    }
    let y = from_channels_last(ctx, out, spatial_rank)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

// ---- DeformConv (2D) ---------------------------------------------------------------------------

fn deform_take(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    y: mlx::mlx_array,
    x_coord: mlx::mlx_array,
    channel_begin: i32,
    channel_count: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let shape = ctx.shape_of(x);
    let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
    let out_shape = ctx.shape_of(y);
    let (out_h, out_w) = (out_shape[1], out_shape[2]);
    let dt = ctx.dtype_of(x);
    let zero = scalar_for_dtype(ctx, 0.0, dt)?;
    let max_y = scalar_for_dtype(ctx, (height - 1) as f32, dt)?;
    let max_x = scalar_for_dtype(ctx, (width - 1) as f32, dt)?;

    let ge_y = ctx.binary(mlx::mlx_greater_equal, y, zero)?;
    let le_y = ctx.binary(mlx::mlx_less_equal, y, max_y)?;
    let ge_x = ctx.binary(mlx::mlx_greater_equal, x_coord, zero)?;
    let le_x = ctx.binary(mlx::mlx_less_equal, x_coord, max_x)?;
    let valid_y = ctx.binary(mlx::mlx_logical_and, ge_y, le_y)?;
    let valid_x = ctx.binary(mlx::mlx_logical_and, ge_x, le_x)?;
    let valid = ctx.binary(mlx::mlx_logical_and, valid_y, valid_x)?;
    let valid = ctx.astype(valid, dt)?;

    let clipped_y = ctx.emit(|res, s| unsafe { mlx::mlx_clip(res, y, zero, max_y, s) })?;
    let clipped_x = ctx.emit(|res, s| unsafe { mlx::mlx_clip(res, x_coord, zero, max_x, s) })?;
    let yi = ctx.astype(clipped_y, mlx::mlx_dtype__MLX_INT32)?;
    let xi = ctx.astype(clipped_x, mlx::mlx_dtype__MLX_INT32)?;
    let width_scalar = ctx.scalar_i32(width);
    let row = ctx.mul(yi, width_scalar)?;
    let spatial = ctx.add(row, xi)?;
    let spatial = ctx.expand_dims(spatial, 1)?;

    let mut bases = Vec::with_capacity((batch * channel_count) as usize);
    for n in 0..batch {
        for c in channel_begin..channel_begin + channel_count {
            bases.push((n * channels + c) * height * width);
        }
    }
    let base = ctx.from_host(
        bases.as_ptr() as *const c_void,
        &[batch, channel_count, 1, 1],
        mlx::mlx_dtype__MLX_INT32,
    );
    let indices = ctx.add(base, spatial)?;
    let flat = ctx.reshape(x, &[batch * channels * height * width])?;
    let gathered = ctx.emit(|res, s| unsafe { mlx::mlx_take(res, flat, indices, s) })?;
    let valid = ctx.expand_dims(valid, 1)?;
    let valid = ctx.emit(|res, s| unsafe {
        mlx::mlx_broadcast_to(
            res,
            valid,
            [batch, channel_count, out_h, out_w].as_ptr(),
            4,
            s,
        )
    })?;
    ctx.mul(gathered, valid)
}

fn deform_bilinear_sample(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    y: mlx::mlx_array,
    x_coord: mlx::mlx_array,
    channel_begin: i32,
    channel_count: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let dt = ctx.dtype_of(x);
    let one = scalar_for_dtype(ctx, 1.0, dt)?;
    let y0 = ctx.unary(mlx::mlx_floor, y)?;
    let x0 = ctx.unary(mlx::mlx_floor, x_coord)?;
    let y1 = ctx.add(y0, one)?;
    let x1 = ctx.add(x0, one)?;
    let wy1 = ctx.sub(y, y0)?;
    let wx1 = ctx.sub(x_coord, x0)?;
    let wy0 = ctx.sub(one, wy1)?;
    let wx0 = ctx.sub(one, wx1)?;

    let v00 = deform_take(ctx, x, y0, x0, channel_begin, channel_count)?;
    let v01 = deform_take(ctx, x, y0, x1, channel_begin, channel_count)?;
    let v10 = deform_take(ctx, x, y1, x0, channel_begin, channel_count)?;
    let v11 = deform_take(ctx, x, y1, x1, channel_begin, channel_count)?;
    let w00 = ctx.mul(wy0, wx0)?;
    let w01 = ctx.mul(wy0, wx1)?;
    let w10 = ctx.mul(wy1, wx0)?;
    let w11 = ctx.mul(wy1, wx1)?;
    let w00 = ctx.expand_dims(w00, 1)?;
    let w01 = ctx.expand_dims(w01, 1)?;
    let w10 = ctx.expand_dims(w10, 1)?;
    let w11 = ctx.expand_dims(w11, 1)?;
    let p00 = ctx.mul(v00, w00)?;
    let p01 = ctx.mul(v01, w01)?;
    let p10 = ctx.mul(v10, w10)?;
    let p11 = ctx.mul(v11, w11)?;
    let top = ctx.add(p00, p01)?;
    let bottom = ctx.add(p10, p11)?;
    ctx.add(top, bottom)
}

fn deform_conv_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let output_dtype = ctx.dtype_of(x);
    // Coordinates must not lose integer or fractional bits in fp16/bf16 before gathering.
    let x = ctx.astype(x, mlx::mlx_dtype__MLX_FLOAT32)?;
    let weight = ctx.resolve(&n.inputs[1])?;
    let weight = ctx.astype(weight, mlx::mlx_dtype__MLX_FLOAT32)?;
    let offset = ctx.resolve(&n.inputs[2])?;
    let offset = ctx.astype(offset, mlx::mlx_dtype__MLX_FLOAT32)?;
    let x_shape = ctx.shape_of(x);
    let w_shape = ctx.shape_of(weight);
    let offset_shape = ctx.shape_of(offset);
    let (batch, channels, out_h, out_w) =
        (x_shape[0], x_shape[1], offset_shape[2], offset_shape[3]);
    let (out_channels, kernel_h, kernel_w) = (w_shape[0], w_shape[2], w_shape[3]);
    let kernel_size = kernel_h * kernel_w;
    let group = n.ints.get("group").copied().unwrap_or(1) as i32;
    let offset_group = n.ints.get("offset_group").copied().unwrap_or(1) as i32;
    let strides = attr_or(n, "strides", 2, 1);
    let pads = attr_or(n, "pads", 4, 0);
    let dilations = attr_or(n, "dilations", 2, 1);
    let dt = ctx.dtype_of(x);

    let mut base_h = Vec::with_capacity((out_h * out_w) as usize);
    let mut base_w = Vec::with_capacity((out_h * out_w) as usize);
    for oh in 0..out_h {
        for ow in 0..out_w {
            base_h.push((oh as i64 * strides[0] - pads[0]) as f32);
            base_w.push((ow as i64 * strides[1] - pads[1]) as f32);
        }
    }
    let base_h = ctx.from_host(
        base_h.as_ptr() as *const c_void,
        &[1, out_h, out_w],
        mlx::mlx_dtype__MLX_FLOAT32,
    );
    let base_w = ctx.from_host(
        base_w.as_ptr() as *const c_void,
        &[1, out_h, out_w],
        mlx::mlx_dtype__MLX_FLOAT32,
    );
    let base_h = ctx.astype(base_h, dt)?;
    let base_w = ctx.astype(base_w, dt)?;
    let channels_per_offset_group = channels / offset_group;
    let mut kernel_samples = Vec::with_capacity(kernel_size as usize);

    for kh in 0..kernel_h {
        for kw in 0..kernel_w {
            let kernel_index = kh * kernel_w + kw;
            let kh_offset = scalar_for_dtype(ctx, (kh as i64 * dilations[0]) as f32, dt)?;
            let kw_offset = scalar_for_dtype(ctx, (kw as i64 * dilations[1]) as f32, dt)?;
            let kernel_base_h = ctx.add(base_h, kh_offset)?;
            let kernel_base_w = ctx.add(base_w, kw_offset)?;
            let mut offset_group_samples = Vec::with_capacity(offset_group as usize);
            for og in 0..offset_group {
                let offset_channel = (og * kernel_size + kernel_index) * 2;
                let off_h = ctx.slice(
                    offset,
                    &[0, offset_channel, 0, 0],
                    &[batch, offset_channel + 1, out_h, out_w],
                )?;
                let off_w = ctx.slice(
                    offset,
                    &[0, offset_channel + 1, 0, 0],
                    &[batch, offset_channel + 2, out_h, out_w],
                )?;
                let off_h = ctx.squeeze(off_h, 1)?;
                let off_w = ctx.squeeze(off_w, 1)?;
                let sample_h = ctx.add(kernel_base_h, off_h)?;
                let sample_w = ctx.add(kernel_base_w, off_w)?;
                let mut sampled = deform_bilinear_sample(
                    ctx,
                    x,
                    sample_h,
                    sample_w,
                    og * channels_per_offset_group,
                    channels_per_offset_group,
                )?;
                if present(n, 4) {
                    let mask = ctx.resolve(&n.inputs[4])?;
                    let mask = ctx.astype(mask, mlx::mlx_dtype__MLX_FLOAT32)?;
                    let mask_channel = og * kernel_size + kernel_index;
                    let mask = ctx.slice(
                        mask,
                        &[0, mask_channel, 0, 0],
                        &[batch, mask_channel + 1, out_h, out_w],
                    )?;
                    sampled = ctx.mul(sampled, mask)?;
                }
                offset_group_samples.push(sampled);
            }
            let mut sampled = offset_group_samples[0];
            for &part in &offset_group_samples[1..] {
                sampled = ctx.concat2(sampled, part, 1)?;
            }
            kernel_samples.push(sampled);
        }
    }

    let columns = ctx.stack(&kernel_samples, 2)?;
    let channels_per_group = channels / group;
    let outputs_per_group = out_channels / group;
    let mut group_outputs = Vec::with_capacity(group as usize);
    for g in 0..group {
        let columns = ctx.slice(
            columns,
            &[0, g * channels_per_group, 0, 0, 0],
            &[
                batch,
                (g + 1) * channels_per_group,
                kernel_size,
                out_h,
                out_w,
            ],
        )?;
        let columns = ctx.transpose(columns, &[0, 3, 4, 1, 2])?;
        let columns = ctx.contiguous(columns)?;
        let columns = ctx.reshape(
            columns,
            &[batch * out_h * out_w, channels_per_group * kernel_size],
        )?;
        let group_weight = ctx.slice(
            weight,
            &[g * outputs_per_group, 0, 0, 0],
            &[
                (g + 1) * outputs_per_group,
                channels_per_group,
                kernel_h,
                kernel_w,
            ],
        )?;
        let group_weight = ctx.reshape(
            group_weight,
            &[outputs_per_group, channels_per_group * kernel_size],
        )?;
        let group_weight = ctx.transpose(group_weight, &[1, 0])?;
        let output = ctx.matmul(columns, group_weight)?;
        let output = ctx.reshape(output, &[batch, out_h, out_w, outputs_per_group])?;
        group_outputs.push(ctx.transpose(output, &[0, 3, 1, 2])?);
    }
    let mut output = group_outputs[0];
    for &part in &group_outputs[1..] {
        output = ctx.concat2(output, part, 1)?;
    }
    if present(n, 3) {
        let bias = ctx.resolve(&n.inputs[3])?;
        let bias = ctx.astype(bias, mlx::mlx_dtype__MLX_FLOAT32)?;
        let bias = ctx.reshape(bias, &[1, out_channels, 1, 1])?;
        output = ctx.add(output, bias)?;
    }
    output = ctx.astype(output, output_dtype)?;
    output = ctx.contiguous(output)?;
    ctx.bind(&n.outputs[0], output);
    Ok(())
}

// ---- ConvTranspose ------------------------------------------------------------------------------

fn transpose_pads(
    auto_pad: &str,
    input: &[i64],
    kernel: &[i64],
    strides: &[i64],
    dilations: &[i64],
    output_padding: &[i64],
    output_shape: Option<&[i64]>,
    explicit_pads: &[i64],
) -> Result<Vec<i64>, MlxError> {
    let rank = input.len();
    if output_shape.is_none() && matches!(auto_pad, "NOTSET" | "VALID") {
        return Ok(if auto_pad == "VALID" {
            vec![0; rank * 2]
        } else {
            explicit_pads.to_vec()
        });
    }
    let mut pads = vec![0; rank * 2];
    for i in 0..rank {
        let desired = output_shape.map_or(input[i] * strides[i], |shape| shape[i]);
        let total =
            strides[i] * (input[i] - 1) + output_padding[i] + dilations[i] * (kernel[i] - 1) + 1
                - desired;
        if total < 0 {
            return Err("ConvTranspose: requested output requires negative total padding".into());
        }
        pads[i] = if auto_pad == "SAME_UPPER" {
            total / 2
        } else {
            total - total / 2
        };
        pads[i + rank] = total - pads[i];
    }
    Ok(pads)
}

fn conv_transpose_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x0 = ctx.resolve(&n.inputs[0])?;
    let xs = ctx.shape_of(x0);
    let rank = xs.len() - 2;
    if xs[1..].iter().any(|&dim| dim <= 0) {
        return Err("ConvTranspose: channel and spatial dimensions must be positive".into());
    }
    let strides = attr_or(n, "strides", rank, 1);
    let dilations = attr_or(n, "dilations", rank, 1);
    let output_padding = attr_or(n, "output_padding", rank, 0);
    let group = n.ints.get("group").copied().unwrap_or(1) as i32;
    let x = to_channels_last(ctx, x0, rank as i32)?;
    let w0 = ctx.resolve(&n.inputs[1])?;
    let ws = ctx.shape_of(w0);
    if xs[1] != ws[0] {
        return Err("ConvTranspose: W input channels must match X".into());
    }
    let pads = transpose_pads(
        n.strings.get("auto_pad").map_or("NOTSET", String::as_str),
        &xs[2..].iter().map(|&v| i64::from(v)).collect::<Vec<_>>(),
        &ws[2..].iter().map(|&v| i64::from(v)).collect::<Vec<_>>(),
        &strides,
        &dilations,
        &output_padding,
        n.int_arrays.get("output_shape").map(Vec::as_slice),
        &attr_or(n, "pads", 2 * rank, 0),
    )?;
    let mut grouped_shape = vec![group, ws[0] / group, ws[1]];
    grouped_shape.extend_from_slice(&ws[2..]);
    let w = ctx.reshape(w0, &grouped_shape)?;
    let mut axes = vec![0, 2];
    axes.extend(3..rank as i32 + 3);
    axes.push(1);
    let w = ctx.transpose(w, &axes)?;
    let mut weight_shape = vec![ws[1] * group];
    weight_shape.extend_from_slice(&ws[2..]);
    weight_shape.push(ws[0] / group);
    let w = ctx.contiguous(w)?;
    let weight = ctx.reshape(w, &weight_shape)?;

    // Compute the uncropped transpose, then crop output rather than using negative conv padding
    // (which would slice the undilated input and lose stride-aligned samples).
    let low: Vec<i32> = (0..rank)
        .map(|i| (dilations[i] * (i64::from(ws[i + 2]) - 1)) as i32)
        .collect();
    let high: Vec<i32> = (0..rank)
        .map(|i| low[i] + output_padding[i] as i32)
        .collect();
    let kernel_dilation: Vec<i32> = dilations.iter().map(|&v| v as i32).collect();
    let input_dilation: Vec<i32> = strides.iter().map(|&v| v as i32).collect();
    let unit = vec![1; rank];
    let mut out = ctx.emit(|res, s| unsafe {
        mlx::mlx_conv_general(
            res,
            x,
            weight,
            unit.as_ptr(),
            rank,
            low.as_ptr(),
            rank,
            high.as_ptr(),
            rank,
            kernel_dilation.as_ptr(),
            rank,
            input_dilation.as_ptr(),
            rank,
            group,
            true,
            s,
        )
    })?;
    let mut start = vec![0; rank + 2];
    let mut stop = ctx.shape_of(out);
    for i in 0..rank {
        start[i + 1] = i32::try_from(pads[i]).map_err(|_| "ConvTranspose: pad exceeds int32")?;
        stop[i + 1] -=
            i32::try_from(pads[i + rank]).map_err(|_| "ConvTranspose: pad exceeds int32")?;
        if stop[i + 1] <= start[i + 1] {
            return Err("ConvTranspose: output spatial dimensions must be positive".into());
        }
    }
    out = ctx.slice(out, &start, &stop)?;

    if present(n, 2) {
        let bias = ctx.resolve(&n.inputs[2])?;
        out = add(ctx, out, bias)?;
    }
    let y = from_channels_last(ctx, out, rank as i32)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

// ---- shared pooling primitives (channels-last) -------------------------------------------------

#[derive(Debug)]
struct PoolGeometry {
    kernel: Vec<i64>,
    strides: Vec<i64>,
    dilations: Vec<i64>,
    pads: Vec<i64>,
    tail: Vec<i64>,
    output: Vec<i32>,
}

impl PoolGeometry {
    fn new(
        spatial: &[i64],
        kernel: Vec<i64>,
        strides: Vec<i64>,
        dilations: Vec<i64>,
        mut pads: Vec<i64>,
        auto_pad: &str,
        ceil: bool,
    ) -> Result<Self, MlxError> {
        let r = spatial.len();
        if !(1..=3).contains(&r)
            || kernel.len() != r
            || strides.len() != r
            || dilations.len() != r
            || pads.len() != 2 * r
            || spatial.iter().any(|&v| v <= 0 || v > i32::MAX as i64)
            || kernel
                .iter()
                .chain(&strides)
                .chain(&dilations)
                .any(|&v| v <= 0 || v > i32::MAX as i64)
            || pads.iter().any(|&v| v < 0 || v > i32::MAX as i64)
        {
            return Err("Pool: invalid spatial dimensions or window attributes".into());
        }
        if auto_pad != "NOTSET" {
            pads = auto_pad_pads(auto_pad, spatial, &kernel, &strides, &dilations)
                .ok_or_else(|| format!("Pool: unsupported auto_pad {auto_pad:?}"))?;
        }
        let mut output = Vec::with_capacity(r);
        let mut tail = Vec::with_capacity(r);
        for i in 0..r {
            // Attribute components fit i32; their products and sums fit i64.
            let effective = dilations[i] * (kernel[i] - 1) + 1;
            let padded = spatial[i] + pads[i] + pads[i + r];
            let delta = padded - effective;
            let mut count = if matches!(auto_pad, "SAME_UPPER" | "SAME_LOWER") {
                (spatial[i] + strides[i] - 1) / strides[i]
            } else {
                delta.div_euclid(strides[i])
                    + i64::from(ceil && delta.rem_euclid(strides[i]) != 0)
                    + 1
            };
            if ceil && count > 0 && (count - 1) * strides[i] >= spatial[i] + pads[i] {
                count -= 1;
            }
            let extra = if count > 0 {
                ((count - 1) * strides[i] + effective - padded).max(0)
            } else {
                0
            };
            if count < 0 || count > i32::MAX as i64 || padded + extra > i32::MAX as i64 {
                return Err("Pool: output or padded extent is not representable by MLX".into());
            }
            output.push(count as i32);
            tail.push(extra);
        }
        Ok(Self {
            kernel,
            strides,
            dilations,
            pads,
            tail,
            output,
        })
    }

    fn materialized_pads(&self) -> Vec<i64> {
        let mut pads = self.pads.clone();
        let r = self.output.len();
        for i in 0..r {
            pads[r + i] += self.tail[i];
        }
        pads
    }

    fn axes(&self) -> Vec<i32> {
        let r = self.output.len() as i32;
        (r + 1..2 * r + 1).collect()
    }
}

fn pad_spatial(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    pads: &[i64],
    value: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    if pads.iter().all(|&p| p == 0) {
        return Ok(x);
    }
    let r = pads.len() / 2;
    let axes: Vec<i32> = (1..=r as i32).collect();
    let low: Vec<i32> = pads[..r].iter().map(|&v| v as i32).collect();
    let high: Vec<i32> = pads[r..].iter().map(|&v| v as i32).collect();
    let mode = b"constant\0";
    let out = ctx.emit(|res, s| unsafe {
        mlx::mlx_pad(
            res,
            x,
            axes.as_ptr(),
            r,
            low.as_ptr(),
            r,
            high.as_ptr(),
            r,
            value,
            mode.as_ptr() as *const c_char,
            s,
        )
    })?;
    ctx.contiguous(out)
}

/// Build [N, out..., kernel..., C] over a contiguous, fully materialized padded input.
fn sliding_windows(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    geometry: &PoolGeometry,
) -> Result<mlx::mlx_array, MlxError> {
    let shape = ctx.shape_of(x);
    let r = geometry.output.len();
    let mut contiguous = vec![1i64; r + 2];
    for i in (0..r + 1).rev() {
        contiguous[i] = contiguous[i + 1]
            .checked_mul(i64::from(shape[i + 1]))
            .ok_or("Pool: padded strides overflow int64")?;
    }
    let mut window_shape = vec![shape[0]];
    window_shape.extend(&geometry.output);
    window_shape.extend(geometry.kernel.iter().map(|&v| v as i32));
    window_shape.push(shape[r + 1]);
    window_shape.iter().try_fold(1i64, |size, &dim| {
        size.checked_mul(i64::from(dim))
            .ok_or("Pool: window element count overflows int64")
    })?;
    let mut window_strides = vec![contiguous[0]];
    for i in 0..r {
        window_strides.push(
            contiguous[i + 1]
                .checked_mul(geometry.strides[i])
                .ok_or("Pool: window strides overflow int64")?,
        );
    }
    for i in 0..r {
        window_strides.push(
            contiguous[i + 1]
                .checked_mul(geometry.dilations[i])
                .ok_or("Pool: dilated strides overflow int64")?,
        );
    }
    window_strides.push(1);
    let windows = ctx.emit(|res, s| unsafe {
        mlx::mlx_as_strided(
            res,
            x,
            window_shape.as_ptr(),
            window_shape.len(),
            window_strides.as_ptr(),
            window_strides.len(),
            0,
            s,
        )
    })?;
    // MLX 0.32.4's strided reduction misreads overlapping dilated views
    // (e.g. [N,3,3,C] with spatial strides [C,2*C]). Materialize new
    // dilated forms; keep the existing undilated reduction fast path.
    if geometry.dilations.iter().any(|&d| d != 1) {
        ctx.contiguous(windows)
    } else {
        Ok(windows)
    }
}

fn pool_sum(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    axes: &[i32],
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_sum_axes(res, a, axes.as_ptr(), axes.len(), false, s) })
}

fn pool_mask_windows(
    ctx: &mut TranslationContext,
    spatial: &[i64],
    geometry: &PoolGeometry,
    include_pad: bool,
) -> Result<mlx::mlx_array, MlxError> {
    let r = spatial.len();
    let mut shape = vec![1];
    for (i, &dim) in spatial.iter().enumerate() {
        let padding = if include_pad {
            geometry.pads[i] + geometry.pads[r + i]
        } else {
            0
        };
        shape.push((dim + padding) as i32);
    }
    shape.push(1);
    let mask = ctx.emit(|res, s| unsafe {
        mlx::mlx_ones(
            res,
            shape.as_ptr(),
            shape.len(),
            mlx::mlx_dtype__MLX_FLOAT32,
            s,
        )
    })?;
    let zero = ctx.scalar_f32(0.0);
    // Ceil-only extension is not declared ONNX padding, even when count_include_pad=1.
    let pads = if include_pad {
        let mut pads = vec![0; 2 * r];
        pads[r..].copy_from_slice(&geometry.tail);
        pads
    } else {
        geometry.materialized_pads()
    };
    let mask = pad_spatial(ctx, mask, &pads, zero)?;
    sliding_windows(ctx, mask, geometry)
}

fn pool_counts(
    ctx: &mut TranslationContext,
    spatial: &[i64],
    geometry: &PoolGeometry,
    include_pad: bool,
) -> Result<mlx::mlx_array, MlxError> {
    let windows = pool_mask_windows(ctx, spatial, geometry, include_pad)?;
    pool_sum(ctx, windows, &geometry.axes())
}

fn pool_indices(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    shape: &[i32],
    windows: mlx::mlx_array,
    maxima: mlx::mlx_array,
    geometry: &PoolGeometry,
) -> Result<(), MlxError> {
    let rank = shape.len();
    let r = rank - 2;
    let mut indices = ctx.scalar_i64(0);
    let mut stride = 1i64;
    for axis in (0..rank).rev() {
        let coord = ctx.arange(0.0, f64::from(shape[axis]), 1.0, mlx::mlx_dtype__MLX_INT64)?;
        let mut broadcast = vec![1; rank];
        broadcast[axis] = shape[axis];
        let coord = ctx.reshape(coord, &broadcast)?;
        let scale = ctx.scalar_i64(stride);
        let coord = ctx.binary(mlx::mlx_multiply, coord, scale)?;
        indices = ctx.binary(mlx::mlx_add, indices, coord)?;
        stride = stride
            .checked_mul(i64::from(shape[axis]))
            .ok_or("MaxPool: original flattened indices overflow int64")?;
    }
    let indices = to_channels_last(ctx, indices, r as i32)?;
    let sentinel = ctx.scalar_i64(i64::MAX);
    let indices = pad_spatial(ctx, indices, &geometry.materialized_pads(), sentinel)?;
    let indices = sliding_windows(ctx, indices, geometry)?;
    let mut expanded = ctx.shape_of(maxima);
    for _ in 0..r {
        expanded.insert(r + 1, 1);
    }
    let maxima = ctx.reshape(maxima, &expanded)?;
    let equal = ctx.binary(mlx::mlx_equal, windows, maxima)?;
    let candidates = ctx.where_(equal, indices, sentinel)?;
    let axes = geometry.axes();
    // Min ORIGINAL row-major index implements the first valid window element on ties,
    // independently of the requested output index encoding and padded sentinel values.
    let selected = ctx.emit(|res, s| unsafe {
        mlx::mlx_min_axes(res, candidates, axes.as_ptr(), axes.len(), false, s)
    })?;
    let selected = from_channels_last(ctx, selected, r as i32)?;
    let valid = ctx.binary(mlx::mlx_not_equal, selected, sentinel)?;
    let zero = ctx.scalar_i64(0);
    let selected = ctx.where_(valid, selected, zero)?;
    let encoded = if n.ints.get("storage_order").copied().unwrap_or(0) == 1 {
        let spatial_size = shape[2..].iter().try_fold(1i64, |size, &v| {
            size.checked_mul(i64::from(v))
                .ok_or("MaxPool: spatial size overflows int64")
        })?;
        let size = ctx.scalar_i64(spatial_size);
        let batch_channel = ctx.binary(mlx::mlx_floor_divide, selected, size)?;
        let mut encoded = ctx.binary(mlx::mlx_multiply, batch_channel, size)?;
        let mut trailing = spatial_size;
        let mut leading = 1i64;
        for &dim in &shape[2..] {
            trailing /= i64::from(dim);
            let divisor = ctx.scalar_i64(trailing);
            let coord = ctx.binary(mlx::mlx_floor_divide, selected, divisor)?;
            let dimension = ctx.scalar_i64(i64::from(dim));
            let coord = ctx.binary(mlx::mlx_remainder, coord, dimension)?;
            let scale = ctx.scalar_i64(leading);
            let coord = ctx.binary(mlx::mlx_multiply, coord, scale)?;
            encoded = ctx.binary(mlx::mlx_add, encoded, coord)?;
            leading *= i64::from(dim);
        }
        encoded
    } else {
        selected
    };
    // ORT encodes the no-sample sentinel as spatial coordinates (-1,...,-1),
    // plus the original batch/channel offset, rather than a global -1.
    let column_major = n.ints.get("storage_order").copied().unwrap_or(0) == 1;
    let mut spatial_size = 1i64;
    let mut sentinel_offset = 0i64;
    let dimensions: Vec<i32> = if column_major {
        shape[2..].to_vec()
    } else {
        shape[2..].iter().copied().rev().collect()
    };
    for dim in dimensions {
        sentinel_offset = sentinel_offset
            .checked_sub(spatial_size)
            .ok_or("MaxPool: missing index offset overflows int64")?;
        spatial_size = spatial_size
            .checked_mul(i64::from(dim))
            .ok_or("MaxPool: spatial size overflows int64")?;
    }
    let mut missing = ctx.scalar_i64(sentinel_offset);
    let mut scale = spatial_size;
    for axis in (0..2).rev() {
        let coord = ctx.arange(0.0, f64::from(shape[axis]), 1.0, mlx::mlx_dtype__MLX_INT64)?;
        let mut broadcast = vec![1; rank];
        broadcast[axis] = shape[axis];
        let coord = ctx.reshape(coord, &broadcast)?;
        let factor = ctx.scalar_i64(scale);
        let coord = ctx.binary(mlx::mlx_multiply, coord, factor)?;
        missing = ctx.binary(mlx::mlx_add, missing, coord)?;
        scale = scale
            .checked_mul(i64::from(shape[axis]))
            .ok_or("MaxPool: original size overflows int64")?;
    }
    let encoded = ctx.where_(valid, encoded, missing)?;
    ctx.bind(&n.outputs[1], encoded);
    Ok(())
}

fn pool_op(ctx: &mut TranslationContext, n: &NodeDesc, average: bool) -> Result<(), MlxError> {
    let x0 = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(x0);
    if !(3..=5).contains(&shape.len()) {
        return Err("Pool: input must have 1-3 spatial dimensions".into());
    }
    if shape[0] != 0 && shape[1] == 0 {
        return Err("Pool: only the batch dimension may be zero".into());
    }
    let r = shape.len() - 2;
    let spatial: Vec<i64> = shape[2..].iter().map(|&v| i64::from(v)).collect();
    let geometry = PoolGeometry::new(
        &spatial,
        n.int_arrays
            .get("kernel_shape")
            .cloned()
            .unwrap_or_default(),
        attr_or(n, "strides", r, 1),
        attr_or(n, "dilations", r, 1),
        attr_or(n, "pads", 2 * r, 0),
        n.strings
            .get("auto_pad")
            .map(String::as_str)
            .unwrap_or("NOTSET"),
        n.ints.get("ceil_mode").copied().unwrap_or(0) == 1,
    )?;
    let axes = geometry.axes();
    let lp = n.op_type == "LpPool";
    let original_dt = ctx.dtype_of(x0);
    if shape[0] == 0 || shape[1] == 0 || geometry.output.contains(&0) {
        let mut output_shape = shape[..2].to_vec();
        output_shape.extend(&geometry.output);
        for (i, output) in n
            .outputs
            .iter()
            .enumerate()
            .filter(|(_, o)| !o.name.is_empty())
        {
            let dtype = if i == 0 {
                original_dt
            } else {
                mlx::mlx_dtype__MLX_INT64
            };
            let empty = ctx.emit(|res, s| unsafe {
                mlx::mlx_zeros(res, output_shape.as_ptr(), output_shape.len(), dtype, s)
            })?;
            ctx.bind(output, empty);
        }
        return Ok(());
    }
    let x = to_channels_last(ctx, x0, r as i32)?;
    let x = if average || lp {
        ctx.astype(x, mlx::mlx_dtype__MLX_FLOAT32)?
    } else {
        x
    };
    let dt = ctx.dtype_of(x);
    let pad_value = if average || lp {
        0.0
    } else {
        match dt {
            mlx::mlx_dtype__MLX_INT8 => -128.0,
            mlx::mlx_dtype__MLX_UINT8 => 0.0,
            _ => f32::NEG_INFINITY,
        }
    };
    let pv = scalar_for_dtype(ctx, pad_value, dt)?;
    let padded = pad_spatial(ctx, x, &geometry.materialized_pads(), pv)?;
    let windows = sliding_windows(ctx, padded, &geometry)?;
    let out = if lp {
        let p = if n.since_version == 1 {
            n.floats.get("p").copied().unwrap_or(2.0)
        } else {
            n.ints.get("p").copied().unwrap_or(2) as f32
        };
        let a = abs_(ctx, windows)?;
        let ps = ctx.scalar_f32(p);
        let mut powered = power(ctx, a, ps)?;
        if p <= 0.0 && geometry.materialized_pads().iter().any(|&v| v != 0) {
            let mask = pool_mask_windows(ctx, &spatial, &geometry, false)?;
            let zero = ctx.scalar_f32(0.0);
            let valid = ctx.binary(mlx::mlx_greater, mask, zero)?;
            powered = ctx.where_(valid, powered, zero)?;
        }
        let summed = pool_sum(ctx, powered, &axes)?;
        let inv = ctx.scalar_f32(1.0 / p);
        power(ctx, summed, inv)?
    } else if average {
        let include_pad = n.ints.get("count_include_pad").copied().unwrap_or(0) != 0;
        let has_padding = geometry.materialized_pads().iter().any(|&p| p != 0);
        if !has_padding {
            let out = ctx.emit(|res, s| unsafe {
                mlx::mlx_mean_axes(res, windows, axes.as_ptr(), axes.len(), false, s)
            })?;
            let out = ctx.astype(out, original_dt)?;
            let y = from_channels_last(ctx, out, r as i32)?;
            ctx.bind(&n.outputs[0], y);
            return Ok(());
        }
        let sums = pool_sum(ctx, windows, &axes)?;
        let counts = pool_counts(ctx, &spatial, &geometry, include_pad)?;
        ctx.binary(mlx::mlx_divide, sums, counts)?
    } else {
        let mut out = ctx.emit(|res, s| unsafe {
            mlx::mlx_max_axes(res, windows, axes.as_ptr(), axes.len(), false, s)
        })?;
        if geometry.materialized_pads().iter().any(|&v| v != 0) {
            let counts = pool_counts(ctx, &spatial, &geometry, false)?;
            let zero = ctx.scalar_f32(0.0);
            let nonempty = ctx.binary(mlx::mlx_greater, counts, zero)?;
            let lowest = match original_dt {
                mlx::mlx_dtype__MLX_INT8 => -128.0,
                mlx::mlx_dtype__MLX_UINT8 => 0.0,
                _ => f32::MIN,
            };
            let lowest = scalar_for_dtype(ctx, lowest, original_dt)?;
            out = ctx.where_(nonempty, out, lowest)?;
        }
        if n.outputs
            .get(1)
            .is_some_and(|output| !output.name.is_empty())
        {
            pool_indices(ctx, n, &shape, windows, out, &geometry)?;
        }
        out
    };
    let out = ctx.astype(out, original_dt)?;
    let y = from_channels_last(ctx, out, r as i32)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn average_pool_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    pool_op(ctx, n, true)
}

fn max_pool_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    pool_op(ctx, n, false)
}

fn global_pool_op(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    average: bool,
) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    if ctx.shape_of(x)[2..].iter().any(|&d| d <= 0) {
        return Err("global Pool: spatial dimensions must be positive".into());
    }
    let axes: Vec<i32> = (2..ctx.ndim(x) as i32).collect();
    let out = if average {
        ctx.emit(|res, s| unsafe {
            mlx::mlx_mean_axes(res, x, axes.as_ptr(), axes.len(), true, s)
        })?
    } else {
        ctx.emit(|res, s| unsafe { mlx::mlx_max_axes(res, x, axes.as_ptr(), axes.len(), true, s) })?
    };
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn global_average_pool_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    global_pool_op(ctx, n, true)
}

fn global_max_pool_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    global_pool_op(ctx, n, false)
}

// ---- LpPool / GlobalLpPool (from normpool.cc) ---------------------------------------------------

fn lp_pool_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    pool_op(ctx, n, false)
}

fn global_lp_pool_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    if ctx.shape_of(x)[2..].iter().any(|&d| d <= 0) {
        return Err("GlobalLpPool: spatial dimensions must be positive".into());
    }
    let rank = ctx.ndim(x) as i32;
    let p = n.ints.get("p").copied().unwrap_or(2) as f32;
    let dt = ctx.dtype_of(x);
    let axes: Vec<i32> = (2..rank).collect();

    let a = abs_(ctx, x)?;
    let ps = scalar_for_dtype(ctx, p, dt)?;
    let powered = power(ctx, a, ps)?;
    let summed = ctx.emit(|res, s| unsafe {
        mlx::mlx_sum_axes(res, powered, axes.as_ptr(), axes.len(), true, s)
    })?;
    let inv = scalar_for_dtype(ctx, 1.0 / p, dt)?;
    let out = power(ctx, summed, inv)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

// ---- claim helpers (port of op_claim.h + conv.cc/normpool.cc local helpers) ----------------------

fn read_spatial_attribute(
    node: &NodeView,
    name: &str,
    spatial_rank: usize,
    default: i64,
) -> Option<Vec<i64>> {
    let (present, mut values) = node.ints_attr(name);
    if !present {
        values = vec![default; spatial_rank];
    }
    if values.len() != spatial_rank || !values.iter().all(|&v| v > 0 && v <= i32::MAX as i64) {
        return None;
    }
    Some(values)
}

fn read_pads(node: &NodeView, spatial_rank: usize) -> Option<Vec<i64>> {
    let (present, mut pads) = node.ints_attr("pads");
    if !present {
        pads = vec![0; 2 * spatial_rank];
    }
    if pads.len() != 2 * spatial_rank || !pads.iter().all(|&v| v >= 0 && v <= i32::MAX as i64) {
        return None;
    }
    Some(pads)
}

/// Resolve an ONNX `auto_pad` mode to explicit begin/end pads from statically known spatial dims.
/// Returns ONNX pad layout `[begin_0..begin_{r-1}, end_0..end_{r-1}]`, or `None` for an
/// unrecognized mode. `VALID` yields zero pads; `SAME_UPPER`/`SAME_LOWER` size the total pad so the
/// output is `ceil(input/stride)`, putting the extra unit of an odd total at the end (`UPPER`) or
/// the begin (`LOWER`). Asymmetric SAME_* pads are fine — `conv_op` routes them through
/// `mlx_conv_general` exactly like explicit asymmetric pads.
fn auto_pad_pads(
    auto_pad: &str,
    in_spatial: &[i64],
    kernel: &[i64],
    strides: &[i64],
    dilations: &[i64],
) -> Option<Vec<i64>> {
    let r = in_spatial.len();
    match auto_pad {
        "VALID" => Some(vec![0; 2 * r]),
        "SAME_UPPER" | "SAME_LOWER" => {
            let mut begin = vec![0i64; r];
            let mut end = vec![0i64; r];
            for i in 0..r {
                let eff_k = dilations[i] * (kernel[i] - 1) + 1;
                let out = (in_spatial[i] + strides[i] - 1) / strides[i]; // ceil(in/stride)
                let total = ((out - 1) * strides[i] + eff_k - in_spatial[i]).max(0);
                if auto_pad == "SAME_LOWER" {
                    end[i] = total / 2;
                    begin[i] = total - end[i];
                } else {
                    begin[i] = total / 2;
                    end[i] = total - begin[i];
                }
            }
            begin.extend(end);
            Some(begin)
        }
        _ => None,
    }
}

fn static_positive_shape(shape: &[i64], rank: usize) -> bool {
    shape.len() == rank && shape.iter().all(|&d| d > 0 && d <= i32::MAX as i64)
}

/// Compare an actual (ONNX-declared) shape against an expected shape, treating any negative
/// (dynamic / symbolic) dim on EITHER side as a wildcard. Dynamic spatial convs declare `-1` output
/// dims and we cannot precompute the expected spatial extent (pads may depend on the runtime shape),
/// so both sides carry wildcards that must be skipped.
fn same_known_shape(actual: &[i64], expected: &[i64]) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    actual
        .iter()
        .zip(expected)
        .all(|(&a, &e)| a < 0 || e < 0 || a == e)
}

fn optional_bias_is_valid(
    node: &NodeView,
    dtype: crate::sys::ort::ONNXTensorElementDataType,
    channels: i64,
) -> bool {
    if !node.input_present(2) {
        return true;
    }
    match node.input_info(2) {
        Some(info) => info.dtype == dtype && info.shape == vec![channels],
        None => false,
    }
}

// ---- claim predicates ---------------------------------------------------------------------------

fn conv_claim(node: &NodeView) -> ClaimResult {
    let ni = node.num_inputs();
    require!(
        (2..=3).contains(&ni) && node.num_outputs() == 1,
        "expects 2 or 3 inputs and 1 output, got {}in/{}out",
        ni,
        node.num_outputs()
    );
    let (x, w, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(w), Some(o)) => (x, w, o),
        _ => deny!("missing tensor type/shape info on X, W, or output"),
    };
    require!(
        is_mlx_float(x.dtype) && w.dtype == x.dtype && out.dtype == x.dtype,
        "X/W/output must share one float dtype (fp32/fp16/bf16), got {}, {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(w.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        (3..=5).contains(&x.shape.len()),
        "only Conv1D/2D/3D are supported (got X rank {}); dynamic batch/spatial dims are supported",
        x.shape.len()
    );
    let spatial_rank = x.shape.len() - 2;
    require!(
        static_positive_shape(&w.shape, spatial_rank + 2),
        "W must have rank {} with all dims statically known and positive (shape {:?})",
        spatial_rank + 2,
        w.shape
    );
    require!(
        out.shape.len() == spatial_rank + 2,
        "output rank must be {} to match X (got rank {})",
        spatial_rank + 2,
        out.shape.len()
    );
    let spatial_dynamic = (0..spatial_rank).any(|i| x.shape[i + 2] <= 0);
    require!(
        node.string_attr("auto_pad", "NOTSET") == "NOTSET" || !node.ints_attr("pads").0,
        "auto_pad and explicit pads cannot both be specified"
    );
    let strides = match read_spatial_attribute(node, "strides", spatial_rank, 1) {
        Some(v) => v,
        None => deny!(
            "strides must contain {} positive spatial values",
            spatial_rank
        ),
    };
    let dilations = match read_spatial_attribute(node, "dilations", spatial_rank, 1) {
        Some(v) => v,
        None => deny!(
            "dilations must contain {} positive spatial values",
            spatial_rank
        ),
    };
    // `auto_pad` (SAME_UPPER/SAME_LOWER/VALID) resolves to explicit pads from the static input
    // spatial shape; NOTSET reads the `pads` attribute. SAME_* pads may be asymmetric — `conv_op`
    // routes those through `mlx_conv_general` just like explicit asymmetric pads.
    let auto_pad = node.string_attr("auto_pad", "NOTSET");
    let pads: Option<Vec<i64>> = if auto_pad == "NOTSET" {
        // Explicit pads are static regardless of dynamic spatial dims.
        match read_pads(node, spatial_rank) {
            Some(v) => Some(v),
            None => deny!("pads must contain {} non-negative values", spatial_rank * 2),
        }
    } else if spatial_dynamic {
        // auto_pad + dynamic spatial: pads depend on the runtime spatial extent, so they cannot be
        // precomputed here. `conv_op` resolves them at trace time from the concrete `ctx.shape_of`.
        None
    } else {
        let in_sp: Vec<i64> = (0..spatial_rank).map(|i| x.shape[i + 2]).collect();
        let kernel: Vec<i64> = (0..spatial_rank).map(|i| w.shape[i + 2]).collect();
        match auto_pad_pads(&auto_pad, &in_sp, &kernel, &strides, &dilations) {
            Some(v) => Some(v),
            None => deny!("unsupported auto_pad value {auto_pad:?}"),
        }
    };
    // Asymmetric pads (`pads[i] != pads[i + spatial_rank]`) are supported: `conv_op` routes them
    // through `mlx_conv_general`, which takes separate `padding_lo`/`padding_hi` vectors. Only the
    // non-negativity checked in `read_pads` is required.
    let (kernel_present, kernel_shape) = node.ints_attr("kernel_shape");
    if kernel_present {
        require!(
            kernel_shape.len() == spatial_rank,
            "kernel_shape must contain {} values (got {:?})",
            spatial_rank,
            kernel_shape
        );
        for i in 0..spatial_rank {
            require!(
                kernel_shape[i] == w.shape[i + 2],
                "kernel_shape {:?} must match W spatial shape {:?}",
                kernel_shape,
                &w.shape[2..]
            );
        }
    }
    let group = node.int_attr("group", 1);
    let out_channels = w.shape[0];
    require!(
        group > 0 && group <= i32::MAX as i64,
        "group must be positive and fit int32 (got {group})"
    );
    require!(
        spatial_rank != 3 || group == 1,
        "MLX Conv3D does not support groups"
    );
    let Some(channels) = w.shape[1].checked_mul(group) else {
        deny!("W channels times group overflows int64");
    };
    require!(
        out_channels % group == 0,
        "group {group} must divide output channels {out_channels}"
    );
    require!(
        x.shape[1] <= 0 || x.shape[1] == channels,
        "X channels {} must equal W channels {} times group {group}",
        x.shape[1],
        w.shape[1]
    );
    require!(
        optional_bias_is_valid(node, x.dtype, out_channels),
        "optional bias must have dtype {} and shape [{}]",
        crate::registry::ort_dtype_name(x.dtype),
        out_channels
    );
    let mut expected = vec![x.shape[0], out_channels];
    for i in 0..spatial_rank {
        let in_dim = x.shape[i + 2];
        match &pads {
            // Static input spatial dim with resolved pads → precompute the expected output extent.
            Some(pads) if in_dim > 0 => {
                let effective_kernel = dilations[i] * (w.shape[i + 2] - 1) + 1;
                let padded = in_dim + pads[i] + pads[i + spatial_rank];
                require!(
                    padded >= effective_kernel,
                    "padded spatial dim {i} ({padded}) is smaller than effective kernel {effective_kernel}"
                );
                expected.push((padded - effective_kernel) / strides[i] + 1);
            }
            // Dynamic spatial dim (or unresolved auto_pad pads) → the output extent is dynamic too;
            // push a wildcard so `same_known_shape` skips it. The handler resolves it at trace time.
            _ => expected.push(-1),
        }
    }
    require!(
        same_known_shape(&out.shape, &expected),
        "output shape {:?} does not match expected {:?} (dynamic batch/spatial dims are wildcards)",
        out.shape,
        expected
    );
    Ok(())
}

fn deform_conv_claim(node: &NodeView) -> ClaimResult {
    require!(
        (3..=5).contains(&node.num_inputs()) && node.num_outputs() == 1,
        "expects 3 to 5 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, w, offset, out) = match (
        node.input_info(0),
        node.input_info(1),
        node.input_info(2),
        node.output_info(0),
    ) {
        (Some(x), Some(w), Some(offset), Some(out)) => (x, w, offset, out),
        _ => deny!("missing tensor type/shape info on X, W, offset, or output"),
    };
    require!(
        is_mlx_float(x.dtype)
            && w.dtype == x.dtype
            && offset.dtype == x.dtype
            && out.dtype == x.dtype,
        "X/W/offset/output must share one float dtype (fp32/fp16/bf16), got {}, {}, {}, -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(w.dtype),
        crate::registry::ort_dtype_name(offset.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        static_positive_shape(&x.shape, 4),
        "DeformConv requires static positive rank-4 NCHW X (shape {:?})",
        x.shape
    );
    require!(
        static_positive_shape(&w.shape, 4),
        "DeformConv requires static positive rank-4 W (shape {:?})",
        w.shape
    );
    require!(
        static_positive_shape(&offset.shape, 4),
        "DeformConv requires static positive rank-4 offset (shape {:?})",
        offset.shape
    );
    require!(
        static_positive_shape(&out.shape, 4),
        "DeformConv requires static positive rank-4 output (shape {:?})",
        out.shape
    );
    let kernel = match read_spatial_attribute(node, "kernel_shape", 2, 0) {
        Some(v) => v,
        None if !node.ints_attr("kernel_shape").0 => vec![w.shape[2], w.shape[3]],
        None => deny!("kernel_shape must contain 2 positive values"),
    };
    require!(
        kernel == w.shape[2..],
        "kernel_shape {:?} must match W spatial shape {:?}",
        kernel,
        &w.shape[2..]
    );
    let strides = match read_spatial_attribute(node, "strides", 2, 1) {
        Some(v) => v,
        None => deny!("strides must contain 2 positive values"),
    };
    let dilations = match read_spatial_attribute(node, "dilations", 2, 1) {
        Some(v) => v,
        None => deny!("dilations must contain 2 positive values"),
    };
    let pads = match read_pads(node, 2) {
        Some(v) => v,
        None => deny!("pads must contain 4 non-negative values"),
    };
    let group = node.int_attr("group", 1);
    let offset_group = node.int_attr("offset_group", 1);
    require!(group > 0, "group must be positive (got {group})");
    require!(
        offset_group > 0,
        "offset_group must be positive (got {offset_group})"
    );
    let channels = x.shape[1];
    let out_channels = w.shape[0];
    require!(
        channels % group == 0 && out_channels % group == 0,
        "group {group} must divide input channels {channels} and output channels {out_channels}"
    );
    require!(
        w.shape[1] == channels / group,
        "W input-channel dim must equal X channels/group (got {} vs {}/{})",
        w.shape[1],
        channels,
        group
    );
    require!(
        channels % offset_group == 0,
        "offset_group {offset_group} must divide input channels {channels}"
    );
    let effective_h = dilations[0] * (kernel[0] - 1) + 1;
    let effective_w = dilations[1] * (kernel[1] - 1) + 1;
    let padded_h = x.shape[2] + pads[0] + pads[2];
    let padded_w = x.shape[3] + pads[1] + pads[3];
    require!(
        padded_h >= effective_h && padded_w >= effective_w,
        "padded input [{padded_h}, {padded_w}] is smaller than effective kernel [{effective_h}, {effective_w}]"
    );
    let out_h = (padded_h - effective_h) / strides[0] + 1;
    let out_w = (padded_w - effective_w) / strides[1] + 1;
    let kernel_size = kernel[0] * kernel[1];
    let expected_offset = vec![x.shape[0], offset_group * kernel_size * 2, out_h, out_w];
    require!(
        offset.shape == expected_offset,
        "offset shape {:?} must be {:?}",
        offset.shape,
        expected_offset
    );
    let expected_output = vec![x.shape[0], out_channels, out_h, out_w];
    require!(
        out.shape == expected_output,
        "output shape {:?} must be {:?}",
        out.shape,
        expected_output
    );
    if node.input_present(3) {
        let bias = match node.input_info(3) {
            Some(info) => info,
            None => deny!("missing tensor type/shape info on bias"),
        };
        require!(
            bias.dtype == x.dtype && bias.shape == vec![out_channels],
            "bias must have dtype {} and shape [{}]",
            crate::registry::ort_dtype_name(x.dtype),
            out_channels
        );
    }
    if node.input_present(4) {
        let mask = match node.input_info(4) {
            Some(info) => info,
            None => deny!("missing tensor type/shape info on mask"),
        };
        let expected_mask = vec![x.shape[0], offset_group * kernel_size, out_h, out_w];
        require!(
            mask.dtype == x.dtype && mask.shape == expected_mask,
            "mask must have dtype {} and shape {:?}",
            crate::registry::ort_dtype_name(x.dtype),
            expected_mask
        );
    }
    Ok(())
}

fn conv_transpose_claim(node: &NodeView) -> ClaimResult {
    let ni = node.num_inputs();
    require!(
        (2..=3).contains(&ni) && node.num_outputs() == 1,
        "expects 2 or 3 inputs and 1 output, got {}in/{}out",
        ni,
        node.num_outputs()
    );
    let (x, w, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(w), Some(o)) => (x, w, o),
        _ => deny!("missing tensor type/shape info on X, W, or output"),
    };
    require!(
        is_mlx_float(x.dtype) && w.dtype == x.dtype && out.dtype == x.dtype,
        "X/W/output must share one float dtype (fp32/fp16/bf16), got {}, {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(w.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        (3..=5).contains(&x.shape.len()) && x.shape[1..].iter().all(|&d| d != 0),
        "ConvTranspose requires rank 3-5 X with nonzero channel/spatial dims (shape {:?}); symbolic dims are supported",
        x.shape
    );
    let rank = x.shape.len() - 2;
    require!(
        static_positive_shape(&w.shape, rank + 2),
        "ConvTranspose requires matching-rank W with static positive dims (shape {:?})",
        w.shape
    );
    require!(
        out.shape.len() == rank + 2,
        "ConvTranspose output must match X rank (got rank {})",
        out.shape.len()
    );
    let auto_pad = node.string_attr("auto_pad", "NOTSET");
    require!(
        matches!(
            auto_pad.as_str(),
            "NOTSET" | "VALID" | "SAME_UPPER" | "SAME_LOWER"
        ),
        "unsupported auto_pad {auto_pad:?}"
    );
    let group = node.int_attr("group", 1);
    require!(
        group > 0
            && group <= i32::MAX as i64
            && w.shape[0] % group == 0
            && (rank != 3 || group == 1),
        "group must divide W input channels; MLX ConvTranspose3D requires group=1"
    );
    require!(
        x.shape[1] < 0 || w.shape[0] == x.shape[1],
        "W input-channel dim {} must match X channels {}",
        w.shape[0],
        x.shape[1]
    );
    let strides = match read_spatial_attribute(node, "strides", rank, 1) {
        Some(v) => v,
        None => deny!("strides must contain {rank} positive spatial values"),
    };
    let dilations = match read_spatial_attribute(node, "dilations", rank, 1) {
        Some(v) => v,
        None => deny!("dilations must contain {rank} positive spatial values"),
    };
    let explicit_pads = match read_pads(node, rank) {
        Some(v) => v,
        None => deny!("pads must contain {} non-negative values", rank * 2),
    };
    let (op_present, mut output_padding) = node.ints_attr("output_padding");
    if !op_present {
        output_padding = vec![0; rank];
    }
    require!(
        output_padding.len() == rank
            && output_padding
                .iter()
                .enumerate()
                .all(|(i, &v)| v >= 0 && v < strides[i].max(dilations[i])),
        "output_padding must contain {rank} non-negative values smaller than stride or dilation {:?} (got {:?})",
        strides,
        output_padding
    );
    let (output_shape_present, output_shape) = node.ints_attr("output_shape");
    require!(
        !output_shape_present
            || (output_shape.len() == rank
                && output_shape.iter().all(|&v| v > 0 && v <= i32::MAX as i64)),
        "output_shape must contain {rank} positive int32 values"
    );
    let (kernel_present, kernel_shape) = node.ints_attr("kernel_shape");
    require!(
        !kernel_present || kernel_shape == w.shape[2..],
        "kernel_shape {:?} must match W spatial shape {:?}",
        kernel_shape,
        &w.shape[2..]
    );
    let Some(out_channels) = w.shape[1]
        .checked_mul(group)
        .filter(|&v| v <= i32::MAX as i64)
    else {
        deny!("output channels exceed int32");
    };
    require!(
        optional_bias_is_valid(node, x.dtype, out_channels),
        "optional bias must have dtype {} and shape [{}]",
        crate::registry::ort_dtype_name(x.dtype),
        out_channels
    );
    require!(
        (0..rank).all(|i| dilations[i]
            .checked_mul(w.shape[i + 2] - 1)
            .and_then(|v| v.checked_add(output_padding[i]))
            .is_some_and(|v| v < i32::MAX as i64)),
        "dilated kernel and output padding must fit int32"
    );
    require!(
        !matches!(auto_pad.as_str(), "SAME_UPPER" | "SAME_LOWER")
            || output_shape_present
            || (0..rank)
                .all(|i| dilations[i] * (w.shape[i + 2] - 1) + 1 + output_padding[i] >= strides[i]),
        "SAME padding requiring output extension is unsupported"
    );
    let mut expected = vec![x.shape[0], out_channels];
    if x.shape[2..].iter().all(|&d| d > 0) {
        let pads = match transpose_pads(
            &auto_pad,
            &x.shape[2..],
            &w.shape[2..],
            &strides,
            &dilations,
            &output_padding,
            output_shape_present.then_some(output_shape.as_slice()),
            &explicit_pads,
        ) {
            Ok(pads) => pads,
            Err(reason) => deny!("{reason}"),
        };
        for i in 0..rank {
            let uncropped = strides[i] * (x.shape[i + 2] - 1)
                + output_padding[i]
                + dilations[i] * (w.shape[i + 2] - 1)
                + 1;
            require!(
                uncropped <= i32::MAX as i64,
                "uncropped output extent must fit int32"
            );
            let extent = uncropped - pads[i] - pads[i + rank];
            require!(
                extent > 0 && extent <= i32::MAX as i64,
                "derived output extent must be positive int32"
            );
            expected.push(extent);
        }
    } else {
        expected.extend(if output_shape_present {
            output_shape
        } else {
            vec![-1; rank]
        });
    }
    require!(
        same_known_shape(&out.shape, &expected),
        "output shape {:?} does not match expected {:?}",
        out.shape,
        expected
    );
    Ok(())
}

fn pool_claim(node: &NodeView, average: bool) -> ClaimResult {
    let max = node.op_type() == "MaxPool";
    require!(
        node.num_inputs() == 1 && (node.num_outputs() == 1 || (max && node.num_outputs() == 2)),
        "expects 1 input and 1 output (plus optional MaxPool indices), got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(o)) => (x, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        (is_mlx_float(x.dtype) || (max && matches!(x.dtype, 2 | 3))) && out.dtype == x.dtype,
        "input/output must share a supported float dtype (or int8/uint8 for MaxPool), got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        (x.dtype != 16 || node.since_version() >= 22)
            && (!matches!(x.dtype, 2 | 3) || node.since_version() >= 12),
        "input dtype is unavailable in this Pool schema version"
    );
    require!(
        (3..=5).contains(&x.shape.len()),
        "Pool requires 1-3 spatial dimensions (got input rank {})",
        x.shape.len()
    );
    require!(
        x.shape.iter().all(|&d| d <= i32::MAX as i64) && x.shape[2..].iter().all(|&d| d != 0),
        "Pool spatial dimensions must be nonzero and dimensions must fit MLX (shape {:?}); symbolic dimensions are supported",
        x.shape
    );
    require!(
        x.shape[0] == 0 || x.shape[1] != 0,
        "Pool: only the batch dimension may be zero"
    );
    require!(
        out.shape.len() == x.shape.len(),
        "Pool output rank must match input (got rank {})",
        out.shape.len()
    );
    require!(
        matches!(node.int_attr("ceil_mode", 0), 0 | 1),
        "ceil_mode must be 0 or 1"
    );
    let r = x.shape.len() - 2;
    let spatial_dynamic = x.shape[2..].iter().any(|&d| d < 0);
    let (kernel_present, kernel) = node.ints_attr("kernel_shape");
    require!(
        kernel_present
            && kernel.len() == r
            && kernel.iter().all(|&v| v > 0 && v <= i32::MAX as i64),
        "kernel_shape must contain {r} positive MLX-representable values (got {:?})",
        kernel
    );
    let strides = match read_spatial_attribute(node, "strides", r, 1) {
        Some(v) => v,
        None => deny!("strides must contain {r} positive MLX-representable values"),
    };
    let dilations = match read_spatial_attribute(node, "dilations", r, 1) {
        Some(v) => v,
        None => deny!("dilations must contain {r} positive MLX-representable values"),
    };
    let auto_pad = node.string_attr("auto_pad", "NOTSET");
    require!(
        matches!(
            auto_pad.as_str(),
            "NOTSET" | "VALID" | "SAME_UPPER" | "SAME_LOWER"
        ),
        "unsupported auto_pad value {auto_pad:?}"
    );
    require!(
        auto_pad == "NOTSET" || !node.ints_attr("pads").0,
        "auto_pad and explicit pads cannot both be specified"
    );
    let pads = match read_pads(node, r) {
        Some(v) => v,
        None => deny!(
            "pads must contain {} non-negative MLX-representable values",
            2 * r
        ),
    };
    if average {
        let cip = node.int_attr("count_include_pad", 0);
        require!(
            cip == 0 || cip == 1,
            "count_include_pad must be 0 or 1 (got {cip})"
        );
    } else if max {
        require!(
            matches!(node.int_attr("storage_order", 0), 0 | 1),
            "storage_order must be 0 or 1"
        );
        if node.output_present(1) {
            let indices = match node.output_info(1) {
                Some(info) => info,
                None => deny!("missing MaxPool indices type/shape info"),
            };
            require!(
                indices.dtype == 7 && same_known_shape(&indices.shape, &out.shape),
                "MaxPool indices must be int64 with the same shape as Y"
            );
            require!(
                x.shape
                    .iter()
                    .filter(|&&d| d >= 0)
                    .try_fold(1i64, |p, &d| p.checked_mul(d))
                    .is_some(),
                "MaxPool flattened indices exceed int64"
            );
        }
    } else {
        if node.since_version() == 1 {
            let p = node.float_attr("p", 2.0);
            require!(p.is_finite(), "LpPool p must be finite");
        }
    }
    let mut expected = x.shape[..2].to_vec();
    if spatial_dynamic {
        expected.extend(vec![-1; r]);
    } else {
        let geometry = match PoolGeometry::new(
            &x.shape[2..],
            kernel,
            strides,
            dilations,
            pads,
            &auto_pad,
            node.int_attr("ceil_mode", 0) == 1,
        ) {
            Ok(geometry) => geometry,
            Err(reason) => deny!("{reason}"),
        };
        expected.extend(geometry.output.iter().map(|&v| i64::from(v)));
    }
    require!(
        same_known_shape(&out.shape, &expected),
        "output shape {:?} does not match expected {:?} (dynamic batch/spatial dims are wildcards)",
        out.shape,
        expected
    );
    Ok(())
}

fn average_pool_claim(node: &NodeView) -> ClaimResult {
    pool_claim(node, true)
}

fn max_pool_claim(node: &NodeView) -> ClaimResult {
    pool_claim(node, false)
}

fn global_pool_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(out)) => (x, out),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        is_mlx_float(x.dtype) && out.dtype == x.dtype,
        "input/output must share one float dtype (fp32/fp16/bf16), got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() >= 3 && x.shape[2..].iter().all(|&d| d != 0),
        "global Pool requires rank >=3 with nonzero spatial dims (shape {:?}); symbolic dims are supported",
        x.shape
    );
    let mut expected = x.shape.clone();
    expected[2..].fill(1);
    require!(
        same_known_shape(&out.shape, &expected),
        "output shape {:?} does not match expected {:?}",
        out.shape,
        expected
    );
    Ok(())
}

fn lp_pool_claim(node: &NodeView) -> ClaimResult {
    pool_claim(node, false)
}

fn global_lp_pool_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(o)) => (x, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        is_mlx_float(x.dtype) && out.dtype == x.dtype,
        "input/output must share one float dtype (fp32/fp16/bf16), got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() >= 3 && x.shape[2..].iter().all(|&d| d != 0),
        "GlobalLpPool requires rank >=3 with nonzero spatial dims (shape {:?}); symbolic dims are supported",
        x.shape
    );
    require!(
        node.int_attr("p", 2) > 0,
        "p must be positive (got {})",
        node.int_attr("p", 2)
    );
    let mut expected = x.shape.clone();
    expected[2..].fill(1);
    require!(
        same_known_shape(&out.shape, &expected),
        "output shape {:?} does not match expected {:?}",
        out.shape,
        expected
    );
    Ok(())
}

// ---- registration -------------------------------------------------------------------------------

fn shapeless(
    registry: &mut OpRegistry,
    op_type: &'static str,
    handler: crate::registry::OpHandler,
    claim: crate::registry::ClaimPredicate,
) {
    registry.register_shapeless(OpRegistration {
        domain: "",
        op_type,
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler,
        claim,
    });
}

fn shape_keyed(
    registry: &mut OpRegistry,
    op_type: &'static str,
    handler: crate::registry::OpHandler,
    claim: crate::registry::ClaimPredicate,
    reason: &'static str,
) {
    registry.register_shape_keyed(
        OpRegistration {
            domain: "",
            op_type,
            min_opset: K_ANY_OPSET,
            max_opset: K_ANY_OPSET,
            handler,
            claim,
        },
        reason,
    );
}

pub fn register_conv(registry: &mut OpRegistry) {
    shapeless(registry, "Conv", conv_op, conv_claim);
    registry.register_shape_keyed(
        OpRegistration {
            domain: "",
            op_type: "DeformConv",
            min_opset: 19,
            max_opset: K_ANY_OPSET,
            handler: deform_conv_op,
            claim: deform_conv_claim,
        },
        crate::registry::MLX_SLICE_SHAPE_REASON,
    );
    shape_keyed(
        registry,
        "ConvTranspose",
        conv_transpose_op,
        conv_transpose_claim,
        crate::registry::MLX_SLICE_SHAPE_REASON,
    );
    shape_keyed(
        registry,
        "AveragePool",
        average_pool_op,
        average_pool_claim,
        "emits MLX AsStrided and may emit Pad; neither implements Primitive::output_shapes in MLX 0.32.2",
    );
    shape_keyed(
        registry,
        "MaxPool",
        max_pool_op,
        max_pool_claim,
        "emits MLX AsStrided and may emit Pad; neither implements Primitive::output_shapes in MLX 0.32.2",
    );
    shapeless(
        registry,
        "GlobalAveragePool",
        global_average_pool_op,
        global_pool_claim,
    );
    shapeless(
        registry,
        "GlobalMaxPool",
        global_max_pool_op,
        global_pool_claim,
    );
    shape_keyed(
        registry,
        "LpPool",
        lp_pool_op,
        lp_pool_claim,
        "emits MLX AsStrided and may emit Pad; neither implements Primitive::output_shapes in MLX 0.32.2",
    );
    shapeless(
        registry,
        "GlobalLpPool",
        global_lp_pool_op,
        global_lp_pool_claim,
    );
}

#[cfg(test)]
mod pooling_tests {
    use super::PoolGeometry;

    #[test]
    fn ceil_tail_and_last_window_correction() {
        let g =
            PoolGeometry::new(&[4], vec![3], vec![2], vec![1], vec![0, 0], "NOTSET", true).unwrap();
        assert_eq!(g.output, [2]);
        assert_eq!(g.tail, [1]);
        assert_eq!(g.materialized_pads(), [0, 1]);
        let g =
            PoolGeometry::new(&[4], vec![2], vec![2], vec![1], vec![0, 1], "NOTSET", true).unwrap();
        assert_eq!(g.output, [2]);
        assert_eq!(g.tail, [0]);
    }

    #[test]
    fn same_padding_uses_dilated_extent() {
        for (auto_pad, expected) in [("SAME_UPPER", [1, 2]), ("SAME_LOWER", [2, 1])] {
            let g = PoolGeometry::new(&[6], vec![3], vec![2], vec![2], vec![0, 0], auto_pad, false)
                .unwrap();
            assert_eq!(g.output, [3]);
            assert_eq!(g.pads, expected);
        }
    }

    #[test]
    fn empty_output_is_not_negative_output() {
        let g = PoolGeometry::new(&[2], vec![3], vec![1], vec![1], vec![0, 0], "NOTSET", false)
            .unwrap();
        assert_eq!(g.output, [0]);
        assert!(
            PoolGeometry::new(&[1], vec![5], vec![1], vec![2], vec![0, 0], "NOTSET", false)
                .is_err()
        );
    }

    #[test]
    fn invalid_or_unrepresentable_geometry_is_rejected() {
        for (spatial, kernel, stride, dilation, pads) in [
            (vec![0], vec![2], vec![1], vec![1], vec![0, 0]),
            (vec![4], vec![2], vec![0], vec![1], vec![0, 0]),
            (vec![4], vec![2], vec![1], vec![0], vec![0, 0]),
            (vec![4], vec![2], vec![1], vec![1], vec![-1, 0]),
            (vec![4], vec![2, 2], vec![1], vec![1], vec![0, 0]),
            (vec![i32::MAX as i64], vec![2], vec![1], vec![1], vec![1, 1]),
            (
                vec![1],
                vec![i32::MAX as i64],
                vec![1],
                vec![i32::MAX as i64],
                vec![0, 0],
            ),
        ] {
            assert!(
                PoolGeometry::new(&spatial, kernel, stride, dilation, pads, "NOTSET", false)
                    .is_err()
            );
        }
    }
}
