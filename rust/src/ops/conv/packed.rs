use super::{add, attr_or, from_channels_last, present, to_channels_last};
use crate::engine::{MlxError, NodeDesc, TranslationContext, mlx_dtype_from_onnx};
use crate::mlx::metal;
use crate::sys::mlx;

pub(crate) fn profile(
    weight: &[i64],
    group: i64,
    strides: Option<&[i64]>,
    pads: Option<&[i64]>,
) -> bool {
    (3..=5).contains(&weight.len())
        && weight[2..].iter().all(|&v| v == 1)
        && group == 1
        && strides.is_none_or(|v| v.len() == weight.len() - 2 && v.iter().all(|&s| s == 1))
        && pads.is_none_or(|v| v.len() == 2 * (weight.len() - 2) && v.iter().all(|&p| p == 0))
}

pub(super) fn execute(
    ctx: &mut TranslationContext,
    node: &NodeDesc,
) -> Result<Option<mlx::mlx_array>, MlxError> {
    let Some(weight) = &node.packed_weight else {
        return Ok(None);
    };
    let ws = weight.shape();
    let wide = ws.iter().copied().map(i64::from).collect::<Vec<_>>();
    if !profile(
        &wide,
        node.ints.get("group").copied().unwrap_or(1),
        node.int_arrays.get("strides").map(Vec::as_slice),
        node.int_arrays.get("pads").map(Vec::as_slice),
    ) || !metal::is_gpu(ctx)?
    {
        return Ok(None);
    }
    let spatial = ws.len() - 2;
    let dilations = attr_or(node, "dilations", spatial, 1);
    if dilations.len() != spatial
        || dilations.iter().any(|&d| d <= 0)
        || node
            .int_arrays
            .get("kernel_shape")
            .is_some_and(|v| v != &vec![1; spatial])
        || !matches!(
            node.strings
                .get("auto_pad")
                .map(String::as_str)
                .unwrap_or("NOTSET"),
            "NOTSET" | "VALID" | "SAME_UPPER" | "SAME_LOWER"
        )
    {
        return Ok(None);
    }
    let x = ctx.resolve(&node.inputs[0])?;
    let xs = ctx.shape_of(x);
    let dtype = mlx_dtype_from_onnx(node.outputs[0].otype);
    if xs.len() != ws.len() || xs[2..].iter().any(|&d| d <= 0) || ctx.dtype_of(x) != dtype {
        return Ok(None);
    }
    if xs[1] != ws[1] {
        return Err(format!(
            "Conv: X channels {} differ from decoded weight channels {}; correct the packed reshape or input channel layout",
            xs[1], ws[1]
        ));
    }
    let Some(m) = xs
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != 1)
        .try_fold(1i32, |m, (_, &d)| m.checked_mul(d))
    else {
        return Ok(None);
    };
    let (k, n) = (ws[1], ws[0]);
    if m.checked_mul(n).is_none() {
        return Ok(None);
    }
    let bias = if present(node, 2) {
        let bias = ctx.resolve(&node.inputs[2])?;
        if ctx.shape_of(bias) != [n] || ctx.dtype_of(bias) != dtype {
            return Err(format!(
                "Conv: bias must have shape [{n}] and match the input/weight dtype; supply one bias value per output channel"
            ));
        }
        Some(bias)
    } else {
        None
    };
    let x = to_channels_last(ctx, x, spatial as i32)?;
    let x = ctx.reshape(x, &[m, k])?;
    let payload = weight.payload(ctx)?;
    let Some(mut y) = weight.matrix(ctx, [x, payload], [m, k, n], &[m, n], true, None)? else {
        return Ok(None);
    };
    if let Some(bias) = bias {
        y = add(ctx, y, bias)?;
    }
    let mut shape = vec![xs[0]];
    shape.extend(&xs[2..]);
    shape.push(n);
    let y = ctx.reshape(y, &shape)?;
    let y = from_channels_last(ctx, y, spatial as i32)?;
    ctx.mark_fast("mlx_packed_pointwise_conv");
    Ok(Some(y))
}
