//! Memory-bounded FP32 attention VJP for onnax local functions.
//! The function body remains executable by other ONNX backends.
use crate::engine::{MlxError, NodeDesc, TranslationContext};
use crate::mlx::VectorArray;
use crate::registry::{ClaimResult, NodeView, OpRegistration, OpRegistry};
use crate::require;
use crate::sys::{mlx, ort};
use std::ffi::CString;

mod dense;
mod native;

const PREFIX: &str = "AttentionVJP_";
const F32: mlx::mlx_dtype = mlx::mlx_dtype__MLX_FLOAT32;
const HEADER: &str = include_str!("attention_vjp/common.h");

#[derive(Debug)]
struct Config {
    recompute: bool,
    rank: usize,
    hq: i32,
    hk: i32,
    d: i32,
    scale: f32,
    causal: i32,
    outputs: Vec<usize>,
    mask: i32,
}
impl Config {
    fn kv_rows(&self) -> i32 {
        // Smaller tiles improve occupancy for wider heads and bound workspace.
        if self.d >= 128 { 8 } else { 16 }
    }

    fn parse(name: &str) -> Option<Self> {
        let recompute = name.starts_with("AttentionVJPRecompute_");
        let fields: Vec<_> = name
            .strip_prefix(if recompute {
                "AttentionVJPRecompute_"
            } else {
                PREFIX
            })?
            .split('_')
            .collect();
        if fields.len() != 9 {
            return None;
        }
        let integer = |i: usize, prefix: &str| fields[i].strip_prefix(prefix)?.parse::<i32>().ok();
        let rank = integer(0, "r")? as usize;
        let hq = integer(1, "q")?;
        let hk = integer(2, "k")?;
        let d = integer(3, "d")?;
        let scale = f32::from_bits(u32::from_str_radix(fields[4].strip_prefix('s')?, 16).ok()?);
        let causal = integer(5, "c")?;
        let outputs: Vec<_> = fields[6]
            .strip_prefix('o')?
            .chars()
            .map(|c| c.to_digit(10).map(|i| i as usize))
            .collect::<Option<_>>()?;
        let mask = integer(7, "m")?;
        if (recompute && !(32..=256).contains(&d))
            || !matches!(rank, 3 | 4)
            || hq <= 0
            || hk <= 0
            || hq % hk != 0
            || !(8..=256).contains(&d)
            || d % 8 != 0
            || hq.checked_mul(d).is_none()
            || hk.checked_mul(d).is_none()
            || !scale.is_finite()
            || scale <= 0.
            || !matches!(causal, 0 | 1)
            || !matches!(mask, 0..=2)
            || outputs.is_empty()
            || outputs.iter().any(|&i| i > 2)
            || outputs.windows(2).any(|p| p[0] >= p[1])
            || fields[8].len() != 16
            || !fields[8].bytes().all(|c| c.is_ascii_hexdigit())
        {
            return None;
        }
        Some(Self {
            recompute,
            rank,
            hq,
            hk,
            d,
            scale,
            causal,
            outputs,
            mask,
        })
    }
}

fn claim(node: &NodeView) -> ClaimResult {
    let config = Config::parse(&node.op_type());
    require!(
        config.is_some(),
        "unsupported AttentionVJP function contract"
    );
    let c = config.unwrap();
    let expected_inputs = if c.recompute { 5 } else { 6 };
    require!(
        node.num_inputs() == expected_inputs && node.num_outputs() == c.outputs.len(),
        "AttentionVJP interface mismatch"
    );
    for i in 0..if c.recompute { 4 } else { 5 } {
        let info = node.input_info(i);
        require!(info.is_some(), "AttentionVJP requires tensor inputs");
        let info = info.unwrap();
        require!(
            info.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                && info.shape.len() == c.rank,
            "AttentionVJP requires rank-matched FP32 tensors"
        );
        let heads = if i == 1 || i == 2 { c.hk } else { c.hq };
        require!(
            info.shape[c.rank - 1]
                == if c.rank == 3 {
                    i64::from(heads * c.d)
                } else {
                    i64::from(c.d)
                },
            "AttentionVJP static width mismatch"
        );
        if c.rank == 4 {
            require!(
                info.shape[1] == i64::from(heads),
                "AttentionVJP head count mismatch"
            );
        }
    }
    let mask = node.input_info(if c.recompute { 4 } else { 5 });
    require!(mask.is_some(), "AttentionVJP mask tensor required");
    let mask = mask.unwrap();
    require!(
        (1..=4).contains(&mask.shape.len()),
        "AttentionVJP mask rank must be 1..4"
    );
    require!(
        if c.mask == 2 {
            mask.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        } else {
            mask.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
        },
        "AttentionVJP mask type mismatch"
    );
    Ok(())
}

struct Kernel {
    name: &'static str,
    source: &'static str,
    inputs: &'static [&'static str],
    outputs: &'static [&'static str],
}

fn dispatch(
    ctx: &mut TranslationContext,
    specification: Kernel,
    inputs: &[mlx::mlx_array],
    shapes: &[Vec<i32>],
    grid: [i32; 3],
    c: &Config,
    sequence: [i32; 2],
) -> Result<Vec<mlx::mlx_array>, MlxError> {
    let [sq, sk] = sequence;
    let names = super::selective_scan::VecString::new(specification.inputs)?;
    let outputs = super::selective_scan::VecString::new(specification.outputs)?;
    let name = CString::new(specification.name).unwrap();
    let source = CString::new(specification.source).unwrap();
    let header = CString::new(HEADER).unwrap();
    let kernel = unsafe {
        mlx::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            names.as_raw(),
            outputs.as_raw(),
            source.as_ptr(),
            header.as_ptr(),
            false,
            false,
        )
    };
    if kernel.ctx.is_null() {
        return Err("AttentionVJP Metal kernel creation failed".into());
    }
    let config = unsafe { mlx::mlx_fast_metal_kernel_config_new() };
    if config.ctx.is_null() {
        unsafe { mlx::mlx_fast_metal_kernel_free(kernel) };
        return Err("AttentionVJP Metal config creation failed".into());
    }
    let mut rc = 0;
    for (key, value) in [
        ("HQ", c.hq),
        ("HK", c.hk),
        ("D", c.d),
        ("KV_ROWS", c.kv_rows()),
        ("SQ", sq),
        ("SK", sk),
        ("SCALE", c.scale.to_bits() as i32),
        ("MASK", c.mask),
        ("CAUSAL", c.causal),
    ] {
        let key = CString::new(key).unwrap();
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_template_arg_int(config, key.as_ptr(), value)
        };
    }
    for shape in shapes {
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_output_arg(
                config,
                shape.as_ptr(),
                shape.len(),
                F32,
            )
        };
    }
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_grid(config, grid[0], grid[1], grid[2]) };
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1) };
    let mut args = VectorArray::new();
    for &input in inputs {
        args.append(input);
    }
    let mut result = VectorArray::new();
    if rc == 0 {
        rc = unsafe {
            mlx::mlx_fast_metal_kernel_apply(
                result.as_mut_ptr(),
                kernel,
                args.as_raw(),
                config,
                ctx.stream(),
            )
        };
    }
    unsafe {
        mlx::mlx_fast_metal_kernel_config_free(config);
        mlx::mlx_fast_metal_kernel_free(kernel);
    }
    if rc != 0 || result.size() != shapes.len() {
        return Err("AttentionVJP Metal dispatch failed".into());
    }
    Ok((0..result.size())
        .map(|i| ctx.keep(result.get(i)))
        .collect())
}

fn use_dense(c: &Config, qs: &[i32], ks: &[i32], explicit_limit: Option<u64>) -> bool {
    if c.recompute && explicit_limit.is_none() {
        return true;
    }
    let limit = explicit_limit.unwrap_or(64 * 1024 * 1024);
    [qs[0], c.hq, qs[2], ks[2]]
        .iter()
        .try_fold(4u64, |bytes, &dimension| {
            bytes.checked_mul(dimension as u64)
        })
        .is_some_and(|bytes| bytes <= limit)
}

fn translate(ctx: &mut TranslationContext, node: &NodeDesc) -> Result<(), MlxError> {
    let c = Config::parse(&node.op_type)
        .ok_or_else(|| MlxError::from("Invalid AttentionVJP contract"))?;
    let mut inputs = Vec::new();
    let mut original = Vec::new();
    let tensors = if c.recompute { 4 } else { 5 };
    for (i, input) in node.inputs.iter().enumerate() {
        let mut a = ctx.resolve(input)?;
        let shape = ctx.shape_of(a);
        original.push(shape.clone());
        if i < tensors && c.rank == 3 {
            let heads = if i == 1 || i == 2 { c.hk } else { c.hq };
            a = ctx.reshape(a, &[shape[0], shape[1], heads, c.d])?;
            a = ctx.transpose(a, &[0, 2, 1, 3])?;
        }
        if i < tensors {
            a = ctx.contiguous(a)?;
        }
        inputs.push(a);
    }
    let explicit_limit = std::env::var("ONNXRUNTIME_EP_MLX_ATTENTION_VJP_DENSE_LIMIT_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok());
    let dense_path = use_dense(
        &c,
        &ctx.shape_of(inputs[0]),
        &ctx.shape_of(inputs[1]),
        explicit_limit,
    );
    if c.recompute && dense_path {
        let grads = native::execute(ctx, &inputs, &c)?;
        for (output, &index) in node.outputs.iter().zip(&c.outputs) {
            let mut value = grads[index];
            if c.rank == 3 {
                value = ctx.transpose(value, &[0, 2, 1, 3])?;
                value = ctx.reshape(value, &original[index])?;
            }
            ctx.bind(output, value);
        }
        return Ok(());
    }
    if c.recompute {
        let y = native::primal(ctx, &inputs, &c)?;
        inputs.insert(3, y);
    }
    let (q, k, v, y, g, m) = (
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
    );
    let qs = ctx.shape_of(q);
    let ks = ctx.shape_of(k);
    let vs = ctx.shape_of(v);
    let (b, sq, sk) = (qs[0], qs[2], ks[2]);
    if b == 0 || sq == 0 || sk == 0 {
        for (output, &index) in node.outputs.iter().zip(&c.outputs) {
            let zero = ctx.zeros(&original[index], F32)?;
            ctx.bind(output, zero);
        }
        return Ok(());
    }
    if ctx.shape_of(y) != qs || ctx.shape_of(g) != qs || vs != ks || ks[0] != b {
        return Err("AttentionVJP incompatible live tensor shapes".into());
    }
    let mask_shape = ctx.shape_of(m);
    let target = [b, c.hq, sq, sk];
    for (axis, &size) in mask_shape.iter().enumerate() {
        let target_size = target[4 - mask_shape.len() + axis];
        if size != 1 && size != target_size && axis + 1 != mask_shape.len() {
            return Err("AttentionVJP incompatible mask broadcast".into());
        }
    }
    let grads = if dense_path {
        let arrays = dense::execute(&inputs, [b, sq, sk], &mask_shape, &c, ctx.stream())?;
        ctx.mark_fast("onnax GEMM attention VJP");
        let mut iter = arrays.into_iter();
        [
            ctx.keep(iter.next().unwrap()),
            ctx.keep(iter.next().unwrap()),
            ctx.keep(iter.next().unwrap()),
        ]
    } else {
        ctx.mark_fast("onnax tiled attention VJP");
        let stats = dispatch(
            ctx,
            Kernel {
                name: "onnax_attention_lse",
                source: include_str!("attention_vjp/lse.metal"),
                inputs: &["Q", "K", "Y", "G", "Mask"],
                outputs: &["LS", "Delta"],
            },
            &[q, k, y, g, m],
            &[vec![b, c.hq, sq], vec![b, c.hq, sq]],
            [(sq + 15) / 16 * 128, c.hq, b],
            &c,
            [sq, sk],
        )?;
        let dq = dispatch(
            ctx,
            Kernel {
                name: "onnax_attention_dq",
                source: include_str!("attention_vjp/dq.metal"),
                inputs: &["Q", "K", "V", "G", "Mask", "LS", "Delta"],
                outputs: &["DQ"],
            },
            &[q, k, v, g, m, stats[0], stats[1]],
            &[qs],
            [(sq + 15) / 16 * 128, c.hq, b],
            &c,
            [sq, sk],
        )?[0];
        let kv = dispatch(
            ctx,
            Kernel {
                name: "onnax_attention_dkv",
                source: include_str!("attention_vjp/dkv.metal"),
                inputs: &["Q", "K", "V", "G", "Mask", "LS", "Delta"],
                outputs: &["DK", "DV"],
            },
            &[q, k, v, g, m, stats[0], stats[1]],
            &[ks, vs],
            [(sk + c.kv_rows() - 1) / c.kv_rows() * 128, c.hk, b],
            &c,
            [sq, sk],
        )?;
        [dq, kv[0], kv[1]]
    };
    for (output, &index) in node.outputs.iter().zip(&c.outputs) {
        let mut value = grads[index];
        if c.rank == 3 {
            value = ctx.transpose(value, &[0, 2, 1, 3])?;
            value = ctx.reshape(value, &original[index])?;
        }
        ctx.bind(output, value);
    }
    Ok(())
}

pub fn register(registry: &mut OpRegistry) {
    registry.register_shape_keyed(
        OpRegistration {
            domain: "pkg.onnax",
            op_type: "AttentionVJP",
            min_opset: 1,
            max_opset: 1,
            handler: translate,
            claim,
        },
        "conservatively shape-key Metal grids and workspace selection; custom kernels lack dynamic output_shapes",
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    const VALID: &str = "AttentionVJP_r3_q16_k8_d128_s3db504f3_c1_o012_m1_0123456789abcdef";

    #[test]
    fn recomputed_contract_is_explicit() {
        let name = "AttentionVJPRecompute_r3_q4_k2_d40_s3e21e89b_c0_o012_m0_0123456789abcdef";
        let config = Config::parse(name).unwrap();
        assert!(config.recompute);
        for (old, new) in [
            ("d40_", "d8_"),
            ("d40_", "d264_"),
            ("c0_", "c2_"),
            ("m0_", "m3_"),
        ] {
            assert!(Config::parse(&name.replace(old, new)).is_none());
        }
        assert!(!Config::parse(VALID).unwrap().recompute);
        assert!(
            Config::parse(&VALID.replace("AttentionVJP_", "AttentionVJPRecompute_"))
                .unwrap()
                .recompute
        );
    }

    #[test]
    fn explicit_score_budget_applies_to_recomputed_pullbacks() {
        let mut c = Config::parse(VALID).unwrap();
        let query = [1, 16, 4096, 128];
        let key = [1, 8, 4096, 128];
        let score_bytes = 1024 * 1024 * 1024;
        assert!(!use_dense(&c, &query, &key, None));
        assert!(use_dense(&c, &query, &key, Some(score_bytes)));
        assert!(!use_dense(&c, &query, &key, Some(score_bytes - 1)));
        c.recompute = true;
        assert!(use_dense(&c, &query, &key, None));
        assert!(use_dense(&c, &query, &key, Some(score_bytes)));
        assert!(!use_dense(&c, &query, &key, Some(0)));
        assert!(!use_dense(&c, &query, &key, Some(score_bytes - 1)));
    }

    #[test]
    fn wide_tiles_fit_threadgroup_memory() {
        for d in (8..=256).step_by(8) {
            let c = Config::parse(&VALID.replace("d128_", &format!("d{d}_"))).unwrap();
            let kv = (4 * c.kv_rows() * 32 + 2 * c.kv_rows() * d) * 4;
            let dq = (3 * 16 * 32 + 16 * d) * 4;
            assert!(kv <= 32 * 1024 && dq <= 32 * 1024);
            assert_eq!(c.kv_rows(), if d >= 128 { 8 } else { 16 });
        }
    }

    #[test]
    fn parameterized_function_contract() {
        let c = Config::parse(VALID).unwrap();
        assert_eq!(
            (c.rank, c.hq, c.hk, c.d, c.causal, c.mask),
            (3, 16, 8, 128, 1, 1)
        );
        assert_eq!(c.outputs, vec![0, 1, 2]);
        for (old, new) in [
            ("AttentionVJP_", "AttentionVJP_v1_"),
            ("q16_", "q15_"),
            ("q16_", "q2147483640_"),
            ("d128_", "d512_"),
            ("d128_", "d0_"),
            ("d128_", "d41_"),
            ("s3db504f3_", "s7f800000_"),
            ("c1_", "c2_"),
            ("o012_", "o210_"),
            ("m1_", "m3_"),
        ] {
            assert!(Config::parse(&VALID.replace(old, new)).is_none(), "{new}");
        }
        assert!(Config::parse("AttentionVJP").is_none());
        assert!(Config::parse(&format!("{VALID}_extra")).is_none());
    }
}
