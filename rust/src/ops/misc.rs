//! Misc op handlers: Constant (scalar/list value forms), TfIdfVectorizer, OneHot, Trilu, Scatter
//! (opset 9/10), Det, sequence consumers, OptionalHasElement/GetElement and the loss ops
//! (NegativeLogLikelihoodLoss / SoftmaxCrossEntropyLoss). Faithful port of the C++ `ops/misc2.cc`
//! (plus OneHot/Trilu from `math.cc` and Constant from `shape.cc`). Det materialises its input,
//! computes on the host, and wraps the result back as a fresh MLX array. Only statically
//! translatable, MLX-supported forms are claimed; every other form is left to ORT CPU.

use std::collections::HashMap;
use std::os::raw::c_void;

use crate::engine::{MlxError, NodeDesc, Src, TensorRef, TranslationContext};
use crate::mlx::VectorArray;
use crate::registry::{
    ClaimPredicate, ClaimResult, K_ANY_OPSET, NodeView, OpHandler, OpRegistration, OpRegistry,
    SlotInfo, ValueKind, is_mlx_float, is_mlx_supported, is_movable, is_signed_integer,
};
use crate::sys::mlx;
use crate::sys::ort;
use crate::{deny, require};

const F32: mlx::mlx_dtype = mlx::mlx_dtype__MLX_FLOAT32;
const I32: mlx::mlx_dtype = mlx::mlx_dtype__MLX_INT32;

const T_FLOAT: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
const T_INT32: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
const T_INT64: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;

// ---- shared helpers -----------------------------------------------------------------------------

fn present(n: &NodeDesc, i: usize) -> bool {
    i < n.inputs.len() && n.inputs[i].source != Src::Absent && !n.inputs[i].name.is_empty()
}

fn str_attr(n: &NodeDesc, name: &str, dflt: &str) -> String {
    n.strings
        .get(name)
        .cloned()
        .unwrap_or_else(|| dflt.to_string())
}

/// Read a constant scalar int (int32/int64) input at translate time.
fn read_const_i64(ctx: &TranslationContext, r: &TensorRef) -> Result<i64, MlxError> {
    let h = ctx.raw_host(r)?;
    if h.data.is_null() || h.count < 1 {
        return Err("MLX misc: expected a scalar int".to_string());
    }
    match h.dtype {
        t if t == T_INT64 => Ok(unsafe { *(h.data as *const i64) }),
        t if t == T_INT32 => Ok(unsafe { *(h.data as *const i32) } as i64),
        _ => Err("MLX misc: scalar int input has an unsupported dtype".to_string()),
    }
}

/// A kept scalar of `val` cast to `x`'s dtype (keeps an int graph in its own dtype).
fn scalar_like(
    ctx: &mut TranslationContext,
    val: i64,
    x: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    let s = ctx.scalar_i64(val);
    ctx.astype(s, ctx.dtype_of(x))
}

fn is_int_index(t: ort::ONNXTensorElementDataType) -> bool {
    t == T_INT32 || t == T_INT64
}

fn static_tensor(info: &SlotInfo) -> bool {
    info.shape.iter().all(|&d| d >= 0)
}

// =============================================================================================
// TfIdfVectorizer — numeric-token, pool_int64s form, statically unrolled in ORT traversal order.
// =============================================================================================
fn tfidf_pool(n: &NodeDesc) -> Result<Vec<HashMap<Vec<i64>, usize>>, MlxError> {
    let counts = n
        .int_arrays
        .get("ngram_counts")
        .ok_or_else(|| "MLX TfIdfVectorizer: missing ngram_counts".to_string())?;
    let pool = n
        .int_arrays
        .get("pool_int64s")
        .ok_or_else(|| "MLX TfIdfVectorizer: missing pool_int64s".to_string())?;
    let indexes = n
        .int_arrays
        .get("ngram_indexes")
        .ok_or_else(|| "MLX TfIdfVectorizer: missing ngram_indexes".to_string())?;
    let mut grams = vec![HashMap::new(); counts.len() + 1];
    let mut gram_id = 0usize;
    for (i, &start64) in counts.iter().enumerate() {
        let size = i + 1;
        let start = usize::try_from(start64)
            .map_err(|_| "MLX TfIdfVectorizer: negative ngram_counts value".to_string())?;
        let end64 = counts.get(i + 1).copied().unwrap_or(pool.len() as i64);
        let end = usize::try_from(end64)
            .map_err(|_| "MLX TfIdfVectorizer: negative ngram_counts value".to_string())?;
        if end < start || end > pool.len() {
            return Err("MLX TfIdfVectorizer: ngram_counts are out of pool bounds".to_string());
        }
        if (end - start) % size != 0 {
            return Err(format!(
                "MLX TfIdfVectorizer: pool items do not form whole {size}-grams"
            ));
        }
        for tokens in pool[start..end].chunks_exact(size) {
            if grams[size].insert(tokens.to_vec(), gram_id).is_some() {
                return Err(format!(
                    "MLX TfIdfVectorizer: duplicate {size}-gram in pool_int64s"
                ));
            }
            gram_id += 1;
        }
    }
    if gram_id != indexes.len() {
        return Err(format!(
            "MLX TfIdfVectorizer: {} pooled n-grams but {} ngram_indexes",
            gram_id,
            indexes.len()
        ));
    }
    Ok(grams)
}

fn tfidf_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(x);
    let (rows, row_size) = if shape.len() == 1 {
        (1usize, shape[0] as usize)
    } else {
        (shape[0] as usize, shape[1] as usize)
    };
    let min_gram = *n.ints.get("min_gram_length").unwrap() as usize;
    let max_gram = *n.ints.get("max_gram_length").unwrap() as usize;
    let max_skip = *n.ints.get("max_skip_count").unwrap() as usize;
    let mode = n.strings.get("mode").unwrap().as_str();
    let indexes = n.int_arrays.get("ngram_indexes").unwrap();
    let weights = n.float_arrays.get("weights");
    if let Some(w) = weights
        && w.len() != indexes.len()
    {
        return Err(format!(
            "MLX TfIdfVectorizer: weights size {} must equal ngram_indexes size {}",
            w.len(),
            indexes.len()
        ));
    }
    if indexes.iter().any(|&index| index < 0) {
        return Err("MLX TfIdfVectorizer: ngram_indexes must be non-negative".to_string());
    }
    let grams = tfidf_pool(n)?;
    let output_size = indexes.iter().copied().max().unwrap() as usize + 1;
    let flat = ctx.reshape(x, &[(rows * row_size) as i32])?;
    let mut output = Vec::with_capacity(rows * output_size);

    for row in 0..rows {
        let row_output_start = output.len();
        output.extend((0..output_size).map(|_| ctx.scalar_f32(0.0)));
        let mut start_gram = min_gram;
        let useful_skip_distances = if max_gram <= 1 {
            1
        } else {
            max_skip
                .saturating_add(1)
                .min(row_size.saturating_sub(1).max(1))
        };
        for skip_distance in 1..=useful_skip_distances {
            for start in 0..row_size {
                let Some(last_min) = (start_gram - 1)
                    .checked_mul(skip_distance)
                    .and_then(|d| start.checked_add(d))
                else {
                    break;
                };
                if last_min >= row_size {
                    break;
                }
                for (gram_size, gram) in grams.iter().enumerate().take(max_gram + 1).skip(1) {
                    let Some(pos) = (gram_size - 1)
                        .checked_mul(skip_distance)
                        .and_then(|d| start.checked_add(d))
                    else {
                        break;
                    };
                    if pos >= row_size {
                        break;
                    }
                    if gram_size < start_gram {
                        continue;
                    }
                    for (tokens, &pool_idx) in gram {
                        let mut matches = None;
                        for (token_offset, &expected_value) in tokens.iter().enumerate() {
                            let token_pos = start + token_offset * skip_distance;
                            let index = ctx.scalar_i64((row * row_size + token_pos) as i64);
                            let token =
                                ctx.emit(|res, s| unsafe { mlx::mlx_take(res, flat, index, s) })?;
                            let expected = scalar_like(ctx, expected_value, flat)?;
                            let eq = ctx.binary(mlx::mlx_equal, token, expected)?;
                            matches = Some(match matches {
                                Some(prefix) => ctx.binary(mlx::mlx_logical_and, prefix, eq)?,
                                None => eq,
                            });
                        }
                        let matches = matches.unwrap();
                        let out_idx = indexes[pool_idx] as usize;
                        let slot = row_output_start + out_idx;
                        let weight = weights.map_or(1.0, |w| w[pool_idx]);
                        output[slot] = match mode {
                            "TF" => {
                                let match_f = ctx.astype(matches, F32)?;
                                ctx.add(output[slot], match_f)?
                            }
                            "IDF" => {
                                let value = ctx.scalar_f32(weight);
                                ctx.emit(|res, s| unsafe {
                                    mlx::mlx_where(res, matches, value, output[slot], s)
                                })?
                            }
                            "TFIDF" => {
                                let value = ctx.scalar_f32(weight);
                                let zero = ctx.scalar_f32(0.0);
                                let increment = ctx.emit(|res, s| unsafe {
                                    mlx::mlx_where(res, matches, value, zero, s)
                                })?;
                                ctx.add(output[slot], increment)?
                            }
                            _ => unreachable!(),
                        };
                    }
                }
            }
            if start_gram == 1 {
                start_gram = 2;
                if start_gram > max_gram {
                    break;
                }
            }
        }
    }

    let result = ctx.stack(&output, 0)?;
    let result = if shape.len() == 1 {
        result
    } else {
        ctx.reshape(result, &[rows as i32, output_size as i32])?
    };
    ctx.bind(&n.outputs[0], result);
    Ok(())
}

fn tfidf_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (input, output) = match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => (i, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        static_tensor(&input) && static_tensor(&output),
        "input and output shapes must be static"
    );
    require!(
        input.dtype == T_INT32 || input.dtype == T_INT64,
        "only numeric int32/int64 token inputs are supported (string tensors are not supported)"
    );
    require!(
        output.dtype == T_FLOAT,
        "output must have float32 element type"
    );
    require!(
        input.shape.len() == 1 || input.shape.len() == 2,
        "input must have rank 1 or 2"
    );
    require!(
        input.shape.iter().all(|&d| i32::try_from(d).is_ok())
            && input
                .shape
                .iter()
                .try_fold(1i64, |n, &d| n.checked_mul(d))
                .is_some_and(|n| i32::try_from(n).is_ok()),
        "input dimensions and element count must fit MLX int32 shapes"
    );
    if input.shape.len() == 2 && input.shape[0] < 1 {
        deny!("rank-2 input batch dimension must be positive");
    }

    require!(
        node.attr_type("mode") == ort::OrtOpAttrType_ORT_OP_ATTR_STRING,
        "mode must be a required STRING attribute"
    );
    let mode = node.string_attr("mode", "");
    require!(
        mode == "TF" || mode == "IDF" || mode == "TFIDF",
        "mode must be TF, IDF, or TFIDF (got {mode})"
    );
    for name in ["min_gram_length", "max_gram_length", "max_skip_count"] {
        if node.attr_type(name) != ort::OrtOpAttrType_ORT_OP_ATTR_INT {
            deny!("{name} must be a required INT attribute");
        }
    }
    let min_gram = node.int_attr("min_gram_length", 0);
    let max_gram = node.int_attr("max_gram_length", 0);
    let max_skip = node.int_attr("max_skip_count", -1);
    require!(min_gram > 0, "min_gram_length must be positive");
    require!(
        max_gram >= min_gram,
        "max_gram_length must be at least min_gram_length"
    );
    require!(max_skip >= 0, "max_skip_count must be non-negative");
    require!(
        usize::try_from(max_skip).is_ok(),
        "max_skip_count exceeds host indexing range"
    );

    let (have_counts, counts) = node.ints_attr("ngram_counts");
    let (have_indexes, indexes) = node.ints_attr("ngram_indexes");
    let (have_pool, pool) = node.ints_attr("pool_int64s");
    require!(
        have_counts && !counts.is_empty(),
        "non-empty INTS ngram_counts is required"
    );
    require!(
        have_indexes && !indexes.is_empty(),
        "non-empty INTS ngram_indexes is required"
    );
    require!(
        have_pool && !pool.is_empty(),
        "non-empty INTS pool_int64s is required"
    );
    require!(
        !node.has_attr("pool_strings"),
        "pool_strings/string-token form is not supported"
    );
    if node.has_attr("weights")
        && node.attr_type("weights") != ort::OrtOpAttrType_ORT_OP_ATTR_FLOATS
    {
        deny!("weights must be a FLOATS attribute");
    }
    require!(
        max_gram as usize <= counts.len(),
        "max_gram_length must be within ngram_counts"
    );
    require!(
        indexes.iter().all(|&v| v >= 0),
        "ngram_indexes values must be non-negative"
    );

    let mut total_grams = 0usize;
    let mut seen = HashMap::<Vec<i64>, ()>::new();
    for (i, &start64) in counts.iter().enumerate() {
        let size = i + 1;
        let start = match usize::try_from(start64) {
            Ok(v) => v,
            Err(_) => deny!("ngram_counts values must be non-negative"),
        };
        let end64 = counts.get(i + 1).copied().unwrap_or(pool.len() as i64);
        let end = match usize::try_from(end64) {
            Ok(v) => v,
            Err(_) => deny!("ngram_counts values must be non-negative"),
        };
        if end < start || end > pool.len() {
            deny!("ngram_counts are not monotonic offsets within pool_int64s");
        }
        if (end - start) % size != 0 {
            deny!("pool_int64s items do not form whole {size}-grams");
        }
        for gram in pool[start..end].chunks_exact(size) {
            if seen.insert(gram.to_vec(), ()).is_some() {
                deny!("duplicate {size}-gram in pool_int64s");
            }
            total_grams += 1;
        }
        seen.clear();
    }
    require!(
        total_grams == indexes.len(),
        "pooled n-gram count ({total_grams}) must equal ngram_indexes size ({})",
        indexes.len()
    );
    let output_size = match indexes.iter().copied().max().and_then(|v| v.checked_add(1)) {
        Some(v) => v,
        None => deny!("ngram_indexes output size overflows"),
    };
    require!(
        i32::try_from(output_size).is_ok(),
        "ngram_indexes output size must fit an MLX int32 dimension"
    );
    let expected = if input.shape.len() == 1 {
        vec![output_size]
    } else {
        vec![input.shape[0], output_size]
    };
    require!(
        output.shape == expected,
        "output shape must be {:?}, got {:?}",
        expected,
        output.shape
    );
    Ok(())
}

// =============================================================================================
// Constant (scalar / list value forms). TENSOR (`value`) / sparse forms are left to ORT CPU.
// =============================================================================================
fn constant_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let result = if let Some(&v) = n.ints.get("value_int") {
        ctx.scalar_i64(v)
    } else if let Some(&v) = n.floats.get("value_float") {
        ctx.scalar_f32(v)
    } else if let Some(values) = n.int_arrays.get("value_ints") {
        ctx.from_host_i64(values, &[values.len() as i32])
    } else if let Some(values) = n.float_arrays.get("value_floats") {
        ctx.from_host(
            values.as_ptr() as *const c_void,
            &[values.len() as i32],
            F32,
        )
    } else {
        return Err("MLX: Constant attribute form is not supported".to_string());
    };
    ctx.bind(&n.outputs[0], result);
    Ok(())
}

fn constant_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 0 && node.num_outputs() == 1,
        "expects 0 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let out = match node.output_info(0) {
        Some(i) if static_tensor(&i) => i,
        Some(_) => deny!("output shape must be static"),
        None => deny!("missing output type/shape info"),
    };
    require!(
        !node.has_attr("value") && !node.has_attr("sparse_value"),
        "tensor and sparse Constant value forms are not supported"
    );
    struct Form {
        name: &'static str,
        attr_type: ort::OrtOpAttrType,
        out_type: ort::ONNXTensorElementDataType,
        scalar: bool,
    }
    let forms = [
        Form {
            name: "value_int",
            attr_type: ort::OrtOpAttrType_ORT_OP_ATTR_INT,
            out_type: T_INT64,
            scalar: true,
        },
        Form {
            name: "value_float",
            attr_type: ort::OrtOpAttrType_ORT_OP_ATTR_FLOAT,
            out_type: T_FLOAT,
            scalar: true,
        },
        Form {
            name: "value_ints",
            attr_type: ort::OrtOpAttrType_ORT_OP_ATTR_INTS,
            out_type: T_INT64,
            scalar: false,
        },
        Form {
            name: "value_floats",
            attr_type: ort::OrtOpAttrType_ORT_OP_ATTR_FLOATS,
            out_type: T_FLOAT,
            scalar: false,
        },
    ];
    let mut matched = 0;
    for f in &forms {
        let at = node.attr_type(f.name);
        if at == ort::OrtOpAttrType_ORT_OP_ATTR_UNDEFINED {
            continue;
        }
        if at != f.attr_type || out.dtype != f.out_type {
            deny!(
                "attribute {} has incompatible type or output dtype {}",
                f.name,
                crate::registry::ort_dtype_name(out.dtype)
            );
        }
        if f.scalar {
            if !out.shape.is_empty() {
                deny!("scalar attribute {} requires a scalar output", f.name);
            }
        } else if out.shape.len() != 1 {
            deny!("list attribute {} requires a rank-1 output", f.name);
        }
        matched += 1;
    }
    require!(
        matched == 1,
        "requires exactly one supported scalar or list value attribute"
    );
    Ok(())
}

// =============================================================================================
// OneHot — depth categories along `axis`; off/on values gathered from the 2-element `values`.
// =============================================================================================
fn one_hot_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let indices = ctx.resolve(&n.inputs[0])?;
    let depth = read_const_i64(ctx, &n.inputs[1])? as i32;
    let values = ctx.resolve(&n.inputs[2])?;
    let output_rank = ctx.ndim(indices) as i32 + 1;
    let mut axis = n.ints.get("axis").copied().unwrap_or(-1) as i32;
    if axis < 0 {
        axis += output_rank;
    }

    let idt = ctx.dtype_of(indices);
    let categories =
        ctx.emit(|res, s| unsafe { mlx::mlx_arange(res, 0.0, depth as f64, 1.0, idt, s) })?;
    let mut cat_shape = vec![1i32; output_rank as usize];
    cat_shape[axis as usize] = depth;
    let categories = ctx.reshape(categories, &cat_shape)?;

    let zero = scalar_like(ctx, 0, indices)?;
    let negative = ctx.binary(mlx::mlx_less, indices, zero)?;
    let depth_s = scalar_like(ctx, depth as i64, indices)?;
    let wrapped = ctx.add(indices, depth_s)?;
    let normalized =
        ctx.emit(|res, s| unsafe { mlx::mlx_where(res, negative, wrapped, indices, s) })?;

    let expanded = ctx.expand_dims(normalized, axis)?;
    let selected = ctx.binary(mlx::mlx_equal, expanded, categories)?;

    let zero_i = ctx.scalar_i64(0);
    let one_i = ctx.scalar_i64(1);
    let off = ctx.emit(|res, s| unsafe { mlx::mlx_take(res, values, zero_i, s) })?;
    let on = ctx.emit(|res, s| unsafe { mlx::mlx_take(res, values, one_i, s) })?;
    let out = ctx.emit(|res, s| unsafe { mlx::mlx_where(res, selected, on, off, s) })?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn is_boundary_value_type(t: ort::ONNXTensorElementDataType) -> bool {
    is_mlx_float(t)
        || is_signed_integer(t)
        || (crate::registry::is_unsigned_integer(t)
            && t != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64)
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
}

fn one_hot_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 3 && node.num_outputs() == 1,
        "expects 3 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (indices, depth, values, out) = match (
        node.input_info(0),
        node.input_info(1),
        node.input_info(2),
        node.output_info(0),
    ) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    if !is_int_index(indices.dtype)
        || depth.dtype != T_INT64
        || values.shape != [2]
        || values.dtype != out.dtype
        || !is_boundary_value_type(values.dtype)
    {
        deny!(
            "indices must be int32/int64; depth int64; values a 2-element supported dtype matching output"
        );
    }
    let depth_val = match node.const_scalar_i64(1) {
        Some(d) if d > 0 => d,
        _ => deny!("depth must be a positive constant int64 scalar"),
    };
    let output_rank = indices.shape.len() as i64 + 1;
    let mut axis = node.int_attr("axis", -1);
    if axis < 0 {
        axis += output_rank;
    }
    if axis < 0 || axis >= output_rank || out.shape.len() as i64 != output_rank {
        deny!(
            "axis must be in output rank and output shape must equal indices shape with depth inserted"
        );
    }
    let mut expected = indices.shape.clone();
    expected.insert(axis as usize, depth_val);
    require!(
        expected == out.shape,
        "output shape must equal indices shape with depth inserted at axis"
    );
    Ok(())
}

// =============================================================================================
// Trilu — upper/lower triangular retain with diagonal offset `k`.
// =============================================================================================
fn trilu_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let k = if present(n, 1) {
        read_const_i64(ctx, &n.inputs[1])? as i32
    } else {
        0
    };
    let upper = n.ints.get("upper").is_none_or(|&v| v != 0);
    let out = ctx.emit(|res, s| unsafe {
        if upper {
            mlx::mlx_triu(res, x, k, s)
        } else {
            mlx::mlx_tril(res, x, k, s)
        }
    })?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn trilu_claim(node: &NodeView) -> ClaimResult {
    let ni = node.num_inputs();
    require!(
        (1..=2).contains(&ni) && node.num_outputs() == 1,
        "expects 1 or 2 inputs and 1 output, got {ni}in/{}out",
        node.num_outputs()
    );
    let (x, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(a), Some(b)) => (a, b),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    if x.shape.len() < 2 || x.dtype != out.dtype || !is_boundary_value_type(x.dtype) {
        deny!("input must have rank >= 2 and a supported dtype matching output");
    }
    if ni == 2 && node.input_present(1) && node.const_scalar_i64(1).is_none() {
        deny!("diagonal offset must be a constant int scalar");
    }
    let upper = node.int_attr("upper", 1);
    require!(upper == 0 || upper == 1, "upper must be 0 or 1");
    Ok(())
}

// =============================================================================================
// Scatter (deprecated, opset 9/10) — alias of ScatterElements (reduction=none): put_along_axis.
// =============================================================================================
fn scatter_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let data = ctx.resolve(&n.inputs[0])?;
    let indices = ctx.resolve(&n.inputs[1])?;
    let updates = ctx.resolve(&n.inputs[2])?;
    let rank = ctx.ndim(data) as i32;
    let mut axis = n.ints.get("axis").copied().unwrap_or(0) as i32;
    if axis < 0 {
        axis += rank;
    }
    let dim = ctx.dim(data, axis);

    let idx = ctx.astype(indices, I32)?;
    let zero = ctx.scalar_i32(0);
    let neg = ctx.binary(mlx::mlx_less, idx, zero)?;
    let dim_s = ctx.scalar_i32(dim);
    let wrapped = ctx.add(idx, dim_s)?;
    let norm = ctx.emit(|res, s| unsafe { mlx::mlx_where(res, neg, wrapped, idx, s) })?;
    let r =
        ctx.emit(|res, s| unsafe { mlx::mlx_put_along_axis(res, data, norm, updates, axis, s) })?;
    let cont = ctx.contiguous(r)?;
    ctx.bind(&n.outputs[0], cont);
    Ok(())
}

fn scatter_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 3 && node.num_outputs() == 1,
        "expects 3 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (data, indices, updates, out) = match (
        node.input_info(0),
        node.input_info(1),
        node.input_info(2),
        node.output_info(0),
    ) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    if !static_tensor(&data)
        || !static_tensor(&indices)
        || !static_tensor(&updates)
        || !static_tensor(&out)
    {
        deny!("all input and output shapes must be static");
    }
    require!(
        is_mlx_float(data.dtype)
            && is_int_index(indices.dtype)
            && updates.dtype == data.dtype
            && out.dtype == data.dtype,
        "data/updates/output must share a float dtype and indices must be int32/int64"
    );
    require!(
        !data.shape.is_empty()
            && indices.shape == updates.shape
            && indices.shape.len() == data.shape.len()
            && out.shape == data.shape,
        "data must be non-scalar; indices/updates ranks and shapes must match; output must match data"
    );
    Ok(())
}

// =============================================================================================
// Det — matrix determinant of the last two (square) dims. Host-computed in double.
// =============================================================================================
fn det_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let xf = ctx.astype(x, F32)?;
    let ev = ctx.contiguous_eval(xf)?;

    let shape = ctx.shape_of(ev);
    let rank = shape.len();
    let m = shape[rank - 1] as usize;
    let mut batch: usize = 1;
    for &d in &shape[..rank - 2] {
        batch *= d as usize;
    }
    let src = ctx.host_ptr(ev) as *const f32;
    let mut out = vec![0f32; batch];
    let mut a = vec![0f64; m * m];
    for (b, out_value) in out.iter_mut().enumerate() {
        for (i, value) in a.iter_mut().enumerate() {
            *value = unsafe { *src.add(b * m * m + i) } as f64;
        }
        let mut det = 1.0f64;
        for col in 0..m {
            let mut pivot = col;
            let mut best = a[col * m + col].abs();
            for row in (col + 1)..m {
                let val = a[row * m + col].abs();
                if val > best {
                    best = val;
                    pivot = row;
                }
            }
            if best == 0.0 {
                det = 0.0;
                break;
            }
            if pivot != col {
                for k in 0..m {
                    a.swap(pivot * m + k, col * m + k);
                }
                det = -det;
            }
            let diag = a[col * m + col];
            det *= diag;
            for row in (col + 1)..m {
                let factor = a[row * m + col] / diag;
                for k in col..m {
                    a[row * m + k] -= factor * a[col * m + k];
                }
            }
        }
        *out_value = det as f32;
    }
    let out_shape: Vec<i32> = shape[..rank - 2].to_vec();
    let res = ctx.from_host(out.as_ptr() as *const c_void, &out_shape, F32);
    let res = ctx.astype(res, ctx.dtype_of(x))?;
    ctx.bind(&n.outputs[0], res);
    Ok(())
}

fn det_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(a), Some(b)) => (a, b),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    if !static_tensor(&x) || !is_mlx_float(x.dtype) || out.dtype != x.dtype {
        deny!("input shape must be static and input/output must share a float dtype");
    }
    let rank = x.shape.len();
    require!(
        rank >= 2 && x.shape[rank - 1] > 0 && x.shape[rank - 1] == x.shape[rank - 2],
        "input must end in non-empty square matrix dimensions"
    );
    Ok(())
}

// =============================================================================================
// Optional family.
// =============================================================================================
fn optional_has_element_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let present = match n.inputs[0].source {
        Src::CtxInput => ctx.ctx_input_is_present(n.inputs[0].ctx_index)?,
        Src::Absent => false,
        // Container-producing nodes are never claimed at an EP boundary, so an intermediate here
        // can only be a tensor value, which is necessarily present.
        Src::Initializer | Src::Intermediate => true,
    };
    let t = ctx.scalar_bool(present);
    ctx.bind(&n.outputs[0], t);
    Ok(())
}

fn optional_has_element_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    match node.input_kind(0) {
        Some(ValueKind::Tensor(_)) => {}
        Some(kind @ ValueKind::Optional(_)) if kind.optional_tensor().is_some() => {}
        Some(kind) => deny!(
            "OptionalHasElement: {} input is unsupported; only tensor and optional(tensor) values can cross the MLX EP boundary",
            kind.description()
        ),
        None => deny!("OptionalHasElement: input is absent"),
    }
    match node.output_info(0) {
        Some(info)
            if info.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
                && info.shape.is_empty() => {}
        Some(info) => deny!(
            "OptionalHasElement: output must be a scalar bool tensor, got {} shape {:?}",
            crate::registry::ort_dtype_name(info.dtype),
            info.shape
        ),
        None => deny!("OptionalHasElement: output must be a tensor"),
    }
    Ok(())
}

fn optional_get_element_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    if n.inputs[0].source == Src::CtxInput && !ctx.ctx_input_is_present(n.inputs[0].ctx_index)? {
        return Err("MLX OptionalGetElement: input optional is empty".to_string());
    }
    let a = ctx.resolve(&n.inputs[0])?;
    ctx.bind(&n.outputs[0], a);
    Ok(())
}

fn sequence_length_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let count = i64::try_from(ctx.sequence_len(&n.inputs[0])?)
        .map_err(|_| "MLX SequenceLength: count exceeds int64".to_string())?;
    let result = ctx.scalar_i64(count);
    ctx.bind(&n.outputs[0], result);
    Ok(())
}

fn sequence_at_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let count = i64::try_from(ctx.sequence_len(&n.inputs[0])?)
        .map_err(|_| "MLX SequenceAt: count exceeds int64".to_string())?;
    let position = ctx.resolve(&n.inputs[1])?;
    if ctx.ndim(position) != 0 {
        return Err("MLX SequenceAt: position must be a scalar".into());
    }
    let position = ctx.read_scalar_i64(position)?;
    let index = if position < 0 {
        position + count
    } else {
        position
    };
    if index < 0 || index >= count {
        return Err(format!(
            "MLX SequenceAt: position {position} is out of range for {count} tensors"
        ));
    }
    let result = ctx.sequence_element(&n.inputs[0], index as usize, n.outputs[0].otype)?;
    ctx.bind(&n.outputs[0], result);
    Ok(())
}

fn concat_from_sequence_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let count = ctx.sequence_len(&n.inputs[0])?;
    if count == 0 {
        return Err("MLX ConcatFromSequence: input sequence must not be empty".into());
    }
    let new_axis = n.ints.get("new_axis").copied().unwrap_or(0);
    if !matches!(new_axis, 0 | 1) {
        return Err("MLX ConcatFromSequence: new_axis must be 0 or 1".into());
    }
    let axis = *n
        .ints
        .get("axis")
        .ok_or_else(|| "MLX ConcatFromSequence: axis is required".to_string())?;
    let first = ctx.sequence_element(&n.inputs[0], 0, n.outputs[0].otype)?;
    let first_shape = ctx.shape_of(first);
    let rank = first_shape.len() as i64 + new_axis;
    if axis < -rank || axis >= rank {
        return Err(format!(
            "MLX ConcatFromSequence: axis {axis} is invalid for rank {rank}"
        ));
    }
    let axis = if axis < 0 { axis + rank } else { axis } as usize;
    if new_axis == 1 {
        crate::engine::dim_i32(
            i64::try_from(count)
                .map_err(|_| "MLX ConcatFromSequence: sequence is too long".to_string())?,
        )?;
    }
    let mut arrays = VectorArray::new();
    let mut extent = 0i64;
    for index in 0..count {
        let array = if index == 0 {
            first
        } else {
            ctx.sequence_element(&n.inputs[0], index, n.outputs[0].otype)?
        };
        let mut shape = ctx.shape_of(array);
        if shape.len() != first_shape.len()
            || shape
                .iter()
                .zip(&first_shape)
                .enumerate()
                .any(|(i, (a, b))| a != b && (new_axis == 1 || i != axis))
        {
            return Err("MLX ConcatFromSequence: incompatible element shapes".into());
        }
        let array = if new_axis == 1 {
            shape.insert(axis, 1);
            ctx.reshape(array, &shape)?
        } else {
            extent = extent
                .checked_add(i64::from(shape[axis]))
                .ok_or_else(|| "MLX ConcatFromSequence: axis extent overflow".to_string())?;
            crate::engine::dim_i32(extent)?;
            array
        };
        arrays.append(array);
    }
    let result = ctx.emit(|res, stream| unsafe {
        mlx::mlx_concatenate_axis(res, arrays.as_raw(), axis as i32, stream)
    })?;
    ctx.bind(&n.outputs[0], result);
    Ok(())
}

fn sequence_claim(node: &NodeView) -> ClaimResult {
    let op = node.op_type();
    let inputs = if op == "SequenceAt" { 2 } else { 1 };
    require!(
        node.num_inputs() == inputs && node.num_outputs() == 1,
        "sequence operator has invalid input/output arity"
    );
    let element = match node.input_kind(0) {
        Some(ValueKind::Sequence(inner)) => match *inner {
            ValueKind::Tensor(info) => info,
            _ => deny!("only sequences of tensors are supported"),
        },
        _ => deny!("input must be a tensor sequence"),
    };
    let output = node
        .output_info(0)
        .ok_or_else(|| "output must be a tensor".to_string())?;
    if op == "SequenceLength" {
        require!(output.dtype == T_INT64, "length output must be int64");
    } else {
        require!(
            is_movable(element.dtype) && element.dtype == output.dtype,
            "sequence elements and output must share an MLX-supported dtype"
        );
    }
    if op == "SequenceAt" {
        let position = node
            .input_info(1)
            .ok_or_else(|| "position must be an integer tensor".to_string())?;
        require!(
            is_int_index(position.dtype),
            "position must be int32 or int64"
        );
        require!(position.shape.is_empty(), "position must be a scalar");
    } else if op == "ConcatFromSequence" {
        require!(
            matches!(node.int_attr("new_axis", 0), 0 | 1),
            "new_axis must be 0 or 1"
        );
        require!(node.has_attr("axis"), "axis is required");
    }
    Ok(())
}

fn optional_get_element_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1 && node.input_present(0),
        "requires one present input and one output"
    );
    let input = match node.input_kind(0) {
        Some(ValueKind::Tensor(info)) => info,
        // ORT's plugin-EP boundary provides no presence bit or safe unwrap operation for an empty
        // optional. Treating the non-null empty placeholder as a tensor crashes in ORT before this
        // EP can return the standard OptionalGetElement error, so the real optional form stays on
        // CPU. A bare tensor is retained for legacy graphs that ORT permits for this operator.
        Some(ValueKind::Optional(_)) => deny!(
            "OptionalGetElement: optional(tensor) may be empty and ORT 1.29's plugin EP boundary cannot safely unwrap it; stays on CPU"
        ),
        Some(kind) => deny!(
            "OptionalGetElement: {} input is unsupported; only optional(tensor) is supported",
            kind.description()
        ),
        None => deny!("OptionalGetElement: input is absent"),
    };
    match node.output_info(0) {
        Some(b) => {
            require!(
                is_mlx_supported(input.dtype) && b.dtype == input.dtype,
                "input/output must share an MLX-supported dtype"
            );
            Ok(())
        }
        None => deny!("OptionalGetElement: output must be a tensor"),
    }
}

/// `KernelContext_GetOutput` accepts dimensions and allocates only a tensor. Although the OrtApi
/// can construct standalone sequences, the plugin-EP Compute ABI has no operation to assign one to
/// a fused-node output, so claiming a producer would make the model invalid at session creation.
fn container_output_op(_ctx: &mut TranslationContext, _n: &NodeDesc) -> Result<(), MlxError> {
    Err(
        "MLX: container-producing nodes cannot execute at the ORT 1.29 plugin EP boundary"
            .to_string(),
    )
}

fn optional_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "Optional expects 1 optional input and 1 output"
    );
    match node.output_kind(0) {
        Some(ValueKind::Optional(inner)) if inner.tensor().is_some() => deny!(
            "Optional: optional(tensor) output cannot cross the ORT 1.29 plugin EP boundary; KernelContext_GetOutput only allocates tensors"
        ),
        Some(kind) => deny!(
            "Optional: {} output is unsupported; only optional(tensor) is representable by this operator and container outputs stay on CPU",
            kind.description()
        ),
        None => deny!("Optional: output type information is unavailable"),
    }
}

// =============================================================================================
// NegativeLogLikelihoodLoss / SoftmaxCrossEntropyLoss.
// =============================================================================================
fn loss_common(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    apply_log_softmax: bool,
) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let t = ctx.resolve(&n.inputs[1])?;
    let fdt = ctx.dtype_of(x);
    let has_weight = present(n, 2);

    let mut logp = x;
    if apply_log_softmax {
        let lse = ctx.emit(|res, s| unsafe { mlx::mlx_logsumexp_axis(res, x, 1, true, s) })?;
        logp = ctx.sub(x, lse)?;
        if n.outputs.len() > 1 && !n.outputs[1].name.is_empty() {
            ctx.bind(&n.outputs[1], logp);
        }
    }

    let ti = ctx.astype(t, I32)?;
    let has_ignore = n.ints.contains_key("ignore_index");
    let mut mask = ti; // only meaningful when has_ignore
    let mut safe_t = ti;
    if has_ignore {
        let ig = ctx.scalar_i32(*n.ints.get("ignore_index").unwrap() as i32);
        mask = ctx.binary(mlx::mlx_equal, ti, ig)?;
        let zeros = ctx.zeros_like(ti)?;
        safe_t = ctx.emit(|res, s| unsafe { mlx::mlx_where(res, mask, zeros, ti, s) })?;
    }

    let idx_e = ctx.expand_dims(safe_t, 1)?;
    let gathered =
        ctx.emit(|res, s| unsafe { mlx::mlx_take_along_axis(res, logp, idx_e, 1, s) })?;
    let picked = ctx.squeeze(gathered, 1)?;
    let mut loss = ctx.emit(|res, s| unsafe { mlx::mlx_negative(res, picked, s) })?;

    let mut w_at = loss; // only meaningful when have_w_at
    let mut have_w_at = false;
    if has_weight {
        let w = ctx.resolve(&n.inputs[2])?;
        w_at = ctx.emit(|res, s| unsafe { mlx::mlx_take(res, w, safe_t, s) })?;
        have_w_at = true;
        loss = ctx.mul(loss, w_at)?;
    }
    if has_ignore {
        let zeros = ctx.zeros_like(loss)?;
        loss = ctx.emit(|res, s| unsafe { mlx::mlx_where(res, mask, zeros, loss, s) })?;
        if have_w_at {
            let wz = ctx.zeros_like(w_at)?;
            w_at = ctx.emit(|res, s| unsafe { mlx::mlx_where(res, mask, wz, w_at, s) })?;
        }
    }

    let reduction = str_attr(n, "reduction", "mean");
    if reduction == "none" {
        ctx.bind(&n.outputs[0], loss);
        return Ok(());
    }
    let sum = ctx.emit(|res, s| unsafe { mlx::mlx_sum(res, loss, false, s) })?;
    if reduction == "sum" {
        ctx.bind(&n.outputs[0], sum);
        return Ok(());
    }
    // mean: divide by the sum of the (non-ignored) weights, or the non-ignored element count.
    let denom = if have_w_at {
        ctx.emit(|res, s| unsafe { mlx::mlx_sum(res, w_at, false, s) })?
    } else if has_ignore {
        let keep = ctx.emit(|res, s| unsafe { mlx::mlx_logical_not(res, mask, s) })?;
        let keepf = ctx.astype(keep, fdt)?;
        ctx.emit(|res, s| unsafe { mlx::mlx_sum(res, keepf, false, s) })?
    } else {
        let c = ctx.scalar_f32(ctx.size_of(ti) as f32);
        ctx.astype(c, fdt)?
    };
    let mean = ctx.binary(mlx::mlx_divide, sum, denom)?;
    ctx.bind(&n.outputs[0], mean);
    Ok(())
}

fn nll_loss_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    loss_common(ctx, n, false)
}

fn sce_loss_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    loss_common(ctx, n, true)
}

fn loss_claim(node: &NodeView, sce: bool) -> ClaimResult {
    let ni = node.num_inputs();
    require!((2..=3).contains(&ni), "expects 2 or 3 inputs, got {ni}");
    let max_out = if sce { 2 } else { 1 };
    let no = node.num_outputs();
    require!(
        no >= 1 && no <= max_out,
        "expects 1 to {max_out} outputs, got {no}"
    );
    let (x, t) = match (node.input_info(0), node.input_info(1)) {
        (Some(a), Some(b)) => (a, b),
        _ => deny!("missing tensor type/shape info on an input"),
    };
    require!(
        is_mlx_float(x.dtype) && is_int_index(t.dtype) && x.shape.len() >= 2,
        "scores must be rank >= 2 float and targets must be int32/int64"
    );
    if node.input_present(2) {
        match node.input_info(2) {
            Some(w) if w.dtype == x.dtype && w.shape.len() == 1 => {}
            _ => deny!("weight must be rank-1 with the scores dtype"),
        }
    }
    let reduction = node.string_attr("reduction", "mean");
    require!(
        reduction == "mean" || reduction == "sum" || reduction == "none",
        "reduction must be mean, sum, or none (got {reduction})"
    );
    Ok(())
}

fn nll_loss_claim(node: &NodeView) -> ClaimResult {
    loss_claim(node, false)
}

fn sce_loss_claim(node: &NodeView) -> ClaimResult {
    loss_claim(node, true)
}

// ---- registration -------------------------------------------------------------------------------

fn shapeless(
    registry: &mut OpRegistry,
    op_type: &'static str,
    min_opset: i32,
    max_opset: i32,
    handler: OpHandler,
    claim: ClaimPredicate,
) {
    registry.register_shapeless(OpRegistration {
        domain: "",
        op_type,
        min_opset,
        max_opset,
        handler,
        claim,
    });
}

fn shape_keyed(
    registry: &mut OpRegistry,
    op_type: &'static str,
    min_opset: i32,
    max_opset: i32,
    handler: OpHandler,
    claim: ClaimPredicate,
    reason: &'static str,
) {
    registry.register_shape_keyed(
        OpRegistration {
            domain: "",
            op_type,
            min_opset,
            max_opset,
            handler,
            claim,
        },
        reason,
    );
}

fn eager(
    registry: &mut OpRegistry,
    op_type: &'static str,
    min_opset: i32,
    max_opset: i32,
    handler: OpHandler,
    claim: ClaimPredicate,
    reason: &'static str,
) {
    registry.register_eager(
        OpRegistration {
            domain: "",
            op_type,
            min_opset,
            max_opset,
            handler,
            claim,
        },
        reason,
    );
}

pub fn register(registry: &mut OpRegistry) {
    eager(
        registry,
        "SequenceLength",
        11,
        K_ANY_OPSET,
        sequence_length_op,
        sequence_claim,
        "conservatively reads live sequence metadata from the ORT kernel context",
    );
    eager(
        registry,
        "SequenceAt",
        11,
        K_ANY_OPSET,
        sequence_at_op,
        sequence_claim,
        "conservatively copies live sequence elements from the ORT kernel context",
    );
    eager(
        registry,
        "ConcatFromSequence",
        11,
        K_ANY_OPSET,
        concat_from_sequence_op,
        sequence_claim,
        "conservatively copies live sequence elements from the ORT kernel context",
    );
    shapeless(
        registry,
        "Constant",
        K_ANY_OPSET,
        K_ANY_OPSET,
        constant_op,
        constant_claim,
    );
    shapeless(
        registry,
        "TfIdfVectorizer",
        9,
        K_ANY_OPSET,
        tfidf_op,
        tfidf_claim,
    );
    shapeless(
        registry,
        "OneHot",
        9,
        K_ANY_OPSET,
        one_hot_op,
        one_hot_claim,
    );
    shapeless(registry, "Trilu", 14, K_ANY_OPSET, trilu_op, trilu_claim);
    shape_keyed(
        registry,
        "Scatter",
        9,
        10,
        scatter_op,
        scatter_claim,
        crate::registry::MLX_SCATTER_SHAPE_REASON,
    );
    shapeless(registry, "Det", K_ANY_OPSET, K_ANY_OPSET, det_op, det_claim);
    shapeless(
        registry,
        "Optional",
        15,
        K_ANY_OPSET,
        container_output_op,
        optional_claim,
    );
    eager(
        registry,
        "OptionalHasElement",
        K_ANY_OPSET,
        K_ANY_OPSET,
        optional_has_element_op,
        optional_has_element_claim,
        "reads optional presence from the ORT kernel context at execution time",
    );
    shapeless(
        registry,
        "OptionalGetElement",
        K_ANY_OPSET,
        K_ANY_OPSET,
        optional_get_element_op,
        optional_get_element_claim,
    );
    shapeless(
        registry,
        "NegativeLogLikelihoodLoss",
        K_ANY_OPSET,
        K_ANY_OPSET,
        nll_loss_op,
        nll_loss_claim,
    );
    shapeless(
        registry,
        "SoftmaxCrossEntropyLoss",
        K_ANY_OPSET,
        K_ANY_OPSET,
        sce_loss_op,
        sce_loss_claim,
    );
}
