//! Private frozen, blocked 4-bit lookup decoding followed by FP32 MatMul.
//! Recognition proves the standard ONNX producer chain; public ONNX remains unchanged.

use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::sync::Arc;

use crate::engine::{InitData, MlxError, NodeDesc, OutRef, Src, TensorRef, TranslationContext};
use crate::mlx::{Array, VectorArray};
use crate::registry::{CompileShapeSafety, NodeView};
use crate::sys::{mlx, ort};

const F32: mlx::mlx_dtype = mlx::mlx_dtype__MLX_FLOAT32;
const FLOAT: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
const INT64: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
const UINT8: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;

#[derive(Clone)]
pub(crate) struct LookupMatMul {
    packed: TensorRef,
    scales: TensorRef,
    book: [f32; 16],
    rows: i32,
    cols: i32,
    block: i32,
    transpose: bool,
}

struct Producers<'a>(HashMap<&'a str, &'a NodeDesc>, bool);

impl<'a> Producers<'a> {
    fn op(&self, value: &TensorRef, op: &str, inputs: usize) -> Option<&'a NodeDesc> {
        let node = *self.0.get(value.name.as_str())?;
        (node.domain.is_empty()
            && node.op_type == op
            && node.inputs.len() == inputs
            && node.outputs.len() == 1)
            .then_some(node)
    }

    fn literal(
        &self,
        value: &TensorRef,
    ) -> Option<(ort::ONNXTensorElementDataType, Vec<i64>, Vec<u8>)> {
        if let Some(init) = &value.init {
            if !(value.source == Src::Initializer || value.constant)
                || init.count > 16
                || init.data.is_null()
            {
                return None;
            }
            let width = if init.dtype == INT64 {
                8
            } else if init.dtype == FLOAT {
                4
            } else {
                return None;
            };
            // Initializer storage is session-owned (or Arc-owned by this descriptor).
            let data =
                unsafe { std::slice::from_raw_parts(init.data.cast::<u8>(), init.count * width) };
            return Some((init.dtype, init.shape.clone(), data.to_vec()));
        }
        let tensor = self.op(value, "Constant", 0)?.tensors.get("value")?;
        (tensor.count <= 16).then(|| (tensor.dtype, tensor.shape.clone(), tensor.data.clone()))
    }

    fn ints(&self, value: &TensorRef) -> Option<Vec<i64>> {
        let (dtype, _, data) = self.literal(value)?;
        if dtype != INT64 || data.len() % 8 != 0 {
            return None;
        }
        Some(
            data.as_chunks::<8>()
                .0
                .iter()
                .map(|v| i64::from_ne_bytes(*v))
                .collect(),
        )
    }

    fn reshape(&self, value: &TensorRef, shape: &[i64]) -> Option<TensorRef> {
        let node = self.op(value, "Reshape", 2)?;
        (self.ints(&node.inputs[1])? == shape).then(|| node.inputs[0].clone())
    }

    fn unsqueeze(&self, value: &TensorRef) -> Option<TensorRef> {
        let node = self.op(value, "Unsqueeze", 2)?;
        (self.ints(&node.inputs[1])? == [1]).then(|| node.inputs[0].clone())
    }

    fn recognize(&self, node: &NodeDesc) -> Option<LookupMatMul> {
        if !node.domain.is_empty()
            || node.op_type != "MatMul"
            || node.inputs.len() != 2
            || node.outputs.len() != 1
            || node.outputs[0].otype != FLOAT
        {
            return None;
        }
        let mut weight = node.inputs[1].clone();
        let transpose = if let Some(parent) = self.op(&weight, "Transpose", 1) {
            if parent.int_arrays.get("perm").is_some_and(|p| p != &[1, 0]) {
                return None;
            }
            weight = parent.inputs[0].clone();
            true
        } else {
            false
        };
        let reshape = self.op(&weight, "Reshape", 2)?;
        let shape = self.ints(&reshape.inputs[1])?;
        if shape.len() != 2 {
            return None;
        }
        let rows = i32::try_from(shape[0]).ok()?;
        let cols = i32::try_from(shape[1]).ok()?;
        let size = rows.checked_mul(cols)?;
        if rows <= 0 || cols <= 0 {
            return None;
        }
        let slice = *self.0.get(reshape.inputs[0].name.as_str())?;
        if !slice.domain.is_empty()
            || slice.op_type != "Slice"
            || slice.outputs.len() != 1
            || !(4..=5).contains(&slice.inputs.len())
            || self.ints(&slice.inputs[1])? != [0]
            || self.ints(&slice.inputs[2])? != [i64::from(size)]
            || self.ints(&slice.inputs[3])? != [0]
            || slice
                .inputs
                .get(4)
                .is_some_and(|v| self.ints(v).as_deref() != Some(&[1]))
        {
            return None;
        }
        let scaled = self.reshape(&slice.inputs[0], &[-1])?;
        let multiply = self.op(&scaled, "Mul", 2)?;
        let (gather, scales) = if let Some(gather) = self.op(&multiply.inputs[0], "Gather", 2) {
            (gather, self.unsqueeze(&multiply.inputs[1])?)
        } else {
            (
                self.op(&multiply.inputs[1], "Gather", 2)?,
                self.unsqueeze(&multiply.inputs[0])?,
            )
        };
        if gather.ints.get("axis").copied().unwrap_or(0) != 0 {
            return None;
        }
        let (dtype, book_shape, bytes) = self.literal(&gather.inputs[0])?;
        if dtype != FLOAT || book_shape != [16] || bytes.len() != 64 {
            return None;
        }
        let book: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_ne_bytes(*v))
            .collect();
        let index_reshape = self.op(&gather.inputs[1], "Reshape", 2)?;
        let index_shape = self.ints(&index_reshape.inputs[1])?;
        if index_shape.len() != 2 || index_shape[0] != -1 {
            return None;
        }
        let block = i32::try_from(index_shape[1]).ok()?;
        if block <= 0 || block % 2 != 0 {
            return None;
        }
        let concat = self.op(&index_reshape.inputs[0], "Concat", 2)?;
        if concat.ints.get("axis").copied() != Some(1) {
            return None;
        }
        let high = self.unsqueeze(&concat.inputs[0])?;
        let low = self.unsqueeze(&concat.inputs[1])?;
        let high = self.op(&high, "Div", 2)?;
        let low = self.op(&low, "Mod", 2)?;
        if high.inputs[0].name != low.inputs[0].name
            || self.ints(&high.inputs[1])? != [16]
            || self.ints(&low.inputs[1])? != [16]
            || low.ints.get("fmod").copied().unwrap_or(0) != 0
        {
            return None;
        }
        let cast = self.op(&high.inputs[0], "Cast", 1)?;
        if cast.ints.get("to").copied() != Some(i64::from(INT64)) {
            return None;
        }
        let packed = cast.inputs[0].clone();
        let p = packed.init.as_ref()?;
        let s = scales.init.as_ref()?;
        let blocks = (i64::from(size) + i64::from(block) - 1) / i64::from(block);
        if !(packed.source == Src::Initializer || packed.constant)
            || !(scales.source == Src::Initializer || scales.constant)
            || p.dtype != UINT8
            || s.dtype != FLOAT
            || p.shape != [blocks * i64::from(block) / 2]
            || s.shape != [blocks]
            || p.count as i64 != p.shape[0]
            || s.count as i64 != blocks
            || self.1 && (p.data.is_null() || s.data.is_null())
        {
            return None;
        }
        let n = if transpose { rows } else { cols };
        (n / 16 + i32::from(n % 16 != 0)).checked_mul(128)?;
        Some(LookupMatMul {
            packed,
            scales,
            book: book.try_into().ok()?,
            rows,
            cols,
            block,
            transpose,
        })
    }
}

pub(crate) fn recognize(nodes: &mut Vec<NodeDesc>, graph_outputs: &HashSet<String>) {
    if std::env::var_os("ONNXRUNTIME_EP_MLX_NO_LUT4_FUSION")
        .is_some_and(|v| v != "0" && !v.is_empty())
    {
        return;
    }
    let (found, candidates) = {
        let producers = Producers(
            nodes
                .iter()
                .flat_map(|n| n.outputs.iter().map(move |v| (v.name.as_str(), n)))
                .collect(),
            true,
        );
        let found = nodes
            .iter()
            .map(|n| producers.recognize(n))
            .collect::<Vec<_>>();
        let indices: HashMap<_, _> = nodes
            .iter()
            .enumerate()
            .flat_map(|(i, n)| n.outputs.iter().map(move |v| (v.name.as_str(), i)))
            .collect();
        let mut candidates = HashSet::new();
        for (i, matched) in found.iter().enumerate() {
            if matched.is_some() {
                let mut pending = vec![nodes[i].inputs[1].name.as_str()];
                while let Some(name) = pending.pop() {
                    if let Some(&parent) = indices.get(name)
                        && candidates.insert(parent)
                    {
                        pending.extend(nodes[parent].inputs.iter().map(|v| v.name.as_str()));
                    }
                }
            }
        }
        (found, candidates)
    };
    for (node, pattern) in nodes.iter_mut().zip(found) {
        if let Some(pattern) = pattern {
            // Internal dependencies drive compiled-region boundaries. Other consumers still
            // depend on the original decode chain, including externally visible weights.
            node.inputs = vec![
                node.inputs[0].clone(),
                pattern.packed.clone(),
                pattern.scales.clone(),
            ];
            node.lut4 = Some(Box::new(pattern));
            node.compile_shape_safety = CompileShapeSafety::EagerOnly {
                reason: "packed lookup MatMul validates live geometry and dispatches a specialized kernel",
            };
        }
    }
    fn captured_names(nodes: &[NodeDesc], used: &mut HashSet<String>) {
        for node in nodes {
            used.extend(node.inputs.iter().map(|v| v.name.clone()));
            for child in &node.subgraphs {
                used.extend(child.output_names.iter().cloned());
                captured_names(&child.nodes, used);
            }
        }
    }
    let mut used = graph_outputs.clone();
    for node in nodes.iter() {
        used.extend(
            node.outputs
                .iter()
                .filter(|v| v.external)
                .map(|v| v.name.clone()),
        );
        for child in &node.subgraphs {
            used.extend(child.output_names.iter().cloned());
            captured_names(&child.nodes, &mut used);
        }
    }
    let mut keep = vec![true; nodes.len()];
    for (index, node) in nodes.iter().enumerate().rev() {
        if candidates.contains(&index) && node.outputs.iter().all(|v| !used.contains(&v.name)) {
            keep[index] = false;
        } else {
            used.extend(node.inputs.iter().map(|v| v.name.clone()));
        }
    }
    let mut index = 0;
    nodes.retain(|_| {
        let retain = keep[index];
        index += 1;
        retain
    });
}

/// Use the same semantic proof at claim time, retaining only bounded literals.
/// Large frozen buffers carry metadata here; execution descriptors require real storage.
pub(crate) fn partition_regions(views: &[NodeView], supported: &[bool]) -> Vec<Vec<usize>> {
    if std::env::var_os("ONNXRUNTIME_EP_MLX_NO_LUT4_FUSION")
        .is_some_and(|v| v != "0" && !v.is_empty())
    {
        return Vec::new();
    }
    if !views
        .iter()
        .any(|v| v.domain().is_empty() && v.op_type() == "Mod")
    {
        return Vec::new();
    }
    let mut nodes = Vec::with_capacity(views.len());
    for view in views {
        let mut node = NodeDesc::new(
            view.op_type(),
            view.domain(),
            view.since_version(),
            CompileShapeSafety::EagerOnly {
                reason: "claim-time lookup4 metadata",
            },
        );
        for key in match node.op_type.as_str() {
            "Cast" => &["to"][..],
            "Concat" | "Gather" => &["axis"][..],
            "Mod" => &["fmod"][..],
            _ => &[][..],
        } {
            node.ints.insert(
                (*key).into(),
                view.int_attr(key, if *key == "to" { -1 } else { 0 }),
            );
        }
        if node.op_type == "Transpose" {
            let (present, perm) = view.ints_attr("perm");
            if present {
                node.int_arrays.insert("perm".into(), perm);
            }
        }
        for (index, name) in view.input_names().into_iter().enumerate() {
            let mut input = TensorRef::absent();
            input.name = name;
            input.source = Src::Intermediate;
            if view.is_constant_initializer(index)
                && let Some(info) = view.input_info(index)
            {
                let count = info
                    .shape
                    .iter()
                    .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?));
                if let Some(count) = count {
                    let bytes = if count <= 16 && info.dtype == INT64 {
                        view.read_const_int64(index)
                            .map(|v| v.into_iter().flat_map(i64::to_ne_bytes).collect::<Vec<_>>())
                    } else if count <= 16 && info.dtype == FLOAT {
                        view.read_const_f32(index)
                            .map(|v| v.into_iter().flat_map(f32::to_ne_bytes).collect::<Vec<_>>())
                    } else {
                        None
                    };
                    let owned = bytes.map(Arc::new);
                    input.source = Src::Initializer;
                    input.constant = true;
                    input.init = Some(InitData {
                        data: owned
                            .as_ref()
                            .map_or(std::ptr::null(), |v| v.as_ptr().cast()),
                        shape: info.shape,
                        dtype: info.dtype,
                        count,
                        owned,
                    });
                }
            }
            node.inputs.push(input);
        }
        for (index, name) in view.output_names().into_iter().enumerate() {
            node.outputs.push(OutRef {
                name,
                external: false,
                ctx_index: 0,
                otype: view.output_info(index).map_or(
                    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
                    |v| v.dtype,
                ),
            });
        }
        nodes.push(node);
    }
    let producer_indices: HashMap<_, _> = nodes
        .iter()
        .enumerate()
        .flat_map(|(i, n)| n.outputs.iter().map(move |v| (v.name.as_str(), i)))
        .collect();
    let producers = Producers(
        nodes
            .iter()
            .flat_map(|n| n.outputs.iter().map(move |v| (v.name.as_str(), n)))
            .collect(),
        false,
    );
    let mut selected = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        if !supported[index] || producers.recognize(node).is_none() {
            continue;
        }
        let mut members = HashSet::from([index]);
        let mut pending = vec![node.inputs[1].name.as_str()];
        while let Some(name) = pending.pop() {
            if let Some(&parent) = producer_indices.get(name)
                && members.insert(parent)
            {
                pending.extend(nodes[parent].inputs.iter().map(|v| v.name.as_str()));
            }
        }
        if members.iter().all(|&i| supported[i]) {
            let mut members: Vec<_> = members.into_iter().collect();
            members.sort_unstable();
            selected.push(members);
        }
    }
    selected
}

const SOURCE: &str = r#"
uint tid = thread_index_in_threadgroup;
uint sg = tid / 32;
uint row0 = threadgroup_position_in_grid.y * 16;
uint col0 = threadgroup_position_in_grid.x * 16;
threadgroup float a[256];
threadgroup float b[256];
threadgroup float c[256];
metal::simdgroup_float8x8 acc = metal::make_filled_simdgroup_matrix<float,8>(0.0f);
metal::simdgroup_float8x8 correction = metal::make_filled_simdgroup_matrix<float,8>(0.0f);
for (uint kk=0; kk<K; kk+=16) {
    metal::simdgroup_float8x8 partial = metal::make_filled_simdgroup_matrix<float,8>(0.0f);
    for (uint i=tid; i<256; i+=128) {
        uint row=row0+i/16, inner=kk+i%16;
        a[i]=(row<M && inner<K) ? x[row*K+inner] : 0.0f;
        uint k=kk+i/16, col=col0+i%16;
        float value=0.0f;
        if (k<K && col<N) {
            uint flat=TRANSPOSE ? col*COLS+k : k*COLS+col;
            uint code=packed[flat/2];
            code=(flat%2==0) ? code>>4 : code&15;
            value=book[code]*scales[flat/BLOCK];
        }
        b[i]=value;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for (uint offset=0; offset<16; offset+=8) {
        metal::simdgroup_float8x8 aa,bb;
        metal::simdgroup_load(aa,a+(sg/2)*128+offset,16);
        metal::simdgroup_load(bb,b+offset*16+(sg%2)*8,16);
        metal::simdgroup_multiply_accumulate(partial,aa,bb,partial);
    }
    for (uint lane=0; lane<2; ++lane) {
        float value=partial.thread_elements()[lane]-correction.thread_elements()[lane];
        float added=acc.thread_elements()[lane]+value;
        correction.thread_elements()[lane]=metal::isfinite(added)
            ? (added-acc.thread_elements()[lane])-value : 0.0f;
        acc.thread_elements()[lane]=added;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}
metal::simdgroup_store(acc,c+(sg/2)*128+(sg%2)*8,16);
metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
for (uint i=tid; i<256; i+=128) {
    uint row=row0+i/16,col=col0+i%16;
    if (row<M && col<N) y[row*N+col]=c[i];
}
"#;

pub(super) fn execute(ctx: &mut TranslationContext, node: &NodeDesc) -> Result<bool, MlxError> {
    let Some(pattern) = &node.lut4 else {
        return Ok(false);
    };
    let x = ctx.resolve(&node.inputs[0])?;
    let mut shape = ctx.shape_of(x);
    let k = if pattern.transpose {
        pattern.cols
    } else {
        pattern.rows
    };
    let n = if pattern.transpose {
        pattern.rows
    } else {
        pattern.cols
    };
    if ctx.dtype_of(x) != F32 || shape.last().copied() != Some(k) {
        return Err("lookup4 MatMul: expected FP32 input with matching reduction dimension".into());
    }
    shape.pop();
    let m = shape
        .iter()
        .try_fold(1i32, |m, &d| m.checked_mul(d))
        .ok_or("lookup4 MatMul: batch size overflow")?;
    shape.push(n);
    m.checked_mul(k)
        .and_then(|_| m.checked_mul(n))
        .ok_or("lookup4 MatMul: flattened tensor exceeds int32 indexing")?;
    let mut device = unsafe { mlx::mlx_device_new() };
    let mut kind = mlx::mlx_device_type__MLX_CPU;
    let mut rc = unsafe { mlx::mlx_stream_get_device(&mut device, ctx.stream()) };
    if rc == 0 {
        rc = unsafe { mlx::mlx_device_get_type(&mut kind, device) };
    }
    unsafe { mlx::mlx_device_free(device) };
    if rc != 0 {
        return Err("lookup4 MatMul: cannot inspect stream device".into());
    }
    let y = if m == 0 {
        ctx.zeros(&shape, F32)?
    } else if kind == mlx::mlx_device_type__MLX_CPU {
        let key = format!(
            "__lookup4_cpu__{}:{}:{}:{}:{}:{:?}",
            pattern.packed.cache_key(),
            pattern.scales.cache_key(),
            pattern.rows,
            pattern.cols,
            pattern.block,
            pattern.book.map(f32::to_bits)
        );
        let weight = if let Some(cached) = ctx.cache_get(&key) {
            cached
        } else {
            let p = pattern.packed.init.as_ref().unwrap();
            let s = pattern.scales.init.as_ref().unwrap();
            let mut values = Vec::with_capacity((pattern.rows * pattern.cols) as usize);
            for index in 0..pattern.rows * pattern.cols {
                let byte = unsafe { *p.data.cast::<u8>().add(index as usize / 2) };
                let code = if index % 2 == 0 { byte >> 4 } else { byte & 15 };
                let scale = unsafe {
                    std::ptr::read_unaligned(
                        s.data.cast::<f32>().add((index / pattern.block) as usize),
                    )
                };
                values.push(pattern.book[code as usize] * scale);
            }
            let owned =
                Array::from_data(values.as_ptr().cast(), &[pattern.rows, pattern.cols], F32);
            ctx.cache_put(key, owned)
        };
        let weight = if pattern.transpose {
            ctx.emit(|out, stream| unsafe { mlx::mlx_transpose(out, weight, stream) })?
        } else {
            weight
        };
        ctx.mark_composed("frozen lookup4 decode + CPU matmul");
        ctx.binary(mlx::mlx_matmul, x, weight)?
    } else {
        let x = ctx.contiguous(x)?;
        let packed = ctx.resolve(&pattern.packed)?;
        let scales = ctx.resolve(&pattern.scales)?;
        let book_key = format!("__lookup4_book__{:?}", pattern.book.map(f32::to_bits));
        let book = if let Some(cached) = ctx.cache_get(&book_key) {
            cached
        } else {
            ctx.cache_put(
                book_key,
                Array::from_data(pattern.book.as_ptr().cast(), &[16], F32),
            )
        };
        let mut inputs = VectorArray::new();
        for value in [x, packed, scales, book] {
            inputs.append(value);
        }
        let names = super::selective_scan::VecString::new(&["x", "packed", "scales", "book"])?;
        let outputs = super::selective_scan::VecString::new(&["y"])?;
        let source = CString::new(SOURCE).unwrap();
        let kernel = unsafe {
            mlx::mlx_fast_metal_kernel_new(
                c"mlx_lookup4_matmul".as_ptr(),
                names.as_raw(),
                outputs.as_raw(),
                source.as_ptr(),
                c"".as_ptr(),
                true,
                false,
            )
        };
        if kernel.ctx.is_null() {
            return Err("lookup4 MatMul: cannot create Metal kernel".into());
        }
        let config = unsafe { mlx::mlx_fast_metal_kernel_config_new() };
        if config.ctx.is_null() {
            unsafe { mlx::mlx_fast_metal_kernel_free(kernel) };
            return Err("lookup4 MatMul: cannot create Metal configuration".into());
        }
        let mut rc = 0;
        for (name, value) in [
            ("M", m),
            ("K", k),
            ("N", n),
            ("COLS", pattern.cols),
            ("BLOCK", pattern.block),
            ("TRANSPOSE", i32::from(pattern.transpose)),
        ] {
            let name = CString::new(name).unwrap();
            rc |= unsafe {
                mlx::mlx_fast_metal_kernel_config_add_template_arg_int(config, name.as_ptr(), value)
            };
        }
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_output_arg(
                config,
                shape.as_ptr(),
                shape.len(),
                F32,
            )
        };
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_set_grid(
                config,
                (n / 16 + i32::from(n % 16 != 0)) * 128,
                m / 16 + i32::from(m % 16 != 0),
                1,
            )
        };
        rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1) };
        let mut output = VectorArray::new();
        if rc == 0 {
            rc = unsafe {
                mlx::mlx_fast_metal_kernel_apply(
                    output.as_mut_ptr(),
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
        if rc != 0 || output.size() != 1 {
            return Err("lookup4 MatMul: Metal dispatch failed".into());
        }
        ctx.mark_fast("mlx_lookup4_matmul");
        ctx.keep(output.get(0))
    };
    ctx.bind(&node.outputs[0], y);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Plan;
    use crate::mlx::Stream;

    fn initializer(
        name: &str,
        bytes: Vec<u8>,
        dtype: ort::ONNXTensorElementDataType,
        count: usize,
    ) -> TensorRef {
        let owned = Arc::new(bytes);
        let mut value = TensorRef::absent();
        value.name = name.into();
        value.source = Src::Initializer;
        value.init = Some(InitData {
            data: owned.as_ptr().cast(),
            shape: vec![count as i64],
            dtype,
            count,
            owned: Some(owned),
        });
        value
    }

    #[test]
    fn cpu_products_keep_branch_weights_and_transposed_layouts_distinct() {
        let book = std::array::from_fn(|i| (i * i) as f32 / 32.0);
        let patterns = [(0x12, 0.125f32), (0x34, 0.25f32)].map(|(code, scale)| LookupMatMul {
            packed: initializer("packed", vec![code; 32], UINT8, 32),
            scales: initializer("scales", scale.to_ne_bytes().to_vec(), FLOAT, 1),
            book,
            rows: 3,
            cols: 5,
            block: 64,
            transpose: false,
        });
        let stream = Stream::new_default_cpu();
        let mut plan = Plan::new(vec![]);
        let mut ctx = TranslationContext::new(
            &mut plan,
            std::ptr::null(),
            std::ptr::null_mut(),
            stream.as_raw(),
        );
        let output = |name: &str| OutRef {
            name: name.into(),
            external: false,
            ctx_index: 0,
            otype: FLOAT,
        };
        let mut input = TensorRef::absent();
        input.name = "x".into();
        input.source = Src::Intermediate;
        for transpose in [false, true] {
            let (k, n) = if transpose { (5, 3) } else { (3, 5) };
            let values = vec![0.25f32; 2 * k as usize];
            let x = ctx.from_host(values.as_ptr().cast(), &[2, k], F32);
            ctx.bind(&output("x"), x);
            for index in [0, 1, 0, 1] {
                let mut pattern = patterns[index].clone();
                pattern.transpose = transpose;
                let mut node = NodeDesc::new(
                    "MatMul".into(),
                    String::new(),
                    13,
                    CompileShapeSafety::Shapeless,
                );
                node.name = "same_name_in_both_branches".into();
                node.inputs = vec![input.clone()];
                node.outputs = vec![output("y")];
                node.lut4 = Some(Box::new(pattern));
                assert!(execute(&mut ctx, &node).unwrap());
                let mut yref = TensorRef::absent();
                yref.name = "y".into();
                yref.source = Src::Intermediate;
                let y = ctx.resolve(&yref).unwrap();
                let y = ctx.contiguous_eval(y).unwrap();
                assert_eq!(ctx.shape_of(y), vec![2, n]);
                let result = unsafe {
                    std::slice::from_raw_parts(mlx::mlx_array_data_float32(y), (2 * n) as usize)
                };
                let (byte, scale) = if index == 0 {
                    (0x12, 0.125)
                } else {
                    (0x34, 0.25)
                };
                for row in 0..2 {
                    for col in 0..n {
                        let expected = (0..k)
                            .map(|inner| {
                                let flat = if transpose {
                                    col * 5 + inner
                                } else {
                                    inner * 5 + col
                                };
                                let code = if flat % 2 == 0 { byte >> 4 } else { byte & 15 };
                                0.25 * book[code as usize] * scale
                            })
                            .sum::<f32>();
                        assert_eq!(result[(row * n + col) as usize], expected);
                    }
                }
            }
        }
    }
}
