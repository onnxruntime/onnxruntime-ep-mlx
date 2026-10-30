//! GetCapability policy and partition orchestration.
//!
//! This module owns the short-lived ORT graph inspection required to apply registry
//! policy. Topology always comes from [`OrtGraphSnapshot`]; raw `OrtNode` handles are
//! used only while this callback is active for registry attribute predicates.

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString, c_char};
use std::ptr;

use crate::engine::is_separate_qkv_attention_op;
use crate::factory::ORT_API_VERSION;
use crate::ort_graph::{NodeMetadata, OrtGraphSnapshot};
use crate::partition::{
    build_contiguous_clusters, build_convex_clusters, infer_layer_boundary_values,
    split_annotated_layer_clusters,
};
use crate::registry::{ClaimResult, CompilePartitionClass, NodeView, claimable};
use crate::sys::ort;

fn native_candidate_claim(
    native_candidate: bool,
    op_type: &str,
    tile_payload: impl FnOnce() -> ClaimResult,
) -> Option<ClaimResult> {
    if !native_candidate {
        return None;
    }
    match op_type {
        "Range" => Some(Ok(())),
        "Tile" => Some(tile_payload()),
        _ => None,
    }
}

/// Apply EP claim policy to one optimized graph.
///
/// # Safety
/// All ABI handles are borrowed from ORT and must remain valid for this call. No
/// borrowed pointer is retained after this function returns.
pub(crate) unsafe fn get_capability(
    ort_api: *const ort::OrtApi,
    ep_api: *const ort::OrtEpApi,
    graph: *const ort::OrtGraph,
    support: *mut ort::OrtEpGraphSupportInfo,
) -> *mut ort::OrtStatus {
    unsafe {
        let api = &*ort_api;
        let ep_api = &*ep_api;
        let snapshot = match OrtGraphSnapshot::from_ort(api, graph) {
            Ok(snapshot) => snapshot,
            Err(error) => return ep_fail_status(api, error),
        };
        let node_order = match snapshot.ir.topological_order() {
            Ok(order) => order,
            Err(error) => {
                return ep_fail_status(
                    api,
                    format!("optimized graph IR has no topological order: {error}"),
                );
            }
        };
        if node_order.is_empty() {
            return ptr::null_mut();
        }

        let mut raw_nodes: Vec<*const ort::OrtNode> = vec![ptr::null(); node_order.len()];
        let st = (api.Graph_GetNodes.unwrap())(graph, raw_nodes.as_mut_ptr(), raw_nodes.len());
        if !st.is_null() {
            return st;
        }
        let mut raw_nodes_by_id = HashMap::with_capacity(raw_nodes.len());
        for node in raw_nodes {
            let mut ort_node_id = 0usize;
            let st = (api.Node_GetId.unwrap())(node, &mut ort_node_id);
            if !st.is_null() {
                return st;
            }
            let Some(&node_id) = snapshot.node_by_ort_id.get(&ort_node_id) else {
                return ep_fail_status(
                    api,
                    format!("optimized graph snapshot is missing ORT node ID {ort_node_id}"),
                );
            };
            raw_nodes_by_id.insert(node_id, node);
        }

        // ORT partitions control-flow bodies before their parent. Never create a
        // private-domain fusion inside a nested body; the parent is claimed whole or
        // ORT retains the body on CPU.
        let mut in_cf_body = false;
        if let Some(get_parent) = api.Graph_GetParentNode {
            let mut parent: *const ort::OrtNode = ptr::null();
            let st = get_parent(graph, &mut parent);
            if st.is_null() && !parent.is_null() {
                in_cf_body = true;
            } else if !st.is_null() {
                release_status(api, st);
            }
        }

        // Foundry Q8 Whisper keeps its dynamic position-ID Range/Tile pair together
        // with the decoder so ORT does not introduce CPU islands.
        let native_q8_attention_graph = node_order.len() > 32
            && node_order.iter().any(|&node| {
                let metadata = &snapshot.nodes[&node];
                is_separate_qkv_attention_op(&metadata.domain, &metadata.op_type)
            })
            && node_order.iter().any(|&node| {
                let view = NodeView::new(ort_api, raw_nodes_by_id[&node]);
                view.op_type() == "MatMulNBits" && view.int_attr("bits", 4) == 8
            })
            && node_order
                .iter()
                .filter(|&&node| {
                    let view = NodeView::new(ort_api, raw_nodes_by_id[&node]);
                    view.op_type() == "Range"
                        && view.read_const_scalar_f64(2) == Some(1.0)
                        && (0..3).any(|index| view.read_const_scalar_f64(index).is_none())
                })
                .count()
                == 1
            && node_order
                .iter()
                .filter(|&&node| {
                    let view = NodeView::new(ort_api, raw_nodes_by_id[&node]);
                    view.op_type() == "Tile" && !view.is_const_int64(1)
                })
                .count()
                == 1;

        let mixed_vision_quant_graph = node_order.iter().any(|&node| {
            let metadata = &snapshot.nodes[&node];
            metadata.domain == "com.microsoft" && metadata.op_type == "PackedMultiHeadAttention"
        }) && node_order.iter().any(|&node| {
            let metadata = &snapshot.nodes[&node];
            metadata.domain == "com.microsoft" && metadata.op_type == "MatMulNBits"
        });

        let supported: Vec<bool> = node_order
            .iter()
            .map(|&node_id| {
                if in_cf_body {
                    return false;
                }
                let view = NodeView::new(ort_api, raw_nodes_by_id[&node_id]);
                if let Some(decision) =
                    native_candidate_claim(native_q8_attention_graph, &view.op_type(), || {
                        crate::ops::shape::tile_payload_dtype(&view).map(|_| ())
                    })
                {
                    return decision.is_ok();
                }
                claimable(&view)
            })
            .collect();
        let float64: Vec<bool> = node_order
            .iter()
            .map(|&node| {
                crate::registry::node_uses_float64(&NodeView::new(ort_api, raw_nodes_by_id[&node]))
            })
            .collect();
        let attention_anchors = node_order
            .iter()
            .copied()
            .filter(|node| is_decoder_attention_anchor(&snapshot.nodes[node]))
            .collect::<Vec<_>>();
        let decoder_graph = !attention_anchors.is_empty();
        let packed_qdq_nodes = if decoder_graph {
            packed_qdq_eager_nodes(
                &snapshot,
                &node_order,
                &supported,
                ort_api,
                &raw_nodes_by_id,
            )
        } else {
            HashSet::new()
        };
        let compile_class: Vec<CompilePartitionClass> = node_order
            .iter()
            .map(|&node| {
                let view = NodeView::new(ort_api, raw_nodes_by_id[&node]);
                let metadata = &snapshot.nodes[&node];
                let mod_boundary = metadata.domain.is_empty() && metadata.op_type == "Mod";
                let nonzero_boundary = metadata.domain.is_empty() && metadata.op_type == "NonZero";
                let live_tile_boundary = crate::registry::requires_live_tile_eager_boundary(&view);
                let native_live_tile = live_tile_boundary
                    && native_q8_attention_graph
                    && matches!(view.domain().as_str(), "" | "ai.onnx")
                    && view.op_type() == "Tile"
                    && view.subgraphs().is_empty();
                let live_range_boundary =
                    crate::registry::requires_live_range_eager_boundary(&view);
                let native_live_range = live_range_boundary
                    && native_q8_attention_graph
                    && view.op_type() == "Range"
                    && view.subgraphs().is_empty();
                if packed_qdq_nodes.contains(&node)
                    || crate::registry::requires_random_eager_boundary(&view)
                    || crate::registry::requires_unique_eager_boundary(&view)
                    || (live_range_boundary && !native_live_range)
                    || crate::registry::requires_elementary_contrib_eager_boundary(&view)
                    || crate::registry::requires_cdist_eager_boundary(&view)
                    || (live_tile_boundary && !native_live_tile)
                    || crate::registry::requires_float64_metadata_eager_boundary(&view)
                {
                    CompilePartitionClass::Eager
                } else if native_live_tile || native_live_range {
                    CompilePartitionClass::Shapeless
                } else if decoder_graph || mod_boundary || nonzero_boundary {
                    crate::registry::compile_shape_safety(&view).partition_class()
                } else {
                    CompilePartitionClass::Shapeless
                }
            })
            .collect();

        let tr = crate::trace::tracer();
        let mut rejected: Vec<(String, usize, String, Vec<String>)> = Vec::new();
        if tr.active() || std::env::var_os("ONNXRUNTIME_EP_MLX_CLAIM_DEBUG").is_some() {
            use std::collections::BTreeMap;
            let mut acc: BTreeMap<String, (usize, String, Vec<String>)> = BTreeMap::new();
            for (&node, &ok) in node_order.iter().zip(supported.iter()) {
                if !ok {
                    let view = NodeView::new(ort_api, raw_nodes_by_id[&node]);
                    let entry = acc
                        .entry(crate::registry::qualified_op_name(&view))
                        .or_insert((0, String::new(), Vec::new()));
                    entry.0 += 1;
                    if entry.1.is_empty() {
                        entry.1 = if in_cf_body {
                            "inside a control-flow subgraph body — claimed as part of the parent \
                             If/Loop/Scan, not individually"
                                .to_string()
                        } else {
                            native_candidate_claim(
                                native_q8_attention_graph,
                                &view.op_type(),
                                || crate::ops::shape::tile_payload_dtype(&view).map(|_| ()),
                            )
                            .unwrap_or_else(|| crate::registry::claim_decision(&view))
                            .err()
                            .map(|cause| cause.into_owned())
                            .unwrap_or_else(|| "declined (no reason reported)".to_string())
                        };
                    }
                    if entry.2.len() < 16 {
                        let name = view.name();
                        if !name.is_empty() {
                            entry.2.push(name);
                        }
                    }
                }
            }
            rejected = acc
                .into_iter()
                .map(|(op, (count, reason, names))| (op, count, reason, names))
                .collect();
            rejected.sort_by_key(|entry| std::cmp::Reverse(entry.1));
            if std::env::var_os("ONNXRUNTIME_EP_MLX_CLAIM_DEBUG").is_some() {
                for (op, count, reason, names) in &rejected {
                    log::debug!("unclaimed {op} x{count} ({reason}): {names:?}");
                }
            }
        }

        let clusters = if mixed_vision_quant_graph {
            build_contiguous_clusters(&node_order, &supported, &float64, &compile_class)
        } else {
            build_convex_clusters(
                &snapshot.ir,
                &node_order,
                &supported,
                &float64,
                &compile_class,
            )
        };
        let layer_boundary_outputs = if in_cf_body {
            HashSet::new()
        } else {
            match graph_metadata_value(api, graph, c"onnxruntime_ep_mlx.layer_boundary_outputs") {
                Ok(Some(value)) => match serde_json::from_str::<Vec<String>>(&value) {
                    Ok(outputs) => outputs
                        .into_iter()
                        .filter_map(|name| snapshot.value_by_name.get(&name).copied())
                        .filter(|&value| snapshot.ir.value(value).producer.is_some())
                        .collect(),
                    Err(error) => {
                        log::warn!(
                            "ignoring invalid onnxruntime_ep_mlx.layer_boundary_outputs metadata: \
                             {error}"
                        );
                        HashSet::new()
                    }
                },
                Ok(None) => {
                    infer_layer_boundary_values(&snapshot.ir, &node_order, &attention_anchors)
                }
                Err(st) => return st,
            }
        };
        let layer_partition_span = select_layer_partition_span(
            std::env::var("ONNXRUNTIME_EP_MLX_LAYER_PARTITIONS")
                .ok()
                .as_deref(),
            layer_boundary_outputs.len(),
        );
        let clusters = match layer_partition_span {
            Some(span) => split_annotated_layer_clusters(
                &snapshot.ir,
                clusters,
                &layer_boundary_outputs,
                span,
            ),
            None => clusters,
        };

        let add_fuse = ep_api.EpGraphSupportInfo_AddNodesToFuse.unwrap();
        let mut claimed = 0usize;
        for cluster in &clusters {
            let group: Vec<*const ort::OrtNode> =
                cluster.iter().map(|node| raw_nodes_by_id[node]).collect();
            let mut options: ort::OrtNodeFusionOptions = std::mem::zeroed();
            options.ort_version_supported = ORT_API_VERSION;
            options.drop_constant_initializers = true;
            let st = add_fuse(support, group.as_ptr(), group.len(), &options);
            if !st.is_null() {
                return st;
            }
            claimed += cluster.len();
        }
        log::debug!(
            "capability: {claimed}/{} nodes in {} fused subgraphs",
            node_order.len(),
            clusters.len()
        );
        tr.record_claim(claimed, node_order.len(), clusters.len(), &rejected);
        ptr::null_mut()
    }
}

fn packed_qdq_eager_nodes(
    snapshot: &OrtGraphSnapshot,
    order: &[onnx_runtime_ir::NodeId],
    supported: &[bool],
    api: *const ort::OrtApi,
    raw_nodes: &HashMap<onnx_runtime_ir::NodeId, *const ort::OrtNode>,
) -> HashSet<onnx_runtime_ir::NodeId> {
    let supported: HashSet<_> = order
        .iter()
        .copied()
        .zip(supported.iter().copied())
        .filter_map(|(node, ok)| ok.then_some(node))
        .collect();
    let producers: HashMap<_, _> = order
        .iter()
        .flat_map(|&node| {
            snapshot.nodes[&node]
                .output_slots
                .iter()
                .flatten()
                .map(move |&value| (value, node))
        })
        .collect();
    let mut result = HashSet::new();
    for &product in order {
        if !supported.contains(&product) {
            continue;
        }
        let view = NodeView::new(api, raw_nodes[&product]);
        if !crate::ops::qdq::partition_product_candidate(&view) {
            continue;
        }
        let inputs = &snapshot.nodes[&product].input_slots;
        let producer = |slot: usize| {
            inputs
                .get(slot)
                .copied()
                .flatten()
                .and_then(|value| producers.get(&value).copied())
        };
        let Some(weight) = producer(1).filter(|node| supported.contains(node)) else {
            continue;
        };
        let Some(weight_packed) =
            crate::ops::qdq::partition_dq_candidate(&NodeView::new(api, raw_nodes[&weight]), true)
        else {
            continue;
        };
        let mut activation = None;
        let mut activation_packed = false;
        if let Some(input) = producer(0) {
            let metadata = &snapshot.nodes[&input];
            if metadata.domain.is_empty() && metadata.op_type == "DequantizeLinear" {
                if !supported.contains(&input) {
                    continue;
                }
                let Some(packed) = crate::ops::qdq::partition_dq_candidate(
                    &NodeView::new(api, raw_nodes[&input]),
                    false,
                ) else {
                    continue;
                };
                activation = Some(input);
                activation_packed = packed;
            }
        }
        if weight_packed || activation_packed {
            result.extend([product, weight]);
            result.extend(activation);
        }
    }
    result
}

fn select_layer_partition_span(value: Option<&str>, layer_count: usize) -> Option<usize> {
    match value.map(str::trim) {
        Some("0" | "off" | "false") => None,
        Some("" | "auto") | None => {
            (layer_count >= 23).then(|| layer_count.div_ceil(7).clamp(4, 8))
        }
        Some(value) => value.parse::<usize>().ok().filter(|&span| span > 0),
    }
}

fn is_decoder_attention_anchor(node: &NodeMetadata) -> bool {
    if node.domain == "com.microsoft" && node.op_type == "PagedAttention" {
        return true;
    }
    let has_present_cache = node.output_slots.get(1).is_some_and(Option::is_some)
        && node.output_slots.get(2).is_some_and(Option::is_some);
    has_present_cache
        && ((node.domain == "com.microsoft"
            && matches!(
                node.op_type.as_str(),
                "GroupQueryAttention" | "MultiHeadAttention"
            ))
            || (node.domain.is_empty() && node.op_type == "Attention"))
}

unsafe fn ep_fail_status(api: &ort::OrtApi, message: impl AsRef<str>) -> *mut ort::OrtStatus {
    let message = CString::new(message.as_ref())
        .unwrap_or_else(|_| CString::new("MLX capability inspection failed").unwrap());
    unsafe { (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, message.as_ptr()) }
}

unsafe fn release_status(api: &ort::OrtApi, status: *mut ort::OrtStatus) {
    if !status.is_null() {
        unsafe { (api.ReleaseStatus.unwrap())(status) };
    }
}

unsafe fn graph_metadata_value(
    api: &ort::OrtApi,
    graph: *const ort::OrtGraph,
    key: &CStr,
) -> Result<Option<String>, *mut ort::OrtStatus> {
    unsafe {
        let mut metadata: *mut ort::OrtModelMetadata = ptr::null_mut();
        let status = (api.Graph_GetModelMetadata.unwrap())(graph, &mut metadata);
        if !status.is_null() {
            return Err(status);
        }
        let mut allocator: *mut ort::OrtAllocator = ptr::null_mut();
        let status = (api.GetAllocatorWithDefaultOptions.unwrap())(&mut allocator);
        if !status.is_null() {
            (api.ReleaseModelMetadata.unwrap())(metadata);
            return Err(status);
        }
        let mut value: *mut c_char = ptr::null_mut();
        let status = (api.ModelMetadataLookupCustomMetadataMap.unwrap())(
            metadata,
            allocator,
            key.as_ptr(),
            &mut value,
        );
        if !status.is_null() {
            (api.ReleaseModelMetadata.unwrap())(metadata);
            return Err(status);
        }
        let result = if value.is_null() {
            None
        } else {
            let owned = CStr::from_ptr(value).to_string_lossy().into_owned();
            ((*allocator).Free.unwrap())(allocator, value.cast());
            Some(owned)
        };
        (api.ReleaseModelMetadata.unwrap())(metadata);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::{native_candidate_claim, select_layer_partition_span};
    use crate::ops::shape::validate_tile_payload;
    use crate::registry::{ClaimResult, SlotInfo};
    use crate::sys::ort::*;

    fn candidate_tile_claim(
        input: Option<SlotInfo>,
        output: Option<SlotInfo>,
        since_version: i32,
    ) -> ClaimResult {
        native_candidate_claim(true, "Tile", || {
            validate_tile_payload(
                input
                    .zip(output)
                    .map(|(input, output)| (input.dtype, output.dtype)),
                since_version,
            )
            .map(|_| ())
        })
        .expect("native Tile must take the override")
    }

    fn slot(dtype: ONNXTensorElementDataType, shape: &[i64]) -> Option<SlotInfo> {
        Some(SlotInfo {
            dtype,
            shape: shape.to_vec(),
        })
    }

    #[test]
    fn native_tile_override_preserves_all_movable_payloads() {
        for dtype in [
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
        ] {
            // Shapes are deliberately not an admission input: [] also represents
            // unknown-rank tensor metadata in SlotInfo, not absent metadata.
            for shape in [&[][..], &[2, 3], &[-1, 3], &[0, 3], &[i64::MAX, 0]] {
                assert_eq!(
                    candidate_tile_claim(slot(dtype, shape), slot(dtype, shape), 13),
                    Ok(()),
                    "dtype {dtype}, shape {shape:?}"
                );
            }
        }
    }

    #[test]
    fn native_tile_override_rejects_unsupported_or_missing_payload_metadata() {
        let f32 = ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        let string = ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
        for (input, output) in [
            (None, slot(f32, &[])),
            (slot(f32, &[]), None),
            (None, None),
            (slot(string, &[]), None),
            (None, slot(string, &[])),
        ] {
            assert_eq!(
                candidate_tile_claim(input, output, 13).unwrap_err(),
                "missing tensor type/shape info on the data input or output"
            );
        }
        for dtype in [
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
        ] {
            for output in [dtype, f32] {
                assert_eq!(
                    candidate_tile_claim(slot(dtype, &[2]), slot(output, &[2]), 13).unwrap_err(),
                    format!(
                        "data dtype {} is not a movable dtype supported on GPU",
                        crate::registry::ort_dtype_name(dtype)
                    )
                );
            }
        }
    }

    #[test]
    fn native_tile_override_preserves_mismatch_and_bfloat16_diagnostic_order() {
        let bf16 = ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
        let f32 = ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        for (input, output, version) in [
            (bf16, f32, 6),
            (bf16, f32, 13),
            (f32, bf16, 13),
            (
                f32,
                ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
                13,
            ),
            (
                f32,
                ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
                13,
            ),
            (
                f32,
                ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
                13,
            ),
        ] {
            assert_eq!(
                candidate_tile_claim(slot(input, &[]), slot(output, &[]), version).unwrap_err(),
                format!(
                    "output dtype {} must match data dtype {}",
                    crate::registry::ort_dtype_name(output),
                    crate::registry::ort_dtype_name(input)
                )
            );
        }
        assert_eq!(
            candidate_tile_claim(slot(bf16, &[]), slot(bf16, &[]), 6).unwrap_err(),
            "Tile: bfloat16 requires opset 13"
        );
        assert_eq!(
            candidate_tile_claim(slot(bf16, &[]), slot(bf16, &[]), 13),
            Ok(())
        );
    }

    #[test]
    fn native_override_is_lazy_and_does_not_run_the_full_tile_claim() {
        for (candidate, op_type, expected) in [
            (false, "Tile", None),
            (false, "Range", None),
            (true, "Range", Some(Ok(()))),
            (true, "Add", None),
        ] {
            assert_eq!(
                native_candidate_claim(candidate, op_type, || {
                    panic!("must not evaluate the Tile payload guard")
                }),
                expected
            );
        }
        let normal_called = std::cell::Cell::new(false);
        let normal_claim = || -> ClaimResult {
            normal_called.set(true);
            Err("ordinary claim rejects geometry".into())
        };
        let dtype = ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        assert_eq!(
            native_candidate_claim(true, "Tile", || {
                validate_tile_payload(Some((dtype, dtype)), 13).map(|_| ())
            })
            .unwrap_or_else(normal_claim),
            Ok(())
        );
        assert!(!normal_called.get());
        assert_eq!(
            native_candidate_claim(true, "Tile", || Err("payload rejected".into()))
                .unwrap_or_else(normal_claim),
            Err("payload rejected".into())
        );
        assert!(!normal_called.get());
        assert_eq!(
            native_candidate_claim(false, "Tile", || panic!("not native"))
                .unwrap_or_else(normal_claim),
            Err("ordinary claim rejects geometry".into())
        );
        assert!(normal_called.get());
    }

    #[test]
    fn auto_layer_partition_span_scales_with_decoder_size() {
        assert_eq!(select_layer_partition_span(None, 16), None);
        assert_eq!(select_layer_partition_span(None, 23), Some(4));
        assert_eq!(select_layer_partition_span(None, 24), Some(4));
        assert_eq!(select_layer_partition_span(None, 52), Some(8));
        assert_eq!(select_layer_partition_span(Some("auto"), 80), Some(8));
    }

    #[test]
    fn layer_partition_span_accepts_overrides() {
        assert_eq!(select_layer_partition_span(Some("0"), 52), None);
        assert_eq!(select_layer_partition_span(Some("off"), 52), None);
        assert_eq!(select_layer_partition_span(Some("5"), 52), Some(5));
        assert_eq!(select_layer_partition_span(Some("invalid"), 52), None);
    }
}
