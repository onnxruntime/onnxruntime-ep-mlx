"""Shape and Size inspect tensor metadata, not its payload representation."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper
import onnxruntime as ort
import pytest


def _model(op, dtype, *, shape=None, opset=24, initializer=None, **attrs):
    output = helper.make_tensor_value_info("y", T.INT64, [None] if op == "Shape" else [])
    graph = helper.make_graph(
        [helper.make_node(op, ["x"], ["y"], **attrs)], "metadata_only",
        [] if initializer is not None else [helper.make_tensor_value_info("x", dtype, shape)],
        [output], initializer=[] if initializer is None else [initializer],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)],
                             ir_version=11).SerializeToString()


def _session(model, *, mixed=False, profile=None):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if not mixed:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    session = ort.InferenceSession(
        model, options,
        providers=["MLXExecutionProvider", "CPUExecutionProvider"] if mixed
        else ["MLXExecutionProvider"],
    )
    session.disable_fallback()
    return session


def _expected(op, shape, *, start=0, end=None):
    if op == "Size":
        return np.array(np.prod(shape, dtype=np.int64), np.int64)
    rank = len(shape)
    end = rank if end is None else end
    begin = max(0, min(rank, start + rank if start < 0 else start))
    stop = max(begin, max(0, min(rank, end + rank if end < 0 else end)))
    return np.array(shape[begin:stop], np.int64)


@pytest.mark.parametrize("op", ["Shape", "Size"])
@pytest.mark.parametrize("dtype", [T.STRING, T.FLOAT, T.DOUBLE, T.INT64, T.UINT8, T.BOOL])
@pytest.mark.parametrize("eager", [False, True])
def test_metadata_only_live_rank_and_geometry(op, dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(op, dtype))
    for shape in [(), (3,), (2, 3), (2, 0, 3), (1, 2, 1, 3), (3,)]:
        data = (np.full(shape, "payload", object) if dtype == T.STRING else
                np.full(shape, 1, helper.tensor_dtype_to_np_dtype(dtype)))
        actual = session.run(None, {"x": data})[0]
        expected = _expected(op, shape)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.int64


@pytest.mark.parametrize("start,end", [
    (0, None), (1, 3), (-3, -1), (-20, 20), (3, 1), (8, -8),
    (-(2**63), 2**63 - 1), (2**63 - 1, -(2**63)),
])
@pytest.mark.parametrize("dtype", [T.STRING, T.FLOAT])
def test_metadata_shape_interval_attributes(start, end, dtype):
    attrs = {"start": start}
    if end is not None:
        attrs["end"] = end
    session = _session(_model("Shape", dtype, **attrs))
    for shape in [(), (3,), (2, 3, 4), (1, 0, 3, 4)]:
        data = np.full(shape, "x", object) if dtype == T.STRING else np.zeros(shape, np.float32)
        np.testing.assert_array_equal(
            session.run(None, {"x": data})[0],
            _expected("Shape", shape, start=start, end=end),
        )


@pytest.mark.parametrize("opset", [1, 13, 15, 19, 21, 24, 26])
@pytest.mark.parametrize("op", ["Shape", "Size"])
def test_metadata_string_schema_versions(op, opset):
    model = _model(op, T.STRING, shape=["B", "C"], opset=opset)
    session = _session(model)
    for shape in [(2, 3), (1, 0), (3, 2)]:
        data = np.full(shape, "x", object)
        np.testing.assert_array_equal(session.run(None, {"x": data})[0], _expected(op, shape))


@pytest.mark.parametrize("op", ["Shape", "Size"])
@pytest.mark.parametrize("dtype", [T.STRING, T.FLOAT])
@pytest.mark.parametrize("declared", [None, [0, 2**31 + 7]])
def test_metadata_extents_do_not_narrow_to_mlx_i32(op, dtype, declared):
    shape = (0, 2**31 + 7)
    data = np.empty(shape, object if dtype == T.STRING else np.float32)
    session = _session(_model(op, dtype, shape=declared))
    np.testing.assert_array_equal(session.run(None, {"x": data})[0], _expected(op, shape))


@pytest.mark.parametrize("op", ["Shape", "Size"])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3), (1,) * 12])
def test_metadata_string_initializers(op, shape):
    data = np.full(shape, "initializer", object)
    initializer = numpy_helper.from_array(data, name="x")
    session = _session(_model(op, T.STRING, initializer=initializer))
    np.testing.assert_array_equal(session.run(None, {})[0], _expected(op, shape))


@pytest.mark.parametrize("op", ["Shape", "Size"])
@pytest.mark.parametrize("shape", [(2, 3), (5,)])
@pytest.mark.parametrize("dtype", [
    T.FLOAT8E4M3FN, T.FLOAT8E4M3FNUZ, T.FLOAT8E5M2, T.FLOAT8E5M2FNUZ,
    T.FLOAT8E8M0, T.INT4, T.UINT4, T.FLOAT4E2M1,
])
def test_metadata_encoded_initializers(op, dtype, shape):
    count = int(np.prod(shape))
    packed = dtype in [T.INT4, T.UINT4, T.FLOAT4E2M1]
    raw = bytes((count + 1) // 2 if packed else count)
    initializer = helper.make_tensor("x", dtype, shape, raw, raw=True)
    session = _session(_model(op, dtype, initializer=initializer))
    np.testing.assert_array_equal(session.run(None, {})[0], _expected(op, shape))


@pytest.mark.parametrize("op", ["Shape", "Size"])
def test_metadata_cpu_string_producer(op, tmp_path):
    model = onnx.load_model_from_string(_model(op, T.STRING))
    del model.graph.input[:]
    model.graph.input.extend([
        helper.make_tensor_value_info("a", T.STRING, None),
        helper.make_tensor_value_info("b", T.STRING, None),
    ])
    model.graph.node.insert(0, helper.make_node("StringConcat", ["a", "b"], ["x"]))
    session = _session(model.SerializeToString(), mixed=True, profile=tmp_path / op)
    for shape in [(2,), (2, 3), (0, 2)]:
        data = np.full(shape, "a", object)
        result = session.run(None, {"a": data, "b": data})[0]
        np.testing.assert_array_equal(result, _expected(op, shape))
    events = json.loads(Path(session.end_profiling()).read_text())
    nodes = [event["args"] for event in events if event.get("cat") == "Node"]
    assert any(event.get("provider") == "MLXExecutionProvider" for event in nodes)
    assert any(event.get("provider") == "CPUExecutionProvider"
               and event.get("op_name") == "StringConcat" for event in nodes)


@pytest.mark.parametrize("op", ["Shape", "Size"])
def test_metadata_string_captured_by_if(op):
    branches = []
    for attrs in [{}, {"start": 1} if op == "Shape" else {}]:
        body = onnx.load_model_from_string(_model(op, T.STRING, **attrs)).graph
        del body.input[:]
        branches.append(body)
    graph = helper.make_graph(
        [helper.make_node("If", ["condition"], ["result"],
                          then_branch=branches[0], else_branch=branches[1])],
        "metadata_capture",
        [helper.make_tensor_value_info("condition", T.BOOL, []),
         helper.make_tensor_value_info("x", T.STRING, None)],
        [helper.make_tensor_value_info("result", T.INT64, [None] if op == "Shape" else [])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)], ir_version=11)
    session = _session(model.SerializeToString())
    for shape, condition in [((2, 3), True), ((4,), False), ((), True), ((0, 3, 1), False)]:
        data = np.full(shape, "captured", object)
        result = session.run(None, {"x": data, "condition": np.array(condition)})[0]
        np.testing.assert_array_equal(
            result, _expected(op, shape, start=0 if condition or op == "Size" else 1),
        )


@pytest.mark.parametrize("op", ["Shape", "Size"])
@pytest.mark.parametrize("dtype", [T.STRING, T.FLOAT8E4M3FN, T.UINT4])
def test_metadata_owned_branch_initializers(op, dtype):
    shapes = [(2, 3), (5,)]
    branches = []
    for shape in shapes:
        if dtype == T.STRING:
            initializer = numpy_helper.from_array(np.full(shape, "local", object), name="x")
        else:
            count = int(np.prod(shape))
            raw = bytes((count + 1) // 2 if dtype == T.UINT4 else count)
            initializer = helper.make_tensor("x", dtype, shape, raw, raw=True)
        branches.append(onnx.load_model_from_string(
            _model(op, dtype, initializer=initializer)).graph)
    graph = helper.make_graph(
        [helper.make_node("If", ["condition"], ["result"],
                          then_branch=branches[0], else_branch=branches[1])],
        "metadata_owned_initializers",
        [helper.make_tensor_value_info("condition", T.BOOL, [])],
        [helper.make_tensor_value_info("result", T.INT64, [None] if op == "Shape" else [])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)], ir_version=11)
    session = _session(model.SerializeToString())
    for condition in [True, False, True, False]:
        result = session.run(None, {"condition": np.array(condition)})[0]
        np.testing.assert_array_equal(result, _expected(op, shapes[0 if condition else 1]))


def test_metadata_non_mlx_compile_route(tmp_path):
    trace = tmp_path / "metadata.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "metadata_string_schema_versions", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0", "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path"]
    assert len(paths) == 42
    assert {event["ops"] for event in paths} == {"Shape", "Size"}
    assert all(event["path"] == "eager" and event["cache"] == "n/a" for event in paths)


@pytest.mark.parametrize("escape", ["direct", "cast", "concat_gather"])
@pytest.mark.parametrize("source", ["present_k", "q", "runtime_reshape", "runtime_expand", "runtime_slice"])
def test_numeric_metadata_after_attention_replays_live_geometry(escape, source):
    parameterized = source.startswith("runtime_")
    metadata_input = "shaped" if parameterized else source
    rank = 4 if source == "present_k" else 2 if source == "runtime_reshape" else 3
    nodes = [
        helper.make_node(
            "Attention", ["q", "k", "v", "", "pk", "pv"], ["y", "present_k", "present_v"],
            q_num_heads=1, kv_num_heads=1, is_causal=1,
        ),
    ]
    inputs = [helper.make_tensor_value_info(name, T.FLOAT, [1, 1, 8])
              for name in ["q", "k", "v"]]
    inputs.extend(helper.make_tensor_value_info(name, T.FLOAT, [1, 1, "past", 8])
                  for name in ["pk", "pv"])
    initializers = []
    if source in ["runtime_reshape", "runtime_expand"]:
        nodes.append(helper.make_node("Reshape" if source == "runtime_reshape" else "Expand",
                                      ["y", "target"], ["shaped"]))
        inputs.append(helper.make_tensor_value_info("target", T.INT64, [rank]))
    elif source == "runtime_slice":
        nodes.append(helper.make_node("Slice", ["y", "start", "end", "axis"], ["shaped"]))
        inputs.extend(helper.make_tensor_value_info(name, T.INT64, [1]) for name in ["start", "end"])
        initializers.append(numpy_helper.from_array(np.array([2], np.int64), name="axis"))
    nodes.extend([
        helper.make_node("Shape", [metadata_input], ["geometry"]),
        helper.make_node("Size", [metadata_input], ["count"]),
    ])
    if escape == "cast":
        for index, name in [(len(nodes) - 2, "geometry"), (len(nodes) - 1, "count")]:
            nodes[index].output[0] = f"{name}_raw"
            nodes.extend([
                helper.make_node("Cast", [f"{name}_raw"], [f"{name}_narrow"], to=T.INT32),
                helper.make_node("Cast", [f"{name}_narrow"], [name], to=T.INT64),
            ])
    elif escape == "concat_gather":
        nodes[-2].output[0] = "geometry_raw"
        nodes[-1].output[0] = "count_raw"
        initializers.extend([
            numpy_helper.from_array(np.zeros(rank, np.int64), name="zeros"),
            numpy_helper.from_array(np.arange(rank, dtype=np.int64), name="indices"),
            numpy_helper.from_array(np.array([0], np.int64), name="unsqueeze_axis"),
            numpy_helper.from_array(np.array(0, np.int64), name="scalar_index"),
        ])
        nodes.extend([
            helper.make_node("Concat", ["geometry_raw", "zeros"], ["geometry_joined"], axis=0),
            helper.make_node("Gather", ["geometry_joined", "indices"], ["geometry"]),
            helper.make_node("Unsqueeze", ["count_raw", "unsqueeze_axis"], ["count_vector"]),
            helper.make_node("Concat", ["count_vector", "zeros"], ["count_joined"], axis=0),
            helper.make_node("Gather", ["count_joined", "scalar_index"], ["count"]),
        ])
    nodes.extend([
        helper.make_node("ReduceSum", ["geometry"], ["extent_sum"], keepdims=0),
        helper.make_node("Add", ["extent_sum", "count"], ["metadata_sum"]),
        helper.make_node("Cast", ["metadata_sum"], ["metadata_float"], to=T.FLOAT),
        helper.make_node("Add", ["y", "metadata_float"], ["adjusted_y"]),
    ])
    outputs = [
        helper.make_tensor_value_info("adjusted_y", T.FLOAT, [1, 1, 8]),
        helper.make_tensor_value_info("present_k", T.FLOAT, [1, 1, "total", 8]),
        helper.make_tensor_value_info("present_v", T.FLOAT, [1, 1, "total", 8]),
        helper.make_tensor_value_info("geometry", T.INT64, [rank]),
        helper.make_tensor_value_info("count", T.INT64, []),
    ]
    graph = helper.make_graph(nodes, "live_attention_geometry", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)], ir_version=11)
    session = _session(model.SerializeToString())
    for invocation, past in enumerate([1, 1, 3, 2, 1]):
        if parameterized:
            past = 1
        feeds = {name: np.full((1, 1, 8), past + invocation, np.float32)
                 for name in ["q", "k", "v"]}
        feeds.update({name: np.ones((1, 1, past, 8), np.float32) for name in ["pk", "pv"]})
        expected_shape = (1, 1, past + 1, 8) if source == "present_k" else (1, 1, 8)
        if source == "runtime_reshape":
            expected_shape = [(2, 4), (4, 2), (1, 8), (8, 1), (2, 4)][invocation]
            feeds["target"] = np.array(expected_shape, np.int64)
        elif source == "runtime_expand":
            expected_shape = [(1, 2, 8), (2, 1, 8), (1, 1, 8), (1, 3, 8), (1, 2, 8)][invocation]
            feeds["target"] = np.array(expected_shape, np.int64)
        elif source == "runtime_slice":
            end = [4, 6, 2, 0, 4][invocation]
            feeds.update({"start": np.array([0], np.int64), "end": np.array([end], np.int64)})
            expected_shape = (1, 1, end)
        actual = session.run(None, feeds)
        np.testing.assert_array_equal(actual[1], np.concatenate(
            [feeds["pk"], feeds["k"].reshape(1, 1, 1, 8)], axis=2))
        np.testing.assert_array_equal(actual[2], np.concatenate(
            [feeds["pv"], feeds["v"].reshape(1, 1, 1, 8)], axis=2))
        value = past + invocation
        current_probability = 1 / (1 + past * np.exp(-np.sqrt(8) * value * (value - 1)))
        expected_y = current_probability * value + (1 - current_probability)
        np.testing.assert_allclose(
            actual[0], expected_y + sum(expected_shape) + np.prod(expected_shape),
            rtol=1e-5, atol=1e-6,
        )
        np.testing.assert_array_equal(actual[3], expected_shape)
        np.testing.assert_array_equal(actual[4], np.prod(expected_shape))
    if source == "runtime_reshape":
        valid = feeds["target"].copy()
        feeds["target"] = np.array([3, 3], np.int64)
        with pytest.raises(Exception, match="MLX subgraph failed"):
            session.run(None, feeds)
        feeds["target"] = valid
        np.testing.assert_array_equal(session.run(None, feeds)[3], valid)


def test_metadata_numeric_replay_trace(tmp_path):
    trace = tmp_path / "numeric_metadata.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "numeric_metadata_after_attention_replays_live_geometry and (q or present_k)",
         "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0",
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path" and "Attention" in event["args"].get("ops", "")]
    assert len(paths) == 30
    decode = [event for event in paths if event["path"] == "decode"
              and "Shape" in event["ops"] and "Size" in event["ops"]]
    assert [event["cache"] for event in decode] == (["MISS"] + ["HIT"] * 4) * 3


@pytest.mark.parametrize("op", ["Shape", "Size"])
def test_metadata_same_shaped_parameters_use_existing_general_specialization(op):
    model = onnx.load_model_from_string(_model(op, T.FLOAT, shape=[None, None]))
    model.graph.input[0].name = "data"
    model.graph.input[0].type.tensor_type.shape.ClearField("dim")
    model.graph.input[0].type.tensor_type.shape.dim.add().dim_value = 8
    model.graph.input.append(helper.make_tensor_value_info("target", T.INT64, [2]))
    model.graph.node.insert(0, helper.make_node("Reshape", ["data", "target"], ["x"]))
    session = _session(model.SerializeToString())
    for invocation, shape in enumerate([(2, 4), (4, 2), (1, 8), (8, 1), (2, 4)]):
        feeds = {"data": np.full(8, invocation, np.float32), "target": np.array(shape, np.int64)}
        np.testing.assert_array_equal(session.run(None, feeds)[0], _expected(op, shape))


@pytest.mark.parametrize("op", ["Reshape", "Expand"])
@pytest.mark.parametrize("eager", [False, True])
def test_static_root_metadata_keeps_unrelated_live_shape_controls_safe(op, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    rank = 2 if op == "Reshape" else 3
    inputs = [helper.make_tensor_value_info(name, T.FLOAT, [1, 1, 8])
              for name in ["q", "k", "v"]]
    inputs.extend(helper.make_tensor_value_info(name, T.FLOAT, [1, 1, 1, 8])
                  for name in ["pk", "pv"])
    inputs.append(helper.make_tensor_value_info("target", T.INT64, [rank]))
    graph = helper.make_graph(
        [helper.make_node(
            "Attention", ["q", "k", "v", "", "pk", "pv"], ["y", "present_k", "present_v"],
            q_num_heads=1, kv_num_heads=1, is_causal=1),
         helper.make_node(op, ["y", "target"], ["shaped"]),
         helper.make_node("Shape", ["q"], ["geometry"])],
        "static_metadata_live_controls", inputs,
        [helper.make_tensor_value_info("shaped", T.FLOAT, [None] * rank),
         helper.make_tensor_value_info("present_k", T.FLOAT, [1, 1, 2, 8]),
         helper.make_tensor_value_info("present_v", T.FLOAT, [1, 1, 2, 8]),
         helper.make_tensor_value_info("geometry", T.INT64, [3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)], ir_version=11)
    session = _session(model.SerializeToString())
    shapes = ([(2, 4), (4, 2), (1, 8), (8, 1), (2, 4)] if op == "Reshape" else
              [(1, 2, 8), (2, 1, 8), (1, 1, 8), (1, 3, 8), (1, 2, 8)])
    for invocation, shape in enumerate(shapes):
        feeds = {name: np.zeros((1, 1, 8), np.float32) for name in ["q", "k"]}
        feeds["v"] = np.full((1, 1, 8), invocation + 1, np.float32)
        feeds.update({name: np.zeros((1, 1, 1, 8), np.float32) for name in ["pk", "pv"]})
        feeds["target"] = np.array(shape, np.int64)
        actual = session.run(None, feeds)
        np.testing.assert_array_equal(actual[0], np.full(shape, (invocation + 1) / 2, np.float32))
        np.testing.assert_array_equal(actual[3], [1, 1, 8])


@pytest.mark.parametrize("op", ["Shape", "Size"])
def test_numeric_metadata_control_flow_same_input_shapes(op):
    branches = []
    for shape in [(2, 4), (4, 2)]:
        initializer = numpy_helper.from_array(np.arange(8, dtype=np.float32).reshape(shape), name="local")
        branches.append(helper.make_graph(
            [helper.make_node("Identity", ["local"], ["result"])], "local_geometry", [],
            [helper.make_tensor_value_info("result", T.FLOAT, list(shape))],
            initializer=[initializer],
        ))
    graph = helper.make_graph(
        [helper.make_node("If", ["condition"], ["x"], then_branch=branches[0], else_branch=branches[1]),
         helper.make_node(op, ["x"], ["y"])],
        "metadata_control_flow", [helper.make_tensor_value_info("condition", T.BOOL, [])],
        [helper.make_tensor_value_info("y", T.INT64, [2] if op == "Shape" else [])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)], ir_version=11)
    session = _session(model.SerializeToString())
    for condition in [True, False, False, True]:
        actual = session.run(None, {"condition": np.array(condition)})[0]
        np.testing.assert_array_equal(actual, _expected(op, (2, 4) if condition else (4, 2)))


def test_type_only_capture_does_not_enable_string_control_flow_outputs():
    data = np.array(["a", "b"], object)
    branch = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["result"])], "return_string", [],
        [helper.make_tensor_value_info("result", T.STRING, [2])])
    graph = helper.make_graph(
        [helper.make_node("If", ["condition"], ["y"],
                          then_branch=branch, else_branch=branch)],
        "string_payload_boundary", [helper.make_tensor_value_info("condition", T.BOOL, [])],
        [helper.make_tensor_value_info("y", T.STRING, [2])],
        initializer=[numpy_helper.from_array(data, name="x")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)],
                              ir_version=11).SerializeToString()
    with pytest.raises(Exception, match="fallback to CPU EP has been explicitly disabled"):
        _session(model)
    np.testing.assert_array_equal(
        _session(model, mixed=True).run(None, {"condition": np.array(True)})[0], data)
