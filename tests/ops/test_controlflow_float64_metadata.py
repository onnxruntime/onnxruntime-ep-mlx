"""Control-flow metadata reads must not import ignored Double payloads into MLX."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper
import pytest

from test_metadata_only_shape import _expected, _model, _session


def _metadata_body(op, *, shift=False, initializer=None, data_op=None):
    body = onnx.load_model_from_string(
        _model(op, T.DOUBLE, initializer=initializer,
               **({"start": 1} if shift and op == "Shape" else {})),
    ).graph
    del body.input[:]
    if shift and op == "Size":
        body.initializer.append(numpy_helper.from_array(np.array(1, np.int64), name="one"))
        body.node[0].output[0] = "size"
        body.node.append(helper.make_node("Add", ["size", "one"], ["y"]))
    if data_op is not None:
        body.node[0].input[0] = "computed"
        body.node.insert(0, helper.make_node(data_op, ["x"], ["computed"]))
    return body


def _if_model(op, *, nested=False, initializer=None, data_op=None):
    branches = [_metadata_body(op, shift=shift, initializer=initializer, data_op=data_op)
                for shift in (False, True)]
    if nested:
        inner = helper.make_node(
            "If", ["inner_condition"], ["y"],
            then_branch=branches[0], else_branch=branches[1],
        )
        branches[0] = helper.make_graph([inner], "nested", [], list(branches[0].output))
    graph = helper.make_graph(
        [helper.make_node("If", ["condition"], ["result"],
                          then_branch=branches[0], else_branch=branches[1])],
        "double_metadata",
        [helper.make_tensor_value_info("condition", T.BOOL, [])]
        + ([helper.make_tensor_value_info("inner_condition", T.BOOL, [])] if nested else [])
        + ([helper.make_tensor_value_info("x", T.DOUBLE, None)] if initializer is None else []),
        [helper.make_tensor_value_info("result", T.INT64, [None] if op == "Shape" else [])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)], ir_version=11)


def _reference(op, shape, shift):
    result = _expected(op, shape, start=int(shift))
    return result + np.int64(1) if shift and op == "Size" else result


def _check(actual, expected):
    assert actual.dtype == expected.dtype == np.int64
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def _profile(session, *, declined=False):
    events = json.loads(Path(session.end_profiling()).read_text())
    compute = [event["args"] for event in events
               if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time")]
    assert compute
    assert all(event.get("provider") for event in compute), compute
    if declined:
        assert any(event.get("op_name") == "If"
                   and event["provider"] == "CPUExecutionProvider" for event in compute)
    else:
        assert all(event["provider"] == "MLXExecutionProvider" for event in compute), compute


@pytest.mark.parametrize("op", ["Shape", "Size"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("eager", [False, True])
def test_if_double_metadata_live_geometry(op, nested, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _if_model(op, nested=nested).SerializeToString()
    session = _session(model, profile=tmp_path / "if")
    for shape in [(), (3,), (2, 3), (1, 0, 3), (0, 2**31 + 7), (2, 3)]:
        for condition, inner in [(True, True), (True, False), (False, True), (True, True)]:
            x = np.full(shape, np.nan if condition else np.inf, np.float64)
            snapshot = x.tobytes()
            feeds = {"x": x, "condition": np.array(condition)}
            if nested:
                feeds["inner_condition"] = np.array(inner)
            actual = session.run(None, feeds)[0]
            _check(actual, _reference(op, shape, not condition or (nested and not inner)))
            assert x.tobytes() == snapshot
    _profile(session)


@pytest.mark.parametrize("op", ["Shape", "Size"])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3), (1, 0, 3)])
@pytest.mark.parametrize("eager", [False, True])
def test_if_double_metadata_branch_initializers(op, shape, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    initializer = numpy_helper.from_array(np.full(shape, np.nan, np.float64), name="x")
    session = _session(_if_model(op, initializer=initializer).SerializeToString(),
                       profile=tmp_path / "initializer")
    for condition in [True, False, True]:
        _check(session.run(None, {"condition": np.array(condition)})[0],
               _reference(op, shape, not condition))
    _profile(session)


def _recurrent_model(flow, op):
    metadata = helper.make_node(op, ["x"], ["metadata"])
    nodes = [metadata]
    initializers = []
    if op == "Shape":
        initializers.append(numpy_helper.from_array(np.array([0], np.int64), name="axes"))
        nodes.append(helper.make_node("ReduceSum", ["metadata", "axes"], ["increment"], keepdims=0))
    else:
        metadata.output[0] = "increment"
    nodes.append(helper.make_node("Add", ["state", "increment"], ["updated"]))
    state = helper.make_tensor_value_info("state", T.INT64, [])
    updated = helper.make_tensor_value_info("updated", T.INT64, [])
    if flow == "Loop":
        body = helper.make_graph(
            nodes, "loop_metadata",
            [helper.make_tensor_value_info("iteration", T.INT64, []),
             helper.make_tensor_value_info("body_condition", T.BOOL, []), state],
            [helper.make_tensor_value_info("body_condition", T.BOOL, []), updated],
            initializer=initializers,
        )
        control = helper.make_node("Loop", ["trip", "condition", "initial"], ["final"], body=body)
        inputs = [helper.make_tensor_value_info("trip", T.INT64, []),
                  helper.make_tensor_value_info("condition", T.BOOL, [])]
        outputs = [helper.make_tensor_value_info("final", T.INT64, [])]
    else:
        nodes.append(helper.make_node("Identity", ["updated"], ["element_out"]))
        body = helper.make_graph(
            nodes, "scan_metadata", [state, helper.make_tensor_value_info("element", T.FLOAT, [])],
            [updated, helper.make_tensor_value_info("element_out", T.INT64, [])],
            initializer=initializers,
        )
        control = helper.make_node(
            "Scan", ["initial", "sequence"], ["final", "scanned"], body=body, num_scan_inputs=1,
        )
        inputs = [helper.make_tensor_value_info("sequence", T.FLOAT, ["trip"])]
        outputs = [helper.make_tensor_value_info("final", T.INT64, []),
                   helper.make_tensor_value_info("scanned", T.INT64, ["trip"])]
    graph = helper.make_graph(
        [control], "recurrent_metadata",
        inputs + [helper.make_tensor_value_info("initial", T.INT64, []),
                  helper.make_tensor_value_info("x", T.DOUBLE, None)],
        outputs,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)],
                             ir_version=11).SerializeToString()


@pytest.mark.parametrize("flow", ["Loop", "Scan"])
@pytest.mark.parametrize("op", ["Shape", "Size"])
@pytest.mark.parametrize("eager", [False, True])
def test_recurrent_double_metadata_capture(flow, op, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_recurrent_model(flow, op), profile=tmp_path / flow)
    for shape in [(2, 3), (0, 3), (3, 2)]:
        x = np.full(shape, np.inf, np.float64)
        snapshot = x.tobytes()
        increment = sum(shape) if op == "Shape" else int(np.prod(shape))
        for trip in [0, 1, 3, 1]:
            initial = np.array(5 - trip, np.int64)
            feeds = {"x": x, "initial": initial}
            if flow == "Loop":
                feeds.update(trip=np.array(trip, np.int64), condition=np.array(True))
            else:
                feeds["sequence"] = np.arange(trip, dtype=np.float32)
            actual = session.run(None, feeds)
            _check(actual[0], initial + trip * np.int64(increment))
            if flow == "Scan":
                _check(actual[1], initial + np.arange(1, trip + 1, dtype=np.int64) * increment)
            assert x.tobytes() == snapshot
    _profile(session)


@pytest.mark.parametrize("data_op", ["Neg", "Identity"])
@pytest.mark.parametrize("op", ["Shape", "Size"])
@pytest.mark.parametrize("eager", [False, True])
def test_double_payload_computation_inside_if_stays_declined(
    data_op, op, eager, monkeypatch, tmp_path,
):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _if_model(op, data_op=data_op).SerializeToString()
    session = _session(model, mixed=True, profile=tmp_path / data_op)
    for condition in [True, False]:
        x = np.array([[np.nan, np.inf, -0.0], [1, 2, 3]], np.float64)
        _check(session.run(None, {"x": x, "condition": np.array(condition)})[0],
               _reference(op, x.shape, not condition))
    _profile(session, declined=True)


def _route_replay(profile):
    model = _if_model("Shape", nested=True)
    graph = model.graph
    graph.initializer.append(numpy_helper.from_array(np.array(1, np.float32), name="one_float"))
    graph.input.extend([helper.make_tensor_value_info("values", T.FLOAT, [3]),
                        helper.make_tensor_value_info("factor", T.FLOAT, [3])])
    graph.node.insert(0, helper.make_node("Add", ["values", "one_float"], ["upstream"]))
    control = graph.node[1]
    control.output.append("passthrough")
    for attribute in control.attribute:
        branch = attribute.g
        if branch.node[0].op_type == "If":
            branch.node[0].output.append("nested_passthrough")
            branch.output.append(helper.make_tensor_value_info("nested_passthrough", T.FLOAT, [3]))
            for inner_attribute in branch.node[0].attribute:
                inner_attribute.g.node.append(
                    helper.make_node("Identity", ["upstream"], ["local_passthrough"]),
                )
                inner_attribute.g.output.append(
                    helper.make_tensor_value_info("local_passthrough", T.FLOAT, [3]),
                )
        else:
            branch.node.append(helper.make_node("Identity", ["upstream"], ["local_passthrough"]))
            branch.output.append(helper.make_tensor_value_info("local_passthrough", T.FLOAT, [3]))
    graph.node.append(helper.make_node("Mul", ["passthrough", "factor"], ["scaled"]))
    graph.output.append(helper.make_tensor_value_info("scaled", T.FLOAT, [3]))
    session = _session(model.SerializeToString(), profile=profile)
    for index, (condition, inner) in enumerate([(True, True), (True, False), (False, True)]):
        values = np.array([1, 2, 3], np.float32) + index
        factor = np.full(3, index + 2, np.float32)
        result, scaled = session.run(None, {
            "x": np.full((2, 3), np.nan, np.float64), "condition": np.array(condition),
            "inner_condition": np.array(inner), "values": values, "factor": factor,
        })
        _check(result, _reference("Shape", (2, 3), not condition or not inner))
        np.testing.assert_array_equal(scaled, (values + np.float32(1)) * factor)
    _profile(session)


def test_metadata_control_flow_eager_boundary_preserves_general_neighbors(tmp_path):
    trace = tmp_path / "metadata-route.json"
    subprocess.run(
        [sys.executable, "-c",
         "import os, onnxruntime as ort; "
         "ort.register_execution_provider_library('MLXExecutionProvider', os.environ['ONNXRUNTIME_MLX_EP_LIB']); "
         "from test_controlflow_float64_metadata import _route_replay; _route_replay(os.environ['MLX_METADATA_ROUTE_PROFILE'])"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "MLX_METADATA_ROUTE_PROFILE": str(tmp_path / "metadata-route-profile"),
             "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0", "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    events = [event["args"] for event in json.loads(trace.read_text()) if event.get("cat") == "ep.path"]
    control_paths = [event["path"] for event in events if "If" in event["ops"].split(",")]
    assert control_paths == ["eager"] * 3, control_paths
    for neighbor in ["Add", "Mul"]:
        paths = [event for event in events if neighbor in event["ops"].split(",")]
        assert [event["path"] for event in paths] == ["general"] * 3, paths
        assert paths[1]["cache"] == "HIT", paths
