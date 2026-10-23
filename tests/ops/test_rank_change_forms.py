"""Squeeze/Unsqueeze versions, exact views, full-width axes and guarded replay."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
import pytest


OPS = ["Squeeze", "Unsqueeze"]


def _model(op, shape, axes, *, dtype=np.float32, opset=26, live=False,
           omitted_slot=False, interior=False):
    tensor_type = numpy_helper.from_array(np.array(0, dtype)).data_type
    inputs = [helper.make_tensor_value_info("x", tensor_type, shape)]
    names = ["x"]
    nodes = []
    initializers = []
    node = helper.make_node(op, names, ["y"])
    if opset < 13:
        if axes is not None:
            node.attribute.append(helper.make_attribute(
                "axes", axes, attr_type=onnx.AttributeProto.INTS))
    elif axes is not None:
        if live or interior:
            inputs.append(helper.make_tensor_value_info("axes", T.INT64, [None]))
        else:
            initializers.append(numpy_helper.from_array(np.array(axes, np.int64), "axes"))
        if interior:
            nodes.append(helper.make_node("Identity", ["axes"], ["interior_axes"]))
        node.input.append("interior_axes" if interior else "axes")
    elif omitted_slot:
        node.input.append("")
    nodes.append(node)
    return helper.make_model(
        helper.make_graph(
            nodes, "rank_change_forms", inputs,
            [helper.make_tensor_value_info("y", tensor_type, None)],
            initializer=initializers,
        ), opset_imports=[helper.make_opsetid("", opset)], ir_version=11,
    )


def _session(model, *, cpu=False):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    if not cpu:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model.SerializeToString(), options,
        providers=["CPUExecutionProvider"] if cpu else ["MLXExecutionProvider"])
    session.disable_fallback()
    return session


def _exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.reshape(-1).view(np.uint8),
                                  expected.reshape(-1).view(np.uint8))


def _expected(op, data, axes):
    if op == "Unsqueeze":
        return np.expand_dims(data, tuple(axes))
    if axes is None:
        return np.squeeze(data)
    normalized = tuple(sorted({axis % data.ndim for axis in axes}))
    return np.squeeze(data, axis=normalized)


@pytest.mark.parametrize("case", ["squeeze_live", "unsqueeze_live", "omitted_slot", "duplicates"])
def test_rank_changes_strict_gaps(case):
    op = "Unsqueeze" if case == "unsqueeze_live" else "Squeeze"
    axes = None if case == "omitted_slot" else [0, 0] if case == "duplicates" else [0]
    live = case.endswith("live")
    model = _model(op, [1, 2, 1], axes, live=live, omitted_slot=True)
    data = np.arange(2, dtype=np.float32).reshape(1, 2, 1)
    feeds = {"x": data}
    if live:
        feeds["axes"] = np.array(axes, np.int64)
    _exact(_session(model).run(None, feeds)[0], _expected(op, data, axes))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("opset", [1, 11, 13, 21, 23, 24, 25, 26])
@pytest.mark.parametrize("eager", [False, True])
def test_rank_changes_schema_versions(op, opset, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.arange(6, dtype=np.float32).reshape(1, 2, 3, 1)
    axes = [3, 0] if op == "Squeeze" else [5, 0]
    model = _model(op, data.shape, axes, opset=opset)
    expected = _expected(op, data, axes)
    _exact(_session(model).run(None, {"x": data})[0], expected)
    _exact(_session(model, cpu=True).run(None, {"x": data})[0], expected)
    # Legacy Python evaluators apply axes sequentially instead of simultaneously.
    reference_model = _model(op, data.shape, sorted(axes), opset=opset) if opset < 13 else model
    _exact(ReferenceEvaluator(reference_model).run(None, {"x": data})[0], expected)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", [
    np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64, np.bool_,
])
@pytest.mark.parametrize("eager", [False, True])
def test_rank_changes_exact_dtypes(op, dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    values = np.arange(6).astype(dtype)
    if dtype == np.int64:
        values = np.array([-(2**63), 2**63 - 1, 2**53 + 1, -2**53 - 1, 0, 7], dtype)
    elif dtype == np.uint64:
        values = np.array([0, 2**64 - 1, 2**63 + 1, 2**53 + 1, 7, 9], dtype)
    elif dtype == np.float64:
        values = np.array([0, 0x8000000000000000, 0x7FF8000000001234,
                           0x3FF0000000000001, 0xFFF0000000000000,
                           0x7FF0000000000000], np.uint64).view(np.float64)
    elif dtype == np.float32:
        values = np.array([0, 0x80000000, 0x7FC01234, 0x3F800001,
                           0xFF800000, 0x7F800000], np.uint32).view(np.float32)
    elif dtype == np.float16:
        values = np.array([0, 0x8000, 0x7E01, 0x3C01, 0xFC00, 0x7C00],
                          np.uint16).view(np.float16)
    data = values.reshape(1, 2, 3, 1)
    axes = [-1, 0]
    model = _model(op, [None] * data.ndim, axes, dtype=dtype, live=True)
    _exact(_session(model).run(None, {"x": data, "axes": np.array(axes, np.int64)})[0],
           _expected(op, data, axes))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("eager", [False, True])
def test_rank_changes_bfloat16(op, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.array([1.5, -2, 3.25, -4.5, 0, -0.0], np.float32).reshape(1, 2, 3, 1)
    model = _model(op, data.shape, [-1, 0], live=True)
    model.graph.node[0].input[0] = "narrow"
    model.graph.node[0].output[0] = "viewed"
    model.graph.node.insert(0, helper.make_node("Cast", ["x"], ["narrow"], to=T.BFLOAT16))
    model.graph.node.append(helper.make_node("Cast", ["viewed"], ["y"], to=T.FLOAT))
    _exact(_session(model).run(None, {"x": data, "axes": np.array([-1, 0], np.int64)})[0],
           _expected(op, data, [-1, 0]))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("shape", [[], [1], [1, 1], [0], [0, 1, 2, 1], [1, 0, 1]])
@pytest.mark.parametrize("eager", [False, True])
def test_rank_changes_scalar_and_empty(op, shape, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.arange(np.prod(shape, dtype=int), dtype=np.int64).reshape(shape)
    axes = None if op == "Squeeze" else [-1, 0]
    model = _model(op, shape, axes, dtype=np.int64, live=axes is not None)
    feeds = {"x": data}
    if axes is not None:
        feeds["axes"] = np.array(axes, np.int64)
    _exact(_session(model).run(None, feeds)[0], _expected(op, data, axes))
    _exact(_session(model, cpu=True).run(None, feeds)[0], _expected(op, data, axes))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("eager", [False, True])
def test_rank_changes_explicit_empty_axes_are_identity(op, live, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _model(op, [1, 2, 1], [], live=live)
    data = np.arange(2, dtype=np.float32).reshape(1, 2, 1)
    feeds = {"x": data}
    if live:
        feeds["axes"] = np.array([], np.int64)
    _exact(_session(model).run(None, feeds)[0], data)
    _exact(ReferenceEvaluator(model).run(None, feeds)[0], data)
    inferred = onnx.shape_inference.infer_shapes(_model(op, data.shape, []), strict_mode=True)
    assert [dim.dim_value for dim in inferred.graph.output[0].type.tensor_type.shape.dim] == [1, 2, 1]
    # ORT 1.31 treats empty Squeeze axes as omitted; the ONNX inference/reference use identity.
    cpu_expected = np.squeeze(data) if op == "Squeeze" else data
    _exact(_session(model, cpu=True).run(None, feeds)[0], cpu_expected)


@pytest.mark.parametrize("opset", [1, 11])
def test_squeeze_legacy_empty_axes_conflict_declined(opset):
    model = _model("Squeeze", [1, 2, 1], [], opset=opset)
    with pytest.raises(Exception, match="fallback|explicitly assigned|not supported"):
        _session(model)
    data = np.arange(2, dtype=np.float32).reshape(1, 2, 1)
    _exact(_session(model, cpu=True).run(None, {"x": data})[0], np.squeeze(data))
    _exact(ReferenceEvaluator(model).run(None, {"x": data})[0], np.squeeze(data))


@pytest.mark.parametrize("opset", [1, 11])
@pytest.mark.parametrize("shape", [[], [2, 3], [0, 2]])
def test_squeeze_legacy_empty_axes_unambiguous_identity(opset, shape):
    data = np.arange(np.prod(shape, dtype=int), dtype=np.float32).reshape(shape)
    model = _model("Squeeze", shape, [], opset=opset)
    _exact(_session(model).run(None, {"x": data})[0], data)
    _exact(_session(model, cpu=True).run(None, {"x": data})[0], data)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("eager", [False, True])
def test_rank_changes_legacy_negative_axes(op, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.arange(6, dtype=np.float32).reshape(1, 2, 3, 1)
    model = _model(op, data.shape, [-1, 0], opset=11)
    expected = _expected(op, data, [-1, 0])
    _exact(_session(model).run(None, {"x": data})[0], expected)
    _exact(_session(model, cpu=True).run(None, {"x": data})[0], expected)


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("eager", [False, True])
def test_squeeze_repeated_normalized_selections(live, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    axes = [0, -4, 3, -1]
    data = np.arange(6, dtype=np.float32).reshape(1, 2, 3, 1)
    model = _model("Squeeze", data.shape, axes, live=live)
    feeds = {"x": data}
    if live:
        feeds["axes"] = np.array(axes, np.int64)
    _exact(_session(model).run(None, feeds)[0], data.reshape(2, 3))
    _exact(_session(model, cpu=True).run(None, feeds)[0], data.reshape(2, 3))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("eager", [False, True])
def test_rank_change_route_replay(op, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(op, [1, 2, 1], [0], live=True))
    for invocation, axes in enumerate([[0], [0], [-1], [-1], [0]]):
        data = (np.arange(2, dtype=np.float32) + invocation * 10).reshape(1, 2, 1)
        _exact(session.run(None, {"x": data, "axes": np.array(axes, np.int64)})[0],
               _expected(op, data, axes))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("eager", [False, True])
def test_rank_change_interior_route(op, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(op, [1, 2, 1], [0], interior=True))
    for invocation, axes in enumerate([[0], [-1], [0]]):
        data = (np.arange(2, dtype=np.float32) + invocation * 10).reshape(1, 2, 1)
        _exact(session.run(None, {"x": data, "axes": np.array(axes, np.int64)})[0],
               _expected(op, data, axes))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("eager", [False, True])
def test_rank_change_empty_route(op, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(op, [1, 2, 1], [], live=True))
    for invocation in range(3):
        data = (np.arange(2, dtype=np.float32) + invocation * 10).reshape(1, 2, 1)
        _exact(session.run(None, {"x": data, "axes": np.array([], np.int64)})[0], data)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("eager", [False, True])
def test_rank_change_shape_derived_route(op, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _model(op, [2, 1, 3, 1], [1], interior=True)
    model.graph.input[1].CopyFrom(helper.make_tensor_value_info("axis_source", T.FLOAT, [None]))
    model.graph.node[0].op_type = "Shape"
    model.graph.node[0].input[0] = "axis_source"
    session = _session(model)
    for invocation, axis in enumerate([1, 1, 3, 3, 1]):
        data = (np.arange(6, dtype=np.float32) + invocation * 10).reshape(2, 1, 3, 1)
        _exact(session.run(None, {"x": data, "axis_source": np.ones(axis, np.float32)})[0],
               _expected(op, data, [axis]))


@pytest.mark.parametrize("op", OPS)
def test_rank_change_actual_routes(tmp_path, op):
    trace = tmp_path / "rank_change_trace.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", f"(rank_change_route_replay or rank_change_interior_route or rank_change_empty_route or rank_change_shape_derived_route) and {op}",
         "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path" and op in event["args"].get("ops", "").split(",")]
    assert [event["cache"] for event in paths if event["path"] == "general"] == [
        "MISS", "HIT", "RETRACE", "HIT", "HIT"] * 2
    assert sum(event["path"] == "eager" for event in paths) == 22


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("eager", [False, True])
def test_rank_changes_unknown_rank_replay(op, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(op, None, [-1], live=True))
    for invocation, shape in enumerate([[1, 2, 1], [2, 3, 1], [1], [0, 1], [1, 2, 1]]):
        data = (np.arange(np.prod(shape), dtype=np.float32) + invocation * 10).reshape(shape)
        _exact(session.run(None, {"x": data, "axes": np.array([-1], np.int64)})[0],
               _expected(op, data, [-1]))


@pytest.mark.parametrize("eager", [False, True])
def test_squeeze_no_axes_changing_singletons(eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model("Squeeze", [None, None, None], None, omitted_slot=True))
    for invocation, shape in enumerate([[1, 2, 1], [2, 1, 3], [1, 1, 1], [0, 1, 3], [1, 2, 1]]):
        data = (np.arange(np.prod(shape), dtype=np.float32) + invocation * 10).reshape(shape)
        _exact(session.run(None, {"x": data})[0], np.squeeze(data))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("bad", [[2**63 - 1], [-(2**63)], [2**32], [-5], [5]])
def test_rank_changes_invalid_full_width_axes_recover(op, bad):
    session = _session(_model(op, [1, 2, 1], [0], live=True))
    good = {"x": np.arange(2, dtype=np.float32).reshape(1, 2, 1),
            "axes": np.array([0], np.int64)}
    _exact(session.run(None, good)[0], _expected(op, good["x"], [0]))
    with pytest.raises(Exception, match="MLX.*(Squeeze|Unsqueeze)"):
        session.run(None, {**good, "axes": np.array(bad, np.int64)})
    good["x"] += 10
    _exact(session.run(None, good)[0], _expected(op, good["x"], [0]))


@pytest.mark.parametrize("op,bad", [("Squeeze", [1]), ("Unsqueeze", [0, -5])])
def test_rank_changes_invalid_extent_or_duplicate_recover(op, bad):
    session = _session(_model(op, [1, 2, 1], [0], live=True))
    good = {"x": np.arange(2, dtype=np.float32).reshape(1, 2, 1),
            "axes": np.array([0], np.int64)}
    with pytest.raises(Exception, match="MLX.*(Squeeze|Unsqueeze)"):
        session.run(None, {**good, "axes": np.array(bad, np.int64)})
    _exact(session.run(None, good)[0], _expected(op, good["x"], [0]))


@pytest.mark.parametrize("op", OPS)
def test_rank_changes_unknown_axes_rank_validates_and_recovers(op):
    model = _model(op, [1, 2, 1], [0], live=True)
    model.graph.input[1].type.tensor_type.ClearField("shape")
    session = _session(model)
    good = {"x": np.arange(2, dtype=np.float32).reshape(1, 2, 1),
            "axes": np.array([0], np.int64)}
    _exact(session.run(None, good)[0], _expected(op, good["x"], [0]))
    for bad in [np.array(0, np.int64), np.array([[0]], np.int64)]:
        with pytest.raises(Exception, match="MLX.*(Squeeze|Unsqueeze)"):
            session.run(None, {**good, "axes": bad})
    good["x"] += 10
    _exact(session.run(None, good)[0], _expected(op, good["x"], [0]))


@pytest.mark.parametrize("op", OPS)
def test_rank_changes_native_wide_empty_geometry_declined(op):
    model = _model(op, [0, 2**32], None if op == "Squeeze" else [0])
    with pytest.raises(Exception, match="fallback|explicitly assigned|not supported"):
        _session(model)
