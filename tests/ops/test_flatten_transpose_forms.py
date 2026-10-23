"""Exact view semantics, full-width attributes and checked folded geometry."""

import json
import math
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


OPS = ["Flatten", "Transpose"]


def _model(op, shape, parameter=None, *, dtype=np.float32, opset=26):
    tensor_type = numpy_helper.from_array(np.array(0, dtype)).data_type
    node = helper.make_node(op, ["x"], ["y"])
    if parameter is not None:
        name = "axis" if op == "Flatten" else "perm"
        node.attribute.append(helper.make_attribute(
            name, parameter, attr_type=onnx.AttributeProto.INT if op == "Flatten"
            else onnx.AttributeProto.INTS))
    return helper.make_model(
        helper.make_graph(
            [node], "flatten_transpose_forms",
            [helper.make_tensor_value_info("x", tensor_type, shape)],
            [helper.make_tensor_value_info("y", tensor_type, None)]),
        opset_imports=[helper.make_opsetid("", opset)], ir_version=11)


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


def _expected(op, data, parameter):
    if op == "Transpose":
        return data.transpose(parameter)
    axis = 1 if parameter is None else parameter
    if axis < 0:
        axis += data.ndim
    return data.reshape(math.prod(data.shape[:axis]), math.prod(data.shape[axis:]))


def _exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.reshape(-1).view(np.uint8),
                                  expected.reshape(-1).view(np.uint8))


@pytest.mark.parametrize("op,opset", [
    *[("Flatten", version) for version in [1, 9, 11, 13, 21, 23, 24, 25, 26]],
    *[("Transpose", version) for version in [1, 13, 21, 23, 24, 25, 26]],
])
@pytest.mark.parametrize("eager", [False, True])
def test_view_schema_versions(op, opset, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    parameter = 2 if op == "Flatten" else [1, 2, 0]
    model = _model(op, data.shape, parameter, opset=opset)
    expected = _expected(op, data, parameter)
    _exact(_session(model).run(None, {"x": data})[0], expected)
    _exact(_session(model, cpu=True).run(None, {"x": data})[0], expected)
    _exact(ReferenceEvaluator(model).run(None, {"x": data})[0], expected)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("eager", [False, True])
def test_view_default_attributes(op, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    model = _model(op, data.shape)
    expected = _expected(op, data, None)
    _exact(_session(model).run(None, {"x": data})[0], expected)
    _exact(_session(model, cpu=True).run(None, {"x": data})[0], expected)
    _exact(ReferenceEvaluator(model).run(None, {"x": data})[0], expected)


@pytest.mark.parametrize("axis", [-3, -1, 3])
@pytest.mark.parametrize("opset", [11, 13, 21, 23, 24, 25, 26])
def test_flatten_signed_axis_versions(axis, opset):
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    model = _model("Flatten", data.shape, axis, opset=opset)
    expected = _expected("Flatten", data, axis)
    _exact(_session(model).run(None, {"x": data})[0], expected)
    _exact(_session(model, cpu=True).run(None, {"x": data})[0], expected)
    _exact(ReferenceEvaluator(model).run(None, {"x": data})[0], expected)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", [
    np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64, np.bool_,
])
@pytest.mark.parametrize("eager", [False, True])
def test_views_exact_payloads(op, dtype, eager, monkeypatch):
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
    parameter = 2 if op == "Flatten" else [2, 0, 3, 1]
    model = _model(op, [None] * data.ndim, parameter, dtype=dtype)
    _exact(_session(model).run(None, {"x": data})[0], _expected(op, data, parameter))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("eager", [False, True])
def test_views_bfloat16(op, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.array([1.5, -2, 3.25, -4.5, 0, -0.0], np.float32).reshape(1, 2, 3, 1)
    parameter = 2 if op == "Flatten" else [2, 0, 3, 1]
    model = _model(op, data.shape, parameter)
    model.graph.node[0].input[0] = "narrow"
    model.graph.node[0].output[0] = "viewed"
    model.graph.node.insert(0, helper.make_node("Cast", ["x"], ["narrow"], to=T.BFLOAT16))
    model.graph.node.append(helper.make_node("Cast", ["viewed"], ["y"], to=T.FLOAT))
    _exact(_session(model).run(None, {"x": data})[0], _expected(op, data, parameter))


@pytest.mark.parametrize("shape", [[], [0], [0, 3], [2, 0, 3], [65536, 65536, 0]])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("eager", [False, True])
def test_flatten_scalar_and_zero_products(shape, axis, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    if not shape and axis != 0:
        with pytest.raises(Exception):
            _session(_model("Flatten", shape, axis))
        return
    data = np.empty(shape, np.int8) if 0 in shape else np.array(7, np.int8)
    model = _model("Flatten", shape, axis, dtype=np.int8)
    expected = _expected("Flatten", data, axis)
    if any(dim > 2**31 - 1 for dim in expected.shape):
        with pytest.raises(Exception, match="fallback|explicitly assigned|not supported"):
            _session(model)
    else:
        _exact(_session(model).run(None, {"x": data})[0], expected)
    _exact(_session(model, cpu=True).run(None, {"x": data})[0], expected)
    # Python's reference uses reshape(outer, -1), which cannot infer from outer=0.
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    assert [dim.dim_value for dim in inferred.graph.output[0].type.tensor_type.shape.dim] == list(expected.shape)


@pytest.mark.parametrize("shape,perm", [([], None), ([], []), ([0], None),
                                      ([0, 2, 3], None), ([2, 0, 3], [2, 0, 1])])
@pytest.mark.parametrize("eager", [False, True])
def test_transpose_scalar_empty_and_default(shape, perm, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.empty(shape, np.float32) if 0 in shape else np.array(7, np.float32)
    model = _model("Transpose", shape, perm)
    expected = _expected("Transpose", data, perm)
    _exact(_session(model).run(None, {"x": data})[0], expected)
    _exact(_session(model, cpu=True).run(None, {"x": data})[0], expected)
    _exact(ReferenceEvaluator(model).run(None, {"x": data})[0], expected)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("eager", [False, True])
def test_view_geometry_route_replay(op, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    parameter = 1 if op == "Flatten" else None
    session = _session(_model(op, [None, None, 2], parameter))
    for invocation, shape in enumerate([[2, 3, 2], [2, 3, 2], [3, 2, 2], [3, 2, 2], [2, 3, 2]]):
        data = (np.arange(12, dtype=np.float32) + invocation * 10).reshape(shape)
        _exact(session.run(None, {"x": data})[0], _expected(op, data, parameter))


@pytest.mark.parametrize("op", OPS)
def test_view_actual_routes(tmp_path, op):
    trace = tmp_path / "view_trace.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", f"view_geometry_route_replay and {op}", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120)
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path" and op in event["args"].get("ops", "").split(",")]
    assert [event["cache"] for event in paths if event["path"] == "general"] == [
        "MISS", "HIT", "RETRACE", "HIT", "HIT"]
    assert sum(event["path"] == "eager" for event in paths) == 5


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("eager", [False, True])
def test_views_unknown_rank_replay(op, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    parameter = 0 if op == "Flatten" else None
    session = _session(_model(op, None, parameter))
    for invocation, shape in enumerate([[], [2, 3], [2, 1, 3], [0, 3], [3], []]):
        data = (np.arange(math.prod(shape), dtype=np.float32) + invocation * 10).reshape(shape)
        _exact(session.run(None, {"x": data})[0], _expected(op, data, parameter))


@pytest.mark.parametrize("op,parameter", [
    ("Flatten", 2**32), ("Flatten", -(2**63)), ("Flatten", 2**63 - 1),
    ("Transpose", [-1, 0]), ("Transpose", [0, 0]), ("Transpose", [2**32, 1]),
    ("Transpose", [-(2**63), 1]), ("Transpose", [2**63 - 1, 1]),
])
def test_views_invalid_attributes_declined(op, parameter):
    with pytest.raises(Exception, match="fallback|explicitly assigned|not supported"):
        _session(_model(op, None, parameter))


@pytest.mark.parametrize("op,parameter,bad_shape", [
    ("Flatten", 2, [6]), ("Transpose", [1, 0], [6]),
])
def test_views_runtime_rank_errors_recover(op, parameter, bad_shape):
    session = _session(_model(op, None, parameter))
    good = np.arange(6, dtype=np.float32).reshape(2, 3)
    _exact(session.run(None, {"x": good})[0], _expected(op, good, parameter))
    with pytest.raises(Exception, match=f"MLX.*{op}"):
        session.run(None, {"x": good.reshape(bad_shape)})
    good += 10
    _exact(session.run(None, {"x": good})[0], _expected(op, good, parameter))


@pytest.mark.parametrize("op,shape,parameter", [
    ("Flatten", [0, 65536, 65536], 1),
    ("Flatten", [65536, 65536, None], 2),
    ("Flatten", [0, 2**32], 1),
    ("Transpose", [0, 2**32], None),
])
def test_views_native_wide_geometry_declined(op, shape, parameter):
    with pytest.raises(Exception, match="fallback|explicitly assigned|not supported"):
        _session(_model(op, shape, parameter))


def test_flatten_runtime_folded_overflow_recovers():
    session = _session(_model("Flatten", None, 1, dtype=np.int8))
    good = np.arange(6, dtype=np.int8).reshape(2, 3)
    _exact(session.run(None, {"x": good})[0], good)
    with pytest.raises(Exception, match="MLX.*Flatten.*folded dimension"):
        session.run(None, {"x": np.empty((0, 65536, 65536), np.int8)})
    good += 10
    _exact(session.run(None, {"x": good})[0], good)
    cpu = _session(_model("Flatten", [0, 65536, 65536], 1, dtype=np.int8), cpu=True)
    _exact(cpu.run(None, {"x": np.empty((0, 65536, 65536), np.int8)})[0],
           np.empty((0, 2**32), np.int8))


@pytest.mark.parametrize("axis", [0, 1, 2, 3, -1, -3])
@pytest.mark.parametrize("eager", [False, True])
def test_flatten_transposed_strides(axis, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.arange(24, dtype=np.int64).reshape(2, 3, 4) + 2**53
    model = _model("Flatten", data.shape, axis, dtype=np.int64)
    model.graph.node[0].input[0] = "strided"
    model.graph.node.insert(0, helper.make_node("Transpose", ["x"], ["strided"], perm=[2, 0, 1]))
    expected = _expected("Flatten", data.transpose(2, 0, 1), axis)
    _exact(_session(model).run(None, {"x": data})[0], expected)
    _exact(_session(model, cpu=True).run(None, {"x": data})[0], expected)
