"""Constant Tile controls, exact payloads and checked empty geometry."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort
from onnx import TensorProto as T, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
import pytest


def _model(shape, repeats, *, dtype=np.float32, opset=26):
    tensor_type = numpy_helper.from_array(np.array(0, dtype)).data_type
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Tile", ["x", "repeats"], ["y"])], "tile_forms",
            [helper.make_tensor_value_info("x", tensor_type, shape)],
            [helper.make_tensor_value_info("y", tensor_type, None)],
            initializer=[numpy_helper.from_array(np.asarray(repeats, np.int64), "repeats")]),
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


def _exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.reshape(-1).view(np.uint8),
                                  expected.reshape(-1).view(np.uint8))


@pytest.mark.parametrize("opset", [6, 13, 26])
@pytest.mark.parametrize("eager", [False, True])
def test_tile_constant_versions(opset, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    repeats = [2, 3]
    model = _model(data.shape, repeats, opset=opset)
    expected = np.tile(data, repeats)
    _exact(_session(model).run(None, {"x": data})[0], expected)
    _exact(_session(model, cpu=True).run(None, {"x": data})[0], expected)
    _exact(ReferenceEvaluator(model).run(None, {"x": data})[0], expected)


@pytest.mark.parametrize("dtype", [
    np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64, np.bool_,
])
@pytest.mark.parametrize("eager", [False, True])
def test_tile_exact_payloads(dtype, eager, monkeypatch):
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
    data = values.reshape(2, 3)
    model = _model([None, None], [2, 3], dtype=dtype)
    _exact(_session(model).run(None, {"x": data})[0], np.tile(data, [2, 3]))


@pytest.mark.parametrize("eager", [False, True])
def test_tile_bfloat16(eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.array([1.5, -2, 3.25, -4.5, 0, -0.0], np.float32).reshape(2, 3)
    model = _model(data.shape, [2, 3])
    model.graph.node[0].input[0] = "narrow"
    model.graph.node[0].output[0] = "tiled"
    model.graph.node.insert(0, helper.make_node("Cast", ["x"], ["narrow"], to=T.BFLOAT16))
    model.graph.node.append(helper.make_node("Cast", ["tiled"], ["y"], to=T.FLOAT))
    _exact(_session(model).run(None, {"x": data})[0], np.tile(data, [2, 3]))


@pytest.mark.parametrize("shape,repeats", [
    ([], []), ([0], [2**63 - 1]), ([0, 2], [2**63 - 1, 3]),
    ([2, 3], [0, 4]), ([0, 2**32], [1, 0]), ([2**32, 0], [0, 1]),
])
@pytest.mark.parametrize("eager", [False, True])
def test_tile_empty_full_width_geometry(shape, repeats, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.zeros(shape, np.int8)
    model = _model(shape, repeats, dtype=np.int8)
    expected = np.zeros([dim * repeat for dim, repeat in zip(shape, repeats)], np.int8)
    _exact(_session(model).run(None, {"x": data})[0], expected)
    _exact(_session(model, cpu=True).run(None, {"x": data})[0], expected)


@pytest.mark.parametrize("repeats", [[-1], [-(2**32)], [-(2**63)]])
def test_tile_negative_repeats_declined(repeats):
    with pytest.raises(Exception, match="fallback|explicitly assigned|not supported"):
        _session(_model(None, repeats))


@pytest.mark.parametrize("shape,repeats", [([1], [2**32 + 1]), ([0, 1], [1, 2**32])])
def test_tile_native_wide_output_declined(shape, repeats):
    with pytest.raises(Exception, match="fallback|explicitly assigned|not supported"):
        _session(_model(shape, repeats))


@pytest.mark.parametrize("eager", [False, True])
def test_tile_unknown_rank_cardinality_recovery(eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(None, [2, 1]))
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    _exact(session.run(None, {"x": data})[0], np.tile(data, [2, 1]))
    with pytest.raises(Exception, match="MLX.*Tile.*repeats length"):
        session.run(None, {"x": data.reshape(6)})
    data += 10
    _exact(session.run(None, {"x": data})[0], np.tile(data, [2, 1]))


def test_tile_unknown_rank_full_width_overflow_recovery():
    session = _session(_model(None, [2**32 + 1]))
    data = np.empty(0, np.float32)
    _exact(session.run(None, {"x": data})[0], data)
    with pytest.raises(Exception, match="MLX.*Tile.*output dimension"):
        session.run(None, {"x": np.arange(2, dtype=np.float32)})
    _exact(session.run(None, {"x": data})[0], data)


@pytest.mark.parametrize("shape", [[], [2, 1], [2, 1, 3], [0, 3], [2, 3]])
@pytest.mark.parametrize("eager", [False, True])
def test_tile_ones_identity(shape, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    model = _model(shape, [1] * len(shape))
    _exact(_session(model).run(None, {"x": data})[0], data)


@pytest.mark.parametrize("eager", [False, True])
def test_tile_constant_route_replay(eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model([None, None], [2, 3]))
    for invocation, shape in enumerate([[2, 3], [2, 3], [3, 2], [3, 2], [2, 3]]):
        data = (np.arange(6, dtype=np.float32) + invocation * 10).reshape(shape)
        _exact(session.run(None, {"x": data})[0], np.tile(data, [2, 3]))


def test_tile_constant_actual_routes(tmp_path):
    trace = tmp_path / "tile_trace.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "tile_constant_route_replay", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120)
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path" and "Tile" in event["args"].get("ops", "").split(",")]
    assert [event["cache"] for event in paths if event["path"] == "general"] == [
        "MISS", "HIT", "RETRACE", "HIT", "HIT"]
    assert sum(event["path"] == "eager" for event in paths) == 5
