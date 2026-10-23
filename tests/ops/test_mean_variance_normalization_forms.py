"""MeanVarianceNormalization function semantics, live geometry and typed arithmetic."""

import json
import os
from pathlib import Path
import subprocess
import sys

import ml_dtypes
import numpy as np
from onnx import AttributeProto, TensorProto as T, helper
from onnx.reference import ReferenceEvaluator
import pytest

from test_metadata_only_shape import _session


TYPES = {T.FLOAT16: np.float16, T.FLOAT: np.float32, T.DOUBLE: np.float64,
         T.BFLOAT16: ml_dtypes.bfloat16}


def _model(dtype=T.FLOAT, shape=None, *, axes=None, opset=13):
    boundary_type = T.FLOAT if dtype == T.BFLOAT16 else dtype
    nodes = []
    name = "x"
    if dtype == T.BFLOAT16:
        nodes.append(helper.make_node("Cast", ["x"], ["internal_x"], to=dtype))
        name = "internal_x"
    operation = helper.make_node(
        "MeanVarianceNormalization", [name], ["internal_y" if dtype == T.BFLOAT16 else "y"],
    )
    if axes is not None:
        operation.attribute.append(helper.make_attribute("axes", axes, attr_type=AttributeProto.INTS))
    nodes.append(operation)
    if dtype == T.BFLOAT16:
        nodes.append(helper.make_node("Cast", ["internal_y"], ["y"], to=T.FLOAT))
    return helper.make_model(
        helper.make_graph(nodes, "mean_variance_forms",
                          [helper.make_tensor_value_info("x", boundary_type, shape)],
                          [helper.make_tensor_value_info("y", boundary_type, shape)]),
        opset_imports=[helper.make_opsetid("", opset)], ir_version=11,
    ).SerializeToString()


def _reference(data, dtype, axes):
    work_type = np.float64 if dtype == T.DOUBLE else np.float32
    x = data.astype(TYPES[dtype]).astype(work_type)
    selected = (0, 2, 3) if axes is None else tuple(axes) if axes else None
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        mean = np.mean(x, axis=selected, keepdims=True, dtype=work_type)
        second_moment = np.mean(x * x, axis=selected, keepdims=True, dtype=work_type)
        epsilon = np.asarray(np.float32(1e-9), dtype=work_type)
        result = (x - mean) / (np.sqrt(second_moment - mean * mean) + epsilon)
    result = result.astype(TYPES[dtype])
    return result.astype(np.float32) if dtype == T.BFLOAT16 else result


def _check(actual, expected, dtype):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    tolerance = {T.FLOAT16: 2e-3, T.BFLOAT16: 1e-2, T.FLOAT: 2e-5,
                 T.DOUBLE: 2e-12}[dtype]
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance * 1e-2,
                               equal_nan=True)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("shape,axes", [
    ((2, 3, 2, 4), None),
    ((3, 4), [1]),
    ((3, 4), [-1, 0]),
    ((3, 4), []),
    ((), []),
])
def test_mean_variance_typed_forms(dtype, shape, axes, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _model(dtype, list(shape), axes=axes)
    session = _session(model)
    rng = np.random.default_rng(210)
    for scale, offset in [(0.5, 0), (1e-6, 0), (0.3, 0.1), (0.5, 0)]:
        data = np.asarray(rng.normal(size=shape) * scale + offset,
                          dtype=np.float32 if dtype == T.BFLOAT16 else TYPES[dtype])
        original = data.copy()
        expected = _reference(data, dtype, axes)
        _check(session.run(None, {"x": data})[0], expected, dtype)
        np.testing.assert_array_equal(data, original)
        if dtype == T.FLOAT:
            np.testing.assert_allclose(
                ReferenceEvaluator(model).run(None, {"x": data})[0], expected,
                rtol=2e-5, atol=2e-7, equal_nan=True,
            )


@pytest.mark.parametrize("opset", [9, 13, 18, 26])
@pytest.mark.parametrize("eager", [False, True])
def test_mean_variance_epsilon_follows_sqrt(opset, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _model(T.FLOAT, [2], axes=[0], opset=opset)
    data = np.array([-1e-6, 1e-6], np.float32)
    expected = ReferenceEvaluator(model).run(None, {"x": data})[0]
    actual = _session(model).run(None, {"x": data})[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
    assert actual[1] > 0.99


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("eager", [False, True])
def test_mean_variance_empty_axes_replays_actual_rank_and_empty_shapes(dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(dtype, axes=[]))
    for shape in [(2, 3), (3, 2), (), (0, 4), (4, 0), (2, 1, 3), (2, 3)]:
        size = int(np.prod(shape))
        data = np.arange(size, dtype=np.float32).reshape(shape) * 0.125
        data = np.asarray(data, dtype=np.float32 if dtype == T.BFLOAT16 else TYPES[dtype])
        expected = data.copy() if size == 0 else _reference(data, dtype, [])
        _check(session.run(None, {"x": data})[0], expected, dtype)


@pytest.mark.parametrize("eager", [False, True])
def test_mean_variance_invalid_live_rank_recovers(eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(axes=[1]))
    for offset in [0, 2]:
        data = np.arange(6, dtype=np.float32).reshape(2, 3) + offset
        _check(session.run(None, {"x": data})[0], _reference(data, T.FLOAT, [1]), T.FLOAT)
        with pytest.raises(Exception, match="(?i)axis|axes"):
            session.run(None, {"x": np.array(1, np.float32)})
    data = np.arange(6, dtype=np.float32).reshape(2, 3) * 0.25
    _check(session.run(None, {"x": data})[0], _reference(data, T.FLOAT, [1]), T.FLOAT)


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("eager", [False, True])
def test_mean_variance_default_axes_replay_actual_rank(dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(dtype))
    for shape in [(2, 2, 2, 2), (1, 2, 2, 2, 3), (2, 2, 2, 2)]:
        data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) * 0.125
        data = data.astype(np.float32 if dtype == T.BFLOAT16 else TYPES[dtype])
        _check(session.run(None, {"x": data})[0], _reference(data, dtype, None), dtype)
        with pytest.raises(Exception, match="(?i)axis|axes"):
            session.run(None, {"x": np.zeros((2, 3, 4), data.dtype)})


@pytest.mark.parametrize("shape,axes", [
    ([2, 3], None), ([], None), ([2, 3], [0, -2]),
    ([2, 3], [-(2**63)]), ([2, 3], [2**63 - 1]),
])
def test_mean_variance_invalid_static_axes_are_not_claimed(shape, axes):
    with pytest.raises(Exception, match="(?i)axis|axes|fallback|assigned|unsupported"):
        _session(_model(shape=shape, axes=axes))


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("data", [
    np.ones(3, np.float32),
    np.array([10000, 10001], np.float32),
    np.array([np.inf, 1], np.float32),
    np.array([np.nan, -1], np.float32),
])
def test_mean_variance_preserves_second_moment_and_nonfinite_behavior(data, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _model(T.FLOAT, list(data.shape), axes=[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = ReferenceEvaluator(model).run(None, {"x": data})[0]
    _check(_session(model).run(None, {"x": data})[0], expected, T.FLOAT)


def test_mean_variance_route_replay(monkeypatch):
    for eager in [False, True]:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
        session = _session(_model(T.FLOAT, ["rows", "columns"], axes=[-1]))
        for call, shape in enumerate([(2, 3), (2, 3), (3, 2), (2, 3), (3, 2)]):
            data = np.arange(6, dtype=np.float32).reshape(shape) * (call + 1) * 0.125
            _check(session.run(None, {"x": data})[0], _reference(data, T.FLOAT, [-1]), T.FLOAT)


def test_mean_variance_actual_routes(tmp_path):
    trace = tmp_path / "mean_variance_trace.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "mean_variance_route_replay", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path"
             and "MeanVarianceNormalization" in event["args"].get("ops", "").split(",")]
    assert [event["cache"] for event in paths if event["path"] == "general"] == [
        "MISS", "HIT", "RETRACE", "HIT", "HIT"]
    assert sum(event["path"] == "eager" for event in paths) == 5
