"""Versioned spatial norms, exact double computation, live geometry and narrow accumulation."""

from decimal import Decimal, localcontext
import json
import os
from pathlib import Path
import subprocess
import sys

import ml_dtypes
import numpy as np
import onnxruntime as ort
from onnx import TensorProto as T, helper, numpy_helper
import pytest


OPS = ["InstanceNormalization", "LpNormalization", "LRN"]
DTYPES = [np.float16, np.float32, np.float64]


def _model(op, shape, *, dtype=np.float32, opset=22, attrs=None, bf16=False):
    dtype = np.dtype(dtype)
    external_type = numpy_helper.from_array(np.array(0, dtype)).data_type
    internal_type = T.BFLOAT16 if bf16 else external_type
    inputs = [helper.make_tensor_value_info("x", external_type, shape)]
    names = ["x"]
    if op == "InstanceNormalization":
        inputs += [helper.make_tensor_value_info(name, external_type, [None])
                   for name in ["s", "b"]]
        names += ["s", "b"]
    nodes = []
    if bf16:
        for name in names:
            nodes.append(helper.make_node("Cast", [name], [name + "_bf16"], to=T.BFLOAT16))
        names = [name + "_bf16" for name in names]
    attributes = {"size": 3} if op == "LRN" else {}
    attributes.update(attrs or {})
    nodes.append(helper.make_node(op, names, ["normalized"], **attributes))
    if bf16:
        nodes.append(helper.make_node("Cast", ["normalized"], ["y"], to=external_type))
    else:
        nodes[-1].output[0] = "y"
    return helper.make_model(
        helper.make_graph(
            nodes, "spatial_norm_forms", inputs,
            [helper.make_tensor_value_info("y", external_type,
                                           None if shape is None else [None] * len(shape))],
            value_info=[helper.make_tensor_value_info("normalized", internal_type, None)]
            if bf16 else [],
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


def _feeds(op, shape, dtype=np.float32, *, magnitude=1.0, offset=0.0):
    x = (np.linspace(-0.7, 1.1, int(np.prod(shape))).reshape(shape) * magnitude + offset)
    feeds = {"x": x.astype(dtype)}
    if op == "InstanceNormalization":
        feeds.update(s=np.linspace(0.5, 1.5, shape[1]).astype(dtype),
                     b=np.linspace(-0.2, 0.3, shape[1]).astype(dtype))
    return feeds


def _instance_reference(data, attrs):
    x = data["x"]
    result = np.empty_like(x)
    with localcontext() as context:
        context.prec = 70
        epsilon = Decimal.from_float(float(np.float32(attrs.get("epsilon", 1e-5))))
        for batch, channel in np.ndindex(*x.shape[:2]):
            values = [Decimal.from_float(float(value)) for value in x[batch, channel].flat]
            mean = sum(values) / len(values)
            centered = [value - mean for value in values]
            variance = sum(value * value for value in centered) / len(values)
            denominator = (variance + epsilon).sqrt()
            scale = Decimal.from_float(float(data["s"][channel]))
            bias = Decimal.from_float(float(data["b"][channel]))
            normalized = [float("nan") if denominator == 0
                          else float(value / denominator * scale + bias) for value in centered]
            result[batch, channel] = np.array(normalized).reshape(x.shape[2:])
    return result


def _reference(op, feeds, attrs=None, *, bf16=False):
    attrs = attrs or {}
    dtype = ml_dtypes.bfloat16 if bf16 else feeds["x"].dtype
    compute_dtype = np.float64 if dtype == np.dtype(np.float64) else np.float32
    data = {key: value.astype(dtype).astype(compute_dtype) for key, value in feeds.items()}
    x = data["x"]
    if x.size == 0:
        return x.astype(dtype).astype(feeds["x"].dtype)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if op == "InstanceNormalization":
            result = _instance_reference(data, attrs)
        elif op == "LpNormalization":
            axis = attrs.get("axis", -1)
            if attrs.get("p", 2) == 1:
                norm = np.abs(x).sum(axis=axis, keepdims=True)
            else:
                norm = np.sqrt((x * x).sum(axis=axis, keepdims=True))
            result = np.where(norm == 0, 0, x / norm)
        else:
            size = attrs.get("size", 3)
            before = (size - 1) // 2
            after = size - 1 - before
            squares = x * x
            sums = np.empty_like(x)
            for channel in range(x.shape[1]):
                sums[:, channel] = squares[
                    :, max(0, channel - before):min(x.shape[1], channel + after + 1)
                ].sum(axis=1)
            alpha = compute_dtype(np.float32(attrs.get("alpha", 1e-4))) / compute_dtype(size)
            bias = np.float32(attrs.get("bias", 1.0))
            beta = np.float32(attrs.get("beta", 0.75))
            result = x / np.power(bias + alpha * sums, beta)
    return result.astype(dtype).astype(feeds["x"].dtype)


def _check(actual, expected, *, bf16=False):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if bf16:
        tolerance = dict(rtol=1e-2, atol=2e-3)
    elif expected.dtype == np.float16:
        tolerance = dict(rtol=2e-3, atol=5e-4)
    elif expected.dtype == np.float64:
        tolerance = dict(rtol=1e-12, atol=1e-12)
    else:
        tolerance = dict(rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(actual, expected, equal_nan=True, **tolerance)


@pytest.mark.parametrize("case", [
    "instance_v1", "instance_dynamic", "instance_empty", "instance_double",
    "lp_double", "lrn_dynamic", "lrn_empty", "lrn_double", "lrn_large_window",
    "half_lp_square", "half_instance_square", "half_lrn_square",
])
def test_spatial_norm_new_forms_strict(case):
    op = ("InstanceNormalization" if "instance" in case else
          "LpNormalization" if "lp" in case else "LRN")
    shape = (0, 3, 2) if "empty" in case else (2, 3, 4)
    dtype = np.float64 if "double" in case else np.float16 if "half" in case else np.float32
    attrs = {"size": (1 << 32) + 3, "alpha": 1e8} if case == "lrn_large_window" else {}
    feeds = _feeds(op, shape, dtype, magnitude=1000 if "half" in case else 1)
    model = _model(op, [None] * len(shape) if "dynamic" in case else list(shape),
                   dtype=dtype, opset=1 if case == "instance_v1" else 22, attrs=attrs)
    actual = _session(model).run(None, feeds)[0]
    _check(actual, _reference(op, feeds, attrs))


@pytest.mark.parametrize("op,opset", [
    ("InstanceNormalization", 1), ("InstanceNormalization", 6), ("InstanceNormalization", 22),
    ("LpNormalization", 1), ("LpNormalization", 22), ("LRN", 1), ("LRN", 13),
])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eager", [False, True])
def test_spatial_norm_schema_versions(op, opset, dtype, eager, monkeypatch):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    feeds = _feeds(op, (2, 3, 4, 2), dtype)
    model = _model(op, [2, 3, 4, 2], dtype=dtype, opset=opset)
    _check(_session(model).run(None, feeds)[0], _reference(op, feeds))
    if dtype == np.float32 and opset != 1:
        _check(_session(model, cpu=True).run(None, feeds)[0], _reference(op, feeds))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 3, 2, 4), (1, 3, 2, 2, 2)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eager", [False, True])
def test_spatial_norm_live_shapes_and_values(op, shape, dtype, eager, monkeypatch):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    session = _session(_model(op, [None] * len(shape), dtype=dtype))
    shapes = [shape, shape, (shape[0] + 1, shape[1] + 1, *shape[2:]), shape]
    for index, current_shape in enumerate(shapes):
        feeds = _feeds(op, current_shape, dtype, magnitude=1 + index * 0.25, offset=index * 0.1)
        _check(session.run(None, feeds)[0], _reference(op, feeds))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 3, 2, 4), (1, 3, 2, 2, 2)])
@pytest.mark.parametrize("eager", [False, True])
def test_spatial_norm_bfloat16(op, shape, eager, monkeypatch):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    model = _model(op, [None] * len(shape), bf16=True)
    session = _session(model)
    for magnitude in [1, 1000]:
        feeds = _feeds(op, shape, magnitude=magnitude)
        _check(session.run(None, feeds)[0], _reference(op, feeds, bf16=True), bf16=True)


@pytest.mark.parametrize("axis", [-3, -2, -1, 0, 1, 2])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eager", [False, True])
def test_lp_axis_orders_and_zero_norms(axis, p, dtype, eager, monkeypatch):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    attrs = dict(axis=axis, p=p)
    model = _model("LpNormalization", [2, 3, 4], dtype=dtype, attrs=attrs)
    session = _session(model)
    for magnitude in [0, 1e-4, 1000]:
        feeds = _feeds("LpNormalization", (2, 3, 4), dtype, magnitude=magnitude)
        _check(session.run(None, feeds)[0], _reference("LpNormalization", feeds, attrs))


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 6, 9, (1 << 31), (1 << 32) + 3, (1 << 63) - 1])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eager", [False, True])
def test_lrn_even_clipped_and_fullwidth_windows(size, dtype, eager, monkeypatch):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    attrs = dict(size=size, alpha=0.25, beta=0.6, bias=1.2)
    feeds = _feeds("LRN", (2, 3, 2), dtype, magnitude=20)
    model = _model("LRN", [2, 3, 2], dtype=dtype, attrs=attrs)
    _check(_session(model).run(None, feeds)[0], _reference("LRN", feeds, attrs))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("shape", [(0, 3, 2), (2, 0, 2), (2, 3, 0), (0, 0, 0)])
@pytest.mark.parametrize("eager", [False, True])
def test_spatial_norm_empty_recovery(op, shape, eager, monkeypatch):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    session = _session(_model(op, [None] * 3))
    for current_shape in [shape, (2, 3, 4), shape]:
        feeds = _feeds(op, current_shape)
        _check(session.run(None, feeds)[0], _reference(op, feeds))


@pytest.mark.parametrize("op", OPS)
def test_spatial_norm_unknown_rank_errors_and_recovery(op):
    attrs = {"axis": 2} if op == "LpNormalization" else {}
    session = _session(_model(op, None, attrs=attrs))
    invalid = {"x": np.ones((), np.float32)}
    if op == "InstanceNormalization":
        invalid.update(s=np.ones(3, np.float32), b=np.zeros(3, np.float32))
    with pytest.raises(Exception, match="rank|axis|MLX|Mlx"):
        session.run(None, invalid)
    for shape in [(2, 3, 4), (1, 2, 3, 4), (2, 3, 4)]:
        feeds = _feeds(op, shape)
        _check(session.run(None, feeds)[0], _reference(op, feeds, attrs))
    if op == "InstanceNormalization":
        feeds["s"] = np.ones(4, np.float32)
        with pytest.raises(Exception, match="affine|shape|MLX|Mlx"):
            session.run(None, feeds)
        feeds["s"] = np.ones(3, np.float32)
        _check(session.run(None, feeds)[0], _reference(op, feeds, attrs))


@pytest.mark.parametrize("axis", [-(1 << 63), -4, 3, (1 << 32) + 1, (1 << 63) - 1])
def test_lp_invalid_axis_is_not_narrowed(axis):
    model = _model("LpNormalization", [2, 3, 4], attrs={"axis": axis})
    with pytest.raises(Exception, match="fallback|axis"):
        _session(model)


@pytest.mark.parametrize("op", OPS)
def test_spatial_norm_native_bounds_and_recovery(op):
    session = _session(_model(op, [None] * 3))
    feeds = _feeds(op, (0, 3, 1 << 31))
    with pytest.raises(Exception, match="native bounds|MLX|Mlx"):
        session.run(None, feeds)
    feeds = _feeds(op, (1, 3, 2))
    _check(session.run(None, feeds)[0], _reference(op, feeds))


@pytest.mark.parametrize("epsilon", [0.0, 1e-5, 0.125])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eager", [False, True])
def test_instance_epsilon_and_constant_channels(epsilon, dtype, eager, monkeypatch):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    attrs = {"epsilon": epsilon}
    session = _session(_model("InstanceNormalization", [None] * 3, dtype=dtype, attrs=attrs))
    for magnitude in [0, 1e-4, 1000]:
        feeds = _feeds("InstanceNormalization", (2, 3, 4), dtype, magnitude=magnitude, offset=2)
        _check(session.run(None, feeds)[0], _reference("InstanceNormalization", feeds, attrs))


@pytest.mark.parametrize("op,attrs", [
    ("LpNormalization", {"p": 0}), ("LpNormalization", {"p": 3}),
    ("LRN", {"size": 0}), ("LRN", {"size": -1}),
])
def test_spatial_norm_invalid_attributes_are_declined(op, attrs):
    with pytest.raises(Exception, match="fallback|p=|size"):
        _session(_model(op, [2, 3, 4], attrs=attrs))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("eager", [False, True])
def test_spatial_norm_route_replay(op, dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(op, [None] * 3, dtype=dtype))
    for index, shape in enumerate([(2, 3, 4), (2, 3, 4), (1, 4, 2), (2, 3, 4)]):
        feeds = _feeds(op, shape, dtype, magnitude=1 + index, offset=index * 0.125)
        _check(session.run(None, feeds)[0], _reference(op, feeds))


def test_spatial_norm_actual_general_and_cpu_routes(tmp_path):
    trace = tmp_path / "spatial_norm_trace.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "spatial_norm_route_replay", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path"]
    for op in OPS:
        selected = [event for event in paths if op in event.get("ops", "")]
        assert [event["cache"] for event in selected if event["path"] == "general"] == [
            "MISS", "HIT", "RETRACE", "HIT"]
        assert sum(event["path"] == "eager" for event in selected) == 12
