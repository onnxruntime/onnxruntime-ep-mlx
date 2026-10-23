"""Strict Pow coverage: typed exponents, schema broadcasting, and live replay."""

from decimal import Decimal, localcontext
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort
from onnx import TensorProto as T, helper, numpy_helper
import pytest


FLOATS = [np.float16, np.float32, np.float64, "bf16"]
INTEGERS = [np.int8, np.uint8, np.int16, np.uint16,
            np.int32, np.uint32, np.int64, np.uint64]


def _type(dtype):
    return T.BFLOAT16 if dtype == "bf16" else helper.np_dtype_to_tensor_dtype(np.dtype(dtype))


def _model(base_dtype, exponent_dtype, base_shape, exponent_shape, *, opset=26,
           attrs=None, constant=None):
    inputs, nodes, initializers, names = [], [], [], []
    for name, dtype, shape in [
        ("base", base_dtype, base_shape), ("exponent", exponent_dtype, exponent_shape)
    ]:
        boundary_type = T.FLOAT if dtype == "bf16" else _type(dtype)
        if constant is not None and name in constant:
            initializers.append(numpy_helper.from_array(constant[name], name=name))
        else:
            inputs.append(helper.make_tensor_value_info(name, boundary_type, shape))
        if dtype == "bf16":
            nodes.append(helper.make_node("Cast", [name], [name + "_bf"], to=T.BFLOAT16))
            names.append(name + "_bf")
        else:
            names.append(name)
    nodes.append(helper.make_node("Pow", names, ["powered"], **(attrs or {})))
    output = "powered"
    if base_dtype == "bf16":
        nodes.append(helper.make_node("Cast", ["powered"], ["out"], to=T.FLOAT))
        output = "out"
    graph = helper.make_graph(
        nodes, "pow_forms", inputs,
        [helper.make_tensor_value_info(
            output, T.FLOAT if base_dtype == "bf16" else _type(base_dtype), None)],
        initializer=initializers,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=11)


def _session(model, *, cpu=False):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    if not cpu:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model.SerializeToString(), options,
        providers=["CPUExecutionProvider"] if cpu else ["MLXExecutionProvider"],
    )
    session.disable_fallback()
    return session


def _array(values, dtype):
    return np.asarray(values, np.float32 if dtype == "bf16" else dtype)


def _exact(actual, expected):
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(np.signbit(actual[finite]), np.signbit(expected[finite]))


@pytest.mark.parametrize("base_dtype", [np.float16, np.float32])
@pytest.mark.parametrize("eager", [False, True])
def test_pow_baseline_integer_parity(base_dtype, eager, monkeypatch):
    """This must fail on the immutable baseline, not merely in cast emulation."""
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    exponent_dtype = np.int16 if base_dtype == np.float16 else np.int32
    boundary = 2**11 if base_dtype == np.float16 else 2**24
    exponents = np.array([boundary, boundary + 1, boundary + 2, -boundary - 1],
                         exponent_dtype)
    base = np.full(4, -1, base_dtype)
    expected = np.array([-1 if int(n) % 2 else 1 for n in exponents], base_dtype)
    model = _model(base_dtype, exponent_dtype, [4], [4])
    feeds = {"base": base, "exponent": exponents}
    if base_dtype == np.float32:
        _exact(_session(model, cpu=True).run(None, feeds)[0], expected)
    _exact(_session(model).run(None, feeds)[0], expected)


@pytest.mark.parametrize("base_dtype", FLOATS)
@pytest.mark.parametrize("exponent_dtype", INTEGERS)
@pytest.mark.parametrize("eager", [False, True])
def test_pow_integer_limits_and_special_bases(base_dtype, exponent_dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    bounds = np.iinfo(exponent_dtype)
    values = [0, 1, 2, 3, bounds.max - 1, bounds.max]
    if bounds.min < 0:
        values += [-1, -2, -3, bounds.min, bounds.min + 1]
    exponents = np.asarray(values, exponent_dtype)[None, :]
    base = _array([-1, 1, -0.0, 0.0, -np.inf, np.inf, np.nan], base_dtype)[:, None]
    expected = np.empty((base.size, exponents.size), dtype=base.dtype)
    for row, value in enumerate(base.flat):
        for col, n in enumerate(exponents.flat):
            n = int(n)
            odd = n % 2 != 0
            if n == 0 or value == 1:
                result = 1
            elif value == -1:
                result = -1 if odd else 1
            elif np.isnan(value):
                result = np.nan
            elif value == 0:
                result = 0.0 if n > 0 else np.inf
                result = -result if np.signbit(value) and odd else result
            else:
                result = np.inf if n > 0 else 0.0
                result = -result if np.signbit(value) and odd else result
            expected[row, col] = result
    model = _model(base_dtype, exponent_dtype, list(base.shape), list(exponents.shape))
    _exact(_session(model).run(None, {"base": base, "exponent": exponents})[0], expected)


@pytest.mark.parametrize("base_dtype", FLOATS)
@pytest.mark.parametrize("exponent_dtype", FLOATS)
@pytest.mark.parametrize("eager", [False, True])
def test_pow_float_types_and_special_values(base_dtype, exponent_dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    base = _array([-1, -0.0, 0.0, 1, 4, np.inf, -np.inf, np.nan], base_dtype)[:, None]
    exponent = _array([-3, -2, -0.5, 0, 0.5, 2, 3, np.inf, -np.inf, np.nan],
                      exponent_dtype)[None, :]
    with np.errstate(all="ignore"):
        expected = np.power(base.astype(np.float64), exponent.astype(np.float64)).astype(base.dtype)
    model = _model(base_dtype, exponent_dtype, list(base.shape), list(exponent.shape))
    _exact(_session(model).run(None, {"base": base, "exponent": exponent})[0], expected)


@pytest.mark.parametrize("base_dtype", FLOATS)
@pytest.mark.parametrize("exponent_dtype", [np.float16, np.float32, np.float64, "bf16"])
def test_pow_tiny_positive_float_exponent(base_dtype, exponent_dtype):
    # Every value is representable in the exponent's original format.
    tiny = {np.float16: 2**-24, np.float32: 2**-100, np.float64: 2**-1000,
            "bf16": 2**-100}[exponent_dtype]
    exponent = _array([tiny, -tiny, 0], exponent_dtype)
    base = _array(0, base_dtype)
    expected = _array([0, np.inf, 1], base_dtype)
    _exact(_session(_model(base_dtype, exponent_dtype, [], [3])).run(
        None, {"base": base, "exponent": exponent})[0], expected)


@pytest.mark.parametrize("base_dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("exponent_dtype", INTEGERS)
def test_pow_integer_decimal_oracle(base_dtype, exponent_dtype):
    # Decimal exponentiation is independent of NumPy's mixed-type promotion and MLX power.
    values = [0.5, -2, 1.25, np.nextafter(base_dtype(1), base_dtype(2))]
    bound = np.iinfo(exponent_dtype).max
    exponents = [0, 3, min(bound, 2**24 + 1), min(bound, 2**48 + 1)]
    if np.iinfo(exponent_dtype).min < 0:
        exponents[0] = -3
    base = np.asarray(values, base_dtype)
    exponent = np.asarray(exponents, exponent_dtype)
    expected = []
    with localcontext() as context:
        context.prec = 100
        for value, n in zip(base, exponent, strict=True):
            decimal_base = Decimal(float(value))
            logarithm = decimal_base.copy_abs().ln() * int(n)
            if logarithm > 800:
                expected.append(np.inf)
            elif logarithm < -800:
                expected.append(0)
            else:
                expected.append(float(decimal_base ** int(n)))
    with np.errstate(over="ignore"):
        expected = np.asarray(expected, base_dtype)
    actual = _session(_model(base_dtype, exponent_dtype, [4], [4])).run(
        None, {"base": base, "exponent": exponent})[0]
    tolerance = {np.float16: 1e-3, np.float32: 1e-5, np.float64: 1e-12}[base_dtype]
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=0)


@pytest.mark.parametrize("base_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("exponent_dtype", [np.int32, np.int64, np.uint64])
@pytest.mark.parametrize("eager", [False, True])
def test_pow_large_integer_near_one(base_dtype, exponent_dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    exponents = [2**24 - 1, 2**24, 2**24 + 1]
    if exponent_dtype != np.int32 and base_dtype == np.float64:
        exponents += [2**53 - 1, 2**53, 2**53 + 1]
    if exponent_dtype != np.uint64:
        exponents += [-n for n in exponents]
    exponent = np.asarray(exponents, exponent_dtype)
    base = np.full(exponent.shape, np.nextafter(base_dtype(1), base_dtype(2)), base_dtype)
    with localcontext() as context:
        context.prec = 100
        expected = np.asarray([float(Decimal(float(x)) ** int(n))
                               for x, n in zip(base, exponent, strict=True)], base_dtype)
    actual = _session(_model(base_dtype, exponent_dtype, [len(exponents)], [len(exponents)])).run(
        None, {"base": base, "exponent": exponent})[0]
    np.testing.assert_allclose(
        actual, expected, rtol=1e-5 if base_dtype == np.float32 else 1e-12, atol=0)


@pytest.mark.parametrize("base_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("exponent_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("eager", [False, True])
def test_pow_float_near_one(base_dtype, exponent_dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    near_one = np.nextafter(base_dtype(1), base_dtype(2))
    base = np.asarray([near_one, -near_one, near_one, -near_one], base_dtype)
    exponent = np.asarray([2**24, 2**24, -(2**24) + 1, -(2**24) + 1], exponent_dtype)
    with localcontext() as context:
        context.prec = 100
        expected = np.asarray([float(Decimal(float(x)) ** int(n))
                               for x, n in zip(base, exponent, strict=True)], base_dtype)
    actual = _session(_model(base_dtype, exponent_dtype, [4], [4])).run(
        None, {"base": base, "exponent": exponent})[0]
    np.testing.assert_allclose(
        actual, expected, rtol=1e-5 if base_dtype == np.float32 else 1e-12, atol=0)


@pytest.mark.parametrize("opset", [1, 6, 7, 11, 12, 13, 15, 26])
@pytest.mark.parametrize("constant", [None, "base", "exponent"])
def test_pow_schema_versions_and_initializers(opset, constant):
    base = np.array([[0.5, 1, 2], [4, 8, 16]], np.float32)
    exponent = np.array([[2, 3, -1], [-2, 0, 0.5]], np.float32)
    model = _model(np.float32, np.float32, [2, 3], [2, 3], opset=opset,
                   constant={constant: {"base": base, "exponent": exponent}[constant]}
                   if constant else None)
    feeds = {name: value for name, value in [("base", base), ("exponent", exponent)]
             if name != constant}
    _exact(_session(model).run(None, feeds)[0], np.power(base, exponent))


@pytest.mark.parametrize("shape,exponent_shape,attrs", [
    ((2, 3, 4), (3,), {"broadcast": 1, "axis": 1}),
    ((2, 3, 4), (2, 3), {"broadcast": 1, "axis": 0}),
    ((2, 3, 4), (4,), {"broadcast": 1}),
    ((2, 3, 4), (), {"broadcast": 1}),
    ((2, 0, 4), (0,), {"broadcast": 1, "axis": 1}),
])
@pytest.mark.parametrize("opset", [1, 6])
def test_pow_legacy_axis(shape, exponent_shape, attrs, opset):
    base = np.full(shape, 2, np.float32)
    exponent = np.arange(np.prod(exponent_shape), dtype=np.float32).reshape(exponent_shape)
    axis = attrs.get("axis", len(shape) - len(exponent_shape))
    aligned = [1] * len(shape)
    aligned[axis:axis + len(exponent_shape)] = exponent_shape
    expected = np.power(base, exponent.reshape(aligned))
    model = _model(np.float32, np.float32, list(shape), list(exponent_shape),
                   attrs=attrs, opset=opset)
    _exact(_session(model).run(None, {"base": base, "exponent": exponent})[0], expected)


@pytest.mark.parametrize("base_shape,exponent_shape", [
    ((), ()), ((), (2, 3)), ((2, 1, 3), (1, 4, 1)), ((0, 3), (1, 3)),
    ((2, 0, 3), (3,)), ((1, 3), (0, 1)), ((3,), (2, 1)),
])
@pytest.mark.parametrize("exponent_dtype", [np.float32, np.int64, np.uint64])
def test_pow_multidirectional_and_empty_shapes(base_shape, exponent_shape, exponent_dtype):
    base = np.full(base_shape, -2, np.float32)
    exponent = np.full(exponent_shape, 3, exponent_dtype)
    expected = np.full(np.broadcast_shapes(base_shape, exponent_shape), -8, np.float32)
    model = _model(np.float32, exponent_dtype, list(base_shape), list(exponent_shape))
    _exact(_session(model).run(None, {"base": base, "exponent": exponent})[0], expected)


@pytest.mark.parametrize("exponent_dtype", [np.int32, np.int64, np.uint64])
@pytest.mark.parametrize("eager", [False, True])
def test_pow_live_replay(exponent_dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(np.float32, exponent_dtype, [None, None], [None]))
    for index, shape in enumerate([(2, 3), (2, 3), (1, 4), (2, 3)]):
        base = np.full(shape, -1 if index % 2 == 0 else 1, np.float32)
        exponent = np.arange(shape[-1], dtype=exponent_dtype) + exponent_dtype(2**24 + index)
        expected = np.asarray([[(-1 if int(n) % 2 else 1) if index % 2 == 0 else 1
                                for n in exponent]] * shape[0], np.float32)
        _exact(session.run(None, {"base": base, "exponent": exponent})[0], expected)


def test_pow_invalid_broadcast_recovers():
    session = _session(_model(np.float32, np.int64, [None, None], [None]))
    base = np.full((2, 3), -1, np.float32)
    feeds = {"base": base, "exponent": np.array([2**24 + 1] * 3, np.int64)}
    _exact(session.run(None, feeds)[0], base)
    with pytest.raises(Exception, match="broadcast|shape|dimension|MLX|Mlx"):
        session.run(None, {**feeds, "exponent": np.ones(4, np.int64)})
    _exact(session.run(None, feeds)[0], base)


@pytest.mark.parametrize("broadcast", [0, 1])
def test_pow_legacy_dynamic_shapes_recover(broadcast):
    attrs = {"broadcast": broadcast}
    if broadcast:
        attrs["axis"] = 0
    session = _session(_model(
        np.float32, np.float32, [None, None], [None, None], opset=6, attrs=attrs))
    for shape in [(2, 3), (2, 3), (1, 4), (2, 3)]:
        base = np.full(shape, 2, np.float32)
        exponent = np.full(shape, 3, np.float32)
        feeds = {"base": base, "exponent": exponent}
        _exact(session.run(None, feeds)[0], np.full(shape, 8, np.float32))
    with pytest.raises(Exception, match="broadcast|shape|rank|MLX|Mlx"):
        session.run(None, {"base": np.ones((1, 3), np.float32),
                           "exponent": np.ones((2, 3), np.float32)})
    _exact(session.run(None, feeds)[0], np.full(shape, 8, np.float32))


@pytest.mark.parametrize("base_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("exponent_dtype", INTEGERS + FLOATS)
def test_pow_integer_base_boundary(base_dtype, exponent_dtype):
    model = _model(base_dtype, exponent_dtype, [2], [2])
    with pytest.raises(Exception, match="fallback|not supported|not implemented|NOT_IMPLEMENTED"):
        _session(model)


@pytest.mark.parametrize("exponent_dtype", [np.int64, np.uint64, np.float32])
def test_pow_shapeless_replay(exponent_dtype):
    model = _model(np.float32, exponent_dtype, [None, 1, 8], [None, 1, 8])
    # The decode configuration requires a real KV-cache contract, not just an input_ids name.
    model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    for name in ("q", "k", "v"):
        model.graph.input.append(helper.make_tensor_value_info(name, T.FLOAT, [1, 1, 8]))
    for name in ("pk", "pv"):
        model.graph.input.append(helper.make_tensor_value_info(name, T.FLOAT, [1, 1, None, 8]))
    for name in ("lengths", "total"):
        model.graph.input.append(helper.make_tensor_value_info(name, T.INT32, [1]))
    model.graph.node.insert(0, helper.make_node(
        "GroupQueryAttention", ["q", "k", "v", "pk", "pv", "lengths", "total"],
        ["attention", "present_k", "present_v"], domain="com.microsoft",
        num_heads=1, kv_num_heads=1, do_rotary=0))
    model.graph.node.append(helper.make_node("Add", ["attention", "powered"], ["out"]))
    model.graph.output[0].name = "out"
    for name in ("present_k", "present_v"):
        model.graph.output.append(helper.make_tensor_value_info(name, T.FLOAT, [1, 1, None, 8]))
    session = _session(model)
    for index, batch in enumerate([2, 2, 3, 2]):
        base = np.full((batch, 1, 8), -1, np.float32)
        values = np.arange(batch * 8).reshape(batch, 1, 8) + 2**24 + index
        exponent = values.astype(exponent_dtype)
        expected = np.asarray([-1 if int(n) % 2 else 1 for n in exponent.flat], np.float32)
        past = [1, 2, 3, 1][index]
        feeds = {"base": base, "exponent": exponent,
                 **{name: np.zeros((1, 1, 8), np.float32) for name in ("q", "k", "v")},
                 **{name: np.zeros((1, 1, past, 8), np.float32) for name in ("pk", "pv")},
                 "lengths": np.array([past], np.int32),
                 "total": np.array([past + 1], np.int32)}
        actual = session.run(None, feeds)
        _exact(actual[0], expected.reshape(base.shape))
        assert actual[1].shape == (1, 1, past + 1, 8)
        assert actual[2].shape == (1, 1, past + 1, 8)


def test_pow_actual_general_route(tmp_path):
    trace = tmp_path / "pow_trace.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "pow_live_replay", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path" and "Pow" in event["args"].get("ops", "")]
    assert [event["cache"] for event in paths if event["path"] == "general"] == [
        "MISS", "HIT", "RETRACE", "HIT"] * 3
    assert sum(event["path"] == "eager" for event in paths) == 12


def test_pow_actual_shapeless_route(tmp_path):
    trace = tmp_path / "pow_decode_trace.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "pow_shapeless_replay", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0"},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path" and "Pow" in event["args"].get("ops", "")]
    assert [event["cache"] for event in paths if event["path"] == "decode"] == [
        "MISS", "HIT", "HIT", "HIT"] * 3, paths
    assert not [event for event in paths if event["path"] != "decode"]
