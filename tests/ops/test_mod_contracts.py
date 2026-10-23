"""Versioned Mod contracts with exact represented-input oracles."""

from fractions import Fraction
import json
import math

import ml_dtypes
import numpy as np
import onnx_ir as ir
from onnx_ir import DataType as DT
import onnxruntime as ort
import pytest

import _models as m


FLOATS = [(DT.FLOAT16, np.float16), (DT.FLOAT, np.float32), (DT.DOUBLE, np.float64)]
INTEGERS = [
    (DT.INT8, np.int8), (DT.INT16, np.int16), (DT.INT32, np.int32), (DT.INT64, np.int64),
    (DT.UINT8, np.uint8), (DT.UINT16, np.uint16), (DT.UINT32, np.uint32), (DT.UINT64, np.uint64),
]


def _model(dtype, a_shape, b_shape, *, fmod=1, opset=13):
    return m.make_model(
        "Mod",
        [m.tensor("a", dtype, a_shape), m.tensor("b", dtype, b_shape)],
        [m.tensor("out", dtype, np.broadcast_shapes(a_shape, b_shape))],
        attributes={} if fmod is None else {"fmod": fmod},
        opset=opset,
    )


def _fmod_reference(a, b):
    a, b = np.broadcast_arrays(a, b)
    result = np.empty(a.shape, dtype=a.dtype)
    for index in np.ndindex(a.shape):
        x, y = float(a[index]), float(b[index])
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or y == 0:
            value = math.nan
        elif math.isinf(y):
            value = x
        else:
            remainder = abs(Fraction(x)) % abs(Fraction(y))
            value = math.copysign(float(remainder), x)
        result[index] = value
    return result


def _assert_float_exact(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[finite], expected[finite])
    np.testing.assert_array_equal(np.signbit(actual[finite]), np.signbit(expected[finite]))


def _float_inputs(dtype):
    info = np.finfo(dtype)
    tiny = dtype(info.smallest_subnormal)
    a = np.array([
        dtype(1e4 if dtype == np.float16 else 1e20),
        info.max, info.max, info.tiny, tiny, 5.5, 5.5, 0, -0.0, np.inf,
        -np.inf, np.nan, 1, 0, 0, tiny, info.tiny, 1, -0.0,
    ], dtype=dtype)
    b = np.array([
        3, 3, info.tiny, tiny * dtype(3), tiny * dtype(3), 2, -2, 3, -3,
        2, np.inf, 3, 0, 0, np.nan, info.max, info.max, np.inf, np.inf,
    ], dtype=dtype)
    return a, b


@pytest.mark.parametrize("dtype,np_dtype", FLOATS)
@pytest.mark.parametrize("opset", [10, 13, 26])
@pytest.mark.parametrize("eager", [False, True], ids=["general", "eager"])
def test_mod_floating_exact_replay(dtype, np_dtype, opset, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    a, b = _float_inputs(np_dtype)
    model = _model(dtype, a.shape, b.shape, opset=opset)
    m.assert_mlx_claims(model, {"a": a, "b": b})
    session = m._session(model, m.EP_PROVIDERS)
    for x, y in [(a, b), (-a, -b), (np.roll(a, 4), np.roll(b, -3)), (a, b)]:
        actual = session.run(None, {"a": x, "b": y})[0]
        _assert_float_exact(actual, _fmod_reference(x, y))


@pytest.mark.parametrize("dtype,np_dtype", FLOATS)
@pytest.mark.parametrize("shapes", [((), ()), ((), (4,)), ((4,), ()), ((2, 1, 4), (3, 1))])
@pytest.mark.parametrize("eager", [False, True], ids=["general", "eager"])
def test_mod_scalar_multidirectional(dtype, np_dtype, shapes, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    a_shape, b_shape = shapes
    a = np.full(a_shape, 10000 if np_dtype == np.float16 else 1e20, dtype=np_dtype)
    b = np.full(b_shape, -3, dtype=np_dtype)
    model = _model(dtype, a_shape, b_shape)
    m.assert_mlx_claims(model, {"a": a, "b": b})
    session = m._session(model, m.EP_PROVIDERS)
    for x, y in [(a, b), (-a, -b), (a, b)]:
        _assert_float_exact(session.run(None, {"a": x, "b": y})[0], _fmod_reference(x, y))


@pytest.mark.parametrize("dtype,np_dtype", FLOATS)
@pytest.mark.parametrize("eager", [False, True], ids=["general", "eager"])
def test_mod_floating_bit_patterns(dtype, np_dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    rng = np.random.default_rng(1907)
    a = np.frombuffer(rng.bytes(128 * np.dtype(np_dtype).itemsize), dtype=np_dtype).copy()
    b = np.frombuffer(rng.bytes(a.nbytes), dtype=np_dtype).copy()
    model = _model(dtype, a.shape, b.shape)
    m.assert_mlx_claims(model, {"a": a, "b": b})
    _assert_float_exact(m.run_mlx(model, {"a": a, "b": b})[0], _fmod_reference(a, b))


@pytest.mark.parametrize("dtype,np_dtype", FLOATS)
@pytest.mark.parametrize("eager", [False, True], ids=["general", "eager"])
def test_mod_symbolic_empty_replay(dtype, np_dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = m.make_model(
        "Mod",
        [m.tensor("a", dtype, ["N", 1, 4]), m.tensor("b", dtype, [1, "M", 1])],
        [m.tensor("out", dtype, ["N", "M", 4])],
        attributes={"fmod": 1},
    )
    session = m._session(model, m.EP_PROVIDERS)
    for n, length in [(2, 3), (1, 1), (0, 3), (2, 3)]:
        a = np.full((n, 1, 4), -10000, dtype=np_dtype)
        b = np.full((1, length, 1), 3, dtype=np_dtype)
        _assert_float_exact(session.run(None, {"a": a, "b": b})[0], _fmod_reference(a, b))


@pytest.mark.parametrize("eager", [False, True], ids=["general", "eager"])
def test_mod_bfloat16_exact_replay(eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    a, b = _float_inputs(np.float32)
    # Keep the BF16 range finite rather than overflowing FLOAT32_MAX during the input Cast.
    a[1:3] = np.float32(3e38)
    a = a.astype(ml_dtypes.bfloat16).astype(np.float32)
    b = b.astype(ml_dtypes.bfloat16).astype(np.float32)
    model = m.bf16_interior_model(
        "Mod", [("a", list(a.shape)), ("b", list(b.shape))], list(a.shape),
        attributes={"fmod": 1},
    )
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "bf16_mod")
    session = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    for x, y in [(a, b), (-a, -b), (a, b)]:
        expected = _fmod_reference(x, y).astype(ml_dtypes.bfloat16).astype(np.float32)
        _assert_float_exact(session.run(None, {"a": x, "b": y})[0], expected)
    with open(session.end_profiling()) as profile:
        events = json.load(profile)
    nodes = [event["args"] for event in events if event.get("cat") == "Node"]
    assert any(node.get("provider") == "MLXExecutionProvider" for node in nodes)
    assert not any(
        node.get("op_name") == "Mod" and node.get("provider") == "CPUExecutionProvider"
        for node in nodes
    )


def _integer_reference(a, b, fmod):
    a, b = np.broadcast_arrays(a, b)
    result = np.empty(a.shape, dtype=a.dtype)
    for index in np.ndindex(a.shape):
        x, y = int(a[index]), int(b[index])
        value = x % y
        if fmod:
            value = abs(x) % abs(y)
            if x < 0:
                value = -value
        result[index] = value
    return result


@pytest.mark.parametrize("dtype,np_dtype", INTEGERS)
@pytest.mark.parametrize("opset,fmod", [(10, 0), (10, 1), (12, 1), (13, 0), (26, None)])
@pytest.mark.parametrize("eager", [False, True], ids=["default-route", "eager"])
def test_mod_integer_contract(dtype, np_dtype, opset, fmod, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    info = np.iinfo(np_dtype)
    signed = info.min < 0
    a = np.array([info.min, info.max, 7, 0, info.max - 1, info.min + 1], dtype=np_dtype)
    b = np.array([3, -3 if signed else 3, -1 if signed else 1, 3, 7, 3], dtype=np_dtype)
    model = _model(dtype, a.shape, b.shape, opset=opset, fmod=fmod)
    m.assert_mlx_claims(model, {"a": a, "b": b})
    session = m._session(model, m.EP_PROVIDERS)
    for x, y in [(a, b), (np.roll(a, 1), b), (a, b)]:
        actual = session.run(None, {"a": x, "b": y})[0]
        assert actual.dtype == np_dtype
        np.testing.assert_array_equal(actual, _integer_reference(x, y, fmod))
    y = np.array([info.max, info.min if signed else 1, 3, 1, info.max, 7], dtype=np_dtype)
    np.testing.assert_array_equal(
        session.run(None, {"a": a, "b": y})[0], _integer_reference(a, y, fmod),
    )
    if signed:
        x = np.full(a.shape, info.min, dtype=np_dtype)
        y = np.full(b.shape, -1, dtype=np_dtype)
        np.testing.assert_array_equal(session.run(None, {"a": x, "b": y})[0], np.zeros_like(x))


@pytest.mark.parametrize("dtype,np_dtype", INTEGERS)
@pytest.mark.parametrize("fmod,opset", [(0, 13), (1, 10)])
def test_mod_integer_zero_divisor_recovery(dtype, np_dtype, fmod, opset):
    a = np.array([7, 8], dtype=np_dtype)
    b = np.array([3, 3], dtype=np_dtype)
    model = _model(dtype, a.shape, b.shape, fmod=fmod, opset=opset)
    session = m._session(model, m.EP_PROVIDERS)
    for divisor in [b, np.array([3, 0], dtype=np_dtype), b]:
        if np.any(divisor == 0):
            with pytest.raises(Exception, match="Mod integer divisor must not contain zero"):
                session.run(None, {"a": a, "b": divisor})
        else:
            np.testing.assert_array_equal(
                session.run(None, {"a": a, "b": divisor})[0],
                _integer_reference(a, divisor, fmod),
            )


@pytest.mark.parametrize("divisor", [0, 3, -1])
def test_mod_initializer_divisor(divisor):
    a = m.tensor("a", DT.INT64, [2])
    b = ir.Value(name="b", const_value=ir.tensor(np.array(divisor, np.int64)))
    out = m.tensor("out", DT.INT64, [2])
    node = ir.node("Mod", [a, b], outputs=[out])
    graph = ir.Graph([a], [out], nodes=[node], initializers=[b], opset_imports={"": 13})
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    data = np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max], np.int64)
    if divisor == 0:
        with pytest.raises(Exception, match="Mod integer divisor must not contain zero"):
            m.run_mlx(model, {"a": data})
    else:
        m.assert_mlx_claims(model, {"a": data})
        np.testing.assert_array_equal(
            m.run_mlx(model, {"a": data})[0], _integer_reference(data, np.array(divisor), False),
        )


@pytest.mark.parametrize("opset", [13, 26])
def test_mod_integer_fmod_declined_at_schema13(opset):
    model = _model(DT.INT64, [2], [2], opset=opset)
    m.assert_op_declined(
        model, {"a": np.array([-7, 7], np.int64), "b": np.array([3, -3], np.int64)}, "Mod",
    )


@pytest.mark.parametrize("fmod", [None, 0, -1, 2])
def test_mod_invalid_float_attribute_rejected(fmod):
    model = _model(DT.FLOAT, [2], [2], fmod=fmod)
    with pytest.raises(Exception, match="fmod"):
        session = m._session(model, m.EP_PROVIDERS)
        session.run(None, {"a": np.array([7, -7], np.float32), "b": np.array([3, 3], np.float32)})
