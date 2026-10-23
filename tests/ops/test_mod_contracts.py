"""Versioned Mod contracts with exact represented-input oracles."""

from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
import sys

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
@pytest.mark.parametrize("eager", [False, True], ids=["default-route", "forced-eager"])
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
@pytest.mark.parametrize("eager", [False, True], ids=["default-route", "forced-eager"])
def test_mod_scalar_multidirectional(dtype, np_dtype, shapes, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    a_shape, b_shape = shapes
    a = np.full(a_shape, 10000 if np_dtype == np.float16 else 1e20, dtype=np_dtype)
    b = np.full(b_shape, -3, dtype=np_dtype)
    model = _model(dtype, a_shape, b_shape)
    m.assert_mlx_claims(model, {"a": a, "b": b})
    session = m._session(model, m.EP_PROVIDERS)
    for x, y in [(a, b), (-a, -b), (a, b)]:
        x, y = np.asarray(x), np.asarray(y)
        _assert_float_exact(session.run(None, {"a": x, "b": y})[0], _fmod_reference(x, y))


@pytest.mark.parametrize("dtype,np_dtype", FLOATS)
@pytest.mark.parametrize("eager", [False, True], ids=["default-route", "forced-eager"])
def test_mod_floating_bit_patterns(dtype, np_dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    rng = np.random.default_rng(1907)
    a = np.frombuffer(rng.bytes(128 * np.dtype(np_dtype).itemsize), dtype=np_dtype).copy()
    b = np.frombuffer(rng.bytes(a.nbytes), dtype=np_dtype).copy()
    model = _model(dtype, a.shape, b.shape)
    m.assert_mlx_claims(model, {"a": a, "b": b})
    _assert_float_exact(m.run_mlx(model, {"a": a, "b": b})[0], _fmod_reference(a, b))


@pytest.mark.parametrize("dtype,np_dtype", FLOATS)
@pytest.mark.parametrize("eager", [False, True], ids=["default-route", "forced-eager"])
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


@pytest.mark.parametrize("eager", [False, True], ids=["default-route", "forced-eager"])
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
    session.disable_fallback()
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
        session = m._session(model, m.EP_PROVIDERS)
        session.disable_fallback()
        with pytest.raises(Exception, match="Mod integer divisor must not contain zero"):
            session.run(None, {"a": data})
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


def _mixed_route_replay(case):
    dtype, np_dtype = {
        "float": (DT.FLOAT, np.float32),
        "double": (DT.DOUBLE, np.float64),
        "int64": (DT.INT64, np.int64),
        "bfloat16": (DT.BFLOAT16, ml_dtypes.bfloat16),
    }[case]
    boundary_dtype = DT.FLOAT if case == "bfloat16" else dtype
    shape = [1, "T", 4]
    a, b, offset = [m.tensor(name, boundary_dtype, shape) for name in ("a", "b", "offset")]
    inputs = [a, b, offset]
    nodes = []
    if case == "bfloat16":
        a, b, offset = [m.tensor(name, dtype, shape) for name in ("a_bf", "b_bf", "offset_bf")]
        for source, dest in zip(inputs, (a, b, offset), strict=True):
            nodes.append(ir.node("Cast", [source], attributes={"to": int(dtype)}, outputs=[dest]))
    added, remainder, result = [m.tensor(name, dtype, shape) for name in ("added", "mod", "result")]
    nodes.extend([
        ir.node("Add", [a, offset], outputs=[added]),
        ir.node("Mod", [added, b], attributes={"fmod": 0 if case == "int64" else 1}, outputs=[remainder]),
        ir.node("Add", [remainder, offset], outputs=[result]),
    ])
    if case == "bfloat16":
        out = m.tensor("out", DT.FLOAT, shape)
        nodes.append(ir.node("Cast", [result], attributes={"to": int(DT.FLOAT)}, outputs=[out]))
    else:
        out = result
    graph = ir.Graph(inputs, [out], nodes=nodes, opset_imports={"": 13})
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    session = m._session(model, m.EP_PROVIDERS)
    session.disable_fallback()
    calls = [(1, 10000), (1, -10000), (3, 7777)] if case == "int64" else [
        (1, 10000), (3, -10000), (1, 7777),
    ]
    for index, (length, value) in enumerate(calls):
        x = np.full((1, length, 4), value, dtype=np_dtype)
        y = np.full(x.shape, 3, dtype=np_dtype)
        z = np.full(x.shape, 1, dtype=np_dtype)
        expected = (
            _integer_reference(x + z, y, False)
            if case == "int64" else _fmod_reference(x + z, y)
        ) + z
        feeds = {"a": x, "b": y, "offset": z}
        if case == "bfloat16":
            feeds = {name: array.astype(np.float32) for name, array in feeds.items()}
            expected = expected.astype(np.float32)
        actual = session.run(None, feeds)[0]
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
        if case == "int64" and index == 0:
            with pytest.raises(Exception, match="Mod integer divisor must not contain zero"):
                session.run(None, {**feeds, "b": np.zeros_like(y)})


def _route_events(replay, case, tmp_path):
    trace = tmp_path / "mod_route.json"
    code = (
        "import os; import onnxruntime as ort; "
        "ort.register_execution_provider_library('MLXExecutionProvider', "
        "os.environ['ONNXRUNTIME_MLX_EP_LIB']); "
        f"from test_mod_contracts import {replay}; "
        f"{replay}({case!r})"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parent,
        env={
            **os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
            "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0",
            "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0",
            "ORT_DISABLE_TELEMETRY": "1",
        },
        check=True,
        timeout=120,
    )
    return json.loads(trace.read_text())


@pytest.mark.parametrize("case", ["float", "double", "int64", "bfloat16"])
def test_mod_actual_route_and_mixed_graph(case, tmp_path):
    events = _route_events("_mixed_route_replay", case, tmp_path)
    paths = [
        event["args"]["path"] for event in events
        if event.get("cat") == "ep.path" and "Mod" in event.get("args", {}).get("ops", "").split(",")
    ]
    assert len(paths) == 3, paths
    assert set(paths) == {"eager"}, paths
    add_paths = [
        event["args"]["path"] for event in events
        if event.get("cat") == "ep.path" and "Add" in event.get("args", {}).get("ops", "").split(",")
    ]
    assert len(add_paths) == (7 if case == "int64" else 6), add_paths
    assert set(add_paths) == ({"eager"} if case == "double" else {"general"}), add_paths


def _shape_consumer_replay(case):
    floating = case == "float_unsqueeze"
    operation = "Unsqueeze" if floating else case
    shape = [2, 3] if operation == "Unsqueeze" else [1, 2, 1] if operation == "Squeeze" else [2]
    control_shape = shape if floating else [2] if operation == "Pad" else [1]
    dtype = DT.FLOAT if floating else DT.INT64
    a, b = [m.tensor(name, dtype, control_shape) for name in ("a", "b")]
    remainder = m.tensor("remainder", dtype, control_shape)
    nodes = [ir.node("Mod", [a, b], attributes={"fmod": int(floating)}, outputs=[remainder])]
    inputs, initializers = [a, b], []
    rank = len(shape) + (operation == "Unsqueeze") - (operation == "Squeeze")
    out = m.tensor("out", DT.FLOAT, [f"D{i}" for i in range(rank)])
    if floating:
        axes = ir.Value(name="axes", const_value=ir.tensor(np.array([0], np.int64)))
        initializers.append(axes)
        nodes.append(ir.node("Unsqueeze", [remainder, axes], outputs=[out]))
    else:
        x = m.tensor("x", DT.FLOAT, shape)
        inputs.append(x)
        nodes.append(ir.node(operation, [x, remainder], outputs=[out]))
    graph = ir.Graph(inputs, [out], nodes=nodes, initializers=initializers, opset_imports={"": 13})
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    session = m._session(model, m.EP_PROVIDERS)
    session.disable_fallback()
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    for index in range(3):
        if floating:
            a = np.full(shape, [1e20, -1e20, 1e20][index], np.float32)
            b = np.full(shape, 3, np.float32)
            expected = np.expand_dims(_fmod_reference(a, b), axis=0)
            feeds = {"a": a, "b": b}
        else:
            controls = (
                [[1, 0], [0, 1], [1, 0]] if operation == "Pad"
                else [[0], [2], [0]]
            )[index]
            a = np.array(controls, np.int64)
            b = np.full(a.shape, 2 if operation == "Pad" else 3, np.int64)
            expected = (
                np.pad(x, (a[0], a[1])) if operation == "Pad"
                else np.squeeze(x, axis=int(a[0])) if operation == "Squeeze"
                else np.expand_dims(x, axis=int(a[0]))
            )
            feeds = {"x": x, "a": a, "b": b}
        actual = session.run(None, feeds)[0]
        _assert_float_exact(actual, expected)
        if not floating and index == 0:
            with pytest.raises(Exception, match="Mod integer divisor must not contain zero"):
                session.run(None, {**feeds, "b": np.zeros_like(b)})


@pytest.mark.parametrize("case", ["Pad", "Squeeze", "Unsqueeze", "float_unsqueeze"])
def test_mod_actual_route_with_live_shape_consumer(case, tmp_path):
    events = _route_events("_shape_consumer_replay", case, tmp_path)
    paths = [event["args"] for event in events if event.get("cat") == "ep.path"]
    mod_paths = [item["path"] for item in paths if "Mod" in item["ops"].split(",")]
    operation = "Unsqueeze" if case == "float_unsqueeze" else case
    consumer_paths = [
        item["path"] for item in paths if operation in item["ops"].split(",")
    ]
    assert mod_paths == ["eager"] * 3, mod_paths
    assert consumer_paths == ["general"] * 3, consumer_paths
