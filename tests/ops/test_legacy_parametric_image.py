"""Strict contracts for deprecated standard-domain schema-1 math operators."""

from contextlib import contextmanager
import json
from decimal import Decimal, localcontext
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from onnx import AttributeProto
import onnx_ir as ir
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_pybind11_state import EPFail, Fail, InvalidGraph
import pytest

import _models as m

DT = ir.DataType
FLOATS = [(DT.FLOAT16, np.float16), (DT.FLOAT, np.float32), (DT.DOUBLE, np.float64)]
CPU_ASSIGNMENT_DISABLED = (
    "[ONNXRuntimeError] : 1 : FAIL : This session contains graph nodes that are "
    "assigned to the default CPU EP, but fallback to CPU EP has been explicitly "
    "disabled by the user."
)


def _value(name, dtype, shape):
    if shape is None:
        return ir.Value(name=name, type=ir.TensorType(dtype))
    return m.tensor(name, dtype, shape)


def _model(op, dtype, shape, attributes, opset=9):
    attrs = [
        ir.AttrFloat32s(key, value) if key == "bias" else ir.AttrFloat32(key, value)
        for key, value in attributes.items()
    ]
    x, y = [_value(name, dtype, shape) for name in ("x", "y")]
    node = ir.Node("", op, [x], attrs, outputs=[y])
    graph = ir.Graph([x], [y], nodes=[node], opset_imports={"": opset})
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


@contextmanager
def _strict(model):
    with TemporaryDirectory(prefix="legacy_mlx_profile_") as directory:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(directory) / "execution")
        session = ort.InferenceSession(
            model, options, providers=["MLXExecutionProvider"], enable_fallback=False,
        )
        session.disable_fallback()
        # ORT implicitly registers CPU even when assigning nodes to it is forbidden.
        assert session.get_providers() in (
            ["MLXExecutionProvider"], ["MLXExecutionProvider", "CPUExecutionProvider"],
        )
        try:
            yield session
        finally:
            profile = Path(session.end_profiling())
        kernels = [
            event for event in json.loads(profile.read_text())
            if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time")
        ]
        assert kernels, "No actual compute events in the strict MLX session"
        assert all(
            event.get("args", {}).get("provider") == "MLXExecutionProvider"
            for event in kernels
        ), kernels


def _assert_not_claimed(model):
    with pytest.raises(Fail) as error:
        with _strict(model):
            pytest.fail("Malformed legacy operator was unexpectedly claimed")
    assert type(error.value) is Fail
    assert str(error.value) == CPU_ASSIGNMENT_DISABLED


def _softplus_reference(x, alpha, beta):
    work = np.float64 if x.dtype == np.float64 else np.float32
    a, b = work(np.float32(alpha)), work(np.float32(beta))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        z = b * x.astype(work)
        y = a * np.logaddexp(work(0), z)
        rescued = (z == np.inf) & np.isfinite(x) & np.isfinite(a) & np.isfinite(b)
        tail = (np.float64(np.float32(alpha)) * np.float64(np.float32(beta))) * x.astype(np.float64)
        y = np.where(rescued, tail, y)
        return y.astype(x.dtype)


def _image_reference(x, bias, scale=1.0):
    work = np.float64 if x.dtype == np.float64 else np.float32
    values = np.asarray(bias, np.float32).astype(work).reshape(1, -1, 1, 1)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return (x.astype(work) * work(np.float32(scale)) + values).astype(x.dtype)


def _check(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    rtol, atol = {
        np.dtype(np.float16): (1e-3, 1e-4),
        np.dtype(np.float32): (3e-6, 2e-7),
        np.dtype(np.float64): (3e-14, 2e-15),
    }[actual.dtype]
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    zero = expected == 0
    np.testing.assert_array_equal(np.signbit(actual[zero]), np.signbit(expected[zero]))


@pytest.mark.parametrize("dtype,np_dtype", FLOATS)
@pytest.mark.parametrize("opset", [1, 9])
@pytest.mark.parametrize("shape", [(), (7,), (2, 1, 7), (0, 7)])
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (-2.0, -0.5), (0.0, 0.0), (1.25, -0.0)])
@pytest.mark.parametrize("eager", [False, True], ids=["default", "forced-eager"])
def test_parametric_softplus_forms(dtype, np_dtype, opset, shape, alpha, beta, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _model("ParametricSoftplus", dtype, shape, {"alpha": alpha, "beta": beta}, opset)
    count = int(np.prod(shape))
    x = np.linspace(-8, 8, count, dtype=np_dtype).reshape(shape)
    with _strict(model) as session:
        for data in [x, -x, x]:
            data = np.asarray(data, dtype=np_dtype)
            before = data.tobytes()
            _check(session.run(None, {"x": data})[0], _softplus_reference(data, alpha, beta))
            assert data.tobytes() == before


@pytest.mark.parametrize("dtype,np_dtype", FLOATS)
@pytest.mark.parametrize("opset", [1, 9])
@pytest.mark.parametrize("shape", [(2, 3, 2, 4), (0, 3, 2, 4), (2, 0, 2, 4), (2, 3, 0, 4), (2, 3, 2, 0)])
@pytest.mark.parametrize("scale", [None, 1.0, 0.0, -1.25])
@pytest.mark.parametrize("eager", [False, True], ids=["default", "forced-eager"])
def test_image_scaler_forms(dtype, np_dtype, opset, shape, scale, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    bias = [0.25, -2.0, 3.5][:shape[1]]
    attrs = {"bias": bias}
    if scale is not None:
        attrs["scale"] = scale
    x = np.arange(int(np.prod(shape)), dtype=np_dtype).reshape(shape)
    with _strict(_model("ImageScaler", dtype, shape, attrs, opset)) as session:
        for data in [x, -x, x]:
            before = data.tobytes()
            _check(
                session.run(None, {"x": data})[0],
                _image_reference(data, bias, 1.0 if scale is None else scale),
            )
            assert data.tobytes() == before


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("dtype,np_dtype", FLOATS)
def test_legacy_live_unknown_rank(dtype, np_dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    with _strict(_model("ParametricSoftplus", dtype, None, {"alpha": 1.25, "beta": -0.75})) as soft:
        for shape in [(), (7,), (2, 1, 7), (0, 7), (7,)]:
            x = np.arange(int(np.prod(shape)), dtype=np_dtype).reshape(shape)
            for data in [x, -x]:
                data = np.asarray(data, dtype=np_dtype)
                _check(soft.run(None, {"x": data})[0], _softplus_reference(data, 1.25, -0.75))
    with _strict(_model("ImageScaler", dtype, None, {"bias": [0.25, -2.0]})) as image:
        for shape in [(1, 2, 2, 3), (3, 2, 1, 2), (0, 2, 3, 0), (1, 2, 2, 3)]:
            x = np.arange(int(np.prod(shape)), dtype=np_dtype).reshape(shape)
            _check(image.run(None, {"x": x})[0], _image_reference(x, [0.25, -2.0]))
        for shape, reason in [((1, 3, 2, 3), "bias.*runtime channel"), ((2, 3), "rank four")]:
            with pytest.raises(EPFail, match=reason):
                image.run(None, {"x": np.zeros(shape, np_dtype)})
        x = np.zeros((1, 2, 2, 3), np_dtype)
        _check(image.run(None, {"x": x})[0], _image_reference(x, [0.25, -2.0]))


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("dtype,np_dtype,exponent", [(DT.FLOAT, np.float32, 20), (DT.DOUBLE, np.float64, 1000)])
def test_parametric_softplus_rescued_positive_overflow(dtype, np_dtype, exponent, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    alpha, beta = 2.0**-120, 2.0**120
    x = np.array([2.0**exponent, -(2.0**exponent)], np_dtype)
    with _strict(_model("ParametricSoftplus", dtype, [2], {"alpha": alpha, "beta": beta})) as session:
        for data in [x, -x]:
            expected = np.maximum(data, np_dtype(0))
            np.testing.assert_array_equal(session.run(None, {"x": data})[0], expected)


@pytest.mark.parametrize("dtype,np_dtype", [(DT.FLOAT, np.float32), (DT.DOUBLE, np.float64)])
@pytest.mark.parametrize("beta", [2.0, -2.0])
@pytest.mark.parametrize("eager", [False, True])
def test_parametric_softplus_rescued_finite_maximum(dtype, np_dtype, beta, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    maximum = np.finfo(np_dtype).max
    x = np.array([maximum, -maximum], np_dtype)
    with _strict(_model(
        "ParametricSoftplus", dtype, [2], {"alpha": 0.5, "beta": beta},
    )) as session:
        for data in [x, -x]:
            expected = np.maximum(data if beta > 0 else -data, np_dtype(0))
            actual = session.run(None, {"x": data})[0]
            assert np.isfinite(actual).all()
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype,np_dtype,value", [(DT.FLOAT, np.float32, -150.0), (DT.DOUBLE, np.float64, -750.0)])
@pytest.mark.parametrize("eager", [False, True])
def test_parametric_softplus_rescued_negative_tail(dtype, np_dtype, value, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    alpha = 2.0**120
    x = np.array([value, value - 1], np_dtype)
    with localcontext() as context:
        context.prec = 100
        # The log1p correction here is < 1e-60 relative, well below either output's precision.
        expected = np.array([
            float(Decimal.from_float(alpha) * Decimal.from_float(float(v)).exp()) for v in x
        ], np_dtype)
    assert np.isfinite(expected).all() and (expected > 0).all()
    with _strict(_model("ParametricSoftplus", dtype, [2], {"alpha": alpha, "beta": 1.0})) as session:
        for data in [x, x[::-1].copy(), x]:
            actual = session.run(None, {"x": data})[0]
            reference = expected if data[0] == x[0] else expected[::-1]
            np.testing.assert_allclose(actual, reference, rtol=3e-6 if np_dtype == np.float32 else 3e-14, atol=0)


@pytest.mark.parametrize("eager", [False, True])
def test_legacy_attribute_precision_and_subnormals(eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    with _strict(_model(
        "ParametricSoftplus", DT.FLOAT16, [2], {"alpha": 2.0**-30, "beta": 2.0**30},
    )) as soft, _strict(_model(
        "ImageScaler", DT.FLOAT, [1, 1, 1, 2], {"scale": 2.0**-133, "bias": [0.0]},
    )) as image, _strict(_model(
        "ImageScaler", DT.FLOAT16, [1, 1, 1, 1], {"bias": [2.0**-20]},
    )) as half_bias:
        for sign in [1, -1, 1]:
            x = sign * np.array([1024.0, -1024.0], np.float16)
            np.testing.assert_array_equal(soft.run(None, {"x": x})[0], np.maximum(x, np.float16(0)))
            x = sign * np.array([2.0**126, -(2.0**126)], np.float32).reshape(1, 1, 1, 2)
            np.testing.assert_array_equal(
                image.run(None, {"x": x})[0],
                sign * np.array([0.0078125, -0.0078125], np.float32).reshape(x.shape),
            )
        np.testing.assert_array_equal(
            half_bias.run(None, {"x": np.zeros((1, 1, 1, 1), np.float16)})[0],
            np.full((1, 1, 1, 1), 2.0**-20, np.float16),
        )


@pytest.mark.parametrize("dtype,np_dtype", FLOATS)
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (-1.0, -1.0), (0.0, 1.0), (1.0, 0.0)])
def test_parametric_softplus_nonfinite_inputs(dtype, np_dtype, alpha, beta):
    x = np.array([-np.inf, -0.0, 0.0, np.inf, np.nan], np_dtype)
    with _strict(_model("ParametricSoftplus", dtype, [5], {"alpha": alpha, "beta": beta})) as session:
        _check(session.run(None, {"x": x})[0], _softplus_reference(x, alpha, beta))


@pytest.mark.parametrize("dtype,np_dtype", FLOATS)
@pytest.mark.parametrize("alpha,beta", [
    (np.nan, 1.0), (np.inf, -1.0), (-np.inf, 0.0),
    (1.0, np.nan), (1.0, np.inf), (1.0, -np.inf),
])
def test_parametric_softplus_nonfinite_attributes(dtype, np_dtype, alpha, beta):
    x = np.array([-1.0, -0.0, 0.0, 1.0], np_dtype)
    with _strict(_model("ParametricSoftplus", dtype, [4], {"alpha": alpha, "beta": beta})) as session:
        _check(session.run(None, {"x": x})[0], _softplus_reference(x, alpha, beta))


@pytest.mark.parametrize("dtype,np_dtype", FLOATS)
@pytest.mark.parametrize("scale", [0.0, np.inf, np.nan])
def test_image_scaler_nonfinite_inputs_and_attributes(dtype, np_dtype, scale):
    x = np.array([-np.inf, -0.0, 0.0, np.inf, np.nan] * 2, np_dtype).reshape(1, 2, 1, 5)
    bias = [np.inf, np.nan]
    with _strict(_model("ImageScaler", dtype, list(x.shape), {"scale": scale, "bias": bias})) as session:
        _check(session.run(None, {"x": x})[0], _image_reference(x, bias, scale))


def test_parametric_softplus_double_transcendental_precision():
    x = np.array([-1.123456789012345, -0.123456789012345, 0.333333333333333], np.float64)
    with _strict(_model("ParametricSoftplus", DT.DOUBLE, [3], {"alpha": 1.25, "beta": 0.75})) as session:
        _check(session.run(None, {"x": x})[0], _softplus_reference(x, 1.25, 0.75))


def test_image_scaler_separate_multiply_add_rounding():
    scale = float(np.float32(1 + 2.0**-23))
    bias = -float(np.float32(1 + 2.0**-22))
    x = np.full((1, 1, 1, 1), scale, np.float32)
    with _strict(_model("ImageScaler", DT.FLOAT, list(x.shape), {"scale": scale, "bias": [bias]})) as session:
        np.testing.assert_array_equal(session.run(None, {"x": x})[0], np.zeros_like(x))


@pytest.mark.parametrize("op,attrs", [
    ("ParametricSoftplus", {"alpha": 1.25, "beta": -0.75}),
    ("ImageScaler", {"scale": -1.25, "bias": [0.25, -2.0]}),
])
def test_legacy_cpu_float32_control(op, attrs):
    x = np.linspace(-4, 4, 12, dtype=np.float32).reshape(1, 2, 2, 3)
    model = _model(op, DT.FLOAT, list(x.shape), attrs)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    cpu = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"], enable_fallback=False)
    cpu.disable_fallback()
    expected = _softplus_reference(x, **attrs) if op == "ParametricSoftplus" else _image_reference(x, **attrs)
    _check(cpu.run(None, {"x": x})[0], expected)
    with _strict(model) as session:
        _check(session.run(None, {"x": x})[0], expected)


@pytest.mark.parametrize("op,shape,attrs", [
    ("ParametricSoftplus", [2], {}),
    ("ParametricSoftplus", [2], {"alpha": 1.0}),
    ("ParametricSoftplus", [2], {"beta": 1.0}),
    ("ImageScaler", [1, 2, 1, 1], {}),
    ("ImageScaler", [1, 2, 1, 1], {"bias": []}),
    ("ImageScaler", [1, 2, 1, 1], {"bias": [1.0]}),
    ("ImageScaler", [1, 2, 1], {"bias": [1.0, 2.0]}),
    ("ImageScaler", [1, 2, 1, 1, 1], {"bias": [1.0, 2.0]}),
])
def test_legacy_invalid_forms_not_claimed(op, shape, attrs):
    _assert_not_claimed(_model(op, DT.FLOAT, shape, attrs))


@pytest.mark.parametrize("op,attrs", [
    ("ParametricSoftplus", {"alpha": 1.0, "beta": 1.0}),
    ("ImageScaler", {"bias": [0.0]}),
])
@pytest.mark.parametrize("opset", [10, 26])
def test_legacy_deprecated_schema_loader_rejection(op, attrs, opset):
    model = _model(op, DT.FLOAT, [1, 1, 1, 1], attrs, opset)
    expected = (
        f"[ONNXRuntimeError] : 1 : FAIL : Fatal error: ai.onnx:{op}(10) "
        "is not a registered function/op"
    )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    with pytest.raises(Fail) as cpu_error:
        ort.InferenceSession(
            model, options, providers=["CPUExecutionProvider"], enable_fallback=False,
        )
    assert type(cpu_error.value) is Fail
    assert str(cpu_error.value) == expected
    with pytest.raises(Fail) as mlx_error:
        with _strict(model):
            pytest.fail("Deprecated schema was unexpectedly accepted")
    assert type(mlx_error.value) is Fail
    assert str(mlx_error.value) == expected


@pytest.mark.parametrize("dtype", [DT.BFLOAT16, DT.INT32, DT.BOOL, DT.COMPLEX64, DT.STRING])
@pytest.mark.parametrize("op,attrs", [
    ("ParametricSoftplus", {"alpha": 1.0, "beta": 1.0}),
    ("ImageScaler", {"bias": [0.0]}),
])
def test_legacy_schema_illegal_tensor_types(dtype, op, attrs):
    with pytest.raises(InvalidGraph, match="Type|type"):
        with _strict(_model(op, dtype, [1, 1, 1, 1], attrs)):
            pytest.fail("Schema-illegal tensor type was unexpectedly accepted")


@pytest.mark.parametrize("op,attrs,wrong", [
    ("ParametricSoftplus", [ir.AttrInt64("alpha", 1), ir.AttrFloat32("beta", 1.0)], "alpha"),
    ("ParametricSoftplus", [ir.AttrFloat32("alpha", 1.0), ir.AttrInt64("beta", 1)], "beta"),
    ("ImageScaler", [ir.AttrFloat32("bias", 0.0)], "bias"),
    ("ImageScaler", [ir.AttrFloat32s("bias", [0.0]), ir.AttrInt64("scale", 1)], "scale"),
])
def test_legacy_wrong_attribute_types(op, attrs, wrong):
    x, y = [m.tensor(name, DT.FLOAT, [1, 1, 1, 1]) for name in ("x", "y")]
    graph = ir.Graph([x], [y], nodes=[ir.Node("", op, [x], attrs, outputs=[y])], opset_imports={"": 9})
    proto = ir.to_proto(ir.Model(graph, ir_version=11))
    attribute = next(attr for attr in proto.graph.node[0].attribute if attr.name == wrong)
    assert attribute.type == (AttributeProto.FLOAT if wrong == "bias" else AttributeProto.INT)
    # Experimental schema attributes can reach EP claiming instead of loader type rejection.
    _assert_not_claimed(proto.SerializeToString())
    legal = {"alpha": 1.0, "beta": 1.0} if op == "ParametricSoftplus" else {"bias": [0.0], "scale": 1.0}
    data = np.full((1, 1, 1, 1), 1.25, np.float32)
    expected = _softplus_reference(data, **legal) if op == "ParametricSoftplus" else _image_reference(data, **legal)
    with _strict(_model(op, DT.FLOAT, list(data.shape), legal)) as session:
        _check(session.run(None, {"x": data})[0], expected)


def _route_replay(op):
    shape = ["N", 2, 1, 3]
    x, offset, factor = [m.tensor(name, DT.FLOAT, shape) for name in ("x", "offset", "factor")]
    added, middle, y = [m.tensor(name, DT.FLOAT, shape) for name in ("added", "middle", "y")]
    attrs = {"alpha": 1.25, "beta": -0.75}
    nodes = [ir.node("Add", [x, offset], outputs=[added])]
    inputs = [x, offset, factor]
    if op == "If":
        condition = m.tensor("condition", DT.BOOL, [])
        inputs.append(condition)
        then_y, else_y = [m.tensor(name, DT.FLOAT, shape) for name in ("then_y", "else_y")]
        then_graph = ir.Graph([], [then_y], nodes=[
            ir.node("ParametricSoftplus", [added], attributes=attrs, outputs=[then_y]),
        ], name="legacy_then")
        else_graph = ir.Graph([], [else_y], nodes=[
            ir.node("Identity", [added], outputs=[else_y]),
        ], name="legacy_else")
        nodes.append(ir.node("If", [condition], attributes={
            "then_branch": then_graph, "else_branch": else_graph,
        }, outputs=[middle]))
    else:
        if op == "ImageScaler":
            attrs = {"scale": -1.25, "bias": [0.25, -2.0]}
        nodes.append(ir.node(op, [added], attributes=attrs, outputs=[middle]))
    nodes.append(ir.node("Mul", [middle, factor], outputs=[y]))
    graph = ir.Graph(inputs, [y], nodes=nodes, opset_imports={"": 9})
    with _strict(ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()) as session:
        for index, batch in enumerate([1, 2, 1]):
            data = np.arange(batch * 6, dtype=np.float32).reshape(batch, 2, 1, 3) - index
            feeds = {"x": data, "offset": np.full_like(data, 0.25), "factor": np.full_like(data, -0.5)}
            value = data + feeds["offset"]
            if op == "If":
                feeds["condition"] = np.array(index != 1, np.bool_)
                expected = _softplus_reference(value, **attrs) if index != 1 else value
            else:
                expected = _softplus_reference(value, **attrs) if op == "ParametricSoftplus" else _image_reference(value, **attrs)
            _check(session.run(None, feeds)[0], expected * feeds["factor"])


@pytest.mark.parametrize("op", ["ParametricSoftplus", "ImageScaler", "If"])
def test_legacy_actual_eager_boundary_with_compiled_neighbors(op, tmp_path):
    trace = tmp_path / "legacy_route.json"
    subprocess.run(
        [sys.executable, "-c",
         "import os, onnxruntime as ort; "
         "ort.register_execution_provider_library('MLXExecutionProvider', os.environ['ONNXRUNTIME_MLX_EP_LIB']); "
         f"from test_legacy_parametric_image import _route_replay; _route_replay({op!r})"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0", "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    events = [event["args"] for event in json.loads(trace.read_text()) if event.get("cat") == "ep.path"]
    paths = [item["path"] for item in events if op in item["ops"].split(",")]
    assert paths == ["eager"] * 3, paths
    for neighbor in ["Add", "Mul"]:
        paths = [item["path"] for item in events if neighbor in item["ops"].split(",")]
        assert paths == ["general"] * 3, paths
