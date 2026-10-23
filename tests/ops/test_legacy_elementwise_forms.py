"""Legacy ORT schema-1 elementwise operators omitted from modern ONNX catalogs."""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT
from onnxruntime.capi.onnxruntime_pybind11_state import Fail


CASES = [
    ("Affine", {}),
    ("Affine", {"alpha": 1.25}),
    ("Affine", {"beta": -0.75}),
    ("Affine", {"alpha": -2.0, "beta": 0.5}),
    ("Scale", {}),
    ("Scale", {"scale": -1.25}),
    ("Scale", {"scale": 0.0}),
    ("ScaledTanh", {"alpha": 1.25, "beta": 0.75}),
    ("ScaledTanh", {"alpha": -2.0, "beta": -0.5}),
    ("ScaledTanh", {"alpha": 0.5, "beta": 0.0}),
]
DTYPES = [(DT.FLOAT16, np.float16), (DT.FLOAT, np.float32), (DT.DOUBLE, np.float64)]


def _model(op, dtype, shape, attributes, *, opset=7):
    x = ir.Value(name="x", type=ir.TensorType(dtype),
                 shape=None if shape is None else ir.Shape(shape))
    y = ir.Value(name="y", type=ir.TensorType(dtype),
                 shape=None if shape is None else ir.Shape(shape))
    node = ir.node(op, [x], attributes=attributes, outputs=[y], name="legacy_elementwise")
    graph = ir.Graph([x], [y], nodes=[node], name="legacy_elementwise",
                     opset_imports={"": opset})
    return ir.to_proto(ir.Model(graph, ir_version=9)).SerializeToString()


def _session(model, *, cpu=False):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    if not cpu:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model, options, enable_fallback=False,
        providers=["CPUExecutionProvider"] if cpu else ["MLXExecutionProvider"],
    )
    session.disable_fallback()
    return session


def _reference(op, x, attributes):
    work = x.astype(np.float64 if x.dtype == np.float64 else np.float32)

    def attr(name, default):
        return np.asarray(np.float32(attributes.get(name, default)), dtype=work.dtype)

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        if op == "Affine":
            result = work * attr("alpha", 1.0) + attr("beta", 0.0)
        elif op == "Scale":
            result = work * attr("scale", 1.0)
        else:
            result = np.float32(attributes["alpha"]) * np.tanh(work * np.float32(attributes["beta"]))
        return result.astype(x.dtype)


def _assert_result(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    tolerance = {np.dtype("float16"): 1e-3, np.dtype("float32"): 2e-6,
                 np.dtype("float64"): 1e-12}[expected.dtype]
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True)


@pytest.fixture(params=[False, True], ids=["default", "eager"])
def mode(request, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if request.param else "0")


@pytest.mark.parametrize("op,attributes", CASES)
@pytest.mark.parametrize("dtype,numpy_dtype", DTYPES)
@pytest.mark.parametrize("shape", [(), (7,), (2, 3), (2, 0, 3)])
def test_legacy_elementwise_forms(mode, op, attributes, dtype, numpy_dtype, shape):
    session = _session(_model(op, dtype, shape, attributes))
    for offset in (-0.75, 0.25):
        data = (np.arange(math.prod(shape)).reshape(shape) * 0.125 + offset).astype(numpy_dtype)
        before = data.tobytes()
        (actual,) = session.run(None, {"x": data})
        _assert_result(actual, _reference(op, data, attributes))
        assert data.tobytes() == before


@pytest.mark.parametrize("op,attributes", [CASES[3], CASES[5], CASES[7]])
@pytest.mark.parametrize("dtype,numpy_dtype", DTYPES)
def test_legacy_unknown_rank_replay(mode, op, attributes, dtype, numpy_dtype):
    session = _session(_model(op, dtype, None, attributes))
    for shape in [(2, 3), (0, 5), (), (3,), (2, 3)]:
        data = (np.arange(math.prod(shape)).reshape(shape) * 0.25 - 0.5).astype(numpy_dtype)
        before = data.tobytes()
        (actual,) = session.run(None, {"x": data})
        _assert_result(actual, _reference(op, data, attributes))
        assert data.tobytes() == before


@pytest.mark.parametrize("op,attributes", [CASES[3], CASES[5], CASES[7]])
def test_legacy_cpu_positive_control(op, attributes):
    data = np.asarray([[-2.0, -0.0, 0.0], [0.25, 1.0, 3.0]], dtype=np.float32)
    model = _model(op, DT.FLOAT, data.shape, attributes)
    (expected,) = _session(model, cpu=True).run(None, {"x": data})
    (actual,) = _session(model).run(None, {"x": data})
    _assert_result(actual, expected)


@pytest.mark.parametrize("op,attributes,data", [
    ("Affine", {"alpha": 3e38, "beta": 0.25}, [0.0, -0.0]),
    ("Scale", {"scale": 3e38}, [0.0, -0.0]),
    ("Scale", {"scale": 1e-10}, [60000.0, -60000.0]),
    ("ScaledTanh", {"alpha": 30000.0, "beta": 1e-10}, [60000.0, -60000.0]),
])
def test_legacy_half_attributes_keep_float_precision(mode, op, attributes, data):
    data = np.asarray(data, dtype=np.float16)
    (actual,) = _session(_model(op, DT.FLOAT16, data.shape, attributes)).run(None, {"x": data})
    expected = _reference(op, data, attributes)
    _assert_result(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


@pytest.mark.parametrize("op,attributes", [CASES[3], CASES[5], CASES[7]])
@pytest.mark.parametrize("dtype,numpy_dtype", DTYPES)
def test_legacy_nonfinite_and_signed_zero(mode, op, attributes, dtype, numpy_dtype):
    data = np.asarray([-np.inf, -0.0, 0.0, np.inf, np.nan], dtype=numpy_dtype)
    (actual,) = _session(_model(op, dtype, data.shape, attributes)).run(None, {"x": data})
    expected = _reference(op, data, attributes)
    _assert_result(actual, expected)
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(np.signbit(actual[finite]), np.signbit(expected[finite]))


@pytest.mark.parametrize("attributes", [{}, {"alpha": 1.0}, {"beta": 1.0}])
def test_scaled_tanh_undefined_defaults_are_not_invented(attributes):
    with pytest.raises(Fail, match="fallback|Fallback|not.*assigned|not.*supported"):
        _session(_model("ScaledTanh", DT.FLOAT, (3,), attributes))
