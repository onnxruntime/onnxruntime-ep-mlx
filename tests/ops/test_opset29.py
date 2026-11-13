"""ONNX-main contracts; runtime tests skip only when ORT lacks opset-29 schemas."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from test_metadata_only_shape import _session


def _model(op, feeds, attrs=None, output_names=("y",)):
    arrays = {name: numpy_helper.from_array(value) for name, value in feeds.items()}
    graph = helper.make_graph(
        [helper.make_node(op, list(feeds), list(output_names), **(attrs or {}))],
        "opset29",
        [helper.make_tensor_value_info(name, arrays[name].data_type, list(value.shape))
         for name, value in feeds.items()],
        [helper.make_tensor_value_info(name, arrays[next(iter(feeds))].data_type, None)
         for name in output_names],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 29)], ir_version=11,
    )


def _cases():
    yield "Expm1", {"x": np.array([-0.0, 1e-8, -1e-8, -np.inf, np.inf, np.nan], np.float32)}, {}, (
        np.array([-0.0, 1e-8, -1e-8, -1, np.inf, np.nan], np.float32),
    )
    yield "Log1p", {"x": np.array([-0.0, 1e-8, -1, -2, np.inf, np.nan], np.float32)}, {}, (
        np.array([-0.0, 1e-8, -np.inf, np.nan, np.inf, np.nan], np.float32),
    )
    yield "LogCumSumExp", {"x": np.array([-1000, -1001, 1000, 1001], np.float32)}, {"axis": -1}, (
        np.logaddexp.accumulate(np.array([-1000, -1001, 1000, 1001], np.float64)).astype(np.float32),
    )
    yield "MeanVarianceNormalization", {"x": np.array([10000, 10002], np.float32)}, {
        "axes": [0], "epsilon": 0.5,
    }, (np.array([-2 / 3, 2 / 3], np.float32),)
    yield "Gemm", {
        "x": np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3),
        "w": np.array([[1, 2], [3, 4]], np.float32),
        "bias": np.array([0.5, -0.5], np.float32),
    }, {"transA": 1}, (
        np.array([13.5, 17.5, 17.5, 23.5, 21.5, 29.5,
                  37.5, 53.5, 41.5, 59.5, 45.5, 65.5], np.float32).reshape(2, 3, 2),
    )
    yield "GroupedMatMul", {
        "x": np.arange(1, 7, dtype=np.float32).reshape(2, 3),
        "w": np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3),
        "indices": np.array([[1, 0], [0, 1]], np.int32),
        "bias": np.array([[0.5, -0.5], [1, -1]], np.float32),
    }, {"transB": 1}, (
        np.array([51, 67, 14.5, 31.5, 32.5, 76.5, 123, 166], np.float32).reshape(2, 2, 2),
    )
    yield "CausalConvWithState", {
        "x": np.array([[[1, 2, 3]]], np.float32),
        "w": np.array([[[1, 10]]], np.float32),
        "bias": np.zeros(1, np.float32),
        "past": np.array([[[4, 5]]], np.float32),
    }, {"dilation": 2}, (
        np.array([[[14, 25, 31]]], np.float32), np.array([[[2, 3]]], np.float32),
    )
    yield "GatherND", {
        "x": np.array([[1, 2], [3, 4]], np.float32),
        "indices": np.array([[0], [-1]], np.int32),
    }, {}, (np.array([[1, 2], [3, 4]], np.float32),)
    yield "ScatterND", {
        "x": np.array([[1, 2], [3, 4]], np.float32),
        "indices": np.array([[0], [0]], np.int32),
        "updates": np.array([[3, 4], [5, 6]], np.float32),
    }, {"reduction": "mean"}, (np.array([[3, 4], [3, 4]], np.float32),)
    yield "ScatterElements", {
        "x": np.array([[1, 2, 3], [4, 5, 6]], np.float32),
        "indices": np.array([[0, -1, 0, 1]], np.int32),
        "updates": np.array([[3, 7, 5, 6]], np.float32),
    }, {"axis": -1, "reduction": "mean"}, (np.array([[3, 4, 5], [4, 5, 6]], np.float32),)


CASES = list(_cases())


@pytest.mark.parametrize("op,feeds,attrs,expected", CASES, ids=[case[0] for case in CASES])
def test_opset29_native_goldens_match_onnx_reference(op, feeds, attrs, expected):
    if onnx.defs.onnx_opset_version() < 29:
        pytest.skip("ONNX reference schemas stop before opset 29; install ONNX from main")
    outputs = ("y", "state") if op == "CausalConvWithState" else ("y",)
    model = _model(op, feeds, attrs, outputs)
    actual = ReferenceEvaluator(model).run(None, feeds)
    for result, golden in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, golden, rtol=2e-6, atol=1e-7, equal_nan=True)
        if op in {"Expm1", "Log1p"}:
            assert np.signbit(result[0])


@pytest.fixture(scope="module")
def runtime_opset29():
    feeds = {"x": np.array([1e-8], np.float32)}
    try:
        ort.InferenceSession(_model("Expm1", feeds).SerializeToString(),
                             providers=["CPUExecutionProvider"])
    except (ort.capi.onnxruntime_pybind11_state.Fail,
            ort.capi.onnxruntime_pybind11_state.InvalidGraph) as error:
        message = str(error)
        if ("Opset 29 is under development" in message
                or "No Op registered for Expm1" in message):
            pytest.skip(f"ORT {ort.__version__} lacks opset-29 schema loading: {message}")
        raise


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("op,feeds,attrs,expected", CASES, ids=[case[0] for case in CASES])
def test_opset29_mlx_against_reference(runtime_opset29, op, feeds, attrs, expected,
                                     eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    outputs = ("y", "state") if op == "CausalConvWithState" else ("y",)
    actual = _session(_model(op, feeds, attrs, outputs).SerializeToString()).run(None, feeds)
    for result, golden in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, golden, rtol=2e-5, atol=1e-6, equal_nan=True)
