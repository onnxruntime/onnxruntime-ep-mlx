"""CDist v1: independent distances, strict provider ownership, and live eager inputs."""

import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import TensorProto as T, helper, numpy_helper
import pytest


DTYPES = [np.float32, np.float64]
METRICS = [None, "sqeuclidean", "euclidean"]
ERRORS = (ort.capi.onnxruntime_pybind11_state.InvalidGraph,
          ort.capi.onnxruntime_pybind11_state.InvalidArgument,
          ort.capi.onnxruntime_pybind11_state.Fail)


def _model(dtype, shapes, metric=None, *, constant=None, crossing=False):
    tensor_type = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    inputs, nodes, names, initializers = [], [], [], []
    for name, shape in zip(["a", "b"], shapes, strict=True):
        if name == "b" and constant is not None:
            initializers.append(numpy_helper.from_array(constant, name))
        else:
            inputs.append(helper.make_tensor_value_info(name, tensor_type, shape))
        if crossing:
            nodes.append(helper.make_node("Identity", [name], [name + "_live"]))
            names.append(name + "_live")
        else:
            names.append(name)
    attrs = {} if metric is None else {"metric": metric}
    nodes.append(helper.make_node("CDist", names, ["y"], domain="com.microsoft",
                                  name="distance_target", **attrs))
    output_shape = [shapes[0][0], shapes[1][0]] if all(len(s) == 2 for s in shapes) else None
    return helper.make_model(
        helper.make_graph(nodes, "cdist", inputs, [
            helper.make_tensor_value_info("y", tensor_type, output_shape)],
            initializer=initializers),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10)


def _session(model, path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.enable_profiling = True
    options.profile_file_prefix = str(path)
    session = ort.InferenceSession(model.SerializeToString(), options,
                                   providers=["MLXExecutionProvider"], enable_fallback=False)
    session.disable_fallback()
    providers = session.get_providers()
    assert providers in (["MLXExecutionProvider"],
                         ["MLXExecutionProvider", "CPUExecutionProvider"]), providers
    return session


def _placement(session):
    events = json.loads(Path(session.end_profiling()).read_text())
    kernels = [e for e in events
               if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")]
    assert kernels, "expected actual node kernel events"
    assert all(e.get("args", {}).get("provider") == "MLXExecutionProvider"
               for e in kernels), kernels


def _reference(a, b, metric):
    # math.dist uses a scaled norm, unlike the lowering's typed square/sum reduction.
    distances = [[math.dist(row_a, row_b) for row_b in b] for row_a in a]
    result = np.asarray(distances, np.float64).reshape(a.shape[0], b.shape[0])
    if metric != "euclidean":
        result = np.square(result)
    return result.astype(a.dtype)


def _compare(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=4 * np.finfo(actual.dtype).eps,
                               atol=0, equal_nan=True)
    zeros = expected == 0
    np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("source", ["input", "intermediate", "initializer"])
def test_cdist_live_values_and_sources(dtype, metric, eager, source, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    b = np.asarray([[1, 2, 3], [-2, 0, 4], [0, 0, 0]], dtype)
    session = _session(_model(dtype, [[2, 3], [3, 3]], metric,
                              constant=b if source == "initializer" else None,
                              crossing=source == "intermediate"), tmp_path / "live")
    for sign in [1, -1, 1]:
        a = np.asarray([[0, 2, -3], [4, 1, 2]], dtype) * sign
        live_b = b if source == "initializer" else b * sign
        feeds = {"a": a}
        if source != "initializer":
            feeds["b"] = live_b
        _compare(session.run(None, feeds)[0], _reference(a, live_b, metric))
    _placement(session)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("eager", [False, True])
def test_cdist_symbolic_geometry_empty_and_recovery(dtype, metric, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(dtype, [[None, None], [None, None]], metric), tmp_path / "shape")
    for m, k, n in [(2, 3, 4), (0, 3, 4), (2, 0, 4), (2, 3, 0), (0, 0, 0), (1, 2, 1)]:
        a = np.arange(m * n, dtype=dtype).reshape(m, n)
        b = -np.arange(k * n, dtype=dtype).reshape(k, n)
        _compare(session.run(None, {"a": a, "b": b})[0], _reference(a, b, metric))
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.EPFail,
                       match="CDist: A and B feature dimensions must match"):
        session.run(None, {"a": np.ones((2, 3), dtype), "b": np.ones((4, 2), dtype)})
    a, b = np.ones((2, 3), dtype), np.zeros((4, 3), dtype)
    _compare(session.run(None, {"a": a, "b": b})[0], _reference(a, b, metric))
    _placement(session)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("metric", ["sqeuclidean", "euclidean"])
def test_cdist_nearby_large_vectors_and_genuine_double(dtype, metric, tmp_path):
    large = 2.0 ** (40 if dtype == np.float64 else 20)
    a = np.asarray([[large, large, large]], dtype)
    b = np.asarray([[large + 1, large + 2, large - 1],
                    [large + 0.125, large, large]], dtype)
    session = _session(_model(dtype, [[1, 3], [2, 3]], metric), tmp_path / "precision")
    expected = np.asarray([[6, 0.015625]], dtype)
    if metric == "euclidean":
        expected = np.sqrt(expected)
    _compare(session.run(None, {"a": a, "b": b})[0], expected)
    _compare(session.run(None, {"a": a, "b": a.repeat(2, axis=0)})[0],
             np.zeros((1, 2), dtype))
    _placement(session)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("metric", ["sqeuclidean", "euclidean"])
def test_cdist_typed_overflow_and_nonfinite(dtype, metric, tmp_path):
    session = _session(_model(dtype, [[1, 1], [3, 1]], metric), tmp_path / "nonfinite")
    # IEEE typed squares overflow even when a scaled mathematical norm would not.
    huge = np.finfo(dtype).max / 4
    for a, b, expected in [
        ([[huge]], [[0], [huge], [-huge]], [[np.inf, 0, np.inf]]),
        ([[np.inf]], [[0], [np.inf], [-np.inf]], [[np.inf, np.nan, np.inf]]),
        ([[np.nan]], [[0], [1], [np.inf]], [[np.nan, np.nan, np.nan]]),
        ([[0]], [[-0.0], [0], [0]], [[0, 0, 0]]),
    ]:
        _compare(session.run(None, {"a": np.asarray(a, dtype),
                                   "b": np.asarray(b, dtype)})[0],
                 np.asarray(expected, dtype))
    _placement(session)


@pytest.mark.parametrize("metric", ["", "cosine", "minkowski", "SQEUCLIDEAN", b"\xff", 1, 1.0])
def test_cdist_invalid_metric(metric, tmp_path):
    with pytest.raises(ERRORS):
        _session(_model(np.float32, [[2, 3], [4, 3]], metric), tmp_path / "metric")


@pytest.mark.parametrize("shapes", [
    [[], [2, 3]], [[3], [2, 3]], [[1, 2, 3], [2, 3]],
    [[2, 3], [3]], [[2, 3], [4, 2]],
])
def test_cdist_invalid_geometry(shapes, tmp_path):
    with pytest.raises(ERRORS):
        _session(_model(np.float32, shapes), tmp_path / "geometry")


@pytest.mark.parametrize("dtype", [np.float16, np.int32])
def test_cdist_invalid_dtype(dtype, tmp_path):
    with pytest.raises(ERRORS):
        _session(_model(dtype, [[2, 3], [4, 3]]), tmp_path / "dtype")


@pytest.mark.parametrize("variant", [
    "mixed", "bf16", "output_dtype", "output_rank",
    "missing_a", "missing_b", "missing_y", "extra_input", "extra_output",
    "no_inputs", "no_outputs", "unknown_attribute",
])
def test_cdist_exact_form_validation(variant, tmp_path):
    model = _model(np.float32, [[2, 3], [4, 3]])
    node = model.graph.node[0]
    if variant in ("mixed", "bf16"):
        model.graph.input[1].type.tensor_type.elem_type = (
            T.DOUBLE if variant == "mixed" else T.BFLOAT16)
        if variant == "bf16":
            model.graph.input[0].type.tensor_type.elem_type = T.BFLOAT16
            model.graph.output[0].type.tensor_type.elem_type = T.BFLOAT16
    elif variant == "output_dtype":
        model.graph.output[0].type.tensor_type.elem_type = T.DOUBLE
    elif variant == "output_rank":
        del model.graph.output[0].type.tensor_type.shape.dim[1]
    elif variant in ("missing_a", "missing_b"):
        node.input[variant == "missing_b"] = ""
    elif variant == "missing_y":
        node.output[0] = ""
    elif variant == "extra_input":
        node.input.append("a")
    elif variant == "extra_output":
        node.output.append("extra")
    elif variant == "no_inputs":
        del node.input[:]
    elif variant == "no_outputs":
        del node.output[:]
    else:
        node.attribute.append(helper.make_attribute("p", 2.0))
    with pytest.raises(ERRORS):
        _session(model, tmp_path / variant)


@pytest.mark.parametrize("depth", [1, 2])
def test_cdist_nested_eager_live(depth, tmp_path):
    model = _model(np.float32, [[2, 3], [4, 3]], "euclidean")
    body = helper.make_graph(list(model.graph.node), "branch", [], list(model.graph.output))
    for level in range(depth):
        branch = helper.make_node("If", ["condition"], ["y"], then_branch=body, else_branch=body)
        body = helper.make_graph([branch], f"parent_{level}", [], list(model.graph.output))
    del model.graph.node[:]
    model.graph.node.extend(body.node)
    model.graph.input.append(helper.make_tensor_value_info("condition", T.BOOL, []))
    session = _session(model, tmp_path / "nested")
    for flag, sign in [(True, 1), (False, -1), (True, 1)]:
        a = np.arange(6, dtype=np.float32).reshape(2, 3) * sign
        b = np.ones((4, 3), np.float32)
        _compare(session.run(None, {"a": a, "b": b, "condition": np.asarray(flag)})[0],
                 _reference(a, b, "euclidean"))
    _placement(session)


def test_cdist_double_body_payload_remains_rejected(tmp_path):
    model = _model(np.float64, [[2, 3], [4, 3]])
    body = helper.make_graph(list(model.graph.node), "double_branch", [],
                             list(model.graph.output))
    del model.graph.node[:]
    model.graph.node.append(helper.make_node(
        "If", ["condition"], ["y"], then_branch=body, else_branch=body))
    model.graph.input.append(helper.make_tensor_value_info("condition", T.BOOL, []))
    with pytest.raises(ERRORS):
        _session(model, tmp_path / "double_body")
