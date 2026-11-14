"""Strict QDQ product sessions, original layouts, and independently checked GPU fusion."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from _schema_runtime import session as schema_session
from onnx import TensorProto, defs, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnxruntime.capi.onnxruntime_pybind11_state import Fail


def _transposed_model(case, layout, dtype):
    rng = np.random.default_rng(42)
    m, k, n, experts, selections = 3, 67, 7, 4, 3
    grouped, trans_a = case == "Grouped", case == "GemmTransA"
    weight_shape = (experts, n, k) if grouped else (n, k)
    a_shape = (k, m) if trans_a else (m, k)
    low, high, zero = (-70, 70, -3) if dtype == np.int8 else (0, 180, 71)
    feeds = {
        "a": rng.integers(low, high, a_shape).astype(dtype),
        "w": rng.integers(low, high, weight_shape).astype(dtype),
    }
    block = 16 if layout == "blocked" else 0
    scale_shape = (*weight_shape[:-1], (k + block - 1) // block) if block else (n,)
    a_scale = np.array([0.125, 0.25, 0.5], np.float32)
    w_scale = np.resize(
        np.array([0.0625, 0.125, 0.25] if layout == "integer" else [0.03, 0.07, 0.011],
                 np.float32),
        scale_shape,
    )
    a_zero = np.full(a_scale.shape, zero, dtype)
    w_zero = rng.integers(low, high, scale_shape).astype(dtype)
    parameters = {"as": a_scale, "ws": w_scale, "az": a_zero, "wz": w_zero}
    initializers = []
    if layout == "integer":
        initializers = [numpy_helper.from_array(parameters.pop(name), name)
                        for name in ("as", "ws")]
    feeds.update(parameters)
    nodes = [
        helper.make_node("DequantizeLinear", ["a", "as", "az"], ["af"],
                         axis=1 if trans_a else 0),
        helper.make_node("DequantizeLinear", ["w", "ws", "wz"], ["wf"],
                         axis=len(weight_shape) - (1 if block else 2),
                         **({"block_size": block} if block else {})),
    ]
    activation = (
        (feeds["a"].astype(np.float32)
         - _expanded(a_zero, a_shape, 1 if trans_a else 0, 0))
        * _expanded(a_scale, a_shape, 1 if trans_a else 0, 0)
    )
    if trans_a:
        activation = activation.T
    weight = (
        (feeds["w"].astype(np.float32)
         - _expanded(w_zero, weight_shape, len(weight_shape) - (1 if block else 2), block))
        * _expanded(w_scale, weight_shape, len(weight_shape) - (1 if block else 2), block)
    )
    if grouped:
        feeds["ids"] = np.array([[3, 0, 3], [1, 1, 0], [0, 3, 1]], np.int64)
        feeds["bias"] = rng.normal(0, 0.2, (experts, n)).astype(np.float32)
        nodes.append(helper.make_node("GroupedMatMul", ["af", "wf", "ids", "bias"],
                                      ["y"], transB=1))
        expected = np.stack([
            activation[row].astype(np.float64) @ weight[ids].astype(np.float64).T
            for row, choices in enumerate(feeds["ids"]) for ids in choices
        ]).reshape(m, selections, n).astype(np.float32) + feeds["bias"][feeds["ids"]]
        shape = [m, selections, n]
    else:
        feeds["c"] = rng.normal(0, 0.2, (1, n)).astype(np.float32)
        nodes.append(helper.make_node("Gemm", ["af", "wf", "c"], ["y"],
                                      transA=int(trans_a), transB=1, alpha=0.75, beta=-0.25))
        # Preserve the MatMul output rounding before the ordered Gemm epilogue.
        product = (activation.astype(np.float64) @ weight.astype(np.float64).T).astype(np.float32)
        expected = product * np.float32(0.75) + feeds["c"] * np.float32(-0.25)
        shape = [m, n]
    nodes.append(helper.make_node("Add", ["wf", "wf"], ["shared"]))
    info = helper.make_tensor_value_info
    model = helper.make_model(helper.make_graph(
        nodes, "original_transposed_qdq_layout",
        [info(name, helper.np_dtype_to_tensor_dtype(value.dtype), list(value.shape))
         for name, value in feeds.items()],
        [info("y", TensorProto.FLOAT, shape),
         info("wf", TensorProto.FLOAT, list(weight_shape)),
         info("shared", TensorProto.FLOAT, list(weight_shape))],
        initializers,
    ), opset_imports=[helper.make_opsetid("", 29 if grouped else 24)], ir_version=11)
    return model.SerializeToString(), feeds, expected, weight


@pytest.mark.parametrize("case", ["Gemm", "GemmTransA", "Grouped"])
@pytest.mark.parametrize("layout", ["integer", "axis", "blocked"])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8], ids=["int8", "uint8"])
def test_transposed_qdq(case, layout, dtype, monkeypatch):
    model, feeds, expected, weight = _transposed_model(case, layout, dtype)
    for disabled in ["0", "1"]:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION", disabled)
        result = schema_session(model).run(None, feeds)
        np.testing.assert_allclose(result[0], expected, rtol=2e-5, atol=8e-4)
        np.testing.assert_array_equal(result[1], weight)
        np.testing.assert_array_equal(result[2], weight + weight)


def test_transposed_qdq_actual_dispatch(tmp_path):
    trace = tmp_path / "transposed-gemm.json"
    subprocess.run(
        [sys.executable, "-m", "pytest",
         f"{Path(__file__)}::test_transposed_qdq[int8-blocked-Gemm]",
         "-q", "-p", "no:cacheprovider"],
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
        check=True, timeout=180,
    )
    kernels = {event.get("args", {}).get("kernel")
               for event in json.loads(trace.read_text()) if event.get("cat") == "op.fast"}
    assert "mlx_qdq_matmul" in kernels, kernels


def _require_grouped_reference():
    if defs.onnx_opset_version() < 29:
        pytest.skip("ONNX lacks the GroupedMatMul-29 reference schema; install ONNX from main")


@pytest.mark.parametrize("case", ["Gemm", "GemmTransA", "Grouped"])
@pytest.mark.parametrize("transpose", [0, 1])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8], ids=["int8", "uint8"])
def test_weight_only_prefill(case, transpose, dtype, monkeypatch):
    if case == "Grouped":
        _require_grouped_reference()
    rng = np.random.default_rng(815)
    m, k, n, experts, selections = 17, 257, 69, 5, 3
    grouped, trans_a = case == "Grouped", case == "GemmTransA"
    weight_shape = ((experts,) if grouped else ()) + ((n, k) if transpose else (k, n))
    low, high, zero = (-70, 70, -3) if dtype == np.int8 else (0, 180, 71)
    feeds = {
        "x": rng.normal(0, 0.2, (k, m) if trans_a else (m, k)).astype(np.float32),
        "w": rng.integers(low, high, weight_shape).astype(dtype),
        "scale": np.resize(np.array([0.03, 0.07, 0.011], np.float32), (n,)),
        "zero": np.full((n,), zero, dtype),
    }
    axis = len(weight_shape) - (2 if transpose else 1)
    weight = ((feeds["w"].astype(np.float32) - _expanded(feeds["zero"], weight_shape, axis, 0))
              * _expanded(feeds["scale"], weight_shape, axis, 0))
    activation = feeds["x"].T if trans_a else feeds["x"]
    nodes = [helper.make_node("DequantizeLinear", ["w", "scale", "zero"], ["wf"], axis=axis)]
    if grouped:
        feeds["ids"] = np.resize(np.array([4, 1, 1, 4, 4, 1], np.int64), (m, selections))
        feeds["bias"] = rng.normal(0, 0.2, (experts, n)).astype(np.float32)
        nodes.append(helper.make_node("GroupedMatMul", ["x", "wf", "ids", "bias"], ["y"],
                                      transB=transpose))
        canonical = weight.transpose(0, 2, 1) if transpose else weight
        expected = np.stack([
            activation[row].astype(np.float64) @ canonical[expert].astype(np.float64)
            for row, choices in enumerate(feeds["ids"]) for expert in choices
        ]).reshape(m, selections, n).astype(np.float32) + feeds["bias"][feeds["ids"]]
        shape = [m, selections, n]
    else:
        feeds["c"] = rng.normal(0, 0.2, (n,)).astype(np.float32)
        nodes.append(helper.make_node("Gemm", ["x", "wf", "c"], ["y"],
                                      transA=int(trans_a), transB=transpose,
                                      alpha=0.75, beta=-0.25))
        canonical = weight.T if transpose else weight
        product = (activation.astype(np.float64) @ canonical.astype(np.float64)).astype(np.float32)
        expected = product * np.float32(0.75) + feeds["c"] * np.float32(-0.25)
        shape = [m, n]
    info = helper.make_tensor_value_info
    model = helper.make_model(helper.make_graph(
        nodes, "weight_only_prefill",
        [info(name, helper.np_dtype_to_tensor_dtype(value.dtype), list(value.shape))
         for name, value in feeds.items()],
        [info("y", TensorProto.FLOAT, shape)],
    ), opset_imports=[helper.make_opsetid("", 29 if grouped else 24)], ir_version=11)
    reference = ReferenceEvaluator(model).run(None, feeds)[0]
    np.testing.assert_allclose(reference, expected, rtol=2e-5, atol=8e-6)
    for disabled in ["0", "1"]:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION", disabled)
        actual = schema_session(model.SerializeToString()).run(None, feeds)[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=8e-6)


def test_weight_only_prefill_actual_dispatch(tmp_path):
    _require_grouped_reference()
    probe, _, _, _ = _transposed_model("Grouped", "axis", np.int8)
    schema_session(probe)
    trace = tmp_path / "weight-only-prefill.json"
    subprocess.run(
        [sys.executable, "-m", "pytest",
         f"{Path(__file__)}::test_weight_only_prefill", "-k", "Grouped and int8",
         "-q", "-p", "no:cacheprovider"],
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
        check=True, timeout=180,
    )
    kernels = {event.get("args", {}).get("kernel")
               for event in json.loads(trace.read_text()) if event.get("cat") == "op.fast"}
    assert "mlx_qdq_tiled_matmul" in kernels, kernels


def _expanded(value, shape, axis, block):
    if value.size == 1 and block == 0:
        return value.reshape(())
    if block == 0:
        broadcast = [1] * len(shape)
        broadcast[axis] = shape[axis]
        return value.reshape(broadcast)
    result = np.repeat(value, block, axis=axis)
    selection = [slice(None)] * len(shape)
    selection[axis] = slice(shape[axis])
    return result[tuple(selection)]


def _model(case, dtype=np.int8):
    k = 256 if case == "integer" else 1025
    n = 7
    rng = np.random.default_rng(51)
    feeds = {
        "a": rng.integers(0, 17, (2, k)).astype(dtype),
        "w": rng.integers(0, 13, (k, n)).astype(dtype),
    }
    zero = 7
    a_axis, w_axis = (0, 0 if case == "blocked" else 1)
    block = 32 if case == "blocked" else 0
    if case == "integer":
        a_scale = np.array(0.25, np.float32)
        w_scale = np.array(0.125, np.float32)
    elif case == "blocked":
        a_scale = np.array([0.03, 0.07], np.float32)
        w_scale = np.linspace(0.01, 0.04, ((k + block - 1) // block) * n,
                              dtype=np.float32).reshape(-1, n)
    else:
        a_scale = np.array(0.03125, np.float32)
        w_scale = np.linspace(0.015, 0.04, n, dtype=np.float32)
    values = {
        "as": a_scale, "ws": w_scale,
        "az": np.full(a_scale.shape, zero, dtype),
        "wz": np.full(w_scale.shape, zero, dtype),
    }
    # Static binary scales enable the proved integer subset; all other parameters stay live.
    initializers = []
    if case == "integer":
        for name in ["as", "ws"]:
            initializers.append(numpy_helper.from_array(values.pop(name), name))
    feeds.update(values)
    nodes = [
        helper.make_node("DequantizeLinear", ["a", "as", "az"], ["af"], axis=a_axis),
        helper.make_node("DequantizeLinear", ["w", "ws", "wz"], ["wf"],
                         axis=w_axis, **({"block_size": block} if block else {})),
        helper.make_node("MatMul", ["af", "wf"], ["y"]),
        helper.make_node("Add", ["wf", "wf"], ["shared"]),
    ]
    info = lambda name, dt, shape: helper.make_tensor_value_info(name, dt, shape)
    model = helper.make_model(
        helper.make_graph(
            nodes, "qdq_direct",
            [info(name, helper.np_dtype_to_tensor_dtype(v.dtype), list(v.shape))
             for name, v in feeds.items()],
            [info("y", TensorProto.FLOAT, ["m", n]),
             info("wf", TensorProto.FLOAT, [k, n]),
             info("shared", TensorProto.FLOAT, [k, n])],
            initializers,
            value_info=[info("af", TensorProto.FLOAT, ["m", k])],
        ),
        opset_imports=[helper.make_opsetid("", 24)], ir_version=11,
    )
    # Token count is live and changes across calls; the matrix dtype stays schema-constrained.
    model.graph.input[0].type.tensor_type.shape.dim[0].ClearField("dim_value")
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "m"
    if case == "blocked":
        # Per-token activation scale/zp extents change with M too.
        for value in model.graph.input:
            if value.name in {"as", "az"}:
                value.type.tensor_type.shape.dim[0].ClearField("dim_value")
                value.type.tensor_type.shape.dim[0].dim_param = "m"
    return model.SerializeToString(), feeds, a_scale, w_scale, a_axis, w_axis, block


@pytest.mark.parametrize("case", ["integer", "axis", "blocked"])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8], ids=["int8", "uint8"])
def test_qdq_session_replay(case, dtype):
    model, feeds, a_scale, w_scale, a_axis, w_axis, block = _model(case, dtype)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"])
    for m in [2, 1, 3, 0, 2]:
        feeds["a"] = np.resize(feeds["a"], (m, feeds["w"].shape[0])).copy()
        if case == "blocked":
            feeds["as"] = np.full((m,), 0.03, np.float32)
            feeds["az"] = np.full((m,), 7, dtype)
        sa = feeds.get("as", a_scale)
        sw = feeds.get("ws", w_scale)
        a = ((feeds["a"].astype(np.float32) - _expanded(feeds["az"], feeds["a"].shape, a_axis, 0))
             * _expanded(sa, feeds["a"].shape, a_axis, 0))
        w = ((feeds["w"].astype(np.float32) - _expanded(feeds["wz"], feeds["w"].shape, w_axis, block))
             * _expanded(sw, feeds["w"].shape, w_axis, block))
        actual = session.run(None, feeds)
        # Accumulate the exact FP32 operand products independently of BLAS.
        # FP32 GEMM itself loses ~2.4e-5 on the repeated-value replay case.
        expected = np.array([
            [math.fsum(float(x) * float(y) for x, y in zip(row, column))
             for column in w.T]
            for row in a
        ], dtype=np.float64).reshape(m, w.shape[1])
        np.testing.assert_allclose(actual[0], expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(actual[1], w)
        np.testing.assert_array_equal(actual[2], w + w)
        # Live bytes and parameters must not become trace/cache constants.
        feeds["w"] = (feeds["w"] // 2).copy()
        if "ws" in feeds:
            feeds["ws"] = (feeds["ws"] * np.float32(0.5)).copy()


@pytest.mark.parametrize("case,kernel", [
    ("integer", "mlx_qdq_integer_matmul"),
    ("axis", "mlx_qdq_matmul"),
    ("blocked", "mlx_qdq_matmul"),
])
def test_qdq_actual_dispatch(tmp_path, case, kernel):
    for disabled in [False, True]:
        trace = tmp_path / f"{case}-{disabled}.json"
        env = {**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)}
        if disabled:
            env["ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION"] = "1"
        else:
            env.pop("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION", None)
        subprocess.run(
            [sys.executable, "-m", "pytest",
             f"{Path(__file__)}::test_qdq_session_replay[int8-{case}]",
             "-q", "-p", "no:cacheprovider"], env=env, check=True, timeout=180,
        )
        events = json.loads(trace.read_text())
        kernels = {e.get("args", {}).get("kernel")
                   for e in events if e.get("cat") == "op.fast"}
        assert (kernel in kernels) is not disabled, kernels


@pytest.mark.parametrize("unsupported", ["int16", "half_scale", "vector"])
def test_qdq_unsupported_retains_unfused_path(unsupported):
    k = 33
    dtype = np.int16 if unsupported == "int16" else np.int8
    scale_dtype = np.float16 if unsupported == "half_scale" else np.float32
    out_dtype = TensorProto.FLOAT16 if unsupported == "half_scale" else TensorProto.FLOAT
    x = np.ones((k,) if unsupported == "vector" else (2, k), scale_dtype)
    w = np.ones((k, 3), dtype)
    scale = np.array(0.5, scale_dtype)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("DequantizeLinear", ["w", "s"], ["wf"]),
             helper.make_node("MatMul", ["x", "wf"], ["y"])],
            "unfused",
            [helper.make_tensor_value_info("x", out_dtype, list(x.shape)),
             helper.make_tensor_value_info("w", helper.np_dtype_to_tensor_dtype(w.dtype), list(w.shape)),
             helper.make_tensor_value_info("s", out_dtype, [])],
            [helper.make_tensor_value_info("y", out_dtype, [3] if x.ndim == 1 else [2, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 24)], ir_version=11,
    ).SerializeToString()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"])
    np.testing.assert_array_equal(session.run(None, {"x": x, "w": w, "s": scale})[0],
                                  x @ (w.astype(scale_dtype) * scale))


@pytest.mark.parametrize("k", [4096, 16384])
def test_qdq_distributed_accumulation(k):
    # A single long serial accumulator loses these representable fractional increments.
    scale = np.float32(1.0000152587890625)
    arrays = {
        "a": np.ones((1, k), np.uint8), "w": np.ones((k, 1), np.uint8),
        "as": np.array(1, np.float32), "ws": np.array(scale, np.float32),
    }
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("DequantizeLinear", ["a", "as"], ["af"]),
             helper.make_node("DequantizeLinear", ["w", "ws"], ["wf"]),
             helper.make_node("MatMul", ["af", "wf"], ["y"])],
            "distributed_qdq",
            [helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(v.dtype), list(v.shape))
             for name, v in arrays.items()],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
        ),
        opset_imports=[helper.make_opsetid("", 24)], ir_version=11,
    ).SerializeToString()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    actual = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"]).run(None, arrays)[0]
    np.testing.assert_allclose(actual, [[np.float32(k) * scale]], rtol=1e-5, atol=1e-6)


def test_qdq_quantize_input_and_output():
    x = np.array([[-1.3, -0.5, 0, 0.5, 1.3]], np.float32)
    w = np.arange(15, dtype=np.int8).reshape(5, 3)
    scales = {"xs": np.array(0.25, np.float32), "ws": np.array(0.125, np.float32),
              "ys": np.array(0.125, np.float32)}
    zeros = {"xz": np.array(4, np.uint8), "wz": np.array(5, np.int8),
             "yz": np.array(128, np.uint8)}
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "xs", "xz"], ["xq"]),
        helper.make_node("DequantizeLinear", ["xq", "xs", "xz"], ["xf"]),
        helper.make_node("DequantizeLinear", ["w", "ws", "wz"], ["wf"]),
        helper.make_node("MatMul", ["xf", "wf"], ["y"]),
        helper.make_node("QuantizeLinear", ["y", "ys", "yz"], ["yq"]),
        helper.make_node("DequantizeLinear", ["yq", "ys", "yz"], ["yf"]),
    ]
    values = {"x": x, "w": w, **scales, **zeros}
    model = helper.make_model(
        helper.make_graph(
            nodes, "complete_qdq",
            [helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(v.dtype), list(v.shape))
             for name, v in values.items()],
            [helper.make_tensor_value_info("yq", TensorProto.UINT8, [1, 3]),
             helper.make_tensor_value_info("yf", TensorProto.FLOAT, [1, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 24)], ir_version=11,
    ).SerializeToString()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    actual = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"]).run(None, values)
    xq = np.clip(np.rint(x / scales["xs"]) + zeros["xz"], 0, 255).astype(np.uint8)
    y = ((xq.astype(np.float32) - zeros["xz"]) * scales["xs"]) @ (
        (w.astype(np.float32) - zeros["wz"]) * scales["ws"])
    yq = np.clip(np.rint(y / scales["ys"]) + zeros["yz"], 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(actual[0], yq)
    np.testing.assert_array_equal(actual[1], (yq.astype(np.float32) - zeros["yz"]) * scales["ys"])


@pytest.mark.parametrize("case", ["overflow_cancellation", "zero_times_inf", "nan"])
def test_qdq_weight_only_nonfinite_and_overflow(case):
    maximum = np.finfo(np.float32).max
    x = np.array([[maximum, maximum]], np.float32)
    w = np.array([[1], [-1]], np.int8)
    if case == "zero_times_inf":
        x.fill(np.inf)
        w.fill(0)
    elif case == "nan":
        x[0, 0] = np.nan
    scale = np.array(0.5, np.float32)
    arrays = {"x": x, "w": w, "s": scale}
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("DequantizeLinear", ["w", "s"], ["wf"]),
             helper.make_node("MatMul", ["x", "wf"], ["y"])],
            "dq_before_dot",
            [helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(v.dtype), list(v.shape))
             for name, v in arrays.items()],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
        ), opset_imports=[helper.make_opsetid("", 24)], ir_version=11,
    ).SerializeToString()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    actual = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"]).run(None, arrays)[0]
    with np.errstate(invalid="ignore", over="ignore"):
        expected = x @ (w.astype(np.float32) * scale)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("unsupported", ["int16", "half_scale", "vector"])
def test_qdq_nonfusion_dispatch(tmp_path, unsupported):
    trace = tmp_path / f"unfused-{unsupported}.json"
    env = {**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)}
    env.pop("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION", None)
    subprocess.run(
        [sys.executable, "-m", "pytest",
         f"{Path(__file__)}::test_qdq_unsupported_retains_unfused_path[{unsupported}]",
         "-q", "-p", "no:cacheprovider"], env=env, check=True, timeout=120,
    )
    events = json.loads(trace.read_text())
    kernels = {e.get("args", {}).get("kernel")
               for e in events if e.get("cat") == "op.fast"}
    assert not {
        "mlx_qdq_matmul", "mlx_qdq_matvec", "mlx_qdq_tiled_matmul", "mlx_qdq_integer_matmul",
    } & kernels


@pytest.mark.skipif(not ort.__version__.startswith("1.31."),
                    reason="Negative loader contract is specific to ORT 1.31")
def test_ort131_groupedmatmul29_loader_limit():
    # This proves a runtime limitation, NOT successful GroupedMatMul execution.
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("GroupedMatMul", ["x", "w", "indices"], ["y"])],
            "official_grouped_schema29",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1]),
             helper.make_tensor_value_info("w", TensorProto.FLOAT, [1, 1, 1]),
             helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1])],
        ), opset_imports=[helper.make_opsetid("", 29)], ir_version=11,
    ).SerializeToString()
    with pytest.raises(
        Fail,
        match=r"(?s)Opset 29.*Current official support for domain ai.onnx is till opset 2[0-8]",
    ):
        ort.InferenceSession(model, providers=["MLXExecutionProvider"])
