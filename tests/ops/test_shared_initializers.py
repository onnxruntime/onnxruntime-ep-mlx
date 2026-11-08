"""Immutable weights share storage while sessions retain independent inputs and lifetimes."""

from __future__ import annotations

import ctypes
import gc
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest


def model(weights, *, name="weight", overridable=False):
    weight = ir.val(name, const_value=ir.tensor(weights))
    indices = ir.val("indices", dtype=ir.DataType.INT64, shape=[1])
    output = ir.val("output", dtype=weight.dtype, shape=[1, weights.shape[1]])
    node = ir.node("Gather", [weight, indices], attributes={"axis": 0}, outputs=[output])
    graph = ir.Graph(
        [indices, weight] if overridable else [indices],
        [output],
        nodes=[node],
        initializers=[weight],
        opset_imports={"": 24},
        name="shared_weights",
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def session(weights, **kwargs):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(
        model(weights, **kwargs), options, providers=["MLXExecutionProvider"]
    )


def run(executor, weights, **feeds):
    result = executor.run(None, {"indices": np.array([3], np.int64), **feeds})[0]
    np.testing.assert_array_equal(result, weights[3:4])


def active_bytes():
    library = ctypes.CDLL(os.environ["ONNXRUNTIME_MLX_EP_LIB"])
    function = library.mlx_get_active_memory
    function.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
    function.restype = ctypes.c_int
    count = ctypes.c_size_t()
    assert function(ctypes.byref(count)) == 0
    return count.value


@pytest.fixture(autouse=True)
def eager(monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    monkeypatch.delenv("ONNXRUNTIME_EP_MLX_NO_SHARED_INITIALIZERS", raising=False)
    gc.collect()


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32, np.int8, np.uint8])
def test_shared_payload_survives_original_session_and_releases_last_owner(dtype):
    weights = (np.arange(512 * 512) % 32).astype(dtype).reshape(512, 512)
    before = active_bytes()
    first = session(weights)
    run(first, weights)
    one = active_bytes()
    second = session(weights.copy(), name="another_module.weight")
    run(second, weights)
    two = active_bytes()
    assert one - before >= weights.nbytes
    assert two - one < weights.nbytes // 4
    del first
    gc.collect()
    run(second, weights)
    del second
    gc.collect()
    assert active_bytes() == before


def test_dtype_shape_and_contents_distinguish_weights():
    values = np.zeros((512, 512), np.float32)
    candidates = [values, values.view(np.int32), values.reshape(256, 1024), values + 1]
    executors = []
    baseline = active_bytes()
    for weights in candidates:
        executor = session(weights)
        run(executor, weights)
        executors.append(executor)
    assert active_bytes() - baseline >= sum(weights.nbytes for weights in candidates)
    del executor
    executors.clear()
    gc.collect()
    assert active_bytes() == baseline


def test_overridable_initializer_uses_live_feed_each_time():
    weights = np.ones((512, 512), np.float32)
    first = session(weights)
    run(first, weights)
    changing = session(weights, overridable=True)
    for value in (2, 3, 1):
        override = weights * value
        run(changing, override, weight=override)
    run(first, weights)


def test_disable_switch_retains_independent_storage(monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_SHARED_INITIALIZERS", "1")
    weights = np.ones((512, 512), np.float32)
    baseline = active_bytes()
    first, second = session(weights), session(weights)
    run(first, weights)
    run(second, weights)
    assert active_bytes() - baseline >= 2 * weights.nbytes


@pytest.mark.parametrize("seeds", [(1, 2), (7, 7)])
def test_independent_run_threads_keep_correct_session_ownership(seeds):
    def worker(seed):
        weights = np.full((128, 128), seed, np.float32)
        first, second = session(weights), session(weights, name="second")
        for _ in range(3):
            run(first, weights)
            run(second, weights)
        del first
        gc.collect()
        run(second, weights)
        return seed

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(worker, seeds)) == list(seeds)


def test_original_session_can_close_on_a_different_thread():
    weights = np.full((128, 128), 7, np.float32)
    first = session(weights)
    second = session(weights, name="second")
    run(first, weights)
    run(second, weights)
    holder = [first]
    del first

    def release():
        holder.clear()
        gc.collect()

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(release).result()
    run(second, weights)


def test_compiled_sessions_reuse_immutable_storage(monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0")
    weights = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    first, second = session(weights), session(weights, name="second")
    for executor in (first, second):
        for row in (3, 4, 3):
            actual = executor.run(None, {"indices": np.array([row], np.int64)})[0]
            np.testing.assert_array_equal(actual, weights[row : row + 1])
    del first
    gc.collect()
    run(second, weights)


def quantized_session(*, prefix, overridable=False):
    packed = np.full((512, 16, 16), 0x99, np.uint8)
    scales = np.full((512, 16), 0.01, np.float16)
    weight_name, scale_name = f"{prefix}.weight", f"{prefix}.scales"
    inputs = [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT16, [1, 512])]
    if overridable:
        inputs.append(onnx.helper.make_tensor_value_info(
            scale_name, onnx.TensorProto.FLOAT16, list(scales.shape)
        ))
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node(
            "MatMulNBits", ["x", weight_name, scale_name], ["y"],
            domain="com.microsoft", K=512, N=512, bits=4, block_size=32,
        )],
        "shared_quantized_views", inputs,
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT16, [1, 512])],
        [
            onnx.numpy_helper.from_array(packed, weight_name),
            onnx.numpy_helper.from_array(scales, scale_name),
        ],
    )
    proto = onnx.helper.make_model(
        graph, ir_version=11,
        opset_imports=[onnx.helper.make_opsetid("", 24), onnx.helper.make_opsetid("com.microsoft", 1)],
    )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(
        proto.SerializeToString(), options, providers=["MLXExecutionProvider"]
    ), packed, scales


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("sharing", [False, True])
def test_quantized_view_sharing_control_and_session_lifetime(monkeypatch, compiled, sharing):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0" if compiled else "1")
    if sharing:
        monkeypatch.delenv("ONNXRUNTIME_EP_MLX_NO_SHARED_QUANT_VIEWS", raising=False)
    else:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_SHARED_QUANT_VIEWS", "1")
    before = active_bytes()
    first, packed, scales = quantized_session(prefix="target")
    feeds = {"x": np.ones((1, 512), np.float16)}
    expected = np.full((1, 512), float(scales[0, 0]) * 512, np.float16)
    np.testing.assert_allclose(first.run(None, feeds)[0], expected, rtol=2e-3)
    one = active_bytes()
    second, _, _ = quantized_session(prefix="draft")
    np.testing.assert_allclose(second.run(None, feeds)[0], expected, rtol=2e-3)
    if sharing:
        assert active_bytes() - one < packed.nbytes // 2
    else:
        assert active_bytes() - one >= packed.nbytes
    del first
    gc.collect()
    np.testing.assert_allclose(second.run(None, feeds)[0], expected, rtol=2e-3)
    del second
    gc.collect()
    assert active_bytes() == before


@pytest.mark.parametrize("compiled", [False, True])
def test_quantized_shared_views_keep_overridable_scales_live(monkeypatch, compiled):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0" if compiled else "1")
    monkeypatch.delenv("ONNXRUNTIME_EP_MLX_NO_SHARED_QUANT_VIEWS", raising=False)
    executor, _, scales = quantized_session(prefix="live", overridable=True)
    for multiplier in (1, 2, 3):
        current = scales * multiplier
        actual = executor.run(None, {
            "x": np.ones((1, 512), np.float16), "live.scales": current,
        })[0]
        expected = np.full((1, 512), float(current[0, 0]) * 512, np.float16)
        np.testing.assert_allclose(actual, expected, rtol=2e-3)
