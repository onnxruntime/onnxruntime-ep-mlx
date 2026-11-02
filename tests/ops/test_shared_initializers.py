"""Immutable weights share storage while sessions retain independent inputs and lifetimes."""

from __future__ import annotations

import ctypes
import gc
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
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


def test_independent_run_threads_keep_correct_session_ownership():
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
        assert list(pool.map(worker, (1, 2))) == [1, 2]
