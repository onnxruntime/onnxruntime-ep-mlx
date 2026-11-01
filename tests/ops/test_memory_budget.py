"""Explicit MLX free-cache budgets and dynamic-shape inference."""

import ctypes
import os

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

ENV_KEY = "ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES"
SESSION_KEY = "ep.mlx.cache_limit_bytes"


def model_bytes():
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["rows", 1024])
    tape = ir.tape.Tape()
    offset = tape.initializer(ir.tensor(np.float32(0.01), name="offset"))
    y = tape.op("Tanh", [tape.op("Add", [x, offset])])
    y.name, y.dtype, y.shape = "y", ir.DataType.FLOAT, ir.Shape(["rows", 1024])
    model = ir.Model(
        ir.Graph(
            [x],
            [y],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="cache_budget",
        ),
        ir_version=10,
    )
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto, full_check=True)
    return proto.SerializeToString()


def runtime(value=None):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if value is not None:
        options.add_session_config_entry(SESSION_KEY, value)
    result = ort.InferenceSession(
        model_bytes(),
        options,
        providers=["MLXExecutionProvider"],
        enable_fallback=False,
    )
    result.disable_fallback()
    return result


@pytest.fixture
def allocator():
    api = ctypes.CDLL(os.environ["ONNXRUNTIME_MLX_EP_LIB"])
    api.mlx_set_cache_limit.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]
    api.mlx_set_cache_limit.restype = ctypes.c_int
    api.mlx_get_cache_memory.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
    api.mlx_get_cache_memory.restype = ctypes.c_int

    def set_limit(value):
        previous = ctypes.c_size_t()
        assert api.mlx_set_cache_limit(ctypes.byref(previous), value) == 0
        return previous.value

    def get_limit():
        previous = set_limit(0)
        set_limit(previous)
        return previous

    previous = get_limit()
    try:
        yield api, set_limit, get_limit
    finally:
        set_limit(previous)


def test_omitted_budget_preserves_application_setting(monkeypatch, allocator):
    monkeypatch.delenv(ENV_KEY, raising=False)
    _, set_limit, get_limit = allocator
    set_limit(12345678)
    executor = runtime()
    assert get_limit() == 12345678
    x = np.zeros((2, 1024), np.float32)
    np.testing.assert_allclose(
        executor.run(None, {"x": x})[0], np.tanh(x + 0.01), atol=1e-6
    )


def test_session_budget_overrides_environment(monkeypatch, allocator):
    monkeypatch.setenv(ENV_KEY, "invalid")
    _, _, get_limit = allocator
    executor = runtime("1048576")
    assert get_limit() == 1048576
    x = np.ones((3, 1024), np.float32)
    np.testing.assert_allclose(
        executor.run(None, {"x": x})[0], np.tanh(x + 0.01), atol=1e-6
    )


@pytest.mark.parametrize("source", ["session", "environment"])
@pytest.mark.parametrize("value", ["", "-1", "1.5", "32MiB", "184467440737095516160"])
def test_invalid_budget_is_an_error(monkeypatch, allocator, source, value):
    _, set_limit, get_limit = allocator
    set_limit(12345678)
    monkeypatch.delenv(ENV_KEY, raising=False)
    if source == "environment":
        monkeypatch.setenv(ENV_KEY, value)
    with pytest.raises(
        RuntimeError, match=r"INVALID_ARGUMENT.*MLX allocator cache budget"
    ):
        runtime(value if source == "session" else None)
    assert get_limit() == 12345678


def test_zero_budget_executes_variable_shapes_without_free_cache(
    monkeypatch, allocator
):
    monkeypatch.setenv(ENV_KEY, "0")
    api, _, get_limit = allocator
    executor = runtime()
    assert get_limit() == 0
    for rows in range(16, 48, 2):
        x = np.linspace(-0.2, 0.2, rows * 1024, dtype=np.float32).reshape(rows, 1024)
        actual = executor.run(None, {"x": x})[0]
        np.testing.assert_allclose(actual, np.tanh(x + 0.01), rtol=1e-5, atol=1e-6)
        cached = ctypes.c_size_t()
        assert api.mlx_get_cache_memory(ctypes.byref(cached)) == 0
        assert cached.value == 0
