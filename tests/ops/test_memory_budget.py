"""Explicit MLX free-cache/residency budgets and dynamic-shape inference."""

import ctypes
import os

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest


@pytest.fixture(params=["cache", "wired"])
def kind(request):
    return request.param


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


def runtime(value=None, kind="cache"):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if value is not None:
        options.add_session_config_entry(f"ep.mlx.{kind}_limit_bytes", value)
    result = ort.InferenceSession(
        model_bytes(),
        options,
        providers=["MLXExecutionProvider"],
        enable_fallback=False,
    )
    result.disable_fallback()
    return result


@pytest.fixture
def allocator(kind):
    api = ctypes.CDLL(os.environ["ONNXRUNTIME_MLX_EP_LIB"])
    setter = getattr(api, f"mlx_set_{kind}_limit")
    setter.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]
    setter.restype = ctypes.c_int
    api.mlx_get_cache_memory.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
    api.mlx_get_cache_memory.restype = ctypes.c_int

    def set_limit(value):
        previous = ctypes.c_size_t()
        assert setter(ctypes.byref(previous), value) == 0
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


def test_omitted_budget_preserves_application_setting(monkeypatch, allocator, kind):
    monkeypatch.delenv(f"ONNXRUNTIME_EP_MLX_{kind.upper()}_LIMIT_BYTES", raising=False)
    _, set_limit, get_limit = allocator
    set_limit(12345678)
    executor = runtime(kind=kind)
    assert get_limit() == 12345678
    x = np.zeros((2, 1024), np.float32)
    np.testing.assert_allclose(
        executor.run(None, {"x": x})[0], np.tanh(x + 0.01), atol=1e-6
    )


def test_session_budget_overrides_environment(monkeypatch, allocator, kind):
    monkeypatch.setenv(f"ONNXRUNTIME_EP_MLX_{kind.upper()}_LIMIT_BYTES", "invalid")
    _, _, get_limit = allocator
    executor = runtime("1048576", kind=kind)
    assert get_limit() == 1048576
    x = np.ones((3, 1024), np.float32)
    np.testing.assert_allclose(
        executor.run(None, {"x": x})[0], np.tanh(x + 0.01), atol=1e-6
    )


@pytest.mark.parametrize("source", ["session", "environment"])
@pytest.mark.parametrize("value", ["", "-1", "1.5", "32MiB", "184467440737095516160"])
def test_invalid_budget_is_an_error(monkeypatch, allocator, kind, source, value):
    _, set_limit, get_limit = allocator
    set_limit(12345678)
    env_key = f"ONNXRUNTIME_EP_MLX_{kind.upper()}_LIMIT_BYTES"
    monkeypatch.delenv(env_key, raising=False)
    if source == "environment":
        monkeypatch.setenv(env_key, value)
    label = "allocator cache" if kind == "cache" else "wired memory"
    with pytest.raises(RuntimeError, match=rf"INVALID_ARGUMENT.*MLX {label} budget"):
        runtime(value if source == "session" else None, kind=kind)
    assert get_limit() == 12345678


def test_zero_budget_executes_variable_shapes(monkeypatch, allocator, kind):
    monkeypatch.setenv(f"ONNXRUNTIME_EP_MLX_{kind.upper()}_LIMIT_BYTES", "0")
    api, _, get_limit = allocator
    executor = runtime(kind=kind)
    assert get_limit() == 0
    for rows in range(16, 48, 2):
        x = np.linspace(-0.2, 0.2, rows * 1024, dtype=np.float32).reshape(rows, 1024)
        actual = executor.run(None, {"x": x})[0]
        np.testing.assert_allclose(actual, np.tanh(x + 0.01), rtol=1e-5, atol=1e-6)
        if kind == "cache":
            cached = ctypes.c_size_t()
            assert api.mlx_get_cache_memory(ctypes.byref(cached)) == 0
            assert cached.value == 0
        assert get_limit() == 0


def test_invalid_second_setting_does_not_change_first(monkeypatch, allocator, kind):
    first = f"ONNXRUNTIME_EP_MLX_{kind.upper()}_LIMIT_BYTES"
    second = f"ONNXRUNTIME_EP_MLX_{'CACHE' if kind == 'wired' else 'WIRED'}_LIMIT_BYTES"
    _, set_limit, get_limit = allocator
    set_limit(12345678)
    monkeypatch.setenv(first, "1048576")
    monkeypatch.setenv(second, "invalid")
    with pytest.raises(RuntimeError, match=r"INVALID_ARGUMENT.*budget"):
        runtime(kind=kind)
    assert get_limit() == 12345678


@pytest.mark.parametrize("kind", ["wired"], indirect=True)
def test_device_rejects_excessive_wired_budget(monkeypatch, allocator, kind):
    monkeypatch.delenv("ONNXRUNTIME_EP_MLX_WIRED_LIMIT_BYTES", raising=False)
    _, set_limit, get_limit = allocator
    set_limit(12345678)
    with pytest.raises(RuntimeError, match=r"wired memory budget.*device-supported"):
        runtime(str(2**64 - 1), kind=kind)
    assert get_limit() == 12345678
