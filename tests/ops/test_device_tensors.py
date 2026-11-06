"""Public ORT device allocation, ownership and transfers (separate opt-in process)."""

import gc
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

pytestmark = pytest.mark.skipif(
    os.environ.get("ONNXRUNTIME_EP_MLX_DEVICE_TENSORS") != "1",
    reason="run in a fresh process with ONNXRUNTIME_EP_MLX_DEVICE_TENSORS=1",
)


def memory_info():
    device = next(
        d for d in ort.get_ep_devices() if d.ep_name == "MLXExecutionProvider"
    )
    info = device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
    assert info.name == "MLX"
    assert info.device_vendor_id == 0x106B
    return info


def session(*, matrix=False):
    nodes = [helper.make_node("Add", ["x", "x"], ["y"])]
    initializers = []
    if matrix:
        initializers = [numpy_helper.from_array(np.eye(3, dtype=np.float32), "w")]
        nodes = [
            helper.make_node("MatMul", ["x", "w"], ["a"]),
            helper.make_node("Add", ["a", "x"], ["b"]),
            helper.make_node("Relu", ["b"], ["y"]),
        ]
    graph = helper.make_graph(
        nodes,
        "device-tensors",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    result = ort.InferenceSession(
        model.SerializeToString(), options, providers=["MLXExecutionProvider"]
    )
    result.disable_fallback()
    return result


def test_cpu_transfers_and_memory_pattern_reuse():
    model = session(matrix=True)
    for i in range(10):
        x = np.arange(6, dtype=np.float32).reshape(2, 3) - i
        np.testing.assert_array_equal(
            model.run(None, {"x": x})[0], np.maximum(x * 2, 0)
        )


def test_device_state_reuse_and_aliased_output():
    model = session()
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    state = ort.OrtValue.ortvalue_from_shape_and_type(
        x.shape, x.dtype, memory_info=memory_info()
    )
    state.update_inplace(x)
    binding = model.io_binding()
    binding.bind_ortvalue_input("x", state)
    binding.bind_ortvalue_output("y", state)
    for _ in range(6):
        model.run_with_iobinding(binding)
    np.testing.assert_array_equal(state.numpy(), x * 64)


def test_device_output_outlives_session():
    model = session(matrix=True)
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    state = ort.OrtValue.ortvalue_from_shape_and_type(
        x.shape, x.dtype, memory_info=memory_info()
    )
    binding = model.io_binding()
    binding.bind_cpu_input("x", x)
    binding.bind_ortvalue_output("y", state)
    model.run_with_iobinding(binding)
    del binding, model
    gc.collect()
    np.testing.assert_array_equal(state.numpy(), x * 2)


@pytest.mark.parametrize("compiled", [False, True])
def test_quantized_device_output_outlives_initializer_storage(monkeypatch, compiled):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0" if compiled else "1")
    k, n, block = 512, 128, 32
    graph = helper.make_graph(
        [
            helper.make_node(
                "MatMulNBits",
                ["x", "weights", "scales"],
                ["y"],
                domain="com.microsoft",
                K=k,
                N=n,
                block_size=block,
                bits=4,
            )
        ],
        "device-quantized-initializers",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, k])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, n])],
        [
            numpy_helper.from_array(
                np.full((n, k // block, block // 2), 0x99, np.uint8), "weights"
            ),
            numpy_helper.from_array(
                np.full((n, k // block), 0.25, np.float32), "scales"
            ),
        ],
    )
    proto = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=10,
    )
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    model = ort.InferenceSession(
        proto.SerializeToString(), options, providers=["MLXExecutionProvider"]
    )
    model.disable_fallback()
    x = ort.OrtValue.ortvalue_from_shape_and_type(
        [2, k], np.float32, memory_info=memory_info()
    )
    x.update_inplace(np.ones((2, k), np.float32))
    state = ort.OrtValue.ortvalue_from_shape_and_type(
        [2, n], np.float32, memory_info=memory_info()
    )
    binding = model.io_binding()
    binding.bind_ortvalue_input("x", x)
    binding.bind_ortvalue_output("y", state)
    model.run_with_iobinding(binding)
    del binding, model, x, proto, graph
    gc.collect()
    np.testing.assert_array_equal(state.numpy(), np.full((2, n), k * 0.25))


def test_memory_pattern_offsets_across_partitions():
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["x", "x"], ["a"]),
            helper.make_node("Mod", ["a", "divisor"], ["b"], fmod=1),
            helper.make_node("Add", ["b", "b"], ["y"]),
        ],
        "device-memory-pattern",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
        [numpy_helper.from_array(np.array(3, np.float32), "divisor")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    model = ort.InferenceSession(
        model.SerializeToString(), options, providers=["MLXExecutionProvider"]
    )
    model.disable_fallback()
    for i in range(10):
        x = np.arange(6, dtype=np.float32).reshape(2, 3) - i
        np.testing.assert_array_equal(
            model.run(None, {"x": x})[0], np.fmod(x * 2, 3) * 2
        )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int32, np.int64])
def test_device_upload_download_preserves_bytes(dtype):
    x = np.arange(6, dtype=dtype).reshape(2, 3)
    state = ort.OrtValue.ortvalue_from_shape_and_type(
        x.shape, x.dtype, memory_info=memory_info()
    )
    for scale in (1, 2, 3):
        source = x * scale
        state.update_inplace(source)
        actual = state.numpy()
        assert actual.dtype == source.dtype
        assert actual.shape == source.shape
        assert actual.tobytes() == source.tobytes()


def test_device_host_parameters_remain_live():
    graph = helper.make_graph(
        [helper.make_node("Reshape", ["x", "shape"], ["y"])],
        "device-host-parameter",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("shape", TensorProto.INT64, [2]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["r", "c"])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    model = ort.InferenceSession(
        model.SerializeToString(), options, providers=["MLXExecutionProvider"]
    )
    model.disable_fallback()
    shape = ort.OrtValue.ortvalue_from_shape_and_type(
        [2], np.int64, memory_info=memory_info()
    )
    binding = model.io_binding()
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    binding.bind_cpu_input("x", x)
    binding.bind_ortvalue_input("shape", shape)
    for dimensions in ([3, 2], [1, 6], [6, 1], [3, 2]):
        shape.update_inplace(np.array(dimensions, np.int64))
        binding.bind_output("y", "cpu")
        model.run_with_iobinding(binding)
        np.testing.assert_array_equal(
            binding.copy_outputs_to_cpu()[0], x.reshape(dimensions)
        )


def test_uninitialized_device_input_is_actionable():
    model = session()
    state = ort.OrtValue.ortvalue_from_shape_and_type(
        [2, 3], np.float32, memory_info=memory_info()
    )
    binding = model.io_binding()
    binding.bind_ortvalue_input("x", state)
    binding.bind_output("y", "cpu")
    with pytest.raises(RuntimeError, match="no initialized payload"):
        model.run_with_iobinding(binding)


def test_device_transfer_rejects_foreign_run_thread():
    model = session()
    state = ort.OrtValue.ortvalue_from_shape_and_type(
        [2, 3], np.float32, memory_info=memory_info()
    )
    binding = model.io_binding()
    binding.bind_cpu_input("x", np.ones((2, 3), np.float32))
    binding.bind_ortvalue_output("y", state)
    model.run_with_iobinding(binding)
    with (
        ThreadPoolExecutor(max_workers=1) as pool,
        pytest.raises(RuntimeError, match="producer's run thread"),
    ):
        pool.submit(state.numpy).result()
