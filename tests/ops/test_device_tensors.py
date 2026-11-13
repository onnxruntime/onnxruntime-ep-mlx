"""Public ORT device allocation, ownership and transfers (separate opt-in process)."""

import gc
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from bench.mtp_replay import make_checkpoint_model, make_prefix_replay_model

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


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("owned", [None, "0", "1"])
@pytest.mark.parametrize("count", [1024, 1025])
def test_cpu_input_ownership_threshold_and_output_lifetime(
    monkeypatch, eager, owned, count
):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    if owned is None:
        monkeypatch.delenv("ONNXRUNTIME_EP_MLX_OWN_SMALL_INPUTS", raising=False)
    else:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_OWN_SMALL_INPUTS", owned)
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "one"], ["y"])],
        "owned_cpu_input",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [count])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [count])],
        [numpy_helper.from_array(np.array(1, np.float32), "one")],
    )
    model = helper.make_model(
        graph, ir_version=11, opset_imports=[helper.make_opsetid("", 24)]
    )
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["MLXExecutionProvider"]
    )
    output = ort.OrtValue.ortvalue_from_shape_and_type(
        [count], np.float32, memory_info=memory_info()
    )
    binding = session.io_binding()
    binding.bind_ortvalue_output("y", output)
    for offset in (0, 7):
        value = np.arange(count, dtype=np.float32) + offset
        expected = value + 1
        binding.bind_cpu_input("x", value)
        session.run_with_iobinding(binding)
        value.fill(-88)
        binding.clear_binding_inputs()
        del value
        np.testing.assert_array_equal(output.numpy(), expected)
    del binding, session
    gc.collect()
    np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("conv_dtype", [np.float16, np.float32])
@pytest.mark.parametrize("geometry", [(2, 6, 8, 8), (16, 48, 128, 128)])
def test_public_mtp_prefix_replay_retains_immutable_checkpoint(
    monkeypatch, eager, conv_dtype, geometry
):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    key_heads, heads, key_width, value_width = geometry
    capacity, batch = 4, 1
    channels = 2 * key_heads * key_width + heads * value_width
    inputs = [
        SimpleNamespace(
            name="past.0.recurrent",
            type="tensor(float)",
            shape=[batch, heads, value_width, key_width],
        ),
        SimpleNamespace(
            name="past.0.conv",
            type="tensor(float16)" if conv_dtype == np.float16 else "tensor(float)",
            shape=[batch, channels, 3],
        ),
    ]
    captures = [
        SimpleNamespace(
            name="state_update.0.recurrent_capsule",
            type="tensor(float)",
            shape=[
                batch,
                capacity * (heads + key_heads * key_width + heads * value_width),
            ],
        ),
        SimpleNamespace(
            name="state_update.0.conv_value",
            type=inputs[1].type,
            shape=[batch, capacity, channels],
        ),
    ]
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    checkpoint_session = ort.InferenceSession(
        make_checkpoint_model(inputs).SerializeToString(),
        options,
        providers=["MLXExecutionProvider"],
    )
    replay_session = ort.InferenceSession(
        make_prefix_replay_model(
            inputs, captures, capacity=capacity
        ).SerializeToString(),
        options,
        providers=["MLXExecutionProvider"],
    )
    checkpoint_session.disable_fallback()
    replay_session.disable_fallback()
    info = memory_info()

    def device_value(shape, dtype, data=None):
        value = ort.OrtValue.ortvalue_from_shape_and_type(
            shape, dtype, memory_info=info
        )
        if data is not None:
            value.update_inplace(data)
        return value

    rng = np.random.default_rng(4620)
    original = {
        value.name: rng.normal(size=value.shape).astype(
            np.float32 if value.type == "tensor(float)" else np.float16
        )
        for value in inputs
    }
    bits_type = np.uint16 if conv_dtype == np.float16 else np.uint32
    bits = (
        [0x8000, 0x7C01, 0x7FFF, 0xFC02]
        if conv_dtype == np.float16
        else [0x80000000, 0x7F800001, 0x7FFFFFFF, 0xFF800002]
    )
    original["past.0.conv"].view(bits_type).flat[:4] = bits
    states, snapshots = {}, {}
    checkpoint_binding = checkpoint_session.io_binding()
    replay_binding = replay_session.io_binding()
    for value in inputs:
        source = original[value.name]
        states[value.name] = device_value(source.shape, source.dtype, source)
        snapshots[value.name] = device_value(source.shape, source.dtype)
        checkpoint_binding.bind_ortvalue_input(value.name, states[value.name])
        checkpoint_binding.bind_ortvalue_output(
            "checkpoint." + value.name, snapshots[value.name]
        )
        replay_binding.bind_ortvalue_input(value.name, snapshots[value.name])
        replay_binding.bind_ortvalue_output(
            "present." + value.name.removeprefix("past."), states[value.name]
        )
    checkpoint_session.run_with_iobinding(checkpoint_binding)
    for value in inputs:
        states[value.name].update_inplace(np.zeros_like(original[value.name]))
    capture_values = {}
    for _ in range(3):
        host_captures = {
            value.name: rng.normal(size=value.shape).astype(
                np.float32 if value.type == "tensor(float)" else np.float16
            )
            for value in captures
        }
        host_captures["state_update.0.conv_value"].view(bits_type).flat[:4] = bits[::-1]
        for name, data in host_captures.items():
            if name not in capture_values:
                capture_values[name] = device_value(data.shape, data.dtype, data)
                replay_binding.bind_ortvalue_input(name, capture_values[name])
            else:
                capture_values[name].update_inplace(data)
        replay_session.run_with_iobinding(replay_binding)
        capsule = host_captures["state_update.0.recurrent_capsule"]
        key_offset = capacity * heads
        delta_offset = key_offset + capacity * key_heads * key_width
        decay = capsule[:, :key_offset].reshape(batch, capacity, heads)[:, 0]
        key = capsule[:, key_offset:delta_offset].reshape(
            batch, capacity, key_heads, key_width
        )[:, 0]
        delta = capsule[:, delta_offset:].reshape(batch, capacity, heads, value_width)[
            :, 0
        ]
        decayed = original["past.0.recurrent"] * decay[:, :, None, None]
        expected = (
            decayed.astype(np.float64)
            + np.repeat(key, heads // key_heads, axis=1)[:, :, None, :].astype(
                np.float64
            )
            * delta[:, :, :, None].astype(np.float64)
        ).astype(np.float32)
        np.testing.assert_array_equal(
            states["past.0.recurrent"].numpy().view(np.uint32), expected.view(np.uint32)
        )
        expected_conv = np.concatenate(
            [
                original["past.0.conv"][:, :, 1:],
                host_captures["state_update.0.conv_value"][:, :1].transpose(0, 2, 1),
            ],
            axis=2,
        )
        np.testing.assert_array_equal(
            states["past.0.conv"].numpy().view(bits_type), expected_conv.view(bits_type)
        )
        for name, snapshot in snapshots.items():
            np.testing.assert_array_equal(
                snapshot.numpy().view(np.uint8), original[name].view(np.uint8)
            )
    del checkpoint_binding, replay_binding, checkpoint_session, replay_session
    gc.collect()
    for name, snapshot in snapshots.items():
        np.testing.assert_array_equal(
            snapshot.numpy().view(np.uint8), original[name].view(np.uint8)
        )


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


def test_float64_upload_reshape_download_and_session_lifetime():
    shape = np.array([3, 2], dtype=np.int64)
    graph = helper.make_graph(
        [helper.make_node("Reshape", ["x", "shape"], ["y"])],
        "float64-device-tensors",
        [helper.make_tensor_value_info("x", TensorProto.DOUBLE, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.DOUBLE, [3, 2])],
        [numpy_helper.from_array(shape, "shape")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 24)], ir_version=11
    )
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["MLXExecutionProvider"]
    )
    session.disable_fallback()
    source = np.arange(6, dtype=np.float64).reshape(2, 3)
    device_input = ort.OrtValue.ortvalue_from_shape_and_type(
        source.shape, source.dtype, memory_info=memory_info()
    )
    device_input.update_inplace(source)
    np.testing.assert_array_equal(device_input.numpy(), source)
    device_output = ort.OrtValue.ortvalue_from_shape_and_type(
        (3, 2), source.dtype, memory_info=memory_info()
    )
    assert device_input.data_type() == "tensor(double)"
    assert device_output.data_type() == "tensor(double)"
    binding = session.io_binding()
    binding.bind_ortvalue_input("x", device_input)
    binding.bind_ortvalue_output("y", device_output)
    session.run_with_iobinding(binding)
    assert device_output.data_type() == "tensor(double)"
    del binding, device_input, session
    gc.collect()
    np.testing.assert_array_equal(device_output.numpy(), source.reshape(3, 2))


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
