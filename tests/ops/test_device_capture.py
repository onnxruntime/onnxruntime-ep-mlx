"""One-shot Metal capture must include GPU-owned asynchronous boundaries."""

import os
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "darwin"
    or os.environ.get("ONNXRUNTIME_EP_MLX_DEVICE_TENSORS") != "1",
    reason="requires a fresh macOS process with public MLX device tensors",
)


@pytest.mark.parametrize("eager", [False, True])
def test_gpu_owned_boundary_capture(tmp_path, eager):
    capture = tmp_path / "boundary.gputrace"
    environment = {
        **os.environ,
        "MTL_CAPTURE_ENABLED": "1",
        "ONNXRUNTIME_EP_MLX_GPU_CAPTURE": str(capture),
        "ONNXRUNTIME_EP_MLX_GPU_CAPTURE_EVAL": "0",
        "ONNXRUNTIME_EP_MLX_NO_COMPILE": "1" if eager else "0",
    }
    script = """
import os
import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ort.register_execution_provider_library(
    "MLXExecutionProvider", os.environ["ONNXRUNTIME_MLX_EP_LIB"]
)
device = next(d for d in ort.get_ep_devices() if d.ep_name == "MLXExecutionProvider")
graph = helper.make_graph(
    [
        helper.make_node("MatMul", ["x", "weight"], ["product"]),
        helper.make_node("Add", ["product", "one"], ["y"]),
    ],
    "captured_device_boundary",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32])],
    [
        numpy_helper.from_array(np.eye(32, dtype=np.float32), "weight"),
        numpy_helper.from_array(np.array(1, np.float32), "one"),
    ],
)
model = helper.make_model(
    graph, ir_version=10, opset_imports=[helper.make_opsetid("", 21)]
)
options = ort.SessionOptions()
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
session = ort.InferenceSession(
    model.SerializeToString(), options, providers=["MLXExecutionProvider"]
)
output = ort.OrtValue.ortvalue_from_shape_and_type(
    [1, 32], np.float32,
    memory_info=device.memory_info(ort.OrtDeviceMemoryType.DEFAULT),
)
binding = session.io_binding()
binding.bind_ortvalue_output("y", output)
for offset in (0, 7, 19):
    x = np.arange(32, dtype=np.float32)[None] + offset
    binding.bind_cpu_input("x", x)
    session.run_with_iobinding(binding)
    np.testing.assert_array_equal(output.numpy(), x + 1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], env=environment,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert capture.exists(), result.stdout + result.stderr
    assert any(capture.iterdir()), "Metal capture contains no recorded data"
