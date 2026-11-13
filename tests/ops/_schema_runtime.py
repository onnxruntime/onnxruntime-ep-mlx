"""Strict provider execution and independent reference checks for development schemas."""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def model(op, feeds, attrs=None, output_names=("y",), *, opset):
    arrays = {name: numpy_helper.from_array(value) for name, value in feeds.items()}
    graph = helper.make_graph(
        [helper.make_node(op, list(feeds), list(output_names), **(attrs or {}))],
        op,
        [helper.make_tensor_value_info(name, arrays[name].data_type, list(value.shape))
         for name, value in feeds.items()],
        [helper.make_tensor_value_info(name, arrays[next(iter(feeds))].data_type, None)
         for name in output_names],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=11,
    )


def session(model_bytes, *, profile=None):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("session.disable_aot_function_inlining", "1")
    if profile is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    try:
        result = ort.InferenceSession(
            model_bytes, options, providers=["MLXExecutionProvider"], enable_fallback=False,
        )
    except (ort.capi.onnxruntime_pybind11_state.Fail,
            ort.capi.onnxruntime_pybind11_state.InvalidGraph) as error:
        message = str(error)
        if ("is under development" in message or "No Op registered for" in message):
            pytest.skip(f"ORT {ort.__version__} lacks the requested schema: {message}")
        raise
    result.disable_fallback()
    return result


def assert_mlx_profile(result):
    events = json.loads(Path(result.end_profiling()).read_text())
    providers = {event["args"].get("provider") for event in events
                 if event.get("cat") == "Node"}
    assert "MLXExecutionProvider" in providers
    assert "CPUExecutionProvider" not in providers


def check_reference(op, feeds, attrs, expected, *, opset):
    if onnx.defs.onnx_opset_version() < opset:
        pytest.skip(f"ONNX reference schemas stop before opset {opset}; install ONNX from main")
    outputs = ("y", "state") if len(expected) == 2 else ("y",)
    actual = ReferenceEvaluator(model(op, feeds, attrs, outputs, opset=opset)).run(None, feeds)
    for result, golden in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, golden, rtol=2e-6, atol=1e-7, equal_nan=True)
    return actual


def check_mlx(op, feeds, attrs, expected, eager, monkeypatch, tmp_path, *, opset):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    outputs = ("y", "state") if len(expected) == 2 else ("y",)
    result = session(model(op, feeds, attrs, outputs, opset=opset).SerializeToString(),
                     profile=tmp_path / op)
    actual = result.run(None, feeds)
    for value, golden in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, golden, rtol=2e-5, atol=1e-6, equal_nan=True)
    assert_mlx_profile(result)
    return actual
