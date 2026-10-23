"""Bernoulli output representation must not replace its probability sampler."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from onnx import TensorProto as T, helper, load_model_from_string, numpy_helper
import onnxruntime as ort
import pytest


PROBABILITY_TYPES = [T.FLOAT, T.FLOAT16, T.BFLOAT16]


def _model(probability_type, *, output_type=T.DOUBLE, opset=22, seed=41.0,
           shape=None, explicit_dtype=True, initializer=False, nested=False):
    source_type = T.FLOAT if probability_type == T.BFLOAT16 else probability_type
    inputs = [] if initializer else [helper.make_tensor_value_info("p", source_type, shape)]
    nodes = []
    initializers = []
    probability = "p"
    if initializer:
        values = np.resize(np.array([0, 1, 0.25, 0.5, 0.75], np.float32), shape)
        if probability_type == T.BFLOAT16:
            bits = (values.view(np.uint32) >> 16).astype(np.uint16)
            initializers.append(helper.make_tensor("p", T.BFLOAT16, shape,
                                                  bits.tobytes(), raw=True))
        else:
            initializers.append(numpy_helper.from_array(
                values.astype(helper.tensor_dtype_to_np_dtype(probability_type)), name="p",
            ))
    elif probability_type == T.BFLOAT16:
        probability = "bf16_p"
        nodes.append(helper.make_node("Cast", ["p"], [probability], to=T.BFLOAT16))
    attrs = {"dtype": output_type} if explicit_dtype else {}
    if seed is not None:
        attrs["seed"] = seed
    if nested:
        branches = [
            helper.make_graph(
                [helper.make_node("Bernoulli", [probability], [name], **attrs)],
                name, [], [helper.make_tensor_value_info(name, output_type, shape)],
            )
            for name in ["then", "else"]
        ]
        inputs.append(helper.make_tensor_value_info("condition", T.BOOL, []))
        nodes.append(helper.make_node("If", ["condition"], ["draw"],
                                      then_branch=branches[0], else_branch=branches[1]))
    else:
        nodes.append(helper.make_node("Bernoulli", [probability], ["draw"], **attrs))
    result_type = output_type
    result = "draw"
    if output_type == T.BFLOAT16:
        result_type = T.FLOAT
        result = "y"
        nodes.append(helper.make_node("Cast", ["draw"], [result], to=T.FLOAT))
    return helper.make_model(
        helper.make_graph(nodes, "bernoulli_output", inputs,
                          [helper.make_tensor_value_info(result, result_type, shape)],
                          initializer=initializers),
        opset_imports=[helper.make_opsetid("", opset)], ir_version=11,
    ).SerializeToString()


def _session(model, *, profile=None):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    session = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"],
                                   enable_fallback=False)
    session.disable_fallback()
    return session


def _assert_profile(session):
    path = Path(session.end_profiling())
    events = json.loads(path.read_text())
    path.unlink()
    kernels = [event["args"] for event in events
               if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time")]
    assert kernels
    assert all(event.get("provider") == "MLXExecutionProvider" for event in kernels), kernels


def _probabilities(probability_type, shape, values):
    dtype = np.float16 if probability_type == T.FLOAT16 else np.float32
    return np.resize(np.asarray(values, dtype=dtype), shape)


@pytest.mark.parametrize("opset,probability_type", [
    (opset, dtype) for opset in [15, 22, 26]
    for dtype in (PROBABILITY_TYPES if opset >= 22 else [T.FLOAT, T.FLOAT16])
])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("shape", [(), (256,), (0, 3)])
def test_explicit_double_output(probability_type, opset, eager, shape, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(probability_type, opset=opset, shape=shape),
                       profile=tmp_path / "double")
    probabilities = _probabilities(probability_type, shape, [0, 1, 0.25, 0.5, 0.75])
    before = probabilities.tobytes()
    first, second = [session.run(None, {"p": probabilities})[0] for _ in range(2)]
    assert first.shape == shape
    assert first.dtype == np.float64
    assert np.isin(first, [0.0, 1.0]).all()
    np.testing.assert_array_equal(first[probabilities == 0], 0)
    np.testing.assert_array_equal(first[probabilities == 1], 1)
    np.testing.assert_array_equal(first, second)
    assert probabilities.tobytes() == before
    _assert_profile(session)


@pytest.mark.parametrize("probability_type", PROBABILITY_TYPES)
@pytest.mark.parametrize("output_type", [T.DOUBLE, None])
@pytest.mark.parametrize("initializer", [False, True])
def test_default_and_explicit_output(probability_type, output_type, initializer, tmp_path):
    dtype = probability_type if output_type is None else output_type
    session = _session(_model(probability_type, output_type=dtype, shape=[256],
                              explicit_dtype=output_type is not None, initializer=initializer),
                       profile=tmp_path / "output")
    feeds = {} if initializer else {
        "p": _probabilities(probability_type, (256,), [0, 1, 0.25, 0.5, 0.75]),
    }
    first, second = [session.run(None, feeds)[0] for _ in range(2)]
    expected_type = T.FLOAT if dtype == T.BFLOAT16 else dtype
    assert first.dtype == helper.tensor_dtype_to_np_dtype(expected_type)
    assert first.shape == (256,)
    assert np.isin(first, [0, 1]).all()
    np.testing.assert_array_equal(first, second)
    _assert_profile(session)


@pytest.mark.parametrize("probability_type", PROBABILITY_TYPES)
@pytest.mark.parametrize("seed", [None, 41.0, 41.75])
@pytest.mark.parametrize("eager", [False, True])
def test_live_probability_and_key(probability_type, seed, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(probability_type, seed=seed))
    for value in [0, 1, 0, 1]:
        probabilities = _probabilities(probability_type, (256,), [value])
        np.testing.assert_array_equal(session.run(None, {"p": probabilities})[0],
                                      np.full(256, value, np.float64))
    probabilities = _probabilities(probability_type, (256,), [0.5])
    outputs = [session.run(None, {"p": probabilities})[0].copy() for _ in range(4)]
    assert all(output.dtype == np.float64 and np.isin(output, [0, 1]).all()
               for output in outputs)
    assert all(np.array_equal(left, right) == (seed is not None)
               for left, right in zip(outputs, outputs[1:]))
    empty = session.run(None, {"p": _probabilities(probability_type, (0, 3), [0.5])})[0]
    assert empty.shape == (0, 3) and empty.dtype == np.float64
    restored = session.run(None, {"p": probabilities})[0]
    assert restored.shape == (256,) and restored.dtype == np.float64
    if seed is not None:
        np.testing.assert_array_equal(restored, outputs[0])
        truncated = _session(_model(probability_type, seed=41.0))
        np.testing.assert_array_equal(restored, truncated.run(None, {"p": probabilities})[0])


@pytest.mark.parametrize("seed", [-1.0, float("nan"), float("inf"), -float("inf"), 2.0**64])
def test_invalid_seed_is_declined(seed):
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail,
                       match="CPU EP.*fallback.*disabled"):
        _session(_model(T.FLOAT, seed=seed))


@pytest.mark.parametrize("opset", [15, 21])
@pytest.mark.parametrize("initializer", [False, True])
def test_bfloat_probability_before_22_is_rejected(opset, initializer):
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidGraph,
                       match="Type.*bfloat16.*invalid"):
        _session(_model(T.BFLOAT16, opset=opset, initializer=initializer, shape=[256]))


def test_double_probabilities_stay_declined():
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail,
                       match="CPU EP.*fallback.*disabled"):
        _session(_model(T.DOUBLE))


def test_dtype_must_match_output():
    proto = load_model_from_string(_model(T.FLOAT, shape=[256]))
    proto.graph.output[0].type.tensor_type.elem_type = T.FLOAT
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail,
                       match="Type Error.*does not match expected type"):
        _session(proto.SerializeToString())


def test_illegal_probability_type_is_rejected():
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidGraph,
                       match="Type Error.*invalid"):
        _session(_model(T.INT32))


def test_unsupported_output_type_stays_declined():
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail,
                       match="CPU EP.*fallback.*disabled"):
        _session(_model(T.FLOAT, output_type=T.STRING))


@pytest.mark.parametrize("probability_type", PROBABILITY_TYPES)
def test_seed_zero_and_typed_casts(probability_type):
    proto = load_model_from_string(_model(probability_type, seed=0.0, shape=[256]))
    for name, dtype in [("single", T.FLOAT), ("integer", T.INT64)]:
        proto.graph.node.append(helper.make_node("Cast", ["draw"], [name], to=dtype))
        proto.graph.output.append(helper.make_tensor_value_info(name, dtype, [256]))
    session = _session(proto.SerializeToString())
    probabilities = _probabilities(probability_type, (256,), [0, 1, 0.5])
    double, single, integer = session.run(None, {"p": probabilities})
    assert double.dtype == np.float64
    assert single.dtype == np.float32
    assert integer.dtype == np.int64
    np.testing.assert_array_equal(double, single)
    np.testing.assert_array_equal(double, integer)


@pytest.mark.parametrize("probability_type", PROBABILITY_TYPES)
def test_double_payload_inside_if_stays_declined(probability_type):
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail,
                       match="CPU EP.*fallback.*disabled"):
        _session(_model(probability_type, seed=None, shape=[256], nested=True))


@pytest.mark.parametrize("probability_type", PROBABILITY_TYPES)
@pytest.mark.parametrize("seed", [None, 41.0])
def test_native_probability_route_case(probability_type, seed, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE", "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PREFILL_COMPILE", "0")
    session = _session(_model(probability_type, output_type=probability_type,
                              seed=seed, shape=[256], explicit_dtype=False))
    probabilities = _probabilities(probability_type, (256,), [0.5])
    outputs = [session.run(None, {"p": probabilities})[0].copy() for _ in range(4)]
    assert all(np.isin(output, [0, 1]).all() for output in outputs)
    assert all(np.array_equal(left, right) == (seed is not None)
               for left, right in zip(outputs, outputs[1:]))


def test_double_output_actual_routes(tmp_path):
    trace = tmp_path / "bernoulli-double.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "test_live_probability_and_key", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path"
             and "Bernoulli" in event.get("args", {}).get("ops", "").split(",")]
    assert len(paths) == 192
    assert all(event["path"] == "eager" for event in paths), paths


def test_native_probability_actual_routes(tmp_path):
    trace = tmp_path / "bernoulli-native.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "test_native_probability_route_case", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path"
             and "Bernoulli" in event.get("args", {}).get("ops", "").split(",")]
    assert sum(event["path"] == "eager" for event in paths) == 12
    compiled = [event for event in paths if event["path"] == "general"]
    assert len(paths) == 24 and len(compiled) == 12
    assert sum(event["cache"] == "MISS" for event in compiled) == 3
    assert sum(event["cache"] == "HIT" for event in compiled) == 9
