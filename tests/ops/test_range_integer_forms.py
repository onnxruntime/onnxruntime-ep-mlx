"""Full-width integer Range values and conservative inferred-count contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort
from onnx import defs, helper, numpy_helper
import pytest


def _model(start, limit, delta, *, vector_input=None, dtype=np.int64):
    initializers = [
        numpy_helper.from_array(
            np.array([value] if vector_input == name else value, dtype), name=name)
        for name, value in zip(("start", "limit", "delta"), (start, limit, delta))
    ]
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Range", ["start", "limit", "delta"], ["range"]),
             helper.make_node("Add", ["range", "offset"], ["out"])],
            "range_int64",
            [helper.make_tensor_value_info("offset", initializers[0].data_type, [])],
            [helper.make_tensor_value_info("out", initializers[0].data_type, [None])],
            initializer=initializers,
        ), opset_imports=[helper.make_opsetid("", 26)], ir_version=11,
    )


def _session(model, *, cpu=False):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    if not cpu:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model.SerializeToString(), options,
        providers=["CPUExecutionProvider"] if cpu else ["MLXExecutionProvider"],
    )
    session.disable_fallback()
    return session


@pytest.mark.parametrize("dtype,start,limit,delta", [
    (np.int64, 2**53 + 1, 2**53 + 5, 1),
    (np.int64, -(2**53) - 5, -(2**53) - 1, 1),
    (np.int64, 2**53 + 5, 2**53 + 1, -1),
    (np.int64, -(2**53) - 1, -(2**53) - 5, -1),
    (np.int32, 2**24 + 1, 2**24 + 5, 1),
    (np.int32, -(2**24) - 5, -(2**24) - 1, 1),
    (np.int32, 2**24 + 5, 2**24 + 1, -1),
    (np.int32, -(2**24) - 1, -(2**24) - 5, -1),
])
@pytest.mark.parametrize("eager", [False, True])
def test_range_integer_large_exact_values(dtype, start, limit, delta, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _model(start, limit, delta, dtype=dtype)
    expected = np.array(list(range(start, limit, delta)), dtype)
    session = _session(model)
    cpu = _session(model, cpu=True)
    for offset in [0, 3, 1, 0]:
        feeds = {"offset": np.array(offset, dtype)}
        np.testing.assert_array_equal(session.run(None, feeds)[0], expected + offset)
        np.testing.assert_array_equal(cpu.run(None, feeds)[0], expected + offset)


def test_range_integer_large_actual_routes(tmp_path):
    trace = tmp_path / "range_trace.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "range_integer_large_exact_values", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path" and "Range" in event["args"].get("ops", "")]
    assert [event["cache"] for event in paths if event["path"] == "general"] == [
        "MISS", "HIT", "HIT", "HIT"] * 8
    assert sum(event["path"] == "eager" for event in paths) == 32


@pytest.mark.parametrize("start,limit,delta", [
    (0, 8, 2), (8, 0, -2), (1, 10, 3), (10, 1, -3),
    (4, 4, -(2**63)), (4, 8, -1), (8, 4, 1),
    (-(2**53) - 2, -(2**53) + 2, 1),
])
@pytest.mark.parametrize("eager", [False, True])
def test_range_integer_empty_and_signed_progressions(start, limit, delta, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _model(start, limit, delta)
    expected = np.array(list(range(start, limit, delta)), np.int64)
    feeds = {"offset": np.array(0, np.int64)}
    np.testing.assert_array_equal(_session(model).run(None, feeds)[0], expected)
    np.testing.assert_array_equal(_session(model, cpu=True).run(None, feeds)[0], expected)


@pytest.mark.parametrize("start,limit,delta", [
    (2**63 - 7, 2**63 - 3, 1),
    (-(2**63) + 2, -(2**63) + 6, 1),
    (2**63 - 3, 2**63 - 7, -1),
    (-(2**63) + 6, -(2**63) + 2, -1),
])
@pytest.mark.parametrize("eager", [False, True])
def test_range_full_width_bounds_follow_onnx_function(start, limit, delta, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _model(start, limit, delta)
    expected = np.array(list(range(start, limit, delta)), np.int64)
    feeds = {"offset": np.array(0, np.int64)}
    np.testing.assert_array_equal(_session(model).run(None, feeds)[0], expected)

    # ORT 1.31's native Range incorrectly returns empty; its expanded ONNX function is exact.
    function = model.functions.add()
    function.CopyFrom(defs.get_schema("Range", 11).function_body)
    function.name = "RangeFunction"
    function.domain = "test.range"
    model.opset_import.add(domain=function.domain, version=1)
    model.graph.node[0].op_type = function.name
    model.graph.node[0].domain = function.domain
    np.testing.assert_array_equal(_session(model, cpu=True).run(None, feeds)[0], expected)


def test_range_inferred_integer_count_disagreement_is_unclaimed():
    start, limit, delta = -(2**63), 2**63 - 1, 2**63 - 1
    model = _model(start, limit, delta)
    with pytest.raises(Exception, match="fallback to CPU EP has been explicitly disabled"):
        _session(model)
    # ORT 1.31 infers two elements; the mathematical count is three.
    actual = _session(model, cpu=True).run(None, {"offset": np.array(0, np.int64)})[0]
    assert actual.size != len(range(start, limit, delta))


@pytest.mark.parametrize("name", ["start", "limit", "delta"])
def test_range_rejects_one_element_vector_bounds(name):
    with pytest.raises(Exception):
        _session(_model(0, 4, 1, vector_input=name))
