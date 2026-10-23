"""Slice schema versions, exact controls, and live value specialization."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort
from onnx import AttributeProto, TensorProto as T, helper, numpy_helper
import pytest


def _model(dtype, shape, starts, ends, axes=None, steps=None, *, opset=26,
           index_dtype=np.int64, live=False):
    inputs = [helper.make_tensor_value_info("data", dtype, shape)]
    names = ["data"]
    initializers = []
    attrs = {}
    if opset < 10:
        attrs = {"starts": starts, "ends": ends}
        if axes is not None:
            attrs["axes"] = axes
    else:
        for name, values in zip(("starts", "ends", "axes", "steps"), (starts, ends, axes, steps)):
            if values is None:
                names.append("")
                continue
            names.append(name)
            if live:
                inputs.append(helper.make_tensor_value_info(
                    name, T.INT32 if index_dtype == np.int32 else T.INT64, [None]))
            else:
                initializers.append(numpy_helper.from_array(np.asarray(values, index_dtype), name=name))
        while not names[-1]:
            names.pop()
    node = helper.make_node("Slice", names, ["out"])
    node.attribute.extend(helper.make_attribute(name, values, attr_type=AttributeProto.INTS)
                          for name, values in attrs.items())
    graph = helper.make_graph(
        [node], "slice_forms", inputs,
        [helper.make_tensor_value_info("out", dtype, None)], initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=11)


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


def _expected(data, starts, ends, axes=None, steps=None):
    axes = list(range(len(starts))) if axes is None else axes
    steps = [1] * len(starts) if steps is None else steps
    result = data
    for start, end, axis, step in zip(starts, ends, axes, steps, strict=True):
        axis %= data.ndim
        dim = data.shape[axis]
        start = start + dim if start < 0 else start
        end = end + dim if end < 0 else end
        if step > 0:
            start, end = max(0, min(dim, start)), max(0, min(dim, end))
        else:
            start, end = min(max(0, start), dim - 1), max(-1, min(dim - 1, end))
        indices = np.fromiter(range(start, end, step), dtype=np.int64)
        result = np.take(result, indices, axis=axis)
    return result


@pytest.mark.parametrize("opset", [1, 9])
@pytest.mark.parametrize("starts,ends,axes", [
    ([0], [2], None), ([0, 1], [3, 4], None), ([-2], [100], [-1]),
    ([3, 0], [1, 2], [1, 0]), ([], [], None),
])
@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.uint64, np.bool_])
@pytest.mark.parametrize("eager", [False, True])
def test_slice_legacy_attributes(opset, starts, ends, axes, dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.arange(12).reshape(3, 4).astype(dtype)
    if dtype == np.bool_:
        data = np.arange(12).reshape(3, 4) % 5 == 1
    elif dtype == np.uint64:
        data += np.uint64(2**63)
    model = _model(helper.np_dtype_to_tensor_dtype(data.dtype), list(data.shape),
                   starts, ends, axes, opset=opset)
    expected = _expected(data, starts, ends, axes)
    np.testing.assert_array_equal(_session(model).run(None, {"data": data})[0], expected)
    np.testing.assert_array_equal(_session(model, cpu=True).run(None, {"data": data})[0], expected)


@pytest.mark.parametrize("opset", [10, 11, 13, 26])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("axes,steps", [(None, None), ([1], None), (None, [2]), ([-1], [-1])])
def test_slice_modern_optional_inputs_and_index_types(opset, index_dtype, axes, steps):
    starts, ends = ([3], [-100]) if steps == [-1] else ([0], [3])
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    model = _model(T.FLOAT, [3, 4], starts, ends, axes, steps, opset=opset, index_dtype=index_dtype)
    expected = _expected(data, starts, ends, axes, steps)
    np.testing.assert_array_equal(_session(model).run(None, {"data": data})[0], expected)
    np.testing.assert_array_equal(_session(model, cpu=True).run(None, {"data": data})[0], expected)


@pytest.mark.parametrize("dtype", [
    np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
    np.int64, np.uint64, np.float16, np.float32, np.float64,
])
def test_slice_exact_payload_families(dtype):
    data = np.arange(24).reshape(4, 6).astype(dtype)
    if dtype == np.bool_:
        data = np.arange(24).reshape(4, 6) % 5 == 1
    elif dtype == np.int64:
        data += np.int64(-(2**63) + 64)
    elif dtype == np.uint64:
        data += np.uint64(2**63 + 64)
    elif dtype == np.float64:
        data += 1 / 3
    model = _model(helper.np_dtype_to_tensor_dtype(data.dtype), [4, 6],
                   [5, 0], [-100, 4], [-1, 0], [-2, 2], index_dtype=np.int32)
    expected = _expected(data, [5, 0], [-100, 4], [-1, 0], [-2, 2])
    np.testing.assert_array_equal(_session(model).run(None, {"data": data})[0], expected)


def test_slice_bfloat16_payload():
    model = _model(T.BFLOAT16, [4, 6], [5], [-100], [-1], [-2], index_dtype=np.int32)
    model.graph.input[0].type.tensor_type.elem_type = T.FLOAT
    model.graph.node[0].input[0] = "narrow"
    model.graph.node[0].output[0] = "sliced"
    model.graph.node.insert(0, helper.make_node("Cast", ["data"], ["narrow"], to=T.BFLOAT16))
    model.graph.node.append(helper.make_node("Cast", ["sliced"], ["out"], to=T.FLOAT))
    model.graph.output[0].type.tensor_type.elem_type = T.FLOAT
    data = np.arange(24, dtype=np.float32).reshape(4, 6) / 2
    np.testing.assert_array_equal(
        _session(model).run(None, {"data": data})[0],
        _expected(data, [5], [-100], [-1], [-2]),
    )


@pytest.mark.parametrize("form", ["legacy", "int32", "live", "huge_step"])
def test_slice_new_forms_strict_claim(form):
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    steps = [2**63 - 1] if form == "huge_step" else [1]
    model = _model(T.FLOAT, [3, 4], [0], [2], [0], steps,
                   opset=9 if form == "legacy" else 26,
                   index_dtype=np.int32 if form == "int32" else np.int64,
                   live=form == "live")
    feeds = {"data": data}
    if form == "live":
        feeds.update({name: np.array(value, np.int64) for name, value in
                      zip(("starts", "ends", "axes", "steps"), ([0], [2], [0], steps))})
    np.testing.assert_array_equal(
        _session(model).run(None, feeds)[0], _expected(data, [0], [2], [0], steps))


@pytest.mark.parametrize("step", [2**31 - 1, 2**31, 2**63 - 1, -(2**31), -(2**31) - 1, -(2**63)])
@pytest.mark.parametrize("shape", [(3, 5), (3, 0), (0, 5)])
def test_slice_extreme_strides_and_empty_geometry(step, shape):
    starts, ends = ([2**63 - 1], [-(2**63)]) if step < 0 else ([-(2**63)], [2**63 - 1])
    data = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    model = _model(T.INT64, list(shape), starts, ends, [-1], [step])
    np.testing.assert_array_equal(
        _session(model).run(None, {"data": data})[0], _expected(data, starts, ends, [-1], [step]))


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("eager", [False, True])
def test_slice_live_control_replay(index_dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _model(T.FLOAT, [None, None], [0], [4], [0], [2], index_dtype=index_dtype, live=True)
    session = _session(model)
    cases = [(0, 4, 0, 2), (0, 4, 0, 2), (3, np.iinfo(index_dtype).min, 1, -1),
             (3, np.iinfo(index_dtype).min, 1, -1), (0, 4, 0, 2)]
    for invocation, values in enumerate(cases):
        data = np.arange(24, dtype=np.float32).reshape(4, 6) + invocation
        feeds = {"data": data}
        feeds.update({name: np.array([value], index_dtype)
                      for name, value in zip(("starts", "ends", "axes", "steps"), values)})
        actual = session.run(None, feeds)[0]
        np.testing.assert_array_equal(actual, _expected(data, *([value] for value in values)))


@pytest.mark.parametrize("eager", [False, True])
def test_slice_intermediate_controls(eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _model(T.FLOAT, [4, 6], [0], [4], [0], [2], live=True)
    model.graph.initializer.extend([
        numpy_helper.from_array(np.array([0], np.int64), name="zero"),
        numpy_helper.from_array(np.array([1], np.int64), name="one"),
    ])
    producers = []
    for name in ("starts", "ends", "axes", "steps"):
        for value in model.graph.input:
            if value.name == name:
                value.name = f"{name}_input"
        producers.append(helper.make_node(
            "Slice", [f"{name}_input", "zero", "one", "zero", "one"], [name]))
    nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend([*producers, *nodes])
    session = _session(model)
    for invocation, values in enumerate([(0, 4, 0, 2), (3, -100, 1, -1), (0, 4, 0, 2)]):
        data = np.arange(24, dtype=np.float32).reshape(4, 6) + invocation
        feeds = {"data": data, **{f"{name}_input": np.array([value, 99], np.int64) for name, value in
                 zip(("starts", "ends", "axes", "steps"), values)}}
        np.testing.assert_array_equal(
            session.run(None, feeds)[0], _expected(data, *([value] for value in values)))


def test_slice_unknown_rank_and_changing_geometry():
    model = _model(T.FLOAT, None, [0], [2], [-1], [1], live=True)
    session = _session(model)
    for invocation, shape in enumerate([(4, 6), (12,), (2, 3, 4), (4, 6)]):
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + invocation
        feeds = {"data": data, **{name: np.array(value, np.int64) for name, value in
                 zip(("starts", "ends", "axes", "steps"), ([0], [2], [-1], [1]))}}
        np.testing.assert_array_equal(session.run(None, feeds)[0], data[..., :2])


def test_slice_live_control_actual_routes(tmp_path):
    trace = tmp_path / "slice_trace.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "slice_live_control_replay or slice_intermediate_controls",
         "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path" and "Slice" in event["args"].get("ops", "")]
    general = [event for event in paths if event["path"] == "general"]
    assert [event["cache"] for event in general] == ["MISS", "HIT", "RETRACE", "HIT", "HIT"] * 2
    assert sum(event["path"] == "eager" for event in paths) == 16


@pytest.mark.parametrize("bad", [
    ([0, 1], [2], [0], [1]), ([0], [2], [0, 1], [1]),
    ([0], [2], [0], [1, 1]), ([0], [2], [0], [0]),
    ([0], [2], [2], [1]), ([0], [2], [-3], [1]),
    ([0, 0], [2, 2], [1, -1], [1, 1]),
])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_slice_invalid_live_controls_recover(bad, index_dtype):
    model = _model(T.FLOAT, [3, 4], [0], [2], [0], [1], index_dtype=index_dtype, live=True)
    session = _session(model)
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    good = {"data": data, **{name: np.array(value, index_dtype) for name, value in
            zip(("starts", "ends", "axes", "steps"), ([0], [2], [0], [1]))}}
    np.testing.assert_array_equal(session.run(None, good)[0], data[:2])
    invalid = {"data": data, **{name: np.array(value, index_dtype) for name, value in
               zip(("starts", "ends", "axes", "steps"), bad)}}
    with pytest.raises(Exception, match="MLX Slice"):
        session.run(None, invalid)
    good["data"] = data + 20
    np.testing.assert_array_equal(session.run(None, good)[0], good["data"][:2])


def test_slice_unknown_rank_rejects_scalar_and_recovers():
    model = _model(T.FLOAT, None, [], [], [], [], live=True)
    session = _session(model)
    feeds = {name: np.array([], np.int64) for name in ("starts", "ends", "axes", "steps")}
    feeds["data"] = np.array(1, np.float32)
    with pytest.raises(Exception, match="cannot slice a scalar"):
        session.run(None, feeds)
    feeds["data"] = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(session.run(None, feeds)[0], feeds["data"])


def test_slice_wide_input_geometry_stays_on_cpu():
    model = _model(T.FLOAT, [0, 2**31 + 7], [0], [1], [1])
    with pytest.raises(Exception, match="fallback to CPU EP has been explicitly disabled"):
        _session(model)
    data = np.empty((0, 2**31 + 7), np.float32)
    assert _session(model, cpu=True).run(None, {"data": data})[0].shape == (0, 1)
