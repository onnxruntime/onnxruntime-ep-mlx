"""Sequence inputs with tensor outputs: strict placement, ownership, and live values."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper
import onnxruntime as ort
import pytest


def _model(op, dtype=T.FLOAT, *, opset=24, position_dtype=T.INT64,
           position_shape=(), **attrs):
    inputs = [helper.make_tensor_sequence_value_info("seq", dtype, None)]
    names = ["seq"]
    if op == "SequenceAt":
        inputs.append(helper.make_tensor_value_info("position", position_dtype, position_shape))
        names.append("position")
    output = helper.make_tensor_value_info(
        "y", T.INT64 if op == "SequenceLength" else dtype,
        [] if op == "SequenceLength" else None,
    )
    model = helper.make_model(
        helper.make_graph([helper.make_node(op, names, ["y"], **attrs)],
                          "sequence_input", inputs, [output]),
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=11,
    )
    return model.SerializeToString()


def _session(model, *, strict=True, providers=None, profile=None):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if strict:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    session = ort.InferenceSession(
        model, options, providers=providers or (
            ["MLXExecutionProvider"] if strict else ["CPUExecutionProvider"]),
    )
    session.disable_fallback()
    return session


@pytest.mark.parametrize("op", ["SequenceLength", "SequenceAt", "ConcatFromSequence"])
def test_sequence_boundary_preflight(op):
    attrs = {"axis": 0} if op == "ConcatFromSequence" else {}
    session = _session(_model(op, **attrs))
    for offset in range(3):
        sequence = [np.arange(3, dtype=np.float32) + offset, np.array([9 + offset], np.float32)]
        feeds = {"seq": sequence}
        if op == "SequenceAt":
            feeds["position"] = np.array(-1, np.int64)
        result = session.run(None, feeds)[0]
        expected = (np.array(2, np.int64) if op == "SequenceLength" else
                    sequence[-1] if op == "SequenceAt" else np.concatenate(sequence))
        np.testing.assert_array_equal(result, expected)


DTYPES = [T.BOOL, T.INT8, T.UINT8, T.INT16, T.UINT16, T.INT32, T.UINT32,
          T.INT64, T.UINT64, T.FLOAT16, T.FLOAT, T.DOUBLE]


def _values(dtype):
    dtype = np.dtype(helper.tensor_dtype_to_np_dtype(dtype))
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        return np.array([info.min, info.max, 0, 1], dtype=dtype)
    if dtype.kind == "b":
        return np.array([True, False, False, True])
    return np.array([np.nextafter(dtype.type(1), dtype.type(2)),
                     -0.0, np.inf, -np.inf, np.nan], dtype=dtype)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op", ["SequenceLength", "SequenceAt", "ConcatFromSequence"])
def test_sequence_numeric_types_and_live_values(dtype, op, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    attrs = {"axis": -1} if op == "ConcatFromSequence" else {}
    session = _session(_model(op, dtype, **attrs))
    values = _values(dtype)
    for sequence in [[values], [values[::-1].copy(), values], [values[:0], values, values[:2]]]:
        feeds = {"seq": sequence}
        if op == "SequenceAt":
            feeds["position"] = np.array(-len(sequence), np.int64)
        actual = session.run(None, feeds)[0]
        expected = (np.array(len(sequence), np.int64) if op == "SequenceLength" else
                    sequence[0] if op == "SequenceAt" else np.concatenate(sequence))
        np.testing.assert_array_equal(actual, expected)
        if dtype in (T.FLOAT16, T.FLOAT, T.DOUBLE) and op != "SequenceLength":
            np.testing.assert_array_equal(actual.view(f"u{actual.itemsize}"),
                                          expected.view(f"u{expected.itemsize}"))


@pytest.mark.parametrize("opset", [11, 24, 26])
@pytest.mark.parametrize("position_dtype", [T.INT32, T.INT64])
def test_sequence_at_live_positions_and_geometry(opset, position_dtype):
    session = _session(_model("SequenceAt", T.INT64, opset=opset,
                              position_dtype=position_dtype))
    index_type = helper.tensor_dtype_to_np_dtype(position_dtype)
    for sequence in [
        [np.array(2**62 + 1, np.int64), np.arange(6, dtype=np.int64).reshape(2, 3)],
        [np.empty((0, 2), np.int64), np.array([-(2**62) - 1], np.int64)],
        [np.array([7, 8, 9], np.int64)],
    ]:
        for index in range(-len(sequence), len(sequence)):
            result = session.run(None, {"seq": sequence,
                                       "position": np.array(index, dtype=index_type)})[0]
            np.testing.assert_array_equal(result, sequence[index])


@pytest.mark.parametrize("new_axis,shape,axis", [
    (0, (2, 3), 0), (0, (2, 3), -1), (0, (2, 0, 3), 0),
    (0, (0,), -1), (1, (), 0), (1, (), -1), (1, (2, 3), 0),
    (1, (2, 3), 1), (1, (2, 3), -1), (1, (0, 2), -3),
])
def test_concat_sequence_axis_and_empty_geometry(new_axis, shape, axis):
    model = _model("ConcatFromSequence", new_axis=new_axis, axis=axis)
    session = _session(model)
    cpu = _session(model, strict=False)
    for length in [1, 3, 2]:
        sequence = [np.full(shape, i, np.float32) for i in range(length)]
        expected = (np.stack(sequence, axis=axis) if new_axis
                    else np.concatenate(sequence, axis=axis))
        result = session.run(None, {"seq": sequence})[0]
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(result, cpu.run(None, {"seq": sequence})[0])


def test_sequence_length_empty_and_strings():
    for dtype, sequence in [
        (T.FLOAT, []),
        (T.STRING, [np.array(["a", "b"], dtype=object), np.array(["c"], dtype=object)]),
    ]:
        session = _session(_model("SequenceLength", dtype))
        for values in [sequence, sequence + sequence, sequence]:
            result = session.run(None, {"seq": values})[0]
            assert result.shape == () and result.dtype == np.int64
            assert result.item() == len(values)


def test_sequence_at_invalid_position_recovers():
    session = _session(_model("SequenceAt"))
    sequence = [np.array([2, 3], np.float32)]
    for position in [1, -2, np.iinfo(np.int64).min, np.iinfo(np.int64).max]:
        with pytest.raises(Exception, match="SequenceAt.*position.*out of range"):
            session.run(None, {"seq": sequence, "position": np.array(position, np.int64)})
        np.testing.assert_array_equal(
            session.run(None, {"seq": sequence, "position": np.array(0, np.int64)})[0],
            sequence[0],
        )


@pytest.mark.parametrize("new_axis,sequence,error", [
    (0, [], "must not be empty"),
    (0, [np.ones((2, 3), np.float32), np.ones((3, 4), np.float32)], "incompatible"),
    (1, [np.ones((2, 3), np.float32), np.ones((3, 3), np.float32)], "incompatible"),
])
def test_concat_sequence_invalid_geometry_recovers(new_axis, sequence, error):
    session = _session(_model("ConcatFromSequence", axis=0, new_axis=new_axis))
    with pytest.raises(Exception, match=f"ConcatFromSequence.*{error}"):
        session.run(None, {"seq": sequence})
    good = [np.array([1, 2], np.float32)]
    np.testing.assert_array_equal(
        session.run(None, {"seq": good})[0],
        np.stack(good) if new_axis else np.concatenate(good),
    )


@pytest.mark.parametrize("op", ["SequenceLength", "SequenceAt", "ConcatFromSequence"])
def test_cpu_produced_sequence_is_consumed_by_mlx(op, tmp_path):
    attrs = {"axis": 0} if op == "ConcatFromSequence" else {}
    model = onnx.load_model_from_string(_model(op, **attrs))
    model.graph.value_info.append(model.graph.input[0])
    del model.graph.input[0]
    model.graph.input.extend([
        helper.make_tensor_value_info("x0", T.FLOAT, None),
        helper.make_tensor_value_info("x1", T.FLOAT, None),
    ])
    model.graph.node.insert(0, helper.make_node("SequenceConstruct", ["x0", "x1"], ["seq"]))
    session = _session(
        model.SerializeToString(), strict=False,
        providers=["MLXExecutionProvider", "CPUExecutionProvider"], profile=tmp_path / op,
    )
    for length in [3, 5, 3]:
        sequence = [np.arange(length, dtype=np.float32), np.array([length + 1], np.float32)]
        feeds = {"x0": sequence[0], "x1": sequence[1]}
        if op == "SequenceAt":
            feeds["position"] = np.array(-1, np.int64)
        expected = (np.array(2, np.int64) if op == "SequenceLength" else
                    sequence[-1] if op == "SequenceAt" else np.concatenate(sequence))
        np.testing.assert_array_equal(session.run(None, feeds)[0], expected)
    events = json.loads(Path(session.end_profiling()).read_text())
    nodes = [event["args"] for event in events if event.get("cat") == "Node"]
    assert any(event.get("provider") == "MLXExecutionProvider" for event in nodes)
    assert any(event.get("provider") == "CPUExecutionProvider"
               and event.get("op_name") == "SequenceConstruct" for event in nodes)


@pytest.mark.parametrize("op", ["SequenceLength", "SequenceAt", "ConcatFromSequence"])
def test_sequence_input_captured_by_if(op):
    attrs = {"axis": 0} if op == "ConcatFromSequence" else {}
    branches = []
    for index in [0, -1]:
        body = onnx.load_model_from_string(_model(op, **attrs)).graph
        del body.input[:]
        if op == "SequenceAt":
            body.initializer.append(
                numpy_helper.from_array(np.array(index, np.int64), name="position"))
        branches.append(body)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("If", ["condition"], ["result"],
                              then_branch=branches[0], else_branch=branches[1])],
            "sequence_capture",
            [helper.make_tensor_value_info("condition", T.BOOL, []),
             helper.make_tensor_sequence_value_info("seq", T.FLOAT, None)],
            [helper.make_tensor_value_info("result", T.INT64 if op == "SequenceLength"
                                           else T.FLOAT, [] if op == "SequenceLength" else None)],
        ),
        opset_imports=[helper.make_opsetid("", 24)], ir_version=11,
    )
    session = _session(model.SerializeToString())
    for length, condition in [(3, True), (2, False), (5, True), (3, False)]:
        sequence = [np.arange(length, dtype=np.float32), np.array([-length], np.float32)]
        expected = (np.array(2, np.int64) if op == "SequenceLength" else
                    sequence[0 if condition else -1] if op == "SequenceAt"
                    else np.concatenate(sequence))
        actual = session.run(None, {"seq": sequence, "condition": np.array(condition)})[0]
        np.testing.assert_array_equal(actual, expected)


def test_sequence_at_empty_and_non_scalar_position_recovers():
    session = _session(_model("SequenceAt", position_shape=None))
    for sequence, position, error in [
        ([], np.array(0, np.int64), "out of range"),
        ([np.array([1], np.float32)], np.array([0], np.int64), "must be a scalar"),
    ]:
        with pytest.raises(Exception, match=f"SequenceAt.*{error}"):
            session.run(None, {"seq": sequence, "position": position})
    np.testing.assert_array_equal(session.run(None, {
        "seq": [np.array([2], np.float32)], "position": np.array(0, np.int64),
    })[0], [2])


def test_sequence_compile_policy_reads_live_values(tmp_path):
    trace = tmp_path / "sequence.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "test_sequence_boundary_preflight", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0", "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path"]
    assert len(paths) == 9
    assert {event["ops"] for event in paths} == {
        "SequenceLength", "SequenceAt", "ConcatFromSequence",
    }
    assert all(event["path"] == "eager" and event["cache"] == "n/a" for event in paths)


def test_sequence_at_data_produced_position():
    model = onnx.load_model_from_string(_model("SequenceAt", T.INT64))
    model.graph.input[1].name = "x"
    model.graph.initializer.append(
        numpy_helper.from_array(np.array(1, np.int64), name="one"))
    model.graph.node.insert(0, helper.make_node("Add", ["x", "one"], ["position"]))
    session = _session(model.SerializeToString())
    sequence = [np.array([2**62 + 1], np.int64), np.array([-(2**62) - 1], np.int64)]
    for value in [-1, 0, -2, -1]:
        actual = session.run(None, {"seq": sequence, "x": np.array(value, np.int64)})[0]
        np.testing.assert_array_equal(actual, sequence[value + 1])


@pytest.mark.parametrize("axis", [-2, 1, -(2**63), 2**63 - 1])
def test_concat_sequence_invalid_axis(axis):
    session = _session(_model("ConcatFromSequence", axis=axis))
    with pytest.raises(Exception, match="ConcatFromSequence.*axis.*invalid"):
        session.run(None, {"seq": [np.array([1, 2], np.float32)]})


@pytest.mark.parametrize("op", ["SequenceAt", "ConcatFromSequence"])
def test_string_sequence_payload_output_stays_on_cpu(op):
    model = _model(op, T.STRING, **({"axis": 0} if op == "ConcatFromSequence" else {}))
    with pytest.raises(Exception, match="fallback to CPU EP has been explicitly disabled"):
        _session(model)
    sequence = [np.array(["a"], object), np.array(["b", "c"], object)]
    feeds = {"seq": sequence}
    if op == "SequenceAt":
        feeds["position"] = np.array(-1, np.int64)
    expected = sequence[-1] if op == "SequenceAt" else np.concatenate(sequence)
    np.testing.assert_array_equal(_session(model, strict=False).run(None, feeds)[0], expected)
