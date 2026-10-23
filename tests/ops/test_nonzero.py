"""Exact NonZero contracts; never substitute ORT CPU fallback for provider execution."""

import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
import onnxruntime as ort
import pytest


DTYPES = [np.bool_, np.int8, np.int16, np.int32, np.int64,
          np.uint8, np.uint16, np.uint32, np.uint64,
          np.float16, np.float32, np.float64]
CPU_TYPES = [np.bool_, np.uint8, np.int32, np.int64, np.float32]


def _model(dtype=np.float32, shape=None, *, opset=26, chain=False,
           crossing=False, initializer=None):
    tensor_type = numpy_helper.from_array(np.array(0, dtype)).data_type
    nodes = []
    source = "x"
    if crossing:
        nodes.append(helper.make_node("Neg", [source], ["negated"]))
        source = "negated"
    nodes.append(helper.make_node("NonZero", [source], ["indices"]))
    if chain:
        nodes.append(helper.make_node("NonZero", ["indices"], ["nested"]))
        source = "nested"
    else:
        source = "indices"
    if crossing:
        nodes.append(helper.make_node("Transpose", [source], ["transposed"], perm=[1, 0]))
        source = "transposed"
    nodes.append(helper.make_node("Identity", [source], ["y"]))
    return helper.make_model(
        helper.make_graph(
            nodes, "nonzero_contract",
            [] if initializer is not None else [
                helper.make_tensor_value_info("x", tensor_type, shape)],
            [helper.make_tensor_value_info("y", T.INT64, [None, None])],
            initializer=[] if initializer is None else [initializer],
        ), opset_imports=[helper.make_opsetid("", opset)], ir_version=11,
    )


def _session(model, *, cpu=False, profile=None):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    if not cpu:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    session = ort.InferenceSession(
        model.SerializeToString(), options,
        providers=["CPUExecutionProvider"] if cpu else ["MLXExecutionProvider"],
    )
    session.disable_fallback()
    return session


def _oracle(data):
    # Enumerate ONNX coordinates directly, including rank zero, not numpy.nonzero's scalar rule.
    coordinates = []
    for coordinate in itertools.product(*(range(extent) for extent in data.shape)):
        if data[coordinate] != 0:
            coordinates.append(coordinate)
    result = np.empty((data.ndim, len(coordinates)), np.int64)
    for column, coordinate in enumerate(coordinates):
        result[:, column] = coordinate
    return result


def _exact(actual, expected):
    assert actual.dtype == expected.dtype == np.int64
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def _values(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        bits_type, bits = {
            2: (np.uint16, [0, 0x8000, 1, 0x8001, 0x7C00, 0xFC00, 0x7E01, 0x3C00]),
            4: (np.uint32, [0, 0x80000000, 1, 0x80000001, 0x7F800000,
                           0xFF800000, 0x7FC00001, 0x3F800000]),
            8: (np.uint64, [0, 1 << 63, 1, (1 << 63) + 1, 0x7FF0000000000000,
                           0xFFF0000000000000, 0x7FF8000000000001, 0x3FF0000000000000]),
        }[dtype.itemsize]
        return np.array(bits, bits_type).view(dtype).reshape(2, 4)
    if dtype.kind == "b":
        values = [False, True, True, False, True, False, True, False]
    else:
        bounds = np.iinfo(dtype)
        values = [0, bounds.max, bounds.min, 1, 0, bounds.max, 1, 0]
        if dtype.itemsize == 8:
            values[3] = 2**53 + 1
            values[6] = -(2**53 + 1) if dtype.kind == "i" else 2**63 + 1
    return np.array(values, dtype).reshape(2, 4)


@pytest.mark.parametrize("opset", [9, 12, 13, 21, 23, 24, 25, 26])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eager", [False, True])
def test_nonzero_exact_schema_types(dtype, opset, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = _values(dtype)
    model = _model(dtype, [None, None], opset=opset)
    expected = _oracle(data)
    _exact(_session(model).run(None, {"x": data})[0], expected)
    _exact(ReferenceEvaluator(model).run(None, {"x": data})[0], expected)
    if dtype in CPU_TYPES:
        _exact(_session(model, cpu=True).run(None, {"x": data})[0], expected)


@pytest.mark.parametrize("opset", [9, 13, 26])
@pytest.mark.parametrize("value", [0.0, -0.0, 1.0, float("nan"), float("inf")])
def test_nonzero_scalar_onnx_contract_and_ort_disagreement(opset, value):
    data = np.array(value, np.float32)
    model = _model(shape=[], opset=opset)
    expected = _oracle(data)
    _exact(_session(model).run(None, {"x": data})[0], expected)
    # ONNX 1.22's reference delegates scalar handling to NumPy; 1.23.2 fixes it.
    # The independent ONNX-coordinate oracle above does not depend on that reference version.
    # Explicit versioned disagreement, not a reason to relax the ONNX shape assertion.
    if ort.__version__.startswith("1.31."):
        cpu = _session(model, cpu=True).run(None, {"x": data})[0]
        assert cpu.shape == (1, expected.shape[1])
        _exact(cpu, np.zeros((1, expected.shape[1]), np.int64))


@pytest.mark.parametrize("shape", [[0], [0, 2], [2, 0, 3], [1, 2, 0, 4]])
def test_nonzero_zero_sized_axes(shape):
    data = np.empty(shape, np.int64)
    model = _model(np.int64, shape)
    expected = _oracle(data)
    _exact(_session(model).run(None, {"x": data})[0], expected)
    _exact(_session(model, cpu=True).run(None, {"x": data})[0], expected)
    _exact(ReferenceEvaluator(model).run(None, {"x": data})[0], expected)


@pytest.mark.parametrize("opset", [13, 21, 26])
def test_nonzero_exact_bfloat16_initializer(opset):
    bits = np.array([0, 0x8000, 1, 0x8001, 0x7F80, 0xFF80, 0x7FC1, 0x3F80], np.uint16)
    tensor = T(name="x", data_type=T.BFLOAT16, dims=[2, 4], raw_data=bits.tobytes())
    model = _model(shape=[2, 4], opset=opset, initializer=tensor)
    expected = np.array([[0, 0, 1, 1, 1, 1], [2, 3, 0, 1, 2, 3]], np.int64)
    _exact(_session(model).run(None, {})[0], expected)
    _exact(ReferenceEvaluator(model).run(None, {})[0], expected)


def test_nonzero_live_bfloat16_producer():
    model = _model(shape=[None, None])
    model.graph.node.insert(0, helper.make_node("Cast", ["x"], ["bf16"], to=T.BFLOAT16))
    model.graph.node[1].input[0] = "bf16"
    session = _session(model)
    for data in [np.zeros((2, 4), np.float32), _values(np.float32),
                 np.ones((2, 4), np.float32), np.zeros((2, 4), np.float32)]:
        # Values rounded to zero by the producer must not be resurrected by NonZero.
        import ml_dtypes
        _exact(session.run(None, {"x": data})[0], _oracle(data.astype(ml_dtypes.bfloat16)))


@pytest.mark.parametrize("chain,crossing", [(False, False), (True, False), (False, True)])
def test_nonzero_route_workload(chain, crossing):
    model = _model(shape=None, chain=chain, crossing=crossing)
    session = _session(model)
    for shape, payload in [
        ((2, 3), [0, 0, 0, 0, 0, 0]),
        ((2, 3), [1, 0, 0, 0, 0, 0]),
        ((2, 3), [0, 0, 0, 0, 0, 7]),
        ((2, 3), [1, 2, 3, 4, 5, 6]),
        ((2, 3), [0, 0, 0, 0, 0, 0]),
        ((0, 3), []),
        ((), [1]),
        ((3,), [0, 1, 0]),
        ((2, 3), [0, 0, 0, 0, 2, 0]),
    ]:
        data = np.array(payload, np.float32).reshape(shape)
        expected = _oracle(data)
        if chain:
            expected = _oracle(expected)
        if crossing:
            expected = expected.T.copy()
        _exact(session.run(None, {"x": data})[0], expected)


def test_nonzero_actual_eager_and_adjacent_compiled_routes(tmp_path):
    trace = tmp_path / "nonzero_trace.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "nonzero_route_workload", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0",
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0",
             "ONNXRUNTIME_EP_MLX_NO_PREFILL_COMPILE": "0"},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path"]
    nonzero = [event for event in paths if "NonZero" in event.get("ops", "").split(",")]
    assert len(nonzero) == 27
    assert all(event["path"] == "eager" for event in nonzero)
    assert any(event.get("ops", "").split(",").count("NonZero") == 2 for event in nonzero)
    for operation in ["Neg", "Transpose"]:
        adjacent = [event for event in paths if operation in event.get("ops", "").split(",")]
        assert adjacent
        assert any(event["path"] == "general" and event["cache"] == "HIT" for event in adjacent)
        assert all("NonZero" not in event.get("ops", "").split(",") for event in adjacent)


def test_nonzero_invalid_runtime_geometry_recovers():
    session = _session(_model(shape=None))
    good = np.array([[0, 2], [3, 0]], np.float32)
    _exact(session.run(None, {"x": good})[0], _oracle(good))
    with pytest.raises(Exception, match="MLX.*(dimension|NonZero)"):
        session.run(None, {"x": np.empty((0, 2**32), np.float32)})
    with pytest.raises(Exception, match="(Unexpected input data type|expected)"):
        session.run(None, {"x": good.astype(np.int64)})
    good[:] = [[4, 0], [0, 0]]
    _exact(session.run(None, {"x": good})[0], _oracle(good))


def test_nonzero_profile_is_provider_owned(tmp_path):
    model = _model(shape=[2, 4], chain=True, crossing=True)
    session = _session(model, profile=tmp_path / "nonzero")
    data = _values(np.float32)
    _exact(session.run(None, {"x": data})[0], _oracle(_oracle(data)).T.copy())
    events = json.loads(Path(session.end_profiling()).read_text())
    kernels = [event for event in events
               if event.get("cat") == "Node" and "provider" in event.get("args", {})]
    assert kernels
    assert all(event["args"]["provider"] == "MLXExecutionProvider" for event in kernels)


def test_nonzero_schema_history_and_rejected_forms():
    histories = [schema for schema in onnx.defs.get_all_schemas_with_history()
                 if schema.name == "NonZero" and schema.domain == ""]
    assert sorted(schema.since_version for schema in histories) == [9, 13]
    for schema in histories:
        assert not schema.attributes
        assert len(schema.inputs) == len(schema.outputs) == 1
        assert schema.outputs[0].type_str == "tensor(int64)"
        assert ("tensor(bfloat16)" in schema.type_constraints[0].allowed_type_strs) == (
            schema.since_version == 13)
    for dtype in [T.STRING, T.COMPLEX64, T.COMPLEX128]:
        model = _model(shape=[2])
        model.graph.input[0].type.tensor_type.elem_type = dtype
        with pytest.raises(Exception, match="fallback|explicitly assigned|not supported|NOT_IMPLEMENTED"):
            _session(model)
    model = _model(shape=[2], opset=9)
    model.graph.input[0].type.tensor_type.elem_type = T.BFLOAT16
    with pytest.raises(Exception):
        _session(model)
    for mutate in [
        lambda model: model.graph.node[0].attribute.append(helper.make_attribute("axis", 0)),
        lambda model: setattr(model.graph.output[0].type.tensor_type, "elem_type", T.INT32),
    ]:
        model = _model(shape=[2])
        mutate(model)
        with pytest.raises(Exception):
            _session(model)
