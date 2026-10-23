"""Strict Unique contracts: typed oracle, live cardinality, and real eager boundaries."""

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
NAMES = ["y", "indices", "inverse", "counts"]
BROKEN_UNSORTED_REFERENCE = onnx.__version__.split("+")[0] in {
    "1.22.0", "1.23.0", "1.23.1", "1.23.2",
}


def _model(dtype=np.float32, shape=None, *, axis=None, sorted=None, outputs=(0, 1, 2, 3),
           opset=26, crossing=False, initializer=None, nested=False):
    tensor_type = (initializer.data_type if initializer is not None
                   else numpy_helper.from_array(np.array(0, dtype)).data_type)
    attributes = {}
    if axis is not None:
        attributes["axis"] = axis
    if sorted is not None:
        attributes["sorted"] = sorted
    slots = [name if slot in outputs else "" for slot, name in enumerate(NAMES)]
    while not slots[-1]:
        slots.pop()
    nodes = []
    source = "x"
    if crossing:
        nodes.append(helper.make_node("Neg", ["x"], ["negated"]))
        source = "negated"
    unique_node = helper.make_node("Unique", [source], slots, **attributes)
    y_shape = [None] if axis is None else (None if shape is None else [None] * len(shape))
    output_infos = [helper.make_tensor_value_info(NAMES[slot],
                    tensor_type if slot == 0 else T.INT64, y_shape if slot == 0 else [None])
                    for slot in outputs]
    if nested:
        branch = helper.make_graph([unique_node], "unique_branch", [], output_infos)
        for level in range(1, int(nested)):
            branch = helper.make_graph([
                helper.make_node("If", ["condition"], [NAMES[i] for i in outputs],
                                 then_branch=branch, else_branch=branch),
            ], f"unique_nested_{level}", [], output_infos)
        nodes.append(helper.make_node("If", ["condition"], [NAMES[i] for i in outputs],
                                     then_branch=branch, else_branch=branch))
    else:
        nodes.append(unique_node)
    if crossing:
        nodes.append(helper.make_node("Neg", ["y"], ["restored"]))
        output_infos[0].name = "restored"
    inputs = [] if initializer is not None else [
        helper.make_tensor_value_info("x", tensor_type, shape)]
    if nested:
        inputs.append(helper.make_tensor_value_info("condition", T.BOOL, []))
    return helper.make_model(
        helper.make_graph(nodes, "unique_contract", inputs, output_infos,
                          initializer=[] if initializer is None else [initializer]),
        opset_imports=[helper.make_opsetid("", opset)], ir_version=11,
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
        providers=["CPUExecutionProvider"] if cpu else ["MLXExecutionProvider"])
    session.disable_fallback()
    return session


def _oracle(data, axis=None, sorted=None):
    """Enumerate typed slice equality, independently of numpy.unique and Rust sorting."""
    if axis is None:
        slices = data.reshape(-1, 1)
    else:
        axis %= data.ndim
        moved = np.moveaxis(data, axis, 0)
        # reshape(axis_length, -1) is ambiguous when the axis has length zero.
        width = int(np.prod(moved.shape[1:], dtype=object))
        slices = moved.reshape(moved.shape[0], width)
    representatives, first, counts, inverse = [], [], [], []
    floating = data.dtype.kind == "f" or data.dtype.name == "bfloat16"
    for index, item in enumerate(slices):
        group = None
        for candidate, previous in enumerate(representatives):
            equal = all(a == b or (axis is None and floating and np.isnan(a) and np.isnan(b))
                        for a, b in zip(item, previous, strict=True))
            if equal:
                group = candidate
                break
        if group is None:
            group = len(representatives)
            representatives.append(item.copy())
            first.append(index)
            counts.append(0)
        counts[group] += 1
        inverse.append(group)
    order = list(range(len(first)))
    if sorted != 0:
        def key(group):
            return tuple((1, 0) if floating and np.isnan(value) else (0, value)
                         for value in representatives[group])
        order.sort(key=key)
    remap = {group: rank for rank, group in enumerate(order)}
    indices = np.array([first[group] for group in order], np.int64)
    y = (data.reshape(-1)[indices] if axis is None else np.take(data, indices, axis=axis))
    return [y, indices, np.array([remap[group] for group in inverse], np.int64),
            np.array([counts[group] for group in order], np.int64)]


def _exact(actual, expected, *, payload=True):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    if payload:
        assert actual.tobytes() == expected.tobytes()


def _check(results, expected, outputs=(0, 1, 2, 3), *, payload=True):
    assert len(results) == len(outputs)
    for result, slot in zip(results, outputs, strict=True):
        _exact(result, expected[slot], payload=payload)


def _reference_check(model, data, expected, *, axis=None, sorted=None, payload=True):
    # ONNX through 1.23.2 takes original X along axis zero, not sorted Y
    # along the requested axis; it can throw, return wrong shapes, or wrong values.
    # Compare its sorted variant only. The EP's unsorted result stays byte-exact against
    # the independent oracle, and a separate test records the actual reference failure.
    if BROKEN_UNSORTED_REFERENCE and sorted == 0:
        reference_model = onnx.ModelProto()
        reference_model.CopyFrom(model)
        node = reference_model.graph.node[-1]
        for attribute in node.attribute:
            if attribute.name == "sorted":
                attribute.i = 1
        result = ReferenceEvaluator(reference_model).run(None, {"x": data})
        _check(result, _oracle(data, axis, 1), payload=payload)
        return
    result = ReferenceEvaluator(model).run(None, {"x": data})
    _check(result, expected, payload=payload)


def test_unique_versioned_reference_unsorted_defect():
    data = np.array([[2, 1, 2], [3, 1, 3]], np.int64)
    if BROKEN_UNSORTED_REFERENCE:
        model = _model(np.int64, [2, 3], sorted=0)
        with pytest.raises(IndexError, match="out of bounds"):
            ReferenceEvaluator(model).run(None, {"x": data})
    else:
        _check(ReferenceEvaluator(_model(np.int64, [2, 3], sorted=0)).run(
            None, {"x": data}), _oracle(data, sorted=0))


def _values(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        values = [True, False, True, True, False, False]
    elif dtype.kind in "iu":
        bounds = np.iinfo(dtype)
        values = [bounds.max, bounds.min, 1, bounds.max, 1, bounds.min]
        if dtype.itemsize == 8:
            values[2] = 2**53 + 1
            values[4] = 2**53 + 1
    else:
        values = [3, -1, 1, 3, 1, -1]
    return np.array(values, dtype).reshape(2, 3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("opset", [11, 13, 21, 26])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("sorted", [None, 0, 1])
def test_unique_schema_types_axis_and_order(dtype, opset, axis, sorted):
    data = _values(dtype)
    model = _model(dtype, [None, None], axis=axis, sorted=sorted, opset=opset)
    expected = _oracle(data, axis, sorted)
    _check(_session(model).run(None, {"x": data}), expected)
    _reference_check(model, data, expected, axis=axis, sorted=sorted)
    # These are ORT 1.31's registered real Unique CPU types, not the schema's full type set.
    if dtype in [np.int8, np.int64, np.float32, np.float64]:
        _check(_session(model, cpu=True).run(None, {"x": data}), expected)


@pytest.mark.parametrize("outputs", [(0,), (0, 1), (0, 2), (0, 3), (0, 1, 3),
                                    (0, 2, 3), (0, 1, 2), (0, 1, 2, 3)])
@pytest.mark.parametrize("axis", [None, -1])
@pytest.mark.parametrize("sorted", [0, 1])
def test_unique_optional_output_omissions(outputs, axis, sorted):
    data = _values(np.int64)
    model = _model(np.int64, [2, 3], outputs=outputs, axis=axis, sorted=sorted)
    _check(_session(model).run(None, {"x": data}), _oracle(data, axis, sorted), outputs)


@pytest.mark.parametrize("shape,axis", [
    ([], None), ([0], None), ([0], 0), ([0, 3], None), ([0, 3], 0), ([0, 3], 1),
    ([2, 0, 3], 0), ([2, 0, 3], 1), ([2, 0, 3], 2),
    ([0, 2, 0, 3], -1), ([2, 3, 0], 1),
])
@pytest.mark.parametrize("sorted", [0, 1])
def test_unique_scalar_and_zero_sized_slices(shape, axis, sorted):
    data = np.full(shape, 7, np.int64)
    model = _model(np.int64, shape, axis=axis, sorted=sorted)
    expected = _oracle(data, axis, sorted)
    _check(_session(model).run(None, {"x": data}), expected)
    _reference_check(model, data, expected, axis=axis, sorted=sorted)
    _check(_session(model, cpu=True).run(None, {"x": data}), expected)


@pytest.mark.parametrize("axis", [0, 1, 2, -2])
@pytest.mark.parametrize("sorted", [0, 1])
def test_unique_whole_nonleading_slices(axis, sorted):
    data = np.array([[[3, 4], [1, 2], [3, 4]],
                     [[5, 6], [2, 7], [5, 6]]], np.int64)
    model = _model(np.int64, [2, 3, 2], axis=axis, sorted=sorted)
    expected = _oracle(data, axis, sorted)
    _check(_session(model).run(None, {"x": data}), expected)
    _reference_check(model, data, expected, axis=axis, sorted=sorted)
    _check(_session(model, cpu=True).run(None, {"x": data}), expected)


@pytest.mark.parametrize("dtype,bits", [
    (np.float16, [0x8000, 0, 1, 0x7C00, 0xFC00, 0x7E11, 0xFE22, 0x7E11]),
    (np.float32, [0x80000000, 0, 1, 0x7F800000, 0xFF800000,
                  0x7FC00011, 0xFFC00022, 0x7FC00011]),
    (np.float64, [1 << 63, 0, 1, 0x7FF0000000000000, 0xFFF0000000000000,
                  0x7FF8000000000011, 0xFFF8000000000022, 0x7FF8000000000011]),
])
@pytest.mark.parametrize("axis", [None, 0])
@pytest.mark.parametrize("sorted", [0, 1])
def test_unique_signed_zeros_infinities_and_nan_payloads(dtype, bits, axis, sorted):
    unsigned = {2: np.uint16, 4: np.uint32, 8: np.uint64}[np.dtype(dtype).itemsize]
    data = np.array(bits, unsigned).view(dtype).reshape(8, 1)
    model = _model(dtype, [8, 1], axis=axis, sorted=sorted)
    expected = _oracle(data, axis, sorted)
    session = _session(model)
    _check(session.run(None, {"x": data}), expected)
    if dtype == np.float16 and axis is not None:
        finite = data.copy()
        finite[np.isnan(finite)] = 1
        _check(session.run(None, {"x": finite}), _oracle(finite, axis, sorted))
        _check(session.run(None, {"x": data}), expected)
    # NaN equivalence follows ONNX Reference/NumPy, not ORT CPU's std::map<float>.
    # Its NaN comparator is not a strict weak ordering; do not bless it as an oracle.
    # NumPy half-axis NaN placement is a separately recorded authority discrepancy.
    # It is not grounds to reject legal input or weaken the strict EP oracle above.
    if dtype != np.float16 or axis is None:
        _reference_check(model, data, expected, axis=axis, sorted=sorted, payload=False)


def test_unique_numpy234_float16_axis_nan_order_dispute():
    if np.__version__ == "2.3.4":
        data = np.array([0x8000, 0, 1, 0x7C00, 0xFC00, 0x7E11, 0xFE22, 0x7E11],
                        np.uint16).view(np.float16).reshape(8, 1)
        model = _model(np.float16, [8, 1], axis=0, sorted=1)
        result = ReferenceEvaluator(model).run(None, {"x": data})
        reference_order = np.array([5, 6, 7, 4, 0, 2, 3], np.int64)
        expected_reference = [
            np.take(data, reference_order, axis=0), reference_order,
            np.array([4, 4, 5, 6, 3, 0, 1, 2], np.int64),
            np.array([1, 1, 1, 1, 2, 1, 1], np.int64),
        ]
        _check(result, expected_reference)
        expected = _oracle(data, 0, 1)
        np.testing.assert_array_equal(expected[1], [4, 0, 2, 3, 5, 6, 7])
        np.testing.assert_array_equal(expected[2], [1, 1, 2, 3, 0, 4, 5, 6])
        np.testing.assert_array_equal(expected[3], [1, 2, 1, 1, 1, 1, 1])


def test_unique_exact_double_neighbors():
    data = np.array([1, 1 + np.finfo(np.float64).eps, 1], np.float64)
    model = _model(np.float64, [3])
    _check(_session(model).run(None, {"x": data}), _oracle(data))


@pytest.mark.parametrize("sorted", [0, 1])
def test_unique_nan_slice_ties_compare_entire_rows(sorted):
    data = np.array([[np.nan, 3], [np.nan, 1], [0, 2], [-0.0, 2], [np.nan, 1]], np.float64)
    model = _model(np.float64, [5, 2], axis=0, sorted=sorted)
    expected = _oracle(data, 0, sorted)
    _check(_session(model).run(None, {"x": data}), expected)
    _reference_check(model, data, expected, axis=0, sorted=sorted, payload=False)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("axis", [None, -1])
def test_unique_exact_initializer(dtype, axis):
    data = _values(dtype)
    model = _model(dtype, [2, 3], initializer=numpy_helper.from_array(data, name="x"),
                   axis=axis, sorted=0)
    _check(_session(model).run(None, {}), _oracle(data, axis, 0))


def test_unique_bfloat16_schema11_is_not_legal():
    bits = np.array([0x8000, 0, 0x3F80, 0x3F80, 1], np.uint16)
    tensor = T(name="x", data_type=T.BFLOAT16, dims=[5], raw_data=bits.tobytes())
    model = _model(initializer=tensor)
    with pytest.raises(Exception, match="(bfloat16|BFLOAT16|Type Error|not.*supported)"):
        _session(model)


def test_unique_live_low_precision_producer():
    model = _model(np.float16, [None, None], sorted=0)
    model.graph.input[0].type.tensor_type.elem_type = T.FLOAT
    model.graph.node.insert(0, helper.make_node("Cast", ["x"], ["rounded"], to=T.FLOAT16))
    model.graph.node[1].input[0] = "rounded"
    session = _session(model)
    for data in [np.zeros((2, 3), np.float32), _values(np.float32),
                 np.full((2, 3), 1e-9, np.float32), np.ones((2, 3), np.float32)]:
        _check(session.run(None, {"x": data}), _oracle(data.astype(np.float16), sorted=0))


@pytest.mark.parametrize("axis", [None, 0, -1])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.uint64])
def test_unique_live_unknown_rank_and_cardinality(axis, eager, dtype, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _model(dtype, shape=None, axis=axis, sorted=0)
    session = _session(model)
    for shape, payload in [
        ((2, 3), [0] * 6), ((2, 3), [1, 2, 1, 2, 1, 2]),
        ((2, 3), [6, 5, 4, 3, 2, 1]), ((2, 3), [1] * 6),
        ((0, 3), []), ((3, 0), []), ((2, 3), [0] * 6),
        ((3,), [3, 1, 3]), ((), [7]), ((2, 3), [2, 3, 2, 3, 2, 3]),
    ]:
        data = np.array(payload, dtype).reshape(shape)
        if axis is not None and data.ndim == 0:
            with pytest.raises(Exception, match="Unique.*axis"):
                session.run(None, {"x": data})
        else:
            _check(session.run(None, {"x": data}), _oracle(data, axis, 0))


@pytest.mark.parametrize("crossing,nested", [(False, False), (True, False), (False, True),
                                           (True, True), (True, 2)])
def test_unique_route_workload(crossing, nested):
    model = _model(shape=[None, None], sorted=0, crossing=crossing, nested=nested)
    session = _session(model)
    for iteration, data in enumerate([
        np.zeros((2, 3), np.float32), _values(np.float32), np.ones((2, 3), np.float32),
        np.arange(6, dtype=np.float32).reshape(2, 3), np.empty((0, 3), np.float32),
        _values(np.float32),
    ]):
        feeds = {"x": data}
        if nested:
            feeds["condition"] = np.array(iteration % 2 == 0)
        expected = _oracle(-data if crossing else data, sorted=0)
        if crossing:
            expected[0] = -expected[0]
        _check(session.run(None, feeds), expected)


def test_unique_actual_routes_and_compiled_neighbor_cache(tmp_path):
    trace = tmp_path / "unique_trace.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "unique_route_workload", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0",
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0",
             "ONNXRUNTIME_EP_MLX_NO_PREFILL_COMPILE": "0"},
        check=True, timeout=120)
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path"]
    eager = [event for event in paths
             if set(event.get("ops", "").split(",")) & {"Unique", "If"}]
    assert len(eager) == 30
    assert all(event["path"] == "eager" for event in eager)
    neighbors = [event for event in paths if "Neg" in event.get("ops", "").split(",")]
    assert neighbors
    assert any(event["path"] == "general" and event["cache"] == "HIT" for event in neighbors)
    assert all(not set(event.get("ops", "").split(",")) & {"Unique", "If"}
               for event in neighbors)


@pytest.mark.parametrize("axis", [-(2**63), -3, 2, 2**63 - 1])
def test_unique_full_width_invalid_axis_recovers(axis):
    session = _session(_model(shape=None, axis=axis))
    with pytest.raises(Exception, match="Unique.*axis"):
        session.run(None, {"x": np.zeros((2, 3), np.float32)})
    # The same attribute can be valid at a larger rank only for small invalid axes.
    if axis in [-3, 2]:
        data = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        _check(session.run(None, {"x": data}), _oracle(data, axis))


def test_unique_invalid_geometry_then_valid_same_session():
    session = _session(_model(shape=None))
    good = _values(np.float32)
    _check(session.run(None, {"x": good}), _oracle(good))
    with pytest.raises(Exception, match="MLX.*(dimension|Unique)"):
        session.run(None, {"x": np.empty((0, 2**32), np.float32)})
    with pytest.raises(Exception, match="(Unexpected input data type|expected)"):
        session.run(None, {"x": good.astype(np.int64)})
    good[:] = 7
    _check(session.run(None, {"x": good}), _oracle(good))


def test_unique_schema_history_and_rejected_forms():
    histories = [schema for schema in onnx.defs.get_all_schemas_with_history()
                 if schema.name == "Unique" and schema.domain == ""]
    assert [schema.since_version for schema in histories if schema.since_version <= 27] == [11]
    schema = next(schema for schema in histories if schema.since_version == 11)
    assert schema.min_output == 1 and schema.max_output == 4
    assert schema.attributes["sorted"].default_value.i == 1
    assert "tensor(bfloat16)" not in schema.type_constraints[0].allowed_type_strs
    for dtype in [T.STRING, T.COMPLEX64, T.COMPLEX128]:
        model = _model(shape=[2])
        model.graph.input[0].type.tensor_type.elem_type = dtype
        model.graph.output[0].type.tensor_type.elem_type = dtype
        with pytest.raises(Exception, match="fallback|explicitly assigned|not supported|NOT_IMPLEMENTED"):
            _session(model)
    for sorted in [-1, 2, 2**63 - 1]:
        with pytest.raises(Exception, match="sorted|fallback|explicitly assigned|not supported"):
            _session(_model(shape=[2], sorted=sorted))


def test_unique_profile_is_provider_owned(tmp_path):
    session = _session(_model(shape=[2, 3], crossing=True), profile=tmp_path / "unique")
    data = _values(np.float32)
    expected = _oracle(-data)
    expected[0] = -expected[0]
    _check(session.run(None, {"x": data}), expected)
    events = json.loads(Path(session.end_profiling()).read_text())
    kernels = [event for event in events
               if event.get("cat") == "Node" and "provider" in event.get("args", {})]
    assert kernels
    assert {event["args"]["provider"] for event in kernels} == {"MLXExecutionProvider"}


def test_unique_model_schema_and_independent_oracle():
    """Pure schema/reference coverage, callable without loading any provider library."""
    for dtype, axis, sorted in itertools.product(DTYPES, [None, 0, -1], [None, 0, 1]):
        data = _values(dtype)
        model = _model(dtype, [None, None], axis=axis, sorted=sorted)
        onnx.checker.check_model(model)
        _reference_check(model, data, _oracle(data, axis, sorted), axis=axis, sorted=sorted)
