"""Split schema partitions, live sizes, and checked multi-output geometry."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort
from onnx import AttributeProto, TensorProto as T, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
import pytest


def _model(dtype=T.FLOAT, shape=(2, 6), sizes=None, *, axis=-1, count=2,
           opset=26, live=False, num_outputs=None):
    inputs = [helper.make_tensor_value_info("data", dtype, shape)]
    names, initializers, attrs = ["data"], [], {"axis": axis}
    if live:
        names.append("sizes")
        inputs.append(helper.make_tensor_value_info("sizes", T.INT64, None))
    elif sizes is not None and opset >= 13:
        names.append("sizes")
        initializers.append(numpy_helper.from_array(np.array(sizes, np.int64), name="sizes"))
    if num_outputs is not None:
        attrs["num_outputs"] = num_outputs
    node = helper.make_node("Split", names, [f"out{i}" for i in range(count)], **attrs)
    if sizes is not None and opset < 13:
        node.attribute.append(helper.make_attribute("split", sizes, attr_type=AttributeProto.INTS))
    return helper.make_model(
        helper.make_graph(
            [node], "split_forms", inputs,
            [helper.make_tensor_value_info(f"out{i}", dtype, None) for i in range(count)],
            initializer=initializers,
        ), opset_imports=[helper.make_opsetid("", opset)], ir_version=11,
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


def _assert_parts(actual, data, sizes, axis=-1):
    expected = np.split(data, np.cumsum(sizes[:-1]), axis=axis)
    assert len(actual) == len(expected)
    for part, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(part, wanted)


@pytest.mark.parametrize("sizes", [[0, 6], [2, 4], [6, 0]])
def test_split_version1_attributes(sizes):
    data = np.arange(12, dtype=np.float32).reshape(2, 6)
    _assert_parts(_session(_model(sizes=sizes, opset=1)).run(None, {"data": data}), data, sizes)


@pytest.mark.parametrize("opset", [2, 11, 13, 18, 26])
@pytest.mark.parametrize("sizes", [[0, 6], [2, 4], [6, 0], [0, 0]])
@pytest.mark.parametrize("axis", [1, -1])
def test_split_explicit_sizes_and_empty_parts(opset, sizes, axis):
    data = np.arange(2 * sum(sizes), dtype=np.float32).reshape(2, sum(sizes))
    model = _model(shape=data.shape, sizes=sizes, axis=axis, opset=opset)
    _assert_parts(_session(model).run(None, {"data": data}), data, sizes, axis)
    _assert_parts(_session(model, cpu=True).run(None, {"data": data}), data, sizes, axis)


@pytest.mark.parametrize("opset", [2, 11, 13, 18, 26])
@pytest.mark.parametrize("shape", [(0, 4), (2, 0), (2, 6)])
@pytest.mark.parametrize("eager", [False, True])
def test_split_equal_dynamic_geometry(opset, shape, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    model = _model(shape=[None, None], opset=opset)
    _assert_parts(_session(model).run(None, {"data": data}), data, [shape[-1] // 2] * 2)
    _assert_parts(_session(model, cpu=True).run(None, {"data": data}), data, [shape[-1] // 2] * 2)


@pytest.mark.parametrize("opset", [18, 26])
@pytest.mark.parametrize("extent,count", [
    (0, 2), (1, 2), (5, 2), (7, 3), (7, 4), (4, 3), (6, 4), (5, 1),
])
@pytest.mark.parametrize("eager", [False, True])
def test_split_num_outputs_smaller_final_part(opset, extent, count, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.arange(2 * extent, dtype=np.float32).reshape(2, extent)
    model = _model(shape=[None, None], count=count, num_outputs=count, opset=opset)
    chunk = (extent + count - 1) // count
    sizes = [chunk] * (count - 1) + [extent - chunk * (count - 1)]
    actual = _session(model).run(None, {"data": data})
    _assert_parts(actual, data, sizes)
    reference = ReferenceEvaluator(model).run(None, {"data": data})
    for part, wanted in zip(actual, reference, strict=True):
        np.testing.assert_array_equal(part, wanted)
    # ORT 1.31's num_outputs path rejects some empty parts; explicit sizes are equivalent.
    oracle = _model(shape=data.shape, sizes=sizes, count=count, opset=opset)
    _assert_parts(_session(oracle, cpu=True).run(None, {"data": data}), data, sizes)


@pytest.mark.parametrize("dtype", [
    np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
    np.int64, np.uint64, np.float16, np.float32, np.float64,
])
def test_split_exact_payload_families(dtype):
    data = np.arange(12).reshape(2, 6).astype(dtype)
    if dtype == np.bool_:
        data = np.arange(12).reshape(2, 6) % 5 == 1
    elif dtype == np.int64:
        data += np.int64(-(2**63) + 64)
    elif dtype == np.uint64:
        data += np.uint64(2**63 + 64)
    elif dtype == np.float64:
        data += 2**-40
    model = _model(helper.np_dtype_to_tensor_dtype(data.dtype), data.shape, count=3, live=True)
    _assert_parts(_session(model).run(None, {"data": data, "sizes": np.array([2, 0, 4], np.int64)}),
                  data, [2, 0, 4])


def test_split_bfloat16_payload():
    model = _model(T.BFLOAT16, sizes=[2, 4])
    model.graph.input[0].type.tensor_type.elem_type = T.FLOAT
    model.graph.node[0].input[0] = "narrow"
    for i, output in enumerate(model.graph.output):
        model.graph.node[0].output[i] = f"narrow{i}"
        output.type.tensor_type.elem_type = T.FLOAT
    model.graph.node.insert(0, helper.make_node("Cast", ["data"], ["narrow"], to=T.BFLOAT16))
    model.graph.node.extend(helper.make_node("Cast", [f"narrow{i}"], [f"out{i}"], to=T.FLOAT)
                            for i in range(2))
    data = np.arange(12, dtype=np.float32).reshape(2, 6) / 2
    _assert_parts(_session(model).run(None, {"data": data}), data, [2, 4])


@pytest.mark.parametrize("form", ["live", "symbolic", "num_outputs"])
def test_split_new_forms_strict_claim(form):
    data = np.arange(10 if form == "num_outputs" else 12, dtype=np.float32).reshape(2, -1)
    model = _model(shape=data.shape if form != "symbolic" else [None, None],
                   sizes=[2, 4] if form != "num_outputs" else None,
                   live=form == "live", num_outputs=2 if form == "num_outputs" else None)
    feeds = {"data": data}
    if form == "live":
        feeds["sizes"] = np.array([2, 4], np.int64)
    _assert_parts(_session(model).run(None, feeds), data,
                  [3, 2] if form == "num_outputs" else [2, 4])


@pytest.mark.parametrize("eager", [False, True])
def test_split_live_sizes_replay(eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(shape=[None, None], live=True))
    for invocation, sizes in enumerate([[2, 4], [2, 4], [0, 6], [0, 6], [2, 4]]):
        data = np.arange(12, dtype=np.float32).reshape(2, 6) + invocation
        _assert_parts(session.run(None, {"data": data, "sizes": np.array(sizes, np.int64)}),
                      data, sizes)


@pytest.mark.parametrize("eager", [False, True])
def test_split_interior_sizes(eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _model(live=True)
    model.graph.input[1].name = "size_values"
    model.graph.initializer.append(
        numpy_helper.from_array(np.array([2, 1], np.int64), name="prefix_sizes"))
    model.graph.node.insert(
        0, helper.make_node("Split", ["size_values", "prefix_sizes"], ["sizes", "unused"], axis=0))
    session = _session(model)
    for invocation, sizes in enumerate([[2, 4], [0, 6], [2, 4]]):
        data = np.arange(12, dtype=np.float32).reshape(2, 6) + invocation
        _assert_parts(
            session.run(None, {"data": data, "size_values": np.array([*sizes, 99], np.int64)}),
            data, sizes,
        )


def test_split_live_sizes_actual_routes(tmp_path):
    trace = tmp_path / "split_trace.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "split_live_sizes_replay or split_interior_sizes",
         "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path" and "Split" in event["args"].get("ops", "")]
    assert [event["cache"] for event in paths if event["path"] == "general"] == [
        "MISS", "HIT", "RETRACE", "HIT", "HIT"]
    assert sum(event["path"] == "eager" for event in paths) == 11


@pytest.mark.parametrize("bad", [
    [-1, 7], [1, 4], [1, 2, 3], [], [[2, 4]], [2**63 - 1, 2**63 - 1],
])
def test_split_live_invalid_sizes_recover(bad):
    session = _session(_model(live=True))
    data = np.arange(12, dtype=np.float32).reshape(2, 6)
    good = {"data": data, "sizes": np.array([2, 4], np.int64)}
    _assert_parts(session.run(None, good), data, [2, 4])
    with pytest.raises(Exception, match="MLX Split"):
        session.run(None, {"data": data, "sizes": np.array(bad, np.int64)})
    good["data"] = data + 20
    _assert_parts(session.run(None, good), good["data"], [2, 4])


def test_split_unknown_rank_and_error_recovery():
    session = _session(_model(shape=None, live=True))
    for shape in [(6,), (2, 6), (2, 1, 6)]:
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        _assert_parts(session.run(None, {"data": data, "sizes": np.array([2, 4], np.int64)}),
                      data, [2, 4])
    with pytest.raises(Exception, match="MLX Split"):
        session.run(None, {"data": np.array(1, np.float32), "sizes": np.array([2, 4], np.int64)})
    data = np.arange(6, dtype=np.float32)
    _assert_parts(session.run(None, {"data": data, "sizes": np.array([2, 4], np.int64)}),
                  data, [2, 4])


def test_split_boundary_specialization_limit():
    session = _session(_model(shape=[None, None], live=True))
    for invocation, left in enumerate([*range(18), 0]):
        data = np.arange(34, dtype=np.float32).reshape(2, 17) + invocation
        sizes = [left, 17 - left]
        _assert_parts(session.run(None, {"data": data, "sizes": np.array(sizes, np.int64)}),
                      data, sizes)


def test_split_num_outputs_invalid_geometry_recovers():
    session = _session(_model(shape=[None, None], count=4, num_outputs=4))
    data = np.arange(14, dtype=np.float32).reshape(2, 7)
    _assert_parts(session.run(None, {"data": data}), data, [2, 2, 2, 1])
    for extent in [1, 2, 5]:
        with pytest.raises(Exception, match="MLX Split"):
            session.run(None, {"data": np.arange(2 * extent, dtype=np.float32).reshape(2, extent)})
    _assert_parts(session.run(None, {"data": data + 20}), data + 20, [2, 2, 2, 1])
    for extent, sizes in [(0, [0, 0, 0, 0]), (3, [1, 1, 1, 0])]:
        data = np.arange(2 * extent, dtype=np.float32).reshape(2, extent)
        _assert_parts(session.run(None, {"data": data}), data, sizes)
