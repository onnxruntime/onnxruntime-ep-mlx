"""Pad versions, signed geometry, live parameters and exact movable payloads."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort
from onnx import TensorProto as T, helper, numpy_helper
import pytest


def _model(shape, pads, axes=None, *, dtype=np.float32, mode="constant", opset=26,
           live=False, live_axes=False, live_cv=False, cv=None, axes_dtype=np.int64):
    tensor_type = numpy_helper.from_array(np.array(0, dtype)).data_type
    inputs = [helper.make_tensor_value_info("data", tensor_type, shape)]
    initializers = []
    names = ["data"]
    attrs = {"mode": mode}
    if opset < 11:
        attrs["paddings" if opset == 1 else "pads"] = pads
        if cv is not None:
            attrs["value"] = float(cv)
    else:
        for name, value, parameter_dtype, dynamic in [
            ("pads", pads, np.int64, live),
            ("cv", cv, dtype, live_cv),
            ("axes", axes, axes_dtype, live or live_axes),
        ]:
            if value is None:
                names.append("")
                continue
            value = np.array(value, parameter_dtype)
            names.append(name)
            if dynamic:
                inputs.append(helper.make_tensor_value_info(
                    name, numpy_helper.from_array(value).data_type,
                    [] if name == "cv" else [None]))
            else:
                initializers.append(numpy_helper.from_array(value, name=name))
        while names[-1] == "":
            names.pop()
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Pad", names, ["out"], **attrs)],
            "pad_forms", inputs,
            [helper.make_tensor_value_info("out", tensor_type,
                                           None if shape is None else [None] * len(shape))],
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
        providers=["CPUExecutionProvider"] if cpu else ["MLXExecutionProvider"])
    session.disable_fallback()
    return session


def _expected(data, pads, axes=None, mode="constant", cv=0):
    axes = list(range(data.ndim)) if axes is None else [axis % data.ndim for axis in axes]
    shape = list(data.shape)
    before = [0] * data.ndim
    after = [0] * data.ndim
    for i, axis in enumerate(axes):
        before[axis], after[axis] = int(pads[i]), int(pads[i + len(axes)])
        shape[axis] += before[axis] + after[axis]
    if mode == "constant":
        result = np.full(shape, cv, dtype=data.dtype)
        for output in np.ndindex(*shape):
            source = tuple(index - pad for index, pad in zip(output, before))
            if all(0 <= index < extent for index, extent in zip(source, data.shape)):
                result[output] = data[source]
        return result
    cropped = data[tuple(slice(max(-low, 0), extent - max(-high, 0))
                         for low, high, extent in zip(before, after, data.shape))]
    return np.pad(cropped, [(max(low, 0), max(high, 0))
                            for low, high in zip(before, after)], mode=mode)


def _assert_exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(
        actual.reshape(-1).view(np.uint8), expected.reshape(-1).view(np.uint8))


def _payload(dtype):
    if dtype == np.int64:
        return np.array([-(2**63), 2**63 - 1, 2**53 + 1, -2**53 - 1, 7, -9], dtype).reshape(2, 3)
    if dtype == np.uint64:
        return np.array([0, 2**64 - 1, 2**63 + 1, 2**53 + 1, 7, 9], dtype).reshape(2, 3)
    result = np.arange(6).astype(dtype).reshape(2, 3)
    if dtype == np.float64:
        result += np.arange(1, 7).reshape(2, 3) * 2.0**-40
    return result


@pytest.mark.parametrize("case", ["legacy", "signed", "int32_axes", "live"])
def test_pad_new_forms_strict_claim(case):
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    pads, axes = [-1, 2], [1]
    options = {}
    if case == "legacy":
        pads, axes = [0, 1, 0, 1], None
        options["opset"] = 2
    elif case == "int32_axes":
        pads = [1, 2]
        options["axes_dtype"] = np.int32
        options["live_axes"] = True
    elif case == "live":
        options["live"] = True
    model = _model(data.shape, pads, axes, **options)
    feeds = {"data": data}
    if case == "live":
        feeds.update(pads=np.array(pads, np.int64), axes=np.array(axes, np.int64))
    elif case == "int32_axes":
        feeds["axes"] = np.array(axes, np.int32)
    _assert_exact(_session(model).run(None, feeds)[0], _expected(data, pads, axes))


@pytest.mark.parametrize("dtype", [
    np.float32, np.float16, np.float64, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64, np.bool_,
])
@pytest.mark.parametrize("mode", ["constant", "reflect", "edge", "wrap"])
@pytest.mark.parametrize("eager", [False, True])
def test_pad_exact_movable_payloads(dtype, mode, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = _payload(dtype)
    cv = np.array(1 if dtype == np.bool_ else 7, dtype)
    if dtype == np.int64:
        cv = np.array(2**53 + 1, dtype)
    elif dtype == np.uint64:
        cv = np.array(2**64 - 1, dtype)
    if dtype == np.float64:
        cv += 2.0**-40
    pads, axes = [1, 2, 2, 1], [-1, 0]
    model = _model(data.shape, pads, axes, dtype=dtype, mode=mode, cv=cv)
    _assert_exact(_session(model).run(None, {"data": data})[0],
                  _expected(data, pads, axes, mode, cv))


@pytest.mark.parametrize("mode", ["constant", "reflect", "edge", "wrap"])
@pytest.mark.parametrize("eager", [False, True])
def test_pad_bfloat16_payload(mode, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.array([[1.5, -2, 3.25], [4.5, -5.25, 6]], np.float32)
    pads, axes = [2, 3], [1]
    model = _model(data.shape, pads, axes, mode=mode)
    model.graph.node[0].input[0] = "narrow"
    model.graph.node[0].output[0] = "padded"
    model.graph.node.insert(0, helper.make_node("Cast", ["data"], ["narrow"], to=T.BFLOAT16))
    model.graph.node.append(helper.make_node("Cast", ["padded"], ["out"], to=T.FLOAT))
    _assert_exact(_session(model).run(None, {"data": data})[0],
                  _expected(data, pads, axes, mode))


@pytest.mark.parametrize("opset", [1, 2, 10])
@pytest.mark.parametrize("mode", ["constant", "reflect", "edge"])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.float64])
@pytest.mark.parametrize("eager", [False, True])
def test_pad_legacy_attributes(opset, mode, dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = _payload(dtype)
    pads = [1, 0, 1, 2]
    model = _model(data.shape, pads, dtype=dtype, mode=mode, opset=opset, cv=-2.5)
    _assert_exact(_session(model).run(None, {"data": data})[0],
                  _expected(data, pads, mode=mode, cv=-2.5))


@pytest.mark.parametrize("mode", ["constant", "reflect", "edge", "wrap"])
@pytest.mark.parametrize("pads", [[-1, 2], [2, -1], [-1, -1]])
@pytest.mark.parametrize("eager", [False, True])
def test_pad_signed_crop_then_pad(mode, pads, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.array([10, 20, 30, 40], np.int64)
    model = _model(data.shape, pads, dtype=np.int64, mode=mode, cv=-7)
    expected = _expected(data, pads, mode=mode, cv=-7)
    _assert_exact(_session(model).run(None, {"data": data})[0], expected)
    _assert_exact(_session(model, cpu=True).run(None, {"data": data})[0], expected)


@pytest.mark.parametrize("pads", [
    [-4, 2], [2, -4], [-3, 2], [-5, 5],
    [-(2**63), 2**63 - 1], [2**63 - 1, -(2**63)],
])
@pytest.mark.parametrize("eager", [False, True])
def test_pad_constant_overcrop_uses_original_coordinates(pads, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.array([10, 20, 30], np.int64)
    model = _model(data.shape, pads, dtype=np.int64, cv=2**53 + 1)
    _assert_exact(_session(model).run(None, {"data": data})[0],
                  _expected(data, pads, cv=2**53 + 1))


@pytest.mark.parametrize("shape,pads,axes", [
    ([], [], None), ([0], [2, 3], None), ([0, 3], [1, 2], [1]),
    ([2, 3], [], []), ([3], [-3, 0], None),
])
@pytest.mark.parametrize("eager", [False, True])
def test_pad_empty_and_scalar_geometry(shape, pads, axes, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.zeros(shape, np.int64)
    model = _model(shape, pads, axes, dtype=np.int64, cv=2**53 + 1)
    _assert_exact(_session(model).run(None, {"data": data})[0],
                  _expected(data, pads, axes, cv=2**53 + 1))


@pytest.mark.parametrize("eager", [False, True])
def test_pad_live_controls_replay(eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _model([None, None], [0, 2], [1], live=True, live_cv=True,
                   cv=0, axes_dtype=np.int32, opset=19)
    session = _session(model)
    for invocation, (pads, axes) in enumerate([
        ([0, 2], [1]), ([0, 2], [1]), ([1, -1], [1]), ([1, -1], [1]),
        ([1, -1], [0]), ([1, -1], [0]), ([0, 2], [1]),
    ]):
        data = np.arange(6, dtype=np.float32).reshape(2, 3) + invocation
        cv = np.array(10 + invocation, np.float32)
        feeds = {"data": data, "pads": np.array(pads, np.int64),
                 "axes": np.array(axes, np.int32), "cv": cv}
        _assert_exact(session.run(None, feeds)[0], _expected(data, pads, axes, cv=cv))


@pytest.mark.parametrize("eager", [False, True])
def test_pad_interior_controls(eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _model([2, 3], [0, 2], [1], live=True, opset=19)
    model.graph.input[1].name = "pad_values"
    model.graph.input[1].type.tensor_type.shape.dim[0].dim_value = 3
    model.graph.initializer.append(numpy_helper.from_array(
        np.array([2, 1], np.int64), name="parts"))
    model.graph.node.insert(0, helper.make_node(
        "Split", ["pad_values", "parts"], ["pads", "unused"], axis=0))
    session = _session(model)
    for invocation, pads in enumerate([[0, 2], [-1, 2], [0, 2]]):
        data = np.arange(6, dtype=np.float32).reshape(2, 3) + invocation
        _assert_exact(session.run(None, {
            "data": data, "pad_values": np.array([*pads, 99], np.int64),
            "axes": np.array([1], np.int64),
        })[0], _expected(data, pads, [1]))


def test_pad_live_controls_actual_routes(tmp_path):
    trace = tmp_path / "pad_trace.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "pad_live_controls_replay or pad_interior_controls",
         "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path" and "Pad" in event["args"].get("ops", "")]
    assert [event["cache"] for event in paths if event["path"] == "general"] == [
        "MISS", "HIT", "RETRACE", "HIT", "RETRACE", "HIT", "HIT"]
    assert sum(event["path"] == "eager" for event in paths) == 13


@pytest.mark.parametrize("pads,axes", [
    ([1], [1]), ([1, 1], [2]), ([1, 1, 0, 0], [-1, 1]),
    ([-4, 0], [1]), ([2**63 - 1, 0], [1]),
])
def test_pad_live_invalid_controls_recover(pads, axes):
    session = _session(_model([2, 3], [0, 1], [1], live=True, opset=19))
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    good = {"data": data, "pads": np.array([0, 1], np.int64),
            "axes": np.array([1], np.int64)}
    _assert_exact(session.run(None, good)[0], _expected(data, [0, 1], [1]))
    with pytest.raises(Exception, match="MLX Pad"):
        session.run(None, {"data": data, "pads": np.array(pads, np.int64),
                           "axes": np.array(axes, np.int64)})
    good["data"] = data + 20
    _assert_exact(session.run(None, good)[0], _expected(data + 20, [0, 1], [1]))


@pytest.mark.parametrize("name", ["pads", "axes", "cv"])
def test_pad_unknown_control_ranks_validate_and_recover(name):
    model = _model([2, 3], [0, 1], [1], live=True, live_cv=True, cv=7, opset=19)
    next(value for value in model.graph.input if value.name == name).type.tensor_type.ClearField("shape")
    session = _session(model)
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    good = {"data": data, "pads": np.array([0, 1], np.int64),
            "axes": np.array([1], np.int64), "cv": np.array(7, np.float32)}
    _assert_exact(session.run(None, good)[0], _expected(data, [0, 1], [1], cv=7))
    bad = dict(good)
    bad[name] = good[name].reshape((1,) if name == "cv" else (1, -1))
    with pytest.raises(Exception, match="MLX Pad"):
        session.run(None, bad)
    good["data"] = data + 20
    _assert_exact(session.run(None, good)[0], _expected(data + 20, [0, 1], [1], cv=7))


@pytest.mark.parametrize("eager", [False, True])
def test_pad_unknown_data_rank_replay(eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _model(None, [0, 1], live=True, opset=19)
    session = _session(model)
    for invocation, shape in enumerate([[3], [2, 3], [3]]):
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + invocation
        pads = [0] * len(shape) + [1] * len(shape)
        _assert_exact(session.run(None, {
            "data": data, "pads": np.array(pads, np.int64),
        })[0], _expected(data, pads))


def test_pad_boundary_variant_cap_preserves_returned_payloads():
    session = _session(_model([2, 3], [0, 1], [1], live=True, opset=19))
    for invocation, before in enumerate([*range(18), 0]):
        data = np.arange(6, dtype=np.float32).reshape(2, 3) + invocation
        pads = [before, 0]
        _assert_exact(session.run(None, {
            "data": data, "pads": np.array(pads, np.int64),
            "axes": np.array([1], np.int64),
        })[0], _expected(data, pads, [1]))


def test_pad_live_output_byte_overflow_recover():
    session = _session(_model([1, 1, 1], [0] * 6, dtype=np.float64, live=True))
    data = np.array([[[1 + 2.0**-40]]], np.float64)
    good = {"data": data, "pads": np.zeros(6, np.int64)}
    _assert_exact(session.run(None, good)[0], data)
    with pytest.raises(Exception, match="MLX Pad: output byte count overflows"):
        session.run(None, {
            "data": data,
            "pads": np.array([2**30 - 1, 2**30 - 1, 1, 0, 0, 0], np.int64),
        })
    good["data"] = data + 2.0**-40
    _assert_exact(session.run(None, good)[0], good["data"])
