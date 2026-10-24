"""Modern constant Tile: exact payloads, checked live geometry and capture routes."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest


DT = ir.DataType
FAIL = ort.capi.onnxruntime_pybind11_state.Fail
EP_FAIL = ort.capi.onnxruntime_pybind11_state.EPFail
DTYPES = [
    np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64, np.bool_,
]


def _value(name, dtype, shape):
    return ir.Value(name=name, type=ir.TensorType(dtype),
                    shape=None if shape is None else ir.Shape(shape))


def _model(shape, repeats, *, dtype=np.float32, opset=13, output=None,
           chain=False, bf16=False):
    tensor_type = ir.tensor(np.array(0, dtype)).dtype
    x = _value("x", tensor_type, shape)
    y = _value("y", tensor_type, output)
    repeats = np.asarray(repeats, np.int64)
    r = ir.Value(name="repeats", type=ir.TensorType(DT.INT64),
                 shape=ir.Shape(repeats.shape), const_value=ir.tensor(repeats))
    nodes = []
    data = x
    if bf16:
        data = _value("narrow", DT.BFLOAT16, shape)
        nodes.append(ir.node("Cast", [x], attributes={"to": int(DT.BFLOAT16)},
                             outputs=[data]))
    elif chain:
        data = _value("transposed", tensor_type, None)
        nodes.append(ir.node("Transpose", [x], attributes={"perm": [1, 0]},
                             outputs=[data]))
    tiled = _value("tiled", DT.BFLOAT16 if bf16 else tensor_type, None) if (
        chain or bf16) else y
    nodes.append(ir.node("Tile", [data, r], outputs=[tiled]))
    if bf16:
        nodes.append(ir.node("Cast", [tiled], attributes={"to": int(DT.FLOAT)},
                             outputs=[y]))
    elif chain:
        nodes.append(ir.node("Transpose", [tiled], attributes={"perm": [1, 0]},
                             outputs=[y]))
    graph = ir.Graph([x], [y], nodes=nodes, initializers=[r],
                     opset_imports={"": opset}, name="tile_forms")
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


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
        model, options,
        providers=["CPUExecutionProvider"] if cpu else ["MLXExecutionProvider"],
        enable_fallback=False)
    session.disable_fallback()
    return session


def _ownership(session):
    profile = Path(session.end_profiling())
    try:
        events = json.loads(profile.read_text())
    finally:
        profile.unlink()
    providers = {
        event["args"]["provider"] for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    }
    assert providers == {"MLXExecutionProvider"}, providers


def _expected(data, repeats):
    shape = tuple(extent * repeat for extent, repeat in zip(data.shape, repeats, strict=True))
    result = np.empty(shape, data.dtype)
    for index in np.ndindex(shape):
        source = tuple(i % extent for i, extent in zip(index, data.shape, strict=True))
        result[index] = data[source]
    return result


def _exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def _payload(dtype):
    if dtype == np.int64:
        return np.array([-(2**63), 2**63 - 1, 2**53 + 1, -2**53 - 1, 0, 7], dtype)
    if dtype == np.uint64:
        return np.array([0, 2**64 - 1, 2**63 + 1, 2**53 + 1, 7, 9], dtype)
    if dtype == np.float64:
        return np.array([0, 0x8000000000000000, 0x7FF8000000001234,
                         0x3FF0000000000001, 0xFFF0000000000000,
                         0x7FF0000000000000], np.uint64).view(dtype)
    if dtype == np.float32:
        return np.array([0, 0x80000000, 0x7FC01234, 0x3F800001,
                         0xFF800000, 0x7F800000], np.uint32).view(dtype)
    if dtype == np.float16:
        return np.array([0, 0x8000, 0x7E01, 0x3C01, 0xFC00, 0x7C00],
                        np.uint16).view(dtype)
    return np.arange(6).astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("opset", [6, 13])
@pytest.mark.parametrize("eager", [False, True])
def test_tile_exact_types(dtype, opset, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = _payload(dtype).reshape(2, 3)
    repeats = [2, 3]
    model = _model(data.shape, repeats, dtype=dtype, opset=opset)
    expected = _expected(data, repeats)
    session = _session(model, profile=tmp_path / "tile")
    _exact(session.run(None, {"x": data})[0], expected)
    _ownership(session)
    cpu_expected = expected
    if dtype == np.float16:
        # Hosted ORT 1.31 CPU Tile canonicalizes this fixture's 0x7e01 NaN to 0x7e00.
        # This is an oracle limitation, not the MLX/ONNX payload contract above.
        cpu_expected = expected.copy()
        bits = cpu_expected.view(np.uint16)
        bits[bits == 0x7E01] = 0x7E00
    _exact(_session(model, cpu=True).run(None, {"x": data})[0], cpu_expected)


@pytest.mark.parametrize("shape,repeats", [
    ([], []), ([0], [2]), ([0, 2], [3, 2]), ([2, 0, 0], [2, 3, 4]),
    ([2, 3], [0, 2]), ([0, 2], [2**63 - 1, 3]),
    ([65536, 65536, 0], [1, 1, 1]),
])
@pytest.mark.parametrize("dtype", [np.int8, np.float32, np.uint64])
@pytest.mark.parametrize("eager", [False, True])
def test_tile_scalar_and_empty(shape, repeats, dtype, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.full(shape, 7, dtype)
    model = _model(shape, repeats, dtype=dtype)
    session = _session(model, profile=tmp_path / "empty_tile")
    _exact(session.run(None, {"x": data})[0], _expected(data, repeats))
    _ownership(session)


@pytest.mark.parametrize("eager", [False, True])
def test_tile_bfloat16_interior(eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.array([1.5, -2, 3.25, -4.5, 0, -0.0], np.float32).reshape(2, 3)
    session = _session(_model(data.shape, [2, 3], bf16=True), profile=tmp_path / "bf16")
    _exact(session.run(None, {"x": data})[0], _expected(data, [2, 3]))
    _ownership(session)


@pytest.mark.parametrize("chain", [False, True])
@pytest.mark.parametrize("eager", [False, True])
def test_tile_shape_replay(chain, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(["a", "b"], [2, 3], chain=chain),
                       profile=tmp_path / "replay")
    for invocation, shape in enumerate([
        [2, 3], [2, 3], [3, 2], [3, 2], [2, 3], [0, 3], [2, 3], [0, 3],
    ]):
        data = (np.arange(math.prod(shape), dtype=np.float32) + invocation * 10).reshape(shape)
        expected = _expected(data.T if chain else data, [2, 3])
        _exact(session.run(None, {"x": data})[0], expected.T if chain else expected)
    _ownership(session)


@pytest.mark.parametrize("eager", [False, True])
def test_tile_noncontiguous_intermediate(eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model([2, 3], [2, 3], chain=True),
                       profile=tmp_path / "intermediate")
    for invocation in range(3):
        data = (np.arange(6, dtype=np.float32) + invocation * 10).reshape(2, 3)
        _exact(session.run(None, {"x": data})[0], _expected(data.T, [2, 3]).T)
    _ownership(session)


def test_tile_actual_capture_routes(tmp_path):
    cases = [
        ("standalone", "test_tile_shape_replay[False-False]", "Tile",
         ["MISS", "HIT", "RETRACE", "HIT", "HIT", "HIT"], 2),
        ("symbolic_chain", "test_tile_shape_replay[False-True]", "Transpose,Tile,Transpose",
         ["MISS", "HIT", "RETRACE", "HIT", "HIT", "HIT"], 2),
        ("wide_metadata", "test_tile_wide_input_metadata_only[False]", "Tile", [], 3),
        ("fixed_chain", "test_tile_noncontiguous_intermediate[False]",
         "Transpose,Tile,Transpose", ["MISS", "HIT", "HIT"], 0),
    ]
    for label, test, ops, caches, eager_calls in cases:
        trace = tmp_path / f"{label}.json"
        subprocess.run(
            [sys.executable, "-m", "pytest", f"{Path(__file__).name}::{test}",
             "-q", "-p", "no:cacheprovider"],
            cwd=Path(__file__).parent,
            env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
                 "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
            check=True, timeout=120)
        paths = [event["args"] for event in json.loads(trace.read_text())
                 if event.get("cat") == "ep.path"]
        assert paths and all(event["ops"] == ops for event in paths), (label, paths)
        assert [event["cache"] for event in paths if event["path"] == "general"] == caches, label
        assert sum(event["path"] == "eager" for event in paths) == eager_calls, label
        assert len(paths) == len(caches) + eager_calls, label


@pytest.mark.parametrize("repeats", [[-1], [-(2**63)]])
def test_tile_negative_constants_declined(repeats):
    # Unknown data rank lets the provider establish the repeat-value rejection.
    with pytest.raises(FAIL, match="CPU EP.*fallback.*disabled"):
        _session(_model(None, repeats))


@pytest.mark.parametrize("repeats,output", [
    ([2, 1], []), ([], [None]), ([0], [3]),
])
def test_tile_inconsistent_declared_output_declined(repeats, output):
    with pytest.raises(FAIL, match="CPU EP.*fallback.*disabled"):
        _session(_model(None, repeats, output=output))


@pytest.mark.parametrize("repeat", [2**32 + 1, 2**63 - 1])
@pytest.mark.parametrize("eager", [False, True])
def test_tile_full_width_runtime_extent_errors_recover(repeat, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model([None], [repeat]))
    empty = np.empty([0], np.float32)
    _exact(session.run(None, {"x": empty})[0], empty)
    with pytest.raises(EP_FAIL, match="MLX Tile: output extent exceeds int32"):
        session.run(None, {"x": np.array([7], np.float32)})
    _exact(session.run(None, {"x": empty})[0], empty)


@pytest.mark.parametrize("shape", [[], [6], [1, 2, 3]])
def test_tile_unknown_rank_cardinality_errors_recover(shape):
    session = _session(_model(None, [2, 1]))
    good = np.arange(6, dtype=np.float32).reshape(2, 3)
    _exact(session.run(None, {"x": good})[0], _expected(good, [2, 1]))
    bad = np.array(7, np.float32) if not shape else good.reshape(shape)
    with pytest.raises(EP_FAIL, match="MLX Tile: repeats length must equal data rank"):
        session.run(None, {"x": bad})
    good += 10
    _exact(session.run(None, {"x": good})[0], _expected(good, [2, 1]))


@pytest.mark.parametrize("eager", [False, True])
def test_tile_unknown_rank_scalar_errors_recover(eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(None, []))
    scalar = np.array(-0.0, np.float32)
    _exact(session.run(None, {"x": scalar})[0], scalar)
    with pytest.raises(EP_FAIL, match="MLX Tile: repeats length must equal data rank"):
        session.run(None, {"x": scalar.reshape(1)})
    _exact(session.run(None, {"x": scalar})[0], scalar)


@pytest.mark.parametrize("repeats,bad_shape,message", [
    ([2**63 - 1, 1], [2, 0], "output extent overflows int64"),
    ([1, 2**32 + 1], [0, 1], "output extent exceeds int32"),
    ([65536, 65536], [1, 1], "element count exceeds int32 indexing"),
])
def test_tile_all_axes_checked_with_empty_recovery(repeats, bad_shape, message):
    session = _session(_model([None] * len(repeats), repeats, dtype=np.uint8))
    empty = np.empty([0] * len(repeats), np.uint8)
    _exact(session.run(None, {"x": empty})[0], _expected(empty, repeats))
    bad = np.empty(bad_shape, np.uint8)
    with pytest.raises(EP_FAIL, match=f"MLX Tile(?: output)?: .*{message}"):
        session.run(None, {"x": bad})
    _exact(session.run(None, {"x": empty})[0], _expected(empty, repeats))


@pytest.mark.parametrize("eager", [False, True])
def test_tile_wide_input_metadata_only(eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.empty([0, 2**31 + 7], np.uint8)
    session = _session(_model(list(data.shape), [1, 0], dtype=np.uint8),
                       profile=tmp_path / "wide")
    for _ in range(3):
        _exact(session.run(None, {"x": data})[0], np.empty([0, 0], np.uint8))
    _ownership(session)
