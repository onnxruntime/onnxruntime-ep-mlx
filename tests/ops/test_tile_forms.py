"""Modern Tile: exact payloads, checked geometry and context-specific capture routes."""

import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest


DT = ir.DataType
FAIL = ort.capi.onnxruntime_pybind11_state.Fail
EP_FAIL = ort.capi.onnxruntime_pybind11_state.EPFail
INVALID_GRAPH = ort.capi.onnxruntime_pybind11_state.InvalidGraph
DTYPES = [
    np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64, np.bool_,
]


def _value(name, dtype, shape):
    return ir.Value(name=name, type=ir.TensorType(dtype),
                    shape=None if shape is None else ir.Shape(shape))


def _model(shape, repeats, *, dtype=np.float32, opset=13, output=None,
           chain=False, bf16=False, live=None, repeats_shape="fixed",
           repeats_dtype=np.int64):
    tensor_type = ir.tensor(np.array(0, dtype)).dtype
    x = _value("x", tensor_type, shape)
    y = _value("y", tensor_type, output)
    repeats = np.asarray(repeats, repeats_dtype)
    rshape = list(repeats.shape) if repeats_shape == "fixed" else repeats_shape
    r = _value("repeats", ir.tensor(repeats).dtype, rshape)
    if live is None:
        r.const_value = ir.tensor(repeats)
    nodes = []
    tile_repeats = r
    if live == "produced":
        tile_repeats = _value("live_repeats", ir.tensor(repeats).dtype, rshape)
        nodes.append(ir.node("Identity", [r], outputs=[tile_repeats],
                             name="live_repeats_producer"))
    data = x
    if bf16:
        data = _value("narrow", DT.BFLOAT16, shape)
        nodes.append(ir.node("Cast", [x], attributes={"to": int(DT.BFLOAT16)},
                             outputs=[data]))
    elif chain:
        data = _value("transposed", tensor_type, None)
        nodes.append(ir.node("Transpose", [x], attributes={"perm": [1, 0]},
                             outputs=[data], name="live_before" if live else ""))
    tiled = _value("tiled", DT.BFLOAT16 if bf16 else tensor_type, None) if (
        chain or bf16) else y
    nodes.append(ir.node("Tile", [data, tile_repeats], outputs=[tiled],
                         name="live_tile" if live else ""))
    if bf16:
        nodes.append(ir.node("Cast", [tiled], attributes={"to": int(DT.FLOAT)},
                             outputs=[y]))
    elif chain:
        nodes.append(ir.node("Transpose", [tiled], attributes={"perm": [1, 0]},
                             outputs=[y], name="live_after" if live else ""))
    graph = ir.Graph([x] if live is None else [x, r], [y], nodes=nodes,
                     initializers=[r] if live is None else [],
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


def _native_candidate_model(extra):
    """Broad Q8 candidate with an exact INT64 position chain, not a detector mock."""
    import onnx
    from test_matmulnbits_asym import _matmulnbits_model

    quantized, _ = _matmulnbits_model(
        M=1, K=32, N=32, block=32, asymmetric=False, bits=8)
    model = onnx.load_from_string(quantized)
    # Import 14 resolves to effective Tile schema 13, not a Tile-14 schema.
    next(opset for opset in model.opset_import if opset.domain == "").version = 14
    initializers = []

    def constant(name, data):
        value = _value(name, DT.INT64, list(np.asarray(data).shape))
        value.const_value = ir.tensor(np.asarray(data, np.int64))
        initializers.append(value)
        return value

    start = _value("start", DT.INT64, [])
    length = _value("length", DT.INT64, [])
    batch = _value("batch", DT.INT64, [1])
    step = constant("step", np.array(1, np.int64))
    one = constant("one", [1])
    axes = constant("axes", [0])
    limit = _value("limit", DT.INT64, [])
    positions = _value("positions", DT.INT64, [None])
    row = _value("position_row", DT.INT64, [1, None])
    length_vector = _value("length_vector", DT.INT64, [1])
    shape = _value("position_shape", DT.INT64, [2])
    repeats = _value("position_repeats", DT.INT64, [2])
    expanded = _value("expanded_positions", DT.INT64, [1, None])
    tiled = _value("native_positions", DT.INT64, [None, None])
    nodes = [
        ir.node("Add", [start, length], outputs=[limit]),
        ir.node("Range", [start, limit, step], outputs=[positions], name="native_range"),
        ir.node("Unsqueeze", [positions, axes], outputs=[row]),
        ir.node("Unsqueeze", [length, axes], outputs=[length_vector]),
        ir.node("Concat", [one, length_vector], attributes={"axis": 0}, outputs=[shape]),
        ir.node("Concat", [batch, one], attributes={"axis": 0}, outputs=[repeats]),
        ir.node("Expand", [row, shape], outputs=[expanded]),
        ir.node("Tile", [expanded, repeats], outputs=[tiled], name="native_tile"),
    ]
    quantized_out = _value("out", DT.FLOAT, [1, 1, 32])
    query = quantized_out
    for index in range(28):
        output = _value(f"query_{index}", DT.FLOAT, [1, 1, 32])
        nodes.append(ir.node("Add", [query, query], outputs=[output]))
        query = output
    key = _value("key", DT.FLOAT, [1, 1, 32])
    value = _value("value", DT.FLOAT, [1, 1, 32])
    attention = _value("attention", DT.FLOAT, [1, 1, 32])
    nodes.append(ir.node(
        "MultiHeadAttention", [query, key, value], outputs=[attention],
        domain="com.microsoft", attributes={"num_heads": 4}))
    inputs = [start, length, batch, key, value]
    outputs = [tiled, attention]
    if extra != "none":
        dtype = {"int64": DT.INT64, "string": DT.STRING,
                 "complex64": DT.COMPLEX64, "complex128": DT.COMPLEX128}[extra]
        data = _value("unrelated_data", dtype, [2])
        output = _value("unrelated_output", dtype, [4])
        constant_repeats = constant("unrelated_repeats", [2])
        nodes.append(ir.node(
            "Tile", [data, constant_repeats], outputs=[output], name="unrelated_tile"))
        inputs.append(data)
        outputs.append(output)
    extension = ir.to_proto(ir.Model(ir.Graph(
        [quantized_out, *inputs], outputs, nodes=nodes, initializers=initializers,
        opset_imports={"": 14, "com.microsoft": 1}, name="native_tile_admission"),
        ir_version=11))
    model.graph.node.extend(extension.graph.node)
    model.graph.input.extend(item for item in extension.graph.input if item.name != "out")
    del model.graph.output[:]
    model.graph.output.extend(extension.graph.output)
    model.graph.initializer.extend(extension.graph.initializer)
    model.graph.value_info.extend(extension.graph.value_info)
    model.graph.value_info.append(onnx.helper.make_tensor_value_info(
        "out", onnx.TensorProto.FLOAT, [1, 1, 32]))
    return model


@pytest.mark.parametrize("extra", ["none", "int64", "string", "complex64", "complex128"])
def test_native_candidate_tile_payload_session(extra):
    """Admission only; complex fixtures fail in ORT loading before provider admission."""
    model = _native_candidate_model(extra).SerializeToString()
    if extra in {"complex64", "complex128"}:
        message = (
            "[ONNXRuntimeError] : 1 : FAIL : Exception during loading: "
            f"MLDataType for: tensor({extra}) is not currently registered or supported"
        )
        with pytest.raises(FAIL, match=rf"\A{re.escape(message)}\Z"):
            _session(model)
    elif extra in {"none", "int64"}:
        _session(model)
    else:
        with pytest.raises(FAIL, match="CPU EP.*fallback.*disabled"):
            _session(model)


def test_native_candidate_tile_payload_actual_capability(tmp_path):
    for extra in ["none", "int64", "string"]:
        trace = tmp_path / f"native_{extra}.json"
        subprocess.run(
            [sys.executable, "-m", "pytest",
             f"{Path(__file__).name}::test_native_candidate_tile_payload_session[{extra}]",
             "-q", "-p", "no:cacheprovider"],
            cwd=Path(__file__).parent,
            env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
            check=True, timeout=120)
        total = len(_native_candidate_model(extra).graph.node)
        claims = [event["args"] for event in json.loads(trace.read_text())
                  if event.get("cat") == "ep.claim" and event["args"]["total"] == total]
        assert total > 32 and len(claims) == 1, (extra, claims)
        claim = claims[0]
        unsupported = extra not in {"none", "int64"}
        assert claim["unclaimed"] == int(unsupported), (extra, claim)
        assert claim["claimed"] == total - int(unsupported), (extra, claim)
        # A live INT64 Range is rejected by the ordinary claim. Its admission in
        # this actual GetCapability trace proves the unchanged broad detector
        # returned true, including when the unrelated constant-repeat Tile exists.
        assert "fallback_Range" not in claim, (extra, claim)
        fallbacks = {key for key in claim if key.startswith("fallback_")}
        assert fallbacks == ({"fallback_Tile"} if unsupported else set()), (extra, claim)
        if unsupported:
            assert claim["fallback_Tile"].startswith(
                "x1: data dtype other is not a movable dtype supported on GPU")
            assert "nodes: [unrelated_tile]" in claim["fallback_Tile"]


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


@pytest.mark.parametrize("live", ["boundary", "produced"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("opset", [6, 13])
@pytest.mark.parametrize("eager", [False, True])
def test_tile_live_exact_types(live, dtype, opset, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = _payload(dtype).reshape(2, 3)
    repeats = np.array([2, 3], np.int64)
    session = _session(_model(data.shape, repeats, dtype=dtype, opset=opset, live=live),
                       profile=tmp_path / "live_exact")
    _exact(session.run(None, {"x": data, "repeats": repeats})[0],
           _expected(data, repeats))
    _ownership(session)


@pytest.mark.parametrize("live", ["boundary", "produced"])
@pytest.mark.parametrize("eager", [False, True])
def test_tile_live_bfloat16(live, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.array([1.5, -2, 3.25, -4.5, 0, -0.0], np.float32).reshape(2, 3)
    repeats = np.array([2, 3], np.int64)
    session = _session(_model(data.shape, repeats, bf16=True, live=live),
                       profile=tmp_path / "live_bf16")
    _exact(session.run(None, {"x": data, "repeats": repeats})[0],
           _expected(data, repeats))
    _ownership(session)


@pytest.mark.parametrize("live", ["boundary", "produced"])
@pytest.mark.parametrize("eager", [False, True])
def test_tile_live_repeats_freshness(live, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model([2, 3], [1, 2], dtype=np.int64, live=live),
                       profile=tmp_path / "live_replay")
    retained = []
    for invocation, values in enumerate([[1, 2], [2, 1], [0, 3], [1, 2], [1, 2]]):
        data = (_payload(np.int64) + invocation).reshape(2, 3)
        repeats = np.array(values, np.int64)
        original = data.tobytes(), repeats.tobytes()
        expected = _expected(data, repeats)
        actual = session.run(None, {"x": data, "repeats": repeats})[0]
        _exact(actual, expected)
        assert (data.tobytes(), repeats.tobytes()) == original
        assert not np.shares_memory(actual, data)
        retained.append((actual, expected.copy()))
    for values in [[-1, 1], [1, -(2**63)]]:
        with pytest.raises(EP_FAIL, match="MLX Tile: repeats must be nonnegative"):
            session.run(None, {"x": data, "repeats": np.array(values, np.int64)})
        _exact(session.run(None, {"x": data, "repeats": repeats})[0], expected)
    for actual, expected in retained:
        _exact(actual, expected)
    _ownership(session)


@pytest.mark.parametrize("live", ["boundary", "produced"])
@pytest.mark.parametrize("shape,repeats", [
    ([], []), ([0], [2**63 - 1]), ([0, 2], [2**63 - 1, 3]),
    ([2, 0, 0], [2, 3, 4]), ([2, 3], [0, 2]),
    ([0, 2**31 + 7], [1, 0]),
])
@pytest.mark.parametrize("eager", [False, True])
def test_tile_live_scalar_and_empty(live, shape, repeats, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.full(shape, 7, np.uint64)
    session = _session(_model(shape, repeats, dtype=np.uint64, live=live),
                       profile=tmp_path / "live_empty")
    repeats = np.array(repeats, np.int64)
    _exact(session.run(None, {"x": data, "repeats": repeats})[0],
           _expected(data, repeats))
    _ownership(session)


@pytest.mark.parametrize("live", ["boundary", "produced"])
@pytest.mark.parametrize("shape", [[], [6], [1, 2, 3]])
def test_tile_live_unknown_rank_recovery(live, shape):
    session = _session(_model(None, [1, 2], live=live, repeats_shape=None))
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    repeats = np.array([1, 2], np.int64)
    for bad in [repeats.reshape(1, 2), np.array(2, np.int64), repeats[:1]]:
        with pytest.raises(EP_FAIL, match="MLX Tile: repeats (?:must have rank one|length must equal data rank)"):
            session.run(None, {"x": data, "repeats": bad})
        _exact(session.run(None, {"x": data, "repeats": repeats})[0],
               _expected(data, repeats))
    bad_data = data.reshape(shape) if shape else np.array(7, np.float32)
    with pytest.raises(EP_FAIL, match="MLX Tile: repeats length must equal data rank"):
        session.run(None, {"x": bad_data, "repeats": repeats})
    _exact(session.run(None, {"x": data, "repeats": repeats})[0],
           _expected(data, repeats))


@pytest.mark.parametrize("live", ["boundary", "produced"])
@pytest.mark.parametrize("eager", [False, True])
def test_tile_live_unknown_metadata_scalar_replay(live, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(None, [], dtype=np.float64, live=live, repeats_shape=None),
                       profile=tmp_path / "unknown_scalar")
    for shape, values in [([], []), ([2, 3], [2, 1]), ([], []),
                          ([0, 2], [2**63 - 1, 3]), ([], [])]:
        data = np.full(shape, -0.0, np.float64)
        repeats = np.array(values, np.int64)
        _exact(session.run(None, {"x": data, "repeats": repeats})[0],
               _expected(data, repeats))
    with pytest.raises(EP_FAIL, match="MLX Tile: repeats length must equal data rank"):
        session.run(None, {"x": data, "repeats": np.array([1], np.int64)})
    _exact(session.run(None, {"x": data, "repeats": repeats})[0], data)
    _ownership(session)


@pytest.mark.parametrize("live", ["boundary", "produced"])
@pytest.mark.parametrize("repeats,error,message", [
    (np.array(2, np.int64), FAIL, "CPU EP.*fallback.*disabled"),
    (np.ones((1, 2), np.int64), FAIL, "CPU EP.*fallback.*disabled"),
    (np.ones(1, np.int64), FAIL, "CPU EP.*fallback.*disabled"),
    (np.ones(2, np.int32), INVALID_GRAPH, "tensor\\(int32\\)"),
])
def test_tile_live_invalid_metadata_rejected(live, repeats, error, message):
    with pytest.raises(error, match=message):
        _session(_model([2, 3], repeats, live=live, repeats_dtype=repeats.dtype))


@pytest.mark.parametrize("live", ["boundary", "produced"])
def test_tile_live_bfloat16_opset6_rejected(live):
    with pytest.raises(INVALID_GRAPH, match="tensor\\(bfloat16\\)"):
        _session(_model([2, 3], [1, 2], live=live, bf16=True, opset=6))


@pytest.mark.parametrize("live", ["boundary", "produced"])
@pytest.mark.parametrize("output", [[], [None], [None, None, None]])
def test_tile_live_inconsistent_output_rank_declined(live, output):
    with pytest.raises(FAIL, match="CPU EP.*fallback.*disabled"):
        _session(_model(None, [1, 2], live=live, output=output))


@pytest.mark.parametrize("live", ["boundary", "produced"])
@pytest.mark.parametrize("repeats,shape,message", [
    ([2**32 + 1], [1], "output extent exceeds int32"),
    ([2**63 - 1], [1], "output extent exceeds int32"),
    ([2**63 - 1, 1], [2, 0], "output extent overflows int64"),
    ([1, 2**32 + 1], [0, 1], "output extent exceeds int32"),
    ([65536, 65536], [1, 1], "element count exceeds int32 indexing"),
    ([0, -1], [0, 2], "repeats must be nonnegative"),
    ([1, -(2**63)], [0, 2], "repeats must be nonnegative"),
])
def test_tile_live_geometry_errors_recover(live, repeats, shape, message):
    session = _session(_model([None] * len(shape), repeats, dtype=np.uint8, live=live))
    data = np.empty(shape, np.uint8)
    good = np.zeros(len(shape), np.int64)
    expected = _expected(data, good)
    _exact(session.run(None, {"x": data, "repeats": good})[0], expected)
    with pytest.raises(EP_FAIL, match=f"MLX Tile(?: output)?: .*{message}"):
        session.run(None, {"x": data, "repeats": np.array(repeats, np.int64)})
    _exact(session.run(None, {"x": data, "repeats": good})[0], expected)


@pytest.mark.parametrize("live", ["boundary", "produced"])
def test_tile_live_partition_fixture(live, tmp_path):
    session = _session(_model([2, 3], [1, 2], chain=True, live=live),
                       profile=tmp_path / "live_partition")
    for invocation, repeats in enumerate([[1, 2], [2, 1], [0, 3], [1, 2]]):
        data = (np.arange(6, dtype=np.float32) + invocation).reshape(2, 3)
        _exact(session.run(None, {"x": data, "repeats": np.array(repeats, np.int64)})[0],
               _expected(data.T, repeats).T)
    _ownership(session)


@pytest.mark.parametrize("live", ["boundary", "produced"])
def test_tile_live_add_partition_fixture(live, tmp_path):
    x = _value("x", DT.FLOAT, [2, 3])
    repeats = _value("repeats", DT.INT64, [2])
    one = _value("one", DT.FLOAT, [])
    one.const_value = ir.tensor(np.array(1, np.float32))
    before = _value("before", DT.FLOAT, [2, 3])
    tiled = _value("tiled", DT.FLOAT, [None, None])
    out = _value("out", DT.FLOAT, [None, None])
    nodes = []
    live_repeats = repeats
    if live == "produced":
        live_repeats = _value("live_repeats", DT.INT64, [2])
        nodes.append(ir.node("Identity", [repeats], outputs=[live_repeats],
                             name="live_repeats_producer"))
    nodes.extend([
        ir.node("Add", [x, one], outputs=[before], name="live_before"),
        ir.node("Tile", [before, live_repeats], outputs=[tiled], name="live_tile"),
        ir.node("Add", [tiled, one], outputs=[out], name="live_after"),
    ])
    model = ir.to_proto(ir.Model(ir.Graph(
        [x, repeats], [out], nodes=nodes, initializers=[one],
        opset_imports={"": 13}, name="live_tile_add"), ir_version=11)).SerializeToString()
    session = _session(model, profile=tmp_path / "live_add")
    for invocation, values in enumerate([[1, 2], [2, 1], [0, 3], [1, 2]]):
        data = (np.arange(6, dtype=np.float32) + invocation).reshape(2, 3)
        repeats_value = np.array(values, np.int64)
        _exact(session.run(None, {"x": data, "repeats": repeats_value})[0],
               _expected(data + 1, values) + 1)
    _ownership(session)


@pytest.mark.parametrize("fixture,neighbor", [
    ("test_tile_live_partition_fixture", "Transpose"),
    ("test_tile_live_add_partition_fixture", "Add"),
])
def test_tile_live_actual_routes(fixture, neighbor, tmp_path):
    library = Path(os.environ["ONNXRUNTIME_MLX_EP_LIB"]).resolve(strict=True)
    assert library.is_file()
    for live in ["boundary", "produced"]:
        trace = tmp_path / f"{neighbor}_{live}.json"
        subprocess.run(
            [sys.executable, "-m", "pytest",
             f"{Path(__file__).resolve()}::{fixture}[{live}]",
             "-q", "-p", "no:cacheprovider"],
            env={**os.environ, "ONNXRUNTIME_MLX_EP_LIB": str(library),
                 "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
                 "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0",
                 "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
            check=True, timeout=120)
        paths = [event["args"] for event in json.loads(trace.read_text())
                 if event.get("cat") == "ep.path"]
        tiles = [event for event in paths if "Tile" in event["ops"].split(",")]
        assert len(tiles) == 4 and all(
            event["path"] == "eager" and event["ops"] == "Tile"
            and event["node_names"] == "live_tile" and event["cache"] == "n/a"
            for event in tiles), paths
        assert len({event["partition_id"] for event in tiles}) == 1, paths
        neighbors = [event for event in paths if event["path"] == "general"
                     and neighbor in event["ops"].split(",")]
        assert neighbors, paths
        before = [event for event in neighbors if "live_before" in event["node_names"].split(",")]
        assert [event["cache"] for event in before] == ["MISS", "HIT", "HIT", "HIT"], paths
        assert len({event["partition_id"] for event in before}) == 1, paths


def _control_model(kind):
    x = _value("x", DT.FLOAT, [2, 3])
    repeats = _value("repeats", DT.INT64, [2])
    result = _value("result", DT.FLOAT, [None, None])
    inputs = [x, repeats]
    outputs = [result]
    if kind in ("If", "NestedIf"):
        condition = _value("condition", DT.BOOL, [])
        inputs.append(condition)
        branches = {}
        for branch in ["then_branch", "else_branch"]:
            local = _value(f"{branch}_repeats", DT.INT64, [2])
            out = _value(f"{branch}_out", DT.FLOAT, [None, None])
            one = _value(f"{branch}_one", DT.INT64, [2])
            one.const_value = ir.tensor(np.ones(2, np.int64))
            producer = ir.node("Identity", [repeats], outputs=[local]) if (
                branch == "then_branch") else ir.node("Add", [repeats, one], outputs=[local])
            branches[branch] = ir.Graph(
                [], [out], nodes=[producer, ir.node("Tile", [x, local], outputs=[out])],
                initializers=[] if branch == "then_branch" else [one],
                opset_imports={"": 13}, name=branch)
        inner = _value("inner", DT.FLOAT, [None, None]) if kind == "NestedIf" else result
        node = ir.node("If", [condition], outputs=[inner], attributes=branches,
                       name="live_tile_If")
        if kind == "NestedIf":
            outer_condition = _value("outer_condition", DT.BOOL, [])
            inputs.append(outer_condition)
            then = ir.Graph([], [inner], nodes=[node], opset_imports={"": 13}, name="outer_then")
            other = _value("outer_else_out", DT.FLOAT, [None, None])
            otherwise = ir.Graph(
                [], [other], nodes=[ir.node("Tile", [x, repeats], outputs=[other])],
                opset_imports={"": 13}, name="outer_else")
            node = ir.node("If", [outer_condition], outputs=[result],
                           attributes={"then_branch": then, "else_branch": otherwise},
                           name="live_tile_NestedIf")
    elif kind == "Loop":
        trip = _value("trip", DT.INT64, [])
        condition = _value("condition", DT.BOOL, [])
        inputs.extend([trip, condition])
        iteration = _value("iteration", DT.INT64, [])
        body_condition = _value("body_condition", DT.BOOL, [])
        carried = _value("carried", DT.INT64, [2])
        last = _value("last", DT.FLOAT, [None, None])
        next_condition = _value("next_condition", DT.BOOL, [])
        next_repeats = _value("next_repeats", DT.INT64, [2])
        tiled = _value("body_tiled", DT.FLOAT, [None, None])
        indices = _value("indices", DT.INT64, [2])
        indices.const_value = ir.tensor(np.array([1, 0], np.int64))
        body = ir.Graph(
            [iteration, body_condition, carried, last], [next_condition, next_repeats, tiled],
            nodes=[ir.node("Identity", [body_condition], outputs=[next_condition]),
                   ir.node("Tile", [x, carried], outputs=[tiled]),
                   ir.node("Gather", [carried, indices], outputs=[next_repeats])],
            initializers=[indices], opset_imports={"": 13}, name="tile_loop")
        final_repeats = _value("final_repeats", DT.INT64, [2])
        outputs = [final_repeats, result]
        node = ir.node("Loop", [trip, condition, repeats, x], outputs=outputs,
                       attributes={"body": body}, name="live_tile_Loop")
    else:
        assert kind == "Scan"
        # Every slice has the same repeat values in each invocation; no ragged outputs.
        repeats = _value("repeats", DT.INT64, [3, 2])
        inputs = [x, repeats]
        body_repeats = _value("body_repeats", DT.INT64, [2])
        tiled = _value("scan_tiled", DT.FLOAT, [None, None])
        body = ir.Graph(
            [body_repeats], [tiled],
            nodes=[ir.node("Tile", [x, body_repeats], outputs=[tiled])],
            opset_imports={"": 13}, name="tile_scan")
        result = _value("result", DT.FLOAT, [3, None, None])
        outputs = [result]
        node = ir.node("Scan", [repeats], outputs=outputs,
                       attributes={"body": body, "num_scan_inputs": 1}, name="live_tile_Scan")
    return ir.to_proto(ir.Model(ir.Graph(
        inputs, outputs, nodes=[node], opset_imports={"": 13}, name=f"live_tile_{kind}"),
        ir_version=11)).SerializeToString()


@pytest.mark.parametrize("kind", ["If", "NestedIf", "Loop", "Scan"])
@pytest.mark.parametrize("eager", [False, True])
def test_tile_live_recursive_freshness(kind, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_control_model(kind), profile=tmp_path / f"tile_{kind}")
    for invocation, values in enumerate([[1, 2], [2, 1], [1, 2], [1, 1]]):
        data = (np.arange(6, dtype=np.float32) + invocation * 10).reshape(2, 3)
        repeats = np.array(values, np.int64)
        feeds = {"x": data, "repeats": repeats}
        if kind in ("If", "NestedIf"):
            selected = bool(invocation % 2)
            feeds["condition"] = np.array(selected)
            actual_repeats = repeats if selected else repeats + 1
            if kind == "NestedIf":
                outer_selected = invocation != 2
                feeds["outer_condition"] = np.array(outer_selected)
                if not outer_selected:
                    actual_repeats = repeats
            expected = _expected(data, actual_repeats)
        elif kind == "Loop":
            trip = invocation + 1
            feeds.update(trip=np.array(trip, np.int64), condition=np.array(True))
            last_repeats = repeats[::-1] if trip % 2 == 0 else repeats
            expected = _expected(data, last_repeats)
        else:
            feeds["repeats"] = np.repeat(repeats[None], 3, axis=0)
            expected = np.stack([_expected(data, repeats)] * 3)
        originals = {name: value.tobytes() for name, value in feeds.items()}
        actual = session.run(None, feeds)
        _exact(actual[-1], expected)
        if kind == "Loop":
            _exact(actual[0], repeats[::-1] if trip % 2 else repeats)
        assert all(feeds[name].tobytes() == value for name, value in originals.items())
    _ownership(session)


def test_tile_live_recursive_actual_routes(tmp_path):
    library = Path(os.environ["ONNXRUNTIME_MLX_EP_LIB"]).resolve(strict=True)
    assert library.is_file()
    for kind in ["If", "NestedIf", "Loop", "Scan"]:
        trace = tmp_path / f"recursive_{kind}.json"
        subprocess.run(
            [sys.executable, "-m", "pytest",
             f"{Path(__file__).resolve()}::test_tile_live_recursive_freshness[False-{kind}]",
             "-q", "-p", "no:cacheprovider"],
            env={**os.environ, "ONNXRUNTIME_MLX_EP_LIB": str(library),
                 "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
                 "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0",
                 "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
            check=True, timeout=120)
        paths = [event["args"] for event in json.loads(trace.read_text())
                 if event.get("cat") == "ep.path"]
        expected_op = "If" if kind == "NestedIf" else kind
        assert len(paths) == 4 and all(event["ops"] == expected_op
                                      and event["path"] == "eager"
                                      and event["cache"] == "n/a"
                                      and event["node_names"] == f"live_tile_{kind}"
                                      for event in paths), paths
        assert len({event["partition_id"] for event in paths}) == 1, paths
