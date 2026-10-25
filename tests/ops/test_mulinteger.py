"""MulInteger v1 source regressions; runtime results require separately authorized CI."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort
from onnx import TensorProto as T, helper, numpy_helper
import pytest


FORMS = {
    "neither": ["a", "", "b"],
    "neither_trailing": ["a", "", "b", ""],
    "a": ["a", "az", "b"],
    "a_trailing": ["a", "az", "b", ""],
    "b": ["a", "", "b", "bz"],
    "both": ["a", "az", "b", "bz"],
}
FAIL = ort.capi.onnxruntime_pybind11_state.Fail
EP_FAIL = ort.capi.onnxruntime_pybind11_state.EPFail
INVALID_GRAPH = ort.capi.onnxruntime_pybind11_state.InvalidGraph


def _oracle(a, b, az=None, bz=None):
    a, b = np.broadcast_arrays(a, b)
    az, bz = 0 if az is None else int(az.item()), 0 if bz is None else int(bz.item())
    out = np.empty(a.shape, dtype=np.int32)
    for index in np.ndindex(a.shape):
        # Python integers, not narrow NumPy subtraction/multiplication.
        out[index] = (int(a[index]) - az) * (int(b[index]) - bz)
    return out


def _model(a, b, az, bz, *, form="both", source="boundary", symbolic=False,
           unknown_payload=False, unknown_zp=False, opset=21):
    arrays = {"a": a, "b": b, "az": az, "bz": bz}
    slots = FORMS[form].copy()
    nodes, inputs, initializers, feeds = [], [], [], {}
    for name in dict.fromkeys(slot for slot in slots if slot):
        array = arrays[name]
        dtype = helper.np_dtype_to_tensor_dtype(array.dtype)
        is_zp = name in ("az", "bz")
        if source == "initializer" and is_zp:
            initializers.append(numpy_helper.from_array(array, name))
            continue
        shape = (None if (unknown_zp if is_zp else unknown_payload)
                 else [] if is_zp else [None] * array.ndim if symbolic else array.shape)
        inputs.append(helper.make_tensor_value_info(name, dtype, shape))
        feeds[name] = array
        if source == "produced":
            live = f"live_{name}"
            nodes.append(helper.make_node("Identity", [name], [live]))
            slots = [live if slot == name else slot for slot in slots]
    nodes.append(helper.make_node("MulInteger", slots, ["y"], domain="com.microsoft",
                                  name="mulinteger_target"))
    output_shape = (None if unknown_payload else np.broadcast_shapes(a.shape, b.shape))
    if symbolic and output_shape is not None:
        output_shape = [None] * len(output_shape)
    return helper.make_model(
        helper.make_graph(nodes, "mulinteger", inputs, [
            helper.make_tensor_value_info("y", T.INT32, output_shape)], initializers),
        opset_imports=[helper.make_opsetid("", opset), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    ), feeds


def _session(model, profile=None):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    session = ort.InferenceSession(model.SerializeToString(), options,
                                   providers=["MLXExecutionProvider"], enable_fallback=False)
    session.disable_fallback()
    return session


def _ownership(session):
    events = json.loads(Path(session.end_profiling()).read_text())
    kernels = [e for e in events if e.get("cat") == "Node"
               and e.get("name", "").endswith("_kernel_time")]
    assert kernels, "must record actual provider kernel events"
    assert all(e.get("args", {}).get("provider") == "MLXExecutionProvider"
               for e in kernels), kernels


def _exact(actual, expected):
    assert actual.dtype == expected.dtype == np.dtype(np.int32)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def _values(dtype, shape, phase):
    limits = np.iinfo(dtype)
    values = [limits.min, limits.max, 0, 1]
    return np.asarray([values[(i + phase) % 4] for i in range(int(np.prod(shape)))],
                      dtype=dtype).reshape(shape)


def _expected(arrays, form):
    a, b, az, bz = arrays
    return _oracle(a, b, az if "az" in FORMS[form] else None,
                   bz if "bz" in FORMS[form] else None)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("source", ["boundary", "initializer", "produced"])
@pytest.mark.parametrize("eager", [False, True])
def test_optional_forms_live_exact(dtype, form, source, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    limits = np.iinfo(dtype)
    arrays = [_values(dtype, (2, 1), 0), _values(dtype, (1, 3), 1),
              np.asarray(limits.min, dtype), np.asarray(limits.max, dtype)]
    model, feeds = _model(*arrays, form=form, source=source)
    session = _session(model, tmp_path / "optional")
    retained = []
    for phase in [0, 1, 2, 0]:
        arrays[0][...] = _values(dtype, arrays[0].shape, phase)
        arrays[1][...] = _values(dtype, arrays[1].shape, phase + 1)
        if source != "initializer":
            arrays[2][...] = [limits.min, limits.max, 0][phase]
            arrays[3][...] = [limits.max, 0, limits.min][phase]
        snapshots = {name: value.copy() for name, value in feeds.items()}
        expected = _expected(arrays, form)
        actual = session.run(None, feeds)[0]
        _exact(actual, expected)
        retained.append((actual, expected.copy()))
        for name, snapshot in snapshots.items():
            np.testing.assert_array_equal(feeds[name], snapshot)
    for actual, expected in retained:
        _exact(actual, expected)
    _ownership(session)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("eager", [False, True])
def test_literal_extreme_products(dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    lo, hi = np.iinfo(dtype).min, np.iinfo(dtype).max
    a = np.asarray([hi, lo, hi, lo], dtype)
    b = np.asarray([hi, lo, lo, hi], dtype)
    az = np.asarray(lo, dtype)
    bz = np.asarray(lo, dtype)
    model, feeds = _model(a, b, az, bz)
    session = _session(model)
    _exact(session.run(None, feeds)[0], np.asarray([65025, 0, 0, 0], np.int32))
    bz[...] = hi
    _exact(session.run(None, feeds)[0], np.asarray([0, 0, -65025, 0], np.int32))
    _exact(session.run(None, feeds)[0], _oracle(a, b, az, bz))


@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("shapes", [
    ((), ()), ((), (2, 3)), ((3,), (2, 1)), ((2, 1), (1, 3)),
    ((2, 1, 3), (4, 3)), ((0,), (1,)), ((2, 0, 3), (1, 1, 3)),
])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("form", list(FORMS))
def test_scalar_broadcast_empty(dtype, shapes, eager, form, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    arrays = [_values(dtype, shapes[0], 0), _values(dtype, shapes[1], 1),
              np.asarray(np.iinfo(dtype).max, dtype), np.asarray(np.iinfo(dtype).min, dtype)]
    model, feeds = _model(*arrays, form=form)
    session = _session(model, tmp_path / "geometry")
    _exact(session.run(None, feeds)[0], _expected(arrays, form))
    _ownership(session)


@pytest.mark.parametrize("eager", [False, True])
def test_symbolic_shape_replay(eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    arrays = [_values(np.int8, (2, 1), 0), _values(np.int8, (1, 3), 1),
              np.asarray(-128, np.int8), np.asarray(127, np.int8)]
    model, _ = _model(*arrays, symbolic=True)
    session = _session(model, tmp_path / "symbolic")
    for phase, (a_shape, b_shape) in enumerate([
        ((2, 1), (1, 3)), ((2, 1), (1, 3)),
        ((4, 1), (1, 2)), ((4, 1), (1, 2)), ((2, 1), (1, 3)),
    ]):
        arrays = [_values(np.int8, a_shape, phase), _values(np.int8, b_shape, phase + 1),
                  np.asarray(-128 + phase, np.int8), np.asarray(127 - phase, np.int8)]
        feeds = dict(zip(["a", "b", "az", "bz"], arrays, strict=True))
        _exact(session.run(None, feeds)[0], _oracle(*arrays))
    _ownership(session)


@pytest.mark.parametrize("eager", [False, True])
def test_unknown_payload_rank_replay(eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    z = np.asarray(0, np.uint8)
    model, _ = _model(z, z, z, z, unknown_payload=True)
    session = _session(model, tmp_path / "unknown_rank")
    for phase, shapes in enumerate([((), ()), ((2, 1), (1, 3)),
                                    ((2, 0, 3), (1, 1, 3)), ((), ())]):
        arrays = [_values(np.uint8, shapes[0], phase), _values(np.uint8, shapes[1], phase + 1),
                  np.asarray(phase, np.uint8), np.asarray(255 - phase, np.uint8)]
        _exact(session.run(None, dict(zip(["a", "b", "az", "bz"], arrays, strict=True)))[0],
               _oracle(*arrays))
    _ownership(session)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("bad_slot,bad_shape", [
    ("az", (1,)), ("az", (0,)), ("az", (1, 1)), ("bz", (1,)), ("bz", (2,)),
    ("broadcast", None),
])
@pytest.mark.parametrize("empty", [False, True])
def test_runtime_validation_recovers(eager, bad_slot, bad_shape, empty, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    arrays = [_values(np.uint8, (0 if empty else 2, 1), 0),
              _values(np.uint8, (1, 3), 1), np.asarray(255, np.uint8), np.asarray(0, np.uint8)]
    model, feeds = _model(*arrays, unknown_payload=True, unknown_zp=True)
    session = _session(model)
    _exact(session.run(None, feeds)[0], _oracle(*arrays))
    bad = feeds.copy()
    if bad_slot == "broadcast":
        bad["b"] = np.ones((3, 2), np.uint8)
        message = "MulInteger: broadcast/multiply"
    else:
        bad[bad_slot] = np.zeros(bad_shape, np.uint8)
        message = "MulInteger: zero point.*scalar"
    with pytest.raises(EP_FAIL, match=message):
        session.run(None, bad)
    feeds["az"][...] = 1
    feeds["bz"][...] = 254
    _exact(session.run(None, feeds)[0], _oracle(*arrays))


@pytest.mark.parametrize("shape", [[1], [0], [2], [1, 1], [None]])
@pytest.mark.parametrize("slot", ["az", "bz"])
def test_static_nonscalar_zero_points_declined(shape, slot):
    z = np.asarray(0, np.uint8)
    model, _ = _model(z, z, z, z)
    value = next(v for v in model.graph.input if v.name == slot)
    value.CopyFrom(helper.make_tensor_value_info(slot, T.UINT8, shape))
    with pytest.raises(FAIL, match="CPU EP.*fallback.*disabled"):
        _session(model)


@pytest.mark.parametrize("case", [
    "mixed_data", "mixed_zp", "output", "missing_a", "missing_b",
    "short", "long", "extra_output", "attribute",
])
def test_loader_rejects_malformed_schema(case):
    z = np.asarray(0, np.uint8)
    model, _ = _model(z, z, z, z)
    node = model.graph.node[-1]
    if case in ("mixed_data", "mixed_zp"):
        name = "b" if case == "mixed_data" else "az"
        next(v for v in model.graph.input if v.name == name).type.tensor_type.elem_type = T.INT8
    elif case == "output":
        model.graph.output[0].type.tensor_type.elem_type = T.UINT8
    elif case in ("missing_a", "missing_b"):
        node.input[0 if case == "missing_a" else 2] = ""
    elif case == "short":
        del node.input[2:]
    elif case == "long":
        node.input.append("")
    elif case == "extra_output":
        node.output.append("extra")
        model.graph.output.append(helper.make_tensor_value_info("extra", T.INT32, []))
    else:
        node.attribute.append(helper.make_attribute("axis", 0))
    # These are schema/type-inference rejections, not provider-claim evidence.
    with pytest.raises((INVALID_GRAPH, FAIL)):
        _session(model)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("eager", [False, True])
def test_recursive_integer_branch_freshness(depth, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    arrays = [_values(np.int8, (2, 1), 0), _values(np.int8, (1, 3), 1),
              np.asarray(-128, np.int8), np.asarray(127, np.int8)]
    model, feeds = _model(*arrays, source="produced", symbolic=True)
    body = helper.make_graph(list(model.graph.node), "integer_branch", [],
                             list(model.graph.output))
    for level in range(depth):
        node = helper.make_node("If", ["condition"], ["y"],
                                then_branch=body, else_branch=body)
        body = helper.make_graph([node], f"integer_parent_{level}", [],
                                 list(model.graph.output))
    del model.graph.node[:]
    model.graph.node.extend(body.node)
    model.graph.input.append(helper.make_tensor_value_info("condition", T.BOOL, []))
    session = _session(model, tmp_path / "recursive")
    for phase, flag in enumerate([True, False, True, False]):
        for index, array in enumerate(arrays):
            array[...] = _values(np.int8, array.shape, phase + index)
        feeds["condition"] = np.asarray(flag)
        _exact(session.run(None, feeds)[0], _oracle(*arrays))
    _ownership(session)


def test_recursive_bad_zero_point_metadata_declines_parent():
    z = np.asarray(0, np.uint8)
    model, _ = _model(z, z, z, z)
    model.graph.input[1].CopyFrom(helper.make_tensor_value_info("az", T.UINT8, [1]))
    # Keep the invalid-but-schema-typed parameter in both lexical branches.
    body = helper.make_graph(list(model.graph.node), "invalid_integer_branch", [],
                             list(model.graph.output))
    del model.graph.node[:]
    model.graph.node.append(helper.make_node("If", ["condition"], ["y"],
                                             then_branch=body, else_branch=body))
    model.graph.input.append(helper.make_tensor_value_info("condition", T.BOOL, []))
    with pytest.raises(FAIL, match="CPU EP.*fallback.*disabled"):
        _session(model)


@pytest.mark.parametrize("kind", ["Loop", "Scan"])
@pytest.mark.parametrize("eager", [False, True])
def test_recursive_integer_loop_scan(kind, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    arrays = [_values(np.int8, (2, 1), 0), _values(np.int8, (1, 3), 1),
              np.asarray(-128, np.int8), np.asarray(127, np.int8)]
    model, feeds = _model(*arrays)
    initializers = [numpy_helper.from_array(np.zeros((2, 3), np.int32), "initial_state")]
    body_nodes = [
        helper.make_node("MulInteger", ["a", "az", "b", "bz"], ["product"],
                         domain="com.microsoft"),
        helper.make_node("Add", ["state", "product"], ["next_state"]),
    ]
    state = helper.make_tensor_value_info("state", T.INT32, [2, 3])
    next_state = helper.make_tensor_value_info("next_state", T.INT32, [2, 3])
    if kind == "Loop":
        body = helper.make_graph(
            [*body_nodes, helper.make_node("Identity", ["condition_in"], ["condition_out"])],
            "integer_loop",
            [helper.make_tensor_value_info("iteration", T.INT64, []),
             helper.make_tensor_value_info("condition_in", T.BOOL, []), state],
            [helper.make_tensor_value_info("condition_out", T.BOOL, []), next_state],
        )
        initializers.extend([
            numpy_helper.from_array(np.asarray(2, np.int64), "trip"),
            numpy_helper.from_array(np.asarray(True), "condition"),
        ])
        node = helper.make_node("Loop", ["trip", "condition", "initial_state"], ["y"],
                                body=body)
    else:
        # Each of two scan slices contributes the same exact live product.
        body_nodes[0].input[0] = "slice"
        body_nodes.append(helper.make_node("Identity", ["next_state"], ["prefix"]))
        body = helper.make_graph(
            body_nodes, "integer_scan",
            [state, helper.make_tensor_value_info("slice", T.INT8, [2, 1])],
            [next_state, helper.make_tensor_value_info("prefix", T.INT32, [2, 3])],
        )
        node = helper.make_node("Scan", ["initial_state", "a"], ["y", "prefixes"],
                                body=body, num_scan_inputs=1)
        value = next(v for v in model.graph.input if v.name == "a")
        value.CopyFrom(helper.make_tensor_value_info("a", T.INT8, [2, 2, 1]))
        model.graph.output.append(helper.make_tensor_value_info("prefixes", T.INT32, [2, 2, 3]))
    del model.graph.node[:]
    model.graph.node.append(node)
    model.graph.initializer.extend(initializers)
    session = _session(model, tmp_path / kind)
    for phase in range(3):
        for index, array in enumerate(arrays):
            array[...] = _values(np.int8, array.shape, phase + index)
        feeds["a"] = np.stack([arrays[0], arrays[0]]) if kind == "Scan" else arrays[0]
        expected = _oracle(*arrays)
        outputs = session.run(None, feeds)
        _exact(outputs[0], expected * np.int32(2))
        if kind == "Scan":
            _exact(outputs[1], np.stack([expected, expected * np.int32(2)]))
    _ownership(session)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("opset", [14, 26])
def test_modern_default_domain_imports(opset, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    a, b = np.asarray([255, 0], np.uint8), np.asarray([255, 255], np.uint8)
    z = np.asarray(0, np.uint8)
    model, feeds = _model(a, b, z, z, form="neither", opset=opset)
    _exact(_session(model).run(None, feeds)[0], _oracle(a, b))


@pytest.mark.parametrize("seq", [1, 2])
def test_attention_cluster_integer_freshness(seq, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PREFILL_COMPILE", "0")
    arrays = [_values(np.int8, (1, seq, 8), 0), np.asarray(127, np.int8),
              np.asarray(-128, np.int8), np.asarray(127, np.int8)]
    model, feeds = _model(*arrays)
    model.graph.node.extend([
        helper.make_node("Cast", ["y"], ["query"], to=T.FLOAT),
        helper.make_node(
            "GroupQueryAttention",
            ["query", "key", "value", "past_key", "past_value", "lengths", "total"],
            ["attention", "present_key", "present_value"], domain="com.microsoft",
            num_heads=1, kv_num_heads=1, do_rotary=0,
        ),
    ])
    for name, array in {
        "key": np.zeros((1, seq, 8), np.float32),
        "value": np.ones((1, seq, 8), np.float32),
        "past_key": np.zeros((1, 1, 1, 8), np.float32),
        "past_value": np.ones((1, 1, 1, 8), np.float32),
        "lengths": np.asarray([seq], np.int32),
        "total": np.asarray([seq + 1], np.int32),
    }.items():
        feeds[name] = array
        model.graph.input.append(helper.make_tensor_value_info(
            name, helper.np_dtype_to_tensor_dtype(array.dtype), array.shape))
    model.graph.output.extend([
        helper.make_tensor_value_info("attention", T.FLOAT, [1, seq, 8]),
        helper.make_tensor_value_info("present_key", T.FLOAT, [1, 1, seq + 1, 8]),
        helper.make_tensor_value_info("present_value", T.FLOAT, [1, 1, seq + 1, 8]),
    ])
    session = _session(model, tmp_path / "attention_cluster")
    for phase in range(4):
        for index, array in enumerate(arrays):
            array[...] = _values(np.int8, array.shape, phase + index)
        feeds["value"].fill(phase + 1)
        feeds["past_value"].fill(phase + 1)
        integer, attention, key, value = session.run(None, feeds)
        _exact(integer, _oracle(*arrays))
        # All keys are zero and all values equal, independently of the live integer query.
        np.testing.assert_allclose(attention, np.full((1, seq, 8), phase + 1, np.float32),
                                   rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(key, np.zeros((1, 1, seq + 1, 8), np.float32))
        np.testing.assert_array_equal(value, np.full((1, 1, seq + 1, 8), phase + 1, np.float32))
    _ownership(session)


def test_actual_capture_routes(tmp_path):
    cases = [
        ("general", "test_symbolic_shape_replay[False]", ["MISS", "HIT", "RETRACE", "HIT", "HIT"]),
        ("general", "test_optional_forms_live_exact[False-boundary-both-int8]",
         ["MISS", "HIT", "HIT", "HIT"]),
        ("eager", "test_symbolic_shape_replay[True]", ["n/a"] * 5),
        ("general", "test_recursive_integer_branch_freshness[False-1]",
         ["MISS", "RETRACE", "HIT", "HIT"]),
        ("general", "test_recursive_integer_loop_scan[False-Loop]", ["MISS", "HIT", "HIT"]),
        ("general", "test_recursive_integer_loop_scan[False-Scan]", ["MISS", "HIT", "HIT"]),
        ("decode", "test_attention_cluster_integer_freshness[1]", ["MISS", "HIT", "HIT", "HIT"]),
        ("prefill", "test_attention_cluster_integer_freshness[2]", ["MISS", "HIT", "HIT", "HIT"]),
    ]
    for index, (route, test, caches) in enumerate(cases):
        trace = tmp_path / f"route_{index}.json"
        subprocess.run(
            [sys.executable, "-m", "pytest", f"{Path(__file__).name}::{test}",
             "-q", "-p", "no:cacheprovider"], cwd=Path(__file__).parent, check=True, timeout=120,
            env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
                 "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        )
        paths = [event["args"] for event in json.loads(trace.read_text())
                 if event.get("cat") == "ep.path"]
        assert paths and all(p["path"] == route for p in paths), paths
        assert [p["cache"] for p in paths] == caches, paths
        expected = ("MulInteger,Cast,GroupQueryAttention" if "attention_cluster" in test
                    else "Loop" if test.endswith("-Loop]")
                    else "Scan" if test.endswith("-Scan]")
                    else "If" if "recursive" in test else "MulInteger")
        assert all(p["ops"] == expected for p in paths), paths
        assert len({p["partition_id"] for p in paths}) == 1, paths
