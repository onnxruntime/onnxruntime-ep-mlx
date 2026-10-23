"""RandomLike consumes geometry and an optional type, never input payloads."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from onnx import TensorProto as T, helper, numpy_helper
import onnxruntime as ort
import pytest


OPS = ["RandomNormalLike", "RandomUniformLike"]
LEGACY_TYPES = [
    T.BOOL, T.UINT8, T.UINT16, T.UINT32, T.UINT64,
    T.INT8, T.INT16, T.INT32, T.INT64, T.FLOAT16, T.FLOAT, T.DOUBLE,
    T.STRING, T.COMPLEX64, T.COMPLEX128,
]
FEED_TYPES = [dtype for dtype in LEGACY_TYPES if dtype not in [T.COMPLEX64, T.COMPLEX128]]


def _data(dtype, shape, call=0):
    if dtype == T.STRING:
        return np.full(shape, f"ignored-{call}", object)
    numpy_dtype = helper.tensor_dtype_to_np_dtype(dtype)
    if dtype == T.BOOL:
        values = [True, False]
    elif np.issubdtype(numpy_dtype, np.integer):
        limits = np.iinfo(numpy_dtype)
        values = [limits.min, limits.max, 0]
    elif dtype in [T.COMPLEX64, T.COMPLEX128]:
        values = [complex(np.nan, np.inf), complex(-0.0, 1), complex(2, -3)]
    else:
        values = [np.nan, np.inf, -np.inf, -0.0, 1]
    return np.resize(np.roll(np.array(values, dtype=numpy_dtype), call), shape)


def _model(op, dtype, *, shape=None, output_dtype=T.FLOAT, opset=22, seed=41.0,
           initializer=None, nested=False, neighbor=False, cast_bfloat=False):
    attrs = {} if output_dtype is None else {"dtype": output_dtype}
    output_dtype = dtype if output_dtype is None else output_dtype
    if seed is not None:
        attrs["seed"] = seed
    inputs = [] if initializer is not None else [
        helper.make_tensor_value_info("source" if cast_bfloat else "x",
                                     T.FLOAT if cast_bfloat else dtype, shape),
    ]
    initializers = [] if initializer is None else [initializer]
    nodes = []
    if cast_bfloat:
        nodes.append(helper.make_node("Cast", ["source"], ["x"], to=T.BFLOAT16))
    if nested:
        branches = []
        for name in ["then", "else"]:
            branches.append(helper.make_graph(
                [helper.make_node(op, ["x"], [name], **attrs)], name, [],
                [helper.make_tensor_value_info(name, output_dtype, shape)],
            ))
        inputs.append(helper.make_tensor_value_info("condition", T.BOOL, []))
        nodes.append(helper.make_node("If", ["condition"], ["draw"],
                                      then_branch=branches[0], else_branch=branches[1]))
    else:
        nodes.append(helper.make_node(op, ["x"], ["draw"], **attrs))
    outputs = [helper.make_tensor_value_info("draw", output_dtype, shape)]
    if neighbor:
        inputs.append(helper.make_tensor_value_info("bias", output_dtype, shape))
        nodes.append(helper.make_node("Add", ["draw", "bias"], ["sum"]))
        outputs.append(helper.make_tensor_value_info("sum", output_dtype, shape))
    return helper.make_model(
        helper.make_graph(nodes, "random_like_metadata", inputs, outputs,
                          initializer=initializers),
        opset_imports=[helper.make_opsetid("", opset)], ir_version=11,
    ).SerializeToString()


def _session(model, *, mixed=False, profile=None):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if not mixed:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    session = ort.InferenceSession(
        model, options,
        providers=["MLXExecutionProvider", "CPUExecutionProvider"] if mixed
        else ["MLXExecutionProvider"],
        enable_fallback=False,
    )
    session.disable_fallback()
    return session


def _assert_draw(op, output, shape, dtype):
    assert output.shape == shape
    assert output.dtype == helper.tensor_dtype_to_np_dtype(dtype)
    assert np.isfinite(output).all()
    if op == "RandomUniformLike":
        assert ((output >= 0) & (output < 1)).all()
    if output.size >= 128:
        assert np.ptp(output.astype(np.float32)) > 0


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("opset,dtype", [
    (version, dtype)
    for version in [1, 22, 26]
    for dtype in LEGACY_TYPES + ([T.BFLOAT16] if version >= 22 else [])
])
@pytest.mark.parametrize("output_dtype", [T.FLOAT, T.FLOAT16])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("shape", [(), (2, 128), (0, 3)])
def test_random_like_all_legal_initializer_types(op, opset, dtype, output_dtype, eager, shape,
                                               monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    if dtype == T.BFLOAT16:
        initializer = helper.make_tensor(
            "x", dtype, shape, np.full(shape, 0x3F80, np.uint16).tobytes(), raw=True,
        )
    else:
        initializer = numpy_helper.from_array(_data(dtype, shape), name="x")
    session = _session(_model(op, dtype, shape=shape, output_dtype=output_dtype,
                              opset=opset, initializer=initializer))
    first, second = [session.run(None, {})[0] for _ in range(2)]
    _assert_draw(op, first, shape, output_dtype)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", FEED_TYPES)
@pytest.mark.parametrize("seed", [None, 41.0])
@pytest.mark.parametrize("eager", [False, True])
def test_random_like_live_unknown_rank_and_payload(op, dtype, seed, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(op, dtype, seed=seed))
    repeated = []
    for call, shape in enumerate([(256,), (), (2, 128), (0, 3), (256,)]):
        data = _data(dtype, shape, call)
        before = data.tobytes()
        output = session.run(None, {"x": data})[0]
        assert data.tobytes() == before
        _assert_draw(op, output, shape, T.FLOAT)
        if shape == (256,):
            repeated.append(output.copy())
    assert np.array_equal(*repeated) == (seed is not None)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", [T.FLOAT, T.FLOAT16])
@pytest.mark.parametrize("eager", [False, True])
def test_random_like_default_output_dtype(op, dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(op, dtype, output_dtype=None))
    for shape in [(), (256,), (0, 3)]:
        _assert_draw(op, session.run(None, {"x": _data(dtype, shape)})[0], shape, dtype)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("seed", [None, 41.0])
@pytest.mark.parametrize("eager", [False, True])
def test_random_like_bfloat_intermediate(op, seed, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(op, T.BFLOAT16, shape=[256], seed=seed, cast_bfloat=True))
    outputs = []
    for call in range(3):
        data = _data(T.FLOAT, (256,), call)
        before = data.tobytes()
        output = session.run(None, {"source": data})[0]
        assert data.tobytes() == before
        _assert_draw(op, output, (256,), T.FLOAT)
        outputs.append(output.copy())
    assert np.array_equal(outputs[0], outputs[-1]) == (seed is not None)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("eager", [False, True])
def test_random_like_unknown_wide_dimension_error_and_recovery(op, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(op, T.STRING))
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.EPFail, match="dimension.*i32"):
        session.run(None, {"x": np.empty((0, 2**31 + 7), object)})
    _assert_draw(op, session.run(None, {"x": _data(T.STRING, (256,))})[0], (256,), T.FLOAT)


@pytest.mark.parametrize("op", OPS)
def test_random_like_known_wide_geometry_is_declined(op):
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail,
                       match="CPU EP.*fallback.*disabled"):
        _session(_model(op, T.STRING, shape=[0, 2**31 + 7]))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("case", ["numeric-seeded", "string-seeded", "string-live", "if-string"])
def test_metadata_like_route_case(op, case, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE", "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PREFILL_COMPILE", "0")
    dtype = T.FLOAT if case == "numeric-seeded" else T.STRING
    seed = None if case == "string-live" else 41.0
    session = _session(_model(op, dtype, shape=[256], seed=seed, neighbor=True,
                              nested=case == "if-string"))
    outputs = []
    for call in range(4):
        feeds = {"x": _data(dtype, (256,), call),
                 "bias": np.full(256, call, np.float32)}
        if case == "if-string":
            feeds["condition"] = np.asarray(call % 2 == 0)
        before = {name: value.tobytes() for name, value in feeds.items()}
        draw, total = session.run(None, feeds)
        _assert_draw(op, draw, (256,), T.FLOAT)
        np.testing.assert_array_equal(total, draw + feeds["bias"])
        assert all(value.tobytes() == before[name] for name, value in feeds.items())
        outputs.append(draw.copy())
    assert np.array_equal(outputs[0], outputs[-1]) == (seed is not None)


def test_random_like_metadata_actual_routes(tmp_path):
    trace = tmp_path / "random-like-metadata.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "test_metadata_like_route_case", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
        check=True, timeout=120,
    )
    events = [event["args"] for event in json.loads(trace.read_text())
              if event.get("cat") == "ep.path"]
    for op in OPS:
        draws = [event for event in events if op in event.get("ops", "").split(",")]
        assert sum(event["path"] == "eager" for event in draws) == 8
        assert [event["cache"] for event in draws if event["path"] == "general"] == [
            "MISS", "HIT", "HIT", "HIT",
        ]
    parents = [event for event in events if "If" in event.get("ops", "").split(",")]
    assert len(parents) == 8 and all(event["path"] == "eager" for event in parents)
    adds = [event for event in events if event.get("ops", "") == "Add"
            and event["path"] == "general"]
    assert len(adds) == 24
    assert sum(event["cache"] == "MISS" for event in adds) == 6
    assert sum(event["cache"] == "HIT" for event in adds) == 18


@pytest.mark.parametrize("op", OPS)
def test_random_like_cpu_string_producer(op, tmp_path):
    from onnx import load_model_from_string

    model = load_model_from_string(_model(op, T.STRING, shape=[256], neighbor=True))
    model.graph.input[0].name = "a"
    model.graph.input.append(helper.make_tensor_value_info("b", T.STRING, [256]))
    model.graph.node.insert(0, helper.make_node("StringConcat", ["a", "b"], ["x"]))
    session = _session(model.SerializeToString(), mixed=True, profile=tmp_path / op)
    feeds = {"a": _data(T.STRING, (256,)), "b": np.full(256, "-b", object),
             "bias": np.ones(256, np.float32)}
    draw, total = session.run(None, feeds)
    _assert_draw(op, draw, (256,), T.FLOAT)
    np.testing.assert_array_equal(total, draw + feeds["bias"])
    nodes = [event["args"] for event in json.loads(Path(session.end_profiling()).read_text())
             if event.get("cat") == "Node"]
    assert any(event.get("provider") == "CPUExecutionProvider"
               and event.get("op_name") == "StringConcat" for event in nodes)
    assert sum(event.get("provider") == "MLXExecutionProvider" for event in nodes) >= 2
