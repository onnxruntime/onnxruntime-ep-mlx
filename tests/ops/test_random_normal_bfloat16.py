"""BF16 Normal rounds the existing F32 draw only after its FLOAT affine transform."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from onnx import AttributeProto, TensorProto as T, helper, load_model_from_string
import onnxruntime as ort
import pytest


OPS = ["RandomNormal", "RandomNormalLike"]


def _model(op, *, shape=None, input_type=T.FLOAT, output_type=T.BFLOAT16,
           explicit_dtype=True, opset=22, seed=41.0, mean=0.0, scale=1.0,
           nested=False, omit_parameters=False):
    attrs = {"mean": mean, "scale": scale} if not omit_parameters else {}
    if explicit_dtype:
        attrs["dtype"] = output_type
    if seed is not None:
        attrs["seed"] = seed
    nodes, inputs = [], []
    random_inputs = []
    if op.endswith("Like"):
        source_type = T.FLOAT if input_type == T.BFLOAT16 else input_type
        inputs.append(helper.make_tensor_value_info("x", source_type, shape))
        random_inputs = ["x"]
        if input_type == T.BFLOAT16:
            nodes.append(helper.make_node("Cast", ["x"], ["bf16_x"], to=T.BFLOAT16))
            random_inputs = ["bf16_x"]
    elif shape:
        attrs["shape"] = list(shape)
    if nested:
        branches = [
            helper.make_graph(
                [helper.make_node(op, random_inputs, [name], **attrs)], name, [],
                [helper.make_tensor_value_info(name, output_type, shape)],
            )
            for name in ["then", "else"]
        ]
        inputs.append(helper.make_tensor_value_info("condition", T.BOOL, []))
        nodes.append(helper.make_node("If", ["condition"], ["draw"],
                                      then_branch=branches[0], else_branch=branches[1]))
    else:
        nodes.append(helper.make_node(op, random_inputs, ["draw"], **attrs))
        if not op.endswith("Like") and not shape:
            nodes[-1].attribute.append(helper.make_attribute(
                "shape", [], attr_type=AttributeProto.INTS,
            ))
    result = "draw"
    if output_type == T.BFLOAT16:
        result = "y"
        nodes.append(helper.make_node("Cast", ["draw"], [result], to=T.FLOAT))
    return helper.make_model(
        helper.make_graph(nodes, "normal_bfloat16", inputs, [
            helper.make_tensor_value_info(
                result, T.FLOAT if output_type == T.BFLOAT16 else output_type, shape,
            ),
        ]),
        opset_imports=[helper.make_opsetid("", opset)], ir_version=11,
    ).SerializeToString()


def _session(model, *, profile=None):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    session = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"],
                                   enable_fallback=False)
    session.disable_fallback()
    return session


def _assert_profile(session):
    path = Path(session.end_profiling())
    events = json.loads(path.read_text())
    path.unlink()
    kernels = [event["args"] for event in events
               if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time")]
    assert kernels
    assert all(event.get("provider") == "MLXExecutionProvider" for event in kernels), kernels


def _bf16_round(values):
    bits = np.asarray(values, np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return np.where(np.isnan(values), np.uint32(0x7FC00000), rounded).view(np.float32)


def _assert_affine(actual, standard, mean, scale, *, eager):
    mean, scale = np.float32(mean), np.float32(scale)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        separate = np.add(np.multiply(standard, scale, dtype=np.float32), mean,
                          dtype=np.float32)
        fused = (standard.astype(np.float64) * np.float64(scale)
                 + np.float64(mean)).astype(np.float32)
    # Compilation may fuse the F32 affine multiply/add. This is not an exact-real oracle.
    expected = [_bf16_round(value).view(np.uint32) for value in [separate, fused]]
    bits = actual.view(np.uint32)
    matches = (bits == expected[0]) | (bits == expected[1])
    if not np.all(matches):
        def describe(array, index):
            value = int(array.view(np.uint32).ravel()[index])
            exponent, fraction = (value >> 23) & 0xFF, value & 0x7FFFFF
            if exponent == 0:
                kind = "subnormal" if fraction else "zero"
            elif exponent == 0xFF:
                kind = "nan" if fraction else "infinity"
            else:
                kind = "normal"
            return {
                "bits": f"0x{value:08x}", "class": kind, "sign": value >> 31,
                "low16": f"0x{value & 0xFFFF:04x}",
                "distance_from_bf16_midpoint_low16": (value & 0xFFFF) - 0x8000,
            }

        indices = np.flatnonzero(~matches)[:32]
        diagnostic = {
            "mismatch_count": int(np.count_nonzero(~matches)),
            "compile_disabled_flag": eager,
            "actual_backend_route": "not measured for this case; flag is not route proof",
            "kernel_provider": "MLXExecutionProvider asserted by ORT kernel profile",
            "mean": describe(np.asarray(mean), 0),
            "scale": describe(np.asarray(scale), 0),
            "mismatches": [
                {
                    "index": int(index),
                    "actual": describe(actual, index),
                    "standard": describe(standard, index),
                    "separate_f32": describe(separate, index),
                    "fused_candidate_f32": describe(fused, index),
                    "separate_bf16": describe(expected[0], index),
                    "fused_candidate_bf16": describe(expected[1], index),
                }
                for index in indices
            ],
        }
        pytest.fail(json.dumps(diagnostic, sort_keys=True))


def _feeds(op, shape, input_type=T.FLOAT, call=0):
    if not op.endswith("Like"):
        return {}
    if input_type == T.STRING:
        values = np.full(shape, f"ignored-{call}", object)
    else:
        dtype = helper.tensor_dtype_to_np_dtype(
            T.FLOAT if input_type == T.BFLOAT16 else input_type,
        )
        data = [np.nan, np.inf, -0.0, call] if np.issubdtype(dtype, np.floating) else [
            0, 1, call, 3,
        ]
        values = np.resize(np.roll(np.array(data, dtype), call), shape)
    return {"x": values}


@pytest.mark.parametrize("value,bits", [
    (0.0, 0x00000000), (-0.0, 0x80000000),
    (1.0 + 2.0**-8, 0x3F800000), (1.0 + 3 * 2.0**-8, 0x3F820000),
    (2.0**-133, 0x00010000), (2.0**-134, 0x00000000),
    (3 * 2.0**-134, 0x00020000),
    (float.fromhex("0x1.fffffep127"), 0x7F800000),
])
def test_independent_bfloat_rounding_golden(value, bits):
    actual = _bf16_round(np.array([value], np.float32))
    np.testing.assert_array_equal(actual.view(np.uint32), [bits])


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("mean,scale", [
    (0.0, 1.0), (0.1001, 1.003), (-3.14159, 0.1001),
    (1.0 + 2.0**-8, 0.0), (-0.0, 0.0), (0.0, -0.0),
    (0.0, 2.0**-120), (2.0**-120, 2.0**-120),
    (2.0**-140, 0.0),
    (0.0, 2.0**126), (2.0**127, 2.0**126),
    (float.fromhex("0x1.fffffep127"), 0.0),
    (0.0, float.fromhex("0x1.fffffep127")),
    (-float.fromhex("0x1.fffffep127"), float.fromhex("0x1.fffffep127")),
])
def test_affine_against_paired_f32_draw(op, eager, mean, scale, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    shape = (2048,)
    feeds = _feeds(op, shape)
    control = _session(_model(op, shape=shape, output_type=T.FLOAT))
    standard = control.run(None, feeds)[0]
    session = _session(_model(op, shape=shape, mean=mean, scale=scale),
                       profile=tmp_path / "affine")
    actual = session.run(None, feeds)[0]
    assert actual.dtype == np.float32 and actual.shape == shape
    assert np.all((actual.view(np.uint32) & 0xFFFF) == 0)
    _assert_profile(session)
    _assert_affine(actual, standard, mean, scale, eager=eager)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("opset", [22, 26])
@pytest.mark.parametrize("shape", [(), (256,), (0, 3)])
def test_default_parameters_and_output_dtype(op, opset, shape):
    input_type = T.BFLOAT16 if op.endswith("Like") else T.FLOAT
    default_type = input_type if op.endswith("Like") else T.FLOAT
    feeds = _feeds(op, shape, input_type)
    default = _session(_model(op, shape=shape, input_type=input_type,
                              output_type=default_type, explicit_dtype=False, opset=opset,
                              omit_parameters=True))
    explicit = _session(_model(op, shape=shape, input_type=input_type,
                               output_type=default_type, opset=opset))
    np.testing.assert_array_equal(default.run(None, feeds)[0], explicit.run(None, feeds)[0])
    bfloat = _session(_model(op, shape=shape, input_type=input_type, opset=opset))
    single = _session(_model(op, shape=shape, input_type=input_type,
                             output_type=T.FLOAT, opset=opset))
    actual, standard = bfloat.run(None, feeds)[0], single.run(None, feeds)[0]
    assert actual.shape == shape and actual.dtype == np.float32
    np.testing.assert_array_equal(actual, _bf16_round(standard))


@pytest.mark.parametrize("input_type", [
    T.FLOAT, T.FLOAT16, T.BFLOAT16, T.BOOL, T.INT64, T.STRING, T.DOUBLE,
])
@pytest.mark.parametrize("seed", [None, 41.75])
@pytest.mark.parametrize("eager", [False, True])
def test_live_like_dynamic_geometry_and_payload(input_type, seed, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model("RandomNormalLike", input_type=input_type, seed=seed,
                              explicit_dtype=input_type != T.BFLOAT16))
    repeated = []
    for call, shape in enumerate([(256,), (), (2, 128), (0, 3), (256,)]):
        feeds = _feeds("RandomNormalLike", shape, input_type, call)
        before = feeds["x"].tobytes()
        actual = session.run(None, feeds)[0]
        assert feeds["x"].tobytes() == before
        assert actual.shape == shape and actual.dtype == np.float32
        assert np.isfinite(actual).all()
        assert np.all((actual.view(np.uint32) & 0xFFFF) == 0)
        if shape == (256,):
            repeated.append(actual.copy())
    assert np.array_equal(*repeated) == (seed is not None)
    if seed is not None:
        truncated = _session(_model("RandomNormalLike", input_type=input_type, seed=41.0))
        np.testing.assert_array_equal(repeated[0], truncated.run(
            None, _feeds("RandomNormalLike", (256,), input_type),
        )[0])


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("seed", [0.0, 41.75])
def test_seeded_replay_and_fractional_truncation(op, seed):
    feeds = _feeds(op, (256,))
    session = _session(_model(op, shape=(256,), seed=seed))
    first, second = [session.run(None, feeds)[0] for _ in range(2)]
    np.testing.assert_array_equal(first, second)
    truncated = _session(_model(op, shape=(256,), seed=float(int(seed))))
    np.testing.assert_array_equal(first, truncated.run(None, feeds)[0])


@pytest.mark.parametrize("eager", [False, True])
def test_unseeded_normal_live_and_empty(eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model("RandomNormal", shape=(256,), seed=None))
    outputs = [session.run(None, {})[0].copy() for _ in range(3)]
    empty = _session(_model("RandomNormal", shape=(0, 3), seed=None))
    result = empty.run(None, {})[0]
    assert result.shape == (0, 3) and result.dtype == np.float32
    outputs.append(session.run(None, {})[0])
    assert all(np.isfinite(output).all() for output in outputs)
    assert all(not np.array_equal(left, right) for left, right in zip(outputs, outputs[1:]))


@pytest.mark.parametrize("op", OPS)
def test_distribution_not_only_paired_prng(op):
    shape, mean, scale = (32768,), 0.125, 1.5
    session = _session(_model(op, shape=shape, mean=mean, scale=scale))
    values = session.run(None, _feeds(op, shape))[0].astype(np.float64)
    assert np.isfinite(values).all()
    assert abs(values.mean() - mean) < 0.05
    assert abs(values.std() - scale) < 0.05
    standardized = (values - mean) / scale
    assert abs(np.median(standardized)) < 0.05
    assert 0.035 < np.mean(np.abs(standardized) > 2) < 0.06


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("opset", [1, 21])
def test_bfloat_output_before_22_is_rejected(op, opset):
    proto = load_model_from_string(_model(op, shape=(256,), opset=opset))
    del proto.graph.node[-1]
    proto.graph.output[0].name = "draw"
    proto.graph.output[0].type.tensor_type.elem_type = T.BFLOAT16
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail,
                       match="CPU EP.*fallback.*disabled"):
        _session(proto.SerializeToString())


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("seed", [-1.0, float("nan"), float("inf"), -float("inf"), 2.0**64])
def test_invalid_seed_stays_declined(op, seed):
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail,
                       match="CPU EP.*fallback.*disabled"):
        _session(_model(op, shape=(256,), seed=seed))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("mean,scale", [
    (float("nan"), 1.0), (float("inf"), 1.0), (-float("inf"), 1.0),
    (0.0, -1.0), (0.0, float("inf")), (0.0, float("nan")),
])
def test_invalid_affine_attributes_stay_declined(op, mean, scale):
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail,
                       match="CPU EP.*fallback.*disabled"):
        _session(_model(op, shape=(256,), mean=mean, scale=scale))


@pytest.mark.parametrize("op", OPS)
def test_dtype_must_match_output(op):
    proto = load_model_from_string(_model(op, shape=(256,), output_type=T.FLOAT))
    for attr in proto.graph.node[0].attribute:
        if attr.name == "dtype":
            attr.i = T.BFLOAT16
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail,
                       match="Type Error.*does not match expected type"):
        _session(proto.SerializeToString())


@pytest.mark.parametrize("op", ["RandomUniform", "RandomUniformLike"])
def test_uniform_bfloat_output_stays_declined(op):
    proto = load_model_from_string(_model(op, shape=(256,)))
    attrs = proto.graph.node[0].attribute
    retained = [attr for attr in attrs if attr.name not in ["mean", "scale"]]
    del attrs[:]
    attrs.extend(retained)
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail,
                       match="CPU EP.*fallback.*disabled"):
        _session(proto.SerializeToString())


@pytest.mark.parametrize("op", OPS)
def test_double_output_stays_declined(op):
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail,
                       match="CPU EP.*fallback.*disabled"):
        _session(_model(op, shape=(256,), output_type=T.DOUBLE))


def test_unknown_wide_geometry_error_and_recovery():
    session = _session(_model("RandomNormalLike", input_type=T.STRING))
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.EPFail, match="dimension.*i32"):
        session.run(None, {"x": np.empty((0, 2**31 + 7), object)})
    result = session.run(None, {"x": np.full(256, "ignored", object)})[0]
    assert result.shape == (256,) and result.dtype == np.float32
    assert np.isfinite(result).all()


@pytest.mark.parametrize("op", OPS)
def test_known_wide_geometry_stays_declined(op):
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail,
                       match="CPU EP.*fallback.*disabled"):
        _session(_model(op, shape=(0, 2**31 + 7), input_type=T.STRING))


@pytest.mark.parametrize("case", ["numeric-seeded", "string-seeded", "numeric-live", "if-live"])
def test_bfloat_route_case(case, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE", "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PREFILL_COMPILE", "0")
    input_type = T.STRING if case == "string-seeded" else T.FLOAT
    seed = None if case.endswith("live") else 41.0
    session = _session(_model("RandomNormalLike", shape=(256,), input_type=input_type,
                              seed=seed, nested=case == "if-live"))
    outputs = []
    for call in range(4):
        feeds = _feeds("RandomNormalLike", (256,), input_type, call)
        if case == "if-live":
            feeds["condition"] = np.asarray(call % 2 == 0)
        outputs.append(session.run(None, feeds)[0].copy())
    assert all(np.isfinite(output).all() for output in outputs)
    assert np.array_equal(outputs[0], outputs[-1]) == (seed is not None)


def test_bfloat_actual_routes(tmp_path):
    trace = tmp_path / "normal-bfloat.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name, "-k", "test_bfloat_route_case",
         "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path"]
    draws = [event for event in paths
             if "RandomNormalLike" in event.get("ops", "").split(",")]
    # If bodies are translated inline, so their four calls have parent-only path events.
    assert len(draws) == 12, paths
    assert sum(event["path"] == "eager" for event in draws) == 8, paths
    assert [event["cache"] for event in draws if event["path"] == "general"] == [
        "MISS", "HIT", "HIT", "HIT",
    ]
    parents = [event for event in paths if "If" in event.get("ops", "").split(",")]
    assert len(parents) == 4 and all(event["path"] == "eager" for event in parents)
