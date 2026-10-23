"""Stable fused LinearAttention recurrence: arithmetic, carry, and actual routing."""

import json
import os
from pathlib import Path
import subprocess
import sys

import ml_dtypes
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

import _models as m
from test_ssm_contracts import bf16_boundaries, linear_reference, strict_session
from test_ssm_quant import _linear_attention_model

DT = ir.DataType
NO_SCAN = "ONNXRUNTIME_EP_MLX_NO_LINEAR_ATTENTION_SCAN"


def typed_reference(feeds, rule, qh, h, kh, dk, dv, window=0, state_dtype=None):
    """Independent FP64 primitives with explicit rounding at every state-dtype boundary."""
    batch, time, _ = feeds["query"].shape
    dtype = state_dtype or feeds.get("past_state", feeds["query"]).dtype

    def rounded(value):
        return np.asarray(value).astype(dtype).astype(np.float64)

    q, k, v = [rounded(feeds[name]) for name in ["query", "key", "value"]]
    q = rounded(q.reshape(batch, time, qh, dk) * rounded(1 / np.sqrt(dk)))
    k = np.repeat(k.reshape(batch, time, kh, dk), h // kh, axis=2)
    v = v.reshape(batch, time, h, dv)
    if qh < h:
        q = np.repeat(q, h // qh, axis=2)
    past = feeds.get("past_state")
    state = rounded(past[-1] if window else past) if past is not None else np.zeros((batch, h, dk, dv))
    output = np.empty((batch, time, max(qh, h), dv), np.float64)
    history = []
    for token in range(time):
        if rule in ["gated", "gated_delta"]:
            decay = rounded(feeds["decay"][:, token]).reshape(batch, h, -1)
            state = rounded(state * rounded(np.exp(decay))[:, :, :, None])
        delta = v[:, token]
        if rule in ["delta", "gated_delta"]:
            retrieval = rounded(np.einsum("bhk,bhkv->bhv", k[:, token], state))
            beta = rounded(feeds["beta"][:, token, :, None])
            delta = rounded(beta * rounded(delta - retrieval))
        outer = rounded(k[:, token, :, :, None] * delta[:, :, None, :])
        state = rounded(state + outer)
        history.append(state.copy())
        readout = np.repeat(state, qh // h, axis=1) if qh > h else state
        output[:, token] = rounded(np.einsum("bhk,bhkv->bhv", q[:, token], readout))
    if window:
        state = np.zeros((window, batch, h, dk, dv), np.float64)
        if history:
            kept = history[-window:]
            state[-len(kept):] = np.stack(kept)
        elif past is not None:
            state = rounded(past)
    return output.reshape(batch, time, max(qh, h) * dv), state


def fp32_working_reference(
    feeds, rule, qh, h, kh, dk, dv, window=0, *, input_dtype=None, state_dtype=None,
):
    """FP64 primitives with FP32 carry/readout, followed by declared T/S stores."""
    input_dtype = input_dtype or feeds["query"].dtype
    state_dtype = state_dtype or feeds.get("past_state", feeds["query"]).dtype
    effective = {
        name: value.astype(state_dtype if name == "past_state" else input_dtype).astype(np.float64)
        for name, value in feeds.items()
    }

    def working(value):
        return np.asarray(value).astype(np.float32).astype(np.float64)

    batch, time, _ = effective["query"].shape
    q = effective["query"].reshape(batch, time, qh, dk)
    k = np.repeat(effective["key"].reshape(batch, time, kh, dk), h // kh, axis=2)
    v = effective["value"].reshape(batch, time, h, dv)
    if qh < h:
        q = np.repeat(q, h // qh, axis=2)
    past = effective.get("past_state")
    state = (past[-1] if window else past) if past is not None else np.zeros((batch, h, dk, dv))
    state = working(state)
    scale = working(1 / working(np.sqrt(dk)))
    output = np.empty((batch, time, max(qh, h), dv), np.float64)
    history = []
    for token in range(time):
        if rule in ["gated", "gated_delta"]:
            decay = effective["decay"][:, token]
            decay = decay.reshape(batch, h, decay.shape[-1] // h)
            state = working(state * working(np.exp(decay))[:, :, :, None])
        delta = v[:, token]
        if rule in ["delta", "gated_delta"]:
            retrieval = working(np.einsum("bhk,bhkv->bhv", k[:, token], state))
            beta = effective["beta"][:, token, :, None]
            delta = working(beta * working(delta - retrieval))
        state = working(state + working(k[:, token, :, :, None] * delta[:, :, None, :]))
        history.append(state.copy())
        readout = np.repeat(state, qh // h, axis=1) if qh > h else state
        dot = working(np.einsum("bhk,bhkv->bhv", q[:, token], readout))
        output[:, token] = working(dot * scale)
    if window:
        state = np.zeros((window, batch, h, dk, dv), np.float64)
        if history:
            kept = history[-window:]
            state[-len(kept):] = np.stack(kept)
        elif past is not None:
            state = past.copy()
    return (
        output.reshape(batch, time, max(qh, h) * dv).astype(input_dtype).astype(np.float64),
        state.astype(state_dtype).astype(np.float64),
    )


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16])
@pytest.mark.parametrize("rule", ["linear", "gated", "delta", "gated_delta"])
@pytest.mark.parametrize("geometry", [
    (1, 2, 1, 1, 1, 1, 1, 0, True, False, False),
    (2, 7, 2, 4, 1, 33, 3, 0, False, True, True),
    (1, 8, 4, 2, 1, 65, 5, 3, True, False, True),
    (1, 2, 2, 2, 2, 31, 2, 4, False, True, True),
    (1, 9, 2, 2, 2, 256, 4, 0, True, True, True),
    (1, 3, 2, 2, 2, 257, 3, 0, False, False, True),
    (2, 7, 2, 4, 1, 257, 3, 0, False, True, False),
    (1, 8, 4, 2, 1, 511, 5, 3, True, False, True),
    (1, 2, 2, 2, 2, 512, 2, 4, False, True, True),
    (1, 3, 2, 2, 2, 513, 3, 0, False, False, True),
])
def test_linear_attention_scan_arithmetic(monkeypatch, eager, dtype, rule, geometry):
    batch, time, qh, h, kh, dk, dv, window, scalar, shared, past = geometry
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    monkeypatch.delenv(NO_SCAN, raising=False)
    model, feeds = _linear_attention_model(
        rule, B=batch, T=time, q_heads=qh, kv_heads=h, key_heads=kh,
        d_k=dk, d_v=dv, with_past=past, state_window=window,
        scalar_decay=scalar, shared_beta=shared, dtype=dtype,
    )
    rng = np.random.default_rng(2430)
    for name, array in feeds.items():
        values = rng.normal(size=array.shape) * (0.05 if name == "key" else 0.1)
        if name == "decay":
            values = -np.abs(values)
        elif name == "beta":
            values = rng.uniform(0.25, 0.75, array.shape)
        feeds[name] = values.astype(array.dtype)
    fused = strict_session(model).run(None, feeds)
    typed = typed_reference(feeds, rule, qh, h, kh, dk, dv, window)
    for actual, expected in zip(fused, typed, strict=True):
        assert actual.shape == expected.shape
        assert actual.dtype == feeds["query"].dtype
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    with monkeypatch.context() as unfused:
        unfused.setenv(NO_SCAN, "1")
        reference = strict_session(model).run(None, feeds)
    for actual, expected in zip(fused, reference, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)

    reference_model, _ = _linear_attention_model(
        rule, B=batch, T=1 if window else time, q_heads=qh, kv_heads=h, key_heads=kh,
        d_k=dk, d_v=dv, with_past=True if window else past,
        scalar_decay=scalar, shared_beta=shared,
    )
    cpu = ort.InferenceSession(reference_model, providers=["CPUExecutionProvider"])
    cpu_feeds = {name: value.astype(np.float32) for name, value in feeds.items()}
    if window:
        state = cpu_feeds["past_state"][-1].copy()
        outputs, states = [], []
        for token in range(time):
            step = {name: value[:, token:token + 1] for name, value in cpu_feeds.items() if name != "past_state"}
            output, state = cpu.run(None, {**step, "past_state": state})
            outputs.append(output)
            states.append(state.copy())
        history = np.zeros((window, batch, h, dk, dv), np.float32)
        history[-min(time, window):] = np.stack(states[-window:])
        reference = [np.concatenate(outputs, axis=1), history]
    else:
        reference = cpu.run(None, cpu_feeds)
    tolerance = 3e-3 if dtype == DT.FLOAT16 else 1e-3
    for actual, expected in zip(fused, reference, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("rule", ["linear", "gated", "delta", "gated_delta"])
@pytest.mark.parametrize("geometry", [
    (1, 7, 1, 1, 1, 33, 3, 0, True, False),
    (1, 7, 1, 1, 1, 257, 3, 0, True, False),
    (1, 3, 2, 4, 1, 512, 3, 4, False, True),
    (1, 1, 4, 2, 1, 257, 3, 3, True, False),
    (1, 0, 2, 2, 2, 257, 3, 4, True, False),
    (0, 3, 2, 2, 1, 512, 3, 0, True, True),
])
def test_linear_attention_scan_bfloat16(monkeypatch, eager, rule, geometry):
    batch, time, qh, h, kh, dk, dv, window, scalar, shared = geometry
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    state_shape = [batch, h, dk, dv]
    if window:
        state_shape.insert(0, window)
    model = bf16_boundaries(
        "LinearAttention",
        [
            ("query", [batch, time, qh * dk]), ("key", [batch, time, kh * dk]),
            ("value", [batch, time, h * dv]), ("past_state", state_shape),
            ("decay", [batch, time, h if scalar else h * dk]),
            ("beta", [batch, time, 1 if shared else h]),
        ],
        [("y", [batch, time, max(qh, h) * dv]), ("state", state_shape)],
        domain="com.microsoft",
        attrs=[
            ir.AttrInt64("q_num_heads", qh), ir.AttrInt64("kv_num_heads", h),
            ir.AttrInt64("state_window", window), ir.AttrString("update_rule", rule),
        ],
    )
    rng = np.random.default_rng(2431)
    feeds = {
        "query": rng.normal(size=(batch, time, qh * dk)).astype(np.float32) / 8,
        "key": rng.normal(size=(batch, time, kh * dk)).astype(np.float32) / 16,
        "value": rng.normal(size=(batch, time, h * dv)).astype(np.float32) / 8,
        "past_state": rng.normal(size=state_shape).astype(np.float32) / 16,
        "decay": np.full((batch, time, h if scalar else h * dk), -0.1, np.float32),
        "beta": np.full((batch, time, 1 if shared else h), 0.5, np.float32),
    }
    original = {name: value.copy() for name, value in feeds.items()}
    actual = strict_session(model).run(None, feeds)
    target = (
        fp32_working_reference(
            feeds, rule, qh, h, kh, dk, dv, window,
            input_dtype=ml_dtypes.bfloat16, state_dtype=ml_dtypes.bfloat16,
        )
        if 257 <= dk <= 512 else
        typed_reference(feeds, rule, qh, h, kh, dk, dv, window, state_dtype=ml_dtypes.bfloat16)
    )
    for output, reference in zip(actual, target, strict=True):
        assert output.shape == reference.shape
        assert output.dtype == np.float32
        assert np.isfinite(output).all() and np.isfinite(reference).all()
        np.testing.assert_allclose(output, reference, rtol=1e-2, atol=1e-3)
    with monkeypatch.context() as unfused:
        unfused.setenv(NO_SCAN, "1")
        expected = strict_session(model).run(None, feeds)
    for output, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(output, reference, rtol=1e-2, atol=1e-3)
    for output, reference in zip(expected, target, strict=True):
        np.testing.assert_allclose(output, reference, rtol=1e-2, atol=1e-3)
    if time == 0:
        np.testing.assert_array_equal(actual[1], target[1])
    for name, value in feeds.items():
        np.testing.assert_array_equal(value, original[name])


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("dk", [257, 512])
@pytest.mark.parametrize("dtype,state_dtype,domain", [
    pytest.param(DT.BFLOAT16, DT.BFLOAT16, "com.microsoft", id="bf16-bf16-contrib"),
    pytest.param(DT.BFLOAT16, DT.BFLOAT16, "", id="bf16-bf16-standard"),
    pytest.param(DT.BFLOAT16, DT.FLOAT16, "", id="bf16-fp16-standard"),
    pytest.param(DT.BFLOAT16, DT.FLOAT, "", id="bf16-fp32-standard"),
    pytest.param(DT.FLOAT16, DT.BFLOAT16, "", id="fp16-bf16-standard"),
    pytest.param(DT.FLOAT16, DT.FLOAT16, "", id="fp16-fp16-standard"),
    pytest.param(DT.FLOAT16, DT.FLOAT, "", id="fp16-fp32-standard"),
    pytest.param(DT.FLOAT, DT.BFLOAT16, "", id="fp32-bf16-standard"),
    pytest.param(DT.FLOAT, DT.FLOAT16, "", id="fp32-fp16-standard"),
    pytest.param(DT.FLOAT, DT.FLOAT, "", id="fp32-fp32-standard"),
    pytest.param(DT.FLOAT, DT.FLOAT, "com.microsoft", id="fp32-fp32-control"),
    pytest.param(DT.FLOAT16, DT.FLOAT16, "com.microsoft", id="fp16-fp16-control"),
])
def test_linear_attention_wide_internal_dtype_route(tmp_path, eager, dk, dtype, state_dtype, domain):
    trace = tmp_path / "linear_attention_dtype.json"
    child = r"""
import os
import sys
import ml_dtypes
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import _models as m
from test_linear_attention_scan import fp32_working_reference

ort.register_execution_provider_library("MLXExecutionProvider", sys.argv[1])
dk = int(sys.argv[2])
dtype, state_dtype = ir.DataType(int(sys.argv[3])), ir.DataType(int(sys.argv[4]))
domain = sys.argv[5]
shapes = {
    "query": [1, 7, dk], "key": [1, 7, dk], "value": [1, 7, 3],
    "past_state": [1, 1, dk, 3], "decay": [1, 7, 1], "beta": [1, 7, 1],
}
inputs, operands, nodes = [], [], []
for name, shape in shapes.items():
    external = m.tensor(name, ir.DataType.FLOAT, shape)
    internal_dtype = state_dtype if name == "past_state" else dtype
    internal = m.tensor(name + "_internal", internal_dtype, shape)
    inputs.append(external)
    operands.append(internal)
    nodes.append(ir.node("Cast", [external], attributes={"to": int(internal_dtype)}, outputs=[internal]))
internal_outputs = [
    m.tensor("y_internal", dtype, [1, 7, 3]),
    m.tensor("state_internal", state_dtype, [1, 1, dk, 3]),
]
nodes.append(ir.node(
    "LinearAttention", operands, domain=domain,
    attributes={"q_num_heads": 1, "kv_num_heads": 1, "update_rule": "gated_delta"},
    outputs=internal_outputs,
))
outputs = []
for internal, name, shape in zip(internal_outputs, ["y", "state"], [[1, 7, 3], [1, 1, dk, 3]], strict=True):
    external = m.tensor(name, ir.DataType.FLOAT, shape)
    nodes.append(ir.node("Cast", [internal], attributes={"to": int(ir.DataType.FLOAT)}, outputs=[external]))
    outputs.append(external)
imports = {"": 27 if not domain else 24, **({domain: 1} if domain else {})}
graph = ir.Graph(inputs, outputs, nodes=nodes, opset_imports=imports, name="wide_internal_dtype")
proto = ir.to_proto(ir.Model(graph, ir_version=11))
op = next(node for node in proto.graph.node if node.op_type == "LinearAttention")
casts = {node.output[0]: node.attribute[0].i for node in proto.graph.node if node.op_type == "Cast"}
assert [casts[name] for name in op.input] == [int(dtype)] * 3 + [int(state_dtype)] + [int(dtype)] * 2
assert all(value.type.tensor_type.elem_type == int(ir.DataType.FLOAT) for value in proto.graph.input)
model = proto.SerializeToString()
rng = np.random.default_rng(2431)
feeds = {
    "query": rng.normal(size=(1, 7, dk)).astype(np.float32) / 8,
    "key": rng.normal(size=(1, 7, dk)).astype(np.float32) / 16,
    "value": rng.normal(size=(1, 7, 3)).astype(np.float32) / 8,
    "past_state": rng.normal(size=(1, 1, dk, 3)).astype(np.float32) / 16,
    "decay": np.full((1, 7, 1), -0.1, np.float32),
    "beta": np.full((1, 7, 1), 0.5, np.float32),
}
original = {name: value.copy() for name, value in feeds.items()}
results = []
for disabled in ["0", "1"]:
    os.environ["ONNXRUNTIME_EP_MLX_NO_LINEAR_ATTENTION_SCAN"] = disabled
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model, options, providers=["MLXExecutionProvider"], enable_fallback=False,
    )
    session.disable_fallback()
    results.append(session.run(None, feeds))
    for name, value in feeds.items():
        np.testing.assert_array_equal(value, original[name])
rtol = 1e-2 if ir.DataType.BFLOAT16 in (dtype, state_dtype) else 1e-3
for actual, expected, shape in zip(results[0], results[1], [[1, 7, 3], [1, 1, dk, 3]], strict=True):
    assert actual.shape == expected.shape == tuple(shape)
    assert actual.dtype == expected.dtype == np.float32
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=1e-3)
if ir.DataType.BFLOAT16 in (dtype, state_dtype):
    types = {
        ir.DataType.FLOAT: np.float32, ir.DataType.FLOAT16: np.float16,
        ir.DataType.BFLOAT16: ml_dtypes.bfloat16,
    }
    target = fp32_working_reference(
        feeds, "gated_delta", 1, 1, 1, dk, 3,
        input_dtype=types[dtype], state_dtype=types[state_dtype],
    )
    for outputs in results:
        for actual, expected in zip(outputs, target, strict=True):
            assert actual.shape == expected.shape
            assert np.isfinite(expected).all()
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=1e-3)
"""
    env = {
        **os.environ,
        "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
        "ONNXRUNTIME_EP_MLX_NO_COMPILE": "1" if eager else "0",
        "ALLOW_RELEASED_ONNX_OPSET_ONLY": "0",
    }
    subprocess.run(
        [
            sys.executable, "-c", child, os.environ["ONNXRUNTIME_MLX_EP_LIB"],
            str(dk), str(int(dtype)), str(int(state_dtype)), domain,
        ],
        cwd=Path(__file__).parent, env=env, check=True, capture_output=True, text=True,
        timeout=120,
    )
    events = json.loads(trace.read_text())
    fast = [
        event for event in events
        if event.get("cat") == "op.fast"
        and event.get("args", {}).get("kernel") == "mlx_linear_attention_scan"
    ]
    bf16_reason = [
        event for event in events
        if event.get("cat") == "op.composed"
        and event.get("args", {}).get("reason")
        == "stable unroll: wide BF16 preserves state-dtype primitives"
    ]
    fp32_reason = [
        event for event in events
        if event.get("cat") == "op.composed"
        and event.get("args", {}).get("reason") == "stable unroll: wide BF16 FP32 working state"
    ]
    # Both dtype families use the FP32 wide scan; the disabled control still composes.
    assert fast
    assert not bf16_reason
    if DT.BFLOAT16 in (dtype, state_dtype):
        assert fp32_reason


@pytest.mark.parametrize("dk,disabled", [
    (33, False), (33, True), (257, False), (512, False), (513, False),
])
def test_linear_attention_scan_route_and_dynamic_replay(tmp_path, dk, disabled):
    trace = tmp_path / "linear_attention.json"
    child = r"""
import sys
import numpy as np
import onnx
import onnxruntime as ort
from test_ssm_contracts import linear_reference, strict_session
from test_ssm_quant import _linear_attention_model
ort.register_execution_provider_library("MLXExecutionProvider", sys.argv[1])
dk = int(sys.argv[2])
model, _ = _linear_attention_model(
    "gated_delta", B=1, T=5, q_heads=2, kv_heads=2, d_k=dk, d_v=3,
    with_past=True, scalar_decay=True, dyn_time=True,
)
model = onnx.load_model_from_string(model)
for value in [*model.graph.input, *model.graph.output]:
    value.type.tensor_type.shape.dim[0].dim_param = "batch"
session = strict_session(model.SerializeToString())
rng = np.random.default_rng(2432)
for batch, time in [(1, 5), (1, 5), (2, 7), (2, 1), (1, 0), (1, 5)]:
    feeds = {
        "query": (rng.normal(size=(batch, time, 2 * dk)) / 8).astype(np.float32),
        "key": (rng.normal(size=(batch, time, 2 * dk)) / 16).astype(np.float32),
        "value": (rng.normal(size=(batch, time, 6)) / 8).astype(np.float32),
        "past_state": (rng.normal(size=(batch, 2, dk, 3)) / 16).astype(np.float32),
        "decay": -rng.uniform(0, .1, (batch, time, 2)).astype(np.float32),
        "beta": rng.uniform(.25, .75, (batch, time, 2)).astype(np.float32),
    }
    for actual, expected in zip(session.run(None, feeds), linear_reference(feeds), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
"""
    env = {**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)}
    env[NO_SCAN] = "1" if disabled else "0"
    subprocess.run(
        [sys.executable, "-c", child, os.environ["ONNXRUNTIME_MLX_EP_LIB"], str(dk)],
        cwd=Path(__file__).parent, env=env, check=True, capture_output=True, text=True,
        timeout=120,
    )
    events = json.loads(trace.read_text())
    fast = [
        event for event in events
        if event.get("cat") == "op.fast"
        and event.get("args", {}).get("kernel") == "mlx_linear_attention_scan"
    ]
    composed = [
        event for event in events
        if event.get("cat") == "op.composed" and event.get("name") == "LinearAttention"
    ]
    if disabled or dk > 512:
        assert not fast
        assert composed
    else:
        assert fast
        assert not composed


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("dk", [257, 512])
def test_linear_attention_wide_bfloat16_dynamic_replay(monkeypatch, eager, disabled, dk):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    monkeypatch.setenv(NO_SCAN, "1" if disabled else "0")
    model = bf16_boundaries(
        "LinearAttention",
        [
            ("query", ["batch", "time", 2 * dk]), ("key", ["batch", "time", 2 * dk]),
            ("value", ["batch", "time", 6]), ("past_state", ["batch", 2, dk, 3]),
            ("decay", ["batch", "time", 2]), ("beta", ["batch", "time", 2]),
        ],
        [("y", ["batch", "time", 6]), ("state", ["batch", 2, dk, 3])],
        domain="com.microsoft",
        attrs=[ir.AttrInt64("q_num_heads", 2), ir.AttrInt64("kv_num_heads", 2)],
    )
    session = strict_session(model)
    rng = np.random.default_rng(2435)
    previous = None
    for batch, time, decay in [
        (1, 5, 0.), (1, 5, -1e-7), (2, 7, -.1), (2, 1, 0.),
        (1, 0, 0.), (0, 3, -.1), (1, 5, -.1),
    ]:
        past = (
            previous.copy() if previous is not None and previous.shape[0] == batch
            else rng.normal(size=(batch, 2, dk, 3)).astype(np.float32) / 16
        )
        feeds = {
            "query": rng.normal(size=(batch, time, 2 * dk)).astype(np.float32) / 8,
            "key": rng.normal(size=(batch, time, 2 * dk)).astype(np.float32) / 16,
            "value": rng.normal(size=(batch, time, 6)).astype(np.float32) / 8,
            "past_state": past,
            "decay": np.full((batch, time, 2), decay, np.float32),
            "beta": rng.uniform(.25, .75, (batch, time, 2)).astype(np.float32),
        }
        original = {name: value.copy() for name, value in feeds.items()}
        actual = session.run(None, feeds)
        target = fp32_working_reference(
            feeds, "gated_delta", 2, 2, 2, dk, 3,
            input_dtype=ml_dtypes.bfloat16, state_dtype=ml_dtypes.bfloat16,
        )
        for output, reference in zip(actual, target, strict=True):
            assert output.shape == reference.shape
            assert output.dtype == np.float32
            assert np.isfinite(output).all() and np.isfinite(reference).all()
            np.testing.assert_allclose(output, reference, rtol=1e-2, atol=1e-3)
        for name, value in feeds.items():
            np.testing.assert_array_equal(value, original[name])
        if time == 0:
            np.testing.assert_array_equal(actual[1], target[1])
        previous = actual[1]


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("dk", [257, 512])
@pytest.mark.parametrize("batch,time", [(0, 3), (1, 0), (0, 0)])
def test_linear_attention_wide_empty(monkeypatch, eager, dk, batch, time):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    model, feeds = _linear_attention_model(
        "gated_delta", B=batch, T=time, q_heads=2, kv_heads=2, d_k=dk, d_v=3,
        with_past=True, scalar_decay=True,
    )
    for actual, expected in zip(strict_session(model).run(None, feeds), linear_reference(feeds), strict=True):
        assert actual.shape == expected.shape
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("dk", [257, 512])
@pytest.mark.parametrize("time", [128, 200])
def test_linear_attention_wide_zero_weak_decay(monkeypatch, eager, dk, time):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    model, feeds = _linear_attention_model(
        "gated_delta", B=1, T=time, q_heads=2, kv_heads=2, d_k=dk, d_v=4,
        with_past=True, scalar_decay=True,
    )
    rng = np.random.default_rng(2434)
    for name, array in feeds.items():
        feeds[name] = (rng.normal(size=array.shape) * .1).astype(array.dtype)
    feeds["beta"].fill(.95)
    session = strict_session(model)
    for decay in [0., -1e-7, -.1, 0.]:
        feeds["decay"].fill(decay)
        expected = linear_reference(feeds)
        for actual, reference in zip(session.run(None, feeds), expected, strict=True):
            assert actual.shape == reference.shape
            assert actual.dtype == np.float32
            assert np.isfinite(actual).all() and np.isfinite(reference).all()
            np.testing.assert_allclose(actual, reference, rtol=2e-3, atol=2e-5)


@pytest.mark.parametrize("float64", [False, True])
def test_linear_attention_scan_control_flow_claims(float64):
    query_input = m.tensor("query", DT.DOUBLE if float64 else DT.FLOAT, [1, 5, 3])
    key = m.tensor("key", DT.FLOAT, [1, 5, 3])
    value = m.tensor("value", DT.FLOAT, [1, 5, 2])
    past = m.tensor("past_state", DT.FLOAT, [1, 1, 3, 2])
    decay = m.tensor("decay", DT.FLOAT, [1, 5, 1])
    beta = m.tensor("beta", DT.FLOAT, [1, 5, 1])

    def branch(suffix):
        query = m.tensor("q_" + suffix, DT.FLOAT, [1, 5, 3])
        output = m.tensor("y_" + suffix, DT.FLOAT, [1, 5, 2])
        state = m.tensor("s_" + suffix, DT.FLOAT, [1, 1, 3, 2])
        copied = m.tensor("copy_" + suffix, DT.DOUBLE if float64 else DT.FLOAT, [1, 5, 3])
        nodes = []
        if float64:
            nodes.append(
                ir.node("Cast", [query_input], attributes={"to": int(DT.FLOAT)}, outputs=[query])
            )
        else:
            query = query_input
        nodes += [
            ir.node(
                "LinearAttention", [query, key, value, past, decay, beta],
                attributes={"q_num_heads": 1, "kv_num_heads": 1},
                outputs=[output, state], domain="com.microsoft",
            ),
            ir.node("Identity", [query_input], outputs=[copied]),
        ]
        return ir.Graph([], [output, state, copied], nodes=nodes, name="branch_" + suffix)

    condition = m.tensor("condition", DT.BOOL, [])
    outputs = [
        m.tensor("y", DT.FLOAT, [1, 5, 2]),
        m.tensor("state", DT.FLOAT, [1, 1, 3, 2]),
        m.tensor("copy", DT.DOUBLE if float64 else DT.FLOAT, [1, 5, 3]),
    ]
    node = ir.node(
        "If", [condition], outputs=outputs,
        attributes={"then_branch": branch("then"), "else_branch": branch("else")},
    )
    graph = ir.Graph(
        [condition, query_input, key, value, past, decay, beta], outputs, nodes=[node],
        opset_imports={"": 24, "com.microsoft": 1}, name="control_flow_linear_attention",
    )
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    rng = np.random.default_rng(2433)
    feeds = {
        "condition": np.array(True),
        "query": rng.normal(size=(1, 5, 3)).astype(np.float64 if float64 else np.float32) / 8,
        "key": rng.normal(size=(1, 5, 3)).astype(np.float32) / 8,
        "value": rng.normal(size=(1, 5, 2)).astype(np.float32) / 8,
        "past_state": rng.normal(size=(1, 1, 3, 2)).astype(np.float32) / 8,
        "decay": np.zeros((1, 5, 1), np.float32),
        "beta": np.full((1, 5, 1), 0.5, np.float32),
    }
    if float64:
        # Control-flow bodies carrying float64 are deliberately declined, even with float32 math.
        with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail, match="fallback to CPU EP"):
            strict_session(model)
        return
    expected = linear_reference(feeds)
    actual = strict_session(model).run(None, feeds)
    for output, reference in zip(actual[:2], expected, strict=True):
        np.testing.assert_allclose(output, reference, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(actual[2].view(np.uint32), feeds["query"].view(np.uint32))
