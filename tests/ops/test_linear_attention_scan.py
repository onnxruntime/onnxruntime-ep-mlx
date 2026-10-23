"""Stable fused LinearAttention recurrence: arithmetic, carry, and actual routing."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

import _models as m
from test_ssm_contracts import bf16_boundaries, linear_reference, strict_session
from test_ssm_quant import _linear_attention_model

DT = ir.DataType
NO_SCAN = "ONNXRUNTIME_EP_MLX_NO_LINEAR_ATTENTION_SCAN"


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
def test_linear_attention_scan_bfloat16(monkeypatch, eager):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    model = bf16_boundaries(
        "LinearAttention",
        [
            ("query", [1, 7, 33]), ("key", [1, 7, 33]), ("value", [1, 7, 3]),
            ("past", [1, 1, 33, 3]), ("decay", [1, 7, 1]), ("beta", [1, 7, 1]),
        ],
        [("y", [1, 7, 3]), ("state", [1, 1, 33, 3])],
        domain="com.microsoft",
        attrs=[ir.AttrInt64("q_num_heads", 1), ir.AttrInt64("kv_num_heads", 1)],
    )
    rng = np.random.default_rng(2431)
    feeds = {
        "query": rng.normal(size=(1, 7, 33)).astype(np.float32) / 8,
        "key": rng.normal(size=(1, 7, 33)).astype(np.float32) / 16,
        "value": rng.normal(size=(1, 7, 3)).astype(np.float32) / 8,
        "past": rng.normal(size=(1, 1, 33, 3)).astype(np.float32) / 16,
        "decay": np.full((1, 7, 1), -0.1, np.float32),
        "beta": np.full((1, 7, 1), 0.5, np.float32),
    }
    actual = strict_session(model).run(None, feeds)
    with monkeypatch.context() as unfused:
        unfused.setenv(NO_SCAN, "1")
        expected = strict_session(model).run(None, feeds)
    for output, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(output, reference, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("dk,disabled", [(33, False), (33, True), (257, False)])
def test_linear_attention_scan_route_and_dynamic_replay(tmp_path, dk, disabled):
    trace = tmp_path / "linear_attention.json"
    child = r"""
import sys
import numpy as np
import onnxruntime as ort
from test_ssm_contracts import linear_reference, strict_session
from test_ssm_quant import _linear_attention_model
ort.register_execution_provider_library("MLXExecutionProvider", sys.argv[1])
dk = int(sys.argv[2])
model, _ = _linear_attention_model(
    "gated_delta", B=1, T=5, q_heads=2, kv_heads=2, d_k=dk, d_v=3,
    with_past=True, scalar_decay=True, dyn_time=True,
)
session = strict_session(model)
rng = np.random.default_rng(2432)
for time in [5, 5, 7, 1, 5]:
    feeds = {
        "query": (rng.normal(size=(1, time, 2 * dk)) / 8).astype(np.float32),
        "key": (rng.normal(size=(1, time, 2 * dk)) / 16).astype(np.float32),
        "value": (rng.normal(size=(1, time, 6)) / 8).astype(np.float32),
        "past_state": (rng.normal(size=(1, 2, dk, 3)) / 16).astype(np.float32),
        "decay": -rng.uniform(0, .1, (1, time, 2)).astype(np.float32),
        "beta": rng.uniform(.25, .75, (1, time, 2)).astype(np.float32),
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
    if disabled or dk > 256:
        assert not fast
        assert composed
    else:
        assert fast
        assert not composed


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
