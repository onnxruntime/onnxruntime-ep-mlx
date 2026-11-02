"""Gated-delta reverse Scan: portable IR, exact structural matching and Metal numerics."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest


def model_and_feeds(
    *,
    width=32,
    value_width=9,
    time=7,
    batch=2,
    heads=2,
    input_direction=1,
    output_direction=1,
    changed=False,
    per_key=False,
    scale_value=None,
    live_scale=False,
):
    dt = ir.DataType.FLOAT
    step_shapes = (
        [[batch, heads, width, value_width]] * 2
        + [[batch, heads, width]] * 2
        + [
            [batch, heads, value_width],
            [batch, heads, width if per_key else 1],
            [batch, heads],
            [batch, heads, value_width],
        ]
    )
    inputs = [
        ir.val(f"input_{i}", dtype=dt, shape=s if i == 0 else [time, *s])
        for i, s in enumerate(step_shapes)
    ]
    values = [ir.val(f"step_{i}", dtype=dt, shape=s) for i, s in enumerate(step_shapes)]
    constants = []
    nodes = []

    def const(array):
        value = ir.val(f"constant_{len(constants)}", const_value=ir.tensor(np.asarray(array)))
        constants.append(value)
        return value

    last, row, both = [const(np.array(x, np.int64)) for x in ([-1], [-2], [-1, -2])]
    scale = const(
        np.array(1 / np.sqrt(width) if scale_value is None else scale_value, np.float32)
    )

    def op(name, *args, **attrs):
        out = ir.val(f"value_{len(nodes)}", dtype=dt)
        nodes.append(ir.node(name, args, attributes=attrs, outputs=[out]))
        return out

    def outer(x, y):
        return op("Mul", op("Unsqueeze", x, last), op("Unsqueeze", y, row))

    def retrieve(x, y):
        return op("Squeeze", op("MatMul", op("Unsqueeze", y, row), x), row)

    def matvec(x, y):
        return op("Squeeze", op("MatMul", x, op("Unsqueeze", y, last)), last)

    c, previous, q, k, v, g, beta, gy = values
    factor = op("Exp", op("Unsqueeze", g, last))
    faded = op("Mul", previous, factor)
    residual = op("Sub", v, retrieve(faded, k))
    present = op("Add", faded, op("Mul", op("Unsqueeze", beta, both), outer(k, residual)))
    total = op("Add", c, outer(op("Mul", q, scale), gy))
    product = retrieve(total, k)
    dv = op("Mul", product, op("Unsqueeze", beta, last))
    db = op("ReduceSum", op("Mul", product, residual), last, keepdims=0)
    dq = op("Mul", matvec(present, gy), scale)
    dk = op(
        "Sub",
        op("Mul", matvec(total, residual), op("Unsqueeze", beta, last)),
        matvec(faded, dv),
    )
    dfaded = op("Add" if changed else "Sub", total, outer(k, dv))
    dg = op("ReduceSum", op("Mul", dfaded, faded), last, keepdims=0)
    if not per_key:
        dg = op("ReduceSum", dg, last, keepdims=1)
    dc = op("Mul", factor, dfaded)
    results = [dc, dq, dk, dv, dg, db]
    shapes = [step_shapes[i] for i in (0, 2, 3, 4, 5, 6)]
    for value, shape in zip(results, shapes, strict=True):
        value.shape = ir.Shape(shape)
    body = ir.Graph(
        values,
        results,
        nodes=nodes,
        initializers=constants[:-1] if live_scale else constants,
        opset_imports={"": 18},
        name="arbitrary_names",
    )
    outputs = [
        ir.val(f"out_{i}", dtype=dt, shape=s if i == 0 else [time, *s])
        for i, s in enumerate(shapes)
    ]
    scan = ir.node(
        "Scan",
        inputs,
        attributes={
            "body": body,
            "num_scan_inputs": 7,
            "scan_input_directions": [input_direction] * 7,
            "scan_output_directions": [output_direction] * 5,
        },
        outputs=outputs,
    )
    outer_inputs = inputs
    if live_scale:
        scale.type = ir.TensorType(dt)
        scale.shape = ir.Shape([])
        outer_inputs = [*inputs, scale]
    graph = ir.Graph(
        outer_inputs,
        outputs,
        nodes=[scan],
        initializers=[scale] if live_scale else [],
        opset_imports={"": 18},
        name="gated_delta_pullback",
    )
    rng = np.random.default_rng(821)
    feeds = {
        x.name: rng.normal(0, 0.1, size=tuple(x.shape)).astype(np.float32) for x in inputs
    }
    feeds["input_3"] /= np.linalg.norm(feeds["input_3"], axis=-1, keepdims=True)
    feeds["input_5"] -= 0.5
    feeds["input_6"] = rng.uniform(0.1, 0.8, size=tuple(inputs[6].shape)).astype(np.float32)
    if live_scale:
        feeds[scale.name] = np.array(0.35, np.float32)
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString(), feeds


def run(model, feeds, *, cpu=False):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if not cpu:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model, options, providers=["CPUExecutionProvider"] if cpu else ["MLXExecutionProvider"]
    )
    return session.run(None, feeds)


@pytest.mark.parametrize(
    "width,value_width,time", [(16, 5, 1), (32, 9, 7), (64, 64, 17), (128, 128, 32)]
)
def test_products_match_cpu(width, value_width, time):
    model, feeds = model_and_feeds(width=width, value_width=value_width, time=time)
    for got, want in zip(run(model, feeds), run(model, feeds, cpu=True), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("input_direction,output_direction", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_scan_directions(input_direction, output_direction):
    model, feeds = model_and_feeds(
        input_direction=input_direction, output_direction=output_direction
    )
    for got, want in zip(run(model, feeds), run(model, feeds, cpu=True), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)


def test_structurally_changed_body_uses_original_semantics():
    model, feeds = model_and_feeds(changed=True)
    for got, want in zip(run(model, feeds), run(model, feeds, cpu=True), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("disabled", [False, True])
def test_fast_path_trace_and_unrolled_control(tmp_path, disabled):
    trace = tmp_path / "trace.json"
    output = tmp_path / "outputs.npz"
    subprocess.run(
        [sys.executable, __file__, str(output)],
        check=True,
        env={
            **os.environ,
            "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
            "ONNXRUNTIME_EP_MLX_NO_GATED_DELTA_REVERSE": str(int(disabled)),
        },
        timeout=180,
    )
    events = json.loads(trace.read_text())
    fast = any(
        e.get("cat") == "op.fast"
        and e.get("args", {}).get("kernel") == "mlx_gated_delta_reverse_scan"
        for e in events
    )
    assert fast != disabled
    model, feeds = model_and_feeds()
    expected = run(model, feeds, cpu=True)
    with np.load(output) as saved:
        for i, want in enumerate(expected):
            np.testing.assert_allclose(saved[f"arr_{i}"], want, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("scale_value", [0.0, -0.25, 1.0])
def test_scale_and_pipeline_cache(scale_value):
    model, feeds = model_and_feeds(scale_value=scale_value)
    for got, want in zip(run(model, feeds), run(model, feeds, cpu=True), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("options", [{"width": 24}, {"per_key": True}])
def test_uncovered_shapes_preserve_generic_scan(options):
    model, feeds = model_and_feeds(**options)
    for got, want in zip(run(model, feeds), run(model, feeds, cpu=True), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(
    "options", [{"changed": True}, {"width": 24}, {"per_key": True}, {"live_scale": True}]
)
def test_fallback_trace(tmp_path, options):
    trace = tmp_path / "trace.json"
    output = tmp_path / "outputs.npz"
    subprocess.run(
        [sys.executable, __file__, str(output)],
        check=True,
        env={
            **os.environ,
            "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
            "GATED_DELTA_TEST_OPTIONS": json.dumps(options),
        },
        timeout=180,
    )
    events = json.loads(trace.read_text())
    assert not any(
        e.get("cat") == "op.fast"
        and e.get("args", {}).get("kernel") == "mlx_gated_delta_reverse_scan"
        for e in events
    )
    model, feeds = model_and_feeds(**options)
    with np.load(output) as saved:
        for i, expected in enumerate(run(model, feeds, cpu=True)):
            np.testing.assert_allclose(saved[f"arr_{i}"], expected, rtol=1e-4, atol=1e-6)


def test_zero_cotangents_produce_zero_products():
    model, feeds = model_and_feeds()
    feeds["input_0"].fill(0)
    feeds["input_7"].fill(0)
    for output in run(model, feeds):
        np.testing.assert_array_equal(output, np.zeros_like(output))


def test_large_negative_decay_matches_cpu():
    model, feeds = model_and_feeds()
    feeds["input_5"].fill(-120)
    for got, expected in zip(run(model, feeds), run(model, feeds, cpu=True), strict=True):
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("compile_disabled", ["0", "1"])
def test_live_scale_updates_across_runs(monkeypatch, compile_disabled):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", compile_disabled)
    model, feeds = model_and_feeds(live_scale=True)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"])
    for value in (0.35, -0.2):
        feeds["constant_3"] = np.array(value, np.float32)
        for got, expected in zip(
            session.run(None, feeds), run(model, feeds, cpu=True), strict=True
        ):
            np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    ort.register_execution_provider_library(
        "MLXExecutionProvider", os.environ["ONNXRUNTIME_MLX_EP_LIB"]
    )
    data, inputs = model_and_feeds(
        **json.loads(os.environ.get("GATED_DELTA_TEST_OPTIONS", "{}"))
    )
    np.savez(sys.argv[1], *run(data, inputs))
