"""State-only and replay Scan fusion against an independent NumPy recurrence."""

from __future__ import annotations

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from test_gated_delta_reverse import run


def model_and_feeds(
    *,
    width=32,
    value_width=9,
    history=True,
    per_key=False,
    input_direction=0,
    output_direction=0,
    changed=False,
    extra_random=False,
):
    batch, heads, time = 2, 2, 7
    shapes = [
        [batch, heads, width, value_width],
        [batch, heads, width],
        [batch, heads, value_width],
        [batch, heads, width if per_key else 1],
        [batch, heads],
    ]
    inputs = [
        ir.val(f"input_{i}", dtype=ir.DataType.FLOAT, shape=s if i == 0 else [time, *s])
        for i, s in enumerate(shapes)
    ]
    steps = [
        ir.val(f"step_{i}", dtype=ir.DataType.FLOAT, shape=s)
        for i, s in enumerate(shapes)
    ]
    constants, nodes = [], []

    def const(axes):
        value = ir.val(
            f"axes_{len(constants)}", const_value=ir.tensor(np.array(axes, np.int64))
        )
        constants.append(value)
        return value

    last, row, both = [const(x) for x in ([-1], [-2], [-1, -2])]

    def op(name, *args):
        value = ir.val(f"value_{len(nodes)}", dtype=ir.DataType.FLOAT)
        nodes.append(ir.node(name, args, outputs=[value]))
        return value

    state, key, value, decay, beta = steps
    faded = op("Mul", state, op("Exp", op("Unsqueeze", decay, last)))
    product = op("Squeeze", op("MatMul", op("Unsqueeze", key, row), faded), row)
    residual = op("Add" if changed else "Sub", value, product)
    update = op(
        "Mul",
        op("Unsqueeze", beta, both),
        op("Mul", op("Unsqueeze", key, last), op("Unsqueeze", residual, row)),
    )
    present = op("Add", faded, update)
    present.shape = ir.Shape(shapes[0])
    body_outputs = [present]
    if history:
        previous = op("Identity", state)
        previous.shape = ir.Shape(shapes[0])
        body_outputs.append(previous)
    if extra_random:
        nodes.append(
            ir.node(
                "RandomNormal",
                [],
                attributes={"shape": [1]},
                outputs=[ir.val("unused_random", dtype=ir.DataType.FLOAT, shape=[1])],
            )
        )
    body = ir.Graph(
        steps,
        body_outputs,
        nodes=nodes,
        initializers=constants,
        opset_imports={"": 18},
        name="transition",
    )
    outputs = [ir.val("final_state", dtype=ir.DataType.FLOAT, shape=shapes[0])]
    if history:
        outputs.append(
            ir.val("previous_states", dtype=ir.DataType.FLOAT, shape=[time, *shapes[0]])
        )
    scan = ir.node(
        "Scan",
        inputs,
        attributes={
            "body": body,
            "num_scan_inputs": 4,
            "scan_input_directions": [input_direction] * 4,
            "scan_output_directions": [output_direction] if history else [],
        },
        outputs=outputs,
    )
    graph = ir.Graph(
        inputs, outputs, nodes=[scan], opset_imports={"": 18}, name="state_scan"
    )
    rng = np.random.default_rng(291)
    feeds = {
        x.name: rng.normal(0, 0.1, tuple(x.shape)).astype(np.float32) for x in inputs
    }
    feeds["input_1"] /= np.linalg.norm(feeds["input_1"], axis=-1, keepdims=True)
    feeds["input_3"] -= 0.5
    feeds["input_4"] = rng.uniform(0.1, 0.8, tuple(inputs[4].shape)).astype(np.float32)
    current = feeds["input_0"].copy()
    previous = []
    order = range(time - 1, -1, -1) if input_direction else range(time)
    for t in order:
        previous.append(current.copy())
        key, value, decay, beta = [feeds[f"input_{i}"][t] for i in range(1, 5)]
        faded = current * np.exp(decay[..., None])
        product = (key[..., None, :] @ faded)[..., 0, :]
        residual = value + product if changed else value - product
        current = faded + beta[..., None, None] * (
            key[..., :, None] * residual[..., None, :]
        )
    expected = [current]
    if history:
        expected.append(np.stack(previous[::-1] if output_direction else previous))
    model = ir.to_proto(ir.Model(graph, ir_version=10))
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString(), feeds, expected


@pytest.mark.parametrize("width,value_width", [(16, 1), (32, 9), (64, 33), (128, 128)])
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("per_key", [False, True])
def test_state_products(width, value_width, history, per_key):
    model, feeds, expected = model_and_feeds(
        width=width, value_width=value_width, history=history, per_key=per_key
    )
    for actual, want in zip(run(model, feeds), expected, strict=True):
        np.testing.assert_allclose(actual, want, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(
    "input_direction,output_direction", [(0, 0), (0, 1), (1, 0), (1, 1)]
)
def test_state_directions(input_direction, output_direction):
    model, feeds, expected = model_and_feeds(
        input_direction=input_direction, output_direction=output_direction
    )
    for actual, want in zip(run(model, feeds), expected, strict=True):
        np.testing.assert_allclose(actual, want, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("options", [{"changed": True}, {"width": 17}])
def test_unmatched_state_preserves_semantics(options):
    model, feeds, expected = model_and_feeds(**options)
    for actual, want in zip(run(model, feeds), expected, strict=True):
        np.testing.assert_allclose(actual, want, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(
    "options,kernel",
    [
        ({"history": False}, "mlx_gated_delta_state_scan"),
        ({"history": True}, "mlx_gated_delta_replay_scan"),
        ({"changed": True}, None),
        ({"width": 17}, None),
        ({"extra_random": True}, None),
    ],
)
def test_state_fusion_trace(tmp_path, options, kernel):
    import json
    import os
    import subprocess
    import sys

    trace = tmp_path / "trace.json"
    subprocess.run(
        [sys.executable, __file__, json.dumps(options)],
        check=True,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
        timeout=180,
    )
    kernels = {
        event.get("args", {}).get("kernel")
        for event in json.loads(trace.read_text())
        if event.get("cat") == "op.fast"
    }
    if kernel:
        assert kernel in kernels
    else:
        assert not kernels.intersection(
            {"mlx_gated_delta_state_scan", "mlx_gated_delta_replay_scan"}
        )


@pytest.mark.parametrize("compiled", [False, True])
def test_dynamic_time_and_changed_values_reuse_session(compiled):
    import onnxruntime as ort

    data, feeds, _ = model_and_feeds()
    model = ir.from_proto(onnx.load_model_from_string(data))
    for value in [*model.graph.inputs[1:], model.graph.outputs[1]]:
        value.shape = ir.Shape(["time", *list(value.shape)[1:]])
    data = ir.to_proto(model).SerializeToString()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("ep.mlx.enable_compile", "1" if compiled else "0")
    session = ort.InferenceSession(data, options, providers=["MLXExecutionProvider"])
    for time, scale in [(7, 1.0), (3, 0.7), (7, -0.2)]:
        changed = {
            name: np.ascontiguousarray(
                (value if name == "input_0" else value[:time]) * scale
            )
            for name, value in feeds.items()
        }
        for actual, want in zip(
            session.run(None, changed), run(data, changed, cpu=True), strict=True
        ):
            np.testing.assert_allclose(actual, want, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    import json
    import os
    import sys

    import onnxruntime as ort

    ort.register_execution_provider_library(
        "MLXExecutionProvider", os.environ["ONNXRUNTIME_MLX_EP_LIB"]
    )
    model, feeds, expected = model_and_feeds(**json.loads(sys.argv[1]))
    for actual, want in zip(run(model, feeds), expected, strict=True):
        np.testing.assert_allclose(actual, want, rtol=1e-4, atol=1e-6)
