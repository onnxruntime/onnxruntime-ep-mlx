"""Frozen precision/layout caches preserve initializer and live-input behavior."""

import ml_dtypes
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

import _models as m


def _model(weight, *, live, bf16):
    x = m.tensor("x", ir.DataType.FLOAT, ["rows", weight.shape[1]])
    w = m.tensor("weight", ir.DataType.FLOAT, weight.shape)
    inputs = [x, w] if live else [x]
    initializers = []
    if not live:
        w.const_value = ir.tensor(weight)
        initializers = [w]
    t = ir.tape.Tape()
    value = w
    if bf16:
        value = t.op(
            "Cast",
            [t.op("Cast", [value], {"to": int(ir.DataType.BFLOAT16)})],
            {"to": int(ir.DataType.FLOAT)},
        )
    value = t.op("Transpose", [value], {"perm": [1, 0]})
    y = t.op("MatMul", [x, value])
    y.name = "y"
    y.dtype = ir.DataType.FLOAT
    y.shape = ir.Shape(["rows", weight.shape[0]])
    model = ir.Model(
        ir.Graph(
            inputs,
            [y],
            nodes=t.nodes,
            initializers=initializers,
            opset_imports={"": 25},
            name="frozen_view",
        ),
        ir_version=11,
    )
    return ir.to_proto(model).SerializeToString()


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
def test_frozen_and_live_rhs_replay(live, bf16, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    rng = np.random.default_rng(21)
    weight = rng.normal(0, 0.01, (128, 8192)).astype(np.float32)
    source = _model(weight, live=live, bf16=bf16)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    executor = ort.InferenceSession(source, options, providers=["MLXExecutionProvider"])
    for step, rows in enumerate([64, 65, 3, 64]):
        current = weight + np.float32(step * 0.002) if live else weight
        x = rng.normal(0, 0.1, (rows, 8192)).astype(np.float32)
        feeds = {"x": x, **({"weight": current} if live else {})}
        expected = current.astype(ml_dtypes.bfloat16).astype(np.float32) if bf16 else current
        actual = executor.run(None, feeds)[0]
        np.testing.assert_allclose(actual, x @ expected.T, rtol=1e-4, atol=1e-6)
