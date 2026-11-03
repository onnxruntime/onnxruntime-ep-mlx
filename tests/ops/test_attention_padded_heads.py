"""Short-head fused SDPA preserves scale, masks, and live sequence geometry."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest


def _model(dtype, mode, head_dim):
    dt = getattr(ir.DataType, dtype)
    inputs, nodes, attention_inputs = [], [], []
    for name, heads, length in [("q", 4, "Q"), ("k", 2, "K"), ("v", 2, "K")]:
        value = ir.Value(
            name=name,
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1, heads, length, head_dim]),
        )
        inputs.append(value)
        cast = ir.Value(name=name + ".cast", type=ir.TensorType(dt), shape=value.shape)
        nodes.append(
            ir.node("Cast", [value], outputs=[cast], attributes={"to": int(dt)})
        )
        attention_inputs.append(cast)
    if mode in ("bool", "additive"):
        mask = ir.Value(
            name="mask",
            type=ir.TensorType(
                ir.DataType.BOOL if mode == "bool" else ir.DataType.FLOAT
            ),
            shape=ir.Shape(["Q", "K"]),
        )
        inputs.append(mask)
        attention_inputs.append(mask)
    result = ir.Value(
        name="result", type=ir.TensorType(dt), shape=ir.Shape([1, 4, "Q", head_dim])
    )
    nodes.append(
        ir.node(
            "Attention",
            attention_inputs,
            outputs=[result],
            attributes={"scale": 0.37, "is_causal": int(mode == "causal")},
        )
    )
    output = ir.Value(
        name="output", type=ir.TensorType(ir.DataType.FLOAT), shape=result.shape
    )
    nodes.append(
        ir.node(
            "Cast",
            [result],
            outputs=[output],
            attributes={"to": int(ir.DataType.FLOAT)},
        )
    )
    graph = ir.Graph(
        inputs, [output], nodes=nodes, name="padded_head", opset_imports={"": 24}
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def _reference(q, k, v, mask):
    k, v = (np.repeat(x.astype(np.float64), 2, axis=1) for x in (k, v))
    outputs = []
    for start in range(0, q.shape[2], 64):
        scores = (
            q[:, :, start : start + 64].astype(np.float64) @ k.swapaxes(-1, -2)
        ) * 0.37
        if mask is not None:
            scores = np.where(mask[start : start + 64], scores, -np.inf)
        maximum = scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores - np.where(np.isfinite(maximum), maximum, 0))
        total = probs.sum(axis=-1, keepdims=True)
        outputs.append((probs / np.where(total == 0, 1, total)) @ v)
    return np.concatenate(outputs, axis=2)


@pytest.mark.parametrize("dtype", ["FLOAT", "FLOAT16", "BFLOAT16"])
@pytest.mark.parametrize("head_dim", [32, 40, 63])
@pytest.mark.parametrize("mode", ["none", "bool", "additive", "causal"])
def test_short_heads_dynamic_gqa(dtype, mode, head_dim):
    import ml_dtypes

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        _model(dtype, mode, head_dim), options, providers=["MLXExecutionProvider"]
    )
    rng = np.random.default_rng(7303)
    np_dtype = {
        "FLOAT": np.float32,
        "FLOAT16": np.float16,
        "BFLOAT16": ml_dtypes.bfloat16,
    }[dtype]
    for length in [2048, 17, 2048]:
        feeds = {
            name: (rng.normal(size=(1, heads, length, head_dim)) * 0.2).astype(
                np.float32
            )
            for name, heads in [("q", 4), ("k", 2), ("v", 2)]
        }
        mask = None
        if mode != "none":
            mask = np.arange(length)[None, :] <= np.arange(length)[:, None]
            if mode in ("bool", "additive"):
                mask[0] = False
                feeds["mask"] = (
                    mask
                    if mode == "bool"
                    else np.where(mask, 0, -np.inf).astype(np.float32)
                )
        expected = _reference(
            *(
                feeds[name].astype(np_dtype).astype(np.float32)
                for name in ["q", "k", "v"]
            ),
            mask,
        )
        actual = session.run(None, feeds)[0]
        atol = {"FLOAT": 1e-6, "FLOAT16": 3e-4, "BFLOAT16": 3e-3}[dtype]
        np.testing.assert_allclose(
            actual, expected, rtol=1e-4 if dtype == "FLOAT" else 0.02, atol=atol
        )
        if mode in ("bool", "additive"):
            np.testing.assert_array_equal(actual[:, :, 0], 0)
