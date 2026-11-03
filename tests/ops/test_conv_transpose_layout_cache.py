"""Frozen ConvTranspose layouts and changing explicit weights."""

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest


def _reference(x, w, group):
    batch, channels, height, width = x.shape
    out_channels = w.shape[1] * group
    out = np.zeros((batch, out_channels, height + 2, width + 2), np.float32)
    for c in range(channels):
        first = (c // (channels // group)) * w.shape[1]
        for i in range(height):
            for j in range(width):
                out[:, first : first + w.shape[1], i : i + 3, j : j + 3] += (
                    x[:, c, i, j, None, None, None] * w[c]
                )
    return out[:, :, 1:-1, 1:-1]


@pytest.mark.parametrize("mode", ["frozen", "live", "overridable"])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
def test_weight_layout_preserves_live_updates(mode, eager, group, dtype, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    frozen = mode == "frozen"
    initialized = mode != "live"
    rng = np.random.default_rng(8)
    dt = np.float16 if dtype == ir.DataType.FLOAT16 else np.float32
    weights = rng.normal(0, 0.1, (4, 3, 3, 3)).astype(dt)
    x = ir.val("x", dtype=dtype, shape=["batch", 4, 5, 6])
    w = ir.val(
        "w",
        dtype=dtype,
        shape=weights.shape,
        const_value=ir.tensor(weights) if initialized else None,
    )
    tape = ir.tape.Tape()
    view = tape.op("Identity", [w])
    y = tape.op("ConvTranspose", [x, view], {"pads": [1, 1, 1, 1], "group": group})
    y.name, y.dtype, y.shape = "y", dtype, ir.Shape(["batch", 3 * group, 5, 6])
    model = ir.Model(
        ir.Graph(
            [x] if frozen else [x, w],
            [y],
            nodes=tape.nodes,
            initializers=[w] if initialized else [],
            opset_imports={"": 24},
            name="conv_layout",
        ),
        ir_version=11,
    )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    executor = ort.InferenceSession(
        ir.to_proto(model).SerializeToString(),
        options,
        providers=["MLXExecutionProvider"],
    )
    executor.disable_fallback()
    for batch in (1, 2, 1):
        data = rng.normal(0, 0.1, (batch, 4, 5, 6)).astype(dt)
        for scale in (1, 2, 1):
            current = weights if frozen else (weights * scale).astype(dt)
            feeds = {
                "x": data,
                **(
                    {}
                    if frozen or (mode == "overridable" and scale == 1)
                    else {"w": current}
                ),
            }
            expected = _reference(
                data.astype(np.float32), current.astype(np.float32), group
            ).astype(dt)
            np.testing.assert_allclose(
                executor.run(None, feeds)[0],
                expected,
                rtol=3e-3 if dt == np.float16 else 1e-4,
                atol=1e-4 if dt == np.float16 else 1e-6,
            )
