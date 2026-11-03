"""ConvTranspose direct padding and output-crop boundaries against a scatter oracle."""

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest


def _bf16(values):
    bits = values.astype(np.float32).view(np.uint32)
    return (
        (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    ).view(np.float32)


def _reference(x, w, attrs):
    strides, dilations, padding, extra = (
        attrs[k] for k in ("strides", "dilations", "pads", "output_padding")
    )
    batch, channels, height, width = x.shape
    full = [
        (n - 1) * s + 2 * d + 1 + e
        for n, s, d, e in zip((height, width), strides, dilations, extra, strict=True)
    ]
    output = np.zeros((batch, 3, *full), np.float32)
    for c in range(channels):
        for h in range(height):
            for j in range(width):
                for kh in range(3):
                    for kw in range(3):
                        output[
                            :,
                            :,
                            h * strides[0] + kh * dilations[0],
                            j * strides[1] + kw * dilations[1],
                        ] += x[:, c, h, j, None] * w[c, :, kh, kw]
    return output[
        :, :, padding[0] : full[0] - padding[2], padding[1] : full[1] - padding[3]
    ]


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize(
    "strides,dilations,pads,extra",
    [
        ([1, 1], [1, 1], [1, 1, 1, 1], [0, 0]),
        ([2, 1], [1, 1], [0, 1, 2, 0], [1, 0]),
        ([2, 2], [1, 1], [3, 0, 0, 1], [1, 0]),
        ([1, 2], [2, 1], [2, 1, 1, 0], [0, 1]),
    ],
)
def test_dynamic_padding_geometry(
    dtype, eager, strides, dilations, pads, extra, monkeypatch
):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    boundary = ir.DataType.FLOAT if dtype == ir.DataType.BFLOAT16 else dtype
    x = ir.val("x", dtype=boundary, shape=["batch", 2, "height", "width"])
    w = ir.val("w", dtype=boundary, shape=[2, 3, 3, 3])
    tape = ir.tape.Tape()
    inputs = (
        [tape.op("Cast", [v], {"to": int(dtype)}) for v in (x, w)]
        if dtype == ir.DataType.BFLOAT16
        else [x, w]
    )
    attrs = {
        "strides": strides,
        "dilations": dilations,
        "pads": pads,
        "output_padding": extra,
    }
    y = tape.op("ConvTranspose", inputs, attrs)
    if dtype == ir.DataType.BFLOAT16:
        y = tape.op("Cast", [y], {"to": int(boundary)})
    y.name, y.dtype, y.shape = (
        "y",
        boundary,
        ir.Shape(["batch", 3, "out_height", "out_width"]),
    )
    model = ir.Model(
        ir.Graph(
            [x, w],
            [y],
            nodes=tape.nodes,
            opset_imports={"": 24},
            name="transpose_padding",
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
    rng = np.random.default_rng(23)
    dt = np.float16 if dtype == ir.DataType.FLOAT16 else np.float32
    for batch, height, width in [(1, 4, 5), (2, 5, 4), (0, 4, 5), (1, 4, 5)]:
        data = rng.normal(0, 0.1, (batch, 2, height, width)).astype(dt)
        weights = rng.normal(0, 0.1, (2, 3, 3, 3)).astype(dt)
        a, b = (
            map(_bf16, (data, weights))
            if dtype == ir.DataType.BFLOAT16
            else (data.astype(np.float32), weights.astype(np.float32))
        )
        expected = _reference(a, b, attrs)
        expected = (
            _bf16(expected) if dtype == ir.DataType.BFLOAT16 else expected.astype(dt)
        )
        tolerance = (
            {"rtol": 1e-4, "atol": 1e-6}
            if dtype == ir.DataType.FLOAT
            else (
                {"rtol": 3e-3, "atol": 1e-4}
                if dtype == ir.DataType.FLOAT16
                else {"rtol": 2e-2, "atol": 1e-3}
            )
        )
        np.testing.assert_allclose(
            executor.run(None, {"x": data, "w": weights})[0], expected, **tolerance
        )
