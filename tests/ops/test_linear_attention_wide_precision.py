"""Retained wide W0 fixture with its unchanged FP64 mathematical error budget."""

import numpy as np
import onnx_ir as ir
import pytest

from test_ssm_contracts import linear_reference, strict_session
from test_ssm_quant import _linear_attention_model


@pytest.mark.parametrize("eager", [False, True])
def test_wide_fp16_working_precision(monkeypatch, eager):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    else:
        monkeypatch.delenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", raising=False)
    monkeypatch.delenv("ONNXRUNTIME_EP_MLX_NO_LINEAR_ATTENTION_SCAN", raising=False)
    model, feeds = _linear_attention_model(
        "gated_delta", B=1, T=200, q_heads=16, kv_heads=16,
        d_k=512, d_v=128, with_past=True, scalar_decay=True,
        dtype=ir.DataType.FLOAT16,
    )
    rng = np.random.default_rng(2422)
    for name in ["query", "key", "value"]:
        feeds[name] = (
            rng.normal(size=feeds[name].shape) * (0.5 if name == "key" else 0.3)
        ).astype(feeds[name].dtype)
    feeds["past_state"] = (
        rng.normal(size=feeds["past_state"].shape) * 0.1
    ).astype(feeds["past_state"].dtype)
    keys = feeds["key"].reshape(1, 200, 16, 512)
    keys[:] = keys / np.linalg.norm(keys.astype(np.float32), axis=-1, keepdims=True)
    feeds["beta"].fill(0.95)
    feeds["decay"].fill(0)
    original = {name: value.copy() for name, value in feeds.items()}
    expected = linear_reference(feeds)
    session = strict_session(model)
    for _ in range(3):
        for actual, reference in zip(session.run(None, feeds), expected, strict=True):
            assert actual.shape == reference.shape
            assert actual.dtype == np.float16
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, reference, rtol=0.003, atol=0.003)
        for name, value in feeds.items():
            np.testing.assert_array_equal(value, original[name])
