"""Narrow comparison policy for two rounding-sensitive released FP16 fixtures."""

from __future__ import annotations

import numpy as np


# Each fixture and its native result are within the original 1e-3 relative tolerance of
# float64 SDPA, but differ from each other by two FP16 ULPs. Keep the original models/data;
# the companion oracle tests additionally enforce the exact two-ULP and forward-error bounds.
ATTENTION_FP16_TEST_KWARGS = {
    name: {"rtol": float(2 * np.finfo(np.float16).eps)}
    for name in (
        "test_attention_4d_causal_fp16",
        "test_attention_4d_causal_fp16_expanded",
    )
}


def causal_attention_float64(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Independent real-arithmetic oracle for the no-cache, top-left causal fixtures."""
    q, k, v = (x.astype(np.float64) for x in (q, k, v))
    scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(q.shape[-1])
    allowed = np.arange(k.shape[-2]) <= np.arange(q.shape[-2])[:, None]
    scores = np.where(allowed, scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v


def assert_fp16_attention_accuracy(
    actual: np.ndarray,
    expected: np.ndarray,
    ideal: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Require finite, shape/type-correct outputs, two ULPs, and the original forward error."""
    assert actual.dtype == expected.dtype == np.float16
    assert actual.shape == expected.shape == ideal.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_array_max_ulp(actual, expected, maxulp=2)
    # Neither the fixture nor the implementation gets a relaxed bound against the oracle.
    np.testing.assert_allclose(expected.astype(np.float64), ideal, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual.astype(np.float64), ideal, rtol=rtol, atol=atol)
