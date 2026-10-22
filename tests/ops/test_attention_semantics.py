"""Attention masking/precision oracles independent of version-specific backend fixtures."""

from __future__ import annotations

import numpy as np
import pytest
from onnx_ir import DataType as DT

import _models as m
from test_attention_ext import _t, assert_mlx_claims, build_model


def _attention_reference(q, k, v, allowed):
    """Stable real-arithmetic SDPA, including zero probabilities for empty key sets."""
    q, k, v = (x.astype(np.float64) for x in (q, k, v))
    groups = q.shape[1] // k.shape[1]
    k = np.repeat(k, groups, axis=1)
    v = np.repeat(v, groups, axis=1)
    scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(q.shape[-1])
    scores = np.where(allowed, scores, -np.inf)
    row_max = scores.max(axis=-1, keepdims=True)
    scores -= np.where(np.isneginf(row_max), 0, row_max)
    probs = np.exp(scores)
    denominator = probs.sum(axis=-1, keepdims=True)
    probs /= np.where(denominator == 0, 1, denominator)
    return probs @ v


@pytest.mark.parametrize("opset", [23, 24])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("mask_type", ["bool", "float"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mask_rank", [0, 1, 2, 3, 4])
def test_attention_fully_masked_rows(opset, dtype, mask_type, causal, mask_rank):
    rng = np.random.default_rng(42)
    q = rng.random((1, 4, 3, 8)).astype(dtype)
    k = rng.random((1, 2, 3, 8)).astype(dtype)
    v = rng.random((1, 2, 3, 8)).astype(dtype)
    allowed = np.array([[False, False, False], [False, False, True], [True, True, True]])
    if mask_rank == 0:
        allowed = np.array(False)
    elif mask_rank == 1:
        allowed = allowed[0]
    elif mask_rank == 3:
        allowed = allowed[None]
    elif mask_rank == 4:
        allowed = allowed[None, None]
    mask = allowed if mask_type == "bool" else np.where(allowed, 0, -np.inf).astype(dtype)
    dt = DT.FLOAT16 if dtype == np.float16 else DT.FLOAT
    model = build_model(
        "Attention",
        [_t("Q", list(q.shape), dt), _t("K", list(k.shape), dt),
         _t("V", list(v.shape), dt), _t("M", list(mask.shape), DT.BOOL if mask_type == "bool" else dt)],
        [_t("Y", list(q.shape), dt)],
        attributes={"is_causal": int(causal)},
        opset=opset,
    )
    feeds = {"Q": q, "K": k, "V": v, "M": mask}
    if causal:
        allowed = allowed & np.tri(3, dtype=bool)
    expected = _attention_reference(q, k, v, allowed)
    assert_mlx_claims(model, feeds)
    actual = m.run_mlx(model, feeds)[0]
    np.testing.assert_allclose(actual, expected, rtol=0.001, atol=1e-7)
    empty = np.broadcast_to(~allowed.any(axis=-1, keepdims=True), actual.shape)
    np.testing.assert_array_equal(actual[empty], np.zeros_like(actual[empty]))


@pytest.mark.parametrize("special_form", [False, True])
@pytest.mark.parametrize("invalid_score", [np.nan, np.inf])
@pytest.mark.parametrize("invalid_input", ["Q", "V"])
def test_attention_mask_guard_is_not_nan_to_num(special_form, invalid_score, invalid_input):
    q = np.ones((1, 2, 3, 8), dtype=np.float32)
    k = np.ones_like(q)
    v = np.ones_like(q)
    if invalid_input == "Q":
        q[:, :, 0] = invalid_score
        q[:, :, 1] = np.nan
    else:
        v[:, :, 0] = invalid_score
    mask = np.array([[False, False, False], [True, True, True], [True, True, True]])
    attributes = {"softmax_precision": int(DT.FLOAT)} if special_form else {}
    model = build_model(
        "Attention",
        [_t("Q", list(q.shape)), _t("K", list(k.shape)), _t("V", list(v.shape)),
         _t("M", list(mask.shape), DT.BOOL)],
        [_t("Y", list(q.shape))],
        attributes=attributes,
    )
    feeds = {"Q": q, "K": k, "V": v, "M": mask}
    assert_mlx_claims(model, feeds)
    actual = m.run_mlx(model, feeds)[0]
    np.testing.assert_array_equal(actual[:, :, 0], 0)
    if invalid_input == "Q":
        assert np.isnan(actual[:, :, 1]).all(), "An attended input NaN must not be cleared"
        np.testing.assert_array_equal(actual[:, :, 2], 1)
    elif np.isnan(invalid_score):
        assert np.isnan(actual[:, :, 1:]).all(), "An attended V NaN must not be cleared"
    else:
        assert np.isposinf(actual[:, :, 1:]).all(), "An attended V infinity must not be cleared"


def test_attention_fp16_external_cache_softmax_accuracy():
    """FP16 storage must not force a low-accuracy FP16 softmax accumulator."""
    rng = np.random.RandomState(0)
    q = rng.rand(2, 4, 1, 8).astype(np.float16)
    k = rng.rand(2, 2, 8, 8).astype(np.float16)
    v = rng.rand(2, 2, 8, 8).astype(np.float16)
    lengths = np.array([8, 5], dtype=np.int64)
    model = build_model(
        "Attention",
        [_t("Q", list(q.shape), DT.FLOAT16), _t("K", list(k.shape), DT.FLOAT16),
         _t("V", list(v.shape), DT.FLOAT16), None, None, None,
         _t("lengths", [2], DT.INT64)],
        [_t("Y", list(q.shape), DT.FLOAT16)],
        attributes={"is_causal": 1},
    )
    feeds = {"Q": q, "K": k, "V": v, "lengths": lengths}
    allowed = np.arange(8) < lengths[:, None, None, None]
    expected = _attention_reference(q, k, v, allowed)
    assert_mlx_claims(model, feeds)
    np.testing.assert_allclose(m.run_mlx(model, feeds)[0], expected, rtol=0.001, atol=1e-7)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_attention_3d_cached_fully_masked_rows(dtype):
    rng = np.random.default_rng(17)
    q = rng.random((1, 4, 2, 8)).astype(dtype)
    k = rng.random((1, 2, 2, 8)).astype(dtype)
    v = rng.random((1, 2, 2, 8)).astype(dtype)
    pk = rng.random(k.shape).astype(dtype)
    pv = rng.random(v.shape).astype(dtype)
    mask = np.array([[False, False, False, False], [True, True, False, True]])
    dt = DT.FLOAT16 if dtype == np.float16 else DT.FLOAT
    q3 = q.transpose(0, 2, 1, 3).reshape(1, 2, 32)
    k3 = k.transpose(0, 2, 1, 3).reshape(1, 2, 16)
    v3 = v.transpose(0, 2, 1, 3).reshape(1, 2, 16)
    model = build_model(
        "Attention",
        [_t("Q", list(q3.shape), dt), _t("K", list(k3.shape), dt),
         _t("V", list(v3.shape), dt), _t("M", list(mask.shape), DT.BOOL),
         _t("PK", list(pk.shape), dt), _t("PV", list(pv.shape), dt)],
        [_t("Y", list(q3.shape), dt), _t("present_K", [1, 2, 4, 8], dt),
         _t("present_V", [1, 2, 4, 8], dt)],
        attributes={"q_num_heads": 4, "kv_num_heads": 2, "is_causal": 1},
    )
    feeds = {"Q": q3, "K": k3, "V": v3, "M": mask, "PK": pk, "PV": pv}
    present_k = np.concatenate((pk, k), axis=2)
    present_v = np.concatenate((pv, v), axis=2)
    expected = _attention_reference(q, present_k, present_v, mask)
    expected = expected.transpose(0, 2, 1, 3).reshape(q3.shape)
    assert_mlx_claims(model, feeds)
    actual, actual_k, actual_v = m.run_mlx(model, feeds)
    np.testing.assert_allclose(actual, expected, rtol=0.001, atol=1e-7)
    np.testing.assert_array_equal(actual[:, 0], 0)
    np.testing.assert_array_equal(actual_k, present_k)
    np.testing.assert_array_equal(actual_v, present_v)


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_attention_empty_output_guard_preserves_qk_diagnostics(mode, invalid_value):
    q = np.ones((1, 2, 2, 8), dtype=np.float32)
    k = np.ones_like(q)
    v = np.full_like(q, invalid_value)
    mask = np.array([[False, False], [True, True]])
    model = build_model(
        "Attention",
        [_t("Q", list(q.shape)), _t("K", list(k.shape)), _t("V", list(v.shape)),
         _t("M", list(mask.shape), DT.BOOL)],
        [_t("Y", list(q.shape)), _t("PK", list(k.shape)), _t("PV", list(v.shape)),
         _t("QK", [1, 2, 2, 2])],
        attributes={"softcap": 2.0, "qk_matmul_output_mode": mode},
    )
    feeds = {"Q": q, "K": k, "V": v, "M": mask}
    assert_mlx_claims(model, feeds)
    actual, present_k, present_v, qk = m.run_mlx(model, feeds)
    np.testing.assert_array_equal(actual[:, :, 0], 0)
    if np.isnan(invalid_value):
        assert np.isnan(actual[:, :, 1]).all()
    else:
        assert np.isposinf(actual[:, :, 1]).all()
    np.testing.assert_array_equal(present_k, k)
    np.testing.assert_array_equal(present_v, v)
    raw = np.sqrt(8.0)
    capped = 2.0 * np.tanh(raw / 2.0)
    expected = np.full((1, 2, 2, 2), raw if mode == 0 else capped)
    if mode == 2:
        expected[:, :, 0] = -np.inf
    elif mode == 3:
        expected[:, :, 0] = 0
        expected[:, :, 1] = 0.5
    np.testing.assert_allclose(qk, expected, rtol=1e-6, atol=1e-7)
