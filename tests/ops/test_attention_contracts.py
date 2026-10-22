"""Live-value regressions for the attention and supplied-cache rotary contracts."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT

import _models as m
from test_attention_ext import _t, assert_mlx_claims, build_model
from test_paged_attention import _inputs, _model, _paged_ref


@pytest.mark.parametrize("opset", [23, 24, 25])
@pytest.mark.parametrize("shape", [[1], [2, 1], [1, 1, 2, 1]])
@pytest.mark.parametrize("boolean", [False, True])
@pytest.mark.parametrize("composed", [False, True])
def test_attention_singleton_key_mask(opset, shape, boolean, composed):
    q = np.zeros((1, 2, 2, 8), np.float32)
    k = np.zeros((1, 1, 3, 8), np.float32)
    v = np.broadcast_to(np.arange(3, dtype=np.float32)[None, None, :, None] * 2, k.shape).copy()
    mask = np.ones(shape, bool) if boolean else np.zeros(shape, np.float32)
    model = build_model(
        "Attention",
        [_t("q", list(q.shape)), _t("k", list(k.shape)), _t("v", list(v.shape)),
         _t("mask", shape, DT.BOOL if boolean else DT.FLOAT)],
        [_t("y", list(q.shape))],
        attributes={"softmax_precision": int(DT.FLOAT)} if composed else {},
        opset=opset,
    )
    feeds = {"q": q, "k": k, "v": v, "mask": mask}
    assert_mlx_claims(model, feeds)
    np.testing.assert_allclose(m.run_mlx(model, feeds)[0], 2, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("causal,window", [(0, -1), (1, -1), (1, 1)])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("with_past", [False, True])
def test_gqa_live_lengths_defaults_and_causality(monkeypatch, eager, causal, window, batch, with_past):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    seq, cap, head = (1, 4, 8) if with_past else (3, 0, 8)
    q = np.zeros((batch, seq, head), np.float32)
    k = q.copy()
    v = np.broadcast_to(np.arange(seq, dtype=np.float32)[None, :, None] + 6, q.shape).copy()
    pk = np.zeros((batch, 1, cap, head), np.float32)
    pv = np.broadcast_to(np.arange(cap, dtype=np.float32)[None, None, :, None] * 2, pk.shape).copy()
    inputs = [_t("q", list(q.shape)), _t("k", list(k.shape)), _t("v", list(v.shape)),
              _t("pk", list(pk.shape)) if with_past else None,
              _t("pv", list(pv.shape)) if with_past else None,
              _t("lengths", [batch], DT.INT32), _t("total", [1], DT.INT32)]
    model = build_model(
        "GroupQueryAttention", inputs,
        [_t("y", list(q.shape)), _t("present_k", [batch, 1, cap or seq, head]),
         _t("present_v", [batch, 1, cap or seq, head])],
        domain="com.microsoft",
        attributes={"num_heads": 1, "kv_num_heads": 1, "causal": causal, "local_window_size": window},
    )
    feeds = {"q": q, "k": k, "v": v}
    if with_past:
        feeds.update(pk=pk, pv=pv)
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    for step in range(2):
        lengths = np.full(batch, 3 + step if with_past else seq, np.int32)
        if with_past and batch == 2:
            lengths[1] = 1 + step
        feeds.update(lengths=lengths - 1, total=np.array([lengths.max()], np.int32))
        feeds["v"] = v + step * 2
        expected = np.empty_like(q)
        expected_pv = pv.copy() if with_past else np.zeros((batch, 1, seq, head), np.float32)
        for b, length in enumerate(lengths):
            past = max(int(length) - seq, 0)
            expected_pv[b, 0, past:length] = feeds["v"][b, :length - past]
            for i in range(seq):
                end = min(past + i + 1, length) if causal else length
                start = max(0, past + i - window + 1) if window > 0 else 0
                expected[b, i] = expected_pv[b, 0, start:end].mean(axis=0)
        assert_mlx_claims(model, feeds)
        actual, _, actual_pv = session.run(None, feeds)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(actual_pv, expected_pv)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("supplied_rotary", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_gqa_default_rotary_ignores_optional_caches(packed, supplied_rotary, causal):
    from test_attention_semantics import _attention_reference

    rng = np.random.default_rng(82)
    q = rng.normal(size=(1, 3, 16)).astype(np.float32)
    k = rng.normal(size=(1, 3, 8)).astype(np.float32)
    v = rng.normal(size=k.shape).astype(np.float32)
    feeds = dict(q=q, k=k, v=v, lengths=np.array([2], np.int32), total=np.array([3], np.int32))
    if packed:
        feeds["q"] = np.concatenate([q, k, v], axis=-1)
        del feeds["k"], feeds["v"]
    inputs = [_t("q", list(feeds["q"].shape)),
              None if packed else _t("k", list(k.shape)), None if packed else _t("v", list(v.shape)),
              None, None, _t("lengths", [1], DT.INT32), _t("total", [1], DT.INT32)]
    if supplied_rotary:
        inputs += [_t("cos", [3, 4]), _t("sin", [3, 4])]
        feeds.update(cos=np.full((3, 4), 2, np.float32), sin=np.ones((3, 4), np.float32))
    model = build_model(
        "GroupQueryAttention", inputs, [_t("y", list(q.shape))], domain="com.microsoft",
        attributes={"num_heads": 2, "kv_num_heads": 1, "causal": int(causal)},
    )
    expected = _attention_reference(
        q.reshape(1, 3, 2, 8).transpose(0, 2, 1, 3),
        k.reshape(1, 3, 1, 8).transpose(0, 2, 1, 3),
        v.reshape(1, 3, 1, 8).transpose(0, 2, 1, 3),
        np.tri(3, dtype=bool) if causal else np.ones((3, 3), bool),
    ).transpose(0, 2, 1, 3).reshape(q.shape)
    assert_mlx_claims(model, feeds)
    np.testing.assert_allclose(m.run_mlx(model, feeds)[0], expected, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("batch,sliding", [(1, False), (2, False), (1, True), (2, True)])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("constant", [False, True])
def test_gqa_exact_rotary_and_ragged_positions(dtype, batch, sliding, interleaved, constant):
    from test_attention_semantics import _attention_reference

    rng = np.random.default_rng(541)
    dt = DT.FLOAT16 if dtype == np.float16 else DT.FLOAT
    q = rng.normal(size=(batch, 1, 16)).astype(dtype)
    k = rng.normal(size=(batch, 1, 8)).astype(dtype)
    v = rng.normal(size=k.shape).astype(dtype)
    pk = rng.normal(size=(batch, 1, 4, 8)).astype(dtype)
    pv = rng.normal(size=pk.shape).astype(dtype)
    cos = rng.uniform(0.5, 2, size=(6, 4)).astype(dtype)
    sin = rng.uniform(-0.5, 0.5, size=cos.shape).astype(dtype)
    arrays = dict(q=q, k=k, v=v, pk=pk, pv=pv, cos=cos, sin=sin)
    inputs = [_t(name, list(arrays[name].shape), dt) for name in ("q", "k", "v", "pk", "pv")]
    inputs += [_t("lengths", [batch], DT.INT32), _t("total", [1], DT.INT32),
               _t("cos", list(cos.shape), dt), _t("sin", list(sin.shape), dt)]
    attrs = {"num_heads": 2, "kv_num_heads": 1, "do_rotary": 1,
             "rotary_interleaved": int(interleaved)}
    if sliding:
        attrs.update(sliding_window_cache=1, local_window_size=2)
    model = onnx.load_model_from_string(build_model(
        "GroupQueryAttention", inputs,
        [_t("y", list(q.shape), dt), _t("pkey", list(pk.shape), dt), _t("pvalue", list(pv.shape), dt)],
        domain="com.microsoft", attributes=attrs,
    ))
    if constant:
        for name in ("cos", "sin"):
            model.graph.initializer.append(onnx.numpy_helper.from_array(arrays[name], name))
            inp = next(i for i in model.graph.input if i.name == name)
            model.graph.input.remove(inp)
    model = model.SerializeToString()
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    for step in range(3):
        lengths = np.full(batch, 3 + min(step, 1), np.int32)
        if batch == 2:
            lengths[1] = 1 + min(step, 1)
        if not constant:
            cos[2:] += dtype(step * 0.5)
            sin[2:] += dtype(step * 0.25)
        v += dtype(step * 0.125)
        feeds = {name: value.copy() for name, value in arrays.items() if not constant or name not in ("cos", "sin")}
        feeds.update(lengths=lengths - 1, total=np.array([lengths.max()], np.int32))
        expected_k, expected_v = pk.copy(), pv.copy()
        expected = np.empty_like(q)
        for b, length in enumerate(lengths):
            past = int(length) - 1
            def rotate(x):
                if interleaved:
                    r = np.empty_like(x)
                    r[..., 0::2], r[..., 1::2] = -x[..., 1::2], x[..., 0::2]
                    c, s = np.repeat(cos[past], 2), np.repeat(sin[past], 2)
                else:
                    r = np.concatenate((-x[..., 4:], x[..., :4]), axis=-1)
                    c, s = np.tile(cos[past], 2), np.tile(sin[past], 2)
                return (x * c + r * s).astype(dtype)
            qh = rotate(q[b].reshape(1, 2, 1, 8))
            kh = rotate(k[b].reshape(1, 1, 1, 8))
            expected_k[b, :, past:past + 1] = kh[0]
            expected_v[b, :, past:past + 1] = v[b].reshape(1, 1, 8)
            allowed = np.arange(length) >= (max(0, past - 1) if sliding else 0)
            expected[b] = _attention_reference(
                qh, expected_k[b:b + 1, :, :length], expected_v[b:b + 1, :, :length], allowed
            ).reshape(1, 16)
        assert_mlx_claims(model, {name: value.copy() for name, value in feeds.items()})
        actual, actual_k, actual_v = session.run(None, feeds)
        np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)
        for b, length in enumerate(lengths):
            np.testing.assert_allclose(actual_k[b, :, :length], expected_k[b, :, :length], rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(actual_v[b, :, :length], expected_v[b, :, :length])


@pytest.mark.parametrize("target", [
    "test_gqa_live_lengths_defaults_and_causality[True-1-1--1-False]",
    "test_gqa_exact_rotary_and_ragged_positions[True-False-1-False-float32]",
])
def test_gqa_decode_trace_proves_replay(tmp_path, target):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    trace = tmp_path / "gqa_replay.json"
    subprocess.run(
        [sys.executable, "-m", "pytest",
         f"{Path(__file__).name}::{target}",
         "-q", "-p", "no:cacheprovider"],
        check=True, cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)}, timeout=120,
    )
    partitions = {}
    for event in json.loads(trace.read_text()):
        if event.get("name") == "mlx.compute[decode]" and event.get("args", {}).get("ops") == "GroupQueryAttention":
            args = event["args"]
            partitions.setdefault(args["partition_id"], []).append(args.get("cache"))
    assert ["MISS", "HIT"] in partitions.values(), partitions


def test_rotary_trace_proves_live_cache_replay(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    trace = tmp_path / "rotary_replay.json"
    subprocess.run(
        [sys.executable, "-m", "pytest",
         f"{Path(__file__).name}::test_rotary_offset_uses_exact_live_cache[4-4-0-False]",
         "-q", "-p", "no:cacheprovider"],
        check=True, cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)}, timeout=120,
    )
    partitions = {}
    for event in json.loads(trace.read_text()):
        if event.get("name") == "mlx.compute[general]" and event.get("args", {}).get("ops") == "RotaryEmbedding":
            args = event["args"]
            partitions.setdefault(args["partition_id"], []).append(args.get("cache"))
    assert ["MISS", "HIT"] in partitions.values(), partitions


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("interleaved", [0, 1])
@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("half", [2, 4])
def test_rotary_offset_uses_exact_live_cache(monkeypatch, eager, interleaved, rows, half):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    x = np.arange(1, 17, dtype=np.float32).reshape(1, 2, 1, 8)
    cos = np.full((rows, half), 2, np.float32)
    sin = np.full((rows, half), 0.25, np.float32)
    model = build_model(
        "RotaryEmbedding",
        [_t("x", list(x.shape)), _t("pos", [1], DT.INT64),
         _t("cos", list(cos.shape)), _t("sin", list(sin.shape))],
        [_t("y", list(x.shape))], domain="com.microsoft",
        attributes={"interleaved": interleaved, "rotary_embedding_dim": half * 2},
    )
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    for step in range(2):
        cos[0] = 2 + step
        sin[0] = 0.25 + step
        feeds = {"x": x, "pos": np.array([0], np.int64), "cos": cos, "sin": sin}
        a = x[..., :half * 2]
        if interleaved:
            rotated = np.empty_like(a)
            rotated[..., 0::2], rotated[..., 1::2] = -a[..., 1::2], a[..., 0::2]
            c, s = np.repeat(cos[0], 2), np.repeat(sin[0], 2)
        else:
            rotated = np.concatenate((-a[..., half:], a[..., :half]), axis=-1)
            c, s = np.tile(cos[0], 2), np.tile(sin[0], 2)
        expected = x.copy()
        expected[..., :half * 2] = a * c + rotated * s
        assert_mlx_claims(model, feeds)
        np.testing.assert_allclose(session.run(None, feeds)[0], expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("offset", [False, True])
@pytest.mark.parametrize("bad_position", [-1, 1, 2**32, 2**63 - 1])
def test_rotary_replay_rejects_invalid_positions(offset, bad_position):
    x = np.ones((1, 1, 1, 8), np.float32)
    pos = np.zeros((1,) if offset else (1, 1), np.int64)
    cos = np.ones((1, 4), np.float32)
    sin = np.zeros_like(cos)
    model = build_model(
        "RotaryEmbedding",
        [_t("x", list(x.shape)), _t("pos", list(pos.shape), DT.INT64),
         _t("cos", list(cos.shape)), _t("sin", list(sin.shape))],
        [_t("y", list(x.shape))], domain="com.microsoft",
    )
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    feeds = dict(x=x, pos=pos, cos=cos, sin=sin)
    session.disable_fallback()
    np.testing.assert_array_equal(session.run(None, feeds)[0], x)
    pos.fill(bad_position)
    with pytest.raises(Exception, match="position_ids exceed"):
        session.run(None, feeds)
    pos.fill(0)
    np.testing.assert_array_equal(session.run(None, feeds)[0], x)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("rotary", [False, True])
@pytest.mark.parametrize("metadata_size", [0, 2, 3])
def test_paged_live_caches_and_metadata(causal, rotary, metadata_size):
    heads, kv, head, blocks, block_size, batch = 4, 2, 8, 8, 4, 2
    model = onnx.load_model_from_string(_model(heads, kv, head, blocks, block_size, batch, int(rotary), 0))
    model.graph.node[0].attribute.append(onnx.helper.make_attribute("is_causal", int(causal)))
    if metadata_size:
        while len(model.graph.node[0].input) < 16:
            model.graph.node[0].input.append("")
        model.graph.node[0].input.append("metadata")
        model.graph.input.append(onnx.helper.make_tensor_value_info("metadata", onnx.TensorProto.INT32, [metadata_size]))
    model = model.SerializeToString()
    feeds, _ = _inputs(heads, kv, head, blocks, block_size, batch, int(rotary), 0, 123)
    if metadata_size:
        feeds["metadata"] = np.array([1000, 1000, 0][:metadata_size], np.int32)
    if rotary:
        feeds["cos_cache"].fill(2)
        feeds["sin_cache"].fill(0.25)
    assert_mlx_claims(model, {k: v.copy() for k, v in feeds.items()})
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    for step in range(2):
        if rotary:
            feeds["cos_cache"] *= np.float16(1.25)
        feeds["value"] += np.float16(step * 0.25)
        expected = _paged_ref(
            feeds["query"], feeds["key"], feeds["value"], feeds["key_cache"], feeds["value_cache"],
            feeds["cumulative_sequence_length"], feeds["past_seqlens"], feeds["block_table"],
            heads, kv, head, feeds.get("cos_cache"), feeds.get("sin_cache"), is_causal=causal,
        )
        actual = session.run(None, {k: v.copy() for k, v in feeds.items()})[0]
        np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("invalid", ["cumulative", "negative_past", "too_long", "block", "slot"])
def test_paged_invalid_live_metadata_errors(invalid):
    model = onnx.load_model_from_string(_model(4, 2, 8, 8, 4, 2, 0, 0))
    feeds, _ = _inputs(4, 2, 8, 8, 4, 2, 0, 0, 123)
    if invalid == "cumulative":
        feeds["cumulative_sequence_length"][1] = -1
    elif invalid == "negative_past":
        feeds["past_seqlens"][0] = -1
    elif invalid == "too_long":
        feeds["past_seqlens"][0] = 1000
    elif invalid == "block":
        feeds["block_table"].fill(-1)
    else:
        while len(model.graph.node[0].input) < 10:
            model.graph.node[0].input.append("")
        model.graph.node[0].input.append("slots")
        tokens = feeds["query"].shape[0]
        model.graph.input.append(onnx.helper.make_tensor_value_info("slots", onnx.TensorProto.INT32, [tokens]))
        feeds["slots"] = np.full(tokens, 1000, np.int32)
    session = ort.InferenceSession(model.SerializeToString(), providers=m.EP_PROVIDERS)
    session.disable_fallback()
    with pytest.raises(Exception, match="PagedAttention:"):
        session.run(None, feeds)
