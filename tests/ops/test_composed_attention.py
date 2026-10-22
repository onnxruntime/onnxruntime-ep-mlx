"""Contrib score composition, pinned to ORT v1.31.0 rather than ONNX Attention modes.

GQA CPU oracle: gqa_attention_base.h:1531-1610. PagedAttention has no CPU kernel;
its NumPy oracle follows paged_attention_impl.cu:873-886,913-947 (W counts the
current token, with no right bound when is_causal=0).
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT

import _models as m
from test_attention_ext import _t, assert_mlx_claims, build_model
from test_paged_attention import _inputs, _model, _paged_ref


def _strict_session(model):
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    providers = [p for p in m.EP_PROVIDERS
                 if (p[0] if isinstance(p, tuple) else p) != "CPUExecutionProvider"]
    return ort.InferenceSession(model, options, providers=providers)


def _gqa(dtype, *, batch=2, seq=2, cap=6, mode=1, softcap=0.7, smooth=0,
         sink=True, bias=True, packed=False, cache=True, holes=False, window=-1,
         causal=True, rotary=False):
    rng = np.random.default_rng(731)
    heads, kv, head = 4, 2, 8
    dt = DT.FLOAT16 if dtype == np.float16 else DT.FLOAT
    q = (rng.normal(size=(batch, seq, heads * head)) * 2).astype(dtype)
    k = (rng.normal(size=(batch, seq, kv * head)) * 2).astype(dtype)
    v = rng.normal(size=k.shape).astype(dtype)
    pk = rng.normal(size=(batch, kv, cap, head)).astype(dtype)
    pv = rng.normal(size=pk.shape).astype(dtype)
    lengths = np.array([seq + (2 if cache else 0) - b for b in range(batch)], np.int32)
    total = int(lengths.max())
    feeds = dict(q=q, k=k, v=v, lengths=lengths - 1, total=np.array([total], np.int32))
    inputs = [_t("q", list(q.shape), dt), _t("k", list(k.shape), dt), _t("v", list(v.shape), dt),
              _t("pk", list(pk.shape), dt) if cache else None,
              _t("pv", list(pv.shape), dt) if cache else None,
              _t("lengths", [batch], DT.INT32), _t("total", [1], DT.INT32)]
    if cache:
        feeds.update(pk=pk, pv=pv)
    if packed:
        feeds["q"] = np.concatenate([q, k, v], axis=-1)
        del feeds["k"], feeds["v"]
        inputs[:3] = [_t("q", list(feeds["q"].shape), dt), None, None]
    inputs += [None, None, None, None, None]
    if rotary:
        for index, name, value in [(7, "cos", 1.25), (8, "sin", 0.25)]:
            feeds[name] = np.full((cap + seq + 2, head // 2), value, dtype)
            inputs[index] = _t(name, list(feeds[name].shape), dt)
    if bias:
        feeds["bias"] = rng.uniform(-4, 4, size=(batch, heads, seq, total)).astype(dtype)
        # Bias must be added after capping, and -inf must never be capped into a finite score.
        feeds["bias"][..., 0] = -np.inf
        inputs[10] = _t("bias", [batch, heads, seq, "total_length"], dt)
    if sink:
        feeds["sink"] = np.array([-1.5, 0.0, 2.0, 5.0], dtype)
        inputs[11] = _t("sink", [heads], dt)
    outputs = [_t("y", list(q.shape), dt),
               _t("present_k", [batch, kv, cap if cache else "capacity", head], dt),
               _t("present_v", [batch, kv, cap if cache else "capacity", head], dt)]
    if mode:
        outputs.append(_t("qk", [batch, heads, seq, "total_length"], dt))
    model = onnx.load_model_from_string(build_model(
        "GroupQueryAttention", inputs, outputs, domain="com.microsoft",
        attributes={"num_heads": heads, "kv_num_heads": kv, "qk_output": mode,
                    "softcap": softcap, "smooth_softmax": smooth,
                    "local_window_size": window, "causal": int(causal), "do_rotary": int(rotary)},
    ))
    if holes:
        model.graph.node[0].output[1:3] = ["", ""]
        del model.graph.output[1:3]
    return model.SerializeToString(), feeds, (q, k, v)


def _gqa_reference(feeds, original, *, mode, softcap, smooth, window=-1, causal=True, scale=None):
    q, k, v = (a.astype(np.float32) for a in original)
    batch, seq, hidden = q.shape
    heads, kv, head = 4, 2, 8
    total = int(feeds["total"][0])
    capacity = max(total, feeds["pk"].shape[2]) if "pk" in feeds else total
    pk = feeds.get("pk", np.zeros((batch, kv, 0, head), q.dtype)).astype(np.float32)
    pv = feeds.get("pv", np.zeros_like(pk)).astype(np.float32)
    present_k = np.zeros((batch, kv, capacity, head), np.float32)
    present_v = np.zeros_like(present_k)
    present_k[:, :, :pk.shape[2]] = pk
    present_v[:, :, :pv.shape[2]] = pv
    out = np.empty((batch, seq, hidden), np.float32)
    diagnostic = np.zeros((batch, heads, seq, total), np.float32)
    for b, length in enumerate(feeds["lengths"] + 1):
        past = max(int(length) - seq, 0)
        rows = int(length) - past
        qh = q[b].reshape(seq, heads, head).transpose(1, 0, 2)
        kh = k[b].reshape(seq, kv, head).transpose(1, 0, 2)
        vh = v[b].reshape(seq, kv, head).transpose(1, 0, 2)
        present_k[b, :, past:length] = kh[:, :rows]
        present_v[b, :, past:length] = vh[:, :rows]
        keys = np.repeat(present_k[b, :, :length], heads // kv, axis=0)
        vals = np.repeat(present_v[b, :, :length], heads // kv, axis=0)
        scores = (qh @ keys.transpose(0, 2, 1)) * (1 / np.sqrt(head) if scale is None else scale)
        if softcap:
            scores = softcap * np.tanh(scores / softcap)
        if "bias" in feeds:
            scores += feeds["bias"][b, :, :, :length].astype(np.float32)
        end = np.minimum(past + np.arange(seq) + 1, length) if causal else np.full(seq, length)
        start = np.maximum(0, end - window) if window > 0 else np.zeros(seq, int)
        col = np.arange(length)[None, :]
        visible = (col >= start[:, None]) & (col < end[:, None])
        before = np.where(visible[None], scores, 0)
        scores = np.where(visible[None], scores, -np.inf)
        sink = feeds.get("sink", np.zeros(heads) if smooth else None)
        if sink is not None:
            logits = np.concatenate([scores, np.broadcast_to(sink[:, None, None], (heads, seq, 1))], -1)
        else:
            logits = scores
        e = np.exp(logits - logits.max(-1, keepdims=True))
        probs = (e / e.sum(-1, keepdims=True))[..., :length]
        diagnostic[b, :, :, :length] = before if mode == 1 else probs
        out[b] = (probs @ vals).transpose(1, 0, 2).reshape(seq, hidden)
    return out, present_k, present_v, diagnostic


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("smooth,sink", [(0, False), (1, False), (0, True), (1, True)])
@pytest.mark.parametrize("packed", [False, True])
def test_gqa_composed_live_ragged(dtype, mode, smooth, sink, packed):
    model, feeds, original = _gqa(dtype, mode=mode, smooth=smooth, sink=sink, packed=packed)
    assert_mlx_claims(model, feeds)
    session = _strict_session(model)
    for step in range(2):
        # Same shapes, changed data/lengths/bias/sink/cache: no frozen trace or cache key.
        if step:
            feeds["lengths"] += 1
            feeds["total"] += 1
            if "sink" in feeds:
                feeds["sink"] += dtype(0.5)
            feeds["pv"] += dtype(0.25)
            bias = feeds["bias"]
            feeds["bias"] = np.concatenate([bias, np.full((*bias.shape[:-1], 1), -0.75, dtype)], -1)
            original[2][:] += dtype(0.5)
            if packed:
                feeds["q"][..., -original[2].shape[-1]:] = original[2]
        expected = _gqa_reference(feeds, original, mode=mode, softcap=0.7, smooth=smooth)
        actual = session.run(None, feeds)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, rtol=3e-3, atol=3e-3)
        np.testing.assert_array_equal(actual[1], expected[1].astype(dtype))
        np.testing.assert_array_equal(actual[2], expected[2].astype(dtype))


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("softcap", [0.0, 0.7])
@pytest.mark.parametrize("causal,window,cache", [(True, -1, True), (True, 2, True),
                                               (False, -1, True), (True, 2, False)])
def test_gqa_qk_cpu_contract(mode, softcap, causal, window, cache):
    # No ragged tail: ORT copies unspecified scratch beyond each member's live length.
    model, feeds, original = _gqa(np.float32, batch=1, mode=mode, softcap=softcap,
                                  smooth=1, sink=True, bias=window == -1,
                                  causal=causal, window=window, cache=cache)
    assert_mlx_claims(model, feeds)
    actual = _strict_session(model).run(None, feeds)
    cpu = m.run_cpu(model, feeds)
    expected = _gqa_reference(feeds, original, mode=mode, softcap=softcap, smooth=1,
                              window=window, causal=causal)
    for got, want in [(actual[0], cpu[0]), (actual[3], cpu[3])]:
        np.testing.assert_allclose(got, want, rtol=3e-3, atol=3e-3)
    for got, want in [(actual[0], expected[0]), (actual[3], expected[3])]:
        np.testing.assert_allclose(got, want, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("cache", [False, True])
def test_gqa_qk_optional_output_holes(cache):
    model, feeds, original = _gqa(np.float32, cache=cache, holes=True)
    assert_mlx_claims(model, feeds)
    actual = _strict_session(model).run(None, feeds)
    expected = _gqa_reference(feeds, original, mode=1, softcap=0.7, smooth=0)
    for got, want in zip(actual, [expected[0], expected[3]]):
        np.testing.assert_allclose(got, want, rtol=3e-3, atol=3e-3)


def test_gqa_invisible_bias_is_not_evaluated():
    model, feeds, original = _gqa(np.float32, batch=1, seq=3, cache=False)
    feeds["bias"][:, :, 0, 1:] = np.nan
    expected = _gqa_reference(feeds, original, mode=1, softcap=0.7, smooth=0)
    actual = _strict_session(model).run(None, feeds)
    for got, want in [(actual[0], expected[0]), (actual[3], expected[3])]:
        np.testing.assert_allclose(got, want, rtol=3e-3, atol=3e-3)
    np.testing.assert_array_equal(actual[3][:, :, 0, 1:], 0)


@pytest.mark.parametrize("mode", [1, 2])
def test_gqa_composed_negative_scale(mode):
    model, feeds, original = _gqa(np.float32, batch=1, mode=mode)
    model = onnx.load_model_from_string(model)
    model.graph.node[0].attribute.append(onnx.helper.make_attribute("scale", -0.2))
    model = model.SerializeToString()
    expected = _gqa_reference(feeds, original, mode=mode, softcap=0.7, smooth=0, scale=-0.2)
    actual = _strict_session(model).run(None, feeds)
    cpu = m.run_cpu(model, feeds)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=3e-3, atol=3e-3)
    for got, want in [(actual[0], cpu[0]), (actual[3], cpu[3])]:
        np.testing.assert_allclose(got, want, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("window", [1, 2])
def test_gqa_ragged_padded_query_window(window):
    model, feeds, original = _gqa(np.float32, seq=3, cache=False, window=window, bias=False)
    expected = _gqa_reference(feeds, original, mode=1, softcap=0.7, smooth=0, window=window)
    actual = _strict_session(model).run(None, feeds)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("packed", [False, True])
def test_gqa_composed_norm_rotary(dtype, packed):
    model, feeds, original = _gqa(dtype, seq=1, mode=2, packed=packed, rotary=True)
    model = onnx.load_model_from_string(model)
    node = model.graph.node[0]
    while len(node.input) < 14:
        node.input.append("")
    for name in ("qw", "kw"):
        node.input.append(name)
        model.graph.input.append(onnx.helper.make_tensor_value_info(
            name, onnx.TensorProto.FLOAT16 if dtype == np.float16 else onnx.TensorProto.FLOAT, [8]))
        feeds[name] = np.linspace(0.5, 1.5, 8).astype(dtype)
    model = model.SerializeToString()
    transformed = [original[0].copy(), original[1].copy(), original[2]]
    session = _strict_session(model)
    for step in range(2):
        feeds["cos"] += dtype(step * 0.25)
        feeds["qw"] += dtype(step * 0.125)
        for i, (heads, weight) in enumerate([(4, "qw"), (2, "kw")]):
            x = original[i].reshape(2, 1, heads, 8).astype(np.float32)
            x = (x / np.sqrt(np.mean(x * x, -1, keepdims=True) + 1e-6)
                 * feeds[weight].astype(np.float32)).astype(dtype)
            for b, length in enumerate(feeds["lengths"] + 1):
                pos = int(length) - 1
                c = np.tile(feeds["cos"][pos], 2)
                s = np.tile(feeds["sin"][pos], 2)
                rotated = np.concatenate([-x[b, ..., 4:], x[b, ..., :4]], -1)
                transformed[i][b] = (x[b] * c + rotated * s).reshape(1, heads * 8)
        expected = _gqa_reference(feeds, transformed, mode=2, softcap=0.7, smooth=0)
        assert_mlx_claims(model, feeds)
        actual = session.run(None, feeds)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("feature", ["softcap", "smooth", "sink", "qk1", "qk2"])
def test_gqa_composed_decode_live(feature):
    mode = int(feature[-1]) if feature.startswith("qk") else 0
    cap = 0.7 if feature == "softcap" else 0.0
    smooth = int(feature == "smooth")
    model, feeds, original = _gqa(np.float32, batch=1, seq=1, mode=mode,
                                  softcap=cap, smooth=smooth, sink=feature == "sink", bias=False)
    session = _strict_session(model)
    for length in (3, 4, 2):
        feeds["lengths"][:] = length - 1
        feeds["total"][:] = length
        original[2][:] += np.float32(0.25)
        expected = _gqa_reference(feeds, original, mode=mode, softcap=cap, smooth=smooth)
        actual = session.run(None, feeds)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, rtol=3e-3, atol=3e-3)


def test_gqa_composed_trace_is_eager(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    trace = tmp_path / "composed.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", f"{Path(__file__).name}::test_gqa_composed_decode_live",
         "-q", "-p", "no:cacheprovider"],
        check=True, cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)}, timeout=120,
    )
    events = [e for e in json.loads(trace.read_text())
              if e.get("name", "").startswith("mlx.compute[")
              and e.get("args", {}).get("ops") == "GroupQueryAttention"]
    assert len(events) == 15, events
    assert all(e["name"] == "mlx.compute[eager]" for e in events), events


@pytest.mark.parametrize("seq,cap", [(1, 2), (3, 4)])
@pytest.mark.parametrize("smooth,sink", [(1, False), (0, True), (1, True)])
def test_gqa_sliding_softcap_sink(seq, cap, smooth, sink):
    model, feeds, _ = _gqa(np.float32, batch=1, seq=seq, cap=cap, mode=0,
                           bias=False, window=2, smooth=smooth, sink=sink)
    model = onnx.load_model_from_string(model)
    model.graph.node[0].attribute.append(onnx.helper.make_attribute("sliding_window_cache", 1))
    model = model.SerializeToString()
    # Before and after eviction; compare only the specified resident cache prefix.
    session = _strict_session(model)
    for total in (cap + seq, cap + seq + 1):
        feeds["lengths"][:] = total - 1
        feeds["total"][:] = total
        assert_mlx_claims(model, feeds)
        actual = session.run(None, feeds)
        cpu = m.run_cpu(model, feeds)
        np.testing.assert_allclose(actual[0], cpu[0], rtol=3e-3, atol=3e-3)
        gap = cap - 2 + 1
        resident = total - gap * ((total - cap + gap - 1) // gap)
        for got, want in zip(actual[1:], cpu[1:]):
            np.testing.assert_array_equal(got[:, :, :resident], want[:, :, :resident])


@pytest.mark.parametrize("invalid", [
    "mode", "coupling", "missing_qk", "qk_type", "sink_shape", "sink_type", "bias_heads", "sliding_qk",
])
def test_gqa_composed_invalid_contract(invalid):
    model, feeds, _ = _gqa(np.float32, batch=1, bias=False)
    model = onnx.load_model_from_string(model)
    node = model.graph.node[0]
    if invalid in ("mode", "coupling"):
        for a in node.attribute:
            if a.name == "qk_output":
                a.i = 3 if invalid == "mode" else 0
    elif invalid == "missing_qk":
        del node.output[3]
        del model.graph.output[3]
    elif invalid == "qk_type":
        model.graph.output[3].type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
    elif invalid.startswith("sink"):
        sink = next(v for v in model.graph.input if v.name == "sink")
        if invalid == "sink_shape":
            sink.type.tensor_type.shape.dim[0].dim_value = 3
        else:
            sink.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
    elif invalid == "bias_heads":
        node.input[10] = "bad_bias"
        model.graph.input.append(onnx.helper.make_tensor_value_info(
            "bad_bias", onnx.TensorProto.FLOAT, [1, 3, 2, 4]))
    else:
        node.attribute.extend([onnx.helper.make_attribute("sliding_window_cache", 1)])
        for a in node.attribute:
            if a.name == "local_window_size":
                a.i = 2
    with pytest.raises(Exception):
        _strict_session(model.SerializeToString())


@pytest.mark.parametrize("batch_broadcast,head_broadcast", [(False, True), (True, False), (True, True)])
def test_gqa_composed_bias_broadcast(batch_broadcast, head_broadcast):
    model, feeds, original = _gqa(np.float32)
    model = onnx.load_model_from_string(model)
    bias = feeds["bias"]
    if batch_broadcast:
        bias = bias[:1]
    if head_broadcast:
        bias = bias[:, :1]
    feeds["bias"] = bias.copy()
    value = next(v for v in model.graph.input if v.name == "bias")
    value.type.tensor_type.shape.dim[0].dim_value = bias.shape[0]
    value.type.tensor_type.shape.dim[1].dim_value = bias.shape[1]
    expected_feeds = dict(feeds, bias=np.broadcast_to(bias, (2, 4, 2, 4)))
    expected = _gqa_reference(expected_feeds, original, mode=1, softcap=0.7, smooth=0)
    actual = _strict_session(model.SerializeToString()).run(None, feeds)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [-1, 1, 3])
@pytest.mark.parametrize("softcap", [0.0, 0.4])
@pytest.mark.parametrize("sink", [False, True])
@pytest.mark.parametrize("extras", [False, True])
def test_paged_composed_live_cache(packed, causal, window, softcap, sink, extras):
    heads, kv, head, blocks, block_size, batch = 4, 2, 8, 16, 4, 2
    model = onnx.load_model_from_string(_model(heads, kv, head, blocks, block_size, batch, 1, 0))
    feeds, _ = _inputs(heads, kv, head, blocks, block_size, batch, 1, 0, 423)
    node = model.graph.node[0]
    node.attribute.extend(onnx.helper.make_attribute(k, v) for k, v in
                          dict(is_causal=int(causal), local_window_size=window, softcap=softcap).items())
    if sink:
        while len(node.input) < 11:
            node.input.append("")
        node.input.append("sink")
        model.graph.input.append(onnx.helper.make_tensor_value_info("sink", onnx.TensorProto.FLOAT16, [heads]))
        feeds["sink"] = np.array([-2, 0, 1, 4], np.float16)
    if extras:
        tokens = len(feeds["query"])
        for index, name, dtype, shape in [
            (10, "slots", onnx.TensorProto.INT32, [tokens]),
            (12, "qw", onnx.TensorProto.FLOAT16, [head]),
            (13, "kw", onnx.TensorProto.FLOAT16, [head]),
            (16, "metadata", onnx.TensorProto.INT32, [3]),
        ]:
            while len(node.input) <= index:
                node.input.append("")
            node.input[index] = name
            model.graph.input.append(onnx.helper.make_tensor_value_info(name, dtype, shape))
        slots = []
        cum, past, bt = (feeds[n] for n in ("cumulative_sequence_length", "past_seqlens", "block_table"))
        for b in range(batch):
            for j in range(int(cum[b + 1] - cum[b])):
                pos = int(past[b]) + j
                slots.append(int(bt[b, pos // block_size]) * block_size + pos % block_size)
        feeds["slots"] = np.array(slots, np.int32)
        feeds["slots"][0] = -1
        feeds["qw"] = np.linspace(0.5, 1.5, head).astype(np.float16)
        feeds["kw"] = np.linspace(1.5, 0.5, head).astype(np.float16)
        feeds["metadata"] = np.array([1000, 1000, 0], np.int32)
    q, k, v = (feeds[n] for n in ("query", "key", "value"))
    if packed:
        node.input[1:3] = ["", ""]
        for name in ("key", "value"):
            model.graph.input.remove(next(i for i in model.graph.input if i.name == name))
            del feeds[name]
        model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = (heads + 2 * kv) * head
        feeds["query"] = np.concatenate([q, k, v], -1)
    model = model.SerializeToString()
    assert_mlx_claims(model, {n: a.copy() for n, a in feeds.items()})
    session = _strict_session(model)
    for step in range(2):
        v += np.float16(step * 0.25)
        feeds["cos_cache"] += np.float16(step * 0.125)
        if sink:
            feeds["sink"] += np.float16(step * 0.5)
        if packed:
            feeds["query"] = np.concatenate([q, k, v], -1)
        expected, kc, vc = _paged_ref(
            q, k, v, feeds["key_cache"], feeds["value_cache"],
            feeds["cumulative_sequence_length"], feeds["past_seqlens"], feeds["block_table"],
            heads, kv, head, feeds["cos_cache"], feeds["sin_cache"],
            head_sink=feeds.get("sink"), is_causal=causal, softcap=softcap,
            local_window_size=window, return_cache=True,
            activation_dtype=np.float16,
            slot_mapping=feeds.get("slots"), q_norm=feeds.get("qw"), k_norm=feeds.get("kw"),
        )
        live = {n: a.copy() for n, a in feeds.items()}
        actual = session.run(None, live)[0]
        np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)
        np.testing.assert_array_equal(live["key_cache"], kc.astype(np.float16))
        np.testing.assert_array_equal(live["value_cache"], vc.astype(np.float16))


def _bf16_boundary(model):
    model = onnx.load_model_from_string(model)
    node = model.graph.node[0]
    before, after = [], []
    for value in model.graph.input:
        if value.type.tensor_type.elem_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16):
            value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            name = value.name + "_bf16"
            before.append(onnx.helper.make_node("Cast", [value.name], [name], to=onnx.TensorProto.BFLOAT16))
            node.input[:] = [name if i == value.name else i for i in node.input]
    for value in model.graph.output:
        value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        name = value.name + "_bf16"
        node.output[:] = [name if o == value.name else o for o in node.output]
        after.append(onnx.helper.make_node("Cast", [name], [value.name], to=onnx.TensorProto.FLOAT))
    nodes = before + [node] + after
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model.SerializeToString()


@pytest.mark.parametrize("mode", [1, 2])
def test_gqa_composed_bfloat16(mode):
    import ml_dtypes

    model, feeds, _ = _gqa(np.float32, mode=mode)
    model = _bf16_boundary(model)
    feeds = {n: a.astype(ml_dtypes.bfloat16).astype(np.float32) if a.dtype == np.float32 else a
             for n, a in feeds.items()}
    expected = _gqa_reference(feeds, (feeds["q"], feeds["k"], feeds["v"]),
                              mode=mode, softcap=0.7, smooth=0)
    actual = _strict_session(model).run(None, feeds)
    for got, want in zip(actual, expected):
        want = want.astype(ml_dtypes.bfloat16).astype(np.float32)
        np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-3)


def test_paged_composed_bfloat16():
    import ml_dtypes

    model = onnx.load_model_from_string(_model(4, 2, 8, 8, 4, 2, 0, 0))
    model.graph.node[0].attribute.extend([
        onnx.helper.make_attribute("softcap", 0.4),
        onnx.helper.make_attribute("local_window_size", 2),
    ])
    model = _bf16_boundary(model.SerializeToString())
    feeds, _ = _inputs(4, 2, 8, 8, 4, 2, 0, 0, 423)
    feeds = {n: a.astype(ml_dtypes.bfloat16).astype(np.float32) if a.dtype == np.float16 else a
             for n, a in feeds.items()}
    expected = _paged_ref(
        feeds["query"], feeds["key"], feeds["value"], feeds["key_cache"], feeds["value_cache"],
        feeds["cumulative_sequence_length"], feeds["past_seqlens"], feeds["block_table"],
        4, 2, 8, softcap=0.4, local_window_size=2,
    )
    actual = _strict_session(model).run(None, feeds)[0]
    np.testing.assert_allclose(actual, expected.astype(ml_dtypes.bfloat16).astype(np.float32),
                               rtol=1e-2, atol=1e-3)
