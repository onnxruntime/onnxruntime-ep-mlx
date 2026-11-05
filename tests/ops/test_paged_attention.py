"""PagedAttention (com.microsoft.PagedAttention) coverage for the MLX EP.

ORT ships PagedAttention CUDA-only (no CPU kernel), so we cannot compare against an ORT reference on
this platform — and, because there is no CPU fallback, the MLX EP claiming it is the *only* way such
a model runs on Apple Silicon. We therefore validate against a numpy port of ORT's own reference
(``attention_ref`` + paged-cache read/update from test_paged_attention_cuda.py), which is exactly what
ORT validates its CUDA kernel against. Each case is proven to actually run on the MLX EP (per-node
profiling) and compared, tolerance-gated, to that reference.

Supported forms include separate/packed QKV, GQA, causal/bidirectional left windows,
softcap, per-head sinks, and optional RoPE using supplied cos/sin cache values.
"""

from __future__ import annotations

import json
import math
import os

import _models as m
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT


# ---------------- numpy reference (port of ORT's attention_ref + paged cache) --------------------
def _rotate_half(x, interleaved):
    if not interleaved:
        d = x.shape[-1] // 2
        return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)
    out = np.empty_like(x)
    out[..., 0::2] = -x[..., 1::2]
    out[..., 1::2] = x[..., 0::2]
    return out


def _apply_rope(x, cos, sin, pos, interleaved):
    c, s = cos[pos], sin[pos]  # [seq, d/2]
    if not interleaved:
        c = np.concatenate([c, c], axis=-1)
        s = np.concatenate([s, s], axis=-1)
    else:
        c = np.repeat(c, 2, axis=-1)
        s = np.repeat(s, 2, axis=-1)
    return x * c[:, None, :] + _rotate_half(x, interleaved) * s[:, None, :]


def _apply_rope_offset(x, cos, sin, pos, interleaved, offset):
    out = x.copy()
    width = cos.shape[-1] * 2
    out[..., offset : offset + width] = _apply_rope(
        x[..., offset : offset + width],
        cos,
        sin,
        pos,
        interleaved,
    )
    return out


def _paged_ref(
    q,
    k,
    v,
    kc,
    vc,
    cum,
    past,
    bt,
    H,
    KV,
    d,
    cos=None,
    sin=None,
    interleaved=0,
    slot_mapping=None,
    head_sink=None,
    q_norm=None,
    k_norm=None,
    norm_eps=1e-6,
    return_cache=False,
    is_causal=True,
    softcap=0.0,
    local_window_size=-1,
    activation_dtype=None,
):
    batch = len(cum) - 1
    scale = 1.0 / math.sqrt(d)
    out = np.zeros((q.shape[0], H * d), np.float32)
    block_size = kc.shape[1]
    kc, vc = kc.copy(), vc.copy()
    g = H // KV
    for b in range(batch):
        s0, s1 = int(cum[b]), int(cum[b + 1])
        nq = s1 - s0
        if nq == 0:
            continue
        pastb = int(past[b])
        total = pastb + nq
        qb = q[s0:s1].reshape(nq, H, d).astype(np.float32)
        kb = k[s0:s1].reshape(nq, KV, d).astype(np.float32)
        vb = v[s0:s1].reshape(nq, KV, d).astype(np.float32)
        if q_norm is not None:
            qb = qb / np.sqrt(np.mean(qb * qb, axis=-1, keepdims=True) + norm_eps) * q_norm
            kb = kb / np.sqrt(np.mean(kb * kb, axis=-1, keepdims=True) + norm_eps) * k_norm
        if cos is not None:
            pos = np.arange(pastb, total)
            if activation_dtype is not None:
                qb, kb = qb.astype(activation_dtype), kb.astype(activation_dtype)
                rope_cos, rope_sin = cos.astype(activation_dtype), sin.astype(activation_dtype)
            else:
                rope_cos, rope_sin = cos, sin
            qb = _apply_rope(qb, rope_cos, rope_sin, pos, interleaved).astype(np.float32)
            kb = _apply_rope(kb, rope_cos, rope_sin, pos, interleaved).astype(np.float32)
        for i in range(nq):
            p = pastb + i
            if slot_mapping is None:
                phys, off = int(bt[b, p // block_size]), p % block_size
            else:
                slot = int(slot_mapping[s0 + i])
                if slot < 0:
                    continue
                phys, off = divmod(slot, block_size)
            kc[phys, off], vc[phys, off] = kb[i], vb[i]
        keys = np.zeros((total, KV, d), np.float32)
        vals = np.zeros((total, KV, d), np.float32)
        for p in range(total):
            if p < pastb:
                phys, off = int(bt[b, p // block_size]), p % block_size
                keys[p], vals[p] = kc[phys, off], vc[phys, off]
            else:
                keys[p], vals[p] = kb[p - pastb], vb[p - pastb]
        keys, vals = np.repeat(keys, g, 1), np.repeat(vals, g, 1)
        qh, kh, vh = qb.transpose(1, 0, 2), keys.transpose(1, 0, 2), vals.transpose(1, 0, 2)
        scores = np.einsum("hqd,hkd->hqk", qh, kh) * scale
        if softcap > 0:
            scores = softcap * np.tanh(scores / softcap)
        col = np.arange(total)[None, :]
        row = (pastb + np.arange(nq))[:, None]
        if is_causal:
            scores = np.where((col > row)[None], -np.inf, scores)
        if local_window_size > 0:
            scores = np.where((col < row - local_window_size + 1)[None], -np.inf, scores)
        if head_sink is None:
            scores -= scores.max(-1, keepdims=True)
            e = np.exp(scores)
            attn = e / e.sum(-1, keepdims=True)
        else:
            max_score = np.maximum(scores.max(-1, keepdims=True), head_sink[:, None, None])
            e = np.exp(scores - max_score)
            sink_e = np.exp(head_sink[:, None, None] - max_score)
            attn = e / (e.sum(-1, keepdims=True) + sink_e)
        ob = np.einsum("hqk,hkd->hqd", attn, vh).transpose(1, 0, 2).reshape(nq, H * d)
        out[s0:s1] = ob
    return (out, kc, vc) if return_cache else out


# ---------------- model + feeds ----------------
def _model(H, KV, d, nb, blk, batch, do_rotary, interleaved):
    hidden, kvh, maxb = H * d, KV * d, nb // batch
    ins = [
        m.tensor("query", DT.FLOAT16, ["nt", hidden]),
        m.tensor("key", DT.FLOAT16, ["nt", kvh]),
        m.tensor("value", DT.FLOAT16, ["nt", kvh]),
        m.tensor("key_cache", DT.FLOAT16, [nb, blk, KV, d]),
        m.tensor("value_cache", DT.FLOAT16, [nb, blk, KV, d]),
        m.tensor("cumulative_sequence_length", DT.INT32, [batch + 1]),
        m.tensor("past_seqlens", DT.INT32, [batch]),
        m.tensor("block_table", DT.INT32, [batch, maxb]),
    ]
    if do_rotary:
        ins += [m.tensor("cos_cache", DT.FLOAT16, ["ms", d // 2]),
                m.tensor("sin_cache", DT.FLOAT16, ["ms", d // 2])]
    outs = [m.tensor("output", DT.FLOAT16, ["nt", hidden])]
    attrs = {"num_heads": H, "kv_num_heads": KV, "do_rotary": do_rotary,
             "rotary_interleaved": interleaved}
    return m.make_model("PagedAttention", ins, outs, domain="com.microsoft", attributes=attrs, opset=17)


def _inputs(H, KV, d, nb, blk, batch, do_rotary, interleaved, seed):
    rng = np.random.default_rng(seed)
    hidden, kvh, maxb = H * d, KV * d, nb // batch
    cap = maxb * blk
    new_lens = rng.integers(1, min(4, cap) + 1, size=batch).astype(np.int32)
    past = np.array([rng.integers(0, cap - nl + 1) for nl in new_lens], dtype=np.int32)
    cum = np.concatenate([[0], np.cumsum(new_lens)]).astype(np.int32)
    nt = int(cum[-1])
    q = (rng.standard_normal((nt, hidden)) * 0.3).astype(np.float16)
    k = (rng.standard_normal((nt, kvh)) * 0.3).astype(np.float16)
    v = (rng.standard_normal((nt, kvh)) * 0.3).astype(np.float16)
    kc = (rng.standard_normal((nb, blk, KV, d)) * 0.3).astype(np.float16)
    vc = (rng.standard_normal((nb, blk, KV, d)) * 0.3).astype(np.float16)
    block_table = rng.permutation(nb).astype(np.int32).reshape(batch, maxb)
    feeds = {"query": q, "key": k, "value": v, "key_cache": kc, "value_cache": vc,
             "cumulative_sequence_length": cum, "past_seqlens": past, "block_table": block_table}
    cos = sin = None
    if do_rotary:
        half = d // 2
        inv = 1.0 / (10000.0 ** (np.arange(half) / half))
        ang = np.arange(cap)[:, None] * inv[None, :]  # structured RoPE cache (what real models emit)
        cos = np.cos(ang).astype(np.float16); sin = np.sin(ang).astype(np.float16)
        feeds["cos_cache"] = cos; feeds["sin_cache"] = sin
    ref = _paged_ref(q, k, v, kc, vc, cum, past, block_table, H, KV, d,
                     cos=(None if cos is None else cos.astype(np.float32)),
                     sin=(None if sin is None else sin.astype(np.float32)), interleaved=interleaved)
    return feeds, ref


def _assert_mlx_claims(model, feeds):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = "mlx_pa_probe"
    sess = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)
    sess.run(None, feeds)
    path = sess.end_profiling()
    try:
        events = json.load(open(path))
    finally:
        os.remove(path)
    providers = {e.get("args", {}).get("provider") for e in events
                 if e.get("cat") == "Node" and e.get("args", {}).get("provider")}
    assert "MLXExecutionProvider" in providers, f"MLX EP did not claim PagedAttention (ran on {providers})"


def _check(H, KV, d, nb, blk, batch, do_rotary=0, interleaved=0, seed=0):
    model = _model(H, KV, d, nb, blk, batch, do_rotary, interleaved)
    feeds, ref = _inputs(H, KV, d, nb, blk, batch, do_rotary, interleaved, seed)
    _assert_mlx_claims(model, feeds)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    out = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS).run(None, feeds)[0].astype(np.float32)
    np.testing.assert_allclose(out, ref, rtol=2e-2, atol=2e-2)


# --- GQA / MQA / MHA, causal, no RoPE -----------------------------------------------------------
@pytest.mark.parametrize("H,KV", [(4, 2), (4, 4), (8, 2), (2, 1)], ids=["gqa", "mha", "gqa8", "mqa"])
@pytest.mark.parametrize("batch", [1, 3])
def test_paged_attention_core(H, KV, batch):
    _check(H, KV, 16, nb=8 * batch, blk=4, batch=batch, seed=H * 10 + KV + batch)


@pytest.mark.parametrize("kv", [1, 4])
def test_paged_attention_grouped_head256(kv):
    _check(24, kv, 256, nb=4, blk=64, batch=1, seed=256 + kv)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [-1, 1, 3])
def test_paged_attention_decode_mask_replay(causal, window):
    geometry = (4, 2, 16, 4, 4, 1)
    proto = onnx.load_from_string(_model(*geometry, 0, 0))
    proto.graph.node[0].attribute.extend([
        onnx.helper.make_attribute("is_causal", int(causal)),
        onnx.helper.make_attribute("local_window_size", window),
    ])
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        proto.SerializeToString(), options, providers=["MLXExecutionProvider"]
    )
    session.disable_fallback()
    rng = np.random.default_rng(943)
    for nt in [3, 1, 2, 1]:
        feeds, _ = _inputs(*geometry, 0, 0, 943)
        for name, width in [("query", 64), ("key", 32), ("value", 32)]:
            feeds[name] = (rng.standard_normal((nt, width)) * 0.3).astype(np.float16)
        feeds["cumulative_sequence_length"] = np.array([0, nt], dtype=np.int32)
        feeds["past_seqlens"] = np.array([5], dtype=np.int32)
        expected = _paged_ref(
            *(feeds[name] for name in [
                "query", "key", "value", "key_cache", "value_cache",
                "cumulative_sequence_length", "past_seqlens", "block_table",
            ]),
            4, 2, 16, is_causal=causal, local_window_size=window,
        )
        actual = session.run(None, feeds)[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)


# --- multi-block gather (large past spanning several physical blocks) ----------------------------
@pytest.mark.parametrize("blk", [2, 8])
def test_paged_attention_multiblock(blk):
    _check(8, 2, 32, nb=8 * 2, blk=blk, batch=2, seed=blk + 7)


@pytest.mark.parametrize("source", ["runtime", "initializer", "intermediate"])
@pytest.mark.parametrize("eager", [False, True])
def test_paged_attention_metadata_sources(source, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    geometry = (4, 2, 16, 16, 4, 2)
    proto = onnx.load_from_string(_model(*geometry, 0, 0))
    metadata = ["cumulative_sequence_length", "past_seqlens", "block_table"]
    initial, _ = _inputs(*geometry, 0, 0, 87)
    if source == "initializer":
        for value in list(proto.graph.input):
            if value.name in metadata:
                proto.graph.input.remove(value)
                proto.graph.initializer.append(
                    onnx.numpy_helper.from_array(initial[value.name], value.name)
                )
    elif source == "intermediate":
        attention = proto.graph.node[0]
        prefix = []
        proto.graph.initializer.append(
            onnx.numpy_helper.from_array(np.array(0, dtype=np.int32), "zero")
        )
        for name in metadata:
            output = name + "_computed"
            prefix.append(onnx.helper.make_node("Add", [name, "zero"], [output]))
            for i, input_name in enumerate(attention.input):
                if input_name == name:
                    attention.input[i] = output
        nodes = [*prefix, *proto.graph.node]
        del proto.graph.node[:]
        proto.graph.node.extend(nodes)
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        proto.SerializeToString(), options, providers=["MLXExecutionProvider"]
    )
    session.disable_fallback()
    for seed in [87, 88, 89]:
        feeds, reference = _inputs(
            *geometry, 0, 0, 87 if source == "initializer" else seed
        )
        if source == "initializer":
            feeds = {name: value for name, value in feeds.items() if name not in metadata}
        actual = session.run(None, feeds)[0]
        np.testing.assert_allclose(actual, reference, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("outputs", ["omitted", "separate", "aliased"])
@pytest.mark.parametrize("remap", [False, True])
def test_paged_attention_partial_cache_writeback(outputs, remap):
    geometry = (4, 2, 16, 16, 4, 2)
    h, kv, d, nb, block, _ = geometry
    proto = onnx.load_from_string(_model(*geometry, 0, 0))
    if outputs != "omitted":
        for name in ["key_cache_out", "value_cache_out"]:
            proto.graph.node[0].output.append(name)
            proto.graph.output.append(
                onnx.helper.make_tensor_value_info(
                    name, onnx.TensorProto.FLOAT16, [nb, block, kv, d]
                )
            )
    if remap:
        proto.graph.node[0].input.extend(["", "", "slot_mapping"])
        proto.graph.input.append(
            onnx.helper.make_tensor_value_info(
                "slot_mapping", onnx.TensorProto.INT32, ["nt"]
            )
        )
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        proto.SerializeToString(), options, providers=["MLXExecutionProvider"]
    )
    session.disable_fallback()
    for seed in [87, 88, 89]:
        feeds, _ = _inputs(*geometry, 0, 0, seed)
        slots = None
        if remap:
            nt = feeds["query"].shape[0]
            slots = np.random.default_rng(seed).permutation(nb * block)[:nt].astype(np.int32)
            slots[0] = -1
            feeds["slot_mapping"] = slots
        expected = _paged_ref(
            *(feeds[name] for name in [
                "query", "key", "value", "key_cache", "value_cache",
                "cumulative_sequence_length", "past_seqlens", "block_table",
            ]),
            h, kv, d, slot_mapping=slots, return_cache=True,
        )
        binding = session.io_binding()
        for name, value in feeds.items():
            binding.bind_cpu_input(name, value)
        binding.bind_output("output", "cpu")
        for name in ["key_cache", "value_cache"] if outputs != "omitted" else []:
            if outputs == "aliased":
                value = feeds[name]
                binding.bind_output(
                    name + "_out", "cpu", element_type=np.float16,
                    shape=value.shape, buffer_ptr=value.ctypes.data,
                )
            else:
                binding.bind_output(name + "_out", "cpu")
        session.run_with_iobinding(binding)
        actual = binding.copy_outputs_to_cpu()
        np.testing.assert_allclose(actual[0], expected[0], rtol=2e-2, atol=2e-2)
        for index, name in enumerate(["key_cache", "value_cache"], 1):
            np.testing.assert_array_equal(feeds[name], expected[index])
            if outputs != "omitted":
                np.testing.assert_array_equal(actual[index], expected[index])


# --- RoPE (structured cache), non-interleaved + interleaved --------------------------------------
@pytest.mark.parametrize("interleaved", [0, 1], ids=["neox", "gptj"])
@pytest.mark.parametrize("H,KV,d", [(4, 2, 16), (2, 1, 64)], ids=["gqa-d16", "mqa-d64"])
def test_paged_attention_rotary(interleaved, H, KV, d):
    _check(H, KV, d, nb=8 * 2, blk=4, batch=2, do_rotary=1, interleaved=interleaved, seed=H + d + interleaved)


def test_paged_attention_latent_layout_and_rotary_offset():
    h, kv, d, vd, rot, nb, blk = 2, 1, 16, 8, 8, 3, 4
    nt, past_len, offset, scale = 2, 3, 8, 0.2
    rng = np.random.default_rng(291)
    q = (rng.standard_normal((nt, h * d)) * 0.2).astype(np.float16)
    k = (rng.standard_normal((nt, d)) * 0.2).astype(np.float16)
    kc = (rng.standard_normal((nb, blk, kv, d)) * 0.2).astype(np.float16)
    cum = np.array([0, nt], dtype=np.int32)
    past = np.array([past_len], dtype=np.int32)
    bt = np.array([[1, 0, 2]], dtype=np.int32)
    half = rot // 2
    inv = 1.0 / (10000.0 ** (np.arange(half) / half))
    angles = np.arange(nb * blk)[:, None] * inv[None, :]
    cos = np.cos(angles).astype(np.float16)
    sin = np.sin(angles).astype(np.float16)

    qh = q.reshape(nt, h, d).astype(np.float32)
    kh_new = k.reshape(nt, kv, d).astype(np.float32)
    positions = np.arange(past_len, past_len + nt)
    qh = _apply_rope_offset(qh, cos.astype(np.float32), sin.astype(np.float32), positions, 0, offset)
    kh_new = _apply_rope_offset(
        kh_new,
        cos.astype(np.float32),
        sin.astype(np.float32),
        positions,
        0,
        offset,
    )
    expected_cache = kc.copy()
    for i in range(nt):
        logical = past_len + i
        expected_cache[bt[0, logical // blk], logical % blk, 0] = kh_new[i, 0].astype(np.float16)
    keys = np.empty((past_len + nt, d), dtype=np.float32)
    for logical in range(past_len):
        keys[logical] = kc[bt[0, logical // blk], logical % blk, 0]
    keys[past_len:] = kh_new[:, 0]
    scores = np.einsum("thd,kd->htk", qh, keys) * scale
    cols = np.arange(past_len + nt)[None, :]
    rows = (past_len + np.arange(nt))[:, None]
    scores = np.where((cols > rows)[None], -np.inf, scores)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    expected = np.einsum("htk,kv->thv", probs, keys[:, :vd]).reshape(nt, h * vd)

    absent = lambda: m.tensor("", DT.FLOAT, [])
    inputs = [
        m.tensor("query", DT.FLOAT16, [nt, h * d]),
        m.tensor("key", DT.FLOAT16, [nt, d]),
        absent(),
        m.tensor("key_cache", DT.FLOAT16, [nb, blk, kv, d]),
        absent(),
        m.tensor("cum", DT.INT32, [2]),
        m.tensor("past", DT.INT32, [1]),
        m.tensor("block_table", DT.INT32, [1, 3]),
        m.tensor("cos", DT.FLOAT16, [nb * blk, half]),
        m.tensor("sin", DT.FLOAT16, [nb * blk, half]),
    ]
    outputs = [
        m.tensor("output", DT.FLOAT16, [nt, h * vd]),
        m.tensor("key_cache_out", DT.FLOAT16, [nb, blk, kv, d]),
    ]
    model = m.make_model(
        "PagedAttention",
        inputs,
        outputs,
        domain="com.microsoft",
        attributes={
            "num_heads": h,
            "kv_num_heads": kv,
            "v_head_size": vd,
            "kv_cache_layout": "LATENT",
            "do_rotary": 1,
            "rotary_offset": offset,
            "scale": scale,
        },
    )
    feeds = {
        "query": q,
        "key": k,
        "key_cache": kc,
        "cum": cum,
        "past": past,
        "block_table": bt,
        "cos": cos,
        "sin": sin,
    }
    m.assert_mlx_claims(model, feeds)
    actual, actual_cache = m.run_mlx(model, feeds)
    np.testing.assert_allclose(actual, expected.astype(np.float16), rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(actual_cache, expected_cache, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("d", [64, 96, 128, 192, 256])
@pytest.mark.parametrize("length,window", [(1, -1), (319, -1), (1023, 3), (1024, -1)])
@pytest.mark.parametrize("h,kv", [(6, 2), (24, 4), (64, 1)])
@pytest.mark.parametrize("tiny_queries", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nq", [1, 2, 5, 8, 9])
def test_paged_attention_vector_decode_fp32_reductions(
    d, length, window, h, kv, tiny_queries, causal, nq
):
    length = max(length, nq)
    nb, block = 5, 256
    geometry = (h, kv, d, nb, block, 1)
    proto = onnx.load_from_string(_model(*geometry, 0, 0))
    proto.graph.node[0].attribute.append(
        onnx.helper.make_attribute("local_window_size", window)
    )
    proto.graph.node[0].attribute.append(
        onnx.helper.make_attribute("is_causal", int(causal))
    )
    rng = np.random.default_rng(1329)
    feeds, _ = _inputs(*geometry, 0, 0, 1329)
    for name, width in [("query", h * d), ("key", kv * d), ("value", kv * d)]:
        scale = 2000 if name == "value" else 0.3
        if tiny_queries and name == "query":
            scale = 2**-18
        elif tiny_queries and name == "key":
            scale = 4096
        feeds[name] = (rng.standard_normal((nq, width)) * scale).astype(np.float16)
    feeds["value_cache"] *= np.float16(2000)
    if tiny_queries:
        feeds["key_cache"] *= np.float16(8192)
    feeds["cumulative_sequence_length"] = np.array([0, nq], dtype=np.int32)
    feeds["past_seqlens"] = np.array([length - nq], dtype=np.int32)
    reference = _paged_ref(
        feeds["query"], feeds["key"], feeds["value"],
        feeds["key_cache"], feeds["value_cache"],
        feeds["cumulative_sequence_length"], feeds["past_seqlens"],
        feeds["block_table"], h, kv, d, local_window_size=window, is_causal=causal,
    )
    actual = m.run_mlx(
        proto.SerializeToString(), {name: value.copy() for name, value in feeds.items()}
    )[0]
    np.testing.assert_allclose(actual, reference.astype(np.float16), rtol=2e-3, atol=0.5)

    keys = np.empty((length, kv, d), dtype=np.float32)
    values = np.empty_like(keys)
    for p in range(length - nq):
        physical = int(feeds["block_table"][0, p // block])
        keys[p] = feeds["key_cache"][physical, p % block]
        values[p] = feeds["value_cache"][physical, p % block]
    keys[-nq:] = feeds["key"].reshape(nq, kv, d)
    values[-nq:] = feeds["value"].reshape(nq, kv, d)
    shape_q, shape_kv = [1, h, nq, d], [1, h, length, d]
    rows = (length - nq + np.arange(nq))[:, None]
    cols = np.arange(length)[None, :]
    mask = np.zeros((nq, length), dtype=np.float32)
    if causal:
        mask[cols > rows] = -np.inf
    if window > 0:
        mask[cols < rows - window + 1] = -np.inf
    mask = mask[None, None]
    fp32_model = m.make_model(
        "Attention",
        [
            m.tensor("Q", DT.FLOAT, shape_q),
            m.tensor("K", DT.FLOAT, shape_kv),
            m.tensor("V", DT.FLOAT, shape_kv),
            m.tensor("mask", DT.FLOAT, [1, 1, nq, length]),
        ],
        [m.tensor("Y", DT.FLOAT, shape_q)],
        attributes={"scale": float(np.float32(1) / np.sqrt(np.float32(d)))},
        opset=23,
    )
    fp32_feeds = {
        "Q": feeds["query"].astype(np.float32).reshape(nq, h, d).transpose(1, 0, 2)[None],
        "K": np.repeat(keys, h // kv, axis=1).transpose(1, 0, 2)[None],
        "V": np.repeat(values, h // kv, axis=1).transpose(1, 0, 2)[None],
        "mask": mask,
    }
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        fp32_model, options, providers=["MLXExecutionProvider"]
    )
    session.disable_fallback()
    fp32_output = session.run(None, fp32_feeds)[0]
    fp32_output = fp32_output[0].transpose(1, 0, 2).reshape(nq, h * d).astype(np.float16)
    if d == 192 or nq > 8:
        # Unchanged FP32 fallbacks can select a different reduction layout.
        np.testing.assert_allclose(actual, fp32_output, rtol=2e-3, atol=2e-2)
    else:
        np.testing.assert_array_equal(actual, fp32_output)


def test_paged_attention_ort129_optional_inputs_and_cache_outputs():
    h, kv, d, nb, blk, batch = 4, 2, 8, 4, 4, 1
    hidden, kv_hidden = h * d, kv * d
    rng = np.random.default_rng(129)
    past = np.array([3], dtype=np.int32)
    cum = np.array([0, 2], dtype=np.int32)
    q = rng.standard_normal((2, hidden)).astype(np.float16)
    k = rng.standard_normal((2, kv_hidden)).astype(np.float16)
    v = rng.standard_normal((2, kv_hidden)).astype(np.float16)
    kc = rng.standard_normal((nb, blk, kv, d)).astype(np.float16)
    vc = rng.standard_normal((nb, blk, kv, d)).astype(np.float16)
    bt = np.array([[2, 0, 1, 3]], dtype=np.int32)
    slots = np.array([3, 9], dtype=np.int32)
    sink = np.array([-0.5, 0.25, 1.0, -1.0], dtype=np.float16)
    qw = np.linspace(0.75, 1.25, d).astype(np.float16)
    kw = np.linspace(1.25, 0.75, d).astype(np.float16)
    metadata = np.array([4, 16], dtype=np.int32)
    absent = lambda: m.tensor("", DT.FLOAT, [])
    inputs = [
        m.tensor("query", DT.FLOAT16, [2, hidden]),
        m.tensor("key", DT.FLOAT16, [2, kv_hidden]),
        m.tensor("value", DT.FLOAT16, [2, kv_hidden]),
        m.tensor("key_cache", DT.FLOAT16, [nb, blk, kv, d]),
        m.tensor("value_cache", DT.FLOAT16, [nb, blk, kv, d]),
        m.tensor("cum", DT.INT32, [2]),
        m.tensor("past", DT.INT32, [1]),
        m.tensor("block_table", DT.INT32, [1, 4]),
        absent(),
        absent(),
        m.tensor("slot_mapping", DT.INT32, [2]),
        m.tensor("head_sink", DT.FLOAT16, [h]),
        m.tensor("q_norm_weight", DT.FLOAT16, [d]),
        m.tensor("k_norm_weight", DT.FLOAT16, [d]),
        absent(),
        absent(),
        m.tensor("attention_metadata", DT.INT32, [2]),
    ]
    outputs = [
        m.tensor("output", DT.FLOAT16, [2, hidden]),
        m.tensor("key_cache_out", DT.FLOAT16, [nb, blk, kv, d]),
        m.tensor("value_cache_out", DT.FLOAT16, [nb, blk, kv, d]),
    ]
    model = m.make_model(
        "PagedAttention",
        inputs,
        outputs,
        domain="com.microsoft",
        attributes={"num_heads": h, "kv_num_heads": kv, "qk_norm_epsilon": 1e-5},
    )
    feeds = {
        "query": q,
        "key": k,
        "value": v,
        "key_cache": kc,
        "value_cache": vc,
        "cum": cum,
        "past": past,
        "block_table": bt,
        "slot_mapping": slots,
        "head_sink": sink,
        "q_norm_weight": qw,
        "k_norm_weight": kw,
        "attention_metadata": metadata,
    }
    expected = _paged_ref(
        q,
        k,
        v,
        kc,
        vc,
        cum,
        past,
        bt,
        h,
        kv,
        d,
        slot_mapping=slots,
        head_sink=sink.astype(np.float32),
        q_norm=qw.astype(np.float32),
        k_norm=kw.astype(np.float32),
        norm_eps=1e-5,
        return_cache=True,
    )
    m.assert_mlx_claims(model, feeds)
    actual = m.run_mlx(model, feeds)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got, want.astype(np.float16), rtol=2e-2, atol=2e-2)

    one_output_inputs = [
        m.tensor("query", DT.FLOAT16, [2, hidden]),
        m.tensor("key", DT.FLOAT16, [2, kv_hidden]),
        m.tensor("value", DT.FLOAT16, [2, kv_hidden]),
        m.tensor("key_cache", DT.FLOAT16, [nb, blk, kv, d]),
        m.tensor("value_cache", DT.FLOAT16, [nb, blk, kv, d]),
        m.tensor("cum", DT.INT32, [2]),
        m.tensor("past", DT.INT32, [1]),
        m.tensor("block_table", DT.INT32, [1, 4]),
        absent(),
        absent(),
        m.tensor("slot_mapping", DT.INT32, [2]),
        m.tensor("head_sink", DT.FLOAT16, [h]),
        m.tensor("q_norm_weight", DT.FLOAT16, [d]),
        m.tensor("k_norm_weight", DT.FLOAT16, [d]),
        absent(),
        absent(),
        m.tensor("attention_metadata", DT.INT32, [2]),
    ]
    one_output_model = m.make_model(
        "PagedAttention",
        one_output_inputs,
        [m.tensor("output", DT.FLOAT16, [2, hidden])],
        domain="com.microsoft",
        attributes={"num_heads": h, "kv_num_heads": kv, "qk_norm_epsilon": 1e-5},
    )
    one_output_feeds = {name: value.copy() for name, value in feeds.items()}
    m.assert_mlx_claims(
        one_output_model,
        {name: value.copy() for name, value in one_output_feeds.items()},
    )
    m.run_mlx(one_output_model, one_output_feeds)
    np.testing.assert_allclose(one_output_feeds["key_cache"], expected[1], rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(one_output_feeds["value_cache"], expected[2], rtol=2e-3, atol=2e-3)
