"""GQA v1 contracts pinned to ORT v1.31.0 (402979ac1bef89dcd55add41eaa7277c407599eb).

CPU sources under onnxruntime/contrib_ops/cpu/bert:
group_query_attention.cc:428-486 (rotary IDs), 507-620 (absolute window bias);
group_query_attention_helper.h:282-305,373-375 (empty new KV);
gqa_attention_base.h:1420-1610 (read-only KV, masks, score ordering).
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx_ir import DataType as DT

import _models as m
from test_attention_ext import _t, assert_mlx_claims, build_model
from test_composed_attention import _bf16_boundary, _strict_session


def _case(dtype=np.float32, *, batch=1, seq=2, new_seq=None, cap=6, rotary=True,
          positions=True, bias=True, mode=0, causal=True, window=-1, sliding=False,
          composed=False, interleaved=False, partial=False, outputs=True):
    rng = np.random.default_rng(365)
    heads, kv, head = 4, 2, 32
    new_seq = seq if new_seq is None else new_seq
    dt = DT.FLOAT16 if dtype == np.float16 else DT.FLOAT
    feeds = dict(
        q=rng.normal(size=(batch, seq, heads * head)).astype(dtype),
        k=rng.normal(size=(batch, new_seq, kv * head)).astype(dtype),
        v=rng.normal(size=(batch, new_seq, kv * head)).astype(dtype),
        pk=rng.normal(size=(batch, kv, cap, head)).astype(dtype),
        pv=rng.normal(size=(batch, kv, cap, head)).astype(dtype),
        lengths=np.full(batch, seq + 1, np.int32),
        total=np.array([seq + 2], np.int32),
    )
    inputs = [_t(n, list(feeds[n].shape), dt) for n in ("q", "k", "v", "pk", "pv")]
    inputs += [_t("lengths", [batch], DT.INT32), _t("total", [1], DT.INT32)]
    inputs += [None] * 5
    if rotary:
        half = 8 if partial else 16
        feeds.update(cos=rng.uniform(0.5, 1.5, (64, half)).astype(dtype),
                     sin=rng.uniform(-0.5, 0.5, (64, half)).astype(dtype))
        inputs[7:9] = [_t(n, list(feeds[n].shape), dt) for n in ("cos", "sin")]
    if positions:
        feeds["pos"] = np.tile(np.arange(seq, dtype=np.int64) + 5, (batch, 1))
        inputs[9] = _t("pos", [batch, "position_width"], DT.INT64)
    if bias:
        feeds["bias"] = rng.uniform(-2, 2, (batch, heads, seq, seq + 2)).astype(dtype)
        inputs[10] = _t("bias", ["bias_batch", "bias_heads", seq, "absolute_total"], dt)
    if composed:
        feeds["sink"] = np.array([-1, 0, 1, 2], dtype)
        inputs[11] = _t("sink", [heads], dt)
    out = [_t("y", list(feeds["q"].shape), dt),
           _t("present_k", [batch, kv, cap, head], dt) if outputs else None,
           _t("present_v", [batch, kv, cap, head], dt) if outputs else None]
    if mode:
        out.append(_t("qk", [batch, heads, seq, "absolute_total"], dt))
    attrs = dict(num_heads=heads, kv_num_heads=kv, do_rotary=int(rotary),
                 rotary_interleaved=int(interleaved), causal=int(causal),
                 local_window_size=window, sliding_window_cache=int(sliding),
                 qk_output=mode, softcap=0.7 if composed else 0.0,
                 smooth_softmax=1 if composed else -1)
    model = build_model("GroupQueryAttention", inputs, out, domain="com.microsoft", attributes=attrs)
    if not outputs and mode:
        model = onnx.load_model_from_string(model)
        model.graph.node[0].output[:] = ["y", "", "", "qk"]
        model = model.SerializeToString()
    return model, feeds


def _rotate(x, cos, sin, positions, interleaved):
    half = cos.shape[1]
    part = x[..., :2 * half].astype(np.float32)
    c, s = cos[positions].astype(np.float32), sin[positions].astype(np.float32)
    if interleaved:
        rotated = np.empty_like(part)
        rotated[..., 0::2], rotated[..., 1::2] = -part[..., 1::2], part[..., 0::2]
        c, s = np.repeat(c, 2, -1), np.repeat(s, 2, -1)
    else:
        rotated = np.concatenate([-part[..., half:], part[..., :half]], -1)
        c, s = np.tile(c, (1, 2)), np.tile(s, (1, 2))
    out = x.copy()
    out[..., :2 * half] = (part * c[None] + rotated * s[None]).astype(x.dtype)
    return out


def _reference(feeds, *, causal=True, window=-1, interleaved=False, composed=False, mode=0):
    batch, seq, hidden = feeds["q"].shape
    heads, kv, head = 4, 2, 32
    total = int(feeds["total"][0])
    pk, pv = feeds["pk"].copy(), feeds["pv"].copy()
    output = np.empty_like(feeds["q"])
    diagnostic = np.zeros((batch, heads, seq, total), feeds["q"].dtype)
    new_seq = feeds["k"].shape[1]
    for b, length in enumerate(feeds["lengths"] + 1):
        length = int(length)
        past = max(length - seq, 0)
        q = feeds["q"][b].reshape(seq, heads, head).transpose(1, 0, 2)
        k = feeds["k"][b].reshape(new_seq, kv, head).transpose(1, 0, 2)
        if "cos" in feeds:
            if "pos" in feeds:
                ids = (int(feeds["pos"].flat[0]) + np.arange(seq) if total == seq else
                       feeds["pos"].reshape(-1)[b * seq:(b + 1) * seq])
            else:
                ids = past + np.arange(seq)
            q = _rotate(q, feeds["cos"], feeds["sin"], ids, interleaved)
            if new_seq:
                k = _rotate(k, feeds["cos"], feeds["sin"], ids, interleaved)
        if new_seq:
            rows = length - past
            pk[b, :, past:length] = k[:, :rows]
            pv[b, :, past:length] = feeds["v"][b, :rows].reshape(rows, kv, head).transpose(1, 0, 2)
        keys = np.repeat(pk[b, :, :length], heads // kv, 0).astype(np.float32)
        vals = np.repeat(pv[b, :, :length], heads // kv, 0).astype(np.float32)
        scores = q.astype(np.float32) @ keys.transpose(0, 2, 1) / np.sqrt(head)
        if composed:
            scores = 0.7 * np.tanh(scores / 0.7)
        if "bias" in feeds:
            bias = feeds["bias"]
            scores += bias[0 if bias.shape[0] == 1 else b, :, :, :length].astype(np.float32)
        end = np.minimum(past + np.arange(seq) + 1, length) if causal else np.full(seq, length)
        start = np.maximum(0, end - window) if window > 0 else np.zeros(seq, int)
        columns = np.arange(length)[None]
        visible = (columns >= start[:, None]) & (columns < end[:, None])
        logits = np.where(visible[None], scores, -np.inf)
        if composed:
            logits = np.concatenate(
                [logits, np.broadcast_to(feeds["sink"][:, None, None], (heads, seq, 1))], -1)
        e = np.exp(logits - logits.max(-1, keepdims=True))
        probs = (e / e.sum(-1, keepdims=True))[..., :length]
        diagnostic[b, :, :, :length] = np.where(visible[None], scores, 0) if mode == 1 else probs
        output[b] = (probs @ vals).transpose(1, 0, 2).reshape(seq, hidden)
    return [output, pk, pv] + ([diagnostic] if mode else [])


def _check(actual, expected, dtype):
    for got, want in zip(actual, expected):
        assert got.dtype == dtype
        np.testing.assert_allclose(got, want, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("interleaved,partial", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("first,batch,seq", [(True, 2, 3), (False, 1, 3), (False, 2, 1)])
@pytest.mark.parametrize("packed", [False, True])
def test_gqa_custom_rotary_ids_live(dtype, interleaved, partial, first, batch, seq, packed):
    model, feeds = _case(dtype, batch=batch, seq=seq, interleaved=interleaved, partial=partial)
    if packed:
        model = onnx.load_model_from_string(model)
        model.graph.node[0].input[1:3] = ["", ""]
        for inp in list(model.graph.input):
            if inp.name in ("k", "v"):
                model.graph.input.remove(inp)
            elif inp.name == "q":
                inp.type.tensor_type.shape.dim[2].dim_value = 256
        model = model.SerializeToString()
    session = _strict_session(model)
    session.disable_fallback()
    for step in range(2):
        total = seq if first else seq + 2
        feeds["lengths"][:] = total - 1
        feeds["total"][:] = total
        feeds["bias"] = feeds["bias"][..., :total].copy()
        # First-prompt format deliberately ignores all IDs except pos[0].
        feeds["pos"] = np.full((batch, seq + 1), 2**40 if first else 9 - step, np.int64)
        feeds["pos"].flat[0] = 2 + step
        if not first:
            feeds["pos"].reshape(-1)[1:batch * seq:2] = 4 + step
        feeds["cos"] += dtype(step * 0.125)
        feeds["bias"] += dtype(step * 0.25)
        expected = _reference(feeds, interleaved=interleaved)
        live = {k: v.copy() for k, v in feeds.items()}
        if packed:
            live["q"] = np.concatenate([live["q"], live.pop("k"), live.pop("v")], -1)
        assert_mlx_claims(model, {k: v.copy() for k, v in live.items()})
        actual = session.run(None, {k: v.copy() for k, v in live.items()})
        cpu = m.run_cpu(model, {k: v.copy() for k, v in live.items()})
        _check(actual, expected, dtype)
        np.testing.assert_allclose(actual[0], cpu[0], rtol=3e-3, atol=3e-3)
        for b, length in enumerate(feeds["lengths"] + 1):
            for got, want in zip(actual[1:], cpu[1:]):
                np.testing.assert_allclose(got[b, :, :length], want[b, :, :length], rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("bad", [-1, 64, 2**32, 2**63 - 1])
def test_gqa_custom_rotary_bounds_every_call(first, bad):
    model, feeds = _case(seq=2, bias=False)
    if first:
        feeds["total"][:] = 2
        feeds["lengths"][:] = 1
    session = _strict_session(model)
    session.disable_fallback()
    session.run(None, feeds)
    feeds["pos"][0, 0 if first else 1] = bad
    with pytest.raises(Exception, match="position_ids exceed"):
        session.run(None, feeds)
    feeds["pos"][:] = 5
    session.run(None, feeds)


def test_gqa_first_prompt_rotary_offset_span():
    model, feeds = _case(seq=3, bias=False)
    feeds["total"][:] = 3
    feeds["lengths"][:] = 2
    feeds["pos"][0, 0] = 62
    session = _strict_session(model)
    session.disable_fallback()
    with pytest.raises(Exception, match="position_ids exceed"):
        session.run(None, feeds)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("batch,seq,total", [(1, 2, 4), (2, 1, 4), (2, 3, 3)])
@pytest.mark.parametrize("causal,window", [(True, -1), (True, 2), (False, -1)])
@pytest.mark.parametrize("mode,composed", [(0, False), (1, True), (2, True)])
def test_gqa_empty_new_kv_read_only(dtype, batch, seq, total, causal, window, mode, composed):
    model, feeds = _case(dtype, batch=batch, seq=seq, new_seq=0, mode=mode,
                         causal=causal, window=window, composed=composed)
    session = _strict_session(model)
    for step in range(2):
        length = total - step
        feeds["total"][:] = length
        feeds["lengths"][:] = length - 1
        if batch == 2 and seq == 1:
            feeds["lengths"][1] -= 1
        # Keep first-prompt batching valid for the CPU oracle.
        if batch == 2 and seq > 1:
            feeds["total"][:] = total
            feeds["lengths"][:] = total - 1
        feeds["bias"] = np.full((batch, 4, seq, int(feeds["total"][0])), -0.2 + step, dtype)
        feeds["pos"][:] = 6 - step
        feeds["pv"] += dtype(step * 0.125)
        original = {k: v.copy() for k, v in feeds.items()}
        expected = _reference(feeds, causal=causal, window=window, composed=composed, mode=mode)
        assert_mlx_claims(model, {k: v.copy() for k, v in feeds.items()})
        actual = session.run(None, feeds)
        cpu = m.run_cpu(model, {k: v.copy() for k, v in original.items()})
        _check(actual, expected, dtype)
        np.testing.assert_array_equal(actual[1], original["pk"])
        np.testing.assert_array_equal(actual[2], original["pv"])
        np.testing.assert_array_equal(feeds["pk"], original["pk"])
        np.testing.assert_array_equal(feeds["pv"], original["pv"])
        np.testing.assert_allclose(actual[0], cpu[0], rtol=3e-3, atol=3e-3)
        if mode:
            for b, length in enumerate(feeds["lengths"] + 1):
                np.testing.assert_allclose(actual[3][b, :, :, :length], cpu[3][b, :, :, :length],
                                           rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("rotary", [False, True])
def test_gqa_empty_new_kv_optional_outputs(mode, rotary):
    model, feeds = _case(seq=1, new_seq=0, outputs=False, mode=mode, composed=True,
                         rotary=rotary, positions=rotary)
    expected = _reference(feeds, composed=True, mode=mode)
    actual = _strict_session(model).run(None, feeds)
    _check(actual, [expected[0]] + expected[3:], np.float32)


@pytest.mark.parametrize("empty", [False, True])
def test_gqa_position_cache_bfloat16(empty):
    import ml_dtypes

    model, feeds = _case(seq=2, new_seq=0 if empty else 2, mode=2, composed=True)
    model = _bf16_boundary(model)
    internal = {name: value.astype(ml_dtypes.bfloat16) if value.dtype == np.float32 else value
                for name, value in feeds.items()}
    feeds = {name: value.astype(np.float32) if value.dtype == ml_dtypes.bfloat16 else value
             for name, value in internal.items()}
    expected = _reference(internal, composed=True, mode=2)
    actual = _strict_session(model).run(None, feeds)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want.astype(np.float32), rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("aligned", [False, True])
def test_gqa_empty_new_kv_writeback_and_dynamic_sequence(monkeypatch, alias, eager, aligned):
    import ctypes

    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    model, feeds = _case(seq=1, rotary=False, positions=False, bias=False)
    model = onnx.load_model_from_string(model)
    for inp in model.graph.input:
        if inp.name in ("k", "v"):
            inp.type.tensor_type.shape.dim[1].dim_param = "new_kv"
    model = model.SerializeToString()
    session = _strict_session(model)
    if aligned:
        for name in ("pk", "pv"):
            original = feeds[name]
            storage = ctypes.create_string_buffer(original.nbytes + 16384)
            offset = (-ctypes.addressof(storage)) % 16384
            feeds[name] = np.ndarray(original.shape, original.dtype, buffer=storage, offset=offset)
            feeds[name][:] = original
            assert feeds[name].ctypes.data % 16384 == 0
    original_k, original_v = feeds["k"].copy(), feeds["v"].copy()
    for new_seq, length in [(1, 3), (1, 4), (0, 4), (0, 3), (1, 4)]:
        feeds["k"] = original_k[:, :new_seq].copy()
        feeds["v"] = original_v[:, :new_seq].copy()
        feeds["lengths"][:] = length - 1
        feeds["total"][:] = length
        feeds["pv"] += np.float32(0.125)
        expected = _reference(feeds)
        binding = session.io_binding()
        for name, value in feeds.items():
            binding.bind_cpu_input(name, value)
        output = np.empty_like(feeds["q"])
        key = feeds["pk"] if alias else np.full_like(feeds["pk"], -19)
        value = feeds["pv"] if alias else np.full_like(feeds["pv"], -23)
        for name, array in [("y", output), ("present_k", key), ("present_v", value)]:
            binding.bind_output(name, "cpu", 0, array.dtype, array.shape, array.ctypes.data)
        session.run_with_iobinding(binding)
        _check([output, key, value], expected, np.float32)


def test_gqa_empty_new_kv_overlapping_writeback():
    import ctypes

    model, feeds = _case(seq=1, new_seq=0, cap=1024, rotary=False, positions=False, bias=False)
    session = _strict_session(model)
    outputs = []
    for name in ("pk", "pv"):
        original = feeds[name]
        storage = ctypes.create_string_buffer(original.nbytes + 32768)
        offset = (-ctypes.addressof(storage)) % 16384
        feeds[name] = np.ndarray(original.shape, original.dtype, buffer=storage, offset=offset)
        feeds[name][:] = original
        outputs.append(np.ndarray(original.shape, original.dtype, buffer=storage, offset=offset + 16384))
        assert np.shares_memory(feeds[name], outputs[-1])
    expected = _reference(feeds)
    binding = session.io_binding()
    for name, value in feeds.items():
        binding.bind_cpu_input(name, value)
    result = np.empty_like(feeds["q"])
    for name, value in zip(("y", "present_k", "present_v"), [result, *outputs]):
        binding.bind_output(name, "cpu", 0, value.dtype, value.shape, value.ctypes.data)
    session.run_with_iobinding(binding)
    _check([result, *outputs], expected, np.float32)


@pytest.mark.parametrize("invalid", ["without_past", "unequal_positive", "sliding_empty", "ids_type", "ids_batch"])
def test_gqa_new_forms_invalid_contract(invalid):
    model, feeds = _case(seq=2, new_seq=0, bias=False)
    model = onnx.load_model_from_string(model)
    node = model.graph.node[0]
    if invalid == "without_past":
        node.input[3:5] = ["", ""]
        for name in ("pk", "pv"):
            model.graph.input.remove(next(v for v in model.graph.input if v.name == name))
    elif invalid == "unequal_positive":
        for name in ("k", "v"):
            inp = next(v for v in model.graph.input if v.name == name)
            inp.type.tensor_type.shape.dim[1].dim_value = 1
    elif invalid == "sliding_empty":
        for attr in node.attribute:
            if attr.name == "sliding_window_cache":
                attr.i = 1
            elif attr.name == "local_window_size":
                attr.i = 2
    else:
        inp = next(v for v in model.graph.input if v.name == "pos")
        if invalid == "ids_type":
            inp.type.tensor_type.elem_type = onnx.TensorProto.INT32
        else:
            inp.type.tensor_type.shape.dim[0].dim_value = 2
    with pytest.raises(Exception):
        _strict_session(model.SerializeToString())


@pytest.mark.parametrize("invalid", ["length", "bias", "ids_width"])
def test_gqa_new_forms_runtime_validation(invalid):
    model, feeds = _case(seq=1, new_seq=0)
    session = _strict_session(model)
    session.disable_fallback()
    session.run(None, feeds)
    if invalid == "length":
        feeds["lengths"][:] = 6
        feeds["total"][:] = 7
    elif invalid == "bias":
        feeds["bias"] = feeds["bias"][..., :-1].copy()
    else:
        feeds["pos"] = np.empty((1, 0), np.int64)
    with pytest.raises(Exception, match="GroupQueryAttention"):
        session.run(None, feeds)


@pytest.mark.parametrize("sliding", [False, True])
def test_gqa_rejects_short_nonpadding_lengths(sliding):
    model, feeds = _case(batch=2 if sliding else 1, seq=3, new_seq=3 if sliding else 0,
                         sliding=sliding, window=2 if sliding else -1)
    if sliding:
        feeds["total"][:] = 3
        feeds["lengths"][:] = 2
        feeds["bias"] = feeds["bias"][..., :3].copy()
    session = _strict_session(model)
    session.disable_fallback()
    actual = session.run(None, feeds)
    cpu = m.run_cpu(model, {name: value.copy() for name, value in feeds.items()})
    np.testing.assert_allclose(actual[0], cpu[0], rtol=3e-3, atol=3e-3)
    if sliding:
        feeds["lengths"][1] = 1
    else:
        feeds["total"][:] = 2
        feeds["lengths"][:] = 1
        feeds["bias"] = feeds["bias"][..., :2].copy()
    with pytest.raises(Exception, match="too small for sequence_length"):
        m.run_cpu(model, {name: value.copy() for name, value in feeds.items()})
    with pytest.raises(Exception, match="invalid.*sequence lengths"):
        session.run(None, feeds)


def _resident(total, cap, window):
    gap = cap - window + 1
    return total if total <= cap else total - gap * ((total - cap + gap - 1) // gap)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("batch,seq,cap,window", [(1, 1, 4, 2), (2, 1, 4, 3),
                                               (1, 3, 4, 2), (1, 5, 3, 2)])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("composed", [False, True])
def test_gqa_sliding_bias_absolute_columns(dtype, batch, seq, cap, window, broadcast, composed):
    model, feeds = _case(dtype, batch=batch, seq=seq, cap=cap, window=window,
                         sliding=True, composed=composed)
    session = _strict_session(model)
    rng = np.random.default_rng(5412)
    all_k = rng.normal(size=(batch, 2, 32, 32)).astype(dtype)
    all_v = rng.normal(size=all_k.shape).astype(dtype)
    # Fresh resident inputs explicitly rematerialize rollbacks that cross an eviction boundary.
    for step, total in enumerate((max(seq, cap), cap + seq, cap + seq + 1, max(seq, cap))):
        lengths = np.full(batch, total, int)
        if batch == 2:
            lengths[1] -= 1
        feeds["total"][:] = total
        feeds["lengths"][:] = lengths - 1
        bias = rng.uniform(-3, 3, (1 if broadcast else batch, 1 if broadcast else 4, seq, total))
        feeds["bias"] = bias.astype(dtype)
        expected = np.empty_like(feeds["q"])
        expected_k, expected_v = np.zeros_like(feeds["pk"]), np.zeros_like(feeds["pv"])
        for b, length in enumerate(lengths):
            past = length - seq
            before, after = _resident(past, cap, window), _resident(length, cap, window)
            feeds["pk"][b].fill(0)
            feeds["pv"][b].fill(0)
            feeds["pk"][b, :, :before] = all_k[b, :, past - before:past]
            feeds["pv"][b, :, :before] = all_v[b, :, past - before:past]
            feeds["k"][b] = all_k[b, :, past:length].transpose(1, 0, 2).reshape(seq, 64)
            feeds["v"][b] = all_v[b, :, past:length].transpose(1, 0, 2).reshape(seq, 64)
            feeds["pos"][b] = np.arange(seq) + 3 + step
            ids = (int(feeds["pos"].flat[0]) + np.arange(seq) if total == seq else feeds["pos"][b])
            q = feeds["q"][b].reshape(seq, 4, 32).transpose(1, 0, 2)
            q = _rotate(q, feeds["cos"], feeds["sin"], ids, False).astype(np.float32)
            rotated_k = _rotate(all_k[b, :, past:length], feeds["cos"], feeds["sin"], ids, False)
            keys = all_k[b, :, :length].copy()
            keys[:, past:] = rotated_k
            expected_k[b, :, :after] = keys[:, length - after:length]
            expected_v[b, :, :after] = all_v[b, :, length - after:length]
            keys = np.repeat(keys, 2, 0).astype(np.float32)
            vals = np.repeat(all_v[b, :, :length], 2, 0).astype(np.float32)
            scores = q @ keys.transpose(0, 2, 1) / np.sqrt(32)
            if composed:
                scores = 0.7 * np.tanh(scores / 0.7)
            scores += bias[0 if broadcast else b, :, :, :length].astype(dtype).astype(np.float32)
            columns = np.arange(length)[None]
            end = past + np.arange(seq) + 1
            visible = (columns < end[:, None]) & (columns >= np.maximum(0, end - window)[:, None])
            scores = np.where(visible[None], scores, -np.inf)
            if composed:
                scores = np.concatenate([scores, np.broadcast_to(feeds["sink"][:, None, None], (4, seq, 1))], -1)
            e = np.exp(scores - scores.max(-1, keepdims=True))
            probs = (e / e.sum(-1, keepdims=True))[..., :length]
            expected[b] = (probs @ vals).transpose(1, 0, 2).reshape(seq, 128)
        original = {k: v.copy() for k, v in feeds.items()}
        assert_mlx_claims(model, {k: v.copy() for k, v in feeds.items()})
        actual = session.run(None, feeds)
        cpu = m.run_cpu(model, {k: v.copy() for k, v in original.items()})
        _check(actual, [expected, expected_k, expected_v], dtype)
        np.testing.assert_allclose(actual[0], cpu[0], rtol=3e-3, atol=3e-3)
        for b, length in enumerate(lengths):
            resident = _resident(length, cap, window)
            for got, want in zip(actual[1:], cpu[1:]):
                np.testing.assert_allclose(got[b, :, :resident], want[b, :, :resident],
                                           rtol=3e-3, atol=3e-3)
