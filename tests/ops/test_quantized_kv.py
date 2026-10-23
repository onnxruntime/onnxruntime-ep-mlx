"""Integer KV contracts pinned to ORT v1.31.0, commit 402979ac1bef89dcd55add41eaa7277c407599eb.

Independent byte equations: core/mlas/lib/qkv_quant.cpp (GQA reciprocal and zero->1),
contrib_ops/cuda/bert/paged_attention_impl.cu (Paged divide and zero->0).
Signed INT4 is +8 biased, even channel LOW nibble, odd channel HIGH; an unused
high nibble is 8. Paged has no CPU kernel; CPU GQA only serves the supported subset.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx_ir import DataType as DT

import _models as m
from test_attention_ext import _t, assert_mlx_claims, build_model
from test_composed_attention import _strict_session
from test_gqa_position_cache import _case, _rotate


def _encode(x, scale, bits, *, paged=False):
    x, scale = np.asarray(x, np.float32), np.asarray(scale, np.float32)
    if paged:
        scaled = np.divide(x, scale, out=np.zeros(np.broadcast_shapes(x.shape, scale.shape), np.float32),
                           where=scale != 0)
    else:
        inverse = np.divide(np.float32(1), scale, out=np.ones_like(scale), where=scale != 0)
        scaled = x * inverse
    limit = 2 ** (bits - 1)
    codes = np.clip(np.rint(scaled), -limit, limit - 1).astype(np.int8)
    if bits == 8:
        return codes
    biased = (codes.astype(np.int16) + 8).astype(np.uint8)
    if x.shape[-1] % 2:
        biased = np.concatenate([biased, np.full((*biased.shape[:-1], 1), 8, np.uint8)], -1)
    return biased[..., ::2] | (biased[..., 1::2] << 4)


def _decode(x, scale, bits, head):
    if bits == 4:
        values = np.stack([x & 15, x >> 4], -1).reshape(*x.shape[:-1], x.shape[-1] * 2)[..., :head]
        values = values.astype(np.float32) - 8
    else:
        values = x.astype(np.float32)
    return values * np.asarray(scale, np.float32)


def _scores(q, k, v, past, *, causal=True, window=-1, bias=None, sink=None, cap=0):
    heads, seq, head = q.shape
    k, v = (np.repeat(a, heads // a.shape[0], 0).astype(np.float32) for a in (k, v))
    logits = q.astype(np.float32) @ k.transpose(0, 2, 1) * np.float32(1 / np.sqrt(head))
    if cap:
        logits = np.float32(cap) * np.tanh(logits / np.float32(cap))
    if bias is not None:
        logits += bias.astype(np.float32)
    col = np.arange(k.shape[1])[None]
    row = past + np.arange(seq)[:, None]
    visible = np.ones((seq, k.shape[1]), bool)
    if causal:
        visible &= col <= row
    if window > 0:
        visible &= col >= row - window + 1
    before = np.where(visible[None], logits, 0)
    logits = np.where(visible[None], logits, -np.inf)
    if sink is not None:
        logits = np.concatenate([logits, np.broadcast_to(sink[:, None, None], (heads, seq, 1))], -1)
    e = np.exp(logits - logits.max(-1, keepdims=True))
    probs = (e / e.sum(-1, keepdims=True))[..., :k.shape[1]]
    return probs @ v, before, probs


def _scales(heads, head, mode, side=0):
    if mode == "PER_TENSOR":
        return np.array([0.125 if side == 0 else 0.25], np.float32)
    scale = np.resize(np.array([0.125, 0.25, 0.5, 1, 0.0625, 0.375, 2, 0], np.float32),
                      (heads, 1, head)).copy()
    return np.roll(scale, side, axis=-1)


def _gqa(bits=8, dtype=np.float32, *, batch=2, seq=2, modes=("PER_CHANNEL", "PER_CHANNEL"),
         packed=False, rotary=False, composed=False, mode=0, holes=False,
         sliding=False, readonly=False, causal=True, window=-1, cap=6):
    model, feeds = _case(dtype, batch=batch, seq=seq, new_seq=0 if readonly else seq, cap=cap,
                         rotary=rotary, positions=rotary, bias=composed, composed=composed,
                         mode=mode, sliding=sliding, window=window, causal=causal)
    model = onnx.load_model_from_string(model)
    node = model.graph.node[0]
    node.attribute.extend([onnx.helper.make_attribute("kv_cache_bit_width", bits)])
    for side, name in enumerate(("k", "v")):
        node.attribute.extend([onnx.helper.make_attribute(f"{name}_quant_type", modes[side])])
        feeds[f"{name}_scale"] = _scales(2, 32, modes[side], side)
        model.graph.input.append(onnx.helper.make_tensor_value_info(
            f"{name}_scale", onnx.TensorProto.FLOAT, list(feeds[f"{name}_scale"].shape)))
        scale = feeds[f"{name}_scale"].reshape(1, 2, 1, 32) if modes[side] == "PER_CHANNEL" else feeds[f"{name}_scale"]
        feeds[f"p{name}"] = _encode(feeds[f"p{name}"], scale, bits)
        for value in [*model.graph.input, *model.graph.output]:
            if value.name in (f"p{name}", f"present_{name}"):
                value.type.tensor_type.elem_type = onnx.TensorProto.INT8 if bits == 8 else onnx.TensorProto.UINT8
                value.type.tensor_type.shape.dim[-1].dim_value = 32 if bits == 8 else 16
                if value.name.startswith("present"):
                    value.type.tensor_type.shape.dim[2].dim_param = "capacity"
    node.input.extend([""] * (12 - len(node.input)))
    node.input.extend(["k_scale", "v_scale"])
    for value in model.graph.input:
        if value.name in ("k", "v"):
            value.type.tensor_type.shape.dim[1].dim_param = "new_sequence"
    if packed:
        feeds["q"] = np.concatenate([feeds["q"], feeds.pop("k"), feeds.pop("v")], -1)
        node.input[1:3] = ["", ""]
        for value in list(model.graph.input):
            if value.name in ("k", "v"):
                model.graph.input.remove(value)
            elif value.name == "q":
                value.type.tensor_type.shape.dim[2].dim_value = feeds["q"].shape[2]
    if holes:
        node.output[1:3] = ["", ""]
        del model.graph.output[1:3]
    return model.SerializeToString(), feeds


def _gqa_ref(feeds, bits, *, packed=False, causal=True, window=-1, sliding=False,
             mode=0, composed=False):
    batch, seq = feeds["q"].shape[:2]
    heads, kv, head = 4, 2, 32
    if packed:
        q, k, v = np.split(feeds["q"], [heads * head, (heads + kv) * head], -1)
    else:
        q, k, v = (feeds[n] for n in ("q", "k", "v"))
    capacity = feeds["pk"].shape[2]
    total = int(feeds["total"][0])
    new_seq = k.shape[1]
    grow = capacity if sliding else max(capacity, total)
    zero = 0 if bits == 8 else 0x88
    result = [np.empty((batch, seq, heads * head), q.dtype)]
    cache = [np.full((batch, kv, grow, 32 if bits == 8 else 16), zero, feeds["pk"].dtype)
             for _ in range(2)]
    if not sliding:
        for array, name in zip(cache, ("pk", "pv")):
            array[:, :, :capacity] = feeds[name]
    diagnostic = np.zeros((batch, heads, seq, total), q.dtype)
    def resident(length):
        gap = capacity - window + 1
        return length if length <= capacity else length - gap * ((length - capacity + gap - 1) // gap)
    lengths = feeds["lengths"] + 1
    staging = sliding and any(resident(int(n)) - seq < min(int(n) - seq, window - 1) for n in lengths)
    for b, length in enumerate(lengths):
        length = int(length)
        past = max(length - seq, 0)
        append_past = length if new_seq == 0 else past
        qh = q[b].reshape(seq, heads, head).transpose(1, 0, 2)
        kh = k[b].reshape(new_seq, kv, head).transpose(1, 0, 2)
        vh = v[b].reshape(new_seq, kv, head).transpose(1, 0, 2)
        if "cos" in feeds:
            positions = (int(feeds["pos"].flat[0]) + np.arange(seq) if total == seq
                         else feeds["pos"].reshape(-1)[b * seq:(b + 1) * seq])
            qh = _rotate(qh, feeds["cos"], feeds["sin"], positions, False)
            if new_seq:
                kh = _rotate(kh, feeds["cos"], feeds["sin"], positions, False)
        values = []
        if sliding:
            before, after = resident(past), resident(length)
            seed_offset = before - (after - seq)
            seed_start = 0 if staging else seed_offset
            row_offset = before - seed_start
            origin = past - before + seed_start
        else:
            row_offset, origin = past, 0
        for side, (name, new) in enumerate((("k", kh), ("v", vh))):
            scale = feeds[f"{name}_scale"]
            scale = scale.reshape(kv, 1, head) if scale.size > 1 else scale
            old = feeds[f"p{name}"][b]
            if sliding:
                encoded = _encode(new, scale, bits)
                work = np.concatenate([old[:, seed_start:before], encoded], 1)
                cache[side][b, :, :after] = work[:, seed_offset if staging else 0:]
                values.append(_decode(work, scale, bits, head))
            else:
                rows = length - append_past
                if rows:
                    cache[side][b, :, append_past:length] = _encode(new[:, :rows], scale, bits)
                values.append(_decode(cache[side][b, :, :length], scale, bits, head))
        bias = feeds.get("bias")
        if bias is not None:
            bias = bias[0 if bias.shape[0] == 1 else b, :, :, origin:origin + values[0].shape[1]]
        out, before_score, probs = _scores(
            qh, *values, row_offset, causal=causal, window=window, bias=bias,
            sink=feeds.get("sink"), cap=0.7 if composed else 0)
        result[0][b] = out.transpose(1, 0, 2).reshape(seq, heads * head)
        if mode:
            diagnostic[b, :, :, :length] = before_score if mode == 1 else probs
    return result + cache + ([diagnostic] if mode else [])


def _check(actual, expected):
    assert len(actual) == len(expected)
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert got.dtype == want.dtype, (index, got.dtype, want.dtype)
        if got.dtype in (np.int8, np.uint8):
            np.testing.assert_array_equal(got, want)
        else:
            np.testing.assert_allclose(got, want, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("modes", [("PER_TENSOR", "PER_TENSOR"), ("PER_CHANNEL", "PER_CHANNEL"),
                                  ("PER_CHANNEL", "PER_TENSOR")])
@pytest.mark.parametrize("eager", [False, True])
def test_gqa_integer_live_cache_scales(monkeypatch, bits, dtype, packed, modes, eager):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    model, feeds = _gqa(bits, dtype, packed=packed, modes=modes)
    session = _strict_session(model)
    for step in range(3):
        if step:
            feeds["lengths"] += 1 if step == 1 else -1
            feeds["total"] += 1 if step == 1 else -1
            for name in ("k_scale", "v_scale"):
                feeds[name] *= np.float32(1.25)
            feeds["pk"] ^= np.array(3, feeds["pk"].dtype)
            feeds["pv"] ^= np.array(5, feeds["pv"].dtype)
        expected = _gqa_ref(feeds, bits, packed=packed)
        assert_mlx_claims(model, {n: a.copy() for n, a in feeds.items()})
        _check(session.run(None, {n: a.copy() for n, a in feeds.items()}), expected)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("mode,holes", [(0, False), (1, True), (2, False)])
@pytest.mark.parametrize("causal,window", [(True, -1), (True, 2), (False, -1)])
def test_gqa_integer_rotary_score_contracts(bits, mode, holes, causal, window):
    model, feeds = _gqa(bits, rotary=True, composed=True, mode=mode, holes=holes,
                        causal=causal, window=window)
    session = _strict_session(model)
    for step in range(2):
        feeds["pos"][:] = 3 + step
        feeds["cos"] += np.float32(step * 0.125)
        feeds["bias"] += np.float32(step * 0.25)
        feeds["sink"] += np.float32(step * 0.5)
        expected = _gqa_ref(feeds, bits, composed=True, mode=mode, causal=causal, window=window)
        if holes:
            expected = [expected[0], *expected[3:]]
        _check(session.run(None, {n: a.copy() for n, a in feeds.items()}), expected)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("batch,seq,capacity,window,length", [
    (2, 1, 6, 3, 8), (1, 3, 6, 3, 9), (1, 8, 6, 3, 12), (1, 2, 6, 3, 4)])
@pytest.mark.parametrize("rotary", [False, True])
def test_gqa_integer_sliding_bytes(bits, batch, seq, capacity, window, length, rotary):
    model, feeds = _gqa(bits, batch=batch, seq=seq, cap=capacity, sliding=True, window=window,
                        rotary=rotary)
    feeds["lengths"][:] = length - 1
    if batch == 2:
        feeds["lengths"][1] -= 2
    feeds["total"][:] = length
    expected = _gqa_ref(feeds, bits, sliding=True, window=window)
    _check(_strict_session(model).run(None, feeds), expected)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("alias", [False, True])
def test_gqa_integer_readonly_append_alias_and_recovery(bits, alias):
    model, feeds = _gqa(bits, batch=1, seq=1)
    session = _strict_session(model)
    session.disable_fallback()
    original = {n: feeds[n].copy() for n in ("k", "v")}
    for new_seq, length in [(1, 3), (0, 3), (0, 2), (1, 4)]:
        for name in ("k", "v"):
            feeds[name] = original[name][:, :new_seq].copy()
        feeds["lengths"][:] = length - 1
        feeds["total"][:] = length
        expected = _gqa_ref(feeds, bits)
        binding = session.io_binding()
        for name, array in feeds.items():
            binding.bind_cpu_input(name, array)
        outputs = [np.empty_like(expected[0]), feeds["pk"] if alias else np.empty_like(expected[1]),
                   feeds["pv"] if alias else np.empty_like(expected[2])]
        for name, array in zip(("y", "present_k", "present_v"), outputs):
            binding.bind_output(name, "cpu", 0, array.dtype, array.shape, array.ctypes.data)
        session.run_with_iobinding(binding)
        _check(outputs, expected)
    bad = {**feeds, "lengths": np.array([6], np.int32)}
    with pytest.raises(Exception, match="sequence|length|capacity"):
        session.run(None, bad)
    _check(session.run(None, feeds), _gqa_ref(feeds, bits))


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_gqa_integer_ties_saturation_cpu_bytes(bits, dtype):
    model, feeds = _gqa(bits, dtype, batch=1, seq=1, modes=("PER_TENSOR", "PER_TENSOR"))
    values = np.resize(np.array([-1024, -128.5, -8.5, -7.5, -2.5, -1.5, -0.5, 0.5,
                                 1.5, 2.5, 6.5, 7.5, 126.5, 127.5, 1024], dtype), (1, 1, 64))
    session = _strict_session(model)
    for scale in (np.float32(1), np.float32(0), np.float32(0.3)):
        feeds["k_scale"][:] = scale
        feeds["v_scale"][:] = scale
        feeds["k"] = values.copy()
        feeds["v"] = -values.copy()
        expected = _gqa_ref(feeds, bits)
        actual = session.run(None, {n: a.copy() for n, a in feeds.items()})
        _check(actual, expected)
        cpu = m.run_cpu(model, {n: a.copy() for n, a in feeds.items()})
        np.testing.assert_allclose(actual[0], cpu[0], rtol=3e-3, atol=3e-3)
        for got, want in zip(actual[1:], cpu[1:]):
            np.testing.assert_array_equal(got[:, :, :3], want[:, :, :3])


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("first", [False, True])
def test_gqa_integer_growing_and_ragged_prompt(bits, first):
    model, feeds = _gqa(bits, batch=2, seq=3, cap=0 if first else 2)
    feeds["total"][:] = 3 if first else 5
    feeds["lengths"][:] = feeds["total"][0] - 1
    feeds["lengths"][1] -= 1
    expected = _gqa_ref(feeds, bits)
    _check(_strict_session(model).run(None, feeds), expected)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("paged", [False, True])
def test_integer_fp32_half_boundary_arithmetic(bits, paged):
    model, feeds = (_paged(bits, modes=("PER_TENSOR", "PER_TENSOR")) if paged else
                    _gqa(bits, batch=1, seq=1, modes=("PER_TENSOR", "PER_TENSOR")))
    session = _strict_session(model)
    feeds["q"][:] = 0
    for scale in map(np.float32, [0.3, 0.7, 1.1, -0.3]):
        centers = np.array([-128.5, -127.5, -8.5, -7.5, -2.5, -1.5, -0.5, 0.5,
                             1.5, 2.5, 6.5, 7.5, 126.5, 127.5], np.float32) * scale
        values = np.stack([np.nextafter(centers, -np.inf, dtype=np.float32), centers,
                           np.nextafter(centers, np.inf, dtype=np.float32)], -1).reshape(-1)
        for name, value in (("k_scale", scale), ("v_scale", scale)):
            feeds[name][:] = value
        for name in ("k", "v"):
            # Paged activation precision is fp16; GQA fp32 exercises reciprocal/division
            # disagreements at adjacent FP32 values around an exact rounding tie.
            feeds[name][:] = np.resize(values, feeds[name].shape)
        reference = _paged_ref(feeds, bits) if paged else _gqa_ref(feeds, bits)
        _check(session.run(None, {n: a.copy() for n, a in feeds.items()}), reference)


def _paged(bits=8, *, head=16, modes=("PER_CHANNEL", "PER_CHANNEL"), packed=False,
           slots=True, rotary=False, norm=False, causal=True, window=-1, composed=False):
    rng = np.random.default_rng(818)
    heads, kv, blocks, block, batch = 4, 2, 6, 4, 3
    lengths = np.array([2, 0, 1], np.int32)
    past = np.array([3, 1, 2], np.int32)
    cumulative = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    tokens = int(cumulative[-1])
    feeds = {name: rng.normal(0, 0.3, (tokens, count * head)).astype(np.float16)
             for name, count in (("q", heads), ("k", kv), ("v", kv))}
    feeds.update(cum=cumulative, past=past, table=rng.permutation(blocks).astype(np.int32).reshape(batch, 2))
    for side, name in enumerate(("k", "v")):
        feeds[f"{name}_scale"] = _scales(kv, head, modes[side], side)
        shape = (blocks, block, kv, head if bits == 8 else (head + 1) // 2)
        feeds[f"p{name}"] = (rng.integers(-128, 128, shape, dtype=np.int8) if bits == 8
                            else rng.integers(0, 256, shape, dtype=np.uint8))
    if slots:
        feeds["slots"] = np.array([feeds["table"][0, 0] * block + 3,
                                   feeds["table"][0, 1] * block, -1], np.int32)
    if rotary:
        feeds["cos"] = rng.uniform(0.5, 1.5, (16, head // 2)).astype(np.float16)
        feeds["sin"] = rng.uniform(-0.5, 0.5, (16, head // 2)).astype(np.float16)
    if norm:
        feeds["qw"] = rng.uniform(0.5, 1.5, head).astype(np.float16)
        feeds["kw"] = rng.uniform(0.5, 1.5, head).astype(np.float16)
    if composed:
        feeds["sink"] = np.linspace(-0.5, 0.5, heads).astype(np.float16)
    feeds["metadata"] = np.array([2, 5, 1], np.int32)
    order = ["q", "k", "v", "pk", "pv", "cum", "past", "table",
             "cos", "sin", "slots", "sink", "qw", "kw", "k_scale", "v_scale", "metadata"]
    if packed:
        feeds["q"] = np.concatenate([feeds["q"], feeds.pop("k"), feeds.pop("v")], -1)
    def tensor(name):
        a = feeds[name]
        dtype = {np.dtype(np.float16): DT.FLOAT16, np.dtype(np.float32): DT.FLOAT,
                 np.dtype(np.int8): DT.INT8, np.dtype(np.uint8): DT.UINT8,
                 np.dtype(np.int32): DT.INT32}[a.dtype]
        return _t(name, list(a.shape), dtype)
    attrs = dict(num_heads=heads, kv_num_heads=kv, k_quant_type=modes[0], v_quant_type=modes[1],
                 k_cache_dtype="int4" if bits == 4 else "int8",
                 v_cache_dtype="int4" if bits == 4 else "int8", do_rotary=int(rotary),
                 is_causal=int(causal), local_window_size=window, softcap=0.7 if composed else 0.0)
    model = build_model(
        "PagedAttention", [tensor(n) if n in feeds else None for n in order],
        [_t("y", [tokens, heads * head], DT.FLOAT16),
         _t("present_k", list(feeds["pk"].shape), DT.INT8 if bits == 8 else DT.UINT8),
         _t("present_v", list(feeds["pv"].shape), DT.INT8 if bits == 8 else DT.UINT8)],
        domain="com.microsoft", attributes=attrs)
    return model, feeds


def _paged_ref(feeds, bits, *, head=16, packed=False, causal=True, window=-1, composed=False):
    heads, kv, block = 4, 2, feeds["pk"].shape[1]
    if packed:
        q, k, v = np.split(feeds["q"], [heads * head, (heads + kv) * head], -1)
    else:
        q, k, v = (feeds[n] for n in ("q", "k", "v"))
    q, k, v = [a.reshape(-1, count, head).copy() for a, count in ((q, heads), (k, kv), (v, kv))]
    if "qw" in feeds:
        for a, name in ((q, "qw"), (k, "kw")):
            x = a.astype(np.float32)
            a[:] = (x / np.sqrt(np.mean(x * x, -1, keepdims=True) + np.float32(1e-6))
                    * feeds[name].astype(np.float32)).astype(a.dtype)
    if "cos" in feeds:
        for b, past in enumerate(feeds["past"]):
            start, stop = feeds["cum"][b:b + 2]
            for a in (q, k):
                a[start:stop] = _rotate(a[start:stop].transpose(1, 0, 2),
                                       feeds["cos"], feeds["sin"], past + np.arange(stop - start),
                                       False).transpose(1, 0, 2)
    caches = [feeds["pk"].copy(), feeds["pv"].copy()]
    for b, past in enumerate(feeds["past"]):
        start, stop = feeds["cum"][b:b + 2]
        for token in range(int(start), int(stop)):
            pos = int(past) + token - int(start)
            slot = (feeds["slots"][token] if "slots" in feeds else
                    feeds["table"][b, pos // block] * block + pos % block)
            if slot >= 0:
                for cache, name, new in zip(caches, ("k", "v"), (k, v)):
                    scale = feeds[f"{name}_scale"]
                    scale = scale.reshape(kv, head) if scale.size > 1 else scale
                    cache[slot // block, slot % block] = _encode(new[token], scale, bits, paged=True)
    values = []
    for cache, name in zip(caches, ("k", "v")):
        scale = feeds[f"{name}_scale"]
        scale = scale.reshape(1, 1, kv, head) if scale.size > 1 else scale
        values.append(_decode(cache, scale, bits, head))
    out = np.empty((len(q), heads * head), q.dtype)
    for b, past in enumerate(feeds["past"]):
        start, stop = feeds["cum"][b:b + 2]
        if start == stop:
            continue
        total = int(past + stop - start)
        positions = np.arange(total)
        rows = (feeds["table"][b, positions // block], positions % block)
        keys, vals = (a[rows].transpose(1, 0, 2) for a in values)
        result, _, _ = _scores(q[start:stop].transpose(1, 0, 2), keys, vals, int(past),
                               causal=causal, window=window, sink=feeds.get("sink"),
                               cap=0.7 if composed else 0)
        out[start:stop] = result.transpose(1, 0, 2).reshape(stop - start, heads * head)
    return [out, *caches]


@pytest.mark.parametrize("bits,head", [(8, 16), (4, 16), (4, 7)])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("modes", [("PER_TENSOR", "PER_TENSOR"), ("PER_CHANNEL", "PER_CHANNEL"),
                                  ("PER_TENSOR", "PER_CHANNEL")])
@pytest.mark.parametrize("slots", [False, True])
def test_paged_integer_live_physical_cache(bits, head, packed, modes, slots):
    model, feeds = _paged(bits, head=head, packed=packed, modes=modes, slots=slots)
    session = _strict_session(model)
    for step in range(2):
        if step:
            feeds["k_scale"] *= np.float32(1.125)
            feeds["v_scale"] *= np.float32(1.25)
            feeds["pk"] ^= np.array(3, feeds["pk"].dtype)
            feeds["pv"] ^= np.array(7, feeds["pv"].dtype)
            feeds["metadata"][:] = [1, 1, 0]
        expected = _paged_ref(feeds, bits, head=head, packed=packed)
        assert_mlx_claims(model, {n: a.copy() for n, a in feeds.items()})
        live = {n: a.copy() for n, a in feeds.items()}
        _check(session.run(None, live), expected)
        np.testing.assert_array_equal(live["pk"], expected[1])
        np.testing.assert_array_equal(live["pv"], expected[2])


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("causal,window", [(True, -1), (True, 2), (False, -1), (False, 2)])
def test_paged_integer_rotary_sink_and_local(bits, causal, window):
    model, feeds = _paged(bits, rotary=True, composed=True, causal=causal, window=window)
    expected = _paged_ref(feeds, bits, composed=True, causal=causal, window=window)
    _check(_strict_session(model).run(None, feeds), expected)


@pytest.mark.parametrize("bits", [8, 4])
def test_paged_integer_ties_saturation_zero_scale(bits):
    model, feeds = _paged(bits, modes=("PER_TENSOR", "PER_TENSOR"), slots=False)
    session = _strict_session(model)
    values = np.array([-1024, -128.5, -8.5, -7.5, -2.5, -1.5, -0.5, 0.5,
                       1.5, 2.5, 6.5, 7.5, 126.5, 127.5, 1024, 0], np.float16)
    for scale in (np.float32(1), np.float32(0), np.float32(0.3)):
        feeds["k"][:] = np.resize(values, feeds["k"].shape)
        feeds["v"][:] = -np.resize(values, feeds["v"].shape)
        feeds["k_scale"][:] = scale
        feeds["v_scale"][:] = scale
        expected = _paged_ref(feeds, bits)
        _check(session.run(None, {n: a.copy() for n, a in feeds.items()}), expected)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("holes", [False, True])
def test_paged_integer_alias_writeback_and_optional_outputs(bits, holes):
    model, feeds = _paged(bits)
    model = onnx.load_model_from_string(model)
    if holes:
        model.graph.node[0].output[1] = ""
        del model.graph.output[1]
    session = _strict_session(model.SerializeToString())
    session.disable_fallback()
    for step in range(2):
        feeds["past"][0] = 3 + step
        feeds["slots"][0] = feeds["table"][0, step] * 4 + 3
        expected = _paged_ref(feeds, bits)
        binding = session.io_binding()
        for name, array in feeds.items():
            binding.bind_cpu_input(name, array)
        output = np.empty_like(expected[0])
        bound = [("y", output), ("present_v", feeds["pv"])]
        if not holes:
            bound.append(("present_k", feeds["pk"]))
        for name, array in bound:
            binding.bind_output(name, "cpu", 0, array.dtype, array.shape, array.ctypes.data)
        session.run_with_iobinding(binding)
        _check([output, feeds["pk"], feeds["pv"]], expected)
    bad = {**feeds, "slots": np.array([-2, 0, 1], np.int32)}
    with pytest.raises(Exception, match="slot"):
        session.run(None, bad)
    expected = _paged_ref(feeds, bits)
    result = session.run(None, feeds)
    _check(result, [expected[0], expected[2]] if holes else expected)


@pytest.mark.parametrize("bits", [8, 4])
def test_paged_integer_qk_norm_before_quantization(bits):
    model, feeds = _paged(bits, norm=True, modes=("PER_TENSOR", "PER_TENSOR"))
    # Exact representable values keep the independent byte oracle away from half-step
    # boundaries; normalization must precede integer conversion, not follow dequantization.
    feeds["q"][:] = np.resize(np.array([0.25, -0.5, 0.75, 1], np.float16), feeds["q"].shape)
    feeds["k"][:] = np.resize(np.array([1, 0.5, -0.25, -0.75], np.float16), feeds["k"].shape)
    feeds["qw"][:] = 1
    feeds["kw"][:] = 1
    expected = _paged_ref(feeds, bits)
    _check(_strict_session(model).run(None, feeds), expected)


@pytest.mark.parametrize("bits", [8, 4])
def test_paged_integer_shared_page_write_before_any_read(bits):
    model, feeds = _paged(bits, modes=("PER_TENSOR", "PER_TENSOR"))
    feeds["cum"][:] = [0, 1, 1, 3]
    feeds["past"][:] = 0
    feeds["table"][:] = 0
    feeds["slots"][:] = [-1, 0, -1]
    feeds["pk"][:] = 0 if bits == 8 else 0x88
    feeds["pv"][:] = 0 if bits == 8 else 0x88
    feeds["q"][:] = 0
    feeds["k"][:] = 0
    feeds["v"][:] = 1
    feeds["k_scale"][:] = 1
    feeds["v_scale"][:] = 1
    expected = _paged_ref(feeds, bits)
    assert np.all(expected[0][0] == 1)
    _check(_strict_session(model).run(None, feeds), expected)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("paged", [False, True])
def test_integer_cache_bfloat16_preserves_fp32_scales(bits, paged):
    import ml_dtypes

    model, feeds = _paged(bits) if paged else _gqa(bits)
    model = onnx.load_model_from_string(model)
    node = model.graph.node[0]
    before, after = [], []
    for value in model.graph.input:
        if value.name in ("k_scale", "v_scale"):
            continue
        if value.type.tensor_type.elem_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16):
            value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            name = value.name + "_bf16"
            before.append(onnx.helper.make_node("Cast", [value.name], [name], to=onnx.TensorProto.BFLOAT16))
            node.input[:] = [name if n == value.name else n for n in node.input]
            feeds[value.name] = feeds[value.name].astype(ml_dtypes.bfloat16).astype(np.float32)
    for value in model.graph.output:
        if value.type.tensor_type.elem_type in (onnx.TensorProto.INT8, onnx.TensorProto.UINT8):
            continue
        name = value.name + "_bf16"
        internal = onnx.ValueInfoProto()
        internal.CopyFrom(value)
        internal.name = name
        internal.type.tensor_type.elem_type = onnx.TensorProto.BFLOAT16
        model.graph.value_info.append(internal)
        value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        node.output[:] = [name if n == value.name else n for n in node.output]
        after.append(onnx.helper.make_node("Cast", [name], [value.name], to=onnx.TensorProto.FLOAT))
    nodes = before + [node] + after
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    expected = _paged_ref(feeds, bits) if paged else _gqa_ref(feeds, bits)
    expected[0] = expected[0].astype(ml_dtypes.bfloat16).astype(np.float32)
    actual = _strict_session(model.SerializeToString()).run(None, feeds)
    _check(actual[1:], expected[1:])
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("paged", [False, True])
def test_integer_live_scale_shape_error_recovery(bits, paged):
    model, feeds = _paged(bits) if paged else _gqa(bits)
    model = onnx.load_model_from_string(model)
    value = next(v for v in model.graph.input if v.name == "k_scale")
    for index, dim in enumerate(value.type.tensor_type.shape.dim):
        dim.dim_param = f"scale_{index}"
    session = _strict_session(model.SerializeToString())
    session.disable_fallback()
    bad = {**feeds, "k_scale": feeds["k_scale"][..., :-1].copy()}
    with pytest.raises(Exception, match="scale"):
        session.run(None, bad)
    expected = _paged_ref(feeds, bits) if paged else _gqa_ref(feeds, bits)
    _check(session.run(None, feeds), expected)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("bad", ["mode", "missing_scale", "scale_dtype", "scale_shape",
                                "cache_dtype", "output_dtype", "packing"])
def test_integer_cache_invalid_static_contract(paged, bad):
    model, feeds = _paged(4) if paged else _gqa(4)
    model = onnx.load_model_from_string(model)
    node = model.graph.node[0]
    if bad in ("mode", "packing"):
        name = "k_quant_type" if bad == "mode" else "k_cache_dtype" if paged else "kv_cache_bit_width"
        attr = next(a for a in node.attribute if a.name == name)
        if bad == "mode":
            attr.s = b"NONE"
        elif paged:
            attr.s = b"float4e2m1"
        else:
            attr.i = 8
    elif bad == "missing_scale":
        node.input[14 if paged else 12] = ""
    else:
        name = "k_scale" if bad.startswith("scale_") else "pk" if bad == "cache_dtype" else "present_k"
        value = next(v for v in [*model.graph.input, *model.graph.output] if v.name == name)
        if bad == "scale_shape":
            value.type.tensor_type.shape.dim[-1].dim_value = 3
        else:
            value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
    with pytest.raises(Exception):
        _strict_session(model.SerializeToString())
