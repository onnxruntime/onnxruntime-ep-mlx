"""Packed hybrid SSM kernels against independent FP64 NumPy recurrences."""

from itertools import pairwise

import _models as m
import ml_dtypes
import numpy as np
import onnx
import onnx_ir as ir
import pytest
from test_ssm_contracts import strict_session
from test_ssm_quant import build


def storage_type(dtype):
    return {
        np.float32: ir.DataType.FLOAT,
        np.float16: ir.DataType.FLOAT16,
        ml_dtypes.bfloat16: ir.DataType.BFLOAT16,
    }[dtype]


@pytest.mark.parametrize("activation", ["silu", "swish", "sigmoid"])
def test_gated_rms_norm_dynamic_replay(activation):
    shape = ["batch", "tokens", 12]
    model = m.make_model(
        "GatedRMSNorm",
        [
            m.tensor("x", ir.DataType.FLOAT, shape),
            m.tensor("scale", ir.DataType.FLOAT, [4]),
            m.tensor("gate", ir.DataType.FLOAT, shape),
        ],
        [m.tensor("y", ir.DataType.FLOAT, shape)],
        domain="com.microsoft",
        attributes={"activation": activation},
    )
    session = strict_session(model)
    rng = np.random.default_rng(42)
    scale = rng.standard_normal(4).astype(np.float32)
    for batch, tokens in [(2, 3), (1, 1), (3, 5), (2, 3)]:
        x = rng.standard_normal((batch, tokens, 12)).astype(np.float32)
        gate = rng.standard_normal(x.shape).astype(np.float32)
        groups = x.astype(np.float64).reshape(-1, 4)
        normalized = groups / np.sqrt(np.mean(groups**2, axis=-1, keepdims=True) + 1e-5)
        sigmoid = 1 / (1 + np.exp(-gate.astype(np.float64)))
        activated = sigmoid if activation == "sigmoid" else gate * sigmoid
        expected = (normalized * scale).reshape(x.shape) * activated
        actual = session.run(None, {"x": x, "scale": scale, "gate": gate})[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("rank", [3, 4])
def test_mrotary_dynamic_replay(rank):
    shape = ["batch", "tokens", 24] if rank == 3 else ["batch", 2, "tokens", 12]
    model = m.make_model(
        "MRotaryEmbedding",
        [
            m.tensor("x", ir.DataType.FLOAT, shape),
            m.tensor("positions", ir.DataType.INT64, [3, "batch", "tokens"]),
            m.tensor("cos", ir.DataType.FLOAT, [17, 6]),
            m.tensor("sin", ir.DataType.FLOAT, [17, 6]),
        ],
        [m.tensor("y", ir.DataType.FLOAT, shape)],
        domain="com.microsoft",
        attributes={"num_heads": 2, "mrope_section": [2, 2, 2]},
    )
    session = strict_session(model)
    rng = np.random.default_rng(43)
    cos = rng.standard_normal((17, 6)).astype(np.float32)
    sin = rng.standard_normal((17, 6)).astype(np.float32)
    for batch, tokens in [(2, 3), (1, 1), (3, 5), (2, 3)]:
        x4 = rng.standard_normal((batch, 2, tokens, 12)).astype(np.float32)
        positions = rng.integers(0, 17, (3, batch, tokens), dtype=np.int64)
        selected = [
            np.concatenate(
                [cache[positions[i]][..., 2 * i : 2 * i + 2] for i in range(3)],
                axis=-1,
            )[:, None]
            for cache in (cos, sin)
        ]
        c, s = selected
        expected = np.concatenate(
            [x4[..., :6] * c - x4[..., 6:] * s, x4[..., 6:] * c + x4[..., :6] * s],
            axis=-1,
        )
        x = x4 if rank == 4 else x4.transpose(0, 2, 1, 3).reshape(batch, tokens, 24)
        if rank == 3:
            expected = expected.transpose(0, 2, 1, 3).reshape(x.shape)
        actual = session.run(
            None, {"x": x, "positions": positions, "cos": cos, "sin": sin}
        )[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def runtime_model(serialized, dtype):
    if dtype != ml_dtypes.bfloat16:
        return serialized
    proto = onnx.load_from_string(serialized)
    casts, tail = [], []
    for value in proto.graph.input:
        if value.type.tensor_type.elem_type != onnx.TensorProto.BFLOAT16:
            continue
        internal = value.name + "_bf16"
        for node in proto.graph.node:
            for i, name in enumerate(node.input):
                if name == value.name:
                    node.input[i] = internal
        value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        casts.append(
            onnx.helper.make_node(
                "Cast", [value.name], [internal], to=onnx.TensorProto.BFLOAT16
            )
        )
    for value in proto.graph.output:
        if value.type.tensor_type.elem_type != onnx.TensorProto.BFLOAT16:
            continue
        internal = value.name + "_bf16"
        for node in proto.graph.node:
            for i, name in enumerate(node.output):
                if name == value.name:
                    node.output[i] = internal
        value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        tail.append(
            onnx.helper.make_node(
                "Cast", [internal], [value.name], to=onnx.TensorProto.FLOAT
            )
        )
    nodes = [*casts, *proto.graph.node, *tail]
    del proto.graph.node[:]
    proto.graph.node.extend(nodes)
    return proto.SerializeToString()


def mrotary_reference(x, positions, cos, sin, sections, layout, interleaved, scale, rank):
    dtype = x.dtype
    half = cos.shape[1]
    assignment = np.zeros(half, dtype=np.int64)
    if layout == 0:
        start = 0
        for stream, width in enumerate(sections):
            assignment[start : start + width] = stream
            start += width
    else:
        for stream, width in enumerate(sections[1:], 1):
            assignment[stream : min(width * 3, half) : 3] = stream

    selected_positions = positions[assignment].transpose(1, 2, 0)
    columns = np.arange(half)
    selected = []
    for cache in (cos, sin):
        values = cache[selected_positions, columns]
        if scale != 1.0:
            values = (values.astype(np.float32) * np.float32(scale)).astype(dtype)
        selected.append(values[:, None])
    cos_rows, sin_rows = selected

    x4 = x if rank == 4 else x.reshape(x.shape[0], x.shape[1], 2, 12).transpose(0, 2, 1, 3)
    rot = 2 * half
    if interleaved:
        pairs = x4[..., :rot].reshape(*x4.shape[:-1], half, 2)
        first, second = pairs[..., 0], pairs[..., 1]
        rotated = np.stack(
            [first * cos_rows - second * sin_rows, first * sin_rows + second * cos_rows],
            axis=-1,
        ).reshape(*x4.shape[:-1], rot)
    else:
        first, second = x4[..., :half], x4[..., half:rot]
        rotated = np.concatenate(
            [first * cos_rows - second * sin_rows, first * sin_rows + second * cos_rows],
            axis=-1,
        )
    expected = np.concatenate([rotated, x4[..., rot:]], axis=-1)
    if rank == 3:
        expected = expected.transpose(0, 2, 1, 3).reshape(x.shape)
    return expected.astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float16, ml_dtypes.bfloat16])
@pytest.mark.parametrize("rank", [3, 4])
@pytest.mark.parametrize("layout,interleaved,rot", [(0, 0, 12), (1, 1, 8)])
@pytest.mark.parametrize("eager", [False, True])
def test_mrotary_column_gather_reference(dtype, rank, layout, interleaved, rot, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    shape = ["batch", "tokens", 24] if rank == 3 else ["batch", 2, "tokens", 12]
    half = rot // 2
    sections = [2, 1, half - 3]
    scale = 1.0003
    model = m.make_model(
        "MRotaryEmbedding",
        [
            m.tensor("x", storage_type(dtype), shape),
            m.tensor("positions", ir.DataType.INT64, [3, "batch", "tokens"]),
            m.tensor("cos", storage_type(dtype), [17, half]),
            m.tensor("sin", storage_type(dtype), [17, half]),
        ],
        [m.tensor("y", storage_type(dtype), shape)],
        domain="com.microsoft",
        attributes={
            "num_heads": 2, "mrope_section": sections,
            "mrope_layout": layout, "interleaved": interleaved,
            "rotary_embedding_dim": rot, "scale": scale,
        },
    )
    model = runtime_model(model, dtype)
    session = strict_session(model)
    rng = np.random.default_rng(413)
    for batch, tokens in [(2, 3), (1, 1), (1, 0), (3, 5), (2, 3)]:
        cos = rng.standard_normal((17, half)).astype(dtype)
        sin = rng.standard_normal((17, half)).astype(dtype)
        x4 = rng.standard_normal((batch, 2, tokens, 12)).astype(dtype)
        x = x4 if rank == 4 else x4.transpose(0, 2, 1, 3).reshape(batch, tokens, 24)
        positions = rng.integers(0, 17, (3, batch, tokens), dtype=np.int64)
        feeds = {
            "x": x, "positions": positions,
            "cos": cos, "sin": sin,
        }
        expected = mrotary_reference(
            x, positions, cos, sin, sections, layout, interleaved, scale, rank
        )
        if dtype == ml_dtypes.bfloat16:
            feeds = {name: value.astype(np.float32) if name != "positions" else value
                     for name, value in feeds.items()}
            expected = expected.astype(np.float32)
        actual = session.run(None, feeds)[0]
        tolerance = 2e-5 if dtype == np.float32 else 3e-3
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def runtime_feeds(feeds):
    return {
        name: array.astype(np.float32) if array.dtype == ml_dtypes.bfloat16 else array
        for name, array in feeds.items()
    }


def model(op, feeds, outputs, attrs, dtype):
    inputs = []
    for name in feeds:
        if not name:
            inputs.append(ir.Value(name=""))
            continue
        array = feeds[name]
        element_type = (
            ir.DataType.INT32
            if array.dtype == np.int32
            else ir.DataType.FLOAT
            if array.dtype == np.float32
            else dtype
        )
        shape = list(array.shape)
        if name in ("qkv", "input", "decay", "beta", "offsets", "state"):
            shape[0] = (
                "batch_plus_one"
                if name == "offsets"
                else "batch"
                if name == "state"
                else "tokens"
            )
        inputs.append(m.tensor(name, element_type, shape))
    return build(
        op,
        inputs,
        [
            m.tensor(name, element_type, list(shape))
            for name, element_type, shape in outputs
        ],
        attrs=[
            ir.Attr(
                k,
                ir.AttributeType.STRING
                if isinstance(v, str)
                else ir.AttributeType.FLOAT
                if isinstance(v, float)
                else ir.AttributeType.INT,
                v,
            )
            for k, v in attrs.items()
        ],
        domain="com.microsoft",
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float16, ml_dtypes.bfloat16])
@pytest.mark.parametrize("activation", ["none", "silu"])
@pytest.mark.parametrize("kernel,dilation", [(1, 1), (4, 1), (3, 2)])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("capacity", [0, 1, 4, 8])
def test_varlen_conv(dtype, activation, kernel, dilation, with_bias, capacity):
    rng = np.random.default_rng(42)
    channels, lengths = 19, [2, 5]
    carry = (kernel - 1) * dilation
    feeds = {
        "input": rng.normal(size=(sum(lengths), channels)).astype(dtype),
        "weight": rng.normal(size=(channels, 1, kernel)).astype(dtype),
        "offsets": np.array([0, 2, 7], np.int32),
        "bias": rng.normal(size=channels).astype(dtype),
        "state": rng.normal(size=(2, channels, carry)).astype(dtype),
    }
    dt = storage_type(dtype)
    outputs = [
        ("output", dt, ["tokens", channels]),
        ("final", dt, [2, channels, carry]),
    ]
    attrs = {"activation": activation, "dilation": dilation}
    if capacity:
        feeds["counts"] = np.array([2, 99], np.int32)
        outputs.append(("updates", dt, [2, capacity, channels]))
        attrs["state_update_capacity"] = capacity
    serialized = model(
        "VarlenCausalConvWithState",
        feeds,
        outputs,
        attrs,
        dt,
    )
    if not with_bias:
        proto = onnx.load_from_string(serialized)
        proto.graph.node[0].input[3] = ""
        del proto.graph.input[3]
        serialized = proto.SerializeToString()
        feeds.pop("bias")
    session = strict_session(runtime_model(serialized, dtype))
    initial_input = feeds["input"]
    for iteration, offsets in enumerate(
        ([0, 2, 7], [0, 0, 7], [0, 7, 7], [0, 6, 7], [0, 0, 0], [0, 1, 2])
    ):
        feeds["input"] = initial_input[: offsets[-1]]
        feeds["offsets"] = np.array(offsets, np.int32)
        if capacity:
            feeds["counts"] = np.array(
                [2, 99] if iteration % 2 == 0 else [-1, 3], np.int32
            )
        expected = np.empty_like(feeds["input"])
        state = np.empty_like(feeds["state"])
        for b, (start, end) in enumerate(pairwise(offsets)):
            history = np.concatenate(
                (feeds["state"][b].T, feeds["input"][start:end])
            ).astype(np.float64)
            for t in range(end - start):
                samples = history[t + np.arange(kernel) * dilation]
                y = np.sum(samples * feeds["weight"][:, 0].T, axis=0) + feeds.get(
                    "bias", 0
                )
                if activation == "silu":
                    y = y / (1 + np.exp(-y))
                expected[start + t] = y
            state[b] = history[-carry:].T if carry else history[:0].T
        actual = session.run(None, runtime_feeds(feeds))
        output, final = actual[:2]
        tolerance = 8e-3 if dtype == ml_dtypes.bfloat16 else 3e-3
        np.testing.assert_allclose(
            output, expected.astype(np.float32), rtol=tolerance, atol=tolerance
        )
        np.testing.assert_array_equal(final, state)
        if capacity:
            updates = np.zeros((2, capacity, channels), dtype=dtype)
            for b, (start, end) in enumerate(pairwise(offsets)):
                count = max(0, min(int(feeds["counts"][b]), capacity, end - start))
                updates[b, :count] = feeds["input"][start : start + count]
            np.testing.assert_array_equal(actual[2], updates)


def delta_reference(feeds, hq, h, k, v, rule, fused, norm, scale=0.0, capacity=0):
    packed = feeds["qkv"].astype(np.float64)
    query = packed[:, : hq * k].reshape(-1, hq, k)
    key = packed[:, hq * k : 2 * hq * k].reshape(-1, hq, k)
    value = packed[:, 2 * hq * k :].reshape(-1, h, v)
    if norm:
        query = query / np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + 1e-12)
        key = key / np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + 1e-12)
    query = np.repeat(query, h // hq, axis=1)
    key = np.repeat(key, h // hq, axis=1)
    output = np.empty((len(packed), h, v), np.float64)
    final = feeds["state"].astype(np.float64).copy()
    capsules = np.zeros((len(final), capacity * (h + hq * k + h * v)), np.float64)
    for b, (start, end) in enumerate(pairwise(feeds["offsets"])):
        for t in range(start, end):
            effective_decay = np.ones(h, np.float64)
            if "decay" in feeds and rule in ("gated", "gated_delta"):
                g = feeds["decay"][t].astype(np.float64)
                if fused:
                    g = -np.exp(feeds["a_log"].astype(np.float64)) * np.logaddexp(
                        0, g + feeds["dt_bias"]
                    )
                effective_decay = np.exp(g)
                final[b] *= effective_decay[:, None, None]
            delta = value[t].copy()
            if rule in ("delta", "gated_delta"):
                retrieved = np.einsum("hvk,hk->hv", final[b], key[t])
                delta -= retrieved
            if "beta" in feeds:
                beta = feeds["beta"][t].astype(np.float64)
                if fused:
                    beta = 1 / (1 + np.exp(-beta))
                delta *= beta[:, None]
            slot = t - start
            if (
                capacity
                and feeds.get("active", [1])[0]
                and slot < min(int(feeds["counts"][b]), capacity)
            ):
                capsules[b, : capacity * h].reshape(capacity, h)[slot] = effective_decay
                capsules[b, capacity * h : capacity * (h + hq * k)].reshape(
                    capacity, hq, k
                )[slot] = key[t, :: h // hq]
                capsules[b, capacity * (h + hq * k) :].reshape(capacity, h, v)[slot] = (
                    delta
                )
            final[b] += delta[:, :, None] * key[t, :, None, :]
            output[t] = np.einsum("hvk,hk->hv", final[b], query[t]) * (
                scale or 1 / np.sqrt(k)
            )
    return (output, final, capsules) if capacity else (output, final)


@pytest.mark.parametrize("dtype", [np.float32, np.float16, ml_dtypes.bfloat16])
@pytest.mark.parametrize(
    "geometry",
    [(2, 6, 8, 8), (2, 6, 16, 32), (16, 48, 128, 128), (2, 6, 256, 64)],
)
@pytest.mark.parametrize("capacity", [0, 4, 8])
def test_chunked_delta_dynamic(dtype, geometry, capacity, monkeypatch):
    hq, h, k, v = geometry
    dt = storage_type(dtype)
    inputs = [
        m.tensor("qkv", dt, ["tokens", 2 * hq * k + h * v]),
        ir.Value(name=""),
        ir.Value(name=""),
        m.tensor("offsets", ir.DataType.INT32, [3]),
        m.tensor("decay", ir.DataType.FLOAT, ["tokens", h]),
        m.tensor("beta", ir.DataType.FLOAT, ["tokens", h]),
        m.tensor("state", ir.DataType.FLOAT, [2, h, v, k]),
        m.tensor("a_log", ir.DataType.FLOAT, [h]),
        m.tensor("dt_bias", ir.DataType.FLOAT, [h]),
    ]
    outputs = [
        m.tensor("output", dt, ["tokens", h, v]),
        m.tensor("final", ir.DataType.FLOAT, [2, h, v, k]),
    ]
    scale = 0.37
    attrs = [
        ir.Attr("qk_l2_norm", ir.AttributeType.INT, 1),
        ir.Attr("gate_activation", ir.AttributeType.STRING, "softplus"),
        ir.Attr("beta_activation", ir.AttributeType.STRING, "sigmoid"),
        ir.Attr("scale", ir.AttributeType.FLOAT, scale),
    ]
    if capacity:
        inputs.extend(
            [
                m.tensor("counts", ir.DataType.INT32, [2]),
                m.tensor("active", ir.DataType.INT32, [1]),
            ]
        )
        attrs.append(ir.Attr("state_update_capacity", ir.AttributeType.INT, capacity))
        outputs.append(
            m.tensor("updates", ir.DataType.FLOAT, [2, capacity * (h + hq * k + h * v)])
        )
    serialized = runtime_model(
        build("GatedDeltaNet", inputs, outputs, attrs=attrs, domain="com.microsoft"),
        dtype,
    )
    baseline, candidate = strict_session(serialized), strict_session(serialized)
    rng = np.random.default_rng(814)
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_GDN_PREPARE", "0")
    for iteration, (tokens, split, mode) in enumerate(
        [
            (0, 0, "normal"),
            (1, 0, "normal"),
            (8, 3, "normal"),
            (31, 31, "normal"),
            (32, 1, "normal"),
            (63, 21, "normal"),
            (64, 0, "zero"),
            (65, 65, "weak"),
            (257, 0, "weak"),
            (64, 21, "mixed"),
            (129, 1, "collinear"),
            (4, 1, "normal"),
        ]
    ):
        feeds = {
            "qkv": (rng.normal(size=(tokens, 2 * hq * k + h * v)) * 0.1).astype(dtype),
            "offsets": np.array([0, split, tokens], np.int32),
            "decay": rng.normal(size=(tokens, h)).astype(np.float32),
            "beta": rng.normal(size=(tokens, h)).astype(np.float32),
            "state": (rng.normal(size=(2, h, v, k)) * 0.1).astype(np.float32),
            "a_log": np.full(
                h, -20 if mode == "weak" else 100 if mode == "zero" else -1, np.float32
            ),
            "dt_bias": np.full(h, -10 if mode == "weak" else 0, np.float32),
        }
        if mode == "mixed":
            feeds["a_log"].fill(0)
            feeds["decay"][::2] = 1000
            feeds["decay"][1::2] = -1000
        elif mode == "collinear":
            feeds["qkv"][:, : 2 * hq * k] = 0.1
            feeds["a_log"].fill(-20)
            feeds["beta"].fill(100)
        if capacity:
            feeds["counts"] = np.array([2, 99] if iteration % 2 else [-1, 99], np.int32)
            feeds["active"] = np.array([iteration % 3 != 0], np.int32)
        expected = delta_reference(
            feeds, hq, h, k, v, "gated_delta", True, True, scale, capacity
        )
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_CHUNKED_PREFILL", "0")
        control = baseline.run(None, runtime_feeds(feeds))
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_CHUNKED_PREFILL", "1")
        actual = candidate.run(None, runtime_feeds(feeds))
        if tokens == 65:
            with monkeypatch.context() as default_environment:
                default_environment.delenv(
                    "ONNXRUNTIME_EP_MLX_CHUNKED_PREFILL", raising=False
                )
                default = strict_session(serialized).run(None, runtime_feeds(feeds))
            for result, enabled in zip(default, actual, strict=True):
                np.testing.assert_array_equal(
                    result.view(np.uint8), enabled.view(np.uint8)
                )
        tolerance = 8e-3 if dtype == ml_dtypes.bfloat16 else 1e-3
        np.testing.assert_allclose(
            actual[0], expected[0], rtol=tolerance, atol=tolerance
        )
        for result, reference in zip(actual[1:], expected[1:], strict=True):
            np.testing.assert_allclose(result, reference, rtol=2e-5, atol=2e-6)
        if capacity and tokens == 257:
            capsule = actual[2][1]
            gates = capsule[: capacity * h].reshape(capacity, h)
            keys = capsule[capacity * h : capacity * (h + hq * k)].reshape(
                capacity, hq, k
            )
            deltas = capsule[capacity * (h + hq * k) :].reshape(capacity, h, v)
            replayed = feeds["state"][1].astype(np.float64)
            for prefix in range(capacity):
                replayed *= gates[prefix, :, None, None]
                replayed += (
                    deltas[prefix, :, :, None].astype(np.float64)
                    * np.repeat(keys[prefix], h // hq, axis=0)[:, None, :]
                )
                prefix_feeds = {
                    **feeds,
                    "qkv": feeds["qkv"][: prefix + 1],
                    "decay": feeds["decay"][: prefix + 1],
                    "beta": feeds["beta"][: prefix + 1],
                    "offsets": np.array([0, 0, prefix + 1], np.int32),
                }
                prefix_state = delta_reference(
                    prefix_feeds,
                    hq,
                    h,
                    k,
                    v,
                    "gated_delta",
                    True,
                    True,
                    scale,
                    capacity,
                )[1][1]
                np.testing.assert_allclose(replayed, prefix_state, rtol=2e-5, atol=2e-6)
        if tokens < 32:
            for result, reference in zip(actual, control, strict=True):
                np.testing.assert_array_equal(
                    result.view(np.uint8), reference.view(np.uint8)
                )
        if tokens == 257:
            state = feeds["state"]
            parts = []
            for start, end in [(0, 65), (65, tokens)]:
                segment = {
                    **feeds,
                    "qkv": feeds["qkv"][start:end],
                    "decay": feeds["decay"][start:end],
                    "beta": feeds["beta"][start:end],
                    "offsets": np.array([0, 0, end - start], np.int32),
                    "state": state,
                }
                result = candidate.run(None, runtime_feeds(segment))
                parts.append(result[0])
                state = result[1]
            np.testing.assert_allclose(
                np.concatenate(parts), actual[0], rtol=tolerance, atol=tolerance
            )
            np.testing.assert_allclose(state, actual[1], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float16, ml_dtypes.bfloat16])
@pytest.mark.parametrize("geometry", [(2, 6, 33, 33), (16, 48, 128, 128)])
@pytest.mark.parametrize("capacity", [0, 4])
def test_prepared_delta_bitwise_dynamic(dtype, geometry, capacity, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_CHUNKED_PREFILL", "0")
    hq, h, k, v = geometry
    dt = storage_type(dtype)
    inputs = [
        m.tensor("qkv", dt, ["tokens", 2 * hq * k + h * v]),
        ir.Value(name=""),
        ir.Value(name=""),
        m.tensor("offsets", ir.DataType.INT32, [3]),
        m.tensor("decay", ir.DataType.FLOAT, ["tokens", h]),
        m.tensor("beta", ir.DataType.FLOAT, ["tokens", h]),
        m.tensor("state", ir.DataType.FLOAT, [2, h, v, k]),
        m.tensor("a_log", ir.DataType.FLOAT, [h]),
        m.tensor("dt_bias", ir.DataType.FLOAT, [h]),
    ]
    outputs = [
        m.tensor("output", dt, ["tokens", h, v]),
        m.tensor("final", ir.DataType.FLOAT, [2, h, v, k]),
    ]
    attrs = [
        ir.Attr("qk_l2_norm", ir.AttributeType.INT, 1),
        ir.Attr("gate_activation", ir.AttributeType.STRING, "softplus"),
        ir.Attr("beta_activation", ir.AttributeType.STRING, "sigmoid"),
    ]
    if capacity:
        inputs.append(m.tensor("counts", ir.DataType.INT32, [2]))
        attrs.append(ir.Attr("state_update_capacity", ir.AttributeType.INT, capacity))
        outputs.append(
            m.tensor("updates", ir.DataType.FLOAT, [2, capacity * (h + hq * k + h * v)])
        )
    serialized = runtime_model(
        build("GatedDeltaNet", inputs, outputs, attrs=attrs, domain="com.microsoft"),
        dtype,
    )
    baseline, candidate = strict_session(serialized), strict_session(serialized)
    rng = np.random.default_rng(433)
    for tokens in (0, 1, 3, 4, 5, 8, 9, 64, 256, 4):
        feeds = {
            "qkv": (rng.normal(size=(tokens, 2 * hq * k + h * v)) * 0.1).astype(dtype),
            "offsets": np.array([0, tokens // 3, tokens], np.int32),
            "decay": rng.normal(size=(tokens, h)).astype(np.float32),
            "beta": rng.normal(size=(tokens, h)).astype(np.float32),
            "state": (rng.normal(size=(2, h, v, k)) * 0.1).astype(np.float32),
            "a_log": rng.normal(size=h).astype(np.float32),
            "dt_bias": rng.normal(size=h).astype(np.float32),
        }
        if capacity:
            feeds["counts"] = np.array([2, 99], np.int32)
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_GDN_PREPARE", "1")
        expected = baseline.run(None, runtime_feeds(feeds))
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_GDN_PREPARE", "0")
        actual = candidate.run(None, runtime_feeds(feeds))
        for result, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(
                result.view(np.uint8), reference.view(np.uint8)
            )


@pytest.mark.parametrize("dtype", [np.float32, np.float16, ml_dtypes.bfloat16])
@pytest.mark.parametrize("rule", ["linear", "gated", "delta", "gated_delta"])
@pytest.mark.parametrize("geometry", [(1, 1, 1, 3), (2, 6, 33, 5), (2, 6, 128, 128)])
@pytest.mark.parametrize("fused,norm", [(False, False), (True, True)])
@pytest.mark.parametrize("with_gates", [False, True])
@pytest.mark.parametrize("capacity,active", [(0, None), (1, None), (4, 1), (8, 0)])
def test_packed_delta(dtype, rule, geometry, fused, norm, with_gates, capacity, active):
    hq, h, k, v = geometry
    rng = np.random.default_rng(613)
    tokens = 9 if capacity else 5
    feeds = {
        "qkv": (rng.normal(size=(tokens, 2 * hq * k + h * v)) * 0.1).astype(dtype),
        "offsets": np.array([0, 1, tokens], np.int32),
        "decay": (-np.abs(rng.normal(size=(tokens, h)))).astype(np.float32),
        "beta": rng.uniform(0.1, 0.9, (tokens, h)).astype(np.float32),
        "state": (rng.normal(size=(2, h, v, k)) * 0.1).astype(np.float32),
        "a_log": rng.normal(size=h).astype(np.float32),
        "dt_bias": rng.normal(size=h).astype(np.float32),
    }
    dt = storage_type(dtype)
    scale = -0.25 if fused else 0.0
    # Two distinct absent operand slots must survive serialization.
    inputs = [
        m.tensor("qkv", dt, ["tokens", feeds["qkv"].shape[1]]),
        ir.Value(name=""),
        ir.Value(name=""),
        m.tensor("offsets", ir.DataType.INT32, [3]),
        m.tensor("decay", ir.DataType.FLOAT, ["tokens", h]),
        m.tensor("beta", ir.DataType.FLOAT, ["tokens", h]),
        m.tensor("state", ir.DataType.FLOAT, [2, h, v, k]),
        m.tensor("a_log", ir.DataType.FLOAT, [h]),
        m.tensor("dt_bias", ir.DataType.FLOAT, [h]),
    ]
    if not with_gates:
        for index in (4, 5, 7, 8):
            feeds.pop(inputs[index].name)
            inputs[index] = ir.Value(name="")
    attrs = [
        ir.Attr("update_rule", ir.AttributeType.STRING, rule),
        ir.Attr(
            "gate_activation", ir.AttributeType.STRING, "softplus" if fused else "none"
        ),
        ir.Attr(
            "beta_activation", ir.AttributeType.STRING, "sigmoid" if fused else "none"
        ),
        ir.Attr("qk_l2_norm", ir.AttributeType.INT, int(norm)),
        ir.Attr("scale", ir.AttributeType.FLOAT, scale),
    ]
    outputs = [
        m.tensor("output", dt, ["tokens", h, v]),
        m.tensor("final", ir.DataType.FLOAT, [2, h, v, k]),
    ]
    if capacity:
        feeds["counts"] = np.array([2, 99], np.int32)
        inputs.append(m.tensor("counts", ir.DataType.INT32, [2]))
        if active is not None:
            feeds["active"] = np.array([active], np.int32)
            inputs.append(m.tensor("active", ir.DataType.INT32, [1]))
        attrs.append(ir.Attr("state_update_capacity", ir.AttributeType.INT, capacity))
        outputs.append(
            m.tensor("updates", ir.DataType.FLOAT, [2, capacity * (h + hq * k + h * v)])
        )
    serialized = build(
        "GatedDeltaNet",
        inputs,
        outputs,
        attrs=attrs,
        domain="com.microsoft",
    )
    session = strict_session(runtime_model(serialized, dtype))
    uncaptured = None
    if capacity:
        proto = onnx.load_from_string(serialized)
        node = next(n for n in proto.graph.node if n.op_type == "GatedDeltaNet")
        next(a for a in node.attribute if a.name == "state_update_capacity").i = 0
        del node.input[9:]
        del node.output[2:]
        del proto.graph.output[2:]
        base_inputs = [
            v for v in proto.graph.input if v.name not in ("counts", "active")
        ]
        del proto.graph.input[:]
        proto.graph.input.extend(base_inputs)
        uncaptured = strict_session(runtime_model(proto.SerializeToString(), dtype))
    initial_tokens = {
        name: feeds[name] for name in ("qkv", "decay", "beta") if name in feeds
    }
    for iteration, offsets in enumerate(
        (
            [0, 1, tokens],
            [0, 0, tokens],
            [0, tokens, tokens],
            [0, 4, tokens],
            [0, 0, 0],
            [0, 1, 2],
        )
    ):
        for name, array in initial_tokens.items():
            feeds[name] = array[: offsets[-1]]
        feeds["offsets"] = np.array(offsets, np.int32)
        if capacity:
            feeds["counts"] = np.array(
                [2, 99] if iteration % 2 == 0 else [-1, 99], np.int32
            )
            if active is not None:
                feeds["active"] = np.array([(active + iteration) % 2], np.int32)
        expected = delta_reference(
            feeds, hq, h, k, v, rule, fused, norm, scale, capacity
        )
        actual = session.run(None, runtime_feeds(feeds))
        if uncaptured is not None:
            baseline = uncaptured.run(
                None,
                runtime_feeds(
                    {k: v for k, v in feeds.items() if k not in ("counts", "active")}
                ),
            )
            for result, reference in zip(actual[:2], baseline):
                np.testing.assert_array_equal(
                    result.view(np.uint8), reference.view(np.uint8)
                )
        for result, reference in zip(actual[:2], expected[:2]):
            tolerance = 8e-3 if dtype == ml_dtypes.bfloat16 else 3e-3
            np.testing.assert_allclose(result, reference, rtol=tolerance, atol=3e-5)
        for b, (start, end) in enumerate(pairwise(offsets)):
            if start == end:
                np.testing.assert_array_equal(actual[1][b], feeds["state"][b])
        if capacity:
            if not feeds.get("active", [1])[0]:
                np.testing.assert_array_equal(actual[2], np.zeros_like(actual[2]))
            else:
                for b, (start, end) in enumerate(pairwise(offsets)):
                    count = max(0, min(int(feeds["counts"][b]), capacity, end - start))
                    for lo, width in [
                        (0, h),
                        (capacity * h, hq * k),
                        (capacity * (h + hq * k), h * v),
                    ]:
                        np.testing.assert_allclose(
                            actual[2][b, lo : lo + count * width],
                            expected[2][b, lo : lo + count * width],
                            rtol=2e-3,
                            atol=3e-5,
                        )
                    decay = actual[2][b, : capacity * h].reshape(capacity, h)
                    key = actual[2][b, capacity * h : capacity * (h + hq * k)].reshape(
                        capacity, hq, k
                    )
                    delta = actual[2][b, capacity * (h + hq * k) :].reshape(
                        capacity, h, v
                    )
                    replayed = feeds["state"][b].copy()
                    replayed_fma = replayed.copy()
                    for accepted in range(1, count + 1):
                        i = accepted - 1
                        replayed *= decay[i, :, None, None]
                        replayed += (
                            delta[i, :, :, None]
                            * np.repeat(key[i], h // hq, axis=0)[:, None, :]
                        )
                        prefix = {
                            name: (
                                array[start : start + accepted]
                                if name in ("qkv", "decay", "beta")
                                else array
                            )
                            for name, array in feeds.items()
                        }
                        prefix["state"] = feeds["state"][b : b + 1]
                        prefix["offsets"] = np.array([0, accepted], np.int32)
                        prefix_state = delta_reference(
                            prefix, hq, h, k, v, rule, fused, norm, scale
                        )[1][0]
                        np.testing.assert_allclose(
                            replayed, prefix_state, rtol=3e-3, atol=3e-5
                        )
                        if (
                            dtype == np.float16
                            and fused
                            and with_gates
                            and rule == "gated_delta"
                            and k == v == 128
                        ):
                            # GenAI replays FP32 state with an explicitly fused multiply-add.
                            replayed_fma *= decay[i, :, None, None]
                            replayed_fma = (
                                replayed_fma.astype(np.float64)
                                + delta[i, :, :, None].astype(np.float64)
                                * np.repeat(key[i], h // hq, axis=0)[:, None, :].astype(
                                    np.float64
                                )
                            ).astype(np.float32)
                            prefix["state"] = feeds["state"]
                            prefix["offsets"] = np.array(
                                [0, accepted, accepted] if b == 0 else [0, 0, accepted],
                                np.int32,
                            )
                            gpu_prefix = uncaptured.run(
                                None,
                                runtime_feeds(
                                    {
                                        name: value
                                        for name, value in prefix.items()
                                        if name not in ("counts", "active")
                                    }
                                ),
                            )[1][b]
                            np.testing.assert_array_equal(
                                replayed_fma.view(np.uint32),
                                gpu_prefix.view(np.uint32),
                            )
        feeds["state"] = actual[1]
    if dtype == np.float16 and fused and rule == "gated_delta" and k == 128:
        expected = delta_reference(
            feeds, hq, h, k, v, rule, fused, norm, scale, capacity
        )
        binding = session.io_binding()
        for name, array in feeds.items():
            binding.bind_cpu_input(name, array)
        binding.bind_output("output", "cpu")
        binding.bind_output(
            "final",
            "cpu",
            0,
            np.float32,
            feeds["state"].shape,
            feeds["state"].ctypes.data,
        )
        if capacity:
            binding.bind_output("updates", "cpu")
        session.run_with_iobinding(binding)
        for result, reference in zip(binding.copy_outputs_to_cpu(), expected):
            np.testing.assert_allclose(result, reference, rtol=3e-3, atol=3e-5)
