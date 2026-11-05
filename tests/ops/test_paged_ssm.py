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
def test_varlen_conv(dtype, activation, kernel, dilation, with_bias):
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
    serialized = model(
        "VarlenCausalConvWithState",
        feeds,
        [("output", dt, ["tokens", channels]), ("final", dt, [2, channels, carry])],
        {"activation": activation, "dilation": dilation},
        dt,
    )
    if not with_bias:
        proto = onnx.load_from_string(serialized)
        proto.graph.node[0].input[3] = ""
        del proto.graph.input[3]
        serialized = proto.SerializeToString()
        feeds.pop("bias")
    session = strict_session(runtime_model(serialized, dtype))
    for offsets in ([0, 2, 7], [0, 6, 7], [0, 1, 2]):
        feeds["input"] = feeds["input"][: offsets[-1]]
        feeds["offsets"] = np.array(offsets, np.int32)
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
        output, final = session.run(None, runtime_feeds(feeds))
        tolerance = 8e-3 if dtype == ml_dtypes.bfloat16 else 3e-3
        np.testing.assert_allclose(
            output, expected.astype(np.float32), rtol=tolerance, atol=tolerance
        )
        np.testing.assert_array_equal(final, state)


def delta_reference(feeds, hq, h, k, v, rule, fused, norm, scale=0.0):
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
    for b, (start, end) in enumerate(pairwise(feeds["offsets"])):
        for t in range(start, end):
            if "decay" in feeds and rule in ("gated", "gated_delta"):
                g = feeds["decay"][t].astype(np.float64)
                if fused:
                    g = -np.exp(feeds["a_log"].astype(np.float64)) * np.logaddexp(
                        0, g + feeds["dt_bias"]
                    )
                final[b] *= np.exp(g)[:, None, None]
            delta = value[t].copy()
            if rule in ("delta", "gated_delta"):
                retrieved = np.einsum("hvk,hk->hv", final[b], key[t])
                delta -= retrieved
            if "beta" in feeds:
                beta = feeds["beta"][t].astype(np.float64)
                if fused:
                    beta = 1 / (1 + np.exp(-beta))
                delta *= beta[:, None]
            final[b] += delta[:, :, None] * key[t, :, None, :]
            output[t] = np.einsum("hvk,hk->hv", final[b], query[t]) * (
                scale or 1 / np.sqrt(k)
            )
    return output, final


@pytest.mark.parametrize("dtype", [np.float32, np.float16, ml_dtypes.bfloat16])
@pytest.mark.parametrize("rule", ["linear", "gated", "delta", "gated_delta"])
@pytest.mark.parametrize("geometry", [(1, 1, 1, 3), (2, 6, 33, 5), (2, 6, 128, 128)])
@pytest.mark.parametrize("fused,norm", [(False, False), (True, True)])
@pytest.mark.parametrize("with_gates", [False, True])
def test_packed_delta(dtype, rule, geometry, fused, norm, with_gates):
    hq, h, k, v = geometry
    rng = np.random.default_rng(613)
    tokens = 5
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
    serialized = build(
        "GatedDeltaNet",
        inputs,
        [
            m.tensor("output", dt, ["tokens", h, v]),
            m.tensor("final", ir.DataType.FLOAT, [2, h, v, k]),
        ],
        attrs=attrs,
        domain="com.microsoft",
    )
    session = strict_session(runtime_model(serialized, dtype))
    for offsets in ([0, 1, tokens], [0, 4, tokens], [0, 1, 2]):
        for name in ("qkv", "decay", "beta"):
            if name in feeds:
                feeds[name] = feeds[name][: offsets[-1]]
        feeds["offsets"] = np.array(offsets, np.int32)
        expected = delta_reference(feeds, hq, h, k, v, rule, fused, norm, scale)
        actual = session.run(None, runtime_feeds(feeds))
        for result, reference in zip(actual, expected):
            tolerance = 8e-3 if dtype == ml_dtypes.bfloat16 else 3e-3
            np.testing.assert_allclose(result, reference, rtol=tolerance, atol=3e-5)
        feeds["state"] = actual[1]
    if dtype == np.float16 and fused and rule == "gated_delta" and k == 128:
        expected = delta_reference(feeds, hq, h, k, v, rule, fused, norm, scale)
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
        session.run_with_iobinding(binding)
        for result, reference in zip(binding.copy_outputs_to_cpu(), expected):
            np.testing.assert_allclose(result, reference, rtol=3e-3, atol=3e-5)
