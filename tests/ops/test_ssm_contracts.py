"""SSM schema contracts, independent recurrences, and live-input replay."""

import numpy as np
import ml_dtypes
import onnx_ir as ir
import onnxruntime as ort
import pytest

import _models as m
from test_ssm_quant import build, initz, _linear_attention_model

DT = ir.DataType


def strict_session(model):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"])
    session.disable_fallback()
    return session


def bf16_boundaries(op, input_shapes, output_shapes, *, domain, attrs=(), native_inputs=()):
    inputs, operands, nodes = [], [], []
    for name, shape in input_shapes:
        if not name:
            operands.append(ir.Value(name=""))
            continue
        dtype = DT.INT64 if name == "indices" else DT.FLOAT
        external = m.tensor(name, dtype, shape)
        inputs.append(external)
        if name in native_inputs:
            operands.append(external)
        else:
            internal = m.tensor(name + "_bf", DT.BFLOAT16, shape)
            nodes.append(ir.node("Cast", [external], attributes={"to": int(DT.BFLOAT16)}, outputs=[internal]))
            operands.append(internal)
    results = [m.tensor(name + "_bf", DT.BFLOAT16, shape) for name, shape in output_shapes]
    nodes.append(ir.node(op, operands, domain=domain, attributes={a.name: a for a in attrs}, outputs=results))
    outputs = []
    for result, (name, shape) in zip(results, output_shapes, strict=True):
        output = m.tensor(name, DT.FLOAT, shape)
        nodes.append(ir.node("Cast", [result], attributes={"to": int(DT.FLOAT)}, outputs=[output]))
        outputs.append(output)
    imports = {"": 24, **({domain: 1} if domain else {})}
    graph = ir.Graph(inputs, outputs, nodes=nodes, opset_imports=imports, name="bf16_ssm")
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


@pytest.mark.parametrize("eager", [False, True])
def test_ssm_bfloat16_native_boundaries(monkeypatch, eager):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    rng = np.random.default_rng(2405)

    def bf(array):
        return array.astype(ml_dtypes.bfloat16).astype(np.float32)

    model = bf16_boundaries(
        "LinearAttentionGate",
        [("a", [2, 3, 4]), ("bias", [4]), ("scale", [4])],
        [("decay", [2, 3, 4])],
        domain="com.microsoft", native_inputs=("bias", "scale"),
    )
    feeds = {
        "a": bf(rng.normal(size=(2, 3, 4))),
        "bias": rng.normal(size=4).astype(np.float32),
        "scale": -rng.uniform(size=4).astype(np.float32),
    }
    expected = bf(np.logaddexp(np.float32(0), feeds["a"] + feeds["bias"]) * feeds["scale"])
    np.testing.assert_array_equal(strict_session(model).run(None, feeds)[0], expected)

    model = bf16_boundaries(
        "TensorScatter",
        [("past", [2, 2, 8, 3]), ("update", [2, 2, 3, 3]), ("indices", [2])],
        [("present", [2, 2, 8, 3])],
        domain="", attrs=[ir.AttrString("mode", "circular")], native_inputs=("indices",),
    )
    session = strict_session(model)
    for indices in [[7, 9], [2, -1]]:
        past = bf(rng.normal(size=(2, 2, 8, 3)))
        update = bf(rng.normal(size=(2, 2, 3, 3)))
        expected = past.copy()
        for batch in range(2):
            for token in range(3):
                expected[batch, :, (indices[batch] + token) % 8] = update[batch, :, token]
        feeds = {"past": past, "update": update, "indices": np.array(indices, np.int64)}
        np.testing.assert_array_equal(session.run(None, feeds)[0], expected)

    shape, state = [1, 4, 2, 2], [1, 4, 2, 2]
    model = bf16_boundaries(
        "CausalConvWithState",
        [("x", shape), ("weight", [4, 1, 3]), ("", []), ("past", state)],
        [("y", shape), ("state", state)],
        domain="com.microsoft",
        attrs=[ir.AttrInt64("channels_last", 1), ir.AttrInt64("dilation", 2)],
    )
    feeds = {
        "x": np.arange(16, dtype=np.float32).reshape(shape) / 8,
        "weight": np.ones((4, 1, 3), np.float32) / 4,
        "past": np.ones(state, np.float32) / 8,
    }
    padded = np.concatenate([feeds["past"], feeds["x"]], axis=1)
    expected = bf((padded[:, :4] + padded[:, 2:6] + padded[:, 4:8]) / 4)
    y, present = strict_session(model).run(None, feeds)
    np.testing.assert_array_equal(y, expected)
    np.testing.assert_array_equal(present, feeds["x"])

    model = bf16_boundaries(
        "LinearAttention",
        [
            ("query", [1, 4, 2]), ("key", [1, 4, 2]), ("value", [1, 4, 2]),
            ("past", [1, 1, 2, 2]), ("decay", [1, 4, 1]), ("beta", [1, 4, 1]),
        ],
        [("y", [1, 4, 2]), ("state", [1, 1, 2, 2])],
        domain="com.microsoft",
        attrs=[ir.AttrInt64("q_num_heads", 1), ir.AttrInt64("kv_num_heads", 1)],
    )
    feeds = {
        "query": bf(rng.normal(size=(1, 4, 2)) * 0.25),
        "key": bf(rng.normal(size=(1, 4, 2)) * 0.25),
        "value": bf(rng.normal(size=(1, 4, 2)) * 0.25),
        "past": bf(rng.normal(size=(1, 1, 2, 2)) * 0.125),
        "decay": np.zeros((1, 4, 1), np.float32),
        "beta": np.full((1, 4, 1), 0.5, np.float32),
    }
    state = feeds["past"].copy()
    query = bf(feeds["query"] * bf(np.array(1 / np.sqrt(2), np.float32)))
    expected = []
    for token in range(4):
        key = feeds["key"][:, token].reshape(1, 1, 1, 2)
        retrieval = bf(key @ state).reshape(1, 1, 2)
        delta = bf(bf(feeds["value"][:, token, None] - retrieval) * 0.5)
        outer = bf(key.swapaxes(-1, -2) @ delta[:, :, None])
        state = bf(state + outer)
        q = query[:, token].reshape(1, 1, 1, 2)
        expected.append(bf(q @ state).reshape(1, 1, 2))
    y, present = strict_session(model).run(None, feeds)
    np.testing.assert_array_equal(y, np.concatenate(expected, axis=1))
    np.testing.assert_array_equal(present, state)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("with_beta", [False, True])
@pytest.mark.parametrize("heads", [0, 4])
def test_gate_three_inputs_and_dynamic_replay(monkeypatch, eager, with_beta, heads):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    shape = ["batch", "time", heads]
    inputs = [
        m.tensor("a", DT.FLOAT, shape),
        m.tensor("bias", DT.FLOAT, [heads]),
        m.tensor("scale", DT.FLOAT, [heads]),
    ]
    outputs = [m.tensor("decay", DT.FLOAT, shape)]
    if with_beta:
        inputs.append(m.tensor("b", DT.FLOAT, shape))
        outputs.append(m.tensor("beta", DT.FLOAT, shape))
    session = strict_session(build("LinearAttentionGate", inputs, outputs, domain="com.microsoft"))
    rng = np.random.default_rng(2401)
    for batch, time in [(2, 3), (2, 3), (1, 1), (1, 0), (0, 2)]:
        feeds = {
            "a": rng.normal(size=(batch, time, heads)).astype(np.float32),
            "bias": rng.normal(size=heads).astype(np.float32),
            "scale": -rng.uniform(size=heads).astype(np.float32),
        }
        expected = [np.logaddexp(np.float32(0), feeds["a"] + feeds["bias"]) * feeds["scale"]]
        if with_beta:
            feeds["b"] = rng.normal(size=(batch, time, heads)).astype(np.float32)
            expected.append(1 / (1 + np.exp(-feeds["b"])))
        for actual, reference in zip(session.run(None, feeds), expected, strict=True):
            np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-7)


def causal_model(*, channels_last, dilation, window, with_bias, kernel=3, dtype=DT.FLOAT):
    channels = [2, 2] if channels_last else [4]
    shape = ["batch", "time", *channels] if channels_last else ["batch", 4, "time"]
    carry = (kernel - 1) * dilation
    state = ["batch", carry, *channels] if channels_last else ["batch", 4, carry]
    if window:
        state = [window, *state]
    inputs = [m.tensor("x", dtype, shape), m.tensor("weight", dtype, [4, 1, kernel])]
    inputs.append(m.tensor("bias", dtype, [4]) if with_bias else ir.Value(name=""))
    inputs.append(m.tensor("past", dtype, state))
    return build(
        "CausalConvWithState",
        inputs,
        [m.tensor("y", dtype, shape), m.tensor("state", dtype, state)],
        domain="com.microsoft",
        attrs=[
            ir.AttrInt64("channels_last", int(channels_last)),
            ir.AttrInt64("dilation", dilation),
            ir.AttrInt64("state_window", window),
            ir.AttrString("activation", "silu"),
        ],
    )


def causal_reference(feeds, *, channels_last, dilation, window):
    x, past, weight = feeds["x"], feeds["past"], feeds["weight"][:, 0]
    if window:
        original_past = past
        past = past[-1]
    if channels_last:
        trailing = x.shape[2:]
        x = x.reshape(x.shape[0], x.shape[1], 4).transpose(0, 2, 1)
        past = past.reshape(past.shape[0], past.shape[1], 4).transpose(0, 2, 1)
    carry = (weight.shape[1] - 1) * dilation
    padded = np.concatenate([past, x], axis=2)
    y = np.zeros_like(x, dtype=np.float64)
    for token in range(x.shape[2]):
        for tap in range(weight.shape[1]):
            y[:, :, token] += padded[:, :, token + tap * dilation].astype(np.float64) * weight[:, tap].astype(np.float64)
    if x.dtype == np.float16:
        y = y.astype(np.float16)
        if "bias" in feeds:
            y = (y + feeds["bias"][None, :, None]).astype(np.float16)
        sigmoid = (1 / (1 + np.exp(-y.astype(np.float32)))).astype(np.float16)
        y = (y * sigmoid).astype(np.float16)
    else:
        if "bias" in feeds:
            y += feeds["bias"][None, :, None]
        y = (y / (1 + np.exp(-y))).astype(np.float32)
    state = padded[:, :, x.shape[2]:x.shape[2] + carry]
    if window:
        states = [padded[:, :, t + 1:t + 1 + carry] for t in range(x.shape[2])]
        state = np.zeros((window, *past.shape), dtype=x.dtype)
        kept = states[-window:]
        if kept:
            state[-len(kept):] = np.stack(kept)
    if channels_last:
        y = y.transpose(0, 2, 1).reshape(y.shape[0], y.shape[2], *trailing)
        if window:
            state = state.transpose(0, 1, 3, 2).reshape(window, x.shape[0], carry, *trailing)
        else:
            state = state.transpose(0, 2, 1).reshape(x.shape[0], carry, *trailing)
    if window and x.shape[2] == 0:
        state = original_past.copy()
    return y, state


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("dilation,window,kernel", [(1, 0, 3), (2, 0, 3), (2, 4, 3), (3, 2, 1)])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("dtype,np_dtype", [(DT.FLOAT, np.float32), (DT.FLOAT16, np.float16)])
def test_causal_layout_dilation_state_replay(monkeypatch, eager, channels_last, dilation, window, kernel, with_bias, dtype, np_dtype):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    session = strict_session(causal_model(
        channels_last=channels_last, dilation=dilation, window=window, with_bias=with_bias, kernel=kernel, dtype=dtype,
    ))
    rng = np.random.default_rng(2402)
    carry = (kernel - 1) * dilation
    previous = None
    for batch, time in [(2, 4), (2, 4), (1, 1), (1, 0), (0, 3)]:
        shape = (batch, time, 2, 2) if channels_last else (batch, 4, time)
        state = (batch, carry, 2, 2) if channels_last else (batch, 4, carry)
        if window:
            state = (window, *state)
        feeds = {
            "x": rng.normal(size=shape).astype(np_dtype),
            "weight": rng.normal(size=(4, 1, kernel)).astype(np_dtype),
            "past": rng.normal(size=state).astype(np_dtype),
        }
        if previous is not None and previous.shape == state:
            feeds["past"] = previous.copy()
        if with_bias:
            feeds["bias"] = rng.normal(size=4).astype(np_dtype)
        expected = causal_reference(feeds, channels_last=channels_last, dilation=dilation, window=window)
        y, state = session.run(None, feeds)
        np.testing.assert_allclose(y, expected[0], rtol=2e-3 if dtype == DT.FLOAT16 else 1e-5, atol=1e-6)
        np.testing.assert_array_equal(state, expected[1])
        previous = state.copy()


def scatter_model(dtype, mode, *, axis=-2):
    past_shape = ["batch", 2, "capacity", 3]
    update_shape = ["batch", 2, "time", 3]
    normalized = axis % 4
    past_shape[2], past_shape[normalized] = past_shape[normalized], past_shape[2]
    update_shape[2], update_shape[normalized] = update_shape[normalized], update_shape[2]
    return build(
        "TensorScatter",
        [
            m.tensor("past", dtype, past_shape),
            m.tensor("update", dtype, update_shape),
            m.tensor("indices", DT.INT64, ["batch"]),
        ],
        [m.tensor("present", dtype, past_shape)],
        attrs=[ir.AttrString("mode", mode), ir.AttrInt64("axis", axis)],
    )


@pytest.mark.parametrize("channels_last,dilation", [(False, 2), (True, 1), (True, 2)])
def test_causal_no_past_attributes(channels_last, dilation):
    shape = [1, 4, 4]
    kernel = 5
    state_shape = [1, (kernel - 1) * dilation, 4] if channels_last else [1, 4, (kernel - 1) * dilation]
    model = build(
        "CausalConvWithState",
        [m.tensor("x", DT.FLOAT, shape), m.tensor("weight", DT.FLOAT, [4, 1, kernel])],
        [m.tensor("y", DT.FLOAT, shape), m.tensor("state", DT.FLOAT, state_shape)],
        domain="com.microsoft",
        attrs=[
            ir.AttrInt64("channels_last", int(channels_last)),
            ir.AttrInt64("dilation", dilation),
            ir.AttrString("activation", "silu"),
        ],
    )
    feeds = {
        "x": np.arange(16, dtype=np.float32).reshape(shape) / 8,
        "weight": np.arange(4 * kernel, dtype=np.float32).reshape(4, 1, kernel) / 8,
    }
    expected = causal_reference(
        {**feeds, "past": np.zeros(state_shape, np.float32)},
        channels_last=channels_last, dilation=dilation, window=0,
    )
    actual = strict_session(model).run(None, feeds)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("mode", ["linear", "circular"])
@pytest.mark.parametrize("dtype,np_dtype", [
    (DT.FLOAT, np.float32), (DT.FLOAT16, np.float16), (DT.DOUBLE, np.float64), (DT.BOOL, np.bool_),
    (DT.INT8, np.int8), (DT.INT16, np.int16), (DT.INT32, np.int32), (DT.INT64, np.int64),
    (DT.UINT8, np.uint8), (DT.UINT16, np.uint16), (DT.UINT32, np.uint32), (DT.UINT64, np.uint64),
])
def test_scatter_batch_modulo_exact_live_replay(monkeypatch, eager, mode, dtype, np_dtype):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    session = strict_session(scatter_model(dtype, mode))
    rng = np.random.default_rng(2403)
    offsets = [[0, 4], [3, 1], [8, 15], [np.iinfo(np.int64).max, np.iinfo(np.int64).min]]
    for iteration, indices in enumerate(offsets):
        if mode == "linear":
            indices = offsets[iteration % 2]
        for time in [3, 0]:
            past = rng.integers(0, 100, (2, 2, 8, 3)).astype(np_dtype)
            update = rng.integers(0, 100, (2, 2, time, 3)).astype(np_dtype)
            if np_dtype == np.int64:
                past += np.int64(1 << 50)
                update += np.int64(1 << 51)
            elif np_dtype == np.uint64:
                past += np.uint64(1 << 63)
                update += np.uint64(1 << 63)
            original_past = past.copy()
            expected = past.copy()
            for batch in range(2):
                for token in range(time):
                    index = int(indices[batch]) + token
                    if mode == "circular":
                        index %= 8
                    expected[batch, :, index] = update[batch, :, token]
            feeds = {"past": past, "update": update, "indices": np.array(indices, np.int64)}
            np.testing.assert_array_equal(session.run(None, feeds)[0], expected)
            np.testing.assert_array_equal(feeds["past"], original_past)


def test_scatter_linear_bounds_and_recovery():
    session = strict_session(scatter_model(DT.FLOAT, "linear"))
    feeds = {
        "past": np.zeros((2, 2, 8, 3), np.float32),
        "update": np.ones((2, 2, 3, 3), np.float32),
        "indices": np.array([0, 2], np.int64),
    }
    expected = session.run(None, feeds)[0]
    for index in [-1, 6, 1 << 32, np.iinfo(np.int64).max]:
        feeds["indices"][1] = index
        with pytest.raises(ort.capi.onnxruntime_pybind11_state.EPFail, match="TensorScatter"):
            session.run(None, feeds)
        feeds["indices"][1] = 2
        np.testing.assert_array_equal(session.run(None, feeds)[0], expected)


@pytest.mark.parametrize("axis", [1, -3, 2, -2, 3, -1])
@pytest.mark.parametrize("mode", ["linear", "circular"])
def test_scatter_normalized_axis_and_empty_geometry(axis, mode):
    session = strict_session(scatter_model(DT.INT64, mode, axis=axis))
    for batch, capacity, time in [(2, 8, 3), (1, 5, 2), (0, 8, 3), (2, 0, 0)]:
        past = np.arange(batch * 2 * capacity * 3, dtype=np.int64).reshape(batch, 2, capacity, 3)
        past += 1 << 50
        update = -np.arange(batch * 2 * time * 3, dtype=np.int64).reshape(batch, 2, time, 3)
        indices = np.arange(batch, dtype=np.int64)
        if capacity == 0:
            indices.fill(0)
        elif mode == "circular":
            indices += 2 * capacity - 1
        expected = past.copy()
        for sample in range(batch):
            for token in range(time):
                destination = int(indices[sample]) + token
                if mode == "circular":
                    destination %= capacity
                expected[sample, :, destination] = update[sample, :, token]
        past = np.swapaxes(past, 2, axis % 4).copy()
        update = np.swapaxes(update, 2, axis % 4).copy()
        expected = np.swapaxes(expected, 2, axis % 4).copy()
        np.testing.assert_array_equal(session.run(None, {"past": past, "update": update, "indices": indices})[0], expected)


@pytest.mark.parametrize("mode", ["linear", "circular"])
def test_scatter_dynamic_geometry_error_recovery(mode):
    session = strict_session(scatter_model(DT.FLOAT, mode))
    feeds = {
        "past": np.zeros((2, 2, 8, 3), np.float32),
        "update": np.ones((2, 2, 3, 3), np.float32),
        "indices": np.array([0, 2], np.int64),
    }
    expected = session.run(None, feeds)[0]
    for name, invalid in [
        ("indices", np.zeros(1, np.int64)),
        ("update", np.ones((1, 2, 3, 3), np.float32)),
        ("update", np.ones((2, 2, 9, 3), np.float32)),
    ]:
        valid = feeds[name]
        feeds[name] = invalid
        with pytest.raises(ort.capi.onnxruntime_pybind11_state.EPFail, match="TensorScatter"):
            session.run(None, feeds)
        feeds[name] = valid
        np.testing.assert_array_equal(session.run(None, feeds)[0], expected)


@pytest.mark.parametrize("mode", ["linear", "circular"])
@pytest.mark.parametrize("dtype,np_dtype", [(DT.INT64, np.int64), (DT.DOUBLE, np.float64)])
def test_scatter_initializer_indices_compiled_replay(mode, dtype, np_dtype):
    offsets = [1, 3] if mode == "linear" else [np.iinfo(np.int64).max, np.iinfo(np.int64).min]
    indices = initz("indices", np.array(offsets, np.int64))
    model = build(
        "TensorScatter",
        [
            m.tensor("past", dtype, [2, 2, "capacity", 3]),
            m.tensor("update", dtype, [2, 2, "time", 3]),
            indices,
        ],
        [m.tensor("present", dtype, [2, 2, "capacity", 3])],
        attrs=[ir.AttrString("mode", mode)],
        inits=(indices,),
    )
    session = strict_session(model)
    rng = np.random.default_rng(2406)
    for capacity, time in [(8, 3), (8, 3), (5, 2), (8, 3), (8, 0)]:
        past = rng.integers(0, 100, (2, 2, capacity, 3), dtype=np.int64).astype(np_dtype) + (1 << 50)
        update = -rng.integers(0, 100, (2, 2, time, 3), dtype=np.int64).astype(np_dtype) - (1 << 51)
        expected = past.copy()
        for batch in range(2):
            for token in range(time):
                destination = int(offsets[batch]) + token
                if mode == "circular":
                    destination %= capacity
                expected[batch, :, destination] = update[batch, :, token]
        np.testing.assert_array_equal(session.run(None, {"past": past, "update": update})[0], expected)


def test_scatter_intermediate_indices_remain_live():
    past = m.tensor("past", DT.INT64, [2, 2, 8, 3])
    update = m.tensor("update", DT.INT64, [2, 2, 3, 3])
    offsets = m.tensor("offsets", DT.INT64, [2])
    shift = m.tensor("shift", DT.INT64, [2])
    indices = m.tensor("indices", DT.INT64, [2])
    output = m.tensor("present", DT.INT64, [2, 2, 8, 3])
    nodes = [
        ir.node("Add", [offsets, shift], outputs=[indices]),
        ir.node("TensorScatter", [past, update, indices], attributes={"mode": "circular"}, outputs=[output]),
    ]
    graph = ir.Graph([past, update, offsets, shift], [output], nodes=nodes, opset_imports={"": 24}, name="live_scatter")
    session = strict_session(ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString())
    feeds = {
        "past": np.arange(96, dtype=np.int64).reshape(2, 2, 8, 3) + (1 << 50),
        "update": -np.arange(36, dtype=np.int64).reshape(2, 2, 3, 3) - (1 << 51),
        "offsets": np.array([7, 1], np.int64),
    }
    for shifts in [[0, 0], [2, 9], [-8, -3], [0, 0]]:
        feeds["shift"] = np.array(shifts, np.int64)
        expected = feeds["past"].copy()
        for batch in range(2):
            for token in range(3):
                destination = (int(feeds["offsets"][batch]) + shifts[batch] + token) % 8
                expected[batch, :, destination] = feeds["update"][batch, :, token]
        np.testing.assert_array_equal(session.run(None, feeds)[0], expected)


@pytest.mark.parametrize("dtype,np_dtype,bits,patterns", [
    (DT.DOUBLE, np.float64, np.uint64, [0x8000000000000000, 0x7FF8123456789ABC, 0xFFF0000000000000, 0x7FF0000000000000]),
    (DT.FLOAT, np.float32, np.uint32, [0x80000000, 0x7FC12345, 0xFF800000, 0x7F800000]),
    (DT.FLOAT16, np.float16, np.uint16, [0x8000, 0x7E15, 0xFC00, 0x7C00]),
])
def test_scatter_float_payload_bits(dtype, np_dtype, bits, patterns):
    session = strict_session(scatter_model(dtype, "circular"))
    past = np.resize(np.array(patterns, dtype=bits), (2, 2, 8, 3)).view(np_dtype)
    update = np.resize(np.array(patterns[::-1], dtype=bits), (2, 2, 3, 3)).view(np_dtype)
    expected = past.copy()
    for batch, offset in enumerate([7, 2]):
        for token in range(3):
            expected[batch, :, (offset + token) % 8] = update[batch, :, token]
    actual = session.run(None, {"past": past, "update": update, "indices": np.array([7, 2], np.int64)})[0]
    np.testing.assert_array_equal(actual.view(bits), expected.view(bits))


@pytest.mark.parametrize("channels_last", [False, True])
def test_causal_dynamic_state_geometry_error_recovery(channels_last):
    model = causal_model(channels_last=channels_last, dilation=2, window=3, with_bias=False)
    session = strict_session(model)
    shape = (2, 3, 2, 2) if channels_last else (2, 4, 3)
    state = (3, 2, 4, 2, 2) if channels_last else (3, 2, 4, 4)
    feeds = {
        "x": np.ones(shape, np.float32),
        "weight": np.ones((4, 1, 3), np.float32),
        "past": np.zeros(state, np.float32),
    }
    expected = session.run(None, feeds)
    good = feeds["past"]
    feeds["past"] = good[:, :1]
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.EPFail, match="CausalConvWithState"):
        session.run(None, feeds)
    feeds["past"] = good
    for actual, reference in zip(session.run(None, feeds), expected, strict=True):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("eager", [False, True])
def test_gate_dynamic_geometry_error_recovery(monkeypatch, eager):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    shape = ["batch", "time", "heads"]
    model = build(
        "LinearAttentionGate",
        [
            m.tensor("a", DT.FLOAT, shape),
            m.tensor("bias", DT.FLOAT, ["heads"]),
            m.tensor("scale", DT.FLOAT, ["heads"]),
            m.tensor("b", DT.FLOAT, shape),
        ],
        [m.tensor("decay", DT.FLOAT, shape), m.tensor("beta", DT.FLOAT, shape)],
        domain="com.microsoft",
    )
    session = strict_session(model)
    feeds = {
        "a": np.ones((2, 3, 4), np.float32),
        "bias": np.zeros(4, np.float32),
        "scale": -np.ones(4, np.float32),
        "b": np.zeros((2, 3, 4), np.float32),
    }
    expected = session.run(None, feeds)
    for name, invalid in [("b", np.zeros((1, 3, 4), np.float32)), ("bias", np.zeros(1, np.float32))]:
        valid = feeds[name]
        feeds[name] = invalid
        with pytest.raises(ort.capi.onnxruntime_pybind11_state.EPFail, match="LinearAttentionGate"):
            session.run(None, feeds)
        feeds[name] = valid
        for actual, reference in zip(session.run(None, feeds), expected, strict=True):
            # A failed trace can switch fused softplus to eager evaluation (one FP32 ULP).
            np.testing.assert_array_max_ulp(actual, reference, maxulp=1)


@pytest.mark.parametrize("eager", [False, True])
def test_linear_attention_dynamic_geometry_recovery(monkeypatch, eager):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    model, feeds = _linear_attention_model(
        "gated_delta", B=1, T=3, q_heads=2, kv_heads=2, d_k=4, d_v=3,
        with_past=True, scalar_decay=True, dyn_time=True,
    )
    session = strict_session(model)
    expected = session.run(None, feeds)
    for name in ["key", "value", "decay", "beta"]:
        valid = feeds[name]
        feeds[name] = valid[:, :1]
        with pytest.raises(ort.capi.onnxruntime_pybind11_state.EPFail, match="LinearAttention"):
            session.run(None, feeds)
        feeds[name] = valid
        for actual, reference in zip(session.run(None, feeds), expected, strict=True):
            np.testing.assert_array_equal(actual, reference)


def linear_reference(feeds):
    q, k, v = (feeds[name].astype(np.float64) for name in ["query", "key", "value"])
    batch, time, _ = q.shape
    state = feeds["past_state"].astype(np.float64).copy()
    heads, dk, dv = state.shape[1:]
    q = q.reshape(batch, time, heads, dk) / np.sqrt(dk)
    k = k.reshape(batch, time, heads, dk)
    v = v.reshape(batch, time, heads, dv)
    output = np.empty((batch, time, heads, dv), np.float64)
    for token in range(time):
        state *= np.exp(feeds["decay"][:, token].astype(np.float64))[:, :, None, None]
        key = k[:, token]
        retrieval = np.einsum("bhk,bhkv->bhv", key, state)
        delta = feeds["beta"][:, token, :, None].astype(np.float64) * (v[:, token] - retrieval)
        state += key[:, :, :, None] * delta[:, :, None, :]
        output[:, token] = np.einsum("bhk,bhkv->bhv", q[:, token], state)
    return output.reshape(batch, time, heads * dv), state


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("time", [128, 200])
@pytest.mark.parametrize("normalized", [False, True])
def test_gated_delta_zero_weak_decay_recurrence(monkeypatch, eager, time, normalized):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    model, feeds = _linear_attention_model(
        "gated_delta", B=1, T=time, q_heads=2, kv_heads=2, d_k=16, d_v=4,
        with_past=True, scalar_decay=True,
    )
    rng = np.random.default_rng(2422)
    for name in ["query", "key", "value"]:
        feeds[name] = (rng.normal(size=feeds[name].shape) * (0.5 if name == "key" else 0.3)).astype(np.float32)
    feeds["past_state"] = (rng.normal(size=feeds["past_state"].shape) * 0.1).astype(np.float32)
    if normalized:
        key = feeds["key"].reshape(1, time, 2, 16)
        key /= np.linalg.norm(key, axis=-1, keepdims=True)
    feeds["beta"].fill(0.95)
    session = strict_session(model)
    delta_model, _ = _linear_attention_model(
        "delta", B=1, T=time, q_heads=2, kv_heads=2, d_k=16, d_v=4,
        with_past=True, scalar_decay=True,
    )
    delta = strict_session(delta_model).run(None, {name: value for name, value in feeds.items() if name != "decay"})
    for decay in [0.0, -1e-7, -0.1, 0.0]:
        feeds["decay"].fill(decay)
        expected = linear_reference(feeds)
        outputs = session.run(None, feeds)
        for actual, reference in zip(outputs, expected, strict=True):
            np.testing.assert_allclose(actual, reference, rtol=2e-3, atol=2e-5)
        if decay == 0:
            for actual, reference in zip(outputs, delta, strict=True):
                np.testing.assert_allclose(actual, reference, rtol=2e-3, atol=2e-5)
