"""Packed SSM correctness against independent, sequential NumPy references.

The packed operators have no ORT CPU kernels. Profiling and disabled EP retry ensure that these
tests exercise MLX, including the chunked prefill path and the V-major FP32 state boundary.
"""

from __future__ import annotations

import json

import ml_dtypes
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnxruntime.capi.onnxruntime_pybind11_state import EPFail, NotImplemented as OrtNotImplemented

import _models as m

DT = ir.DataType
DTYPES = [(DT.FLOAT, np.float32), (DT.FLOAT16, np.float16), (DT.BFLOAT16, ml_dtypes.bfloat16)]


def _model(op, inputs, outputs, attributes):
    # Python ORT cannot feed BF16 arrays directly, so keep those boundaries FP32 and cast on MLX.
    graph_inputs, graph_outputs, before, after = [], [], [], []
    for value in inputs:
        if not value.name:
            continue
        if value.dtype == DT.BFLOAT16:
            external = ir.Value(
                name=value.name + "_fp32", type=ir.TensorType(DT.FLOAT), shape=value.shape
            )
            before.append(ir.node("Cast", [external], attributes={"to": int(DT.BFLOAT16)}, outputs=[value]))
            graph_inputs.append(external)
        else:
            graph_inputs.append(value)
    for value in outputs:
        if value.dtype == DT.BFLOAT16:
            external = ir.Value(
                name=value.name + "_fp32", type=ir.TensorType(DT.FLOAT), shape=value.shape
            )
            after.append(ir.node("Cast", [value], attributes={"to": int(DT.FLOAT)}, outputs=[external]))
            graph_outputs.append(external)
        else:
            graph_outputs.append(value)
    node = ir.node(op, inputs, attributes=attributes, domain="com.microsoft", outputs=outputs)
    graph = ir.Graph(
        graph_inputs, graph_outputs, nodes=[*before, node, *after],
        name=f"packed_{op}", opset_imports={"": 24, "com.microsoft": 1},
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def _session(model, tmp_path):
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "packed_ssm")
    session = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    session.disable_fallback()
    return session


def _assert_mlx(session, runs=1):
    with open(session.end_profiling()) as profile:
        events = json.load(profile)
    providers = [
        event["args"]["provider"] for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    ]
    assert len(providers) >= runs
    assert set(providers) == {"MLXExecutionProvider"}, providers


def _feeds_for_session(session, feeds):
    return {
        value.name: (
            feeds[value.name[:-5]].astype(np.float32)
            if value.name.endswith("_fp32") else feeds[value.name]
        )
        for value in session.get_inputs()
    }


def _assert_results(actual, expected, dtype):
    tolerance = 1.5e-2 if dtype == DT.BFLOAT16 else 2e-3 if dtype == DT.FLOAT16 else 3e-5
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance, err_msg=f"output {index}")
        assert got.shape == want.shape


def _conv_model(channels, kernel=4, dtype=DT.FLOAT16, activation="silu", bias=True, dilation=1, capture=0):
    pad = (kernel - 1) * dilation
    inputs = [
        m.tensor("input", dtype, ["tokens", channels]),
        m.tensor("weight", dtype, [channels, 1, kernel]),
        m.tensor("cu", DT.INT32, ["offset_count"]),
        m.tensor("bias", dtype, [channels]) if bias else ir.Value(name=""),
        m.tensor("initial_state", dtype, ["batch", channels, pad]),
    ]
    outputs = [
        m.tensor("output", dtype, ["tokens", channels]),
        m.tensor("final_state", dtype, ["batch", channels, pad]),
    ]
    if capture:
        inputs.append(m.tensor("capture_count", DT.INT32, ["batch"]))
        outputs.append(m.tensor("state_update", dtype, ["batch", capture, channels]))
    return _model("VarlenCausalConvWithState", inputs, outputs, {
        "activation": activation, "dilation": dilation, "state_update_capacity": capture,
    })


def _conv_feeds(lengths, channels, kernel, np_dtype, dilation=1, bias=True):
    rng = np.random.default_rng(710)
    feeds = {
        "input": rng.normal(0, 0.4, (sum(lengths), channels)).astype(np_dtype),
        "weight": rng.normal(0, 0.4, (channels, 1, kernel)).astype(np_dtype),
        "initial_state": rng.normal(
            0, 0.4, (len(lengths), channels, (kernel - 1) * dilation)
        ).astype(np_dtype),
        "cu": np.array([0, *np.cumsum(lengths)], dtype=np.int32),
    }
    if bias:
        feeds["bias"] = rng.normal(0, 0.2, channels).astype(np_dtype)
    return feeds


def _conv_reference(feeds, activation="silu", dilation=1):
    x = feeds["input"].astype(np.float64)
    weight = feeds["weight"].astype(np.float64)[:, 0, :]
    state = feeds["initial_state"].copy()
    output = np.empty_like(x)
    pad = state.shape[-1]
    for b, (start, end) in enumerate(zip(feeds["cu"][:-1], feeds["cu"][1:], strict=True)):
        raw = np.concatenate([state[b].astype(np.float64).T, x[start:end]], axis=0)
        for local in range(end - start):
            taps = raw[local + np.arange(weight.shape[1]) * dilation]
            result = np.sum(taps.T * weight, axis=-1)
            if "bias" in feeds:
                result += feeds["bias"].astype(np.float64)
            if activation in ("silu", "swish"):
                result *= 1 / (1 + np.exp(-result))
            output[start + local] = result
        if pad:
            state[b] = raw[-pad:].T
    return [output.astype(feeds["input"].dtype).astype(np.float32), state.astype(np.float32)]


@pytest.mark.parametrize("dtype,np_dtype", DTYPES)
@pytest.mark.parametrize(
    "kernel,dilation,activation,bias",
    [(4, 1, "none", True), (4, 1, "silu", True), (3, 2, "swish", False), (1, 1, "none", False)],
)
def test_varlen_causal_conv_ragged(tmp_path, dtype, np_dtype, kernel, dilation, activation, bias):
    feeds = _conv_feeds([1, 5, 2], 7, kernel, np_dtype, dilation, bias)
    initial = feeds["initial_state"].copy()
    session = _session(_conv_model(7, kernel, dtype, activation, bias, dilation), tmp_path)
    actual = session.run(None, _feeds_for_session(session, feeds))
    _assert_results(actual, _conv_reference(feeds, activation, dilation), dtype)
    np.testing.assert_array_equal(feeds["initial_state"], initial)
    np.testing.assert_array_equal(actual[1], _conv_reference(feeds, activation, dilation)[1])
    _assert_mlx(session)


def test_varlen_causal_conv_qwen35_geometry(tmp_path):
    feeds = _conv_feeds([5, 1], 6144, 4, np.float16)
    session = _session(_conv_model(6144), tmp_path)
    _assert_results(session.run(None, feeds), _conv_reference(feeds), DT.FLOAT16)
    _assert_mlx(session)


def _gdn_model(
    hq=2, hv=6, dk=8, dv=5, dtype=DT.FLOAT16, gate="qwen", beta="sigmoid",
    norm=True, chunk=64, packing="ragged", with_state=True, scale=0.0,
    rule="gated_delta", per_key_decay=False, capture=0,
):
    leading = ["tokens"] if packing != "uniform4" else ["batch", "sequence"]
    inputs = [
        m.tensor("query", dtype, [*leading, hq, dk]),
        m.tensor("key", dtype, [*leading, hq, dk]),
        m.tensor("value", dtype, [*leading, hv, dv]),
        m.tensor("cu", DT.INT32, ["offset_count"]) if packing == "ragged" else ir.Value(name=""),
        m.tensor("decay", DT.FLOAT, [*leading, hv, dk] if per_key_decay else [*leading, hv]),
        m.tensor("beta", DT.FLOAT, [*leading, hv]),
        m.tensor("initial_state", DT.FLOAT, ["batch", hv, dv, dk]) if with_state else ir.Value(name=""),
    ]
    if gate == "qwen":
        inputs += [m.tensor("a_log", DT.FLOAT, [hv]), m.tensor("dt_bias", DT.FLOAT, [hv])]
    outputs = [
        m.tensor("output", dtype, [*leading, hv, dv]),
        m.tensor("final_state", DT.FLOAT, ["batch", hv, dv, dk]),
    ]
    if capture:
        assert gate == "qwen"
        inputs.append(m.tensor("capture_count", DT.INT32, ["batch"]))
        outputs.append(m.tensor("state_update", DT.FLOAT, ["batch", capture * (hv + hq * dk + hv * dv)]))
    return _model(
        "GatedDeltaNet", inputs, outputs,
        {
            "update_rule": rule, "gate_activation": gate, "beta_activation": beta,
            "qk_l2_norm": int(norm), "chunk_size": chunk, "scale": scale, "state_update_capacity": capture,
        },
    )


def _gdn_feeds(lengths, hq=2, hv=6, dk=8, dv=5, np_dtype=np.float16):
    rng = np.random.default_rng(720)
    tokens = sum(lengths)
    return {
        "query": rng.normal(0, 0.7, (tokens, hq, dk)).astype(np_dtype),
        "key": rng.normal(0, 0.7, (tokens, hq, dk)).astype(np_dtype),
        "value": rng.normal(0, 0.7, (tokens, hv, dv)).astype(np_dtype),
        "decay": rng.normal(0, 0.5, (tokens, hv)).astype(np.float32),
        "beta": rng.normal(0, 0.5, (tokens, hv)).astype(np.float32),
        "a_log": rng.uniform(-3, -1, hv).astype(np.float32),
        "dt_bias": rng.normal(0, 0.3, hv).astype(np.float32),
        "initial_state": rng.normal(0, 0.1, (len(lengths), hv, dv, dk)).astype(np.float32),
        "cu": np.array([0, *np.cumsum(lengths)], dtype=np.int32),
    }


def _gdn_reference(feeds, lengths, gate="qwen", beta_activation="sigmoid", norm=True, scale=0.0):
    hq, dk = feeds["query"].shape[-2:]
    hv, dv = feeds["value"].shape[-2:]
    q = feeds["query"].reshape(-1, hq, dk).astype(np.float64)
    k = feeds["key"].reshape(-1, hq, dk).astype(np.float64)
    v = feeds["value"].reshape(-1, hv, dv).astype(np.float64)
    if norm:
        q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-12)
        k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-12)
    q, k = np.repeat(q, hv // hq, axis=1), np.repeat(k, hv // hq, axis=1)
    gates = feeds["decay"].reshape(-1, hv).astype(np.float64)
    beta = feeds["beta"].reshape(-1, hv).astype(np.float64)
    if gate == "qwen":
        gates = -np.exp(feeds["a_log"].astype(np.float64)) * np.logaddexp(
            0, gates + feeds["dt_bias"].astype(np.float64)
        )
    if beta_activation == "sigmoid":
        beta = 1 / (1 + np.exp(-beta))
    state = feeds.get("initial_state", np.zeros((len(lengths), hv, dv, dk))).astype(np.float64).copy()
    output = np.empty_like(v)
    start = 0
    for b, length in enumerate(lengths):
        for token in range(start, start + length):
            state[b] *= np.exp(gates[token])[:, None, None]
            retrieved = np.einsum("hvk,hk->hv", state[b], k[token])
            delta = beta[token, :, None] * (v[token] - retrieved)
            state[b] += delta[:, :, None] * k[token, :, None, :]
            output[token] = np.einsum("hvk,hk->hv", state[b], q[token]) * (scale or dk**-0.5)
        start += length
    output = output.reshape(feeds["value"].shape).astype(feeds["value"].dtype).astype(np.float32)
    return [output, state.astype(np.float32)]


@pytest.mark.parametrize("dtype,np_dtype", DTYPES)
@pytest.mark.parametrize("lengths", [(1, 1), (5, 3), (64, 1), (65, 2), (129, 7), (257, 13), (0, 3)])
def test_gated_delta_net_ragged_inverse_gqa(tmp_path, dtype, np_dtype, lengths):
    feeds = _gdn_feeds(lengths, np_dtype=np_dtype)
    initial = feeds["initial_state"].copy()
    session = _session(_gdn_model(dtype=dtype), tmp_path)
    actual = session.run(None, _feeds_for_session(session, feeds))
    _assert_results(actual, _gdn_reference(feeds, lengths), dtype)
    np.testing.assert_allclose(actual[1], _gdn_reference(feeds, lengths)[1], rtol=3e-5, atol=3e-5)
    assert actual[1].dtype == np.float32
    np.testing.assert_array_equal(feeds["initial_state"], initial)
    for batch, length in enumerate(lengths):
        if length == 0:
            np.testing.assert_array_equal(actual[1][batch], initial[batch])
    _assert_mlx(session)


def test_gated_delta_net_qwen35_geometry(tmp_path):
    feeds = _gdn_feeds([67, 1], hq=16, hv=16, dk=128, dv=128)
    session = _session(_gdn_model(hq=16, hv=16, dk=128, dv=128), tmp_path)
    actual = session.run(None, feeds)
    expected = _gdn_reference(feeds, [67, 1])
    _assert_results(actual, expected, DT.FLOAT16)
    np.testing.assert_allclose(actual[1], expected[1], rtol=3e-5, atol=3e-5)
    _assert_mlx(session)


@pytest.mark.parametrize("packing", ["ragged", "uniform3", "uniform4"])
def test_gated_delta_net_effective_gates_chunk32_and_uniform_packing(tmp_path, packing):
    feeds = _gdn_feeds([33, 33], np_dtype=np.float32)
    for name in ("query", "key"):
        feeds[name] /= np.sqrt(np.sum(feeds[name] ** 2, axis=-1, keepdims=True) + 1e-12)
    feeds["decay"] = -np.abs(feeds["decay"])
    feeds["beta"] = 1 / (1 + np.exp(-feeds["beta"]))
    if packing == "uniform4":
        for name in ("query", "key", "value", "decay", "beta"):
            feeds[name] = feeds[name].reshape(2, 33, *feeds[name].shape[1:])
    session = _session(
        _gdn_model(dtype=DT.FLOAT, gate="none", beta="none", norm=False, chunk=32,
                   packing=packing, scale=0.25),
        tmp_path,
    )
    actual = session.run(None, _feeds_for_session(session, feeds))
    _assert_results(actual, _gdn_reference(feeds, [33, 33], "none", "none", False, 0.25), DT.FLOAT)
    _assert_mlx(session)


def test_gated_delta_net_zero_initial_state_and_extreme_gates(tmp_path):
    feeds = _gdn_feeds([65])
    del feeds["initial_state"]
    feeds["decay"][::2] = 80
    feeds["decay"][1::2] = -80
    feeds["beta"][::2] = 80
    feeds["beta"][1::2] = -80
    feeds["query"][0] = 0
    feeds["key"][0] = 0
    session = _session(_gdn_model(with_state=False), tmp_path)
    actual = session.run(None, _feeds_for_session(session, feeds))
    _assert_results(actual, _gdn_reference(feeds, [65]), DT.FLOAT16)
    assert all(np.isfinite(value).all() for value in actual)
    _assert_mlx(session)


@pytest.mark.parametrize("operator", ["conv", "gdn"])
def test_packed_ssm_prefill_decode_state_handoff(tmp_path, operator):
    model = _conv_model(7) if operator == "conv" else _gdn_model()
    session = _session(model, tmp_path)
    state = None
    for lengths in ([65, 3], [1, 1], [5], [1, 1]):
        feeds = (
            _conv_feeds(lengths, 7, 4, np.float16)
            if operator == "conv" else _gdn_feeds(lengths)
        )
        if state is not None:
            count = min(len(state), len(lengths))
            feeds["initial_state"][:count] = state[:count]
        expected = _conv_reference(feeds) if operator == "conv" else _gdn_reference(feeds, lengths)
        actual = session.run(None, _feeds_for_session(session, feeds))
        _assert_results(actual, expected, DT.FLOAT16)
        state = actual[1]
    _assert_mlx(session, runs=4)


@pytest.mark.parametrize("operator", ["conv", "gdn"])
def test_packed_ssm_in_place_state_output(tmp_path, operator):
    feeds = _conv_feeds([65, 2], 7, 4, np.float16) if operator == "conv" else _gdn_feeds([65, 2])
    model = _conv_model(7) if operator == "conv" else _gdn_model()
    expected = _conv_reference(feeds) if operator == "conv" else _gdn_reference(feeds, [65, 2])
    session = _session(model, tmp_path)
    binding = session.io_binding()
    for name, value in _feeds_for_session(session, feeds).items():
        binding.bind_cpu_input(name, value)
    state = feeds["initial_state"]
    binding.bind_output("output")
    binding.bind_output(
        "final_state", device_type="cpu", device_id=0, element_type=state.dtype,
        shape=state.shape, buffer_ptr=state.ctypes.data,
    )
    session.run_with_iobinding(binding)
    actual = binding.copy_outputs_to_cpu()
    _assert_results(actual, expected, DT.FLOAT16)
    np.testing.assert_allclose(state, expected[1], rtol=3e-5, atol=3e-5)
    _assert_mlx(session)


@pytest.mark.parametrize(
    "model",
    [
        _conv_model(7, capture=1),
        _gdn_model(capture=1),
        _gdn_model(per_key_decay=True),
        _gdn_model(rule="linear"),
    ],
    ids=["conv-capture", "gdn-capture", "per-key-decay", "other-update-rule"],
)
def test_packed_ssm_declines_unsupported_modes(model):
    with pytest.raises(OrtNotImplemented, match="VarlenCausalConvWithState|GatedDeltaNet"):
        ort.InferenceSession(model, providers=m.EP_PROVIDERS)


@pytest.mark.parametrize("operator", ["conv", "gdn"])
@pytest.mark.parametrize("offsets", [[1, 2, 4], [0, 2, 3], [0, 5, 4], [0, -1, 4]])
def test_packed_ssm_rejects_invalid_offsets(tmp_path, operator, offsets):
    feeds = _conv_feeds([2, 2], 7, 4, np.float32) if operator == "conv" else _gdn_feeds([2, 2])
    feeds["cu"] = np.array(offsets, dtype=np.int32)
    model = _conv_model(7, dtype=DT.FLOAT) if operator == "conv" else _gdn_model()
    session = _session(model, tmp_path)
    with pytest.raises(EPFail, match="cumulative_sequence_length must partition all tokens"):
        session.run(None, _feeds_for_session(session, feeds))
    _assert_mlx(session)


@pytest.mark.parametrize("operator", ["conv", "gdn"])
def test_packed_ssm_rejects_state_batch_mismatch(tmp_path, operator):
    feeds = _conv_feeds([2, 2], 7, 4, np.float32) if operator == "conv" else _gdn_feeds([2, 2])
    feeds["initial_state"] = feeds["initial_state"][:1].copy()
    model = _conv_model(7, dtype=DT.FLOAT) if operator == "conv" else _gdn_model()
    session = _session(model, tmp_path)
    with pytest.raises(EPFail, match="initial_state"):
        session.run(None, _feeds_for_session(session, feeds))
    _assert_mlx(session)


def test_varlen_causal_conv_rejects_empty_sequence(tmp_path):
    feeds = _conv_feeds([2, 2], 7, 4, np.float16)
    feeds["cu"] = np.array([0, 0, 4], dtype=np.int32)
    session = _session(_conv_model(7), tmp_path)
    with pytest.raises(EPFail, match="cumulative_sequence_length must partition all tokens"):
        session.run(None, feeds)
    _assert_mlx(session)


def test_qwen35_convolution_to_chunked_gated_delta_pipeline(tmp_path):
    shapes = {
        "input": (DT.FLOAT16, ["tokens", 6144]),
        "weight": (DT.FLOAT16, [6144, 1, 4]),
        "bias": (DT.FLOAT16, [6144]),
        "cu": (DT.INT32, ["offset_count"]),
        "conv_initial_state": (DT.FLOAT16, ["batch", 6144, 3]),
        "initial_state": (DT.FLOAT, ["batch", 16, 128, 128]),
        "decay": (DT.FLOAT, ["tokens", 16]),
        "beta": (DT.FLOAT, ["tokens", 16]),
        "a_log": (DT.FLOAT, [16]),
        "dt_bias": (DT.FLOAT, [16]),
    }
    inputs = {name: m.tensor(name, dtype, shape) for name, (dtype, shape) in shapes.items()}
    convolved = m.tensor("convolved", DT.FLOAT16, ["tokens", 6144])
    conv_final = m.tensor("conv_final_state", DT.FLOAT16, ["batch", 6144, 3])
    conv = ir.node(
        "VarlenCausalConvWithState",
        [inputs[name] for name in ("input", "weight", "cu", "bias", "conv_initial_state")],
        attributes={"activation": "silu"}, domain="com.microsoft",
        outputs=[convolved, conv_final],
    )
    split = [m.tensor(name, DT.FLOAT16, ["tokens", 2048]) for name in ("q_flat", "k_flat", "v_flat")]
    split_node = ir.node("Split", [convolved], attributes={"axis": 1, "num_outputs": 3}, outputs=split)
    target = ir.Value(
        name="head_shape", type=ir.TensorType(DT.INT64), shape=ir.Shape([3]),
        const_value=ir.tensor(np.array([0, 16, 128], dtype=np.int64)),
    )
    heads = [m.tensor(name, DT.FLOAT16, ["tokens", 16, 128]) for name in ("q", "k", "v")]
    reshapes = [
        ir.node("Reshape", [value, target], outputs=[head])
        for value, head in zip(split, heads, strict=True)
    ]
    output = m.tensor("output", DT.FLOAT16, ["tokens", 16, 128])
    final = m.tensor("final_state", DT.FLOAT, ["batch", 16, 128, 128])
    gdn = ir.node(
        "GatedDeltaNet",
        [*heads, *(inputs[name] for name in ("cu", "decay", "beta", "initial_state", "a_log", "dt_bias"))],
        attributes={
            "update_rule": "gated_delta", "gate_activation": "qwen",
            "beta_activation": "sigmoid", "qk_l2_norm": 1, "chunk_size": 64,
        },
        domain="com.microsoft", outputs=[output, final],
    )
    graph = ir.Graph(
        list(inputs.values()), [output, conv_final, final], nodes=[conv, split_node, *reshapes, gdn],
        initializers=[target], opset_imports={"": 24, "com.microsoft": 1}, name="qwen35_packed_pipeline",
    )
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    conv_feeds = _conv_feeds([65, 1], 6144, 4, np.float16)
    gdn_feeds = _gdn_feeds([65, 1], hq=16, hv=16, dk=128, dv=128)
    convolved, conv_state = _conv_reference(conv_feeds)
    for name, value in zip(("query", "key", "value"), np.split(convolved, 3, axis=-1), strict=True):
        gdn_feeds[name] = value.reshape(-1, 16, 128).astype(np.float16)
    expected_output, expected_state = _gdn_reference(gdn_feeds, [65, 1])
    feeds = {name: value for name, value in conv_feeds.items() if name != "initial_state"}
    feeds["conv_initial_state"] = conv_feeds["initial_state"]
    feeds.update({name: gdn_feeds[name] for name in ("decay", "beta", "initial_state", "a_log", "dt_bias")})
    session = _session(model, tmp_path)
    actual = session.run(None, feeds)
    _assert_results(actual, [expected_output, conv_state, expected_state], DT.FLOAT16)
    np.testing.assert_array_equal(actual[1], conv_state)
    np.testing.assert_allclose(actual[2], expected_state, rtol=3e-5, atol=3e-5)
    _assert_mlx(session)
