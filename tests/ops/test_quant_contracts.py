"""Focused quantized operator contracts, with independent exact-valued oracles."""

import json
import os
from pathlib import Path
import subprocess
import sys

import ml_dtypes
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

import _models as m
from test_matmulnbits_asym import _matmulnbits_model
from test_qmoe import _qmoe_model


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("k", [64, 51])
@pytest.mark.parametrize("intermediate", [False, True])
def test_matmulnbits_live_weight_replay(monkeypatch, eager, k, intermediate):
    if eager:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    else:
        monkeypatch.delenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", raising=False)
    model, feeds = _matmulnbits_model(M=1, K=k, N=4, block=32, asymmetric=False)
    if intermediate:
        proto = onnx.load_from_string(model)
        proto.graph.node[0].input[1] = "live_b"
        proto.graph.node.insert(0, helper.make_node("Identity", ["b"], ["live_b"]))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        model = proto.SerializeToString()
    else:
        options = ort.SessionOptions()
    feeds["a"].fill(1)
    feeds["scales"].fill(1)
    feeds["b"].fill(0x88)
    session = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    m.assert_mlx_claims(model, feeds)
    for value in [0x88, 0x99, 0x77]:
        feeds["b"].fill(value)
        expected = np.full((1, 1, 4), k * ((value & 15) - 8), np.float32)
        np.testing.assert_array_equal(session.run(None, feeds)[0], expected)
        feeds["b"] = feeds["b"].copy()
        np.testing.assert_array_equal(session.run(None, feeds)[0], expected)


@pytest.mark.parametrize("k", [64, 51])
def test_matmulnbits_live_weight_compiled_route(tmp_path, k):
    trace = tmp_path / "trace.json"
    subprocess.run(
        [sys.executable, "-m", "pytest",
         f"{Path(__file__)}::test_matmulnbits_live_weight_replay[False-{k}-False]",
         "-q", "-p", "no:cacheprovider"],
        check=True, timeout=120,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
    )
    events = json.loads(trace.read_text())
    paths = [e["args"] for e in events if e.get("cat") == "ep.path"
             and e.get("args", {}).get("ops") == "MatMulNBits"]
    assert any(p["cache"] == "HIT" and p["path"] != "eager" for p in paths), paths


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("block", [16, 32, 256])
@pytest.mark.parametrize("floating_zp", [False, True])
@pytest.mark.parametrize("initializer", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, ml_dtypes.bfloat16])
def test_matmulnbits_bits_float_zero_point_bias(bits, block, floating_zp, initializer, dtype):
    k, n = block + 3, 4
    blocks = 2
    pack = 8 // bits
    midpoint = 1 << (bits - 1)
    packed = sum((midpoint + 1) << (lane * bits) for lane in range(pack))
    b = np.full((n, blocks, block // pack), packed, np.uint8)
    scales = np.full((n, blocks), 0.5, dtype)
    tensor_type = helper.np_dtype_to_tensor_dtype(scales.dtype)
    zp = (
        np.full((n, blocks), midpoint - 0.5, dtype)
        if floating_zp
        else np.full((n, (blocks + pack - 1) // pack),
                     sum(midpoint << (lane * bits) for lane in range(pack)), np.uint8)
    )
    inputs = [
        helper.make_tensor_value_info("a", tensor_type, [1, k]),
        helper.make_tensor_value_info("b", TensorProto.UINT8, list(b.shape)),
        helper.make_tensor_value_info("s", tensor_type, list(scales.shape)),
        helper.make_tensor_value_info("zp", tensor_type if floating_zp else TensorProto.UINT8, list(zp.shape)),
        helper.make_tensor_value_info("bias", tensor_type, [n]),
    ]
    feeds = {"a": np.zeros((1, k), dtype), "b": b, "s": scales,
             "zp": zp, "bias": np.arange(n, dtype=np.float32).astype(dtype)}
    feeds["a"][:, :4] = 1
    feeds["a"][:, -3:] = 1
    inits = []
    if initializer:
        for name in ["b", "s", "zp"]:
            inits.append(numpy_helper.from_array(feeds.pop(name), name))
        inputs = [v for v in inputs if v.name not in {"b", "s", "zp"}]
    node = helper.make_node("MatMulNBits", ["a", "b", "s", "zp", "", "bias"], ["y"],
                            domain="com.microsoft", K=k, N=n, block_size=block, bits=bits)
    proto = helper.make_model(
        helper.make_graph([node], "nbits", inputs,
                          [helper.make_tensor_value_info("y", tensor_type, [1, n])], inits),
        opset_imports=[helper.make_opsetid("", 24), helper.make_opsetid("com.microsoft", 1)],
        ir_version=11,
    )
    if dtype == ml_dtypes.bfloat16:
        casts = []
        for value in proto.graph.input:
            if value.type.tensor_type.elem_type == TensorProto.BFLOAT16:
                name = value.name
                value.name = f"{name}_fp32"
                value.type.tensor_type.elem_type = TensorProto.FLOAT
                feeds[value.name] = feeds.pop(name).astype(np.float32)
                casts.append(helper.make_node("Cast", [value.name], [name], to=TensorProto.BFLOAT16))
        proto.graph.node[0].output[0] = "y_bf16"
        proto.graph.node.extend([helper.make_node("Cast", ["y_bf16"], ["y"], to=TensorProto.FLOAT)])
        for cast in reversed(casts):
            proto.graph.node.insert(0, cast)
        proto.graph.output[0].type.tensor_type.elem_type = TensorProto.FLOAT
    model = proto.SerializeToString()
    m.assert_mlx_claims(model, feeds)
    expected = np.arange(n, dtype=np.float32)[None] + 7 * (0.75 if floating_zp else 0.5)
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    actual = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"]).run(None, feeds)[0]
    assert actual.dtype == np.dtype(np.float32 if dtype == ml_dtypes.bfloat16 else dtype)
    np.testing.assert_array_equal(actual.astype(np.float32), expected)


@pytest.mark.parametrize("width", [8, 24, 40])
@pytest.mark.parametrize("asymmetric", [False, True])
@pytest.mark.parametrize("negative_axes", [False, True])
def test_gather_block_partial_tail(width, asymmetric, negative_axes):
    blocks = (width + 15) // 16
    scales = np.arange(1, 2 * blocks + 1, dtype=np.float32).reshape(2, blocks)
    names = ["data", "indices", "scales"] + (["zp"] if asymmetric else [])
    feeds = {"data": np.full((2, width // 2), 0x99, np.uint8),
             "indices": np.array([1, 0], np.int64), "scales": scales}
    if asymmetric:
        feeds["zp"] = np.full((2, (blocks + 1) // 2), 0x77, np.uint8)
    inputs = [helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(feeds[name].dtype),
                                           list(feeds[name].shape)) for name in names]
    node = helper.make_node("GatherBlockQuantized", names, ["y"], domain="com.microsoft",
                            bits=4, block_size=16, gather_axis=-2 if negative_axes else 0,
                            quantize_axis=-1 if negative_axes else 1)
    model = helper.make_model(
        helper.make_graph([node], "gather", inputs,
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, width])]),
        opset_imports=[helper.make_opsetid("", 24), helper.make_opsetid("com.microsoft", 1)],
        ir_version=11,
    ).SerializeToString()
    expected = np.repeat(scales[[1, 0]], 16, axis=1)[:, :width] * (2 if asymmetric else 1)
    m.assert_mlx_claims(model, feeds)
    m.assert_matches_ref(model, feeds, [expected], rtol=0, atol=0)


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("nan_unselected", [False, True])
def test_qmoe_exact_topk_ties(k, nan_unselected):
    model, feeds = _qmoe_model(E=3, H=4, I=4, K=k, bits=4, act="identity",
                               fusion=0, fc2b=True, T=1)
    feeds["fc1w"].fill(0x88)
    feeds["fc2w"].fill(0x88)
    feeds["fc1s"].fill(1)
    feeds["fc2s"].fill(1)
    feeds["fc2b"][:] = np.array([0, 2, 9], np.float32)[:, None]
    if nan_unselected:
        feeds["fc2b"][0] = np.nan
    feeds["router"].fill(0)
    expected = np.full((1, 4), 9 if k == 1 else 5.5, np.float32)
    m.assert_mlx_claims(model, feeds)
    np.testing.assert_array_equal(m.run_cpu(model, feeds)[0], expected)
    m.assert_matches_ref(model, feeds, [expected], rtol=0, atol=0)


@pytest.mark.parametrize("bits1,bits2", [(8, 4), (4, 8)])
@pytest.mark.parametrize("half_scales", [False, True])
def test_qmoe_projection_bit_overrides(bits1, bits2, half_scales):
    model, feeds = _qmoe_model(E=2, H=4, I=4, K=1, bits=4, act="identity",
                               fusion=0, T=1)
    proto = onnx.load_from_string(model)
    for projection, width in [(1, bits1), (2, bits2)]:
        name = f"fc{projection}w"
        pack = 8 // width
        q = (1 << (width - 1)) + 1
        byte = sum(q << (lane * width) for lane in range(pack))
        feeds[name] = np.full((2, 4, 4 // pack), byte, np.uint8)
        value = next(v for v in proto.graph.input if v.name == name)
        value.type.tensor_type.shape.dim[2].dim_value = 4 // pack
        proto.graph.node[0].attribute.append(helper.make_attribute(f"fc{projection}_expert_weight_bits", width))
        feeds[f"fc{projection}s"].fill(1)
        if half_scales:
            scale_name = f"fc{projection}s"
            feeds[scale_name] = feeds[scale_name].astype(np.float16)
            value = next(v for v in proto.graph.input if v.name == scale_name)
            value.type.tensor_type.elem_type = TensorProto.FLOAT16
    feeds["input"].fill(1)
    feeds["router"][:] = [2, 0]
    model = proto.SerializeToString()
    m.assert_mlx_claims(model, feeds)
    m.assert_matches_ref(model, feeds, [np.full((1, 4), 16, np.float32)], rtol=0, atol=0)


@pytest.mark.parametrize("layout", [1, 2])
def test_matmulnbits_declines_prepacked_layout(layout):
    model, _ = _matmulnbits_model(M=1, K=64, N=4, block=32, asymmetric=False)
    proto = onnx.load_from_string(model)
    proto.graph.node[0].attribute.append(helper.make_attribute("weight_prepacked", layout))
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail, match="fallback|assigned"):
        ort.InferenceSession(proto.SerializeToString(), options, providers=["MLXExecutionProvider"])


def test_qmoe_declines_conflicting_fused_projection_bits():
    model, _ = _qmoe_model(E=2, H=4, I=4, K=1, bits=4, act="swiglu",
                           fusion=1, T=1)
    proto = onnx.load_from_string(model)
    proto.graph.node[0].attribute.append(helper.make_attribute("fc1_expert_weight_bits", 8))
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail, match="fallback|assigned"):
        ort.InferenceSession(proto.SerializeToString(), options, providers=["MLXExecutionProvider"])


def test_matmulnbits_invalid_dynamic_activation_recovers():
    model, feeds = _matmulnbits_model(M=1, K=64, N=4, block=32, asymmetric=False)
    proto = onnx.load_from_string(model)
    dim = proto.graph.input[0].type.tensor_type.shape.dim[2]
    dim.Clear()
    dim.dim_param = "K"
    session = ort.InferenceSession(proto.SerializeToString(), providers=m.EP_PROVIDERS)
    session.disable_fallback()
    good = feeds["a"]
    feeds["a"] = good[..., :-1].copy()
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.EPFail, match="reshape|MatMulNBits"):
        session.run(None, feeds)
    feeds["a"] = good
    np.testing.assert_allclose(session.run(None, feeds)[0], m.run_cpu(model, feeds)[0],
                               rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("vary", ["weight", "scale", "packed_zp", "float_zp"])
def test_matmulnbits_owned_branch_initializer_cache(vary):
    def branch(name, changed):
        byte = 0xAA if changed and vary == "weight" else 0x99
        scale = 2 if changed and vary == "scale" else 1
        zero = 7 if changed and vary in {"packed_zp", "float_zp"} else 8
        zp = (np.full((4, 2), zero, np.float32) if vary == "float_zp"
              else np.full((4, 1), zero * 17, np.uint8))
        inits = [
            numpy_helper.from_array(np.full((4, 2, 16), byte, np.uint8), "b"),
            numpy_helper.from_array(np.full((4, 2), scale, np.float32), "s"),
            numpy_helper.from_array(zp, "zp"),
        ]
        node = helper.make_node("MatMulNBits", ["a", "b", "s", "zp"], ["y"],
                                domain="com.microsoft", K=64, N=4, bits=4, block_size=32)
        return helper.make_graph([node], name, [],
                                 [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
                                 inits)

    node = helper.make_node("If", ["cond"], ["y"],
                            then_branch=branch("then", False), else_branch=branch("else", True))
    model = helper.make_model(
        helper.make_graph([node], "scoped", [
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 64]),
        ], [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])]),
        opset_imports=[helper.make_opsetid("", 24), helper.make_opsetid("com.microsoft", 1)],
        ir_version=11,
    ).SerializeToString()
    feeds = {"cond": np.array(True), "a": np.ones((1, 64), np.float32)}
    m.assert_mlx_claims(model, feeds)
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    session.disable_fallback()
    for cond in [True, False, True, False]:
        feeds["cond"] = np.array(cond)
        np.testing.assert_array_equal(
            session.run(None, feeds)[0],
            np.full((1, 4), 64 if cond else 128, np.float32),
        )


@pytest.mark.parametrize("op", ["QuantizeLinear", "DequantizeLinear"])
@pytest.mark.parametrize("shape,axis", [([], 1), ([3], 0), ([3], -1), (["N"], -1)])
@pytest.mark.parametrize("scalar_vector_param", [False, True])
def test_quant_scalar_and_vector_live_axis(op, shape, axis, scalar_vector_param):
    scalar = shape == []
    quantize = op == "QuantizeLinear"
    values = np.array(4, np.float32) if scalar else np.array([2, 4, 8], np.float32)
    scales = np.array(0.5, np.float32) if scalar else np.array([0.5, 1, 2], np.float32)
    if scalar and scalar_vector_param:
        scales = scales.reshape(1)
    x = values if quantize else values.astype(np.uint8)
    zp = np.zeros(scales.shape, np.uint8)
    node = helper.make_node(op, ["x", "s", "zp"], ["y"], axis=axis)
    model = helper.make_model(
        helper.make_graph([node], "quant", [
            helper.make_tensor_value_info("x", TensorProto.FLOAT if quantize else TensorProto.UINT8, shape),
            helper.make_tensor_value_info("s", TensorProto.FLOAT, list(scales.shape) if scalar or shape != ["N"] else ["N"]),
            helper.make_tensor_value_info("zp", TensorProto.UINT8, list(zp.shape) if scalar or shape != ["N"] else ["N"]),
        ], [helper.make_tensor_value_info("y", TensorProto.UINT8 if quantize else TensorProto.FLOAT, shape)]),
        opset_imports=[helper.make_opsetid("", 24)], ir_version=11,
    ).SerializeToString()
    feeds = {"x": x, "s": scales, "zp": zp}
    m.assert_mlx_claims(model, feeds)
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    session.disable_fallback()
    for factor in [1, 2]:
        feeds["s"] = np.asarray(scales * factor)
        expected = np.rint(x / feeds["s"]).astype(np.uint8) if quantize else x * feeds["s"]
        actual = session.run(None, feeds)[0]
        assert actual.shape == x.shape
        np.testing.assert_array_equal(actual, np.asarray(expected).reshape(x.shape))
    if shape == ["N"]:
        bad = {"x": np.full(1, 4, dtype=x.dtype), "s": np.ones(3, np.float32),
               "zp": np.zeros(3, np.uint8)}
        with pytest.raises(ort.capi.onnxruntime_pybind11_state.EPFail, match="per-axis"):
            session.run(None, bad)
        bad = {"x": np.full(3, 4, dtype=x.dtype), "s": np.ones(3, np.float32),
               "zp": np.zeros(1, np.uint8)}
        with pytest.raises(ort.capi.onnxruntime_pybind11_state.EPFail, match="shapes must match"):
            session.run(None, bad)
        for length in [5, 1, 3]:
            feeds = {"x": np.full(length, 4, dtype=x.dtype),
                     "s": np.full(length, 0.5, np.float32),
                     "zp": np.zeros(length, np.uint8)}
            expected = np.full(length, 8 if quantize else 2, dtype=values.dtype)
            np.testing.assert_array_equal(session.run(None, feeds)[0], expected)
