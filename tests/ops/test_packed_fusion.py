"""Packed decode/consumer equivalence, including routing and original-layout views."""

import json
import os
import subprocess
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from _schema_runtime import assert_mlx_profile, session
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

_reference = pytest.importorskip("onnx.reference.ops.op_dequantize_packed")
FORMATS = [
    "gguf:q4_0", "gguf:q4_1", "gguf:q5_0", "gguf:q5_1", "gguf:q8_0", "gguf:q8_1",
    "gguf:q2_k", "gguf:q3_k", "gguf:q4_k", "gguf:q5_k", "gguf:q6_k",
]


def _weights(name, shape):
    elements, size = _reference.GGUF_BLOCK_SIZES[_reference.GGUF_FORMATS[name]]
    count = int(np.prod(shape)) // elements
    data = np.random.default_rng(42).integers(0, 256, (count, size), dtype=np.uint8)
    offsets = {
        "gguf:q2_k": (80, 82), "gguf:q3_k": (108,), "gguf:q6_k": (208,),
        "gguf:q4_1": (0, 2), "gguf:q5_1": (0, 2), "gguf:q8_1": (0, 2),
        "gguf:q4_k": (0, 2), "gguf:q5_k": (0, 2),
    }.get(name, (0,))
    for offset in offsets:
        scales = (0.001 * (1 + np.arange(count) % (7 if offset == offsets[0] else 5)))
        data[:, offset:offset + 2] = scales.astype("<f2").view(np.uint8).reshape(-1, 2)
    return numpy_helper.from_array(data.reshape(-1), "packed")


def _model(nodes, feeds, initializers, outputs):
    info = helper.make_tensor_value_info
    return helper.make_model(
        helper.make_graph(
            nodes, "packed-consumers",
            [info(name, helper.np_dtype_to_tensor_dtype(value.dtype), list(value.shape))
             for name, value in feeds.items()],
            [info(name, TensorProto.FLOAT, shape) for name, shape in outputs],
            initializers,
        ), opset_imports=[helper.make_opsetid("", 29)], ir_version=11,
    )


def _grouped(name, trans_b, rows):
    block, _ = _reference.GGUF_BLOCK_SIZES[_reference.GGUF_FORMATS[name]]
    k, n, experts, selections = block, block + 13 if trans_b else block * 2, 4, 3
    shape = [experts, n, k] if trans_b else [experts, k, n]
    feeds = {
        "x": np.random.default_rng(0).normal(0, 0.1, (rows, k)).astype(np.float32),
        "ids": np.tile(np.array([[3, 0, 3]], np.int64), (rows, 1)),
        "bias": np.arange(experts * n, dtype=np.float32).reshape(experts, n) * np.float32(0.0007),
    }
    nodes = [
        helper.make_node("DequantizePacked", ["packed"], ["w"], format=name, shape=shape),
        helper.make_node("GroupedMatMul", ["x", "w", "ids", "bias"], ["y"], transB=trans_b),
    ]
    return _model(nodes, feeds, [_weights(name, shape)], [("y", [rows, selections, n])]), feeds


@pytest.mark.parametrize("name", FORMATS)
@pytest.mark.parametrize("trans_b", [0, 1])
@pytest.mark.parametrize("rows", [1, 17, 65])
def test_packed_grouped_matches_reference(name, trans_b, rows, monkeypatch, tmp_path):
    model, feeds = _grouped(name, trans_b, rows)
    expected = ReferenceEvaluator(model).run(None, feeds)[0]
    for disabled in ["0", "1"]:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", disabled)
        runtime = session(model.SerializeToString(), profile=tmp_path / f"grouped-{disabled}")
        actual = runtime.run(None, feeds)[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-4)
        assert_mlx_profile(runtime)


def test_packed_live_bytes_and_dynamic_routing(monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", "0")
    model, feeds = _grouped("gguf:q4_0", 1, 3)
    payload = numpy_helper.to_array(model.graph.initializer[0]).copy()
    model.graph.ClearField("initializer")
    model.graph.input.append(helper.make_tensor_value_info("packed", TensorProto.UINT8, ["bytes"]))
    for value in [model.graph.input[0], model.graph.input[1], model.graph.output[0]]:
        value.type.tensor_type.shape.dim[0].ClearField("dim_value")
        value.type.tensor_type.shape.dim[0].dim_param = "m"
    for value in [model.graph.input[1], model.graph.output[0]]:
        value.type.tensor_type.shape.dim[1].ClearField("dim_value")
        value.type.tensor_type.shape.dim[1].dim_param = "s"
    runtime = session(model.SerializeToString())
    rng = np.random.default_rng(42)
    for m, selections in [(3, 3), (1, 2), (0, 3), (4, 0), (9, 2), (3, 3)]:
        feeds["x"] = rng.normal(0, 0.1, (m, 32)).astype(np.float32)
        feeds["ids"] = np.resize(np.array([3, 0, 3, 1], np.int64), (m, selections))
        feeds["packed"] = payload
        expected = (ReferenceEvaluator(model).run(None, feeds)[0] if m and selections
                    else np.zeros((m, selections, 45), np.float32))
        np.testing.assert_allclose(runtime.run(None, feeds)[0], expected, rtol=2e-5, atol=1e-5)
        payload.reshape(-1, 18)[:, 2] ^= 0x55
    with pytest.raises(Exception, match="input has .* format and shape require"):
        runtime.run(None, {**feeds, "packed": payload[:-1].copy()})


@pytest.mark.parametrize("value", [-1, 4, 2**32, 2**63 - 1, -(2**63)])
def test_packed_grouped_validates_wide_indices(value, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", "0")
    model, feeds = _grouped("gguf:q4_0", 1, 1)
    feeds["ids"][0, 0] = value
    with pytest.raises(Exception, match=r"group_indices\[0\].*outside"):
        session(model.SerializeToString()).run(None, feeds)


@pytest.mark.parametrize("name", FORMATS)
@pytest.mark.parametrize("case", ["MatMul", "MatMulTranspose", "Gemm", "GemmTransA"])
@pytest.mark.parametrize("m", [9, 65])
@pytest.mark.parametrize("aligned", [False, True])
def test_packed_dense_views(name, case, m, aligned, monkeypatch):
    block, _ = _reference.GGUF_BLOCK_SIZES[_reference.GGUF_FORMATS[name]]
    k, n = block, block if aligned else block + 13
    transposed = case != "MatMul"
    shape = [n, k] if transposed else [1, k, block * 2]
    if not transposed:
        n = block * 2
    trans_a = case == "GemmTransA"
    rng = np.random.default_rng(7)
    feeds = {"x": rng.normal(0, 0.1, (k, m) if trans_a else (m, k)).astype(np.float32)}
    nodes = [helper.make_node("DequantizePacked", ["packed"], ["w"], format=name, shape=shape)]
    initializers = [_weights(name, shape)]
    if case == "MatMul":
        initializers.append(numpy_helper.from_array(np.array([-1, n], np.int64), "shape"))
        nodes.append(helper.make_node("Reshape", ["w", "shape"], ["view"]))
    elif case == "MatMulTranspose":
        nodes.append(helper.make_node("Transpose", ["w"], ["view"], perm=[1, 0]))
    if case.startswith("MatMul"):
        nodes.append(helper.make_node("MatMul", ["x", "view"], ["y"]))
    else:
        feeds["bias"] = rng.normal(0, 0.1, (1, n)).astype(np.float32)
        nodes.append(helper.make_node(
            "Gemm", ["x", "w", "bias"], ["y"], transA=int(trans_a), transB=1,
            alpha=0.75, beta=-0.5,
        ))
    model = _model(nodes, feeds, initializers, [("y", [m, n])])
    expected = ReferenceEvaluator(model).run(None, feeds)[0]
    for disabled in ["0", "1"]:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", disabled)
        actual = session(model.SerializeToString()).run(None, feeds)[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-4)


def _decoder(kind):
    k, n = (256 if kind == "packed-reshape" else 64), 64
    feeds = {"x": np.random.default_rng(19).normal(0, 0.1, (1, k)).astype(np.float32)}
    if kind == "qdq":
        initializers = [
            numpy_helper.from_array(np.random.default_rng(42).integers(
                -70, 70, (n, k), dtype=np.int8), "codes"),
            numpy_helper.from_array(np.array(0.03, np.float32), "scale"),
            numpy_helper.from_array(np.array(-3, np.int8), "zero"),
        ]
        nodes = [helper.make_node("DequantizeLinear", ["codes", "scale", "zero"], ["w"])]
    else:
        shape = [1, n, k] if kind == "packed-reshape" else [n, k]
        name = "gguf:q4_k" if kind == "packed-reshape" else "gguf:q4_0"
        initializers = [_weights(name, shape)]
        nodes = [helper.make_node("DequantizePacked", ["packed"], ["decoded"],
                                  format=name, shape=shape)]
        if kind == "packed-reshape":
            initializers.append(numpy_helper.from_array(np.array([n, k], np.int64), "wshape"))
            nodes.append(helper.make_node("Reshape", ["decoded", "wshape"], ["w"]))
        else:
            nodes.append(helper.make_node("Transpose", ["decoded"], ["w"], perm=[1, 0]))
    nodes.append(helper.make_node(
        "MatMul" if kind == "packed-transpose" else "Gemm", ["x", "w"], ["product"],
        **({} if kind == "packed-transpose" else {"transB": 1}),
    ))
    initializers.append(numpy_helper.from_array(np.array([1, 1, n], np.int64), "qshape"))
    nodes.extend([
        helper.make_node("Reshape", ["product", "qshape"], ["q"]),
        helper.make_node("Attention", ["q", "q", "q"], ["y", "pk", "pv"],
                         q_num_heads=1, kv_num_heads=1),
    ])
    model = _model(nodes, feeds, initializers, [
        ("y", [1, 1, n]), ("pk", [1, 1, 1, n]), ("pv", [1, 1, 1, n]),
    ])
    return model, feeds


@pytest.mark.parametrize("kind", ["packed-transpose", "packed-reshape", "qdq"])
def test_decoder_weight_decode(kind, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION", "0")
    model, feeds = _decoder(kind)
    expected = ReferenceEvaluator(model).run(None, feeds)
    runtime = session(model.SerializeToString(), profile=tmp_path / "decoder")
    for actual, golden in zip(runtime.run(None, feeds), expected, strict=True):
        np.testing.assert_allclose(actual, golden, rtol=2e-5, atol=1e-5)
    assert_mlx_profile(runtime)


@pytest.mark.parametrize("kind", ["packed-transpose", "packed-reshape", "qdq"])
def test_decoder_weight_decode_actual_dispatch(kind, tmp_path):
    model, _ = _decoder(kind)
    session(model.SerializeToString())
    trace = tmp_path / "decoder.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", f"{Path(__file__)}::test_decoder_weight_decode[{kind}]",
         "-q", "-p", "no:cacheprovider"], check=True, timeout=180,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
    )
    kernels = {event.get("args", {}).get("kernel")
               for event in json.loads(trace.read_text()) if event.get("cat") == "op.fast"}
    assert ("mlx_qdq_matvec" if kind == "qdq" else "mlx_packed_matvec") in kernels, kernels


@pytest.mark.parametrize("dtype", [TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16])
def test_packed_rounding_and_shared_outputs(dtype, monkeypatch):
    m, k, n = 11, 256, 19
    shape = [n, k]
    feeds = {"x": np.random.default_rng(7).normal(0, 0.1, (m, k)).astype(np.float32)}
    nodes = [
        helper.make_node("DequantizePacked", ["packed"], ["w"],
                         format="gguf:q4_k", shape=shape, output_dtype=dtype),
        helper.make_node("Cast", ["x"], ["a"], to=dtype),
        helper.make_node("Gemm", ["a", "w"], ["product"], transB=1),
        helper.make_node("Cast", ["product"], ["y"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["w", "w"], ["shared"]),
        helper.make_node("Cast", ["shared"], ["visible"], to=TensorProto.FLOAT),
    ]
    payload = _weights("gguf:q4_k", shape)
    model = _model(nodes, feeds, [payload], [("y", [m, n]), ("visible", shape)])
    reference = _model(
        [nodes[0], helper.make_node("Cast", ["w"], ["decoded"], to=TensorProto.FLOAT)],
        {}, [payload], [("decoded", shape)],
    )
    weight = ReferenceEvaluator(reference).run(None, {})[0]
    compute = {TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16,
               TensorProto.BFLOAT16: ml_dtypes.bfloat16}[dtype]
    a = feeds["x"].astype(compute).astype(np.float32)
    expected = (a.astype(np.float64) @ weight.astype(np.float64).T).astype(compute).astype(np.float32)
    shared = (weight.astype(compute) + weight.astype(compute)).astype(np.float32)
    for disabled in ["0", "1"]:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", disabled)
        actual = session(model.SerializeToString()).run(None, feeds)
        np.testing.assert_allclose(actual[0], expected, rtol=1e-5 if dtype == TensorProto.FLOAT else 0.01,
                                   atol=3e-4 if dtype == TensorProto.FLOAT else 0.01)
        np.testing.assert_array_equal(actual[1], shared)


@pytest.mark.parametrize("dtype,step", [(TensorProto.FLOAT16, 2**-11),
                                        (TensorProto.BFLOAT16, 2**-8)])
def test_packed_rounds_weights_before_multiplication(dtype, step, monkeypatch):
    payload = np.zeros(20, np.uint8)
    payload[:4] = np.array([step, 1], dtype="<f2").view(np.uint8)
    payload[5] = 1
    feeds = {"x": np.array([[1, -1, *([0] * 30)]], np.float32)}
    nodes = [
        helper.make_node("DequantizePacked", ["packed"], ["w"],
                         format="gguf:q4_1", shape=[1, 32], output_dtype=dtype),
        helper.make_node("Cast", ["x"], ["a"], to=dtype),
        helper.make_node("Gemm", ["a", "w"], ["product"], transB=1),
        helper.make_node("Cast", ["product"], ["y"], to=TensorProto.FLOAT),
    ]
    model = _model(nodes, feeds, [numpy_helper.from_array(payload, "packed")], [("y", [1, 1])])
    for disabled in ["0", "1"]:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", disabled)
        actual = session(model.SerializeToString()).run(None, feeds)[0]
        # Deferring the weight conversion until after the dot incorrectly returns -step.
        np.testing.assert_array_equal(actual, np.zeros((1, 1), np.float32))


@pytest.mark.parametrize("name,dtype", [("gguf:iq2_s", TensorProto.FLOAT),
                                        ("gguf:q4_0", TensorProto.DOUBLE)])
def test_packed_nonfused_formats_and_double(name, dtype):
    shape = [7, 256]
    feeds = {"x": np.random.default_rng(42).normal(0, 0.1, (3, 256)).astype(np.float32)}
    nodes = [
        helper.make_node("DequantizePacked", ["packed"], ["w"],
                         format=name, shape=shape, output_dtype=dtype),
        helper.make_node("Cast", ["x"], ["a"], to=dtype),
        helper.make_node("Gemm", ["a", "w"], ["product"], transB=1),
        helper.make_node("Cast", ["product"], ["y"], to=TensorProto.FLOAT),
    ]
    model = _model(nodes, feeds, [_weights(name, shape)], [("y", [3, 7])])
    expected = ReferenceEvaluator(model).run(None, feeds)[0]
    actual = session(model.SerializeToString()).run(None, feeds)[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)


@pytest.mark.parametrize("view", [False, True])
def test_packed_selected_embedding_rows(view, monkeypatch):
    name, shape = "gguf:q4_1", [7, 64]
    payload = _weights(name, shape)
    feeds = {"ids": np.array([[6, -7, 6], [1, -2, 0]], np.int64)}
    nodes = [helper.make_node("DequantizePacked", ["packed"], ["w"], format=name, shape=shape)]
    if view:
        nodes.append(helper.make_node("Transpose", ["w"], ["table"], perm=[1, 0]))
        table, result_shape = "table", [2, 3, 7]
    else:
        table, result_shape = "w", [2, 3, 64]
    nodes.append(helper.make_node("Gather", [table, "ids"], ["y"], axis=0))
    model = _model(nodes, feeds, [payload], [("y", result_shape)])
    expected = ReferenceEvaluator(model).run(None, feeds)[0]
    for disabled in ["0", "1"]:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", disabled)
        actual = session(model.SerializeToString()).run(None, feeds)[0]
        np.testing.assert_array_equal(actual, expected)


def test_packed_embedding_preserves_nonfinites_and_signed_zero(monkeypatch):
    payload = np.full((4, 18), 0xF8, np.uint8)
    payload[:, :2] = np.array([-0.0, np.inf, -np.inf, 2**-24], dtype="<f2").view(np.uint8).reshape(-1, 2)
    feeds = {"ids": np.array([3, 0, 2, 1, 0], np.int64)}
    nodes = [
        helper.make_node("DequantizePacked", ["packed"], ["w"],
                         format="gguf:q4_0", shape=[4, 32]),
        helper.make_node("Gather", ["w", "ids"], ["y"]),
    ]
    model = _model(nodes, feeds, [numpy_helper.from_array(payload.reshape(-1), "packed")],
                   [("y", [5, 32])])
    with np.errstate(invalid="ignore"):
        expected = ReferenceEvaluator(model).run(None, feeds)[0]
    for disabled in ["0", "1"]:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", disabled)
        actual = session(model.SerializeToString()).run(None, feeds)[0]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
        zero = expected == 0
        np.testing.assert_array_equal(np.signbit(actual[zero]), np.signbit(expected[zero]))


@pytest.mark.parametrize("value", [7, -8, 2**32, 2**63 - 1])
def test_packed_embedding_validates_wide_indices(value, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", "0")
    feeds = {"ids": np.array([value], np.int64)}
    nodes = [
        helper.make_node("DequantizePacked", ["packed"], ["w"],
                         format="gguf:q4_0", shape=[7, 32]),
        helper.make_node("Gather", ["w", "ids"], ["y"]),
    ]
    model = _model(nodes, feeds, [_weights("gguf:q4_0", [7, 32])], [("y", [1, 32])])
    with pytest.raises(Exception, match=r"indices\[0\].*outside"):
        session(model.SerializeToString()).run(None, feeds)


def test_packed_transpose_then_reshape():
    shape, m = [3, 64, 32], 5
    feeds = {"x": np.random.default_rng(7).normal(0, 0.1, (m, 32)).astype(np.float32)}
    nodes = [
        helper.make_node("DequantizePacked", ["packed"], ["w"],
                         format="gguf:q4_0", shape=shape),
        helper.make_node("Transpose", ["w"], ["t"], perm=[2, 0, 1]),
        helper.make_node("Reshape", ["t", "shape"], ["matrix"]),
        helper.make_node("MatMul", ["x", "matrix"], ["y"]),
    ]
    initializers = [_weights("gguf:q4_0", shape),
                    numpy_helper.from_array(np.array([32, -1], np.int64), "shape")]
    model = _model(nodes, feeds, initializers, [("y", [m, 192])])
    expected = ReferenceEvaluator(model).run(None, feeds)[0]
    actual = session(model.SerializeToString()).run(None, feeds)[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)


def test_packed_grouped_actual_dispatch(tmp_path):
    # Probe the schema in this process before spawning: released runtimes skip honestly.
    model, _ = _grouped("gguf:q4_0", 1, 17)
    session(model.SerializeToString())
    trace = tmp_path / "packed-consumers.json"
    subprocess.run(
        [sys.executable, "-m", "pytest",
         f"{Path(__file__)}::test_packed_grouped_matches_reference[17-1-gguf:q4_0]",
         f"{Path(__file__)}::test_packed_grouped_matches_reference[1-1-gguf:q4_0]",
         f"{Path(__file__)}::test_packed_selected_embedding_rows[False]",
         "-q", "-p", "no:cacheprovider"],
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
        check=True, timeout=180,
    )
    kernels = {event.get("args", {}).get("kernel")
               for event in json.loads(trace.read_text()) if event.get("cat") == "op.fast"}
    assert {"mlx_packed_matvec", "mlx_packed_tiled_matmul", "mlx_packed_gather"} <= kernels, kernels


@pytest.mark.parametrize("name,shape", [("gguf:q4_0", [1031, 64]),
                                       ("gguf:q6_k", [263, 256])])
def test_packed_decode_two_dimensional_grid(name, shape):
    model = _model(
        [helper.make_node("DequantizePacked", ["packed"], ["y"], format=name, shape=shape)],
        {}, [_weights(name, shape)], [("y", shape)],
    )
    expected = ReferenceEvaluator(model).run(None, {})[0]
    np.testing.assert_array_equal(session(model.SerializeToString()).run(None, {})[0], expected)


@pytest.mark.parametrize("dtype,large", [(TensorProto.FLOAT16, False),
                                        (TensorProto.BFLOAT16, False),
                                        (TensorProto.BFLOAT16, True)])
@pytest.mark.parametrize("rows", [17, 65])
def test_packed_prefill_preserves_accumulator_and_bfloat_range(dtype, large, rows, monkeypatch):
    payload = np.full(18, 0xCC, np.uint8)
    payload[:2] = np.array([16384 if large else 1], dtype="<f2").view(np.uint8)
    x = np.full((rows, 32), 131072 if large else 16384, np.float32)
    if not large:
        x[:, 16:] *= -1
    feeds = {"x": x}
    nodes = [
        helper.make_node("DequantizePacked", ["packed"], ["w"],
                         format="gguf:q4_0", shape=[1, 1, 32], output_dtype=dtype),
        helper.make_node("Cast", ["x"], ["a"], to=dtype),
        helper.make_node("GroupedMatMul", ["a", "w", "ids"], ["product"], transB=1),
        helper.make_node("Cast", ["product"], ["y"], to=TensorProto.FLOAT),
    ]
    feeds["ids"] = np.zeros((rows, 1), np.int64)
    model = _model(nodes, feeds, [numpy_helper.from_array(payload, "packed")], [("y", [rows, 1, 1])])
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", "0")
    expected = np.full((rows, 1, 1), 2**38 if large else 0, np.float32)
    np.testing.assert_array_equal(session(model.SerializeToString()).run(None, feeds)[0], expected)


def _conv(spatial, dtype, fallback=None, *, view=False):
    group = 2 if fallback == "group" else 1
    kernel = [3] * len(spatial) if fallback == "spatial" else [1] * len(spatial)
    shape = [7 if group == 1 else 8, 32, *kernel]
    packed_shape = [shape[0], int(np.prod(shape[1:]))]
    decoded_shape = [shape[0], *([1] * len(spatial)), 32] if view else packed_shape
    payload = _weights("gguf:q4_0", packed_shape)
    attrs = {"group": group}
    if fallback == "stride":
        attrs["strides"] = [2] * len(spatial)
    elif fallback == "padding":
        attrs["pads"] = [1] * (2 * len(spatial))
    feeds = {
        "x": np.random.default_rng(51).normal(
            0, 0.1, (2, 32 * group, *spatial)).astype(np.float32),
        "bias": np.random.default_rng(11).normal(0, 0.1, (shape[0],)).astype(np.float32),
    }
    nodes = [
        helper.make_node("DequantizePacked", ["packed"], ["w"],
                         format="gguf:q4_0", shape=decoded_shape, output_dtype=dtype),
        (helper.make_node("Transpose", ["w"], ["conv_w"],
                          perm=[0, len(spatial) + 1, *range(1, len(spatial) + 1)])
         if view else helper.make_node("Reshape", ["w", "shape"], ["conv_w"])),
        helper.make_node("Cast", ["x"], ["a"], to=dtype),
        helper.make_node("Cast", ["bias"], ["b"], to=dtype),
        helper.make_node("Conv", ["a", "conv_w", "b"], ["product"], **attrs),
        helper.make_node("Cast", ["product"], ["y"], to=TensorProto.FLOAT),
    ]
    initializers = [payload, numpy_helper.from_array(np.array(shape, np.int64), "shape")]
    model = _model(nodes, feeds, initializers, [("y", [2, shape[0], *([None] * len(spatial))])])
    weight = ReferenceEvaluator(_model(
        [nodes[0], helper.make_node("Cast", ["w"], ["decoded"], to=TensorProto.FLOAT)],
        {}, [payload], [("decoded", decoded_shape)],
    )).run(None, {})[0].reshape(shape)
    compute = {TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16,
               TensorProto.BFLOAT16: ml_dtypes.bfloat16}[dtype]
    a, b = (feeds[name].astype(compute).astype(np.float32) for name in ["x", "bias"])
    # The ONNX reference does not implement BF16 Conv; decode/round independently,
    # then use its Float64 spatial contraction and round the graph's product once.
    reference = _model(
        [helper.make_node("Conv", ["a", "weight", "bias"], ["product"], **attrs)],
        {"a": a.astype(np.float64), "weight": weight.astype(np.float64),
         "bias": b.astype(np.float64)}, [], [("product", [2, shape[0], *([None] * len(spatial))])],
    )
    reference.graph.output[0].type.tensor_type.elem_type = TensorProto.DOUBLE
    # The EP, like the unfused MLX path, rounds the product before adding bias.
    no_bias = ReferenceEvaluator(reference).run(None, {
        "a": a.astype(np.float64), "weight": weight.astype(np.float64),
        "bias": np.zeros_like(b, dtype=np.float64),
    })[0]
    expected = (no_bias.astype(compute).astype(np.float32)
                + b.reshape((1, -1, *([1] * len(spatial))))).astype(compute).astype(np.float32)
    return model, feeds, expected


@pytest.mark.parametrize("spatial", [[19], [3, 7], [2, 3, 4]])
@pytest.mark.parametrize("dtype", [TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16])
@pytest.mark.parametrize("view", [False, True], ids=["reshape", "transpose"])
def test_packed_pointwise_conv(spatial, dtype, view, monkeypatch, tmp_path):
    model, feeds, expected = _conv(spatial, dtype, view=view)
    for disabled in ["0", "1"]:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", disabled)
        runtime = session(model.SerializeToString(), profile=tmp_path / f"conv-{disabled}")
        actual = runtime.run(None, feeds)[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-5 if dtype == TensorProto.FLOAT else 0.01,
                                   atol=2e-5 if dtype == TensorProto.FLOAT else 0.002)
        assert_mlx_profile(runtime)


@pytest.mark.parametrize("fallback", ["group", "stride", "padding", "spatial"])
def test_packed_conv_unsupported_profiles_use_mlx_conv(fallback, monkeypatch, tmp_path):
    model, feeds, expected = _conv([5, 7], TensorProto.FLOAT, fallback)
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", "0")
    runtime = session(model.SerializeToString(), profile=tmp_path / "conv-fallback")
    actual = runtime.run(None, feeds)[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert_mlx_profile(runtime)


def test_packed_pointwise_conv_actual_dispatch(tmp_path):
    model, _, _ = _conv([19], TensorProto.FLOAT)
    session(model.SerializeToString())
    trace = tmp_path / "conv.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", f"{Path(__file__)}::test_packed_pointwise_conv",
         "-q", "-p", "no:cacheprovider"], check=True, timeout=180,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
    )
    kernels = {event.get("args", {}).get("kernel")
               for event in json.loads(trace.read_text()) if event.get("cat") == "op.fast"}
    assert "mlx_packed_pointwise_conv" in kernels, kernels


def _constant_prefill(trans_b):
    n = 37 if trans_b else 64
    shape = [n, 32] if trans_b else [32, n]
    feeds = {"x": np.zeros((65, 32), np.float32)}
    nodes = [
        helper.make_node("DequantizePacked", ["packed"], ["w"], format="gguf:q4_0", shape=shape),
        helper.make_node("Gemm", ["x", "w"], ["y"], transB=trans_b),
    ]
    model = _model(nodes, feeds, [_weights("gguf:q4_0", shape)], [("y", [None, n])])
    model.graph.input[0].type.tensor_type.shape.dim[0].ClearField("dim_value")
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "m"
    return model, feeds


@pytest.mark.parametrize("trans_b", [0, 1])
def test_packed_constant_prefill_replays_values_and_shapes(trans_b, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", "0")
    model, feeds = _constant_prefill(trans_b)
    shape = [37, 32] if trans_b else [32, 64]
    reference = _model([model.graph.node[0]], {}, [model.graph.initializer[0]], [("w", shape)])
    weight = ReferenceEvaluator(reference).run(None, {})[0]
    matrix = weight.T if trans_b else weight
    runtime = session(model.SerializeToString())
    rng = np.random.default_rng(82)
    for rows in [65, 9, 32, 33, 65]:
        feeds["x"] = rng.normal(0, 0.1, (rows, 32)).astype(np.float32)
        expected = (feeds["x"].astype(np.float64) @ matrix.astype(np.float64)).astype(np.float32)
        np.testing.assert_allclose(runtime.run(None, feeds)[0], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("trans_b", [0, 1])
def test_packed_constant_prefill_actual_dispatch(trans_b, tmp_path):
    model, _ = _constant_prefill(trans_b)
    session(model.SerializeToString())
    trace = tmp_path / "constant-prefill.json"
    subprocess.run(
        [sys.executable, "-m", "pytest",
         f"{Path(__file__)}::test_packed_constant_prefill_replays_values_and_shapes[{trans_b}]",
         "-q", "-p", "no:cacheprovider"], check=True, timeout=180,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
    )
    kernels = {event.get("args", {}).get("kernel")
               for event in json.loads(trace.read_text()) if event.get("cat") == "op.fast"}
    assert "mlx_packed_quantized_matmul" in kernels, kernels


@pytest.mark.parametrize("name", [name for name in FORMATS
                                 if name not in {"gguf:q2_k", "gguf:q3_k", "gguf:q6_k"}])
@pytest.mark.parametrize("exceptional", [False, True], ids=["signed-zero", "nonfinite"])
def test_packed_prefill_preserves_decode_classes_and_shared_zeros(name, exceptional, monkeypatch):
    block, size = _reference.GGUF_BLOCK_SIZES[_reference.GGUF_FORMATS[name]]
    shape = [37, block]
    payload = numpy_helper.to_array(_weights(name, shape)).copy().reshape(37, size)
    parameters = [0.0, -0.0] if not exceptional else [0.0, -0.0, np.inf, -np.inf, np.nan]
    d = np.resize(np.array(parameters, dtype="<f2"), 37)
    payload[:, :2] = d.view(np.uint8).reshape(37, 2)
    if name in {"gguf:q4_1", "gguf:q5_1", "gguf:q4_k", "gguf:q5_k"}:
        payload[:, 2:4] = (-d).view(np.uint8).reshape(37, 2)
    elif name == "gguf:q8_1":
        # The stored dot-product sum is not part of DequantizePacked decoding.
        payload[:, 2:4] = np.array([np.nan], dtype="<f2").view(np.uint8)
    feeds = {"x": np.ones((65, block), np.float32)}
    nodes = [
        helper.make_node("DequantizePacked", ["packed"], ["w"], format=name, shape=shape),
        helper.make_node("Gemm", ["x", "w"], ["y"], transB=1),
    ]
    model = _model(nodes, feeds, [numpy_helper.from_array(payload.ravel(), "packed")],
                   [("y", [65, 37]), ("w", shape)])
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", "0")
    runtime = session(model.SerializeToString())
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", "1")
    baseline = session(model.SerializeToString())
    with np.errstate(invalid="ignore"):
        weight = ReferenceEvaluator(_model(
            [nodes[0]], {}, [model.graph.initializer[0]], [("w", shape)],
        )).run(None, {})[0]
        for sign in [1, -1]:
            feeds["x"].fill(sign)
            expected = (feeds["x"].astype(np.float64) @ weight.astype(np.float64).T).astype(np.float32)
            monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", "1")
            unfused = baseline.run(None, feeds)[0]
            monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", "0")
            actual, decoded = runtime.run(None, feeds)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=True)
            np.testing.assert_array_equal(decoded, weight)
            np.testing.assert_array_equal(np.signbit(decoded[weight == 0]), np.signbit(weight[weight == 0]))
            # BLAS dot initialization differs on signed zero. Preserve the
            # unfused MLX contraction while checking decoded signs independently.
            np.testing.assert_array_equal(np.signbit(actual[expected == 0]), np.signbit(unfused[expected == 0]))


@pytest.mark.parametrize("rows", [32, 33, 65])
@pytest.mark.parametrize("code,scale", [(9, 1), (15, 65504)], ids=["unit-weight", "large-weight"])
def test_packed_native_prefill_preserves_float32_activation_range_and_classes(rows, code, scale, monkeypatch):
    payload = np.full((2, 18), code | (code << 4), np.uint8)
    payload[:, :2] = np.array([scale], dtype="<f2").view(np.uint8)
    payload[1, 2:] = 0x88
    x = np.ones((rows, 32), np.float32)
    x[0, 0] = np.nan
    x[1, 0] = np.inf
    x[2, 0] = -np.inf
    x[3, :2] = [np.inf, -np.inf]
    x[4].fill(2**100)
    feeds = {"x": x}
    nodes = [
        helper.make_node("DequantizePacked", ["packed"], ["w"], format="gguf:q4_0", shape=[2, 32]),
        helper.make_node("Gemm", ["x", "w"], ["y"], transB=1),
    ]
    model = _model(nodes, feeds, [numpy_helper.from_array(payload.ravel(), "packed")],
                   [("y", [rows, 2])])
    weight = ReferenceEvaluator(_model(
        [nodes[0]], {}, [model.graph.initializer[0]], [("w", [2, 32])],
    )).run(None, {})[0]
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION", "0")
    with np.errstate(invalid="ignore"):
        expected = (x.astype(np.float64) @ weight.astype(np.float64).T).astype(np.float32)
        np.testing.assert_array_equal(session(model.SerializeToString()).run(None, feeds)[0], expected)
