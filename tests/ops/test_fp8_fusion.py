"""FP8 DequantizeLinear transport, GPU decode, and fused weight products.

References decode with ml_dtypes, apply the ONNX per-element ``(x - zero) * scale`` rounding in
float32, and accumulate in float64.
"""

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from _schema_runtime import session
from onnx import TensorProto, helper, numpy_helper

ml_dtypes = pytest.importorskip("ml_dtypes")

FORMATS = {
    "e4m3fn": (ml_dtypes.float8_e4m3fn, TensorProto.FLOAT8E4M3FN),
    "e4m3fnuz": (ml_dtypes.float8_e4m3fnuz, TensorProto.FLOAT8E4M3FNUZ),
    "e5m2": (ml_dtypes.float8_e5m2, TensorProto.FLOAT8E5M2),
    "e5m2fnuz": (ml_dtypes.float8_e5m2fnuz, TensorProto.FLOAT8E5M2FNUZ),
}
OUTPUTS = {
    np.float32: TensorProto.FLOAT,
    np.float16: TensorProto.FLOAT16,
    ml_dtypes.bfloat16: TensorProto.BFLOAT16,
}

def _block_product_model(kind, dtype, shape, n, block, *, bias=True, activation_scale=False,
                         one_hot=False, shared=False):
    """CUDA contrib contract and its ONNX equivalent, including each rounding boundary."""
    rng = np.random.default_rng(57)
    k = shape[-1]
    values = rng.normal(0, 1, (n, k)).astype(ml_dtypes.float8_e4m3fn)
    scales = rng.uniform(0.013, 0.19, (n, (k + block - 1) // block)).astype(np.float32)
    x = rng.normal(0, 1, shape).astype(np.float32)
    if one_hot:
        x.fill(0)
        x[..., 0] = 1
    low = OUTPUTS[dtype]
    b = rng.normal(0, 0.2, n).astype(dtype)
    nodes = [helper.make_node("Cast", ["x"], ["a"], to=low)]
    initializers = [_initializer("w", values.view(np.uint8), TensorProto.FLOAT8E4M3FN),
                    _initializer("ws", scales)]
    a = x.astype(dtype).astype(np.float32)
    if activation_scale:
        scale = np.array(0.25, np.float32)
        initializers.append(_initializer("as", scale))
        a = ((a / scale).astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
             * scale).astype(dtype).astype(np.float32)
    if bias:
        initializers.append(numpy_helper.from_array(b, "bias"))
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)]
    if kind == "contrib":
        nodes.append(helper.make_node(
            "MatMulBlockQuantizedFp8Weight",
            ["a", "w", "ws", "as" if activation_scale else "", "bias" if bias else ""],
            ["low_y"], domain="com.microsoft", **({"block_size": block} if block != 128 else {})))
    else:
        nodes += [helper.make_node("DequantizeLinear", ["w", "ws"], ["wf"], axis=1,
                                   block_size=block),
                  helper.make_node("Cast", ["wf"], ["wl"], to=low)]
        activation = "a"
        if activation_scale:
            nodes += [helper.make_node("Cast", ["a"], ["af"], to=TensorProto.FLOAT),
                      helper.make_node("QuantizeLinear", ["af", "as"], ["aq"],
                                       output_dtype=TensorProto.FLOAT8E4M3FN),
                      helper.make_node("DequantizeLinear", ["aq", "as"], ["ad"]),
                      helper.make_node("Cast", ["ad"], ["al"], to=low)]
            activation = "al"
        if kind == "gemm":
            nodes.append(helper.make_node("Gemm",
                [activation, "wl", *(["bias"] if bias else [])], ["low_y"], transB=1))
        else:
            nodes += [helper.make_node("Transpose", ["wl"], ["wt"], perm=[1, 0]),
                      helper.make_node("MatMul", [activation, "wt"],
                                       ["product" if bias else "low_y"])]
            if bias:
                nodes.append(helper.make_node("Add", ["product", "bias"], ["low_y"]))
        if shared:
            nodes.append(helper.make_node("Cast", ["wl"], ["weight_out"], to=TensorProto.FLOAT))
            outputs.append(helper.make_tensor_value_info("weight_out", TensorProto.FLOAT, None))
    nodes.append(helper.make_node("Cast", ["low_y"], ["y"], to=TensorProto.FLOAT))
    graph = helper.make_graph(nodes, "fp8_block_product",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape))], outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21),
                                                  helper.make_opsetid("com.microsoft", 1)],
                              ir_version=10)
    weight = (values.astype(np.float32) * np.repeat(scales, block, axis=1)[:, :k]).astype(dtype)
    expected = (a.astype(np.float64) @ weight.astype(np.float64).T).astype(dtype)
    if bias:
        expected = (expected.astype(np.float32) + b.astype(np.float32)).astype(dtype)
    return model.SerializeToString(), {"x": x}, expected.astype(np.float32), weight.astype(np.float32)


@pytest.mark.parametrize("dtype", [np.float16, ml_dtypes.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("kind", ["contrib", "gemm", "matmul"])
@pytest.mark.parametrize("shape,block,bias,activation_scale", [
    ((1, 256), 128, True, False),
    ((3, 259), 128, False, True),
    ((20, 131), 32, True, True),
])
def test_block_fp8_products(dtype, kind, shape, block, bias, activation_scale, monkeypatch):
    model, feeds, expected, weight = _block_product_model(
        kind, dtype, shape, 17, block, bias=bias, activation_scale=activation_scale,
        shared=kind != "contrib")
    for disabled in ["0", "1"]:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION", disabled)
        result = session(model)
        actual = result.run(None, feeds)
        np.testing.assert_allclose(actual[0], expected,
            rtol=0.02 if dtype is ml_dtypes.bfloat16 else 0.003, atol=0.003)
        if kind != "contrib":
            np.testing.assert_array_equal(actual[1], weight)
        # Changed values replay through the same session without capturing stale activation bytes.
        zero_feeds = {"x": np.zeros_like(feeds["x"])}
        replay = result.run(None, zero_feeds)[0]
        reference = session(model).run(None, zero_feeds)[0]
        np.testing.assert_array_equal(replay, reference)


@pytest.mark.parametrize("dtype", [np.float16, ml_dtypes.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("kind", ["contrib", "matmul"])
def test_block_fp8_rounding_and_batched(dtype, kind):
    model, feeds, expected, _ = _block_product_model(
        kind, dtype, (2, 3, 256), 33, 128, bias=False, one_hot=True)
    actual = session(model).run(None, feeds)[0]
    # A one-hot product exposes the weight Cast exactly (not just matmul tolerance).
    np.testing.assert_array_equal(actual, expected)


def test_fp8_activation_quantization_rounding():
    magnitudes = np.arange(127, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    midpoints = (magnitudes[:-1] + magnitudes[1:]) * np.float32(0.5)
    values = np.concatenate([magnitudes, midpoints,
        np.nextafter(midpoints, np.float32(-np.inf)),
        np.nextafter(midpoints, np.float32(np.inf)), [0.0, np.inf, np.nan]]).astype(np.float32)
    values = np.concatenate([values, -values])
    graph = helper.make_graph(
        [helper.make_node("QuantizeLinear", ["x", "s"], ["q"],
                          output_dtype=TensorProto.FLOAT8E4M3FN),
         helper.make_node("DequantizeLinear", ["q", "s"], ["y"])],
        "fp8_activation_rounding",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(values.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(values.shape)),
         helper.make_tensor_value_info("q", TensorProto.FLOAT8E4M3FN, list(values.shape))],
        [numpy_helper.from_array(np.array(1, np.float32), "s")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    expected = np.clip(values, -448, 448).astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    result = session(model.SerializeToString())
    actual = result.run(["y"], {"x": values})[0]
    np.testing.assert_array_equal(actual, expected)
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(np.signbit(actual[finite]), np.signbit(expected[finite]))
    q = result.run_with_ort_values(["q"], {"x": ort.OrtValue.ortvalue_from_numpy(values)})[0]
    bits = np.frombuffer(ctypes.string_at(q.data_ptr(), values.size), np.uint8)
    expected_bits = np.clip(values, -448, 448).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    np.testing.assert_array_equal(bits[finite], expected_bits[finite])
    np.testing.assert_array_equal(bits[~finite] & 0x7F, 0x7F)


@pytest.mark.parametrize("kind", ["contrib", "gemm", "matmul"])
@pytest.mark.parametrize("rows,path", [(1, "mlx_fp8_gemv_k"), (20, "mlx_fp8_dequant_matmul")])
def test_block_fp8_dispatch(tmp_path, kind, rows, path):
    code = """
import os
import onnxruntime as ort
from bench.fp8_block_products import model
ort.register_execution_provider_library("MLXExecutionProvider", os.environ["ONNXRUNTIME_MLX_EP_LIB"])
data, feeds = model(KIND, "bf16", ROWS, 256, 33, 128, True)
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
session = ort.InferenceSession(data, options, providers=["MLXExecutionProvider"], enable_fallback=False)
session.run(None, feeds)
""".replace("KIND", repr(kind)).replace("ROWS", str(rows))
    for disabled in [False, True]:
        trace = tmp_path / f"block_{disabled}.json"
        env = {**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
               "ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION": str(int(disabled))}
        subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=180)
        kernels = {e.get("args", {}).get("kernel") for e in json.loads(trace.read_text())
                   if e.get("cat") == "op.fast"}
        assert (path in kernels) == (kind == "contrib" or not disabled), kernels
        assert not {k for k in kernels if k and k.startswith("mlx_qdq")}, kernels


def _initializer(name, array, onnx_type=None):
    if onnx_type is None:
        return numpy_helper.from_array(array, name)
    return helper.make_tensor(name, onnx_type, array.shape, array.reshape(-1).view(np.uint8).tobytes(),
                              raw=True)


def _decode(bits, fmt):
    return bits.view(FORMATS[fmt][0]).astype(np.float32)


def _weights(rng, shape, fmt, special=False):
    """Finite FP8 weights; ``special`` mixes in subnormals, signed zeros, and the max."""
    values = rng.normal(0, 1, shape).astype(FORMATS[fmt][0])
    bits = values.view(np.uint8).copy()
    if special:
        finfo = ml_dtypes.finfo(FORMATS[fmt][0])
        extremes = np.array([0.0, finfo.max, -finfo.max, finfo.smallest_subnormal,
                             -finfo.smallest_subnormal, finfo.smallest_normal],
                            np.float32).astype(FORMATS[fmt][0]).view(np.uint8)
        flat = bits.reshape(-1)
        flat[: extremes.size * 3: 3] = extremes[: len(flat[: extremes.size * 3: 3])]
        if "uz" not in fmt:
            flat[1] = 0x80  # negative zero
    return bits


def _expand(value, shape, axis, block):
    if value.ndim == 0:
        return value
    if block:
        value = np.repeat(value, block, axis=axis)
        return value[tuple(slice(0, d) for d in shape)]
    view = [1] * len(shape)
    view[axis] = shape[axis]
    return value.reshape(view)


def _product_model(fmt, case, layout, m, k, n, *, batch=None, shared=False, special=False,
                   zero=True, edit=None):
    rng = np.random.default_rng(29)
    trans_a = case == "GemmTransA"
    trans_b = case.startswith("Gemm")
    weight_shape = (n, k) if trans_b else (k, n)
    k_axis = 1 if trans_b else 0
    bits = _weights(rng, weight_shape, fmt, special)
    if edit:
        edit(bits)
    if layout == "tensor":
        axis, block, scale = 0, 0, np.array(0.0371, np.float32)
    elif layout == "axis":
        axis = 1 - k_axis
        block = 0
        scale = rng.uniform(0.01, 0.2, weight_shape[axis]).astype(np.float32)
    else:
        axis, block = k_axis, 32
        scale_shape = list(weight_shape)
        scale_shape[k_axis] = (k + block - 1) // block
        scale = rng.uniform(0.01, 0.2, scale_shape).astype(np.float32)
    zero_bits = np.zeros(scale.shape, np.uint8)
    weight = (_decode(bits, fmt) - _expand(_decode(zero_bits, fmt), weight_shape, axis, block)
              ) * _expand(scale, weight_shape, axis, block)
    x_shape = (k, m) if trans_a else (*(batch or ()), m, k)
    x = rng.normal(0, 1, x_shape).astype(np.float32)
    inputs = ["w", "ws", "wz"] if zero else ["w", "ws"]
    attrs = {"axis": axis, **({"block_size": block} if block else {})}
    nodes = [helper.make_node("DequantizeLinear", inputs, ["wf"], **attrs)]
    initializers = [_initializer("w", bits, FORMATS[fmt][1]), _initializer("ws", scale)]
    if zero:
        initializers.append(_initializer("wz", zero_bits, FORMATS[fmt][1]))
    feeds = {"x": x}
    graph_inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))]
    dense = weight.T if trans_b else weight
    with np.errstate(invalid="ignore"):
        product = ((x.T if trans_a else x).astype(np.float64)
                   @ dense.astype(np.float64)).astype(np.float32)
    if trans_b:
        c = rng.normal(0, 0.2, (1, n)).astype(np.float32)
        feeds["c"] = c
        graph_inputs.append(helper.make_tensor_value_info("c", TensorProto.FLOAT, [1, n]))
        nodes.append(helper.make_node("Gemm", ["x", "wf", "c"], ["y"], transA=int(trans_a),
                                      transB=1, alpha=0.75, beta=-0.25))
        expected = np.float32(0.75) * product + np.float32(-0.25) * c
    else:
        nodes.append(helper.make_node("MatMul", ["x", "wf"], ["y"]))
        expected = product
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)]
    if shared:
        nodes.append(helper.make_node("Add", ["wf", "wf"], ["twice"]))
        outputs += [helper.make_tensor_value_info("wf", TensorProto.FLOAT, None),
                    helper.make_tensor_value_info("twice", TensorProto.FLOAT, None)]
    graph = helper.make_graph(nodes, "fp8", graph_inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    return model.SerializeToString(), feeds, expected, weight


def _check(result, expected, x_scale):
    # Float32 accumulation versus the float64 reference.
    atol = 4e-6 * x_scale
    np.testing.assert_allclose(result, expected, rtol=2e-5, atol=atol)


@pytest.mark.parametrize("output", list(OUTPUTS), ids=["float", "float16", "bfloat16"])
@pytest.mark.parametrize("fmt", list(FORMATS))
def test_standalone_decode_all_bit_patterns(fmt, output):
    bits = np.arange(256, dtype=np.uint8).reshape(16, 16)
    scale = np.array(0.5, np.float32).astype(output)
    graph = helper.make_graph(
        # ORT Python cannot return bfloat16; the widening Cast is exact.
        [helper.make_node("DequantizeLinear", ["w", "s"], ["d"]),
         helper.make_node("Cast", ["d"], ["y"], to=TensorProto.FLOAT)],
        "fp8_dq", [],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [_initializer("w", bits, FORMATS[fmt][1]),
         _initializer("s", scale, OUTPUTS[output] if output is ml_dtypes.bfloat16 else None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    (result,) = session(model.SerializeToString()).run(None, {})
    expected = (_decode(bits, fmt).astype(output) * scale).astype(output)
    expected = expected.astype(np.float32)
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(result[finite], expected[finite])
    np.testing.assert_array_equal(np.signbit(result[finite]), np.signbit(expected[finite]))


PRODUCT_CASES = [
    # (case, layout, m, k, n, batch)
    ("Gemm", "axis", 1, 256, 96, None),
    ("Gemm", "blocked", 3, 200, 40, None),
    ("Gemm", "tensor", 8, 64, 33, None),
    ("Gemm", "blocked", 20, 96, 48, None),
    ("Gemm", "axis", 2, 67, 7, None),
    ("GemmTransA", "blocked", 3, 64, 40, None),
    ("GemmTransA", "axis", 12, 64, 40, None),
    ("MatMul", "axis", 1, 256, 128, None),
    ("MatMul", "blocked", 4, 1000, 132, None),
    ("MatMul", "tensor", 8, 64, 256, None),
    ("MatMul", "blocked", 24, 80, 64, None),
    ("MatMul", "axis", 3, 33, 7, None),
    ("MatMul", "blocked", 3, 64, 64, (2,)),
]


def _case_id(case, layout, m, k, n, batch):
    return f"{case}-{layout}-m{m}-k{k}-n{n}" + ("-batched" if batch else "")


PRODUCT_IDS = [_case_id(*c) for c in PRODUCT_CASES]


@pytest.mark.parametrize("case,layout,m,k,n,batch", PRODUCT_CASES, ids=PRODUCT_IDS)
@pytest.mark.parametrize("fmt", list(FORMATS))
def test_fused_product(fmt, case, layout, m, k, n, batch, monkeypatch):
    model, feeds, expected, weight = _product_model(fmt, case, layout, m, k, n, batch=batch,
                                                    shared=True, special=True)
    x_scale = float(np.abs(feeds["x"]).max() * np.abs(weight).sum(axis=-2 if case == "MatMul"
                                                                   else -1).max())
    for disabled in ["0", "1"]:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION", disabled)
        y, wf, twice = session(model).run(None, feeds)
        _check(y, expected, x_scale)
        np.testing.assert_array_equal(wf, weight)
        np.testing.assert_array_equal(twice, weight + weight)


def test_fused_nonfinite_weights():
    def edit(bits):
        bits[1, 5], bits[3, 9], bits[3, 10], bits[5, 0] = 0x7F, 0x7C, 0xFC, 0x7C

    model, feeds, expected, _ = _product_model("e5m2", "Gemm", "axis", 2, 64, 8, edit=edit)
    y = session(model).run(None, feeds)[0]
    np.testing.assert_array_equal(np.isnan(y), np.isnan(expected))
    assert np.isnan(expected).any() and np.isinf(expected).any()
    np.testing.assert_array_equal(y[np.isinf(expected)], expected[np.isinf(expected)])
    finite = np.isfinite(expected)
    np.testing.assert_allclose(y[finite], expected[finite], rtol=2e-5, atol=1e-4)


@pytest.mark.parametrize("case,layout,m,k,n,batch", PRODUCT_CASES, ids=PRODUCT_IDS)
def test_fused_replay(case, layout, m, k, n, batch):
    model, feeds, expected, weight = _product_model("e4m3fn", case, layout, m, k, n, batch=batch)
    _check(session(model).run(None, feeds)[0], expected, float(np.abs(weight).sum()))


ACTIVATION_CASES = ["MatMul", "Gemm", "GroupedMatMul"]
ACTIVATION_WEIGHTS = {"int8": (np.int8, TensorProto.INT8), "uint8": (np.uint8, TensorProto.UINT8),
                      "e4m3fn": FORMATS["e4m3fn"]}


def _activation_model(fmt, case, weight_type):
    """FP8 activation DQ feeding a product whose weight is also DequantizeLinear."""
    rng = np.random.default_rng(3)
    m, k, n = 3, 32, 8
    x_bits = _weights(rng, (m, k), fmt)
    one = FORMATS[fmt][0]
    # +-1 codes (E4M3FN 0x38/0xB8) are +56/+184 if read as integers.
    x_bits[:, :2] = np.array([1, -1], np.float32).astype(one).view(np.uint8)
    x_scale, x_zero = np.float32(0.5), np.array(-2, np.float32).astype(one).view(np.uint8)
    x = (_decode(x_bits, fmt) - _decode(x_zero, fmt)) * x_scale
    w_np, w_onnx = ACTIVATION_WEIGHTS[weight_type]
    grouped = case == "GroupedMatMul"
    trans_b = case != "MatMul"
    shape = (2, n, k) if grouped else (n, k) if trans_b else (k, n)
    if weight_type in FORMATS:
        w_q = _weights(rng, shape, weight_type)
        w_real = _decode(w_q, weight_type)
    else:
        info = np.iinfo(w_np)
        w_q = rng.integers(info.min, info.max + 1, shape).astype(w_np)
        w_real = w_q.astype(np.float32)
    w_scale, w_zero = np.float32(0.03), np.zeros((), np.uint8).astype(w_q.dtype)
    weight = (w_real - (0 if weight_type in FORMATS else w_zero.astype(np.float32))) * w_scale
    nodes = [
        helper.make_node("DequantizeLinear", ["xq", "xs", "xz"], ["xf"]),
        helper.make_node("DequantizeLinear", ["w", "ws", "wz"], ["wf"]),
    ]
    feeds_info = [helper.make_tensor_value_info("xq", FORMATS[fmt][1], [m, k])]
    feeds = {}
    if grouped:
        ids = rng.integers(0, 2, (m, 2)).astype(np.int64)
        feeds["ids"] = ids
        feeds_info.append(helper.make_tensor_value_info("ids", TensorProto.INT64, [m, 2]))
        nodes.append(helper.make_node("GroupedMatMul", ["xf", "wf", "ids"], ["y"], transB=1))
        expected = np.einsum("mk,msnk->msn", x.astype(np.float64), weight[ids].astype(np.float64))
        opset, ir = 29, 11
    else:
        dense = weight.T if trans_b else weight
        nodes.append(helper.make_node(case, ["xf", "wf"], ["y"],
                                      **({"transB": 1} if trans_b else {})))
        expected = x.astype(np.float64) @ dense.astype(np.float64)
        opset, ir = 21, 10
    graph = helper.make_graph(
        nodes, "fp8_activation", feeds_info,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [_initializer("xs", np.array(x_scale)),
         _initializer("xz", x_zero.reshape(()), FORMATS[fmt][1]),
         _initializer("w", w_q, w_onnx if weight_type in FORMATS else None),
         _initializer("ws", np.array(w_scale)),
         _initializer("wz", w_zero.reshape(()), w_onnx if weight_type in FORMATS else None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)],
                              ir_version=ir)
    xq = ort.OrtValue.ortvalue_from_shape_and_type([m, k], FORMATS[fmt][1])
    ctypes.memmove(xq.data_ptr(), x_bits.ctypes.data, x_bits.nbytes)
    feeds["xq"] = xq
    return model.SerializeToString(), feeds, expected.astype(np.float32)


@pytest.mark.parametrize("weight_type", list(ACTIVATION_WEIGHTS))
@pytest.mark.parametrize("case", ACTIVATION_CASES)
@pytest.mark.parametrize("fmt", ["e4m3fn", "e5m2"])
def test_fp8_activation_not_fused(fmt, case, weight_type, monkeypatch):
    model, feeds, expected = _activation_model(fmt, case, weight_type)
    for disabled in ["0", "1"]:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION", disabled)
        y = session(model).run(None, feeds)[0]  # GroupedMatMul skips without opset 29.
        np.testing.assert_allclose(y, expected, rtol=2e-5, atol=1e-4)


PATTERN_CASES = {
    # id: (case, m, k, trans_a, n_pad, kernel)
    "gemv_k": ("Gemm", 1, 8, False, 0, "mlx_fp8_gemv_k"),
    "gemv_k_rows": ("Gemm", 3, 8, False, 0, "mlx_fp8_gemv_k"),
    "gemv_n": ("MatMul", 1, 8, False, 0, "mlx_fp8_gemv_n"),
    "dequant_matmul": ("MatMul", 9, 8, False, 0, "mlx_fp8_dequant_matmul"),
    "matvec_odd_k": ("Gemm", 1, 7, False, 0, "mlx_fp8_matvec"),
    "matvec_trans_a": ("Gemm", 3, 8, True, 0, "mlx_fp8_matvec"),
    "tiled_odd_n": ("MatMul", 3, 8, False, 1, "mlx_fp8_tiled_matmul"),
    "tiled_odd_k": ("Gemm", 6, 7, False, 0, "mlx_fp8_tiled_matmul"),
}


GROUPED_CASES = {
    # id: (trans_b, m, k, n, layout, kernel); two selections per row.
    "gemv_k": (True, 2, 64, 24, "blocked", "mlx_fp8_gemv_k"),
    "gemv_k_routes12": (True, 6, 64, 24, "axis", "mlx_fp8_gemv_k"),
    "matvec_odd_k": (True, 2, 67, 24, "blocked", "mlx_fp8_matvec"),
    "tiled_trans_b": (True, 20, 64, 24, "blocked", "mlx_fp8_tiled_matmul"),
    "tiled_odd_k": (True, 20, 67, 24, "tensor", "mlx_fp8_tiled_matmul"),
    "gemv_n": (False, 2, 64, 24, "blocked", "mlx_fp8_gemv_n"),
    "tiled_odd_n": (False, 3, 64, 7, "axis", "mlx_fp8_tiled_matmul"),
    "tiled": (False, 20, 64, 24, "blocked", "mlx_fp8_tiled_matmul"),
}


def _grouped_model(fmt, trans_b, m, k, n, layout, *, bits=None, x=None, ids=None, scale=None,
                   shared=False, bias=True):
    rng = np.random.default_rng(7)
    experts, selections = 4, 2
    shape = (experts, n, k) if trans_b else (experts, k, n)
    k_axis, n_axis = (2, 1) if trans_b else (1, 2)
    if bits is None:
        bits = _weights(rng, shape, fmt, special=True)
    experts = bits.shape[0]
    if scale is not None:
        axis, block = 0, 0
    elif layout == "tensor":
        axis, block, scale = 0, 0, np.array(0.0371, np.float32)
    elif layout == "axis":
        axis, block = n_axis, 0
        scale = rng.uniform(0.01, 0.2, n).astype(np.float32)
    else:
        axis, block = k_axis, 16
        scale_shape = list(shape)
        scale_shape[k_axis] = (k + block - 1) // block
        scale = rng.uniform(0.01, 0.2, scale_shape).astype(np.float32)
    zero_bits = np.zeros(scale.shape, np.uint8)
    weight = (_decode(bits, fmt) - _expand(_decode(zero_bits, fmt), shape, axis, block)
              ) * _expand(scale, shape, axis, block)
    feeds = {
        "x": rng.normal(0, 1, (m, k)).astype(np.float32) if x is None else x,
        "ids": (rng.integers(0, experts, (m, selections)) if ids is None else ids
                ).astype(np.int64),
    }
    b = rng.normal(0, 0.2, (experts, n)).astype(np.float32)
    dense = np.swapaxes(weight, 1, 2) if trans_b else weight
    with np.errstate(invalid="ignore"):
        expected = np.einsum("mk,mskn->msn", feeds["x"].astype(np.float64),
                             dense[feeds["ids"]].astype(np.float64)).astype(np.float32)
    if bias:
        expected = expected + b[feeds["ids"]]
    attrs = {"axis": axis, **({"block_size": block} if block else {})}
    nodes = [helper.make_node("DequantizeLinear", ["w", "ws", "wz"], ["wf"], **attrs),
             helper.make_node("GroupedMatMul", ["x", "wf", "ids", *(["b"] if bias else [])],
                              ["y"], transB=int(trans_b))]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)]
    if shared:
        outputs.append(helper.make_tensor_value_info("wf", TensorProto.FLOAT, None))
    initializers = [_initializer("w", bits, FORMATS[fmt][1]), _initializer("ws", scale),
                    _initializer("wz", zero_bits, FORMATS[fmt][1])]
    if bias:
        initializers.append(_initializer("b", b))
    graph = helper.make_graph(
        nodes, "fp8_grouped",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(feeds["x"].shape)),
         helper.make_tensor_value_info("ids", TensorProto.INT64, list(feeds["ids"].shape))],
        outputs, initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 29)], ir_version=11)
    return model.SerializeToString(), feeds, expected, weight


@pytest.mark.parametrize("shared", [False, True], ids=["fused", "shared"])
@pytest.mark.parametrize("case", list(GROUPED_CASES))
@pytest.mark.parametrize("fmt", list(FORMATS))
def test_fused_grouped(fmt, case, shared, monkeypatch):
    trans_b, m, k, n, layout, _ = GROUPED_CASES[case]
    model, feeds, expected, weight = _grouped_model(fmt, trans_b, m, k, n, layout, shared=shared)
    x_scale = float(np.abs(feeds["x"]).max() * np.abs(weight).sum(axis=2 if trans_b else 1).max())
    for disabled in ["0", "1"]:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION", disabled)
        # Skips where ORT cannot load opset 29.
        outputs = session(model).run(None, feeds)
        _check(outputs[0], expected, x_scale)
        if shared:
            np.testing.assert_array_equal(outputs[1], weight)


@pytest.mark.parametrize("case", list(GROUPED_CASES))
def test_fused_grouped_replay(case):
    trans_b, m, k, n, layout, _ = GROUPED_CASES[case]
    model, feeds, expected, weight = _grouped_model("e4m3fn", trans_b, m, k, n, layout)
    _check(session(model).run(None, feeds)[0], expected, float(np.abs(weight).sum()))


@pytest.mark.parametrize("trans_b,m", [(True, 1), (True, 9), (False, 1), (False, 9)],
                         ids=["gemv_k", "tiled_trans_b", "gemv_n", "tiled"])
@pytest.mark.parametrize("fmt", list(FORMATS))
def test_fused_grouped_decodes_all_bit_patterns(fmt, trans_b, m):
    """Each expert holds all 256 codes; one-hot rows select every decoded weight exactly."""
    k = 8
    patterns = np.arange(256, dtype=np.uint8)
    codes = np.stack([patterns, patterns[::-1]])
    bits = np.zeros((2, 256, k) if trans_b else (2, k, 256), np.uint8)
    if trans_b:
        bits[:, :, 0] = codes
    else:
        bits[:, 0, :] = codes
    x = np.zeros((m, k), np.float32)
    x[:, 0] = 1
    ids = np.tile([[0, 1]], (m, 1))
    model, feeds, expected, _ = _grouped_model(
        fmt, trans_b, m, k, 256, None, bits=bits, x=x, ids=ids,
        scale=np.array(0.125, np.float32), bias=False)
    y = session(model).run(None, feeds)[0]
    np.testing.assert_array_equal(expected, np.broadcast_to(
        _decode(codes, fmt) * np.float32(0.125), expected.shape))
    np.testing.assert_array_equal(np.isnan(y), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(y[finite], expected[finite])


DISPATCH_CASES = [
    *[(f"test_fused_replay[{_case_id(*c)}]", kernel) for *c, kernel in [
        ("Gemm", "axis", 1, 256, 96, None, "mlx_fp8_gemv_k"),
        ("MatMul", "blocked", 3, 64, 64, (2,), "mlx_fp8_gemv_n"),
        ("Gemm", "blocked", 20, 96, 48, None, "mlx_fp8_dequant_matmul"),
        ("GemmTransA", "blocked", 3, 64, 40, None, "mlx_fp8_matvec"),
        ("MatMul", "axis", 3, 33, 7, None, "mlx_fp8_tiled_matmul"),
    ]],
    *[(f"test_fused_decodes_all_bit_patterns[e4m3fn-{name}]", case[-1])
      for name, case in PATTERN_CASES.items()],
    *[(f"test_fused_grouped_replay[{name}]", case[-1])
      for name, case in GROUPED_CASES.items()],
    *[(f"test_fused_grouped_decodes_all_bit_patterns[e4m3fn-{name}]", kernel)
      for name, kernel in [("gemv_k", "mlx_fp8_gemv_k"), ("tiled_trans_b", "mlx_fp8_tiled_matmul"),
                           ("gemv_n", "mlx_fp8_gemv_n"), ("tiled", "mlx_fp8_tiled_matmul")]],
    # FP8 activations are never fused; None asserts that no QDQ/FP8 product kernel ran.
    *[(f"test_fp8_activation_not_fused[e4m3fn-{case}-{w}]", None)
      for case in ACTIVATION_CASES for w in ACTIVATION_WEIGHTS],
]


@pytest.mark.parametrize("test,kernel", DISPATCH_CASES, ids=[t for t, _ in DISPATCH_CASES])
def test_fused_dispatch(tmp_path, test, kernel):
    if "grouped" in test.lower():
        session(_grouped_model("e4m3fn", True, 1, 8, 4, "tensor")[0])  # Skips without opset 29.
    for disabled in [False, True]:
        trace = tmp_path / f"{disabled}.json"
        env = {**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)}
        env.pop("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION", None)
        if disabled:
            env["ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION"] = "1"
        subprocess.run(
            [sys.executable, "-m", "pytest", f"{Path(__file__)}::{test}",
             "-q", "-p", "no:cacheprovider"], env=env, check=True, timeout=180,
        )
        kernels = {e.get("args", {}).get("kernel") for e in json.loads(trace.read_text())
                   if e.get("cat") == "op.fast"}
        if kernel is None:
            # FP8 activations may use FP8 weight-only products, never integer QDQ readers.
            assert not {k for k in kernels if k and k.startswith("mlx_qdq")}, kernels
            break
        assert (kernel in kernels) is not disabled, kernels




@pytest.mark.parametrize("pattern", list(PATTERN_CASES))
@pytest.mark.parametrize("fmt", list(FORMATS))
def test_fused_decodes_all_bit_patterns(fmt, pattern):
    """A one-hot activation selects each decoded weight exactly, including NaN and infinities."""
    case, m, k, trans_a, pad, _ = PATTERN_CASES[pattern]
    n = 256 + pad
    patterns = np.arange(256, dtype=np.uint8)
    bits = np.zeros((n, k) if case == "Gemm" else (k, n), np.uint8)
    if case == "Gemm":
        bits[:256, 0] = patterns
    else:
        bits[0, :256] = patterns
    x = np.zeros((k, m) if trans_a else (m, k), np.float32)
    x[(0, slice(None)) if trans_a else (slice(None), 0)] = 1
    scale = np.array(0.125, np.float32)
    attrs = {"transA": int(trans_a), "transB": 1} if case == "Gemm" else {}
    graph = helper.make_graph(
        [helper.make_node("DequantizeLinear", ["w", "ws"], ["wf"]),
         helper.make_node(case, ["x", "wf"], ["y"], **attrs)],
        "fp8_patterns", [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [_initializer("w", bits, FORMATS[fmt][1]), _initializer("ws", scale)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    y = session(model.SerializeToString()).run(None, {"x": x})[0]
    assert y.shape == (m, n)
    np.testing.assert_array_equal(y[:, 256:], 0)
    y = y[:, :256]
    expected = np.broadcast_to(_decode(patterns, fmt) * scale, y.shape)
    np.testing.assert_array_equal(np.isnan(y), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(y[finite], expected[finite])
