"""FP8 DequantizeLinear transport, GPU decode, and fused weight products.

References decode with ml_dtypes, apply the ONNX per-element ``(x - zero) * scale`` rounding in
float32, and accumulate in float64.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
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


@pytest.mark.parametrize("case,layout,m,k,n,batch,kernel", [
    ("Gemm", "axis", 1, 256, 96, None, "mlx_fp8_gemv_k"),
    ("MatMul", "blocked", 3, 64, 64, (2,), "mlx_fp8_gemv_n"),
    ("Gemm", "blocked", 20, 96, 48, None, "mlx_fp8_dequant_matmul"),
    ("GemmTransA", "blocked", 3, 64, 40, None, "mlx_fp8_matvec"),
    ("MatMul", "axis", 3, 33, 7, None, "mlx_fp8_tiled_matmul"),
])
def test_fused_dispatch(tmp_path, case, layout, m, k, n, batch, kernel):
    for disabled in [False, True]:
        trace = tmp_path / f"{disabled}.json"
        env = {**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)}
        env.pop("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION", None)
        if disabled:
            env["ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION"] = "1"
        name = _case_id(case, layout, m, k, n, batch)
        subprocess.run(
            [sys.executable, "-m", "pytest",
             f"{Path(__file__)}::test_fused_replay[{name}]",
             "-q", "-p", "no:cacheprovider"], env=env, check=True, timeout=180,
        )
        kernels = {e.get("args", {}).get("kernel") for e in json.loads(trace.read_text())
                   if e.get("cat") == "op.fast"}
        assert (kernel in kernels) is not disabled, kernels


@pytest.mark.parametrize("m,k", [(2, 64), (20, 67)])
@pytest.mark.parametrize("fmt", ["e4m3fn", "e5m2"])
def test_fused_grouped(fmt, m, k):
    rng = np.random.default_rng(7)
    experts, n, selections = 4, 24, 2
    bits = _weights(rng, (experts, n, k), fmt, special=True)
    scale = rng.uniform(0.01, 0.2, (experts, n, (k + 15) // 16)).astype(np.float32)
    weight = _decode(bits, fmt) * _expand(scale, bits.shape, 2, 16)
    feeds = {
        "x": rng.normal(0, 1, (m, k)).astype(np.float32),
        "ids": rng.integers(0, experts, (m, selections)).astype(np.int64),
    }
    graph = helper.make_graph(
        [helper.make_node("DequantizeLinear", ["w", "ws"], ["wf"], axis=2, block_size=16),
         helper.make_node("GroupedMatMul", ["x", "wf", "ids"], ["y"], transB=1)],
        "fp8_grouped",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [m, k]),
         helper.make_tensor_value_info("ids", TensorProto.INT64, [m, selections])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [_initializer("w", bits, FORMATS[fmt][1]), _initializer("ws", scale)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 29)], ir_version=11)
    result = session(model.SerializeToString())  # Skips where ORT cannot load opset 29.
    expected = np.stack([
        feeds["x"][row].astype(np.float64) @ weight[e].astype(np.float64).T
        for row in range(m) for e in feeds["ids"][row]
    ]).reshape(m, selections, n).astype(np.float32)
    _check(result.run(None, feeds)[0], expected, float(np.abs(weight).sum(axis=-1).max() * 5))


@pytest.mark.parametrize("case,m", [("Gemm", 1), ("MatMul", 1), ("MatMul", 9), ("Gemm", 3)],
                         ids=["gemv_k", "gemv_n", "dequant_matmul", "gemv_k_rows"])
@pytest.mark.parametrize("fmt", list(FORMATS))
def test_fused_decodes_all_bit_patterns(fmt, case, m):
    """A one-hot activation selects each decoded weight exactly, including NaN and infinities."""
    k = 8
    patterns = np.arange(256, dtype=np.uint8)
    bits = np.zeros((256, k) if case == "Gemm" else (k, 256), np.uint8)
    if case == "Gemm":
        bits[:, 0] = patterns
    else:
        bits[0] = patterns
    x = np.zeros((m, k), np.float32)
    x[:, 0] = 1
    scale = np.array(0.125, np.float32)
    nodes = [helper.make_node("DequantizeLinear", ["w", "ws"], ["wf"]),
             helper.make_node(case, ["x", "wf"], ["y"], **({"transB": 1} if case == "Gemm"
                                                         else {}))]
    graph = helper.make_graph(
        nodes, "fp8_patterns", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [m, k])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [_initializer("w", bits, FORMATS[fmt][1]), _initializer("ws", scale)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    y = session(model.SerializeToString()).run(None, {"x": x})[0]
    expected = np.broadcast_to(_decode(patterns, fmt) * scale, y.shape)
    np.testing.assert_array_equal(np.isnan(y), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(y[finite], expected[finite])
