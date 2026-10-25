"""Official ONNX packed tensors, not byte-valued MatMulNBits inputs.

Live inputs use ORT's typed allocation API and copy exactly ceil(N/2) bytes.
References use unpacked integers independently of the device's nibble loader.
GroupedMatMul-29 coverage is direct Rust handler coverage, not an ORT session.
"""

import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper


PACKED = (TensorProto.INT4, TensorProto.UINT4)


def _pack(values):
    flat = np.asarray(values).reshape(-1).astype(np.int32) & 15
    result = np.zeros((flat.size + 1) // 2, np.uint8)
    result[:] = flat[::2]
    result[:flat.size // 2] |= (flat[1::2] << 4).astype(np.uint8)
    if flat.size % 2:
        # Poison the unused nibble: it is not a logical zero or a row pad.
        result[-1] |= np.uint8(0xA0)
    return result


def _initializer(name, values, dtype):
    if dtype in PACKED:
        return TensorProto(name=name, data_type=dtype, dims=values.shape,
                           raw_data=_pack(values).tobytes())
    return numpy_helper.from_array(values, name)


def _ortvalue(values, dtype):
    values = np.asarray(values)
    if dtype not in PACKED:
        return ort.OrtValue.ortvalue_from_numpy(values)
    result = ort.OrtValue.ortvalue_from_shape_and_type(list(values.shape), dtype)
    assert result.element_type() == dtype
    assert result.shape() == list(values.shape)
    payload = _pack(values)
    if payload.size:
        ctypes.memmove(result.data_ptr(), payload.ctypes.data, payload.nbytes)
    return result


def _session(model, tmp_path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "packed-profile")
    session = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"])
    session.disable_fallback()
    return session


def _completed_on_mlx(session):
    events = json.loads(Path(session.end_profiling()).read_text())
    providers = {event.get("args", {}).get("provider")
                 for event in events if event.get("cat") == "Node"}
    assert "MLXExecutionProvider" in providers, providers
    assert "CPUExecutionProvider" not in providers, providers


def _values(shape, dtype, offset=0):
    count = int(np.prod(shape))
    if dtype == TensorProto.FLOAT:
        return ((np.arange(count) % 9 - 4) / 8).astype(np.float32).reshape(shape)
    low = -8 if dtype == TensorProto.INT4 else -128 if dtype == TensorProto.INT8 else 0
    width = 16 if dtype in PACKED else 256
    dt = np.int8 if dtype in (TensorProto.INT4, TensorProto.INT8) else np.uint8
    return ((np.arange(count) + offset) % width + low).astype(dt).reshape(shape)


def _binary32_coefficient(bits):
    bits = int(bits)
    exponent = (bits >> 23) & 255
    if exponent == 255:
        raise ValueError("exact finite-dot oracle requires finite binary32 operands")
    significand = bits & 0x7FFFFF
    if exponent:
        significand |= 1 << 23
    coefficient = significand << max(0, exponent - 1)
    return -coefficient if bits >> 31 else coefficient


def _round_product_sum(total):
    # Each binary32 coefficient is in units of 2^-149; products are in 2^-298.
    sign = int(total < 0) << 31
    magnitude = abs(total)
    shift = max(149, magnitude.bit_length() - 24)
    significand, remainder = divmod(magnitude, 1 << shift)
    halfway = 1 << (shift - 1)
    significand += remainder > halfway or (remainder == halfway and significand % 2 == 1)
    if significand == 1 << 24:
        significand >>= 1
        shift += 1
    if significand < 1 << 23:
        bits = significand
    else:
        exponent = shift - 148
        bits = 0x7F800000 if exponent >= 255 else (
            (exponent << 23) | (significand & 0x7FFFFF))
    return np.asarray(sign | bits, dtype=np.uint32).view(np.float32)[()]


def _exact_projection(a, w):
    assert a.dtype == w.dtype == np.float32
    rows = [[_binary32_coefficient(value) for value in row]
            for row in np.atleast_2d(a).view(np.uint32)]
    columns = [[_binary32_coefficient(value) for value in column]
               for column in w.T.copy().view(np.uint32)]
    result = np.empty((len(rows), len(columns)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, column in enumerate(columns):
            result[i, j] = _round_product_sum(sum(x * y for x, y in zip(row, column)))
    return result[0] if a.ndim == 1 else result


def _matrix_model(xdtype, wdtype, layout, initializers, missing_zero=False, k=None,
                  decoder=False, vector=False):
    if k is None:
        k = 256 if layout == "integer" else 65 if layout == "blocked" else 1025
    n = 8 if decoder else 5
    wshape = (k, n)
    scale_shape = () if layout == "integer" else ((k + 31) // 32, n) if layout == "blocked" else (n,)
    values = {"a": _values((k,) if vector else (3, k), xdtype), "w": _values(wshape, wdtype)}
    types = {"a": xdtype, "w": wdtype, "ws": TensorProto.FLOAT, "wz": wdtype}
    ws = np.full(scale_shape, 0.125, np.float32) if layout == "integer" else (
        np.linspace(0.015, 0.04, max(1, int(np.prod(scale_shape))), dtype=np.float32)
        .reshape(scale_shape))
    values["ws"] = ws
    if not missing_zero:
        values["wz"] = _values(scale_shape, wdtype, 3)
    attrs = {"axis": 0, "block_size": 32} if layout == "blocked" else {"axis": -1}
    nodes = [helper.make_node("DequantizeLinear",
                              ["w", "ws"] + ([] if missing_zero else ["wz"]), ["wf"], **attrs)]
    if xdtype != TensorProto.FLOAT:
        values["as"] = np.array(0.25, np.float32)
        types.update({"as": TensorProto.FLOAT, "az": xdtype})
        if not missing_zero:
            values["az"] = _values((), xdtype, 5)
        nodes.insert(0, helper.make_node("DequantizeLinear",
                                       ["a", "as"] + ([] if missing_zero else ["az"]), ["af"]))
    nodes.extend([
        helper.make_node("MatMul", ["a" if xdtype == TensorProto.FLOAT else "af", "wf"], ["y"]),
        helper.make_node("Add", ["wf", "wf"], ["shared"]),
    ])
    extra_initializers = []
    if decoder:
        for name in ("pk", "pv"):
            values[name] = _values((1, 1, 2, n), TensorProto.FLOAT)
            types[name] = TensorProto.FLOAT
        extra_initializers = [
            numpy_helper.from_array(np.array([0, 1] if vector else [0], np.int64), "query_axes"),
            numpy_helper.from_array(np.array(1 / 1024, np.float32), "query_scale"),
        ]
        nodes.extend([
            helper.make_node("Mul", ["y", "query_scale"], ["small_y"]),
            helper.make_node("Unsqueeze", ["small_y", "query_axes"], ["query"]),
            helper.make_node("Attention", ["query", "query", "query", "", "pk", "pv"],
                             ["attention", "present_k", "present_v"],
                             q_num_heads=1, kv_num_heads=1),
        ])
    # Binary initializer scales enable the existing exact integer subset, not a new policy.
    constants = {name for name in values
                 if (initializers and name != "a")
                 or (layout == "integer" and name in ("as", "ws"))}
    info = lambda name, dtype, shape: helper.make_tensor_value_info(name, dtype, shape)
    outputs = [info("y", TensorProto.FLOAT, [n] if vector else ["m", n]),
               info("wf", TensorProto.FLOAT, wshape),
               info("shared", TensorProto.FLOAT, wshape)]
    if xdtype != TensorProto.FLOAT:
        outputs.append(info("af", TensorProto.FLOAT, ["m", k]))
    if decoder:
        outputs.extend([
            info("attention", TensorProto.FLOAT, [1, 1 if vector else "m", n]),
            info("present_k", TensorProto.FLOAT, [1, 1, "total", n]),
            info("present_v", TensorProto.FLOAT, [1, 1, "total", n]),
        ])
    model = helper.make_model(
        helper.make_graph(
            nodes, "packed_qdq",
            [info(name, types[name], (list(value.shape) if vector else ["m", k])
                  if name == "a" else list(value.shape))
             for name, value in values.items() if name not in constants],
            outputs,
            [_initializer(name, values[name], types[name]) for name in sorted(constants)]
            + extra_initializers,
        ),
        opset_imports=[helper.make_opsetid("", 24)], ir_version=11,
    )
    return model.SerializeToString(), values, types, constants


def _matrix_replay(tmp_path, xdtype, wdtype, layout, initializers, missing_zero=False, k=None,
                   decoder=False, vector=False, record_property=None):
    model, values, types, constants = _matrix_model(
        xdtype, wdtype, layout, initializers, missing_zero, k, decoder, vector)
    session = _session(model, tmp_path)
    k = values["w"].shape[0]
    # Repeat M=3 before changing shape: a shape-keyed stale-data cache cannot pass.
    for iteration, m in enumerate([3, 3, 1, 2, 3] if decoder else [3, 3, 1, 0, 2, 3]):
        values["a"] = _values((k,) if vector else (m, k), xdtype, iteration)
        if "w" not in constants:
            values["w"] = _values(values["w"].shape, wdtype, iteration * 3)
        for name in ("az", "wz"):
            if name in values and name not in constants:
                values[name] = _values(values[name].shape, types[name], iteration + 2)
        for name in ("as", "ws"):
            if name in values and name not in constants:
                values[name] = (values[name] * np.float32(0.5)).copy()
        if decoder:
            values["pk"] = _values(values["pk"].shape, TensorProto.FLOAT) + np.float32(iteration / 8)
            values["pv"] = _values(values["pv"].shape, TensorProto.FLOAT) - np.float32(iteration / 8)
        feeds = {name: _ortvalue(value, types[name])
                 for name, value in values.items() if name not in constants}
        actual = [value.numpy() for value in session.run_with_ort_values(None, feeds)]
        sw = values["ws"]
        zw = values.get("wz", np.array(0, np.float32)).astype(np.float32)
        if layout == "blocked":
            sw = np.repeat(sw, 32, axis=0)[:k]
            if not missing_zero:
                zw = np.repeat(zw, 32, axis=0)[:k]
        w = (values["w"].astype(np.float32) - zw) * sw
        a = values["a"].astype(np.float32)
        if xdtype != TensorProto.FLOAT:
            a = (a - values.get("az", np.float32(0))) * values["as"]
        reference = _exact_projection(a, w)
        original = a @ w
        disagreements = np.argwhere(~np.isclose(original, reference, rtol=1e-5, atol=1e-6))
        if disagreements.size:
            details = [{
                "coordinate": coordinate.tolist(),
                "original_numpy_fp32": float(original[tuple(coordinate)]),
                "exact_sum_rounded_fp32": float(reference[tuple(coordinate)]),
                "actual": float(actual[0][tuple(coordinate)]),
            } for coordinate in disagreements]
            print("PACKED_QDQ_ORIGINAL_NUMPY_DISAGREEMENT " + json.dumps({
                "iteration": iteration, "disagreements": details,
            }))
            if record_property is not None:
                record_property(f"original_numpy_fp32_disagreement_iteration_{iteration}",
                                json.dumps(details))
        try:
            np.testing.assert_allclose(actual[0], reference, rtol=1e-5, atol=1e-6)
        except AssertionError:
            if initializers and layout == "axis" and xdtype in PACKED and wdtype in PACKED:
                _diagnose_projection(actual, a, w, values, xdtype, wdtype, iteration)
            raise
        np.testing.assert_array_equal(actual[1], w)
        np.testing.assert_array_equal(actual[2], w + w)
        if xdtype != TensorProto.FLOAT:
            np.testing.assert_array_equal(actual[3], a)
        if decoder:
            query = (reference * np.float32(1 / 1024)).reshape(-1, w.shape[1])
            keys = np.concatenate([values["pk"][0, 0], query], axis=0)
            cached_values = np.concatenate([values["pv"][0, 0], query], axis=0)
            scores = query.astype(np.float64) @ keys.astype(np.float64).T / np.sqrt(query.shape[1])
            probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            np.testing.assert_allclose(actual[-3], (probabilities @ cached_values)[None],
                                       rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(actual[-2], keys[None, None], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(actual[-1], cached_values[None, None], rtol=1e-5, atol=1e-6)
    _completed_on_mlx(session)


def _diagnose_projection(actual, a, w, values, xdtype, wdtype, iteration):
    reference = a @ w
    independent_reference = _exact_projection(a, w)
    coordinates = np.argwhere(
        ~np.isclose(actual[0], reference, rtol=1e-5, atol=1e-6)
        | ~np.isclose(actual[0], independent_reference, rtol=1e-5, atol=1e-6))

    def ordered_float32(value):
        bits = int(np.asarray(value, dtype=np.float32).view(np.uint32))
        return 0x80000000 - (bits & 0x7FFFFFFF) if bits & 0x80000000 else 0x80000000 + bits

    for row, column in coordinates:
        # Keep the original NumPy reduction and hosted diagnostic evidence identifiable.
        if np.isfinite(a[row]).all() and np.isfinite(w[:, column]).all():
            independent = math.fsum(float(x) * float(y) for x, y in zip(a[row], w[:, column]))
        else:
            independent = float(a[row].astype(np.float64) @ w[:, column].astype(np.float64))
        observed = float(actual[0][row, column])
        expected = float(reference[row, column])
        print("PACKED_QDQ_NUMERICAL_DIAGNOSTIC " + json.dumps({
            "iteration": iteration,
            "coordinate": [int(row), int(column)],
            "logical_types": {"a": xdtype, "w": wdtype},
            "shapes": {"a": list(a.shape), "w": list(w.shape)},
            "actual": observed,
            "original_fp32_matmul_reference": expected,
            "independent_exact_sum_rounded_fp32": float(independent_reference[row, column]),
            "independent_fp64_dot_of_fp32_dq_operands": independent,
            "finite_product_sum_method": "math.fsum of exact binary64 products of binary32 DQ operands",
            "absolute_difference": {
                "actual_vs_fp32": abs(observed - expected),
                "actual_vs_fp64": abs(observed - independent),
                "fp32_vs_fp64": abs(expected - independent),
            },
            "float32_ulp_distance": {
                "actual_vs_fp32": abs(ordered_float32(observed) - ordered_float32(expected)),
                "actual_vs_rounded_fp64": abs(ordered_float32(observed) - ordered_float32(independent)),
                "fp32_vs_rounded_fp64": abs(ordered_float32(expected) - ordered_float32(independent)),
            },
            "parameters": {
                name: np.asarray(values[name]).tolist() if name in values else None
                for name in ("as", "ws", "az", "wz")
            },
            "quantized_dot_operands": {
                "a_row": values["a"][row].tolist(),
                "w_column": values["w"][:, column].tolist(),
            },
            "fp32_dq_dot_operands": {
                "reference_a_row": a[row].tolist(),
                "reference_w_column": w[:, column].tolist(),
                "actual_a_row": actual[3][row].tolist(),
                "actual_w_column": actual[1][:, column].tolist(),
            },
            "dq_actual_equals_reference": {
                "a": bool(np.array_equal(actual[3], a)),
                "w": bool(np.array_equal(actual[1], w)),
            },
            "operation_order": "FP32 subtraction, FP32 DQ multiplication, then dot; no bias",
            "bias": None,
        }, separators=(",", ":")))


@pytest.mark.parametrize("xdtype", [
    TensorProto.FLOAT, TensorProto.INT4, TensorProto.UINT4, TensorProto.INT8, TensorProto.UINT8,
], ids=["float", "i4", "u4", "i8", "u8"])
@pytest.mark.parametrize("wdtype", PACKED, ids=["i4", "u4"])
@pytest.mark.parametrize("layout", ["integer", "axis", "blocked"])
@pytest.mark.parametrize("initializers", [False, True], ids=["live", "initializer"])
def test_packed_qdq_replay(tmp_path, record_property, xdtype, wdtype, layout, initializers):
    _matrix_replay(tmp_path, xdtype, wdtype, layout, initializers,
                   record_property=record_property)


@pytest.mark.parametrize("bits,coefficient", [
    (0, 0), (0x80000000, 0), (1, 1),
    (0x007FFFFF, (1 << 23) - 1), (0x00800000, 1 << 23),
    (0x3F800000, 1 << 149), (0xBF800000, -(1 << 149)),
    (0x7F7FFFFF, ((1 << 24) - 1) << 253),
])
def test_packed_exact_oracle_coefficients(bits, coefficient):
    assert _binary32_coefficient(bits) == coefficient


@pytest.mark.parametrize("total,bits", [
    (0, 0),
    (1 << 148, 0),  # Half of the smallest subnormal, ties to even.
    (-(1 << 148), 0x80000000),
    (1 << 149, 1),
    (3 << 148, 2),
    (1 << 172, 0x00800000),
    (((1 << 24) - 1) << 148, 0x00800000),
    ((1 << 298) + (1 << 274), 0x3F800000),
    ((1 << 298) + (1 << 274) + 1, 0x3F800001),
    (-((1 << 298) + (1 << 274) + 1), 0xBF800001),
    ((1 << 426) - (1 << 401) - 1, 0x7F7FFFFF),
    ((1 << 426) - (1 << 401), 0x7F800000),
])
def test_packed_exact_oracle_rounding(total, bits):
    assert int(_round_product_sum(total).view(np.uint32)) == bits


def test_packed_exact_oracle_products():
    e = np.float32(1 + 2**-23)
    a = np.array([[e, -1]], np.float32)
    w = np.array([[e], [e]], np.float32)
    assert int(_exact_projection(a, w)[0, 0].view(np.uint32)) == 0x34000001
    with pytest.raises(ValueError, match="finite"):
        _exact_projection(np.array([[np.inf]], np.float32), np.ones((1, 1), np.float32))


NUMERIC_CASES = (
    "lane-cancellation", "tree-cancellation", "product-residual", "tiny-normal",
    "partial-overflow", "product-overflow", "reduction-overflow", "overflow-cancellation",
    "finite-fma-product-overflow",
    "nan", "positive-infinity", "negative-infinity", "opposite-infinities",
    "zero-times-infinity", "dq-overflow",
)


@pytest.mark.parametrize("case", NUMERIC_CASES)
def test_packed_general_numerics(tmp_path, case):
    a = np.zeros((1, 65), np.float32)
    w = np.ones((65, 1), np.int8)
    scale = np.array(1, np.float32)
    expected = None
    if case in ("lane-cancellation", "tree-cancellation"):
        positions = [0, 32, 64] if case == "lane-cancellation" else [0, 1, 16]
        a[0, positions] = [2**24, 1, -(2**24)]
    elif case == "product-residual":
        a[0, [0, 32]] = [1 + 2**-23, -1]
        scale = np.array(1 + 2**-23, np.float32)
    elif case == "tiny-normal":
        a[0, [0, 32, 64]] = [2**-90, 2**-120, -(2**-90)]
    elif case in ("partial-overflow", "reduction-overflow", "overflow-cancellation"):
        positions = [0, 1] if case == "reduction-overflow" else [0, 32]
        a[0, positions] = 2**127
        if case == "overflow-cancellation":
            a[0, 64] = -(2**127)
        # Preserve the original F32 exceptional path, not exact-real overflow recovery.
        expected = np.inf
    elif case == "product-overflow":
        a[0, 0] = 2**127
        w.fill(2)
        expected = np.inf
    elif case == "finite-fma-product-overflow":
        # A nonzero lane must reject its overflowing TwoProd, although the FMA is finite.
        a[0, [17, 49]] = [-(2**126), 2**127]
        w.fill(2)
        expected = 2**127
    elif case == "nan":
        a[0, 0] = np.nan
        expected = np.nan
    elif case in ("positive-infinity", "negative-infinity", "opposite-infinities"):
        a[0, 0] = -np.inf if case == "negative-infinity" else np.inf
        expected = a[0, 0]
        if case == "opposite-infinities":
            a[0, 32] = -np.inf
            expected = np.nan
    elif case == "zero-times-infinity":
        scale = np.array(np.inf, np.float32)
        expected = np.nan
    elif case == "dq-overflow":
        a.fill(1)
        w.fill(7)
        scale = np.array(2**127, np.float32)
        expected = np.inf
    else:
        raise AssertionError(f"unhandled numeric case {case}")
    info = lambda name, dt, shape: helper.make_tensor_value_info(name, dt, shape)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("DequantizeLinear", ["w", "s"], ["wf"]),
             helper.make_node("MatMul", ["a", "wf"], ["y"])],
            "packed_general_numerics",
            [info("a", TensorProto.FLOAT, [1, 65]), info("w", TensorProto.INT4, [65, 1])],
            [info("y", TensorProto.FLOAT, [1, 1]), info("wf", TensorProto.FLOAT, [65, 1])],
            [numpy_helper.from_array(scale, "s")],
        ), opset_imports=[helper.make_opsetid("", 24)], ir_version=11,
    ).SerializeToString()
    session = _session(model, tmp_path)
    actual, actual_w = [value.numpy() for value in session.run_with_ort_values(
        None, {"a": _ortvalue(a, TensorProto.FLOAT), "w": _ortvalue(w, TensorProto.INT4)})]
    with np.errstate(over="ignore", invalid="ignore"):
        wf = w.astype(np.float32) * scale
    np.testing.assert_array_equal(actual_w, wf)
    reference = _exact_projection(a, wf) if expected is None else np.full((1, 1), expected, np.float32)
    np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
    # The finite constructions are exactly representable; exceptional classes also match exactly.
    np.testing.assert_array_equal(actual, reference)
    _completed_on_mlx(session)


def test_packed_general_numerics_actual_dispatch(tmp_path):
    trace = tmp_path / "packed-general-numerics.json"
    env = {**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)}
    env.pop("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION", None)
    subprocess.run(
        [sys.executable, "-m", "pytest",
         f"{Path(__file__)}::test_packed_general_numerics",
         "-q", "-p", "no:cacheprovider"],
        env=env, check=True, timeout=180,
    )
    events = json.loads(trace.read_text())
    kernels = [event.get("args", {}).get("kernel")
               for event in events if event.get("cat") == "op.fast"]
    assert kernels.count("mlx_qdq_packed4_matmul") == len(NUMERIC_CASES), kernels


@pytest.mark.parametrize("wdtype", [TensorProto.INT8, TensorProto.UINT8], ids=["i8", "u8"])
def test_packed_activation_byte_weight(tmp_path, wdtype):
    _matrix_replay(tmp_path, TensorProto.INT4, wdtype, "axis", False)


@pytest.mark.parametrize("layout", ["integer", "axis", "blocked"])
def test_packed_missing_zero(tmp_path, layout):
    _matrix_replay(tmp_path, TensorProto.INT4, TensorProto.UINT4, layout, False, True)


def test_packed_empty_reduction(tmp_path):
    _matrix_replay(tmp_path, TensorProto.INT4, TensorProto.UINT4, "integer", False, k=0)


def test_packed_vector_uses_ordinary_dq(tmp_path):
    x = np.arange(5, dtype=np.float32)
    w = _values((5, 3), TensorProto.INT4)
    scale = np.array(0.125, np.float32)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("DequantizeLinear", ["w", "s"], ["wf"]),
             helper.make_node("MatMul", ["x", "wf"], ["y"])],
            "packed_vector",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [5]),
             helper.make_tensor_value_info("w", TensorProto.INT4, [5, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])],
            [numpy_helper.from_array(scale, "s")],
        ), opset_imports=[helper.make_opsetid("", 21)], ir_version=10,
    ).SerializeToString()
    session = _session(model, tmp_path)
    actual = session.run_with_ort_values(
        None, {"x": _ortvalue(x, TensorProto.FLOAT),
               "w": _ortvalue(w, TensorProto.INT4)})[0].numpy()
    np.testing.assert_array_equal(actual, x @ (w.astype(np.float32) * scale))
    _completed_on_mlx(session)


@pytest.mark.parametrize("case", ["integer", "general"])
def test_packed_qdq_dispatch_case(tmp_path, case):
    _matrix_replay(tmp_path, TensorProto.INT4, TensorProto.UINT4,
                   "integer" if case == "integer" else "blocked", False)


@pytest.mark.parametrize("case,kernel", [
    ("integer", "mlx_qdq_packed4_integer_matmul"),
    ("general", "mlx_qdq_packed4_matmul"),
])
def test_packed_qdq_actual_dispatch(tmp_path, case, kernel):
    for disabled in (False, True):
        trace = tmp_path / f"packed-{case}-{disabled}.json"
        env = {**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)}
        if disabled:
            env["ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION"] = "1"
        else:
            env.pop("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION", None)
        subprocess.run(
            [sys.executable, "-m", "pytest",
             f"{Path(__file__)}::test_packed_qdq_dispatch_case[{case}]",
             "-q", "-p", "no:cacheprovider"],
            env=env, check=True, timeout=180,
        )
        events = json.loads(trace.read_text())
        kernels = {event.get("args", {}).get("kernel")
                   for event in events if event.get("cat") == "op.fast"}
        assert (kernel in kernels) is not disabled, kernels
        if disabled:
            assert not any(name and name.startswith("mlx_qdq_") for name in kernels), kernels


@pytest.mark.parametrize("vector", [False, True], ids=["matrix", "vector"])
def test_packed_decoder_dispatch_case(tmp_path, vector):
    _matrix_replay(tmp_path, TensorProto.FLOAT if vector else TensorProto.INT4,
                   TensorProto.UINT4, "blocked", False, decoder=True, vector=vector)


@pytest.mark.parametrize("vector", [False, True], ids=["matrix", "vector"])
def test_packed_decoder_partition_and_dispatch(tmp_path, vector):
    # Attention with both present-cache slots triggers capability's decoder classification.
    for disabled in (False, True):
        trace = tmp_path / f"packed-decoder-{disabled}.json"
        env = {**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)}
        if disabled:
            env["ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION"] = "1"
        else:
            env.pop("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION", None)
        subprocess.run(
            [sys.executable, "-m", "pytest",
             f"{Path(__file__)}::test_packed_decoder_dispatch_case[{'vector' if vector else 'matrix'}]",
             "-q", "-p", "no:cacheprovider"],
            env=env, check=True, timeout=180,
        )
        events = json.loads(trace.read_text())
        paths = [event for event in events if event.get("cat") == "ep.path"]
        products = [event for event in paths
                    if "MatMul" in event["args"].get("ops", "").split(",")]
        assert products, paths
        if vector:
            assert all("DequantizeLinear" not in event["args"]["ops"].split(",")
                       for event in products), products
        else:
            assert all(event["name"] == "mlx.compute[eager]" for event in products), products
            assert all("DequantizeLinear" in event["args"]["ops"].split(",")
                       for event in products), products
        if not vector:
            assert all("Attention" not in event["args"]["ops"].split(",")
                       for event in products), products
        assert len({event["args"]["partition_id"] for event in products}) == 1, products
        assert any("Attention" in event["args"].get("ops", "").split(",") for event in paths), paths
        kernels = {event.get("args", {}).get("kernel")
                   for event in events if event.get("cat") == "op.fast"}
        assert ("mlx_qdq_packed4_matmul" in kernels) is (not disabled and not vector), kernels
        if disabled or vector:
            assert not any(name and name.startswith("mlx_qdq_") for name in kernels), kernels


@pytest.mark.parametrize("dtype", PACKED, ids=["i4", "u4"])
@pytest.mark.parametrize("shape", [(), (1,), (5,), (3, 5), (3, 3, 5), (0,), (3, 0)])
@pytest.mark.parametrize("half", [False, True], ids=["float", "half"])
def test_packed_ordinary_dq(tmp_path, dtype, shape, half):
    x = _values(shape, dtype)
    dt = TensorProto.FLOAT16 if half else TensorProto.FLOAT
    scale = np.array(0.125, np.float16 if half else np.float32)
    zero = _values((), dtype, 3)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("DequantizeLinear", ["x", "s", "z"], ["y"])],
            "ordinary_packed_dq",
            [helper.make_tensor_value_info("x", dtype, shape),
             helper.make_tensor_value_info("z", dtype, [])],
            [helper.make_tensor_value_info("y", dt, shape)],
            [numpy_helper.from_array(scale, "s")],
        ), opset_imports=[helper.make_opsetid("", 21)], ir_version=10,
    ).SerializeToString()
    session = _session(model, tmp_path)
    actual = session.run_with_ort_values(
        None, {"x": _ortvalue(x, dtype), "z": _ortvalue(zero, dtype)})[0].numpy()
    expected = ((x.astype(np.float32) - zero.astype(np.float32)) * scale).astype(scale.dtype)
    np.testing.assert_array_equal(actual, expected)
    _completed_on_mlx(session)


def test_packed_bad_live_axis_geometry_recovers(tmp_path):
    dtype = TensorProto.INT4
    info = lambda name, dt, shape: helper.make_tensor_value_info(name, dt, shape)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("DequantizeLinear", ["x", "s", "z"], ["y"], axis=-1)],
            "packed_live_geometry",
            [info("x", dtype, ["m", "k"]), info("s", TensorProto.FLOAT, ["p"]),
             info("z", dtype, ["p"])],
            [info("y", TensorProto.FLOAT, ["m", "k"])],
        ), opset_imports=[helper.make_opsetid("", 21)], ir_version=10,
    ).SerializeToString()
    session = _session(model, tmp_path)
    x = _values((3, 5), dtype)
    for size in (5, 4, 5):
        s = np.full((size,), 0.125, np.float32)
        z = _values((size,), dtype)
        feeds = {"x": _ortvalue(x, dtype), "s": _ortvalue(s, TensorProto.FLOAT),
                 "z": _ortvalue(z, dtype)}
        if size == 4:
            with pytest.raises(ort.capi.onnxruntime_pybind11_state.EPFail, match="geometry"):
                session.run_with_ort_values(None, feeds)
        else:
            actual = session.run_with_ort_values(None, feeds)[0].numpy()
            np.testing.assert_array_equal(actual, (x.astype(np.float32) - z) * s)
    _completed_on_mlx(session)


@pytest.mark.parametrize("blocked", [False, True], ids=["axis", "blocked"])
def test_packed_invalid_static_geometry_is_unclaimed(tmp_path, blocked):
    shape = [5, 3] if blocked else [3, 5]
    parameter = [3, 3] if blocked else [4]
    attrs = {"axis": 0, "block_size": 4} if blocked else {"axis": -1}
    info = lambda name, dt, dims: helper.make_tensor_value_info(name, dt, dims)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("DequantizeLinear", ["x", "s"], ["wf"], **attrs),
             helper.make_node("MatMul", ["a", "wf"], ["y"])],
            "invalid_packed_qdq",
            [info("x", TensorProto.INT4, shape), info("s", TensorProto.FLOAT, parameter),
             info("a", TensorProto.FLOAT, [2, shape[0]])],
            [info("y", TensorProto.FLOAT, [2, shape[1]])],
        ), opset_imports=[helper.make_opsetid("", 24)], ir_version=11,
    ).SerializeToString()
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail, match="fallback|assigned"):
        _session(model, tmp_path)
