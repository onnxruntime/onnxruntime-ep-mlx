"""Schema-1 FusedMatMul contracts; runtime evidence requires an admitted run."""

import itertools
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import TensorProto as T, helper, numpy_helper
import pytest


DTYPES = [np.float16, np.float32, np.float64, "bf16"]
FLAGS = list(itertools.product([0, 1], repeat=4))
FLAG_NAMES = ["transA", "transB", "transBatchA", "transBatchB"]


def _round_bf16(values):
    values = np.asarray(values, np.float32)
    bits = values.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    # Preserve infinities and canonicalize NaNs rather than rounding their payload into infinity.
    rounded = np.where(np.isnan(values), np.uint32(0x7FC00000), rounded).astype(np.uint32)
    return rounded.view(np.float32)


def _typed(values, dtype):
    return _round_bf16(values) if dtype == "bf16" else np.asarray(values, dtype)


def _storage(logical, trans, trans_batch):
    """Encode logical (..., M, K) into upstream's physical input layout."""
    if logical.ndim < 2:
        return logical.copy()
    physical = logical.swapaxes(-1, -2) if trans else logical
    if trans_batch:
        physical = np.moveaxis(physical, -2, 0)
    return np.ascontiguousarray(physical)


def _oracle(a, b, dtype, alpha):
    # Numerical oracle takes logical operands, never the lowering's physical permutation.
    work = np.float64 if dtype == np.float64 else np.float32
    scale = _typed(np.float32(alpha), dtype).astype(work)
    with np.errstate(all="ignore"):
        result = np.matmul(a.astype(work), b.astype(work)) * scale
        return _typed(result, dtype)


def _model(dtype, shapes, attrs, *, constant_b=None):
    tensor_type = T.BFLOAT16 if dtype == "bf16" else helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    boundary_type = T.FLOAT if dtype == "bf16" else tensor_type
    inputs, nodes, names, initializers = [], [], [], []
    for name, shape in zip(["a", "b"], shapes, strict=True):
        if name == "b" and constant_b is not None:
            initializers.append(numpy_helper.from_array(constant_b, name))
        else:
            inputs.append(helper.make_tensor_value_info(name, boundary_type, shape))
        if dtype == "bf16":
            nodes.append(helper.make_node("Cast", [name], [name + "_bf"], to=T.BFLOAT16))
            names.append(name + "_bf")
        else:
            names.append(name)
    nodes.append(helper.make_node("FusedMatMul", names, ["product"], domain="com.microsoft", **attrs))
    output = "product"
    if dtype == "bf16":
        nodes.append(helper.make_node("Cast", ["product"], ["y"], to=T.FLOAT))
        output = "y"
    return helper.make_model(
        helper.make_graph(
            nodes, "fused_matmul", inputs,
            [helper.make_tensor_value_info(output, boundary_type, None)],
            initializer=initializers),
        opset_imports=[helper.make_opsetid("", 23), helper.make_opsetid("com.microsoft", 1)],
        ir_version=11)


def _session(model, tmp_path, *, cpu=False):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    options.enable_profiling = not cpu
    options.profile_file_prefix = str(tmp_path / "fused_matmul")
    if not cpu:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model.SerializeToString(), options,
        providers=["CPUExecutionProvider"] if cpu else ["MLXExecutionProvider"])
    session.disable_fallback()
    return session


def _placement(session):
    path = Path(session.end_profiling())
    try:
        events = json.loads(path.read_text())
    finally:
        path.unlink()
    compute = [event for event in events
               if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time")]
    assert compute, "profiling must contain actual compute events"
    for event in compute:
        assert "provider" in event.get("args", {}), event
        assert event["args"]["provider"] == "MLXExecutionProvider", event
    return compute


def _check(actual, expected, dtype, *, exact=False):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if exact:
        np.testing.assert_array_equal(actual, expected)
    else:
        tolerance = {np.float16: 2e-3, np.float32: 2e-6, np.float64: 2e-14, "bf16": 1e-2}[dtype]
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("shapes", [
    ((2, 3, 4), (1, 4, 5)),
    ((2, 1, 3, 4), (1, 5, 4, 6)),
    ((2, 1, 3, 4, 5), (1, 3, 1, 5, 2)),
])
@pytest.mark.parametrize("eager", [False, True])
def test_all_transpose_combinations(dtype, flags, shapes, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    # Distinct extents and non-symmetric values discriminate every axis ordering.
    a = _typed((np.arange(np.prod(shapes[0])).reshape(shapes[0]) % 7 - 3) / 4, dtype)
    b = _typed((np.arange(np.prod(shapes[1])).reshape(shapes[1]) % 9 - 4) / 4, dtype)
    ta, tb, tba, tbb = flags
    physical_a, physical_b = _storage(a, ta, tba), _storage(b, tb, tbb)
    attrs = dict(zip(FLAG_NAMES, flags, strict=True), alpha=-0.5)
    model = _model(dtype, [physical_a.shape, physical_b.shape], attrs)
    feeds = {"a": physical_a, "b": physical_b}
    expected = _oracle(a, b, dtype, -0.5)
    session = _session(model, tmp_path)
    _check(session.run(None, feeds)[0], expected, dtype, exact=True)
    _placement(session)
    # ORT CPU registers this contrib op only for float and double.
    if dtype in [np.float32, np.float64]:
        _check(_session(model, tmp_path, cpu=True).run(None, feeds)[0], expected, dtype, exact=True)


GEOMETRIES = [
    ((4,), (4,)), ((4,), (4, 3)), ((3, 4), (4,)),
    ((4,), (2, 5, 4, 3)), ((2, 5, 3, 4), (4,)),
    ((3, 4), (4, 2)), ((2, 1, 3, 4), (5, 4, 2)),
    ((0,), (0,)), ((2, 0), (0, 3)), ((0, 4), (4, 3)),
    ((2, 4), (4, 0)), ((0, 2, 4), (0, 4, 3)),
    ((0, 1, 2, 4), (1, 3, 4, 5)),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shapes", GEOMETRIES)
@pytest.mark.parametrize("trans", list(itertools.product([0, 1], repeat=2)))
@pytest.mark.parametrize("eager", [False, True])
def test_vectors_broadcast_and_empty(dtype, shapes, trans, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    a, b = [_typed((np.arange(np.prod(shape)).reshape(shape) % 5 - 2) / 2, dtype)
            for shape in shapes]
    ta, tb = trans
    a_physical, b_physical = _storage(a, ta, False), _storage(b, tb, False)
    model = _model(dtype, [a_physical.shape, b_physical.shape], {"transA": ta, "transB": tb})
    session = _session(model, tmp_path)
    _check(session.run(None, {"a": a_physical, "b": b_physical})[0],
           _oracle(a, b, dtype, 1), dtype, exact=True)
    _placement(session)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("eager", [False, True])
def test_dynamic_live_replays(dtype, flags, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    attrs = dict(zip(FLAG_NAMES, flags, strict=True), alpha=0.25)
    model = _model(dtype, [[None] * 4, [None] * 4], attrs)
    session = _session(model, tmp_path)
    for batch, rows, k, cols, offset in [(2, 3, 4, 5, 0), (2, 3, 4, 5, 3),
                                       (1, 2, 3, 4, 1), (0, 2, 3, 4, 2),
                                       (2, 3, 0, 5, 1), (2, 3, 4, 5, 2)]:
        a = _typed((np.arange(batch * rows * k).reshape(batch, 1, rows, k) % 7 - offset) / 2, dtype)
        b = _typed((np.arange(k * cols).reshape(1, 1, k, cols) % 5 - offset) / 2, dtype)
        ta, tb, tba, tbb = flags
        feeds = {"a": _storage(a, ta, tba), "b": _storage(b, tb, tbb)}
        _check(session.run(None, feeds)[0], _oracle(a, b, dtype, 0.25), dtype, exact=True)
    _placement(session)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("alpha", [1.0, 0.0, -0.0, -2.0, 0.1, 1.00390625,
                                 2**-20, 2**20, np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("eager", [False, True])
def test_alpha_precision_and_specials(dtype, alpha, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    a = _typed([[1.5, -0.25], [0, 0], [1, 2]], dtype)
    b = _typed([[0.5, -2], [1.5, 1]], dtype)
    model = _model(dtype, [a.shape, b.shape], {"alpha": float(alpha)})
    session = _session(model, tmp_path)
    expected = _oracle(a, b, dtype, alpha)
    actual = session.run(None, {"a": a, "b": b})[0]
    _check(actual, expected, dtype, exact=True)
    finite_nonzero = np.isfinite(expected) & (expected != 0)
    np.testing.assert_array_equal(np.signbit(actual[finite_nonzero]), np.signbit(expected[finite_nonzero]))
    _placement(session)
    # The double CPU constructor rejects nonfinite alpha despite no such schema restriction.
    if dtype in [np.float32, np.float64] and np.isfinite(alpha) and alpha != 0:
        _check(_session(model, tmp_path, cpu=True).run(None, {"a": a, "b": b})[0],
               expected, dtype)


@pytest.mark.parametrize("dtype", [np.float16, "bf16"])
@pytest.mark.parametrize("eager", [False, True])
def test_scaling_precedes_output_rounding(dtype, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    # An unscaled F16 result overflows; a separate narrow matmul cannot recover it.
    a, b = _typed([[256]], dtype), _typed([[256]], dtype)
    alpha = 0.25
    expected = _oracle(a, b, dtype, alpha)
    if dtype == "bf16":
        # The unscaled product is a BF16 tie; prematurely rounding it loses the final upward round.
        a = _typed([[1, 1]], dtype)
        b = _typed([[1], [2**-8]], dtype)
        alpha = 1.0078125
        expected = _typed([[1.01171875 + 2**-15]], dtype)
    model = _model(dtype, [a.shape, b.shape], {"alpha": alpha})
    session = _session(model, tmp_path)
    _check(session.run(None, {"a": a, "b": b})[0], expected, dtype, exact=True)
    _placement(session)


@pytest.mark.parametrize("eager", [False, True])
def test_half_scaling_precedes_underflow(eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    a = np.array([[2**-13]], np.float16)
    model = _model(np.float16, [a.shape, a.shape], {"alpha": 4.0})
    session = _session(model, tmp_path)
    _check(session.run(None, {"a": a, "b": a})[0], np.array([[2**-24]], np.float16),
           np.float16, exact=True)
    _placement(session)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("value", [-1, 2, 2**32, 2**32 + 1, -(2**63), -(2**63) + 1, 2**63 - 1])
def test_nonboolean_full_width_flags(dtype, value, tmp_path):
    a = _typed(np.arange(24).reshape(2, 3, 4) / 4, dtype)
    b = _typed(np.arange(40).reshape(2, 4, 5) / 4, dtype)
    physical_a, physical_b = _storage(a, True, True), _storage(b, True, True)
    attrs = dict.fromkeys(FLAG_NAMES, value)
    model = _model(dtype, [[None] * 3, [None] * 3], attrs)
    session = _session(model, tmp_path)
    _check(session.run(None, {"a": physical_a, "b": physical_b})[0],
           _oracle(a, b, dtype, 1), dtype, exact=True)
    _placement(session)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eager", [False, True])
def test_defaults_and_constant_b(dtype, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    a = _typed([[1.25, -2], [0.5, 3]], dtype)
    b = _typed([[0.5, 2, 1], [-1, 0, 3]], dtype)
    model = _model(dtype, [a.shape, b.shape], {}, constant_b=b)
    session = _session(model, tmp_path)
    for value in [a, a + 1, a]:
        _check(session.run(None, {"a": value})[0], _oracle(value, b, dtype, 1),
               dtype, exact=True)
    _placement(session)


@pytest.mark.parametrize("eager", [False, True])
def test_float64_full_width_values(eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    a = np.array([[1 + 2**-40, 2**-900], [-(1 + 2**-45), 2**800]], np.float64)
    b = np.array([[1, 0], [0, 1]], np.float64)
    model = _model(np.float64, [a.shape, b.shape], {"alpha": 0.5})
    session = _session(model, tmp_path)
    _check(session.run(None, {"a": a, "b": b})[0], a * 0.5, np.float64, exact=True)
    _placement(session)


@pytest.mark.parametrize("eager", [False, True])
def test_float64_claimed_edge_preserves_partition_boundary(eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = _model(np.float64, [[2, 2], [2, 2]], {})
    graph = model.graph
    graph.ClearField("output")
    graph.input.extend([
        helper.make_tensor_value_info("expected", T.DOUBLE, [2, 2]),
        helper.make_tensor_value_info("x", T.FLOAT, [2, 2]),
        helper.make_tensor_value_info("z", T.FLOAT, [2, 2]),
    ])
    graph.node.extend([
        helper.make_node("Equal", ["product", "expected"], ["equal"]),
        helper.make_node("Where", ["equal", "x", "z"], ["y"]),
    ])
    graph.output.append(helper.make_tensor_value_info("y", T.FLOAT, [2, 2]))
    a = np.array([[1 + 2**-40, 0], [0, 2**800]], np.float64)
    x = np.array([[1, 2], [3, 4]], np.float32)
    feeds = {"a": a, "b": np.eye(2, dtype=np.float64), "expected": a, "x": x, "z": -x}
    session = _session(model, tmp_path)
    _check(session.run(None, feeds)[0], x, np.float32, exact=True)
    compute = _placement(session)
    assert len({event["name"] for event in compute}) >= 2, (
        "the float64 FusedMatMul/Equal region must not fuse across its bool edge into float32 Where")


@pytest.mark.parametrize("shapes", [([2, 3], [3, 4]), ([3], [3]),
                                    ([2, 3, 4], [4, 5]),
                                    ([2, 3, 4], [1, 2, 4, 5])])
@pytest.mark.parametrize("batch_flag", ["transBatchA", "transBatchB"])
def test_invalid_batch_transpose_rank_is_not_claimed(shapes, batch_flag, tmp_path):
    # Loader/claim rejection, not a numerical gap: upstream MatMulComputeHelper forbids these.
    model = _model(np.float32, [[None] * len(shape) for shape in shapes], {batch_flag: 1})
    with pytest.raises((ort.capi.onnxruntime_pybind11_state.Fail,
                        ort.capi.onnxruntime_pybind11_state.InvalidArgument,
                        ort.capi.onnxruntime_pybind11_state.RuntimeException),
                       match="FusedMatMul|fallback|rank|assigned"):
        _session(model, tmp_path)
