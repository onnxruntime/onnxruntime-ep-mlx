"""Gemm geometry, integer arithmetic and schema-version contracts."""

import _schema_runtime as schema_runtime
import numpy as np
import onnx
import pytest
from onnx.reference import ReferenceEvaluator


def _gemm_rank_cases():
    yield "Gemm", {
        "x": np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3),
        "w": np.array([[1, 2], [3, 4]], np.float32),
        "bias": np.array([0.5, -0.5], np.float32),
    }, {"transA": 1}, (
        np.array([13.5, 17.5, 17.5, 23.5, 21.5, 29.5,
                  37.5, 53.5, 41.5, 59.5, 45.5, 65.5], np.float32).reshape(2, 3, 2),
    )


GEMM_RANK_CASES = list(_gemm_rank_cases())


@pytest.mark.parametrize("op,feeds,attrs,expected", GEMM_RANK_CASES, ids=[case[0] for case in GEMM_RANK_CASES])
def test_gemm_rank_goldens_match_onnx_reference(op, feeds, attrs, expected):
    schema_runtime.check_reference(op, feeds, attrs, expected, opset=29)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("op,feeds,attrs,expected", GEMM_RANK_CASES, ids=[case[0] for case in GEMM_RANK_CASES])
def test_gemm_rank_mlx_against_reference(op, feeds, attrs, expected, eager, monkeypatch, tmp_path):
    schema_runtime.check_mlx(op, feeds, attrs, expected, eager, monkeypatch, tmp_path, opset=29)


def _integer_gemm_case(dtype, case):
    if case == "batched-transposes":
        feeds = {"a": np.arange(1, 13, dtype=dtype).reshape(2, 2, 3),
                 "b": np.array([[1, 3], [2, 4]], dtype),
                 "c": np.array([1, 2], dtype)}
        attrs = {"transA": 1, "transB": 1, "alpha": 2.0, "beta": -1.0}
    elif case in {"vector", "full-width"}:
        large = ((2**62 + 1) if dtype == np.int64 else (2**63 + 1)
                 ) if case == "full-width" else 2**25 + 1
        feeds = {"a": np.array([large, 2], dtype),
                 "b": np.ones((2, 1), dtype),
                 "c": np.array([np.iinfo(dtype).max], dtype)}
        attrs = {"transA": 1, "beta": 0.0}
    elif case == "zero-k":
        feeds = {"a": np.empty((2, 0), dtype), "b": np.empty((0, 3), dtype),
                 "c": np.array([1, 2, 3], dtype)}
        attrs = {}
    else:
        feeds = {"a": np.zeros((4, 2), dtype), "b": np.empty((2, 0), dtype),
                 "c": np.empty((0,), dtype)}
        attrs = {}
    a, b = feeds["a"].astype(object), feeds["b"].astype(object)
    if attrs.get("transA") and a.ndim > 1:
        a = a.swapaxes(-1, -2)
    if attrs.get("transB"):
        b = b.T
    expected = a @ b * int(attrs.get("alpha", 1))
    if attrs.get("beta", 1) != 0:
        expected = expected + feeds["c"].astype(object) * int(attrs.get("beta", 1))
    return feeds, attrs, expected.astype(dtype)


INTEGER_CASES = [(dtype, case) for dtype in [np.int32, np.uint32, np.int64, np.uint64]
                 for case in ["batched-transposes", "vector", "zero-k", "empty-width"]]
INTEGER_CASES += [(dtype, "full-width") for dtype in [np.int64, np.uint64]]


@pytest.mark.parametrize("dtype,case", INTEGER_CASES[:16])
def test_gemm29_integer_oracle_matches_onnx_reference(dtype, case):
    if onnx.defs.onnx_opset_version() < 29:
        pytest.skip("ONNX reference schemas stop before opset 29; install ONNX from main")
    feeds, attrs, expected = _integer_gemm_case(dtype, case)
    actual = ReferenceEvaluator(schema_runtime.model("Gemm", feeds, attrs, opset=29)).run(None, feeds)[0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("dtype,case", INTEGER_CASES)
def test_gemm29_integer_exact_oracle(dtype, case, eager,
                                   monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    feeds, attrs, expected = _integer_gemm_case(dtype, case)
    session = schema_runtime.session(schema_runtime.model("Gemm", feeds, attrs, opset=29).SerializeToString(),
                       profile=tmp_path / "integer-gemm")
    actual = session.run(None, feeds)[0]
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    schema_runtime.assert_mlx_profile(session)
