"""Grouped matrix products with live expert routing and layout contracts."""

import _schema_runtime as schema_runtime
import numpy as np
import pytest


def _grouped_transpose_cases():
    yield "GroupedMatMul", {
        "x": np.arange(1, 7, dtype=np.float32).reshape(2, 3),
        "w": np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3),
        "indices": np.array([[1, 0], [0, 1]], np.int32),
        "bias": np.array([[0.5, -0.5], [1, -1]], np.float32),
    }, {"transB": 1}, (
        np.array([51, 67, 14.5, 31.5, 32.5, 76.5, 123, 166], np.float32).reshape(2, 2, 2),
    )


GROUPED_TRANSPOSE_CASES = list(_grouped_transpose_cases())


@pytest.mark.parametrize("op,feeds,attrs,expected", GROUPED_TRANSPOSE_CASES, ids=[case[0] for case in GROUPED_TRANSPOSE_CASES])
def test_grouped_transpose_goldens_match_onnx_reference(op, feeds, attrs, expected):
    schema_runtime.check_reference(op, feeds, attrs, expected, opset=29)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("op,feeds,attrs,expected", GROUPED_TRANSPOSE_CASES, ids=[case[0] for case in GROUPED_TRANSPOSE_CASES])
def test_grouped_transpose_mlx_against_reference(op, feeds, attrs, expected, eager, monkeypatch, tmp_path):
    schema_runtime.check_mlx(op, feeds, attrs, expected, eager, monkeypatch, tmp_path, opset=29)
