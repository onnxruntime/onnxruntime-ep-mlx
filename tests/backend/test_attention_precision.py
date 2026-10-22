"""Forward-error/ULP guards for the exact two-case FP16 comparison policy."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, numpy_helper
from onnx.backend.test.loader import load_model_tests

from _attention_precision import (
    ATTENTION_FP16_TEST_KWARGS,
    assert_fp16_attention_accuracy,
    causal_attention_float64,
)
from _backend import MlxBackend


def test_attention_precision_policy_scope():
    assert set(ATTENTION_FP16_TEST_KWARGS) == {
        "test_attention_4d_causal_fp16",
        "test_attention_4d_causal_fp16_expanded",
    }
    assert all(
        kwargs == {"rtol": 2**-9} for kwargs in ATTENTION_FP16_TEST_KWARGS.values()
    )


@pytest.mark.parametrize("bad", ["three-ulps", "mask", "nan", "forward-error", "dtype", "shape"])
def test_attention_precision_policy_rejects_real_errors(bad):
    expected = np.array([0.39794921875], dtype=np.float16)
    ideal = np.array([0.398295323199067], dtype=np.float64)
    actual = np.array([0.3984375], dtype=np.float16)
    assert_fp16_attention_accuracy(actual, expected, ideal, rtol=0.001, atol=1e-7)
    if bad == "three-ulps":
        actual = np.nextafter(actual, np.float16(np.inf))
    elif bad == "mask":
        actual[:] = 0
    elif bad == "nan":
        actual[:] = np.nan
    elif bad == "forward-error":
        # Even exact fixture agreement cannot excuse a wrong real-arithmetic result.
        actual = expected.copy()
        ideal[:] = 0.4
    elif bad == "dtype":
        actual = actual.astype(np.float32)
    else:
        actual = actual.reshape(1, 1)
    with pytest.raises(AssertionError):
        assert_fp16_attention_accuracy(actual, expected, ideal, rtol=0.001, atol=1e-7)


@pytest.fixture(scope="module")
def official_cases():
    lib = os.environ.get("ONNXRUNTIME_MLX_EP_LIB") or os.environ.get("MLX_EP_LIB")
    if not lib or not Path(lib).is_file():
        pytest.skip("Native MLX library is required for official-case forward-error checks")
    return {
        c.name: c for c in load_model_tests(kind="node")
        if c.name in ATTENTION_FP16_TEST_KWARGS
    }


@pytest.mark.parametrize("name", ATTENTION_FP16_TEST_KWARGS)
def test_official_attention_fp16_forward_error(name, official_cases):
    if name not in official_cases:
        pytest.skip("This ONNX release does not contain the FP16 fixture")
    case = official_cases[name]
    assert case.model is not None and case.data_sets is not None
    backend = MlxBackend.prepare(case.model)
    for inputs, outputs in case.data_sets:
        inputs = [
            numpy_helper.to_array(x) if isinstance(x, TensorProto) else x for x in inputs
        ]
        outputs = [
            numpy_helper.to_array(x) if isinstance(x, TensorProto) else x for x in outputs
        ]
        assert len(inputs) == 3 and len(outputs) == 1
        ideal = causal_attention_float64(*inputs)
        actual, = backend.run(inputs)
        assert_fp16_attention_accuracy(
            actual, outputs[0], ideal, rtol=case.rtol, atol=case.atol
        )
