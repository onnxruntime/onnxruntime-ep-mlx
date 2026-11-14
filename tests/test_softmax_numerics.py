"""Regression checks for the softmax numerical audit controls."""

import functools
from itertools import permutations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from bench import softmax_numerics


@pytest.mark.parametrize("count", [2, 3, 4])
def test_variant_order_balances_positions_and_pairs(count):
    variants = tuple(range(count))
    cycle = count if count == 2 else 2 * count
    orders = [
        softmax_numerics.counterbalanced_order(variants, case) for case in range(cycle)
    ]
    for variant in variants:
        for position in range(count):
            assert sum(order[position] == variant for order in orders) == cycle // count
    for first, second in permutations(variants, 2):
        assert (
            sum(order.index(first) < order.index(second) for order in orders)
            == cycle // 2
        )


@pytest.mark.parametrize("count", [2, 3, 4])
def test_reversed_sweep_mirrors_each_case(count):
    variants = tuple(range(count))
    for case in range(2 * count):
        original = softmax_numerics.counterbalanced_order(variants, case)
        reverse = softmax_numerics.counterbalanced_order(variants, case, reverse=True)
        assert reverse == original[::-1]


def test_attention_upcast_changes_precision_not_softmax_formula():
    logits = np.array([3.796875, 2.6875, 4.4375], dtype=np.float32)
    q = mx.array([[[[1.0]]]], dtype=mx.bfloat16)
    k = mx.array(logits.reshape(1, 1, 3, 1), dtype=mx.bfloat16)
    v = mx.array(np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3), dtype=mx.bfloat16)
    function = mx.compile(
        functools.partial(softmax_numerics.composed_attention, variant="upcast")
    )
    actual = np.array(function(q, k, v).astype(mx.float32)).reshape(-1)

    exp_logits = np.exp(logits.astype(np.float64) - logits.max())
    expected = exp_logits / exp_logits.sum()
    expected = np.array(
        mx.array(expected.astype(np.float32), dtype=mx.bfloat16).astype(mx.float32)
    )
    np.testing.assert_array_equal(actual, expected)


def test_compiled_upcast_retains_plain_logsumexp_cancellation():
    logits = mx.array([[0.0, -64.0, -64.0]], dtype=mx.bfloat16)
    upcast = mx.compile(softmax_numerics.upcast_logsoftmax)
    paper = mx.compile(softmax_numerics.paper_logsoftmax)
    assert np.array(upcast(logits).astype(mx.float32))[0, 0] == 0
    assert np.array(paper(logits).astype(mx.float32))[0, 0] < -3e-28


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_compiled_upcast_remains_sensitive_to_large_common_offset(dtype):
    logits = mx.array([[32768.0, 32768.0, 32768.0]], dtype=dtype)
    expected = np.array(mx.array([-np.log(3)], dtype=dtype).astype(mx.float32))[0]
    upcast = mx.compile(softmax_numerics.upcast_logsoftmax)
    paper = mx.compile(softmax_numerics.paper_logsoftmax)
    assert np.array(upcast(logits).astype(mx.float32))[0, 0] != expected
    assert np.array(paper(logits).astype(mx.float32))[0, 0] == expected
