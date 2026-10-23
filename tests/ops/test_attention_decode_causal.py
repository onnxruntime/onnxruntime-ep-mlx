"""Cached causal attention must not replay a first-call, host-sized zero mask."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from onnx import TensorProto as T, helper, numpy_helper
import pytest

from test_metadata_only_shape import _session


PAST_LENGTHS = [1, 1, 3, 2, 1, 7, 31, 127, 257, 1023, 4095, 8191, 127, 3, 1, 0, 1, 1023, 1]


def _run_growth(control, geometry="single"):
    query_length = 2 if geometry == "multiquery" else 1
    query_shape = [1, query_length, 8]
    key_shape = [1, "keys" if geometry == "changing_keys" else query_length, 8]
    inputs = [helper.make_tensor_value_info(name, T.FLOAT, shape)
              for name, shape in [("q", query_shape), ("k", key_shape), ("v", key_shape)]]
    inputs.extend(helper.make_tensor_value_info(name, T.FLOAT, [1, 1, "past", 8])
                  for name in ["pk", "pv"])
    initializers = []
    mask_name = ""
    if control == "external_zero":
        mask_name = "mask"
        initializers.append(numpy_helper.from_array(np.zeros(1, np.float32), name=mask_name))
    nodes = [
        helper.make_node("Attention", ["q", "k", "v", mask_name, "pk", "pv"],
                         ["y", "present_k", "present_v"], q_num_heads=1, kv_num_heads=1,
                         is_causal=int(control != "noncausal")),
        helper.make_node("Shape", ["q"], ["geometry"]),
        helper.make_node("Size", ["q"], ["count"]),
        helper.make_node("ReduceSum", ["geometry"], ["extent_sum"], keepdims=0),
        helper.make_node("Add", ["extent_sum", "count"], ["metadata_sum"]),
        helper.make_node("Cast", ["metadata_sum"], ["metadata_float"], to=T.FLOAT),
        helper.make_node("Add", ["y", "metadata_float"], ["adjusted_y"]),
    ]
    outputs = [helper.make_tensor_value_info("adjusted_y", T.FLOAT, query_shape),
               helper.make_tensor_value_info("present_k", T.FLOAT, [1, 1, "total", 8]),
               helper.make_tensor_value_info("present_v", T.FLOAT, [1, 1, "total", 8]),
               helper.make_tensor_value_info("geometry", T.INT64, [3]),
               helper.make_tensor_value_info("count", T.INT64, [])]
    model = helper.make_model(helper.make_graph(nodes, "causal_decode_growth", inputs, outputs,
                                               initializer=initializers),
                              opset_imports=[helper.make_opsetid("", 24)], ir_version=11)
    session = _session(model.SerializeToString())
    for invocation, past in enumerate(PAST_LENGTHS):
        current_keys = [1, 2, 3, 1][invocation % 4] if geometry == "changing_keys" else query_length
        value = past + invocation if invocation < 5 else 20 + invocation - 5
        query = value if invocation < 5 else 3
        key = value if invocation < 5 else 4
        current_v = np.full((1, current_keys, 8), value, np.float32)
        if current_keys > 1:
            current_v[:, 1:, :] += np.arange(1, current_keys, dtype=np.float32)[None, :, None] * 10
        feeds = {"q": np.full(query_shape, query, np.float32),
                 "k": np.full((1, current_keys, 8), key, np.float32), "v": current_v,
                 "pk": np.ones((1, 1, past, 8), np.float32),
                 "pv": np.full((1, 1, past, 8), 1 if invocation < 5 else -7, np.float32)}
        snapshots = {name: data.copy() for name, data in feeds.items()}
        actual = session.run(None, feeds)
        keys = np.concatenate([snapshots["pk"], snapshots["k"].reshape(1, 1, current_keys, 8)], axis=2)
        values = np.concatenate([snapshots["pv"], snapshots["v"].reshape(1, 1, current_keys, 8)], axis=2)
        np.testing.assert_array_equal(actual[1], keys)
        np.testing.assert_array_equal(actual[2], values)
        np.testing.assert_array_equal(actual[3], query_shape)
        np.testing.assert_array_equal(actual[4], np.prod(query_shape))
        for name, data in feeds.items():
            np.testing.assert_array_equal(data, snapshots[name], err_msg=f"mutated {name}")
        scores = snapshots["q"].astype(np.float64) @ keys[0, 0].astype(np.float64).T / np.sqrt(8)
        if control != "noncausal":
            allowed = np.arange(past + current_keys)[None, :] <= past + np.arange(query_length)[:, None]
            scores = np.where(allowed, scores, -np.inf)
        probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        expected = probabilities @ values[0, 0].astype(np.float64)
        expected += sum(query_shape) + np.prod(query_shape)
        np.testing.assert_allclose(actual[0], expected, rtol=1e-5, atol=1e-6,
                                   err_msg=f"invocation={invocation}, past={past}, keys={current_keys}")


@pytest.mark.parametrize("eager", [False, True], ids=["compiled", "eager"])
@pytest.mark.parametrize("control", ["causal", "noncausal", "external_zero"])
def test_attention_decode_causal_growth(control, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    _run_growth(control)


@pytest.mark.parametrize("geometry", ["changing_keys", "multiquery"])
@pytest.mark.parametrize("eager", [False, True], ids=["default", "eager"])
def test_attention_causal_alignment_is_not_assumed(geometry, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    _run_growth("causal", geometry)


@pytest.mark.parametrize("control", ["causal", "noncausal", "external_zero"])
def test_attention_decode_causal_growth_routes(control, tmp_path):
    trace = tmp_path / "causal_decode.json"
    subprocess.run(
        [sys.executable, "-m", "pytest",
         f"{Path(__file__).name}::test_attention_decode_causal_growth[{control}-compiled]",
         "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
        check=True, timeout=60,
    )
    events = json.loads(trace.read_text())
    decode = [event for event in events if event.get("name") == "mlx.compute[decode]"
              and "Attention" in event.get("args", {}).get("ops", "").split(",")]
    assert [event["args"].get("cache") for event in decode] == ["MISS"] + ["HIT"] * (len(PAST_LENGTHS) - 1)
    assert len({event["args"]["partition_id"] for event in decode}) == 1
