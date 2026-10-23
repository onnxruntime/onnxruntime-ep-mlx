"""Default random keys stay live across compiled neighbors and nested graphs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT

import _models as m


OPS = ["RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike",
       "Bernoulli", "Multinomial"]


def _model(op, seed):
    categorical = op == "Multinomial"
    shape = [2, 256] if categorical else [256]
    dtype = DT.INT64 if categorical else DT.FLOAT
    bias, random, output = [m.tensor(name, dtype, shape) for name in ["bias", "random", "y"]]
    inputs = [bias]
    feeds = {"bias": np.zeros(shape, np.int64 if categorical else np.float32)}
    random_inputs = []
    attributes = {} if seed is None else {"seed": seed}
    if op.endswith("Like"):
        x = m.tensor("x", DT.INT32, shape)
        inputs.append(x)
        random_inputs.append(x)
        feeds["x"] = np.arange(256, dtype=np.int32)
        attributes["dtype"] = int(DT.FLOAT)
    elif op == "Bernoulli":
        p = m.tensor("p", DT.FLOAT, shape)
        inputs.append(p)
        random_inputs.append(p)
        feeds["p"] = np.full(shape, 0.5, np.float32)
        attributes["dtype"] = int(DT.FLOAT)
    elif categorical:
        logits = m.tensor("logits", DT.FLOAT, [2, 4])
        inputs.append(logits)
        random_inputs.append(logits)
        feeds["logits"] = np.zeros((2, 4), np.float32)
        attributes.update(dtype=int(DT.INT64), sample_size=256)
    else:
        attributes.update(shape=shape, dtype=int(DT.FLOAT))
    nodes = [
        ir.node(op, random_inputs, attributes=attributes, outputs=[random]),
        ir.node("Add", [random, bias], outputs=[output]),
    ]
    graph = ir.Graph(inputs, [output], nodes=nodes, name="live_random_key",
                     opset_imports={"": 17})
    return ir.to_proto(ir.Model(graph, ir_version=9)).SerializeToString(), feeds


def _session(model):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(model, options, providers=["MLXExecutionProvider"])


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("seed", [None, 41.0], ids=["default-key", "explicit-seed"])
@pytest.mark.parametrize("eager", [False, True], ids=["default", "eager"])
def test_random_key_replay(monkeypatch, op, seed, eager):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE", "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PREFILL_COMPILE", "0")
    model, feeds = _model(op, seed)
    session = _session(model)
    outputs = []
    for call in range(4):
        if "x" in feeds:
            feeds["x"] = np.arange(256, dtype=np.int32) * (call + 1)
        before = {name: value.tobytes() for name, value in feeds.items()}
        output = session.run(None, feeds)[0].copy()
        assert output.shape == feeds["bias"].shape
        assert output.dtype == feeds["bias"].dtype
        assert np.isfinite(output).all()
        assert all(value.tobytes() == before[name] for name, value in feeds.items())
        if op.startswith("RandomUniform"):
            assert ((output >= 0) & (output < 1)).all()
        elif op == "Bernoulli":
            assert np.isin(output, [0, 1]).all()
        elif op == "Multinomial":
            assert ((output >= 0) & (output < 4)).all()
        outputs.append(output)
    for left, right in zip(outputs, outputs[1:]):
        assert np.array_equal(left, right) == (seed is not None)


def test_random_key_actual_routes(tmp_path):
    trace = tmp_path / "random-key.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name, "-k", "random_key_replay",
         "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path"]
    for op in OPS:
        events = [event for event in paths if op in event.get("ops", "").split(",")]
        assert sum(event["path"] == "eager" for event in events) == 12
        assert [event["cache"] for event in events if event["path"] == "general"] == [
            "MISS", "HIT", "HIT", "HIT",
        ]
    adds = [event for event in paths
            if event.get("ops", "").split(",") == ["Add"] and event["path"] == "general"]
    assert len(adds) == 24
    assert sum(event["cache"] == "MISS" for event in adds) == 6
    assert sum(event["cache"] == "HIT" for event in adds) == 18


@pytest.mark.parametrize("eager", [False, True], ids=["default", "eager"])
def test_random_default_key_inside_if(monkeypatch, eager):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE", "0")
    branches = []
    for name in ["then", "else"]:
        output = m.tensor(f"{name}_output", DT.FLOAT, [256])
        branches.append(ir.Graph(
            [], [output], nodes=[ir.node(
                "RandomUniform", [], attributes={"shape": [256]}, outputs=[output],
            )], name=name, opset_imports={"": 17},
        ))
    condition = ir.Value(name="condition", const_value=ir.tensor(np.asarray(True)))
    random = m.tensor("random", DT.FLOAT, [256])
    bias = m.tensor("bias", DT.FLOAT, [256])
    output = m.tensor("y", DT.FLOAT, [256])
    nodes = [
        ir.node("If", [condition],
                attributes={"then_branch": branches[0], "else_branch": branches[1]},
                outputs=[random]),
        ir.node("Add", [random, bias], outputs=[output]),
    ]
    graph = ir.Graph([bias], [output], nodes=nodes, initializers=[condition],
                     name="nested_default_key", opset_imports={"": 17})
    session = _session(ir.to_proto(ir.Model(graph, ir_version=9)).SerializeToString())
    outputs = [session.run(None, {"bias": np.zeros(256, np.float32)})[0].copy()
               for _ in range(4)]
    assert all(np.isfinite(output).all() and ((output >= 0) & (output < 1)).all()
               for output in outputs)
    assert all(not np.array_equal(left, right) for left, right in zip(outputs, outputs[1:]))


def test_random_nested_actual_routes(tmp_path):
    trace = tmp_path / "nested-random-key.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "random_default_key_inside_if", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path"]
    parents = [event for event in paths if "If" in event.get("ops", "").split(",")]
    assert len(parents) == 8 and all(event["path"] == "eager" for event in parents)
    adds = [event for event in paths
            if event.get("ops", "").split(",") == ["Add"] and event["path"] == "general"]
    assert [event["cache"] for event in adds] == ["MISS", "HIT", "HIT", "HIT"]
