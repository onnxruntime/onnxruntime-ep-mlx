"""AffineGrid geometry, live size specialization, precision, and narrow fallback contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import ml_dtypes
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
from onnx.reference import ReferenceEvaluator
import pytest

import _models as m

DT = ir.DataType


def _model(rank, dtype=DT.FLOAT, *, size=None, interior=False, align=None, opset=20):
    theta_in = m.tensor("theta", DT.FLOAT if dtype == DT.BFLOAT16 else dtype,
                        ["n", rank, rank + 1])
    theta = theta_in
    nodes = []
    if dtype == DT.BFLOAT16:
        theta = m.tensor("theta_bf16", dtype, ["n", rank, rank + 1])
        nodes.append(ir.node("Cast", [theta_in], attributes={"to": int(dtype)}, outputs=[theta]))
    size_in = m.tensor("size", DT.INT64, [rank + 2])
    inputs = [theta_in]
    initializers = []
    if size is not None:
        size_in.const_value = ir.tensor(np.asarray(size, dtype=np.int64))
        initializers.append(size_in)
    else:
        inputs.append(size_in)
    grid_size = size_in
    if interior:
        delta = m.tensor("delta", DT.INT64, [rank + 2])
        inputs.append(delta)
        grid_size = m.tensor("grid_size", DT.INT64, [rank + 2])
        nodes.append(ir.node("Add", [size_in, delta], outputs=[grid_size]))
    grid = m.tensor("grid", dtype, ["n", *[f"s{i}" for i in range(rank)], rank])
    attrs = {} if align is None else {"align_corners": align}
    nodes.append(ir.node("AffineGrid", [theta, grid_size], attributes=attrs, outputs=[grid]))
    out = grid
    if dtype == DT.BFLOAT16:
        out = m.tensor("out", DT.FLOAT, ["n", *[f"s{i}" for i in range(rank)], rank])
        nodes.append(ir.node("Cast", [grid], attributes={"to": int(DT.FLOAT)}, outputs=[out]))
    graph = ir.Graph(inputs, [out], nodes=nodes, initializers=initializers,
                     opset_imports={"": opset}, name="affine_grid_contract")
    model = ir.to_proto(ir.Model(graph, ir_version=11))
    onnx.checker.check_model(model)
    return model.SerializeToString()


def _session(model, *, strict=True):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    if strict:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    session.disable_fallback()
    return session


def _theta(n, rank, dtype=DT.FLOAT, offset=0):
    values = (np.arange(n * rank * (rank + 1), dtype=np.float32) % 11 - 5) / 8 + offset
    return values.reshape(n, rank, rank + 1).astype(
        {DT.FLOAT16: np.float16, DT.DOUBLE: np.float64}.get(dtype, np.float32))


def _expected(theta, size, dtype=DT.FLOAT, align=0):
    # Evaluate each affine expression independently, with width-first coordinates.
    if dtype == DT.BFLOAT16:
        theta = theta.astype(ml_dtypes.bfloat16).astype(np.float32)
    else:
        theta = theta.astype(np.float64 if dtype == DT.DOUBLE else np.float32)
    rank = len(size) - 2
    result = np.empty((size[0], *size[2:], rank),
                      dtype=np.float64 if dtype == DT.DOUBLE else np.float32)
    for index in np.ndindex(*size[2:]):
        coords = [
            (2 * i / (dim - 1) - 1) if align else ((2 * i + 1) / dim - 1)
            for i, dim in zip(index, size[2:], strict=True)
        ][::-1]
        for axis in range(rank):
            result[(slice(None), *index, axis)] = theta[:, axis, -1] + sum(
                theta[:, axis, j] * coord for j, coord in enumerate(coords))
    if dtype == DT.BFLOAT16:
        return result.astype(ml_dtypes.bfloat16).astype(np.float32)
    return result.astype({DT.FLOAT16: np.float16, DT.DOUBLE: np.float64}.get(dtype, np.float32))


def _assert_result(actual, expected, dtype):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # Low-precision payloads accumulate in fp32; doubles must retain real fp64 precision.
    tolerance = {DT.FLOAT: 3e-6, DT.FLOAT16: 1e-3, DT.BFLOAT16: 8e-3, DT.DOUBLE: 1e-12}[dtype]
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.fixture(params=[False, True], ids=["compiled", "eager"])
def execution_mode(request, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if request.param else "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE", "0")


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16, DT.BFLOAT16, DT.DOUBLE])
@pytest.mark.parametrize("kind", ["constant", "boundary", "interior"])
def test_affine_grid_live_geometry(rank, dtype, kind, execution_mode):
    first = [2, 1, *([2] * (rank - 1)), 3]
    session = _session(_model(rank, dtype, size=first if kind == "constant" else None,
                              interior=kind == "interior"))
    sizes = [first] * 4 if kind == "constant" else [
        first, [2, 0, *([3] * (rank - 1)), 2],
        [1, 2**40, *([1] * (rank - 1)), 4], [0, 0, *([2] * rank)], first,
    ]
    for iteration, size in enumerate(sizes):
        theta = _theta(size[0], rank, dtype, offset=iteration / 8)
        feeds = {"theta": theta}
        if kind != "constant":
            feeds["size"] = np.array(size, np.int64)
        if kind == "interior":
            # Same-shaped inputs with changing values must not freeze an interior host read.
            feeds["size"] = np.array(first, np.int64)
            feeds["delta"] = np.array(size, np.int64) - feeds["size"]
        _assert_result(session.run(None, feeds)[0], _expected(theta, size, dtype), dtype)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("align", [None, 0, 1])
@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16, DT.BFLOAT16, DT.DOUBLE])
def test_affine_grid_constant_align_and_opset(rank, align, dtype, execution_mode):
    size = [2, 3, *([3] * rank)]
    model = _model(rank, dtype, size=size, align=align, opset=24)
    theta = _theta(2, rank, dtype)
    actual = _session(model).run(None, {"theta": theta})[0]
    _assert_result(actual, _expected(theta, size, dtype, align or 0), dtype)
    if dtype == DT.FLOAT:
        proto = onnx.load_model_from_string(model)
        reference = ReferenceEvaluator(proto).run(None, {"theta": theta})[0]
        np.testing.assert_allclose(actual, reference, rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(actual, m.run_cpu(model, {"theta": theta})[0],
                                   rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("constant", [False, True])
def test_affine_grid_empty_batch(rank, constant, execution_mode):
    size = [0, 0, *([2] * rank)]
    feeds = {"theta": _theta(0, rank)}
    if not constant:
        feeds["size"] = np.array(size, np.int64)
    model = _model(rank, size=size if constant else None)
    actual = _session(model).run(None, feeds)[0]
    assert actual.shape == (0, *size[2:], rank)
    assert actual.dtype == np.float32


@pytest.mark.parametrize("spatial", [(1, 4), (4, 1), (1, 1),
                                   (1, 2, 3), (2, 1, 3), (2, 3, 1), (1, 1, 1)])
def test_affine_grid_unaligned_singleton_axes(spatial, execution_mode):
    rank = len(spatial)
    size = [2, 1, *spatial]
    theta = _theta(2, rank)
    model = _model(rank, size=size)
    actual = _session(model).run(None, {"theta": theta})[0]
    _assert_result(actual, _expected(theta, size), DT.FLOAT)
    np.testing.assert_allclose(actual, m.run_cpu(model, {"theta": theta})[0],
                               rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("rank", [2, 3])
def test_affine_grid_double_preserves_sub_float32_bits(rank, execution_mode):
    size = [2, 1, *([3] * rank)]
    theta = _theta(2, rank, DT.DOUBLE) + np.float64("0.0000000001234567")
    model = _model(rank, DT.DOUBLE)
    feeds = {"theta": theta, "size": np.array(size, np.int64)}
    actual = _session(model).run(None, feeds)[0]
    _assert_result(actual, _expected(theta, size, DT.DOUBLE), DT.DOUBLE)
    np.testing.assert_allclose(actual, m.run_cpu(model, feeds)[0], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("rank", [2, 3])
def test_affine_grid_invalid_live_size_then_recovery(rank, execution_mode):
    session = _session(_model(rank))
    theta = _theta(2, rank)
    valid = [2, 1, *([2] * rank)]
    for bad, message in [
        ([3, *valid[1:]], "theta must have shape"),
        ([2, -1, *valid[2:]], "nonnegative"),
        ([2, 1, 0, *valid[3:]], "positive spatial"),
        ([2, 1, 2**32, *valid[3:]], "within int32"),
        ([2, 1, 2**30, *valid[3:]], "element count"),
    ]:
        with pytest.raises(Exception, match=message):
            session.run(None, {"theta": theta, "size": np.array(bad, np.int64)})
        feeds = {"theta": theta, "size": np.array(valid, np.int64)}
        _assert_result(session.run(None, feeds)[0], _expected(theta, valid), DT.FLOAT)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("live", [False, True])
def test_affine_grid_unresolved_singleton_is_declined(rank, live):
    size = [2, 1, *([1] * rank)]
    model = _model(rank, size=None if live else size, align=1)
    feeds = {"theta": _theta(2, rank)}
    if live:
        feeds["size"] = np.array(size, np.int64)
    m.assert_op_declined(model, feeds, "AffineGrid")


def _trace_workload():
    ort.register_execution_provider_library("MLXExecutionProvider",
                                           os.environ["ONNXRUNTIME_MLX_EP_LIB"])
    for interior in [False, True]:
        session = _session(_model(3, interior=interior))
        for iteration, spatial in enumerate([(2, 3, 4), (2, 3, 4), (3, 2, 4),
                                            (3, 2, 4), (2, 3, 4)]):
            size = [2, 1, *spatial]
            theta = _theta(2, 3, offset=iteration / 8)
            feeds = {"theta": theta, "size": np.array(size, np.int64)}
            if interior:
                feeds["delta"] = feeds["size"] - np.array([2, 1, 2, 3, 4], np.int64)
                feeds["size"] = np.array([2, 1, 2, 3, 4], np.int64)
            _assert_result(session.run(None, feeds)[0], _expected(theta, size), DT.FLOAT)


def test_affine_grid_value_guarded_compile_and_interior_eager(tmp_path):
    trace = tmp_path / "affine_grid.json"
    subprocess.run(
        [sys.executable, "-c", "import test_affine_grid as t; t._trace_workload()"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0",
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    events = json.loads(trace.read_text())
    paths = [e for e in events if e.get("cat") == "ep.path"]
    assert [(e["name"], e["args"]["cache"]) for e in paths] == [
        ("mlx.compute[general]", cache) for cache in ["MISS", "HIT", "RETRACE", "HIT", "HIT"]
    ] + [("mlx.compute[eager]", "n/a")] * 5
