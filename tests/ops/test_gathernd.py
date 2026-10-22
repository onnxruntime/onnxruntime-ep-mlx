"""GatherND batch, dtype, empty-extent and replay coverage for the MLX EP.

Gathers full slices of ``data`` addressed by the last axis of ``indices``. The output shape is static
(from the operand shapes) so the op is compile-safe; only the index values are runtime. Each case is
proven to run on the MLX EP (per-node profiling) and compared, tolerance-gated, to ORT's CPU EP.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT

import _models as m


def _indices(ishape, dshape):
    idx = np.zeros(ishape, dtype=np.int64)
    width = ishape[-1]
    for mi in np.ndindex(*ishape[:-1]):
        for j in range(width):
            idx[mi + (j,)] = np.random.randint(-dshape[j], dshape[j])  # exercise negatives
    return idx


def _assert_mlx_claims(model, feeds):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = "mlx_gnd_probe"
    sess = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)
    sess.run(None, feeds)
    path = sess.end_profiling()
    try:
        events = json.load(open(path))
    finally:
        os.remove(path)
    providers = {e.get("args", {}).get("provider") for e in events
                 if e.get("cat") == "Node" and e.get("args", {}).get("provider")}
    assert "MLXExecutionProvider" in providers, f"MLX EP did not claim GatherND (ran on {providers})"


def _check(dshape, ishape, seed=0):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    data = rng.standard_normal(dshape).astype(np.float32)
    idx = _indices(ishape, dshape)
    width = ishape[-1]
    oshape = list(ishape[:-1]) + list(dshape[width:])
    model = m.make_model(
        "GatherND",
        [m.tensor("data", DT.FLOAT, list(dshape)), m.tensor("indices", DT.INT64, list(ishape))],
        [m.tensor("out", DT.FLOAT, oshape)],
        attributes={"batch_dims": 0},
    )
    feeds = {"data": data, "indices": idx}
    _assert_mlx_claims(model, feeds)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    expected = ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    actual = m.run_mlx(model, feeds)[0]
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dshape,ishape",
    [
        ((2, 2), (2, 2)),        # full index tuples -> rank-1 output
        ((2, 2), (2, 1)),        # slice a row
        ((2, 3, 4), (2, 2)),     # m=2 -> [.., 4]
        ((2, 3, 4), (5, 3)),     # m=3 scalars
        ((4, 5, 6), (2, 3, 1)),  # nested index grid, m=1
        ((3, 4, 5, 6), (2, 2)),  # 3-D tail
        ((8,), (3, 1)),          # 1-D data
    ],
)
def test_gathernd(dshape, ishape):
    _check(dshape, ishape, seed=hash((dshape, ishape)) & 0xFFFF)


def _batched_reference(data, indices, batch_dims):
    width = indices.shape[-1]
    shape = indices.shape[:-1] + data.shape[batch_dims + width:]
    out = np.empty(shape, dtype=data.dtype)
    for position in np.ndindex(indices.shape[:-1]):
        selection = position[:batch_dims] + tuple(indices[position])
        out[position] = data[selection]
    return out


@pytest.mark.parametrize(
    "dshape,ishape,batch_dims",
    [
        ((2, 3, 4), (2, 5, 1), 1),
        ((2, 3, 4), (2, 5, 2), 1),
        ((2, 3, 4, 5), (2, 3, 2, 1), 2),
        ((2, 3, 4), (2, 1), 1),
        ((2, 3, 4), (0, 2), 0),
        ((2, 3, 0), (4, 2), 0),
        ((0, 3, 4), (0, 5, 1), 1),
        ((2, 3, 4), (2, 0, 2), 1),
        ((2, 0, 3, 4), (2, 0, 5, 1), 2),
        ((3,), (1,), 0),
    ],
)
@pytest.mark.parametrize(
    "dtype,np_dtype",
    [
        (DT.FLOAT, np.float32),
        (DT.FLOAT16, np.float16),
        (DT.DOUBLE, np.float64),
        (DT.INT8, np.int8),
        (DT.INT16, np.int16),
        (DT.INT32, np.int32),
        (DT.INT64, np.int64),
        (DT.UINT8, np.uint8),
        (DT.UINT16, np.uint16),
        (DT.UINT32, np.uint32),
        (DT.UINT64, np.uint64),
        (DT.BOOL, np.bool_),
    ],
)
@pytest.mark.parametrize("eager", [False, True], ids=["compiled", "eager"])
def test_gathernd_batches_and_empty(dshape, ishape, batch_dims, dtype, np_dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    data = np.arange(np.prod(dshape)).reshape(dshape).astype(np_dtype)
    if dtype == DT.INT64:
        data += 2**53
    elif dtype == DT.UINT64:
        data += 2**63
    elif dtype == DT.DOUBLE:
        data += 2**-30
    rng = np.random.default_rng(42)
    indices = np.empty(ishape, np.int64)
    for j in range(ishape[-1]):
        extent = dshape[batch_dims + j]
        indices[..., j] = rng.integers(-extent, extent, size=ishape[:-1])
    expected = _batched_reference(data, indices, batch_dims)
    model = m.make_model(
        "GatherND",
        [m.tensor("data", dtype, dshape), m.tensor("indices", DT.INT64, ishape)],
        [m.tensor("out", dtype, expected.shape)],
        attributes={"batch_dims": batch_dims},
    )
    m.assert_mlx_claims(model, {"data": data, "indices": indices})
    session = m._session(model, m.EP_PROVIDERS)
    for live in [indices, np.where(indices < 0, indices + 1, indices - 1), indices]:
        actual = session.run(None, {"data": data, "indices": live})[0]
        np.testing.assert_array_equal(actual, _batched_reference(data, live, batch_dims))


@pytest.mark.parametrize("eager", [False, True], ids=["compiled", "eager"])
def test_gathernd_symbolic_batches(eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    model = m.make_model(
        "GatherND",
        [
            m.tensor("data", DT.FLOAT, ["B", "N", "D"]),
            m.tensor("indices", DT.INT64, ["B", "P", 1]),
        ],
        [m.tensor("out", DT.FLOAT, ["B", "P", "D"])],
        attributes={"batch_dims": 1},
    )
    session = m._session(model, m.EP_PROVIDERS)
    rng = np.random.default_rng(42)
    for batch, length, width, count in [(1, 3, 4, 2), (2, 5, 3, 4), (2, 3, 0, 1), (1, 3, 4, 2)]:
        data = rng.normal(size=(batch, length, width)).astype(np.float32)
        indices = rng.integers(-length, length, size=(batch, count, 1), dtype=np.int64)
        actual = session.run(None, {"data": data, "indices": indices})[0]
        np.testing.assert_array_equal(actual, _batched_reference(data, indices, 1))


def test_gathernd_shape_keyed_replay(tmp_path):
    trace = tmp_path / "gathernd.json"
    subprocess.run(
        [
            sys.executable, "-m", "pytest",
            f"{Path(__file__).name}::test_gathernd_symbolic_batches[compiled]",
            "-q", "-p", "no:cacheprovider",
        ],
        cwd=Path(__file__).parent,
        env={
            **os.environ,
            "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
            "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0",
            "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0",
        },
        check=True,
        timeout=120,
    )
    paths = [e for e in json.loads(trace.read_text()) if e.get("cat") == "ep.path"]
    assert [e["args"]["path"] for e in paths] == ["general", "general", "eager", "general"]
    assert [e["args"]["cache"] for e in paths] == ["MISS", "RETRACE", "n/a", "HIT"]
