"""Strict RoiAlign placement, live geometry, precision, and independent sampling oracle.

Contracts: ONNX 1.23.2 schemas 10/16/22 and ORT 1.31 roialign.cc. Max pooling
reduces the four weighted corners, not their bilinear sum. Adaptive empty grids
contribute zero; output_half_pixel first clamps ROI extents to at least one.
"""

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
import pytest

import _models as m

DT = ir.DataType


def reference(x, rois, batches, *, opset=22, mode="avg", output_height=1,
              output_width=1, sampling_ratio=0, spatial_scale=1.0,
              coordinate_transformation_mode=None):
    """Direct per-ROI/bin/sample equations, independent of the tensor lowering."""
    f = np.float64 if x.dtype == np.float64 else np.float32
    half = coordinate_transformation_mode == "half_pixel" or (
        coordinate_transformation_mode is None and opset >= 16
    )
    offset = f(0.5 if half else 0)
    scale = f(np.float32(spatial_scale))
    out = np.zeros((len(rois), x.shape[1], output_height, output_width), dtype=x.dtype)
    for r, (roi, batch) in enumerate(zip(rois, batches, strict=True)):
        x1, y1, x2, y2 = [f(f(v) * scale - offset) for v in roi]
        rh, rw = f(y2 - y1), f(x2 - x1)
        if not half:
            rh, rw = max(rh, f(1)), max(rw, f(1))
        bh, bw = f(rh / f(output_height)), f(rw / f(output_width))
        gh = sampling_ratio or max(0, int(np.ceil(bh)))
        gw = sampling_ratio or max(0, int(np.ceil(bw)))
        if gh == 0 or gw == 0:
            continue
        for c in range(x.shape[1]):
            image = x[int(batch), c].astype(f)
            for ph in range(output_height):
                for pw in range(output_width):
                    samples = []
                    for iy in range(gh):
                        yy = f(f(y1 + f(f(ph) * bh)) + f(f(f(iy + 0.5) * bh) / f(gh)))
                        for ix in range(gw):
                            xx = f(f(x1 + f(f(pw) * bw)) + f(f(f(ix + 0.5) * bw) / f(gw)))
                            if yy < -1 or yy > image.shape[0] or xx < -1 or xx > image.shape[1]:
                                terms = [f(0)] * 4
                            else:
                                y = min(max(yy, f(0)), f(image.shape[0] - 1))
                                z = min(max(xx, f(0)), f(image.shape[1] - 1))
                                y0, z0 = int(np.floor(y)), int(np.floor(z))
                                y3 = min(y0 + 1, image.shape[0] - 1)
                                z3 = min(z0 + 1, image.shape[1] - 1)
                                ly, lx = f(y - f(y0)), f(z - f(z0))
                                weights = [f(f(1 - ly) * f(1 - lx)), f(f(1 - ly) * lx),
                                           f(ly * f(1 - lx)), f(ly * lx)]
                                pixels = [image[y0, z0], image[y0, z3],
                                          image[y3, z0], image[y3, z3]]
                                terms = [f(w * v) for w, v in zip(weights, pixels, strict=True)]
                            samples.append(max(terms) if mode == "max" else
                                           f(f(f(terms[0] + terms[1]) + terms[2]) + terms[3]))
                    if samples:
                        out[r, c, ph, pw] = (max(samples) if mode == "max" else
                                            np.sum(samples, dtype=f) / f(gh * gw))
    return out


def model(*, dtype=DT.FLOAT, opset=22, attrs=None, interior=False, constant=False):
    attrs = attrs or {}
    x = m.tensor("x", dtype, ["n", "c", "h", "w"])
    rois = m.tensor("rois", dtype, ["r", 4])
    batch = m.tensor("batch", DT.INT64, ["r"])
    y = m.tensor("y", dtype, ["r", "c", attrs.get("output_height", 1),
                               attrs.get("output_width", 1)])
    nodes = []
    inputs = [x, rois, batch]
    initializers = []
    actual_rois = rois
    actual_batch = batch
    if constant:
        rois.const_value = ir.tensor(np.array([[0, 0, 3, 3]], dtype=np.float32))
        batch.const_value = ir.tensor(np.array([0], dtype=np.int64))
        initializers = [rois, batch]
        inputs = [x]
    if interior:
        delta = m.tensor("delta", dtype, ["r", 4])
        actual_rois = m.tensor("live_rois", dtype, ["r", 4])
        actual_batch = m.tensor("live_batch", DT.INT64, ["r"])
        nodes.extend([
            ir.node("Add", [rois, delta], outputs=[actual_rois]),
            ir.node("Identity", [batch], outputs=[actual_batch]),
        ])
        inputs.append(delta)
    nodes.append(ir.node("RoiAlign", [x, actual_rois, actual_batch],
                         attributes=attrs, outputs=[y]))
    graph = ir.Graph(inputs, [y], nodes=nodes, initializers=initializers,
                     opset_imports={"": opset})
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def strict_session(data, feeds, tmp_path):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "roi_claim")
    session = ort.InferenceSession(data, options, providers=m.EP_PROVIDERS)
    session.disable_fallback()
    session.run(None, feeds)
    profile = Path(session.end_profiling())
    try:
        events = json.loads(profile.read_text())
    finally:
        profile.unlink()
    nodes = [e["args"] for e in events if e.get("cat") == "Node"
             and e.get("args", {}).get("provider")]
    assert any(n["provider"] == "MLXExecutionProvider" for n in nodes)
    assert not any(n["provider"] == "CPUExecutionProvider" and n.get("op_name") == "RoiAlign"
                   for n in nodes), nodes
    return session


@pytest.fixture(params=[False, True], ids=["compiled", "eager"])
def execution_mode(request, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if request.param else "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE", "0")


@pytest.mark.parametrize("opset", [10, 16, 22])
@pytest.mark.parametrize("mode", ["avg", "max"])
@pytest.mark.parametrize("ratio", [None, 0, 1, 3])
def test_roi_align_live_grids(opset, mode, ratio, execution_mode, tmp_path):
    attrs = dict(mode=mode, output_height=2, output_width=3, spatial_scale=0.75)
    if ratio is not None:
        attrs["sampling_ratio"] = ratio
    data = model(opset=opset, attrs=attrs)
    x = (np.arange(2 * 2 * 5 * 6, dtype=np.float32).reshape(2, 2, 5, 6) % 19) - 12
    feeds = {"x": x, "rois": np.array([[0, 0, 1, 1], [-2, -1, 7, 6]], dtype=np.float32),
             "batch": np.array([0, 1], dtype=np.int64)}
    session = strict_session(data, feeds, tmp_path)
    cpu_model = onnx.load_model_from_string(data)
    cpu_model.opset_import[0].version = min(opset, 16)
    cpu = ort.InferenceSession(cpu_model.SerializeToString(), providers=["CPUExecutionProvider"])
    for rois, batches, shift in [
        ([[0, 0, 1, 1], [-2, -1, 7, 6]], [0, 1], 0),
        ([[0, 0, 7, 6], [1, 1, 2, 2]], [1, 0], 3),
        ([[-8, -8, -4, -4], [0, 0, 4, 5]], [0, 1], -2),
    ]:
        feeds["rois"] = np.array(rois, dtype=np.float32)
        feeds["batch"] = np.array(batches, dtype=np.int64)
        feeds["x"] = x + shift
        want = reference(feeds["x"], feeds["rois"], feeds["batch"], opset=opset, **attrs)
        np.testing.assert_allclose(cpu.run(None, feeds)[0], want, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(session.run(None, feeds)[0], want, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("ctm", ["half_pixel", "output_half_pixel"])
@pytest.mark.parametrize("mode", ["avg", "max"])
@pytest.mark.parametrize("ratio", [0, 2])
def test_roi_align_boundary_and_degenerate(ctm, mode, ratio, execution_mode, tmp_path):
    attrs = dict(coordinate_transformation_mode=ctm, mode=mode, sampling_ratio=ratio)
    data = model(attrs=attrs)
    rois = [[v, v, v, v] for v in [-2, -1, -0.999, 0, 0.5, 2.5, 3, 3.001, 6]]
    rois += [[2, 2, 0, 0], [1, 0, 1, 3], [0, 2, 3, 2]]
    if ratio == 0 and ctm == "half_pixel":
        rois += [[0, 0, 3_000_000_000, 0], [0, 0, 0, 3_000_000_000]]
    feeds = {"x": np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3) - 6,
             "rois": np.array(rois, dtype=np.float32),
             "batch": np.zeros(len(rois), dtype=np.int64)}
    session = strict_session(data, feeds, tmp_path)
    want = reference(feeds["x"], feeds["rois"], feeds["batch"], **attrs)
    np.testing.assert_allclose(session.run(None, feeds)[0], want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("ratio", [0, 2])
@pytest.mark.parametrize("constant", [False, True])
def test_roi_align_interior_and_constant(ratio, constant, execution_mode, tmp_path):
    attrs = dict(sampling_ratio=ratio, output_height=2, output_width=2)
    data = model(attrs=attrs, interior=not constant, constant=constant)
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    rois = np.array([[0, 0, 3, 3]], dtype=np.float32)
    batch = np.array([0], dtype=np.int64)
    feeds = {"x": x}
    if not constant:
        feeds.update(rois=rois, batch=batch, delta=np.zeros_like(rois))
    session = strict_session(data, feeds, tmp_path)
    for delta, value in [(0, 1), (1.5, -2), (-0.25, 3)]:
        feeds["x"] = x * value
        if not constant:
            feeds["delta"].fill(delta)
        actual_rois = rois if constant else rois + delta
        want = reference(feeds["x"], actual_rois, batch, **attrs)
        np.testing.assert_allclose(session.run(None, feeds)[0], want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [DT.FLOAT16, DT.BFLOAT16, DT.DOUBLE])
@pytest.mark.parametrize("ratio", [0, 2])
@pytest.mark.parametrize("mode", ["avg", "max"])
def test_roi_align_precision(dtype, ratio, mode, execution_mode, tmp_path):
    np_dtype = {DT.FLOAT16: np.float16, DT.BFLOAT16: ml_dtypes.bfloat16,
                DT.DOUBLE: np.float64}[dtype]
    attrs = dict(sampling_ratio=ratio, mode=mode, output_height=2, output_width=2)
    data = model(dtype=DT.FLOAT if dtype == DT.BFLOAT16 else dtype, attrs=attrs)
    if dtype == DT.BFLOAT16:
        proto = onnx.load_model_from_string(data)
        roi_node = proto.graph.node[0]
        for index in [0, 1]:
            source = roi_node.input[index]
            roi_node.input[index] = source + "_bf"
            proto.graph.node.insert(index, onnx.helper.make_node(
                "Cast", [source], [source + "_bf"], to=int(DT.BFLOAT16)))
        roi_node.output[0] = "y_bf"
        proto.graph.node.append(onnx.helper.make_node("Cast", ["y_bf"], ["y"], to=int(DT.FLOAT)))
        data = proto.SerializeToString()
    x = (np.arange(32).reshape(2, 1, 4, 4) - 19).astype(np_dtype)
    if dtype == DT.DOUBLE:
        x = x * (1 + 2**-30) + 2**-35
    feeds = {"x": x, "rois": np.array([[0.125, 0.25, 3.125, 3.25]], dtype=np_dtype),
             "batch": np.array([1], dtype=np.int64)}
    if dtype == DT.BFLOAT16:
        feeds = {k: v.astype(np.float32) if v.dtype == ml_dtypes.bfloat16 else v
                 for k, v in feeds.items()}
    session = strict_session(data, feeds, tmp_path)
    for shift in [0, 0.125, -0.5]:
        feeds["rois"] += shift
        values = {k: v.astype(np_dtype).astype(np.float32) if dtype == DT.BFLOAT16
                  and v.dtype == np.float32 else v for k, v in feeds.items()}
        want = reference(values["x"], values["rois"], values["batch"], **attrs)
        if dtype == DT.BFLOAT16:
            want = want.astype(ml_dtypes.bfloat16).astype(np.float32)
        tolerance = {DT.FLOAT16: 1e-3, DT.BFLOAT16: 0, DT.DOUBLE: 2e-14}[dtype]
        np.testing.assert_allclose(session.run(None, feeds)[0], want,
                                   rtol=tolerance, atol=tolerance)
        if dtype == DT.DOUBLE:
            proto = onnx.load_model_from_string(data)
            proto.opset_import[0].version = 16
            np.testing.assert_allclose(m.run_cpu(proto.SerializeToString(), feeds)[0], want,
                                       rtol=2e-14, atol=2e-14)


def test_roi_align_all_defaults(execution_mode, tmp_path):
    data = model()
    feeds = {"x": np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4),
             "rois": np.array([[0, 0, 4, 4]], dtype=np.float32),
             "batch": np.zeros(1, dtype=np.int64)}
    session = strict_session(data, feeds, tmp_path)
    want = reference(feeds["x"], feeds["rois"], feeds["batch"])
    np.testing.assert_allclose(session.run(None, feeds)[0], want, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [0.0, -0.5, 2.0])
@pytest.mark.parametrize("ctm", ["half_pixel", "output_half_pixel"])
def test_roi_align_scale_contract(scale, ctm, execution_mode, tmp_path):
    attrs = dict(spatial_scale=scale, coordinate_transformation_mode=ctm)
    data = model(attrs=attrs)
    feeds = {"x": np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4),
             "rois": np.array([[0, 0, 2, 3], [3, 3, 1, 1]], dtype=np.float32),
             "batch": np.zeros(2, dtype=np.int64)}
    session = strict_session(data, feeds, tmp_path)
    want = reference(feeds["x"], feeds["rois"], feeds["batch"], **attrs)
    np.testing.assert_allclose(session.run(None, feeds)[0], want, rtol=0, atol=0)


def test_roi_align_double_adaptive_threshold_and_empty(execution_mode, tmp_path):
    data = model(dtype=DT.DOUBLE)
    feeds = {"x": np.arange(16, dtype=np.float64).reshape(1, 1, 4, 4) ** 2,
             "rois": np.array([[0, 0, 1 + 2**-40, 1]], dtype=np.float64),
             "batch": np.zeros(1, dtype=np.int64)}
    session = strict_session(data, feeds, tmp_path)
    for roi in [[0, 0, 1 + 2**-40, 1], [0, 0, 1, 1],
                [0, 0, 3_000_000_000, 0], [0, 0, 0, 3_000_000_000]]:
        feeds["rois"] = np.array([roi], dtype=np.float64)
        want = reference(feeds["x"], feeds["rois"], feeds["batch"])
        np.testing.assert_allclose(session.run(None, feeds)[0], want, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("ratio", [0, 2])
@pytest.mark.parametrize("interior", [False, True])
def test_roi_align_route_replay(ratio, interior, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE", "0")
    attrs = dict(sampling_ratio=ratio, output_height=2, output_width=2)
    data = model(attrs=attrs, interior=interior)
    session = ort.InferenceSession(data, providers=m.EP_PROVIDERS)
    session.disable_fallback()
    feeds = {"x": np.arange(32, dtype=np.float32).reshape(2, 1, 4, 4),
             "rois": np.array([[0, 0, 3, 3]], dtype=np.float32),
             "batch": np.zeros(1, dtype=np.int64)}
    if interior:
        feeds["delta"] = np.zeros_like(feeds["rois"])
    for count, end, batch, delta in [(1, 3, 0, 0), (1, 1, 1, 0.25),
                                      (2, 4, 0, -0.25), (1, 3, 1, 0.5)]:
        feeds["x"] += 0.125
        feeds["rois"] = np.tile(np.array([[0, 0, end, end]], dtype=np.float32), (count, 1))
        feeds["batch"] = np.full(count, batch, dtype=np.int64)
        if interior:
            feeds["delta"] = np.full_like(feeds["rois"], delta)
        live_rois = feeds["rois"] + delta if interior else feeds["rois"]
        want = reference(feeds["x"], live_rois, feeds["batch"], **attrs)
        np.testing.assert_allclose(session.run(None, feeds)[0], want, rtol=1e-6, atol=1e-6)
    good = {key: value.copy() for key, value in feeds.items()}
    for bad in ["batch", "rois"]:
        if bad == "batch":
            feeds["batch"][0] = 2**32
        else:
            feeds["rois"][0, 0] = np.nan
        with pytest.raises(Exception, match="RoiAlign:"):
            session.run(None, feeds)
        feeds = {key: value.copy() for key, value in good.items()}
    np.testing.assert_allclose(session.run(None, feeds)[0], want, rtol=1e-6, atol=1e-6)


def test_roi_align_observed_compile_routes(tmp_path):
    trace = tmp_path / "roi_routes.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "roi_align_route_replay", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0",
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path" and "RoiAlign" in event["args"].get("ops", "")]
    assert [event["cache"] for event in paths if event["path"] == "general"] == [
        "MISS", "HIT", "RETRACE", "HIT", "HIT"]
    assert sum(event["path"] == "eager" for event in paths) == 15


@pytest.mark.parametrize("ratio", [0, 2])
@pytest.mark.parametrize("shape,count", [([0, 2, 3, 3], 0), ([1, 0, 3, 3], 1),
                                        ([1, 2, 3, 3], 0)])
def test_roi_align_empty(shape, count, ratio, execution_mode, tmp_path):
    data = model(attrs={"sampling_ratio": ratio})
    feeds = {"x": np.empty(shape, dtype=np.float32),
             "rois": np.zeros((count, 4), dtype=np.float32),
             "batch": np.zeros(count, dtype=np.int64)}
    session = strict_session(data, feeds, tmp_path)
    result = session.run(None, feeds)[0]
    assert result.shape == (count, shape[1], 1, 1)
    assert result.dtype == np.float32


@pytest.mark.parametrize("ratio", [0, 2])
@pytest.mark.parametrize("bad,match", [
    ("negative_batch", "batch_indices must be within"),
    ("large_batch", "batch_indices must be within"),
    ("nan_roi", "ROI coordinates must be finite"),
    ("inf_roi", "ROI coordinates must be finite"),
    ("overflow_roi", "scaled ROI extents must be finite"),
    ("mismatched_batch", "expected X"),
])
def test_roi_align_invalid_recovery(ratio, bad, match, execution_mode, tmp_path):
    data = model(attrs={"sampling_ratio": ratio, "spatial_scale": 2.0})
    good = {"x": np.ones((1, 1, 3, 3), dtype=np.float32),
            "rois": np.array([[0, 0, 2, 2]], dtype=np.float32),
            "batch": np.zeros(1, dtype=np.int64)}
    session = strict_session(data, good, tmp_path)
    feeds = {k: v.copy() for k, v in good.items()}
    if bad == "negative_batch":
        feeds["batch"][0] = -1
    elif bad == "large_batch":
        feeds["batch"][0] = 2**32
    elif bad in {"nan_roi", "inf_roi", "overflow_roi"}:
        feeds["rois"][0, 2] = {"nan_roi": np.nan, "inf_roi": np.inf,
                                "overflow_roi": np.finfo(np.float32).max}[bad]
    else:
        feeds["batch"] = np.zeros(2, dtype=np.int64)
    with pytest.raises(Exception, match=match):
        session.run(None, feeds)
    want = reference(good["x"], good["rois"], good["batch"],
                     sampling_ratio=ratio, spatial_scale=2.0)
    np.testing.assert_allclose(session.run(None, good)[0], want, rtol=1e-6, atol=1e-6)
