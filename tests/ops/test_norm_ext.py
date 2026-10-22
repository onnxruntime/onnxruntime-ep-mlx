"""Correctness tests for the MLX EP extended-normalization op family.

Each op runs through the MLX EP (CPU fallback available) and is compared against ORT's CPU EP,
tolerance-gated (fp32 tight, fp16 loose). Scale/bias/mean/var are supplied as runtime inputs — the
EP does not require them to be constant. Standard norms also exercise suffix axes,
broadcast affine shapes, explicit stash precision, and optional statistic outputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import ml_dtypes
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

import _models as m

DT = ir.DataType

_IR_OF = {np.dtype("float32"): DT.FLOAT, np.dtype("float16"): DT.FLOAT16, np.dtype("float64"): DT.DOUBLE}
_TOL = {
    np.dtype("float32"): dict(rtol=1e-4, atol=1e-5),
    np.dtype("float16"): dict(rtol=2e-2, atol=2e-2),
}
DTYPES = [np.float32, np.float16]


def ir_of(dt) -> ir.DataType:
    return _IR_OF[np.dtype(dt)]


def tol(dt) -> dict:
    return _TOL[np.dtype(dt)]


def build(
    op: str,
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    *,
    attrs: list[ir.Attr] | None = None,
    domain: str = "",
    opset: int = 24,
) -> bytes:
    node = ir.node(
        op,
        inputs,
        attributes={attr.name: attr for attr in (attrs or [])},
        domain=domain,
        outputs=outputs,
    )
    imports = {"": opset}
    if domain:
        imports[domain] = 1
    graph = ir.Graph(inputs, outputs, nodes=[node], opset_imports=imports, name=f"mlx_{op}")
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def randn(dt, *shape) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(shape).astype(dt)


def _strict_mlx_session(model):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"])
    session.disable_fallback()
    return session


def _norm_reference(x, scale, axis, centered, *, stash=np.float32, bias=None, epsilon=1e-5):
    axes = tuple(range(axis % x.ndim, x.ndim))
    accumulation = np.float64 if np.dtype(stash) == np.dtype(np.float64) else np.float32
    value = x.astype(stash)
    mean = value.astype(accumulation).mean(axis=axes, keepdims=True).astype(stash)
    values = (value - mean).astype(stash) if centered else value
    squared = (values * values).astype(stash)
    variance = squared.astype(accumulation).mean(axis=axes, keepdims=True).astype(stash)
    stddev = np.sqrt((variance + stash(np.float32(epsilon))).astype(stash)).astype(stash)
    inv = (stash(1) / stddev).astype(stash)
    normalized = ((values * inv) if centered else (values / stddev)).astype(x.dtype)
    output = normalized * scale
    if bias is not None:
        output = output + bias
    return output.astype(scale.dtype), mean, inv


@pytest.mark.parametrize("op", ["RMSNormalization", "LayerNormalization"])
@pytest.mark.parametrize("axis", [-3, -2, -1, 0, 1, 2])
@pytest.mark.parametrize("affine_shape", [(), (1,), (4,), (1, 1, 4)])
@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("eager", [False, True], ids=["compiled-enabled", "eager"])
def test_standard_norm_axes_broadcast(op, axis, affine_shape, dt, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    shape = [2, 3, 4]
    x = (randn(dt, *shape) * 1e-3).astype(dt)
    scale = np.full(affine_shape, 1.25, dt)
    inputs = [m.tensor("x", ir_of(dt), shape), m.tensor("scale", ir_of(dt), list(affine_shape))]
    feeds = {"x": x, "scale": scale}
    if op == "LayerNormalization":
        inputs.append(m.tensor("bias", ir_of(dt), list(affine_shape)))
        feeds["bias"] = np.full(affine_shape, 0.125, dt)
    model = build(
        op, inputs, [m.tensor("out", ir_of(dt), shape)],
        attrs=[ir.AttrInt64("axis", axis)],
    )
    session = _strict_mlx_session(model)
    for factor in [1, 2, -1]:
        feeds["x"] = x * dt(factor)
        actual = session.run(None, feeds)[0]
        expected, _, _ = _norm_reference(
            feeds["x"], scale, axis, op == "LayerNormalization", bias=feeds.get("bias")
        )
        np.testing.assert_allclose(actual, expected, **tol(dt))


@pytest.mark.parametrize("axes", [-1, 0])
@pytest.mark.parametrize("statistics", ["mean", "inv", "both"])
@pytest.mark.parametrize("stash,stash_numpy", [(DT.FLOAT, np.float32), (DT.BFLOAT16, ml_dtypes.bfloat16)])
def test_layer_norm_optional_statistics(axes, statistics, stash, stash_numpy):
    shape = [2, 3, 4]
    stats_shape = [2, 3, 1] if axes == -1 else [1, 1, 1]
    x = m.tensor("x", DT.FLOAT, shape)
    scale = m.tensor("scale", DT.FLOAT, [4])
    y = m.tensor("y", DT.FLOAT, shape)
    mean = m.tensor("mean", stash, stats_shape) if statistics != "inv" else ir.Value(name="")
    inv = m.tensor("inv", stash, stats_shape) if statistics != "mean" else ir.Value(name="")
    ports = [y, mean] if statistics == "mean" else [y, mean, inv]
    node = ir.node(
        "LayerNormalization", [x, scale], attributes={"axis": axes, "stash_type": int(stash)}, outputs=ports
    )
    nodes, outputs = [node], [y]
    for stat in ports[1:]:
        if not stat.name:
            continue
        out = m.tensor(f"{stat.name}_float", DT.FLOAT, stats_shape)
        nodes.append(ir.node("Cast", [stat], attributes={"to": int(DT.FLOAT)}, outputs=[out]))
        outputs.append(out)
    graph = ir.Graph(
        [x, scale], outputs, nodes=nodes, name="normalization_statistics", opset_imports={"": 23}
    )
    proto = ir.to_proto(ir.Model(graph, ir_version=11))
    if statistics == "inv":
        assert proto.graph.node[0].output[1] == ""
    model = proto.SerializeToString()
    data = randn(np.float32, *shape)
    weights = np.arange(1, 5, dtype=np.float32)
    actual = _strict_mlx_session(model).run(None, {"x": data, "scale": weights})
    expected = _norm_reference(data, weights, axes, True, stash=stash_numpy)
    selected = [expected[0]]
    if statistics != "inv":
        selected.append(expected[1].astype(np.float32))
    if statistics != "mean":
        selected.append(expected[2].astype(np.float32))
    for result, reference in zip(actual, selected, strict=True):
        np.testing.assert_allclose(result, reference, rtol=1e-2 if stash == DT.BFLOAT16 else 1e-5, atol=1e-5)


@pytest.mark.parametrize("x_dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("scale_dtype", [np.float16, np.float32, np.float64])
def test_rms_norm_mixed_types(x_dtype, scale_dtype):
    model = build(
        "RMSNormalization",
        [m.tensor("x", ir_of(x_dtype), [2, 4]), m.tensor("scale", ir_of(scale_dtype), [4])],
        [m.tensor("out", ir_of(scale_dtype), [2, 4])],
    )
    x = randn(x_dtype, 2, 4)
    scale = np.arange(1, 5, dtype=scale_dtype)
    actual = _strict_mlx_session(model).run(None, {"x": x, "scale": scale})[0]
    expected, _, _ = _norm_reference(x, scale, -1, False)
    assert actual.dtype == np.dtype(scale_dtype)
    np.testing.assert_allclose(actual, expected, rtol=2e-2 if np.float16 in [x_dtype, scale_dtype] else 1e-5, atol=1e-6)


@pytest.mark.parametrize("op", ["RMSNormalization", "LayerNormalization"])
@pytest.mark.parametrize("shape", [(0, 3), (2, 0), (0, 0)])
def test_standard_norm_empty(op, shape):
    model = build(
        op, [m.tensor("x", DT.FLOAT, list(shape)), m.tensor("scale", DT.FLOAT, [])],
        [m.tensor("out", DT.FLOAT, list(shape))],
    )
    actual = _strict_mlx_session(model).run(
        None, {"x": np.empty(shape, np.float32), "scale": np.array(1, np.float32)}
    )[0]
    assert actual.shape == shape


@pytest.mark.parametrize("op", ["RMSNormalization", "LayerNormalization"])
def test_standard_norm_dynamic_affine_dispatch(op):
    model = build(
        op, [m.tensor("x", DT.FLOAT, ["B", "D"]), m.tensor("scale", DT.FLOAT, ["G"])],
        [m.tensor("out", DT.FLOAT, ["B", "D"])],
    )
    session = _strict_mlx_session(model)
    for index, (batch, dim, gamma) in enumerate([(2, 1, 1), (2, 3, 1), (1, 3, 3), (2, 1, 1)]):
        x = randn(np.float32, batch, dim) * (index + 1)
        scale = np.arange(1, gamma + 1, dtype=np.float32) * (index + 1)
        actual = session.run(None, {"x": x, "scale": scale})[0]
        expected, _, _ = _norm_reference(x, scale, -1, op == "LayerNormalization")
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("op", ["RMSNormalization", "LayerNormalization"])
@pytest.mark.parametrize("shape", [(4,), (2, 1, 3, 4), (1, 2, 1, 2, 4)])
@pytest.mark.parametrize("axis", [0, -1])
def test_standard_norm_rank_boundaries(op, shape, axis):
    model = build(
        op, [m.tensor("x", DT.FLOAT, list(shape)), m.tensor("scale", DT.FLOAT, [])],
        [m.tensor("out", DT.FLOAT, list(shape))], attrs=[ir.AttrInt64("axis", axis)],
    )
    data, scale = randn(np.float32, *shape), np.array(1.25, np.float32)
    actual = _strict_mlx_session(model).run(None, {"x": data, "scale": scale})[0]
    expected, _, _ = _norm_reference(data, scale, axis, op == "LayerNormalization")
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dt", DTYPES)
def test_layer_norm_omitted_bias_slot(dt):
    x, scale = m.tensor("x", ir_of(dt), [2, 4]), m.tensor("scale", ir_of(dt), [4])
    y = m.tensor("y", ir_of(dt), [2, 4])
    node = ir.node("LayerNormalization", [x, scale, None], outputs=[y])
    graph = ir.Graph([x, scale], [y], nodes=[node], name="no_norm_bias", opset_imports={"": 23})
    proto = ir.to_proto(ir.Model(graph, ir_version=11))
    assert proto.graph.node[0].input[-1] == ""
    model = proto.SerializeToString()
    feeds = {"x": randn(dt, 2, 4), "scale": np.ones(4, dt)}
    actual = _strict_mlx_session(model).run(None, feeds)[0]
    # ORT 1.31 crashes initializing a CPU session with this explicit empty input.
    expected, _, _ = _norm_reference(feeds["x"], feeds["scale"], -1, True)
    np.testing.assert_allclose(actual, expected, **tol(dt))


@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
def test_layer_norm_empty_statistics(shape):
    stats_shape = [shape[0], 1]
    model = build(
        "LayerNormalization",
        [m.tensor("x", DT.FLOAT, list(shape)), m.tensor("scale", DT.FLOAT, [])],
        [m.tensor("y", DT.FLOAT, list(shape)), m.tensor("mean", DT.FLOAT, stats_shape),
         m.tensor("inv", DT.FLOAT, stats_shape)],
    )
    y, mean, inv = _strict_mlx_session(model).run(
        None, {"x": np.empty(shape, np.float32), "scale": np.array(1, np.float32)}
    )
    assert y.shape == shape
    expected = np.full(stats_shape, np.nan, np.float32)
    np.testing.assert_array_equal(mean, expected)
    np.testing.assert_array_equal(inv, expected)


@pytest.mark.parametrize("op", ["RMSNormalization", "LayerNormalization"])
@pytest.mark.parametrize("stash,np_stash", [(DT.FLOAT16, np.float16), (DT.DOUBLE, np.float64)])
def test_standard_norm_stash(op, stash, np_stash):
    model = build(
        op, [m.tensor("x", DT.FLOAT, [2, 4]), m.tensor("scale", DT.FLOAT, [])],
        [m.tensor("out", DT.FLOAT, [2, 4])],
        attrs=[ir.AttrInt64("stash_type", int(stash))],
    )
    x, scale = randn(np.float32, 2, 4), np.array(1.25, np.float32)
    actual = _strict_mlx_session(model).run(None, {"x": x, "scale": scale})[0]
    expected, _, _ = _norm_reference(x, scale, -1, op == "LayerNormalization", stash=np_stash)
    np.testing.assert_allclose(actual, expected, rtol=1e-3 if stash == DT.FLOAT16 else 1e-6, atol=1e-6)


@pytest.mark.parametrize("op", ["RMSNormalization", "LayerNormalization"])
@pytest.mark.parametrize("axis", [-1, 0])
@pytest.mark.parametrize("affine_shape", [(), (1,), (4,), (1, 1, 4)])
@pytest.mark.parametrize("stash", [DT.FLOAT, DT.BFLOAT16])
def test_standard_norm_bfloat16_interior(op, axis, affine_shape, stash):
    x = m.tensor("x", DT.FLOAT, [2, 3, 4])
    scale = m.tensor("scale", DT.FLOAT, list(affine_shape))
    xb = m.tensor("xb", DT.BFLOAT16, [2, 3, 4])
    sb = m.tensor("sb", DT.BFLOAT16, list(affine_shape))
    yb = m.tensor("yb", DT.BFLOAT16, [2, 3, 4])
    y = m.tensor("y", DT.FLOAT, [2, 3, 4])
    nodes = [
        ir.node("Cast", [x], attributes={"to": int(DT.BFLOAT16)}, outputs=[xb]),
        ir.node("Cast", [scale], attributes={"to": int(DT.BFLOAT16)}, outputs=[sb]),
        ir.node(op, [xb, sb], attributes={"axis": axis, "stash_type": int(stash)}, outputs=[yb]),
        ir.node("Cast", [yb], attributes={"to": int(DT.FLOAT)}, outputs=[y]),
    ]
    graph = ir.Graph(
        [x, scale], [y], nodes=nodes, name="bfloat_normalization", opset_imports={"": 23}
    )
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    data, weights = randn(np.float32, 2, 3, 4), np.full(affine_shape, 1.25, np.float32)
    actual = _strict_mlx_session(model).run(None, {"x": data, "scale": weights})[0]
    expected, _, _ = _norm_reference(
        data.astype(ml_dtypes.bfloat16), weights.astype(ml_dtypes.bfloat16), axis,
        op == "LayerNormalization", stash=np.float32 if stash == DT.FLOAT else ml_dtypes.bfloat16,
    )
    np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=2e-2, atol=1e-3)


@pytest.mark.parametrize("x_dtype", [np.float32, np.float64])
def test_rms_norm_double_affine_exact(x_dtype):
    model = build(
        "RMSNormalization",
        [m.tensor("x", ir_of(x_dtype), [2, 4]), m.tensor("scale", DT.DOUBLE, [4])],
        [m.tensor("out", DT.DOUBLE, [2, 4])],
        attrs=[ir.AttrFloat32("epsilon", 0)],
    )
    scale = np.ones(4, np.float64) + np.arange(1, 5) * 2**-30
    actual = _strict_mlx_session(model).run(None, {"x": np.ones((2, 4), x_dtype), "scale": scale})[0]
    np.testing.assert_array_equal(actual, np.broadcast_to(scale, (2, 4)))


@pytest.mark.parametrize("op", ["RMSNormalization", "LayerNormalization"])
def test_standard_norm_float64_statistics_exact(op):
    model = build(
        op, [m.tensor("x", DT.DOUBLE, [2]), m.tensor("scale", DT.DOUBLE, [2])],
        [m.tensor("out", DT.DOUBLE, [2])],
        attrs=[ir.AttrInt64("stash_type", int(DT.DOUBLE)), ir.AttrFloat32("epsilon", 0)],
    )
    x = np.array([1 + 2**-30, 1 - 2**-30], np.float64)
    actual = _strict_mlx_session(model).run(None, {"x": x, "scale": np.ones(2, np.float64)})[0]
    expected = x if op == "RMSNormalization" else np.array([1, -1], np.float64)
    np.testing.assert_array_equal(actual, expected)


def test_standard_norm_compile_and_cpu_routes(tmp_path):
    trace = tmp_path / "normalization.json"
    subprocess.run(
        [
            sys.executable, "-m", "pytest", Path(__file__).name,
            "-k", "test_standard_norm_dynamic_affine_dispatch or (test_standard_norm_stash and DOUBLE)",
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
    events = json.loads(trace.read_text())
    paths = [e["args"] for e in events if e.get("cat") == "ep.path"]
    assert [e["path"] for e in paths] == ["general"] * 8 + ["eager"] * 2
    assert [e["cache"] for e in paths] == ["MISS", "RETRACE", "RETRACE", "HIT"] * 2 + ["n/a"] * 2
    assert {e["ops"] for e in paths[-2:]} == {"RMSNormalization", "LayerNormalization"}


# --- LayerNormalization -----------------------------------------------------------------------
@pytest.mark.parametrize("dt", DTYPES)
def test_layer_norm_scale_bias(dt):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 4, 8)).astype(dt)
    scale = rng.standard_normal((8,)).astype(dt)
    bias = rng.standard_normal((8,)).astype(dt)
    model = build(
        "LayerNormalization",
        [m.tensor("x", ir_of(dt), [2, 4, 8]), m.tensor("s", ir_of(dt), [8]),
         m.tensor("b", ir_of(dt), [8])],
        [m.tensor("o", ir_of(dt), [2, 4, 8])],
        attrs=[ir.AttrInt64("axis", -1), ir.AttrFloat32("epsilon", 1e-5)],
        opset=17,
    )
    m.assert_matches_cpu(model, {"x": x, "s": scale, "b": bias}, **tol(dt))


@pytest.mark.parametrize("dt", DTYPES)
def test_layer_norm_no_bias(dt):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 6)).astype(dt)
    scale = rng.standard_normal((6,)).astype(dt)
    model = build(
        "LayerNormalization",
        [m.tensor("x", ir_of(dt), [3, 6]), m.tensor("s", ir_of(dt), [6])],
        [m.tensor("o", ir_of(dt), [3, 6])],
        attrs=[ir.AttrFloat32("epsilon", 1e-5)],
        opset=17,
    )
    m.assert_matches_cpu(model, {"x": x, "s": scale}, **tol(dt))


# --- SimplifiedLayerNormalization (com.microsoft) ---------------------------------------------
def test_simplified_layer_norm():
    """RMS-style norm. Skipped when the running ORT build lacks the contrib schema."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    scale = rng.standard_normal((8,)).astype(np.float32)
    model = build(
        "SimplifiedLayerNormalization",
        [m.tensor("x", DT.FLOAT, [2, 4, 8]), m.tensor("s", DT.FLOAT, [8])],
        [m.tensor("o", DT.FLOAT, [2, 4, 8])],
        attrs=[ir.AttrFloat32("epsilon", 1e-5)],
        domain="com.microsoft",
    )
    try:
        ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001 - schema availability probe
        if "not a registered" in str(exc):
            pytest.skip("SimplifiedLayerNormalization not in this ORT build")
        raise
    m.assert_matches_cpu(model, {"x": x, "s": scale}, rtol=1e-4, atol=1e-5)


def test_simplified_layer_norm_default_domain():
    """SPECIAL CASE: Microsoft's exporter sometimes stamps the com.microsoft contrib op
    SimplifiedLayerNormalization into the DEFAULT domain (domain ""). We register it there too, so
    it must be CLAIMED (not fall back) and match CPU. (gemma-4-E2B vision encoder has 113 of these.)"""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    scale = rng.standard_normal((8,)).astype(np.float32)
    model = build(
        "SimplifiedLayerNormalization",
        [m.tensor("x", DT.FLOAT, [2, 4, 8]), m.tensor("s", DT.FLOAT, [8])],
        [m.tensor("o", DT.FLOAT, [2, 4, 8])],
        attrs=[ir.AttrFloat32("epsilon", 1e-5), ir.AttrInt64("axis", -1)],
        domain="",  # the mis-stamped default domain
    )
    try:
        ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001 - schema availability probe
        if "not a registered" in str(exc):
            pytest.skip("SimplifiedLayerNormalization not in this ORT build")
        raise
    m.assert_op_claimed(
        model,
        {"x": x, "s": scale},
        "SimplifiedLayerNormalization",
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("dt", DTYPES)
def test_skip_layer_norm(dt):
    rng = np.random.default_rng(4)
    shape = [2, 4, 8]
    x = rng.standard_normal(shape).astype(dt)
    skip = rng.standard_normal(shape).astype(dt)
    gamma = rng.standard_normal((8,)).astype(dt)
    beta = rng.standard_normal((8,)).astype(dt)
    bias = rng.standard_normal((8,)).astype(dt)
    model = build(
        "SkipLayerNormalization",
        [m.tensor("x", ir_of(dt), shape), m.tensor("skip", ir_of(dt), shape),
         m.tensor("gamma", ir_of(dt), [8]), m.tensor("beta", ir_of(dt), [8]),
         m.tensor("bias", ir_of(dt), [8])],
        [m.tensor("o", ir_of(dt), shape)],
        attrs=[ir.AttrFloat32("epsilon", 1e-5)],
        domain="com.microsoft",
    )
    feeds = {"x": x, "skip": skip, "gamma": gamma, "beta": beta, "bias": bias}
    m.assert_matches_cpu(model, feeds, **tol(dt))


# --- GroupNormalization -----------------------------------------------------------------------
@pytest.mark.parametrize("dt", DTYPES)
def test_group_norm(dt):
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 6, 4, 4)).astype(dt)
    scale = rng.standard_normal((6,)).astype(dt)
    bias = rng.standard_normal((6,)).astype(dt)
    model = build(
        "GroupNormalization",
        [m.tensor("x", ir_of(dt), [2, 6, 4, 4]), m.tensor("s", ir_of(dt), [6]),
         m.tensor("b", ir_of(dt), [6])],
        [m.tensor("o", ir_of(dt), [2, 6, 4, 4])],
        attrs=[ir.AttrInt64("num_groups", 3), ir.AttrFloat32("epsilon", 1e-5)],
    )
    m.assert_matches_cpu(model, {"x": x, "s": scale, "b": bias}, **tol(dt))


def test_group_norm_dynamic_spatial():
    """GroupNormalization with dynamic batch/spatial dims (the standard diffusion-UNet export
    form, e.g. [B, C, H, W] with B/H/W symbolic). Only the channel count needs to be static to
    split into groups; the reshape is built from the concrete trace-time shape (like Conv). Must
    be CLAIMED by MLX (not forced to CPU, which fragmented every UNet resnet/attention block) and
    match CPU numerically."""
    import contextlib
    import os
    import tempfile

    @contextlib.contextmanager
    def _capture_stderr():
        with tempfile.TemporaryFile(mode="w+b") as tmp:
            old = os.dup(2)
            os.dup2(tmp.fileno(), 2)
            try:
                yield tmp
            finally:
                os.dup2(old, 2)
                os.close(old)

    rng = np.random.default_rng(15)
    x = rng.standard_normal((2, 6, 5, 7)).astype(np.float32)
    scale = rng.standard_normal((6,)).astype(np.float32)
    bias = rng.standard_normal((6,)).astype(np.float32)
    model = build(
        "GroupNormalization",
        [
            m.tensor("x", DT.FLOAT, ["B", 6, "H", "W"]),
            m.tensor("s", DT.FLOAT, [6]),
            m.tensor("b", DT.FLOAT, [6]),
        ],
        [m.tensor("o", DT.FLOAT, ["B", 6, "H", "W"])],
        attrs=[ir.AttrInt64("num_groups", 3), ir.AttrFloat32("epsilon", 1e-5)],
    )
    feeds = {"x": x, "s": scale, "b": bias}

    prev = os.environ.get("ONNXRUNTIME_EP_MLX_CLAIM_DEBUG")
    os.environ["ONNXRUNTIME_EP_MLX_CLAIM_DEBUG"] = "1"
    try:
        with _capture_stderr() as cap:
            m.run_mlx(model, feeds)
            cap.seek(0)
            claim_log = cap.read().decode("utf-8", "replace")
    finally:
        if prev is None:
            os.environ.pop("ONNXRUNTIME_EP_MLX_CLAIM_DEBUG", None)
        else:
            os.environ["ONNXRUNTIME_EP_MLX_CLAIM_DEBUG"] = prev

    assert "unclaimed GroupNormalization" not in claim_log, (
        f"dynamic-spatial GroupNormalization must be claimed by MLX, got:\n{claim_log}"
    )
    m.assert_matches_cpu(model, feeds, rtol=1e-4, atol=1e-5)


# --- LpNormalization --------------------------------------------------------------------------
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("axis", [-1, 0])
def test_lp_norm(p, axis):
    rng = np.random.default_rng(6)
    x = (rng.standard_normal((3, 5)) + 2.0).astype(np.float32)
    model = build(
        "LpNormalization",
        [m.tensor("x", DT.FLOAT, [3, 5])],
        [m.tensor("o", DT.FLOAT, [3, 5])],
        attrs=[ir.AttrInt64("axis", axis), ir.AttrInt64("p", p)],
    )
    m.assert_matches_cpu(model, {"x": x}, rtol=1e-5, atol=1e-6)


# --- BatchNormalization -----------------------------------------------------------------------
@pytest.mark.parametrize("dt", DTYPES)
def test_batch_norm_inference(dt):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, 4, 3, 3)).astype(dt)
    scale = rng.standard_normal((4,)).astype(dt)
    bias = rng.standard_normal((4,)).astype(dt)
    mean = rng.standard_normal((4,)).astype(dt)
    var = (np.abs(rng.standard_normal((4,))) + 0.5).astype(dt)
    model = build(
        "BatchNormalization",
        [m.tensor("x", ir_of(dt), [2, 4, 3, 3]), m.tensor("s", ir_of(dt), [4]),
         m.tensor("b", ir_of(dt), [4]), m.tensor("mean", ir_of(dt), [4]),
         m.tensor("var", ir_of(dt), [4])],
        [m.tensor("o", ir_of(dt), [2, 4, 3, 3])],
        attrs=[ir.AttrFloat32("epsilon", 1e-5)],
    )
    feeds = {"x": x, "s": scale, "b": bias, "mean": mean, "var": var}
    m.assert_matches_cpu(model, feeds, **tol(dt))


def test_skip_simplified_layer_norm_unused_mean_invstd():
    """SkipSimplifiedLayerNormalization declaring mean(out1)/inv-std(out2) that are UNUSED (no
    consumer) — the common transformers.js/Mobius export shape. The RMS handler doesn't produce
    them; since they aren't fused-subgraph boundary outputs, MLX must DCE them, claim the node, and
    still match CPU on the consumed outputs (normalized out0 + residual sum out3)."""
    rng = np.random.default_rng(11)
    shape = [1, 5, 8]
    x = rng.standard_normal(shape).astype(np.float32)
    skip = rng.standard_normal(shape).astype(np.float32)
    gamma = rng.standard_normal((8,)).astype(np.float32)
    xi = m.tensor("x", DT.FLOAT, shape)
    si = m.tensor("skip", DT.FLOAT, shape)
    gi = m.tensor("gamma", DT.FLOAT, [8])
    o = m.tensor("o", DT.FLOAT, shape)
    mean = ir.Value(name="mean_unused", type=None)  # dangling: no consumer, not a graph output
    invstd = ir.Value(name="invstd_unused", type=None)
    resid = m.tensor("resid", DT.FLOAT, shape)
    node = ir.node(
        "SkipSimplifiedLayerNormalization",
        [xi, si, gi],
        attributes={"epsilon": 1e-5},
        domain="com.microsoft",
        outputs=[o, mean, invstd, resid],
    )
    graph = ir.Graph([xi, si, gi], [o, resid], nodes=[node],
                     opset_imports={"": 24, "com.microsoft": 1}, name="mlx_SkipSimplifiedLN_unused")
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    try:
        ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        if "not a registered" in str(exc):
            pytest.skip("SkipSimplifiedLayerNormalization not in this ORT build")
        raise
    m.assert_matches_cpu(model, {"x": x, "skip": skip, "gamma": gamma}, rtol=1e-4, atol=1e-5)
