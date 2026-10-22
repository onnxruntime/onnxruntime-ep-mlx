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
import onnx
import onnx_ir as ir
import onnxruntime as ort
from onnx.reference import ReferenceEvaluator
import pytest

import _models as m

DT = ir.DataType

_IR_OF = {np.dtype("float32"): DT.FLOAT, np.dtype("float16"): DT.FLOAT16, np.dtype("float64"): DT.DOUBLE}
_TOL = {
    np.dtype("float32"): dict(rtol=1e-4, atol=1e-5),
    np.dtype("float16"): dict(rtol=2e-2, atol=2e-2),
}
DTYPES = [np.float32, np.float16]
STANDARD_NORMS = ["RMSNormalization", "LayerNormalization", "SimplifiedLayerNormalization"]


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


def _norm_reference(x, scale, axis, centered, *, stash=np.float32, bias=None, epsilon=1e-5,
                    divide_stddev=False, moments_variance=False):
    axes = tuple(range(axis % x.ndim, x.ndim))
    accumulation = np.float64 if np.dtype(stash) == np.dtype(np.float64) else np.float32
    value = x.astype(stash)
    mean = value.astype(accumulation).mean(axis=axes, keepdims=True).astype(stash)
    values = (value - mean).astype(stash) if centered else value
    squared = (value * value if moments_variance else values * values).astype(stash)
    variance = squared.astype(accumulation).mean(axis=axes, keepdims=True).astype(stash)
    if moments_variance:
        variance = (variance - (mean * mean).astype(stash)).astype(stash)
    stddev = np.sqrt((variance + stash(np.float32(epsilon))).astype(stash)).astype(stash)
    inv = (stash(1) / stddev).astype(stash)
    normalized = ((values * inv) if centered and not divide_stddev else (values / stddev)).astype(x.dtype)
    output = normalized * scale
    if bias is not None:
        output = output + bias
    return output.astype(scale.dtype), mean, inv


@pytest.mark.parametrize("op", STANDARD_NORMS)
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
@pytest.mark.parametrize("op", ["RMSNormalization", "SimplifiedLayerNormalization"])
def test_rms_norm_mixed_types(x_dtype, scale_dtype, op):
    model = build(
        op,
        [m.tensor("x", ir_of(x_dtype), [2, 4]), m.tensor("scale", ir_of(scale_dtype), [4])],
        [m.tensor("out", ir_of(scale_dtype), [2, 4])],
    )
    x = randn(x_dtype, 2, 4)
    scale = np.arange(1, 5, dtype=scale_dtype)
    actual = _strict_mlx_session(model).run(None, {"x": x, "scale": scale})[0]
    expected, _, _ = _norm_reference(x, scale, -1, False)
    assert actual.dtype == np.dtype(scale_dtype)
    np.testing.assert_allclose(actual, expected, rtol=2e-2 if np.float16 in [x_dtype, scale_dtype] else 1e-5, atol=1e-6)


@pytest.mark.parametrize("op", STANDARD_NORMS)
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


@pytest.mark.parametrize("op", STANDARD_NORMS)
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


@pytest.mark.parametrize("op", STANDARD_NORMS)
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


@pytest.mark.parametrize("op", STANDARD_NORMS)
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


@pytest.mark.parametrize("op", STANDARD_NORMS)
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
@pytest.mark.parametrize("op", ["RMSNormalization", "SimplifiedLayerNormalization"])
def test_rms_norm_double_affine_exact(x_dtype, op):
    model = build(
        op,
        [m.tensor("x", ir_of(x_dtype), [2, 4]), m.tensor("scale", DT.DOUBLE, [4])],
        [m.tensor("out", DT.DOUBLE, [2, 4])],
        attrs=[ir.AttrFloat32("epsilon", 0)],
    )
    scale = np.ones(4, np.float64) + np.arange(1, 5) * 2**-30
    actual = _strict_mlx_session(model).run(None, {"x": np.ones((2, 4), x_dtype), "scale": scale})[0]
    np.testing.assert_array_equal(actual, np.broadcast_to(scale, (2, 4)))


@pytest.mark.parametrize("op", STANDARD_NORMS)
def test_standard_norm_float64_statistics_exact(op):
    model = build(
        op, [m.tensor("x", DT.DOUBLE, [2]), m.tensor("scale", DT.DOUBLE, [2])],
        [m.tensor("out", DT.DOUBLE, [2])],
        attrs=[ir.AttrInt64("stash_type", int(DT.DOUBLE)), ir.AttrFloat32("epsilon", 0)],
    )
    x = np.array([1 + 2**-30, 1 - 2**-30], np.float64)
    actual = _strict_mlx_session(model).run(None, {"x": x, "scale": np.ones(2, np.float64)})[0]
    expected = np.array([1, -1], np.float64) if op == "LayerNormalization" else x
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
    assert [e["path"] for e in paths] == ["general"] * 12 + ["eager"] * 3
    assert [e["cache"] for e in paths] == ["MISS", "RETRACE", "RETRACE", "HIT"] * 3 + ["n/a"] * 3
    assert {e["ops"] for e in paths[-3:]} == set(STANDARD_NORMS)


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
    """ORT exposes this extension in the default domain; it must be claimed."""
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
@pytest.mark.parametrize("axis", [-1, 0])
@pytest.mark.parametrize("stash,precision", [(DT.FLOAT, np.float32), (DT.DOUBLE, np.float64)])
@pytest.mark.parametrize("eager", [False, True])
def test_simplified_layer_norm_inverse_statistics(dt, axis, stash, precision, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    shape = [2, 3, 4]
    stats_shape = [2, 3, 1] if axis == -1 else [1, 1, 1]
    model = build(
        "SimplifiedLayerNormalization",
        [m.tensor("x", ir_of(dt), shape), m.tensor("s", ir_of(dt), [4])],
        [m.tensor("y", ir_of(dt), shape), m.tensor("inv", stash, stats_shape)],
        attrs=[ir.AttrInt64("axis", axis), ir.AttrInt64("stash_type", int(stash))],
    )
    session = _strict_mlx_session(model)
    for factor in [1, 2, -1]:
        feeds = {"x": randn(dt, *shape) * dt(factor), "s": np.full(4, 1.25, dt)}
        actual, inv = session.run(None, feeds)
        expected, _, expected_inv = _norm_reference(feeds["x"], feeds["s"], axis, False, stash=precision)
        np.testing.assert_allclose(actual, expected, **tol(dt))
        assert inv.dtype == np.dtype(precision)
        np.testing.assert_allclose(inv, expected_inv, rtol=1e-12 if stash == DT.DOUBLE else 1e-6, atol=1e-12)
        if dt == np.float32 and stash == DT.FLOAT:
            cpu = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
            cpu_out = cpu.run(None, feeds)
            np.testing.assert_allclose(actual, cpu_out[0], **tol(dt))
            np.testing.assert_allclose(inv, cpu_out[1], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
def test_simplified_layer_norm_empty_inverse_statistics(shape):
    model = build(
        "SimplifiedLayerNormalization",
        [m.tensor("x", DT.FLOAT, list(shape)), m.tensor("s", DT.FLOAT, [])],
        [m.tensor("y", DT.FLOAT, list(shape)), m.tensor("inv", DT.FLOAT, [shape[0], 1])],
    )
    y, inv = _strict_mlx_session(model).run(
        None, {"x": np.empty(shape, np.float32), "s": np.array(1, np.float32)}
    )
    assert y.shape == shape
    np.testing.assert_array_equal(inv, np.full([shape[0], 1], np.nan, np.float32))


def test_simplified_layer_norm_omitted_inverse_slot():
    x, scale = m.tensor("x", DT.FLOAT, [2, 4]), m.tensor("scale", DT.FLOAT, [])
    y, omitted = m.tensor("y", DT.FLOAT, [2, 4]), ir.Value(name="")
    node = ir.node("SimplifiedLayerNormalization", [x, scale], outputs=[y, omitted])
    graph = ir.Graph([x, scale], [y], nodes=[node], name="omitted_inverse", opset_imports={"": 23})
    proto = ir.to_proto(ir.Model(graph, ir_version=11))
    proto.graph.node[0].output.append("")
    assert proto.graph.node[0].output[1] == ""
    feeds = {"x": randn(np.float32, 2, 4), "scale": np.array(1.25, np.float32)}
    actual = _strict_mlx_session(proto.SerializeToString()).run(None, feeds)[0]
    expected, _, _ = _norm_reference(feeds["x"], feeds["scale"], -1, False)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


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
def _group_norm_model(shape, version, groups, dt, stash):
    affine = groups if version < 21 else shape[1]
    payload_type = DT.BFLOAT16 if dt == ml_dtypes.bfloat16 else ir_of(dt)
    boundary_type = DT.FLOAT if payload_type == DT.BFLOAT16 else payload_type
    inputs = [m.tensor("x", boundary_type, shape), m.tensor("s", boundary_type, [affine]),
              m.tensor("b", boundary_type, [affine])]
    nodes, arguments = [], []
    for value in inputs:
        if boundary_type == payload_type:
            arguments.append(value)
        else:
            converted = m.tensor(f"{value.name}_converted", payload_type, list(value.shape))
            nodes.append(ir.node("Cast", [value], attributes={"to": int(payload_type)}, outputs=[converted]))
            arguments.append(converted)
    output = m.tensor("normalized", payload_type, shape)
    attrs = {"num_groups": groups}
    if version >= 21:
        attrs["stash_type"] = int(stash)
    nodes.append(ir.node("GroupNormalization", arguments, attributes=attrs, outputs=[output]))
    if boundary_type != payload_type:
        converted = m.tensor("out", boundary_type, shape)
        nodes.append(ir.node("Cast", [output], attributes={"to": int(boundary_type)}, outputs=[converted]))
        output = converted
    graph = ir.Graph(inputs, [output], nodes=nodes, name="group_normalization", opset_imports={"": version})
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


@pytest.mark.parametrize("version", [18, 21])
@pytest.mark.parametrize("shape,groups", [([2, 6], 1), ([2, 6, 3], 2), ([1, 6, 2, 3, 2], 6)])
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.float64, ml_dtypes.bfloat16])
@pytest.mark.parametrize("eager", [False, True])
def test_group_norm_schema_forms(version, shape, groups, dt, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    stash = DT.DOUBLE if dt == np.float64 else DT.FLOAT
    model = _group_norm_model(shape, version, groups, dt, stash)
    session = _strict_mlx_session(model)
    rng = np.random.default_rng(121)
    affine_size = groups if version < 21 else shape[1]
    for _ in range(3):
        x = rng.standard_normal(shape).astype(dt)
        scale = rng.standard_normal(affine_size).astype(dt)
        bias = rng.standard_normal(affine_size).astype(dt)
        boundary = np.float32 if dt == ml_dtypes.bfloat16 else dt
        actual = session.run(None, {"x": x.astype(boundary), "s": scale.astype(boundary),
                                   "b": bias.astype(boundary)})[0]
        grouped_shape = [shape[0], groups, int(np.prod(shape[1:])) // groups]
        precision = dt if version < 21 else np.float64 if stash == DT.DOUBLE else np.float32
        normalized, _, _ = _norm_reference(
            x.reshape(grouped_shape), np.ones((), dt), 2, True, stash=precision,
            divide_stddev=True, moments_variance=True,
        )
        if version < 21:
            normalized = normalized * scale.reshape(1, groups, 1) + bias.reshape(1, groups, 1)
            expected = normalized.reshape(shape)
        else:
            affine_shape = [1, shape[1]] + [1] * (len(shape) - 2)
            expected = normalized.reshape(shape) * scale.reshape(affine_shape) + bias.reshape(affine_shape)
        tolerance = dict(rtol=2e-2, atol=2e-2) if dt in [np.float16, ml_dtypes.bfloat16] else dict(rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(actual, expected.astype(boundary), **tolerance)
        if dt == np.float32:
            cpu = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
            np.testing.assert_allclose(
                actual, cpu.run(None, {"x": x, "s": scale, "b": bias})[0], **tolerance
            )


@pytest.mark.parametrize("stash,precision", [(DT.FLOAT16, np.float16), (DT.BFLOAT16, ml_dtypes.bfloat16),
                                           (DT.DOUBLE, np.float64)])
def test_group_norm_stash_precision(stash, precision):
    shape, groups = [2, 4, 3], 2
    model = _group_norm_model(shape, 21, groups, np.float32, stash)
    feeds = {"x": randn(np.float32, *shape), "s": np.ones(4, np.float32), "b": np.zeros(4, np.float32)}
    actual = _strict_mlx_session(model).run(None, feeds)[0]
    expected, _, _ = _norm_reference(
        feeds["x"].reshape(2, groups, 6), np.ones((), np.float32), 2, True, stash=precision,
        divide_stddev=True, moments_variance=True,
    )
    np.testing.assert_allclose(actual, expected.reshape(shape), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("version", [18, 21])
def test_group_norm_dynamic_channels_and_empty(version):
    model = _group_norm_model(["N", "C", "S"], version, 2, np.float32, DT.FLOAT)
    session = _strict_mlx_session(model)
    for shape in ([2, 6, 3], [1, 8, 2], [0, 4, 3], [2, 6, 0], [2, 0, 3], [2, 6, 3]):
        affine_size = 2 if version < 21 else shape[1]
        feeds = {"x": randn(np.float32, *shape), "s": np.ones(affine_size, np.float32),
                 "b": np.zeros(affine_size, np.float32)}
        actual = session.run(None, feeds)[0]
        assert actual.shape == tuple(shape)
        if np.prod(shape):
            expected, _, _ = _norm_reference(
                feeds["x"].reshape(shape[0], 2, int(np.prod(shape[1:])) // 2),
                np.ones((), np.float32), 2, True, divide_stddev=True, moments_variance=True,
            )
            np.testing.assert_allclose(actual, expected.reshape(shape), rtol=1e-5, atol=1e-6)


def test_group_norm_double_statistics_exact():
    model = build(
        "GroupNormalization",
        [m.tensor("x", DT.DOUBLE, [1, 2]), m.tensor("s", DT.DOUBLE, [2]),
         m.tensor("b", DT.DOUBLE, [2])],
        [m.tensor("out", DT.DOUBLE, [1, 2])],
        attrs=[ir.AttrInt64("num_groups", 1), ir.AttrInt64("stash_type", int(DT.DOUBLE)),
               ir.AttrFloat32("epsilon", 0)],
    )
    feeds = {"x": np.array([[1 + 2**-20, 1 - 2**-20]], np.float64),
             "s": np.ones(2, np.float64), "b": np.zeros(2, np.float64)}
    np.testing.assert_array_equal(_strict_mlx_session(model).run(None, feeds)[0], [[1, -1]])


@pytest.mark.parametrize("version,stash,x", [(18, DT.FLOAT, [4096, 4097]), (21, DT.FLOAT16, [32, 33])])
def test_group_norm_schema_function_variance(version, stash, x):
    shape = [1, 2]
    feeds = {"x": np.array([x], np.float32), "s": np.ones(2 if version == 21 else 1, np.float32),
             "b": np.zeros(2 if version == 21 else 1, np.float32)}
    model = _group_norm_model(shape, version, 1, np.float32, stash)
    node = onnx.load_model_from_string(model).graph.node[0]
    schema = onnx.defs.get_schema("GroupNormalization", version)
    types = [
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, list(value.shape)).SerializeToString()
        for value in feeds.values()
    ]
    function = onnx.FunctionProto.FromString(
        schema.get_context_dependent_function(node.SerializeToString(), types)
    )
    reference = ReferenceEvaluator(function)
    expected = reference.run(
        None, dict(zip(function.input, feeds.values())),
        attributes={attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute},
    )[0]
    actual = _strict_mlx_session(model).run(None, feeds)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("malformed", ["channels", "scale", "bias"])
def test_group_norm_invalid_dynamic_geometry_recovers(malformed):
    session = _strict_mlx_session(
        _group_norm_model(["N", "C", "S"], 21, 2, np.float32, DT.FLOAT)
    )
    feeds = {"x": randn(np.float32, 2, 6, 3), "s": np.ones(6, np.float32),
             "b": np.zeros(6, np.float32)}
    expected = session.run(None, feeds)[0]
    bad = feeds.copy()
    if malformed == "channels":
        bad["x"] = randn(np.float32, 2, 7, 3)
        bad["s"], bad["b"] = np.ones(7, np.float32), np.zeros(7, np.float32)
    else:
        bad["s" if malformed == "scale" else "b"] = np.ones(5, np.float32)
    with pytest.raises(Exception, match="GroupNormalization.*(divide channel count|affine parameters)"):
        session.run(None, bad)
    np.testing.assert_array_equal(session.run(None, feeds)[0], expected)


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
