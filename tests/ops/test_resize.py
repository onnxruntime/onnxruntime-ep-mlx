"""Correctness tests for the MLX EP Resize op (nearest + linear).

Each case runs a single-node Resize model through the MLX EP (with ORT CPU fallback available) and
compares against ORT's CPU EP, tolerance-gated. `scales` / `sizes` are supplied as constant
initializers — the only forms the EP claims — mirroring a constant-folded real model. bf16 keeps the
compute inside an MLX-claimed subgraph (fp32 boundaries) and compares against a numpy reference,
since ORT CPU has no bf16 Resize kernel.

Covered claimed matrix: {nearest, linear} x {half_pixel, asymmetric, align_corners,
pytorch_half_pixel} x {scale-up, scale-down, sizes} x {fp32, fp16, bf16}, plus every nearest_mode
and a 1-D (NCW) linear case. Cubic / roi / exclude_outside / antialias / dynamic params are left to
ORT CPU and are not exercised here. Same-extent fractional scales also cover Upsample-9,
independent coordinate/index goldens, identity controls, and default/eager value/shape replay.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

import _models as m

DT = ir.DataType

_IR_OF = {
    np.dtype("float32"): DT.FLOAT,
    np.dtype("float16"): DT.FLOAT16,
}

_TOL = {
    np.dtype("float32"): dict(rtol=1e-6, atol=2e-3),
    np.dtype("float16"): dict(rtol=2e-3, atol=2e-3),
}


def initz(name: str, arr: np.ndarray) -> ir.Value:
    """A constant initializer value (const_value set) — read by the EP at translate time."""
    t = ir.tensor(arr, name=name)
    return ir.Value(
        name=name, type=ir.TensorType(t.dtype), shape=ir.Shape(list(arr.shape)), const_value=t
    )


def build_resize(
    *,
    dt,
    in_shape: list[int],
    out_shape: list[int],
    scales: np.ndarray | None = None,
    sizes: np.ndarray | None = None,
    mode: str,
    ctm: str,
    nmode: str | None = None,
) -> bytes:
    """Single-node Resize model; scales/sizes are graph initializers (the claimed constant form)."""
    ir_dt = _IR_OF[np.dtype(dt)]
    x = m.tensor("x", ir_dt, in_shape)
    inits: list[ir.Value] = []
    # roi is omitted (empty-name optional slot) so `scales`/`sizes` land at input index 2 / 3.
    inputs: list[ir.Value] = [x, initz("", np.array([], np.float32))]
    if scales is not None:
        inputs.append(initz("scales", scales))
        inits.append(initz("scales", scales))
    else:
        inputs.append(initz("", np.array([], np.float32)))  # scales omitted
        inputs.append(initz("sizes", sizes))
        inits.append(initz("sizes", sizes))
    out = m.tensor("o", ir_dt, out_shape)
    attrs = {"mode": mode, "coordinate_transformation_mode": ctm}
    if nmode is not None:
        attrs["nearest_mode"] = nmode
    node = ir.node("Resize", inputs, attributes=attrs, outputs=[out])
    graph = ir.Graph(
        [x], [out], nodes=[node], initializers=inits, opset_imports={"": 19}, name="mlx_Resize"
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


# --- numpy reference (fp32) used for bf16 coverage ------------------------------------------------
def _coord(mode: str, j: int, scale: float, li: int, lo: int) -> float:
    if mode == "align_corners":
        return 0.0 if lo == 1 else j * (li - 1) / (lo - 1)
    if mode == "asymmetric":
        return j / scale
    if mode == "pytorch_half_pixel":
        return (j + 0.5) / scale - 0.5 if lo > 1 else 0.0
    return (j + 0.5) / scale - 0.5  # half_pixel


def _nearest_index(nmode: str, src: float, li: int) -> int:
    if nmode == "floor":
        v = np.floor(src)
    elif nmode == "ceil":
        v = np.ceil(src)
    elif nmode == "round_prefer_ceil":
        v = np.floor(src + 0.5)
    else:
        v = np.ceil(src - 0.5)  # round_prefer_floor
    return int(min(max(int(v), 0), li - 1))


def resize_ref(
    data: np.ndarray,
    out_len: list[int],
    scales: list[float],
    *,
    mode: str,
    ctm: str,
    nmode: str = "round_prefer_floor",
) -> np.ndarray:
    d = data.astype(np.float64)
    rank = data.ndim
    for ax in range(rank):
        li, lo = data.shape[ax], out_len[ax]
        coords = [_coord(ctm, j, scales[ax], li, lo) for j in range(lo)]
        # Check the actual clamped geometry, independently of the handler's scale predicate.
        if lo == li and all(min(max(src, 0), li - 1) == j for j, src in enumerate(coords)):
            continue
        if mode == "nearest":
            idx = [_nearest_index(nmode, src, li) for src in coords]
            d = np.take(d, idx, axis=ax)
        else:
            new = np.zeros(d.shape[:ax] + (lo,) + d.shape[ax + 1 :], np.float64)
            for j, src in enumerate(coords):
                x0 = int(np.floor(src))
                frac = src - x0
                lo_i = min(max(x0, 0), li - 1)
                hi_i = min(max(x0 + 1, 0), li - 1)
                sj = [slice(None)] * rank
                sj[ax] = j
                slo = [slice(None)] * rank
                slo[ax] = lo_i
                shi = [slice(None)] * rank
                shi[ax] = hi_i
                new[tuple(sj)] = (1 - frac) * d[tuple(slo)] + frac * d[tuple(shi)]
            d = new
    return d.astype(np.float32)


def _sample(shape: list[int]) -> np.ndarray:
    rng = np.random.default_rng(hash(tuple(shape)) & 0xFFFFFFFF)
    return rng.standard_normal(shape).astype(np.float32)


CTMS = ["half_pixel", "asymmetric", "align_corners", "pytorch_half_pixel"]

# (name, in_shape, kind, param, out_shape, scales_for_ref)
SPATIAL_CASES = [
    ("up2x", [1, 2, 4, 4], "scales", [1.0, 1.0, 2.0, 2.0], [1, 2, 8, 8], [1, 1, 2.0, 2.0]),
    ("down2x", [1, 2, 4, 6], "scales", [1.0, 1.0, 0.5, 0.5], [1, 2, 2, 3], [1, 1, 0.5, 0.5]),
    ("sizes", [1, 2, 4, 4], "sizes", [1, 2, 7, 3], [1, 2, 7, 3], [1, 1, 7 / 4, 3 / 4]),
]


# --- nearest ------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", [np.float32, np.float16])
@pytest.mark.parametrize("ctm", CTMS)
@pytest.mark.parametrize("case", SPATIAL_CASES, ids=[c[0] for c in SPATIAL_CASES])
def test_nearest_matches_cpu(dt, ctm, case):
    _name, in_shape, kind, param, out_shape, _sc = case
    data = _sample(in_shape).astype(dt)
    kw = (
        {"scales": np.array(param, np.float32)}
        if kind == "scales"
        else {"sizes": np.array(param, np.int64)}
    )
    model = build_resize(
        dt=dt,
        in_shape=in_shape,
        out_shape=out_shape,
        mode="nearest",
        ctm=ctm,
        nmode="round_prefer_floor",
        **kw,
    )
    m.assert_matches_cpu(model, {"x": data}, **_TOL[np.dtype(dt)])


@pytest.mark.parametrize("nmode", ["round_prefer_floor", "round_prefer_ceil", "floor", "ceil"])
def test_nearest_modes_match_cpu(nmode):
    in_shape, out_shape = [1, 1, 5, 5], [1, 1, 3, 3]
    data = _sample(in_shape)
    model = build_resize(
        dt=np.float32,
        in_shape=in_shape,
        out_shape=out_shape,
        sizes=np.array(out_shape, np.int64),
        mode="nearest",
        ctm="half_pixel",
        nmode=nmode,
    )
    m.assert_matches_cpu(model, {"x": data}, **_TOL[np.dtype(np.float32)])


# --- linear -------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", [np.float32, np.float16])
@pytest.mark.parametrize("ctm", CTMS)
@pytest.mark.parametrize("case", SPATIAL_CASES, ids=[c[0] for c in SPATIAL_CASES])
def test_linear_matches_cpu(dt, ctm, case):
    _name, in_shape, kind, param, out_shape, _sc = case
    data = _sample(in_shape).astype(dt)
    kw = (
        {"scales": np.array(param, np.float32)}
        if kind == "scales"
        else {"sizes": np.array(param, np.int64)}
    )
    model = build_resize(
        dt=dt, in_shape=in_shape, out_shape=out_shape, mode="linear", ctm=ctm, **kw
    )
    m.assert_matches_cpu(model, {"x": data}, **_TOL[np.dtype(dt)])


def test_linear_1d_ncw_matches_cpu():
    # 1-D (NCW) linear: only the innermost (W) axis is resized; N,C stay put.
    in_shape, out_shape = [1, 2, 4], [1, 2, 9]
    data = _sample(in_shape)
    model = build_resize(
        dt=np.float32,
        in_shape=in_shape,
        out_shape=out_shape,
        scales=np.array([1.0, 1.0, 9 / 4], np.float32),
        mode="linear",
        ctm="half_pixel",
    )
    m.assert_matches_cpu(model, {"x": data}, **_TOL[np.dtype(np.float32)])


# --- bf16 (MLX-interior subgraph vs numpy reference) --------------------------------------------
@pytest.mark.parametrize("mode", ["nearest", "linear"])
@pytest.mark.parametrize("scale,out_len", [(2.0, 8), (1.2, 4)])
def test_bf16_interior_matches_ref(resize_route, mode, scale, out_len):
    in_shape, out_shape = [1, 2, 4, 4], [1, 2, out_len, out_len]
    scales = np.array([1.0, 1.0, scale, scale], np.float32)
    data = _sample(in_shape)
    attrs = {"mode": mode, "coordinate_transformation_mode": "half_pixel"}
    if mode == "nearest":
        attrs["nearest_mode"] = "round_prefer_floor"
    # scales as an initializer inside a bf16-interior Cast->Resize->Cast subgraph.
    x = m.tensor("x", DT.FLOAT, in_shape)
    x_bf = m.tensor("x_bf", DT.BFLOAT16, in_shape)
    scales_v = initz("scales", np.array(scales, np.float32))
    roi_v = initz("", np.array([], np.float32))
    y_bf = m.tensor("y_bf", DT.BFLOAT16, out_shape)
    out = m.tensor("out", DT.FLOAT, out_shape)
    nodes = [
        ir.node("Cast", [x], attributes={"to": int(DT.BFLOAT16)}, outputs=[x_bf]),
        ir.node(
            "Resize",
            [x_bf, roi_v, scales_v],
            attributes={
                "mode": attrs["mode"],
                "coordinate_transformation_mode": "half_pixel",
                **(
                    {"nearest_mode": attrs["nearest_mode"]}
                    if mode == "nearest"
                    else {}
                ),
            },
            outputs=[y_bf],
        ),
        ir.node("Cast", [y_bf], attributes={"to": int(DT.FLOAT)}, outputs=[out]),
    ]
    graph = ir.Graph(
        [x],
        [out],
        nodes=nodes,
        initializers=[scales_v],
        opset_imports={"": 19},
        name="mlx_bf16_Resize",
    )
    model = ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()
    ref = resize_ref(
        data,
        out_shape,
        scales.tolist(),
        mode=mode,
        ctm="half_pixel",
        nmode="round_prefer_floor",
    )
    m.assert_matches_ref(model, {"x": data}, [ref], rtol=1e-2, atol=1e-2)


# --- dynamic (symbolic) spatial dims --------------------------------------------------------------
# FPN / super-resolution / GAN exports declare Resize inputs as [N, C, -1, -1] (symbolic H/W) with
# CONSTANT scales or sizes. The claim accepts dynamic spatial dims; the shape-keyed general path
# resolves the concrete extent at trace time, so these must run ON MLX (not fall back) and match CPU.
# (Runtime/data-dependent sizes — e.g. FCN's Shape-derived sizes — stay on CPU and aren't covered.)


def _dyn_resize(dt, channels: int, *, scales=None, sizes=None, mode: str, ctm: str) -> bytes:
    ir_dt = _IR_OF[np.dtype(dt)]
    dims: list[int | str] = [1, channels, "H", "W"]
    x = ir.Value(name="x", type=ir.TensorType(ir_dt), shape=ir.Shape(dims))
    inputs = [x, initz("", np.array([], np.float32))]
    inits = []
    if scales is not None:
        inputs.append(initz("scales", scales))
        inits.append(initz("scales", scales))
    else:
        inputs.append(initz("", np.array([], np.float32)))
        inputs.append(initz("sizes", sizes))
        inits.append(initz("sizes", sizes))
    out = ir.Value(
        name="o", type=ir.TensorType(ir_dt), shape=ir.Shape([1, channels, "OH", "OW"])
    )
    node = ir.node(
        "Resize",
        inputs,
        attributes={"mode": mode, "coordinate_transformation_mode": ctm},
        outputs=[out],
    )
    graph = ir.Graph([x], [out], nodes=[node], initializers=inits, opset_imports={"": 19}, name="mlx_Resize_dyn")
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


def _assert_resize_claimed(model: bytes, feeds, tol) -> None:
    m.assert_op_claimed(model, feeds, "Resize", **tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize(
    "mode,ctm", [("nearest", "asymmetric"), ("linear", "pytorch_half_pixel")], ids=["nearest", "linear"]
)
def test_resize_dynamic_spatial_scales(dtype, mode, ctm) -> None:
    # constant 2x spatial scales, dynamic H/W input
    scales = np.array([1, 1, 2, 2], np.float32)
    model = _dyn_resize(dtype, 3, scales=scales, mode=mode, ctm=ctm)
    x = np.random.default_rng(1).random((1, 3, 7, 5)).astype(dtype)
    _assert_resize_claimed(model, {"x": x}, _TOL[np.dtype(dtype)])


@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize(
    "mode,ctm", [("nearest", "asymmetric"), ("linear", "pytorch_half_pixel")], ids=["nearest", "linear"]
)
def test_resize_dynamic_spatial_sizes(dtype, mode, ctm) -> None:
    # constant target sizes, dynamic H/W input
    sizes = np.array([1, 3, 10, 12], np.int64)
    model = _dyn_resize(dtype, 3, sizes=sizes, mode=mode, ctm=ctm)
    x = np.random.default_rng(2).random((1, 3, 7, 5)).astype(dtype)
    _assert_resize_claimed(model, {"x": x}, _TOL[np.dtype(dtype)])


# --- same integer extent does not imply identity coordinates ------------------------------------
@pytest.fixture(params=[False, True], ids=["default", "eager"])
def resize_route(request, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if request.param else "0")


def _strict_resize_session(model):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"])
    session.disable_fallback()
    return session


def test_resize_same_extent_nearest_counterexample(resize_route):
    shape = [1, 1, 1, 4]
    data = np.array([0, 1, 2, 3], np.float32).reshape(shape)
    expected = np.array([0, 1, 2, 2], np.float32).reshape(shape)
    model = build_resize(
        dt=np.float32, in_shape=shape, out_shape=shape,
        scales=np.array([1, 1, 1, 1.2], np.float32),
        mode="nearest", ctm="half_pixel", nmode="round_prefer_floor",
    )
    actual = _strict_resize_session(model).run(None, {"x": data})[0]
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def _upsample9(*, dt, shape, scales, mode):
    x = m.tensor("x", _IR_OF[np.dtype(dt)], shape)
    out = m.tensor("o", _IR_OF[np.dtype(dt)], shape)
    scales_v = initz("scales", scales)
    node = ir.node("Upsample", [x, scales_v], attributes={"mode": mode}, outputs=[out])
    graph = ir.Graph(
        [x], [out], nodes=[node], initializers=[scales_v],
        opset_imports={"": 9}, name="mlx_Upsample9_same_extent",
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


_SAME_EXTENT_CASES = [
    ("Resize", "nearest", ctm, nmode)
    for ctm in CTMS
    for nmode in ["round_prefer_floor", "round_prefer_ceil", "floor", "ceil"]
] + [
    ("Resize", "linear", ctm, None) for ctm in CTMS
] + [
    ("Upsample", "nearest", "asymmetric", "floor"),
    ("Upsample", "linear", "asymmetric", None),
]


def _same_extent_golden(mode, ctm, nmode, axis):
    data = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    if mode == "nearest":
        if ctm == "align_corners" or nmode == "ceil":
            indices = [0, 1, 2, 3]
        elif nmode == "floor":
            indices = [0, 0, 1, 2]
        else:
            indices = [0, 1, 2, 2]
        return np.take(data, indices, axis=axis)
    # float32(1.2) is exactly 5033165/4194304. These clamped coordinates are derived
    # from that rational, not from resize_ref or the production coordinate helper.
    if ctm == "align_corners":
        coords = [0, 1, 2, 3]
    elif ctm == "asymmetric":
        coords = [0, 4194304 / 5033165, 8388608 / 5033165, 12582912 / 5033165]
    else:
        coords = [0, 7549747 / 10066330, 3187671 / 2013266, 24326963 / 10066330]
    coords = np.array(coords, np.float32)
    if axis == 2:
        return (4 * coords[:, None] + np.arange(4, dtype=np.float32)).reshape(1, 1, 4, 4)
    return (4 * np.arange(4, dtype=np.float32)[:, None] + coords).reshape(1, 1, 4, 4)


@pytest.mark.parametrize("op,mode,ctm,nmode", _SAME_EXTENT_CASES)
@pytest.mark.parametrize("axis", [2, 3], ids=["height", "width"])
def test_resize_ref_same_extent_goldens(op, mode, ctm, nmode, axis):
    data = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    scales = np.ones(4, np.float32)
    scales[axis] = np.float32(1.2)
    actual = resize_ref(
        data, list(data.shape), scales.tolist(), mode=mode, ctm=ctm, nmode=nmode,
    )
    expected = _same_extent_golden(mode, ctm, nmode, axis)
    if mode == "nearest":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, **_TOL[np.dtype(np.float32)])


@pytest.mark.parametrize("op,mode,ctm,nmode", _SAME_EXTENT_CASES)
@pytest.mark.parametrize("axis", [2, 3], ids=["height", "width"])
@pytest.mark.parametrize("dt", [np.float32, np.float16])
def test_same_extent_resampling_replays(resize_route, op, mode, ctm, nmode, axis, dt):
    shape = [1, 1, 4, 4]
    scales = np.ones(4, np.float32)
    scales[axis] = np.float32(1.2)
    if op == "Upsample":
        model = _upsample9(dt=dt, shape=shape, scales=scales, mode=mode)
    else:
        model = build_resize(
            dt=dt, in_shape=shape, out_shape=shape, scales=scales,
            mode=mode, ctm=ctm, nmode=nmode,
        )
    session = _strict_resize_session(model)
    data = np.arange(16, dtype=np.float32).reshape(shape)
    golden = _same_extent_golden(mode, ctm, nmode, axis)
    for factor, offset in [(1, 0), (2, 7), (1, 0)]:
        actual = session.run(None, {"x": (data * factor + offset).astype(dt)})[0]
        expected = (golden * factor + offset).astype(dt)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        if mode == "nearest":
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, **_TOL[np.dtype(dt)])


@pytest.mark.parametrize("mode", ["nearest", "linear"])
@pytest.mark.parametrize("ctm", CTMS)
def test_same_extent_shape_replay(resize_route, mode, ctm):
    scales = np.array([1, 1, 1, 1.2], np.float32)
    session = _strict_resize_session(
        _dyn_resize(np.float32, 1, scales=scales, mode=mode, ctm=ctm)
    )
    for width in [4, 5, 0, 4]:
        data = np.arange(4 * width, dtype=np.float32).reshape(1, 1, 4, width)
        out_shape = [1, 1, 4, int(np.floor(width * float(scales[-1])))]
        expected = (
            _same_extent_golden(mode, ctm, "round_prefer_floor", 3)
            if width == 4 else resize_ref(
                data, out_shape, scales.tolist(), mode=mode, ctm=ctm,
            )
        )
        actual = session.run(None, {"x": data})[0]
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected, **_TOL[np.dtype(np.float32)])


_IDENTITY_CASES = [
    (kind, ctm)
    for kind in ["unit_scales", "equal_sizes", "singleton"]
    for ctm in CTMS
] + [("fractional_align_corners", "align_corners")]


@pytest.mark.parametrize("kind,ctm", _IDENTITY_CASES)
@pytest.mark.parametrize("mode", ["nearest", "linear"])
@pytest.mark.parametrize("dt", [np.float32, np.float16])
def test_resize_identity_preserves_nonfinite(resize_route, kind, ctm, mode, dt):
    data = np.array([np.nan, np.inf, -np.inf, -0.0], dtype=dt).reshape(
        [1, 1, 4, 1] if kind == "singleton" else [1, 1, 1, 4]
    )
    shape = list(data.shape)
    scales = np.ones(4, np.float32)
    if kind in ["singleton", "fractional_align_corners"]:
        scales[-1] = np.float32(1.2)
    params = (
        {"sizes": np.array(shape, np.int64)}
        if kind == "equal_sizes" else {"scales": scales}
    )
    session = _strict_resize_session(build_resize(
        dt=dt, in_shape=shape, out_shape=shape, mode=mode, ctm=ctm, **params,
    ))
    actual = session.run(None, {"x": data})[0]
    assert actual.shape == data.shape
    assert actual.dtype == data.dtype
    np.testing.assert_array_equal(actual, data)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(data))
    reference = resize_ref(
        data, shape, scales.tolist(), mode=mode, ctm=ctm,
    )
    np.testing.assert_array_equal(reference, data)


@pytest.mark.parametrize("singleton", [False, True], ids=["unit_scales", "singleton"])
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_upsample9_identity_preserves_nonfinite(resize_route, singleton, mode):
    data = np.array([np.nan, np.inf, -np.inf, -0.0], np.float32).reshape(
        [1, 1, 4, 1] if singleton else [1, 1, 1, 4]
    )
    scales = np.array([1, 1, 1, 1.2 if singleton else 1], np.float32)
    session = _strict_resize_session(_upsample9(
        dt=np.float32, shape=list(data.shape), scales=scales, mode=mode,
    ))
    actual = session.run(None, {"x": data})[0]
    assert actual.shape == data.shape
    assert actual.dtype == data.dtype
    np.testing.assert_array_equal(actual, data)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(data))
