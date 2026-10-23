"""Strict GridSample placement, schema histories, tensor-product equations, and live replay.

ONNX schemas 16/20/22 define the contract. ORT 1.31's CPU kernel only covers matching
float/double 2D and non-cubic 3D. Its equations cross-check this independent separable oracle;
ONNX's reference additionally checks general-N floating interpolation (not double or empty,
where the reference narrows coordinates / returns the wrong output shape).
"""

from __future__ import annotations

import itertools
import json
import os
from fractions import Fraction
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
FLOATS = [DT.FLOAT16, DT.BFLOAT16, DT.FLOAT, DT.DOUBLE]
NUMPY = {
    DT.FLOAT16: np.float16, DT.BFLOAT16: ml_dtypes.bfloat16,
    DT.FLOAT: np.float32, DT.DOUBLE: np.float64,
    DT.INT8: np.int8, DT.INT16: np.int16, DT.INT32: np.int32, DT.INT64: np.int64,
    DT.UINT8: np.uint8, DT.UINT16: np.uint16, DT.UINT32: np.uint32, DT.UINT64: np.uint64,
    DT.BOOL: np.bool_,
}


def _model(rank, dtype=DT.FLOAT, grid_dtype=DT.FLOAT, *, mode=None, padding="zeros",
           align=0, opset=22, x_shape=None, grid_shape=None):
    x_shape = x_shape or ["n", "c", *[f"d{i}" for i in range(rank)]]
    grid_shape = grid_shape or ["n", *[f"o{i}" for i in range(rank)], rank]
    inputs, values, nodes = [], [], []
    for name, dtype_, shape in [("x", dtype, x_shape), ("grid", grid_dtype, grid_shape)]:
        value = m.tensor(name, DT.FLOAT if dtype_ == DT.BFLOAT16 else dtype_, shape)
        inputs.append(value)
        if dtype_ == DT.BFLOAT16:
            cast = m.tensor(name + "_bf16", dtype_, shape)
            nodes.append(ir.node("Cast", [value], attributes={"to": int(dtype_)}, outputs=[cast]))
            value = cast
        values.append(value)
    out_shape = [x_shape[0], x_shape[1], *grid_shape[1:-1]]
    y = m.tensor("y", dtype, out_shape)
    attrs = {"padding_mode": padding, "align_corners": align}
    if mode is not None:
        attrs["mode"] = mode
    nodes.append(ir.node("GridSample", values, attributes=attrs, outputs=[y]))
    out = y
    if dtype == DT.BFLOAT16:
        out = m.tensor("out", DT.FLOAT, out_shape)
        nodes.append(ir.node("Cast", [y], attributes={"to": int(DT.FLOAT)}, outputs=[out]))
    graph = ir.Graph(inputs, [out], nodes=nodes, opset_imports={"": opset}, name="grid_sample")
    proto = ir.to_proto(ir.Model(graph, ir_version=11))
    onnx.checker.check_model(proto)
    return proto.SerializeToString()


def _session(model):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"])
    session.disable_fallback()
    return session


def _cast(value, dtype):
    value = value.astype(NUMPY[dtype])
    return value.astype(np.float32) if dtype == DT.BFLOAT16 else value


def _reflect(value, dim, align):
    if dim == 1:
        return 0.0
    low, span = (0, dim - 1) if align else (-0.5, dim)
    # Repeated folding, independently of the device's floating-remainder expression.
    quotient, residual = divmod(value - low, span)
    return low + (residual if int(quotient) % 2 == 0 else span - residual)


def _kernel(distance):
    distance = abs(distance)
    if distance <= 1:
        return 1.25 * distance**3 - 2.25 * distance**2 + 1
    if distance < 2:
        return -0.75 * distance**3 + 3.75 * distance**2 - 6 * distance + 3
    return 0.0


def _expected(x, grid, dtype=DT.FLOAT, grid_dtype=DT.FLOAT, *,
              mode="linear", padding="zeros", align=0):
    x, grid = _cast(x, dtype), _cast(grid, grid_dtype)
    calc = np.float64 if DT.DOUBLE in (dtype, grid_dtype) else np.float32
    output = np.empty((x.shape[0], x.shape[1], *grid.shape[1:-1]), dtype=NUMPY[dtype])
    rank = x.ndim - 2
    for n in range(x.shape[0]):
        for point in np.ndindex(grid.shape[1:-1]):
            coords = []
            for axis, dim in enumerate(x.shape[2:]):
                normalized = calc(grid[(n, *point, rank - axis - 1)])
                if padding == "reflection":
                    normalized = calc(np.fmod(normalized, calc(4)))
                position = ((normalized + calc(1)) * calc(dim - 1) / calc(2) if align else
                            ((normalized + calc(1)) * calc(dim) - calc(1)) / calc(2))
                coords.append(float(position))
            if mode == "nearest":
                index, valid = [], True
                for position, dim in zip(coords, x.shape[2:], strict=True):
                    position = float(np.rint(position))
                    if padding == "zeros":
                        valid &= 0 <= position < dim
                    elif padding == "reflection":
                        position = _reflect(position, dim, align)
                    index.append(int(np.clip(position, 0, dim - 1)))
                output[(n, slice(None), *point)] = x[(n, slice(None), *index)] if valid else 0
                continue
            # Contract one complete axis at a time, instead of enumerating device gather branches.
            values = x[n].astype(calc)
            for position, dim in zip(coords, x.shape[2:], strict=True):
                base = int(np.floor(position))
                offsets = (-1, 0, 1, 2) if mode == "cubic" else (0, 1)
                samples = []
                for offset in offsets:
                    index = base + offset
                    distance = position - index
                    weight = _kernel(distance) if mode == "cubic" else 1 - abs(distance)
                    valid = padding != "zeros" or 0 <= index < dim
                    if padding == "reflection":
                        index = int(_reflect(index, dim, align))
                    index = min(max(index, 0), dim - 1)
                    samples.append(values[:, index] * calc(weight) if valid
                                   else np.zeros_like(values[:, 0]))
                values = sum(samples)
            output[(n, slice(None), *point)] = values
    return output.astype(np.float32) if dtype == DT.BFLOAT16 else output


def _assert(actual, expected, dtype=DT.FLOAT):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if dtype not in FLOATS:
        np.testing.assert_array_equal(actual, expected)
    else:
        tolerance = {DT.FLOAT: 2e-5, DT.FLOAT16: 2e-3, DT.BFLOAT16: 1.6e-2,
                     DT.DOUBLE: 2e-12}[dtype]
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def _inputs(rank, *, offset=0, spatial=None, output=None):
    spatial = spatial or [3, *([2] * (rank - 1))]
    output = output or [1] * (rank - 1) + [4]
    x = (np.arange(2 * 2 * np.prod(spatial)).reshape(2, 2, *spatial) % 17 - 8) / 8 + offset
    grid = np.random.default_rng(3101 + rank).uniform(-3.5, 3.5, (2, *output, rank))
    return x.astype(np.float32), grid.astype(np.float32)


@pytest.fixture(params=[False, True], ids=["compiled", "eager"])
def execution_mode(request, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if request.param else "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE", "0")


@pytest.mark.parametrize("rank", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("mode,padding,align", list(itertools.product(
    ["linear", "nearest", "cubic"], ["zeros", "border", "reflection"], [0, 1])))
def test_grid_sample_forms(rank, mode, padding, align, execution_mode):
    x, grid = _inputs(rank)
    model = _model(rank, mode=mode, padding=padding, align=align)
    session = _session(model)
    for offset in [0, 0.375]:
        feeds = {"x": x + offset, "grid": grid + offset}
        expected = _expected(**feeds, mode=mode, padding=padding, align=align)
        _assert(session.run(None, feeds)[0], expected)
        if rank in (2, 3) and (rank == 2 or mode != "cubic"):
            _assert(m.run_cpu(model, feeds)[0], expected)
        if rank in (1, 4) and mode != "nearest" and padding != "reflection":
            reference = ReferenceEvaluator(onnx.load_model_from_string(model))
            # ONNX's reference clips continuous cubic border coordinates, losing edge overshoot.
            # ORT's independently checked per-neighbor padding equations do not. Cross-check the
            # general-N reference where the two contracts agree, while the full live grid above
            # still verifies all out-of-bound neighbors against the independently derived oracle.
            reference_feeds = {**feeds, "grid": np.clip(feeds["grid"], -0.4, 0.4)} \
                if mode == "cubic" and padding == "border" else feeds
            _assert(reference.run(None, reference_feeds)[0],
                    _expected(**reference_feeds, mode=mode, padding=padding, align=align))


@pytest.mark.parametrize("dtype,grid_dtype", list(itertools.product(FLOATS, repeat=2)))
@pytest.mark.parametrize("mode", ["linear", "nearest", "cubic"])
@pytest.mark.parametrize("align", [0, 1])
def test_grid_sample_independent_float_types(dtype, grid_dtype, mode, align, execution_mode):
    for rank, padding in [(2, "zeros"), (4, "reflection")]:
        x, grid = _inputs(rank)
        x = _cast(x, dtype)
        grid = _cast(grid, grid_dtype)
        model = _model(rank, dtype, grid_dtype, mode=mode, padding=padding, align=align)
        actual = _session(model).run(None, {"x": x, "grid": grid})[0]
        _assert(actual, _expected(x, grid, dtype, grid_dtype, mode=mode,
                                 padding=padding, align=align), dtype)


@pytest.mark.parametrize("mode,padding,align", list(itertools.product(
    ["linear", "nearest", "cubic"], ["zeros", "border", "reflection"], [0, 1])))
def test_grid_sample_separable_reordered_axes(mode, padding, align, execution_mode):
    x, grid = _inputs(4, spatial=[2, 3, 1, 4], output=[2, 1, 3, 2])
    model = _model(4, mode=mode, padding=padding, align=align)
    actual = _session(model).run(None, {"x": x, "grid": grid})[0]
    _assert(actual, _expected(x, grid, mode=mode, padding=padding, align=align))


def test_grid_sample_separable_oversized_intermediate_then_recovery(execution_mode):
    session = _session(_model(4, mode="cubic", padding="reflection"))
    with pytest.raises(Exception, match="element count exceeds int32"):
        session.run(None, {"x": np.ones((1, 1, 32, 32, 32, 32), np.float32),
                           "grid": np.zeros((1, 1, 1, 1, 65536, 4), np.float32)})
    x, grid = _inputs(4)
    _assert(session.run(None, {"x": x, "grid": grid})[0],
            _expected(x, grid, mode="cubic", padding="reflection"))


@pytest.mark.parametrize("dtype", [d for d in NUMPY if d not in FLOATS])
@pytest.mark.parametrize("grid_dtype", FLOATS)
@pytest.mark.parametrize("padding", ["zeros", "border", "reflection"])
def test_grid_sample_nearest_exact_payload(dtype, grid_dtype, padding, execution_mode):
    # These half indices are exact with align_corners=1 and dimension 5.
    pixel_coords = np.array([-10.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 9.5])
    grid = _cast(np.stack([pixel_coords / 2 - 1, np.zeros_like(pixel_coords)], -1)
                 .reshape(1, 1, -1, 2), grid_dtype)
    x = np.arange(10).reshape(1, 1, 2, 5).astype(NUMPY[dtype])
    if dtype in (DT.INT64, DT.UINT64):
        x += np.array(2**60, dtype=NUMPY[dtype])
    elif dtype in (DT.INT32, DT.UINT32):
        x += np.array(2**28, dtype=NUMPY[dtype])
    elif dtype in (DT.INT8, DT.INT16):
        x -= np.array(7, dtype=NUMPY[dtype])
    model = _model(2, dtype, grid_dtype, mode="nearest", padding=padding, align=1)
    session = _session(model)
    for values in [x, np.flip(x).copy()]:
        actual = session.run(None, {"x": values, "grid": grid})[0]
        _assert(actual, _expected(values, grid, dtype, grid_dtype, mode="nearest",
                                 padding=padding, align=1), dtype)


@pytest.mark.parametrize("opset,mode", [(16, None), (16, "bilinear"), (16, "bicubic"),
                                       (16, "nearest"), (20, None), (20, "cubic"),
                                       (22, None), (24, "nearest")])
def test_grid_sample_schema_history(opset, mode, execution_mode):
    x, grid = _inputs(2)
    model = _model(2, mode=mode, opset=opset, padding="reflection")
    actual = _session(model).run(None, {"x": x, "grid": grid})[0]
    canonical = {None: "linear", "bilinear": "linear", "bicubic": "cubic"}.get(mode, mode)
    expected = _expected(x, grid, mode=canonical, padding="reflection")
    _assert(actual, expected)
    _assert(m.run_cpu(model, {"x": x, "grid": grid})[0], expected)


@pytest.mark.parametrize("mode", ["linear", "nearest", "cubic"])
@pytest.mark.parametrize("padding", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align", [0, 1])
def test_grid_sample_dynamic_empty_singleton_and_recovery(mode, padding, align, execution_mode):
    session = _session(_model(2, mode=mode, padding=padding, align=align))
    for shape, grid_shape in [
        ((2, 3, 4, 5), (2, 2, 3, 2)),
        ((1, 2, 1, 5), (1, 3, 2, 2)),
        ((2, 0, 3, 1), (2, 1, 4, 2)),
        ((0, 2, 3, 4), (0, 3, 4, 2)),
        ((0, 1, 2, 2), (0, 65536, 65536, 2)),
        ((2, 1, 1, 1), (2, 0, 3, 2)),
        ((2, 1, 1, 1), (2, 2, 0, 2)),
        ((2, 1, 1, 1), (2, 2, 2, 2)),
    ]:
        x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 8
        grid = np.random.default_rng(12).uniform(-7, 7, grid_shape).astype(np.float32)
        _assert(session.run(None, {"x": x, "grid": grid})[0],
                _expected(x, grid, mode=mode, padding=padding, align=align))
    for x, grid, message in [
        (np.ones((2, 1, 0, 3), np.float32), np.ones((2, 1, 1, 2), np.float32), "spatial"),
        (np.ones((2, 1, 3, 3), np.float32), np.ones((1, 1, 1, 2), np.float32), "batch"),
    ]:
        with pytest.raises(Exception, match=message):
            session.run(None, {"x": x, "grid": grid})
        x, grid = _inputs(2)
        _assert(session.run(None, {"x": x, "grid": grid})[0],
                _expected(x, grid, mode=mode, padding=padding, align=align))


def test_grid_sample_large_empty_output_geometry(execution_mode):
    for static in [False, True]:
        x = np.ones((1, 1, 2, 2, 2), np.float32)
        grid = np.empty((1, 65536, 65536, 0, 3), np.float32)
        session = _session(_model(
            3, mode="cubic", padding="reflection", align=1,
            x_shape=list(x.shape) if static else None,
            grid_shape=list(grid.shape) if static else None))
        _assert(session.run(None, {"x": x, "grid": grid})[0],
                _expected(x, grid, mode="cubic", padding="reflection", align=1))
        if not static:
            x, grid = _inputs(3)
            _assert(session.run(None, {"x": x, "grid": grid})[0],
                    _expected(x, grid, mode="cubic", padding="reflection", align=1))


@pytest.mark.parametrize("mode", ["linear", "nearest", "cubic"])
@pytest.mark.parametrize("align", [0, 1])
def test_grid_sample_double_precision_and_far_reflection(mode, align, execution_mode):
    x = np.array([0, 1, -2, 3, 4, -5, 6, 7, 8, -9, 10, 11], np.float64).reshape(1, 1, 3, 4)
    x += np.float64("0.0000000001234567")
    grid = np.array([[-1.125, 0.1234567890123], [0.250000000123, -0.25],
                     [1000000.375, -1000000.625], [-1000000.375, 1000000.625]], np.float64)
    grid = grid.reshape(1, 1, -1, 2)
    model = _model(2, DT.DOUBLE, DT.DOUBLE, mode=mode, padding="reflection", align=align)
    actual = _session(model).run(None, {"x": x, "grid": grid})[0]
    expected = _expected(x, grid, DT.DOUBLE, DT.DOUBLE, mode=mode,
                         padding="reflection", align=align)
    _assert(actual, expected, DT.DOUBLE)
    _assert(m.run_cpu(model, {"x": x, "grid": grid})[0], expected, DT.DOUBLE)


@pytest.mark.parametrize("mode", ["linear", "nearest", "cubic"])
@pytest.mark.parametrize("padding", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align", [0, 1])
def test_grid_sample_far_float32_coordinates(mode, padding, align, execution_mode):
    x = np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)
    grid = np.array([[2**25 + 8, -2**25 - 8], [-2**25 - 16, 2**25 + 16],
                     [1000000.375, -1000000.625], [-1000000.375, 1000000.625]],
                    np.float32).reshape(1, 1, -1, 2)
    model = _model(2, mode=mode, padding=padding, align=align)
    actual = _session(model).run(None, {"x": x, "grid": grid})[0]
    # Beyond float32's half-integer resolution, use the exact repeated-folding integer oracle,
    # not ORT's floating border subtraction (which can itself lose the half-pixel offset).
    _assert(actual, _expected(x, grid, mode=mode, padding=padding, align=align))


@pytest.mark.parametrize("grid_dtype,value", [(DT.FLOAT, 8388604), (DT.DOUBLE, 4503599627370492)])
@pytest.mark.parametrize("mode", ["linear", "nearest", "cubic"])
def test_grid_sample_reflection_below_old_reduction_threshold(grid_dtype, value, mode,
                                                            execution_mode):
    x = np.array([10, 20, 30], np.float32).reshape(1, 1, 3)
    grid = np.array([value, -value], NUMPY[grid_dtype]).reshape(1, 2, 1)
    model = _model(1, grid_dtype=grid_dtype, mode=mode, padding="reflection")
    actual = _session(model).run(None, {"x": x, "grid": grid})[0]
    # Both normalized values are multiples of four, so both locations are exactly the center.
    np.testing.assert_array_equal(actual, np.full((1, 1, 2), 20, np.float32))


@pytest.mark.parametrize("dim", [2**24 + 1, 2**24 + 2, 2**24 + 3, 2**24 + 4])
def test_grid_sample_large_axis_exact_bounds_and_rounding(dim, execution_mode):
    # At most ~80 MiB of payloads, independently of parameter combinations. This includes the
    # 16 MiB int8 counterexample where float32 rounds dim-1 down to the penultimate pixel.
    integer = np.arange(dim, dtype=np.int8)
    floating = integer.astype(np.float32)
    tiny = np.nextafter(np.float32(0), np.float32(1))
    coords = np.array([-8, -1, -0.5, -2**-40, -tiny, 0, tiny, 2**-40,
                       0.125, 0.5, 1, 8], np.float32)
    grid = coords.reshape(1, -1, 1)
    for mode, padding, align in itertools.product(
            ["nearest", "linear", "cubic"], ["zeros", "border", "reflection"], [0, 1]):
        dtype = DT.INT8 if mode == "nearest" else DT.FLOAT
        data = integer if mode == "nearest" else floating
        expected = []
        for normalized in coords:
            exact = Fraction(float(normalized))
            position = (exact + 1) * (dim - 1) / 2 if align else ((exact + 1) * dim - 1) / 2
            if mode == "nearest":
                indices, weights = [round(position)], [1]
            else:
                base = position.numerator // position.denominator
                indices = [base + i for i in ([-1, 0, 1, 2] if mode == "cubic" else [0, 1])]
                weights = [(_kernel(float(position - i)) if mode == "cubic"
                            else 1 - abs(float(position - i))) for i in indices]
            value = 0
            for index, weight in zip(indices, weights, strict=True):
                if padding == "zeros" and not 0 <= index < dim:
                    continue
                if padding == "reflection":
                    index = int(_reflect(index, dim, align))
                index = min(max(index, 0), dim - 1)
                value += float(data[index]) * weight
            expected.append(value)
        model = _model(1, dtype, mode=mode, padding=padding, align=align)
        session = _session(model)
        actual = session.run(None, {"x": data.reshape(1, 1, -1), "grid": grid})[0]
        _assert(actual, np.array(expected, NUMPY[dtype]).reshape(1, 1, -1), dtype)
        replay = session.run(None, {"x": data.reshape(1, 1, -1),
                                    "grid": grid[:, ::-1].copy()})[0]
        _assert(replay, actual[..., ::-1], dtype)


@pytest.mark.parametrize("grid_dtype", [DT.FLOAT, DT.DOUBLE])
@pytest.mark.parametrize("mode", ["linear", "nearest", "cubic"])
@pytest.mark.parametrize("padding", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align", [0, 1])
def test_grid_sample_extreme_finite_padding(grid_dtype, mode, padding, align, execution_mode):
    x = np.array([10, 20, 30], np.float32).reshape(1, 1, 3)
    maximum = np.finfo(NUMPY[grid_dtype]).max
    grid = np.array([-maximum, maximum, -(2**63), 2**63], NUMPY[grid_dtype]).reshape(1, 4, 1)
    model = _model(1, grid_dtype=grid_dtype, mode=mode, padding=padding, align=align)
    actual = _session(model).run(None, {"x": x, "grid": grid})[0]
    # These representable normalized magnitudes are exact multiples of the reflection period
    # four. No float->integer conversion or overflow-prone denormalization is needed by this
    # analytical oracle. In particular, finite out-of-range grids must never become pixel zero.
    expected = {"zeros": [0, 0, 0, 0], "border": [10, 30, 10, 30],
                "reflection": [20, 20, 20, 20]}[padding]
    np.testing.assert_array_equal(actual, np.array(expected, np.float32).reshape(1, 1, 4))


@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.INT64, DT.UINT64])
def test_grid_sample_double_grid_nearest_thresholds(dtype, execution_mode):
    x = np.arange(9).reshape(1, 1, 3, 3).astype(NUMPY[dtype])
    if dtype in (DT.INT64, DT.UINT64):
        x += np.array(2**60, dtype=NUMPY[dtype])
    epsilon = 2**-40
    grid = np.array([[-0.5 - epsilon, 0], [-0.5, 0], [-0.5 + epsilon, 0],
                     [0.5 - epsilon, 0], [0.5, 0], [0.5 + epsilon, 0]], np.float64)
    grid = grid.reshape(1, 1, -1, 2)
    model = _model(2, dtype, DT.DOUBLE, mode="nearest", align=1)
    actual = _session(model).run(None, {"x": x, "grid": grid})[0]
    np.testing.assert_array_equal(actual, x[:, :, 1:2, [0, 0, 1, 1, 2, 2]])


@pytest.mark.parametrize("padding", ["zeros", "border", "reflection"])
def test_grid_sample_nearest_preserves_float_bits(padding, execution_mode):
    x = np.array([-0.0, 0.0, np.inf, -np.inf, np.nan], np.float32).reshape(1, 1, 1, 5)
    grid = np.stack([np.linspace(-1, 1, 5), np.zeros(5)], axis=-1).astype(np.float32)
    grid = grid.reshape(1, 1, 5, 2)
    model = _model(2, mode="nearest", padding=padding, align=1)
    actual = _session(model).run(None, {"x": x, "grid": grid})[0]
    np.testing.assert_array_equal(actual.view(np.uint32), x.view(np.uint32))


@pytest.mark.parametrize("grid_dtype", FLOATS)
@pytest.mark.parametrize("align", [0, 1])
def test_grid_sample_standard_axis_subnormal_normalization_contract(grid_dtype, align,
                                                                  execution_mode):
    dtype = NUMPY[grid_dtype]
    info = ml_dtypes.finfo(dtype) if grid_dtype == DT.BFLOAT16 else np.finfo(dtype)
    smallest = info.smallest_subnormal
    largest = np.nextafter(dtype(info.tiny), dtype(0), dtype=dtype)
    values = np.array([-largest, -smallest, 0, smallest, largest], dtype=dtype)
    grid = np.stack([np.full(5, -1, dtype=dtype), values], axis=-1).reshape(1, 1, 5, 2)
    grid = _cast(grid, grid_dtype)
    x = np.array([10, 20, 30, 40], np.float32).reshape(1, 1, 2, 2)
    model = _model(2, grid_dtype=grid_dtype, mode="nearest", align=align)
    actual = _session(model).run(None, {"x": x, "grid": grid})[0]
    expected = _expected(x, grid, grid_dtype=grid_dtype, mode="nearest", align=align)
    np.testing.assert_array_equal(actual, expected)

    # Ordinary axes use fp32 (or fp64 for a double grid) denormalization, like ORT. Adding a
    # positive subnormal to one can round to one before nearest-even is applied; this is not
    # evidence that Metal flushed the grid itself. Prove the distinction against an actual CPU
    # kernel and an exact-real rational oracle, instead of silently weakening nearest checks.
    cpu_dtype = DT.DOUBLE if grid_dtype == DT.DOUBLE else DT.FLOAT
    cpu_model = _model(2, cpu_dtype, cpu_dtype, mode="nearest", align=align)
    cpu_x = x.astype(NUMPY[cpu_dtype])
    cpu_grid = grid.astype(NUMPY[cpu_dtype])
    cpu = m.run_cpu(cpu_model, {"x": cpu_x, "grid": cpu_grid})[0]
    np.testing.assert_array_equal(cpu, expected.astype(NUMPY[cpu_dtype]))
    rational = []
    for value in values:
        position = (Fraction(float(value)) + 1) / 2 if align else Fraction(float(value)) + Fraction(1, 2)
        rational.append(x[0, 0, round(position), 0])
    rational = np.array(rational).reshape(1, 1, 1, 5)
    assert not np.array_equal(cpu, rational), "control must expose real-vs-floating normalization"


@pytest.mark.parametrize("kwargs", [
    {"opset": 16, "mode": "linear"}, {"opset": 20, "mode": "bilinear"},
    {"opset": 16, "mode": "cubic"}, {"opset": 22, "mode": "bicubic"},
    {"mode": "invalid"}, {"padding": "invalid"}, {"align": 2},
    {"dtype": DT.INT64, "mode": "linear"}, {"dtype": DT.BOOL, "mode": "cubic"},
    {"x_shape": [1, 1, 0, 3], "grid_shape": [1, 1, 1, 2]},
    {"x_shape": [1, 1, 3, 3], "grid_shape": [2, 1, 1, 2]},
    {"x_shape": [1, 1, 2**30, 3], "grid_shape": [1, 1, 1, 2]},
])
def test_grid_sample_invalid_or_unresolved_forms_are_not_claimed(kwargs):
    # Model loading may reject invalid geometry before GetCapability. Either way, it must not
    # create a success-shaped MLX session or rely on a CPU retry.
    with pytest.raises(Exception, match="fallback|shape|dimension|type|[Ee]rror"):
        _session(_model(2, **kwargs))


def _trace_workload():
    ort.register_execution_provider_library("MLXExecutionProvider",
                                           os.environ["ONNXRUNTIME_MLX_EP_LIB"])
    for rank in [2, 4]:
        session = _session(_model(rank, mode="cubic", padding="reflection"))
        for iteration, size in enumerate([3, 3, 4, 4, 3]):
            x, grid = _inputs(rank, spatial=[size] * rank, offset=iteration / 8)
            grid += iteration / 8
            _assert(session.run(None, {"x": x, "grid": grid})[0],
                    _expected(x, grid, mode="cubic", padding="reflection"))


def test_grid_sample_actual_compile_routes(tmp_path):
    for eager in [False, True]:
        trace = tmp_path / f"grid_sample_{eager}.json"
        subprocess.run(
            [sys.executable, "-c", "import test_grid_sample as t; t._trace_workload()"],
            cwd=Path(__file__).parent,
            env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
                 "ONNXRUNTIME_EP_MLX_NO_COMPILE": str(int(eager)),
                 "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
            check=True, timeout=120)
        events = json.loads(trace.read_text())
        paths = [event for event in events if event.get("cat") == "ep.path"]
        if eager:
            assert [event["name"] for event in paths] == ["mlx.compute[eager]"] * 10
        else:
            assert [(event["name"], event["args"]["cache"]) for event in paths] == [
                ("mlx.compute[general]", cache)
                for cache in ["MISS", "HIT", "RETRACE", "HIT", "HIT"] * 2]
