"""Correctness tests for the MLX EP shape / data-movement op family.

Each op is exercised through the MLX EP (with ORT CPU fallback available) and compared against
ORT's CPU EP, tolerance-gated. Data-movement ops are dtype-agnostic, so they are parametrized over
float/half/int/bool to cover the dtype-generic translate path. Shape parameters are generally
supplied as constant initializers, mirroring a constant-folded model. Dedicated live-parameter
cases also check eager runtime freshness.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

import _models as m

DT = ir.DataType

_IR_OF = {
    np.dtype("float32"): DT.FLOAT,
    np.dtype("float16"): DT.FLOAT16,
    np.dtype("int64"): DT.INT64,
    np.dtype("int32"): DT.INT32,
    np.dtype("bool"): DT.BOOL,
}

# Tolerances: movement ops copy bytes exactly, so tight tolerances hold for every dtype.
_TOL = {
    np.dtype("float32"): dict(rtol=1e-6, atol=0.0),
    np.dtype("float16"): dict(rtol=1e-3, atol=0.0),
    np.dtype("int64"): dict(rtol=0.0, atol=0.0),
    np.dtype("int32"): dict(rtol=0.0, atol=0.0),
    np.dtype("bool"): dict(rtol=0.0, atol=0.0),
}

# Dtypes used to prove the pure data-movement path is dtype-generic.
MOVE_DTYPES = [np.float32, np.float16, np.int64, np.bool_]


def ir_of(dt) -> ir.DataType:
    return _IR_OF[np.dtype(dt)]


def sample(dt, shape) -> np.ndarray:
    """Deterministic test data of the requested numpy dtype and shape."""
    npd = np.dtype(dt)
    n = int(np.prod(shape))
    if npd == np.dtype("bool"):
        return (np.arange(n) % 2 == 0).reshape(shape)
    return np.arange(n, dtype=npd).reshape(shape)


def initz(name: str, arr: np.ndarray) -> ir.Value:
    """A constant initializer value (const_value set) — read by the EP at translate time."""
    t = ir.tensor(arr, name=name)
    return ir.Value(
        name=name, type=ir.TensorType(t.dtype), shape=ir.Shape(list(arr.shape)), const_value=t
    )


def build(
    op: str,
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    *,
    inits: tuple[ir.Value, ...] = (),
    attrs: list[ir.Attr] | None = None,
    domain: str = "",
    opset: int = 24,
) -> bytes:
    """Single-node model; initializer inputs are pulled out into the graph initializer list."""
    node = ir.node(op, inputs, attributes={a.name: a for a in (attrs or [])}, domain=domain, outputs=outputs)
    graph_inputs = [i for i in inputs if i.const_value is None]
    imports = {"": opset}
    if domain:
        imports[domain] = 1
    graph = ir.Graph(
        graph_inputs,
        outputs,
        nodes=[node],
        initializers=list(inits),
        opset_imports=imports,
        name=f"mlx_{op}",
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def tol(dt) -> dict:
    return _TOL[np.dtype(dt)]


# --- Gather / GatherElements ----------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_gather_axis0_negative_indices(dt):
    data = sample(dt, [4, 3])
    idx = np.array([-1, 0, 2], np.int64)  # negative index wraps to the last row
    model = build(
        "Gather",
        [m.tensor("d", ir_of(dt), [4, 3]), m.tensor("i", DT.INT64, [3])],
        [m.tensor("o", ir_of(dt), [3, 3])],
        attrs=[ir.AttrInt64("axis", 0)],
    )
    m.assert_matches_cpu(model, {"d": data, "i": idx}, **tol(dt))


@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_gather_axis1(dt):
    data = sample(dt, [3, 5])
    idx = np.array([[0, 4], [2, 1], [3, 0]], np.int64)
    model = build(
        "Gather",
        [m.tensor("d", ir_of(dt), [3, 5]), m.tensor("i", DT.INT64, [3, 2])],
        [m.tensor("o", ir_of(dt), [3, 3, 2])],
        attrs=[ir.AttrInt64("axis", 1)],
    )
    m.assert_matches_cpu(model, {"d": data, "i": idx}, **tol(dt))


@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_gather_elements(dt):
    data = sample(dt, [3, 3])
    idx = np.array([[0, 1, 2], [2, 1, 0], [1, 1, 1]], np.int64)
    model = build(
        "GatherElements",
        [m.tensor("d", ir_of(dt), [3, 3]), m.tensor("i", DT.INT64, [3, 3])],
        [m.tensor("o", ir_of(dt), [3, 3])],
        attrs=[ir.AttrInt64("axis", 1)],
    )
    m.assert_matches_cpu(model, {"d": data, "i": idx}, **tol(dt))


# --- Concat -----------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_concat(dt, axis):
    a = sample(dt, [2, 3])
    b = sample(dt, [2, 3])
    out_shape = [4, 3] if axis == 0 else [2, 6]
    model = build(
        "Concat",
        [m.tensor("a", ir_of(dt), [2, 3]), m.tensor("b", ir_of(dt), [2, 3])],
        [m.tensor("o", ir_of(dt), out_shape)],
        attrs=[ir.AttrInt64("axis", axis)],
    )
    m.assert_matches_cpu(model, {"a": a, "b": b}, **tol(dt))


# --- Reshape ----------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_reshape(dt):
    data = sample(dt, [2, 3, 4])
    shape = np.array([6, 4], np.int64)
    model = build(
        "Reshape",
        [m.tensor("d", ir_of(dt), [2, 3, 4]), initz("s", shape)],
        [m.tensor("o", ir_of(dt), [6, 4])],
        inits=(initz("s", shape),),
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


def test_reshape_zero_and_infer():
    # 0 copies the input dim (allowzero default), -1 is inferred by MLX.
    data = sample(np.float32, [2, 3, 4])
    shape = np.array([0, -1], np.int64)
    model = build(
        "Reshape",
        [m.tensor("d", DT.FLOAT, [2, 3, 4]), initz("s", shape)],
        [m.tensor("o", DT.FLOAT, [2, 12])],
        inits=(initz("s", shape),),
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(np.float32))


def test_reshape_allowzero_without_zeros():
    # dynamo exporter sets allowzero=1 on every Reshape; with no 0 in the target it is a plain reshape.
    data = sample(np.float32, [2, 3, 4])
    shape = np.array([2, -1, 4], np.int64)
    model = build(
        "Reshape",
        [m.tensor("d", DT.FLOAT, [2, 3, 4]), initz("s", shape)],
        [m.tensor("o", DT.FLOAT, [2, 3, 4])],
        inits=(initz("s", shape),),
        attrs=[ir.AttrInt64("allowzero", 1)],
    )
    m.assert_op_claimed(model, {"d": data}, "Reshape", **tol(np.float32))


def test_reshape_allowzero_literal_zero():
    # allowzero=1: a 0 is a real zero-size dim, not a copy of the input dim ([0, 4] -> [4, 0]).
    data = np.zeros((0, 4), np.float32)
    shape = np.array([4, 0], np.int64)
    model = build(
        "Reshape",
        [m.tensor("d", DT.FLOAT, [0, 4]), initz("s", shape)],
        [m.tensor("o", DT.FLOAT, [4, 0])],
        inits=(initz("s", shape),),
        attrs=[ir.AttrInt64("allowzero", 1)],
    )
    m.assert_op_claimed(model, {"d": data}, "Reshape", **tol(np.float32))
    assert m.run_mlx(model, {"d": data})[0].shape == (4, 0)


# --- Transpose --------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_transpose_perm(dt):
    data = sample(dt, [2, 3, 4])
    model = build(
        "Transpose",
        [m.tensor("d", ir_of(dt), [2, 3, 4])],
        [m.tensor("o", ir_of(dt), [4, 2, 3])],
        attrs=[ir.AttrInt64s("perm", [2, 0, 1])],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


def test_transpose_default_reverse():
    # No perm => reverse all axes.
    data = sample(np.float32, [2, 3, 4])
    model = build(
        "Transpose",
        [m.tensor("d", DT.FLOAT, [2, 3, 4])],
        [m.tensor("o", DT.FLOAT, [4, 3, 2])],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(np.float32))


# --- Unsqueeze / Squeeze ----------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_unsqueeze(dt):
    data = sample(dt, [2, 3])
    axes = np.array([0, 3], np.int64)
    model = build(
        "Unsqueeze",
        [m.tensor("d", ir_of(dt), [2, 3]), initz("ax", axes)],
        [m.tensor("o", ir_of(dt), [1, 2, 3, 1])],
        inits=(initz("ax", axes),),
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_squeeze_axes(dt):
    data = sample(dt, [1, 2, 1, 3])
    axes = np.array([0, 2], np.int64)
    model = build(
        "Squeeze",
        [m.tensor("d", ir_of(dt), [1, 2, 1, 3]), initz("ax", axes)],
        [m.tensor("o", ir_of(dt), [2, 3])],
        inits=(initz("ax", axes),),
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


def test_squeeze_all():
    # No axes => drop every size-1 dim.
    data = sample(np.float32, [1, 3, 1])
    model = build(
        "Squeeze",
        [m.tensor("d", DT.FLOAT, [1, 3, 1])],
        [m.tensor("o", DT.FLOAT, [3])],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(np.float32))


# --- Flatten ----------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_flatten(dt, axis):
    data = sample(dt, [2, 3, 4])
    rows = int(np.prod([2, 3, 4][:axis])) if axis else 1
    cols = int(np.prod([2, 3, 4][axis:]))
    model = build(
        "Flatten",
        [m.tensor("d", ir_of(dt), [2, 3, 4])],
        [m.tensor("o", ir_of(dt), [rows, cols])],
        attrs=[ir.AttrInt64("axis", axis)],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


# --- Expand -----------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_expand_broadcast(dt):
    data = sample(dt, [3, 1])
    shape = np.array([1, 4], np.int64)  # bidirectional broadcast -> [3, 4]
    model = build(
        "Expand",
        [m.tensor("d", ir_of(dt), [3, 1]), initz("s", shape)],
        [m.tensor("o", ir_of(dt), [3, 4])],
        inits=(initz("s", shape),),
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


def test_expand_add_leading_dim():
    data = sample(np.float32, [3, 1])
    shape = np.array([2, 3, 4], np.int64)
    model = build(
        "Expand",
        [m.tensor("d", DT.FLOAT, [3, 1]), initz("s", shape)],
        [m.tensor("o", DT.FLOAT, [2, 3, 4])],
        inits=(initz("s", shape),),
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(np.float32))


# --- Slice ------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_slice_step(dt):
    data = sample(dt, [5, 4])
    starts = np.array([1, 0], np.int64)
    ends = np.array([5, 3], np.int64)
    axes = np.array([0, 1], np.int64)
    steps = np.array([2, 1], np.int64)  # step>1 on axis 0
    ins = (initz("s", starts), initz("e", ends), initz("a", axes), initz("t", steps))
    model = build(
        "Slice",
        [m.tensor("d", ir_of(dt), [5, 4]), *ins],
        [m.tensor("o", ir_of(dt), [2, 3])],
        inits=ins,
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


def test_slice_no_axes_no_steps():
    data = sample(np.float32, [4, 6])
    starts = np.array([0, 1], np.int64)
    ends = np.array([4, 5], np.int64)
    ins = (initz("s", starts), initz("e", ends))
    model = build(
        "Slice",
        [m.tensor("d", DT.FLOAT, [4, 6]), *ins],
        [m.tensor("o", DT.FLOAT, [4, 4])],
        inits=ins,
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(np.float32))


@pytest.mark.parametrize("dt", MOVE_DTYPES)
@pytest.mark.parametrize(
    ("starts", "ends", "axes", "steps"),
    [
        ([np.iinfo(np.int64).max], [np.iinfo(np.int64).min], [1], [-1]),
        ([-1], [-6], [1], [-2]),
        ([-10], [np.iinfo(np.int64).min], [1], [-1]),
        ([2, 0], [0, 5], [0, 1], [-1, 2]),
    ],
    ids=["full-reverse", "negative-bounds", "start-before-dimension", "mixed-directions"],
)
def test_slice_negative_step(dt, starts, ends, axes, steps):
    data = sample(dt, [3, 5])
    values = tuple(
        initz(name, np.asarray(value, np.int64))
        for name, value in zip(("s", "e", "a", "t"), (starts, ends, axes, steps))
    )
    expected = data[
        tuple(
            slice(starts[axes.index(axis)], ends[axes.index(axis)], steps[axes.index(axis)])
            if axis in axes
            else slice(None)
            for axis in range(data.ndim)
        )
    ]
    if starts == [-10]:
        # ONNX clamps a reverse Slice start to [0, dim-1], unlike NumPy's empty result here.
        expected = data[:, :1]
    model = build(
        "Slice",
        [m.tensor("d", ir_of(dt), [3, 5]), *values],
        [m.tensor("o", ir_of(dt), list(expected.shape))],
        inits=values,
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))
    np.testing.assert_array_equal(m.run_mlx(model, {"d": data})[0], expected)


# --- Split ------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_split_sizes(dt):
    data = sample(dt, [2, 6])
    split = np.array([2, 4], np.int64)
    outs = [m.tensor("o0", ir_of(dt), [2, 2]), m.tensor("o1", ir_of(dt), [2, 4])]
    model = build(
        "Split",
        [m.tensor("d", ir_of(dt), [2, 6]), initz("sp", split)],
        outs,
        inits=(initz("sp", split),),
        attrs=[ir.AttrInt64("axis", 1)],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


def test_split_num_outputs():
    # Even split into num_outputs equal chunks (opset-18 attr form, no split input).
    data = sample(np.float32, [2, 6])
    outs = [m.tensor("o0", DT.FLOAT, [2, 3]), m.tensor("o1", DT.FLOAT, [2, 3])]
    model = build(
        "Split",
        [m.tensor("d", DT.FLOAT, [2, 6])],
        outs,
        attrs=[ir.AttrInt64("axis", 1), ir.AttrInt64("num_outputs", 2)],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(np.float32))


# --- SplitToSequence-24 -------------------------------------------------------------------------
def _split_to_sequence_model(
    data_shape: list[int],
    *,
    split: np.ndarray | None,
    axis: int,
    keepdims: int,
) -> bytes:
    data = m.tensor("d", DT.FLOAT, data_shape)
    output = ir.Value(name="seq", type=ir.SequenceType(ir.TensorType(DT.FLOAT)))
    inputs = [data]
    initializers = []
    if split is not None:
        split_value = initz("split", split)
        inputs.append(split_value)
        initializers.append(split_value)
    node = ir.node(
        "SplitToSequence",
        inputs,
        attributes={"axis": axis, "keepdims": keepdims},
        outputs=[output],
    )
    graph = ir.Graph(
        [data],
        [output],
        nodes=[node],
        initializers=initializers,
        name="mlx_SplitToSequence",
        opset_imports={"": 24},
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


@pytest.mark.parametrize(
    ("split", "axis", "keepdims"),
    [
        (None, 0, 1),  # Default split size one.
        (np.array(2, dtype=np.int64), 1, 1),  # Scalar split size.
        (np.array([1, 3], dtype=np.int64), 1, 1),  # Per-output split sizes.
        (np.array([2, 2], dtype=np.int64), 1, 0),  # Remove split axis from each output.
    ],
)
def test_split_to_sequence_v24_declines_container_output(
    split: np.ndarray | None, axis: int, keepdims: int
) -> None:
    data = sample(np.float32, [2, 4])
    model = _split_to_sequence_model(
        [2, 4],
        split=split,
        axis=axis,
        keepdims=keepdims,
    )
    m.assert_op_declined(model, {"d": data}, "SplitToSequence")
    expected = m.run_cpu(model, {"d": data})[0]
    actual = m.run_mlx(model, {"d": data})[0]
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)


# --- Tile -------------------------------------------------------------------------------------


def test_tile_runtime_repeats_are_claimed_and_fresh():
    model = m.make_model(
        "Tile",
        [m.tensor("x", DT.FLOAT, [2, 2]), m.tensor("repeats", DT.INT64, [2])],
        [m.tensor("out", DT.FLOAT, ["rows", "columns"])],
    )
    feeds = {
        "x": np.arange(4, dtype=np.float32).reshape(2, 2),
        "repeats": np.array([1, 2], dtype=np.int64),
    }
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model, options, providers=["MLXExecutionProvider"], enable_fallback=False)
    session.disable_fallback()
    for repeats in [[1, 2], [2, 1], [0, 3], [1, 2]]:
        feeds["repeats"] = np.array(repeats, dtype=np.int64)
        np.testing.assert_array_equal(session.run(None, feeds)[0], np.tile(feeds["x"], repeats))


def test_range_runtime_integer_bounds_stay_fresh_on_mlx():
    model = m.make_model(
        "Range",
        [m.tensor(name, DT.INT64, []) for name in ["start", "limit", "delta"]],
        [m.tensor("out", DT.INT64, ["length"])],
    )
    session = _native_range_session(model)
    for start, limit, delta in [
        (1, 7, 2),
        (4, 0, -1),
        (1, 7, 1),
        (7, 1, 2),
        (2**63 - 6, 2**63 - 2, 1),
        (-(2**63), 2**63 - 1, 2**63 - 1),
        (2**63 - 1, -(2**63), -(2**63)),
        (1, 7, 2),
    ]:
        feeds = {
            name: np.array(value, dtype=np.int64)
            for name, value in zip(
                ["start", "limit", "delta"], [start, limit, delta], strict=True
            )
        }
        np.testing.assert_array_equal(
            session.run(None, feeds)[0],
            np.array(list(range(start, limit, delta)), dtype=np.int64),
        )


def _native_range_session(model):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model, options, providers=["MLXExecutionProvider"], enable_fallback=False
    )
    session.disable_fallback()
    return session


def test_range_live_integer_error_then_recovery():
    model = m.make_model(
        "Range",
        [m.tensor(n, DT.INT64, []) for n in ("start", "limit", "delta")],
        [m.tensor("out", DT.INT64, ["length"])],
    )
    session = _native_range_session(model)

    def feeds(start, limit, delta):
        return {
            name: np.array(v, np.int64)
            for name, v in zip(
                ("start", "limit", "delta"), (start, limit, delta), strict=True
            )
        }

    for invalid in ((0, 3, 0), (-(2**63), 2**63 - 1, 1)):
        with pytest.raises(ort.capi.onnxruntime_pybind11_state.EPFail, match="Range"):
            session.run(None, feeds(*invalid))
        np.testing.assert_array_equal(session.run(None, feeds(1, 7, 2))[0], [1, 3, 5])


def test_range_from_shape_keeps_causal_mask_on_mlx():
    x = m.tensor("x", DT.FLOAT, ["batch", "sequence", 3])
    tape = ir.tape.Tape()

    def constant(value):
        return tape.initializer(
            ir.tensor(np.asarray(value, np.int64), name=f"c{len(tape.initializers)}")
        )

    length = tape.op("Gather", [tape.op("Shape", [x]), constant(1)], {"axis": 0})
    positions = tape.op("Range", [constant(0), length, constant(1)])
    positions.name, positions.dtype, positions.shape = (
        "positions",
        DT.INT64,
        ir.Shape(["sequence"]),
    )
    keys = tape.op("Unsqueeze", [positions, constant([0])])
    queries = tape.op("Unsqueeze", [positions, constant([1])])
    mask = tape.op("LessOrEqual", [keys, queries])
    mask.name, mask.dtype, mask.shape = (
        "mask",
        DT.BOOL,
        ir.Shape(["sequence", "sequence"]),
    )
    graph = ir.Graph(
        [x],
        [positions, mask],
        nodes=tape.nodes,
        initializers=tape.initializers,
        opset_imports={"": 24},
        name="range_mask",
    )
    session = _native_range_session(
        ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    )
    for length in (5, 8, 3, 5):
        positions, mask = session.run(None, {"x": np.ones((1, length, 3), np.float32)})
        np.testing.assert_array_equal(positions, np.arange(length))
        np.testing.assert_array_equal(mask, np.tri(length, dtype=bool))


def test_range_live_axes_feed_compiled_reduction():
    inputs = [m.tensor(n, DT.INT64, []) for n in ("start", "limit", "delta")]
    x = m.tensor("x", DT.FLOAT, [2, 3, 4])
    tape = ir.tape.Tape()
    axes = tape.op("Range", inputs)
    axes.dtype, axes.shape = DT.INT64, ir.Shape(["axes"])
    out = tape.op("ReduceSum", [x, axes], {"keepdims": 1, "noop_with_empty_axes": 1})
    out.name, out.dtype, out.shape = "out", DT.FLOAT, ir.Shape(["a", "b", "c"])
    graph = ir.Graph(
        [*inputs, x], [out], nodes=tape.nodes, opset_imports={"": 24}, name="range_axes"
    )
    session = _native_range_session(
        ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    )
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    for bounds in ((0, 2, 1), (1, 3, 1), (2, -1, -1), (0, 0, 1), (0, 2, 1)):
        feeds = {
            name: np.array(v, np.int64)
            for name, v in zip(("start", "limit", "delta"), bounds, strict=True)
        }
        expected = values.sum(axis=tuple(range(*bounds)), keepdims=True)
        np.testing.assert_array_equal(
            session.run(None, {**feeds, "x": values})[0], expected
        )


@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_tile(dt):
    data = sample(dt, [2, 2])
    reps = np.array([2, 3], np.int64)
    model = build(
        "Tile",
        [m.tensor("d", ir_of(dt), [2, 2]), initz("r", reps)],
        [m.tensor("o", ir_of(dt), [4, 6])],
        inits=(initz("r", reps),),
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


# --- Pad --------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.int64])
def test_pad_constant_default_zero(dt):
    data = sample(dt, [2, 3])
    pads = np.array([1, 0, 1, 2], np.int64)  # begin/end for each of the 2 axes
    model = build(
        "Pad",
        [m.tensor("d", ir_of(dt), [2, 3]), initz("p", pads)],
        [m.tensor("o", ir_of(dt), [4, 5])],
        inits=(initz("p", pads),),
        attrs=[ir.AttrString("mode", "constant")],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


def test_pad_constant_value():
    data = sample(np.float32, [2, 3])
    pads = np.array([1, 1, 1, 1], np.int64)
    cval = np.array(7.0, np.float32)
    ins = (initz("p", pads), initz("cv", cval))
    model = build(
        "Pad",
        [m.tensor("d", DT.FLOAT, [2, 3]), *ins],
        [m.tensor("o", DT.FLOAT, [4, 5])],
        inits=ins,
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(np.float32))


@pytest.mark.parametrize("mode", ["reflect", "edge", "wrap"])
def test_pad_modes_with_axes(mode: str):
    """Opset-25 Pad modes apply only to the selected, non-leading dimensions."""
    data = sample(np.float32, [2, 3, 4])
    pads = initz("p", np.array([1, 2, 1, 1], np.int64))
    constant_value = initz("cv", np.array(0.0, np.float32))
    axes = initz("a", np.array([1, 2], np.int64))
    model = build(
        "Pad",
        [m.tensor("d", DT.FLOAT, [2, 3, 4]), pads, constant_value, axes],
        [m.tensor("o", DT.FLOAT, [2, 5, 7])],
        inits=(pads, constant_value, axes),
        attrs=[ir.AttrString("mode", mode)],
        opset=25,
    )
    m.assert_mlx_claims(model, {"d": data})
    m.assert_matches_cpu(model, {"d": data}, **tol(np.float32))


# --- Identity ---------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_identity(dt):
    data = sample(dt, [2, 3])
    model = build(
        "Identity",
        [m.tensor("d", ir_of(dt), [2, 3])],
        [m.tensor("o", ir_of(dt), [2, 3])],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


# --- ConstantOfShape --------------------------------------------------------------------------
@pytest.mark.parametrize("intermediate", [False, True])
@pytest.mark.parametrize("eager", [False, True], ids=["compiled-enabled", "eager"])
@pytest.mark.parametrize(
    "dtype,np_dtype,fill",
    [
        (DT.FLOAT, np.float32, None),
        (DT.INT64, np.int64, 2**53 + 7),
        (DT.UINT64, np.uint64, 2**63 + 5),
        (DT.DOUBLE, np.float64, 1e-300),
        (DT.BOOL, np.bool_, True),
    ],
)
def test_constant_of_shape_live_values(intermediate, eager, dtype, np_dtype, fill, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    shape = m.tensor("shape", DT.INT64, [2])
    out = m.tensor("out", dtype, ["A", "B"])
    nodes = []
    parameter = shape
    if intermediate:
        parameter = m.tensor("parameter", DT.INT64, [2])
        nodes.append(ir.node("Identity", [shape], outputs=[parameter]))
    attributes = {} if fill is None else {
        "value": ir.AttrTensor("value", ir.tensor(np.array([fill], np_dtype)))
    }
    nodes.append(ir.node("ConstantOfShape", [parameter], attributes=attributes, outputs=[out]))
    graph = ir.Graph(
        [shape], [out], nodes=nodes, name="live_fill_shape", opset_imports={"": 23}
    )
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"])
    for dims in [(2, 3), (3, 2), (0, 4), (2, 3)]:
        actual = session.run(None, {"shape": np.asarray(dims, np.int64)})[0]
        np.testing.assert_array_equal(actual, np.full(dims, 0 if fill is None else fill, np_dtype))


@pytest.mark.parametrize("dimension", [-1, 2**31, 2**32 + 1])
def test_constant_of_shape_rejects_narrowed_dimension(dimension):
    model = build(
        "ConstantOfShape", [m.tensor("shape", DT.INT64, [1])],
        [m.tensor("out", DT.FLOAT, ["N"])],
    )
    session = m._session(model, m.EP_PROVIDERS)
    session.disable_fallback()
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.EPFail, match="ConstantOfShape.*i32"):
        session.run(None, {"shape": np.array([dimension], np.int64)})


def test_expand_rejects_incompatible_dimensions():
    model = build(
        "Expand", [m.tensor("x", DT.FLOAT, [2, 2]), m.tensor("shape", DT.INT64, [2])],
        [m.tensor("out", DT.FLOAT, ["A", "B"])],
    )
    session = m._session(model, m.EP_PROVIDERS)
    session.disable_fallback()
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.EPFail, match="Expand.*cannot broadcast"):
        session.run(None, {"x": np.ones((2, 2), np.float32), "shape": np.array([2, 3], np.int64)})


def test_constant_of_shape_empty_shape_is_scalar():
    model = build(
        "ConstantOfShape", [m.tensor("shape", DT.INT64, [0])],
        [m.tensor("out", DT.FLOAT, [])],
    )
    feeds = {"shape": np.array([], np.int64)}
    m.assert_mlx_claims(model, feeds)
    np.testing.assert_array_equal(m.run_mlx(model, feeds)[0], np.array(0, np.float32))


def test_constant_of_shape_value_guard_replay(tmp_path):
    trace = tmp_path / "fill_shape.json"
    subprocess.run(
        [
            sys.executable, "-m", "pytest", Path(__file__).name,
            "-k", "test_constant_of_shape_live_values and compiled-enabled",
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
    events = [
        e["args"] for e in json.loads(trace.read_text())
        if e.get("cat") == "ep.path" and "ConstantOfShape" in e["args"].get("ops", "").split(",")
    ]
    paths, caches = [], []
    for compiled in [True, True, True, False, True]:
        paths.extend(["general" if compiled else "eager"] * 4 + ["eager"] * 4)
        caches.extend(
            (["MISS", "RETRACE", "RETRACE", "HIT"] if compiled else ["n/a"] * 4) + ["n/a"] * 4
        )
    assert [e["path"] for e in events] == paths
    assert [e["cache"] for e in events] == caches


def _assert_matches_cpu_noopt(model: bytes, feeds: dict, *, rtol: float, atol: float) -> None:
    """Like ``m.assert_matches_cpu`` but with graph optimization disabled.

    ConstantOfShape with a constant shape input is otherwise constant-folded by ORT before the EP
    sees it; ORT_DISABLE_ALL keeps the node in the graph so the MLX handler is actually exercised.
    """
    def run(providers):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(model, opts, providers=providers).run(None, feeds)

    expected = run(["CPUExecutionProvider"])
    actual = run(["MLXExecutionProvider", "CPUExecutionProvider"])
    for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=f"output {i}")


def test_constant_of_shape_default_zero():
    shape = np.array([2, 3], np.int64)
    model = build(
        "ConstantOfShape",
        [initz("s", shape)],
        [m.tensor("o", DT.FLOAT, [2, 3])],
        inits=(initz("s", shape),),
    )
    _assert_matches_cpu_noopt(model, {}, rtol=0.0, atol=0.0)


def test_constant_of_shape_explicit_value():
    """An explicit one-element `value` sets both the fill and the output dtype.

    Previously refused, which stranded the node on CPU; each such node is a partition boundary.
    """
    shape = np.array([2, 3], np.int64)
    model = build(
        "ConstantOfShape",
        [initz("s", shape)],
        [m.tensor("o", DT.FLOAT, [2, 3])],
        inits=(initz("s", shape),),
        attrs=[ir.AttrTensor("value", ir.tensor(np.array([2.5], np.float32)))],
    )
    _assert_matches_cpu_noopt(model, {}, rtol=0.0, atol=0.0)


def test_constant_of_shape_explicit_value_int64():
    """A non-float `value` must keep its own dtype rather than being narrowed through a float."""
    shape = np.array([4], np.int64)
    model = build(
        "ConstantOfShape",
        [initz("s", shape)],
        [m.tensor("o", DT.INT64, [4])],
        inits=(initz("s", shape),),
        attrs=[ir.AttrTensor("value", ir.tensor(np.array([7], np.int64)))],
    )
    _assert_matches_cpu_noopt(model, {}, rtol=0.0, atol=0.0)


def test_constant_of_shape_runtime_shape_from_input():
    """Shape input derived from Shape(x) at run time, not a constant initializer.

    This is the form a Scan's zero-init carried state takes when the batch dim is dynamic; refusing
    it split a dynamic-shape Mamba graph into 79 fused subgraphs instead of 18.
    """
    x = ir.Value(name="x", type=ir.TensorType(DT.FLOAT), shape=ir.Shape(["B", 5]))
    shp = ir.Value(name="shp")
    o = ir.Value(name="o", type=ir.TensorType(DT.FLOAT), shape=ir.Shape(["B", 5]))
    nodes = [
        ir.node("Shape", [x], outputs=[shp]),
        ir.node("ConstantOfShape", [shp], outputs=[o]),
    ]
    graph = ir.Graph([x], [o], nodes=nodes, name="g", opset_imports={"": 20})
    model = ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()
    _assert_matches_cpu_noopt(model, {"x": np.zeros((3, 5), np.float32)}, rtol=0.0, atol=0.0)


def _fp64_constant_of_shape_model() -> bytes:
    """ConstantOfShape whose `value` is float64.

    1e-300 is unrepresentable in float32, so a narrowed fill would flush to zero.
    """
    shape = np.array([2, 3], np.int64)
    return build(
        "ConstantOfShape",
        [initz("s", shape)],
        [m.tensor("o", DT.DOUBLE, [2, 3])],
        inits=(initz("s", shape),),
        attrs=[ir.AttrTensor("value", ir.tensor(np.array([1e-300], np.float64)))],
    )


def test_build_fp64_constant_of_shape_session() -> None:
    """Child of the test below: creating the session is what runs GetCapability."""
    _assert_matches_cpu_noopt(_fp64_constant_of_shape_model(), {}, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    os.environ.get("ONNXRUNTIME_EP_MLX_CLAIM_TEST_CHILD") == "1",
    reason="child process: it builds the session, it does not spawn another",
)
def test_constant_of_shape_float64_value_is_claimed(tmp_path) -> None:
    """A float64 `value` must be claimed, not stranded on CPU.

    ConstantOfShape only fills, so it takes the data-movement dtype set rather than the arithmetic
    one; float64 belongs there because MLX carries the dtype through `full` unchanged. The node is
    coloured float64 by GetCapability and its cluster runs on an MLX CPU stream, so the general
    "MLX rejects float64" rule does not apply to it.

    Asserted from the EP trace rather than from the output: an unclaimed node still produces the
    right answer via ORT CPU, so numerical agreement cannot tell the two apart. The EP reads its
    trace configuration once per process, hence the child process.
    """
    trace = tmp_path / "trace.json"
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{Path(__file__).name}::test_build_fp64_constant_of_shape_session",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        check=False,
        env={
            **os.environ,
            "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
            "ONNXRUNTIME_EP_MLX_CLAIM_TEST_CHILD": "1",
        },
        cwd=str(Path(__file__).parent),
        timeout=300,
    )
    if child.returncode not in (0, 134, -6) or (
        child.returncode in (134, -6) and not trace.exists()
    ):
        child.check_returncode()
    events = json.loads(trace.read_text())
    claims = [event for event in events if event.get("cat") == "ep.claim"]
    assert claims, "the EP recorded no capability decision"
    declined = {
        key.removeprefix("fallback_")
        for claim in claims
        for key in claim["args"]
        if key.startswith("fallback_")
    }
    assert "ConstantOfShape" not in declined, (
        "a float64 `value` was declined; the op only fills, so it takes the data-movement "
        f"dtype set and the fp64 cluster runs on a CPU stream. Declined: {sorted(declined)}"
    )


def _build_dynamic_reshape_model():
    """Runtime Reshape whose target comes from Shape->Gather->Concat with a const tail."""
    import numpy as np
    import onnx_ir as ir
    import _models as m
    DT = ir.DataType
    B, H = 1, 12
    x = ir.Value(name="x", type=ir.TensorType(DT.FLOAT), shape=ir.Shape([B, "S", H]))
    shp = ir.Value(name="shp")
    dims01 = ir.Value(name="dims01")
    tail = m.tensor("tail", DT.INT64, [2])  # const [3,4]
    idx = m.tensor("idx", DT.INT64, [2])    # const [0,1]
    target = ir.Value(name="target")
    o = ir.Value(name="o", type=ir.TensorType(DT.FLOAT), shape=ir.Shape([B, "S", 3, 4]))
    nodes = [
        ir.node("Shape", [x], outputs=[shp]),
        ir.node("Gather", [shp, idx], attributes={"axis": 0}, outputs=[dims01]),
        ir.node("Concat", [dims01, tail], attributes={"axis": 0}, outputs=[target]),
        ir.node("Reshape", [x, target], outputs=[o]),
    ]
    inits = [
        ir.Value(name="idx", type=ir.TensorType(DT.INT64), shape=ir.Shape([2]),
                 const_value=ir.tensor(np.array([0, 1], np.int64))),
        ir.Value(name="tail", type=ir.TensorType(DT.INT64), shape=ir.Shape([2]),
                 const_value=ir.tensor(np.array([3, 4], np.int64))),
    ]
    g = ir.Graph([x], [o], nodes=nodes, initializers=inits, opset_imports={"": 24}, name="mlx_dynreshape")
    return ir.to_proto(ir.Model(g, ir_version=11)).SerializeToString()


def test_build_dynamic_reshape_session() -> None:
    """Child of the test below: creating the session is what runs GetCapability."""
    import numpy as np
    import _models as m
    model = _build_dynamic_reshape_model()
    feed = {"x": np.random.default_rng(0).standard_normal((1, 5, 12)).astype(np.float32)}
    m.assert_matches_cpu(model, feed, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    os.environ.get("ONNXRUNTIME_EP_MLX_CLAIM_TEST_CHILD") == "1",
    reason="child process: it builds the session, it does not spawn another",
)
def test_reshape_dynamic_shape_from_input_shape(tmp_path) -> None:
    """Runtime Reshape whose target is derived from Shape(input) — a SHAPE-CONST value.

    The EP must claim it via the shape-const mid-trace eval, keep the fused partition acyclic, and
    match ORT CPU. Guards the decode de-fragmentation path.

    Asserted from the EP trace, in a child process. The previous version of this test set
    ``ONNXRUNTIME_EP_MLX_CLAIM_DEBUG`` with ``monkeypatch`` and scanned ``capfd``, which could never
    fire: the EP reads its trace configuration once per process, long before a mid-test ``setenv``.
    It also could not have worked even if the variable had been read in time, because an unclaimed
    node still produces the right answer via ORT CPU — so ``assert_matches_cpu`` passing says
    nothing about who claimed it.
    """
    trace = tmp_path / "trace.json"
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{Path(__file__).name}::test_build_dynamic_reshape_session",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        check=False,
        env={
            **os.environ,
            "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
            "ONNXRUNTIME_EP_MLX_CLAIM_TEST_CHILD": "1",
        },
        cwd=str(Path(__file__).parent),
        timeout=300,
    )
    if child.returncode not in (0, 134, -6) or (
        child.returncode in (134, -6) and not trace.exists()
    ):
        child.check_returncode()
    events = json.loads(trace.read_text())
    claims = [event for event in events if event.get("cat") == "ep.claim"]
    assert claims, "the EP recorded no capability decision"
    declined = {
        key.removeprefix("fallback_")
        for claim in claims
        for key in claim["args"]
        if key.startswith("fallback_")
    }
    assert "Reshape" not in declined, (
        "the shape-const Reshape was declined; its target is derivable at trace time, and each "
        f"declined node is a partition boundary. Declined: {sorted(declined)}"
    )


@pytest.mark.skipif(
    os.environ.get("ONNXRUNTIME_EP_MLX_BOUNDARY_SHAPE_CHILD") != "1",
    reason="run by test_boundary_shapes_use_value_guarded_compile",
)
def test_boundary_shape_compile_child() -> None:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    def check(model, feeds):
        mlx = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
        cpu = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
        for feed in feeds:
            actual = mlx.run(None, feed)
            expected = cpu.run(None, feed)
            for a, e in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(a, e)

    for op, data, targets in [
        ("Reshape", np.arange(12, dtype=np.float32), [[3, 4], [3, 4], [2, 6], [2, 6], [3, 4]]),
        ("Expand", np.arange(3, dtype=np.float32).reshape(1, 3),
         [[2, 3], [2, 3], [4, 3], [4, 3], [2, 3]]),
    ]:
        model = build(
            op,
            [m.tensor("x", DT.FLOAT, data.shape), m.tensor("shape", DT.INT64, [2])],
            [m.tensor("y", DT.FLOAT, ["a", "b"])],
        )
        check(model, [
            {"x": data + iteration, "shape": np.array(target, np.int64)}
            for iteration, target in enumerate(targets)
        ])

    model = build(
        "Slice",
        [m.tensor("x", DT.FLOAT, [32]), m.tensor("start", DT.INT64, [1]),
         m.tensor("end", DT.INT64, [1])],
        [m.tensor("y", DT.FLOAT, ["n"])],
    )
    check(model, [
        {"x": np.arange(32, dtype=np.float32) + iteration,
         "start": np.array([start], np.int64), "end": np.array([end], np.int64)}
        for iteration, (start, end) in enumerate([(0, 8), (0, 8), (2, 10), (2, 10), (0, 8)])
    ])
    # The 17th distinct value must use eager; an earlier cached value must still be reusable.
    check(model, [
        {"x": np.arange(32, dtype=np.float32) + iteration,
         "start": np.array([0], np.int64), "end": np.array([end], np.int64)}
        for iteration, end in enumerate([1, 1, *range(2, 18), 1])
    ])


@pytest.mark.skipif(
    os.environ.get("ONNXRUNTIME_EP_MLX_BOUNDARY_SHAPE_CHILD") == "1",
    reason="child process executes the traced workload",
)
def test_boundary_shapes_use_value_guarded_compile(tmp_path: Path) -> None:
    trace = tmp_path / "boundary_shapes.json"
    subprocess.run(
        [sys.executable, "-m", "pytest",
         f"{Path(__file__).name}::test_boundary_shape_compile_child", "-q", "-p", "no:cacheprovider"],
        check=True,
        cwd=str(Path(__file__).parent),
        env={
            **os.environ,
            "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
            "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0",
            "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0",
            "ONNXRUNTIME_EP_MLX_BOUNDARY_SHAPE_CHILD": "1",
        },
        timeout=300,
    )
    events = json.loads(trace.read_text())
    paths = [event for event in events if event.get("cat") == "ep.path"]
    general = [event["args"]["cache"] for event in paths
               if event["name"] == "mlx.compute[general]"]
    assert general[:15] == ["MISS", "HIT", "RETRACE", "HIT", "HIT"] * 3
    assert general[15:] == ["MISS", "HIT", *(["RETRACE"] * 15), "HIT"]
    eager = [event for event in paths if event["name"] == "mlx.compute[eager]"]
    assert len(eager) == 1, "only the uncached 17th value specialization should use eager"


def test_boundary_reshape_invalid_target_then_recovery() -> None:
    model = build(
        "Reshape",
        [m.tensor("x", DT.FLOAT, [12]), m.tensor("shape", DT.INT64, [2])],
        [m.tensor("y", DT.FLOAT, ["a", "b"])],
    )
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    x = np.arange(12, dtype=np.float32)
    session.run(None, {"x": x, "shape": np.array([3, 4], np.int64)})
    with pytest.raises(Exception, match="Reshape|reshape"):
        session.run(None, {"x": x, "shape": np.array([5, 5], np.int64)})
    actual = session.run(None, {"x": x + 1, "shape": np.array([2, 6], np.int64)})[0]
    np.testing.assert_array_equal(actual, (x + 1).reshape(2, 6))
