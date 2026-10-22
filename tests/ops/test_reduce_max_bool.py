"""Boolean ReduceMax semantics, including the empty-set identity missing in ORT CPU."""

from __future__ import annotations

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx_ir import DataType as DT

import _models as m


def _model(
    x: np.ndarray,
    axes: tuple[int, ...] | None,
    *,
    keepdims: int,
    noop: int = 0,
    axes_source: str = "input",
    opset: int = 20,
) -> tuple[bytes, dict[str, np.ndarray], np.ndarray]:
    if not axes and noop:
        expected = x.copy()
    else:
        expected = np.any(x, axis=axes or None, keepdims=bool(keepdims))
    expected = np.asarray(expected, dtype=np.bool_)
    data = m.tensor("x", DT.BOOL, list(x.shape))
    out = m.tensor("out", DT.BOOL, list(expected.shape))
    inputs = [data]
    graph_inputs = [data]
    initializers = []
    feeds = {"x": x}
    if axes is not None:
        axes_array = np.asarray(axes, dtype=np.int64)
        axis = m.tensor("axes", DT.INT64, [len(axes)])
        if axes_source == "initializer":
            axis.const_value = ir.tensor(axes_array)
            initializers.append(axis)
        else:
            graph_inputs.append(axis)
            feeds["axes"] = axes_array
        inputs.append(axis)
    node = ir.node(
        "ReduceMax",
        inputs,
        attributes={"keepdims": keepdims, "noop_with_empty_axes": noop},
        outputs=[out],
    )
    graph = ir.Graph(
        graph_inputs,
        [out],
        nodes=[node],
        initializers=initializers,
        name="mlx_bool_reduce_max",
        opset_imports={"": opset},
    )
    model = ir.to_proto(ir.Model(graph, ir_version=9))
    onnx.checker.check_model(model)
    return model.SerializeToString(), feeds, expected


def _assert_mlx_result(
    model: bytes, feeds: dict[str, np.ndarray], expected: np.ndarray
) -> None:
    actual = m.run_mlx(model, feeds)[0]
    np.testing.assert_array_equal(actual, expected, strict=True)
    m.assert_mlx_claims(model, feeds)


@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("noop", [0, 1])
@pytest.mark.parametrize("axes_source", ["input", "initializer"])
@pytest.mark.parametrize("axes", [None, (), (1,), (-1,), (0, -1)])
@pytest.mark.parametrize("all_false", [False, True], ids=["mixed", "all-false"])
def test_reduce_max_bool(
    keepdims: int,
    noop: int,
    axes_source: str,
    axes: tuple[int, ...] | None,
    all_false: bool,
) -> None:
    x = np.zeros((2, 3, 4), dtype=np.bool_)
    if not all_false:
        x[0, 1, 2] = True
        x[1, 2, 0] = True
    _assert_mlx_result(
        *_model(x, axes, keepdims=keepdims, noop=noop, axes_source=axes_source)
    )


@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("noop", [0, 1])
@pytest.mark.parametrize("axes_source", ["input", "initializer"])
@pytest.mark.parametrize(
    "shape,axes",
    [
        ((2, 0, 4), (1,)),  # Exact ONNX 1.23.2 empty-set fixture when keepdims=1.
        ((2, 0, 4), (0,)),  # A zero dimension survives reduction.
        ((2, 0, 4), (-2, -1)),
        ((2, 0, 4), None),
        ((2, 0, 4), ()),
        ((0,), None),
    ],
)
def test_reduce_max_bool_empty(
    keepdims: int,
    noop: int,
    axes_source: str,
    shape: tuple[int, ...],
    axes: tuple[int, ...] | None,
) -> None:
    _assert_mlx_result(
        *_model(
            np.empty(shape, dtype=np.bool_),
            axes,
            keepdims=keepdims,
            noop=noop,
            axes_source=axes_source,
        )
    )


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("noop", [0, 1])
@pytest.mark.parametrize("axes", [None, ()])
@pytest.mark.parametrize("opset", [20, 24])
def test_reduce_max_bool_scalar(
    value: bool, keepdims: int, noop: int, axes: tuple[int, ...] | None, opset: int
) -> None:
    _assert_mlx_result(
        *_model(
            np.array(value, dtype=np.bool_),
            axes,
            keepdims=keepdims,
            noop=noop,
            opset=opset,
        )
    )


@pytest.mark.parametrize("opset", [1, 11, 12, 13, 17, 18, 20])
@pytest.mark.parametrize("keepdims", [0, 1])
def test_reduce_max_numeric_schemas(opset: int, keepdims: int) -> None:
    x = np.array([[1, -2, 3], [-4, -5, -6]], dtype=np.float32)
    expected = np.max(x, axis=-1, keepdims=bool(keepdims))
    inputs = [m.tensor("x", DT.FLOAT, [2, 3])]
    feeds = {"x": x}
    attributes = {"keepdims": keepdims}
    if opset < 18:
        attributes["axes"] = [-1]
    else:
        inputs.append(m.tensor("axes", DT.INT64, [1]))
        feeds["axes"] = np.array([-1], dtype=np.int64)
    model = m.make_model(
        "ReduceMax",
        inputs,
        [m.tensor("out", DT.FLOAT, list(expected.shape))],
        attributes=attributes,
        opset=opset,
    )
    onnx.checker.check_model(onnx.load_from_string(model))
    _assert_mlx_result(model, feeds, expected)
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape,axes_form",
    [
        (shape, axes_form)
        for shape in [(), (0,), (2, 3), (2, 0, 3)]
        for axes_form in ["omitted", "empty", "last"]
        if shape or axes_form != "last"
    ],
)
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("opset", [20, 23, 26])
def test_reduce_min_bool(
    shape: tuple[int, ...], keepdims: int, opset: int, axes_form: str
) -> None:
    x = np.asarray(np.arange(np.prod(shape), dtype=np.int64).reshape(shape) % 3 != 0)
    axes = None if axes_form != "last" else (-1,)
    expected = np.all(x, axis=axes, keepdims=bool(keepdims))
    inputs = [m.tensor("x", DT.BOOL, list(shape))]
    feeds = {"x": x}
    if axes_form != "omitted":
        values = np.asarray(axes or [], dtype=np.int64)
        inputs.append(m.tensor("axes", DT.INT64, [len(values)]))
        feeds["axes"] = values
    model = m.make_model(
        "ReduceMin",
        inputs,
        [m.tensor("out", DT.BOOL, list(expected.shape))],
        attributes={"keepdims": keepdims},
        opset=opset,
    )
    _assert_mlx_result(model, feeds, expected)
