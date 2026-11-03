"""Overwrite ScatterND with symbolic extents and changing runtime geometry."""

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from test_scatternd_add import native


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
@pytest.mark.parametrize("width", [1, 2])
def test_symbolic_overwrite_changes_shapes_and_indices(dtype, width):
    dt = np.float16 if dtype == ir.DataType.FLOAT16 else np.float32
    executor = native(
        ["rows", "cols"],
        ["count", width],
        ["count", "cols"] if width == 1 else ["count"],
        dtype,
        reduction="none",
    )
    for rows, cols, count in [(3, 4, 2), (5, 2, 1), (3, 4, 0), (3, 4, 2)]:
        data = np.arange(rows * cols, dtype=dt).reshape(rows, cols)
        for reverse in (False, True):
            indices = np.array([[0, 0], [-1, -1]], np.int64)[:count, :width].copy()
            if reverse:
                indices = indices[::-1].copy()
            shape = [count, cols] if width == 1 else [count]
            updates = np.arange(np.prod(shape), dtype=dt).reshape(shape) + dt(50)
            expected = data.copy()
            for position in range(count):
                expected[tuple(indices[position])] = updates[position]
            actual = executor.run(
                None, {"data": data, "indices": indices, "updates": updates}
            )[0]
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(
                data, np.arange(rows * cols, dtype=dt).reshape(rows, cols)
            )


def test_symbolic_overwrite_rejects_invalid_updates_and_recovers():
    executor = native(["rows", "cols"], ["count", 1], ["u", "v"], reduction="none")
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    indices = np.array([[0], [2]], np.int64)
    updates = np.full((2, 4), 100, np.float32)
    feeds = {"data": data, "indices": indices, "updates": updates}
    expected = data.copy()
    expected[[0, 2]] = updates
    np.testing.assert_array_equal(executor.run(None, feeds)[0], expected)
    with pytest.raises(
        ort.capi.onnxruntime_pybind11_state.EPFail, match="ScatterND: updates shape"
    ):
        executor.run(None, {**feeds, "updates": updates[:, :1]})
    np.testing.assert_array_equal(executor.run(None, feeds)[0], expected)
