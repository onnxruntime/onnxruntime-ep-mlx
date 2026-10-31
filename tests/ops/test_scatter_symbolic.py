"""ScatterElements symbolic metadata and partial non-axis geometry."""

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

import _models as m


def _native(model):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model, options, providers=["MLXExecutionProvider"], enable_fallback=False
    )
    session.disable_fallback()
    return session


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("rows", [0, 1, 2])
def test_scatter_partial_nonaxis_region(dtype, rows):
    dt = ir.DataType.FLOAT if dtype == np.float32 else ir.DataType.FLOAT16
    model = m.make_model(
        "ScatterElements",
        [
            m.tensor("data", dt, [3, 4]),
            m.tensor("indices", ir.DataType.INT64, [rows, 2]),
            m.tensor("updates", dt, [rows, 2]),
        ],
        [m.tensor("out", dt, [3, 4])],
        attributes={"axis": -1},
    )
    session = _native(model)
    data = np.arange(12, dtype=dtype).reshape(3, 4)
    indices = np.tile(np.array([0, -1], np.int64), (rows, 1))
    updates = np.arange(rows * 2, dtype=dtype).reshape(rows, 2) + dtype(100)
    expected = data.copy()
    for row in range(rows):
        for col in range(2):
            expected[row, indices[row, col]] = updates[row, col]
    actual = session.run(None, {"data": data, "indices": indices, "updates": updates})[
        0
    ]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(data, np.arange(12, dtype=dtype).reshape(3, 4))


def test_scatter_symbolic_updates_metadata_and_runtime_checks():
    model = m.make_model(
        "ScatterElements",
        [
            m.tensor("data", ir.DataType.FLOAT, ["b", 4, "s", 6]),
            m.tensor("indices", ir.DataType.INT64, ["b", 4, "s", 1]),
            m.tensor("updates", ir.DataType.FLOAT, ["ub", "uh", "us", "uw"]),
        ],
        [m.tensor("out", ir.DataType.FLOAT, ["b", 4, "s", 6])],
        attributes={"axis": -1},
    )
    session = _native(model)
    for batch, length in [(1, 3), (2, 5), (1, 3)]:
        data = np.arange(batch * 4 * length * 6, dtype=np.float32).reshape(
            batch, 4, length, 6
        )
        indices = np.full((batch, 4, length, 1), -1, np.int64)
        updates = np.full(indices.shape, 1000, np.float32)
        feeds = {"data": data, "indices": indices, "updates": updates}
        expected = data.copy()
        expected[..., -1] = 1000
        np.testing.assert_array_equal(session.run(None, feeds)[0], expected)
        with pytest.raises(
            ort.capi.onnxruntime_pybind11_state.EPFail, match="ScatterElements"
        ):
            session.run(None, {**feeds, "updates": updates[:, :1]})
        bad_indices = np.full((batch, 4, length + 1, 1), -1, np.int64)
        with pytest.raises(
            ort.capi.onnxruntime_pybind11_state.EPFail, match="ScatterElements"
        ):
            session.run(
                None,
                {
                    **feeds,
                    "indices": bad_indices,
                    "updates": np.ones(bad_indices.shape, np.float32),
                },
            )
        np.testing.assert_array_equal(session.run(None, feeds)[0], expected)
