"""Keep supported eager/compiled branches native when fusions interleave."""

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

import _models as m


def test_interleaved_eager_regions_preserve_native_coverage():
    x = m.tensor("x", ir.DataType.INT64, [1])
    bounds = [
        m.tensor(name, ir.DataType.INT64, []) for name in ("start", "limit", "delta")
    ]
    tape = ir.tape.Tape()
    zero = tape.initializer(ir.tensor(np.array(0, np.int64), name="zero"))
    a = tape.op("Add", [x, zero])
    repeats = tape.op("Range", bounds)
    repeats.dtype, repeats.shape = ir.DataType.INT64, ir.Shape([1])
    added = tape.op("Add", [a, repeats])
    added.name, added.dtype, added.shape = "added", ir.DataType.INT64, ir.Shape([1])
    tiled = tape.op("Tile", [a, repeats])
    tiled.name, tiled.dtype, tiled.shape = (
        "tiled",
        ir.DataType.INT64,
        ir.Shape(["copies"]),
    )
    model = ir.Model(
        ir.Graph(
            [x, *bounds],
            [added, tiled],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 24},
            name="interleaved",
        ),
        ir_version=11,
    )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        ir.to_proto(model).SerializeToString(),
        options,
        providers=["MLXExecutionProvider"],
        enable_fallback=False,
    )
    session.disable_fallback()
    for repeat in (2, 3, 0, 1, 2):
        values = np.array([10 + repeat], np.int64)
        feeds = {
            "x": values,
            "start": np.array(repeat, np.int64),
            "limit": np.array(repeat + 1, np.int64),
            "delta": np.array(1, np.int64),
        }
        actual = session.run(None, feeds)
        np.testing.assert_array_equal(actual[0], values + repeat)
        np.testing.assert_array_equal(actual[1], np.tile(values, repeat))
