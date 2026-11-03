"""Small shape programs preserve geometry and live payload behavior."""

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest


@pytest.mark.parametrize("gpu_shapes", [False, True])
@pytest.mark.parametrize("eager", [False, True])
def test_dynamic_broadcast_axes_and_live_nonzero(gpu_shapes, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    if gpu_shapes:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_GPU_SHAPE_OPS", "1")
    else:
        monkeypatch.delenv("ONNXRUNTIME_EP_MLX_GPU_SHAPE_OPS", raising=False)
    target = ir.val("target", dtype=ir.DataType.FLOAT, shape=["target_batch", 3])
    gradient = ir.val("gradient", dtype=ir.DataType.FLOAT, shape=["gradient_batch", 3])
    tape = ir.tape.Tape()
    constant = lambda x: tape.op(
        "Constant", [], {"value": ir.tensor(np.asarray(x, np.int64))}
    )
    shape = tape.op("Shape", [target])
    axes = tape.op(
        "Squeeze",
        [tape.op("NonZero", [tape.op("Equal", [shape, constant(1)])]), constant([0])],
    )
    axes = tape.op("Concat", [axes, constant([-1])], {"axis": 0})
    expanded = tape.op("Unsqueeze", [gradient, constant([-1])])
    reduced = tape.op("ReduceSum", [expanded, axes], {"keepdims": 1})
    result = tape.op("Reshape", [reduced, shape])
    result.name, result.dtype, result.shape = "result", ir.DataType.FLOAT, target.shape
    live = tape.op("NonZero", [gradient])
    live.name, live.dtype, live.shape = "live", ir.DataType.INT64, ir.Shape([2, None])
    model = ir.Model(
        ir.Graph(
            [target, gradient],
            [result, live],
            nodes=tape.nodes,
            opset_imports={"": 24},
            name="shape_programs",
        ),
        ir_version=11,
    )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    executor = ort.InferenceSession(
        ir.to_proto(model).SerializeToString(),
        options,
        providers=["MLXExecutionProvider"],
    )
    executor.disable_fallback()
    for target_rows, source_rows in [(1, 5), (3, 3), (0, 0), (1, 0), (1, 1), (1, 5)]:
        for offset in (0, 1):
            values = np.arange(source_rows * 3, dtype=np.float32).reshape(
                source_rows, 3
            )
            values = np.where((values + offset) % 2 == 0, values + 1, 0).astype(
                np.float32
            )
            actual, indices = executor.run(
                None,
                {"target": np.zeros((target_rows, 3), np.float32), "gradient": values},
            )
            expected = values.sum(axis=0, keepdims=True) if target_rows == 1 else values
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(
                indices, np.array(np.nonzero(values), np.int64)
            )
