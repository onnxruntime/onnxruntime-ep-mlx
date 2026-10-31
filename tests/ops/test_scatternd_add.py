"""Native ScatterND additive updates, live geometry and checked indices."""

import _models as m
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest


def native(shape, index_shape, update_shape, dtype=ir.DataType.FLOAT):
    graph = m.make_model(
        "ScatterND",
        [
            m.tensor("data", dtype, shape),
            m.tensor("indices", ir.DataType.INT64, index_shape),
            m.tensor("updates", dtype, update_shape),
        ],
        [m.tensor("out", dtype, shape)],
        attributes={"reduction": "add"},
    )
    if dtype == ir.DataType.BFLOAT16:
        model = onnx.load_model_from_string(graph)
        scatter = model.graph.node[0]
        scatter.input[0], scatter.input[2] = "data_bf16", "updates_bf16"
        scatter.output[0] = "out_bf16"
        for index in (0, 2):
            model.graph.input[index].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        model.graph.output[0].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        prefix = [
            onnx.helper.make_node(
                "Cast", [name], [name + "_bf16"], to=onnx.TensorProto.BFLOAT16
            )
            for name in ("data", "updates")
        ]
        model.graph.node.extend(
            [
                onnx.helper.make_node(
                    "Cast", ["out_bf16"], ["out"], to=onnx.TensorProto.FLOAT
                )
            ]
        )
        for node in reversed(prefix):
            model.graph.node.insert(0, node)
        onnx.checker.check_model(model, full_check=True)
        graph = model.SerializeToString()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    executor = ort.InferenceSession(
        graph, options, providers=["MLXExecutionProvider"], enable_fallback=False
    )
    executor.disable_fallback()
    return executor


def oracle(data, indices, updates):
    result = data.copy()
    for position in np.ndindex(indices.shape[:-1]):
        result[tuple(indices[position])] += updates[position]
    return result


@pytest.mark.parametrize(
    "dtype,dt", [(np.float32, ir.DataType.FLOAT), (np.float16, ir.DataType.FLOAT16)]
)
@pytest.mark.parametrize("width", [1, 2, 3])
def test_native_add_negative_and_duplicate_tuples(dtype, dt, width):
    shape = [2, 3, 4]
    indices = np.array([[0, 1, 2], [-1, -1, -1], [0, 1, 2]], np.int64)[:, :width].copy()
    updates = np.arange(3 * np.prod(shape[width:]), dtype=dtype).reshape(
        [3, *shape[width:]]
    )
    data = np.arange(24, dtype=dtype).reshape(shape)
    original = data.copy()
    result = native(shape, list(indices.shape), list(updates.shape), dt).run(
        None, {"data": data, "indices": indices, "updates": updates}
    )[0]
    np.testing.assert_array_equal(result, oracle(data, indices, updates))
    np.testing.assert_array_equal(data, original)


def test_symbolic_add_reuses_session_and_recovers_after_invalid_inputs():
    executor = native(["a", "b", "c"], ["n", 2], ["n", "c"])
    for shape in [(2, 3, 4), (3, 2, 2), (2, 3, 4)]:
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        indices = np.array([[0, 0], [-1, -1], [0, 0]], np.int64)
        updates = np.ones((3, shape[-1]), np.float32)
        feeds = {"data": data, "indices": indices, "updates": updates}
        expected = oracle(data, indices, updates)
        np.testing.assert_array_equal(executor.run(None, feeds)[0], expected)
        for bad in (shape[0], -shape[0] - 1, np.iinfo(np.int64).min):
            invalid = indices.copy()
            invalid[0, 0] = bad
            with pytest.raises(
                ort.capi.onnxruntime_pybind11_state.EPFail, match="ScatterND"
            ):
                executor.run(None, {**feeds, "indices": invalid})
        with pytest.raises(
            ort.capi.onnxruntime_pybind11_state.EPFail, match="ScatterND"
        ):
            executor.run(
                None, {**feeds, "updates": np.ones((2, shape[-1]), np.float32)}
            )
        np.testing.assert_array_equal(executor.run(None, feeds)[0], expected)


@pytest.mark.parametrize(
    "shape,count", [((2, 3, 4), 0), ((0, 3, 4), 0), ((2, 3, 0), 2)]
)
def test_symbolic_empty_updates_and_empty_data(shape, count):
    executor = native(["a", "b", "c"], ["n", 2], ["n", "c"])
    data = np.zeros(shape, np.float32)
    indices = np.zeros((count, 2), np.int64)
    updates = np.zeros((count, shape[-1]), np.float32)
    np.testing.assert_array_equal(
        executor.run(None, {"data": data, "indices": indices, "updates": updates})[0],
        data,
    )


def test_single_tuple_with_scalar_update():
    executor = native([2, 3], [2], [])
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    indices = np.array([-1, 1], np.int64)
    updates = np.array(10, np.float32)
    np.testing.assert_array_equal(
        executor.run(None, {"data": data, "indices": indices, "updates": updates})[0],
        oracle(data, indices, updates),
    )


def test_add_bfloat16_payload_with_float32_runtime_boundary():
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    indices = np.array([[0], [-1], [0]], np.int64)
    updates = np.ones((3, 4), np.float32)
    executor = native([3, 4], [3, 1], [3, 4], ir.DataType.BFLOAT16)
    actual = executor.run(None, {"data": data, "indices": indices, "updates": updates})[
        0
    ]
    np.testing.assert_array_equal(actual, oracle(data, indices, updates))
