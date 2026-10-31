"""Eager outputs and views own their arrays after translation temporaries are released."""

import _models as m
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort


def chain_model(*, layers=32):
    x = m.tensor("x", ir.DataType.FLOAT, ["rows", "columns"])
    tape = ir.tape.Tape()
    offset = tape.initializer(ir.tensor(np.array(0.01, np.float32), name="offset"))
    value = x
    for index in range(layers):
        value = tape.op("Tanh", [tape.op("Add", [value, offset])])
        if index == 3:
            checkpoint = value
            checkpoint.name = "checkpoint"
    value.name = "last"
    transposed = tape.op("Transpose", [value], {"perm": [1, 0]})
    transposed.name = "transposed"
    for output in (checkpoint, value):
        output.dtype, output.shape = ir.DataType.FLOAT, ir.Shape(["rows", "columns"])
    transposed.dtype, transposed.shape = (
        ir.DataType.FLOAT,
        ir.Shape(["columns", "rows"]),
    )
    model = ir.Model(
        ir.Graph(
            [x],
            [checkpoint, value, transposed],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="eager_chain",
        ),
        ir_version=10,
    )
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto, full_check=True)
    return proto.SerializeToString()


def test_eager_roots_keep_intermediate_outputs_and_views_alive(monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    runtime = ort.InferenceSession(
        chain_model(),
        options,
        providers=["MLXExecutionProvider"],
        enable_fallback=False,
    )
    runtime.disable_fallback()
    for rows, columns in ((2, 5), (3, 7), (2, 5)):
        x = np.linspace(-0.2, 0.2, rows * columns, dtype=np.float32).reshape(
            rows, columns
        )
        expected = x.copy()
        for index in range(32):
            expected = np.tanh(expected + np.float32(0.01))
            if index == 3:
                checkpoint = expected.copy()
        before = x.copy()
        actual = runtime.run(None, {"x": x})
        for value, wanted in zip(
            actual, (checkpoint, expected, expected.T), strict=True
        ):
            np.testing.assert_allclose(value, wanted, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(x, before)


def measure(library):
    """Measure this fixture in a fresh process using the EP's linked MLX allocator."""
    import ctypes
    import hashlib
    import os
    from pathlib import Path

    os.environ["ONNXRUNTIME_EP_MLX_NO_COMPILE"] = "1"
    ort.register_execution_provider_library("MLXExecutionProvider", library)
    api = ctypes.CDLL(library)
    api.mlx_get_peak_memory.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
    api.mlx_get_peak_memory.restype = ctypes.c_int
    api.mlx_reset_peak_memory.argtypes = []
    api.mlx_reset_peak_memory.restype = ctypes.c_int
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    graph = chain_model(layers=48)
    runtime = ort.InferenceSession(
        graph, options, providers=["MLXExecutionProvider"], enable_fallback=False
    )
    runtime.disable_fallback()
    x = np.linspace(-0.2, 0.2, 1024 * 1024, dtype=np.float32).reshape(1024, 1024)
    if api.mlx_reset_peak_memory() != 0:
        raise RuntimeError("Unable to reset the MLX allocator peak")
    actual = runtime.run(None, {"x": x})
    peak = ctypes.c_size_t()
    if api.mlx_get_peak_memory(ctypes.byref(peak)) != 0:
        raise RuntimeError("Unable to read the MLX allocator peak")
    expected = x.copy()
    for index in range(48):
        expected = np.tanh(expected + np.float32(0.01))
        if index == 3:
            checkpoint = expected.copy()
    for a, b in zip(actual, (checkpoint, expected, expected.T), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    return {
        "library": library,
        "library_sha256": hashlib.sha256(Path(library).read_bytes()).hexdigest(),
        "graph_sha256": hashlib.sha256(graph).hexdigest(),
        "layers": 48,
        "nodes": 97,
        "shape": list(x.shape),
        "input_bytes": x.nbytes,
        "checked_outputs": len(actual),
        "peak_mlx_active_bytes": peak.value,
        "scope": "Fresh process; one eager ORT execution; MLX allocator peak reset after session creation, before run; cache and CPU allocations excluded",
    }


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = measure(args.library)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
