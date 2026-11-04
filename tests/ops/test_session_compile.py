"""Independent compiled/eager policies for sessions sharing one MLX process."""

import json
import os
import subprocess
import sys

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

KEY = "ep.mlx.enable_compile"


def model_bytes(prefix="policy"):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["rows", 8])
    tape = ir.tape.Tape()
    offset = tape.initializer(ir.tensor(np.float32(0.1), name="offset"))
    scale = tape.initializer(ir.tensor(np.float32(0.7), name="scale"))
    y = tape.op("Tanh", [tape.op("Mul", [tape.op("Add", [x, offset]), scale])])
    y.name, y.dtype, y.shape = "y", ir.DataType.FLOAT, ir.Shape(["rows", 8])
    for index, node in enumerate(tape.nodes):
        node.name = f"{prefix}.{index}"
    model = ir.Model(
        ir.Graph(
            [x],
            [y],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name=prefix,
        ),
        ir_version=10,
    )
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto, full_check=True)
    return proto.SerializeToString()


def runtime(value, prefix="policy"):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if value is not None:
        options.add_session_config_entry(KEY, value)
    result = ort.InferenceSession(
        model_bytes(prefix), options, providers=["MLXExecutionProvider"], enable_fallback=False
    )
    result.disable_fallback()
    return result


@pytest.mark.parametrize("value", ["", "true", "false", "2", "-1", " 1", "01"])
def test_invalid_session_compile_option(value):
    with pytest.raises(RuntimeError, match=r"INVALID_ARGUMENT.*ep.mlx.enable_compile.*0 or 1"):
        runtime(value)


@pytest.mark.parametrize("global_disable", [False, True])
def test_interleaved_sessions_preserve_policy_and_live_shapes(tmp_path, global_disable):
    trace = tmp_path / "trace.json"
    environment = {
        **os.environ,
        "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
        "ONNXRUNTIME_EP_MLX_NO_COMPILE": "1" if global_disable else "0",
        "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0",
    }
    subprocess.run(
        [sys.executable, __file__, "--probe"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    with open(trace) as stream:
        events = json.load(stream)
    paths = {name: [] for name in ("eager", "compiled", "default")}
    for event in events:
        if event.get("cat") != "ep.path":
            continue
        args = event["args"]
        for name, calls in paths.items():
            if args.get("node_names", "").startswith(name + "."):
                calls.append(args["path"])
    assert paths["eager"] and set(paths["eager"]) == {"eager"}
    for name in ("compiled", "default"):
        assert paths[name]
        if global_disable:
            assert set(paths[name]) == {"eager"}
        else:
            assert "general" in paths[name]


def probe():
    ort.register_execution_provider_library(
        "MLXExecutionProvider", os.environ["ONNXRUNTIME_MLX_EP_LIB"]
    )
    executors = [runtime("0", "eager"), runtime("1", "compiled"), runtime(None, "default")]
    for iteration, rows in enumerate((1, 3, 1, 3)):
        x = np.linspace(-1, 1, rows * 8, dtype=np.float32).reshape(rows, 8)
        x += np.float32(iteration * 0.13)
        for executor in executors:
            actual = executor.run(None, {"x": x})[0]
            np.testing.assert_allclose(actual, np.tanh((x + 0.1) * 0.7), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    probe()
