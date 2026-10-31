"""Frozen blocked lookup-4 products retain ordinary ONNX semantics and outputs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper


def _model(
    *,
    transpose=False,
    shared=False,
    live=False,
    low_first=False,
    rank=2,
    k=19,
    n=23,
    block=64,
):
    blocks = (k * n + block - 1) // block
    rng = np.random.default_rng(23)
    packed = rng.integers(0, 256, blocks * block // 2, dtype=np.uint8)
    scales = rng.uniform(0.01, 0.1, blocks).astype(np.float32)
    book = np.linspace(-1, 1, 16, dtype=np.float32) ** 3
    values = {
        "packed": packed,
        "scales": scales,
        "book": book,
        "sixteen": np.array(16, np.int64),
        "axes": np.array([1], np.int64),
        "blocks": np.array([-1, block], np.int64),
        "flat": np.array([-1], np.int64),
        "start": np.array([0], np.int64),
        "end": np.array([k * n], np.int64),
        "axis": np.array([0], np.int64),
        "matrix": np.array([k, n], np.int64),
    }
    node = helper.make_node
    nodes = [
        node("Cast", ["packed"], ["integer"], to=TensorProto.INT64),
        node("Div", ["integer", "sixteen"], ["high"]),
        node("Mod", ["integer", "sixteen"], ["low"]),
        node("Unsqueeze", ["high", "axes"], ["hi"]),
        node("Unsqueeze", ["low", "axes"], ["lo"]),
        node("Concat", ["lo", "hi"] if low_first else ["hi", "lo"], ["pairs"], axis=1),
        node("Reshape", ["pairs", "blocks"], ["indices"]),
        node("Gather", ["book", "indices"], ["decoded"], axis=0),
        node("Unsqueeze", ["scales", "axes"], ["scale_column"]),
        node("Mul", ["decoded", "scale_column"], ["scaled"]),
        node("Reshape", ["scaled", "flat"], ["flattened"]),
        node("Slice", ["flattened", "start", "end", "axis"], ["trimmed"]),
        node("Reshape", ["trimmed", "matrix"], ["weight"]),
    ]
    if transpose:
        nodes.append(node("Transpose", ["weight"], ["weight_t"], perm=[1, 0]))
    nodes.append(
        node(
            "MatMul",
            ["x", "weight_t" if transpose else "weight"],
            ["y"],
            name="product",
        )
    )
    inner, outer = (n, k) if transpose else (k, n)
    prefix = [] if rank == 1 else ["m"] if rank == 2 else [2, "m"]
    info = helper.make_tensor_value_info
    inputs = [info("x", TensorProto.FLOAT, [*prefix, inner])]
    if live:
        inputs.extend(
            info(
                name,
                helper.np_dtype_to_tensor_dtype(values[name].dtype),
                values[name].shape,
            )
            for name in ("packed", "scales")
        )
    outputs = [info("y", TensorProto.FLOAT, [*prefix, outer])]
    if shared:
        nodes.append(node("Add", ["weight", "weight"], ["shared"]))
        outputs.extend(
            [
                info("weight", TensorProto.FLOAT, [k, n]),
                info("shared", TensorProto.FLOAT, [k, n]),
            ]
        )
    initializers = [
        numpy_helper.from_array(v, name)
        for name, v in values.items()
        if not live or name not in {"packed", "scales"}
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "lookup4", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString(), packed, scales, book, (k, n)


def _session(model, provider):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    if provider == "MLXExecutionProvider":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(model, options, providers=[provider])


def _reference(packed, scales, book, shape, low_first=False):
    codes = np.stack(
        [packed & 15, packed >> 4] if low_first else [packed >> 4, packed & 15], axis=1
    ).reshape(-1)
    decoded = book[codes].reshape(len(scales), -1) * scales[:, None]
    return decoded.reshape(-1)[: np.prod(shape)].reshape(shape)


@pytest.mark.parametrize("transpose", [False, True], ids=["forward", "transpose"])
@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("shared", [False, True])
def test_lut4_session_replay(transpose, rank, shared):
    model, p, s, book, shape = _model(transpose=transpose, rank=rank, shared=shared)
    runtime = _session(model, "MLXExecutionProvider")
    cpu = _session(model, "CPUExecutionProvider")
    weight = _reference(p, s, book, shape)
    rng = np.random.default_rng(17)
    for m in [1] if rank == 1 else [1, 17, 0, 3, 1]:
        prefix = () if rank == 1 else (m,) if rank == 2 else (2, m)
        x = rng.normal(0, 0.1, (*prefix, shape[1] if transpose else shape[0])).astype(
            np.float32
        )
        actual = runtime.run(None, {"x": x})
        expected = x @ (weight.T if transpose else weight)
        np.testing.assert_allclose(actual[0], expected, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(
            actual[0], cpu.run(None, {"x": x})[0], rtol=1e-4, atol=1e-6
        )
        if shared:
            np.testing.assert_array_equal(actual[1], weight)
            np.testing.assert_array_equal(actual[2], weight + weight)


@pytest.mark.parametrize("low_first,live", [(True, False), (False, True)])
def test_lut4_unmatched_and_live_parameters(low_first, live):
    model, p, s, book, shape = _model(low_first=low_first, live=live)
    runtime = _session(model, "MLXExecutionProvider")
    x = np.ones((3, shape[0]), np.float32)
    for _ in range(3):
        feeds = {"x": x, **({"packed": p, "scales": s} if live else {})}
        expected = x @ _reference(p, s, book, shape, low_first)
        np.testing.assert_allclose(
            runtime.run(None, feeds)[0], expected, rtol=1e-4, atol=1e-6
        )
        if live:
            p = (p // 2).copy()
            s = (s * np.float32(0.5)).copy()


def test_lut4_dispatch_probe():
    model, p, s, book, shape = _model(transpose=True)
    runtime = _session(model, "MLXExecutionProvider")
    for m in [1, 3]:
        x = np.full((m, shape[1]), 0.1, np.float32)
        np.testing.assert_allclose(
            runtime.run(None, {"x": x})[0],
            x @ _reference(p, s, book, shape).T,
            rtol=1e-4,
            atol=1e-6,
        )


def test_lut4_actual_dispatch(tmp_path):
    for disabled in [False, True]:
        trace = tmp_path / f"lookup-{disabled}.json"
        env = {**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)}
        if disabled:
            env["ONNXRUNTIME_EP_MLX_NO_LUT4_FUSION"] = "1"
        else:
            env.pop("ONNXRUNTIME_EP_MLX_NO_LUT4_FUSION", None)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{Path(__file__)}::test_lut4_composed_dispatch_probe",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            env=env,
            check=True,
            timeout=180,
        )
        events = json.loads(trace.read_text())
        kernels = {
            e.get("args", {}).get("kernel") for e in events if e.get("cat") == "op.fast"
        }
        assert ("mlx_lookup4_matmul" in kernels) is not disabled, kernels
        if not disabled:
            assert (
                sum(
                    e.get("args", {}).get("kernel") == "mlx_lookup4_matmul"
                    for e in events
                )
                == 6
            )
        if not disabled:
            calls = [e for e in events if e["name"].startswith("mlx.compute[")]
            assert calls and all(e["args"]["nodes"] == 1 for e in calls)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_lut4_nonfinite_input_classification(value):
    model, p, s, book, shape = _model()
    runtime = _session(model, "MLXExecutionProvider")
    x = np.zeros((1, shape[0]), np.float32)
    x[0, 0] = value
    with np.errstate(invalid="ignore"):
        expected = x @ _reference(p, s, book, shape)
    np.testing.assert_allclose(
        runtime.run(None, {"x": x})[0], expected, rtol=1e-4, atol=1e-6, equal_nan=True
    )


def test_lut4_overridable_initializers():
    blob, p, s, book, shape = _model()
    model = onnx.load_from_string(blob)
    for name, value in [("packed", p), ("scales", s)]:
        model.graph.input.append(
            helper.make_tensor_value_info(
                name, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
            )
        )
    runtime = _session(model.SerializeToString(), "MLXExecutionProvider")
    x = np.full((2, shape[0]), 0.1, np.float32)
    for _ in range(3):
        expected = x @ _reference(p, s, book, shape)
        np.testing.assert_allclose(
            runtime.run(None, {"x": x, "packed": p, "scales": s})[0],
            expected,
            rtol=1e-4,
            atol=1e-6,
        )
        p = (p // 2).copy()
        s = (s * np.float32(0.5)).copy()


def test_lut4_nested_outputs_and_branch_weights():
    blob, p, s, book, shape = _model()
    source = onnx.load_from_string(blob)
    then = onnx.GraphProto()
    then.CopyFrom(source.graph)
    then.name = "then"
    then.ClearField("input")
    then.output.append(
        helper.make_tensor_value_info("weight", TensorProto.FLOAT, list(shape))
    )
    otherwise = onnx.GraphProto()
    otherwise.CopyFrom(then)
    otherwise.name = "else"
    alternative = np.bitwise_xor(p, np.uint8(15))
    for value in otherwise.initializer:
        if value.name == "packed":
            value.CopyFrom(numpy_helper.from_array(alternative, "packed"))
    info = helper.make_tensor_value_info
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "If",
                    ["condition"],
                    ["y", "w"],
                    then_branch=then,
                    else_branch=otherwise,
                )
            ],
            "nested_lookup",
            [
                info("condition", TensorProto.BOOL, []),
                info("x", TensorProto.FLOAT, ["m", shape[0]]),
            ],
            [
                info("y", TensorProto.FLOAT, ["m", shape[1]]),
                info("w", TensorProto.FLOAT, list(shape)),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.checker.check_model(model, full_check=True)
    runtime = _session(model.SerializeToString(), "MLXExecutionProvider")
    for condition, m in [(True, 1), (False, 3), (True, 2), (False, 1)]:
        x = np.full((m, shape[0]), 0.1, np.float32)
        expected = _reference(p if condition else alternative, s, book, shape)
        y, w = runtime.run(None, {"x": x, "condition": np.array(condition)})
        np.testing.assert_allclose(y, x @ expected, rtol=1e-4, atol=1e-6)
        np.testing.assert_array_equal(w, expected)


def test_lut4_composed_dispatch_probe():
    nodes, initializers, weights = [], [], []
    previous = "x"
    for layer in range(3):
        blob, p, s, book, shape = _model(k=23, n=23)
        original = onnx.load_from_string(blob)
        p = np.bitwise_xor(p, np.uint8(layer * 17))

        def rename(name, layer=layer, previous=previous):
            return previous if name == "x" else f"layer{layer}_{name}"

        for value in original.graph.initializer:
            if value.name == "packed":
                value.CopyFrom(numpy_helper.from_array(p, "packed"))
            value.name = rename(value.name)
            initializers.append(value)
        for i, node in enumerate(original.graph.node):
            node.name = f"layer{layer}_{node.op_type}_{i}"
            node.input[:] = [rename(n) for n in node.input]
            node.output[:] = [rename(n) for n in node.output]
            nodes.append(node)
        previous = f"layer{layer}_y"
        weights.append(_reference(p, s, book, shape))
        if layer < 2:
            nodes.append(
                helper.make_node("Tanh", [previous], [f"layer{layer}_activation"])
            )
            previous = f"layer{layer}_activation"
    info = helper.make_tensor_value_info
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "composed_lookup",
            [info("x", TensorProto.FLOAT, ["m", 23])],
            [info(previous, TensorProto.FLOAT, ["m", 23])],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.checker.check_model(model, full_check=True)
    runtime = _session(model.SerializeToString(), "MLXExecutionProvider")
    for m in [1, 3]:
        x = np.full((m, 23), 0.1, np.float32)
        expected = x
        for i, weight in enumerate(weights):
            expected = expected @ weight
            if i < 2:
                expected = np.tanh(expected)
        np.testing.assert_allclose(
            runtime.run(None, {"x": x})[0], expected, rtol=1e-4, atol=1e-6
        )
