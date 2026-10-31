"""Frozen blocked lookup-4 products retain ordinary ONNX semantics and outputs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import _models as models
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest


def _buffers(k, n, block):
    blocks = (k * n + block - 1) // block
    rng = np.random.default_rng(23)
    return {
        "packed": rng.integers(0, 256, blocks * block // 2, dtype=np.uint8),
        "scales": rng.uniform(0.01, 0.1, blocks).astype(np.float32),
        "book": np.linspace(-1, 1, 16, dtype=np.float32) ** 3,
        "sixteen": np.array(16, np.int64),
        "axes": np.array([1], np.int64),
        "blocks": np.array([-1, block], np.int64),
        "flat": np.array([-1], np.int64),
        "start": np.array([0], np.int64),
        "end": np.array([k * n], np.int64),
        "axis": np.array([0], np.int64),
        "matrix": np.array([k, n], np.int64),
    }


def _lookup(
    tape, x, buffers, *, prefix="", transpose=False, low_first=False, live=False
):
    values, inputs = {}, []
    for name, array in buffers.items():
        tensor = ir.tensor(array, name=prefix + name)
        if live and name in {"packed", "scales"}:
            values[name] = models.tensor(prefix + name, tensor.dtype, array.shape)
            inputs.append(values[name])
        else:
            values[name] = tape.initializer(tensor)
    integer = tape.op("Cast", [values["packed"]], {"to": int(ir.DataType.INT64)})
    high = tape.op("Div", [integer, values["sixteen"]])
    low = tape.op("Mod", [integer, values["sixteen"]])
    hi, lo = [tape.op("Unsqueeze", [v, values["axes"]]) for v in (high, low)]
    pairs = tape.op("Concat", [lo, hi] if low_first else [hi, lo], {"axis": 1})
    indices = tape.op("Reshape", [pairs, values["blocks"]])
    decoded = tape.op("Gather", [values["book"], indices], {"axis": 0})
    scale = tape.op("Unsqueeze", [values["scales"], values["axes"]])
    scaled = tape.op("Mul", [decoded, scale])
    flat = tape.op("Reshape", [scaled, values["flat"]])
    trimmed = tape.op("Slice", [flat, values["start"], values["end"], values["axis"]])
    weight = tape.op("Reshape", [trimmed, values["matrix"]])
    k, n = buffers["matrix"].tolist()
    weight.name, weight.dtype, weight.shape = (
        prefix + "weight",
        ir.DataType.FLOAT,
        ir.Shape([k, n]),
    )
    right = tape.op("Transpose", [weight], {"perm": [1, 0]}) if transpose else weight
    output = tape.op("MatMul", [x, right])
    tape.nodes[-1].name = prefix + "product"
    output.name, output.dtype = prefix + "y", ir.DataType.FLOAT
    output.shape = ir.Shape([*x.shape[:-1], k if transpose else n])
    return output, weight, inputs


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
    buffers = _buffers(k, n, block)
    inner = n if transpose else k
    prefix = [] if rank == 1 else ["m"] if rank == 2 else [2, "m"]
    x = models.tensor("x", ir.DataType.FLOAT, [*prefix, inner])
    tape = ir.tape.Tape()
    output, weight, inputs = _lookup(
        tape, x, buffers, transpose=transpose, low_first=low_first, live=live
    )
    outputs = [output]
    if shared:
        added = tape.op("Add", [weight, weight])
        added.name, added.dtype, added.shape = "shared", weight.dtype, weight.shape
        outputs.extend([weight, added])
    model = ir.Model(
        ir.Graph(
            [x, *inputs],
            outputs,
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="lookup4",
        ),
        ir_version=10,
    )
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    return model, buffers["packed"], buffers["scales"], buffers["book"], (k, n)


def _session(model, provider):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    if provider == "MLXExecutionProvider":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(
        ir.to_proto(model).SerializeToString(), options, providers=[provider]
    )


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
    model, p, s, book, shape = _model()
    for name in ("packed", "scales"):
        model.graph.inputs.append(model.graph.initializers[name])
    runtime = _session(model, "MLXExecutionProvider")
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
    shape = (19, 23)
    buffers = _buffers(*shape, 64)
    p, s, book = (buffers[name] for name in ("packed", "scales", "book"))
    alternative = np.bitwise_xor(p, np.uint8(15))
    condition = models.tensor("condition", ir.DataType.BOOL, [])
    x = models.tensor("x", ir.DataType.FLOAT, ["m", shape[0]])

    def branch(name, packed):
        tape = ir.tape.Tape()
        output, weight, _ = _lookup(tape, x, {**buffers, "packed": packed})
        return ir.Graph(
            [],
            [output, weight],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name=name,
        )

    y = models.tensor("y", ir.DataType.FLOAT, ["m", shape[1]])
    w = models.tensor("w", ir.DataType.FLOAT, list(shape))
    node = ir.node(
        "If",
        [condition],
        outputs=[y, w],
        attributes={
            "then_branch": branch("then", p),
            "else_branch": branch("else", alternative),
        },
    )
    model = ir.Model(
        ir.Graph(
            [condition, x],
            [y, w],
            nodes=[node],
            opset_imports={"": 18},
            name="nested_lookup",
        ),
        ir_version=10,
    )
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    runtime = _session(model, "MLXExecutionProvider")
    for condition, m in [(True, 1), (False, 3), (True, 2), (False, 1)]:
        x = np.full((m, shape[0]), 0.1, np.float32)
        expected = _reference(p if condition else alternative, s, book, shape)
        y, w = runtime.run(None, {"x": x, "condition": np.array(condition)})
        np.testing.assert_allclose(y, x @ expected, rtol=1e-4, atol=1e-6)
        np.testing.assert_array_equal(w, expected)


def test_lut4_composed_dispatch_probe():
    tape, weights = ir.tape.Tape(), []
    x = models.tensor("x", ir.DataType.FLOAT, ["m", 23])
    previous = x
    for layer in range(3):
        buffers = _buffers(23, 23, 64)
        buffers["packed"] = np.bitwise_xor(buffers["packed"], np.uint8(layer * 17))
        previous, _, _ = _lookup(tape, previous, buffers, prefix=f"layer{layer}_")
        weights.append(
            _reference(buffers["packed"], buffers["scales"], buffers["book"], (23, 23))
        )
        if layer < 2:
            previous = tape.op("Tanh", [previous])
            previous.name = f"layer{layer}_activation"
            previous.dtype, previous.shape = ir.DataType.FLOAT, ir.Shape(["m", 23])
    model = ir.Model(
        ir.Graph(
            [x],
            [previous],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="composed_lookup",
        ),
        ir_version=10,
    )
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    runtime = _session(model, "MLXExecutionProvider")
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
