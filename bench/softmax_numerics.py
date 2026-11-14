"""Reproduce arXiv:2610.31642 forward collapse and measure proposed fixes on Metal.

The ONNX candidates run through the real EP with CPU fallback disabled. The MLX
microbenchmarks isolate FP32 upcasting from the paper's normalizer rewrite,
measure an inference-only log1p formulation, and separate attention
normalization overhead from the cost of losing fused SDPA.
"""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import TypeVar

import mlx.core as mx
import numpy as np
import onnx_ir as ir
import onnxruntime as ort

DT = ir.DataType
DTYPES = {
    "fp32": (DT.FLOAT, mx.float32),
    "fp16": (DT.FLOAT16, mx.float16),
    "bf16": (DT.BFLOAT16, mx.bfloat16),
}
T = TypeVar("T")


def counterbalanced_order(
    variants: tuple[T, ...], case_index: int, *, reverse: bool = False
) -> tuple[T, ...]:
    """Rotate variants and reverse every rotation cycle to balance run order."""
    if not variants:
        raise ValueError(
            "Cannot counterbalance an empty variant list; provide at least one variant."
        )
    if case_index < 0:
        raise ValueError(
            f"Cannot counterbalance case index {case_index}; use a non-negative index."
        )
    size = len(variants)
    if size == 1:
        return variants

    # All rotations in both directions balance first and pairwise ordering for
    # two and three variants. Larger groups receive the same treatment without
    # coupling ordering to the growing benchmark record count.
    cycle_size = size if size == 2 else 2 * size
    phase = case_index % cycle_size
    ordered = variants
    if phase >= size:
        ordered = tuple(reversed(variants))
        phase -= size
    ordered = ordered[phase:] + ordered[:phase]
    return tuple(reversed(ordered)) if reverse else ordered


def reference(x: np.ndarray) -> np.ndarray:
    """Finite-input fp64 log probabilities without adding a tiny tail to one."""
    x = x.astype(np.float64)
    index = x.argmax(axis=-1, keepdims=True)
    shifted = x - np.take_along_axis(x, index, axis=-1)
    tail = np.exp(shifted)
    np.put_along_axis(tail, index, 0.0, axis=-1)
    return shifted - np.log1p(tail.sum(axis=-1, keepdims=True))


def paper_logsoftmax(x: mx.array) -> mx.array:
    work = x.astype(mx.float32)
    index = mx.argmax(work, axis=-1, keepdims=True)
    mask = mx.arange(x.shape[-1]) == index
    shifted = mx.where(mask, 0.0, work - mx.take_along_axis(work, index, axis=-1))
    others = mx.where(mask, -mx.inf, shifted)
    normalizer = mx.logaddexp(
        mx.logsumexp(others, axis=-1, keepdims=True), mx.array(0.0)
    )
    return (shifted - normalizer).astype(x.dtype)


def current_logsoftmax(x: mx.array) -> mx.array:
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def upcast_logsoftmax(x: mx.array) -> mx.array:
    """Use the existing forward formula, but accumulate and normalize in FP32."""
    work = x.astype(mx.float32)
    return (work - mx.logsumexp(work, axis=-1, keepdims=True)).astype(x.dtype)


def forward_logsoftmax(x: mx.array) -> mx.array:
    """Equivalent finite-input forward formula, not the paper's autodiff rule."""
    work = x.astype(mx.float32)
    index = mx.argmax(work, axis=-1, keepdims=True)
    mask = mx.arange(x.shape[-1]) == index
    shifted = work - mx.take_along_axis(work, index, axis=-1)
    tail = mx.sum(mx.where(mask, 0.0, mx.exp(shifted)), axis=-1, keepdims=True)
    return (shifted - mx.log1p(tail)).astype(x.dtype)


def model(
    op: str,
    shape: tuple[int, ...],
    dtype: DT,
    *,
    variant: str = "current",
    log_prob: bool = False,
) -> bytes:
    """FP32 boundaries permit BF16 testing without NumPy/ORT BF16 feeds."""
    if variant not in ("current", "upcast", "paper"):
        raise ValueError(
            f"Unknown softmax benchmark variant {variant!r}; use current, upcast, or paper."
        )
    nodes: list[ir.Node] = []
    initializers: list[ir.Value] = []

    def value(name, dt, dims):
        return ir.Value(name=name, type=ir.TensorType(dt), shape=ir.Shape(dims))

    def emit(kind, args, dt, dims, **attrs):
        out = value(f"v{len(nodes)}", dt, dims)
        nodes.append(ir.node(kind, args, attributes=attrs, outputs=[out]))
        return out

    def const(name, data):
        array = np.asarray(data)
        out = ir.Value(name=name, const_value=ir.tensor(array))
        initializers.append(out)
        return out

    x = value("x", DT.FLOAT, shape)
    inputs = [x]
    stored = emit("Cast", [x], dtype, shape, to=int(dtype))
    loss_op = op in ("SoftmaxCrossEntropyLoss", "NegativeLogLikelihoodLoss")
    loss_shape = shape[:-1]
    target = None
    if loss_op:
        if len(shape) != 2:
            raise ValueError(
                f"{op} audit expects [batch, classes], got {shape}; use a rank-2 shape."
            )
        target = value("target", DT.INT64, loss_shape)
        inputs.append(target)
    outputs = []
    if variant in ("current", "upcast"):
        compute_dtype = DT.FLOAT if variant == "upcast" else dtype
        work = stored
        if compute_dtype != dtype:
            work = emit("Cast", [stored], compute_dtype, shape, to=int(compute_dtype))
        result_shape = loss_shape if loss_op else shape
        out = value("result", compute_dtype, result_shape)
        op_outputs = [out]
        attrs = {"reduction": "none"} if loss_op else {"axis": -1}
        if log_prob:
            op_outputs.append(value("log_prob", compute_dtype, shape))
        nodes.append(
            ir.node(
                op,
                [work, target] if loss_op else [work],
                attributes=attrs,
                outputs=op_outputs,
            )
        )
        for result in op_outputs:
            if compute_dtype != dtype:
                result = emit(
                    "Cast",
                    [result],
                    dtype,
                    list(result.shape),
                    to=int(dtype),
                )
            outputs.append(
                emit("Cast", [result], DT.FLOAT, list(result.shape), to=int(DT.FLOAT))
            )
    else:
        work = emit("Cast", [stored], DT.FLOAT, shape, to=int(DT.FLOAT))
        row_shape = (*shape[:-1], 1)
        if target is None:
            index = emit("ArgMax", [work], DT.INT64, row_shape, axis=-1, keepdims=1)
        else:
            index = emit(
                "Unsqueeze", [target, const("unsqueeze_axis", [1])], DT.INT64, row_shape
            )
        center = emit("GatherElements", [work, index], DT.FLOAT, row_shape, axis=-1)
        mask = emit(
            "Equal",
            [const("classes", np.arange(shape[-1], dtype=np.int64)), index],
            DT.BOOL,
            shape,
        )
        shifted = emit("Sub", [work, center], DT.FLOAT, shape)
        if target is None:
            shifted = emit(
                "Where", [mask, const("zero", np.float32(0)), shifted], DT.FLOAT, shape
            )
        others = emit(
            "Where",
            [mask, const("neg_inf", np.float32(-np.inf)), shifted],
            DT.FLOAT,
            shape,
        )
        lse = emit(
            "ReduceLogSumExp",
            [others, const("reduce_axis", [-1])],
            DT.FLOAT,
            row_shape,
            keepdims=1,
        )
        normalizer = emit("Softplus", [lse], DT.FLOAT, row_shape)
        if target is None:
            result = emit("Sub", [shifted, normalizer], DT.FLOAT, shape)
            if op == "Softmax":
                result = emit("Exp", [result], DT.FLOAT, shape)
        else:
            result = emit(
                "Squeeze",
                [normalizer, const("squeeze_axis", [1])],
                DT.FLOAT,
                loss_shape,
            )
        result = emit("Cast", [result], dtype, list(result.shape), to=int(dtype))
        outputs.append(
            emit("Cast", [result], DT.FLOAT, list(result.shape), to=int(DT.FLOAT))
        )
    graph = ir.Graph(
        inputs,
        outputs,
        nodes=nodes,
        initializers=initializers,
        name=f"{op}_{variant}",
        opset_imports={"": 24},
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def session(data: bytes) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(data, options, providers=["MLXExecutionProvider"])


def rounded(x: np.ndarray, dtype: mx.Dtype) -> np.ndarray:
    return np.array(mx.array(x.astype(np.float32), dtype=dtype).astype(mx.float32))


def attention_accuracy() -> list[dict]:
    records = []
    for name, (onnx_dt, mlx_dt) in DTYPES.items():
        for logits in ([3.0, -3.0, -3.0], [9.0, -9.0, -9.0]):
            feeds = {}
            for key, shape in (
                ("q", (1, 1, 1, 64)),
                ("k", (1, 1, 3, 64)),
                ("v", (1, 1, 3, 64)),
            ):
                feeds[key] = np.zeros(shape, dtype=np.float32)
            feeds["q"][..., 0] = 1
            feeds["k"][0, 0, :, 0] = logits
            feeds["v"][0, 0, :, :3] = np.eye(3)
            expected = rounded(np.exp(reference(np.array([logits]))), mlx_dt)
            for composed in (False, True):
                inputs, stored, nodes = [], [], []
                for key, array in feeds.items():
                    inp = ir.Value(
                        name=key,
                        type=ir.TensorType(DT.FLOAT),
                        shape=ir.Shape(array.shape),
                    )
                    cast = ir.Value(
                        name=f"{key}_stored",
                        type=ir.TensorType(onnx_dt),
                        shape=ir.Shape(array.shape),
                    )
                    inputs.append(inp)
                    stored.append(cast)
                    nodes.append(
                        ir.node(
                            "Cast",
                            [inp],
                            attributes={"to": int(onnx_dt)},
                            outputs=[cast],
                        )
                    )
                dims = ir.Shape([1, 1, 1, 64])
                out = ir.Value(
                    name="attention", type=ir.TensorType(onnx_dt), shape=dims
                )
                result = ir.Value(
                    name="result", type=ir.TensorType(DT.FLOAT), shape=dims
                )
                attrs = {"scale": 1.0}
                if composed:
                    attrs["softmax_precision"] = int(DT.FLOAT)
                nodes.append(
                    ir.node("Attention", stored, attributes=attrs, outputs=[out])
                )
                nodes.append(
                    ir.node(
                        "Cast",
                        [out],
                        attributes={"to": int(DT.FLOAT)},
                        outputs=[result],
                    )
                )
                graph = ir.Graph(
                    inputs,
                    [result],
                    nodes=nodes,
                    name="attention_accuracy",
                    opset_imports={"": 25 if composed else 24},
                )
                actual = session(
                    ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
                ).run(None, feeds)[0]
                np.testing.assert_allclose(
                    actual[0, 0, 0, :3],
                    expected[0],
                    rtol={"fp32": 5e-6, "fp16": 2e-3, "bf16": 1e-2}[name],
                    atol=0,
                    err_msg=f"Attention/{name}/{logits}/{composed}",
                )
                records.append(
                    {
                        "dtype": name,
                        "logits": logits,
                        "composed": composed,
                        "probabilities": actual[0, 0, 0, :3].tolist(),
                        "rounded_reference": expected[0].tolist(),
                    }
                )
    return records


def accuracy() -> list[dict]:
    records = []
    mlx_variants = {
        "current": current_logsoftmax,
        "upcast": upcast_logsoftmax,
        "paper": paper_logsoftmax,
        "forward_log1p": forward_logsoftmax,
    }
    compiled_variants = {name: mx.compile(fn) for name, fn in mlx_variants.items()}
    cases = {
        "bf16_cliff": [3.0, -3.0, -3.0],
        "fp16_cliff": [4.0, -4.0, -4.0],
        "paper": [9.0, -9.0, -9.0],
        "nonmax_target": [-9.0, 9.0, -9.0],
        "large_margin": [0.0, -64.0, -64.0],
        "ties": [9.0, 9.0, -9.0],
        "singleton": [9.0],
        "uniform": [0.0, 0.0, 0.0],
        "uniform_offset": [1000.0, 1000.0, 1000.0],
        "wide_uniform_offset": [32768.0, 32768.0, 32768.0],
    }
    for name, (onnx_dt, mlx_dt) in DTYPES.items():
        for case, logits in cases.items():
            x = np.array([logits], dtype=np.float32)
            stored = rounded(x, mlx_dt)
            ref = reference(stored)
            feeds = {"x": x}
            target_feeds = {**feeds, "target": np.zeros(1, dtype=np.int64)}
            for op in ("Softmax", "LogSoftmax", "SoftmaxCrossEntropyLoss"):
                is_loss = op == "SoftmaxCrossEntropyLoss"
                feed = target_feeds if is_loss else feeds
                current = session(model(op, x.shape, onnx_dt, log_prob=is_loss)).run(
                    None, feed
                )
                upcast = session(
                    model(
                        op,
                        x.shape,
                        onnx_dt,
                        variant="upcast",
                        log_prob=is_loss,
                    )
                ).run(None, feed)
                paper = session(model(op, x.shape, onnx_dt, variant="paper")).run(
                    None, feed
                )[0]
                expected = (
                    -ref[:, 0] if is_loss else (np.exp(ref) if op == "Softmax" else ref)
                )
                quantized = rounded(expected, mlx_dt)
                # Compiled Metal log1p is less accurate than the standalone primitive.
                tolerance = {"fp32": 1e-4, "fp16": 2e-3, "bf16": 1e-2}[name]
                # The control rounds logsumexp before subtracting the logits.
                upcast_atol = max(5e-7, float(np.spacing(np.abs(stored)).max()))
                np.testing.assert_allclose(
                    upcast[0],
                    quantized,
                    rtol=tolerance,
                    atol=upcast_atol,
                    err_msg=f"upcast candidate: {name}/{case}/{op}",
                )
                if is_loss:
                    np.testing.assert_allclose(
                        upcast[1],
                        rounded(ref, mlx_dt),
                        rtol=tolerance,
                        atol=upcast_atol,
                        err_msg=f"upcast log_prob: {name}/{case}",
                    )
                np.testing.assert_allclose(
                    paper,
                    quantized,
                    rtol=tolerance,
                    atol=0,
                    err_msg=f"paper candidate: {name}/{case}/{op}",
                )
                mlx_results = {}
                if op == "LogSoftmax":
                    mlx_input = mx.array(x, dtype=mlx_dt)
                    for mode, functions in (
                        ("mlx", mlx_variants),
                        ("mlx_compiled", compiled_variants),
                    ):
                        for variant, function in functions.items():
                            actual = np.array(function(mlx_input).astype(mx.float32))
                            mlx_results[f"{mode}_{variant}"] = actual
                            if variant == "current":
                                continue
                            np.testing.assert_allclose(
                                actual,
                                quantized,
                                rtol=tolerance,
                                atol=upcast_atol if variant == "upcast" else 0,
                                err_msg=f"{mode} {variant} candidate: {name}/{case}",
                            )
                record = {
                    "dtype": name,
                    "case": case,
                    "op": op,
                    "reference": float(expected.flat[0]),
                    "rounded_reference": float(quantized.flat[0]),
                    "current": float(current[0].flat[0]),
                    "upcast": float(upcast[0].flat[0]),
                    "paper": float(paper.flat[0]),
                    "optional_log_prob": float(current[1].flat[0]) if is_loss else None,
                    "upcast_optional_log_prob": float(upcast[1].flat[0])
                    if is_loss
                    else None,
                }
                if mlx_results:
                    record.update(
                        {
                            key: float(actual.flat[0])
                            for key, actual in mlx_results.items()
                        }
                    )
                records.append(record)
            # NLL consumes log probabilities; it does not normalize or create the collapse.
            nll_feed = {
                "x": rounded(ref, mlx_dt),
                "target": np.zeros(1, dtype=np.int64),
            }
            nll = session(model("NegativeLogLikelihoodLoss", x.shape, onnx_dt)).run(
                None, nll_feed
            )[0]
            np.testing.assert_array_equal(
                nll,
                -rounded(ref[:, 0], mlx_dt),
                err_msg=f"NLL passthrough: {name}/{case}",
            )
            records.append(
                {
                    "dtype": name,
                    "case": case,
                    "op": "NegativeLogLikelihoodLoss",
                    "reference": float(-ref[0, 0]),
                    "current": float(nll[0]),
                }
            )
    return records


def measure(function, warmup: int, samples: int) -> dict:
    for _ in range(warmup):
        function()
    times = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        function()
        times.append((time.perf_counter_ns() - start) / 1e3)
    return {
        "median_us": statistics.median(times),
        "p10_us": float(np.percentile(times, 10)),
        "p90_us": float(np.percentile(times, 90)),
        "samples_us": times,
    }


def evaluate(function, *arrays) -> None:
    mx.eval(function(*arrays))


def composed_attention(q, k, v, *, variant="current"):
    scores = (q * q.shape[-1] ** -0.5) @ mx.swapaxes(k, -1, -2)
    if variant == "current":
        probs = mx.softmax(scores, axis=-1, precise=True)
    elif variant == "upcast":
        probs = mx.softmax(scores.astype(mx.float32), axis=-1, precise=True).astype(
            scores.dtype
        )
    elif variant == "paper":
        probs = mx.exp(paper_logsoftmax(scores.astype(mx.float32))).astype(scores.dtype)
    else:
        raise ValueError(
            f"Unknown attention benchmark variant {variant!r}; "
            "use current, upcast, or paper."
        )
    return probs @ v


def benchmark(warmup: int, samples: int, *, reverse_order: bool = False) -> list[dict]:
    rng = np.random.default_rng(42)
    records = []
    for shape_index, shape in enumerate(
        (
            (1, 32768),
            (1, 131072),
            (1, 262144),
            (8, 131072),
            (1, 8, 128, 128),
            (1, 8, 512, 512),
        )
    ):
        x_np = rng.standard_normal(shape).astype(np.float32)
        for dtype_index, (name, (onnx_dt, mlx_dt)) in enumerate(DTYPES.items()):
            for op_index, op in enumerate(
                ("Softmax", "LogSoftmax", "SoftmaxCrossEntropyLoss")
            ):
                if op == "SoftmaxCrossEntropyLoss" and len(shape) != 2:
                    continue
                feeds = {"x": x_np}
                if op == "SoftmaxCrossEntropyLoss":
                    feeds["target"] = x_np.argmax(axis=-1).astype(np.int64)
                for variant in counterbalanced_order(
                    ("current", "upcast", "paper"),
                    shape_index + dtype_index + op_index,
                    reverse=reverse_order,
                ):
                    run = session(model(op, shape, onnx_dt, variant=variant))
                    output = run.run(None, feeds)[0]
                    ref = reference(rounded(x_np, mlx_dt))
                    expected = (
                        -np.take_along_axis(ref, feeds["target"][:, None], axis=-1)[
                            :, 0
                        ]
                        if op == "SoftmaxCrossEntropyLoss"
                        else (np.exp(ref) if op == "Softmax" else ref)
                    )
                    np.testing.assert_allclose(
                        output,
                        rounded(expected, mlx_dt),
                        rtol={"fp32": 1e-5, "fp16": 2e-2, "bf16": 8e-2}[name],
                        atol=1e-6,
                        err_msg=f"{op}/{name}/{shape}/{variant}",
                    )
                    record = {
                        "backend": "EP",
                        "shape": shape,
                        "dtype": name,
                        "op": op,
                        "variant": variant,
                        **measure(
                            functools.partial(run.run, None, feeds), warmup, samples
                        ),
                    }
                    records.append(record)
                    print(
                        json.dumps(
                            {k: v for k, v in record.items() if k != "samples_us"}
                        ),
                        flush=True,
                    )
            x = mx.array(x_np, dtype=mlx_dt)
            mx.eval(x)
            variants = (
                ("current", current_logsoftmax),
                ("upcast", upcast_logsoftmax),
                ("paper", paper_logsoftmax),
                ("forward_log1p", forward_logsoftmax),
            )
            for variant, function in counterbalanced_order(
                variants, shape_index + dtype_index, reverse=reverse_order
            ):
                compiled = mx.compile(function)
                record = {
                    "backend": "MLX compiled",
                    "shape": shape,
                    "dtype": name,
                    "op": "LogSoftmax",
                    "variant": variant,
                    **measure(
                        functools.partial(evaluate, compiled, x), warmup, samples
                    ),
                }
                records.append(record)
                print(
                    json.dumps({k: v for k, v in record.items() if k != "samples_us"}),
                    flush=True,
                )
    for shape_index, shape in enumerate(
        ((1, 8, 1, 4096, 64), (1, 8, 128, 128, 64), (1, 8, 512, 512, 64))
    ):
        b, h, qlen, klen, dim = shape
        for dtype_index, (name, (_, mlx_dt)) in enumerate(DTYPES.items()):
            q, k, v = [
                mx.array(rng.standard_normal(dims).astype(np.float32), dtype=mlx_dt)
                for dims in ((b, h, qlen, dim), (b, h, klen, dim), (b, h, klen, dim))
            ]
            mx.eval(q, k, v)
            variants = (
                (
                    "fused_current",
                    lambda q, k, v: mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=q.shape[-1] ** -0.5
                    ),
                ),
                ("composed_current", composed_attention),
                (
                    "composed_upcast",
                    functools.partial(composed_attention, variant="upcast"),
                ),
                (
                    "composed_paper",
                    functools.partial(composed_attention, variant="paper"),
                ),
            )
            for variant, function in counterbalanced_order(
                variants, shape_index + dtype_index, reverse=reverse_order
            ):
                compiled = mx.compile(function)
                output = np.array(compiled(q, k, v).astype(mx.float32))
                q64, k64, v64 = [
                    np.array(a.astype(mx.float32)).astype(np.float64) for a in (q, k, v)
                ]
                scores = (q64 * dim**-0.5) @ np.swapaxes(k64, -1, -2)
                expected = np.exp(reference(scores)) @ v64
                np.testing.assert_allclose(
                    output,
                    rounded(expected, mlx_dt),
                    rtol={"fp32": 1e-4, "fp16": 2e-2, "bf16": 8e-2}[name],
                    atol={"fp32": 2e-6, "fp16": 1e-3, "bf16": 1e-2}[name],
                    err_msg=f"attention benchmark/{name}/{shape}/{variant}",
                )
                record = {
                    "backend": "MLX compiled",
                    "shape": shape,
                    "dtype": name,
                    "op": "Attention",
                    "variant": variant,
                    "max_abs_error": float(np.max(np.abs(output - expected))),
                    **measure(
                        functools.partial(evaluate, compiled, q, k, v), warmup, samples
                    ),
                }
                records.append(record)
                print(
                    json.dumps({k: v for k, v in record.items() if k != "samples_us"}),
                    flush=True,
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("accuracy", "benchmark", "all"), default="all"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--reverse-order",
        action="store_true",
        help="reverse every case's variant order for a counterbalanced second sweep",
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 1:
        parser.error(
            "--warmup and --samples must be positive; use at least 20/100 for reported timings."
        )
    lib = os.environ.get("ONNXRUNTIME_MLX_EP_LIB")
    if not lib or not Path(lib).is_file():
        parser.error(
            "Set ONNXRUNTIME_MLX_EP_LIB to the built libonnxruntime_mlx_ep.dylib; see docs/BUILDING.md."
        )
    ort.register_execution_provider_library(
        "MLXExecutionProvider", str(Path(lib).resolve())
    )
    result = {
        "environment": {
            "platform": platform.platform(),
            "device": mx.device_info(),
            "mlx": importlib.metadata.version("mlx"),
            "ort": ort.__version__,
            "library": str(Path(lib).resolve()),
            "no_compile": os.environ.get("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0"),
        },
        "benchmark_config": {
            "warmup": args.warmup,
            "samples": args.samples,
            "reverse_order": args.reverse_order,
        },
        "accuracy": accuracy() if args.mode != "benchmark" else [],
        "attention_accuracy": attention_accuracy() if args.mode != "benchmark" else [],
        "benchmarks": benchmark(
            args.warmup, args.samples, reverse_order=args.reverse_order
        )
        if args.mode != "accuracy"
        else [],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved numerical results and raw latency samples to {args.output}")


if __name__ == "__main__":
    main()
