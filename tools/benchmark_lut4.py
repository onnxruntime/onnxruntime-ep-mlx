"""Measure packed lookup4, fused lookup4 and cached FP32 projection sessions."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

BOOK = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    np.float32,
)


def quantize(weight, block=64):
    size = weight.size
    padded = np.pad(weight.reshape(-1), (0, -size % block)).reshape(-1, block)
    scales = np.abs(padded).max(axis=1)
    divisor = np.where(scales == 0, np.float32(1), scales)
    codes = np.abs((padded / divisor[:, None])[:, :, None] - BOOK).argmin(axis=-1)
    codes = codes.astype(np.uint8).reshape(-1)
    packed = (codes[::2] << 4) | codes[1::2]
    decoded = (BOOK[codes].reshape(-1, block) * scales[:, None]).reshape(-1)[:size]
    return packed, scales, decoded.reshape(weight.shape)


def model(k, n, packed, scales, decoded, transpose, cached):
    node = helper.make_node
    values = {}
    nodes = []
    if cached:
        values["weight"] = decoded
    else:
        values = {
            "packed": packed,
            "scales": scales,
            "book": BOOK,
            "sixteen": np.array(16, np.int64),
            "axes": np.array([1], np.int64),
            "blocks": np.array([-1, 64], np.int64),
            "flat": np.array([-1], np.int64),
            "start": np.array([0], np.int64),
            "end": np.array([k * n], np.int64),
            "axis": np.array([0], np.int64),
            "matrix": np.array([k, n], np.int64),
        }
        nodes = [
            node("Cast", ["packed"], ["integer"], to=TensorProto.INT64),
            node("Div", ["integer", "sixteen"], ["high"]),
            node("Mod", ["integer", "sixteen"], ["low"]),
            node("Unsqueeze", ["high", "axes"], ["hi"]),
            node("Unsqueeze", ["low", "axes"], ["lo"]),
            node("Concat", ["hi", "lo"], ["pairs"], axis=1),
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
    nodes.append(node("MatMul", ["x", "weight_t" if transpose else "weight"], ["y"]))
    inner, outer = (n, k) if transpose else (k, n)
    info = helper.make_tensor_value_info
    result = helper.make_model(
        helper.make_graph(
            nodes,
            "lookup4_projection",
            [info("x", TensorProto.FLOAT, ["m", inner])],
            [info("y", TensorProto.FLOAT, ["m", outer])],
            [numpy_helper.from_array(value, name) for name, value in values.items()],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.checker.check_model(result, full_check=True)
    return result


def worker(args):
    import onnxruntime as ort

    ort.register_execution_provider_library("MLXExecutionProvider", args.library)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    start = time.perf_counter()
    runtime = ort.InferenceSession(
        args.model, options, providers=["MLXExecutionProvider"]
    )
    construction = time.perf_counter() - start
    with np.load(args.data, allow_pickle=False) as data:
        x, expected = data["x"], data["expected"]
    actual = runtime.run(None, {"x": x})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)
    for _ in range(args.warmup):
        runtime.run(None, {"x": x})
    times = []
    for _ in range(args.samples):
        start = time.perf_counter()
        runtime.run(None, {"x": x})
        times.append(time.perf_counter() - start)
    report = {
        "construction_seconds": construction,
        "median_seconds": float(np.median(times)),
        "p95_seconds": float(np.percentile(times, 95)),
        "warmup": args.warmup,
        "samples": args.samples,
        "process_peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "max_abs": float(np.max(abs(actual - expected))),
        "shape": list(x.shape),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def shape(value):
    try:
        k, n = map(int, value.lower().split("x"))
        if min(k, n) <= 0:
            raise ValueError
        return k, n
    except ValueError:
        raise argparse.ArgumentTypeError("Expected positive KxN dimensions") from None


def run(args):
    rng = np.random.default_rng(29)
    rows = []
    with tempfile.TemporaryDirectory(prefix="mlx-lut4-benchmark-") as directory:
        directory = Path(directory)
        for k, n in args.shape or [(1024, 2048), (3072, 1024)]:
            weight = rng.normal(0, 0.02, (k, n)).astype(np.float32)
            packed, scales, decoded = quantize(weight)
            for transpose in (False, True):
                inner = n if transpose else k
                prefix = f"{k}-{n}-{transpose}"
                paths = {}
                for cached in (False, True):
                    path = directory / f"{prefix}-{cached}.onnx"
                    onnx.save(
                        model(k, n, packed, scales, decoded, transpose, cached), path
                    )
                    paths[cached] = path
                x = rng.normal(0, 0.1, (args.rows, inner)).astype(np.float32)
                data = directory / f"{prefix}.npz"
                np.savez(data, x=x, expected=x @ (decoded.T if transpose else decoded))
                row = {
                    "k": k,
                    "n": n,
                    "transpose": transpose,
                    "packed_bytes": packed.nbytes + scales.nbytes,
                    "float_weight_bytes": decoded.nbytes,
                    "model_bytes": paths[False].stat().st_size,
                    "cached_fp32_model_bytes": paths[True].stat().st_size,
                }
                for variant in ("baseline", "fused", "cached_fp32"):
                    env = dict(os.environ)
                    env.pop("ONNXRUNTIME_EP_MLX_TRACE", None)
                    if variant == "baseline":
                        env["ONNXRUNTIME_EP_MLX_NO_LUT4_FUSION"] = "1"
                    else:
                        env.pop("ONNXRUNTIME_EP_MLX_NO_LUT4_FUSION", None)
                    output = directory / f"{prefix}-{variant}.json"
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--library",
                        args.library,
                        "--model",
                        str(paths[variant == "cached_fp32"]),
                        "--data",
                        str(data),
                        "--output",
                        str(output),
                        "--warmup",
                        str(args.warmup),
                        "--samples",
                        str(args.samples),
                    ]
                    subprocess.run(command, env=env, check=True)
                    row[variant] = json.loads(output.read_text())
                    trace = directory / f"{prefix}-{variant}-trace.json"
                    env["ONNXRUNTIME_EP_MLX_TRACE"] = str(trace)
                    subprocess.run(
                        command + ["--warmup", "0", "--samples", "1"],
                        env=env,
                        check=True,
                    )
                    events = json.loads(trace.read_text())
                    kernels = sum(
                        e.get("args", {}).get("kernel") == "mlx_lookup4_matmul"
                        for e in events
                    )
                    if bool(kernels) != (variant == "fused"):
                        raise AssertionError(
                            f"Unexpected packed-kernel dispatch: {variant}, calls={kernels}"
                        )
                    row[variant]["fused_calls_observed"] = kernels
                    row[variant]["sampled_metal_allocated_bytes"] = max(
                        (
                            e["args"]["bytes"]
                            for e in events
                            if e["name"] == "mlx.gpu_mem_bytes"
                        ),
                        default=0,
                    )
                rows.append(row)
                print(json.dumps(row), flush=True)
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        cpu = result.stdout.strip() or cpu
    return {
        "hardware": {
            "cpu": cpu,
            "logical_cores": os.cpu_count(),
            "physical_memory_bytes": os.sysconf("SC_PAGE_SIZE")
            * os.sysconf("SC_PHYS_PAGES"),
        },
        "compute_dtype": "float32",
        "rows": rows,
        "library_sha256": hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "threads": 1,
        "ort_optimizer": "disabled",
        "scope": "Isolated worker per variant; synchronous ORT run includes CPU input/output exchange. RSS includes imports, construction and runtime; parent model preparation is separate. Metal allocation sampled in a separate trace run. Timing has tracing disabled.",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("onnxruntime", "onnx", "numpy")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shape", type=shape, action="append")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--data", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rows <= 0 or args.samples <= 0 or args.warmup < 0:
        parser.error("Rows and samples must be positive; warmup must be nonnegative")
    args.library = str(Path(args.library).resolve())
    if args.worker:
        worker(args)
    else:
        Path(args.output).write_text(
            json.dumps(run(args), indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
