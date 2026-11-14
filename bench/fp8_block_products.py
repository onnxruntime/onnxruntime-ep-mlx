"""Same-library, synchronized FP8 contrib / ONNX-pattern / unfused comparison."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import ml_dtypes
import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def model(kind, dtype, m, k, n, block, activation_scale):
    rng = np.random.default_rng(57)
    low = TensorProto.FLOAT16 if dtype == "fp16" else TensorProto.BFLOAT16
    np_low = np.float16 if dtype == "fp16" else ml_dtypes.bfloat16
    weights = rng.normal(0, 1, (n, k)).astype(ml_dtypes.float8_e4m3fn)
    scale = rng.uniform(0.013, 0.19, (n, (k + block - 1) // block)).astype(np.float32)
    bias = rng.normal(0, 0.2, n).astype(np_low)
    initializers = [numpy_helper.from_array(weights, "w"), numpy_helper.from_array(scale, "ws"),
                    numpy_helper.from_array(bias, "bias")]
    nodes = [helper.make_node("Cast", ["x"], ["a"], to=low)]
    if activation_scale:
        initializers.append(numpy_helper.from_array(np.array(0.25, np.float32), "as"))
    if kind == "contrib":
        nodes.append(helper.make_node("MatMulBlockQuantizedFp8Weight",
            ["a", "w", "ws", "as" if activation_scale else "", "bias"], ["low_y"],
            domain="com.microsoft", block_size=block))
    else:
        nodes += [helper.make_node("DequantizeLinear", ["w", "ws"], ["wf"], axis=1,
                                   block_size=block),
                  helper.make_node("Cast", ["wf"], ["wl"], to=low)]
        activation = "a"
        if activation_scale:
            nodes += [helper.make_node("Cast", ["a"], ["af"], to=TensorProto.FLOAT),
                      helper.make_node("QuantizeLinear", ["af", "as"], ["aq"],
                                       output_dtype=TensorProto.FLOAT8E4M3FN),
                      helper.make_node("DequantizeLinear", ["aq", "as"], ["ad"]),
                      helper.make_node("Cast", ["ad"], ["al"], to=low)]
            activation = "al"
        if kind == "gemm":
            nodes.append(helper.make_node("Gemm", [activation, "wl", "bias"], ["low_y"],
                                           transB=1))
        else:
            nodes += [helper.make_node("Transpose", ["wl"], ["wt"], perm=[1, 0]),
                      helper.make_node("MatMul", [activation, "wt"], ["p"]),
                      helper.make_node("Add", ["p", "bias"], ["low_y"])]
    nodes.append(helper.make_node("Cast", ["low_y"], ["y"], to=TensorProto.FLOAT))
    graph = helper.make_graph(nodes, "fp8_block_products",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [m, k])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [m, n])], initializers)
    serialized = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21),
        helper.make_opsetid("com.microsoft", 1)], ir_version=10).SerializeToString()
    return serialized, {"x": rng.normal(0, 1, (m, k)).astype(np.float32)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--inner", type=int, default=4096)
    parser.add_argument("--outer", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--activation-scale", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(*args.rows, args.inner, args.outer, args.block_size, args.iterations) <= 0:
        parser.error("dimensions, block size and iterations must be positive")
    if args.warmup < 1:
        parser.error("warmup must be positive")
    if os.environ.get("ONNXRUNTIME_EP_MLX_TRACE") or any(
        os.environ.get(key, "0") not in ("", "0")
        for key in ["ONNXRUNTIME_EP_MLX_TRACE_FINE", "ONNXRUNTIME_EP_MLX_VERBOSE"]
    ):
        parser.error("unset MLX tracing and verbose output before timing")
    library = args.library.resolve()
    ort.register_execution_provider_library("MLXExecutionProvider", str(library))
    report = {"machine": platform.machine(), "platform": platform.platform(),
              "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
              "ort": ort.__version__, "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
              "dtype": args.dtype, "K": args.inner, "N": args.outer,
              "block_size": args.block_size, "activation_scale": args.activation_scale,
              "warmup": args.warmup, "iterations": args.iterations, "results": []}
    variants = [("contrib", False), ("gemm", False), ("matmul", False),
                ("gemm", True), ("matmul", True)]
    previous = os.environ.get("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION")
    try:
        for m in args.rows:
            sessions = []
            for kind, disabled in variants:
                os.environ["ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION"] = str(int(disabled))
                data, feeds = model(kind, args.dtype, m, args.inner, args.outer,
                                    args.block_size, args.activation_scale)
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                result = ort.InferenceSession(data, options, providers=["MLXExecutionProvider"],
                                               enable_fallback=False)
                result.disable_fallback()
                for _ in range(args.warmup):
                    result.run(None, feeds)
                sessions.append((result, feeds))
            baseline = sessions[-1][0].run(None, sessions[-1][1])[0]
            for result, feeds in sessions:
                np.testing.assert_allclose(result.run(None, feeds)[0], baseline,
                    rtol=0.02 if args.dtype == "bf16" else 0.003, atol=0.02)
            timings = [[] for _ in variants]
            # Rotate order to avoid assigning thermal/clock drift to one implementation.
            for iteration in range(args.iterations):
                for offset in range(len(variants)):
                    index = (iteration + offset) % len(variants)
                    result, feeds = sessions[index]
                    start = time.perf_counter_ns()
                    result.run(None, feeds)  # Host output completes GPU execution.
                    timings[index].append((time.perf_counter_ns() - start) / 1e6)
            for (kind, disabled), times in zip(variants, timings, strict=True):
                report["results"].append({"M": m, "kind": kind, "fusion": not disabled,
                    "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
                    "samples_ms": times})
    finally:
        if previous is None:
            os.environ.pop("ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION", None)
        else:
            os.environ["ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION"] = previous
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
