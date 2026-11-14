"""Synchronized, same-library A/B measurements of static weight-decode consumers."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

_FORMATS = {
    "gguf:q4_0": (32, 18, (0,)),
    "gguf:q4_1": (32, 20, (0, 2)),
    "gguf:q8_0": (32, 34, (0,)),
    "gguf:q4_k": (256, 144, (0, 2)),
    "gguf:q5_k": (256, 176, (0, 2)),
    "gguf:q6_k": (256, 210, (208,)),
}


def _model(args, *, dtype="float32", unique_routes=False):
    rng = np.random.default_rng(42)
    m, k, n = args.rows, args.inner, args.outer
    grouped = args.op == "grouped"
    shape = ([args.experts] if grouped else []) + (
        [n, k] if args.trans_b else [k, n]
    )
    element_type = TensorProto.FLOAT16 if dtype == "float16" else TensorProto.FLOAT
    numpy_type = np.float16 if dtype == "float16" else np.float32
    nodes, initializers = [], []
    if args.format == "float":
        weights = rng.normal(0, 0.02, shape).astype(numpy_type)
        initializers.append(numpy_helper.from_array(weights, "w"))
        packed_bytes = weights.nbytes
    elif args.format == "int8":
        weights = rng.integers(-127, 128, shape, dtype=np.int8)
        scale = np.full(shape[-2:-1] if args.trans_b else shape[-1:], 0.012, np.float32)
        zero = np.full(scale.shape, -3, np.int8)
        initializers.extend(numpy_helper.from_array(v, name) for name, v in [
            ("codes", weights), ("scale", scale), ("zero", zero)
        ])
        nodes.append(helper.make_node(
            "DequantizeLinear", ["codes", "scale", "zero"], ["w"],
            axis=len(shape) - (2 if args.trans_b else 1),
        ))
        packed_bytes = weights.nbytes + scale.nbytes + zero.nbytes
    else:
        block, size, offsets = _FORMATS[args.format]
        if shape[-1] % block:
            raise ValueError(f"{args.format} requires the last weight dimension divisible by {block}")
        payload = rng.integers(0, 256, (int(np.prod(shape)) // block, size), dtype=np.uint8)
        scale = np.array([0.002], dtype="<f2").view(np.uint8)
        for offset in offsets:
            payload[:, offset:offset + 2] = scale
        initializers.append(numpy_helper.from_array(payload.reshape(-1), "codes"))
        nodes.append(helper.make_node(
            "DequantizePacked", ["codes"], ["w"], format=args.format, shape=shape,
            output_dtype=element_type,
        ))
        packed_bytes = payload.nbytes
    feeds = {"x": rng.normal(0, 0.1, (m, k)).astype(numpy_type)}
    info = lambda name, dtype, dims: helper.make_tensor_value_info(name, dtype, dims)
    inputs = [info("x", element_type, [m, k])]
    if grouped:
        feeds["ids"] = rng.integers(
            0, min(args.active_experts, args.experts), (m, args.selections), dtype=np.int64
        )
        if unique_routes:
            if args.selections > min(args.active_experts, args.experts):
                raise ValueError("Unique routing requires selections <= active experts <= experts")
            feeds["ids"][:] = np.arange(args.selections)
        inputs.append(info("ids", TensorProto.INT64, [m, args.selections]))
        nodes.append(helper.make_node(
            "GroupedMatMul", ["x", "w", "ids"], ["y"], transB=args.trans_b
        ))
        output_shape = [m, args.selections, n]
    elif args.op == "gemm":
        nodes.append(helper.make_node("Gemm", ["x", "w"], ["y"], transB=args.trans_b))
        output_shape = [m, n]
    else:
        if args.trans_b:
            nodes.append(helper.make_node("Transpose", ["w"], ["wt"], perm=[1, 0]))
        nodes.append(helper.make_node("MatMul", ["x", "wt" if args.trans_b else "w"], ["y"]))
        output_shape = [m, n]
    model = helper.make_model(
        helper.make_graph(
            nodes, "static_weight_decode", inputs,
            [info("y", element_type, output_shape)], initializers,
        ), opset_imports=[helper.make_opsetid("", 29)], ir_version=11,
    )
    return model, feeds, packed_bytes, int(np.prod(shape)) * np.dtype(numpy_type).itemsize


class _Stream(ctypes.Structure):
    _fields_ = [("ctx", ctypes.c_void_p)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--op", choices=["grouped", "gemm", "matmul"], default="grouped")
    parser.add_argument("--format", choices=["float", "int8", *_FORMATS], default="gguf:q4_0")
    parser.add_argument("--trans-b", type=int, choices=[0, 1], default=1)
    for name, default in [
        ("rows", 1), ("inner", 4096), ("outer", 4096),
        ("experts", 8), ("active-experts", 2), ("selections", 2),
        ("samples", 30), ("warmup", 10),
    ]:
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--disable-fusion", action="store_true")
    parser.add_argument("--unique-routes", action="store_true",
                        help="Route each token to distinct experts [0..selections-1]")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    args = parser.parse_args()
    if min(args.rows, args.inner, args.outer, args.experts, args.active_experts,
           args.selections, args.samples) < 1 or args.warmup < 0:
        parser.error("Use positive dimensions/sample counts and nonnegative warmup")
    if args.active_experts > args.experts:
        parser.error("--active-experts must not exceed --experts")
    if args.unique_routes and args.selections > args.active_experts:
        parser.error("Unique routes require --selections <= --active-experts")
    if args.dtype != "float32" and args.format == "int8":
        parser.error("The native int8 fused comparison requires --dtype float32")
    if os.environ.get("ONNXRUNTIME_EP_MLX_TRACE") or any(
        os.environ.get(key, "0") not in ["", "0"]
        for key in ["ONNXRUNTIME_EP_MLX_TRACE_FINE", "ONNXRUNTIME_EP_MLX_VERBOSE"]
    ):
        parser.error("Unset MLX trace/fine-trace/verbose variables before measuring latency")
    library = args.library.resolve(strict=True)
    for key in ["ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION", "ONNXRUNTIME_EP_MLX_NO_PACKED_FUSION"]:
        os.environ[key] = "1" if args.disable_fusion else "0"
    os.environ["ALLOW_RELEASED_ONNX_OPSET_ONLY"] = "0"
    ort.register_execution_provider_library("MLXExecutionProvider", str(library))
    mlx = ctypes.CDLL(str(library))
    mlx.mlx_default_gpu_stream_new.restype = _Stream
    mlx.mlx_synchronize.argtypes = [_Stream]
    mlx.mlx_stream_free.argtypes = [_Stream]
    stream = mlx.mlx_default_gpu_stream_new()

    def checked(fn, *values):
        rc = fn(*values)
        if rc:
            raise RuntimeError(f"MLX {fn.__name__} returned {rc}; check the native MLX error log")

    model, feeds, packed_bytes, decoded_bytes = _model(
        args, dtype=args.dtype, unique_routes=args.unique_routes
    )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("session.disable_aot_function_inlining", "1")
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["MLXExecutionProvider"]
    )
    checked(mlx.mlx_reset_peak_memory)
    timings = []
    try:
        for iteration in range(args.warmup + args.samples):
            checked(mlx.mlx_synchronize, stream)
            start = time.perf_counter_ns()
            output = session.run(None, feeds)[0]
            checked(mlx.mlx_synchronize, stream)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if iteration >= args.warmup:
                timings.append(elapsed)
        peak = ctypes.c_size_t()
        checked(mlx.mlx_get_peak_memory, ctypes.byref(peak))
    finally:
        checked(mlx.mlx_stream_free, stream)
    report = {
        "format": args.format, "op": args.op, "transB": args.trans_b,
        "fusion_disabled": args.disable_fusion, "dtype": args.dtype,
        "unique_routes": args.unique_routes,
        "shape": {"M": args.rows, "K": args.inner, "N": args.outer,
                  "E": args.experts if args.op == "grouped" else 1,
                  "S": args.selections if args.op == "grouped" else 1},
        "selected_experts": int(np.unique(feeds["ids"]).size) if "ids" in feeds else 1,
        "warmup": args.warmup, "samples_ms": timings,
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "packed_bytes": packed_bytes, "full_decoded_weight_bytes": decoded_bytes,
        "mlx_peak_active_bytes": peak.value, "runtime": ort.__version__,
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "platform": platform.platform(),
        "scope": "Warm synchronous ORT calls, CPU input/output boundaries included, explicit GPU synchronization; setup excluded",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.save(args.output.with_suffix(".npy"), output)
    print(json.dumps({"output": str(args.output), "median_ms": report["median_ms"],
                      "mlx_peak_active_bytes": peak.value}))


if __name__ == "__main__":
    main()
