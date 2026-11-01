"""Measure precision-preserving lookup4 fusion using the correctness-test graphs.

Run each configuration in a fresh process with an absolute --library path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "ops"))
from test_lut4_fusion import _model, _session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], required=True)
    parser.add_argument("--adjoint", action="store_true")
    parser.add_argument("--disable-fusion", action="store_true")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--inner", type=int, default=1024)
    parser.add_argument("--outer", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.rows, args.inner, args.outer, args.samples) < 1 or args.warmup < 0:
        parser.error("Expected positive dimensions/sample count and nonnegative warmup")
    os.environ["ONNXRUNTIME_EP_MLX_NO_COMPILE"] = "1"
    os.environ["ONNXRUNTIME_EP_MLX_NO_LUT4_FUSION"] = (
        "1" if args.disable_fusion else "0"
    )
    ort.register_execution_provider_library("MLXExecutionProvider", args.library)
    model, *_ = _model(
        k=args.inner,
        n=args.outer,
        precision=None if args.precision == "fp32" else args.precision,
        adjoint=args.adjoint,
        transpose=args.adjoint,
    )
    runtime = _session(model, "MLXExecutionProvider")
    inner, outer = (
        (args.outer, args.inner) if args.adjoint else (args.inner, args.outer)
    )
    x = np.random.default_rng(12).normal(0, 0.1, (args.rows, inner)).astype(np.float32)
    for _ in range(args.warmup):
        runtime.run(None, {"x": x})
    timings = []
    for _ in range(args.samples):
        start = time.perf_counter()
        output = runtime.run(None, {"x": x})[0]
        timings.append((time.perf_counter() - start) * 1000)
    np.save(args.output + ".npy", output)
    report = {
        "version": 1,
        "precision": args.precision,
        "adjoint": args.adjoint,
        "fusion_disabled": args.disable_fusion,
        "shape": [args.rows, inner, outer],
        "warmup": args.warmup,
        "samples": args.samples,
        "samples_ms": timings,
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "platform": platform.platform(),
        "runtime_version": ort.__version__,
        "library_sha256": hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
        "scope": "Synchronous runtime.run including FP32 host boundaries; session setup excluded; CPU fallback and ORT graph optimization disabled",
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": args.output, "median_ms": report["median_ms"]}))


if __name__ == "__main__":
    main()
