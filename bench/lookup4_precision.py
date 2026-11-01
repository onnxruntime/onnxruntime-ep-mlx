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

import ml_dtypes
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "ops"))
from test_lut4_fusion import _model, _session


def _affine_model(bits, precision, rows, inner, outer, block=64):
    if inner % block:
        raise ValueError("Affine benchmark inner dimension must be a multiple of 64")
    dtype = {"fp32": np.float32, "fp16": np.float16, "bf16": ml_dtypes.bfloat16}[
        precision
    ]
    compute = {
        "fp32": ir.DataType.FLOAT,
        "fp16": ir.DataType.FLOAT16,
        "bf16": ir.DataType.BFLOAT16,
    }[precision]
    rng = np.random.default_rng(44)
    codes = rng.integers(0, 1 << bits, (outer, inner // block, block), dtype=np.uint8)
    packed = codes if bits == 8 else codes[..., ::2] | (codes[..., 1::2] << 4)
    scales = rng.uniform(0.001, 0.01, (outer, inner // block)).astype(dtype)
    weight = (
        (
            (codes.astype(np.float32) - (1 << (bits - 1)))
            * scales.astype(np.float32)[..., None]
        )
        .reshape(outer, inner)
        .T
    )
    tape = ir.tape.Tape()
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[rows, inner])
    a = (
        x
        if compute == ir.DataType.FLOAT
        else tape.op("Cast", [x], {"to": int(compute)})
    )
    p = tape.initializer(ir.tensor(packed, name="packed"))
    scale = tape.initializer(
        ir.tensor(scales.reshape(-1), name="scales", dtype=compute)
    )
    product = tape.op(
        "MatMulNBits",
        [a, p, scale],
        {"K": inner, "N": outer, "bits": bits, "block_size": block},
        domain="com.microsoft",
    )
    output = (
        product
        if compute == ir.DataType.FLOAT
        else tape.op("Cast", [product], {"to": int(ir.DataType.FLOAT)})
    )
    output.name, output.dtype, output.shape = (
        "y",
        ir.DataType.FLOAT,
        ir.Shape([rows, outer]),
    )
    model = ir.Model(
        ir.Graph(
            [x],
            [output],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18, "com.microsoft": 1},
        ),
        ir_version=10,
    )
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    return model, weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--format", choices=["nf4", "int4", "int8"], default="nf4")
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
    weight = None
    if args.format == "nf4":
        model, *_ = _model(
            k=args.inner,
            n=args.outer,
            precision=None if args.precision == "fp32" else args.precision,
            adjoint=args.adjoint,
            transpose=args.adjoint,
        )
    else:
        if args.adjoint or args.disable_fusion:
            parser.error(
                "Affine formats measure forward MatMulNBits; --adjoint/--disable-fusion select NF4 only"
            )
        model, weight = _affine_model(
            int(args.format[3:]), args.precision, args.rows, args.inner, args.outer
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
        "format": args.format,
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
    if weight is not None:
        dtype = {"fp32": np.float32, "fp16": np.float16, "bf16": ml_dtypes.bfloat16}[
            args.precision
        ]
        reference = (
            (x.astype(dtype).astype(np.float32) @ weight)
            .astype(dtype)
            .astype(np.float32)
        )
        delta = output - reference
        report["numpy_reference"] = {
            "max_abs": float(np.max(np.abs(delta))),
            "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(reference)),
        }
        epsilon = (
            1e-4
            if args.precision == "fp32"
            else float(np.finfo(dtype).eps)
            if args.precision == "fp16"
            else 2**-7
        )
        if report["numpy_reference"]["relative_l2"] > epsilon:
            raise AssertionError(report["numpy_reference"])
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": args.output, "median_ms": report["median_ms"]}))


if __name__ == "__main__":
    main()
