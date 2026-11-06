"""Compare CPU versus persistent public-device state bindings on one paged export.

Run as ``python -m bench.ort_paged``. This is a target-only ORT loop, not a GenAI
Engine or MTP benchmark. Sampling remains CPU argmax over the unchanged logits.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

import numpy as np

from bench.genai_paged import summarize


def state_output(name):
    if name.startswith("past_key_values."):
        return "present." + name.removeprefix("past_key_values.")
    if name.startswith("past."):
        return "present." + name.removeprefix("past.")
    raise ValueError(f"{name!r} is not a paged state input")


def run_once(
    session, prompt, generated, *, info, blocks, block_size, prefill_chunk_size=None
):
    import onnxruntime as ort

    if len(prompt) == 0 or (prefill_chunk_size is not None and prefill_chunk_size <= 0):
        raise ValueError("Provide a nonempty prompt and a positive prefill chunk size")
    dimensions = {
        "num_blocks": blocks,
        "block_size": block_size,
        "batch_size": 1,
        "max_num_blocks": blocks,
    }
    binding = session.io_binding()
    states = []
    for desc in session.get_inputs():
        if not desc.name.startswith(("past.", "past_key_values.")):
            continue
        shape = [d if isinstance(d, int) else dimensions[d] for d in desc.shape]
        dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16}[desc.type]
        zero = np.zeros(shape, dtype=dtype)
        if info is None:
            value = ort.OrtValue.ortvalue_from_numpy(zero)
        else:
            value = ort.OrtValue.ortvalue_from_shape_and_type(
                shape, dtype, memory_info=info
            )
            value.update_inplace(zero)
        states.append(value)
        binding.bind_ortvalue_input(desc.name, value)
        binding.bind_ortvalue_output(state_output(desc.name), value)
    if not states:
        raise ValueError(
            "Export has no recognized paged/recurrent states; use a paged builder export"
        )
    outputs = {v.name: v for v in session.get_outputs()}
    if "logits" not in outputs:
        raise ValueError("Export must expose logits; GPU sampling is not substituted")
    logits_desc = outputs["logits"]
    dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16}[
        logits_desc.type
    ]
    logits = ort.OrtValue.ortvalue_from_shape_and_type(
        [1, logits_desc.shape[-1]], dtype
    )
    chunk_size = prefill_chunk_size or len(prompt)
    empty_logits = (
        ort.OrtValue.ortvalue_from_shape_and_type([0, logits_desc.shape[-1]], dtype)
        if chunk_size < len(prompt)
        else None
    )
    binding.bind_ortvalue_output("logits", logits)
    inputs = {v.name for v in session.get_inputs()}
    metadata_names = {
        "input_ids",
        "position_ids",
        "block_table",
        "cumulative_sequence_lengths",
        "past_sequence_lengths",
        "attention_metadata",
        "logits_indices",
    }
    unexpected = (
        inputs
        - metadata_names
        - {
            v.name
            for v in session.get_inputs()
            if v.name.startswith(("past.", "past_key_values."))
        }
    )
    if unexpected:
        raise ValueError(
            f"Unsupported decoder metadata {sorted(unexpected)}; use the GenAI runner"
        )
    if "logits_indices" not in inputs:
        raise ValueError(
            "Use a pruned target-only export with logits_indices for this driver"
        )
    table = np.arange(blocks, dtype=np.int32)[None]
    binding.bind_cpu_input("block_table", table)
    tokens, times = [], []
    current, past = prompt[:chunk_size], 0
    started = time.perf_counter()
    while len(tokens) < generated:
        n = len(current)
        intermediate = past + n < len(prompt)
        if past + n > blocks * block_size:
            raise ValueError(
                "Prompt plus generation exceeds paged capacity; increase --blocks"
            )
        feeds = {
            "input_ids": np.asarray(current, np.int64),
            "position_ids": np.tile(np.arange(past, past + n, dtype=np.int64), (3, 1)),
            "cumulative_sequence_lengths": np.array([0, n], np.int32),
            "past_sequence_lengths": np.array([past], np.int32),
            "attention_metadata": np.array([n, past + n, 0], np.int32),
            "logits_indices": np.array([] if intermediate else [n - 1], np.int32),
        }
        for name, value in feeds.items():
            if name in inputs:
                binding.bind_cpu_input(name, value)
        binding.bind_ortvalue_output(
            "logits", empty_logits if intermediate else logits
        )
        session.run_with_iobinding(binding)
        past += n
        if intermediate:
            current = prompt[past : past + chunk_size]
            continue
        token = int(logits.numpy().reshape(-1).argmax())
        times.append(time.perf_counter())
        tokens.append(token)
        current = [token]
    return {
        "tokens": tokens,
        "ttft_seconds": times[0] - started,
        "decode_seconds": times[-1] - times[0],
        "decode_intervals_seconds": [b - a for a, b in pairwise(times)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Flat GenAI report with prompt_tokens and runs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory", choices=("cpu", "mlx"), required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        help="Prefill in bounded chunks, retaining state bindings and skipping intermediate logits",
    )
    args = parser.parse_args()
    if args.tokens < 2 or args.warmup < 3 or args.repetitions < 1:
        parser.error("require tokens >= 2, warmup >= 3 and repetitions >= 1")
    if args.blocks < 1 or args.block_size < 1:
        parser.error("require positive --blocks and --block-size")
    if args.prefill_chunk_size is not None and args.prefill_chunk_size <= 0:
        parser.error("--prefill-chunk-size must be positive")
    if any(
        os.environ.get(k)
        for k in ("ONNXRUNTIME_EP_MLX_TRACE", "ONNXRUNTIME_EP_MLX_GPU_CAPTURE")
    ):
        parser.error("unset tracing/capture before final measurements")
    if args.memory == "mlx":
        os.environ["ONNXRUNTIME_EP_MLX_DEVICE_TENSORS"] = "1"
    elif os.environ.get("ONNXRUNTIME_EP_MLX_DEVICE_TENSORS", "0") != "0":
        parser.error("unset ONNXRUNTIME_EP_MLX_DEVICE_TENSORS for --memory cpu")
    if not any(
        os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")
        for name in ("CI", "GITHUB_ACTIONS")
    ):
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    import onnxruntime as ort

    config = json.loads((args.model / "genai_config.json").read_text())
    if config.get("model", {}).get("mtp", {}).get("enabled"):
        parser.error(
            "this target-only loop cannot measure MTP; use bench/genai_paged.py --mtp"
        )
    reference = json.loads(args.reference.read_text())
    prompt = reference["prompt_tokens"]
    expected = reference["runs"][0]["tokens"][: args.tokens]
    if len(expected) != args.tokens:
        parser.error("reference must contain at least --tokens generated tokens")
    ort.register_execution_provider_library(
        "MLXExecutionProvider", str(args.library.resolve())
    )
    info = None
    if args.memory == "mlx":
        device = next(
            d for d in ort.get_ep_devices() if d.ep_name == "MLXExecutionProvider"
        )
        info = device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(args.model / "model.onnx"), options, providers=["MLXExecutionProvider"]
    )
    session.disable_fallback()
    construction = time.perf_counter() - started
    runs = []
    for i in range(args.warmup + args.repetitions):
        result = run_once(
            session,
            prompt,
            args.tokens,
            info=info,
            blocks=args.blocks,
            block_size=args.block_size,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        if result["tokens"] != expected:
            mismatch = next(
                j for j, (a, b) in enumerate(zip(result["tokens"], expected)) if a != b
            )
            raise RuntimeError(
                f"Greedy reference mismatch at generated token {mismatch}; do not compare throughput"
            )
        print(
            json.dumps(
                {
                    "warmup": i < args.warmup,
                    "ttft_seconds": result["ttft_seconds"],
                    "decode_seconds": result["decode_seconds"],
                }
            ),
            flush=True,
        )
        if i >= args.warmup:
            runs.append(result)
    metrics = summarize(runs)
    metrics["decode_interval_p95_ms"] = float(
        np.percentile(
            [v * 1000 for r in runs for v in r["decode_intervals_seconds"]],
            95,
        )
    )
    report = {
        "driver": "public ORT IOBinding, target-only; CPU argmax over unchanged logits",
        "memory": args.memory,
        "prompt_tokens": prompt,
        "generated_tokens": args.tokens,
        "warmups": args.warmup,
        "repetitions": args.repetitions,
        "blocks": args.blocks,
        "block_size": args.block_size,
        "prefill_chunk_size": args.prefill_chunk_size,
        "model_directory": str(args.model.resolve()),
        "config_sha256": hashlib.sha256(
            (args.model / "genai_config.json").read_bytes()
        ).hexdigest(),
        "plugin_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "mlx_options": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("ONNXRUNTIME_EP_MLX_")
        },
        "onnxruntime": ort.__version__,
        "onnxruntime_distribution": version("onnxruntime"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "memory_bytes": int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        ),
        "model_construction_seconds": construction,
        "runs": runs,
        **metrics,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
