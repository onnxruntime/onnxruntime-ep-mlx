"""Benchmark one unchanged GenAI Engine export in a fresh process per EP."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path


def summarize(runs):
    decoded_tokens = sum(len(run["tokens"]) - 1 for run in runs)
    decode_seconds = sum(run["decode_seconds"] for run in runs)
    if not decoded_tokens or decode_seconds <= 0:
        raise ValueError(
            "Decode timing requires at least two generated tokens and positive elapsed time"
        )
    return {
        "warm_ttft_median_ms": statistics.median(
            run["ttft_seconds"] * 1000 for run in runs
        ),
        "decode_median_ms_per_token": statistics.median(
            run["decode_seconds"] * 1000 / (len(run["tokens"]) - 1) for run in runs
        ),
        "aggregate_decode_tokens_per_second": decoded_tokens / decode_seconds,
    }


def run_once(og, model, tokens, generated, *, require_mtp=False):
    import numpy as np

    # A fresh Engine avoids cross-run prefix-cache hits; the model's sessions stay warm.
    engine = og.Engine(model)
    if require_mtp and not engine.max_draft_tokens_per_proposal():
        raise RuntimeError(
            "GenAI Engine reports max_draft_tokens_per_proposal=0: this model cannot "
            "execute MTP verification. Re-export hybrid models with positive "
            "state_update_capacity and use a provider supporting compact state replay"
        )
    options = og.RequestOptions()
    options.set_max_session_tokens(len(tokens) + generated)
    request = engine.create_request(options=options)
    turn = og.TurnOptions(request)
    turn.set_do_sample(False)
    turn.set_min_generated_tokens(generated)
    turn.set_max_generated_tokens(generated)
    events = engine.create_event_buffer(1)
    output, timestamps = [], []
    start = time.perf_counter()
    try:
        request.begin_turn(np.asarray(tokens, dtype=np.int32), turn)
        while engine.has_pending_requests():
            for event in engine.run(events):
                if event.flags & og.EngineEventFlags.FAILED:
                    raise RuntimeError(
                        f"GenAI request failed with error_code={event.error_code}; inspect EP logs"
                    )
                if event.flags & og.EngineEventFlags.TOKEN:
                    if event.request is not request:
                        raise RuntimeError(
                            "GenAI returned a token for an unexpected request"
                        )
                    timestamps.append(time.perf_counter())
                    output.append(int(event.token))
    finally:
        request.close()
    if len(output) != generated:
        raise RuntimeError(
            f"Requested {generated} generated tokens but received {len(output)}; "
            "check context capacity and generation settings before comparing latency"
        )
    result = {
        "tokens": output,
        "ttft_seconds": timestamps[0] - start,
        "decode_seconds": timestamps[-1] - timestamps[0],
        "decode_intervals_seconds": [b - a for a, b in pairwise(timestamps)],
        "speculative_stats": engine.get_speculative_stats(),
        "max_draft_tokens_per_proposal": engine.max_draft_tokens_per_proposal(),
    }
    del request, engine
    gc.collect()
    return result


def validate_speculation(run):
    stats = run["speculative_stats"]
    if stats["mtp_failures"]:
        raise RuntimeError(
            f"MTP reported {stats['mtp_failures']} failures; inspect runtime logs "
            "and resolve draft execution errors before comparing performance"
        )
    if not (
        stats["draft_forward_passes"]
        and stats["draft_tokens_evaluated"]
        and stats["completed_rounds"]
    ):
        raise RuntimeError(
            "MTP did not complete draft verification rounds; "
            "check MTP configuration and increase --tokens before comparing performance"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ep", choices=["mlx", "webgpu"], required=True)
    parser.add_argument(
        "--library", type=Path, help="Development MLX EP dylib (required for mlx)"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--mtp", action="store_true", help="Require an enabled and active MTP drafter"
    )
    parser.add_argument(
        "--max-draft-tokens",
        type=int,
        choices=range(1, 17),
        help="Override the MTP draft limit without changing the exported model",
    )
    parser.add_argument(
        "--prompt",
        default="Explain how a compiler transforms source code into machine code.",
    )
    args = parser.parse_args()
    if args.tokens < 2 or args.warmup < 3 or args.repetitions < 1:
        parser.error("require tokens >= 2, warmup >= 3, and repetitions >= 1")
    if args.ep == "mlx" and (args.library is None or not args.library.is_file()):
        parser.error("--library must point to the development MLX EP dylib")
    if args.max_draft_tokens is not None and not args.mtp:
        parser.error("--max-draft-tokens requires --mtp")
    for name in ("ONNXRUNTIME_EP_MLX_TRACE", "ONNXRUNTIME_EP_MLX_GPU_CAPTURE"):
        if os.environ.get(name):
            parser.error(f"unset {name} before collecting final latency measurements")
    if not any(
        os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")
        for name in ("CI", "GITHUB_ACTIONS")
    ):
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    import numpy as np
    import onnxruntime_genai as og

    config_path = args.model / "genai_config.json"
    exported = json.loads(config_path.read_text())
    mtp_enabled = bool(exported.get("model", {}).get("mtp", {}).get("enabled"))
    if mtp_enabled != args.mtp:
        parser.error(
            "MTP export requires --mtp"
            if mtp_enabled
            else "--mtp requires an export with model.mtp.enabled=true; re-export "
            "with exclude_mtp=false prune_lm_head=false"
        )
    if args.ep == "mlx":
        library = args.library.resolve()
        provider = "MLXExecutionProvider"
    else:
        import onnxruntime_ep_webgpu as webgpu

        library = Path(webgpu.get_library_path())
        provider = webgpu.get_ep_name()
    og.register_execution_provider_library(provider, str(library))
    config = og.Config(str(args.model.resolve()))
    config.clear_providers()
    config.append_provider(provider)
    overlay = {}
    if args.max_draft_tokens is not None:
        overlay = {"speculative": {"max_draft_tokens": args.max_draft_tokens}}
        config.overlay(json.dumps(overlay))
    started = time.perf_counter()
    model = og.Model(config)
    construction_seconds = time.perf_counter() - started
    tokenizer = og.Tokenizer(model)
    prompt = tokenizer.apply_chat_template(
        messages=json.dumps([{"role": "user", "content": args.prompt}]),
        add_generation_prompt=True,
    )
    tokens = [int(token) for token in tokenizer.encode(prompt)]
    for warmup in range(args.warmup):
        run = run_once(og, model, tokens, args.tokens, require_mtp=args.mtp)
        if args.mtp:
            validate_speculation(run)
        print(
            json.dumps(
                {
                    "warmup": warmup + 1,
                    "ttft_seconds": run["ttft_seconds"],
                    "decode_seconds": run["decode_seconds"],
                }
            ),
            flush=True,
        )
    runs = []
    for _ in range(args.repetitions):
        run = run_once(og, model, tokens, args.tokens, require_mtp=args.mtp)
        if args.mtp:
            validate_speculation(run)
        runs.append(run)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in run.items()
                    if k
                    not in ("tokens", "decode_intervals_seconds", "speculative_stats")
                }
            ),
            flush=True,
        )
    if any(run["tokens"] != runs[0]["tokens"] for run in runs[1:]):
        raise RuntimeError(
            "Greedy tokens differ between repetitions; investigate correctness before comparing performance"
        )
    metrics = summarize(runs)
    metrics["decode_interval_p95_ms"] = float(
        np.percentile(
            [t * 1000 for run in runs for t in run["decode_intervals_seconds"]],
            95,
        )
    )
    report = {
        "provider": provider,
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "memory_bytes": int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        ),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in (
                "onnxruntime",
                "onnxruntime-genai",
                *(("onnxruntime-ep-webgpu",) if args.ep == "webgpu" else ()),
            )
        },
        "mlx_options": {
            name: os.environ[name]
            for name in (
                "ONNXRUNTIME_EP_MLX_NO_COMPILE",
                "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE",
                "ONNXRUNTIME_EP_MLX_NO_PREFILL_COMPILE",
                "ONNXRUNTIME_EP_MLX_LAYER_PARTITIONS",
            )
            if name in os.environ
        },
        "plugin_library": str(library),
        "model_directory": str(args.model.resolve()),
        "plugin_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "mode": "mtp" if args.mtp else "non_speculative",
        "runtime_overlay": overlay,
        "initializer_sharing": {
            name: bool(
                exported.get("model", {}).get(name, {}).get("shared_initializers")
            )
            for name in ("decoder", "mtp")
        },
        "prompt_tokens": list(tokens),
        "generated_tokens": args.tokens,
        "warmups": args.warmup,
        "repetitions": args.repetitions,
        "construction_seconds": construction_seconds,
        "runs": runs,
        **metrics,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
