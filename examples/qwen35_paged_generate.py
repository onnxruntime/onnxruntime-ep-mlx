#!/usr/bin/env python3
"""Benchmark a Qwen3.5 paged ONNX model with the MLX, WebGPU, or CPU EP."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="GenAI model directory containing genai_config.json and model.onnx",
    )
    parser.add_argument(
        "--ep",
        choices=("mlx", "webgpu", "cpu"),
        default="mlx",
        help="Execution provider to benchmark (overrides the model configuration)",
    )
    parser.add_argument(
        "--ep-library",
        type=Path,
        help="Plugin library; defaults to ONNXRUNTIME_MLX_EP_LIB or ONNXRUNTIME_WEBGPU_EP_LIB",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        help="Write provider, timings, and generated token IDs to a JSON file",
    )
    parser.add_argument(
        "--ort-profile-prefix",
        type=Path,
        help="Enable ORT session profiling with this output filename prefix",
    )
    parser.add_argument(
        "--prompt",
        default="Explain briefly why paged attention is useful for language model inference.",
        help="User prompt passed through the model's chat template",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=8,
        help="Tokens generated in one untimed warmup per prompt length; use 0 to disable",
    )
    parser.add_argument(
        "--prompt-repeats",
        type=int,
        nargs="+",
        default=[1, 4, 16],
        help="Repeat the prompt text this many times to benchmark several input lengths",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Measured runs per prompt length",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Skip the tokenizer chat template and pass --prompt as-is",
    )
    parser.add_argument(
        "--print-output",
        action="store_true",
        help="Decode and print generated text after each timed run",
    )
    parser.add_argument(
        "--mlx-verbose",
        action="store_true",
        help="Enable MLX EP claim/summary diagnostics before loading the provider",
    )
    return parser.parse_args()


def validate_model_dir(model_dir: Path) -> dict:
    config_path = model_dir / "genai_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing GenAI config: {config_path}")
    for filename in ("model.onnx", "tokenizer.json"):
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing model artifact: {path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    dynamic_batching = config.get("engine", {}).get("dynamic_batching", {})
    num_blocks = dynamic_batching.get("num_blocks")
    block_size = dynamic_batching.get("block_size")
    if not isinstance(num_blocks, int) or num_blocks <= 0:
        raise ValueError("engine.dynamic_batching.num_blocks must be a positive integer")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("engine.dynamic_batching.block_size must be a positive integer")
    config["_paged_cache_token_capacity"] = num_blocks * block_size
    return config


def configure_provider(og, model_dir: Path, ep: str, ep_library: Path | None):
    if ep != "cpu":
        env_name = f"ONNXRUNTIME_{ep.upper()}_EP_LIB"
        if ep_library is None:
            library_env = os.environ.get(env_name)
            ep_library = Path(library_env) if library_env else None
        if ep_library is None:
            raise ValueError(f"Pass --ep-library or set {env_name} for --ep {ep}")
        ep_library = ep_library.expanduser().resolve()
        if not ep_library.is_file():
            raise FileNotFoundError(f"EP library not found: {ep_library}")
        provider_name = "MLXExecutionProvider" if ep == "mlx" else "WebGpuExecutionProvider"
        og.register_execution_provider_library(provider_name, str(ep_library))
    elif ep_library is not None:
        raise ValueError("--ep-library is not used with --ep cpu")

    config = og.Config(str(model_dir))
    config.clear_providers()
    if ep != "cpu":
        config.append_provider("MLXExecutionProvider" if ep == "mlx" else "webgpu")
    config.overlay(json.dumps({"engine": {"dynamic_batching": {"prefix_caching": False}}}))
    return config


def generate_once(og, engine, input_tokens, max_new_tokens):
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(len(input_tokens) + max_new_tokens)
    request = engine.create_request(options=request_options)
    turn_options = og.TurnOptions(request)
    turn_options.set_max_generated_tokens(max_new_tokens)
    turn_options.set_do_sample(False)

    generated_tokens = []
    token_times = []
    event_buffer = engine.create_event_buffer(1)
    started = time.perf_counter()
    request.begin_turn(np.asarray(input_tokens, dtype=np.int32), turn_options)
    while engine.has_pending_requests():
        for event in engine.run(event_buffer):
            if event.flags & og.EngineEventFlags.FAILED:
                raise RuntimeError(f"Generation failed; error_code={event.error_code}")
            if event.flags & og.EngineEventFlags.CAPACITY_BLOCKED:
                raise RuntimeError(f"Generation capacity blocked; error_code={event.error_code}")
            if event.request is not request:
                raise RuntimeError(f"Engine returned an unexpected request; error_code={event.error_code}")
            if event.flags & og.EngineEventFlags.TOKEN:
                generated_tokens.append(int(event.token))
                token_times.append(time.perf_counter())
            if event.flags & og.EngineEventFlags.TURN_FINISHED:
                event.request.close()

    finished = time.perf_counter()
    ttft = token_times[0] - started if token_times else None
    decode_seconds = token_times[-1] - token_times[0] if len(token_times) > 1 else None
    decode_tokens_per_second = (
        (len(token_times) - 1) / decode_seconds
        if decode_seconds is not None and decode_seconds > 0
        else None
    )
    return {
        "tokens": generated_tokens,
        "ttft": ttft,
        "decode_tokens_per_second": decode_tokens_per_second,
        "total_seconds": finished - started,
    }


def main() -> int:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.warmup_tokens < 0:
        raise ValueError("--warmup-tokens must be non-negative")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if not args.prompt_repeats or any(repeats <= 0 for repeats in args.prompt_repeats):
        raise ValueError("--prompt-repeats values must be positive")
    model_dir = args.model_dir.expanduser().resolve()

    config_data = validate_model_dir(model_dir)
    if args.mlx_verbose:
        os.environ["ONNXRUNTIME_EP_MLX_VERBOSE"] = "1"
        os.environ["ONNXRUNTIME_EP_MLX_CLAIM_DEBUG"] = "1"

    try:
        import onnxruntime_genai as og
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime_genai is not installed in this Python environment. Install or build "
            "the ONNX Runtime GenAI Python package in the selected environment, then rerun."
        ) from error

    config = configure_provider(og, model_dir, args.ep, args.ep_library)
    if args.ort_profile_prefix is not None:
        config.overlay(json.dumps({
            "model": {"decoder": {"session_options": {
                "enable_profiling": str(args.ort_profile_prefix.expanduser().resolve()),
            }}},
        }))

    model_started = time.perf_counter()
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    model_load_seconds = time.perf_counter() - model_started
    engine_started = time.perf_counter()
    engine = og.Engine(model)
    engine_init_seconds = time.perf_counter() - engine_started
    model_config = config_data["model"]
    context_capacity = model_config.get("context_length")
    cache_capacity = config_data["_paged_cache_token_capacity"]

    report = {
        "ep": args.ep,
        "model_dir": str(model_dir),
        "genai_version": og.__version__,
        "prefix_caching": False,
        "model_load_seconds": model_load_seconds,
        "engine_init_seconds": engine_init_seconds,
        "max_new_tokens": args.max_new_tokens,
        "warmup_tokens": args.warmup_tokens,
        "prompts": [],
    }
    print(f"Provider: {args.ep}; prefix caching disabled")
    print(f"Model/tokenizer load: {model_load_seconds:.2f}s | Engine init: {engine_init_seconds:.2f}s")
    print(f"Measured output: {args.max_new_tokens} tokens | Runs per prompt: {args.runs}")
    for repeats in args.prompt_repeats:
        content = " ".join([args.prompt] * repeats)
        if args.raw_prompt:
            prompt = content
        else:
            prompt = tokenizer.apply_chat_template(
                messages=json.dumps([{"role": "user", "content": content}]),
                add_generation_prompt=True,
            )
        input_tokens = [int(token) for token in tokenizer.encode(prompt)]
        input_token_count = len(input_tokens)
        if input_token_count == 0:
            raise ValueError("The prompt encoded to zero tokens")

        max_requested_tokens = input_token_count + max(args.max_new_tokens, args.warmup_tokens)
        if context_capacity and max_requested_tokens > context_capacity:
            raise ValueError(
                f"Prompt plus requested output ({max_requested_tokens} tokens) exceeds model context "
                f"({context_capacity})"
            )
        if max_requested_tokens > cache_capacity:
            raise ValueError(
                f"Prompt plus requested output ({max_requested_tokens} tokens) exceeds configured paged KV "
                f"capacity ({cache_capacity} tokens = num_blocks * block_size)"
            )

        print(f"\nPrompt repeats: {repeats}; prompt tokens: {input_token_count}")
        if args.warmup_tokens:
            warmup = generate_once(og, engine, input_tokens, args.warmup_tokens)
            print(f"Warmup: generated {len(warmup['tokens'])} tokens (not timed)")

        results = []
        for run_index in range(args.runs):
            result = generate_once(og, engine, input_tokens, args.max_new_tokens)
            results.append(result)
            ttft = f"{result['ttft']:.3f}s" if result["ttft"] is not None else "n/a"
            decode_rate = (
                f"{result['decode_tokens_per_second']:.2f} tok/s"
                if result["decode_tokens_per_second"] is not None
                else "n/a"
            )
            print(
                f"Run {run_index + 1}: TTFT {ttft} | steady decode {decode_rate} "
                f"| total {result['total_seconds']:.2f}s | generated {len(result['tokens'])}"
            )
            if args.print_output:
                print(f"Output: {tokenizer.decode(np.asarray(result['tokens'], dtype=np.int32))}")

        ttfts = [result["ttft"] for result in results if result["ttft"] is not None]
        decode_rates = [
            result["decode_tokens_per_second"]
            for result in results
            if result["decode_tokens_per_second"] is not None
        ]
        if args.runs > 1:
            median_ttft = f"{statistics.median(ttfts):.3f}s" if ttfts else "n/a"
            median_decode = f"{statistics.median(decode_rates):.2f} tok/s" if decode_rates else "n/a"
            print(f"Median: TTFT {median_ttft} | steady decode {median_decode}")
        report["prompts"].append({
            "prompt_repeats": repeats,
            "input_tokens": list(input_tokens),
            "runs": results,
            "median_ttft_seconds": statistics.median(ttfts) if ttfts else None,
            "median_decode_tokens_per_second": statistics.median(decode_rates) if decode_rates else None,
        })
    if args.results_json is not None:
        args.results_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Results written to {args.results_json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
