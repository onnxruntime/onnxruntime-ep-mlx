"""Measure decoder forward, token loss and adapter gradients with fixed parameters.

Run with the onnax checkout on PYTHONPATH and its Hugging Face example dependencies.
Each invocation loads one exported decoder bundle and reuses its runtime sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version


def git_state(directory):
    result = subprocess.run(
        ["git", "-C", directory, "rev-parse", "HEAD"], capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "-C", directory, "status", "--porcelain"], capture_output=True, text=True
    )
    return {
        "commit": result.stdout.strip() if result.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def digest_arrays(arrays):
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gradients-output")
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--unfused", action="store_true")
    args = parser.parse_args()
    if args.sequence < 2 or args.batch < 1 or args.warmup < 0 or args.samples < 1:
        parser.error("Require sequence >= 2, positive batch/samples and nonnegative warmup")
    os.environ["ONNXRUNTIME_EP_MLX_NO_GATED_DELTA_REVERSE"] = "1" if args.unfused else "0"
    os.environ["ONNXRUNTIME_EP_MLX_NO_COMPILE"] = "1"
    os.environ.setdefault("ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES", "536870912")
    import numpy as np
    import torch

    from examples.huggingface.decoder_sft import OnnxDecoder
    from examples.huggingface.pretrained import PretrainedSource, load_tokenizer
    from examples.huggingface.sft import tokenizer
    from examples.lora.mlx_runtime import register_library, session
    from examples.process_memory import physical_footprint

    torch.set_num_threads(1)
    register_library(args.library)
    start = time.perf_counter()
    session_times = []

    def timed_session(graph):
        before = time.perf_counter()
        result = session(graph)
        session_times.append(time.perf_counter() - before)
        return result

    model = OnnxDecoder.load(args.export, session_factory=timed_session)
    construction = time.perf_counter() - start
    try:
        if args.sequence > model.config.max_position_embeddings:
            parser.error("Sequence exceeds the source model's position limit")
        provenance = getattr(model.config, "onnax_pretrained", None)
        if provenance:
            source = PretrainedSource(
                provenance["model_id"],
                provenance["revision"],
                tuple(provenance["checkpoint_to_reference_prefix"].items()),
            )
            tok = load_tokenizer(source, cache_dir=args.cache_dir)
        else:
            tok = tokenizer()
        text = "ONNX models learn from examples. Low-rank adapters update a small set of parameters. "
        tokens = tok.encode(text, add_special_tokens=False)
        if not tokens:
            raise ValueError("Benchmark text produced no tokens")
        repetitions = max(1, (args.sequence + len(tokens) - 1) // len(tokens))
        while len(tokens) < args.sequence:
            tokens = tok.encode(text * repetitions, add_special_tokens=False)
            repetitions *= 2
        ids = torch.tensor([tokens[: args.sequence]], dtype=torch.int64).repeat(args.batch, 1)
        data = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "position_ids": torch.arange(args.sequence)[None].repeat(args.batch, 1),
            "labels": ids.clone(),
        }
        original_hash = digest_arrays(model.onnx.arrays())
        report = {
            "version": 1,
            "mode": "unrolled" if args.unfused else "fused",
            "scope": "forward, mean shifted-token loss and all adapter gradients; fixed parameters",
            "component_scope": "block times include native host/device transfers; input_snapshot times CPU buffer cloning",
            "input_shape": list(ids.shape),
            "prediction_targets_per_step": args.batch * (args.sequence - 1),
            "source_identity": model.onnx._identity,
            "adapter_sha256": original_hash,
            "input_sha256": digest_arrays(
                {name: value.numpy() for name, value in data.items()}
            ),
            "pretrained_source": provenance,
            "checkpoint_boundaries": model.onnx.boundaries,
            "bundle_load_and_sessions_seconds": construction,
            "session_creation_seconds": sum(session_times),
            "session_creation_samples_seconds": session_times,
            "warmup_steps": args.warmup,
            "measured_steps": args.samples,
            "samples": [],
            "platform": platform.platform(),
            "python": platform.python_version(),
            "native_source": git_state(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "onnax_source": git_state(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(sys.modules[OnnxDecoder.__module__].__file__)
                    )
                )
            ),
            "versions": {
                name: version(name)
                for name in (
                    "onnax",
                    "onnx",
                    "onnxruntime",
                    "torch",
                    "transformers",
                    "mobius-onnx",
                )
            },
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "ONNXRUNTIME_EP_MLX_NO_COMPILE",
                    "ONNXRUNTIME_EP_MLX_NO_GATED_DELTA_REVERSE",
                    "ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES",
                )
            },
            "torch_threads": torch.get_num_threads(),
        }
        opts = model.onnx._sessions[0].get_session_options()
        report["ort"] = {
            "intra_threads": opts.intra_op_num_threads,
            "inter_threads": opts.inter_op_num_threads,
            "graph_optimization": str(opts.graph_optimization_level),
            "cpu_fallback_disabled": opts.get_session_config_entry(
                "session.disable_cpu_ep_fallback"
            )
            == "1",
        }
        report["adapter_bytes"] = sum(value.nbytes for value in model.onnx.arrays().values())
        report["export_bundle_bytes"] = sum(
            os.path.getsize(os.path.join(root, name))
            for root, _, names in os.walk(args.export)
            for name in names
        )
        import onnx

        report["primal_graph_nodes"] = len(
            onnx.load(
                os.path.join(args.export, "execution", "model.onnx"), load_external_data=False
            ).graph.node
        )
        report["status"] = "running"
        with open(args.library, "rb") as stream:
            report["library_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        with open(__file__, "rb") as stream:
            report["benchmark_source_sha256"] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

        def save():
            temporary = args.output + ".tmp"
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
            os.replace(temporary, args.output)

        save()
        for index in range(args.warmup + args.samples):
            model.zero_grad(set_to_none=True)
            before = time.perf_counter()
            output = model(**data)
            after_forward = time.perf_counter()
            output.loss.backward()
            after_backward = time.perf_counter()
            if not torch.isfinite(output.loss) or any(
                p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()
            ):
                raise FloatingPointError(
                    "Nonfinite loss or missing/nonfinite adapter gradient"
                )
            if index >= args.warmup:
                report["samples"].append(
                    {
                        "gradient_step_seconds": after_backward - before,
                        "task_forward_seconds": after_forward - before,
                        "task_backward_seconds": after_backward - after_forward,
                        "block_forward_seconds": model.onnx.timings["forward"][-1],
                        "block_backward_seconds": model.onnx.timings["backward"][-1],
                        "input_snapshot_seconds": model.onnx.timings["exchange"][-1],
                        "loss": float(output.loss.detach()),
                    }
                )
                report["physical_memory"] = physical_footprint()
                save()
                if len(report["samples"]) % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "mode": report["mode"],
                                "samples": len(report["samples"]),
                                "last_step_seconds": report["samples"][-1][
                                    "gradient_step_seconds"
                                ],
                            }
                        ),
                        flush=True,
                    )
            del output
        report["summary"] = {
            key: {
                "median": float(np.median([row[key] for row in report["samples"]])),
                "p95": float(np.percentile([row[key] for row in report["samples"]], 95)),
            }
            for key in report["samples"][0]
            if key.endswith("seconds")
        }
        report["parameters_unchanged"] = digest_arrays(model.onnx.arrays()) == original_hash
        if not report["parameters_unchanged"]:
            raise AssertionError("Benchmark changed adapter parameters")
        if args.gradients_output:
            os.makedirs(os.path.dirname(os.path.abspath(args.gradients_output)), exist_ok=True)
            np.savez(
                args.gradients_output,
                **{
                    name: parameter.grad.detach().numpy()
                    for name, parameter in model.onnx.parameters_by_name().items()
                },
            )
        import resource

        report["process_peak_rss_bytes"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        report["status"] = "complete"
        save()
        print(
            json.dumps(
                {
                    "mode": report["mode"],
                    "summary": report["summary"],
                    "physical_memory": report["physical_memory"],
                }
            ),
            flush=True,
        )
    finally:
        model.close()


if __name__ == "__main__":
    main()
