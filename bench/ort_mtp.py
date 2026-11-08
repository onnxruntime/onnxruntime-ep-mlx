"""Fixed-length greedy K=1 MTP through public ORT GPU OrtValues.

The target and draft exports are unchanged unless an explicit last-two-logit
target is supplied. This outer loop is not
GenAI Engine: its draft head starts cold from the last target prompt hidden row.
Only sampled token IDs are materialized on CPU. Prefix replay and sampling use
separate helper ONNX sessions.
"""

import argparse
import copy
import hashlib
import json
import os
import platform
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from bench.genai_paged import summarize
from bench.mtp_replay import make_checkpoint_model, make_prefix_replay_model
from bench.ort_paged import state_output

_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}

def make_last_two_logit_model(model):
    """Copy a target graph, preserving every one-/two-token verification logit."""
    result = copy.deepcopy(model)
    outputs = {value.name: value for value in result.graph.output}
    producers = [
        (index, node)
        for index, node in enumerate(result.graph.node)
        if "logits" in node.output
    ]
    if (
        not {"logits", "hidden_states"} <= outputs.keys()
        or len(producers) != 1
        or producers[0][1].op_type != "MatMulNBits"
        or producers[0][1].domain != "com.microsoft"
        or not producers[0][1].input
        or producers[0][1].input[0] != "hidden_states"
        or any("logits" in node.input for node in result.graph.node)
        or any(
            len(outputs[name].type.tensor_type.shape.dim) != 2
            for name in ("logits", "hidden_states")
        )
    ):
        raise ValueError(
            "Last-two-logit target requires rank-two logits and hidden_states with "
            "a direct com.microsoft::MatMulNBits vocabulary projection and no "
            "internal logits consumers; use a compatible unpruned target export"
        )
    names = {
        name for node in result.graph.node for name in (*node.input, *node.output)
    } | {value.name for value in (*result.graph.input, *result.graph.initializer)}
    prefix = "last_two_logits"
    while any(name.startswith(prefix) for name in names):
        prefix += "_"
    keys = [f"{prefix}.{name}" for name in ("starts", "ends", "axes", "steps")]
    for name, value in zip(
        keys, (-2, np.iinfo(np.int64).max, 0, 1), strict=True
    ):
        result.graph.initializer.append(
            numpy_helper.from_array(np.array([value], np.int64), name)
        )
    last_hidden = f"{prefix}.hidden"
    index, projection = producers[0]
    projection.input[0] = last_hidden
    result.graph.node.insert(
        index,
        helper.make_node("Slice", ["hidden_states", *keys], [last_hidden]),
    )
    for value in (*result.graph.output, *result.graph.value_info):
        if value.name == "logits":
            if len(value.type.tensor_type.shape.dim) != 2:
                raise ValueError(
                    "Last-two-logit target found inconsistent logits value metadata; "
                    "provide a target with rank-two logits throughout the graph"
                )
            value.type.tensor_type.shape.dim[0].dim_param = "logit_tokens"
    return result


def _shape(value, dimensions):
    try:
        return [d if isinstance(d, int) else dimensions[d] for d in value.shape]
    except KeyError as error:
        raise ValueError(
            f"{value.name}: unresolved shape {value.shape}; use a packed paged export "
            "with concrete head geometry and batch_size/num_tokens dimensions"
        ) from error


def _selector_model(logits, hidden):
    dtype = {
        "tensor(float)": TensorProto.FLOAT,
        "tensor(float16)": TensorProto.FLOAT16,
    }
    if logits.type not in dtype or hidden.type not in dtype:
        raise ValueError("GPU selection requires FP16/FP32 logits and hidden states")
    nodes = [
        helper.make_node("Gather", ["logits", "indices"], ["rows"], axis=0),
        helper.make_node("ArgMax", ["rows"], ["tokens"], axis=1, keepdims=0),
        helper.make_node(
            "Gather", ["hidden", "hidden_index"], ["selected_hidden"], axis=0
        ),
    ]
    return helper.make_model(
        helper.make_graph(
            nodes,
            "public_mtp_selection",
            [
                helper.make_tensor_value_info(
                    "logits", dtype[logits.type], ["rows", logits.shape[-1]]
                ),
                helper.make_tensor_value_info(
                    "hidden", dtype[hidden.type], ["hidden_rows", hidden.shape[-1]]
                ),
                helper.make_tensor_value_info(
                    "indices", TensorProto.INT64, ["selected"]
                ),
                helper.make_tensor_value_info("hidden_index", TensorProto.INT64, [1]),
            ],
            [
                helper.make_tensor_value_info(
                    "tokens", TensorProto.INT64, ["selected"]
                ),
                helper.make_tensor_value_info(
                    "selected_hidden", dtype[hidden.type], [1, hidden.shape[-1]]
                ),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )


def make_runner(
    target,
    draft,
    prompt,
    generated,
    *,
    info,
    blocks,
    block_size,
    prefill_chunk_size,
    capacity,
    speculative=True,
    last_two_logits=False,
):
    import onnxruntime as ort

    if (
        not len(prompt)
        or generated < 1
        or blocks < 1
        or block_size < 1
        or prefill_chunk_size < 1
        or capacity < 1
        or info is None
    ):
        raise ValueError(
            "Provide a nonempty prompt, positive generation/chunk/cache/capture "
            "sizes and public MLX device memory_info"
        )
    if len(prompt) + generated > blocks * block_size:
        raise ValueError(
            "Prompt plus generation exceeds paged capacity; increase blocks"
        )
    dimensions = {
        "batch_size": 1,
        "num_blocks": blocks,
        "block_size": block_size,
        "max_num_blocks": blocks,
    }
    target_inputs = target.get_inputs()
    target_outputs = {value.name: value for value in target.get_outputs()}
    draft_outputs = {value.name: value for value in draft.get_outputs()}
    if not {"logits", "hidden_states"} <= target_outputs.keys():
        raise ValueError(
            "Target must expose unchanged logits and hidden_states outputs"
        )
    if not {"logits", "hidden_states_out"} <= draft_outputs.keys():
        raise ValueError("Draft must expose logits and hidden_states_out outputs")
    for name, other in [
        ("logits", draft_outputs["logits"]),
        ("hidden_states", draft_outputs["hidden_states_out"]),
    ]:
        value = target_outputs[name]
        if (
            len(value.shape) != 2
            or len(other.shape) != 2
            or value.shape[-1] != other.shape[-1]
            or value.type != other.type
        ):
            raise ValueError(f"Target and draft {name} widths/dtypes must match")
    if last_two_logits != (target_outputs["logits"].shape[0] == "logit_tokens"):
        raise ValueError(
            "Last-two-logit execution requires a target with logits dimension "
            "'logit_tokens' and last_two_logits=True; build it with "
            "make_last_two_logit_model"
        )
    metadata = {
        "input_ids",
        "position_ids",
        "block_table",
        "cumulative_sequence_lengths",
        "past_sequence_lengths",
        "attention_metadata",
    }
    for session, extra in [
        (target, {"state_update_capture_count", "state_update_active"}),
        (draft, {"hidden_states"}),
    ]:
        names = {value.name for value in session.get_inputs()}
        unexpected = {
            name
            for name in names - metadata - extra
            if not name.startswith(("past.", "past_key_values."))
        }
        if unexpected or not metadata <= names or not extra <= names:
            raise ValueError(
                f"Unsupported packed MTP input contract: unexpected={sorted(unexpected)}, "
                f"missing={sorted((metadata | extra) - names)}; use a captured "
                "unpruned target and paged draft export"
            )
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    def helper_session(model):
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["MLXExecutionProvider"]
        )
        session.disable_fallback()
        return session

    checkpoint = helper_session(make_checkpoint_model(target_inputs))
    replay = helper_session(
        make_prefix_replay_model(
            target_inputs, list(target_outputs.values()), capacity=capacity
        )
    )
    selector = helper_session(
        _selector_model(target_outputs["logits"], target_outputs["hidden_states"])
    )

    def allocate(shape, dtype, *, zero=False):
        value = ort.OrtValue.ortvalue_from_shape_and_type(
            shape, dtype, memory_info=info
        )
        if zero:
            value.update_inplace(np.zeros(shape, dtype))
        return value

    bindings, states, buffers, persistent_outputs = {}, {}, {}, {}
    zero_buffers, resets = {}, []
    for name, session in [("target", target), ("draft", draft)]:
        binding = session.io_binding()
        bindings[name] = binding
        states[name], buffers[name] = {}, {}
        persistent_outputs[name] = {}
        for value in session.get_inputs():
            if not value.name.startswith(("past.", "past_key_values.")):
                continue
            state = allocate(_shape(value, dimensions), _DTYPES[value.type], zero=True)
            if value.name.startswith("past."):
                key = (tuple(_shape(value, dimensions)), _DTYPES[value.type])
                if key not in zero_buffers:
                    zero_buffers[key] = np.zeros(key[0], key[1])
                resets.append((state, zero_buffers[key]))
            states[name][value.name] = state
            binding.bind_ortvalue_input(value.name, state)
            binding.bind_ortvalue_output(state_output(value.name), state)
            persistent_outputs[name][state_output(value.name)] = state
        binding.bind_cpu_input("block_table", np.arange(blocks, dtype=np.int32)[None])
    checkpoint_binding, replay_binding = checkpoint.io_binding(), replay.io_binding()
    snapshots = {}
    for value in checkpoint.get_outputs():
        original_name = value.name.removeprefix("checkpoint.")
        snapshot = allocate(_shape(value, dimensions), _DTYPES[value.type])
        snapshots[original_name] = snapshot
        checkpoint_binding.bind_ortvalue_input(
            original_name, states["target"][original_name]
        )
        checkpoint_binding.bind_ortvalue_output(value.name, snapshot)
        replay_binding.bind_ortvalue_input(original_name, snapshot)
        replay_binding.bind_ortvalue_output(
            state_output(original_name), states["target"][original_name]
        )
    for value in replay.get_inputs():
        if not value.name.startswith("state_update."):
            continue
        capture = allocate(_shape(value, dimensions), _DTYPES[value.type])
        replay_binding.bind_ortvalue_input(value.name, capture)
        bindings["target"].bind_ortvalue_output(value.name, capture)
        persistent_outputs["target"][value.name] = capture
    selection_binding = selector.io_binding()
    selected_hidden = allocate(
        [1, target_outputs["hidden_states"].shape[-1]],
        _DTYPES[target_outputs["hidden_states"].type],
    )
    selection_binding.bind_ortvalue_output("selected_hidden", selected_hidden)
    cpu_tokens = {
        n: ort.OrtValue.ortvalue_from_shape_and_type([n], np.int64) for n in (1, 2)
    }

    def execute(name, session, ids, past, position, *, hidden=None, capture=False):
        n = len(ids)
        if past + n > blocks * block_size:
            raise ValueError(
                f"{name}: cache capacity exceeded at past={past}, tokens={n}"
            )
        binding = bindings[name]
        feeds = {
            "input_ids": np.asarray(ids, np.int64),
            "position_ids": np.tile(
                np.arange(position, position + n, dtype=np.int64), (3, 1)
            ),
            "cumulative_sequence_lengths": np.array([0, n], np.int32),
            "past_sequence_lengths": np.array([past], np.int32),
            "attention_metadata": np.array([n, past + n, 0], np.int32),
        }
        if name == "target":
            feeds["state_update_capture_count"] = np.array([int(capture)], np.int32)
            feeds["state_update_active"] = np.array([int(capture)], np.int32)
        else:
            binding.bind_ortvalue_input("hidden_states", hidden)
        for key, value in feeds.items():
            binding.bind_cpu_input(key, value)
        if n not in buffers[name]:
            buffers[name][n] = {
                value.name: allocate(
                    _shape(value, {
                        **dimensions, "num_tokens": n, "logit_tokens": min(n, 2)
                    }), _DTYPES[value.type]
                )
                for value in session.get_outputs()
                if not value.name.startswith(("present.", "state_update."))
            }
        for key, value in buffers[name][n].items():
            binding.bind_ortvalue_output(key, value)
        session.run_with_iobinding(binding)
        return buffers[name][n]

    def select(output, rows, hidden_row, *, draft=False):
        selection_binding.bind_ortvalue_input("logits", output["logits"])
        selection_binding.bind_ortvalue_input(
            "hidden", output["hidden_states_out" if draft else "hidden_states"]
        )
        selection_binding.bind_cpu_input("indices", np.array(rows, np.int64))
        selection_binding.bind_cpu_input(
            "hidden_index", np.array([hidden_row], np.int64)
        )
        selection_binding.bind_ortvalue_output("tokens", cpu_tokens[len(rows)])
        selector.run_with_iobinding(selection_binding)
        return cpu_tokens[len(rows)].numpy().tolist()

    first_run = True

    def generate(prompt, generated):
        nonlocal first_run
        if not len(prompt) or generated < 1:
            raise ValueError("Provide a nonempty prompt and positive generation size")
        if len(prompt) + generated > blocks * block_size:
            raise ValueError(
                "Prompt plus generation exceeds paged capacity; increase blocks"
            )
        if not first_run:
            # KV validity is reset by past=0; future slots are overwritten before
            # use. Clearing full KV tensors would allocate new device arrays.
            for value, zero in resets:
                value.update_inplace(zero)
        first_run = False
        tokens, times = [], []
        started = time.perf_counter()
        past = 0
        for start in range(0, len(prompt), prefill_chunk_size):
            chunk = prompt[start : start + prefill_chunk_size]
            output = execute("target", target, chunk, past, past)
            past += len(chunk)
        anchor = int(select(
            output, [
                min(len(chunk), 2) - 1 if last_two_logits else len(chunk) - 1
            ], len(chunk) - 1
        )[0])
        del output
        selection_binding.clear_binding_inputs()
        bindings["target"].clear_binding_outputs()
        for name, state in persistent_outputs["target"].items():
            bindings["target"].bind_ortvalue_output(name, state)
        for n in list(buffers["target"]):
            if n > 2:
                del buffers["target"][n]
        tokens.append(anchor)
        times.append(time.perf_counter())
        pending_hidden, head_tokens, head_past = selected_hidden, [anchor], 0
        rounds = accepted = rejections = draft_input_tokens = 0
        while len(tokens) < generated:
            if not speculative:
                output = execute("target", target, [anchor], past, past)
                anchor = int(select(output, [0], 0)[0])
                past += 1
                tokens.append(anchor)
                times.append(time.perf_counter())
                continue
            # Re-feed accepted drafts with target hidden rows, then the bonus
            # anchor. K=1 never writes speculative drafts into the head KV.
            draft_output = execute(
                "draft",
                draft,
                head_tokens,
                head_past,
                past - len(head_tokens) + 1,
                hidden=pending_hidden,
            )
            head_past += len(head_tokens)
            draft_input_tokens += len(head_tokens)
            proposal = int(
                select(
                    draft_output,
                    [len(head_tokens) - 1],
                    len(head_tokens) - 1,
                    draft=True,
                )[0]
            )
            checkpoint.run_with_iobinding(checkpoint_binding)
            output = execute(
                "target", target, [anchor, proposal], past, past, capture=True
            )
            prediction, bonus = map(int, select(output, [0, 1], 0))
            rounds += 1
            if proposal == prediction:
                accepted += 1
                past += 2
                pending_hidden = output["hidden_states"]
                head_tokens = [proposal, bonus]
                emitted = [proposal, bonus]
                anchor = bonus
            else:
                rejections += 1
                replay.run_with_iobinding(replay_binding)
                past += 1
                pending_hidden = selected_hidden
                head_tokens = [prediction]
                emitted = [prediction]
                anchor = prediction
            now = time.perf_counter()
            for token in emitted[: generated - len(tokens)]:
                tokens.append(token)
                times.append(now)
        return {
            "tokens": tokens,
            "ttft_seconds": times[0] - started,
            "decode_seconds": times[-1] - times[0],
            "decode_intervals_seconds": [b - a for a, b in pairwise(times)],
            "speculative_stats": {
                "completed_rounds": rounds,
                "draft_forward_passes": rounds,
                "draft_input_tokens": draft_input_tokens,
                "draft_tokens_proposed": rounds,
                "draft_tokens_evaluated": rounds,
                "draft_tokens_accepted": accepted,
                "prefix_replays": rejections,
            },
        }

    return generate


def run_once(target, draft, prompt, generated, **options):
    return make_runner(target, draft, prompt, generated, **options)(prompt, generated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=70)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument(
        "--target-last-two-logits", type=Path,
        help="Explicit last-two-logit target graph; K1 verification/decode retain all rows",
    )
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Run a non-speculative control on the same unpruned target export",
    )
    args = parser.parse_args()
    if args.tokens < 2 or args.warmup < 3 or args.repetitions < 1:
        parser.error("require tokens >= 2, warmup >= 3 and repetitions >= 1")
    if min(args.blocks, args.block_size, args.prefill_chunk_size, args.capacity) < 1:
        parser.error("cache, prefill chunk and capture capacity must be positive")
    if any(
        os.environ.get(k)
        for k in ("ONNXRUNTIME_EP_MLX_TRACE", "ONNXRUNTIME_EP_MLX_GPU_CAPTURE")
    ):
        parser.error("unset tracing/capture before final measurements")
    os.environ["ONNXRUNTIME_EP_MLX_DEVICE_TENSORS"] = "1"
    if not any(
        os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")
        for name in ("CI", "GITHUB_ACTIONS")
    ):
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    import onnxruntime as ort

    reference = json.loads(args.reference.read_text())
    source_sha256 = {
        f"bench/{name}": hashlib.sha256(
            Path(__file__).with_name(name).read_bytes()
        ).hexdigest()
        for name in ("ort_mtp.py", "mtp_replay.py", "ort_paged.py", "genai_paged.py")
    }
    prompt = reference["prompt_tokens"]
    expected = reference["runs"][0]["tokens"][: args.tokens]
    if not len(prompt) or len(expected) != args.tokens:
        parser.error(
            "reference must contain a nonempty prompt and --tokens output tokens"
        )
    if len(prompt) + args.tokens > args.blocks * args.block_size:
        parser.error("prompt plus generation exceeds paged capacity; increase --blocks")
    ort.register_execution_provider_library(
        "MLXExecutionProvider", str(args.library.resolve())
    )
    device = next(
        (d for d in ort.get_ep_devices() if d.ep_name == "MLXExecutionProvider"),
        None,
    )
    if device is None:
        parser.error(
            f"Registered {args.library} but no MLX device is available; verify "
            "the plugin/runtime ABI and Metal availability before retrying"
        )
    info = device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    started = time.perf_counter()
    target = ort.InferenceSession(
        str(args.target_last_two_logits or args.model / "model.onnx"),
        options, providers=["MLXExecutionProvider"]
    )
    draft = ort.InferenceSession(
        str(args.model / "mtp.onnx"), options, providers=["MLXExecutionProvider"]
    )
    target.disable_fallback()
    draft.disable_fallback()
    construction = time.perf_counter() - started
    started = time.perf_counter()
    runner = make_runner(
        target,
        draft,
        prompt,
        args.tokens,
        info=info,
        blocks=args.blocks,
        block_size=args.block_size,
        prefill_chunk_size=args.prefill_chunk_size,
        capacity=args.capacity,
        speculative=not args.target_only,
        last_two_logits=args.target_last_two_logits is not None,
    )
    setup = time.perf_counter() - started
    runs = []
    for i in range(args.warmup + args.repetitions):
        result = runner(prompt, args.tokens)
        if result["tokens"] != expected:
            mismatch = next(
                j
                for j, (a, b) in enumerate(zip(result["tokens"], expected, strict=True))
                if a != b
            )
            raise RuntimeError(
                f"Greedy reference mismatch at generated token {mismatch}: "
                f"actual={result['tokens'][mismatch]}, expected={expected[mismatch]}; "
                "investigate numerical/state alignment before comparing throughput"
            )
        print(
            json.dumps(
                {
                    "warmup": i < args.warmup,
                    "ttft_seconds": result["ttft_seconds"],
                    "decode_seconds": result["decode_seconds"],
                    "speculative_stats": result["speculative_stats"],
                }
            ),
            flush=True,
        )
        if i >= args.warmup:
            runs.append(result)
    report = {
        "driver": (
            "Public ORT GPU IOBinding target-only control on the unchanged export"
            if args.target_only
            else "Public ORT GPU IOBinding K=1 MTP; separate checkpoint/replay/"
            "greedy-selection helpers, not GenAI Engine"
        ) + (
            "; explicit last-two-logit target"
            if args.target_last_two_logits is not None else ""
        ),
        "target_last_two_logits": (
            None if args.target_last_two_logits is None else {
                "path": str(args.target_last_two_logits.resolve()),
                "sha256": hashlib.sha256(args.target_last_two_logits.read_bytes()).hexdigest(),
                "scope": "Model-native last-two-logit target rewrite; one-token "
                "decode and two-token K1 verification retain every original row",
            }
        ),
        "draft_priming": (
            None
            if args.target_only
            else "Cold head, starting from the final target prompt hidden row; "
            "accepted drafts are re-fed with target hidden rows"
        ),
        "cpu_materialization": "Sampled token IDs only; state, captures, hidden states and graph-exposed logits remain GPU-owned",
        "prompt_tokens": prompt,
        "generated_tokens": args.tokens,
        "warmups": args.warmup,
        "repetitions": args.repetitions,
        "blocks": args.blocks,
        "block_size": args.block_size,
        "prefill_chunk_size": args.prefill_chunk_size,
        "capture_capacity": args.capacity,
        "max_draft_tokens": 0 if args.target_only else 1,
        "model_directory": str(args.model.resolve()),
        "model_graph_sha256": {
            name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
            for name in ("model.onnx", "mtp.onnx")
        },
        "plugin_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "python_source_sha256": source_sha256,
        "mlx_options": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("ONNXRUNTIME_EP_MLX_")
        },
        "onnxruntime": ort.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model_construction_seconds": construction,
        "runtime_setup_seconds": setup,
        "memory_pattern_enabled": False,
        "kv_reset": "Logical invalidation with past=0; full KV buffers are not physically cleared between requests",
        "timing_scope": "One-time construction/helper/buffer setup and per-request recurrent-state resets excluded; TTFT includes all target prefill and first greedy selection. Decode counts emitted tokens, including same-round bursts.",
        "runs": runs,
        **summarize(runs),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("warm_ttft_median_ms", "aggregate_decode_tokens_per_second")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
