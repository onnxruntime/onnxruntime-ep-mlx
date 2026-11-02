# Gated-delta reverse scans

The provider recognizes the standard ONNX `Scan` data-flow used for gated-delta
pullbacks. It checks all six output expressions, static axes and the FP32 scale.
The optimized path executes the recurrence in a Metal kernel, followed by MLX
reductions of the independent value tiles.

Each thread owns a key row and four value columns. Its adjoint state stays in
registers across the time loop. SIMD reductions and small threadgroup scratch
compute the key dot products. The outputs are the final adjoint state and the
query, key, value, log-decay and beta gradients.

## Shapes and execution

The kernel uses FP32 arrays, key widths 16, 32, 64 or 128, and value widths from
1 through 128. It supports positive dynamic batch, head and time dimensions,
scalar per-head log-decay, nonzero incoming adjoints and value tails.

| Input | Shape |
|---|---|
| Adjoint carry | `[B,H,K,V]` |
| Previous forward states | `[T,B,H,K,V]` |
| Query and key | `[T,B,H,K]` |
| Value and output cotangent | `[T,B,H,V]` |
| Log-decay | `[T,B,H,1]` |
| Beta | `[T,B,H]` |

Sequence axes are zero. Input sequences share a direction, and output sequences
share a direction; all four forward/reverse combinations are covered. Live scales,
per-key decay and other Scan forms use the generic implementation. Matching has
bounded graph and expression sizes, and runtime shapes are checked before dispatch.

Set `ONNXRUNTIME_EP_MLX_NO_GATED_DELTA_REVERSE=1` to select the generic Scan control.
The fast-path trace event is `op.fast` with kernel `mlx_gated_delta_reverse_scan`.
A matched body taking the generic path records its reason in `op.composed`.

## Real decoder measurement

[`bench/onnax_decoder.py`](../bench/onnax_decoder.py) measures a reused exported
onnax decoder bundle. The timed region includes forward execution, mean causal
token loss and all adapter gradients. Parameters stay fixed. Block execution
includes native host/device transfers; `input_snapshot_seconds` measures CPU
buffer cloning. The report records session creation separately from bundle
loading, median/P95 timings, process memory, library/source hashes and input and
adapter fingerprints.

Use an onnax checkout with its Hugging Face example dependencies and an exported
Qwen3.5 decoder from the
[onnax tutorial](https://www.justinchuby.com/onnax/decoder-families.html).
Run each mode in a fresh process:

```sh
PYTHONPATH=/path/to/onnax HF_HUB_OFFLINE=1 \
  /path/to/onnax/.venv/bin/python /path/to/onnxruntime-ep-mlx/bench/onnax_decoder.py \
  --export /path/to/run/export --cache-dir /path/to/model-cache \
  --library /path/to/libonnxruntime_mlx_ep.dylib --sequence 512 \
  --output /tmp/unrolled.json --gradients-output /tmp/unrolled.npz --unfused
```

Repeat with `/tmp/fused.json`, `/tmp/fused.npz`, and omit `--unfused`. Defaults are
10 warmup steps and 100 measured steps. The benchmark fixes MLX compilation off
and records its allocator cache limit. Both modes require full MLX node assignment
and disable ORT graph optimization. The named gradient snapshots support a
same-parameter numerical comparison alongside the timings.

For a macOS physical-memory budget, run the command through onnax's
`python -m examples.huggingface.guard --output-dir GUARD -- COMMAND ...`.

### M1 Max result

The pinned Qwen3.5-0.8B FP32 adapter at sequence length 512 produced:

| Reverse Scan | Forward/loss/all-gradients median / P95 | Process peak physical footprint |
|---|---:|---:|
| Generic unrolled | 5.261 / 6.652 s | 12.25 GiB |
| Fused Metal | 3.229 / 3.461 s | 11.05 GiB |

Each mode ran in its own process with 10 warmups and 100 samples. The same
binary, inputs, frozen source and 60 adapter tensors were used; the switch
above selects the generic control. All 60 final gradient arrays matched at
`rtol=1e-4, atol=1e-6`. The median throughput ratio for this forward/backward
workload is 1.63. Physical footprint includes model setup and the runtime.

The complete [generic](../bench/reports/qwen35-512-unrolled-m1max.json) and
[fused](../bench/reports/qwen35-512-fused-m1max.json) reports contain raw samples,
fingerprints, software versions and source/library hashes. The measured kernel
commit is `d7acb25`; the subsequent Rust lint fix changes axes decoding syntax.

The same binary completed 30 SFT updates on the pinned real model at sequence
length 64, microbatch one and four accumulation steps. Training loss fell from
5.223939 to 0.302991, and validation loss from 6.019480 to 5.621203. A fresh
process resumed checkpoint 15 through update 30. Its 61 checkpoint state tensors,
AdamW moments and scheduler state matched continuous training byte for byte.
Fresh loaded models also produced identical logits, loss and all 60 adapter
gradients.

The disk guard interrupted the resumed run during final export, after its
step-30 checkpoint was saved. After state equality was verified, the immutable
graph bundle was restored with hardlinks and each final checkpoint was loaded
strictly before execution. The [training/recovery record](../bench/reports/qwen35-fused-sft-recovery-m1max.json)
preserves this event, guard summaries and the verifier source.
