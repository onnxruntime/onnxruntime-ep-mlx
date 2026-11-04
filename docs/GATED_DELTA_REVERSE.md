# Gated-delta scans

The provider recognizes the standard ONNX `Scan` data-flow used for gated-delta
pullbacks. It checks all six output expressions, static axes and the FP32 scale.
The optimized path executes the recurrence in a Metal kernel, followed by MLX
reductions of the independent value tiles.

Each thread owns a key row and four value columns. Its adjoint state stays in
registers across the time loop. SIMD reductions and small threadgroup scratch
compute the key dot products. The outputs are the final adjoint state and the
query, key, value, log-decay and beta gradients.

## State and replay scans

State-only scans compute checkpoint boundaries. Replay scans also return the
state before each update for the reverse recurrence. The provider recognizes
both forms and keeps four value columns per thread in registers across time.
SIMD reductions compute the key/state product once per time step and value tile.

For key `k`, value `v`, log-decay `g`, rate `b` and incoming state `S`, the update is

```text
F = S * exp(g[..., None])
r = v - k[..., None, :] @ F
S_next = F + b[..., None, None] * (k[..., :, None] * r[..., None, :])
```

The optional replay output is `S`, before the update. All arrays are FP32.
Key widths are 16, 32, 64 and 128; value widths are 1–128. Log-decay may be
per-head or per-key. Batch, head and positive sequence lengths are dynamic.
Sequence axes are zero, input directions agree, and output directions agree.
The independent input/output direction combinations are supported.

Matching checks the entire body: changed expressions and unrelated operations
retain generic Scan execution. Shapes and index bounds are checked at dispatch.
The trace kernel names are `mlx_gated_delta_state_scan` and
`mlx_gated_delta_replay_scan`.

## Reverse scan shapes and execution

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

## 4K checkpoint replay measurement

A real Qwen3.5-0.8B linear-attention block at 4,032 tokens ran in independent
processes on M1 Max. Each process performed 10 warmups and 100 measured calls.
The timed region is block recomputation and its VJP, including runtime array
exchange. Parameters are fixed; the other decoder blocks, loss and optimizer
run outside this measurement.

| State/replay scans | Median / P95 | Process peak physical footprint | MLX window peak |
|---|---:|---:|---:|
| Generic, current parent | 851.03 / 863.48 ms | 3.36 GiB | 3.43 GiB |
| Fused Metal | 280.25 / 285.59 ms | 3.99 GiB | 3.44 GiB |

The median throughput ratio is 3.04.
Both adapter gradients match the saved independent capture at
`rtol=1e-4, atol=1e-6`; maximum absolute differences are 1.19e-9 and 1.16e-8.
Inputs, exported model and gradient mappings are identical. ORT optimization,
CPU fallback and MLX compilation are disabled, with one CPU thread and a
512 MiB allocator cache.

A separate trace contains two executions. State and replay each dispatch 252
fused calls, alongside the existing 252 reverse calls. MatMul dispatches fall
from 16,162 to 34. These counts include qualification and one measured execution;
the timings above come from runs with tracing disabled.

[Raw samples, guard observations, hashes, validation logs and trace counts](benchmarks/gated-delta-state-m1max/)
record the parent and candidate binaries. The process peak includes imports,
initialization and the runtime; the allocator window is reset after warmup.

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
