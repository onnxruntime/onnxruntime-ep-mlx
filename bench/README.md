# MLX EP op benchmark

A small perf-regression harness for the MLX execution provider. It benchmarks a
fixed set of tiny, self-contained ONNX graphs — one per op family the EP claims
(GEMM, MatMulNBits, LayerNorm/RMSNorm, GroupQueryAttention, Softmax, Gelu,
elementwise, Conv) — through the real ORT + MLX path (`session.run`), and reports
the median wall-clock time.

The graphs reuse the exact builders the correctness tests use
(`tests/ops/_models.py`), so the benchmark can't drift from what's tested.

## Run locally

```sh
ORT_LIB=<dir containing libonnxruntime.dylib>
DYLD_LIBRARY_PATH=$ORT_LIB \
ONNXRUNTIME_MLX_EP_LIB=rust/target/release/libonnxruntime_mlx_ep.dylib \
  python bench/bench.py --out pr.json --label PR
```

Each line prints `mlx=…ms cpu=…ms Nx` and flags any case that falls back to CPU
(`⚠ fallback`). Sub-millisecond cases (norm/softmax/elementwise) are dispatch-bound
and slower than CPU — that's expected; the point is to detect *changes* over time,
and to catch an op silently dropping out of the claimed set.

## Compare two builds

```sh
python bench/compare.py --base base.json --pr pr.json --threshold 10 > comment.md
```

Emits a Markdown table (regressions 🔴, improvements 🟢, new CPU fallbacks ⛔).
Timing flags are threshold-based observations, not proof of a performance change.

## CI

`.github/workflows/bench.yml` is **manual-only**. In Actions, select the **Bench**
workflow, choose the branch to measure, and use **Run workflow**. It builds that ref
and `main` on the same Apple-silicon runner, then benchmarks them sequentially with
the selected ref's harness. Results appear in the Actions job summary; the workflow
does not run automatically on PRs or post PR comments.

Shared-runner timings are too noisy for reliable regression detection, even when
both builds use the same runner. Reproduce timing changes locally on an idle machine
before drawing conclusions. Correctness tests remain automatic and separate.

## onnax decoder workloads

`onnax_decoder.py` loads an exported onnax decoder task and reuses its sessions
for fixed-parameter forward, token loss and adapter gradients. Run it with the
onnax checkout on `PYTHONPATH` and that project's Hugging Face dependencies:

```sh
PYTHONPATH=/path/to/onnax/src:/path/to/onnax python bench/onnax_decoder.py \
  --export /path/to/export --library /path/to/libonnxruntime_mlx_ep.dylib \
  --sequence 4032 --warmup 10 --samples 100 \
  --output /tmp/decoder-eager.json --gradients-output /tmp/decoder-eager.npz
```

Add `--compile-head` to permit compilation for the chunked output head while
keeping decoder sessions eager. The default runs both eagerly. The report
records this per-session policy, session construction, raw step samples and
memory. Measure each policy in a separate process with the same export and
library, then compare the loss and every named adapter gradient before comparing
timing. ORT graph optimization and CPU fallback remain disabled by the onnax
session factory.

The [Qwen3.5 BF16/NF4 qualification reports](reports/onnax-head-policy-m1max)
cover 64 and 4,032 tokens. Both policies reproduce every adapter gradient bit for bit and match loss
at `rtol=1e-4, atol=1e-6`. The 4K qualification uses one warmup and three samples per
independent process. Head-only traces cover changed inputs and 128/63-token
chunks: the compiled route records an initial miss, a tail-shape retrace and
cache hits on repeated shapes. The folder also retains synchronized PyTorch
head-stage diagnostics, process guards and source checksums.

## Gated-delta Scan value tiles

The [4032-token backward-block report](reports/onnax-scan-value-tiles-m1max/README.md)
measures eight-column reverse tiles: 8.0% higher block throughput and 22.5%
lower MLX allocator peak on M1 Max. The report includes raw samples, numerical
checks and runtime digests.

## Paged GenAI inference

`genai_paged.py` measures a current GenAI Engine export, non-speculative by default. Use
the exact same exported directory for both providers, in separate sequential
processes, with tracing and GPU capture disabled:

```sh
python bench/genai_paged.py --model /path/to/export --ep mlx \
  --library "$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib" \
  --output /tmp/paged-mlx.json
python bench/genai_paged.py --model /path/to/export --ep webgpu \
  --output /tmp/paged-webgpu.json
```

Install compatible `onnxruntime`, `onnxruntime-genai` and
`onnxruntime-ep-webgpu` versions in one environment. The runner defaults to
three warmups, five repetitions and 256 generated tokens. It reports warm TTFT,
decode ms/token, aggregate decode tokens/s and inter-token p95, and retains
greedy tokens, raw timing samples, runtime versions and plugin/config hashes.
Compare prompt and generated token arrays before comparing speed. A fresh Engine
per repetition prevents cross-run prefix-cache reuse while model sessions stay
warm. TTFT includes request submission; decode excludes the first generated token.

The current upstream builder rejects `exclude_mtp=false` together with
`prune_lm_head=true`. For a non-speculative baseline use `exclude_mtp=true`.
Do not substitute an older installed builder: versions without
`use_paged_attention` can accept that extra option without producing a paged graph.

For hybrid Qwen MTP, export with `exclude_mtp=false prune_lm_head=false
state_update_capacity=4 max_draft_tokens=4`, retaining the same quantization and
paged-cache settings, and add `--mtp` to both provider commands. With GenAI
0.19.0, capacity zero leaves the head configured but clamps this hybrid Engine's
draft capacity to zero: it runs target-only decoding.
The runner requires an enabled MTP head, records per-run speculative telemetry,
and rejects runs with draft failures or no draft/target-verification execution.
GenAI 0.19.0's MTP path leaves `target_verify_forward_passes` at zero; activation
is checked using draft forwards, evaluated draft tokens and completed rounds.
Use `--max-draft-tokens N` to compare draft limits on the same exported weights;
the runtime overlay is recorded separately from the unchanged export config hash.
The Engine's `max_draft_tokens_per_proposal` is a capacity ceiling, not the
selected limit; inspect the overlay and proposed tokens per round together.
Compare both providers at the same limit before tuning each independently.
Acceptance rate, verification and reconciliation forwards, and emitted tokens
must accompany speculative throughput: a configured but inactive head is not an
MTP measurement. Inter-token intervals include the bursts of accepted tokens
emitted after verification and are not uniform per-token GPU execution times.

GenAI 0.19.0's shared-initializer loader calls the CPU device's unsupported
`GetMemoryInfo` method for MLX's CPU-visible I/O, failing model construction with
`GetMemoryInfo for CPU should not be used.` If encountered, retain the original
builder config and set both `model.decoder.shared_initializers` and
`model.mtp.shared_initializers` to empty arrays in the benchmark config before
loading it, after verifying that both ONNX files contain all their initializers.
The inspected export is self-contained, so no weights need to change. An empty
runtime overlay does not clear these arrays in this GenAI
version. Reports record sharing configuration, and comparisons must disclose
whether sharing is enabled for each provider.

Check builder/runtime schema compatibility before loading a capture export.
The inspected builder emits a seventh `state_update_active` input on
`VarlenCausalConvWithState`, but ORT 1.31.1's official schema accepts only six
inputs. For that version, remove the extra convolution input, not the GDN
control, and retain the original graph for audit. The convolution contract uses
`capture_count`; `GatedDeltaNet` separately supports the active control.

MLX supports packed `GatedDeltaNet` with scalar-per-value-head gates, FP32
V-major state and key widths up to 256, plus packed
`VarlenCausalConvWithState`. These GPU kernels keep cumulative sequence lengths
on-device. Compact state-update capture supports capacities 1 through 8:
convolution exports raw token rows, and recurrence exports FP32 decay, shared
normalized keys and delta rows in ORT's capsule layout. Captured prefixes can
restore the state after partial draft acceptance without rerunning the target.
GDN's optional inactive control zeroes its capsule without changing recurrence;
capacity zero retains the ordinary two-output kernels.

Packed exports can leave batch/token dimensions symbolic. `GatedRMSNorm` and
`MRotaryEmbedding` claim these forms without CPU boundaries; concrete reshape
and slice geometry remains shape-keyed across prefill and decode.
Packed SSM decoders coalesce capture-compatible neighbours into shape-keyed
regions rather than splitting at every shapeless/shape-keyed edge. Paged
attention keeps its separate cache-alias boundary; existing eager and
live-metadata boundaries remain eager.

### Qwen3.8-27B MTP qualification

[Full measurements, tokens, export/library hashes and numerical probes](reports/qwen38-paged-mtp-m1max.json)
cover genuine MTP on the M1 Max (32 GPU cores, 32 GiB), with ORT 1.31.1,
GenAI 0.19.0 and native MLX 0.32.4. The unpruned INT4/group-32 export uses
capture capacity four and draft limit four. Batch size is one, with 64 prompt
tokens, 256 generated tokens, three warmups and five measured repetitions,
without tracing. Both initializer-sharing arrays are disabled, and the extra
convolution input is normalized as described above; weight files are unchanged.

| MLX implementation | Warm TTFT median | Aggregate decode | Inter-token p95 |
|---|---:|---:|---:|
| Retained compact capture | 10,298 ms | 7.25 tokens/s | 417 ms |
| Rejected draft-head coalescing | 11,030 ms | 5.70 tokens/s | 608 ms |
| Coalescing repeat | 9,481 ms | 6.46 tokens/s | 522 ms |

Retained decode median is 135.1 ms/token. Each retained run completes 89
speculative rounds, with 355 proposed, 236 evaluated and 166 accepted draft
tokens (70.34% evaluated-token acceptance), 355 draft forwards, 90 target
forwards and zero MTP failures. All 256 tokens match across the fifteen
measurements, including the rejected builds. Short 64-token pilots suggested
that coalescing helped, but both full measurements regressed; that change was
reverted. Multi-query FP16 attention and forced small-batch QMM experiments
were also not retained. This enables MTP, but does not establish a speedup over
ordinary decoding; the earlier non-MTP measurements use a different, pruned
export.

MTP does **not** exactly match the single-token greedy reference. The first
difference is generated token 129 (zero-based): IDs 25 and 593 tie at 25.375
in single-token FP16 logits. Teacher-forcing the same target prefix in chunks
of three or five, without an MTP head or any prefix replay, selects 593 at
25.375 versus 25 at 25.359375. Batching alone therefore reproduces the choice.
Captured/uncaptured recurrence and FP32-fma capsule replay have bitwise
regressions, but no exact lossless-token claim is made for end-to-end MTP.

WebGPU EP 0.5.0 fails before the first token because
`com.microsoft:GatedDeltaNetStateReplay` is not registered. There is no valid
activated-MTP WebGPU throughput comparison, and no MTP speed-win claim.

### Qwen3.8-27B initial local qualification

[Raw samples, tokens, options and model/plugin hashes](reports/qwen38-paged-m1max.json)
cover the same INT4/block-32 paged export on an M1 Max (32 GPU cores, 32 GiB).
Runtime versions: ORT 1.31.1, GenAI 0.19.0, WebGPU EP 0.5.0 and native MLX 0.32.4.
MTP is disabled and LM-head pruning retained. Batch size is one, with 64 prompt
tokens, 256 generated tokens, three warmups and five measured repetitions per EP.
Providers run in separate sequential processes, without tracing or GPU capture.

| Provider/build | Warm TTFT median | Aggregate decode | Inter-token p95 |
|---|---:|---:|---:|
| MLX runnable baseline | 1,636 ms | 3.60 tokens/s | 292 ms |
| MLX optimized | 615 ms | 7.91 tokens/s | 132 ms |
| WebGPU | 1,395 ms | 9.02 tokens/s | 126 ms |

The runnable baseline already includes the new packed SSM kernels; it is not an
unmodified-main measurement. Dynamic norm/rotary claims and capture coalescing
raise placement from 1,382/1,462 to 1,462/1,462 nodes and reduce fused partitions
from 641 to 33. Decode improves **2.20x**, but WebGPU still leads decode by **14%**.
MLX TTFT is **2.27x faster** than WebGPU for this prompt.

All 256 greedy tokens match across every run, both MLX builds and WebGPU.
A same-binary `ONNXRUNTIME_EP_MLX_NO_COMPILE=1` comparison gives 7.28 tokens/s
and 1,045 ms TTFT, so the compiled-region policy is retained. These measurements
do not establish long-prompt or concurrent-request performance. On this 32 GiB
host, export required external-backed staging and column-batched LM-head
quantization to avoid memory exhaustion; the quantization algorithm and options
were unchanged, with column batching checked for bit-identical results.

### Paged-cache writeback follow-up

[Follow-up raw samples and hashes](reports/qwen38-paged-writeback-m1max.json)
use the same model and measurement settings, with a fresh WebGPU run:

| Provider/build | Warm TTFT median | Aggregate decode | Inter-token p95 |
|---|---:|---:|---:|
| MLX direct host metadata, full cache writes | 715 ms | 8.10 tokens/s | 138 ms |
| MLX sparse cache writeback | 612 ms | 9.77 tokens/s | 109 ms |
| WebGPU recheck | 1,281 ms | 10.42 tokens/s | 102 ms |

Sparse writeback improves decode by **20.6%** over the metadata-only reference.
WebGPU still leads decode by **6.6%** in this round; its fresh result, rather
than the earlier 9.02 tokens/s, is the comparison. All 256 greedy tokens match
every measured run and the initial qualification. PagedAttention writes only
updated physical input slots; aliased cache outputs copy an enclosing changed
row interval, while independent outputs still receive the full cache.

A native grouped-SDPA variant avoided repeated KV-head views but reduced
end-to-end decode to **9.24 tokens/s**, despite matching all 256 tokens. It was
reverted; its raw samples are retained as a rejected experiment in the follow-up
report. Fewer materializations alone do not establish an end-to-end win.

### FP32-accumulating vector-decode follow-up

[Raw samples, tokens, hashes, and both provider orders](reports/qwen38-paged-vector-m1max.json)
retain the same export, prompt, warmups, repetitions, and 256-token completion.
Eligible short single-query paged attention keeps FP16 Q/K/V buffers while
MLX 0.32.4's vector kernel accumulates scores, softmax, and weighted values in
FP32. Other geometries and custom attention modes retain the previous path.

| Round and execution order | MLX reference | MLX vector decode | WebGPU |
|---|---:|---:|---:|
| Reference, candidate, WebGPU | 9.54 tokens/s | 10.16 tokens/s | 8.89 tokens/s |
| WebGPU, candidate after cooldown | - | 10.59 tokens/s | 9.65 tokens/s |

The same-round MLX improvement is **6.5%**, and every measured token matches the
original qualification. Reverse-order warm TTFT medians are 588 ms for MLX and
1,281 ms for WebGPU. Both aggregate comparisons favor MLX, but the reverse-order
WebGPU run includes one 35.65-second decode sample: its other four runs average
about **10.57 tokens/s**, nearly matching MLX's 10.59. All samples are retained;
these results do **not** establish a durable steady-state lead.

Head width 192 and contexts of at least 1,024 keys retain FP32 buffers because
their FP16 paths do not preserve the existing rounding. Large/canceling-value
tests compare eligible outputs exactly to an FP32 MLX attention reference and
also check an independent NumPy reference. A separate mask-elision experiment
matched tokens but measured 7.79 tokens/s and was reverted.

Public [WebGPU source](https://github.com/microsoft/onnxruntime/tree/74e4120bbc307ecb5c81ef84a9cf42f1fe19519e/onnxruntime/contrib_ops/webgpu)
uses direct paged decode with shared physical-row offsets and real GPU buffer
allocation/transfers. This source pin is not verified as the installed 0.5.0
wheel's build commit. Device tensor residency and avoiding dense KV gathers are
the next structural leads, not evidence that replacing stock MLX QMV will help.

### Qualified typed-mask decode lead

[Complete samples, tokens, hashes, and reverse-order confirmation](reports/qwen38-paged-typed-mask-m1max.json)
omit the redundant zero mask only within the validated short, single-query FP16
GPU vector path without a local window. FP32 accumulation, state/cache outputs,
prefill, windows, long contexts, and custom attention modes are preserved.

| Build, WebGPU-first round | Warm TTFT median | Aggregate decode | Inter-token p95 |
|---|---:|---:|---:|
| WebGPU | 1,280 ms | 10.56 tokens/s | 94.54 ms |
| MLX vector reference | 595 ms | 10.47 tokens/s | 100.81 ms |
| MLX typed-mask decode | 586 ms | **10.94 tokens/s** | 95.31 ms |

The candidate improves same-round MLX decode by **4.5%** and leads fresh WebGPU
by **3.5%**. Every candidate decode sample is faster than every WebGPU sample in
this round, without dropping samples. Reversing provider order after cooldown
gives MLX **10.87 tokens/s** versus WebGPU **8.63 tokens/s**; WebGPU slows in that
round, so the smaller WebGPU-first lead is the primary comparison.

Every measured 256-token sequence matches the original qualification. This is
an end-to-end, non-speculative batch-one prompt64/output256 result, not evidence
for longer contexts, concurrent requests, or the external 15-17 tokens/s claim.
Expanded reference coverage includes causal/bidirectional windows, the model's
24-query/4-KV head layout, subnormal queries, wide values, and fallback geometry.

Intervening controls remain in the vector report: typed native GQA did not beat
the repeated-head reference and was reverted. The exact vector-buffer-only build
also measured 10.36 versus WebGPU 10.43 tokens/s before this mask follow-up.

Source research at
[MLX-LM](https://github.com/ml-explore/mlx-lm/tree/736ab0d8c1d0edc79c3d798044fe6904e2fa23be),
[MTPLX](https://github.com/youssofal/mtplx/tree/9882703f3105363ddc37eca9f97aa09a1d387112),
and [Lithos Metal](https://github.com/lithos-ai/lithos-metal/tree/6653b7df5f2c4392ec7b038960f2796b1499923f)
prioritizes cache ownership/traffic, projection concatenation with library GEMV,
and fusion between projections. Their native GPU state/sampling interfaces,
different quantization, speculative projections, and newer-chip measurements
are not interchangeable with this standard-ORT, M1, group-32 comparison.

## Historical model measurements

These results were previously recorded in the root README on an M1 Max.
They are **not new ONNX Runtime 1.31 / MLX 0.32.4 benchmarks** or promises for
other models and machines. Exact runtime revisions and reproduction commands
were not recorded alongside these tables. Use the harness above for current
op-level comparisons and measure your own end-to-end workload.

### Encoders

Warm end-to-end inference, median of 10 runs, MLX EP versus ORT CPU on the same
machine. The original report recorded identical top-1 results and maximum
absolute differences no greater than 6e-5 for these cases.

| Model | Workload | CPU EP | MLX EP | Speedup |
|---|---|---:|---:|---:|
| Perch v2 | audio encoder (with DFT front-end) | 64.0 ms | 12.0 ms | 5.3x |
| Perch v2 (no DFT) | audio encoder | 56.5 ms | 12.0 ms | 4.7x |
| BirdNET | audio classifier (CNN) | 14.9 ms | 7.3 ms | 2.0x |
| gemma-4-E2B | vision encoder (fp16 ViT) | 267 ms | 47 ms | 5.7x |

### Decoders

Foundry Local q4f16 models, warm, MLX EP versus ORT CPU on the same M1 Max.
Prefill is a 128-token step; decode is one token with 128 past tokens.

| Model | Architecture | Prefill speedup | Decode speedup |
|---|---|---:|---:|
| Qwen2.5-0.5B | GQA, external rotary | 5.2x | dispatch-bound (CPU-favored) |
| Phi-3.5-mini | Phi3, GQA | 5.29x | 1.19x |
| Phi-4-mini | Phi4, long-context RoPE | 5.78x | 1.10x |
| Mistral-7B-Instruct | GQA, growing KV | 11.89x | 3.30x |
| gemma-4-E2B | Gemma3n, 15-layer | 3.3x | 3.3x |

The README also reported Muse-Glimmer-30B INT4 at 138.67 prefill tok/s
(512 tokens) and 14.79 decode tok/s (200 generated tokens), using the then-bundled
MLX runtime and automatic 8-layer groups. A same-quantization llama.cpp baseline
was reported at 137.84 / 13.50 tok/s. Those figures lack the fuller reproduction
details needed for a current backend comparison.

Compilation, partitioning, CPU fallback, and weight bandwidth all affect
latency. Small or dispatch-bound workloads can favor CPU. See
[tracing and profiling](../examples/traces/README.md) to understand a run,
and [decoder tuning](../docs/INTEGRATION.md) for partition and precision options.
