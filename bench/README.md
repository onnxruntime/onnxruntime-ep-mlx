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

## Softmax numerical audit

`softmax_numerics.py` reproduces the forward collapse described in
[A Fix for Softmax and Cross-Entropy](https://arxiv.org/html/2610.31642v1).
It compares Softmax, LogSoftmax, SoftmaxCrossEntropyLoss (including its optional
log-probability output), NLL, and fused/composed attention with an independent
FP64 reference. Cases include high margins, tied maxima, a single class,
non-max targets, and large common offsets. FP16/BF16 inputs are quantized before
constructing the reference; unavoidable output underflow is distinguished from
lost log-probabilities.

With the built plugin and Python `mlx`, `onnxruntime`, `onnx_ir`, and `numpy`
available:

```sh
ONNXRUNTIME_MLX_EP_LIB=rust/target/release/libonnxruntime_mlx_ep.dylib \
  python bench/softmax_numerics.py --output /tmp/softmax-numerics.json
```

Use `--mode accuracy` or `--mode benchmark` to run only one part. The ONNX
baseline and paper-style candidate run through the EP with CPU fallback and ORT
graph optimizations disabled. Separate compiled-MLX timings compare an
inference-only `log1p` formulation and distinguish loss of fused SDPA from
normalization overhead. Each latency sample synchronizes output evaluation;
warmup/compilation is excluded. JSON includes raw samples and device/version
metadata. Repeat in separate processes on an idle machine before interpreting
small differences.

The candidates are benchmark graphs, not production replacements. The existing
`x - logsumexp(x)` loss/log-softmax path can lose representable values, including
BF16 uniform logits with a large common offset. Rewriting the normalizer
improves those outputs but can add significant dispatch/memory overhead; the
paper's training-gradient benefits do not establish an inference speedup.

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

The runner registers WebGPU's native `WebGpuExecutionProvider` plugin but selects
GenAI's **`WebGPU`** provider alias. Selecting the literal plugin name instead
routes GenAI through its generic CPU device interface, including CPU-owned cache
buffers. Older reports produced by this runner used that generic configuration;
their measured comparisons are not device-resident WebGPU baselines or evidence
of a lead over the best-supported WebGPU configuration.

Source-build/startup provenance is retained in
[`qwen38-paged-webgpu-main-m1max.json`](reports/qwen38-paged-webgpu-main-m1max.json).
The separate [multi-query MTP A/B](reports/qwen38-paged-multiquery-mtp-m1max.json)
preserves five full 256-token samples per mode. Its aggregate throughput improved,
but median decode latency and TTFT did not; it does not establish a robust
short-context speedup, a WebGPU lead, or a 16K end-to-end result.
The [grouped-KV prefill experiment](reports/qwen38-paged-prefill-gqa-rejected-m1max.json)
was removed after its isolated attention gains did not survive full-model MTP
qualification; all timing samples, including the stall, remain in the report.

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
Use `--disable-mtp` instead of `--mtp` for a matched non-speculative control
on the **same unpruned MTP export**. This changes only the recorded runtime
overlay (`model.mtp.enabled=false`); it does not edit the export. Non-speculative
runs reject actual draft execution. Do not compare MTP against a separately
pruned target-only model to calculate its speculative speedup.

For a WebGPU build from ONNX Runtime main, pass its library explicitly instead
of using the installed plugin wheel:

```sh
python bench/genai_paged.py --model /path/to/mtp-export --ep webgpu \
  --library /path/to/libonnxruntime_providers_webgpu.dylib \
  --source-commit "$(git -C /path/to/onnxruntime rev-parse HEAD)" \
  --mtp --max-draft-tokens 4 --prompt-file /path/to/long-prompt.txt \
  --num-blocks 128 --max-scheduled-tokens 256 \
  --output /tmp/long-webgpu-mtp.json
```

`--source-commit` records the supplied source revision alongside the binary
hash; build that exact checkout before measuring. `--prompt-file` reads UTF-8
user text, applies the normal chat template, and records the actual prompt
tokens and text hash. Use those actual token counts, not the file size or an
estimated "16K" label. Scheduling overrides are recorded in the runtime
overlay and require a dynamic-batching export; its ONNX cache/input dimensions
must also admit the requested capacity. Eight 256-token blocks cannot hold a
16K prompt. Use identical prompt, capacity, scheduling and draft settings for
both providers and for the `--disable-mtp` control. Measure long-context TTFT
and decode separately, record memory pressure, and do not compare different
Apple GPU generations as an EP speedup.

Prefill has **two independent limits**: the Engine's total token budget
(`--max-scheduled-tokens`) and the request's `search.chunk_size`
(`--prefill-chunk-size`). The smaller limit wins for a single request; concurrent
requests also share the total budget. Raising only the scheduling budget above
the export's chunk cap does not produce larger prefill calls. For example, this
export retained 256-row GDN inputs with a 1,024-token scheduling budget until
`search.chunk_size` was also raised. To test actual 512-token chunks, use both
`--max-scheduled-tokens 512 --prefill-chunk-size 512`. These options change only
the recorded runtime overlay, not the exported model. Larger chunks can increase
working memory and delay other requests, and are not automatically faster.
The runner also records `ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES` and
`ONNXRUNTIME_EP_MLX_WIRED_LIMIT_BYTES` when present. Public GPU tensor ownership
does not itself wire buffers into the GPU working set. Choose an explicit
residency budget for large-model measurements without exceeding the device's
recommended working set; the process-wide effects are described in
[`MEMORY.md`](../docs/MEMORY.md).

The plugin's minimum ABI version is not sufficient to establish operator-schema
compatibility. Main revision `0e21de25ea368aa3dff00daabee9bd0399ec4bed` requires a
matching ORT 1.32.0 runtime for its `GatedDeltaNet` type constraints; the installed
1.31.1 runtime cannot load that model/provider combination. Use an isolated
matching runtime for both providers, and retain its library path and actual
imported version in the report. Build the plugin version as valid SemVer
(for example `0.5.0-dev.20261105`, not `0.5.0.dev20261105`).
[Source-build provenance and genuine-MTP startup checks](reports/qwen38-paged-webgpu-main-m1max.json)
retain the pinned revision and library hashes. Those short diagnostics are not
a qualified throughput comparison or a reproduced 50% MTP speedup.

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
Keep that compatibility config separate from the builder config. Do not remove
working shared-weight metadata from WebGPU's device-backed configuration:
this Qwen export shares about 3.03 GiB between the decoder and MTP head, which
can materially affect memory pressure on a 32 GiB machine. If provider-specific
sharing configurations are necessary, record that difference rather than
attributing its allocation cost to arithmetic kernels.
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

Packed GDN prefill uses an eight-token block recurrence by default.
`ONNXRUNTIME_EP_MLX_CHUNKED_PREFILL=0` selects the original serial recurrence
for an A/B control; `1` explicitly enables blocks. It requires at least 32 packed tokens,
key/value widths divisible by eight, L2-normalized Q/K, sigmoid beta, softplus
decay and enabled coefficient preparation. Calls with fewer than 32 packed
tokens, including this batch-capacity-two model's short decode and MTP
verification calls, retain the existing recurrence. The block kernel
keeps state and matrix arithmetic in FP32, uses direct decay products without
clamping or dividing by a decay, and supports ragged tails, empty sequences and
compact prefix capture. Its reassociated reductions are not bitwise identical
to token-serial recurrence. An isolated kernel speedup is not an end-to-end TTFT claim.
The [initial full-model qualification](reports/qwen38-paged-chunked-prefill-mtp-m1max.json)
preserved all generated tokens and speculative statistics, but median TTFT
regressed from 18.430 to 23.961 seconds. A scheduler-256 short pilot did not
establish a robust improvement either. This remains a CPU-visible regression
case: default block execution does not imply an MTP speedup, and the serial
control remains available.
The separate [GPU-resident target-only qualification](reports/qwen38-public-chunked-prefill-m1max.json)
uses the public ORT chunked driver at the same 64-token chunk size. Its full
1K-input/256-output run improved median warm TTFT from 7.836 to 7.433 seconds,
with every token matching the independent reference. A reversed-order short
screen showed a smaller 1.6% TTFT reduction; all samples and timing caveats are
retained. This is not an MTP or WebGPU speedup claim.
The [16K-input/256-output check](reports/qwen38-public-chunked-prefill-16k-m1max.json)
on M1 Max uses 1,024-token public-driver chunks, 65 cache blocks, a 512 MiB
free-cache limit and an explicit 26,800,603,136-byte wired limit. With three
warmups and one measured repetition per mode, serial/block TTFT was
146.527/141.153 seconds; decode was 9.697/9.560 tok/s. All 256 tokens matched
each other and the independent GenAI target-only and MTP runs.
The observed 3.7% TTFT reduction is a bounded check, not a statistically robust
speedup; decode did not improve. GPU ownership alone was insufficient for stable
long-context latency: an earlier unwired block run did not finish its first
warmup in 40 minutes. That failed run is retained separately, not counted as a
performance sample or a matched wiring A/B. The wired budget is opt-in and
device-specific; do not copy this machine's limit without checking available
working-set capacity.

The [genuine GenAI 16K MTP check](reports/qwen38-paged-mtp-16k-wired-m1max.json)
uses the same unpruned export for target-only and MTP, CPU-owned GenAI state,
256-token scheduling/chunks and 70 cache blocks. Three warmups and one measured
repetition gave target/MTP TTFT of 171.190/173.191 seconds and decode of
6.859/5.998 tok/s. MTP completed 116 speculative rounds with 248 draft tokens
evaluated, 139 accepted and zero failures; all 256 output tokens matched.
This is a negative MTP result, not a 50% speedup or a matched WebGPU comparison.
The initial 66-block MTP attempt was rejected before generation because GenAI's
speculative scheduling reserve reduced its maximum request length to 15,873;
both final GenAI modes were rerun at 70 blocks.

The [16K proposal-budget sweep](reports/qwen38-paged-mtp-proposal-budget-16k-m1max.json)
keeps the original capacity-four target/draft graphs and the same native binary.
Reducing the runtime proposal budget avoids much of the rejected draft work:

| GenAI MLX configuration | Warm TTFT | Decode |
|---|---:|---:|
| K4, 256-token chunks | 173.191 s | 5.998 tok/s |
| K1, 256-token chunks | 174.909 s | 9.452 tok/s |
| K2, 256-token chunks | 176.909 s | 7.699 tok/s |
| K1, 1,024-token chunks | 156.832 s | 9.554 tok/s |
| Separate selected-head K1, 1,024-token chunks | 145.479 s | 9.137 tok/s |

All 256 MLX tokens match the target-only control. The original-export
K1/chunk-1024 result is an observed 59.3% increase over K4 MTP, but only 39.3%
over the non-speculative GenAI control. These are single measured repetitions
after three warmups, not universal defaults or statistically robust speedups.
Use explicit `--mtp --max-draft-tokens 1 --prefill-chunk-size 1024
--max-scheduled-tokens 1024 --num-blocks 70` for this configuration.
GenAI's reported `max_draft_tokens_per_proposal` remains the export capacity,
four; the runtime overlay and actual proposal counts establish K1.
The selected-head variant lowers TTFT but regresses measured decode, so no
production model-interface rewrite is retained.

WebGPU rebuilt from main at `f795db61a827f2d09710a62dc704289b18811a43`
completed K1/chunk-256 at 525.951-second TTFT and 5.766 tok/s on this M1 Max.
Its generated tokens diverge from MLX starting at zero-based index 39, so the
result does **not** establish an exact-output cross-EP throughput win.
The K4 attempts exceeded separate 40- and 75-minute orchestration bounds;
their warmups and failure evidence remain in the report, not final throughput
samples. WebGPU retains its working device interface and initializer sharing;
MLX uses CPU-owned GenAI state and its explicit residency budget.

Compare this option at fixed `--max-scheduled-tokens` before tuning the
scheduler budget, and compare generated tokens and speculative statistics.
`ONNXRUNTIME_EP_MLX_NO_GDN_PREPARE=1` also disables the block route.

Packed exports can leave batch/token dimensions symbolic. `GatedRMSNorm` and
`MRotaryEmbedding` claim these forms without CPU boundaries; concrete reshape
and slice geometry remains shape-keyed across prefill and decode.
Packed SSM decoders coalesce capture-compatible neighbours into shape-keyed
regions rather than splitting at every shapeless/shape-keyed edge. Paged
attention keeps its separate cache-alias boundary; existing eager and
live-metadata boundaries remain eager.

### Public-ORT GPU-resident K1 MTP

`python -m bench.ort_mtp` is a separate fixed-length greedy outer loop, not
GenAI Engine or a transparent EP rewrite. It uses the unchanged unpruned
target/draft exports, public GPU OrtValues, and GPU greedy selection.
Only sampled token IDs are read on CPU. State, captures, hidden states and
full logits stay GPU-owned; intermediate prefill logits are released before
decode. Bindings and state allocations are reused across requests. Recurrent and
convolution states are reset; `past=0` logically invalidates KV without clearing
the full cache. ORT memory patterns are disabled. Repeated 16K requests qualified
with the explicit residency budget below; earlier larger-budget attempts hit GPU OOM.
The draft head starts cold from the final target prompt hidden row.
Accepted drafts are re-fed with their corresponding target hidden rows.

`bench.mtp_replay` builds separate checkpoint and prefix-one replay models
using existing ONNX operators. Recurrent replay applies the captured FP32
decay with `Mul`, then a linear `GatedDeltaNet` update with captured keys and
deltas. Convolution history uses bit-preserving slices and concatenation.
Rejection restores from the pre-verification checkpoint; paged KV is retained,
with its logical length rolled back so invalid future slots are excluded and
overwritten on the next verification. This helper is not an implementation of
the distinct `GatedDeltaNetStateReplay`/`ConvStateReplay` operator ABIs.

Run with a flat GenAI reference containing the exact prompt and output IDs:

```bash
python -m bench.ort_mtp --model MODEL --library MLX_DYLIB \
  --reference TARGET_REFERENCE_JSON --output RESULT_JSON \
  --blocks 70 --block-size 256 --capacity 4 \
  --prefill-chunk-size 512 --tokens 256 --warmup 3 --repetitions 5
```

Configure a device-appropriate explicit wired/cache budget before execution;
GPU allocation alone does not guarantee residency. The capture capacity must
match the export, even though the runtime proposal budget is one. Every warmup
and measured output must match the reference before a timing report is written.
The report distinguishes this interface and its cold-head priming from GenAI.
One-time helper/buffer setup and per-request state resets are excluded from
TTFT; prefill and first greedy selection are included. Same-round emitted tokens
retain their shared timestamp rather than inventing evenly spaced inter-token timings.

Add `--target-only` for a non-speculative control using the same unpruned target
export, GPU state, and greedy selector. It runs one target token at a time
without draft, checkpoint, or replay execution. Its report sets
`max_draft_tokens` to zero and all speculative counters to zero; MTP remains
the default. This control isolates execution differences within the ONNX
export, not differences from native-runtime model conversions.

For an explicit model-native prefill optimization, build a separate target
graph that projects only the last two hidden rows:

```python
from pathlib import Path
import onnx
from bench.ort_mtp import make_last_two_logit_model

model = Path("MODEL")
target = onnx.load(model / "model.onnx", load_external_data=False)
onnx.save(make_last_two_logit_model(target), model / "model.last-two-logits.onnx")
```

Keep it beside the original graph so relative external-weight references remain
valid; weights are not loaded or rewritten. Add
`--target-last-two-logits MODEL/model.last-two-logits.onnx` to the benchmark.

The [16K M1 Max qualification](reports/qwen38-last-two-logits-m1max.json)
measured 128.559 s median TTFT versus 134.969 s for the original target, with
unchanged reference tokens and effectively unchanged decode throughput.
This is a model-native vocabulary-projection reduction, not an EP kernel gain.
Only the vocabulary projection input is sliced. All hidden, cache and recurrent
outputs are preserved, and one-token decode/two-token K1 verification retain
every original logit row. This uses one target session, not a second prefill
session. It is an explicit graph rewrite, not an unchanged-export EP comparison
or permission for ordinary public sessions to omit output rows.

The original target remains the default comparison. Before treating the clipped
graph as a performance improvement, qualify exact reference tokens and measure
prefill and decode independently; different QMM/QMV paths can change rounding.

[The small-input ownership qualification](reports/qwen38-owned-small-inputs-16k-m1max.json)
keeps CPU runtime inputs of at most 4096 bytes in owned MLX arrays, allowing
GPU-only boundaries to use their existing asynchronous evaluation path.
Large borrowed CPU inputs, CPU outputs, and input writebacks remain synchronous.
The owned path is default; `ONNXRUNTIME_EP_MLX_OWN_SMALL_INPUTS=0` restores
borrowed wrapping for small inputs.

On M1 Max 32 GiB, the same-binary 16K/256 qualification used three warmups and
three measured requests per arm. Aggregate K1 MTP decode rose from 11.483 to
11.936 tokens/s, while median TTFT changed from 135.401 to 136.275 seconds.
All twelve requests matched the unchanged target reference. Subsequent
shipping-binary default/opt-out checks also matched, but timings varied
substantially: 9.321 versus 7.502 tokens/s and 134.453 versus 171.101 seconds
TTFT. These checks are not a clean additional speedup measurement.

The earlier native llama.cpp reference remains 128.896 seconds TTFT and
16.166 tokens/s. Its mixed GGUF quantization differs from the ONNX affine
INT4/group32 export, and generated tokens differ across native runtimes.
The llama.cpp throughput target has not been reached.

[The M1 Max 32 GiB residency screen](reports/qwen38-public-mtp-residency-16k-m1max.json)
completed three warmups and one measured 256-token request at 16,384 input tokens;
all four outputs matched the reference. With 512-token chunks, a 22 GiB wired cap
(`23622320128` bytes) and a 512 MiB cache limit, the measured result was
183.044 s TTFT and 9.296 tok/s. This qualifies a GPU-resident MTP baseline, not a
new speed win or a universal memory-budget default. Earlier 24.96 GiB wired-cap
attempts failed during repeated prefill. Native allocation counters did not
establish a large monotonic leak; the driver-OOM root cause remains unresolved.

### Qwen3.8-27B compact MTP capture qualification

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
activated-MTP WebGPU throughput comparison, and no MTP-over-WebGPU speed-win claim.

### Persistent public ORT device tensors

`ONNXRUNTIME_EP_MLX_DEVICE_TENSORS=1` enables experimental MLX-owned GPU
allocations and public ORT data transfers. Set it before registering the plugin
in a fresh process; the allocator policy is fixed at registration. The default
CPU-visible interface remains unchanged. This mode retains native arrays across
compiled and eager partitions instead of copying each boundary output to host.
Device tensors and their transfers must stay on their producer's inference
thread. MLX device addresses are opaque allocator identifiers, not host buffers
or an interop contract with another GPU provider. Use the registered memory
information, not a hand-constructed memory descriptor:

When this mode is enabled, a co-registered GPU provider with an equivalent ORT
memory-device descriptor can match MLX's transfer callback. Foreign allocations
are rejected safely, but ORT cannot retry a later provider's transfer. Use
separate processes for such providers, or copy through the owning EP to CPU
before passing data to MLX. Float64 device views and downloads use the MLX CPU
stream; they do not enable float64 Metal computation.

```python
device = next(d for d in ort.get_ep_devices() if d.ep_name == "MLXExecutionProvider")
info = device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
state = ort.OrtValue.ortvalue_from_shape_and_type(shape, dtype, memory_info=info)
state.update_inplace(initial_state)
binding.bind_ortvalue_input("past.0.recurrent", state)
binding.bind_ortvalue_output("present.0.recurrent", state)
```

Allocate each state once and reuse the binding across decode calls. Outputs
retain their native storage after session teardown. Readback through `.numpy()`
or a CPU output binding synchronizes and transfers data; device I/O alone does
not remove an application's explicit readbacks. CPU feeds are uploaded, not
mutable GPU cache allocations: when a PagedAttention cache output is omitted,
observe in-place updates through the original **device** OrtValue. CPU-visible
input/output aliasing continues to work through explicitly bound outputs.
If a device Run fails, reinitialize bound state before retrying: outputs may
already alias inputs. Device evaluation errors are reported rather than replayed
through an eager path that could consume partially updated state.

`ort_paged.py` measures a target-only public-ORT loop with CPU versus persistent
GPU state ownership on the same unchanged paged export:

```sh
python -m bench.ort_paged --model /path/to/non-mtp-export \
  --library "$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib" \
  --reference /tmp/paged-mlx.json --memory cpu --output /tmp/ort-cpu.json
python -m bench.ort_paged --model /path/to/non-mtp-export \
  --library "$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib" \
  --reference /tmp/paged-mlx.json --memory mlx --output /tmp/ort-device.json
```

Add `--prefill-chunk-size 64` to the public ORT driver for bounded prefill.
Every chunk reuses the same state/input/output OrtValues. Intermediate chunks
request an empty `logits_indices` tensor and bind a zero-row logits output, so
they update state without evaluating an unnecessary output head or sampling a
token. Only the final prefill chunk requests logits; TTFT includes all chunks.
The default remains unsegmented prefill. This driver requires a pruned,
target-only export and does not run MTP.

Both processes use the reference report's exact prompt tokens and require every
generated token to match it. State initialization is outside timing; sampling
remains CPU first-max argmax over the unchanged logits. This is a different
driver from GenAI, not evidence that GenAI itself retains final states on GPU.
The runner rejects enabled MTP configurations rather than silently benchmarking
their target alone. Use `genai_paged.py --mtp` for activated MTP measurements.

Paged attention also has a direct FP16 single-query path for head widths
64/96/128/256 and contexts below 1,024 tokens. It reads physical KV pages with
GQA head mapping, without gathering a dense cache or repeating heads. Other
geometries, multi-query verification and unsupported attention options keep the
existing implementation. Packed dirty-row writeback can avoid rooting a full
GPU cache scatter when cache outputs are omitted or verified to alias the
CPU-visible input; independent outputs retain the complete updated cache.
Native device aliases retain functional cache updates rather than mutating
arrays that could have other owners.

For same-binary diagnostics, `ONNXRUNTIME_EP_MLX_NO_PAGED_DIRECT=1` and
`ONNXRUNTIME_EP_MLX_NO_PAGED_PATCHES=1` independently disable these changes.

MRotaryEmbedding shares per-column position indices and gathers only the
selected cosine/sine cache values, rather than gathering all three full streams
and assembling them with per-column Slice/Concat chains. Position validation,
stream assignment, dtype, and scale arithmetic are unchanged. The column and
stream maps are cached; caches too large for safe int32 linear addressing retain
the original path. All other valid inputs use the column gather without model,
hardware, or process-flag gating; no chip-specific kernel is selected. The
[M1 Max qualification](reports/qwen38-mrope-column-gather-m1max.json) measured
about 27% faster aggregate short-context decode and 30% faster 16K decode with
unchanged tokens and FP32 state. 16K TTFT was effectively unchanged. The report
retains all samples, including the slower candidate request, and records the
temporary diagnostic opt-out used for its historical control arm. Newer Macs
were not measured.

The [follow-up decode experiments](reports/qwen38-decode-followups-m1max.json)
retain the completed non-speculative 16K control and all short-context samples.
Fixed K2, confidence-gated K2, draft last-logit pruning and forcing native
matrix kernels did not establish a gain over the qualified K1 defaults, so
none is enabled. More accepted drafts or fewer projected rows alone are not
an end-to-end speedup. Native aggregate parity and newer-Mac gains remain
unproven.

Packed GDN prepares FP32 normalized Q/K and decay/beta coefficients once when
there are more than eight packed tokens and at least 32 value rows, rather than
recalculating them for each value row. Short-query decode and verification retain
their original kernel. The recurrence and compact capture keep their update and
reduction order. Set `ONNXRUNTIME_EP_MLX_NO_GDN_PREPARE=1` for the same-binary
control; the benchmark records this setting alongside the other MLX options.

The [1K-input MTP qualification](reports/qwen38-paged-gdn-prefill-mtp-m1max.json)
observed median TTFT **28.07 -> 23.06 seconds** with identical tokens and
speculative statistics (three warmups, five 256-token completions per mode).
The fixed ordering and system memory pressure limit causal precision; this is
not a WebGPU or 16K comparison. The
[broader preparation experiment](reports/qwen38-paged-gdn-prepared-mtp-m1max.json)
did not establish a decode gain in reversed order and is not retained.
Collect tracing separately from final latency measurements; public GPU-to-CPU
transfers and sparse input writebacks are included in copy accounting.

### Qwen state/cache qualification

[Raw samples, tokens, source/library hashes, ownership diagnostics and rejected
experiments](reports/qwen38-paged-residency-m1max.json) separate each binary and
driver. Final target-only GenAI measurements use the same unchanged INT4/group-32
export on the M1 Max (32 GPU cores, 32 GiB), with Python 3.13.5, ORT 1.31.1,
GenAI 0.19.0, native MLX 0.32.4 and WebGPU EP 0.5.0. Prompt length is 64,
generation length 256, with three warmups and five measured repetitions in
fresh sequential processes; tracing and capture are disabled.

| Provider/mode | Warm TTFT median | Decode median | Aggregate decode | Inter-token p95 |
|---|---:|---:|---:|---:|
| MLX, `ONNXRUNTIME_EP_MLX_DEVICE_TENSORS=1` | 605 ms | 85.03 ms/token | 11.71 tokens/s | 97.04 ms |
| WebGPU | 1,283 ms | 95.37 ms/token | 10.48 tokens/s | 94.69 ms |

MLX aggregate decode is **11.8% higher** in this qualification. Every generated
token matches across all ten measured runs. No stalls are discarded; this is a
short-context, single-request result, not a claim about every latency percentile
or longer contexts.

Activated MTP uses the separate unpruned capacity-four export and the default
CPU-visible interface (`ONNXRUNTIME_EP_MLX_DEVICE_TENSORS=0`): **8.27 tokens/s**,
versus **7.70 tokens/s** with both direct paging and packed patches disabled on
the same binary (**7.4% higher decode throughput**). Warm TTFT is 10,010 ms
versus 8,941 ms for that control; inter-token p95 is 370 versus 394 ms.
Each run completes 89 rounds,
evaluates 236 draft tokens and accepts 166, with zero failures. All 256 tokens
match the committed MTP reference, including its known FP16 tie-dependent
difference from target-only decoding. WebGPU still lacks
`GatedDeltaNetStateReplay`, so no activated-MTP WebGPU speed comparison exists.

Allocator enablement alone is **not GenAI state residency**. Separate four-token
traces find 192 three-MiB recurrent-state readbacks in GenAI in either memory
mode. Explicit public GPU state bindings eliminate those readbacks and download
only 1,986,560 logits bytes in total. The older public-loop ownership A/B is
archived separately; its timings must not be mixed with GenAI measurements.
Keep GenAI MTP CPU-visible for now: its allocator-only GPU-mode series included
a 595.65-second decode stall in GPU-to-CPU transfer. All five samples are
retained, with the CPU-sampling and memory-pressure diagnostic disclosed.

Boundary reduction remains unresolved. There are **33 total ORT regions**,
including 16 PagedAttention boundaries, not 33 compiled regions plus another 16.
The four-call trace records 16 compiled and 17 eager executions per call.
A genuine single-partition hybrid retained compilation but regressed
11.89 to 11.23 tokens/s; deferred GPU-only submission regressed 11.73 to 10.42.
Both experiments were removed.

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
