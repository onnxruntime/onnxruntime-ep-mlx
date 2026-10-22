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
