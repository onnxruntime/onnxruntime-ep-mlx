# onnxruntime-mlx

[![PyPI Version](https://img.shields.io/pypi/v/onnxruntime-ep-mlx)](https://pypi.org/project/onnxruntime-ep-mlx/)
[![Crates.io Version](https://img.shields.io/crates/v/onnxruntime-ep-mlx)](https://crates.io/crates/onnxruntime-ep-mlx)
[![Rust](https://shields.io/badge/-Rust-3776AB?style=flat&logo=rust)](https://rustup.rs/)

> **PyPI package: [`onnxruntime-ep-mlx`](https://pypi.org/project/onnxruntime-ep-mlx/)** — `pip install onnxruntime-ep-mlx`, `import onnxruntime_ep_mlx`. (Formerly published as `onnxruntime-mlx`, now renamed.)

An **MLX-native execution provider** for ONNX Runtime on Apple Silicon, built as an out-of-tree
**plugin EP** (ORT plugin-EP C ABI, ORT 1.29 / `ORT_API_VERSION 29`). It ships as a standalone
`libonnxruntime_mlx_ep.dylib` loaded by a stock prebuilt `libonnxruntime.dylib` via
`RegisterExecutionProviderLibrary` — **no ONNX Runtime fork required**.

The EP translates fused ONNX subgraphs into [MLX](https://github.com/ml-explore/mlx) graphs for
encoders, LLM prefill, and token-at-a-time decode.

## How it works

`ONNX fused subgraph → MLX graph → mlx_compile → mlx_eval → ORT outputs`

The EP translates supported ONNX regions into MLX, compiles reusable closures, and leaves unsupported
ops on ORT CPU. It covers common encoder and decoder operators, including quantized matmul, GQA/MHA,
PagedAttention, RoPE, RMS and gated RMS normalization, packed causal convolution, chunked
GatedDeltaNet, pooling, reductions, and shape operations. See
[`docs/OP_ARCHITECTURE.md`](docs/OP_ARCHITECTURE.md) for the coverage table.

Large, fully claimed regions are fastest. Compatible shapeless and shape-keyed regions can fuse,
with the stricter shape-keyed mode selected for the combined region. Dynamic GatedRMSNorm and
MRotaryEmbedding geometries are validated before compiled replay for each new input-shape/dtype
tuple; runtime position tensors remain live. Host-data-dependent operators still execute eagerly.
Autoregressive decode uses a
shapeless path so growing KV length does not retrace. Use
`ONNXRUNTIME_EP_MLX_VERBOSE=1` or `ONNXRUNTIME_EP_MLX_CLAIM_DEBUG=1` to diagnose fallback and
fragmentation.

### Packed Qwen state-space operators

`VarlenCausalConvWithState` supports token-major ragged batches, dilated depthwise convolution,
optional bias/SiLU, and activation-typed raw carry state. `GatedDeltaNet` supports the scalar-decay
`gated_delta` form, Qwen gate activations, optional Q/K L2 normalization, and inverse grouped-query
heads. Prefill uses the shared chunked batched-matmul core; single-token requests use a recurrent
step. Recurrent state and gate arithmetic stay FP32 with FP32/FP16/BF16 activation inputs.

Both operators read cumulative sequence boundaries on the host and run eagerly, with tensor
arithmetic on MLX. Compact speculative-state capture, per-key-dimension GatedDeltaNet decay, and
other GatedDeltaNet update rules are not supported and are declined. Tests cover Qwen3.5-0.8B
geometry, mixed-length batches, chunk tails, changing runtime shapes, and aliased state outputs.
Full-model GenAI generation has been smoke-tested and matched WebGPU token-for-token on the
benchmark prompts below. Broader correctness against the original model and performance tuning
have not been validated.

### Quantized weights and paged-cache execution

`MatMulNBits` accepts INT4/INT8 block sizes 16/32/64/128/256. Block 256 is lowered losslessly
to two MLX group-128 blocks sharing the original scale and affine bias; constant parameters are
expanded once and cached, and packed weights stay compressed. Block 16 retains the composed
dequantize-and-matmul path.

Standalone separate-cache `PagedAttention` with head widths up to 256 uses a direct block-table
GQA Metal kernel with FP32 online softmax. It does not expand KV heads or gather a contiguous
prefix. Cache updates are evaluated with attention, then only the changed physical slots are
written back. Verified aliased `present` outputs avoid full-cache copies; non-aliased outputs
receive the complete updated cache. Latent layouts, larger heads, and cache values used within
another eager subgraph retain the composed path. Set `ONNXRUNTIME_EP_MLX_NO_PAGED_GQA=1` for A/B
comparison with that path.

### Run the exported Qwen3.5 paged model

The example below uses ONNX Runtime GenAI's Engine API, which drives the paged model's packed
variable-length inputs and the model's chat template. `--ep` selects the provider without editing
`genai_config.json` (default: `mlx`):

```sh
export ONNXRUNTIME_MLX_EP_LIB="$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib"
python examples/qwen35_paged_generate.py \
  --model-dir /path/to/qwen3.5-0.8b-int4-paged-mlx \
  --max-new-tokens 128 \
  --warmup-tokens 8 \
  --prompt-repeats 1 4 16 \
  --runs 3 \
  --prompt "Why is paged attention useful?"
```

To compare WebGPU, first build its plugin from the ONNX Runtime checkout:

```sh
python tools/ci_build/build.py \
  --build_dir build/WebGPUPlugin --config Release --update --build \
  --use_webgpu shared_lib --build_shared_lib \
  --skip_tests --skip_submodule_sync --skip_pip_install --no_telemetry \
  --parallel 2 --target onnxruntime_providers_webgpu
```

Then run the same benchmark command with `--ep webgpu --ep-library
/path/to/onnxruntime/build/WebGPUPlugin/Release/libonnxruntime_providers_webgpu.dylib`.
Alternatively, set `ONNXRUNTIME_WEBGPU_EP_LIB`. `--ep cpu` needs no plugin, but the current CPU EP
cannot execute this exact packed hybrid graph: it has no kernels for `PagedAttention`,
`VarlenCausalConvWithState`, or `GatedDeltaNet`. A non-paged CPU-compatible export would be a
different workload, not an apples-to-apples provider comparison.

Run this with a source-built ONNX Runtime GenAI package that includes the Engine API and is ABI
compatible with the ONNX Runtime used to build the MLX EP. The released `onnxruntime-genai` 0.18.0
wheel uses the classic Generator API, which supplies rank-2 `input_ids` and cannot run this paged
model's rank-1 packed input. The GenAI source build used for validation also requires setting
`tokenizer_class` to `Qwen2Tokenizer` in `tokenizer_config.json`, because its pinned
onnxruntime-extensions revision does not recognize `Qwen3_5Tokenizer`.

The script repeats the prompt text to produce several input lengths and reports the exact encoded
token count for each. It runs an untimed warmup per length, then measures time-to-first-token (from
request submission through the first generated token) and steady-state decode throughput (excluding
the first token's latency). Prefix caching is disabled so repeated prompts still execute prefill.
Multiple measured runs report the median. Model/tokenizer loading and Engine initialization are
reported separately. `--results-json /path/to/results.json` saves timings and token IDs for comparison;
`--print-output` decodes text outside the timed region. Output length is a cap, so EOS may end a run
early; the actual generated count is printed. Use fewer prompt repeats or runs for a quicker pass;
`--mlx-verbose` prints provider claim and execution summaries.

#### Qwen3.5-0.8B provider comparison

On an Apple M1 with 16 GiB RAM, using ORT 1.31.0 Release and source-built GenAI 0.18.0-dev:

| Prompt tokens | MLX TTFT | WebGPU TTFT | MLX decode | WebGPU decode |
| ---: | ---: | ---: | ---: | ---: |
| 19 | 0.163 s | 0.095 s | 10.14 tok/s | 43.79 tok/s |
| 40 | 0.232 s | 0.158 s | 9.72 tok/s | 43.70 tok/s |
| 124 | 0.334 s | 0.350 s | 9.77 tok/s | 43.61 tok/s |

These are medians of three runs per prompt length, each generating 128 tokens after an eight-token
warmup. Both providers used the identical INT4/FP16 paged graph with prefix caching disabled and
were benchmarked sequentially after compilation finished. WebGPU was built in Release from ORT
commit `8755f3ef11`; graph capture was not enabled. Every measured output matched between providers.
WebGPU decoded 4.3-4.5 times faster than this MLX implementation; this is not a comparison of the
backends' theoretical limits. CPU could not load this graph and has no timing result.

## Requirements

- macOS on Apple Silicon, ORT 1.29 prebuilt (`ORT_API_VERSION >= 29`)
- CMake 3.25+, a C++20 compiler, `curl`, and `unzip` to build the pinned MLX runtime
- **`mlx-c` (and `mlx`) — a HARD build dependency**: run the setup block below and export both
  `MLX_PREFIX` and `MLXC_PREFIX` to its prefix. This pins MLX 0.32.2 commit
  `1f8e74e3` and mlx-c commit `c74db530`; wheels bundle these runtime libraries.
- A **Rust toolchain** (`rustup`) to build the EP from source

## Versioning (ORT compatibility)

A plugin EP targets one ORT C-ABI version. Package versions use
`0.<ORT_API_VERSION>.<patch>`:

| onnxruntime-ep-mlx | ONNX Runtime | `ORT_API_VERSION` |
|---|---|---|
| `0.29.x` | 1.29.x | 29 |

For example, ORT 1.28 moves the EP to `0.28.x`.

## Build

The EP is a Rust `cdylib` crate under [`rust/`](rust/). Point it at an ONNX Runtime C-API
include directory and `cargo build`:

```sh
setup_mlx=rust/scripts/setup_mlx.sh                     # repository checkout
test -x "$setup_mlx" || setup_mlx=scripts/setup_mlx.sh  # published Rust crate
"$setup_mlx" "$PWD/.deps/mlx-0.32.2"                   # HARD dependency (mlx-c + mlx)
export MLX_PREFIX="$PWD/.deps/mlx-0.32.2"
export MLXC_PREFIX="$MLX_PREFIX"
cd rust
# Either point ORT_INCLUDE_DIR at the ORT headers directly, or set ORT_HOME to an
# ONNX Runtime release root (build.rs will look in $ORT_HOME/include):
export ORT_INCLUDE_DIR=/path/to/onnxruntime/include   # or: export ORT_HOME=/path/to/onnxruntime-osx-arm64-1.29.0
cargo build --release
# => rust/target/release/libonnxruntime_mlx_ep.dylib  (registers the EP as "MLXExecutionProvider")
```

Set `MLX_PREFIX` and `MLXC_PREFIX` to link and bundle a custom MLX runtime
instead of the Homebrew installation. Each prefix must contain `include/` and
`lib/`; the wheel builder also bundles `mlx.metallib` and an optional
`libjaccl.dylib` from `MLX_PREFIX/lib`.

The crate binds the ORT plugin-EP C ABI and `mlx-c` directly via `bindgen`; it does **not** link
`libonnxruntime` (ORT is reached through the `OrtApi` function-pointer table passed to
`CreateEpFactories`).

## Install & use

### Python (recommended)

```sh
pip install -U onnxruntime-ep-mlx        # macOS/Apple-Silicon wheel; bundles the mlx runtime
```

```python
import onnxruntime as ort
import onnxruntime_ep_mlx

# Register the plugin EP once, then select it (with CPU fallback) like any provider.
onnxruntime_ep_mlx.register_execution_provider_library()          # name: "MLXExecutionProvider"
sess = ort.InferenceSession(
    "model.onnx",
    providers=["MLXExecutionProvider", "CPUExecutionProvider"],
)
out = sess.run(None, feeds)
```

`onnxruntime_ep_mlx` also exposes `library_path()`, `ep_name()`, `version()`, and
`append_to_session_options(so)`.

### C / C++ (or any onnxruntime binding)

Point onnxruntime at the built dylib and select the provider by name:

```c
// 1. Register the plugin library with the environment (once).
RegisterExecutionProviderLibrary(env, "MLXExecutionProvider",
                                 "/abs/path/libonnxruntime_mlx_ep.dylib");
// 2. Append it to a session's options (falls back to CPU for unclaimed ops).
const char* ep = "MLXExecutionProvider";
SessionOptionsAppendExecutionProvider_V2(options, env, &ep, /*count*/ 1, ...);
```

From Rust via **onnx-genai**: `ONNX_GENAI_EP=metal` +
`ONNX_GENAI_METAL_EP_LIB=/abs/path/libonnxruntime_mlx_ep.dylib`.

## Large-decoder partition metadata

The EP automatically infers residual layer boundaries from graph topology for decoders of about 24
layers or larger. Exporters can make the boundaries explicit with the
ONNX custom metadata key `onnxruntime_ep_mlx.layer_boundary_outputs`. Its value is a JSON array of
residual output tensor names, one per transformer layer:

```python
import json
import onnx

model = onnx.load("decoder/model.onnx", load_external_data=False)
entry = model.metadata_props.add()
entry.key = "onnxruntime_ep_mlx.layer_boundary_outputs"
entry.value = json.dumps(["layer.0.output", "layer.1.output", "layer.2.output"])
onnx.save(model, "decoder/model.onnx")
```

Metadata takes precedence over inference and does not rely on node names. The EP chooses a dynamic
group size of 4-8 layers, targeting about seven partitions. Override it with
`ONNXRUNTIME_EP_MLX_LAYER_PARTITIONS=<layers>`; use `0` or `off` to disable partitioning.

## Performance (M1 Max, warm)

Real end-to-end models, median of 10 runs, MLX EP vs the ORT **CPU** EP on the same machine — top-1
identical and max abs diff ≤ 6e-5 in every case:

| Model | Workload | CPU EP | MLX EP | Speedup |
|---|---|---:|---:|---:|
| Perch v2 | audio encoder (with DFT front-end) | 64.0 ms | 12.0 ms | **5.3×** |
| Perch v2 (no DFT) | audio encoder | 56.5 ms | 12.0 ms | **4.7×** |
| BirdNET | audio classifier (CNN) | 14.9 ms | 7.3 ms | **2.0×** |
| gemma-4-E2B | vision encoder (fp16 ViT) | 267 ms | 47 ms | **5.7×** |

Feed-forward encoders (audio / CNN / vision) are the EP's sweet spot: the whole graph fuses into a
single MLX closure that is traced + `mlx_compile`d once and replayed, so a static-shape model runs
end-to-end on the GPU with one dispatch (e.g. Perch: 725/725 nodes claimed, 1 fused subgraph).

Supported control flow stays inside that compiled boundary. `Scan` bodies are shape-specialized and
unrolled at trace time (including dynamic sequence dimensions), with per-input/per-output axes and
forward/reverse directions; carried-state `Loop` bodies are also unrolled. Host-readable `If`
branches and `Loop` trip counts are specialized and cached per decision (capped at 16 variants per
session; additional variants run eager), so surrounding tensor ops and the selected body replay as
one MLX closure. Data-dependent conditions, early-exit loops, and loop scan outputs remain on ORT
CPU.

Eligible BF16 INT4/INT8 prefill matmuls (block size 32/64/128/256, at least 32 rows) explicitly use stock
MLX's FP16 compute path by default. Set `ONNXRUNTIME_EP_MLX_BF16_QMM_FP16=0` to disable it.

The **Foundry Local** q4f16 decoders below run on the same M1 Max, warm, MLX EP vs the ORT CPU EP
(decode = 1 token with 128 past; prefill = 128-token step):

| Model | Arch | Prefill | Decode |
|---|---|---:|---:|
| Qwen2.5-0.5B | GQA, external rotary | **5.2×** | dispatch-bound (CPU-favored) |
| Phi-3.5-mini | Phi3, GQA | **5.29×** | 1.19× |
| Phi-4-mini | Phi4, long-context RoPE | **5.78×** | 1.10× |
| Mistral-7B-Instruct | GQA, growing KV | **11.89×** | **3.30×** |
| gemma-4-E2B | Gemma3n, 15-layer | 3.3× | **3.3×** |

Muse-Glimmer-30B INT4, with the optimized bundled MLX runtime and automatic 8-layer groups, reaches
**138.67 prefill tok/s** at 512 tokens and **14.79 decode tok/s** over 200 generated tokens. The
same-quantization llama.cpp baseline reaches **137.84 / 13.50 tok/s**.

Decode is weight-bandwidth-bound: small models can favor CPU, while larger q4 decoders benefit from
MLX. Unclaimed ops fall back to ORT CPU.

## Profiling & tracing (Perfetto)

The EP ships a built-in tracer (compiled in by default, **near-zero cost when off**). Recording is
gated entirely by environment variables — set one, run your model, and inspect the result.

**Get a Perfetto/Chrome trace.** Point `ONNXRUNTIME_EP_MLX_TRACE` at an output path; the JSON trace is
written when the inference session is torn down:

```bash
ONNXRUNTIME_EP_MLX_TRACE=/tmp/mlx_trace.json python your_script.py
# then open https://ui.perfetto.dev  (or chrome://tracing) and load /tmp/mlx_trace.json
```

The timeline shows one span per fused subgraph (`mlx.subgraph`), a nested span around the synchronous
`mlx_eval` (`mlx.eval` — its CPU wall time is the GPU-inclusive time of the whole fused subgraph),
per-op build spans with shapes/dtype/bytes, and counter tracks for GPU memory / utilisation. Ops that
fell back to a slower *composed* path (despite a fused kernel existing) are coloured distinctly with a
`reason=…`, and a top-10 slowest-ops summary is emitted at teardown.

**Lighter options** (no JSON file):

| Env var | Effect |
|---|---|
| `ONNXRUNTIME_EP_MLX_VERBOSE=1` | Print the end-of-run session summary (claim rate, compute-path breakdown, time attribution) to stderr. |
| `ONNXRUNTIME_EP_MLX_CLAIM_DEBUG=1` | Print each unclaimed node + the actionable reason (why the graph fragmented). |
| `ONNXRUNTIME_EP_MLX_SIGNPOST=1` | Emit `os_signpost` intervals so an Instruments *Metal System Trace* correlates. |
| `ONNXRUNTIME_EP_MLX_NO_STABLE_CROSS_CACHE=1` | Disable per-generation MLX reuse of immutable MHA cross-attention K/V inputs for performance A/B. |

**Per-kernel GPU detail (Xcode).** MLX hides its Metal command buffers inside one fused `mlx_eval`, so
the JSON trace times the fused eval as a whole. To see *inside* it, capture a boundary eval to a
`.gputrace` bundle (full per-kernel timing / occupancy / bandwidth) and open it in Xcode:

```bash
MTL_CAPTURE_ENABLED=1 \
ONNXRUNTIME_EP_MLX_GPU_CAPTURE=/tmp/mlx.gputrace \
ONNXRUNTIME_EP_MLX_GPU_CAPTURE_EVAL=5 \
python your_script.py
```

`MTL_CAPTURE_ENABLED=1` must be set before process start. `…_GPU_CAPTURE_EVAL` picks which eval to
capture (0-based, default 0); for decode, eval 0 is prefill/warmup, so pick a steady-state token.

## Concurrency

MLX evaluation is thread-affine. Use one `InferenceSession` per thread; do not call `Run()` on one
shared session from multiple threads.
Model loading may occur on a different thread: CPU/GPU streams are initialized lazily on the
first inference thread, which remains the owner for subsequent calls.

## Numerical accuracy

Outputs are tolerance-matched against ORT CPU but are not bit-identical. Long greedy generations can
diverge after near-tied logits because MLX and CPU use different floating-point reduction orders.

## Layout

```
docs/     design docs (DESIGN, OP_ARCHITECTURE, COMPILED_CAPTURE, MLX_EVALUATION)
rust/     the Rust EP: plugin-EP C-ABI vtables (factory/ep) + the modular ONNX->MLX
          translator (engine, registry, ops/*.rs) over a mlx-c RAII layer (mlx.rs)
python/   pure-Python pip package (onnxruntime-ep-mlx): a locator that bundles + registers
          the cargo-built dylib (hatchling build hook, hatch_build.py)
tests/    MLX op-correctness (tests/ops, pytest) + ONNX-standard conformance (tests/conformance)
.github/  CI (cargo build + op tests) and PyPI trusted-publishing workflows
```

## Testing

Build the EP (above), then run the pytest op-correctness suite (MLX vs ORT CPU reference):

```sh
export ONNXRUNTIME_MLX_EP_LIB=$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib
export DYLD_LIBRARY_PATH=<ort-prebuilt/lib>
python -m pytest tests/ops -q
```

- `tests/ops` — each translated decoder op via MLX vs. ORT CPU reference (tolerance-gated, pytest)
- `tests/conformance` — opt-in fuzz-conformance of the MLX EP against the ONNX standard
  (`cbourjau/onnx-tests`); see [`tests/conformance/README.md`](tests/conformance/README.md)
