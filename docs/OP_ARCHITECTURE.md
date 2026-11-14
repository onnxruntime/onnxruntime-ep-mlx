# ONNX→MLX Op Translation Architecture

**Status:** Final Rust post-pivot architecture
**Date:** 2026-07-13
**Repo:** `onnxruntime-mlx`
**Companion:** [`DESIGN.md`](./DESIGN.md)

---

## 0. Summary

The op architecture is now a Rust ONNX→MLX translator, not a hand-kernel registry.

`ep.rs` under `rust/src/` decides which ONNX nodes are claimable, asks ORT to fuse maximal convex supported partitions, and compiles each partition into an MLX-oriented node-descriptor plan. `rust/src/engine.rs` runs that plan through `mlx-c`, dispatching each node through a **modular, opset-aware, dtype-generic op registry** (`rust/src/registry.rs` plus per-family handler modules under `rust/src/ops/`). MLX is the **only** compute path for both prefill and decode.

The registered EP name is **`MLXExecutionProvider`** (the name passed to `RegisterExecutionProviderLibrary` and returned by the factory/EP `GetName`). The repo, vendor string, crate, and artifact are MLX-native:

- Repo/vendor: `onnxruntime-mlx`
- Crate: `rust/`
- Dylib: `rust/target/release/libonnxruntime_mlx_ep.dylib`

There is no independent Metal backend or shader registry. Specialized kernels,
including the opt-in packed GDN chunk kernel in `paged_ssm_chunked.metal`, run
through MLX's custom-kernel API. Op translations live in the modular registry
described in §2.2 — adding an op is one handler + one claim predicate + one
registration line in a single `rust/src/ops/<family>.rs` module, with **zero
`ep.rs` under `rust/src/` edits**.

---

## 1. Current pipeline

### 1.1 Claim

The EP claims only nodes whose domain, op type, dtype, attributes, and input form exactly match the translation inventory in §2. The claim set is intentionally a subset of what MLX can execute through this EP.

If a node is unsupported, ambiguous, or only supported by the old hand-kernel path, the EP does not claim it. ORT assigns that node to CPU.

### 1.2 Fuse

Claimed nodes are grouped into maximal convex connected ORT partitions for the plugin EP.
If interleaved groups create a cyclic quotient, topological contiguous runs retain every
supported node while preserving dtype and compile-class boundaries. The resulting fused
regions have an acyclic dependency graph.

### 1.3 Compile

`ep.rs` under `rust/src/` owns `Compile` and builds the ONNX→MLX plan. The plan records:

- The translated MLX operation sequence.
- Input/output binding information for ORT tensors.
- Constants and live-context data that should be converted and cached once.
- KV-cache output bindings used across prefill and decode.

### 1.4 Run

`rust/src/engine.rs` materializes and executes the MLX graph via `mlx-c`, dispatching each node through the op registry (§2.2). The runtime evaluates once at the fused subgraph boundary with `mlx_eval`, then writes outputs back to the ORT tensors expected by the session.

### 1.5 Fallback

Unclaimed nodes run on ORT CPU. This includes both genuinely unsupported ops and forms that are deliberately excluded from the translator, such as asymmetric `GatherBlockQuantized` with `zero_points`.

---

## 2. Authoritative op translation inventory

The following table is the current support contract. Do not broaden claims without adding the corresponding MLX translation and tests.

| ONNX op | Domain | Claimed ONNX form | MLX op(s) | Notes |
|---|---|---|---|---|
| **MatMulNBits** | `com.microsoft` | int4 block quantized weights, `bits=4`, `block_size=32` | `mlx_quantized_matmul` | Packed uint8 int4 weights are repacked once to MLX affine-quant format and cached persistently on the compiled plan. |
| **GroupQueryAttention** | `com.microsoft` | 9-input separate-QKV form; matching `fp32`/`fp16`/`bf16` Q/K/V/past_k/past_v/cos/sin; `int32` `seqlens_k` and `total_seq`; RoPE applied in-op | `mlx_fast_scaled_dot_product_attention` + `mlx_fast_rope` | Writes present K/V back to the same ORT ctx outputs in `[B, kv_heads, total_seq, head]` native-float layout. This keeps prefill→decode KV handoff layout- and position-continuous. |
| **RMSNormalization** | `ai.onnx` | Any valid suffix axis; broadcast scale; independent floating X/scale types | Fused last-axis vector form or composed statistics/affine | Live scale remains a dependency. Default epsilon is `1e-5`; explicit stash precision is honored. |
| **LayerNormalization** | `ai.onnx` | Any valid suffix axis; broadcast scale/optional bias; optional mean/inverse-standard-deviation outputs | Fused last-axis vector Y-only form or composed statistics/affine | Statistics retain stash dtype. Float64 tensors or internal statistics use an isolated MLX CPU stream. |
| **SkipSimplifiedLayerNormalization** | `com.microsoft` | `fp32`/`fp16`/`bf16` input/skip/gamma, optional bias, inverse statistic and residual output | skip-add + fused RMS normalization or composed saved statistics | Float32 accumulation; live RMS mean output remains unclaimed because its semantics are undefined upstream. |
| **GatherBlockQuantized** | `com.microsoft` | int4 embedding, symmetric (3-input, `zp=8`) **and** asymmetric (4-input `zero_points`) | gather + int4 dequant | Symmetric: `w=(q-8)·scale`. Asymmetric: `w=(q-zp)·scale`. |
| **Softmax** | `ai.onnx` | any valid axis; `fp32`/`fp16`/`bf16` | `mlx_softmax_axis` / `mlx_softmax_axes` | Opset <13 normalizes the complete suffix starting at `axis` (default 1); opset >=13 normalizes one axis (default -1). Legacy multi-axis suffixes use shape-keyed compilation because MLX selects the lowering using singleton extents; single-axis forms retain shapeless compilation. |
| **NonZero** | `ai.onnx` | schema 9/13; bool, all 8/16/32/64-bit signed/unsigned integers, fp16/fp32/fp64 and (schema 13) bf16 | provider-owned exact eager host scan + int64 MLX output | Live payload determines row-major coordinates and `[rank, count]` shape on every call. Scalar output is `[0, count]` per ONNX, unlike ORT 1.31 CPU's `[1, count]`. |
| **Unique** | `ai.onnx` | schema 11; bool, all 8/16/32/64-bit signed/unsigned integers, fp16/fp32/fp64; flat or axis slices, scalar/unknown rank/empty shapes, all optional outputs | provider-owned exact eager host sort/group + byte-preserving MLX outputs | [Contracts and authority limits](UNIQUE.md). Live cardinality and exact metadata; recursive eager parent boundaries. Legal NaNs use deterministic NaN-last sorting; NumPy half-axis ordering disagreement is explicit. Future28 dispatch is preserved; bf16 Unique28 remains unclaimed/unvalidated. Native validation pending. |
| **MatMul** | `ai.onnx` | rank >=1 floating operands, including vector-vector, vector-matrix and batched vector forms | `mlx_matmul` | ONNX vector promotion/squeezing and batch broadcasting are native MLX semantics. Empty contraction dimensions, empty batches and symbolic extents are supported. Float64 uses the isolated MLX CPU stream; integer MatMul stays on ORT CPU. |
| **GroupedMatMul** | `ai.onnx`, schema 29 (unreleased) | X[M,K], W[G,K,N] or transB W[G,N,K], indices[M,S] int32/int64, optional bias[G,N]; matching fp16/bf16/fp32/fp64 data | GPU gather-MM or direct QDQ/packed decode-compute; typed CPU gather + matmul | Output [M,S,N]. Live full-width route validation precedes dispatch. Both transB layouts avoid an expanded expert bank; fused kernels decode only selected experts. Empty/repeated/unused routes retain ONNX semantics. Native and isolated schema-enabled runtime qualification; released ORT 1.31 lacks the schema. |
| **CDist** | `com.microsoft` | v1; homogeneous `fp32`/`fp64` rank-two A[M,N], B[K,N]; default/explicit `sqeuclidean` or `euclidean`; C[M,K] | provider-owned eager typed host distances | [Contracts, resource bounds and remaining metrics](CDIST.md). Direct differences; no [M,K,N] workspace. No ORT CPU fallback or GPU-acceleration claim; recursive eager boundary, unchanged DOUBLE-body restrictions. |
| **Inverse** | `com.microsoft` | v1; homogeneous `fp32`/`fp64` rank >=2, square final axes, same-shape output, no attributes | eager `mlx_linalg_inv` on the run thread's reused default MLX CPU stream | [Contracts and limits](INVERSE.md). Typed LAPACK, explicit nonfinite-input errors, checked empty short-circuit and synchronous native errors; zero-trip Scan shape templates, recursive eager parent boundary, unchanged context stream and DOUBLE-body rejection. FLOAT16 promotion remains OPEN. |
| **FusedMatMul** | `com.microsoft`, schema 1 | matching fp16/fp32/fp64/bf16 A/B/Y, alpha and all four transpose flags | transpose + `mlx_matmul` + scale | Batch transpose rotates axis 0 behind all batch axes, before last-two transpose; either batch flag requires equal ranks >=3, matching ORT's compute helper. Vector last-two transpose is a no-op. Narrow scaling uses a dtype-rounded alpha and FP32 product/scaling before the final output cast, avoiding an intermediate fp16/bf16 result. Empty-result materialization uses shape-keyed compilation. Float64 retains the existing isolated MLX CPU stream. Source implementation and fixtures only; runtime/compiled evidence pending. |
| **GatherND** | `ai.onnx` | numeric/bool data; any valid `batch_dims`; statically known positive index-tuple width | flattened gather plus per-batch offsets | Preserves scalar slices, zero-size index grids/tails/batches, and live negative indices. Symbolic extents use shape-keyed compilation because flattening strides depend on concrete shapes. Float64 data uses the isolated MLX CPU stream. |
| **Add** | `ai.onnx` | `fp32`/`fp16`/`bf16` | MLX elementwise add | Floating forms only (fp32 via the residual-add predicate, fp16/bf16 via the elementwise predicate). |
| **Mul** | `ai.onnx` | `fp32`/`fp16`/`bf16` | MLX elementwise multiply | Floating forms only. |
| **Sub** | `ai.onnx` | `fp32`/`fp16`/`bf16`, `int64` | MLX elementwise subtract | Covers supported numeric/bookkeeping forms only. |
| **Sigmoid** | `ai.onnx`, `com.microsoft` | `fp32`/`fp16`/`bf16` | MLX elementwise sigmoid | Standalone `SiLU`/`Swish` are not claimed. |
| **QuickGelu** | `com.microsoft` | v1; `fp16`/`bf16`/`fp32`/`fp64`; FLOAT alpha default 1.702 | provider-owned eager typed sigmoid/product | [Elementary contracts and evidence](ELEMENTARY_CONTRIB.md). Independent mathematical reference; not a claim of CUDA execution or CPU bit identity. |
| **BiasAdd** | `com.microsoft` | v1; `fp16`/`fp32`; exactly X[N,S,C], bias[C], skip[N,S,C] | provider-owned eager typed (X + bias) + skip | No generic broadcasting. Exact live geometry and typed addition order. |
| **IsAllFinite** | `com.microsoft` | v1; homogeneous variadic `fp16`/`bf16`/`fp32`/`fp64`; scalar bool output | provider-owned eager exact predicates/reduction | Empty identity true; nonzero flags select NaN/Inf checks; both nonzero invalid. |
| **ScaledSiLU** | `com.microsoft` | v1; independent X/scale `fp16`/`bf16`/`fp32`; optional rank-zero scale overrides FLOAT alpha default 1 | provider-owned eager literal typed SiLU(scale * X) | All nine type pairs; round Z, sigmoid and product to X type. FastGelu remains a separate [OPEN audit obligation](ELEMENTARY_CONTRIB.md#fastgelu-remains-a-separate-open-lifecycle), not registered by this batch. |
| **Cast / CastLike** | `ai.onnx` | all pairs among bool, signed/unsigned 8/16/32/64-bit integers, `fp16`/`bf16`/`fp32`/`fp64` (13 types, including same-type) | MLX astype | Float64 conversions use the isolated MLX CPU stream; other types use Metal. CastLike reads only the target's declared dtype, not its values or shape. |
| **Div** | `ai.onnx` | same-dtype float or `int32`/`int64`, scalar-or-suffix broadcast | MLX divide; integer floor quotient corrected toward zero | MLX 0.32.4 floors integer division; opposite-sign inputs with a nonzero remainder add one to preserve ONNX truncation without float conversion. Zero and correction use the input dtype, including for values beyond exact floating-point representation. Integer division by zero and unrepresentable quotients are not defined by this support contract. |
| **Mod** | `ai.onnx` | Mod10/13, same-dtype numeric inputs with multidirectional broadcasting | magnitude remainder/sign restoration for floats; integer remainder/correction | [Versioned contracts](MOD.md): Mod10 permits integer `fmod=1`; Mod13 requires integers for `fmod=0` and floats for `fmod=1`. Floating arithmetic uses explicit MLX CPU/DOUBLE primitives to avoid Metal remainder precision loss. All forms are eager-only; integer zero divisors are validated on every call. Mod28 is not registered. |
| **Pow** | `ai.onnx` | Version-legal floating bases and numeric exponents; legacy `broadcast`/`axis` and modern multidirectional broadcasting | typed MLX power or exact integer-exponent decomposition | Output retains the base dtype. Integer exponents are not rounded through the base's floating dtype: full-width decomposition preserves sign/parity, including large odd exponents. Float64 calculations use the isolated MLX CPU stream, not ORT fallback. Integer bases remain unclaimed pending an authoritative overflow and negative-exponent contract; this is a semantic coverage gap, not a claim that MLX lacks integer arithmetic. |
| **Conv** | `ai.onnx` | rank-3/4, static positive W shape; symbolic X batch/channel/spatial dimensions allowed | MLX conv1d/conv2d/conv_general | W and group determine the required input channels; concrete channels are checked at execution. |
| **RNN / GRU / LSTM** | `ai.onnx` | layout=0, default activations, no sequence_lens; symbolic sequence length allowed | shape-keyed unrolling | Each concrete input shape gets its own trace. Empty sequences preserve initial states and produce an empty Y. |
| **Dropout** | `ai.onnx` | Versioned floating data, optional ratio/training controls and mask; legacy `is_test` | identity/ones for inference, uniform sampling and scaled masking for training | Opsets 1/6 honor `is_test`; 7/10 retain inference behavior because they provide no per-node training selector. Mask dtype follows the schema (data dtype before opset 10, bool thereafter). Modern controls are live scalar tensors, including intermediate producers; inference ignores the ratio value. BF16 data requires opset 13 and BF16 ratio requires opset 22. Training is an eager partition boundary, preserving compiled neighbors and fresh default keys; explicit INT64 seed bits are preserved. Float64 data uses the MLX CPU stream. |
| **RandomNormal / RandomNormalLike / RandomUniform / RandomUniformLike / Bernoulli / Multinomial** | `ai.onnx` | Existing versioned dtype/attribute restrictions; Like reads tensor geometry only; Normal supports bfloat16 output since schema 22; Bernoulli accepts float32/float16/float64 probabilities since schema 15 and bfloat16 since schema 22, with default or explicit native numeric/Boolean outputs | MLX random primitives | Unseeded forms are eager partition boundaries: each call consumes the live default key, including empty draws. Explicit float-seed forms retain the finite/non-negative/<2^64 claim guard and literal integer truncation; they remain fixed-key and shape-keyed when eligible. Normal's bfloat16 path keeps one existing F32 standard-normal draw/key on the context stream, then uses the current thread's existing default MLX CPU stream for F32 multiplication by the original FLOAT scale, F32 addition of the original FLOAT mean, and the final bfloat16 cast. This is provider-owned CPU affine execution, not an all-GPU path or ORT CPU fallback. It completes evaluation and checks CPU synchronization before exposing the result to consumers, also checking synchronization after construction/evaluation errors. This default stream can share other same-thread work; no exclusive ownership, new per-plan stream/worker/pool, or exhaustive native-error isolation is claimed. BF16 Normal/NormalLike and their recursive parents are eager even when seeded; old float32/float16 compilation behavior is unchanged. Parameters and intermediate samples are not pre-rounded to bfloat16; no new sampler, F64 Normal route or exact-real/higher-precision guarantee is introduced. Ordinary F32 arithmetic and final bfloat16 rounding can overflow or underflow; finite attributes do not guarantee finite outputs, and actual subnormal, signed-zero and cross-stream fidelity require runtime qualification. NormalLike defaults to the input dtype, including bfloat16 since schema 22; RandomNormal still defaults to float32. Bernoulli preserves its native bool draw and casts 0/1 exactly to the requested output type, including uint64 and DOUBLE. DOUBLE input or output selects the existing provider-owned MLX CPU-stream eager path, not ORT CPU fallback. Original DOUBLE probabilities are not downcast: native uint32 bits are compared in F64 against the F64 product of p and the native F32 upper constant (2^32+512). This preserves the native finite 32-bit draw/threshold policy, not a new sampler, 53-bit uniform resolution, exact-real probability guarantee, or cross-provider seeded sequence contract. The default output dtype remains the input dtype. Like's metadata claim accepts every legal input type in schema 1/22, including string/complex/double with explicit float32/float16 output dtype (also bfloat16 for NormalLike since schema 22), without converting its payload; non-MLX input types remain eager even with a seed. [ORT 1.31 initializer sizing](https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/core/framework/tensorprotoutils.cc#L1402-L1437) rejects complex initializers, as seen in [hosted run 39913638545](https://github.com/onnxruntime/onnxruntime-ep-mlx/actions/runs/39913638545/job/120282096717): these cases assert the exact session-initialization error against an independent CPU Shape control, not RNG execution. Scalar, empty and unknown-rank Like geometry is checked without narrowing wide dimensions. Recursive eager boundaries preserve neighboring deterministic compilation; control-flow bodies performing DOUBLE payload computation remain declined. Uniform bfloat16 output, Normal/Uniform DOUBLE outputs, full-precision DOUBLE uniform sampling and extreme uniform endpoints remain separate coverage gaps. |

### Opset 29 (ONNX main, unreleased)

The schema inventory is pinned to ONNX main
[`3c41120`](https://github.com/onnx/onnx/blob/3c41120b9c69657a1dd13407b080bd451ed06c9c/onnx/defs/operator_sets.h):
seven new operators and six schema updates. Source support does not upgrade the
host runtime's schema registry. ORT 1.31 rejects opset 29 by default; even with
`ALLOW_RELEASED_ONNX_OPSET_ONLY=0`, it lacks the new operator schemas. ORT main
[`6080d7c`](https://github.com/microsoft/onnxruntime/tree/6080d7c7cc598db70335be28c7946c429dfb2671)
also pins released ONNX 1.23.2 by default. An isolated build of that ORT main with
the pinned ONNX-main source substituted successfully executes the version-29
models through MLX, with CPU fallback and ahead-of-time function inlining
disabled and MLX provider execution verified in profiles. This is a custom
development runtime, not official ORT 1.31 support. Native handler tests and
ONNX-main reference fixtures remain separate evidence; unavailable released
runtime loading is explicitly skipped.

| Operator | Change and implementation |
|---|---|
| Expm1, Log1p (new) | Native MLX stable unary primitives, preserving tiny representable results and signed zero. fp16/bf16/fp32 on Metal; fp64 on MLX CPU. |
| LogCumSumExp (new) | Native inclusive forward MLX log-add-exp scan, arbitrary axis, fp32 accumulation for narrow inputs and fp64 for double. NaN and signed infinities propagate prefix-wise; empty tensors retain shape. Shape-keyed compilation. |
| GroupedMatMul (new) | Native gather-MM and selected-expert QDQ/packed kernels support both transB layouts without a materialized weight transpose. |
| Pack, Unpack (new) | UINT8 logical codes, UINT8/UINT32 words, arbitrary axis, bits 1-8 or grouped radix with the signed-INT64 power bound. See packed-format limits below. |
| DequantizePacked (new) | Exact little-endian GGUF block decoding in fp32, one final output conversion; all 27 normative formats implemented, including the seven IQ codebook formats below. |
| Gemm (updated) | Rank >=1 A, rank-2 B, last-two-axis transA (no-op for vectors), unidirectional C broadcasting, beta=0 ignores C. Floating dtypes retain native matmul and shape-keyed empty-result replay. Version 29 adds an exact eager host path for int32/uint32/int64/uint64 with checked i128 accumulation and integral scaling. |
| MeanVarianceNormalization (updated) | Version 29 uses centered variance and configurable epsilon after sqrt; earlier schemas retain their second-moment expression and fixed epsilon. |
| CausalConvWithState (updated) | Standard-domain dilation from version 29 uses native depthwise MLX conv1d and raw chronological carry of `(K-1)*dilation`. |
| GatherND (updated) | Existing int32/int64 path plus full-width live bounds validation for version 29, before any unchecked gather. Version 29 is eager. |
| ScatterElements, ScatterND (updated) | Version-29 mean includes original data once and all duplicate updates. Floating means use MLX scatter sums/counts, fp32 narrow accumulation, and a scaled retry for finite-sum overflow; double uses MLX CPU. Integer means use exact eager i128 host sums and floor division, avoiding 64-bit Metal scatter limitations. ScatterND accepts int32 indices and version-29 zero-width tuples. Other previously unclaimed reductions remain unclaimed. |

DequantizePacked supports `gguf:q1_0`, `q2_0`, `q4_0`, `q4_1`, `q5_0`,
`q5_1`, `q8_0`, `q8_1`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`,
`q8_k`, `iq4_nl`, `iq4_xs`, `tq1_0`, `tq2_0`, `mxfp4`, `nvfp4`,
`iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq1_s`, and `iq1_m`
(each identifier has the `gguf:` prefix).
IQ decoding uses the pinned normative codebooks and format-specific
bit/sign/scale layouts, with independently generated ONNX-reference fixtures
covering randomized nonzero single/multiblock data and scale/sign edge cases.
Pack/Unpack's bits spelling has vectorized MLX fast paths. Pack with bits 1–7
checks the live maximum via an MLX reduction and scalar read on every run, so
those nodes are eager; bits=8 retains shape-keyed compilation because all UINT8
codes are valid. Empty inputs return a correctly shaped empty tensor directly,
after Unpack validates its packed-axis length. Custom Metal decoding supports
Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 and Q2_K/Q3_K/Q4_K/Q5_K/Q6_K,
including lazy standalone decoding and the fused consumers described below.
Grouped radix and the other 16 quantization formats retain exact eager host
fallbacks; their correctness coverage is not a GPU-speed claim.
String/complex forms unsupported by the array boundary remain unclaimed.

Gemm-13 and Gemm-29 have the same integer dtype set: version 29 adds generalized
geometry, not new integer dtypes. The new integer path preserves full-width
values without floating-point narrowing, including values above 2^53.
Fractional/nonfinite or out-of-i128-range coefficients remain unclaimed; checked
accumulator overflow and results outside the output dtype produce actionable
errors rather than assuming unspecified wraparound or silently rounding.
MLX has no native integer GEMM primitive, so this is a correctness fallback, not
an accelerated integer matrix product. Older-version integer Gemm remains
unclaimed. Full-width fixtures use Python arbitrary-precision arithmetic because
ONNX's NumPy reference applies floating scalar multiplication and can itself
round large integers; smaller integral cases also match ReferenceEvaluator.

#### Reproducing schema-enabled runtime validation

Use fresh, ignored directories inside this worktree, not another checkout.
With the normal test dependencies installed and the pinned MLX plugin built,
download ORT `6080d7c` to `.deps/ort-opset29` and ONNX `3c41120` to
`.deps/onnx-opset29`. The validated Apple-arm64 build route is:

```bash
python -m venv --system-site-packages .venv-opset29
.venv-opset29/bin/python .deps/ort-opset29/tools/ci_build/build.py \
  --config Release --build_dir "$PWD/.deps/ort-opset29-build" \
  --parallel 6 --skip_tests --skip_submodule_sync --cmake_generator Ninja \
  --build_shared_lib --build_wheel --compile_no_warning_as_error --update --build \
  --cmake_extra_defines \
    "FETCHCONTENT_SOURCE_DIR_ONNX=$PWD/.deps/onnx-opset29" \
    FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER CMAKE_OSX_ARCHITECTURES=arm64 \
    onnxruntime_ENABLE_TELEMETRY=OFF onnxruntime_BUILD_UNIT_TESTS=OFF
.venv-opset29/bin/python -m pip install \
  .deps/ort-opset29-build/Release/dist/onnxruntime-*.whl \
  "onnx @ git+https://github.com/onnx/onnx.git@3c41120b9c69657a1dd13407b080bd451ed06c9c" \
  "protobuf>=7.36.2"
ALLOW_RELEASED_ONNX_OPSET_ONLY=0 \
ONNXRUNTIME_MLX_EP_LIB="$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib" \
  .venv-opset29/bin/python -m pytest -q \
    tests/ops/test_math.py tests/ops/test_reduction.py \
    tests/ops/test_gemm.py tests/ops/test_grouped_matmul.py \
    tests/ops/test_mean_variance_normalization_forms.py tests/ops/test_ssm_contracts.py \
    tests/ops/test_gathernd.py tests/ops/test_scatter_symbolic.py tests/ops/test_packed_ops.py
```

Changing the model's opset stamp, merely registering schemas, or allowing CPU
fallback does not validate these handlers. These tests run version-29 models,
disable function inlining, and require MLX provider profiles.
Validation passed 156 focused opset-29 tests without skips, 219 native Rust
tests, and 2,299 selected Python operator/regression tests. The only regression
skip was an unrelated negative loader contract specific to ORT 1.31, not missing
opset-29 execution.

### GroupedMatMul and direct QDQ products

The GroupedMatMul contract is ONNX main
[`3c41120`](https://github.com/onnx/onnx/blob/3c41120b9c69657a1dd13407b080bd451ed06c9c/onnx/defs/math/defs.cc),
including its standard-operator function decomposition and float64 support. The pinned
MLX GatherMM CPU kernel accepts only float32; the typed CPU fallback uses ordinary
Matmul (double GEMM for float64), not a narrowing cast. GPU routing does not construct
a gathered [M,S,K,N] expert-weight tensor.

`qdq.rs` recognizes default-domain DequantizeLinear weights feeding MatMul,
Gemm, or GroupedMatMul, optionally recognizing activation DQ too. Gemm and
GroupedMatMul support both transB layouts; Gemm also honors transA and its ordered
alpha/beta/bias epilogue. On Metal, rank-two activations
and rank-two dense/rank-three expert int4/uint4/int8/uint8 weights with float32 scale/output use a
private direct kernel. Per-tensor, per-axis (including negative axes), and blocked
parameters are indexed in their original ONNX layouts; live weights, scales and zero
points are never cached as constants. Weights need not be initializers. Activation
QuantizeLinear and output Q/DQ remain in the graph. Shared DQ consumers and graph outputs
are retained; MLX laziness avoids evaluating unused dequantized tensors.

For K <= 256 with both operands quantized, static positive power-of-two scales in
[2^-40,2^40], and scale/zero-point layouts constant along K, the kernel uses an exact
int32 dot before scaling. Each centered integer lies in [-255,255]; every partial
absolute sum is at most 16,646,400 < 2^24. DQ values, products and all partial sums are
exact normal float32 values under the exponent guard. Both final binary scalings remain
exact, finite and normal (or zero). Bias is added separately after the matrix result.

The general path reads quantized bytes directly, performs float32 DQ rounding in
registers, then uses SIMD-distributed float32 accumulation. With either packed operand,
it compensates rounded products using FMA residuals and TwoSum errors, including
the 32-lane reduction, before rounding the dot to float32. Bias is added only after
that rounding. The original FMA/SIMD sum is retained for the entire output if any
lane's operand, product, compensated partial/correction, or reduction becomes
nonfinite, or if the original reduction is nonfinite. Thus it preserves the
original exceptional arithmetic rather than promising exact-real recovery after
intermediate overflow. Underflow can lose product residuals; Metal subnormal fidelity
and blanket correctly-rounded dots are not claimed. Ordinary INT8/UINT8 general
arithmetic and the proven integer subset retain their existing numerical contracts.
It supports realistic K and blocked scales, but is **not integer-only ALU or broadly
bit-exact**. Finite packed regression references independently accumulate exact
products of the already-float32-rounded DQ operands using binary integer
coefficients, then round the sum to float32 with ties to even. Existing
`rtol=1e-5, atol=1e-6` thresholds are unchanged. Original NumPy float32 reductions
remain named disagreement diagnostics: neither their reduction order nor a fitted
output is the oracle. Exceptional regressions separately check the retained
No expanded floating activation or
weight matrix is produced by this kernel. Weight-only byte DQ additionally uses
bounded SIMDgroup matrix tiles for prefill and a SIMD matvec for contiguous K
decode; each weight is rounded to float32 before multiplication. Arbitrary
float32 QDQ is not rewritten to affine QMM: scale-after-dot/bias correction can
change ONNX DQ overflow/cancellation semantics. Non-float32 DQ, vector/batched MatMul, and CPU streams use ordinary DQ
and the existing unfused matrix-product path. `ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION=1` disables
recognition for same-binary comparisons. Nonempty dispatched products expose
`mlx_qdq_integer_matmul` or `mlx_qdq_matmul` in `op.fast` trace events; products
with either packed operand expose `mlx_qdq_packed4_integer_matmul` or
`mlx_qdq_packed4_matmul`, respectively. These names identify dispatch, not measured
speedup.

Packed INT4/UINT4 support starts at DequantizeLinear-21. Its
[versioned schema](https://github.com/onnx/onnx/blob/0834d5e3a5c41e6e3c36a6853e7540e81c704d07/onnx/defs/quantization/old.cc)
and [tensor storage definition](https://github.com/onnx/onnx/blob/0834d5e3a5c41e6e3c36a6853e7540e81c704d07/onnx/onnx.in.proto)
define globally flattened, low-nibble-first packing with ceil(N/2) bytes; there is
no row or expert padding. INT4 uses two's complement, not a biased unsigned code.
The [ORT Int4x2 transport](https://github.com/microsoft/onnxruntime/blob/v1.23.0/include/onnxruntime/core/framework/int4.h)
uses that same layout. Private kernels extract data and zero-point nibbles at their
logical indices, including odd K, odd expert boundaries, and blocked tails, without
an expanded integer or floating weight tensor. Packed and byte operand types may
be mixed; missing zero points are logical zero for the original ONNX input type.

Only initializers and ORT boundary inputs carry packed storage. An explicit resolver
keeps their logical shape and signedness separate from the one-dimensional MLX byte
array; generic dtype mapping, movement operators, QuantizeLinear packed outputs, and
copy-out are not broadened. Live input byte counts are checked with ORT
`GetTensorSizeInBytes`; owned initializers use the IR's checked storage size.
Logical dimensions and unpack indexing must fit int32 (at most 2^31-2 elements).
Ordinary DQ supports scalar, per-axis and statically valid blocked parameters,
float32/float16/bfloat16 output, scalar data and empty per-tensor/per-axis data.
Empty blocked geometry remains unclaimed under the existing block validator.
It unpacks only for legitimate unfused/shared consumers. Packed DQ is eager-only,
including nested-body capture checks; compiled input wrapping also rejects packed
types before treating a logical shape as a byte-array shape. Unsupported matrix
forms still use ordinary DQ rather than reinterpreting storage or approximating it.

**Decoder co-location:** capability classification ordinarily uses per-node compile
classes when it sees any `com.microsoft::PagedAttention`, or default-domain
`Attention` / contrib `GroupQueryAttention` / `MultiHeadAttention` with both
present-cache output slots 1 and 2 populated. Without pattern integration, packed
DQ's Eager class and FP32 MatMul's Shapeless class would prevent their union.
The QDQ pattern places only already-claimable, direct FP32 DQ producers
and their eligible default MatMul / Gemm / GroupedMatMul-29 consumer in the existing
Eager class. It requires rank-two activations, rank-two/rank-three weights as
appropriate, FP32 scales/output, and INT4/UINT4/INT8/UINT8 quantized operands.
Packed transport retains eager co-location even
when compute fusion is disabled; byte-only promotion honors that switch.
The signature predicates are shared with runtime
recognition; ordinary claims validate types/parameters first, and execution
independently revalidates live geometry and the GPU device.

All surrounding compile classes, attention-anchor detection, NativeRoPE routing,
convex/contiguous algorithms, unsupported-node barriers and explicit/inferred layer
cuts remain unchanged. Existing eligible Eager neighbors can join under those same
convexity/dtype rules; a class label is not a separate hard island boundary. Shared
DQ consumers still receive their original floating results, and an explicit layer
cut that separates the product from DQ is respected (ordinary DQ/matmul then runs).
Vector/batched, non-FP32 and unclaimable patterns are not promoted to packed-product
regions. Current-runtime decoder-shaped regression source connects packed MatMul
to cached Attention-24 with both present outputs, checks repeated live packed data
and shared results against independent references, and asserts actual eager
DQ/product co-location plus private dispatch with fusion enabled, absence with
fusion disabled, and no promotion for vector MatMul. This is bounded integration,
not a claim of full quantized-MoE model coverage.

**Runtime qualification:** ordinary QDQ regressions also run on released ORT.
GroupedMatMul-29 and packed-consumer models run on the isolated schema-enabled
ORT-main/ONNX-main build described above, with CPU fallback and AOT function
inlining disabled, provider profiles checked, and private dispatch asserted.
Released-runtime schema skips and native direct-handler tests are distinct evidence.

### FP8 DequantizeLinear and fused products

`float8.rs` defines FLOAT8E4M3FN, E4M3FNUZ, E5M2 and E5M2FNUZ decoding. MLX
has no float8 dtype, so `engine::resolve_float8` transports initializer and
boundary payloads as uint8 bits (copying unaligned initializer bytes); generic
dtype mapping and Cast remain unchanged. Standalone DQ decodes bits through an
exact inline GPU decoder (a 256-entry table on CPU), applies `(d - zero) * scale` in fp32 (zero point must
share the data format and scale shape), then rounds once to the output dtype.
FP8 DQ is eager-only, like packed transport.

FP32-output FP8 weight DQ feeding MatMul / Gemm / GroupedMatMul-29 fuses
without materializing a floating weight on the decode path. Kernels decode
inline with exact fp16 bit construction (four bytes per `uint`; E5M2FNUZ uses
the generic decoder since exponent 31 is finite) and round each element to fp32
before an fp32 dot:

| Shape | Kernel (`op.fast`) |
| --- | --- |
| M≤8, K-contiguous weight (Gemm transB=1, transB GroupedMatMul), K%8==0 | `mlx_fp8_gemv_k`: SIMD-group per 4 output rows, M tiled by 4 |
| M≤8, N-contiguous weight (MatMul, Gemm transB=0), N%4==0 | `mlx_fp8_gemv_n`: K split across threadgroups, then sum |
| M>8, not grouped | `mlx_fp8_dequant_matmul`: one GPU decode to the weight's consumer dtype, then MLX GEMM |
| other (odd N/K, transA, grouped M>16) | `mlx_fp8_matvec` / `mlx_fp8_tiled_matmul` reader |

MatMul with rank≥3 X is flattened when the weight is FP8. The recognizer follows
an optional FP16/BF16 Cast and rank-two Transpose after FP32 weight DQ. Decode
rounds each scaled weight through that Cast before FP32 accumulation; prefill
decodes directly into the consumer dtype. Product output rounding precedes the
ordinary Gemm epilogue or Add bias. Shared DQ/Cast outputs retain their ordinary
values. Cast fusion is FP8-only: INT8/UINT8 DQ followed by FP16/BF16 Cast retains
ordinary compiled matrix products, while direct FP32 integer DQ through a rank-two
Transpose still fuses. Packed INT4/UINT4 transport remains eager independently
of fusion. Partition discovery checks these same dtype and view constraints.
FP8 activation DQ stays in the graph and supplies floating activations
to the weight-only kernels; its byte codes never enter integer QDQ readers.
Scalar FP32, saturated E4M3FN QuantizeLinear uses device round-to-nearest-even
encoding (including subnormals and signed zero), enabling activation Q/DQ patterns.
Other FP8 quantization forms, FP16/BF16-output weight DQ, and device-tensor FP8
inputs remain outside fusion.

`MatMulBlockQuantizedFp8Weight` uses the same GPU decode/prefill kernels with
FP16/BF16 weight/output rounding, partial final K blocks, optional bias and
activation Q/DQ. Its CPU path retains host decoding. FP8 transport and product
geometry are eager-only. `ONNXRUNTIME_EP_MLX_NO_QDQ_FUSION=1` disables standard
pattern fusion, not the contrib operator's native implementation.

`test_fp8_fusion.py` checks all 256 codes of each format through every kernel
above, including transA, odd-N/K and grouped routes, and asserts the kernel
via trace. GroupedMatMul-29 cases skip on released ORT 1.31; they pass on the
schema-enabled runtime described in "Reproducing schema-enabled runtime
validation".

Synchronized session runs on M1 Max (E4M3FN, block 128 along K, median µs, 400
runs; footprint MB; unfused is the same binary with fusion disabled):

| Case | Fused | Unfused | Dense fp16 | Dense fp32 |
| --- | --- | --- | --- | --- |
| Gemm transB 4096², M=1 | 394 (145) | 1443 (399) | 314 | 411 |
| Gemm transB 4096², M=8 | 515 | 1468 | 374 | 460 |
| Gemm transB 4096², M=128 | 1429 (277) | 2145 (416) | 905 | 1161 |
| MatMul 4096², M=1 | 377 | 1543 | 379 | 522 |
| MatMul 4096², M=8 | 451 | 1469 | 370 | 434 |
| MatMul 4096², M=128 | 1308 | 2003 | 917 | 1010 |
| Gemm transB 8192², M=1 | 540 (323) | 5074 (1089) | 657 (566) | — |
| MatMul 8192², M=1 | 605 | 5383 | 911 | — |

Times include roughly 0.2–0.25 ms of session overhead. The prefill route
allocates a transient fp32 weight per run.

GroupedMatMul-29 on the schema-enabled runtime (8 experts of 2048², two
selections; E4M3FN, block 128 along K, median µs over 300 runs; footprint MB):

| Case | Fused | Unfused |
| --- | --- | --- |
| transB=1, M=1 | 349 (227) | 2447 (609) |
| transB=1, M=8 | 468 | 3043 |
| transB=1, M=64 | 1989 | 7005 |
| transB=0, M=1 | 334 | 2485 |
| transB=0, M=64 | 2002 | 6546 |

### Packed static-weight consumers

`packed/fusion.rs` recognizes DequantizePacked feeding MatMul, Gemm,
GroupedMatMul and Gather through bounded constant Reshape/Transpose chains.
`conv/packed.rs` recognizes unpadded, unit-stride, group-one pointwise Conv in
one to three spatial dimensions. Decode/view/consumer atomic regions preserve
these patterns in decoder partitions; explicit layer cuts remain respected.
Shared decoded outputs keep their original nodes and dtype conversions.

The 11 Metal formats listed above share one normative FLOAT decoder.
`matmul/fused.rs` loads decoded weights into bounded tiles, converts to the
graph dtype before multiplication, and accumulates in float32. FP16 tiles are
used only for FP16 graphs; FP32/BF16 operands are not silently narrowed to FP16.
Both expert layouts map source addresses directly, without a floating transpose
or [M,S,K,N] expansion. Repeated experts are legal; routing buckets reuse tiles
only for selected experts. Gather decodes only requested embedding rows.
Bias follows the original rounded product; activation/residual epilogues are
not additionally fused.

For dense FP32 prefill with M>=33, immutable host-backed weights, an affine
matrix layout and a physical row width divisible by 32, `packed/qmm.rs`
repackages Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q4_K/Q5_K codes and compact
scale/bias parameters once into native MLX QMM storage. A one-time host check
rejects nonfinite scale/minimum fields from this optimization; it does not
decode float weights. The GPU repacking and native FLOAT tile reader never
write a full floating weight. Half scales, bounded K-subscales and small codes
make the affine products exactly representable in FLOAT; this justification
does not extend to arbitrary QDQ scales. Native QMM is not used for narrow
output dtypes, where earlier rounding could change the decoded weights.
The minimum M exceeds every vector cutoff in pinned MLX 0.32.4, ensuring
matrix readers rather than scale-after-dot vector arithmetic.

Dense custom tiled prefill regressed against native GEMM in measurements and
is not selected by default. Dense M>4 profiles outside the native-QMM subset
(including Q2_K/Q3_K/Q6_K, narrow dtypes, live weights, nonfinite parameters,
and unsupported views/widths) retain ordinary GPU decode plus matmul. Small
dense products and routed experts retain direct kernels. Thus fallback profiles
may materialize floating weights; the no-full-weight claim applies only to
eligible fused routes. Arbitrary QDQ view chains,
general spatial/grouped Conv, and the other 16 packed formats' GPU fusion
remain unimplemented, not hardware-impossibility claims.

DequantizePacked checks each logical dimension and its one-dimensional encoded
byte count against the MLX int32 dimension limit, but uses checked size_t
logical offsets and a two-dimensional standalone decode grid. A live Q4_0
bank [128,4097,4096] executes selected experts across the signed-int32 logical
offset boundary without an 8.59 GB floating bank. This is execution evidence,
not a steady-state timing or process-RSS measurement.
See [synchronized consumer benchmarks](../bench/README.md#static-weight-decode-consumers).

The FusedMatMul source contract follows immutable ORT
[`402979ac`](https://github.com/microsoft/onnxruntime/tree/402979ac1bef89dcd55add41eaa7277c407599eb):
`contrib_defs.cc` for schema/vector behavior, `matmul_helper.h` for batch permutation/rank rules,
and CUDA `math/matmul.cc` plus `shared_inc/fpgeneric.h` for narrow alpha conversion and FP32
accumulation. Nonzero INT flags are interpreted at full width, as in the runtime kernels;
upstream shape inference truncates them to C++ `int`, so unusually large flags can infer
different extents. No inference workaround or global claim/stream boundary is introduced.
The schema does not prohibit nonfinite alpha; the fixtures use an explicit NumPy oracle
for those cases because ORT's double CPU constructor rejects them. CPU comparisons are
restricted to its registered float/double forms, never exception-driven oracle substitution.
For a zero contraction length, the result is literal positive zeros before alpha conversion
or scaling, matching ORT CUDA even for nonfinite alpha or a finite fp16 alpha that overflows
when narrowed. The zero-K fixtures cross all four transpose flags, dtype families, symbolic
replays, vector promotion, and empty-output cases with an explicit zero-fill oracle.
Tests in `tests/ops/test_fused_matmul.py` cover default/eager requests, not yet proven compiled
or CPU-stream execution. Absence of an earlier native registration does not establish a
baseline failure: ORT expansion/rewrites and actual placement still require admitted execution.

Runtime-bounded INT64 **Range** and modern runtime-repeat **Tile** use eager MLX
translation: live values determine output extents and are refreshed before compiled consumers.
Constant initializer forms remain supported. The exact native-attention position proof
preserves its existing Range/Tile capture exception.

Modern **Tile-6/13** INT64 repeats use checked rank-one/cardinality, nonnegative
full-width per-axis multiplication, native int32 output extents and checked element/byte bounds.
Every axis is validated even when another axis is zero. Valid empty results use typed zeros
before importing data or narrowing repeats; `0 * INT64_MAX` is zero, not an overflowing repeat.
Scalar data requires a rank-one empty repeats vector and is an identity. Known declared output
dimensions must agree; unknown rank/extents remain live metadata, not frozen scalar/zero values.
Movable real numeric/bool payloads retain their dtype (BF16 since schema 13, DOUBLE on the
existing MLX CPU stream); string, complex and encoded transports remain gaps.

The pinned MLX Tile implementation constructs concrete reshape/broadcast/reshape targets.
With constant repeats, fixed nonempty geometry and scalar identity are capture-safe; symbolic,
unknown-rank and potentially empty forms are shape-keyed. Known wide input metadata canceled by zero repeats
is eager-only so compilation cannot pre-import an unrepresentable input. Actual ORT boundary
acceptance of extreme empty metadata is separate from the checked geometry contract.
Live boundary and produced repeats are reread on every eager invocation after rank, cardinality
and INT64 metadata validation; the returned value count must match the metadata. Unknown metadata
is deferred to runtime. Generic live Tiles and recursively containing parents are eager partition
boundaries; eligible neighbors retain their own compile routes. They are not added to generic
host-parameter value keys, and even shape-constant live repeats cannot be captured.
The existing native RoPE identity condition and Range/Tile Q8/position proofs are unchanged.
Only the sole top-level live Tile owned by `Plan::new`'s exact native-attention proof regains
its prior shapeless descriptor, after that proof and before route selection. The broad Q8
capability override preserves the candidate partition, not capture permission. Native Tile
admission shares the ordinary Tile payload validator: both tensor metadata records are required,
the input must be movable, the output dtype must match, and BF16 requires effective schema 13.
Unknown rank/extents are not missing tensor metadata. This payload-only guard does not re-run
ordinary repeats or geometry checks or alter the Range exception or native detector/proof.
Near misses and nested live Tiles stay eager; translation defensively rejects capture after
the native shortcut.
Legacy Tile-1 has a distinct three-input signature and remains a separate authority/support
gap, not a modern alias or a demonstrated backend impossibility. These are bounded source
contracts and regression fixtures, not proof of every combination or physical GPU execution.
Live-repeat fixtures are source-only pending independently authorized CI qualification.
The native-admission source regression imports opset 14 (effective Tile schema 13) and retains
an unrelated constant-INT64-repeat string/complex Tile as a graph output with optimizations
disabled. Capability traces require the supported live INT64 Range and Tile to remain admitted
with an unrelated constant-repeat Tile. Rust plan tests separately guard the exact
native-position capture exception. These assertions have
not been executed locally and are not device reproduction or numerical evidence.

**NonZero** uses exact original representations, not float32 conversions: signed floating zeros
are false, NaNs/infinities/subnormals are true, and large integers retain all bits. Empty tensors
produce `[rank, 0]`. Symbolic extents and unknown rank resolve from live input metadata; in-graph
inputs are resolved and synchronously materialized before scanning. The eager partition boundary
excludes NonZero from general/decode/prefill capture without disabling eligible adjacent
producer/consumer compilation. Host scanning costs linear CPU work and may force a GPU
synchronization/materialization; it is provider-owned execution, **not GPU acceleration**.
Input extents and nonempty input/output element counts must fit native int32 indexing.
Complex64/complex128 and string forms admitted by the ONNX schemas remain **unimplemented
coverage gaps**, not backend impossibilities; they need end-to-end provider dtype/host-value
support and dedicated oracles. ORT 1.31 CPU itself implements only bool/uint8/int32/int64/fp32
NonZero and disagrees with ONNX on scalar output rank. No tolerance or fallback is used to
hide these differences. Unique shares the checked exact live host reader, without changing
NonZero's coordinate/equality rules; see [Unique contracts](UNIQUE.md).

**GridSample** supports opset-16 2D `bilinear`/`nearest`/`bicubic` and opset-20+
general-N `linear`/`nearest`/`cubic`, all three padding modes (`zeros`, `border`,
`reflection`), and either `align_corners` value. Floating payload and coordinate types are
independent (`fp16`/`fp32`/`fp64`, plus `bf16` from opset 22). Float64 stays on the MLX CPU stream.
Nearest uses ties-to-even and exact gathers for all supported integer/bool payloads, without
lossy float conversions. Symbolic geometry, empty batch/channel/output axes, singleton input
axes and changed live payload/grid values use checked shape-keyed execution. Input spatial
axes must be positive; nonempty flattened counts and separable intermediates must fit int32 indexing.
Empty outputs read only input metadata, so an empty batch can retain large spatial extents
without constructing a sampling-point count or importing unused payloads; output strides are checked.
Higher-rank interpolation contracts spatial axes rather than allocating an exponential map.
Reflection reduces normalized coordinates periodically before they can overflow or lose pixel
offsets. Axes longer than float32's exact integer range use binary-rational integer indexing and
ties-to-even, rather than rounding image bounds or nearest indices through float32.
Ordinary axes retain fp32 coordinate arithmetic (fp64 if either tensor is double), matching
ORT's accumulation-precision denormalization: a perturbation smaller than that arithmetic can
round away before nearest-even. Standard even-axis subnormal controls distinguish this from
Metal flushing and from exact-real binary-rational indexing; they do not claim exact-real
coordinate arithmetic for every ordinary-axis grid.
Non-floating linear/cubic remains an **open semantic-contract gap**, not a backend impossibility:
the schema admits these types but specifies no intermediate precision/cast rule, ORT 1.31
has no corresponding CPU kernel, and ONNX's reference casts interpolation coefficients to the
payload type. String/complex payloads lack the EP's end-to-end dtype mapping and remain unclaimed.
The numerical oracle is independently derived and cross-checked against ORT's implemented forms
and ONNX's general-N reference, not inferred from successful registration.

**RoiAlign** supports default/adaptive `sampling_ratio=0` and positive fixed ratios,
`avg`/`max`, both coordinate modes, and the opset-10/16 default transition. X, ROIs and
output share one floating dtype (`fp16`/`fp32`/`fp64`, plus `bf16` from opset 22);
batch indices are int64. Adaptive sampling reads current ROI values on every eager
invocation and schedules each ROI's ceiling-sized grid independently. Fixed grids
retain shape-keyed compilation, with live boundary-coordinate and batch-index checks
before every compiled apply; interior controls execute eagerly. Float64 uses actual
double-precision host arithmetic on the isolated CPU route, not a float32 round trip.
Max pooling reduces weighted corners as specified by the pinned ONNX reference and
ORT implementation, rather than reducing bilinear sums. Nonpositive adaptive grids
produce zero; `output_half_pixel` first clamps ROI extents to at least one.
Empty ROI/channel outputs are supported; spatial axes remain positive. Coordinates
and spatial scale must be finite, batch indices must be in range, and nonempty
grids/output counts must fit checked int32 indexing.

Cast/CastLike integer narrowing discards high bits, floating-to-integer conversions truncate toward
zero for representable values, and numeric-to-bool uses zero/nonzero (including signed floating
zeros, infinities and NaNs). Floating-to-integer out-of-range values are undefined by ONNX and are
not assigned a new contract here. Tests cover the full type-pair matrix against NumPy and ORT CPU,
large integer rounding, integer boundaries, floating subnormals/signed zeros and eager/compiled
execution. An empty or changing-shape CastLike target does not become a compiled data dependency;
other consumers of that tensor still read its live values.

String, float8/float4 and packed int4/uint4 conversions remain on ORT CPU: the pinned MLX dtype
mapping/conversion backend does not support those ONNX types. MLX complex64 is not a Cast/CastLike
option because ONNX's schemas exclude complex types. The float8-only `saturate`/`round_mode`
attributes are not approximated for unsupported types; those nodes retain CPU fallback.

### 2.1 Numeric `ai.onnx.ml` acceleration

The EP claims the numeric tensor forms of **ArrayFeatureExtractor, Binarizer, FeatureVectorizer,
Imputer, LabelEncoder, Normalizer, OneHotEncoder, Scaler, LinearClassifier, LinearRegressor,
SVMClassifier, and SVMRegressor**. String/map forms remain on ORT CPU. TreeEnsemble operators are
also left on CPU because the MLX translation was substantially slower than ORT's optimized CPU
kernel. The retained SVM path is compute-bound enough to benefit from Metal; a 1024×128 RBF
regression case with 256 support vectors measured about **10.9×** faster than ORT CPU on the
development machine.

### 2.2 Coverage and form restrictions

Registrations cover numeric/Boolean tensor operators and selected contrib operators,
but registration alone does not mean every schema version, attribute combination,
dtype, optional input/output, rank, or dynamic/empty shape is supported. The
per-node claim predicate and compile policy define the implemented subset.

The MLX array boundary carries bool/int/uint 8-64/fp16/bf16/fp32, with selective
float64 support on isolated MLX CPU streams. Coverage includes elementwise/math/trig/activations,
logical/bitwise, all reductions, shape/data-movement, normalizations, attention, MatMul/Gemm,
conv/pooling, quantization (Quantize/Dequantize/DynamicQuantize/MatMulInteger/ConvInteger/
QLinearMatMul/QLinearConv), random, signal/FFT (DFT/STFT/windows/MelWeightMatrix — audio),
vision/detection, **recurrent (RNN/GRU/LSTM and
com.microsoft LinearAttention via stable Metal prefill scans for eligible `Dk <= 512`,
with static unrolling for single-token, larger-width, and CPU-stream calls)**, **control flow (If/Scan/Loop via recursive
subgraph-body translation)**, and host-computed data-dependent operators.

LinearAttention preserves the original typed scan for `Dk <= 256`. The separate
`257 <= Dk <= 512` scan includes BF16 activation/output and recurrent-state
storage. It widens original live inputs and
keeps FP32 working state, rounding only output and final/window state stores to
their declared types.
Stored window slots do not round the ongoing recurrence. This preserves the
recurrence, not literal intermediate half-precision rounding.
For BF16-associated calls in this wide range, the composed path also uses FP32
working state and readout, including scan-disabled, single-token, and CPU-stream
calls. Query scaling follows the FP32 readout; inputs are not first quantized to
the state storage dtype. Each final/window state is stored as S and each output
as T, without feeding window-store rounding into the next token. Empty-time calls
still return the unchanged stored state. This intentionally replaces historical
wide-BF16 primitive-S arithmetic and its no-fast-scan expectation, following
official ONNX/CUDA-first precision rather than typed compatibility.
The original `Dk <= 256` and `Dk > 512` precision policies, and non-BF16
single-token paths, are unchanged. This is a scoped correctness candidate, not
general standard-domain precision compliance, numerical acceptance, or a BF16
performance claim.
For source isolation, the wider path leaves the original lazy preparation in
place. Those unused arrays are not additional evaluation roots, but compiled
`trace_transient` retains their arena handles until plan teardown. Fresh resource
and performance evidence must include that per-cached-trace ownership cost;
results for the earlier typed wide scan do not validate this implementation.

Before schema 29, `MeanVarianceNormalization` follows the schema-9/13 second-moment expression:
the epsilon is added after the square root, not inside it. Default axes are
exactly `[0, 2, 3]`; explicit empty axes reduce all dimensions, including scalar
inputs. Axes are checked at full integer width before native reduction. Narrow
floating inputs use float32 arithmetic and round the result back to the input
type; float64 uses MLX CPU. Empty outputs are built from live metadata without
reading payloads. Dynamic ranks and empty geometry use shape-keyed compilation,
so replay cannot freeze a first-call axis list or empty-output branch.

These value-kind forms remain on ORT CPU because the current array-only EP boundary
cannot materialize their results:

| Category | Ops |
|---|---|
| Sequence-valued outputs | SequenceConstruct/Empty/Erase/Insert/Map, SplitToSequence |
| String tensors | RegexFullMatch, StringConcat, StringNormalizer, StringSplit |
| Optional value | Optional |

`Shape` and `Size` read tensor geometry without converting payloads into MLX
arrays. This includes string and encoded float8/float4/int4/uint4 tensors that ORT
can materialize, string initializers, changing ranks, scalar tensors, and empty
tensors with dimensions beyond MLX's int32 shape range. `Shape` honors clipped
positive/negative `start` and `end`; `Size` returns a checked int64 element count
and zero for any empty dimension. Non-MLX payloads use eager metadata reads;
dynamic or intermediate numeric geometry uses shape-keyed compilation. This does not enable
string/encoded payload outputs or control-flow array boundaries.
Fully static graph-input/initializer metadata retains shapeless eligibility.
Double inputs captured by `If`, `Loop`, or `Scan` bodies may be read by `Shape`
and `Size` without importing their payloads. These bodies and their enclosing
control-flow nodes stay eager, including for static geometry; unrelated numeric
work retains its existing compilation eligibility. Actual Double computation
inside control-flow bodies remains declined.
Existing host-value specialization and eager fences still protect geometry
controlled by changing parameter values; root input shapes alone are not a
sufficient cache key. General capture value-specializes supported boundary
parameters; decode and prefill reject host-read parameters unless they are
immutable or proven shape-constant. This also protects live Reshape/Expand
outputs when an unrelated metadata input has fully static geometry.

`Loop` reads its trip count and initial condition on the host. As with `If`,
node-produced controls are left to ORT CPU instead of being fused with their
producers into a plan whose intermediate values cannot be read as host inputs.
Graph-input and initializer controls retain live-value specialization.

`Flatten` validates full-width axes with the schema's bounds (nonnegative before
opset 11) and checks both folded dimensions before narrowing to MLX int32 shapes.
Products account for zero extents before multiplication, including scalar
axis-zero inputs and large empty shapes; unrepresentable folded dimensions
remain on CPU when known, or report an explicit runtime EP error otherwise.
Dynamic or unknown-rank Flatten geometry uses shape-keyed compilation because
its reshape target contains concrete products. `Transpose` requires a genuine
permutation of every nonnegative input axis; negative, repeated and truncated
wide axes are not normalized into valid permutations. Omitted permutations
reverse the actual rank, including scalars; unknown ranks are shape-keyed.
Both views preserve exact numeric/Boolean payload bits, with float64 on MLX CPU.

`Squeeze` and `Unsqueeze` support legacy axes attributes and live int64 axes
vectors, including omitted optional Squeeze slots, unknown ranks, scalars and
empty dimensions. Full-width axes are validated before narrowing; Unsqueeze
requires unique normalized axes, while Squeeze removes repeated selections
once, matching ONNX shape inference and ORT. Nonempty boundary axes use bounded
full-value general-cache keys; empty or data-dependent interior axes stay eager.
Changing ranks or singleton selection cannot enter unguarded shapeless
decode/prefill traces. Exact numeric/Boolean payloads retain their bits; float64
uses MLX CPU. Modern explicitly empty axes mean identity, as specified by ONNX
shape inference/reference, even though ORT 1.31 squeezes all singleton dimensions.
Legacy explicitly empty Squeeze attributes remain unclaimed when singleton
dimensions are possible because ONNX inference disagrees with CPU/reference.
String, complex, encoded payloads and native-wide dimensions remain outside this
MLX array materialization path.

Sequence **inputs** are supported by `SequenceLength`, `SequenceAt`, and
`ConcatFromSequence` from opset 11 onward. These handlers execute eagerly and
re-read the live ORT sequence on every invocation. Element copies own their MLX
storage before temporary ORT values are released. `SequenceLength` reads only the
container count, including string-tensor sequences, and returns a scalar int64.
`SequenceAt` accepts scalar int32/int64 positions, including negative indices.
Element access is bounded by ORT's signed 32-bit `GetValue` index API.
`ConcatFromSequence` supports concatenation or stacking (`new_axis=0/1`), negative
axes, and empty tensor dimensions; empty sequences, incompatible shapes, and
invalid indices/axes report errors. Payload outputs support the native movable
numeric/Boolean types, including precise float64 on MLX CPU, but not strings or
complex tensors. Sequence-producing nodes remain on CPU; their tensor-sequence
outputs can feed these MLX consumers across the ORT boundary.

`Slice` supports legacy starts/ends/axes attributes and modern int32/int64
control tensors, including omitted axes/steps, negative axes, reverse slices,
empty results, and changing boundary or intermediate controls. Lengths, unique
axes, axis bounds, vector ranks, and nonzero steps are checked before indexing.
Full-width integer bounds and strides are preserved; strides larger than the
dimension are clipped in integer arithmetic because they select at most one
element. Boundary values use the existing bounded general-capture keys, while
unguarded interior controls execute eagerly. Scalar inputs and known input
dimensions beyond MLX's int32 array geometry remain unclaimed.

`Pad` handles version-1 `paddings` and version-2 size/value attributes, modern
int64 `pads`, and optional int32/int64 axes. Constant and live boundary
parameters use the existing bounded full-value general-cache keys; interior
data-dependent pads/axes stay eager. The fill value remains live graph data,
not a host specialization parameter. Signed pads crop before edge, wrap and
reflection padding. Constant mode preserves original-coordinate overlap,
including removal beyond an edge compensated by padding at the other edge.
Empty/scalar outputs, exact numeric/Boolean payloads, and float64 CPU movement
retain their types and bits. Reflection composes repeated folds and singleton
broadcasts where native padding kernels reject wider pads. Dimensions,
products, vector/scalar ranks and normalized unique axes are checked before
native allocation; nonconstant modes cannot extend an empty cropped axis.
Wrap requires schema 19 or newer. String, complex and encoded element formats
remain outside this tensor materialization path, not proven impossible
backend operations.
ORT 1.31's shape inference rejects constant int32 axes before provider
selection; int64 initializers and int32 graph values avoid that frontend
limitation.

`InstanceNormalization`, `LpNormalization`, and `LRN` support their legacy
schemas, live numeric payloads, symbolic/unknown ranks, and empty dimensions.
Instance normalization requires rank at least three and exact channel vectors;
Lp normalization checks its full-width axis before narrowing and supports
orders one and two, including zero norms. LRN applies the schema's channel
window to general `[N,C,...]` tensors, including asymmetric even windows.
Full-width positive window sizes are clipped to contributing channel offsets
without changing the original `alpha/size` coefficient or allocating padding
proportional to the requested size. Windows covering every channel reduce once.
Narrow inputs accumulate and apply affine parameters in float32 before the
final cast; float64 computes on MLX CPU. Instance statistics compensate the
residual mean to avoid cancellation for small variance around a large offset.
Dynamic rank/channel/empty dispatch
uses shape-keyed capture, retaining shapeless eligibility for the existing
fully static positive Instance/Lp forms. Runtime channel-vector, rank, axis,
and native-dimension errors are explicit and do not poison later valid runs.
The explicit schema formulas are used for cases where ORT's LRN kernel only
accepts rank four or the Python reference mishandles channel iteration or
negative-input L1 norms.

`Split` supports legacy size attributes, modern int64 size tensors, equal
partitions, and `num_outputs` partitions with a smaller final part. Symbolic
geometry and changing boundary sizes use shape-keyed capture with bounded
value specialization; data-dependent interior sizes execute eagerly. Sizes
must be nonnegative, have one entry per output, and sum exactly to the axis
extent without integer overflow. Zero-length parts are preserved. Scalar
inputs, native-wide dimensions, and the legacy version-1 optional floating
size input remain unclaimed. `num_outputs` cannot accompany explicit sizes or
produce a negative final chunk. Empty partitions follow ONNX shape inference,
the reference, and equivalent explicit-size partitions, even where ORT 1.31's
`num_outputs` kernel rejects them.

INT64 `Range` values are generated with checked integer arithmetic. Live bounds
are reread in an eager partition before downstream compiled consumers. All three bounds
must be scalars, the output length must fit native int32 geometry, and its
exact count must agree with ORT's inferred extent for constant bounds. Full-width bounds with small
spans follow the ONNX function, including cases where ORT 1.31's native kernel
incorrectly returns empty. Forms whose exact count disagrees with the inferred
extent remain unclaimed. Live INT64 bounds support positive/negative steps,
empty results, full-width values and error recovery; invalid geometry raises
an EP error. Other Range dtypes continue to require constant scalar bounds.

`BatchNormalization` implements versioned inference and training, including
legacy version-7 nonspatial features and version-9 rank-one inputs (`C=1`).
Version 14 separates statistic types from data/affine types; version 15 also
allows independent affine types. Supported floating groups, including mixed
float64, compute in float32 or MLX CPU float64 and cast each output to its
declared type. Training uses population variance, live momentum updates, and
optional output slots without freezing inputs. Legacy version-7/9 training
selects five output slots and returns inverse standard deviation in the
historically named `saved_var` slot, matching ORT; modern training selects
three slots. Version-7 nonspatial training composes the equivalent flattened
feature operation. Empty training populations produce NaN channel statistics,
while empty channel vectors remain empty. Runtime shape/parameter validation
and capture classification follow the spatial normalization rules above.
Version-1/6 inference honors `is_test` instead of silently treating training
as inference; their training forms remain unclaimed pending the explicit
mandatory in-place running-statistic contract, not an unavailable MLX kernel.

Metal has no float64 kernels; supported float64 forms run through MLX CPU, while
other forms fall back to ORT CPU. Empty/dynamic support is per-handler, not a
blanket guarantee. Claimed control-flow nodes translate supported tensor bodies;
other control-flow forms remain on CPU.

Already-claimed nested `If` conditions and `Loop` trip counts/initial conditions
can read bound body formal inputs eagerly, including BOOL/INT64 slices supplied
by `Scan` and the iteration/condition formals supplied by an enclosing `Loop`.
Reads validate exact dtype and one-element payloads; INT64 controls never pass
through floating point. Direct graph-input/initializer controls retain their
host-value-specialized compiled path. Intermediate controls remain ineligible
for compiled specialization and use checked evaluated host reads only in eager
execution. This does not widen claims for node-produced `If` conditions or change
Loop/Scan arity, passthrough-condition, optional/dynamic termination, scan-output,
container, or float64 restrictions. The focused nested-control fixtures are
source coverage pending native validation, not a full control-flow audit claim.

The **core Rust modules** register:

| Module | Registered ops |
|---|---|
| `elementwise.rs` / `math.rs` | Arithmetic, comparison, logical/bitwise, casts/bitcasts, transcendental functions, and activations including Mish/PRelu/HardSwish |
| `reduction.rs` | Reduce family, ArgMin/ArgMax, CumSum/CumProd/LogCumSumExp, Hardmax, TopK |
| `shape.rs` | Gather/scatter, reshape/layout, crop/pad, depth/space transforms, Resize/Upsample, Compress, ReverseSequence, EyeLike |
| `norm.rs` | RMS/Layer/Group/Instance/Batch/Lp normalization, MeanVarianceNormalization, LRN, Microsoft skip-norm variants |
| `attention.rs` | GroupQueryAttention, Attention (opset 23 & 24), MultiHeadAttention, RotaryEmbedding |
| `matmul.rs` | MatMul, GroupedMatMul, Gemm, com.microsoft::FusedMatMul |
| `qdq.rs` | Private QDQ matrix-product recognition and direct packed int4/uint4, int8/uint8 and FP8 Metal kernels |
| `float8.rs` | FP8 formats, exact host/Metal decoders for DequantizeLinear transport and fusion |
| `packed/` | Pack/Unpack/DequantizePacked; private family decoders/codebooks, Metal decoding, constant views and consumer fusion |
| `conv.rs` | Conv/ConvTranspose/DeformConv and local/global pooling |
| `quant.rs` | MatMulNBits, GatherBlockQuantized, Quantize/Dequantize/DynamicQuantize, MatMulInteger, ConvInteger, QLinearMatMul, QLinearConv |
| `ssm.rs` | TensorScatter (opset 24), CausalConvWithState, LinearAttention (linear/gated/delta/gated_delta, GQA) |
| `paged_ssm.rs` | Packed VarlenCausalConvWithState and GatedDeltaNet; device-resident offsets, FP32 V-major recurrence, empty rows preserve initial state, compact prefix-transition capture up to capacity 8; with more than eight tokens and at least 32 value rows, prepare FP32 Q/K normalization and decay/beta coefficients once rather than repeating them for every value row |
| `selective_scan.rs` | Mamba-1 selective scan: recognises the `Scan` body and replaces the per-timestep unroll with one custom Metal kernel (state kept in registers) |
| `nonzero.rs` | Exact eager NonZero host scan with live cardinality and int64 coordinates |
| `unique.rs` / `host_scan.rs` | Exact eager Unique grouping/axis slices; shared checked live host payload access |
| `misc.rs` / `image.rs` | Constants, losses, Det, sequence-to-tensor consumers, numeric TF-IDF, and image decoding |
| `random.rs` / `signal.rs` / `recurrent.rs` / `vision.rs` / `controlflow.rs` | Random, signal/FFT/window, RNN/GRU/LSTM, vision/detection, and recursive control-flow translations |

Reductions accept scalar inputs. `ReduceMax` and `ReduceMin` also claim Boolean
tensors from opset 20 onward, using logical OR and AND respectively. Empty
extrema reductions use exact dtype identities: Boolean `False`/`True`, integer
minimum/maximum, or floating-point negative/positive infinity. Live axes are
read on each invocation, not frozen into a compiled closure; initializer axes
remain compilable, and proven shape-derived axes use shape-keyed compilation.
Other reduction operators retain their numeric/float-only dtype restrictions.

`ConstantOfShape` preserves exact fill values, scalar output for an empty shape
vector, and zero dimensions. When compilation is supported, live boundary shapes
use bounded value specialization; float64 and data-dependent intermediate shapes
execute eagerly, and proven
shape-derived shapes use shape-keyed compilation. Invalid dimensions are rejected
before narrowing to MLX extents. `Expand` reports incompatible broadcasts instead
of returning an invented zero tensor.

Each registration supplies a claim predicate and compilation policy. A declined
form falls back to ORT CPU. Regression tests cover selected schema partitions and
interaction boundaries; they are not proof of arbitrary Cartesian combinations.
Examples of remaining form exclusions include some `Resize` interpolation options,
ConvTranspose geometries, and attention mask/layout combinations. Such exclusions
are implementation restrictions, not necessarily mathematical/backend impossibilities.

**Engine fix (landed):** ORT returns a **null** `OrtValueInfo` for an omitted *interior* optional input (e.g. `Resize` `roi`, `Clip` `min`); the Rust clustering pass now null-guards input names before constructing dataflow edges, so interior-optional-gap ops are safe. Earlier interior-gap claim guards in attention/SSM modules can now be relaxed as a follow-up.

---

## 2.2 The modular op registry

The translator is a **registry**, not an if-chain. Both the claim-time membership check and the run-time dispatch consult the SAME table, so a claimed op is always translatable and vice-versa.

### Files

| File | Role |
|---|---|
| `rust/src/registry.rs` | The `OpRegistry` singleton: the `(domain, op_type, [min_opset, max_opset]) → {handler, claim}` table, lookup, `translate`, and `claimable`. Also contains `NodeView` and claim helpers such as `is_mlx_float` and `suffix_broadcast`. |
| `rust/src/engine.rs` | `mlx_dtype_from_onnx()` (the dtype mapping), `Plan` (persistent MLX state), and `TranslationContext` (the object handlers use to resolve inputs, bind outputs, keep arrays alive, run boundary eval, and copy out). |
| `rust/src/mlx.rs` | Safe RAII layer over `mlx-c`: `Stream`, `Array`, and `VectorArray` wrappers with `Drop`; raw bindgen stays in `rust/src/sys.rs`. |
| `ep.rs` under `rust/src/` | ORT `OrtEp` vtable boundary: `GetCapability`, convex clustering, `Compile`, `OrtNodeComputeInfo` ownership, and runtime compute entry points. No op-specific translation logic. |
| `rust/src/ops/elementwise.rs` | `Add`, `Mul`, `Sub`, `Sigmoid`, `Softmax`, `Cast` handlers **+ claim predicates** + registration. |
| `rust/src/ops/norm.rs` | Normalization handlers and claim predicates. |
| `rust/src/ops/attention.rs` | Attention, GQA, MHA, and RoPE handlers and claim predicates. |
| `rust/src/ops/quant.rs` | Quantization handlers including `MatMulNBits` and `GatherBlockQuantized`; the former split quantization families are consolidated here. |
| `rust/src/ops/{math,matmul,misc,random,recurrent,reduction,shape,signal,ssm,vision,conv,controlflow}.rs` | Remaining op families. |

Each module registers into the registry; `rust/src/registry.rs` `register_builtin_ops` wires the modules into a process-wide `OnceLock` singleton. Registration is explicit — no reliance on static-init ordering.

### The registry key: `(domain, op_type, opset range)` + claim predicate

A handler has the Rust type `fn(&mut TranslationContext, &NodeDesc) -> Result<(), MlxError>`. It is registered under a domain (`""` = `ai.onnx`, or `com.microsoft`), an op type, an inclusive opset range `[min_opset, max_opset]`, the handler, **and a claim predicate** `fn(&NodeView) -> bool`. `K_ANY_OPSET` (`-1`) means "unbounded on that side"; a version-insensitive or contrib op registers with `{K_ANY_OPSET, K_ANY_OPSET}`.

```rust
registry.register(OpRegistration {
    domain,
    op_type: "MyOp",
    min_opset,
    max_opset,
    handler: my_op,
    claim: my_op_claim,
});
```

The claim predicate answers, for a concrete ONNX node whose `(domain, op, opset)` already matched this entry: **can the MLX backend translate THIS node exactly** — right dtypes, shapes, attributes, input/output form? It lives in the same `rust/src/ops/<family>.rs` module as its handler, using shared helpers in `rust/src/registry.rs` (`SlotInfo`, `is_mlx_float`, `suffix_broadcast`, and related helpers).

`ep.rs` under `rust/src/` contains no per-op claim logic. `GetCapability` calls a single hook — `claimable(&NodeView)` — which looks up the matching registry entry and runs its claim predicate.

```rust
pub fn claimable(node: &NodeView) -> bool {
    match registry().find_entry(&node.domain(), &node.op_type(), node.since_version()) {
        Some(entry) => (entry.claim)(node),
        None => false,
    }
}
```

Because the same lookup backs both claim and run-time dispatch, "claimed" and "translatable" can never disagree.

### The registry key: opset dispatch

The opset is threaded end-to-end: `ep.rs` under `rust/src/` reads the node's since-version into `NodeDesc::since_version`, and `OpRegistry::find_entry` dispatches by matching the range. This is the seam that lets opset-23 and opset-24 variants of an op (e.g. `Attention`, `TensorScatter`) map to different handlers — a version-split op registers two handlers with adjacent, non-overlapping ranges (e.g. `[1, 22]` and `[23, K_ANY_OPSET]`). `RMSNormalization` uses a bounded range as a live example.

### Generic node attributes (`NodeDesc`)

`ep.rs` under `rust/src/` copies ONNX attributes on each node into `NodeDesc` generically. Attributes are split by ONNX attribute type into typed maps:

| ONNX attr type | `NodeDesc` map | Handler reads |
|---|---|---|
| `ORT_OP_ATTR_INT` | `ints` (`HashMap<String, i64>`) | `n.ints.get("axis")` |
| `ORT_OP_ATTR_FLOAT` | `floats` (`HashMap<String, f32>`) | `n.floats.get("epsilon")` |
| `ORT_OP_ATTR_INTS` | `int_arrays` (`HashMap<String, Vec<i64>>`) | `n.int_arrays.get("axes")` — Slice/Reduce/Transpose/Conv/Split |
| `ORT_OP_ATTR_FLOATS` | `float_arrays` (`HashMap<String, Vec<f32>>`) | `n.float_arrays.get("scales")` |
| `ORT_OP_ATTR_STRING` | `strings` (`HashMap<String, String>`) | `n.strings.get("mode")` |
| `ORT_OP_ATTR_TENSOR` | `tensors` (`HashMap<String, ConstTensor>`) | Constant and ConstantOfShape tensor payloads |
| `ORT_OP_ATTR_GRAPH` | `subgraphs` (`Vec<SubgraphDesc>`) | If/Scan/Loop body translation |

Absent attributes are simply not present in the map, so a handler reads an optional attr with `get(...).copied().unwrap_or(default)` and a required attr with `get(...).ok_or_else(...)`. This schema is what unblocks the long tail (Slice, Reduce*, Transpose, Conv, Attention, Split, LayerNorm) whose attributes are arrays and strings.

### The dtype mapping

`mlx_dtype_from_onnx()` maps every ONNX tensor element type `mlx-c` can carry to its `mlx_dtype`: `fp32`, `fp16`, **`bf16`**, `fp64`, the signed/unsigned integer widths (`int8/16/32/64`, `uint8/16/32/64`), and `bool`. It is used in input resolution, constant materialization, boundary casts, and copy-out, so **every tensor honors its actual dtype** rather than a hard-coded fp32.

Because MLX carries the resolved dtype through its ops with no per-dtype code, the dtype-generic handlers (elementwise, activation, softmax, normalization, cast) work in fp32, fp16 **and** bf16 with a single implementation. `GroupQueryAttention` also accepts matching fp32/fp16/bf16 tensors end to end. `MatMulNBits` and `GatherBlockQuantized` remain fp32-only where their quantized representations match the cpu-recipe graph.

---

## 3. Translation details by family

### 3.1 Quantized matmul

`MatMulNBits` is claimed only for the int4 block-32 form. The ONNX packed uint8 weight tensor is repacked once into the affine-quant layout MLX expects, then stored on the compiled plan for reuse across prefill and decode.

Runtime target: `mlx_quantized_matmul`.

### 3.2 Attention and KV cache

`GroupQueryAttention` is the fused attention op used by the target decoder graphs. The claimed form is the 9-input separate-QKV `com.microsoft` op with matching `fp32`/`fp16`/`bf16` Q/K/V, past K/V, cos/sin, and `int32` sequence-length inputs.

The translator maps it to:

- `mlx_fast_rope` for the in-op RoPE transform.
- `mlx_fast_scaled_dot_product_attention` for attention.

The backend writes present K/V to the same ORT context outputs in `[B, kv_heads, total_seq, head]` native-float layout. This preserves the runtime-owned KV-cache handoff across the prefill→decode boundary.

`PagedAttention` reads CPU boundary/initializer integer metadata directly and
evaluates graph-produced metadata before reading it. Its eager cache writeback
copies only the physical slots updated by the current call, coalescing adjacent
slots and preserving skipped slots. Declared cache outputs still return the full
cache. When an output actually aliases its cache input, boundary copy-out uses
the enclosing changed row interval in each physical block; non-aliased outputs
retain a full copy. Cache ownership and all observable input/output updates stay
with ORT.

Eligible GPU paged attention reads physical KV pages directly for one to eight
query rows, without dense gathers or repeating KV heads. FP16 Q/K/V stay FP16
when their matching head width is 64, 96, 128, or 256, with no sinks, softcap, or
local window. Scores, softmax, weighted values, and two-pass partial outputs
accumulate in FP32; only the final output is cast to FP16. Architecture-dependent
one/two-pass dispatch, partition counts, and reduction order match the FP32
dense fallback with its already-expanded KV heads. This avoids the intermediate
FP16 rounding in MLX's ordinary two-pass FP16 attention.

Single-batch unique cache writes can read original pages plus the current
query rows, including writes that overwrite cached-prefix slots. Duplicate
writes and multiple sequences retain functional updated-cache reads. Other
geometries and an explicit `MLX_SDPA_BLOCKS` override retain dense attention
lowering; all existing masks and cache writeback semantics are preserved.

Standard causal `Attention` with one query and one current key can attend every
past/current key. It omits the redundant causal array rather than capturing a
first-call KV extent during shapeless decode. An external mask still applies,
including its fully masked row semantics. Shapeless causal replay requires
statically proven single-query/single-current-key geometry; other causal shapes
retain shape-keyed eligibility and eager execution when no shape-keyed attention
route is available. Non-square alignment still uses the actual past offset, not
MLX's lower-right causal alignment. This does not change GQA or sliding windows.

### 3.3 Normalization

Standard `RMSNormalization` and `LayerNormalization` normalize any valid suffix
of dimensions and accept unidirectional broadcast affine parameters. The ordinary
last-axis, vector-affine, Y-only forms retain MLX fast kernels. Other forms compose
the schema's arithmetic, honor `stash_type`, and cast normalized values back before
the affine stage. Narrow statistics accumulate in float32 and round back to stash
precision; float64 statistics stay on an isolated MLX CPU stream even when all
boundary tensors are float32. LayerNormalization also produces optional mean and
inverse-standard-deviation outputs, including omitted output slots.

ORT's default-domain `SimplifiedLayerNormalization` extension shares RMS
normalization's suffix axes, broadcast/mixed affine types, stash precision, and
empty/dynamic support, and can produce its optional inverse-standard-deviation
output. The existing `com.microsoft` compatibility alias is retained for runtimes
that expose it. This extension is not an ONNX standard operator.

`GroupNormalization` supports opset 18's per-group affine parameters and opset
21+'s per-channel affine parameters and stash precision. Batch, channel, and
spatial extents may change between calls; channels must be divisible by the
positive group count, and both affine vectors are validated. Empty extents are
supported. Floating-point payloads include float64 on the MLX CPU stream;
float64 stash statistics also select that stream independently of payload dtype.
Concrete group reshapes use shape-keyed compilation.

`SkipLayerNormalization` and `SkipSimplifiedLayerNormalization` support optional
bias, broadcast skip inputs, and the residual sum at fixed output slot 3.
Both use their schema's default epsilon of `1e-12`. Narrow inputs accumulate the
residual and normalization in float32 before casting output tensors back.
Empty extents are supported; dynamic forms use shape-keyed compilation.
`SkipLayerNormalization` supplies float32 mean and inverse-standard-deviation
outputs at slots 1 and 2, including independent output holes. The RMS variant
supplies its inverse-root-mean-square statistic at slot 2. Saved statistics have
shape `input.shape[:-1] + [1]` and describe the float32 residual before its
optional cast back to the input dtype. Statistics use shape-keyed composition;
ordinary inference retains the fused normalization paths.

Only the RMS variant's live mean output at slot 1 remains declined when consumed
or exported. Its upstream schema does not define which RMS statistic it denotes,
and the [portable reference](https://github.com/onnxruntime/mobius/blob/main/src/mobius/functions/skip_layer_normalization.py)
emits unused placeholders, not an observable value contract. Declared-but-unused
RMS mean outputs remain accepted. This is a semantic specification gap, not a
backend limitation.

### 3.4 Quantized embedding gather

`GatherBlockQuantized` is claimed for both the symmetric int4 embedding form (three inputs, `zp=8`, `w=(q-8)·scale`) and the asymmetric four-input form with explicit `zero_points` (`w=(q-zp)·scale`). The backend performs gather plus int4 dequant, uses the schema's default block size of 128, and returns zero slices for out-of-range indices as required by ORT 1.31. Bounds are checked before narrowing int64 indices, and the result retains the scale tensor's dtype.

`GatedRMSNorm` supports the ORT 1.31 `activation` attribute: `silu` (default), its `swish` alias, and `sigmoid`. Unknown activations are not claimed. Dynamic leading dimensions are supported with a static feature width; compilation is shape-keyed because grouped reshapes capture concrete runtime geometry. `MRotaryEmbedding` likewise accepts dynamic batch/sequence dimensions with static head and rotary widths.

### 3.5 Softmax and elementwise

The translator supports the exact elementwise and cast forms listed in §2:

- `Softmax`: last-axis, `fp32`/`fp16`/`bf16` → `mlx_softmax`.
- `Add`: `fp32`/`fp16`/`bf16`.
- `Mul`: `fp32`/`fp16`/`bf16`.
- `Sub`: `fp32`/`fp16`/`bf16` and `int64`.
- `Sigmoid`: `fp32`/`fp16`/`bf16`.
- `Cast` / `CastLike`: the verified conversion set in the support table, including bool→`fp32`/`fp16`.

Unsupported dtype combinations fall back to CPU.

`TopK` handles legacy `k` attributes and current initializer, runtime, or
shape-derived K inputs. Native floating-point and signed/unsigned integer
payloads retain exact values and lower-index tie-breaking. Float64 runs on the
MLX CPU stream. Both `largest` settings and `sorted=0/1` are supported;
`sorted=0` may return sorted results because its output order is unspecified.
K must remain positive and within the selected axis, including after replay;
empty outer dimensions are supported. Live boundary K values use bounded
value specialization, while live same-partition data-derived K runs eagerly.
Dynamic/empty geometry and live K use shape-keyed compilation; static,
immutable-K forms retain shapeless compilation. Float8 and sub-byte payloads
remain unsupported.

---

## 4. Caching and lifetime model

The compiled plan owns persistent conversions that are expensive or unnecessary to repeat:

- Repacked `MatMulNBits` weights.
- Cos/sin caches.
- Gammas.
- Embedding table data.
- Biases.

Some values are available only through live context data, so they are cached on the first `Run`. Subsequent prefill/decode invocations reuse the plan cache.

This replaces the old per-kernel pipeline-state model. There are no Metal compute pipeline states, shader names, or MSL dtype suffixes in the current architecture.

Rust ownership is a deliberate design win: ORT factory/EP objects cross the C ABI with `Box::into_raw` / `Box::from_raw`, while MLX streams, arrays, and vector arrays are owned by wrappers in `rust/src/mlx.rs` whose `Drop` implementations call the matching `mlx_*_free`. Runtime-produced arrays live in `TranslationContext`'s arena for the duration of a compute call; persistent constants live in the plan cache and are freed with the plan.

---

## 5. Claiming rules

A claim predicate should answer one question: **can the current ONNX node be represented by the MLX translator exactly enough to run inside the fused subgraph?**

That means every claim must validate:

1. Domain and op type.
2. Input count and output count where the op has multiple forms.
3. Dtypes.
4. Required attributes such as `bits`, `block_size`, `axis`, or last-axis softmax.
5. Layout assumptions, especially KV-cache shape for GQA.
6. Whether constants/initializers can be cached in the current plan.

The claim predicate is registered **next to its translate handler** in the op module (`rust/src/ops/<family>.rs`) as `OpRegistration::claim`, and is additionally AND-gated by registry membership: `claimable(&NodeView)` (called from `ep.rs` under `rust/src/` `GetCapability`) looks up the matching `(domain, op, opset)` entry and runs its claim predicate. A node with no matching entry — or whose entry's predicate rejects it — is never claimed, so "claimed" can never outrun "translatable". **`ep.rs` under `rust/src/` contains no per-op claim logic.**

When in doubt, do not claim. CPU fallback is preferred to an approximate translation.

---

## 6. Adding or changing a translated op

The extension path is **purely additive** and **registry-centric**: adding an op is one self-contained handler module plus one registration line — **no `ep.rs` under `rust/src/` edits, ever.** To add a long-tail op:

1. **Handler + claim predicate — in one module.** In the appropriate `rust/src/ops/<family>.rs` (or a new module), add:
   - `fn my_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError>` — resolve inputs with the `TranslationContext` helpers, read attributes generically (`n.ints`, `n.int_arrays`, `n.strings`, …), emit MLX ops through safe `mlx.rs` wrappers or raw `mlx-c` calls kept by the context, and bind results to `n.outputs[i]`. Read the tensor's actual dtype through `mlx_dtype_from_onnx` — never hard-code fp32.
   - `fn my_op_claim(node: &NodeView) -> bool` — the dtype/shape/attribute claim checks, using shared helpers in `rust/src/registry.rs`.
2. **Register — one line.** Add a `registry.register(OpRegistration { ... })` call in that module's `register` function. Use `K_ANY_OPSET` for a version-insensitive op, or a bounded range for an opset-split op.
3. **New module wiring only if needed.** If you created a new `rust/src/ops/<family>.rs`, add it to `rust/src/ops/mod.rs` and call its `register` function from `register_builtin_ops` in `rust/src/registry.rs`.
4. **Attributes — usually nothing to do.** `ep.rs` under `rust/src/` already copies node attributes into `NodeDesc` generically. Your handler and claim predicate just read them. Only if a future op needs a new attribute payload shape is a localized `NodeDesc` extraction extension required.
5. **Tests.** Add op-correctness coverage in `tests/ops/` (ONNX IR API via `_models.py`). For a dtype-generic op, add fp16 (vs ORT CPU) and bf16 (bf16-interior subgraph vs a numpy reference) cases.
6. **Conformance / E2E.** Add opt-in conformance coverage when broad ONNX behavior matters, and add/update E2E coverage if the op affects decoder coherence or KV-cache behavior.
7. **Docs.** Update the §2 table and, if relevant, §2.2.

**Summary of touch points for a new op in an existing family:** one handler function + one claim predicate + one `registry.register(...)` line in `rust/src/ops/<family>.rs`, plus tests. **Zero changes to `ep.rs` under `rust/src/`, `rust/src/engine.rs`, or the registry core.**

**Add a new opset variant:** register a second handler for the new range and narrow the existing registration (e.g. change `[23, K_ANY_OPSET]` to `[23, 23]` and add `[24, K_ANY_OPSET]`).

**Add a new dtype:** if `mlx-c` exposes it, add the `ONNX → mlx_dtype` case to `mlx_dtype_from_onnx`, ensure copy-out handles the byte layout, and widen the relevant claim predicate. Dtype-generic handlers need no change.

Do not add a new `.metal` kernel or a dtype-traits/MSL specialization layer for new coverage.

---

## 7. Testing expectations

Use the Rust plugin artifact with the Python tests:

```sh
DYLD_LIBRARY_PATH=<ort-prebuilt/lib> \
ONNXRUNTIME_MLX_EP_LIB=$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib \
python -m pytest tests/ops -q
```

| Layer | Test | Purpose |
|---|---|---|
| Op correctness | `tests/ops/` / `python -m pytest tests/ops -q` | Confirms each claimed translation matches a reference within accepted tolerances. fp32/fp16 compare against ORT CPU; bf16 keeps the compute inside an MLX-claimed subgraph (fp32 boundaries) and compares against a numpy fp32 reference (~1e-2), since ORT CPU has no bf16 kernels. |
| Conformance | `tests/conformance/` | Opt-in broader ONNX coverage. |
| Memory stability | macOS `leaks` around Rust stress scripts | Confirms Rust RAII teardown stays leak-free across repeated sessions. |

Current post-pivot baseline:

- Rust release build green.
- Python op tests green through `MLXExecutionProvider`.
- Qwen2.5-0.5B emits `The capital of France is Paris`.
- CPU token stream match for the first 14 tokens; known fp32 decode drift after that is accepted.
- Prefill/TTFT improves from ~33 ms CPU to ~15 ms `MLXExecutionProvider` for a 26-token prompt.
- Prefill/TTFT improves from ~575 ms CPU to ~165 ms `MLXExecutionProvider` for a 512-token prompt.
- Warm decode is ~122–148 tok/s at short context.
- Leak stress shows flat allocator memory and 0 leaked MLX handles across bounded cycles.

---

## 8. Build and dependency implications

`mlx-c` is a hard dependency. Build the pinned MLX/mlx-c pair:

```sh
rust/scripts/setup_mlx.sh "$PWD/.deps/mlx-0.32.4"
export MLX_PREFIX="$PWD/.deps/mlx-0.32.4"
export MLXC_PREFIX="$MLX_PREFIX"
```

Build with Cargo from the Rust crate:

```sh
cd rust
ORT_INCLUDE_DIR=<ORT include dir> cargo build --release
```

`rust/build.rs` also honors `$ORT_HOME/include` as a fallback. It runs bindgen over the ORT plugin-EP C ABI and `mlx-c` headers, and it does **not** link `libonnxruntime`.

Use only the current crate/artifact names:

- `rust/`
- `rust/target/release/libonnxruntime_mlx_ep.dylib`

The registered EP name is `MLXExecutionProvider`; do not use the dylib name as evidence that the runtime-facing EP name differs.

---

## 9. Removed / historical architecture

This document replaces the older modular-op and dtype plan. The following are historical and must not be described as active:

- Hand-written Metal shaders for matmulnbits, GQA, norm, softmax, RoPE, elementwise, data movement, and quantized gather.
- Metal shader compile/registry/encode machinery and the former allocator bridge.
- Generated Metal-kernel include fragments.
- The old hand-kernel op-registry scaffold.
- The dtype/MSL specialization plan.
- The old MPS target and dylib artifact.
- Transitional MLX/Metal feature flags and any hand-kernel fallback path.

For the data that justified the pivot, see [`docs/MLX_EVALUATION.md`](./MLX_EVALUATION.md).

---

## 10. Open follow-ups

| Follow-up | Current behavior |
|---|---|
| `GatherBlockQuantized` with explicit `zero_points` | Not claimed; CPU fallback. |
| Additional decoder ops | Not claimed unless/until an MLX translation and tests are added. |
| Further fp32 drift analysis | Current drift is accepted after the first 14 CPU-matching tokens; keep monitoring with E2E tests. |
| Broader model-family coverage | Out of scope for the current claim table; CPU fallback remains the default. |


## Symbolic ScatterElements geometry

`ScatterElements` reduction `none` accepts compatible symbolic shape metadata.
Known dimensions must agree; runtime shapes validate indices/update equality,
rank, axis and non-axis bounds. Partial non-axis updates scatter into the selected
prefix and merge it back into the original data, preserving untouched elements.
This avoids MLX broadcasting a one-row update across multiple data rows. Empty
updates return the original data. The existing shape-keyed compile route checks
new geometry before reuse.

Native regressions cover FP32/FP16 partial and empty updates, 4-D symbolic
metadata, invalid geometry and recovery. Dynamic INT64 Range also covers shape-derived
causal masks, full-width integer values and live axes consumed by reductions.
