# Compiled Capture (`mlx_compile`) — what can be dynamic, and the model invariants

**Status:** Final Rust post-pivot architecture
**Date:** 2026-07-18
**Repo:** `onnxruntime-mlx`
**Companion:** [`OP_ARCHITECTURE.md`](./OP_ARCHITECTURE.md), [`DESIGN.md`](./DESIGN.md)
**Code:** `rust/src/compiled.rs`, `rust/src/engine.rs`, `rust/src/ops/attention.rs`

---

## 0. Summary

The eager translator (`engine::TranslationContext::execute`) rebuilds and dispatches **every** node of a
claimed subgraph as a separate, unfused MLX primitive launch on **every** `Compute` call — ~393 kernel
launches per token for a decoder (`compiled.rs:3-6`). The compiled fast path traces the whole subgraph
**once** into an `mlx_closure` over its dynamic inputs, compiles it with `mlx_compile` (kernel fusion),
caches the closure on the `Plan`, and on each later call just applies the compiled closure to the
freshly-wrapped live buffers.

This doc explains **how that capture works**, **what is allowed to be dynamic** while still being
captured, and **the invariants a model must satisfy** for the (shapeless) decode capture to be both
correct and retrace-free.

---

## 1. The mechanism

1. **Trace once.** Build an `mlx_closure` over just the **dynamic (non-constant) ctx inputs**. Constants
   (weights, cos/sin caches) are baked in; dynamic inputs become **placeholders** (`build_closure`,
   `trace_body` in `compiled.rs`).
2. **Compile.** `mlx_compile` fuses the traced graph into one optimized kernel program
   (`Closure::compile`, `compiled.rs`).
3. **Cache.** The compiled closure is stored on the `Plan` (per `Slot`).
4. **Apply per call.** Zero-copy wrap the live ORT input buffers (`Array::from_data_managed`) and run the
   fused closure; copy the boundary outputs back (`try_compiled`, `compiled.rs`).

**The central constraint:** *inside a trace, placeholders carry no data.* Anything that must **read a
dynamic value mid-trace** — to decide a shape, an axis, or a slice bound — is illegal, because
`mlx_compile` cannot shape-infer or eval it. Every allowance and every invariant below follows from this
one fact.

**Safety.** Every path falls back to the eager translator on any doubt (ineligible plan, missing cache,
trace/apply/eval error). The compiled path never crashes and never diverges (`compiled.rs:27-28`).

---

## 2. Two capture modes (`ShapeMode`) — the crux of "what can be dynamic"

`ShapeMode` (`engine.rs:241-248`) is the whole answer:

| Mode | Behavior on a changed input shape | Used by |
|---|---|---|
| **`Shapeless`** | The compiled closure is shape-*agnostic*: a dimension may **grow/change every call with zero retrace**. One closure serves every step. | **decode** (growing KV length costs one compile ever, not one per token) |
| **`ShapeKeyed`** | `mlx_compile` keys on input shapes/dtypes and **transparently retraces** (re-invokes the thunk) when they change, so a new shape recompiles rather than miscomputes. | **general** static-shape subgraphs; **prefill** (varying query length `S`) |

MLX 0.32's shapeless `mlx_compile` cannot shape-infer several primitives and cannot eval mid-trace.
Every registered lowering therefore makes an explicit choice at its registration boundary:

- `register_shapeless` is an affirmative promise that its **shapeless-specific** trace emits only
  primitives implementing `Primitive::output_shapes`;
- `register_shape_keyed` carries the exact missing-primitive reason;
- `register_shape_classified` derives `Shapeless`, `ShapeKeyedOnly`, or the stricter `EagerOnly`
  from concrete node properties (used by quant handlers and per-form GroupQueryAttention).

There is no unclassified registration API. `NodeDesc` stores the resulting decision during Compile,
so decoder partition colouring and every compiled execution route consume the same proof. Eager MLX
execution is unaffected.

The MLX 0.32.2 audit found these missing-`output_shapes` primitives in current lowerings:

| MLX primitive | Registered lowerings that can emit it |
|---|---|
| `Slice` | ONNX `Slice`; QMoE; CausalConvWithState; LinearAttention; Scan; DeformConv; LRN; RNN/GRU/LSTM; DFT; `ai.onnx.ml` SVMClassifier/FeatureVectorizer; `com.microsoft` Attention/PackedMultiHeadAttention/RotaryEmbedding/MRotaryEmbedding/PagedAttention; sliding-window and attention-bias GroupQueryAttention; GridSample/RoiAlign/MaxRoiPool; conditional block-quantized matmul, MatMulNBits zero-point trim, and GatherBlockQuantized zero-point trim forms |
| `Scan` | CumSum; CumProd; internal LinearAttention cumulative gating |
| `AsStrided` | AveragePool/MaxPool/LpPool; STFT |
| `Pad` | AveragePool/MaxPool/LpPool; LRN; CenterCropPad; ONNX Pad; LinearAttention; conditional unaligned MatMulNBits |
| `RandomBits` | RandomNormal/RandomNormalLike/RandomUniform/RandomUniformLike/Bernoulli/Multinomial |
| `Scatter` | Scatter/ScatterElements/ScatterND; PagedAttention; Col2Im; MaxUnpool |
| `Split` | ONNX Split |
| `View` | BitCast |
| `FFT` | DFT/STFT |
| `CustomKernel` | the fused selective-scan branch inside ONNX Scan |

`GroupQueryAttention` is classified per node. Claimable non-sliding forms without attention bias use
the slice-free shapeless decode configuration (data-fed RoPE rows plus shape-preserving cache
updates). Sliding-window and attention-bias forms emit `Slice` and remain `EagerOnly`; unsupported
forms are also denied compilation conservatively.

In decoder graphs, shapeless, shape-keyed-only, and eager-only nodes receive distinct partition
colours. Stricter nodes remain claimed by MLX without disabling a more capable neighbouring
partition. ShapeKeyed pays one recompile per distinct `(shape, dtype)` tuple — cheap when the shape
set is small and repeating (prefill prompt lengths, CNN batch sizes), pathological if unbounded.

### Value-guarded boundary shape parameters

The general route can also specialize direct ORT boundary parameters of `Reshape`, `Expand`,
`Slice`, `DFT` (length/axis), `STFT` (step/frame length), and the three cosine-sum windows (size).
Before **every** apply it reads their integer values from the current kernel context,
including dtype and tensor shape in an explicit host key. An unused synthetic input with a unique
shape selects the corresponding MLX trace, using the same mechanism as static control-flow
specialization. Changing parameter values with otherwise identical input shapes therefore retraces;
returning to an earlier value reuses its trace. Ordinary activation/cache inputs remain live data.

This does not infer or restore `shape_const` provenance. It is safe even when the boundary values
really are data-dependent. Each parameter must contain 1–64 int32/int64 elements, and the general
plan retains at most 16 host-value specializations (shared with control-flow decisions). New values
after that limit use eager translation; previously cached values remain reusable. Empty/unsupported
parameters also use eager translation. Runtime-valued **intermediates** still require proven
shape-constant provenance; the compiler never evaluates a data-dependent tracer.

Claim rules are unchanged: runtime `Range`/`Tile` remain on CPU, and dynamic Slice axes/steps are
not newly claimed. The shapeless decode and specialized attention configurations are unchanged.

Signal controls must be rank-zero host-readable int32/int64 tensors (DFT axis is int64).
Lengths and steps are checked in the positive int32 range before narrowing. FFT signal shapes and
STFT window shapes may vary between calls; windows must match an explicit frame length. If both
STFT optional inputs are absent, the frame spans the signal, matching the ONNX reference.
DFT supports full forward/inverse transforms, real forward onesided output, and complex inverse
onesided output (IRFFT, with trailing component dimension 1). The omitted IRFFT length is
`2 * (input_axis_length - 1)`. Complex STFT requires `onesided=0`. FFTs use float32/complex64
internally and explicitly cast back to fp32/fp16/bf16; double FFTs remain unclaimed rather than
silently losing precision.

Window and MelWeightMatrix outputs support all ONNX numeric dtypes, with an explicit cast before
any fused consumer. Double windows use host double-precision cosine because MLX's CPU cosine
is only float32-accurate; double Mel matrices retain host double-precision weights.
Integer generator outputs truncate host double values before any float32 rounding.
Mel scalar frequency bounds support fp16/bf16/fp32/fp64. Runtime Mel parameters use eager
translation because their float-valued controls do not participate in the bounded integer cache;
constant Mel parameters can be compiled. Negative/zero sizes, invalid frequency ranges, and
unrepresentable allocations are rejected. Symmetric singleton floating windows preserve the
reference's NaN; integer outputs for that undefined conversion produce an explicit error.

### Two distinct senses of "dynamic"

- **Symbolic-dynamic at claim time** (a `-1`/symbolic dim in the graph): always fine **as long as the
  hidden/feature dim is static**. E.g. the `Attention` claim requires `qshape[2]` (hidden) `> 0` but lets
  `qshape[0]` (batch) and seq be dynamic — literally *"batch/seq may be dynamic"*
  (`ops/attention.rs:1207-1211`).
- **Runtime-varying across calls:** free in `Shapeless`; a transparent recompile in `ShapeKeyed`.

### The three configurations

`CompiledConfig` (`engine.rs:251-315`) expresses all three fast paths as one parameterised core:

- **decode** = `{ Shapeless, kv_alias, rope_as_data, delta_copyout }` — shapeless so a growing KV length
  never retraces; RoPE position + valid-past fed as **data**; delta KV copy-out.
- **general** = `{ ShapeKeyed, contiguous_outputs }` — retraces on a shape change; no attention/KV
  specialisation.
- **prefill** = `{ ShapeKeyed, kv_alias, rope_as_data, delta_copyout }` — the **same** decoder subgraph
  as decode but at query length `S > 1`, so it is shape-keyed (see §4).

---

## 3. Invariants for the shapeless **decode** capture

These are the gates in `general_enabled` / `build_closure` and the reasons behind each. A subgraph that
violates one falls back to eager (or, for attention, to the growing-concat route).

1. **No control flow** (`If` / `Loop` / `Scan`). The graph *structure* would depend on runtime data
   (`compile_enabled`, `compiled.rs:46-51`).

2. **No data-dependent translation parameters.** `Reshape` / `Expand` target (input 1), `Slice`
   starts/ends/axes/steps (1–4), `Range` bounds (0–2), and the `CumSum`/`CumProd` axis (1) must be **constant or
   shape-const** (derived from `Shape`/`Size`), never plain runtime data
   (`reads_data_dependent_parameter`). The general route additionally permits the explicitly
   value-guarded boundary parameters described above. Shape-const is OK because it changes with the shape-keyed cache
   key; an ordinary runtime tensor value does not participate in that key and would otherwise be
   silently baked into later calls. (This is exactly why runtime-bounded `Range` is deferred — see
   `OP_ARCHITECTURE`/README Range note.)

3. **No host-computed / data-dependent-output-shape ops** — `Det`, `NonZero`, `Unique` GPU-eval their
   input data mid-translate and/or emit a shape known only at runtime (`is_general_compile_unsafe`,
   `compiled.rs:61-73`).

4. **Attention uses the shared-buffer (fixed-capacity) KV contract, not growing concat.** With a
   runtime-owned max KV length, `past_k`'s seq axis is a **fixed capacity** every step, so **all shapes
   are static across tokens** (`detect_shared_kv`, `compiled.rs:167-213`). The only thing that varies —
   the valid-past offset — is fed as **data**: the new K/V are written with a `slice_update_dynamic` at
   that offset and attention runs over the whole capacity under a **static-shape additive mask** (buffer
   tail beyond `valid_past + S` masked to `-inf`; causal within the valid prefix)
   (`ops/attention.rs:420-425, 582-590`). Every op stays statically shaped, so the shapeless closure can
   carry it. A *growing* concat KV would change shape each token and force retrace/eager.

5. **`valid_past` must be recoverable as data** — from `total_sequence_length` (GQA input 6) or, when
   that scalar is computed in-subgraph, from the `attention_mask` width (`detect_shared_kv:193-209`,
   `try_compiled:355-360`). That is what lets the per-step position be **data** rather than a baked
   constant.

6. **RoPE must be full-rotary (`rotary_dim == head_dim`).** Then rotate-half is expressed as a `[hd,hd]`
   **matmul** — carrying no `Slice`, which shapeless compile cannot shape-infer — and the cos/sin rows
   are **pre-sliced outside the graph and fed as synthetic closure inputs**, so the per-step position
   never bakes in (`compiled.rs:21-25`, the `rope_as_data` feature). Partial rotary → the build declines
   → eager.

### Mental model

> **Shapeless capture = "the only things allowed to vary at runtime are values fed as *data*; everything
> the graph's *structure / shapes* depend on must be a compile-time constant or a fixed-capacity
> buffer."** Position, valid-past, and the KV delta are data. Capacity, head counts, hidden dims, and
> rotary structure are static. That contract is what lets one compiled closure serve an unbounded decode.

---

## 4. Why prefill is `ShapeKeyed`, not `Shapeless`

Prefill is the same decoder subgraph but at query length `S > 1`. The causal mask's query-position
`arange(0, S)` extent and the KV write width are read as **Rust ints during the trace and baked as
constants** (`CompiledConfig::prefill`, `engine.rs:300-314`). One shapeless closure traced at one `S`
would miscompute at another — so prefill retraces per distinct `S` (a handful of prompt lengths) while
replaying the fused closure for repeated lengths. The shape-keyed prefill trace also statically narrows
attention to the valid prefix `[0, valid_past + S)` (both static per shape key), so it does not run SDPA
over the full KV capacity — which is what turned prefill from a TTFT regression into a win
(`compiled.rs:113-123`).

---

## 5. Empty-tensor guard

For `ShapeKeyed` paths, `try_compiled` skips to eager on **every** call if any dynamic input has a
zero-size dim (an empty tensor): `mlx_compile` over a zero-size shape can abort inside MLX, and a
shape-keyed closure would otherwise retrace straight back into the same abort. Shapeless decode never
sees empty inputs (`compiled.rs:309-321`).

---

## 6. Current limitation (deferred)

The shapeless growing-KV decode capture is currently hardwired to `com.microsoft::GroupQueryAttention`
(`detect_shared_kv`, and the `kv_alias` / `rope_as_data` / `delta_copyout` machinery). Generalizing it to
ai.onnx `Attention` is a larger change because that op has a different KV/mask contract (additive mask +
`is_causal`, no `seqlens_k`), so it does not yet get the shapeless decode fast path. Static-shape
`Attention` subgraphs are still eligible for the **general** `ShapeKeyed` capture.

---

## 7. Nemotron streaming attribution (issue #10)

The supplied English fp32 Nemotron Speech Streaming 0.6B export and `parakeet-rs` streaming engine
were measured on an M1 Max, 32 GiB, macOS 27.0.1, AC power, ORT 1.31.0, MLX 0.32.2. Model revision:
`altunenes/parakeet-rs@4d2a8bc71f5c896ec40faa59732e6716295edaf2` on Hugging Face, directory
`nemotron-speech-streaming-en-0.6b`. Rust engine revision:
`altunenes/parakeet-rs@6ad1501898842fda76f86f7bcad6b361daf5df9e`.

This is **not** a reproduction of the historical CPU 10.5s / MLX 10.3s result: the reporter's
original audio/runtime/timer are unknown. The authorized alternate audio fixture repeats/truncates
Whisper's public `tests/jfk.flac` to 40 seconds after mono 16 kHz PCM16 conversion. Fixture SHA256:
`d20746b9aae79111ce0775605dc02960454df04f4827f34450aa648c64044f67`.

The measurement adapter retains the supplied engine's preprocessing, cache updates, greedy decoder,
8960-sample chunks, final-chunk padding, and three zero flush chunks. It times in-memory processing,
excluding loading/audio I/O/printing; resets all streaming state before every utterance; and records
token IDs/logprobs plus encoder/decoder wrapper wall times. Each fresh, sequential provider process
reuses its sessions for three full warmups followed by ten timed utterances. Session settings:
Level3 graph optimization, four intra-op threads, one inter-op thread. The MLX process explicitly
registers its canonical provider library path; the separate CPU process registers no MLX library.
Final timings have tracing, ORT profiling, verbose/claim logs, captures, and compile kill switches off.

| Provider/source | Median seconds | p95 seconds | Median encoder / decoder seconds |
|---|---:|---:|---:|
| ORT CPU | 8.204 | 8.658 | 7.884 / 0.292 |
| MLX main `57a0d2e` | 7.312 | 7.786 | 6.029 / 1.257 |
| MLX coverage fixes `605f11e` | 5.862 | 5.932 | 5.435 / 0.399 |
| MLX coverage fixes + current main, `5a52393` | 5.849 | 5.996 | 5.421 / 0.400 |
| MLX plus reviewed value-guarded boundary specialization | 2.671 | 2.748 | 2.254 / 0.391 |

Run order was CPU, main, coverage fixes, specialization; the initial specialization candidate measured
2.912s median / 3.178s p95. A later confirmation ran updated coverage fixes, specialization, CPU and
measured medians 5.849s, 2.809s, 8.062s respectively; p95s were 5.996s, 3.552s, 8.222s. After review,
the boundary exemption was explicitly restricted to general capture, preserving prefill's stricter
guard. The rebuilt/revalidated final source produced the 2.671s result above in another fresh
three-warmup/ten-timed process. These are sequential local observations, not confidence intervals.
The machine already had about 4.76 GiB swap in use; free-memory percentage was 54–56%, with no
recorded thermal/performance warnings. Existing swap and other host activity limit precision.

The reviewed provider's SHA256 is
`4860d51c27093afe1946e1d16753500cad8d4ab2040ec739334e7d6a06b0f60f`; the loaded ORT dylib's is
`83973572676032b4ba9418afed0b51f45599d35bb4e4628ca8ae1954c8362e31`. Separate loader diagnostics
verified the intended provider/ORT/MLX/MLX-C libraries, not just the requested environment variable.
Final first-chunk median was 34.93ms, steady chunk median/p95 35.76/39.02ms, final audio chunk
35.25ms, and flush-chunk median 34.53ms. Compute/audio RTF was 0.0668.

All warmup/timed utterances produced the same 131 token IDs and transcript as CPU, with 75 encoder
and 656 decoder calls. Maximum emitted-token logprob difference was `1.15e-5` for coverage fixes
and `7.63e-6` with specialization. All encoder outputs/caches on identical first/steady/final feeds
also agreed with CPU at `rtol=1e-4, atol=1e-4`; integer lengths were exact. Maximum absolute errors
were `8.05e-7` for encoded features, `3.91e-6` for channel cache, and `1.26e-4` for convolution cache.
Model weights, audio, transcripts, and raw profiles remain local.

Attribution used separate diagnostic runs, never their timings as final performance results:

- Coverage fixes reduced 22 partitions to nine, claiming 2739/2746 optimized nodes. The remaining
  CPU `Range` x6 and `Tile` x1 took about 4ms combined in the profiled utterance; their direct compute
  cost is not the main bottleneck. Main's tiny decoder partitions also imposed substantial overhead.
- Six encoder partitions, including the 2293-node block, still reconstructed eagerly: 450 eager
  executions per utterance. Translation spans totaled about 3.72s, dominated by Reshape/Slice.
  These spans can include synchronization, not just CPU graph construction.
- Original-graph diagnostic outputs at calls 0/37/74 showed Slice ends `[77]` and the first-layer
  reshape target `[1,-1,1024]`, while live cache valid length changed from 0 to 70. The fix does
  **not** assume those observed values stay constant: it guards boundary parameter values every call.
- Specialization produced 1837 general executions, nine first-use misses and no eager executions
  or retraces in the fixed-chunk diagnostic utterance. Full output copies remained about 1849 MiB;
  copy volume was not eliminated. Evaluation remains GPU-inclusive wall time, including lazy JIT
  work on first use, not a measurement of pure GPU kernel duration. Warmup excludes that first-use
  work from the steady-state table above; closure-build counters alone do not quantify it.
