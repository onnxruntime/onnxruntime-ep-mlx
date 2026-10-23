# Elementary contrib operators (source checkpoint)

The MLX EP registers **com.microsoft v1** FastGelu, QuickGelu, BiasAdd,
IsAllFinite and ScaledSiLU. This is source implementation and fixture coverage,
not runtime qualification or completion of the full operator audit.

All five deliberately execute **eagerly on provider-owned host data**. Live graph
inputs, initializers and evaluated intermediates use the existing checked host
reader. No ORT CPU fallback is used. Compile classification excludes decode,
general and prefill capture, recursively through control-flow bodies, and splits
these eager boundaries even outside decoder graphs. Existing float64 CPU-stream
coloring applies; no global dtype/transport policy is widened. Float64 payloads
inside control flow remain refused by the existing body guard.

## Implemented contracts

| Operator | Types | Inputs and behavior |
| --- | --- | --- |
| QuickGelu | F16, BF16, F32, F64 | One X; FLOAT alpha defaults to 1.702. Typed v1 function: X * sigmoid(cast_T(alpha) * X). |
| FastGelu | F16, BF16, F32, F64 | X, optional homogeneous bias; absent and explicit empty-name slot are equivalent. Typed schema function constants 0.797885, 0.035677 and 0.5, with rounding at each ONNX operation. |
| BiasAdd | F16, F32 | Exactly X[N,S,C], bias[C], skip[N,S,C]; Y = (X + bias) + skip, with typed additions. No generic residual broadcasting. |
| IsAllFinite | homogeneous F16, BF16, F32, F64 | At least one tensor, arbitrary individual shapes; scalar BOOL output. Every element of every live input participates; empty inputs contribute true. |
| ScaledSiLU | independent T and M: F16, BF16, F32 | X plus optional **rank-zero** scale; scale overrides FLOAT alpha (default 1). Z = cast_T(float32(X) * float32(scale)); S = cast_T(sigmoid(float32(Z))); Y = cast_T(float32(Z) * float32(S)). |

IsAllFinite INT flags use nonzero truth, including negative integers. Both flags
nonzero is **invalid**, not a precedence choice. With isnan_only only NaNs fail;
with isinf_only only either infinity fails; otherwise either kind fails.

ScaledSiLU is **SiLU(scale * X)**, not Swish(X, scale). Scale is never pre-rounded
to X's type. Host evaluation makes the two intermediate T roundings observable
and prevents compiled optimization/fusion from erasing them. A vector of length
one is not a scalar. BiasAdd and all optional inputs validate live geometry as
well as static/symbolic metadata. Zero dimensions are legal.

FastGelu's schema function explicitly uses standard Add(X, bias); it therefore
defines more bias shapes than the CPU/CUDA kernel's rank-one, last-dimension
helper. The EP implements these function-defined broadcasts **when they preserve
X's shape**, including scalar X/bias and multidimensional bias. Expanding X to a
larger result is **OPEN and declined with a specific claim reason**: the function
would expand it, but schema shape inference unconditionally propagates X's shape.
This boundary is not presented as a backend impossibility or kernel restriction.

QuickGelu and FastGelu follow the typed schema function rather than claiming
bitwise identity with every fused upstream kernel. ScaledSiLU follows the
explicit staged CPU implementation. Transcendental results retain host libm
behavior, including IEEE NaN/infinity arithmetic. No performance claim is made.

## Immutable semantic authority

Matching upstream tag **v1.31.0** resolves to commit
[`402979ac1bef89dcd55add41eaa7277c407599eb`](https://github.com/microsoft/onnxruntime/commit/402979ac1bef89dcd55add41eaa7277c407599eb).
These are source contracts, **not proof of the installed runtime binary's
identity**, generator provenance or a fresh runtime query.

- [bert_defs.cc, FastGelu schema/function, lines 3591-3636](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/core/graph/contrib_ops/bert_defs.cc#L3591-L3636):
  typed constants, optional Add, shape propagation.
- [bias_gelu_helper.h, lines 17-45](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/contrib_ops/cpu/bert/bias_gelu_helper.h#L17-L45):
  narrower fused-kernel geometry, not the function's full broadcast definition.
- [contrib_defs.cc, QuickGelu, lines 628-660](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/core/graph/contrib_ops/contrib_defs.cc#L628-L660):
  default alpha, full type set and context-dependent typed function.
- [contrib_defs.cc, IsAllFinite, lines 965-1006](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/core/graph/contrib_ops/contrib_defs.cc#L965-L1006)
  and [CUDA constructor, lines 14-27](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/contrib_ops/cuda/math/isfinite.h#L14-L27):
  homogeneous variadic types, scalar bool, both-true rejection.
- [IsAllFinite CUDA implementation, lines 25-64](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/contrib_ops/cuda/math/isfinite.cc#L25-L64):
  empty identity, multi-input scan, float/half/double CUDA registrations.
- [BiasAdd CUDA validation, lines 33-70](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/contrib_ops/cuda/diffusion/bias_add.cc#L33-L70)
  and [addition order, lines 31-53](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/contrib_ops/cuda/diffusion/bias_add_impl.cu#L31-L53);
  [BiasAdd schema, diffusion_defs.cc, lines 198-228](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/core/graph/contrib_ops/diffusion_defs.cc#L198-L228).
- [ScaledSiLU schema, lines 5587-5611](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/core/graph/contrib_ops/bert_defs.cc#L5587-L5611)
  and [CPU implementation, lines 12-80](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/contrib_ops/cpu/bert/scaled_silu.cc#L12-L80):
  independent M, rank-zero scale, float32 work and three T casts.

The retained inventory SHA256 is
`136a948551de882468e8b9f4d16c6b7d11340dd8cb92120f191acd169224b323`;
its five `runtime_schemas` records, not its empty standard `schemas`, were read
with stdlib JSON. Historical schema/source discovery was at 755; the scheduler
checked F470 runtime bytes separately. Neither fact proves catalog generator
binary identity. Current upstream main `cb2a4f584be4f18d60bfb072e0813b9d110a6f9b`
was also read for comparison, but release-tag sources above are the cited authority.

## Fixtures and pending runtime admission

`tests/ops/test_elementary_contrib.py` contains typed independent oracles,
literal ScaledSiLU goldens, every T/M pair, attribute defaults/overrides,
missing/empty optional slots, zero/scalar/symbolic shapes, same-session changing
values and shapes, initializer/live/intermediate inputs, nested If parents,
malformed arities/types/geometry and default/forced-eager routes.
BF16 is exercised through explicit Cast boundaries rather than pretending that
the Python ORT loader accepts a NumPy BF16 feed. CPU kernels are not used as
proxies: source registrations cover less than schema types (FastGelu CPU F32;
IsAllFinite CUDA F16/F32/F64), and CPU availability does not define EP feasibility.

Every positive session disables CPU assignment fallback and runtime fallback.
Profiles select a **nonempty** set of Node `*_kernel_time` events independently
of provider, then require **every event** to report MLX, rejecting missing provider
metadata. Automatically listed CPU providers are not mistaken for actual CPU
assignment. Rust fixtures cover typed literal stages, geometry and recursive
compiled-route exclusion. Fixture source inventory is not test collection.
Only source/AST inspection and Rust 1.98 formatting are permitted at this
checkpoint; build, collection, inference and GPU validation remain unrun pending
the scheduler's explicit local-window admission.
