# Four elementary contrib operators

The MLX EP registers **com.microsoft v1 QuickGelu, BiasAdd, IsAllFinite and
ScaledSiLU**. This four-op scope supersedes PR #72's original five-op addition;
FastGelu remains a separate OPEN audit obligation, detailed below. Source
support is not complete runtime qualification or completion of the operator audit.

| Operator | Legal types | Contract |
| --- | --- | --- |
| QuickGelu | F16, BF16, F32, F64 | Required X/Y of matching type/shape; FLOAT alpha defaults to 1.702. Typed X * sigmoid(cast_T(alpha) * X). No optional input. |
| BiasAdd | F16, F32 | Exactly X[N,S,C], bias[C], skip[N,S,C]; typed (X + bias) + skip. No generic broadcasting or optional inputs. |
| IsAllFinite | Homogeneous F16, BF16, F32, F64 | One or more independently shaped inputs, scalar BOOL output. Empty tensors contribute true. INT isnan_only/isinf_only default to zero; any nonzero value is true. Both nonzero is invalid. |
| ScaledSiLU | Independent T and M: F16, BF16, F32 | Required X plus optional rank-zero scale, including all nine T/M pairs. Present scale overrides FLOAT alpha (default 1); absent/empty-name slot uses alpha. |

ScaledSiLU computes Z_T = cast_T(float32(X) * effective_scale), then
S_T = cast_T(sigmoid(float32(Z_T))), then Y = cast_T(float32(Z_T) * float32(S_T)).
It is not Swish(X, scale). Scale is independently decoded into F32, never
prematurely rounded to T. A one-element vector is not a scalar. Live geometry,
type/arity and attribute types are validated rather than silently coerced.

All four use provider-owned eager host arithmetic/scans, not ORT CPU fallback.
Recursive eager boundaries exclude decode/general/prefill capture, including
nested bodies, without globally widening host readers or movable dtypes.
Existing float64 CPU coloring and control-flow payload restrictions remain.
The RandomLike metadata-only input policy is not a payload transport grant.
BF16 fixtures use explicit Cast boundaries rather than assuming direct NumPy
BF16 loader support. Scalar/empty/symbolic/live/initializer/intermediate inputs,
signed zero and nonfinite values retain their existing independent references
and thresholds.

## Mathematical contracts and source authority

For `com.microsoft`, official CUDA behavior takes precedence over conflicting
CPU behavior. References are independent mathematical/typed contracts, not
observed-output transplants or an assumption that CPU is the oracle. Bit identity
between different exp/libm/device implementations is not a prerequisite for
mathematical support. The existing literal goldens, typed stages and comparison
thresholds are nevertheless binding; a concrete disagreement must be classified,
not hidden by widening a tolerance or substituting an oracle.

Pinned ORT v1.31.0 source commit
[`402979ac1bef89dcd55add41eaa7277c407599eb`](https://github.com/microsoft/onnxruntime/commit/402979ac1bef89dcd55add41eaa7277c407599eb)
is semantic source authority, not installed-binary identity proof:

- [QuickGelu schema/function, contrib_defs.cc 628-660](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/core/graph/contrib_ops/contrib_defs.cc#L628-L660).
  The F32 [CPU implementation, activations.h 62-97](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/contrib_ops/cpu/activations.h#L62-L97)
  uses MLAS logistic or, for alpha=1, MLAS SiLU. This does not supersede the
  schema's typed mathematical contract or prove CPU/CUDA numerical equivalence.
  Fixtures retain independent references and include alpha=1, default/explicit
  1.702, zero, negative and other positive alpha. Native78's two default-F32 B1
  positives establish only those historical CPU controls, not this candidate.
- [BiasAdd schema, diffusion_defs.cc 198-228](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/core/graph/contrib_ops/diffusion_defs.cc#L198-L228)
  and [CUDA additions, bias_add_impl.cu 31-53](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/contrib_ops/cuda/diffusion/bias_add_impl.cu#L31-L53).
  Independent typed addition is the reference; no CPU contrib kernel is invented.
- [IsAllFinite schema, contrib_defs.cc 965-1006](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/core/graph/contrib_ops/contrib_defs.cc#L965-L1006)
  and [constructor rejection, isfinite.h 14-27](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/contrib_ops/cuda/math/isfinite.h#L14-L27).
  Exact boolean classification is independently referenced without a fabricated
  CPU kernel. Direct backend registration does not narrow schema legality.
- [ScaledSiLU schema, bert_defs.cc 5587-5611](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/core/graph/contrib_ops/bert_defs.cc#L5587-L5611)
  and [CPU stages, scaled_silu.cc 12-80](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/contrib_ops/cpu/bert/scaled_silu.cc#L12-L80).
  Line 61 requires NumDimensions()==0, not Size()==1. Retained official CUDA
  discovery source `contrib_ops/cuda/bert/scaled_silu.cc` independently checks
  rank zero and registers all nine T/M combinations; `scaled_silu_impl.cu`
  independently loads scale into F32 and explicitly casts Z, sigmoid and product
  to T (lines 20-40). Those stages agree with the implementation's mathematical
  contract. These retained discovery bytes are not among the twelve immutably
  verified v1.31 source files, and are not proof of installed CUDA identity or
  execution. Existing exact literal goldens stay binding; Native78 never reached
  B6/B7 CPU controls.

Pinned schemas and the twelve independently verified upstream source files are
semantic evidence only: neither they nor historical discovery sources prove the
identity of a loaded runtime, a schema-catalog generator, or actual device
dispatch. No CUDA execution is claimed.

## FastGelu remains a separate OPEN lifecycle

FastGelu is not registered by this four-op addition. It is not removed from the
audit queue, declared impossible, or represented as completed. Historical
five-op commit `77f3ae388da3b296ea1c54d0204381439280ed8f`, its reviews and original
299 manually enumerated static Python cases remain preserved in history. Its
original minus-two and present-bias goldens/tolerances and failed Native78 raw
evidence remain immutable, not silently replaced by this filtered fixture module.

The pinned FastGelu schema function and CPU kernel use different polynomial
constants and arithmetic/transcendental paths. Native78 failed a CPU-versus-schema
reference check before MLX registration, on a separate historical baseline, not
this candidate. That failure is not a CUDA defect or a failed current MLX result.
CPU matching is not the prospective contract gate under CUDA-first policy.
Retained authorized CUDA evidence stops at the wrapper's registration, bias
validation and `LaunchFastGeluKernel` call; it does not establish device
coefficients, rounding stages or scalar/half2 behavior. The remaining obligation
is an authorized CUDA-contract arithmetic/reference assessment, followed by any
justified source correction and focused validation without weakening original
goldens/thresholds. No guessed CPU tanh substitute or excluded-source refetch is
authorized. Shape-expanding bias's schema/function shape-inference conflict is a
separate OPEN boundary. Simpler well-defined scalar/shape-preserving
generalizations are not rejected merely because of CPU-kernel restrictions.

## Evidence boundaries

The filtered module has **17 Python test functions / 233 manually enumerated
STATIC parameter cases**, plus **3 Rust test functions**. These are not observed
collection/execution counts. The original five-op hosted run `39934916520` checked
out merge `18de3d008ea0c2793c1247f609d226d008bafff4`, tree
`2c4fdfae9cf8370d021833ae136720d0532467af`; its successful results do not qualify
this four-op final diff or establish module-specific execution of the old 299.
Strict sessions and nonempty actual Node `*_kernel_time` profiles retain the
constructor/runtime/assignment fallback prohibitions and all-MLX requirements.
The provider descriptor must be exactly ordered `[MLX]` or `[MLX, CPU]`; listing
CPU is not actual CPU assignment. Function-expanded/rewritten already-owned
positives are not missing-support evidence, and provider ownership does not prove
physical GPU/compiled/host route. Exact dtype/shape/value/live replays, BF16 Cast
boundaries, all nine scale pairs and nested eager routing remain focused evidence
targets. Numerical/provider failure, invalid model, strict assignment gap and
actual loader limits must remain distinct. No oracle substitution, tolerance
weakening, skipped failure or retry is authorized. Final-diff CI and separately
admitted focused runtime evidence must be attributed to their exact inputs and
checkouts; this source reconciliation grants no local execution or binary reuse.
