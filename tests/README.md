# MLX EP tests

The EP is MLX-native and written in Rust (`rust/`): an ONNX fused decoder subgraph is translated to
an MLX graph, with MLX as the sole compute path. The suites are **Python** (pytest); there is no
longer any C++/CTest build.

Build the EP and configure its dependencies first (see [building](../docs/BUILDING.md)).
Run these commands from the repository root:

```bash
ORT_INCLUDE_DIR=<ort-include-dir> cargo build --manifest-path rust/Cargo.toml --release
# => rust/target/release/libonnxruntime_mlx_ep.dylib
```

Install the Python test dependencies used by CI:

```sh
python -m pip install "onnxruntime==1.31.0" "onnx==1.23.2" numpy onnx_ir pytest
export ONNXRUNTIME_MLX_EP_LIB="$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib"
export DYLD_LIBRARY_PATH="$ORT_HOME/lib"
cargo test --manifest-path rust/Cargo.toml --release --lib
python -m pytest tests/ops -q
python -m pytest tests/backend -q
```

Set `ORT_HOME` to the release root from the build guide, or point
`DYLD_LIBRARY_PATH` at the corresponding ORT `lib/` directory.

## `tests/ops` — op-correctness (pytest)

Each ONNX decoder op the EP translates to MLX (MatMulNBits, GroupQueryAttention, RMSNormalization,
SkipSimplifiedLayerNormalization, GatherBlockQuantized, Softmax, Add/Mul/Sub/Sigmoid/Cast, and the
full modular registry in `rust/src/ops/*.rs`) is run through the plugin and compared, tolerance-gated,
against ORT's CPU EP reference (fp16 too) or a numpy reference (bf16). Parametrized `pytest`
(`test_*.py` + `_models.py` builders); the EP is registered once by `conftest.py` from
`ONNXRUNTIME_MLX_EP_LIB`. Models are built with the ONNX IR (`onnx_ir`:
`ir.Value`/`ir.Node`/`ir.Graph`/`ir.Model`), not `onnx.helper`.

```bash
export ONNXRUNTIME_MLX_EP_LIB="$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib"
export DYLD_LIBRARY_PATH=<ort-prebuilt/lib>
python -m pytest tests/ops -q
```

Running `pytest` without `ONNXRUNTIME_MLX_EP_LIB` set **skips** the suite (rather than failing), so
it is safe to include in any pytest invocation.

`test_attention_semantics.py` uses small, locally built models and an independent float64 SDPA
oracle rather than version-specific ONNX backend fixture names. It covers fully masked rows
(including the combined causal/explicit mask), broadcast mask ranks, FP16/FP32, GQA and cached
3D layouts, and preserves input NaNs/infinities in rows with attendable keys. Both fused and composed
forms zero fully masked output rows even when Q or V is nonfinite; zero probabilities alone would
still produce `0 * NaN` in the value contraction. The cached decode tests in
`test_attention_ext.py` also require a compiled MISS/HIT replay as an empty mask becomes nonempty.
An empty allowed key set must produce zero, not a uniform attention average or NaN; the guard is
based on the additive mask, never a blanket NaN replacement.

### Attention FP16 comparisons

Two ONNX 1.23.2 cases, `test_attention_4d_causal_fp16` and its `_expanded` variant, have a
case-specific backend comparison bound of `2 * finfo(float16).eps` (2^-9). The original models,
inputs, expected arrays and absolute tolerance are unchanged. This is **not** an Attention-wide
tolerance change: the GQA FP16 case and both fully masked robustness cases keep their original
comparisons.

The [released reference](https://github.com/onnx/onnx/blob/v1.23.2/onnx/reference/ops/op_attention.py)
rounds scaled Q/K, logits, and the softmax exponential/sum/division to FP16. The expanded graph's
Softmax has FP16 input/output, but does not require every internal arithmetic operation to round to
FP16. MLX's precise softmax accumulates in FP32 before rounding its output; the composed Attention
path uses the same accurate helper as standalone Softmax, while retaining `softmax_precision`'s
input cast and the probability/output dtype boundaries. We do not add intermediate rounding to
make a more accurate result reproduce the reference's rounding error.

The [same-Mac pre-fix diagnostic](https://github.com/onnxruntime/onnxruntime-ep-mlx/actions/runs/39703543192)
confirms that ORT CPU and MLX both return `0.3984375` at `[0,0,2,4]`, versus the fixture's
`0.39794921875`; independent float64 SDPA gives `0.398295323199067`. For the expanded MLX graph,
`[0,0,1,4]` is `0.468994140625` versus `0.469482421875`, with float64 value
`0.469189668630952`. In both examples the native value is closer to float64 than the fixture, yet
fails the fixture's 1e-3 relative comparison. Both discrepancies are exactly two FP16 ULPs.

`tests/backend/test_attention_precision.py` therefore also runs the exact two public in-memory
fixtures and requires **both** the expected data and native output to meet the **original**
`rtol=0.001, atol=1e-7` against an independent float64 oracle, plus an exact two-ULP maximum
distance from each other. Shape, dtype and finiteness are checked separately. Negative tests reject
three-ULP drift, wrong masks/zero outputs, NaNs, wrong shapes/types, and forward-error violations
even when fixture agreement is exact. The public runner's slightly wider relative bound alone is
not the acceptance criterion.

## `tests/backend` — official ONNX fixtures

The public ONNX `BackendTest` runner loads node, simple, and PyTorch
operator/converted cases. CI pins ONNX 1.23.2. Node models and input/expected
datasets are obtained in memory via `load_model_tests(kind="node")`; packaged
node fixture directories and the generate-data CLI are not needed. Collection
fails if any intended category is empty.

Models run unchanged with their original datasets and tolerances, except for
the case-specific FP16 Attention comparisons described above. Explicit runtime
IR/domain-opset limit rejections become version-reasoned skips. Attention-25
window attributes added in ONNX 1.23 are skipped only when ORT rejects that exact
official attribute on the matching valid model node; other invalid-graph errors
fail.

BF16 tensor inputs/outputs skip only when independent public ORT CPU probes
demonstrate the required NumPy input/output conversion is unavailable. These
Shape/Constant probes use no casts, preserve dtype/values when supported, and
are cached per process. They do not rewrite real models or suppress
internal-BF16-only models, malformed graphs, or unrelated runtime failures.
Kernel errors and output mismatches still fail, with CPU fallback available
for ops MLX does not claim.

The loader/adapter regressions can run without Metal:

```sh
python -m pytest tests/backend/test_backend_adapter.py -q
```

## `tests/conformance` — ONNX-standard fuzz-conformance (opt-in)

Bounded fuzz-conformance of the MLX EP against the ONNX standard via `cbourjau/onnx-tests`. Each op
is fuzzed in its own subprocess so a single native crash cannot abort the run. It reads the EP dylib
from `MLX_EP_LIB`. See [`tests/conformance/README.md`](conformance/README.md).

## Memory-leak checks

RAII (`impl Drop` in `rust/src/mlx.rs`) gives deterministic teardown, so leak-checking is done ad hoc
with macOS `leaks` against the Rust stress scripts rather than a dedicated CTest target:

```bash
MallocStackLogging=1 leaks --atExit -- \
  env ONNXRUNTIME_MLX_EP_LIB="$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib" \
      DYLD_LIBRARY_PATH=<ort-prebuilt/lib> \
  python rust/stress_add.py
```

The stress scripts (`rust/stress_add.py`, `rust/stress_norm_attn.py`, `rust/stress_wave2.py`) exercise
the fast-norm / fast-SDPA / RoPE / multi-output paths across many back-to-back sessions and report
**0 leaks / 0 bytes**.

## ONNX Runtime telemetry is disabled locally, not in CI

Local tests and benchmarks set `ORT_DISABLE_TELEMETRY=1` so a developer machine does not phone
home. CI deliberately leaves ORT's default enabled so its maintainers receive a usage signal.
An explicit environment value wins in either context.

ORT reads the variable when its native library loads, so local entry points must set it **before**
anything imports `onnxruntime` — a pytest fixture runs too late. The policy is implemented in three
places because none of these entry points passes through the others:

| Where | Covers |
| --- | --- |
| `conftest.py` (rootdir) | every pytest suite; imported before the per-suite conftests import ORT |
| `bench/bench.py` | the benchmark runner, which is invoked directly |
| `tests/conformance/run_conformance.sh` | onnx-tests subprocesses, driven from *its* checkout, so this repo's conftest is never collected |

ORT's telemetry worker can rarely abort during interpreter teardown after a green test summary.
An affected CI job may opt out at job or step scope. `tests/ops/test_telemetry_policy.py` guards the
local-vs-CI policy and prevents workflow-wide opt-outs.
