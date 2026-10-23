# Unique contracts and execution

`ai.onnx::Unique` schema 11 is implemented by provider-owned eager host
sorting/grouping. This expands the previous flattened, non-scalar
float32/int32/int64 implementation; Unique was not a missing operator.

The registered input types are bool, signed/unsigned 8/16/32/64-bit integers,
float16, float32 and float64. Scalar, unknown-rank, symbolic, dynamic and empty
inputs resolve their actual metadata and values on every invocation.
Without `axis`, the input is flattened in row-major order and Y is one-dimensional,
even for scalar input. With `axis`, whole slices are compared lexicographically
in the row-major order of the remaining dimensions; Y retains the input rank and
replaces only the selected dimension. Negative axes are normalized at full int64
width and must lie in `[-rank, rank-1]`. An axis is invalid for rank zero.

`sorted=1` (the default) orders values/slices ascending; `sorted=0` orders groups
by their first occurrence. All optional-output combinations, including holes,
preserve the same Y and group numbering. Indices identify first occurrences,
inverse indices reconstruct input values/slices, and counts follow Y's order.
Metadata outputs are int64, not int32 indices widened after a gather.
Original representative bytes are copied, not reconstructed through float32.
In particular, adjacent large int64/uint64 and float64 values stay distinct.
Floating signed zeros compare equal, with the first occurrence's bytes retained.
Infinities and subnormals use their numeric ordering.

## NaN authority limits

The schema links NumPy but does not specify a NaN equivalence relation, NaN
ordering, or representative payload/sign selection. The implementation follows
the ONNX Reference/NumPy convention for the supported forms: flattened NaNs form one group, sorted last;
NaN-containing axis slices are distinct, ordered lexicographically with NaNs
after non-NaNs. Equivalent sort keys retain input order. Y always retains the
first representative's original bytes, including NaN payloads and signed zero.
This is an explicit reference-aligned choice, not a claim that all authorities
agree on NaNs or on representative bytes.

One narrow form remains an explicit runtime error: **sorted float16 axis slices
containing NaNs**. NumPy 2.3.4's structured half comparison puts these NaNs before
finite values, unlike its float32/float64 ordering. Neither the schema nor ORT's
NaN comparator resolves this discrepancy. Unsorted float16 axis NaNs and
flattened float16 NaNs remain implemented. A regression records the exact
Reference disagreement and checks invalid-then-finite recovery in the same EP
session; it does not pretend this unsupported form passed.

ORT 1.31 CPU uses `std::map` with floating `<`, which does not define a strict
weak ordering when NaNs occur. It is not a reliable NaN oracle. ONNX 1.22 and
the pinned ONNX 1.23.2 Reference have a separate unsorted bug: they gather original X on axis zero
instead of reordering sorted Y on the requested axis (and returns early for
single-output models). Tests record that defect explicitly; on those versions,
Reference comparisons use its sorted variant while every EP result, including
unsorted results, is checked against an independent typed enumeration oracle.
No EP output is reshaped, tolerance-relaxed, or replaced with CPU fallback.

## Empty geometry, live reads and partitioning

An empty axis contains zero slices. A nonempty axis with zero-sized slices
contains one unique empty slice: its first index is zero, its count is the
axis length, and all inverse indices are zero. For example `[2, 0, 3]` with
axis 0 produces Y shape `[1, 0, 3]`, indices `[0]`, inverse `[0, 0]`, counts `[2]`.
With axis 1 the same input produces `[2, 0, 3]` and empty metadata outputs.

The checked live host reader and geometry validation are shared with NonZero;
NonZero's coordinate/equality contracts are unchanged. Empty buffers never
become Rust raw slices from null pointers. Every extent and nonempty total
element count must fit native int32 indexing; byte products are checked before
reading. Shape/count disagreement and invalid geometry are explicit errors.
Boundary inputs are read live from ORT, initializers from owned data, and
intermediate or control-flow-bound values from the current evaluated MLX array.
No data-dependent cardinality is captured in general, decode or prefill caches.
Unique and parents recursively containing Unique have eager partition
boundaries, keeping eligible adjacent producers/consumers separately compilable.
Existing Mod/NonZero partition rules are retained.

Host sorting takes O(N log N) comparisons (each comparison can inspect an entire
axis slice), plus host materialization and output copies. It can synchronize
pending MLX work, consumes host memory, and is **not a GPU speedup** or an ORT CPU
kernel fallback. Float64 transport uses the existing isolated MLX CPU stream.

## Remaining forms and proof gates

Complex/string and encoded transport remain explicit provider-pipeline gaps,
not backend impossibilities. Bfloat16 is absent from schema 11 and only added in
schema 28; schema 28 is not registered pending a supporting ORT/schema runtime
and real-route validation. The shared scanner understands bf16 representations
for existing NonZero and pure typed unit coverage, but this does not authorize
illegal bf16 Unique11 graphs.

Existing control-flow pipeline restrictions still apply: a parent with float64
anywhere in its body is declined because bodies share the parent's GPU plan.
This change does not expand float64 control-flow transport or non-tensor parent
boundaries. The recursive neighbor regressions use claimable float32 If bodies.

The source baseline is main `965304020994ed3da4157232102d151b64025ca5`,
tree `044024a9a213be4eb974e6280af0076b402c69dc`; its pristine git archive SHA256 is
`3529465d5e2569948d197c97ad869cd3017e962eb25cb027faba99e95987ed16`.
The pinned manifests are `rust/Cargo.toml` blob
`d5c2fe2742c9f9f250ea3d080477f87ce28cb471`, lockfile blob
`f59cfd0eee7051adb51e3f7ca6c38aa820334ca4`, and `python/pyproject.toml` blob
`f2d588d37a9f31005498c9f8edc6a64a67f37649`. CI selects ORT 1.31.0,
MLX 0.32.4 commit `36969c094354086e15d2a327a9c50feb20b9cf11`, and
mlx-c commit `57731b00e8bed710e0750e7356da9f18f8eb3e20`.

Native baseline/candidate execution, focused Unique/affected suites, route/cache
assertions, full ops/backend/leaks, wheel ABI/Python 3.13 registration and
fresh-current-main source interaction remain **pending** until explicitly
scheduler-granted and executed. Source/schema/oracle checks are not native
proof. Regression route counts are expectations, not observed passes.

Authorities:
[ONNX Unique11/28](https://onnx.ai/onnx/operators/onnx__Unique.html),
[ONNX 1.23.2 Reference](https://github.com/onnx/onnx/blob/e1ac8e74b620bbdc7ba07f0440806bd93f651a28/onnx/reference/ops/op_unique.py),
and [ORT 1.31 CPU](https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/core/providers/cpu/tensor/unique.cc).
