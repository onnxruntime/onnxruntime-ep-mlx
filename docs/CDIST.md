# CDist v1: squared Euclidean and Euclidean

The MLX provider claims `com.microsoft::CDist` v1 for **`sqeuclidean`**
(absent/default or explicit) and **`euclidean`**. Both inputs are required
homogeneous FLOAT or DOUBLE rank-two tensors: A[M,N], B[K,N]. The one required
output is C[M,K] of the same type. Feature dimensions must agree, including on
every live replay; symbolic extents are not frozen. `metric` must be STRING.
Unknown metrics, wrong attribute types, missing slots, extra inputs/outputs,
mixed dtypes, FLOAT16/BFLOAT16 and invalid geometry are not claimed.

The retained ORT 1.31.0 runtime schema describes these types, ranks and the
default. This is retained semantic evidence, not a fresh runtime schema query
or proof of a particular loaded binary. The schema advertises 23 metric names,
but has no p, weights, variance or inverse-covariance parameter. The other 21
metrics remain separate audit obligations; this implementation does not invent
their defaults or claim they are impossible. The independent mathematical
contracts here are sum of squared coordinate differences and its square root.
CPU acceptance or rejection is not the definition of the contract, and no
CUDA execution or CUDA bit identity is claimed.

## Execution and numerical behavior

Execution is **provider-owned eager host arithmetic**, not GPU acceleration,
an MLX compiled reduction, or ORT CPU fallback. It reuses checked live host
readers and materializes an MLX output. DOUBLE outputs retain the existing
isolated MLX CPU-stream policy; FLOAT outputs use the existing output transport.
These are source-derived route descriptions, not measured per-case physical
execution claims. Recursive eager partition boundaries prevent capture of this
host work, including in parent bodies. DOUBLE control-flow payload computation
remains rejected; no global host-reader, dtype or stream policy is widened.

For each pair, subtraction, multiplication and accumulation use the input
precision, in coordinate order; Euclidean applies the same precision's sqrt.
Direct differences avoid cancellation from the norm/dot-product identity.
Empty M or K preserves [M,K], and N=0 produces positive zeros. IEEE NaN/Inf
propagation and typed overflow remain visible: in particular, a squared
intermediate may overflow even when a scaled mathematical Euclidean norm would
be finite. There is no hidden clamp, widening, stable-norm substitution or
CPU-oracle retry. Fixtures distinguish independent mathematical accuracy checks
from these explicitly declared overflow/nonfinite cases, with unchanged fixed
comparison thresholds and hand-derived precision goldens.

Work is O(M*K*N), storage O((M+K)*N + M*K), with constant scratch per pair.
There is no [M,K,N] intermediate or tiling framework. Decoded inputs and the
output require host storage in addition to existing ORT/MLX buffers. Existing
checked layout limits apply separately to both inputs and the output: dimensions
and nonempty element counts must fit int32 indexing, and byte sizes must be
representable. These are safety bounds, not a claim that every such allocation
fits available memory or that large pairwise workloads are fast.

## Focused evidence

`tests/ops/test_cdist.py` uses independent `math.dist` references for finite
mathematical checks, plus exact nearby-large-vector and genuine DOUBLE
discrimination cases, declared typed-overflow/nonfinite expectations, and live
input/initializer/intermediate sources. Default/explicit metrics, default/eager
modes, changing symbolic M/K/N, empty axes, runtime feature mismatch followed by
recovery, strict forms and nested eager routes are covered by source fixtures.
Strict sessions disable constructor/assignment/runtime fallback, accept only
ordered [MLX] or [MLX, CPU] descriptors, and require independently selected,
nonempty actual kernel events whose providers are all MLX.

Source fixtures are not an observed collection count. Actual hosted job
checkouts, JUnit case outcomes and source-only review must be bound separately;
provider ownership does not establish physical GPU or CPU execution. No local
native, scientific import, model/checker, collection, build, inference or
benchmark execution is implied by this source addition.
