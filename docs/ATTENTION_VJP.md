# Attention pullbacks for training

The provider executes versioned `pkg.onnax::AttentionVJP_v1_*` local functions
emitted by onnax's optional deployment pass. Each function also contains an
ordinary ONNX implementation for other runtimes.

The six inputs are Q, K, V, the forward output Y, the upstream gradient G, and a
mask. The function name records rank, head counts, head dimension, FP32 scaling,
causality, selected Q/K/V gradient outputs, mask type, and a fallback-body digest.
The provider checks this contract before claiming the node. FP32 rank-three and
rank-four inputs support head dimensions 16, 32, 64 and 128, including GQA/MQA.
Masks support boolean or additive values, broadcast dimensions, short key tails,
and singleton key broadcasting. Causal attention uses top-left alignment.

The tiled Metal pullback recomputes log-sum-exp, then computes dQ and grouped
dK/dV. It uses fixed-size scratch storage and skips causal tiles above the
diagonal. Fully masked rows produce zero gradients. dK/dV sum shared-head
contributions without atomic writes.

Small score matrices use MLX GEMM and softmax operations. Their intermediates have
local ownership and expire after their final consumer. The policy is controlled
by `ONNXRUNTIME_EP_MLX_ATTENTION_VJP_DENSE_LIMIT_BYTES`: its default is 64 MiB,
measured as `batch * query_heads * query_length * key_length * 4`. Set it to `0`
for tiled execution, or increase it to spend more temporary memory on GEMM
throughput. This is the size of one score tensor, not a process memory limit.

## Large blocked lookup-4 products

For at least 128 activation rows, frozen lookup-4 MatMul decodes an execution-local
FP32 weight and invokes MLX's tuned GEMM. Smaller products use the fused lookup
kernel. Weights remain packed in the ONNX model. The temporary decoded weight is
released with the execution graph; it is not added to the persistent weight cache.
`ONNXRUNTIME_EP_MLX_LUT4_GEMM=0` selects the fused lookup kernel for comparisons.

The regression suite builds models with ONNX IR and compares both attention
policies with independent NumPy derivatives and portable ONNX function bodies.
Lookup-4 tests replay both weight layouts across the 128-row dispatch boundary.

## Reuse frozen precision and layout views

Immutable initializers retain provenance through Cast, Transpose and Identity.
Large FP16/BF16/FP32 conversions are cached once per session. A very wide frozen
projection's transposed FP32 matrix is packed contiguously for its backward
GEMM. This avoids the slow tall-reduction transposed-RHS route while preserving
all weight values.

Runtime inputs and shape-derived data keep their live semantics. Control-flow
formals shadow outer provenance and restore it on scope exit. Tests replay
changing weights and sizes, and check that a loop/body formal cannot enter the
immutable cache. `ONNXRUNTIME_EP_MLX_PACK_FROZEN_VIEWS=0` disables this policy.
These derived arrays live with the session; the allocator's idle-buffer budget
controls a separate cache.

## Half-precision lookup weights

Lookup-4 recognition also follows FP16/BF16 casts around decoded weights.
Forward products decode directly into the selected half dtype and call MLX GEMM.
FP32 adjoints widen those same rounded weights before multiplication. A transpose
preserves the rounding boundary. Decode buffers are local to the execution;
packed weights stay compressed between calls. Shared decoded outputs retain their
original producer graph.

Measure the precision paths with the shared correctness graphs:

```sh
python bench/lookup4_precision.py --library /absolute/path/to/libonnxruntime_mlx_ep.dylib \
  --precision bf16 --rows 1024 --inner 1024 --outer 2048 --output bf16-forward.json
```

Add `--adjoint` for a transposed FP32 derivative product, or `--disable-fusion`
for the portable decode chain. Each invocation uses ten warmups, 100 synchronous
samples, disabled CPU fallback, and disabled ORT graph optimization. Output files
include raw samples, runtime/library identity and a NumPy output for comparison.
