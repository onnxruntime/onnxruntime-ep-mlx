# Attention pullbacks for training

The provider executes versioned `ai.onnax::AttentionVJP_v1_*` local functions
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
