# Attention pullbacks for training

The provider executes parameterized `pkg.onnax::AttentionVJP_*` local functions
emitted by onnax's optional deployment pass. Each function also contains an
ordinary ONNX implementation for other runtimes.

The six inputs are Q, K, V, the forward output Y, the upstream gradient G, and a
mask. The function name records rank, head counts, head dimension, FP32 scaling,
causality, selected Q/K/V gradient outputs, mask type, and a fallback-body digest.
The provider checks this contract before claiming the node. FP32 rank-three and
rank-four inputs support head dimensions from 8 through 256 in multiples of eight, including GQA/MQA.
Masks support boolean or additive values, broadcast dimensions, short key tails,
and singleton key broadcasting. Causal attention uses top-left alignment.

The tiled Metal pullback recomputes log-sum-exp, then computes dQ and grouped
dK/dV. It uses fixed-size scratch storage and skips causal tiles above the
diagonal. Fully masked rows produce zero gradients. dK/dV sum shared-head
contributions without atomic writes. Heads from 128 through 256 use eight key rows
per threadgroup. Four score tiles and two FP32 accumulators occupy 12 KiB at
width 128 and 20 KiB at width 256. Narrower heads use sixteen key rows.

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

Use `--format int4` or `--format int8` for affine MatMulNBits forward products.
These modes also check an independent NumPy decoded-weight product and record
its error. NF4 uses its 16-entry NormalFloat table; affine formats use their own
scale/zero-point encoding.


## 256-wide Qwen3.5 block

A complete 4032-token Qwen3.5-0.8B FP32 block (eight query heads, two KV heads)
uses the same binary for each of these paired forward/backward measurements.
Each isolated worker performs ten warmups and 100 samples, with ORT graph
optimization and CPU fallback disabled.

| Pullback | Pair median / P95 | MLX allocator peak | Process peak physical |
|---|---:|---:|---:|
| Portable graph | 329 / 350 ms | 3.97 GiB | 4.30 GiB |
| Tiled, 64 MiB score budget | 376 / 390 ms | 1.74 GiB | 2.43 GiB |
| GEMM, 1 GiB score budget | 260 / 272 ms | 2.54 GiB | 2.95 GiB |

The GEMM route reduces this block's median paired time by 21%. The tiled route
uses the least memory. All five captured block gradients agree at
`rtol=1e-4, atol=1e-6`; maximum absolute differences are below 1.1e-8. The block
uses fixed parameters during measurement. Full-task training adds the other
blocks, loss, accumulation and optimizer execution.

For short forward heads, see [fused attention padding](PADDED_ATTENTION.md).

## Recomputed short-head pullbacks

`pkg.onnax::AttentionVJPRecompute_*` takes Q, K, V, G and Mask. Mask carries a boolean
or additive bias; an unmasked function ignores its placeholder value.
Its portable ONNX body recomputes Y before applying the pullback. `AttentionVJP_*` uses the supplied Y instead.

The native recomputing route supports FP32 attention, rank three or four, boolean and
additive masks, causal attention, equal Q/K/V head widths from 32 through 256
in multiples of eight, and GQA. It pads heads to a supported training width,
preserves the original scale, and invokes MLX's public
VJP transform around fast SDPA. MLX then chooses its fused full-attention
backward kernel. Short queries are padded to 16 rows and cropped so training
uses the full-attention kernel that produces log-sum-exp. Zero padded query
cotangents contribute no K/V gradient. Empty geometry returns zero gradients.

MLX owns the callback payload and retained stream. RAII owns every temporary
array and closure, and the callback catches Rust unwinds at the C boundary.
Shape-keyed compilation specializes the live geometry while data inputs remain
live on every invocation.

A matched M1 Max FP32 UNet LoRA run uses one 512-pixel image, ten warmups and
100 measured steps, joint scalar-objective execution and external AdamW. The
same native binary measures 465.21/472.09 ms median/P95 with saved-output
pullbacks and 372.10/381.60 ms with six recomputed pullbacks. MLX peak allocation
is 3.846/3.978 GB; whole-process lifetime physical peak is 5.209/5.113 GB.
ORT graph optimization and CPU fallback are disabled. Prediction, loss and all
72 adapter gradients pass the existing norm/pointwise gates. A 20-update
real-image run aligns all gradients and updates with Diffusers MPS, preserves
frozen weight hashes, and reloads its exported predictions bitwise.


Causal attention follows ONNX's upper-left alignment. The built-in causal mode
handles equal query/key extents without an external mask; other cases construct
an explicit upper-left mask. Singleton dimensions and short key tails retain
their broadcast/padding semantics. Changing mask data remains live in compiled
specializations.

Fully masked rows receive a finite softmax input and their output is selected
to zero. Their upstream gradients into SDPA are zero, keeping additive-mask
backward kernels free of `0 * NaN`. This matches the portable function's zero
row contribution. FP32 heads above 128 pad to 256: the 192-wide save-LSE forward
kernel asks for 40,448 bytes of threadgroup memory on M1 Max, above its 32,768-byte
limit. The 256-wide kernel uses its split-head layout.

The Qwen3.5 4032-token block qualification uses causal attention, additive masks,
eight query heads, two KV heads and head width 256. Paired median/P95 is
217.67/222.03 ms, versus 218.28/221.27 ms for the same-binary saved-output GEMM
control with a 1 GiB score budget. All five captured gradients pass the existing
gates, with maximum absolute error 1.40e-9. These measurements cover the block's
forward/backward calls; optimizer and whole-model execution are separate phases.

## Key-tile workspace and throughput

Eight key rows halve the dK/dV workspace for 128–192-wide heads. On an M1 Max,
isolated FP32 causal GQA pullbacks with 16 query heads and 8 KV heads measure:

| Sequence / head width | Previous tile median | Eight-row tile median | Throughput ratio |
| --- | ---: | ---: | ---: |
| 4096 / 128 | 190.51 ms | 152.51 ms | 1.249× |
| 1024 / 160 | 18.63 ms | 14.93 ms | 1.248× |
| 1024 / 192 | 21.09 ms | 17.26 ms | 1.222× |

Each case has three warmups and ten measured synchronous Session.run calls,
including array exchange. Both tiles match independent PyTorch SDPA gradients;
their dQ, dK and dV arrays are bitwise equal. The 4096-token process peak remains
about 0.84 GiB.

A separate Qwen3-0.6B pilot uses 4096 tokens, NF4 frozen decoder weights, FP32
adapters and AdamW, four-layer checkpoints, a compiled 128-token chunked head
and eager decoder sessions. After two warmups, three complete training steps
measure 10.894 s versus 9.757 s median. The first two updates' 448 adapter gradient
and parameter arrays, and all five losses, are bitwise equal.

[Raw samples and provenance](benchmarks/attention-tile-occupancy-m1max.json)
include the hardware, versions, source/library hashes and original driver
snapshots. The drivers run from an onnax source checkout with its example
dependencies installed.
