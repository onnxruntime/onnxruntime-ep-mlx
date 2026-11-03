# Fused attention for short heads

MLX's full-attention kernel supports a 64-element head, while a head width of 40
uses composed operations that materialize the score matrix. The EP can extend Q,
K, and V with zero columns, execute tiled attention at width 64, and crop the
result to its original width. The scale remains the ONNX scale for the original
head: zero columns leave every dot product unchanged.

The selection uses live geometry. It applies to equal Q/K/V head widths from 32
to 63 and at least 16 million score elements (`B * Hq * Sq * Sk`). This bounds
the padding expansion to 2x and targets score matrices of at least 64 MiB in
FP32. Smaller attention matrices keep their existing execution path. Attention
nodes with eligible or unknown head widths use shape-keyed compilation because
MLX Pad and Slice require concrete output geometry. Each sequence shape gets
its own compiled variant. Other head widths retain their existing compilation policy.

Masks, grouped query heads, output dtype, and the original attention scale pass
through the same SDPA entry point. Diagnostic/composed attention forms keep
using their existing implementation.

## Measurement on M1 Max

The matched FP32 UNet workload uses `segmind/tiny-sd`, 512-pixel images, 72 LoRA
parameters, joint scalar loss/gradient execution, and external AdamW. ORT graph
optimization and CPU fallback are disabled. Each process warms up for 10 steps
and measures 100 complete steps.

| Metric | Original | Padded short heads |
| --- | ---: | ---: |
| Step median | 473.88 ms | 464.24 ms |
| Step P95 | 482.46 ms | 471.66 ms |
| MLX allocator peak | 4.325 GB | 3.846 GB |
| Process lifetime physical peak | 5.404 GB | 4.975 GB |

Whole-process memory includes construction, runtime, captures, and allocator
caches. The cache limit is 512 MiB in both runs. Peak memory and step timing are
measured in isolated processes.

The captured loss and all 72 adapter gradients pass the existing
`rtol=1e-4, atol=1e-6` pointwise and tensor-norm checks. Prediction relative L2
error is `9.75e-7`; 9 of 16,384 values cross the pointwise threshold near zero.
The existing whole-prediction norm check passes. The two SDPA implementations
use different FP32 reduction orders.

The 20-update real-image run uses two microbatches per update and compares loss,
all adapter gradients, and parameter updates with Diffusers on MPS. Training
loss changes from 0.11625 to 0.09504; validation loss changes from 0.11048 to
0.10962. Frozen-weight hashes stay unchanged, and the exported ONNX reload
reproduces its predictions bitwise.

Matched [baseline](benchmarks/padded-attention-baseline-m1max.json),
[candidate](benchmarks/padded-attention-m1max.json),
[tensor comparisons](benchmarks/padded-attention-comparison-m1max.json),
[training](benchmarks/padded-attention-training-m1max.json), and
[provenance](benchmarks/padded-attention-provenance-m1max.json) reports record the
workload, versions, binary hashes, and measurement scope.
