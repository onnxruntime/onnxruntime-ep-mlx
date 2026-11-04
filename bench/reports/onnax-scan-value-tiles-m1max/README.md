# Gated-delta reverse value tiles on M1 Max

The reverse Scan kernel now carries eight value columns per threadgroup.
For a value width of 128, this halves the number of groups and the temporary
query/key/decay/beta partial buffers. One internal constant controls Metal
storage, loops, indexing and output geometry. State and replay kernels keep
four columns per group.

## Fixed-input backward block

This pair executes the saved Qwen3.5-0.8B layer-8 linear-attention backward
block with a 4032-token input, NF4 frozen projections, BF16 projection compute
and FP32 recurrent state and adapter gradients. Both processes load the same
symbolically simplified ONNX graph and captured input arrays. ORT graph
optimization and MLX compilation are disabled; the MLX cache limit is 512 MiB.
Each process runs ten warmups and 100 measured calls. `caffeinate -i` covers
the serial measurement queue.

| Reverse value columns | Block median | Block P95 | MLX allocator peak |
| --- | ---: | ---: | ---: |
| Four | 292.69 ms | 299.82 ms | 3,738,491,876 bytes |
| Eight | 271.05 ms | 275.91 ms | 2,898,779,108 bytes |

Block throughput improves by 8.0%, and the MLX allocator peak falls by 22.5%.
The timing includes runtime execution and output transfer for this backward
block. The allocator peak includes setup and all calls in that process.
Every call checks all outputs against the saved reference at
`rtol=1e-4, atol=1e-6`; the largest absolute difference is 6.98e-10.

`baseline.json` and `candidate.json` retain all call timings, output checks,
model/library digests and allocator snapshots. The execution reports retain
whole-process memory observations. `environment.json` identifies the source
and input-manifest digests; copied drivers record the exact commands and
captured-array paths used on this machine.

## Numerical coverage

The native Scan suite passes 56 tests with ORT graph optimization and CPU
fallback disabled for MLX execution. Coverage includes both scan directions,
key widths 16–128, value widths around the eight-column boundary, a partial
17-column tile, zero cotangents, decay extremes, changing inputs, compilation
and structurally different Scan bodies.

## Whole-model gradient check

The 4032-token decoder also runs in separate original/candidate processes,
with the same 25 simplified backward graphs, frozen forward/head graphs,
inputs and adapter parameters. All 60 adapter gradients and the token loss
match at `rtol=1e-4, atol=1e-6`. `full4032/` records the per-gradient errors,
input/adapter digests, graph overrides and process guards. Each process runs
one warmup and one checked call; the 100-sample timing table above measures
the single backward block.
