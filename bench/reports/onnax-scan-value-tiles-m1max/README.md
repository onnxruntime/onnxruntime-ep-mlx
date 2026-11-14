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

## Full decoder measurement

The complete Qwen3.5-0.8B comparison uses a 4032-token input with 4031 shifted
prediction targets. Both independent processes load the same 25 symbolically
simplified backward graphs, frozen forward/head graphs, inputs and adapters.
The [symbolic-adjoint change](https://github.com/justinchuby/onnax/pull/200)
identifies the graph-generation work. Each library runs ten warmups and
100 measured forward/loss/VJP calls with fixed parameters. Both body and head
sessions are eager; ORT optimization is disabled.

| Reverse value columns | Step median | Step P95 | Prediction targets/s | Process peak physical footprint |
| --- | ---: | ---: | ---: | ---: |
| Four | 10.501 s | 10.650 s | 383.9 | 8.572 GB |
| Eight | 10.675 s | 11.832 s | 377.6 | 7.819 GB |

The candidate's whole-process peak is 8.8% lower, including setup and runtime.
Its full-step median is 1.7% higher in this pair, with a broader tail. Forward
and output-head work has medians of 3.359 / 3.564 seconds; body backward has
medians of 7.120 / 7.098 seconds. The raw samples retain this phase variation.

All 100 token losses are equal. All 60 final adapter gradients match at
`rtol=1e-4, atol=1e-6`, with maximum absolute difference 1.54e-8. Input/adapter
digests and loaded graph identities agree, and both runs leave parameters unchanged.
Both process guards complete with zero observation errors and maximum gaps
below 1.05 seconds; `caffeinate -i` covers the measurement.

`full4032-formal/` contains raw samples, phase timings, process observations,
all gradient checks, session overrides and copied drivers. The earlier
single-call whole-model qualification remains in `full4032/`.

## Numerical coverage

The native Scan suite passes 56 tests with ORT graph optimization and CPU
fallback disabled for MLX execution. Coverage includes both scan directions,
key widths 16–128, value widths around the eight-column boundary, a partial
17-column tile, zero cotangents, decay extremes, changing inputs, compilation
and structurally different Scan bodies.
