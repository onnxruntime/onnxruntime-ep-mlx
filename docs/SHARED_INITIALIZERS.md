# Shared immutable initializers

Sessions share identical immutable weights automatically. On the FP32
Qwen3.5-0.8B workload below, this reduces whole-process peak physical footprint
from 11.10 to 8.16 GiB while preserving adapter gradients and the final training state.

## Ownership and configuration

Compile-time host buffers of at least 4 KiB are interned by SHA-256, ONNX dtype
and shape. Names and ORT storage addresses can differ. Overridable graph inputs
continue to read their live feeds.

The host-byte pool is process-wide. MLX array pools follow the evaluator's run
thread: sessions on that thread reuse a materialized leaf and retain private C
handles. Sessions on different threads share host bytes through independent array
pools. Atomic owners permit ORT to release a session on another thread.

Each entry owns its host buffer explicitly, preserving the address used as its
identity throughout the entry's lifetime. A managed-payload callback also retains
host storage for MLX's no-copy buffer. This covers the copy fallback, which can
release that callback immediately after copying. Retained leaf ownership protects
weights from donation as plans reuse them.

Both indexes hold weak references. Closing the original session leaves other
owners valid; closing the last owner releases active storage. Expired keys are
removed on insertion. Freed storage follows MLX's allocator cache policy.

Sharing is enabled by default. Set
`ONNXRUNTIME_EP_MLX_NO_SHARED_INITIALIZERS=1` before execution for the per-session
control. Compilation hashes large initializers using ARM SHA-256 dispatch;
subsequent runs use each plan's cached handles. Small constants, live inputs and
derived layout/quantization caches keep their existing paths.

## Full-task timing and memory

The pinned model is `Qwen/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`. The M1 Max measurement uses FP32,
batch one, 512 tokens and fixed trained rank-four adapters. Each variant runs in
a fresh process with ten warmups and 100 timed calls, using the same inputs,
parameters and benchmark source. ORT graph optimization, CPU fallback and MLX
compile are disabled; the allocator cache limit is 512 MiB.

One call includes the ONNX forward, external shifted-token mean loss, ONNX VJP
and runtime host/device exchange. The benchmark keeps parameters fixed. External
optimizer updates are covered by the training qualification below.

| Metric | Per-session storage | Shared storage, ARM hashing |
|---|---:|---:|
| Warmed call median | 3.198 s | 3.104 s |
| Warmed call P95 | 3.240 s | 3.130 s |
| Whole-process peak physical footprint | 11.10 GiB | 8.16 GiB |
| Bundle load and session creation, one call | 10.911 s | 13.724 s |
| Session creation within that load | 3.381 s | 6.037 s |

Whole-process footprint includes imports, bundle verification, construction,
runtime allocations and CPU arrays. Session timing is recorded
separately from warmed execution. All 100 losses and all 60 adapter gradients
match the per-session control byte for byte, and parameters remain unchanged.
The [final report](benchmarks/shared-initializers-owner-timing-m1max.json) records
all samples, gradient hashes, model/input identities, source commits, library
hashes and successful process guards. Its control is preserved in the
[paired report](benchmarks/shared-initializers-timing-m1max.json).

The [allocator probe](benchmarks/shared-initializers-m1max.json) explains the
storage reduction: end-of-task active memory falls from 6.68 to 3.81 GiB, and
active peak from 7.53 to 4.67 GiB. Active memory returns to zero after closing all
sessions. That probe records one complete forward/backward and allocator
observations after each of its 51 session calls.

ARM dispatch reduces a 256 MiB SHA-256 sample from 0.752 to 0.122 seconds in the
[independent hashing test](benchmarks/hash-dispatch-m1max.json), with exact Python
hashlib agreement. The earlier software-hash session creation took 22.712 seconds;
the final build takes 6.037 seconds. Raw reports retain both configurations.

## Training and recovery

The final build completes 30 Qwen3.5-0.8B SFT updates with TRL, sequence length
64, microbatch one and gradient accumulation four. FP32 adapters use external
AdamW; the frozen base remains in ONNX.

| Loss | Before training | After 30 updates |
|---|---:|---:|
| Fixed training set | 5.223939 | 0.302991 |
| Validation set | 6.019480 | 5.621203 |

A fresh process resumes checkpoint 15 and reproduces checkpoint 30 exactly,
including all 61 state tensors, AdamW and scheduler state, and PEFT adapters.
Those final states also match the per-session-storage control byte for byte.
Both runs independently export their ONNX bundles; 53 models per export pass the
full ONNX checker. Fresh loaded execution reproduces logits, loss and all 60
adapter gradients exactly.

Freeze verification compares 3,084 named tensors by dtype, shape and raw bytes
across exported graphs, functions and tensor attributes. Their shared external
files can pack tensors in different orders. The
[training report](benchmarks/shared-initializers-owner-training-m1max.json)
includes successful guard records, loss endpoints, output hashes and executable
verification source; the [canonical tensor manifest](benchmarks/shared-initializers-arm-frozen-tensors.json)
contains every named tensor record.

The ONNX IR regression tests cover FP32, FP16, INT32, INT8 and UINT8 payloads;
renamed weights; different contents/shapes/dtypes; overridable inputs; original
and final session release; copied-buffer identity; foreign-thread teardown;
concurrent run threads; and compiled execution. Qwen3, Phi-3 and Qwen3.5 also
pass independent forward/gradient, multi-step-update and PEFT audits.
