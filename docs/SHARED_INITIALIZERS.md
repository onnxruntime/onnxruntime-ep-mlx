# Shared immutable initializers

Sessions running on the same evaluator thread share identical immutable initializer
arrays of at least 4 KiB. The pool identifies weights by SHA-256 of their exact
bytes, ONNX dtype and shape. Parameter names and ORT storage addresses can differ.
Overridable graph inputs continue to read their live feeds.

Compile-time host buffers are interned across sessions using weak references.
Each shared MLX array retains that owned host buffer through a managed-payload
callback, independently of ORT. Each cache entry also retains the host buffer
explicitly, keeping its identity alive when MLX chooses a copied buffer. Plans
retain their own C handles and a strong entry owner; the index holds weak references.
Closing the originating session leaves other sessions valid. Closing the final
owner releases active storage, and expired index entries are removed on insertion.
The MLX allocator may retain freed storage in its separately configured cache.

The host-byte pool is process-wide. MLX array pools follow the runtime's evaluator
thread ownership: sessions first run on different threads use independent array
pools while sharing immutable host bytes. Atomic owners permit ORT to release
a session on another thread; the index never transfers an MLX handle across run
threads. Published arrays are evaluated immutable leaves, and retained ownership
protects their storage from donation while plans reuse it.

Set `ONNXRUNTIME_EP_MLX_NO_SHARED_INITIALIZERS=1` before execution to use the
original per-plan initializer wrappers. Sharing is enabled by default. Compilation
hashes large initializers; repeated runs use the plan's cached handle. Small constants, live inputs and derived layout/quantization
caches retain their existing paths.

## Validation

The ONNX IR tests cover FP32, FP16, INT32, INT8 and UINT8 weights; renamed weights;
different contents, shapes and dtypes; overridable inputs; release of the original
and final sessions; foreign-thread teardown; concurrent independent run threads;
and compiled execution. The disable switch provides a paired memory control.

## Qwen3.5-0.8B measurement

A paired full forward/loss/backward probe uses the trained FP32 export at revision
`2fc06364715b967f1860aea9cf38778875588b17`, batch one and 512 tokens on M1 Max.
Both fresh processes use the same library, inputs and adapters, ORT optimization
and CPU fallback disabled, MLX compile disabled, and a 512 MiB allocator cache limit.

| Metric | Sharing off | Sharing on |
|---|---:|---:|
| End-of-task active allocation | 6.68 GiB | 3.81 GiB |
| MLX active allocation peak | 7.53 GiB | 4.67 GiB |
| Whole-process physical footprint peak | 10.55 GiB | 7.66 GiB |
| Active allocation after closing all sessions | 0 | 0 |

All 51 session calls produce byte-identical outputs, including logits, token loss
and all adapter gradients. The probe includes construction, one task backward,
output hashing and cleanup; warmed timing is measured separately. Compilation
owns and hashes host weights once per plan. The [raw report](benchmarks/shared-initializers-m1max.json)
includes each allocator observation, output hashes, guarded-process results and
reproducible worker source.

## ARM hashing

The default build enables SHA-256 hardware dispatch on ARM. In the independent
M1 Max [hashing test](benchmarks/hash-dispatch-m1max.json), a 256 MiB sample takes
0.752 seconds through the software backend and 0.122 seconds with ARM dispatch.
Both outputs match Python hashlib exactly. The complete-model
[ARM qualification](benchmarks/shared-initializers-arm-hash-m1max.json) preserves
all 51 session output hashes and releases active storage after closing the model.

The [paired timing report](benchmarks/shared-initializers-timing-m1max.json)
records the earlier software-hash comparison: 10 warmups and 100 measured full
forward/loss/backward calls per process, fixed parameters, all 60 gradients and
losses byte-identical. Median execution falls from 3.198 to 3.111 seconds and
physical peak from 11.10 to 8.25 GiB. Its construction times identify the hashing
cost addressed by ARM dispatch. Final-build initialization and execution are
qualified separately.
