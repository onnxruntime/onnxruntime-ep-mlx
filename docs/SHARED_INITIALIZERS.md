# Shared immutable initializers

Sessions running on the same evaluator thread share identical immutable initializer
arrays of at least 4 KiB. The pool identifies weights by SHA-256 of their exact
bytes, ONNX dtype and shape. Parameter names and ORT storage addresses can differ.
Overridable graph inputs continue to read their live feeds.

Compile-time host buffers are interned across sessions using weak references.
Each shared MLX array retains that owned host buffer through a managed-payload
callback, independently of ORT. Plans retain their own C
handles and a strong storage owner; a thread-local index holds weak references.
Closing the originating session leaves other sessions valid. Closing the final
owner releases active storage, and expired index entries are removed on insertion.
The MLX allocator may retain freed storage in its separately configured cache.

The pool follows the runtime's existing evaluator-thread ownership. Sessions first
run on different threads use independent pools. Atomic owners permit ORT to release
a session on another thread; the index never transfers an MLX handle across run
threads. Published arrays are evaluated immutable leaves, and retained ownership
protects their storage from donation while plans reuse it.

Set `ONNXRUNTIME_EP_MLX_NO_SHARED_INITIALIZERS=1` before execution to use the
original per-plan initializer wrappers. The default is sharing enabled. Compilation hashes each large initializer once; repeated runs use the
plan's cached handle. Small constants, live inputs and derived layout/quantization
caches retain their existing paths.

## Validation

The ONNX IR tests cover FP32, FP16, INT32, INT8 and UINT8 weights; renamed weights;
different contents, shapes and dtypes; overridable inputs; release of the original
and final sessions; foreign-thread teardown; concurrent independent run threads;
and compiled execution. The disable switch provides a paired memory control.
