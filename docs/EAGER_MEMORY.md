# Eager intermediate array lifetimes

The eager translator builds one lazy MLX graph and evaluates its boundary
outputs synchronously. Keeping an owning array handle for every translated
operation pins each intermediate allocation through that evaluation. Large
lookup decoders can retain many unpacked integer and floating-point matrices.

After translation, the output vector owns references to all boundary outputs
and state writebacks. These roots keep their dependencies alive. The translator
then releases its temporary arena, name lookup and cast cache before evaluation,
so MLX can recycle intermediates after their final consumer. Constant initializer
and repack arrays retain their plan lifetime. Output views and ordered state
writebacks are copied from owning references obtained from the output vector.
Runtime input wrappers remain referenced through evaluation, protecting
caller-owned zero-copy buffers from donation and reuse. Regressions cover
16 KiB-aligned inputs as well as ordinary NumPy allocations.

## Reproduce the allocation comparison

Run each binary in a separate process:

```bash
python tests/ops/test_eager_array_lifetimes.py \
  --library /path/to/libonnxruntime_mlx_ep.dylib --output memory.json
```

The fixture uses ONNX IR to build 48 Add/Tanh layers and a final transpose over
1024 × 1024 FP32 values. Its three outputs include an intermediate checkpoint
and an output view. All outputs are checked against independent NumPy arithmetic.
The command disables ORT optimization and compiled capture, resets the linked
MLX allocator peak after session construction, then runs one execution.

On Apple M1 Max, allocator active-memory peak was 411,041,796 bytes before the
change and 16,777,220 bytes afterward (approximately 392 MiB and 16 MiB). Both
runs used the same graph hash. This metric covers MLX active allocations;
process RSS and cached Metal allocations have separate scopes.

`benchmarks/eager-array-lifetimes-m1max.json` records binary hashes, graph hash,
measurement scope and regression results.
