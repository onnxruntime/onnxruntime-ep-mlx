# MLX allocator cache budgets

Variable tensor shapes can leave reusable buffers of many sizes in MLX's free
cache. Set a cache budget when the application needs to limit retained free
buffers. Active arrays and persistent model weights have their own lifetimes.

Configure a session before creating it:

```python
options = ort.SessionOptions()
options.add_session_config_entry("ep.mlx.cache_limit_bytes", "536870912")
session = ort.InferenceSession(
    "model.onnx", options, providers=["MLXExecutionProvider"]
)
```

Or apply the same budget to sessions created by an existing application:

```bash
ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES=536870912 python application.py
```

Values are nonnegative decimal byte counts. `0` disables the free cache.
Session configuration takes precedence over the environment. Omitting both
preserves the application's current MLX setting; invalid values fail session
creation with `INVALID_ARGUMENT`.

The allocator setting is process-wide across EP sessions linked to the same
MLX runtime. Each explicit configuration updates that shared setting, and
session teardown preserves it. Applications using multiple sessions can apply
one consistent budget through the environment.

MLX reclaims excess free-cache memory on the next allocation, so a sampled
cache can briefly exceed its target after buffers are released. Process RSS
also includes active tensors, runtime state and host allocations. See the
[MLX cache-limit API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html).
