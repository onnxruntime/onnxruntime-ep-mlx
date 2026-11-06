# MLX memory budgets

## Free-cache budget

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

## GPU-residency budget

On macOS 15 or newer, MLX can keep Metal allocations resident instead of leaving
their residency entirely to the system. MLX defaults to a wired limit of `0`.
Large-model latency can become highly variable without residency, even when
active arrays fit within the device's recommended working set.

Set `ep.mlx.wired_limit_bytes` before creating a session, or use
`ONNXRUNTIME_EP_MLX_WIRED_LIMIT_BYTES` for applications that create their own
sessions. Values are nonnegative decimal byte counts, and the session setting
overrides the environment. `0` disables wiring; omission preserves the existing
MLX limit. The limit is a maximum, not a request to allocate that many bytes.
Invalid syntax fails session creation before either memory budget is changed.
A device-rejected budget also fails explicitly.

Choose a budget that covers the measured model/state/temporary working set
without exceeding the device's `max_recommended_working_set_size`, and leave
headroom for the OS and other applications. A free-cache budget limits unused
buffers; it does not replace a residency budget. No residency budget is enabled
automatically by this EP.

Like the cache setting, the wired limit is process-wide and is not restored on
session teardown. Configure a consistent limit before inference, and do not
change it while asynchronous work is in flight in any session sharing that MLX
runtime. MLX-LM likewise
[sets a recommended wired limit](https://github.com/ml-explore/mlx-lm/blob/736ab0d8c1d0edc79c3d798044fe6904e2fa23be/mlx_lm/utils.py#L79-L89)
for generation; see the
[MLX wired-limit API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html).
