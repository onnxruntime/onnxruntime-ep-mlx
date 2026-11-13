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

## Immutable quantized initializer views

Large immutable initializers are interned by content, dtype, and shape. The
packed uint32 weight, reshaped scale views, and immutable symmetric affine biases
used by `MatMulNBits` also reuse
evaluated MLX leaves across plans on the same run thread. Each plan retains its
own handle and a strong backing-storage owner; initializer names are not
identities, and overridable inputs remain live. GPU outputs may retain this
storage after their originating session closes.

Set `ONNXRUNTIME_EP_MLX_NO_SHARED_QUANT_VIEWS=1` to disable only quantized-view
sharing, or `ONNXRUNTIME_EP_MLX_NO_SHARED_INITIALIZERS=1` to disable all
initializer interning. These controls do not change quantization arithmetic.
Derived biases preserve the existing CPU FP16/BF16/FP32 rounding and are interned
by their resulting bytes, dtype, and shape; runtime scales are never interned.
A Qwen 27B target/draft diagnostic reduced reported MLX active allocations by
about 707 MiB after draft execution. Prefill peak allocations were unchanged;
this is not a measurement of total physical memory or a decode speedup claim.

## Runtime input ownership

CPU runtime inputs of at most 4096 bytes are copied into owned MLX arrays by
default. This removes their borrowed-buffer synchronization requirement when
all boundary outputs are GPU-owned. Public GPU inputs retain their zero-copy
wrapping. Larger CPU inputs still use borrowed wrapping and synchronous
evaluation; CPU outputs and side-effecting input writebacks also retain their
synchronous boundary.

Set `ONNXRUNTIME_EP_MLX_OWN_SMALL_INPUTS=0` to restore borrowed wrapping for
small CPU inputs. The byte limit applies to actual tensor storage, including
packed-input storage, rather than just the element count.

## Paged-prefill workspace

By default, paged prefill keeps the native KV head count in FP32
SDPA for eligible GPU `PagedAttention` calls with more than eight queries.
This avoids explicitly repeating grouped KV heads before attention. Decode,
quantized or latent caches, softcap, and head-sink paths are unchanged.
Causal/window masks and visible cache outputs retain their existing semantics.
Eligible FP16 windowed prefill also gathers only physical pages intersecting
the attention window instead of materializing the full cached prefix.
Set `ONNXRUNTIME_EP_MLX_NO_PAGED_WINDOW_GATHER=1` to restore the full-prefix
gather for diagnostics.

Set `ONNXRUNTIME_EP_MLX_PAGED_PREFILL_GQA=0` to restore explicit KV-head
repetition. The native grouped path passed independent attention/cache checks
and six exact-token Qwen 16K MTP requests (three warmups, three measured).
The measured median TTFT was 143.913 seconds and aggregate decode was
10.945 tokens/second on M1 Max 32 GiB with 1024-token chunks and explicit
phase-specific residency budgets. This is not a same-binary A/B speedup claim.
It reduces one source of temporary storage, not the model's total working set
or the need for a device-appropriate residency budget.
