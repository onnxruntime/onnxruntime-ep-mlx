# Integration and decoder tuning

For the Python install and basic session example, see the [README](../README.md).

## ONNX Runtime compatibility

Package versions follow `0.<ORT_API_VERSION>.<patch>`. The minor version identifies
the target C ABI, not the ONNX model opset.

| Package series | Minimum ONNX Runtime | Target `ORT_API_VERSION` |
|---|---|---|
| `0.31.x` | 1.31 | 31 |
| `0.29.x` (earlier releases) | 1.29 | 29 |

Version `0.31.0` targets the C ABI from
[ONNX Runtime 1.31.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0).
CI and release builds use 1.31 headers and runtime. Registration rejects runtimes
that cannot provide API 31. Newer runtimes must retain that API and expose the
plugin registration interface.

## Python helpers

`onnxruntime_ep_mlx` exposes `library_path()`, `ep_name()`, `version()`, and
`append_to_session_options(so)`. The latter requires an ONNX Runtime build with
`SessionOptions.add_provider`; the explicit `providers=[...]` pattern in the
README is preferred.

## C / C++ and other native hosts

Load `libonnxruntime_mlx_ep.dylib` through ORT's plugin-EP API:

1. Register the dylib with `OrtApi::RegisterExecutionProviderLibrary`.
2. Find its `OrtEpDevice` through `OrtApi::GetEpDevices`.
3. Pass the device to `OrtApi::SessionOptionsAppendExecutionProvider_V2`.

The device's `EpDevice_EpName` is `MLXExecutionProvider`. The V2 API takes
devices, not provider-name strings. Python installations can locate the bundled
dylib with `onnxruntime_ep_mlx.library_path()`; for a standalone build, see
[building](BUILDING.md).

For Rust hosts using **onnx-genai**, set `ONNX_GENAI_EP=metal` and
`ONNX_GENAI_METAL_EP_LIB=/abs/path/libonnxruntime_mlx_ep.dylib`.

## Large-decoder partition metadata

The EP infers residual layer boundaries from graph topology for decoders of
about 24 layers or larger. Exporters can make the boundaries explicit with the
ONNX custom metadata key `onnxruntime_ep_mlx.layer_boundary_outputs`. Its value
is a JSON array of residual output tensor names, one per transformer layer:

```python
import json
import onnx

model = onnx.load("decoder/model.onnx", load_external_data=False)
entry = model.metadata_props.add()
entry.key = "onnxruntime_ep_mlx.layer_boundary_outputs"
entry.value = json.dumps(["layer.0.output", "layer.1.output", "layer.2.output"])
onnx.save(model, "decoder/model.onnx")
```

Metadata takes precedence over inference and does not rely on node names.
The EP chooses groups of 4-8 layers, targeting about seven partitions.
Override the group size with `ONNXRUNTIME_EP_MLX_LAYER_PARTITIONS=<layers>`;
use `0` or `off` to disable partitioning.

## Compilation and precision

Supported ONNX regions become MLX graphs with reusable compiled closures;
unsupported regions remain on ORT CPU. Dynamic shapes use shape-keyed
compilation where required. Eligible autoregressive decode uses shapeless
capture so growing KV length does not retrace. See
[compiled capture](COMPILED_CAPTURE.md) for eligibility and invariants, and the
[operator guide](OP_ARCHITECTURE.md) for control-flow restrictions.

Eligible BF16 INT4/INT8 prefill matmuls (block size 32/64/128, at least 32 rows)
use stock MLX's FP16 compute path by default. Set
`ONNXRUNTIME_EP_MLX_BF16_QMM_FP16=0` to disable it.

Use [tracing](../examples/traces/README.md) to diagnose fallback, fragmentation,
and compilation. Outputs are not bit-identical to ORT CPU; validate accuracy
and latency for your model, including long generations.
