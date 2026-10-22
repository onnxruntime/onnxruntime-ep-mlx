# onnxruntime-ep-mlx

[![PyPI Version](https://img.shields.io/pypi/v/onnxruntime-ep-mlx)](https://pypi.org/project/onnxruntime-ep-mlx/)
[![Rust crate](https://img.shields.io/crates/v/onnxruntime-ep-mlx)](https://crates.io/crates/onnxruntime-ep-mlx)

MLX-native ONNX Runtime execution provider for Apple Silicon. It is an
out-of-tree plugin EP for ONNX Runtime 1.31 or newer and requires no ONNX Runtime fork.

The wheel bundles MLX 0.32.4 commit `36969c09` and mlx-c commit `57731b00`. It supports encoder, LLM
prefill, and token-at-a-time decode workloads, including quantized matmul,
attention, normalization, convolution, shape operations, and compiled
shape-specialized `If`, `Loop`, and `Scan` control flow.

See the [complete README](https://github.com/onnxruntime/onnxruntime-ep-mlx#readme)
for operator coverage, performance results, diagnostics, build instructions,
and C/C++ integration.

## Install

```sh
pip install -U onnxruntime-ep-mlx
```

```python
import onnxruntime as ort
import onnxruntime_ep_mlx

onnxruntime_ep_mlx.register_execution_provider_library()
session = ort.InferenceSession(
    "model.onnx",
    providers=["MLXExecutionProvider", "CPUExecutionProvider"],
)
outputs = session.run(None, feeds)
```

The package requires macOS 14+ on Apple Silicon, Python 3.11+, and ONNX Runtime 1.31 or newer.
The `0.31.x` package series targets ABI 31 and is built with ONNX Runtime 1.31 headers.
