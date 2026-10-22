# onnxruntime-ep-mlx

[![PyPI Version](https://img.shields.io/pypi/v/onnxruntime-ep-mlx)](https://pypi.org/project/onnxruntime-ep-mlx/)
[![Rust crate](https://img.shields.io/crates/v/onnxruntime-ep-mlx)](https://crates.io/crates/onnxruntime-ep-mlx)

MLX-native ONNX Runtime execution provider for Apple Silicon. It is an
out-of-tree plugin EP for ONNX Runtime 1.31 or newer and requires no ONNX Runtime fork.

The wheel bundles the MLX runtime. Supported parts of your model run on the GPU;
unsupported operators stay on ONNX Runtime's CPU provider, where a matching
CPU kernel is available. Coverage depends on operator attributes, dtypes, and shapes.

See the [complete README](https://github.com/onnxruntime/onnxruntime-ep-mlx#readme)
for requirements, limitations, and links to operator coverage, benchmarks,
diagnostics, source builds, and native integration.

## Install

```sh
python -m pip install -U onnxruntime-ep-mlx
```

```python
import onnxruntime as ort
import onnxruntime_ep_mlx

onnxruntime_ep_mlx.register_execution_provider_library()
session = ort.InferenceSession(
    "model.onnx",
    providers=["MLXExecutionProvider", "CPUExecutionProvider"],
)
feeds = {"input_name": input_array}
outputs = session.run(None, feeds)
```

Replace `model.onnx`, `input_name`, and `input_array` with your model and NumPy
inputs. Use `session.get_inputs()` to inspect input names, shapes, and types.

The package requires macOS 14+ on Apple Silicon, Python 3.11+, and ONNX Runtime 1.31 or newer.
The `0.31.x` package series targets ABI 31 and is built with ONNX Runtime 1.31 headers.
Use one session per thread, not concurrent runs on a shared session. Performance
is workload-dependent and floating-point results are not bit-identical to CPU.
