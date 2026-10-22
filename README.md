# onnxruntime-ep-mlx

[![PyPI Version](https://img.shields.io/pypi/v/onnxruntime-ep-mlx)](https://pypi.org/project/onnxruntime-ep-mlx/)
[![Crates.io Version](https://img.shields.io/crates/v/onnxruntime-ep-mlx)](https://crates.io/crates/onnxruntime-ep-mlx)

Run ONNX models on Apple Silicon GPUs with [MLX](https://github.com/ml-explore/mlx)
and ONNX Runtime. This execution provider is a plugin: no ONNX Runtime fork is
needed. Supported parts of your model run in MLX; unsupported operators stay on
ONNX Runtime's CPU provider.

## Install

Requires **macOS 14+ on Apple Silicon**, **Python 3.11+**, and **ONNX Runtime 1.31+**.
The wheel bundles the MLX runtime and installs ONNX Runtime as a dependency;
no source-build tools are needed.

```sh
python -m pip install -U onnxruntime-ep-mlx
```

## Use

Register the plugin once per process, before creating your session:

```python
import onnxruntime as ort
import onnxruntime_ep_mlx

onnxruntime_ep_mlx.register_execution_provider_library()
session = ort.InferenceSession(
    "model.onnx",
    providers=["MLXExecutionProvider", "CPUExecutionProvider"],
)

# Supply your model's input names and NumPy arrays.
feeds = {"input_name": input_array}
outputs = session.run(None, feeds)
```

Replace `model.onnx`, `input_name`, and `input_array` with your model and inputs.
Use `session.get_inputs()` to inspect the required input names, shapes, and types.

## Before using it

- **Coverage is conditional.** Operator attributes, dtypes, and shapes determine
  what runs in MLX. See the [operator support guide](docs/OP_ARCHITECTURE.md).
  CPU fallback requires a matching ONNX Runtime CPU kernel.
- **Speed depends on the model.** Compilation adds startup cost; CPU fallback and
  small workloads can outweigh GPU gains. Measure your own workload with the
  [benchmark guide](bench/README.md).
- **Use one session per thread.** Do not run a shared session concurrently from
  multiple threads; MLX evaluation is thread-affine.
- **Results are not bit-identical to CPU.** Floating-point differences can change
  long greedy generations when logits are nearly tied.

## Documentation

| For | Guide |
|---|---|
| Runtime compatibility, native bindings, decoder tuning | [Integration](docs/INTEGRATION.md) |
| Building the plugin or Python wheel | [Build from source](docs/BUILDING.md) |
| Diagnosing fallback and GPU execution | [Tracing and profiling](examples/traces/README.md) |
| Running correctness and conformance tests | [Testing](tests/README.md) |
| Understanding or extending the provider | [Design](docs/DESIGN.md), [operator architecture](docs/OP_ARCHITECTURE.md), [compiled capture](docs/COMPILED_CAPTURE.md) |
