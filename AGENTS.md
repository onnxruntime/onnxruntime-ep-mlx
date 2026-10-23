Be succinct, robust and elegant.

## Operator scope and semantics

- Operator audits and implementation are inference-only. Optimizer, autodiff, and
  training operators (such as Adam, Adagrad, Momentum, and Gradient), as well as
  training-only modes, are outside the required scope. Inference RNG, control
  flow, and stateful SSM remain in scope.
- Default-domain operators must follow official ONNX definitions. Neither
  unofficial default-domain extensions nor officially deprecated ONNX operators
  require new support. An older opset alone does not imply deprecation; use the
  official ONNX deprecation status. Preserve existing behavior and regression
  coverage.
- For `com.microsoft` operators, prefer official ORT CUDA semantics and
  capabilities when CPU behavior differs. CPU rejection, missing kernels, or
  output differences alone must not exclude CUDA-only operators or define the
  contract. Establish the contract from authorized CUDA source/specifications
  and validate MLX against independent references; do not claim CUDA execution
  without actually running it. Prefer simpler, mathematically well-defined
  generalizations (such as scalar/rank-0 inputs) over needless CPU-kernel
  restrictions. Preserve real backend, dtype, safety, and unambiguous semantic
  constraints. When the CPU implementation rejects valid cases, validate them
  against independent references rather than treating CPU rejection as normative.
