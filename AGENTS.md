Be succinct, robust and elegant.

## Operator scope and semantics

- Default-domain operators must follow official ONNX definitions. Unofficial
  default-domain extensions need not be implemented; deprecated but officially
  defined ONNX operators are not excluded by this rule.
- For `com.microsoft` operators, prefer simpler, mathematically well-defined
  generalizations (such as scalar/rank-0 inputs) over needless CPU-kernel
  restrictions. Preserve real backend, dtype, safety, and unambiguous semantic
  constraints. When the CPU implementation rejects valid cases, validate them
  against independent references rather than treating CPU rejection as normative.
