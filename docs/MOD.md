# Mod contracts and precision

The provider implements the resolved ONNX Mod10 and Mod13 schemas with
same-dtype inputs/output and multidirectional (NumPy-style) broadcasting,
including scalars.

| Resolved schema | `fmod=0` (default) | `fmod=1` |
| --- | --- | --- |
| 10 | Signed/unsigned integers, divisor-signed remainder | Integers or floating point, dividend-signed remainder |
| 13 | Signed/unsigned integers, divisor-signed remainder | Floating point only, dividend-signed remainder |

Mapped integer types are 8/16/32/64-bit signed and unsigned. Floating-point
types are FLOAT16, FLOAT, and DOUBLE; schema13 additionally admits BFLOAT16.
The claim predicate declines invalid attribute/dtype combinations, even if an
ORT CPU kernel happens to execute them. `fmod` must be zero or one.

Floating remainder uses MLX remainder on nonnegative magnitudes and restores
the dividend sign. It does not reconstruct a remainder by dividing, rounding,
multiplying, and subtracting: that loses finite remainders when a quotient
rounds or overflows. The primitive chain explicitly uses the MLX CPU stream
with exact DOUBLE inputs, then rounds the result back to the declared dtype.
This avoids Metal's fused remainder precision loss and subnormal-divisor
flushing without introducing a custom kernel or falling back to ORT CPU.
DOUBLE remains DOUBLE throughout. Zero divisors, NaNs, and infinite operands follow the floating
`fmod` contract. Signed-zero inputs are preserved when the result is zero.
Floating Mod therefore trades GPU fusion for correctness; it remains a live
dependency in general compiled replay, rather than a host-computed constant.

Integer Mod checks for zero divisors on every execution and reports an EP
error, including after a successful run of the same shape. It uses the eager
MLX route so this validation cannot be captured once and omitted on replay.
Signed minimum divided by minus one produces a zero remainder without
executing overflowing signed division. Integer arithmetic never passes
through a floating-point representation.

Mod28 expands the legal dtype/attribute combinations. It is deliberately not
registered here: ORT 1.31 supports model opsets only through 26, so declaring a
newer import cannot demonstrate execution of the resolved Mod28 schema.

Authoritative definitions:
[Mod10](https://github.com/onnx/onnx/blob/v1.23.2/onnx/defs/math/old.cc),
[Mod13/Mod28 descriptions](https://github.com/onnx/onnx/blob/v1.23.2/onnx/defs/doc_strings.cc),
and [Mod28 schema](https://github.com/onnx/onnx/blob/v1.23.2/onnx/defs/math/defs.cc).
The strict regression matrix in `tests/ops/test_mod_contracts.py` uses exact
rational arithmetic on represented floating-point inputs and Python integers,
not a floating-point quotient composition or a lossy integer CPU oracle.
