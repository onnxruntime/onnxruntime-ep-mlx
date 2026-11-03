# Shape-aware broadcast adjoints

`pkg.onnax::SumToShape_v1` reduces a gradient to the shape of a reference tensor.
Its inputs are `Gradient` and `Reference`; its single output has the gradient's
dtype and the reference's shape. Reference values are unused.

Align the reference dimensions with the trailing gradient dimensions. Sum the
leading dimensions and every aligned reference dimension equal to one, then
reshape to the reference shape. Each aligned dimension must either equal the
gradient dimension or be one. An empty gradient produces zeros of the target
shape, including empty outputs. Scalars follow the same rule.

The onnax deployment pass includes a standard ONNX function body with dynamic
axis discovery. MLX executes FP32, FP16 and BF16 gradients directly: derive axes
from both live shapes, reduce and reshape. This shape-keyed implementation avoids
materializing axis tensors or evaluating their values on the host. Both input
shapes participate in compiled specialization; changing a singleton dimension
selects the corresponding reduction.

Tests cover eager and compiled execution, changing reduction axes within one
session, leading axes, scalars, empty tensors, ignored reference values and
recovery after an invalid shape. NumPy supplies the independent reduction oracle.

A matched M1 Max UNet LoRA run uses FP32, one 512-pixel image, 10 warmup
steps and 100 measured steps. With the same native binary and a 512 MiB cache,
compiled step median/P95 changes from 0.959/0.984 s to 0.651/0.661 s. Whole-process
peak physical footprint changes from 7.15 GB to 5.37 GB; MLX peak allocation
changes from 4.46 GB to 4.31 GB. Prediction, loss and all 72 adapter gradients
match bitwise. ORT graph optimization and CPU fallback are disabled.

The one-warmup/three-step diagnostic reduces subgraph execution spans from 1172
to 8 and recorded materialized output bytes from 4.27 GB to 3.72 MB. The full
measurement includes setup, warmup and execution in its process peak; diagnostic
counters include setup events. Qwen3, Qwen3.5 and Phi3 also pass native compiled
adapter-gradient, three-step update and re-export comparisons against PyTorch.
