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
