# Shape-aware broadcast adjoints

`pkg.onnax::SumToShape(Gradient, TargetShape)` reduces a gradient to target
dimensions supplied as a one-dimensional INT64 vector. Dimensions are
nonnegative int32 values; an empty vector selects a scalar output. The output
keeps the gradient's dtype.

Align the target dimensions with the trailing gradient dimensions. Sum leading
dimensions and aligned dimensions equal to one, then reshape to the target.
Each aligned dimension must equal the gradient dimension or be one. Empty
gradients produce zeros of the target shape, including empty outputs.

The onnax deployment pass emits `Shape(Reference)` and a standard ONNX function
body with dynamic axis discovery. Saved-value planning can retain the dimension
vector instead of the reference activation. MLX executes FP32, FP16 and BF16
gradients by selecting reduction axes from the live gradient shape and target
values.

Target vectors are host shape parameters. Their values participate in compiled
specialization alongside tensor shapes: changing dimensions selects a new
reduction even when the vector length and gradient shape stay unchanged.
Shape-derived vectors reuse the host shape-program path. Tests cover eager and
compiled execution, changing axes, zeros, scalars, invalid dimensions and reuse
of a cached valid reduction after an error. NumPy provides the independent oracle.

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
