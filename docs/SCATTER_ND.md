# Additive ScatterND

`ScatterND` supports `reduction="add"` from ONNX opset 16. It starts from the
input data and adds each update to its indexed slice. Duplicate tuples contribute
all their updates; negative coordinates count from the corresponding dimension's
end. See the [ONNX operator definition](https://onnx.ai/onnx/operators/onnx__ScatterND.html).

The additive path supports FP32, FP16 and BF16 payloads, symbolic data/update
extents, scalar and slice updates, empty update lists and empty trailing slices.
The tuple width is fixed by the graph. Shape relationships and every live index
are checked before dispatch to MLX's scatter-add operation. Invalid inputs return
an EP error, and the session can be reused afterward.

Index validation executes eagerly so changing tuples at identical shapes are
checked on every call. The numerical payload uses native MLX scatter-add. The
existing `reduction="none"` path retains its shape-keyed execution policy.

Run the numerical and geometry regressions with the development library:

```bash
ONNXRUNTIME_MLX_EP_LIB=/path/to/libonnxruntime_mlx_ep.dylib \
  python -m pytest tests/ops/test_scatternd_add.py tests/ops/test_scatter_symbolic.py tests/ops/test_exp_shape.py -q
```

The focused tests disable CPU EP fallback and use an independent NumPy update
loop. BF16 payloads use FP32 inputs/outputs with explicit casts at the runtime
boundary. Duplicate/negative tuples, multiple shapes per session and recovery
following out-of-bounds values are covered.
