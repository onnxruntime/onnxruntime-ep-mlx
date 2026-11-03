"""Shape-keyed broadcast adjoints against independent NumPy reductions."""

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest


def _function(shape_vector=False):
    gradient, reference = ir.val("Gradient"), ir.val("Reference")
    t = ir.tape.Tape()

    def op(name, *inputs, **attrs):
        return t.op(name, inputs, attrs)

    def constant(value):
        return op("Constant", value=ir.tensor(np.asarray(value, np.int64)))

    shape = reference if shape_vector else op("Shape", reference)
    gs = op("Shape", gradient)
    leading = op("Sub", op("Size", gs), op("Size", shape))
    ones = op(
        "ConstantOfShape",
        op("Unsqueeze", leading, constant([0])),
        value=ir.tensor(np.array([1], np.int64)),
    )
    padded = op("Concat", ones, shape, axis=0)
    axes = op("Squeeze", op("NonZero", op("Equal", padded, constant(1))), constant([0]))
    axes = op("Concat", axes, constant([-1]), axis=0)
    expanded = op("Unsqueeze", gradient, constant([-1]))
    reduced = op("ReduceSum", expanded, axes, keepdims=1)
    result = op("Reshape", reduced, shape, allowzero=1)
    return ir.Function(
        "pkg.onnax",
        "SumToShape_v2" if shape_vector else "SumToShape_v1",
        graph=ir.Graph(
            [gradient, reference], [result], nodes=t.nodes, opset_imports={"": 18}
        ),
        attributes=[],
    )


def _bf16(values):
    bits = values.astype(np.float32).view(np.uint32)
    return (
        (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    ).view(np.float32)


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
@pytest.mark.parametrize("shape_vector", [False, True])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("ranks", [(2, 2), (2, 1), (2, 0), (0, 0)])
def test_live_shapes_and_values(dtype, eager, ranks, shape_vector, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    boundary = ir.DataType.FLOAT if dtype == ir.DataType.BFLOAT16 else dtype
    g = ir.val("g", dtype=boundary, shape=[None] * ranks[0])
    r = ir.val(
        "r",
        dtype=ir.DataType.INT64 if shape_vector else boundary,
        shape=[ranks[1]] if shape_vector else [None] * ranks[1],
    )
    t = ir.tape.Tape()
    inputs = (
        [
            t.op("Cast", [g], {"to": int(dtype)}),
            r if shape_vector else t.op("Cast", [r], {"to": int(dtype)}),
        ]
        if dtype == ir.DataType.BFLOAT16
        else [g, r]
    )
    y = t.op(
        "SumToShape_v2" if shape_vector else "SumToShape_v1",
        inputs,
        domain="pkg.onnax",
        version=1,
    )
    if dtype == ir.DataType.BFLOAT16:
        y = t.op("Cast", [y], {"to": int(boundary)})
    y.name, y.dtype, y.shape = "y", boundary, ir.Shape([None] * ranks[1])
    f = _function(shape_vector)
    model = ir.Model(
        ir.Graph(
            [g, r],
            [y],
            nodes=t.nodes,
            opset_imports={"": 18, "pkg.onnax": 1},
            name="broadcast_adjoint",
        ),
        ir_version=10,
        functions=[f],
    )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    runtime = ort.InferenceSession(
        ir.to_proto(model).SerializeToString(),
        options,
        providers=["MLXExecutionProvider"],
    )
    runtime.disable_fallback()
    shapes = {
        (2, 2): [
            ((2, 3), (2, 1)),
            ((2, 3), (1, 3)),
            ((0, 3), (1, 3)),
            ((2, 0), (2, 0)),
            ((2, 3), (2, 3)),
            ((2, 3), (2, 1)),
        ],
        (2, 1): [((2, 3), (3,)), ((0, 3), (3,)), ((4, 1), (1,)), ((2, 3), (3,))],
        (2, 0): [((2, 3), ()), ((2, 0), ()), ((1, 1), ()), ((2, 3), ())],
        (0, 0): [((), ()), ((), ())],
    }[ranks]
    rng = np.random.default_rng(27)
    for source, target in shapes:
        data = np.asarray(rng.normal(0, 0.1, source), dtype=boundary.numpy())
        values = (
            _bf16(data) if dtype == ir.DataType.BFLOAT16 else data.astype(np.float32)
        )
        padded = (1,) * (len(source) - len(target)) + target
        axes = tuple(i for i, dimension in enumerate(padded) if dimension == 1)
        expected = values.sum(axis=axes, keepdims=True).reshape(target)
        expected = (
            _bf16(expected)
            if dtype == ir.DataType.BFLOAT16
            else expected.astype(boundary.numpy())
        )
        actual = runtime.run(
            None,
            {
                "g": data,
                "r": np.asarray(target, np.int64)
                if shape_vector
                else np.full(target, np.nan, dtype=boundary.numpy()),
            },
        )[0]
        assert actual.shape == target
        assert actual.dtype == boundary.numpy()
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-4 if dtype == ir.DataType.FLOAT else 0.01,
            atol=1e-6 if dtype == ir.DataType.FLOAT else 1e-4,
        )

        if (
            dtype == ir.DataType.FLOAT
            and not eager
            and ranks == (2, 2)
            and source == (2, 3)
        ):
            with pytest.raises(Exception, match="incompatible live broadcast extents"):
                runtime.run(
                    None,
                    {
                        "g": data,
                        "r": np.array([2, 2], np.int64)
                        if shape_vector
                        else np.ones((2, 2), np.float32),
                    },
                )
            if shape_vector:
                for invalid in ([-1, 3], [2**40, 1]):
                    with pytest.raises(Exception, match="nonnegative int32"):
                        runtime.run(
                            None, {"g": data, "r": np.asarray(invalid, np.int64)}
                        )
            # A rejected shape must not poison the previously cached specialization.
            np.testing.assert_allclose(
                runtime.run(
                    None,
                    {
                        "g": data,
                        "r": np.asarray(target, np.int64)
                        if shape_vector
                        else np.ones(target, np.float32),
                    },
                )[0],
                expected,
                rtol=1e-4,
                atol=1e-6,
            )
