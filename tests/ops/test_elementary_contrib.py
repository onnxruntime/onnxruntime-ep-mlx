"""Source fixtures for elementary contrib v1; strict MLX placement, typed live oracles."""

import json
import math
from pathlib import Path

import ml_dtypes
import numpy as np
from onnx import TensorProto as T, helper, numpy_helper
import onnxruntime as ort
import pytest


FLOATS = [T.FLOAT16, T.BFLOAT16, T.FLOAT, T.DOUBLE]
SCALED_FLOATS = FLOATS[:-1]
NP = {T.FLOAT16: np.float16, T.BFLOAT16: ml_dtypes.bfloat16,
      T.FLOAT: np.float32, T.DOUBLE: np.float64}


def _model(op, types, shapes, *, attrs=None, omitted=False, symbolic=False,
           outputs=None, crossing=False):
    inputs, nodes, names = [], [], []
    for i, (dtype, shape) in enumerate(zip(types, shapes, strict=True)):
        name = f"x{i}"
        inputs.append(helper.make_tensor_value_info(
            name, T.FLOAT if dtype == T.BFLOAT16 else dtype,
            [None] * len(shape) if symbolic else shape))
        if dtype == T.BFLOAT16:
            nodes.append(helper.make_node("Cast", [name], [f"typed{i}"], to=dtype))
            name = f"typed{i}"
        if crossing:
            nodes.append(helper.make_node("Identity", [name], [f"live{i}"]))
            name = f"live{i}"
        names.append(name)
    if omitted:
        names.append("")
    dtype = T.BOOL if op == "IsAllFinite" else types[0]
    output_shape = [] if op == "IsAllFinite" else shapes[0]
    output_shape = ([None] * len(output_shape) if symbolic else output_shape)
    slots = ["result"] if outputs is None else outputs
    nodes.append(helper.make_node(op, names, slots, domain="com.microsoft",
                                  name="elementary_target", **(attrs or {})))
    output_name = "result"
    if dtype == T.BFLOAT16:
        nodes.append(helper.make_node("Cast", ["result"], ["out"], to=T.FLOAT))
        output_name, dtype = "out", T.FLOAT
    return helper.make_model(
        helper.make_graph(nodes, "elementary_contrib", inputs, [
            helper.make_tensor_value_info(output_name, dtype, output_shape)]),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )


def _session(model, path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.enable_profiling = True
    options.profile_file_prefix = str(path)
    session = ort.InferenceSession(model.SerializeToString(), options,
                                   providers=["MLXExecutionProvider"])
    session.disable_fallback()
    assert "MLXExecutionProvider" in session.get_providers()
    return session


def _placement(session):
    events = json.loads(Path(session.end_profiling()).read_text())
    kernels = [event for event in events
               if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time")]
    assert kernels, "profiling must contain actual node kernel events"
    assert all(event.get("args", {}).get("provider") == "MLXExecutionProvider"
               for event in kernels), kernels


def _typed(values, dtype):
    return np.asarray(values, dtype=NP[dtype])


def _feed(values, types):
    return {f"x{i}": np.asarray(x, dtype=np.float32 if dtype == T.BFLOAT16 else NP[dtype])
            for i, (x, dtype) in enumerate(zip(values, types, strict=True))}


def _sigmoid(value, dtype):
    work = np.float64 if dtype == T.DOUBLE else np.float32
    value = work(value)
    if math.isnan(value):
        return NP[dtype](np.nan)
    exponent = work(math.exp(-abs(float(value))))
    gate = (work(1) / work(work(1) + exponent) if value >= 0
            else exponent / work(work(1) + exponent))
    return NP[dtype](gate)


def _oracle(op, arrays, types, attrs=None):
    attrs = attrs or {}
    dtype = types[0]
    cast = NP[dtype]
    work = np.float64 if dtype == T.DOUBLE else np.float32
    x = _typed(arrays[0], dtype)
    if op == "IsAllFinite":
        nan_only, inf_only = attrs.get("isnan_only", 0), attrs.get("isinf_only", 0)
        assert not (nan_only and inf_only)
        values = [_typed(value, dtype).astype(work) for value in arrays]
        invalid = np.isnan if nan_only else np.isinf if inf_only else lambda x: ~np.isfinite(x)
        return np.asarray(not any(invalid(value).any() for value in values), dtype=np.bool_)
    result = np.empty(x.shape, dtype=cast)
    bias = None
    if len(arrays) > 1 and op in ("FastGelu", "BiasAdd"):
        bias = np.broadcast_to(_typed(arrays[1], dtype), x.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        for index in np.ndindex(x.shape):
            value = x[index]
            if bias is not None:
                value = cast(work(value) + work(bias[index]))
            if op == "FastGelu":
                square = cast(work(value) * work(value))
                factor = cast(work(cast(0.035677)) * work(square))
                inner = cast(work(cast(0.797885)) + work(factor))
                inner = cast(work(value) * work(inner))
                gate = cast(math.tanh(float(inner)))
                gate = cast(work(1) + work(gate))
                product = cast(work(value) * work(gate))
                result[index] = cast(work(cast(0.5)) * work(product))
            elif op == "QuickGelu":
                alpha = cast(np.float32(attrs.get("alpha", 1.702)))
                z = cast(work(value) * work(alpha))
                result[index] = cast(work(value) * work(_sigmoid(z, dtype)))
            elif op == "BiasAdd":
                result[index] = cast(work(value) + work(_typed(arrays[2], dtype)[index]))
            elif op == "ScaledSiLU":
                scale = (np.float32(_typed(arrays[1], types[1]).item())
                         if len(arrays) == 2 else np.float32(attrs.get("alpha", 1)))
                z = cast(np.float32(value) * scale)
                gate = _sigmoid(z, dtype)
                result[index] = cast(np.float32(z) * np.float32(gate))
            else:
                raise AssertionError(op)
    return result.astype(np.float32) if dtype == T.BFLOAT16 else result


def _compare(actual, expected, *, exact=False):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if exact or actual.dtype == np.bool_:
        np.testing.assert_array_equal(actual, expected)
    else:
        # Fixed four-ulp-scale bound for host libm vs the independent scalar oracle.
        np.testing.assert_allclose(actual, expected, rtol=4 * np.finfo(actual.dtype).eps,
                                   atol=0, equal_nan=True)
    zeros = expected == 0
    np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("alpha", [None, 1.702, 0.0, -1.0, 2.0])
@pytest.mark.parametrize("eager", [False, True])
def test_quick_gelu_live_types_attributes(dtype, alpha, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    attrs = {} if alpha is None else {"alpha": alpha}
    session = _session(_model("QuickGelu", [dtype], [[6]], attrs=attrs, crossing=True),
                       tmp_path / "quick")
    for values in [[-3, -1, -0.0, 0, 1, 3], [1e-3, -2, 4, 2, 0, -4],
                   [-3, -1, -0.0, 0, 1, 3]]:
        arrays = [_typed(values, dtype)]
        _compare(session.run(None, _feed(arrays, [dtype]))[0],
                 _oracle("QuickGelu", arrays, [dtype], attrs))
    _placement(session)


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("bias_shape", [None, (), (3,), (2, 1), (2, 3)])
@pytest.mark.parametrize("eager", [False, True])
def test_fast_gelu_function_bias_forms(dtype, bias_shape, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    shapes = [(2, 3)] + ([] if bias_shape is None else [bias_shape])
    types = [dtype] * len(shapes)
    session = _session(_model("FastGelu", types, shapes, crossing=True), tmp_path / "fast")
    for shift in [0, 1, -1, 0]:
        arrays = [_typed(np.arange(6).reshape(2, 3) / 2 - 1 + shift, dtype)]
        if bias_shape is not None:
            arrays.append(_typed(np.full(bias_shape, shift / 4), dtype))
        _compare(session.run(None, _feed(arrays, types))[0], _oracle("FastGelu", arrays, types))
    _placement(session)


@pytest.mark.parametrize("op,types", [
    ("FastGelu", [dtype]) for dtype in FLOATS
] + [("QuickGelu", [dtype]) for dtype in FLOATS]
    + [("ScaledSiLU", [dtype]) for dtype in SCALED_FLOATS])
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3)])
def test_scalar_and_empty(op, types, shape, tmp_path):
    arrays = [_typed(np.full(shape, -0.0), types[0])]
    session = _session(_model(op, types, [shape]), tmp_path / "empty")
    _compare(session.run(None, _feed(arrays, types))[0], _oracle(op, arrays, types), exact=True)
    _placement(session)


@pytest.mark.parametrize("op", ["FastGelu", "ScaledSiLU"])
@pytest.mark.parametrize("dtype", SCALED_FLOATS)
def test_explicit_missing_optional(op, dtype, tmp_path):
    arrays = [_typed([1, -1, 0], dtype)]
    session = _session(_model(op, [dtype], [[3]], omitted=True), tmp_path / "missing")
    _compare(session.run(None, _feed(arrays, [dtype]))[0], _oracle(op, arrays, [dtype]))
    _placement(session)


@pytest.mark.parametrize("dtype", [T.FLOAT16, T.FLOAT])
@pytest.mark.parametrize("eager", [False, True])
def test_bias_add_exact_geometry_live_empty(dtype, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    types = [dtype] * 3
    session = _session(_model("BiasAdd", types, [[2, 3, 4], [4], [2, 3, 4]],
                              symbolic=True, crossing=True), tmp_path / "bias")
    for n, channels, shift in [(2, 4, 0), (1, 2, 2), (0, 4, -1), (2, 0, 0), (2, 4, 0)]:
        shape = (n, 3, channels)
        arrays = [_typed(np.full(shape, 2048 + shift), dtype),
                  _typed(np.full((channels,), 1), dtype),
                  _typed(np.full(shape, -2048), dtype)]
        _compare(session.run(None, _feed(arrays, types))[0],
                 _oracle("BiasAdd", arrays, types), exact=True)
    _placement(session)


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("flags", [{}, {"isnan_only": 0, "isinf_only": 0},
                                   {"isnan_only": 1}, {"isinf_only": 1},
                                   {"isnan_only": -3}, {"isinf_only": 2}])
@pytest.mark.parametrize("eager", [False, True])
def test_is_all_finite_variadic_scalar_empty_live(dtype, flags, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    types = [dtype] * 3
    session = _session(_model("IsAllFinite", types, [[], [4], [0, 2]], attrs=flags,
                              symbolic=True, crossing=True), tmp_path / "finite")
    for scalar, vector in [(0, [1, -0.0, 2, 3]), (np.nan, [1, 2, 3, 4]),
                           (0, [np.inf, -np.inf, 2, 3]), (0, [np.nan, np.inf, 0, 1]),
                           (0, [1, -0.0, 2, 3])]:
        arrays = [_typed(scalar, dtype), _typed(vector, dtype), _typed(np.empty((0, 2)), dtype)]
        _compare(session.run(None, _feed(arrays, types))[0],
                 _oracle("IsAllFinite", arrays, types, flags), exact=True)
    _placement(session)


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3)])
def test_is_all_finite_single_input(dtype, shape, tmp_path):
    arrays = [_typed(np.zeros(shape), dtype)]
    session = _session(_model("IsAllFinite", [dtype], [shape]), tmp_path / "single")
    _compare(session.run(None, _feed(arrays, [dtype]))[0], np.asarray(True), exact=True)
    _placement(session)


@pytest.mark.parametrize("dtype", SCALED_FLOATS)
@pytest.mark.parametrize("scale_dtype", SCALED_FLOATS)
@pytest.mark.parametrize("eager", [False, True])
def test_scaled_silu_independent_m_scalar_override_stages(dtype, scale_dtype, eager,
                                                        monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    types = [dtype, scale_dtype]
    attrs = {"alpha": -7.0}
    session = _session(_model("ScaledSiLU", types, [[5], []], attrs=attrs, crossing=True),
                       tmp_path / "scale")
    for scale in [2.0, 1.00390625, -0.75, 0.0, 2.0]:
        arrays = [_typed([-3, -1, 0, 1, 3], dtype), _typed(scale, scale_dtype)]
        _compare(session.run(None, _feed(arrays, types))[0],
                 _oracle("ScaledSiLU", arrays, types, attrs), exact=True)
    _placement(session)


@pytest.mark.parametrize("dtype,golden", [
    (T.FLOAT, 1.7615940570831299), (T.FLOAT16, 1.76171875), (T.BFLOAT16, 1.7578125)])
@pytest.mark.parametrize("alpha", [None, 1.0, 0.0, -2.0, 2.0])
def test_scaled_silu_literal_goldens_and_alpha(dtype, golden, alpha, tmp_path):
    attrs = {} if alpha is None else {"alpha": alpha}
    arrays = [_typed([1], dtype)]
    session = _session(_model("ScaledSiLU", [dtype], [[1]], attrs=attrs), tmp_path / "golden")
    actual = session.run(None, _feed(arrays, [dtype]))[0]
    expected = (_typed([golden], dtype).astype(np.float32) if dtype == T.BFLOAT16
                else _typed([golden], dtype)) if alpha == 2 else _oracle(
                    "ScaledSiLU", arrays, [dtype], attrs)
    _compare(actual, expected, exact=True)
    _placement(session)


@pytest.mark.parametrize("op,types,shapes,attrs", [
    ("BiasAdd", [T.FLOAT] * 3, [[2, 3], [3], [2, 3]], {}),
    ("BiasAdd", [T.FLOAT] * 3, [[2, 3, 4], [1], [2, 3, 4]], {}),
    ("BiasAdd", [T.FLOAT] * 3, [[2, 3, 4], [4], [1, 3, 4]], {}),
    ("BiasAdd", [T.DOUBLE] * 3, [[2, 3, 4], [4], [2, 3, 4]], {}),
    ("ScaledSiLU", [T.FLOAT] * 2, [[2, 3], [1]], {}),
    ("ScaledSiLU", [T.FLOAT] * 2, [[2, 3], [0]], {}),
    ("ScaledSiLU", [T.DOUBLE], [[2, 3]], {}),
    ("QuickGelu", [T.INT32], [[2]], {}),
    ("IsAllFinite", [T.FLOAT, T.DOUBLE], [[2], [2]], {}),
    ("IsAllFinite", [T.FLOAT], [[2]], {"isnan_only": 1, "isinf_only": -1}),
    ("FastGelu", [T.FLOAT] * 2, [[1, 3], [2, 3]], {}),
])
def test_invalid_schema_or_claim_forms_fail_closed(op, types, shapes, attrs, tmp_path):
    # No CPU fallback proxy is accepted as successful execution of an unsupported form.
    with pytest.raises((ort.capi.onnxruntime_pybind11_state.InvalidGraph,
                        ort.capi.onnxruntime_pybind11_state.Fail,
                        ort.capi.onnxruntime_pybind11_state.InvalidArgument)):
        _session(_model(op, types, shapes, attrs=attrs), tmp_path / "invalid")


@pytest.mark.parametrize("op", ["FastGelu", "QuickGelu", "BiasAdd", "IsAllFinite", "ScaledSiLU"])
def test_required_inputs_and_outputs_cannot_be_omitted(op, tmp_path):
    types = [T.FLOAT] * (3 if op == "BiasAdd" else 1)
    shapes = [[1, 1, 1], [1], [1, 1, 1]] if op == "BiasAdd" else [[1]]
    for missing_input in [False, True]:
        model = _model(op, types, shapes)
        target = next(node for node in model.graph.node if node.domain == "com.microsoft")
        if missing_input:
            target.input[0] = ""
        else:
            target.output[0] = ""
        with pytest.raises((ort.capi.onnxruntime_pybind11_state.InvalidGraph,
                            ort.capi.onnxruntime_pybind11_state.Fail,
                            ort.capi.onnxruntime_pybind11_state.InvalidArgument)):
            _session(model, tmp_path / f"omitted_{missing_input}")


@pytest.mark.parametrize("op", ["FastGelu", "QuickGelu", "BiasAdd", "IsAllFinite", "ScaledSiLU"])
def test_wrong_input_and_output_arities(op, tmp_path):
    types = [T.FLOAT] * (3 if op == "BiasAdd" else 1)
    shapes = [[1, 1, 1], [1], [1, 1, 1]] if op == "BiasAdd" else [[1]]
    for variant in ["no_inputs", "extra_input", "no_outputs", "extra_output"]:
        if op == "IsAllFinite" and variant == "extra_input":
            continue  # The variadic positive fixture already covers this legal form.
        model = _model(op, types, shapes)
        target = next(node for node in model.graph.node if node.domain == "com.microsoft")
        if variant == "no_inputs":
            del target.input[:]
        elif variant == "extra_input":
            target.input.extend(["x0"] * 3)
        elif variant == "no_outputs":
            del target.output[:]
        else:
            target.output.append("extra")
        with pytest.raises((ort.capi.onnxruntime_pybind11_state.InvalidGraph,
                            ort.capi.onnxruntime_pybind11_state.Fail,
                            ort.capi.onnxruntime_pybind11_state.InvalidArgument)):
            _session(model, tmp_path / variant)


@pytest.mark.parametrize("op", ["FastGelu", "QuickGelu", "ScaledSiLU"])
@pytest.mark.parametrize("dtype", SCALED_FLOATS)
@pytest.mark.parametrize("eager", [False, True])
def test_symbolic_shape_changing_same_session(op, dtype, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    types = [dtype, dtype] if op == "FastGelu" else [dtype]
    shapes = [[2, 3], [3]] if op == "FastGelu" else [[2, 3]]
    session = _session(_model(op, types, shapes, symbolic=True, crossing=True),
                       tmp_path / "dynamic")
    for n, channels in [(2, 3), (1, 2), (0, 3), (2, 0), (2, 3)]:
        arrays = [_typed(np.full((n, channels), 0.75), dtype)]
        if op == "FastGelu":
            arrays.append(_typed(np.full((channels,), -0.25), dtype))
        _compare(session.run(None, _feed(arrays, types))[0], _oracle(op, arrays, types),
                 exact=op == "ScaledSiLU")
    _placement(session)


@pytest.mark.parametrize("op,dtype", [
    (op, dtype) for op in ("FastGelu", "QuickGelu") for dtype in FLOATS
] + [("ScaledSiLU", dtype) for dtype in SCALED_FLOATS])
def test_nonfinite_live_values(op, dtype, tmp_path):
    session = _session(_model(op, [dtype], [[4]]), tmp_path / "nonfinite")
    for values in [[np.nan, np.inf, -np.inf, -0.0], [1, 0, -1, 3]]:
        arrays = [_typed(values, dtype)]
        _compare(session.run(None, _feed(arrays, [dtype]))[0],
                 _oracle(op, arrays, [dtype]), exact=op == "ScaledSiLU")
    _placement(session)


def _example(op):
    if op == "BiasAdd":
        return [np.ones((1, 2, 3)), np.ones(3), -np.ones((1, 2, 3))]
    if op == "IsAllFinite":
        return [np.ones(3), np.empty((0, 2))]
    if op in ("FastGelu", "ScaledSiLU"):
        return [np.ones(3), np.asarray(2.0)]
    return [np.ones(3)]


@pytest.mark.parametrize("op,dtype", [
    (op, dtype) for op in ("FastGelu", "QuickGelu", "IsAllFinite", "ScaledSiLU")
    for dtype in SCALED_FLOATS
] + [("BiasAdd", dtype) for dtype in (T.FLOAT16, T.FLOAT)])
def test_initializer_and_live_input_sources(op, dtype, tmp_path):
    arrays = [_typed(x, dtype) for x in _example(op)]
    types = [dtype] * len(arrays)
    model = _model(op, types, [x.shape for x in arrays])
    feeds = _feed(arrays, types)
    name = "x1" if len(arrays) > 1 else "x0"
    model.graph.initializer.append(numpy_helper.from_array(feeds.pop(name), name=name))
    for index, value in enumerate(model.graph.input):
        if value.name == name:
            del model.graph.input[index]
            break
    session = _session(model, tmp_path / "initializer")
    _compare(session.run(None, feeds)[0], _oracle(op, arrays, types),
             exact=op in ("BiasAdd", "IsAllFinite", "ScaledSiLU"))
    _placement(session)


@pytest.mark.parametrize("op,attrs", [
    ("QuickGelu", {"alpha": 1}), ("ScaledSiLU", {"alpha": 1}),
    ("IsAllFinite", {"isnan_only": 1.0}), ("IsAllFinite", {"isinf_only": 1.0}),
])
def test_attribute_types_are_not_coerced(op, attrs, tmp_path):
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidGraph):
        _session(_model(op, [T.FLOAT], [[3]], attrs=attrs), tmp_path / "attribute_type")


@pytest.mark.parametrize("op", ["FastGelu", "QuickGelu", "BiasAdd", "IsAllFinite", "ScaledSiLU"])
def test_output_dtype_is_not_coerced(op, tmp_path):
    arrays = _example(op)
    model = _model(op, [T.FLOAT] * len(arrays), [x.shape for x in arrays])
    model.graph.output[0].type.tensor_type.elem_type = T.INT32
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail):
        _session(model, tmp_path / "output_type")


@pytest.mark.parametrize("op", ["FastGelu", "QuickGelu", "BiasAdd", "IsAllFinite", "ScaledSiLU"])
@pytest.mark.parametrize("depth", [1, 2])
def test_nested_eager_parent_with_live_branch(op, depth, tmp_path):
    arrays = [_typed(x, T.FLOAT) for x in _example(op)]
    types = [T.FLOAT] * len(arrays)
    model = _model(op, types, [x.shape for x in arrays])
    body = helper.make_graph(list(model.graph.node), "elementary_branch", [],
                             list(model.graph.output))
    for level in range(depth):
        node = helper.make_node("If", ["condition"], ["result"],
                                then_branch=body, else_branch=body)
        body = helper.make_graph([node], f"parent_{level}", [], list(model.graph.output))
    del model.graph.node[:]
    model.graph.node.extend(body.node)
    model.graph.input.append(helper.make_tensor_value_info("condition", T.BOOL, []))
    session = _session(model, tmp_path / "nested")
    for flag, value in [(True, 1), (False, -1), (True, 1)]:
        live = [array * value for array in arrays]
        feeds = _feed(live, types)
        feeds["condition"] = np.asarray(flag)
        _compare(session.run(None, feeds)[0], _oracle(op, live, types),
                 exact=op in ("BiasAdd", "IsAllFinite", "ScaledSiLU"))
    _placement(session)
