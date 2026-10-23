"""Versioned Dropout controls, masks, live values, and numerical boundaries."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT
from onnxruntime.capi.onnxruntime_pybind11_state import EPFail


def _value(name, dtype, shape):
    return ir.Value(name=name, type=ir.TensorType(dtype),
                    shape=None if shape is None else ir.Shape(shape))


def _numpy_dtype(dtype):
    return {DT.FLOAT16: np.float16, DT.DOUBLE: np.float64}.get(dtype, np.float32)


def _model(dtype=DT.FLOAT, *, shape=(256,), opset=13, ratio_dtype=DT.FLOAT,
           controls="boundary", mask="present", seed=None, legacy_training=False,
           omit_ratio=False, omit_training=False, control_shape=()):
    inputs, nodes, initializers = [], [], []
    boundary_dtype = DT.FLOAT if dtype == DT.BFLOAT16 else dtype
    x = _value("x", boundary_dtype, shape)
    inputs.append(x)
    data = x
    if dtype == DT.BFLOAT16:
        data = _value("data_bf", dtype, shape)
        nodes.append(ir.node("Cast", [x], attributes={"to": int(dtype)}, outputs=[data]))
    arguments = [data]
    attributes = {}
    if opset < 12:
        attributes["ratio"] = 0.5
        if opset < 7:
            attributes["is_test"] = 0 if legacy_training else 1
    else:
        for name, dt, omitted, constant in [
            ("ratio", ratio_dtype, omit_ratio, 0.5),
            ("training", DT.BOOL, omit_training, True),
        ]:
            if omitted:
                arguments.append(ir.Value(name=""))
                continue
            source_dtype = DT.FLOAT if dt == DT.BFLOAT16 else dt
            source = _value(name, source_dtype, control_shape)
            if controls == "initializer":
                source.const_value = ir.tensor(
                    np.asarray(constant, dtype=_numpy_dtype(dt) if dt != DT.BOOL else np.bool_),
                    name=name,
                )
                initializers.append(source)
            else:
                inputs.append(source)
            value = source
            if dt == DT.BFLOAT16 or controls == "intermediate":
                value = _value(f"{name}_value", dt, control_shape)
                nodes.append(ir.node(
                    "Cast" if dt == DT.BFLOAT16 else "Identity", [source],
                    attributes={"to": int(dt)} if dt == DT.BFLOAT16 else {}, outputs=[value],
                ))
            arguments.append(value)
        if seed is not None:
            attributes["seed"] = seed
    y = _value("y", dtype, shape)
    mask_dtype = dtype if opset < 10 else DT.BOOL
    mask_value = _value("mask" if mask == "present" else "", mask_dtype, shape)
    node_outputs = [y] if mask == "omitted" else [y, mask_value]
    nodes.append(ir.node("Dropout", arguments, attributes=attributes, outputs=node_outputs))
    output = y
    if dtype == DT.BFLOAT16:
        output = _value("output", DT.FLOAT, shape)
        nodes.append(ir.node("Cast", [y], attributes={"to": int(DT.FLOAT)}, outputs=[output]))
    outputs = [output, mask_value] if mask == "present" else [output]
    graph = ir.Graph(inputs, outputs, nodes=nodes, initializers=initializers,
                     name="dropout_forms", opset_imports={"": opset})
    return ir.to_proto(ir.Model(graph, ir_version=9)).SerializeToString()


def _session(model):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"])
    session.disable_fallback()
    return session


@pytest.fixture(params=[False, True], ids=["default", "eager"])
def execution(request, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if request.param else "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE", "0")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_PREFILL_COMPILE", "0")


@pytest.mark.parametrize("dtype", [DT.FLOAT16, DT.FLOAT, DT.DOUBLE])
@pytest.mark.parametrize("ratio_dtype", [DT.FLOAT16, DT.FLOAT, DT.DOUBLE])
@pytest.mark.parametrize("controls", ["boundary", "initializer", "intermediate"])
def test_dropout_training_type_and_control_matrix(execution, dtype, ratio_dtype, controls):
    session = _session(_model(dtype, ratio_dtype=ratio_dtype, controls=controls, seed=2**40 + 7))
    x = (np.arange(256) % 17 - 8).astype(_numpy_dtype(dtype))
    feeds = {"x": x}
    if controls != "initializer":
        feeds.update(ratio=np.asarray(0.5, _numpy_dtype(ratio_dtype)),
                     training=np.asarray(True))
    before = {name: value.tobytes() for name, value in feeds.items()}
    y, mask = session.run(None, feeds)
    assert y.dtype == x.dtype and mask.dtype == np.bool_
    assert y.shape == mask.shape == x.shape
    assert mask.any() and not mask.all()
    np.testing.assert_array_equal(y, x * mask.astype(x.dtype) * np.asarray(2, x.dtype))
    assert all(value.tobytes() == before[name] for name, value in feeds.items())


@pytest.mark.parametrize("opset", [1, 6, 7, 10, 12, 13, 22])
def test_dropout_versioned_inference_masks(execution, opset):
    session = _session(_model(opset=opset))
    x = np.arange(256, dtype=np.float32)
    feeds = {"x": x}
    if opset >= 12:
        feeds.update(ratio=np.asarray(np.nan, np.float32), training=np.asarray(False))
    y, mask = session.run(None, feeds)
    assert y.tobytes() == x.tobytes()
    assert mask.dtype == (np.float32 if opset < 10 else np.bool_)
    np.testing.assert_array_equal(mask, np.ones(256, dtype=mask.dtype))


@pytest.mark.parametrize("opset", [1, 6])
def test_dropout_legacy_training(execution, opset):
    session = _session(_model(opset=opset, legacy_training=True))
    x = np.ones(256, np.float32)
    previous = None
    for _ in range(3):
        y, mask = session.run(None, {"x": x})
        assert mask.dtype == np.float32
        assert np.isin(mask, [0, 1]).all() and mask.any() and not mask.all()
        np.testing.assert_array_equal(y, x * mask * np.float32(2))
        if previous is not None:
            assert not np.array_equal(mask, previous)
        previous = mask.copy()


def test_dropout_live_controls_and_error_recovery(execution):
    session = _session(_model(seed=19))
    x = np.ones(256, np.float32)
    for training, ratio in [
        (False, np.nan), (True, 0.5), (False, np.inf), (True, 0.25), (True, 0),
    ]:
        y, mask = session.run(None, {
            "x": x, "ratio": np.asarray(ratio, np.float32), "training": np.asarray(training),
        })
        expected = x if not training else x * mask / np.float32(1 - ratio)
        np.testing.assert_array_equal(y, expected)
        if not training or ratio == 0:
            assert mask.all()
    for ratio in [np.nan, np.inf, -0.5, 1, 2]:
        with pytest.raises(EPFail, match="Dropout training ratio"):
            session.run(None, {
                "x": x, "ratio": np.asarray(ratio, np.float32), "training": np.asarray(True),
            })
        y, mask = session.run(None, {
            "x": x, "ratio": np.asarray(0.5, np.float32), "training": np.asarray(True),
        })
        np.testing.assert_array_equal(y, x * mask * np.float32(2))


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3)])
def test_dropout_scalar_and_empty_geometry(execution, shape):
    session = _session(_model(shape=None, seed=4))
    x = np.ones(shape, np.float32)
    y, mask = session.run(None, {
        "x": x, "ratio": np.asarray(0.5, np.float32), "training": np.asarray(True),
    })
    assert y.shape == mask.shape == shape
    np.testing.assert_array_equal(y, x * mask * np.float32(2))


@pytest.mark.parametrize("mask", ["present", "omitted", "empty"])
def test_dropout_optional_controls_and_outputs(execution, mask):
    session = _session(_model(mask=mask, omit_ratio=True, seed=17))
    outputs = session.run(None, {"x": np.ones(256, np.float32), "training": np.asarray(True)})
    assert np.isin(outputs[0], [0, 2]).all()
    assert (outputs[0] == 0).any() and (outputs[0] == 2).any()
    session = _session(_model(mask=mask, omit_training=True))
    x = np.ones(256, np.float32)
    outputs = session.run(None, {"x": x, "ratio": np.asarray(np.inf, np.float32)})
    assert outputs[0].tobytes() == x.tobytes()


@pytest.mark.parametrize("opset,ratio_dtype", [(13, DT.FLOAT), (22, DT.BFLOAT16)])
def test_dropout_bfloat16_interior(execution, opset, ratio_dtype):
    session = _session(_model(DT.BFLOAT16, ratio_dtype=ratio_dtype, opset=opset, seed=23))
    x = (np.arange(256) % 17 - 8).astype(np.float32)
    y, mask = session.run(None, {
        "x": x, "ratio": np.asarray(0.5, np.float32), "training": np.asarray(True),
    })
    np.testing.assert_array_equal(y, x * mask * np.float32(2))


def test_dropout_double_ratio_precision(execution):
    session = _session(_model(DT.DOUBLE, ratio_dtype=DT.DOUBLE, seed=29))
    x = np.ones(256, np.float64)
    for ratio in [0.123456789123, np.nextafter(1.0, 0.0)]:
        y, mask = session.run(None, {
            "x": x, "ratio": np.asarray(ratio), "training": np.asarray(True),
        })
        np.testing.assert_allclose(y, x * mask / (1 - ratio), rtol=1e-15, atol=0)


def test_dropout_unknown_control_rank_validation(execution):
    session = _session(_model(control_shape=None, seed=31))
    x = np.ones(256, np.float32)
    for ratio, training in [(np.array([0.5], np.float32), np.asarray(True)),
                            (np.asarray(0.5, np.float32), np.array([True]))]:
        with pytest.raises(EPFail, match="scalar"):
            session.run(None, {"x": x, "ratio": ratio, "training": training})
    y, mask = session.run(None, {
        "x": x, "ratio": np.asarray(0.5, np.float32), "training": np.asarray(True),
    })
    np.testing.assert_array_equal(y, x * mask * np.float32(2))


def test_dropout_seed_preserves_high_integer_bits(execution):
    masks = []
    feeds = {"x": np.ones(256, np.float32), "ratio": np.asarray(0.5, np.float32),
             "training": np.asarray(True)}
    for seed in [7, 2**40 + 7, -1, -(2**63)]:
        session = _session(_model(seed=seed))
        _, first = session.run(None, feeds)
        _, second = session.run(None, feeds)
        np.testing.assert_array_equal(first, second)
        masks.append(first.copy())
    assert not np.array_equal(masks[0], masks[1])


def test_dropout_nonfinite_and_signed_zero_semantics(execution):
    x = np.tile(np.array([0, 0x80000000, 0x7FC00041, 0x7F800000, 0xFF800000],
                        np.uint32).view(np.float32), 64)
    session = _session(_model(shape=x.shape, seed=37))
    y, mask = session.run(None, {
        "x": x, "ratio": np.asarray(np.nan, np.float32), "training": np.asarray(False),
    })
    assert y.tobytes() == x.tobytes() and mask.all()
    y, mask = session.run(None, {
        "x": x, "ratio": np.asarray(0.5, np.float32), "training": np.asarray(True),
    })
    with np.errstate(invalid="ignore"):
        expected = x * mask * np.float32(2)
    np.testing.assert_array_equal(y, expected)
    np.testing.assert_array_equal(np.signbit(y[x == 0]), np.signbit(expected[x == 0]))
