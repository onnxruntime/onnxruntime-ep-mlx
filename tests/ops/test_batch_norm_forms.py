"""BatchNormalization versioned types, modes, optional statistics and live geometry."""

import json
import os
from pathlib import Path
import subprocess
import sys

import ml_dtypes
import numpy as np
import onnxruntime as ort
from onnx import TensorProto as T, helper
import pytest

from test_spatial_norm_forms import _check, _session


TYPES = [T.FLOAT16, T.FLOAT, T.DOUBLE, T.BFLOAT16]
NP_TYPES = {T.FLOAT16: np.float16, T.FLOAT: np.float32,
            T.DOUBLE: np.float64, T.BFLOAT16: ml_dtypes.bfloat16}


def _model(shape, *, version=15, types=(T.FLOAT, T.FLOAT, T.FLOAT),
           training=False, spatial=True, attrs=None, holes=(), only_y=False, parameters=None):
    x_type, affine_type, statistics_type = types
    input_types = [x_type, affine_type, affine_type, statistics_type, statistics_type]
    names = ["x", "s", "b", "m", "v"]
    inputs, nodes = [], []
    internal = []
    for name, dtype in zip(names, input_types, strict=True):
        external_type = T.FLOAT if dtype == T.BFLOAT16 else dtype
        inputs.append(helper.make_tensor_value_info(
            name, external_type, shape if name == "x" else parameters))
        if dtype == T.BFLOAT16:
            internal.append(name + "_bf16")
            nodes.append(helper.make_node("Cast", [name], [internal[-1]], to=dtype))
        else:
            internal.append(name)
    attributes = dict(attrs or {})
    if version < 7:
        attributes.setdefault("is_test", 0 if training else 1)
    if version < 9:
        attributes["spatial"] = int(spatial)
    if version == 1:
        attributes["consumed_inputs"] = [0] * 5
    if version >= 14:
        attributes["training_mode"] = int(training)
    output_names = ["y", "rm", "rv", "sm", "inv"][:1 if not training else 5 if version < 14 else 3]
    node_outputs = ["" if i in holes else name for i, name in enumerate(output_names)]
    nodes.append(helper.make_node("BatchNormalization", internal,
                                  [name + "_internal" if name else "" for name in node_outputs],
                                  **attributes))
    outputs = []
    for index, name in enumerate(node_outputs):
        if not name:
            continue
        dtype = x_type if index == 0 else statistics_type
        internal_name = name + "_internal"
        if dtype == T.BFLOAT16:
            nodes.append(helper.make_node("Cast", [internal_name], [name], to=T.FLOAT))
        else:
            nodes.append(helper.make_node("Identity", [internal_name], [name]))
        if not only_y or index == 0:
            outputs.append(helper.make_tensor_value_info(
                name, T.FLOAT if dtype == T.BFLOAT16 else dtype,
                None if shape is None or index else [None] * len(shape)))
    return helper.make_model(
        helper.make_graph(nodes, "batch_norm_forms", inputs, outputs),
        opset_imports=[helper.make_opsetid("", version)], ir_version=11,
    )


def _quantize(value, dtype):
    return np.asarray(value).astype(NP_TYPES[dtype]).astype(
        np.float32 if dtype == T.BFLOAT16 else NP_TYPES[dtype])


def _feeds(shape, types=(T.FLOAT, T.FLOAT, T.FLOAT), *, spatial=True, change=0):
    x_type, affine_type, statistics_type = types
    parameters = (1 if len(shape) == 1 else shape[1],) if spatial else shape[1:]
    count = int(np.prod(parameters))
    return {
        "x": _quantize(np.linspace(-2, 3, int(np.prod(shape))).reshape(shape)
                       * (1 + change) + change * 0.25, x_type),
        "s": _quantize(np.linspace(0.5, 1.5, count).reshape(parameters) + change * 0.05, affine_type),
        "b": _quantize(np.linspace(-0.25, 0.5, count).reshape(parameters) + change * 0.01, affine_type),
        "m": _quantize(np.linspace(2, 4, count).reshape(parameters) + change, statistics_type),
        "v": _quantize(np.linspace(0.75, 2, count).reshape(parameters) + change * 0.25,
                       statistics_type),
    }


def _expected(feeds, types=(T.FLOAT, T.FLOAT, T.FLOAT), *, version=15, training=False,
              spatial=True, attrs=None, holes=(), only_y=False):
    attrs = attrs or {}
    compute = np.float64 if T.DOUBLE in types else np.float32
    x, scale, bias, input_mean, input_variance = [
        feeds[name].astype(compute) for name in ["x", "s", "b", "m", "v"]]
    parameter_shape = scale.shape
    broadcast_shape = ([1] * x.ndim if spatial else [1, *parameter_shape])
    if spatial and x.ndim > 1:
        broadcast_shape[1] = x.shape[1]
    if training:
        axes = tuple(i for i in range(x.ndim) if x.ndim == 1 or i != 1) if spatial else (0,)
        with np.errstate(invalid="ignore", divide="ignore"):
            if x.size == 0:
                mean = np.full(parameter_shape, np.nan, compute)
                variance = mean.copy()
            else:
                mean = x.mean(axis=axes).reshape(parameter_shape)
                centered = x - mean.reshape(broadcast_shape)
                variance = (centered * centered).mean(axis=axes).reshape(parameter_shape)
    else:
        mean, variance = input_mean, input_variance
    epsilon = compute(np.float32(attrs.get("epsilon", 1e-5)))
    with np.errstate(invalid="ignore", divide="ignore"):
        denominator = np.sqrt(variance + epsilon)
        y = ((x - mean.reshape(broadcast_shape)) / denominator.reshape(broadcast_shape)
             * scale.reshape(broadcast_shape) + bias.reshape(broadcast_shape))
    results = [_quantize(y, types[0])]
    if training:
        momentum = compute(np.float32(attrs.get("momentum", 0.9)))
        for previous, current in [(input_mean, mean), (input_variance, variance)]:
            results.append(_quantize(previous * momentum + current * (1 - momentum), types[2]))
        if version < 14:
            results.extend([_quantize(mean, types[2]), _quantize(1 / denominator, types[2])])
    return [result for index, result in enumerate(results)
            if index not in holes and (not only_y or index == 0)]


def _assert_outputs(actual, expected, types):
    assert len(actual) == len(expected)
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        _check(got, want, bf16=types[0 if index == 0 else 2] == T.BFLOAT16)


@pytest.mark.parametrize("case", [
    "rank_one", "double", "mixed", "training", "legacy_training",
    "nonspatial", "unknown_rank", "narrow_affine",
])
def test_batch_norm_new_forms_strict(case):
    shape = (4,) if case == "rank_one" else (2, 3, 2)
    types = (T.DOUBLE,) * 3 if case == "double" else (
        (T.FLOAT16, T.FLOAT, T.DOUBLE) if case == "mixed" else
        (T.FLOAT16,) * 3 if case == "narrow_affine" else (T.FLOAT,) * 3)
    training = case in {"training", "legacy_training"}
    version = 9 if case == "legacy_training" else 7 if case == "nonspatial" else 15
    spatial = case != "nonspatial"
    attrs = {"epsilon": 0.25}
    model = _model(None if case == "unknown_rank" else list(shape), types=types,
                   version=version, training=training, spatial=spatial, attrs=attrs)
    feeds = _feeds(shape, types, spatial=spatial)
    if case == "narrow_affine":
        feeds["x"] = np.array([1000, 1000.5, 999.5, 1001, 998, 1002] * 2,
                              np.float16).reshape(shape)
        feeds["m"][:] = 1000
    _assert_outputs(_session(model).run(None, feeds),
                    _expected(feeds, types, version=version, training=training,
                              spatial=spatial, attrs=attrs), types)


@pytest.mark.parametrize("version", [1, 6, 7, 9, 14, 15])
@pytest.mark.parametrize("dtype", [T.FLOAT16, T.FLOAT, T.DOUBLE])
@pytest.mark.parametrize("eager", [False, True])
def test_batch_norm_inference_versions(version, dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    shape = (2, 3, 2, 2)
    types = (dtype,) * 3
    feeds = _feeds(shape, types)
    model = _model(list(shape), version=version, types=types, parameters=[3])
    expected = _expected(feeds, types, version=version)
    _assert_outputs(_session(model).run(None, feeds), expected, types)
    if dtype in {T.FLOAT, T.DOUBLE} and version >= 7:
        _assert_outputs(_session(model, cpu=True).run(None, feeds), expected, types)


@pytest.mark.parametrize("types", [(x, a, s) for x in TYPES for a in TYPES for s in TYPES])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("eager", [False, True])
def test_batch_norm_independent_dtype_groups(types, training, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model([None] * 3, types=types, training=training))
    for change, shape in enumerate([(2, 3, 2), (2, 3, 2), (1, 4, 3), (2, 3, 2)]):
        feeds = _feeds(shape, types, change=change)
        _assert_outputs(session.run(None, feeds), _expected(feeds, types, training=training), types)


@pytest.mark.parametrize("dtype,statistics", [(x, s) for x in TYPES for s in TYPES])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("eager", [False, True])
def test_batch_norm_version14_statistic_group(dtype, statistics, training, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    types = (dtype, dtype, statistics)
    model = _model([None] * 3, version=14, types=types, training=training)
    feeds = _feeds((2, 3, 2), types)
    _assert_outputs(_session(model).run(None, feeds),
                    _expected(feeds, types, version=14, training=training), types)


@pytest.mark.parametrize("version,spatial", [(7, True), (7, False), (9, True),
                                            (14, True), (15, True)])
@pytest.mark.parametrize("dtype", [T.FLOAT16, T.FLOAT, T.DOUBLE])
@pytest.mark.parametrize("eager", [False, True])
def test_batch_norm_population_and_momentum(version, spatial, dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    types = (dtype,) * 3
    for momentum in [0.0, 0.25, 1.0, -0.5]:
        attrs = {"momentum": momentum, "epsilon": 0.125}
        model = _model([None] * 3, version=version, types=types, training=True,
                       spatial=spatial, attrs=attrs)
        feeds = _feeds((2, 3, 2), types, spatial=spatial)
        expected = _expected(feeds, types, version=version, training=True,
                             spatial=spatial, attrs=attrs)
        _assert_outputs(_session(model).run(None, feeds), expected, types)
        if dtype in {T.FLOAT, T.DOUBLE}:
            if spatial:
                cpu_model, cpu_feeds = model, {name: value.copy() for name, value in feeds.items()}
            else:
                cpu_model = _model([2, 6], version=version, types=types, training=True, attrs=attrs)
                cpu_feeds = {name: value.reshape((2, 6) if name == "x" else (6,)).copy()
                             for name, value in feeds.items()}
            actual = _session(cpu_model, cpu=True).run(None, cpu_feeds)
            actual = [value.reshape(want.shape) for value, want in zip(actual, expected, strict=True)]
            _assert_outputs(actual, expected, types)


@pytest.mark.parametrize("shape", [(4,), (0,), (0, 3, 2), (2, 0, 2), (2, 3, 0), (1, 3, 2, 2, 2)])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("eager", [False, True])
def test_batch_norm_empty_rank_one_and_unknown_recovery(shape, training, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(None, training=training))
    for change, current in enumerate([shape, (2, 3, 4), shape]):
        feeds = _feeds(current, change=change)
        _assert_outputs(session.run(None, feeds), _expected(feeds, training=training), (T.FLOAT,) * 3)


@pytest.mark.parametrize("version,holes", [(9, (1,)), (9, (2,)), (9, (3,)), (9, (4,)),
                                         (9, (1, 2, 3, 4)), (15, (1,)), (15, (2,)),
                                         (15, (1, 2))])
def test_batch_norm_optional_statistic_slots(version, holes):
    feeds = _feeds((2, 3, 2))
    model = _model([None] * 3, version=version, training=True, holes=holes)
    actual = _session(model).run(None, feeds)
    _assert_outputs(actual, _expected(feeds, version=version, training=True, holes=holes),
                    (T.FLOAT,) * 3)


@pytest.mark.parametrize("version", [7, 9, 15])
def test_batch_norm_training_with_unused_statistics(version):
    feeds = _feeds((2, 3, 2))
    model = _model([None] * 3, version=version, training=True, only_y=True)
    _assert_outputs(_session(model).run(None, feeds),
                    _expected(feeds, version=version, training=True, only_y=True), (T.FLOAT,) * 3)


@pytest.mark.parametrize("version", [1, 6])
def test_batch_norm_legacy_inplace_training_is_not_false_inference(version):
    with pytest.raises(Exception, match="fallback|in.place"):
        _session(_model([2, 3, 2, 2], version=version, attrs={"is_test": 0}))


@pytest.mark.parametrize("training", [False, True])
def test_batch_norm_parameter_errors_and_recovery(training):
    session = _session(_model(None, training=training))
    feeds = _feeds((2, 3, 2))
    with pytest.raises(Exception, match="rank|MLX|Mlx"):
        session.run(None, {**feeds, "x": np.array(1, np.float32)})
    for name in ["s", "b", "m", "v"]:
        with pytest.raises(Exception, match="parameter shape|MLX|Mlx"):
            session.run(None, {**feeds, name: np.ones((1, 3), np.float32)})
        _assert_outputs(session.run(None, feeds), _expected(feeds, training=training), (T.FLOAT,) * 3)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype", [T.FLOAT, T.DOUBLE])
@pytest.mark.parametrize("eager", [False, True])
def test_batch_norm_route_replay(training, dtype, eager, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    types = (dtype,) * 3
    session = _session(_model([None] * 3, training=training, types=types))
    for change, shape in enumerate([(2, 3, 2), (2, 3, 2), (1, 4, 3), (2, 3, 2)]):
        feeds = _feeds(shape, types, change=change)
        before = {name: value.copy() for name, value in feeds.items()}
        _assert_outputs(session.run(None, feeds), _expected(feeds, types, training=training), types)
        for name in feeds:
            np.testing.assert_array_equal(feeds[name], before[name])


def test_batch_norm_actual_routes(tmp_path):
    trace = tmp_path / "batch_norm_trace.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", "batch_norm_route_replay", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    paths = [event["args"] for event in json.loads(trace.read_text())
             if event.get("cat") == "ep.path"
             and "BatchNormalization" in event["args"].get("ops", "")]
    assert [event["cache"] for event in paths if event["path"] == "general"] == [
        "MISS", "HIT", "RETRACE", "HIT"] * 2
    assert sum(event["path"] == "eager" for event in paths) == 24
