"""Claimed spatial regressions, including live controls and legacy schema defaults."""

from __future__ import annotations

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

import _models as m

DT = ir.DataType


@pytest.fixture(params=[False, True], ids=["compiled", "eager"])
def execution_mode(request, monkeypatch):
    if request.param:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")


@pytest.mark.parametrize("opset", [10, 16, 22])
@pytest.mark.parametrize("mode", ["avg", "max"])
def test_roi_align_schema_default(opset, mode, execution_mode):
    model = m.make_model(
        "RoiAlign",
        [
            m.tensor("x", DT.FLOAT, [1, 1, 4, 4]),
            m.tensor("rois", DT.FLOAT, [2, 4]),
            m.tensor("batch", DT.INT64, [2]),
        ],
        [m.tensor("y", DT.FLOAT, [2, 1, 2, 2])],
        attributes={"sampling_ratio": 2, "output_height": 2, "output_width": 2, "mode": mode},
        opset=opset,
    )
    feeds = {
        "x": np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4) - 8,
        "rois": np.array([[0, 0, 3, 3], [-2, -1, 2, 3]], dtype=np.float32),
        "batch": np.zeros(2, dtype=np.int64),
    }
    # ORT 1.31 has no CPU RoiAlign-22 kernel; 16 has the same fp32 semantics.
    reference = onnx.load_model_from_string(model)
    reference.opset_import[0].version = min(opset, 16)
    reference = reference.SerializeToString()
    m.assert_matches_ref(model, feeds, m.run_cpu(reference, feeds), rtol=1e-6, atol=1e-6)
    m.assert_mlx_claims(model, feeds)
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    for shift in [0.25, -0.5]:
        feeds["rois"] += shift
        np.testing.assert_allclose(session.run(None, feeds)[0], m.run_cpu(reference, feeds)[0],
                                   rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16])
def test_max_roi_pool_extreme_and_empty_bins(dtype, execution_mode):
    np_dtype = np.float32 if dtype == DT.FLOAT else np.float16
    low = np.finfo(np_dtype).min
    model = m.make_model(
        "MaxRoiPool",
        [m.tensor("x", dtype, [1, 1, 2, 2]), m.tensor("rois", dtype, [2, 5])],
        [m.tensor("y", dtype, [2, 1, 2, 2])],
        attributes={"pooled_shape": [2, 2]},
        opset=22,
    )
    feeds = {
        "x": np.full((1, 1, 2, 2), low, dtype=np_dtype),
        "rois": np.array([[0, 0, 0, 1, 1], [0, 3, 3, 4, 4]], dtype=np_dtype),
    }
    expected = np.zeros((2, 1, 2, 2), dtype=np_dtype)
    expected[0] = low
    m.assert_matches_ref(model, feeds, [expected], rtol=0, atol=0)
    m.assert_mlx_claims(model, feeds)


@pytest.mark.parametrize("rank", [3, 4, 5])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16])
def test_max_unpool_assignment_and_live_controls(rank, explicit, dtype, execution_mode):
    shape = [1, 1] + [2] * (rank - 2)
    out_shape = [1, 1] + [4] * (rank - 2)
    inputs = [m.tensor("x", dtype, shape), m.tensor("indices", DT.INT64, shape)]
    if explicit:
        inputs.append(m.tensor("shape", DT.INT64, [rank]))
    model = m.make_model(
        "MaxUnpool", inputs, [m.tensor("y", dtype, ["d" + str(i) for i in range(rank)])],
        attributes={"kernel_shape": [2] * (rank - 2), "strides": [2] * (rank - 2)},
        opset=22,
    )
    np_dtype = np.float32 if dtype == DT.FLOAT else np.float16
    feeds = {"x": np.full(shape, 3, dtype=np_dtype),
             "indices": np.zeros(shape, dtype=np.int64)}
    if explicit:
        feeds["shape"] = np.array(out_shape, dtype=np.int64)
    m.assert_op_claimed(model, feeds, "MaxUnpool", rtol=0, atol=0)
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    for index, value in [(0, -7), (3, 11), (1, 2)]:
        feeds["indices"].fill(index)
        feeds["x"].fill(value)
        if explicit:
            out_shape[-1] += 1
            feeds["shape"] = np.array(out_shape, dtype=np.int64)
        result = session.run(None, feeds)[0]
        expected = np.zeros(out_shape, dtype=np_dtype)
        expected.flat[index] = value
        np.testing.assert_array_equal(result, expected)


def test_max_unpool_invalid_indices():
    model = m.make_model(
        "MaxUnpool", [m.tensor("x", DT.FLOAT, [1, 1, 2]),
                      m.tensor("indices", DT.INT64, [1, 1, 2])],
        [m.tensor("y", DT.FLOAT, [1, 1, 4])],
        attributes={"kernel_shape": [2], "strides": [2]},
    )
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    session.disable_fallback()
    feeds = {"x": np.ones((1, 1, 2), dtype=np.float32),
             "indices": np.array([[[0, 4]]], dtype=np.int64)}
    with pytest.raises(Exception, match="indices must be within"):
        session.run(None, feeds)


def test_max_unpool_constant_controls(execution_mode):
    x = m.tensor("x", DT.FLOAT, [1, 1, 2])
    indices = m.tensor("indices", DT.INT64, [1, 1, 2])
    indices.const_value = ir.tensor(np.array([[[1, 3]]], dtype=np.int64))
    shape = m.tensor("shape", DT.INT64, [3])
    shape.const_value = ir.tensor(np.array([1, 1, 5], dtype=np.int64))
    y = m.tensor("y", DT.FLOAT, [1, 1, 5])
    node = ir.node("MaxUnpool", [x, indices, shape],
                   attributes={"kernel_shape": [2], "strides": [2]}, outputs=[y])
    graph = ir.Graph([x], [y], nodes=[node], initializers=[indices, shape],
                     opset_imports={"": 22})
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    for value in [1, 3, -2]:
        feeds = {"x": np.full([1, 1, 2], value, dtype=np.float32)}
        expected = np.array([[[0, value, 0, value, 0]]], dtype=np.float32)
        np.testing.assert_array_equal(session.run(None, feeds)[0], expected)
        m.assert_mlx_claims(model, feeds)


@pytest.mark.parametrize("opset", [19, 22])
def test_deform_conv_large_fp16_coordinates(opset, execution_mode):
    width = 2053
    shape = [1, 1, 1, width]
    model = m.make_model(
        "DeformConv",
        [m.tensor("x", DT.FLOAT16, shape), m.tensor("w", DT.FLOAT16, [1, 1, 1, 1]),
         m.tensor("offset", DT.FLOAT16, [1, 2, 1, width])],
        [m.tensor("y", DT.FLOAT16, shape)],
        opset=opset,
    )
    x = ((np.arange(width) % 7) - 3).astype(np.float16).reshape(shape)
    offset = np.zeros((1, 2, 1, width), dtype=np.float16)
    offset[:, 1] = 0.25
    feeds = {"x": x, "w": np.ones((1, 1, 1, 1), dtype=np.float16), "offset": offset}
    expected = x.astype(np.float32) * 0.75
    expected[..., :-1] += x[..., 1:].astype(np.float32) * 0.25
    m.assert_matches_ref(model, feeds, [expected.astype(np.float16)], rtol=0, atol=0)
    m.assert_mlx_claims(model, feeds)
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    for fraction, weight in [(0.5, 2), (0, -1), (0.25, 1)]:
        feeds["offset"][:, 1] = fraction
        feeds["w"].fill(weight)
        expected = x.astype(np.float32) * (1 - fraction)
        expected[..., :-1] += x[..., 1:].astype(np.float32) * fraction
        expected *= weight
        np.testing.assert_array_equal(session.run(None, feeds)[0], expected.astype(np.float16))


@pytest.mark.parametrize("rank", [3, 4, 5, 6])
@pytest.mark.parametrize("op", ["GlobalAveragePool", "GlobalMaxPool", "GlobalLpPool"])
@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16])
def test_global_pool_general_rank(rank, op, dtype, execution_mode):
    shape = [2, 3] + [2] * (rank - 2)
    model = m.make_model(
        op, [m.tensor("x", dtype, ["n", "c"] + ["s" + str(i) for i in range(rank - 2)])],
        [m.tensor("y", dtype, ["n", "c"] + [1] * (rank - 2))],
    )
    np_dtype = np.float32 if dtype == DT.FLOAT else np.float16
    rng = np.random.default_rng(1981)
    feeds = {"x": (rng.standard_normal(shape) * 0.2).astype(np_dtype)}
    def reference(x):
        axes = tuple(range(2, rank))
        x = x.astype(np.float32)
        if op == "GlobalAveragePool":
            return np.mean(x, axis=axes, keepdims=True).astype(np_dtype)
        if op == "GlobalMaxPool":
            return np.max(x, axis=axes, keepdims=True).astype(np_dtype)
        return np.sqrt(np.sum(x * x, axis=axes, keepdims=True)).astype(np_dtype)

    m.assert_matches_ref(model, feeds, [reference(feeds["x"])], rtol=3e-3, atol=2e-4)
    m.assert_mlx_claims(model, feeds)
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    for new_shape in [[1, 2] + [3] * (rank - 2), shape]:
        feeds["x"] = (rng.standard_normal(new_shape) * 0.2).astype(np_dtype)
        np.testing.assert_allclose(session.run(None, feeds)[0], reference(feeds["x"]),
                                   rtol=3e-3, atol=2e-4)


@pytest.mark.parametrize("auto_pad", ["NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"])
@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16])
def test_conv3d_attributes(auto_pad, dtype, execution_mode):
    shape = [1, 2, 5, 4, 6]
    wshape = [3, 2, 2, 2, 2]
    attrs = {"strides": [2, 1, 2], "dilations": [1, 2, 1], "auto_pad": auto_pad}
    if auto_pad == "NOTSET":
        attrs["pads"] = [0, 1, 2, 1, 0, 0]
    model = m.make_model(
        "Conv", [m.tensor("x", dtype, ["n", 2, "d", "h", "w"]),
                 m.tensor("w", dtype, wshape), m.tensor("b", dtype, [3])],
        [m.tensor("y", dtype, ["n", 3, "od", "oh", "ow"])], attributes=attrs,
    )
    rng = np.random.default_rng(338)
    np_dtype = np.float32 if dtype == DT.FLOAT else np.float16
    feeds = {"x": (rng.standard_normal(shape) * 0.2).astype(np_dtype),
             "w": (rng.standard_normal(wshape) * 0.2).astype(np_dtype),
             "b": (rng.standard_normal(3) * 0.1).astype(np_dtype)}
    def reference(feeds):
        if not auto_pad.startswith("SAME_"):
            return m.run_cpu(model, feeds)[0]
        # ORT CPU rejects SAME+dilation. Explicit equivalent padding retains this cross-case.
        pads = []
        for size, stride, dilation in zip(feeds["x"].shape[2:], attrs["strides"],
                                           attrs["dilations"], strict=True):
            out_size = (size + stride - 1) // stride
            total = max((out_size - 1) * stride + dilation + 1 - size, 0)
            begin = total // 2 if auto_pad == "SAME_UPPER" else (total + 1) // 2
            pads.append((begin, total - begin))
        ref_attrs = {**attrs, "auto_pad": "NOTSET",
                     "pads": [p[0] for p in pads] + [p[1] for p in pads]}
        ref = m.make_model(
            "Conv", [m.tensor("x", dtype, list(feeds["x"].shape)),
                     m.tensor("w", dtype, wshape), m.tensor("b", dtype, [3])],
            [m.tensor("y", dtype, ["n", 3, "od", "oh", "ow"])], attributes=ref_attrs,
        )
        return m.run_cpu(ref, feeds)[0]

    m.assert_matches_ref(model, feeds, [reference(feeds)], rtol=3e-3, atol=3e-4)
    m.assert_mlx_claims(model, feeds)
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    for xshape in [[2, 2, 4, 5, 5], shape]:
        for name in ["w", "b"]:
            feeds[name] *= -0.5
        feeds["x"] = (rng.standard_normal(xshape) * 0.2).astype(np_dtype)
        np.testing.assert_allclose(session.run(None, feeds)[0], reference(feeds),
                                   rtol=3e-3, atol=3e-4)


TRANSPOSE_CASES = [
    pytest.param(1, 1, {"pads": [1, 0]}, id="1d-asymmetric"),
    pytest.param(1, 2, {"pads": [0, 2]}, id="1d-groups"),
    pytest.param(2, 2, {"pads": [0, 1, 2, 0]}, id="2d-groups-asymmetric"),
    pytest.param(3, 1, {"pads": [1, 0, 2, 0, 1, 0]}, id="3d-asymmetric"),
    *[pytest.param(rank, 1, {"auto_pad": ap}, id=f"{rank}d-{ap}")
      for rank in [1, 2, 3] for ap in ["VALID", "SAME_UPPER", "SAME_LOWER"]],
    pytest.param(1, 1, {"output_shape": [7], "pads": [9, 9]}, id="explicit-ignores-pads"),
    pytest.param(2, 2, {"output_shape": [7, 7]}, id="explicit-grouped"),
    pytest.param(3, 1, {"output_shape": [7, 7, 7], "auto_pad": "SAME_UPPER"},
                 id="explicit-3d-upper"),
    pytest.param(1, 1, {"pads": [4, 0]}, id="crop-beyond-effective-kernel"),
]


@pytest.mark.parametrize("rank,group,geometry", TRANSPOSE_CASES)
@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16])
def test_conv_transpose_forms(rank, group, geometry, dtype, execution_mode):
    shape = [1, 4] + [4] * rank
    wshape = [4, 3] + [2] * rank
    attrs = {"strides": [2] * rank, "dilations": [2] * rank,
             "output_padding": [1] * rank, "group": group, **geometry}
    model = m.make_model(
        "ConvTranspose",
        [m.tensor("x", dtype, ["n", 4] + ["s" + str(i) for i in range(rank)]),
         m.tensor("w", dtype, wshape), m.tensor("b", dtype, [3 * group])],
        [m.tensor("y", dtype, ["n", 3 * group] + ["o" + str(i) for i in range(rank)])],
        attributes=attrs,
    )
    rng = np.random.default_rng(453)
    np_dtype = np.float32 if dtype == DT.FLOAT else np.float16
    feeds = {"x": (rng.standard_normal(shape) * 0.2).astype(np_dtype),
             "w": (rng.standard_normal(wshape) * 0.2).astype(np_dtype),
             "b": (rng.standard_normal(3 * group) * 0.1).astype(np_dtype)}
    m.assert_op_claimed(model, feeds, "ConvTranspose", rtol=3e-3, atol=3e-4)
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    for xshape in [[2, 4] + [5] * rank, shape]:
        feeds["x"] = (rng.standard_normal(xshape) * 0.2).astype(np_dtype)
        feeds["w"] *= -0.5
        feeds["b"] *= -0.5
        np.testing.assert_allclose(session.run(None, feeds)[0], m.run_cpu(model, feeds)[0],
                                   rtol=3e-3, atol=3e-4)


def _bf16_spatial_model(op, specs, output_shape, attrs):
    inputs = [m.tensor(name, dtype, shape) for name, dtype, shape in specs]
    inner_inputs = []
    nodes = []
    for value in inputs:
        if value.dtype == DT.FLOAT:
            inner = m.tensor(value.name + "_bf", DT.BFLOAT16, value.shape)
            nodes.append(ir.node("Cast", [value], attributes={"to": int(DT.BFLOAT16)},
                                 outputs=[inner]))
        else:
            inner = value
        inner_inputs.append(inner)
    output = m.tensor("result_bf", DT.BFLOAT16, output_shape)
    nodes.append(ir.node(op, inner_inputs, attributes=attrs, outputs=[output]))
    result = m.tensor("result", DT.FLOAT, output_shape)
    nodes.append(ir.node("Cast", [output], attributes={"to": int(DT.FLOAT)}, outputs=[result]))
    graph = ir.Graph(inputs, [result], nodes=nodes, opset_imports={"": 22})
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


@pytest.mark.parametrize("op", ["Conv", "ConvTranspose", "DeformConv", "RoiAlign",
                               "MaxRoiPool", "MaxUnpool", "GlobalAveragePool",
                               "GlobalMaxPool", "GlobalLpPool"])
def test_spatial_bf16_kernels(op, execution_mode, tmp_path):
    import json

    shape = [1, 1, 2, 2]
    x = np.arange(4, dtype=np.float32).reshape(shape)
    specs = [("x", DT.FLOAT, shape)]
    feeds = {"x": x}
    attrs = {}
    expected = x
    if op in {"Conv", "ConvTranspose"}:
        shape = [1, 1, 2, 2, 2]
        feeds["x"] = np.arange(8, dtype=np.float32).reshape(shape)
        specs = [("x", DT.FLOAT, shape), ("w", DT.FLOAT, [1, 1, 1, 1, 1])]
        feeds["w"] = np.ones([1, 1, 1, 1, 1], dtype=np.float32)
        expected = feeds["x"]
    elif op == "DeformConv":
        width = 263
        shape = [1, 1, 1, width]
        x = (np.arange(width) % 7).astype(np.float32).reshape(shape)
        offset = np.zeros([1, 2, 1, width], dtype=np.float32)
        offset[:, 1] = 0.25
        specs = [("x", DT.FLOAT, shape), ("w", DT.FLOAT, [1, 1, 1, 1]),
                 ("offset", DT.FLOAT, list(offset.shape))]
        feeds = {"x": x, "w": np.ones([1, 1, 1, 1], dtype=np.float32), "offset": offset}
        expected = x * 0.75
        expected[..., :-1] += x[..., 1:] * 0.25
    elif op == "RoiAlign":
        specs += [("rois", DT.FLOAT, [1, 4]), ("batch", DT.INT64, [1])]
        feeds.update(rois=np.array([[0, 0, 2, 2]], dtype=np.float32),
                     batch=np.zeros(1, dtype=np.int64))
        attrs = {"sampling_ratio": 2}
        shape = [1, 1, 1, 1]
        expected = np.full(shape, 1.5, dtype=np.float32)
    elif op == "MaxRoiPool":
        specs += [("rois", DT.FLOAT, [1, 5])]
        feeds["rois"] = np.array([[0, 0, 0, 1, 1]], dtype=np.float32)
        attrs = {"pooled_shape": [1, 1]}
        shape = [1, 1, 1, 1]
        expected = np.full(shape, 3, dtype=np.float32)
    elif op == "MaxUnpool":
        specs += [("idx", DT.INT64, [1, 1, 2, 2]), ("shape", DT.INT64, [4])]
        shape = [1, 1, 4, 5]
        feeds.update(idx=np.array([[[[0, 3], [7, 9]]]], dtype=np.int64),
                     shape=np.array(shape, dtype=np.int64))
        attrs = {"kernel_shape": [2, 2], "strides": [2, 2]}
        expected = np.zeros(shape, dtype=np.float32)
        expected.flat[[0, 3, 7, 9]] = x.ravel()
    else:
        shape = [1, 1, 1, 1]
        if op == "GlobalAveragePool":
            value = 1.5
        elif op == "GlobalMaxPool":
            value = 3
        else:
            value = 6
            attrs["p"] = 1
        expected = np.full(shape, value, dtype=np.float32)
    model = _bf16_spatial_model(op, specs, shape, attrs)
    m.assert_matches_ref(model, feeds, [expected], rtol=0, atol=0)
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "bf16_claim")
    session = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    session.run(None, feeds)
    with open(session.end_profiling()) as profile:
        providers = {e.get("args", {}).get("provider") for e in json.load(profile)
                     if e.get("cat") == "Node" and e.get("args", {}).get("provider")}
    assert providers == {"MLXExecutionProvider"}, f"{op} BF16 did not execute wholly on MLX"


@pytest.mark.parametrize("op", ["RoiAlign", "MaxRoiPool", "MaxUnpool",
                               "Conv", "ConvTranspose", "GlobalAveragePool",
                               "GlobalMaxPool", "GlobalLpPool"])
def test_spatial_empty_batch_or_rois(op, execution_mode):
    shape = [0, 1, 2, 2]
    specs = [m.tensor("x", DT.FLOAT, shape)]
    feeds = {"x": np.empty(shape, dtype=np.float32)}
    attrs = {}
    if op in {"RoiAlign", "MaxRoiPool"}:
        specs = [m.tensor("x", DT.FLOAT, [1, 1, 2, 2])]
        feeds["x"] = np.ones([1, 1, 2, 2], dtype=np.float32)
        roi_shape = [0, 4 if op == "RoiAlign" else 5]
        specs += [m.tensor("rois", DT.FLOAT, roi_shape)]
        feeds["rois"] = np.empty(roi_shape, dtype=np.float32)
        if op == "RoiAlign":
            specs += [m.tensor("batch", DT.INT64, [0])]
            feeds["batch"] = np.empty([0], dtype=np.int64)
            attrs = {"sampling_ratio": 2}
        else:
            attrs = {"pooled_shape": [1, 1]}
        shape = [0, 1, 1, 1]
    elif op in {"Conv", "ConvTranspose"}:
        specs += [m.tensor("w", DT.FLOAT, [1, 1, 1, 1])]
        feeds["w"] = np.ones([1, 1, 1, 1], dtype=np.float32)
    elif op == "MaxUnpool":
        specs += [m.tensor("indices", DT.INT64, shape)]
        feeds["indices"] = np.empty(shape, dtype=np.int64)
        attrs = {"kernel_shape": [2, 2], "strides": [2, 2]}
        shape = [0, 1, 4, 4]
    else:
        shape = [0, 1, 1, 1]
    model = m.make_model(op, specs, [m.tensor("y", DT.FLOAT, shape)], attributes=attrs)
    m.assert_matches_ref(model, feeds, [np.empty(shape, dtype=np.float32)], rtol=0, atol=0)
    m.assert_mlx_claims(model, feeds)


@pytest.mark.parametrize("op", ["RoiAlign", "MaxRoiPool", "MaxUnpool"])
def test_vision_symbolic_extents(op, execution_mode):
    inputs = [m.tensor("x", DT.FLOAT, ["n", "c", "h", "w"])]
    attrs = {}
    if op == "MaxUnpool":
        inputs += [m.tensor("indices", DT.INT64, ["n", "c", "h", "w"]),
                   m.tensor("shape", DT.INT64, [4])]
        attrs = {"kernel_shape": [2, 2], "strides": [2, 2]}
    else:
        inputs += [m.tensor("rois", DT.FLOAT, ["r", 4 if op == "RoiAlign" else 5])]
        if op == "RoiAlign":
            inputs += [m.tensor("batch", DT.INT64, ["r"])]
            attrs = {"sampling_ratio": 1}
        else:
            attrs = {"pooled_shape": [1, 1]}
    model = m.make_model(op, inputs,
                         [m.tensor("y", DT.FLOAT, ["a", "b", "oh", "ow"])],
                         attributes=attrs, opset=16)
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    for size, count in [(2, 1), (3, 2), (4, 0), (2, 3)]:
        xshape = [2, 2, size, size]
        feeds = {"x": np.arange(np.prod(xshape), dtype=np.float32).reshape(xshape)}
        if op == "MaxUnpool":
            feeds["indices"] = np.arange(np.prod(xshape), dtype=np.int64).reshape(xshape)
            feeds["shape"] = np.array([2, 2, 2 * size, 2 * size], dtype=np.int64)
        else:
            coords = [0, 0, size - 1, size - 1]
            roi = coords if op == "RoiAlign" else [0, *coords]
            feeds["rois"] = np.array([roi] * count, dtype=np.float32).reshape(count, len(roi))
            if op == "RoiAlign":
                feeds["batch"] = np.zeros(count, dtype=np.int64)
        expected = m.run_cpu(model, feeds)[0]
        np.testing.assert_allclose(session.run(None, feeds)[0], expected, rtol=1e-6, atol=1e-6)
        m.assert_mlx_claims(model, feeds)


def test_grouped_conv3d_is_declined():
    model = m.make_model(
        "Conv", [m.tensor("x", DT.FLOAT, [1, 2, 3, 3, 3]),
                 m.tensor("w", DT.FLOAT, [2, 1, 2, 2, 2])],
        [m.tensor("y", DT.FLOAT, [1, 2, 2, 2, 2])], attributes={"group": 2},
    )
    feeds = {"x": np.ones([1, 2, 3, 3, 3], dtype=np.float32),
             "w": np.ones([2, 1, 2, 2, 2], dtype=np.float32)}
    m.assert_op_declined(model, feeds, "Conv")
