"""Convolution composition and runtime spatial geometry, with live same-session inputs."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx.reference import ReferenceEvaluator

import _models as m

DT = ir.DataType


@pytest.fixture(params=[False, True], ids=["compiled", "eager"])
def execution_mode(request, monkeypatch):
    if request.param:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")


def _claimed_session(model):
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(model, options, providers=["MLXExecutionProvider"])


@pytest.mark.parametrize("op,geometry", [
    (op, geometry) for op in ["Conv", "ConvTranspose"]
    for geometry in ["asymmetric", "SAME_UPPER", "SAME_LOWER", "output_shape"]
    if op == "ConvTranspose" or geometry != "output_shape"
])
@pytest.mark.parametrize("group", [2, 4])
@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16])
def test_grouped_conv3d(op, group, dtype, geometry, execution_mode):
    np_dtype = np.float32 if dtype == DT.FLOAT else np.float16
    attrs = {"group": group, "strides": [2, 1, 2], "dilations": [1, 2, 1]}
    if op == "ConvTranspose":
        attrs["output_padding"] = [1, 0, 1]
    if geometry == "asymmetric":
        attrs["pads"] = [1, 0, 1, 0, 1, 0]
    elif geometry == "output_shape":
        attrs["output_shape"] = [7, 5, 7]
    else:
        attrs["auto_pad"] = geometry
    wshape = [4, 2] + [2] * 3 if op == "ConvTranspose" else [8, 4 // group] + [2] * 3
    oc = 2 * group if op == "ConvTranspose" else 8
    model = m.make_model(
        op, [m.tensor("x", dtype, ["n", 4, "d", "h", "w"]),
             m.tensor("w", dtype, wshape), m.tensor("b", dtype, [oc])],
        [m.tensor("y", dtype, ["n", oc, "od", "oh", "ow"])], attributes=attrs,
    )
    rng = np.random.default_rng(7482)
    feeds = {"w": (rng.standard_normal(wshape) * .2).astype(np_dtype),
             "b": (rng.standard_normal(oc) * .1).astype(np_dtype)}

    def reference():
        ref = model
        if op == "Conv" and geometry.startswith("SAME"):
            pads = []
            for size, stride, dilation in zip(feeds["x"].shape[2:], attrs["strides"],
                                              attrs["dilations"], strict=True):
                total = max(((size + stride - 1) // stride - 1) * stride + dilation + 1 - size, 0)
                begin = total // 2 if geometry == "SAME_UPPER" else (total + 1) // 2
                pads.append((begin, total - begin))
            proto = onnx.load_model_from_string(model)
            del proto.graph.node[0].attribute[:]
            explicit = {**attrs, "auto_pad": "NOTSET",
                        "pads": [v[0] for v in pads] + [v[1] for v in pads]}
            proto.graph.node[0].attribute.extend(
                onnx.helper.make_attribute(k, v) for k, v in explicit.items())
            ref = proto.SerializeToString()
        return m.run_cpu(ref, feeds)[0]

    session = _claimed_session(model)
    for shape in [[1, 4, 4, 4, 4], [2, 4, 5, 3, 5], [1, 4, 4, 4, 4]]:
        feeds["x"] = (rng.standard_normal(shape) * .2).astype(np_dtype)
        feeds["w"] *= -.5
        feeds["b"] *= -.5
        np.testing.assert_allclose(session.run(None, feeds)[0], reference(), rtol=3e-3, atol=3e-4)
    m.assert_mlx_claims(model, feeds)


@pytest.mark.parametrize("group,offset_group", [(1, 2), (2, 1), (2, 4), (4, 2)])
@pytest.mark.parametrize("optional", ["none", "bias", "mask", "both"])
@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16])
def test_dynamic_deform_conv(group, offset_group, optional, dtype, execution_mode):
    inputs = [m.tensor(name, dtype, [name + str(i) for i in range(4)])
              for name in ["x", "w", "offset"]]
    if optional in {"bias", "both"}:
        inputs.append(m.tensor("bias", dtype, ["oc"]))
    elif optional == "mask":
        inputs.append(ir.Value(name="", type=None))
    if optional in {"mask", "both"}:
        inputs.append(m.tensor("mask", dtype, ["mn", "mc", "mh", "mw"]))
    model = m.make_model(
        "DeformConv", inputs, [m.tensor("y", dtype, ["n", "oc", "oh", "ow"])],
        attributes={"group": group, "offset_group": offset_group, "strides": [1, 2],
                    "dilations": [2, 1], "pads": [1, 0, 0, 1]}, opset=22,
    )
    rng = np.random.default_rng(1390)
    np_dtype = np.float32 if dtype == DT.FLOAT else np.float16
    session = _claimed_session(model)
    for batch, channels, height, width, kh, kw in [
        (1, 4, 5, 6, 2, 2), (2, 8, 4, 5, 1, 3), (1, 4, 5, 6, 2, 2),
    ]:
        oc = group * 2
        oh, ow = height + 1 - 2 * (kh - 1), (width + 1 - kw) // 2 + 1
        feeds = {
            "x": (rng.standard_normal([batch, channels, height, width]) * .2).astype(np_dtype),
            "w": (rng.standard_normal([oc, channels // group, kh, kw]) * .2).astype(np_dtype),
            "offset": rng.uniform(-.6, .6, [batch, 2 * offset_group * kh * kw, oh, ow]).astype(np_dtype),
        }
        if optional in {"bias", "both"}:
            feeds["bias"] = (rng.standard_normal(oc) * .1).astype(np_dtype)
        if optional in {"mask", "both"}:
            feeds["mask"] = rng.uniform(-.5, 1, [batch, offset_group * kh * kw, oh, ow]).astype(np_dtype)
        for _ in range(2):
            np.testing.assert_allclose(session.run(None, feeds)[0], m.run_cpu(model, feeds)[0],
                                       rtol=3e-3, atol=3e-4)
            feeds["offset"] *= -.5
            feeds["w"] *= -.5
            if "mask" in feeds:
                feeds["mask"] *= -.5
    m.assert_mlx_claims(model, feeds)


def _col2im_model(rank, dtype, controls, attrs=None):
    data = m.tensor("data", dtype, ["n", "ck", "length"])
    image = m.tensor("image", DT.INT64, [rank])
    block = m.tensor("block", DT.INT64, [rank])
    output = m.tensor("y", dtype, ["n", "c"] + ["s" + str(i) for i in range(rank)])
    nodes, inputs, initializers = [], [data], []
    if controls == "constant":
        image.const_value = ir.tensor(np.array([4] * rank, dtype=np.int64))
        block.const_value = ir.tensor(np.array([2] * rank, dtype=np.int64))
        initializers = [image, block]
    elif controls == "intermediate":
        delta = m.tensor("delta", DT.INT64, [rank])
        inputs += [image, block, delta]
        image_live = m.tensor("image_live", DT.INT64, [rank])
        block_live = m.tensor("block_live", DT.INT64, [rank])
        nodes += [ir.node("Add", [image, delta], outputs=[image_live]),
                  ir.node("Add", [block, delta], outputs=[block_live])]
        image, block = image_live, block_live
    else:
        inputs += [image, block]
    nodes.append(ir.node("Col2Im", [data, image, block], attributes=attrs or {}, outputs=[output]))
    graph = ir.Graph(inputs, [output], nodes=nodes, initializers=initializers, opset_imports={"": 18})
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def _col2im_ref(data, image, block, strides, dilations, pads):
    rank = len(image)
    npos = [(image[d] + pads[d] + pads[d + rank] - dilations[d] * (block[d] - 1) - 1)
            // strides[d] + 1 for d in range(rank)]
    channels = data.shape[1] // int(np.prod(block))
    output = np.zeros([data.shape[0], channels, *image], dtype=data.dtype)
    for k, kernel in enumerate(itertools.product(*(range(v) for v in block))):
        for l, position in enumerate(itertools.product(*(range(v) for v in npos))):
            coord = tuple(position[d] * strides[d] + kernel[d] * dilations[d] - pads[d]
                          for d in range(rank))
            if all(0 <= coord[d] < image[d] for d in range(rank)):
                output[(slice(None), slice(None), *coord)] += data[:, k::int(np.prod(block)), l]
    return output


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("controls", ["constant", "live", "intermediate"])
@pytest.mark.parametrize("dtype,np_dtype", [(DT.FLOAT, np.float32), (DT.FLOAT16, np.float16)])
def test_col2im_runtime_geometry(rank, controls, dtype, np_dtype, execution_mode):
    model = _col2im_model(rank, dtype, controls)
    session = _claimed_session(model)
    rng = np.random.default_rng(305)
    for batch, channels, image, block in [
        (1, 2, [4] * rank, [2] * rank),
        (2, 3, [5] * rank, [3] * rank),
        (1, 2, [4] * rank, [2] * rank),
    ]:
        if controls == "constant":
            image, block = [4] * rank, [2] * rank
        length = int(np.prod(np.array(image) - block + 1))
        data = (rng.integers(-8, 9, [batch, channels * int(np.prod(block)), length]) / 4).astype(np_dtype)
        feeds = {"data": data}
        if controls != "constant":
            feeds |= {"image": np.array(image, dtype=np.int64), "block": np.array(block, dtype=np.int64)}
        if controls == "intermediate":
            feeds["delta"] = np.zeros(rank, dtype=np.int64)
        expected = _col2im_ref(data, image, block, [1] * rank, [1] * rank, [0] * (2 * rank))
        np.testing.assert_array_equal(session.run(None, feeds)[0], expected)
    m.assert_mlx_claims(model, feeds)


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("controls", ["live", "intermediate"])
def test_col2im_same_shape_control_changes(rank, controls, execution_mode):
    model = _col2im_model(rank, DT.FLOAT, controls)
    session = _claimed_session(model)
    data = np.ones([1, 6**rank, 3**rank], dtype=np.float32)
    for delta in [1, 2, 1]:
        image, block = [3 + delta] * rank, [1 + delta] * rank
        feeds = {"data": data, "image": np.array(image, dtype=np.int64),
                 "block": np.array(block, dtype=np.int64)}
        if controls == "intermediate":
            feeds["image"].fill(3)
            feeds["block"].fill(1)
            feeds["delta"] = np.full(rank, delta, dtype=np.int64)
        expected = _col2im_ref(data, image, block, [1] * rank, [1] * rank, [0] * (2 * rank))
        np.testing.assert_array_equal(session.run(None, feeds)[0], expected)
        data *= -.5


@pytest.mark.parametrize("np_dtype", [np.float32, np.float16, np.float64, np.uint8, np.uint16, np.uint32, np.uint64,
                                     np.int8, np.int16, np.int32, np.int64, np.bool_])
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_col2im_exact_payloads(rank, np_dtype, execution_mode):
    dtype = ir.DataType.from_numpy(np.dtype(np_dtype))
    attrs = {"strides": [2] * rank, "dilations": [2] * rank, "pads": [1] * rank + [2] * rank}
    model = _col2im_model(rank, dtype, "live", attrs)
    session = _claimed_session(model)
    for image, block in [([4] * rank, [2] * rank), ([6] * rank, [3] * rank)]:
        npos = [(i + 3 - 2 * (k - 1) - 1) // 2 + 1 for i, k in zip(image, block, strict=True)]
        shape = [2, 2 * int(np.prod(block)), int(np.prod(npos))]
        data = (np.arange(int(np.prod(shape))).reshape(shape) % 7).astype(np_dtype)
        if np.issubdtype(np_dtype, np.integer) and np.dtype(np_dtype).itemsize >= 4:
            data += np_dtype(1 << (55 if np.dtype(np_dtype).itemsize == 8 else 25))
        elif np_dtype == np.float64:
            data += 2.0**-40
        if np.issubdtype(np_dtype, np.signedinteger):
            data[:, ::2] *= -1
        feeds = {"data": data, "image": np.array(image, dtype=np.int64),
                 "block": np.array(block, dtype=np.int64)}
        expected = _col2im_ref(data, image, block, [2] * rank, [2] * rank, [1] * rank + [2] * rank)
        np.testing.assert_array_equal(session.run(None, feeds)[0], expected)
        # Independently confirm the additive oracle against ONNX's current reference.
        np.testing.assert_array_equal(ReferenceEvaluator(model).run(None, feeds)[0], expected)
    m.assert_mlx_claims(model, feeds)


@pytest.mark.parametrize("bad_image,bad_block,shape,error", [
    ([0], [2], [1, 2, 3], "invalid dimensions"),
    ([4], [0], [1, 2, 3], "invalid dimensions"),
    ([4], [3], [1, 2, 2], "channels must"),
    ([4], [2], [1, 2, 2], "input length"),
    ([1], [3], [1, 3, 1], "smaller than"),
    ([2**40], [2], [1, 2, 3], "invalid dimensions"),
])
def test_col2im_invalid_geometry_recovers(bad_image, bad_block, shape, error, execution_mode):
    model = _col2im_model(1, DT.FLOAT, "live")
    session = _claimed_session(model)
    session.disable_fallback()
    feeds = {"data": np.ones(shape, dtype=np.float32), "image": np.array(bad_image, dtype=np.int64),
             "block": np.array(bad_block, dtype=np.int64)}
    with pytest.raises(Exception, match="Col2Im:.*" + error):
        session.run(None, feeds)
    feeds = {"data": np.ones([1, 2, 3], dtype=np.float32), "image": np.array([4], dtype=np.int64),
             "block": np.array([2], dtype=np.int64)}
    np.testing.assert_array_equal(session.run(None, feeds)[0], [[[1, 2, 2, 1]]])


@pytest.mark.parametrize("bad", ["channels", "offset", "mask", "bias", "spatial"])
def test_deform_invalid_geometry_recovers(bad, execution_mode):
    model = m.make_model(
        "DeformConv",
        [m.tensor(name, DT.FLOAT, [name + str(i) for i in range(4)])
         for name in ["x", "w", "offset"]]
        + [m.tensor("bias", DT.FLOAT, ["oc"]),
           m.tensor("mask", DT.FLOAT, ["mn", "mc", "mh", "mw"])],
        [m.tensor("y", DT.FLOAT, ["n", "c", "h", "w"])],
        attributes={"group": 2, "offset_group": 1}, opset=22,
    )
    session = _claimed_session(model)
    session.disable_fallback()
    feeds = {"x": np.ones([1, 4, 3, 3], dtype=np.float32),
             "w": np.ones([4, 2, 1, 1], dtype=np.float32),
             "offset": np.zeros([1, 2, 3, 3], dtype=np.float32),
             "bias": np.ones([4], dtype=np.float32),
             "mask": np.ones([1, 1, 3, 3], dtype=np.float32)}
    key, shape = {
        "channels": ("w", [4, 1, 1, 1]), "offset": ("offset", [1, 4, 3, 3]),
        "mask": ("mask", [1, 2, 3, 3]), "bias": ("bias", [3]),
        "spatial": ("x", [1, 4, 0, 3]),
    }[bad]
    invalid = {**feeds, key: np.ones(shape, dtype=np.float32)}
    with pytest.raises(Exception, match="DeformConv:"):
        session.run(None, invalid)
    np.testing.assert_array_equal(session.run(None, feeds)[0], np.full([1, 4, 3, 3], 3))


@pytest.mark.parametrize("op", ["DeformConv", "Col2Im"])
def test_empty_batch(op, execution_mode):
    if op == "Col2Im":
        model = _col2im_model(2, DT.FLOAT, "live")
        feeds = {"data": np.empty([0, 8, 9], dtype=np.float32),
                 "image": np.array([4, 4], dtype=np.int64),
                 "block": np.array([2, 2], dtype=np.int64)}
        shape = [0, 2, 4, 4]
    else:
        model = m.make_model(
            op, [m.tensor(name, DT.FLOAT, [name + str(i) for i in range(4)])
                 for name in ["x", "w", "offset"]],
            [m.tensor("y", DT.FLOAT, ["n", "c", "h", "w"])], opset=22,
        )
        feeds = {"x": np.empty([0, 2, 3, 3], dtype=np.float32),
                 "w": np.ones([4, 2, 1, 1], dtype=np.float32),
                 "offset": np.empty([0, 2, 3, 3], dtype=np.float32)}
        shape = [0, 4, 3, 3]
    session = _claimed_session(model)
    assert session.run(None, feeds)[0].shape == tuple(shape)


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("dtype,np_dtype", [(DT.FLOAT, np.float32), (DT.INT64, np.int64)])
def test_col2im_empty_contributions(rank, dtype, np_dtype, execution_mode):
    for image, block, attrs, length in [
        ([1] * rank, [2] * rank, {}, 0),
        ([1] * rank, [2] * rank,
         {"dilations": [2] * rank, "pads": [1] * (2 * rank)}, 1),
    ]:
        model = _col2im_model(rank, dtype, "live", attrs)
        feeds = {"data": np.ones([1, 2**rank, length], dtype=np_dtype),
                 "image": np.array(image, dtype=np.int64), "block": np.array(block, dtype=np.int64)}
        expected = np.zeros([1, 1] + image, dtype=np_dtype)
        np.testing.assert_array_equal(ReferenceEvaluator(model).run(None, feeds)[0], expected)
        np.testing.assert_array_equal(_claimed_session(model).run(None, feeds)[0], expected)


@pytest.mark.parametrize("op", ["Conv", "ConvTranspose", "DeformConv", "Col2Im"])
def test_bf16_runtime_geometry(op, execution_mode):
    from test_spatial_semantics import _bf16_spatial_model

    if op in {"Conv", "ConvTranspose"}:
        specs = [("x", DT.FLOAT, ["n", 4, "d", "h", "w"]),
                 ("w", DT.FLOAT, [4, 2, 1, 1, 1])]
        model = _bf16_spatial_model(op, specs, ["n", 4, "d", "h", "w"], {"group": 2})
        feeds = {"x": np.arange(4 * 3 * 3 * 3, dtype=np.float32).reshape([1, 4, 3, 3, 3]) % 7,
                 "w": np.ones([4, 2, 1, 1, 1], dtype=np.float32)}
        expected = np.repeat(feeds["x"].reshape(1, 2, 2, 3, 3, 3).sum(axis=2), 2, axis=1)
    elif op == "DeformConv":
        model = _bf16_spatial_model(
            op, [("x", DT.FLOAT, ["n", "c", "h", "w"]),
                 ("w", DT.FLOAT, [1, 1, 1, 1]),
                 ("offset", DT.FLOAT, ["n", 2, "h", "w"]),
                 ("bias", DT.FLOAT, [1]),
                 ("mask", DT.FLOAT, ["n", 1, "h", "w"])],
            ["n", 1, "h", "w"], {},
        )
        x = (np.arange(261, dtype=np.float32) % 7).reshape(1, 1, 1, 261)
        feeds = {"x": x, "w": np.ones([1, 1, 1, 1], dtype=np.float32),
                 "offset": np.zeros([1, 2, 1, 261], dtype=np.float32),
                 "bias": np.zeros([1], dtype=np.float32), "mask": np.ones_like(x)}
        expected = x
    else:
        model = _bf16_spatial_model(
            op, [("data", DT.FLOAT, ["n", "ck", "l"]),
                 ("image", DT.INT64, [1]), ("block", DT.INT64, [1])],
            ["n", "c", "s"], {},
        )
        feeds = {"data": np.ones([1, 2, 3], dtype=np.float32),
                 "image": np.array([4], dtype=np.int64), "block": np.array([2], dtype=np.int64)}
        expected = np.array([[[1, 2, 2, 1]]], dtype=np.float32)
    m.assert_matches_ref(model, feeds, [expected], rtol=0, atol=0)
    np.testing.assert_array_equal(_claimed_session(model).run(None, feeds)[0], expected)
    m.assert_mlx_claims(model, feeds)


def test_geometry_compile_routes(tmp_path):
    trace = tmp_path / "conv_geometry.json"
    selectors = [
        "test_grouped_conv3d[compiled-FLOAT-2-Conv-asymmetric]",
        "test_grouped_conv3d[compiled-FLOAT-2-ConvTranspose-asymmetric]",
        "test_dynamic_deform_conv[compiled-FLOAT-none-1-2]",
        "test_col2im_runtime_geometry[compiled-FLOAT-float32-constant-1]",
        "test_col2im_runtime_geometry[compiled-FLOAT-float32-live-1]",
        "test_col2im_runtime_geometry[compiled-FLOAT-float32-intermediate-1]",
    ]
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         *[f"{Path(__file__).name}::{selector}" for selector in selectors]],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0", "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    paths = [e["args"] for e in json.loads(trace.read_text()) if e.get("cat") == "ep.path"]
    for op in ["Conv", "ConvTranspose", "DeformConv"]:
        actual = [e for e in paths if e["ops"] == op]
        assert actual and {e["path"] for e in actual} == {"general"}
        assert {"MISS", "HIT", "RETRACE"} <= {e["cache"] for e in actual}
    col = [e for e in paths if "Col2Im" in e["ops"]]
    assert {e["path"] for e in col} == {"general", "eager"}
    assert {"MISS", "HIT", "RETRACE", "n/a"} <= {e["cache"] for e in col}
