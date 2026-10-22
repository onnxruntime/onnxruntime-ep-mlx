"""Window-pooling geometry, exact original-coordinate indices, and compiled replay."""

from __future__ import annotations

import itertools

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import onnx_ir as ir
import onnxruntime as ort
import pytest

import _models as m

DT = ir.DataType


@pytest.fixture(params=[False, True], ids=["compiled", "eager"])
def execution_mode(request, monkeypatch):
    if request.param:
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")


def pool_model(op, shape, attrs, dtype=DT.FLOAT, *, indices=False, opset=22):
    outputs = [m.tensor("y", dtype, [f"y{i}" for i in range(len(shape))])]
    if indices:
        outputs.append(m.tensor("indices", DT.INT64, [f"y{i}" for i in range(len(shape))]))
    return m.make_model(
        op, [m.tensor("x", dtype, shape)], outputs, attributes=attrs, opset=opset
    )


def geometry(rank, case):
    attrs = {"kernel_shape": [3] * rank, "strides": [2] * rank}
    if case == "floor":
        pass
    elif case == "asymmetric":
        attrs["pads"] = [1] * rank + [0] * rank
    elif case == "ceil":
        attrs["ceil_mode"] = 1
    elif case == "last-window":
        attrs.update(kernel_shape=[2] * rank, pads=[0] * rank + [1] * rank, ceil_mode=1)
    elif case == "dilated":
        attrs.update(dilations=[2] * rank, pads=[1] * rank + [2] * rank)
    elif case == "ceil-dilated":
        attrs.update(dilations=[2] * rank, ceil_mode=1)
    elif case == "dilated-overlap":
        attrs.update(dilations=[2] * rank, strides=[1] * rank)
    elif case in ("SAME_UPPER", "SAME_LOWER", "VALID"):
        attrs["auto_pad"] = case
    elif case == "same-dilated":
        attrs.update(dilations=[2] * rank, auto_pad="SAME_LOWER")
    elif case == "valid-ceil":
        attrs.update(auto_pad="VALID", ceil_mode=1)
    else:
        raise AssertionError(case)
    return attrs


def assert_outputs(actual, expected, *, exact=False, fp16=False):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got.shape == want.shape
        assert got.dtype == want.dtype
        if exact or want.dtype == np.int64:
            np.testing.assert_array_equal(got, want)
        else:
            np.testing.assert_allclose(
                got, want, rtol=2e-2 if fp16 else 1e-5, atol=2e-2 if fp16 else 1e-6
            )


def pool_reference(op, x, attrs, *, indices=False):
    """Scalar ONNX-coordinate oracle; no padded arrays or strided window views.

    ONNX 1.23.2's LpPool reference rescales a padded mean, and its common ceil
    helper can move extra tail padding to the beginning. Neither is an oracle
    for those interactions, so cross-check ORT CPU with the coordinate definition.
    """
    rank = x.ndim - 2
    kernel = attrs["kernel_shape"]
    strides = attrs.get("strides", [1] * rank)
    dilations = attrs.get("dilations", [1] * rank)
    pads = list(attrs.get("pads", [0] * (2 * rank)))
    auto_pad = attrs.get("auto_pad", "NOTSET")
    effective = [d * (k - 1) + 1 for k, d in zip(kernel, dilations, strict=True)]
    spatial = x.shape[2:]
    out = []
    for i in range(rank):
        if auto_pad.startswith("SAME"):
            size = (spatial[i] + strides[i] - 1) // strides[i]
            total = max(0, (size - 1) * strides[i] + effective[i] - spatial[i])
            pads[i] = (total + (auto_pad == "SAME_LOWER")) // 2
            pads[rank + i] = total - pads[i]
        else:
            numerator = spatial[i] + pads[i] + pads[rank + i] - effective[i]
            size = numerator // strides[i] + 1
            if attrs.get("ceil_mode", 0) and numerator % strides[i]:
                size += 1
        if attrs.get("ceil_mode", 0) and (size - 1) * strides[i] >= spatial[i] + pads[i]:
            size -= 1
        out.append(size)
    values = np.empty((*x.shape[:2], *out), dtype=x.dtype)
    locations = np.full(values.shape, -1, dtype=np.int64)
    for nc in np.ndindex(x.shape[:2]):
        base = (nc[0] * x.shape[1] + nc[1]) * int(np.prod(spatial))
        for position in np.ndindex(*out):
            samples = []
            valid = []
            padded_count = 0
            for offset in itertools.product(*(range(k) for k in kernel)):
                coord = tuple(
                    position[i] * strides[i] - pads[i] + offset[i] * dilations[i]
                    for i in range(rank)
                )
                if all(-pads[i] <= coord[i] < spatial[i] + pads[rank + i] for i in range(rank)):
                    padded_count += 1
                if all(0 <= coord[i] < spatial[i] for i in range(rank)):
                    samples.append(float(x[(*nc, *coord)]))
                    valid.append(coord)
            target = (*nc, *position)
            if op == "AveragePool":
                count = padded_count if attrs.get("count_include_pad", 0) else len(samples)
                values[target] = sum(samples) / count if count else np.nan
            elif op == "LpPool":
                p = attrs.get("p", 2)
                values[target] = sum(abs(v) ** p for v in samples) ** (1 / p)
            elif samples:
                best = max(range(len(samples)), key=samples.__getitem__)
                values[target] = samples[best]
                locations[target] = base + np.ravel_multi_index(
                    valid[best], spatial, order="F" if attrs.get("storage_order", 0) else "C"
                )
            else:
                values[target] = (np.finfo(x.dtype).min if x.dtype.kind == "f"
                                  else np.iinfo(x.dtype).min)
    return [values, locations] if indices else [values]


def cpu_pool(op, x, attrs, dtype=DT.FLOAT, *, indices=False):
    attrs = dict(attrs)
    auto_pad = attrs.get("auto_pad", "NOTSET")
    if auto_pad.startswith("SAME") and any(d != 1 for d in attrs.get("dilations", [])):
        # ORT 1.31 computes SAME padding from the undilated kernel, disagreeing
        # with both the schema and ONNX shape inference. Equivalent explicit pads
        # exercise its actual dilated pooling kernels without inheriting that bug.
        rank = x.ndim - 2
        strides = attrs.get("strides", [1] * rank)
        pads = []
        ends = []
        for size, kernel, stride, dilation in zip(
            x.shape[2:], attrs["kernel_shape"], strides, attrs["dilations"], strict=True
        ):
            output = (size + stride - 1) // stride
            total = max(0, (output - 1) * stride + dilation * (kernel - 1) + 1 - size)
            begin = (total + (auto_pad == "SAME_LOWER")) // 2
            pads.append(begin)
            ends.append(total - begin)
        del attrs["auto_pad"]
        attrs["pads"] = pads + ends
    model = pool_model(op, list(x.shape), attrs, dtype, indices=indices)
    return m.run_cpu(model, {"x": x})


@pytest.mark.parametrize("op", ["AveragePool", "MaxPool", "LpPool"])
@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16])
@pytest.mark.parametrize(
    "case",
    ["floor", "asymmetric", "ceil", "last-window", "dilated", "ceil-dilated", "dilated-overlap",
     "SAME_UPPER", "SAME_LOWER", "VALID", "same-dilated", "valid-ceil"],
)
def test_pool_geometry_matrix(op, rank, dtype, case, execution_mode):
    shape = [2, 2] + [6 + i for i in range(rank)]
    attrs = geometry(rank, case)
    if op == "LpPool":
        attrs["p"] = 3
    model = pool_model(op, shape, attrs, dtype, indices=op == "MaxPool")
    np_dtype = np.float16 if dtype == DT.FLOAT16 else np.float32
    x = np.random.default_rng(93).uniform(-1, 1, shape).astype(np_dtype)
    feeds = {"x": x}
    expected = cpu_pool(op, x, attrs, dtype, indices=op == "MaxPool")
    assert_outputs(expected, pool_reference(op, x, attrs, indices=op == "MaxPool"),
                   exact=op == "MaxPool", fp16=dtype == DT.FLOAT16)
    assert_outputs(m.run_mlx(model, feeds), expected, exact=op == "MaxPool",
                   fp16=dtype == DT.FLOAT16)
    m.assert_mlx_claims(model, feeds)


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("include_pad", [0, 1])
@pytest.mark.parametrize("dilation", [1, 2])
def test_average_explicit_padding_vs_ceil_extension(rank, include_pad, dilation, execution_mode):
    shape = [1, 2] + [6] * rank
    attrs = {
        "kernel_shape": [3] * rank, "strides": [3] * rank,
        "pads": [1] * rank + [0] * rank, "ceil_mode": 1,
        "dilations": [dilation] * rank, "count_include_pad": include_pad,
    }
    model = pool_model("AveragePool", shape, attrs)
    feeds = {"x": np.full(shape, 6, dtype=np.float32)}
    assert_outputs(m.run_mlx(model, feeds), m.run_cpu(model, feeds))
    m.assert_mlx_claims(model, feeds)


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("storage_order", [0, 1])
@pytest.mark.parametrize(
    "dtype,fill", [(DT.FLOAT, -np.inf), (DT.FLOAT16, -np.inf),
                   (DT.INT8, -128), (DT.UINT8, 0)]
)
def test_max_indices_ties_extremes_and_storage(rank, storage_order, dtype, fill, execution_mode):
    shape = [2, 3] + [4 + i for i in range(rank)]
    attrs = {
        "kernel_shape": [3] * rank, "strides": [2] * rank,
        "pads": [1] * rank + [1] * rank, "ceil_mode": 1,
        "storage_order": storage_order,
    }
    model = pool_model("MaxPool", shape, attrs, dtype, indices=True)
    np_dtype = {DT.FLOAT: np.float32, DT.FLOAT16: np.float16,
                DT.INT8: np.int8, DT.UINT8: np.uint8}[dtype]
    feeds = {"x": np.full(shape, fill, dtype=np_dtype)}
    assert_outputs(m.run_mlx(model, feeds), m.run_cpu(model, feeds), exact=True)
    m.assert_mlx_claims(model, feeds)


@pytest.mark.parametrize("dtype", [DT.INT8, DT.UINT8])
@pytest.mark.parametrize("indices", [False, True])
def test_max_integer_dilated_kernels(dtype, indices, execution_mode):
    shape = [2, 3, 5, 7, 6]
    model = pool_model(
        "MaxPool", shape, {"kernel_shape": [2, 3, 2], "dilations": [2, 2, 1],
                          "strides": [2, 1, 3], "pads": [1, 1, 0, 0, 1, 1],
                          "ceil_mode": 1, "storage_order": 1}, dtype, indices=indices
    )
    x = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    x = ((x % 256) - (128 if dtype == DT.INT8 else 0)).astype(
        np.int8 if dtype == DT.INT8 else np.uint8
    )
    feeds = {"x": x}
    assert_outputs(m.run_mlx(model, feeds), m.run_cpu(model, feeds), exact=True)
    m.assert_mlx_claims(model, feeds)


@pytest.mark.parametrize("op", ["AveragePool", "MaxPool", "LpPool"])
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_dynamic_shape_and_value_replay(op, rank, execution_mode):
    attrs = geometry(rank, "same-dilated")
    model = pool_model(op, ["n", "c"] + [f"s{i}" for i in range(rank)], attrs,
                       indices=op == "MaxPool")
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    for size, channels, offset in [(6, 2, 0), (6, 2, 7), (8, 3, -4), (6, 2, 1)]:
        shape = [2, channels] + [size + i for i in range(rank)]
        x = np.random.default_rng(33).uniform(-1, 1, shape).astype(np.float32) + offset
        feeds = {"x": x}
        expected = cpu_pool(op, x, attrs, indices=op == "MaxPool")
        assert_outputs(session.run(None, feeds), expected, exact=op == "MaxPool")
    m.assert_mlx_claims(model, feeds)


@pytest.mark.parametrize(
    "op,opset,attrs",
    [("AveragePool", 1, {}), ("AveragePool", 7, {"count_include_pad": 1}),
     ("AveragePool", 10, {"ceil_mode": 1}), ("AveragePool", 19, {"dilations": [2]}),
     ("MaxPool", 1, {}), ("MaxPool", 8, {"storage_order": 1}),
     ("MaxPool", 10, {"ceil_mode": 1, "dilations": [2]}),
     ("LpPool", 1, {"p": 1.5}), ("LpPool", 2, {"p": 3}),
     ("LpPool", 18, {"ceil_mode": 1, "dilations": [2]})],
)
def test_schema_revisions_and_defaults(op, opset, attrs, execution_mode):
    model = pool_model(op, [1, 2, 7], {"kernel_shape": [3], **attrs}, opset=opset,
                       indices=op == "MaxPool" and opset >= 8)
    feeds = {"x": np.linspace(-2, 2, 14, dtype=np.float32).reshape(1, 2, 7)}
    # ORT 1.31 has no AveragePool-1 / LpPool-1 CPU kernel. These unpadded
    # legacy defaults have a working official reference, including float p in LpPool-1.
    expected = (ReferenceEvaluator(onnx.load_model_from_string(model)).run(None, feeds)
                if opset == 1 and op != "MaxPool" else m.run_cpu(model, feeds))
    assert_outputs(m.run_mlx(model, feeds), expected, exact=op == "MaxPool")
    m.assert_mlx_claims(model, feeds)


@pytest.mark.parametrize("op", ["AveragePool", "MaxPool", "LpPool"])
def test_invalid_runtime_geometry_then_recovery(op, execution_mode):
    model = pool_model(op, [1, 2, "width"], {"kernel_shape": [5], "dilations": [2]})
    session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)
    session.disable_fallback()
    valid = {"x": np.ones([1, 2, 11], dtype=np.float32)}
    assert_outputs(session.run(None, valid), m.run_cpu(model, valid), exact=op == "MaxPool")
    with pytest.raises(Exception, match="Pool:"):
        session.run(None, {"x": np.ones([1, 2, 1], dtype=np.float32)})
    assert_outputs(session.run(None, valid), m.run_cpu(model, valid), exact=op == "MaxPool")


@pytest.mark.parametrize("op", ["AveragePool", "MaxPool", "LpPool"])
def test_bf16_pool_composed_claim(op, execution_mode):
    shape = [2, 2, 6, 7, 5]
    attrs = {"kernel_shape": [2, 3, 2], "pads": [1, 1, 0, 0, 1, 1],
             "strides": [2, 2, 2], "dilations": [2, 1, 2], "ceil_mode": 1}
    fp32 = pool_model(op, shape, attrs)
    x = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) % 7) / 4
    expected = m.run_cpu(fp32, {"x": x})[0]
    model = m.bf16_interior_model(op, [("x", shape)], list(expected.shape), attributes=attrs)
    # Every op, including both boundary casts, must be claimed; no CPU bf16 kernel is assumed.
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"])
    assert_outputs(session.run(None, {"x": x}), [expected], fp16=True)


def test_pool_schemas_match_pinned_onnx():
    for op, first_dilation in [("AveragePool", 19), ("MaxPool", 10), ("LpPool", 18)]:
        assert "dilations" not in onnx.defs.get_schema(op, first_dilation - 1).attributes
        assert "dilations" in onnx.defs.get_schema(op, first_dilation).attributes
        assert "tensor(bfloat16)" in onnx.defs.get_schema(op, 22).type_constraints[0].allowed_type_strs


@pytest.mark.parametrize("op", ["AveragePool", "MaxPool", "LpPool"])
@pytest.mark.parametrize("shape", [[0, 2, 5], [1, 2, 2], [0, 2, 4, 5, 6]])
def test_legal_empty_outputs(op, shape, execution_mode):
    attrs = {"kernel_shape": [3] * (len(shape) - 2)}
    model = pool_model(op, shape, attrs, indices=op == "MaxPool")
    feeds = {"x": np.empty(shape, dtype=np.float32)}
    assert_outputs(m.run_mlx(model, feeds), m.run_cpu(model, feeds), exact=True)
    # Profiling an empty batch can omit all kernel events; output-zero with a nonempty
    # input still proves the formerly rejected kernel-larger-than-input form is claimed.
    if shape[0]:
        m.assert_mlx_claims(model, feeds)


@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16, DT.INT8, DT.UINT8])
@pytest.mark.parametrize("storage_order", [0, 1])
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_max_empty_dilated_window(dtype, storage_order, rank, execution_mode):
    shape = [2, 2] + [1] * rank
    model = pool_model(
        "MaxPool", shape, {"kernel_shape": [2] * rank, "dilations": [2] * rank,
                          "pads": [1] * (2 * rank), "storage_order": storage_order},
        dtype, indices=True
    )
    np_dtype = {DT.FLOAT: np.float32, DT.FLOAT16: np.float16,
                DT.INT8: np.int8, DT.UINT8: np.uint8}[dtype]
    feeds = {"x": np.ones(shape, dtype=np_dtype)}
    assert_outputs(m.run_mlx(model, feeds), m.run_cpu(model, feeds), exact=True)
    m.assert_mlx_claims(model, feeds)


def test_max_indices_beyond_float32_exact_integer_range(execution_mode):
    shape = [2, 129, 65537]
    model = pool_model(
        "MaxPool", shape, {"kernel_shape": [65537], "strides": [65537]},
        DT.UINT8, indices=True
    )
    x = np.zeros(shape, dtype=np.uint8)
    x[..., -1] = 255
    expected_values = np.full([2, 129, 1], 255, dtype=np.uint8)
    expected_indices = ((np.arange(258, dtype=np.int64) + 1) * 65537 - 1).reshape(2, 129, 1)
    assert expected_indices.max() > 2**24
    assert_outputs(m.run_mlx(model, {"x": x}), [expected_values, expected_indices], exact=True)
    assert_outputs(m.run_cpu(model, {"x": x}), [expected_values, expected_indices], exact=True)
    m.assert_mlx_claims(model, {"x": x})


@pytest.mark.parametrize("storage_order", [0, 1])
def test_max_ties_follow_original_traversal_not_storage_encoding(storage_order, execution_mode):
    x = np.array([[[[0, 5], [5, 0]]]], dtype=np.float32)
    attrs = {"kernel_shape": [2, 2], "storage_order": storage_order}
    model = pool_model("MaxPool", list(x.shape), attrs, indices=True)
    expected = [np.array([[[[5]]]], dtype=np.float32),
                np.array([[[[1 if storage_order == 0 else 2]]]], dtype=np.int64)]
    assert_outputs(m.run_cpu(model, {"x": x}), expected, exact=True)
    assert_outputs(m.run_mlx(model, {"x": x}), expected, exact=True)
    m.assert_mlx_claims(model, {"x": x})


@pytest.mark.parametrize("p", [0, -1, -2])
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_lp_nonpositive_exponent_excludes_padding(p, rank, execution_mode):
    shape = [1, 2] + [4] * rank
    attrs = {"kernel_shape": [3] * rank, "pads": [1] * (2 * rank), "p": p}
    model = pool_model("LpPool", shape, attrs)
    feeds = {"x": np.linspace(0.5, 2, np.prod(shape), dtype=np.float32).reshape(shape)}
    assert_outputs(m.run_mlx(model, feeds), m.run_cpu(model, feeds))
    m.assert_mlx_claims(model, feeds)


@pytest.mark.parametrize("op", ["AveragePool", "MaxPool", "LpPool"])
@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("layout", ["transpose", "broadcast", "slice"])
def test_pool_noncontiguous_and_broadcast_input(op, rank, layout, execution_mode):
    shape = [2, 2] + [6 + i for i in range(rank)]
    np_dtype = np.float32
    target = m.tensor("pool_input", DT.FLOAT, shape)
    constants = []
    if layout == "transpose":
        raw_shape = shape[:2] + shape[:1:-1]
        x = m.tensor("x", DT.FLOAT, raw_shape)
        producer = ir.node(
            "Transpose", [x], attributes={"perm": [0, 1] + list(range(rank + 1, 1, -1))},
            outputs=[target]
        )
    elif layout == "broadcast":
        raw_shape = [1, 2] + [1] * rank
        x = m.tensor("x", DT.FLOAT, raw_shape)
        expanded = m.tensor("shape", DT.INT64, [rank + 2])
        expanded.const_value = ir.tensor(np.array(shape, dtype=np.int64))
        constants.append(expanded)
        producer = ir.node("Expand", [x, expanded], outputs=[target])
    else:
        raw_shape = shape[:2] + [2 * d for d in shape[2:]]
        x = m.tensor("x", DT.FLOAT, raw_shape)
        for name, values in [
            ("starts", [0] * rank), ("ends", raw_shape[2:]),
            ("axes", list(range(2, rank + 2))), ("steps", [2] * rank)
        ]:
            value = m.tensor(name, DT.INT64, [rank])
            value.const_value = ir.tensor(np.array(values, dtype=np.int64))
            constants.append(value)
        producer = ir.node("Slice", [x, *constants], outputs=[target])
    outputs = [m.tensor("y", DT.FLOAT, [f"y{i}" for i in range(rank + 2)])]
    if op == "MaxPool":
        outputs.append(m.tensor("indices", DT.INT64, [f"y{i}" for i in range(rank + 2)]))
    pool = ir.node(op, [target], attributes=geometry(rank, "ceil-dilated"), outputs=outputs)
    graph = ir.Graph([x], outputs, nodes=[producer, pool], initializers=constants,
                     opset_imports={"": 22})
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    feeds = {"x": np.random.default_rng(16).uniform(-1, 1, raw_shape).astype(np_dtype)}
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model, options, providers=["MLXExecutionProvider"])
    assert_outputs(session.run(None, feeds), m.run_cpu(model, feeds), exact=op == "MaxPool")
