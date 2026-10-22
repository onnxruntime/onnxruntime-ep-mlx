"""ONNX numeric Cast/CastLike coverage, including type-only targets and conversion edges."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

import _models as m

DT = ir.DataType
TYPES = [
    (DT.BOOL, np.bool_), (DT.INT8, np.int8), (DT.UINT8, np.uint8),
    (DT.INT16, np.int16), (DT.UINT16, np.uint16),
    (DT.INT32, np.int32), (DT.UINT32, np.uint32),
    (DT.INT64, np.int64), (DT.UINT64, np.uint64),
    (DT.FLOAT16, np.float16), (DT.BFLOAT16, ml_dtypes.bfloat16),
    (DT.FLOAT, np.float32), (DT.DOUBLE, np.float64),
]


def cast_model(op, src, dst, size, *, target_size=0):
    """Expose BF16 through fp32 boundaries because ORT NumPy I/O has no BF16 conversion."""
    x = m.tensor("x", DT.FLOAT if src == DT.BFLOAT16 else src, [size])
    value = x
    nodes = []
    if src == DT.BFLOAT16:
        value = m.tensor("source_bf16", src, [size])
        nodes.append(ir.node("Cast", [x], attributes={"to": int(src)}, outputs=[value]))
    inputs = [x]
    cast_inputs = [value]
    if op == "CastLike":
        target = m.tensor("target", DT.FLOAT if dst == DT.BFLOAT16 else dst, [target_size])
        inputs.append(target)
        like = target
        if dst == DT.BFLOAT16:
            like = m.tensor("target_bf16", dst, [target_size])
            nodes.append(ir.node("Cast", [target], attributes={"to": int(dst)}, outputs=[like]))
        cast_inputs.append(like)
    result = m.tensor("result", dst, [size])
    nodes.append(ir.node(op, cast_inputs, attributes={"to": int(dst)} if op == "Cast" else {},
                         outputs=[result]))
    if dst == DT.BFLOAT16:
        output = m.tensor("output", DT.FLOAT, [size])
        nodes.append(ir.node("Cast", [result], attributes={"to": int(DT.FLOAT)}, outputs=[output]))
    else:
        output = result
    graph = ir.Graph(inputs, [output], nodes=nodes, opset_imports={"": 24}, name="cast_coverage")
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def check_conversion(op, src, src_np, dst, dst_np, values):
    x = np.array(values, dtype=src_np)
    feeds = {"x": x.astype(np.float32) if src == DT.BFLOAT16 else x}
    if op == "CastLike":
        feeds["target"] = np.empty(0, np.float32 if dst == DT.BFLOAT16 else dst_np)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_profiling = True
    options.profile_file_prefix = "cast_coverage"
    model = cast_model(op, src, dst, x.size)
    session = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    cpu_options = ort.SessionOptions()
    cpu_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    cpu = ort.InferenceSession(model, cpu_options, providers=["CPUExecutionProvider"])
    with np.errstate(over="ignore", invalid="ignore"):
        expected = x.astype(dst_np)
    if dst == DT.BFLOAT16:
        expected = expected.astype(np.float32)
    try:
        for _ in range(2):
            actual = session.run(None, feeds)[0]
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual, cpu.run(None, feeds)[0])
            if np.issubdtype(expected.dtype, np.floating):
                np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    finally:
        profile = Path(session.end_profiling())
        events = json.loads(profile.read_text())
        profile.unlink()
    providers = [e["args"]["provider"] for e in events
                 if e.get("cat") == "Node" and "provider" in e.get("args", {})]
    assert providers and set(providers) == {"MLXExecutionProvider"}, providers


@pytest.mark.parametrize("op", ["Cast", "CastLike"])
@pytest.mark.parametrize("src,src_np", TYPES, ids=[t.name for t, _ in TYPES])
@pytest.mark.parametrize("dst,dst_np", TYPES, ids=[t.name for t, _ in TYPES])
def test_numeric_cast_pairs(op, src, src_np, dst, dst_np):
    if src == DT.BOOL:
        values = [False, True, False, True]
    elif np.issubdtype(np.dtype(src_np), np.integer):
        limits = np.iinfo(src_np)
        values = [0, 1, limits.min, limits.max]
    elif dst in [DT.BOOL, DT.FLOAT16, DT.BFLOAT16, DT.FLOAT, DT.DOUBLE]:
        values = [0.0, -0.0, 1.5, -2.75, float("inf"), -float("inf"), float("nan")]
    elif np.issubdtype(np.dtype(dst_np), np.unsignedinteger):
        values = [0.0, -0.0, 1.75, 2.5]
    else:
        values = [0.0, -0.0, 1.75, -2.5]
    check_conversion(op, src, src_np, dst, dst_np, values)


@pytest.mark.parametrize("op", ["Cast", "CastLike"])
@pytest.mark.parametrize("src,src_np", [(DT.INT64, np.int64), (DT.UINT64, np.uint64)])
@pytest.mark.parametrize("dst,dst_np", TYPES[-4:], ids=[t.name for t, _ in TYPES[-4:]])
def test_large_integer_rounding(op, src, src_np, dst, dst_np):
    values = [2**24 - 1, 2**24 + 1, 2**24 + 3, 2**53 - 1, 2**53 + 1, 2**53 + 3]
    if src == DT.INT64:
        values += [-v for v in values]
    check_conversion(op, src, src_np, dst, dst_np, values)


FLOAT_TYPES = TYPES[-4:]
INTEGER_TYPES = TYPES[1:9]


@pytest.mark.parametrize("op", ["Cast", "CastLike"])
@pytest.mark.parametrize("src,src_np", FLOAT_TYPES, ids=[t.name for t, _ in FLOAT_TYPES])
@pytest.mark.parametrize("dst,dst_np", [(DT.BOOL, np.bool_), *FLOAT_TYPES],
                         ids=["BOOL", *[t.name for t, _ in FLOAT_TYPES]])
def test_float_conversion_edges(op, src, src_np, dst, dst_np):
    info = ml_dtypes.finfo(src_np) if src == DT.BFLOAT16 else np.finfo(src_np)
    values = [0.0, -0.0, info.smallest_subnormal, -info.smallest_subnormal,
              info.tiny, -info.tiny, info.max, -info.max, 1 + 2**-8, 1 + 3 * 2**-8,
              np.inf, -np.inf, np.nan]
    check_conversion(op, src, src_np, dst, dst_np, values)


@pytest.mark.parametrize("src,src_np", INTEGER_TYPES, ids=[t.name for t, _ in INTEGER_TYPES])
@pytest.mark.parametrize("dst,dst_np", INTEGER_TYPES, ids=[t.name for t, _ in INTEGER_TYPES])
def test_integer_narrowing_wraps_bits(src, src_np, dst, dst_np):
    limits = np.iinfo(src_np)
    values = [limits.min, limits.min + 1, limits.max, limits.max - 1, 0, 1]
    values += [v for v in [-65537, -257, -129, -1, 127, 128, 255, 256, 32768, 65536,
                          2**31, 2**32, 2**63] if limits.min <= v <= limits.max]
    check_conversion("Cast", src, src_np, dst, dst_np, values)


def test_castlike_type_only_target_compile_child():
    if os.environ.get("MLX_CAST_TYPE_CHILD") != "1":
        pytest.skip("run by test_castlike_type_only_target_uses_compile")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    model = cast_model("CastLike", DT.FLOAT, DT.UINT16, 4, target_size="T")
    session = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    for size in [0, 1, 1024, 0]:
        for values in [[0.0, 1.75, 128.5, 65535.0], [1.5, 7.25, 256.0, 8192.0]]:
            x = np.array(values, np.float32)
            actual = session.run(None, {"x": x, "target": np.full(size, 17, np.uint16)})[0]
            np.testing.assert_array_equal(actual, x.astype(np.uint16))


@pytest.mark.skipif(os.environ.get("MLX_CAST_TYPE_CHILD") == "1", reason="child")
def test_castlike_type_only_target_uses_compile(tmp_path):
    trace = tmp_path / "castlike.json"
    subprocess.run(
        [sys.executable, "-m", "pytest",
         f"{Path(__file__).name}::test_castlike_type_only_target_compile_child", "-q"],
        cwd=Path(__file__).parent,
        env={**os.environ, "MLX_CAST_TYPE_CHILD": "1",
             "ONNXRUNTIME_EP_MLX_TRACE": str(trace), "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0",
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0"},
        check=True, timeout=120,
    )
    events = json.loads(trace.read_text())
    paths = [e for e in events if e.get("cat") == "ep.path"]
    assert len(paths) == 8
    assert all(e["name"] == "mlx.compute[general]" for e in paths)
    assert [e["args"]["cache"] for e in paths] == ["MISS", *(["HIT"] * 7)]
    assert all(e["args"]["shape_key"] == "0:1:[4]" for e in paths), paths


def test_castlike_target_used_elsewhere_stays_live():
    x = m.tensor("x", DT.FLOAT, [3])
    target = m.tensor("target", DT.UINT16, [3])
    y = m.tensor("y", DT.UINT16, [3])
    target_out = m.tensor("target_out", DT.FLOAT, [3])
    graph = ir.Graph(
        [x, target], [y, target_out],
        nodes=[
            ir.node("CastLike", [x, target], outputs=[y]),
            ir.node("Cast", [target], outputs=[target_out], attributes={"to": int(DT.FLOAT)}),
        ],
        opset_imports={"": 24},
    )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString(),
        options, providers=m.EP_PROVIDERS,
    )
    for values in [[0, 1, 65535], [7, 8192, 256]]:
        target_data = np.array(values, np.uint16)
        actual = session.run(None, {"x": np.array([0.5, 1.75, 7.25], np.float32),
                                    "target": target_data})
        np.testing.assert_array_equal(actual[0], np.array([0, 1, 7], np.uint16))
        np.testing.assert_array_equal(actual[1], target_data.astype(np.float32))


@pytest.mark.parametrize("op", ["Cast", "CastLike"])
def test_string_conversion_stays_on_cpu(op):
    x = m.tensor("x", DT.STRING, [3])
    inputs = [x]
    feeds = {"x": np.array(["-2.5", "0", "17.75"], dtype=object)}
    if op == "CastLike":
        inputs.append(m.tensor("target", DT.FLOAT, [0]))
        feeds["target"] = np.empty(0, np.float32)
    model = m.make_model(op, inputs, [m.tensor("y", DT.FLOAT, [3])],
                         attributes={"to": int(DT.FLOAT)} if op == "Cast" else None)
    # ORT rewrites CastLike to Cast when optimizing a known target dtype.
    m.assert_op_declined(model, feeds, "Cast")
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)


@pytest.mark.parametrize("saturate", [0, 1])
def test_float8_attributes_retain_cpu_fallback(saturate):
    x = m.tensor("x", DT.FLOAT, [6])
    low = m.tensor("fp8", DT.FLOAT8E4M3FN, [6])
    y = m.tensor("y", DT.FLOAT, [6])
    graph = ir.Graph(
        [x], [y],
        nodes=[
            ir.node("Cast", [x], outputs=[low],
                    attributes={"to": int(DT.FLOAT8E4M3FN), "saturate": saturate}),
            ir.node("Cast", [low], outputs=[y], attributes={"to": int(DT.FLOAT)}),
        ],
        opset_imports={"": 24},
    )
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    feeds = {"x": np.array([-1000, -1.5, 0, 1.5, 448, 1000], np.float32)}
    m.assert_op_declined(model, feeds, "Cast")
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)
