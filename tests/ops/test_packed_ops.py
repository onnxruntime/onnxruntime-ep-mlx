"""ONNX-reference vectors and gated runtime coverage for opset-29 packed operators."""

from __future__ import annotations

import hashlib
import struct

import _models as m
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator
from onnx_ir import DataType as DT

_packed_reference = pytest.importorskip(
    "onnx.reference.ops.op_dequantize_packed",
    reason="Packed operator reference tests require ONNX main with opset-29 schemas",
)
GGUF_BLOCK_SIZES = _packed_reference.GGUF_BLOCK_SIZES
GGUF_FORMATS = _packed_reference.GGUF_FORMATS


_ORT_SKIP = pytest.mark.skip(
    reason=(
        f"ONNX Runtime {ort.__version__} does not register main-domain opset-29 schemas; "
        "the ONNX 1.24 reference vectors are covered here and by Rust unit tests"
    )
)

_REFERENCE_HASHES = {
    "gguf:q4_0": "a90720e173d085dc9909350e325b5aedcde971c741dddaffd79be99bf2440e95",
    "gguf:q4_1": "3477c46e645e6efc46e1f4db0a3a9da86f67bc42b4590b07d208f247a35b43b8",
    "gguf:q5_0": "83de917dca8f6fc5b443c1df9ca0f56d5e4beee705be983a5b57ebe1b85df0d0",
    "gguf:q5_1": "22901be3fbd743e1cb8033f1fa8dd2df41ef750eac1d66b90b08fdfab0ab0d7b",
    "gguf:q8_0": "81c210fbb597956e5bb5886548765bcb89b4543c859ada164db6ec4108480a61",
    "gguf:q8_1": "6a88aa6f38aae39f127b4d409d132df3c72270a963b7ddeb689c316d75ac04a2",
    "gguf:q2_k": "23eeb776edeb5dd8108599db95ad3e4a9da2df69c25e9babb69e373048961ce8",
    "gguf:q3_k": "4ca578e80d4c1667d452dddbda42202bb5ac50d7550a7e9c07c590b620d0d476",
    "gguf:q4_k": "0b7d60960bbe06d8729b29efd26faa5596d34e3d3bd6b4e9429deeb9bd74ccfa",
    "gguf:q5_k": "bace8af8173213f13e200bc12909fcc01d91fcde7b0f194aa912c28f72c007db",
    "gguf:q6_k": "2fa306b0594050ec51f42446383458c63c9e35d6ca49c0edab12e683d52c17c8",
    "gguf:q8_k": "8dfa3856e7e2d6a013d7972667db95a0ef582090fd06aa181d0b6bf17cc6b816",
    "gguf:iq4_nl": "f6a109ad08842a7434e5ede115bb231181fb91f3d074ab73fe7d0b80b543f34c",
    "gguf:iq4_xs": "71290fdeb45737ab8df10e7c517805451e222c5321efc5f4274b6208d8ce382b",
    "gguf:tq1_0": "828722236022425372a33c139a735a5df1e2b0531a37e843ed999b443e51f644",
    "gguf:tq2_0": "4d47fb49ce1830cb43d7e7aa1671a68c6c8881388b6af66a674b7b9fb9ed1cf6",
    "gguf:mxfp4": "225c2bd8325c321b0f614e4223e7c5926faa1aae8f9147d5b46261151b4a971c",
    "gguf:nvfp4": "7786b973f254aeba6c190e61c7205102f8816b034e5fdded7490fe67ca1c4305",
    "gguf:q1_0": "aa4b746ef275e19003de06231cfce05503506bb9e3578e3e0f6fe2265c35380d",
    "gguf:q2_0": "a737a64244923a3ed27285517e505c26992139cd2300f0269d4e85048abed865",
}


def _payload(name: str, size: int) -> np.ndarray:
    data = bytearray((index * 37 + 11) & 255 for index in range(size))

    def half(offset: int, value: float) -> None:
        data[offset : offset + 2] = struct.pack("<e", value)

    if name in {"gguf:q4_0", "gguf:q5_0", "gguf:q8_0", "gguf:iq4_nl", "gguf:q1_0", "gguf:q2_0"}:
        half(0, 0.75)
    elif name in {"gguf:q4_1", "gguf:q5_1", "gguf:q8_1"}:
        half(0, 0.75)
        half(2, -0.5)
    elif name == "gguf:q2_k":
        half(80, 0.75)
        half(82, 0.25)
    elif name == "gguf:q3_k":
        half(108, 0.75)
    elif name in {"gguf:q4_k", "gguf:q5_k"}:
        half(0, 0.75)
        half(2, 0.25)
    elif name == "gguf:q6_k":
        half(208, 0.75)
    elif name == "gguf:q8_k":
        data[:4] = struct.pack("<f", 0.75)
    elif name == "gguf:iq4_xs":
        half(0, 0.75)
    elif name == "gguf:tq1_0":
        half(52, 0.75)
    elif name == "gguf:tq2_0":
        half(64, 0.75)
    elif name == "gguf:mxfp4":
        data[0] = 126
    elif name == "gguf:nvfp4":
        data[:4] = bytes([0x38, 0x21, 0x01, 0x7F])
    else:
        raise AssertionError(name)
    return np.frombuffer(bytes(data), dtype=np.uint8)


def _reference_model(op: str, inputs, output, **attributes):
    node = helper.make_node(op, [item.name for item in inputs], [output.name], **attributes)
    graph = helper.make_graph([node], "packed-reference", inputs, [output])
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 29)], ir_version=onnx.IR_VERSION
    )


def test_dequantize_supported_formats_reference_hashes() -> None:
    assert onnx.defs.onnx_opset_version() >= 29
    for name, expected_hash in _REFERENCE_HASHES.items():
        elements, size = GGUF_BLOCK_SIZES[GGUF_FORMATS[name]]
        model = _reference_model(
            "DequantizePacked",
            [helper.make_tensor_value_info("x", TensorProto.UINT8, [size])],
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [elements]),
            format=name,
            shape=[elements],
        )
        output = ReferenceEvaluator(model).run(None, {"x": _payload(name, size)})[0]
        assert hashlib.sha256(output.astype("<f4", copy=False).tobytes()).hexdigest() == expected_hash


def test_pack_unpack_reference_axis_and_grouped_vectors() -> None:
    cases = [
        (
            np.array([[1, 7, 3], [2, 0, 4], [5, 6, 1], [7, 2, 0]], dtype=np.uint8),
            TensorProto.UINT8,
            {"bits": 3, "axis": 0},
            np.array([[81, 135, 99], [15, 5, 0]], dtype=np.uint8),
        ),
        (
            np.array([[1, 2, 3, 4, 5], [31, 0, 16, 7, 9]], dtype=np.uint8),
            TensorProto.UINT32,
            {"bits": 5, "axis": 1},
            np.array([[5_377_089], [9_682_975]], dtype=np.uint32),
        ),
        (
            np.array(
                [[0, 2, 1], [1, 1, 2], [2, 0, 0], [0, 2, 1], [1, 1, 2], [2, 0, 0]],
                dtype=np.uint8,
            ),
            TensorProto.UINT8,
            {"radix": 3, "group_size": 5, "axis": 0},
            np.array([[102, 140, 196], [2, 0, 0]], dtype=np.uint8),
        ),
        (
            np.array([[0, 1, 2, 0, 1, 2], [2, 2, 2, 2, 2, 1]], dtype=np.uint8),
            TensorProto.UINT32,
            {"radix": 3, "group_size": 5, "axis": 1},
            np.array([[614], [498]], dtype=np.uint32),
        ),
    ]
    for source, output_type, attributes, expected in cases:
        pack = _reference_model(
            "Pack",
            [helper.make_tensor_value_info("x", TensorProto.UINT8, list(source.shape))],
            helper.make_tensor_value_info("y", output_type, list(expected.shape)),
            output_type=output_type,
            **attributes,
        )
        packed = ReferenceEvaluator(pack).run(None, {"x": source})[0]
        np.testing.assert_array_equal(packed, expected)
        unpack = _reference_model(
            "Unpack",
            [
                helper.make_tensor_value_info("x", output_type, list(expected.shape)),
                helper.make_tensor_value_info("count", TensorProto.INT64, []),
            ],
            helper.make_tensor_value_info("y", TensorProto.UINT8, list(source.shape)),
            **attributes,
        )
        count = np.array(source.shape[attributes["axis"]], dtype=np.int64)
        actual = ReferenceEvaluator(unpack).run(None, {"x": packed, "count": count})[0]
        np.testing.assert_array_equal(actual, source)


@_ORT_SKIP
def test_pack_unpack_bits_arbitrary_axis() -> None:
    pack = m.make_model(
        "Pack",
        [m.tensor("x", DT.UINT8, [2, 3])],
        [m.tensor("packed", DT.UINT8, [2, 2])],
        attributes={"axis": 1, "bits": 5},
        opset=29,
    )
    x = np.array([[1, 2, 3], [31, 0, 16]], dtype=np.uint8)
    m.assert_mlx_claims(pack, {"x": x})
    actual = m._session(pack, m.EP_PROVIDERS).run(None, {"x": x})[0]
    np.testing.assert_array_equal(actual, [[65, 12], [31, 64]])

    unpack = m.make_model(
        "Unpack",
        [m.tensor("x", DT.UINT8, [2, 2]), m.tensor("count", DT.INT64, [])],
        [m.tensor("out", DT.UINT8, [2, 3])],
        attributes={"axis": 1, "bits": 5},
        opset=29,
    )
    feeds = {
        "x": np.array([[65, 12], [31, 64]], dtype=np.uint8),
        "count": np.array(3, dtype=np.int64),
    }
    m.assert_mlx_claims(unpack, feeds)
    np.testing.assert_array_equal(m._session(unpack, m.EP_PROVIDERS).run(None, feeds)[0], x)


@_ORT_SKIP
def test_pack_grouped_radix_uint32() -> None:
    model = m.make_model(
        "Pack",
        [m.tensor("x", DT.UINT8, [5])],
        [m.tensor("out", DT.UINT32, [1])],
        attributes={"radix": 3, "group_size": 5, "output_type": 12},
        opset=29,
    )
    feeds = {"x": np.array([0, 1, 2, 0, 1], dtype=np.uint8)}
    m.assert_mlx_claims(model, feeds)
    np.testing.assert_array_equal(
        m._session(model, m.EP_PROVIDERS).run(None, feeds)[0],
        np.array([102], dtype=np.uint32),
    )


@_ORT_SKIP
def test_dequantize_packed_q4_0() -> None:
    payload = np.zeros(18, dtype=np.uint8)
    payload[:2] = np.frombuffer(np.float16(1).tobytes(), dtype=np.uint8)
    model = m.make_model(
        "DequantizePacked",
        [m.tensor("x", DT.UINT8, [18])],
        [m.tensor("out", DT.FLOAT, [32])],
        attributes={"format": "gguf:q4_0", "shape": [32]},
        opset=29,
    )
    m.assert_mlx_claims(model, {"x": payload})
    actual = m._session(model, m.EP_PROVIDERS).run(None, {"x": payload})[0]
    np.testing.assert_array_equal(actual, np.full(32, -8, dtype=np.float32))
