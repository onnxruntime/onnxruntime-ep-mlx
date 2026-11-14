"""ONNX-reference vectors and gated runtime coverage for opset-29 packed operators."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

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
    "gguf:iq2_xxs": "2d847c629a20c0680fd803ff4a3e6b9e52c1c0370827c1f2e298460232a9d4ac",
    "gguf:iq2_xs": "3794aef862517c04fe35779d80807c6e11a0ac68192c5642f665214c0441f114",
    "gguf:iq3_xxs": "0837edfdefd1d16019a57cd13b9c1e3da84b2b4111f0ac533bdfad8e5562eae1",
    "gguf:iq1_s": "d684261b0f868b731aa93073192579ffe72cb620d8687c698fdbeed1b3485189",
    "gguf:iq4_nl": "f6a109ad08842a7434e5ede115bb231181fb91f3d074ab73fe7d0b80b543f34c",
    "gguf:iq3_s": "a1eb566ccfbfc5cb89a10a48c2d91d897508160a4f23ee7b85f4b639fb9e40a4",
    "gguf:iq2_s": "6fab6c67ea45a243db184a07bb39fea51fcb2085cc89a512bd88e5751496e802",
    "gguf:iq4_xs": "71290fdeb45737ab8df10e7c517805451e222c5321efc5f4274b6208d8ce382b",
    "gguf:iq1_m": "78a282a22eb9b4f25a773fcaaf6d2d11555620e31031e04fb8b7c6a9f377f228",
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

    if name in {
        "gguf:q4_0",
        "gguf:q5_0",
        "gguf:q8_0",
        "gguf:iq2_xxs",
        "gguf:iq2_xs",
        "gguf:iq3_xxs",
        "gguf:iq1_s",
        "gguf:iq4_nl",
        "gguf:iq3_s",
        "gguf:iq2_s",
        "gguf:q1_0",
        "gguf:q2_0",
    }:
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
    elif name == "gguf:iq1_m":
        bits = struct.unpack("<H", struct.pack("<e", 0.75))[0]
        for index in range(4):
            offset = 48 + 2 * index
            low = struct.unpack("<H", data[offset : offset + 2])[0] & 0x0FFF
            data[offset : offset + 2] = struct.pack(
                "<H", low | (((bits >> (4 * index)) & 15) << 12)
            )
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


_IQ_EDGE_HASHES = {
    "gguf:iq2_xxs": "9dae46c6ae7941e2eda7453fb20b8982176eea4f39ee83f11ad5d8eb3efeb2da",
    "gguf:iq2_xs": "0e88985fd051c2ceb6e0445c29bb7310f62995166b3add6e4323b128d319ae6a",
    "gguf:iq2_s": "bbe22508be22eb4de5426c94a7a23e319db5444871772ff4ec0db41c334aa537",
    "gguf:iq3_xxs": "60ca09a43f9f8a511ec6574ad2fa91b494f19af494a857548c45d5f57032d3d0",
    "gguf:iq3_s": "25529fa9c30b113bd83e14155b4a2e5b25377f28e8f92e08c43b060e6f502e4f",
    "gguf:iq1_s": "aa41c544e32edfa8c6e6ba342d8680504107498a183efba6246d8b64bc5a4e64",
    "gguf:iq1_m": "868b328bfa9e3547645fde1deae4f97a91fc4f4d655181d795bba6351c023699",
}


def _iq_edge_payload(name: str, block_size: int) -> np.ndarray:
    state = 0x6D2B79F5
    data = bytearray()
    for _ in range(3 * block_size):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        data.append(state % 255 + 1)
    for block, scale in enumerate((0.5, -1.0, 65504.0)):
        base = block * block_size
        if name == "gguf:iq1_m":
            bits = struct.unpack("<H", struct.pack("<e", scale))[0]
            for index, low in enumerate((0x000, 0x249, 0xDB6, 0xFFF)):
                offset = base + 48 + 2 * index
                data[offset : offset + 2] = struct.pack(
                    "<H", low | (((bits >> (4 * index)) & 15) << 12)
                )
            data[base + 32 : base + 48] = bytes([0x08, 0xF7]) * 8
        else:
            data[base : base + 2] = struct.pack("<e", scale)
        if name == "gguf:iq2_xxs":
            data[base + 6 : base + 10] = b"\xff" * 4
            data[base + 14 : base + 18] = b"\0" * 4
        elif name == "gguf:iq2_xs":
            data[base + 2 : base + 4] = struct.pack("<H", 0xFE00)
            data[base + 4 : base + 6] = b"\0\0"
            data[base + 66 : base + 68] = bytes([0xF0, 0x0F])
        elif name == "gguf:iq2_s":
            data[base + 34 : base + 36] = bytes([0, 255])
            data[base + 66] = 255
            data[base + 74 : base + 76] = bytes([0xF0, 0x0F])
        elif name == "gguf:iq3_xxs":
            data[base + 66 : base + 70] = b"\xff" * 4
            data[base + 70 : base + 74] = b"\0" * 4
        elif name == "gguf:iq3_s":
            data[base + 66] = 255
            data[base + 74 : base + 76] = bytes([0, 255])
            data[base + 106 : base + 108] = bytes([0xF0, 0x0F])
        elif name == "gguf:iq1_s":
            data[base + 34 : base + 36] = struct.pack("<H", 0x8000)
            data[base + 36 : base + 38] = struct.pack("<H", 0x7FFF)
    return np.frombuffer(bytes(data), dtype=np.uint8)


def _reference_model(op: str, inputs, output, **attributes):
    node = helper.make_node(op, [item.name for item in inputs], [output.name], **attributes)
    graph = helper.make_graph([node], "packed-reference", inputs, [output])
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 29)], ir_version=11
    )


def _mlx_session(model: bytes | onnx.ModelProto, profile: Path | None = None):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("session.disable_aot_function_inlining", "1")
    if profile is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    if isinstance(model, onnx.ModelProto):
        model = model.SerializeToString()
    session = ort.InferenceSession(
        model,
        options,
        providers=["MLXExecutionProvider"],
        enable_fallback=False,
    )
    session.disable_fallback()
    return session


def _assert_mlx_profile(session: ort.InferenceSession) -> None:
    profile_path = Path(session.end_profiling())
    try:
        events = json.loads(profile_path.read_text())
    finally:
        profile_path.unlink()
    providers = {
        event.get("args", {}).get("provider")
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    }
    assert "MLXExecutionProvider" in providers, providers
    assert "CPUExecutionProvider" not in providers, providers


@pytest.fixture(scope="module")
def runtime_opset29() -> None:
    name = "gguf:q4_0"
    elements, size = GGUF_BLOCK_SIZES[GGUF_FORMATS[name]]
    model = _reference_model(
        "DequantizePacked",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [size])],
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [elements]),
        format=name,
        shape=[elements],
    )
    try:
        _mlx_session(model).run(None, {"x": _payload(name, size)})
    except (
        ort.capi.onnxruntime_pybind11_state.Fail,
        ort.capi.onnxruntime_pybind11_state.InvalidGraph,
    ) as error:
        message = str(error)
        missing_schema = (
            "Opset 29 is under development" in message
            or "No Op registered for DequantizePacked" in message
            or (
                "DequantizePacked" in message
                and "not a registered function/op" in message
            )
        )
        if missing_schema:
            pytest.skip(
                f"ORT {ort.__version__} lacks opset-29 DequantizePacked schema loading: "
                f"{message}"
            )
        raise


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


@pytest.mark.parametrize("compile_disabled", ["0", "1"], ids=["compiled", "eager"])
@pytest.mark.parametrize("name", _REFERENCE_HASHES)
def test_dequantize_all_formats_mlx_only(
    runtime_opset29,
    name: str,
    compile_disabled: str,
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", compile_disabled)
    elements, size = GGUF_BLOCK_SIZES[GGUF_FORMATS[name]]
    model = _reference_model(
        "DequantizePacked",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [size])],
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [elements]),
        format=name,
        shape=[elements],
    )
    payload = _payload(name, size)
    expected = ReferenceEvaluator(model).run(None, {"x": payload})[0]
    session = _mlx_session(model, tmp_path / f"{name.replace(':', '-')}-{compile_disabled}")
    actual = session.run(None, {"x": payload})[0]
    np.testing.assert_array_equal(actual, expected)
    _assert_mlx_profile(session)


def test_dequantize_iq_multiblock_edge_reference_hashes() -> None:
    for name, expected_hash in _IQ_EDGE_HASHES.items():
        elements, size = GGUF_BLOCK_SIZES[GGUF_FORMATS[name]]
        model = _reference_model(
            "DequantizePacked",
            [helper.make_tensor_value_info("x", TensorProto.UINT8, [3 * size])],
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [3 * elements]),
            format=name,
            shape=[3 * elements],
        )
        output = ReferenceEvaluator(model).run(
            None, {"x": _iq_edge_payload(name, size)}
        )[0]
        assert np.isfinite(output).all()
        assert hashlib.sha256(output.astype("<f4", copy=False).tobytes()).hexdigest() == expected_hash


@pytest.mark.parametrize("compile_disabled", ["0", "1"], ids=["compiled", "eager"])
@pytest.mark.parametrize("name", _IQ_EDGE_HASHES)
def test_dequantize_iq_multiblock_edges_mlx_only(
    runtime_opset29,
    name: str,
    compile_disabled: str,
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", compile_disabled)
    elements, size = GGUF_BLOCK_SIZES[GGUF_FORMATS[name]]
    model = _reference_model(
        "DequantizePacked",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [3 * size])],
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [3 * elements]),
        format=name,
        shape=[3 * elements],
    )
    payload = _iq_edge_payload(name, size)
    expected = ReferenceEvaluator(model).run(None, {"x": payload})[0]
    session = _mlx_session(
        model, tmp_path / f"{name.replace(':', '-')}-edges-{compile_disabled}"
    )
    actual = session.run(None, {"x": payload})[0]
    np.testing.assert_array_equal(actual, expected)
    _assert_mlx_profile(session)


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


def test_pack_unpack_bits_arbitrary_axis(runtime_opset29, tmp_path: Path) -> None:
    pack = m.make_model(
        "Pack",
        [m.tensor("x", DT.UINT8, [2, 3])],
        [m.tensor("packed", DT.UINT8, [2, 2])],
        attributes={"axis": 1, "bits": 5},
        opset=29,
    )
    x = np.array([[1, 2, 3], [31, 0, 16]], dtype=np.uint8)
    session = _mlx_session(pack, tmp_path / "pack-bits")
    actual = session.run(None, {"x": x})[0]
    np.testing.assert_array_equal(actual, [[65, 12], [31, 64]])
    _assert_mlx_profile(session)

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
    session = _mlx_session(unpack, tmp_path / "unpack-bits")
    np.testing.assert_array_equal(session.run(None, feeds)[0], x)
    _assert_mlx_profile(session)


@pytest.mark.parametrize("bits", range(1, 8))
@pytest.mark.parametrize("storage", [TensorProto.UINT8, TensorProto.UINT32], ids=["u8", "u32"])
@pytest.mark.parametrize("intermediate", [False, True], ids=["input", "intermediate"])
@pytest.mark.parametrize("disable_compile", ["0", "1"], ids=["compile-enabled", "eager"])
def test_pack_bits_checks_live_codes(
    runtime_opset29, bits, storage, intermediate, disable_compile, monkeypatch
) -> None:
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", disable_compile)
    word_bits = 8 if storage == TensorProto.UINT8 else 32
    shape = [2, 3]
    packed_shape = [2, (shape[1] * bits + word_bits - 1) // word_bits]
    model = _reference_model(
        "Pack",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, shape)],
        helper.make_tensor_value_info("y", storage, packed_shape),
        bits=bits,
        output_type=storage,
    )
    if intermediate:
        model.graph.node[0].input[0] = "live"
        model.graph.node.insert(0, helper.make_node("Identity", ["x"], ["live"]))
    session = _mlx_session(model)
    reference = ReferenceEvaluator(model)
    limit = (1 << bits) - 1
    valid = np.array([[0, limit, 0], [limit, 0, limit]], dtype=np.uint8)
    other_valid = limit - valid
    invalid = valid.copy()
    invalid[1, 1] = limit + 1
    for values in [valid, other_valid]:
        np.testing.assert_array_equal(
            session.run(None, {"x": values})[0], reference.run(None, {"x": values})[0]
        )
    with pytest.raises(ValueError, match="Input codes must be representable"):
        reference.run(None, {"x": invalid})
    message = (
        rf"Pack: maximum input code {limit + 1} exceeds {limit} for bits={bits}; "
        rf"provide UINT8 codes in \[0, {limit}\]"
    )
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.EPFail, match=message):
        session.run(None, {"x": invalid})
    np.testing.assert_array_equal(
        session.run(None, {"x": valid})[0], reference.run(None, {"x": valid})[0]
    )


@pytest.mark.parametrize("storage", [TensorProto.UINT8, TensorProto.UINT32], ids=["u8", "u32"])
@pytest.mark.parametrize("disable_compile", ["0", "1"], ids=["compile-enabled", "eager"])
def test_pack_full_byte_accepts_255_after_empty_replay(
    runtime_opset29, storage, disable_compile, monkeypatch
) -> None:
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", disable_compile)
    word_bits = 8 if storage == TensorProto.UINT8 else 32
    model = _reference_model(
        "Pack",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [None, 5])],
        helper.make_tensor_value_info("y", storage, [None, (40 + word_bits - 1) // word_bits]),
        bits=8,
        output_type=storage,
    )
    session = _mlx_session(model)
    reference = ReferenceEvaluator(model)
    nonempty = np.array([[0, 1, 127, 254, 255], [255, 254, 127, 1, 0]], dtype=np.uint8)
    for values in [
        np.empty((0, 5), np.uint8),
        nonempty,
        255 - nonempty,
        np.empty((0, 5), np.uint8),
    ]:
        np.testing.assert_array_equal(
            session.run(None, {"x": values})[0], reference.run(None, {"x": values})[0]
        )


def test_pack_unpack_large_empty_non_axis(runtime_opset29) -> None:
    # Keep a regression in graph-construction complexity bounded to a subprocess.
    worker = "_MLX_PACK_EMPTY_TEST_WORKER"
    if os.environ.get(worker) != "1":
        result = subprocess.run(
            [sys.executable, "-m", "pytest", f"{Path(__file__)}::test_pack_unpack_large_empty_non_axis",
             "-q", "-p", "no:cacheprovider"],
            env={**os.environ, worker: "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=45,
            check=False,
        )
        assert result.returncode == 0, result.stdout
        return

    cases = [
        ([0, 1_000_000], 1, {"bits": 8}, TensorProto.UINT8),
        ([1_000_000, 0], 0, {"bits": 3}, TensorProto.UINT32),
        ([2, 0, 100_000], -1, {"bits": 5}, TensorProto.UINT8),
        ([50_000, 50_000, 5, 0], 2, {"radix": 3, "group_size": 5}, TensorProto.UINT8),
    ]
    for shape, axis, attributes, storage in cases:
        axis %= len(shape)
        word_bits = 8 if storage == TensorProto.UINT8 else 32
        packed_shape = shape.copy()
        if "bits" in attributes:
            total_bits = shape[axis] * attributes["bits"]
        else:
            radix, group_size = attributes["radix"], attributes["group_size"]
            groups = (shape[axis] + group_size - 1) // group_size
            total_bits = groups * (radix**group_size - 1).bit_length()
        packed_shape[axis] = (total_bits + word_bits - 1) // word_bits
        pack = _reference_model(
            "Pack",
            [helper.make_tensor_value_info("x", TensorProto.UINT8, shape)],
            helper.make_tensor_value_info("y", storage, packed_shape),
            axis=axis,
            output_type=storage,
            **attributes,
        )
        inputs = {"x": np.empty(shape, dtype=np.uint8)}
        actual = _mlx_session(pack).run(None, inputs)[0]
        expected = ReferenceEvaluator(pack).run(None, inputs)[0]
        np.testing.assert_array_equal(actual, expected)
        assert list(actual.shape) == packed_shape

        unpack = _reference_model(
            "Unpack",
            [helper.make_tensor_value_info("x", storage, packed_shape),
             helper.make_tensor_value_info("count", TensorProto.INT64, [])],
            helper.make_tensor_value_info("y", TensorProto.UINT8, [None] * len(shape)),
            axis=axis,
            **attributes,
        )
        feeds = {"x": actual, "count": np.array(shape[axis], dtype=np.int64)}
        session = _mlx_session(unpack)
        actual = session.run(None, feeds)[0]
        expected = ReferenceEvaluator(unpack).run(None, feeds)[0]
        np.testing.assert_array_equal(actual, expected)
        assert list(actual.shape) == shape
        feeds["count"] = np.array(0, dtype=np.int64)
        with pytest.raises(ort.capi.onnxruntime_pybind11_state.EPFail, match="packed axis"):
            session.run(None, feeds)
        with pytest.raises(ValueError, match="Packed axis dimension must equal"):
            ReferenceEvaluator(unpack).run(None, feeds)


@pytest.mark.parametrize("shape,axis", [([0], 0), ([0, 5], 0), ([2, 0, 3], 1), ([0, 0], -1)])
@pytest.mark.parametrize("storage", [TensorProto.UINT8, TensorProto.UINT32], ids=["u8", "u32"])
def test_pack_unpack_zero_selected_axis(runtime_opset29, shape, axis, storage) -> None:
    pack = _reference_model(
        "Pack",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, shape)],
        helper.make_tensor_value_info("y", storage, shape),
        bits=3,
        axis=axis,
        output_type=storage,
    )
    packed = _mlx_session(pack).run(None, {"x": np.empty(shape, np.uint8)})[0]
    np.testing.assert_array_equal(
        packed, ReferenceEvaluator(pack).run(None, {"x": np.empty(shape, np.uint8)})[0]
    )
    unpack = _reference_model(
        "Unpack",
        [helper.make_tensor_value_info("x", storage, shape),
         helper.make_tensor_value_info("count", TensorProto.INT64, [])],
        helper.make_tensor_value_info("y", TensorProto.UINT8, shape),
        bits=3,
        axis=axis,
    )
    feeds = {"x": packed, "count": np.array(0, np.int64)}
    np.testing.assert_array_equal(
        _mlx_session(unpack).run(None, feeds)[0], ReferenceEvaluator(unpack).run(None, feeds)[0]
    )


def test_unpack_empty_count_exceeding_mlx_dimension_limit(runtime_opset29) -> None:
    count = 1 << 31
    shape = [0, count // 8]
    model = _reference_model(
        "Unpack",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, shape),
         helper.make_tensor_value_info("count", TensorProto.INT64, [])],
        helper.make_tensor_value_info("y", TensorProto.UINT8, [0, None]),
        bits=1,
    )
    feeds = {"x": np.empty(shape, np.uint8), "count": np.array(count, np.int64)}
    assert ReferenceEvaluator(model).run(None, feeds)[0].shape == (0, count)
    with pytest.raises(
        ort.capi.onnxruntime_pybind11_state.EPFail,
        match=r"Unpack: count 2147483648 exceeds the int32 dimension limit; reduce count",
    ):
        _mlx_session(model).run(None, feeds)


@pytest.mark.parametrize("disable_compile", ["0", "1"], ids=["compile-enabled", "eager"])
def test_unpack_empty_then_nonempty_replay(runtime_opset29, disable_compile, monkeypatch) -> None:
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", disable_compile)
    graph = helper.make_graph(
        [helper.make_node("Unpack", ["x", "count"], ["y"], bits=5)],
        "unpack-constant-count-replay",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [None, 2])],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [None, 3])],
        [helper.make_tensor("count", TensorProto.INT64, [], [3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 29)], ir_version=11)
    session = _mlx_session(model)
    reference = ReferenceEvaluator(model)
    for values in [
        np.empty((0, 2), np.uint8),
        np.array([[65, 12], [31, 64]], np.uint8),
        np.array([[0, 0], [255, 127]], np.uint8),
        np.empty((0, 2), np.uint8),
    ]:
        np.testing.assert_array_equal(
            session.run(None, {"x": values})[0], reference.run(None, {"x": values})[0]
        )


def test_pack_bits_compile_policy(runtime_opset29, tmp_path: Path) -> None:
    cases = [
        ("test_pack_bits_checks_live_codes[compile-enabled-input-u8-3]", "Pack", False),
        ("test_pack_full_byte_accepts_255_after_empty_replay[compile-enabled-u8]", "Pack", True),
        ("test_unpack_empty_then_nonempty_replay[compile-enabled]", "Unpack", True),
    ]
    for index, (case, op, compiled) in enumerate(cases):
        trace = tmp_path / f"packing-{index}.json"
        subprocess.run(
            [sys.executable, "-m", "pytest", f"{Path(__file__)}::{case}", "-q",
             "-p", "no:cacheprovider"],
            env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
            check=True,
            timeout=90,
        )
        events = [event["args"] for event in json.loads(trace.read_text())
                  if event.get("cat") == "ep.path"
                  and event.get("args", {}).get("ops") == op]
        if compiled:
            assert any(event["path"] == "general" for event in events), events
            assert any(event["cache"] == "HIT" for event in events), events
        else:
            assert len(events) >= 3, events
            assert all(event["path"] == "eager" for event in events), events


def test_pack_grouped_radix_uint32(runtime_opset29, tmp_path: Path) -> None:
    model = m.make_model(
        "Pack",
        [m.tensor("x", DT.UINT8, [5])],
        [m.tensor("out", DT.UINT32, [1])],
        attributes={"radix": 3, "group_size": 5, "output_type": 12},
        opset=29,
    )
    feeds = {"x": np.array([0, 1, 2, 0, 1], dtype=np.uint8)}
    session = _mlx_session(model, tmp_path / "pack-grouped")
    np.testing.assert_array_equal(
        session.run(None, feeds)[0],
        np.array([102], dtype=np.uint32),
    )
    _assert_mlx_profile(session)


def test_dequantize_packed_q4_0(runtime_opset29, tmp_path: Path) -> None:
    payload = np.zeros(18, dtype=np.uint8)
    payload[:2] = np.frombuffer(np.float16(1).tobytes(), dtype=np.uint8)
    model = m.make_model(
        "DequantizePacked",
        [m.tensor("x", DT.UINT8, [18])],
        [m.tensor("out", DT.FLOAT, [32])],
        attributes={"format": "gguf:q4_0", "shape": [32]},
        opset=29,
    )
    session = _mlx_session(model, tmp_path / "dequantize-q4-0")
    actual = session.run(None, {"x": payload})[0]
    np.testing.assert_array_equal(actual, np.full(32, -8, dtype=np.float32))
    _assert_mlx_profile(session)
