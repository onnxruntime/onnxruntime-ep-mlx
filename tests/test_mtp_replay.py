"""Independent reference checks for the explicit public-ORT prefix helper."""

import ctypes
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

from bench.mtp_replay import make_checkpoint_model, make_prefix_replay_model


def descriptors(
    batch=2,
    heads=6,
    key_heads=2,
    key_width=8,
    value_width=8,
    capacity=4,
    conv_width=3,
    conv_type="tensor(float16)",
):
    channels = 2 * key_heads * key_width + heads * value_width
    width = capacity * (heads + key_heads * key_width + heads * value_width)
    inputs = [
        SimpleNamespace(
            name="past.0.recurrent",
            type="tensor(float)",
            shape=[batch, heads, value_width, key_width],
        ),
        SimpleNamespace(
            name="past.0.conv", type=conv_type, shape=[batch, channels, conv_width]
        ),
    ]
    outputs = [
        SimpleNamespace(
            name="state_update.0.recurrent_capsule",
            type="tensor(float)",
            shape=[batch, width],
        ),
        SimpleNamespace(
            name="state_update.0.conv_value",
            type=conv_type,
            shape=[batch, capacity, channels],
        ),
    ]
    return inputs, outputs


def replay_reference(state, capsule, capacity, key_heads):
    """Use libc fmaf, not the graph implementation or an unfused multiply/add."""
    fma = ctypes.CDLL(None).fmaf
    fma.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    fma.restype = ctypes.c_float
    batch, heads, value_width, key_width = state.shape
    decay = capsule[:, : capacity * heads].reshape(batch, capacity, heads)
    key_end = capacity * (heads + key_heads * key_width)
    key = capsule[:, capacity * heads : key_end].reshape(
        batch, capacity, key_heads, key_width
    )
    delta = capsule[:, key_end:].reshape(batch, capacity, heads, value_width)
    result = state * decay[:, 0, :, None, None]
    for b, h, v, k in np.ndindex(result.shape):
        result[b, h, v, k] = fma(
            key[b, 0, h * key_heads // heads, k],
            delta[b, 0, h, v],
            result[b, h, v, k],
        )
    return result


class GatedDeltaNet(OpRun):
    op_domain = "com.microsoft"

    def _run(self, qkv, q, k, offsets, decay, beta, initial, **attributes):
        assert attributes["update_rule"] == "linear"
        assert q is k is decay is beta is None
        batch, heads, value_width, key_width = initial.shape
        key_elements = (qkv.shape[1] - heads * value_width) // 2
        key_heads = key_elements // key_width
        np.testing.assert_array_equal(offsets, np.arange(batch + 1, dtype=np.int32))
        capsule = np.concatenate(
            [
                np.ones([batch, heads], np.float32),
                qkv[:, key_elements : 2 * key_elements],
                qkv[:, 2 * key_elements :],
            ],
            axis=1,
        )
        final = replay_reference(initial, capsule, 1, key_heads)
        return np.zeros([batch, heads, value_width], np.float32), final


@pytest.mark.parametrize("capacity", [1, 4, 8])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("conv_width", [0, 1, 3])
def test_prefix_replay_matches_independent_reference(capacity, batch, conv_width):
    inputs, outputs = descriptors(batch=batch, capacity=capacity, conv_width=conv_width)
    model = make_prefix_replay_model(
        inputs, outputs, capacity=capacity, batch_size=batch
    )
    rng = np.random.default_rng(46)
    feeds = {
        d.name: rng.normal(size=d.shape).astype(
            np.float32 if d.type == "tensor(float)" else np.float16
        )
        for d in [*inputs, *outputs]
    }
    original = {name: value.copy() for name, value in feeds.items()}
    recurrent, conv = ReferenceEvaluator(model, new_ops=[GatedDeltaNet]).run(
        None, feeds
    )
    expected = replay_reference(
        feeds["past.0.recurrent"],
        feeds["state_update.0.recurrent_capsule"],
        capacity,
        2,
    )
    np.testing.assert_array_equal(recurrent.view(np.uint32), expected.view(np.uint32))
    old = feeds["past.0.conv"]
    expected_conv = (
        np.concatenate(
            [
                old[:, :, 1:],
                feeds["state_update.0.conv_value"][:, :1].transpose(0, 2, 1),
            ],
            axis=2,
        )
        if conv_width
        else old
    )
    np.testing.assert_array_equal(conv.view(np.uint16), expected_conv.view(np.uint16))
    for name, value in feeds.items():
        np.testing.assert_array_equal(
            value.view(np.uint8), original[name].view(np.uint8)
        )


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_conv_replay_preserves_raw_bits(dtype):
    typ = "tensor(float16)" if dtype == np.float16 else "tensor(float)"
    inputs, outputs = descriptors(batch=1, conv_type=typ)
    inputs, outputs = inputs[1:], outputs[1:]
    integer = np.uint16 if dtype == np.float16 else np.uint32
    bits = (
        [0x8000, 0x7C01, 0x7FFF, 0xFC02]
        if dtype == np.float16
        else [0x80000000, 0x7F800001, 0x7FFFFFFF, 0xFF800002]
    )
    feeds = {
        d.name: np.resize(np.array(bits, integer), d.shape).view(dtype)
        for d in [*inputs, *outputs]
    }
    model = make_prefix_replay_model(inputs, outputs, capacity=4)
    actual = ReferenceEvaluator(model).run(None, feeds)[0]
    expected = np.concatenate(
        [
            feeds["past.0.conv"][:, :, 1:],
            feeds["state_update.0.conv_value"][:, :1].transpose(0, 2, 1),
        ],
        axis=2,
    )
    np.testing.assert_array_equal(actual.view(integer), expected.view(integer))


def test_checkpoint_has_only_identities_and_no_capture_dependency():
    inputs, _ = descriptors(batch="batch_size")
    model = make_checkpoint_model(inputs, batch_size=2)
    assert [node.op_type for node in model.graph.node] == ["Identity", "Identity"]
    assert [out.name for out in model.graph.output] == [
        "checkpoint.past.0.recurrent",
        "checkpoint.past.0.conv",
    ]
    assert all(
        v.type.tensor_type.shape.dim[0].dim_value == 2
        for v in [*model.graph.input, *model.graph.output]
    )
    onnx.checker.check_model(model, full_check=True)


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing_capture", "missing.*state capture"),
        ("wrong_capacity", "invalid packed key section"),
        ("bad_recurrent_type", "capture dtype must match"),
        ("bad_capture_type", "capture dtype must match"),
        ("bad_capture_rank", "expected rank"),
        ("bad_state_rank", "expected positive FP32"),
        ("bad_key_section", "invalid packed key section"),
        ("bad_batch", "batch dimension"),
        ("bad_capture_batch", "capture batch"),
        ("unknown_geometry", "concrete nonnegative"),
        ("unknown_state", "only.*recurrent.*conv"),
    ],
)
def test_replay_rejects_incompatible_exports(change, match):
    inputs, outputs = descriptors(batch=1)
    capacity = 4
    if change == "missing_capture":
        outputs.pop(0)
    elif change == "wrong_capacity":
        capacity = 1
    elif change == "bad_recurrent_type":
        inputs[0].type = "tensor(float16)"
    elif change == "bad_capture_type":
        outputs[0].type = "tensor(float16)"
    elif change == "bad_capture_rank":
        outputs[0].shape = [1, 2, 3]
    elif change == "bad_state_rank":
        inputs[0].shape = [1, 6, 8]
    elif change == "bad_key_section":
        outputs[0].shape[1] = 4 * (6 + 6 * 8)
    elif change == "bad_batch":
        inputs[0].shape[0] = 2
    elif change == "bad_capture_batch":
        outputs[0].shape[0] = 2
    elif change == "unknown_geometry":
        inputs[0].shape[-1] = "key_width"
    else:
        inputs[0].name = "past.0.other"
    with pytest.raises(ValueError, match=match):
        make_prefix_replay_model(inputs, outputs, capacity=capacity)


@pytest.mark.parametrize("capacity", [0, -1])
def test_replay_requires_active_capture_capacity(capacity):
    inputs, outputs = descriptors(batch=1)
    with pytest.raises(ValueError, match="positive capture capacity"):
        make_prefix_replay_model(inputs, outputs, capacity=capacity)
