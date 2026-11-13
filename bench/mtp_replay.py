"""Explicit one-token state replay helpers for public-ORT MTP experiments.

These are separate ONNX sessions, not changes to the decoder export or EP.
Replay starts from a pre-verification checkpoint and an active prefix-one
capture. It does not restore paged KV lengths; the caller must do that separately.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from bench.ort_paged import state_output


def _states(inputs, batch_size):
    if batch_size <= 0:
        raise ValueError("Replay requires a positive batch_size")
    states = [value for value in inputs if value.name.startswith("past.")]
    if not states:
        raise ValueError("No past.* recurrent/convolution states were provided")
    result = []
    for value in states:
        shape = list(value.shape)
        if not shape:
            raise ValueError(f"{value.name}: state must have a batch dimension")
        if isinstance(shape[0], int) and shape[0] != batch_size:
            raise ValueError(
                f"{value.name}: batch dimension {shape[0]} does not match "
                f"batch_size={batch_size}"
            )
        shape[0] = batch_size
        if any(not isinstance(d, int) or d < 0 for d in shape):
            raise ValueError(
                f"{value.name}: replay needs concrete nonnegative state geometry; "
                f"received {shape}"
            )
        types = {
            "tensor(float)": TensorProto.FLOAT,
            "tensor(float16)": TensorProto.FLOAT16,
        }
        if value.type not in types:
            raise ValueError(
                f"{value.name}: unsupported state dtype {value.type}; "
                "use FP32 recurrent state and FP16/FP32 convolution state"
            )
        result.append((value.name, shape, types[value.type]))
    return result


def _model(nodes, inputs, outputs, constants, values=()):
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "public_mtp_state_helper",
            inputs,
            outputs,
            constants,
            value_info=values,
        ),
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    return model


def make_checkpoint_model(inputs, *, batch_size=1):
    """Identity outputs can retain immutable GPU arrays through public IOBinding."""
    sources, outputs, nodes = [], [], []
    for name, shape, dtype in _states(inputs, batch_size):
        output = "checkpoint." + name
        sources.append(helper.make_tensor_value_info(name, dtype, shape))
        outputs.append(helper.make_tensor_value_info(output, dtype, shape))
        nodes.append(helper.make_node("Identity", [name], [output]))
    return _model(nodes, sources, outputs, [])


def make_prefix_replay_model(inputs, capture_outputs, *, capacity, batch_size=1):
    """Replay exactly the first captured update, without CPU state materialization.

    Recurrent replay materializes FP32 decay multiplication before the linear
    GatedDeltaNet update (key * captured_delta + state). Convolution replay uses
    only slicing/transposition/concatenation, with no floating-point arithmetic.
    The capsule section offsets depend on the export's full capture capacity,
    even though only its first entry is replayed.
    """
    if capacity <= 0:
        raise ValueError("Replay requires the export's positive capture capacity")
    captures = {value.name: value for value in capture_outputs}
    nodes, sources, outputs, constants, values = [], [], [], [], []

    def constant(name, data):
        constants.append(numpy_helper.from_array(np.asarray(data, np.int64), name))
        return name

    def reshape(name, source, shape):
        nodes.append(
            helper.make_node(
                "Reshape", [source, constant(name + ".shape", shape)], [name]
            )
        )
        return name

    def slice_axis(name, source, start, end, axis):
        nodes.append(
            helper.make_node(
                "Slice",
                [
                    source,
                    constant(name + ".start", [start]),
                    constant(name + ".end", [end]),
                    constant(name + ".axis", [axis]),
                ],
                [name],
            )
        )
        return name

    for name, shape, dtype in _states(inputs, batch_size):
        suffix = name.removeprefix("past.")
        recurrent = suffix.endswith(".recurrent")
        if not recurrent and not suffix.endswith(".conv"):
            raise ValueError(
                f"{name}: only .recurrent and .conv state replay is supported"
            )
        capture_name = "state_update." + (
            suffix.removesuffix(".recurrent") + ".recurrent_capsule"
            if recurrent
            else suffix.removesuffix(".conv") + ".conv_value"
        )
        capture = captures.get(capture_name)
        if capture is None:
            raise ValueError(
                f"{name}: missing {capture_name}; use an export with state capture"
            )
        capture_shape = list(capture.shape)
        expected_rank = 2 if recurrent else 3
        if len(capture_shape) != expected_rank:
            raise ValueError(
                f"{capture_name}: expected rank {expected_rank}, "
                f"received {capture_shape}"
            )
        if isinstance(capture_shape[0], int) and capture_shape[0] != batch_size:
            raise ValueError(f"{capture_name}: capture batch does not match state")
        capture_shape[0] = batch_size
        if capture.type != (
            "tensor(float)" if dtype == TensorProto.FLOAT else "tensor(float16)"
        ):
            raise ValueError(f"{capture_name}: capture dtype must match {name}")
        output = state_output(name)
        sources.append(helper.make_tensor_value_info(name, dtype, shape))
        sources.append(
            helper.make_tensor_value_info(capture_name, dtype, capture_shape)
        )
        outputs.append(helper.make_tensor_value_info(output, dtype, shape))
        if not recurrent:
            if len(shape) != 3 or shape[1] <= 0:
                raise ValueError(f"{name}: expected [batch, positive channels, width]")
            _, channels, width = shape
            if capture_shape != [batch_size, capacity, channels]:
                raise ValueError(
                    f"{capture_name}: expected [{batch_size}, {capacity}, "
                    f"{channels}], received {capture_shape}"
                )
            if width == 0:
                nodes.append(helper.make_node("Identity", [name], [output]))
                continue
            tail = slice_axis(name + ".tail", name, 1, width, 2)
            first = slice_axis(name + ".first", capture_name, 0, 1, 1)
            last = name + ".last"
            nodes.append(helper.make_node("Transpose", [first], [last], perm=[0, 2, 1]))
            nodes.append(helper.make_node("Concat", [tail, last], [output], axis=2))
            continue
        if len(shape) != 4 or dtype != TensorProto.FLOAT or min(shape) <= 0:
            raise ValueError(f"{name}: expected positive FP32 [batch, Hv, Dv, Dk]")
        _, heads, value_width, key_width = shape
        capsule_width = capture_shape[1]
        if not isinstance(capsule_width, int) or capsule_width % capacity:
            raise ValueError(f"{capture_name}: width must be divisible by capacity")
        key_elements = capsule_width // capacity - heads - heads * value_width
        if key_elements <= 0 or key_elements % key_width:
            raise ValueError(
                f"{capture_name}: invalid packed key section geometry for "
                f"width={capsule_width}, capacity={capacity}, state={shape}; "
                "check the export's capture capacity and state dimensions"
            )
        key_heads = key_elements // key_width
        if heads % key_heads or key_width > 256:
            raise ValueError(
                f"{name}: linear GatedDeltaNet requires Hv divisible by Hk "
                "and key width <= 256"
            )
        decay = slice_axis(name + ".decay", capture_name, 0, heads, 1)
        decay = reshape(name + ".decay4", decay, [batch_size, heads, 1, 1])
        decayed = name + ".decayed"
        nodes.append(helper.make_node("Mul", [name, decay], [decayed]))
        key_offset = capacity * heads
        key = slice_axis(
            name + ".key", capture_name, key_offset, key_offset + key_elements, 1
        )
        delta_offset = key_offset + capacity * key_elements
        delta = slice_axis(
            name + ".delta",
            capture_name,
            delta_offset,
            delta_offset + heads * value_width,
            1,
        )
        query = name + ".query"
        constants.append(
            numpy_helper.from_array(
                np.zeros([batch_size, key_elements], np.float32), query
            )
        )
        packed = name + ".packed"
        nodes.append(helper.make_node("Concat", [query, key, delta], [packed], axis=1))
        offsets = name + ".offsets"
        constants.append(
            numpy_helper.from_array(np.arange(batch_size + 1, dtype=np.int32), offsets)
        )
        ignored = name + ".ignored"
        nodes.append(
            helper.make_node(
                "GatedDeltaNet",
                [packed, "", "", offsets, "", "", decayed],
                [ignored, output],
                domain="com.microsoft",
                update_rule="linear",
                scale=1.0,
            )
        )
        values.append(
            helper.make_tensor_value_info(
                ignored, TensorProto.FLOAT, [batch_size, heads, value_width]
            )
        )
    return _model(nodes, sources, outputs, constants, values)
