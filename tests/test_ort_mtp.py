"""K=1 alignment, rollback and CPU-readback boundaries in the public ORT loop."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from onnx import TensorProto, checker, helper, load_from_string, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
from test_mtp_replay import GatedDeltaNet

from bench.ort_mtp import main, make_last_two_logit_model, make_runner, run_once


def descriptor(name, shape, typ="tensor(float)"):
    return SimpleNamespace(name=name, shape=shape, type=typ)


def install_runtime(monkeypatch, mode):
    calls, values = [], []

    class Value:
        def __init__(self, shape, dtype, memory_info=None):
            self.data = np.zeros(shape, dtype)
            self.device = memory_info is not None
            self.readbacks = 0
            values.append(self)

        def update_inplace(self, data):
            self.data = data.copy()

        def numpy(self):
            assert not self.device, "state/hidden/logits must never be read on CPU"
            assert self.data.dtype == np.int64
            self.readbacks += 1
            return self.data

    class Binding:
        def __init__(self):
            self.inputs, self.outputs, self.cpu = {}, {}, {}

        def bind_ortvalue_input(self, name, value):
            self.inputs[name] = value

        def bind_ortvalue_output(self, name, value):
            self.outputs[name] = value

        def bind_cpu_input(self, name, value):
            self.cpu[name] = value.copy()

        def clear_binding_inputs(self):
            self.inputs.clear()
            self.cpu.clear()

        def clear_binding_outputs(self):
            self.outputs.clear()

    class Session:
        instances = []

        def __init__(self, model=None, *_args, kind=None, **_kwargs):
            self.instances.append(self)
            self.kind = kind
            self.head_past = 0
            self.last_binding = None
            self.last_two_logits = False
            if model is not None:
                self.model = load_from_string(model)
                self.reference = ReferenceEvaluator(self.model, new_ops=[GatedDeltaNet])

        def io_binding(self):
            return Binding()

        def disable_fallback(self):
            pass

        def proto_descriptors(self, items):
            types = {
                TensorProto.FLOAT: "tensor(float)",
                TensorProto.FLOAT16: "tensor(float16)",
                TensorProto.INT64: "tensor(int64)",
                TensorProto.INT32: "tensor(int32)",
            }
            return [
                descriptor(
                    item.name,
                    [
                        d.dim_value if d.HasField("dim_value") else d.dim_param
                        for d in item.type.tensor_type.shape.dim
                    ],
                    types[item.type.tensor_type.elem_type],
                )
                for item in items
            ]

        def get_inputs(self):
            if self.kind is None:
                return self.proto_descriptors(self.model.graph.input)
            inputs = [
                descriptor("input_ids", ["num_tokens"], "tensor(int64)"),
                descriptor("position_ids", [3, "num_tokens"], "tensor(int64)"),
                descriptor("block_table", [1, "max_num_blocks"], "tensor(int32)"),
                descriptor("cumulative_sequence_lengths", [2], "tensor(int32)"),
                descriptor("past_sequence_lengths", [1], "tensor(int32)"),
                descriptor("attention_metadata", [3], "tensor(int32)"),
                descriptor("past_key_values.0.key", ["num_blocks", "block_size", 1, 1]),
                descriptor(
                    "past_key_values.0.value", ["num_blocks", "block_size", 1, 1]
                ),
            ]
            if self.kind == "target":
                inputs += [
                    descriptor("past.0.recurrent", [1, 1, 8, 8]),
                    descriptor("past.0.conv", [1, 24, 3]),
                    descriptor("state_update_capture_count", [1], "tensor(int32)"),
                    descriptor("state_update_active", [1], "tensor(int32)"),
                ]
            else:
                inputs.append(descriptor("hidden_states", ["num_tokens", 2]))
            return inputs

        def get_outputs(self):
            if self.kind is None:
                return self.proto_descriptors(self.model.graph.output)
            outputs = [
                descriptor("logits", [
                    "logit_tokens" if self.last_two_logits else "num_tokens", 5
                ]),
                descriptor(
                    "hidden_states" if self.kind != "draft" else "hidden_states_out",
                    ["num_tokens", 2],
                ),
                descriptor("present.0.key", ["num_blocks", "block_size", 1, 1]),
                descriptor("present.0.value", ["num_blocks", "block_size", 1, 1]),
            ]
            if self.kind == "target":
                outputs += [
                    descriptor("present.0.recurrent", [1, 1, 8, 8]),
                    descriptor("present.0.conv", [1, 24, 3]),
                    descriptor("state_update.0.recurrent_capsule", [1, 68]),
                    descriptor("state_update.0.conv_value", [1, 4, 24]),
                ]
            return outputs

        def run_with_iobinding(self, binding):
            self.last_binding = binding
            feeds = {**binding.cpu, **{k: v.data for k, v in binding.inputs.items()}}
            if self.kind is None:
                arrays = self.reference.run(None, feeds)
                for output, data in zip(self.get_outputs(), arrays, strict=True):
                    binding.outputs[output.name].data = data
                calls.append(
                    ("helper", tuple(n.op_type for n in self.model.graph.node))
                )
                return
            ids, positions = feeds["input_ids"], feeds["position_ids"][0]
            past = int(feeds["past_sequence_lengths"][0])
            assert positions.size == ids.size
            np.testing.assert_array_equal(
                feeds["attention_metadata"], [len(ids), past + len(ids), 0]
            )
            calls.append((self.kind, past, ids.tolist(), positions.tolist()))
            prediction = (ids + 1) % 5
            result = {}
            if self.kind == "draft":
                if past == 0:
                    self.head_past = 0
                assert past == self.head_past
                self.head_past += len(ids)
                hidden = feeds["hidden_states"]
                np.testing.assert_array_equal((hidden[:, 0] + 1) % 5, ids)
                np.testing.assert_array_equal(hidden[:, 1] + 1, positions)
                if mode == "reject":
                    prediction = (ids + 2) % 5
                elif mode == "mixed" and ids[-1] % 3 == 1:
                    prediction[-1] = (ids[-1] + 2) % 5
            else:
                np.testing.assert_array_equal(
                    positions, np.arange(past, past + len(ids))
                )
                prefix = feeds["past_key_values.0.key"].reshape(-1)[:past]
                state = feeds["past.0.recurrent"]
                np.testing.assert_array_equal(state, np.full_like(state, prefix.sum()))
                history = np.pad(prefix, (max(0, 3 - past), 0))[-3:]
                np.testing.assert_array_equal(
                    feeds["past.0.conv"], np.tile(history, [1, 24, 1])
                )
                capture = bool(feeds["state_update_active"][0])
                assert feeds["state_update_capture_count"][0] == int(capture)
                capsule = np.zeros([1, 68], np.float32)
                conv_capture = np.zeros([1, 4, 24], np.float32)
                if capture:
                    capsule[0, 0] = 1
                    capsule[0, 4:12] = 1
                    capsule[0, 36:44] = ids[0]
                    conv_capture[0, 0] = ids[0]
                result["state_update.0.recurrent_capsule"] = capsule
                result["state_update.0.conv_value"] = conv_capture
                result["present.0.recurrent"] = state + np.float32(ids.sum())
                result["present.0.conv"] = np.tile(
                    np.r_[history, ids][-3:].astype(np.float32), [1, 24, 1]
                )
            for kind in ("key", "value"):
                cache = feeds["past_key_values.0." + kind].copy()
                cache.reshape(-1)[past : past + len(ids)] = ids
                result["present.0." + kind] = cache
            logits = np.full([len(ids), 5], -10, np.float32)
            logits[np.arange(len(ids)), prediction] = 10
            result["logits"] = (
                logits[-2:] if self.last_two_logits else logits
            )
            result[
                "hidden_states" if self.kind != "draft" else "hidden_states_out"
            ] = np.stack([ids, positions], axis=1).astype(np.float32)
            for name, data in result.items():
                assert binding.outputs[name].data.shape == data.shape
                binding.outputs[name].data = data

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            OrtValue=SimpleNamespace(ortvalue_from_shape_and_type=Value),
            InferenceSession=Session,
            SessionOptions=lambda: SimpleNamespace(
                add_session_config_entry=lambda *_: None
            ),
        ),
    )
    return Session(kind="target"), Session(kind="draft"), calls, values


class MatMulNBits(OpRun):
    op_domain = "com.microsoft"

    def _run(self, x, packed, scales, **_attributes):
        assert np.all(packed == 0x99)
        return (x.sum(axis=1, keepdims=True) * scales.reshape(1, -1),)


def companion_source_model():
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["x"], ["hidden_states"]),
                helper.make_node("Identity", ["state"], ["present"]),
                helper.make_node(
                    "MatMulNBits", ["hidden_states", "weight", "scales"], ["logits"],
                    domain="com.microsoft", K=32, N=3, bits=4, block_size=32,
                ),
            ],
            "prefill_companion",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, ["tokens", 32]),
                helper.make_tensor_value_info("state", TensorProto.FLOAT, [2]),
            ],
            [
                helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["tokens", 3]),
                helper.make_tensor_value_info(
                    "hidden_states", TensorProto.FLOAT, ["tokens", 32]
                ),
                helper.make_tensor_value_info("present", TensorProto.FLOAT, [2]),
            ],
            [
                numpy_helper.from_array(np.full((3, 1, 16), 0x99, np.uint8), "weight"),
                numpy_helper.from_array(np.array([[.25], [.5], [.75]], np.float32), "scales"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )


@pytest.mark.parametrize("tokens", [1, 2, 8, 32])
def test_last_two_logit_target_preserves_hidden_state_and_last_logits(tokens):
    original = companion_source_model()
    serialized = original.SerializeToString()
    companion = make_last_two_logit_model(original)
    checker.check_model(companion)
    assert original.SerializeToString() == serialized
    feeds = {
        "x": np.arange(tokens * 32, dtype=np.float32).reshape(tokens, 32),
        "state": np.array([7, 19], np.float32),
    }
    expected = ReferenceEvaluator(original, new_ops=[MatMulNBits]).run(None, feeds)
    actual = ReferenceEvaluator(companion, new_ops=[MatMulNBits]).run(None, feeds)
    np.testing.assert_array_equal(actual[0], expected[0][-2:])
    for result, reference in zip(actual[1:], expected[1:], strict=True):
        np.testing.assert_array_equal(result, reference)
    dimension = companion.graph.output[0].type.tensor_type.shape.dim[0]
    assert dimension.dim_param == "logit_tokens"


def test_last_two_logit_target_preserves_external_weight_metadata():
    from onnx.external_data_helper import set_external_data

    original = companion_source_model()
    set_external_data(original.graph.initializer[0], "weights.data", offset=128)
    original.graph.initializer[0].data_location = TensorProto.EXTERNAL
    companion = make_last_two_logit_model(original)
    assert companion.graph.initializer[0] == original.graph.initializer[0]


@pytest.mark.parametrize("fault", ["producer", "consumer", "rank"])
def test_last_two_logit_target_rejects_incompatible_graph(fault):
    original = companion_source_model()
    if fault == "producer":
        original.graph.node[-1].op_type = "MatMul"
    elif fault == "consumer":
        original.graph.node.append(helper.make_node("Identity", ["logits"], ["extra"]))
    else:
        original.graph.output[1].type.tensor_type.ClearField("shape")
    with pytest.raises(ValueError, match="compatible unpruned target export"):
        make_last_two_logit_model(original)


@pytest.mark.parametrize("mode", ["accept", "reject", "mixed"])
@pytest.mark.parametrize("chunk", [1, 2, 5, 8])
@pytest.mark.parametrize("speculative", [False, True])
def test_single_last_two_logit_target_preserves_prefill_and_mtp(
    monkeypatch, mode, chunk, speculative
):
    target, draft, calls, values = install_runtime(monkeypatch, mode)
    target.last_two_logits = True
    runner = make_runner(
        target, draft, [0, 1, 2, 3, 4], 12, info=object(), blocks=8,
        block_size=8, prefill_chunk_size=chunk, capacity=4,
        speculative=speculative, last_two_logits=True,
    )
    for prompt in ([0, 1, 2, 3, 4], [2, 3, 4], [3, 4]):
        assert runner(prompt, 12)["tokens"] == [i % 5 for i in range(12)]
    assert all(value.readbacks == 0 for value in values if value.device)
    assert any(call[0] == "target" for call in calls)

@pytest.mark.parametrize("declared,enabled", [(True, False), (False, True)])
def test_last_two_logit_target_requires_matching_contract(monkeypatch, declared, enabled):
    target, draft, _, _ = install_runtime(monkeypatch, "accept")
    target.last_two_logits = declared
    with pytest.raises(ValueError, match="last_two_logits=True"):
        make_runner(
            target, draft, [1], 2, info=object(), blocks=8, block_size=8,
            prefill_chunk_size=2, capacity=4, last_two_logits=enabled,
        )

def test_prefill_releases_bound_bulk_outputs_and_selector_inputs(monkeypatch):
    target, draft, _, _ = install_runtime(monkeypatch, "accept")
    run_once(
        target, draft, [0, 1, 2, 3, 4], 1, info=object(), blocks=8,
        block_size=8, prefill_chunk_size=5, capacity=4,
    )
    assert not {"logits", "hidden_states"} & target.last_binding.outputs.keys()
    for name, state in target.last_binding.inputs.items():
        if name.startswith(("past.", "past_key_values.")):
            from bench.ort_paged import state_output

            assert target.last_binding.outputs[state_output(name)] is state
    assert type(target).instances[-1].last_binding.inputs == {}


@pytest.mark.parametrize("mode", ["accept", "reject", "mixed"])
@pytest.mark.parametrize("generated", [1, 2, 8, 12])
@pytest.mark.parametrize("chunk", [2, 5])
def test_public_mtp_alignment_rollback_and_tail(monkeypatch, mode, generated, chunk):
    target, draft, calls, values = install_runtime(monkeypatch, mode)
    result = run_once(
        target,
        draft,
        [0, 1, 2, 3, 4],
        generated,
        info=object(),
        blocks=8,
        block_size=8,
        prefill_chunk_size=chunk,
        capacity=4,
    )
    assert result["tokens"] == [i % 5 for i in range(generated)]
    assert len(result["decode_intervals_seconds"]) == generated - 1
    stats = result["speculative_stats"]
    assert stats["completed_rounds"] == stats["draft_tokens_proposed"]
    assert stats["draft_forward_passes"] == len([c for c in calls if c[0] == "draft"])
    assert (
        stats["prefix_replays"]
        == stats["completed_rounds"] - stats["draft_tokens_accepted"]
    )
    if generated > 1:
        assert stats["completed_rounds"] > 0
        if mode == "accept":
            assert stats["prefix_replays"] == 0
            assert stats["draft_tokens_accepted"] == stats["completed_rounds"]
        elif mode == "reject":
            assert stats["draft_tokens_accepted"] == 0
        elif generated == 12:
            assert stats["draft_tokens_accepted"] and stats["prefix_replays"]
    assert all(v.readbacks == 0 for v in values if v.device)


def test_public_target_only_control_skips_speculation(monkeypatch):
    target, draft, calls, values = install_runtime(monkeypatch, "reject")
    result = run_once(
        target, draft, [0, 1, 2, 3, 4], 12, info=object(), blocks=8,
        block_size=8, prefill_chunk_size=2, capacity=4, speculative=False,
    )
    assert result["tokens"] == [i % 5 for i in range(12)]
    assert all(value == 0 for value in result["speculative_stats"].values())
    assert not any(call[0] == "draft" for call in calls)
    assert not any(
        call[0] == "helper" and ("GatedDeltaNet" in call[1] or set(call[1]) == {"Identity"})
        for call in calls
    )
    assert len([call for call in calls if call[0] == "target"]) == 14
    assert all(v.readbacks == 0 for v in values if v.device)


@pytest.mark.parametrize(
    "field,value",
    [
        ("generated", 0),
        ("blocks", 0),
        ("block_size", 0),
        ("prefill_chunk_size", 0),
        ("capacity", 0),
        ("info", None),
    ],
)
def test_public_mtp_invalid_sizes(monkeypatch, field, value):
    target, draft, _, _ = install_runtime(monkeypatch, "accept")
    args = {
        "generated": 2,
        "info": object(),
        "blocks": 8,
        "block_size": 8,
        "prefill_chunk_size": 2,
        "capacity": 4,
    }
    args[field] = value
    with pytest.raises(ValueError, match="nonempty prompt.*positive"):
        run_once(target, draft, [1], **args)


@pytest.mark.parametrize("name", ["logits", "hidden_states_out"])
@pytest.mark.parametrize("shape", [[], [2]])
def test_public_mtp_rejects_invalid_draft_output_rank(monkeypatch, name, shape):
    target, draft, _, _ = install_runtime(monkeypatch, "accept")
    outputs = draft.get_outputs()
    next(value for value in outputs if value.name == name).shape = shape
    monkeypatch.setattr(draft, "get_outputs", lambda: outputs)
    with pytest.raises(ValueError, match="widths/dtypes must match"):
        run_once(
            target,
            draft,
            [1],
            2,
            info=object(),
            blocks=8,
            block_size=8,
            prefill_chunk_size=2,
            capacity=4,
        )


def test_public_mtp_reports_missing_device(monkeypatch, tmp_path, capsys):
    install_runtime(monkeypatch, "accept")
    runtime = sys.modules["onnxruntime"]
    monkeypatch.setattr(
        runtime, "register_execution_provider_library", lambda *_: None, raising=False
    )
    monkeypatch.setattr(runtime, "get_ep_devices", lambda: [], raising=False)
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps({"prompt_tokens": [1], "runs": [{"tokens": [1] * 256}]})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ort_mtp",
            "--model",
            str(tmp_path),
            "--library",
            str(tmp_path / "plugin.dylib"),
            "--reference",
            str(reference),
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "no MLX device is available" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["accept", "reject", "mixed"])
def test_persistent_runner_resets_requests_without_reallocating_states(
    monkeypatch, mode
):
    target, draft, _, values = install_runtime(monkeypatch, mode)
    runner = make_runner(
        target,
        draft,
        [0, 1, 2, 3, 4],
        12,
        info=object(),
        blocks=8,
        block_size=8,
        prefill_chunk_size=2,
        capacity=4,
    )
    first = runner([0, 1, 2, 3, 4], 12)
    allocated = len(values)
    for prompt in ([0, 1, 2, 3, 4], [2, 3, 4], [3, 4]):
        result = runner(prompt, 12)
        assert result["tokens"] == first["tokens"]
        assert result["speculative_stats"] == first["speculative_stats"]
        assert len(values) == allocated
    assert all(v.readbacks == 0 for v in values if v.device)
