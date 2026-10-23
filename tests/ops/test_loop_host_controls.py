"""Loop controls must be readable by both eager translation and specialization."""

import json

import numpy as np
import onnx
from onnx import TensorProto as T, helper
import onnxruntime as ort
import pytest


def _model(controls):
    scalar = lambda name, dtype: helper.make_tensor_value_info(name, dtype, [])
    vector = lambda name: helper.make_tensor_value_info(name, T.FLOAT, [2])
    one = helper.make_tensor("one", T.FLOAT, [], [1.0])
    body = helper.make_graph(
        [
            helper.make_node("Identity", ["body_cond"], ["next_cond"]),
            helper.make_node("Add", ["body_state", "one"], ["next_state"]),
        ],
        "loop_body",
        [scalar("iteration", T.INT64), scalar("body_cond", T.BOOL), vector("body_state")],
        [scalar("next_cond", T.BOOL), vector("next_state")],
        [one],
    )
    nodes = []
    trip, condition = "trip", "condition"
    if controls in ("trip", "both"):
        nodes.append(helper.make_node("Add", ["trip", "delta"], ["computed_trip"]))
        trip = "computed_trip"
    if controls in ("condition", "both"):
        nodes.append(helper.make_node("And", ["condition", "gate"], ["computed_condition"]))
        condition = "computed_condition"
    nodes.append(helper.make_node("Loop", [trip, condition, "state"], ["result"], body=body))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "loop_host_controls",
            [
                scalar("trip", T.INT64),
                scalar("delta", T.INT64),
                scalar("condition", T.BOOL),
                scalar("gate", T.BOOL),
                vector("state"),
            ],
            [vector("result")],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


@pytest.mark.parametrize("eager", [False, True], ids=["default", "eager"])
@pytest.mark.parametrize("controls", ["host", "trip", "condition", "both"])
def test_loop_live_host_control_boundary(controls, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", str(int(eager)))
    model = _model(controls)
    cpu = ort.InferenceSession(model, providers=["CPUExecutionProvider"], enable_fallback=False)
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "loop_controls")
    if controls == "host":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model,
        options,
        providers=["MLXExecutionProvider", "CPUExecutionProvider"] if controls != "host"
        else ["MLXExecutionProvider"],
        enable_fallback=False,
    )
    session.disable_fallback()
    for index, (trip, condition, gate) in enumerate(
        [(0, True, True), (3, True, True), (1, True, False), (2, False, True), (2, True, True)]
    ):
        state = np.array([index, -index], dtype=np.float32)
        feeds = {
            "trip": np.array(trip - int(controls in ("trip", "both")), dtype=np.int64),
            "delta": np.array(1, dtype=np.int64),
            "condition": np.array(condition),
            "gate": np.array(gate),
            "state": state,
        }
        active = condition and (gate if controls in ("condition", "both") else True)
        expected = state + np.float32(trip if active else 0)
        reference = cpu.run(None, feeds)[0]
        actual = session.run(None, feeds)[0]
        assert actual.dtype == np.float32 and actual.shape == state.shape
        np.testing.assert_array_equal(reference, expected)
        np.testing.assert_array_equal(actual, expected)
    with open(session.end_profiling()) as source:
        events = json.load(source)
    kernels = [
        event.get("args", {}) for event in events
        if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time")
    ]
    assert kernels, events
    assert all(event.get("provider") for event in kernels), kernels
    if controls == "host":
        assert all(event["provider"] == "MLXExecutionProvider" for event in kernels), kernels
    else:
        loops = [event for event in kernels if event.get("op_name") == "Loop"]
        assert len(loops) == 5, kernels
        assert all(event["provider"] == "CPUExecutionProvider" for event in loops), loops
