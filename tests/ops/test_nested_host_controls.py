"""Bound body controls stay exact and eager; direct host controls retain capture support.

Source-only fixtures until a scheduler-approved native validation window is available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT

from test_rem_controlflow import FLOAT, _const, _model, _t, if_model, loop_model


def _increment_body(prefix: str) -> ir.Graph:
    iteration = _t(f"{prefix}_iteration", DT.INT64, [])
    condition = _t(f"{prefix}_condition", DT.BOOL, [])
    state = _t(f"{prefix}_state", DT.FLOAT, [2])
    condition_out = _t(f"{prefix}_condition_out", DT.BOOL, [])
    state_out = _t(f"{prefix}_state_out", DT.FLOAT, [2])
    one = _const(f"{prefix}_one", np.ones(2, FLOAT))
    return ir.Graph(
        [iteration, condition, state],
        [condition_out, state_out],
        nodes=[
            ir.node("Identity", [condition], outputs=[condition_out]),
            ir.node("Add", [state, one], outputs=[state_out]),
        ],
        initializers=[one],
        name=f"{prefix}_body",
        opset_imports={"": 18},
    )


def nested_loop_model(*, initializer_controls: bool = False) -> bytes:
    """Inner trip/condition are the outer iteration/condition formal inputs."""
    trip = _t("trip", DT.INT64, [])
    condition = _t("condition", DT.BOOL, [])
    if initializer_controls:
        trip = _const("trip", np.array(4, np.int64))
        condition = _const("condition", np.array(True))
    state = _t("state", DT.FLOAT, [2])
    output = _t("output", DT.FLOAT, [2])
    iteration = _t("outer_iteration", DT.INT64, [])
    body_condition = _t("outer_condition", DT.BOOL, [])
    body_state = _t("outer_state", DT.FLOAT, [2])
    condition_out = _t("outer_condition_out", DT.BOOL, [])
    state_out = _t("outer_state_out", DT.FLOAT, [2])
    body = ir.Graph(
        [iteration, body_condition, body_state],
        [condition_out, state_out],
        nodes=[
            ir.node("Identity", [body_condition], outputs=[condition_out]),
            ir.node(
                "Loop", [iteration, body_condition, body_state], outputs=[state_out],
                attributes={"body": _increment_body("inner")},
            ),
        ],
        name="outer_body",
        opset_imports={"": 18},
    )
    return _model(ir.Graph(
        [state] if initializer_controls else [trip, condition, state],
        [output],
        nodes=[ir.node("Loop", [trip, condition, state], outputs=[output],
                       attributes={"body": body})],
        initializers=[trip, condition] if initializer_controls else [],
        name="nested_loop",
        opset_imports={"": 18},
    ))


def scan_control_model(kind: str) -> bytes:
    """Scan slices become nested If/Loop controls; include exact control echo outputs."""
    state = _t("state", DT.FLOAT, [2])
    conditions = _t("conditions", DT.BOOL, ["T"])
    final = _t("final", DT.FLOAT, [2])
    sequence = _t("sequence", DT.FLOAT, ["T", 2])
    condition_echo = _t("condition_echo", DT.BOOL, ["T"])
    body_state = _t("scan_state", DT.FLOAT, [2])
    condition = _t("scan_condition", DT.BOOL, [])
    next_state = _t("scan_next", DT.FLOAT, [2])
    step_state = _t("scan_step", DT.FLOAT, [2])
    step_condition = _t("scan_condition_echo", DT.BOOL, [])
    inputs = [state, conditions]
    outputs = [final, sequence, condition_echo]
    body_inputs = [body_state, condition]
    body_outputs = [next_state, step_state, step_condition]
    if kind == "If":
        branches = {}
        for name, op in [("then_branch", "Add"), ("else_branch", "Sub")]:
            one = _const(f"{name}_one", np.ones(2, FLOAT))
            result = _t(f"{name}_result", DT.FLOAT, [2])
            branches[name] = ir.Graph(
                [], [result], nodes=[ir.node(op, [body_state, one], outputs=[result])],
                initializers=[one], name=name, opset_imports={"": 18},
            )
        control = ir.node("If", [condition], outputs=[next_state], attributes=branches)
    else:
        assert kind == "Loop"
        trips = _t("trips", DT.INT64, ["T"])
        trip = _t("scan_trip", DT.INT64, [])
        trip_echo = _t("trip_echo", DT.INT64, ["T"])
        step_trip = _t("scan_trip_echo", DT.INT64, [])
        inputs.append(trips)
        outputs.append(trip_echo)
        body_inputs.append(trip)
        body_outputs.append(step_trip)
        control = ir.node(
            "Loop", [trip, condition, body_state], outputs=[next_state],
            attributes={"body": _increment_body("scan_inner")},
        )
    nodes = [
        control,
        ir.node("Identity", [next_state], outputs=[step_state]),
        ir.node("Identity", [condition], outputs=[step_condition]),
    ]
    if kind == "Loop":
        nodes.append(ir.node("Identity", [trip], outputs=[step_trip]))
    body = ir.Graph(
        body_inputs, body_outputs, nodes=nodes, name=f"scan_{kind}_body",
        opset_imports={"": 18},
    )
    return _model(ir.Graph(
        inputs, outputs,
        nodes=[ir.node("Scan", inputs, outputs=outputs,
                       attributes={"body": body, "num_scan_inputs": len(inputs) - 1})],
        name=f"scan_nested_{kind}", opset_imports={"": 18},
    ))


def _session(model: bytes, *, cpu: bool, profile: Path | None = None):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    if not cpu:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    session = ort.InferenceSession(
        model, options,
        providers=["CPUExecutionProvider"] if cpu else ["MLXExecutionProvider"],
    )
    session.disable_fallback()
    return session


def _assert_owned_profile(events: list[dict]) -> None:
    kernels = [
        event for event in events
        if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time")
    ]
    assert kernels, "no compute kernel events; provider ownership is unproven"
    assert all(
        event.get("args", {}).get("provider") == "MLXExecutionProvider"
        for event in kernels
    ), kernels


def _assert_exact_outputs(outputs: list, expected: list) -> None:
    assert len(outputs) == len(expected)
    for got, manual in zip(outputs, expected, strict=True):
        assert got.dtype == manual.dtype
        assert got.shape == manual.shape
        np.testing.assert_array_equal(got, manual)


def _check_replays(
    model: bytes,
    cases: list[tuple[dict, list]],
    profile: Path,
    *,
    manual_empty_scan: bool = False,
) -> None:
    cpu = _session(model, cpu=True)
    mlx = _session(model, cpu=False, profile=profile)
    try:
        for feeds, expected in cases:
            # Scan-16 defines zero-trip outputs, but ORT 1.31 CPU and ONNX 1.23.2's
            # reference Scan reject them. Only this explicit feed uses the schema oracle.
            empty_scan = manual_empty_scan and feeds["conditions"].shape == (0,)
            if not empty_scan:
                _assert_exact_outputs(cpu.run(None, feeds), expected)
            _assert_exact_outputs(mlx.run(None, feeds), expected)
    finally:
        profile_path = Path(mlx.end_profiling())
        try:
            events = json.loads(profile_path.read_text())
        finally:
            profile_path.unlink()
    _assert_owned_profile(events)


@pytest.fixture(params=["default", "eager"])
def control_mode(request, monkeypatch) -> None:
    for name in [
        "ONNXRUNTIME_EP_MLX_NO_COMPILE",
        "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE",
        "ONNXRUNTIME_EP_MLX_NO_PREFILL_COMPILE",
    ]:
        monkeypatch.delenv(name, raising=False)
    if request.param == "eager":
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")


@pytest.mark.parametrize("initializer_controls", [False, True])
def test_nested_loop_bound_controls(control_mode, tmp_path, initializer_controls) -> None:
    cases = []
    for trip, condition in [(0, True), (1, True), (3, True), (4, False), (4, True), (3, True)]:
        state = np.array([7, -5], FLOAT)
        feeds = {"state": state}
        if initializer_controls:
            trip, condition = 4, True
        else:
            feeds.update(trip=np.array(trip, np.int64), condition=np.array(condition))
        count = trip * (trip - 1) // 2 if condition else 0
        cases.append((feeds, [state + np.float32(count)]))
    _check_replays(nested_loop_model(initializer_controls=initializer_controls),
                   cases, tmp_path / "nested_loop")


@pytest.mark.parametrize("kind", ["If", "Loop"])
def test_scan_bound_controls_live_replays(control_mode, tmp_path, kind) -> None:
    cases = []
    for conditions, trips in [
        ([True, False, True], [0, 1, 3]),
        ([False, True, False], [3, 1, 0]),
        ([True, True, False], [3, 0, 1]),
        ([False, True, False], [2**63 - 1, -(2**63), 2**53 + 1]),
        ([], []),
        ([True, False, True], [0, 1, 3]),
    ]:
        state = np.array([11, -7], FLOAT)
        flags = np.array(conditions, np.bool_)
        counts = np.array(trips, np.int64)
        feeds = {"state": state, "conditions": flags}
        steps = []
        current = state.copy()
        for condition, trip in zip(conditions, trips, strict=True):
            delta = (1 if condition else -1) if kind == "If" else (
                max(trip, 0) if condition else 0
            )
            current = current + np.float32(delta)
            steps.append(current.copy())
        expected = [current, np.array(steps, FLOAT).reshape(-1, 2), flags]
        if kind == "Loop":
            feeds["trips"] = counts
            expected.append(counts)
        cases.append((feeds, expected))
    _check_replays(
        scan_control_model(kind), cases, tmp_path / f"scan_{kind}", manual_empty_scan=True,
    )


@pytest.mark.parametrize("kind", ["If", "Loop"])
def test_empty_scan_bound_controls_are_owned(control_mode, tmp_path, kind) -> None:
    """Empty-only profiles cannot borrow ownership evidence from nonempty replays."""
    state = np.array([11, -7], FLOAT)
    feeds = {"state": state, "conditions": np.empty(0, np.bool_)}
    expected = [state.copy(), np.empty((0, 2), FLOAT), np.empty(0, np.bool_)]
    if kind == "Loop":
        feeds["trips"] = np.empty(0, np.int64)
        expected.append(np.empty(0, np.int64))
    _check_replays(
        scan_control_model(kind), [(feeds, expected)], tmp_path / f"empty_scan_{kind}",
        manual_empty_scan=True,
    )


def test_direct_host_controls_live_replays(control_mode, tmp_path) -> None:
    """Neighbor coverage: direct runtime controls must not be evaluated trace placeholders."""
    state = np.array([2, -4], FLOAT)
    cases = [
        ({"M": np.array(trip, np.int64), "condin": np.array(condition), "acc0": state},
         [state + np.float32(trip if condition else 0)])
        for trip, condition in [(3, True), (3, True), (1, False), (1, True), (3, True)]
    ]
    _check_replays(loop_model(), cases, tmp_path / "direct_loop")
    x = np.array([2, -4, 8], FLOAT)
    cases = [
        ({"x": x, "cond": np.array(condition)}, [x + (1 if condition else -1)])
        for condition in [True, True, False, False, True]
    ]
    _check_replays(if_model(), cases, tmp_path / "direct_if")


@pytest.mark.parametrize("events", [
    [],
    [{"cat": "Node", "name": "fused_kernel_time", "args": {}}],
    [{"cat": "Node", "name": "fused_kernel_time",
      "args": {"provider": "CPUExecutionProvider"}}],
    [{"cat": "Node", "name": "fused_fence_before",
      "args": {"provider": "MLXExecutionProvider"}}],
])
def test_ownership_probe_rejects_missing_compute_or_provider(events) -> None:
    with pytest.raises(AssertionError):
        _assert_owned_profile(events)
