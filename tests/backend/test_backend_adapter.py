"""Released ONNX acquisition/adapter regressions that do not require Metal."""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.backend.base import Backend, BackendRep
from onnx.backend.test.loader import load_model_tests
from onnx.backend.test.runner import Runner
from onnx.reference import ReferenceEvaluator
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidArgument, NotImplemented

import _backend
from _backend import MlxBackend, MlxBackendRep, WANTED_CATEGORIES, backend_test_cases


@pytest.fixture(scope="module")
def node_cases():
    cases = load_model_tests(kind="node")
    assert cases, "ONNX generated no node models"
    return {case.name: case for case in cases}


@pytest.fixture
def cpu_adapter(monkeypatch):
    create_session = ort.InferenceSession

    def cpu_session(model, *, sess_options, providers):
        assert providers == [_backend.EP_NAME, "CPUExecutionProvider"]
        assert sess_options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return create_session(model, sess_options=sess_options, providers=["CPUExecutionProvider"])

    monkeypatch.setattr(_backend, "_ensure_registered", lambda: None)
    monkeypatch.setattr(_backend.ort, "InferenceSession", cpu_session)


def identity_model(*, ir=7, opset=13, domain="", domain_opset=1):
    imports = [helper.make_opsetid("", opset)]
    if domain:
        imports.append(helper.make_opsetid(domain, domain_opset))
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])],
            "identity",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        ),
        ir_version=ir,
        opset_imports=imports,
    )


def test_released_node_acquisition_is_in_memory(node_cases):
    for case in node_cases.values():
        assert isinstance(case.model, onnx.ModelProto)
        assert case.model_dir is None
        assert case.url is None
        assert case.data_sets
        for inputs, expected in case.data_sets:
            assert inputs is not None
            assert expected
        assert case.rtol >= 0 and case.atol >= 0
    assert "test_abs" in node_cases
    assert "test_pow_types_float32_int64" in node_cases
    assert "test_sequence_map_identity_1_sequence" in node_cases


class ReferenceRep(BackendRep):
    def __init__(self, model):
        self.ref = ReferenceEvaluator(model)

    def run(self, inputs, **kwargs):
        return self.ref.run(None, dict(zip(self.ref.input_names, inputs)))


class ReferenceBackend(Backend):
    @classmethod
    def prepare(cls, model, device="CPU", **kwargs):
        return ReferenceRep(model)

    @classmethod
    def supports_device(cls, device):
        return device == "CPU"


@pytest.fixture(scope="module")
def reference_runner():
    return Runner(ReferenceBackend, __name__)


def test_runner_keeps_all_intended_categories(reference_runner, node_cases):
    cases = backend_test_cases(reference_runner)
    assert tuple(cases) == WANTED_CATEGORIES
    node_names = unittest.defaultTestLoader.getTestCaseNames(cases[WANTED_CATEGORIES[0]])
    assert len([name for name in node_names if name.endswith("_cpu")]) == len(node_cases)
    assert not any("Real" in name for name in cases)


@pytest.mark.parametrize(
    "name",
    ["test_abs_cpu", "test_pow_types_float32_int64_cpu", "test_sequence_map_identity_1_sequence_cpu"],
)
def test_public_runner_executes_reference_data(reference_runner, name):
    case = backend_test_cases(reference_runner)[WANTED_CATEGORIES[0]]
    result = unittest.TestResult()
    case(name).run(result)
    assert result.testsRun == 1
    assert not result.skipped
    assert result.wasSuccessful(), (result.errors, result.failures)


def test_runner_preserves_all_datasets_tensor_protos_and_tolerances(monkeypatch, node_cases):
    inputs = np.array([-1.0, 2.0], dtype=np.float32)
    case = replace(
        node_cases["test_abs"],
        name="test_adapter_datasets",
        data_sets=[
            ([numpy_helper.from_array(inputs)], [numpy_helper.from_array(np.abs(inputs))]),
            ([inputs * 2], [np.abs(inputs * 2) + np.float32(1e-4)]),
        ],
        rtol=2e-3,
        atol=3e-5,
    )
    observed = []
    load = load_model_tests

    def load_cases(*args, kind=None, **kwargs):
        return [case] if kind == "node" else load(*args, kind=kind, **kwargs)

    class CheckingRunner(Runner):
        @classmethod
        def assert_similar_outputs(cls, ref_outputs, outputs, rtol, atol, model_dir=None):
            observed.append((rtol, atol, model_dir))
            super().assert_similar_outputs(ref_outputs, outputs, rtol, atol, model_dir)

    monkeypatch.setattr("onnx.backend.test.runner.load_model_tests", load_cases)
    runner = CheckingRunner(ReferenceBackend, __name__)
    tests = backend_test_cases(runner)[WANTED_CATEGORIES[0]]
    result = unittest.TestResult()
    tests("test_adapter_datasets_cpu").run(result)
    assert result.wasSuccessful(), (result.errors, result.failures)
    assert not result.skipped
    assert observed == [(case.rtol, case.atol, None)] * len(case.data_sets)


@pytest.mark.parametrize("missing", WANTED_CATEGORIES)
def test_empty_or_missing_category_is_collection_error(missing):
    class Nonempty(unittest.TestCase):
        def test_present_cpu(self):
            pass

    cases = dict.fromkeys(WANTED_CATEGORIES, Nonempty)
    del cases[missing]
    with pytest.raises(RuntimeError, match=missing):
        backend_test_cases(SimpleNamespace(test_cases=cases))
    cases[missing] = unittest.TestCase
    with pytest.raises(RuntimeError, match=missing):
        backend_test_cases(SimpleNamespace(test_cases=cases))


@pytest.mark.parametrize(
    "name",
    ["test_abs", "test_pow_types_float32_int64", "test_sequence_map_identity_1_sequence"],
)
def test_supported_generated_models_run_on_cpu(cpu_adapter, node_cases, name):
    case = node_cases[name]
    original = case.model.SerializeToString()
    rep = MlxBackend.prepare(case.model)
    for inputs, expected in case.data_sets:
        inputs = [numpy_helper.to_array(x) if isinstance(x, TensorProto) else x for x in inputs]
        expected = [numpy_helper.to_array(x) if isinstance(x, TensorProto) else x for x in expected]
        Runner.assert_similar_outputs(expected, rep.run(inputs), case.rtol, case.atol)
    assert case.model.SerializeToString() == original


@pytest.mark.parametrize(
    ("model", "reason"),
    [
        (identity_model(ir=14), "Unsupported model IR version: 14"),
        (identity_model(opset=999), r"ai.onnx opset 999"),
        (identity_model(domain="ai.onnx.ml", domain_opset=999), r"ai.onnx.ml opset 999"),
    ],
)
def test_runtime_version_fences_skip_without_rewriting(cpu_adapter, model, reason):
    original = model.SerializeToString()
    with pytest.raises(unittest.SkipTest, match=reason):
        MlxBackend.prepare(model)
    assert model.SerializeToString() == original


def test_generated_node_runner_reports_version_skip(cpu_adapter):
    class VersionBackend(MlxBackend):
        @classmethod
        def is_compatible(cls, model, device="CPU", **kwargs):
            pytest.fail("Generated-node Runner must exercise prepare, not is_compatible")

    # A genuine latest-ONNX node model that exceeds ORT 1.31's IR limit.
    runner = Runner(VersionBackend, __name__)
    case = backend_test_cases(runner)[WANTED_CATEGORIES[0]]
    result = unittest.TestResult()
    case("test_optional_get_element_tensor_cpu").run(result)
    assert result.testsRun == 1
    assert len(result.skipped) == 1
    assert "Unsupported model IR version" in result.skipped[0][1]
    assert result.wasSuccessful()


@pytest.mark.parametrize(
    "error",
    [
        Fail("provider initialization failed"),
        Fail("malformed graph: invalid input"),
        InvalidArgument("invalid input shape"),
        NotImplemented("Could not find an implementation for Pow(15)"),
        RuntimeError("MLX provider failure"),
        Fail("Unsupported model IR version: 999, max supported IR version: 13"),
        Fail("ONNX Runtime only *guarantees* support. "
             "Current official support for domain ai.onnx.ml is till opset 5."),
        Fail("ONNX Runtime only *guarantees* support. Opset 28 is under development. "
             "Current official support for domain ai.onnx is till opset 5."),
        Fail("Current official support for domain ai.onnx is till opset 5."),
    ],
)
def test_unrelated_session_errors_propagate(monkeypatch, error):
    monkeypatch.setattr(_backend, "_ensure_registered", lambda: None)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(_backend.ort, "InferenceSession", fail)
    with pytest.raises(type(error)) as caught:
        MlxBackend.prepare(identity_model())
    assert caught.value is error


def test_registration_errors_propagate(monkeypatch):
    def fail():
        raise RuntimeError("provider registration failed")

    monkeypatch.setattr(_backend, "_ensure_registered", fail)
    with pytest.raises(RuntimeError, match="provider registration failed"):
        MlxBackend.prepare(identity_model())


def test_malformed_graph_is_not_a_compatibility_skip(cpu_adapter):
    model = identity_model()
    model.graph.node[0].input[0] = "not_a_graph_input"
    with pytest.raises(onnx.checker.ValidationError):
        MlxBackend.prepare(model)


def test_known_pow_cpu_kernel_failure_is_not_skipped(cpu_adapter, node_cases):
    with pytest.raises(NotImplemented, match="Pow"):
        MlxBackend.prepare(node_cases["test_pow_types_float32_uint64"].model)


def test_sequence_optional_and_multiple_runs_preserve_values():
    seen = []

    class Session:
        def run(self, outputs, feeds):
            seen.append((outputs, feeds))
            return [feeds["sequence"], feeds["optional"]]

    rep = MlxBackendRep(Session(), ["sequence", "optional"], ["s", "o"])
    for value in (1, 2):
        sequence = [np.array([value], dtype=np.float32), np.array([value + 1], dtype=np.float32)]
        assert rep.run([sequence, None])[1] is None
        assert isinstance(seen[-1][1]["sequence"], list)
        for actual, expected in zip(seen[-1][1]["sequence"], sequence):
            np.testing.assert_array_equal(actual, expected)
    assert len(seen) == 2


def test_output_mismatch_is_a_failure(reference_runner):
    with pytest.raises(AssertionError):
        reference_runner.assert_similar_outputs(
            [np.array([1.0], dtype=np.float32)],
            [np.array([2.0], dtype=np.float32)],
            rtol=1e-3,
            atol=1e-7,
        )
