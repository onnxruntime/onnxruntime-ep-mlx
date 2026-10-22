"""ORT adapter and collection guards for the official ONNX backend runner."""

from __future__ import annotations

import os
import re
import unittest

import numpy as np
import onnxruntime as ort
from onnx import ModelProto, defs
from onnx.backend.base import Backend, BackendRep
from onnx.backend.test.runner import Runner
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidGraph

EP_NAME = os.environ.get("MLX_EP_NAME", "MLXExecutionProvider")
_registered = False

WANTED_CATEGORIES = (
    "OnnxBackendNodeModelTest",
    "OnnxBackendSimpleModelTest",
    "OnnxBackendPyTorchOperatorModelTest",
    "OnnxBackendPyTorchConvertedModelTest",
)


def backend_test_cases(runner: Runner) -> dict[str, type[unittest.TestCase]]:
    cases = runner.test_cases
    for name in WANTED_CATEGORIES:
        if name not in cases or not any(
            test.endswith("_cpu")
            for test in unittest.defaultTestLoader.getTestCaseNames(cases[name])
        ):
            raise RuntimeError(f"ONNX backend acquisition produced no tests for {name}")
    return {name: cases[name] for name in WANTED_CATEGORIES}


def _ensure_registered() -> None:
    global _registered
    if _registered:
        return
    lib = os.environ.get("ONNXRUNTIME_MLX_EP_LIB") or os.environ.get("MLX_EP_LIB")
    if not lib:
        raise RuntimeError("ONNXRUNTIME_MLX_EP_LIB/MLX_EP_LIB is not set")
    try:
        ort.register_execution_provider_library(EP_NAME, os.path.abspath(lib))
    except Exception as exc:  # a second registration in-process is benign
        if "already registered" not in str(exc).lower():
            raise
    _registered = True


class MlxBackendRep(BackendRep):
    def __init__(self, sess: ort.InferenceSession, inputs: list[str], outputs: list[str]):
        self._sess = sess
        self._inputs = inputs
        self._outputs = outputs

    def run(self, inputs, **kwargs):  # noqa: ANN001
        def _feed(val):
            if val is None:
                return None
            # Preserve sequence/optional inputs rather than collapsing to one tensor.
            if isinstance(val, (list, tuple)):
                return [np.asarray(v) for v in val]
            return np.asarray(val)

        feeds = {name: _feed(val) for name, val in zip(self._inputs, inputs)}
        return self._sess.run(self._outputs, feeds)


def _version_rejection(model: ModelProto, message: str) -> str | None:
    """Recognize only ORT's explicit IR/released-domain-opset limits."""
    ir = re.search(
        r"Unsupported model IR version: (\d+), max supported IR version: (\d+)",
        message,
    )
    if ir and model.ir_version == int(ir[1]) > int(ir[2]):
        return ir[0]
    opset = re.search(
        r"Current official support for domain (\S+) is till opset (\d+)\.",
        message,
    )
    requested = re.search(r"Opset (\d+) is under development", message)
    if opset and requested and "ONNX Runtime only *guarantees* support" in message:
        for imported in model.opset_import:
            domain = imported.domain or "ai.onnx"
            if domain == opset[1] and imported.version == int(requested[1]) > int(opset[2]):
                return f"{domain} opset {imported.version}: {opset[0]}"
    return None


def _schema_rejection(model: ModelProto, message: str) -> str | None:
    # ONNX 1.23 adds Attention-25 window attributes absent from ORT 1.31's
    # vendored schemas, even though its domain-level opset fence admits 25.
    rejected = re.search(
        r'In Node, \("([^"]*)", Attention, "", -1\).*?'
        r"Error Unrecognized attribute: (left_window_size|right_window_size) "
        r"for operator Attention\s*$",
        message,
        re.DOTALL,
    )
    if not rejected:
        return None
    version = next(
        (item.version for item in model.opset_import if item.domain == ""), None
    )
    if version is None or version < 25:
        return None
    schema = defs.get_schema("Attention", version, "")
    if schema.since_version < 25 or rejected[2] not in schema.attributes:
        return None
    for node in model.graph.node:
        if (
            node.name == rejected[1]
            and node.op_type == "Attention"
            and node.domain == ""
            and any(attr.name == rejected[2] for attr in node.attribute)
        ):
            return (
                f"Attention-{schema.since_version}: runtime rejected official "
                f"attribute {rejected[2]}"
            )
    return None


class MlxBackend(Backend):
    """Execute the unmodified model on MLX with CPU fallback available."""

    @classmethod
    def prepare(cls, model, device="CPU", **kwargs):  # noqa: ANN001
        super().prepare(model, device, **kwargs)
        _ensure_registered()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        try:
            sess = ort.InferenceSession(
                model.SerializeToString(),
                sess_options=opts,
                providers=[EP_NAME, "CPUExecutionProvider"],
            )
        except Fail as exc:
            # The generated-node Runner does not call Backend.is_compatible.
            reason = _version_rejection(model, str(exc))
            if reason is None:
                raise
            raise unittest.SkipTest(f"ONNX Runtime {ort.__version__}: {reason}") from exc
        except InvalidGraph as exc:
            reason = _schema_rejection(model, str(exc))
            if reason is None:
                raise
            raise unittest.SkipTest(f"ONNX Runtime {ort.__version__}: {reason}") from exc
        return MlxBackendRep(
            sess,
            [i.name for i in sess.get_inputs()],
            [o.name for o in sess.get_outputs()],
        )

    @classmethod
    def supports_device(cls, device: str) -> bool:
        # Apple unified memory: ORT tensors are CPU-addressable.
        return device == "CPU"
