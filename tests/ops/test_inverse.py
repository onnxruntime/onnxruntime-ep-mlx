"""Inverse v1: typed CPU primitive, independent references and strict live contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from onnx import TensorProto as T, helper, numpy_helper
import onnxruntime as ort
import pytest


DTYPES = [np.float32, np.float64]
ERRORS = (ort.capi.onnxruntime_pybind11_state.InvalidGraph,
          ort.capi.onnxruntime_pybind11_state.InvalidArgument,
          ort.capi.onnxruntime_pybind11_state.Fail)
EPFAIL = ort.capi.onnxruntime_pybind11_state.EPFail


def _model(dtype, shape, *, constant=None, crossing=False):
    tensor_type = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    nodes = []
    if crossing:
        nodes.append(helper.make_node("Neg", ["x"], ["negated"]))
    nodes.append(helper.make_node("Inverse", ["negated" if crossing else "x"], ["inverse"],
                                  domain="com.microsoft", name="inverse_target"))
    if crossing:
        nodes.append(helper.make_node("Neg", ["inverse"], ["y"]))
    return helper.make_model(
        helper.make_graph(nodes, "inverse",
                          [] if constant is not None else [
                              helper.make_tensor_value_info("x", tensor_type, shape)],
                          [helper.make_tensor_value_info(
                              "y" if crossing else "inverse", tensor_type, shape)],
                          initializer=[] if constant is None else [
                              numpy_helper.from_array(constant, "x")]),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10)


def _session(model, path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.enable_profiling = True
    options.profile_file_prefix = str(path)
    session = ort.InferenceSession(model.SerializeToString(), options,
                                   providers=["MLXExecutionProvider"], enable_fallback=False)
    session.disable_fallback()
    providers = session.get_providers()
    assert providers in (["MLXExecutionProvider"],
                         ["MLXExecutionProvider", "CPUExecutionProvider"]), providers
    return session


def _placement(session):
    events = json.loads(Path(session.end_profiling()).read_text())
    kernels = [e for e in events
               if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")]
    assert kernels, "expected actual kernel events"
    assert all(e.get("args", {}).get("provider") == "MLXExecutionProvider"
               for e in kernels), kernels


def _matrix(shape, dtype, offset=0):
    if not np.prod(shape):
        return np.empty(shape, dtype)
    n = shape[-1]
    raw = ((np.arange(np.prod(shape)).reshape(shape) + offset) % 7 - 3).astype(dtype)
    return raw + np.eye(n, dtype=dtype) * (np.sum(np.abs(raw), axis=-1)[..., :, None] + 1)


def _compare(actual, x, expected=None):
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    eps = np.finfo(x.dtype).eps
    if expected is None:
        expected = np.linalg.inv(x) if x.size else np.empty_like(x)
    np.testing.assert_allclose(actual, expected, rtol=8 * eps, atol=8 * eps)
    if x.size:
        identity = np.broadcast_to(np.eye(x.shape[-1]), x.shape)
        np.testing.assert_allclose(x.astype(np.float64) @ actual.astype(np.float64),
                                   identity, rtol=0, atol=16 * x.shape[-1] * eps)
        np.testing.assert_allclose(actual.astype(np.float64) @ x.astype(np.float64),
                                   identity, rtol=0, atol=16 * x.shape[-1] * eps)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("source", ["input", "intermediate", "initializer"])
@pytest.mark.parametrize("shape", [(2, 2), (2, 2, 2), (2, 1, 3, 3)])
def test_inverse_live_batched_sources(dtype, eager, source, shape, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    constant = _matrix(shape, dtype) if source == "initializer" else None
    session = _session(_model(dtype, shape, constant=constant,
                              crossing=source == "intermediate"), tmp_path / "live")
    for offset in [0, 3, 0]:
        x = constant if constant is not None else _matrix(shape, dtype, offset)
        _compare(session.run(None, {} if constant is not None else {"x": x})[0], x)
    _placement(session)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("rank", [2, 3])
def test_inverse_symbolic_empty_geometry_and_recovery(dtype, eager, rank, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    session = _session(_model(dtype, [None] * rank), tmp_path / "dynamic")
    shapes = ([(2, 2), (0, 0), (1, 1), (3, 3)] if rank == 2 else
              [(2, 2, 2), (0, 3, 3), (2, 0, 0), (0, 50_000, 50_000), (3, 1, 1)])
    for shape in shapes:
        x = _matrix(shape, dtype)
        _compare(session.run(None, {"x": x})[0], x)
    shape = (2, 3) if rank == 2 else (2, 2, 3)
    with pytest.raises(EPFAIL, match="Inverse: final two dimensions must be square"):
        session.run(None, {"x": np.ones(shape, dtype)})
    x = _matrix((2, 2) if rank == 2 else (2, 2, 2), dtype, 1)
    _compare(session.run(None, {"x": x})[0], x)
    _placement(session)


@pytest.mark.parametrize("dtype", DTYPES)
def test_inverse_manual_goldens_and_typed_range(dtype, tmp_path):
    session = _session(_model(dtype, [2, 2]), tmp_path / "golden")
    for x, expected in [
        ([[1, 2], [3, 5]], [[-5, 2], [3, -1]]),
        ([[0, 2], [4, 0]], [[0, 0.25], [0.5, 0]]),
    ]:
        _compare(session.run(None, {"x": np.asarray(x, dtype)})[0], np.asarray(x, dtype),
                 np.asarray(expected, dtype))
    # Both finite exponents distinguish real DOUBLE from an unnoticed FLOAT round trip.
    exponent = 200 if dtype == np.float64 else 20
    x = np.diag(np.asarray([2.0 ** exponent, 2.0 ** -exponent], dtype))
    expected = np.diag(np.asarray([2.0 ** -exponent, 2.0 ** exponent], dtype))
    actual = session.run(None, {"x": x})[0]
    np.testing.assert_array_equal(actual, expected)
    _compare(actual, x, expected)
    # Invertible but ill-conditioned: an exact binary triangular reference and residual.
    delta = 2.0 ** (-30 if dtype == np.float64 else -10)
    x = np.asarray([[1, 1], [0, delta]], dtype)
    expected = np.asarray([[1, -1 / delta], [0, 1 / delta]], dtype)
    _compare(session.run(None, {"x": x})[0], x, expected)
    _placement(session)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_inverse_nonfinite_rejection_and_recovery(dtype, value, tmp_path):
    session = _session(_model(dtype, [2, 2]), tmp_path / "nonfinite")
    x = np.eye(2, dtype=dtype)
    x[0, 0] = value
    with pytest.raises(EPFAIL, match="Inverse: nonfinite input is not supported"):
        session.run(None, {"x": x})
    x = _matrix((2, 2), dtype)
    _compare(session.run(None, {"x": x})[0], x)
    _placement(session)


def _singular_recovery(dtype_name, directory):
    dtype = np.dtype(dtype_name).type
    session = _session(_model(dtype, [None, None, None]), Path(directory) / "singular")
    for x in [np.asarray([[[1, 2], [2, 4]]], dtype),
              np.asarray([[[1, 0], [0, 1]], [[0, 0], [0, 0]]], dtype)]:
        with pytest.raises(EPFAIL, match="Inverse: CPU inversion/evaluation failed"):
            session.run(None, {"x": x})
        valid = _matrix((2, 2, 2), dtype, 2)
        _compare(session.run(None, {"x": valid})[0], valid)
    _placement(session)
    print("SINGULAR_ERROR_AND_SAME_SESSION_RECOVERY_OK", flush=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_inverse_singular_error_and_same_session_recovery_subprocess(dtype, tmp_path):
    child = subprocess.run(
        [sys.executable, "-c",
         "import os,resource,runpy,sys,onnxruntime as ort; "
         "resource.setrlimit(resource.RLIMIT_CORE,(0,0)); "
         "ort.register_execution_provider_library('MLXExecutionProvider', "
         "os.environ['ONNXRUNTIME_MLX_EP_LIB']); "
         "d=runpy.run_path(sys.argv[1]); d['_singular_recovery'](sys.argv[2],sys.argv[3])",
         str(Path(__file__).resolve()), np.dtype(dtype).name, str(tmp_path)],
        capture_output=True, text=True, timeout=60)
    (tmp_path / "singular.stdout").write_text(child.stdout)
    (tmp_path / "singular.stderr").write_text(child.stderr)
    assert child.returncode == 0, (child.returncode, child.stdout, child.stderr)
    assert "SINGULAR_ERROR_AND_SAME_SESSION_RECOVERY_OK" in child.stdout


def _concurrent_owner_threads(dtype_name, directory):
    dtype = np.dtype(dtype_name).type
    barrier = threading.Barrier(2, timeout=30)

    def worker(singular):
        session = _session(_model(dtype, [2, 2]), Path(directory) / str(singular))
        for offset in range(12):
            barrier.wait()
            x = _matrix((2, 2), dtype, offset)
            if singular:
                with pytest.raises(EPFAIL, match="Inverse: CPU inversion/evaluation failed"):
                    session.run(None, {"x": np.zeros((2, 2), dtype)})
            _compare(session.run(None, {"x": x})[0], x)
        _placement(session)
        return threading.get_ident()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, singular) for singular in [False, True]]
        owners = [future.result() for future in futures]
    assert len(set(owners)) == 2, owners
    print("INVERSE_TWO_OWNER_THREADS_RECOVERY_OK", owners, flush=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_inverse_concurrent_owner_threads_subprocess(dtype, tmp_path):
    child = subprocess.run(
        [sys.executable, "-c",
         "import os,resource,runpy,sys,onnxruntime as ort; "
         "resource.setrlimit(resource.RLIMIT_CORE,(0,0)); "
         "ort.register_execution_provider_library('MLXExecutionProvider', "
         "os.environ['ONNXRUNTIME_MLX_EP_LIB']); "
         "d=runpy.run_path(sys.argv[1]); d['_concurrent_owner_threads'](sys.argv[2],sys.argv[3])",
         str(Path(__file__).resolve()), np.dtype(dtype).name, str(tmp_path)],
        capture_output=True, text=True, timeout=90)
    (tmp_path / "concurrent.stdout").write_text(child.stdout)
    (tmp_path / "concurrent.stderr").write_text(child.stderr)
    assert child.returncode == 0, (child.returncode, child.stdout, child.stderr)
    assert "INVERSE_TWO_OWNER_THREADS_RECOVERY_OK" in child.stdout


def _session_churn(dtype_name, directory):
    dtype = np.dtype(dtype_name).type
    for index in range(12):
        session = _session(_model(dtype, [2, 2]), Path(directory) / f"churn_{index}")
        x = _matrix((2, 2), dtype, index)
        _compare(session.run(None, {"x": x})[0], x)
        with pytest.raises(EPFAIL, match="Inverse: CPU inversion/evaluation failed"):
            session.run(None, {"x": np.zeros((2, 2), dtype)})
        _compare(session.run(None, {"x": x})[0], x)
        _placement(session)
        del session
    print("INVERSE_SESSION_CHURN_RECOVERY_OK", flush=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_inverse_session_churn_leaks_subprocess(dtype, tmp_path):
    child = subprocess.run(
        ["leaks", "--atExit", "--", sys.executable, "-c",
         "import os,resource,runpy,sys,onnxruntime as ort; "
         "resource.setrlimit(resource.RLIMIT_CORE,(0,0)); "
         "ort.register_execution_provider_library('MLXExecutionProvider', "
         "os.environ['ONNXRUNTIME_MLX_EP_LIB']); "
         "d=runpy.run_path(sys.argv[1]); d['_session_churn'](sys.argv[2],sys.argv[3])",
         str(Path(__file__).resolve()), np.dtype(dtype).name, str(tmp_path)],
        env={**os.environ, "MallocStackLogging": "1"},
        capture_output=True, text=True, timeout=120)
    (tmp_path / "churn.stdout").write_text(child.stdout)
    (tmp_path / "churn.stderr").write_text(child.stderr)
    assert child.returncode == 0, (child.returncode, child.stdout, child.stderr)
    assert "INVERSE_SESSION_CHURN_RECOVERY_OK" in child.stdout
    assert "0 leaks for 0 total leaked bytes." in child.stdout


@pytest.mark.parametrize("shape", [[], [1], [2, 3], [2, 3, 2], None])
def test_inverse_invalid_geometry_and_unknown_rank(shape, tmp_path):
    with pytest.raises(ERRORS):
        _session(_model(np.float32, shape), tmp_path / "shape")


@pytest.mark.parametrize("dtype", [np.float16, np.int32, np.complex64])
def test_inverse_unclaimed_dtype(dtype, tmp_path):
    with pytest.raises(ERRORS):
        _session(_model(dtype, [2, 2]), tmp_path / "dtype")


@pytest.mark.parametrize("variant", [
    "missing_x", "missing_y", "extra_input", "extra_output", "no_inputs", "no_outputs",
    "output_dtype", "output_rank", "bf16", "attribute",
])
def test_inverse_exact_form_validation(variant, tmp_path):
    model = _model(np.float32, [2, 2])
    node = model.graph.node[0]
    if variant == "missing_x":
        node.input[0] = ""
    elif variant == "missing_y":
        node.output[0] = ""
    elif variant == "extra_input":
        node.input.append("x")
    elif variant == "extra_output":
        node.output.append("extra")
    elif variant == "no_inputs":
        del node.input[:]
    elif variant == "no_outputs":
        del node.output[:]
    elif variant == "output_dtype":
        model.graph.output[0].type.tensor_type.elem_type = T.DOUBLE
    elif variant == "output_rank":
        del model.graph.output[0].type.tensor_type.shape.dim[-1]
    elif variant == "bf16":
        model.graph.input[0].type.tensor_type.elem_type = T.BFLOAT16
        model.graph.output[0].type.tensor_type.elem_type = T.BFLOAT16
    else:
        node.attribute.append(helper.make_attribute("axis", 0))
    with pytest.raises(ERRORS):
        _session(model, tmp_path / variant)


@pytest.mark.parametrize("depth", [1, 2])
def test_inverse_nested_cpu_eager_live(depth, tmp_path):
    model = _model(np.float32, [2, 2], crossing=True)
    body = helper.make_graph(list(model.graph.node), "branch", [], list(model.graph.output))
    for level in range(depth):
        branch = helper.make_node("If", ["condition"], ["y"], then_branch=body, else_branch=body)
        body = helper.make_graph([branch], f"parent_{level}", [], list(model.graph.output))
    del model.graph.node[:]
    model.graph.node.extend(body.node)
    model.graph.input.append(helper.make_tensor_value_info("condition", T.BOOL, []))
    session = _session(model, tmp_path / "nested")
    for flag, offset in [(True, 0), (False, 2), (True, 0)]:
        x = _matrix((2, 2), np.float32, offset)
        _compare(session.run(None, {"x": x, "condition": np.asarray(flag)})[0], x)
    _placement(session)


def test_inverse_double_body_payload_remains_rejected(tmp_path):
    model = _model(np.float64, [2, 2])
    body = helper.make_graph(list(model.graph.node), "double_branch", [], list(model.graph.output))
    del model.graph.node[:]
    model.graph.node.append(helper.make_node(
        "If", ["condition"], ["inverse"], then_branch=body, else_branch=body))
    model.graph.input.append(helper.make_tensor_value_info("condition", T.BOOL, []))
    with pytest.raises(ERRORS):
        _session(model, tmp_path / "double_body")


@pytest.mark.parametrize("flow", ["Loop", "Scan", "ScanIf"])
@pytest.mark.parametrize("eager", [False, True])
def test_inverse_loop_scan_cpu_eager_live(flow, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    state = helper.make_tensor_value_info("state", T.FLOAT, [2, 2])
    next_state = helper.make_tensor_value_info("next_state", T.FLOAT, [2, 2])
    if flow == "Loop":
        body = helper.make_graph([
            helper.make_node("Inverse", ["state"], ["next_state"], domain="com.microsoft"),
        ], "inverse_loop", [
            helper.make_tensor_value_info("iteration", T.INT64, []),
            helper.make_tensor_value_info("condition", T.BOOL, []), state],
            [helper.make_tensor_value_info("condition", T.BOOL, []), next_state])
        node = helper.make_node("Loop", ["trip", "condition", "x"], ["y"], body=body)
        initializers = [numpy_helper.from_array(np.asarray(3, np.int64), "trip"),
                        numpy_helper.from_array(np.asarray(True), "condition")]
        shape, outputs = [2, 2], [helper.make_tensor_value_info("y", T.FLOAT, [2, 2])]
    else:
        body = helper.make_graph([
            helper.make_node("Inverse", ["element"], ["next_state"], domain="com.microsoft"),
            helper.make_node("Identity", ["next_state"], ["element_out"]),
        ], "inverse_scan", [state, helper.make_tensor_value_info("element", T.FLOAT, [2, 2])],
            [next_state, helper.make_tensor_value_info("element_out", T.FLOAT, [2, 2])])
        if flow == "ScanIf":
            branch = helper.make_graph(list(body.node), "inverse_branch", [], list(body.output))
            del body.node[:]
            body.node.append(helper.make_node(
                "If", ["template_condition"], ["next_state", "element_out"],
                then_branch=branch, else_branch=branch))
        node = helper.make_node("Scan", ["initial", "x"], ["final", "y"],
                                body=body, num_scan_inputs=1)
        initializers = [numpy_helper.from_array(np.eye(2, dtype=np.float32), "initial")]
        if flow == "ScanIf":
            initializers.append(numpy_helper.from_array(np.asarray(True), "template_condition"))
        shape = [None, 2, 2]
        outputs = [helper.make_tensor_value_info("final", T.FLOAT, [2, 2]),
                   helper.make_tensor_value_info("y", T.FLOAT, shape)]
    model = helper.make_model(
        helper.make_graph([node], "inverse_control",
                          [helper.make_tensor_value_info("x", T.FLOAT, shape)], outputs,
                          initializer=initializers),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10)
    session = _session(model, tmp_path / flow)
    for count, offset in [(3, 0), (1, 2), (0, 0), (3, 0)]:
        x = _matrix((2, 2) if flow == "Loop" else (count, 2, 2), np.float32, offset)
        actual = session.run(None, {"x": x})
        _compare(actual[-1], x)
        if flow != "Loop":
            if count:
                _compare(actual[0], x[-1])
            else:
                np.testing.assert_array_equal(actual[0], np.eye(2, dtype=np.float32))
                assert actual[-1].shape == (0, 2, 2)
    _placement(session)


def test_inverse_actual_eager_and_neighbor_routes(tmp_path):
    trace = tmp_path / "inverse_trace.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", Path(__file__).name,
         "-k", ("test_inverse_live_batched_sources or test_inverse_nested_cpu_eager_live "
                "or test_inverse_loop_scan_cpu_eager_live"),
         "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
             "ONNXRUNTIME_EP_MLX_NO_COMPILE": "0",
             "ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE": "0",
             "ONNXRUNTIME_EP_MLX_NO_PREFILL_COMPILE": "0"},
        check=True, timeout=120)
    paths = [e["args"] for e in json.loads(trace.read_text()) if e.get("cat") == "ep.path"]
    inverses = [e for e in paths
                if set(e.get("ops", "").split(",")) & {"Inverse", "If", "Loop", "Scan"}]
    assert inverses and all(e["path"] == "eager" for e in inverses)
    neighbors = [e for e in paths if e.get("ops") == "Neg"]
    assert neighbors
    assert any(e["path"] == "general" and e["cache"] == "HIT" for e in neighbors)
