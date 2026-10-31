"""Inverse v1: typed CPU primitive, independent references and strict live contracts."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

T = ir.DataType
DTYPES = [np.float32, np.float64]
ERRORS = (ort.capi.onnxruntime_pybind11_state.InvalidGraph,
          ort.capi.onnxruntime_pybind11_state.InvalidArgument,
          ort.capi.onnxruntime_pybind11_state.Fail)
EPFAIL = ort.capi.onnxruntime_pybind11_state.EPFail
LEAK_ENV_KEYS = ("PATH", "MallocStackLogging", "DYLD_INSERT_LIBRARIES",
                 "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH")


def _value(name, dtype, shape):
    return ir.Value(name=name, type=ir.TensorType(dtype),
                    shape=None if shape is None else ir.Shape(shape))


def _model(dtype, shape, *, constant=None, crossing=False):
    tensor_type = ir.DataType.from_numpy(np.dtype(dtype))
    x = _value("x", tensor_type, shape)
    if constant is not None:
        x.const_value = ir.tensor(constant, name="x")
    inverse = _value("inverse", tensor_type, shape)
    nodes = []
    operand = x
    if crossing:
        operand = _value("negated", tensor_type, shape)
        nodes.append(ir.node("Neg", [x], outputs=[operand]))
    nodes.append(ir.node("Inverse", [operand], outputs=[inverse],
                         domain="com.microsoft", name="inverse_target"))
    output = inverse
    if crossing:
        output = _value("y", tensor_type, shape)
        nodes.append(ir.node("Neg", [inverse], outputs=[output]))
    return ir.Model(ir.Graph(
        [] if constant is not None else [x], [output], nodes=nodes,
        initializers=[] if constant is None else [x], name="inverse",
        opset_imports={"": 21, "com.microsoft": 1}), ir_version=10)


def _session(model, path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.enable_profiling = True
    options.profile_file_prefix = str(path)
    session = ort.InferenceSession(ir.to_proto(model).SerializeToString(), options,
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
         ("import os,resource,runpy,sys,onnxruntime as ort; "
         "resource.setrlimit(resource.RLIMIT_CORE,(0,0)); "
         "ort.register_execution_provider_library('MLXExecutionProvider', "
         "os.environ['ONNXRUNTIME_MLX_EP_LIB']); "
         "d=runpy.run_path(sys.argv[1]); d['_singular_recovery'](sys.argv[2],sys.argv[3])"),
         str(Path(__file__).resolve()), np.dtype(dtype).name, str(tmp_path)],
        check=False, capture_output=True, text=True, timeout=60)
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
         ("import os,resource,runpy,sys,onnxruntime as ort; "
         "resource.setrlimit(resource.RLIMIT_CORE,(0,0)); "
         "ort.register_execution_provider_library('MLXExecutionProvider', "
         "os.environ['ONNXRUNTIME_MLX_EP_LIB']); "
         "d=runpy.run_path(sys.argv[1]); d['_concurrent_owner_threads'](sys.argv[2],sys.argv[3])"),
         str(Path(__file__).resolve()), np.dtype(dtype).name, str(tmp_path)],
        check=False, capture_output=True, text=True, timeout=90)
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
def test_inverse_session_churn_leaks_subprocess(dtype, tmp_path, capsys):
    env = {**os.environ, "MallocStackLogging": "1"}
    tool = shutil.which("leaks", path=env.get("PATH"))
    assert tool is not None, "macOS leaks executable not found on PATH"
    tool_hash = hashlib.sha256(Path(tool).read_bytes()).hexdigest()
    ready = tmp_path / "churn.ready"
    command = [sys.executable, "-c",
         ("import gc,json,os,sys; "
         "import resource,runpy,onnxruntime as ort; "
         "resource.setrlimit(resource.RLIMIT_CORE,(0,0)); "
         "ort.register_execution_provider_library('MLXExecutionProvider', "
         "os.environ['ONNXRUNTIME_MLX_EP_LIB']); "
         "d=runpy.run_path(sys.argv[1]); d['_session_churn'](sys.argv[2],sys.argv[3]); "
         "gc.collect(); "
         "open(sys.argv[4],'w').write(str(os.getpid())); sys.stdin.readline()"),
         str(Path(__file__).resolve()), np.dtype(dtype).name, str(tmp_path), str(ready)]
    # Keep the final Python image alive after session destruction. Scanning its
    # PID avoids relying on an atExit hook surviving a framework Python exec.
    with (tmp_path / "churn.stdout").open("w+") as stdout, (
        tmp_path / "churn.stderr"
    ).open("w+") as stderr:
        child = subprocess.Popen(command, env=env, stdin=subprocess.PIPE,
                                 stdout=stdout, stderr=stderr, text=True)
        try:
            deadline = time.monotonic() + 120
            while not ready.exists():
                assert child.poll() is None, "session churn child exited before scan"
                assert time.monotonic() < deadline, "session churn readiness timed out"
                time.sleep(0.05)
            assert int(ready.read_text()) == child.pid
            report = subprocess.run([tool, str(child.pid)], env=env,
                                    check=False, capture_output=True, text=True, timeout=60)
        finally:
            try:
                child.communicate("\n", timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate()
                raise
            stdout.seek(0)
            stderr.seek(0)
            child_stdout, child_stderr = stdout.read(), stderr.read()
    (tmp_path / "leaks.stdout").write_text(report.stdout)
    (tmp_path / "leaks.stderr").write_text(report.stderr)
    with capsys.disabled():
        print(json.dumps({
            "record_kind": "inverse_leaks_launcher",
            "dtype": np.dtype(dtype).name,
            "resolved_tool": tool,
            "tool_realpath": str(Path(tool).resolve()),
            "tool_sha256": tool_hash,
            "argv": report.args,
            "child_argv": command,
            "launcher_environment": {k: env.get(k) for k in LEAK_ENV_KEYS},
            "returncode": report.returncode,
            "stdout": report.stdout,
            "stderr": report.stderr,
            "child_returncode": child.returncode,
            "child_stdout": child_stdout,
            "child_stderr": child_stderr,
        }, indent=2), flush=True)
    assert child.returncode == 0, (child.returncode, child_stdout, child_stderr)
    assert "INVERSE_SESSION_CHURN_RECOVERY_OK" in child_stdout
    assert report.returncode == 0, (report.returncode, report.stdout, report.stderr)
    assert "0 leaks for 0 total leaked bytes." in report.stdout + report.stderr, (
        report.stdout, report.stderr)


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
    graph = model.graph
    node = graph[0]
    inputs, outputs, attrs = list(node.inputs), list(node.outputs), {}
    if variant == "missing_x":
        inputs[0] = None
    elif variant == "missing_y":
        outputs[0].name = ""
    elif variant == "extra_input":
        inputs.append(inputs[0])
    elif variant == "extra_output":
        outputs.append(ir.val("extra"))
    elif variant == "no_inputs":
        inputs.clear()
    elif variant == "no_outputs":
        outputs.clear()
    elif variant == "output_dtype":
        outputs[0].type = ir.TensorType(T.DOUBLE)
    elif variant == "output_rank":
        outputs[0].shape = ir.Shape([2])
    elif variant == "bf16":
        inputs[0].type = outputs[0].type = ir.TensorType(T.BFLOAT16)
    else:
        attrs["axis"] = 0
    graph.remove(node)
    graph.append(ir.node("Inverse", inputs, outputs=outputs,
                         domain="com.microsoft", attributes=attrs))
    with pytest.raises(ERRORS):
        _session(model, tmp_path / variant)


@pytest.mark.parametrize("depth", [1, 2])
def test_inverse_nested_cpu_eager_live(depth, tmp_path):
    original = _model(np.float32, [2, 2], crossing=True)
    x = original.graph.inputs[0]
    condition = _value("condition", T.BOOL, [])
    body = original.graph
    body.inputs.clear()
    for level in range(depth):
        output = _value(f"nested_{level}", T.FLOAT, [2, 2])
        branch = ir.node("If", [condition], outputs=[output],
                         attributes={"then_branch": body, "else_branch": body.clone(allow_outer_scope_values=True)})
        body = ir.Graph([], [output], nodes=[branch], name=f"parent_{level}")
    body.inputs.extend([x, condition])
    body.opset_imports.update({"": 21, "com.microsoft": 1})
    model = ir.Model(body, ir_version=10)
    session = _session(model, tmp_path / "nested")
    for flag, offset in [(True, 0), (False, 2), (True, 0)]:
        x = _matrix((2, 2), np.float32, offset)
        _compare(session.run(None, {"x": x, "condition": np.asarray(flag)})[0], x)
    _placement(session)


def test_inverse_double_body_payload_remains_rejected(tmp_path):
    original = _model(np.float64, [2, 2])
    body = original.graph
    x = body.inputs.pop()
    condition = _value("condition", T.BOOL, [])
    output = _value("result", T.DOUBLE, [2, 2])
    node = ir.node("If", [condition], outputs=[output],
                   attributes={"then_branch": body, "else_branch": body.clone(allow_outer_scope_values=True)})
    model = ir.Model(ir.Graph([x, condition], [output], nodes=[node],
                             opset_imports={"": 21, "com.microsoft": 1}), ir_version=10)
    with pytest.raises(ERRORS):
        _session(model, tmp_path / "double_body")


@pytest.mark.parametrize("flow", ["Loop", "Scan", "ScanIf"])
@pytest.mark.parametrize("eager", [False, True])
def test_inverse_loop_scan_cpu_eager_live(flow, eager, monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if eager else "0")
    state = _value("state", T.FLOAT, [2, 2])
    next_state = _value("next_state", T.FLOAT, [2, 2])
    if flow == "Loop":
        iteration = _value("iteration", T.INT64, [])
        condition = _value("condition", T.BOOL, [])
        body = ir.Graph([iteration, condition, state], [condition, next_state], nodes=[
            ir.node("Inverse", [state], outputs=[next_state], domain="com.microsoft")],
            name="inverse_loop")
        trip = ir.val("trip", const_value=ir.tensor(np.asarray(3, np.int64)))
        flag = ir.val("condition", const_value=ir.tensor(np.asarray(True)))
        shape = [2, 2]
        x, y = _value("x", T.FLOAT, shape), _value("y", T.FLOAT, shape)
        node = ir.node("Loop", [trip, flag, x], outputs=[y], attributes={"body": body})
        initializers, outputs = [trip, flag], [y]
    else:
        element = _value("element", T.FLOAT, [2, 2])
        element_out = _value("element_out", T.FLOAT, [2, 2])
        nodes = [ir.node("Inverse", [element], outputs=[next_state], domain="com.microsoft"),
                 ir.node("Identity", [next_state], outputs=[element_out])]
        initial = ir.val("initial", const_value=ir.tensor(np.eye(2, dtype=np.float32)))
        initializers = [initial]
        if flow == "ScanIf":
            branch = ir.Graph([], [next_state, element_out], nodes=nodes, name="inverse_branch")
            condition = ir.val("template_condition", const_value=ir.tensor(np.asarray(True)))
            initializers.append(condition)
            next_state = _value("if_state", T.FLOAT, [2, 2])
            element_out = _value("if_element", T.FLOAT, [2, 2])
            nodes = [ir.node("If", [condition], outputs=[next_state, element_out],
                             attributes={"then_branch": branch, "else_branch": branch.clone(allow_outer_scope_values=True)})]
        body = ir.Graph([state, element], [next_state, element_out], nodes=nodes, name="inverse_scan")
        shape = [None, 2, 2]
        x = _value("x", T.FLOAT, shape)
        outputs = [_value("final", T.FLOAT, [2, 2]), _value("y", T.FLOAT, shape)]
        node = ir.node("Scan", [initial, x], outputs=outputs,
                       attributes={"body": body, "num_scan_inputs": 1})
    model = ir.Model(ir.Graph([x], outputs, nodes=[node], initializers=initializers,
                             name="inverse_control", opset_imports={"": 21, "com.microsoft": 1}),
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
