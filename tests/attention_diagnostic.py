"""Opt-in exact released Attention fixture diagnostics (not the backend adapter)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx.backend.test.loader import load_model_tests


NAMES = {
    "test_attention_23_boolmask_fullymasked_row_nan_robustness",
    "test_attention_causal_boolmask_nan_robustness",
    "test_attention_4d_causal_fp16",
    "test_attention_4d_causal_fp16_expanded",
    "test_attention_4d_gqa_causal_nonpad_decode_fp16",
}


def metrics(label, got, want, rtol, atol):
    matches = np.isclose(got, want, rtol=rtol, atol=atol, equal_nan=True)
    print(label, "mismatches", np.count_nonzero(~matches), "/", got.size, flush=True)
    for index in np.argwhere(~matches)[:12]:
        index = tuple(index)
        print("  ", index, "actual", got[index], "expected", want[index], flush=True)


def oracle(model, feeds):
    """Real-arithmetic SDPA, independently evaluated in float64."""
    q, k, v = (feeds[name].astype(np.float64) for name in ("Q", "K", "V"))
    k = np.repeat(k, q.shape[1] // k.shape[1], axis=1)
    v = np.repeat(v, q.shape[1] // v.shape[1], axis=1)
    scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(q.shape[-1])
    allowed = np.arange(k.shape[-2])[None, :] <= np.arange(q.shape[-2])[:, None]
    node = model.graph.node[0]
    if len(node.input) > 6 and node.input[6]:
        lengths = feeds[node.input[6]]
        offset = lengths[:, None, None, None] - q.shape[-2]
        allowed = np.arange(k.shape[-2]) <= (
            np.arange(q.shape[-2])[None, None, :, None] + offset
        )
        allowed &= np.arange(k.shape[-2]) < lengths[:, None, None, None]
    scores = np.where(allowed, scores, -np.inf)
    scores -= np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs @ v


def main():
    print("VERSIONS", onnx.__version__, ort.__version__, np.__version__, flush=True)
    lib = os.environ.get("ONNXRUNTIME_MLX_EP_LIB")
    if lib:
        print("DEVELOPMENT LIB", lib, flush=True)
        ort.register_execution_provider_library("MLXExecutionProvider", lib)
    cases = {c.name: c for c in load_model_tests(kind="node") if c.name in NAMES}
    assert set(cases) == NAMES
    for name, case in cases.items():
        print("\nCASE", name, "TOL", case.rtol, case.atol, flush=True)
        inputs, outputs = case.data_sets[0]
        inputs = [
            onnx.numpy_helper.to_array(x) if isinstance(x, onnx.TensorProto) else x
            for x in inputs
        ]
        feeds = dict(zip((v.name for v in case.model.graph.input), inputs, strict=True))
        expected = outputs[0]
        if "fp16" in name:
            ideal = oracle(case.model, feeds)
            metrics("float64 vs fixture", ideal, expected, case.rtol, case.atol)
            print("ideal maxabs error", np.max(np.abs(ideal - expected)), flush=True)
        providers = [["CPUExecutionProvider"]]
        if lib:
            providers.append(["MLXExecutionProvider", "CPUExecutionProvider"])
        for eps in providers:
            for disabled in [False, True]:
                with tempfile.TemporaryDirectory() as directory:
                    options = ort.SessionOptions()
                    options.log_severity_level = 3
                    options.enable_profiling = True
                    options.profile_file_prefix = str(Path(directory) / "profile")
                    options.optimized_model_filepath = str(Path(directory) / "optimized.onnx")
                    if disabled:
                        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                    session = ort.InferenceSession(case.model.SerializeToString(), options, providers=eps)
                    actual = session.run(None, feeds)[0]
                    events = json.loads(Path(session.end_profiling()).read_text())
                    optimized = onnx.load(options.optimized_model_filepath)
                    print("PROVIDERS", eps, "DISABLE_OPT", disabled, flush=True)
                    print("OPTIMIZED", [n.op_type for n in optimized.graph.node], flush=True)
                    print("EXECUTION", [
                        (e.get("name"), e["args"].get("provider"), e["args"].get("op_name"))
                        for e in events if e.get("cat") == "Node" and "provider" in e.get("args", {})
                    ], flush=True)
                    metrics("actual vs fixture", actual, expected, case.rtol, case.atol)
                    if "fp16" in name:
                        print("actual vs ideal maxabs", np.max(np.abs(actual - ideal)), flush=True)
                        for index in np.argwhere(~np.isclose(actual, expected, rtol=case.rtol, atol=case.atol))[:12]:
                            index = tuple(index)
                            print("  ideal", index, ideal[index], "rounded", np.float16(ideal[index]), flush=True)


if __name__ == "__main__":
    main()
