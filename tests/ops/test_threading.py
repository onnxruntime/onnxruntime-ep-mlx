"""Loading on one thread and running exclusively on another must initialize MLX there."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT

import _models as m


@pytest.mark.parametrize("dtype,np_dtype", [(DT.FLOAT, np.float32), (DT.DOUBLE, np.float64)])
def test_load_on_main_thread_run_on_worker(dtype, np_dtype, tmp_path):
    model = m.make_model(
        "Add",
        [m.tensor("a", dtype, ["tokens", 8]), m.tensor("b", dtype, ["tokens", 8])],
        [m.tensor("out", dtype, ["tokens", 8])],
    )
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "mlx_worker")
    session = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    session.disable_fallback()
    with ThreadPoolExecutor(max_workers=1) as worker:
        for tokens in [3, 1, 3]:
            a = np.arange(tokens * 8, dtype=np_dtype).reshape(tokens, 8) / 7
            b = np.full((tokens, 8), 0.25, dtype=np_dtype)
            actual = worker.submit(session.run, None, {"a": a, "b": b}).result()[0]
            np.testing.assert_array_equal(actual, a + b)
    with open(session.end_profiling()) as profile:
        events = json.load(profile)
    providers = {
        event.get("args", {}).get("provider")
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    }
    assert providers == {"MLXExecutionProvider"}
