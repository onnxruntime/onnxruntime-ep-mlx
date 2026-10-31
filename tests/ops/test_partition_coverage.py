"""Keep supported eager/compiled branches native when fusions interleave."""

import json

import _models as m
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest


@pytest.mark.parametrize("no_compile", [False, True])
def test_interleaved_eager_regions_preserve_native_coverage(
    monkeypatch, tmp_path, no_compile
):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1" if no_compile else "0")
    x = m.tensor("x", ir.DataType.INT64, [1])
    bounds = [
        m.tensor(name, ir.DataType.INT64, []) for name in ("start", "limit", "delta")
    ]
    tape = ir.tape.Tape()
    zero = tape.initializer(ir.tensor(np.array(0, np.int64), name="zero"))
    a = tape.op("Add", [x, zero])
    repeats = tape.op("Range", bounds)
    repeats.dtype, repeats.shape = ir.DataType.INT64, ir.Shape([1])
    added = tape.op("Add", [a, repeats])
    added.name, added.dtype, added.shape = "added", ir.DataType.INT64, ir.Shape([1])
    tiled = tape.op("Tile", [a, repeats])
    tiled.name, tiled.dtype, tiled.shape = (
        "tiled",
        ir.DataType.INT64,
        ir.Shape(["copies"]),
    )
    model = ir.Model(
        ir.Graph(
            [x, *bounds],
            [added, tiled],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 24},
            name="interleaved",
        ),
        ir_version=11,
    )
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "partition")
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        ir.to_proto(model).SerializeToString(),
        options,
        providers=["MLXExecutionProvider"],
        enable_fallback=False,
    )
    session.disable_fallback()
    for repeat in (2, 3, 0, 1, 2):
        values = np.array([10 + repeat], np.int64)
        feeds = {
            "x": values,
            "start": np.array(repeat, np.int64),
            "limit": np.array(repeat + 1, np.int64),
            "delta": np.array(1, np.int64),
        }
        actual = session.run(None, feeds)
        np.testing.assert_array_equal(actual[0], values + repeat)
        np.testing.assert_array_equal(actual[1], np.tile(values, repeat))

    with open(session.end_profiling()) as stream:
        events = json.load(stream)
    kernels = [
        event
        for event in events
        if event.get("args", {}).get("provider") == "MLXExecutionProvider"
        and event.get("name", "").endswith("_kernel_time")
    ]
    # Five executions keep Range/Tile values live. Eager execution needs one
    # native region, while capture retains its stricter operation boundaries.
    if no_compile:
        assert len(kernels) == 5, len(kernels)
    else:
        assert len(kernels) > 5, len(kernels)

    if no_compile:
        # Changing the diagnostic switch after construction must not capture
        # Range/Tile's first live values in a coalesced plan.
        monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0")
        for repeat in (4, 0, 1):
            values = np.array([20 + repeat], np.int64)
            actual = session.run(
                None,
                {
                    "x": values,
                    "start": np.array(repeat, np.int64),
                    "limit": np.array(repeat + 1, np.int64),
                    "delta": np.array(1, np.int64),
                },
            )
            np.testing.assert_array_equal(actual[0], values + repeat)
            np.testing.assert_array_equal(actual[1], np.tile(values, repeat))
