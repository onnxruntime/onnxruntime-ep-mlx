"""Correctness + MLX-placement tests for the signal / FFT op family (signal.cc).

Placement assertions prevent vacuous CPU-fallback passes. NumPy is the FFT oracle for combinations
not implemented by ORT CPU (including IRFFT and reduced precision). Live scalar controls are
replayed in the same session, with compilation enabled and disabled.
"""

from __future__ import annotations

import json
import os

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT

import _models as m


def initz(name: str, arr: np.ndarray) -> ir.Value:
    """A constant-initializer value (const_value set) — read by the EP at translate time."""
    t = ir.tensor(arr, name=name)
    return ir.Value(
        name=name, type=ir.TensorType(t.dtype), shape=ir.Shape(list(arr.shape)), const_value=t
    )


def val(name: str, dtype: ir.DataType, shape: list) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def build(
    op: str,
    inputs: list[ir.Value | None],
    outputs: list[ir.Value],
    *,
    attrs: dict[str, object] | None = None,
    opset: int = 17,
) -> bytes:
    """Single-node model; constant-initializer inputs are pulled into the graph initializer list."""
    node = ir.node(op, inputs, attributes=attrs or {}, outputs=outputs)
    graph_inputs = [i for i in inputs if i is not None and i.const_value is None]
    initializers = [i for i in inputs if i is not None and i.const_value is not None]
    graph = ir.Graph(
        graph_inputs,
        outputs,
        nodes=[node],
        initializers=initializers,
        opset_imports={"": opset},
        name=f"mlx_{op}",
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


def _run_and_assert_mlx(
    model: bytes, feeds: dict[str, np.ndarray], *, disable_opt: bool = False
) -> list[np.ndarray]:
    """Run through the MLX EP and assert (via the ORT profile) at least one node ran on MLX.

    The window / mel ops take only constant inputs, so ORT's constant-folding pass would evaluate
    and erase the node before the EP ever sees it. ``disable_opt`` turns graph optimization off so
    the node survives to run on the MLX EP (matching how it executes inside a real fused subgraph).
    """
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = True
    options.profile_file_prefix = "mlx_signal_claim_probe"
    if disable_opt:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    session.disable_fallback()
    outputs = session.run(None, feeds)
    profile_path = session.end_profiling()
    try:
        with open(profile_path, encoding="utf-8") as profile:
            events = json.load(profile)
    finally:
        os.remove(profile_path)
    providers = {
        event.get("args", {}).get("provider")
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    }
    assert providers == {"MLXExecutionProvider"}, f"signal graph ran on {providers or 'no EP'}"
    return outputs


def _cpu(model: bytes, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    return ort.InferenceSession(model, providers=["CPUExecutionProvider"]).run(None, feeds)


def _assert_mlx_matches_cpu(model, feeds, *, rtol=1e-3, atol=1e-4, disable_opt=False) -> None:
    actual = _run_and_assert_mlx(model, feeds, disable_opt=disable_opt)
    expected = _cpu(model, feeds)
    assert len(actual) == len(expected)
    for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=f"output {i}")


# --- Cosine-sum windows -------------------------------------------------------------------------
@pytest.mark.parametrize("op", ["HannWindow", "HammingWindow", "BlackmanWindow"])
@pytest.mark.parametrize("periodic", [1, 0])
def test_window(op, periodic) -> None:
    size = 32
    model = build(
        op,
        [initz("size", np.array(size, np.int64))],
        [val("Y", DT.FLOAT, [size])],
        attrs={"periodic": periodic, "output_datatype": int(DT.FLOAT)},
    )
    # Windows are exact-ish: single cos per sample, so a tight tolerance holds against ORT CPU.
    _assert_mlx_matches_cpu(model, {}, rtol=1e-5, atol=1e-5, disable_opt=True)


def test_hann_window_fp16_output() -> None:
    size = 24
    model = build(
        "HannWindow",
        [initz("size", np.array(size, np.int64))],
        [val("Y", DT.FLOAT16, [size])],
        attrs={"periodic": 1, "output_datatype": int(DT.FLOAT16)},
    )
    _assert_mlx_matches_cpu(model, {}, rtol=1e-3, atol=1e-3, disable_opt=True)


# --- DFT ----------------------------------------------------------------------------------------
def test_dft_real_onesided_forward() -> None:
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((2, 16, 1)).astype(np.float32)
    model = build(
        "DFT",
        [val("X", DT.FLOAT, [2, 16, 1])],
        [val("Y", DT.FLOAT, ["b", "f", "c"])],
        attrs={"axis": 1, "inverse": 0, "onesided": 1},
    )
    _assert_mlx_matches_cpu(model, {"X": sig})


def test_dft_real_full_forward_with_dft_length() -> None:
    rng = np.random.default_rng(1)
    sig = rng.standard_normal((3, 10, 1)).astype(np.float32)
    model = build(
        "DFT",
        [val("X", DT.FLOAT, [3, 10, 1]), initz("dft_length", np.array(20, np.int64))],
        [val("Y", DT.FLOAT, ["b", "f", "c"])],
        attrs={"axis": 1, "inverse": 0, "onesided": 0},
    )
    _assert_mlx_matches_cpu(model, {"X": sig})


def test_dft_complex_full_forward() -> None:
    rng = np.random.default_rng(2)
    sig = rng.standard_normal((2, 12, 2)).astype(np.float32)
    model = build(
        "DFT",
        [val("X", DT.FLOAT, [2, 12, 2])],
        [val("Y", DT.FLOAT, ["b", "f", "c"])],
        attrs={"axis": 1, "inverse": 0, "onesided": 0},
    )
    _assert_mlx_matches_cpu(model, {"X": sig})


def test_dft_complex_inverse() -> None:
    rng = np.random.default_rng(3)
    sig = rng.standard_normal((2, 8, 2)).astype(np.float32)
    model = build(
        "DFT",
        [val("X", DT.FLOAT, [2, 8, 2])],
        [val("Y", DT.FLOAT, ["b", "f", "c"])],
        attrs={"axis": 1, "inverse": 1, "onesided": 0},
    )
    _assert_mlx_matches_cpu(model, {"X": sig})


def test_dft_opset20_axis_input() -> None:
    rng = np.random.default_rng(4)
    sig = rng.standard_normal((2, 16, 1)).astype(np.float32)
    model = build(
        "DFT",
        [
            val("X", DT.FLOAT, [2, 16, 1]),
            initz("dft_length", np.array(16, np.int64)),
            initz("axis", np.array(1, np.int64)),
        ],
        [val("Y", DT.FLOAT, ["b", "f", "c"])],
        attrs={"inverse": 0, "onesided": 1},
        opset=20,
    )
    _assert_mlx_matches_cpu(model, {"X": sig})


# --- STFT ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("onesided", [1, 0])
def test_stft_windowed(onesided) -> None:
    rng = np.random.default_rng(5)
    batch, signal_len, frame_length, frame_step = 2, 48, 16, 8
    sig = rng.standard_normal((batch, signal_len, 1)).astype(np.float32)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame_length) / frame_length)).astype(
        np.float32
    )
    model = build(
        "STFT",
        [
            val("signal", DT.FLOAT, [batch, signal_len, 1]),
            initz("frame_step", np.array(frame_step, np.int64)),
            initz("window", window),
            initz("frame_length", np.array(frame_length, np.int64)),
        ],
        [val("Y", DT.FLOAT, ["b", "f", "bins", "c"])],
        attrs={"onesided": onesided},
    )
    _assert_mlx_matches_cpu(model, {"signal": sig})


# --- MelWeightMatrix ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_mel_bins,dft_length,sample_rate,lower,upper",
    [(8, 64, 16000, 0.0, 8000.0), (20, 128, 16000, 20.0, 7600.0)],
)
def test_mel_weight_matrix(num_mel_bins, dft_length, sample_rate, lower, upper) -> None:
    num_spectrogram_bins = dft_length // 2 + 1
    model = build(
        "MelWeightMatrix",
        [
            initz("num_mel_bins", np.array(num_mel_bins, np.int64)),
            initz("dft_length", np.array(dft_length, np.int64)),
            initz("sample_rate", np.array(sample_rate, np.int64)),
            initz("lower_edge_hertz", np.array(lower, np.float32)),
            initz("upper_edge_hertz", np.array(upper, np.float32)),
        ],
        [val("Y", DT.FLOAT, [num_spectrogram_bins, num_mel_bins])],
        attrs={"output_datatype": int(DT.FLOAT)},
    )
    _assert_mlx_matches_cpu(model, {}, rtol=1e-5, atol=1e-6, disable_opt=True)


def _fft_expected(x, length, axis, inverse, onesided):
    signal = x[..., 0].astype(np.float32)
    if x.shape[-1] == 2:
        signal = signal + 1j * x[..., 1].astype(np.float32)
    if inverse and onesided:
        return np.fft.irfft(signal, n=length, axis=axis)[..., None].astype(x.dtype)
    result = (np.fft.ifft if inverse else np.fft.fft)(signal, n=length, axis=axis)
    if onesided:
        indices = [slice(None)] * result.ndim
        indices[axis] = slice(0, length // 2 + 1)
        result = result[tuple(indices)]
    return np.stack([result.real, result.imag], axis=-1).astype(x.dtype)


@pytest.fixture(params=[False, True], ids=["compiled", "eager"])
def execution_mode(request, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", str(int(request.param)))


@pytest.mark.parametrize("opset,axis", [(17, 0), (17, -2), (20, 0), (20, -2)])
@pytest.mark.parametrize("inverse,onesided,components", [(0, 0, 1), (0, 0, 2), (0, 1, 1),
                                                        (1, 0, 1), (1, 0, 2), (1, 1, 2)])
@pytest.mark.parametrize("length", [None, 5, 12])
def test_dft_combinations(opset, axis, inverse, onesided, components, length, execution_mode):
    x = np.random.default_rng(29).normal(size=(3, 7, components)).astype(np.float32)
    inputs = [val("X", DT.FLOAT, list(x.shape))]
    attrs = {"inverse": inverse, "onesided": onesided}
    if length is not None or opset >= 20:
        inputs.append(None if length is None else initz("length", np.array(length, np.int64)))
    if opset >= 20:
        inputs.append(initz("axis", np.array(axis, np.int64)))
    else:
        attrs["axis"] = axis
    model = build("DFT", inputs, [val("Y", DT.FLOAT, [None, None, 1 if inverse and onesided else 2])],
                  attrs=attrs, opset=opset)
    n = length or (2 * (x.shape[axis % x.ndim] - 1) if inverse and onesided else x.shape[axis % x.ndim])
    expected = _fft_expected(x, n, axis % x.ndim, inverse, onesided)
    got = _run_and_assert_mlx(model, {"X": x})[0]
    np.testing.assert_allclose(got, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("inverse,onesided,components", [(0, 0, 2), (0, 1, 1), (1, 0, 2), (1, 1, 2)])
def test_dft_float16(inverse, onesided, components, execution_mode):
    x = np.random.default_rng(30).normal(size=(2, 9, components)).astype(np.float16)
    model = build("DFT", [val("X", DT.FLOAT16, list(x.shape))],
                  [val("Y", DT.FLOAT16, [2, None, 1 if inverse and onesided else 2])],
                  attrs={"axis": 1, "inverse": inverse, "onesided": onesided})
    length = 16 if inverse and onesided else 9
    got = _run_and_assert_mlx(model, {"X": x})[0]
    assert got.dtype == np.float16
    np.testing.assert_allclose(got, _fft_expected(x, length, 1, inverse, onesided),
                               atol=4e-3, rtol=1e-3)


@pytest.mark.parametrize("components,onesided", [(1, 1), (1, 0), (2, 0)])
@pytest.mark.parametrize("optional", ["window", "length", "both"])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_stft_optional_complex_and_half(components, onesided, optional, dtype, execution_mode):
    x = np.random.default_rng(31).normal(size=(2, 29, components)).astype(dtype)
    frame, step = 7, 3
    window = np.linspace(0.25, 1, frame).astype(dtype)
    dt = DT.FLOAT if dtype == np.float32 else DT.FLOAT16
    inputs = [val("X", dt, [2, None, components]), initz("step", np.array(step, np.int32)),
              None if optional == "length" else val("window", dt, [None])]
    if optional != "window":
        inputs.append(initz("frame", np.array(frame, np.int32)))
    model = build("STFT", inputs, [val("Y", dt, [2, None, None, 2])],
                  attrs={"onesided": onesided})
    feeds = {"X": x}
    if optional != "length":
        feeds["window"] = window
    frames = np.stack([x[:, i:i + frame] for i in range(0, x.shape[1] - frame + 1, step)], axis=1)
    # Window multiplication is computed in fp32, like the FFT, before the final dtype restoration.
    frames = frames.astype(np.float32)
    if optional != "length":
        frames *= window.astype(np.float32)[None, None, :, None]
    expected = _fft_expected(frames, frame, 2, False, onesided).astype(dtype)
    got = _run_and_assert_mlx(model, feeds)[0]
    np.testing.assert_allclose(got, expected, atol=4e-3 if dtype == np.float16 else 2e-5, rtol=1e-3)


def _profile_providers(session):
    path = session.end_profiling()
    try:
        with open(path, encoding="utf-8") as f:
            return {e.get("args", {}).get("provider") for e in json.load(f) if e.get("cat") == "Node"}
    finally:
        os.remove(path)


def _live_session(model):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_profiling = True
    session = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    session.disable_fallback()
    return session


def test_dft_live_axis_length_and_specialization_limit(execution_mode):
    x = np.random.default_rng(32).normal(size=(4, 9, 1)).astype(np.float32)
    model = build("DFT", [val("X", DT.FLOAT, [None, None, 1]),
                          val("length", DT.INT64, []), val("axis", DT.INT64, [])],
                  [val("Y", DT.FLOAT, [None, None, 2])], opset=20)
    session = _live_session(model)
    # Exceed the bounded 16-entry cache, then revisit an earlier specialization.
    for length, axis in [(n, n % 2) for n in range(1, 20)] + [(3, 1), (6, 0), (3, 1)]:
        got = session.run(None, {"X": x, "length": np.array(length, np.int64),
                                  "axis": np.array(axis, np.int64)})[0]
        np.testing.assert_allclose(got, _fft_expected(x, length, axis, False, False), atol=2e-5)
    assert "MLXExecutionProvider" in _profile_providers(session)


def test_stft_live_step_length_and_shapes(execution_mode):
    model = build("STFT", [val("X", DT.FLOAT, [None, None, 1]), val("step", DT.INT64, []),
                           None, val("length", DT.INT64, [])],
                  [val("Y", DT.FLOAT, [None, None, None, 2])])
    session = _live_session(model)
    for batch, size, frame, step in [(2, 30, 8, 3), (1, 19, 5, 6), (3, 22, 7, 2), (2, 30, 8, 3)]:
        x = np.random.default_rng(size).normal(size=(batch, size, 1)).astype(np.float32)
        got = session.run(None, {"X": x, "step": np.array(step, np.int64),
                                 "length": np.array(frame, np.int64)})[0]
        frames = np.stack([x[:, i:i + frame] for i in range(0, size - frame + 1, step)], axis=1)
        np.testing.assert_allclose(got, _fft_expected(frames, frame, 2, False, True), atol=2e-5)
    assert "MLXExecutionProvider" in _profile_providers(session)


@pytest.mark.parametrize("op", ["HannWindow", "HammingWindow", "BlackmanWindow"])
@pytest.mark.parametrize("periodic", [0, 1])
def test_window_live_size(op, periodic, execution_mode):
    model = build(op, [val("size", DT.INT64, [])], [val("Y", DT.FLOAT, [None])],
                  attrs={"periodic": periodic})
    session = _live_session(model)
    for size in [7, 16, 2, 7]:
        feed = {"size": np.array(size, np.int64)}
        np.testing.assert_allclose(session.run(None, feed)[0], _cpu(model, feed)[0], atol=2e-6)
    assert "MLXExecutionProvider" in _profile_providers(session)


def test_mel_live_parameters(execution_mode):
    model = build("MelWeightMatrix", [val("bins", DT.INT64, []), val("length", DT.INT64, []),
                                      val("rate", DT.INT64, []), val("lower", DT.FLOAT, []),
                                      val("upper", DT.FLOAT, [])],
                  [val("Y", DT.FLOAT, [None, None])])
    session = _live_session(model)
    for bins, length, rate, lower, upper in [(8, 64, 16000, 0, 8000), (5, 32, 8000, 20, 3900),
                                           (8, 64, 16000, 100, 7000), (8, 64, 16000, 0, 8000)]:
        feed = {k: np.array(v, dt) for k, v, dt in
                [("bins", bins, np.int64), ("length", length, np.int64), ("rate", rate, np.int64),
                 ("lower", lower, np.float32), ("upper", upper, np.float32)]}
        np.testing.assert_allclose(session.run(None, feed)[0], _cpu(model, feed)[0], atol=1e-6)
    assert "MLXExecutionProvider" in _profile_providers(session)


@pytest.mark.parametrize("op", ["HannWindow", "HammingWindow", "BlackmanWindow", "MelWeightMatrix"])
@pytest.mark.parametrize("dtype", [DT.FLOAT16, DT.BFLOAT16])
def test_generator_interior_dtype(op, dtype, execution_mode):
    if op == "MelWeightMatrix":
        inputs = [initz("bins", np.array(5, np.int64)), initz("length", np.array(32, np.int64)),
                  initz("rate", np.array(8000, np.int64)), initz("lower", np.array(0, np.float32)),
                  initz("upper", np.array(4000, np.float32))]
        shape = [17, 5]
    else:
        inputs = [initz("size", np.array(20, np.int64))]
        shape = [20]
    y = val("generated", dtype, shape)
    z = val("Y", DT.FLOAT, shape)
    nodes = [ir.node(op, inputs, attributes={"output_datatype": int(dtype)}, outputs=[y]),
             ir.node("Cast", [y], attributes={"to": int(DT.FLOAT)}, outputs=[z])]
    graph = ir.Graph([], [z], nodes=nodes, initializers=inputs, opset_imports={"": 17})
    model = ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()
    # A float32 generator followed by a cast-to-float32 is the unrounded oracle.
    fp32 = build(op, [initz(i.name, i.const_value.numpy()) for i in inputs],
                 [val("Y", DT.FLOAT, shape)])
    expected = _cpu(fp32, {})[0]
    if dtype == DT.FLOAT16:
        expected = expected.astype(np.float16).astype(np.float32)
    else:
        import ml_dtypes
        expected = expected.astype(ml_dtypes.bfloat16).astype(np.float32)
    got = _run_and_assert_mlx(model, {}, disable_opt=True)[0]
    np.testing.assert_allclose(got, expected, atol=2e-5, rtol=0)
    # Verify the rounding is observable, rather than a vacuous terminal-output cast.
    assert np.any(got != _cpu(fp32, {})[0])


@pytest.mark.parametrize("op,feeds,error", [
    ("DFT", {"length": np.array(0, np.int64)}, "length/rate"),
    ("DFT", {"length": np.array(2**32 + 7, np.int64)}, "length/rate"),
    ("DFT", {"length": np.array(-1, np.int64)}, "length/rate"),
    ("STFT", {"step": np.array(0, np.int64), "length": np.array(4, np.int64)}, "length/rate"),
    ("STFT", {"step": np.array(2, np.int64), "length": np.array(99, np.int64)}, "exceed"),
    ("HannWindow", {"size": np.array(-1, np.int64)}, "length/rate"),
])
def test_live_invalid_scalar_is_error(op, feeds, error, execution_mode):
    if op == "DFT":
        inputs = [val("X", DT.FLOAT, [1, 8, 1]), val("length", DT.INT64, [])]
        outputs = [val("Y", DT.FLOAT, [1, None, 2])]
        feeds = {"X": np.ones((1, 8, 1), np.float32), **feeds}
    elif op == "STFT":
        inputs = [val("X", DT.FLOAT, [1, 8, 1]), val("step", DT.INT64, []),
                  None, val("length", DT.INT64, [])]
        outputs = [val("Y", DT.FLOAT, [1, None, None, 2])]
        feeds = {"X": np.ones((1, 8, 1), np.float32), **feeds}
    else:
        inputs = [val("size", DT.INT64, [])]
        outputs = [val("Y", DT.FLOAT, [None])]
    session = _live_session(build(op, inputs, outputs))
    with pytest.raises(Exception, match=error):
        session.run(None, feeds)
    assert "MLXExecutionProvider" in _profile_providers(session)


@pytest.mark.parametrize("op,inverse,onesided,components",
                         [("DFT", 0, 1, 1), ("DFT", 0, 0, 2), ("DFT", 1, 1, 2),
                          ("STFT", 0, 1, 1), ("STFT", 0, 0, 2)])
def test_fft_bfloat16_interior(op, inverse, onesided, components, execution_mode):
    import ml_dtypes

    x = np.random.default_rng(40).normal(size=(2, 15, components)).astype(np.float32)
    source = val("X", DT.FLOAT, list(x.shape))
    reduced = val("reduced", DT.BFLOAT16, list(x.shape))
    shape = [2, None, 1 if inverse and onesided else 2] if op == "DFT" else [2, None, None, 2]
    transformed = val("transformed", DT.BFLOAT16, shape)
    output = val("Y", DT.FLOAT, shape)
    inputs = [reduced]
    initializers = []
    attrs = {"onesided": onesided}
    if op == "DFT":
        attrs.update(axis=1, inverse=inverse)
    else:
        step = initz("step", np.array(3, np.int64))
        frame = initz("frame", np.array(7, np.int64))
        inputs += [step, None, frame]
        initializers = [step, frame]
    nodes = [ir.node("Cast", [source], attributes={"to": int(DT.BFLOAT16)}, outputs=[reduced]),
             ir.node(op, inputs, attributes=attrs, outputs=[transformed]),
             ir.node("Cast", [transformed], attributes={"to": int(DT.FLOAT)}, outputs=[output])]
    graph = ir.Graph([source], [output], nodes=nodes, initializers=initializers,
                     opset_imports={"": 17})
    model = ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()
    rounded = x.astype(ml_dtypes.bfloat16).astype(np.float32)
    if op == "STFT":
        rounded = np.stack([rounded[:, i:i + 7] for i in range(0, 9, 3)], axis=1)
        length, axis = 7, 2
    else:
        length, axis = (28 if inverse and onesided else 15), 1
    expected = _fft_expected(rounded, length, axis, inverse, onesided)
    expected = expected.astype(ml_dtypes.bfloat16).astype(np.float32)
    np.testing.assert_allclose(_run_and_assert_mlx(model, {"X": x})[0], expected, atol=2e-3, rtol=2e-3)


def test_stft_mismatched_live_window_and_length(execution_mode):
    model = build("STFT", [val("X", DT.FLOAT, [1, 16, 1]), initz("step", np.array(2, np.int64)),
                           val("window", DT.FLOAT, [None]), val("length", DT.INT64, [])],
                  [val("Y", DT.FLOAT, [1, None, None, 2])])
    session = _live_session(model)
    with pytest.raises(Exception, match="window length must equal"):
        session.run(None, {"X": np.ones((1, 16, 1), np.float32),
                           "window": np.ones(6, np.float32), "length": np.array(4, np.int64)})
    assert "MLXExecutionProvider" in _profile_providers(session)


@pytest.mark.parametrize("name,value", [("bins", -1), ("bins", 2**31-1), ("length", 0),
                                       ("rate", 0), ("lower", -1), ("upper", float("nan")),
                                       ("upper", 9000), ("upper", 0)])
def test_mel_invalid_live_parameters(name, value, execution_mode):
    inputs = [val(k, dt, []) for k, dt in [("bins", DT.INT64), ("length", DT.INT64),
                                          ("rate", DT.INT64), ("lower", DT.FLOAT),
                                          ("upper", DT.FLOAT)]]
    model = build("MelWeightMatrix", inputs, [val("Y", DT.FLOAT, [None, None])])
    session = _live_session(model)
    feed = {"bins": np.array(5, np.int64), "length": np.array(32, np.int64),
            "rate": np.array(16000, np.int64), "lower": np.array(0, np.float32),
            "upper": np.array(8000, np.float32)}
    feed[name] = np.array(value, feed[name].dtype)
    with pytest.raises(Exception, match="MLX"):
        session.run(None, feed)
    assert "MLXExecutionProvider" in _profile_providers(session)


@pytest.mark.parametrize("edge_type", [np.float16, np.float64])
def test_mel_edge_types(edge_type, execution_mode):
    inputs = [initz("bins", np.array(5, np.int32)), initz("length", np.array(32, np.int32)),
              initz("rate", np.array(8000, np.int32)),
              initz("lower", np.array(0, edge_type)), initz("upper", np.array(4000, edge_type))]
    model = build("MelWeightMatrix", inputs, [val("Y", DT.FLOAT, [17, 5])])
    expected = _cpu(build("MelWeightMatrix", [initz(i.name, i.const_value.numpy()) for i in inputs[:3]] +
                          [initz("lower32", np.array(0, np.float32)),
                           initz("upper32", np.array(4000, np.float32))],
                          [val("Y", DT.FLOAT, [17, 5])]), {})[0]
    np.testing.assert_allclose(_run_and_assert_mlx(model, {}, disable_opt=True)[0], expected, atol=1e-6)


@pytest.mark.parametrize("dtype", [DT.INT8, DT.UINT8, DT.INT16, DT.UINT16, DT.INT32,
                                  DT.UINT32, DT.INT64, DT.UINT64, DT.DOUBLE])
@pytest.mark.parametrize("op", ["HannWindow", "HammingWindow", "BlackmanWindow", "MelWeightMatrix"])
def test_generator_numeric_outputs(op, dtype, execution_mode):
    import onnx
    from onnx.reference import ReferenceEvaluator

    if op == "MelWeightMatrix":
        inputs = [initz("bins", np.array(5, np.int64)), initz("length", np.array(32, np.int64)),
                  initz("rate", np.array(8000, np.int64)), initz("lower", np.array(0, np.float64)),
                  initz("upper", np.array(4000, np.float64))]
        shape = [17, 5]
    else:
        inputs = [initz("size", np.array(20, np.int64))]
        shape = [20]
    model = build(op, inputs, [val("Y", dtype, shape)], attrs={"output_datatype": int(dtype)})
    got = _run_and_assert_mlx(model, {}, disable_opt=True)[0]
    expected = ReferenceEvaluator(onnx.load_from_string(model)).run(None, {})[0]
    assert got.dtype == expected.dtype
    np.testing.assert_allclose(got, expected, rtol=2e-13, atol=2e-15)


@pytest.mark.parametrize("op,inputs,shape", [
    ("DFT", [val("X", DT.FLOAT, [1, 8, 1]), initz("length", np.array([4], np.int64))],
     [1, None, 2]),
    ("HannWindow", [initz("size", np.array([4], np.int64))], [None]),
])
def test_non_scalar_control_is_rejected(op, inputs, shape):
    model = build(op, inputs, [val("Y", DT.FLOAT, shape)])
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail, match="must be a scalar"):
        ort.InferenceSession(model, providers=m.EP_PROVIDERS)


@pytest.mark.parametrize("op", ["HannWindow", "HammingWindow", "BlackmanWindow"])
@pytest.mark.parametrize("periodic", [0, 1])
@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16, DT.DOUBLE])
def test_singleton_window(op, periodic, dtype, execution_mode):
    import onnx
    from onnx.reference import ReferenceEvaluator

    model = build(op, [initz("size", np.array(1, np.int64))], [val("Y", dtype, [1])],
                  attrs={"periodic": periodic, "output_datatype": int(dtype)})
    with np.errstate(invalid="ignore"):
        expected = ReferenceEvaluator(onnx.load_from_string(model)).run(None, {})[0]
    got = _run_and_assert_mlx(model, {}, disable_opt=True)[0]
    np.testing.assert_allclose(got, expected, atol=2e-7, equal_nan=True)


@pytest.mark.parametrize("onesided", [0, 1])
def test_stft_both_optional_inputs_absent(onesided, execution_mode):
    x = np.random.default_rng(41).normal(size=(2, 11, 1)).astype(np.float32)
    model = build("STFT", [val("X", DT.FLOAT, [2, None, 1]), val("step", DT.INT64, [])],
                  [val("Y", DT.FLOAT, [2, 1, None, 2])], attrs={"onesided": onesided})
    expected = _fft_expected(x[:, None], 11, 2, False, onesided)
    got = _run_and_assert_mlx(model, {"X": x, "step": np.array(2, np.int64)})[0]
    np.testing.assert_allclose(got, expected, atol=2e-5)


@pytest.mark.parametrize("op", ["HannWindow", "HammingWindow", "BlackmanWindow"])
@pytest.mark.parametrize("size", [20000, 20001])
def test_integer_window_truncation_before_float_rounding(op, size, execution_mode):
    model = build(op, [initz("size", np.array(size, np.int64))], [val("Y", DT.INT64, [size])],
                  attrs={"output_datatype": int(DT.INT64)})
    expected = np.zeros(size, np.int64)
    if size % 2 == 0:
        expected[size // 2] = 1
    np.testing.assert_array_equal(_run_and_assert_mlx(model, {}, disable_opt=True)[0], expected)


@pytest.mark.parametrize("op", ["DFT", "STFT"])
def test_fft_dynamic_component_dimension(op, execution_mode):
    inputs = [val("X", DT.FLOAT, [None, None, None])]
    attrs = {"onesided": 0}
    shape = [None, None, 2]
    if op == "STFT":
        inputs += [initz("step", np.array(3, np.int64)), None, initz("frame", np.array(7, np.int64))]
        shape = [None, None, None, 2]
    model = build(op, inputs, [val("Y", DT.FLOAT, shape)], attrs=attrs, opset=20)
    session = _live_session(model)
    for batch, length, components in [(2, 13, 1), (1, 16, 2), (3, 14, 1)]:
        x = np.random.default_rng(components).normal(size=(batch, length, components)).astype(np.float32)
        if op == "STFT":
            frames = np.stack([x[:, i:i + 7] for i in range(0, length - 6, 3)], axis=1)
            expected = _fft_expected(frames, 7, 2, False, False)
        else:
            expected = _fft_expected(x, length, 1, False, False)
        np.testing.assert_allclose(session.run(None, {"X": x})[0], expected, atol=2e-5)
    assert "MLXExecutionProvider" in _profile_providers(session)
