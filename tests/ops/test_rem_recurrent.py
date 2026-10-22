"""Op-correctness tests for the recurrent family (rust/src/ops/recurrent.rs) vs ORT CPU.

RNN / GRU / LSTM are unrolled over the concrete sequence length into a shape-keyed MLX graph. Each
claimed form is checked against ORT CPU or an explicit equation-based reference where CPU does
not support the form (batch-first layout, reduced precision, or empty sequences).
"""

from __future__ import annotations

import _models as m
import numpy as np
import onnx
import onnx_ir as ir
import pytest
from _models import DataType, assert_matches_cpu, tensor

# Gate multiplier packed into W/R per op.
_GATES = {"RNN": 1, "GRU": 3, "LSTM": 4}


def _reference(model: bytes, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    """ONNX equations, including clipping every activation input, independently of ORT kernels."""
    node = onnx.load_model_from_string(model).graph.node[0]
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    op = node.op_type
    layout = attrs.get("layout", 0)
    x = feeds["X"].astype(np.float64)
    if layout:
        x = x.swapaxes(0, 1)
    seq, batch, _ = x.shape
    directions, _, hidden = feeds["R"].shape
    defaults = {
        "RNN": ["Tanh"],
        "GRU": ["Sigmoid", "Tanh"],
        "LSTM": ["Sigmoid", "Tanh", "Tanh"],
    }[op]
    names = [s.decode() for s in attrs.get("activations", [])] or defaults * directions
    alphas, betas = (
        iter(attrs.get("activation_alpha", [])),
        iter(attrs.get("activation_beta", [])),
    )
    functions = []
    for name in names:
        alpha = (
            next(
                alphas,
                {
                    "LeakyRelu": 0.01,
                    "HardSigmoid": 0.2,
                    "Affine": 0.0,
                    "ScaledTanh": 0.0,
                }.get(name, 1.0),
            )
            if name
            in {
                "Affine",
                "LeakyRelu",
                "ThresholdedRelu",
                "ScaledTanh",
                "HardSigmoid",
                "Elu",
            }
            else 1.0
        )
        beta = (
            next(betas, {"HardSigmoid": 0.5}.get(name, 0.0))
            if name in {"Affine", "ScaledTanh", "HardSigmoid"}
            else 0.0
        )
        functions.append((name, alpha, beta))

    def activate(value: np.ndarray, index: int) -> np.ndarray:
        if "clip" in attrs:
            value = np.clip(value, -attrs["clip"], attrs["clip"])
        name, alpha, beta = functions[index]
        if name == "Relu":
            return np.maximum(value, 0)
        if name == "Tanh":
            return np.tanh(value)
        if name == "Sigmoid":
            return 1 / (1 + np.exp(-value))
        if name == "Affine":
            return alpha * value + beta
        if name == "LeakyRelu":
            return np.where(value >= 0, value, alpha * value)
        if name == "ThresholdedRelu":
            return np.where(value >= alpha, value, 0)
        if name == "ScaledTanh":
            return alpha * np.tanh(beta * value)
        if name == "HardSigmoid":
            return np.clip(alpha * value + beta, 0, 1)
        if name == "Elu":
            return np.where(value >= 0, value, alpha * np.expm1(value))
        if name == "Softsign":
            return value / (1 + np.abs(value))
        if name == "Softplus":
            return np.logaddexp(0, value)
        raise AssertionError(name)

    state_shape = (directions, batch, hidden)
    states = {}
    for name in ["initial_h", "initial_c"]:
        state = feeds.get(name)
        states[name] = (
            np.zeros(state_shape)
            if state is None
            else (state.swapaxes(0, 1) if layout else state).astype(np.float64).copy()
        )
    y = np.zeros((seq, directions, batch, hidden))
    lengths = feeds.get("sequence_lens", np.full(batch, seq))
    for d in range(directions):
        w, r = feeds["W"][d].astype(np.float64), feeds["R"][d].astype(np.float64)
        bias = feeds.get("B", np.zeros((directions, 2 * _GATES[op] * hidden)))[d]
        wb, rb = np.split(bias.astype(np.float64), 2)
        h, c = states["initial_h"][d], states["initial_c"][d]
        pi, po, pf = np.split(feeds.get("P", np.zeros((directions, 3 * hidden)))[d], 3)
        reverse = attrs.get("direction", b"forward") == b"reverse" or d == 1
        f = d * len(defaults)
        for t in reversed(range(seq)) if reverse else range(seq):
            xp, hp = x[t] @ w.T + wb, h @ r.T + rb
            if op == "RNN":
                new_h = activate(xp + hp, f)
                new_c = c
            elif op == "GRU":
                xz, xr, xh = np.split(xp, 3, axis=1)
                hz, hr, hh = np.split(hp, 3, axis=1)
                z, reset = activate(xz + hz, f), activate(xr + hr, f)
                candidate = xh + (
                    reset * hh
                    if attrs.get("linear_before_reset", 0)
                    else (reset * h) @ r[2 * hidden :].T + rb[2 * hidden :]
                )
                new_h = (1 - z) * activate(candidate, f + 1) + z * h
                new_c = c
            else:
                ip, out, fp, cp = np.split(xp + hp, 4, axis=1)
                i = activate(ip + pi * c, f)
                forget = (
                    1 - i if attrs.get("input_forget", 0) else activate(fp + pf * c, f)
                )
                new_c = forget * c + i * activate(cp, f + 1)
                new_h = activate(out + po * new_c, f) * activate(new_c, f + 2)
            valid = (t < lengths)[:, None]
            y[t, d] = np.where(valid, new_h, 0)
            h, c = np.where(valid, new_h, h), np.where(valid, new_c, c)
        if "sequence_lens" in feeds:
            h, c = (
                np.where(lengths[:, None] > 0, h, 0),
                np.where(lengths[:, None] > 0, c, 0),
            )
        states["initial_h"][d], states["initial_c"][d] = h, c
    result = [y, states["initial_h"], states["initial_c"]]
    if layout:
        result = [
            y.transpose(2, 0, 1, 3),
            result[1].swapaxes(0, 1),
            result[2].swapaxes(0, 1),
        ]
    return [
        value.astype(feeds["X"].dtype)
        for name, value in zip(node.output, result)
        if name
    ]


def _assert_reference(
    model: bytes, feeds: dict[str, np.ndarray], *, tolerance: float = 1e-3
) -> None:
    for got, want in zip(
        m.run_mlx(model, feeds), _reference(model, feeds), strict=True
    ):
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance)


def _recurrent_model(
    op: str,
    *,
    seq: int,
    batch: int,
    input_size: int,
    hidden: int,
    direction: str,
    bias: bool,
    initial: bool,
    peephole: bool = False,
    linear_before_reset: int | None = None,
    input_forget: int | None = None,
    clip: float | None = None,
    outputs: tuple[str, ...] = ("Y", "Y_h", "Y_c"),
    dynamic_sequence: bool = False,
    layout: int = 0,
    sequence_lens: list[int] | None = None,
    activations: list[str] | None = None,
    activation_alpha: list[float] | None = None,
    activation_beta: list[float] | None = None,
    omit_hidden_size: bool = False,
    dtype: np.dtype | None = None,
    dynamic_batch: bool = False,
    cell_initial: bool | None = None,
    opset: int = 17,
) -> tuple[bytes, dict[str, np.ndarray]]:
    """Build a single-node RNN/GRU/LSTM model plus random feeds."""
    num_dir = 2 if direction == "bidirectional" else 1
    gates = _GATES[op]
    rng = np.random.default_rng(123)
    dtype = np.dtype("float32") if dtype is None else dtype
    ir_dtype = {
        np.dtype("float16"): DataType.FLOAT16,
        np.dtype("float64"): DataType.DOUBLE,
    }.get(dtype, DataType.FLOAT)
    batch_dim = "batch" if dynamic_batch else batch

    def rand(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.5).astype(dtype)

    x = rand(seq, batch, input_size)
    if layout:
        x = x.transpose(1, 0, 2).copy()
    w = rand(num_dir, gates * hidden, input_size)
    r = rand(num_dir, gates * hidden, hidden)

    in_values: list[ir.Value | None] = [
        tensor(
            "X",
            ir_dtype,
            (
                [batch_dim, "sequence" if dynamic_sequence else seq, input_size]
                if layout
                else ["sequence" if dynamic_sequence else seq, batch_dim, input_size]
            ),
        ),
        tensor("W", ir_dtype, [num_dir, gates * hidden, input_size]),
        tensor("R", ir_dtype, [num_dir, gates * hidden, hidden]),
    ]
    feeds: dict[str, np.ndarray] = {"X": x, "W": w, "R": r}

    def add_optional(index: int, name: str, value: np.ndarray | None) -> None:
        while len(in_values) < index:
            in_values.append(None)
        if value is None:
            in_values.append(None)
            return
        shape = list(value.shape)
        if dynamic_batch:
            if name == "sequence_lens":
                shape[0] = batch_dim
            elif name in {"initial_h", "initial_c"}:
                shape[0 if layout else 1] = batch_dim
        in_values.append(
            tensor(name, DataType.INT32 if name == "sequence_lens" else ir_dtype, shape)
        )
        feeds[name] = value

    add_optional(3, "B", rand(num_dir, 2 * gates * hidden) if bias else None)
    add_optional(
        4,
        "sequence_lens",
        np.array(sequence_lens, dtype=np.int32) if sequence_lens is not None else None,
    )
    state_shape = (batch, num_dir, hidden) if layout else (num_dir, batch, hidden)
    add_optional(5, "initial_h", rand(*state_shape) if initial else None)
    if op == "LSTM":
        add_optional(
            6,
            "initial_c",
            rand(*state_shape)
            if (initial if cell_initial is None else cell_initial)
            else None,
        )
        add_optional(7, "P", rand(num_dir, 3 * hidden) if peephole else None)

    out_values: list[ir.Value] = []
    out_specs = {
        "Y": (
            [batch_dim, "sequence" if dynamic_sequence else seq, num_dir, hidden]
            if layout
            else ["sequence" if dynamic_sequence else seq, num_dir, batch_dim, hidden]
        ),
        "Y_h": [batch_dim, num_dir, hidden] if layout else [num_dir, batch_dim, hidden],
        "Y_c": [batch_dim, num_dir, hidden] if layout else [num_dir, batch_dim, hidden],
    }
    node_outputs: list[ir.Value] = []
    for name in ("Y", "Y_h", "Y_c") if op == "LSTM" else ("Y", "Y_h"):
        if name not in outputs:
            node_outputs.append(ir.Value(name=""))
            continue
        value = tensor(name, ir_dtype, out_specs[name])
        out_values.append(value)
        node_outputs.append(value)

    attrs: list[ir.Attr] = [
        ir.AttrString("direction", direction),
    ]
    if opset >= 14:
        attrs.append(ir.AttrInt64("layout", layout))
    if not omit_hidden_size:
        attrs.append(ir.AttrInt64("hidden_size", hidden))
    if activations is not None:
        attrs.append(ir.AttrStrings("activations", activations))
    for name, values in [
        ("activation_alpha", activation_alpha),
        ("activation_beta", activation_beta),
    ]:
        if values is not None:
            attrs.append(ir.AttrFloat32s(name, values))
    if linear_before_reset is not None:
        attrs.append(ir.AttrInt64("linear_before_reset", linear_before_reset))
    if input_forget is not None:
        attrs.append(ir.AttrInt64("input_forget", input_forget))
    if clip is not None:
        attrs.append(ir.AttrFloat32("clip", clip))

    node = ir.node(
        op,
        in_values,
        attributes={attr.name: attr for attr in attrs},
        outputs=node_outputs,
    )
    graph = ir.Graph(
        [v for v in in_values if v is not None],
        out_values,
        nodes=[node],
        name=f"mlx_{op}",
        opset_imports={"": opset},
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString(), feeds


_DIRECTIONS = ["forward", "reverse"]
_FLAGS = [
    pytest.param(False, False, id="nobias-noinit"),
    pytest.param(True, False, id="bias-noinit"),
    pytest.param(True, True, id="bias-init"),
]


# --- RNN -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("direction", _DIRECTIONS)
@pytest.mark.parametrize("bias,initial", _FLAGS)
def test_rnn(direction: str, bias: bool, initial: bool) -> None:
    model, feeds = _recurrent_model(
        "RNN",
        seq=5,
        batch=3,
        input_size=4,
        hidden=6,
        direction=direction,
        bias=bias,
        initial=initial,
        outputs=("Y", "Y_h"),
    )
    assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


# --- GRU -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("direction", _DIRECTIONS)
@pytest.mark.parametrize("bias,initial", _FLAGS)
@pytest.mark.parametrize("linear_before_reset", [0, 1])
def test_gru(
    direction: str, bias: bool, initial: bool, linear_before_reset: int
) -> None:
    model, feeds = _recurrent_model(
        "GRU",
        seq=5,
        batch=3,
        input_size=4,
        hidden=6,
        direction=direction,
        bias=bias,
        initial=initial,
        linear_before_reset=linear_before_reset,
        outputs=("Y", "Y_h"),
    )
    assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


# --- LSTM ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("direction", _DIRECTIONS)
@pytest.mark.parametrize("bias,initial", _FLAGS)
def test_lstm(direction: str, bias: bool, initial: bool) -> None:
    model, feeds = _recurrent_model(
        "LSTM",
        seq=5,
        batch=3,
        input_size=4,
        hidden=6,
        direction=direction,
        bias=bias,
        initial=initial,
        outputs=("Y", "Y_h", "Y_c"),
    )
    assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("direction", _DIRECTIONS)
def test_lstm_peephole(direction: str) -> None:
    model, feeds = _recurrent_model(
        "LSTM",
        seq=4,
        batch=2,
        input_size=3,
        hidden=5,
        direction=direction,
        bias=True,
        initial=True,
        peephole=True,
        outputs=("Y", "Y_h", "Y_c"),
    )
    assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


def test_lstm_input_forget() -> None:
    model, feeds = _recurrent_model(
        "LSTM",
        seq=4,
        batch=2,
        input_size=3,
        hidden=5,
        direction="forward",
        bias=True,
        initial=False,
        input_forget=1,
        outputs=("Y", "Y_h", "Y_c"),
    )
    assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
def test_clip(op: str) -> None:
    model, feeds = _recurrent_model(
        op,
        seq=4,
        batch=2,
        input_size=3,
        hidden=5,
        direction="forward",
        bias=True,
        initial=False,
        clip=1.5,
        outputs=("Y", "Y_h"),
    )
    assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


# --- Bidirectional (two directions concatenated along the num_directions axis) --------------------
@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
def test_bidirectional(op: str) -> None:
    outs = ("Y", "Y_h", "Y_c") if op == "LSTM" else ("Y", "Y_h")
    model, feeds = _recurrent_model(
        op,
        seq=5,
        batch=3,
        input_size=4,
        hidden=6,
        direction="bidirectional",
        bias=True,
        initial=True,
        outputs=outs,
    )
    assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
@pytest.mark.parametrize("direction", ["forward", "reverse", "bidirectional"])
@pytest.mark.parametrize("initial", [False, True])
def test_dynamic_sequence(op: str, direction: str, initial: bool) -> None:
    model, feeds = _recurrent_model(
        op,
        seq=3,
        batch=2,
        input_size=4,
        hidden=5,
        direction=direction,
        bias=True,
        initial=initial,
        dynamic_sequence=True,
    )
    m.assert_mlx_claims(model, feeds)
    sessions = [
        m._session(model, providers)
        for providers in (m.EP_PROVIDERS, ["CPUExecutionProvider"])
    ]
    rng = np.random.default_rng(123)
    for sequence in [3, 1, 0, 5, 3]:
        feeds["X"] = (rng.standard_normal((sequence, 2, 4)) * 0.5).astype(np.float32)
        if initial:
            for name in ["initial_h", "initial_c"] if op == "LSTM" else ["initial_h"]:
                feeds[name] = (rng.standard_normal(feeds[name].shape) * 0.5).astype(
                    np.float32
                )
        actual = sessions[0].run(None, feeds)
        if sequence == 0:
            # ORT CPU's zero-length recurrent kernels can abort. With no timesteps, ONNX's
            # recurrence leaves the initial states unchanged and Y has an empty sequence axis.
            directions = 2 if direction == "bidirectional" else 1
            state_shape = (directions, 2, 5)
            expected = [
                np.empty((0, directions, 2, 5), dtype=np.float32),
                feeds["initial_h"]
                if initial
                else np.zeros(state_shape, dtype=np.float32),
            ]
            if op == "LSTM":
                expected.append(
                    feeds["initial_c"]
                    if initial
                    else np.zeros(state_shape, dtype=np.float32)
                )
        else:
            expected = sessions[1].run(None, feeds)
        for got, want in zip(actual, expected, strict=True):
            assert got.shape == want.shape
            np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
@pytest.mark.parametrize("clip", [0.0, 0.15])
@pytest.mark.parametrize("direction", ["forward", "reverse", "bidirectional"])
def test_clip_activation_inputs(op: str, clip: float, direction: str) -> None:
    model, feeds = _recurrent_model(
        op,
        seq=3,
        batch=2,
        input_size=2,
        hidden=3,
        direction=direction,
        bias=True,
        initial=True,
        peephole=op == "LSTM",
        clip=clip,
    )
    if op == "LSTM":
        feeds["initial_c"].fill(6.0)
    m.assert_mlx_claims(model, feeds)
    # Check the equations independently: clipping is before activations, not on carried state.
    if op == "LSTM":
        actual = m.run_mlx(model, feeds)
        assert (
            np.max(np.abs(actual[0]))
            <= np.tanh(clip) * (1 / (1 + np.exp(-clip))) + 1e-6
        )
        assert np.max(np.abs(actual[2])) > clip
    _assert_reference(model, feeds)
    # ORT 1.31 rejects zero clip on GRU/LSTM and does not clip LSTM's h(Ct) input.
    if op == "RNN" or (op == "GRU" and clip > 0):
        assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
@pytest.mark.parametrize("direction", ["forward", "reverse", "bidirectional"])
@pytest.mark.parametrize("initial", [False, True])
def test_live_sequence_lens(op: str, direction: str, initial: bool) -> None:
    model, feeds = _recurrent_model(
        op,
        seq=4,
        batch=3,
        input_size=2,
        hidden=3,
        direction=direction,
        bias=True,
        initial=initial,
        peephole=op == "LSTM",
        sequence_lens=[4, 2, 1],
    )
    m.assert_mlx_claims(model, feeds)
    ep = m._session(model, m.EP_PROVIDERS)
    cpu = m._session(model, ["CPUExecutionProvider"])
    for lengths in [[4, 2, 1], [1, 4, 3], [0, 2, 4], [4, 4, 4]]:
        feeds["sequence_lens"][:] = lengths
        feeds["W"] *= 0.9
        if initial:
            feeds["initial_h"] += 0.1
        for got, want in zip(ep.run(None, feeds), cpu.run(None, feeds), strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)


_ACTIVATIONS = [
    "Relu",
    "Tanh",
    "Sigmoid",
    "Affine",
    "LeakyRelu",
    "ThresholdedRelu",
    "ScaledTanh",
    "HardSigmoid",
    "Elu",
    "Softsign",
    "Softplus",
]


@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
@pytest.mark.parametrize("activation", _ACTIVATIONS)
@pytest.mark.parametrize("parameters", [False, True])
def test_activations(op: str, activation: str, parameters: bool) -> None:
    count = {"RNN": 1, "GRU": 2, "LSTM": 3}[op]
    # Distinct activations/parameters in the reverse direction also exercise STRINGS capture order.
    names = [activation] * count + ["HardSigmoid"] * count
    model, feeds = _recurrent_model(
        op,
        seq=3,
        batch=2,
        input_size=2,
        hidden=3,
        direction="bidirectional",
        bias=True,
        initial=True,
        peephole=op == "LSTM",
        activations=names,
        activation_alpha=[0.3 + i * 0.1 for i in range(2 * count)]
        if parameters
        else None,
        activation_beta=[0.4 - i * 0.1 for i in range(2 * count)]
        if parameters
        else None,
    )
    m.assert_mlx_claims(model, feeds)
    _assert_reference(model, feeds)
    # ONNX ThresholdedRelu defaults alpha to 1; ORT's recurrent kernel defaults it to 0.
    if parameters or activation != "ThresholdedRelu":
        assert_matches_cpu(model, feeds, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
@pytest.mark.parametrize("direction", ["forward", "reverse", "bidirectional"])
@pytest.mark.parametrize("initial", [False, True])
def test_batch_first(op: str, direction: str, initial: bool) -> None:
    kwargs = {
        "seq": 4,
        "batch": 3,
        "input_size": 2,
        "hidden": 3,
        "direction": direction,
        "bias": True,
        "initial": initial,
        "peephole": op == "LSTM",
        "sequence_lens": [4, 2, 0],
    }
    model, feeds = _recurrent_model(op, layout=1, **kwargs)
    cpu_model, _ = _recurrent_model(op, **kwargs)
    cpu_feeds = {name: value.copy() for name, value in feeds.items()}
    for name in ["X", "initial_h", "initial_c"]:
        if name in cpu_feeds:
            cpu_feeds[name] = cpu_feeds[name].swapaxes(0, 1).copy()
    m.assert_mlx_claims(model, feeds)
    expected = m.run_cpu(cpu_model, cpu_feeds)
    for i, (got, want) in enumerate(
        zip(m.run_mlx(model, feeds), expected, strict=True)
    ):
        want = want.transpose(2, 0, 1, 3) if i == 0 else want.swapaxes(0, 1)
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
@pytest.mark.parametrize("outputs", [("Y",), ("Y_h",), ("Y", "Y_h")])
def test_optional_outputs_and_inferred_hidden(
    op: str, outputs: tuple[str, ...]
) -> None:
    model, feeds = _recurrent_model(
        op,
        seq=3,
        batch=2,
        input_size=2,
        hidden=3,
        direction="forward",
        bias=False,
        initial=True,
        outputs=outputs,
        omit_hidden_size=True,
    )
    m.assert_mlx_claims(model, feeds)
    _assert_reference(model, feeds)


@pytest.mark.parametrize("outputs", [("Y_c",), ("Y_h", "Y_c"), ("Y", "Y_c")])
def test_lstm_optional_cell_output(outputs: tuple[str, ...]) -> None:
    model, feeds = _recurrent_model(
        "LSTM",
        seq=3,
        batch=2,
        input_size=2,
        hidden=3,
        direction="bidirectional",
        bias=False,
        initial=True,
        peephole=True,
        outputs=outputs,
    )
    m.assert_mlx_claims(model, feeds)
    assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
@pytest.mark.parametrize("layout", [0, 1])
def test_dynamic_batch_and_empty_shapes(op: str, layout: int) -> None:
    kwargs = {
        "input_size": 2,
        "hidden": 3,
        "direction": "bidirectional",
        "bias": True,
        "initial": True,
        "peephole": op == "LSTM",
        "sequence_lens": [3, 1],
        "layout": layout,
        "dynamic_batch": True,
        "dynamic_sequence": True,
    }
    model, feeds = _recurrent_model(op, seq=3, batch=2, **kwargs)
    m.assert_mlx_claims(model, feeds)
    session = m._session(model, m.EP_PROVIDERS)
    for seq, batch in [(3, 2), (1, 1), (3, 0), (0, 2), (4, 3), (3, 2)]:
        local = dict(kwargs, sequence_lens=[seq] * batch)
        _, feeds = _recurrent_model(op, seq=seq, batch=batch, **local)
        expected = _reference(model, feeds)
        for got, want in zip(session.run(None, feeds), expected, strict=True):
            assert got.shape == want.shape
            np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("initial,cell_initial", [(True, False), (False, True)])
@pytest.mark.parametrize("input_forget", [0, 1])
def test_independent_lstm_states(
    initial: bool, cell_initial: bool, input_forget: int
) -> None:
    model, feeds = _recurrent_model(
        "LSTM",
        seq=3,
        batch=2,
        input_size=0,
        hidden=3,
        direction="bidirectional",
        bias=False,
        initial=initial,
        cell_initial=cell_initial,
        peephole=True,
        input_forget=input_forget,
        sequence_lens=[3, 1],
    )
    m.assert_mlx_claims(model, feeds)
    _assert_reference(model, feeds)
    assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
def test_float16(op: str) -> None:
    model, feeds = _recurrent_model(
        op,
        seq=3,
        batch=2,
        input_size=2,
        hidden=3,
        direction="bidirectional",
        bias=True,
        initial=True,
        peephole=op == "LSTM",
        sequence_lens=[1, 3],
        dtype=np.dtype("float16"),
    )
    m.assert_mlx_claims(model, feeds)
    _assert_reference(model, feeds, tolerance=4e-3)


@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
def test_exact_float64_activations(op: str) -> None:
    count = {"RNN": 1, "GRU": 2, "LSTM": 3}[op]
    model, feeds = _recurrent_model(
        op,
        seq=3,
        batch=2,
        input_size=2,
        hidden=3,
        direction="bidirectional",
        bias=True,
        initial=True,
        peephole=op == "LSTM",
        sequence_lens=[1, 3],
        dtype=np.dtype("float64"),
        activations=["Tanh"] * (count * 2),
    )
    m.assert_mlx_claims(model, feeds)
    _assert_reference(model, feeds, tolerance=2e-12)


@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
@pytest.mark.parametrize("opset", [1, 7, 14, 22])
def test_schema_versions(op: str, opset: int) -> None:
    model, feeds = _recurrent_model(
        op,
        seq=3,
        batch=2,
        input_size=2,
        hidden=3,
        direction="reverse",
        bias=True,
        initial=True,
        opset=opset,
    )
    m.assert_mlx_claims(model, feeds)
    _assert_reference(model, feeds)


@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
def test_bfloat16(op: str) -> None:
    gates = _GATES[op]
    shapes = [("X", [3, 2, 2]), ("W", [1, gates * 3, 2]), ("R", [1, gates * 3, 3])]
    model = m.bf16_interior_model(
        op, shapes, [3, 1, 2, 3], attributes={"hidden_size": 3}
    )
    reference_model, feeds = _recurrent_model(
        op,
        seq=3,
        batch=2,
        input_size=2,
        hidden=3,
        direction="forward",
        bias=False,
        initial=False,
        outputs=("Y",),
    )
    m.assert_mlx_claims(model, feeds)
    actual = m.run_mlx(model, feeds)
    for got, want in zip(actual, _reference(reference_model, feeds), strict=True):
        np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "activation,alpha,beta,expected",
    [
        ("ThresholdedRelu", 0.5, 0.0, [-0.0, 0.0, 0.5, 1.0]),
        ("HardSigmoid", 0.5, 0.5, [0.0, 0.5, 0.75, 1.0]),
    ],
)
def test_activation_boundaries(
    activation: str, alpha: float, beta: float, expected: list[float]
) -> None:
    model, feeds = _recurrent_model(
        "RNN",
        seq=1,
        batch=4,
        input_size=1,
        hidden=1,
        direction="forward",
        bias=False,
        initial=False,
        activations=[activation],
        activation_alpha=[alpha],
        activation_beta=[beta],
        outputs=("Y",),
    )
    feeds["X"][:] = np.array([-1, 0, 0.5, 1], dtype=np.float32).reshape(1, 4, 1)
    feeds["W"].fill(1)
    feeds["R"].fill(0)
    m.assert_mlx_claims(model, feeds)
    actual = m.run_mlx(model, feeds)[0]
    np.testing.assert_allclose(actual.ravel(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
def test_recurrent_rejects_inconsistent_hidden_size(op: str) -> None:
    model, feeds = _recurrent_model(
        op,
        seq=3,
        batch=2,
        input_size=2,
        hidden=3,
        direction="forward",
        bias=False,
        initial=False,
    )
    # Claim-time checks must reject invalid known shapes instead of tracing unsafe slices.
    proto = onnx.load_model_from_string(model)
    next(a for a in proto.graph.node[0].attribute if a.name == "hidden_size").i = 0
    with pytest.raises(Exception, match="hidden_size|shape|dimension"):
        m.run_mlx(proto.SerializeToString(), feeds)


@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
@pytest.mark.parametrize("interior", [False, True])
def test_live_lengths_validate_after_cache_hits(op: str, interior: bool) -> None:
    model, feeds = _recurrent_model(
        op,
        seq=3,
        batch=2,
        input_size=2,
        hidden=3,
        direction="bidirectional",
        bias=True,
        initial=True,
        sequence_lens=[3, 1],
        dynamic_sequence=True,
    )
    if interior:
        parsed = ir.from_proto(onnx.load_model_from_string(model))
        recurrent = parsed.graph[0]
        computed = tensor("computed_lengths", DataType.INT32, [2])
        delta = tensor("length_delta", DataType.INT32, [2])
        parsed.graph.inputs.append(delta)
        feeds["length_delta"] = np.zeros(2, dtype=np.int32)
        parsed.graph.insert_before(
            recurrent, ir.node("Add", [recurrent.inputs[4], delta], outputs=[computed])
        )
        recurrent.replace_input_with(4, computed)
        model = ir.to_proto(parsed).SerializeToString()
    m.assert_op_claimed(model, feeds, op, rtol=1e-3, atol=1e-3)
    session = m._session(model, m.EP_PROVIDERS)
    session.disable_fallback()
    for _ in range(3):
        session.run(None, feeds)
    for invalid in [-1, 4]:
        feeds["sequence_lens"][0] = invalid
        with pytest.raises(Exception, match="sequence_lens.*\\[0, sequence_length\\]"):
            session.run(None, feeds)
    feeds["sequence_lens"][0] = 3
    for got, want in zip(
        session.run(None, feeds),
        _reference(model, feeds) if not interior else m.run_cpu(model, feeds),
        strict=True,
    ):
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)
    # Even with an already cached length vector, a shorter X changes the valid upper bound.
    feeds["X"] = feeds["X"][:1].copy()
    with pytest.raises(Exception, match="sequence_lens.*\\[0, sequence_length\\]"):
        session.run(None, feeds)


@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
@pytest.mark.parametrize("parameters", [[], [0.3]])
def test_partial_legacy_activation_parameters(op: str, parameters: list[float]) -> None:
    count = {"RNN": 1, "GRU": 2, "LSTM": 3}[op]
    model, feeds = _recurrent_model(
        op,
        seq=3,
        batch=2,
        input_size=2,
        hidden=3,
        direction="bidirectional",
        bias=True,
        initial=True,
        activations=["Affine", "ScaledTanh"] * count,
        activation_alpha=parameters,
        activation_beta=parameters,
    )
    m.assert_mlx_claims(model, feeds)
    _assert_reference(model, feeds)
    assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)
