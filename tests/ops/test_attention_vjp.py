"""Tiled attention VJP contract, checked against independent NumPy reverse mode."""

import struct

import _models as m
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest


def _model(
    rank, d, hq, hk, causal, kind, selected=(0, 1, 2), mask_rank=4, recompute=False
):
    shape = lambda h, s: ["batch", h, s, d] if rank == 4 else ["batch", s, h * d]
    q, k, v, y, g = [
        m.tensor(n, ir.DataType.FLOAT, shape(h, s))
        for n, h, s in [
            ("q", hq, "sq"),
            ("k", hk, "sk"),
            ("v", hk, "sk"),
            ("y", hq, "sq"),
            ("g", hq, "sq"),
        ]
    ]
    mask = m.tensor(
        "mask",
        ir.DataType.BOOL if kind != 2 else ir.DataType.FLOAT,
        ["batch", 1, "sq", "width"][-mask_rank:],
    )
    # Function body uses ordinary ONNX. Native execution is selected by its parameterized name.
    t = ir.tape.Tape()
    const = lambda a: t.op(
        "Constant", [], {"value": ir.tensor(np.asarray(a, np.int64))}
    )

    def to4(x, heads):
        return (
            x
            if rank == 4
            else t.op(
                "Transpose",
                [t.op("Reshape", [x, const([0, 0, heads, d])])],
                {"perm": [0, 2, 1, 3]},
            )
        )

    aq, ak, av, ay, ag = [
        to4(x, h) for x, h in zip([q, k, v, y, g], [hq, hk, hk, hq, hq])
    ]
    indices = const(np.repeat(np.arange(hk), hq // hk))
    ek = t.op("Gather", [ak, indices], {"axis": 1})
    ev = t.op("Gather", [av, indices], {"axis": 1})
    scale = float(np.float32(d**-0.5))
    sc = t.op("Constant", [], {"value": ir.tensor(np.array(scale, np.float32))})
    scores = t.op(
        "Mul",
        [t.op("MatMul", [aq, t.op("Transpose", [ek], {"perm": [0, 1, 3, 2]})]), sc],
    )
    floating = lambda x: t.op(
        "Constant", [], {"value": ir.tensor(np.array(x, np.float32))}
    )
    dim = lambda x, axis: t.op(
        "Gather", [t.op("Shape", [x]), const([axis])], {"axis": 0}
    )
    if kind:
        width = dim(mask, -1)
        padding = t.op(
            "Where",
            [
                t.op("Equal", [width, const([1])]),
                const([0]),
                t.op("Sub", [dim(ak, 2), width]),
            ],
        )
        pads = t.op("Concat", [const([0] * (2 * mask_rank - 1)), padding], {"axis": 0})
        fill = (
            t.op("Constant", [], {"value": ir.tensor(np.array(False))})
            if kind == 1
            else floating(-np.inf)
        )
        padded = t.op("Pad", [mask, pads, fill])
        bias = (
            t.op("Where", [padded, floating(0), floating(-np.inf)])
            if kind == 1
            else padded
        )
        scores = t.op("Add", [scores, bias])
    if causal:
        scalar = lambda value: t.op("Squeeze", [value, const([0])])
        qi = t.op("Range", [scalar(const([0])), scalar(dim(aq, 2)), scalar(const([1]))])
        ki = t.op("Range", [scalar(const([0])), scalar(dim(ak, 2)), scalar(const([1]))])
        allowed = t.op("LessOrEqual", [ki, t.op("Unsqueeze", [qi, const([1])])])
        scores = t.op(
            "Add", [scores, t.op("Where", [allowed, floating(0), floating(-np.inf)])]
        )
    maximum = t.op("ReduceMax", [scores, const([-1])], {"keepdims": 1})
    empty = t.op("Equal", [maximum, floating(-np.inf)])
    p = t.op(
        "Where",
        [
            empty,
            floating(0),
            t.op(
                "Softmax", [t.op("Where", [empty, floating(0), scores])], {"axis": -1}
            ),
        ],
    )
    dp = t.op("MatMul", [ag, t.op("Transpose", [ev], {"perm": [0, 1, 3, 2]})])
    delta = t.op("ReduceSum", [t.op("Mul", [ay, ag]), const([-1])], {"keepdims": 1})
    ds = t.op("Mul", [t.op("Mul", [p, t.op("Sub", [dp, delta])]), sc])
    dq = t.op("MatMul", [ds, ek])
    dk = t.op("MatMul", [t.op("Transpose", [ds], {"perm": [0, 1, 3, 2]}), aq])
    dv = t.op("MatMul", [t.op("Transpose", [p], {"perm": [0, 1, 3, 2]}), ag])
    reduced = []
    for value in [dk, dv]:
        dims = t.op("Shape", [value])
        batch = t.op("Gather", [dims, const([0])], {"axis": 0})
        tail = t.op("Gather", [dims, const([2, 3])], {"axis": 0})
        grouped = t.op("Concat", [batch, const([hk, hq // hk]), tail], {"axis": 0})
        reduced.append(
            t.op(
                "ReduceSum",
                [t.op("Reshape", [value, grouped]), const([2])],
                {"keepdims": 0},
            )
        )
    grads = [dq, *reduced]
    if rank == 3:
        grads = [
            t.op(
                "Reshape",
                [t.op("Transpose", [a], {"perm": [0, 2, 1, 3]}), t.op("Shape", [x])],
            )
            for a, x in zip(grads, [q, k, v])
        ]
    bits = struct.unpack("I", struct.pack("f", scale))[0]
    name = f"AttentionVJP_r{rank}_q{hq}_k{hk}_d{d}_s{bits:08x}_c{int(causal)}_o{''.join(map(str, selected))}_m{kind}_0000000000000000"
    outputs = [
        m.tensor(
            f"d{i}",
            ir.DataType.FLOAT,
            shape(hq if i == 0 else hk, "sq" if i == 0 else "sk"),
        )
        for i in selected
    ]
    call_inputs = [q, k, v, y, g, mask]
    # Clone formals so the body and call graph own independent values.
    body = ir.GraphView(
        [q, k, v, y, g, mask],
        [grads[i] for i in selected],
        nodes=t.nodes,
        opset_imports={"": 25},
    ).clone()
    if recompute:
        bq, bk, bv, by, _, bm = body.inputs
        forward_y = ir.Value(name="recomputed_y", type=by.type, shape=by.shape)
        attrs = {"scale": scale, "is_causal": int(causal)}
        if rank == 3:
            attrs.update(q_num_heads=hq, kv_num_heads=hk)
        body.insert_before(
            body[0],
            ir.node(
                "Attention",
                [bq, bk, bv] + ([bm] if kind else []),
                outputs=[forward_y],
                attributes=attrs,
            ),
        )
        by.replace_all_uses_with(forward_y)
        body.inputs.remove(by)
        call_inputs.remove(y)
        name = name.replace("AttentionVJP_", "AttentionVJPRecompute_")
    node = ir.node(name, call_inputs, domain="pkg.onnax", outputs=outputs)
    fn = ir.Function("pkg.onnax", name, graph=body, attributes=[])
    model = ir.Model(
        ir.Graph(
            call_inputs,
            outputs,
            nodes=[node],
            name="attention_vjp",
            opset_imports={"": 25, "pkg.onnax": 1},
        ),
        ir_version=11,
        functions=[fn],
    )
    return ir.to_proto(model).SerializeToString()


def _reference(q, k, v, g, mask, causal, scale):
    hq, hk = q.shape[1], k.shape[1]
    kq, vq = [np.repeat(a.astype(np.float64), hq // hk, axis=1) for a in (k, v)]
    q, g = q.astype(np.float64), g.astype(np.float64)
    scores = (q @ kq.swapaxes(-1, -2)) * scale
    if mask is not None:
        if mask.shape[-1] != 1 and mask.shape[-1] < k.shape[2]:
            mask = np.pad(
                mask,
                [(0, 0)] * (mask.ndim - 1) + [(0, k.shape[2] - mask.shape[-1])],
                constant_values=False if mask.dtype == bool else -np.inf,
            )
        scores += np.where(mask, 0, -np.inf) if mask.dtype == bool else mask
    if causal:
        scores = np.where(
            np.arange(k.shape[2])[None, :] <= np.arange(q.shape[2])[:, None],
            scores,
            -np.inf,
        )
    mx = scores.max(-1, keepdims=True)
    p = np.exp(scores - np.where(np.isneginf(mx), 0, mx))
    total = p.sum(-1, keepdims=True)
    p /= np.where(total == 0, 1, total)
    y = p @ vq
    dp = g @ vq.swapaxes(-1, -2)
    ds = p * (dp - (p * dp).sum(-1, keepdims=True)) * scale
    dq = ds @ kq
    dk = (
        (ds.swapaxes(-1, -2) @ q)
        .reshape(q.shape[0], hk, hq // hk, k.shape[2], k.shape[3])
        .sum(2)
    )
    dv = (
        (p.swapaxes(-1, -2) @ g)
        .reshape(q.shape[0], hk, hq // hk, k.shape[2], k.shape[3])
        .sum(2)
    )
    return y.astype(np.float32), [a.astype(np.float32) for a in (dq, dk, dv)]


@pytest.mark.parametrize("mask_rank", [1, 2, 3, 4])
@pytest.mark.parametrize("dense_limit", [0, 64 * 1024**2])
@pytest.mark.parametrize(
    "rank,d,hq,hk,causal,kind,selected",
    [
        (4, 16, 4, 2, False, 0, (0, 1, 2)),
        (4, 8, 4, 2, False, 0, (0, 1, 2)),
        (4, 40, 4, 2, False, 0, (0, 1, 2)),
        (3, 80, 4, 1, True, 1, (0, 1, 2)),
        (4, 160, 2, 2, False, 2, (0, 1, 2)),
        (4, 192, 2, 2, False, 2, (0, 1, 2)),
        (4, 200, 2, 2, False, 2, (0, 1, 2)),
        (4, 248, 2, 2, False, 2, (0, 1, 2)),
        (3, 32, 4, 1, True, 1, (0, 1, 2)),
        (4, 64, 2, 2, False, 2, (0, 2)),
        (3, 128, 4, 2, True, 2, (1, 2)),
        (3, 256, 8, 2, True, 1, (0, 1, 2)),
        (4, 256, 4, 2, False, 2, (1, 2)),
    ],
)
def test_tiled_attention_live_inputs(
    rank, d, hq, hk, causal, kind, selected, dense_limit, mask_rank, monkeypatch
):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    monkeypatch.setenv(
        "ONNXRUNTIME_EP_MLX_ATTENTION_VJP_DENSE_LIMIT_BYTES", str(dense_limit)
    )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        _model(rank, d, hq, hk, causal, kind, selected, mask_rank),
        options,
        providers=["MLXExecutionProvider"],
    )
    cpu_options = ort.SessionOptions()
    cpu_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    cpu_options.intra_op_num_threads = cpu_options.inter_op_num_threads = 1
    cpu = ort.InferenceSession(
        _model(rank, d, hq, hk, causal, kind, selected, mask_rank),
        cpu_options,
        providers=["CPUExecutionProvider"],
    )
    rng = np.random.default_rng(45)
    for b, s, t, width in [
        (2, 5, 7, 7),
        (1, 33, 35, 29),
        (1, 32, 32, 1),
        (1, 17, 18, 18),
    ]:
        q = rng.normal(0, 0.2, (b, hq, s, d)).astype(np.float32)
        k, v = [rng.normal(0, 0.2, (b, hk, t, d)).astype(np.float32) for _ in range(2)]
        g = rng.normal(size=q.shape).astype(np.float32)
        allowed = rng.random((b, 1, s, width)) > 0.2
        allowed[0, :, 0] = False
        if mask_rank == 1:
            allowed = allowed[0, 0, -1]
        elif mask_rank == 2:
            allowed = allowed[0, 0]
        elif mask_rank == 3:
            allowed = allowed[0]
        mask = (
            allowed if kind != 2 else np.where(allowed, 0, -np.inf).astype(np.float32)
        )
        y, expected = _reference(
            q, k, v, g, mask if kind else None, causal, float(np.float32(d**-0.5))
        )
        layout = lambda a, b=b: (
            a if rank == 4 else a.transpose(0, 2, 1, 3).reshape(b, a.shape[2], -1)
        )
        feeds = dict(zip(["q", "k", "v", "y", "g"], map(layout, [q, k, v, y, g])))
        feeds["mask"] = mask
        portable = cpu.run(None, feeds)
        for i, (got, index) in enumerate(zip(session.run(None, feeds), selected)):
            np.testing.assert_allclose(
                got, layout(expected[index]), rtol=1e-4, atol=1e-6
            )
            np.testing.assert_allclose(
                portable[i], layout(expected[index]), rtol=1e-4, atol=1e-6
            )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("rank", [3, 4])
@pytest.mark.parametrize("d", [32, 40, 64, 80, 128, 160, 192, 256])
@pytest.mark.parametrize("kind", [0, 1, 2])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("selected", [(0,), (1, 2), (0, 1, 2)])
def test_recomputed_attention_vjp(
    rank, d, selected, compiled, kind, causal, monkeypatch
):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0" if compiled else "1")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        _model(rank, d, 4, 2, causal, kind, selected, recompute=True),
        options,
        providers=["MLXExecutionProvider"],
    )
    session.disable_fallback()
    rng = np.random.default_rng(191)
    for b, sq, sk, width in [
        (2, 5, 7, 3),
        (1, 33, 35, 29),
        (1, 1, 1, 1),
        (2, 17, 9, 1),
        (1, 33, 35, 35),
        (1, 33, 35, 35),
    ]:
        q = rng.normal(0, 0.2, (b, 4, sq, d)).astype(np.float32)
        k, v = [rng.normal(0, 0.2, (b, 2, sk, d)).astype(np.float32) for _ in range(2)]
        g = rng.normal(size=q.shape).astype(np.float32)
        allowed = rng.random((b, 1, sq, width)) > 0.2
        allowed[0, :, 0] = False
        mask = (
            allowed
            if kind != 2
            else np.where(allowed, rng.normal(0, 0.03, allowed.shape), -np.inf).astype(
                np.float32
            )
        )
        _, expected = _reference(
            q, k, v, g, mask if kind else None, causal, float(np.float32(d**-0.5))
        )
        values = [q, k, v, g]
        if rank == 3:
            values = [
                x.transpose(0, 2, 1, 3).reshape(b, x.shape[2], -1) for x in values
            ]
            expected = [
                x.transpose(0, 2, 1, 3).reshape(b, x.shape[2], -1) for x in expected
            ]
        feeds = dict(zip(["q", "k", "v", "g"], values))
        feeds["mask"] = mask
        actual = session.run(None, feeds)
        for a, i in zip(actual, selected):
            np.testing.assert_allclose(a, expected[i], rtol=1e-4, atol=1e-6)
