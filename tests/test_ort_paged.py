"""Public-state chunking advances the cache without sampling intermediate logits."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bench import ort_paged
from bench.ort_paged import run_once


@pytest.mark.parametrize("chunk", [None, 2, 16])
@pytest.mark.parametrize("device", [False, True])
def test_prefill_chunks_reuse_state_and_sample_only_last_chunk(
    monkeypatch, chunk, device
):
    values = []

    class Value:
        def __init__(self, shape):
            self.shape = list(shape)
            self.readbacks = 0
            values.append(self)

        def update_inplace(self, _):
            pass

        def numpy(self):
            self.readbacks += 1
            assert self.shape == [1, 3]
            return np.array([[0, 2, 1]], np.float32)

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            OrtValue=SimpleNamespace(
                ortvalue_from_numpy=lambda x: Value(x.shape),
                ortvalue_from_shape_and_type=lambda shape, *_args, **_kwargs: Value(
                    shape
                ),
            )
        ),
    )
    feeds, outputs, inputs, calls = {}, {}, {}, []
    binding = SimpleNamespace(
        bind_ortvalue_input=lambda name, value: inputs.update({name: value}),
        bind_ortvalue_output=lambda name, value: outputs.update({name: value}),
        bind_cpu_input=lambda name, value: feeds.update({name: value.copy()}),
    )
    metadata = [
        "input_ids",
        "position_ids",
        "block_table",
        "cumulative_sequence_lengths",
        "past_sequence_lengths",
        "attention_metadata",
        "logits_indices",
    ]
    session = SimpleNamespace(
        io_binding=lambda: binding,
        get_inputs=lambda: [
            SimpleNamespace(
                name="past.0.recurrent", shape=[1, 1, 1, 1], type="tensor(float)"
            ),
            *(SimpleNamespace(name=n) for n in metadata),
        ],
        get_outputs=lambda: [
            SimpleNamespace(name="logits", shape=["rows", 3], type="tensor(float)"),
            SimpleNamespace(name="present.0.recurrent"),
        ],
        run_with_iobinding=lambda _: calls.append(
            ({k: v.copy() for k, v in feeds.items()}, dict(outputs))
        ),
    )
    result = run_once(
        session,
        [10, 11, 12, 13, 14],
        2,
        info=object() if device else None,
        blocks=8,
        block_size=256,
        prefill_chunk_size=chunk,
    )
    assert result["tokens"] == [1, 1]
    assert len(result["decode_intervals_seconds"]) == 1
    state = inputs["past.0.recurrent"]
    assert all(out["present.0.recurrent"] is state for _, out in calls)
    if chunk == 2:
        assert [f["input_ids"].tolist() for f, _ in calls] == [
            [10, 11],
            [12, 13],
            [14],
            [1],
        ]
        assert [f["past_sequence_lengths"].tolist() for f, _ in calls] == [
            [0],
            [2],
            [4],
            [5],
        ]
        assert [f["logits_indices"].tolist() for f, _ in calls] == [[], [], [0], [0]]
        assert [o["logits"].shape for _, o in calls] == [[0, 3], [0, 3], [1, 3], [1, 3]]
        assert sum(v.readbacks for v in values) == 2
    else:
        assert [f["input_ids"].tolist() for f, _ in calls] == [
            [10, 11, 12, 13, 14],
            [1],
        ]


@pytest.mark.parametrize("prompt,chunk", [([], None), ([1], 0), ([1], -1)])
def test_invalid_prefill_arguments(monkeypatch, prompt, chunk):
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace())
    with pytest.raises(ValueError, match="nonempty prompt.*positive"):
        run_once(
            object(),
            prompt,
            2,
            info=None,
            blocks=8,
            block_size=256,
            prefill_chunk_size=chunk,
        )


def test_report_records_imported_runtime_and_chunk_size(monkeypatch, tmp_path):
    (tmp_path / "genai_config.json").write_text("{}")
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps({"prompt_tokens": [1], "runs": [{"tokens": [2, 3]}]})
    )
    library = tmp_path / "plugin.dylib"
    library.write_bytes(b"test")
    output = tmp_path / "report.json"
    monkeypatch.delenv("ONNXRUNTIME_EP_MLX_DEVICE_TENSORS", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            __version__="isolated-runtime",
            register_execution_provider_library=lambda *_: None,
            SessionOptions=lambda: SimpleNamespace(
                add_session_config_entry=lambda *_: None
            ),
            InferenceSession=lambda *_args, **_kwargs: SimpleNamespace(
                disable_fallback=lambda: None
            ),
        ),
    )
    monkeypatch.setattr(ort_paged, "version", lambda _: "installed-distribution")
    monkeypatch.setattr(ort_paged.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(
        ort_paged.subprocess, "check_output", lambda *_args, **_kwargs: "32"
    )

    def run(*_args, **kwargs):
        assert kwargs["prefill_chunk_size"] == 64
        return {
            "tokens": [2, 3],
            "ttft_seconds": 0.1,
            "decode_seconds": 0.2,
            "decode_intervals_seconds": [0.2],
        }

    monkeypatch.setattr(ort_paged, "run_once", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ort_paged",
            "--model",
            str(tmp_path),
            "--library",
            str(library),
            "--reference",
            str(reference),
            "--output",
            str(output),
            "--memory",
            "cpu",
            "--tokens",
            "2",
            "--repetitions",
            "1",
            "--prefill-chunk-size",
            "64",
        ],
    )
    ort_paged.main()
    report = json.loads(output.read_text())
    assert report["onnxruntime"] == "isolated-runtime"
    assert report["onnxruntime_distribution"] == "installed-distribution"
    assert report["prefill_chunk_size"] == 64
