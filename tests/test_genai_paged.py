"""Timing aggregation excludes the first generated token."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bench import genai_paged
from bench.genai_paged import summarize


def test_summary_uses_aggregate_decode_time():
    runs = [
        {"tokens": [1, 2, 3], "ttft_seconds": 0.1, "decode_seconds": 0.2},
        {"tokens": [1, 2, 3, 4], "ttft_seconds": 0.3, "decode_seconds": 0.6},
    ]
    result = summarize(runs)
    assert result["warm_ttft_median_ms"] == pytest.approx(200)
    assert result["decode_median_ms_per_token"] == pytest.approx(150)
    assert result["aggregate_decode_tokens_per_second"] == pytest.approx(5 / 0.8)


@pytest.mark.parametrize("tokens,seconds", [([1], 0), ([1, 2], 0)])
def test_summary_rejects_missing_decode_measurement(tokens, seconds):
    with pytest.raises(ValueError, match="at least two"):
        summarize([{"tokens": tokens, "ttft_seconds": 0.1, "decode_seconds": seconds}])


def test_report_serializes_numpy_tokenizer_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0")
    (tmp_path / "genai_config.json").write_text("{}")
    library = tmp_path / "plugin.dylib"
    library.write_bytes(b"test")
    output = tmp_path / "report.json"
    config = SimpleNamespace(
        clear_providers=lambda: None, append_provider=lambda _: None
    )
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda **_: "prompt",
        encode=lambda _: np.array([3, 4], dtype=np.int32),
    )
    og = SimpleNamespace(
        register_execution_provider_library=lambda *_: None,
        Config=lambda _: config,
        Model=lambda _: object(),
        Tokenizer=lambda _: tokenizer,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", og)
    monkeypatch.setattr(genai_paged, "version", lambda _: "test")
    monkeypatch.setattr(genai_paged.subprocess, "check_output", lambda *_, **__: "32")

    def run_once(og, model, tokens, generated):
        assert all(type(token) is int for token in tokens)
        return {
            "tokens": [5, 6],
            "ttft_seconds": 0.1,
            "decode_seconds": 0.01,
            "decode_intervals_seconds": [0.01],
        }

    monkeypatch.setattr(genai_paged, "run_once", run_once)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "genai_paged",
            "--model",
            str(tmp_path),
            "--ep",
            "mlx",
            "--library",
            str(library),
            "--output",
            str(output),
            "--tokens",
            "2",
            "--repetitions",
            "1",
        ],
    )
    genai_paged.main()
    report = json.loads(output.read_text())
    assert report["prompt_tokens"] == [3, 4]
    assert report["mlx_options"]["ONNXRUNTIME_EP_MLX_NO_COMPILE"] == "0"
