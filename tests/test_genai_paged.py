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


@pytest.mark.parametrize("mtp,draft_limit", [(False, None), (True, None), (True, 2)])
def test_report_serializes_numpy_tokenizer_ids(monkeypatch, tmp_path, mtp, draft_limit):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0")
    (tmp_path / "genai_config.json").write_text(
        json.dumps({"model": {"mtp": {"enabled": mtp}}})
    )
    library = tmp_path / "plugin.dylib"
    library.write_bytes(b"test")
    output = tmp_path / "report.json"
    overlays = []
    config = SimpleNamespace(
        clear_providers=lambda: None,
        append_provider=lambda _: None,
        overlay=lambda value: overlays.append(json.loads(value)),
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

    def run_once(og, model, tokens, generated, *, require_mtp=False):
        assert all(type(token) is int for token in tokens)
        assert require_mtp == mtp
        return {
            "tokens": [5, 6],
            "ttft_seconds": 0.1,
            "decode_seconds": 0.01,
            "decode_intervals_seconds": [0.01],
            "speculative_stats": {
                "mtp_failures": 0,
                "draft_forward_passes": int(mtp),
                "draft_tokens_evaluated": int(mtp),
                "completed_rounds": int(mtp),
                "target_verify_forward_passes": int(mtp),
            },
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
            *(["--mtp"] if mtp else []),
            *(["--max-draft-tokens", str(draft_limit)] if draft_limit else []),
        ],
    )
    genai_paged.main()
    report = json.loads(output.read_text())
    assert report["prompt_tokens"] == [3, 4]
    assert report["mlx_options"]["ONNXRUNTIME_EP_MLX_NO_COMPILE"] == "0"
    assert report["mode"] == ("mtp" if mtp else "non_speculative")
    expected_overlay = (
        {"speculative": {"max_draft_tokens": draft_limit}} if draft_limit else {}
    )
    assert report["runtime_overlay"] == expected_overlay
    assert overlays == ([expected_overlay] if draft_limit else [])
    assert report["runs"][0]["speculative_stats"]["draft_forward_passes"] == int(mtp)


@pytest.mark.parametrize(
    "failures,draft,verify,rounds,message",
    [
        (1, 1, 1, 1, "failures"),
        (0, 0, 1, 1, "did not complete"),
        (0, 1, 0, 1, "did not complete"),
        (0, 1, 1, 0, "did not complete"),
    ],
)
def test_mtp_rejects_failed_or_inactive_speculation(
    failures, draft, verify, rounds, message
):
    with pytest.raises(RuntimeError, match=message):
        genai_paged.validate_speculation(
            {
                "speculative_stats": {
                    "mtp_failures": failures,
                    "draft_forward_passes": draft,
                    "target_verify_forward_passes": verify,
                    "draft_tokens_evaluated": verify,
                    "completed_rounds": rounds,
                }
            }
        )


def test_mtp_accepts_completed_rounds_without_granular_verify_counter():
    genai_paged.validate_speculation(
        {
            "speculative_stats": {
                "mtp_failures": 0,
                "draft_forward_passes": 18,
                "target_verify_forward_passes": 0,
                "draft_tokens_evaluated": 14,
                "completed_rounds": 5,
            }
        }
    )


@pytest.mark.parametrize("enabled,opt_in", [(True, False), (False, True)])
def test_mtp_requires_matching_export_and_opt_in(
    monkeypatch, tmp_path, capsys, enabled, opt_in
):
    (tmp_path / "genai_config.json").write_text(
        json.dumps({"model": {"mtp": {"enabled": enabled}}})
    )
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", SimpleNamespace())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "genai_paged",
            "--model",
            str(tmp_path),
            "--ep",
            "webgpu",
            "--output",
            str(tmp_path / "report.json"),
            *(["--mtp"] if opt_in else []),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        genai_paged.main()
    assert exc.value.code == 2
    expected = (
        "MTP export requires --mtp"
        if enabled
        else "--mtp requires an export with model.mtp.enabled=true"
    )
    assert expected in capsys.readouterr().err


def test_mtp_rejects_zero_engine_capacity_before_request_creation():
    og = SimpleNamespace(
        Engine=lambda _: SimpleNamespace(max_draft_tokens_per_proposal=lambda: 0)
    )
    with pytest.raises(RuntimeError, match="state_update_capacity"):
        genai_paged.run_once(og, object(), [1], 256, require_mtp=True)
