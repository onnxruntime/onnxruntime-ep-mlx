"""Timing aggregation excludes the first generated token."""

import hashlib
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


@pytest.mark.parametrize("contents", [None, b" \n", b"\xff"])
def test_invalid_prompt_fails_before_loading_runtime(
    monkeypatch, tmp_path, capsys, contents
):
    prompt = tmp_path / "prompt.txt"
    if contents is not None:
        prompt.write_bytes(contents)
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", None)
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
            str(tmp_path / "result.json"),
            "--prompt-file",
            str(prompt),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        genai_paged.main()
    assert exc.value.code == 2
    assert "provide" in capsys.readouterr().err


@pytest.mark.parametrize(
    "mtp,draft_limit,disabled,provider",
    [
        (False, None, False, "mlx"),
        (True, None, False, "mlx"),
        (True, 2, False, "mlx"),
        (True, None, True, "mlx"),
        (True, None, True, "webgpu"),
        (True, 2, False, "webgpu"),
    ],
)
@pytest.mark.parametrize("prompt_file", [False, True])
@pytest.mark.parametrize("scheduling", [False, True])
def test_report_serializes_numpy_tokenizer_ids(
    monkeypatch, tmp_path, mtp, draft_limit, disabled, provider, prompt_file, scheduling
):
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0")
    (tmp_path / "genai_config.json").write_text(
        json.dumps(
            {
                "model": {"mtp": {"enabled": mtp}},
                "engine": {
                    "dynamic_batching": {"num_blocks": 8, "max_scheduled_tokens": 64}
                },
            }
        )
    )
    library = tmp_path / "plugin.dylib"
    library.write_bytes(b"test")
    output = tmp_path / "report.json"
    overlays = []
    providers = []
    config = SimpleNamespace(
        clear_providers=lambda: None,
        append_provider=providers.append,
        overlay=lambda value: overlays.append(json.loads(value)),
    )
    prompts = []

    def apply_chat_template(**kwargs):
        prompts.append(json.loads(kwargs["messages"])[0]["content"])
        return "prompt"

    tokenizer = SimpleNamespace(
        apply_chat_template=apply_chat_template,
        encode=lambda _: np.array([3, 4], dtype=np.int32),
    )
    og = SimpleNamespace(
        register_execution_provider_library=lambda *_: None,
        Config=lambda _: config,
        Model=lambda _: object(),
        Tokenizer=lambda _: tokenizer,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", og)
    monkeypatch.setitem(
        sys.modules, "onnxruntime", SimpleNamespace(__version__="source")
    )
    monkeypatch.setenv("ORT_LIB_PATH", str(tmp_path / "runtime.dylib"))
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime_ep_webgpu",
        SimpleNamespace(
            get_ep_name=lambda: "WebGpuExecutionProvider",
            get_library_path=lambda: pytest.fail(
                "explicit library must override the wheel"
            ),
        ),
    )
    monkeypatch.setattr(genai_paged, "version", lambda _: "test")
    monkeypatch.setattr(genai_paged.subprocess, "check_output", lambda *_, **__: "32")

    def run_once(og, model, tokens, generated, *, require_mtp=False):
        assert all(type(token) is int for token in tokens)
        assert require_mtp == (mtp and not disabled)
        return {
            "tokens": [5, 6],
            "ttft_seconds": 0.1,
            "decode_seconds": 0.01,
            "decode_intervals_seconds": [0.01],
            "speculative_stats": {
                "mtp_failures": 0,
                "draft_forward_passes": int(require_mtp),
                "draft_tokens_evaluated": int(require_mtp),
                "completed_rounds": int(require_mtp),
                "target_verify_forward_passes": int(require_mtp),
            },
        }

    monkeypatch.setattr(genai_paged, "run_once", run_once)
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("A long context", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "genai_paged",
            "--model",
            str(tmp_path),
            "--ep",
            provider,
            "--library",
            str(library),
            "--source-commit",
            "a" * 40,
            "--output",
            str(output),
            "--tokens",
            "2",
            "--repetitions",
            "1",
            *(["--mtp"] if mtp and not disabled else []),
            *(["--disable-mtp"] if disabled else []),
            *(["--max-draft-tokens", str(draft_limit)] if draft_limit else []),
            *(["--prompt-file", str(prompt_path)] if prompt_file else []),
            *(
                ["--num-blocks", "128", "--max-scheduled-tokens", "256"]
                if scheduling
                else []
            ),
        ],
    )
    genai_paged.main()
    report = json.loads(output.read_text())
    assert report["prompt_tokens"] == [3, 4]
    assert report["versions"]["onnxruntime"] == "source"
    assert report["ort_runtime_library"] == str(tmp_path / "runtime.dylib")
    assert report["mlx_options"]["ONNXRUNTIME_EP_MLX_NO_COMPILE"] == "0"
    assert report["mode"] == ("mtp" if mtp and not disabled else "non_speculative")
    assert report["plugin_source_commit"] == "a" * 40
    assert report["plugin_library"] == str(library)
    expected_provider = "WebGPU" if provider == "webgpu" else "MLXExecutionProvider"
    assert providers == [expected_provider]
    assert report["genai_provider"] == expected_provider
    expected_prompt = (
        "A long context"
        if prompt_file
        else "Explain how a compiler transforms source code into machine code."
    )
    assert prompts == [expected_prompt]
    assert (
        report["prompt_text_sha256"]
        == hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
    )
    expected_overlay = (
        {"speculative": {"max_draft_tokens": draft_limit}} if draft_limit else {}
    )
    if disabled:
        expected_overlay["model"] = {"mtp": {"enabled": False}}
    if scheduling:
        expected_overlay["engine"] = {
            "dynamic_batching": {"num_blocks": 128, "max_scheduled_tokens": 256}
        }
    assert report["runtime_overlay"] == expected_overlay
    assert overlays == ([expected_overlay] if expected_overlay else [])
    assert report["runs"][0]["speculative_stats"]["draft_forward_passes"] == int(
        mtp and not disabled
    )


def test_non_speculative_run_rejects_actual_drafting():
    pending = iter([True, False])
    request = SimpleNamespace(begin_turn=lambda *_: None, close=lambda: None)
    engine = SimpleNamespace(
        create_request=lambda **_: request,
        create_event_buffer=lambda _: object(),
        has_pending_requests=lambda: next(pending),
        run=lambda _: [
            SimpleNamespace(flags=2, request=request, token=5),
            SimpleNamespace(flags=2, request=request, token=6),
        ],
        get_speculative_stats=lambda: {
            "draft_forward_passes": 1,
            "draft_tokens_evaluated": 1,
        },
    )
    og = SimpleNamespace(
        Engine=lambda _: engine,
        RequestOptions=lambda: SimpleNamespace(set_max_session_tokens=lambda _: None),
        TurnOptions=lambda _: SimpleNamespace(
            set_do_sample=lambda _: None,
            set_min_generated_tokens=lambda _: None,
            set_max_generated_tokens=lambda _: None,
        ),
        EngineEventFlags=SimpleNamespace(FAILED=1, TOKEN=2),
    )
    with pytest.raises(RuntimeError, match="Non-speculative run executed MTP drafts"):
        genai_paged.run_once(og, object(), [1], 2)


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
