import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "qwen35_paged_generate.py"
SPEC = importlib.util.spec_from_file_location("qwen35_paged_generate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


@pytest.mark.parametrize(
    ("ep", "registered_name", "appended_name"),
    [
        ("mlx", "MLXExecutionProvider", "MLXExecutionProvider"),
        ("webgpu", "WebGpuExecutionProvider", "webgpu"),
    ],
)
def test_plugin_provider_selection(tmp_path, ep, registered_name, appended_name):
    library = tmp_path / "plugin.dylib"
    library.touch()
    config = Mock()
    og = SimpleNamespace(Config=Mock(return_value=config), register_execution_provider_library=Mock())

    assert benchmark.configure_provider(og, tmp_path, ep, library) is config

    og.register_execution_provider_library.assert_called_once_with(registered_name, str(library))
    config.clear_providers.assert_called_once_with()
    config.append_provider.assert_called_once_with(appended_name)
    assert json.loads(config.overlay.call_args.args[0]) == {
        "engine": {"dynamic_batching": {"prefix_caching": False}},
    }


def test_cpu_does_not_register_or_append_plugin(tmp_path):
    config = Mock()
    og = SimpleNamespace(Config=Mock(return_value=config), register_execution_provider_library=Mock())

    benchmark.configure_provider(og, tmp_path, "cpu", None)

    og.register_execution_provider_library.assert_not_called()
    config.clear_providers.assert_called_once_with()
    config.append_provider.assert_not_called()


def test_webgpu_uses_its_own_environment_variable(tmp_path, monkeypatch):
    library = tmp_path / "webgpu.dylib"
    library.touch()
    monkeypatch.setenv("ONNXRUNTIME_WEBGPU_EP_LIB", str(library))
    monkeypatch.setenv("ONNXRUNTIME_MLX_EP_LIB", "/not/the/webgpu/plugin")
    og = SimpleNamespace(Config=Mock(), register_execution_provider_library=Mock())

    benchmark.configure_provider(og, tmp_path, "webgpu", None)

    og.register_execution_provider_library.assert_called_once_with("WebGpuExecutionProvider", str(library))


def test_missing_plugin_is_an_explicit_error(tmp_path, monkeypatch):
    monkeypatch.delenv("ONNXRUNTIME_WEBGPU_EP_LIB", raising=False)
    with pytest.raises(ValueError, match="ONNXRUNTIME_WEBGPU_EP_LIB"):
        benchmark.configure_provider(Mock(), tmp_path, "webgpu", None)


def test_timing_excludes_first_token_from_decode(monkeypatch):
    flags = SimpleNamespace(FAILED=1, CAPACITY_BLOCKED=2, TOKEN=4, TURN_FINISHED=8)
    request = Mock()
    events = [
        SimpleNamespace(flags=flags.TOKEN, request=request, token=np.int32(10)),
        SimpleNamespace(flags=flags.TOKEN, request=request, token=20),
        SimpleNamespace(flags=flags.TURN_FINISHED, request=request),
    ]
    engine = Mock()
    engine.create_request.return_value = request
    engine.has_pending_requests.side_effect = [True, True, True, False]
    engine.run.side_effect = [[event] for event in events]
    og = SimpleNamespace(RequestOptions=Mock(), TurnOptions=Mock(), EngineEventFlags=flags)
    monkeypatch.setattr(benchmark.time, "perf_counter", Mock(side_effect=[1.0, 1.5, 1.75, 1.8]))

    result = benchmark.generate_once(og, engine, [1, 2, 3], 2)

    assert result["tokens"] == [10, 20]
    assert result["ttft"] == 0.5
    assert result["decode_tokens_per_second"] == 4.0
    assert result["total_seconds"] == pytest.approx(0.8)
    assert json.loads(json.dumps(result))["tokens"] == [10, 20]
    request.close.assert_called_once_with()
