"""Phase 3 Wave 2 (§6.2) smoke tests for the ported OpenClaw native runner.

The runner is a near-verbatim copy of
``Workspace-Bench/evaluation/src/agents/openclaw.py``. These tests do **not**
exercise the OpenClaw CLI — that's the Wave 3 §6.5 e2e job and would require
``openclaw`` to be installed on the test host. The goal here is purely static:

- the module imports cleanly,
- ``run()`` exposes the new ``extra_plugins`` keyword-only parameter,
- ``_extract_openclaw_trace`` keeps its public signature,
- every other public-ish helper listed in the wave spec is reachable.
"""

from __future__ import annotations

import inspect
import json

import pytest


def test_module_imports() -> None:
    # Importing the module is itself a smoke test — the source pulled in from
    # Workspace-Bench only uses the stdlib, so this should never fail unless
    # the port accidentally introduced a bad import.
    from claw_eval.harnesses import _openclaw_native  # noqa: F401


def test_run_signature_exposes_extra_plugins() -> None:
    from claw_eval.harnesses import _openclaw_native

    sig = inspect.signature(_openclaw_native.run)
    assert "extra_plugins" in sig.parameters, (
        "run() must accept extra_plugins (Wave 2 §6.2)"
    )
    param = sig.parameters["extra_plugins"]
    assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
        "extra_plugins must be keyword-only — run() uses bare * to enforce kwargs"
    )
    assert param.default is None, (
        "extra_plugins must default to None so existing callers don't break"
    )

    # The pre-existing parameters must still be there with the same kind. This
    # guards against accidentally renaming or reordering arguments during the
    # port.
    for name in (
        "prompt",
        "work_dir",
        "sandbox_dir",
        "timeout_s",
        "api_provider",
        "agent_id",
    ):
        assert name in sig.parameters, f"run() lost original kwarg {name!r}"
        assert sig.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY


def test_extract_openclaw_trace_signature() -> None:
    from claw_eval.harnesses import _openclaw_native

    sig = inspect.signature(_openclaw_native._extract_openclaw_trace)
    # Wave 3's trace adapter will call this directly; lock its shape.
    for name in ("session_jsonl_path", "base_url", "model"):
        assert name in sig.parameters, (
            f"_extract_openclaw_trace must keep keyword arg {name!r}"
        )
        assert sig.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY


def test_copy_session_retains_rich_trajectory_sidecar(tmp_path) -> None:
    from claw_eval.harnesses import _openclaw_native

    sessions = tmp_path / "state/agents/main/sessions"
    sessions.mkdir(parents=True)
    session = sessions / "session-1.jsonl"
    sidecar = sessions / "session-1.trajectory.jsonl"
    session.write_text('{"type":"session","id":"session-1"}\n')
    sidecar.write_text('{"type":"context.compiled"}\n')
    output = tmp_path / "output"
    output.mkdir()

    retained = _openclaw_native._copy_session_jsonl(
        state_dir=str(tmp_path / "state"),
        agent_id="main",
        session_id="session-1",
        dst_dir=str(output),
    )

    assert retained == str(output / "session.jsonl")
    assert (output / "session.jsonl").read_bytes() == session.read_bytes()
    assert (output / "openclaw.trajectory.jsonl").read_bytes() == sidecar.read_bytes()


def test_copy_session_allows_legacy_session_without_sidecar(tmp_path) -> None:
    from claw_eval.harnesses import _openclaw_native

    sessions = tmp_path / "state/agents/main/sessions"
    sessions.mkdir(parents=True)
    session = sessions / "session-1.jsonl"
    session.write_text('{"type":"session","id":"session-1"}\n')
    output = tmp_path / "output"
    output.mkdir()

    retained = _openclaw_native._copy_session_jsonl(
        state_dir=str(tmp_path / "state"),
        agent_id="main",
        session_id="session-1",
        dst_dir=str(output),
    )

    assert retained == str(output / "session.jsonl")
    assert not (output / "openclaw.trajectory.jsonl").exists()


def test_extract_places_assistant_before_its_tool_call(tmp_path) -> None:
    from claw_eval.harnesses import _openclaw_native

    session = tmp_path / "session.jsonl"
    session.write_text(
        json.dumps({
            "type": "message", "timestamp": "2026-01-01T00:00:00Z",
            "message": {
                "role": "assistant", "provider": "openai", "model": "m",
                "content": [{"type": "toolCall", "id": "call-1", "name": "Bash",
                             "arguments": {"command": "true"}}],
                "usage": {"input": 1, "output": 1, "totalTokens": 2},
            },
        }) + "\n",
        encoding="utf-8",
    )
    trace = _openclaw_native._extract_openclaw_trace(
        session_jsonl_path=str(session), base_url="http://localhost/v1", model="m"
    )
    assert [(e["type"], e["role"]) for e in trace["executionTrace"]] == [
        ("text", "assistant"), ("tool", "tool")
    ]


def test_public_helpers_present() -> None:
    # These are the helpers Wave 3 (bridge module + trace adapter) is going to
    # reach into. If the port accidentally drops one, the breakage shows up
    # here instead of three weeks later when the bridge gets wired in.
    from claw_eval.harnesses import _openclaw_native

    for name in (
        "_outputs_from_openclaw_result",
        "_merge_proxy_usage_into_trace",
        "_merge_fetch_log_usage_into_trace",
        "_start_openclaw_usage_proxy",
        "_patch_openclaw_models_file",
        "_openclaw_default_agent_id",
        "_capture_openclaw_preflight",
        "_write_fetch_hook",
    ):
        assert hasattr(_openclaw_native, name), (
            f"_openclaw_native missing required helper {name!r}"
        )
        assert callable(getattr(_openclaw_native, name))


# ---------------------------------------------------------------------------
# Model input modalities — the OpenClaw model entry must declare image/video
# support, else ``modelSupportsInput(entry, "image")`` is false and OpenClaw
# silently drops every image a media tool produces (the openclaw-arm vision
# gap). See ``_resolve_model_input_modalities`` / ``_build_openclaw_temp_config``.


def test_resolve_model_input_modalities_default(monkeypatch) -> None:
    from claw_eval.harnesses import _openclaw_native

    monkeypatch.delenv("CLAWEVAL_MODEL_INPUT_MODALITIES", raising=False)
    mods = _openclaw_native._resolve_model_input_modalities()
    assert "image" in mods
    assert "text" in mods


def test_resolve_model_input_modalities_env_override(monkeypatch) -> None:
    from claw_eval.harnesses import _openclaw_native

    monkeypatch.setenv("CLAWEVAL_MODEL_INPUT_MODALITIES", "text, VIDEO,video")
    mods = _openclaw_native._resolve_model_input_modalities()
    assert mods == ["text", "video"]


def test_resolve_model_input_modalities_rejects_env_drift(monkeypatch) -> None:
    from claw_eval.harnesses import _openclaw_native

    monkeypatch.setenv("CLAWEVAL_MODEL_INPUT_MODALITIES", "text,image")
    with pytest.raises(ValueError, match="modality mismatch"):
        _openclaw_native._resolve_model_input_modalities(["text"])

    monkeypatch.setenv("CLAWEVAL_MODEL_INPUT_MODALITIES", "text,bogus")
    with pytest.raises(ValueError, match="unsupported modality"):
        _openclaw_native._resolve_model_input_modalities(["text"])


def test_build_openclaw_temp_config_declares_image_input(tmp_path, monkeypatch) -> None:
    import json

    from claw_eval.harnesses import _openclaw_native

    monkeypatch.delenv("CLAWEVAL_MODEL_INPUT_MODALITIES", raising=False)
    monkeypatch.delenv("OPENCLAW_CONFIG_PATH", raising=False)
    dst = str(tmp_path / "openclaw.json")
    _openclaw_native._build_openclaw_temp_config(
        dst_path=dst,
        provider_id="openai",
        target_base_url="https://api.z.ai/api/coding/paas/v4",
        target_model="glm-5v-turbo",
        target_api_key="sk-test",
        workspace_dir=str(tmp_path),
    )
    cfg = json.loads((tmp_path / "openclaw.json").read_text())
    entry = cfg["models"]["providers"]["openai"]["models"][0]
    assert entry["id"] == "glm-5v-turbo"
    assert "image" in entry.get("input", []), (
        "model entry must declare image input so OpenClaw sends frames to the model"
    )


@pytest.mark.parametrize("provider_api", ["openai-completions", "anthropic-messages"])
def test_build_openclaw_temp_config_materializes_strict_text_input(
    tmp_path, monkeypatch, provider_api
) -> None:
    import json

    from claw_eval.harnesses import _openclaw_native

    monkeypatch.setenv("CLAWEVAL_MODEL_INPUT_MODALITIES", "text")
    monkeypatch.delenv("OPENCLAW_CONFIG_PATH", raising=False)
    dst = str(tmp_path / "openclaw.json")
    _openclaw_native._build_openclaw_temp_config(
        dst_path=dst,
        provider_id="strict-text",
        target_base_url=(
            "http://127.0.0.1:8010"
            if provider_api == "anthropic-messages"
            else "http://127.0.0.1:8010/v1"
        ),
        target_model="Qwen3.5-9B",
        target_api_key="local-placeholder",
        workspace_dir=str(tmp_path),
        provider_api=provider_api,
        thinking=provider_api == "anthropic-messages",
        reasoning=True,
        context_window=262144,
        max_tokens=262144,
    )

    cfg = json.loads((tmp_path / "openclaw.json").read_text())
    entry = cfg["models"]["providers"]["strict-text"]["models"][0]
    assert entry["input"] == ["text"]
    assert not ({"image", "video", "audio", "document"} & set(entry["input"]))


def test_build_openclaw_temp_config_materializes_context_window(
    tmp_path, monkeypatch
) -> None:
    import json

    from claw_eval.harnesses import _openclaw_native

    monkeypatch.delenv("OPENCLAW_CONFIG_PATH", raising=False)
    dst = str(tmp_path / "openclaw.json")
    _openclaw_native._build_openclaw_temp_config(
        dst_path=dst,
        provider_id="openai",
        target_base_url="http://127.0.0.1:8000/v1",
        target_model="sft-4b-dsv4prolh",
        target_api_key="local-placeholder",
        workspace_dir=str(tmp_path),
        context_window=65536,
        max_tokens=65536,
        provider_timeout_sec=1800,
    )

    cfg = json.loads((tmp_path / "openclaw.json").read_text())
    provider = cfg["models"]["providers"]["openai"]
    assert cfg["models"]["mode"] == "replace"
    entry = provider["models"][0]
    assert entry["contextWindow"] == 65536
    assert entry["maxTokens"] == 65536
    assert provider["timeoutSeconds"] == 1800
    assert "reasoning" not in entry
    assert "compat" not in entry


def test_build_openclaw_temp_config_materializes_deepseek_thinking(
    tmp_path, monkeypatch
) -> None:
    import json

    from claw_eval.harnesses import _openclaw_native

    monkeypatch.delenv("OPENCLAW_CONFIG_PATH", raising=False)
    dst = str(tmp_path / "openclaw.json")
    _openclaw_native._build_openclaw_temp_config(
        dst_path=dst,
        provider_id="vllm",
        target_base_url="https://newapi.deepwisdom.ai",
        target_model="deepseek-v4-flash",
        target_api_key="test-placeholder",
        workspace_dir=str(tmp_path),
        thinking=True,
        thinking_format="deepseek",
    )

    cfg = json.loads((tmp_path / "openclaw.json").read_text())
    entry = cfg["models"]["providers"]["vllm"]["models"][0]
    assert entry["reasoning"] is True
    assert entry["compat"] == {"thinkingFormat": "deepseek"}


def test_build_openclaw_temp_config_materializes_explicit_medium_independently(
    tmp_path, monkeypatch
) -> None:
    from claw_eval.harnesses import _openclaw_native

    monkeypatch.delenv("OPENCLAW_CONFIG_PATH", raising=False)
    dst = str(tmp_path / "openclaw.json")
    _openclaw_native._build_openclaw_temp_config(
        dst_path=dst,
        provider_id="openai",
        target_base_url="http://127.0.0.1:8010/v1",
        target_model="Qwen3.5-9B",
        target_api_key="local-placeholder",
        workspace_dir=str(tmp_path),
        thinking=True,
        reasoning=True,
        thinking_format="qwen-chat-template",
        reasoning_effort="medium",
    )

    cfg = json.loads((tmp_path / "openclaw.json").read_text())
    entry = cfg["models"]["providers"]["openai"]["models"][0]
    assert entry["reasoning"] is True
    assert "compat" not in entry
    assert cfg["agents"]["defaults"]["thinkingDefault"] == "medium"


def test_reasoning_effort_attestation_requires_config_cli_and_session(
    tmp_path,
) -> None:
    from claw_eval.harnesses import _openclaw_native

    config_path = tmp_path / "openclaw.json"
    invocation_path = tmp_path / "openclaw_invocation.json"
    session_path = tmp_path / "session.jsonl"
    config_path.write_text(
        json.dumps({"agents": {"defaults": {"thinkingDefault": "medium"}}}),
        encoding="utf-8",
    )
    invocation_path.write_text(
        json.dumps({"cmd": ["openclaw", "agent", "--thinking", "medium"]}),
        encoding="utf-8",
    )
    session_path.write_text(
        json.dumps(
            {"type": "thinking_level_change", "thinkingLevel": "medium"}
        )
        + "\n",
        encoding="utf-8",
    )

    evidence, error = _openclaw_native._attest_reasoning_effort(
        expected="medium",
        config_path=str(config_path),
        invocation_path=str(invocation_path),
        session_jsonl_path=str(session_path),
        terminal_status="ok",
    )

    assert error is None
    assert evidence["materialized"] == "medium"
    assert evidence["transcriptObserved"] == "medium"
    persisted = json.loads(invocation_path.read_text(encoding="utf-8"))
    assert persisted["reasoningEffort"]["attestation"]["status"] == "ok"


def test_reasoning_effort_attestation_fails_on_session_drift(tmp_path) -> None:
    from claw_eval.harnesses import _openclaw_native

    config_path = tmp_path / "openclaw.json"
    invocation_path = tmp_path / "openclaw_invocation.json"
    session_path = tmp_path / "session.jsonl"
    config_path.write_text(
        json.dumps({"agents": {"defaults": {"thinkingDefault": "medium"}}}),
        encoding="utf-8",
    )
    invocation_path.write_text(
        json.dumps({"cmd": ["openclaw", "agent", "--thinking", "medium"]}),
        encoding="utf-8",
    )
    session_path.write_text(
        json.dumps({"type": "thinking_level_change", "thinkingLevel": "low"})
        + "\n",
        encoding="utf-8",
    )

    _, error = _openclaw_native._attest_reasoning_effort(
        expected="medium",
        config_path=str(config_path),
        invocation_path=str(invocation_path),
        session_jsonl_path=str(session_path),
        terminal_status="ok",
    )

    assert error is not None
    assert "session thinking level mismatch" in error


def test_build_openclaw_temp_config_materializes_anthropic_messages(
    tmp_path, monkeypatch
) -> None:
    from claw_eval.harnesses import _openclaw_native

    monkeypatch.delenv("OPENCLAW_CONFIG_PATH", raising=False)
    dst = str(tmp_path / "openclaw.json")
    _openclaw_native._build_openclaw_temp_config(
        dst_path=dst,
        provider_id="anthropic",
        target_base_url="http://127.0.0.1:8010",
        target_model="Qwen3.5-9B",
        target_api_key="local-placeholder",
        workspace_dir=str(tmp_path),
        provider_api="anthropic-messages",
        thinking=True,
        reasoning=True,
        context_window=262144,
        max_tokens=65536,
    )

    cfg = json.loads((tmp_path / "openclaw.json").read_text())
    provider = cfg["models"]["providers"]["anthropic"]
    entry = provider["models"][0]
    assert provider["api"] == "anthropic-messages"
    assert entry["api"] == "anthropic-messages"
    assert entry["reasoning"] is True
    assert entry["contextWindow"] == 262144
    assert entry["maxTokens"] == 65536
    assert "compat" not in entry


def test_build_openclaw_temp_config_rejects_invalid_anthropic_contract(
    tmp_path, monkeypatch
) -> None:
    from claw_eval.harnesses import _openclaw_native

    monkeypatch.delenv("OPENCLAW_CONFIG_PATH", raising=False)
    kwargs = {
        "dst_path": str(tmp_path / "openclaw.json"),
        "provider_id": "anthropic",
        "target_base_url": "http://127.0.0.1:8010",
        "target_model": "Qwen3.5-9B",
        "target_api_key": "local-placeholder",
        "workspace_dir": str(tmp_path),
    }

    with pytest.raises(ValueError, match="unsupported OpenClaw provider_api"):
        _openclaw_native._build_openclaw_temp_config(
            **kwargs, provider_api="anthropic-responses"
        )
    with pytest.raises(ValueError, match="must not set OpenAI thinking_format"):
        _openclaw_native._build_openclaw_temp_config(
            **kwargs,
            provider_api="anthropic-messages",
            thinking=True,
            thinking_format="qwen-chat-template",
        )
