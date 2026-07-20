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

    monkeypatch.setenv("CLAWEVAL_MODEL_INPUT_MODALITIES", "text, bogus ,VIDEO")
    mods = _openclaw_native._resolve_model_input_modalities()
    assert mods == ["text", "video"]  # bogus dropped, case-normalized, deduped


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
