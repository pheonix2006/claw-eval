"""Wave 3-E (§3.7) — run OpenClaw subprocess inside the sandbox container.

This module is the container counterpart to ``_openclaw_native.run``. The
host-side ``_openclaw_native`` does enormous local state management (proxy
server, fetch-hook, models.json patching) that doesn't translate cleanly
into a docker exec — and the design doc forbids editing that file. So this
module instead does the minimum needed to make OpenClaw work inside the
container:

1. Mount host ``case_dir`` into the container at the same absolute path
   (handled by the caller via ``SandboxRunner.start_container(volumes=)``).
2. Materialise host-side artefacts (config file, temp dirs) under that
   ``case_dir`` so they're visible inside the container at the same paths.
3. Run ``openclaw agent --local --json --message ...`` via ``docker exec``
   with env vars pointing at the in-container (== host) paths.
4. After the subprocess returns, locate ``session.jsonl`` under the
   container's OpenClaw state dir and feed it to the existing
   ``_extract_openclaw_trace`` helper.

What we deliberately DROP vs. host mode:

* **usage proxy server**: would require host networking + reachability from
  inside the container; session.jsonl already includes per-message
  ``llm.usage`` blocks (see ``_extract_openclaw_trace`` lines 820-939 in
  ``_openclaw_native.py``) so the loss is bounded to the cases where
  OpenClaw doesn't emit usage there. Wave 3-E acceptance is "usage in
  trace is non-zero, not necessarily byte-equal to host mode".

* **fetch hook .mjs**: ditto — session.jsonl is the source of truth.

These omissions are documented as part of the Wave 3-E known limitations.
They DO NOT affect callID consistency, robustness, completion, or any of
the §6.5 acceptance checks.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._openclaw_native import (
    _build_openclaw_temp_config,
    _attest_reasoning_effort,
    _extract_openclaw_trace,
    _find_latest_session,
    _json_first_value,
    _normalize_reasoning_effort,
    _outputs_from_openclaw_result,
    _safe_json_loads,
)

_log = logging.getLogger(__name__)


def _docker_exec(
    container: Any,
    cmd: List[str],
    *,
    env: Dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: float | None = None,
) -> "subprocess.CompletedProcess[str]":
    """Run ``cmd`` inside ``container`` via ``docker exec``.

    Captures stdout/stderr. Honours ``timeout``. Does NOT raise on non-zero
    exit code — the caller inspects ``returncode`` to decide whether the
    OpenClaw run timed out / errored, matching the host-mode policy in
    ``_openclaw_native.run``.
    """
    container_id = container.id if hasattr(container, "id") else str(container)
    exec_cmd: list[str] = ["docker", "exec"]
    if env:
        for k, v in env.items():
            exec_cmd.extend(["-e", f"{k}={v}"])
    if cwd is not None:
        exec_cmd.extend(["-w", str(cwd)])
    exec_cmd.append(container_id)
    exec_cmd.extend(cmd)
    return subprocess.run(
        exec_cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _build_agent_cmd(
    *,
    prompt: str,
    agent_id: str,
    timeout_s: float,
    session_key: Optional[str] = None,
    thinking: bool = False,
    reasoning_effort: Optional[str] = None,
) -> "list[str]":
    """Build the ``openclaw agent`` argv for a single in-container turn.

    Pure (no I/O) so the flag wiring — notably the multi-turn
    ``--session-key`` — is unit-testable without a Docker container. When
    ``session_key`` is None the flag is omitted, leaving single-turn behaviour
    byte-identical to the pre-multi-turn command.
    """
    cmd: list[str] = [
        "openclaw", "--no-color", "--log-level", "silent",
        "agent", "--local", "--json",
        "--message", prompt,
        "--agent", str(agent_id),
    ]
    if session_key:
        cmd.extend(["--session-key", str(session_key)])
    reasoning_effort = _normalize_reasoning_effort(reasoning_effort)
    if reasoning_effort is not None:
        cmd.extend(["--thinking", reasoning_effort])
    elif thinking:
        cmd.extend(["--thinking", "on"])
    if isinstance(timeout_s, (int, float)) and timeout_s > 0:
        cmd.extend(["--timeout", str(int(timeout_s))])
    return cmd


def run_in_container(
    *,
    prompt: str,
    container: Any,
    work_dir_host: str,
    case_dir_host: str,
    timeout_s: float,
    api_provider: Dict[str, Any],
    extra_plugins: Optional[List[str]] = None,
    agent_id: Optional[str] = None,
    seeded_config_path: Optional[str] = None,
    session_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Drive an OpenClaw subprocess inside ``container``.

    The shape of the return dict matches ``_openclaw_native.run`` so the
    OpenClawHarness can swallow it through the same trace adapter.

    Path contract: ``work_dir_host`` and ``case_dir_host`` are host paths
    that must also exist inside the container at the SAME absolute path
    (via volume mounts on container start). Files we write under
    ``case_dir_host/raw/`` are immediately readable from the container.

    Parameters
    ----------
    prompt:
        ``task.prompt.text``.
    container:
        Docker SDK container object (from
        ``SandboxRunner.start_container(...).container``).
    work_dir_host:
        Host path to the agent's workdir (also mounted in container).
    case_dir_host:
        Host path to the per-task scratch dir.
    timeout_s:
        Wall-clock cap; OpenClaw's ``--timeout`` flag mirrors this and the
        docker exec timeout is set slightly above it.
    api_provider:
        ``{"baseUrl", "model", "apiKey", "provider_type", "provider_api"}``.
    extra_plugins:
        Plugin ids to surface in ``CLAWEVAL_EXTRA_PLUGINS`` (exposed to the
        OpenClaw subprocess via env var; today informational only since
        bridge install runs separately).
    agent_id:
        Defaults to ``"main"``.
    seeded_config_path:
        Optional pre-built ``openclaw.json`` (the host harness uses this to
        seed ``tools.deny``). When given, we still merge model/agent keys
        into it via ``_build_openclaw_temp_config``.
    session_key:
        Optional OpenClaw session key (``agent:<id>:<key>``). When set, it is
        passed via ``--session-key`` so consecutive calls with the SAME key
        reuse OpenClaw's persisted session context — the mechanism the
        multi-turn user_agent loop relies on to accumulate the conversation
        across independent CLI invocations. ``None`` (default) omits the flag,
        keeping single-turn behaviour byte-identical.
    """
    started_at = time.time()
    raw_dir_host = os.path.join(case_dir_host, "raw")
    os.makedirs(raw_dir_host, exist_ok=True)

    prompt = str(prompt or "").strip()
    if not prompt:
        return {"status": "error", "paths": [], "errorMessage": "Missing prompt"}

    # Host-side filesystem layout (mirrors _openclaw_native.run lines 973-980).
    state_dir = os.path.join(raw_dir_host, "openclaw_state")
    case_home = os.path.join(raw_dir_host, "openclaw_home")
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(case_home, exist_ok=True)
    os.makedirs(os.path.join(case_home, ".openclaw"), exist_ok=True)

    base_url = api_provider.get("baseUrl") if isinstance(api_provider, dict) else None
    model = api_provider.get("model") if isinstance(api_provider, dict) else None
    api_key = api_provider.get("apiKey") if isinstance(api_provider, dict) else None
    thinking = bool(api_provider.get("thinking")) if isinstance(api_provider, dict) else False
    reasoning = bool(api_provider.get("reasoning")) if isinstance(api_provider, dict) else False
    thinking_format = api_provider.get("thinking_format") if isinstance(api_provider, dict) else None
    reasoning_effort = _normalize_reasoning_effort(
        api_provider.get("reasoning_effort") if isinstance(api_provider, dict) else None
    )
    provider_api = (
        str(api_provider.get("provider_api") or "openai-completions")
        if isinstance(api_provider, dict)
        else "openai-completions"
    )
    context_window = api_provider.get("context_window") if isinstance(api_provider, dict) else None
    max_tokens = api_provider.get("max_tokens") if isinstance(api_provider, dict) else None
    input_modalities = api_provider.get("input_modalities") if isinstance(api_provider, dict) else None
    provider_id = (
        str(api_provider.get("provider_type") or "openai")
        if isinstance(api_provider, dict)
        else "openai"
    )
    # thinking on 时归一 provider id 为 "vllm"（openclaw 的 --thinking gate
    # isVllmQwenThinkingCompat 只认 vllm；对齐外层 OpenClawRuntime 做法）。
    if (
        thinking
        and provider_api == "openai-completions"
        and reasoning_effort is None
    ):
        provider_id = "vllm"

    resolved_agent_id = str(agent_id or "").strip() or "main"

    # Build/seed openclaw.json. _build_openclaw_temp_config reads
    # OPENCLAW_CONFIG_PATH (we set it to the seeded path) and merges
    # models/agents keys on top, preserving any tools.deny we wrote there.
    config_path_host = (
        seeded_config_path
        if seeded_config_path
        else os.path.join(raw_dir_host, "openclaw.json")
    )
    if isinstance(base_url, str) and base_url.strip() and isinstance(model, str) and model.strip():
        prev_cfg_env = os.environ.get("OPENCLAW_CONFIG_PATH")
        os.environ["OPENCLAW_CONFIG_PATH"] = config_path_host
        try:
            _build_openclaw_temp_config(
                dst_path=config_path_host,
                provider_id=provider_id,
                target_base_url=base_url.strip(),
                target_model=model.strip(),
                target_api_key=(
                    api_key.strip()
                    if isinstance(api_key, str) and api_key.strip()
                    else None
                ),
                workspace_dir=os.path.abspath(work_dir_host),
                provider_api=provider_api,
                thinking=thinking,
                reasoning=reasoning,
                thinking_format=thinking_format,
                reasoning_effort=reasoning_effort,
                input_modalities=input_modalities,
                context_window=(
                    context_window
                    if isinstance(context_window, int)
                    and not isinstance(context_window, bool)
                    else None
                ),
                max_tokens=(
                    max_tokens
                    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool)
                    else None
                ),
                provider_timeout_sec=(
                    int(timeout_s)
                    if isinstance(timeout_s, (int, float))
                    and not isinstance(timeout_s, bool)
                    and timeout_s > 0
                    else None
                ),
            )
        finally:
            if prev_cfg_env is None:
                os.environ.pop("OPENCLAW_CONFIG_PATH", None)
            else:
                os.environ["OPENCLAW_CONFIG_PATH"] = prev_cfg_env

    # Env vars to pass into docker exec. The OpenClaw subprocess uses these
    # to find the isolated state dir + models config.
    container_env: Dict[str, str] = {
        "OPENCLAW_STATE_DIR": state_dir,
        "OPENCLAW_HOME": case_home,
        "HOME": case_home,
        "OPENCLAW_CONFIG_PATH": config_path_host,
    }
    if provider_api == "anthropic-messages":
        if isinstance(base_url, str) and base_url.strip():
            container_env["ANTHROPIC_BASE_URL"] = base_url.strip()
        if isinstance(model, str) and model.strip():
            container_env["ANTHROPIC_MODEL"] = model.strip()
        if isinstance(api_key, str) and api_key.strip():
            container_env["ANTHROPIC_API_KEY"] = api_key.strip()
    else:
        if isinstance(base_url, str) and base_url.strip():
            container_env["OPENAI_BASE_URL"] = base_url.strip()
        if isinstance(model, str) and model.strip():
            container_env["OPENAI_MODEL"] = model.strip()
        if isinstance(api_key, str) and api_key.strip():
            container_env["OPENAI_API_KEY"] = api_key.strip()
    extra_plugin_ids = [
        str(p).strip() for p in (extra_plugins or []) if str(p or "").strip()
    ]
    if extra_plugin_ids:
        container_env["CLAWEVAL_EXTRA_PLUGINS"] = json.dumps(extra_plugin_ids)

    # ---- Capture an OpenClaw preflight from inside the container. ----
    # Mirrors _openclaw_native._capture_openclaw_preflight, but via docker
    # exec — so the snapshot reflects the container view of config/models.
    preflight_path = os.path.join(raw_dir_host, "openclaw_preflight.json")
    preflight: Dict[str, Any] = {
        "cwd": os.path.abspath(work_dir_host),
        "stateDir": container_env["OPENCLAW_STATE_DIR"],
        "configPath": container_env.get("OPENCLAW_CONFIG_PATH"),
        "providerApi": provider_api,
        "envModel": container_env.get("ANTHROPIC_MODEL")
        or container_env.get("OPENAI_MODEL"),
        "envBaseUrl": container_env.get("ANTHROPIC_BASE_URL")
        or container_env.get("OPENAI_BASE_URL"),
        "commands": {},
    }
    diag_commands = {
        "config_file": ["openclaw", "config", "file"],
        "default_model": ["openclaw", "config", "get", "agents.defaults.model"],
        "agents_list": ["openclaw", "agents", "list", "--json"],
    }
    for name, cmd in diag_commands.items():
        try:
            p = _docker_exec(
                container, cmd,
                env=container_env,
                cwd=os.path.abspath(work_dir_host),
                timeout=15,
            )
            preflight["commands"][name] = {
                "cmd": cmd,
                "exitCode": int(p.returncode),
                "stdout": (p.stdout or "")[:4000],
                "stderr": (p.stderr or "")[:4000],
            }
        except Exception as exc:
            preflight["commands"][name] = {"cmd": cmd, "error": str(exc)}
    try:
        with open(preflight_path, "w", encoding="utf-8") as f:
            json.dump(preflight, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # ---- The actual OpenClaw subprocess. ----
    oc_cmd = _build_agent_cmd(
        prompt=prompt,
        agent_id=resolved_agent_id,
        timeout_s=timeout_s,
        session_key=session_key,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
    )

    # docker exec needs a slightly bigger timeout than OpenClaw's own
    # ``--timeout`` so we get to read the JSON output before docker kills
    # the exec session. +30s slack mirrors WorkspaceBench's CodexHarness.
    exec_timeout = float(timeout_s) + 30.0 if timeout_s and timeout_s > 0 else None

    exit_code = 1
    stdout_text = ""
    stderr_text = ""
    try:
        p = _docker_exec(
            container, oc_cmd,
            env=container_env,
            cwd=os.path.abspath(work_dir_host),
            timeout=exec_timeout,
        )
        exit_code = int(p.returncode)
        stdout_text = p.stdout or ""
        stderr_text = p.stderr or ""
    except subprocess.TimeoutExpired as e:
        exit_code = 124
        stdout_text = e.stdout.decode("utf-8", errors="ignore") if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or "")
        stderr_text = e.stderr.decode("utf-8", errors="ignore") if isinstance(e.stderr, (bytes, bytearray)) else (e.stderr or "")
    except Exception as e:
        exit_code = 1
        stderr_text = str(e)

    # ---- Persist subprocess outputs (mirrors host-mode behaviour). ----
    try:
        with open(os.path.join(raw_dir_host, "openclaw_invocation.json"), "w", encoding="utf-8") as f:
            json.dump({
                "cmd": oc_cmd,
                "cwd": os.path.abspath(work_dir_host),
                "workspaceDir": os.path.abspath(work_dir_host),
                "homeDir": case_home,
                "stateDir": state_dir,
                "agentId": str(resolved_agent_id),
                "exitCode": exit_code,
                "startedAt": started_at,
                "finishedAt": time.time(),
                "containerId": container.id if hasattr(container, "id") else None,
                "reasoningEffort": {
                    "requested": reasoning_effort,
                    "materialized": reasoning_effort,
                    "transport": "agent_cli_--thinking"
                    if reasoning_effort is not None
                    else "legacy_binary_thinking"
                    if thinking
                    else "framework_default",
                },
            }, f, ensure_ascii=False, indent=2)
        with open(os.path.join(raw_dir_host, "stdout.txt"), "w", encoding="utf-8") as f:
            f.write(stdout_text)
        with open(os.path.join(raw_dir_host, "stderr.txt"), "w", encoding="utf-8") as f:
            f.write(stderr_text)
    except Exception:
        pass

    # ---- Parse JSON output + locate session.jsonl. ----
    parsed = _json_first_value(stdout_text)
    if isinstance(parsed, (dict, list)):
        try:
            with open(os.path.join(raw_dir_host, "result.parsed.json"), "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    session_id: Optional[str] = None
    if isinstance(parsed, dict):
        meta = parsed.get("meta")
        if isinstance(meta, dict):
            am = meta.get("agentMeta")
            if isinstance(am, dict) and isinstance(am.get("sessionId"), str) and am.get("sessionId"):
                session_id = str(am.get("sessionId"))

    session_jsonl: Optional[str] = None
    if session_id:
        candidate = os.path.join(
            state_dir, "agents", str(resolved_agent_id), "sessions", f"{session_id}.jsonl"
        )
        if os.path.exists(candidate) and os.path.isfile(candidate):
            dst = os.path.join(raw_dir_host, "session.jsonl")
            try:
                shutil.copy2(candidate, dst)
                session_jsonl = dst
            except Exception:
                session_jsonl = None
    if not session_jsonl:
        sid2, src2 = _find_latest_session(state_dir=state_dir, agent_id=str(resolved_agent_id))
        if sid2 and not session_id:
            session_id = sid2
        if src2:
            try:
                dst = os.path.join(raw_dir_host, "session.jsonl")
                shutil.copy2(src2, dst)
                session_jsonl = dst
            except Exception:
                session_jsonl = None

    outs = _outputs_from_openclaw_result(
        parsed if isinstance(parsed, dict) else {},
        os.path.abspath(work_dir_host),
    )
    last_text = ""
    if isinstance(parsed, dict):
        tos = parsed.get("textOutputs")
        if isinstance(tos, list) and tos and isinstance(tos[-1], str):
            last_text = tos[-1]
        elif isinstance(parsed.get("reply"), str):
            last_text = str(parsed.get("reply"))
        elif isinstance(parsed.get("text"), str):
            last_text = str(parsed.get("text"))

    status = "ok"
    if exit_code == 124:
        status = "timeout"
    elif exit_code != 0:
        status = "error"

    duration_ms = int((time.time() - started_at) * 1000)
    trace_core: Dict[str, Any] = {
        "executionTrace": [],
        "usageTotal": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "llm": {"provider": None, "baseUrl": base_url, "model": model},
    }
    if session_jsonl and os.path.exists(session_jsonl) and os.path.isfile(session_jsonl):
        try:
            trace_core = _extract_openclaw_trace(
                session_jsonl_path=session_jsonl,
                base_url=str(base_url) if isinstance(base_url, str) else None,
                model=str(model) if isinstance(model, str) else None,
            )
        except Exception as exc:
            _log.warning("openclaw trace extraction failed: %s", exc)
    if isinstance(trace_core.get("lastText"), str) and trace_core.get("lastText"):
        last_text = str(trace_core.get("lastText"))

    _, reasoning_contract_error = _attest_reasoning_effort(
        expected=reasoning_effort,
        config_path=config_path_host,
        invocation_path=os.path.join(raw_dir_host, "openclaw_invocation.json"),
        session_jsonl_path=session_jsonl,
        terminal_status=status,
    )
    if reasoning_contract_error is not None:
        status = "error"

    return {
        "status": status,
        "paths": outs,
        "errorMessage": (
            reasoning_contract_error
            or (
                f"Timeout after {timeout_s}s"
                if status == "timeout"
                else (
                    stderr_text[:2000]
                    if status == "error" and isinstance(stderr_text, str)
                    else None
                )
            )
        ),
        "trace": {
            "runner": "openclaw",
            "agentId": str(resolved_agent_id),
            "sessionId": session_id,
            "sessionJsonlPath": session_jsonl,
            "rawDir": raw_dir_host,
            "lastText": last_text,
            "executionTrace": trace_core.get("executionTrace") if isinstance(trace_core.get("executionTrace"), list) else [],
            "llm": trace_core.get("llm") if isinstance(trace_core.get("llm"), dict) else {},
            "usageTotal": trace_core.get("usageTotal") if isinstance(trace_core.get("usageTotal"), dict) else {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
            },
            # No proxy / fetch hook in container mode — these stay zero by design.
            "proxyLogPath": None,
            "proxyCapturedCalls": 0,
            "fetchLogPath": None,
            "fetchCapturedCalls": 0,
        },
        "metrics": {
            "turns": trace_core.get("turns") if isinstance(trace_core.get("turns"), int) else None,
            "promptTokens": (trace_core.get("usageTotal").get("prompt_tokens") if isinstance(trace_core.get("usageTotal"), dict) else None),
            "completionTokens": (trace_core.get("usageTotal").get("completion_tokens") if isinstance(trace_core.get("usageTotal"), dict) else None),
            "totalTokens": (trace_core.get("usageTotal").get("total_tokens") if isinstance(trace_core.get("usageTotal"), dict) else None),
        },
        "durationMs": duration_ms,
    }
