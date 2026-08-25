"""CLI: claw-eval run | grade | list | build-image | _run-inner."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Ensure localhost traffic (mock services) bypasses any HTTP proxy,
# while external API requests (OpenRouter etc.) still go through proxy.
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")


def _load_dotenv_local() -> None:
    """Load ``.env.local`` (KEY=VALUE per line) into os.environ.

    Zero-dependency loader so local runs can keep secrets (e.g. SERP_DEV_KEY)
    out of shell profiles and out of git. Existing env vars always win, so an
    explicit ``export`` on the command line overrides the file. Searched in
    CWD first, then the project root (where pyproject.toml lives).
    """
    candidates = [
        Path.cwd() / ".env.local",
        Path(__file__).resolve().parent.parent.parent / ".env.local",
    ]
    seen: set[Path] = set()
    for env_file in candidates:
        if env_file in seen or not env_file.is_file():
            continue
        seen.add(env_file)
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv_local()


def _resolve_task_yaml(task_arg: str) -> Path:
    """Resolve --task to a YAML file path.

    Accepts either a directory (tasks/T01zh_email_triage) or a file (tasks/T01zh_email_triage/task.yaml).
    """
    p = Path(task_arg)
    if p.is_dir():
        yaml_path = p / "task.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"No task.yaml found in {p}")
        return yaml_path
    return p


def _resolve_tasks_dir(task_yaml: Path) -> Path:
    """Given a task YAML path like tasks/T01zh_email_triage/task.yaml, return the tasks/ root dir.

    The path is ``.resolve()``-d first so that a *symlinked* task dir resolves to
    its real location. Without this, a symlink ``tmp/T002 -> repo/tasks/T002``
    would make ``task_yaml.parent.parent`` = ``tmp`` (the symlink's parent), and
    callers that use ``tasks_dir.parent`` as the mock-service CWD would land in
    ``/tmp`` instead of the repo root — breaking the relative
    ``python mock_services/.../server.py`` service commands. Resolving makes
    symlinked task dirs behave identically to real ones; for a non-symlink path
    ``.resolve()`` only absolutises it, leaving ``parent.parent`` unchanged.
    """
    # task.yaml is at tasks/<ID>/task.yaml — parent.parent is tasks/
    return task_yaml.resolve().parent.parent


def _make_trace_dir(base_dir: str | Path, model_id: str) -> Path:
    """Build a trace output directory: ``<base_dir>/<YYYYMMDD_HHMMSS>_<model>/``.

    Model names like ``anthropic/claude-opus-4-6`` are sanitised to
    ``anthropic_claude-opus-4-6`` (slashes replaced with underscores).
    """
    from datetime import datetime

    date_str = datetime.now().strftime("%y-%m-%d-%H-%M")
    safe_model = model_id.replace("/", "_")
    trace_dir = Path(base_dir) / f"{safe_model}_{date_str}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


def _make_judge(cfg, args):
    """Create an LLMJudge instance if enabled, or None."""
    if getattr(args, "no_judge", False):
        return None
    if not cfg.judge.enabled:
        return None
    # Need at least an API key to use the judge
    api_key = cfg.judge.api_key
    if not api_key:
        return None
    from .graders.llm_judge import LLMJudge

    model_id = getattr(args, "judge_model", None) or cfg.judge.model_id
    return LLMJudge(
        model_id=model_id,
        api_key=api_key,
        base_url=cfg.judge.base_url,
    )


def _apply_proxy(proxy_url: str | None) -> None:
    """Set HTTP(S)_PROXY env vars for model/judge API traffic.

    Mock services are unaffected because ``services.py`` strips proxy vars
    from subprocess environments, and ``no_proxy`` already covers localhost.
    """
    if not proxy_url:
        return
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url
    print(f"[proxy] Model/judge traffic via {proxy_url}")


def _require_harness_ok(result) -> None:
    """Fail loudly before grading an infrastructure-invalid rollout."""
    status = str(getattr(result, "status", "ok") or "ok")
    if status == "ok":
        return
    detail = getattr(result, "error_message", None) or "no error detail"
    raise RuntimeError(
        f"harness rollout {status}: {detail}; trace={result.trace_path}"
    )


def _require_provider_transport(cfg, args: argparse.Namespace) -> None:
    """Fail before rollout if CLI and config disagree on the model wire."""

    requested = getattr(args, "provider_transport", None)
    if requested is None:
        return
    configured = str(cfg.model.provider_transport)
    if requested != configured:
        raise RuntimeError(
            "provider transport mismatch: "
            f"CLI requested {requested!r}, config materialized {configured!r}"
        )


def _grade_with_optional_params(
    grader, messages, dispatches, task,
    *, audit_data, judge, media_events, env_snapshot=None,
):
    """Call grader.grade, passing optional params only when the grader accepts them.

    Returns (scores, judge_calls) where judge_calls is a list of dicts
    captured from the LLMJudge call log (empty if judge has no logging).

    When ``judge`` is ``None`` (``--no-judge`` or no judge key), it is replaced
    with a :class:`NoJudge` null-object so graders that call ``judge.evaluate()``
    unconditionally contribute a neutral 0.0 sub-score instead of crashing with
    ``AttributeError: 'NoneType' object has no attribute 'evaluate'``.
    """
    from .graders.base import AbstractGrader
    from .graders.llm_judge import NoJudge

    if judge is None:
        judge = NoJudge()

    if hasattr(judge, "reset_call_log"):
        judge.reset_call_log()

    params = inspect.signature(grader.grade).parameters
    kwargs = {"audit_data": audit_data, "judge": judge}
    if "media_events" in params:
        kwargs["media_events"] = media_events
    if "env_snapshot" in params and env_snapshot is not None:
        kwargs["env_snapshot"] = env_snapshot
    scores = grader.grade(messages, dispatches, task, **kwargs)

    judge_calls = judge.get_call_log() if hasattr(judge, "get_call_log") else []
    return scores, judge_calls


def _grade_run_result(
    *,
    args: argparse.Namespace,
    task_yaml: Path,
    task,
    tasks_dir: Path,
    trace_path: Path,
    env_snapshot: dict | None,
    raw_dir: Path | None,
    rollout_status: str,
    model_id: str,
    judge,
):
    """Grade a rollout directly or through its immutable replay bundle.

    ``--freeze-rollout-bundle`` is the Phase-C lifecycle seam used by Harbor:
    the rollout, exact snapshot, authoritative grader inputs, and native
    OpenClaw artifacts are frozen before the grader is loaded. The score is
    then computed from that bundle, proving that grading no longer depends on
    the live sandbox or mutable task directory.
    """

    from .graders.registry import get_grader
    from .trace.reader import load_trace

    if getattr(args, "freeze_rollout_bundle", False):
        if args.harness != "openclaw":
            raise RuntimeError(
                "--freeze-rollout-bundle currently requires --harness openclaw"
            )
        from .rollout_bundle import grade_native_rollout, write_native_rollout_bundle

        bundle_root = trace_path.with_name(f"{trace_path.stem}_rollout_bundle")
        manifest = write_native_rollout_bundle(
            bundle_root=bundle_root,
            task_yaml=task_yaml,
            trace_path=trace_path,
            env_snapshot=env_snapshot or {},
            raw_dir=raw_dir,
            harness=args.harness,
            model=model_id,
            rollout_status=rollout_status,
        )
        grade = grade_native_rollout(manifest, judge=judge)
        print(f"Rollout bundle: {manifest}")
        return grade.scores, grade.judge_calls

    _start, messages, dispatches, media_events, _end, audit_data = load_trace(trace_path)
    grader = get_grader(task.task_id, tasks_dir=tasks_dir, task_dir=task_yaml.parent)
    return _grade_with_optional_params(
        grader, messages, dispatches, task,
        audit_data=audit_data, judge=judge, media_events=media_events,
        env_snapshot=env_snapshot,
    )


def _make_user_agent(cfg, task):
    """Create a UserAgent instance if the task has user_agent enabled, or None."""
    if not task.user_agent.enabled:
        return None
    from .runner.user_agent import UserAgent
    ua_model_cfg = cfg.user_agent_model
    api_key = ua_model_cfg.api_key or cfg.judge.api_key
    if not api_key:
        return None
    return UserAgent(
        model_id=ua_model_cfg.model_id,
        api_key=api_key,
        base_url=ua_model_cfg.base_url,
    )


def _cfg_with_model_overrides(
    cfg,
    *,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
):
    """Return a copy of ``cfg`` with CLI model/api_key/base_url overrides applied.

    The harness layer is given a single ``cfg`` — we baked the overrides into a
    cloned cfg so harnesses don't need to know about argparse. None values fall
    back to the existing cfg.model fields (no overwrite with None).
    """
    overrides: dict = {}
    if model is not None:
        overrides["model_id"] = model
    if api_key is not None:
        overrides["api_key"] = api_key
    if base_url is not None:
        overrides["base_url"] = base_url
    if not overrides:
        return cfg
    new_model = cfg.model.model_copy(update=overrides)
    return cfg.model_copy(update={"model": new_model})


def _run_harness_preflight(harness, task) -> None:
    """Run harness.preflight; print errors to stderr and exit on failure."""
    errs = harness.preflight(task)
    for err in errs:
        print(f"[preflight] {err}", file=sys.stderr)
    if errs:
        raise SystemExit(1)


def _collect_env_snapshot(sandbox_url: str, task) -> dict:
    """Collect environment data from the container after the agent loop finishes.

    Called between agent loop completion and container destruction.
    What to collect is declared in task.yaml via ``env_snapshot_files``
    and ``env_snapshot_commands``.

    Individual collection failures are recorded as ``{"error": ...}``
    entries in the snapshot dict rather than aborting the entire snapshot.
    """
    import httpx

    timeout = getattr(task.environment, "env_snapshot_timeout", 10) if hasattr(task, "environment") else 10
    client = httpx.Client(timeout=max(timeout + 5, 15.0))
    snapshot: dict = {}

    try:
        # Run commands FIRST — they typically generate the files we need to collect
        for cmd in getattr(task, "env_snapshot_commands", []):
            try:
                resp = client.post(
                    f"{sandbox_url}/exec",
                    json={"command": cmd, "timeout_seconds": timeout},
                )
                cmd_result = resp.json()
                snapshot[f"cmd:{cmd}"] = cmd_result
                # Debug: show command results
                exit_code = cmd_result.get("exit_code", "?")
                stdout = (cmd_result.get("stdout") or "")[:200]
                stderr = (cmd_result.get("stderr") or "")[:200]
                print(f"[env_snapshot] cmd exit={exit_code}: {cmd[:80]}")
                if stderr:
                    print(f"[env_snapshot]   stderr: {stderr}")
            except Exception as exc:
                snapshot[f"cmd:{cmd}"] = {"error": str(exc)}
                print(f"[WARNING] env_snapshot command failed: {cmd}: {exc}")

        def _normalize_read_response(data: dict) -> dict:
            """Convert media-format /read responses to the standard encoding/content format.

            The /read endpoint returns a media-format response for image files
            (with "frames" containing image_b64) instead of the simple
            {"encoding": "base64", "content": ...} that _save_env_snapshot and
            graders expect.  Extract the raw image data from the first frame.
            """
            if data.get("frames") and not data.get("encoding"):
                frames = data["frames"]
                if frames and frames[0].get("image_b64"):
                    return {
                        "content": frames[0]["image_b64"],
                        "encoding": "base64",
                        "mime_type": frames[0].get("mime_type", "image/png"),
                    }
            return data

        # Collect files AFTER commands (commands may generate the files)
        for pattern in getattr(task, "env_snapshot_files", []):
            try:
                if "*" in pattern or "?" in pattern:
                    resp = client.post(
                        f"{sandbox_url}/glob",
                        json={"pattern": pattern, "max_files": 50},
                    )
                    file_list = resp.json().get("files", [])
                    print(f"[env_snapshot] glob '{pattern}' → {len(file_list)} file(s)")
                    for f in file_list:
                        try:
                            resp2 = client.post(
                                f"{sandbox_url}/read",
                                json={"path": f["path"]},
                            )
                            snapshot[f"file:{f['path']}"] = _normalize_read_response(resp2.json())
                        except Exception as exc:
                            snapshot[f"file:{f['path']}"] = {"error": str(exc)}
                            print(f"[WARNING] env_snapshot file read failed: {f['path']}: {exc}")
                else:
                    resp = client.post(
                        f"{sandbox_url}/read",
                        json={"path": pattern},
                    )
                    snapshot[f"file:{pattern}"] = _normalize_read_response(resp.json())
            except Exception as exc:
                snapshot[f"file:{pattern}"] = {"error": str(exc)}
                print(f"[WARNING] env_snapshot file failed: {pattern}: {exc}")
    finally:
        client.close()

    return snapshot


def _save_env_snapshot(snapshot: dict, trace_path: Path, task_id: str) -> None:
    """Persist env_snapshot artifacts alongside the trace for debugging.

    Saves:
    - PNG screenshots as individual files in <trace_dir>/<task_id>_snapshot/
    - Command stdout/stderr as snapshot_commands.json
    - A summary index as snapshot_index.json
    """
    import base64

    if not snapshot:
        return

    snapshot_dir = trace_path.parent / f"{trace_path.stem}_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    index: dict = {"files": [], "commands": []}

    for key, entry in sorted(snapshot.items()):
        if key.startswith("file:"):
            path = key[len("file:"):]
            if entry.get("encoding") == "base64" and entry.get("content"):
                # Save binary file (e.g. PNG screenshot)
                filename = Path(path).name
                out_path = snapshot_dir / filename
                try:
                    out_path.write_bytes(base64.b64decode(entry["content"]))
                    index["files"].append({"container_path": path, "saved_as": filename, "size": out_path.stat().st_size})
                except Exception as exc:
                    index["files"].append({"container_path": path, "error": str(exc)})
            elif entry.get("error"):
                index["files"].append({"container_path": path, "error": entry["error"]})
            else:
                # Text file — save as-is
                filename = Path(path).name
                out_path = snapshot_dir / filename
                try:
                    out_path.write_text(entry.get("content", ""), encoding="utf-8")
                    index["files"].append({"container_path": path, "saved_as": filename})
                except Exception as exc:
                    index["files"].append({"container_path": path, "error": str(exc)})

        elif key.startswith("cmd:"):
            cmd = key[len("cmd:"):]
            index["commands"].append({
                "cmd": cmd,
                "exit_code": entry.get("exit_code"),
                "stdout": (entry.get("stdout") or "")[:2000],
                "stderr": (entry.get("stderr") or "")[:2000],
                "error": entry.get("error"),
            })

    # Write index
    index_path = snapshot_dir / "snapshot_index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"[env_snapshot] saved {len(index['files'])} file(s) + {len(index['commands'])} cmd(s) → {snapshot_dir}")


def _trace_totals(end) -> dict[str, int | float]:
    """Extract model token/time totals from a TraceEnd event."""
    if end is None:
        return {
            "model_input_tokens": 0,
            "model_output_tokens": 0,
            "total_tokens": 0,
            "model_time_s": 0.0,
            "tool_time_s": 0.0,
            "other_time_s": 0.0,
            "wall_time_s": 0.0,
        }

    model_input_tokens = getattr(end, "model_input_tokens", getattr(end, "input_tokens", 0))
    model_output_tokens = getattr(end, "model_output_tokens", getattr(end, "output_tokens", 0))
    total_tokens = getattr(end, "total_tokens", model_input_tokens + model_output_tokens)
    model_time_s = getattr(end, "model_time_s", 0.0)
    tool_time_s = getattr(end, "tool_time_s", 0.0)
    other_time_s = getattr(end, "other_time_s", 0.0)
    wall_time_s = getattr(end, "wall_time_s", 0.0)

    # Backward compatibility for older traces.
    if not total_tokens:
        total_tokens = model_input_tokens + model_output_tokens
    if not other_time_s and wall_time_s:
        other_time_s = max(0.0, wall_time_s - model_time_s - tool_time_s)

    return {
        "model_input_tokens": model_input_tokens,
        "model_output_tokens": model_output_tokens,
        "total_tokens": total_tokens,
        "model_time_s": wall_time_s if not model_time_s and not tool_time_s else model_time_s,
        "tool_time_s": tool_time_s,
        "other_time_s": other_time_s,
        "wall_time_s": wall_time_s,
    }


def cmd_run(args: argparse.Namespace) -> None:
    """Run an agent on a task."""
    _apply_proxy(getattr(args, "proxy", None))

    from .config import load_config
    from .graders.registry import get_grader
    from .harnesses import get_harness
    from .models.scoring import compute_pass_at_k, compute_pass_hat_k, compute_task_score, is_pass
    from .models.task import TaskDefinition
    from .trace.reader import load_trace

    cfg = load_config(args.config)
    _require_provider_transport(cfg, args)
    harness = get_harness(args.harness)

    task_yaml = _resolve_task_yaml(args.task)
    task = TaskDefinition.from_yaml(task_yaml)
    tasks_dir = _resolve_tasks_dir(task_yaml)
    _run_harness_preflight(harness, task)

    port_offset = getattr(args, "port_offset", 0) or 0
    if port_offset:
        task.apply_port_offset(port_offset)

    # Resolve model_id early (used for trace dir naming)
    model_id = args.model or cfg.model.model_id
    base_trace_dir = args.trace_dir or cfg.defaults.trace_dir
    trace_dir = _make_trace_dir(base_trace_dir, model_id)

    # Bake CLI model/api_key/base_url overrides into cfg so the harness sees one
    # source of truth.
    cfg = _cfg_with_model_overrides(
        cfg, model=args.model, api_key=args.api_key, base_url=args.base_url,
    )

    # ---- Sandbox mode: container-based evaluation ----
    # Agent loop stays on host; container runs only the sandbox HTTP server.
    # When ``--harness openclaw`` is used, the container *also* runs the
    # OpenClaw subprocess (Wave 3-E §3.7).
    sandbox_mode = getattr(args, "sandbox", False) or cfg.sandbox.enabled

    # Phase 3 Wave 3-E §7: OpenClaw must NOT run in host mode for production
    # evaluation. ``--harness openclaw`` without ``--sandbox`` is refused
    # unless the user opts into host smoke explicitly via the escape hatch
    # env var (used by ``tests/test_openclaw_e2e.py``).
    if args.harness == "openclaw" and not sandbox_mode:
        if os.environ.get("CLAWEVAL_ALLOW_OPENCLAW_HOST_SMOKE") != "1":
            print(
                "ERROR: --harness openclaw requires --sandbox for production.\n"
                "OpenClaw on host can read tasks/<id>/grader.py (no isolation,\n"
                "design doc §3.7 / §7). To run the legacy Wave 3-D host smoke\n"
                "test, set CLAWEVAL_ALLOW_OPENCLAW_HOST_SMOKE=1 (not for\n"
                "production evaluation).",
                file=sys.stderr,
            )
            raise SystemExit(2)

    # Phase 4 Wave 4-D §4.2: AOrchestra is asymmetric vs OpenClaw — it's OK
    # in host mode UNLESS the task declares SANDBOX_TOOL_NAMES. AOrchestra is
    # a Python in-process library; host mode is safe when the agent doesn't
    # ask for Bash/Read/Write/... tools that would otherwise need a sandbox
    # server. Refuse the SANDBOX-tools-without-sandbox combo loudly here so
    # the harness's preflight error has a clearer call site.
    if args.harness == "aorchestra":
        from .runner.sandbox_tools import SANDBOX_TOOL_NAMES as _SBX

        task_needs_sandbox = any(t.name in _SBX for t in task.tools)
        if task_needs_sandbox and not sandbox_mode:
            sbx_names = [t.name for t in task.tools if t.name in _SBX]
            print(
                "ERROR: --harness aorchestra requires --sandbox when the task\n"
                f"declares sandbox tools {sbx_names}. AOrchestra runs as a Python\n"
                "library on host; sandbox tools need the in-container sandbox\n"
                "server (design doc §4.2).",
                file=sys.stderr,
            )
            raise SystemExit(2)

    if sandbox_mode:
        from .runner.sandbox_runner import SandboxRunner
        from .runner.services import ServiceManager

        sandbox_image = getattr(args, "sandbox_image", None) or cfg.sandbox.image
        # OpenClaw harness defaults to its own image (OpenClaw CLI + sandbox
        # server). The user can still override via --sandbox-image.
        if args.harness == "openclaw" and getattr(args, "sandbox_image", None) is None:
            sandbox_image = "claw-eval-agent-openclaw:latest"
        runner = SandboxRunner(cfg.sandbox, image=sandbox_image)
        judge = _make_judge(cfg, args)
        trials = args.trials or 1
        trial_scores: list[float] = []
        trace_paths: list[Path] = []

        with ServiceManager(task.services, mock_today=task.environment.mock_today) as svc:
            for i in range(trials):
                if trials > 1:
                    print(f"\n--- Trial {i + 1}/{trials} ---")
                if i > 0:
                    svc.reset_all()

                run_id = f"{task.task_id}-trial{i}"
                # OpenClaw container needs host networking (so OpenClaw inside
                # can reach host mock services + so host can probe the
                # in-container sandbox server through localhost) plus a volume
                # mount of the per-task case_dir at the same path on both
                # sides. The bridge plugin writes its traffic log there;
                # OpenClaw state dir lives there; everything materialised on
                # host is readable inside the container.
                start_kwargs: dict = {"run_id": run_id}
                case_dir_mount: Path | None = None
                if args.harness == "openclaw":
                    case_dir_mount = (
                        Path(trace_dir) / f"{task.task_id}_{run_id}_raw"
                    )
                    case_dir_mount.mkdir(parents=True, exist_ok=True)
                    # Network mode: default "host" (Linux). On macOS Docker
                    # Desktop --network host does not bridge host<->container
                    # localhost, so opt into bridge + port-mapping +
                    # host.docker.internal via CLAWEVAL_SANDBOX_NET=bridge.
                    # Bridge mode: do NOT set network_mode -> SandboxRunner falls
                    # to port mapping; the bridge plugin reaches host mocks via
                    # host.docker.internal (rewritten in openclaw harness).
                    _net = os.environ.get("CLAWEVAL_SANDBOX_NET", "").strip().lower()
                    if _net != "bridge":
                        start_kwargs["network_mode"] = "host"
                        # Host networking makes the in-container sandbox server
                        # bind a fixed port (default 8080) directly on the host,
                        # so concurrent `run --sandbox` workers collide. Give each
                        # a unique port derived from its port_offset — mirrors
                        # cmd_batch (see sandbox_port = 8080 + port_offset below).
                        start_kwargs["sandbox_port"] = 8080 + port_offset
                    start_kwargs["volumes"] = {str(case_dir_mount): str(case_dir_mount)}
                    bridge_log_in_container = case_dir_mount / "raw" / "bridge_traffic.jsonl"
                    start_kwargs["extra_env"] = {
                        "CLAWEVAL_BRIDGE_LOG": str(bridge_log_in_container),
                    }
                handle = runner.start_container(**start_kwargs)
                try:
                    n_injected = runner.inject_files(handle, task, task_dir=str(task_yaml.parent))
                    expected_files = len(task.sandbox_files) if task.sandbox_files else len(getattr(task.environment, "fixtures", []))
                    if expected_files and n_injected < expected_files:
                        print(f"[WARNING] inject_files: only {n_injected}/{expected_files} files injected")
                    result = harness.run(
                        task,
                        trace_dir=trace_dir,
                        run_id=run_id,
                        cfg=cfg,
                        sandbox_handle=handle,
                        user_agent=_make_user_agent(cfg, task),
                        services_ctx=svc,
                        sandbox_tools=True,
                    )
                    _require_harness_ok(result)
                    trace_path = result.trace_path
                    # Inject grader-only files (e.g. verify scripts with answers)
                    # AFTER the agent loop so the agent cannot read them.
                    n_grader = runner.inject_grader_files(handle, task, task_dir=str(task_yaml.parent))
                    if task.sandbox_grader_files and n_grader < len(task.sandbox_grader_files):
                        print(f"[WARNING] inject_grader_files: only {n_grader}/{len(task.sandbox_grader_files)} files injected")
                    # Collect env snapshot before destroying container.
                    # ClawEvalHarness intentionally leaves env_snapshot=None so
                    # this collection can run AFTER inject_grader_files; other
                    # harnesses that populate result.env_snapshot win that race.
                    env_snapshot = result.env_snapshot
                    if env_snapshot is None:
                        env_snapshot = _collect_env_snapshot(handle.sandbox_url, task)
                    _save_env_snapshot(env_snapshot, trace_path, task.task_id)
                finally:
                    runner.stop_container(handle)

                # Read local grader files from host (GT files, never touched by agent)
                if task.local_grader_files:
                    import base64 as _b64
                    task_root = Path(str(task_yaml.parent))
                    for rel_path in task.local_grader_files:
                        local_path = task_root / rel_path
                        if local_path.exists():
                            content = _b64.b64encode(local_path.read_bytes()).decode()
                            env_snapshot[f"local_file:{rel_path}"] = {
                                "encoding": "base64",
                                "content": content,
                            }
                        else:
                            env_snapshot[f"local_file:{rel_path}"] = {
                                "error": f"not found: {local_path}",
                            }

                trace_paths.append(trace_path)
                print(f"Trace: {trace_path}")

                # Grade locally, optionally by replaying a frozen native bundle.
                start, _messages, _dispatches, _media_events, end, _audit_data = load_trace(trace_path)
                scores, judge_calls = _grade_run_result(
                    args=args,
                    task_yaml=task_yaml,
                    task=task,
                    tasks_dir=tasks_dir,
                    trace_path=trace_path,
                    env_snapshot=env_snapshot,
                    raw_dir=result.raw_dir,
                    rollout_status=result.status,
                    model_id=model_id,
                    judge=judge,
                )
                task_score = compute_task_score(scores)
                passed = is_pass(task_score)
                trial_scores.append(task_score)
                user_agent_meta = {}
                if end and end.user_agent_rounds > 0:
                    user_agent_meta = {
                        "rounds_used": end.user_agent_rounds,
                        "max_rounds": end.user_agent_max_rounds,
                        "done_reached": end.user_agent_done,
                    }
                _append_grading_to_trace(
                    trace_path,
                    trace_id=start.trace_id,
                    task_id=task.task_id,
                    scores=scores,
                    task_score=task_score,
                    passed=passed,
                    judge_calls=judge_calls,
                    user_agent_meta=user_agent_meta,
                )
                totals = _trace_totals(end)

                print(f"  completion:     {scores.completion:.2f}")
                print(f"  robustness:     {scores.robustness:.2f}")
                print(f"  communication:  {scores.communication:.2f}")
                print(f"  safety:         {scores.safety:.1f}")
                print(f"  task_score:     {task_score:.2f}")
                print(f"  passed:         {passed}")
                print(
                    f"  model_tokens:   {totals['total_tokens']} "
                    f"({totals['model_input_tokens']} in / {totals['model_output_tokens']} out)"
                )
                print(
                    f"  time_s:         wall={totals['wall_time_s']:.2f} "
                    f"model={totals['model_time_s']:.2f} tool={totals['tool_time_s']:.2f} "
                    f"other={totals['other_time_s']:.2f}"
                )

        if trials > 1:
            print(f"\n--- Multi-trial summary ({trials} trials) ---")
            for i, (score, path) in enumerate(zip(trial_scores, trace_paths)):
                print(f"  Trial {i+1}: score={score:.2f} pass={is_pass(score)} trace={path}")
            pass_at_1 = compute_pass_at_k(trial_scores, k=1)
            pass_hat_k = compute_pass_hat_k(trial_scores, k=trials)
            print(f"  pass@1:  {pass_at_1:.3f}")
            print(f"  pass^{trials}:  {pass_hat_k:.3f}")
        return

    # ---- Normal (local) mode ----
    judge = _make_judge(cfg, args)
    sandbox_tools = getattr(args, "sandbox_tools", False)

    from .runner.services import ServiceManager

    trials = args.trials or 1
    trial_scores_local: list[float] = []
    trace_paths_local: list[Path] = []

    with ServiceManager(task.services, mock_today=task.environment.mock_today) as svc:
        for i in range(trials):
            if trials > 1:
                print(f"\n--- Trial {i + 1}/{trials} ---")

            # Reset mock service state between trials
            if i > 0:
                svc.reset_all()

            run_id = f"{task.task_id}-trial{i}"
            result = harness.run(
                task,
                trace_dir=trace_dir,
                run_id=run_id,
                cfg=cfg,
                sandbox_handle=None,
                user_agent=_make_user_agent(cfg, task),
                services_ctx=svc,
                sandbox_tools=sandbox_tools,
            )
            _require_harness_ok(result)
            trace_path = result.trace_path
            trace_paths_local.append(trace_path)
            print(f"Trace: {trace_path}")

            # Read local grader files from host (GT files, never touched by agent)
            env_snapshot: dict | None = result.env_snapshot
            if task.local_grader_files:
                if env_snapshot is None:
                    env_snapshot = {}
                import base64 as _b64
                task_root = Path(str(task_yaml.parent))
                for rel_path in task.local_grader_files:
                    local_path = task_root / rel_path
                    if local_path.exists():
                        content = _b64.b64encode(local_path.read_bytes()).decode()
                        env_snapshot[f"local_file:{rel_path}"] = {
                            "encoding": "base64",
                            "content": content,
                        }
                    else:
                        env_snapshot[f"local_file:{rel_path}"] = {
                            "error": f"not found: {local_path}",
                        }

            # Grade, optionally by replaying a frozen native bundle.
            _start, _messages, _dispatches, _media_events, end, _audit_data = load_trace(trace_path)
            scores, judge_calls = _grade_run_result(
                args=args,
                task_yaml=task_yaml,
                task=task,
                tasks_dir=tasks_dir,
                trace_path=trace_path,
                env_snapshot=env_snapshot,
                raw_dir=result.raw_dir,
                rollout_status=result.status,
                model_id=model_id,
                judge=judge,
            )
            task_score = compute_task_score(scores)
            passed = is_pass(task_score)
            trial_scores_local.append(task_score)
            totals = _trace_totals(end)

            print(f"  completion:     {scores.completion:.2f}")
            print(f"  robustness:     {scores.robustness:.2f}")
            print(f"  communication:  {scores.communication:.2f}")
            print(f"  safety:         {scores.safety:.1f}")
            print(f"  task_score:     {task_score:.2f}")
            print(f"  passed:         {passed}")
            print(
                f"  model_tokens:   {totals['total_tokens']} "
                f"({totals['model_input_tokens']} in / {totals['model_output_tokens']} out)"
            )
            print(
                f"  time_s:         wall={totals['wall_time_s']:.2f} "
                f"model={totals['model_time_s']:.2f} tool={totals['tool_time_s']:.2f} "
                f"other={totals['other_time_s']:.2f}"
            )

    if trials > 1:
        print(f"\n--- Multi-trial summary ({trials} trials) ---")
        for i, (score, path) in enumerate(zip(trial_scores_local, trace_paths_local)):
            print(f"  Trial {i+1}: score={score:.2f} pass={is_pass(score)} trace={path}")
        pass_at_1 = compute_pass_at_k(trial_scores_local, k=1)
        pass_hat_k = compute_pass_hat_k(trial_scores_local, k=trials)
        print(f"  pass@1:  {pass_at_1:.3f}")
        print(f"  pass^{trials}:  {pass_hat_k:.3f}")


def cmd_run_inner(args: argparse.Namespace) -> None:
    """Run a single trial inside a sandbox container (internal command)."""
    _apply_proxy(getattr(args, "proxy", None))

    from .config import load_config
    from .graders.registry import get_grader
    from .harnesses import get_harness
    from .models.scoring import compute_task_score, is_pass
    from .models.task import TaskDefinition
    from .runner.services import ServiceManager
    from .trace.reader import load_trace

    cfg = load_config(args.config)
    _require_provider_transport(cfg, args)
    harness = get_harness(args.harness)

    task_yaml = _resolve_task_yaml(args.task)
    task = TaskDefinition.from_yaml(task_yaml)
    tasks_dir = _resolve_tasks_dir(task_yaml)
    _run_harness_preflight(harness, task)

    model_id = args.model or cfg.model.model_id
    # _run-inner has an extra fallback to $OPENAI_API_KEY that the other paths
    # don't — preserved here by resolving the effective key before baking into cfg.
    effective_api_key = args.api_key or cfg.model.api_key or os.environ.get("OPENAI_API_KEY")
    cfg = _cfg_with_model_overrides(
        cfg, model=args.model, api_key=effective_api_key, base_url=args.base_url,
    )

    sandbox_tools = getattr(args, "sandbox_tools", False)
    # _run-inner receives the final trace dir from the caller (e.g. submit script).
    # Only fall back to _make_trace_dir when --trace-dir is not provided.
    if args.trace_dir:
        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
    else:
        trace_dir = _make_trace_dir(cfg.defaults.trace_dir, model_id)

    run_id = f"{task.task_id}-inner"
    with ServiceManager(task.services, mock_today=task.environment.mock_today) as svc:
        result = harness.run(
            task,
            trace_dir=trace_dir,
            run_id=run_id,
            cfg=cfg,
            sandbox_handle=None,
            user_agent=_make_user_agent(cfg, task),
            services_ctx=svc,
            sandbox_tools=sandbox_tools,
        )
    _require_harness_ok(result)
    trace_path = result.trace_path

    print(f"Trace: {trace_path}")

    # --- Inline grading ---
    judge = _make_judge(cfg, args)
    start, messages, dispatches, media_events, end, audit_data = load_trace(trace_path)
    grader = get_grader(task.task_id, tasks_dir=tasks_dir, task_dir=task_yaml.parent)
    scores, judge_calls = _grade_with_optional_params(
        grader, messages, dispatches, task,
        audit_data=audit_data, judge=judge, media_events=media_events,
    )
    task_score = compute_task_score(scores)
    passed = is_pass(task_score)

    totals = _trace_totals(end)
    result = {
        "task_id": task.task_id,
        "task_name": task.task_name,
        "model": model_id,
        "trace": trace_path.name,
        "turns": end.total_turns if end else 0,
        "model_input_tokens": totals["model_input_tokens"],
        "model_output_tokens": totals["model_output_tokens"],
        "input_tokens": totals["model_input_tokens"],
        "output_tokens": totals["model_output_tokens"],
        "tokens": totals["total_tokens"],
        "model_time_s": totals["model_time_s"],
        "tool_time_s": totals["tool_time_s"],
        "other_time_s": totals["other_time_s"],
        "wall_time_s": totals["wall_time_s"],
        "completion": scores.completion,
        "robustness": scores.robustness,
        "communication": scores.communication,
        "safety": scores.safety,
        "task_score": task_score,
        "passed": passed,
    }
    result_path = trace_path.with_suffix(".result.json")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Result: {result_path}")
    print(f"  task_score={task_score:.3f}  passed={passed}")


def cmd_build_image(args: argparse.Namespace) -> None:
    """Build the sandbox Docker image."""
    from .config import load_config

    cfg = load_config(getattr(args, "config", None))

    from .runner.sandbox_runner import SandboxRunner

    image = getattr(args, "image", None) or cfg.sandbox.image
    context = getattr(args, "context", ".")
    dockerfile = getattr(args, "dockerfile", "Dockerfile.agent")
    runner = SandboxRunner(cfg.sandbox, image=image)
    runner.build_image(context_path=context, dockerfile=dockerfile)


def cmd_grade(args: argparse.Namespace) -> None:
    """Grade an existing trace file."""
    _apply_proxy(getattr(args, "proxy", None))

    from .config import load_config
    from .graders.registry import get_grader
    from .models.scoring import compute_task_score, is_pass
    from .models.task import TaskDefinition
    from .trace.reader import load_trace

    cfg = load_config(args.config if hasattr(args, "config") else None)
    judge = _make_judge(cfg, args)

    start, messages, dispatches, media_events, end, audit_data = load_trace(args.trace)

    task_yaml = _resolve_task_yaml(args.task)
    task = TaskDefinition.from_yaml(task_yaml)
    tasks_dir = _resolve_tasks_dir(task_yaml)

    grader = get_grader(task.task_id, tasks_dir=tasks_dir, task_dir=task_yaml.parent)
    scores, judge_calls = _grade_with_optional_params(
        grader, messages, dispatches, task,
        audit_data=audit_data, judge=judge, media_events=media_events,
    )
    task_score = compute_task_score(scores)
    passed = is_pass(task_score)

    print(f"Trace:   {args.trace}")
    print(f"Task:    {task.task_id} ({task.task_name})")
    print(f"Model:   {start.model}")
    print(f"Turns:   {end.total_turns if end else '?'}")
    totals = _trace_totals(end)
    print(
        f"Tokens:  {totals['total_tokens']} "
        f"({totals['model_input_tokens']} in / {totals['model_output_tokens']} out)"
    )
    print(
        f"Time:    wall={totals['wall_time_s']:.2f}s "
        f"model={totals['model_time_s']:.2f}s "
        f"tool={totals['tool_time_s']:.2f}s "
        f"other={totals['other_time_s']:.2f}s"
    )
    print()
    print(f"completion:     {scores.completion:.2f}")
    print(f"robustness:     {scores.robustness:.2f}")
    print(f"communication:  {scores.communication:.2f}")
    print(f"safety:         {scores.safety:.1f}")
    print(f"task_score:     {task_score:.2f}")
    print(f"passed:         {passed}")
    if judge_calls:
        print(f"\n--- Judge Calls ({len(judge_calls)}) ---")
        for i, c in enumerate(judge_calls):
            print(f"  [{i+1}] {c['method']} score={c['score']:.2f}")
            print(f"       {c['reasoning'][:200]}")


def cmd_grade_bundle(args: argparse.Namespace) -> None:
    """Grade an immutable native rollout bundle."""
    _apply_proxy(getattr(args, "proxy", None))

    from .config import load_config
    from .rollout_bundle import grade_native_rollout, load_native_rollout_bundle

    cfg = load_config(args.config if hasattr(args, "config") else None)
    judge = _make_judge(cfg, args)
    bundle = load_native_rollout_bundle(Path(args.bundle))
    grade = grade_native_rollout(bundle.root, judge=judge)
    scores = grade.scores

    print(f"Bundle:  {bundle.root / 'bundle.json'}")
    print(f"Task:    {bundle.manifest['task_id']}")
    print(f"Model:   {bundle.manifest['model']}")
    print(f"Harness: {bundle.manifest['harness']}")
    print()
    print(f"completion:     {scores.completion:.2f}")
    print(f"robustness:     {scores.robustness:.2f}")
    print(f"communication:  {scores.communication:.2f}")
    print(f"safety:         {scores.safety:.1f}")
    print(f"task_score:     {grade.task_score:.2f}")
    print(f"passed:         {grade.passed}")


def _append_grading_to_trace(
    trace_path: Path,
    trace_id: str,
    task_id: str,
    scores,
    task_score: float,
    passed: bool,
    judge_calls: list[dict] | None = None,
    user_agent_meta: dict | None = None,
) -> None:
    """Append a grading_result event to the end of a trace JSONL file."""
    from .models.trace import GradingResult, DimensionScores

    event = GradingResult(
        trace_id=trace_id,
        task_id=task_id,
        scores=DimensionScores(
            completion=scores.completion,
            robustness=scores.robustness,
            communication=scores.communication,
            safety=scores.safety,
        ),
        task_score=task_score,
        passed=passed,
        judge_calls=judge_calls or [],
        user_agent_meta=user_agent_meta or {},
    )
    with open(trace_path, "a") as fh:
        fh.write(event.model_dump_json() + "\n")


def _run_single_task(
    task_dir: str,
    config_path: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    trace_dir: str | None,
    port_offset: int,
    no_judge: bool,
    judge_model: str | None,
    trials: int,
    proxy: str | None = None,
    sandbox: bool = False,
    sandbox_image: str | None = None,
    sandbox_tools: bool = False,
    harness_name: str = "claweval",
) -> dict:
    """Run a single task in a worker process. Returns a result dict."""
    # Ensure localhost bypasses proxy in worker processes.
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    # Re-apply proxy for model/judge API calls (services.py strips proxy
    # from mock-service subprocesses independently).
    _apply_proxy(proxy)

    from .config import load_config
    from .graders.registry import get_grader
    from .harnesses import get_harness
    from .models.scoring import compute_pass_at_k, compute_pass_hat_k, compute_task_score, is_pass
    from .models.task import TaskDefinition
    from .runner.services import ServiceManager
    from .trace.reader import load_trace

    harness = get_harness(harness_name)

    task_yaml = _resolve_task_yaml(task_dir)
    task = TaskDefinition.from_yaml(task_yaml)
    tasks_dir = _resolve_tasks_dir(task_yaml)

    # Preflight: if blocked, surface as a task-level error (worker can't raise
    # SystemExit cleanly inside ProcessPoolExecutor).
    preflight_errs = harness.preflight(task)
    if preflight_errs:
        return {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "difficulty": task.difficulty,
            "trials": [],
            "error": "preflight: " + "; ".join(preflight_errs),
        }

    if port_offset:
        task.apply_port_offset(port_offset)

    cfg = load_config(config_path)
    cfg = _cfg_with_model_overrides(cfg, model=model, api_key=api_key, base_url=base_url)

    # Build judge if needed
    judge = None
    if not no_judge and cfg.judge.enabled and cfg.judge.api_key:
        from .graders.llm_judge import LLMJudge
        judge = LLMJudge(
            model_id=judge_model or cfg.judge.model_id,
            api_key=cfg.judge.api_key,
            base_url=cfg.judge.base_url,
        )

    # Resolve sandbox mode
    sandbox_mode = sandbox or cfg.sandbox.enabled
    sandbox_runner = None
    if sandbox_mode:
        from .runner.sandbox_runner import SandboxRunner
        # Mirror cmd_run's openclaw image default (cli.py:437) so batch and
        # single-task runs behave identically. Without this, --harness openclaw
        # in batch mode tries to pull "claw-eval-agent:latest" (the generic
        # default) instead of the locally-built "claw-eval-agent-openclaw:latest".
        #
        # Same default for aorchestra: it doesn't ship its own sandbox image,
        # and the openclaw image's sandbox-server portion is all aorchestra
        # actually needs from the container (sandbox tools route through it).
        resolved_image = sandbox_image or cfg.sandbox.image
        if harness_name in ("openclaw", "aorchestra") and sandbox_image is None:
            resolved_image = "claw-eval-agent-openclaw:latest"
        sandbox_runner = SandboxRunner(cfg.sandbox, image=resolved_image)

    # Phase 4 Wave 4-D §4.2 — same asymmetric AOrchestra gate as cmd_run.
    if harness_name == "aorchestra":
        from .runner.sandbox_tools import SANDBOX_TOOL_NAMES as _SBX

        task_needs_sandbox = any(t.name in _SBX for t in task.tools)
        if task_needs_sandbox and not sandbox_mode:
            sbx_names = [t.name for t in task.tools if t.name in _SBX]
            return {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "difficulty": task.difficulty,
                "trials": [],
                "error": (
                    f"--harness aorchestra requires --sandbox when task declares "
                    f"sandbox tools {sbx_names} (design doc §4.2)"
                ),
            }

    result = {
        "task_id": task.task_id,
        "task_name": task.task_name,
        "difficulty": task.difficulty,
        "trials": [],
        "error": None,
    }

    import time
    from openai import APIConnectionError, APITimeoutError, InternalServerError

    max_retries = 3
    for attempt in range(max_retries):
        result["trials"] = []
        result["error"] = None
        try:
            with ServiceManager(task.services, cwd=tasks_dir.parent, mock_today=task.environment.mock_today) as svc:
                for i in range(trials):
                    if i > 0:
                        svc.reset_all()

                    try:
                        env_snapshot = None
                        if sandbox_runner:
                            run_id = f"{task.task_id}-t{i}-p{port_offset}"
                            # OpenClaw needs host networking + a volume mount of
                            # the per-task case_dir + the bridge-log env, exactly
                            # like cmd_run (cli.py). Without these the bridge
                            # install (docker exec npm install in a host path not
                            # mounted in the container) fails with rc=127. Under
                            # concurrency, host networking makes the sandbox server
                            # bind a fixed port (8080) that would collide across
                            # workers, so we give each worker a unique sandbox_port
                            # derived from its port_offset (mock services are
                            # already offset by apply_port_offset).
                            start_kwargs: dict = {"run_id": run_id}
                            _resolved_trace_dir = trace_dir or cfg.defaults.trace_dir
                            if harness_name == "openclaw":
                                _case_dir_mount = (
                                    Path(_resolved_trace_dir) / f"{task.task_id}_{run_id}_raw"
                                )
                                _case_dir_mount.mkdir(parents=True, exist_ok=True)
                                _bridge_log_in_container = _case_dir_mount / "raw" / "bridge_traffic.jsonl"
                                start_kwargs["network_mode"] = "host"
                                start_kwargs["volumes"] = {str(_case_dir_mount): str(_case_dir_mount)}
                                start_kwargs["extra_env"] = {
                                    "CLAWEVAL_BRIDGE_LOG": str(_bridge_log_in_container),
                                }
                                start_kwargs["sandbox_port"] = 8080 + port_offset
                            handle = sandbox_runner.start_container(**start_kwargs)
                            try:
                                n_injected = sandbox_runner.inject_files(handle, task, task_dir=task_dir)
                                expected_files = len(task.sandbox_files) if task.sandbox_files else len(getattr(task.environment, "fixtures", []))
                                if expected_files and n_injected < expected_files:
                                    print(f"[WARNING] inject_files: only {n_injected}/{expected_files} files injected")
                                result_h = harness.run(
                                    task,
                                    trace_dir=trace_dir or cfg.defaults.trace_dir,
                                    run_id=run_id,
                                    cfg=cfg,
                                    sandbox_handle=handle,
                                    user_agent=_make_user_agent(cfg, task),
                                    services_ctx=svc,
                                    sandbox_tools=True,
                                )
                                _require_harness_ok(result_h)
                                trace_path = result_h.trace_path
                                n_grader = sandbox_runner.inject_grader_files(handle, task, task_dir=task_dir)
                                if task.sandbox_grader_files and n_grader < len(task.sandbox_grader_files):
                                    print(f"[WARNING] inject_grader_files: only {n_grader}/{len(task.sandbox_grader_files)} files injected")
                                # ClawEvalHarness leaves env_snapshot=None so
                                # the collection can run AFTER inject_grader_files;
                                # other harnesses that populate it win that race.
                                env_snapshot = result_h.env_snapshot
                                if env_snapshot is None:
                                    env_snapshot = _collect_env_snapshot(handle.sandbox_url, task)
                                _save_env_snapshot(env_snapshot, trace_path, task.task_id)
                            finally:
                                sandbox_runner.stop_container(handle)
                        else:
                            run_id = f"{task.task_id}-t{i}-p{port_offset}"
                            result_h = harness.run(
                                task,
                                trace_dir=trace_dir or cfg.defaults.trace_dir,
                                run_id=run_id,
                                cfg=cfg,
                                sandbox_handle=None,
                                user_agent=_make_user_agent(cfg, task),
                                services_ctx=svc,
                                sandbox_tools=sandbox_tools,
                            )
                            _require_harness_ok(result_h)
                            trace_path = result_h.trace_path
                            env_snapshot = result_h.env_snapshot

                        # Read local grader files from host (GT files, never touched by agent)
                        if task.local_grader_files:
                            if env_snapshot is None:
                                env_snapshot = {}
                            import base64 as _b64
                            task_root = Path(task_dir)
                            for rel_path in task.local_grader_files:
                                local_path = task_root / rel_path
                                if local_path.exists():
                                    content = _b64.b64encode(local_path.read_bytes()).decode()
                                    env_snapshot[f"local_file:{rel_path}"] = {
                                        "encoding": "base64",
                                        "content": content,
                                    }
                                else:
                                    env_snapshot[f"local_file:{rel_path}"] = {
                                        "error": f"not found: {local_path}",
                                    }

                        start, messages, dispatches, media_events, end, audit_data = load_trace(trace_path)
                        grader = get_grader(task.task_id, tasks_dir=tasks_dir, task_dir=task_dir)
                        scores, judge_calls = _grade_with_optional_params(
                            grader, messages, dispatches, task,
                            audit_data=audit_data, judge=judge, media_events=media_events,
                            env_snapshot=env_snapshot,
                        )
                        task_score = compute_task_score(scores)
                        user_agent_meta = {}
                        if end and end.user_agent_rounds > 0:
                            user_agent_meta = {
                                "rounds_used": end.user_agent_rounds,
                                "max_rounds": end.user_agent_max_rounds,
                                "done_reached": end.user_agent_done,
                            }
                        _append_grading_to_trace(
                            trace_path,
                            trace_id=start.trace_id,
                            task_id=task.task_id,
                            scores=scores,
                            task_score=task_score,
                            passed=is_pass(task_score),
                            judge_calls=judge_calls,
                            user_agent_meta=user_agent_meta,
                        )
                        totals = _trace_totals(end)
                        result["trials"].append({
                            "trace": str(trace_path),
                            "model_input_tokens": totals["model_input_tokens"],
                            "model_output_tokens": totals["model_output_tokens"],
                            "input_tokens": totals["model_input_tokens"],
                            "output_tokens": totals["model_output_tokens"],
                            "tokens": totals["total_tokens"],
                            "model_time_s": totals["model_time_s"],
                            "tool_time_s": totals["tool_time_s"],
                            "other_time_s": totals["other_time_s"],
                            "wall_time_s": totals["wall_time_s"],
                            "completion": scores.completion,
                            "robustness": scores.robustness,
                            "communication": scores.communication,
                            "safety": scores.safety,
                            "task_score": task_score,
                            "passed": is_pass(task_score),
                        })
                    except Exception as trial_exc:
                        result["trials"].append({
                            "trial": i,
                            "error": str(trial_exc),
                            "task_score": 0.0,
                            "passed": False,
                        })
            break  # success — exit retry loop
        except (APIConnectionError, APITimeoutError, InternalServerError, ConnectionError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s
                print(f"  [{task.task_id}] retry {attempt + 1}/{max_retries} after {type(e).__name__}, waiting {wait}s")
                time.sleep(wait)
            else:
                result["error"] = str(e)
        except Exception as e:
            result["error"] = str(e)
            break  # non-retryable error

    # Compute multi-trial aggregate metrics (exclude errored trials)
    valid_trials = [t for t in result["trials"] if not t.get("error")]
    if not valid_trials and result["trials"]:
        # All trials errored — propagate as task-level error for summary stats
        result["error"] = result["trials"][0].get("error", "all trials errored")
    trial_scores = [t["task_score"] for t in valid_trials]
    n_trials = len(trial_scores)
    if n_trials > 0:
        result["avg_score"] = sum(trial_scores) / n_trials
        result["pass_at_1"] = compute_pass_at_k(trial_scores, k=1)
        result["pass_hat_k"] = compute_pass_hat_k(trial_scores, k=n_trials)
        result["avg_passed"] = is_pass(result["avg_score"])
    else:
        result["avg_score"] = 0.0
        result["pass_at_1"] = 0.0
        result["pass_hat_k"] = 0.0
        result["avg_passed"] = False

    return result


def _scan_completed_trials(trace_dir: Path) -> dict[str, int]:
    """Scan a trace directory and return {task_id: completed_trial_count}.

    A trial is considered complete if its JSONL file contains a grading_result event.
    """
    from collections import defaultdict

    completed: dict[str, int] = defaultdict(int)
    for f in trace_dir.glob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "grading_result":
                    task_id = ev.get("task_id", "")
                    if task_id:
                        completed[task_id] += 1
                    break  # one grading_result per file is enough
    return dict(completed)


def _load_completed_results(trace_dir: Path) -> list[dict]:
    """Load per-trial results from grading_result events in a trace directory.

    Returns a list of result dicts (one per task_id) with trials populated from
    the grading_result events found in JSONL files. This allows merging with
    new results when using --continue.
    """
    from collections import defaultdict

    # task_id -> list of trial info dicts
    task_trials: dict[str, list[dict]] = defaultdict(list)

    for f in sorted(trace_dir.glob("*.jsonl")):
        grading = None
        trace_end = None
        for line_str in open(f):
            line_str = line_str.strip()
            if not line_str:
                continue
            try:
                ev = json.loads(line_str)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "grading_result":
                grading = ev
            elif ev.get("type") == "trace_end":
                trace_end = ev

        if grading is None:
            continue

        task_id = grading.get("task_id", "")
        if not task_id:
            continue

        scores = grading.get("scores", {})
        trial_info = {
            "trace": str(f),
            "model_input_tokens": trace_end.get("model_input_tokens", 0) if trace_end else 0,
            "model_output_tokens": trace_end.get("model_output_tokens", 0) if trace_end else 0,
            "input_tokens": trace_end.get("model_input_tokens", 0) if trace_end else 0,
            "output_tokens": trace_end.get("model_output_tokens", 0) if trace_end else 0,
            "tokens": trace_end.get("total_tokens", 0) if trace_end else 0,
            "model_time_s": trace_end.get("model_time_s", 0.0) if trace_end else 0.0,
            "tool_time_s": trace_end.get("tool_time_s", 0.0) if trace_end else 0.0,
            "other_time_s": trace_end.get("other_time_s", 0.0) if trace_end else 0.0,
            "wall_time_s": trace_end.get("wall_time_s", 0.0) if trace_end else 0.0,
            "completion": scores.get("completion", 0.0),
            "robustness": scores.get("robustness", 0.0),
            "communication": scores.get("communication", 0.0),
            "safety": scores.get("safety", 1.0),
            "task_score": grading.get("task_score", 0.0),
            "passed": grading.get("passed", False),
        }
        task_trials[task_id].append(trial_info)

    # Build result dicts per task
    from .models.scoring import compute_pass_at_k, compute_pass_hat_k, is_pass

    results = []
    for task_id, trials in task_trials.items():
        trial_scores = [t["task_score"] for t in trials]
        n = len(trial_scores)
        result = {
            "task_id": task_id,
            "task_name": "",
            "difficulty": "",
            "trials": trials,
            "error": None,
        }
        if n > 0:
            result["avg_score"] = sum(trial_scores) / n
            result["pass_at_1"] = compute_pass_at_k(trial_scores, k=1)
            result["pass_hat_k"] = compute_pass_hat_k(trial_scores, k=n)
            result["avg_passed"] = is_pass(result["avg_score"])
        results.append(result)

    return results


def _fmt_duration(seconds: float) -> str:
    """Format seconds as e.g. '3m22s' or '1h05m'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def cmd_batch(args: argparse.Namespace) -> None:
    """Run all (or filtered) tasks in parallel."""
    _apply_proxy(getattr(args, "proxy", None))

    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.exists():
        print(f"Tasks directory not found: {tasks_dir}")
        sys.exit(1)

    # --rerun-errors: load previous results and filter to errored tasks only
    rerun_dir = getattr(args, "rerun_errors", None)
    prev_results: list[dict] | None = None
    errored_task_ids: set[str] = set()
    if rerun_dir:
        rerun_path = Path(rerun_dir)
        prev_results_file = rerun_path / "batch_results.json"
        if not prev_results_file.exists():
            print(f"batch_results.json not found in {rerun_path}")
            sys.exit(1)
        with open(prev_results_file) as f:
            prev_results = json.load(f)
        errored_task_ids = {r["task_id"] for r in prev_results if r.get("error")}
        if not errored_task_ids:
            print("No errored tasks found in previous run — nothing to rerun.")
            return
        print(f"[rerun-errors] Found {len(errored_task_ids)} errored tasks to rerun:")
        for tid in sorted(errored_task_ids):
            err_msg = next((r["error"] for r in prev_results if r["task_id"] == tid), "")
            print(f"  {tid}: {err_msg[:80]}")
        print()

    # --continue: scan existing trace dir for completed trials
    continue_dir = getattr(args, "continue_dir", None)
    completed_trials: dict[str, int] = {}
    continue_prev_results: list[dict] = []
    if continue_dir:
        continue_path = Path(continue_dir)
        if not continue_path.exists():
            print(f"Continue directory not found: {continue_path}")
            sys.exit(1)
        completed_trials = _scan_completed_trials(continue_path)
        continue_prev_results = _load_completed_results(continue_path)
        total_completed = sum(completed_trials.values())
        print(f"[continue] Scanning {continue_path} — found {total_completed} completed trial(s) "
              f"across {len(completed_trials)} task(s)")
        if completed_trials:
            for tid in sorted(completed_trials):
                print(f"  {tid}: {completed_trials[tid]} trial(s) done")
            print()

    # Discover tasks
    task_dirs = sorted(
        str(d) for d in tasks_dir.iterdir()
        if d.is_dir() and (d / "task.yaml").exists()
    )

    # --task-ids: authoritative selection of an arbitrary, possibly
    # non-contiguous set of tasks (e.g. T002,T008,T012,T018,T077). It is
    # mutually exclusive with the other selectors — combining them is almost
    # always a mistake, so we error loudly rather than silently picking a winner.
    task_ids_arg = getattr(args, "task_ids", None)
    if task_ids_arg:
        conflicting = [
            name for name, val in (
                ("--filter", args.filter),
                ("--tag", args.tag),
                ("--range", getattr(args, "range", None)),
            )
            if val
        ]
        if conflicting:
            print(
                f"[ERROR] --task-ids is mutually exclusive with "
                f"{', '.join(conflicting)} — pass only one selection method."
            )
            sys.exit(1)

        import re as _re

        requested = [tok.strip() for tok in task_ids_arg.split(",") if tok.strip()]
        if not requested:
            print("[ERROR] --task-ids was given but contained no task IDs.")
            sys.exit(1)

        # Build a lookup from both the full dir name (== task_id) and the bare
        # numeric ID prefix (T002) to the task dir path(s).
        by_name: dict[str, str] = {}
        by_numeric: dict[str, list[str]] = {}
        for d in task_dirs:
            name = Path(d).name
            by_name[name] = d
            nm = _re.match(r"(T\d+)", name)
            if nm:
                by_numeric.setdefault(nm.group(1), []).append(d)

        selected: list[str] = []
        not_found: list[str] = []
        seen: set[str] = set()
        for tok in requested:
            match = None
            if tok in by_name:
                match = by_name[tok]
            elif tok in by_numeric:
                cands = by_numeric[tok]
                if len(cands) > 1:
                    print(
                        f"[ERROR] --task-ids token {tok!r} is ambiguous — matches "
                        f"{[Path(c).name for c in cands]}. Use the full task-dir name."
                    )
                    sys.exit(1)
                match = cands[0]
            if match is None:
                not_found.append(tok)
            elif match not in seen:
                seen.add(match)
                selected.append(match)

        if not_found:
            print(
                f"[ERROR] --task-ids: {len(not_found)} requested ID(s) not found "
                f"in {tasks_dir}: {', '.join(not_found)}"
            )
            sys.exit(1)

        task_dirs = selected
    else:
        if args.filter:
            filt = args.filter.lower()
            task_dirs = [d for d in task_dirs if filt in d.lower()]

        if args.tag:
            from .models.task import TaskDefinition as _TD
            filtered = []
            for d in task_dirs:
                td = _TD.from_yaml(Path(d) / "task.yaml")
                if args.tag in td.tags:
                    filtered.append(d)
            task_dirs = filtered

        if getattr(args, "range", None):
            import re as _re
            _m = _re.match(r"(\d+)-(\d+)$", args.range)
            if not _m:
                print(f"[ERROR] Invalid --range format: {args.range}  (expected L-R, e.g. 1-104)")
                sys.exit(1)
            lo, hi = int(_m.group(1)), int(_m.group(2))
            def _in_range(d):
                name = Path(d).name
                m = _re.match(r"T(\d+)", name)
                return m is not None and lo <= int(m.group(1)) <= hi
            task_dirs = [d for d in task_dirs if _in_range(d)]

    # If rerunning errors, only keep the errored task dirs
    if errored_task_ids:
        task_dirs = [d for d in task_dirs if Path(d).name in errored_task_ids]

    workers = args.parallel
    trials = args.trials or 1

    # If continuing, filter out fully-completed tasks and compute remaining trials per task
    skipped_task_ids: set[str] = set()
    remaining_trials: dict[str, int] = {}  # task_dir -> number of trials still needed
    if continue_dir:
        remaining_dirs = []
        for d in task_dirs:
            task_id = Path(d).name
            done = completed_trials.get(task_id, 0)
            if done >= trials:
                skipped_task_ids.add(task_id)
            else:
                remaining_dirs.append(d)
                remaining_trials[d] = trials - done
        n_skipped = len(task_dirs) - len(remaining_dirs)
        task_dirs = remaining_dirs
        if n_skipped:
            print(f"[continue] Skipping {n_skipped} task(s) with {trials}+ completed trial(s)")
        for d in task_dirs:
            needed = remaining_trials[d]
            if needed < trials:
                print(f"  {Path(d).name}: {trials - needed}/{trials} done, running {needed} more")

    if not task_dirs:
        if continue_dir:
            print("All tasks already completed — nothing to run.")
        else:
            print("No tasks matched.")
        return

    total = len(task_dirs)

    # Build a shared trace output directory for this batch run
    from .config import load_config as _load_cfg_early
    _cfg_early = _load_cfg_early(args.config)
    _model_id = args.model or _cfg_early.model.model_id
    _base_trace_dir = args.trace_dir or _cfg_early.defaults.trace_dir

    if rerun_dir:
        # Reuse the existing trace directory
        batch_trace_dir = str(Path(rerun_dir))
    elif continue_dir:
        # Reuse the continue trace directory
        batch_trace_dir = str(Path(continue_dir))
    else:
        batch_trace_dir = str(_make_trace_dir(_base_trace_dir, _model_id))

    print(f"Running {total} tasks with {workers} parallel workers, {trials} trial(s) each")
    print(f"Traces → {batch_trace_dir}\n")

    results: list[dict] = []
    # Progress tracking
    start_time = time.monotonic()
    n_pass_hat = 0      # pass^k: all trials passed
    n_pass_at = 0       # pass@k: at least one trial passed
    score_sum = 0.0
    finished_tasks = 0

    # Each worker slot gets a unique port offset: slot 0 → 0, slot 1 → 50, ...
    # Tasks use ports 9100-9129 (span=30); stride of 50 leaves headroom.
    # We map futures to their assigned slot so we can recycle offsets.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Slot pool: available port offsets
        available_slots = list(range(workers))
        pending: dict = {}  # future → (task_dir, slot_index)

        task_queue = list(task_dirs)
        finished = 0

        port_base_offset = getattr(args, "port_base_offset", 0)

        # Sanity check: max port must stay below ephemeral range (32768)
        _STRIDE = 50  # port gap between adjacent worker slots
        max_port = 9129 + port_base_offset + (workers - 1) * _STRIDE
        if max_port >= 32768:
            max_safe = (32767 - 9129 - port_base_offset) // _STRIDE + 1
            print(
                f"[ERROR] --port-base-offset {port_base_offset} with {workers} workers "
                f"would use port {max_port} (>=32768, collides with ephemeral range). "
                f"Max workers for this offset: {max_safe}"
            )
            return

        def _submit(td: str) -> None:
            slot = available_slots.pop(0)
            offset = port_base_offset + slot * _STRIDE
            # Use per-task remaining trials when continuing, otherwise full trials
            task_trials = remaining_trials.get(td, trials)
            fut = pool.submit(
                _run_single_task,
                task_dir=td,
                config_path=args.config,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                trace_dir=batch_trace_dir,
                port_offset=offset,
                no_judge=args.no_judge,
                judge_model=getattr(args, "judge_model", None),
                trials=task_trials,
                proxy=getattr(args, "proxy", None),
                sandbox=getattr(args, "sandbox", False),
                sandbox_image=getattr(args, "sandbox_image", None),
                sandbox_tools=getattr(args, "sandbox_tools", False),
                harness_name=getattr(args, "harness", "claweval"),
            )
            pending[fut] = (td, slot)

        # Seed initial batch
        while task_queue and available_slots:
            _submit(task_queue.pop(0))

        # Process completions
        while pending:
            for fut in as_completed(pending):
                td, slot = pending.pop(fut)
                available_slots.append(slot)
                finished += 1

                try:
                    res = fut.result()
                except Exception as e:
                    res = {"task_id": Path(td).name, "error": str(e), "trials": []}

                results.append(res)

                # Incrementally write batch_results.json after each task
                _partial_out = Path(batch_trace_dir)
                _partial_out.mkdir(parents=True, exist_ok=True)
                _partial_file = _partial_out / "batch_results.json"
                try:
                    with open(_partial_file, "w") as _pf:
                        json.dump(results, _pf, indent=2, ensure_ascii=False)
                except Exception:
                    pass  # best-effort; don't crash on incremental write failure

                # Update progress counters
                finished_tasks += 1
                if res.get("error"):
                    score_sum += 0.0
                else:
                    trials_list = res["trials"]
                    score_sum += sum(tr["task_score"] for tr in trials_list) / len(trials_list)
                    if all(tr["passed"] for tr in trials_list):
                        n_pass_hat += 1
                    if any(tr["passed"] for tr in trials_list):
                        n_pass_at += 1

                # Print task result
                tid = res.get("task_id", Path(td).name)
                if res.get("error"):
                    print(f"  [{finished}/{total}] {tid}: ERROR — {res['error'][:80]}")
                else:
                    for i, tr in enumerate(res["trials"]):
                        label = f" trial {i+1}" if trials > 1 else ""
                        status = "PASS" if tr["passed"] else "FAIL"
                        print(
                            f"  [{finished}/{total}] {tid}{label}: {tr['task_score']:.2f} {status} "
                            f"| tok={tr.get('tokens', 0)} "
                            f"({tr.get('model_input_tokens', tr.get('input_tokens', 0))} in/"
                            f"{tr.get('model_output_tokens', tr.get('output_tokens', 0))} out) "
                            f"| time=wall {tr.get('wall_time_s', 0.0):.2f}s "
                            f"model {tr.get('model_time_s', 0.0):.2f}s "
                            f"tool {tr.get('tool_time_s', 0.0):.2f}s"
                        )
                    if trials > 1 and res["trials"]:
                        avg_s = res.get("avg_score", 0.0)
                        avg_status = "PASS" if res.get("avg_passed", False) else "FAIL"
                        print(
                            f"  [{finished}/{total}] {tid} avg: {avg_s:.2f} {avg_status} "
                            f"| pass@1={res.get('pass_at_1', 0.0):.2f} "
                            f"pass^{trials}={res.get('pass_hat_k', 0.0):.2f}"
                        )

                # Print progress bar
                elapsed = time.monotonic() - start_time
                pct = finished * 100 // total
                if finished < total:
                    eta = elapsed / finished * (total - finished)
                    eta_str = f" | ETA ~{_fmt_duration(eta)}"
                else:
                    eta_str = ""
                avg_score = score_sum / finished_tasks if finished_tasks else 0.0
                print(
                    f"  [Progress] {finished}/{total} done ({pct}%) "
                    f"| avg {avg_score:.2f} "
                    f"pass^{trials} {n_pass_hat}/{finished_tasks} "
                    f"pass@{trials} {n_pass_at}/{finished_tasks} "
                    f"| elapsed {_fmt_duration(elapsed)}{eta_str}"
                )

                # Submit next task if any
                if task_queue and available_slots:
                    _submit(task_queue.pop(0))

                break  # restart as_completed loop with updated pending

    # --- Merge with previous results if rerunning errors ---
    if prev_results is not None:
        rerun_by_id = {r["task_id"]: r for r in results}
        still_errored = sum(1 for r in results if r.get("error"))
        fixed = len(results) - still_errored
        print(f"\n[rerun-errors] {fixed}/{len(results)} previously errored tasks now succeeded"
              f" ({still_errored} still errored)")

        # Merge: replace errored entries in prev_results with new results
        merged = []
        for prev in prev_results:
            if prev["task_id"] in rerun_by_id:
                merged.append(rerun_by_id[prev["task_id"]])
            else:
                merged.append(prev)
        results = merged
        total = len(results)

    # --- Merge with previously completed results if continuing ---
    # Re-scan all JSONL traces to build authoritative results (avoids
    # stale / partial data from the in-memory `results` list, which only
    # contains tasks that were re-run in *this* invocation).
    if continue_dir:
        all_from_traces = _load_completed_results(Path(continue_dir))
        if all_from_traces:
            results = all_from_traces
            total = len(results)
            print(f"\n[continue] Rebuilt results from {total} task(s) in trace directory")

    # --- Summary ---
    print(f"\n{'='*60}")
    if prev_results is not None:
        print(f"BATCH COMPLETE (rerun-errors merge) — {total} tasks")
    elif continue_dir:
        print(f"BATCH COMPLETE (continue merge) — {total} tasks")
    else:
        print(f"BATCH COMPLETE — {total} tasks, {workers} workers")
    print(f"{'='*60}\n")

    errored = sum(1 for r in results if r.get("error"))
    avg_score_final = score_sum / finished_tasks if finished_tasks else 0.0
    total_model_input_tokens = sum(
        tr.get("model_input_tokens", tr.get("input_tokens", 0))
        for r in results for tr in r.get("trials", [])
    )
    total_model_output_tokens = sum(
        tr.get("model_output_tokens", tr.get("output_tokens", 0))
        for r in results for tr in r.get("trials", [])
    )
    total_tokens = sum(tr.get("tokens", 0) for r in results for tr in r.get("trials", []))
    total_model_time_s = sum(tr.get("model_time_s", 0.0) for r in results for tr in r.get("trials", []))
    total_tool_time_s = sum(tr.get("tool_time_s", 0.0) for r in results for tr in r.get("trials", []))
    total_other_time_s = sum(tr.get("other_time_s", 0.0) for r in results for tr in r.get("trials", []))
    total_wall_time_s = sum(tr.get("wall_time_s", 0.0) for r in results for tr in r.get("trials", []))

    print(f"  Avg score: {avg_score_final:.3f}")
    print(f"  pass^{trials}: {n_pass_hat}/{finished_tasks}")
    print(f"  pass@{trials}: {n_pass_at}/{finished_tasks}")
    print(f"  Errored: {errored}/{finished_tasks}")
    print(
        f"  Total model tokens: {total_tokens} "
        f"({total_model_input_tokens} in / {total_model_output_tokens} out)"
    )
    print(
        f"  Total time: wall={total_wall_time_s:.2f}s "
        f"model={total_model_time_s:.2f}s tool={total_tool_time_s:.2f}s "
        f"other={total_other_time_s:.2f}s"
    )

    print(f"\n{'─'*60}")
    # Sort by task_id for readability
    for r in sorted(results, key=lambda x: x.get("task_id", "")):
        tid = r.get("task_id", "?")
        if r.get("error"):
            print(f"  {tid:40s}  ERROR: {r['error'][:50]}")
        elif r["trials"]:
            valid_trials = [t for t in r["trials"] if not t.get("error")]
            if not valid_trials:
                tr = r["trials"][0]
                print(f"  {tid:40s}  0.00  ERR   {tr.get('error', 'unknown')[:60]}")
            elif len(valid_trials) == 1:
                # Single trial: show as before
                tr = valid_trials[0]
                status = "PASS" if tr["passed"] else "FAIL"
                print(f"  {tid:40s}  {tr['task_score']:.2f}  {status}  "
                      f"C={tr['completion']:.2f} R={tr['robustness']:.2f} "
                      f"M={tr['communication']:.2f} S={tr['safety']:.0f} "
                      f"TOK={tr.get('tokens', 0)} "
                      f"({tr.get('model_input_tokens', tr.get('input_tokens', 0))}in/"
                      f"{tr.get('model_output_tokens', tr.get('output_tokens', 0))}out) "
                      f"TIME=wall {tr.get('wall_time_s', 0.0):.2f}s "
                      f"model {tr.get('model_time_s', 0.0):.2f}s "
                      f"tool {tr.get('tool_time_s', 0.0):.2f}s")
            else:
                # Multi-trial: show avg score + per-trial scores + pass^k/pass@k
                tl = r["trials"]
                avg_sc = sum(tr["task_score"] for tr in tl) / len(tl)
                trial_strs = "/".join(f"{t['task_score']:.2f}" for t in tl)
                p_hat = "Y" if all(tr["passed"] for tr in tl) else "N"
                p_at = "Y" if any(tr["passed"] for tr in tl) else "N"
                total_tok = sum(t.get("tokens", 0) for t in tl)
                total_in = sum(t.get("model_input_tokens", t.get("input_tokens", 0)) for t in tl)
                total_out = sum(t.get("model_output_tokens", t.get("output_tokens", 0)) for t in tl)
                total_wall = sum(t.get("wall_time_s", 0.0) for t in tl)
                total_model = sum(t.get("model_time_s", 0.0) for t in tl)
                total_tool = sum(t.get("tool_time_s", 0.0) for t in tl)
                print(f"  {tid:40s}  {avg_sc:.2f}  "
                      f"trials=[{trial_strs}] "
                      f"pass^{len(tl)}={p_hat} pass@{len(tl)}={p_at} "
                      f"TOK={total_tok} ({total_in}in/{total_out}out) "
                      f"TIME=wall {total_wall:.2f}s "
                      f"model {total_model:.2f}s "
                      f"tool {total_tool:.2f}s")

    # Write JSON results into the same trace subdir
    out_dir = Path(batch_trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "batch_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    summary_file = out_dir / "batch_summary.json"
    summary_data = {
        "tasks": total,
        "trials_per_task": trials,
        f"pass_hat_{trials}": n_pass_hat,
        f"pass_at_{trials}": n_pass_at,
        "errored": errored,
        "avg_score": avg_score_final,
        "total_model_input_tokens": total_model_input_tokens,
        "total_model_output_tokens": total_model_output_tokens,
        "total_input_tokens": total_model_input_tokens,
        "total_output_tokens": total_model_output_tokens,
        "total_tokens": total_tokens,
        "total_model_time_s": total_model_time_s,
        "total_tool_time_s": total_tool_time_s,
        "total_other_time_s": total_other_time_s,
        "total_wall_time_s": total_wall_time_s,
    }
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {results_file}")
    print(f"  Summary saved to {summary_file}")


def cmd_cleanup(args: argparse.Namespace) -> None:
    """Remove all claw-eval Docker containers."""
    from .config import load_config

    cfg = load_config(getattr(args, "config", None))

    from .runner.sandbox_runner import SandboxRunner

    runner = SandboxRunner(cfg.sandbox, image=cfg.sandbox.image)
    count = runner.cleanup_all()
    if count:
        print(f"Removed {count} claw-eval container(s).")
    else:
        print("No claw-eval containers found.")


def cmd_list(args: argparse.Namespace) -> None:
    """List available tasks."""
    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.exists():
        print(f"Tasks directory not found: {tasks_dir}")
        return

    from .models.task import TaskDefinition

    for yaml_file in sorted(tasks_dir.glob("*/task.yaml")):
        try:
            task = TaskDefinition.from_yaml(yaml_file)
            print(f"  {task.task_id:6s}  {task.task_name:30s}  difficulty={task.difficulty}  category={task.category}")
        except Exception as e:
            print(f"  {yaml_file.parent.name}: error loading - {e}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="claw-eval", description="Claw evaluation framework")
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run agent on a task")
    p_run.add_argument("--task", required=True, help="Path to task dir or YAML (e.g. tasks/T01zh_email_triage)")
    p_run.add_argument("--model", default=None, help="Model ID (default: from config.yaml)")
    p_run.add_argument("--api-key", default=None, help="API key (default: from config.yaml / $OPENAI_API_KEY)")
    p_run.add_argument("--base-url", default=None, help="Base URL for OpenAI-compatible API")
    p_run.add_argument(
        "--provider-transport",
        choices=["openai-completions", "anthropic-messages"],
        default=None,
        help="Required model wire; must match config.model.provider_transport",
    )
    p_run.add_argument("--config", default=None, help="Path to config.yaml")
    p_run.add_argument("--trials", type=int, default=1, help="Number of trials")
    p_run.add_argument("--trace-dir", default=None, help="Output directory for traces")
    p_run.add_argument("--judge-model", default=None, help="Override judge model ID")
    p_run.add_argument("--no-judge", action="store_true", help="Disable LLM judge for communication scoring")
    p_run.add_argument("--port-offset", type=int, default=0, help="Offset for all service ports (enables parallel runs)")
    p_run.add_argument("--sandbox", action="store_true", help="Run inside a Docker sandbox container")
    p_run.add_argument("--sandbox-image", default=None, help="Override sandbox Docker image name")
    p_run.add_argument("--sandbox-tools", action="store_true", help="Inject sandbox tools (shell/file/browser) without Docker")
    p_run.add_argument(
        "--freeze-rollout-bundle",
        action="store_true",
        help=(
            "Freeze exact OpenClaw rollout/grader inputs and compute the score "
            "by replaying that immutable bundle"
        ),
    )
    p_run.add_argument("--proxy", default=None, help="HTTP proxy URL for model/judge API traffic (e.g. http://proxy:port)")
    p_run.add_argument(
        "--harness", default="claweval", choices=["claweval", "openclaw", "aorchestra", "codex", "claudecode"],
        help="Agent harness driving the rollout (default: claweval)",
    )

    # _run-inner (hidden — used inside sandbox containers)
    p_inner = sub.add_parser("_run-inner", help=argparse.SUPPRESS)
    p_inner.add_argument("--task", required=True)
    p_inner.add_argument("--model", default=None)
    p_inner.add_argument("--api-key", default=None)
    p_inner.add_argument("--base-url", default=None)
    p_inner.add_argument(
        "--provider-transport",
        choices=["openai-completions", "anthropic-messages"],
        default=None,
    )
    p_inner.add_argument("--config", default=None)
    p_inner.add_argument("--trace-dir", default=None)
    p_inner.add_argument("--sandbox-tools", action="store_true")
    p_inner.add_argument("--judge-model", default=None)
    p_inner.add_argument("--no-judge", action="store_true")
    p_inner.add_argument("--proxy", default=None)
    p_inner.add_argument(
        "--harness", default="claweval", choices=["claweval", "openclaw", "aorchestra", "codex", "claudecode"],
        help="Agent harness driving the rollout (default: claweval)",
    )

    # build-image
    p_build = sub.add_parser("build-image", help="Build the sandbox Docker image")
    p_build.add_argument("--image", default=None, help="Image name/tag (default: from config)")
    p_build.add_argument("--context", default=".", help="Docker build context path")
    p_build.add_argument("--dockerfile", default="Dockerfile.agent", help="Dockerfile name (default: Dockerfile.agent)")
    p_build.add_argument("--config", default=None, help="Path to config.yaml")

    # grade
    p_grade = sub.add_parser("grade", help="Grade an existing trace")
    p_grade.add_argument("--trace", required=True, help="Path to JSONL trace file")
    p_grade.add_argument("--task", required=True, help="Path to task dir or YAML (e.g. tasks/T01zh_email_triage)")
    p_grade.add_argument("--config", default=None, help="Path to config.yaml")
    p_grade.add_argument("--judge-model", default=None, help="Override judge model ID")
    p_grade.add_argument("--no-judge", action="store_true", help="Disable LLM judge for communication scoring")
    p_grade.add_argument("--proxy", default=None, help="HTTP proxy URL for judge API traffic")

    # grade-bundle
    p_grade_bundle = sub.add_parser(
        "grade-bundle", help="Grade an immutable native rollout bundle"
    )
    p_grade_bundle.add_argument(
        "--bundle", required=True, help="Path to bundle.json or its directory"
    )
    p_grade_bundle.add_argument("--config", default=None, help="Path to config.yaml")
    p_grade_bundle.add_argument("--judge-model", default=None, help="Override judge model ID")
    p_grade_bundle.add_argument("--no-judge", action="store_true", help="Disable LLM judge")
    p_grade_bundle.add_argument("--proxy", default=None, help="HTTP proxy URL for judge API traffic")

    # batch
    p_batch = sub.add_parser("batch", help="Run all tasks in parallel")
    p_batch.add_argument("--tasks-dir", default="tasks", help="Tasks directory")
    p_batch.add_argument("--filter", default=None, help="Only run tasks matching this substring (e.g. 'en_' or 'T01')")
    p_batch.add_argument("--tag", default=None, help="Only run tasks with this tag (e.g. 'multimodal', 'general')")
    p_batch.add_argument("--range", default=None, help="Only run tasks in numeric ID range (e.g. '1-104')")
    p_batch.add_argument(
        "--task-ids", default=None, dest="task_ids", metavar="ID1,ID2,...",
        help="Run an explicit comma-separated set of tasks, e.g. "
             "'T002_email_triage,T008_todo_management' or short form 'T002,T008'. "
             "Authoritative selection — mutually exclusive with --filter/--tag/--range. "
             "Errors if any requested ID does not exist.",
    )
    p_batch.add_argument("--parallel", type=int, default=4, help="Number of parallel workers (default: 4)")
    p_batch.add_argument("--model", default=None)
    p_batch.add_argument("--api-key", default=None)
    p_batch.add_argument("--base-url", default=None)
    p_batch.add_argument("--config", default=None, help="Path to config.yaml")
    p_batch.add_argument("--trials", type=int, default=1)
    p_batch.add_argument("--trace-dir", default=None, help="Output directory for traces")
    p_batch.add_argument("--judge-model", default=None)
    p_batch.add_argument("--no-judge", action="store_true")
    p_batch.add_argument("--proxy", default=None, help="HTTP proxy URL for model/judge API traffic")
    p_batch.add_argument("--port-base-offset", type=int, default=0, help="Base port offset to avoid conflicts when running multiple batch jobs (e.g. 400)")
    p_batch.add_argument("--sandbox", action="store_true", help="Run sandbox tools inside Docker containers")
    p_batch.add_argument("--sandbox-image", default=None, help="Override sandbox Docker image name")
    p_batch.add_argument("--sandbox-tools", action="store_true", help="Inject sandbox tools (shell/file/browser) without Docker")
    p_batch.add_argument(
        "--harness", default="claweval", choices=["claweval", "openclaw", "aorchestra", "codex", "claudecode"],
        help="Agent harness driving the rollout (default: claweval)",
    )
    p_batch.add_argument("--rerun-errors", default=None, metavar="TRACE_DIR",
                         help="Re-run only errored tasks from a previous batch run. "
                              "Reads batch_results.json from TRACE_DIR, re-runs errored tasks, "
                              "and merges results back into the same directory.")
    p_batch.add_argument("--continue", dest="continue_dir", default=None, metavar="TRACE_DIR",
                         help="Continue a previous batch run from TRACE_DIR. "
                              "Scans existing trace files for grading_result events, "
                              "skips tasks with enough completed trials, and only runs the rest. "
                              "Results are merged into the same directory.")

    # cleanup
    p_cleanup = sub.add_parser("cleanup", help="Remove all claw-eval Docker containers")
    p_cleanup.add_argument("--config", default=None, help="Path to config.yaml")

    # list
    p_list = sub.add_parser("list", help="List available tasks")
    p_list.add_argument("--tasks-dir", default="tasks", help="Tasks directory")

    args = parser.parse_args(argv)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "_run-inner":
        cmd_run_inner(args)
    elif args.command == "build-image":
        cmd_build_image(args)
    elif args.command == "grade":
        cmd_grade(args)
    elif args.command == "grade-bundle":
        cmd_grade_bundle(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "cleanup":
        cmd_cleanup(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
