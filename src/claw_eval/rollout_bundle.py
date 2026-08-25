"""Immutable rollout bundles and grader replay for lifecycle decomposition."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


BUNDLE_SCHEMA_VERSION = 1


class RolloutBundleError(ValueError):
    """Raised when a rollout bundle is unsafe, incomplete, or has drifted."""


@dataclass(frozen=True)
class LoadedRolloutBundle:
    root: Path
    manifest: dict[str, Any]
    task_yaml: Path
    trace: Path
    env_snapshot: dict[str, Any]


@dataclass(frozen=True)
class NativeGrade:
    scores: Any
    task_score: float
    passed: bool
    judge_calls: list[dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(task_yaml: Path) -> dict[str, Any]:
    """Record the ClawEval checkout that supplied task and runtime semantics."""

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=task_yaml.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "repository": run("config", "--get", "remote.origin.url"),
            "commit": run("rev-parse", "HEAD"),
            "tree": run("rev-parse", "HEAD^{tree}"),
            "tracked_dirty": bool(
                run("status", "--porcelain", "--untracked-files=no")
            ),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "repository": None,
            "commit": None,
            "tree": None,
            "tracked_dirty": None,
        }


def _safe_relative(root: Path, value: str, *, label: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RolloutBundleError(f"unsafe {label} path: {value!r}")
    path = (root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RolloutBundleError(f"{label} escapes bundle: {value!r}") from exc
    return path


def _copy_member(
    *, source: Path, bundle_root: Path, relative: str, members: dict[str, Any]
) -> Path:
    if source.is_symlink() or not source.is_file():
        raise RolloutBundleError(f"bundle source is not a regular file: {source}")
    destination = _safe_relative(bundle_root, relative, label="bundle member")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    members[relative] = {
        "sha256": _sha256(destination),
        "size_bytes": destination.stat().st_size,
    }
    return destination


def _task_document(task_yaml: Path) -> dict[str, Any]:
    import yaml

    try:
        value = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RolloutBundleError(f"cannot load task YAML {task_yaml}: {exc}") from exc
    if not isinstance(value, dict) or not value.get("task_id"):
        raise RolloutBundleError(f"task YAML has no task_id: {task_yaml}")
    return value


def _copy_declared_grader_inputs(
    *, task_yaml: Path, task: dict[str, Any], bundle_root: Path, members: dict[str, Any]
) -> None:
    task_root = task_yaml.parent.resolve()
    grader = task_root / "grader.py"
    _copy_member(
        source=grader,
        bundle_root=bundle_root,
        relative="task/grader.py",
        members=members,
    )
    for key in ("local_grader_files", "sandbox_grader_files"):
        values = task.get(key) or []
        if not isinstance(values, list):
            raise RolloutBundleError(f"task.{key} must be a list")
        for raw in values:
            relative = PurePosixPath(str(raw))
            if relative.is_absolute() or ".." in relative.parts:
                raise RolloutBundleError(f"unsafe task.{key} path: {raw!r}")
            source = (task_root / Path(*relative.parts)).resolve()
            try:
                source.relative_to(task_root)
            except ValueError as exc:
                raise RolloutBundleError(
                    f"task.{key} path escapes task root: {raw!r}"
                ) from exc
            _copy_member(
                source=source,
                bundle_root=bundle_root,
                relative=f"task/{key}/{relative.as_posix()}",
                members=members,
            )
    _copy_peer_graders(
        grader=grader,
        tasks_root=task_root.parent,
        bundle_root=bundle_root,
        members=members,
    )


def _declared_peer_graders(grader: Path) -> list[str]:
    """Return literal ``load_peer_grader`` dependencies from a grader module."""

    try:
        tree = ast.parse(grader.read_text(encoding="utf-8"), filename=str(grader))
    except (OSError, SyntaxError) as exc:
        raise RolloutBundleError(
            f"cannot inspect grader dependencies {grader}: {exc}"
        ) from exc
    task_ids: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        function_name = (
            function.id
            if isinstance(function, ast.Name)
            else function.attr
            if isinstance(function, ast.Attribute)
            else None
        )
        if function_name != "load_peer_grader":
            continue
        if (
            not node.args
            or not isinstance(node.args[0], ast.Constant)
            or not isinstance(node.args[0].value, str)
        ):
            raise RolloutBundleError(
                f"grader has a non-literal load_peer_grader dependency: {grader}"
            )
        task_ids.add(node.args[0].value)
    return sorted(task_ids)


def _copy_peer_graders(
    *,
    grader: Path,
    tasks_root: Path,
    bundle_root: Path,
    members: dict[str, Any],
    seen: set[str] | None = None,
) -> None:
    """Freeze peer-grader source dependencies recursively."""

    seen = seen if seen is not None else set()
    for task_id in _declared_peer_graders(grader):
        if task_id in seen:
            continue
        seen.add(task_id)
        peer = tasks_root / task_id / "grader.py"
        destination = f"tasks/{task_id}/grader.py"
        _copy_member(
            source=peer,
            bundle_root=bundle_root,
            relative=destination,
            members=members,
        )
        _copy_peer_graders(
            grader=peer,
            tasks_root=tasks_root,
            bundle_root=bundle_root,
            members=members,
            seen=seen,
        )


def _copy_optional_native_artifacts(
    *, raw_dir: Path | None, bundle_root: Path, members: dict[str, Any]
) -> None:
    if raw_dir is None or not raw_dir.is_dir():
        return
    names = (
        "session.jsonl",
        "openclaw.trajectory.jsonl",
        "bridge_traffic.jsonl",
        "model_budget.json",
        "openclaw_invocation.json",
        "openclaw_preflight.json",
    )
    for name in names:
        candidates = sorted(raw_dir.rglob(name))
        if not candidates:
            continue
        if len(candidates) != 1:
            raise RolloutBundleError(
                f"expected at most one native {name}, observed {len(candidates)}"
            )
        _copy_member(
            source=candidates[0],
            bundle_root=bundle_root,
            relative=f"native/{name}",
            members=members,
        )


def write_native_rollout_bundle(
    *,
    bundle_root: Path,
    task_yaml: Path,
    trace_path: Path,
    env_snapshot: dict[str, Any],
    raw_dir: Path | None,
    harness: str,
    model: str,
    rollout_status: str,
) -> Path:
    """Freeze every current grader input before a grading call."""

    if bundle_root.exists():
        raise RolloutBundleError(f"rollout bundle already exists: {bundle_root}")
    bundle_root.mkdir(parents=True)
    members: dict[str, Any] = {}
    task = _task_document(task_yaml)
    _copy_member(
        source=task_yaml,
        bundle_root=bundle_root,
        relative="task/task.yaml",
        members=members,
    )
    _copy_declared_grader_inputs(
        task_yaml=task_yaml,
        task=task,
        bundle_root=bundle_root,
        members=members,
    )
    _copy_member(
        source=trace_path,
        bundle_root=bundle_root,
        relative="native/trace.jsonl",
        members=members,
    )
    snapshot_path = bundle_root / "native/env_snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(env_snapshot, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    members["native/env_snapshot.json"] = {
        "sha256": _sha256(snapshot_path),
        "size_bytes": snapshot_path.stat().st_size,
    }
    _copy_optional_native_artifacts(
        raw_dir=raw_dir,
        bundle_root=bundle_root,
        members=members,
    )
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "task_id": str(task["task_id"]),
        "harness": harness,
        "model": model,
        "rollout_status": rollout_status,
        "claw_eval_source": _source_identity(task_yaml),
        "task_yaml": "task/task.yaml",
        "trace": "native/trace.jsonl",
        "env_snapshot": "native/env_snapshot.json",
        "members": members,
    }
    manifest_path = bundle_root / "bundle.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_native_rollout_bundle(path: Path) -> LoadedRolloutBundle:
    """Load a bundle and fail if any member hash or size has drifted."""

    manifest_path = path if path.name == "bundle.json" else path / "bundle.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RolloutBundleError(
            f"cannot load rollout bundle {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RolloutBundleError("unsupported rollout bundle schema")
    members = manifest.get("members")
    if not isinstance(members, dict):
        raise RolloutBundleError("rollout bundle has no member inventory")
    root = manifest_path.parent.resolve()
    for relative, contract in members.items():
        if not isinstance(relative, str) or not isinstance(contract, dict):
            raise RolloutBundleError("invalid rollout bundle member contract")
        member = _safe_relative(root, relative, label="bundle member")
        if member.is_symlink() or not member.is_file():
            raise RolloutBundleError(f"bundle member is missing or unsafe: {relative}")
        if member.stat().st_size != contract.get("size_bytes"):
            raise RolloutBundleError(f"bundle member size mismatch: {relative}")
        if _sha256(member) != contract.get("sha256"):
            raise RolloutBundleError(f"bundle member SHA-256 mismatch: {relative}")
    task_yaml = _safe_relative(root, str(manifest["task_yaml"]), label="task YAML")
    trace = _safe_relative(root, str(manifest["trace"]), label="trace")
    snapshot = _safe_relative(
        root, str(manifest["env_snapshot"]), label="environment snapshot"
    )
    try:
        env_snapshot = json.loads(snapshot.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RolloutBundleError(
            f"cannot load exact environment snapshot: {exc}"
        ) from exc
    if not isinstance(env_snapshot, dict):
        raise RolloutBundleError("environment snapshot must be a JSON object")
    return LoadedRolloutBundle(
        root=root,
        manifest=manifest,
        task_yaml=task_yaml,
        trace=trace,
        env_snapshot=env_snapshot,
    )


def grade_native_rollout(
    path: Path,
    *,
    judge: Any | None = None,
) -> NativeGrade:
    """Run the authoritative task grader from a frozen rollout bundle."""

    from .graders.llm_judge import NoJudge
    from .graders.base import peer_grader_tasks_dir
    from .graders.registry import get_grader
    from .models.scoring import compute_task_score, is_pass
    from .models.task import TaskDefinition
    from .trace.reader import load_trace

    bundle = load_native_rollout_bundle(path)
    task = TaskDefinition.from_yaml(bundle.task_yaml)
    start, messages, dispatches, media_events, _end, audit_data = load_trace(
        bundle.trace
    )
    if start.task_id != task.task_id or bundle.manifest.get("task_id") != task.task_id:
        raise RolloutBundleError("bundle task identity mismatch")
    if start.harness != bundle.manifest.get("harness"):
        raise RolloutBundleError("bundle harness identity mismatch")
    with peer_grader_tasks_dir(bundle.root / "tasks"):
        grader = get_grader(
            task.task_id,
            tasks_dir=bundle.task_yaml.parent.parent,
            task_dir=bundle.task_yaml.parent,
        )
    effective_judge = judge if judge is not None else NoJudge()
    if hasattr(effective_judge, "reset_call_log"):
        effective_judge.reset_call_log()
    params = inspect.signature(grader.grade).parameters
    kwargs: dict[str, Any] = {
        "audit_data": audit_data,
        "judge": effective_judge,
    }
    if "media_events" in params:
        kwargs["media_events"] = media_events
    if "env_snapshot" in params:
        kwargs["env_snapshot"] = bundle.env_snapshot
    scores = grader.grade(messages, dispatches, task, **kwargs)
    judge_calls = (
        list(effective_judge.get_call_log())
        if hasattr(effective_judge, "get_call_log")
        else []
    )
    task_score = float(compute_task_score(scores))
    return NativeGrade(
        scores=scores,
        task_score=task_score,
        passed=bool(is_pass(task_score)),
        judge_calls=judge_calls,
    )
