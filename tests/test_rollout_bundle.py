from __future__ import annotations

import json
from pathlib import Path

import pytest

from claw_eval.rollout_bundle import (
    RolloutBundleError,
    grade_native_rollout,
    load_native_rollout_bundle,
    write_native_rollout_bundle,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    task = tmp_path / "source/tasks/T001"
    task.mkdir(parents=True)
    (task / "task.yaml").write_text(
        """task_id: T001
task_name: Test
difficulty: easy
category: test
prompt:
  text: Do the task
tools: []
services: []
environment:
  max_turns: 2
""",
        encoding="utf-8",
    )
    (task / "grader.py").write_text(
        """from claw_eval.graders.base import AbstractGrader
from claw_eval.models.trace import DimensionScores

class FixtureGrader(AbstractGrader):
    def grade(self, messages, dispatches, task, audit_data=None, judge=None, env_snapshot=None):
        del messages, dispatches, task, audit_data, judge
        completion = 1.0 if env_snapshot.get('cmd:check', {}).get('stdout') == 'ok' else 0.0
        return DimensionScores(completion=completion, robustness=1.0, communication=1.0, safety=1.0)
""",
        encoding="utf-8",
    )
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_start",
                        "trace_id": "trace-1",
                        "task_id": "T001",
                        "model": "model",
                        "harness": "openclaw",
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "trace_id": "trace-1",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "done"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "trace_end",
                        "trace_id": "trace-1",
                        "total_turns": 1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    snapshot = {"cmd:check": {"stdout": "ok", "exit_code": 0}}
    return task / "task.yaml", trace, snapshot


def test_bundle_round_trip_replays_exact_snapshot_grade(tmp_path: Path) -> None:
    task_yaml, trace, snapshot = _fixture(tmp_path)
    manifest = write_native_rollout_bundle(
        bundle_root=tmp_path / "bundle",
        task_yaml=task_yaml,
        trace_path=trace,
        env_snapshot=snapshot,
        raw_dir=None,
        harness="openclaw",
        model="model",
        rollout_status="ok",
    )

    loaded = load_native_rollout_bundle(manifest)
    grade = grade_native_rollout(manifest)

    assert loaded.env_snapshot == snapshot
    assert grade.task_score == 1.0
    assert grade.passed is True
    assert grade.judge_calls == []


def test_bundle_rejects_member_drift(tmp_path: Path) -> None:
    task_yaml, trace, snapshot = _fixture(tmp_path)
    manifest = write_native_rollout_bundle(
        bundle_root=tmp_path / "bundle",
        task_yaml=task_yaml,
        trace_path=trace,
        env_snapshot=snapshot,
        raw_dir=None,
        harness="openclaw",
        model="model",
        rollout_status="ok",
    )
    (manifest.parent / "native/env_snapshot.json").write_text("{}\n")
    with pytest.raises(RolloutBundleError, match="size mismatch|SHA-256 mismatch"):
        load_native_rollout_bundle(manifest)


@pytest.mark.parametrize("field", ["task_yaml", "trace", "env_snapshot"])
def test_bundle_rejects_uninventoried_replay_inputs(
    tmp_path: Path, field: str
) -> None:
    task_yaml, trace, snapshot = _fixture(tmp_path)
    manifest_path = write_native_rollout_bundle(
        bundle_root=tmp_path / "bundle",
        task_yaml=task_yaml,
        trace_path=trace,
        env_snapshot=snapshot,
        raw_dir=None,
        harness="openclaw",
        model="model",
        rollout_status="ok",
    )
    replacement = manifest_path.parent / "replacement.json"
    replacement.write_text("{}\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = replacement.name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RolloutBundleError, match="member inventory"):
        load_native_rollout_bundle(manifest_path)


def test_bundle_retains_native_session_and_rich_sidecar(tmp_path: Path) -> None:
    task_yaml, trace, snapshot = _fixture(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "session.jsonl").write_text("session\n")
    (raw / "openclaw.trajectory.jsonl").write_text("trajectory\n")
    manifest = write_native_rollout_bundle(
        bundle_root=tmp_path / "bundle",
        task_yaml=task_yaml,
        trace_path=trace,
        env_snapshot=snapshot,
        raw_dir=raw,
        harness="openclaw",
        model="model",
        rollout_status="ok",
    )
    loaded = load_native_rollout_bundle(manifest)
    assert (loaded.root / "native/session.jsonl").read_text() == "session\n"
    assert (
        loaded.root / "native/openclaw.trajectory.jsonl"
    ).read_text() == "trajectory\n"


def test_bundle_freezes_and_replays_peer_grader_dependency(tmp_path: Path) -> None:
    task_yaml, trace, snapshot = _fixture(tmp_path)
    task_root = task_yaml.parent
    peer = task_root.parent / "T000_peer"
    peer.mkdir()
    (peer / "grader.py").write_text(
        """from claw_eval.graders.base import AbstractGrader
from claw_eval.models.trace import DimensionScores

class PeerGrader(AbstractGrader):
    def grade(self, messages, dispatches, task, audit_data=None, judge=None, env_snapshot=None):
        del messages, dispatches, task, audit_data, judge, env_snapshot
        return DimensionScores(completion=0.75, robustness=1.0, communication=1.0, safety=1.0)
""",
        encoding="utf-8",
    )
    (task_root / "grader.py").write_text(
        """from claw_eval.graders.base import load_peer_grader

PeerGrader = load_peer_grader("T000_peer")

class FixtureGrader(PeerGrader):
    pass
""",
        encoding="utf-8",
    )
    manifest = write_native_rollout_bundle(
        bundle_root=tmp_path / "bundle",
        task_yaml=task_yaml,
        trace_path=trace,
        env_snapshot=snapshot,
        raw_dir=None,
        harness="openclaw",
        model="model",
        rollout_status="ok",
    )
    source_peer = peer / "grader.py"
    source_peer.write_text("raise RuntimeError('mutable source must not be read')\n")

    loaded = load_native_rollout_bundle(manifest)
    grade = grade_native_rollout(manifest)

    assert "tasks/T000_peer/grader.py" in loaded.manifest["members"]
    assert grade.scores.completion == 0.75
