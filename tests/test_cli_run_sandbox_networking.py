"""Regression tests for ``cmd_run`` OpenClaw sandbox networking arguments."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from claw_eval import cli
from claw_eval.models.task import TaskDefinition


class _StartContainerReached(RuntimeError):
    """Stop ``cmd_run`` after capturing the container launch contract."""


def _capture_start_kwargs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    offset: int,
    net: str | None,
) -> dict:
    applied_offsets: list[int] = []
    task = SimpleNamespace(
        task_id="T_test_sandbox_ports",
        services=[],
        environment=SimpleNamespace(mock_today=None),
        apply_port_offset=applied_offsets.append,
    )
    config = SimpleNamespace(
        model=SimpleNamespace(model_id="test-model"),
        defaults=SimpleNamespace(trace_dir=str(tmp_path / "default-traces")),
        sandbox=SimpleNamespace(enabled=True, image="test-sandbox-image"),
    )
    harness = SimpleNamespace(preflight=lambda _task: [])
    captured: dict = {}

    class FakeSandboxRunner:
        def __init__(self, sandbox_config, *, image=None):
            assert sandbox_config is config.sandbox
            assert image == "claw-eval-agent-openclaw:latest"

        def start_container(self, **kwargs):
            captured.update(kwargs)
            raise _StartContainerReached

    class FakeServiceManager:
        def __init__(self, services, *, mock_today=None):
            assert services is task.services
            assert mock_today is None

        def __enter__(self):
            return SimpleNamespace()

        def __exit__(self, exc_type, exc, traceback):
            return False

    task_yaml = tmp_path / "tasks" / task.task_id / "task.yaml"
    task_yaml.parent.mkdir(parents=True)
    task_yaml.write_text("task_id: ignored-by-fake\n", encoding="utf-8")

    monkeypatch.setattr("claw_eval.config.load_config", lambda _path: config)
    monkeypatch.setattr("claw_eval.harnesses.get_harness", lambda _name: harness)
    monkeypatch.setattr(
        TaskDefinition,
        "from_yaml",
        classmethod(lambda cls, _path: task),
    )
    monkeypatch.setattr(cli, "_make_judge", lambda _cfg, _args: None)
    monkeypatch.setattr(
        "claw_eval.runner.sandbox_runner.SandboxRunner",
        FakeSandboxRunner,
    )
    monkeypatch.setattr(
        "claw_eval.runner.services.ServiceManager",
        FakeServiceManager,
    )
    if net is None:
        monkeypatch.delenv("CLAWEVAL_SANDBOX_NET", raising=False)
    else:
        monkeypatch.setenv("CLAWEVAL_SANDBOX_NET", net)

    args = argparse.Namespace(
        api_key=None,
        base_url=None,
        config="unused-config.yaml",
        harness="openclaw",
        model=None,
        no_judge=True,
        port_offset=offset,
        proxy=None,
        sandbox=True,
        sandbox_image=None,
        task=str(task_yaml),
        trace_dir=str(tmp_path / "traces"),
        trials=1,
    )
    with pytest.raises(_StartContainerReached):
        cli.cmd_run(args)

    assert applied_offsets == ([offset] if offset else [])
    assert captured["run_id"] == f"{task.task_id}-trial0"

    volumes = captured["volumes"]
    assert len(volumes) == 1
    [(host_case_dir, container_case_dir)] = volumes.items()
    assert host_case_dir == container_case_dir
    assert Path(host_case_dir).is_dir()
    assert captured["extra_env"] == {
        "CLAWEVAL_BRIDGE_LOG": str(
            Path(container_case_dir) / "raw" / "bridge_traffic.jsonl"
        )
    }
    return captured


@pytest.mark.parametrize("offset", [0, 50, 350])
def test_cmd_run_openclaw_host_uses_offset_sandbox_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offset: int,
) -> None:
    kwargs = _capture_start_kwargs(
        tmp_path,
        monkeypatch,
        offset=offset,
        net=None,
    )
    assert kwargs["network_mode"] == "host"
    assert kwargs["sandbox_port"] == 8080 + offset


def test_cmd_run_openclaw_bridge_does_not_force_network_or_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _capture_start_kwargs(
        tmp_path,
        monkeypatch,
        offset=350,
        net="  BRIDGE  ",
    )
    assert "network_mode" not in kwargs
    assert "sandbox_port" not in kwargs
