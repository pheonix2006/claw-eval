"""Unit tests for the OpenClaw container-mode agent command builder.

``_build_agent_cmd`` is the pure command-construction slice of
``run_in_container`` — extracted so we can assert flag wiring (notably the
multi-turn ``--session-key``) without standing up a Docker container.
"""

from __future__ import annotations

from claw_eval.harnesses._openclaw_container import _build_agent_cmd


def test_cmd_basic_shape() -> None:
    cmd = _build_agent_cmd(prompt="hello", agent_id="main", timeout_s=600)
    assert cmd[:7] == [
        "openclaw", "--no-color", "--log-level", "silent",
        "agent", "--local", "--json",
    ]
    assert cmd[cmd.index("--message") + 1] == "hello"
    assert cmd[cmd.index("--agent") + 1] == "main"
    assert cmd[cmd.index("--timeout") + 1] == "600"


def test_cmd_omits_session_key_by_default() -> None:
    """Single-turn path must be byte-identical to today: no --session-key."""
    cmd = _build_agent_cmd(prompt="hi", agent_id="main", timeout_s=600)
    assert "--session-key" not in cmd


def test_cmd_threads_session_key_when_given() -> None:
    """Multi-turn: a session_key is passed through so consecutive calls
    share OpenClaw's persisted session context."""
    cmd = _build_agent_cmd(
        prompt="round 2",
        agent_id="main",
        timeout_s=600,
        session_key="agent:main:claweval-run42",
    )
    assert cmd[cmd.index("--session-key") + 1] == "agent:main:claweval-run42"
    # message/agent still correct
    assert cmd[cmd.index("--message") + 1] == "round 2"
    assert cmd[cmd.index("--agent") + 1] == "main"


def test_cmd_no_timeout_when_nonpositive() -> None:
    cmd = _build_agent_cmd(prompt="x", agent_id="main", timeout_s=0)
    assert "--timeout" not in cmd
