"""Harness Protocol, HarnessResult, HarnessFeature.

Phase 3 §3.2 — see docs/harness_design.md.

A Harness is the abstract layer that drives one task rollout from prompt to
trace JSONL + env_snapshot + audit_data. claw-eval's own ``run_task`` loop is
one such harness (see ``claweval.py``); external CLI agents (OpenClaw, Codex,
Claude Code) get their own implementations in later waves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from ..config import Config
    from ..models.task import TaskDefinition
    from ..runner.sandbox_runner import ContainerHandle
    from ..runner.services import ServiceManager
    from ..runner.user_agent import UserAgent


# HarnessFeature describes "which task.yaml fields are honoured by this harness".
# This is task-compatibility semantics, not symmetric capability comparison —
# a value missing from ``supported_features`` means the matching task.yaml field
# is either ignored or rejected when running under this harness.
HarnessFeature = Literal[
    "http_services",      # task.services / tool_endpoints HTTP mock tasks can run
    "sandbox_tools",      # task can use claw-eval's built-in SANDBOX_TOOLS
    "user_agent",         # task.user_agent.enabled=true tasks can run
    "compact",            # task.environment.enable_compact is honoured
    "max_turns_strict",   # task.environment.max_turns is a hard cap (else advisory)
]


@dataclass
class HarnessResult:
    """The output contract every harness must satisfy.

    ``trace_path`` is the only mandatory artifact. ``env_snapshot`` /
    ``audit_data`` are optional inputs the host CLI may pass into the grader;
    harnesses that don't produce them leave them empty / None. ``raw_dir`` is
    a per-harness scratch area (session.jsonl / stdout / proxy logs) kept for
    debugging but never fed to the grader.
    """

    trace_path: Path                  # standard claw-eval JSONL — feed to load_trace()
    env_snapshot: dict | None         # workspace snapshot, if any
    audit_data: dict[str, dict]       # mock service /audit data, keyed by service name
    raw_dir: Path | None              # harness-private debug area
    status: Literal["ok", "error", "timeout"] = "ok"
    error_message: str | None = None


class Harness(Protocol):
    """Abstract harness contract.

    Implementations live as plain classes; the Protocol is only used for type
    hints. The ``services_ctx`` argument is the host CLI's
    ``runner.services.ServiceManager`` — declared as a forward-ref string to
    avoid an import cycle between this module and ``runner.services``.
    """

    name: str
    supported_features: frozenset[HarnessFeature]

    def preflight(self, task: "TaskDefinition") -> list[str]:
        """Check whether ``task`` can run on this harness.

        Returns a list of blocking error messages. An empty list means the
        task is admissible. Caller (CLI) prints them to stderr and aborts.
        """
        ...

    def run(
        self,
        task: "TaskDefinition",
        *,
        trace_dir: Path,
        run_id: str,
        cfg: "Config",
        sandbox_handle: "ContainerHandle | None",
        user_agent: "UserAgent | None",
        services_ctx: "ServiceManager | None",
        sandbox_tools: bool = False,
    ) -> HarnessResult:
        """Drive one rollout of ``task`` and return its standardised result.

        ``sandbox_tools`` is a wave-1 carve-out for claw-eval's
        ``--sandbox-tools`` flag (use SANDBOX_TOOLS locally without a docker
        container). It is independent of ``sandbox_handle``: a docker run sets
        both, a local sandbox-tools run sets only ``sandbox_tools=True``, a
        plain run sets neither. External harnesses can ignore the flag.
        """
        ...
