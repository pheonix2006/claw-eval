"""Tests for the OpenClaw → claw-eval trace translator.

Phase 3 Wave 2 (§6.4). Fixtures live under ``tests/fixtures/openclaw/``:

- ``executionTrace_sample.json`` — hand-constructed (see the Wave-2 report).
  Real OpenClaw runs from Workspace-Bench all aborted on a context-overflow
  400 before emitting any toolCall, so we synthesise a minimal session that
  exercises the assistant / toolCall / toolResult interleaving plus an
  error-injection turn.
- ``bridge_log_sample.jsonl`` — hand-constructed to share ``callID`` with
  the executionTrace; one HTTP-200 record and one HTTP-500 to drive the
  ``is_error`` rules.

These tests must be runnable from a clean checkout (no docker, no OpenClaw
install). They feed the adapter with fixtures, then re-load the produced
JSONL through the real ``load_trace`` to prove the byte representation is
contract-compliant.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claw_eval.harnesses._trace_adapter import translate_openclaw
from claw_eval.models.content import TextBlock, ToolResultBlock, ToolUseBlock
from claw_eval.models.task import (
    Environment,
    Prompt,
    TaskDefinition,
)
from claw_eval.models.tool import ToolSpec
from claw_eval.models.trace import (
    AuditSnapshot,
    DimensionScores,
    MediaLoad,
    ToolDispatch,
    TraceMessage,
)
from claw_eval.trace.reader import load_trace

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "openclaw"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_execution_trace() -> list[dict]:
    with open(FIXTURES_DIR / "executionTrace_sample.json") as fh:
        return json.load(fh)


@pytest.fixture
def execution_trace() -> list[dict]:
    return _load_execution_trace()


@pytest.fixture
def usage_total() -> dict:
    # Matches the per-turn usage in the fixture (320+22 + 410+8 + 480+36).
    return {"prompt_tokens": 1210, "completion_tokens": 66, "total_tokens": 1276}


@pytest.fixture
def llm_meta() -> dict:
    return {
        "provider": "openai",
        "baseUrl": "http://localhost:8100/v1",
        "model": "qwen3-32b",
    }


@pytest.fixture
def bridge_log_path() -> Path:
    return FIXTURES_DIR / "bridge_log_sample.jsonl"


@pytest.fixture
def task() -> TaskDefinition:
    return TaskDefinition(
        task_id="T_test_adapter_001",
        task_name="adapter unit test stub",
        prompt=Prompt(text="dummy", language="zh"),
        environment=Environment(timeout_seconds=300, max_turns=20),
    )


@pytest.fixture
def audit_data() -> dict[str, dict]:
    # Mirrors the shape ServiceManager.collect_audit() returns.
    return {
        "officeqa": {
            "calls": [
                {
                    "endpoint": "/officeqa/query_spending",
                    "request_body": {"department": "sales", "period": "last_month"},
                    "response_body": {"department": "sales", "total": 128500.5},
                    "status": 200,
                },
                {
                    "endpoint": "/officeqa/query_spending",
                    "request_body": {"department": "purchasing", "period": "last_month"},
                    "response_body": {"error": "internal_server_error"},
                    "status": 500,
                },
            ]
        }
    }


# ---------------------------------------------------------------------------
# 1) Basic translation produces the expected event mix
# ---------------------------------------------------------------------------


def test_translate_basic(
    tmp_path,
    execution_trace,
    usage_total,
    llm_meta,
    bridge_log_path,
    audit_data,
    task,
):
    trace_path = translate_openclaw(
        execution_trace=execution_trace,
        usage_total=usage_total,
        llm_meta=llm_meta,
        bridge_log_path=bridge_log_path,
        audit_data=audit_data,
        task=task,
        run_id="run0",
        trace_dir=tmp_path,
        duration_ms=6_120,
        status="ok",
    )
    assert trace_path.exists()
    assert trace_path.name == "T_test_adapter_001_run0.jsonl"

    # Read raw lines and bucket by ``type`` so we don't depend on the reader.
    with trace_path.open() as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    types = [ev["type"] for ev in lines]

    # TraceStart at the head, TraceEnd at the tail.
    assert types[0] == "trace_start"
    assert types[-1] == "trace_end"

    # Header fields on TraceStart.
    head = lines[0]
    assert head["task_id"] == "T_test_adapter_001"
    assert head["harness"] == "openclaw"
    assert head["model"] == "qwen3-32b"

    # Expected counts based on the fixture:
    # - 2 user text messages + 3 assistant text messages + 2 tool results
    #   = 7 message events
    # - 2 ToolDispatch events
    # - 1 AuditSnapshot event (officeqa)
    assert sum(1 for t in types if t == "message") == 7
    assert sum(1 for t in types if t == "tool_dispatch") == 2
    assert sum(1 for t in types if t == "audit_snapshot") == 1

    # Sanity-check TraceEnd token rollup.
    end = lines[-1]
    assert end["input_tokens"] == 1210
    assert end["output_tokens"] == 66
    assert end["total_tokens"] == 1276
    assert end["total_turns"] == 3  # 3 assistant messages
    assert end["wall_time_s"] == pytest.approx(6.12, abs=1e-3)
    assert end["failure_modes"] == []


# ---------------------------------------------------------------------------
# 2) Round-trip through load_trace
# ---------------------------------------------------------------------------


def test_load_trace_roundtrip(
    tmp_path,
    execution_trace,
    usage_total,
    llm_meta,
    bridge_log_path,
    audit_data,
    task,
):
    trace_path = translate_openclaw(
        execution_trace=execution_trace,
        usage_total=usage_total,
        llm_meta=llm_meta,
        bridge_log_path=bridge_log_path,
        audit_data=audit_data,
        task=task,
        run_id="run0",
        trace_dir=tmp_path,
        duration_ms=6_120,
        status="ok",
    )

    start, messages, dispatches, media_events, end, audit_out = load_trace(trace_path)

    assert start.harness == "openclaw"
    assert start.task_id == task.task_id
    assert start.model == "qwen3-32b"

    # 7 messages, 2 dispatches, 0 media events, audit by service.
    assert len(messages) == 7
    assert len(dispatches) == 2
    assert media_events == []
    assert end is not None
    assert end.total_turns == 3
    assert end.input_tokens == 1210
    assert end.output_tokens == 66
    assert "officeqa" in audit_out
    assert audit_out["officeqa"]["calls"][0]["status"] == 200

    # First assistant message should hold a ToolUseBlock for call_sales_001.
    first_assistant = next(m for m in messages if m.message.role == "assistant")
    tool_uses = [b for b in first_assistant.message.content if isinstance(b, ToolUseBlock)]
    assert len(tool_uses) == 1
    assert tool_uses[0].id == "call_sales_001"
    assert tool_uses[0].name == "officeqa_query_spending"
    assert tool_uses[0].input == {"department": "sales", "period": "last_month"}

    # Each tool result is a user-role message containing one ToolResultBlock.
    tool_results = [
        b
        for m in messages
        if m.message.role == "user"
        for b in m.message.content
        if isinstance(b, ToolResultBlock)
    ]
    assert {tr.tool_use_id for tr in tool_results} == {
        "call_sales_001",
        "call_purch_002",
    }


def test_openclaw_core_exec_is_recorded_as_successful_sandbox_bash(tmp_path) -> None:
    task = TaskDefinition(
        task_id="T_exec",
        task_name="exec fallback",
        prompt=Prompt(text="run"),
        environment=Environment(timeout_seconds=30, max_turns=3),
        tools=[ToolSpec(name="Bash", description="shell", input_schema={"type": "object"})],
    )
    events = [
        {"type": "text", "role": "assistant", "content": ""},
        {"type": "tool", "role": "tool", "tool": "exec", "callID": "c1",
         "input": {"command": "true"}, "output": {"text": ""},
         "exitCode": 0, "status": "completed", "durationMs": 4},
    ]
    path = translate_openclaw(
        execution_trace=events, usage_total={}, llm_meta={"model": "m"},
        bridge_log_path=None, audit_data={}, task=task, run_id="r",
        trace_dir=tmp_path, duration_ms=4, status="ok",
    )
    _, messages, dispatches, _, _, _ = load_trace(path)
    use = next(block for msg in messages for block in msg.message.content
               if isinstance(block, ToolUseBlock))
    result = next(block for msg in messages for block in msg.message.content
                  if isinstance(block, ToolResultBlock))
    assert use.name == "Bash"
    assert result.is_error is False
    assert dispatches[0].tool_name == "Bash"
    assert dispatches[0].response_status == 200

# ---------------------------------------------------------------------------
# 3) Grader can consume the translated trace
# ---------------------------------------------------------------------------


class _DummyGrader:
    """Minimal AbstractGrader-shaped consumer.

    We don't subclass AbstractGrader (which is abstract); we just mimic the
    signature to demonstrate the contract is followed. The point is to
    exercise every accessor pattern that real graders use against the data
    returned by load_trace.
    """

    def grade(
        self,
        messages: list[TraceMessage],
        dispatches: list[ToolDispatch],
        task: TaskDefinition,
        audit_data: dict[str, dict] | None = None,
        judge=None,
        media_events: list[MediaLoad] | None = None,
        env_snapshot: dict | None = None,
    ) -> DimensionScores:
        # final assistant text
        final = ""
        for msg in reversed(messages):
            if msg.message.role == "assistant":
                final = msg.message.text
                break
        # robustness — touch ToolDispatch.response_status
        errors = sum(1 for d in dispatches if d.response_status >= 400)
        completion = 1.0 if final else 0.0
        robustness = 0.5 if errors else 1.0
        # touch audit_data so missing keys would crash here, not later
        if audit_data:
            assert isinstance(audit_data, dict)
            _ = audit_data.get("officeqa", {}).get("calls", [])
        return DimensionScores(
            completion=completion,
            robustness=robustness,
            safety=1.0,
            efficiency_turns=sum(1 for m in messages if m.message.role == "assistant"),
            efficiency_tokens=sum(d.latency_ms == 0 for d in dispatches),
        )


def test_grader_can_consume(
    tmp_path,
    execution_trace,
    usage_total,
    llm_meta,
    bridge_log_path,
    audit_data,
    task,
):
    trace_path = translate_openclaw(
        execution_trace=execution_trace,
        usage_total=usage_total,
        llm_meta=llm_meta,
        bridge_log_path=bridge_log_path,
        audit_data=audit_data,
        task=task,
        run_id="run0",
        trace_dir=tmp_path,
        duration_ms=6_120,
        status="ok",
    )
    _, messages, dispatches, media_events, _, audit_out = load_trace(trace_path)

    grader = _DummyGrader()
    scores = grader.grade(
        messages=messages,
        dispatches=dispatches,
        task=task,
        audit_data=audit_out,
        media_events=media_events,
    )

    assert isinstance(scores, DimensionScores)
    assert scores.completion == 1.0  # final assistant text present
    assert scores.robustness == 0.5  # one HTTP 500 detected
    assert scores.efficiency_turns == 3


# ---------------------------------------------------------------------------
# 4) ``is_error`` rules — bridge status >= 400, == 200, missing record
# ---------------------------------------------------------------------------


def test_is_error_rules(
    tmp_path,
    execution_trace,
    usage_total,
    llm_meta,
    audit_data,
    task,
):
    # Augment fixture: rewrite one bridge record to drop the second toolCall
    # entirely (forcing the degraded path).
    real_bridge = json.loads(
        (FIXTURES_DIR / "bridge_log_sample.jsonl").read_text().splitlines()[0]
    )
    only_sales = tmp_path / "bridge_sales_only.jsonl"
    with only_sales.open("w") as fh:
        fh.write(json.dumps(real_bridge) + "\n")

    # ---- Case A: full bridge — call_sales (200) clean, call_purch (500) error.
    trace_a = translate_openclaw(
        execution_trace=execution_trace,
        usage_total=usage_total,
        llm_meta=llm_meta,
        bridge_log_path=FIXTURES_DIR / "bridge_log_sample.jsonl",
        audit_data=audit_data,
        task=task,
        run_id="rule_a",
        trace_dir=tmp_path,
        duration_ms=1000,
        status="ok",
    )
    _, msgs_a, disp_a, _, _, _ = load_trace(trace_a)
    tr_by_id_a = {
        b.tool_use_id: b
        for m in msgs_a
        if m.message.role == "user"
        for b in m.message.content
        if isinstance(b, ToolResultBlock)
    }
    assert tr_by_id_a["call_sales_001"].is_error is False, "200 must NOT be error"
    assert tr_by_id_a["call_purch_002"].is_error is True, "500 MUST be error"

    # ToolDispatch should carry the bridge HTTP status verbatim.
    disp_by_id_a = {d.tool_use_id: d for d in disp_a}
    assert disp_by_id_a["call_sales_001"].response_status == 200
    assert disp_by_id_a["call_purch_002"].response_status == 500

    # ---- Case B: degraded — bridge has no record for call_purch_002.
    trace_b = translate_openclaw(
        execution_trace=execution_trace,
        usage_total=usage_total,
        llm_meta=llm_meta,
        bridge_log_path=only_sales,
        audit_data=audit_data,
        task=task,
        run_id="rule_b",
        trace_dir=tmp_path,
        duration_ms=1000,
        status="ok",
    )
    _, msgs_b, disp_b, _, _, _ = load_trace(trace_b)
    tr_by_id_b = {
        b.tool_use_id: b
        for m in msgs_b
        if m.message.role == "user"
        for b in m.message.content
        if isinstance(b, ToolResultBlock)
    }
    # call_sales still 200 → not an error.
    assert tr_by_id_b["call_sales_001"].is_error is False
    # call_purch missing from bridge → forced error.
    assert tr_by_id_b["call_purch_002"].is_error is True

    disp_by_id_b = {d.tool_use_id: d for d in disp_b}
    assert disp_by_id_b["call_purch_002"].response_status == 500
    assert disp_by_id_b["call_purch_002"].endpoint_url == ""  # placeholder


# ---------------------------------------------------------------------------
# 5) Empty bridge log (pure-text task, no tools) still produces a valid trace
# ---------------------------------------------------------------------------


def test_empty_bridge_log(tmp_path, llm_meta, task):
    # A pure-text trace: one user prompt, one assistant reply. No tool events.
    exec_trace = [
        {
            "type": "text",
            "role": "user",
            "content": "用一句话解释什么是复利。",
            "timestamp": "2026-06-20T10:00:00.000Z",
        },
        {
            "type": "text",
            "role": "assistant",
            "content": "复利就是利息再产生利息,雪球越滚越大。",
            "timestamp": "2026-06-20T10:00:02.000Z",
            "turn": 1,
            "llm": {
                "provider": "openai",
                "baseUrl": "http://localhost:8100/v1",
                "model": "qwen3-32b",
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 18,
                    "total_tokens": 68,
                    "cache_read": 0,
                    "cache_write": 0,
                },
                "stopReason": "stop",
                "errorMessage": None,
            },
        },
    ]
    trace_path = translate_openclaw(
        execution_trace=exec_trace,
        usage_total={"prompt_tokens": 50, "completion_tokens": 18, "total_tokens": 68},
        llm_meta=llm_meta,
        bridge_log_path=None,  # ← the empty-tool-set case
        audit_data={},  # ← no mock services
        task=task,
        run_id="empty",
        trace_dir=tmp_path,
        duration_ms=2_000,
        status="ok",
    )

    start, messages, dispatches, media_events, end, audit_out = load_trace(trace_path)
    assert start.harness == "openclaw"
    assert len(messages) == 2
    assert dispatches == []          # no bridge records to translate
    assert audit_out == {}           # no services
    assert media_events == []
    assert end is not None
    assert end.total_turns == 1
    assert end.failure_modes == []

    # And the dummy grader still consumes it cleanly.
    scores = _DummyGrader().grade(
        messages=messages,
        dispatches=dispatches,
        task=task,
        audit_data=audit_out,
        media_events=media_events,
    )
    assert scores.completion == 1.0
    assert scores.robustness == 1.0  # no errors ⇒ clean
    # Verify the assistant text survived the round-trip.
    final_text = next(
        m.message.text for m in reversed(messages) if m.message.role == "assistant"
    )
    assert "复利" in final_text


# ---------------------------------------------------------------------------
# Edge case: timeout status surfaces in failure_modes
# ---------------------------------------------------------------------------


def test_timeout_status_recorded(tmp_path, llm_meta, task):
    exec_trace = [
        {
            "type": "text",
            "role": "user",
            "content": "do the thing",
            "timestamp": "2026-06-20T10:00:00.000Z",
        }
    ]
    trace_path = translate_openclaw(
        execution_trace=exec_trace,
        usage_total={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        llm_meta=llm_meta,
        bridge_log_path=None,
        audit_data={},
        task=task,
        run_id="timeout",
        trace_dir=tmp_path,
        duration_ms=300_000,
        status="timeout",
    )
    _, _, _, _, end, _ = load_trace(trace_path)
    assert end is not None
    assert end.failure_modes == ["timeout"]
