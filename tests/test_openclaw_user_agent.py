"""Tests for the OpenClaw arm's multi-turn simulated-user (user_agent) support.

C (spec 2026-06-30-claweval-openclaw-multiturn-user-agent): the OpenClaw
harness itself drives multi-turn dialogue — each in-container ``openclaw agent``
call is one turn, the simulated ``UserAgent`` reacts to the agent's answer, and
the next turn re-invokes the CLI with the SAME ``--session-key`` so OpenClaw's
persisted session carries the conversation forward. This is NOT the AO arm's
in-process ``context +=`` mechanism (independent processes can't share memory);
the only equivalence is the loop shape, [DONE]/max_rounds semantics, the
[user_agent] marker and the reuse of claw_eval's UserAgent.
"""

from __future__ import annotations

from claw_eval.harnesses import get_harness
from claw_eval.harnesses.openclaw import (
    _drive_user_agent_turns,
    _merge_turn_traces,
)
from claw_eval.models.message import Message
from claw_eval.models.task import (
    Environment,
    Prompt,
    TaskDefinition,
    UserAgentTaskConfig,
)


def _make_task(*, ua_enabled: bool, max_rounds: int = 3) -> TaskDefinition:
    return TaskDefinition(
        task_id="UA_openclaw_task",
        task_name="openclaw user_agent multi-turn",
        prompt=Prompt(text="帮我算一下提前还房贷划不划算。"),
        environment=Environment(max_turns=10, timeout_seconds=300),
        user_agent=UserAgentTaskConfig(
            enabled=ua_enabled,
            persona="你是一个有房贷的借款人。",
            max_rounds=max_rounds,
        ),
    )


# --------------------------------------------------------------------------- #
# preflight + supported_features
# --------------------------------------------------------------------------- #
def test_preflight_no_longer_rejects_user_agent_task() -> None:
    """The old hard rejection ('openclaw harness does not support simulated
    user_agent') is gone — the OpenClaw arm now drives the loop itself."""
    harness = get_harness("openclaw")
    task = _make_task(ua_enabled=True)
    errs = harness.preflight(task)
    assert errs == [], f"preflight should accept user_agent now: {errs}"


def test_supported_features_includes_user_agent() -> None:
    harness = get_harness("openclaw")
    assert "user_agent" in harness.supported_features


def test_preflight_still_ok_for_non_user_agent() -> None:
    """A task with user_agent disabled is unaffected."""
    harness = get_harness("openclaw")
    task = _make_task(ua_enabled=False)
    assert harness.preflight(task) == []


# --------------------------------------------------------------------------- #
# driver loop  (_drive_user_agent_turns)
# --------------------------------------------------------------------------- #
def _raw(text: str) -> dict:
    """A minimal run_in_container-shaped result carrying a turn's final text."""
    return {"status": "ok", "trace": {"lastText": text, "executionTrace": []}}


class _StubUserAgent:
    """Returns scripted replies in order; ``None`` ends (== [DONE])."""

    def __init__(self, replies: list[str | None]) -> None:
        self._replies = list(replies)
        self.calls: list[list[Message]] = []
        self.personas: list[str] = []

    def generate_response(
        self, persona: str, conversation_messages: list[Message]
    ) -> str | None:
        self.personas.append(persona)
        self.calls.append(list(conversation_messages))
        if not self._replies:
            return None
        return self._replies.pop(0)


def _record_runner():
    """A run_turn callback that records (message, session_key) and replies."""
    calls: list[tuple[str, str | None]] = []

    def run_turn(message: str, session_key: str | None) -> dict:
        calls.append((message, session_key))
        return _raw(f"answer-to:{message[:24]}")

    return run_turn, calls


def test_single_turn_when_user_agent_none() -> None:
    """user_agent=None → exactly one turn, no session_key needed, zero
    behaviour change vs the pre-multi-turn single call."""
    run_turn, calls = _record_runner()
    result = _drive_user_agent_turns(
        prompt="原始任务",
        run_turn=run_turn,
        user_agent=None,
        persona="",
        max_rounds=8,
        agent_id="main",
        run_id="run1",
    )
    assert len(calls) == 1
    assert calls[0][0] == "原始任务"
    assert result["rounds"] == 0
    assert result["done"] is False
    assert len(result["turns"]) == 1


def test_multi_turn_runs_until_done() -> None:
    """UserAgent replies twice then returns None ([DONE]); the loop runs the
    first turn + 2 follow-up turns = 3 calls, all sharing one session_key."""
    run_turn, calls = _record_runner()
    ua = _StubUserAgent(["追问1", "追问2", None])
    result = _drive_user_agent_turns(
        prompt="原始任务",
        run_turn=run_turn,
        user_agent=ua,
        persona="借款人",
        max_rounds=8,
        agent_id="main",
        run_id="run1",
    )
    assert len(calls) == 3
    # turn 1 = task prompt; turns 2,3 = injected [user_agent] replies
    assert calls[0][0] == "原始任务"
    assert calls[1][0] == "[user_agent]\n追问1"
    assert calls[2][0] == "[user_agent]\n追问2"
    # all turns share ONE stable session_key (OpenClaw session continuity)
    keys = {sk for _, sk in calls}
    assert len(keys) == 1 and next(iter(keys))  # non-empty, single value
    assert result["rounds"] == 2
    assert result["done"] is True
    # persona threaded into every UserAgent call (3 calls: 2 replies + the
    # final [DONE] decision)
    assert ua.personas == ["借款人", "借款人", "借款人"]


def test_multi_turn_caps_at_max_rounds() -> None:
    """UserAgent never says [DONE]; the loop stops after max_rounds injected
    replies (max_rounds=2 → 1 initial + 2 follow-ups = 3 calls)."""
    run_turn, calls = _record_runner()
    ua = _StubUserAgent(["q1", "q2", "q3", "q4"])  # never None
    result = _drive_user_agent_turns(
        prompt="task",
        run_turn=run_turn,
        user_agent=ua,
        persona="p",
        max_rounds=2,
        agent_id="main",
        run_id="run1",
    )
    assert len(calls) == 3
    assert result["rounds"] == 2
    assert result["done"] is False  # capped, not satisfied


def test_session_key_includes_run_id() -> None:
    """session_key is per-run unique so concurrent/repeat runs don't collide."""
    run_turn, calls = _record_runner()
    ua = _StubUserAgent([None])
    _drive_user_agent_turns(
        prompt="t",
        run_turn=run_turn,
        user_agent=ua,
        persona="p",
        max_rounds=4,
        agent_id="main",
        run_id="RUNXYZ",
    )
    sk = calls[0][1]
    assert sk is not None and "RUNXYZ" in sk


def test_user_agent_sees_agent_answer_in_transcript() -> None:
    """The UserAgent must react to what the agent actually said — its
    transcript includes the prior assistant answer."""
    run_turn, calls = _record_runner()
    ua = _StubUserAgent(["再问", None])
    _drive_user_agent_turns(
        prompt="原始任务",
        run_turn=run_turn,
        user_agent=ua,
        persona="p",
        max_rounds=4,
        agent_id="main",
        run_id="r",
    )
    # first UserAgent call sees the agent's first answer as an assistant message
    first_transcript = ua.calls[0]
    assistant_texts = [m.text for m in first_transcript if m.role == "assistant"]
    assert any("answer-to:原始任务" in t for t in assistant_texts)



# --------------------------------------------------------------------------- #
# trace merge  (_merge_turn_traces)
# --------------------------------------------------------------------------- #
def _turn_with_trace(events: list[dict]) -> dict:
    return {"status": "ok", "trace": {"lastText": "", "executionTrace": events}}


def test_merge_single_turn_is_passthrough() -> None:
    """One turn → executionTrace unchanged (byte-identical to single-turn)."""
    ev = [
        {"type": "text", "role": "user", "content": "原始任务"},
        {"type": "text", "role": "assistant", "content": "答复"},
    ]
    merged = _merge_turn_traces([_turn_with_trace(ev)], injected=["x"])
    assert merged == ev


def test_merge_concatenates_without_duplicating_injected_marker() -> None:
    """REAL container behaviour: OpenClaw records the injected --message
    (already ``[user_agent]``-prefixed) as the FIRST user event of the NEXT
    turn's executionTrace. So _merge_turn_traces must only concatenate the
    per-turn traces — it must NOT re-insert its own [user_agent] sentinel, or
    the marker (and surrounding context) would appear twice. The injected reply
    shows up exactly once, carried by OpenClaw's own trace."""
    t1 = _turn_with_trace([
        {"type": "text", "role": "user", "content": "原始任务"},
        {"type": "text", "role": "assistant", "content": "答1"},
    ])
    # OpenClaw's next-turn trace already opens with the injected [user_agent] msg
    t2 = _turn_with_trace([
        {"type": "text", "role": "user", "content": "[user_agent]\n追问1"},
        {"type": "text", "role": "assistant", "content": "答2"},
    ])
    merged = _merge_turn_traces([t1, t2], injected=["追问1"])
    contents = [e.get("content") for e in merged]
    # exactly ONE [user_agent] marker, no duplication
    ua = [c for c in contents if str(c).startswith("[user_agent]")]
    assert len(ua) == 1, contents
    assert contents == ["原始任务", "答1", "[user_agent]\n追问1", "答2"]


# --------------------------------------------------------------------------- #
# integration: merged trace → translate_openclaw → readable trace
# --------------------------------------------------------------------------- #
def test_merged_trace_translates_with_user_agent_marker(tmp_path) -> None:
    """End-to-end: a 2-turn merged trace round-trips through translate_openclaw
    into a loadable trace whose user message carries [user_agent] and whose
    TraceEnd records the user_agent rounds/done."""
    from claw_eval.harnesses._trace_adapter import translate_openclaw
    from claw_eval.trace.reader import load_trace

    t1 = _turn_with_trace([
        {"type": "text", "role": "user", "content": "原始任务"},
        {"type": "text", "role": "assistant", "content": "答1"},
    ])
    # next turn opens with OpenClaw's own record of the injected reply
    t2 = _turn_with_trace([
        {"type": "text", "role": "user", "content": "[user_agent]\n追问1"},
        {"type": "text", "role": "assistant", "content": "答2"},
    ])
    merged = _merge_turn_traces([t1, t2], injected=["追问1"])

    task = _make_task(ua_enabled=True, max_rounds=8)
    out = translate_openclaw(
        execution_trace=merged,
        usage_total={},
        llm_meta={"model": "glm-5v-turbo"},
        bridge_log_path=None,
        audit_data={},
        task=task,
        run_id="runZ",
        trace_dir=tmp_path,
        duration_ms=1234,
        status="ok",
        user_agent_rounds=1,
        user_agent_max_rounds=8,
        user_agent_done=True,
    )
    start, messages, dispatches, media, end, audit = load_trace(out)
    # a [user_agent]-prefixed user message exists
    user_texts = [m.message.text for m in messages if m.message.role == "user"]
    assert any(t.startswith("[user_agent]") for t in user_texts), user_texts
    # TraceEnd carries the user_agent rollup
    assert end is not None
    assert end.user_agent_rounds == 1
    assert end.user_agent_done is True


# --------------------------------------------------------------------------- #
# _turn_final_text robustness (regression: empty lastText must fall back)
# --------------------------------------------------------------------------- #
from claw_eval.harnesses.openclaw import _turn_final_text


def test_turn_final_text_prefers_lasttext() -> None:
    raw = {"trace": {"lastText": "最终答案", "executionTrace": []}}
    assert _turn_final_text(raw) == "最终答案"


def test_turn_final_text_falls_back_to_execution_trace_when_lasttext_empty() -> None:
    """Container --json output is {payloads, meta} → run_in_container often
    leaves lastText="" even though the executionTrace holds the real assistant
    answer. An empty lastText must NOT be treated as the final text (that made
    the UserAgent see an empty turn and immediately say [DONE], collapsing
    multi-turn to a single turn). Fall back to the last assistant event."""
    raw = {
        "trace": {
            "lastText": "",
            "executionTrace": [
                {"type": "text", "role": "user", "content": "问题"},
                {"type": "text", "role": "assistant", "content": "我需要先确认几个信息……"},
            ],
        }
    }
    assert _turn_final_text(raw) == "我需要先确认几个信息……"


def test_turn_final_text_empty_when_nothing() -> None:
    assert _turn_final_text({"trace": {"lastText": "", "executionTrace": []}}) == ""
    assert _turn_final_text({}) == ""
