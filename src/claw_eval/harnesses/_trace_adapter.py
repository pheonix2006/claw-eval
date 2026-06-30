"""OpenClaw session → claw-eval trace JSONL translator.

Phase 3 Wave 2 (§3.5 / §6.4) — see ``docs/harness_design.md``.

This module is the **read side** of the harness contract for external CLI
agents.  ``_openclaw_native.run`` returns a payload with a normalised
``executionTrace`` array, a ``usageTotal`` aggregate, and ``llm`` metadata.
The bridge plugin records every HTTP fan-out into a sibling JSONL file.  The
two streams are merged here into a standard claw-eval trace that the existing
``load_trace`` / ``AbstractGrader.grade`` pipeline can consume unchanged.

The translation table (§3.5) is implemented in :func:`translate_openclaw`.

Assumed input schema
--------------------

``execution_trace`` entries (produced by ``_openclaw_native._extract_openclaw_trace``):

- ``{"type": "text", "role": "user", "content": str, "timestamp": str|None}``
- ``{"type": "text", "role": "assistant", "content": str, "timestamp": str|None,
    "turn": int, "llm": {"provider": str, "baseUrl": str, "model": str,
    "usage": {"prompt_tokens": int, "completion_tokens": int,
              "total_tokens": int, "cache_read": int, "cache_write": int},
    "stopReason": str|None, "errorMessage": str|None}}``
- ``{"type": "tool", "role": "tool", "tool": str, "callID": str,
    "timestamp": str|None, "startedAt": str|None, "finishedAt": str|None,
    "durationMs": int|None, "input": dict, "output": {"text": str}|None,
    "exitCode": int|None, "status": str|None}``

``bridge_log_path`` JSONL entries (produced by the bridge plugin's
``recordCall``, §3.4a):

- ``{"toolCallId": str, "tool": str, "url": str, "method": str,
    "request": dict, "status": int, "response": Any, "durationMs": int}``
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..models.content import TextBlock, ToolResultBlock, ToolUseBlock
from ..models.message import Message
from ..models.trace import (
    AuditSnapshot,
    TokenUsage,
    ToolDispatch,
    TraceEnd,
    TraceMessage,
    TraceStart,
)
from ..trace.writer import TraceWriter

if TYPE_CHECKING:
    from ..models.task import TaskDefinition


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_tool_output(output: Any) -> str:
    """Render the OpenClaw ``tool.output`` field as a single text string.

    OpenClaw's ``_extract_openclaw_trace`` packs every toolResult as
    ``{"text": "..."}``, but be liberal in what we accept — fall back to JSON
    encoding for unexpected shapes so downstream graders still see something
    readable.
    """
    if output is None:
        return ""
    if isinstance(output, dict):
        text = output.get("text")
        if isinstance(text, str):
            return text
        # Some bridge / native shapes may put structured data here. Render it.
        try:
            return json.dumps(output, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(output)
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(output)


def _load_bridge_log(
    bridge_log_path: Path | None,
) -> dict[str, dict[str, Any]]:
    """Read the bridge JSONL log and key each record by ``toolCallId``.

    Returns an empty mapping when ``bridge_log_path`` is ``None`` or missing —
    those are the legitimate empty-tool-set or bridge-disabled cases.
    Malformed lines are skipped with a warning rather than aborting the
    translation; we'd rather emit a partial trace than refuse to grade.
    """
    if bridge_log_path is None:
        return {}
    path = Path(bridge_log_path)
    if not path.exists() or not path.is_file():
        return {}

    index: dict[str, dict[str, Any]] = {}
    with path.open() as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                _log.warning(
                    "bridge log %s line %d: malformed JSON (%s) — skipped",
                    path, lineno, exc,
                )
                continue
            if not isinstance(rec, dict):
                continue
            tcid = rec.get("toolCallId")
            if not isinstance(tcid, str) or not tcid:
                _log.warning(
                    "bridge log %s line %d: missing toolCallId — skipped",
                    path, lineno,
                )
                continue
            # Last-write-wins: should not occur (OpenClaw guarantees unique
            # callIDs), but keep the most recent if it does.
            index[tcid] = rec
    return index


def _coerce_status_int(value: Any) -> int | None:
    """Cast a bridge ``status`` field to int; return None on failure."""
    if isinstance(value, bool):  # bool is an int subclass — exclude explicitly
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _coerce_duration_ms(value: Any) -> float:
    """Cast a duration field to float milliseconds; default 0.0."""
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _compute_is_error(
    session_event: dict[str, Any],
    bridge_record: dict[str, Any] | None,
) -> bool:
    """Apply the §3.5 ``is_error`` rule.

    Truthy if any of:
    1. bridge HTTP status >= 400
    2. session toolCall has errorMessage/isError=true
    3. bridge has no record for this callID (degraded path)
    """
    # Rule 3 (degraded): missing bridge record ⇒ error
    if bridge_record is None:
        return True

    # Rule 1: bridge HTTP-level failure
    status = _coerce_status_int(bridge_record.get("status"))
    if status is not None and status >= 400:
        return True

    # Rule 2: native-side error markers. _openclaw_native sometimes propagates
    # ``status`` ("ok"/"error") and / or ``exitCode`` on the tool event.
    if session_event.get("isError") is True:
        return True
    if session_event.get("errorMessage"):
        return True
    exit_code = session_event.get("exitCode")
    if isinstance(exit_code, int) and exit_code != 0:
        return True
    sess_status = session_event.get("status")
    if isinstance(sess_status, str) and sess_status.lower() in {"error", "failed", "fail"}:
        return True

    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def translate_openclaw(
    *,
    execution_trace: list[dict[str, Any]],
    usage_total: dict[str, Any],
    llm_meta: dict[str, Any],
    bridge_log_path: Path | None,
    audit_data: dict[str, dict[str, Any]],
    task: "TaskDefinition",
    run_id: str,
    trace_dir: Path,
    duration_ms: int,
    status: str,
    user_agent_rounds: int = 0,
    user_agent_max_rounds: int = 0,
    user_agent_done: bool = False,
) -> Path:
    """Translate an OpenClaw session + bridge log into a claw-eval trace JSONL.

    Returns the absolute path to the written trace file.  The output is
    suitable for ``claw_eval.trace.reader.load_trace`` and any grader
    consuming the standard contract.

    See ``docs/harness_design.md`` §3.5 for the per-event mapping, including
    the ``is_error`` rule and how the two data sources (OpenClaw session vs.
    bridge plugin HTTP log) are merged through the shared ``callID``.
    """
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{task.task_id}_{run_id}.jsonl"

    # Stable trace_id for the whole stream — load_trace doesn't enforce, but
    # consumers correlate events through it.
    trace_id = f"{task.task_id}_{run_id}_{uuid.uuid4().hex[:8]}"

    # Pull the model id from the LLM metadata block; OpenClaw's session
    # writer populates llm.model with the resolved model path. Fall back to
    # "unknown" so TraceStart never crashes on a partial trace.
    model = (
        llm_meta.get("model")
        if isinstance(llm_meta, dict) and isinstance(llm_meta.get("model"), str)
        else None
    ) or "unknown"

    # Pre-index bridge records by toolCallId for O(1) lookup.
    bridge_index = _load_bridge_log(bridge_log_path)

    # We must remember the last assistant TraceMessage we emitted so that the
    # next ``type=tool`` event can splice its ToolUseBlock into the assistant's
    # content list. The OpenClaw native runner emits the assistant text BEFORE
    # the corresponding tool entries (see _extract_openclaw_trace lines ~862,
    # 887), so this lookback is always to the most recent assistant message.
    pending_messages: list[TraceMessage] = []
    pending_dispatches: list[ToolDispatch] = []
    last_assistant_msg: TraceMessage | None = None

    assistant_turn_count = 0
    failure_modes: list[str] = []

    # ------------------------------------------------------------------
    # Pass 1: walk executionTrace, build messages + ToolDispatch records.
    # ------------------------------------------------------------------
    for event in execution_trace:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        role = event.get("role")

        if etype == "text" and role == "user":
            content_str = event.get("content")
            if not isinstance(content_str, str):
                content_str = "" if content_str is None else str(content_str)
            msg = TraceMessage(
                trace_id=trace_id,
                message=Message(
                    role="user",
                    content=[TextBlock(text=content_str)],
                ),
            )
            pending_messages.append(msg)
            # A new user message resets the "current assistant" anchor — any
            # subsequent tool events should pair with the next assistant
            # message (which OpenClaw's writer guarantees).
            last_assistant_msg = None
            continue

        if etype == "text" and role == "assistant":
            content_str = event.get("content")
            if not isinstance(content_str, str):
                content_str = "" if content_str is None else str(content_str)
            llm_block = event.get("llm") if isinstance(event.get("llm"), dict) else {}
            usage_block = (
                llm_block.get("usage") if isinstance(llm_block.get("usage"), dict) else {}
            )
            usage = TokenUsage(
                input_tokens=int(usage_block.get("prompt_tokens") or 0),
                output_tokens=int(usage_block.get("completion_tokens") or 0),
            )
            msg = TraceMessage(
                trace_id=trace_id,
                message=Message(
                    role="assistant",
                    content=[TextBlock(text=content_str)],
                ),
                usage=usage,
            )
            pending_messages.append(msg)
            last_assistant_msg = msg
            assistant_turn_count += 1
            # Propagate per-turn errors into failure_modes (e.g. the
            # 400-token-overflow seen in real OpenClaw runs) so graders can
            # see why a run ended short. Don't add duplicates.
            err_msg = llm_block.get("errorMessage")
            if isinstance(err_msg, str) and err_msg and err_msg not in failure_modes:
                failure_modes.append(err_msg)
            continue

        if etype == "tool":
            call_id = event.get("callID")
            tool_name = event.get("tool")
            if not isinstance(call_id, str) or not call_id:
                _log.warning("tool event missing callID — skipped: %r", event)
                continue
            if not isinstance(tool_name, str) or not tool_name:
                _log.warning("tool event missing tool name — skipped: %r", event)
                continue

            tool_input = (
                event.get("input") if isinstance(event.get("input"), dict) else {}
            )
            tool_output = event.get("output")
            duration = _coerce_duration_ms(event.get("durationMs"))

            # 1) Splice ToolUseBlock into the most recent assistant message.
            if last_assistant_msg is None:
                # No assistant message preceded this tool event — synthesise a
                # bare assistant message so the ToolUseBlock has a home. This
                # is a degraded path; OpenClaw's writer normally always emits
                # an assistant 'text' event first.
                _log.warning(
                    "tool event %s arrived without preceding assistant message — "
                    "synthesising empty assistant message",
                    call_id,
                )
                synthetic = TraceMessage(
                    trace_id=trace_id,
                    message=Message(role="assistant", content=[TextBlock(text="")]),
                )
                pending_messages.append(synthetic)
                last_assistant_msg = synthetic
                assistant_turn_count += 1

            last_assistant_msg.message.content.append(
                ToolUseBlock(id=call_id, name=tool_name, input=tool_input)
            )

            # 2) Pair with bridge record (if any) to compute is_error.
            bridge_rec = bridge_index.get(call_id)
            is_error = _compute_is_error(event, bridge_rec)

            # 3) Emit ToolResultBlock as the next user message (claw-eval
            #    folds toolResult into a user-role message; see
            #    runner/loop.py and grader format_conversation_detailed).
            result_text = _serialize_tool_output(tool_output)
            tool_result_msg = TraceMessage(
                trace_id=trace_id,
                message=Message(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_use_id=call_id,
                            content=[TextBlock(text=result_text)],
                            is_error=is_error,
                        )
                    ],
                ),
            )
            pending_messages.append(tool_result_msg)

            # 4) Emit ToolDispatch from the bridge record (preferred) or a
            #    degraded placeholder (status=500) when bridge missed it.
            if bridge_rec is not None:
                status_code = _coerce_status_int(bridge_rec.get("status"))
                if status_code is None:
                    # Bridge wrote a record but no usable status — degrade,
                    # but still surface the request/response payloads.
                    status_code = 500
                req_body = (
                    bridge_rec.get("request")
                    if isinstance(bridge_rec.get("request"), dict)
                    else {}
                )
                pending_dispatches.append(
                    ToolDispatch(
                        trace_id=trace_id,
                        tool_use_id=call_id,
                        tool_name=tool_name,
                        endpoint_url=str(bridge_rec.get("url") or ""),
                        request_body=req_body,
                        response_status=status_code,
                        response_body=bridge_rec.get("response"),
                        latency_ms=_coerce_duration_ms(bridge_rec.get("durationMs"))
                        or duration,
                    )
                )
            else:
                # Degraded: session toolCall has no bridge record. Default
                # to status=500 + empty endpoint so robustness penalises it,
                # rather than silently masking a failure as a clean run.
                _log.warning(
                    "tool event %s has no matching bridge record — "
                    "emitting placeholder ToolDispatch(response_status=500)",
                    call_id,
                )
                pending_dispatches.append(
                    ToolDispatch(
                        trace_id=trace_id,
                        tool_use_id=call_id,
                        tool_name=tool_name,
                        endpoint_url="",
                        request_body=tool_input,
                        response_status=500,
                        response_body=None,
                        latency_ms=duration,
                    )
                )
            continue

        # Unknown event shapes (e.g. media events the native runner might
        # add later) are silently ignored; we trade extensibility for
        # forward-compat. Log at debug for traceability.
        _log.debug("unhandled executionTrace event: %r", event)

    # ------------------------------------------------------------------
    # Pass 2: derive TraceEnd tokens / failure modes from the rollup data.
    # ------------------------------------------------------------------
    ut = usage_total if isinstance(usage_total, dict) else {}
    input_tokens = int(ut.get("prompt_tokens") or 0)
    output_tokens = int(ut.get("completion_tokens") or 0)
    total_tokens = int(ut.get("total_tokens") or (input_tokens + output_tokens))

    if status == "timeout":
        if "timeout" not in failure_modes:
            failure_modes.insert(0, "timeout")
    elif status == "error":
        # If a per-turn errorMessage already populated failure_modes, leave
        # it; otherwise just mark "error" so the grader has something to see.
        if not failure_modes:
            failure_modes.append("error")

    wall_time_s = float(duration_ms) / 1000.0 if duration_ms else 0.0

    # ------------------------------------------------------------------
    # Pass 3: write the trace JSONL through TraceWriter (pydantic-serialised
    # so the field set matches the claweval path byte-for-byte).
    # ------------------------------------------------------------------
    with TraceWriter(trace_path) as writer:
        writer.write_event(
            TraceStart(
                trace_id=trace_id,
                task_id=task.task_id,
                model=model,
                harness="openclaw",
            )
        )

        for msg in pending_messages:
            writer.write_event(msg)

        for disp in pending_dispatches:
            writer.write_event(disp)

        # Audit snapshots — one per service. The audit_url is informational
        # under the OpenClaw path (we collect /audit out-of-band, after the
        # fact), so a placeholder is acceptable per §6.4.
        for svc_name, svc_data in (audit_data or {}).items():
            writer.write_event(
                AuditSnapshot(
                    trace_id=trace_id,
                    service_name=svc_name,
                    audit_url=f"openclaw://post-hoc/{svc_name}/audit",
                    audit_data=svc_data if isinstance(svc_data, dict) else {},
                )
            )

        writer.write_event(
            TraceEnd(
                trace_id=trace_id,
                total_turns=assistant_turn_count,
                model_input_tokens=input_tokens,
                model_output_tokens=output_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                model_time_s=0.0,
                tool_time_s=0.0,
                other_time_s=0.0,
                wall_time_s=wall_time_s,
                failure_modes=failure_modes,
                user_agent_rounds=user_agent_rounds,
                user_agent_max_rounds=user_agent_max_rounds,
                user_agent_done=user_agent_done,
            )
        )

    return trace_path
