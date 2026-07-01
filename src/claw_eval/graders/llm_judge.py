"""LLM-as-judge for subjective communication quality scoring."""

from __future__ import annotations

import json
import random
import re
import time

from openai import OpenAI
from pydantic import BaseModel

from ..models.trace import _now


class _NonCompletionResponse(RuntimeError):
    """The endpoint returned a body that is not an OpenAI chat completion.

    The classic trigger is a gateway (e.g. new-api) whose ``base_url`` is
    missing the ``/v1`` API path: the request falls through to the SPA, which
    answers the unknown route with its HTML homepage and HTTP 200. The OpenAI
    SDK then deserialises that HTML into a plain ``str``, so ``resp.choices``
    raises ``AttributeError``. That is a configuration error, never transient —
    retrying it 30 times just hides the cause, so we raise immediately with an
    actionable message instead of feeding it into the retry loop.
    """


class JudgeResult(BaseModel):
    score: float  # 0.0-1.0
    reasoning: str


_SYSTEM_PROMPT = """\
You are an evaluation judge for an AI assistant.
You will be given a task prompt, a conversation, a summary of actions taken, and a rubric.
Follow the rubric to score the assistant's response on a 0.0-1.0 scale.
Respond with JSON only: {"score": <float>, "reasoning": "<brief explanation>"}
"""

_ACTIONS_SYSTEM_PROMPT = """\
You are an evaluation judge for an AI agent's actions.
You will be given a task prompt, a record of actions the agent actually performed \
(extracted from the server-side audit log, not from the agent's self-report), \
and a rubric.
Follow the rubric to score the quality of the agent's actions on a 0.0-1.0 scale.
Respond with JSON only: {"score": <float>, "reasoning": "<brief explanation>"}
"""

_VISUAL_SYSTEM_PROMPT = """\
You are a STRICT visual evaluation judge. Your job is to compare candidate images \
against reference images and/or a detailed rubric, then assign a score from 0.0 to 1.0.

CRITICAL RULES:
- You must be HARSH and PRECISE. Do NOT give generous scores.
- If the rubric describes specific content (e.g., specific notes, pitches, patterns, \
station names, colors), you MUST verify each detail. Getting the general layout right \
but the specific content wrong should score LOW (0.1-0.3).
- A visually "nice-looking" output that has WRONG content is a FAILURE.
- Only score above 0.5 if the MAJORITY of rubric criteria are clearly satisfied.
- Only score above 0.7 if the content is substantially correct with minor issues.
- Only score above 0.9 if the output is nearly perfect.
- Score 0.0-0.2 if the output is mostly wrong or unrecognizable.
- When reference images are provided, compare the candidate DIRECTLY against them — \
the reference is ground truth.

Respond with JSON only: {"score": <float>, "reasoning": "<brief explanation>"}
"""


class LLMJudge:
    """Judge communication quality using an LLM via OpenAI-compatible API."""

    # Marker distinguishing a real judge from the :class:`NoJudge` null-object.
    # Graders that bypass evaluate() and hit ``judge.client`` directly can guard
    # on ``getattr(judge, "enabled", True)`` to skip the LLM call when disabled.
    enabled = True

    def __init__(
        self,
        model_id: str = "google/gemini-2.5-flash",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self.client = OpenAI(api_key=api_key or "dummy", base_url=base_url)
        self.model_id = model_id
        self.base_url = base_url
        self._call_log: list[dict] = []

    def _content_or_raise(self, resp) -> str:
        """Return the first choice's text, or raise.

        Raises :class:`_NonCompletionResponse` (NOT retried) when ``resp`` is not
        a completion object at all — the tell-tale sign of a ``base_url`` missing
        the ``/v1`` API path. Raises ``ValueError`` (retried) when choices are
        present but empty (a genuine transient hiccup).
        """
        if isinstance(resp, str) or not hasattr(resp, "choices"):
            raise _NonCompletionResponse(
                f"judge endpoint returned a non-completion response "
                f"(type={type(resp).__name__}) — base_url {self.base_url!r} "
                f"likely needs the '/v1' API path. Body preview: "
                f"{str(resp)[:200]!r}"
            )
        if not resp.choices:
            raise ValueError("judge endpoint returned empty choices")
        return resp.choices[0].message.content or "{}"

    def evaluate(
        self,
        task_prompt: str,
        conversation: str,
        actions_summary: str,
        rubric: str,
    ) -> JudgeResult:
        """Evaluate communication quality and return a JudgeResult."""
        user_msg = (
            f"## Task Prompt\n{task_prompt}\n\n"
            f"## Conversation\n{conversation}\n\n"
            f"## Actions Taken\n{actions_summary}\n\n"
            f"## Rubric\n{rubric}"
        )
        max_retries = 30
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=8192,
                )
                raw = self._content_or_raise(resp)
                # Strip markdown code fences if present
                raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
                raw = re.sub(r"\s*```$", "", raw.strip())
                m = re.search(r'\{[^{}]*\}', raw)
                if m:
                    raw = m.group(0)
                try:
                    parsed = json.loads(raw)
                    score, reasoning = parsed["score"], parsed["reasoning"]
                except (json.JSONDecodeError, KeyError):
                    # Fallback: extract score and reasoning directly
                    score_m = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
                    reason_m = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
                    if score_m:
                        score = float(score_m.group(1))
                        reasoning = reason_m.group(1) if reason_m else ""
                    else:
                        raise json.JSONDecodeError("No score found in raw", raw, 0)

                result = JudgeResult(
                    score=max(0.0, min(1.0, float(score))),
                    reasoning=str(reasoning),
                )
                self._call_log.append({
                    "method": "evaluate",
                    "rubric_preview": rubric[:300],
                    "score": result.score,
                    "reasoning": result.reasoning,
                    "timestamp": _now(),
                })
                return result
            except _NonCompletionResponse:
                # Misconfigured endpoint (non-JSON body) — retrying never helps.
                raise
            except Exception as exc:
                last_exc = exc
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                delay = min(2 ** (attempt + 1), 8) + random.uniform(0, 1)
                print(f"[judge-retry] ({status or type(exc).__name__}), "
                      f"attempt {attempt + 1}/{max_retries}, waiting {delay:.1f}s ...")
                time.sleep(delay)

        raise RuntimeError(
            f"LLMJudge.evaluate failed after {max_retries} retries"
        ) from last_exc

    def evaluate_actions(
        self,
        task_prompt: str,
        artifacts: str,
        rubric: str,
    ) -> JudgeResult:
        """Evaluate the quality of agent actions/artifacts from audit log.

        Unlike ``evaluate`` which scores conversation quality, this method
        scores the actual operations the agent performed, as recorded by
        server-side audit logs.  The agent cannot manipulate this data.
        """
        user_msg = (
            f"## Task Prompt\n{task_prompt}\n\n"
            f"## Agent Actions (from server audit log)\n{artifacts}\n\n"
            f"## Rubric\n{rubric}"
        )
        max_retries = 30
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": _ACTIONS_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=8192,
                )
                raw = self._content_or_raise(resp)
                raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
                raw = re.sub(r"\s*```$", "", raw.strip())
                m = re.search(r'\{[^{}]*\}', raw)
                if m:
                    raw = m.group(0)
                try:
                    parsed = json.loads(raw)
                    score, reasoning = parsed["score"], parsed["reasoning"]
                except (json.JSONDecodeError, KeyError):
                    score_m = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
                    reason_m = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
                    if score_m:
                        score = float(score_m.group(1))
                        reasoning = reason_m.group(1) if reason_m else ""
                    else:
                        raise json.JSONDecodeError("No score found in raw", raw, 0)

                result = JudgeResult(
                    score=max(0.0, min(1.0, float(score))),
                    reasoning=str(reasoning),
                )
                self._call_log.append({
                    "method": "evaluate_actions",
                    "rubric_preview": rubric[:300],
                    "score": result.score,
                    "reasoning": result.reasoning,
                    "timestamp": _now(),
                })
                return result
            except _NonCompletionResponse:
                # Misconfigured endpoint (non-JSON body) — retrying never helps.
                raise
            except Exception as exc:
                last_exc = exc
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                delay = min(2 ** (attempt + 1), 8) + random.uniform(0, 1)
                print(f"[judge-retry] ({status or type(exc).__name__}), "
                      f"attempt {attempt + 1}/{max_retries}, waiting {delay:.1f}s ...")
                time.sleep(delay)

        raise RuntimeError(
            f"LLMJudge.evaluate_actions failed after {max_retries} retries"
        ) from last_exc

    def evaluate_visual(
        self,
        rubric: str,
        reference_images_b64: list[str],
        candidate_images_b64: list[str],
        context: str = "",
    ) -> JudgeResult:
        """Evaluate visual similarity between reference and candidate images.

        Constructs a message with inline base64 images for a vision-capable
        judge model and returns a JudgeResult (score + reasoning).
        """
        content_parts: list[dict] = []

        # Context / rubric text
        header = "## Visual Evaluation\n"
        if context:
            header += f"{context}\n\n"
        header += f"## Rubric\n{rubric}\n\n"
        header += (
            "## Scoring Calibration\n"
            "- 0.0-0.2: Output is mostly wrong, unrecognizable, or missing most required content\n"
            "- 0.2-0.4: Some elements present but major content errors (wrong notes, wrong colors, wrong layout)\n"
            "- 0.4-0.6: General structure is right but significant content inaccuracies remain\n"
            "- 0.6-0.8: Most content is correct with some minor issues\n"
            "- 0.8-1.0: Content is substantially correct, matching reference closely\n\n"
            "IMPORTANT: Looking nice is NOT enough. The CONTENT must be accurate. "
            "Check each rubric criterion individually and sum up the weighted scores.\n\n"
        )
        header += "Below are reference images followed by candidate images.\n"
        header += 'Respond with JSON only: {"score": <float>, "reasoning": "<brief explanation>"}'
        content_parts.append({"type": "text", "text": header})

        # Reference images
        if reference_images_b64:
            content_parts.append({"type": "text", "text": f"\n### Reference ({len(reference_images_b64)} images)"})
            for img_b64 in reference_images_b64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })

        # Candidate images
        if candidate_images_b64:
            content_parts.append({"type": "text", "text": f"\n### Candidate ({len(candidate_images_b64)} images)"})
            for img_b64 in candidate_images_b64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })

        max_retries = 30
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": _VISUAL_SYSTEM_PROMPT},
                        {"role": "user", "content": content_parts},
                    ],
                    temperature=0.0,
                    max_tokens=8192,
                )
                raw = self._content_or_raise(resp)
                raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
                raw = re.sub(r"\s*```$", "", raw.strip())
                m = re.search(r'\{[^{}]*\}', raw)
                if m:
                    raw = m.group(0)
                try:
                    parsed = json.loads(raw)
                    score, reasoning = parsed["score"], parsed["reasoning"]
                except (json.JSONDecodeError, KeyError):
                    score_m = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
                    reason_m = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
                    if score_m:
                        score = float(score_m.group(1))
                        reasoning = reason_m.group(1) if reason_m else ""
                    else:
                        raise json.JSONDecodeError("No score found in raw", raw, 0)
                result = JudgeResult(
                    score=max(0.0, min(1.0, float(score))),
                    reasoning=str(reasoning),
                )
                self._call_log.append({
                    "method": "evaluate_visual",
                    "rubric_preview": rubric[:300],
                    "n_ref_images": len(reference_images_b64),
                    "n_cand_images": len(candidate_images_b64),
                    "context_preview": context[:200],
                    "score": result.score,
                    "reasoning": result.reasoning,
                    "timestamp": _now(),
                })
                print(f"[judge-visual] score={result.score:.2f} reasoning={result.reasoning[:200]}")
                return result
            except _NonCompletionResponse:
                # Misconfigured endpoint (non-JSON body) — retrying never helps.
                raise
            except Exception as exc:
                last_exc = exc
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                delay = min(2 ** (attempt + 1), 8) + random.uniform(0, 1)
                print(f"[judge-visual-retry] ({status or type(exc).__name__}), "
                      f"attempt {attempt + 1}/{max_retries}, waiting {delay:.1f}s ...")
                time.sleep(delay)

        raise RuntimeError(
            f"LLMJudge.evaluate_visual failed after {max_retries} retries"
        ) from last_exc

    def get_call_log(self) -> list[dict]:
        return list(self._call_log)

    def reset_call_log(self) -> None:
        self._call_log.clear()


class NoJudge:
    """Null-object substitute for ``LLMJudge`` when the judge is disabled.

    When a run uses ``--no-judge`` (or no judge API key is configured), graders
    that call ``judge.evaluate()`` / ``judge.evaluate_actions()`` /
    ``judge.evaluate_visual()`` would otherwise crash with
    ``AttributeError: 'NoneType' object has no attribute 'evaluate'``.

    Substituting this null object for ``None`` at the single grading chokepoint
    (``_grade_with_optional_params``) makes every judge-dependent component
    contribute a **neutral 0.0** sub-score instead of crashing. 0.0 (not a
    passing score) is the safe choice: a judge-scored dimension that could not
    be evaluated should not silently pass. The neutral evaluation is recorded in
    the call log (with ``"no_judge": True``) so downstream tooling can see the
    judge was skipped.

    Methods accept ``*args, **kwargs`` so they are a drop-in regardless of the
    exact argument shape individual graders use.
    """

    # Graders that bypass evaluate() and access ``judge.client`` directly cannot
    # rely on the evaluate* methods; they can instead guard on
    # ``getattr(judge, "enabled", True)`` to skip the LLM call when disabled.
    enabled = False

    def __init__(self) -> None:
        self._call_log: list[dict] = []

    _REASON = "judge disabled (--no-judge): neutral 0.0 score"

    def _neutral(self, method: str) -> "JudgeResult":
        self._call_log.append(
            {"method": method, "no_judge": True, "score": 0.0, "reasoning": self._REASON}
        )
        return JudgeResult(score=0.0, reasoning=self._REASON)

    def evaluate(self, *args, **kwargs) -> "JudgeResult":
        return self._neutral("evaluate")

    def evaluate_actions(self, *args, **kwargs) -> "JudgeResult":
        return self._neutral("evaluate_actions")

    def evaluate_visual(self, *args, **kwargs) -> "JudgeResult":
        return self._neutral("evaluate_visual")

    def get_call_log(self) -> list[dict]:
        return list(self._call_log)

    def reset_call_log(self) -> None:
        self._call_log.clear()
