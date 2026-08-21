from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from claw_eval.harnesses._openclaw_media import (
    build_media_agent_command,
    project_prompt_attachments,
)
from claw_eval.harnesses._trace_adapter import translate_openclaw
from claw_eval.models.task import Prompt, TaskDefinition
from claw_eval.trace.reader import load_trace


def test_projects_images_and_utf8_documents_with_truthful_evidence(
    tmp_path: Path,
) -> None:
    image = tmp_path / "fixtures/image.jpg"
    csv = tmp_path / "fixtures/data.csv"
    text = tmp_path / "fixtures/notes.txt"
    image.parent.mkdir()
    image.write_bytes(b"jpeg-fixture")
    csv.write_text("name,value\na,1\n", encoding="utf-8")
    text.write_text("hello document", encoding="utf-8")

    projection = project_prompt_attachments(
        message="answer from attachments",
        attachments=["fixtures/image.jpg", "fixtures/data.csv", "fixtures/notes.txt"],
        task_dir=tmp_path,
    )

    assert projection.images == (
        {
            "type": "image",
            "data": base64.b64encode(b"jpeg-fixture").decode("ascii"),
            "mimeType": "image/jpeg",
        },
    )
    assert "[attached document: data.csv]\nname,value\na,1\n" in projection.message
    assert "[attached document: notes.txt]\nhello document" in projection.message
    assert [event["status"] for event in projection.events] == [
        "loaded",
        "loaded",
        "loaded",
    ]
    assert projection.events[0]["sha256"] == hashlib.sha256(b"jpeg-fixture").hexdigest()
    assert projection.events[0]["size_bytes"] == len(b"jpeg-fixture")
    assert all(
        event["note"] == "prompt_attachment; projected_to_openclaw_agent_input"
        for event in projection.events
    )


def test_missing_or_unsupported_attachment_is_not_reported_loaded(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "fixture.bin"
    binary.write_bytes(b"binary")

    projection = project_prompt_attachments(
        message="prompt",
        attachments=["missing.jpg", "fixture.bin", "../escape.jpg"],
        task_dir=tmp_path,
    )

    assert projection.message == "prompt"
    assert projection.images == ()
    assert [event["status"] for event in projection.events] == [
        "error",
        "skipped",
        "error",
    ]


def test_empty_attachment_list_preserves_the_prompt(tmp_path: Path) -> None:
    projection = project_prompt_attachments(
        message="unchanged",
        attachments=[],
        task_dir=tmp_path,
    )

    assert projection.message == "unchanged"
    assert projection.images == ()
    assert projection.events == ()


def test_invalid_utf8_document_is_error_and_is_not_appended(tmp_path: Path) -> None:
    document = tmp_path / "invalid.txt"
    document.write_bytes(b"\xff\xfe")

    projection = project_prompt_attachments(
        message="prompt",
        attachments=["invalid.txt"],
        task_dir=tmp_path,
    )

    assert projection.message == "prompt"
    assert projection.images == ()
    assert len(projection.events) == 1
    assert projection.events[0]["status"] == "error"
    assert projection.events[0]["size_bytes"] == 2
    assert "utf8_decode_error=UnicodeDecodeError" in projection.events[0]["note"]


def test_media_sdk_command_matches_native_agent_command_contract(
    tmp_path: Path,
) -> None:
    command = build_media_agent_command(
        raw_dir=tmp_path,
        openclaw_package_root="/usr/local/lib/node_modules/openclaw",
        message="look",
        agent_id="main",
        images=[{"type": "image", "data": "YWJj", "mimeType": "image/jpeg"}],
        timeout_s=179.9,
        session_key="agent:main:test",
        reasoning_effort="medium",
    )

    assert command == [
        "node",
        str(tmp_path / "openclaw_media_runner.mjs"),
        str(tmp_path / "openclaw_media_input.json"),
    ]
    payload = json.loads((tmp_path / "openclaw_media_input.json").read_text())
    assert payload == {
        "openclawPackageRoot": "/usr/local/lib/node_modules/openclaw",
        "options": {
            "message": "look",
            "agentId": "main",
            "images": [{"type": "image", "data": "YWJj", "mimeType": "image/jpeg"}],
            "json": True,
            "senderIsOwner": True,
            "cleanupBundleMcpOnRunEnd": True,
            "cleanupCliLiveSessionOnRunEnd": True,
            "oneShotCliRun": True,
            "sessionKey": "agent:main:test",
            "thinking": "medium",
            "timeout": "179",
        },
    }
    assert "agentCommand" in (tmp_path / "openclaw_media_runner.mjs").read_text()


def test_trace_adapter_emits_media_load_before_messages(tmp_path: Path) -> None:
    task = TaskDefinition(
        task_id="media-task",
        task_name="media task",
        prompt=Prompt(text="look"),
    )
    path = translate_openclaw(
        execution_trace=[{"type": "text", "role": "user", "content": "look"}],
        usage_total={},
        llm_meta={"model": "model"},
        bridge_log_path=None,
        audit_data={},
        task=task,
        run_id="run",
        trace_dir=tmp_path,
        duration_ms=1,
        status="ok",
        media_events=[
            {
                "modality": "image",
                "source_path": "/task/image.jpg",
                "mime_type": "image/jpeg",
                "size_bytes": 3,
                "sha256": hashlib.sha256(b"abc").hexdigest(),
                "status": "loaded",
                "note": "prompt_attachment; projected_to_openclaw_agent_input",
            }
        ],
    )

    event_types = [json.loads(line)["type"] for line in path.read_text().splitlines()]
    assert event_types[:3] == ["trace_start", "media_load", "message"]
    _, _, _, media_events, _, _ = load_trace(path)
    assert len(media_events) == 1
    assert media_events[0].source_path == "/task/image.jpg"
    assert media_events[0].status == "loaded"
