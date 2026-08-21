"""Prompt-attachment projection for the OpenClaw black-box harness.

The OpenClaw CLI ``agent --message`` entrypoint is text-only.  Image prompt
attachments therefore use OpenClaw's public plugin SDK ``agentCommand`` with
its native ``images`` option.  UTF-8 text and CSV attachments are appended to
the first user message.  Every attempted projection produces evidence that the
trace adapter can turn into a truthful ``MediaLoad`` event.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


_TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt"}
_MEDIA_RUNNER_SOURCE = """import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const input = JSON.parse(await readFile(process.argv[2], "utf8"));
const agentRuntimeUrl = pathToFileURL(
  join(input.openclawPackageRoot, "dist", "plugin-sdk", "agent-runtime.js"),
).href;
const { agentCommand } = await import(agentRuntimeUrl);
const runtime = {
  log: () => {},
  error: (...args) => process.stderr.write(`${args.map(String).join(" ")}\\n`),
  exit: (code) => {
    if (Number(code) !== 0) process.exitCode = Number(code);
  },
};
const result = await agentCommand(input.options, runtime);
process.stdout.write(`${JSON.stringify(result)}\\n`);
"""


@dataclass(frozen=True)
class PromptMediaProjection:
    """Effective first-turn message, native images, and trace evidence."""

    message: str
    images: tuple[dict[str, str], ...]
    events: tuple[dict[str, Any], ...]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _event(
    *,
    path: Path,
    modality: str,
    mime_type: str,
    data: bytes,
    status: str,
    note: str,
) -> dict[str, Any]:
    return {
        "modality": modality,
        "source_path": str(path),
        "mime_type": mime_type,
        "size_bytes": len(data),
        "sha256": _sha256(data),
        "status": status,
        "note": note,
    }


def _resolve_attachment(task_dir: Path, value: object) -> Path:
    root = task_dir.resolve()
    raw = Path(str(value or "").strip())
    if not str(raw):
        raise ValueError("prompt attachment path is empty")
    path = (raw if raw.is_absolute() else root / raw).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"prompt attachment escapes task directory: {value!r}")
    return path


def project_prompt_attachments(
    *,
    message: str,
    attachments: Sequence[object],
    task_dir: Path,
) -> PromptMediaProjection:
    """Project declared prompt attachments without claiming unsupported loads."""

    effective_message = str(message or "")
    images: list[dict[str, str]] = []
    events: list[dict[str, Any]] = []

    for raw_path in attachments:
        try:
            path = _resolve_attachment(task_dir, raw_path)
        except ValueError:
            display = (task_dir / str(raw_path or "")).resolve()
            events.append(
                _event(
                    path=display,
                    modality="document",
                    mime_type="application/octet-stream",
                    data=b"",
                    status="error",
                    note="prompt_attachment; invalid_path",
                )
            )
            continue

        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        modality = (
            "image"
            if mime_type.startswith("image/")
            else "audio"
            if mime_type.startswith("audio/")
            else "video"
            if mime_type.startswith("video/")
            else "document"
        )
        try:
            data = path.read_bytes()
        except OSError as exc:
            events.append(
                _event(
                    path=path,
                    modality=modality,
                    mime_type=mime_type,
                    data=b"",
                    status="error",
                    note=f"prompt_attachment; read_error={type(exc).__name__}",
                )
            )
            continue

        if modality == "image":
            images.append(
                {
                    "type": "image",
                    "data": base64.b64encode(data).decode("ascii"),
                    "mimeType": mime_type,
                }
            )
        elif path.suffix.lower() in _TEXT_SUFFIXES:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                events.append(
                    _event(
                        path=path,
                        modality=modality,
                        mime_type=mime_type,
                        data=data,
                        status="error",
                        note=(
                            f"prompt_attachment; utf8_decode_error={type(exc).__name__}"
                        ),
                    )
                )
                continue
            effective_message += f"\n\n[attached document: {path.name}]\n{text}"
        else:
            events.append(
                _event(
                    path=path,
                    modality=modality,
                    mime_type=mime_type,
                    data=data,
                    status="skipped",
                    note="prompt_attachment; unsupported_projection",
                )
            )
            continue

        events.append(
            _event(
                path=path,
                modality=modality,
                mime_type=mime_type,
                data=data,
                status="loaded",
                note="prompt_attachment; projected_to_openclaw_agent_input",
            )
        )

    return PromptMediaProjection(
        message=effective_message,
        images=tuple(images),
        events=tuple(events),
    )


def build_media_agent_command(
    *,
    raw_dir: Path,
    openclaw_package_root: str,
    message: str,
    agent_id: str,
    images: Sequence[dict[str, str]],
    timeout_s: float,
    session_key: str | None = None,
    thinking: bool = False,
    reasoning_effort: str | None = None,
) -> list[str]:
    """Materialise an SDK runner and return its ``node`` argv."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    runner_path = raw_dir / "openclaw_media_runner.mjs"
    input_path = raw_dir / "openclaw_media_input.json"
    runner_path.write_text(_MEDIA_RUNNER_SOURCE, encoding="utf-8")

    options: dict[str, Any] = {
        "message": message,
        "agentId": agent_id,
        "images": list(images),
        "json": True,
        "senderIsOwner": True,
        "cleanupBundleMcpOnRunEnd": True,
        "cleanupCliLiveSessionOnRunEnd": True,
        "oneShotCliRun": True,
    }
    if session_key:
        options["sessionKey"] = session_key
    if reasoning_effort is not None:
        options["thinking"] = reasoning_effort
    elif thinking:
        options["thinking"] = "on"
    if isinstance(timeout_s, (int, float)) and timeout_s > 0:
        options["timeout"] = str(int(timeout_s))

    input_path.write_text(
        json.dumps(
            {
                "openclawPackageRoot": openclaw_package_root,
                "options": options,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return ["node", str(runner_path), str(input_path)]


def resolve_openclaw_package_root(env: dict[str, str] | None = None) -> str:
    """Locate the installed OpenClaw package for plugin-SDK imports."""

    environment = env or os.environ
    candidates: list[Path] = []
    explicit = environment.get("OPENCLAW_PACKAGE_ROOT")
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(Path("/usr/local/lib/node_modules/openclaw"))

    executable = shutil.which("openclaw", path=environment.get("PATH"))
    if executable:
        resolved = Path(executable).resolve()
        candidates.extend(
            parent for parent in resolved.parents if parent.name == "openclaw"
        )
    try:
        npm_root = subprocess.run(
            ["npm", "root", "-g"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        npm_root = ""
    if npm_root:
        candidates.append(Path(npm_root) / "openclaw")

    for candidate in candidates:
        root = candidate.resolve()
        if (root / "dist/plugin-sdk/agent-runtime.js").is_file():
            return str(root)
    raise FileNotFoundError("cannot locate OpenClaw plugin SDK package root")
