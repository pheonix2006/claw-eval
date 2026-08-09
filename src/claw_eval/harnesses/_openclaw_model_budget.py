"""Task-scoped streaming proxy enforcing ClawEval's model-turn budget.

OpenClaw can perform many provider calls inside one CLI invocation and exposes
no generic ``--max-turns`` flag.  This proxy sits between OpenClaw and the
configured OpenAI-compatible endpoint, forwards bytes without interpreting the
payload, and rejects the first completion request beyond the task budget.
Only request metadata is persisted; credentials and bodies are never logged.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx


_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


_PROVIDER_TRANSPORTS = ("openai-completions", "anthropic-messages")


def _target_url(
    base: str,
    path: str,
    provider_transport: str = "openai-completions",
) -> str:
    base = base.rstrip("/")
    # ``base_url`` is already the provider's complete API root.  The local
    # budget proxy advertises a synthetic ``/v1`` prefix to OpenClaw, so strip
    # that prefix before appending an endpoint whenever the provider root
    # already owns a path (for example Z.AI's ``/api/coding/paas/v4``).
    # Preserve it only for origin-only bases such as ``https://api.example``.
    if (
        provider_transport == "openai-completions"
        and urlsplit(base).path.rstrip("/")
        and path.startswith("/v1/")
    ):
        path = path[3:]
    return f"{base}/{path.lstrip('/')}"


class OpenClawModelBudgetProxy(AbstractContextManager["OpenClawModelBudgetProxy"]):
    def __init__(
        self,
        *,
        target_base_url: str,
        max_completions: int,
        evidence_path: Path,
        provider_transport: str = "openai-completions",
    ):
        if max_completions <= 0:
            raise ValueError("max_completions must be positive")
        parsed = urlsplit(target_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("target_base_url must be an HTTP(S) URL")
        if provider_transport not in _PROVIDER_TRANSPORTS:
            raise ValueError(
                "provider_transport must be openai-completions or "
                "anthropic-messages"
            )
        self.target_base_url = target_base_url
        self.provider_transport = provider_transport
        self.max_completions = max_completions
        self.evidence_path = Path(evidence_path)
        self.completion_count = 0
        self.rejected_count = 0
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def exhausted(self) -> bool:
        return self.rejected_count > 0

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("proxy is not running")
        origin = f"http://127.0.0.1:{self._server.server_address[1]}"
        if self.provider_transport == "anthropic-messages":
            return origin
        return f"{origin}/v1"

    def _record(self, path: str, decision: str) -> None:
        self.requests.append({"path": path.split("?", 1)[0], "decision": decision,
                              "completion_index": self.completion_count,
                              "timestamp": time.time()})
        self._write_evidence()

    def _reserve(self, path: str) -> bool:
        normalized_path = path.split("?", 1)[0].rstrip("/")
        if self.provider_transport == "anthropic-messages":
            is_completion = normalized_path == "/v1/messages"
        else:
            is_completion = normalized_path.endswith(
                ("/chat/completions", "/responses")
            )
        with self._lock:
            if not is_completion:
                self._record(path, "forward_non_completion")
                return True
            if self.completion_count >= self.max_completions:
                self.rejected_count += 1
                self._record(path, "reject_max_turns")
                return False
            self.completion_count += 1
            self._record(path, "forward")
            return True

    def _write_evidence(self) -> None:
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "target_origin": urlsplit(self.target_base_url)._replace(path="", query="", fragment="").geturl(),
            "provider_transport": self.provider_transport,
            "max_completions": self.max_completions,
            "completion_count": self.completion_count,
            "rejected_count": self.rejected_count,
            "requests": self.requests,
        }
        tmp = self.evidence_path.with_suffix(self.evidence_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.evidence_path)

    def __enter__(self) -> "OpenClawModelBudgetProxy":
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format: str, *_args: object) -> None:
                return

            def do_POST(self) -> None:
                if not owner._reserve(self.path):
                    body = json.dumps({"error": {"message": "ClawEval max_turns exceeded",
                                                  "type": "invalid_request_error",
                                                  "code": "claw_eval_max_turns_exceeded"}}).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_HEADERS}
                try:
                    with httpx.stream(
                        "POST",
                        _target_url(
                            owner.target_base_url,
                            self.path,
                            owner.provider_transport,
                        ),
                        content=body,
                        headers=headers,
                        timeout=httpx.Timeout(
                            connect=30, read=None, write=30, pool=30
                        ),
                        trust_env=False,
                    ) as upstream:
                        self.send_response(upstream.status_code)
                        has_length = False
                        for key, value in upstream.headers.items():
                            if key.lower() in _HOP_HEADERS:
                                continue
                            if key.lower() == "content-length":
                                has_length = True
                            self.send_header(key, value)
                        if not has_length:
                            self.send_header("Connection", "close")
                            self.close_connection = True
                        self.end_headers()
                        for chunk in upstream.iter_raw():
                            self.wfile.write(chunk)
                            self.wfile.flush()
                except (httpx.HTTPError, OSError) as exc:
                    if not self.wfile.closed:
                        owner._record(self.path, f"upstream_error:{type(exc).__name__}")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # A cancelled OpenClaw request may leave the upstream SSE read blocked.
        # Such request threads must never hold task teardown (or the outer batch)
        # open after the task deadline has fired.
        self._server.daemon_threads = True
        self._server.block_on_close = False
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._write_evidence()
        return self

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
        self._write_evidence()

    def __exit__(self, *_exc: object) -> None:
        self.close()
