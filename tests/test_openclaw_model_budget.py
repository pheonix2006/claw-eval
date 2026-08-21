from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from claw_eval.harnesses._openclaw_model_budget import (
    OpenClawModelBudgetProxy,
    _target_url,
)


class _UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        body = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _AnthropicUpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    paths: list[str] = []

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:
        type(self).paths.append(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        body = b'{"type":"message","content":[{"type":"text","text":"ok"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_target_url_treats_configured_base_as_complete_api_root() -> None:
    assert _target_url(
        "https://api.openai.example/v1", "/v1/chat/completions"
    ) == "https://api.openai.example/v1/chat/completions"
    assert _target_url(
        "https://api.z.ai/api/coding/paas/v4", "/v1/chat/completions"
    ) == "https://api.z.ai/api/coding/paas/v4/chat/completions"
    assert _target_url(
        "https://api.example", "/v1/chat/completions"
    ) == "https://api.example/v1/chat/completions"


def test_model_budget_proxy_forwards_stream_and_rejects_n_plus_one(tmp_path) -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    evidence = tmp_path / "model-budget.json"
    target = f"http://127.0.0.1:{upstream.server_address[1]}/v1"

    try:
        with OpenClawModelBudgetProxy(
            target_base_url=target,
            max_completions=2,
            evidence_path=evidence,
        ) as proxy:
            for _ in range(2):
                with httpx.stream(
                    "POST",
                    f"{proxy.base_url}/chat/completions",
                    json={"model": "m", "stream": True},
                    timeout=5,
                ) as response:
                    assert response.status_code == 200
                    assert "[DONE]" in "".join(response.iter_text())

            rejected = httpx.post(
                f"{proxy.base_url}/chat/completions",
                json={"model": "m", "stream": True},
                timeout=5,
            )
            assert rejected.status_code == 400
            assert rejected.json()["error"]["code"] == "claw_eval_max_turns_exceeded"
            assert proxy.completion_count == 2
            assert proxy.rejected_count == 1
            assert proxy.exhausted is True
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["max_completions"] == 2
    assert payload["completion_count"] == 2
    assert payload["rejected_count"] == 1
    assert [row["decision"] for row in payload["requests"]] == [
        "forward",
        "forward",
        "reject_max_turns",
    ]
    serialized = evidence.read_text(encoding="utf-8")
    assert "Authorization" not in serialized
    assert "api_key" not in serialized


def test_anthropic_model_budget_counts_messages_but_not_count_tokens(tmp_path) -> None:
    _AnthropicUpstreamHandler.paths = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _AnthropicUpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    evidence = tmp_path / "anthropic-model-budget.json"
    target = f"http://127.0.0.1:{upstream.server_address[1]}"

    try:
        with OpenClawModelBudgetProxy(
            target_base_url=target,
            max_completions=2,
            evidence_path=evidence,
            provider_transport="anthropic-messages",
        ) as proxy:
            assert not proxy.base_url.endswith("/v1")
            counted = httpx.post(
                f"{proxy.base_url}/v1/messages/count_tokens",
                json={"model": "m", "messages": []},
                timeout=5,
            )
            assert counted.status_code == 200
            for _ in range(2):
                response = httpx.post(
                    f"{proxy.base_url}/v1/messages",
                    json={"model": "m", "max_tokens": 16, "messages": []},
                    timeout=5,
                )
                assert response.status_code == 200

            rejected = httpx.post(
                f"{proxy.base_url}/v1/messages",
                json={"model": "m", "max_tokens": 16, "messages": []},
                timeout=5,
            )
            assert rejected.status_code == 400
            assert proxy.completion_count == 2
            assert proxy.rejected_count == 1
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    assert _AnthropicUpstreamHandler.paths == [
        "/v1/messages/count_tokens",
        "/v1/messages",
        "/v1/messages",
    ]
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["provider_transport"] == "anthropic-messages"
    assert [row["decision"] for row in payload["requests"]] == [
        "forward_non_completion",
        "forward",
        "forward",
        "reject_max_turns",
    ]
