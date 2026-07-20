from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from claw_eval.harnesses._openclaw_model_budget import OpenClawModelBudgetProxy


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
