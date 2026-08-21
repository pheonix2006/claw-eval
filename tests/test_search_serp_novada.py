"""Unit tests for the Novada SERP backend (sync + async) and error propagation.

These tests mock ``requests`` so they never hit the network. They lock in the
fix for the async (``/request`` + Bearer) endpoint and the rule that business
errors must surface as an ``error`` key rather than being swallowed into an
empty result set.

Run::

    .venv/bin/python -m pytest tests/test_search_serp_novada.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "mock_services" / "web_real" / "search_serp.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("search_serp_under_test", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


serp = _load_module()


class _FakeResp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


# --------------------------------------------------------------------------
# _extract_async_organic — structural navigation + business errors
# --------------------------------------------------------------------------

def _async_payload(organic):
    return {"data": {"data": {"json": [{"rest": {"organic": organic}}]}}}


def test_async_extract_happy_path():
    organic = [{"title": "T", "link": "https://x", "description": "snip"}]
    out, err = serp._extract_async_organic(_async_payload(organic))
    assert err is None
    assert out == organic


def test_async_extract_empty_is_not_error():
    out, err = serp._extract_async_organic(_async_payload([]))
    assert err is None
    assert out == []


def test_async_extract_official_direct_list_with_organic_results():
    organic = [{"title": "T", "link": "https://x", "snippet": "snip"}]
    out, err = serp._extract_async_organic(
        [{"spider_code": 200, "rest": {"organic_results": organic}}]
    )
    assert err is None
    assert out == organic


def test_async_extract_business_error_code():
    out, err = serp._extract_async_organic({"code": 401, "msg": "invalid key"})
    assert out == []
    assert err and "401" in err and "invalid key" in err


def test_async_extract_inner_business_error_code():
    out, err = serp._extract_async_organic(
        {"code": 0, "data": {"code": 400, "msg": "serp returns empty", "data": None}}
    )
    assert out == []
    assert err and "missing 'data'" in err and "serp returns empty" in err


def test_async_extract_spider_error_code():
    out, err = serp._extract_async_organic(
        [{"spider_code": 503, "rest": {"organic": []}}]
    )
    assert out == []
    assert err and "spider error code=503" in err


def test_async_extract_missing_structure():
    out, err = serp._extract_async_organic({"data": {"data": {}}})
    assert out == []
    assert err and "json" in err


# --------------------------------------------------------------------------
# search_serp — async path end-to-end (mocked transport)
# --------------------------------------------------------------------------

def test_search_async_maps_items(monkeypatch):
    monkeypatch.setenv("SERP_PROVIDER", "novada_async")
    monkeypatch.setenv("SERP_DEV_KEY", "k")
    organic = [
        {"title": "A", "link": "https://a", "description": "da", "date": "2026"},
        {"title": "B", "url": "https://b", "snippet": "db"},
    ]
    captured = {}

    def fake_post(url, headers=None, data=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["data"] = data
        return _FakeResp(200, _async_payload(organic))

    monkeypatch.setattr(serp.requests, "post", fake_post)
    res = serp.search_serp("python", num=5)

    assert res["status"] == 200
    assert res["error"] is None
    assert res["output"] == [
        {"title": "A", "link": "https://a", "snippet": "da", "date": "2026", "query": "python"},
        {"title": "B", "link": "https://b", "snippet": "db", "date": "", "query": "python"},
    ]
    # endpoint + auth + scraper pairing
    assert captured["url"] == serp._DEFAULT_ASYNC_URL
    assert captured["headers"]["Authorization"] == "Bearer k"
    assert captured["data"]["scraper_name"] == "google.com"
    assert captured["data"]["scraper_id"] == "google_search"


def test_search_async_business_error_propagates(monkeypatch):
    monkeypatch.setenv("SERP_PROVIDER", "novada_async")
    monkeypatch.setattr(
        serp.requests, "post",
        lambda *a, **k: _FakeResp(200, {"code": 403, "msg": "quota exceeded"}),
    )
    res = serp.search_serp("python")
    assert res["output"] == []
    assert res["status"] == 200
    assert res["error"] and "403" in res["error"]


def test_search_async_http_error_propagates(monkeypatch):
    monkeypatch.setenv("SERP_PROVIDER", "novada_async")
    monkeypatch.setattr(
        serp.requests, "post",
        lambda *a, **k: _FakeResp(429, None, text="Too Many Requests"),
    )
    res = serp.search_serp("python")
    assert res["output"] == []
    assert res["status"] == 429
    assert res["error"] and "429" in res["error"]


def test_search_transport_exception_propagates(monkeypatch):
    monkeypatch.setenv("SERP_PROVIDER", "novada_async")

    def boom(*a, **k):
        raise ConnectionError("dns fail")

    monkeypatch.setattr(serp.requests, "post", boom)
    res = serp.search_serp("python")
    assert res["status"] == -1
    assert res["output"] == []
    assert res["error"] and "ConnectionError" in res["error"]


# --------------------------------------------------------------------------
# search_serp — sync (legacy) path still works
# --------------------------------------------------------------------------

def test_search_sync_legacy_path(monkeypatch):
    monkeypatch.setenv("SERP_PROVIDER", "novada_sync")
    payload = {"data": {"organic_results": [
        {"title": "L", "url": "https://l", "description": "ds"},
    ]}}
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        return _FakeResp(200, payload)

    monkeypatch.setattr(serp.requests, "get", fake_get)
    res = serp.search_serp("hello")
    assert res["status"] == 200
    assert res["error"] is None
    assert res["output"][0]["link"] == "https://l"
    assert captured["url"] == serp._DEFAULT_SYNC_URL


def test_search_sync_business_error_propagates(monkeypatch):
    # Legacy endpoint answers HTTP 200 with a wrapped business error when a
    # new-style dashboard key is used — must surface, not silently empty.
    monkeypatch.setenv("SERP_PROVIDER", "novada_sync")
    monkeypatch.setattr(
        serp.requests, "get",
        lambda *a, **k: _FakeResp(200, {"code": 402, "msg": "Api Key error"}),
    )
    res = serp.search_serp("hello")
    assert res["output"] == []
    assert res["status"] == 200
    assert res["error"] and "402" in res["error"]
