"""
Search SERP — raw web skill (Novada)

Two provider backends, selected by env ``SERP_PROVIDER``:

* ``novada_sync``  (legacy, default) — GET https://scraperapi.novada.com/search
      ?engine=google&api_key=<SERP_DEV_KEY>&q=...
      organic results live in  data.organic_results

* ``novada_async`` (current dashboard keys) — POST https://scraper.novada.com/request
      Authorization: Bearer <SERP_DEV_KEY>
      Content-Type: application/x-www-form-urlencoded
      organic results live in  data.data.json[0].rest.organic

New-style dashboard keys ONLY work with the async endpoint; the legacy
``/search`` + ``api_key`` path returns business errors for them. Those errors
used to be swallowed into an empty result set ("HTTP 200, total 0, error
none"), which silently poisons benchmark failure analysis. We now surface
them explicitly via an ``error`` key.

Env:
    SERP_PROVIDER        novada_sync | novada_async   (default novada_sync)
    SERP_API_URL         override endpoint (else provider default)
    SERP_DEV_KEY         Novada key
    NOVADA_SCRAPER_NAME  async scraper_name (default google.com)
    NOVADA_SCRAPER_ID    async scraper_id   (default google_search)

Input:  query (str), timeout (int), num (int), start (int)
Output: {"status": <int>, "output": <list[dict]>, "error": <str|None>}
"""

import os
import re
import requests

_DEFAULT_SYNC_URL = "https://scraperapi.novada.com/search"
_DEFAULT_ASYNC_URL = "https://scraper.novada.com/request"


def _provider() -> str:
    return os.getenv("SERP_PROVIDER", "novada_sync").strip().lower()


def _dev_key() -> str:
    return os.getenv("SERP_DEV_KEY", "YOUR_API_KEY")


def _api_url(provider: str) -> str:
    explicit = os.getenv("SERP_API_URL")
    if explicit:
        return explicit
    return _DEFAULT_ASYNC_URL if provider == "novada_async" else _DEFAULT_SYNC_URL


def _detect_language(query: str) -> tuple[str, str]:
    if re.search(r"[一-鿿]", query):
        return "zh", "cn"
    return "en", "us"


def _save_raw(raw_save_path: str | None, text: str) -> None:
    if not raw_save_path:
        return
    os.makedirs(os.path.dirname(raw_save_path) or ".", exist_ok=True)
    with open(raw_save_path, "w", encoding="utf-8") as f:
        f.write(text)


def _business_error(payload: dict) -> str | None:
    """Return a message if Novada wrapped a business error in code/msg.

    Both endpoints answer HTTP 200 even on auth/quota failures, packing the
    real status into a top-level ``code`` (e.g. 402 "Api Key error"). Treat
    anything outside the success codes as an error so it is not swallowed.
    """
    if not isinstance(payload, dict):
        return None
    code = payload.get("code")
    if code not in (None, 0, "0", 200, "200"):
        msg = payload.get("msg") or payload.get("message") or ""
        return f"Novada business error code={code} msg={msg}"[:300]
    return None


def _extract_async_organic(payload) -> tuple[list, str | None]:
    """Extract current envelope or official direct-list Novada responses.

    Returns (organic_list, error). A genuinely empty result set is
    (``[]``, ``None``) — *not* an error. A structural mismatch or a Novada
    business error code is ([], "<message>").
    """
    if isinstance(payload, list):
        node = payload
    elif isinstance(payload, dict):
        # Dashboard responses wrap the scraper payload in two business-status
        # envelopes.  Check both: HTTP 200 and outer code=0 do not imply that
        # the inner scrape succeeded.
        err = _business_error(payload)
        if err:
            return [], err
        outer_data = payload.get("data")
        if not isinstance(outer_data, dict):
            return [], f"Novada async missing 'data' in response: {str(payload)[:200]}"
        err = _business_error(outer_data)
        if err:
            # Preserve the historical error prefix/payload consumed by the
            # benchmark's frozen Novada recovery evidence parser.
            return [], f"Novada async missing 'data' in response: {str(payload)[:200]}"
        inner_data = outer_data.get("data")
        if not isinstance(inner_data, dict):
            return [], f"Novada async missing inner 'data' in response: {str(payload)[:200]}"
        node = inner_data.get("json")
    else:
        return [], f"Novada async unsupported payload: {str(payload)[:200]}"

    if not isinstance(node, list) or not node:
        return [], f"Novada async empty 'json' list: {str(payload)[:200]}"
    first = node[0]
    spider_code = first.get("spider_code") if isinstance(first, dict) else None
    if spider_code not in (None, 0, "0", 200, "200"):
        return [], f"Novada spider error code={spider_code}: {str(first)[:200]}"
    rest = first.get("rest") if isinstance(first, dict) else None
    if not isinstance(rest, dict):
        return [], f"Novada async missing 'rest': {str(first)[:200]}"
    err = _business_error(rest)
    if err:
        return [], f"Novada result business error: {err}"
    organic = rest.get("organic")
    if organic is None:
        # The current live dashboard envelope uses ``organic`` while Novada's
        # public Google Scraper output preview names it ``organic_results``.
        organic = rest.get("organic_results")
    if organic is None:
        return [], f"Novada async missing 'organic': {str(rest)[:200]}"
    if not isinstance(organic, list):
        return [], "Novada async 'organic' is not a list"
    return organic, None


def _map_item(item: dict, query: str) -> dict:
    return {
        "title": item.get("title", ""),
        "link": item.get("link") or item.get("url", ""),
        "snippet": item.get("description") or item.get("snippet", ""),
        "date": item.get("date", ""),
        "query": query,
    }


def _search_sync(query, timeout, num, start, raw_save_path) -> dict:
    hl, gl = _detect_language(query)
    params = {
        "engine": "google",
        "api_key": _dev_key(),
        "q": query,
        "num": str(min(max(num, 1), 10)),
        "hl": hl,
        "gl": gl,
        "start": str(max(start, 1)),
        "fetch_mode": "static",
        "no_cache": "true",
    }
    resp = requests.get(_api_url("novada_sync"), params=params, timeout=timeout)
    if resp.status_code == 200:
        _save_raw(raw_save_path, resp.text)
    if resp.status_code != 200:
        return {
            "status": resp.status_code,
            "output": [],
            "error": f"Novada sync HTTP {resp.status_code}: {resp.text[:200]}",
        }
    payload = resp.json()
    err = _business_error(payload)
    if err:
        return {"status": resp.status_code, "output": [], "error": err}
    data = payload.get("data", {})
    results = [_map_item(item, query) for item in data.get("organic_results", [])]
    return {"status": resp.status_code, "output": results, "error": None}


def _search_async(query, timeout, num, start, raw_save_path) -> dict:
    hl, gl = _detect_language(query)
    headers = {
        "Authorization": f"Bearer {_dev_key()}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    # NOTE: scraper_name MUST be google.com when scraper_id=google_search.
    # Novada's own examples sometimes show amazon.com here, but that pairing
    # fails for Google search.
    data = {
        "scraper_name": os.getenv("NOVADA_SCRAPER_NAME", "google.com"),
        "scraper_id": os.getenv("NOVADA_SCRAPER_ID", "google_search"),
        "scraper_errors": "true",
        "q": query,
        "device": "desktop",
        "json": "1",
        "render_js": "false",
        "no_cache": "false",
        "ai_overview": "false",
        "domain": "google.com",
        "country": gl,
        "hl": hl,
        "safe": "off",
    }
    resp = requests.post(_api_url("novada_async"), headers=headers, data=data, timeout=timeout)
    if resp.status_code == 200:
        _save_raw(raw_save_path, resp.text)
    if resp.status_code != 200:
        return {
            "status": resp.status_code,
            "output": [],
            "error": f"Novada async HTTP {resp.status_code}: {resp.text[:200]}",
        }
    try:
        payload = resp.json()
    except ValueError:
        return {
            "status": resp.status_code,
            "output": [],
            "error": f"Novada async non-JSON body: {resp.text[:200]}",
        }
    organic, err = _extract_async_organic(payload)
    if err:
        return {"status": resp.status_code, "output": [], "error": err}
    results = [_map_item(item, query) for item in organic]
    return {"status": resp.status_code, "output": results, "error": None}


def search_serp(
    query: str,
    timeout: int = 20,
    num: int = 10,
    start: int = 1,
    raw_save_path: str | None = None,
) -> dict:
    """Search Google via Novada (sync or async) and return extracted results.

    Args:
        query: Search query string.
        timeout: Request timeout in seconds.
        num: Number of results (1-10).
        start: 1-based result offset.
        raw_save_path: Optional path to dump the raw HTTP body.

    Returns:
        dict with keys:
            status (int): HTTP status code, or -1 on transport error.
            output (list[dict]): Result dicts (title, link, snippet, date, query).
            error (str | None): Business/transport error, or None on success.
                An empty ``output`` with ``error is None`` means the search
                genuinely returned zero results.
    """
    provider = _provider()
    try:
        if provider == "novada_async":
            return _search_async(query, timeout, num, start, raw_save_path)
        return _search_sync(query, timeout, num, start, raw_save_path)
    except Exception as e:
        return {
            "status": -1,
            "output": [],
            "error": f"{provider} request failed: {type(e).__name__}: {str(e)[:200]}",
        }


if __name__ == "__main__":
    import json

    result = search_serp("Python web scraping", num=3)
    print(
        f"provider={_provider()}  status={result['status']}  "
        f"count={len(result['output'])}  error={result.get('error')}"
    )
    print(json.dumps(result["output"], indent=2, ensure_ascii=False)[:1000])
