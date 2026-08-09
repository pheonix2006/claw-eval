"""Bridge plugin generator — task.yaml -> OpenClaw tool plugin.

Phase 3 Wave 2 §6.3 — see ``docs/harness_design.md`` (§3.4 / §3.4a / §6.3).

Two-stage pipeline:

1. ``compile_plugin_source(task, plugin_id)`` -> ``str``
     Pure, side-effect-free. Returns the TypeScript source for the plugin's
     ``src/index.ts``. Only depends on the task and a generated plugin id, so
     unit tests can pin behaviour without spawning ``npm`` / ``openclaw``.

2. ``generate_and_install(task, case_dir, services_ctx, run_id=...)``
     -> ``BridgeHandle``
     The IO-heavy half: copies the static template, writes the rendered
     ``index.ts``, runs ``npm install`` + ``tsc`` + ``openclaw plugins
     install/enable`` under per-task isolated env (``OPENCLAW_STATE_DIR`` /
     ``OPENCLAW_HOME`` / ``HOME``), and returns a ``BridgeHandle`` whose
     ``cleanup()`` ``rm -rf``'s everything.

A context manager :func:`bridge_install` wraps step 2 with try/finally for
callers that want a ``with`` block.

Wave 2 carve-out: this module never invokes ``npm`` or ``openclaw`` in the
test suite. Tests exercise only ``compile_plugin_source`` and the schema
translator; the CLI commands are guarded by the host harness in later waves.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from ...runner.sandbox_dispatcher import (
    _ALWAYS_MEDIA_TOOLS as _DISPATCHER_ALWAYS_MEDIA_TOOLS,
    _CONDITIONAL_MEDIA_TOOLS as _DISPATCHER_CONDITIONAL_MEDIA_TOOLS,
    SandboxToolDispatcher,
)
from ...runner.sandbox_tools import SANDBOX_TOOL_NAMES
from .schema_translate import SchemaTranslationError, json_schema_to_typebox

if TYPE_CHECKING:
    from ...models.task import TaskDefinition
    from ...runner.services import ServiceManager


# Re-export the sandbox dispatcher's tool-name -> endpoint-path map under a
# stable name. ``SandboxToolDispatcher._PATH_MAP`` is the canonical source
# (used by claweval runs) — mirror it here so bridge plugin route resolution
# can never drift from the dispatcher's contract.
SANDBOX_ENDPOINTS: dict[str, str] = dict(SandboxToolDispatcher._PATH_MAP)

# Sandbox tools whose response can carry a ``frames`` array of base64 images:
# ReadMedia / BrowserScreenshot always, and Read conditionally (image/PDF). These
# need the *factory* tool form so the bridge can return native image content
# blocks; see :func:`_render_one_tool`. Derived from the dispatcher's own media
# sets so the bridge can never drift from the loop's frame-extraction contract.
# The generated handler still gates on ``Array.isArray(body.frames)`` at runtime,
# so a plain text ``Read`` (no frames) is unaffected.
_MEDIA_FRAME_TOOLS: frozenset[str] = (
    _DISPATCHER_ALWAYS_MEDIA_TOOLS | _DISPATCHER_CONDITIONAL_MEDIA_TOOLS
)

__all__ = [
    "BridgeHandle",
    "BridgeInstallError",
    "compile_plugin_source",
    "generate_and_install",
    "bridge_install",
]


# ---------------------------------------------------------------------------
# Public types


class BridgeInstallError(RuntimeError):
    """Raised when one of the npm / openclaw install steps fails.

    The original ``subprocess.CalledProcessError`` is chained via ``__cause__``
    so callers can still inspect ``stdout`` / ``stderr`` if they need to.
    Wraps it in a domain-specific type so OpenClawHarness can catch precisely
    bridge-install failures without swallowing unrelated errors.
    """


@dataclass
class BridgeHandle:
    """Lifecycle handle returned by :func:`generate_and_install`.

    ``plugin_id`` / ``traffic_log_path`` / ``plugin_dir`` are all ``None`` for
    the "empty tool set" early-return path (``task.tool_endpoints == []``).
    OpenClawHarness uses these as sentinels: ``plugin_id is None`` means
    "don't pass ``--extra-plugins`` and don't read the bridge log".

    ``case_state_dir`` and ``case_home_dir`` are *always* populated so the
    OpenClaw native runner can use them as its isolated ``OPENCLAW_STATE_DIR``
    / ``OPENCLAW_HOME`` regardless of whether a plugin was installed.

    ``cleanup`` is idempotent — repeated calls (e.g. from a context manager
    ``__exit__`` after the caller already called it explicitly) are no-ops.
    """

    case_state_dir: Path
    case_home_dir: Path
    plugin_id: str | None = None
    traffic_log_path: Path | None = None
    plugin_dir: Path | None = None
    # Internal tombstone — flipped by cleanup() to make it idempotent.
    _cleaned: bool = field(default=False, repr=False)

    def cleanup(self) -> None:
        """Best-effort ``rm -rf`` of all per-task scratch dirs.

        We deliberately do NOT call ``openclaw plugins disable / uninstall``:
        the plugin lives under the isolated state dir, so removing the dir
        wipes it cleanly. Trying to disable/uninstall from a half-installed
        state frequently returns "not found" and obscures the real error.
        """
        if self._cleaned:
            return
        self._cleaned = True
        for path in (self.plugin_dir, self.case_state_dir, self.case_home_dir):
            if path is None:
                continue
            # ``ignore_errors=True`` swallows missing-path and partial-cleanup
            # cases (e.g. a directory was already removed by another caller).
            # Idempotency is the contract, not "always logs a warning".
            shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Template + source compilation (pure)


# Resolve once at import time. ``Path(__file__).parent`` is the
# ``_openclaw_bridge`` package dir; the template tree sits next to this file.
_TEMPLATE_DIR = Path(__file__).resolve().parent / "plugin_template"
_INDEX_TS_TEMPLATE = _TEMPLATE_DIR / "src" / "index.ts.template"
_RECORDER_TS = _TEMPLATE_DIR / "src" / "recorder.ts"
_STATIC_FILES = ("package.json", "tsconfig.json")
_NPM_INSTALL_MAX_ATTEMPTS = 3
_NPM_INSTALL_RETRY_DELAY_S = 1.0
_NPM_INSTALL_RETRYABLE_MARKERS = (
    "econnreset",
    "network aborted",
    "network connectivity",
    "socket hang up",
    "etimedout",
    "eai_again",
    "enotfound",
)

# `plugin_id` must satisfy OpenClaw's plugin id pattern. We don't have an
# authoritative regex from the SDK, but the demo plugin uses lowercase
# alphanumerics + dashes; matching that is the safe bet.
_PLUGIN_ID_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


def derive_plugin_id(task_id: str, run_id: str) -> str:
    """Build the per-task / per-run plugin id.

    Format: ``claweval-bridge-<sanitized_task_id>-<sanitized_run_id>``.
    Strict sanitisation (anything outside ``[A-Za-z0-9_-]`` -> ``-``) keeps
    task ids with dots / slashes from breaking plugin install.
    """
    safe_task = _PLUGIN_ID_SAFE.sub("-", task_id).strip("-") or "task"
    safe_run = _PLUGIN_ID_SAFE.sub("-", run_id).strip("-") or "run"
    return f"claweval-bridge-{safe_task}-{safe_run}"


def _bridgeable_tools(task: "TaskDefinition") -> list[str]:
    """Names of every tool the bridge can route, in stable iteration order.

    Order rules (§3.4a):

    1. ``task.tools`` order first — task authors define tools in the order
       they expect the LLM to call them (and the design doc mandates this
       primary iteration order).
    2. Anything in ``task.tool_endpoints`` but absent from ``task.tools`` is
       appended afterwards in endpoint order. This covers the (rare) case
       of an endpoint without a matching ``tools[]`` spec.

    A tool is bridgeable if and only if:

    * its name is in ``SANDBOX_TOOL_NAMES`` (routed to the container's
      sandbox server), OR
    * it has a matching ``tool_endpoints[*]`` entry (routed to the host
      mock service).

    Tools that fail both checks are silently skipped — preflight (and the
    URL resolver) will refuse to install a plugin whose only tools have no
    reachable target, but a partially-targetable task is still useful.
    """
    endpoint_names = {ep.tool_name for ep in (task.tool_endpoints or [])}
    seen: set[str] = set()
    out: list[str] = []
    for spec in (task.tools or []):
        if spec.name in seen:
            continue
        if spec.name in SANDBOX_TOOL_NAMES or spec.name in endpoint_names:
            out.append(spec.name)
            seen.add(spec.name)
    # Endpoints not also in task.tools — covers tasks that declare an
    # endpoint without a tool spec (rare but supported per §3.4a).
    for ep in (task.tool_endpoints or []):
        if ep.tool_name in seen:
            continue
        out.append(ep.tool_name)
        seen.add(ep.tool_name)
    return out


def _resolve_tool_url(
    tool_name: str,
    task: "TaskDefinition",
    sandbox_url: str | None,
    host_gateway: str | None = None,
) -> tuple[str, str]:
    """Return ``(url, method)`` for a bridged tool.

    Routing rules per §3.4a:

    1. Tool is in ``SANDBOX_TOOL_NAMES`` -> route to the container's sandbox
       server (``{sandbox_url}{SANDBOX_ENDPOINTS[tool_name]}``). Raises if
       ``sandbox_url`` is missing — preflight should have caught this.
    2. Otherwise look up ``task.tool_endpoints`` for a matching ``tool_name``.
       Use its ``url`` + ``method``.
    3. Neither: raise ``SchemaTranslationError`` (preflight should have
       caught this too, but bridge generation is the last line of defence).

    ``host_gateway`` (bridge network mode — macOS compat): when set, mock
    service URLs (rule 2) have their host (``localhost`` / ``127.0.0.1``)
    rewritten to ``host_gateway`` (e.g. ``host.docker.internal``) so a
    bridge-networked container can still reach host mock services. The port
    and path are preserved. SANDBOX_TOOL routes (rule 1) are NOT rewritten —
    they already point at the in-container sandbox server. Default ``None`` =
    no rewrite (host network mode / Linux), URLs verbatim.
    """
    if tool_name in SANDBOX_TOOL_NAMES:
        if sandbox_url is None:
            raise SchemaTranslationError(
                f"tool {tool_name!r} requires sandbox server but sandbox_url not provided"
            )
        endpoint = SANDBOX_ENDPOINTS.get(tool_name)
        if not endpoint:
            raise SchemaTranslationError(
                f"tool {tool_name!r} is in SANDBOX_TOOL_NAMES but has no endpoint in SANDBOX_ENDPOINTS"
            )
        return (f"{sandbox_url.rstrip('/')}{endpoint}", "POST")

    for ep in (task.tool_endpoints or []):
        if ep.tool_name == tool_name:
            return (_rewrite_host(ep.url, host_gateway), ep.method or "POST")

    raise SchemaTranslationError(
        f"tool {tool_name!r} declared without endpoint"
    )


def _rewrite_host(url: str, host_gateway: str | None) -> str:
    """Rewrite a mock URL's host to ``host_gateway`` (bridge mode), preserving
    scheme/port/path. No-op when ``host_gateway`` is None or the host is not a
    loopback name."""
    if not host_gateway:
        return url
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    host = parts.hostname
    if host not in ("localhost", "127.0.0.1"):
        return url
    netloc = host_gateway
    if parts.port is not None:
        netloc = f"{host_gateway}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))



def compile_plugin_source(
    task: "TaskDefinition",
    plugin_id: str | None = None,
    *,
    sandbox_url: str | None = None,
    host_gateway: str | None = None,
) -> str | None:
    """Render the plugin ``src/index.ts`` source.

    Returns ``None`` when the task has no bridgeable tool — neither a
    ``tool_endpoints`` entry nor a ``task.tools`` entry whose name is in
    ``SANDBOX_TOOL_NAMES``. The caller is expected to short-circuit the
    whole install flow in that case (§6.3 step 1, "empty tool set early
    return"). This is a sentinel, not an empty string, so accidental
    ``len(source)`` checks don't mask the early return.

    When ``plugin_id`` is omitted, we synthesise a deterministic placeholder
    (``claweval-bridge-<task_id>-snapshot``) so the function stays pure /
    deterministic for tests. Real installs always pass an explicit
    ``plugin_id`` derived from the run id.

    ``sandbox_url`` (Wave 3-E §3.4a) — when present, ``SANDBOX_TOOL_NAMES``
    tools are routed to ``{sandbox_url}{SANDBOX_ENDPOINTS[name]}``; when
    absent, a SANDBOX_TOOLS-bearing task raises ``SchemaTranslationError``.
    HTTP mock-service tools (``task.tool_endpoints``) always use the host
    URL verbatim regardless of ``sandbox_url`` — the container runs with
    ``--network host`` so the same string resolves to the same socket.
    """
    bridgeable = _bridgeable_tools(task)
    if not bridgeable:
        return None

    if plugin_id is None:
        plugin_id = derive_plugin_id(task.task_id, "snapshot")

    tools_block = _render_tools_block(
        task, sandbox_url=sandbox_url, host_gateway=host_gateway
    )
    template = _INDEX_TS_TEMPLATE.read_text(encoding="utf-8")
    description = (
        f"Generated from task {task.task_id}. "
        f"Bridges {len(bridgeable)} tool(s) to claw-eval (mock services + sandbox server)."
    )
    return (
        template
        .replace("{{PLUGIN_ID}}", _js_string(plugin_id, strip_quotes=True))
        .replace("{{PLUGIN_DESCRIPTION}}", _js_string(description, strip_quotes=True))
        .replace("{{TOOLS_BLOCK}}", tools_block)
    )


def _render_tools_block(
    task: "TaskDefinition",
    *,
    sandbox_url: str | None = None,
    host_gateway: str | None = None,
) -> str:
    """Compile the bridgeable tool table into a comma-separated list of
    ``tool({...})`` calls.

    Iteration order matches :func:`_bridgeable_tools` — ``task.tools`` first,
    then any ``task.tool_endpoints`` not also present in ``task.tools``.
    Snapshot tests rely on this being stable.

    Each tool's URL is resolved via :func:`_resolve_tool_url`: SANDBOX_TOOL
    names route to ``{sandbox_url}{SANDBOX_ENDPOINTS[name]}``, HTTP mock
    services use the ``tool_endpoints`` URL verbatim.
    """
    spec_map = {spec.name: spec for spec in (task.tools or [])}
    endpoint_map = {ep.tool_name: ep for ep in (task.tool_endpoints or [])}

    rendered: list[str] = []
    for tool_name in _bridgeable_tools(task):
        spec = spec_map.get(tool_name)
        description = spec.description if spec else ""
        schema = spec.input_schema if spec else {}
        try:
            params_expr = json_schema_to_typebox(schema)
        except SchemaTranslationError:
            # Re-raise unchanged so preflight can catch the spec-mandated
            # exception type.
            raise
        url, method = _resolve_tool_url(tool_name, task, sandbox_url, host_gateway)
        # Cross-reference the endpoint method override (when the route came
        # from task.tool_endpoints) — _resolve_tool_url already returns the
        # right method, but keep this assertion for spec drift safety.
        ep = endpoint_map.get(tool_name)
        if ep is not None:
            method = ep.method or method
        rendered.append(
            _render_one_tool(
                tool_name=tool_name,
                description=description,
                params_expr=params_expr,
                url=url,
                method=method,
            )
        )
    return ",\n".join(rendered)


def _render_one_tool(
    *,
    tool_name: str,
    description: str,
    params_expr: str,
    url: str,
    method: str,
) -> str:
    """Render a single ``tool({...})`` clause.

    Kept as one template string (rather than a multi-line builder) so the
    generated TS is easy to diff against the design doc sample in §3.4a.

    Media tools (``ReadMedia`` / ``BrowserScreenshot``) are rendered via the
    *factory* form so the bridge can return native image content blocks — see
    :func:`_render_media_tool`. Every other tool keeps the plain ``execute``
    form whose return value OpenClaw stringifies into a text tool result.
    """
    if tool_name in _MEDIA_FRAME_TOOLS:
        return _render_media_tool(
            tool_name=tool_name,
            description=description,
            params_expr=params_expr,
            url=url,
            method=method,
        )
    # 4-space indent matches the surrounding `tools: (tool) => [...]` block.
    return f"""    tool({{
      name: {_js_string(tool_name)},
      description: {_js_string(description)},
      parameters: {params_expr},
      execute: async (params, _config, ctx) => {{
        const url = {_js_string(url)};
        const method = {_js_string(method.upper())};
        const started = Date.now();
        let status = -1;
        let body: unknown = null;
        let errMsg: string | undefined;
        try {{
          const resp = await fetch(url, {{
            method,
            headers: {{ "content-type": "application/json" }},
            body: JSON.stringify(params ?? {{}}),
          }});
          status = resp.status;
          const text = await resp.text();
          try {{
            body = text ? JSON.parse(text) : null;
          }} catch {{
            // Mock service returned non-JSON (or empty). Surface the raw text
            // so the grader can still see what came back.
            body = text;
          }}
        }} catch (e) {{
          errMsg = e instanceof Error ? e.message : String(e);
          body = {{ error: errMsg }};
        }}
        const finished = Date.now();
        recordCall({{
          toolCallId: ctx?.toolCallId ?? null,
          tool: {_js_string(tool_name)},
          url,
          method,
          request: params,
          status,
          response: body,
          startedAtMs: started,
          finishedAtMs: finished,
          durationMs: finished - started,
          ...(errMsg ? {{ error: errMsg }} : {{}}),
        }});
        return body;
      }},
    }})"""


def _render_media_tool(
    *,
    tool_name: str,
    description: str,
    params_expr: str,
    url: str,
    method: str,
) -> str:
    """Render a media tool (``ReadMedia`` / ``BrowserScreenshot``) clause.

    Why the *factory* form: ``defineToolPlugin``'s convenience ``execute`` path
    runs every return value through ``wrapToolPluginResult``, which
    ``JSON.stringify``s objects into a *single text block*. The sandbox server
    returns ``{frames: [{image_b64, mime_type}, ...]}``; stringified, those
    frames become inert base64 text that the runtime then truncates (~64KB),
    collapsing e.g. 8 sampled video frames down to one corrupt frame. The model
    is effectively blind on video.

    The factory form returns an ``AgentToolResult`` verbatim (no
    ``wrapToolPluginResult``), so we can emit one ``{type:"image"}`` content
    block per frame — exactly how the native ``SandboxDispatcher`` feeds frames
    to the loop. Base64 is stripped from the text summary to save tokens.

    Frame budget (ReadMedia only): we inject a compressed-frame contract
    (<=1280px, JPEG q60, <=8 frames) into the request. OpenClaw's
    openai-completions provider only forwards image parts from the *latest*
    user message and caps them at ``maxImageParts`` (DEFAULT_OPENAI_MAX_IMAGE_PARTS
    = 8); above that the whole image batch is dropped and the model goes blind.
    Capping to 8 uniformly-sampled, downscaled frames keeps every ReadMedia
    result deliverable. BrowserScreenshot / Read keep their own request bodies
    (their endpoints don't accept these fields).
    """
    if tool_name == "ReadMedia":
        req_prep = (
            "\n          // Mirror OpenClaw's openai-completions image contract: the"
            "\n          // provider forwards only the latest user message's image parts"
            "\n          // and caps them at maxImageParts (default 8) — more are dropped"
            "\n          // whole, blinding the model. Request downscaled JPEG + a <=8"
            "\n          // frame budget so each ReadMedia result stays deliverable."
            "\n          const _req = { ...(params ?? {}) };"
            "\n          if (_req.screen_size == null) _req.screen_size = \"1280x1280\";"
            "\n          if (_req.frame_format == null) _req.frame_format = \"jpeg\";"
            "\n          if (_req.frame_quality == null) _req.frame_quality = 60;"
            "\n          if (_req.frame_budget == null) _req.frame_budget = 8;"
        )
        req_body = "_req"
    else:
        req_prep = ""
        req_body = "params ?? {}"
    return f"""    tool({{
      name: {_js_string(tool_name)},
      description: {_js_string(description)},
      parameters: {params_expr},
      factory: (_ctx) => ({{
        name: {_js_string(tool_name)},
        label: {_js_string(tool_name)},
        description: {_js_string(description)},
        parameters: {params_expr},
        execute: async (toolCallId: string, params: any, _signal?: unknown, _onUpdate?: unknown) => {{
          const url = {_js_string(url)};
          const method = {_js_string(method.upper())};
          const started = Date.now();
          let status = -1;
          let body: any = null;
          let errMsg: string | undefined;{req_prep}
          try {{
            const resp = await fetch(url, {{
              method,
              headers: {{ "content-type": "application/json" }},
              body: JSON.stringify({req_body}),
            }});
            status = resp.status;
            const text = await resp.text();
            try {{
              body = text ? JSON.parse(text) : null;
            }} catch {{
              body = text;
            }}
          }} catch (e) {{
            errMsg = e instanceof Error ? e.message : String(e);
            body = {{ error: errMsg }};
          }}
          const finished = Date.now();
          recordCall({{
            toolCallId: toolCallId ?? null,
            tool: {_js_string(tool_name)},
            url,
            method,
            request: params,
            status,
            response: body,
            startedAtMs: started,
            finishedAtMs: finished,
            durationMs: finished - started,
            ...(errMsg ? {{ error: errMsg }} : {{}}),
          }});
          // Expand frames[] into native image content blocks so the multimodal
          // model actually SEES them (see _render_media_tool docstring).
          if (body && typeof body === "object" && Array.isArray(body.frames)) {{
            const frames = body.frames.filter(
              (f: any) => f && typeof f.image_b64 === "string" && f.image_b64.length > 0,
            );
            if (frames.length > 0) {{
              const summary: any = {{}};
              for (const k of Object.keys(body)) {{
                if (k !== "frames") summary[k] = body[k];
              }}
              summary.frame_count = frames.length;
              const content: any[] = [{{ type: "text", text: JSON.stringify(summary) }}];
              for (const f of frames) {{
                content.push({{
                  type: "image",
                  data: f.image_b64,
                  mimeType: typeof f.mime_type === "string" && f.mime_type ? f.mime_type : "image/png",
                }});
              }}
              return {{ content, details: summary }};
            }}
          }}
          // No frames (error or empty): preserve JSON-as-text behavior.
          if (typeof body === "string") {{
            return {{ content: [{{ type: "text", text: body }}], details: body }};
          }}
          return {{ content: [{{ type: "text", text: JSON.stringify(body ?? null) }}], details: body }};
        }},
      }}),
    }})"""


def _js_string(value: str, *, strip_quotes: bool = False) -> str:
    """Render a Python string as a JavaScript/JSON string literal.

    Using ``json.dumps`` here handles every escape (newlines, quotes, control
    chars, non-ASCII) the JS lexer cares about, which is critical because
    task descriptions in the YAML routinely contain Chinese, quotes, and
    embedded backticks.

    ``strip_quotes`` is for template substitution into already-quoted
    placeholders (e.g. ``id: "{{PLUGIN_ID}}"``): we still escape but drop the
    surrounding double quotes.
    """
    encoded = json.dumps(value, ensure_ascii=False)
    if strip_quotes:
        # ``json.dumps`` always emits the leading + trailing ``"``; safe to
        # peel them since ``ensure_ascii=False`` preserves the content.
        return encoded[1:-1]
    return encoded


# ---------------------------------------------------------------------------
# Install + lifecycle (IO)


def generate_and_install(
    task: "TaskDefinition",
    case_dir: Path,
    services_ctx: "ServiceManager | None" = None,
    *,
    run_id: str | None = None,
    skip_subprocess: bool = False,
    sandbox_url: str | None = None,
    host_gateway: str | None = None,
    container: "Any | None" = None,
) -> BridgeHandle:
    """Compile ``task`` into a plugin, install it into an isolated profile.

    Returns a populated ``BridgeHandle``. On failure, ``cleanup()`` is called
    before re-raising as ``BridgeInstallError``.

    Parameters
    ----------
    task:
        Task whose tools drive plugin generation. Both ``task.tools`` (for
        SANDBOX_TOOLS routing) and ``task.tool_endpoints`` (for HTTP mock
        service routing) feed the rendered plugin source.
    case_dir:
        Per-task scratch dir. Must be writable; this function creates it if
        missing. The plugin sources land at ``case_dir / "bridge_plugin"``,
        the isolated OpenClaw state at ``case_dir / "openclaw_state"``, etc.
    services_ctx:
        Currently unused (mock services run on host with stable URLs from
        ``task.tool_endpoints``). Accepted to keep the §3.4 call signature
        stable when later waves need it.
    run_id:
        Used to disambiguate the plugin id when the same task runs multiple
        times in parallel. If omitted, derived from ``case_dir`` basename.
    skip_subprocess:
        Escape hatch for tests / dry runs — skip the ``npm install`` + tsc +
        ``openclaw plugins ...`` chain. Source files are still generated.
    sandbox_url:
        Wave 3-E §3.4a. When provided, ``SANDBOX_TOOL_NAMES`` tools route to
        ``{sandbox_url}{SANDBOX_ENDPOINTS[name]}`` (the container's sandbox
        server). When ``None``, a SANDBOX_TOOLS-bearing task raises
        ``SchemaTranslationError`` at source-render time. Backward-compat:
        host-mode callers (Wave 3-D) omit this and only get HTTP mock
        services routed.
    container:
        Wave 3-E §3.7. When provided, the npm / openclaw install chain runs
        **inside** the container via ``docker exec`` (so the plugin gets
        installed under the container's OpenClaw state dir). Volume mounts
        wire the host plugin dir to the container at the same path so source
        files materialised on host are visible inside the container. When
        ``None``, install runs on host (Wave 3-D behaviour).

    Notes
    -----
    ``services_ctx`` is accepted, not used in this wave. The recorder uses
    ``task.tool_endpoints`` URLs verbatim (mock services bind to localhost
    on stable ports), so no extra context is needed.
    """
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    case_state_dir = case_dir / "openclaw_state"
    case_home_dir = case_dir / "openclaw_home"
    # Even for the empty-tool early return we still need the isolation dirs —
    # the OpenClaw native runner uses them as OPENCLAW_STATE_DIR / HOME.
    case_state_dir.mkdir(parents=True, exist_ok=True)
    case_home_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — empty bridgeable tool set early return (§6.3).
    # NB: ``not task.tool_endpoints`` is the WRONG check post-Wave 3-E — a
    # task with only SANDBOX_TOOLS has no ``tool_endpoints`` but still has
    # routable tools. Use the unified ``_bridgeable_tools`` helper instead.
    bridgeable = _bridgeable_tools(task)
    if not bridgeable:
        return BridgeHandle(
            case_state_dir=case_state_dir,
            case_home_dir=case_home_dir,
            plugin_id=None,
            traffic_log_path=None,
            plugin_dir=None,
        )

    effective_run_id = run_id or case_dir.name
    plugin_id = derive_plugin_id(task.task_id, effective_run_id)

    plugin_dir = case_dir / "bridge_plugin"
    traffic_log_path = case_dir / "bridge_traffic.jsonl"

    handle = BridgeHandle(
        case_state_dir=case_state_dir,
        case_home_dir=case_home_dir,
        plugin_id=plugin_id,
        traffic_log_path=traffic_log_path,
        plugin_dir=plugin_dir,
    )

    try:
        # Step 2 — render plugin sources (sandbox_url flows into TS).
        _materialise_plugin_dir(
            task, plugin_dir, plugin_id, sandbox_url=sandbox_url, host_gateway=host_gateway
        )
        # Pre-create the traffic log so plugin code can append to it without
        # an extra ``mkdir`` round-trip on first call.
        traffic_log_path.touch(exist_ok=True)

        if skip_subprocess:
            return handle

        # Step 4 — install + enable under the isolated env (§3.4a).
        env = {
            **os.environ,
            "OPENCLAW_STATE_DIR": str(case_state_dir),
            "OPENCLAW_HOME": str(case_home_dir),
            "HOME": str(case_home_dir),
            "CLAWEVAL_BRIDGE_LOG": str(traffic_log_path),
        }
        _run_install_chain(
            plugin_dir=plugin_dir,
            plugin_id=plugin_id,
            env=env,
            container=container,
        )
    except BaseException as exc:  # noqa: BLE001 — failure path needs to clean up *everything*
        handle.cleanup()
        if isinstance(exc, BridgeInstallError):
            raise
        if isinstance(exc, subprocess.CalledProcessError):
            raise BridgeInstallError(
                f"bridge install failed running {exc.cmd!r}: "
                f"rc={exc.returncode}, stderr={exc.stderr!r}"
            ) from exc
        raise

    return handle


def _materialise_plugin_dir(
    task: "TaskDefinition",
    plugin_dir: Path,
    plugin_id: str,
    *,
    sandbox_url: str | None = None,
    host_gateway: str | None = None,
) -> None:
    """Lay out the plugin directory: template files + generated ``index.ts``.

    Mirrors the structure of ``scratch/openclaw_tool_probe/demo_plugin/`` — a
    bare-minimum Node project (``package.json``, ``tsconfig.json``, ``src/``)
    plus the OpenClaw plugin manifest. The manifest is regenerated each run
    so it carries the per-task plugin id and tool list (OpenClaw uses
    ``contracts.tools`` for capability discovery).
    """
    if plugin_dir.exists():
        # A leftover from a crashed run — start clean. ``ignore_errors=False``
        # surfaces permission issues fast rather than producing a half-merged
        # tree on top.
        shutil.rmtree(plugin_dir)
    src_dir = plugin_dir / "src"
    src_dir.mkdir(parents=True)

    # Static files first — same bytes for every plugin.
    for fname in _STATIC_FILES:
        shutil.copy2(_TEMPLATE_DIR / fname, plugin_dir / fname)
    shutil.copy2(_RECORDER_TS, src_dir / "recorder.ts")

    # Rendered index.ts.
    source = compile_plugin_source(
        task, plugin_id=plugin_id, sandbox_url=sandbox_url, host_gateway=host_gateway
    )
    assert source is not None, "early-return path should have been handled by caller"
    (src_dir / "index.ts").write_text(source, encoding="utf-8")

    # Plugin manifest. We re-author rather than copy so the id/name and
    # tool contracts stay in sync with the generated source.
    bridgeable = _bridgeable_tools(task)
    manifest = {
        "id": plugin_id,
        "name": "ClawEval Bridge",
        "description": (
            f"Generated bridge plugin for task {task.task_id}. "
            f"Routes {len(bridgeable)} tool(s) to claw-eval (mock services + sandbox server)."
        ),
        "version": "0.1.0",
        "configSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
        "activation": {"onStartup": False},
        "contracts": {
            "tools": list(bridgeable),
        },
    }
    (plugin_dir / "openclaw.plugin.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _run_install_chain(
    *,
    plugin_dir: Path,
    plugin_id: str,
    env: dict[str, str],
    container: "Any | None" = None,
) -> None:
    """Run npm install + tsc + openclaw plugins install/enable, in order.

    Each step is a separate subprocess call so the first failure aborts the
    chain with a usable ``CalledProcessError``. We don't stream output —
    captured stdout/stderr is attached to the error object for the caller
    to log.

    When ``container`` is provided (Wave 3-E §3.7), every step runs
    **inside** that container via ``docker exec``. This requires the host
    ``plugin_dir`` to be visible at the same absolute path inside the
    container (achieved via volume mount in ``SandboxRunner``). The bridge
    install env vars (OPENCLAW_STATE_DIR / HOME / ...) are passed through
    so the plugin lives under the isolated state dir inside the container.
    """
    if container is None:
        def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
            # capture_output=True so the failure mode (BridgeInstallError)
            # surfaces useful stderr without us paying for a logger here.
            subprocess.run(
                cmd,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
    else:
        # Bridge-only env vars to forward to docker exec. We include the
        # OpenClaw isolation vars + the bridge log location; everything else
        # in ``env`` (PATH inside the container is already correct, HOME we
        # explicitly override) we leave to the image defaults.
        forwarded_keys = (
            "OPENCLAW_STATE_DIR",
            "OPENCLAW_HOME",
            "HOME",
            "CLAWEVAL_BRIDGE_LOG",
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
        )
        forwarded_env = {k: env[k] for k in forwarded_keys if k in env}

        def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
            # docker exec -e KEY=VAL ... -w <cwd> <container_id> <cmd...>
            # cwd is host-side here; mounted-at-same-path means it works
            # inside the container too.
            exec_cmd: list[str] = ["docker", "exec"]
            for k, v in forwarded_env.items():
                exec_cmd.extend(["-e", f"{k}={v}"])
            if cwd is not None:
                exec_cmd.extend(["-w", str(cwd)])
            container_id = container.id if hasattr(container, "id") else str(container)
            exec_cmd.append(container_id)
            exec_cmd.extend(cmd)
            subprocess.run(
                exec_cmd,
                check=True,
                capture_output=True,
                text=True,
            )

    def _is_retryable_npm_install_error(
        cmd: list[str], exc: subprocess.CalledProcessError
    ) -> bool:
        if list(cmd)[:2] != ["npm", "install"]:
            return False
        details = "\n".join(
            part for part in (exc.stderr, exc.stdout) if isinstance(part, str) and part
        ).lower()
        return any(marker in details for marker in _NPM_INSTALL_RETRYABLE_MARKERS)

    def _run_with_retry(cmd: list[str], *, cwd: Path | None = None) -> None:
        attempt = 1
        while True:
            try:
                _run(cmd, cwd=cwd)
                return
            except subprocess.CalledProcessError as exc:
                if (
                    attempt >= _NPM_INSTALL_MAX_ATTEMPTS
                    or not _is_retryable_npm_install_error(cmd, exc)
                ):
                    raise
                time.sleep(_NPM_INSTALL_RETRY_DELAY_S * attempt)
                attempt += 1

    # 1) npm install — pulls TypeBox + the openclaw peer (the latter typically
    #    via the dev dep in package.json; in container builds we expect
    #    openclaw to also be globally installed so plugins build/install can
    #    find the CLI).
    _run_with_retry(["npm", "install"], cwd=plugin_dir)

    # 2) tsc -> dist/. Plugin SDK consumes the compiled JS via the
    #    ``openclaw.extensions`` field in package.json.
    _run(["npx", "tsc", "-p", "tsconfig.json"], cwd=plugin_dir)

    # 3) Bundle/validate using the OpenClaw CLI (writes plugin metadata).
    _run(
        ["openclaw", "plugins", "build", "--entry", "./dist/index.js"],
        cwd=plugin_dir,
    )

    # 4) Install into the isolated state dir.
    _run(["openclaw", "plugins", "install", str(plugin_dir)])

    # 5) Enable so it shows up in the agent's tool list. ``--id`` form is
    #    safer than relying on auto-detection of the just-installed package.
    _run(["openclaw", "plugins", "enable", plugin_id])


# ---------------------------------------------------------------------------
# Context manager


@contextmanager
def bridge_install(
    task: "TaskDefinition",
    case_dir: Path,
    services_ctx: "ServiceManager | None" = None,
    *,
    run_id: str | None = None,
    skip_subprocess: bool = False,
    sandbox_url: str | None = None,
    container: "Any | None" = None,
) -> Iterator[BridgeHandle]:
    """Context manager wrapper around :func:`generate_and_install`.

    Always calls ``handle.cleanup()`` on exit, whether the body succeeded or
    raised. The handle itself is yielded so the caller can read
    ``plugin_id`` / ``traffic_log_path`` / state dirs inside the ``with``
    block.
    """
    handle = generate_and_install(
        task,
        case_dir,
        services_ctx,
        run_id=run_id,
        skip_subprocess=skip_subprocess,
        sandbox_url=sandbox_url,
        container=container,
    )
    try:
        yield handle
    finally:
        handle.cleanup()


# ---------------------------------------------------------------------------
# Misc helpers — exposed for the trace adapter / unit tests


def parse_traffic_log(log_path: Path | str) -> list[dict[str, Any]]:
    """Read the JSONL traffic log into a list of records.

    Kept here (rather than in the trace adapter) so the generator owns both
    sides of the recorder contract — schema changes touch one file only.
    Skips blank / malformed lines defensively; the adapter then decides how
    to surface partial reads.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    out: list[dict[str, Any]] = []
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # Malformed line — skip but don't crash. The trace adapter logs
            # a warning when the count of records < count of toolCalls in
            # session.jsonl, so this stays observable.
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out
