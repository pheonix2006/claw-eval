// Bridge traffic recorder — one JSONL line per tool call.
//
// The Python-side trace adapter consumes this file. Schema is documented in
// harness_design.md §3.5; in short, each line is a JSON object with:
//
//   toolCallId   — OpenClaw SDK callID, used to align with session.jsonl
//   tool         — tool name (matches task.tools[*].name)
//   url, method  — actual HTTP endpoint hit (from task.tool_endpoints[*])
//   request      — params the LLM passed in (after TypeBox validation)
//   status       — HTTP status code from the mock service, or -1 on transport error
//   response     — parsed JSON body, or the raw string when JSON parsing fails
//   startedAtMs / finishedAtMs — wall-clock bounds used only for cross-worker
//                  operational ordering and degradation supervision
//   durationMs   — wall-clock time spent in fetch+parse
//   error        — string, present only when fetch itself threw
//
// Concurrency: OpenClaw awaits tools serially within a turn (no parallel
// fan-out, since generator.py never sets `parallel: true`), so an unguarded
// appendFileSync is safe. If that ever changes, swap to a per-PID lock.

import { appendFileSync } from "node:fs";

export interface BridgeRecord {
  toolCallId: string | null;
  tool: string;
  url: string;
  method: string;
  request: unknown;
  status: number;
  response: unknown;
  startedAtMs: number;
  finishedAtMs: number;
  durationMs: number;
  error?: string;
}

// The Python generator injects the log path through the environment at
// `openclaw agent` startup, so plugin code stays static. If the var is unset
// (e.g. someone runs the plugin outside the bridge flow) we degrade to a
// no-op write and surface the misconfiguration through stderr — this is
// better than crashing the whole agent loop.
const LOG_PATH = process.env.CLAWEVAL_BRIDGE_LOG ?? "";

export function recordCall(entry: BridgeRecord): void {
  if (!LOG_PATH) {
    // eslint-disable-next-line no-console
    console.warn(
      "[claweval-bridge] CLAWEVAL_BRIDGE_LOG not set; dropping tool record for",
      entry.tool
    );
    return;
  }
  try {
    appendFileSync(LOG_PATH, JSON.stringify(entry) + "\n", "utf8");
  } catch (err) {
    // Logging must never break the tool call — surface and continue.
    // eslint-disable-next-line no-console
    console.warn("[claweval-bridge] failed to append record:", err);
  }
}
