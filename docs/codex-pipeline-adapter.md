# Codex-to-pipeline adapter

`eva_agent.codex_pipeline` connects the official asynchronous Codex app-server
runtime to the synchronous `VerifiableDataPipeline` ports. It is a compatibility
layer: no EvaMed tool, skill, rubric, policy, evidence, `ProviderRollout`, or
`JudgeAssessment` schema is changed.

## Actor boundary

`CodexRolloutAdapter` implements the existing `RolloutProvider.run` port. For a
given benchmark episode and cohort it atomically consumes one trajectory claim,
starts one fresh ephemeral Codex thread, runs exactly one app-server turn, and
never resumes or retries it. A failed attempt stays consumed in the adapter and
is raised as `CodexPipelineError`; its terminal `CodexTurnReceipt`, when one
exists, remains attached to the exception for infrastructure quarantine. A
campaign ledger remains the durable source of truth across process restarts.

The adapter derives its offers from the `RolloutRequest.available_tools` emitted
by the existing `ToolRegistry`. `name`, `description`, and `parameters` become a
`CodexToolOffer` without modifying the parameters value. Before returning, it
reopens the receipt and verifies:

- terminal status, fresh-thread marker, role, cohort, model, provider, sandbox,
  and redacted config-key identity;
- exact offered-tool names and the BLAKE3 of the canonical tool catalog;
- MCP server/tool identity, object arguments, completed lifecycle, and nested
  event/tool/turn receipts;
- observable parallel call frontiers reconstructed from app-server
  `item/started` and `item/completed` events;
- every MCP result carries a bridge receipt for the **same** existing
  `ToolResult` already committed to the pipeline `ToolTrace`.

The returned object is the unchanged `ProviderRollout`. Its policy events keep
atomic same-turn tool-call IDs and observations. `safe_metadata` contains the
complete redacted `eva.codex-turn-receipt.v1`, all ordered Codex events and tool
receipts, the reconstructed call groups, and `semantic_retry_count: 0`.

The pipeline creates the real filesystem sandbox. The options factory receives
that absolute root plus a candidate-scoped `CodexToolExecutionBridge`. Its local
MCP transport must call `bridge.execute_group(...)` once for each observed Codex
frontier and return each resulting `structured_content` value verbatim as MCP
`result.structuredContent`. That bridge call is the one semantic execution: it
invokes the existing `ToolRuntimePort.execute` and therefore populates the same
`ToolTrace` that the pipeline later seals. The adapter only reopens the returned
bridge receipts; it never replays a tool or duplicates a workspace mutation.

The final binding requires exact equality among bridge-issued call IDs, Codex
MCP observations, `ToolTrace.results`, declared IDs, joined IDs, arguments,
names, outputs, and BLAKE3 receipts, with `retry_count == 0`. The projected
assistant/tool event IDs are the existing pipeline call IDs, so
`PipelineVerifier` can prove that visible policy calls and evidence results are
the same executions. A direct plugin handler or subprocess that cannot share
this bridge is not production-compatible: a tool-using receipt from that path
lacks bridge evidence and fails before `ProviderRollout` is returned.

## Synchronous lifecycle and parallelism

`SyncCodexTurnRunner` creates a fresh runtime and local asyncio loop for one
invocation, opens it, starts one thread, runs one turn, and closes it in an async
context manager even on failure. It rejects use from a thread that already owns
an active event loop; async callers must move the synchronous pipeline call to a
worker thread. It has no module semaphore, singleton client, lock around provider
work, or global throttle. Independent weak, middle, and strong worker threads can
therefore overlap, while the campaign and app-server retain their own bounds.

For high-width campaigns, both adapters accept an injected
`CodexTurnRunnerPort`. `PersistentCodexRuntimeRunner` implements that port with
one long-lived app-server and a fresh isolated Codex thread for every `run_once`.
The adapter adds no semaphore or retry around it. Actor MCP bindings remain
per-candidate: the actor options factory receives the current bridge on every
call and must place the corresponding local transport endpoint in that thread's
configuration. The caller owns the persistent runner lifecycle and drains it at
shutdown. Omitting `runner` preserves the per-call `SyncCodexTurnRunner` path.

## Opus 5 workspace AgentJudge

`CodexOpus5AgentJudge` implements the existing synchronous `AgentJudge` port and
requires an Opus 5 model identity. It passes the policy-visible evidence
projection publicly and passes the judge-only reference plus the **exact existing
compiled rubric object** through `judge_only_context`. Its four offers come
directly from `JudgeWorkspaceTools.response_api_schemas()` and are marked
judge-only, read-only, and parallel-safe in side metadata; the canonical input
schemas are unchanged.

The configured MCP handler must return this plugin-local structured observation
inside the ordinary MCP `result.structuredContent` envelope:

```json
{
  "status": "completed",
  "output": {},
  "error_code": null,
  "inspected_evidence_refs": ["workspace:after:path"],
  "content_inspection": true,
  "evidence_bundle_blake3": "<64 lowercase hex>"
}
```

This is not trusted on sight. After the Codex turn finishes, the adapter groups
the captured calls by observed overlap and replays every call against a fresh
`JudgeWorkspaceTools` instance over the immutable committed `EvidenceBundle`.
The MCP observation must equal the local replay byte-semantically. Any extra,
missing, stale, or fabricated observation fails closed.

Admission to semantic scoring additionally requires:

- at least one successfully completed `workspace_read` or `workspace_search`;
- at least one score citation returned by that inspected workspace evidence;
- no citation to an uninspected `workspace:*` reference;
- one row, in order, for every compiled rubric item;
- a score from that item's exact compiled partial-credit levels;
- hard-gate truth recomputed from the same compiled item definitions;
- a nonempty summary and an exact terminal JSON shape.

The result is the existing `JudgeAssessment` unchanged. Its existing
`JudgeAgentTrace` contains replay-verified results, content-inspection refs,
atomic parallel groups, and retry count zero in the exact event shape required
by the existing verifier. The full redacted Codex receipt is retained outside
that immutable schema and is available as `receipt_for(judgment_id)` for the
calling campaign to persist beside the assessment. Reward computation therefore
continues to use the existing compiled rubric and verifier without a translated
judge schema.

## Construction

A production runtime factory normally returns a new
`CodexRuntime(OpenAICodexBackend(CodexLaunchOptions(...)))`. The actor factory is
called as `runtime_factory(bridge)` on the default ephemeral path; its options
factory is called as `options_factory(request, cwd, exact_offers, bridge)`. The
judge options factory receives `(request, exact_offers)`. When a shared runner is
injected, either runtime factory may be `None`; the shared runner's already-open
runtime is used while the options factory still receives the current actor
bridge. All returned `CodexThreadOptions` must describe a fresh ephemeral thread
and preserve the exact offers, role, model, provider, sandbox, and cwd.
Credential values stay in environment-backed Codex/MCP configuration and are
not copied into receipts.

Provider, app-server, terminal-status, identity, schema, MCP observation, and
judge-output failures are never converted to a semantic zero and never retried.
The calling campaign should retain the `CodexPipelineError` and optional receipt
as infrastructure evidence under its existing quarantine policy.

## Offline verification

The focused tests use only injected fake backends. They prove parallel actor
overlap, exactly-once in-process claims, same-execution nonempty `ToolTrace`
binding, one workspace mutation, a real `PipelineVerifier` pass, complete
receipt projection, active-event-loop rejection, shared persistent-runner reuse,
Opus 5 read/search enforcement, local observation replay, exact rubric mapping,
citation enforcement, and zero retry:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest -q tests/test_codex_pipeline_adapter.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest -q
```

Neither command starts app-server, invokes a provider, or performs network I/O.
