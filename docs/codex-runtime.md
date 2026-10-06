# `evamed-codex` runtime

`eva_agent.codex_runtime` is a compatibility adapter around the official
[`openai-codex` Python SDK](https://service.example.invalid/docs/codex-sdk). The SDK
starts the local [Codex app-server](https://service.example.invalid/docs/app-server)
and speaks its typed JSON-RPC protocol. EvaMed no longer needs a second custom
OpenAI-style agent loop for this execution path.

The same runtime serves weak, middle, and strong actors and the Opus agent
judge. Model and provider names are passed through; they are not hard-coded by
cohort. The existing `ProviderRouter` remains the temporary deployment fallback
until callers migrate, but it is not called from this runtime.

## Boundary and lifecycle

Production composition is deliberately small:

```python
from eva_agent.codex_runtime import (
    CodexRole, CodexRuntime, CodexSandbox, CodexThreadOptions,
    CodexTurnInput, OpenAICodexBackend,
)

async with CodexRuntime(OpenAICodexBackend()) as runtime:
    thread = await runtime.start_thread(CodexThreadOptions(
        role=CodexRole.STRONG_ACTOR,
        model="gpt-5.6-sol",
        provider="openai",
        cwd="/absolute/verified/sandbox",
        sandbox=CodexSandbox.WORKSPACE_WRITE,
        config={"mcp_servers": {"evamed": {"command": "..."}}},
    ))
    receipt = await runtime.run_turn(
        thread,
        CodexTurnInput(public_text="Complete the verified task."),
    )
```

`resume_thread(upstream_thread_id, options)` resumes an app-server thread while
issuing a new Eva UUID runtime handle. Both start and resume force
`ApprovalMode.deny_all`. Only `read-only` and `workspace-write` are available;
the adapter intentionally offers no full-access preset.

The SDK is lazy-loaded. Tests inject `CodexBackendPort` or `CodexSdkBindings`,
so they start no process and make no provider request. Production deployments
can use the SDK's pinned CLI or set `CodexLaunchOptions.codex_bin` to the
operator-approved local CLI. Launch env and config values are never copied into
receipts.

The runtime adds no candidate semaphore, retry loop, or artificial model lane.
The campaign orchestrator owns outer concurrency. Codex owns the inner turn and
may issue independent tools in parallel; the receipt derives
`max_parallelism_observed` from overlapping `item/started` and
`item/completed` lifecycles.

The schema-preserving pipeline accepts a 64-call read-only frontier. That is a
capability ceiling, not eager thread allocation: only calls actually emitted in
one frontier create work, and every mutating or non-`parallel_safe` definition
remains exclusive. A live regression test synchronizes 32 read-only calls in
one frontier, so the widened path is verified execution rather than metadata.

## High-width app-server pool

`PersistentCodexRuntimeRunner` multiplexes fresh threads through one long-lived
app-server. `ShardedPersistentCodexRuntimeRunner` removes that local process as
a single scheduling and failure domain: it starts a fixed number of independent
persistent runners in parallel and atomically assigns each accepted turn to a
currently least-loaded shard, rotating equal-load ties. It does not queue,
throttle, or retry a failed turn on another shard. Calls accepted before close
are drained before all app-servers close in parallel.

```python
from eva_agent.codex_runtime import (
    CodexRuntime,
    OpenAICodexBackend,
    ShardedPersistentCodexRuntimeRunner,
)

pool = ShardedPersistentCodexRuntimeRunner(
    lambda: CodexRuntime(OpenAICodexBackend()),
    shard_count=64,
)
with pool:
    # Synchronous campaign workers may call pool.run_once(...) concurrently.
    ...
```

The production shard count is a versioned deployment input, not an admission
criterion. Start a real campaign with one shard for the exact candidate/tool
canary, then increase shards and candidate workers only after measuring
provider error rate, latency, and retained receipt integrity. A provider-free
host smoke test on the GB300 node initialized and closed 64 pinned SDK
app-servers in 2.08 seconds with no residual process; this establishes local
control-plane capacity but does not substitute for a real provider canary.

## Immutable EvaMed schemas

Codex is transport, never a schema compiler. An offered tool comes from the
already-verified candidate/sandbox policy through
`CodexToolOffer.from_definition`. Its original `name`, `description`, and
`input_schema`/`parameters` values are retained exactly. MCP's required
`inputSchema` spelling changes only the outer protocol key; its value is
canonical-deep-equal and digest-locked.

Tool FQNs, server names, visibility, permissions, parallel annotations, and
discovery hints live in a separate sidecar catalog or receipt. They are never
inserted into the canonical tool, skill, rubric, policy, or evidence schema.
`offered_tool_schema_blake3` commits the exact ordered canonical tool catalog.
Tests reopen the canonical bytes for ordinary Eva tools and the unmodified
`SkillCatalog` `search_skills`/`load_skill` definitions.

Arbitrary original medical skills may be selected; there is no fixed list or
count. Each `CodexSkill` supplies its existing ID, name, absolute path, and
content BLAKE3. The SDK receives the original name/path as a `SkillInput`; the
receipt records selected IDs and commits the exact mount catalog. The runtime
does not read or rewrite `SKILL.md`.

`CodexRolloutAdapter.skills_factory` is the candidate-scoped production hook.
It resolves the verified public skills for one rollout and mounts them through
the SDK in the same turn; the unchanged `search_skills`/`load_skill` MCP tools
remain available for progressive discovery. Duplicate IDs or names fail before
the provider boundary.

`VerifiedActorSkillCatalog` is the production factory for the pinned medical
skill snapshot. At startup it reopens and verifies the external manifest,
source README, all 24 unique `SKILL.md` bodies, byte counts, stage scopes, and
content digests. Each actor receives the repo-owned `stage-rollout` skill plus
only the legacy skills explicitly allowed for its stage. E2E receives the
native workflow skill and does not silently broaden the legacy catalog, whose
source declares no E2E grants. The resulting mount-catalog BLAKE3 belongs in
the deployment recipe.

## Privacy and receipts

Actor and judge inputs are structurally separate. Any actor call containing
`judge_only_context`, judge-only tools, or reserved private-reference fields
fails before backend execution. Judge-only context is sent only on a judge
thread. Private rubric/reference material must never be placed in an actor's
public fields, base instructions, MCP server, or mounted skills.

Each terminal turn produces `eva.codex-turn-receipt.v1` with:

- Eva UUID receipt/thread/turn IDs and opaque upstream thread/turn IDs;
- role, model, provider, sandbox, terminal status, and final response;
- ordered, BLAKE3-sealed structured notifications with hidden reasoning and
  echoed user input removed;
- structured tool lifecycles, arguments, results, MCP server/tool FQNs, and
  observed same-turn parallelism;
- usage notifications, selected skill IDs/catalog commitment, exact offered
  tool catalog commitment, and logical-input commitment;
- SDK and app-server versions.

`verify_codex_turn_receipt` reopens the event, tool-call, and outer receipt
commitments. Raw input and config values are intentionally absent.

## MCP and plugin connection plan

No plugin files are changed by this foundation. A deployment should connect
EvaMed in four independently verifiable steps:

1. Expose only the verified stage-permitted data-plane tools over an MCP server.
   Preserve each original schema verbatim and handle independent `tools/call`
   requests concurrently within the policy limit.
2. Expose the existing `search_skills` and `load_skill` definitions unchanged.
   Direct skill mounts remain explicit SDK `SkillInput` items with catalog
   provenance; the two mechanisms are complementary.
3. Configure a public actor MCP server separately from the read-only committed
   snapshot server used by the judge. Never mount the judge server on actor
   threads. Preflight server status and compare the offered catalog digest with
   the verified sandbox policy before provider execution.
4. If packaged later, let a Codex plugin bundle the MCP server and medical
   skills, while keeping stable MCP `server/tool` FQNs. Record the offered
   schema digest and selected skill IDs in the runtime receipt. Dynamic tools
   should remain experimental until their schema and identity verification is
   equivalent to MCP.

The source policy verifier, rather than the transport adapter, remains the
authority deciding which tools and skills are eligible for a candidate.
