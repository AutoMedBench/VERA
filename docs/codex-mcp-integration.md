# Policy-bound Codex/MCP integration

`eva_agent.deployment.CodexMCPDeployment` is the fail-closed seam between the
schema-preserving MCP catalog, the `evamed-codex` stdio server, and the shared
Codex runtime. It does not change an EvaMed policy, tool, skill, rubric, or
evidence schema.

## What the preflight proves

One deployment binds all of the following to one candidate:

- the literal BLAKE3 of one source policy file;
- one verified `PolicyCatalogBinding` and its candidate ID;
- the unchanged `search_skills` and `load_skill` contracts from one
  `MCPToolCatalog`;
- one MCP configuration name and one initialized server identity;
- one actor `CodexThreadOptions`, including any OpenAI-compatible model
  provider configuration.

The asynchronous preflight starts the configured process with
`EVAMED_MCP_ACTOR_MODE=1`, sends only `initialize` and `tools/list`, and closes
stdin. Provider credentials are not forwarded to this probe. It requires the
listed tool set to be exactly the selected policy tools plus
`search_skills`/`load_skill`. For every tool it checks name, description,
canonical `inputSchema` bytes, per-schema BLAKE3, catalog BLAKE3, public
visibility, read-only/mutating classification, parallel-safety metadata, and
MCP server/tool identity. Extra control-plane, judge-only, private, duplicate,
or tampered tools fail before a Codex thread can start.

The policy file is read before and after the child process and must remain
byte-identical. Same-name tools with different schemas remain in separate
candidate-scoped processes and bindings; there is no global tool-name merge.

The success receipt commits the exact transport catalog and sidecar metadata,
and explicitly records:

```text
probe_methods = [initialize, tools/list]
provider_calls_made = 0
mcp_tool_calls_made = 0
```

This follows Codex's documented local stdio MCP model, required-server setting,
and per-server `enabled_tools` allowlist. See the
[official OpenAI MCP documentation](https://developers.openai.com/codex/mcp/).

## Building the executable thread configuration

```python
from eva_agent.codex_runtime import CodexRole, CodexSandbox, CodexThreadOptions
from eva_agent.deployment import CodexMCPDeployment, StdioMCPServerSpec

actor = CodexThreadOptions(
    role=CodexRole.STRONG_ACTOR,
    model="gemini-3.1-pro-preview",
    provider="gemini",
    cwd="/absolute/candidate/workspace",
    sandbox=CodexSandbox.WORKSPACE_WRITE,
    config={
        "model_providers": {
            "gemini": {
                "base_url": "https://gateway.example/v1",
                "env_key": "GEMINI_API_KEY",
            }
        }
    },
)

server = StdioMCPServerSpec(
    config_name="evamed",
    server_name="evamed-codex",
    server_version="0.1.0",
    command=("/absolute/path/to/python", "/absolute/path/to/evamed_mcp.py", "--stdio"),
    cwd="/absolute/path/to/EVA-Agent",
    environment={
        "EVAMED_MCP_REGISTRY_FACTORY": "trusted_runtime:build_registry",
        "EVAMED_STAGE": "S3",
        "PYTHONPATH": "/absolute/path/to/trusted/runtime",
    },
)

deployment = CodexMCPDeployment(
    binding=policy_catalog_binding,
    skill_catalog=skill_mcp_catalog,
    source_policy_path="/absolute/candidate/policy.json",
    server=server,
    actor_thread=actor,
)
prepared = await deployment.preflight()

# This object is now suitable for the official SDK-backed runtime:
handle = await codex_runtime.start_thread(prepared.thread_options)
```

The provider name is intentionally opaque to this seam. GPT-5.6 Sol, Gemini,
Qwen, DeepSeek, or another configured Codex model provider receives the same
candidate-bound MCP contract. Provider differences cannot change tool schemas.

## Live in-memory execution bridge

`eva_agent.codex_pipeline.TurnMCPBridgeFactory` closes the provider-turn gap
without serializing Python handlers into a second process. For each actor or
judge turn it creates a mode-`0700` private directory, a mode-`0600` Unix
socket, and a 256-bit nonce. The Codex-launched stdio child is a small proxy;
the candidate-bound `ParallelToolRuntime` (actor) or `JudgeWorkspaceTools`
(judge) remains in the campaign process.

The supplied `temp_root` is mandatory and must be absolute, non-symlink,
process-UID-owned, and mode `0700`. The factory creates and removes an actual
tempfile probe and rejects any resulting socket path of 108 bytes or more
before a provider turn. Production uses the short stable
`/tmp/eva-mcp-v2-<uid>` root; its creation/reopen checks live in the concrete
deployment factory.

```python
from pathlib import Path
import sys
from eva_agent.codex_pipeline import (
    CodexOpus5AgentJudge,
    CodexRolloutAdapter,
    TurnMCPBridgeFactory,
)

turn_mcp = TurnMCPBridgeFactory(
    proxy_python=Path(sys.executable).resolve(),
    proxy_script=Path(
        "/absolute/EVA-Agent/src/eva_agent/codex_pipeline/turn_mcp_proxy.py"
    ),
    temp_root=verified_private_short_root,
    # Match the candidate runtime's bound (transport supports 1..256; default 64).
    maximum_parallel_calls=64,
)

actor = CodexRolloutAdapter(
    options_factory=actor_base_options,  # must not pre-mount mcp_servers
    runner=codex_runner,
    turn_mcp_factory=turn_mcp,
)
judge = CodexOpus5AgentJudge(
    options_factory=judge_base_options,
    runner=codex_runner,
    model_id="claude-opus-5",
    turn_mcp_factory=turn_mcp,
)
```

The exact turn offers become that process's complete `tools/list` result. MCP
calls execute once through the same in-memory bridge that populates the
pipeline `ToolTrace`; the receipt is matched to the returned structured
execution commitments before the socket is removed. Judge calls execute once
against the immutable `EvidenceBundle` snapshots and are not replayed after
the provider turn. Independent calls are coalesced only while no earlier
result has been released. Transport width is explicitly bounded in `[1, 256]`
(default `64`); production should pass the exact lower actor/judge runtime bound.
The underlying runtime remains authoritative for parallel safety and keeps
mutating or non-parallel-safe calls exclusive.

This transport does not invent missing handlers. Production composition must
resolve a handler-backed registry from the candidate's signed promoted-policy
and construction pointers. A schema-only legacy candidate is an explicit
production blocker. Same-name schema variants remain in separate per-candidate
registries and per-turn MCP servers; a global merged `ToolRegistry` is not a
valid production input.

The campaign integration port is deliberately small:

- resolve the selected EVA candidate and its immutable source candidate into
  one handler-backed `CandidateExecutionBinding`;
- construct that candidate's pipeline with `binding.tool_registry` (never a
  worker-global registry);
- pass the shared structural `run_once` runner and one
  `TurnMCPBridgeFactory` to both Codex adapters;
- keep actor/judge base `CodexThreadOptions.config` free of `mcp_servers`;
- commit the resolver's complete policy/binding inventory digest and coverage
  in the deployment recipe, rather than hashing a merged schema catalog.

Missing policy bytes, source construction/workspace evidence, executable
handlers, or an inventory commitment are production blockers. They may not be
replaced with schema-only handlers.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest -q \
  tests/test_codex_mcp_integration.py tests/test_turn_mcp_bridge.py
```

The focused suite uses the real `evamed-codex` stdio process for preflight and
a Codex-compatible fake app-server that launches the real per-turn MCP proxy.
It makes no provider request or production-ledger write.
