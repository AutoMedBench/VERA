# EvaMed Codex plugin

`plugins/evamed-codex` is the repository-native Codex surface for EVA-Agent.
It contributes four native workflow skills and a local stdio MCP server that
can directly call the existing EvaMed medical/workspace handlers. No model or
provider call occurs merely by loading, listing, validating, or testing it.

## Two planes

The five plugin-owned control tools are `capabilities`, `search`, `load`,
`validate_work_order`, and `verify_receipts`. They progressively reveal public
workflow contracts; they are not migrated medical tools.

The data plane is compatibility-first:

- `search_skills` and `load_skill` are created by the existing
  `eva_agent.harness.SkillCatalog`. Their names, descriptions, argument schemas,
  parameters, and result contracts are unchanged.
- Candidate tools come from a BLAKE3-bound sandbox policy and handlers come
  from an injected existing `ToolRegistry`. The bridge only maps the canonical
  `input_schema`/`parameters` key to MCP `inputSchema`. Deep equality and a
  canonical schema BLAKE3 are checked before exposure.
- Candidate policies currently contain six names—`materialize_plan`,
  `retrieve_frozen_evidence`, `materialize_evidence_selection`, `execute_code`,
  `submit_results`, and occasionally `reopen_s4_artifact`—but there are multiple
  exact schemas for some names. The selected candidate policy, not a global
  name lookup, determines the exact description/schema/digest. Variants are
  never merged by name across candidates.
- If only a policy schema is available, `tools/list` marks the tool unavailable
  in `_meta`; `tools/call` returns a typed `handler_unavailable` failure. It
  never fabricates an execution result.

Discovery, availability, stage visibility, concurrency, and permission hints
live in MCP annotations or `_meta.evamed`. They do not modify a canonical
EvaMed input or output schema.

## Skills

| Native workflow skill | Runtime boundary |
| --- | --- |
| `sandbox-construction` | Pinned episode, answer-free projection, one compiled rubric binding |
| `stage-rollout` | Weak/middle/strong exactly once, bounded multi-tool groups, no retry |
| `workspace-agent-judge` | Opus 5 over immutable context plus read-only workspace evidence |
| `trajectory-sft` | Signed-admitted strong trajectory only; atomic multi-tool examples |

The pinned H000–H007 harness contains 174 legacy `SKILL.md` occurrences but 24
unique byte strings. The source declares `license: other` and private research
data, with no redistributable license in the snapshot. The plugin therefore
does not silently relicense or bundle those bodies. Instead,
`references/legacy-skill-manifest.v1.json` records all 24 content BLAKE3 values,
byte lengths, exact catalog stage filters, canonical source paths, all 174
provenance paths, H000–H007 origins, and pinned revision `79dd2a31f5f`.

At startup the bridge reopens all 24 external canonical files and the pinned
README, verifies literal bytes, and then mounts them through the unchanged
`SkillCatalog` handlers. Missing or changed bytes fail closed. Set
`EVAMED_LEGACY_SKILL_ROOT` when the pinned source is not at the standard sibling
checkout path. Only the four new workflow skills are native plugin skills, so
Codex validation never requires rewriting legacy frontmatter.

## Runtime configuration

`.mcp.json` starts `scripts/evamed_mcp.py --stdio`. In this source checkout the
script re-executes in the repository `.venv`; a packaged environment must have
EVA-Agent and its BLAKE3 dependency installed.

| Variable | Contract |
| --- | --- |
| `EVAMED_MCP_POLICY_PATH` | One selected candidate's verified policy JSON |
| `EVAMED_MCP_POLICY_BLAKE3` | BLAKE3 of the policy's literal bytes; required with the path |
| `EVAMED_MCP_REGISTRY_FACTORY` | Synchronous `module:function` returning an existing harness/pipeline `ToolRegistry`, or `{registry, workspace, stage, source_ref}` |
| `EVAMED_STAGE` | Registry projection stage; `S1`–`S5` or `E2E` |
| `EVAMED_MCP_MAX_PARALLEL` | Global MCP call bound, default 16, valid 1–256 |
| `EVAMED_MCP_ACTOR_MODE=1` | Data-only actor surface; requires a policy binding |
| `EVAMED_LEGACY_SKILL_ROOT` | Optional pinned legacy skill snapshot root |

In actor mode, `tools/list` is exactly the selected policy's tools plus
`search_skills` and `load_skill`. Control tools are neither listed nor callable.
This lets policy preflight compare the actor surface without plugin helpers.

Pipeline registry handlers require an injected sandbox workspace. Harness
handlers already accept the canonical argument mapping. Synchronous handlers
run off the event loop; asynchronous handlers are awaited directly. The adapter
does not depend on provider-specific private types.

## Parallel calls and failure semantics

The stdio reader assigns every JSON-RPC request its own task and correlates
out-of-order responses by request ID. A write lock prevents interleaved JSON,
and a bounded gate applies backpressure. Calls whose source ToolRegistry marks
`parallel_safe=true` may overlap. An unsafe call receives an exclusive frontier:
it waits for active calls to drain and blocks new parallel calls until complete.

Every error is returned once with its type, `retry_performed: false`, and a
BLAKE3 failure receipt. Provider, judge, timeout, malformed-output, and handler
failures remain observable failures; the bridge never retries or substitutes a
candidate.

## Data invariants

- One immutable sandbox binds one compiled domain × `S1`–`S5`/`E2E` rubric.
  The agent judge and rewarder consume that same compiled object.
- Actor context contains only a rubric binding. Private rubric items, answers,
  and judge-only references stay outside tools/list, tool arguments, public
  receipts, and SFT output.
- Weak, middle, and strong each receive exactly one trajectory. Infrastructure
  failures are quarantined without semantic score or replacement.
- UUIDs identify runtime objects; BLAKE3 commits content and receipts.
- A pipeline recommendation is not admission. SFT requires independently
  verified signed admission and supervisor-transition evidence.
- Same-turn independent multi-tool calls remain one atomic SFT target.

## Offline verification

Run from the EVA-Agent repository root; these commands make no network or
provider calls:

```bash
export CODEX_SKILLS_ROOT=/path/to/codex/skills

python3 plugins/evamed-codex/scripts/build_legacy_skill_manifest.py --verify
python3 plugins/evamed-codex/scripts/evamed_mcp.py --self-test

for skill in plugins/evamed-codex/skills/*; do
  python3 "$CODEX_SKILLS_ROOT/.system/skill-creator/scripts/quick_validate.py" "$skill"
done

python3 "$CODEX_SKILLS_ROOT/.system/plugin-creator/scripts/validate_plugin.py" \
  plugins/evamed-codex

.venv/bin/python -m pytest -q tests/test_evamed_codex_plugin.py
```

The focused suite checks manifest provenance, literal legacy bytes, schema
deep-equality/digest locks, per-candidate same-name variants, exact actor tool
sets, actual tools/list and tools/call forwarding, stage filtering, absence of
private rubric keys, unavailable-handler failure, and overlapping versus
exclusive stdio calls. This is repo source; no personal marketplace is created.
