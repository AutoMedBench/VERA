---
name: stage-rollout
description: Execute or inspect a stage-wise or E2E EvaMed ability-separation rollout. Use when a constructed sandbox must run once on weak, middle, and strong cohorts with bounded parallel tool use; do not use for retries or admission.
---

# Stage Rollout

Run the sandbox through `eva_agent.pipeline.VerifiableDataPipeline`, using `eva_agent.harness.EvaMedHarness` when the task needs the Responses-style multi-turn tool loop.

- Require exactly three distinct model targets: `weak`, `middle`, and `strong`. Run exactly one fresh trajectory for each, concurrently when capacity permits.
- Keep the same immutable sandbox bytes and same compiled rubric binding across all three workspaces. The rollout sees only the rubric ID/version/digest, not private items or judge-only references.
- Keep `parallel_tool_calls` enabled. Execute independent same-turn calls as one bounded group; preserve call IDs, group membership, observations, workspace before/after state, and maximum observed parallelism. Respect dependency edges and serialize unsafe writes.
- Set retry count to zero. A provider, judge, timeout, or malformed-output failure is infrastructure quarantine: retain its typed evidence, create no semantic score, and never rerun or substitute that candidate.
- Compute each reward from the exact compiled rubric object passed to the judge. Retain raw per-item scores and weak/middle/strong deltas; a perfect staircase is not required.
- Treat a pipeline recommendation as unsigned. Only a separately verified signed supervisor transition can admit the sandbox.

Use `validate_work_order` before dispatch when the MCP bridge is available. It validates these invariants and returns a BLAKE3 work-order commitment without making provider calls. For progressive MCP discovery and exact policy binding, read [references/mcp-usage.md](references/mcp-usage.md).
