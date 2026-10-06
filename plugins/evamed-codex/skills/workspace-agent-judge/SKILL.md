---
name: workspace-agent-judge
description: Judge an EvaMed rollout against its bound rubric by inspecting committed conversation, tool, and workspace evidence. Use for Opus 5 agent judging and independent verification; do not use live mutable files or expose judge-only material to the actor.
---

# Workspace Agent Judge

Evaluate each retained trajectory with an Opus 5 agent judge over the full policy-visible context plus immutable evidence bundle.

1. Reopen the committed evidence and its BLAKE3 bindings. Never judge against the live sandbox filesystem.
2. Use `eva_agent.pipeline.JudgeWorkspaceTools`: `workspace_list`, `workspace_read`, `workspace_search`, and `workspace_diff`. These are read-only snapshot operations and may run concurrently in bounded independent groups.
3. Inspect file content, not only metadata. Cite only observed context events, actor tool calls, judge references, and workspace paths actually read through the judge tools.
4. Give private reference material only to the judge boundary. It must never appear in actor context, tool observations, public summaries, or SFT output.
5. Score all atomic items from the exact `CompiledRubricTable` used by `WeightedRubricRewarder`; retain item scores, hard-gate status, evidence references, the judge tool trajectory, and BLAKE3 receipts.
6. Perform one judgment attempt with zero retry. Preserve provider/judge failures as infrastructure evidence without producing a semantic reward.
7. Reopen the result with `eva_agent.pipeline.PipelineVerifier`. A valid judgment or separation recommendation still cannot admit a candidate without external signed supervisor evidence.
