---
name: trajectory-sft
description: Verify and slice a signed-admitted EvaMed strong-model trajectory into lineage-complete SFT examples. Use only after independent pipeline and supervisor verification; do not export rejected, quarantined, unsigned, weak, or middle trajectories.
---

# Trajectory Sft

Call `eva_agent.pipeline.slice_admitted_trajectory` only after all eligibility gates pass.

- Independently verify the pipeline result, its strong-cohort evaluation, the signed admission receipt, and the signed supervisor transition. Bind every input by BLAKE3; a recommendation alone is insufficient.
- Require that the same compiled rubric drove the agent judge and reward. Require an eligible ability-separation report and one admitted strong trajectory originating from exactly one weak/middle/strong cascade with zero retries.
- Slice at policy-visible assistant-decision boundaries. Preserve the complete visible prefix and source rollout/evaluation/result lineage.
- Keep independent same-turn multi-tool calls as one atomic supervised target. Tool observations become context for later targets, not separate supervised answers.
- Exclude private rubric items, answer keys, judge-only references, hidden reasoning, chain of thought, and credentials recursively. Never infer or reconstruct them from scores.
- Publish new UUID example identities and BLAKE3 lineage receipts without replacing prior output. If any signature, digest, event ordering, or call/observation join fails, stop without producing SFT data; do not retry the source rollout.
