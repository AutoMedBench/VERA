---
name: sandbox-construction
description: Construct or verify an answer-free, rubric-bound EVA medical-research sandbox from a pinned benchmark episode. Use for candidate materialization before any model rollout; do not use to judge, admit, or export trajectories.
---

# Sandbox Construction

Create one immutable candidate identity for one source episode, domain, and `S1`–`S5` or `E2E` stage.

1. Pin the source revision and retain the source episode identifier and content digest.
2. Resolve exactly one table through `eva_agent.rubrics.load_and_compile_registry`. Bind its ID, version, and BLAKE3 digest; never copy or hand-translate its items.
3. Project only answer-free task context and required initial files. Keep answer keys, reference answers, judge-only evidence, private rubric items, and hidden reasoning outside policy-visible context. A rollout may receive the rubric binding, never its private scoring contents.
4. Use UUID runtime identities and BLAKE3 content commitments. Materialize with the contracts in `eva_agent.pipeline.BenchmarkEpisode` and `eva_agent.pipeline.VerifiableDataPipeline`.
5. Reopen every generated file/manifest before declaring construction complete. Construction is not admission and must not create signed supervisor evidence.

Never replace a candidate after a provider boundary. The later weak/middle/strong cascade is one rollout per cohort with zero retry; infrastructure failures remain quarantined rather than being selected away.
