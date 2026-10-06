---
license: other
language:
- en
task_categories:
- text-generation
pretty_name: BaTv2 Medical RL Data
---

# BaTv2-Med-RL-data

Private, synthetic BaTMed post-SFT data for separate GRPO and OPD arms. The
release preserves the Medical SFT message, tool, skill, stage, sandbox, and
semantic-action contracts. It defines an agentic completion as host-verified
environment interaction and state change, not a question-answer response.

## Formal release

`rounds/R001` through `rounds/R003` contain the accepted Qwen3.5-4B schedules,
source banks, policy-visible prompts, frozen round contracts, and separate
GRPO/OPD training contracts. `rounds/R004` through `rounds/R006` contain the
Qwen3.5-9B instruction-model OPD branch. Every round has 400 prompt groups and
1,600 rollout requests; private OPD teacher targets are excluded. R006 is
retained as a formal negative round and was not promoted: its narrow S3 teacher
templates did not transfer to new paths and schemas, and its S4/S5 behavior
regressed. R005 remains the latest promoted 9B parent.

`contracts/post_sft_rubrics_v3.json` is the accepted exact binary-rubric
contract. `contracts/agentic_execution_integrity_v1.json` defines EC001: code
actions must actually run, create the declared files, and produce host-verifiable
execution receipts. H005 adds EC001 to the 9B branch; H006 adds the bounded S1
planning skill without changing cases, scorers, tool schemas, or permissions.
H007 exposes the pilot-recovery-validation skill at S3 and preserves strict
host verification that code was written, executed, and produced a validated
artifact. L001 through L005 are quarantined AutoMedBench Lite adapters and do
not redefine the full-39 headline.

`results/formal_post_sft_summary.json` contains aggregate full-39 scores,
attribution vote outcomes, promotion decisions, and the fixed six-cell Lite
native/process series.
These are adapted private research results, not official upstream leaderboard
scores.

Policy-visible schedules retain frozen rubric and scoring-table identifiers so a
sandbox can be audited against its declared contract. This release contains no
benchmark gold, scorer implementation, expected answers, raw judge messages,
credentials, local paths, host endpoints, model checkpoints, patient data,
private OPD teacher targets, or online GRPO trajectories/rewards.
