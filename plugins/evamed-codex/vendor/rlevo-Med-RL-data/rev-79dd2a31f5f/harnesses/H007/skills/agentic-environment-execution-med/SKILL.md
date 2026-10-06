---
name: agentic-environment-execution-med
description: Complete S3-S5 medical-research work through real isolated execution, exact host-bound artifacts, and independent validation.
allowed_stages: [S1, S3, S4, S5]
---

# Agentic environment execution

Use this skill to distinguish research work from a plausible written answer.

1. Read the current host observation and its `execution_integrity` contract. Paths,
   episode IDs, input IDs, and schemas are task-specific; never guess or reuse them.
2. In S3, submit executable code rather than comments or pseudocode. Read the
   host-owned input, run one bounded pilot, and write the exact required pilot JSON.
   Include the observed input file sizes, admitted record IDs, positive processed
   count, and finite metrics requested by the contract.
3. Read the S3 gate result. A zero exit code does not mean the pilot passed. If the
   host reports a failed check, correct that exact check during the one allowed clean
   retry; do not claim that S3 completed and do not move to S4 yourself.
4. In S4, execute the full bounded transformation and newly write the exact native
   artifact path. Use every expected task ID once, in the required order and schema.
   Writing an unrelated file, copying a prior artifact, or printing a result is not a
   materialized deliverable.
5. In S5, request independent validation before submission. The validator reopens
   the artifact from disk; base the terminal report on its receipt, not on memory or
   intent.

For MedX or AgentClinic, the same rule applies without arbitrary code: use fresh
episode-bound evidence-service or simulator receipts, carry their IDs into the
materialized artifact, and require the host to reopen it before submission.
