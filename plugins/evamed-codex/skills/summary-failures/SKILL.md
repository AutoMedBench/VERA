---
name: summary-failures
description: Maintain evidence-linked task memory after tool errors, failed checks, stage transitions, or resumed medical-research work. Use to avoid repeating observed mistakes; not to infer hidden benchmark answers or rewrite outcomes.
metadata:
  runtime_alias: summary_failures
---

# Failure summary and recovery

Use this skill as `summary_failures` in the EvaMed runtime. Keep useful state in the current task workspace, not only in conversation history.

At a stage boundary, before a large context expansion, or after an error, update the existing public task-state note with the current objective, completed artifacts and their paths, pending jobs, unresolved questions, and the next concrete action. Keep it short; link to full logs instead of copying them. On resume or after compaction, read that note and reopen the relevant artifacts before acting.

For a consequential failure, maintain a compact entry in `notes/failures.md`:

- Observed symptom and a public tool-result or workspace-log reference.
- Cause: confirmed from evidence, or explicitly a hypothesis.
- Correction attempted and its observed result; mark untested corrections as untested.
- A specific check to perform before repeating the operation.

Use the currently offered note-writing or filesystem tools with their actual schema. If the note tool takes `name`, pass `failures.md`, not a nonexistent `path` argument. If that tool is absent, use an allowed workspace-writing tool; never invent a tool. Coordinate parallel workers so only one writes the same note at a time.

Before another similar operation, read the relevant failure entry and perform its check. For example, a schema error calls for checking the current tool signature; an unfinished model job calls for inspecting its actual status before consuming outputs. Reopen artifacts when memory and the workspace disagree. A remembered claim is not verification.

Summarize only policy-visible tool responses, work products, and decisions. Exclude secrets, hidden references, judge-only material, other models' solutions, and private reasoning. Preserve original failures and uncertainty. Do not convert an infrastructure error into a medical score, retry a frozen benchmark attempt, or expand task permissions. Keep task-specific facts scoped to that task; cross-task promotion requires a separately reviewed, non-answer-bearing lesson.

Completion evidence is a saved note that a later stage or resumed agent actually reads and uses correctly. Creating the note or enabling a memory flag alone is not proof that the agent learned from it.
