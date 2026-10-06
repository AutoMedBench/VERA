# EvaMed Codex v1.2 research

An opt-in context and task-memory layer on the pinned Codex 0.153.4 runtime,
inheriting the lightweight v1.1 feature profile. Existing medical tool schemas,
skill bodies, rubric definitions, provider routes and security settings are not
rewritten. This is an SDK composition layer, not a newly compiled Codex binary.

## Context and memory behavior

- Keep task threads across stages where possible. Persist public state, artifact
  references, pending jobs and next actions in `notes/task-state.md`; recover from
  the note even if a new process needs a new thread.
- Mount the new `summary_failures` runtime skill (native skill name
  `summary-failures`). It records evidence-linked mistakes and preventive checks
  in `notes/failures.md`, then asks subsequent stages to read and use them.
- Default to early compaction at 12,288 tokens for the actual 32,768-token Qwen
  context, reserving room for projected history, 4,096 output tokens and margin.
  This is measured-headroom-informed configuration, not a proof that every
  possible tool response fits. Use bounded reads and workspace references.
- Memory is task-scoped. Do not feed hidden labels, private judge information or
  other evaluated models' solutions into the actor. Generic lessons need separate
  review before cross-task promotion.

## SDK integration

```python
from eva_agent.codex_runtime.research_memory import (
    memory_launch_options, memory_thread_options, memory_turn_input,
)

launch = memory_launch_options(existing_launch_options)
thread_options = memory_thread_options(existing_actor_thread_options)
turn_input = memory_turn_input(existing_turn_input)
# Pass these objects to the existing CodexRuntime lifecycle. Keep the same
# thread handle for subsequent stages; use the runtime's normal resume API
# when a persisted thread is available. Persisted notes support fresh threads.
```

Explicit caller configuration takes precedence, including stricter benchmark
restrictions. The new skill is appended through the existing `CodexSkill` mount;
no tool or protocol fields are introduced. Archived evaluation attempts do not
acquire this new profile retroactively.

Inspect the profile without provider calls:

```bash
PYTHONPATH=src python scripts/evamed_codex_research_v1_2.py
```

`--codex-bin /path/to/pinned/codex` additionally checks the v1.1 feature defaults
with isolated strict initialization; it does not run a medical task or verify
recall. Current CPU checks establish schema preservation, bounded configuration
and exact skill mounting only. Actual Qwen tool use, saved failure notes, later
reads and correction behavior must be reported separately, including failures.
