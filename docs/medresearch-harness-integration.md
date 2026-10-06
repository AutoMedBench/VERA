# Medical training with the separately maintained research harness

Branch ownership is main=data curation, EVA-MedResearch=training/evaluation/RSI,
harness-evamed=Codex runtime, tools, skills and context/memory. Shared package
names do not mean installing all three branches into one venv selects them.

Use an explicit frozen harness checkout containing policy_budget.py and
research_memory.py (v1.2 source commit a361f9c74e7df0047f9d5c82a7bdfcb46599f6e1).
The prospective Supra version is a separate choice; changing that selection
does not upgrade a running or frozen experiment.

```sh
export EVA_HARNESS_ROOT=/absolute/frozen/harness-checkout
export PYTHONPATH="$EVA_HARNESS_ROOT/src:$EVA_HARNESS_ROOT/tests:."
.venv/bin/python -m pytest -q -o pythonpath="$EVA_HARNESS_ROOT/src" \
  tests/test_harness_source.py tests/test_automedbench_policy_capture.py \
  tests/test_eva_rsi_skill_identity.py tests/test_failure_memory_v12.py \
  tests/test_failure_memory_compaction.py tests/test_eva_rsi_production.py
```

The training entrypoints pin the selected eva_agent package before local source
path modifications; child training/Ray processes retain the same source choice.
The benchmark's public MCP subprocess still drops private environment variables
and uses its canonical public tool implementation. Selecting a harness does not
widen actor permissions, grant hidden inputs, or modify canonical tool schemas.
Without an explicit selection, historical source ordering remains unchanged;
new consumers need the new runtime dependency and must not fall back silently.

Integrated consumers include exact owned-turn interruption/terminal drain before
final snapshots; explicit opt-in skill-content comparison across relocated
unchanged mounts; and a real three-stage local-Qwen failure-memory diagnostic.
The latter preserves each actual failure, thread restart, fresh-thread recovery,
tool receipt and snapshot. A supplemental item-type observer records actual
context compaction and discloses when subsequent work targets the wrong stage.

Observed diagnostic: 44 actual tool calls, notes reused across restart/new
thread/compaction, but all three final messages empty and post-compaction stage
drift. This is partial operational memory evidence, not a full pass. Original
results remain unchanged. Low scores are allowed; missing or infrastructure
results are not clinical zero. No checkpoint or training step is authorized by
CPU tests or by the presence of a profile alone.
