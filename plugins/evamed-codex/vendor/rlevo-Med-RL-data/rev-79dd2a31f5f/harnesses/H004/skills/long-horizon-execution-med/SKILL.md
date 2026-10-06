---
name: long-horizon-execution-med
description: Plan and complete a bounded multi-stage medical-research task with checkpointed artifacts, pilots, validation, and final delivery.
---

# Long-horizon medical execution

1. Create the S1 plan artifact first. State the intended deliverable, permitted
   tools, success checks, and a recovery point.
2. In S2, use only host-admitted resources. Confirm an actual bounded setup or
   forward operation before treating an environment as ready. When the task
   will be repeated, measure one representative pilot's item count, elapsed
   time, and output validity; use that observation to estimate whether the
   allowed budget can finish the declared deliverable.
3. In S3, produce a small pilot artifact and verify its declared format,
   non-emptiness, shape or schema, and permitted value range.
4. In S4, freeze a pilot that passed validation, process bounded batches, and
   checkpoint enough state to resume without claiming unfinished work. Reserve
   sufficient time and budget for S5 (normally at least one fifth of the
   remaining run budget); do not consume all resources on inference and leave
   no path to validation or submission. If the pilot forecast cannot meet the
   host budget, report the constraint and use only a host-declared reduced or
   fast mode rather than silently changing the task contract.
5. In S5, run an independent declared validator, reconcile counts with the
   plan, and submit only the verified artifact.

Treat a successful command as evidence only of that command. Do not treat it
as evidence that a clinical or research artifact is complete. Keep evidence
and assumptions in explicit artifacts; ne