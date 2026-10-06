---
name: artifact-verification-med
description: Independently inspect and validate bounded medical-research artifacts before batch continuation or delivery.
---

# Artifact verification

1. Inspect the host-declared artifact identifier rather than assuming a path
   or a command exit status proves its contents.
2. Check the declared schema, required fields, item count, non-empty outputs,
   type or shape, and finite permitted value range.
3. Compare the observed count with the current stage plan and identify missing,
   duplicated, malformed, or unverifiable items explicitly.
4. Run only a host-registered validator. Preserve its bounded receipt and fix
   the earliest failed invariant before continuing.
5. Recheck a completed batch independently before submission.

Do not fabricate validation results, silently coerce invalid clinical data, or
use an artifact inspection as permission to access other data.
