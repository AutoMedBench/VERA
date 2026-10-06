---
name: bounded-plan-readiness-med
description: Build an S1 medical-research plan that is ready for real harness execution, with host-owned inputs, executable actions, exact artifacts, validation gates, budgets, stop rules, and recovery points. Use before a multi-stage S1-S5 workflow, especially when later stages must run code or interact with a sandbox rather than answer in prose.
---

# Bounded plan readiness

Turn the research objective into an environment-executable contract before
advancing from S1.

1. Name the exact deliverable and its required host path or service receipt.
   Do not substitute a narrative answer for a file, tool result, or state change.
2. List the host-owned inputs and the tool that will inspect each one. Treat
   paths, schemas, episode IDs, and allowed classes as values to read from the
   current observation, never as values to guess.
3. Map S2-S5 to concrete effects:
   - S2 inspects or prepares admitted inputs.
   - S3 sends syntactically complete executable code for one bounded pilot and
     writes the exact pilot artifact.
   - S4 sends complete executable code for native materialization and writes
     the exact required output.
   - S5 independently reopens, validates, and submits that output.
4. State measurable readiness checks for every transition: successful tool
   receipt, zero exit code, positive captured output when required, expected
   filesystem delta, exact artifact identity, and schema-valid content.
5. Allocate turns and time, reserve capacity for S5, and declare a stop rule.
   If the pilot cannot fit the host budget, stop with the typed constraint; do
   not silently shrink the requested deliverable.
6. Declare one recovery point and what evidence may trigger it. A recovery
   retries executable work with corrected code; it never replaces execution
   with a claim that the work ran.

Use code, not labels. For example, `bounded_detection_pilot` and
`load_skill artifact-verification-med` are not Python programs. A valid S3
action contains imports, reads the admitted input, writes the required JSON,
and prints or returns bounded verification evidence in the same execution.
