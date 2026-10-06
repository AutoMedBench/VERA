---
name: pilot-recovery-validation-med
description: Execute one real bounded S3 pilot through the host, validate its schema and non-degeneracy, and use typed failure evidence for one corrected rerun. Use when the workflow offers an S3 execution or synthesis action; do not spend the only S3 action trying to load this already preloaded guidance.
allowed_stages: [S3]
---

# Pilot recovery validation

Treat S3 as an environment transition, not a written analysis.

1. Read the current host observation for the admitted input, exact pilot path,
   artifact identity, schema, and execution-integrity checks. Never guess a path,
   ID, class set, shape, or expected count.
2. Use the offered S3 action itself. For `execute_code`, send syntactically
   complete executable code—not comments, pseudocode, a skill name, or a shell
   phrase. In that same process, read the admitted input, run one bounded item,
   write the exact pilot artifact, reopen it, and print compact verification
   evidence.
3. A successful exit status is necessary but insufficient. Require the host
   observation to confirm a positive processed count, the expected filesystem
   delta, the exact artifact identity, a loadable native schema, finite bounded
   values, and a non-empty, non-degenerate pilot result.
4. If the host returns a typed failure, correct that named check during the one
   allowed S3 retry. Preserve valid code and provenance, change only the failing
   assumption, execute again, and inspect the new receipt. Do not replace a rerun
   with an explanation that the code should work.
5. For semantic S3 tools, use the same loop with the episode-bound service
   receipt: execute the offered synthesis/simulator action, inspect its typed
   fields, and make one evidence-backed correction when the host permits it.
6. Stop after a passing pilot or an exhausted typed retry. Let the host advance
   to S4; do not claim that the full native artifact or S5 submission already
   exists.

A valid pilot leaves inspectable host evidence. Text that merely describes an
analysis, code that writes no required file, and output that is empty or constant
receive no S3 process credit.
