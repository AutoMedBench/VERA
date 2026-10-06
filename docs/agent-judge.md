# Workspace-inspecting Agent Judge

EVA-Agent does not treat a single prompt/response grading call as a valid
judge. A valid judgment is an Opus 5 tool-using trajectory over the exact
`EvidenceBundle` produced by one rollout.

## Required execution

1. The judge receives the complete policy-visible actor context, conversation,
   tool trace, terminal answer, private judge reference, exact compiled rubric,
   and metadata-only before/after workspace manifests.
2. Workspace bytes are not copied into the initial prompt. The judge must use
   strict read-only `workspace_list`, `workspace_read`, `workspace_search`, or
   `workspace_diff` tools over the committed snapshots.
3. The first judge turn requires a tool call. If either snapshot contains a
   file, at least one successful `read` or `search` is required before scoring.
   Same-turn independent calls run concurrently within a configured bound.
4. Every item is scored at one of the levels allowed by that same compiled
   rubric object. Workspace citations must have the form
   `workspace:before:<path>` or `workspace:after:<path>` and must refer to a
   file the judge actually inspected.
5. No hidden reasoning is requested or stored. Observable assistant decisions,
   tool calls, tool observations, item scores, concise rationales, and evidence
   references are retained.

## Immutable evidence

Each `JudgeAssessment` embeds a `JudgeAgentTrace`. The trace binds:

- all visible judge trajectory events and same-turn call groups;
- strict tool arguments and bounded observations;
- inspected snapshot paths and their committed file/tree digests;
- per-call BLAKE3 receipts and the source evidence-bundle BLAKE3;
- provider turn count, frontier count, observed parallelism, and zero retries.

The independent verifier recomputes those receipts, reopens both workspace
snapshots, checks every cited path, proves that a content inspection occurred,
and rejects mismatched call inventories or altered events. A failed or missing
agent trace is infrastructure quarantine, never a semantic zero and never an
admission.

## Scaling boundary

Candidate judgments can fan out across provider capacity, and independent
workspace calls can fan out within each judgment. Concurrency changes neither
the rubric nor evidence. A GPU may accelerate local vision, OCR, embedding,
deduplication, or open-model rollout work, but deterministic verification and
signature checks remain hardware-independent.
