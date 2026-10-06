# Installed skill catalog

This branch includes 38 existing skill capabilities: 24 unique legacy skills,
five plugin skills, and nine project medical-research skills. Unpromoted experimental
drafts are excluded entirely from this privacy-filtered export.
Skill bodies are checked against the export manifest. These installation counts do not indicate
that every skill is mounted for every actor, or demonstrate model performance.

| Group | Location | Use and activation |
| --- | --- | --- |
| 24 legacy skills | `plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/` | The pinned legacy manifest supplies canonical identities and unchanged stage grants. |
| Five plugin skills | `plugins/evamed-codex/skills/` | Existing construction, rollout, recovery, trajectory export, and independent evaluation capabilities. |
| Nine project skills | `plugins/evamed-codex/skills-project/` | Existing medical-research instructions; copying does not change production mounts or native harness selection. |

The plugin skills are `sandbox-construction`, `stage-rollout`,
`summary-failures`, `trajectory-sft`, and `workspace-agent-judge`.
The recovery skill retains its runtime identity `summary_failures`.
Construction, export, and judge instructions have distinct workflow roles; their
presence does not grant a rollout actor evaluator access.

The project skills are `medresearch-core`, `med-data-audit`,
`med-classification`, `med-image-synthesis`, `med-localization`, `med-vqa`,
`med-report`, `med-evidence-attribution`, and `med-recovery-submit`.
Their original catalog was `evamed-skills-v1.1`. Source paths, lengths, and
content commitments are retained in `skills-project/provenance.v1.json`.

## Legacy provenance

The legacy source namespace is redacted for privacy; its retained revision is
`79dd2a31f5f3e5018608bf20f982ad70d2e5dafa`. All 174 skill occurrences, eight
catalogs, and the original README are included, comprising 183 source files.
The privacy-filtered `references/legacy-skill-manifest.v1.json` has a new export commitment and
pins the 24 unique bodies and their stage grants. `vendor/provenance.v1.json`
also records every copied file's length and BLAKE3 commitment.

The source README declares `license: other`. The retained snapshot contains no
LICENSE, COPYING, or NOTICE terms file. No substitute license or new license
grant is asserted; the repository license does not relicense these materials.
See `plugins/evamed-codex/vendor/NOTICE.md`. The manifest's historical
`external-only-license-unresolved` marker is preserved for original verification;
the repository owner explicitly requested this vendored copy.

`training/automedbench_lite/skill_surface.py` resolves this vendored root first.
When it is absent, the original sibling-workspace source path remains supported.
A present but invalid vendored source fails pinned verification instead of
silently falling back. No source task banks, evaluator references, generated
rollouts, credentials, or model outputs are included in the vendored snapshot.

## Legacy skill index

| Skill body | Permitted stages |
| --- | --- |
| [agentic-environment-execution-med](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H005/skills/agentic-environment-execution-med/SKILL.md) | S1, S3, S4, S5 |
| [artifact-verification-med](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/artifact-verification-med/SKILL.md) | S3, S5 |
| [bounded-plan-readiness-med](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H006/skills/bounded-plan-readiness-med/SKILL.md) | S1 |
| [clinical-data-pipeline-mini](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/clinical-data-pipeline-mini/SKILL.md) | S2, S3, S4 |
| [clinical-dialogue-mini](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/clinical-dialogue-mini/SKILL.md) | S2, S3 |
| [ehr-longitudinal-mini](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/ehr-longitudinal-mini/SKILL.md) | S3 |
| [evidence-summary-med](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/evidence-summary-med/SKILL.md) | S2, S3, S4, S5 |
| [expert-consultation-mini](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/expert-consultation-mini/SKILL.md) | S1 |
| [fhir-interop-mini](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/fhir-interop-mini/SKILL.md) | S2, S3, S4, S5 |
| [interactive-clinical-protocol-med](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/interactive-clinical-protocol-med/SKILL.md) | S1, S2, S3 |
| [long-horizon-execution-med](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/long-horizon-execution-med/SKILL.md) | S1, S2, S3, S4, S5 |
| [medical-s2-evidence-mini](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/medical-s2-evidence-mini/SKILL.md) | S2, S3 |
| [medical-s3-judgment-mini](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/medical-s3-judgment-mini/SKILL.md) | S3 |
| [medical-s4-integration-mini](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/medical-s4-integration-mini/SKILL.md) | S4 |
| [medical-s5-delivery-mini](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/medical-s5-delivery-mini/SKILL.md) | S3, S4, S5 |
| [medical-visual-mini](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/medical-visual-mini/SKILL.md) | S2, S3 |
| [medication-safety-mini](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/medication-safety-mini/SKILL.md) | S2, S3 |
| [pathology-volume-mini](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/pathology-volume-mini/SKILL.md) | S2, S3 |
| [pilot-recovery-validation-med](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H007/skills/pilot-recovery-validation-med/SKILL.md) | S3 |
| [research-governance-mini](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/research-governance-mini/SKILL.md) | S3, S4, S5 |
| [research-protocol-mini](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/research-protocol-mini/SKILL.md) | S4 |
| [structured-delivery-mini](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/structured-delivery-mini/SKILL.md) | S5 |
| [tool-routing-mini](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/tool-routing-mini/SKILL.md) | S2, S3, S4, S5 |
| [trial-matching-mini](../plugins/evamed-codex/vendor/rlevo-Med-RL-data/rev-79dd2a31f5f/harnesses/H000/skills/trial-matching-mini/SKILL.md) | S2, S3 |

## Offline readiness check

Install the repository's Python dependencies, then run from the repository root:

```sh
python plugins/evamed-codex/scripts/verify_skill_readiness.py \
  --output /absolute/workspace/path/outside-checkout/skill-readiness-new
```

The output directory must be new and outside this checkout, within your permitted
workspace. The verifier places its materialized skills, temporary files, caches,
and `receipt.json` there. It verifies source bytes against the pinned manifest
and provenance records, then invokes the actual `search_skills` and `load_skill`
handlers for all 38 installed capabilities. It checks legacy stage filtering,
rejects forbidden stage loads, and excludes all experimental drafts. The local
test makes the five plugin and nine project instructions available to its own
catalog solely to check discovery and content delivery; it changes no actor's
permissions, mounts, production configuration, or native harness candidate.

This test makes no API calls and submits no GPU jobs. It proves offline source
and tool readiness. Candidate quality, reward validity, and full workflow
completion require their own preserved execution evidence. The archived
artifact-handoff drafts have no supported promotion claim; API-fixture results
must not be described as trained-policy results.
