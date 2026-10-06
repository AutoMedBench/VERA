# Explicit executable training-domain selection

This prospective MED integration changes no default and does not select an
S-target or authorize training. The target still comes from actual verified
Qwen evaluation. Public rubric identities and source domains remain unchanged.

Optional settings consumed by `training.eva_rsi.production`:

```json
{
  "execution_catalog_path": "/absolute/path/to/signed-catalog.json",
  "trust_store_path": "/absolute/path/to/public-trust-store.json",
  "training_domains_by_stage": {
    "S1": ["automedbench-classification", "automedbench-segmentation"],
    "S2": ["automedbench-segmentation"],
    "S3": ["automedbench-segmentation"]
  },
  "training_sandbox_limit": 128
}
```

This is an illustrative explicit subset, not the selected live curriculum. Both
catalog paths must be supplied together. Without a domain mapping, all observed
domains remain requested and missing signed coverage fails before the training
subprocess. With a mapping, its selected-stage entry must be present and its
domains must be a subset of that stage's actually observed domains; there is no
implicit transfer curriculum or domain renaming. The mapping alone is allowed
but does not assert signed execution verification.

The plan records the original evaluated domain rows, requested training domains,
and omitted observed domains separately. The existing signed selector retains
exact source/rubric bindings. A new `training-data-selection.json` binds the
existing preparation receipt and records requested-domain source-match and
signed-executable counts, actual distinct emitted sandboxes, and limit shortfall.
A blocked selection records zero emitted sandboxes even if some eligible rows
existed. An unsigned selection reports signed counts as unknown, not inferred.
Batching or repeated online rollouts do not create additional distinct starts.

## Explicit broad-medical stage transfer

The separately opt-in `"training_domain_scope": "verified_stage_transfer"`
requests all signed executable original source domains at the evaluated S-target
stage, not just domains present in the AutoMed evaluation. It requires both
catalog paths and cannot be combined with `training_domains_by_stage`. It does
not change S-target selection, remap domains/rubrics, accept source-only rows, or
mean that unsupported AutoMed domains acquired training sources.

The plan retains evaluation-domain coverage. The data-selection receipt records
actual selected training domains, eligible signed domains, evaluation domains
without selected training sources, and training domains absent from the eval.
The existing `training_sandbox_limit` still caps distinct rows: with limit128,
stage-wide eligible246/248/251 does not become emitted246/248/251. A future
operator may explicitly set a limit of at least251 to include every currently
eligible start. No live scope, limit, S-target or training launch was changed.

## Read-only actual source inventory

Audit: September 10, 2026; 6000 released rows intersected with the signature-
verified primary/train/executable-legacy catalog. This is signed membership, not
a new resolver execution or provider call.

| Source domain | S1 | S2 | S3 |
| --- | ---: | ---: | ---: |
| agentclinic | 235 | 237 | 235 |
| automedbench-classification | 1 | 0 | 0 |
| automedbench-detection | 0 | 0 | 0 |
| automedbench-research | 1 | 1 | 2 |
| automedbench-segmentation | 1 | 1 | 1 |
| healthbench-professional | 1 | 2 | 4 |
| medxpertqa | 7 | 7 | 9 |
| **All source domains** | **246** | **248** | **251** |

The four new AutoMed Lite synthesis/VQA/report/enhancement domains have no rows
in this release. Across the seven AutoMed evaluation domains, only two distinct
S1 tasks and one each S2/S3 are executable. Detection is source-only even though
it has release rows. Other source domains are not implicit substitutes within
observed-domain selection; the separately explicit stage-transfer mode keeps
their original labels and discloses their cross-domain use.

Actual existing local inputs (not copied or changed):

- Release: `/localhome/local-operator/operator_GB300-2/EVA-Agent/runs/bulk-rl-sandboxes.v1`.
- Catalog: `/localhome/local-operator/operator_GB300-2/EVA-Agent/runs/prospective-execution-binding-catalog.v3.r1.json`.
- Public trust store: `/localhome/local-operator/operator_GB300-2/rlevo-med-research/config/host-trust-store.v1.json`.
- Verified catalog BLAKE3: `12ed96b77fbecf03093f30e0e2b5d434c4e8d1f2364101731b6b141423b93c1f`.

Validation (synthetic signed CPU fixtures; no training/provider calls):

```sh
PYTHONPATH=src:. python -m pytest -q tests/test_eva_rsi_training_domain_selection.py tests/test_eva_rsi_production.py tests/test_grpo_signed_stage_data.py tests/test_grpo_stage_data.py
```
