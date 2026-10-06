"""Stage-scoped discovery preferences; canonical skills and access never change."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from eva_agent.pipeline.digests import blake3_hex
from .evidence import commitment, read, require

SCHEMA = "eva.rsi-skill-selection-guidance.v1"
STAGES = ("S1", "S2", "S3", "S4", "S5", "E2E")
PROTECTED = {"eva-workflow/stage-rollout", "summary_failures"}


def read_selection(reference, *, catalog_id=None):
    if reference is None:
        return None
    path = Path(reference["path"])
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 65536,
            "skill_selection_file_unavailable")
    require(commitment(path) == reference, "skill_selection_commitment_differs")
    document = read(path)
    require(document.get("schema") == SCHEMA
            and document.get("document_blake3") == blake3_hex({k: v for k, v in document.items()
                if k != "document_blake3"}), "skill_selection_document_differs")
    if catalog_id is not None:
        require(document.get("catalog_id") == catalog_id, "skill_selection_catalog_differs")
    stages = document.get("stages")
    require(isinstance(stages, dict) and set(stages) == set(STAGES), "skill_selection_stages_differ")
    for row in stages.values():
        require(isinstance(row, dict) and set(row) == {"preferred", "deprioritized"},
                "skill_selection_stage_shape")
        for ids in row.values():
            require(isinstance(ids, list) and len(ids) <= 24
                and all(isinstance(value, str) and value and len(value) <= 240 for value in ids)
                and ids == sorted(set(ids)) and not set(ids) & PROTECTED, "skill_selection_ids_differ")
        require(not set(row["preferred"]) & set(row["deprioritized"]), "skill_selection_conflict")
    return document


def render_selection(reference, *, stage, visible_ids, catalog_id=None):
    """The SAME bounded actor-visible text is used in training and evaluation.

    No Judge scores, rationale, evidence paths or other tasks' answers enter the
    prompt. Stage permissions are checked against the actual verified surface.
    """
    document = read_selection(reference, catalog_id=catalog_id)
    if document is None:
        return ""
    require(stage in STAGES, "skill_selection_stage_unavailable")
    row = document["stages"][stage]
    require(set(row["preferred"]) | set(row["deprioritized"]) <= set(visible_ids),
            "skill_selection_not_stage_eligible")
    if not row["preferred"] and not row["deprioritized"]:
        return ""
    return ("\n\nStage-scoped skill discovery preferences (not permissions or task answers):\n"
        + json.dumps({"stage": stage, **row}, sort_keys=True, separators=(",", ":"))
        + "\nPrioritize searching/loading preferred skills when relevant. Deprioritized skills are not "
          "default recommendations, but remain available when the actual task requires them. "
          "Use the unchanged canonical search_skills/load_skill schemas. These preferences do not "
          "establish skill use, stage completion, or causal benefit.")


def apply_verified_attribution(output_root, decision, *, previous, catalog_id):
    """Only verified existing-ID recommendations affect the next attempt.

    Unsupported or contradictory proposals retain the previous selection with
    an explicit reason; never block the automated loop or author a new skill.
    """
    old = read_selection(previous, catalog_id=catalog_id)
    stages = (json.loads(json.dumps(old["stages"])) if old else
              {stage: {"preferred": [], "deprioritized": []} for stage in STAGES})
    source = decision.get("verified_selection_source")
    if decision.get("opus_attribution") != "verified" or not isinstance(source, dict):
        return {"status": "retained", "reason": "verified_opus_selection_evidence_unavailable",
                "selection": previous, "selection_guidance_changed": False, "canonical_catalog_changed": False}
    stage = source.get("stage")
    require(stage in STAGES and source.get("catalog_id") == catalog_id, "attribution_selection_source_differs")
    eligible = {row["skill_id"] for row in source["catalog"]
                if stage in row.get("allowed_stages", []) and row["skill_id"] not in PROTECTED}
    actions = {}
    skipped = []
    for suggestion in decision["suggestions"]:
        action, identifier = suggestion["action"], suggestion["skill_id"]
        if action == "retain":
            continue
        if identifier not in eligible:
            skipped.append({"action": action, "skill_id": identifier,
                            "reason": "new_protected_or_stage_ineligible_id"})
            continue
        actions.setdefault(identifier, set()).add(action)
    row = stages[stage]
    for identifier, choices in sorted(actions.items()):
        if len(choices) != 1:
            skipped.append({"skill_id": identifier, "reason": "conflicting_suggestions_retained"})
            continue
        action = next(iter(choices))
        require(action in {"add", "remove"}, "attribution_selection_action")
        destination = "preferred" if action == "add" else "deprioritized"
        other = "deprioritized" if action == "add" else "preferred"
        row[destination] = sorted(set(row[destination]) | {identifier})
        row[other] = sorted(set(row[other]) - {identifier})
    changed = stages != (old["stages"] if old else {s: {"preferred": [], "deprioritized": []} for s in STAGES})
    reference = previous
    if changed:
        from .controller import write
        core = {"schema": SCHEMA, "selection_id": str(uuid4()), "catalog_id": catalog_id,
                "stages": stages, "previous_selection": previous, "source_attribution": decision["result"],
                "canonical_catalog_changed": False, "causal_effect_established": False}
        document = {**core, "document_blake3": blake3_hex(core)}
        path = Path(output_root) / document["selection_id"] / "selection.json"
        write(path, document, exclusive=True)
        reference = commitment(path)
    return {"status": "applied" if changed else "retained",
            "reason": "verified_stage_discovery_preferences" if changed else "no_supported_selection_change",
            "selection": reference, "selection_guidance_changed": changed, "skipped": skipped,
            "canonical_catalog_changed": False, "causal_effect_established": False}
