"""Explicit comparison family; raw Judge identities and evidence never change."""
from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import re
import subprocess

from eva_agent.pipeline.digests import blake3_bytes, blake3_hex
from .evidence import commitment, read, require

SCHEMA = "eva.rsi-judge-comparison-family.v1"
DIFFERENCE = "explicit-current-after-judge-only-annotation"
FIELDS = {"judge_backend", "requested_model", "provider", "maximum_execution_frontiers",
    "maximum_workspace_calls", "maximum_transport_projection_turns", "turn_timeout_seconds",
    "implementation_blake3", "material_view", "material_view_implementation_blake3",
    "materializer_implementation_blake3"}
RELATIVE = "src/eva_agent/training/slime_agent_judge.py"
MATERIAL_SOURCES = {
    "material_view_implementation_blake3": "src/eva_agent/pipeline/judge_material_view.py",
    "materializer_implementation_blake3": "src/eva_agent/training/agent_judge_worker.py",
}


def _source(row):
    root = Path(row["core_root"]).resolve(strict=True)
    require(re.fullmatch("[0-9a-f]{40}", row["core_commit"]) is not None, "judge_family_commit_invalid")
    raw = subprocess.check_output(["git", "-C", str(root), "show", row["core_commit"] + ":" + RELATIVE])
    actual = (root / RELATIVE).read_bytes()
    require(actual == raw and blake3_bytes(raw) == row["profile"]["implementation_blake3"],
            "judge_family_source_version_differs")
    for field, relative in MATERIAL_SOURCES.items():
        frozen = subprocess.check_output(["git", "-C", str(root), "show", row["core_commit"] + ":" + relative])
        require((root / relative).read_bytes() == frozen and blake3_bytes(frozen) == row["profile"][field],
                "judge_family_material_source_version_differs")
    return raw, str(root / RELATIVE)


def declared_judge_comparison_family(old_root, new_root, old_profile):
    """Prepare an explicit declaration; it creates no grade or acceptance claim."""
    rows=[]
    for root, supported in ((Path(old_root),False),(Path(new_root),True)):
        profile={**old_profile,"implementation_blake3":blake3_bytes((root/RELATIVE).read_bytes())}
        rows.append({"judge_id":blake3_hex(profile),"profile":profile,"core_root":str(root.resolve()),
            "core_commit":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),
            "annotation_support":supported})
    core={"schema":SCHEMA,"declared_difference":DIFFERENCE,"profiles":rows}
    return {**core,"document_blake3":blake3_hex(core)}


def _annotation_only(old, new):
    """Prove the declared source difference, not just a version-name assertion."""
    original, changed = ast.parse(old), ast.parse(new)
    helpers = [node for node in changed.body if isinstance(node, ast.FunctionDef)
               and node.name == "_current_after_recovery_annotation"]
    require(len(helpers) == 1, "judge_family_annotation_helper_absent")
    changed.body.remove(helpers[0])
    functions = {node.name: node for node in changed.body if isinstance(node, ast.FunctionDef)}
    prepare = functions["prepare_workspace_rollout"]
    assignment, call, result = prepare.body[-3:]
    require(isinstance(assignment, ast.Assign) and len(assignment.targets) == 1
        and isinstance(assignment.targets[0], ast.Name) and assignment.targets[0].id == "prepared"
        and isinstance(call, ast.Expr) and isinstance(call.value, ast.Call)
        and isinstance(call.value.func, ast.Name) and call.value.func.id == "_current_after_recovery_annotation"
        and len(call.value.args) == 1 and isinstance(call.value.args[0], ast.Name)
        and call.value.args[0].id == "prepared" and not call.value.keywords
        and isinstance(result, ast.Return) and isinstance(result.value, ast.Name) and result.value.id == "prepared",
        "judge_family_prepare_differs")
    prepare.body[-3:] = [ast.Return(value=assignment.value)]
    removed = 0
    for node in ast.walk(functions["grade_prepared_rollout"]):
        if not isinstance(node, ast.Dict):
            continue
        for index in range(len(node.keys) - 1, -1, -1):
            value = node.values[index]
            if (node.keys[index] is None and isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "_current_after_recovery_annotation"):
                require(ast.dump(value) == ast.dump(ast.parse("_current_after_recovery_annotation(prepared)", mode="eval").body),
                        "judge_family_annotation_call_differs")
                del node.keys[index]; del node.values[index]; removed += 1
    require(removed == 1 and ast.dump(original) == ast.dump(changed), "judge_family_undeclared_source_difference")


def verify_judge_comparison_family(reference):
    require(isinstance(reference, dict) and set(reference) == {"path", "blake3"}
        and commitment(reference["path"]) == reference, "judge_family_sidecar_commitment")
    value = read(reference["path"])
    require(set(value) == {"schema", "declared_difference", "profiles", "document_blake3"}
        and value["schema"] == SCHEMA and value["declared_difference"] == DIFFERENCE
        and value["document_blake3"] == blake3_hex({k:v for k,v in value.items() if k != "document_blake3"}),
        "judge_family_document_invalid")
    rows = value["profiles"]
    require(isinstance(rows, list) and len(rows) == 2, "judge_family_requires_declared_old_new_pair")
    require(all(isinstance(row,dict) and set(row)=={"judge_id","profile","core_root","core_commit","annotation_support"}
        and type(row["annotation_support"]) is bool and isinstance(row["profile"],dict)
        and set(row["profile"]) == FIELDS and row["judge_id"] == blake3_hex(row["profile"])
        for row in rows), "judge_family_profile_fields_differ")
    require({row["annotation_support"] for row in rows} == {False, True}
        and len({row["judge_id"] for row in rows}) == 2, "judge_family_profile_pair_invalid")
    old, new = sorted(rows, key=lambda row: row["annotation_support"])
    protocol = {k:v for k,v in old["profile"].items() if k != "implementation_blake3"}
    require(protocol == {k:v for k,v in new["profile"].items() if k != "implementation_blake3"}
        and protocol["judge_backend"] == "native_astra" and protocol["requested_model"] == "gpt-6-astra"
        and protocol["material_view"] == "policy-visible-audit-v2", "judge_family_protocol_differs")
    old_raw, old_path = _source(old); new_raw, new_path = _source(new)
    _annotation_only(old_raw, new_raw)
    family_id = blake3_hex({"schema": SCHEMA, "protocol": protocol,
        "declared_difference": DIFFERENCE, "raw_judge_ids": sorted(row["judge_id"] for row in rows)})
    return {"schema": SCHEMA, "comparison_judge_id": family_id, "sidecar": reference,
        "profiles": {row["judge_id"]: {**row, "implementation_source": path}
                     for row,path in ((old,old_path),(new,new_path))},
        "original_evidence_mutated": False, "implementations_identical": False,
        "declared_difference": DIFFERENCE}


def reopen_family_feedback(root, family, *, include_skill_source_binding=False):
    from training.benchmark_feedback.automed_codex import verify_feedback
    claimed = read(Path(root) / "verification.json")["round_identity"]["judge_id"]
    require(claimed in family["profiles"], "judge_family_undeclared_profile")
    row = family["profiles"][claimed]
    value = verify_feedback(Path(root), include_skill_source_binding=include_skill_source_binding,
        judge_implementation_source=row["implementation_source"],
        judge_material_sources={field: str(Path(row["core_root"]) / relative)
                                for field, relative in MATERIAL_SOURCES.items()})
    require(value["round_identity"]["judge_id"] == claimed and value["judge_profile"] == row["profile"],
            "judge_family_reopened_profile_differs")
    return value


def normalize_judge_identity(feedback, family):
    raw = deepcopy(feedback["round_identity"])
    require(raw["judge_id"] in family["profiles"]
        and feedback["judge_profile"] == family["profiles"][raw["judge_id"]]["profile"],
        "judge_family_undeclared_profile")
    return {"raw_round_identity": raw,
        "comparison_round_identity": {**raw, "judge_id": family["comparison_judge_id"]}}
