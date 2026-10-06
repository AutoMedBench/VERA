from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from eva_agent.codex_runtime import CodexSkill
from eva_agent.pipeline import Stage, ToolRegistry
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value
from eva_agent.training.progressive_skills import ProgressiveTeacherSkillSurface
from training.automedbench_lite.skill_discovery import (
    build_public_skill_discovery, render_public_skill_discovery,
)
from training.automedbench_lite.skill_surface import SKILL_TOOLS


@pytest.fixture
def surface(tmp_path):
    """Tiny verified-byte catalog through the actual canonical stage handlers."""
    rows = (
        ("bounded-plan-readiness-med", "Make an evidence-gap plan with a declared stop rule.", ("S1",)),
        ("artifact-verification-med", "Independently reopen actual artifacts.", ("S3", "S5")),
    )
    inventory, documents = [], {}
    bodies = {"eva-workflow/stage-rollout": b"---\nname: stage-rollout\ndescription: Stage workflow guidance.\n---\nNATIVE-BODY-NOT-IN-HINT\n"}
    bodies.update({name: f"ACTUAL-BODY-NOT-IN-HINT-{name}\n".encode() for name, _, _ in rows})
    for index, (name, body) in enumerate(bodies.items()):
        path = tmp_path / f"{index}.md"
        path.write_bytes(body)
        documents[name] = CodexSkill(skill_id=name, name=name, path=str(path), content_blake3=blake3_bytes(body))
    for name, desc, stages in rows:
        inventory.append({"skill_id": name, "description": desc, "allowed_stages": stages,
                          "source_path": "/private/not-public", "path": documents[name].path,
                          "content_blake3": documents[name].content_blake3})
    verified = SimpleNamespace(inventory=lambda: tuple(inventory), catalog_blake3=blake3_hex(inventory),
        for_stage=lambda stage: (documents["eva-workflow/stage-rollout"],
            *(documents[name] for name, _, allowed in rows if stage.value in allowed)))
    progressive = ProgressiveTeacherSkillSurface(verified)
    definitions = {stage.value: {d.name: d for d in progressive.augment_registry(ToolRegistry(()), stage).definitions()}
                   for stage in Stage}
    calls = []
    def call(name, arguments, *, stage):
        calls.append((name, deepcopy(arguments), stage))
        return definitions[stage][name].handler(None, arguments)
    return SimpleNamespace(inventory=canonical_value(inventory), catalog_blake3=verified.catalog_blake3,
                           definitions=definitions, call=call, calls=calls, bodies=bodies)


def test_exact_substring_diagnosis_and_real_body_load_from_hint(surface):
    assert surface.call("search_skills", {"query": "bounded plan", "stage": "S1"}, stage="S1") == {"matches": []}
    sidecar = build_public_skill_discovery(surface, "S1")
    row = next(r for r in sidecar["skills"] if r["skill_id"] == "bounded-plan-readiness-med")
    result = surface.call("search_skills", row["search_arguments"], stage="S1")
    assert result["matches"] == [{"skill_id": row["skill_id"], "description": row["description"]}]
    loaded = surface.call("load_skill", row["load_arguments"], stage="S1")
    assert loaded["content"].encode() == surface.bodies[row["skill_id"]]
    assert loaded["content_blake3"] == blake3_hex(loaded["content"])
    assert loaded["delivery"] == "policy-visible-tool-observation"


def test_discovery_does_not_load_bodies_or_leak_catalog_paths(surface):
    original_inventory = deepcopy(surface.inventory)
    original_schemas = deepcopy(SKILL_TOOLS)
    sidecar = build_public_skill_discovery(surface, Stage.S1)
    assert surface.calls == [("search_skills", {"query": " ", "stage": "S1"}, "S1")]
    assert sidecar["contains_skill_bodies"] is False and sidecar["counts_as_skill_use"] is False
    assert sidecar["visible_skill_count"] == 2
    text = render_public_skill_discovery(sidecar)
    encoded = json.dumps(sidecar) + text
    assert "NOT-IN-HINT" not in encoded and "/private/" not in encoded and "source_path" not in encoded
    assert "artifact-verification-med" not in text
    assert "bounded-plan-readiness-med" in text and "stage='S1'" in text
    assert "contiguous substring" in text and "do not add" in text
    assert surface.inventory == original_inventory and SKILL_TOOLS == original_schemas


def test_stage_permissions_remain_exact_and_unavailable_ids_stay_errors(surface):
    sidecar = build_public_skill_discovery(surface, "S3")
    assert {row["skill_id"] for row in sidecar["skills"]} == {
        "artifact-verification-med", "eva-workflow/stage-rollout"}
    for arguments in ({"skill_id": "bounded-plan-readiness-med", "stage": "S3"},
                      {"skill_id": "automed_eval/artifact-verification-med", "stage": "S3"},
                      {"skill_id": "artifact-verification-med", "stage": "E2E"}):
        with pytest.raises(ValueError):
            surface.call("load_skill", arguments, stage="S3")


def test_unverified_extra_or_missing_descriptor_is_not_advertised(surface):
    original = surface.call
    def changed(name, arguments, *, stage):
        result = original(name, arguments, stage=stage)
        result["matches"].append({"skill_id": "invented", "description": "Not verified."})
        return result
    surface.call = changed
    with pytest.raises(ValueError, match="inventory differ"):
        build_public_skill_discovery(surface, "S1")


def test_changed_canonical_tool_or_mutated_sidecar_fails(surface):
    sidecar = build_public_skill_discovery(surface, "S1")
    sidecar["skills"][0]["skill_id"] = "invented"
    with pytest.raises(ValueError, match="commitment"):
        render_public_skill_discovery(sidecar)
    real = surface.definitions["S1"]["search_skills"]
    surface.definitions["S1"]["search_skills"] = SimpleNamespace(
        name=real.name, description="changed", input_schema=real.input_schema)
    with pytest.raises(ValueError, match="canonical"):
        build_public_skill_discovery(surface, "S1")


@pytest.mark.parametrize("index,stage", list(enumerate(("S1", "S2", "S3", "S4", "S5"))))
def test_future_actor_phase_uses_only_current_public_descriptors(surface, index, stage):
    from training.automedbench_lite.track_actor import PHASES, phase_prompt

    phase, actual_stage, prompt, sidecar = phase_prompt(surface, index, "vqa")
    assert (phase, actual_stage) == PHASES[index][:2]
    assert actual_stage == sidecar["stage"] == stage
    assert prompt.startswith(PHASES[index][2])
    assert f"CURRENT phase {stage}" in prompt and "NOT-IN-HINT" not in prompt
    assert sidecar["counts_as_skill_use"] is False
    assert surface.calls == [("search_skills", {"query": " ", "stage": stage}, stage)]
    if stage == "S3":
        assert "15 distinct public cases" in prompt
