from __future__ import annotations

from itertools import product
import json
from pathlib import Path

from blake3 import blake3
import pytest

from eva_agent.rubrics import load_and_compile_registry, RubricRegistryError, RubricScoreError


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "rubrics/source"
OLD = SOURCE / "domain-stage-tables.v1.json"
NEW = SOURCE / "domain-stage-tables.v2.json"
AUDIT = json.loads((SOURCE / "automedbench-lite-additions-audit.v1.json").read_text())
TRACKS = ("synthesis", "enhancement", "vqa", "report")
STAGES = ("S1", "S2", "S3", "S4", "S5")
REVISION = "8928073d5c3f3b842a4a4278d9b44f6e8ceaa9c5"


@pytest.fixture(scope="module")
def registry():
    return load_and_compile_registry(NEW)


def test_old_source_bytes_and_all_42_compiled_tables_remain_exact(registry):
    assert "blake3:" + blake3(OLD.read_bytes()).hexdigest() == AUDIT["prior_source_blake3"]
    previous = load_and_compile_registry(OLD)
    assert previous.digest == AUDIT["prior_compiled_registry_blake3"]
    old_doc, new_doc = json.loads(OLD.read_text()), json.loads(NEW.read_text())
    assert new_doc["rubrics"][:42] == old_doc["rubrics"]
    assert previous.registry_version == 2 and registry.registry_version == 3
    assert previous.registry_id == registry.registry_id
    assert len(previous.rubrics) == 42
    for old in previous.rubrics:
        current = registry.resolve(old.domain, old.stage)
        assert current.to_document() == old.to_document()
        assert current.digest == old.digest
    with pytest.raises(RubricRegistryError):
        previous.resolve("automedbench-report", "S1")


def test_twenty_additions_are_explicitly_versioned_and_stage_local(registry):
    assert len(registry.rubrics) == 62
    assert sum(len(r.items) for r in registry.rubrics) == 367
    assert registry.digest == AUDIT["compiled_registry_blake3"]
    added = [r for r in registry.rubrics if r.domain in {"automedbench-" + t for t in TRACKS}]
    assert {(r.domain, r.stage) for r in added} == {
        ("automedbench-" + t, s) for t, s in product(TRACKS, STAGES)
    }
    assert sum(len(r.items) for r in added) == AUDIT["added_items"] == 115
    for r in added:
        assert 5 <= len(r.items) <= 10
        assert sum(i["normalized_weight_bps"] for i in r.items) == 10000
        assert all(i["weight"] == 1 and i["atomic"] and i["observable"] for i in r.items)
        assert all("hard_gate" not in i for i in r.items)
        assert {e["evidence_stage"] for i in r.items for e in i["evidence_selectors"]} == {r.stage}
        assert {e["source"] for i in r.items for e in i["evidence_selectors"]} == {"context", "workspace"}
    with pytest.raises(RubricRegistryError):
        registry.resolve("automedbench-report", "E2E")


@pytest.mark.parametrize("track,stage", tuple(product(TRACKS, STAGES)))
def test_shared_judge_reward_object_and_real_item_scores(registry, track, stage):
    binding = registry.bind_sandbox("740f1718-a65f-4b56-b4c3-1028f372bf2d", "automedbench-" + track, stage)
    table = binding.for_benchmark_judging()
    assert table is binding.for_rollout_reward()
    scores = {item["item_id"]: 0 for item in table.items}
    assert table.score(scores).reward_bps == 0  # no automatic Lite waiver gift
    first = next(iter(scores))
    scores[first] = 10000
    result = table.score(scores)
    n = len(scores)
    assert result.reward_bps == result.weighted_score_bps == (10000 + n // 2) // n
    assert result.hard_gate_passed  # not a fabricated host or clinical gate
    assert table.score(dict.fromkeys(scores, 10000)).reward_bps == 10000
    with pytest.raises(RubricScoreError):
        table.score({k: v for k, v in scores.items() if k != first})


def test_every_new_item_has_a_pinned_source_or_explicit_supplement(registry):
    sources = {row["item_id"]: row for row in AUDIT["item_sources"]}
    assert len(sources) == 115
    seen, supplements = set(), 0
    for track, stage in product(TRACKS, STAGES):
        for item in registry.resolve("automedbench-" + track, stage).items:
            row = sources[item["item_id"]]
            seen.add(item["item_id"])
            assert row["domain"] == "automedbench-" + track and row["stage"] == stage
            assert row["source_file"] in AUDIT["source_files"]
            assert row["source_locator"]
            p = item["provenance"]
            assert row["origin"] == p["origin"]
            if p["origin"] == "benchmark":
                assert p["source_revision"] == REVISION
                assert p["source_file"] == row["source_file"]
                assert p["source_benchmark"] == "AutoMedBench-Lite"
                assert p["extraction_status"] == "adapted"
                assert "supplemental" not in item["labels"]
            else:
                supplements += 1
                assert {"supplemental", p["supplemental_label"]} <= set(item["labels"])
                assert REVISION in p["rationale"] and row["source_file"] in p["rationale"]
    assert seen == sources.keys() and supplements > 0


def test_source_audit_distinguishes_native_contract_from_eva_feedback():
    assert AUDIT["revision"] == REVISION and AUDIT["native_score_equivalence"] is False
    assert AUDIT["private_reference_payloads_read"] is False
    assert AUDIT["provider_calls"] == AUDIT["gpu_calls"] == 0
    assert len(AUDIT["source_files"]) == 32
    assert sum(r.get("upstream_manifest_pin_verified", False) for r in AUDIT["source_files"].values()) == 29
    for row in AUDIT["source_files"].values():
        assert row["bytes"] > 0 and row["blake3"].startswith("blake3:") and len(row["blake3"]) == 71
    assert AUDIT["tracks"]["synthesis"]["lite_model"]["checkpoint"] == "weight/PlainCNN_trilinear_interpolation_x4.pth"
    assert len(AUDIT["tracks"]["synthesis"]["native_fixed_items_not_rl_achievements"]) == 3
    assert len(AUDIT["tracks"]["vqa"]["native_fixed_items_not_rl_achievements"]) == 1
    assert AUDIT["tracks"]["report"]["native_fixed_items_not_rl_achievements"] == []


def test_observable_track_specific_constraints_not_stale_cross_track_rules(registry):
    def desc(track, stage, slug):
        match = next(r for r in AUDIT["item_sources"] if r["domain"] == "automedbench-" + track
                     and r["stage"] == stage and r["slug"] == slug)
        return next(i["description"] for i in registry.resolve("automedbench-" + track, stage).items
                    if i["item_id"] == match["item_id"])
    assert "ct.nii.gz" in desc("synthesis", "S1", "input-task")
    assert "exactly 15" in desc("vqa", "S3", "calibration-count")
    assert "[-1024, 3000]" in desc("enhancement", "S3", "range")
    assert "40 to 8000" in desc("report", "S4", "length")
    assert "finite zero SSIM is accepted" in desc("synthesis", "S5", "accepted-result")
    assert "private gold" in desc("vqa", "S3", "public-only")
    assert "not a clinical-summary proxy" in desc("report", "S4", "execution")
