"""Focused CPU contracts for disjoint, same-checkpoint evaluation sources."""
from copy import deepcopy

import pytest

from training.automedbench_lite.track_adapter import BY_TRACK
from training.eva_rsi import composite_eval
from training.eva_rsi.terminal_attempts import (
    verify_terminal_attempts, verify_terminal_track_source,
)
from test_terminal_attempt_gate import replace_doc, seven


def test_all_seven_wrapper_shape_unchanged_and_subset_reopens_same_actor(seven):
    document, _ = seven
    historical = verify_terminal_attempts(document)
    assert set(historical) == {"schema", "valid", "run_attempt_document_blake3",
        "checkpoint_identity_blake3", "tracks", "provider_calls",
        "model_score_threshold", "unreachable_is_not_zero"}
    selected = ("synthesis", "segmentation", "vqa", "report", "enhancement")
    subset = verify_terminal_track_source(document, selected_tracks=selected,
        expected_attempt_tracks=tuple(BY_TRACK))
    assert subset["schema"] == "eva.verified-terminal-track-source.v1"
    assert subset["selected_tracks"] == list(selected)
    assert subset["source_attempt_tracks"] == list(BY_TRACK)
    assert subset["tracks"] == {track: historical["tracks"][track] for track in selected}


def test_two_track_supplement_actor_is_verified_as_two_not_relabelled_all_seven(seven):
    document, _ = seven
    from pathlib import Path
    import json
    root = Path(document["benchmark_run_root"]) / "track-rollouts"
    attempt = json.loads((root / "attempt.json").read_text())
    attempt.update(tracks=list(composite_eval.SUPPLEMENTED), planned_coding_rollouts=2)
    replace_doc(root / "attempt.json", attempt)
    summary = json.loads((root / "summary.json").read_text())
    summary["tracks"] = [row for row in summary["tracks"]
                         if row["track"] in composite_eval.SUPPLEMENTED]
    replace_doc(root / "summary.json", summary)
    proof = verify_terminal_track_source(document,
        selected_tracks=composite_eval.SUPPLEMENTED,
        expected_attempt_tracks=composite_eval.SUPPLEMENTED)
    assert proof["source_attempt_tracks"] == ["classification", "detection"]
    assert set(proof["tracks"]) == set(composite_eval.SUPPLEMENTED)


def _source(name, tracks, stable):
    terminal = {track: {"attempted_stages": ["S1"],
        "unreachable_stages": ["S2", "S3", "S4", "S5"],
        "sources": [], "policy_terminal": None, "unreachable_scores": None}
        for track in tracks}
    return {"index": {"path": f"/{name}-index", "blake3": name},
        "benchmark_run_root": f"/fixture/{name}", "checkpoint_identity": {
            "path": f"/{name}-identity", "blake3": name},
        "checkpoint": {"identity": {"path": f"/{name}-identity", "blake3": name},
            "stable": stable, "stable_blake3": "a" * 64},
        "skill_content_binding": {"path": f"/{name}-skills", "blake3": name},
        "skill_content_id": "b" * 64, "mounted_catalog_blake3": "c" * 64,
        "skill_attempt_document_blake3": "d" * 64,
        "attempt_document_blake3": "e" * 64, "selected_tracks": list(tracks),
        "terminal_proof": {"tracks": terminal}, "judge_attempt_isolation": [],
        "feedback": [{"root": f"/{name}-feedback", "source_name": name,
            "track": tracks[0], "stage": "S1", "judge_receipt_blake3": "f" * 64,
            "raw_round_identity": {"checkpoint_id": name}}]}


def _outer():
    return {"schema": composite_eval.SCHEMA, "composition_mode": composite_eval.MODE,
        "original_index": {"source": "original"}, "supplement_index": {"source": "supplement"},
        "original_skill_content_binding": {"source": "original"},
        "supplement_skill_content_binding": {"source": "supplement"},
        "supplemented_tracks": list(composite_eval.SUPPLEMENTED),
        "track_sources": {track: "supplement" if track in composite_eval.SUPPLEMENTED else "original"
                          for track in BY_TRACK},
        "skill_selection": None, "source_indexes_mutated": False,
        "synthetic_seven_track_run_created": False,
        "canonical_source_schemas_unchanged": True, "missing_is_not_zero": True}


def test_composite_union_preserves_raw_sources_and_requires_stable_checkpoint(monkeypatch):
    stable = {"same_hf": True}
    sources = {"original": _source("original", composite_eval.ORIGINAL, stable),
               "supplement": _source("supplement", composite_eval.SUPPLEMENTED, stable)}
    monkeypatch.setattr(composite_eval, "_source",
        lambda *args, source_name, **kwargs: deepcopy(sources[source_name]))
    monkeypatch.setattr(composite_eval, "read_document", lambda path: {"tracks": []})
    monkeypatch.setattr(composite_eval, "read", lambda path: {"exact_final_model_path": "/fixture/hf"})
    monkeypatch.setattr(composite_eval, "_zero_turn_eligibility",
        lambda run, track, identity, rows: {"track": track, "actual_turn_count": 0,
            "eligible_for_supplement": True})
    proof = composite_eval.verify_composite_index(_outer())
    assert proof["valid"] and set(proof["tracks"]) == set(BY_TRACK)
    assert proof["tracks"]["classification"]["source_name"] == "supplement"
    assert proof["tracks"]["synthesis"]["source_name"] == "original"
    assert [row["source_name"] for row in proof["feedback"]] == ["original", "supplement"]
    assert proof["comparison_checkpoint_identity"] == sources["original"]["checkpoint_identity"]
    assert proof["originals_mutated"] is False and proof["provider_calls"] == 0

    changed = deepcopy(sources)
    changed["supplement"]["checkpoint"]["stable"] = {"same_hf": False}
    monkeypatch.setattr(composite_eval, "_source",
        lambda *args, source_name, **kwargs: deepcopy(changed[source_name]))
    with pytest.raises(ValueError, match="checkpoint_lineage_or_assets_differ"):
        composite_eval.verify_composite_index(_outer())


def test_composite_track_map_is_fixed_not_a_fake_all_seven_root(monkeypatch):
    value = _outer()
    value["track_sources"]["detection"] = "original"
    with pytest.raises(ValueError, match="composite_index_policy_invalid"):
        composite_eval.verify_composite_index(value)
