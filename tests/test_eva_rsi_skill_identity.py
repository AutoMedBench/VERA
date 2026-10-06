"""CPU-only explicit RSI normalization; Judge fixtures are never real grades."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from eva_agent.codex_pipeline.skill_identity import SkillContentIdentityError
from eva_agent.rubrics import load_and_compile_registry
from training.automedbench_lite.adapter import write_once
from training.eva_rsi.evidence import EvidenceError, commitment, verify_evaluation
from training.eva_rsi.skill_identity import build_skill_content_binding, verify_skill_content_binding
from test_skill_content_identity import skill_mount_factory

ROOT = Path(__file__).resolve().parents[1]


def attempt(args):
    root = args["materialization_root"].parent.parent.parent
    path = root / "track-rollouts/attempt.json"
    write_once(path, {"schema": "eva.automedbench-track-attempt.v1", "skill_inventory": args["inventory"],
        "verified_skill_catalog_blake3": args["mounted_catalog_blake3"], "synthetic_test_fixture": True})
    return root


def sidecar(args):
    root = attempt(args)
    value = build_skill_content_binding(root, legacy_manifest_path=args["legacy_manifest_path"])
    path = root / "skill-content-binding.json"
    path.write_text(json.dumps(value))
    return root, path, value


def fixture_evaluation(tmp_path, monkeypatch, source, *, version=2):
    root, binding_path, binding = sidecar(source)
    model = tmp_path / "model"
    identity_path = tmp_path / "checkpoint-identity.json"
    identity_path.write_text(json.dumps({"exact_final_model_path": str(model)}))
    diagnostic = tmp_path / "memory-diagnostic.json"
    diagnostic.write_text('{"synthetic_test_fixture": true, "memory_context_pass": null}')
    rubric = load_and_compile_registry(ROOT / "rubrics/source/domain-stage-tables.v1.json").resolve("automedbench-classification", "S1")
    score = rubric.score({item["item_id"]: 0 for item in rubric.items}).to_document()
    feedback = {"valid": True, "status": "scored", "domain": rubric.domain, "stage": "S1", "case_id": "fixture-track",
        "round_identity": {"model_id": "fixture-model", "checkpoint_id": commitment(identity_path)["blake3"],
            "skill_catalog_id": binding["mounted_catalog_blake3"], "judge_id": "fixture-only-judge"},
        "judge_codex_receipt_blake3": "1" * 64, "score": score}
    requested = []
    def verified(path, **kwargs):
        requested.append(kwargs)
        result = deepcopy(feedback)
        if kwargs.get("include_skill_source_binding"):
            result["skill_source_binding"] = {"run_attempt_document_blake3": binding["attempt"]["document_blake3"],
                "mounted_skill_catalog_blake3": binding["mounted_catalog_blake3"]}
        return result
    monkeypatch.setattr("training.benchmark_feedback.automed_codex.verify_feedback", verified)
    index = {"schema": f"eva.rsi-evaluation-index.v{version}", "evaluation_mode": "diagnostic_subset",
        "checkpoint_identity": str(identity_path), "benchmark_run_root": str(root),
        "feedback_roots": [str(root / "fixture-feedback")], "diagnostic_artifacts": [str(diagnostic)]}
    if version == 2:
        index.update(skill_catalog_identity_mode="verified-content-v1", skill_content_identity=commitment(binding_path))
    index_path = root / "evaluation-index.json"
    index_path.write_text(json.dumps(index))
    return index_path, {"model_path": str(model), "skill_catalog_id": binding["content_identity_blake3"]}, feedback, binding, requested


def test_v2_reopens_all_files_and_preserves_raw_round_identity(tmp_path, monkeypatch, skill_mount_factory):
    first = fixture_evaluation(tmp_path, monkeypatch, skill_mount_factory("first"))
    original = deepcopy(first[2])
    a = verify_evaluation(first[0], first[1])
    second = fixture_evaluation(tmp_path, monkeypatch, skill_mount_factory("second"))
    b = verify_evaluation(second[0], first[1])
    assert a["proposal"]["round_identity"] == b["proposal"]["round_identity"]
    assert first[2] == original
    assert a["skill_identity_normalization"]["feedback"][0]["raw_round_identity"] == original["round_identity"]
    assert a["skill_identity_normalization"]["mounted_catalog_blake3"] != b["skill_identity_normalization"]["mounted_catalog_blake3"]
    assert first[4] == [{"include_skill_source_binding": True}]
    assert a["memory_context_pass"] is None


def test_v1_remains_strict_mounted_id_no_content_fallback(tmp_path, monkeypatch, skill_mount_factory):
    path, context, feedback, binding, calls = fixture_evaluation(tmp_path, monkeypatch, skill_mount_factory("old"), version=1)
    with pytest.raises(EvidenceError, match="feedback_checkpoint_or_catalog_differs"):
        verify_evaluation(path, context)
    result = verify_evaluation(path, {**context, "skill_catalog_id": binding["mounted_catalog_blake3"]})
    assert result["proposal"]["round_identity"] == feedback["round_identity"]
    assert "skill_identity_normalization" not in result
    assert calls == [{}, {}]


@pytest.mark.parametrize("field", ["skill_catalog_identity_mode", "sidecar_commitment", "source_attempt", "mounted_identity"])
def test_v2_fail_closed_for_missing_optin_or_wrong_binding(tmp_path, monkeypatch, skill_mount_factory, field):
    path, context, feedback, binding, _ = fixture_evaluation(tmp_path, monkeypatch, skill_mount_factory("invalid"))
    index = json.loads(path.read_text())
    if field == "skill_catalog_identity_mode":
        del index[field]
    elif field == "sidecar_commitment":
        index["skill_content_identity"]["blake3"] = "0" * 64
    elif field == "source_attempt":
        binding["attempt"]["document_blake3"] = "0" * 64
    else:
        feedback["round_identity"]["skill_catalog_id"] = "0" * 64
    path.write_text(json.dumps(index))
    with pytest.raises(EvidenceError):
        verify_evaluation(path, context)


def test_changed_native_body_not_admitted_under_old_context(tmp_path, skill_mount_factory):
    _, _, old = sidecar(skill_mount_factory("old-content"))
    root, path, _ = sidecar(skill_mount_factory("new-content", native=b"An explicitly different native body."))
    with pytest.raises(SkillContentIdentityError, match="skill_content_identity_differs"):
        verify_skill_content_binding(path, expected_run_root=root, expected_content_id=old["content_identity_blake3"])


def test_sidecar_is_not_trusted_without_actual_reopen(tmp_path, skill_mount_factory):
    args = skill_mount_factory("tamper")
    root, path, value = sidecar(args)
    victim = Path(args["inventory"][0]["path"])
    victim.chmod(0o600)
    victim.write_bytes(b"Changed after the sidecar was created.")
    victim.chmod(0o400)
    with pytest.raises(SkillContentIdentityError, match="retained_skill_bytes_differ"):
        verify_skill_content_binding(path, expected_run_root=root, expected_content_id=value["content_identity_blake3"])
