"""CPU integration fixtures, not real medical scores or serving attestations."""
from copy import deepcopy
import os
from pathlib import Path

import pytest

from eva_agent.rubrics import load_and_compile_registry
from training.eva_rsi.controller import read, write
from training.eva_rsi.evidence import EvidenceError, commitment, verify_evaluation
from training.eva_rsi.production import require_complete_baseline, require_real_evaluation
from test_eva_rsi_loop import setup, table

ROOT = Path(__file__).resolve().parents[1]


def composite_fixture(tmp_path, monkeypatch):
    import training.eva_rsi.composite_eval as composite
    registry = ROOT / "rubrics/source/domain-stage-tables.v1.json"
    model = tmp_path / "same-hf"
    proof = {"schema": "eva.verified-composite-evaluation.v1", "valid": True,
             "sources": {}, "feedback": [], "skill_content_id": "same-content",
             "checkpoint_equivalence_blake3": "e" * 64}
    feedback = {}
    for number, source in enumerate(("original", "supplement")):
        directory = tmp_path / source
        write(directory / "identity.json", {"exact_final_model_path": str(model), "fresh_identity": source})
        write(directory / "diagnostic.json", {"memory_pass": None, "synthetic_test_fixture": True})
        write(directory / "index.json", {"diagnostic_artifacts": [str(directory / "diagnostic.json")]})
        ref = commitment(directory / "identity.json")
        proof["sources"][source] = {"index": commitment(directory / "index.json"), "checkpoint_identity": ref}
        if source == "original":
            proof["comparison_checkpoint_identity"] = ref
        rubric = load_and_compile_registry(registry).resolve("automedbench-classification", f"S{number+1}")
        raw_identity = {"model_id": "same-model", "checkpoint_id": ref["blake3"],
                        "skill_catalog_id": "mounted-" + source, "judge_id": "same-judge"}
        score = rubric.score({item["item_id"]: 0 for item in rubric.items}).to_document()
        root = str(directory / "feedback")
        feedback[root] = {"valid": True, "status": "scored", "round_identity": raw_identity,
            "score": score, "domain": rubric.domain, "stage": rubric.stage, "case_id": "case-" + source,
            "judge_codex_receipt_blake3": str(number + 1) * 64}
        proof["feedback"].append({"root": root, "source_name": source,
            "judge_receipt_blake3": str(number + 1) * 64, "raw_round_identity": deepcopy(raw_identity)})
    path = tmp_path / "composite.json"
    write(path, {"schema": "eva.rsi-composite-evaluation-index.v1", "skill_selection": None})
    monkeypatch.setattr(composite, "verify_composite_index", lambda value: deepcopy(proof))
    monkeypatch.setattr("training.benchmark_feedback.automed_codex.verify_feedback",
                        lambda root, **kwargs: deepcopy(feedback[str(root)]))
    return path, {"model_path": str(model), "skill_catalog_id": "same-content"}, proof, feedback


def test_composite_normalizes_comparison_only_and_keeps_raw_receipts(tmp_path, monkeypatch):
    path, context, proof, feedback = composite_fixture(tmp_path, monkeypatch)
    before = deepcopy(feedback)
    result = verify_evaluation(path, context)
    assert feedback == before
    assert result["index"] == commitment(path)
    assert len(result["feedback"]) == 2 and result["memory_context_pass"] is None
    rows = result["comparison_identity_normalization"]["feedback"]
    assert rows[0]["raw_round_identity"] != rows[1]["raw_round_identity"]
    assert rows[0]["comparison_round_identity"] == rows[1]["comparison_round_identity"]
    assert result["composite_provenance"] == proof
    assert require_complete_baseline(read(path)) == proof
    assert require_real_evaluation({"previous_evaluation": result}) == read(path)


def test_composite_context_and_retained_proof_must_match(tmp_path, monkeypatch):
    path, context, proof, _ = composite_fixture(tmp_path, monkeypatch)
    with pytest.raises(EvidenceError, match="context_skill"):
        verify_evaluation(path, {**context, "skill_catalog_id": "other"})
    result = verify_evaluation(path, context)
    result["composite_provenance"]["skill_content_id"] = "other"
    with pytest.raises(EvidenceError, match="proof_changed"):
        require_real_evaluation({"previous_evaluation": result})


def attachment_fixture(tmp_path, table):
    controller = setup(tmp_path, table, mode="bad_evaluation")
    assert controller.run(execute=True, poll_seconds=.01)["status"] == "blocked"
    state, _ = controller.load()
    root = controller.root / "attempts" / state["active_attempt"]
    # Reap the fixture worker after its durable exit file; live-process guards
    # correctly reject even an unreaped same-birth worker.
    os.waitpid(read(root / "worker.json")["pid"], 0)
    # Synthetic worker emitted exit-zero despite invalid evidence. The explicit
    # fixture failure below tests a retained pre-turn launcher failure instead.
    exit_value = read(root / "exit.json")
    write(root / "exit.json", {**exit_value, "exit_code": 7})
    run = root / "benchmark" / "synthetic-run"
    run.mkdir(parents=True)
    write(root / "identity.json", {"synthetic_test_fixture": True})
    identity = commitment(root / "identity.json")
    write(root / "evaluation-actor-binding.json", {"equivalent_actor_cli": ["actor", "--run-root", str(run)],
                                                 "checkpoint_identity": identity})
    write(tmp_path / "original-index.json", {"benchmark_run_root": str(run),
        "actor_runtime_binding": commitment(root / "evaluation-actor-binding.json")})
    index = tmp_path / "composite.json"
    write(index, {"schema": "eva.rsi-composite-evaluation-index.v1"})
    verification = {"index": commitment(index), "synthetic_test_fixture": True,
        "proposal": {"status": "target_proposed", "s_target": "S1"},
        "composite_provenance": {"sources": {"original": {
            "index": commitment(tmp_path / "original-index.json"), "checkpoint_identity": identity}}}}
    controller.evaluation_verifier = lambda path, context: deepcopy(verification)
    return controller, root, index


def test_attach_then_resume_reconciles_without_relaunch_or_source_write(tmp_path, table):
    controller, root, index = attachment_fixture(tmp_path, table)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    initial = controller.status()
    record = controller.attach_evaluation_supplement(index, reason="user approved C-D only", execute=True)
    assert record["actors_launched"] == record["updates_credited"] == 0
    assert controller.status()["status"] == "blocked"
    controller.resume()
    result = controller.run(execute=True, max_transitions=1, poll_seconds=.01)
    assert result["retained_attempts"] == initial["retained_attempts"]
    assert result["phase"] == "skill_decision" and result["active_attempt"] is None
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    state, _ = controller.load()
    assert state["attempts"][-1]["status"] == "verified_with_retained_command_failure"
    assert Path(state["evaluation"]).is_relative_to(controller.root / "recovery-records")


def test_attachment_requires_explicit_execution_and_cannot_attach_twice(tmp_path, table):
    controller, root, index = attachment_fixture(tmp_path, table)
    with pytest.raises(EvidenceError, match="explicit"):
        controller.attach_evaluation_supplement(index, reason="user approved")
    controller.attach_evaluation_supplement(index, reason="user approved", execute=True)
    with pytest.raises(EvidenceError, match="already_attached"):
        controller.attach_evaluation_supplement(index, reason="repeat", execute=True)


def test_attachment_rejects_a_different_actor_runtime_binding(tmp_path, table):
    controller, root, index = attachment_fixture(tmp_path, table)
    path = tmp_path / "original-index.json"
    value = read(path)
    value["actor_runtime_binding"]["blake3"] = "0" * 64
    write(path, value)
    with pytest.raises(EvidenceError, match="original_source_differs"):
        controller.attach_evaluation_supplement(index, reason="user approved", execute=True)
