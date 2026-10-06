"""Two explicitly approved, one-shot Report S4/S5 Judges; no actor replay."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib
from pathlib import Path
import subprocess
from uuid import uuid4

from training.automedbench_lite.adapter import file_digest, read_document, write_once
from training.automedbench_lite.judge_attempt_isolation import completed_invalid_verdict
from training.automedbench_lite.track_feedback import prepare_track_feedback
from training.benchmark_feedback.automed_codex import judge_once, read_json, verify_feedback, write_private

STAGES = ("S4", "S5")
POLICY = "user-approved-report-s4s5-one-shot-v1"
FAILURE_CATEGORY = "judge hard-gate or summary claim differs"


def commit(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("rejudge_source_topology")
    path = path.resolve(strict=True)
    return {"path": str(path), "blake3": file_digest(path)}


def reopen(reference):
    if commit(reference["path"]) != reference:
        raise ValueError("rejudge_source_changed")
    return Path(reference["path"])


def execution_binding():
    modules = ("eva_agent.codex_pipeline.adapter", "eva_agent.training.slime_agent_judge",
               "eva_agent.pipeline.judge_material_view", "eva_agent.training.agent_judge_worker")
    sources = {name: commit(importlib.import_module(name).__file__) for name in modules}
    core = Path(sources[modules[0]]["path"]).parents[3]
    revision = subprocess.check_output(["git", "-C", str(core), "rev-parse", "HEAD"], text=True).strip()
    return {"core_root": str(core), "core_revision": revision, "sources": sources,
            "producer": commit(__file__), "feedback_adapter": commit(prepare_track_feedback.__code__.co_filename),
            "judge_entrypoint": commit(judge_once.__code__.co_filename)}


def _old_failures(observer_root, stage):
    path = Path(observer_root) / "stages" / "report" / stage / "feedback" / stage / "judge-attempt-isolation.json"
    policy = read_document(path)
    if (policy.get("track") != "report" or policy.get("stage") != stage
            or policy.get("status") != "unavailable" or policy.get("replacement_limit_exhausted") is not True):
        raise ValueError("original_report_failure_scope_differs")
    failures = []
    for reference in policy["attempt_results"]:
        result_path = reopen(reference)
        result = read_document(result_path)
        proof = completed_invalid_verdict(result_path.parent)
        if result.get("status") != "unavailable" or proof["category"] != FAILURE_CATEGORY:
            raise ValueError("original_report_failure_category_differs")
        failures.append({"result": reference, "proof": proof})
    if not failures:
        raise ValueError("original_report_failure_missing")
    return {"isolation": commit(path), "failures": failures,
            "prepared_source": policy["prepared_source"]}


def prepare_packet(*, run_root, original_observer_root, checkpoint_identity,
                   actor_runtime_binding, registry_path, output_root):
    """Provider-free preparation against the unchanged original actor evidence."""
    run_root, output_root = Path(run_root).resolve(strict=True), Path(output_root).resolve()
    observer = Path(original_observer_root).resolve(strict=True)
    if output_root.is_relative_to(run_root) or output_root.is_relative_to(observer):
        raise ValueError("rejudge_output_overlaps_source")
    binding = execution_binding()
    failures = {stage: _old_failures(observer, stage) for stage in STAGES}
    source = {"actor_attempt": commit(run_root / "track-rollouts/attempt.json"),
              "run_manifest": commit(run_root / "track-run-manifest.json"),
              "checkpoint_identity": commit(checkpoint_identity),
              "actor_runtime_binding": commit(actor_runtime_binding), "registry": commit(registry_path)}
    runtime = read_json(Path(actor_runtime_binding))
    output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    stages = {}
    for stage in STAGES:
        target = output_root / stage
        rollout, _, preflight = prepare_track_feedback(run_root=run_root,
            checkpoint_identity=Path(checkpoint_identity), output_root=target, track="report", stage=stage,
            registry_path=Path(registry_path), material_view="policy-visible-audit-v2",
            context_terminal_source_binding=runtime)
        if not preflight.get("judge_eligible"):
            raise ValueError("original_report_prefix_not_judge_eligible")
        old = read_json(reopen(failures[stage]["prepared_source"]["rollout.json"]))
        old_preflight = read_json(reopen(failures[stage]["prepared_source"]["preflight.json"]))
        # Fresh projection UUIDs are permitted; the source turns/workspaces/rubric are not changed.
        for key in ("source_turns", "checkpoint_identity_blake3"):
            if rollout["provider_metadata"][key] != old["provider_metadata"][key]:
                raise ValueError("original_report_source_projection_differs")
        if (rollout["rubric_table"] != old["rubric_table"]
                or preflight["rubric_digest"] != old_preflight["rubric_digest"]):
            raise ValueError("original_report_rubric_differs")
        stages[stage] = {"root": str(target), "prepared": {
            name: commit(target / name) for name in ("preflight.json", "rollout.json")},
            "original_unavailable": failures[stage], "source_turns": rollout["provider_metadata"]["source_turns"],
            "rubric_digest": preflight["rubric_digest"]}
    return write_once(output_root / "packet.json", {"schema": "eva.approved-report-rejudge-packet.v1",
        "policy": POLICY, "packet_id": str(uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_run_root": str(run_root), "source": source, "execution_binding": binding,
        "stages": stages, "permitted_pairs": [["report", stage] for stage in STAGES],
        "semantic_attempts_per_stage": 1, "retry_count": 0, "maximum_parallel_judges": 2,
        "native_turn_timeout_seconds": 600, "actor_rerun": False, "canonical_rubric_changed": False,
        "judge_calls": 0, "provider_free_preparation": True})


def _verify_packet(output_root, *, current_binding=False):
    output_root = Path(output_root).resolve(strict=True)
    packet = read_document(output_root / "packet.json")
    if (packet.get("policy") != POLICY or set(packet.get("stages", {})) != set(STAGES)
            or packet.get("permitted_pairs") != [["report", stage] for stage in STAGES]
            or packet.get("retry_count") != 0 or packet.get("semantic_attempts_per_stage") != 1
            or packet.get("native_turn_timeout_seconds") != 600):
        raise ValueError("rejudge_packet_scope_differs")
    if current_binding and execution_binding() != packet["execution_binding"]:
        raise ValueError("rejudge_execution_source_changed")
    for reference in packet["source"].values():
        reopen(reference)
    for stage, row in packet["stages"].items():
        if Path(row["root"]) != output_root / stage:
            raise ValueError("rejudge_stage_root_differs")
        for reference in row["prepared"].values():
            reopen(reference)
        reopen(row["original_unavailable"]["isolation"])
        for failure in row["original_unavailable"]["failures"]:
            reopen(failure["result"])
            for key in ("failure", "judge_receipt", "native_attempt"):
                reopen(failure["proof"][key])
    return packet


def execute_packet(output_root, *, execute=False):
    """Consume one exclusive packet claim, settle both calls; never resume/retry it."""
    if not execute:
        raise ValueError("explicit_execute_required")
    output_root = Path(output_root).resolve(strict=True)
    packet = _verify_packet(output_root, current_binding=True)
    claim = write_once(output_root / "execute-claim.json", {"schema": "eva.approved-report-rejudge-claim.v1",
        "claim_id": str(uuid4()), "packet": commit(output_root / "packet.json"),
        "semantic_attempts_per_stage": 1, "retry_count": 0})

    def run(stage):
        target = Path(packet["stages"][stage]["root"])
        write_once(target / "one-shot-claim.json", {"schema": "eva.approved-report-stage-claim.v1",
            "stage": stage, "packet_claim_blake3": claim["document_blake3"], "retry_count": 0})
        result = {"schema": "eva.approved-report-stage-result.v1", "stage": stage, "track": "report",
                  "root": str(target), "semantic_attempt_count": 1, "retry_count": 0, "score": None}
        try:
            feedback = judge_once(target, registry_path=Path(packet["source"]["registry"]["path"]),
                                  native_turn_timeout_seconds=600)
            write_private(target / "feedback.json", feedback)
            verified = verify_feedback(target)
            if verified.get("valid") is not True or verified.get("status") != "scored":
                raise ValueError("rejudge_not_independently_verified")
            write_private(target / "verification.json", verified)
            result.update(status="verified", score=verified["score"],
                          verification=commit(target / "verification.json"))
        except Exception as error:
            result.update(status="unavailable", error_type=type(error).__name__)
            try:
                result["completed_invalid_verdict"] = completed_invalid_verdict(target, error)
            except Exception:
                pass  # Native error artifacts remain untouched; no raw exception text is copied.
        write_once(target / "one-shot-result.json", result)
        return commit(target / "one-shot-result.json")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, STAGES))
    return write_once(output_root / "execution-result.json", {
        "schema": "eva.approved-report-rejudge-execution.v1", "policy": POLICY,
        "packet": commit(output_root / "packet.json"), "claim": commit(output_root / "execute-claim.json"),
        "stage_results": results, "judge_attempts_made": 2, "retry_count": 0, "actor_calls": 0,
        "all_verified": all(read_document(Path(ref["path"]))["status"] == "verified" for ref in results)})


def _verified_execution(packet_root):
    """Reopen one claimed call per stage and its canonical persisted assessment."""
    packet_root = Path(packet_root).resolve(strict=True)
    packet = _verify_packet(packet_root)
    for reference in packet["execution_binding"]["sources"].values():
        reopen(reference)
    for key in ("producer", "feedback_adapter", "judge_entrypoint"):
        reopen(packet["execution_binding"][key])
    execution = read_document(packet_root / "execution-result.json")
    claim_path = reopen(execution["claim"])
    claim = read_document(claim_path)
    if (claim_path != packet_root / "execute-claim.json"
            or execution.get("packet") != commit(packet_root / "packet.json")
            or claim.get("packet") != execution["packet"]
            or execution.get("policy") != POLICY or execution.get("all_verified") is not True
            or execution.get("judge_attempts_made") != 2 or execution.get("retry_count") != 0
            or claim.get("semantic_attempts_per_stage") != 1 or claim.get("retry_count") != 0
            or len(execution.get("stage_results", [])) != 2):
        raise ValueError("approved_rejudge_execution_not_two_verified_one_shots")
    verified = []
    for stage, reference in zip(STAGES, execution["stage_results"]):
        target = packet_root / stage
        if reopen(reference) != target / "one-shot-result.json":
            raise ValueError("approved_rejudge_result_root_differs")
        result = read_document(target / "one-shot-result.json")
        stage_claim = read_document(target / "one-shot-claim.json")
        if (result.get("stage") != stage or result.get("track") != "report"
                or result.get("root") != str(target) or result.get("status") != "verified"
                or result.get("semantic_attempt_count") != 1 or result.get("retry_count") != 0
                or stage_claim.get("stage") != stage or stage_claim.get("retry_count") != 0
                or stage_claim.get("packet_claim_blake3") != claim["document_blake3"]):
            raise ValueError("approved_rejudge_stage_claim_differs")
        retained = read_json(reopen(result["verification"]))
        feedback = verify_feedback(target)
        native = read_json(target / "judge-attempt.json")
        if (feedback.get("valid") is not True or feedback.get("status") != "scored"
                or feedback.get("case_id") != "report" or feedback.get("stage") != stage
                or feedback.get("score") != result["score"]
                or {k: v for k, v in retained.items() if k != "verification_id"}
                   != {k: v for k, v in feedback.items() if k != "verification_id"}
                or native.get("semantic_attempt_count") != 1
                or native.get("retry_count") != 0 or native.get("automatic_fallback") is not False
                or native.get("native_turn_timeout_seconds") != 600):
            raise ValueError("approved_rejudge_canonical_verification_differs")
        verified.append({"root": str(target), "stage": stage, "track": "report",
            "judge_receipt_blake3": feedback["judge_codex_receipt_blake3"],
            "round_identity": feedback["round_identity"], "result": reference})
    return packet, verified


def _validate_index_sources(original, packet, standalone):
    if ("approved_stage_rejudgments" in original
            or original.get("benchmark_run_root") != packet["benchmark_run_root"]
            or commit(original["checkpoint_identity"]) != packet["source"]["checkpoint_identity"]
            or original.get("actor_runtime_binding") != packet["source"]["actor_runtime_binding"]):
        raise ValueError("approved_rejudge_original_index_source_differs")
    roots = original.get("feedback_roots")
    if not isinstance(roots, list) or len(set(roots)) != len(roots):
        raise ValueError("approved_rejudge_original_roots_invalid")
    for root in roots:
        preflight = read_json(Path(root) / "preflight.json")
        if preflight.get("case_id") == "report" and preflight.get("stage") in STAGES:
            raise ValueError("approved_rejudge_cannot_replace_valid_original_grade")
    if set(roots) & set(standalone):
        raise ValueError("approved_rejudge_roots_overlap")


def compose_index(*, original_index, packet_root, output_root):
    """New diagnostic index: original roots unchanged, then new S4 and S5 only."""
    original_path = Path(original_index).resolve(strict=True)
    original = read_document(original_path)
    packet_root = Path(packet_root).resolve(strict=True)
    packet, results = _verified_execution(packet_root)
    standalone = [row["root"] for row in results]
    _validate_index_sources(original, packet, standalone)
    output = Path(output_root).resolve()
    if any(output.is_relative_to(path) for path in (
            original_path.parent, packet_root, Path(packet["benchmark_run_root"]))):
        raise ValueError("approved_rejudge_index_output_overlaps_source")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    sidecar = {"schema": "eva.approved-report-rejudge-index-sidecar.v1", "policy": POLICY,
        "original_index": commit(original_path), "packet": commit(packet_root / "packet.json"),
        "execution": commit(packet_root / "execution-result.json"),
        "permitted_pairs": [["report", stage] for stage in STAGES],
        "standalone_roots": standalone, "stage_results": results,
        "original_isolation_references": original.get("postround_judge_attempt_isolation"),
        "original_failure_records": {stage: packet["stages"][stage]["original_unavailable"] for stage in STAGES},
        "execution_binding": packet["execution_binding"], "actor_reruns": 0,
        "failed_records_rewritten": False, "automatic_replacements": False,
        "diagnostic_only": True, "training_admission": False}
    write_once(output / "approved-stage-rejudgments.json", sidecar)
    index = {key: value for key, value in original.items() if key != "document_blake3"}
    index.update(feedback_roots=original["feedback_roots"] + standalone,
                 approved_stage_rejudgments=commit(output / "approved-stage-rejudgments.json"))
    verify_rejudge_index_sidecar(index)
    return write_once(output / "evaluation-index.json", index)


def verify_rejudge_index_sidecar(index):
    """Source-strict outer policy, reusable by the composite admission consumer."""
    reference = index["approved_stage_rejudgments"]
    sidecar = read_document(reopen(reference))
    if (sidecar.get("schema") != "eva.approved-report-rejudge-index-sidecar.v1"
            or sidecar.get("policy") != POLICY
            or sidecar.get("permitted_pairs") != [["report", stage] for stage in STAGES]
            or sidecar.get("actor_reruns") != 0 or sidecar.get("failed_records_rewritten") is not False
            or sidecar.get("automatic_replacements") is not False):
        raise ValueError("approved_rejudge_sidecar_policy_differs")
    original = read_document(reopen(sidecar["original_index"]))
    packet_path = reopen(sidecar["packet"])
    if reopen(sidecar["execution"]) != packet_path.parent / "execution-result.json":
        raise ValueError("approved_rejudge_execution_location_differs")
    packet, results = _verified_execution(packet_path.parent)
    standalone = [row["root"] for row in results]
    _validate_index_sources(original, packet, standalone)
    if (sidecar["standalone_roots"] != standalone or sidecar["stage_results"] != results
            or sidecar["execution_binding"] != packet["execution_binding"]
            or sidecar["original_failure_records"] != {
                stage: packet["stages"][stage]["original_unavailable"] for stage in STAGES}
            or sidecar.get("original_isolation_references") != original.get("postround_judge_attempt_isolation")
            or index.get("feedback_roots") != original["feedback_roots"] + standalone):
        raise ValueError("approved_rejudge_selection_differs")
    # Every original field, including policy, source and coverage metadata, stays untouched.
    if ({k: v for k, v in index.items() if k not in (
            "document_blake3", "feedback_roots", "approved_stage_rejudgments")}
            != {k: v for k, v in original.items() if k not in ("document_blake3", "feedback_roots")}):
        raise ValueError("approved_rejudge_original_index_fields_changed")
    return {"valid": True, "original_index": sidecar["original_index"],
            "standalone_roots": standalone, "stage_results": results, "sidecar": reference}
