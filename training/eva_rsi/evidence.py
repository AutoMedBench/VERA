"""Offline adapters for existing real Judge and Slime evidence, never launch flags."""
from __future__ import annotations

import json
import math
from pathlib import Path

from blake3 import blake3

from eva_agent.rubrics import load_and_compile_registry
from eva_agent.training.coevolution import (
    EvaluationStatus, RoundIdentity, VerifiedStageEvaluation, plan_stage_target,
)


class EvidenceError(ValueError):
    """Safe fixed error codes only; do not interpolate provider/private text."""


def require(value, code):
    if not value:
        raise EvidenceError(code)


def read(path):
    return json.loads(Path(path).read_text())


def commitment(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "blake3": blake3(path.read_bytes()).hexdigest()}


def verify_evaluation(index_path: Path, context: dict) -> dict:
    """Reopen real workspace reads/citations/score through the established bridge.

    The seven-track bridge is an external integration dependency. Diagnostic
    artifacts are retained, never promoted to a memory/context pass by this adapter.
    """
    from training.benchmark_feedback.automed_codex import verify_feedback

    index = read(index_path)
    require(index.get("schema") == "eva.rsi-evaluation-index.v1", "evaluation_index_schema")
    require(index.get("evaluation_mode") in ("diagnostic_subset", "full_single_pass"), "evaluation_scope_mode_missing")
    identity_path = Path(index["checkpoint_identity"])
    identity = read(identity_path)
    require(Path(identity["exact_final_model_path"]).resolve() == Path(context["model_path"]).resolve(),
            "evaluated_checkpoint_path_differs")
    checkpoint_id = commitment(identity_path)["blake3"]
    registry = load_and_compile_registry(Path(__file__).resolve().parents[2] / "rubrics/source/domain-stage-tables.v1.json")
    rows, refs = [], []
    for root in index["feedback_roots"]:
        feedback = verify_feedback(Path(root))
        require(feedback["valid"] is True and feedback["status"] == "scored", "feedback_not_verified")
        binding = RoundIdentity(**feedback["round_identity"])
        require(binding.checkpoint_id == checkpoint_id and binding.skill_catalog_id == context["skill_catalog_id"],
                "feedback_checkpoint_or_catalog_differs")
        # verify_feedback has already reopened the exact recorded registry.
        # New seven-track tables may live in v2; historical unbound evidence
        # retains its original v1 registry rather than silently migrating.
        bound = feedback.get("exact_rubric_registry")
        exact_registry = load_and_compile_registry(Path(bound["path"])) if bound else registry
        rubric = exact_registry.resolve(feedback["domain"], feedback["stage"])
        score_doc = feedback["score"]
        score = rubric.score(score_doc["item_scores_bps"], evaluation_id=score_doc["evaluation_id"])
        require(score.to_document() == score_doc, "feedback_numeric_score_differs")
        rows.append(VerifiedStageEvaluation(feedback["case_id"], binding, rubric, EvaluationStatus.SCORED,
                                            str(Path(root).resolve()), score))
        refs.append({"root": str(Path(root).resolve()), "judge_receipt_blake3": feedback["judge_codex_receipt_blake3"]})
    require(bool(rows), "evaluation_has_no_verified_stage_evidence")
    diagnostics = [commitment(path) for path in index.get("diagnostic_artifacts", [])]
    require(bool(diagnostics), "task_memory_context_diagnostic_artifacts_missing")
    return {"index": commitment(index_path), "checkpoint_identity": commitment(identity_path),
            "evaluation_mode": index["evaluation_mode"],
            "proposal": plan_stage_target(rows), "feedback": refs, "diagnostic_artifacts": diagnostics,
            "memory_context_pass": None, "memory_context_semantics": "retained actual artifacts; no automatic pass from flags"}


def optimizer_events(path: Path) -> dict:
    """Deduplicate exact repeated events, reject conflicting identities and bad steps."""
    events, duplicates = {}, 0
    if not path.exists():
        return {"events": [], "duplicate_events": 0, "executions": 0, "learning_updates": 0}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "optimizer_step":
            continue
        key = (row.get("rollout_id"), row.get("step_id"))
        require(all(type(value) is int and value >= 0 for value in key), "optimizer_event_identity")
        require(key[1] == 0, "requires_one_optimizer_step_per_rollout")
        if key in events:
            require(events[key] == row, "conflicting_optimizer_event_identity")
            duplicates += 1
            continue
        norm = row.get("gradient_norm")
        require(row.get("trainable_parameters") == 8_953_803_264, "full_language_parameter_scope_differs")
        require(type(norm) in (int, float) and math.isfinite(norm) and norm >= 0
                and row.get("successful_update") is True, "optimizer_skipped_or_invalid_gradient")
        require(type(row.get("sampled_changed_values")) is int and row["sampled_changed_values"] >= 0,
                "weight_change_observation_missing")
        events[key] = row
    ordered = [events[key] for key in sorted(events)]
    require([row["rollout_id"] for row in ordered] == list(range(len(ordered))), "optimizer_event_gap")
    return {"events": ordered, "duplicate_events": duplicates, "executions": len(ordered),
            "learning_updates": sum(row["gradient_norm"] > 0 and row["sampled_changed_values"] > 0 for row in ordered)}


def verify_training(training_root: Path, context: dict) -> dict:
    receipt_path = training_root / "run-receipt.json"
    receipt = read(receipt_path)
    require(receipt.get("stage") == "grpo" and receipt.get("judge_backend") == "native_astra", "training_stage_or_judge")
    argv = receipt["argv"]
    require("--use-rollout-logprobs" in argv and argv[argv.index("--num-steps-per-rollout") + 1] == "1",
            "training_on_policy_or_update_contract")
    requested = int(argv[argv.index("--num-rollout") + 1])
    require(1 <= requested <= context["remaining_updates"], "requested_update_count_differs")
    if requested != context["remaining_updates"]:
        plan = read(training_root.parent / "training-plan.json")
        require(plan.get("planned_chunk_updates") == requested
                and plan.get("remaining_updates") == context["remaining_updates"]
                and plan.get("first_update_checkpoint_probe") is True
                and context["round"] == 1 and requested == 1 and context["remaining_updates"] == 50,
                "unplanned_training_prefix")
    for flag, expected in (("--hf-checkpoint", context["architecture_model_path"]),
                           ("--load", context["checkpoint_root"]), ("--save", training_root / "checkpoints")):
        require(Path(argv[argv.index(flag) + 1]).resolve() == Path(expected).resolve(), "training_checkpoint_lineage_differs")
    data_path = Path(argv[argv.index("--prompt-data") + 1])
    require(commitment(data_path)["blake3"] == receipt["training_data_blake3"], "training_data_commitment_differs")
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    require(rows and all(row["metadata"]["stage"] == context["s_target"]
                        and row["metadata"]["judge_backend"] == "native_astra" for row in rows),
            "training_curriculum_target_or_judge_differs")
    events = optimizer_events(training_root / "optimizer-steps.jsonl")
    require(events["executions"] <= requested, "optimizer_update_overshoot")
    return {"run_receipt": commitment(receipt_path), "event_source": commitment(training_root / "optimizer-steps.jsonl"),
            **events, "launch_status": receipt["status"], "checkpoint_root": str(training_root / "checkpoints"),
            "training_root": str(training_root), "exact_optimizer_resume": False}


def verify_checkpoint_evidence(training: dict, context: dict) -> dict:
    from training.slime.checkpoint_preflight import verify_checkpoint

    root = Path(training["checkpoint_root"])
    report = verify_checkpoint(root, Path(context["architecture_model_path"]))
    iteration = report["selection"].get("iteration")
    require(type(iteration) is int and 0 <= iteration < len(training["events"]), "checkpoint_optimizer_frontier_differs")
    prefix = training["events"][:iteration + 1]
    hf = Path(training["training_root"]) / "hf" / f"iter_{iteration:07d}"
    require((hf / "config.json").is_file() and (hf / "tokenizer_config.json").is_file()
            and bool(list(hf.glob("*.safetensors"))), "checkpoint_hf_export_missing")
    return {"checkpoint_root": str(root), "model_path": str(hf), "durable_updates": len(prefix),
            "durable_learning_updates": sum(row["gradient_norm"] > 0 and row["sampled_changed_values"] > 0 for row in prefix),
            "audit": report, "exact_optimizer_resume": False,
            "resume_semantics": "model_only_warm_start; optimizer/RNG/dataset cursor reset; nondurable suffix not credited"}
