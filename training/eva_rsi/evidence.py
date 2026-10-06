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
    if index.get("schema") == "eva.rsi-composite-evaluation-index.v1":
        return _verify_composite_evaluation(index_path, context)
    require(index.get("skill_selection") == context.get("skill_selection"),
            "evaluation_skill_selection_context_differs")
    if index.get("skill_selection"):
        from .skill_selection import read_selection
        read_selection(index["skill_selection"], catalog_id=context["skill_catalog_id"])
        actor_attempt = read(Path(index["benchmark_run_root"]) / "track-rollouts/attempt.json")
        require(actor_attempt.get("skill_selection") == index["skill_selection"],
                "evaluation_actor_skill_selection_differs")
    require(index.get("schema") in ("eva.rsi-evaluation-index.v1", "eva.rsi-evaluation-index.v2"),
            "evaluation_index_schema")
    content_binding = None
    judge_family = None
    if "judge_comparison_family" in index:
        from .judge_identity import verify_judge_comparison_family
        require(index["schema"] == "eva.rsi-evaluation-index.v2"
            and index.get("judge_identity_mode") == "declared-annotation-family-v1",
            "judge_family_opt_in_required")
        judge_family = verify_judge_comparison_family(index["judge_comparison_family"])
    else:
        require("judge_identity_mode" not in index, "judge_family_sidecar_required")
    if index["schema"] == "eva.rsi-evaluation-index.v2":
        from .skill_identity import verify_skill_content_binding
        require(index.get("skill_catalog_identity_mode") == "verified-content-v1", "skill_identity_opt_in_required")
        sidecar = index["skill_content_identity"]
        require(commitment(sidecar["path"]) == sidecar, "skill_identity_sidecar_commitment")
        content_binding = verify_skill_content_binding(Path(sidecar["path"]),
            expected_run_root=Path(index["benchmark_run_root"]), expected_content_id=context["skill_catalog_id"])
    require(index.get("evaluation_mode") in ("diagnostic_subset", "full_single_pass"), "evaluation_scope_mode_missing")
    identity_path = Path(index["checkpoint_identity"])
    identity = read(identity_path)
    require(Path(identity["exact_final_model_path"]).resolve() == Path(context["model_path"]).resolve(),
            "evaluated_checkpoint_path_differs")
    checkpoint_id = commitment(identity_path)["blake3"]
    registry = load_and_compile_registry(Path(__file__).resolve().parents[2] / "rubrics/source/domain-stage-tables.v1.json")
    from training.automedbench_lite.judge_attempt_isolation import (
        BINDING_NAME, replacement_limit, verify_stage_judge_attempts,
    )
    isolation_refs = index.get("postround_judge_attempt_isolation")
    isolation_results = None
    if isolation_refs is not None:
        replacement_limit(index.get("postround_judge_verdict_replacements"))
        require(isinstance(isolation_refs, list) and len(isolation_refs) == len(index["feedback_roots"]),
                "postround_judge_attempt_isolation_count")
        require(all(commitment(ref["path"]) == ref for ref in isolation_refs),
                "postround_judge_attempt_isolation_commitment")
        isolation_results = [verify_stage_judge_attempts(Path(ref["path"]), verify=verify_feedback)
                             for ref in isolation_refs]
        require([result["accepted_attempt_root"] for result in isolation_results]
                == [str(Path(root).resolve()) for root in index["feedback_roots"]],
                "postround_judge_attempt_isolation_feedback_roots")
    else:
        require("postround_judge_verdict_replacements" not in index,
                "postround_judge_attempt_isolation_sidecar_required")
        require(not any((Path(root) / BINDING_NAME).exists() for root in index["feedback_roots"]),
                "unbound_postround_judge_attempt_isolation")
    rows, refs, normalizations, judge_normalizations = [], [], [], []
    for root in index["feedback_roots"]:
        if judge_family:
            from .judge_identity import reopen_family_feedback, normalize_judge_identity
            feedback = reopen_family_feedback(root, judge_family, include_skill_source_binding=bool(content_binding))
        else:
            feedback = (verify_feedback(Path(root), include_skill_source_binding=True) if content_binding
                        else verify_feedback(Path(root)))
        require(feedback["valid"] is True and feedback["status"] == "scored", "feedback_not_verified")
        binding = RoundIdentity(**feedback["round_identity"])
        if content_binding:
            require(binding.skill_catalog_id == content_binding["mounted_catalog_blake3"]
                    and feedback["skill_source_binding"] == {
                        "run_attempt_document_blake3": content_binding["attempt"]["document_blake3"],
                        "mounted_skill_catalog_blake3": content_binding["mounted_catalog_blake3"]},
                    "feedback_skill_source_attempt_differs")
            # Normalize only the planner's in-memory comparison. The original
            # verified RoundIdentity and feedback files remain unchanged.
            normalized = {**feedback["round_identity"], "skill_catalog_id": content_binding["content_identity_blake3"]}
            normalizations.append({"root": str(Path(root).resolve()),
                "raw_round_identity": feedback["round_identity"], "comparison_round_identity": normalized})
            binding = RoundIdentity(**normalized)
        if judge_family:
            comparison = normalize_judge_identity(feedback, judge_family)
            normalized = {"model_id": binding.model_id, "checkpoint_id": binding.checkpoint_id,
                "skill_catalog_id": binding.skill_catalog_id,
                "judge_id": comparison["comparison_round_identity"]["judge_id"]}
            judge_normalizations.append({"root":str(Path(root).resolve()),
                "raw_round_identity":feedback["round_identity"], "comparison_round_identity":normalized})
            binding = RoundIdentity(**normalized)
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
            "memory_context_pass": None, "memory_context_semantics": "retained actual artifacts; no automatic pass from flags",
            **({"postround_judge_attempt_isolation": isolation_results}
               if isolation_results is not None else {}),
            **({"skill_selection": index["skill_selection"]} if index.get("skill_selection") else {}),
            **({"judge_identity_normalization": {**judge_family, "feedback": judge_normalizations}}
               if judge_family else {}),
            **({"skill_identity_normalization": {"schema": "eva.rsi-skill-identity-normalization.v1",
                "sidecar": index["skill_content_identity"],
                "content_identity_blake3": content_binding["content_identity_blake3"],
                "mounted_catalog_blake3": content_binding["mounted_catalog_blake3"],
                "original_evidence_mutated": False, "feedback": normalizations}} if content_binding else {})}


def _verify_composite_evaluation(index_path: Path, context: dict) -> dict:
    """Compare retained, source-verified grades without manufacturing a rollout.

    A fresh serving receipt has its own identity. Only the planner's in-memory
    comparison uses the independently verified same-checkpoint/content aliases.
    Every raw actor/grade document and per-source receipt stays unchanged.
    """
    from .composite_eval import verify_composite_index
    from training.benchmark_feedback.automed_codex import verify_feedback

    index = read(index_path)
    proof = verify_composite_index(index_path)
    require(index.get("skill_selection") == context.get("skill_selection"),
            "evaluation_skill_selection_context_differs")
    require(proof["skill_content_id"] == context["skill_catalog_id"],
            "composite_context_skill_content_differs")
    identity_ref = proof["comparison_checkpoint_identity"]
    require(Path(read(identity_ref["path"])["exact_final_model_path"]).resolve()
            == Path(context["model_path"]).resolve(), "evaluated_checkpoint_path_differs")
    if context.get("skill_selection"):
        from .skill_selection import read_selection
        read_selection(context["skill_selection"], catalog_id=context["skill_catalog_id"])
    rows, refs, normalizations = [], [], []
    registry = load_and_compile_registry(Path(__file__).resolve().parents[2]
                                        / "rubrics/source/domain-stage-tables.v1.json")
    for entry in proof["feedback"]:
        root = Path(entry["root"])
        feedback = verify_feedback(root, include_skill_source_binding=True)
        require(feedback["valid"] is True and feedback["status"] == "scored"
                and feedback["judge_codex_receipt_blake3"] == entry["judge_receipt_blake3"]
                and feedback["round_identity"] == entry["raw_round_identity"],
                "composite_feedback_reopen_differs")
        comparison = {**feedback["round_identity"], "checkpoint_id": identity_ref["blake3"],
                      "skill_catalog_id": proof["skill_content_id"]}
        exact = feedback.get("exact_rubric_registry")
        table = load_and_compile_registry(Path(exact["path"])) if exact else registry
        rubric = table.resolve(feedback["domain"], feedback["stage"])
        score_doc = feedback["score"]
        score = rubric.score(score_doc["item_scores_bps"], evaluation_id=score_doc["evaluation_id"])
        require(score.to_document() == score_doc, "feedback_numeric_score_differs")
        rows.append(VerifiedStageEvaluation(feedback["case_id"], RoundIdentity(**comparison),
                    rubric, EvaluationStatus.SCORED, str(root.resolve()), score))
        refs.append({"root": str(root.resolve()),
                     "judge_receipt_blake3": feedback["judge_codex_receipt_blake3"]})
        normalizations.append({"root": str(root.resolve()), "source_name": entry["source_name"],
            "raw_round_identity": feedback["round_identity"], "comparison_round_identity": comparison})
    require(bool(rows), "evaluation_has_no_verified_stage_evidence")
    diagnostics = []
    for source in proof["sources"].values():
        source_index = read(source["index"]["path"])
        source_diagnostics = [commitment(path) for path in source_index.get("diagnostic_artifacts", [])]
        require(bool(source_diagnostics), "task_memory_context_diagnostic_artifacts_missing")
        diagnostics.extend(source_diagnostics)
    return {"index": commitment(index_path), "checkpoint_identity": identity_ref,
        "evaluation_mode": "composite_full_track_coverage", "proposal": plan_stage_target(rows),
        "feedback": refs, "diagnostic_artifacts": diagnostics, "memory_context_pass": None,
        "memory_context_semantics": "retained per-source artifacts; no automatic pass from flags",
        "composite_provenance": proof,
        "comparison_identity_normalization": {"original_evidence_mutated": False,
            "same_checkpoint_equivalence": proof["checkpoint_equivalence_blake3"],
            "feedback": normalizations},
        **({"skill_selection": context["skill_selection"]} if context.get("skill_selection") else {})}


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
    isolated_groups = {}
    if "judge_group_isolation" in receipt:
        from training.slime.judge_group_isolation import (
            GENERATION_FUNCTION, POLICY, replacement_limit, resolve_accepted_group,
        )
        policy = receipt["judge_group_isolation"]
        require(policy.get("policy") == POLICY and replacement_limit(policy.get("replacement_limit")) > 0
                and argv[argv.index("--rollout-function-path") + 1] == GENERATION_FUNCTION,
                "judge_group_isolation_policy_differs")
        isolated_groups["accepted_rollout_groups"] = [
            resolve_accepted_group(training_root, event["rollout_id"]) for event in events["events"]
        ]
    return {"run_receipt": commitment(receipt_path), "event_source": commitment(training_root / "optimizer-steps.jsonl"),
            **events, "launch_status": receipt["status"], "checkpoint_root": str(training_root / "checkpoints"),
            "training_root": str(training_root), "exact_optimizer_resume": False, **isolated_groups}


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
            **({"accepted_rollout_groups": training["accepted_rollout_groups"][:len(prefix)]}
               if "accepted_rollout_groups" in training else {}),
            "audit": report, "exact_optimizer_resume": False,
            "resume_semantics": "model_only_warm_start; optimizer/RNG/dataset cursor reset; nondurable suffix not credited"}
