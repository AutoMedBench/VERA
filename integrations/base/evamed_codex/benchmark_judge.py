"""Project one real E2E Codex attempt into exact, existing rubric judgments.

No phase receipts or stage success are synthesized. Two blind Opus workspace
judges inspect the same committed evidence; disagreement cannot emit RL reward.
Native benchmark metrics are separate from these stage-scoped assessments.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
from uuid import uuid4

from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value
from eva_agent.rubrics import load_and_compile_registry
from training.automedbench_lite.adapter import read_document, write_once
from training.automedbench_lite.track_feedback import convert_track_turn, track_snapshot

ROOT = Path(__file__).resolve().parents[3]
OPUS = "aws/anthropic/bedrock-claude-opus-5"
REGISTRY = ROOT / "EVA-Agent/rubrics/source/domain-stage-tables.v2.json"


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _json(path):
    return json.loads(path.read_bytes())


def prepare_evidence(run: Path, track: str, stage: str, output: Path):
    """Reopen actor/MCP/byte commitments before any provider request."""
    from eva_agent.training.slime_agent_judge import prepare_workspace_rollout

    run = run.resolve(strict=True)
    output = output.resolve()
    manifest = read_document(run / "track-run-manifest.json")
    rows = [row for row in manifest["tracks"] if row["track"] == track]
    _require(len(rows) == 1, "track_membership_invalid")
    spec = rows[0]
    workspace = (run / spec["workspace_relative"]).resolve()
    _require(not output.is_relative_to(workspace), "judge_output_visible_to_actor")
    audit = run / "track-rollouts" / track
    actor = read_document(audit / "rollout.json")
    _require(actor.get("score_admissible", True) is True and
             actor.get("purpose") != "infrastructure_smoke", "diagnostic_not_score_admissible")
    from .benchmark_admission import require_scoring_admission
    disposition = require_scoring_admission(run, track, actor)
    terminal_proof = None
    partial = actor.get("actual_partial_native_turn_count", 0) == 1
    if disposition == "policy-budget-exhausted" and not partial:
        proof = read_document(audit / "policy-budget-terminal.json")
        terminal_proof = {"valid": True, "receipt_blake3": proof["receipt_blake3"],
                          "actual_terminal_status": proof["actual_terminal_status"]}
    registry = load_and_compile_registry(REGISTRY)
    rubric = registry.resolve("automedbench-" + track, stage)
    target = audit / "turns/01-e2e"
    request = read_document(target / "request.json")
    specification = importlib.util.spec_from_file_location('evamed_frozen_instruction_proof',
        ROOT / 'evamed-codex/scripts/benchmark-instruction-proof.py')
    instruction_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(instruction_module)
    instruction_proof = instruction_module.verify_instruction_projection(run, track)
    before, before_doc = track_snapshot(target / "before", "before")
    after, after_doc = track_snapshot(target / "after", "after")
    _require(before_doc["immutable_inputs_manifest_blake3"] ==
             after_doc["immutable_inputs_manifest_blake3"] == spec["input_manifest_file_blake3"],
             "immutable_input_binding_changed")
    host_path = audit / "mcp-events.jsonl"
    hosts = [json.loads(line) for line in host_path.read_bytes().splitlines() if line] if host_path.exists() else []
    if partial:
        from .benchmark_partial_judge import reopen_and_project
        projection = reopen_and_project(run, track, actor, request, spec["input_manifest_file_blake3"])
        messages, results, frontiers = (projection[key] for key in ("messages", "results", "frontiers"))
        source_receipt = projection["source_receipt_blake3"]
        model, provider = projection["model"], projection["provider"]
        assistant_output, peak = projection["assistant_output"], projection["max_parallelism_observed"]
        host_count = projection["host_count"]
        native_metadata = {"verified_partial_native_turn": projection["metadata"]}
    else:
        raw = _json(target / "receipt.json")
        receipt, messages, results, used, frontiers = convert_track_turn(
            request, raw, hosts, audit, spec["input_manifest_file_blake3"], terminal_proof=terminal_proof)
        _require((receipt.status == "completed" or terminal_proof is not None) and len(used) == len(hosts),
                 "unjoined_or_nonterminal_actor")
        _require(actor["turn_receipt_blake3s"] == [receipt.receipt_blake3] and
                 actor["requested_model"] == receipt.model, "actor_identity_changed")
        source_receipt, model, provider = receipt.receipt_blake3, receipt.model, receipt.provider
        assistant_output, peak, host_count = receipt.final_response or "", receipt.max_parallelism_observed, len(used)
        native_metadata = {"verified_codex_receipt": {"receipt_blake3": source_receipt,
            "model": model, "provider": provider,
            "source_relative_path": f"track-rollouts/{track}/turns/01-e2e/receipt.json"}}
    trace = {"results": results, "declared_call_ids": [row["call_id"] for row in results],
             "joined_call_ids": [row["call_id"] for row in results], "frontier_count": frontiers,
             "max_parallelism_observed": peak, "retry_count": 0}
    sample = str(uuid4())
    binding = {"source_file_blake3": blake3_bytes(REGISTRY.read_bytes()),
               "compiled_registry_digest": registry.digest, "rubric_digest": rubric.digest}
    rollout = {"schema": "eva.codex-e2e-stage-assessment-source.v1", "task_id": sample,
        "sandbox_id": workspace.name, "candidate_id": sample, "executable_episode_id": track,
        "route_id": actor["route"],
        "model_id": model, "provider": provider,
        "messages": canonical_value(messages), "assistant_output": assistant_output,
        "provider_receipt_blake3": source_receipt,
        "workspace_before": canonical_value(before), "workspace_after": canonical_value(after),
        "tool_trace": {**trace, "trace_blake3": blake3_hex(trace)},
        "rubric_table": canonical_value(rubric.to_document()),
        "provider_metadata": {"track": track, "harness": actor["harness"], "runner": actor["runner"],
            "evaluated_stage": stage, "observation_boundary": "whole_actual_E2E_attempt",
            "terminal_disposition": disposition,
            "stage_completion_claimed": False, "stage_prefix_receipts_synthesized": False,
            "native_scores_included": False, "private_references_included": False,
            "hidden_reasoning_included": False, "exact_rubric_registry": binding,
            "frozen_instruction_projection": instruction_proof,
            "actor_document_blake3": actor["document_blake3"],
            **native_metadata,
            "input_manifest_file_blake3": spec["input_manifest_file_blake3"],
            "full_immutable_input_bytes_included": False,
            "before_manifest_blake3": before_doc["document_blake3"],
            "after_manifest_blake3": after_doc["document_blake3"]}}
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    path = output / "source.json"
    path.write_text(json.dumps(rollout, separators=(",", ":")))
    path.chmod(0o600)
    prepared = prepare_workspace_rollout(_json(path), rubric=rubric, source_path=path,
                                         judge_model_id=OPUS, judge_route_id="opus_5")
    write_once(output / "preflight.json", {"schema": "eva.codex-e2e-judge-preflight.v1",
        "sample_id": sample, "track": track, "stage": stage, "source_blake3": prepared.trajectory.source_blake3,
        "registry": binding, "actual_host_calls": host_count, "providers_called": 0,
        "observation_boundary": "whole_actual_E2E_attempt", "stage_completion_claimed": False})
    return rubric


def grade_once(source: Path, rubric, output: Path, route, codex: Path):
    from eva_agent.training.slime_agent_judge import (
        _open_opus_judge, grade_prepared_rollout, prepare_workspace_rollout)

    _require(route.model_id == OPUS and route.route_id == "opus_5", "exact_opus_route_required")
    version = subprocess.check_output([str(codex), "--version"], text=True, timeout=30).strip()
    _require(version == "codex-cli 0.153.4", "judge_codex_pin_differs")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    prepared = prepare_workspace_rollout(_json(source), rubric=rubric, source_path=source,
                                         judge_model_id=OPUS, judge_route_id="opus_5")
    write_once(output / "attempt.json", {"schema": "eva.opus-workspace-assessment-attempt.v1",
        "assessment_id": prepared.task.judge_task_id, "model": OPUS, "codex_version": version,
        "codex_binary_blake3": blake3_bytes(codex.read_bytes()),
        "source_blake3": prepared.trajectory.source_blake3, "rubric_digest": rubric.digest,
        "blind_independent_assessment": True, "retry_count": 0, "automatic_fallback": False})
    judge = None
    try:
        with _open_opus_judge(route, prepared, output, codex_bin=codex) as judge:
            grade = grade_prepared_rollout(prepared, judge=judge, output_root=output,
                                           require_original_source_citations=True)
            receipt = judge.receipt_for(prepared.task.judge_task_id)
            write_once(output / "codex-receipt.json", canonical_value(receipt))
        write_once(output / "grade.json", grade)
        return verify_assessment(source, rubric, output)
    except Exception as exc:
        if judge is not None:
            try:
                receipt = judge.receipt_for(prepared.task.judge_task_id)
                write_once(output / "failure-codex-receipt.json", canonical_value(receipt))
            except Exception:
                pass
        write_once(output / "failure.json", {"error_type": type(exc).__name__, "reward_emitted": False,
            "retry_count": 0, "automatic_fallback": False})
        raise


def verify_assessment(source: Path, rubric, output: Path):
    """Recompute score and reopen reads and actual Codex tool-result joins."""
    from eva_agent.codex_runtime import codex_turn_receipt_from_document, verify_codex_turn_receipt
    from eva_agent.codex_pipeline.adapter import (_agent_judge_terminal_object, _judge_offers,
                                                 _mcp_structured_output, _validate_tool_inventory)
    from eva_agent.pipeline.codec import _judgment
    from eva_agent.pipeline.judge_workspace import JudgeWorkspaceTools
    from eva_agent.training.agent_judge import _validated_judgment
    from eva_agent.training.agent_judge_worker import judgment_document
    from eva_agent.training.slime_agent_judge import prepare_workspace_rollout

    attempt = read_document(output / "attempt.json")
    prepared = prepare_workspace_rollout(_json(source), rubric=rubric, source_path=source,
                                         judge_model_id=OPUS, judge_route_id="opus_5")
    raw = _json(output / "assessment.json")
    _require(raw["assessment_blake3"] == blake3_hex({k: v for k, v in raw.items() if k != "assessment_blake3"}),
             "assessment_commitment_changed")
    assessment = _judgment(raw)
    prepared = replace(prepared, task=replace(prepared.task, judge_task_id=assessment.judgment_id))
    _require(attempt["assessment_id"] == assessment.judgment_id and attempt["source_blake3"] ==
             prepared.trajectory.source_blake3 and attempt["rubric_digest"] == rubric.digest,
             "assessment_attempt_binding_changed")
    snapshots = {"before": prepared.evidence.workspace_before, "after": prepared.evidence.workspace_after}
    for row in assessment.agent_trace.results:
        if row.name != "workspace_read" or row.status != "completed":
            continue
        args, result = row.arguments, row.output
        snapshot = snapshots[args["snapshot"]]
        file = next(file for file in snapshot.files if file.path == args["path"])
        data = result["content"].encode() if result["encoding"] == "utf-8" else base64.b64decode(result["content"], validate=True)
        _require(data == file.content[args["offset"]:args["offset"] + args["max_bytes"]] and
                 result["content_blake3"] == file.content_blake3 and result["tree_blake3"] == snapshot.tree_blake3,
                 "judge_read_bytes_changed")
    judgment = judgment_document(prepared, assessment)
    _require(canonical_value(judgment) == _json(output / "judgment.json"), "judgment_changed")
    scores, _ = _validated_judgment(judgment, task=prepared.task, trajectory=prepared.trajectory)
    score = rubric.score(scores, evaluation_id=assessment.judgment_id)
    _require(canonical_value(score.to_document()) == _json(output / "score.json"), "reward_recompute_differs")
    grade = read_document(output / "grade.json", maximum=64 * 1024**2)
    _require(grade["reward_bps"] == score.reward_bps and grade["item_scores_bps"] == scores
             and grade["source_rollout_blake3"] == prepared.trajectory.source_blake3
             and grade["judge_model_id"] == OPUS, "grade_binding_changed")
    receipt = codex_turn_receipt_from_document({k: v for k, v in read_document(output / "codex-receipt.json").items()
                                               if k != "document_blake3"})
    verify_codex_turn_receipt(receipt)
    _require(receipt.status == "completed" and receipt.visibility == "judge-only" and receipt.model == OPUS,
             "judge_runtime_identity_differs")
    _validate_tool_inventory(receipt, _judge_offers(JudgeWorkspaceTools(prepared.evidence), "evamed-judge"),
                             verify_event_peak=True)
    results = {row.call_id: canonical_value(row) for row in assessment.agent_trace.results}
    seen = set()
    for call in receipt.tool_calls:
        if call.fully_qualified_name not in receipt.offered_mcp_tool_names:
            continue
        value = _mcp_structured_output(call)
        core = {k: v for k, v in value.items() if k != "bridge_receipt_blake3"}
        _require(value["bridge_receipt_blake3"] == blake3_hex(core) and value["call_id"] not in seen
                 and value["judge_tool_result"] == results.get(value["call_id"])
                 and value["arguments"] == canonical_value(call.arguments) and value["name"] == call.mcp_tool,
                 "judge_result_transport_binding_changed")
        seen.add(value["call_id"])
    _require(seen == set(results), "judge_tool_coverage_differs")
    _require(_agent_judge_terminal_object(receipt.final_response or "") == {
        "item_scores": canonical_value(assessment.item_scores), "hard_gates_passed": assessment.hard_gates_passed,
        "summary": assessment.summary}, "judge_terminal_differs")
    return {"verified": True, "assessment_id": assessment.judgment_id,
        "source_blake3": prepared.trajectory.source_blake3, "rubric_digest": rubric.digest,
        "receipt_blake3": receipt.receipt_blake3, "item_scores_bps": scores,
        "reward_bps": score.reward_bps, "hard_gate_passed": score.hard_gate_passed}


def judge_and_verify(output: Path, route, codex: Path):
    preflight = read_document(output / "preflight.json")
    registry = load_and_compile_registry(REGISTRY)
    _require(preflight["registry"]["source_file_blake3"] == blake3_bytes(REGISTRY.read_bytes()) and
             preflight["registry"]["compiled_registry_digest"] == registry.digest, "registry_changed")
    rubric = registry.resolve("automedbench-" + preflight["track"], preflight["stage"])
    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {role: pool.submit(grade_once, output / "source.json", rubric, output / role, route, codex)
                   for role in ("judge", "reward-verifier")}
        for role, future in futures.items():
            try:
                results[role] = future.result()
            except Exception as exc:
                results[role] = {"verified": False, "error_type": type(exc).__name__}
    valid = all(row["verified"] for row in results.values())
    agree = valid and results["judge"]["item_scores_bps"] == results["reward-verifier"]["item_scores_bps"]
    if valid:
        _require(results["judge"]["assessment_id"] != results["reward-verifier"]["assessment_id"], "judge_not_independent")
    return write_once(output / "pair.json", {"schema": "eva.opus-independent-reward-verification.v1",
        "pair_id": str(uuid4()), "track": preflight["track"], "stage": preflight["stage"],
        "source_blake3": preflight["source_blake3"], "rubric_digest": rubric.digest, "results": results,
        "both_assessments_verified": valid, "exact_item_agreement": agree,
        "reward_admitted": agree, "reward_bps": results["judge"]["reward_bps"] if agree else None,
        "disagreement_policy": "retain_both_and_withhold_training_reward",
        "native_benchmark_score": False, "attribution_reward_weight": 0})
