"""One-shot Opus evidence review; suggestions never edit a skill or a reward.

The exact existing rubric and workspace tools remain authoritative. A versioned
suggestion document occupies the existing Judge ``summary`` string, so neither
canonical tool schemas nor the canonical judgment schema need modification.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value
from .evidence import commitment, read, require

SCHEMA = "eva.rsi-opus-skill-attribution.v1"
SUMMARY_SCHEMA = "eva.rsi-skill-suggestions.v2"


def retain_decision(catalog_id, status, **extra):
    return {"action": "retain_catalog", "catalog_id": catalog_id,
            "opus_attribution": status, "skill_change_performed": False,
            "attribution_claimed": status == "verified", "causal_effect_established": False,
            "suggestions": [], **extra}


def instruction(catalog):
    return {
        "schema": SUMMARY_SCHEMA,
        "instruction": (
            "This is an observational skill-attribution review, not a causal experiment. "
            "Score the exact unchanged rubric using actual inspected actor context, tool results, "
            "and before/after workspace. hard_gates_passed is the conjunction of ONLY the compiled "
            "items' non-null hard_gate minimum_score_bps conditions against rounded item score*10000. "
            "If no items define hard_gate, hard_gates_passed MUST be true even when all item scores "
            "are zero. This flag is not overall task success; do not invent additional hard gates. "
            "In the existing summary STRING put one JSON object "
            "with exactly schema, suggestions. suggestions must contain 1..8 objects with exactly "
            "action (retain/add/remove), skill_id (existing exact skill ID, or null for a proposed "
            "new capability), capability (short public description), evidence_refs (nonempty "
            "list of workspace references you actually read), and rationale (brief observable "
            "evidence, not hidden reasoning). A retain recommendation is valid: missing use or "
            "low task score alone does not prove a skill is harmful. Removal or retention of an "
            "individual skill requires inspecting its actual load observation/body, not merely "
            "the catalog. A null skill_id with retain means retain the whole unchanged catalog. "
            "For stage-scoped discovery preferences, add/remove may name an existing eligible skill "
            "whose actual loaded body you inspected: add prioritizes it, remove deprioritizes it. "
            "All canonical definitions and on-demand access remain unchanged. A null-ID add is "
            "an unsupported new-capability proposal and retains selection; do not invent a skill ID, body, "
            "tool, permission, or completed stage. Cite at least one original source workspace "
            "file per suggestion; audit context/tool sidecars can be additional evidence. "
            "Only verified existing-ID discovery preferences may be applied to future attempts. "
            "There is no automatic skill authoring or permission change. The schema value is " + SUMMARY_SCHEMA
        ),
        "verified_catalog_metadata": catalog,
        "scope": "one deterministically selected weakest verified post-round stage",
        "catalog_metadata_is_not_skill_use_evidence": True,
    }


class AttributionJudge:
    """Add judge-only instructions; do not change actor evidence or rubric."""
    def __init__(self, judge, catalog):
        self.judge_impl, self.catalog = judge, catalog

    def judge(self, request, rubric):
        private = {**request.judge_only_reference, "skill_attribution": instruction(self.catalog)}
        return self.judge_impl.judge(replace(request, judge_only_reference=private), rubric)


def validate_suggestions(assessment, prepared, catalog):
    value = json.loads(assessment.summary)
    require(isinstance(value, dict) and set(value) == {"schema", "suggestions"}
            and value["schema"] == SUMMARY_SCHEMA, "attribution_summary_schema")
    rows = value["suggestions"]
    require(isinstance(rows, list) and 1 <= len(rows) <= 8, "attribution_suggestion_count")
    known = {row["skill_id"] for row in catalog}
    inspected = set(assessment.agent_trace.inspected_evidence_refs)
    sources = set(prepared.source_workspace_refs)
    # An individual skill recommendation requires a real successful load,
    # retained body, and the Judge's read of the exact actor-tool sidecar.
    from eva_agent.codex_pipeline import CODEX_JUDGE_REQUIRED_MATERIAL_PATHS
    trace_ref = "workspace:after:" + next(path for path in CODEX_JUDGE_REQUIRED_MATERIAL_PATHS
                                        if path.endswith("tool-trace.json"))
    loaded = set()
    for result in prepared.evidence.tool_trace.results:
        if result.name != "load_skill" or result.status != "completed":
            continue
        host = result.output.get("host_event", {})
        payload = host.get("result", {})
        body = payload.get("content")
        skill_id = payload.get("skill_id")
        if (host.get("is_error") is False and host.get("arguments", {}).get("skill_id") == skill_id
                and isinstance(body, str) and body and payload.get("content_blake3") == blake3_hex(body)):
            loaded.add(skill_id)
    for row in rows:
        require(isinstance(row, dict) and set(row) == {
            "action", "skill_id", "capability", "evidence_refs", "rationale"}, "attribution_suggestion_shape")
        require(row["action"] in {"retain", "add", "remove"}, "attribution_action")
        require(row["skill_id"] in known or row["skill_id"] is None, "attribution_unknown_skill")
        require(row["action"] != "remove" or row["skill_id"] in known, "attribution_remove_requires_existing_skill")
        require(row["skill_id"] is None or (row["skill_id"] in loaded and trace_ref in inspected),
                "attribution_individual_skill_not_inspected")
        require(all(isinstance(row[key], str) and 0 < len(row[key]) <= 2000
                    for key in ("capability", "rationale")), "attribution_description")
        refs = row["evidence_refs"]
        require(isinstance(refs, list) and refs and all(isinstance(ref, str) for ref in refs)
                and set(refs) <= inspected and bool(set(refs) & sources), "attribution_uninspected_evidence")
    return rows


def _prepare_composite_source(context, evaluation_path, evaluation, index_path):
    """Select across the verified union; Judge only the chosen original evidence."""
    from training.benchmark_feedback.automed_codex import (
        _bound_feedback_rubric, read_feedback_rollout, verify_feedback)
    from .composite_eval import verify_composite_index
    from .skill_identity import verify_skill_content_binding

    proof = verify_composite_index(index_path)
    require(proof.get("valid") is True and proof["skill_content_id"] == context["skill_catalog_id"]
            and proof["comparison_checkpoint_identity"] == evaluation["checkpoint_identity"],
            "attribution_composite_comparison_identity")
    verified_rows = proof["feedback"]
    expected = {row["root"]: row["judge_receipt_blake3"] for row in verified_rows}
    supplied = {row["root"]: row["judge_receipt_blake3"] for row in evaluation["feedback"]}
    require(len(expected) == len(verified_rows) and len(supplied) == len(evaluation["feedback"])
            and supplied == expected, "attribution_composite_feedback_coverage")
    sources = {}
    for name, source in proof["sources"].items():
        identity = source["checkpoint_identity"]
        require(commitment(identity["path"]) == identity
                and Path(read(identity["path"])["exact_final_model_path"]).resolve()
                == Path(context["model_path"]).resolve(), "attribution_checkpoint_differs")
        sidecar = source["skill_content_binding"]
        require(commitment(sidecar["path"]) == sidecar, "attribution_catalog_sidecar_changed")
        skills = verify_skill_content_binding(Path(sidecar["path"]),
            expected_run_root=Path(source["benchmark_run_root"]), expected_content_id=context["skill_catalog_id"])
        sources[name] = (source, skills)
    choices = []
    for row in verified_rows:
        source, skills = sources[row["source_name"]]
        root = Path(row["root"])
        feedback = verify_feedback(root, include_skill_source_binding=True)
        require(feedback.get("valid") is True and feedback.get("status") == "scored"
                and feedback["judge_codex_receipt_blake3"] == row["judge_receipt_blake3"]
                and feedback["round_identity"] == row["raw_round_identity"]
                and feedback["score"] == row["score_document"], "attribution_feedback_changed")
        require(feedback["round_identity"]["checkpoint_id"] == source["checkpoint_identity"]["blake3"]
                and feedback["round_identity"]["skill_catalog_id"] == skills["mounted_catalog_blake3"],
                "attribution_feedback_identity")
        choices.append((feedback["score"]["reward_bps"], feedback["stage"], feedback["domain"],
                        str(root), feedback, row["source_name"]))
    require(bool(choices), "attribution_no_verified_evidence")
    _, _, _, chosen, feedback, source_name = min(choices, key=lambda row: row[:4])
    source, skills = sources[source_name]
    root = Path(chosen)
    preflight = read(root / "preflight.json")
    rollout = read_feedback_rollout(root / "rollout.json", preflight)
    rubric = _bound_feedback_rubric(preflight, rollout, None)
    require(blake3_hex(rollout) == feedback["source_rollout_blake3"], "attribution_rollout_changed")
    content = skills["verified_content"]["content"]
    catalog = [*content["legacy_skill_metadata"], content["mount_path_independent_catalog"]["native_skill"]]
    selection = {"evaluation": commitment(evaluation_path), "index": evaluation["index"],
        "feedback_root": chosen, "source_rollout": commitment(root / "rollout.json"),
        "rubric_digest": rubric.digest, "catalog_id": context["skill_catalog_id"],
        "catalog_metadata": catalog, "selection": "lowest_reward_then_stage_domain_path",
        "verified_candidate_count": len(choices), "selected_stage": feedback["stage"],
        "selected_domain": feedback["domain"], "all_round_trajectories_inspected_by_opus": False,
        "composite_source": {"source_name": source_name, "source_index": source["index"],
            "benchmark_run_root": source["benchmark_run_root"],
            "raw_checkpoint_identity": source["checkpoint_identity"],
            "raw_round_identity": feedback["round_identity"],
            "skill_content_binding": source["skill_content_binding"],
            "mounted_catalog_blake3": skills["mounted_catalog_blake3"],
            "comparison_content_identity": proof["skill_content_id"],
            "checkpoint_equivalence_blake3": proof["checkpoint_equivalence_blake3"],
            "canonical_source_evidence_normalized": False}}
    return rollout, rubric, selection


def prepare_source(context):
    """Reopen verified feedback, then select one bounded evidence case, not scores alone."""
    from training.benchmark_feedback.automed_codex import (
        _bound_feedback_rubric, read_feedback_rollout, verify_feedback)
    from .skill_identity import verify_skill_content_binding

    evaluation_path = Path(context["previous_evaluation"])
    evaluation = read(evaluation_path)
    index_path = Path(evaluation["index"]["path"])
    require(commitment(index_path) == evaluation["index"], "attribution_evaluation_changed")
    index = read(index_path)
    if index.get("schema") == "eva.rsi-composite-evaluation-index.v1":
        return _prepare_composite_source(context, evaluation_path, evaluation, index_path)
    require(index["schema"] == "eva.rsi-evaluation-index.v2", "attribution_requires_verified_content_identity")
    sidecar = index["skill_content_identity"]
    require(commitment(sidecar["path"]) == sidecar, "attribution_catalog_sidecar_changed")
    skills = verify_skill_content_binding(Path(sidecar["path"]), expected_run_root=Path(index["benchmark_run_root"]),
                                         expected_content_id=context["skill_catalog_id"])
    require(commitment(index["checkpoint_identity"]) == evaluation["checkpoint_identity"],
            "attribution_checkpoint_identity_changed")
    require(Path(read(index["checkpoint_identity"])["exact_final_model_path"]).resolve()
            == Path(context["model_path"]).resolve(), "attribution_checkpoint_differs")
    choices = []
    family = None
    if "judge_comparison_family" in index:
        from .judge_identity import verify_judge_comparison_family
        require(index.get("judge_identity_mode") == "declared-annotation-family-v1",
                "attribution_judge_family_opt_in_required")
        family = verify_judge_comparison_family(index["judge_comparison_family"])
    for row in evaluation["feedback"]:
        root = Path(row["root"])
        if family:
            from .judge_identity import reopen_family_feedback
            feedback = reopen_family_feedback(root, family, include_skill_source_binding=True)
        else:
            feedback = verify_feedback(root, include_skill_source_binding=True)
        require(feedback["valid"] is True and feedback["status"] == "scored"
                and feedback["judge_codex_receipt_blake3"] == row["judge_receipt_blake3"],
                "attribution_feedback_changed")
        require(feedback["round_identity"]["checkpoint_id"] == evaluation["checkpoint_identity"]["blake3"]
                and feedback["round_identity"]["skill_catalog_id"] == skills["mounted_catalog_blake3"],
                "attribution_feedback_identity")
        choices.append((feedback["score"]["reward_bps"], feedback["stage"], feedback["domain"], str(root), feedback))
    require(bool(choices), "attribution_no_verified_evidence")
    _, _, _, chosen, feedback = min(choices, key=lambda row: row[:4])
    root = Path(chosen)
    preflight = read(root / "preflight.json")
    rollout = read_feedback_rollout(root / "rollout.json", preflight)
    rubric = _bound_feedback_rubric(preflight, rollout, None)
    require(blake3_hex(rollout) == feedback["source_rollout_blake3"], "attribution_rollout_changed")
    # Catalog metadata is explicit context, not inferred tool use or skill quality.
    content = skills["verified_content"]["content"]
    native = content["mount_path_independent_catalog"]["native_skill"]
    catalog = [*content["legacy_skill_metadata"], native]
    selection = {"evaluation": commitment(evaluation_path), "index": evaluation["index"],
        "feedback_root": chosen, "source_rollout": commitment(root / "rollout.json"),
        "rubric_digest": rubric.digest, "catalog_id": context["skill_catalog_id"],
        "catalog_metadata": catalog, "selection": "lowest_reward_then_stage_domain_path",
        "verified_candidate_count": len(choices), "selected_stage": feedback["stage"],
        "selected_domain": feedback["domain"], "all_round_trajectories_inspected_by_opus": False}
    return rollout, rubric, selection


@contextmanager
def open_opus(route, prepared, output_root, *, codex_bin):
    """Existing immutable Judge/MCP APIs; explicit binary and selected venv paths."""
    from eva_agent.codex_pipeline import CodexOpus5AgentJudge, TurnMCPBridgeFactory
    from eva_agent.codex_providers import AdapterLimits, ResponsesAdapterGateway
    from eva_agent.codex_runtime import (CodexRole, CodexRuntime, CodexSandbox, CodexThreadOptions,
        OpenAICodexBackend, PersistentCodexRuntimeRunner)
    from eva_agent.deployment.campaign import CODEX_FIRST_RELEASE_CONFIG_OVERRIDES, CODEX_FIRST_RELEASE_THREAD_CONFIG
    from eva_agent.deployment.rollout_adapters import materialize_rollout_model_catalog
    from eva_agent.pipeline import RandomUUIDFactory
    from eva_agent.training.agent_judge import persist_agent_judge_adapter_receipt
    from eva_agent.training.teacher_launch import isolated_teacher_launch_options
    from eva_agent import codex_pipeline

    source_root = Path(codex_pipeline.__file__).resolve().parents[2]
    with TemporaryDirectory(prefix="eva-opus-attribution-") as temporary:
        root = Path(temporary)
        (root / "mcp").mkdir(mode=0o700)
        bridge = TurnMCPBridgeFactory(proxy_python=Path(sys.executable).resolve(strict=True),
            proxy_script=source_root / "eva_agent/codex_pipeline/turn_mcp_proxy.py",
            temp_root=root / "mcp", maximum_parallel_calls=64)
        gateway = ResponsesAdapterGateway({"opus_5": route},
            limits=AdapterLimits(max_concurrency=1, upstream_timeout_seconds=240),
            local_credential_env_name="EVA_CODEX_OPUS5_ADAPTER_TOKEN", upstream_max_tokens_override=16384,
            receipt_sink=lambda receipt: persist_agent_judge_adapter_receipt(output_root, receipt,
                judge_task_id=prepared.task.judge_task_id, expected_route_id="opus_5"))
        with gateway as opened:
            adapted = opened.adapted_routes()["opus_5"]
            catalog = materialize_rollout_model_catalog({"opus_5": adapted}, root=root / "catalog",
                route_order=("opus_5",), auto_compact_token_limit=114688)
            _, child_env, private = adapted.config.for_subprocess()
            endpoint, _ = private
            launch = isolated_teacher_launch_options(codex_bin=Path(codex_bin).resolve(strict=True),
                cwd=source_root.parent, isolation_root=root / "isolated", child_env=child_env,
                config_overrides=(*CODEX_FIRST_RELEASE_CONFIG_OVERRIDES, catalog.config_override))
            runner = PersistentCodexRuntimeRunner(lambda: CodexRuntime(OpenAICodexBackend(launch)))

            def options(request, offers):
                return CodexThreadOptions(role=CodexRole.JUDGE, model=request.judge_model_id,
                    provider=adapted.config.provider_id, cwd=str(source_root.parent),
                    sandbox=CodexSandbox.READ_ONLY, offered_tools=offers, ephemeral=True,
                    config={"project_doc_max_bytes": 0, "web_search": "disabled",
                        "features": dict(CODEX_FIRST_RELEASE_THREAD_CONFIG["features"]),
                        "model_providers": {adapted.config.provider_id: {
                            "name": "EVA Responses Gateway", "base_url": endpoint,
                            "env_key": adapted.config.credential_env_name, "requires_openai_auth": False,
                            "wire_api": "responses", "request_max_retries": 0, "stream_max_retries": 0}}})

            judge = CodexOpus5AgentJudge(options_factory=options, model_id=route.model_id, runner=runner,
                id_factory=RandomUUIDFactory(), maximum_parallel_tools=64, turn_mcp_factory=bridge,
                compact_evidence_index=True, fast_workspace_judge=True, maximum_workspace_tool_calls=64,
                maximum_workspace_tool_frontiers=4, turn_timeout_seconds=240)
            try:
                runner.start()
                yield judge
            finally:
                runner.close()


def _reopen_assessment(root, prepared, catalog):
    """Existing typed trace/rubric checks plus byte and native receipt joins."""
    from eva_agent.pipeline.codec import _judgment
    from eva_agent.pipeline.judge_workspace import JudgeWorkspaceTools
    from eva_agent.codex_runtime import codex_turn_receipt_from_document
    from eva_agent.codex_pipeline.adapter import _agent_judge_terminal_object, _judge_offers, _mcp_structured_output, _validate_tool_inventory
    from eva_agent.training.agent_judge_worker import judgment_document
    from eva_agent.training.agent_judge import _validated_judgment

    raw = read(root / "assessment.json")
    require(raw["assessment_blake3"] == blake3_hex({k: v for k, v in raw.items() if k != "assessment_blake3"}),
            "attribution_assessment_commitment")
    assessment = _judgment(raw)
    prepared = replace(prepared, task=replace(prepared.task, judge_task_id=assessment.judgment_id))
    snapshots = {"before": prepared.evidence.workspace_before, "after": prepared.evidence.workspace_after}
    for result in assessment.agent_trace.results:
        if result.name != "workspace_read" or result.status != "completed":
            continue
        args, out = result.arguments, result.output
        snap = snapshots[args["snapshot"]]
        file = next(row for row in snap.files if row.path == args["path"])
        data = out["content"].encode() if out["encoding"] == "utf-8" else base64.b64decode(out["content"], validate=True)
        require(data == file.content[args["offset"]:args["offset"] + args["max_bytes"]]
                and out["content_blake3"] == file.content_blake3 and out["tree_blake3"] == snap.tree_blake3,
                "attribution_read_bytes_changed")
    judgment = judgment_document(prepared, assessment)
    require(canonical_value(judgment) == read(root / "judgment.json"), "attribution_judgment_changed")
    scores, _ = _validated_judgment(judgment, task=prepared.task, trajectory=prepared.trajectory)
    require(prepared.trajectory.rubric.score(scores, evaluation_id=assessment.judgment_id).to_document()
            == read(root / "score.json"), "attribution_rubric_score_changed")
    receipt = codex_turn_receipt_from_document(read(root / "judge-codex-receipt.json"))
    require(receipt.status == "completed" and receipt.visibility == "judge-only"
            and receipt.model == assessment.judge_model_id and receipt.provider == "eva_adapter_opus_5",
            "attribution_not_actual_opus_receipt")
    _validate_tool_inventory(receipt, _judge_offers(JudgeWorkspaceTools(prepared.evidence), "evamed-judge"),
                             verify_event_peak=True)
    results = {row.call_id: canonical_value(row) for row in assessment.agent_trace.results}
    observed = set()
    for call in receipt.tool_calls:
        if call.fully_qualified_name not in receipt.offered_mcp_tool_names:
            continue
        doc = _mcp_structured_output(call)
        require(doc["bridge_receipt_blake3"] == blake3_hex({k: v for k, v in doc.items() if k != "bridge_receipt_blake3"})
                and doc["call_id"] not in observed and doc["judge_tool_result"] == results.get(doc["call_id"])
                and doc["name"] == call.mcp_tool and doc["arguments"] == canonical_value(call.arguments),
                "attribution_transport_trace_changed")
        observed.add(doc["call_id"])
    require(observed == set(results), "attribution_trace_coverage")
    require(_agent_judge_terminal_object(receipt.final_response or "") == {
        "item_scores": canonical_value(assessment.item_scores), "hard_gates_passed": assessment.hard_gates_passed,
        "summary": assessment.summary}, "attribution_summary_not_provider_output")
    return validate_suggestions(assessment, prepared, catalog)


def adapter_statuses(root):
    """Only signature-checked safe HTTP categories; no raw provider response."""
    from eva_agent.codex_providers.adapter import SignedAdapterReceipt, verify_adapter_receipt
    rows = []
    for path in sorted((root / "adapter-receipts").rglob("*.json")):
        doc = read(path)
        receipt = SignedAdapterReceipt(**{key: doc[key] for key in SignedAdapterReceipt.__dataclass_fields__})
        verify_adapter_receipt(receipt)
        require(receipt.payload.get("route_id") in {None, "opus_5"}, "attribution_adapter_route")
        rows.append({"receipt": commitment(path),
            "adapter_http_status": receipt.payload["adapter_http_status"],
            "upstream_http_status": receipt.payload["upstream_http_status"]})
    return rows


def run_once(context, *, route_registry, env_files, codex_bin):
    """Explicit execution entry; root is consumed before any route/provider work."""
    from .controller import write
    from eva_agent.training.slime_agent_judge import prepare_workspace_rollout, grade_prepared_rollout
    from eva_agent.codex_providers import load_codex_provider_routes

    root = Path(context["attempt_root"]) / "skill-attribution"
    root.mkdir(mode=0o700, exist_ok=False)
    attempt = {"schema": SCHEMA, "attempt_id": str(uuid4()), "round": context["round"],
        "backend": "opus_5", "retry_count": 0, "automatic_fallback": False,
        "context_blake3": blake3_hex(context), "requested_model": None,
        "provider_setup_started": False, "stage": "evidence_preparation",
        "implementation": commitment(Path(__file__))}
    write(root / "attempt.json", attempt, exclusive=True)
    judge = prepared = None
    try:
        rollout, rubric, source = prepare_source(context)
        write(root / "source.json", source, exclusive=True)
        attempt["stage"] = "route_resolution"
        write(root / "attempt.json", attempt)
        routes = load_codex_provider_routes(env_files=tuple(Path(p) for p in env_files),
            registry_path=Path(route_registry), route_ids=("opus_5",))
        route = routes.get("opus_5")
        require(route is not None and "opus-5" in route.model_id.casefold(), "exact_opus_route_unavailable")
        attempt.update(requested_model=route.model_id, stage="provider_dispatch", provider_setup_started=True,
                       codex_binary=commitment(codex_bin))
        write(root / "attempt.json", attempt)
        prepared = prepare_workspace_rollout(rollout, rubric=rubric, source_path=Path(source["source_rollout"]["path"]),
                                            judge_model_id=route.model_id)
        with open_opus(route, prepared, root, codex_bin=codex_bin) as judge:
            grade_prepared_rollout(prepared, judge=AttributionJudge(judge, source["catalog_metadata"]), output_root=root)
            write(root / "judge-codex-receipt.json", canonical_value(judge.receipt_for(prepared.task.judge_task_id)), exclusive=True)
        attempt["stage"] = "evidence_verification"
        write(root / "attempt.json", attempt)
        suggestions = _reopen_assessment(root, prepared, source["catalog_metadata"])
        decision = retain_decision(context["skill_catalog_id"], "verified", suggestions=suggestions,
            reviewed_source=source["source_rollout"], evidence_scope=source["selection"],
            all_round_trajectories_inspected_by_opus=False,
            application_status="retain_verified" if all(row["action"] == "retain" for row in suggestions)
                else "verified_selection_guidance_for_next_attempt")
    except Exception as exc:
        receipt = getattr(exc, "receipt", None)
        if receipt is None and judge is not None and prepared is not None:
            try:
                receipt = judge.receipt_for(prepared.task.judge_task_id)
            except Exception:
                pass
        if receipt is not None and not (root / "judge-codex-receipt.json").exists():
            write(root / "judge-codex-failure-receipt.json", canonical_value(receipt), exclusive=True)
        failure = {"schema": "eva.rsi-attribution-failure.v1", "stage": attempt["stage"],
            "category": "route_unavailable" if attempt["stage"] == "route_resolution" else
                "provider_or_transport_failure" if attempt["stage"] == "provider_dispatch" else "evidence_validation_failure",
            "error_type": type(exc).__name__, "provider_setup_started": attempt["provider_setup_started"],
            "backend": "opus_5", "retry_count": 0, "automatic_fallback": False}
        statuses = adapter_statuses(root)
        failure["signed_adapter_statuses"] = statuses
        failure["observed_upstream_responses"] = sum(row["upstream_http_status"] is not None for row in statuses)
        if any(row["upstream_http_status"] == 429 for row in statuses):
            failure["category"] = "upstream_rate_or_budget_rejected"
        elif any(row["upstream_http_status"] in {401, 403} for row in statuses):
            failure["category"] = "upstream_authentication_rejected"
        # No exception text, endpoint, credentials, or upstream body is copied.
        write(root / "failure.json", failure, exclusive=True)
        decision = retain_decision(context["skill_catalog_id"], "failed_attempt", failure=failure)
    # Existing canonical grade writers inherit umask; tighten only our new
    # private output files without changing any retained content bytes.
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(0o600)
    result = {"schema": SCHEMA, "context_blake3": blake3_hex(context), "attempt_id": attempt["attempt_id"],
              "decision": decision}
    write(root / "result.json", result, exclusive=True)
    return result


def verify_result(attempt_root, context):
    """Controller reopens successful suggestions; failures are never attribution."""
    from eva_agent.training.slime_agent_judge import prepare_workspace_rollout
    root = Path(attempt_root) / "skill-attribution"
    result, attempt = read(root / "result.json"), read(root / "attempt.json")
    require(result["schema"] == SCHEMA and result["attempt_id"] == attempt["attempt_id"]
            and result["context_blake3"] == attempt["context_blake3"] == blake3_hex(context)
            and attempt["backend"] == "opus_5" and attempt["retry_count"] == 0
            and attempt["automatic_fallback"] is False, "attribution_attempt_binding")
    require(attempt["implementation"] == commitment(Path(__file__)), "attribution_implementation_changed")
    decision = result["decision"]
    require(decision["action"] == "retain_catalog" and decision["catalog_id"] == context["skill_catalog_id"]
            and decision["skill_change_performed"] is False and decision["causal_effect_established"] is False,
            "attribution_cannot_change_catalog")
    if decision["opus_attribution"] == "verified":
        require(decision["attribution_claimed"] is True and "opus-5" in attempt["requested_model"].casefold(),
                "attribution_opus_identity")
        rollout, rubric, source = prepare_source(context)
        require(read(root / "source.json") == source, "attribution_source_reopen_changed")
        prepared = prepare_workspace_rollout(rollout, rubric=rubric,
            source_path=Path(source["source_rollout"]["path"]), judge_model_id=attempt["requested_model"])
        require(decision["suggestions"] == _reopen_assessment(root, prepared, source["catalog_metadata"]),
                "attribution_suggestions_changed")
        statuses = adapter_statuses(root)
        require(statuses and all(row["adapter_http_status"] == 200 and row["upstream_http_status"] == 200
                                 for row in statuses), "attribution_successful_opus_requests_unproven")
    else:
        require(decision["opus_attribution"] == "failed_attempt" and decision["attribution_claimed"] is False
                and decision["suggestions"] == [] and decision["failure"] == read(root / "failure.json"),
                "attribution_failure_claim_differs")
        require(decision["failure"]["signed_adapter_statuses"] == adapter_statuses(root),
                "attribution_failure_http_evidence_changed")
    verified = {**decision, "result": commitment(root / "result.json")}
    if decision["opus_attribution"] == "verified":
        verified["verified_selection_source"] = {"stage": source["selected_stage"],
            "catalog_id": context["skill_catalog_id"], "catalog": source["catalog_metadata"]}
    return verified
