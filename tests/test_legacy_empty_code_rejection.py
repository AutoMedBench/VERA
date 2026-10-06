"""CPU-only retained CCC shape: real legacy guard → Unix MCP → strict Judge reopen."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy
import json

import pytest

from eva_agent.codex_pipeline import CodexPipelineError, CodexRolloutAdapter
from eva_agent.codex_pipeline.empty_code_rejection import empty_code_rejection_ids
from eva_agent.codex_runtime import CodexRole, CodexSandbox, CodexThreadOptions
from eva_agent.pipeline import (BenchmarkEpisode, BenchmarkSource, Cohort, DeterministicUUIDFactory,
    EvidenceBundle, ImmutableArtifactStore, ModelTarget, PipelineVerifier, Stage, ToolDefinition, ToolRegistry)
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value
from eva_agent.pipeline.judge_material_view import POLICY_VISIBLE_VIEW
from eva_agent.pipeline.runner import evidence_core
from eva_agent.pipeline.verify import PipelineVerificationError
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.sources.legacy_execution import CandidateToolCatalogEntry, _LegacyCandidateToolBackend
from eva_agent.training.agent_judge import AgentJudgeSelectionError
from eva_agent.training.agent_judge_worker import _workspace, _policy_events, _tool_trace
from eva_agent.training.slime_agent_judge import prepare_workspace_rollout, grade_prepared_rollout
from eva_agent.training.teacher_worker import TeacherCandidateContext, execute_single_rollout
from test_turn_mcp_bridge import _MCPBackend, _RuntimeFactory, _transport
from test_agent_judge_worker import _ProviderFreeWorkspaceJudge

ROOT = Path(__file__).resolve().parents[1]
# Exact canonical offer retained by CCC; intentionally no minLength restriction.
OFFER = {"description": "Execute complete bounded code for S3 or S4.", "name": "execute_code",
    "input_schema": {"additionalProperties": False, "properties": {
        "code": {"type": "string"}, "stage": {"enum": ["S3", "S4"]}},
        "required": ["stage", "code"], "type": "object"}}


def _rollout(tmp_path, *, terminal="budget", opt_in=True, arbitrary_backend=False, stage="S4"):
    rubric = load_and_compile_registry(ROOT / "rubrics/source/domain-stage-tables.v1.json").resolve(
        "medxpertqa", stage)
    policy = json.dumps({"schema": "rlevo.med-research-sandbox-policy.v2", "tools": [OFFER]}).encode()
    execution = {"schema": "eva.legacy-candidate-runtime-context.v1",
        "policy_blake3": blake3_bytes(policy),
        "tool_catalog_blake3": blake3_hex([CandidateToolCatalogEntry(**OFFER).to_document()])}
    legacy = _LegacyCandidateToolBackend(modules=SimpleNamespace(), binding_identity={},
        runtime_state_root=tmp_path / "backend", contracts={f"s{i}": {} for i in range(1, 6)},
        evidence_root=tmp_path, image_refs={}, trust_store={}, key_id="fixture", private_key=None,
        docker_binary="must-not-run")

    def unrelated(workspace, arguments):
        raise ValueError("unrelated backend infrastructure error")

    registry = ToolRegistry((ToolDefinition(OFFER["name"], OFFER["description"], OFFER["input_schema"],
        unrelated if arbitrary_backend else legacy.handler("execute_code")),))
    backend = _MCPBackend(server="evamed", groups=((("execute_code", {"stage": stage, "code": ""}),),),
        final_text="", tool_status="completed", turn_status="completed" if terminal == "empty_final" else "failed",
        strip_mcp_is_error=True)
    transport, temp_root = _transport(tmp_path)
    captured = {}

    def options(request, cwd, offers, bridge):
        return CodexThreadOptions(role=CodexRole.STRONG_ACTOR, model=request.model.model_id,
            provider=request.model.provider, cwd=cwd, sandbox=CodexSandbox.READ_ONLY, offered_tools=offers)

    class Actor(CodexRolloutAdapter):
        def run(self, request, tools):
            captured["request"] = request
            return super().run(request, tools)

    actor = Actor(_RuntimeFactory(backend, "legacy-empty-code"), options,
        id_factory=DeterministicUUIDFactory("legacy-empty-adapter"), turn_mcp_factory=transport,
        allow_legacy_empty_code_rejections=opt_in, allow_empty_final_response=terminal == "empty_final",
        controlled_budget_terminal=(lambda: {"kind": "total_output_budget", "count": 8192,
            "limit": 8192, "requests": 5}) if terminal == "budget" else None)
    episode = BenchmarkEpisode(episode_id="fixture-source", source=BenchmarkSource(
        "MedXpertQA", "public-fixture.json", "fixture-v1"), domain="medxpertqa", stage=Stage(stage),
        instruction="Execute the required stage code.", policy_context={"execution_binding": execution},
        initial_files={".eva/source-policy.json": policy,
            ".eva/runtime-context.json": json.dumps(execution).encode(), "input.txt": b"public evidence\n"})
    ids = DeterministicUUIDFactory("legacy-empty-task")
    record = {"sandbox_id": ids.new("sandbox"), "candidate_id": ids.new("candidate"),
        "episode_id": episode.episode_id, "domain": episode.domain, "stage": stage,
        "reward_contract": {"rubric_table": rubric.to_document()},
        "workspace_initial_state": {"files": [], "file_count": 0, "byte_count": 0}}
    try:
        document = execute_single_rollout(record=record, context=TeacherCandidateContext(
            episode=episode, rubric=rubric, tool_registry=registry, turn_mcp_factory=transport,
            skills_factory=None), provider=actor, target=ModelTarget(Cohort.STRONG, "fixture-model", "fixture-provider"),
            workspace_root=tmp_path / "actor", task_id=ids.new("task"), route_id="fixture")
    finally:
        assert list(temp_root.iterdir()) == []
        temp_root.rmdir()
    assert all(not session.s3_attempts and not session.s4_attempts for session in legacy._sessions.values())
    value = canonical_value(document)
    request = captured["request"]
    evidence = EvidenceBundle(bundle_id=ids.new("bundle"), rollout_id=request.rollout_id,
        sandbox_manifest=request.sandbox, model=request.model, policy_visible_context=request.policy_visible_context,
        context_blake3=blake3_hex(request.policy_visible_context),
        workspace_before=_workspace(value["workspace_before"], expected_label="before"),
        workspace_after=_workspace(value["workspace_after"], expected_label="after"),
        tool_trace=_tool_trace(value["tool_trace"]), policy_events=_policy_events(value["messages"]),
        assistant_output=value["assistant_output"], provider_receipt_blake3=value["provider_receipt_blake3"],
        safe_provider_metadata=value["provider_metadata"], bundle_blake3="")
    return value, rubric, replace(evidence, bundle_blake3=blake3_hex(evidence_core(evidence)))


@pytest.mark.parametrize("terminal,stage", [("budget", "S4"), ("empty_final", "S4"), ("semantic", "S3")])
def test_real_empty_code_guard_reopens_and_requires_workspace_judge(tmp_path, terminal, stage):
    value, rubric, evidence = _rollout(tmp_path, terminal=terminal, stage=stage)
    PipelineVerifier(ImmutableArtifactStore(tmp_path / "artifacts"))._verify_trajectory(
        SimpleNamespace(evidence=evidence, cohort=Cohort.STRONG))
    assert value["judge_calls"] == 0 and value["selection_pending"] is True
    assert value["score_kind"] == "selection_pending"
    result, = evidence.tool_trace.results
    assert (result.status, result.error_code, result.output) == ("immutable_failure", "ValueError", None)
    assert result.workspace_before_blake3 == result.workspace_after_blake3
    assert evidence.workspace_before.tree_blake3 == evidence.workspace_after.tree_blake3
    metadata = value["provider_metadata"]
    assert metadata["legacy_empty_code_rejections"]["call_ids"] == [result.call_id]
    call, = metadata["codex_turn_receipt"]["tool_calls"]
    assert call["status"] == "completed" and "isError" not in call["output"]["result"]
    assert call["output"]["result"]["structuredContent"]["tool_result"] == canonical_value(result)
    prepared = prepare_workspace_rollout(value, rubric=rubric, source_path=tmp_path / "rollout.json",
        judge_model_id="claude-opus-5", material_view=POLICY_VISIBLE_VIEW)
    assert prepared.evidence.tool_trace == evidence.tool_trace
    assert prepared.trajectory.document == value
    grade = grade_prepared_rollout(prepared, judge=_ProviderFreeWorkspaceJudge(), output_root=tmp_path)
    assert grade["workspace_inspected"] is True
    assert grade["judge_tool_trace"]["workspace_read_count"] > 0


@pytest.mark.parametrize("terminal", ["budget", "empty_final", "semantic"])
@pytest.mark.parametrize("opt_in,arbitrary_backend", [(False, False), (True, True)])
def test_opt_in_and_real_handler_are_required(tmp_path, terminal, opt_in, arbitrary_backend):
    with pytest.raises(CodexPipelineError, match="infrastructure failure|turn status is not completed"):
        _rollout(tmp_path, terminal=terminal, opt_in=opt_in, arbitrary_backend=arbitrary_backend)


def test_only_exact_guard_and_bound_original_evidence_match(tmp_path):
    value, _, evidence = _rollout(tmp_path)
    metadata = value["provider_metadata"]
    receipt = copy.deepcopy(metadata["codex_turn_receipt"])
    original = copy.deepcopy(receipt["tool_calls"][0])
    result, = evidence.tool_trace.results
    bootstrap = {row.path: row.content for row in evidence.workspace_before.files}
    catalog = metadata["legacy_empty_code_rejections"]["offered_catalog"]
    mapping = metadata["codex_to_pipeline_call_ids"]

    def match(*, arguments=None, outcome=result, call_change=None, source=bootstrap, offers=catalog):
        call = copy.deepcopy(original)
        if arguments is not None:
            call["arguments"] = arguments
        core = {"schema": "eva.codex-pipeline-tool-observation.v1", "call_id": outcome.call_id,
            "name": outcome.name, "arguments": call["arguments"], "tool_result": canonical_value(outcome)}
        call["output"]["result"]["structuredContent"] = {**core, "bridge_receipt_blake3": blake3_hex(core)}
        if call_change is not None:
            call_change(call)
        return empty_code_rejection_ids(receipt=SimpleNamespace(**{**receipt,
            "tool_calls": (SimpleNamespace(**call),)}), trace=SimpleNamespace(results=(outcome,)),
            mapping=mapping, catalog=offers, context=evidence.policy_visible_context, bootstrap=source)

    assert match() == (result.call_id,)
    for arguments in ({"stage": "S4", "code": "pass"}, {"stage": "S4", "code": " "},
        {"stage": "S2", "code": ""}, {"stage": "S4", "code": "", "skill_id": "x"}):
        assert match(arguments=arguments) == ()
    for change in ({"error_code": "RuntimeError"}, {"output": {}}, {"status": "completed"},
        {"workspace_after_blake3": blake3_hex("changed")}):
        assert match(outcome=replace(result, **change)) == ()
    assert match(call_change=lambda call: call.update(status="failed")) == ()
    assert match(call_change=lambda call: call["output"]["result"]["structuredContent"].update(
        bridge_receipt_blake3=blake3_hex("wrong"))) == ()
    assert match(source={**bootstrap, ".eva/source-policy.json": b"{}"}) == ()
    assert match(offers=[{**OFFER, "description": "unbound"}]) == ()


@pytest.mark.parametrize("mutation", ["ids", "catalog", "missing"])
def test_invalid_sidecar_cannot_reach_judge(tmp_path, mutation):
    value, rubric, _ = _rollout(tmp_path)
    metadata = value["provider_metadata"]
    if mutation == "ids":
        metadata["legacy_empty_code_rejections"]["call_ids"] = []
    elif mutation == "catalog":
        metadata["legacy_empty_code_rejections"]["offered_catalog"][0]["description"] = "unbound"
    else:
        del metadata["legacy_empty_code_rejections"]
    with pytest.raises((AgentJudgeSelectionError, PipelineVerificationError)):
        prepare_workspace_rollout(value, rubric=rubric, source_path=tmp_path / "rollout.json",
            judge_model_id="claude-opus-5", material_view=POLICY_VISIBLE_VIEW)
    assert not (tmp_path / "grade.json").exists()
