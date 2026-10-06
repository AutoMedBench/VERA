"""CPU-only real Unix-MCP → rollout → strict reopen → workspace Judge fixtures."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from eva_agent.codex_pipeline import CodexRolloutAdapter
from eva_agent.codex_runtime import CodexRole, CodexSandbox, CodexThreadOptions
from eva_agent.pipeline import (BenchmarkEpisode, BenchmarkSource, Cohort, DeterministicUUIDFactory,
    EvidenceBundle, ImmutableArtifactStore, ModelTarget, PipelineVerifier, Stage, ToolDefinition, ToolRegistry)
from eva_agent.pipeline.digests import blake3_hex, canonical_value
from eva_agent.pipeline.judge_material_view import POLICY_VISIBLE_VIEW, REQUIRED_PATHS
from eva_agent.pipeline.runner import evidence_core
from eva_agent.pipeline.verify import PipelineVerificationError, verify_codex_judge_source
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.training.agent_judge import AgentJudgeSelectionError
from eva_agent.training.agent_judge_worker import _workspace, _policy_events, _tool_trace
from eva_agent.training.slime_agent_judge import prepare_workspace_rollout, grade_prepared_rollout
from eva_agent.training.teacher_worker import TeacherCandidateContext, execute_single_rollout
from test_turn_mcp_bridge import _MCPBackend, _RuntimeFactory, _transport
from test_agent_judge_worker import _ProviderFreeWorkspaceJudge

ROOT = Path(__file__).resolve().parents[1]


def _rollout(tmp_path, *, valid_action=False, terminal="failed", strip_is_error=False):
    rubric = load_and_compile_registry(ROOT / "rubrics/source/domain-stage-tables.v1.json").resolve(
        "medxpertqa", "S3")
    schema = {"type": "object", "properties": {"stage": {"const": "S3"},
        "code": {"type": "string", "minLength": 1}}, "required": ["stage", "code"],
        "additionalProperties": False}
    calls = []

    def execute(workspace, arguments):
        calls.append(arguments)
        return {"seed": workspace.read_bytes("input.txt").decode()}

    registry = ToolRegistry((ToolDefinition("execute_code", "Exact S3 fixture program.", schema, execute),))
    groups = ((("execute_code", {"code": "pass"}),),)
    if valid_action:
        groups += ((("execute_code", {"stage": "S3", "code": "pass"}),),)
    backend = _MCPBackend(server="evamed", groups=groups, final_text="fixture terminal",
        tool_status="completed" if valid_action else "failed", turn_status=terminal,
        strip_mcp_is_error=strip_is_error)
    transport, temp_root = _transport(tmp_path)
    captured = {}

    def options(request, cwd, offers, bridge):
        del bridge
        return CodexThreadOptions(role=CodexRole.STRONG_ACTOR, model=request.model.model_id,
            provider=request.model.provider, cwd=cwd, sandbox=CodexSandbox.READ_ONLY, offered_tools=offers)

    class Actor(CodexRolloutAdapter):
        def run(self, request, tools):
            captured["request"] = request
            return super().run(request, tools)

    actor = Actor(_RuntimeFactory(backend, "native-rejection-e2e"), options,
        id_factory=DeterministicUUIDFactory("native-rejection-adapter"), turn_mcp_factory=transport,
        allow_schema_validation_rejections=True,
        controlled_budget_terminal=(lambda: {"kind": "total_output_budget", "count": 8192,
            "limit": 8192, "requests": 5}) if terminal == "failed" else None)
    episode = BenchmarkEpisode(episode_id="fixture-source", source=BenchmarkSource(
        "MedXpertQA", "public-fixture.json", "fixture-v1"), domain="medxpertqa", stage=Stage.S3,
        instruction="Call execute_code with stage S3 and code.", policy_context={"task": "public fixture"},
        initial_files={"input.txt": b"public evidence\n"})
    ids = DeterministicUUIDFactory("native-rejection-task")
    record = {"sandbox_id": ids.new("sandbox"), "candidate_id": ids.new("candidate"),
        "episode_id": episode.episode_id, "domain": episode.domain, "stage": "S3",
        "reward_contract": {"rubric_table": rubric.to_document()},
        "workspace_initial_state": {"files": [], "file_count": 0, "byte_count": 0}}
    document = execute_single_rollout(record=record, context=TeacherCandidateContext(
        episode=episode, rubric=rubric, tool_registry=registry, turn_mcp_factory=transport,
        skills_factory=None), provider=actor, target=ModelTarget(Cohort.STRONG, "fixture-model", "fixture-provider"),
        workspace_root=tmp_path / "actor", task_id=ids.new("task"), route_id="fixture")
    assert list(temp_root.iterdir()) == []
    temp_root.rmdir()
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
    evidence = replace(evidence, bundle_blake3=blake3_hex(evidence_core(evidence)))
    assert len(calls) == int(valid_action)
    return value, rubric, evidence


@pytest.mark.parametrize("valid_action", [False, True])
def test_real_unix_proxy_rollout_failed_action_is_judgeable_without_host_success(tmp_path, valid_action):
    value, rubric, evidence = _rollout(tmp_path, valid_action=valid_action)
    verifier = PipelineVerifier(ImmutableArtifactStore(tmp_path / "artifacts"))
    verifier._verify_trajectory(SimpleNamespace(evidence=evidence, cohort=Cohort.STRONG))
    assert len(evidence.tool_trace.results) == int(valid_action)
    proof = value["provider_metadata"]["schema_validation_rejections"][0]["rejection"]
    assert proof["host_execution_attempted"] is False and proof["host_result_claimed"] is False
    prepared = prepare_workspace_rollout(value, rubric=rubric, source_path=tmp_path / "rollout.json",
        judge_model_id="claude-opus-5", material_view=POLICY_VISIBLE_VIEW)
    files = {row.path: row.content for row in prepared.evidence.workspace_after.files}
    visible = json.loads(files[REQUIRED_PATHS[2]])
    assert visible["events"] == value["messages"]
    assert visible["native_original_mcp_responses"][0]["output"]["result"]["isError"] is True
    assert "Missing required properties: stage." in json.dumps(visible)
    assert prepared.evidence.tool_trace == evidence.tool_trace
    assert prepared.trajectory.document == value
    # CPU fixture Judge performs actual read operations. Its synthetic assessment
    # is a test outcome only, never a medical/provider score.
    grade = grade_prepared_rollout(prepared, judge=_ProviderFreeWorkspaceJudge(), output_root=tmp_path)
    assert grade["workspace_inspected"] is True
    assert grade["judge_tool_trace"]["workspace_read_count"] > 0


@pytest.mark.parametrize("mutation", ["host_claim", "schema", "required_feedback", "missing_budget", "unknown_key"])
def test_rejection_tampering_or_unproven_failure_cannot_reach_judge(tmp_path, mutation):
    value, rubric, _ = _rollout(tmp_path)
    metadata = value["provider_metadata"]
    if mutation == "host_claim":
        metadata["schema_validation_rejections"][0]["rejection"]["host_execution_attempted"] = True
    elif mutation == "schema":
        metadata["schema_validation_rejection_catalog"][0]["input_schema"]["required"] = ["code"]
    elif mutation == "required_feedback":
        metadata["schema_validation_rejections"][0]["rejection"]["validation_issues"][0]["required_properties"] = []
    elif mutation == "missing_budget":
        del metadata["controlled_budget_termination"]
        metadata["schema"] = "eva.codex-provider-rollout-projection.v1"
    else:
        metadata["schema_validation_rejections"][0]["rejection"]["imagined_result"] = {}
    with pytest.raises((AgentJudgeSelectionError, PipelineVerificationError)):
        prepare_workspace_rollout(value, rubric=rubric, source_path=tmp_path / "rollout.json",
            judge_model_id="claude-opus-5", material_view=POLICY_VISIBLE_VIEW)
    assert not (tmp_path / "grade.json").exists()


def test_normal_pipeline_still_rejects_changed_original_provider_binding(tmp_path):
    _, _, evidence = _rollout(tmp_path)
    bad = replace(evidence, provider_receipt_blake3=blake3_hex("not-original"))
    verifier = PipelineVerifier(ImmutableArtifactStore(tmp_path / "artifacts"))
    with pytest.raises(PipelineVerificationError, match="provider projection receipt"):
        verifier._verify_trajectory(SimpleNamespace(evidence=bad, cohort=Cohort.STRONG))


def test_actual_codex_failed_item_shape_omits_mcp_is_error(tmp_path):
    value, rubric, evidence = _rollout(tmp_path, strip_is_error=True)
    verify_codex_judge_source(evidence)
    prepared = prepare_workspace_rollout(value, rubric=rubric, source_path=tmp_path / "rollout.json",
        judge_model_id="claude-opus-5", material_view=POLICY_VISIBLE_VIEW)
    response = prepared.trajectory.document["provider_metadata"]["codex_turn_receipt"]["tool_calls"][0]
    assert response["status"] == "failed"
    assert "isError" not in response["output"]["result"]


def test_completed_turn_with_rejected_action_keeps_actual_terminal(tmp_path):
    value, rubric, evidence = _rollout(tmp_path, terminal="completed")
    verify_codex_judge_source(evidence)
    prepared = prepare_workspace_rollout(value, rubric=rubric, source_path=tmp_path / "rollout.json",
        judge_model_id="claude-opus-5", material_view=POLICY_VISIBLE_VIEW)
    assert "controlled_budget_termination" not in prepared.trajectory.document["provider_metadata"]
    assert prepared.evidence.assistant_output == "fixture terminal"
