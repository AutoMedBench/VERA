from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eva_agent.pipeline.digests import canonical_json_bytes
from eva_agent.training.agent_judge import AgentJudgeSelectionError
from eva_agent.training import slime_agent_judge
from test_agent_judge_worker import _fixture, _ProviderFreeWorkspaceJudge


def _generic_rollout(tmp_path):
    registry, path, task = _fixture(tmp_path)
    rollout = json.loads(path.read_text())
    rollout.update(schema="eva.sglang-grpo-rollout.v1", provider="sglang-local",
                   model_id="Qwen/Qwen3.5-9B", route_id="qwen_sglang_training")
    path.write_bytes(canonical_json_bytes(rollout))
    return rollout, registry.resolve("medxpertqa", "E2E"), path, task.judge_model_id


def test_generic_actor_preserves_provenance_and_uses_workspace_rubric(tmp_path):
    rollout, rubric, path, judge_model = _generic_rollout(tmp_path)
    prepared = slime_agent_judge.prepare_workspace_rollout(
        rollout, rubric=rubric, source_path=path, judge_model_id=judge_model)
    grade = slime_agent_judge.grade_prepared_rollout(
        prepared, judge=_ProviderFreeWorkspaceJudge(), output_root=tmp_path)
    assert grade["reward"] == 1.0
    assert grade["item_scores_bps"] == dict(rubric.score(grade["item_scores_bps"]).item_scores_bps)
    assert grade["rubric_digest"] == rubric.digest
    assert grade["source_actor_provider"] == "sglang-local"
    assert grade["source_actor_schema"] == "eva.sglang-grpo-rollout.v1"
    assert grade["judge_tool_trace"]["workspace_read_count"] > 0
    assert all(grade["evidence_by_item"].values())
    assert prepared.trajectory.document == rollout


def test_missing_real_workspace_coverage_produces_no_reward(tmp_path):
    rollout, rubric, path, judge_model = _generic_rollout(tmp_path)
    prepared = slime_agent_judge.prepare_workspace_rollout(
        rollout, rubric=rubric, source_path=path, judge_model_id=judge_model)
    with pytest.raises(AgentJudgeSelectionError, match="did not completely read"):
        slime_agent_judge.grade_prepared_rollout(
            prepared, judge=_ProviderFreeWorkspaceJudge(complete=False), output_root=tmp_path)
    assert not (tmp_path / "score.json").exists()


def test_service_failure_propagates_without_synthetic_reward(tmp_path, monkeypatch):
    rollout, rubric, _, _ = _generic_rollout(tmp_path)

    def failing(*args, **kwargs):
        raise RuntimeError("service unavailable")

    monkeypatch.setattr(slime_agent_judge, "_grade_rollout_sync", failing)
    with pytest.raises(RuntimeError, match="service unavailable"):
        asyncio.run(slime_agent_judge.grade_rollout(rollout, rubric=rubric, output_root=tmp_path, sample_id="sample-0"))
    assert not (tmp_path / "score.json").exists()


def test_backend_selection_is_explicit_and_default_stays_opus(monkeypatch):
    monkeypatch.delenv("EVA_SLIME_JUDGE_BACKEND", raising=False)
    assert slime_agent_judge.resolve_judge_backend() == "opus_5"
    monkeypatch.setenv("EVA_SLIME_JUDGE_BACKEND", "native_astra")
    assert slime_agent_judge.resolve_judge_backend() == "native_astra"
    assert slime_agent_judge.resolve_judge_backend("opus_5") == "opus_5"
    with pytest.raises(AgentJudgeSelectionError, match="backend must"):
        slime_agent_judge.resolve_judge_backend("automatic")


def test_original_opus_guard_and_explicit_astra_guard_remain():
    from eva_agent.codex_pipeline import CodexOpus5AgentJudge, CodexPipelineError

    with pytest.raises(CodexPipelineError, match="must be Opus 5"):
        CodexOpus5AgentJudge(model_id="gpt-6-astra")
    with pytest.raises(CodexPipelineError, match="exact gpt-6-astra"):
        slime_agent_judge.NativeAstraWorkspaceJudge(model_id="claude-opus-5")


def test_native_engine_uses_same_replayed_workspace_checks(tmp_path):
    from dataclasses import replace
    from eva_agent.pipeline import DeterministicUUIDFactory, JudgeRequest
    from test_codex_pipeline_adapter import _RuntimeFactory, _evidence, _judge_options, _judge_script, _rubric

    rubric = _rubric()
    evidence = _evidence(tmp_path)
    factory = _RuntimeFactory(_judge_script(evidence, rubric))
    judge = slime_agent_judge.NativeAstraWorkspaceJudge(
        factory, lambda request, offers: replace(_judge_options(request, offers), provider="eva_native_astra"),
        id_factory=DeterministicUUIDFactory("native-judge"),
    )
    request = JudgeRequest(
        judgment_id=DeterministicUUIDFactory("native-judgment").new("judgment"),
        judge_model_id="gpt-6-astra", policy_visible_context=evidence.policy_visible_context,
        workspace_evidence=evidence, judge_only_reference={"reference_answer": "private fixture"},
    )
    assessment = judge.judge(request, rubric)
    assert assessment.judge_model_id == "gpt-6-astra"
    assert assessment.agent_trace.content_inspection_count == 1
    assert assessment.agent_trace.inspected_evidence_refs == ("workspace:after:seed.txt",)
    receipt = judge.receipt_for(request.judgment_id)
    assert receipt.model == "gpt-6-astra" and receipt.provider == "eva_native_astra"


def test_explicit_native_dispatch_never_resolves_hub_or_claims_opus(tmp_path, monkeypatch):
    import eva_agent.codex_providers as providers
    rollout, rubric, _, _ = _generic_rollout(tmp_path)
    monkeypatch.setattr(providers, "load_codex_provider_routes", lambda **_: pytest.fail("native lane opened Hub routes"))
    monkeypatch.setattr(slime_agent_judge, "_open_opus_judge", lambda *_: pytest.fail("implicit Opus fallback"))

    @contextmanager
    def native(prepared, output_root):
        assert prepared.task.judge_route_id == "gpt_6_astra"
        assert prepared.task.judge_model_id == "gpt-6-astra"
        yield _ProviderFreeWorkspaceJudge()

    monkeypatch.setattr(slime_agent_judge, "_open_native_astra_judge", native)
    grade = asyncio.run(slime_agent_judge.grade_rollout(
        rollout, rubric=rubric, output_root=tmp_path / "native", sample_id="sample-0", backend="native_astra"))
    assert grade["reward"] == 1.0 and grade["workspace_inspected"] is True
    assert grade["judge_backend"] == "native_astra"
    assert grade["judge_model_id"] == "gpt-6-astra"
    provenance = grade["judge_provenance"]
    assert provenance["returned_model"] is None
    assert provenance["returned_model_status"] == "not-exposed-by-codex-app-server"
    assert provenance["automatic_fallback"] is False
    assert provenance["observed_provider_roundtrips"] is None
    assert provenance["maximum_transport_projection_turns"] == 65
    assert "not-observed-provider-roundtrips" in provenance["provider_turn_count_semantics"]
    assert "opus" not in json.dumps(provenance).lower()
    assert json.loads((tmp_path / "native/sample-0/rollout.json").read_text()) == rollout


def test_native_service_failure_has_no_reward_or_backend_switch(tmp_path, monkeypatch):
    import eva_agent.codex_providers as providers
    rollout, rubric, _, _ = _generic_rollout(tmp_path)
    monkeypatch.setattr(providers, "load_codex_provider_routes", lambda **_: pytest.fail("fallback attempted"))

    @contextmanager
    def unavailable(*_):
        raise RuntimeError("native service unavailable")
        yield  # pragma: no cover

    monkeypatch.setattr(slime_agent_judge, "_open_native_astra_judge", unavailable)
    with pytest.raises(RuntimeError, match="native service unavailable"):
        asyncio.run(slime_agent_judge.grade_rollout(
            rollout, rubric=rubric, output_root=tmp_path / "native", sample_id="sample-0", backend="native_astra"))
    root = tmp_path / "native/sample-0"
    assert not (root / "grade.json").exists() and not (root / "score.json").exists()
    failure = json.loads((root / "failure.json").read_text())
    assert failure["reward_emitted"] is False and failure["automatic_fallback"] is False


def test_native_launch_keeps_auth_private_and_judge_options_read_only(tmp_path, monkeypatch):
    from eva_agent import codex_runtime
    from eva_agent.training import native_astra_teacher as native

    rollout, rubric, path, _ = _generic_rollout(tmp_path)
    prepared = slime_agent_judge.prepare_workspace_rollout(
        rollout, rubric=rubric, source_path=path, judge_model_id="gpt-6-astra", judge_route_id="gpt_6_astra")
    source_auth = tmp_path / "source-auth.json"
    source_auth.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "fixture-private-token"}}))
    original_auth = source_auth.read_bytes()
    output = tmp_path / "artifacts"
    output.mkdir()
    launches = []
    original_launch = native.native_astra_launch_options

    def launch(**kwargs):
        launches.append(kwargs)
        return original_launch(**kwargs)

    class Runner:
        def __init__(self, factory):
            self.started = self.closed = False
        def start(self):
            self.started = True
        def close(self):
            self.closed = True
        def run_once(self, *_):
            pytest.fail("provider must not be called during startup fixture")

    monkeypatch.setattr(native, "native_astra_launch_options", launch)
    monkeypatch.setattr(codex_runtime, "PersistentCodexRuntimeRunner", Runner)
    monkeypatch.setattr(slime_agent_judge.shutil, "which", lambda _: "/bin/true")
    with slime_agent_judge._open_native_astra_judge(prepared, output, source_auth) as judge:
        assert judge._maximum_workspace_tool_frontiers == 32
        assert judge._maximum_workspace_tool_calls == 64
        assert judge._turn_timeout_seconds == 240
        private = launches[0]["isolation_root"]
        assert output not in private.parents and private.stat().st_mode & 0o777 == 0o700
        assert (private / "codex/auth.json").stat().st_mode & 0o777 == 0o600
        assert (private / "codex/auth.json").read_bytes() == original_auth
        options = judge._options(SimpleNamespace(judge_model_id="gpt-6-astra"), ())
        assert options.role.value == "judge" and options.sandbox.value == "read-only"
        assert options.model == "gpt-6-astra" and options.provider == "eva_native_astra"
        assert options.config["model_providers"]["eva_native_astra"]["requires_openai_auth"] is True
        assert "base_url" not in options.config["model_providers"]["eva_native_astra"]
        assert "For EVERY rubric item, including a zero score" in options.base_instructions
        assert "never as an item's only workspace evidence" in options.base_instructions
        for ref in prepared.source_workspace_refs:
            assert ref in options.base_instructions
        assert not list(output.rglob("auth.json"))
    assert not private.exists() and source_auth.read_bytes() == original_auth


def test_native_temp_root_inside_artifacts_fails_before_auth_copy(tmp_path, monkeypatch):
    from eva_agent.training import native_astra_teacher as native
    rollout, rubric, path, _ = _generic_rollout(tmp_path)
    prepared = slime_agent_judge.prepare_workspace_rollout(
        rollout, rubric=rubric, source_path=path, judge_model_id="gpt-6-astra", judge_route_id="gpt_6_astra")
    output = tmp_path / "artifacts"
    monkeypatch.setattr(slime_agent_judge, "gettempdir", lambda: str(output / "tmp"))
    monkeypatch.setattr(native, "private_native_auth_copy", lambda *_: pytest.fail("auth copy happened before boundary check"))
    with pytest.raises(AgentJudgeSelectionError, match="outside artifacts"):
        with slime_agent_judge._open_native_astra_judge(prepared, output, tmp_path / "source-auth.json"):
            pytest.fail("unsafe temporary location accepted")


@pytest.mark.parametrize("shared_execution", [False, True])
def test_native_split_transport_frontiers_use_only_exact_issued_results(tmp_path, shared_execution):
    from eva_agent.codex_pipeline import CodexJudgeToolExecutionBridge, CodexOpus5AgentJudge, CodexPipelineError
    from eva_agent.codex_runtime import CodexToolCall
    from eva_agent.pipeline import DeterministicUUIDFactory
    from eva_agent.pipeline.digests import blake3_hex
    from eva_agent.pipeline.judge_workspace import JudgeWorkspaceTools
    from test_codex_pipeline_adapter import _evidence

    ids = DeterministicUUIDFactory("native-split-reads")
    bridge = CodexJudgeToolExecutionBridge(JudgeWorkspaceTools(_evidence(tmp_path), id_factory=ids), id_factory=ids)
    calls = []
    arguments = {"snapshot": "after", "path": "seed.txt", "offset": 0, "max_bytes": 65536}
    if shared_execution:
        observations = bridge.execute_group((("workspace_read", arguments),) * 2)
    else:
        observations = tuple(bridge.execute_group((("workspace_read", arguments),))[0] for _ in range(2))
    for index, observation in enumerate(observations):
        core = {
            "tool_call_id": ids.new("codex-call"), "upstream_item_id": f"native-{index}",
            "tool_type": "mcpToolCall", "name": "workspace_read", "mcp_server": "evamed-judge",
            "mcp_tool": "workspace_read", "fully_qualified_name": "evamed-judge/workspace_read",
            "status": "completed", "arguments": arguments,
            "output": {"result": {"content": [], "structuredContent": observation.structured_content}, "error": None},
            "lifecycle": ("item/started", "item/completed"), "first_event_sequence": index,
        }
        calls.append(CodexToolCall(**core, receipt_blake3=blake3_hex(core)))
    receipt = SimpleNamespace(receipt_blake3="a" * 64, max_parallelism_observed=2)
    transport = tuple((call,) for call in calls) if shared_execution else (tuple(calls),)
    runner = SimpleNamespace(run_once=lambda *_: pytest.fail("no provider call"))
    options = lambda *_: None
    opus = CodexOpus5AgentJudge(options_factory=options, model_id="claude-opus-5", runner=runner)
    assert opus._live_judge_groups(receipt, transport, bridge) is transport
    with pytest.raises(CodexPipelineError, match="transport frontier differs|receipts and immutable execution differ"):
        bridge.bind_receipt(receipt, transport)

    native = slime_agent_judge.NativeAstraWorkspaceJudge(options_factory=options, runner=runner)
    partition = native._live_judge_groups(receipt, transport, bridge)
    assert partition == ((tuple(calls),) if shared_execution else ((calls[0],), (calls[1],)))
    assert len(bridge.bind_receipt(receipt, partition)) == 2
    audit = native.transport_frontier_audit
    assert audit["original_transport_groups"] == [[call.tool_call_id for call in group] for group in transport]
    assert audit["actual_execution_max_parallelism"] == (2 if shared_execution else 1)
    assert audit["actual_execution_frontier_count"] == (1 if shared_execution else 2)
    assert audit["calls_reexecuted"] is False
    if not shared_execution:
        # Notification order is retained as transport evidence, while only
        # immutable local frontier numbers determine actual execution order.
        reordered = native._live_judge_groups(receipt, ((calls[1],), (calls[0],)), bridge)
        assert reordered == ((calls[0],), (calls[1],))
        assert native.transport_frontier_audit["original_transport_groups"] == [
            [calls[1].tool_call_id], [calls[0].tool_call_id]]
    for invalid in (((calls[0],),), ((calls[1], calls[0]), (calls[0],))):
        with pytest.raises(CodexPipelineError):
            bridge.execution_groups_for_receipt(receipt, invalid)
    altered = {**calls[0].core(), "arguments": {**calls[0].arguments, "offset": 1}}
    tampered = CodexToolCall(**altered, receipt_blake3=blake3_hex(altered))
    with pytest.raises(CodexPipelineError, match="exact issued result"):
        bridge.execution_groups_for_receipt(receipt, ((tampered, calls[1]),))


def test_native_transport_overlap_peak_is_independent_of_disjoint_projection():
    from eva_agent.codex_pipeline import CodexPipelineError
    from eva_agent.codex_pipeline.adapter import _tool_groups

    calls, events = [], []
    for index, (start, end) in enumerate(((0, 4), (1, 7), (5, 8), (6, 9))):
        item_id = f"call-{index}"
        calls.append(SimpleNamespace(upstream_item_id=item_id, first_event_sequence=start, tool_call_id=item_id))
        for sequence, method in ((start, "item/started"), (end, "item/completed")):
            events.append(SimpleNamespace(sequence=sequence, method=method, payload={"item": {"id": item_id}}))
    receipt = SimpleNamespace(tool_calls=calls, events=events, max_parallelism_observed=3)
    with pytest.raises(CodexPipelineError, match="parallel call evidence differs"):
        _tool_groups(receipt)
    groups = _tool_groups(receipt, verify_event_peak=True)
    assert groups == (tuple(calls[:2]), tuple(calls[2:]))
    assert max(map(len, groups)) == 2 and receipt.max_parallelism_observed == 3
    receipt.max_parallelism_observed = 4
    with pytest.raises(CodexPipelineError, match="parallel call evidence differs"):
        _tool_groups(receipt, verify_event_peak=True)


def test_native_frontier_budget_reaches_independent_coverage_without_changing_default(tmp_path, monkeypatch):
    import test_agent_judge_worker as fixtures
    from eva_agent.pipeline import JudgeRequest
    from eva_agent.pipeline.judge_workspace import JudgeWorkspaceTools
    from eva_agent.training.agent_judge_worker import judgment_document

    class SerialWorkspaceTools(JudgeWorkspaceTools):
        def execute_group(self, calls):
            return tuple(result for call in calls for result in super(SerialWorkspaceTools, self).execute_group((call,)))

    monkeypatch.setattr(fixtures, "JudgeWorkspaceTools", SerialWorkspaceTools)
    rollout, rubric, path, _ = _generic_rollout(tmp_path)
    prepared = slime_agent_judge.prepare_workspace_rollout(
        rollout, rubric=rubric, source_path=path, judge_model_id="gpt-6-astra", judge_route_id="gpt_6_astra")
    request = JudgeRequest(
        judgment_id=prepared.task.judge_task_id, judge_model_id=prepared.task.judge_model_id,
        policy_visible_context=prepared.evidence.policy_visible_context,
        workspace_evidence=prepared.evidence, judge_only_reference={},
    )
    assessment = _ProviderFreeWorkspaceJudge().judge(request, rubric)
    assert 4 < assessment.agent_trace.frontier_count <= 32
    with pytest.raises(AgentJudgeSelectionError, match="exceeded its inspection budget"):
        judgment_document(prepared, assessment)
    judgment_document(prepared, assessment, maximum_workspace_tool_frontiers=32)
    with pytest.raises(AgentJudgeSelectionError, match="exceeded its inspection budget"):
        judgment_document(prepared, assessment, maximum_workspace_tool_frontiers=32,
                            maximum_provider_turn_count=1)
    judgment_document(prepared, assessment, maximum_workspace_tool_frontiers=32,
                       maximum_provider_turn_count=65)
    judge = slime_agent_judge.NativeAstraWorkspaceJudge(
        options_factory=lambda *_: None, runner=SimpleNamespace(run_once=lambda *_: None),
        maximum_workspace_tool_frontiers=32)
    monkeypatch.setattr(judge, "judge", _ProviderFreeWorkspaceJudge().judge)
    grade = slime_agent_judge.grade_prepared_rollout(prepared, judge=judge, output_root=tmp_path)
    assert grade["reward"] == 1.0
    retained = json.loads((tmp_path / "assessment.json").read_text())
    assert retained["agent_trace"]["frontier_count"] > 4


@pytest.mark.parametrize("native", [False, True])
def test_original_source_citation_rule_reaches_model_request_only_for_native(tmp_path, native):
    from eva_agent.codex_pipeline import CodexOpus5AgentJudge
    from eva_agent.codex_runtime import CodexRole, CodexRuntimeError, CodexSandbox, CodexThreadOptions
    from test_agent_judge_worker import _CaptureCompactTurnRunner

    rollout, rubric, path, opus_model = _generic_rollout(tmp_path)
    prepared = slime_agent_judge.prepare_workspace_rollout(
        rollout, rubric=rubric, source_path=path,
        judge_model_id="gpt-6-astra" if native else opus_model,
        judge_route_id="gpt_6_astra" if native else "opus_5")
    capture = _CaptureCompactTurnRunner()

    def options(request, offers):
        return CodexThreadOptions(role=CodexRole.JUDGE, model=request.judge_model_id,
            provider="provider-free", cwd=str(tmp_path), sandbox=CodexSandbox.READ_ONLY,
            offered_tools=offers, ephemeral=True)

    cls = slime_agent_judge.NativeAstraWorkspaceJudge if native else CodexOpus5AgentJudge
    judge = cls(options_factory=options, model_id=prepared.task.judge_model_id,
                runner=capture, compact_evidence_index=True, fast_workspace_judge=True,
                turn_timeout_seconds=240)
    with pytest.raises(CodexRuntimeError, match="provider-free capture"):
        slime_agent_judge.grade_prepared_rollout(prepared, judge=judge, output_root=tmp_path)
    # This is the actual CodexTurnInput assembled by the unchanged adapter,
    # captured at the provider boundary rather than merely a local helper value.
    reference = capture.turn_input.judge_only_context["judge_only_reference"]
    if not native:
        assert "native_source_citation_requirements" not in reference
        return
    requirement = reference["native_source_citation_requirements"]
    assert tuple(requirement["original_source_workspace_refs"]) == prepared.source_workspace_refs
    assert requirement["minimum_original_source_citations_per_item"] == 1
    assert requirement["actual_successful_read_required"] is True
    assert requirement["applies_to_zero_scores"] is True
    assert requirement["allowlist_does_not_prove_inspection"] is True
    assert requirement["audit_sidecars_alone_satisfy_requirement"] is False
    assert any(row.path.startswith(".eva-agent-judge/") for row in prepared.evidence.workspace_after.files)
    assert not any(".eva-agent-judge/" in ref for ref in requirement["original_source_workspace_refs"])
    assert "Never cite a nonexistent output" in requirement["instruction"]
    assert "unrelated source citation" in requirement["instruction"]
    assert not (tmp_path / "grade.json").exists()


@pytest.mark.parametrize("include_original_source", [False, True])
def test_missing_artifact_zero_requires_real_source_citation_plus_read_sidecar(tmp_path, include_original_source):
    from eva_agent.pipeline import JudgeAssessment
    from eva_agent.pipeline.digests import blake3_hex
    from dataclasses import replace

    requirement = b"The synthetic task requires work/required-plan.json; no claim is accepted without it.\n"
    registry, path, _ = _fixture(tmp_path, extra_before={"input.txt": requirement},
                                extra_after={"input.txt": requirement})
    rollout = json.loads(path.read_text())
    rubric = registry.resolve("medxpertqa", "E2E")
    prepared = slime_agent_judge.prepare_workspace_rollout(
        rollout, rubric=rubric, source_path=path, judge_model_id="gpt-6-astra", judge_route_id="gpt_6_astra")
    assert not any(row.path == "work/required-plan.json" for row in prepared.evidence.workspace_after.files)
    original_ref = "workspace:before:input.txt"
    sidecar_ref = "workspace:after:.eva-agent-judge/actor/workspace-bindings.json"

    class MissingArtifactJudge(_ProviderFreeWorkspaceJudge):
        def judge(self, request, rubric):
            assessment = super().judge(request, rubric)
            assert original_ref in assessment.agent_trace.inspected_evidence_refs
            assert sidecar_ref in assessment.agent_trace.inspected_evidence_refs
            refs = (original_ref, sidecar_ref) if include_original_source else (sidecar_ref,)
            scores = tuple(replace(score, score=0.0, evidence_refs=refs,
                rationale="The inspected original input requires an output absent from the inspected workspace inventory.")
                for score in assessment.item_scores)
            zero = rubric.score({score.item_id: 0 for score in scores})
            core = {name: getattr(assessment, name) for name in assessment.__dataclass_fields__
                    if name != "assessment_blake3"}
            core.update(item_scores=scores, hard_gates_passed=zero.hard_gate_passed,
                        summary="Synthetic missing required artifact; no output citation was invented.")
            return JudgeAssessment(**core, assessment_blake3=blake3_hex(core))

    if not include_original_source:
        with pytest.raises(AgentJudgeSelectionError, match="must cite inspected workspace evidence"):
            slime_agent_judge.grade_prepared_rollout(prepared, judge=MissingArtifactJudge(), output_root=tmp_path)
        assert not (tmp_path / "score.json").exists()
    else:
        grade = slime_agent_judge.grade_prepared_rollout(prepared, judge=MissingArtifactJudge(), output_root=tmp_path)
        assert grade["reward"] == 0.0
        assert all(refs == [original_ref] for refs in grade["evidence_by_item"].values())
        retained = json.loads((tmp_path / "assessment.json").read_text())
        assert all(row["evidence_refs"] == [original_ref, sidecar_ref] for row in retained["item_scores"])
