"""Provider-free fixtures, never evidence of a real Opus attribution."""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eva_agent.pipeline.digests import blake3_hex, canonical_value
from eva_agent.training import slime_agent_judge
from training.eva_rsi import skill_attribution as attribution
from training.eva_rsi.controller import Controller, read, write
from training.eva_rsi.evidence import EvidenceError
from test_agent_judge_worker import _ProviderFreeWorkspaceJudge
from test_slime_agent_judge import _generic_rollout


def suggestion(**changes):
    return {"action": "retain", "skill_id": None, "capability": "Current catalog",
        "evidence_refs": ["workspace:before:input.txt"],
        "rationale": "Observed evidence does not establish a skill change is warranted.", **changes}


class FixtureJudge(_ProviderFreeWorkspaceJudge):
    def judge(self, request, rubric):
        self.request = request
        original = super().judge(request, rubric)
        core = {key: value for key, value in original.__dict__.items() if key != "assessment_blake3"}
        core["summary"] = json.dumps({"schema": attribution.SUMMARY_SCHEMA, "suggestions": [suggestion()]})
        return replace(original, summary=core["summary"], assessment_blake3=blake3_hex(core))


def fixture(tmp_path):
    rollout, rubric, path, model = _generic_rollout(tmp_path)
    prepared = slime_agent_judge.prepare_workspace_rollout(rollout, rubric=rubric, source_path=path,
                                                         judge_model_id=model)
    catalog = [{"skill_id": "fixture-skill", "description": "fixture only", "allowed_stages": ["E2E"]}]
    judge = FixtureJudge()
    grade = slime_agent_judge.grade_prepared_rollout(prepared,
        judge=attribution.AttributionJudge(judge, catalog), output_root=tmp_path)
    return prepared, catalog, judge, grade


def test_exact_existing_rubric_and_complete_workspace_reads_preserved(tmp_path):
    prepared, catalog, judge, grade = fixture(tmp_path)
    assert grade["rubric_digest"] == prepared.trajectory.rubric.digest
    assert grade["judge_tool_trace"]["workspace_read_count"] >= 8
    assert canonical_value(judge.request.workspace_evidence) == canonical_value(prepared.evidence)
    assert judge.request.judge_only_reference["skill_attribution"] == attribution.instruction(catalog)
    guidance = judge.request.judge_only_reference["skill_attribution"]["instruction"]
    assert "If no items define hard_gate, hard_gates_passed MUST be true" in guidance
    assert "even when all item scores are zero" in guidance
    assert "not overall task success" in guidance
    from eva_agent.pipeline.codec import _judgment
    assessment = _judgment(read(tmp_path / "assessment.json"))
    assert attribution.validate_suggestions(assessment, prepared, catalog) == [suggestion()]
    assert prepared.trajectory.document == read(prepared.task.trajectory_path)


@pytest.mark.parametrize("changes,code", [
    ({"evidence_refs": ["workspace:after:not-read.txt"]}, "uninspected"),
    ({"action": "remove", "skill_id": "invented"}, "unknown_skill"),
    ({"action": "remove", "skill_id": None}, "existing_skill"),
    ({"action": "remove", "skill_id": "fixture-skill"}, "not_inspected"),
    ({"action": "add", "skill_id": "fixture-skill"}, "not_inspected"),
    ({"action": "execute"}, "attribution_action"),
])
def test_suggestions_never_invent_or_skip_evidence(tmp_path, changes, code):
    prepared, catalog, _, _ = fixture(tmp_path)
    from eva_agent.pipeline.codec import _judgment
    assessment = _judgment(read(tmp_path / "assessment.json"))
    modified = replace(assessment, summary=json.dumps({"schema": attribution.SUMMARY_SCHEMA,
        "suggestions": [suggestion(**changes)]}))
    with pytest.raises(EvidenceError, match=code):
        attribution.validate_suggestions(modified, prepared, catalog)


def test_add_is_a_proposal_not_a_new_skill_identity(tmp_path):
    prepared, catalog, _, _ = fixture(tmp_path)
    from eva_agent.pipeline.codec import _judgment
    assessment = _judgment(read(tmp_path / "assessment.json"))
    row = suggestion(action="add", capability="A bounded evidence-reading checklist")
    modified = replace(assessment, summary=json.dumps({"schema": attribution.SUMMARY_SCHEMA, "suggestions": [row]}))
    assert attribution.validate_suggestions(modified, prepared, catalog) == [row]
    decision = attribution.retain_decision("catalog", "verified", suggestions=[row])
    assert decision["action"] == "retain_catalog" and decision["skill_change_performed"] is False


@pytest.mark.parametrize("body_valid", [True, False])
def test_individual_skill_requires_real_loaded_body_and_inspected_trace(tmp_path, body_valid):
    prepared, catalog, _, _ = fixture(tmp_path)
    from eva_agent.pipeline.codec import _judgment
    from eva_agent.pipeline.digests import blake3_bytes
    assessment = _judgment(read(tmp_path / "assessment.json"))
    body = "A real retained fixture skill body."
    result = SimpleNamespace(name="load_skill", status="completed", output={"host_event": {
        "is_error": False, "arguments": {"skill_id": "fixture-skill"}, "result": {
            "skill_id": "fixture-skill", "content": body,
            "content_blake3": blake3_hex(body) if body_valid else "0" * 64}}})
    view = SimpleNamespace(source_workspace_refs=prepared.source_workspace_refs,
                           evidence=SimpleNamespace(tool_trace=SimpleNamespace(results=[result])))
    row = suggestion(action="remove", skill_id="fixture-skill")
    modified = replace(assessment, summary=json.dumps({"schema": attribution.SUMMARY_SCHEMA, "suggestions": [row]}))
    if body_valid:
        assert attribution.validate_suggestions(modified, view, catalog) == [row]
        added = suggestion(action="add", skill_id="fixture-skill")
        modified = replace(assessment, summary=json.dumps({"schema": attribution.SUMMARY_SCHEMA, "suggestions": [added]}))
        assert attribution.validate_suggestions(modified, view, catalog) == [added]
    else:
        with pytest.raises(EvidenceError, match="not_inspected"):
            attribution.validate_suggestions(modified, view, catalog)


def test_missing_or_partial_workspace_cannot_become_verified_attribution(tmp_path):
    rollout, rubric, path, model = _generic_rollout(tmp_path)
    prepared = slime_agent_judge.prepare_workspace_rollout(rollout, rubric=rubric, source_path=path, judge_model_id=model)
    from eva_agent.training.agent_judge import AgentJudgeSelectionError
    with pytest.raises(AgentJudgeSelectionError, match="completely read"):
        slime_agent_judge.grade_prepared_rollout(prepared,
            judge=attribution.AttributionJudge(FixtureJudge(complete=False), []), output_root=tmp_path)
    assert not (tmp_path / "score.json").exists()


def test_no_real_codex_receipt_cannot_be_verified(tmp_path):
    prepared, catalog, _, _ = fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        attribution._reopen_assessment(tmp_path, prepared, catalog)


def source_stub(tmp_path):
    rollout, rubric, path, _ = _generic_rollout(tmp_path)
    return rollout, rubric, {"catalog_metadata": [], "source_rollout": {"path": str(path)},
                            "selection": "fixture-only"}


def test_real_attempted_route_failure_is_safe_retained_and_consumed(tmp_path, monkeypatch):
    import eva_agent.codex_providers as providers
    source = source_stub(tmp_path)
    context = {"attempt_root": str(tmp_path), "round": 1, "skill_catalog_id": "fixture"}
    monkeypatch.setattr(attribution, "prepare_source", lambda _: source)
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("Bearer NEVER_RETAIN_THIS private endpoint")
    monkeypatch.setattr(providers, "load_codex_provider_routes", fail)
    result = attribution.run_once(context, route_registry=tmp_path / "registry", env_files=[], codex_bin=Path(__file__))
    decision = attribution.verify_result(tmp_path, context)
    assert len(calls) == 1 and calls[0]["route_ids"] == ("opus_5",)
    assert decision["opus_attribution"] == "failed_attempt" and not decision["attribution_claimed"]
    assert result["decision"]["failure"]["category"] == "route_unavailable"
    assert "NEVER_RETAIN" not in (tmp_path / "skill-attribution/failure.json").read_text()
    with pytest.raises(FileExistsError):
        attribution.run_once(context, route_registry=tmp_path / "registry", env_files=[], codex_bin=Path(__file__))
    assert len(calls) == 1
    assert (tmp_path / "skill-attribution/result.json").stat().st_mode & 0o777 == 0o600


def test_provider_setup_failure_no_astra_fallback(tmp_path, monkeypatch):
    import eva_agent.codex_providers as providers
    source = source_stub(tmp_path)
    context = {"attempt_root": str(tmp_path), "round": 1, "skill_catalog_id": "fixture"}
    monkeypatch.setattr(attribution, "prepare_source", lambda _: source)
    monkeypatch.setattr(providers, "load_codex_provider_routes", lambda **_: {
        "opus_5": SimpleNamespace(model_id="claude-opus-5")})
    calls = []
    @contextmanager
    def failed(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("private provider response")
        yield
    monkeypatch.setattr(attribution, "open_opus", failed)
    result = attribution.run_once(context, route_registry=tmp_path / "registry", env_files=[], codex_bin=Path(__file__))
    assert len(calls) == 1 and result["decision"]["failure"]["category"] == "provider_or_transport_failure"
    assert result["decision"]["suggestions"] == []
    assert not (tmp_path / "skill-attribution/assessment.json").exists()


def test_opus_route_resolves_reference_across_explicit_env_sources(tmp_path):
    """Exercise the actual loader only; no provider or attribution is started."""
    from eva_agent.codex_providers.routes import (
        CodexProviderConfigurationError, load_codex_provider_routes)

    source = tmp_path / ".env"
    aliases = tmp_path / "keys.env"
    source.write_text("NVIDIA_API_KEY_CAN=fixture-not-a-real-key\n"
                      "NVIDIA_INFERENCE_BASE_URL=https://provider.invalid/v1\n")
    aliases.write_text("NVIDIA_INFERENCE_API_KEY=${NVIDIA_API_KEY_CAN}\n"
                       "MODEL_OPUS_5=aws/anthropic/bedrock-claude-opus-5\n")
    with pytest.raises(CodexProviderConfigurationError, match="provider credential is missing"):
        load_codex_provider_routes(env_files=(aliases,), environment={}, route_ids=("opus_5",))
    routes = load_codex_provider_routes(env_files=(source, aliases), environment={},
                                       route_ids=("opus_5",))
    assert set(routes) == {"opus_5"}
    route = routes["opus_5"]
    assert route.model_id == "aws/anthropic/bedrock-claude-opus-5"
    assert route.config.credential_env_name == "NVIDIA_INFERENCE_API_KEY"
    assert route.config.endpoint_env_name == "NVIDIA_INFERENCE_BASE_URL"
    assert "fixture-not-a-real-key" not in repr(route)


def test_opus_mcp_resolves_venv_python_before_real_factory(tmp_path, monkeypatch):
    """Real MCP preflight with a symlinked venv entry; stop before any gateway."""
    import sys
    import eva_agent.codex_pipeline as pipeline
    import eva_agent.codex_providers as providers

    executable = Path(sys.executable).resolve(strict=True)
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(executable)
    monkeypatch.setattr(sys, "executable", str(venv_python))
    real_factory = pipeline.TurnMCPBridgeFactory
    observed = []

    def capture_factory(**kwargs):
        factory = real_factory(**kwargs)
        observed.append(factory.proxy_python)
        return factory

    class StopBeforeProvider(Exception):
        pass

    def stop_gateway(*args, **kwargs):
        raise StopBeforeProvider

    monkeypatch.setattr(pipeline, "TurnMCPBridgeFactory", capture_factory)
    monkeypatch.setattr(providers, "ResponsesAdapterGateway", stop_gateway)
    with pytest.raises(StopBeforeProvider):
        with attribution.open_opus(None, SimpleNamespace(), tmp_path / "output", codex_bin=executable):
            pytest.fail("provider boundary should not be crossed")
    assert observed == [executable]


def test_invalid_raw_http_status_is_not_treated_as_signed_proof(tmp_path):
    path = tmp_path / "adapter-receipts/task/opus_5/fake.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"payload": {"upstream_http_status": 429}}))
    with pytest.raises((KeyError, ValueError)):
        attribution.adapter_statuses(tmp_path)


def controller_fixture(tmp_path, *, command=True, attribution_verifier=None):
    config = {"schema": "eva.rsi-loop-config.v1", "rounds": 10, "updates_per_round": 50,
        "cwd": str(tmp_path), "architecture_model_path": "original", "initial_model_path": "model",
        "initial_checkpoint_root": "dcp", "skill_catalog_id": "catalog",
        "commands": {key: ["fixture", "{context}"] for key in ("baseline_eval", "train", "evaluation")}}
    if command:
        config["commands"]["skill_attribution"] = ["fixture-attribution", "{context}"]
    write(tmp_path / "config.json", config)
    original = Controller.initialize(tmp_path / "loop", tmp_path / "config.json")
    controller = Controller(original.root, attribution_verifier=attribution_verifier)
    state, _ = controller.load()
    state.update(phase="skill_decision", evaluation="evaluation.json")
    state["rounds"] = [{"round": 1, "durable_updates": 50, "durable_learning_updates": 0,
        "post_evaluation": "evaluation.json", "complete": False}]
    controller.save(state)
    return controller, state


def test_controller_default_says_not_requested_not_unavailable(tmp_path):
    controller, state = controller_fixture(tmp_path, command=False)
    controller.advance_internal(state)
    state, _ = controller.load()
    assert state["rounds"][0]["skill_decision"]["opus_attribution"] == "not_requested"
    assert state["rounds"][0]["complete"] is True and state["round"] == 2


@pytest.mark.parametrize("with_result", [False, True])
def test_optional_attribution_is_one_distinct_attempt_and_never_retried(tmp_path, with_result):
    decision = attribution.retain_decision("catalog", "failed_attempt")
    controller, state = controller_fixture(tmp_path, attribution_verifier=lambda *_: decision)
    controller.advance_internal(state)
    assert state["phase"] == "skill_attribution" and state["rounds"][0]["complete"] is False
    root = tmp_path / "owned-attempt"
    root.mkdir()
    write(root / "context.json", {"skill_catalog_id": "catalog"})
    write(root / "exit.json", {"attempt_id": "actual-fixture-attempt", "exit_code": 1, "error_type": None})
    if with_result:
        (root / "skill-attribution").mkdir()
        write(root / "skill-attribution/result.json", {"fixture": True})
    state["active_attempt"] = "actual-fixture-attempt"
    state["attempts"] = [{"attempt_id": "actual-fixture-attempt", "root": str(root), "phase": "skill_attribution"}]
    controller.save(state)
    assert controller.finish_attempt(state)
    assert state["phase"] == "skill_decision" and state["active_attempt"] is None
    assert state["rounds"][0]["skill_decision"]["action"] == "retain_catalog"
    controller.advance_internal(state)
    assert state["rounds"][0]["complete"] is True and state["phase"] == "target_selection"
    assert len(state["attempts"]) == 1 and state["skill_catalog_id"] == "catalog"
