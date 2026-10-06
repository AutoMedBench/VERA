"""Provider-free fake Codex notifications; never medical/model evidence."""
import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest

from eva_agent.codex_runtime import CodexRole, CodexRuntime, CodexSandbox, CodexThreadOptions, CodexToolOffer, CodexTurnInput
from eva_agent.codex_runtime.runtime import _logical_input
from eva_agent.pipeline.digests import blake3_bytes, canonical_json_bytes, canonical_value
from training.automedbench_lite.adapter import write_once, file_digest
from training.automedbench_lite.track_actor import snapshot
from training.automedbench_lite.track_tools import TrackTools, TOOLS
from training.automedbench_lite.track_feedback import prepare_track_feedback, judge_track_once, track_snapshot
from training.automedbench_lite import track_feedback
from test_codex_runtime import _FakeBackend, _turn_event


def fixture(tmp_path, failed=False, host_error=False, final_text="Observed public fixture", interrupted=False):
    run = tmp_path / "run"
    workspace = run / "actors" / str(uuid4())
    for directory in ("inputs/CASE", "notes", "code", "outputs/agents_outputs", "public-guidance"):
        (workspace / directory).mkdir(parents=True)
    (workspace / "inputs/CASE/image.jpg").write_bytes(b"public fixture")
    write_once(workspace / "task.json", {"track": "classification", "case_ids": ["CASE"]})
    write_once(workspace / "inputs-manifest.json", {"case_ids": ["CASE"], "files": [{"path": "inputs/CASE/image.jpg",
        "bytes": 14, "blake3": blake3_bytes(b"public fixture")}]})
    write_once(run / "track-run-manifest.json", {"run_id": str(uuid4()), "tracks": [{"track": "classification",
        "workspace_relative": workspace.relative_to(run).as_posix(), "input_manifest_file_blake3": file_digest(workspace / "inputs-manifest.json")}]})
    audit = run / "track-rollouts/classification"
    turn = audit / "turns/01-planning"
    turn.mkdir(parents=True)
    identity = {"exact_final_model_path": "/fixture/model"}
    identity_path = tmp_path / "identity.json"
    identity_path.write_bytes(canonical_json_bytes(identity))
    identity_digest = file_digest(identity_path)
    write_once(run / "track-rollouts/attempt.json", {"verified_skill_catalog_blake3": "b" * 64,
        "server_binding": {"identity_file_blake3": identity_digest, "identity": identity,
            "canary": {"checkpoint_identity_blake3": identity_digest, "exact_final_model_path": identity["exact_final_model_path"],
                       "model_info": {"model_path": identity["exact_final_model_path"]}}, "actual_process_argv_rechecked": True}})
    tools = TrackTools(workspace=workspace, audit_root=audit, image="fixture", skills=None,
        model_manifest=tmp_path / "unused", public_python=Path("/unused/python"))
    snapshot(tools.inventory, turn / "before")
    arguments = {"path": "unavailable.txt" if host_error else "task.json"}
    response = tools.call("automed_read_file", arguments, 1)
    if host_error:
        response.pop("isError")
        response["_meta"] = None
    snapshot(tools.inventory, turn / "after")
    offer = CodexToolOffer(fully_qualified_name="automed_eval/automed_read_file", description=TOOLS[0]["description"],
        input_schema=TOOLS[0]["inputSchema"], parallel_safe=False, read_only=True, allowed_stages=("E2E",))
    options = CodexThreadOptions(role=CodexRole.STRONG_ACTOR, model="fixture-Qwen", provider="fixture",
        cwd=str(workspace), sandbox=CodexSandbox.READ_ONLY, offered_tools=(offer,), base_instructions="Fixture", developer_instructions="Fixture")
    value = CodexTurnInput(public_text="Inspect public fixture")
    write_once(turn / "request.json", {"logical_input": canonical_value(_logical_input(options, value)),
        "base_instructions": "Fixture", "developer_instructions": "Fixture", "public_tool_catalog": [TOOLS[0]],
        "verified_skill_catalog_blake3": "b" * 64})
    def script(thread_id, turn_id):
        item = {"id": "fixture-call", "type": "mcpToolCall", "server": "automed_eval", "tool": "automed_read_file",
                "arguments": arguments, "status": "inProgress"}
        return (_turn_event("turn/started", thread_id, turn_id, turn={"id": turn_id, "status": "inProgress"}),
            _turn_event("item/started", thread_id, turn_id, item=item),
            _turn_event("item/completed", thread_id, turn_id, item={**item, "status": "failed" if host_error else "completed", "result": response}),
            _turn_event("item/completed", thread_id, turn_id, item={"id": "answer", "type": "agentMessage", "phase": "final_answer", "text": final_text}),
            _turn_event("turn/completed", thread_id, turn_id, turn={"id": turn_id, "status": "interrupted" if interrupted else "failed" if failed else "completed", "items": []}))
    async def generate():
        async with CodexRuntime(_FakeBackend(script)) as runtime:
            handle = await runtime.start_thread(options)
            return await runtime.run_turn(handle, value)
    receipt = asyncio.run(generate())
    (turn / "receipt.json").write_bytes(canonical_json_bytes(receipt))
    return dict(run_root=run, checkpoint_identity=identity_path, output_root=tmp_path / "feedback", stage="S1")


def test_mutable_track_projection_keeps_actual_calls_and_separate_inputs(tmp_path):
    args = fixture(tmp_path)
    rollout, rubric, report = prepare_track_feedback(**args)
    assert report["valid"] and report["judge_eligible"] and report["provider_calls"] == 0
    assert report["actual_host_call_count"] == 1
    assert rubric.domain == "automedbench-classification" and rubric.stage == "S1"
    assert not rollout["provider_metadata"]["all_immutable_input_bytes_present_in_judge_snapshot"]
    assert rollout["tool_trace"]["results"][0]["output"]["host_event"]["schema"] == "eva.automedbench-track-tool-event.v1"


def test_failed_turn_retained_but_never_scored_as_zero(tmp_path):
    args = fixture(tmp_path, failed=True)
    _, _, report = prepare_track_feedback(**args)
    assert report["actual_turn_statuses"] == ["failed"] and not report["judge_eligible"]
    with pytest.raises(ValueError, match="not_gradeable"): judge_track_once(args["output_root"])
    assert not (args["output_root"] / "judge-attempt.json").exists()


def test_archival_hardlinks_validated_by_actual_bytes(tmp_path):
    args = fixture(tmp_path)
    target = args["run_root"] / "track-rollouts/classification/turns/01-planning/before"
    assert (target / "files/task.json").stat().st_nlink > 1
    value, _ = track_snapshot(target, "before")
    assert value.file_count == 2
    (target / "files/task.json").chmod(0o600)
    (target / "files/task.json").write_bytes(b"tampered")
    with pytest.raises(ValueError): track_snapshot(target, "before")


def test_actual_failed_domain_call_not_rewritten_success(tmp_path):
    args = fixture(tmp_path, host_error=True)
    rollout, _, report = prepare_track_feedback(**args)
    assert report["judge_eligible"]  # Complete model turn; domain failure is observed evidence.
    result = rollout["tool_trace"]["results"][0]
    assert result["status"] == "failed" and result["output"]["host_event"]["is_error"]
    assert result["error_code"] == "public_text_path_not_allowed"


def test_complete_s1_and_absent_later_phases_grade_only_s1(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    calls = []
    def judge(root):
        calls.append(root.name)
        return {"valid": True, "status": "scored"}
    monkeypatch.setattr(track_feedback, "judge_track_once", judge)
    monkeypatch.setattr(track_feedback, "verify_feedback", lambda root: {
        "valid": True, "score": {"reward_bps": 1234}, "source_rollout_blake3": "f" * 64})
    roots = track_feedback.evaluate_track_feedback(args["run_root"], args["checkpoint_identity"], args["output_root"])
    assert calls == ["S1"] and roots == (args["output_root"] / "S1",)
    coverage = json.loads((args["output_root"] / "coverage.json").read_text())
    assert [row["status"] for row in coverage["stages"]] == ["independently_verified", "unknown", "unknown"]
    assert all(row["rubric_score"] is None for row in coverage["stages"][1:])


def test_corrupt_present_s1_does_not_become_unknown(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    request = args["run_root"] / "track-rollouts/classification/turns/01-planning/request.json"
    request.write_text("{}")
    calls = []
    monkeypatch.setattr(track_feedback, "judge_track_once", lambda root: calls.append(root))
    with pytest.raises(ValueError):
        track_feedback.evaluate_track_feedback(args["run_root"], args["checkpoint_identity"], args["output_root"])
    assert not calls


def test_completed_empty_text_remains_gradeable_from_actual_workspace(tmp_path):
    args = fixture(tmp_path, final_text="")
    rollout, _, report = prepare_track_feedback(**args)
    assert report["judge_eligible"] and rollout["assistant_output"] == ""
    assert report["actual_host_call_count"] == 1


def synthetic_budget_binding(args, *, infrastructure_error=None):
    """CPU fixture only; never written into an actual model run."""
    from eva_agent.codex_runtime import codex_turn_receipt_from_document
    from training.automedbench_lite.adapter import read_document
    from training.automedbench_lite.policy_capture import require_joined_host_results
    audit = args["run_root"] / "track-rollouts/classification"
    target = audit / "turns/01-planning"
    receipt = codex_turn_receipt_from_document(json.loads((target / "receipt.json").read_bytes()))
    budget = {"schema": "eva.codex-policy-turn-budget.v1", "budget_exhausted": True,
        "turn_id": receipt.turn_id, "receipt_blake3": receipt.receipt_blake3,
        "actual_terminal_status": receipt.status, "infrastructure_error": None, "reward": None,
        "interrupt_requested_ns": 1, "interrupt_acknowledged_ns": 2, "terminal_observed_ns": 3,
        "timeout_seconds": 900, "drain_seconds": 30}
    return write_once(target / "policy-budget-terminal.json", {
        "schema": "eva.automedbench-policy-budget-terminal.v1", "phase_intent": target.name,
        "receipt_blake3": receipt.receipt_blake3, "actual_terminal_status": receipt.status,
        "workspace_quiescence_verified": True, "infrastructure_error": infrastructure_error,
        "reward": None, "later_stages_evaluated": False, "policy_budget": budget,
        "host_quiescence_observed_ns": 4,
        "after_snapshot_document_blake3": read_document(target / "after/manifest.json")["document_blake3"],
        **require_joined_host_results(receipt, audit)})


def test_explicit_verified_stop_is_partial_evidence_not_completed_stage(tmp_path):
    args = fixture(tmp_path, interrupted=True)
    proof = synthetic_budget_binding(args)
    with pytest.raises(ValueError, match="unfinished_or_private"):
        prepare_track_feedback(**args)
    rollout, _, report = prepare_track_feedback(**args, allow_policy_budget_terminal=True)
    assert report["judge_eligible"] and report["actual_turn_statuses"] == ["interrupted"]
    assert report["stage_completion_claimed"] is False
    assert report["verified_policy_terminal_proofs"][0]["terminal_document_blake3"] == proof["document_blake3"]
    assert rollout["provider_metadata"]["codex_turn_receipts"][0]["status"] == "interrupted"
    assert rollout["provider_metadata"]["budget_stop_is_not_stage_completion"] is True


def test_policy_stop_with_transport_error_cannot_be_admitted(tmp_path):
    args = fixture(tmp_path, interrupted=True)
    synthetic_budget_binding(args, infrastructure_error="local429")
    with pytest.raises(ValueError, match="binding_or_quiescence_invalid"):
        prepare_track_feedback(**args, allow_policy_budget_terminal=True)
    assert not args["output_root"].exists()


def test_context_optin_does_not_admit_generic_failed_actor_turn(tmp_path):
    args = fixture(tmp_path, failed=True)
    with pytest.raises((ValueError, KeyError)):
        prepare_track_feedback(**args, allow_context_budget_terminal=True)
    assert not (args["output_root"] / "judge-attempt.json").exists()


def test_context_proof_optin_is_separate_and_preserves_raw_failed_status(tmp_path, monkeypatch):
    from training.automedbench_lite import context_terminal
    from training.automedbench_lite.adapter import read_document
    args = fixture(tmp_path, failed=True)
    calls = []
    # Routing fixture only; exact signed/error/quiescence proof is independently
    # covered by test_context_terminal_verifier and real retained-source preflight.
    def validated(target, audit, raw_receipt, *, source_binding):
        calls.append((target, source_binding))
        return {"valid": True, "actual_terminal_status": "failed", "numeric_prompt_tokens": None,
            "numeric_guard_record_joined": False, "reward": None,
            "before_manifest_blake3": read_document(target / "before/manifest.json")["document_blake3"],
            "after_manifest_blake3": read_document(target / "after/manifest.json")["document_blake3"]}
    monkeypatch.setattr(context_terminal, "verify_context_terminal", validated)
    rollout, _, report = prepare_track_feedback(**args, allow_context_budget_terminal=True,
                                                context_terminal_source_binding={"fixture": "source"})
    assert calls[0][1] == {"fixture": "source"}
    assert report["judge_eligible"] and report["actual_turn_statuses"] == ["failed"]
    assert "verified_policy_terminal_proofs" not in report
    assert report["verified_context_terminal_proofs"][0]["numeric_prompt_tokens"] is None
    assert rollout["provider_metadata"]["codex_turn_receipts"][0]["status"] == "failed"
    assert rollout["provider_metadata"]["context_budget_stop_is_not_stage_completion"]


def test_new_material_view_is_explicitly_bound_and_uses_actual_imported_judge_source(tmp_path, monkeypatch):
    from eva_agent.training import slime_agent_judge as selected_source
    from training.benchmark_feedback import automed_codex as bridge
    args = fixture(tmp_path)
    rollout, _, preflight = prepare_track_feedback(**args, material_view="policy-visible-audit-v2")
    assert preflight["judge_material_view"] == "policy-visible-audit-v2"
    assert "host_event" in rollout["messages"][3]["content"]["output"]
    calls = []
    async def fixture_service_unavailable(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("CPU fixture; no service invoked")
    monkeypatch.setattr(bridge, "grade_rollout", fixture_service_unavailable)
    with pytest.raises(RuntimeError, match="CPU fixture"):
        judge_track_once(args["output_root"], native_turn_timeout_seconds=600)
    assert calls[0]["material_view"] == "policy-visible-audit-v2"
    attempt = json.loads((args["output_root"] / "judge-attempt.json").read_bytes())
    assert attempt["actual_imported_judge_source"] == str(Path(selected_source.__file__).resolve())
    assert attempt["judge_implementation_blake3"] == file_digest(Path(selected_source.__file__))
    assert attempt["material_view"] == "policy-visible-audit-v2"
    assert not (args["output_root"] / "feedback.json").exists()


def test_helper_evidence_is_readable_without_rewriting_actor_workspace_or_host_response(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    audit = args["run_root"] / "track-rollouts/classification"
    host = json.loads((audit / "mcp-events.jsonl").read_text())
    source = "# Provided CPU fixture, not actor code\n"
    supplied = {"job_id": str(uuid4()), "host_event_id": host["event_id"], "authored_by": "provided_analysis_tool",
        "sources": [{"path": "executed-helper.py", "blake3": blake3_bytes(source.encode()), "text": source}]}
    monkeypatch.setattr(track_feedback, "completed_model_evidence", lambda **kwargs: [supplied])
    rollout, _, report = prepare_track_feedback(**args)
    result = rollout["tool_trace"]["results"][0]
    supplemental = result["output"]["supplemental_host_verified_provided_analysis_tool"]
    assert supplemental["not_actor_authored_code"] and supplemental["not_original_tool_response_content"]
    assert result["output"]["host_event"] == host
    assert [row["path"] for row in rollout["workspace_after"]["files"]] == ["inputs-manifest.json", "task.json"]
    assert not any("executed-helper" in row["path"] for row in rollout["workspace_before"]["files"])
    phase_sources = rollout["provider_metadata"]["source_turns"]
    assert len(phase_sources) == 1 and phase_sources[0]["phase_intent"] == "01-planning"
    assert phase_sources[0]["stage_intent"] == "S1" and "codex_receipt_blake3" in phase_sources[0]
    assert report["judge_eligible"]


def test_all_five_actual_phase_slots_route_to_requested_track(tmp_path, monkeypatch):
    from types import SimpleNamespace
    run, output = tmp_path / "run", tmp_path / "feedback"
    stages = tuple(track_feedback.PHASES)
    for phase in track_feedback.PHASES.values():
        target = run / "track-rollouts/vqa/turns" / phase
        target.mkdir(parents=True)
        (target / "receipt.json").write_text("CPU mock only")
    monkeypatch.setattr(track_feedback, "load_and_compile_registry", lambda path: SimpleNamespace(
        rubrics=[SimpleNamespace(domain="automedbench-vqa", stage=stage) for stage in stages]))
    calls = []
    def prepare(**kwargs):
        calls.append((kwargs["track"], kwargs["stage"]))
        kwargs["output_root"].mkdir()
        return None, None, {"judge_eligible": True}
    monkeypatch.setattr(track_feedback, "prepare_track_feedback", prepare)
    monkeypatch.setattr(track_feedback, "judge_track_once", lambda root: {"fixture": True})
    monkeypatch.setattr(track_feedback, "verify_feedback", lambda root: {"score": {"fixture": True}, "source_rollout_blake3": "f" * 64})
    roots = track_feedback.evaluate_track_feedback(run, tmp_path / "identity", output, stages=stages, track="vqa")
    assert calls == [("vqa", stage) for stage in stages] and len(roots) == 5


def test_sdk_source_binding_forwards_without_enabling_context_terminal(tmp_path, monkeypatch):
    from types import SimpleNamespace
    run, output = tmp_path / "run", tmp_path / "feedback"
    target = run / "track-rollouts/vqa/turns/01-planning"
    target.mkdir(parents=True)
    (target / "receipt.json").write_text("CPU mock only")
    monkeypatch.setattr(track_feedback, "load_and_compile_registry", lambda path: SimpleNamespace(
        rubrics=[SimpleNamespace(domain="automedbench-vqa", stage="S1")]))
    calls = []
    def prepare(**kwargs):
        calls.append(kwargs)
        kwargs["output_root"].mkdir()
        return None, None, {"judge_eligible": True}
    monkeypatch.setattr(track_feedback, "prepare_track_feedback", prepare)
    monkeypatch.setattr(track_feedback, "judge_track_once", lambda root: {"fixture": True})
    monkeypatch.setattr(track_feedback, "verify_feedback", lambda root: {
        "score": {"fixture": True}, "source_rollout_blake3": "f" * 64})
    source = {"schema": "eva.rsi-evaluation-actor-binding.v1"}
    track_feedback.evaluate_track_feedback(run, tmp_path / "identity", output, stages=("S1",), track="vqa",
        allow_context_budget_terminal=False, context_terminal_source_binding=source)
    assert calls[0]["context_terminal_source_binding"] is source
    assert "allow_context_budget_terminal" not in calls[0]


def test_missing_exact_rubric_is_unknown_not_another_domains_score(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(track_feedback, "load_and_compile_registry", lambda path: SimpleNamespace(rubrics=[]))
    monkeypatch.setattr(track_feedback, "judge_track_once", lambda root: pytest.fail("No rubric means no Judge call"))
    output = tmp_path / "feedback"
    with pytest.raises(ValueError, match="no_verified_grade"):
        track_feedback.evaluate_track_feedback(tmp_path / "run", tmp_path / "identity", output, stages=("S1",), track="vqa")
    coverage = json.loads((output / "coverage.json").read_text())
    assert coverage["stages"][0]["reason"] == "exact_compiled_track_stage_rubric_absent"
    assert coverage["stages"][0]["rubric_score"] is None


def test_explicit_registry_is_reopened_by_commitment_and_old_default_unchanged(tmp_path):
    from training.benchmark_feedback.automed_codex import _bound_feedback_rubric
    args = fixture(tmp_path)
    source = track_feedback.ROOT / "rubrics/source/domain-stage-tables.v1.json"
    registry = tmp_path / "explicit-registry.json"
    registry.write_bytes(source.read_bytes())
    rollout, rubric, report = prepare_track_feedback(**args, registry_path=registry)
    assert report["exact_rubric_registry"]["path"] == str(registry)
    assert _bound_feedback_rubric(report, rollout).digest == rubric.digest
    with pytest.raises(ValueError, match="registry_path_differs"):
        _bound_feedback_rubric(report, rollout, source)
    legacy_report = {key: value for key, value in report.items() if key != "exact_rubric_registry"}
    assert _bound_feedback_rubric(legacy_report, rollout).digest == rubric.digest
    registry.write_bytes(registry.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="registry_source_changed"):
        _bound_feedback_rubric(report, rollout)


def test_full_track_snapshot_accepts_more_than_256_actual_files(tmp_path):
    from training.automedbench_lite.adapter import read_document
    args = fixture(tmp_path)
    target = args["run_root"] / "track-rollouts/classification/turns/01-planning/after"
    manifest = read_document(target / "manifest.json")
    for index in range(257):
        relative = f"notes/fixture-{index:04}.txt"
        path = target / "files" / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"x")
        manifest["files"].append({"path": relative, "bytes": 1, "blake3": blake3_bytes(b"x"), "mode": 0o400})
    manifest["files"].sort(key=lambda row: row["path"])
    manifest["file_count"] = len(manifest["files"])
    manifest["bytes"] = sum(row["bytes"] for row in manifest["files"])
    manifest.pop("document_blake3")
    (target / "manifest.json").unlink()
    write_once(target / "manifest.json", manifest)
    snapshot, _ = track_snapshot(target, "after")
    assert snapshot.file_count == 259 and snapshot.files[1].content == b"x"


def test_full_track_actor_budget_rejected_before_reading_large_payloads(tmp_path, monkeypatch):
    from training.automedbench_lite.adapter import read_document
    args = fixture(tmp_path)
    target = args["run_root"] / "track-rollouts/classification/turns/01-planning/after"
    manifest = read_document(target / "manifest.json")
    manifest["bytes"] = 8 * 1024**3 + 1
    manifest.pop("document_blake3")
    (target / "manifest.json").unlink()
    write_once(target / "manifest.json", manifest)
    monkeypatch.setattr(track_feedback, "archived_payload", lambda *args: pytest.fail("Must reject before loading bytes"))
    with pytest.raises(ValueError, match="exceeds_actor_budget"):
        track_snapshot(target, "after")


def test_only_explicit_track_rollout_gets_expanded_reader_budget(tmp_path, monkeypatch):
    from training.benchmark_feedback import automed_codex as bridge
    policy = bridge.TRACK_FEEDBACK_RESOURCE_POLICY
    calls, original = [], bridge.safe_file
    def record(root, relative, *, maximum):
        calls.append(maximum)
        return original(root, relative, maximum=maximum)
    monkeypatch.setattr(bridge, "safe_file", record)
    path = tmp_path / "rollout.json"
    value = {"schema": "eva.automedbench-track-workspace-feedback.v1", "provider_metadata": {
        "snapshot_resource_policy": policy, "exact_rubric_registry": {"fixture": True}}}
    path.write_bytes(canonical_json_bytes(value))
    preflight = {"schema": "eva.automedbench-track-feedback-preflight.v1", "snapshot_resource_policy": policy,
        "exact_rubric_registry": {"fixture": True}}
    assert bridge.read_feedback_rollout(path, preflight) == value
    bridge.read_feedback_rollout(path, {})
    bridge.read_json(path)
    assert calls == [24 * 1024**3, 128 * 1024**2, 128 * 1024**2]
    value["provider_metadata"]["snapshot_resource_policy"] = {}
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValueError, match="scope_changed"):
        bridge.read_feedback_rollout(path, preflight)
