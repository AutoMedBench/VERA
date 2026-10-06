from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from eva_agent.pipeline import FilesystemSandbox, RandomUUIDFactory, Stage, ToolDefinition, ToolRegistry
from eva_agent.pipeline.digests import blake3_bytes
from eva_agent.training.archived_predecessors import (
    ArchivedPredecessor, load_archived_predecessors, replay_archived_predecessors)
from eva_agent.training.teacher_batch import TeacherBatchError


def fixture(tmp_path, stage="S4"):
    def put(path, value):
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        b = value if isinstance(value, bytes) else json.dumps(value).encode()
        p.write_bytes(b)
        return {"path": path, "sha256": hashlib.sha256(b).hexdigest()}
    evidence = {}
    for prior in (("S3",) if stage == "S4" else ("S3", "S4")):
        code, artifact = b"print('fixture')\n", b'{"fixture":true}'
        base = f"executions/{prior}"
        put(base + "/submission/solution.py", code)
        put(base + "/workspace/work/result.json", artifact)
        ref = put(base + "/host-receipt.json", {"payload": {
            "sandbox_id": "original", "episode_id": "old-episode",
            "observations": {"execution": {"stage": prior, "gate_passed": True,
                "exit_code": 0, "process_started": True},
                "contract": {"submission_sha256": hashlib.sha256(code).hexdigest()},
                "artifact": {"host_reopened": True, "reopen_sha256_match": True,
                    "expected_path": "work/result.json", "byte_count": len(artifact),
                    "sha256": hashlib.sha256(artifact).hexdigest()}}}})
        evidence[prior.lower() + "_execution_attempts"] = [ref]
    rollout = put("rollout.json", {"sandbox_id": "original", "episode_id": "old-episode",
        "status": "completed", "infrastructure_healthy": True, "stage_evidence": evidence})
    put("manifest.json", {"sandbox_id": "original", "focus": stage,
        "production_construction_eligible": True,
        "artifacts": {"construction_solver_rollout_receipt": rollout}})
    return SimpleNamespace(stage=Stage(stage), construction_root=tmp_path,
        construction_manifest_path=tmp_path / "manifest.json", source_candidate_id="original",
        construction_manifest_blake3=blake3_bytes((tmp_path / "manifest.json").read_bytes()))


@pytest.mark.parametrize("stage,expected", [("S4", ("S3",)), ("S5", ("S3", "S4"))])
def test_only_prior_successful_stages_are_reopened(tmp_path, stage, expected):
    binding = fixture(tmp_path, stage)
    calls = []
    def verify(r, trust):
        calls.append(r)
        assert trust == {"trusted": "fixture"}
        return r["payload"]
    rows = load_archived_predecessors(binding, verify_host_receipt=verify, trust_store={"trusted": "fixture"})
    assert tuple(r.stage for r in rows) == expected
    assert len(calls) == len(expected)
    assert all(r.provenance["new_execution_verified"] is False for r in rows)
    assert "print" not in repr(rows)


@pytest.mark.parametrize("tamper", ["code", "artifact", "signature"])
def test_tampered_prerequisites_fail_before_replay(tmp_path, tamper):
    binding = fixture(tmp_path)
    if tamper != "signature":
        path = ("submission/solution.py" if tamper == "code" else "workspace/work/result.json")
        (tmp_path / "executions/S3" / path).write_bytes(b"changed")
    def verify(r, trust):
        if tamper == "signature":
            raise TeacherBatchError("signature")
        return r["payload"]
    with pytest.raises(TeacherBatchError):
        load_archived_predecessors(binding, verify_host_receipt=verify, trust_store={})


def test_replay_uses_original_tool_and_keeps_target_stage(tmp_path):
    calls = []
    def execute(workspace, args):
        calls.append(dict(args))
        return {"stage": args["stage"], "gate_passed": True, "exit_code": 0,
                "active_episode_id": "current", "failed_check_ids": []}
    code = "print('unchanged')"
    rows = (ArchivedPredecessor("S3", code, {"code_blake3": blake3_bytes(code.encode()),
        "code_path": "/host-only/construction-private/source.py", "artifact_path": "/host-only/result.json"}),)
    ctx = SimpleNamespace(episode=SimpleNamespace(stage=Stage.S4,
        policy_context={"execution_binding": {"active_episode_id": "current"}}),
        archived_predecessors=rows, tool_registry=ToolRegistry([ToolDefinition(
            name="execute_code", description="original", input_schema={"type": "object"}, handler=execute)]))
    workspace = FilesystemSandbox(tmp_path, "f5cf70bc-148a-4d05-9c84-020b44e48b16", {})
    out = replay_archived_predecessors(context=ctx, workspace=workspace, id_factory=RandomUUIDFactory())
    assert calls == [{"stage": "S3", "code": code}]
    assert out["focus"] == "S4" and not out["archived_receipts_imported"]
    assert out["actor_tokens_generated"] == 0 and not out["actor_reward_emitted"]
    assert "host-only" not in json.dumps(out) and "construction-private" not in json.dumps(out)


def test_replay_accepts_canonical_gate_with_advisory_failed_checks(tmp_path):
    def execute(workspace, args):
        return {"stage": args["stage"], "gate_passed": True, "exit_code": 0,
                "active_episode_id": "current",
                "failed_check_ids": ["hypotheses_exact", "source_units_exact"]}

    code = "print('unchanged')"
    rows = (ArchivedPredecessor("S3", code, {
        "code_blake3": blake3_bytes(code.encode())}),)
    ctx = SimpleNamespace(episode=SimpleNamespace(stage=Stage.S4,
        policy_context={"execution_binding": {"active_episode_id": "current"}}),
        archived_predecessors=rows, tool_registry=ToolRegistry([ToolDefinition(
            name="execute_code", description="original",
            input_schema={"type": "object"}, handler=execute)]))
    workspace = FilesystemSandbox(
        tmp_path, "74990867-e242-4877-8fa1-6d3f31060936", {})

    result = replay_archived_predecessors(
        context=ctx, workspace=workspace, id_factory=RandomUUIDFactory())

    assert result["tool_results"][0]["output"]["gate_passed"] is True
    assert result["tool_results"][0]["output"]["failed_check_ids"] == [
        "hypotheses_exact", "source_units_exact"]


@pytest.mark.parametrize("failure", ["gate", "execution", "episode"])
def test_replay_still_rejects_noncanonical_execution_outcomes(tmp_path, failure):
    def execute(workspace, args):
        if failure == "execution":
            raise RuntimeError("fixture execution failure")
        return {"stage": args["stage"], "gate_passed": failure != "gate",
                "exit_code": 0,
                "active_episode_id": "different" if failure == "episode" else "current",
                "failed_check_ids": []}

    code = "print('unchanged')"
    rows = (ArchivedPredecessor("S3", code, {
        "code_blake3": blake3_bytes(code.encode())}),)
    ctx = SimpleNamespace(episode=SimpleNamespace(stage=Stage.S4,
        policy_context={"execution_binding": {"active_episode_id": "current"}}),
        archived_predecessors=rows, tool_registry=ToolRegistry([ToolDefinition(
            name="execute_code", description="original",
            input_schema={"type": "object"}, handler=execute)]))
    workspace = FilesystemSandbox(
        tmp_path, {
            "gate": "ad9a57a7-d974-4a5d-89c0-52fd9089cd76",
            "execution": "c252ef18-ae15-488a-a12c-a4476a4dc2ee",
            "episode": "b191029c-64b9-45d5-849f-d7dc1e67bdf2",
        }[failure], {})

    with pytest.raises(TeacherBatchError, match="predecessor replay host gate failed"):
        replay_archived_predecessors(
            context=ctx, workspace=workspace, id_factory=RandomUUIDFactory())


def test_s5_public_view_is_exact_current_s4_file_not_archived_solution(tmp_path):
    public = b'{"public_result":"current"}'
    calls = []
    def execute(workspace, args):
        calls.append(args["stage"])
        if args["stage"] == "S4":
            workspace.write_bytes("work/result.json", public, create_only=True)
        return {"stage": args["stage"], "gate_passed": True, "exit_code": 0,
                "active_episode_id": "current", "failed_check_ids": []}
    code = "print('unchanged')"
    rows = tuple(ArchivedPredecessor(s, code, {"code_blake3": blake3_bytes(code.encode())})
                 for s in ("S3", "S4"))
    ctx = SimpleNamespace(episode=SimpleNamespace(stage=Stage.S5, policy_context={
        "execution_binding": {"active_episode_id": "current", "execution_stages": {
            "S4": {"artifact_relative_path": "work/result.json"}}}}),
        archived_predecessors=rows, tool_registry=ToolRegistry([ToolDefinition(
            name="execute_code", description="original", input_schema={"type": "object"}, handler=execute)]))
    workspace = FilesystemSandbox(tmp_path, "eca690d5-53a1-4e59-9607-82a8ea074c58", {})
    out = replay_archived_predecessors(context=ctx, workspace=workspace, id_factory=RandomUUIDFactory())
    assert calls == ["S3", "S4"] and out["focus"] == "S5"
    assert out["public_artifacts"][0]["content"].encode() == public
    assert out["public_artifacts"][0]["content_blake3"] == blake3_bytes(public)
    assert code not in json.dumps(out)
