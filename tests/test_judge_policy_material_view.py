"""Provider-free lossless material projection and real read-tool coverage tests."""
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from eva_agent.pipeline import ToolCall, RandomUUIDFactory
from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes, canonical_value
from eva_agent.pipeline.judge_material_view import AUDIT_PATHS, REQUIRED_PATHS, POLICY_VISIBLE_VIEW, material_layout
from eva_agent.pipeline.judge_workspace import JudgeWorkspaceTools
from eva_agent.codex_pipeline import codex_judge_evidence_index
from eva_agent.training.agent_judge_worker import _fast_workspace_read_coverage
from eva_agent.training.slime_agent_judge import prepare_workspace_rollout
from test_agent_judge_worker import _fixture, _event


def fixture(tmp_path, *, unknown=False):
    registry, path, _ = _fixture(tmp_path)
    source = json.loads(path.read_bytes())
    call_id = str(uuid4())
    actual = {"content": [{"type": "text", "text": "Actual public observation\n" * 25}],
              "structuredContent": {"event_id": str(uuid4()), "result": {"skill_text": "Exact loaded skill body"}}}
    output = {"actual_mcp_response_text_and_binary_commitments": actual,
              "host_event": {"arguments": {"path": "input.txt"}, "inventory": "host-only" * 20000},
              "supplemental_host_verified_provided_analysis_tool": {"source": "provided-code" * 10000}}
    if unknown:
        output["unknown_actual_observation"] = "Must not disappear"
    core = {"result_id": str(uuid4()), "call_id": call_id, "name": "automed_read_file", "frontier": 0,
        "parallel_group_id": str(uuid4()), "status": "completed", "output": output, "error_code": None,
        "workspace_before_blake3": "a" * 64, "workspace_after_blake3": "b" * 64}
    result = {**core, "receipt_blake3": blake3_hex(core)}
    source["messages"].insert(2, _event("assistant", {"name": "automed_read_file", "arguments": {"path": "input.txt"}}, (call_id,)))
    source["messages"].insert(3, _event("tool", result, (call_id,)))
    trace = {"results": [result], "declared_call_ids": [call_id], "joined_call_ids": [call_id],
             "frontier_count": 1, "max_parallelism_observed": 1, "retry_count": 0}
    source["tool_trace"] = {**trace, "trace_blake3": blake3_hex(trace)}
    source["schema"] = "eva.automedbench-track-workspace-feedback.v1"
    return source, registry.resolve("medxpertqa", "E2E"), path


def prepare(args, view=POLICY_VISIBLE_VIEW):
    source, rubric, path = args
    return prepare_workspace_rollout(source, rubric=rubric, source_path=path,
        judge_model_id="gpt-6-astra", judge_route_id="gpt_6_astra", material_view=view)


def test_every_visible_byte_and_original_object_survives_with_deduplicated_host_audit(tmp_path):
    args = fixture(tmp_path)
    source = canonical_json_bytes(args[0])
    prepared, old = prepare(args), prepare(args, "historical-v1")
    evidence = prepared.evidence
    assert canonical_value(evidence.policy_events) == args[0]["messages"]
    assert canonical_value(evidence.tool_trace) == args[0]["tool_trace"]
    assert canonical_json_bytes(args[0]) == source
    files = {row.path: row for row in evidence.workspace_after.files}
    assert files[AUDIT_PATHS[0]].content == canonical_json_bytes(args[0]["messages"])
    assert files[AUDIT_PATHS[1]].content == canonical_json_bytes(args[0]["tool_trace"])
    view = json.loads(files[REQUIRED_PATHS[2]].content)["events"]
    for original, projected in zip(args[0]["messages"], view):
        if original["role"] != "tool":
            assert original == projected
        else:
            assert projected["content"]["output"] == {
                "actual_mcp_response_text_and_binary_commitments": original["content"]["output"]["actual_mcp_response_text_and_binary_commitments"]}
            assert projected["source_event_blake3"] == original["event_blake3"]
    assert JudgeWorkspaceTools(evidence).response_api_schemas() == JudgeWorkspaceTools(old.evidence).response_api_schemas()
    index = codex_judge_evidence_index(evidence)["actor_evidence"]
    assert index["required_complete_reads"] == REQUIRED_PATHS
    assert index["available_targeted_audit_reads"] == AUDIT_PATHS
    assert sum(files[path].byte_count for path in REQUIRED_PATHS) < sum(row.byte_count for row in old.evidence.workspace_after.files) / 10
    assert all(".eva-agent-judge/" not in ref for ref in prepared.source_workspace_refs)


def test_unknown_tool_content_fails_closed_instead_of_being_dropped(tmp_path):
    with pytest.raises(ValueError, match="cannot be dropped"):
        prepare(fixture(tmp_path, unknown=True))


def reads(prepared, *, source, audits=False):
    tools = JudgeWorkspaceTools(prepared.evidence)
    ids = RandomUUIDFactory()
    paths = [("after", path) for path in REQUIRED_PATHS]
    if source:
        paths += [("before", "input.txt"), ("after", "input.txt")]
    if audits:
        paths += [("after", path) for path in AUDIT_PATHS]
    calls = []
    for snapshot, path in paths:
        files = getattr(prepared.evidence, "workspace_" + snapshot).files
        size = next(row.byte_count for row in files if row.path == path)
        for offset in range(0, max(1, size), 65536):
            calls.append(ToolCall(call_id=ids.new("read"), name="workspace_read", arguments={
                "snapshot": snapshot, "path": path, "offset": offset, "max_bytes": 65536}))
    results = tools.execute_group(tuple(calls))
    return SimpleNamespace(agent_trace=SimpleNamespace(results=results, frontier_count=1, provider_turn_count=2))


def test_required_view_plus_before_after_source_reads_pass_without_full_optional_audits(tmp_path):
    prepared = prepare(fixture(tmp_path))
    _fast_workspace_read_coverage(reads(prepared, source=True), prepared.evidence)


def test_optional_audit_reads_never_replace_original_source_inspection(tmp_path):
    prepared = prepare(fixture(tmp_path))
    with pytest.raises(ValueError, match="source workspace before and after"):
        _fast_workspace_read_coverage(reads(prepared, source=False, audits=True), prepared.evidence)


def test_audit_citation_cannot_replace_per_item_source_citation(tmp_path):
    from dataclasses import replace
    from eva_agent.pipeline import JudgeRequest
    from eva_agent.training.agent_judge_worker import judgment_document
    from test_agent_judge_worker import _ProviderFreeWorkspaceJudge
    prepared = prepare(fixture(tmp_path))
    request = JudgeRequest(judgment_id=prepared.task.judge_task_id, judge_model_id=prepared.task.judge_model_id,
        policy_visible_context=prepared.evidence.policy_visible_context,
        workspace_evidence=prepared.evidence, judge_only_reference={})
    assessment = _ProviderFreeWorkspaceJudge().judge(request, prepared.trajectory.rubric)
    rows = tuple(replace(row, evidence_refs=("workspace:after:" + AUDIT_PATHS[0],)) for row in assessment.item_scores)
    with pytest.raises(ValueError, match="every rubric item must cite inspected workspace evidence"):
        judgment_document(prepared, replace(assessment, item_scores=rows))
