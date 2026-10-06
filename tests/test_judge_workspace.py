from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from eva_agent.pipeline.contracts import (
    BenchmarkSource,
    Cohort,
    EvidenceBundle,
    FileSnapshot,
    ModelTarget,
    RubricBinding,
    SandboxManifest,
    Stage,
    ToolCall,
    ToolTrace,
    TrajectoryEvent,
    WorkspaceSnapshot,
)
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex
from eva_agent.pipeline.ids import DeterministicUUIDFactory
from eva_agent.pipeline.judge_workspace import JudgeWorkspaceError, JudgeWorkspaceTools


def _snapshot(label: str, files: Mapping[str, bytes], *, modes: Mapping[str, str] | None = None):
    rows = tuple(
        FileSnapshot(
            path=path,
            content=payload,
            byte_count=len(payload),
            mode=(modes or {}).get(path, "0600"),
            content_blake3=blake3_bytes(payload),
        )
        for path, payload in sorted(files.items())
    )
    core = {"files": rows, "file_count": len(rows), "byte_count": sum(row.byte_count for row in rows)}
    return WorkspaceSnapshot(
        label=label,
        files=rows,
        file_count=len(rows),
        byte_count=core["byte_count"],
        tree_blake3=blake3_hex(core),
    )


def _evidence_core(evidence: EvidenceBundle) -> dict[str, Any]:
    return {
        "bundle_id": evidence.bundle_id,
        "rollout_id": evidence.rollout_id,
        "sandbox_manifest": evidence.sandbox_manifest,
        "model": evidence.model,
        "policy_visible_context": evidence.policy_visible_context,
        "context_blake3": evidence.context_blake3,
        "workspace_before": evidence.workspace_before,
        "workspace_after": evidence.workspace_after,
        "tool_trace": evidence.tool_trace,
        "policy_events": evidence.policy_events,
        "assistant_output": evidence.assistant_output,
        "provider_receipt_blake3": evidence.provider_receipt_blake3,
        "safe_provider_metadata": evidence.safe_provider_metadata,
    }


def _event(ids, role: str, content: Any, call_ids=()):
    core = {
        "event_id": ids.new("judge-policy-event"),
        "role": role,
        "content": content,
        "tool_call_ids": tuple(call_ids),
    }
    return TrajectoryEvent(**core, event_blake3=blake3_hex(core))


def _bundle(
    ids: DeterministicUUIDFactory,
    *,
    before: WorkspaceSnapshot | None = None,
    after: WorkspaceSnapshot | None = None,
) -> EvidenceBundle:
    before = before or _snapshot(
        "before-rollout",
        {"notes.txt": b"baseline\n", "unchanged.txt": b"same\n"},
    )
    after = after or _snapshot(
        "after-rollout",
        {
            "notes.txt": b"baseline\nfinding: aspirin evidence verified\n",
            "result.json": b'{"status":"verified"}\n',
            "unchanged.txt": b"same\n",
        },
    )
    context = {"question": "Inspect the committed output."}
    binding = RubricBinding(
        rubric_id="fixture-rubric",
        version=1,
        digest=blake3_hex("rubric"),
        domain="medxpertqa",
        stage=Stage.E2E,
    )
    sandbox = SandboxManifest(
        sandbox_id=ids.new("sandbox"),
        episode_id=ids.new("episode"),
        source=BenchmarkSource("MedXpertQA", "fixture.json", "fixture-v1"),
        domain="medxpertqa",
        stage=Stage.E2E,
        instruction="Inspect and judge the evidence.",
        policy_context=context,
        initial_files={"notes.txt": b"baseline\n"},
        rubric=binding,
        manifest_blake3=blake3_hex("sandbox-manifest"),
    )
    empty_trace_core = {
        "results": (),
        "declared_call_ids": (),
        "joined_call_ids": (),
        "frontier_count": 0,
        "max_parallelism_observed": 0,
        "retry_count": 0,
    }
    trace = ToolTrace(**empty_trace_core, trace_blake3=blake3_hex(empty_trace_core))
    policy_events = (
        _event(ids, "system", "policy"),
        _event(ids, "user", context),
        _event(ids, "assistant", "completed rollout"),
    )
    partial = EvidenceBundle(
        bundle_id=ids.new("evidence-bundle"),
        rollout_id=ids.new("rollout"),
        sandbox_manifest=sandbox,
        model=ModelTarget(Cohort.STRONG, "fixture-strong", "fixture"),
        policy_visible_context=context,
        context_blake3=blake3_hex(context),
        workspace_before=before,
        workspace_after=after,
        tool_trace=trace,
        policy_events=policy_events,
        assistant_output="completed rollout",
        provider_receipt_blake3=blake3_hex("provider"),
        safe_provider_metadata={"fixture": True},
        bundle_blake3="",
    )
    return replace(partial, bundle_blake3=blake3_hex(_evidence_core(partial)))


def _call(ids, name: str, arguments: Mapping[str, Any]) -> ToolCall:
    return ToolCall(call_id=ids.new("judge-tool-call"), name=name, arguments=arguments)


def test_strict_snapshot_tools_inspect_content_in_parallel_without_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = DeterministicUUIDFactory("judge-tools")
    evidence = _bundle(ids)
    tools = JudgeWorkspaceTools(evidence, id_factory=ids, maximum_parallel_tools=8)

    schemas = tools.response_api_schemas()
    assert [schema["name"] for schema in schemas] == [
        "workspace_diff",
        "workspace_list",
        "workspace_read",
        "workspace_search",
    ]
    for schema in schemas:
        assert schema["strict"] is True
        parameters = schema["parameters"]
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == set(parameters["properties"])

    # There is deliberately no filesystem handle in the runtime. Even a global open()
    # denial cannot affect committed-snapshot inspection.
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    calls = (
        _call(ids, "workspace_list", {"snapshot": "after", "prefix": None, "max_entries": 20}),
        _call(
            ids,
            "workspace_read",
            {"snapshot": "after", "path": "result.json", "offset": 0, "max_bytes": 4096},
        ),
        _call(
            ids,
            "workspace_search",
            {
                "snapshot": "after",
                "query": "ASPIRIN",
                "path_prefix": None,
                "case_sensitive": False,
                "max_matches": 20,
            },
        ),
        _call(ids, "workspace_diff", {"path": None, "max_files": 20}),
    )
    results = tools.execute_group(calls)
    assert len(results) == 4
    assert all(result.status == "completed" for result in results)
    assert tools.max_parallelism_observed == 4
    by_name = {result.name: result for result in results}
    assert by_name["workspace_read"].content_inspection is True
    assert by_name["workspace_search"].content_inspection is True
    assert by_name["workspace_list"].content_inspection is False
    assert by_name["workspace_diff"].content_inspection is False
    assert by_name["workspace_read"].output["content"] == '{"status":"verified"}\n'
    assert by_name["workspace_search"].output["matches"][0]["path"] == "notes.txt"
    assert all(
        ref.startswith(("workspace:before:", "workspace:after:"))
        for result in results
        for ref in result.inspected_evidence_refs
    )
    assert by_name["workspace_read"].inspected_evidence_refs == (
        "workspace:after:result.json",
    )
    assert "workspace:before:notes.txt" in by_name["workspace_diff"].inspected_evidence_refs
    assert "workspace:after:notes.txt" in by_name["workspace_diff"].inspected_evidence_refs

    policy_events = (
        _event(ids, "system", "You are an evidence judge."),
        _event(ids, "user", "Inspect committed workspace evidence."),
        _event(ids, "assistant", {"tool_calls": [call.name for call in calls]}, [c.call_id for c in calls]),
        *(
            _event(ids, "tool", {"receipt_blake3": result.receipt_blake3}, [result.call_id])
            for result in results
        ),
        _event(ids, "assistant", "Evidence inspected; scoring follows."),
    )
    trace = tools.trace(policy_events=policy_events, provider_turn_count=2)
    assert trace.content_inspection_count == 2
    assert trace.max_parallelism_observed == 4
    assert trace.retry_count == 0
    assert trace.joined_call_ids == tuple(call.call_id for call in calls)
    assert trace.evidence_bundle_blake3 == evidence.bundle_blake3


def test_tool_arguments_fail_closed_with_immutable_receipts() -> None:
    ids = DeterministicUUIDFactory("judge-tool-errors")
    tools = JudgeWorkspaceTools(_bundle(ids), id_factory=ids)
    unsafe = _call(
        ids,
        "workspace_read",
        {"snapshot": "after", "path": "../secret", "offset": 0, "max_bytes": 10},
    )
    extra = _call(
        ids,
        "workspace_list",
        {"snapshot": "after", "prefix": None, "max_entries": 10, "unexpected": True},
    )
    results = tools.execute_group((unsafe, extra))
    assert [(row.status, row.error_code) for row in results] == [
        ("immutable_failure", "unsafe_path"),
        ("immutable_failure", "invalid_arguments"),
    ]
    assert all(row.output is None and not row.inspected_evidence_refs for row in results)
    assert all(row.content_inspection is False for row in results)
    assert len({row.receipt_blake3 for row in results}) == 2


def test_same_turn_call_group_cannot_exceed_parallel_bound() -> None:
    ids = DeterministicUUIDFactory("judge-tool-width")
    tools = JudgeWorkspaceTools(_bundle(ids), id_factory=ids, maximum_parallel_tools=1)
    calls = (
        _call(ids, "workspace_list", {"snapshot": "before", "prefix": None, "max_entries": 10}),
        _call(ids, "workspace_list", {"snapshot": "after", "prefix": None, "max_entries": 10}),
    )
    with pytest.raises(JudgeWorkspaceError, match="parallel-tool bound"):
        tools.execute_group(calls)


@pytest.mark.parametrize(
    "bad_after,error",
    [
        (
            _snapshot("bad-path", {"safe/../secret.txt": b"secret"}),
            "unsafe or symlink-like path",
        ),
        (
            _snapshot("prefix-collision", {"alias": b"file", "alias/child": b"child"}),
            "symlink-like prefix topology",
        ),
        (
            _snapshot("symlink-mode", {"link": b"target"}, modes={"link": "120777"}),
            "file commitment differs",
        ),
    ],
)
def test_unsafe_or_symlink_like_snapshot_topology_is_rejected(bad_after, error: str) -> None:
    ids = DeterministicUUIDFactory("judge-bad-topology")
    evidence = _bundle(ids, after=bad_after)
    with pytest.raises(JudgeWorkspaceError, match=error):
        JudgeWorkspaceTools(evidence, id_factory=ids)


def test_snapshot_content_and_bundle_tamper_are_rejected() -> None:
    ids = DeterministicUUIDFactory("judge-tamper")
    valid = _bundle(ids)
    original = valid.workspace_after.files[0]
    tampered_file = replace(original, content=original.content + b"tamper")
    rows = (tampered_file, *valid.workspace_after.files[1:])
    bad_snapshot_core = {
        "files": rows,
        "file_count": len(rows),
        "byte_count": sum(row.byte_count for row in rows),
    }
    bad_snapshot = WorkspaceSnapshot(
        label="tampered-after",
        files=rows,
        file_count=len(rows),
        byte_count=bad_snapshot_core["byte_count"],
        tree_blake3=blake3_hex(bad_snapshot_core),
    )
    partial = replace(valid, workspace_after=bad_snapshot, bundle_blake3="")
    committed_tamper = replace(partial, bundle_blake3=blake3_hex(_evidence_core(partial)))
    with pytest.raises(JudgeWorkspaceError, match="file commitment differs"):
        JudgeWorkspaceTools(committed_tamper, id_factory=ids)

    with pytest.raises(JudgeWorkspaceError, match="evidence bundle BLAKE3 differs"):
        JudgeWorkspaceTools(replace(valid, assistant_output="changed"), id_factory=ids)
