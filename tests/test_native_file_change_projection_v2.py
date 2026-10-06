from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import pytest

from eva_agent.codex_pipeline import (
    CodexOpus5AgentJudge,
    CodexPipelineError,
    CodexRolloutAdapter,
    CodexToolExecutionBridge,
)
from eva_agent.pipeline import (
    Cohort,
    DeterministicUUIDFactory,
    EvidenceBundle,
    FilesystemSandbox,
    ImmutableArtifactStore,
    JudgeRequest,
    ParallelToolRuntime,
    PipelineVerifier,
    PipelineVerificationError,
    Stage,
    ToolRegistry,
)
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value
from eva_agent.pipeline.runner import evidence_core

from test_codex_pipeline_adapter import (
    _RuntimeFactory,
    _evidence as fixture_evidence,
    _event,
    _judge_options,
    _judge_script,
    _native_actor_options,
    _request,
    _rubric,
)


Mutation = Callable[[FilesystemSandbox], None]


def _native_script(
    changes: Sequence[Mapping[str, Any]],
    mutate: Mutation,
):
    def script(
        thread: str,
        turn: str,
        bridge: CodexToolExecutionBridge | None = None,
    ):
        assert bridge is not None
        mutate(bridge._tools._workspace)
        item = {
            "id": "native-file-change",
            "type": "fileChange",
            "changes": [dict(change) for change in changes],
            "status": "inProgress",
        }
        return (
            _event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"}),
            _event("item/started", thread, turn, item=item),
            _event("item/completed", thread, turn, item={**item, "status": "completed"}),
            _event(
                "item/completed",
                thread,
                turn,
                item={
                    "id": "answer",
                    "type": "agentMessage",
                    "phase": "final_answer",
                    "text": "native workspace result committed",
                },
            ),
            _event("turn/completed", thread, turn, turn={"id": turn, "status": "completed"}),
        )

    return script


def _native_tool_free_script(mutate: Mutation):
    def script(
        thread: str,
        turn: str,
        bridge: CodexToolExecutionBridge | None = None,
    ):
        assert bridge is not None
        mutate(bridge._tools._workspace)
        return (
            _event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"}),
            _event(
                "item/completed",
                thread,
                turn,
                item={
                    "id": "answer",
                    "type": "agentMessage",
                    "phase": "final_answer",
                    "text": "done",
                },
            ),
            _event("turn/completed", thread, turn, turn={"id": turn, "status": "completed"}),
        )

    return script


def _run(
    tmp_path: Path,
    *,
    script,
    initial_files: Mapping[str, bytes],
    stage: Stage = Stage.E2E,
    suffix: str = "native-v2",
):
    request, _unused = _request(Cohort.STRONG, tmp_path, suffix=suffix, stage=stage)
    workspace = FilesystemSandbox(
        tmp_path / f"workspace-{suffix}",
        DeterministicUUIDFactory(f"workspace-{suffix}").new("workspace"),
        initial_files,
    )
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=ToolRegistry(()),
        id_factory=DeterministicUUIDFactory(f"tools-{suffix}"),
    )
    factory = _RuntimeFactory(script)
    adapter = CodexRolloutAdapter(
        factory,
        _native_actor_options,
        id_factory=DeterministicUUIDFactory(f"adapter-{suffix}"),
        enable_native_actions=True,
    )
    before = workspace.snapshot("independent-before")
    rollout = adapter.run(replace(request, available_tools=()), runtime)
    after = workspace.snapshot("independent-after")
    return request, workspace, runtime, rollout, before, after


def _evidence(request, runtime, rollout, before, after) -> EvidenceBundle:
    ids = DeterministicUUIDFactory("native-v2-evidence")
    partial = EvidenceBundle(
        bundle_id=ids.new("bundle"),
        rollout_id=request.rollout_id,
        sandbox_manifest=request.sandbox,
        model=request.model,
        policy_visible_context=request.policy_visible_context,
        context_blake3=blake3_hex(request.policy_visible_context),
        workspace_before=before,
        workspace_after=after,
        tool_trace=runtime.trace(),
        policy_events=rollout.policy_events,
        assistant_output=rollout.assistant_output,
        provider_receipt_blake3=rollout.provider_receipt_blake3,
        safe_provider_metadata=rollout.safe_metadata,
        bundle_blake3="",
    )
    return replace(partial, bundle_blake3=blake3_hex(evidence_core(partial)))


@pytest.mark.parametrize("stage", (Stage.S3, Stage.S4, Stage.E2E))
def test_native_v2_binds_add_delete_update_move_and_exact_digests(
    tmp_path: Path,
    stage: Stage,
) -> None:
    changes = (
        {"path": "added.txt", "kind": {"type": "add"}, "diff": "+added\n"},
        {"path": "deleted.txt", "kind": {"type": "delete"}, "diff": "-deleted\n"},
        {"path": "updated.txt", "kind": {"type": "update"}, "diff": "-old\n+new\n"},
        {
            "path": "move-source.txt",
            "kind": {"type": "update", "movePath": "move-target.txt"},
            "diff": "rename and retain bytes",
        },
    )

    def mutate(workspace: FilesystemSandbox) -> None:
        workspace.write_bytes("added.txt", b"added\n", create_only=True)
        (workspace.root / "deleted.txt").unlink()
        workspace.write_bytes("updated.txt", b"new\n")
        os.replace(workspace.root / "move-source.txt", workspace.root / "move-target.txt")

    request, _workspace, runtime, rollout, before, after = _run(
        tmp_path,
        script=_native_script(changes, mutate),
        initial_files={
            "seed.txt": b"stable\n",
            "deleted.txt": b"deleted\n",
            "updated.txt": b"old\n",
            "move-source.txt": b"move bytes\n",
        },
        stage=stage,
        suffix=f"semantics-{stage.value}",
    )
    binding = rollout.safe_metadata["native_file_change_effect_binding"]
    expected_paths = (
        "added.txt",
        "deleted.txt",
        "move-source.txt",
        "move-target.txt",
        "updated.txt",
    )
    assert rollout.safe_metadata["schema"] == (
        "eva.codex-provider-rollout-projection.v2-native-file-change"
    )
    assert binding["declared_changed_paths"] == expected_paths
    assert binding["actual_changed_paths"] == expected_paths
    assert binding["workspace_before_tree_blake3"] == before.tree_blake3
    assert binding["workspace_after_tree_blake3"] == after.tree_blake3
    assert tuple(effect["kind"] for effect in binding["effects"]) == (
        "add",
        "delete",
        "update",
        "update",
    )
    assert binding["effects"][0]["path_effects"][0]["after"][
        "content_blake3"
    ] == blake3_bytes(b"added\n")
    assert binding["effects"][2]["path_effects"][0]["before"][
        "content_blake3"
    ] == blake3_bytes(b"old\n")
    assert binding["effects"][2]["path_effects"][0]["after"][
        "content_blake3"
    ] == blake3_bytes(b"new\n")
    assert binding["effects"][3]["destination_path"] == "move-target.txt"

    evidence = _evidence(request, runtime, rollout, before, after)
    verifier = PipelineVerifier(
        ImmutableArtifactStore(tmp_path / f"artifacts-{stage.value}")
    )
    verifier._verify_trajectory(
        SimpleNamespace(evidence=evidence, cohort=Cohort.STRONG)
    )


def test_native_v2_fails_on_undeclared_workspace_mutation_with_receipt(
    tmp_path: Path,
) -> None:
    changes = (
        {"path": "declared.txt", "kind": {"type": "add"}, "diff": "+declared\n"},
    )

    def mutate(workspace: FilesystemSandbox) -> None:
        workspace.write_bytes("declared.txt", b"declared\n", create_only=True)
        workspace.write_bytes("undeclared.txt", b"hidden\n", create_only=True)

    request, _unused = _request(Cohort.STRONG, tmp_path, suffix="undeclared")
    workspace = FilesystemSandbox(
        tmp_path / "undeclared-workspace",
        DeterministicUUIDFactory("undeclared-workspace").new("workspace"),
        {"seed.txt": b"stable\n"},
    )
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=ToolRegistry(()),
        id_factory=DeterministicUUIDFactory("undeclared-tools"),
    )
    adapter = CodexRolloutAdapter(
        _RuntimeFactory(_native_script(changes, mutate)),
        _native_actor_options,
        enable_native_actions=True,
    )
    with pytest.raises(
        CodexPipelineError,
        match="declared paths and actual workspace effects differ",
    ) as captured:
        adapter.run(replace(request, available_tools=()), runtime)
    assert captured.value.receipt is not None


@pytest.mark.parametrize(
    "scenario",
    ("add-existing", "delete-retained", "update-added", "move-copied"),
)
def test_native_v2_rejects_false_change_semantics(
    tmp_path: Path,
    scenario: str,
) -> None:
    initial = {"seed.txt": b"old\n"}
    if scenario == "add-existing":
        change = {"path": "seed.txt", "kind": {"type": "add"}, "diff": "+new\n"}

        def mutate(workspace: FilesystemSandbox) -> None:
            workspace.write_bytes("seed.txt", b"new\n")

    elif scenario == "delete-retained":
        change = {"path": "seed.txt", "kind": {"type": "delete"}, "diff": "-old\n"}

        def mutate(workspace: FilesystemSandbox) -> None:
            workspace.write_bytes("seed.txt", b"retained\n")

    elif scenario == "update-added":
        initial = {"stable.txt": b"stable\n"}
        change = {"path": "seed.txt", "kind": {"type": "update"}, "diff": "+new\n"}

        def mutate(workspace: FilesystemSandbox) -> None:
            workspace.write_bytes("seed.txt", b"new\n", create_only=True)

    else:
        change = {
            "path": "seed.txt",
            "kind": {"type": "update", "movePath": "moved.txt"},
            "diff": "copy is not move",
        }

        def mutate(workspace: FilesystemSandbox) -> None:
            workspace.write_bytes("moved.txt", b"old\n", create_only=True)

    request, _unused = _request(Cohort.STRONG, tmp_path, suffix=scenario)
    workspace = FilesystemSandbox(
        tmp_path / f"{scenario}-workspace",
        DeterministicUUIDFactory(f"{scenario}-workspace").new("workspace"),
        initial,
    )
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=ToolRegistry(()),
        id_factory=DeterministicUUIDFactory(f"{scenario}-tools"),
    )
    adapter = CodexRolloutAdapter(
        _RuntimeFactory(_native_script((change,), mutate)),
        _native_actor_options,
        enable_native_actions=True,
    )
    with pytest.raises(CodexPipelineError, match="effect differs") as captured:
        adapter.run(replace(request, available_tools=()), runtime)
    assert captured.value.receipt is not None


def test_native_v2_fails_when_workspace_changes_without_file_change(
    tmp_path: Path,
) -> None:
    def mutate(workspace: FilesystemSandbox) -> None:
        workspace.write_bytes("hidden.txt", b"hidden\n", create_only=True)

    request, _unused = _request(Cohort.STRONG, tmp_path, suffix="hidden-no-call")
    workspace = FilesystemSandbox(
        tmp_path / "hidden-no-call-workspace",
        DeterministicUUIDFactory("hidden-no-call-workspace").new("workspace"),
        {"seed.txt": b"stable\n"},
    )
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=ToolRegistry(()),
        id_factory=DeterministicUUIDFactory("hidden-no-call-tools"),
    )
    adapter = CodexRolloutAdapter(
        _RuntimeFactory(_native_tool_free_script(mutate)),
        _native_actor_options,
        enable_native_actions=True,
    )
    with pytest.raises(CodexPipelineError, match="declared paths") as captured:
        adapter.run(replace(request, available_tools=()), runtime)
    assert captured.value.receipt is not None


def test_native_v2_tool_free_turn_binds_an_exact_empty_effect(tmp_path: Path) -> None:
    request, _workspace, runtime, rollout, before, after = _run(
        tmp_path,
        script=_native_tool_free_script(lambda workspace: None),
        initial_files={"seed.txt": b"stable\n"},
        suffix="empty-effect",
    )
    binding = rollout.safe_metadata["native_file_change_effect_binding"]
    assert binding["declared_changed_paths"] == ()
    assert binding["actual_changed_paths"] == ()
    assert binding["effects"] == ()
    assert before.tree_blake3 == after.tree_blake3
    evidence = _evidence(request, runtime, rollout, before, after)
    PipelineVerifier(ImmutableArtifactStore(tmp_path / "empty-artifacts"))._verify_trajectory(
        SimpleNamespace(evidence=evidence, cohort=Cohort.STRONG)
    )


@pytest.mark.parametrize("topology", ("symlink", "hardlink"))
def test_native_v2_rejects_post_turn_link_topology_and_retains_receipt(
    tmp_path: Path,
    topology: str,
) -> None:
    outside = tmp_path / f"outside-{topology}.txt"
    outside.write_bytes(b"outside\n")
    change = {"path": "linked.txt", "kind": {"type": "add"}, "diff": "+outside\n"}

    def mutate(workspace: FilesystemSandbox) -> None:
        linked = workspace.root / "linked.txt"
        if topology == "symlink":
            linked.symlink_to(outside)
        else:
            linked.hardlink_to(outside)

    request, _unused = _request(Cohort.STRONG, tmp_path, suffix=f"post-{topology}")
    workspace = FilesystemSandbox(
        tmp_path / f"post-{topology}-workspace",
        DeterministicUUIDFactory(f"post-{topology}-workspace").new("workspace"),
        {"seed.txt": b"stable\n"},
    )
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=ToolRegistry(()),
        id_factory=DeterministicUUIDFactory(f"post-{topology}-tools"),
    )
    adapter = CodexRolloutAdapter(
        _RuntimeFactory(_native_script((change,), mutate)),
        _native_actor_options,
        enable_native_actions=True,
    )
    with pytest.raises(CodexPipelineError, match="snapshot failed closed") as captured:
        adapter.run(replace(request, available_tools=()), runtime)
    assert captured.value.receipt is not None


def test_native_v2_binding_tamper_is_rejected_independently(tmp_path: Path) -> None:
    change = {"path": "result.txt", "kind": {"type": "add"}, "diff": "+result\n"}

    def mutate(workspace: FilesystemSandbox) -> None:
        workspace.write_bytes("result.txt", b"result\n", create_only=True)

    request, _workspace, runtime, rollout, before, after = _run(
        tmp_path,
        script=_native_script((change,), mutate),
        initial_files={"seed.txt": b"stable\n"},
        suffix="tamper",
    )
    evidence = _evidence(request, runtime, rollout, before, after)
    metadata = canonical_value(evidence.safe_provider_metadata)
    metadata["native_file_change_effect_binding"]["effects"][0]["path_effects"][0][
        "after"
    ]["content_blake3"] = "0" * 64
    tampered = replace(evidence, safe_provider_metadata=metadata)
    verifier = PipelineVerifier(ImmutableArtifactStore(tmp_path / "tamper-artifacts"))
    with pytest.raises(PipelineVerificationError, match="effect binding differs"):
        verifier._verify_trajectory(
            SimpleNamespace(evidence=tampered, cohort=Cohort.STRONG)
        )


def test_native_v2_resumed_receipt_is_rejected_independently(tmp_path: Path) -> None:
    change = {"path": "result.txt", "kind": {"type": "add"}, "diff": "+result\n"}

    def mutate(workspace: FilesystemSandbox) -> None:
        workspace.write_bytes("result.txt", b"result\n", create_only=True)

    request, _workspace, runtime, rollout, before, after = _run(
        tmp_path,
        script=_native_script((change,), mutate),
        initial_files={"seed.txt": b"stable\n"},
        suffix="resume-tamper",
    )
    evidence = _evidence(request, runtime, rollout, before, after)
    metadata = canonical_value(evidence.safe_provider_metadata)
    receipt = metadata["codex_turn_receipt"]
    receipt["thread_resumed"] = True
    receipt["receipt_blake3"] = blake3_hex(
        {key: value for key, value in receipt.items() if key != "receipt_blake3"}
    )
    tampered = replace(evidence, safe_provider_metadata=metadata)
    verifier = PipelineVerifier(ImmutableArtifactStore(tmp_path / "resume-artifacts"))
    with pytest.raises(PipelineVerificationError, match="receipt identity differs"):
        verifier._verify_trajectory(
            SimpleNamespace(evidence=tampered, cohort=Cohort.STRONG)
        )


def test_judge_file_change_is_forbidden_and_receipt_is_retained(tmp_path: Path) -> None:
    evidence = fixture_evidence(tmp_path)
    rubric = _rubric()

    def judge_file_change(thread: str, turn: str, bridge=None):
        del bridge
        item = {
            "id": "judge-file-change",
            "type": "fileChange",
            "changes": [{
                "path": "forbidden.txt",
                "kind": {"type": "add"},
                "diff": "+forbidden\n",
            }],
            "status": "inProgress",
        }
        return (
            _event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"}),
            _event("item/started", thread, turn, item=item),
            _event("item/completed", thread, turn, item={**item, "status": "completed"}),
            _event(
                "item/completed",
                thread,
                turn,
                item={
                    "id": "answer",
                    "type": "agentMessage",
                    "phase": "final_answer",
                    "text": "{}",
                },
            ),
            _event("turn/completed", thread, turn, turn={"id": turn, "status": "completed"}),
        )

    judge = CodexOpus5AgentJudge(
        _RuntimeFactory(judge_file_change),
        _judge_options,
        model_id="claude-opus-5",
        id_factory=DeterministicUUIDFactory("judge-file-change"),
    )
    with pytest.raises(
        CodexPipelineError,
        match="forbidden by this turn capability",
    ) as captured:
        judge.judge(
            JudgeRequest(
                judgment_id=DeterministicUUIDFactory("judge-file-change-id").new(
                    "judgment"
                ),
                judge_model_id="claude-opus-5",
                policy_visible_context=evidence.policy_visible_context,
                workspace_evidence=evidence,
                judge_only_reference=None,
            ),
            rubric,
        )
    assert captured.value.receipt is not None
    assert captured.value.receipt.tool_calls[0].tool_type == "fileChange"


def test_read_only_opus_judge_reopens_and_cites_native_written_file(
    tmp_path: Path,
) -> None:
    change = {
        "path": "native-result.txt",
        "kind": {"type": "add"},
        "diff": "+clinically verified\n",
    }

    def mutate(workspace: FilesystemSandbox) -> None:
        workspace.write_bytes(
            "native-result.txt", b"clinically verified\n", create_only=True
        )

    request, _workspace, runtime, rollout, before, after = _run(
        tmp_path,
        script=_native_script((change,), mutate),
        initial_files={"seed.txt": b"stable\n"},
        suffix="judge-reopen",
    )
    evidence = _evidence(request, runtime, rollout, before, after)
    verifier = PipelineVerifier(ImmutableArtifactStore(tmp_path / "judge-artifacts"))
    verifier._verify_trajectory(
        SimpleNamespace(evidence=evidence, cohort=Cohort.STRONG)
    )

    rubric = _rubric()
    cite = "workspace:after:native-result.txt"
    judge = CodexOpus5AgentJudge(
        _RuntimeFactory(
            _judge_script(
                evidence,
                rubric,
                cite=cite,
                path="native-result.txt",
            )
        ),
        _judge_options,
        model_id="claude-opus-5",
        id_factory=DeterministicUUIDFactory("native-v2-judge"),
    )
    assessment = judge.judge(
        JudgeRequest(
            judgment_id=DeterministicUUIDFactory("native-v2-judgment").new("judgment"),
            judge_model_id="claude-opus-5",
            policy_visible_context=evidence.policy_visible_context,
            workspace_evidence=evidence,
            judge_only_reference={"reference_answer": "private"},
        ),
        rubric,
    )
    assert assessment.agent_trace.inspected_evidence_refs == (cite,)
    assert all(score.evidence_refs == (cite,) for score in assessment.item_scores)
    native_decision_index = next(
        index
        for index, event in enumerate(evidence.policy_events)
        if event.role == "assistant" and event.tool_call_ids
    )
    assert evidence.policy_events[native_decision_index + 1].role == "tool"
    assert evidence.policy_events[native_decision_index + 2].role == "assistant"
