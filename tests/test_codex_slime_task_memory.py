"""CPU-only prospective task-note composition; no provider or optimizer calls."""
from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eva_agent.codex_runtime import CodexRole, CodexSandbox, CodexThreadOptions
from eva_agent.pipeline import ToolDefinition, ToolRegistry
from eva_agent.training import codex_slime_rollout as rollout
from eva_agent.training.teacher_worker import TeacherCandidateContext
from eva_agent.training.task_memory_tools import TASK_MEMORY_PROFILE, TASK_MEMORY_TOOL_NAMES
from test_codex_slime_rollout import _settings, native_fixture
from test_legacy_execution_binding import resolver


def registry():
    names = ("execute_code", "load_skill", "materialize_evidence_selection", "materialize_plan",
             "retrieve_frozen_evidence", "search_skills", "submit_results")
    return ToolRegistry(tuple(ToolDefinition(name=name, description="fixture original " + name,
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda workspace, arguments: {}, kind="tool", parallel_safe=False, read_only=False)
        for name in names))


def context():
    return TeacherCandidateContext(episode=object(), rubric=object(), tool_registry=registry(),
        turn_mcp_factory=object(), skills_factory=lambda request: (), skill_delivery={"fixture": True})


def test_opt_out_is_identity_and_opt_in_replaces_only_each_trajectory_registry():
    cached = context()
    before = cached.tool_registry.public_schemas()
    assert rollout._task_memory_context(cached, _settings()) is cached
    configured = {**_settings(), "task_memory_profile": TASK_MEMORY_PROFILE}
    first = rollout._task_memory_context(cached, configured)
    second = rollout._task_memory_context(cached, configured)
    assert first is not cached and second is not first
    assert first.tool_registry is not second.tool_registry
    assert cached.tool_registry.public_schemas() == before
    for field in fields(cached):
        if field.name != "tool_registry":
            assert getattr(first, field.name) is getattr(cached, field.name)
    for original in cached.tool_registry.definitions():
        assert first.tool_registry.definition(original.name) is original
    extra = set(row.name for row in first.tool_registry.definitions()) - set(row.name for row in cached.tool_registry.definitions())
    assert extra == set(TASK_MEMORY_TOOL_NAMES)


def test_note_guidance_is_opt_in_without_changing_sandbox_config_or_skill_bytes(tmp_path):
    base = CodexThreadOptions(role=CodexRole.STRONG_ACTOR, model="Qwen/Qwen3.5-9B",
        provider="eva_local_qwen", cwd=str(tmp_path), sandbox=CodexSandbox.READ_ONLY,
        config={"model_context_window": 24576, "model_auto_compact_token_limit": 19456},
        developer_instructions="original stage focus", offered_tools=())
    configured = {**_settings(), "task_memory_profile": TASK_MEMORY_PROFILE}
    old = rollout._research_thread_options(base, _settings(), endpoint="http://127.0.0.1:30910/v1")
    selected = rollout._research_thread_options(base, configured, endpoint="http://127.0.0.1:30910/v1")
    assert "public-notes-v1" not in old.developer_instructions
    assert selected.developer_instructions.startswith(old.developer_instructions)
    assert all(name in selected.developer_instructions for name in TASK_MEMORY_TOOL_NAMES)
    assert "already embedded in the SDK" in selected.developer_instructions
    assert "not via evamed/load_skill" in selected.developer_instructions
    assert "execute_code container does not contain these notes" in selected.developer_instructions
    assert selected.config == old.config and selected.sandbox is CodexSandbox.READ_ONLY
    assert selected.offered_tools == old.offered_tools
    assert selected.model == old.model and selected.provider == old.provider
    old_skills = rollout._research_skills_factory(lambda request: (), _settings())(object())
    selected_skills = rollout._research_skills_factory(lambda request: (), configured)(object())
    assert selected_skills == old_skills
    assert tuple(Path(skill.path).read_bytes() for skill in selected_skills) == tuple(
        Path(skill.path).read_bytes() for skill in old_skills)


def test_native_environment_selector_is_default_off_and_requires_measured_profile(monkeypatch, tmp_path):
    binary = tmp_path / "codex"
    binary.write_text("fixture"); binary.chmod(0o700)
    monkeypatch.setattr(rollout, "_binary_identity", lambda path: ("codex-cli 0.153.4", "a" * 64))
    monkeypatch.setenv("EVA_GRPO_CODEX_BIN", str(binary))
    monkeypatch.setenv("EVA_GRPO_CONTEXT_PROFILE", rollout.MEASURED_CONTEXT_PROFILE)
    monkeypatch.delenv(rollout.TASK_MEMORY_ENVIRONMENT_KEY, raising=False)
    args = SimpleNamespace(seq_length=24576, rollout_max_response_len=8192)
    old = rollout.native_settings(args)
    assert "task_memory_profile" not in old
    monkeypatch.setenv(rollout.TASK_MEMORY_ENVIRONMENT_KEY, TASK_MEMORY_PROFILE)
    assert rollout.native_settings(args) == {**old, "task_memory_profile": TASK_MEMORY_PROFILE}
    monkeypatch.setenv(rollout.TASK_MEMORY_ENVIRONMENT_KEY, "unreviewed")
    with pytest.raises(ValueError, match="Task memory requires"):
        rollout.native_settings(args)
    monkeypatch.setenv(rollout.TASK_MEMORY_ENVIRONMENT_KEY, TASK_MEMORY_PROFILE)
    monkeypatch.delenv("EVA_GRPO_CONTEXT_PROFILE")
    with pytest.raises(ValueError, match="Task memory requires"):
        rollout.native_settings(args)


def test_real_native_runner_uses_composed_context_and_records_profile(native_fixture, monkeypatch):
    """Reuse the existing native runner fixture; generation and Judge stay mocked."""
    from eva_agent.training import teacher_worker
    run, state, root = native_fixture
    compose = rollout._task_memory_context
    observed = []

    def fixture_context(source, settings):
        # The historical fixture uses SimpleNamespace; adapt only that fixture
        # into the same immutable dataclass supplied by the production cache.
        cached = TeacherCandidateContext(episode=source.episode, rubric=source.rubric,
            tool_registry=registry(), turn_mcp_factory=source.turn_mcp_factory,
            skills_factory=source.skills_factory)
        selected = compose(cached, settings)
        observed.append((cached, selected))
        return selected

    execute = teacher_worker.execute_single_rollout
    def inspect_execute(**kwargs):
        assert kwargs["context"] is observed[-1][1]
        return execute(**kwargs)

    monkeypatch.setattr(rollout, "_task_memory_context", fixture_context)
    monkeypatch.setattr(teacher_worker, "execute_single_rollout", inspect_execute)
    settings = {**_settings(), "task_memory_profile": TASK_MEMORY_PROFILE, "max_requests": 12,
        "codex_bin": Path("/fixture/codex"), "codex_version": "codex-cli 0.153.4",
        "codex_binary_blake3": "fixture"}
    result = run(settings=settings)
    assert len(observed) == 1 and observed[0][0] is not observed[0][1]
    assert len(observed[0][0].tool_registry.definitions()) == 7
    assert len(observed[0][1].tool_registry.definitions()) == 9
    assert state["thread_options"].sandbox is CodexSandbox.READ_ONLY
    assert state["selected_skill_ids"] == ("summary_failures",)
    assert state["judge_input"]["provider_metadata"]["task_memory_profile"] == TASK_MEMORY_PROFILE
    assert json.loads((root / "trajectory.json").read_text())["provider_metadata"]["task_memory_profile"] == TASK_MEMORY_PROFILE
    assert [sample.reward for sample in result] == [0.3333, 0.3333]
    assert [sample.tokens for sample in result] == [[10, 11, 12, 40, 41], [91, 92, 50]]


def test_s3_host_session_excludes_actor_notes_while_judge_snapshot_retains_them(resolver, tmp_path, monkeypatch):
    """Real signed prerequisite handlers; stop at the container launch boundary."""
    from uuid import uuid4
    from eva_agent.codex_pipeline.native_policy_v2 import build_stage_tool_guidance_v1
    from eva_agent.pipeline import DeterministicUUIDFactory, FilesystemSandbox, ParallelToolRuntime, ToolCall
    from eva_agent.pipeline.judge_workspace import JudgeWorkspaceTools
    from eva_agent.training.teacher_worker import hydrate_teacher_stage_prerequisites
    from test_legacy_execution_binding import AUTOMEDBENCH_S3_SOURCE_ID, _plain
    from test_judge_workspace import _bundle

    binding = resolver.resolve(str(uuid4()), source_candidate_id=AUTOMEDBENCH_S3_SOURCE_ID)
    original_schemas = binding.tool_registry.public_schemas()
    guidance = build_stage_tool_guidance_v1(public_runtime_context=_plain(binding.public_runtime_context),
        source_tool_catalog=original_schemas)
    cached = TeacherCandidateContext(
        episode=SimpleNamespace(policy_context={"execution_binding": binding.public_runtime_context}),
        rubric=object(), tool_registry=binding.tool_registry, turn_mcp_factory=None,
        skills_factory=None, stage_tool_guidance=guidance)
    selected = rollout._task_memory_context(cached, {**_settings(), "task_memory_profile": TASK_MEMORY_PROFILE})
    ids = DeterministicUUIDFactory("native-note-s3-host-separation")
    sandbox = FilesystemSandbox(tmp_path / "actor", ids.new("sandbox"), binding.initial_workspace_files)
    hydration = hydrate_teacher_stage_prerequisites(context=selected, workspace=sandbox, id_factory=ids)
    assert hydration["focus"] == "S3" and hydration["provider_calls"] == 0
    before = sandbox.snapshot("before-notes")
    runtime = ParallelToolRuntime(workspace=sandbox, registry=selected.tool_registry,
        id_factory=ids, maximum_parallel_calls=1)
    note = runtime.execute((ToolCall(ids.new("note-call"), "automed_write_note",
        {"name": "task-state.md", "content": "Fixture task note; S3 has not executed."}),))[0]
    assert note.status == "completed"
    observed = []

    class FixtureContainerBoundary(RuntimeError):
        pass

    def stop_before_container(contract, code, **kwargs):
        # The unchanged host passes only its frozen evidence input directory,
        # not the actor tree. No Docker/process execution occurs in this test.
        assert kwargs["input_dir"].resolve() != sandbox.root.resolve()
        assert not (kwargs["input_dir"] / "notes/task-state.md").exists()
        episode = kwargs["episode_dir"]
        assert episode.name == "s3-attempt-1" and not episode.exists()
        assert not episode.is_relative_to(sandbox.root)
        assert kwargs["prerequisite_receipt"] is not None
        observed.append(episode)
        raise FixtureContainerBoundary("CPU fixture stops before container execution")

    monkeypatch.setattr(resolver.modules.execution, "execute_python_contract", stop_before_container)
    execution = runtime.execute((ToolCall(ids.new("execute-call"), "execute_code",
        {"stage": "S3", "code": "print('fixture')"}),))[0]
    assert execution.status == "immutable_failure" and execution.error_code == "FixtureContainerBoundary"
    assert len(observed) == 1  # Reached actual S3 host path, not a prerequisite rejection.
    assert binding.tool_registry.public_schemas() == original_schemas
    after = sandbox.snapshot("after-notes-and-s3-boundary")
    judge = JudgeWorkspaceTools(_bundle(ids, before=before, after=after), id_factory=ids)
    read = judge.execute_group((ToolCall(ids.new("judge-read"), "workspace_read",
        {"snapshot": "after", "path": "notes/task-state.md", "offset": 0, "max_bytes": 1024}),))[0]
    assert read.status == "completed"
    assert read.output["content"] == "Fixture task note; S3 has not executed."
