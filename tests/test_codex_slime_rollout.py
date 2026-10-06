from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from eva_agent.training.codex_slime_rollout import (
    EARLY_COMPACT_CONTEXT_PROFILE,
    EARLY_COMPACT_TOKENS,
    LOSSLESS_HEADROOM_CONTEXT_PROFILE,
    LOSSLESS_HEADROOM_CONTEXT_TOKENS,
    MEASURED_COMPACT_TOKENS,
    MEASURED_CONTEXT_PROFILE,
    MEASURED_CONTEXT_TOKENS,
    MEASURED_OUTPUT_TOKENS,
    MEASURED_RESERVE_TOKENS,
    MEASURED_TOOL_OUTPUT_TOKENS,
    MEASURED_TOTAL_OUTPUT_BUDGET,
    _research_skills_factory,
    _research_thread_options,
    generate_grpo_rollout,
    native_compaction_limit,
    native_settings,
    run_native_trajectory,
    samples_from_segments,
)


@pytest.mark.parametrize("context,expected", [(8192, 3072), (16384, 10240), (24576, 10240)])
def test_native_compacts_before_measured_transcript_expansion(context, expected):
    assert native_compaction_limit(context) == expected
    assert expected + 4096 + 1024 <= context


def _settings(profile=True):
    result = {
        "max_context_tokens": MEASURED_CONTEXT_TOKENS,
        "max_output_tokens": MEASURED_OUTPUT_TOKENS,
        "total_output_budget": MEASURED_TOTAL_OUTPUT_BUDGET,
        "auto_compact_token_limit": MEASURED_COMPACT_TOKENS,
        "context_reserve_tokens": MEASURED_RESERVE_TOKENS,
        "tool_output_token_limit": MEASURED_TOOL_OUTPUT_TOKENS,
    }
    if profile:
        result["context_profile"] = MEASURED_CONTEXT_PROFILE
    return result


def test_measured_profile_is_explicit_and_historical_settings_stay_unchanged(monkeypatch, tmp_path):
    import eva_agent.training.codex_slime_rollout as rollout

    binary = tmp_path / "codex"
    binary.write_text("fixture")
    binary.chmod(0o700)
    monkeypatch.setattr(rollout, "_binary_identity", lambda _: ("codex-cli 0.153.4", "a" * 64))
    monkeypatch.setenv("EVA_GRPO_CODEX_BIN", str(binary))
    monkeypatch.delenv("EVA_GRPO_CONTEXT_PROFILE", raising=False)
    args = SimpleNamespace(seq_length=24576, rollout_max_response_len=8192)
    historical = native_settings(args)
    assert "recover_completed_invalid_tool_history" not in historical
    monkeypatch.setenv("EVA_GRPO_RECOVER_COMPLETED_INVALID_TOOL_HISTORY", "1")
    recovered = native_settings(args)
    assert recovered == {**historical, "recover_completed_invalid_tool_history": True}
    monkeypatch.setenv("EVA_GRPO_RECOVER_COMPLETED_INVALID_TOOL_HISTORY", "true")
    with pytest.raises(ValueError, match="Invalid completed tool history"):
        native_settings(args)
    monkeypatch.delenv("EVA_GRPO_RECOVER_COMPLETED_INVALID_TOOL_HISTORY")
    assert historical["auto_compact_token_limit"] == 10240
    assert "context_profile" not in historical and "tool_output_token_limit" not in historical

    monkeypatch.setenv("EVA_GRPO_CONTEXT_PROFILE", MEASURED_CONTEXT_PROFILE)
    measured = native_settings(args)
    assert {name: measured[name] for name in (
        "max_context_tokens", "max_output_tokens", "total_output_budget",
        "auto_compact_token_limit", "context_reserve_tokens", "tool_output_token_limit",
    )} == {name: _settings()[name] for name in (
        "max_context_tokens", "max_output_tokens", "total_output_budget",
        "auto_compact_token_limit", "context_reserve_tokens", "tool_output_token_limit",
    )}
    assert measured["baseline_sampled_first_request_max_tokens"] == 14841
    assert measured["composed_sampled_first_request_max_tokens"] == 15963
    assert measured["sampled_first_request_is_not_domain_ceiling"] is True

    monkeypatch.setenv("EVA_GRPO_CONTEXT_PROFILE", EARLY_COMPACT_CONTEXT_PROFILE)
    early = native_settings(args)
    assert early == {
        **measured,
        "context_profile": EARLY_COMPACT_CONTEXT_PROFILE,
        "auto_compact_token_limit": EARLY_COMPACT_TOKENS,
    }
    assert early["max_context_tokens"] == 24576
    assert early["max_output_tokens"] == 4096
    assert early["total_output_budget"] == 8192
    assert early["context_reserve_tokens"] == 1024
    assert early["tool_output_token_limit"] == 2048


def test_measured_profile_requires_matched_slime_geometry(monkeypatch, tmp_path):
    import eva_agent.training.codex_slime_rollout as rollout

    binary = tmp_path / "codex"
    binary.write_text("fixture")
    binary.chmod(0o700)
    monkeypatch.setattr(rollout, "_binary_identity", lambda _: ("codex-cli 0.153.4", "a" * 64))
    monkeypatch.setenv("EVA_GRPO_CODEX_BIN", str(binary))
    monkeypatch.setenv("EVA_GRPO_CONTEXT_PROFILE", MEASURED_CONTEXT_PROFILE)
    with pytest.raises(ValueError, match="seq_length=24576"):
        native_settings(SimpleNamespace(seq_length=24576, rollout_max_response_len=4096))


def test_lossless_headroom_profile_covers_retained_compaction_prompt(monkeypatch, tmp_path):
    import eva_agent.training.codex_slime_rollout as rollout

    binary = tmp_path / "codex"
    binary.write_text("fixture")
    binary.chmod(0o700)
    monkeypatch.setattr(rollout, "_binary_identity", lambda _: ("codex-cli 0.153.4", "a" * 64))
    monkeypatch.setenv("EVA_GRPO_CODEX_BIN", str(binary))
    monkeypatch.setenv("EVA_GRPO_CONTEXT_PROFILE", LOSSLESS_HEADROOM_CONTEXT_PROFILE)

    settings = native_settings(SimpleNamespace(
        seq_length=LOSSLESS_HEADROOM_CONTEXT_TOKENS,
        rollout_max_response_len=8192,
    ))
    assert settings["max_context_tokens"] == 32768
    assert settings["auto_compact_token_limit"] == 16000
    assert settings["max_output_tokens"] == 4096
    assert settings["total_output_budget"] == 8192
    assert settings["tool_output_token_limit"] == 2048
    assert settings["measured_failed_compaction_prompt_max_tokens"] == 29684
    assert settings["measured_compaction_prompt_headroom_tokens"] == 3084
    assert settings["prompt_content_truncated_for_headroom"] is False

    with pytest.raises(ValueError, match="seq_length=32768"):
        native_settings(SimpleNamespace(seq_length=24576, rollout_max_response_len=8192))


def test_measured_profile_composes_actual_sdk_memory_and_preserves_contracts(tmp_path):
    from eva_agent.codex_runtime import CodexRole, CodexSandbox, CodexSkill, CodexThreadOptions
    from eva_agent.codex_runtime.research_memory import MEMORY_INSTRUCTIONS, SKILL_PATH
    from eva_agent.codex_runtime.supra import WORKFLOW_INSTRUCTIONS
    from eva_agent.pipeline.digests import blake3_bytes

    canonical_path = tmp_path / "canonical-skill.md"
    canonical_path.write_text("canonical skill")
    canonical = CodexSkill(skill_id="canonical", name="canonical", path=str(canonical_path),
                           content_blake3=blake3_bytes(canonical_path.read_bytes()))
    factory = _research_skills_factory(lambda request: (canonical,), _settings())
    selected = factory(object())
    assert selected[0] is canonical
    assert len(selected) == 2 and selected[1].skill_id == "summary_failures"
    assert Path(selected[1].path).read_bytes() == SKILL_PATH.read_bytes()

    base = CodexThreadOptions(role=CodexRole.STRONG_ACTOR, model="Qwen/Qwen3.5-9B",
        provider="eva_local_qwen", cwd=str(tmp_path), sandbox=CodexSandbox.READ_ONLY,
        config={"model_context_window": 24576, "model_auto_compact_token_limit": 19456},
        offered_tools=(), developer_instructions="canonical focus", ephemeral=True)
    actual = _research_thread_options(base, _settings(), endpoint="http://127.0.0.1:30910/v1")
    assert actual.model == base.model and actual.provider == base.provider
    assert actual.offered_tools == base.offered_tools and actual.sandbox is base.sandbox
    assert actual.config["model_context_window"] == 24576
    assert actual.config["model_auto_compact_token_limit"] == 19456
    assert actual.config["tool_output_token_limit"] == 2048
    assert actual.developer_instructions.count(MEMORY_INSTRUCTIONS) == 1
    assert actual.developer_instructions.count(WORKFLOW_INSTRUCTIONS) == 1

    early_settings = {
        **_settings(),
        "context_profile": EARLY_COMPACT_CONTEXT_PROFILE,
        "auto_compact_token_limit": EARLY_COMPACT_TOKENS,
    }
    early = _research_thread_options(base, early_settings, endpoint="http://127.0.0.1:30910/v1")
    assert early.config["model_context_window"] == 24576
    assert early.config["model_auto_compact_token_limit"] == 16000
    assert early.config["tool_output_token_limit"] == 2048


def test_sampled_first_request_measurement_is_not_an_admission_ceiling():
    record = segments()[0]
    record["prompt_token_ids"] = list(range(16000))
    record["output_token_ids"] = [16000]
    record["output_token_logprobs"] = [-0.1]
    record["loss_mask"] = [1]
    sample = build([record])[0]
    assert sample.metadata["prompt_token_count"] == 16000
    assert sample.response_length == 1 and sample.loss_mask == [1]


class Sample(SimpleNamespace):
    class Status(Enum):
        COMPLETED = "completed"
        TRUNCATED = "truncated"

    def append_response_tokens(self, args, *, tokens, log_probs, trainable, meta_info, update_terminal_info):
        assert trainable and update_terminal_info
        self.tokens += list(tokens)
        self.response_length += len(tokens)
        self.loss_mask += [1] * len(tokens)
        self.rollout_log_probs = list(log_probs)
        self.status = self.Status.TRUNCATED if meta_info["finish_reason"]["type"] == "length" else self.Status.COMPLETED
        self.received_meta_info = meta_info


def segments():
    return [dict(segment_index=0, prompt_token_ids=[10, 11, 12], output_token_ids=[40, 41],
                 output_token_logprobs=[-0.2, -1.3], loss_mask=[1, 1], request_id=str(uuid.uuid4()),
                 requested_model="Qwen/Qwen3.5-9B", returned_model=None,
                 meta_info={"finish_reason": {"type": "stop"}, "completion_tokens": 2}),
            dict(segment_index=1, prompt_token_ids=[91, 92], output_token_ids=[50],
                 output_token_logprobs=[-0.8], loss_mask=[1], request_id=str(uuid.uuid4()),
                 requested_model="Qwen/Qwen3.5-9B", returned_model=None,
                 meta_info={"finish_reason": {"type": "length"}, "completion_tokens": 1})]


def original(index=7, group=3):
    return Sample(index=index, group_index=group, metadata={"actor_backend": "native_codex_sglang", "judge_backend": "native_astra", "stage": "S1"})


def build(records=None, source=None):
    return samples_from_segments(SimpleNamespace(seq_length=24576), source or original(), records or segments(),
        max_requests=12, training_iteration=2, sample_id="trajectory", trajectory_path=Path("/private/trajectory.json"), sample_type=Sample)


def test_exact_discontiguous_prompts_never_concatenated_or_resupervised():
    source = original()
    before = deepcopy(source.__dict__)
    first, compacted = build(source=source)
    assert first.tokens == [10, 11, 12, 40, 41]
    assert first.response_length == 2 and first.loss_mask == [1, 1]
    assert first.rollout_log_probs == [-0.2, -1.3]
    assert compacted.tokens == [91, 92, 50]
    assert compacted.response_length == 1 and compacted.loss_mask == [1]
    assert first.rollout_id == compacted.rollout_id == 7
    assert first.group_index == compacted.group_index == 3
    assert (first.index, compacted.index) == (84, 85)
    assert first.metadata["training_iteration"] == 2
    assert first.metadata["returned_model_identity"] is None
    assert compacted.status == Sample.Status.TRUNCATED
    assert source.__dict__ == before


@pytest.mark.parametrize("field,value", [("loss_mask", [0, 1]), ("output_token_logprobs", [-0.1]),
                                         ("output_token_logprobs", [float("nan"), -1]),
                                         ("output_token_ids", [True, 3]), ("segment_index", 5)])
def test_invalid_exact_segment_fails_closed(field, value):
    records = segments()
    records[0][field] = value
    with pytest.raises(ValueError):
        build(records)


def test_shared_group_different_original_rollouts_have_no_segment_index_collision():
    a, b = build(source=original(index=1)), build(source=original(index=2))
    assert {s.index for s in a}.isdisjoint(s.index for s in b)
    assert {s.rollout_id for s in a} == {1} and {s.rollout_id for s in b} == {2}


def test_native_hook_requires_original_trajectory_reward_normalizer_before_any_runtime():
    with pytest.raises(ValueError, match="normalization"):
        generate_grpo_rollout(SimpleNamespace(custom_reward_post_process_path=None), 0, None)


@pytest.fixture
def native_fixture(monkeypatch, tmp_path):
    import eva_agent.codex_pipeline as pipeline
    import eva_agent.training.codex_sglang_transport as transport_module
    import eva_agent.training.teacher_worker as worker
    import eva_agent.training.teacher_focus as focus
    import eva_agent.training.slime_agent_judge as judge
    from training.automedbench_lite import local_qwen
    from eva_agent.training.qwen_tool_types import HostArgumentProjector
    monkeypatch.setattr(HostArgumentProjector, "from_context", lambda context: "fixture-bound-host-types")
    state = {"judge_failure": False, "actor_failure": False, "provider_calls": 0}
    records = segments()
    class Transport:
        def __init__(self, **kwargs):
            state["transport_options"] = kwargs
            self.segments = deepcopy(records)
            self.safe_metadata = {"segments": len(records), "generated_tokens": 3, "returned_model": None}
            kwargs["evidence_root"].mkdir(mode=0o700)
    class Adapter:
        def __init__(self, **kwargs):
            state["adapter_options"] = kwargs
            self.options = kwargs["options_factory"]
            self.skills = kwargs["skills_factory"]
            assert callable(kwargs["runtime_factory"])
            assert kwargs["turn_mcp_factory"] is context.turn_mcp_factory
    @contextmanager
    def setup(**kwargs):
        state["setup_options"] = kwargs
        assert kwargs["thinking"] is True and kwargs["exact_tool_schemas"] is True
        assert kwargs["normalize_priority_messages"] is True
        yield SimpleNamespace(provider="local", thread_config={
            "model_context_window": kwargs["context_length"],
            "model_auto_compact_token_limit": kwargs["auto_compact_token_limit"],
        }, backend=lambda: None, safe_metadata={},
            bind_canonical_tools=lambda offers: state.setdefault("canonical_offers", offers))
    def execute(**kwargs):
        state["provider_calls"] += 1
        state["workspace_root"] = kwargs["workspace_root"]
        request = SimpleNamespace(model=kwargs["target"])
        options = kwargs["provider"].options(request, str(tmp_path), (), None)
        state["thread_options"] = options
        state["selected_skill_ids"] = tuple(
            skill.skill_id for skill in kwargs["provider"].skills(request))
        if state["actor_failure"]:
            raise RuntimeError("fixture native terminal failure")
        return deepcopy(state.get("actor_document", {
            "provider_metadata": {"codex_turn_receipt": {"fixture": True}},
            "assistant_output": "visible only"}))
    async def grade(rollout, **kwargs):
        state["judge_options"] = kwargs
        state["judge_input"] = deepcopy(rollout)
        sample_root = kwargs["output_root"].parent
        files = sorted(sample_root.glob("sampled-policy-tokens-*.json"))
        assert len(files) == 2
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in files)
        assert all(json.loads(path.read_text())["reward_emitted"] is False for path in files)
        assert rollout["provider_metadata"]["native_codex_actor"] is True
        assert kwargs["backend"] == "native_astra"
        if state["judge_failure"]:
            raise RuntimeError("fixture Judge failure")
        return {"reward": state.get("judge_reward", 0.3333), "backend": "native_astra"}
    def canonical_skills(_request):
        return ()
    context = SimpleNamespace(turn_mcp_factory=object(), skills_factory=canonical_skills, rubric=object())
    monkeypatch.setattr(transport_module, "CodexSGLangTransport", Transport)
    monkeypatch.setattr(pipeline, "CodexRolloutAdapter", Adapter)
    monkeypatch.setattr(local_qwen, "local_qwen_setup", setup)
    monkeypatch.setattr(worker, "execute_single_rollout", execute)
    monkeypatch.setattr(focus, "focus_only_teacher_instructions", lambda _: "exact focus guidance")
    monkeypatch.setattr(judge, "resolve_judge_backend", lambda: "native_astra")
    monkeypatch.setattr(judge, "grade_rollout", grade)
    def run(sample_root=None, settings=None, recover_history=False, stage="S1"):
        settings = settings or {
            "max_context_tokens": 24576, "max_output_tokens": 4096,
            "total_output_budget": 4096, "max_requests": 12,
            "codex_bin": Path("/fixture/codex"), "codex_version": "codex-cli 0.153.4",
            "codex_binary_blake3": "fixture",
        }
        if recover_history:
            settings = {**settings, "recover_completed_invalid_tool_history": True}
        from eva_agent.pipeline import Stage
        context.episode = SimpleNamespace(stage=Stage(stage))
        context.stage_tool_guidance = None
        context.archived_predecessors = ()
        source = original()
        source.metadata["stage"] = stage
        return run_native_trajectory(SimpleNamespace(seq_length=24576, sglang_router_ip="127.0.0.1", sglang_router_port=30910),
            source, tokenizer=object(), sampling_params={"temperature": 0.8}, record={"stage": stage}, context=context,
            sample_root=sample_root or tmp_path / "sample", sample_id=str(uuid.uuid4()), training_iteration=0, sample_type=Sample,
            settings=settings)
    return run, state, tmp_path / "sample"


@pytest.mark.parametrize("recover_history", [False, True])
def test_native_trajectory_uses_real_adapter_contract_and_judges_once(native_fixture, recover_history):
    run, state, root = native_fixture
    result = run(recover_history=recover_history)
    assert state["setup_options"].get("recover_completed_invalid_tool_history", False) is recover_history
    assert state["provider_calls"] == 1
    assert state["canonical_offers"] == ()
    assert state["adapter_options"]["allow_schema_validation_rejections"] is True
    assert state["adapter_options"]["allow_legacy_empty_code_rejections"] is True
    assert [sample.reward for sample in result] == [0.3333, 0.3333]
    assert len(list(root.glob("training-tokens-*.json"))) == 2
    assert json.loads((root / "summary.json").read_text())["codex_turn_receipt_retained"] is True
    assert state["setup_options"]["auto_compact_token_limit"] == 10240
    assert state["thread_options"].developer_instructions == "exact focus guidance"
    assert state["selected_skill_ids"] == ()


def test_original_e2e_runs_one_whole_native_attempt_and_keeps_exact_sampled_segments(native_fixture):
    """Outer native runner fixture; external generation is mocked, not claimed."""
    run, state, root = native_fixture
    result = run(stage="E2E")
    instructions = state["thread_options"].developer_instructions
    assert "complete the original full medical research task" in instructions
    assert "no intermediate judge result" in instructions
    assert "exact focus guidance" not in instructions
    assert state["provider_calls"] == 1
    assert state["judge_input"]["provider_metadata"]["native_codex_actor"] is True
    assert [sample.tokens for sample in result] == [[10, 11, 12, 40, 41], [91, 92, 50]]
    assert [sample.loss_mask for sample in result] == [[1, 1], [1]]
    assert [sample.rollout_log_probs for sample in result] == [[-0.2, -1.3], [-0.8]]
    assert all(sample.metadata["stage"] == "E2E" and sample.reward == 0.3333 for sample in result)
    assert len(list(root.glob("training-tokens-*.json"))) == 2


def test_e2e_instructions_reject_stage_relabeling_and_borrowed_predecessors():
    from eva_agent.pipeline import Stage
    from eva_agent.training.teacher_batch import TeacherBatchError
    from eva_agent.training.teacher_focus import native_training_instructions

    context = SimpleNamespace(episode=SimpleNamespace(stage=Stage.S5),
        stage_tool_guidance=None, archived_predecessors=())
    with pytest.raises(TeacherBatchError, match="original full-workflow"):
        native_training_instructions(context, "E2E")
    context.episode.stage, context.archived_predecessors = Stage.E2E, ("fixture predecessor",)
    with pytest.raises(TeacherBatchError, match="borrow"):
        native_training_instructions(context, "E2E")


def test_native_group_retains_mixed_draw_on_all_four_original_trajectories(monkeypatch, tmp_path):
    import sys
    from types import ModuleType
    import eva_agent.training.codex_slime_rollout as module
    import eva_agent.training.teacher_worker as worker

    originals = [original() for _ in range(4)]
    sampling = {"policy": "target-mix-e2e-3-3-2-v1", "s_target": "S3", "bucket": "e2e",
        "cursor_before": 2, "cursor_after": 3, "sandbox_id": "fixture-full-task",
        "source_stage": "E2E", "dataset_metadata_blake3": "fixture"}
    for index, sample in enumerate(originals):
        sample.index, sample.group_index = 8 + index, 2
        sample.metadata.update(stage="E2E", sandbox_id="fixture-full-task", bulk_root=str(tmp_path),
            task_sampling=deepcopy(sampling))
    state = ModuleType("slime.rollout.sglang_rollout")
    state.GenerateState = lambda args: SimpleNamespace(tokenizer=object(), sampling_params={})
    monkeypatch.setitem(sys.modules, state.__name__, state)
    monkeypatch.setattr(module, "native_settings", lambda args: {})
    monkeypatch.setattr(module, "_CONTEXTS", SimpleNamespace(load=lambda record: (object(), None)))
    monkeypatch.setattr(worker, "load_bulk_record", lambda root, sandbox: {"stage": "E2E"})
    calls = []
    def run(args, source, **kwargs):
        calls.append((source.index, kwargs["record"]["stage"]))
        result = build(source=source)
        for segment in result:
            segment.reward = 0.5
        return result
    monkeypatch.setattr(module, "run_native_trajectory", run)
    args = SimpleNamespace(custom_reward_post_process_path=module.REWARD_HOOK,
        rollout_batch_size=1, n_samples_per_prompt=4, save=str(tmp_path / "checkpoints"))
    result = module.generate_grpo_rollout(args, 0, SimpleNamespace(get_samples=lambda count: [originals]))
    assert sorted(calls) == [(index, "E2E") for index in range(8, 12)]
    assert len(result) == 1 and len(result[0]) == 4
    assert all(len(trajectory) == 2 for trajectory in result[0])
    assert all(segment.metadata["task_sampling"] == sampling for trajectory in result[0] for segment in trajectory)
    summary = json.loads((tmp_path / "rollouts/000000/summary.json").read_text())
    assert summary["groups"][0]["task_sampling"] == sampling
    assert summary["original_trajectories"] == 4 and summary["segments"] == 8


def test_schema_rejection_and_owned_budget_stay_failed_in_judge_input(native_fixture):
    """Consumer-only fixture: adapter proof validation is tested in its own suite."""
    from eva_agent.codex_pipeline.budget import TERMINAL_MARKER, budget_metadata

    run, state, root = native_fixture
    original = {"provider_metadata": {
        "codex_turn_receipt": {"fixture": True, "status": "failed"},
        "schema_validation_rejections": [{"fixture": True,
            "host_execution_attempted": False, "host_result_claimed": False,
            "raw_arguments": {"code": "print('fixture')"}}],
        "controlled_budget_termination": budget_metadata({"kind": "total_output_budget",
            "count": 8192, "limit": 8192, "requests": 5}),
    }, "assistant_output": TERMINAL_MARKER,
        "messages": [{"fixture": True, "role": "tool", "content": {
            "status": "failed", "arguments": {"code": "print('fixture')"}}}]}
    state["actor_document"] = deepcopy(original)
    state["judge_reward"] = 0.0
    result = run()
    assert state["provider_calls"] == 1
    assert state["actor_document"] == original
    judged = state["judge_input"]
    for name, value in original["provider_metadata"].items():
        assert judged["provider_metadata"][name] == value
    assert judged["messages"] == original["messages"]
    assert judged["assistant_output"] == TERMINAL_MARKER
    assert all(sample.reward == 0.0 for sample in result)
    assert [sample.tokens for sample in result] == [[10, 11, 12, 40, 41], [91, 92, 50]]
    assert [sample.loss_mask for sample in result] == [[1, 1], [1]]
    assert [sample.rollout_log_probs for sample in result] == [[-0.2, -1.3], [-0.8]]
    assert len(list(root.glob("training-tokens-*.json"))) == 2


def test_native_trajectory_uses_opt_in_memory_skill_and_tool_output_profile(native_fixture):
    from eva_agent.codex_runtime.research_memory import MEMORY_INSTRUCTIONS
    from eva_agent.codex_runtime.supra import WORKFLOW_INSTRUCTIONS

    run, state, _ = native_fixture
    settings = {**_settings(), "max_requests": 12,
        "codex_bin": Path("/fixture/codex"), "codex_version": "codex-cli 0.153.4",
        "codex_binary_blake3": "fixture"}
    run(settings=settings)
    assert state["setup_options"]["auto_compact_token_limit"] == 19456
    options = state["thread_options"]
    assert options.config["tool_output_token_limit"] == 2048
    assert options.developer_instructions.count(MEMORY_INSTRUCTIONS) == 1
    assert options.developer_instructions.count(WORKFLOW_INSTRUCTIONS) == 1
    assert state["selected_skill_ids"] == ("summary_failures",)
    assert state["judge_options"]["material_view"] == "policy-visible-audit-v2"
    assert state["judge_options"]["native_turn_timeout_seconds"] == 600


def test_relative_owned_sample_root_binds_absolute_native_workspace(native_fixture, monkeypatch):
    run, state, root = native_fixture
    monkeypatch.chdir(root.parent)
    run(Path('sample'))
    assert state['workspace_root'] == root / 'workspace'
    assert state['workspace_root'].is_absolute()


def test_judge_failure_retains_private_exact_tokens_without_reward_fallback(native_fixture):
    run, state, root = native_fixture
    state["judge_failure"] = True
    with pytest.raises(RuntimeError, match="Judge"):
        run()
    assert len(list(root.glob("sampled-policy-tokens-*.json"))) == 2
    assert not list(root.glob("training-tokens-*.json"))
    failure = json.loads((root / "failure.json").read_text())
    assert failure["phase"] == "workspace_agent_judge" and failure["reward_emitted"] is False


def test_codex_terminal_failure_preserves_transport_failure_without_fabricated_rollout(native_fixture):
    run, state, root = native_fixture
    state["actor_failure"] = True
    with pytest.raises(RuntimeError, match="native"):
        run()
    assert not (root / "trajectory.json").exists()
    assert json.loads((root / "failure.json").read_text())["transport"]["segments"] == 2
