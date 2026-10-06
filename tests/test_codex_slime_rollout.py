from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from eva_agent.training.codex_slime_rollout import (
    samples_from_segments, run_native_trajectory, generate_grpo_rollout, native_compaction_limit,
)


@pytest.mark.parametrize("context,expected", [(8192, 3072), (16384, 10240), (24576, 10240)])
def test_native_compacts_before_measured_transcript_expansion(context, expected):
    assert native_compaction_limit(context) == expected
    assert expected + 4096 + 1024 <= context


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
            self.options = kwargs["options_factory"]
            assert callable(kwargs["runtime_factory"])
            assert kwargs["turn_mcp_factory"] is context.turn_mcp_factory
            assert kwargs["skills_factory"] is context.skills_factory
    @contextmanager
    def setup(**kwargs):
        state["setup_options"] = kwargs
        assert kwargs["thinking"] is True and kwargs["exact_tool_schemas"] is True
        assert kwargs["normalize_priority_messages"] is True
        assert kwargs["auto_compact_token_limit"] == 10240
        yield SimpleNamespace(provider="local", thread_config={}, backend=lambda: None, safe_metadata={})
    def execute(**kwargs):
        state["provider_calls"] += 1
        state["workspace_root"] = kwargs["workspace_root"]
        request = SimpleNamespace(model=kwargs["target"])
        options = kwargs["provider"].options(request, str(tmp_path), (), None)
        assert options.developer_instructions == "exact focus guidance"
        if state["actor_failure"]:
            raise RuntimeError("fixture native terminal failure")
        return {"provider_metadata": {"codex_turn_receipt": {"fixture": True}}, "assistant_output": "visible only"}
    async def grade(rollout, **kwargs):
        sample_root = kwargs["output_root"].parent
        files = sorted(sample_root.glob("sampled-policy-tokens-*.json"))
        assert len(files) == 2
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in files)
        assert all(json.loads(path.read_text())["reward_emitted"] is False for path in files)
        assert rollout["provider_metadata"]["native_codex_actor"] is True
        assert kwargs["backend"] == "native_astra"
        if state["judge_failure"]:
            raise RuntimeError("fixture Judge failure")
        return {"reward": 0.3333, "backend": "native_astra"}
    context = SimpleNamespace(turn_mcp_factory=object(), skills_factory=object(), rubric=object())
    monkeypatch.setattr(transport_module, "CodexSGLangTransport", Transport)
    monkeypatch.setattr(pipeline, "CodexRolloutAdapter", Adapter)
    monkeypatch.setattr(local_qwen, "local_qwen_setup", setup)
    monkeypatch.setattr(worker, "execute_single_rollout", execute)
    monkeypatch.setattr(focus, "focus_only_teacher_instructions", lambda _: "exact focus guidance")
    monkeypatch.setattr(judge, "resolve_judge_backend", lambda: "native_astra")
    monkeypatch.setattr(judge, "grade_rollout", grade)
    def run(sample_root=None):
        return run_native_trajectory(SimpleNamespace(seq_length=24576, sglang_router_ip="127.0.0.1", sglang_router_port=30910),
            original(), tokenizer=object(), sampling_params={"temperature": 0.8}, record={"stage": "S1"}, context=context,
            sample_root=sample_root or tmp_path / "sample", sample_id=str(uuid.uuid4()), training_iteration=0, sample_type=Sample,
            settings={"max_context_tokens": 24576, "max_output_tokens": 4096, "total_output_budget": 4096,
                      "max_requests": 12, "codex_bin": Path("/fixture/codex"), "codex_version": "codex-cli 0.153.4",
                      "codex_binary_blake3": "fixture"})
    return run, state, tmp_path / "sample"


def test_native_trajectory_uses_real_adapter_contract_and_judges_once(native_fixture):
    run, state, root = native_fixture
    result = run()
    assert state["provider_calls"] == 1
    assert [sample.reward for sample in result] == [0.3333, 0.3333]
    assert len(list(root.glob("training-tokens-*.json"))) == 2
    assert json.loads((root / "summary.json").read_text())["codex_turn_receipt_retained"] is True


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
