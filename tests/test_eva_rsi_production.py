"""CPU plan tests; none of these are evidence of medical evaluation or learning."""
from pathlib import Path

import pytest

from training.eva_rsi.production import loop_config, train_plan, NATIVE_GENERATOR, SEGMENT_REWARD


def context(tmp_path, stage="S1", steps=50):
    return {"phase": "train", "s_target": stage, "remaining_updates": steps,
        "attempt_root": str(tmp_path), "architecture_model_path": "/original-qwen",
        "checkpoint_root": "/completed-sft/checkpoints", "previous_evaluation": {"proposal": {
            "stages": {stage: {"domains": {"automedbench-classification": {"sample_count": 1},
                                         "unobserved": {"sample_count": 0}}}}}}}


def settings():
    return {"training_python": "/training/python", "control_python": "/control/python",
        "codex_bin": "/codex-0.153.4/bin/codex", "architecture_model_path": "/original-qwen",
        "initial_model_path": "/completed-sft/hf", "initial_checkpoint_root": "/completed-sft/checkpoints",
        "skill_catalog_id": "observed-catalog"}


@pytest.mark.parametrize("remaining", [1, 25, 49, 50])
def test_native_plan_keeps_remainder_model_architecture_and_segment_rewards(tmp_path, remaining):
    plan = train_plan(context(tmp_path, steps=remaining), settings())
    argv = plan["argv"]
    assert argv[argv.index("--steps") + 1] == str(remaining)
    assert argv[argv.index("--model") + 1] == "/original-qwen"
    assert argv[argv.index("--load") + 1] == "/completed-sft/checkpoints"
    assert argv[argv.index("--generation-function") + 1] == NATIVE_GENERATOR
    assert argv[argv.index("--reward-post-process-function") + 1] == SEGMENT_REWARD
    assert int(argv[argv.index("--save-interval") + 1]) <= 25
    assert plan["domains"] == ["automedbench-classification"]
    assert "--decode-full-graphs" in argv
    assert not list(tmp_path.iterdir())
    assert not plan["actor_execution_verified_by_plan"]


def test_unsupported_selected_stage_not_relabeled(tmp_path):
    with pytest.raises(ValueError, match="selected_stage_not_executable"):
        train_plan(context(tmp_path, stage="E2E"), settings())


@pytest.mark.parametrize("stage", ["S4", "S5"])
def test_late_target_plan_keeps_original_stage_and_native_actor(tmp_path, stage):
    plan = train_plan(context(tmp_path, stage=stage), settings())
    assert plan["stage"] == stage
    assert NATIVE_GENERATOR in plan["argv"]
    assert not plan["actor_execution_verified_by_plan"]


def test_graph_performance_configuration_can_be_explicitly_disabled(tmp_path):
    plan = train_plan(context(tmp_path), {**settings(), "decode_full_graphs": False})
    assert "--decode-full-graphs" not in plan["argv"]


def test_abort_save_is_explicit_and_does_not_change_round_or_loss_plan(tmp_path):
    old = train_plan(context(tmp_path, steps=49), settings())
    plan = train_plan(context(tmp_path, steps=49), {**settings(), "save_on_rollout_error": True})
    assert "--save-on-rollout-error" not in old['argv']
    assert plan['argv'] == [*old['argv'], '--save-on-rollout-error']
    assert plan['remaining_updates'] == plan['planned_chunk_updates'] == 49
    assert plan['domains'] == old['domains']
    with pytest.raises(ValueError, match='invalid_save_on_rollout_error'):
        train_plan(context(tmp_path), {**settings(), "save_on_rollout_error": 'true'})


def test_first_checkpoint_keeps_full_remaining_schedule_and_periodic_cadence(tmp_path):
    configured = {**settings(), "save_on_rollout_error": True}
    prior = train_plan(context(tmp_path, steps=48), configured)
    plan = train_plan(context(tmp_path, steps=48), {**configured, "checkpoint_first_update": True})
    assert plan['argv'] == [*prior['argv'], '--checkpoint-first-update']
    assert plan['remaining_updates'] == plan['planned_chunk_updates'] == 48
    assert plan['argv'][plan['argv'].index('--save-interval') + 1] == '25'
    assert 'same optimizer' in plan['checkpoint_policy']
    assert plan['domains'] == prior['domains']
    with pytest.raises(ValueError, match='first_checkpoint_requires_abort_save_driver'):
        train_plan(context(tmp_path), {**settings(), "checkpoint_first_update": True})
    with pytest.raises(ValueError, match='invalid_checkpoint_first_update_configuration'):
        train_plan(context(tmp_path), {**configured, "checkpoint_first_update": 'true'})


def test_group_isolation_setting_reaches_actual_launcher_only_when_enabled(monkeypatch, tmp_path):
    import sys
    from run_full_parameter import arguments, build_command
    from judge_group_isolation import GENERATION_FUNCTION

    configured = {**settings(), "save_on_rollout_error": True}
    old = train_plan(context(tmp_path, steps=19), configured)
    plan = train_plan(context(tmp_path, steps=19), {**configured, "judge_group_replacements": 1})
    assert plan['argv'] == [*old['argv'], '--judge-group-replacements', '1']
    assert plan['remaining_updates'] == plan['planned_chunk_updates'] == 19
    assert plan['judge_group_replacements'] == 1
    monkeypatch.setattr(sys, 'argv', plan['argv'][1:])
    parsed = arguments()
    command = build_command(parsed)
    assert parsed.judge_group_replacements == 1
    assert command[command.index('--rollout-function-path') + 1] == GENERATION_FUNCTION
    assert command[command.index('--num-rollout') + 1] == '19'
    assert train_plan(context(tmp_path, steps=19), {**configured, "judge_group_replacements": 0}) == old
    for value in (-1, 4, True, '1'):
        with pytest.raises(ValueError, match='invalid_judge_group_replacements'):
            train_plan(context(tmp_path), {**configured, "judge_group_replacements": value})
    with pytest.raises(ValueError, match='isolation_requires_abort_save'):
        train_plan(context(tmp_path), {**settings(), "judge_group_replacements": 1})


@pytest.mark.parametrize('profile', [
    'evamed-grpo-native-24576-v1',
    'evamed-grpo-native-24576-earlycompact-v2',
    'evamed-grpo-native-32768-lossless-headroom-v3',
])
def test_measured_grpo_context_profile_has_explicit_worker_environment(tmp_path, profile):
    old = train_plan(context(tmp_path), settings())
    assert 'context_profile_environment' not in old
    configured = {**settings(), 'grpo_context_profile': profile}
    plan = train_plan(context(tmp_path), configured)
    assert plan['context_profile_environment'] == {'EVA_GRPO_CONTEXT_PROFILE': profile}
    expected_context = (
        '32768'
        if profile == 'evamed-grpo-native-32768-lossless-headroom-v3'
        else '24576'
    )
    assert plan['argv'][plan['argv'].index('--max-tokens') + 1] == expected_context
    with pytest.raises(ValueError, match='requires_8192'):
        train_plan(context(tmp_path), {**configured, 'trajectory_output_budget': 4096})


def test_full_parameter_launcher_requires_profile_matched_context_geometry():
    import runpy

    module = runpy.run_path(str(
        Path(__file__).resolve().parents[1] / 'training/slime/run_full_parameter.py'
    ))
    validate = module['validate_native_context_profile']
    validate(24576, 8192, 'evamed-grpo-native-24576-earlycompact-v2')
    validate(32768, 8192, 'evamed-grpo-native-32768-lossless-headroom-v3')
    with pytest.raises(ValueError, match='matched context'):
        validate(24576, 8192, 'evamed-grpo-native-32768-lossless-headroom-v3')
    with pytest.raises(ValueError):
        validate(32768, 8192, 'unsupported-profile')


def test_historical_provisional_evaluation_cannot_start_training(tmp_path):
    import json
    from training.eva_rsi.production import require_real_evaluation
    from training.eva_rsi.evidence import commitment
    index = tmp_path / "old-index.json"
    index.write_text(json.dumps({"baseline_semantics": "existing diagnostic; provisional curriculum only"}))
    value = {**context(tmp_path), "previous_evaluation": {"index": commitment(index)}}
    with pytest.raises(ValueError, match="real_evaluation_required"):
        require_real_evaluation(value)
    assert not (tmp_path / "training").exists()


def test_baseline_gate_requires_actual_seven_workflows_not_high_scores(tmp_path):
    import json
    from training.eva_rsi.production import require_complete_baseline
    from training.automedbench_lite.track_adapter import BY_TRACK
    root = tmp_path / "track-rollouts"
    root.mkdir()
    rows = [{"track": track, "completed_requested_turns": True, "errors": []} for track in BY_TRACK]
    document = {"benchmark_run_root": str(tmp_path), "evaluation_mode": "full_single_pass"}
    summary = root / "summary.json"
    summary.write_text(json.dumps({"tracks": rows}))
    require_complete_baseline(document)  # No high-score or cascade requirement.
    summary.write_text(json.dumps({"tracks": rows[:-1]}))
    with pytest.raises(ValueError, match="complete_seven_track_baseline_required"):
        require_complete_baseline(document)
    rows[0]["errors"] = [{"error": "runner_failure"}]
    summary.write_text(json.dumps({"tracks": rows}))
    with pytest.raises(ValueError, match="complete_seven_track_baseline_required"):
        require_complete_baseline(document)


def test_supervisor_previous_evaluation_is_a_persisted_path(tmp_path):
    import json
    value = context(tmp_path)
    path = tmp_path / "evaluation-verification.json"
    path.write_text(json.dumps(value["previous_evaluation"]))
    value["previous_evaluation"] = str(path)
    assert train_plan(value, settings())["domains"] == ["automedbench-classification"]


def test_loop_wiring_has_exact_round_count_and_no_shell(tmp_path):
    config = loop_config(settings(), tmp_path / "settings.json")
    assert config["rounds"] == 10 and config["updates_per_round"] == 50
    assert all("--execute" in argv and "{context}" in argv for argv in config["commands"].values())
    assert not list(tmp_path.iterdir())


def test_first_durable_update_is_a_disclosed_prefix_not_a_shorter_round(tmp_path):
    initial = {**context(tmp_path), "round": 1}
    configured = {**settings(), "first_update_checkpoint_probe": True}
    plan = train_plan(initial, configured)
    assert plan["remaining_updates"] == 50 and plan["planned_chunk_updates"] == 1
    assert plan["argv"][plan["argv"].index("--steps") + 1] == "1"
    continuation = train_plan({**initial, "remaining_updates": 49}, configured)
    assert continuation["planned_chunk_updates"] == 49
    assert continuation["first_update_checkpoint_probe"] is False
    assert train_plan({**initial, "round": 2}, configured)["planned_chunk_updates"] == 50


def test_prefix_verifier_requires_actual_one_update_and_explicit_plan(tmp_path):
    import json
    from training.eva_rsi.evidence import verify_training, commitment

    root = tmp_path / "training"
    root.mkdir()
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"metadata": {"stage": "S1", "judge_backend": "native_astra"}}))
    receipt = {"stage": "grpo", "judge_backend": "native_astra", "status": "synthetic_fixture",
        "argv": ["--use-rollout-logprobs", "--num-steps-per-rollout", "1", "--num-rollout", "1",
                 "--hf-checkpoint", "/original-qwen", "--load", "/completed-sft/checkpoints",
                 "--save", str(root / "checkpoints"), "--prompt-data", str(data)],
        "training_data_blake3": commitment(data)["blake3"]}
    (root / "run-receipt.json").write_text(json.dumps(receipt))
    row = {"event": "optimizer_step", "rollout_id": 0, "step_id": 0,
        "trainable_parameters": 8953803264, "gradient_norm": 0.5,
        "sampled_changed_values": 1, "successful_update": True}
    (root / "optimizer-steps.jsonl").write_text(json.dumps(row))
    plan = {"planned_chunk_updates": 1, "remaining_updates": 50, "first_update_checkpoint_probe": True}
    (tmp_path / "training-plan.json").write_text(json.dumps(plan))
    ctx = {**context(tmp_path), "round": 1}
    assert verify_training(root, ctx)["executions"] == 1
    (tmp_path / "training-plan.json").write_text(json.dumps({**plan, "first_update_checkpoint_probe": False}))
    with pytest.raises(ValueError, match="unplanned_training_prefix"):
        verify_training(root, ctx)
