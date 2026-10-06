from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from eva_agent.training.codex_segment_rewards import normalize_segment_rewards, normalize_segment_rewards_with_audit


class Sample(SimpleNamespace):
    def get_reward_value(self, args):
        return self.reward if not args.reward_key else self.reward[args.reward_key]


def args(**kwargs):
    return SimpleNamespace(**{**dict(advantage_estimator="grpo", n_samples_per_prompt=2, rollout_batch_size=1,
                                    rewards_normalization=True, grpo_std_normalization=True, reward_key=None), **kwargs})


def samples(rewards=(0.0, 1.0), counts=(3, 1), group=0, start=0, rollout_start=0):
    result = []
    for offset, (reward, count) in enumerate(zip(rewards, counts, strict=True)):
        for _ in range(count):
            result.append(Sample(index=start + len(result), rollout_id=rollout_start + offset, group_index=group,
                                 reward=reward, tokens=[1, 2], response_length=1, loss_mask=[1], rollout_log_probs=[-0.5]))
    return result


def test_unequal_segment_counts_do_not_bias_original_trajectory_rewards():
    batch = samples()
    before = deepcopy([s.__dict__ for s in batch])
    raw, normalized, audit = normalize_segment_rewards_with_audit(args(), batch)
    magnitude = 0.5 / (math.sqrt(0.5) + 1e-6)
    assert raw == [0, 0, 0, 1]
    assert normalized == pytest.approx([-magnitude, -magnitude, -magnitude, magnitude])
    assert audit["groups"][0]["mean_reward_once_per_trajectory"] == 0.5
    assert audit["original_trajectory_count"] == 2
    assert [s.__dict__ for s in batch] == before


def test_independent_groups_with_interleaved_segments_and_shared_original_identity():
    first = samples(rewards=(0.0, 1.0), counts=(1, 3))
    second = samples(rewards=(0.4, 0.8), counts=(2, 1), group=1, start=4, rollout_start=2)
    batch = [first[0], second[0], first[1], second[1], first[2], second[2], first[3]]
    _, norm, audit = normalize_segment_rewards_with_audit(args(rollout_batch_size=2, grpo_std_normalization=False), batch)
    assert norm == pytest.approx([-0.5, -0.2, 0.5, -0.2, 0.5, 0.2, 0.5])
    assert audit["group_count"] == 2


def test_zero_variance_stays_zero_without_reward_changes():
    raw, norm, audit = normalize_segment_rewards_with_audit(args(), samples((0.1667, 0.1667), (4, 1)))
    assert raw == [0.1667] * 5
    assert norm == [0.0] * 5
    assert audit["groups"][0]["zero_variance_group"] is True
    assert audit["groups"][0]["zero_advantage_group"] is True


def test_no_normalization_preserves_actual_reward_and_reports_zero_variance():
    raw, norm, audit = normalize_segment_rewards_with_audit(args(rewards_normalization=False), samples((0.5, 0.5)))
    assert raw == norm == [0.5] * 4
    assert audit["groups"][0]["zero_advantage_group"] is False


def test_reward_key_and_exact_hook_pair_shape():
    batch = samples(({"score": 0.0}, {"score": 1.0}))
    result = normalize_segment_rewards(args(reward_key="score"), batch)
    assert len(result) == 2 and result[0] == [0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize("field,value", [("index", None), ("index", True), ("rollout_id", None), ("group_index", -1),
                                         ("reward", float("nan")), ("reward", float("inf")), ("reward", "0.5"), ("reward", True)])
def test_invalid_segment_identity_or_reward_fails(field, value):
    batch = samples()
    setattr(batch[0], field, value)
    with pytest.raises(ValueError):
        normalize_segment_rewards(args(), batch)


def test_duplicate_segment_index_fails():
    batch = samples()
    batch[1].index = batch[0].index
    with pytest.raises(ValueError, match="unique"):
        normalize_segment_rewards(args(), batch)


def test_inconsistent_reward_within_original_trajectory_fails():
    batch = samples()
    batch[1].reward = 0.3
    with pytest.raises(ValueError, match="disagree"):
        normalize_segment_rewards(args(), batch)


def test_original_trajectory_cannot_cross_groups():
    batch = samples()
    batch[1].group_index = 1
    with pytest.raises(ValueError, match="multiple"):
        normalize_segment_rewards(args(rollout_batch_size=2), batch)


@pytest.mark.parametrize("options", [dict(n_samples_per_prompt=4), dict(rollout_batch_size=2), dict(n_samples_per_prompt=1), dict(advantage_estimator="ppo")])
def test_incomplete_or_wrong_groups_fail_closed(options):
    with pytest.raises(ValueError):
        normalize_segment_rewards(args(**options), samples())
