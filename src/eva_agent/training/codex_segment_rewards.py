"""GRPO reward normalization once per original Codex trajectory, not segment.

Installed Slime hook signature: ``hook(args, flattened_samples) -> (raw, rewards)``.
Each exact provider-context segment has a unique ``Sample.index``; siblings from
one original trajectory share ``rollout_id`` and ``group_index``. Compaction may
change segment counts but must not change a trajectory's weight in the GRPO
reward mean/standard deviation. Slime's existing per-rollout token-loss reducer
handles the separate loss aggregation using those same rollout IDs.
"""
from __future__ import annotations

import json
import logging
import math
from numbers import Real
from typing import Any, Sequence


LOGGER = logging.getLogger(__name__)
EPSILON = 1e-6  # Matches the installed Slime GRPO post-processing denominator.


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def normalize_segment_rewards_with_audit(args: Any, samples: Sequence[Any]):
    """Return raw per-segment rewards, broadcast GRPO rewards, and a small audit.

    No sample field, actual reward, mask, token, or log probability is modified.
    Statistics use float64 Python arithmetic and sample std (correction=1), the
    same estimator as Slime's float32 torch.std. Numerical roundoff may differ.
    The loss pipeline converts broadcast values to its normal training dtype.
    """
    if args.advantage_estimator != "grpo":
        raise ValueError("The native Codex segment reward hook requires GRPO")
    expected_rollouts = _integer(args.n_samples_per_prompt, "n_samples_per_prompt", minimum=2)
    expected_groups = _integer(args.rollout_batch_size, "rollout_batch_size", minimum=1)
    if not samples:
        raise ValueError("Cannot normalize an empty Codex rollout batch")

    groups: dict[int, dict[int, dict[str, Any]]] = {}
    rollout_group: dict[int, int] = {}
    indices: set[int] = set()
    identities: list[tuple[int, int]] = []
    raw: list[float] = []
    for sample in samples:
        index = _integer(sample.index, "Sample.index")
        rollout = _integer(sample.rollout_id, "Sample.rollout_id")
        group = _integer(sample.group_index, "Sample.group_index")
        if index in indices:
            raise ValueError("Codex segment Sample.index must be unique within the batch")
        indices.add(index)
        if rollout in rollout_group and rollout_group[rollout] != group:
            raise ValueError("A Codex trajectory cannot belong to multiple GRPO prompt groups")
        rollout_group[rollout] = group
        value = sample.get_reward_value(args)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError("Every Codex trajectory requires an actual finite numeric reward")
        reward = float(value)
        trajectory = groups.setdefault(group, {}).setdefault(rollout, {"reward": reward, "segments": 0})
        if trajectory["reward"] != reward:
            raise ValueError("Segments of one Codex trajectory disagree on its actual reward")
        trajectory["segments"] += 1
        identities.append((group, rollout))
        raw.append(reward)

    if len(groups) != expected_groups:
        raise ValueError("Observed Codex prompt groups differ from rollout_batch_size")
    processed: dict[tuple[int, int], float] = {}
    group_audits = []
    normalize = bool(args.rewards_normalization)
    use_std = bool(args.grpo_std_normalization)
    for group, trajectories in groups.items():
        if len(trajectories) != expected_rollouts:
            raise ValueError("Each Codex group must contain exactly n_samples_per_prompt original trajectories")
        rewards = [trajectory["reward"] for trajectory in trajectories.values()]
        mean = math.fsum(rewards) / len(rewards)
        centered = [value - mean for value in rewards]
        std = math.sqrt(math.fsum(value * value for value in centered) / (len(rewards) - 1))
        zero_variance = all(value == rewards[0] for value in rewards)
        if not math.isfinite(mean) or not math.isfinite(std):
            raise ValueError("Codex reward statistics are not finite")
        for (rollout, trajectory), centered_reward in zip(trajectories.items(), centered, strict=True):
            if not normalize:
                normalized = trajectory["reward"]
            elif zero_variance:
                normalized = 0.0
            elif use_std:
                normalized = centered_reward / (std + EPSILON)
            else:
                normalized = centered_reward
            processed[(group, rollout)] = normalized
        group_audits.append({
            "group_index": group,
            "original_trajectories": len(trajectories),
            "segments": sum(item["segments"] for item in trajectories.values()),
            "trajectory_segment_counts": {str(key): value["segments"] for key, value in trajectories.items()},
            "trajectory_rewards": {str(key): value["reward"] for key, value in trajectories.items()},
            "mean_reward_once_per_trajectory": mean,
            "sample_std_once_per_trajectory": std,
            "zero_variance_group": zero_variance,
            "zero_advantage_group": normalize and zero_variance,
        })
    audit = {"schema": "eva.codex-segment-grpo-normalization.v1", "segment_count": len(samples),
             "original_trajectory_count": len(rollout_group), "group_count": len(groups),
             "rewards_normalization": normalize, "std_normalization": use_std,
             "sample_std_correction": 1, "epsilon": EPSILON,
             "reward_or_sample_mutation": False, "groups": group_audits}
    return raw, [processed[identity] for identity in identities], audit


def normalize_segment_rewards(args: Any, samples: Sequence[Any]):
    """Slime --custom-reward-post-process-path entry point."""
    raw, normalized, audit = normalize_segment_rewards_with_audit(args, samples)
    LOGGER.info("EVA_CODEX_SEGMENT_REWARDS %s", json.dumps(audit, sort_keys=True))
    return raw, normalized
