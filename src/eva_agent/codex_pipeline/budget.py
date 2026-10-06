"""Explicit host-budget termination, distinct from model/infrastructure success."""

from typing import Any, Mapping


PROJECTION = "eva.codex-provider-rollout-projection.v3-controlled-budget"
TERMINAL_MARKER = "[host stopped trajectory at its declared rollout budget; not a model answer]"
WORKSPACE_PROJECTION = "eva.codex-provider-rollout-projection.v4-workspace-terminal"
WORKSPACE_TERMINAL_MARKER = "[host recorded a completed tool trajectory without a final text answer]"
WORKSPACE_TERMINAL = {
    "classification": "completed_without_final_text",
    "codex_turn_status": "completed",
    "synthetic_terminal_marker": True,
    "sft_eligible": False,
    "reward_requires_workspace_agent_judge": True,
}


def budget_details(value: Mapping[str, Any]) -> dict[str, Any]:
    """Allow only counters supplied by the trajectory's owned generation transport."""
    if not isinstance(value, Mapping):
        raise ValueError("controlled rollout budget record is not a mapping")
    other = {"max_requests": "generated_tokens", "total_output_budget": "requests"}.get(value.get("kind"))
    if other is None or set(value) != {"kind", "count", "limit", other}:
        raise ValueError("controlled rollout budget category or fields differ")
    if any(type(value[key]) is not int for key in ("count", "limit", other)):
        raise ValueError("controlled rollout budget counters must be integers")
    if value["limit"] <= 0 or value["count"] != value["limit"] or value[other] < 0:
        raise ValueError("controlled rollout budget was not exhausted exactly")
    return dict(value)


def budget_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "classification": "host_rollout_budget_exhausted",
        "budget": budget_details(value),
        "codex_turn_status": "failed",
        "synthetic_terminal_marker": True,
        "sft_eligible": False,
        "reward_requires_workspace_agent_judge": True,
    }
