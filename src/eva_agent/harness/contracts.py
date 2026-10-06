"""Public contracts for the EvaMed OpenAI-SDK harness.

Provider call identifiers are preserved for protocol continuity. EVA-owned
runtime records use UUIDs and every committed content receipt uses BLAKE3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class HarnessContractError(ValueError):
    """A model, tool, or trajectory crossed an invalid harness boundary."""


@dataclass(frozen=True)
class FunctionCall:
    event_id: str
    provider_call_id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ModelTurn:
    event_id: str
    response_id: str | None
    model_id: str
    output_items: tuple[Mapping[str, Any], ...]
    function_calls: tuple[FunctionCall, ...]
    output_text: str
    usage: Mapping[str, Any]
    receipt_blake3: str


@dataclass(frozen=True)
class ToolObservation:
    event_id: str
    provider_call_id: str
    tool_name: str
    ok: bool
    output: Any
    error_type: str | None
    duration_ms: float
    receipt_blake3: str


@dataclass(frozen=True)
class ToolGroup:
    event_id: str
    observations: tuple[ToolObservation, ...]
    parallel: bool
    max_parallelism_observed: int
    receipt_blake3: str


@dataclass(frozen=True)
class HarnessTrajectory:
    run_id: str
    model_id: str
    stage: str
    initial_input: tuple[Mapping[str, Any], ...]
    turns: tuple[ModelTurn, ...]
    tool_groups: tuple[ToolGroup, ...]
    final_text: str
    terminal: bool
    max_parallelism_observed: int
    trajectory_blake3: str


__all__ = [
    "FunctionCall",
    "HarnessContractError",
    "HarnessTrajectory",
    "ModelTurn",
    "ToolGroup",
    "ToolObservation",
]
