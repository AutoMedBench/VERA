"""Bounded multi-turn EvaMed harness loop."""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

from eva_agent.pipeline.digests import blake3_hex, canonical_value

from .contracts import HarnessContractError, HarnessTrajectory, ModelTurn
from .tools import ToolRegistry, VALID_STAGES


class ResponsesModel(Protocol):
    model_id: str

    async def create_turn(
        self,
        *,
        input_items: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        instructions: str | None = None,
    ) -> ModelTurn: ...


class EvaMedHarness:
    """Execute one answer-free sandbox trajectory with auditable tool groups."""

    def __init__(
        self,
        *,
        model: ResponsesModel,
        tools: ToolRegistry,
        max_turns: int = 64,
        max_parallel_tools: int = 32,
        tool_timeout_seconds: float = 300.0,
    ) -> None:
        if max_turns < 1 or max_parallel_tools < 1 or tool_timeout_seconds <= 0:
            raise HarnessContractError("harness limits must be positive")
        self.model = model
        self.tools = tools
        self.max_turns = max_turns
        self.max_parallel_tools = max_parallel_tools
        self.tool_timeout_seconds = tool_timeout_seconds

    async def run(
        self,
        *,
        initial_input: Sequence[Mapping[str, Any]],
        stage: str,
        instructions: str | None = None,
    ) -> HarnessTrajectory:
        if stage not in VALID_STAGES:
            raise HarnessContractError("stage must be S1-S5 or E2E")
        frozen_initial = tuple(canonical_value(copy.deepcopy(list(initial_input))))
        input_items: list[Mapping[str, Any]] = [copy.deepcopy(item) for item in frozen_initial]
        tool_schemas = self.tools.schemas_for_stage(stage)
        turns: list[ModelTurn] = []
        groups = []
        final_text = ""
        terminal = False
        for _ in range(self.max_turns):
            turn = await self.model.create_turn(
                input_items=input_items,
                tools=tool_schemas,
                instructions=instructions,
            )
            turns.append(turn)
            input_items.extend(copy.deepcopy(list(turn.output_items)))
            if not turn.function_calls:
                final_text = turn.output_text
                terminal = True
                break
            group = await self.tools.execute_group(
                turn.function_calls,
                stage=stage,
                max_parallel_tools=self.max_parallel_tools,
                timeout_seconds=self.tool_timeout_seconds,
            )
            groups.append(group)
            for observation in group.observations:
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": observation.provider_call_id,
                        "output": json.dumps(
                            {
                                "ok": observation.ok,
                                "result": observation.output,
                                "error_type": observation.error_type,
                                "receipt_blake3": observation.receipt_blake3,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                )
        run_id = str(uuid4())
        maximum = max((group.max_parallelism_observed for group in groups), default=0)
        core = {
            "run_id": run_id,
            "model_id": self.model.model_id,
            "stage": stage,
            "initial_input": frozen_initial,
            "turns": turns,
            "tool_groups": groups,
            "final_text": final_text,
            "terminal": terminal,
            "max_parallelism_observed": maximum,
        }
        return HarnessTrajectory(
            run_id=run_id,
            model_id=self.model.model_id,
            stage=stage,
            initial_input=frozen_initial,
            turns=tuple(turns),
            tool_groups=tuple(groups),
            final_text=final_text,
            terminal=terminal,
            max_parallelism_observed=maximum,
            trajectory_blake3=blake3_hex(core),
        )


__all__ = ["EvaMedHarness", "ResponsesModel"]
