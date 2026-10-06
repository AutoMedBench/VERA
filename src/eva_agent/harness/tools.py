"""Stage-aware tool registry with bounded independent parallel execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence
from uuid import uuid4

from jsonschema import Draft202012Validator

from eva_agent.pipeline.digests import blake3_hex, canonical_value

from .contracts import FunctionCall, HarnessContractError, ToolGroup, ToolObservation


ToolHandler = Callable[[Mapping[str, Any]], Any | Awaitable[Any]]
VALID_STAGES = frozenset({"S1", "S2", "S3", "S4", "S5", "E2E"})


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: ToolHandler
    allowed_stages: frozenset[str] = VALID_STAGES
    parallel_safe: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.description or not callable(self.handler):
            raise HarnessContractError("tool name, description, and handler are required")
        if not self.allowed_stages or not self.allowed_stages <= VALID_STAGES:
            raise HarnessContractError("tool stages must be a non-empty S1-S5/E2E subset")
        Draft202012Validator.check_schema(dict(self.parameters))

    def response_api_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": canonical_value(self.parameters),
            "strict": True,
        }


class ToolRegistry:
    """Offered tools are explicit; model-invented tools fail before execution."""

    def __init__(self, definitions: Sequence[ToolDefinition] = ()) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._definitions:
            raise HarnessContractError(f"duplicate tool: {definition.name}")
        self._definitions[definition.name] = definition

    def schemas_for_stage(self, stage: str) -> tuple[dict[str, Any], ...]:
        self._validate_stage(stage)
        return tuple(
            definition.response_api_schema()
            for _, definition in sorted(self._definitions.items())
            if stage in definition.allowed_stages
        )

    def _validate_stage(self, stage: str) -> None:
        if stage not in VALID_STAGES:
            raise HarnessContractError("stage must be S1-S5 or E2E")

    def _resolve(self, call: FunctionCall, stage: str) -> ToolDefinition:
        definition = self._definitions.get(call.name)
        if definition is None:
            raise HarnessContractError(f"model requested an unoffered tool: {call.name}")
        if stage not in definition.allowed_stages:
            raise HarnessContractError(
                f"tool {call.name} is not allowed during {stage}"
            )
        errors = sorted(
            Draft202012Validator(definition.parameters).iter_errors(
                dict(call.arguments)
            ),
            key=lambda error: tuple(str(part) for part in error.path),
        )
        if errors:
            raise HarnessContractError(
                f"tool arguments violate {call.name} schema: {errors[0].message}"
            )
        return definition

    async def execute_group(
        self,
        calls: Sequence[FunctionCall],
        *,
        stage: str,
        max_parallel_tools: int,
        timeout_seconds: float,
    ) -> ToolGroup:
        self._validate_stage(stage)
        if not calls:
            raise HarnessContractError("tool group cannot be empty")
        if max_parallel_tools < 1 or timeout_seconds <= 0:
            raise HarnessContractError("tool parallelism and timeout must be positive")
        provider_ids = [call.provider_call_id for call in calls]
        if len(set(provider_ids)) != len(provider_ids):
            raise HarnessContractError("provider tool call IDs must be unique per turn")
        definitions = [self._resolve(call, stage) for call in calls]
        parallel = len(calls) > 1 and all(item.parallel_safe for item in definitions)
        semaphore = asyncio.Semaphore(max_parallel_tools if parallel else 1)
        state_lock = asyncio.Lock()
        active = 0
        maximum = 0

        async def execute_one(
            call: FunctionCall, definition: ToolDefinition
        ) -> ToolObservation:
            nonlocal active, maximum
            started = time.monotonic()
            async with semaphore:
                async with state_lock:
                    active += 1
                    maximum = max(maximum, active)
                try:
                    if inspect.iscoroutinefunction(definition.handler):
                        result = await asyncio.wait_for(
                            definition.handler(call.arguments), timeout_seconds
                        )
                    else:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(definition.handler, call.arguments),
                            timeout_seconds,
                        )
                        if inspect.isawaitable(result):
                            result = await asyncio.wait_for(result, timeout_seconds)
                    result = canonical_value(result)
                    ok = True
                    error_type = None
                except asyncio.TimeoutError:
                    result = {"error": "tool_timeout"}
                    ok = False
                    error_type = "timeout"
                except Exception as exc:  # Preserve a safe typed observation.
                    result = {"error": "tool_execution_failed"}
                    ok = False
                    error_type = type(exc).__name__
                finally:
                    async with state_lock:
                        active -= 1
            duration_ms = (time.monotonic() - started) * 1000.0
            core = {
                "provider_call_id": call.provider_call_id,
                "tool_name": call.name,
                "ok": ok,
                "output": result,
                "error_type": error_type,
            }
            return ToolObservation(
                event_id=str(uuid4()),
                provider_call_id=call.provider_call_id,
                tool_name=call.name,
                ok=ok,
                output=result,
                error_type=error_type,
                duration_ms=duration_ms,
                receipt_blake3=blake3_hex(core),
            )

        if parallel:
            observations = tuple(
                await asyncio.gather(
                    *(execute_one(call, definition) for call, definition in zip(calls, definitions))
                )
            )
        else:
            sequential: list[ToolObservation] = []
            for call, definition in zip(calls, definitions):
                sequential.append(await execute_one(call, definition))
            observations = tuple(sequential)
        group_core = {
            "observations": observations,
            "parallel": parallel,
            "max_parallelism_observed": maximum,
        }
        return ToolGroup(
            event_id=str(uuid4()),
            observations=observations,
            parallel=parallel,
            max_parallelism_observed=maximum,
            receipt_blake3=blake3_hex(group_core),
        )


__all__ = ["ToolDefinition", "ToolHandler", "ToolRegistry", "VALID_STAGES"]
