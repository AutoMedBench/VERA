"""OpenAI Responses API adapter with explicit parallel function calling."""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Sequence
from uuid import uuid4

from eva_agent.pipeline.digests import blake3_hex, canonical_value

from .contracts import FunctionCall, HarnessContractError, ModelTurn


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class OpenAIResponsesModel:
    """Thin async adapter usable with OpenAI or compatible routed endpoints."""

    def __init__(
        self,
        *,
        model_id: str,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_output_tokens: int = 16_384,
    ) -> None:
        if not model_id or max_output_tokens < 1:
            raise HarnessContractError("model ID and output budget are required")
        if client is None:
            from openai import AsyncOpenAI

            kwargs: dict[str, Any] = {}
            if api_key is not None:
                kwargs["api_key"] = api_key
            if base_url is not None:
                kwargs["base_url"] = base_url
            client = AsyncOpenAI(**kwargs)
        self.client = client
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens

    async def create_turn(
        self,
        *,
        input_items: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        instructions: str | None = None,
    ) -> ModelTurn:
        request: dict[str, Any] = {
            "model": self.model_id,
            "input": copy.deepcopy(list(input_items)),
            "tools": copy.deepcopy(list(tools)),
            "parallel_tool_calls": True,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if instructions:
            request["instructions"] = instructions
        response = await self.client.responses.create(**request)
        raw_output = _plain(getattr(response, "output", ()))
        if not isinstance(raw_output, list):
            raise HarnessContractError("Responses output must be a list")
        output_items: tuple[Mapping[str, Any], ...] = tuple(
            item for item in raw_output if isinstance(item, Mapping)
        )
        if len(output_items) != len(raw_output):
            raise HarnessContractError("Responses output contains a non-object item")
        calls: list[FunctionCall] = []
        for item in output_items:
            if item.get("type") != "function_call":
                continue
            provider_call_id = item.get("call_id")
            name = item.get("name")
            arguments = item.get("arguments")
            if not isinstance(provider_call_id, str) or not provider_call_id:
                raise HarnessContractError("function call omitted call_id")
            if not isinstance(name, str) or not name:
                raise HarnessContractError("function call omitted name")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise HarnessContractError("function arguments are not JSON") from exc
            if not isinstance(arguments, Mapping):
                raise HarnessContractError("function arguments must be an object")
            calls.append(
                FunctionCall(
                    event_id=str(uuid4()),
                    provider_call_id=provider_call_id,
                    name=name,
                    arguments=canonical_value(arguments),
                )
            )
        usage = _plain(getattr(response, "usage", {})) or {}
        if not isinstance(usage, Mapping):
            usage = {}
        response_model = getattr(response, "model", None)
        response_id = getattr(response, "id", None)
        output_text = getattr(response, "output_text", "") or ""
        core = {
            "response_id": response_id,
            "model_id": response_model or self.model_id,
            "output_items": output_items,
            "usage": usage,
        }
        return ModelTurn(
            event_id=str(uuid4()),
            response_id=response_id if isinstance(response_id, str) else None,
            model_id=response_model if isinstance(response_model, str) else self.model_id,
            output_items=output_items,
            function_calls=tuple(calls),
            output_text=str(output_text),
            usage=canonical_value(usage),
            receipt_blake3=blake3_hex(core),
        )


__all__ = ["OpenAIResponsesModel"]
