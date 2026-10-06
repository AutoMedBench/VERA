from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from types import SimpleNamespace

from eva_agent.harness import EvaMedHarness, OpenAIResponsesModel, ToolDefinition, ToolRegistry


@dataclass
class _FakeResponse:
    id: str
    model: str
    output: list[dict]
    output_text: str = ""
    usage: dict | None = None


class _FakeResponses:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def create(self, **request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return _FakeResponse(
                id="response-1",
                model="test-model",
                output=[
                    {
                        "type": "function_call",
                        "call_id": "provider-call-a",
                        "name": "inspect_a",
                        "arguments": '{"value":1}',
                    },
                    {
                        "type": "function_call",
                        "call_id": "provider-call-b",
                        "name": "inspect_b",
                        "arguments": '{"value":2}',
                    },
                ],
                usage={"input_tokens": 10, "output_tokens": 5},
            )
        return _FakeResponse(
            id="response-2",
            model="test-model",
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
            output_text="done",
            usage={"input_tokens": 20, "output_tokens": 2},
        )


def test_independent_multi_tool_calls_execute_in_parallel_and_stay_grouped() -> None:
    active = 0
    maximum = 0
    both_started = asyncio.Event()

    async def handler(arguments):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        if active == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.5)
        await asyncio.sleep(0)
        active -= 1
        return {"value": arguments["value"]}

    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    tools = ToolRegistry(
        [
            ToolDefinition("inspect_a", "Inspect source A.", schema, handler),
            ToolDefinition("inspect_b", "Inspect source B.", schema, handler),
        ]
    )
    responses = _FakeResponses()
    model = OpenAIResponsesModel(
        model_id="test-model", client=SimpleNamespace(responses=responses)
    )
    harness = EvaMedHarness(model=model, tools=tools, max_parallel_tools=8)
    trajectory = asyncio.run(
        harness.run(
            initial_input=[{"role": "user", "content": "inspect both"}],
            stage="S2",
        )
    )

    assert trajectory.terminal is True
    assert trajectory.final_text == "done"
    assert trajectory.max_parallelism_observed == 2
    assert maximum == 2
    assert len(trajectory.tool_groups) == 1
    assert trajectory.tool_groups[0].parallel is True
    assert [item.provider_call_id for item in trajectory.tool_groups[0].observations] == [
        "provider-call-a",
        "provider-call-b",
    ]
    assert responses.requests[0]["parallel_tool_calls"] is True
    outputs = [
        item
        for item in responses.requests[1]["input"]
        if item.get("type") == "function_call_output"
    ]
    assert [item["call_id"] for item in outputs] == [
        "provider-call-a",
        "provider-call-b",
    ]
    assert [json.loads(item["output"])["result"]["value"] for item in outputs] == [1, 2]
