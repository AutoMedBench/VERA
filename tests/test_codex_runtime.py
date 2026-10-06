from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import json
from types import SimpleNamespace
from typing import Any, AsyncIterator, Mapping, Sequence

import pytest

from eva_agent.codex_runtime import (
    BackendInputItem,
    BackendTurnOptions,
    CodexLaunchOptions,
    CodexRole,
    CodexRuntime,
    CodexRuntimeError,
    CodexSandbox,
    CodexSdkBindings,
    CodexSkill,
    CodexThreadOptions,
    CodexToolOffer,
    CodexTurnInput,
    OpenAICodexBackend,
    PersistentCodexRuntimeRunner,
    verify_codex_turn_receipt,
)
from eva_agent.codex_runtime.runtime import _new_exact_construction_input_policy
from eva_agent.harness.skills import SkillCatalog, SkillDocument
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes, canonical_value
from eva_agent.pipeline.ids import DeterministicUUIDFactory
from eva_agent.pipeline.tools import ToolDefinition
from eva_agent.training.teacher_worker import teacher_safe_actor_instruction


@dataclass
class _Notification:
    method: str
    payload: Mapping[str, Any]


class _FakeTurn:
    def __init__(
        self,
        turn_id: str,
        notifications: Sequence[_Notification],
        backend: "_FakeBackend",
    ) -> None:
        self.id = turn_id
        self._notifications = notifications
        self._backend = backend

    async def stream(self) -> AsyncIterator[_Notification]:
        self._backend.active_streams += 1
        self._backend.max_active_streams = max(
            self._backend.max_active_streams, self._backend.active_streams
        )
        try:
            for notification in self._notifications:
                await asyncio.sleep(0)
                yield notification
        finally:
            self._backend.active_streams -= 1


class _FakeThread:
    def __init__(self, thread_id: str, backend: "_FakeBackend") -> None:
        self.id = thread_id
        self._backend = backend

    async def turn(
        self,
        items: Sequence[BackendInputItem],
        options: BackendTurnOptions,
    ) -> _FakeTurn:
        self._backend.turn_inputs.append((self.id, tuple(items), options))
        turn_id = f"turn-{self._backend.turn_count}"
        self._backend.turn_count += 1
        notifications = self._backend.script(self.id, turn_id)
        return _FakeTurn(turn_id, notifications, self._backend)


class _FakeBackend:
    sdk_version = "fake-sdk-1"
    server_version = "fake-app-server-1"

    def __init__(self, script=None) -> None:
        self.opened = False
        self.closed = False
        self.started: list[CodexThreadOptions] = []
        self.resumed: list[tuple[str, CodexThreadOptions]] = []
        self.turn_inputs: list[tuple[str, tuple[BackendInputItem, ...], BackendTurnOptions]] = []
        self.turn_count = 0
        self.active_streams = 0
        self.max_active_streams = 0
        self.script = script or _simple_script

    async def open(self) -> None:
        self.opened = True

    async def close(self) -> None:
        self.closed = True

    async def start_thread(self, options: CodexThreadOptions) -> _FakeThread:
        self.started.append(options)
        return _FakeThread(f"thread-{len(self.started)}", self)

    async def resume_thread(self, thread_id: str, options: CodexThreadOptions) -> _FakeThread:
        self.resumed.append((thread_id, options))
        return _FakeThread(thread_id, self)


def _turn_event(method: str, thread_id: str, turn_id: str, **extra: Any) -> _Notification:
    return _Notification(method, {"threadId": thread_id, "turnId": turn_id, **extra})


def _simple_script(thread_id: str, turn_id: str) -> tuple[_Notification, ...]:
    return (
        _turn_event("turn/started", thread_id, turn_id, turn={"id": turn_id, "status": "inProgress"}),
        _turn_event(
            "item/completed",
            thread_id,
            turn_id,
            item={"id": "answer", "type": "agentMessage", "phase": "final_answer", "text": "done"},
        ),
        _turn_event(
            "turn/completed",
            thread_id,
            turn_id,
            turn={"id": turn_id, "status": "completed", "items": []},
        ),
    )


def _parallel_script(thread_id: str, turn_id: str) -> tuple[_Notification, ...]:
    first_started = {
        "id": "mcp-1",
        "type": "mcpToolCall",
        "server": "evamed",
        "tool": "search",
        "arguments": {"query": "trial"},
        "status": "inProgress",
    }
    second_started = {
        "id": "mcp-2",
        "type": "mcpToolCall",
        "server": "evamed",
        "tool": "read",
        "arguments": {"path": "paper.txt"},
        "status": "inProgress",
    }
    first_done = {
        **first_started,
        "status": "completed",
        "result": {"content": [{"type": "text", "text": "one"}]},
    }
    second_done = {
        **second_started,
        "status": "completed",
        "result": {"content": [{"type": "text", "text": "two"}]},
    }
    return (
        _turn_event("turn/started", thread_id, turn_id, turn={"id": turn_id, "status": "inProgress"}),
        _turn_event(
            "item/started",
            thread_id,
            turn_id,
            item={"id": "private-echo", "type": "userMessage", "content": "never commit me"},
        ),
        _turn_event(
            "item/reasoning/textDelta",
            thread_id,
            turn_id,
            itemId="reasoning-1",
            delta="hidden chain of thought",
        ),
        _turn_event("item/started", thread_id, turn_id, item=first_started),
        _turn_event("item/started", thread_id, turn_id, item=second_started),
        _turn_event("item/completed", thread_id, turn_id, item=second_done),
        _turn_event("item/completed", thread_id, turn_id, item=first_done),
        _turn_event(
            "thread/tokenUsage/updated",
            thread_id,
            turn_id,
            tokenUsage={"total": {"inputTokens": 100, "outputTokens": 25, "totalTokens": 125}},
        ),
        _turn_event(
            "item/completed",
            thread_id,
            turn_id,
            item={"id": "answer", "type": "agentMessage", "phase": "final_answer", "text": "verified"},
        ),
        _turn_event(
            "turn/completed",
            thread_id,
            turn_id,
            turn={"id": turn_id, "status": "completed", "items": []},
        ),
    )


def _core_resource_script(
    *,
    server: str,
    tool: str,
    arguments: Mapping[str, Any],
    status: str,
    result: Any,
    error: Any,
):
    def script(thread_id: str, turn_id: str) -> tuple[_Notification, ...]:
        started = {
            "id": "core-resource",
            "type": "mcpToolCall",
            "server": server,
            "tool": tool,
            "arguments": arguments,
            "status": "inProgress",
            "result": None,
            "error": None,
        }
        completed = {
            **started,
            "status": status,
            "result": result,
            "error": error,
            "durationMs": 1,
        }
        return (
            _turn_event(
                "turn/started",
                thread_id,
                turn_id,
                turn={"id": turn_id, "status": "inProgress"},
            ),
            _turn_event("item/started", thread_id, turn_id, item=started),
            _turn_event("item/completed", thread_id, turn_id, item=completed),
            _turn_event(
                "item/completed",
                thread_id,
                turn_id,
                item={
                    "id": "answer",
                    "type": "agentMessage",
                    "phase": "final_answer",
                    "text": "done",
                },
            ),
            _turn_event(
                "turn/completed",
                thread_id,
                turn_id,
                turn={"id": turn_id, "status": "completed", "items": []},
            ),
        )

    return script


def _offers() -> tuple[CodexToolOffer, CodexToolOffer]:
    return (
        CodexToolOffer(
            fully_qualified_name="evamed/search",
            description="Search evidence.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        CodexToolOffer(
            fully_qualified_name="evamed/read",
            description="Read evidence.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
    )


def _actor_options(**overrides: Any) -> CodexThreadOptions:
    values = {
        "role": CodexRole.STRONG_ACTOR,
        "model": "gpt-5.6-sol",
        "provider": "openai",
        "cwd": "/tmp/evamed-fixture",
        "sandbox": CodexSandbox.WORKSPACE_WRITE,
        "config": {"mcp_servers": {"evamed": {"bearer_token_env_var": "EVAMED_TOKEN"}}},
        "offered_tools": _offers(),
    }
    values.update(overrides)
    return CodexThreadOptions(**values)


def test_actor_receipt_captures_parallel_mcp_usage_skills_and_redacts_reasoning(tmp_path: Path) -> None:
    async def scenario():
        backend = _FakeBackend(_parallel_script)
        runtime = CodexRuntime(backend, id_factory=DeterministicUUIDFactory("codex-actor"))
        skill_path = tmp_path / "medical-literature-review" / "SKILL.md"
        skill_path.parent.mkdir()
        skill_path.write_text("original skill bytes")
        skill = CodexSkill(
            skill_id="medical-literature-review",
            name="medical-literature-review",
            path=str(skill_path.resolve()),
            content_blake3=blake3_bytes(b"original skill bytes"),
        )
        async with runtime:
            handle = await runtime.start_thread(_actor_options())
            receipt = await runtime.run_turn(
                handle,
                CodexTurnInput(
                    public_text="Find and verify the evidence.",
                    public_context={"stage": "S3", "question": "Which trial?"},
                    skills=(skill,),
                    output_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                ),
            )
        return backend, receipt

    backend, receipt = asyncio.run(scenario())
    verify_codex_turn_receipt(receipt)
    assert receipt.final_response == "verified"
    assert receipt.max_parallelism_observed == 2
    assert receipt.offered_mcp_tool_names == ("evamed/search", "evamed/read")
    assert tuple(call.fully_qualified_name for call in receipt.tool_calls) == (
        "evamed/search",
        "evamed/read",
    )
    assert receipt.selected_skill_ids == ("medical-literature-review",)
    assert receipt.usage["total"]["totalTokens"] == 125
    assert receipt.config_values_recorded is False
    assert receipt.input_payload_recorded is False
    committed = canonical_json_bytes(receipt)
    assert b"hidden chain of thought" not in committed
    assert b"never commit me" not in committed
    assert any(event.content_redacted for event in receipt.events)
    _, input_items, turn_options = backend.turn_inputs[0]
    assert not any(item.kind == "skill" for item in input_items)
    skill_text = next(
        item.text for item in input_items
        if item.kind == "text" and item.text and "eva.codex-skill-mount-sidecar.v1" in item.text
    )
    assert "original skill bytes" in skill_text
    assert "no filesystem or MCP resource read is offered" in skill_text
    assert turn_options.sandbox is CodexSandbox.WORKSPACE_WRITE


@pytest.mark.parametrize(
    ("server", "tool", "arguments", "status", "result", "error"),
    (
        (
            "evamed", "read_mcp_resource",
            {"server": "evamed", "uri": "eva://unavailable"},
            "failed", None,
            {"message": "resources/read failed: Mcp error: -32601: method not found"},
        ),
        (
            "evamed",
            "list_mcp_resources",
            {"server": "evamed"},
            "failed",
            None,
            {"message": "resources/list failed: Mcp error: -32601: method not found"},
        ),
        (
            "codex",
            "list_mcp_resources",
            {},
            "completed",
            {
                "content": [{"type": "text", "text": '{"resources":[]}'}],
                "structuredContent": None,
                "_meta": None,
            },
            None,
        ),
        (
            "codex",
            "list_mcp_resource_templates",
            {},
            "completed",
            {
                "content": [
                    {"type": "text", "text": '{"resourceTemplates":[]}'}
                ],
                "structuredContent": None,
                "_meta": None,
            },
            None,
        ),
        (
            "codex-dev",
            "list_mcp_resources",
            {"server": "codex-dev"},
            "failed",
            None,
            {"message": "resources/list failed: unknown MCP server 'codex-dev'"},
        ),
    ),
)
def test_runtime_retains_exact_codex_core_resource_calls(
    server: str,
    tool: str,
    arguments: Mapping[str, Any],
    status: str,
    result: Any,
    error: Any,
) -> None:
    async def scenario():
        backend = _FakeBackend(
            _core_resource_script(
                server=server,
                tool=tool,
                arguments=arguments,
                status=status,
                result=result,
                error=error,
            )
        )
        async with CodexRuntime(
            backend, id_factory=DeterministicUUIDFactory(f"core-resource-{server}-{tool}")
        ) as runtime:
            handle = await runtime.start_thread(_actor_options())
            return await runtime.run_turn(
                handle, CodexTurnInput(public_text="Use the exact EvaMed tools.")
            )

    receipt = asyncio.run(scenario())
    assert len(receipt.tool_calls) == 1
    assert receipt.tool_calls[0].fully_qualified_name == f"{server}/{tool}"
    assert receipt.tool_calls[0].status == status
    assert receipt.offered_mcp_tool_names == ("evamed/search", "evamed/read")


@pytest.mark.parametrize(
    ("server", "tool"),
    (
        ("other", "execute_code"),
        ("evamed", "get_tools"),
    ),
)
def test_runtime_still_rejects_non_core_unoffered_mcp_calls(
    server: str, tool: str
) -> None:
    arguments = (
        {"server": server, "uri": "eva://unavailable"}
        if tool == "read_mcp_resource"
        else ({"server": server} if server != "codex" else {})
    )

    async def scenario() -> None:
        backend = _FakeBackend(
            _core_resource_script(
                server=server,
                tool=tool,
                arguments=arguments,
                status="failed",
                result=None,
                error={"message": "method not found"},
            )
        )
        async with CodexRuntime(
            backend, id_factory=DeterministicUUIDFactory(f"unoffered-{server}-{tool}")
        ) as runtime:
            handle = await runtime.start_thread(_actor_options())
            with pytest.raises(
                CodexRuntimeError,
                match=rf"Codex invoked unoffered MCP tool: {server}/{tool}",
            ):
                await runtime.run_turn(
                    handle, CodexTurnInput(public_text="Use the exact EvaMed tools.")
                )

    asyncio.run(scenario())


def test_actor_private_material_fails_before_backend_turn() -> None:
    async def scenario() -> _FakeBackend:
        backend = _FakeBackend()
        async with CodexRuntime(
            backend, id_factory=DeterministicUUIDFactory("codex-private")
        ) as runtime:
            handle = await runtime.start_thread(_actor_options())
            with pytest.raises(CodexRuntimeError, match="private field"):
                await runtime.run_turn(
                    handle,
                    CodexTurnInput(
                        public_text="Answer with tools.",
                        public_context={"nested": {"ground_truth": "do not leak"}},
                    ),
                )
            with pytest.raises(CodexRuntimeError, match="judge-only context"):
                await runtime.run_turn(
                    handle,
                    CodexTurnInput(
                        public_text="Answer with tools.",
                        judge_only_context={"rubric": "private"},
                    ),
                )
            with pytest.raises(CodexRuntimeError, match="private judging material"):
                await runtime.run_turn(
                    handle,
                    CodexTurnInput(public_text="Preserve the gold answer label."),
                )
        return backend

    backend = asyncio.run(scenario())
    assert backend.turn_inputs == []


def test_teacher_projection_allows_public_rubric_without_private_actor_input() -> None:
    async def scenario() -> _FakeBackend:
        instruction, changed = teacher_safe_actor_instruction(
            "Score the public rubric and do not request the hidden answer key."
        )
        assert changed is True
        backend = _FakeBackend()
        async with CodexRuntime(
            backend, id_factory=DeterministicUUIDFactory("teacher-public-rubric")
        ) as runtime:
            handle = await runtime.start_thread(_actor_options())
            receipt = await runtime.run_turn(
                handle,
                CodexTurnInput(
                    public_text=instruction,
                    public_context={
                        "public_reward_contract": {
                            "rubric_table": {
                                "items": [
                                    {
                                        "criterion": (
                                            "Do not request the hidden answer key."
                                        )
                                    }
                                ]
                            }
                        },
                        "teacher_actor_projection": {
                            "judge_only_material_included": False
                        },
                    },
                ),
            )
        assert receipt.status == "completed"
        return backend

    backend = asyncio.run(scenario())
    assert len(backend.turn_inputs) == 1


def test_exact_construction_capability_is_one_shot_and_context_bound() -> None:
    policy = _new_exact_construction_input_policy()
    backend = _FakeBackend()
    options = _actor_options(
        sandbox=CodexSandbox.READ_ONLY,
        offered_tools=(),
        config={"model_providers": {"eva_opus": {"base_url": "http://127.0.0.1:1"}}},
        service_name="evamed-codex-premium-construction",
    )
    live_options = replace(
        options,
        config={"model_providers": {"eva_opus": {"base_url": "http://127.0.0.1:2"}}},
    )
    assert options.config_keys == live_options.config_keys
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    turn_input = CodexTurnInput(
        public_text="Preserve the gold answer label from the frozen request.",
        output_schema=schema,
        sandbox=CodexSandbox.READ_ONLY,
        model=options.model,
    )
    runner = PersistentCodexRuntimeRunner(
        lambda: CodexRuntime(
            backend,
            id_factory=DeterministicUUIDFactory("codex-construction-policy"),
            _construction_input_policy=policy,
        )
    )
    runner.start()
    try:
        with policy.authorize(
            options=options,
            turn_input=turn_input,
            request_blake3="a" * 64,
            source_request_blake3="b" * 64,
            source_phase_request_blake3="c" * 64,
            source_output_schema_blake3=blake3_hex(schema),
        ):
            receipt = runner.run_once(live_options, turn_input)
        assert receipt.status == "completed"
        assert len(backend.turn_inputs) == 1

        with pytest.raises(CodexRuntimeError, match="private judging material"):
            runner.run_once(live_options, turn_input)
    finally:
        runner.close()


def test_judge_resume_receives_separate_private_context_read_only() -> None:
    async def scenario():
        backend = _FakeBackend()
        options = CodexThreadOptions(
            role=CodexRole.JUDGE,
            model="claude-opus-5",
            provider="anthropic",
            cwd="/tmp/judge-snapshot",
            sandbox=CodexSandbox.READ_ONLY,
            offered_tools=(),
        )
        async with CodexRuntime(
            backend, id_factory=DeterministicUUIDFactory("codex-judge")
        ) as runtime:
            handle = await runtime.resume_thread("judge-thread", options)
            receipt = await runtime.run_turn(
                handle,
                CodexTurnInput(
                    public_text="Inspect the committed snapshot.",
                    public_context={"candidate": "public"},
                    judge_only_context={"rubric": {"criterion": "private"}},
                ),
            )
        return backend, receipt

    backend, receipt = asyncio.run(scenario())
    assert backend.resumed[0][0] == "judge-thread"
    assert receipt.thread_resumed is True
    assert receipt.visibility == "judge-only"
    assert receipt.sandbox is CodexSandbox.READ_ONLY
    private_items = [item.text for _, items, _ in backend.turn_inputs for item in items if item.kind == "text"]
    assert any("judge-only context" in (text or "") for text in private_items)


def test_runtime_adds_no_outer_concurrency_throttle() -> None:
    async def scenario() -> int:
        backend = _FakeBackend()
        async with CodexRuntime(
            backend, id_factory=DeterministicUUIDFactory("codex-wide")
        ) as runtime:
            first = await runtime.start_thread(_actor_options(offered_tools=()))
            second = await runtime.start_thread(_actor_options(offered_tools=()))
            await asyncio.gather(
                runtime.run_turn(first, CodexTurnInput(public_text="first")),
                runtime.run_turn(second, CodexTurnInput(public_text="second")),
            )
        return backend.max_active_streams

    assert asyncio.run(scenario()) == 2


def test_receipt_tamper_fails_closed() -> None:
    async def scenario():
        backend = _FakeBackend()
        async with CodexRuntime(
            backend, id_factory=DeterministicUUIDFactory("codex-tamper")
        ) as runtime:
            handle = await runtime.start_thread(_actor_options(offered_tools=()))
            return await runtime.run_turn(handle, CodexTurnInput(public_text="work"))

    receipt = asyncio.run(scenario())
    with pytest.raises(CodexRuntimeError, match="receipt BLAKE3"):
        replace(receipt, final_response="altered")
    with pytest.raises(CodexRuntimeError, match="event BLAKE3"):
        replace(receipt.events[0], method="altered")


def test_original_tool_and_skill_schemas_are_canonical_unchanged() -> None:
    async def no_op(_workspace, _arguments):  # pragma: no cover - never executed
        return {}

    pipeline_definition = ToolDefinition(
        name="medical_search",
        description="Search a medical index.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=no_op,
    )
    skill_catalog = SkillCatalog(
        [SkillDocument("medical-synthesis", "Synthesize evidence.", "# exact skill")]
    )
    skill_definitions = skill_catalog.tool_definitions()
    definitions = (pipeline_definition, *skill_definitions)
    schema_bytes_before = tuple(
        canonical_json_bytes(
            getattr(definition, "input_schema", getattr(definition, "parameters", None))
        )
        for definition in definitions
    )
    offers = tuple(
        CodexToolOffer.from_definition(server="evamed", definition=definition)
        for definition in definitions
    )
    options = _actor_options(offered_tools=offers)

    async def scenario():
        backend = _FakeBackend()
        async with CodexRuntime(
            backend, id_factory=DeterministicUUIDFactory("codex-schemas")
        ) as runtime:
            handle = await runtime.start_thread(options)
            await runtime.run_turn(handle, CodexTurnInput(public_text="Use the offered tools."))
        return backend

    backend = asyncio.run(scenario())
    schema_bytes_after = tuple(
        canonical_json_bytes(
            getattr(definition, "input_schema", getattr(definition, "parameters", None))
        )
        for definition in definitions
    )
    assert schema_bytes_after == schema_bytes_before
    assert tuple(offer.name for offer in offers) == (
        "medical_search",
        "search_skills",
        "load_skill",
    )
    assert offers[0].parallel_safe is False
    assert offers[0].read_only is False
    assert offers[1].parallel_safe is True
    assert offers[1].allowed_stages == ("E2E", "S1", "S2", "S3", "S4", "S5")
    assert tuple(canonical_json_bytes(offer.input_schema) for offer in offers) == schema_bytes_before

    _, sent, _ = backend.turn_inputs[0]
    catalog = next(
        json.loads(item.text)
        for item in sent
        if item.kind == "text" and item.text and "eva.codex-tool-catalog-sidecar.v1" in item.text
    )
    for index, tool in enumerate(catalog["tools"]):
        assert canonical_json_bytes(tool["inputSchema"]) == schema_bytes_before[index]
        assert catalog["metadata"][index]["input_schema_blake3"] == blake3_hex(
            offers[index].input_schema
        )
    assert options.offered_tools == offers


def test_policy_rubric_evidence_and_output_schema_remain_exact_sidecar_values() -> None:
    policy = {
        "stage": "S4",
        "allowed_tools": ["workspace_list", "workspace_read"],
        "limits": {"max_parallel_tools": 8},
    }
    rubric = {
        "rubric_id": "clinical-evidence-s4-v1",
        "digest": "blake3:" + "a" * 64,
        "items": [{"id": "source_quality", "weight_bps": 5000}],
    }
    evidence = {
        "bundle_blake3": "b" * 64,
        "workspace_before_blake3": "c" * 64,
        "workspace_after_blake3": "d" * 64,
    }
    output_schema = {
        "type": "object",
        "properties": {"scores": {"type": "array", "items": {"type": "integer"}}},
        "required": ["scores"],
        "additionalProperties": False,
    }
    values = (policy, rubric, evidence, output_schema)
    bytes_before = tuple(canonical_json_bytes(value) for value in values)
    digests_before = tuple(blake3_hex(value) for value in values)

    async def scenario():
        backend = _FakeBackend()
        options = CodexThreadOptions(
            role=CodexRole.JUDGE,
            model="claude-opus-5",
            provider="anthropic",
            cwd="/tmp/committed-evidence",
            sandbox=CodexSandbox.READ_ONLY,
        )
        async with CodexRuntime(
            backend, id_factory=DeterministicUUIDFactory("canonical-sidecars")
        ) as runtime:
            handle = await runtime.start_thread(options)
            receipt = await runtime.run_turn(
                handle,
                CodexTurnInput(
                    public_text="Judge only committed evidence.",
                    public_context={"policy": policy},
                    judge_only_context={"rubric": rubric, "evidence": evidence},
                    output_schema=output_schema,
                ),
            )
        return backend, receipt

    backend, receipt = asyncio.run(scenario())
    bytes_after = tuple(canonical_json_bytes(value) for value in values)
    digests_after = tuple(blake3_hex(value) for value in values)
    assert bytes_after == bytes_before
    assert digests_after == digests_before
    assert receipt.visibility == "judge-only"

    _, sent, turn_options = backend.turn_inputs[0]
    texts = tuple(item.text or "" for item in sent if item.kind == "text")
    public_payload = json.loads(
        next(text.split("\n", 1)[1] for text in texts if text.startswith("EVA public context"))
    )
    private_payload = json.loads(
        next(
            text.split("\n", 1)[1]
            for text in texts
            if text.startswith("EVA judge-only context")
        )
    )
    assert canonical_json_bytes(public_payload["policy"]) == bytes_before[0]
    assert canonical_json_bytes(private_payload["rubric"]) == bytes_before[1]
    assert canonical_json_bytes(private_payload["evidence"]) == bytes_before[2]
    assert canonical_json_bytes(turn_options.output_schema) == bytes_before[3]


@dataclass
class _SdkInput:
    name: str | None = None
    path: str | None = None
    text: str | None = None


class _SdkHandle:
    id = "official-turn"

    async def stream(self):
        yield _turn_event(
            "turn/completed",
            "official-thread",
            self.id,
            turn={"id": self.id, "status": "completed", "items": []},
        )


class _SdkThread:
    id = "official-thread"

    def __init__(self, owner: "_SdkClient") -> None:
        self.owner = owner

    async def turn(self, items, **kwargs):
        self.owner.turn_args = (items, kwargs)
        return _SdkHandle()


class _SdkClient:
    last: "_SdkClient | None" = None

    def __init__(self, *, config) -> None:
        self.config = config
        self.start_args = None
        self.resume_args = None
        self.turn_args = None
        self.metadata = SimpleNamespace(serverInfo=SimpleNamespace(version="0.152.1"))
        _SdkClient.last = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def close(self):
        return None

    async def thread_start(self, **kwargs):
        self.start_args = kwargs
        return _SdkThread(self)

    async def thread_resume(self, thread_id, **kwargs):
        self.resume_args = (thread_id, kwargs)
        return _SdkThread(self)


def test_official_sdk_adapter_maps_async_thread_turn_and_sandbox_without_request() -> None:
    configs: list[dict[str, Any]] = []

    def config_factory(**kwargs):
        configs.append(kwargs)
        return kwargs

    bindings = CodexSdkBindings(
        async_codex=_SdkClient,
        codex_config=config_factory,
        approval_deny_all="deny_all",
        sandbox_read_only="readOnly",
        sandbox_workspace_write="workspaceWrite",
        text_input=lambda **kwargs: _SdkInput(**kwargs),
        skill_input=lambda **kwargs: _SdkInput(**kwargs),
        mention_input=lambda **kwargs: _SdkInput(**kwargs),
        sdk_version="0.147.0",
    )
    launch = CodexLaunchOptions(
        codex_bin="/usr/bin/codex",
        config_overrides=("features.multi_tool=true",),
        env={"SAFE_REFERENCE": "value"},
    )
    output_schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    output_before = canonical_json_bytes(output_schema)

    async def scenario():
        backend = OpenAICodexBackend(launch, bindings=bindings)
        async with CodexRuntime(
            backend, id_factory=DeterministicUUIDFactory("official-adapter")
        ) as runtime:
            handle = await runtime.start_thread(_actor_options(offered_tools=()))
            return await runtime.run_turn(
                handle,
                CodexTurnInput(public_text="work", output_schema=output_schema),
            )

    receipt = asyncio.run(scenario())
    client = _SdkClient.last
    assert client is not None
    assert configs[0]["codex_bin"] == "/usr/bin/codex"
    assert client.start_args["model"] == "gpt-5.6-sol"
    assert client.start_args["model_provider"] == "openai"
    assert client.start_args["approval_mode"] == "deny_all"
    assert client.start_args["sandbox"] == "workspaceWrite"
    assert client.turn_args[1]["sandbox"] == "workspaceWrite"
    assert canonical_json_bytes(client.turn_args[1]["output_schema"]) == output_before
    assert canonical_json_bytes(output_schema) == output_before
    assert receipt.sdk_version == "0.147.0"
    assert receipt.server_version == "0.152.1"
    assert receipt.turn_id == "official-turn"


def test_options_reject_inline_credentials_and_actor_judge_tools() -> None:
    with pytest.raises(CodexRuntimeError, match="inline credential"):
        _actor_options(config={"api_key": "secret"})
    with pytest.raises(CodexRuntimeError, match="judge-only"):
        _actor_options(
            offered_tools=(
                replace(_offers()[0], visibility="judge-only"),
            )
        )
    assert "value" not in repr(CodexLaunchOptions(env={"TOKEN_ENV_REFERENCE": "value"}))
