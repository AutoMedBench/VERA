"""Real SDK handle + mock app-server RPC/queues; no process/provider/GPU."""
import asyncio
from dataclasses import dataclass
import json
import queue
from threading import Lock
from types import SimpleNamespace

import pytest
from blake3 import blake3

from eva_agent.codex_runtime import (CodexRuntime, CodexRuntimeError, CodexSdkBindings,
    CodexThreadOptions, CodexRole, CodexSandbox, CodexToolOffer, CodexTurnInput,
    OpenAICodexBackend, verify_codex_turn_receipt)
from eva_agent.codex_runtime.policy_budget import CodexPolicyBudgetExceeded, CodexPolicyBudgetDrainError


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass
class Turn:
    id: str
    status: str


@dataclass
class Terminal:
    threadId: str
    turn: Turn


class MockAppServer:
    def __init__(self, mode, audit=None):
        self.mode, self.audit = mode, audit
        self.calls, self.order = [], []
        self._client = self
        self._sync = SimpleNamespace(_router=SimpleNamespace(_lock=Lock(), _turn_notifications={}))
        self.metadata = SimpleNamespace(serverInfo=SimpleNamespace(version="mock-app-server"))
        self.item = {"id": "call-A", "type": "mcpToolCall", "server": "automed_eval",
            "tool": "read", "arguments": {"path": "notes/progress.md"}, "status": "inProgress"}

    async def __aenter__(self): return self
    async def __aexit__(self, *_): return None
    async def close(self): return None
    async def _ensure_initialized(self): return None

    async def thread_start(self, **_):
        async def turn(_items, **_kwargs):
            # The actual installed SDK method performs turn_interrupt(thread,id).
            cls = pytest.importorskip("openai_codex.api").AsyncTurnHandle
            return cls(self, "thread-A", "turn-A")
        return SimpleNamespace(id="thread-A", turn=turn)

    def emit(self, method, payload):
        self._sync._router._turn_notifications["turn-A"].put(SimpleNamespace(method=method, payload=payload))

    def item_event(self, method, item):
        self.emit(method, {"threadId": "thread-A", "turnId": "turn-A", "item": item})

    def terminal(self, status):
        self.emit("turn/completed", Terminal("thread-A", Turn("turn-A", status)))

    def register_turn_notifications(self, turn_id):
        self._sync._router._turn_notifications[turn_id] = queue.Queue()
        self.order.append("registered")
        self.item_event("item/started", self.item)
        if self.mode == "normal":
            self.finish_host()
            self.terminal("completed")

    def unregister_turn_notifications(self, turn_id):
        self.order.append("unregistered")
        self._sync._router._turn_notifications.pop(turn_id)

    async def next_turn_notification(self, turn_id):
        q = self._sync._router._turn_notifications[turn_id]
        item = await asyncio.to_thread(q.get)
        if isinstance(item, BaseException): raise item
        return item

    def finish_host(self):
        structured = {"event_id": "host-A", "name": "read", "result": {"content": "fixture"}}
        response = {"content": [{"type": "text", "text": canonical(structured).decode()}],
            "structuredContent": structured, "isError": False}
        host = {"schema": "eva.automedbench-track-tool-event.v1", "event_id": "host-A",
            "name": "read", "arguments": self.item["arguments"], "result": structured["result"],
            "is_error": False, "response_blake3": blake3(canonical(response)).hexdigest()}
        host["event_blake3"] = blake3(canonical(host)).hexdigest()
        if self.audit:
            (self.audit / "mcp-events.jsonl").write_bytes(canonical(host) + b"\n")
        self.order.append("host-finished")
        self.item_event("item/completed", {**self.item, "status": "completed", "result": response, "error": None})

    async def turn_interrupt(self, thread_id, turn_id):
        self.calls.append(("turn/interrupt", thread_id, turn_id))
        self.order.append("interrupt")
        if self.mode == "rpc_failure": raise RuntimeError("not retained in sidecar")
        if self.mode == "no_terminal": return {}
        await asyncio.sleep(.005)
        if self.mode == "cancelled_tool":
            self.item_event("item/completed", {**self.item, "status": "failed", "result": None,
                                                "error": {"message": "cancelled"}})
        elif self.mode != "active_tool":
            self.finish_host()
        if self.mode == "provider_error":
            self.emit("error", {"threadId": thread_id, "turnId": turn_id, "error": {"message": "fixture"}})
        self.terminal("failed" if self.mode == "provider_error" else "interrupted")
        return {}


def backend(server):
    return OpenAICodexBackend(bindings=CodexSdkBindings(async_codex=lambda **_: server,
        codex_config=lambda **kw: kw, approval_deny_all="deny_all", sandbox_read_only="readOnly",
        sandbox_workspace_write="workspaceWrite", text_input=lambda **kw: kw,
        skill_input=lambda **kw: kw, mention_input=lambda **kw: kw, sdk_version="0.147.0"))


def options(tmp_path):
    return CodexThreadOptions(role=CodexRole.STRONG_ACTOR, model="fixture", provider="fixture",
        cwd=str(tmp_path), sandbox=CodexSandbox.READ_ONLY, offered_tools=(CodexToolOffer(
            fully_qualified_name="automed_eval/read", description="Read fixture",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),))


async def run(server, tmp_path):
    async with CodexRuntime(backend(server)) as runtime:
        handle = await runtime.start_thread(options(tmp_path))
        return await runtime.run_turn(handle, CodexTurnInput(public_text="fixture"),
                                      policy_timeout_seconds=.02, interruption_grace_seconds=.08)


def test_actual_sdk_interrupt_owned_turn_drains_late_host_event(tmp_path):
    server = MockAppServer("interrupt", tmp_path)
    with pytest.raises(CodexPolicyBudgetExceeded) as caught:
        asyncio.run(run(server, tmp_path))
    receipt = caught.value.receipt
    verify_codex_turn_receipt(receipt)
    assert receipt.status == "interrupted" and len(receipt.tool_calls) == 1
    assert receipt.events[-1].method == "turn/completed"
    assert server.calls == [("turn/interrupt", "thread-A", "turn-A")]
    assert server.order == ["registered", "interrupt", "host-finished", "unregistered"]
    assert caught.value.outcome["receipt_blake3"] == receipt.receipt_blake3
    assert caught.value.outcome["terminal_observed_ns"] >= caught.value.outcome["interrupt_requested_ns"]


def test_completed_before_budget_never_interrupts(tmp_path):
    server = MockAppServer("normal")
    assert asyncio.run(run(server, tmp_path)).status == "completed"
    assert server.calls == []


@pytest.mark.parametrize("mode", ["no_terminal", "rpc_failure"])
def test_missing_terminal_never_yields_receipt_or_zero_reward(tmp_path, mode):
    server = MockAppServer(mode)
    with pytest.raises(CodexPolicyBudgetDrainError) as caught:
        asyncio.run(run(server, tmp_path))
    assert not hasattr(caught.value, "receipt")
    assert caught.value.outcome["reward"] is None
    assert caught.value.outcome["infrastructure_error"]
    if mode == "rpc_failure":
        assert caught.value.outcome["infrastructure_error"] == "policy_interrupt_rpc_failed"
    assert server.order[-1] == "unregistered"


def test_terminal_with_active_call_rejected_by_unchanged_receipt_validation(tmp_path):
    with pytest.raises(CodexRuntimeError, match="active tool calls"):
        asyncio.run(run(MockAppServer("active_tool"), tmp_path))


def test_continuous_events_do_not_reset_wall_clock_budget():
    from eva_agent.codex_runtime.policy_budget import PolicyBudgetStream
    class ContinuousTurn:
        id = "continuous"
        interrupted = False
        async def interrupt(self): self.interrupted = True
        async def stream(self):
            while not self.interrupted:
                yield SimpleNamespace(method="thread/tokenUsage/updated")
            yield SimpleNamespace(method="turn/completed")
    async def scenario():
        turn = ContinuousTurn()
        budget = PolicyBudgetStream(.01, .2)
        async for _ in budget.notifications(turn): pass
        assert turn.interrupted and budget.outcome["budget_exhausted"]
        assert budget.outcome["terminal_observed_ns"]
    asyncio.run(scenario())
