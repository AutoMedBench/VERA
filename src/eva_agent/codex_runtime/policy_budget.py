"""Opt-in host wall-clock budget; interruption is not a fabricated terminal.

The event consumer stays alive through interruption. A provider terminal and
ordinary receipt validation remain mandatory. This does not prove that an
external MCP worker or detached model job has stopped; the caller must join
those host results before calling a workspace snapshot final.
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
import math
import time

from .contracts import CodexRuntimeError


class CodexPolicyBudgetExceeded(CodexRuntimeError):
    """The actual terminal receipt survived a requested policy interruption."""

    def __init__(self, receipt, outcome):
        super().__init__("codex_policy_turn_budget_exhausted")
        self.receipt = receipt
        self.outcome = {**outcome, "receipt_blake3": receipt.receipt_blake3,
                        "actual_terminal_status": receipt.status}


class CodexPolicyBudgetDrainError(CodexRuntimeError):
    """No claim of terminal/quiescent execution; reward must remain unknown."""

    def __init__(self, category, outcome):
        super().__init__(category)
        self.outcome = {**outcome, "infrastructure_error": category, "reward": None}


class PolicyBudgetStream:
    def __init__(self, timeout_seconds, drain_seconds):
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(x) or x <= 0 for x in (timeout_seconds, drain_seconds)):
            raise CodexRuntimeError("policy timeout and drain must be positive finite seconds")
        self.deadline = asyncio.get_running_loop().time() + timeout_seconds
        self.drain_seconds = drain_seconds
        self.outcome = {"schema": "eva.codex-policy-turn-budget.v1",
            "timeout_seconds": timeout_seconds, "drain_seconds": drain_seconds,
            "budget_exhausted": False, "interrupt_requested_ns": None,
            "interrupt_acknowledged_ns": None, "terminal_observed_ns": None,
            "infrastructure_error": None, "reward": None}

    async def notifications(self, turn):
        self.outcome["turn_id"] = turn.id
        source = turn.stream().__aiter__()
        pending = None
        interrupt = None
        async def request_interrupt():
            await turn.interrupt()
            self.outcome["interrupt_acknowledged_ns"] = time.time_ns()
        try:
            while True:
                if pending is None:
                    pending = asyncio.create_task(anext(source))
                remaining = max(0, self.deadline - asyncio.get_running_loop().time())
                if remaining == 0:
                    if interrupt is not None:
                        raise CodexPolicyBudgetDrainError("policy_interrupt_terminal_drain_timeout", self.outcome)
                    self.outcome.update(budget_exhausted=True, interrupt_requested_ns=time.time_ns())
                    self.deadline = asyncio.get_running_loop().time() + self.drain_seconds
                    if not callable(getattr(turn, "interrupt", None)):
                        raise CodexPolicyBudgetDrainError("policy_interrupt_unsupported", self.outcome)
                    interrupt = asyncio.create_task(request_interrupt())
                    # Do not cancel the pending read: its real terminal belongs
                    # to this very stream and must not be lost on budget expiry.
                    continue
                if interrupt is not None and interrupt.done() and interrupt.exception() is not None:
                    raise CodexPolicyBudgetDrainError("policy_interrupt_rpc_failed", self.outcome)
                waiters = (pending, interrupt) if interrupt is not None and not interrupt.done() else (pending,)
                done, _ = await asyncio.wait(waiters, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
                if pending not in done:
                    continue
                try:
                    notification = pending.result()
                except StopAsyncIteration:
                    break
                pending = None
                if getattr(notification, "method", None) == "turn/completed":
                    self.outcome["terminal_observed_ns"] = time.time_ns()
                yield notification
            if interrupt is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(interrupt),
                        max(0, self.deadline - asyncio.get_running_loop().time()))
                except Exception as exc:
                    raise CodexPolicyBudgetDrainError("policy_interrupt_rpc_failed", self.outcome) from exc
        finally:
            for task in (pending, interrupt):
                if task is not None and not task.done():
                    task.cancel()  # Local wait cleanup only, never a stop claim.
            await asyncio.gather(*(task for task in (pending, interrupt) if task is not None), return_exceptions=True)
            with suppress(StopAsyncIteration):
                await source.aclose()
