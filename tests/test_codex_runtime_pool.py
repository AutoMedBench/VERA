from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any, AsyncIterator, Mapping, Sequence

import pytest

from eva_agent.codex_runtime import (
    BackendInputItem,
    BackendTurnOptions,
    CodexRole,
    CodexRuntime,
    CodexRuntimeError,
    CodexSandbox,
    CodexThreadOptions,
    CodexTurnInput,
    ShardedPersistentCodexRuntimeRunner,
    verify_codex_turn_receipt,
)


@dataclass
class _Notification:
    method: str
    payload: Mapping[str, Any]


class _Turn:
    def __init__(self, backend: "_Backend", thread_id: str, turn_id: str) -> None:
        self.id = turn_id
        self._backend = backend
        self._thread_id = thread_id

    async def stream(self) -> AsyncIterator[_Notification]:
        with self._backend.lock:
            self._backend.active += 1
            self._backend.maximum = max(self._backend.maximum, self._backend.active)
        try:
            yield _Notification(
                "turn/started",
                {
                    "threadId": self._thread_id,
                    "turnId": self.id,
                    "turn": {"id": self.id, "status": "inProgress"},
                },
            )
            await asyncio.sleep(0.04)
            yield _Notification(
                "item/completed",
                {
                    "threadId": self._thread_id,
                    "turnId": self.id,
                    "item": {
                        "id": f"answer-{self.id}",
                        "type": "agentMessage",
                        "phase": "final_answer",
                        "text": "done",
                    },
                },
            )
            yield _Notification(
                "turn/completed",
                {
                    "threadId": self._thread_id,
                    "turnId": self.id,
                    "turn": {"id": self.id, "status": "completed", "items": []},
                },
            )
        finally:
            with self._backend.lock:
                self._backend.active -= 1


class _BackendThread:
    def __init__(self, backend: "_Backend", thread_id: str) -> None:
        self._backend = backend
        self.id = thread_id

    async def turn(
        self,
        items: Sequence[BackendInputItem],
        options: BackendTurnOptions,
    ) -> _Turn:
        assert items and options.model == "model"
        with self._backend.lock:
            self._backend.turns += 1
            turn_id = f"shard-{self._backend.index}-turn-{self._backend.turns}"
        return _Turn(self._backend, self.id, turn_id)


class _Backend:
    sdk_version = "fake-sdk"
    server_version = "fake-app-server"

    def __init__(self, index: int) -> None:
        self.index = index
        self.opens = 0
        self.closes = 0
        self.threads = 0
        self.turns = 0
        self.active = 0
        self.maximum = 0
        self.lock = Lock()

    async def open(self) -> None:
        self.opens += 1

    async def close(self) -> None:
        self.closes += 1

    async def start_thread(self, options: CodexThreadOptions) -> _BackendThread:
        del options
        with self.lock:
            self.threads += 1
            thread_id = f"shard-{self.index}-thread-{self.threads}"
        return _BackendThread(self, thread_id)

    async def resume_thread(
        self, thread_id: str, options: CodexThreadOptions
    ) -> _BackendThread:
        del options
        return _BackendThread(self, thread_id)


def _options(tmp_path) -> CodexThreadOptions:
    return CodexThreadOptions(
        role=CodexRole.STRONG_ACTOR,
        model="model",
        provider="provider",
        cwd=str(tmp_path.resolve()),
        sandbox=CodexSandbox.WORKSPACE_WRITE,
    )


def test_sharded_pool_starts_once_balances_live_load_and_drains(tmp_path) -> None:
    backends: list[_Backend] = []
    factory_lock = Lock()

    def factory() -> CodexRuntime:
        with factory_lock:
            backend = _Backend(len(backends))
            backends.append(backend)
        return CodexRuntime(backend)

    service = ShardedPersistentCodexRuntimeRunner(factory, shard_count=3)
    with service:
        with ThreadPoolExecutor(max_workers=12) as executor:
            receipts = tuple(
                executor.map(
                    lambda _: service.run_once(
                        _options(tmp_path),
                        CodexTurnInput(public_text="work", model="model"),
                    ),
                    range(12),
                )
            )
        assert service.active_turns == 0
        assert service.shard_active_turns == (0, 0, 0)
        assert all(receipt.final_response == "done" for receipt in receipts)
        for receipt in receipts:
            verify_codex_turn_receipt(receipt)
        assert len(backends) == 3
        assert [backend.opens for backend in backends] == [1, 1, 1]
        assert [backend.turns for backend in backends] == [4, 4, 4]
        assert all(backend.maximum == 4 for backend in backends)
    assert [backend.closes for backend in backends] == [1, 1, 1]
    with pytest.raises(CodexRuntimeError, match="closed"):
        service.run_once(
            _options(tmp_path), CodexTurnInput(public_text="late", model="model")
        )


def test_sharded_pool_validates_count_and_close_before_start(tmp_path) -> None:
    with pytest.raises(CodexRuntimeError, match="positive"):
        ShardedPersistentCodexRuntimeRunner(lambda: CodexRuntime(_Backend(0)), shard_count=0)

    service = ShardedPersistentCodexRuntimeRunner(
        lambda: CodexRuntime(_Backend(0)), shard_count=2
    )
    service.close()
    service.close()
    with pytest.raises(CodexRuntimeError, match="closed"):
        service.start()


def test_sharded_pool_rolls_back_every_started_shard_on_partial_start_failure() -> None:
    backend = _Backend(0)
    calls = 0
    lock = Lock()

    def factory():
        nonlocal calls
        with lock:
            calls += 1
            ordinal = calls
        if ordinal == 2:
            return object()
        return CodexRuntime(backend)

    service = ShardedPersistentCodexRuntimeRunner(factory, shard_count=2)
    with pytest.raises(CodexRuntimeError, match="could not start"):
        service.start()
    assert calls == 2
    assert backend.opens == backend.closes == 1
    assert service.active_turns == 0
    service.close()
