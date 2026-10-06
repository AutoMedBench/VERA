from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import queue
from threading import Event, Lock
import time
from types import SimpleNamespace
from typing import Any, AsyncIterator, Mapping, Sequence

import pytest

from eva_agent.codex_runtime import (
    BackendInputItem,
    BackendTurnOptions,
    CodexRole,
    CodexRuntime,
    CodexRuntimeError,
    CodexSandbox,
    CodexSdkBindings,
    CodexThreadOptions,
    CodexTurnInput,
    OpenAICodexBackend,
    PersistentCodexRuntimeRunner,
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
            # All submitted turns can reach this await concurrently on the one
            # app-server event loop; no hidden outer semaphore serializes them.
            await asyncio.sleep(self._backend.delay)
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
            if self._backend.cancellation_cleanup_delay:
                await asyncio.sleep(self._backend.cancellation_cleanup_delay)
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
            turn_id = f"turn-{self._backend.turns}"
        return _Turn(self._backend, self.id, turn_id)


class _Backend:
    sdk_version = "fake-sdk"
    server_version = "fake-app-server"

    def __init__(self) -> None:
        self.opens = 0
        self.closes = 0
        self.threads = 0
        self.turns = 0
        self.active = 0
        self.maximum = 0
        self.lock = Lock()
        self.delay = 0.03
        self.cancellation_cleanup_delay = 0.0

    async def open(self) -> None:
        self.opens += 1

    async def close(self) -> None:
        self.closes += 1

    async def start_thread(self, options: CodexThreadOptions) -> _BackendThread:
        del options
        with self.lock:
            self.threads += 1
            thread_id = f"thread-{self.threads}"
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


def test_persistent_service_reuses_one_runtime_and_multiplexes_fresh_threads(
    tmp_path,
) -> None:
    backend = _Backend()
    service = PersistentCodexRuntimeRunner(lambda: CodexRuntime(backend))
    with service:
        with ThreadPoolExecutor(max_workers=8) as pool:
            receipts = tuple(
                pool.map(
                    lambda _: service.run_once(
                        _options(tmp_path),
                        CodexTurnInput(public_text="work", model="model"),
                    ),
                    range(8),
                )
            )
        assert service.active_turns == 0
        assert all(receipt.final_response == "done" for receipt in receipts)
        for receipt in receipts:
            verify_codex_turn_receipt(receipt)
        assert len({receipt.thread_id for receipt in receipts}) == 8
        assert backend.opens == 1
        assert backend.threads == backend.turns == 8
        assert backend.maximum == 8
    assert backend.closes == 1
    with pytest.raises(CodexRuntimeError, match="closed"):
        service.run_once(
            _options(tmp_path), CodexTurnInput(public_text="late", model="model")
        )


def test_persistent_service_close_before_start_is_idempotent(tmp_path) -> None:
    backend = _Backend()
    service = PersistentCodexRuntimeRunner(lambda: CodexRuntime(backend))
    service.close()
    service.close()
    assert backend.opens == backend.closes == 0
    with pytest.raises(CodexRuntimeError, match="closed"):
        service.start()


def test_persistent_service_hard_timeout_cancels_turn_before_clean_close(
    tmp_path,
) -> None:
    backend = _Backend()
    backend.delay = 60
    service = PersistentCodexRuntimeRunner(lambda: CodexRuntime(backend))

    with service:
        with pytest.raises(CodexRuntimeError, match="exceeded its wall-clock timeout"):
            service.run_once_with_timeout(
                _options(tmp_path),
                CodexTurnInput(public_text="work", model="model"),
                timeout_seconds=1,
            )
        assert service.active_turns == 0
        assert backend.active == 0

    assert backend.opens == backend.closes == 1


def test_persistent_service_timeout_waits_for_delayed_stream_cleanup(tmp_path) -> None:
    backend = _Backend()
    backend.delay = 60
    backend.cancellation_cleanup_delay = 0.1
    service = PersistentCodexRuntimeRunner(lambda: CodexRuntime(backend))

    with service:
        with pytest.raises(CodexRuntimeError, match="exceeded its wall-clock timeout"):
            service.run_once_with_timeout(
                _options(tmp_path),
                CodexTurnInput(public_text="work", model="model"),
                timeout_seconds=1,
            )
        assert service.active_turns == 0
        assert backend.active == 0

    assert backend.opens == backend.closes == 1


def test_persistent_service_close_drains_sdk_default_executor(tmp_path) -> None:
    """SDK ``asyncio.to_thread`` workers cannot outlive a closed service."""

    backend = _Backend()
    release = Event()
    executor_finished = Event()

    def sdk_background_worker() -> None:
        release.wait()
        # Keep the executor worker alive briefly after backend.close() so this
        # distinguishes an explicit executor drain from bare loop.close().
        time.sleep(0.1)
        executor_finished.set()

    original_open = backend.open
    original_close = backend.close

    async def open_with_sdk_worker() -> None:
        await original_open()
        asyncio.create_task(asyncio.to_thread(sdk_background_worker))

    async def close_and_release_sdk_worker() -> None:
        await original_close()
        release.set()

    backend.open = open_with_sdk_worker  # type: ignore[method-assign]
    backend.close = close_and_release_sdk_worker  # type: ignore[method-assign]
    service = PersistentCodexRuntimeRunner(lambda: CodexRuntime(backend))

    service.start()
    service.close()

    assert executor_finished.is_set()
    assert backend.opens == backend.closes == 1


class _BlockingRouter:
    def __init__(self) -> None:
        self._lock = Lock()
        self._turn_notifications: dict[str, queue.Queue[object]] = {}


class _BlockingStreamClient:
    """Exact pinned-SDK queue shape with a cancellation-hostile to_thread."""

    def __init__(self) -> None:
        self._sync = SimpleNamespace(_router=_BlockingRouter())
        self.worker_started = Event()
        self.worker_finished = Event()
        self.unregistered = Event()

    def register_turn_notifications(self, turn_id: str) -> None:
        with self._sync._router._lock:
            self._sync._router._turn_notifications[turn_id] = queue.Queue()

    def unregister_turn_notifications(self, turn_id: str) -> None:
        with self._sync._router._lock:
            self._sync._router._turn_notifications.pop(turn_id, None)
        self.unregistered.set()

    def _blocking_next(self, turn_id: str) -> object:
        with self._sync._router._lock:
            turn_queue = self._sync._router._turn_notifications[turn_id]
        self.worker_started.set()
        try:
            item = turn_queue.get()
            if isinstance(item, BaseException):
                raise item
            return item
        finally:
            self.worker_finished.set()

    async def next_turn_notification(self, turn_id: str) -> object:
        return await asyncio.to_thread(self._blocking_next, turn_id)


class _BlockingSdkHandle:
    id = "blocking-turn"

    async def stream(self):  # pragma: no cover - safe adapter must replace this
        raise AssertionError("unsafe published SDK stream was used")
        yield


class _BlockingSdkThread:
    id = "blocking-thread"

    async def turn(self, _items, **_kwargs):
        return _BlockingSdkHandle()


class _BlockingSdkClient:
    last: "_BlockingSdkClient | None" = None

    def __init__(self, *, config) -> None:
        del config
        self._client = _BlockingStreamClient()
        self.metadata = SimpleNamespace(serverInfo=SimpleNamespace(version="fake"))
        _BlockingSdkClient.last = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def close(self):
        return None

    async def thread_start(self, **_kwargs):
        return _BlockingSdkThread()

    async def thread_resume(self, _thread_id, **_kwargs):
        return _BlockingSdkThread()


def test_persistent_service_timeout_wakes_pinned_sdk_queue_before_unregister(
    tmp_path,
) -> None:
    """A timed-out SDK queue worker must not hang default-executor shutdown."""

    bindings = CodexSdkBindings(
        async_codex=_BlockingSdkClient,
        codex_config=lambda **kwargs: kwargs,
        approval_deny_all="deny_all",
        sandbox_read_only="readOnly",
        sandbox_workspace_write="workspaceWrite",
        text_input=lambda **kwargs: SimpleNamespace(**kwargs),
        skill_input=lambda **kwargs: SimpleNamespace(**kwargs),
        mention_input=lambda **kwargs: SimpleNamespace(**kwargs),
        sdk_version="0.147.0",
    )
    backend = OpenAICodexBackend(bindings=bindings)
    service = PersistentCodexRuntimeRunner(lambda: CodexRuntime(backend))

    started = time.monotonic()
    with service:
        with pytest.raises(CodexRuntimeError, match="exceeded its wall-clock timeout"):
            service.run_once_with_timeout(
                _options(tmp_path),
                CodexTurnInput(public_text="provider-free blocked stream", model="model"),
                timeout_seconds=1,
            )
        client = _BlockingSdkClient.last
        assert client is not None
        assert client._client.worker_started.is_set()
        assert client._client.worker_finished.wait(timeout=2)
        assert client._client.unregistered.wait(timeout=2)

    assert time.monotonic() - started < 5
