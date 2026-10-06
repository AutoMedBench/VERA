"""Long-lived, thread-safe Codex app-server service for high-width campaigns.

The synchronous data pipeline uses worker threads, while the official Codex
SDK is asynchronous. Starting one app-server process for every model turn is
correct but needlessly expensive. This bridge keeps one runtime open on a
dedicated event loop and multiplexes fresh, isolated Codex threads over it.

It deliberately applies no concurrency semaphore and no retry policy. The
campaign owns provider capacity, while every call here remains one fresh
Codex thread and one semantic attempt.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from threading import Condition, Event, RLock, Thread
from typing import Callable

from .contracts import (
    CodexRuntimeError,
    CodexThreadOptions,
    CodexTurnInput,
    CodexTurnReceipt,
)
from .runtime import CodexRuntime


RuntimeFactory = Callable[[], CodexRuntime]


class PersistentCodexRuntimeRunner:
    """Multiplex synchronous callers through one open async Codex runtime.

    Use this service as a context manager in production so shutdown first
    stops accepting work, then drains all already-started turns before closing
    app-server.
    """

    def __init__(self, runtime_factory: RuntimeFactory) -> None:
        if not callable(runtime_factory):
            raise CodexRuntimeError("Codex runtime factory is required")
        self._factory = runtime_factory
        self._condition = Condition(RLock())
        self._ready = Event()
        self._thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runtime: CodexRuntime | None = None
        self._startup_error: BaseException | None = None
        self._accepting = True
        self._closed = False
        self._active = 0

    def __enter__(self) -> "PersistentCodexRuntimeRunner":
        self.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    @property
    def active_turns(self) -> int:
        with self._condition:
            return self._active

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runtime: CodexRuntime | None = None
        try:
            candidate = self._factory()
            if not isinstance(candidate, CodexRuntime):
                raise CodexRuntimeError("runtime factory did not return CodexRuntime")
            runtime = candidate
            loop.run_until_complete(runtime.open())
            with self._condition:
                self._loop = loop
                self._runtime = runtime
        except BaseException as exc:  # retained and re-raised to every caller
            with self._condition:
                self._startup_error = exc
        finally:
            self._ready.set()
        if runtime is None or self._startup_error is not None:
            loop.close()
            return
        try:
            loop.run_forever()
        finally:
            try:
                loop.run_until_complete(runtime.close())
            finally:
                # The pinned async Codex SDK implements its blocking stdio
                # operations with ``asyncio.to_thread``.  ``loop.close()``
                # does not wait for that default executor, so its workers can
                # otherwise survive a completed campaign and hold Python's
                # interpreter shutdown indefinitely.  Drain those workers
                # after closing app-server and before relinquishing the loop.
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
                loop.close()

    def start(self) -> None:
        with self._condition:
            if self._closed or not self._accepting:
                raise CodexRuntimeError("persistent Codex runtime is closed")
            if self._thread is None:
                self._thread = Thread(
                    target=self._thread_main,
                    name="eva-codex-app-server",
                    daemon=False,
                )
                self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise CodexRuntimeError("persistent Codex runtime could not start") from self._startup_error
        if self._loop is None or self._runtime is None:
            raise CodexRuntimeError("persistent Codex runtime startup differed")

    async def _run_fresh(
        self,
        options: CodexThreadOptions,
        turn_input: CodexTurnInput,
    ) -> CodexTurnReceipt:
        runtime = self._runtime
        if runtime is None:  # guarded by start(), retained for fail-closed typing
            raise CodexRuntimeError("persistent Codex runtime is unavailable")
        handle = await runtime.start_thread(options)
        if handle.resumed:
            raise CodexRuntimeError("pipeline turns must start a fresh Codex thread")
        return await runtime.run_turn(handle, turn_input)

    async def _run_fresh_and_release(
        self,
        options: CodexThreadOptions,
        turn_input: CodexTurnInput,
        cleanup_done: Event,
    ) -> CodexTurnReceipt:
        try:
            return await self._run_fresh(options, turn_input)
        finally:
            # This runs on the service event loop only after cancellation has
            # propagated through the SDK stream and its async cleanup.
            with self._condition:
                self._active -= 1
                self._condition.notify_all()
            cleanup_done.set()

    def run_once(
        self,
        options: CodexThreadOptions,
        turn_input: CodexTurnInput,
    ) -> CodexTurnReceipt:
        """Run one fresh thread; concurrent callers are submitted immediately."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise CodexRuntimeError(
                "synchronous persistent Codex runner cannot block an active event loop"
            )
        self.start()
        with self._condition:
            if not self._accepting or self._closed:
                raise CodexRuntimeError("persistent Codex runtime is draining")
            loop = self._loop
            if loop is None:
                raise CodexRuntimeError("persistent Codex runtime loop is unavailable")
            self._active += 1
        future: Future[CodexTurnReceipt] = asyncio.run_coroutine_threadsafe(
            self._run_fresh(options, turn_input), loop
        )
        try:
            return future.result()
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def run_once_with_timeout(
        self,
        options: CodexThreadOptions,
        turn_input: CodexTurnInput,
        *,
        timeout_seconds: int,
    ) -> CodexTurnReceipt:
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 900:
            raise CodexRuntimeError("Codex turn wall-clock timeout differs")
        self.start()
        with self._condition:
            if not self._accepting or self._closed:
                raise CodexRuntimeError("persistent Codex runtime is draining")
            loop = self._loop
            if loop is None:
                raise CodexRuntimeError("persistent Codex runtime loop is unavailable")
            self._active += 1
        cleanup_done = Event()
        try:
            future: Future[CodexTurnReceipt] = asyncio.run_coroutine_threadsafe(
                self._run_fresh_and_release(options, turn_input, cleanup_done), loop
            )
        except BaseException:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()
            raise
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            # A provider coroutine may itself raise TimeoutError. Preserve that
            # completed result rather than misclassifying it as our deadline.
            if future.done():
                return future.result()
            future.cancel()
            # concurrent.futures.Future becomes cancelled before the event-loop
            # Task has necessarily drained. The explicit event is set only by
            # the coroutine's finally block. If cleanup is uncooperative, keep
            # the active slot retained so close() cannot tear down underneath it.
            cleanup_done.wait(timeout=10)
            raise CodexRuntimeError("Codex turn exceeded its wall-clock timeout") from None

    def close(self) -> None:
        """Stop accepting calls, drain current turns, and close app-server once."""

        with self._condition:
            if self._closed:
                return
            self._accepting = False
            thread = self._thread
        if thread is None:
            with self._condition:
                self._closed = True
            return
        self._ready.wait()
        with self._condition:
            while self._active:
                self._condition.wait()
            loop = self._loop
            runtime = self._runtime
        close_error: BaseException | None = None
        if loop is not None and runtime is not None:
            try:
                asyncio.run_coroutine_threadsafe(runtime.close(), loop).result()
            except BaseException as exc:
                close_error = exc
            finally:
                loop.call_soon_threadsafe(loop.stop)
        thread.join()
        with self._condition:
            self._closed = True
            self._loop = None
            self._runtime = None
        if close_error is not None:
            raise CodexRuntimeError("persistent Codex runtime could not close") from close_error


__all__ = ["PersistentCodexRuntimeRunner"]
