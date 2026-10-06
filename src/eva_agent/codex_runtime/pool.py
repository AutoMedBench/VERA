"""Unthrottled pool of persistent Codex app-server runtimes.

One app-server can multiplex many fresh Codex threads, but a large campaign
should not make that single local process its only scheduling and failure
domain.  This pool starts a fixed number of
:class:`PersistentCodexRuntimeRunner` shards and sends each turn to the
currently least-loaded shard.  It adds no semaphore, queue, or retry policy.

Every semantic attempt still calls ``run_once`` exactly once.  Provider
capacity and durable retry/quarantine decisions remain owned by the campaign.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Condition, RLock

from .contracts import (
    CodexRuntimeError,
    CodexThreadOptions,
    CodexTurnInput,
    CodexTurnReceipt,
)
from .service import PersistentCodexRuntimeRunner, RuntimeFactory


class ShardedPersistentCodexRuntimeRunner:
    """Least-loaded dispatch over independent persistent app-server shards.

    ``runtime_factory`` is invoked once by each shard, so it must return a new
    :class:`~eva_agent.codex_runtime.CodexRuntime` on every invocation.  Shards
    start and close in parallel.  Calls accepted before ``close`` are drained;
    calls arriving after drain begins fail closed.
    """

    def __init__(self, runtime_factory: RuntimeFactory, *, shard_count: int) -> None:
        if not callable(runtime_factory):
            raise CodexRuntimeError("Codex runtime factory is required")
        if type(shard_count) is not int or shard_count < 1:
            raise CodexRuntimeError("Codex app-server shard count must be positive")
        self._runners = tuple(
            PersistentCodexRuntimeRunner(runtime_factory) for _ in range(shard_count)
        )
        self._condition = Condition(RLock())
        self._loads = [0] * shard_count
        self._cursor = 0
        self._state = "new"
        self._terminal_error: BaseException | None = None

    def __enter__(self) -> "ShardedPersistentCodexRuntimeRunner":
        self.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    @property
    def shard_count(self) -> int:
        return len(self._runners)

    @property
    def active_turns(self) -> int:
        with self._condition:
            return sum(self._loads)

    @property
    def shard_active_turns(self) -> tuple[int, ...]:
        """Return observability counters without exposing mutable pool state."""

        with self._condition:
            return tuple(self._loads)

    def _raise_terminal(self, message: str) -> None:
        error = CodexRuntimeError(message)
        if self._terminal_error is None:
            raise error
        raise error from self._terminal_error

    def start(self) -> None:
        """Open all app-server shards in parallel and fail as one unit."""

        leader = False
        with self._condition:
            while self._state == "starting":
                self._condition.wait()
            if self._state == "open":
                return
            if self._state in {"draining", "closed"}:
                raise CodexRuntimeError("sharded Codex runtime is closed")
            if self._state == "failed":
                self._raise_terminal("sharded Codex runtime could not start")
            if self._state == "new":
                self._state = "starting"
                leader = True
        if not leader:  # pragma: no cover - state machine exhaustiveness
            raise CodexRuntimeError("sharded Codex runtime state differs")

        failures: list[BaseException] = []
        with ThreadPoolExecutor(
            max_workers=self.shard_count,
            thread_name_prefix="eva-codex-shard-start",
        ) as executor:
            futures = tuple(executor.submit(runner.start) for runner in self._runners)
            for future in futures:
                try:
                    future.result()
                except BaseException as exc:
                    failures.append(exc)

        if failures:
            # Each child close is idempotent, including a child whose startup
            # failed before an app-server became available.
            with ThreadPoolExecutor(
                max_workers=self.shard_count,
                thread_name_prefix="eva-codex-shard-rollback",
            ) as executor:
                rollback = tuple(executor.submit(runner.close) for runner in self._runners)
                for future in rollback:
                    try:
                        future.result()
                    except BaseException:
                        pass
            with self._condition:
                self._terminal_error = failures[0]
                self._state = "failed"
                self._condition.notify_all()
            self._raise_terminal("sharded Codex runtime could not start")

        with self._condition:
            self._state = "open"
            self._condition.notify_all()

    def _reserve_shard(self) -> int:
        """Atomically reserve a least-loaded shard with rotating tie breaks."""

        with self._condition:
            if self._state != "open":
                raise CodexRuntimeError("sharded Codex runtime is draining")
            minimum = min(self._loads)
            for offset in range(self.shard_count):
                index = (self._cursor + offset) % self.shard_count
                if self._loads[index] == minimum:
                    self._loads[index] += 1
                    self._cursor = (index + 1) % self.shard_count
                    return index
        raise CodexRuntimeError("sharded Codex runtime selection differed")  # pragma: no cover

    def run_once(
        self,
        options: CodexThreadOptions,
        turn_input: CodexTurnInput,
    ) -> CodexTurnReceipt:
        """Run one fresh thread on one shard; never retry on another shard."""

        self.start()
        index = self._reserve_shard()
        try:
            return self._runners[index].run_once(options, turn_input)
        finally:
            with self._condition:
                self._loads[index] -= 1
                if self._loads[index] < 0:  # pragma: no cover - invariant guard
                    self._terminal_error = CodexRuntimeError(
                        "sharded Codex runtime active count underflow"
                    )
                    self._state = "failed"
                self._condition.notify_all()

    def run_once_with_timeout(self, options, turn_input, *, timeout_seconds):
        self.start()
        index = self._reserve_shard()
        try:
            return self._runners[index].run_once_with_timeout(
                options, turn_input, timeout_seconds=timeout_seconds
            )
        finally:
            with self._condition:
                self._loads[index] -= 1
                if self._loads[index] < 0:  # pragma: no cover
                    self._terminal_error = CodexRuntimeError(
                        "sharded Codex runtime active count underflow"
                    )
                    self._state = "failed"
                self._condition.notify_all()

    def close(self) -> None:
        """Stop admission, drain accepted turns, and close all shards in parallel."""

        with self._condition:
            while self._state == "starting":
                self._condition.wait()
            if self._state == "closed":
                return
            if self._state == "new":
                self._state = "closed"
                self._condition.notify_all()
                return
            if self._state == "draining":
                while self._state == "draining":
                    self._condition.wait()
                if self._terminal_error is not None:
                    self._raise_terminal("sharded Codex runtime could not close")
                return
            if self._state == "failed":
                self._state = "closed"
                self._condition.notify_all()
                return
            if self._state != "open":  # pragma: no cover - invariant guard
                raise CodexRuntimeError("sharded Codex runtime state differs")
            self._state = "draining"
            while sum(self._loads):
                self._condition.wait()

        failures: list[BaseException] = []
        with ThreadPoolExecutor(
            max_workers=self.shard_count,
            thread_name_prefix="eva-codex-shard-close",
        ) as executor:
            futures = tuple(executor.submit(runner.close) for runner in self._runners)
            for future in futures:
                try:
                    future.result()
                except BaseException as exc:
                    failures.append(exc)
        with self._condition:
            if failures:
                self._terminal_error = failures[0]
            self._state = "closed"
            self._condition.notify_all()
        if failures:
            self._raise_terminal("sharded Codex runtime could not close")


__all__ = ["ShardedPersistentCodexRuntimeRunner"]
