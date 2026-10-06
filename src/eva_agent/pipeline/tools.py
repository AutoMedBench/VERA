"""Deterministic tool/skill execution with parallel independent frontiers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier, Lock
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
    ContractError,
    JsonValue,
    RuntimeIdFactory,
    ToolCall,
    ToolResult,
    ToolTrace,
    freeze_json,
)
from .digests import blake3_hex
from .workspace import FilesystemSandbox
from .execution_diagnostics import diagnostic_scope


ToolHandler = Callable[[FilesystemSandbox, Mapping[str, JsonValue]], Any]


# One Codex turn may issue a wide read-only frontier.  This is deliberately a
# capability ceiling rather than a process-wide throttle: mutating or
# non-parallel-safe definitions are still executed one at a time below.
MAXIMUM_PARALLEL_TOOL_CALLS = 64


class ToolExecutionError(ContractError):
    """A tool graph or execution contract failed closed."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, JsonValue]
    handler: ToolHandler
    kind: str = "tool"
    parallel_safe: bool = False
    read_only: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.description or self.kind not in {"tool", "skill"}:
            raise ToolExecutionError("tool definition identity differs")
        schema = freeze_json(self.input_schema)
        if not isinstance(schema, Mapping):
            raise ToolExecutionError("tool input schema must be an object")
        if self.read_only and not self.parallel_safe:
            raise ToolExecutionError("read-only definitions should declare parallel safety")
        object.__setattr__(self, "input_schema", schema)

    def public_schema(self) -> Mapping[str, JsonValue]:
        return freeze_json(
            {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.input_schema,
                    "x-eva-kind": self.kind,
                    "x-eva-parallel-safe": self.parallel_safe,
                },
            }
        )  # type: ignore[return-value]


class ToolRegistry:
    def __init__(self, definitions: Sequence[ToolDefinition]) -> None:
        self._definitions = {definition.name: definition for definition in definitions}
        if len(self._definitions) != len(definitions):
            raise ToolExecutionError("tool/skill names must be unique")

    def definition(self, name: str) -> ToolDefinition:
        try:
            return self._definitions[name]
        except KeyError:
            raise ToolExecutionError(f"unknown tool or skill: {name}") from None

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return the exact definitions for schema-preserving composition."""

        return tuple(self._definitions.values())

    def public_schemas(self) -> tuple[Mapping[str, JsonValue], ...]:
        return tuple(self._definitions[name].public_schema() for name in sorted(self._definitions))


class ParallelToolRuntime:
    """Run dependency-ready, read-only calls concurrently and join by call UUID."""

    def __init__(
        self,
        *,
        workspace: FilesystemSandbox,
        registry: ToolRegistry,
        id_factory: RuntimeIdFactory,
        maximum_parallel_calls: int = MAXIMUM_PARALLEL_TOOL_CALLS,
    ) -> None:
        if not 1 <= maximum_parallel_calls <= MAXIMUM_PARALLEL_TOOL_CALLS:
            raise ToolExecutionError(
                f"parallel tool width must be in [1, {MAXIMUM_PARALLEL_TOOL_CALLS}]"
            )
        self._workspace = workspace
        self._registry = registry
        self._ids = id_factory
        self._maximum_parallel_calls = maximum_parallel_calls
        self._results: dict[str, ToolResult] = {}
        self._declared: list[str] = []
        self._joined: list[str] = []
        self._frontier_count = 0
        self._active = 0
        self._max_active = 0
        self._active_lock = Lock()
        self._supplemental_content: dict[str, tuple] = {}

    def supplemental_content(self, call_id: str) -> tuple:
        """Host-captured transport sidecar, never part of canonical ToolResult."""
        with self._active_lock:
            return self._supplemental_content.get(call_id, ())

    def _track_enter(self) -> None:
        with self._active_lock:
            self._active += 1
            self._max_active = max(self._max_active, self._active)

    def _track_exit(self) -> None:
        with self._active_lock:
            self._active -= 1

    def _invoke_handler(
        self,
        call: ToolCall,
        *,
        start_barrier: Barrier | None = None,
    ) -> tuple[str, JsonValue | None, str | None]:
        definition = self._registry.definition(call.name)
        output: JsonValue | None = None
        error_code: str | None = None
        status = "completed"
        self._track_enter()
        try:
            if start_barrier is not None:
                start_barrier.wait(timeout=5)
            with diagnostic_scope(call.call_id, call.name) as diagnostics:
                output = freeze_json(definition.handler(self._workspace, call.arguments))
                if diagnostics.content:
                    with self._active_lock:
                        self._supplemental_content[call.call_id] = diagnostics.content
        except Exception as exc:  # retained as immutable outcome, never retried
            status = "immutable_failure"
            error_code = f"{type(exc).__name__}"
        finally:
            self._track_exit()
        return status, output, error_code

    @staticmethod
    def _result(
        call: ToolCall,
        *,
        result_id: str,
        group_id: str,
        frontier: int,
        status: str,
        output: JsonValue | None,
        error_code: str | None,
        before_blake3: str,
        after_blake3: str,
    ) -> ToolResult:
        core = {
            "result_id": result_id,
            "call_id": call.call_id,
            "name": call.name,
            "frontier": frontier,
            "parallel_group_id": group_id,
            "status": status,
            "output": output,
            "error_code": error_code,
            "workspace_before_blake3": before_blake3,
            "workspace_after_blake3": after_blake3,
        }
        return ToolResult(**core, receipt_blake3=blake3_hex(core))

    def _run_one(
        self,
        call: ToolCall,
        *,
        result_id: str,
        group_id: str,
        frontier: int,
    ) -> ToolResult:
        definition = self._registry.definition(call.name)
        before = self._workspace.snapshot(f"tool-{call.call_id}-before")
        status, output, error_code = self._invoke_handler(call)
        after = self._workspace.snapshot(f"tool-{call.call_id}-after")
        if definition.read_only and before.tree_blake3 != after.tree_blake3:
            status = "immutable_failure"
            error_code = "read_only_workspace_mutation"
            output = None
        return self._result(
            call,
            result_id=result_id,
            group_id=group_id,
            frontier=frontier,
            status=status,
            output=output,
            error_code=error_code,
            before_blake3=before.tree_blake3,
            after_blake3=after.tree_blake3,
        )

    def _dependency_failure(
        self,
        call: ToolCall,
        *,
        result_id: str,
        group_id: str,
        frontier: int,
    ) -> ToolResult:
        snapshot = self._workspace.snapshot(f"tool-{call.call_id}-dependency-failed")
        return self._result(
            call,
            result_id=result_id,
            group_id=group_id,
            frontier=frontier,
            status="immutable_failure",
            output=None,
            error_code="dependency_failed",
            before_blake3=snapshot.tree_blake3,
            after_blake3=snapshot.tree_blake3,
        )

    def _execute_group(self, calls: Sequence[ToolCall], *, parallel: bool) -> list[ToolResult]:
        frontier = self._frontier_count
        self._frontier_count += 1
        group_id = self._ids.new("parallel-tool-frontier" if parallel else "serial-tool-frontier")
        prepared = [
            (call, self._ids.new("tool-result"))
            for call in sorted(calls, key=lambda item: item.call_id)
        ]
        failed_dependencies = {
            call.call_id: any(self._results[dependency].status != "completed" for dependency in call.depends_on)
            for call, _result_id in prepared
        }
        runnable = [(call, result_id) for call, result_id in prepared if not failed_dependencies[call.call_id]]
        results: dict[str, ToolResult] = {}
        if parallel:
            # A parallel read-only frontier observes one common workspace
            # transition. Per-call full-tree snapshots scale as 2N and can
            # dominate the actual tool work at width 64; two shared snapshots
            # preserve the same ToolResult fields and make any false read-only
            # declaration fail the complete frontier consistently.
            before = self._workspace.snapshot(f"tool-{group_id}-before")
            barrier = Barrier(len(runnable)) if len(runnable) > 1 else None
            outcomes: dict[str, tuple[str, JsonValue | None, str | None]] = {}
            if len(runnable) > 1:
                with ThreadPoolExecutor(
                    max_workers=min(len(runnable), self._maximum_parallel_calls)
                ) as pool:
                    futures = {
                        call.call_id: pool.submit(
                            self._invoke_handler,
                            call,
                            start_barrier=barrier,
                        )
                        for call, _result_id in runnable
                    }
                    for call_id in sorted(futures):
                        outcomes[call_id] = futures[call_id].result()
            elif runnable:
                call, _result_id = runnable[0]
                outcomes[call.call_id] = self._invoke_handler(call)
            after = self._workspace.snapshot(f"tool-{group_id}-after")
            workspace_changed = before.tree_blake3 != after.tree_blake3
            for call, result_id in prepared:
                if workspace_changed:
                    status, output, error_code = (
                        "immutable_failure",
                        None,
                        "read_only_workspace_mutation",
                    )
                elif failed_dependencies[call.call_id]:
                    status, output, error_code = "immutable_failure", None, "dependency_failed"
                else:
                    status, output, error_code = outcomes[call.call_id]
                results[call.call_id] = self._result(
                    call,
                    result_id=result_id,
                    group_id=group_id,
                    frontier=frontier,
                    status=status,
                    output=output,
                    error_code=error_code,
                    before_blake3=before.tree_blake3,
                    after_blake3=after.tree_blake3,
                )
        else:
            for call, result_id in prepared:
                if failed_dependencies[call.call_id]:
                    results[call.call_id] = self._dependency_failure(
                        call, result_id=result_id, group_id=group_id, frontier=frontier
                    )
            for call, result_id in runnable:
                results[call.call_id] = self._run_one(
                    call, result_id=result_id, group_id=group_id, frontier=frontier
                )
        return [results[call.call_id] for call, _result_id in prepared]

    def execute(self, calls: Sequence[ToolCall]) -> tuple[ToolResult, ...]:
        if not calls:
            return ()
        incoming = list(calls)
        incoming_ids = [call.call_id for call in incoming]
        if len(set(incoming_ids)) != len(incoming_ids) or set(incoming_ids) & set(self._results):
            raise ToolExecutionError("tool call identity is duplicated or reused")
        known = set(self._results) | set(incoming_ids)
        if any(not set(call.depends_on) <= known or call.call_id in call.depends_on for call in incoming):
            raise ToolExecutionError("tool dependency refers to an unknown or self call")
        self._declared.extend(incoming_ids)
        pending = {call.call_id: call for call in incoming}
        batch_results: list[ToolResult] = []
        completed = set(self._results)
        while pending:
            ready = sorted(
                (call for call in pending.values() if set(call.depends_on) <= completed),
                key=lambda item: item.call_id,
            )
            if not ready:
                raise ToolExecutionError("tool dependency graph contains a cycle")
            parallel_ready = [
                call
                for call in ready
                if self._registry.definition(call.name).parallel_safe
                and self._registry.definition(call.name).read_only
            ]
            if parallel_ready:
                selected = parallel_ready[: self._maximum_parallel_calls]
                observed = self._execute_group(selected, parallel=len(selected) > 1)
            else:
                selected = [ready[0]]
                observed = self._execute_group(selected, parallel=False)
            for result in observed:
                self._results[result.call_id] = result
                self._joined.append(result.call_id)
                batch_results.append(result)
                pending.pop(result.call_id)
                completed.add(result.call_id)
        return tuple(batch_results)

    def trace(self) -> ToolTrace:
        results = tuple(
            sorted(self._results.values(), key=lambda item: (item.frontier, item.call_id))
        )
        joined = tuple(result.call_id for result in results)
        core = {
            "results": results,
            "declared_call_ids": tuple(self._declared),
            "joined_call_ids": joined,
            "frontier_count": self._frontier_count,
            "max_parallelism_observed": self._max_active,
            "retry_count": 0,
        }
        return ToolTrace(**core, trace_blake3=blake3_hex(core))


__all__ = [
    "MAXIMUM_PARALLEL_TOOL_CALLS",
    "ParallelToolRuntime",
    "ToolDefinition",
    "ToolExecutionError",
    "ToolHandler",
    "ToolRegistry",
]
