"""Read-only tools for an agent judge inspecting committed workspace evidence.

This module never accepts a filesystem root.  It reopens the byte-bearing
``WorkspaceSnapshot`` objects already committed by an ``EvidenceBundle`` and
serves bounded list/read/search/diff observations from those immutable values.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from threading import Barrier, Lock
from typing import Any, Callable, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator

from .contracts import (
    EvidenceBundle,
    FileSnapshot,
    JudgeAgentTrace,
    JudgeToolResult,
    JsonValue,
    RuntimeIdFactory,
    ToolCall,
    TrajectoryEvent,
    WorkspaceSnapshot,
    freeze_json,
)
from .digests import blake3_bytes, blake3_hex, is_blake3
from .tools import MAXIMUM_PARALLEL_TOOL_CALLS
from .ids import RandomUUIDFactory


TOOL_NAMES = (
    "workspace_diff",
    "workspace_list",
    "workspace_read",
    "workspace_search",
)
_MODE = re.compile(r"^[0-7]{4}$")


class JudgeWorkspaceError(ValueError):
    """A committed snapshot or judge-tool boundary failed closed."""


class _ToolFailure(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class _ToolDefinition:
    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[[Mapping[str, Any]], tuple[Mapping[str, Any], tuple[str, ...], bool]]

    def schema(self) -> Mapping[str, JsonValue]:
        return freeze_json(
            {
                "type": "function",
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": True,
            }
        )  # type: ignore[return-value]


def _object_schema(properties: Mapping[str, Any]) -> dict[str, Any]:
    """All properties are required, as required by strict function schemas."""

    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }


_SNAPSHOT = {"type": "string", "enum": ["before", "after"]}
_NULLABLE_PATH = {
    "anyOf": [
        {"type": "string", "minLength": 1, "maxLength": 4096},
        {"type": "null"},
    ]
}


def _path(value: Any, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise _ToolFailure("unsafe_path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise _ToolFailure("unsafe_path")
    return value


def _snapshot_core(snapshot: WorkspaceSnapshot) -> dict[str, Any]:
    return {
        "files": snapshot.files,
        "file_count": snapshot.file_count,
        "byte_count": snapshot.byte_count,
    }


def _evidence_core(evidence: EvidenceBundle) -> dict[str, Any]:
    # Kept local so this read-only layer does not import the mutable pipeline runner.
    return {
        "bundle_id": evidence.bundle_id,
        "rollout_id": evidence.rollout_id,
        "sandbox_manifest": evidence.sandbox_manifest,
        "model": evidence.model,
        "policy_visible_context": evidence.policy_visible_context,
        "context_blake3": evidence.context_blake3,
        "workspace_before": evidence.workspace_before,
        "workspace_after": evidence.workspace_after,
        "tool_trace": evidence.tool_trace,
        "policy_events": evidence.policy_events,
        "assistant_output": evidence.assistant_output,
        "provider_receipt_blake3": evidence.provider_receipt_blake3,
        "safe_provider_metadata": evidence.safe_provider_metadata,
    }


def _result_core(
    *,
    result_id: str,
    call: ToolCall,
    frontier: int,
    group_id: str,
    status: str,
    output: JsonValue | None,
    error_code: str | None,
    evidence_refs: tuple[str, ...],
    content_inspection: bool,
    evidence_bundle_blake3: str,
) -> dict[str, Any]:
    return {
        "result_id": result_id,
        "call_id": call.call_id,
        "name": call.name,
        "arguments": call.arguments,
        "frontier": frontier,
        "parallel_group_id": group_id,
        "status": status,
        "output": output,
        "error_code": error_code,
        "inspected_evidence_refs": evidence_refs,
        "content_inspection": content_inspection,
        "evidence_bundle_blake3": evidence_bundle_blake3,
    }


class JudgeWorkspaceTools:
    """Bounded parallel tool runtime over one immutable evidence bundle."""

    def __init__(
        self,
        evidence: EvidenceBundle,
        *,
        id_factory: RuntimeIdFactory | None = None,
        maximum_parallel_tools: int = MAXIMUM_PARALLEL_TOOL_CALLS,
    ) -> None:
        if not isinstance(evidence, EvidenceBundle):
            raise JudgeWorkspaceError("judge tools require an EvidenceBundle")
        if not 1 <= maximum_parallel_tools <= MAXIMUM_PARALLEL_TOOL_CALLS:
            raise JudgeWorkspaceError(
                f"judge tool width must be in [1,{MAXIMUM_PARALLEL_TOOL_CALLS}]"
            )
        if not is_blake3(evidence.bundle_blake3) or evidence.bundle_blake3 != blake3_hex(
            _evidence_core(evidence)
        ):
            raise JudgeWorkspaceError("evidence bundle BLAKE3 differs")
        self._evidence = evidence
        self._ids = id_factory or RandomUUIDFactory()
        self._maximum_parallel_tools = maximum_parallel_tools
        self._snapshots = {
            "before": self._validate_snapshot(evidence.workspace_before, "before"),
            "after": self._validate_snapshot(evidence.workspace_after, "after"),
        }
        self._definitions = self._build_definitions()
        self._results: list[JudgeToolResult] = []
        self._declared: list[str] = []
        self._seen: set[str] = set()
        self._frontier_count = 0
        self._active = 0
        self._max_active = 0
        self._lock = Lock()

    @staticmethod
    def _validate_snapshot(
        snapshot: WorkspaceSnapshot, label: str
    ) -> Mapping[str, FileSnapshot]:
        if not isinstance(snapshot, WorkspaceSnapshot) or not snapshot.label:
            raise JudgeWorkspaceError(f"workspace {label} snapshot differs")
        files: dict[str, FileSnapshot] = {}
        previous = ""
        total = 0
        for row in snapshot.files:
            if not isinstance(row, FileSnapshot):
                raise JudgeWorkspaceError(f"workspace {label} file row differs")
            try:
                normalized = _path(row.path)
            except _ToolFailure as exc:
                raise JudgeWorkspaceError(
                    f"workspace {label} contains an unsafe or symlink-like path"
                ) from exc
            assert normalized is not None
            # A committed file cannot simultaneously be an ancestor directory.
            if any(
                normalized.startswith(existing + "/") or existing.startswith(normalized + "/")
                for existing in files
            ):
                raise JudgeWorkspaceError(
                    f"workspace {label} contains symlink-like prefix topology"
                )
            if normalized in files or (previous and normalized < previous):
                raise JudgeWorkspaceError(f"workspace {label} paths differ")
            if (
                not isinstance(row.content, bytes)
                or type(row.byte_count) is not int
                or row.byte_count != len(row.content)
                or not isinstance(row.mode, str)
                or _MODE.fullmatch(row.mode) is None
                or not is_blake3(row.content_blake3)
                or row.content_blake3 != blake3_bytes(row.content)
            ):
                raise JudgeWorkspaceError(f"workspace {label} file commitment differs")
            files[normalized] = row
            previous = normalized
            total += row.byte_count
        if (
            snapshot.file_count != len(files)
            or snapshot.byte_count != total
            or not is_blake3(snapshot.tree_blake3)
            or snapshot.tree_blake3 != blake3_hex(_snapshot_core(snapshot))
        ):
            raise JudgeWorkspaceError(f"workspace {label} tree commitment differs")
        return files

    def _build_definitions(self) -> Mapping[str, _ToolDefinition]:
        definitions = (
            _ToolDefinition(
                "workspace_list",
                "List paths and committed metadata in the before or after workspace snapshot.",
                _object_schema(
                    {
                        "snapshot": _SNAPSHOT,
                        "prefix": _NULLABLE_PATH,
                        "max_entries": {"type": "integer", "minimum": 1, "maximum": 1000},
                    }
                ),
                self._list,
            ),
            _ToolDefinition(
                "workspace_read",
                "Read bounded file content from a committed workspace snapshot.",
                _object_schema(
                    {
                        "snapshot": _SNAPSHOT,
                        "path": {"type": "string", "minLength": 1, "maxLength": 4096},
                        "offset": {"type": "integer", "minimum": 0},
                        "max_bytes": {"type": "integer", "minimum": 1, "maximum": 65536},
                    }
                ),
                self._read,
            ),
            _ToolDefinition(
                "workspace_search",
                "Search literal text across committed file contents and return bounded matches.",
                _object_schema(
                    {
                        "snapshot": _SNAPSHOT,
                        "query": {"type": "string", "minLength": 1, "maxLength": 512},
                        "path_prefix": _NULLABLE_PATH,
                        "case_sensitive": {"type": "boolean"},
                        "max_matches": {"type": "integer", "minimum": 1, "maximum": 200},
                    }
                ),
                self._search,
            ),
            _ToolDefinition(
                "workspace_diff",
                "Compare committed before/after path, mode, size, and content-digest metadata.",
                _object_schema(
                    {
                        "path": _NULLABLE_PATH,
                        "max_files": {"type": "integer", "minimum": 1, "maximum": 500},
                    }
                ),
                self._diff,
            ),
        )
        for definition in definitions:
            Draft202012Validator.check_schema(dict(definition.parameters))
        return {definition.name: definition for definition in definitions}

    def response_api_schemas(self) -> tuple[Mapping[str, JsonValue], ...]:
        """Return stable strict OpenAI Responses-style function schemas."""

        return tuple(self._definitions[name].schema() for name in TOOL_NAMES)

    def _snapshot(self, name: Any) -> tuple[WorkspaceSnapshot, Mapping[str, FileSnapshot]]:
        if name not in self._snapshots:
            raise _ToolFailure("unknown_snapshot")
        snapshot = self._evidence.workspace_before if name == "before" else self._evidence.workspace_after
        return snapshot, self._snapshots[name]

    @staticmethod
    def _ref(snapshot: str, path: str) -> str:
        return f"workspace:{snapshot}:{path}"

    def _list(
        self, arguments: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], tuple[str, ...], bool]:
        name = arguments["snapshot"]
        prefix = _path(arguments["prefix"], nullable=True)
        snapshot, files = self._snapshot(name)
        selected = [
            row
            for path, row in files.items()
            if prefix is None or path == prefix or path.startswith(prefix + "/")
        ]
        truncated = len(selected) > arguments["max_entries"]
        selected = selected[: arguments["max_entries"]]
        output = {
            "snapshot": name,
            "tree_blake3": snapshot.tree_blake3,
            "entries": [
                {
                    "path": row.path,
                    "byte_count": row.byte_count,
                    "mode": row.mode,
                    "content_blake3": row.content_blake3,
                }
                for row in selected
            ],
            "truncated": truncated,
        }
        refs = tuple(self._ref(name, row.path) for row in selected)
        return output, refs, False

    def _read(
        self, arguments: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], tuple[str, ...], bool]:
        name = arguments["snapshot"]
        path = _path(arguments["path"])
        assert path is not None
        snapshot, files = self._snapshot(name)
        row = files.get(path)
        if row is None:
            raise _ToolFailure("path_not_found")
        offset = arguments["offset"]
        if offset > row.byte_count:
            raise _ToolFailure("offset_out_of_range")
        selected = row.content[offset : offset + arguments["max_bytes"]]
        try:
            content = selected.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            content = base64.b64encode(selected).decode("ascii")
            encoding = "base64"
        output = {
            "snapshot": name,
            "tree_blake3": snapshot.tree_blake3,
            "path": path,
            "content_blake3": row.content_blake3,
            "total_bytes": row.byte_count,
            "offset": offset,
            "returned_bytes": len(selected),
            "truncated": offset + len(selected) < row.byte_count,
            "encoding": encoding,
            "content": content,
        }
        return output, (self._ref(name, path),), True

    def _search(
        self, arguments: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], tuple[str, ...], bool]:
        name = arguments["snapshot"]
        prefix = _path(arguments["path_prefix"], nullable=True)
        snapshot, files = self._snapshot(name)
        selected = [
            row
            for path, row in files.items()
            if prefix is None or path == prefix or path.startswith(prefix + "/")
        ]
        if not selected:
            raise _ToolFailure("no_files_to_search")
        query = arguments["query"]
        needle = query if arguments["case_sensitive"] else query.casefold()
        matches: list[dict[str, Any]] = []
        inspected: list[FileSnapshot] = []
        limit = arguments["max_matches"]
        for row in selected:
            inspected.append(row)
            text = row.content.decode("utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), start=1):
                comparable = line if arguments["case_sensitive"] else line.casefold()
                start = comparable.find(needle)
                if start < 0:
                    continue
                matches.append(
                    {
                        "path": row.path,
                        "line_number": line_number,
                        "column": start + 1,
                        "line": line[:2000],
                        "content_blake3": row.content_blake3,
                    }
                )
                if len(matches) >= limit:
                    break
            if len(matches) >= limit:
                break
        output = {
            "snapshot": name,
            "tree_blake3": snapshot.tree_blake3,
            "query": query,
            "matches": matches,
            "match_count": len(matches),
            "truncated": len(matches) >= limit,
            "searched_file_count": len(inspected),
        }
        refs = tuple(self._ref(name, row.path) for row in inspected)
        return output, refs, True

    def _diff(
        self, arguments: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], tuple[str, ...], bool]:
        requested = _path(arguments["path"], nullable=True)
        before_snapshot, before = self._snapshot("before")
        after_snapshot, after = self._snapshot("after")
        paths = sorted(set(before) | set(after))
        if requested is not None:
            if requested not in before and requested not in after:
                raise _ToolFailure("path_not_found")
            paths = [requested]
        changed = [
            path
            for path in paths
            if path not in before
            or path not in after
            or before[path].content_blake3 != after[path].content_blake3
            or before[path].mode != after[path].mode
        ]
        total_changed = len(changed)
        truncated_files = total_changed > arguments["max_files"]
        changed = changed[: arguments["max_files"]]
        entries: list[dict[str, Any]] = []
        refs: list[str] = []
        for path in changed:
            old = before.get(path)
            new = after.get(path)
            status = "added" if old is None else "deleted" if new is None else "modified"
            entries.append(
                {
                    "path": path,
                    "status": status,
                    "before_content_blake3": None if old is None else old.content_blake3,
                    "after_content_blake3": None if new is None else new.content_blake3,
                    "before_byte_count": None if old is None else old.byte_count,
                    "after_byte_count": None if new is None else new.byte_count,
                    "before_mode": None if old is None else old.mode,
                    "after_mode": None if new is None else new.mode,
                }
            )
            if old is not None:
                refs.append(self._ref("before", path))
            if new is not None:
                refs.append(self._ref("after", path))
        output = {
            "before_tree_blake3": before_snapshot.tree_blake3,
            "after_tree_blake3": after_snapshot.tree_blake3,
            "entries": entries,
            "changed_file_count": total_changed,
            "returned_file_count": len(changed),
            "truncated_files": truncated_files,
        }
        return output, tuple(dict.fromkeys(refs)), False

    def _track_enter(self) -> None:
        with self._lock:
            self._active += 1
            self._max_active = max(self._max_active, self._active)

    def _track_exit(self) -> None:
        with self._lock:
            self._active -= 1

    def _execute_one(
        self,
        call: ToolCall,
        *,
        result_id: str,
        group_id: str,
        frontier: int,
        start_barrier: Barrier | None,
    ) -> JudgeToolResult:
        definition = self._definitions[call.name]
        status = "completed"
        output: JsonValue | None = None
        error_code: str | None = None
        evidence_refs: tuple[str, ...] = ()
        content_inspection = False
        self._track_enter()
        try:
            if start_barrier is not None:
                start_barrier.wait(timeout=5)
            errors = sorted(
                Draft202012Validator(definition.parameters).iter_errors(dict(call.arguments)),
                key=lambda error: tuple(str(part) for part in error.path),
            )
            if errors:
                raise _ToolFailure("invalid_arguments")
            raw_output, evidence_refs, content_inspection = definition.handler(call.arguments)
            frozen = freeze_json(raw_output)
            if not isinstance(frozen, Mapping):
                raise _ToolFailure("invalid_tool_output")
            output = frozen
        except _ToolFailure as exc:
            status = "immutable_failure"
            error_code = exc.code
            output = None
            evidence_refs = ()
            content_inspection = False
        except Exception as exc:  # bounded safe outcome; no raw exception text is exposed
            status = "immutable_failure"
            error_code = type(exc).__name__
            output = None
            evidence_refs = ()
            content_inspection = False
        finally:
            self._track_exit()
        core = _result_core(
            result_id=result_id,
            call=call,
            frontier=frontier,
            group_id=group_id,
            status=status,
            output=output,
            error_code=error_code,
            evidence_refs=evidence_refs,
            content_inspection=content_inspection,
            evidence_bundle_blake3=self._evidence.bundle_blake3,
        )
        return JudgeToolResult(
            **core,
            receipt_blake3=blake3_hex(core),
        )

    def execute_group(self, calls: Sequence[ToolCall]) -> tuple[JudgeToolResult, ...]:
        """Execute one model-turn call group concurrently, without retries."""

        values = tuple(calls)
        if (
            not values
            or len(values) > self._maximum_parallel_tools
            or any(not isinstance(call, ToolCall) for call in values)
        ):
            raise JudgeWorkspaceError(
                "judge tool group must fit the configured parallel-tool bound"
            )
        call_ids = [call.call_id for call in values]
        if len(set(call_ids)) != len(call_ids) or self._seen.intersection(call_ids):
            raise JudgeWorkspaceError("judge tool call identity is duplicated or reused")
        if any(call.name not in self._definitions for call in values):
            raise JudgeWorkspaceError("judge requested an unoffered workspace tool")
        if any(call.depends_on for call in values):
            raise JudgeWorkspaceError("same-turn judge workspace calls cannot declare dependencies")
        frontier = self._frontier_count
        group_id = self._ids.new("judge-parallel-tool-group")
        prepared = [
            (call, self._ids.new("judge-tool-result")) for call in values
        ]
        parallel_width = len(prepared)
        barrier = Barrier(parallel_width) if parallel_width > 1 else None
        with ThreadPoolExecutor(max_workers=parallel_width) as pool:
            futures = [
                pool.submit(
                    self._execute_one,
                    call,
                    result_id=result_id,
                    group_id=group_id,
                    frontier=frontier,
                    start_barrier=barrier if index < parallel_width else None,
                )
                for index, (call, result_id) in enumerate(prepared)
            ]
            results = tuple(future.result() for future in futures)
        self._declared.extend(call_ids)
        self._seen.update(call_ids)
        self._results.extend(results)
        self._frontier_count += 1
        return results

    @property
    def max_parallelism_observed(self) -> int:
        return self._max_active

    def trace(
        self,
        *,
        policy_events: Iterable[TrajectoryEvent],
        provider_turn_count: int,
    ) -> JudgeAgentTrace:
        """Commit the provider-visible judge events and accumulated tool evidence."""

        events = tuple(policy_events)
        results = tuple(self._results)
        joined = tuple(result.call_id for result in results)
        inspected = tuple(
            sorted(
                {
                    reference
                    for result in results
                    if result.status == "completed"
                    for reference in result.inspected_evidence_refs
                }
            )
        )
        content_count = sum(
            result.status == "completed" and result.content_inspection for result in results
        )
        core = {
            "policy_events": events,
            "results": results,
            "declared_call_ids": tuple(self._declared),
            "joined_call_ids": joined,
            "inspected_evidence_refs": inspected,
            "content_inspection_count": content_count,
            "frontier_count": self._frontier_count,
            "max_parallelism_observed": self._max_active,
            "provider_turn_count": provider_turn_count,
            "retry_count": 0,
            "evidence_bundle_blake3": self._evidence.bundle_blake3,
        }
        return JudgeAgentTrace(**core, trace_blake3=blake3_hex(core))


__all__ = ["JudgeWorkspaceError", "JudgeWorkspaceTools", "TOOL_NAMES"]
