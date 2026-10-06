"""Opt-in, task-local note tools; canonical training definitions stay unchanged."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from eva_agent.pipeline.contracts import ContractError, JsonValue
from eva_agent.pipeline.digests import blake3_bytes, canonical_value
from eva_agent.pipeline.tools import ToolDefinition, ToolRegistry
from eva_agent.pipeline.workspace import FilesystemSandbox


TASK_MEMORY_PROFILE = "public-notes-v1"
# Exact public descriptors from automedbench_lite/public_tools.py.  Training
# adds its normal execution annotations only in ToolDefinition's outer schema.
TASK_MEMORY_TOOLS = (
    {"name": "automed_read_file", "description": "Read a bounded public task, notes, or output text file from this case workspace.",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string", "maxLength": 160},
        "offset": {"type": "integer", "minimum": 0, "maximum": 1048576},
        "limit": {"type": "integer", "minimum": 1, "maximum": 32768}}, "required": ["path"], "additionalProperties": False}},
    {"name": "automed_write_note", "description": "Persist a public progress/decision note in notes/ for subsequent turns and resumed work.",
     "inputSchema": {"type": "object", "properties": {"name": {"type": "string", "pattern": "^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$"},
        "content": {"type": "string", "maxLength": 16384}}, "required": ["name", "content"], "additionalProperties": False}},
)
TASK_MEMORY_TOOL_NAMES = tuple(item["name"] for item in TASK_MEMORY_TOOLS)
_READ_VALIDATOR, _NOTE_VALIDATOR = (
    Draft202012Validator(item["inputSchema"]) for item in TASK_MEMORY_TOOLS
)


def _read_file(workspace: FilesystemSandbox, arguments: Mapping[str, JsonValue]) -> dict[str, Any]:
    _READ_VALIDATOR.validate(canonical_value(arguments))
    relative = arguments["path"]
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or not pure.parts
        or ".." in pure.parts
        or (relative != "task.json" and not relative.startswith(("notes/", "outputs/")))
    ):
        raise ContractError("public_text_path_not_allowed")
    # FilesystemSandbox.read_bytes checks topology and byte stability.  Reject
    # absent paths first because its path helper can create missing parents.
    if not (workspace.root / relative).is_file():
        raise ContractError("public_text_file_absent")
    data = workspace.read_bytes(relative)
    if len(data) > 1048576:
        raise ContractError("public_text_file_too_large")
    offset, limit = arguments.get("offset", 0), arguments.get("limit", 16384)
    return {"path": relative, "offset": offset, "total_bytes": len(data),
            "file_blake3": blake3_bytes(data),
            "content": data[offset:offset + limit].decode("utf-8", errors="replace"),
            "complete": offset == 0 and limit >= len(data)}


def _write_note(workspace: FilesystemSandbox, arguments: Mapping[str, JsonValue]) -> dict[str, Any]:
    _NOTE_VALIDATOR.validate(canonical_value(arguments))
    relative = "notes/" + arguments["name"]
    path = workspace.root / relative
    before = (blake3_bytes(workspace.read_bytes(relative))
              if path.exists() or path.is_symlink() else None)
    data = arguments["content"].encode()
    workspace.write_bytes(relative, data)
    return {"path": relative, "before_blake3": before,
            "file_blake3": blake3_bytes(workspace.read_bytes(relative)), "bytes": len(data)}


def augment_registry(registry: ToolRegistry) -> ToolRegistry:
    """Add only the public note tools, bound at invocation to this task's sandbox.

    There is no ambient workspace, cross-task store, container write, or custom
    receipt layer.  ParallelToolRuntime serializes the note mutation and records
    its actual workspace snapshots and ordinary tool result.
    """
    original = registry.definitions()
    if {item.name for item in original}.intersection(TASK_MEMORY_TOOL_NAMES):
        raise ContractError("task memory tool names collide with existing registry")
    definitions = tuple(
        ToolDefinition(name=descriptor["name"], description=descriptor["description"],
                       input_schema=descriptor["inputSchema"], handler=handler,
                       parallel_safe=read_only, read_only=read_only)
        for descriptor, handler, read_only in (
            (TASK_MEMORY_TOOLS[0], _read_file, True),
            (TASK_MEMORY_TOOLS[1], _write_note, False),
        )
    )
    return ToolRegistry((*original, *definitions))


__all__ = ["TASK_MEMORY_PROFILE", "TASK_MEMORY_TOOLS", "TASK_MEMORY_TOOL_NAMES", "augment_registry"]
