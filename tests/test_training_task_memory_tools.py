from __future__ import annotations

import ast
import os
from pathlib import Path
from uuid import uuid4

from jsonschema import ValidationError
import pytest

from eva_agent.pipeline import (
    ContractError, FilesystemSandbox, ParallelToolRuntime, RandomUUIDFactory,
    ToolCall, ToolDefinition, ToolRegistry,
)
from eva_agent.pipeline.digests import blake3_bytes, canonical_json_bytes, canonical_value
from eva_agent.pipeline.workspace import SandboxError
from eva_agent.training.task_memory_tools import TASK_MEMORY_TOOLS, augment_registry


def _workspace(tmp_path, initial=None):
    return FilesystemSandbox(tmp_path, str(uuid4()), initial or {})


def _registry():
    return augment_registry(ToolRegistry(()))


def test_exact_automed_descriptors_and_original_seven_schemas_unchanged():
    names = ("execute_code", "materialize_evidence_selection", "materialize_plan",
             "retrieve_frozen_evidence", "submit_results", "search_skills", "load_skill")
    original = ToolRegistry(tuple(ToolDefinition(
        name=name, description="Immutable original " + name,
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda _workspace, _arguments: {},
    ) for name in names))
    before = canonical_json_bytes(original.public_schemas())
    augmented = augment_registry(original)
    retained = tuple(row for row in augmented.public_schemas()
                     if row["function"]["name"] in names)
    assert canonical_json_bytes(retained) == before
    assert canonical_json_bytes(original.public_schemas()) == before
    for name in names:
        assert augmented.definition(name) is original.definition(name)
    for descriptor in TASK_MEMORY_TOOLS:
        added = augmented.definition(descriptor["name"])
        assert added.description == descriptor["description"]
        assert canonical_value(added.input_schema) == descriptor["inputSchema"]
    assert augmented.definition("automed_write_note").parallel_safe is False
    assert augmented.definition("automed_write_note").read_only is False
    assert augmented.definition("automed_read_file").parallel_safe is True
    assert augmented.definition("automed_read_file").read_only is True
    with pytest.raises(ContractError, match="collide"):
        augment_registry(augmented)


def test_descriptors_equal_frozen_med_reference():
    reference = Path(os.environ.get("EVA_TASK_MEMORY_REFERENCE", str(
        Path(__file__).resolve().parents[2] / "EVA-Agent-training-timeout-isolation-20260914"
        / "training/automedbench_lite/public_tools.py")))
    if not reference.is_file():
        pytest.skip("optional frozen MED public-tool reference is unavailable")
    tree = ast.parse(reference.read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "TOOLS"
                              for target in node.targets))
    expected = tuple(ast.literal_eval(item) for item in assignment.value.left.elts[:2])
    assert canonical_json_bytes(TASK_MEMORY_TOOLS) == canonical_json_bytes(expected)


def test_real_note_survives_fresh_registry_and_runtime_with_normal_receipts(tmp_path):
    workspace = _workspace(tmp_path)
    runtime = ParallelToolRuntime(workspace=workspace, registry=_registry(), id_factory=RandomUUIDFactory())
    call = ToolCall(call_id=str(uuid4()), name="automed_write_note",
                    arguments={"name": "task-state.md", "content": "Next: validate S3 artifact."})
    result = runtime.execute((call,))[0]
    assert result.status == "completed"
    payload = b"Next: validate S3 artifact."
    assert (workspace.root / "notes/task-state.md").read_bytes() == payload
    assert canonical_value(result.output) == {
        "path": "notes/task-state.md", "before_blake3": None,
        "file_blake3": blake3_bytes(payload), "bytes": len(payload),
    }
    assert result.workspace_before_blake3 != result.workspace_after_blake3
    # Rebuild all tool definitions and runtime state: persistence is in the
    # caller's actual task workspace, not an earlier handler's transient state.
    reopened = ParallelToolRuntime(workspace=workspace, registry=_registry(), id_factory=RandomUUIDFactory())
    read = reopened.execute((ToolCall(call_id=str(uuid4()), name="automed_read_file",
        arguments={"path": "notes/task-state.md"}),))[0]
    assert read.status == "completed"
    assert canonical_value(read.output) == {
        "path": "notes/task-state.md", "offset": 0, "total_bytes": len(payload),
        "file_blake3": blake3_bytes(payload), "content": payload.decode(), "complete": True,
    }
    assert read.workspace_before_blake3 == read.workspace_after_blake3 == result.workspace_after_blake3
    updated = _registry().definition("automed_write_note").handler(
        workspace, {"name": "task-state.md", "content": "Validated; continue S4."})
    assert updated["before_blake3"] == blake3_bytes(payload)
    assert _registry().definition("automed_read_file").handler(
        workspace, {"path": "notes/task-state.md"})["content"] == "Validated; continue S4."
    other = _workspace(tmp_path)
    with pytest.raises(ContractError):
        _registry().definition("automed_read_file").handler(other, {"path": "notes/task-state.md"})
    assert not (other.root / "notes").exists()


def test_bounded_read_matches_public_result_fields_and_does_not_create_parents(tmp_path):
    workspace = _workspace(tmp_path, {"task.json": b"abcdef", "outputs/result.txt": b"ok",
                                     "notes/large.txt": b"x" * 1048577})
    read = _registry().definition("automed_read_file").handler
    assert read(workspace, {"path": "task.json", "offset": 1, "limit": 3}) == {
        "path": "task.json", "offset": 1, "total_bytes": 6,
        "file_blake3": blake3_bytes(b"abcdef"), "content": "bcd", "complete": False,
    }
    assert read(workspace, {"path": "outputs/result.txt"})["content"] == "ok"
    with pytest.raises(ContractError, match="too_large"):
        read(workspace, {"path": "notes/large.txt"})
    before = workspace.snapshot("before").tree_blake3
    with pytest.raises(ContractError):
        read(workspace, {"path": "notes/missing/note.md"})
    assert workspace.snapshot("after").tree_blake3 == before
    assert not (workspace.root / "notes/missing").exists()


@pytest.mark.parametrize("relative", ("private/judge.json", "../other/notes/a.md", "/etc/passwd",
                                      "notes/../../private.json", "notes//a.md"))
def test_read_rejects_private_and_escaping_paths(tmp_path, relative):
    workspace = _workspace(tmp_path, {"private/judge.json": b"not policy visible"})
    with pytest.raises(ContractError):
        _registry().definition("automed_read_file").handler(workspace, {"path": relative})


@pytest.mark.parametrize("arguments", (
    {"name": "../escape", "content": "bad"}, {"name": "nested/note", "content": "bad"},
    {"name": "x" * 65, "content": "bad"}, {"name": "ok", "content": "x" * 16385},
    {"name": "ok", "content": "bad", "path": "elsewhere"},
))
def test_note_rejects_arguments_outside_existing_schema(tmp_path, arguments):
    workspace = _workspace(tmp_path)
    with pytest.raises(ValidationError):
        _registry().definition("automed_write_note").handler(workspace, arguments)
    assert not (workspace.root / "notes").exists()


@pytest.mark.parametrize("link_kind", ("parent", "file", "hardlink"))
def test_read_and_write_reject_linked_notes_without_touching_other_task(tmp_path, link_kind):
    outside = tmp_path / "other-task"
    outside.mkdir()
    target = outside / "state.md"
    target.write_bytes(b"other task state")
    workspace = _workspace(tmp_path / "sandboxes")
    if link_kind == "parent":
        (workspace.root / "notes").symlink_to(outside, target_is_directory=True)
    else:
        (workspace.root / "notes").mkdir()
        if link_kind == "file":
            (workspace.root / "notes/state.md").symlink_to(target)
        else:
            os.link(target, workspace.root / "notes/state.md")
    registry = _registry()
    with pytest.raises(SandboxError):
        registry.definition("automed_read_file").handler(workspace, {"path": "notes/state.md"})
    with pytest.raises(SandboxError):
        registry.definition("automed_write_note").handler(workspace, {"name": "state.md", "content": "changed"})
    assert target.read_bytes() == b"other task state"
