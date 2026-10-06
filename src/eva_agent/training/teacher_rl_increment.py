"""Offline RL continuation states from unchanged Codex teacher receipts.

Only prefixes whose workspace matches a completely retained snapshot are
materialized. The full teacher source is an audit sidecar, never part of the
actor prefix. These are continuation training records; provisioning a fresh
candidate host session and replaying its prior effects remains a runtime job.
"""

from __future__ import annotations

import base64
from collections import Counter
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid5

from eva_agent.codex_runtime import codex_turn_receipt_from_document, verify_codex_turn_receipt
from eva_agent.harness.skills import SkillCatalog
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes, canonical_value
from eva_agent.rubrics import CompiledRubricRegistry
from eva_agent.sandboxes.trajectory_rl_v1 import _reward_contract
from eva_agent.training.agent_judge import validate_full_trajectory


ROW_SCHEMA = "eva.teacher-continuation-rl-row.v1"
DATASET_SCHEMA = "eva.teacher-continuation-rl-increment.v1"
NAMESPACE = UUID("cf0d249a-82bf-4fa8-82a4-b33be18bf074")
_RUNTIME_ID_KEYS = frozenset({
    "event_id", "event_blake3", "tool_call_ids", "call_id", "codex_tool_call_id",
    "upstream_item_id", "receipt_blake3", "bridge_receipt_blake3",
})


class TeacherRLIncrementError(ValueError):
    """A source, policy prefix, workspace, or increment cannot be reopened."""


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise TeacherRLIncrementError("duplicate JSON key")
        value[key] = child
    return value


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise TeacherRLIncrementError("artifact topology differs")
    raw = path.read_bytes()
    document = json.loads(raw, object_pairs_hook=_object)
    if not isinstance(document, dict):
        raise TeacherRLIncrementError("artifact is not an object")
    return document, raw


def _member(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or pure.as_posix() != relative or any(part in {"", ".", ".."} for part in pure.parts):
        raise TeacherRLIncrementError("artifact path differs")
    path = root.joinpath(*pure.parts)
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root.parent):
        raise TeacherRLIncrementError("artifact parent topology differs")
    return path


def _snapshot_files(snapshot: Mapping[str, Any]) -> dict[str, bytes]:
    # validate_full_trajectory already reopens file bytes, modes and tree roots.
    return {
        row["path"]: base64.b64decode(row["content"]["$bytes_base64"], validate=True)
        for row in snapshot["files"]
    }


def _state_content(value: Any) -> Any:
    """Ignore freshly allocated trace IDs when deduplicating identical states."""
    if isinstance(value, Mapping):
        return {key: _state_content(child) for key, child in value.items() if key not in _RUNTIME_ID_KEYS}
    if isinstance(value, (tuple, list)):
        return [_state_content(child) for child in value]
    return value


def _source_tools(files: Mapping[str, bytes], skill_delivery: Mapping[str, Any],
                  offered_names: Sequence[str]) -> list[dict[str, Any]]:
    try:
        policy = json.loads(files[".eva/source-policy.json"], object_pairs_hook=_object)
        source_tools = policy["tools"]
    except (KeyError, TypeError, ValueError) as exc:
        raise TeacherRLIncrementError("retained source tool policy is absent") from exc
    if not isinstance(source_tools, list) or not source_tools:
        raise TeacherRLIncrementError("retained source tool policy differs")
    if policy.get("target_split") != "train":
        raise TeacherRLIncrementError("retained source is not training data")
    tools = []
    for tool in source_tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("input_schema"), dict):
            raise TeacherRLIncrementError("canonical source tool schema differs")
        tools.append({"type": "function", "function": {
            "name": tool["name"], "description": tool["description"],
            "parameters": tool["input_schema"],
        }})
    names = {item["function"]["name"] for item in tools}
    if len(names) != len(tools):
        raise TeacherRLIncrementError("canonical source tool name is duplicated")
    if skill_delivery.get("mode") == "search-then-load":
        for skill_tool in SkillCatalog(()).tool_definitions():
            if skill_tool.name not in names:
                tools.append({"type": "function", "function": {
                    "name": skill_tool.name, "description": skill_tool.description,
                    "parameters": canonical_value(skill_tool.parameters),
                }})
    available = {f"evamed/{tool['function']['name']}": tool for tool in tools}
    if set(offered_names) - set(available):
        raise TeacherRLIncrementError("offered source tool has no retained schema")
    return [available[name] for name in sorted(offered_names)]


def _validate_source(path: Path, registry: CompiledRubricRegistry):
    document, raw = _read(path)
    validated = validate_full_trajectory(path, registry=registry)
    if document != validated.document:
        raise TeacherRLIncrementError("source changed while reopening")
    metadata = document.get("provider_metadata")
    if not isinstance(metadata, dict) or metadata.get("raw_input_recorded") is not False:
        raise TeacherRLIncrementError("source provider capture differs")
    receipt = codex_turn_receipt_from_document(metadata.get("codex_turn_receipt", {}))
    verify_codex_turn_receipt(receipt)
    if (receipt.status != "completed" or receipt.visibility != "actor-public"
            or document["model_id"] != receipt.model or document.get("provider") != receipt.provider
            or document.get("assistant_output") != receipt.final_response):
        raise TeacherRLIncrementError("source Codex model or completion binding differs")
    results = document["tool_trace"]["results"]
    by_call: dict[str, Mapping[str, Any]] = {}
    for result in results:
        core = {key: value for key, value in result.items() if key != "receipt_blake3"}
        if result.get("receipt_blake3") != blake3_hex(core) or result["call_id"] in by_call:
            raise TeacherRLIncrementError("source tool result commitment differs")
        by_call[result["call_id"]] = result
    bindings = metadata.get("codex_to_pipeline_call_ids")
    if not isinstance(bindings, dict):
        raise TeacherRLIncrementError("source Codex tool binding is absent")
    codex_calls = {}
    for call in receipt.tool_calls:
        pipeline_id = bindings.get(call.tool_call_id)
        if pipeline_id not in by_call:
            continue
        observation = call.output.get("result", {}).get("structuredContent") if isinstance(call.output, Mapping) else None
        if not isinstance(observation, Mapping):
            raise TeacherRLIncrementError("source MCP observation is absent")
        observation_core = {key: value for key, value in observation.items() if key != "bridge_receipt_blake3"}
        if (observation.get("bridge_receipt_blake3") != blake3_hex(observation_core)
                or canonical_value(observation.get("tool_result")) != by_call[pipeline_id]
                or canonical_value(observation.get("arguments")) != canonical_value(call.arguments)):
            raise TeacherRLIncrementError("source MCP observation binding differs")
        codex_calls[pipeline_id] = call
    if set(codex_calls) != set(by_call):
        raise TeacherRLIncrementError("source has an unbound tool result")
    return document, raw, validated.rubric, by_call, codex_calls


def derive_teacher_rl_rows(path: str | Path, *, registry: CompiledRubricRegistry) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Derive exact continuation prefixes without provider calls or source edits."""
    document, raw, rubric, results, codex_calls = _validate_source(Path(path), registry)
    source_digest = blake3_bytes(raw)
    snapshots = {document[key]["tree_blake3"]: document[key] for key in ("workspace_after", "workspace_before")}
    initial = document["workspace_before"]
    files = _snapshot_files(initial)
    try:
        source_binding = json.loads(files["input/source-binding.json"], object_pairs_hook=_object)
        task_contract = json.loads(files["input/task-contract.json"], object_pairs_hook=_object)
    except (KeyError, ValueError) as exc:
        raise TeacherRLIncrementError("source task or benchmark binding is absent") from exc
    if (source_binding.get("candidate_id") != document["candidate_id"]
            or source_binding.get("domain") != rubric.domain or source_binding.get("stage") != rubric.stage
            or task_contract.get("domain") != rubric.domain or task_contract.get("stage") != rubric.stage
            or task_contract.get("private_reference_visible") is not False):
        raise TeacherRLIncrementError("source benchmark or stage binding differs")
    skills = document.get("skill_delivery", {})
    tools = _source_tools(files, skills, document["provider_metadata"]["codex_turn_receipt"]["offered_mcp_tool_names"])
    messages = document["messages"]
    rows: list[dict[str, Any]] = []
    declared: set[str] = set()
    observed: list[str] = []
    seen_events: set[str] = set()
    state_digest = initial["tree_blake3"]
    unavailable = 0
    for ordinal, event in enumerate(messages):
        if event["event_id"] in seen_events:
            raise TeacherRLIncrementError("source policy event is duplicated")
        seen_events.add(event["event_id"])
        ids = event["tool_call_ids"]
        if event["role"] == "assistant" and ids:
            if set(declared) != set(observed) or len(ids) != len(set(ids)) or any(call_id in declared for call_id in ids):
                raise TeacherRLIncrementError("source tool frontier is not joined")
            calls = event.get("content", {}).get("codex_tool_calls")
            if not isinstance(calls, list) or [call.get("call_id") for call in calls] != ids:
                raise TeacherRLIncrementError("source policy tool declarations differ")
            for call in calls:
                call_id = call["call_id"]
                codex = codex_calls.get(call_id)
                if (codex is None or call.get("codex_tool_call_id") != codex.tool_call_id
                        or call.get("fully_qualified_name") != codex.fully_qualified_name
                        or call.get("arguments") != canonical_value(codex.arguments)
                        or results[call_id]["workspace_before_blake3"] != state_digest):
                    raise TeacherRLIncrementError("source policy call or workspace boundary differs")
            if observed:
                snapshot = snapshots.get(state_digest)
                if snapshot is None:
                    unavailable += 1
                else:
                    prefix = messages[:ordinal]
                    state_key = blake3_hex({
                        "candidate_id": document["candidate_id"], "stage": rubric.stage,
                        "workspace_tree_blake3": state_digest,
                        "policy_content": _state_content(prefix),
                    })
                    sandbox_id = str(uuid5(NAMESPACE, state_key))
                    core = {
                        "schema": ROW_SCHEMA, "sandbox_id": sandbox_id, "split": "train",
                        "domain": rubric.domain, "stage": rubric.stage,
                        "instruction": task_contract["instruction"],
                        "starting_point": {
                            "origin": "retained_teacher_policy_prefix", "position": "before_tool_frontier",
                            "before_event_id": event["event_id"], "before_event_ordinal": ordinal,
                            "observed_tool_result_count": len(observed), "state_blake3": state_key,
                            "source_workspace_snapshot_label": snapshot["label"],
                        },
                        "actor_context": {"messages": prefix, "tools": tools, "skill_delivery": skills},
                        "workspace_initial_state": snapshot,
                        "benchmark_source_binding": source_binding,
                        "rubric_table": document["rubric_table"],
                        "reward_calculation_contract": _reward_contract(registry, rubric),
                        "source": {
                            "schema": document["schema"], "task_id": document["task_id"],
                            "sandbox_id": document["sandbox_id"], "candidate_id": document["candidate_id"],
                            "model_id": document["model_id"], "route_id": document["route_id"],
                            "file_blake3": source_digest, "audit_path": f"sources/{source_digest}.json",
                        },
                        "runtime_contract": {
                            "existing_candidate_resolver_required": True,
                            "candidate_id": document["candidate_id"],
                            "replay_prior_host_effects_required": True,
                            "prior_call_ids": list(observed),
                            "host_session_materialized": False,
                            "future_source_events_actor_visible": False,
                        },
                        "construction_policy": {
                            "quality_selection_deferred": True, "admission_claimed": False,
                            "benchmark_initial_duplicate": False,
                            "full_source_retained_as_audit_only": True,
                            "source_tool_and_skill_trace_retained": True,
                            "provider_calls": 0,
                        },
                    }
                    rows.append({**canonical_value(core), "row_blake3": blake3_hex(core)})
            declared.update(ids)
        elif event["role"] == "tool":
            if len(ids) != 1 or ids[0] not in declared or ids[0] in observed:
                raise TeacherRLIncrementError("source policy tool join differs")
            result = results[ids[0]]
            expected_content = {key: result[key] for key in ("error_code", "output", "receipt_blake3", "status")}
            if event["content"] != expected_content:
                raise TeacherRLIncrementError("source policy observation differs")
            observed.append(ids[0])
            state_digest = result["workspace_after_blake3"]
    if set(declared) != set(results) or set(observed) != set(results) or state_digest != document["workspace_after"]["tree_blake3"]:
        raise TeacherRLIncrementError("source terminal tool or workspace continuity differs")
    return rows, {
        "path": f"sources/{source_digest}.json", "file_blake3": source_digest,
        "byte_count": len(raw), "task_id": document["task_id"],
        "model_id": document["model_id"], "route_id": document["route_id"],
        "derived_row_count": len(rows), "unretained_workspace_frontier_count": unavailable,
        "tool_call_count": len(results),
        "skill_call_count": sum(row["name"] in {"search_skills", "load_skill"} for row in results.values()),
    }


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def _excluded_states(roots: Sequence[Path], registry: CompiledRubricRegistry) -> set[str]:
    states: set[str] = set()
    for root in roots:
        verify_teacher_rl_increment(root, registry=registry)
        with (root / "continuations.jsonl").open() as stream:
            states.update(json.loads(line)["starting_point"]["state_blake3"] for line in stream)
    return states


def build_teacher_rl_increment(*, source_paths: Sequence[Path], output_root: Path,
                               registry: CompiledRubricRegistry, exclude_roots: Sequence[Path] = ()) -> dict[str, Any]:
    """Write a new standalone increment and independently reopen its contents."""
    if output_root.exists() or output_root.is_symlink():
        raise TeacherRLIncrementError("increment output already exists")
    excluded = _excluded_states(exclude_roots, registry)
    seen = set(excluded)
    sources = []
    source_bytes: dict[str, bytes] = {}
    rows = []
    duplicates = 0
    for path in sorted(set(map(Path, source_paths))):
        derived, metadata = derive_teacher_rl_rows(path, registry=registry)
        if metadata["file_blake3"] in source_bytes:
            continue
        raw = path.read_bytes()
        if blake3_bytes(raw) != metadata["file_blake3"]:
            raise TeacherRLIncrementError("source changed during export")
        sources.append(metadata)
        source_bytes[metadata["file_blake3"]] = raw
        for row in derived:
            state = row["starting_point"]["state_blake3"]
            if state in seen:
                duplicates += 1
                continue
            seen.add(state)
            rows.append(row)
    rows.sort(key=lambda row: row["sandbox_id"])
    payload = b"".join(canonical_json_bytes(row) for row in rows)
    core = {
        "schema": DATASET_SCHEMA, "row_count": len(rows), "source_count": len(sources),
        "sources": sources, "rubric_registry_digest": registry.digest,
        "rows_path": "continuations.jsonl", "rows_file_blake3": blake3_bytes(payload),
        "rows_byte_count": len(payload), "excluded_state_blake3": sorted(excluded),
        "duplicate_state_count": duplicates,
        "counts_by_model": dict(sorted(Counter(row["source"]["model_id"] for row in rows).items())),
        "counts_by_stage": dict(sorted(Counter(row["stage"] for row in rows).items())),
        "provider_calls": 0, "host_sessions_materialized": 0,
        "full_sources_actor_visible": False, "published_baseline_modified": False,
    }
    manifest = {**core, "manifest_blake3": blake3_hex(core)}
    output_root.mkdir(mode=0o700, parents=True)
    for metadata in sources:
        _write(output_root / metadata["path"], source_bytes[metadata["file_blake3"]])
    _write(output_root / "continuations.jsonl", payload)
    _write(output_root / "manifest.json", canonical_json_bytes(manifest))
    return verify_teacher_rl_increment(output_root, registry=registry)


def verify_teacher_rl_increment(output_root: str | Path, *, registry: CompiledRubricRegistry) -> dict[str, Any]:
    """Reopen original source receipts and independently derive every row."""
    root = Path(output_root)
    manifest, _ = _read(root / "manifest.json")
    core = {key: value for key, value in manifest.items() if key != "manifest_blake3"}
    if (manifest.get("schema") != DATASET_SCHEMA or manifest.get("manifest_blake3") != blake3_hex(core)
            or manifest.get("rubric_registry_digest") != registry.digest
            or manifest.get("provider_calls") != 0 or manifest.get("host_sessions_materialized") != 0
            or manifest.get("full_sources_actor_visible") is not False
            or manifest.get("published_baseline_modified") is not False):
        raise TeacherRLIncrementError("increment manifest commitment differs")
    expected = []
    seen = set(manifest["excluded_state_blake3"])
    source_digests = set()
    duplicates = 0
    for source in manifest["sources"]:
        path = _member(root, source["path"])
        derived, reopened = derive_teacher_rl_rows(path, registry=registry)
        if reopened != source or source["file_blake3"] in source_digests:
            raise TeacherRLIncrementError("increment source inventory differs")
        source_digests.add(source["file_blake3"])
        for row in derived:
            state = row["starting_point"]["state_blake3"]
            if state in seen:
                duplicates += 1
                continue
            seen.add(state)
            expected.append(row)
    expected.sort(key=lambda row: row["sandbox_id"])
    payload = _member(root, manifest["rows_path"]).read_bytes()
    if (payload != b"".join(canonical_json_bytes(row) for row in expected)
            or len(payload) != manifest["rows_byte_count"]
            or blake3_bytes(payload) != manifest["rows_file_blake3"]
            or len(expected) != manifest["row_count"] or len(source_digests) != manifest["source_count"]
            or duplicates != manifest["duplicate_state_count"]
            or dict(Counter(row["source"]["model_id"] for row in expected)) != manifest["counts_by_model"]
            or dict(Counter(row["stage"] for row in expected)) != manifest["counts_by_stage"]):
        raise TeacherRLIncrementError("increment rows differ from retained sources")
    return {
        "schema": "eva.teacher-continuation-rl-verification.v1", "valid": True,
        "row_count": len(expected), "source_count": len(source_digests),
        "manifest_blake3": manifest["manifest_blake3"], "provider_calls": 0,
        "host_sessions_materialized": 0,
        "checks": ["unchanged_source_bytes", "codex_receipt_and_tool_observations", "complete_tool_joins",
                   "exact_retained_workspace", "prior_policy_context_only", "compiled_stage_rubric",
                   "source_tool_and_skill_capture", "unique_continuation_states", "independent_row_reconstruction"],
    }


def select_nonterminal_teacher_rl_increment(output_root: Path, *, selection_output: Path) -> dict[str, Any]:
    """Index useful candidate starts without changing the raw audit export.

    This is not a quality/admission gate: a failed intermediate state is useful,
    but a state whose own stage is already finished is not a new task. Selected
    candidates still require the host-session provisioning recorded in each row.
    """
    from .execution_verified_sft import _STAGE_COMPLETION_TOOL

    manifest, _ = _read(output_root / "manifest.json")
    core = {key: value for key, value in manifest.items() if key != "manifest_blake3"}
    if manifest.get("schema") != DATASET_SCHEMA or manifest["manifest_blake3"] != blake3_hex(core):
        raise TeacherRLIncrementError("raw increment manifest differs")
    payload = _member(output_root, manifest["rows_path"]).read_bytes()
    if blake3_bytes(payload) != manifest["rows_file_blake3"]:
        raise TeacherRLIncrementError("raw increment rows differ")
    sources = {}
    selected, excluded = [], []
    for line in payload.splitlines():
        row = json.loads(line, object_pairs_hook=_object)
        source = row["source"]
        digest = source["file_blake3"]
        if digest not in sources:
            document, raw = _read(_member(output_root, source["audit_path"]))
            if blake3_bytes(raw) != digest:
                raise TeacherRLIncrementError("raw increment source differs")
            sources[digest] = {result["call_id"]: result for result in document["tool_trace"]["results"]}
        prior = [sources[digest][call_id] for call_id in row["runtime_contract"]["prior_call_ids"]]
        finished = any(
            result["name"] == _STAGE_COMPLETION_TOOL[row["stage"]]
            and result["status"] == "completed"
            and result["output"].get("gate_passed") is True
            and result["output"].get("stage") == ("S5" if row["stage"] == "E2E" else row["stage"])
            for result in prior
        )
        if finished:
            excluded.append({"sandbox_id": row["sandbox_id"], "reason": "target_stage_already_completed"})
        else:
            selected.append(row["sandbox_id"])
    report = {
        "schema": "eva.teacher-continuation-nonterminal-selection.v1",
        "raw_manifest_blake3": manifest["manifest_blake3"],
        "raw_candidate_count": len(selected) + len(excluded),
        "selected_candidate_count": len(selected), "selected_sandbox_ids": selected,
        "excluded": excluded, "host_sessions_materialized": 0,
        "usable_hosted_sandbox_count": 0, "provider_calls": 0,
        "raw_export_modified": False, "quality_selection_deferred": True,
    }
    _write(selection_output, canonical_json_bytes(report))
    return report
