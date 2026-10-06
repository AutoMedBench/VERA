"""Versioned Tier-A policy for Codex-native medical-research edits.

This is an additive sidecar contract.  It does not translate or extend any
EvaMed tool, MCP, rubric, sandbox, or evidence schema.  The narrow Tier-A
surface permits a fresh S3/S4/E2E actor turn to emit receipt-visible
``fileChange`` items while the existing, schema-preserving MCP tools remain
available.  Every other Codex-native capability remains closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from eva_agent.codex_runtime import CodexSkill
from eva_agent.codex_pipeline.turn_mcp import (
    TURN_MCP_DIRECTORY_PREFIX,
    TURN_MCP_MAXIMUM_PARALLEL_CALLS,
    TURN_MCP_PRIVATE_ROOT_MODE,
    TURN_MCP_PROTOCOL_VERSION,
    TURN_MCP_SERVER_VERSION,
    TURN_MCP_SOCKET_FILENAME,
    TURN_MCP_UNIX_SOCKET_PATH_CAPACITY_BYTES,
)
from eva_agent.pipeline.contracts import (
    JsonValue,
    Stage,
    ToolCall,
    ToolResult,
    ToolRuntimePort,
    ToolTrace,
    freeze_json,
    uuid_text,
)
from eva_agent.pipeline.digests import (
    blake3_bytes,
    blake3_hex,
    canonical_value,
    is_blake3,
)


NATIVE_POLICY_V2_SCHEMA = "eva.codex-native-policy.v2"
NATIVE_POLICY_V2_TIER = "tier-a-file-change-only"
NATIVE_POLICY_V2_STAGES = frozenset({Stage.S3, Stage.S4, Stage.E2E})
NATIVE_POLICY_V2_ACTION_TYPES = ("fileChange",)
REQUIRED_NATIVE_POLICY_CODEX_SDK_VERSION = "0.147.0"
REQUIRED_NATIVE_POLICY_CODEX_CLI_VERSION = "0.147.0"

STAGE_TOOL_GUIDANCE_V1_SCHEMA = "eva.codex-stage-tool-guidance.v1"
STAGE_TOOL_CALL_PROOF_V1_SCHEMA = "eva.codex-stage-tool-call-proof.v1"
STAGE_TOOL_FRONTIER_STATE_V1_SCHEMA = "eva.codex-stage-tool-frontier-state.v1"
_LEGACY_RUNTIME_CONTEXT_SCHEMA = "eva.legacy-candidate-runtime-context.v1"
_LEGACY_PLAN_SCHEMA = "rlevo.med-research-stage-plan-artifact.v1"
_LEGACY_SELECTION_SCHEMA = "rlevo.med-research-evidence-selection.v1"
_LEGACY_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# These are launch-policy commitments, not additions to Codex's or EvaMed's
# configuration schemas.  A deployment/backend must prove that it applied
# the same values before it may claim this policy BLAKE3.
NATIVE_POLICY_V2_CONFIG_OVERRIDES = (
    "project_doc_max_bytes=0",
    'web_search="disabled"',
    "features.browser_use=false",
    "features.code_mode_host=false",
    "features.computer_use=false",
    "features.shell_tool=false",
    "features.standalone_web_search=false",
    "features.unified_exec=false",
    "sandbox_workspace_write.network_access=false",
    'sandbox_workspace_write.writable_roots=["{candidate_root}"]',
    "sandbox_workspace_write.exclude_tmpdir_env_var=true",
    "sandbox_workspace_write.exclude_slash_tmp=true",
)

NATIVE_POLICY_V2_NATIVE_SURFACE = MappingProxyType(
    {
        "fileChange": True,
        "commandExecution": False,
        "dynamicToolCall": False,
        "imageGeneration": False,
        "mcpResourceRead": False,
        "webSearch": False,
    }
)

NATIVE_POLICY_V2_BACKEND_WIRE = MappingProxyType(
    {
        "thread_config_key": "sandbox_workspace_write",
        "thread_sandbox_fields": (
            "network_access",
            "writable_roots",
            "exclude_slash_tmp",
            "exclude_tmpdir_env_var",
        ),
        "turn_sandbox_policy_type": "workspaceWrite",
        "turn_sandbox_policy_fields": (
            "type",
            "networkAccess",
            "writableRoots",
            "excludeSlashTmp",
            "excludeTmpdirEnvVar",
        ),
        "approval_policy": "never",
        "approvals_reviewer": None,
        "approval_request_disposition": "decline-and-fail-turn",
    }
)


class NativePolicyV2Error(ValueError):
    """A Tier-A native policy or one of its public bindings differs."""


def _clean_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise NativePolicyV2Error(f"{label} differs")
    return value


def _stage(value: Stage | str) -> Stage:
    try:
        result = value if isinstance(value, Stage) else Stage(value)
    except (TypeError, ValueError):
        raise NativePolicyV2Error("native policy stage differs") from None
    if result not in NATIVE_POLICY_V2_STAGES:
        raise NativePolicyV2Error("native policy permits only S3, S4, or E2E actors")
    return result


def native_policy_v2_config_overrides(candidate_root: str) -> tuple[str, ...]:
    """Materialize the one candidate-root write grant in launch syntax."""

    root = _clean_text(candidate_root, label="candidate workspace root")
    path = Path(root)
    if not path.is_absolute() or ".." in path.parts or str(path) != root:
        raise NativePolicyV2Error(
            "candidate workspace root must be an absolute normalized path"
        )
    # A JSON string literal avoids config injection by quotes or backslashes in
    # an otherwise valid POSIX path.
    encoded = json.dumps(root, ensure_ascii=False)
    return tuple(
        row.replace('["{candidate_root}"]', f"[{encoded}]")
        for row in NATIVE_POLICY_V2_CONFIG_OVERRIDES
    )


def _public_mapping(value: Mapping[str, Any], *, label: str) -> Mapping[str, JsonValue]:
    try:
        frozen = freeze_json(value)
    except Exception as exc:
        raise NativePolicyV2Error(f"{label} is not JSON-shaped") from exc
    if not isinstance(frozen, Mapping):  # pragma: no cover - typing guard
        raise NativePolicyV2Error(f"{label} must be an object")
    return frozen


def _stable_file_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_regular_file_anchored(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    """Open one path without following a symlink in any path component."""

    if any(
        not hasattr(os, flag)
        for flag in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    ):
        raise NativePolicyV2Error(f"{label} requires no-follow host support")
    components = path.parts
    if (
        not path.is_absolute()
        or path.anchor != "/"
        or len(components) < 2
        or any(component in {"", ".", ".."} for component in components[1:])
    ):
        raise NativePolicyV2Error(f"{label} path is unsafe")
    common_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_fd = -1
    file_fd = -1
    try:
        directory_fd = os.open("/", common_flags | os.O_DIRECTORY)
        for component in components[1:-1]:
            next_fd = os.open(
                component,
                common_flags | os.O_DIRECTORY,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        filename = components[-1]
        before = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise NativePolicyV2Error(f"{label} is not a regular no-follow file")
        file_fd = os.open(filename, common_flags, dir_fd=directory_fd)
        opened = os.fstat(file_fd)
        after = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stable_file_signature(before) != _stable_file_signature(opened)
            or _stable_file_signature(after) != _stable_file_signature(opened)
        ):
            raise NativePolicyV2Error(f"{label} changed while it was opened")
        returned_fd = file_fd
        file_fd = -1
        return returned_fd, opened
    except NativePolicyV2Error:
        raise
    except OSError as exc:
        raise NativePolicyV2Error(f"{label} cannot be reopened without symlinks") from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if directory_fd >= 0:
            os.close(directory_fd)


def _read_regular_file_no_follow(path: Path, *, label: str) -> bytes:
    """Read twice through anchored FDs and prove stable inode and topology."""

    try:
        resolved_before = path.resolve(strict=True)
    except OSError as exc:
        raise NativePolicyV2Error(f"{label} cannot be resolved") from exc
    if resolved_before != path:
        raise NativePolicyV2Error(f"{label} path contains a symlink")
    observations: list[tuple[bytes, os.stat_result]] = []
    for _ in range(2):
        file_fd, opened = _open_regular_file_anchored(path, label=label)
        try:
            chunks: list[bytes] = []
            while True:
                chunk = os.read(file_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after_read = os.fstat(file_fd)
        except OSError as exc:
            raise NativePolicyV2Error(f"{label} cannot be read") from exc
        finally:
            os.close(file_fd)
        if _stable_file_signature(after_read) != _stable_file_signature(opened):
            raise NativePolicyV2Error(f"{label} changed while it was read")
        observations.append((b"".join(chunks), opened))
    try:
        final_metadata = path.lstat()
        resolved_after = path.resolve(strict=True)
    except OSError as exc:
        raise NativePolicyV2Error(f"{label} cannot be reopened") from exc
    first_payload, first_metadata = observations[0]
    second_payload, second_metadata = observations[1]
    if (
        resolved_after != path
        or stat.S_ISLNK(final_metadata.st_mode)
        or not stat.S_ISREG(final_metadata.st_mode)
        or _stable_file_signature(first_metadata)
        != _stable_file_signature(second_metadata)
        or _stable_file_signature(second_metadata)
        != _stable_file_signature(final_metadata)
        or first_payload != second_payload
    ):
        raise NativePolicyV2Error(f"{label} inode or topology changed")
    return first_payload


def _turn_mcp_commitment(metadata: Mapping[str, Any]) -> tuple[Mapping[str, JsonValue], str]:
    frozen = _public_mapping(metadata, label="TurnMCP launch metadata")
    required = {
        "schema",
        "protocol_version",
        "server_version",
        "proxy_python",
        "proxy_python_blake3",
        "proxy_exec_script",
        "proxy_exec_script_blake3",
        "proxy_script",
        "proxy_script_blake3",
        "maximum_parallel_calls",
        "startup_timeout_seconds",
        "tool_timeout_seconds",
        "final_child_environment_names",
        "inherited_parent_environment",
        "environment_values_recorded",
        "turn_nonce_recorded",
        "unix_socket_preflight",
        "launch_blake3",
    }
    launch_blake3 = frozen.get("launch_blake3")
    if set(frozen) != required or not is_blake3(launch_blake3):
        raise NativePolicyV2Error("TurnMCP launch metadata lacks a BLAKE3 commitment")
    core = {key: value for key, value in frozen.items() if key != "launch_blake3"}
    if blake3_hex(core) != launch_blake3:
        raise NativePolicyV2Error("TurnMCP launch metadata BLAKE3 differs")
    if (
        frozen.get("schema") != "eva.turn-mcp-bridge-launch.v1"
        or frozen.get("protocol_version") != TURN_MCP_PROTOCOL_VERSION
        or frozen.get("server_version") != TURN_MCP_SERVER_VERSION
        or type(frozen.get("maximum_parallel_calls")) is not int
        or not 1
        <= frozen["maximum_parallel_calls"]
        <= TURN_MCP_MAXIMUM_PARALLEL_CALLS
        or type(frozen.get("startup_timeout_seconds")) not in {int, float}
        or not 0 < frozen["startup_timeout_seconds"] <= 120
        or type(frozen.get("tool_timeout_seconds")) is not int
        or not 1 <= frozen["tool_timeout_seconds"] <= 86_400
        or frozen.get("final_child_environment_names")
        != (
            "EVA_TURN_MCP_MAXIMUM",
            "EVA_TURN_MCP_NONCE",
            "EVA_TURN_MCP_SOCKET",
            "LANG",
            "LC_ALL",
            "PATH",
            "TZ",
        )
        or frozen.get("inherited_parent_environment") is not False
        or frozen.get("environment_values_recorded") is not False
        or frozen.get("turn_nonce_recorded") is not False
    ):
        raise NativePolicyV2Error("TurnMCP launch metadata capability differs")
    for path_key, digest_key in (
        ("proxy_python", "proxy_python_blake3"),
        ("proxy_exec_script", "proxy_exec_script_blake3"),
        ("proxy_script", "proxy_script_blake3"),
    ):
        path = Path(str(frozen[path_key]))
        if (
            not path.is_absolute()
            or ".." in path.parts
            or str(path) != frozen[path_key]
        ):
            raise NativePolicyV2Error("TurnMCP launch file commitment differs")
        payload = _read_regular_file_no_follow(path, label="TurnMCP launch file")
        if blake3_bytes(payload) != frozen[digest_key]:
            raise NativePolicyV2Error("TurnMCP launch file commitment differs")
    _verify_turn_mcp_socket_preflight(frozen["unix_socket_preflight"])
    return frozen, launch_blake3


def _verify_turn_mcp_socket_preflight(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise NativePolicyV2Error("TurnMCP socket preflight differs")
    required = {
        "schema",
        "temp_root",
        "temp_root_uid",
        "temp_root_mode",
        "temp_root_owned_by_process",
        "temp_root_is_symlink",
        "directory_prefix",
        "random_component_bytes",
        "socket_filename",
        "probed_socket_path_bytes",
        "sockaddr_un_path_capacity_bytes",
        "terminator_bytes",
        "name_length_source",
        "preflight_passed",
    }
    root = Path(str(value.get("temp_root", "")))
    try:
        root_metadata = root.lstat()
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise NativePolicyV2Error("TurnMCP private socket root cannot be reopened") from exc
    random_bytes = value.get("random_component_bytes")
    probed_bytes = value.get("probed_socket_path_bytes")
    if (
        set(value) != required
        or value.get("schema") != "eva.turn-mcp-unix-socket-preflight.v1"
        or not root.is_absolute()
        or ".." in root.parts
        or str(root) != value.get("temp_root")
        or resolved != root
        or stat.S_ISLNK(root_metadata.st_mode)
        or not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != os.getuid()
        or stat.S_IMODE(root_metadata.st_mode) != TURN_MCP_PRIVATE_ROOT_MODE
        or value.get("temp_root_uid") != os.getuid()
        or value.get("temp_root_mode") != TURN_MCP_PRIVATE_ROOT_MODE
        or value.get("temp_root_owned_by_process") is not True
        or value.get("temp_root_is_symlink") is not False
        or value.get("directory_prefix") != TURN_MCP_DIRECTORY_PREFIX
        or type(random_bytes) is not int
        or random_bytes < 1
        or value.get("socket_filename") != TURN_MCP_SOCKET_FILENAME
        or type(probed_bytes) is not int
        or not 1 <= probed_bytes < TURN_MCP_UNIX_SOCKET_PATH_CAPACITY_BYTES
        or value.get("sockaddr_un_path_capacity_bytes")
        != TURN_MCP_UNIX_SOCKET_PATH_CAPACITY_BYTES
        or value.get("terminator_bytes") != 1
        or value.get("name_length_source") != "tempfile.mkdtemp_probe_removed"
        or value.get("preflight_passed") is not True
    ):
        raise NativePolicyV2Error("TurnMCP socket preflight commitment differs")


def _skill_catalog(
    skills: Sequence[CodexSkill],
) -> tuple[tuple[CodexSkill, ...], tuple[Mapping[str, str], ...], str]:
    selected = tuple(skills)
    if any(not isinstance(skill, CodexSkill) for skill in selected):
        raise NativePolicyV2Error("SkillInput catalog contains a non-Codex skill")
    if len({skill.skill_id for skill in selected}) != len(selected):
        raise NativePolicyV2Error("SkillInput IDs must be unique")
    if len({skill.name for skill in selected}) != len(selected):
        raise NativePolicyV2Error("SkillInput names must be unique")
    for skill in selected:
        path = Path(skill.path)
        payload = _read_regular_file_no_follow(path, label="SkillInput file")
        if blake3_bytes(payload) != skill.content_blake3:
            raise NativePolicyV2Error("SkillInput content commitment differs")
    catalog = tuple(skill.catalog_entry() for skill in selected)
    return selected, catalog, blake3_hex(catalog)


def _legacy_sha256(value: Any) -> str:
    """Reopen an immutable EvaMed SHA-256 field without creating a new ID."""

    payload = json.dumps(
        canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _legacy_json_size(value: Any) -> int:
    return len(
        json.dumps(
            canonical_value(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _source_tool_catalog(
    source_tool_catalog: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, JsonValue], ...], str]:
    """Normalize public source entries without changing their input schemas."""

    rows: list[Mapping[str, JsonValue]] = []
    for value in source_tool_catalog:
        try:
            row = canonical_value(value)
        except Exception as exc:
            raise NativePolicyV2Error("source tool catalog is not JSON-shaped") from exc
        if not isinstance(row, Mapping):
            raise NativePolicyV2Error("source tool catalog row must be an object")
        if row.get("type") == "function" and set(row) == {"type", "function"}:
            function = row.get("function")
            if not isinstance(function, Mapping):
                raise NativePolicyV2Error("source tool function row differs")
            if set(function) != {
                "name",
                "description",
                "parameters",
                "x-eva-kind",
                "x-eva-parallel-safe",
            }:
                raise NativePolicyV2Error("source tool function fields differ")
            row = {
                "name": function.get("name"),
                "description": function.get("description"),
                "input_schema": function.get("parameters"),
                "visibility": "actor_public",
                "handler_origin": "signed_legacy_evamed",
                "parallel_safe": function.get("x-eva-parallel-safe"),
            }
            if function.get("x-eva-kind") != "tool":
                raise NativePolicyV2Error("source tool kind differs")
        if set(row) != {
            "name",
            "description",
            "input_schema",
            "visibility",
            "handler_origin",
            "parallel_safe",
        }:
            raise NativePolicyV2Error("source tool catalog fields differ")
        if (
            not isinstance(row.get("name"), str)
            or not row["name"]
            or not isinstance(row.get("description"), str)
            or not row["description"]
            or not isinstance(row.get("input_schema"), Mapping)
            or row.get("visibility") != "actor_public"
            or row.get("handler_origin") != "signed_legacy_evamed"
            or type(row.get("parallel_safe")) is not bool
        ):
            raise NativePolicyV2Error("source tool catalog value differs")
        frozen = freeze_json(row)
        assert isinstance(frozen, Mapping)
        rows.append(frozen)
    selected = tuple(sorted(rows, key=lambda row: str(row["name"])))
    names = tuple(str(row["name"]) for row in selected)
    if not selected or len(names) != len(set(names)):
        raise NativePolicyV2Error("source tool catalog names differ")
    return selected, blake3_hex(selected)


def _legacy_runtime_context(
    public_runtime_context: Mapping[str, Any],
) -> tuple[Mapping[str, JsonValue], Stage]:
    context = _public_mapping(
        public_runtime_context, label="public legacy runtime context"
    )
    required = {
        "schema",
        "source_candidate_id",
        "source_family",
        "focus",
        "target_split",
        "active_episode_id",
        "policy_blake3",
        "tool_catalog_blake3",
        "s1_plan_contract",
        "s2_evidence_contract",
        "execution_stages",
        "s5_terminal_json_schema",
    }
    if set(context) != required or context.get("schema") != _LEGACY_RUNTIME_CONTEXT_SCHEMA:
        raise NativePolicyV2Error("public legacy runtime context fields differ")
    try:
        focus = Stage(context["focus"])
    except (TypeError, ValueError):
        raise NativePolicyV2Error("public legacy runtime focus differs") from None
    if (
        not isinstance(context.get("source_candidate_id"), str)
        or not context["source_candidate_id"]
        or not isinstance(context.get("source_family"), str)
        or not context["source_family"]
        or not isinstance(context.get("active_episode_id"), str)
        or not context["active_episode_id"]
        or not is_blake3(context.get("policy_blake3"))
        or not is_blake3(context.get("tool_catalog_blake3"))
    ):
        raise NativePolicyV2Error("public legacy runtime identity differs")
    s1 = context.get("s1_plan_contract")
    s2 = context.get("s2_evidence_contract")
    if not isinstance(s1, Mapping) or not isinstance(s2, Mapping):
        raise NativePolicyV2Error("public S1/S2 contracts differ")
    s1_keys = {
        "schema",
        "contract_id",
        "sandbox_id",
        "episode_id",
        "focus",
        "policy_sha256",
        "task_contract",
        "host_inputs",
        "stage_artifacts",
        "budgets",
        "max_plan_bytes",
    }
    s2_keys = {
        "schema",
        "contract_id",
        "sandbox_id",
        "episode_id",
        "focus",
        "policy_sha256",
        "s1_plan_contract_sha256",
        "s1_receipt_sha256",
        "question",
        "evidence_need",
        "evidence_objects",
        "required_evidence_ids",
        "required_claim_ids",
        "limits",
        "selection_relative_path",
    }
    if set(s1) != s1_keys or set(s2) != s2_keys:
        raise NativePolicyV2Error("public S1/S2 contract fields differ")
    if (
        s1.get("schema") != "rlevo.med-research-stage-plan-contract.v1"
        or s2.get("schema") != "rlevo.med-research-evidence-service-contract.v1"
        or s1.get("sandbox_id") != context["source_candidate_id"]
        or s2.get("sandbox_id") != s1.get("sandbox_id")
        or s1.get("episode_id") != context["active_episode_id"]
        or s2.get("episode_id") != s1.get("episode_id")
        or s1.get("focus") != focus.value
        or s2.get("focus") != focus.value
        or s2.get("s1_plan_contract_sha256") != _legacy_sha256(s1)
    ):
        raise NativePolicyV2Error("public S1/S2 contract binding differs")
    if any(
        not isinstance(value, str) or _LEGACY_SHA256.fullmatch(value) is None
        for value in (
            s1.get("policy_sha256"),
            s2.get("policy_sha256"),
            s2.get("s1_plan_contract_sha256"),
            s2.get("s1_receipt_sha256"),
        )
    ):
        raise NativePolicyV2Error("public legacy SHA-256 boundary differs")
    return context, focus


def _bound_pipeline(s1: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = s1.get("stage_artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
    }:
        raise NativePolicyV2Error("public S1 stage artifacts differ")
    checks = {
        "S2": [
            "signed_host_receipt",
            "exact_artifact_identity",
            "schema_valid_content",
        ],
        "S3": [
            "signed_host_receipt",
            "zero_exit_code",
            "expected_filesystem_delta",
            "exact_artifact_identity",
            "schema_valid_content",
        ],
        "S4": [
            "signed_host_receipt",
            "zero_exit_code",
            "expected_filesystem_delta",
            "exact_artifact_identity",
            "schema_valid_content",
        ],
        "S5": [
            "signed_host_receipt",
            "independent_host_reopen",
            "exact_artifact_identity",
            "schema_valid_content",
            "single_submission",
        ],
    }
    effects = {
        "S2": "frozen_evidence_selection",
        "S3": "bounded_pilot_execution",
        "S4": "native_materialization",
        "S5": "independent_validation_and_submission",
    }
    return [
        {
            "stage": stage,
            "effect": effects[stage],
            "artifact_or_receipt": artifacts[stage],
            "readiness_checks": checks[stage],
        }
        for stage in ("S2", "S3", "S4", "S5")
    ]


def _bound_budget(s1: Mapping[str, Any], minimum_s2_retrievals: int) -> dict[str, Any]:
    budgets = s1.get("budgets")
    if not isinstance(budgets, Mapping) or set(budgets) != {
        "max_turns",
        "wall_time_seconds",
        "minimum_s5_reserved_turns",
        "max_clean_retries_per_execution_stage",
    }:
        raise NativePolicyV2Error("public S1 budgets differ")
    values = tuple(budgets.values())
    if (
        any(type(value) is not int for value in values)
        or minimum_s2_retrievals < 1
        or budgets["max_clean_retries_per_execution_stage"] != 1
    ):
        raise NativePolicyV2Error("public S1 budget values differ")
    valid: list[dict[str, Any]] = []
    for total in range(5, budgets["max_turns"] + 1):
        reserved = max(budgets["minimum_s5_reserved_turns"], (total + 4) // 5)
        stage_turns = {
            "S2": minimum_s2_retrievals + 1,
            "S3": 2,
            "S4": 2,
            "S5": reserved,
        }
        if sum(stage_turns.values()) <= total:
            valid.append(
                {
                    "total_turns": total,
                    "wall_time_seconds": budgets["wall_time_seconds"],
                    "stage_turns": stage_turns,
                    "s5_reserved_turns": reserved,
                }
            )
    if not valid:
        raise NativePolicyV2Error("public S1 budget cannot complete S2-S5")
    preferred = [row for row in valid if row["total_turns"] <= 10]
    return preferred[-1] if preferred else valid[0]


def _s1_plan_example(
    s1: Mapping[str, Any], *, minimum_s2_retrievals: int
) -> dict[str, Any]:
    task = s1.get("task_contract")
    host_inputs = s1.get("host_inputs")
    if (
        not isinstance(task, Mapping)
        or set(task)
        != {
            "task_ids",
            "final_artifact_relative_path",
            "final_artifact_schema_sha256",
        }
        or not isinstance(host_inputs, (tuple, list))
        or not host_inputs
    ):
        raise NativePolicyV2Error("public S1 task or host inputs differ")
    plan = {
        "schema": _LEGACY_PLAN_SCHEMA,
        "contract_sha256": _legacy_sha256(s1),
        "sandbox_id": s1["sandbox_id"],
        "episode_id": s1["episode_id"],
        "objective": (
            "Complete the bound medical-research objective and materialize the "
            f"required artifact at {task['final_artifact_relative_path']}."
        ),
        "deliverable": {
            "relative_path": task["final_artifact_relative_path"],
            "schema_sha256": task["final_artifact_schema_sha256"],
            "task_ids": canonical_value(task["task_ids"]),
        },
        "host_inputs": canonical_value(host_inputs),
        "pipeline": _bound_pipeline(s1),
        "candidate_method": (
            "Retrieve only bound frozen evidence, run the declared S3 pilot, "
            "materialize the S4 artifact, and submit it through S5 validation."
        ),
        "uncertainties": [
            "Frozen evidence may leave clinically relevant ambiguity to report."
        ],
        "budgets": _bound_budget(s1, minimum_s2_retrievals),
        "stop_rule": {
            "condition": "required_evidence_unavailable",
            "on_trigger": "stop_and_report_typed_constraint",
            "typed_constraint": "required_frozen_evidence_unavailable",
        },
        "recovery": {
            "stage": "S3",
            "trigger": "host_effect_gate_failed",
            "action": "retry_corrected_complete_code",
            "max_clean_retries": 1,
        },
        "claims_unseen_results": False,
    }
    max_bytes = s1.get("max_plan_bytes")
    if type(max_bytes) is not int or _legacy_json_size(plan) > max_bytes:
        raise NativePolicyV2Error("derived S1 plan exceeds its public byte budget")
    return plan


def _s2_materials(s2: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    evidence_objects = s2.get("evidence_objects")
    required_ids = s2.get("required_evidence_ids")
    required_claim_ids = s2.get("required_claim_ids")
    limits = s2.get("limits")
    if (
        not isinstance(evidence_objects, (tuple, list))
        or not isinstance(required_ids, (tuple, list))
        or not isinstance(required_claim_ids, (tuple, list))
        or not isinstance(limits, Mapping)
        or not required_ids
        or not required_claim_ids
        or any(not isinstance(value, str) or not value for value in required_ids)
        or any(
            not isinstance(value, str) or not value for value in required_claim_ids
        )
        or len(set(required_ids)) != len(required_ids)
        or len(set(required_claim_ids)) != len(required_claim_ids)
    ):
        raise NativePolicyV2Error("public S2 identifiers differ")
    expected_limit_keys = {
        "max_retrievals",
        "min_selected",
        "max_selected",
        "max_selection_bytes",
        "max_selection_attempts",
    }
    if (
        set(limits) != expected_limit_keys
        or any(type(value) is not int for value in limits.values())
        or limits["min_selected"] != len(required_ids)
        or limits["max_retrievals"] < len(required_ids)
        or limits["max_selected"] < len(required_ids)
    ):
        raise NativePolicyV2Error("public S2 limits differ")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in evidence_objects:
        if not isinstance(row, Mapping):
            raise NativePolicyV2Error("public S2 evidence row differs")
        evidence_id = row.get("evidence_id")
        statement_ids = row.get("statement_ids")
        if (
            not isinstance(evidence_id, str)
            or not evidence_id
            or evidence_id in by_id
            or not isinstance(row.get("source_id"), str)
            or not row["source_id"]
            or not isinstance(row.get("source_revision"), str)
            or not row["source_revision"]
            or not isinstance(statement_ids, (tuple, list))
            or not statement_ids
            or any(not isinstance(value, str) or not value for value in statement_ids)
            or len(set(statement_ids)) != len(statement_ids)
        ):
            raise NativePolicyV2Error("public S2 evidence row binding differs")
        by_id[evidence_id] = row
    if any(evidence_id not in by_id for evidence_id in required_ids):
        raise NativePolicyV2Error("public S2 required evidence is absent")
    selected: list[dict[str, Any]] = []
    all_statement_ids: list[str] = []
    runtime_bindings: list[dict[str, str]] = [
        {
            "json_pointer": "/contract_sha256",
            "source": "materialize_plan.result.evidence_contract_sha256",
        }
    ]
    for index, evidence_id in enumerate(required_ids):
        evidence = by_id[evidence_id]
        statements = canonical_value(evidence["statement_ids"])
        all_statement_ids.extend(statements)
        selected.append(
            {
                "evidence_id": evidence_id,
                "source_id": evidence["source_id"],
                "source_revision": evidence["source_revision"],
                "statement_ids": statements,
                "retrieval_receipt_sha256": {
                    "$runtime": (
                        "retrieve_frozen_evidence.result."
                        f"{evidence_id}.retrieval_receipt_sha256"
                    )
                },
            }
        )
        runtime_bindings.append(
            {
                "json_pointer": f"/selected_evidence/{index}/retrieval_receipt_sha256",
                "source": (
                    "retrieve_frozen_evidence.result."
                    f"{evidence_id}.retrieval_receipt_sha256"
                ),
            }
        )
    if len(all_statement_ids) != len(set(all_statement_ids)):
        raise NativePolicyV2Error("public S2 required statement IDs overlap")
    blueprint = {
        "schema": _LEGACY_SELECTION_SCHEMA,
        "contract_sha256": {
            "$runtime": "materialize_plan.result.evidence_contract_sha256"
        },
        "sandbox_id": s2["sandbox_id"],
        "episode_id": s2["episode_id"],
        "question": s2["question"],
        "evidence_need": s2["evidence_need"],
        "selected_evidence": selected,
        "inferences": [
            {
                "inference_id": claim_id,
                "text": {
                    "$author": "evidence-derived inference text",
                    "min_length": 1,
                    "max_length": 4096,
                },
                "supporting_statement_ids": list(all_statement_ids),
            }
            for claim_id in required_claim_ids
        ],
        "unresolved_gaps": {
            "$author": "one or more evidence gaps",
            "min_items": 1,
            "max_items": 64,
        },
        "limitations": {
            "$author": "one or more evidence limitations",
            "min_items": 1,
            "max_items": 64,
        },
        "care_directive": False,
    }
    return list(required_ids), {
        "dispatchable": False,
        "argument_blueprint": blueprint,
        "runtime_bindings": runtime_bindings,
        "max_argument_bytes": limits["max_selection_bytes"],
    }


@dataclass(frozen=True, slots=True)
class StageToolGuidanceV1:
    """Digest-locked prompt guidance beside, never inside, immutable tool schemas."""

    source_candidate_id: str
    focus: Stage
    active_episode_id: str
    source_policy_blake3: str
    source_tool_catalog_blake3: str
    public_runtime_context_blake3: str
    frontiers: tuple[Mapping[str, JsonValue], ...]

    def __post_init__(self) -> None:
        if (
            not self.source_candidate_id
            or not self.active_episode_id
            or not isinstance(self.focus, Stage)
            or not is_blake3(self.source_policy_blake3)
            or not is_blake3(self.source_tool_catalog_blake3)
            or not is_blake3(self.public_runtime_context_blake3)
        ):
            raise NativePolicyV2Error("stage tool guidance identity differs")
        frozen = freeze_json(self.frontiers)
        if not isinstance(frozen, tuple) or len(frozen) < 3:
            raise NativePolicyV2Error("stage tool guidance frontiers differ")
        for index, frontier in enumerate(frozen):
            if (
                not isinstance(frontier, Mapping)
                or frontier.get("frontier_index") != index
                or frontier.get("parallel_allowed") is not False
                or frontier.get("single_call_required") is not True
                or frontier.get("must_observe_gate_passed_before_next") is not True
            ):
                raise NativePolicyV2Error("stage tool guidance frontier differs")
        object.__setattr__(self, "frontiers", frozen)

    def core_document(self) -> dict[str, Any]:
        return {
            "schema": STAGE_TOOL_GUIDANCE_V1_SCHEMA,
            "source_candidate_id": self.source_candidate_id,
            "focus": self.focus.value,
            "active_episode_id": self.active_episode_id,
            "source_policy_blake3": self.source_policy_blake3,
            "source_tool_catalog_blake3": self.source_tool_catalog_blake3,
            "public_runtime_context_blake3": self.public_runtime_context_blake3,
            # Later-stage actors consume exact predecessors that the host
            # replays before the provider turn.  Keep those frontiers in the
            # signed guidance document as the derivation/provenance boundary,
            # while identifying only the first actor-owned frontier.  The S1
            # and S2 branches deliberately retain their byte representation.
            "first_frontier_index": (
                1
                if self.focus is Stage.S2
                else len(self.frontiers)
                if self.focus is Stage.S3
                else 0
            ),
            "frontiers": canonical_value(self.frontiers),
            "execution_rules": {
                "empty_first_call_allowed": False,
                "s1_s2_parallel_allowed": False,
                "advance_only_after_gate_passed": True,
                "one_shot_s1_consumption_protected_by_preflight": True,
            },
            "native_coding_boundary": {
                "stages": ["S3", "S4", "E2E"],
                "actor_native_action_types": ["fileChange"],
                "native_actions_parallel_allowed": False,
                "command_execution_allowed": False,
                "judge_native_actions_allowed": False,
            },
            "schema_compatibility": {
                "source_tool_schema_changed": False,
                "mcp_wire_schema_changed": False,
                "evamed_schema_changed": False,
                "rubric_schema_changed": False,
                "sandbox_schema_changed": False,
            },
        }

    @property
    def guidance_blake3(self) -> str:
        return blake3_hex(self.core_document())

    @property
    def prompt_text(self) -> str:
        first = canonical_value(self.frontiers[0])
        retrievals = [
            canonical_value(row)
            for row in self.frontiers
            if row.get("tool_name") == "retrieve_frozen_evidence"
        ]
        selection = canonical_value(self.frontiers[-1])
        if self.focus is Stage.S2:
            return "\n".join(
                (
                    "EVA deterministic stage-tool guidance.",
                    f"guidance_blake3={self.guidance_blake3}",
                    (
                        "This is a prompt sidecar; every signed source tool and MCP "
                        "wire schema remains byte-for-byte unchanged."
                    ),
                    (
                        "The host has already executed and independently checked the "
                        "exact S1 predecessor frontier below before this actor turn. "
                        "Do not call materialize_plan and do not reconstruct S1."
                    ),
                    "HOST_HYDRATED_S1_FRONTIER="
                    + json.dumps(
                        first,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    (
                        "S2 is a strict sequential frontier. Start with each listed "
                        "retrieve_frozen_evidence call in order, observe "
                        "gate_passed=true after each call, then issue exactly one "
                        "materialize_evidence_selection call. Never parallelize or "
                        "skip these S2 frontiers."
                    ),
                    "S2_RETRIEVAL_FRONTIERS="
                    + json.dumps(
                        retrievals,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "S2_SELECTION_BLUEPRINT="
                    + json.dumps(
                        selection,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    (
                        "For the S2 selection, replace each $runtime value only with "
                        "the exact preceding signed result and each $author value "
                        "with evidence-grounded text. Never submit the blueprint "
                        "markers themselves and never submit {}."
                    ),
                    (
                        "Stop this stage only after work/evidence-selection.json is "
                        "materialized and the host observation reports "
                        "gate_passed=true with no failed checks."
                    ),
                )
            )
        if self.focus is Stage.S3:
            return "\n".join(
                (
                    "EVA deterministic stage-tool guidance.",
                    f"guidance_blake3={self.guidance_blake3}",
                    (
                        "This is a prompt sidecar; every signed source tool and MCP "
                        "wire schema remains byte-for-byte unchanged."
                    ),
                    (
                        "The host has already executed and independently reopened "
                        "the exact S1 plan plus every required S2 frozen-evidence "
                        "retrieval and the S2 evidence selection in this same "
                        "isolated workspace before this actor turn."
                    ),
                    (
                        "Continue at S3 only. Do not call materialize_plan, "
                        "retrieve_frozen_evidence, or "
                        "materialize_evidence_selection, and do not reconstruct "
                        "S1 or S2."
                    ),
                    (
                        "Use the host-hydrated work/stage-plan.json and "
                        "work/evidence-selection.json as the authoritative "
                        "prerequisites for the bound S3 pilot."
                    ),
                    (
                        "At S3, Codex-native execution is fileChange-only and "
                        "sequential: no commandExecution, shell, network, tmp, "
                        "resume, extra writable root, or judge mutation."
                    ),
                    (
                        "Stop this stage only after the S3 host observation reports "
                        "gate_passed=true with no failed checks."
                    ),
                )
            )
        return "\n".join(
            (
                "EVA deterministic stage-tool guidance.",
                f"guidance_blake3={self.guidance_blake3}",
                (
                    "This is a prompt sidecar; every signed source tool and MCP "
                    "wire schema remains byte-for-byte unchanged."
                ),
                (
                    "S1 and S2 are a strict sequential frontier. Issue exactly one "
                    "call, observe gate_passed=true, then advance. Never parallelize "
                    "S1 or S2."
                ),
                (
                    "S1 FIRST: call materialize_plan before every other tool. Do not "
                    "probe tools or schemas with empty, partial, placeholder, or "
                    "invented arguments. Copy the complete FIRST_FRONTIER shape."
                ),
                (
                    "The first schema-valid materialize_plan call is one-shot; a "
                    "schema-invalid protocol call is rejected before that semantic "
                    "attempt is consumed and returns an exact JSON field diagnostic. "
                    "Never call it with {}. Use one complete object with every field "
                    "shown below; only "
                    "objective, candidate_method, and uncertainties are authored prose."
                ),
                "FIRST_FRONTIER="
                + json.dumps(
                    first,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "S2_RETRIEVAL_FRONTIERS="
                + json.dumps(
                    retrievals,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "S2_SELECTION_BLUEPRINT="
                + json.dumps(
                    selection,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                (
                    "For the S2 selection, replace each $runtime value only with the "
                    "exact preceding signed result and each $author value with "
                    "evidence-grounded text. Never submit the blueprint markers "
                    "themselves and never submit {}."
                ),
                (
                    "At S3/S4, Codex-native execution is fileChange-only and "
                    "sequential: no commandExecution, shell, network, tmp, resume, "
                    "extra writable root, or judge mutation."
                ),
            )
        )

    @property
    def prompt_blake3(self) -> str:
        return blake3_bytes(self.prompt_text.encode("utf-8"))

    def to_document(self) -> dict[str, Any]:
        return {
            **self.core_document(),
            "guidance_blake3": self.guidance_blake3,
            "prompt_blake3": self.prompt_blake3,
        }


def build_stage_tool_guidance_v1(
    *,
    public_runtime_context: Mapping[str, Any],
    source_tool_catalog: Sequence[Mapping[str, Any]],
) -> StageToolGuidanceV1:
    """Derive non-empty S1 and sequential S2 examples from public bytes only."""

    context, focus = _legacy_runtime_context(public_runtime_context)
    catalog, catalog_blake3 = _source_tool_catalog(source_tool_catalog)
    if catalog_blake3 != context["tool_catalog_blake3"]:
        raise NativePolicyV2Error("source tool catalog BLAKE3 differs from context")
    catalog_names = {str(row["name"]) for row in catalog}
    required_tools = {
        "materialize_plan",
        "retrieve_frozen_evidence",
        "materialize_evidence_selection",
    }
    if not required_tools.issubset(catalog_names):
        raise NativePolicyV2Error("public S1/S2 tools are unavailable")
    s1 = context["s1_plan_contract"]
    s2 = context["s2_evidence_contract"]
    assert isinstance(s1, Mapping) and isinstance(s2, Mapping)
    required_evidence_ids, selection = _s2_materials(s2)
    plan = _s1_plan_example(s1, minimum_s2_retrievals=len(required_evidence_ids))
    frontiers: list[dict[str, Any]] = [
        {
            "frontier_index": 0,
            "stage": "S1",
            "tool_name": "materialize_plan",
            "arguments_kind": "complete-dispatchable-example",
            "arguments": plan,
            "authored_fields": ["objective", "candidate_method", "uncertainties"],
            "max_argument_bytes": s1["max_plan_bytes"],
            "parallel_allowed": False,
            "single_call_required": True,
            "must_observe_gate_passed_before_next": True,
        }
    ]
    for evidence_id in required_evidence_ids:
        frontiers.append(
            {
                "frontier_index": len(frontiers),
                "stage": "S2",
                "tool_name": "retrieve_frozen_evidence",
                "arguments_kind": "exact-dispatchable-example",
                "arguments": {"evidence_id": evidence_id},
                "parallel_allowed": False,
                "single_call_required": True,
                "must_observe_gate_passed_before_next": True,
            }
        )
    frontiers.append(
        {
            "frontier_index": len(frontiers),
            "stage": "S2",
            "tool_name": "materialize_evidence_selection",
            "arguments_kind": "runtime-bound-blueprint",
            **selection,
            "parallel_allowed": False,
            "single_call_required": True,
            "must_observe_gate_passed_before_next": True,
        }
    )
    guidance = StageToolGuidanceV1(
        source_candidate_id=str(context["source_candidate_id"]),
        focus=focus,
        active_episode_id=str(context["active_episode_id"]),
        source_policy_blake3=str(context["policy_blake3"]),
        source_tool_catalog_blake3=catalog_blake3,
        public_runtime_context_blake3=blake3_hex(context),
        frontiers=tuple(frontiers),
    )
    if not is_blake3(guidance.guidance_blake3) or not is_blake3(
        guidance.prompt_blake3
    ):
        raise NativePolicyV2Error("stage tool guidance digest differs")
    return guidance


def verify_stage_tool_guidance_v1(
    guidance: StageToolGuidanceV1,
    *,
    public_runtime_context: Mapping[str, Any],
    source_tool_catalog: Sequence[Mapping[str, Any]],
) -> str:
    if not isinstance(guidance, StageToolGuidanceV1):
        raise NativePolicyV2Error("value is not stage tool guidance v1")
    expected = build_stage_tool_guidance_v1(
        public_runtime_context=public_runtime_context,
        source_tool_catalog=source_tool_catalog,
    )
    if guidance.to_document() != expected.to_document() or (
        guidance.prompt_text != expected.prompt_text
    ):
        raise NativePolicyV2Error("stage tool guidance derivation differs")
    return guidance.guidance_blake3


def verify_stage_tool_guidance_document_v1(
    document: Mapping[str, Any],
    *,
    public_runtime_context: Mapping[str, Any],
    source_tool_catalog: Sequence[Mapping[str, Any]],
) -> str:
    expected = build_stage_tool_guidance_v1(
        public_runtime_context=public_runtime_context,
        source_tool_catalog=source_tool_catalog,
    )
    if not isinstance(document, Mapping) or canonical_value(document) != canonical_value(
        expected.to_document()
    ):
        raise NativePolicyV2Error("serialized stage tool guidance differs")
    return expected.guidance_blake3


def _authored_text(value: Any, *, label: str, minimum: int, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not minimum <= len(value) <= maximum
        or not value.strip()
        or "\x00" in value
    ):
        raise NativePolicyV2Error(f"{label} differs")
    return value


def _authored_texts(
    value: Any,
    *,
    label: str,
    minimum_items: int,
    maximum_items: int,
    minimum_length: int = 8,
    maximum_length: int = 2048,
) -> list[str]:
    if not isinstance(value, (tuple, list)) or not (
        minimum_items <= len(value) <= maximum_items
    ):
        raise NativePolicyV2Error(f"{label} differs")
    rows = [
        _authored_text(
            row,
            label=label,
            minimum=minimum_length,
            maximum=maximum_length,
        )
        for row in value
    ]
    if len(rows) != len(set(rows)):
        raise NativePolicyV2Error(f"{label} must be unique")
    return rows


def _verify_s1_arguments(
    frontier: Mapping[str, Any], arguments: Mapping[str, Any]
) -> None:
    expected = frontier["arguments"]
    if not isinstance(expected, Mapping) or set(arguments) != set(expected):
        raise NativePolicyV2Error("guided S1 argument fields differ")
    authored = {"objective", "candidate_method", "uncertainties"}
    if any(
        canonical_value(arguments[key]) != canonical_value(expected[key])
        for key in expected
        if key not in authored
    ):
        raise NativePolicyV2Error("guided S1 public contract binding differs")
    _authored_text(
        arguments["objective"], label="guided S1 objective", minimum=16, maximum=4096
    )
    _authored_text(
        arguments["candidate_method"],
        label="guided S1 candidate method",
        minimum=16,
        maximum=4096,
    )
    _authored_texts(
        arguments["uncertainties"],
        label="guided S1 uncertainties",
        minimum_items=1,
        maximum_items=32,
    )
    if _legacy_json_size(arguments) > frontier["max_argument_bytes"]:
        raise NativePolicyV2Error("guided S1 arguments exceed public byte budget")


def _selection_frontier(guidance: StageToolGuidanceV1) -> Mapping[str, JsonValue]:
    rows = tuple(
        row
        for row in guidance.frontiers
        if row.get("tool_name") == "materialize_evidence_selection"
    )
    if len(rows) != 1:
        raise NativePolicyV2Error("guided S2 selection frontier differs")
    return rows[0]


@dataclass(frozen=True, slots=True)
class StageToolFrontierStateV1:
    """Immutable gate chain for the one-shot S1/S2 execution frontier."""

    guidance_blake3: str
    next_frontier_index: int
    accepted_call_proof_blake3s: tuple[str, ...] = ()
    tool_result_receipt_blake3s: tuple[str, ...] = ()
    evidence_contract_sha256: str | None = None
    retrieval_receipt_sha256_by_evidence_id: Mapping[str, str] = MappingProxyType(
        {}
    )

    def __post_init__(self) -> None:
        if (
            not is_blake3(self.guidance_blake3)
            or type(self.next_frontier_index) is not int
            or self.next_frontier_index < 0
        ):
            raise NativePolicyV2Error("guided frontier state identity differs")
        call_proofs = tuple(self.accepted_call_proof_blake3s)
        results = tuple(self.tool_result_receipt_blake3s)
        if (
            len(call_proofs) != self.next_frontier_index
            or len(results) != self.next_frontier_index
            or any(not is_blake3(value) for value in call_proofs + results)
        ):
            raise NativePolicyV2Error("guided frontier state history differs")
        if self.evidence_contract_sha256 is not None and (
            not isinstance(self.evidence_contract_sha256, str)
            or _LEGACY_SHA256.fullmatch(self.evidence_contract_sha256) is None
        ):
            raise NativePolicyV2Error("guided frontier evidence contract differs")
        receipts = freeze_json(self.retrieval_receipt_sha256_by_evidence_id)
        if not isinstance(receipts, Mapping) or any(
            not isinstance(evidence_id, str)
            or not evidence_id
            or not isinstance(digest, str)
            or _LEGACY_SHA256.fullmatch(digest) is None
            for evidence_id, digest in receipts.items()
        ):
            raise NativePolicyV2Error("guided frontier retrieval receipts differ")
        object.__setattr__(self, "accepted_call_proof_blake3s", call_proofs)
        object.__setattr__(self, "tool_result_receipt_blake3s", results)
        object.__setattr__(self, "retrieval_receipt_sha256_by_evidence_id", receipts)

    def core_document(self) -> dict[str, Any]:
        return {
            "schema": STAGE_TOOL_FRONTIER_STATE_V1_SCHEMA,
            "guidance_blake3": self.guidance_blake3,
            "next_frontier_index": self.next_frontier_index,
            "accepted_call_proof_blake3s": list(
                self.accepted_call_proof_blake3s
            ),
            "tool_result_receipt_blake3s": list(
                self.tool_result_receipt_blake3s
            ),
            "evidence_contract_sha256": self.evidence_contract_sha256,
            "retrieval_receipt_sha256_by_evidence_id": canonical_value(
                self.retrieval_receipt_sha256_by_evidence_id
            ),
        }

    @property
    def state_blake3(self) -> str:
        return blake3_hex(self.core_document())


def start_guided_stage_frontier_v1(
    guidance: StageToolGuidanceV1,
) -> StageToolFrontierStateV1:
    """Create the only valid pre-S1 frontier state."""

    if not isinstance(guidance, StageToolGuidanceV1):
        raise NativePolicyV2Error("value is not stage tool guidance v1")
    return StageToolFrontierStateV1(
        guidance_blake3=guidance.guidance_blake3,
        next_frontier_index=0,
    )


def verify_guided_stage_frontier_state_v1(
    guidance: StageToolGuidanceV1,
    state: StageToolFrontierStateV1,
) -> str:
    """Reopen a frontier state and prove that no S1/S2 gate was skipped."""

    if (
        not isinstance(guidance, StageToolGuidanceV1)
        or not isinstance(state, StageToolFrontierStateV1)
        or state.guidance_blake3 != guidance.guidance_blake3
        or state.next_frontier_index > len(guidance.frontiers)
    ):
        raise NativePolicyV2Error("guided frontier state binding differs")
    completed = guidance.frontiers[: state.next_frontier_index]
    expected_evidence_ids = tuple(
        str(row["arguments"]["evidence_id"])
        for row in completed
        if row.get("tool_name") == "retrieve_frozen_evidence"
    )
    if (
        (state.next_frontier_index == 0)
        != (state.evidence_contract_sha256 is None)
        or tuple(state.retrieval_receipt_sha256_by_evidence_id)
        != tuple(sorted(expected_evidence_ids))
    ):
        raise NativePolicyV2Error("guided frontier prerequisite history differs")
    return state.state_blake3


def _verify_s2_selection_arguments(
    frontier: Mapping[str, Any],
    arguments: Mapping[str, Any],
    *,
    state: StageToolFrontierStateV1,
) -> None:
    blueprint = frontier.get("argument_blueprint")
    if not isinstance(blueprint, Mapping) or set(arguments) != set(blueprint):
        raise NativePolicyV2Error("guided S2 selection fields differ")
    for key in (
        "schema",
        "sandbox_id",
        "episode_id",
        "question",
        "evidence_need",
        "care_directive",
    ):
        if canonical_value(arguments[key]) != canonical_value(blueprint[key]):
            raise NativePolicyV2Error("guided S2 public contract binding differs")
    contract_sha256 = arguments.get("contract_sha256")
    if not isinstance(contract_sha256, str) or _LEGACY_SHA256.fullmatch(
        contract_sha256
    ) is None:
        raise NativePolicyV2Error("guided S2 dynamic contract SHA-256 differs")
    if contract_sha256 != state.evidence_contract_sha256:
        raise NativePolicyV2Error("guided S2 contract is not bound to the S1 result")
    selected = arguments.get("selected_evidence")
    selected_blueprint = blueprint.get("selected_evidence")
    if not isinstance(selected, (tuple, list)) or not isinstance(
        selected_blueprint, (tuple, list)
    ) or len(selected) != len(selected_blueprint):
        raise NativePolicyV2Error("guided S2 selected evidence differs")
    static_selected = {"evidence_id", "source_id", "source_revision", "statement_ids"}
    for row, expected in zip(selected, selected_blueprint, strict=True):
        if (
            not isinstance(row, Mapping)
            or not isinstance(expected, Mapping)
            or set(row) != static_selected | {"retrieval_receipt_sha256"}
            or any(
                canonical_value(row[key]) != canonical_value(expected[key])
                for key in static_selected
            )
            or not isinstance(row.get("retrieval_receipt_sha256"), str)
            or _LEGACY_SHA256.fullmatch(row["retrieval_receipt_sha256"]) is None
            or row["retrieval_receipt_sha256"]
            != state.retrieval_receipt_sha256_by_evidence_id.get(
                str(row.get("evidence_id"))
            )
        ):
            raise NativePolicyV2Error("guided S2 selected evidence binding differs")
    inferences = arguments.get("inferences")
    inference_blueprint = blueprint.get("inferences")
    if not isinstance(inferences, (tuple, list)) or not isinstance(
        inference_blueprint, (tuple, list)
    ) or len(inferences) != len(inference_blueprint):
        raise NativePolicyV2Error("guided S2 inferences differ")
    for row, expected in zip(inferences, inference_blueprint, strict=True):
        if (
            not isinstance(row, Mapping)
            or not isinstance(expected, Mapping)
            or set(row) != {"inference_id", "text", "supporting_statement_ids"}
            or row.get("inference_id") != expected.get("inference_id")
            or canonical_value(row.get("supporting_statement_ids"))
            != canonical_value(expected.get("supporting_statement_ids"))
        ):
            raise NativePolicyV2Error("guided S2 inference binding differs")
        _authored_text(
            row.get("text"),
            label="guided S2 inference text",
            minimum=1,
            maximum=4096,
        )
    _authored_texts(
        arguments.get("unresolved_gaps"),
        label="guided S2 unresolved gaps",
        minimum_items=1,
        maximum_items=64,
    )
    _authored_texts(
        arguments.get("limitations"),
        label="guided S2 limitations",
        minimum_items=1,
        maximum_items=64,
    )
    if _legacy_json_size(arguments) > frontier["max_argument_bytes"]:
        raise NativePolicyV2Error("guided S2 arguments exceed public byte budget")


def materialize_s2_selection_arguments_v1(
    guidance: StageToolGuidanceV1,
    *,
    state: StageToolFrontierStateV1,
    inference_text_by_claim_id: Mapping[str, str],
    unresolved_gaps: Sequence[str],
    limitations: Sequence[str],
) -> Mapping[str, JsonValue]:
    """Fill only runtime/authored holes in the public S2 selection blueprint."""

    frontier = _selection_frontier(guidance)
    verify_guided_stage_frontier_state_v1(guidance, state)
    if (
        state.next_frontier_index >= len(guidance.frontiers)
        or guidance.frontiers[state.next_frontier_index] is not frontier
    ):
        raise NativePolicyV2Error("guided S2 selection is not the next frontier")
    blueprint = canonical_value(frontier["argument_blueprint"])
    selected = blueprint["selected_evidence"]
    inferences = blueprint["inferences"]
    evidence_ids = tuple(row["evidence_id"] for row in selected)
    claim_ids = tuple(row["inference_id"] for row in inferences)
    if set(state.retrieval_receipt_sha256_by_evidence_id) != set(evidence_ids):
        raise NativePolicyV2Error("guided S2 retrieval receipt history differs")
    if set(inference_text_by_claim_id) != set(claim_ids):
        raise NativePolicyV2Error("guided S2 inference text IDs differ")
    arguments = {
        **blueprint,
        "contract_sha256": state.evidence_contract_sha256,
        "selected_evidence": [
            {
                **row,
                "retrieval_receipt_sha256": (
                    state.retrieval_receipt_sha256_by_evidence_id[
                        row["evidence_id"]
                    ]
                ),
            }
            for row in selected
        ],
        "inferences": [
            {**row, "text": inference_text_by_claim_id[row["inference_id"]]}
            for row in inferences
        ],
        "unresolved_gaps": list(unresolved_gaps),
        "limitations": list(limitations),
    }
    _verify_s2_selection_arguments(frontier, arguments, state=state)
    frozen = freeze_json(arguments)
    assert isinstance(frozen, Mapping)
    return frozen


def verify_guided_stage_call_v1(
    guidance: StageToolGuidanceV1,
    *,
    state: StageToolFrontierStateV1,
    frontier_index: int,
    tool_call_id: str,
    tool_name: str,
    arguments: Mapping[str, Any],
    concurrent_call_count: int = 1,
) -> Mapping[str, JsonValue]:
    """Reject empty/parallel S1-S2 calls before a one-shot host effect is spent."""

    verify_guided_stage_frontier_state_v1(guidance, state)
    try:
        uuid_text(tool_call_id, label="guided stage tool_call_id")
    except Exception as exc:
        raise NativePolicyV2Error("guided stage tool call identity differs") from exc
    if (
        not isinstance(guidance, StageToolGuidanceV1)
        or type(frontier_index) is not int
        or not 0 <= frontier_index < len(guidance.frontiers)
        or frontier_index != state.next_frontier_index
        or concurrent_call_count != 1
        or type(concurrent_call_count) is not int
    ):
        raise NativePolicyV2Error("guided stage call frontier differs")
    if not isinstance(arguments, Mapping) or not arguments:
        raise NativePolicyV2Error("guided stage call arguments must not be empty")
    frontier = guidance.frontiers[frontier_index]
    if (
        frontier.get("tool_name") != tool_name
        or frontier.get("parallel_allowed") is not False
        or frontier.get("single_call_required") is not True
    ):
        raise NativePolicyV2Error("guided stage call tool or parallel frontier differs")
    kind = frontier.get("arguments_kind")
    if kind == "complete-dispatchable-example":
        _verify_s1_arguments(frontier, arguments)
    elif kind == "exact-dispatchable-example":
        if canonical_value(arguments) != canonical_value(frontier.get("arguments")):
            raise NativePolicyV2Error("guided S2 retrieval arguments differ")
    elif kind == "runtime-bound-blueprint":
        _verify_s2_selection_arguments(frontier, arguments, state=state)
    else:  # pragma: no cover - constructor makes this unreachable
        raise NativePolicyV2Error("guided stage call argument kind differs")
    proof_core = {
        "schema": STAGE_TOOL_CALL_PROOF_V1_SCHEMA,
        "guidance_blake3": guidance.guidance_blake3,
        "frontier_state_blake3": state.state_blake3,
        "frontier_index": frontier_index,
        "tool_call_id": tool_call_id,
        "stage": frontier["stage"],
        "tool_name": tool_name,
        "arguments_blake3": blake3_hex(arguments),
        "concurrent_call_count": concurrent_call_count,
        "parallel_frontier": False,
    }
    proof = freeze_json({**proof_core, "proof_blake3": blake3_hex(proof_core)})
    assert isinstance(proof, Mapping)
    return proof


def advance_guided_stage_frontier_v1(
    guidance: StageToolGuidanceV1,
    *,
    state: StageToolFrontierStateV1,
    frontier_index: int,
    tool_call_id: str,
    tool_name: str,
    arguments: Mapping[str, Any],
    call_proof: Mapping[str, Any],
    tool_result: ToolResult,
) -> StageToolFrontierStateV1:
    """Advance exactly once after reopening a successful signed host result."""

    expected_proof = verify_guided_stage_call_v1(
        guidance,
        state=state,
        frontier_index=frontier_index,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        arguments=arguments,
    )
    if canonical_value(call_proof) != canonical_value(expected_proof):
        raise NativePolicyV2Error("guided stage call proof differs")
    if not isinstance(tool_result, ToolResult):
        raise NativePolicyV2Error("guided stage ToolResult receipt differs")
    try:
        uuid_text(tool_result.result_id, label="guided stage result_id")
        uuid_text(tool_result.call_id, label="guided stage result call_id")
        uuid_text(
            tool_result.parallel_group_id,
            label="guided stage parallel_group_id",
        )
    except Exception as exc:
        raise NativePolicyV2Error("guided stage ToolResult identity differs") from exc
    result_core = {
        "result_id": tool_result.result_id,
        "call_id": tool_result.call_id,
        "name": tool_result.name,
        "frontier": tool_result.frontier,
        "parallel_group_id": tool_result.parallel_group_id,
        "status": tool_result.status,
        "output": tool_result.output,
        "error_code": tool_result.error_code,
        "workspace_before_blake3": tool_result.workspace_before_blake3,
        "workspace_after_blake3": tool_result.workspace_after_blake3,
    }
    if (
        tool_result.call_id != tool_call_id
        or tool_result.name != tool_name
        or tool_result.status != "completed"
        or tool_result.error_code is not None
        or type(tool_result.frontier) is not int
        or tool_result.frontier != frontier_index
        or not is_blake3(tool_result.workspace_before_blake3)
        or not is_blake3(tool_result.workspace_after_blake3)
        or not is_blake3(tool_result.receipt_blake3)
        or tool_result.receipt_blake3 != blake3_hex(result_core)
        or not isinstance(tool_result.output, Mapping)
    ):
        raise NativePolicyV2Error("guided stage ToolResult receipt differs")
    result = tool_result.output
    frontier = guidance.frontiers[frontier_index]
    if (
        not isinstance(result, Mapping)
        or result.get("gate_passed") is not True
        or result.get("stage") != frontier["stage"]
        or "error" in result
    ):
        raise NativePolicyV2Error("guided stage result did not pass its gate")
    evidence_contract_sha256 = state.evidence_contract_sha256
    retrieval_receipts = canonical_value(
        state.retrieval_receipt_sha256_by_evidence_id
    )
    if tool_name == "materialize_plan":
        evidence_contract_sha256 = result.get("evidence_contract_sha256")
        if (
            not isinstance(evidence_contract_sha256, str)
            or _LEGACY_SHA256.fullmatch(evidence_contract_sha256) is None
            or result.get("next_stage") != "S2"
        ):
            raise NativePolicyV2Error("guided S1 gate result binding differs")
    elif tool_name == "retrieve_frozen_evidence":
        evidence_id = frontier["arguments"]["evidence_id"]
        receipt_sha256 = result.get("retrieval_receipt_sha256")
        if (
            result.get("evidence_id") != evidence_id
            or not isinstance(receipt_sha256, str)
            or _LEGACY_SHA256.fullmatch(receipt_sha256) is None
        ):
            raise NativePolicyV2Error("guided S2 retrieval result binding differs")
        retrieval_receipts[evidence_id] = receipt_sha256
    elif tool_name == "materialize_evidence_selection" and result.get(
        "next_stage"
    ) != "S3":
        raise NativePolicyV2Error("guided S2 selection gate result binding differs")
    advanced = StageToolFrontierStateV1(
        guidance_blake3=guidance.guidance_blake3,
        next_frontier_index=state.next_frontier_index + 1,
        accepted_call_proof_blake3s=(
            *state.accepted_call_proof_blake3s,
            str(expected_proof["proof_blake3"]),
        ),
        tool_result_receipt_blake3s=(
            *state.tool_result_receipt_blake3s,
            tool_result.receipt_blake3,
        ),
        evidence_contract_sha256=evidence_contract_sha256,
        retrieval_receipt_sha256_by_evidence_id=retrieval_receipts,
    )
    verify_guided_stage_frontier_state_v1(guidance, advanced)
    return advanced


class GuidedStageToolRuntimeV1:
    """Host-owned pre-effect guard for one exact E2E S1/S2 frontier.

    The model never supplies or restores the private state. Calls are checked
    before the wrapped runtime can consume an S1/S2 attempt, then advanced only
    from the exact ``ToolResult`` returned by that same runtime invocation.
    """

    def __init__(
        self,
        tools: ToolRuntimePort,
        guidance: StageToolGuidanceV1,
    ) -> None:
        if not callable(getattr(tools, "execute", None)) or not callable(
            getattr(tools, "trace", None)
        ):
            raise NativePolicyV2Error("guided tool runtime port differs")
        if not isinstance(guidance, StageToolGuidanceV1):
            raise NativePolicyV2Error("value is not stage tool guidance v1")
        self._tools = tools
        self._guidance = guidance
        self._state = start_guided_stage_frontier_v1(guidance)
        self._terminal_failure = False
        self._lock = Lock()

    @property
    def workspace_root(self) -> Any:
        return getattr(self._tools, "workspace_root", None)

    @property
    def workspace(self) -> Any:
        return getattr(self._tools, "workspace", None)

    @property
    def _workspace(self) -> Any:
        return getattr(self._tools, "_workspace", None)

    @property
    def frontier_state_document(self) -> Mapping[str, JsonValue]:
        document = freeze_json(
            {
                **self._state.core_document(),
                "state_blake3": self._state.state_blake3,
            }
        )
        assert isinstance(document, Mapping)
        return document

    def execute(self, calls: Sequence[ToolCall]) -> tuple[ToolResult, ...]:
        values = tuple(calls)
        with self._lock:
            if self._terminal_failure:
                raise NativePolicyV2Error(
                    "guided S1/S2 runtime is permanently failed after a consumed effect"
                )
            state = self._state
            guided_names = {
                str(row["tool_name"]) for row in self._guidance.frontiers
            }
            if state.next_frontier_index < len(self._guidance.frontiers):
                if len(values) != 1:
                    raise NativePolicyV2Error(
                        "guided S1/S2 runtime requires one sequential call"
                    )
                call = values[0]
                if not isinstance(call, ToolCall):
                    raise NativePolicyV2Error("guided S1/S2 ToolCall differs")
                frontier_index = state.next_frontier_index
                proof = verify_guided_stage_call_v1(
                    self._guidance,
                    state=state,
                    frontier_index=frontier_index,
                    tool_call_id=call.call_id,
                    tool_name=call.name,
                    arguments=call.arguments,
                )
                # Latch before crossing the effect boundary. Only an exact,
                # receipt-bound advancement unlocks the next frontier. Any
                # exception, failed gate, malformed result, or cancellation
                # therefore makes this runtime permanently non-retryable.
                self._terminal_failure = True
                results = tuple(self._tools.execute(values))
                if len(results) != 1:
                    raise NativePolicyV2Error(
                        "guided S1/S2 runtime returned a non-unit result group"
                    )
                advanced = advance_guided_stage_frontier_v1(
                    self._guidance,
                    state=state,
                    frontier_index=frontier_index,
                    tool_call_id=call.call_id,
                    tool_name=call.name,
                    arguments=call.arguments,
                    call_proof=proof,
                    tool_result=results[0],
                )
                self._state = advanced
                self._terminal_failure = False
                return results
            if any(call.name in guided_names for call in values):
                raise NativePolicyV2Error(
                    "guided S1/S2 runtime forbids consumed-frontier reuse"
                )
            return tuple(self._tools.execute(values))

    def trace(self) -> ToolTrace:
        return self._tools.trace()


def guard_guided_stage_tool_runtime_v1(
    tools: ToolRuntimePort,
    guidance: StageToolGuidanceV1,
) -> GuidedStageToolRuntimeV1:
    """Construct the host-owned runtime guard used before TurnMCP dispatch."""

    return GuidedStageToolRuntimeV1(tools, guidance)


@dataclass(frozen=True, slots=True)
class NativePolicyV2:
    """One deterministic, candidate-turn-independent Tier-A policy.

    Candidate and workspace identity are deliberately deployment bindings, not
    policy permissions.  Consequently identical runtime/skill/MCP inputs for
    the same stage produce the same policy BLAKE3 across processes.
    """

    stage: Stage
    sdk_version: str
    sdk_protocol: str
    cli_version: str
    cli_executable_blake3: str
    skills: tuple[CodexSkill, ...]
    skill_input_catalog: tuple[Mapping[str, str], ...]
    skill_input_catalog_blake3: str
    turn_mcp_metadata: Mapping[str, JsonValue]
    turn_mcp_launch_blake3: str

    def __post_init__(self) -> None:
        stage = _stage(self.stage)
        sdk_version = _clean_text(self.sdk_version, label="Codex SDK version")
        sdk_protocol = _clean_text(self.sdk_protocol, label="Codex SDK protocol")
        cli_version = _clean_text(self.cli_version, label="Codex CLI version")
        if sdk_version != REQUIRED_NATIVE_POLICY_CODEX_SDK_VERSION:
            raise NativePolicyV2Error("Codex SDK version is not the pinned Tier-A version")
        if cli_version != REQUIRED_NATIVE_POLICY_CODEX_CLI_VERSION:
            raise NativePolicyV2Error("Codex CLI version is not the pinned Tier-A version")
        if not is_blake3(self.cli_executable_blake3):
            raise NativePolicyV2Error("Codex CLI executable BLAKE3 differs")
        skills, catalog, catalog_blake3 = _skill_catalog(self.skills)
        supplied_catalog = tuple(canonical_value(row) for row in self.skill_input_catalog)
        if supplied_catalog != catalog or self.skill_input_catalog_blake3 != catalog_blake3:
            raise NativePolicyV2Error("SkillInput catalog commitment differs")
        turn_mcp, turn_mcp_blake3 = _turn_mcp_commitment(self.turn_mcp_metadata)
        if self.turn_mcp_launch_blake3 != turn_mcp_blake3:
            raise NativePolicyV2Error("TurnMCP launch binding differs")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "sdk_version", sdk_version)
        object.__setattr__(self, "sdk_protocol", sdk_protocol)
        object.__setattr__(self, "cli_version", cli_version)
        object.__setattr__(self, "skills", skills)
        object.__setattr__(self, "skill_input_catalog", catalog)
        object.__setattr__(self, "skill_input_catalog_blake3", catalog_blake3)
        object.__setattr__(self, "turn_mcp_metadata", turn_mcp)
        object.__setattr__(self, "turn_mcp_launch_blake3", turn_mcp_blake3)

    def core_document(self) -> dict[str, Any]:
        return {
            "schema": NATIVE_POLICY_V2_SCHEMA,
            "tier": NATIVE_POLICY_V2_TIER,
            "stage": self.stage.value,
            "role": "actor",
            "thread_lifecycle": {
                "fresh_thread_required": True,
                "ephemeral_required": True,
                "resume_allowed": False,
                "one_turn_only": True,
            },
            "actor_sandbox": {
                "mode": "workspace-write",
                "workspace_root_only": True,
                "writable_roots": ["{candidate_root}"],
                "additional_writable_roots": [],
                "network_access": False,
                "tmpdir_env_writable": False,
                "slash_tmp_writable": False,
            },
            "judge_boundary": {
                "mode": "read-only",
                "native_actions_allowed": False,
                "fresh_thread_required": True,
                "actor_thread_reuse_allowed": False,
            },
            "native_surface": dict(NATIVE_POLICY_V2_NATIVE_SURFACE),
            "native_action_types": list(NATIVE_POLICY_V2_ACTION_TYPES),
            "native_actions_parallel_allowed": False,
            "mcp_parallelism_unchanged": True,
            "approval_policy": "never",
            "config_overrides": list(NATIVE_POLICY_V2_CONFIG_OVERRIDES),
            "backend_wire_binding": canonical_value(NATIVE_POLICY_V2_BACKEND_WIRE),
            "sdk_binding": {
                "package": "openai-codex",
                "version": self.sdk_version,
                "protocol": self.sdk_protocol,
                "thread_method": "start_thread",
                "turn_method": "run_streamed",
            },
            "cli_binding": {
                "version": self.cli_version,
                "executable_blake3": self.cli_executable_blake3,
            },
            "skill_input_binding": {
                "sdk_input_type": "SkillInput",
                "catalog": list(self.skill_input_catalog),
                "catalog_blake3": self.skill_input_catalog_blake3,
            },
            "turn_mcp_binding": {
                "launch_blake3": self.turn_mcp_launch_blake3,
                "metadata": self.turn_mcp_metadata,
            },
            "schema_compatibility": {
                "evamed_schema_changed": False,
                "mcp_schema_changed": False,
                "rubric_schema_changed": False,
                "sandbox_schema_changed": False,
            },
        }

    @property
    def policy_blake3(self) -> str:
        return blake3_hex(self.core_document())

    def to_document(self) -> dict[str, Any]:
        return {**self.core_document(), "policy_blake3": self.policy_blake3}


def build_native_policy_v2(
    *,
    stage: Stage | str,
    sdk_version: str,
    sdk_protocol: str,
    cli_version: str,
    cli_executable_blake3: str,
    skills: Sequence[CodexSkill],
    turn_mcp_metadata: Mapping[str, Any],
) -> NativePolicyV2:
    """Build and immediately reopen one deterministic Tier-A policy."""

    selected, catalog, catalog_blake3 = _skill_catalog(skills)
    turn_mcp, turn_mcp_blake3 = _turn_mcp_commitment(turn_mcp_metadata)
    policy = NativePolicyV2(
        stage=_stage(stage),
        sdk_version=sdk_version,
        sdk_protocol=sdk_protocol,
        cli_version=cli_version,
        cli_executable_blake3=cli_executable_blake3,
        skills=selected,
        skill_input_catalog=catalog,
        skill_input_catalog_blake3=catalog_blake3,
        turn_mcp_metadata=turn_mcp,
        turn_mcp_launch_blake3=turn_mcp_blake3,
    )
    verify_native_policy_v2(policy)
    return policy


def verify_native_policy_v2(policy: NativePolicyV2) -> None:
    """Reopen all nested commitments without provider or filesystem access."""

    if not isinstance(policy, NativePolicyV2):
        raise NativePolicyV2Error("value is not a native policy v2")
    # Re-instantiation reopens skill and TurnMCP commitments and exact gates.
    reopened = NativePolicyV2(
        stage=policy.stage,
        sdk_version=policy.sdk_version,
        sdk_protocol=policy.sdk_protocol,
        cli_version=policy.cli_version,
        cli_executable_blake3=policy.cli_executable_blake3,
        skills=policy.skills,
        skill_input_catalog=policy.skill_input_catalog,
        skill_input_catalog_blake3=policy.skill_input_catalog_blake3,
        turn_mcp_metadata=policy.turn_mcp_metadata,
        turn_mcp_launch_blake3=policy.turn_mcp_launch_blake3,
    )
    if reopened.to_document() != policy.to_document() or not is_blake3(
        policy.policy_blake3
    ):
        raise NativePolicyV2Error("native policy v2 commitment differs")


def verify_native_policy_document_v2(document: Mapping[str, Any]) -> str:
    """Verify a serialized sidecar and return its exact policy BLAKE3."""

    if not isinstance(document, Mapping):
        raise NativePolicyV2Error("native policy document must be an object")
    policy_blake3 = document.get("policy_blake3")
    if not is_blake3(policy_blake3):
        raise NativePolicyV2Error("native policy document lacks policy BLAKE3")
    core = {key: value for key, value in document.items() if key != "policy_blake3"}
    expected_core_keys = {
        "schema",
        "tier",
        "stage",
        "role",
        "thread_lifecycle",
        "actor_sandbox",
        "judge_boundary",
        "native_surface",
        "native_action_types",
        "native_actions_parallel_allowed",
        "mcp_parallelism_unchanged",
        "approval_policy",
        "config_overrides",
        "backend_wire_binding",
        "sdk_binding",
        "cli_binding",
        "skill_input_binding",
        "turn_mcp_binding",
        "schema_compatibility",
    }
    if set(core) != expected_core_keys or blake3_hex(core) != policy_blake3:
        raise NativePolicyV2Error("native policy document BLAKE3 or fields differ")
    if (
        core.get("schema") != NATIVE_POLICY_V2_SCHEMA
        or core.get("tier") != NATIVE_POLICY_V2_TIER
        or core.get("stage") not in {stage.value for stage in NATIVE_POLICY_V2_STAGES}
        or core.get("role") != "actor"
        or core.get("native_action_types") != ["fileChange"]
        or core.get("native_surface") != dict(NATIVE_POLICY_V2_NATIVE_SURFACE)
        or core.get("native_actions_parallel_allowed") is not False
        or core.get("mcp_parallelism_unchanged") is not True
        or core.get("approval_policy") != "never"
        or core.get("config_overrides") != list(NATIVE_POLICY_V2_CONFIG_OVERRIDES)
    ):
        raise NativePolicyV2Error("native policy document capability differs")
    lifecycle = core.get("thread_lifecycle")
    actor = core.get("actor_sandbox")
    judge = core.get("judge_boundary")
    compatibility = core.get("schema_compatibility")
    if (
        lifecycle
        != {
            "fresh_thread_required": True,
            "ephemeral_required": True,
            "resume_allowed": False,
            "one_turn_only": True,
        }
        or actor
        != {
            "mode": "workspace-write",
            "workspace_root_only": True,
            "writable_roots": ["{candidate_root}"],
            "additional_writable_roots": [],
            "network_access": False,
            "tmpdir_env_writable": False,
            "slash_tmp_writable": False,
        }
        or judge
        != {
            "mode": "read-only",
            "native_actions_allowed": False,
            "fresh_thread_required": True,
            "actor_thread_reuse_allowed": False,
        }
        or compatibility
        != {
            "evamed_schema_changed": False,
            "mcp_schema_changed": False,
            "rubric_schema_changed": False,
            "sandbox_schema_changed": False,
        }
    ):
        raise NativePolicyV2Error("native policy document safety boundary differs")
    if core.get("backend_wire_binding") != canonical_value(
        NATIVE_POLICY_V2_BACKEND_WIRE
    ):
        raise NativePolicyV2Error("native policy backend wire binding differs")
    skill_binding = core.get("skill_input_binding")
    if not isinstance(skill_binding, Mapping):
        raise NativePolicyV2Error("native policy SkillInput binding differs")
    catalog = skill_binding.get("catalog")
    if (
        set(skill_binding) != {"sdk_input_type", "catalog", "catalog_blake3"}
        or skill_binding.get("sdk_input_type") != "SkillInput"
        or not isinstance(catalog, list)
        or not is_blake3(skill_binding.get("catalog_blake3"))
        or blake3_hex(catalog) != skill_binding["catalog_blake3"]
    ):
        raise NativePolicyV2Error("native policy SkillInput commitment differs")
    skill_ids: set[str] = set()
    skill_names: set[str] = set()
    for row in catalog:
        if not isinstance(row, Mapping):
            raise NativePolicyV2Error("native policy SkillInput catalog row differs")
        skill_path = Path(str(row.get("path", "")))
        if (
            set(row) != {"skill_id", "name", "path", "content_blake3"}
            or not all(
                isinstance(row.get(key), str) and row[key]
                for key in ("skill_id", "name", "path")
            )
            or not skill_path.is_absolute()
            or ".." in skill_path.parts
            or skill_path.as_posix() != row.get("path")
            or not is_blake3(row.get("content_blake3"))
            or row["skill_id"] in skill_ids
            or row["name"] in skill_names
        ):
            raise NativePolicyV2Error("native policy SkillInput catalog row differs")
        skill_payload = _read_regular_file_no_follow(
            skill_path,
            label="native policy SkillInput file",
        )
        if blake3_bytes(skill_payload) != row.get("content_blake3"):
            raise NativePolicyV2Error("native policy SkillInput catalog row differs")
        skill_ids.add(row["skill_id"])
        skill_names.add(row["name"])
    turn_mcp = core.get("turn_mcp_binding")
    if (
        not isinstance(turn_mcp, Mapping)
        or set(turn_mcp) != {"launch_blake3", "metadata"}
        or not isinstance(turn_mcp.get("metadata"), Mapping)
    ):
        raise NativePolicyV2Error("native policy TurnMCP binding differs")
    metadata = turn_mcp["metadata"]
    launch_blake3 = turn_mcp.get("launch_blake3")
    try:
        _, reopened_launch_blake3 = _turn_mcp_commitment(metadata)
    except NativePolicyV2Error as exc:
        raise NativePolicyV2Error("native policy TurnMCP commitment differs") from exc
    if reopened_launch_blake3 != launch_blake3:
        raise NativePolicyV2Error("native policy TurnMCP commitment differs")
    sdk = core.get("sdk_binding")
    cli = core.get("cli_binding")
    if (
        not isinstance(sdk, Mapping)
        or set(sdk)
        != {"package", "version", "protocol", "thread_method", "turn_method"}
        or sdk.get("package") != "openai-codex"
        or sdk.get("version") != REQUIRED_NATIVE_POLICY_CODEX_SDK_VERSION
        or not all(
            isinstance(sdk.get(key), str) and sdk[key]
            for key in ("version", "protocol")
        )
        or sdk.get("thread_method") != "start_thread"
        or sdk.get("turn_method") != "run_streamed"
        or not isinstance(cli, Mapping)
        or set(cli) != {"version", "executable_blake3"}
        or not isinstance(cli.get("version"), str)
        or cli["version"] != REQUIRED_NATIVE_POLICY_CODEX_CLI_VERSION
        or not is_blake3(cli.get("executable_blake3"))
    ):
        raise NativePolicyV2Error("native policy SDK/CLI binding differs")
    return policy_blake3


__all__ = [
    "NATIVE_POLICY_V2_ACTION_TYPES",
    "NATIVE_POLICY_V2_BACKEND_WIRE",
    "NATIVE_POLICY_V2_CONFIG_OVERRIDES",
    "NATIVE_POLICY_V2_NATIVE_SURFACE",
    "NATIVE_POLICY_V2_SCHEMA",
    "NATIVE_POLICY_V2_STAGES",
    "NATIVE_POLICY_V2_TIER",
    "REQUIRED_NATIVE_POLICY_CODEX_CLI_VERSION",
    "REQUIRED_NATIVE_POLICY_CODEX_SDK_VERSION",
    "STAGE_TOOL_CALL_PROOF_V1_SCHEMA",
    "STAGE_TOOL_FRONTIER_STATE_V1_SCHEMA",
    "STAGE_TOOL_GUIDANCE_V1_SCHEMA",
    "NativePolicyV2",
    "NativePolicyV2Error",
    "GuidedStageToolRuntimeV1",
    "StageToolGuidanceV1",
    "StageToolFrontierStateV1",
    "advance_guided_stage_frontier_v1",
    "build_native_policy_v2",
    "build_stage_tool_guidance_v1",
    "guard_guided_stage_tool_runtime_v1",
    "materialize_s2_selection_arguments_v1",
    "native_policy_v2_config_overrides",
    "start_guided_stage_frontier_v1",
    "verify_guided_stage_call_v1",
    "verify_guided_stage_frontier_state_v1",
    "verify_native_policy_document_v2",
    "verify_native_policy_v2",
    "verify_stage_tool_guidance_document_v1",
    "verify_stage_tool_guidance_v1",
]
