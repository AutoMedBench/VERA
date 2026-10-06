"""Deterministic, private Hugging Face release packaging for EVA-Med data.

The packager is deliberately an offline boundary.  It verifies existing SFT
and RL commitments, upgrades observable assistant decisions to a
Qwen-compatible chat representation, and writes an ignored release directory.
It never creates a Hub repository, uploads data, calls a model provider, or
manufactures hidden reasoning.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import gzip
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from typing import Any, BinaryIO
from uuid import NAMESPACE_URL, UUID, uuid5

from blake3 import blake3

from eva_agent.admission.receipts import SignedEnvelope, verify_signed_envelope
from eva_agent.pipeline.digests import (
    blake3_bytes,
    blake3_hex,
    canonical_json_bytes,
    canonical_value,
    is_blake3,
)


SFT_TARGET = "operator/EVA-Med-SFT-data"
RL_TARGET = "operator/EVA-Med-RL-data"
RELEASE_SCHEMA = "eva.private-hf-release-bundle.v1"
SFT_MANIFEST_SCHEMA = "eva.private-hf-qwen-sft-release.v1"
SFT_ROW_SCHEMA = "eva.qwen-observable-assistant-sft-row.v1"
RL_MANIFEST_SCHEMA = "eva.private-hf-rl-release.v1"
MIGRATION_SCHEMA = "eva.hf-release-migration-audit.v1"
LOSS_SCHEMA = "eva.qwen-assistant-decision-loss-contract.v1"
LEGACY_SFT_REPO = "operator/rlevo-Med-SFT-data"
LEGACY_SFT_REVISION = "227f87f04a5f2f4c2ec6013219bd594af391b482"
LEGACY_RL_REPO = "operator/rlevo-Med-RL-data"
LEGACY_RL_REVISION = "79dd2a31f5f3e5018608bf20f982ad70d2e5dafa"
PORTABLE_EVIDENCE_SCHEMA = "eva.rl-portable-evidence-index.v1"
PORTABLE_EVIDENCE_FILE_COUNT = 30_030
PORTABLE_EVIDENCE_ARCHIVE_COUNT = 25
VERIFIER_BUNDLE_SCHEMA = "eva.dataset-verifier-source-bundle.v1"
VERIFIER_BUNDLE_FILES = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "requirements-verifier.txt",
    "README.md",
    "scripts/build_bulk_rl_sandboxes_v1.py",
    "scripts/build_eva_hf_release_v1.py",
    "scripts/verify_eva_hf_dataset_v1.py",
)

SUPPORTED_SFT_DATASETS = {
    "eva.legacy-teacher-sft-dataset.v1": (
        "eva.legacy-teacher-sft-slice.v1",
        "strict_full_trajectory",
    ),
    "eva.execution-verified-frontier-prefix-sft-dataset.v2": (
        "eva.execution-verified-frontier-prefix-sft-slice.v2",
        "execution_verified_s1_frontier",
    ),
    "eva.execution-verified-frontier-prefix-sft-dataset.v3": (
        "eva.execution-verified-frontier-prefix-sft-slice.v3",
        "execution_verified_s1_frontier",
    ),
    "eva.execution-verified-s2-frontier-prefix-sft-dataset.v1": (
        "eva.execution-verified-s2-frontier-prefix-sft-slice.v1",
        "execution_verified_s2_frontier",
    ),
    "eva.execution-verified-s2-frontier-prefix-sft-dataset.v2": (
        "eva.execution-verified-s2-frontier-prefix-sft-slice.v2",
        "execution_verified_s2_frontier",
    ),
}

# Require a token boundary.  This intentionally does not match ordinary words
# such as ``risk-stratified`` or an ``sk-`` substring inside a public URL.
_CREDENTIAL_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{12,}", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])(?:ghp_|github_pat_)[A-Za-z0-9_]{16,}"),
    re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])AIza[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}"),
    re.compile(r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?i)(?<![A-Za-z0-9])Bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


class EVAHFReleaseError(ValueError):
    """A release input, migration, or deterministic output differs."""


@dataclass(frozen=True, slots=True)
class EVAHFReleaseBuild:
    output_root: Path
    sft_rows: int
    rl_rows: int
    rationale_rows: int
    credential_quarantine_rows: int
    release_blake3: str


@dataclass(frozen=True, slots=True)
class _VerifiedSFTRoot:
    root: Path
    dataset_id: str
    schema: str
    row_schema: str
    tier: str
    rows: tuple[Mapping[str, Any], ...]
    manifest_blake3: str


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EVAHFReleaseError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> Mapping[str, Any]:
    path = Path(path)
    try:
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise EVAHFReleaseError("JSON input topology differs")
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object
        )
    except EVAHFReleaseError:
        raise
    except (OSError, UnicodeError, ValueError):
        raise EVAHFReleaseError("JSON input could not be decoded") from None
    if not isinstance(value, Mapping):
        raise EVAHFReleaseError("JSON input must be an object")
    return value


def _regular_file(path: Path, *, label: str) -> Path:
    path = Path(path)
    try:
        metadata = path.lstat()
    except OSError:
        raise EVAHFReleaseError(f"{label} is unavailable") from None
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise EVAHFReleaseError(f"{label} topology differs")
    return path


def _safe_relative(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise EVAHFReleaseError(f"{label} path differs")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise EVAHFReleaseError(f"{label} path escapes its root")
    root_resolved = Path(root).resolve(strict=True)
    candidate = (root_resolved / relative).resolve(strict=True)
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        raise EVAHFReleaseError(f"{label} path escapes its root") from None
    return _regular_file(candidate, label=label)


def _stream_blake3(path: Path) -> str:
    digest = blake3()
    with _regular_file(path, label="content file").open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_json(value: Any) -> str:
    return json.dumps(
        canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _credential_rule(value: Any) -> str | None:
    text = _compact_json(value)
    for index, pattern in enumerate(_CREDENTIAL_PATTERNS):
        if pattern.search(text):
            return f"credential_pattern_{index + 1}"
    return None


def _credential_bytes_rule(payload: bytes) -> str | None:
    """Apply the same explicit boundary rules to portable artifact bytes."""

    text = payload.decode("utf-8", errors="ignore")
    for index, pattern in enumerate(_CREDENTIAL_PATTERNS):
        if pattern.search(text):
            return f"credential_pattern_{index + 1}"
    return None


def _uuid(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise EVAHFReleaseError(f"{label} must be UUID text")
    try:
        parsed = UUID(value)
    except ValueError:
        raise EVAHFReleaseError(f"{label} must be UUID text") from None
    if str(parsed) != value:
        raise EVAHFReleaseError(f"{label} UUID is not canonical")
    return value


def _arguments(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value, object_pairs_hook=_strict_object)
        except (TypeError, ValueError):
            raise EVAHFReleaseError("tool-call arguments are not JSON") from None
    if not isinstance(value, Mapping):
        raise EVAHFReleaseError("tool-call arguments must be an object")
    return _compact_json(value)


def _tool_call(
    value: Mapping[str, Any], *, fallback_id: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EVAHFReleaseError("tool call must be an object")
    function = value.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = value.get("name")
        arguments = value.get("arguments")
    if not isinstance(name, str) or not name:
        raise EVAHFReleaseError("tool-call name differs")
    call_id = value.get("id", fallback_id)
    if not isinstance(call_id, str) or not call_id:
        raise EVAHFReleaseError("tool-call id differs")
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": _arguments(arguments)},
    }


def _content(value: Any, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return _compact_json(value)


def _normalise_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    row_id: str,
    target_is_last: bool,
    target_visible_rationale: str | None = None,
) -> tuple[list[Mapping[str, Any]], list[str]]:
    """Normalise messages and synthesize only protocol-level call IDs.

    Historical assistant rationales are deliberately not promoted to target
    content.  Only a visible rationale on the final supervised target may be
    wrapped in ``<think>``.
    """

    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise EVAHFReleaseError("messages must be a sequence")
    result: list[Mapping[str, Any]] = []
    pending: deque[tuple[str, str]] = deque()
    target_call_ids: list[str] = []
    seen_call_ids: set[str] = set()
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise EVAHFReleaseError("message must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise EVAHFReleaseError("message role differs")
        if role in {"system", "user"}:
            result.append({"role": role, "content": _content(message.get("content"))})
            continue
        if role == "assistant":
            embedded = message.get("content")
            calls_value = message.get("tool_calls")
            if calls_value is None and isinstance(embedded, Mapping):
                calls_value = embedded.get("tool_calls")
                embedded = None
            if calls_value is None:
                calls_value = []
            if not isinstance(calls_value, list):
                raise EVAHFReleaseError("assistant tool_calls differ")
            calls: list[Mapping[str, Any]] = []
            for call_index, call in enumerate(calls_value):
                fallback = "call_" + str(
                    uuid5(
                        NAMESPACE_URL,
                        f"eva-hf-release:{row_id}:{message_index}:{call_index}",
                    )
                )
                normalised = _tool_call(call, fallback_id=fallback)
                call_id = str(normalised["id"])
                if call_id in seen_call_ids:
                    raise EVAHFReleaseError("tool-call id is duplicated")
                seen_call_ids.add(call_id)
                pending.append((call_id, str(normalised["function"]["name"])))
                calls.append(normalised)
            is_target = target_is_last and message_index == len(messages) - 1
            if is_target and target_visible_rationale is not None:
                assistant_content: str | None = (
                    f"<think>{target_visible_rationale}</think>"
                )
            elif calls:
                # Do not turn legacy per-step rationale fields into model
                # targets or expose opaque internal content objects.
                assistant_content = (
                    embedded if isinstance(embedded, str) and embedded else None
                )
            else:
                assistant_content = _content(embedded, allow_none=True)
            output: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_content,
            }
            if calls:
                output["tool_calls"] = calls
            result.append(output)
            if is_target:
                target_call_ids.extend(str(call["id"]) for call in calls)
            continue

        # Tool messages in the legacy semantic corpus carry ``observation``;
        # current verified frontier rows carry structured ``content``.
        if not pending:
            raise EVAHFReleaseError("tool observation has no preceding call")
        call_id, expected_name = pending.popleft()
        explicit_id = message.get("tool_call_id")
        if explicit_id is not None and explicit_id != call_id:
            raise EVAHFReleaseError("tool observation call id differs")
        name = message.get("name", expected_name)
        if name != expected_name:
            raise EVAHFReleaseError("tool observation name differs")
        observation = (
            message["observation"] if "observation" in message else message.get("content")
        )
        result.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": _content(observation),
            }
        )

    if pending and not target_is_last:
        raise EVAHFReleaseError("message history has unresolved tool calls")
    if pending and any(call_id not in target_call_ids for call_id, _ in pending):
        raise EVAHFReleaseError("non-target tool call lacks its observation")
    return result, target_call_ids


def _normalise_tools(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise EVAHFReleaseError("tools must be a list")
    result: list[Mapping[str, Any]] = []
    names: set[str] = set()
    for tool in value:
        if not isinstance(tool, Mapping) or tool.get("type") != "function":
            raise EVAHFReleaseError("Qwen tool schema differs")
        function = tool.get("function")
        if not isinstance(function, Mapping):
            raise EVAHFReleaseError("Qwen tool function differs")
        name = function.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise EVAHFReleaseError("Qwen tool name differs")
        if not isinstance(function.get("parameters"), Mapping):
            raise EVAHFReleaseError("Qwen tool parameters differ")
        names.add(name)
        result.append(canonical_value(tool))
    return result


def _frontier_tool_catalog(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    workspace_root = row.get("workspace_root")
    if not isinstance(workspace_root, str) or not workspace_root:
        raise EVAHFReleaseError("frontier workspace root differs")
    policy_path = Path(workspace_root) / ".eva" / "source-policy.json"
    policy_file = _regular_file(policy_path, label="frontier source policy")
    if row.get("source_policy_file_blake3") != _stream_blake3(policy_file):
        raise EVAHFReleaseError("frontier source policy BLAKE3 differs")
    policy = _read_json(policy_file)
    source_tools = policy.get("tools")
    if not isinstance(source_tools, list) or not source_tools:
        raise EVAHFReleaseError("frontier source tool catalog differs")
    projected: list[Mapping[str, Any]] = []
    for tool in source_tools:
        if not isinstance(tool, Mapping):
            raise EVAHFReleaseError("frontier source tool differs")
        name = tool.get("name")
        description = tool.get("description")
        parameters = tool.get("input_schema")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(description, str)
            or not isinstance(parameters, Mapping)
        ):
            raise EVAHFReleaseError("frontier source tool schema differs")
        function: dict[str, Any] = {
            "name": name,
            "description": description,
            "parameters": canonical_value(parameters),
        }
        for key, value in tool.items():
            if isinstance(key, str) and key.startswith("x-eva-"):
                function[key] = canonical_value(value)
        projected.append({"type": "function", "function": function})
    if row.get("schema") == "eva.execution-verified-s2-frontier-prefix-sft-slice.v2":
        from eva_agent.harness.skills import SkillCatalog

        expected = {
            item["function"]["name"]: {
                "name": item["function"]["name"], "description": item["function"]["description"],
                "input_schema": item["function"]["parameters"],
            } for item in projected
        }
        for definition in SkillCatalog(()).tool_definitions():
            skill = {"name": definition.name, "description": definition.description,
                     "input_schema": canonical_value(definition.parameters)}
            if definition.name in expected and expected[definition.name] != skill:
                raise EVAHFReleaseError("source policy changes a canonical skill tool")
            if definition.name not in expected:
                projected.append({"type": "function", "function": {
                    "name": definition.name, "description": definition.description,
                    "parameters": canonical_value(definition.parameters),
                }})
            expected[definition.name] = skill
        offered = row.get("offered_tool_catalog")
        if (not isinstance(offered, list) or len(offered) != len(expected)
                or any(not isinstance(item, Mapping) for item in offered)
                or {item.get("name"): item for item in offered} != expected
                or row.get("offered_tool_catalog_blake3") != blake3_hex(offered)
                or row.get("codex_offered_tool_schema_blake3") != blake3_hex(offered)):
            raise EVAHFReleaseError("skill-aware S2 offered tool catalog differs")
    return _normalise_tools(projected)


def _verify_sft_root(root: Path) -> _VerifiedSFTRoot:
    root = Path(root)
    manifest = _read_json(root / "manifest.json")
    schema = manifest.get("schema")
    if schema not in SUPPORTED_SFT_DATASETS:
        raise EVAHFReleaseError("SFT source schema is not supported")
    row_schema, tier = SUPPORTED_SFT_DATASETS[str(schema)]
    core = {key: value for key, value in manifest.items() if key != "manifest_blake3"}
    if manifest.get("manifest_blake3") != blake3_hex(core):
        raise EVAHFReleaseError("SFT source manifest BLAKE3 differs")
    dataset_id = _uuid(manifest.get("dataset_id"), label="SFT dataset_id")
    if manifest.get("hidden_reasoning_included") is not False:
        raise EVAHFReleaseError("SFT source includes hidden reasoning")
    if schema.startswith("eva.execution-verified-s2"):
        if manifest.get("private_reference_observations_included") is not False:
            raise EVAHFReleaseError("S2 source includes private reference observations")
    elif manifest.get("private_reference_included") is not False:
        raise EVAHFReleaseError("SFT source includes private reference material")
    descriptors = manifest.get("shards")
    if not isinstance(descriptors, list) or not descriptors:
        raise EVAHFReleaseError("SFT source shard inventory differs")
    rows: list[Mapping[str, Any]] = []
    seen_examples: set[str] = set()
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise EVAHFReleaseError("SFT source shard descriptor differs")
        shard = _safe_relative(root, descriptor.get("path"), label="SFT source shard")
        if descriptor.get("content_blake3") != _stream_blake3(shard):
            raise EVAHFReleaseError("SFT source shard BLAKE3 differs")
        shard_count = 0
        try:
            handle = shard.open("r", encoding="utf-8")
            with handle:
                for line in handle:
                    row = json.loads(line, object_pairs_hook=_strict_object)
                    if not isinstance(row, Mapping):
                        raise EVAHFReleaseError("SFT source row differs")
                    row_core = {
                        key: value for key, value in row.items() if key != "slice_blake3"
                    }
                    if (
                        row.get("schema") != row_schema
                        or row.get("dataset_id") != dataset_id
                        or row.get("slice_blake3") != blake3_hex(row_core)
                        or row.get("hidden_reasoning_included") is not False
                    ):
                        raise EVAHFReleaseError("SFT source row commitment differs")
                    if row.get("private_reference_included", False) is not False:
                        raise EVAHFReleaseError("SFT source row exposes a private reference")
                    if row.get("private_reference_observations_included", False) is not False:
                        raise EVAHFReleaseError("SFT source row exposes private observations")
                    example = _uuid(row.get("example_id"), label="SFT example_id")
                    if example in seen_examples:
                        raise EVAHFReleaseError("SFT example_id is duplicated")
                    if _credential_rule(row) is not None:
                        raise EVAHFReleaseError("verified SFT source contains a credential")
                    seen_examples.add(example)
                    rows.append(row)
                    shard_count += 1
        except EVAHFReleaseError:
            raise
        except (OSError, UnicodeError, ValueError):
            raise EVAHFReleaseError("SFT source shard could not be decoded") from None
        if descriptor.get("slice_count") != shard_count:
            raise EVAHFReleaseError("SFT source shard count differs")
    if manifest.get("slice_count") != len(rows):
        raise EVAHFReleaseError("SFT source total differs")
    return _VerifiedSFTRoot(
        root=root,
        dataset_id=dataset_id,
        schema=str(schema),
        row_schema=row_schema,
        tier=tier,
        rows=tuple(rows),
        manifest_blake3=str(manifest["manifest_blake3"]),
    )


def _verified_sft_row(
    row: Mapping[str, Any], source: _VerifiedSFTRoot
) -> Mapping[str, Any]:
    row_id = str(
        uuid5(
            NAMESPACE_URL,
            f"eva-hf-release:verified:{source.dataset_id}:{row['example_id']}",
        )
    )
    if source.tier == "strict_full_trajectory":
        prefix, _ = _normalise_messages(
            row.get("prefix", []), row_id=row_id, target_is_last=False
        )
        decision = row.get("supervised_assistant_decision")
        if not isinstance(decision, Mapping) or decision.get("role") != "assistant":
            raise EVAHFReleaseError("strict SFT target decision differs")
        target, target_calls = _normalise_messages(
            [decision], row_id=row_id, target_is_last=True
        )
        messages = prefix + target
        observation_evidence = canonical_value(row.get("target_tool_observations", []))
        tools: list[Mapping[str, Any]] = []
        tool_catalog_scope = "unavailable_legacy_verified"
    else:
        source_messages = row.get("messages")
        if not isinstance(source_messages, list) or len(source_messages) < 2:
            raise EVAHFReleaseError("frontier SFT messages differ")
        if source_messages[-1].get("role") != "tool" or source_messages[-2].get("role") != "assistant":
            raise EVAHFReleaseError("frontier target boundary differs")
        # The host observation proves the action, but is not a model-output
        # token.  Strip exactly that final observation from training messages.
        prefix_messages = source_messages[:-2]
        target_message = source_messages[-2]
        prefix, _ = _normalise_messages(
            prefix_messages, row_id=row_id, target_is_last=False
        )
        target, target_calls = _normalise_messages(
            [target_message], row_id=row_id, target_is_last=True
        )
        messages = prefix + target
        observation_evidence = canonical_value(source_messages[-1])
        tools = _frontier_tool_catalog(row)
        tool_catalog_scope = "candidate_source_policy_reopened"
    if not messages or messages[-1].get("role") != "assistant":
        raise EVAHFReleaseError("SFT training target is not final assistant")
    commitments = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "messages",
            "prefix",
            "supervised_assistant_decision",
            "target_tool_observations",
        }
    }
    output: dict[str, Any] = {
        "schema": SFT_ROW_SCHEMA,
        "row_id": row_id,
        "messages": messages,
        "tools": tools,
        "parallel_tool_calls_supported": True,
        "metadata": {
            "migration_status": "upgrade",
            "quality_tier": source.tier,
            "stage": row.get("stage"),
            "model_id": row.get("model_id"),
            "sandbox_id": row.get("sandbox_id"),
            "source_dataset_id": source.dataset_id,
            "source_example_id": row.get("example_id"),
            "source_schema": source.row_schema,
            "tool_catalog_scope": tool_catalog_scope,
        },
        "evidence": {
            "source_manifest_blake3": source.manifest_blake3,
            "source_slice_blake3": row.get("slice_blake3"),
            "source_commitments": canonical_value(commitments),
            "target_host_observation": observation_evidence,
        },
        "loss_contract": {
            "schema": LOSS_SCHEMA,
            "assistant_message_index": len(messages) - 1,
            "target_tool_call_ids": target_calls,
            "supervise_exactly_one_assistant_message": True,
            "subsequent_observations_in_training_messages": False,
        },
    }
    return canonical_value(output)


def _gzip_rows(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    path = _regular_file(path, label="legacy gzip corpus")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line, object_pairs_hook=_strict_object)
                if not isinstance(row, Mapping):
                    raise EVAHFReleaseError("legacy semantic row differs")
                yield line_number, row
    except EVAHFReleaseError:
        raise
    except (OSError, UnicodeError, ValueError):
        raise EVAHFReleaseError("legacy gzip corpus could not be decoded") from None


def _legacy_row(
    row: Mapping[str, Any], *, line_number: int, release: str
) -> Mapping[str, Any]:
    target = row.get("assistant_target")
    history = row.get("messages")
    metadata = row.get("metadata")
    if not isinstance(target, Mapping) or not isinstance(history, list) or not isinstance(metadata, Mapping):
        raise EVAHFReleaseError("legacy semantic row shape differs")
    row_id = str(
        uuid5(
            NAMESPACE_URL,
            (
                "eva-hf-release:legacy-semantic:"
                f"{LEGACY_SFT_REPO}:{LEGACY_SFT_REVISION}:"
                f"data/curated20k_semantic.jsonl.gz:{line_number}"
            ),
        )
    )
    prefix, _ = _normalise_messages(history, row_id=row_id, target_is_last=False)
    rationale = target.get("rationale")
    visible_rationale = rationale.strip() if isinstance(rationale, str) and rationale.strip() else None
    calls = target.get("tool_calls", [])
    if not isinstance(calls, list):
        raise EVAHFReleaseError("legacy semantic target tool calls differ")
    target_message: dict[str, Any] = {
        "role": "assistant",
        "content": target.get("content"),
    }
    if calls:
        target_message["tool_calls"] = calls
    target_messages, target_calls = _normalise_messages(
        [target_message],
        row_id=row_id,
        target_is_last=True,
        target_visible_rationale=visible_rationale,
    )
    messages = prefix + target_messages
    if not calls and not isinstance(target.get("content"), str):
        raise EVAHFReleaseError("legacy terminal target content differs")
    output: dict[str, Any] = {
        "schema": SFT_ROW_SCHEMA,
        "row_id": row_id,
        "messages": messages,
        "tools": _normalise_tools(row.get("tools")),
        "parallel_tool_calls_supported": True,
        "metadata": {
            "migration_status": "upgrade",
            "quality_tier": "legacy_semantic_upgrade",
            "stage": metadata.get("stage"),
            "source_release": release,
            "source_repo": LEGACY_SFT_REPO,
            "source_revision": LEGACY_SFT_REVISION,
            "source_relative_path": "data/curated20k_semantic.jsonl.gz",
            "source_line": line_number,
            "legacy_lineage_row_id": metadata.get("row_id"),
            "legacy_metadata": canonical_value(metadata),
        },
        "evidence": {
            "source_kind": "legacy_semantic_corpus",
            "source_line": line_number,
            "source_revision": LEGACY_SFT_REVISION,
            "provider_private_reasoning_included": False,
        },
        "loss_contract": {
            "schema": LOSS_SCHEMA,
            "assistant_message_index": len(messages) - 1,
            "target_tool_call_ids": target_calls,
            "supervise_exactly_one_assistant_message": True,
            "subsequent_observations_in_training_messages": False,
        },
    }
    if visible_rationale is not None:
        output["decision_summary"] = {
            "format": "think",
            "source": "assistant_target.rationale",
            "verbatim_visible": True,
            "provider_private_cot": False,
            "text": f"<think>{visible_rationale}</think>",
        }
    return canonical_value(output)


class _ShardWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: BinaryIO = path.open("xb")
        self.digest = blake3()
        self.rows = 0
        self.bytes = 0
        self.first_row_id: str | None = None
        self.last_row_id: str | None = None

    def write(self, row: Mapping[str, Any]) -> None:
        payload = canonical_json_bytes(row)
        self.handle.write(payload)
        self.digest.update(payload)
        row_id = str(row["row_id"])
        self.first_row_id = self.first_row_id or row_id
        self.last_row_id = row_id
        self.rows += 1
        self.bytes += len(payload)

    def close(self, *, relative_path: str) -> Mapping[str, Any]:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        return {
            "path": relative_path,
            "row_count": self.rows,
            "byte_count": self.bytes,
            "content_blake3": self.digest.hexdigest(),
            "first_row_id": self.first_row_id,
            "last_row_id": self.last_row_id,
        }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise EVAHFReleaseError("release output already exists") from None


def _write_text(path: Path, value: str) -> Mapping[str, Any]:
    payload = value.encode("utf-8")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise EVAHFReleaseError("release output already exists") from None
    return {
        "path": path.name,
        "byte_count": len(payload),
        "content_blake3": blake3_bytes(payload),
    }


def _verifier_source_files(repository_root: Path) -> list[tuple[Path, PurePosixPath]]:
    """Return the deterministic, secret-free verifier source snapshot."""

    fixed = [
        repository_root / "LICENSE",
        repository_root / "THIRD_PARTY_NOTICES.md",
        repository_root / "pyproject.toml",
        repository_root / "scripts" / "build_bulk_rl_sandboxes_v1.py",
        repository_root / "scripts" / "build_eva_hf_release_v1.py",
        repository_root / "scripts" / "verify_eva_hf_dataset_v1.py",
    ]
    discovered = [
        *sorted((repository_root / "src" / "eva_agent").rglob("*.py")),
        *sorted((repository_root / "rubrics").rglob("*.json")),
        *sorted((repository_root / "schemas").rglob("*.json")),
    ]
    rows: list[tuple[Path, PurePosixPath]] = []
    seen: set[str] = set()
    for source in [*fixed, *discovered]:
        source = source.resolve(strict=True)
        relative = PurePosixPath(source.relative_to(repository_root.resolve(strict=True)).as_posix())
        if relative.as_posix() in seen:
            raise EVAHFReleaseError("verifier source inventory is duplicated")
        seen.add(relative.as_posix())
        rows.append((source, relative))
    return sorted(rows, key=lambda row: row[1].as_posix())


def _copy_verifier_bundle(output: Path, *, dataset_kind: str) -> Mapping[str, Any]:
    """Embed an executable verifier source tree in a released dataset."""

    if dataset_kind not in {"sft", "rl"}:
        raise EVAHFReleaseError("verifier dataset kind differs")
    repository_root = Path(__file__).resolve().parents[3]
    verifier_root = output / "verifier"
    verifier_root.mkdir()
    inventory: list[Mapping[str, Any]] = []
    total_bytes = 0
    for source, relative in _verifier_source_files(repository_root):
        source = _regular_file(source, label="verifier source")
        if source.stat().st_size > 32 << 20:
            raise EVAHFReleaseError("verifier source is oversized")
        payload = source.read_bytes()
        if _credential_bytes_rule(payload) is not None:
            raise EVAHFReleaseError("verifier source contains a credential")
        target = verifier_root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        inventory.append(
            {
                "path": relative.as_posix(),
                "byte_count": len(payload),
                "content_blake3": blake3_bytes(payload),
            }
        )
        total_bytes += len(payload)
    requirements = (
        "blake3==1.0.9\n"
        "cryptography==50.0.1\n"
        "jsonschema==4.26.0\n"
        "openai==2.54.0\n"
        "openai-codex==0.147.0\n"
    ).encode("utf-8")
    readme = (
        "# Embedded EVA dataset verifier\n\n"
        "This is a byte-committed source snapshot of the verifier used for this "
        "release. It contains the EVA verifier modules, rubric sources, JSON schemas, "
        "and entrypoints. Python 3.11+ and the `zstd` executable are required.\n\n"
        "```bash\n"
        "python -m venv .verify-venv\n"
        ".verify-venv/bin/pip install -r verifier/requirements-verifier.txt\n"
        ".verify-venv/bin/python verifier/scripts/verify_eva_hf_dataset_v1.py --dataset-root .\n"
        "```\n"
    ).encode("utf-8")
    for relative, payload in (
        (PurePosixPath("requirements-verifier.txt"), requirements),
        (PurePosixPath("README.md"), readme),
    ):
        target = verifier_root.joinpath(*relative.parts)
        with target.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        inventory.append(
            {
                "path": relative.as_posix(),
                "byte_count": len(payload),
                "content_blake3": blake3_bytes(payload),
            }
        )
        total_bytes += len(payload)
    inventory.sort(key=lambda row: str(row["path"]))
    core: dict[str, Any] = {
        "schema": VERIFIER_BUNDLE_SCHEMA,
        "dataset_kind": dataset_kind,
        "python_requires": ">=3.11",
        "system_requirements": ["zstd"],
        "entrypoint": "scripts/verify_eva_hf_dataset_v1.py",
        "file_count": len(inventory),
        "byte_count": total_bytes,
        "files": inventory,
    }
    manifest = {**core, "bundle_blake3": blake3_hex(core)}
    _write_json(verifier_root / "manifest.json", manifest)
    return {
        "schema": VERIFIER_BUNDLE_SCHEMA,
        "path": "verifier/manifest.json",
        "dataset_kind": dataset_kind,
        "file_count": len(inventory),
        "byte_count": total_bytes,
        "bundle_blake3": manifest["bundle_blake3"],
        "manifest_content_blake3": _stream_blake3(verifier_root / "manifest.json"),
        "entrypoint": "verifier/scripts/verify_eva_hf_dataset_v1.py",
    }


def _verify_verifier_bundle(dataset_root: Path, descriptor: Any) -> Mapping[str, Any]:
    if not isinstance(descriptor, Mapping):
        raise EVAHFReleaseError("verifier bundle descriptor differs")
    manifest_path = _safe_relative(
        dataset_root, descriptor.get("path"), label="verifier bundle manifest"
    )
    manifest = _read_json(manifest_path)
    core = {key: value for key, value in manifest.items() if key != "bundle_blake3"}
    files = manifest.get("files")
    if (
        manifest.get("schema") != VERIFIER_BUNDLE_SCHEMA
        or descriptor.get("schema") != VERIFIER_BUNDLE_SCHEMA
        or manifest.get("bundle_blake3") != blake3_hex(core)
        or descriptor.get("bundle_blake3") != manifest.get("bundle_blake3")
        or descriptor.get("manifest_content_blake3") != _stream_blake3(manifest_path)
        or descriptor.get("dataset_kind") != manifest.get("dataset_kind")
        or not isinstance(files, list)
    ):
        raise EVAHFReleaseError("verifier bundle manifest differs")
    seen: set[str] = set()
    total_bytes = 0
    for row in files:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise EVAHFReleaseError("verifier bundle file row differs")
        path = str(row["path"])
        if path in seen:
            raise EVAHFReleaseError("verifier bundle file is duplicated")
        source = _safe_relative(dataset_root / "verifier", path, label="verifier bundle file")
        source = _regular_file(source, label="verifier bundle file")
        if source.stat().st_size > 32 << 20:
            raise EVAHFReleaseError("verifier bundle file is oversized")
        payload = source.read_bytes()
        if (
            row.get("byte_count") != len(payload)
            or row.get("content_blake3") != blake3_bytes(payload)
            or _credential_bytes_rule(payload) is not None
        ):
            raise EVAHFReleaseError("verifier bundle file commitment differs")
        seen.add(path)
        total_bytes += len(payload)
    required = set(VERIFIER_BUNDLE_FILES)
    if (
        not required.issubset(seen)
        or manifest.get("file_count") != len(files)
        or descriptor.get("file_count") != len(files)
        or manifest.get("byte_count") != total_bytes
        or descriptor.get("byte_count") != total_bytes
        or descriptor.get("entrypoint") != "verifier/scripts/verify_eva_hf_dataset_v1.py"
    ):
        raise EVAHFReleaseError("verifier bundle inventory differs")
    return manifest


def _card_files(
    output: Path, *, target: str, row_count: int, kind: str,
    quality_counts: Mapping[str, int] | None = None,
) -> list[Mapping[str, Any]]:
    if kind == "sft":
        counts = dict(quality_counts or {})
        if sum(counts.values()) != row_count:
            raise EVAHFReleaseError("dataset card quality counts differ from written SFT rows")
        legacy_count = counts.get("legacy_semantic_upgrade", 0)
        verified_count = row_count - legacy_count
        notes = (
            "Rows are Qwen-chat compatible and supervise exactly one final assistant "
            "decision. Earlier visible tool/skill observations may appear as masked "
            "conditioning history; the final target's host verification result remains "
            "audit evidence outside causal training messages. Visible <think> summaries "
            "are copied only from explicit legacy "
            "decision summaries; provider-private chain of thought is never included. "
            f"The release contains {verified_count:,} current trajectory/frontier-verified rows plus "
            f"{legacy_count:,} same-owner legacy semantic rows validated for schema, offered target "
            "tools, arguments, credentials, and duplicates. The legacy tier is explicitly "
            "labelled and does not claim current execution attestation. Schedule rows are "
            "excluded."
        )
    else:
        notes = (
            f"The {row_count:,} train sandboxes are direct byte-for-byte reuse of the signed EVA "
            "RL release. Each sandbox retains its exact rubric and verifiable evidence "
            "contract."
        )
    readme = (
        "---\n"
        "license: other\n"
        "task_categories:\n"
        "- text-generation\n"
        "language:\n"
        "- en\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files:\n"
        "  - split: train\n"
        "    path: data/train-*.jsonl\n"
        "pretty_name: EVA Medical Research Training Data\n"
        "---\n\n"
        f"# {target}\n\n"
        "Private training release generated offline by EVA-Agent.\n\n"
        f"Rows: {row_count}\n\n"
        f"{notes}\n\n"
        "The offline release builder performs no upload and creates no Hugging Face repository.\n"
    )
    readme_descriptor = _write_text(output / "README.md", readme)
    version = {
        "schema": "eva.private-hf-dataset-version.v1",
        "dataset": target,
        "release": "eva-med-hf-release-v1",
        "visibility": "private",
        "row_count": row_count,
        "split_counts": {"train": row_count},
        "offline_builder_upload_performed": False,
        "offline_builder_repo_created": False,
    }
    _write_json(output / "VERSION.json", version)
    version_path = output / "VERSION.json"
    return [
        readme_descriptor,
        {
            "path": "VERSION.json",
            "byte_count": version_path.stat().st_size,
            "content_blake3": _stream_blake3(version_path),
        },
    ]


def _semantic_key(row: Mapping[str, Any]) -> str:
    return _compact_json({"messages": row["messages"], "tools": row.get("tools", [])})


def _write_sft_release(
    output: Path,
    *,
    verified_roots: Sequence[_VerifiedSFTRoot],
    legacy_root: Path,
    expected_current_rows: int,
    expected_legacy_semantic_rows: int,
    expected_legacy_schedule_rows: int,
    expected_credential_quarantine_rows: int | None,
    shard_size: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    version = _read_json(legacy_root / "VERSION.json")
    release = version.get("release")
    if not isinstance(release, str) or not release:
        raise EVAHFReleaseError("legacy SFT release identity differs")
    semantic_path = legacy_root / "data" / "curated20k_semantic.jsonl.gz"
    schedule_path = legacy_root / "data" / "curated20k_schedule.jsonl.gz"
    schedule_rows = sum(1 for _ in _gzip_rows(schedule_path))
    if schedule_rows != expected_legacy_schedule_rows:
        raise EVAHFReleaseError("legacy schedule row count differs")

    data_root = output / "data"
    data_root.mkdir(parents=True)
    descriptors: list[Mapping[str, Any]] = []
    tier_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_semantics: set[str] = set()
    quarantine: list[Mapping[str, Any]] = []
    current_scanned = 0
    current_rows = 0
    legacy_rows = 0
    rationale_rows = 0
    target_tool_calls = 0
    offered_tools = 0
    duplicate_skips: Counter[tuple[str, str]] = Counter()
    writer: _ShardWriter | None = None
    shard_index = 0

    def emit(row: Mapping[str, Any], *, source_identity: str) -> bool:
        nonlocal writer, shard_index, target_tool_calls, offered_tools
        row_id = _uuid(row.get("row_id"), label="release SFT row_id")
        if row_id in seen_ids:
            raise EVAHFReleaseError("release SFT row_id is duplicated")
        semantic = _semantic_key(row)
        if semantic in seen_semantics:
            duplicate_skips[(source_identity, str(row["metadata"]["quality_tier"]))] += 1
            return False
        if _credential_rule(row) is not None:
            raise EVAHFReleaseError("release SFT row contains a credential")
        if writer is None:
            path = data_root / f"train-{shard_index:05d}.jsonl"
            writer = _ShardWriter(path)
        writer.write(row)
        seen_ids.add(row_id)
        seen_semantics.add(semantic)
        metadata = row["metadata"]
        tier_counts[str(metadata["quality_tier"])] += 1
        stage_counts[str(metadata["stage"])] += 1
        target_tool_calls += len(row["loss_contract"]["target_tool_call_ids"])
        offered_tools += len(row.get("tools", []))
        if writer.rows == shard_size:
            relative = f"data/{writer.path.name}"
            descriptors.append(writer.close(relative_path=relative))
            writer = None
            shard_index += 1
        return True

    priority = {
        "strict_full_trajectory": 0,
        "execution_verified_s2_frontier": 1,
        "execution_verified_s1_frontier": 2,
    }
    included_by_dataset: Counter[str] = Counter()
    for source in sorted(verified_roots, key=lambda item: (priority[item.tier], item.dataset_id)):
        for source_row in source.rows:
            current_scanned += 1
            if emit(
                _verified_sft_row(source_row, source),
                source_identity=source.dataset_id,
            ):
                current_rows += 1
                included_by_dataset[source.dataset_id] += 1
    if current_scanned != expected_current_rows:
        raise EVAHFReleaseError("verified SFT source total differs")

    semantic_scanned = 0
    for line_number, source_row in _gzip_rows(semantic_path):
        semantic_scanned += 1
        rule = _credential_rule(source_row)
        if rule is not None:
            quarantine.append(
                {"source_line": line_number, "status": "reject", "reason": rule}
            )
            continue
        row = _legacy_row(source_row, line_number=line_number, release=release)
        if emit(row, source_identity=f"{LEGACY_SFT_REPO}@{LEGACY_SFT_REVISION}"):
            legacy_rows += 1
            rationale_rows += "decision_summary" in row
    if semantic_scanned != expected_legacy_semantic_rows:
        raise EVAHFReleaseError("legacy semantic row count differs")
    if (
        expected_credential_quarantine_rows is not None
        and len(quarantine) != expected_credential_quarantine_rows
    ):
        raise EVAHFReleaseError("legacy credential quarantine count differs")
    if writer is not None:
        relative = f"data/{writer.path.name}"
        descriptors.append(writer.close(relative_path=relative))

    card_files = _card_files(
        output,
        target=SFT_TARGET,
        row_count=current_rows + legacy_rows,
        kind="sft",
        quality_counts=tier_counts,
    )
    verifier_bundle = _copy_verifier_bundle(output, dataset_kind="sft")
    local_migration = {
        "schema": "eva.private-hf-dataset-migration-report.v1",
        "target_dataset": SFT_TARGET,
        "legacy_repo": LEGACY_SFT_REPO,
        "legacy_revision": LEGACY_SFT_REVISION,
        "legacy_semantic_scanned": semantic_scanned,
        "legacy_semantic_included": legacy_rows,
        "credential_quarantine_count": len(quarantine),
        "semantic_duplicate_skips": [
            {
                "source": source,
                "quality_tier": tier,
                "status": "reject",
                "reason": "lower_priority_semantic_duplicate",
                "count": count,
            }
            for (source, tier), count in sorted(duplicate_skips.items())
        ],
        "statuses": ["upgrade", "reject"],
        "reproducibility_command": (
            "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -B "
            "scripts/build_eva_hf_release_v1.py build"
        ),
    }
    _write_json(output / "migration-report.json", local_migration)
    migration_markdown = (
        "# Migration and reproducibility\n\n"
        f"Legacy source: `{LEGACY_SFT_REPO}@{LEGACY_SFT_REVISION}`.\n\n"
        "Only semantic rows are upgraded. Schedule duplicates are rejected. "
        "Credential matches and lower-priority semantic duplicates are excluded "
        "and counted in `migration-report.json`.\n\n"
        "Rebuild from the EVA-Agent repository root:\n\n"
        "```bash\nPYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -B "
        "scripts/build_eva_hf_release_v1.py build\n```\n"
    )
    card_files.extend(
        [
            _write_text(output / "MIGRATION.md", migration_markdown),
            {
                "path": "migration-report.json",
                "byte_count": (output / "migration-report.json").stat().st_size,
                "content_blake3": _stream_blake3(output / "migration-report.json"),
            },
        ]
    )

    manifest_core: dict[str, Any] = {
        "schema": SFT_MANIFEST_SCHEMA,
        "target_dataset": SFT_TARGET,
        "visibility": "private",
        "offline_builder_upload_performed": False,
        "offline_builder_repo_created": False,
        "row_schema": SFT_ROW_SCHEMA,
        "row_count": current_rows + legacy_rows,
        "split_counts": {"train": current_rows + legacy_rows},
        "verified_current_row_count": current_rows,
        "verified_current_scanned": current_scanned,
        "legacy_semantic_scanned": semantic_scanned,
        "legacy_semantic_included": legacy_rows,
        "credential_quarantine_count": len(quarantine),
        "visible_decision_summary_count": rationale_rows,
        "schedule_rows_included": 0,
        "provider_private_reasoning_included": False,
        "semantic_rows_unique": True,
        "target_tool_call_count": target_tool_calls,
        "offered_tool_count": offered_tools,
        "parallel_tool_calls_supported": True,
        "verifier_bundle": verifier_bundle,
        "card_files": card_files,
        "counts_by_quality_tier": dict(sorted(tier_counts.items())),
        "counts_by_stage": dict(sorted(stage_counts.items())),
        "semantic_duplicate_skip_count": sum(duplicate_skips.values()),
        "semantic_duplicate_skips": [
            {
                "source": source,
                "quality_tier": tier,
                "reason": "lower_priority_semantic_duplicate",
                "count": count,
            }
            for (source, tier), count in sorted(duplicate_skips.items())
        ],
        "shards": descriptors,
    }
    manifest = {**manifest_core, "manifest_blake3": blake3_hex(manifest_core)}
    _write_json(output / "manifest.json", manifest)
    migration = {
        "legacy_release": release,
        "legacy_semantic_scanned": semantic_scanned,
        "legacy_semantic_included": legacy_rows,
        "credential_quarantine": quarantine,
        "legacy_schedule_rows": schedule_rows,
        "legacy_repo": LEGACY_SFT_REPO,
        "legacy_revision": LEGACY_SFT_REVISION,
        "included_by_dataset": dict(sorted(included_by_dataset.items())),
        "semantic_duplicate_skips": manifest_core["semantic_duplicate_skips"],
    }
    return manifest, migration


def _verify_rl_source(
    root: Path, *, trust_store_path: Path, expected_rows: int
) -> tuple[
    Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]]
]:
    envelope_document = _read_json(root / "manifest.json")
    envelope = SignedEnvelope.from_document(envelope_document)
    verify_signed_envelope(envelope, trust_store_path=Path(trust_store_path))
    payload = canonical_value(envelope.payload)
    if (
        payload.get("schema") != "eva.bulk-rl-sandbox-sharded-dataset.v1"
        or payload.get("sandbox_count") != expected_rows
        or payload.get("split_counts") != {"train": expected_rows}
    ):
        raise EVAHFReleaseError("RL source manifest contract differs")
    descriptors = payload.get("shards")
    if not isinstance(descriptors, list) or not descriptors:
        raise EVAHFReleaseError("RL source shards differ")
    seen: set[str] = set()
    rows: list[Mapping[str, Any]] = []
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise EVAHFReleaseError("RL source descriptor differs")
        shard = _safe_relative(root, descriptor.get("path"), label="RL source shard")
        if descriptor.get("file_blake3") != _stream_blake3(shard):
            raise EVAHFReleaseError("RL source shard BLAKE3 differs")
        count = 0
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line, object_pairs_hook=_strict_object)
                if not isinstance(row, Mapping):
                    raise EVAHFReleaseError("RL source row differs")
                core = {key: value for key, value in row.items() if key != "record_blake3"}
                if row.get("record_blake3") != blake3_hex(core):
                    raise EVAHFReleaseError("RL source row commitment differs")
                sandbox_id = _uuid(row.get("sandbox_id"), label="RL sandbox_id")
                if sandbox_id in seen:
                    raise EVAHFReleaseError("RL sandbox_id is duplicated")
                if row.get("split") != "train":
                    raise EVAHFReleaseError("RL source contains a non-train row")
                if _credential_rule(row) is not None:
                    raise EVAHFReleaseError("RL source contains a credential")
                seen.add(sandbox_id)
                rows.append(row)
                count += 1
        if descriptor.get("record_count") != count:
            raise EVAHFReleaseError("RL source shard count differs")
    if len(rows) != expected_rows:
        raise EVAHFReleaseError("RL source total differs")
    return envelope_document, list(descriptors), rows


def _portable_authority_members(
    rows: Sequence[Mapping[str, Any]],
) -> set[str]:
    members: set[str] = set()
    for row in rows:
        binding = row.get("source_binding")
        if not isinstance(binding, Mapping):
            raise EVAHFReleaseError("RL source binding differs")
        relative = binding.get("authority_relative_path")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise EVAHFReleaseError("RL authority path differs")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise EVAHFReleaseError("RL authority path is unsafe")
        members.add((PurePosixPath("rlevo-med-research") / pure).as_posix())
    return members


def _verify_portable_evidence(
    root: Path,
    *,
    expected_rows: int,
    required_authority_members: set[str],
) -> Mapping[str, Any]:
    """Reopen the complete compressed evidence overlay without extracting it.

    The overlay's own builder performs a deeper semantic reconstruction.  This
    consumer independently checks every indexed archive member and proves that
    every released RL authority path is present before claiming portability.
    """

    root = Path(root)
    document = _read_json(root / "index.json")
    core = {key: value for key, value in document.items() if key != "index_blake3"}
    dataset = document.get("dataset")
    controls = document.get("controls")
    descriptors = document.get("archives")
    indexed_files = document.get("files")
    if (
        document.get("schema") != PORTABLE_EVIDENCE_SCHEMA
        or document.get("index_blake3") != blake3_hex(core)
        or document.get("file_count") != PORTABLE_EVIDENCE_FILE_COUNT
        or document.get("archive_count") != PORTABLE_EVIDENCE_ARCHIVE_COUNT
        or not isinstance(dataset, Mapping)
        or dataset.get("sandbox_count") != expected_rows
        or dataset.get("split_counts") != {"train": expected_rows}
        or dataset.get("construction_input_count") != expected_rows
        or dataset.get("unique_portable_member_count")
        != PORTABLE_EVIDENCE_FILE_COUNT
        or not isinstance(controls, Mapping)
        or controls.get("provider_calls") != 0
        or controls.get("uploads") != 0
        or controls.get("private_keys_included") is not False
        or controls.get("environment_files_included") is not False
        or controls.get("credential_pattern_findings") != 0
        or controls.get("new_content_id_digest") != "BLAKE3-256"
        or controls.get("release_visibility_required") != "private"
        or not isinstance(descriptors, list)
        or not isinstance(indexed_files, list)
    ):
        raise EVAHFReleaseError("portable RL evidence index differs")

    expected_by_archive: dict[str, dict[str, Mapping[str, Any]]] = {}
    indexed_member_names: set[str] = set()
    for row in indexed_files:
        if not isinstance(row, Mapping):
            raise EVAHFReleaseError("portable RL evidence file row differs")
        archive = row.get("archive")
        member = row.get("path")
        if not isinstance(archive, str) or not isinstance(member, str):
            raise EVAHFReleaseError("portable RL evidence file path differs")
        pure = PurePosixPath(member)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise EVAHFReleaseError("portable RL evidence member is unsafe")
        if member in indexed_member_names:
            raise EVAHFReleaseError("portable RL evidence member is duplicated")
        if not isinstance(row.get("byte_count"), int) or not is_blake3(
            row.get("file_blake3")
        ):
            raise EVAHFReleaseError("portable RL evidence file commitment differs")
        indexed_member_names.add(member)
        expected_by_archive.setdefault(archive, {})[member] = row
    if (
        len(indexed_member_names) != PORTABLE_EVIDENCE_FILE_COUNT
        or not required_authority_members.issubset(indexed_member_names)
    ):
        raise EVAHFReleaseError("portable RL evidence authority coverage differs")

    seen_archives: set[str] = set()
    seen_members: set[str] = set()
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise EVAHFReleaseError("portable RL evidence archive row differs")
        archive_relative = descriptor.get("path")
        if not isinstance(archive_relative, str) or archive_relative in seen_archives:
            raise EVAHFReleaseError("portable RL evidence archive path differs")
        descriptor_core = {
            key: value for key, value in descriptor.items() if key != "descriptor_blake3"
        }
        archive_path = _safe_relative(
            root, archive_relative, label="portable RL evidence archive"
        )
        if (
            descriptor.get("descriptor_blake3") != blake3_hex(descriptor_core)
            or descriptor.get("archive_byte_count") != archive_path.stat().st_size
            or descriptor.get("archive_blake3") != _stream_blake3(archive_path)
        ):
            raise EVAHFReleaseError("portable RL evidence archive commitment differs")
        expected = expected_by_archive.get(archive_relative)
        if expected is None:
            raise EVAHFReleaseError("portable RL evidence archive is unindexed")
        observed: list[Mapping[str, Any]] = []
        process = subprocess.Popen(
            ["zstd", "-q", "-dc", str(archive_path)], stdout=subprocess.PIPE
        )
        if process.stdout is None:
            raise EVAHFReleaseError("portable RL evidence decoder is unavailable")
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive_handle:
                for member in archive_handle:
                    pure = PurePosixPath(member.name)
                    if (
                        not member.isreg()
                        or pure.is_absolute()
                        or any(part in {"", ".", ".."} for part in pure.parts)
                        or member.name in seen_members
                        or member.mode != 0o444
                        or member.uid != 0
                        or member.gid != 0
                        or member.mtime != 0
                    ):
                        raise EVAHFReleaseError(
                            "portable RL evidence tar topology differs"
                        )
                    expected_row = expected.get(member.name)
                    source = archive_handle.extractfile(member)
                    if expected_row is None or source is None:
                        raise EVAHFReleaseError(
                            "portable RL evidence tar inventory differs"
                        )
                    payload = source.read()
                    if (
                        len(payload) != expected_row.get("byte_count")
                        or blake3_bytes(payload) != expected_row.get("file_blake3")
                        or _credential_bytes_rule(payload) is not None
                    ):
                        raise EVAHFReleaseError(
                            "portable RL evidence tar payload differs"
                        )
                    observed.append(
                        {
                            "path": member.name,
                            "byte_count": len(payload),
                            "file_blake3": blake3_bytes(payload),
                        }
                    )
                    seen_members.add(member.name)
            process.stdout.close()
            if process.wait() != 0:
                raise EVAHFReleaseError("portable RL evidence decoder failed")
        except BaseException:
            process.kill()
            process.wait()
            raise
        observed.sort(key=lambda row: str(row["path"]))
        expected_core = [
            {
                "path": row["path"],
                "byte_count": row["byte_count"],
                "file_blake3": row["file_blake3"],
            }
            for row in sorted(expected.values(), key=lambda row: str(row["path"]))
        ]
        if (
            observed != expected_core
            or descriptor.get("file_count") != len(observed)
            or descriptor.get("uncompressed_byte_count")
            != sum(int(row["byte_count"]) for row in observed)
            or descriptor.get("members_root_blake3") != blake3_hex(observed)
        ):
            raise EVAHFReleaseError("portable RL evidence archive members differ")
        seen_archives.add(archive_relative)
    if (
        len(seen_archives) != PORTABLE_EVIDENCE_ARCHIVE_COUNT
        or seen_members != indexed_member_names
    ):
        raise EVAHFReleaseError("portable RL evidence inventory differs")
    return document


def _copy_portable_evidence(
    *,
    source_root: Path,
    output_root: Path,
    expected_rows: int,
    required_authority_members: set[str],
) -> Mapping[str, Any]:
    index = _verify_portable_evidence(
        source_root,
        expected_rows=expected_rows,
        required_authority_members=required_authority_members,
    )
    output_root.mkdir()
    archive_output = output_root / "archives"
    archive_output.mkdir()
    copied_archives: list[Mapping[str, Any]] = []
    for descriptor in index["archives"]:
        source = _safe_relative(
            source_root, descriptor["path"], label="portable RL evidence archive"
        )
        target = archive_output / Path(str(descriptor["path"])).name
        with source.open("rb") as input_handle, target.open("xb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if (
            target.stat().st_size != descriptor["archive_byte_count"]
            or _stream_blake3(target) != descriptor["archive_blake3"]
        ):
            raise EVAHFReleaseError("portable RL evidence copy differs")
        copied_archives.append(
            {
                "path": f"portable-evidence/{descriptor['path']}",
                "byte_count": target.stat().st_size,
                "content_blake3": descriptor["archive_blake3"],
            }
        )
    index_target = output_root / "index.json"
    with (Path(source_root) / "index.json").open("rb") as input_handle, index_target.open(
        "xb"
    ) as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    return {
        "schema": PORTABLE_EVIDENCE_SCHEMA,
        "path": "portable-evidence/index.json",
        "byte_count": index_target.stat().st_size,
        "content_blake3": _stream_blake3(index_target),
        "index_blake3": index["index_blake3"],
        "file_count": index["file_count"],
        "archive_count": index["archive_count"],
        "archive_byte_count": index["archive_byte_count"],
        "uncompressed_byte_count": index["uncompressed_byte_count"],
        "archives": copied_archives,
    }


def _write_rl_release(
    output: Path,
    *,
    source_root: Path,
    portable_evidence_root: Path,
    trust_store_path: Path,
    expected_rows: int,
) -> Mapping[str, Any]:
    envelope, descriptors, source_rows = _verify_rl_source(
        source_root, trust_store_path=trust_store_path, expected_rows=expected_rows
    )
    data_root = output / "data"
    data_root.mkdir(parents=True)
    copied: list[Mapping[str, Any]] = []
    for ordinal, descriptor in enumerate(descriptors):
        source = _safe_relative(
            source_root, descriptor.get("path"), label="RL direct-reuse shard"
        )
        name = f"train-{ordinal:05d}.jsonl"
        target = data_root / name
        with source.open("rb") as input_handle, target.open("xb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        content_blake3 = _stream_blake3(target)
        if content_blake3 != descriptor.get("file_blake3"):
            raise EVAHFReleaseError("RL direct-reuse bytes differ")
        copied.append(
            {
                "path": f"data/{name}",
                "row_count": descriptor.get("record_count"),
                "byte_count": target.stat().st_size,
                "content_blake3": content_blake3,
                "source_path": descriptor.get("path"),
            }
        )
    _write_json(output / "source-signed-manifest.json", envelope)
    portable = _copy_portable_evidence(
        source_root=Path(portable_evidence_root),
        output_root=output / "portable-evidence",
        expected_rows=expected_rows,
        required_authority_members=_portable_authority_members(source_rows),
    )
    card_files = _card_files(
        output, target=RL_TARGET, row_count=expected_rows, kind="rl"
    )
    verifier_bundle = _copy_verifier_bundle(output, dataset_kind="rl")
    local_migration = {
        "schema": "eva.private-hf-dataset-migration-report.v1",
        "target_dataset": RL_TARGET,
        "current_source_status": "direct_reuse",
        "current_source_envelope_blake3": envelope.get("envelope_blake3"),
        "legacy_repo": LEGACY_RL_REPO,
        "legacy_revision": LEGACY_RL_REVISION,
        "legacy_source_status": "reject",
        "legacy_reject_reason": "historical_controls_and_results_not_rl_sandbox_rows",
        "source_artifacts_self_contained": True,
        "portable_evidence_index_blake3": portable["index_blake3"],
        "reproducibility_command": (
            "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -B "
            "scripts/build_eva_hf_release_v1.py build"
        ),
    }
    _write_json(output / "migration-report.json", local_migration)
    migration_markdown = (
        "# Migration and reproducibility\n\n"
        "The signed current 6,000-row RL dataset is copied byte-for-byte. "
        f"The legacy snapshot `{LEGACY_RL_REPO}@{LEGACY_RL_REVISION}` is rejected "
        "because it contains controls/results rather than RL sandbox rows.\n\n"
        "The private release includes a verified compressed portable-evidence overlay. "
        "It preserves every authority-relative source artifact needed by the 6,000 "
        "rows without expanding 30,030 files in the Hub checkout.\n\n"
        "```bash\nPYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -B "
        "scripts/build_eva_hf_release_v1.py build\n```\n"
    )
    card_files.extend(
        [
            _write_text(output / "MIGRATION.md", migration_markdown),
            {
                "path": "migration-report.json",
                "byte_count": (output / "migration-report.json").stat().st_size,
                "content_blake3": _stream_blake3(output / "migration-report.json"),
            },
        ]
    )
    core: dict[str, Any] = {
        "schema": RL_MANIFEST_SCHEMA,
        "target_dataset": RL_TARGET,
        "visibility": "private",
        "offline_builder_upload_performed": False,
        "offline_builder_repo_created": False,
        "migration_status": "direct_reuse",
        "row_count": expected_rows,
        "split_counts": {"train": expected_rows},
        "source_envelope_blake3": envelope.get("envelope_blake3"),
        "source_payload_blake3": envelope.get("payload_blake3"),
        "source_bytes_reencoded": False,
        "source_artifacts_self_contained": True,
        "portable_evidence": portable,
        "verifier_bundle": verifier_bundle,
        "card_files": card_files,
        "shards": copied,
    }
    manifest = {**core, "manifest_blake3": blake3_hex(core)}
    _write_json(output / "manifest.json", manifest)
    return manifest


def _legacy_rl_audit(root: Path) -> Mapping[str, Any]:
    version = _read_json(root / "VERSION.json")
    files = [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()]
    return {
        "status": "reject",
        "included": False,
        "reason": "historical_controls_and_results_not_rl_sandbox_rows",
        "version_schema": version.get("schema"),
        "file_count": len(files),
    }


def build_eva_hf_release(
    *,
    output_root: Path,
    sft_roots: Sequence[Path],
    rl_root: Path,
    rl_portable_evidence_root: Path,
    legacy_audit_root: Path,
    trust_store_path: Path,
    expected_current_sft_rows: int = 1_282,
    expected_legacy_semantic_rows: int = 20_000,
    expected_legacy_schedule_rows: int = 25_600,
    expected_credential_quarantine_rows: int | None = 0,
    expected_rl_rows: int = 6_000,
    sft_shard_size: int = 5_000,
) -> EVAHFReleaseBuild:
    """Build a deterministic local release bundle without any remote action."""

    output_root = Path(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise EVAHFReleaseError("release output already exists")
    if sft_shard_size < 1:
        raise EVAHFReleaseError("SFT shard size differs")
    verified = tuple(_verify_sft_root(Path(root)) for root in sft_roots)
    if len({source.dataset_id for source in verified}) != len(verified):
        raise EVAHFReleaseError("SFT source dataset is duplicated")
    legacy_sft_root = Path(legacy_audit_root) / "rlevo-Med-SFT-data"
    legacy_rl_root = Path(legacy_audit_root) / "rlevo-Med-RL-data"
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        sft_output = temporary / "EVA-Med-SFT-data"
        rl_output = temporary / "EVA-Med-RL-data"
        sft_output.mkdir()
        rl_output.mkdir()
        sft_manifest, legacy_sft = _write_sft_release(
            sft_output,
            verified_roots=verified,
            legacy_root=legacy_sft_root,
            expected_current_rows=expected_current_sft_rows,
            expected_legacy_semantic_rows=expected_legacy_semantic_rows,
            expected_legacy_schedule_rows=expected_legacy_schedule_rows,
            expected_credential_quarantine_rows=expected_credential_quarantine_rows,
            shard_size=sft_shard_size,
        )
        rl_manifest = _write_rl_release(
            rl_output,
            source_root=Path(rl_root),
            portable_evidence_root=Path(rl_portable_evidence_root),
            trust_store_path=Path(trust_store_path),
            expected_rows=expected_rl_rows,
        )
        migration_core: dict[str, Any] = {
            "schema": MIGRATION_SCHEMA,
            "entries": [
                *[
                    {
                        "source_dataset_id": source.dataset_id,
                        "source_schema": source.schema,
                        "status": "upgrade",
                        "included": True,
                        "row_count": len(source.rows),
                        "quality_tier": source.tier,
                        "reason": "qwen_message_projection_with_observable_target_boundary",
                    }
                    for source in sorted(
                        verified,
                        key=lambda item: (
                            {
                                "strict_full_trajectory": 0,
                                "execution_verified_s2_frontier": 1,
                                "execution_verified_s1_frontier": 2,
                            }[item.tier],
                            item.dataset_id,
                        ),
                    )
                ],
                {
                    "source_release": legacy_sft["legacy_release"],
                    "source_repo": legacy_sft["legacy_repo"],
                    "source_revision": legacy_sft["legacy_revision"],
                    "source_kind": "legacy_semantic",
                    "status": "upgrade",
                    "included": True,
                    "scanned_rows": legacy_sft["legacy_semantic_scanned"],
                    "included_rows": legacy_sft["legacy_semantic_included"],
                    "credential_quarantine": legacy_sft["credential_quarantine"],
                    "semantic_duplicate_skips": legacy_sft["semantic_duplicate_skips"],
                    "reason": "semantic_targets_projected_once_to_qwen_messages",
                },
                {
                    "source_release": legacy_sft["legacy_release"],
                    "source_repo": legacy_sft["legacy_repo"],
                    "source_revision": legacy_sft["legacy_revision"],
                    "source_kind": "legacy_schedule",
                    "status": "reject",
                    "included": False,
                    "row_count": legacy_sft["legacy_schedule_rows"],
                    "reason": "schedule_rows_duplicate_semantic_training_content",
                },
                {
                    "source_kind": "current_rl_6000",
                    "status": "direct_reuse",
                    "included": True,
                    "row_count": expected_rl_rows,
                    "source_envelope_blake3": rl_manifest["source_envelope_blake3"],
                    "reason": "signed_exact_train_sandboxes_copied_byte_for_byte",
                },
                {
                    "source_kind": "legacy_rl_snapshot",
                    "source_repo": LEGACY_RL_REPO,
                    "source_revision": LEGACY_RL_REVISION,
                    **_legacy_rl_audit(legacy_rl_root),
                },
            ],
            "allowed_statuses": ["direct_reuse", "upgrade", "reject"],
        }
        migration = {
            **migration_core,
            "audit_blake3": blake3_hex(migration_core),
        }
        _write_json(temporary / "migration-audit.json", migration)
        release_core: dict[str, Any] = {
            "schema": RELEASE_SCHEMA,
            "visibility": "private",
            "targets": [SFT_TARGET, RL_TARGET],
            "offline_builder_upload_performed": False,
            "offline_builder_repo_created": False,
            "generated_data_git_policy": "ignored",
            "provider_calls": 0,
            "sft": {
                "path": "EVA-Med-SFT-data",
                "row_count": sft_manifest["row_count"],
                "manifest_blake3": sft_manifest["manifest_blake3"],
            },
            "rl": {
                "path": "EVA-Med-RL-data",
                "row_count": rl_manifest["row_count"],
                "manifest_blake3": rl_manifest["manifest_blake3"],
            },
            "migration_audit_blake3": migration["audit_blake3"],
        }
        release = {**release_core, "release_blake3": blake3_hex(release_core)}
        _write_json(temporary / "release-manifest.json", release)
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verification = verify_eva_hf_release(
        output_root=output_root,
        expected_sft_rows=int(sft_manifest["row_count"]),
        expected_rl_rows=expected_rl_rows,
    )
    return EVAHFReleaseBuild(
        output_root=output_root,
        sft_rows=int(verification["sft_rows"]),
        rl_rows=int(verification["rl_rows"]),
        rationale_rows=int(verification["rationale_rows"]),
        credential_quarantine_rows=int(verification["credential_quarantine_rows"]),
        release_blake3=str(verification["release_blake3"]),
    )


def _verify_manifest(path: Path, *, digest_key: str) -> Mapping[str, Any]:
    value = _read_json(path)
    core = {key: item for key, item in value.items() if key != digest_key}
    if value.get(digest_key) != blake3_hex(core):
        raise EVAHFReleaseError(f"{path.name} BLAKE3 differs")
    return value


def _verify_qwen_row(row: Mapping[str, Any]) -> None:
    if row.get("schema") != SFT_ROW_SCHEMA:
        raise EVAHFReleaseError("release SFT schema differs")
    _uuid(row.get("row_id"), label="release SFT row_id")
    messages = row.get("messages")
    loss = row.get("loss_contract")
    if (
        not isinstance(messages, list)
        or not messages
        or not isinstance(loss, Mapping)
        or loss.get("schema") != LOSS_SCHEMA
        or loss.get("assistant_message_index") != len(messages) - 1
        or loss.get("supervise_exactly_one_assistant_message") is not True
        or loss.get("subsequent_observations_in_training_messages") is not False
        or messages[-1].get("role") != "assistant"
    ):
        raise EVAHFReleaseError("release SFT loss boundary differs")
    if row.get("parallel_tool_calls_supported") is not True:
        raise EVAHFReleaseError("release SFT parallel tool-call capability differs")
    tools = _normalise_tools(row.get("tools"))
    offered_names = {str(tool["function"]["name"]) for tool in tools}
    catalog_scope = row.get("metadata", {}).get("tool_catalog_scope")
    pending: set[str] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise EVAHFReleaseError("release SFT message differs")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise EVAHFReleaseError("release SFT role differs")
        if role == "assistant":
            for call in message.get("tool_calls", []):
                normalised = _tool_call(call, fallback_id="invalid")
                call_id = str(normalised["id"])
                if call_id in pending:
                    raise EVAHFReleaseError("release SFT tool-call id is duplicated")
                pending.add(call_id)
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise EVAHFReleaseError("release SFT tool join differs")
            pending.remove(str(call_id))
        if index < len(messages) - 1 and role == "assistant" and not message.get("tool_calls"):
            # Plain assistant history is permitted, but is never another loss
            # target because the explicit loss contract selects only the end.
            pass
    target_ids = loss.get("target_tool_call_ids")
    expected_target_ids = [
        str(call["id"]) for call in messages[-1].get("tool_calls", [])
    ]
    if target_ids != expected_target_ids or pending != set(expected_target_ids):
        raise EVAHFReleaseError("release SFT target tool-call boundary differs")
    target_names = {
        str(call["function"]["name"])
        for call in messages[-1].get("tool_calls", [])
    }
    if catalog_scope == "unavailable_legacy_verified":
        if tools:
            raise EVAHFReleaseError("unavailable legacy tool catalog was invented")
    elif not target_names.issubset(offered_names):
        raise EVAHFReleaseError("target tool is absent from offered tools")
    decision = row.get("decision_summary")
    if decision is not None:
        if (
            not isinstance(decision, Mapping)
            or decision.get("format") != "think"
            or decision.get("verbatim_visible") is not True
            or decision.get("provider_private_cot") is not False
            or not isinstance(decision.get("text"), str)
            or not decision["text"].startswith("<think>")
            or not decision["text"].endswith("</think>")
            or messages[-1].get("content") != decision["text"]
        ):
            raise EVAHFReleaseError("release SFT decision summary differs")
    if _credential_rule(row) is not None:
        raise EVAHFReleaseError("release SFT row contains a credential")


def _verify_dataset_support_files(
    dataset_root: Path, manifest: Mapping[str, Any]
) -> None:
    if (
        manifest.get("offline_builder_upload_performed") is not False
        or manifest.get("offline_builder_repo_created") is not False
    ):
        raise EVAHFReleaseError("offline builder status differs")
    card_paths: set[str] = set()
    for descriptor in manifest.get("card_files", []):
        card = _safe_relative(
            dataset_root, descriptor.get("path"), label="dataset card file"
        )
        if (
            descriptor.get("content_blake3") != _stream_blake3(card)
            or descriptor.get("byte_count") != card.stat().st_size
        ):
            raise EVAHFReleaseError("dataset card commitment differs")
        card_paths.add(str(descriptor.get("path")))
    if card_paths != {"README.md", "VERSION.json", "MIGRATION.md", "migration-report.json"}:
        raise EVAHFReleaseError("dataset card inventory differs")
    _verify_verifier_bundle(dataset_root, manifest.get("verifier_bundle"))


def verify_eva_hf_sft_dataset(
    *, dataset_root: Path, expected_rows: int | None = None
) -> Mapping[str, Any]:
    """Verify a downloaded EVA-Med-SFT-data repository in isolation."""

    dataset_root = Path(dataset_root)
    manifest = _verify_manifest(dataset_root / "manifest.json", digest_key="manifest_blake3")
    if manifest.get("schema") != SFT_MANIFEST_SCHEMA:
        raise EVAHFReleaseError("release SFT manifest schema differs")
    _verify_dataset_support_files(dataset_root, manifest)
    seen_ids: set[str] = set()
    seen_semantics: set[str] = set()
    rows = 0
    rationale_rows = 0
    target_tool_calls = 0
    offered_tools = 0
    tier_counts: Counter[str] = Counter()
    for descriptor in manifest.get("shards", []):
        shard = _safe_relative(dataset_root, descriptor.get("path"), label="release SFT shard")
        if (
            descriptor.get("content_blake3") != _stream_blake3(shard)
            or descriptor.get("byte_count") != shard.stat().st_size
        ):
            raise EVAHFReleaseError("release SFT shard differs")
        count = 0
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line, object_pairs_hook=_strict_object)
                if not isinstance(row, Mapping):
                    raise EVAHFReleaseError("release SFT row differs")
                _verify_qwen_row(row)
                row_id = str(row["row_id"])
                semantic = _semantic_key(row)
                if row_id in seen_ids or semantic in seen_semantics:
                    raise EVAHFReleaseError("release SFT row is duplicated")
                seen_ids.add(row_id)
                seen_semantics.add(semantic)
                rationale_rows += "decision_summary" in row
                target_tool_calls += len(row["loss_contract"]["target_tool_call_ids"])
                offered_tools += len(row.get("tools", []))
                tier_counts[str(row["metadata"]["quality_tier"])] += 1
                count += 1
                rows += 1
        if descriptor.get("row_count") != count:
            raise EVAHFReleaseError("release SFT shard count differs")
    if (
        (expected_rows is not None and rows != expected_rows)
        or manifest.get("row_count") != rows
        or manifest.get("split_counts") != {"train": rows}
        or manifest.get("schedule_rows_included") != 0
        or manifest.get("provider_private_reasoning_included") is not False
        or manifest.get("semantic_rows_unique") is not True
        or manifest.get("visible_decision_summary_count") != rationale_rows
        or manifest.get("counts_by_quality_tier") != dict(sorted(tier_counts.items()))
        or manifest.get("target_tool_call_count") != target_tool_calls
        or manifest.get("offered_tool_count") != offered_tools
        or manifest.get("parallel_tool_calls_supported") is not True
    ):
        raise EVAHFReleaseError("release SFT totals differ")
    return {
        "schema": "eva.private-hf-dataset-verification.v1",
        "valid": True,
        "dataset_kind": "sft",
        "rows": rows,
        "rationale_rows": rationale_rows,
        "credential_quarantine_rows": manifest.get("credential_quarantine_count"),
        "manifest_blake3": manifest.get("manifest_blake3"),
        "verifier_bundle_blake3": manifest.get("verifier_bundle", {}).get("bundle_blake3"),
    }


def verify_eva_hf_rl_dataset(
    *, dataset_root: Path, expected_rows: int = 6_000
) -> Mapping[str, Any]:
    """Verify a downloaded EVA-Med-RL-data repository in isolation."""

    dataset_root = Path(dataset_root)
    manifest = _verify_manifest(dataset_root / "manifest.json", digest_key="manifest_blake3")
    if manifest.get("schema") != RL_MANIFEST_SCHEMA:
        raise EVAHFReleaseError("release RL manifest schema differs")
    _verify_dataset_support_files(dataset_root, manifest)
    rows = 0
    released_rows: list[Mapping[str, Any]] = []
    for descriptor in manifest.get("shards", []):
        shard = _safe_relative(dataset_root, descriptor.get("path"), label="release RL shard")
        if (
            descriptor.get("content_blake3") != _stream_blake3(shard)
            or descriptor.get("byte_count") != shard.stat().st_size
        ):
            raise EVAHFReleaseError("release RL shard differs")
        count = 0
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line, object_pairs_hook=_strict_object)
                if not isinstance(row, Mapping):
                    raise EVAHFReleaseError("release RL row differs")
                core = {key: value for key, value in row.items() if key != "record_blake3"}
                if (
                    row.get("record_blake3") != blake3_hex(core)
                    or row.get("split") != "train"
                    or _credential_rule(row) is not None
                ):
                    raise EVAHFReleaseError("release RL row contract differs")
                released_rows.append(row)
                count += 1
                rows += 1
        if descriptor.get("row_count") != count:
            raise EVAHFReleaseError("release RL shard count differs")
    if (
        rows != expected_rows
        or manifest.get("row_count") != rows
        or manifest.get("split_counts") != {"train": rows}
        or manifest.get("migration_status") != "direct_reuse"
        or manifest.get("source_bytes_reencoded") is not False
        or manifest.get("source_artifacts_self_contained") is not True
    ):
        raise EVAHFReleaseError("release RL totals differ")
    portable = manifest.get("portable_evidence")
    if not isinstance(portable, Mapping):
        raise EVAHFReleaseError("portable RL evidence descriptor differs")
    portable_root = dataset_root / "portable-evidence"
    portable_index = _safe_relative(
        dataset_root, portable.get("path"), label="portable RL evidence index"
    )
    if (
        portable.get("schema") != PORTABLE_EVIDENCE_SCHEMA
        or portable.get("byte_count") != portable_index.stat().st_size
        or portable.get("content_blake3") != _stream_blake3(portable_index)
    ):
        raise EVAHFReleaseError("portable RL evidence copy descriptor differs")
    reopened = _verify_portable_evidence(
        portable_root,
        expected_rows=expected_rows,
        required_authority_members=_portable_authority_members(released_rows),
    )
    if (
        portable.get("index_blake3") != reopened.get("index_blake3")
        or portable.get("file_count") != reopened.get("file_count")
        or portable.get("archive_count") != reopened.get("archive_count")
    ):
        raise EVAHFReleaseError("portable RL evidence cross-commitment differs")
    return {
        "schema": "eva.private-hf-dataset-verification.v1",
        "valid": True,
        "dataset_kind": "rl",
        "rows": rows,
        "portable_evidence_files": reopened.get("file_count"),
        "portable_evidence_archives": reopened.get("archive_count"),
        "manifest_blake3": manifest.get("manifest_blake3"),
        "verifier_bundle_blake3": manifest.get("verifier_bundle", {}).get("bundle_blake3"),
    }


def verify_eva_hf_release(
    *,
    output_root: Path,
    expected_sft_rows: int | None = 21_282,
    expected_rl_rows: int = 6_000,
) -> Mapping[str, Any]:
    """Independently reopen all release bytes and validate their boundaries."""

    output_root = Path(output_root)
    release = _verify_manifest(output_root / "release-manifest.json", digest_key="release_blake3")
    migration = _verify_manifest(output_root / "migration-audit.json", digest_key="audit_blake3")
    sft = _verify_manifest(
        output_root / "EVA-Med-SFT-data" / "manifest.json",
        digest_key="manifest_blake3",
    )
    rl = _verify_manifest(
        output_root / "EVA-Med-RL-data" / "manifest.json",
        digest_key="manifest_blake3",
    )
    if (
        release.get("schema") != RELEASE_SCHEMA
        or release.get("offline_builder_upload_performed") is not False
        or release.get("offline_builder_repo_created") is not False
        or release.get("provider_calls") != 0
        or release.get("migration_audit_blake3") != migration.get("audit_blake3")
        or release.get("sft", {}).get("manifest_blake3") != sft.get("manifest_blake3")
        or release.get("rl", {}).get("manifest_blake3") != rl.get("manifest_blake3")
    ):
        raise EVAHFReleaseError("release cross-commitment differs")
    for dataset_root, manifest in (
        (output_root / "EVA-Med-SFT-data", sft),
        (output_root / "EVA-Med-RL-data", rl),
    ):
        if (
            manifest.get("offline_builder_upload_performed") is not False
            or manifest.get("offline_builder_repo_created") is not False
        ):
            raise EVAHFReleaseError("offline builder status differs")
        card_paths: set[str] = set()
        for descriptor in manifest.get("card_files", []):
            card = _safe_relative(
                dataset_root, descriptor.get("path"), label="dataset card file"
            )
            if (
                descriptor.get("content_blake3") != _stream_blake3(card)
                or descriptor.get("byte_count") != card.stat().st_size
            ):
                raise EVAHFReleaseError("dataset card commitment differs")
            card_paths.add(str(descriptor.get("path")))
        if card_paths != {"README.md", "VERSION.json", "MIGRATION.md", "migration-report.json"}:
            raise EVAHFReleaseError("dataset card inventory differs")
        _verify_verifier_bundle(dataset_root, manifest.get("verifier_bundle"))

    seen_ids: set[str] = set()
    seen_semantics: set[str] = set()
    sft_rows = 0
    rationale_rows = 0
    target_tool_calls = 0
    offered_tools = 0
    tier_counts: Counter[str] = Counter()
    for descriptor in sft.get("shards", []):
        shard = _safe_relative(
            output_root / "EVA-Med-SFT-data",
            descriptor.get("path"),
            label="release SFT shard",
        )
        if (
            descriptor.get("content_blake3") != _stream_blake3(shard)
            or descriptor.get("byte_count") != shard.stat().st_size
        ):
            raise EVAHFReleaseError("release SFT shard differs")
        count = 0
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line, object_pairs_hook=_strict_object)
                if not isinstance(row, Mapping):
                    raise EVAHFReleaseError("release SFT row differs")
                _verify_qwen_row(row)
                row_id = str(row["row_id"])
                semantic = _semantic_key(row)
                if row_id in seen_ids or semantic in seen_semantics:
                    raise EVAHFReleaseError("release SFT row is duplicated")
                seen_ids.add(row_id)
                seen_semantics.add(semantic)
                rationale_rows += "decision_summary" in row
                target_tool_calls += len(row["loss_contract"]["target_tool_call_ids"])
                offered_tools += len(row.get("tools", []))
                tier_counts[str(row["metadata"]["quality_tier"])] += 1
                count += 1
                sft_rows += 1
        if descriptor.get("row_count") != count:
            raise EVAHFReleaseError("release SFT shard count differs")
    if (
        (expected_sft_rows is not None and sft_rows != expected_sft_rows)
        or sft.get("row_count") != sft_rows
        or sft.get("split_counts") != {"train": sft_rows}
        or sft.get("schedule_rows_included") != 0
        or sft.get("provider_private_reasoning_included") is not False
        or sft.get("semantic_rows_unique") is not True
        or sft.get("visible_decision_summary_count") != rationale_rows
        or sft.get("counts_by_quality_tier") != dict(sorted(tier_counts.items()))
        or sft.get("target_tool_call_count") != target_tool_calls
        or sft.get("offered_tool_count") != offered_tools
        or sft.get("parallel_tool_calls_supported") is not True
    ):
        raise EVAHFReleaseError("release SFT totals differ")

    rl_root = output_root / "EVA-Med-RL-data"
    rl_rows = 0
    for descriptor in rl.get("shards", []):
        shard = _safe_relative(rl_root, descriptor.get("path"), label="release RL shard")
        if (
            descriptor.get("content_blake3") != _stream_blake3(shard)
            or descriptor.get("byte_count") != shard.stat().st_size
        ):
            raise EVAHFReleaseError("release RL shard differs")
        with shard.open("r", encoding="utf-8") as handle:
            count = sum(1 for _ in handle)
        if descriptor.get("row_count") != count:
            raise EVAHFReleaseError("release RL shard count differs")
        rl_rows += count
    if (
        rl_rows != expected_rl_rows
        or rl.get("row_count") != rl_rows
        or rl.get("migration_status") != "direct_reuse"
        or rl.get("source_bytes_reencoded") is not False
        or rl.get("source_artifacts_self_contained") is not True
    ):
        raise EVAHFReleaseError("release RL totals differ")
    portable = rl.get("portable_evidence")
    if not isinstance(portable, Mapping):
        raise EVAHFReleaseError("portable RL evidence descriptor differs")
    portable_root = rl_root / "portable-evidence"
    portable_index = _safe_relative(
        rl_root, portable.get("path"), label="portable RL evidence index"
    )
    if (
        portable.get("schema") != PORTABLE_EVIDENCE_SCHEMA
        or portable.get("byte_count") != portable_index.stat().st_size
        or portable.get("content_blake3") != _stream_blake3(portable_index)
    ):
        raise EVAHFReleaseError("portable RL evidence copy descriptor differs")
    released_rl_rows: list[Mapping[str, Any]] = []
    for descriptor in rl.get("shards", []):
        shard = _safe_relative(rl_root, descriptor.get("path"), label="release RL shard")
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line, object_pairs_hook=_strict_object)
                if not isinstance(row, Mapping):
                    raise EVAHFReleaseError("release RL row differs")
                released_rl_rows.append(row)
    reopened_portable = _verify_portable_evidence(
        portable_root,
        expected_rows=expected_rl_rows,
        required_authority_members=_portable_authority_members(released_rl_rows),
    )
    if (
        portable.get("index_blake3") != reopened_portable.get("index_blake3")
        or portable.get("file_count") != reopened_portable.get("file_count")
        or portable.get("archive_count") != reopened_portable.get("archive_count")
    ):
        raise EVAHFReleaseError("portable RL evidence cross-commitment differs")
    statuses = {entry.get("status") for entry in migration.get("entries", [])}
    if statuses != {"direct_reuse", "upgrade", "reject"}:
        raise EVAHFReleaseError("migration status coverage differs")
    return {
        "schema": "eva.private-hf-release-verification.v1",
        "valid": True,
        "sft_rows": sft_rows,
        "rl_rows": rl_rows,
        "rationale_rows": rationale_rows,
        "credential_quarantine_rows": sft.get("credential_quarantine_count"),
        "release_blake3": release.get("release_blake3"),
        "provider_calls": 0,
        "upload_performed": False,
        "repo_created": False,
    }


__all__ = [
    "EVAHFReleaseBuild",
    "EVAHFReleaseError",
    "build_eva_hf_release",
    "verify_eva_hf_rl_dataset",
    "verify_eva_hf_release",
    "verify_eva_hf_sft_dataset",
]
