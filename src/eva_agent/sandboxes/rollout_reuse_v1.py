"""Sparse reuse overlay for independently reopenable legacy rollouts.

The overlay never changes the base 6,000-row dataset.  It assigns each usable
trajectory step start to at most one compatible target sandbox; every target
without a row continues to use its benchmark-native initial state.  Existing
legacy SHA-256 references are verified as upstream facts.  This module creates
only one new commitment, the BLAKE3 root covering the overlay and every source
byte it references.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from blake3 import blake3

from eva_agent.pipeline.digests import canonical_json_bytes, canonical_value
from eva_agent.pipeline.legacy_sft import (
    LEGACY_ROLLOUT_SCHEMAS,
    LegacySFTError,
    _read_json,
    _read_sha_reference,
    _sha256_value,
    _trust_keys,
    _verify_host_receipt,
    _verify_stage_references,
    verify_legacy_teacher_source,
)


OVERLAY_SCHEMA = "eva.legacy-rollout-reuse-overlay.v1"
ROW_SCHEMA = "eva.legacy-rollout-reuse-row.v1"
VERIFY_SCHEMA = "eva.legacy-rollout-reuse-verification.v1"
STAGES = ("S1", "S2", "S3", "S4", "S5", "E2E")
VALID_INTERMEDIATE_TERMINATIONS = frozenset(
    {"tool_budget_exhausted", "model_final_before_submission"}
)


class RolloutReuseError(ValueError):
    """A rollout start or sparse overlay cannot be independently reopened."""


@dataclass(frozen=True, slots=True)
class ReusableStart:
    source_receipt_path: str
    source_sandbox_id: str
    source_rollout_id: str
    source_family: str
    stage: str
    producer_model_id: str
    disposition: str
    score: float
    transcript_path: str
    prefix_message_count: int
    next_assistant_message_index: int
    workspace_paths: tuple[str, ...]
    evidence_paths: tuple[str, ...]
    qualification_paths: tuple[str, ...]


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RolloutReuseError(f"{label} path differs")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise RolloutReuseError(f"{label} path differs")
    return value


def _source_family(sandbox_id: Any) -> str | None:
    if not isinstance(sandbox_id, str) or not sandbox_id.startswith(
        "rlevo-medres-trajectory-"
    ):
        return None
    for marker, family in (
        ("healthbench-professional", "healthbench-professional"),
        ("automedbench", "automedbench"),
        ("medxpertqa", "medxpertqa"),
        ("agentclinic", "agentclinic"),
        ("hbp", "healthbench-professional"),
    ):
        if f"-{marker}-" in sandbox_id:
            return family
    return None


def _assistant_stage(message: Mapping[str, Any], previous: str) -> str:
    calls = message.get("tool_calls")
    if not isinstance(calls, (list, tuple)) or not calls:
        return previous
    first = calls[0]
    function = first.get("function") if isinstance(first, Mapping) else None
    name = function.get("name") if isinstance(function, Mapping) else None
    if name == "materialize_plan":
        return "S1"
    if name in {"retrieve_frozen_evidence", "materialize_evidence_selection"}:
        return "S2"
    if name == "submit_results":
        return "S5"
    if name == "execute_code":
        try:
            arguments = json.loads(function.get("arguments", ""))
        except (TypeError, ValueError):
            arguments = {}
        declared = arguments.get("stage") if isinstance(arguments, Mapping) else None
        if declared in {"S3", "S4"}:
            return declared
        return "S3" if previous in {"S1", "S2"} else "S4"
    return previous


def _paths_under(root: Path, relative_root: str) -> tuple[str, ...]:
    directory = root / relative_root
    if not directory.exists():
        return ()
    if directory.is_symlink() or not directory.is_dir():
        raise RolloutReuseError("workspace topology differs")
    rows: list[str] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RolloutReuseError("workspace contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RolloutReuseError("workspace contains a non-regular file")
        rows.append(path.relative_to(root).as_posix())
    return tuple(rows)


def _walk_references(value: Any) -> tuple[str, ...]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        if set(value) == {"path", "sha256"}:
            paths.append(_safe_relative(value["path"], label="stage evidence"))
        else:
            for child in value.values():
                paths.extend(_walk_references(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            paths.extend(_walk_references(child))
    return tuple(paths)


def _prefix_materials(
    receipt_root: Path, receipt: Mapping[str, Any], stage: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    stage_evidence = receipt.get("stage_evidence")
    stage_evidence = stage_evidence if isinstance(stage_evidence, Mapping) else {}
    order = {
        "S1": (),
        "E2E": (),
        "S2": ("s1_plan_contract", "s1_plan_attempts"),
        "S3": (
            "s1_plan_contract", "s1_plan_attempts", "s2_evidence_contract",
            "s2_retrieval_attempts", "s2_selection_attempts",
        ),
        "S4": (
            "s1_plan_contract", "s1_plan_attempts", "s2_evidence_contract",
            "s2_retrieval_attempts", "s2_selection_attempts", "s3_execution_attempts",
        ),
        "S5": tuple(stage_evidence),
    }
    evidence: set[str] = set()
    for key in order[stage]:
        if key in stage_evidence:
            evidence.update(_walk_references(stage_evidence[key]))
    if stage in {"S3", "S4", "S5"}:
        evidence.update(
            path.relative_to(receipt_root).as_posix()
            for path in receipt_root.glob("stages/s2/retrievals/*/retrieval-result.json")
            if path.is_file() and not path.is_symlink()
        )
    workspace_root = {
        "S2": "stages/s1-attempt-1/workspace",
        "S3": "stages/s2/workspace",
        "S5": "executions/s4-attempt-1/workspace",
    }.get(stage)
    workspace = () if workspace_root is None else _paths_under(receipt_root, workspace_root)
    return tuple(sorted(workspace)), tuple(sorted(evidence))


def _failed_source(
    receipt_path: Path,
    *,
    source_root: Path,
    trust_store_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    receipt, _ = _read_json(receipt_path)
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") not in LEGACY_ROLLOUT_SCHEMAS
        or receipt.get("status") != "failed"
        or receipt.get("termination") not in VALID_INTERMEDIATE_TERMINATIONS
        or receipt.get("infrastructure_healthy") is not True
        or receipt.get("exact_identity_all_turns") is not True
        or receipt.get("hidden_reasoning_recorded") is not False
        or receipt.get("raw_provider_response_recorded") is not False
    ):
        raise RolloutReuseError("failed rollout is not a valid intermediate source")
    turns = receipt.get("turns")
    if (
        not isinstance(turns, list)
        or not turns
        or len(turns) != receipt.get("turn_count")
        or any(row.get("identity_match") is not True for row in turns)
    ):
        raise RolloutReuseError("failed rollout turn identity differs")
    root = receipt_path.parent
    transcript, _ = _read_sha_reference(root, receipt.get("transcript"), label="transcript")
    grade, _ = _read_sha_reference(root, receipt.get("final_grade"), label="final grade")
    final_receipt, _ = _read_sha_reference(
        root, receipt.get("final_host_receipt"), label="final host receipt"
    )
    attestation, _ = _read_sha_reference(
        root, receipt.get("host_attestation"), label="rollout attestation"
    )
    keys = _trust_keys(trust_store_path)
    final_payload = _verify_host_receipt(final_receipt, keys)
    attestation_payload = _verify_host_receipt(attestation, keys)
    identity = (receipt.get("sandbox_id"), receipt.get("episode_id"))
    if (
        (final_payload.get("sandbox_id"), final_payload.get("episode_id")) != identity
        or (attestation_payload.get("sandbox_id"), attestation_payload.get("episode_id"))
        != identity
        or not isinstance(grade, Mapping)
        or grade.get("grade_sha256")
        != _sha256_value({key: value for key, value in grade.items() if key != "grade_sha256"})
        or grade.get("sandbox_id") != receipt.get("sandbox_id")
        or grade.get("episode_id") != receipt.get("episode_id")
        or not isinstance(transcript, Mapping)
        or transcript.get("rollout_id") != receipt.get("rollout_id")
        or transcript.get("sandbox_id") != receipt.get("sandbox_id")
        or transcript.get("episode_id") != receipt.get("episode_id")
        or transcript.get("model_id") != receipt.get("model_id")
    ):
        raise RolloutReuseError("failed rollout signed lineage differs")
    messages = transcript.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise RolloutReuseError("failed rollout transcript is incomplete")
    _verify_stage_references(root, receipt)
    return receipt, transcript, grade


def _starts(
    *,
    source_root: Path,
    receipt_path: Path,
    receipt: Mapping[str, Any],
    transcript: Mapping[str, Any],
    grade: Mapping[str, Any],
    disposition: str,
) -> tuple[ReusableStart, ...]:
    family = _source_family(receipt.get("sandbox_id"))
    if family is None:
        return ()
    messages = transcript.get("messages")
    if not isinstance(messages, (list, tuple)):
        return ()
    stage_indexes: dict[str, int] = {}
    previous = "S1"
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        if message.get("source") == "harness":
            continue
        previous = _assistant_stage(message, previous)
        stage_indexes[previous] = index
    first = min(stage_indexes.values(), default=-1)
    if first < 2:
        return ()
    stage_indexes["E2E"] = first
    root = receipt_path.parent
    relative_receipt = receipt_path.relative_to(source_root).as_posix()
    qualification = [relative_receipt]
    for key in ("transcript", "final_grade", "final_host_receipt", "host_attestation"):
        reference = receipt.get(key)
        if isinstance(reference, Mapping) and isinstance(reference.get("path"), str):
            qualification.append((receipt_path.parent / reference["path"]).relative_to(source_root).as_posix())
    result: list[ReusableStart] = []
    for stage in STAGES:
        index = stage_indexes.get(stage)
        if index is None:
            continue
        workspace, evidence = _prefix_materials(root, receipt, stage)
        result.append(
            ReusableStart(
                source_receipt_path=relative_receipt,
                source_sandbox_id=str(receipt["sandbox_id"]),
                source_rollout_id=str(receipt["rollout_id"]),
                source_family=family,
                stage=stage,
                producer_model_id=str(receipt["model_id"]),
                disposition=disposition,
                score=float(grade.get("normalized_score", 0.0)),
                transcript_path=(
                    receipt_path.parent / receipt["transcript"]["path"]
                ).relative_to(source_root).as_posix(),
                prefix_message_count=index,
                next_assistant_message_index=index,
                workspace_paths=tuple(
                    (receipt_path.parent / path).relative_to(source_root).as_posix()
                    for path in workspace
                ),
                evidence_paths=tuple(
                    (receipt_path.parent / path).relative_to(source_root).as_posix()
                    for path in evidence
                ),
                qualification_paths=tuple(sorted(set(qualification))),
            )
        )
    return tuple(result)


def discover_reusable_starts(
    *,
    source_root: Path,
    trust_store_path: Path,
    eligible_model_ids: Sequence[str],
    minimum_score: float = 0.8,
) -> tuple[tuple[ReusableStart, ...], Mapping[str, int]]:
    """Scan real receipts and retain high-score or valid-prefix starts."""

    source_root = source_root.resolve(strict=True)
    receipts = tuple(sorted(source_root.rglob("rollout-receipt.json")))
    counts: Counter[str] = Counter(scanned_receipts=len(receipts))
    candidates: list[ReusableStart] = []
    for path in receipts:
        try:
            preview, _ = _read_json(path)
        except Exception:
            counts["excluded_unverifiable_receipt"] += 1
            continue
        if not isinstance(preview, Mapping) or _source_family(preview.get("sandbox_id")) is None:
            counts["excluded_non_teacher_rollout"] += 1
            continue
        score_ref = preview.get("final_grade")
        score = score_ref.get("normalized_score") if isinstance(score_ref, Mapping) else None
        if preview.get("status") == "completed" and isinstance(score, (int, float)) and score >= minimum_score:
            try:
                verified = verify_legacy_teacher_source(
                    path,
                    source_root=source_root,
                    trust_store_path=trust_store_path,
                    eligible_model_ids=eligible_model_ids,
                    minimum_score=minimum_score,
                )
                starts = _starts(
                    source_root=source_root,
                    receipt_path=path,
                    receipt=verified.receipt,
                    transcript=verified.transcript,
                    grade=verified.grade,
                    disposition="high_score_success",
                )
            except (LegacySFTError, RolloutReuseError, OSError, ValueError):
                counts["excluded_unverifiable_success"] += 1
                continue
            counts["high_score_success_receipts"] += 1
        elif preview.get("status") == "failed":
            try:
                receipt, transcript, grade = _failed_source(
                    path, source_root=source_root, trust_store_path=trust_store_path
                )
                starts = _starts(
                    source_root=source_root,
                    receipt_path=path,
                    receipt=receipt,
                    transcript=transcript,
                    grade=grade,
                    disposition="valid_intermediate_failure",
                )
            except (LegacySFTError, RolloutReuseError, OSError, ValueError):
                counts["excluded_invalid_failure"] += 1
                continue
            counts["valid_intermediate_failure_receipts"] += 1
        else:
            counts["excluded_below_quality_or_incomplete"] += 1
            continue
        if not starts:
            counts["excluded_no_reopenable_step"] += 1
            continue
        candidates.extend(starts)

    model_rank = {model: rank for rank, model in enumerate(eligible_model_ids)}
    candidates.sort(
        key=lambda row: (
            0 if row.disposition == "high_score_success" else 1,
            -row.score,
            model_rank.get(row.producer_model_id, len(model_rank)),
            row.source_receipt_path,
            STAGES.index(row.stage),
        )
    )
    unique: dict[tuple[str, str], ReusableStart] = {}
    for row in candidates:
        unique.setdefault((row.source_sandbox_id, row.stage), row)
    counts["deduplicated_step_starts"] = len(unique)
    return tuple(unique.values()), dict(sorted(counts.items()))


def map_reusable_starts(
    starts: Iterable[ReusableStart], base_records: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    """Assign sparse starts once to compatible base targets."""

    pools: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in base_records:
        pools[(str(record.get("source_family")), str(record.get("stage")))].append(record)
    for rows in pools.values():
        rows.sort(key=lambda row: (int(row["queue_ordinal"]), str(row["sandbox_id"])))
    cursors: Counter[tuple[str, str]] = Counter()
    mapped: list[dict[str, Any]] = []
    for start in starts:
        key = (start.source_family, start.stage)
        cursor = cursors[key]
        if cursor >= len(pools[key]):
            continue
        target = pools[key][cursor]
        cursors[key] += 1
        mapped.append(
            {
                "schema": ROW_SCHEMA,
                "target_sandbox_id": target["sandbox_id"],
                "target_candidate_id": target["candidate_id"],
                "target_queue_ordinal": target["queue_ordinal"],
                "source_family": start.source_family,
                "stage": start.stage,
                "disposition": start.disposition,
                "score": start.score,
                "producer_model_id": start.producer_model_id,
                "trajectory_start": {
                    "source_receipt_path": start.source_receipt_path,
                    "source_sandbox_id": start.source_sandbox_id,
                    "source_rollout_id": start.source_rollout_id,
                    "transcript_path": start.transcript_path,
                    "prefix_message_count": start.prefix_message_count,
                    "next_assistant_message_index": start.next_assistant_message_index,
                    "workspace_paths": list(start.workspace_paths),
                    "evidence_paths": list(start.evidence_paths),
                },
                "qualification": {
                    "actor_visible": False,
                    "paths": list(start.qualification_paths),
                },
                "base_fallback_replaced": True,
            }
        )
    mapped.sort(key=lambda row: (int(row["target_queue_ordinal"]), row["target_sandbox_id"]))
    return tuple(mapped)


def referenced_source_paths(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    values: set[str] = set()
    for row in rows:
        start = row["trajectory_start"]
        values.add(start["source_receipt_path"])
        values.add(start["transcript_path"])
        values.update(start["workspace_paths"])
        values.update(start["evidence_paths"])
        values.update(row["qualification"]["paths"])
    return tuple(sorted(values))


def overlay_root(
    core: Mapping[str, Any],
    *,
    shard_payloads: Sequence[bytes],
    source_root: Path,
    source_paths: Sequence[str],
) -> str:
    """Create the overlay's only new digest over contract, shards, and sources."""

    digest = blake3()

    def add(label: str, payload: bytes) -> None:
        name = label.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(16, "big"))
        digest.update(payload)

    add("manifest-core", canonical_json_bytes(core))
    for ordinal, payload in enumerate(shard_payloads):
        add(f"shard/{ordinal:05d}", payload)
    root = source_root.resolve(strict=True)
    for relative in source_paths:
        safe = _safe_relative(relative, label="source material")
        path = root.joinpath(*PurePosixPath(safe).parts)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            raise RolloutReuseError("source material escapes source root") from None
        if path.is_symlink() or not path.is_file():
            raise RolloutReuseError("source material topology differs")
        add(f"source/{safe}", path.read_bytes())
    return digest.hexdigest()


__all__ = [
    "OVERLAY_SCHEMA",
    "ROW_SCHEMA",
    "VERIFY_SCHEMA",
    "ReusableStart",
    "RolloutReuseError",
    "discover_reusable_starts",
    "map_reusable_starts",
    "overlay_root",
    "referenced_source_paths",
]
