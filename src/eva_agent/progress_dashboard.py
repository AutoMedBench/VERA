"""Credential-free, read-only progress views for long-running EVA campaigns.

The dashboard deliberately observes only durable local state.  It never opens a
provider route, reads a process environment, or treats a persistent Codex
app-server as proof that an API request is in flight.  Admission remains owned
by :mod:`eva_agent.campaign`.  The primary training counter is the signed
bulk-RL package: reopening it verifies the small manifest and stats its shard
descriptors, but deliberately does not read or hash JSONL records on every
refresh.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from eva_agent.admission.receipts import (
    AdmissionReceiptError,
    SignedEnvelope,
    verify_signed_envelope,
)
from eva_agent.construction.premium_campaign import (
    PremiumConstructionCampaignError,
    PremiumConstructionQueue,
    PremiumConstructionTerminalRecord,
    _verify_claim,
)
from eva_agent.pipeline.digests import blake3_hex


_MAX_LOCAL_DOCUMENT_BYTES = 64 * 1024 * 1024
_BULK_RL_TARGET = 6_000
_BULK_RL_SCHEMA = "eva.bulk-rl-sandbox-sharded-dataset.v1"
_SFT_DATASET_SCHEMAS = frozenset({"eva.legacy-teacher-sft-dataset.v1"})
_FRONTIER_SFT_DATASET_SCHEMAS = frozenset(
    {
        "eva.execution-verified-frontier-prefix-sft-dataset.v1",
        "eva.execution-verified-frontier-prefix-sft-dataset.v2",
        "eva.execution-verified-frontier-prefix-sft-dataset.v3",
        "eva.execution-verified-s2-frontier-prefix-sft-dataset.v1",
    }
)
_ROLLOUT_LAUNCHER = "run_campaign_v2.py"
_CONSTRUCTION_LAUNCHER = "run_premium_codex_construction.py"
_TEACHER_BATCH_LAUNCHER = "run_codex_teacher_batch_v1.py"
_TEACHER_TASK_LAUNCHER = "run_one_codex_teacher_rollout_v1.py"
_PROFILE_WIDTHS = MappingProxyType(
    {"canary": 1, "1": 1, "128": 128, "256": 256, "512": 512}
)


class ProgressDashboardError(ValueError):
    """A local progress source could not be reopened safely."""


@dataclass(frozen=True, slots=True)
class BulkRLSandboxSnapshot:
    """Compact view of the signed bulk dataset without reopening its records."""

    target: int
    declared: int
    materialized_verified: int
    shard_count: int
    materialized_verified_shards: int
    manifest_signature_verified: bool
    state_exists: bool

    def __post_init__(self) -> None:
        counts = (
            self.target,
            self.declared,
            self.materialized_verified,
            self.shard_count,
            self.materialized_verified_shards,
        )
        if (
            any(type(value) is not int or value < 0 for value in counts)
            or self.target != _BULK_RL_TARGET
            or type(self.manifest_signature_verified) is not bool
            or type(self.state_exists) is not bool
            or self.materialized_verified > self.declared
            or self.declared > self.target
            or self.materialized_verified_shards > self.shard_count
        ):
            raise ProgressDashboardError("bulk RL progress snapshot differs")
        if not self.state_exists and (any(counts[1:]) or self.manifest_signature_verified):
            raise ProgressDashboardError("absent bulk RL snapshot is nonzero")
        if self.materialized_verified and not self.manifest_signature_verified:
            raise ProgressDashboardError("untrusted bulk RL shards cannot count")

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": "eva.bulk-rl-progress-snapshot.v1",
            "target": self.target,
            "declared": self.declared,
            "materialized_verified": self.materialized_verified,
            "shard_count": self.shard_count,
            "materialized_verified_shards": self.materialized_verified_shards,
            "manifest_signature_verified": self.manifest_signature_verified,
            "verification_scope": "signed_manifest_and_shard_metadata_no_record_rehash",
            "state_exists": self.state_exists,
        }


@dataclass(frozen=True, slots=True)
class SFTTrainingSnapshot:
    """Manifest-indexed teacher trajectory and slice counts."""

    full_trajectories: int
    declared_slices: int
    materialized_slices: int
    shard_count: int
    materialized_shards: int
    state_exists: bool

    def __post_init__(self) -> None:
        counts = (
            self.full_trajectories,
            self.declared_slices,
            self.materialized_slices,
            self.shard_count,
            self.materialized_shards,
        )
        if (
            any(type(value) is not int or value < 0 for value in counts)
            or self.materialized_slices > self.declared_slices
            or self.materialized_shards > self.shard_count
            or (not self.state_exists and any(counts))
        ):
            raise ProgressDashboardError("SFT progress snapshot differs")

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": "eva.sft-training-progress-snapshot.v1",
            "category": "canonical_strict_full_trajectory_sft",
            "full_trajectories": self.full_trajectories,
            "declared_slices": self.declared_slices,
            "materialized_slices": self.materialized_slices,
            "shard_count": self.shard_count,
            "materialized_shards": self.materialized_shards,
            "verification_scope": "manifest_and_shard_metadata_no_record_rehash",
            "state_exists": self.state_exists,
        }


@dataclass(frozen=True, slots=True)
class FrontierSFTTrainingSnapshot:
    """Manifest-indexed execution-verified stage-prefix SFT counts."""

    dataset_count: int
    source_prefixes: int
    declared_slices: int
    materialized_slices: int
    shard_count: int
    materialized_shards: int
    state_exists: bool

    def __post_init__(self) -> None:
        counts = (
            self.dataset_count,
            self.source_prefixes,
            self.declared_slices,
            self.materialized_slices,
            self.shard_count,
            self.materialized_shards,
        )
        if (
            any(type(value) is not int or value < 0 for value in counts)
            or self.materialized_slices > self.declared_slices
            or self.materialized_shards > self.shard_count
            or (not self.state_exists and any(counts))
            or (self.state_exists and self.dataset_count < 1)
        ):
            raise ProgressDashboardError("frontier SFT progress snapshot differs")

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": "eva.frontier-sft-training-progress-snapshot.v1",
            "category": "execution_verified_stage_prefix_sft",
            "dataset_count": self.dataset_count,
            "source_prefixes": self.source_prefixes,
            "declared_slices": self.declared_slices,
            "materialized_slices": self.materialized_slices,
            "shard_count": self.shard_count,
            "materialized_shards": self.materialized_shards,
            "agent_judged": False,
            "strict_full_trajectory_sft_eligible": False,
            "verification_scope": (
                "manifest_commitment_and_shard_metadata_no_record_rehash"
            ),
            "state_exists": self.state_exists,
        }


@dataclass(frozen=True, slots=True)
class TeacherBatchSnapshot:
    """Read-only aggregate view of the durable teacher checkpoint."""

    queued: int
    running: int
    succeeded: int
    failed: int
    high_score_sft_slices: int | None
    state_exists: bool

    def __post_init__(self) -> None:
        counts = (self.queued, self.running, self.succeeded, self.failed)
        if (
            any(type(value) is not int or value < 0 for value in counts)
            or type(self.state_exists) is not bool
            or (
                self.high_score_sft_slices is not None
                and (
                    type(self.high_score_sft_slices) is not int
                    or not 0 <= self.high_score_sft_slices <= self.succeeded
                )
            )
            or (not self.state_exists and any(counts))
        ):
            raise ProgressDashboardError("teacher batch progress snapshot differs")

    @property
    def total(self) -> int:
        return self.queued + self.running + self.succeeded + self.failed

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": "eva.teacher-batch-progress-snapshot.v1",
            "queued": self.queued,
            "running": self.running,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "total": self.total,
            "high_score_sft_slices": self.high_score_sft_slices,
            "source": "checkpoint_sqlite_aggregate_only",
            "state_exists": self.state_exists,
        }


@dataclass(frozen=True, slots=True)
class PremiumConstructionSnapshot:
    queue_total: int
    queued: int
    claimed: int
    active: int
    succeeded: int
    quarantined: int
    logical_provider_calls_started_known: int
    state_exists: bool

    def __post_init__(self) -> None:
        counts = (
            self.queue_total,
            self.queued,
            self.claimed,
            self.active,
            self.succeeded,
            self.quarantined,
            self.logical_provider_calls_started_known,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ProgressDashboardError("premium construction snapshot count differs")
        if type(self.state_exists) is not bool:
            raise ProgressDashboardError("premium construction snapshot state flag differs")
        if self.state_exists:
            if not (
                self.completed <= self.claimed <= self.queue_total
                and self.queued == self.queue_total - self.claimed
                and self.active == self.claimed - self.completed
            ):
                raise ProgressDashboardError("premium construction snapshot totals differ")
        elif any(counts):
            raise ProgressDashboardError("absent premium construction snapshot is nonzero")

    @property
    def completed(self) -> int:
        return self.succeeded + self.quarantined

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": "eva.premium-construction-dashboard-snapshot.v1",
            "queue_total": self.queue_total,
            "queued": self.queued,
            "claimed": self.claimed,
            "active": self.active,
            "succeeded": self.succeeded,
            "quarantined": self.quarantined,
            "completed": self.completed,
            "logical_provider_calls_started_known": (
                self.logical_provider_calls_started_known
            ),
            "state_exists": self.state_exists,
        }


@dataclass(frozen=True, slots=True)
class LiveUtilizationSnapshot:
    process_scan_available: bool
    rollout_launchers: int
    construction_launchers: int
    persistent_codex_app_servers: int
    rollout_worker_lanes_configured: int
    construction_worker_lanes_configured: int
    rollout_active_by_stage: Mapping[str, int]
    teacher_batch_processes: int = 0
    teacher_task_processes: int = 0

    def __post_init__(self) -> None:
        counts = (
            self.rollout_launchers,
            self.construction_launchers,
            self.persistent_codex_app_servers,
            self.rollout_worker_lanes_configured,
            self.construction_worker_lanes_configured,
            self.teacher_batch_processes,
            self.teacher_task_processes,
        )
        if type(self.process_scan_available) is not bool or any(
            type(value) is not int or value < 0 for value in counts
        ):
            raise ProgressDashboardError("live utilization snapshot count differs")
        stages = dict(self.rollout_active_by_stage)
        if any(
            not isinstance(stage, str)
            or not stage
            or type(count) is not int
            or count < 1
            for stage, count in stages.items()
        ):
            raise ProgressDashboardError("live utilization active-stage count differs")
        object.__setattr__(self, "rollout_active_by_stage", MappingProxyType(stages))

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": "eva.local-live-utilization.v1",
            "process_scan_available": self.process_scan_available,
            "rollout_launchers": self.rollout_launchers,
            "construction_launchers": self.construction_launchers,
            "persistent_codex_app_servers": self.persistent_codex_app_servers,
            "rollout_worker_lanes_configured": self.rollout_worker_lanes_configured,
            "construction_worker_lanes_configured": (
                self.construction_worker_lanes_configured
            ),
            "teacher_batch_processes": self.teacher_batch_processes,
            "teacher_task_processes": self.teacher_task_processes,
            "teacher_worker_processes": (
                self.teacher_batch_processes + self.teacher_task_processes
            ),
            "rollout_active_by_stage": dict(self.rollout_active_by_stage),
            "provider_requests_in_flight": None,
            "provider_requests_note": (
                "not inferred from persistent processes; requires provider telemetry"
            ),
        }


def _read_regular_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    """Read one local artifact without following a symlink or pathname swap."""

    try:
        before = path.lstat()
    except FileNotFoundError:
        raise ProgressDashboardError(f"local progress artifact is missing: {path}") from None
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size < 1
        or before.st_size > maximum_bytes
    ):
        raise ProgressDashboardError(f"local progress artifact topology differs: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProgressDashboardError(f"cannot safely open local progress artifact: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
            or opened.st_nlink != 1
            or not stat.S_ISREG(opened.st_mode)
        ):
            raise ProgressDashboardError(
                f"local progress artifact changed during reopen: {path}"
            )
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) != opened.st_size or len(raw) > maximum_bytes:
            raise ProgressDashboardError(f"local progress artifact size differs: {path}")
        return raw
    finally:
        os.close(descriptor)


def _read_json_object(path: Path) -> Mapping[str, Any]:
    raw = _read_regular_bytes(path, maximum_bytes=_MAX_LOCAL_DOCUMENT_BYTES)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProgressDashboardError(f"local progress artifact is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProgressDashboardError(f"local progress artifact must be an object: {path}")
    return MappingProxyType(value)


def _existing_regular_file_has_size(path: Path, expected_size: int) -> bool:
    """Use metadata only; progress refreshes must never stream dataset shards."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if path.is_symlink():
        return False
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and metadata.st_size == expected_size
    )


def _count_manifest_shards(
    *,
    dataset_root: Path,
    shards: Any,
    declared_shard_count: Any,
    record_count_key: str,
    path_prefix: str,
    filename_prefix: str,
) -> tuple[int, int, int]:
    if (
        type(declared_shard_count) is not int
        or declared_shard_count < 0
        or not isinstance(shards, (list, tuple))
        or len(shards) != declared_shard_count
    ):
        raise ProgressDashboardError("dataset shard inventory differs")
    materialized_records = 0
    materialized_shards = 0
    declared_records = 0
    for ordinal, shard in enumerate(shards):
        expected_path = f"{path_prefix}{filename_prefix}{ordinal:05d}.jsonl"
        if (
            not isinstance(shard, Mapping)
            or ("ordinal" in shard and shard.get("ordinal") != ordinal)
            or shard.get("path") != expected_path
            or type(shard.get(record_count_key)) is not int
            or shard[record_count_key] < 1
            or type(shard.get("byte_count")) is not int
            or shard["byte_count"] < 1
        ):
            raise ProgressDashboardError("dataset shard descriptor differs")
        declared_records += shard[record_count_key]
        shard_path = dataset_root.joinpath(*expected_path.split("/"))
        if _existing_regular_file_has_size(shard_path, shard["byte_count"]):
            materialized_shards += 1
            materialized_records += shard[record_count_key]
    return declared_records, materialized_records, materialized_shards


def read_bulk_rl_sandbox_progress(
    root: Path,
    *,
    trust_store_path: Path,
) -> BulkRLSandboxSnapshot:
    """Verify one compact manifest and stat shards without reading JSONL rows."""

    root = Path(root)
    try:
        metadata = root.lstat()
    except FileNotFoundError:
        return BulkRLSandboxSnapshot(_BULK_RL_TARGET, 0, 0, 0, 0, False, False)
    if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
        raise ProgressDashboardError("bulk RL dataset root topology differs")
    try:
        envelope = SignedEnvelope.from_document(_read_json_object(root / "manifest.json"))
        verify_signed_envelope(envelope, trust_store_path=Path(trust_store_path))
    except AdmissionReceiptError as exc:
        raise ProgressDashboardError("bulk RL manifest signature verification failed") from exc
    manifest = envelope.payload
    if manifest.get("schema") != _BULK_RL_SCHEMA:
        raise ProgressDashboardError("bulk RL manifest schema differs")
    declared = manifest.get("sandbox_count")
    if (
        type(declared) is not int
        or not 0 <= declared <= _BULK_RL_TARGET
        or manifest.get("split_counts") != {"train": declared}
    ):
        raise ProgressDashboardError("bulk RL manifest totals differ")
    shard_records, materialized, materialized_shards = _count_manifest_shards(
        dataset_root=root,
        shards=manifest.get("shards"),
        declared_shard_count=manifest.get("shard_count"),
        record_count_key="record_count",
        path_prefix="",
        filename_prefix="sandboxes-",
    )
    if shard_records != declared:
        raise ProgressDashboardError("bulk RL shard totals differ")
    return BulkRLSandboxSnapshot(
        target=_BULK_RL_TARGET,
        declared=declared,
        materialized_verified=materialized,
        shard_count=manifest["shard_count"],
        materialized_verified_shards=materialized_shards,
        manifest_signature_verified=True,
        state_exists=True,
    )


def read_sft_training_progress(output: Path | None) -> SFTTrainingSnapshot:
    """Read a teacher-SFT manifest and stat shards without reopening slices."""

    if output is None:
        return SFTTrainingSnapshot(0, 0, 0, 0, 0, False)
    output = Path(output)
    try:
        metadata = output.lstat()
    except FileNotFoundError:
        return SFTTrainingSnapshot(0, 0, 0, 0, 0, False)
    if output.is_symlink():
        raise ProgressDashboardError("SFT output topology differs")
    if stat.S_ISDIR(metadata.st_mode):
        dataset_root = output
        manifest_path = output / "manifest.json"
    elif stat.S_ISREG(metadata.st_mode):
        dataset_root = output.parent
        manifest_path = output
    else:
        raise ProgressDashboardError("SFT output topology differs")
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema") not in _SFT_DATASET_SCHEMAS:
        raise ProgressDashboardError("SFT manifest schema differs")
    full_trajectories = manifest.get("source_count")
    declared_slices = manifest.get("slice_count")
    if (
        type(full_trajectories) is not int
        or full_trajectories < 0
        or type(declared_slices) is not int
        or declared_slices < 0
    ):
        raise ProgressDashboardError("SFT manifest totals differ")
    shard_slices, materialized_slices, materialized_shards = _count_manifest_shards(
        dataset_root=dataset_root,
        shards=manifest.get("shards"),
        declared_shard_count=manifest.get("shard_count"),
        record_count_key="slice_count",
        path_prefix="shards/",
        filename_prefix="part-",
    )
    if shard_slices != declared_slices:
        raise ProgressDashboardError("SFT shard totals differ")
    return SFTTrainingSnapshot(
        full_trajectories=full_trajectories,
        declared_slices=declared_slices,
        materialized_slices=materialized_slices,
        shard_count=manifest["shard_count"],
        materialized_shards=materialized_shards,
        state_exists=True,
    )


def _frontier_sft_output_paths(
    output: Path | Sequence[Path] | None,
) -> tuple[Path, ...]:
    if output is None:
        return ()
    values = (output,) if isinstance(output, (str, os.PathLike)) else tuple(output)
    paths: list[Path] = []
    identities: set[str] = set()
    for value in values:
        path = Path(value)
        identity = os.path.abspath(path)
        if identity in identities:
            raise ProgressDashboardError("frontier SFT output is duplicated")
        identities.add(identity)
        paths.append(path)
    return tuple(paths)


def _read_one_frontier_sft_progress(
    output: Path,
) -> tuple[FrontierSFTTrainingSnapshot, str | None]:
    try:
        metadata = output.lstat()
    except FileNotFoundError:
        return FrontierSFTTrainingSnapshot(0, 0, 0, 0, 0, 0, False), None
    if output.is_symlink():
        raise ProgressDashboardError("frontier SFT output topology differs")
    if stat.S_ISDIR(metadata.st_mode):
        dataset_root = output
        manifest_path = output / "manifest.json"
    elif stat.S_ISREG(metadata.st_mode):
        dataset_root = output.parent
        manifest_path = output
    else:
        raise ProgressDashboardError("frontier SFT output topology differs")
    manifest = _read_json_object(manifest_path)
    schema = manifest.get("schema")
    core = {key: value for key, value in manifest.items() if key != "manifest_blake3"}
    selection = manifest.get("selection")
    source_count = manifest.get("source_count")
    declared_slices = manifest.get("slice_count")
    is_s2 = schema == "eva.execution-verified-s2-frontier-prefix-sft-dataset.v1"
    if is_s2:
        schema_specific_valid = (
            manifest.get("quality_tier") == "execution_verified_s2_frontier_prefix"
            and manifest.get("selection_status")
            == "s2_prefix_verified_full_trajectory_not_claimed"
            and manifest.get("counts_by_stage") == {"S2": declared_slices}
            and manifest.get("private_reference_observations_included") is False
            and isinstance(selection, Mapping)
            and selection.get("require_completed_s2_selection_frontier") is True
            and selection.get("require_exact_sequential_prerequisites") is True
            and selection.get("require_authoritative_s1_s2_guidance") is True
            and selection.get("export_prerequisite_tool_observations") is False
            and selection.get("export_private_reference_observations") is False
            and selection.get("export_later_stages") is False
            and selection.get("export_terminal_answer") is False
            and selection.get("export_hidden_reasoning") is False
        )
    else:
        schema_specific_valid = (
            manifest.get("quality_tier") == "execution_verified_frontier_prefix"
            and manifest.get("selection_status")
            == "stage_prefix_verified_full_trajectory_not_claimed"
            and manifest.get("counts_by_stage") == {"S1": declared_slices}
            and manifest.get("private_reference_included") is False
            and isinstance(selection, Mapping)
            and selection.get("require_completed_frontier") is True
            and selection.get("require_exact_declared_artifact") is True
            and selection.get("export_later_stages") is False
            and selection.get("export_terminal_answer") is False
            and selection.get("export_hidden_reasoning") is False
            and selection.get("export_private_reference") is False
        )
    if (
        schema not in _FRONTIER_SFT_DATASET_SCHEMAS
        or not schema_specific_valid
        or manifest.get("manifest_blake3") != blake3_hex(core)
        or not isinstance(manifest.get("dataset_id"), str)
        or dataset_root.name != manifest["dataset_id"]
        or type(source_count) is not int
        or source_count < 1
        or type(declared_slices) is not int
        or declared_slices != source_count
        or manifest.get("agent_judged") is not False
        or manifest.get("strict_full_trajectory_sft_eligible") is not False
        or manifest.get("hidden_reasoning_included") is not False
        or not isinstance(selection, Mapping)
        or selection.get("require_gate_passed_true") is not True
        or selection.get("require_empty_failed_check_ids") is not True
        or selection.get("require_reopened_workspace_delta") is not True
    ):
        raise ProgressDashboardError("frontier SFT manifest identity differs")
    shards = manifest.get("shards")
    shard_records, materialized, materialized_shards = _count_manifest_shards(
        dataset_root=dataset_root,
        shards=shards,
        declared_shard_count=len(shards) if isinstance(shards, list) else None,
        record_count_key="slice_count",
        path_prefix="shards/",
        filename_prefix="part-",
    )
    if shard_records != declared_slices:
        raise ProgressDashboardError("frontier SFT shard totals differ")
    return (
        FrontierSFTTrainingSnapshot(
            dataset_count=1,
            source_prefixes=source_count,
            declared_slices=declared_slices,
            materialized_slices=materialized,
            shard_count=len(shards),
            materialized_shards=materialized_shards,
            state_exists=True,
        ),
        manifest["dataset_id"],
    )


def read_frontier_sft_training_progress(
    output: Path | Sequence[Path] | None,
) -> FrontierSFTTrainingSnapshot:
    """Aggregate stage-prefix datasets using manifests and shard metadata only."""

    paths = _frontier_sft_output_paths(output)
    if not paths:
        return FrontierSFTTrainingSnapshot(0, 0, 0, 0, 0, 0, False)
    opened = tuple(_read_one_frontier_sft_progress(path) for path in paths)
    present = tuple(snapshot for snapshot, _ in opened if snapshot.state_exists)
    dataset_ids = tuple(dataset_id for _, dataset_id in opened if dataset_id is not None)
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ProgressDashboardError("frontier SFT dataset is duplicated")
    if not present:
        return FrontierSFTTrainingSnapshot(0, 0, 0, 0, 0, 0, False)
    return FrontierSFTTrainingSnapshot(
        dataset_count=sum(snapshot.dataset_count for snapshot in present),
        source_prefixes=sum(snapshot.source_prefixes for snapshot in present),
        declared_slices=sum(snapshot.declared_slices for snapshot in present),
        materialized_slices=sum(snapshot.materialized_slices for snapshot in present),
        shard_count=sum(snapshot.shard_count for snapshot in present),
        materialized_shards=sum(snapshot.materialized_shards for snapshot in present),
        state_exists=True,
    )


def _teacher_output_paths(
    output: Path | Sequence[Path] | None,
) -> tuple[Path, ...]:
    if output is None:
        return ()
    values = (output,) if isinstance(output, (str, os.PathLike)) else tuple(output)
    paths: list[Path] = []
    identities: set[str] = set()
    for value in values:
        path = Path(value)
        identity = os.path.abspath(path)
        if identity in identities:
            raise ProgressDashboardError("teacher output is duplicated")
        identities.add(identity)
        paths.append(path)
    return tuple(paths)


def _read_one_teacher_batch_progress(output: Path) -> TeacherBatchSnapshot:
    try:
        metadata = output.lstat()
    except FileNotFoundError:
        return TeacherBatchSnapshot(0, 0, 0, 0, None, False)
    if output.is_symlink():
        raise ProgressDashboardError("teacher output topology differs")
    if stat.S_ISDIR(metadata.st_mode):
        checkpoint = output / "checkpoint.sqlite3"
        try:
            checkpoint_metadata = checkpoint.lstat()
        except FileNotFoundError:
            return TeacherBatchSnapshot(0, 0, 0, 0, None, False)
    elif stat.S_ISREG(metadata.st_mode):
        checkpoint = output
        checkpoint_metadata = metadata
    else:
        raise ProgressDashboardError("teacher output topology differs")
    if (
        checkpoint.is_symlink()
        or not stat.S_ISREG(checkpoint_metadata.st_mode)
        or checkpoint_metadata.st_size < 1
    ):
        raise ProgressDashboardError("teacher checkpoint topology differs")
    try:
        uri = f"{checkpoint.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=5) as connection:
            connection.execute("PRAGMA query_only=ON")
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
            }
            required = {"state", "result_path"}
            if not required <= columns:
                raise ProgressDashboardError("teacher checkpoint schema differs")
            rows = connection.execute(
                "SELECT state,COUNT(*) FROM tasks GROUP BY state"
            ).fetchall()
            slice_count = None
            if {"score", "sft_slice_path"} <= columns:
                slice_row = connection.execute(
                    "SELECT COUNT(*) FROM tasks WHERE state='succeeded' "
                    "AND score IS NOT NULL AND sft_slice_path IS NOT NULL"
                ).fetchone()
                slice_count = int(slice_row[0])
    except sqlite3.Error as exc:
        raise ProgressDashboardError("cannot read teacher checkpoint") from exc
    counts = {state: 0 for state in ("queued", "running", "succeeded", "failed")}
    for state, count in rows:
        if state not in counts or type(count) is not int or count < 0:
            raise ProgressDashboardError("teacher checkpoint state differs")
        counts[state] = count
    return TeacherBatchSnapshot(
        queued=counts["queued"],
        running=counts["running"],
        succeeded=counts["succeeded"],
        failed=counts["failed"],
        high_score_sft_slices=slice_count,
        state_exists=True,
    )


def read_teacher_batch_progress(
    output: Path | Sequence[Path] | None,
) -> TeacherBatchSnapshot:
    """Aggregate counters from one or more SQLite checkpoints only.

    The function never follows ``result_path`` or ``sft_slice_path`` and never
    opens, scans, or hashes a trajectory receipt.  Passing one path retains the
    original API; repeated CLI flags supply a sequence.
    """

    paths = _teacher_output_paths(output)
    if not paths:
        return TeacherBatchSnapshot(0, 0, 0, 0, None, False)
    snapshots = tuple(_read_one_teacher_batch_progress(path) for path in paths)
    present = tuple(snapshot for snapshot in snapshots if snapshot.state_exists)
    if not present:
        return TeacherBatchSnapshot(0, 0, 0, 0, None, False)
    slices = (
        sum(snapshot.high_score_sft_slices or 0 for snapshot in present)
        if all(snapshot.high_score_sft_slices is not None for snapshot in present)
        else None
    )
    return TeacherBatchSnapshot(
        queued=sum(snapshot.queued for snapshot in present),
        running=sum(snapshot.running for snapshot in present),
        succeeded=sum(snapshot.succeeded for snapshot in present),
        failed=sum(snapshot.failed for snapshot in present),
        high_score_sft_slices=slices,
        state_exists=True,
    )


def _directory_documents(root: Path, name: str) -> dict[str, Mapping[str, Any]]:
    directory = root / name
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        raise ProgressDashboardError(f"premium construction {name} directory is missing") from None
    if not stat.S_ISDIR(metadata.st_mode) or directory.is_symlink():
        raise ProgressDashboardError(f"premium construction {name} topology differs")
    documents: dict[str, Mapping[str, Any]] = {}
    for path in sorted(directory.iterdir(), key=lambda value: value.name):
        if path.suffix != ".json" or not path.stem:
            raise ProgressDashboardError(
                f"premium construction {name} filename differs"
            )
        documents[path.stem] = _read_json_object(path)
    return documents


def read_premium_construction_progress(root: Path) -> PremiumConstructionSnapshot:
    """Reopen append-only construction state without creating any path."""

    root = Path(root)
    try:
        metadata = root.lstat()
    except FileNotFoundError:
        return PremiumConstructionSnapshot(0, 0, 0, 0, 0, 0, 0, False)
    if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
        raise ProgressDashboardError("premium construction state root topology differs")
    try:
        queue = PremiumConstructionQueue.from_document(
            _read_json_object(root / "queue.v1.json")
        )
        claims = _directory_documents(root, "claims")
        terminal_documents = _directory_documents(root, "terminal")
        terminals = {
            candidate_id: PremiumConstructionTerminalRecord.from_document(document)
            for candidate_id, document in terminal_documents.items()
        }
    except PremiumConstructionCampaignError as exc:
        raise ProgressDashboardError("premium construction state verification failed") from exc

    entries = {entry.candidate_id: entry for entry in queue.entries}
    if set(claims) - set(entries) or set(terminals) - set(entries):
        raise ProgressDashboardError("premium construction state candidate set differs")
    for candidate_id, claim in claims.items():
        if claim.get("candidate_id") != candidate_id:
            raise ProgressDashboardError("premium construction claim filename differs")
        try:
            _verify_claim(claim, queue=queue, entry=entries[candidate_id])
        except PremiumConstructionCampaignError as exc:
            raise ProgressDashboardError("premium construction claim verification failed") from exc
    for candidate_id, record in terminals.items():
        entry = entries[candidate_id]
        claim = claims.get(candidate_id)
        if not (
            record.candidate_id == candidate_id
            and record.queue_id == queue.queue_id
            and record.queue_blake3 == queue.queue_blake3
            and record.selection_blake3 == queue.selection_blake3
            and record.source_candidate_id == entry.source_candidate_id
            and record.campaign_ordinal == entry.campaign_ordinal
            and claim is not None
            and record.claim_blake3 == claim.get("claim_blake3")
        ):
            raise ProgressDashboardError("premium construction terminal binding differs")

    claimed = len(claims)
    succeeded = sum(record.status == "succeeded" for record in terminals.values())
    quarantined = sum(record.status == "quarantined" for record in terminals.values())
    completed = succeeded + quarantined
    active = claimed - completed
    queued = len(queue.entries) - claimed
    if min(claimed, active, queued) < 0 or completed != len(terminals):
        raise ProgressDashboardError("premium construction progress totals differ")
    calls = sum(record.logical_provider_calls_started or 0 for record in terminals.values())
    return PremiumConstructionSnapshot(
        queue_total=len(queue.entries),
        queued=queued,
        claimed=claimed,
        active=active,
        succeeded=succeeded,
        quarantined=quarantined,
        logical_provider_calls_started_known=calls,
        state_exists=True,
    )


def read_rollout_active_stages(ledger: Path) -> Mapping[str, int]:
    """Read only stage-level active occupancy from an already verified ledger."""

    ledger = Path(ledger)
    if not ledger.exists():
        return MappingProxyType({})
    if ledger.is_symlink() or not ledger.is_file():
        raise ProgressDashboardError("admission ledger must be a regular non-symlink file")
    try:
        uri = f"{ledger.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=5) as connection:
            rows = connection.execute(
                "SELECT stage,COUNT(*) FROM candidates "
                "WHERE status='active' GROUP BY stage ORDER BY stage"
            ).fetchall()
    except sqlite3.Error as exc:
        raise ProgressDashboardError("cannot read active rollout stages") from exc
    result: dict[str, int] = {}
    for stage, count in rows:
        if not isinstance(stage, str) or not stage or type(count) is not int or count < 1:
            raise ProgressDashboardError("active rollout stage projection differs")
        result[stage] = count
    return MappingProxyType(result)


def _profile_width(arguments: tuple[bytes, ...], launcher_index: int) -> int:
    """Extract only an allowlisted profile value; never retain command text."""

    for index in range(launcher_index + 1, len(arguments) - 1):
        if arguments[index] != b"--profile":
            continue
        try:
            value = arguments[index + 1].decode("ascii")
        except UnicodeDecodeError:
            return 0
        return int(_PROFILE_WIDTHS.get(value, 0))
    return 0


def _launcher_command(arguments: tuple[bytes, ...], launcher_index: int) -> str:
    if launcher_index + 1 >= len(arguments):
        return ""
    try:
        return arguments[launcher_index + 1].decode("ascii")
    except UnicodeDecodeError:
        return ""


def _read_proc_cmdline(path: Path) -> bytes:
    """Read a procfs cmdline whose reported stat size is normally zero."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProgressDashboardError("cannot inspect local process metadata") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ProgressDashboardError("local process metadata topology differs")
        chunks: list[bytes] = []
        remaining = 1024 * 1024
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if not raw or remaining == 0:
            raise ProgressDashboardError("local process metadata size differs")
        return raw
    finally:
        os.close(descriptor)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def read_live_utilization(
    *,
    ledger: Path,
    project_root: Path,
    proc_root: Path = Path("/proc"),
    runtime_worktrees: Sequence[Path] = (),
) -> LiveUtilizationSnapshot:
    """Count allowlisted local processes without exposing argv, PIDs, or env."""

    active_by_stage = read_rollout_active_stages(ledger)
    project_root = Path(project_root).resolve()
    allowed_roots = [project_root]
    for value in runtime_worktrees:
        path = Path(value)
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ProgressDashboardError("runtime worktree is unavailable") from exc
        if path.is_symlink() or not resolved.is_dir():
            raise ProgressDashboardError("runtime worktree topology differs")
        if resolved not in allowed_roots:
            allowed_roots.append(resolved)
    proc_root = Path(proc_root)
    if not proc_root.is_dir():
        return LiveUtilizationSnapshot(
            False, 0, 0, 0, 0, 0, active_by_stage
        )
    rollout_launchers = construction_launchers = app_servers = 0
    teacher_batches = teacher_tasks = 0
    rollout_width = construction_width = 0
    try:
        process_directories = tuple(proc_root.iterdir())
    except OSError:
        return LiveUtilizationSnapshot(
            False, 0, 0, 0, 0, 0, active_by_stage
        )
    for process in process_directories:
        if not process.name.isdecimal():
            continue
        try:
            cwd = (process / "cwd").resolve(strict=True)
            if not any(_is_under(cwd, root) for root in allowed_roots):
                continue
            raw = _read_proc_cmdline(process / "cmdline")
        except (OSError, ProgressDashboardError):
            continue
        arguments = tuple(part for part in raw.split(b"\0") if part)
        basenames = tuple(os.path.basename(part) for part in arguments)
        rollout_indices = [
            index for index, value in enumerate(basenames) if value == _ROLLOUT_LAUNCHER.encode()
        ]
        construction_indices = [
            index
            for index, value in enumerate(basenames)
            if value == _CONSTRUCTION_LAUNCHER.encode()
        ]
        teacher_batch_indices = [
            index
            for index, value in enumerate(basenames)
            if value == _TEACHER_BATCH_LAUNCHER.encode()
        ]
        teacher_task_indices = [
            index
            for index, value in enumerate(basenames)
            if value == _TEACHER_TASK_LAUNCHER.encode()
        ]
        if rollout_indices:
            index = rollout_indices[0]
            command = _launcher_command(arguments, index)
            if command in {"canary", "production"}:
                rollout_launchers += 1
                rollout_width += (
                    1 if command == "canary" else _profile_width(arguments, index)
                )
        if construction_indices:
            index = construction_indices[0]
            command = _launcher_command(arguments, index)
            if command in {"canary", "next-canary", "production"}:
                construction_launchers += 1
                construction_width += _profile_width(arguments, index)
        if teacher_batch_indices:
            index = teacher_batch_indices[0]
            if _launcher_command(arguments, index) == "run":
                teacher_batches += 1
        if teacher_task_indices:
            teacher_tasks += 1
        if any(value == b"app-server" for value in arguments) and any(
            b"codex" in value.lower() for value in basenames
        ):
            app_servers += 1
    return LiveUtilizationSnapshot(
        process_scan_available=True,
        rollout_launchers=rollout_launchers,
        construction_launchers=construction_launchers,
        persistent_codex_app_servers=app_servers,
        rollout_worker_lanes_configured=rollout_width,
        construction_worker_lanes_configured=construction_width,
        rollout_active_by_stage=active_by_stage,
        teacher_batch_processes=teacher_batches,
        teacher_task_processes=teacher_tasks,
    )


def render_premium_construction_progress(
    snapshot: PremiumConstructionSnapshot, *, width: int = 30
) -> str:
    if type(width) is not int or not 10 <= width <= 100:
        raise ProgressDashboardError("premium construction progress width differs")
    if not snapshot.state_exists:
        return "premium construction: state absent (provider-free local view)"
    target = snapshot.queue_total
    completed = snapshot.completed
    filled = width if target == 0 else min(width, completed * width // target)
    bar = "█" * filled + "░" * (width - filled)
    percent = 100.0 if target == 0 else completed * 100.0 / target
    return (
        f"premium construction [{bar}] {completed:,}/{target:,} ({percent:.2f}%); "
        f"queued={snapshot.queued:,}; active={snapshot.active:,}; "
        f"succeeded={snapshot.succeeded:,}; quarantined={snapshot.quarantined:,}; "
        f"known_calls={snapshot.logical_provider_calls_started_known:,}"
    )


def render_bulk_rl_sandbox_progress(
    snapshot: BulkRLSandboxSnapshot, *, width: int = 30
) -> str:
    if type(width) is not int or not 10 <= width <= 100:
        raise ProgressDashboardError("bulk RL progress width differs")
    filled = min(width, snapshot.materialized_verified * width // snapshot.target)
    bar = "█" * filled + "░" * (width - filled)
    percent = snapshot.materialized_verified * 100.0 / snapshot.target
    source = (
        "signed manifest + shard metadata; JSONL not rehashed"
        if snapshot.state_exists
        else "dataset absent; fail-closed zero"
    )
    return (
        f"RL sandboxes [{bar}] {snapshot.materialized_verified:,}/"
        f"{snapshot.target:,} materialized+verified ({percent:.2f}%); "
        f"shards={snapshot.materialized_verified_shards}/"
        f"{snapshot.shard_count}; source={source}"
    )


def render_sft_training_progress(snapshot: SFTTrainingSnapshot) -> str:
    if not snapshot.state_exists:
        return "SFT canonical strict: output absent"
    return (
        f"SFT canonical strict: full_trajectories={snapshot.full_trajectories:,}; "
        f"slices={snapshot.materialized_slices:,}/{snapshot.declared_slices:,}; "
        f"shards={snapshot.materialized_shards}/{snapshot.shard_count}; "
        "manifest-indexed, records not rehashed"
    )


def render_frontier_sft_training_progress(
    snapshot: FrontierSFTTrainingSnapshot,
) -> str:
    if not snapshot.state_exists:
        return "SFT execution-verified stage prefixes: output absent"
    return (
        "SFT execution-verified stage prefixes: "
        f"datasets={snapshot.dataset_count:,}; sources={snapshot.source_prefixes:,}; "
        f"slices={snapshot.materialized_slices:,}/{snapshot.declared_slices:,}; "
        f"shards={snapshot.materialized_shards}/{snapshot.shard_count}; "
        "not Agent-Judged, not full-trajectory; manifest-indexed, records not rehashed"
    )


def render_sft_total_training_records(
    strict: SFTTrainingSnapshot,
    frontier: FrontierSFTTrainingSnapshot,
) -> str:
    strict_records = strict.materialized_slices
    frontier_records = frontier.materialized_slices
    return (
        f"SFT training records: total={strict_records + frontier_records:,}; "
        f"canonical_strict={strict_records:,}; "
        f"execution_verified_stage_prefix={frontier_records:,}"
    )


def render_teacher_batch_progress(snapshot: TeacherBatchSnapshot) -> str:
    if not snapshot.state_exists:
        return "teacher batch: checkpoint absent"
    slices = (
        f"{snapshot.high_score_sft_slices:,}"
        if snapshot.high_score_sft_slices is not None
        else "unavailable-in-checkpoint"
    )
    return (
        f"teacher batch: queued={snapshot.queued:,}; running={snapshot.running:,}; "
        f"succeeded={snapshot.succeeded:,}; failed={snapshot.failed:,}; "
        f"high_score_sft_slices={slices}; SQLite aggregates only"
    )


def render_live_utilization(
    live: LiveUtilizationSnapshot,
    *,
    rollout_active: int,
    construction_active: int,
) -> str:
    if not live.process_scan_available:
        return "live local utilization: process scan unavailable"
    stages = ",".join(
        f"{stage}={count}" for stage, count in live.rollout_active_by_stage.items()
    ) or "none"
    rollout_capacity = live.rollout_worker_lanes_configured or "?"
    construction_capacity = live.construction_worker_lanes_configured or "?"
    return (
        "live local utilization: "
        f"rollout={rollout_active}/{rollout_capacity} candidate lanes "
        f"({live.rollout_launchers} launcher); "
        f"construction={construction_active}/{construction_capacity} candidate lanes "
        f"({live.construction_launchers} launcher); "
        f"teacher_workers={live.teacher_batch_processes + live.teacher_task_processes} "
        f"(batch={live.teacher_batch_processes},task={live.teacher_task_processes}); "
        f"active_stages={stages}; persistent_codex_app_servers="
        f"{live.persistent_codex_app_servers}; provider_requests_in_flight=unobserved"
    )


__all__ = [
    "BulkRLSandboxSnapshot",
    "FrontierSFTTrainingSnapshot",
    "LiveUtilizationSnapshot",
    "PremiumConstructionSnapshot",
    "ProgressDashboardError",
    "SFTTrainingSnapshot",
    "TeacherBatchSnapshot",
    "read_bulk_rl_sandbox_progress",
    "read_frontier_sft_training_progress",
    "read_live_utilization",
    "read_premium_construction_progress",
    "read_rollout_active_stages",
    "read_sft_training_progress",
    "read_teacher_batch_progress",
    "render_bulk_rl_sandbox_progress",
    "render_frontier_sft_training_progress",
    "render_live_utilization",
    "render_premium_construction_progress",
    "render_sft_training_progress",
    "render_sft_total_training_records",
    "render_teacher_batch_progress",
]
