"""Outcome-blind, append-only premium construction campaign.

The campaign in this module is deliberately narrower than the rollout and
admission campaign.  It consumes only ``frozen`` rows from an already verified
selection-v2 object, executes the versioned premium construction DAG,
and publishes either a legacy-validated EVA executable binding or an
immutable quarantine record.  It never signs or advances supervisor state.

Durability is filesystem based and append-only.  A candidate claim is created
with ``O_EXCL`` before any provider call.  A process restart never retries an
unfinished claim: after obtaining the campaign's exclusive process lock it
materializes an ``abandoned_after_process_loss`` quarantine record.  Running
the process under tmux therefore survives an SSH disconnect; a host/process
loss remains fail-closed and non-retriable.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import math
import os
from pathlib import Path
import stat
from threading import Lock
import time
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence
from uuid import NAMESPACE_URL, uuid5

from eva_agent.campaign.selection_v2 import CampaignSelectionV2
from eva_agent.pipeline.contracts import JsonValue, RuntimeIdFactory, freeze_json, uuid_text
from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes, canonical_value, is_blake3
from eva_agent.pipeline.ids import RandomUUIDFactory

from .legacy_exact import (
    LegacyConstructionComposition,
    _decode_object,
    _ensure_directory,
    _existing_directory,
    _open_directory_chain,
    _read_relative,
    _safe_relative,
    _write_new_relative,
    verify_legacy_construction_publication,
)
from .premium_codex import (
    AUTHOR_LANES,
    CRITIC_QUORUM_LANES,
    ConstructionAttemptEvidence,
    ConstructionFailureReceipt,
    ConstructionLane,
    ConstructionPublicationReceipt,
    ConstructionTurnRequest,
    ConstructionWaveError,
    FrozenConstructionSource,
    PremiumCodexConstructionOrchestrator,
    PremiumConstructionRoutes,
    PublishedConstructionResult,
    V3_LANE_ORDER,
    V4_LANE_ORDER,
    PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
    PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4,
    verify_construction_attempt_evidence,
    verify_construction_failure_receipt,
)


_ALLOWED_WORKER_WIDTHS = frozenset({1, 128, 256, 512})
_APP_SERVER_SHARDS = 64
_MAX_CAMPAIGN_DOCUMENT_BYTES = 64 * 1024 * 1024


class PremiumConstructionCampaignError(ValueError):
    """A construction campaign boundary failed closed."""


def _now(clock: Callable[[], str]) -> str:
    value = clock()
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PremiumConstructionCampaignError("campaign clock must return UTC text")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise PremiumConstructionCampaignError("campaign clock differs") from None
    return value


def _digest_tuple(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not is_blake3(value) for value in result):
        raise PremiumConstructionCampaignError(f"{label} BLAKE3 differs")
    return result


def _sha256(value: str, *, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise PremiumConstructionCampaignError(f"{label} SHA-256 differs")
    return value


@dataclass(frozen=True, slots=True)
class PremiumConstructionQueueEntry:
    candidate_id: str
    source_candidate_id: str
    source_family: str
    source_artifact_sha256: str
    domain: str
    stage: str
    selection_tier: str
    selection_queue_ordinal: int
    campaign_ordinal: int
    readiness_proof_root_blake3: str
    entry_blake3: str

    @classmethod
    def create(cls, row: Any, *, campaign_ordinal: int) -> "PremiumConstructionQueueEntry":
        core = {
            "candidate_id": row.candidate_id,
            "source_candidate_id": row.source_candidate_id,
            "source_family": row.source_family,
            "source_artifact_sha256": row.source_artifact_sha256,
            "domain": row.domain,
            "stage": row.stage,
            "selection_tier": row.selection_tier,
            "selection_queue_ordinal": row.queue_ordinal,
            "campaign_ordinal": campaign_ordinal,
            "readiness_proof_root_blake3": row.readiness_proof_root_blake3,
        }
        return cls(**core, entry_blake3=blake3_hex(core))

    def __post_init__(self) -> None:
        uuid_text(self.candidate_id, label="construction queue candidate_id")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.source_candidate_id,
                self.source_family,
                self.domain,
                self.stage,
                self.selection_tier,
            )
        ):
            raise PremiumConstructionCampaignError("construction queue identity differs")
        _sha256(self.source_artifact_sha256, label="construction source artifact")
        if not is_blake3(self.readiness_proof_root_blake3):
            raise PremiumConstructionCampaignError("construction readiness proof differs")
        if (
            type(self.selection_queue_ordinal) is not int
            or self.selection_queue_ordinal < 1
            or type(self.campaign_ordinal) is not int
            or self.campaign_ordinal < 1
            or self.entry_blake3 != blake3_hex(self.core())
        ):
            raise PremiumConstructionCampaignError("construction queue entry differs")

    def core(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_candidate_id": self.source_candidate_id,
            "source_family": self.source_family,
            "source_artifact_sha256": self.source_artifact_sha256,
            "domain": self.domain,
            "stage": self.stage,
            "selection_tier": self.selection_tier,
            "selection_queue_ordinal": self.selection_queue_ordinal,
            "campaign_ordinal": self.campaign_ordinal,
            "readiness_proof_root_blake3": self.readiness_proof_root_blake3,
        }

    def to_document(self) -> dict[str, Any]:
        return {**self.core(), "entry_blake3": self.entry_blake3}

    @classmethod
    def from_document(cls, value: Any) -> "PremiumConstructionQueueEntry":
        if not isinstance(value, Mapping):
            raise PremiumConstructionCampaignError("construction queue entry must be an object")
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise PremiumConstructionCampaignError("construction queue entry keys differ")
        return cls(**{key: value[key] for key in expected})


@dataclass(frozen=True, slots=True)
class PremiumConstructionTerminalRecord:
    schema: str
    record_id: str
    queue_id: str
    queue_blake3: str
    selection_blake3: str
    candidate_id: str
    source_candidate_id: str
    campaign_ordinal: int
    session_id: str
    claim_blake3: str
    status: str
    logical_provider_calls_started: int | None
    semantic_retry_count: int
    error_code: str | None
    failure_receipt_blake3s: tuple[str, ...]
    publication_receipt: Mapping[str, JsonValue] | None
    binding_blake3: str | None
    validator_authority_blake3: str | None
    executable_material_blake3: str | None
    completed_at_utc: str
    exception_text_recorded: bool
    supervisor_transition_claimed: bool
    record_blake3: str
    attempt_evidence_blake3s: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema not in {
            "eva.premium-construction-terminal.v1",
            "eva.premium-construction-terminal.v2",
            "eva.premium-construction-terminal.v3",
        }:
            raise PremiumConstructionCampaignError("construction terminal schema differs")
        maximum_calls = (
            5 if self.schema == "eva.premium-construction-terminal.v1" else 7
        )
        for value, label in (
            (self.record_id, "terminal record_id"),
            (self.queue_id, "terminal queue_id"),
            (self.candidate_id, "terminal candidate_id"),
            (self.session_id, "terminal session_id"),
        ):
            uuid_text(value, label=label)
        if not self.source_candidate_id or type(self.campaign_ordinal) is not int or self.campaign_ordinal < 1:
            raise PremiumConstructionCampaignError("construction terminal identity differs")
        for digest in (self.queue_blake3, self.selection_blake3, self.claim_blake3):
            if not is_blake3(digest):
                raise PremiumConstructionCampaignError("construction terminal commitment differs")
        failures = _digest_tuple(self.failure_receipt_blake3s, label="construction failure receipt")
        if len(failures) != len(set(failures)):
            raise PremiumConstructionCampaignError(
                "construction failure receipt is duplicated"
            )
        object.__setattr__(self, "failure_receipt_blake3s", failures)
        attempts = _digest_tuple(
            self.attempt_evidence_blake3s,
            label="construction attempt evidence",
        )
        if len(attempts) != len(set(attempts)):
            raise PremiumConstructionCampaignError(
                "construction attempt evidence is duplicated"
            )
        object.__setattr__(self, "attempt_evidence_blake3s", attempts)
        if self.schema in {
            "eva.premium-construction-terminal.v1",
            "eva.premium-construction-terminal.v2",
        } and attempts:
            raise PremiumConstructionCampaignError(
                "legacy construction terminal embeds v3 attempt evidence"
            )
        publication = None if self.publication_receipt is None else freeze_json(self.publication_receipt)
        if publication is not None and not isinstance(publication, Mapping):
            raise PremiumConstructionCampaignError("construction publication receipt differs")
        object.__setattr__(self, "publication_receipt", publication)
        if self.status == "succeeded":
            if not (
                self.logical_provider_calls_started == maximum_calls
                and self.error_code is None
                and (
                    not failures
                    if self.schema == "eva.premium-construction-terminal.v1"
                    else len(failures) <= 3
                )
                and publication is not None
                and (
                    len(attempts) == 7
                    if self.schema == "eva.premium-construction-terminal.v3"
                    else not attempts
                )
                and all(
                    is_blake3(value)
                    for value in (
                        self.binding_blake3,
                        self.validator_authority_blake3,
                        self.executable_material_blake3,
                    )
                )
            ):
                raise PremiumConstructionCampaignError("successful construction proof differs")
        elif self.status == "quarantined":
            if not (
                self.logical_provider_calls_started is None
                or type(self.logical_provider_calls_started) is int
                and 0 <= self.logical_provider_calls_started <= maximum_calls
            ):
                raise PremiumConstructionCampaignError("quarantine call count differs")
            if not (
                isinstance(self.error_code, str)
                and self.error_code
                and publication is None
                and self.binding_blake3 is None
                and self.validator_authority_blake3 is None
                and self.executable_material_blake3 is None
            ):
                raise PremiumConstructionCampaignError("construction quarantine proof differs")
            if self.schema == "eva.premium-construction-terminal.v3" and not (
                not attempts
                if self.logical_provider_calls_started is None
                else len(attempts) <= self.logical_provider_calls_started
            ):
                raise PremiumConstructionCampaignError(
                    "construction quarantine attempt evidence differs"
                )
        else:
            raise PremiumConstructionCampaignError("construction terminal status differs")
        if (
            self.semantic_retry_count != 0
            or self.exception_text_recorded is not False
            or self.supervisor_transition_claimed is not False
        ):
            raise PremiumConstructionCampaignError("construction terminal controls differ")
        _now(lambda: self.completed_at_utc)
        if self.record_blake3 != blake3_hex(self.core()):
            raise PremiumConstructionCampaignError("construction terminal BLAKE3 differs")

    def core(self) -> dict[str, Any]:
        core = {
            "schema": self.schema,
            "record_id": self.record_id,
            "queue_id": self.queue_id,
            "queue_blake3": self.queue_blake3,
            "selection_blake3": self.selection_blake3,
            "candidate_id": self.candidate_id,
            "source_candidate_id": self.source_candidate_id,
            "campaign_ordinal": self.campaign_ordinal,
            "session_id": self.session_id,
            "claim_blake3": self.claim_blake3,
            "status": self.status,
            "logical_provider_calls_started": self.logical_provider_calls_started,
            "semantic_retry_count": self.semantic_retry_count,
            "error_code": self.error_code,
            "failure_receipt_blake3s": self.failure_receipt_blake3s,
            "publication_receipt": self.publication_receipt,
            "binding_blake3": self.binding_blake3,
            "validator_authority_blake3": self.validator_authority_blake3,
            "executable_material_blake3": self.executable_material_blake3,
            "completed_at_utc": self.completed_at_utc,
            "exception_text_recorded": self.exception_text_recorded,
            "supervisor_transition_claimed": self.supervisor_transition_claimed,
        }
        if self.schema == "eva.premium-construction-terminal.v3":
            core["attempt_evidence_blake3s"] = self.attempt_evidence_blake3s
        return core

    def to_document(self) -> dict[str, Any]:
        return {**canonical_value(self.core()), "record_blake3": self.record_blake3}

    @classmethod
    def from_document(cls, value: Any) -> "PremiumConstructionTerminalRecord":
        if not isinstance(value, Mapping):
            raise PremiumConstructionCampaignError("construction terminal keys differ")
        expected = set(cls.__dataclass_fields__)
        legacy_expected = expected - {"attempt_evidence_blake3s"}
        if set(value) == legacy_expected and value.get("schema") in {
            "eva.premium-construction-terminal.v1",
            "eva.premium-construction-terminal.v2",
        }:
            fields = {key: value[key] for key in legacy_expected}
            fields["attempt_evidence_blake3s"] = ()
        elif set(value) == expected:
            fields = {key: value[key] for key in expected}
        else:
            raise PremiumConstructionCampaignError("construction terminal keys differ")
        fields["failure_receipt_blake3s"] = tuple(fields["failure_receipt_blake3s"])
        fields["attempt_evidence_blake3s"] = tuple(
            fields["attempt_evidence_blake3s"]
        )
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class PremiumConstructionQueue:
    schema: str
    queue_id: str
    selection_id: str
    selection_blake3: str
    readiness_authority_blake3: str
    entries: tuple[PremiumConstructionQueueEntry, ...]
    consumed_record_blake3s: tuple[str, ...]
    frozen_candidate_count_before_exclusion: int
    bound_excluded_count: int
    quarantined_excluded_count: int
    queue_blake3: str

    def __post_init__(self) -> None:
        if self.schema != "eva.premium-construction-queue.v1":
            raise PremiumConstructionCampaignError("premium construction queue schema differs")
        uuid_text(self.queue_id, label="premium construction queue_id")
        uuid_text(self.selection_id, label="premium construction selection_id")
        if not is_blake3(self.selection_blake3) or not is_blake3(self.readiness_authority_blake3):
            raise PremiumConstructionCampaignError("premium construction selection commitment differs")
        entries = tuple(self.entries)
        object.__setattr__(self, "entries", entries)
        if (
            tuple(row.campaign_ordinal for row in entries) != tuple(range(1, len(entries) + 1))
            or tuple(row.selection_queue_ordinal for row in entries)
            != tuple(sorted(row.selection_queue_ordinal for row in entries))
            or len({row.candidate_id for row in entries}) != len(entries)
        ):
            raise PremiumConstructionCampaignError("premium construction queue order differs")
        consumed = _digest_tuple(self.consumed_record_blake3s, label="consumed construction record")
        if consumed != tuple(sorted(set(consumed))):
            raise PremiumConstructionCampaignError("consumed construction records differ")
        object.__setattr__(self, "consumed_record_blake3s", consumed)
        for count in (
            self.frozen_candidate_count_before_exclusion,
            self.bound_excluded_count,
            self.quarantined_excluded_count,
        ):
            if type(count) is not int or count < 0:
                raise PremiumConstructionCampaignError("premium construction queue count differs")
        if (
            len(entries) + self.bound_excluded_count + self.quarantined_excluded_count
            != self.frozen_candidate_count_before_exclusion
            or self.queue_blake3 != blake3_hex(self.core())
        ):
            raise PremiumConstructionCampaignError("premium construction queue commitment differs")

    def core(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "queue_id": self.queue_id,
            "selection_id": self.selection_id,
            "selection_blake3": self.selection_blake3,
            "readiness_authority_blake3": self.readiness_authority_blake3,
            "entries": tuple(row.to_document() for row in self.entries),
            "consumed_record_blake3s": self.consumed_record_blake3s,
            "frozen_candidate_count_before_exclusion": self.frozen_candidate_count_before_exclusion,
            "bound_excluded_count": self.bound_excluded_count,
            "quarantined_excluded_count": self.quarantined_excluded_count,
            "controls": {
                "selection_readiness": "frozen",
                "ordering": "selection_v2_queue_ordinal_after_consumed_exclusion",
                "outcome_fields_read_for_ordering": False,
                "semantic_retry_count": 0,
                "supervisor_transition_claimed": False,
            },
        }

    def to_document(self) -> dict[str, Any]:
        return {**canonical_value(self.core()), "queue_blake3": self.queue_blake3}

    @classmethod
    def from_document(cls, value: Any) -> "PremiumConstructionQueue":
        if not isinstance(value, Mapping):
            raise PremiumConstructionCampaignError("premium construction queue must be an object")
        document = dict(value)
        controls = document.pop("controls", None)
        if controls != {
            "selection_readiness": "frozen",
            "ordering": "selection_v2_queue_ordinal_after_consumed_exclusion",
            "outcome_fields_read_for_ordering": False,
            "semantic_retry_count": 0,
            "supervisor_transition_claimed": False,
        }:
            raise PremiumConstructionCampaignError("premium construction queue controls differ")
        if set(document) != set(cls.__dataclass_fields__):
            raise PremiumConstructionCampaignError("premium construction queue keys differ")
        document["entries"] = tuple(
            PremiumConstructionQueueEntry.from_document(row) for row in document["entries"]
        )
        document["consumed_record_blake3s"] = tuple(document["consumed_record_blake3s"])
        return cls(**document)


def build_premium_construction_queue(
    selection: CampaignSelectionV2,
    *,
    consumed_records: Sequence[PremiumConstructionTerminalRecord] = (),
) -> PremiumConstructionQueue:
    """Freeze the selection-v2 ``frozen``/unconsumed prefix without outcomes."""

    if not isinstance(selection, CampaignSelectionV2):
        raise PremiumConstructionCampaignError("verified selection-v2 object is required")
    CampaignSelectionV2.from_document(selection.to_document())
    consumed_by_candidate: dict[str, PremiumConstructionTerminalRecord] = {}
    for record in consumed_records:
        if not isinstance(record, PremiumConstructionTerminalRecord):
            raise PremiumConstructionCampaignError("consumed construction record differs")
        if record.selection_blake3 != selection.selection_blake3:
            raise PremiumConstructionCampaignError("consumed construction selection differs")
        if record.candidate_id in consumed_by_candidate:
            raise PremiumConstructionCampaignError("consumed construction candidate is duplicated")
        consumed_by_candidate[record.candidate_id] = record
    frozen = tuple(row for row in selection.entries if row.construction_readiness == "frozen")
    unknown = set(consumed_by_candidate) - {row.candidate_id for row in frozen}
    if unknown:
        raise PremiumConstructionCampaignError("consumed candidate is not frozen in selection-v2")
    remaining = tuple(row for row in frozen if row.candidate_id not in consumed_by_candidate)
    entries = tuple(
        PremiumConstructionQueueEntry.create(row, campaign_ordinal=index)
        for index, row in enumerate(remaining, 1)
    )
    consumed_digests = tuple(sorted(record.record_blake3 for record in consumed_records))
    readiness_digest = selection.readiness_authority.get("authority_blake3")
    if not is_blake3(readiness_digest):
        raise PremiumConstructionCampaignError("selection-v2 readiness authority differs")
    identity_core = {
        "schema": "eva.premium-construction-queue-identity.v1",
        "selection_blake3": selection.selection_blake3,
        "consumed_record_blake3s": consumed_digests,
        "entry_blake3s": tuple(row.entry_blake3 for row in entries),
    }
    queue_id = str(uuid5(NAMESPACE_URL, f"eva-agent:{blake3_hex(identity_core)}"))
    fields = {
        "schema": "eva.premium-construction-queue.v1",
        "queue_id": queue_id,
        "selection_id": selection.selection_id,
        "selection_blake3": selection.selection_blake3,
        "readiness_authority_blake3": readiness_digest,
        "entries": entries,
        "consumed_record_blake3s": consumed_digests,
        "frozen_candidate_count_before_exclusion": len(frozen),
        "bound_excluded_count": sum(
            record.status == "succeeded" for record in consumed_records
        ),
        "quarantined_excluded_count": sum(
            record.status == "quarantined" for record in consumed_records
        ),
    }
    core = {
        **fields,
        "entries": tuple(row.to_document() for row in entries),
        "controls": {
            "selection_readiness": "frozen",
            "ordering": "selection_v2_queue_ordinal_after_consumed_exclusion",
            "outcome_fields_read_for_ordering": False,
            "semantic_retry_count": 0,
            "supervisor_transition_claimed": False,
        },
    }
    return PremiumConstructionQueue(**fields, queue_blake3=blake3_hex(core))


@dataclass(frozen=True, slots=True)
class PremiumConstructionCampaignConfig:
    worker_width: int
    app_server_shards: int = _APP_SERVER_SHARDS
    progress_seconds: float = 2.0
    max_candidates: int | None = None

    def __post_init__(self) -> None:
        if type(self.worker_width) is not int or self.worker_width not in _ALLOWED_WORKER_WIDTHS:
            raise PremiumConstructionCampaignError("worker_width must be one of 1/128/256/512")
        if type(self.app_server_shards) is not int or self.app_server_shards != _APP_SERVER_SHARDS:
            raise PremiumConstructionCampaignError("premium construction requires 64 app-server shards")
        if type(self.progress_seconds) not in {int, float} or not 0 < self.progress_seconds <= 60:
            raise PremiumConstructionCampaignError("progress_seconds must be in (0,60]")
        if self.max_candidates is not None and (
            type(self.max_candidates) is not int or self.max_candidates < 1
        ):
            raise PremiumConstructionCampaignError("max_candidates must be positive")

    def to_document(self) -> dict[str, Any]:
        return {
            "worker_width": self.worker_width,
            "app_server_shards": self.app_server_shards,
            "progress_seconds": float(self.progress_seconds),
            "max_candidates": self.max_candidates,
            "maximum_simultaneous_logical_provider_calls": 4 * self.worker_width,
            "per_candidate_parallel_frontiers": [2, 4, 1],
            "semantic_retry_count": 0,
        }


@dataclass(frozen=True, slots=True)
class PremiumConstructionPreflightReceipt:
    schema: str
    queue_id: str
    queue_blake3: str
    route_catalog_blake3: str
    worker_width: int
    phase_plan: tuple[Mapping[str, JsonValue], ...]
    candidate_proofs: tuple[Mapping[str, JsonValue], ...]
    provider_call_count: int
    receipt_blake3: str

    def core(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "queue_id": self.queue_id,
            "queue_blake3": self.queue_blake3,
            "route_catalog_blake3": self.route_catalog_blake3,
            "worker_width": self.worker_width,
            "phase_plan": self.phase_plan,
            "candidate_proofs": self.candidate_proofs,
            "provider_call_count": self.provider_call_count,
            "controls": {
                "source_and_author_requests_reopened": True,
                "output_schemas_unchanged": True,
                "runner_opened": False,
                "provider_calls": 0,
                "source_reopen_thread": "caller-main-thread",
            },
        }

    def __post_init__(self) -> None:
        uuid_text(self.queue_id, label="preflight queue_id")
        if self.schema not in {
            "eva.premium-construction-preflight.v1",
            "eva.premium-construction-preflight.v2",
            "eva.premium-construction-preflight.v3",
        }:
            raise PremiumConstructionCampaignError("construction preflight schema differs")
        if not is_blake3(self.queue_blake3) or not is_blake3(self.route_catalog_blake3):
            raise PremiumConstructionCampaignError("construction preflight commitment differs")
        proofs = tuple(freeze_json(value) for value in self.candidate_proofs)
        if any(not isinstance(value, Mapping) for value in proofs):
            raise PremiumConstructionCampaignError("construction preflight proof differs")
        object.__setattr__(self, "candidate_proofs", proofs)
        phase_plan = tuple(freeze_json(value) for value in self.phase_plan)
        if self.schema == "eva.premium-construction-preflight.v1":
            expected_lanes = (
                "opus5_draft",
                "gemini_alternate",
                "opus48_critique",
                "gpt56_comparison",
                "opus5_revision",
            )
            expected_frontiers = (1, 1, 2, 2, 3)
            expected_roles = (
                "strong_actor",
                "strong_actor",
                "middle_actor",
                "strong_actor",
                "strong_actor",
            )
        else:
            expected_lanes = tuple(
                lane.value
                for lane in (
                    V4_LANE_ORDER
                    if self.schema == "eva.premium-construction-preflight.v3"
                    else V3_LANE_ORDER
                )
            )
            expected_frontiers = (1, 1, 2, 2, 2, 2, 3)
            expected_roles = (
                "strong_actor",
                "strong_actor",
                "middle_actor",
                "strong_actor",
                "strong_actor",
                "strong_actor",
                "strong_actor",
            )
        expected_keys = {
            "lane",
            "frontier",
            "route_id",
            "model",
            "provider",
            "role",
            "workspace_mode",
            "semantic_attempt_count",
            "retry_count",
        }
        if not (
            len(phase_plan) == len(expected_lanes)
            and all(isinstance(row, Mapping) for row in phase_plan)
            and all(set(row) == expected_keys for row in phase_plan)
            and tuple(row.get("lane") for row in phase_plan) == expected_lanes
            and tuple(row.get("frontier") for row in phase_plan)
            == expected_frontiers
            and tuple(row.get("workspace_mode") for row in phase_plan)
            == ("read-only",) * len(expected_lanes)
            and tuple(row.get("role") for row in phase_plan) == expected_roles
            and all(
                isinstance(row.get(field), str) and row.get(field)
                for row in phase_plan
                for field in ("route_id", "model", "provider")
            )
            and all(row.get("semantic_attempt_count") == 1 for row in phase_plan)
            and all(row.get("retry_count") == 0 for row in phase_plan)
            and all(
                phase_plan[0].get(field)
                == phase_plan[-1].get(field)
                for field in ("route_id", "model", "provider")
            )
            and (
                self.schema == "eva.premium-construction-preflight.v1"
                or self.schema == "eva.premium-construction-preflight.v3"
                and all(
                    phase_plan[0].get(field) == phase_plan[3].get(field)
                    and phase_plan[0].get(field) == phase_plan[5].get(field)
                    for field in ("route_id", "model", "provider")
                )
                or all(
                    phase_plan[0].get(field) == phase_plan[3].get(field)
                    and phase_plan[1].get(field) == phase_plan[5].get(field)
                    for field in ("route_id", "model", "provider")
                )
            )
        ):
            raise PremiumConstructionCampaignError("construction preflight phase plan differs")
        projected_routes_list: list[dict[str, Any]] = []
        observed_route_ids: set[Any] = set()
        for row in phase_plan:
            if row["route_id"] in observed_route_ids:
                continue
            observed_route_ids.add(row["route_id"])
            projected_routes_list.append(
                {
                    "route_id": row["route_id"],
                    "model": row["model"],
                    "provider": row["provider"],
                }
            )
        projected_routes = tuple(projected_routes_list)
        if self.route_catalog_blake3 != blake3_hex(projected_routes):
            raise PremiumConstructionCampaignError("construction preflight route catalog differs")
        object.__setattr__(self, "phase_plan", phase_plan)
        if (
            self.worker_width not in _ALLOWED_WORKER_WIDTHS
            or self.provider_call_count != 0
            or self.receipt_blake3 != blake3_hex(self.core())
        ):
            raise PremiumConstructionCampaignError("construction preflight receipt differs")

    def to_document(self) -> dict[str, Any]:
        return {**canonical_value(self.core()), "receipt_blake3": self.receipt_blake3}

    @classmethod
    def from_document(cls, value: Any) -> "PremiumConstructionPreflightReceipt":
        if not isinstance(value, Mapping):
            raise PremiumConstructionCampaignError("construction preflight must be an object")
        document = dict(value)
        controls = document.pop("controls", None)
        if controls != {
            "source_and_author_requests_reopened": True,
            "output_schemas_unchanged": True,
            "runner_opened": False,
            "provider_calls": 0,
            "source_reopen_thread": "caller-main-thread",
        }:
            raise PremiumConstructionCampaignError("construction preflight controls differ")
        if set(document) != set(cls.__dataclass_fields__):
            raise PremiumConstructionCampaignError("construction preflight keys differ")
        document["phase_plan"] = tuple(document["phase_plan"])
        document["candidate_proofs"] = tuple(document["candidate_proofs"])
        return cls(**document)


class _CompositionFactoryPort(Protocol):
    selection: CampaignSelectionV2

    def compose(
        self, candidate_id: str, routes: PremiumConstructionRoutes
    ) -> LegacyConstructionComposition: ...


class _RunnerPort(Protocol):
    shard_count: int

    def run_once(self, options: Any, turn_input: Any) -> Any: ...


def _routes_document(routes: PremiumConstructionRoutes) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "route_id": route.route_id,
            "model": route.model,
            "provider": route.provider,
        }
        for route in (routes.opus5, routes.gemini31, routes.opus48, routes.gpt56)
    )


def _phase_plan_document(
    routes: PremiumConstructionRoutes,
    *,
    result_schema: str = PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
) -> tuple[Mapping[str, JsonValue], ...]:
    lanes = (
        (ConstructionLane.OPUS5_DRAFT, 1, "strong_actor", "read-only"),
        (ConstructionLane.GEMINI_ALTERNATE, 1, "strong_actor", "read-only"),
        (ConstructionLane.OPUS48_CRITIQUE, 2, "middle_actor", "read-only"),
        (ConstructionLane.OPUS5_CRITIQUE_BACKUP, 2, "strong_actor", "read-only"),
        (ConstructionLane.GPT56_COMPARISON, 2, "strong_actor", "read-only"),
        (ConstructionLane.GEMINI_COMPARISON_BACKUP, 2, "strong_actor", "read-only"),
        (ConstructionLane.OPUS5_REVISION, 3, "strong_actor", "read-only"),
    )
    return tuple(
        freeze_json(
            {
                "lane": lane.value,
                "frontier": frontier,
                "route_id": (
                    routes.opus5
                    if result_schema == PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4
                    and lane is ConstructionLane.GEMINI_COMPARISON_BACKUP
                    else routes.for_lane(lane)
                ).route_id,
                "model": (
                    routes.opus5
                    if result_schema == PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4
                    and lane is ConstructionLane.GEMINI_COMPARISON_BACKUP
                    else routes.for_lane(lane)
                ).model,
                "provider": (
                    routes.opus5
                    if result_schema == PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4
                    and lane is ConstructionLane.GEMINI_COMPARISON_BACKUP
                    else routes.for_lane(lane)
                ).provider,
                "role": role,
                "workspace_mode": workspace,
                "semantic_attempt_count": 1,
                "retry_count": 0,
            }
        )
        for lane, frontier, role, workspace in lanes
    )


def preflight_premium_construction_campaign(
    *,
    queue: PremiumConstructionQueue,
    factory: _CompositionFactoryPort,
    routes: PremiumConstructionRoutes,
    config: PremiumConstructionCampaignConfig,
    candidate_ids: Iterable[str] | None = None,
    result_schema: str = PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
) -> PremiumConstructionPreflightReceipt:
    """Reopen exact sources and author requests without opening the runner."""

    if factory.selection.selection_blake3 != queue.selection_blake3:
        raise PremiumConstructionCampaignError("construction factory selection differs")
    by_id = {entry.candidate_id: entry for entry in queue.entries}
    identities = tuple(by_id) if candidate_ids is None else tuple(candidate_ids)
    if len(identities) != len(set(identities)) or any(value not in by_id for value in identities):
        raise PremiumConstructionCampaignError("construction preflight candidate set differs")
    entries = tuple(sorted((by_id[value] for value in identities), key=lambda row: row.campaign_ordinal))

    def inspect(entry: PremiumConstructionQueueEntry) -> Mapping[str, JsonValue]:
        composition = factory.compose(entry.candidate_id, routes)
        if not isinstance(composition.source, FrozenConstructionSource):
            raise PremiumConstructionCampaignError("construction source composition differs")
        requests = tuple(composition.adapter.prepare_authors(composition.source, routes))
        if (
            len(requests) != 2
            or tuple(request.lane for request in requests) != AUTHOR_LANES
            or any(not isinstance(request, ConstructionTurnRequest) for request in requests)
            or any(request.source_request_blake3 != composition.source.request_blake3 for request in requests)
            or any(request.dependencies for request in requests)
            or len({request.options.cwd for request in requests}) != 2
        ):
            raise PremiumConstructionCampaignError("construction author preflight differs")
        return freeze_json(
            {
                "candidate_id": entry.candidate_id,
                "entry_blake3": entry.entry_blake3,
                "source_request_blake3": composition.source.request_blake3,
                "author_request_blake3s": [request.request_blake3 for request in requests],
                "author_output_schema_blake3s": [
                    request.source_output_schema_blake3 for request in requests
                ],
            }
        )

    # The authoritative legacy snapshotter coordinates a process-wide Linux
    # read lease and therefore must execute on the caller's main thread.
    # Provider work remains high-width and is a separate boundary.
    proofs = tuple(inspect(entry) for entry in entries)
    core = {
        "schema": (
            "eva.premium-construction-preflight.v3"
            if result_schema == PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4
            else "eva.premium-construction-preflight.v2"
        ),
        "queue_id": queue.queue_id,
        "queue_blake3": queue.queue_blake3,
        "route_catalog_blake3": blake3_hex(_routes_document(routes)),
        "worker_width": config.worker_width,
        "phase_plan": _phase_plan_document(routes, result_schema=result_schema),
        "candidate_proofs": proofs,
        "provider_call_count": 0,
        "controls": {
            "source_and_author_requests_reopened": True,
            "output_schemas_unchanged": True,
            "runner_opened": False,
            "provider_calls": 0,
            "source_reopen_thread": "caller-main-thread",
        },
    }
    return PremiumConstructionPreflightReceipt(
        schema=core["schema"],
        queue_id=core["queue_id"],
        queue_blake3=core["queue_blake3"],
        route_catalog_blake3=core["route_catalog_blake3"],
        worker_width=core["worker_width"],
        phase_plan=core["phase_plan"],
        candidate_proofs=proofs,
        provider_call_count=0,
        receipt_blake3=blake3_hex(core),
    )


def _receipt_from_document(value: Mapping[str, Any]) -> ConstructionPublicationReceipt:
    return ConstructionPublicationReceipt(
        schema=value["schema"],
        publication_id=value["publication_id"],
        sink_name=value["sink_name"],
        construction_receipt_blake3=value["construction_receipt_blake3"],
        artifact_blake3=value["artifact_blake3"],
        legacy_manifest_compatibility_verified=value[
            "legacy_manifest_compatibility_verified"
        ],
        metadata=value["metadata"],
        receipt_blake3=value["receipt_blake3"],
    )


@dataclass(frozen=True, slots=True)
class PremiumConstructionProofCatalog:
    schema: str
    catalog_id: str
    queue_id: str
    queue_blake3: str
    selection_blake3: str
    records: tuple[PremiumConstructionTerminalRecord, ...]
    succeeded: int
    quarantined: int
    pending_claimed: int
    unclaimed: int
    created_at_utc: str
    catalog_blake3: str

    def core(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "catalog_id": self.catalog_id,
            "queue_id": self.queue_id,
            "queue_blake3": self.queue_blake3,
            "selection_blake3": self.selection_blake3,
            "records": tuple(record.to_document() for record in self.records),
            "succeeded": self.succeeded,
            "quarantined": self.quarantined,
            "pending_claimed": self.pending_claimed,
            "unclaimed": self.unclaimed,
            "created_at_utc": self.created_at_utc,
            "controls": {
                "catalog_is_supervisor_transition": False,
                "admission_eligibility_claimed": False,
                "publication_roots_recorded": False,
                "append_only_terminal_records": True,
            },
        }

    def __post_init__(self) -> None:
        if self.schema != "eva.premium-construction-proof-catalog.v1":
            raise PremiumConstructionCampaignError("construction proof catalog schema differs")
        uuid_text(self.catalog_id, label="construction catalog_id")
        uuid_text(self.queue_id, label="construction catalog queue_id")
        if not is_blake3(self.queue_blake3) or not is_blake3(self.selection_blake3):
            raise PremiumConstructionCampaignError("construction catalog commitment differs")
        records = tuple(self.records)
        object.__setattr__(self, "records", records)
        if (
            self.succeeded != sum(record.status == "succeeded" for record in records)
            or self.quarantined != sum(record.status == "quarantined" for record in records)
            or any(type(value) is not int or value < 0 for value in (self.pending_claimed, self.unclaimed))
            or self.catalog_blake3 != blake3_hex(self.core())
        ):
            raise PremiumConstructionCampaignError("construction proof catalog differs")
        _now(lambda: self.created_at_utc)

    def to_document(self) -> dict[str, Any]:
        return {**canonical_value(self.core()), "catalog_blake3": self.catalog_blake3}

    @classmethod
    def from_document(cls, value: Any) -> "PremiumConstructionProofCatalog":
        if not isinstance(value, Mapping):
            raise PremiumConstructionCampaignError("construction proof catalog must be an object")
        document = dict(value)
        controls = document.pop("controls", None)
        if controls != {
            "catalog_is_supervisor_transition": False,
            "admission_eligibility_claimed": False,
            "publication_roots_recorded": False,
            "append_only_terminal_records": True,
        }:
            raise PremiumConstructionCampaignError("construction proof catalog controls differ")
        if set(document) != set(cls.__dataclass_fields__):
            raise PremiumConstructionCampaignError("construction proof catalog keys differ")
        document["records"] = tuple(
            PremiumConstructionTerminalRecord.from_document(record)
            for record in document["records"]
        )
        return cls(**document)


def verify_premium_construction_proof_catalog(
    catalog: PremiumConstructionProofCatalog,
    *,
    queue: PremiumConstructionQueue,
    publication_output_root: str | Path | None = None,
) -> Mapping[str, JsonValue]:
    """Return the exact successful binding allowlist input, without promoting it."""

    if not isinstance(catalog, PremiumConstructionProofCatalog):
        raise PremiumConstructionCampaignError("construction proof catalog type differs")
    if not (
        catalog.queue_id == queue.queue_id
        and catalog.queue_blake3 == queue.queue_blake3
        and catalog.selection_blake3 == queue.selection_blake3
        and catalog.catalog_blake3 == blake3_hex(catalog.core())
    ):
        raise PremiumConstructionCampaignError("construction proof catalog queue differs")
    entries = {entry.candidate_id: entry for entry in queue.entries}
    ordinals = []
    allowlist: list[Mapping[str, JsonValue]] = []
    for record in catalog.records:
        entry = entries.get(record.candidate_id)
        if not (
            entry is not None
            and record.source_candidate_id == entry.source_candidate_id
            and record.campaign_ordinal == entry.campaign_ordinal
            and record.queue_id == queue.queue_id
            and record.queue_blake3 == queue.queue_blake3
            and record.selection_blake3 == queue.selection_blake3
            and record.record_blake3 == blake3_hex(record.core())
        ):
            raise PremiumConstructionCampaignError("construction terminal queue binding differs")
        ordinals.append(record.campaign_ordinal)
        if record.status != "succeeded":
            continue
        assert record.publication_receipt is not None
        receipt = _receipt_from_document(record.publication_receipt)
        if not (
            receipt.legacy_manifest_compatibility_verified is False
            and receipt.metadata.get("binding_blake3") == record.binding_blake3
            and receipt.metadata.get("validator_authority_blake3")
            == record.validator_authority_blake3
            and receipt.metadata.get("executable_material_blake3")
            == record.executable_material_blake3
            and receipt.metadata.get("append_only") is True
            and receipt.metadata.get("retry_count") == 0
        ):
            raise PremiumConstructionCampaignError("construction terminal binding differs")
        if publication_output_root is not None:
            verify_legacy_construction_publication(publication_output_root, receipt)
        allowlist.append(
            freeze_json(
                {
                    "candidate_id": record.candidate_id,
                    "source_candidate_id": record.source_candidate_id,
                    "selection_blake3": record.selection_blake3,
                    "record_blake3": record.record_blake3,
                    "publication_id": receipt.publication_id,
                    "publication_receipt_blake3": receipt.receipt_blake3,
                    "publication_artifact_blake3": receipt.artifact_blake3,
                    "binding_blake3": record.binding_blake3,
                    "validator_authority_blake3": record.validator_authority_blake3,
                    "executable_material_blake3": record.executable_material_blake3,
                }
            )
        )
    if ordinals != sorted(set(ordinals)):
        raise PremiumConstructionCampaignError("construction catalog record order differs")
    if len(catalog.records) + catalog.pending_claimed + catalog.unclaimed != len(queue.entries):
        raise PremiumConstructionCampaignError("construction catalog inventory differs")
    return freeze_json(
        {
            "schema": "eva.premium-construction-supervisor-input.v1",
            "catalog_id": catalog.catalog_id,
            "catalog_blake3": catalog.catalog_blake3,
            "queue_blake3": queue.queue_blake3,
            "selection_blake3": queue.selection_blake3,
            "successful_bindings": allowlist,
            "successful_binding_count": len(allowlist),
            "supervisor_transition_authorized": False,
            "admission_eligibility_claimed": False,
        }
    )


def _verify_claim(
    claim: Mapping[str, Any],
    *,
    queue: PremiumConstructionQueue,
    entry: PremiumConstructionQueueEntry,
) -> None:
    expected = {
        "schema",
        "claim_id",
        "session_id",
        "queue_id",
        "queue_blake3",
        "selection_blake3",
        "candidate_id",
        "source_candidate_id",
        "campaign_ordinal",
        "entry_blake3",
        "started_at_utc",
        "semantic_retry_count",
        "claim_blake3",
    }
    if not isinstance(claim, Mapping) or set(claim) != expected:
        raise PremiumConstructionCampaignError("construction claim keys differ")
    core = {key: claim[key] for key in expected if key != "claim_blake3"}
    for value, label in (
        (claim["claim_id"], "construction claim_id"),
        (claim["session_id"], "construction claim session_id"),
    ):
        uuid_text(value, label=label)
    if not (
        claim["schema"] == "eva.premium-construction-claim.v1"
        and claim["queue_id"] == queue.queue_id
        and claim["queue_blake3"] == queue.queue_blake3
        and claim["selection_blake3"] == queue.selection_blake3
        and claim["candidate_id"] == entry.candidate_id
        and claim["source_candidate_id"] == entry.source_candidate_id
        and claim["campaign_ordinal"] == entry.campaign_ordinal
        and claim["entry_blake3"] == entry.entry_blake3
        and claim["semantic_retry_count"] == 0
        and claim["claim_blake3"] == blake3_hex(core)
    ):
        raise PremiumConstructionCampaignError("construction claim commitment differs")
    _now(lambda: claim["started_at_utc"])


class _AppendOnlyCampaignStore:
    def __init__(self, root: str | Path, queue: PremiumConstructionQueue) -> None:
        self.root = _ensure_directory(root, label="construction campaign root")
        self.queue = queue
        for name in (
            "claims",
            "terminal",
            "failures",
            "attempts",
            "catalogs",
            "sessions",
        ):
            _ensure_directory(self.root / name, label=f"construction campaign {name}")
        queue_path = self.root / "queue.v1.json"
        try:
            metadata = queue_path.lstat()
        except FileNotFoundError:
            _write_new_relative(
                self.root, "queue.v1.json", canonical_json_bytes(queue.to_document())
            )
        else:
            if not os.path.isfile(queue_path) or os.path.islink(queue_path) or metadata.st_nlink != 1:
                raise PremiumConstructionCampaignError("construction queue topology differs")
            observed = _decode_object(
                _read_relative(
                    self.root,
                    _safe_relative("queue.v1.json", label="construction queue path"),
                    label="construction queue",
                    maximum_bytes=_MAX_CAMPAIGN_DOCUMENT_BYTES,
                ),
                label="construction queue",
            )
            reopened = PremiumConstructionQueue.from_document(observed)
            if reopened != queue:
                raise PremiumConstructionCampaignError("installed construction queue differs")

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        descriptor = _open_directory_chain(self.root, create=False)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise PremiumConstructionCampaignError(
                    "another construction campaign process owns this root"
                ) from None
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _read_directory_documents(self, directory: str) -> dict[str, Mapping[str, Any]]:
        root = _existing_directory(self.root / directory, label=f"construction {directory}")
        descriptor = _open_directory_chain(root, create=False)
        try:
            names = sorted(os.listdir(descriptor))
        finally:
            os.close(descriptor)
        result: dict[str, Mapping[str, Any]] = {}
        for name in names:
            if not name.endswith(".json"):
                raise PremiumConstructionCampaignError("construction state filename differs")
            result[name[:-5]] = _decode_object(
                _read_relative(
                    root,
                    _safe_relative(name, label="construction state path"),
                    label="construction state",
                    maximum_bytes=_MAX_CAMPAIGN_DOCUMENT_BYTES,
                ),
                label="construction state",
            )
        return result

    def claims(self) -> dict[str, Mapping[str, Any]]:
        claims = self._read_directory_documents("claims")
        if any(document.get("candidate_id") != candidate_id for candidate_id, document in claims.items()):
            raise PremiumConstructionCampaignError("construction claim filename differs")
        return claims

    def terminals(self) -> dict[str, PremiumConstructionTerminalRecord]:
        terminals = {
            candidate_id: PremiumConstructionTerminalRecord.from_document(document)
            for candidate_id, document in self._read_directory_documents("terminal").items()
        }
        if any(record.candidate_id != candidate_id for candidate_id, record in terminals.items()):
            raise PremiumConstructionCampaignError("construction terminal filename differs")
        return terminals

    def failure_receipts(self) -> dict[str, ConstructionFailureReceipt]:
        receipts = {
            digest: ConstructionFailureReceipt.from_document(document)
            for digest, document in self._read_directory_documents("failures").items()
        }
        if any(receipt.receipt_blake3 != digest for digest, receipt in receipts.items()):
            raise PremiumConstructionCampaignError(
                "construction failure receipt filename differs"
            )
        for receipt in receipts.values():
            verify_construction_failure_receipt(receipt)
        return receipts

    def attempt_evidence(self) -> dict[str, ConstructionAttemptEvidence]:
        evidence = {
            digest: ConstructionAttemptEvidence.from_document(document)
            for digest, document in self._read_directory_documents("attempts").items()
        }
        if any(item.evidence_blake3 != digest for digest, item in evidence.items()):
            raise PremiumConstructionCampaignError(
                "construction attempt evidence filename differs"
            )
        for item in evidence.values():
            verify_construction_attempt_evidence(item)
        return evidence

    def start_session(
        self,
        session_id: str,
        *,
        clock: Callable[[], str],
        config: PremiumConstructionCampaignConfig,
        routes: PremiumConstructionRoutes,
        result_schema: str = PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
    ) -> None:
        phase_plan = _phase_plan_document(routes, result_schema=result_schema)
        core = {
            "schema": "eva.premium-construction-session-start.v2",
            "session_id": session_id,
            "queue_id": self.queue.queue_id,
            "queue_blake3": self.queue.queue_blake3,
            "started_at_utc": _now(clock),
            "config": config.to_document(),
            "phase_plan": phase_plan,
            "phase_plan_blake3": blake3_hex(phase_plan),
            "credential_values_recorded": False,
        }
        _write_new_relative(
            self.root,
            f"sessions/{session_id}.start.json",
            canonical_json_bytes({**core, "session_blake3": blake3_hex(core)}),
        )

    def finish_session(self, session_id: str, report_core: Mapping[str, Any]) -> None:
        core = {
            "schema": "eva.premium-construction-session-finish.v1",
            "session_id": session_id,
            "queue_id": self.queue.queue_id,
            "report": report_core,
        }
        _write_new_relative(
            self.root,
            f"sessions/{session_id}.finish.json",
            canonical_json_bytes({**core, "session_blake3": blake3_hex(core)}),
        )

    def claim(
        self,
        entry: PremiumConstructionQueueEntry,
        *,
        session_id: str,
        clock: Callable[[], str],
    ) -> Mapping[str, Any] | None:
        core = {
            "schema": "eva.premium-construction-claim.v1",
            "claim_id": str(uuid5(NAMESPACE_URL, f"{session_id}:{entry.candidate_id}")),
            "session_id": session_id,
            "queue_id": self.queue.queue_id,
            "queue_blake3": self.queue.queue_blake3,
            "selection_blake3": self.queue.selection_blake3,
            "candidate_id": entry.candidate_id,
            "source_candidate_id": entry.source_candidate_id,
            "campaign_ordinal": entry.campaign_ordinal,
            "entry_blake3": entry.entry_blake3,
            "started_at_utc": _now(clock),
            "semantic_retry_count": 0,
        }
        document = {**core, "claim_blake3": blake3_hex(core)}
        _verify_claim(document, queue=self.queue, entry=entry)
        try:
            _write_new_relative(
                self.root,
                f"claims/{entry.candidate_id}.json",
                canonical_json_bytes(document),
            )
        except Exception as exc:
            claim_path = self.root / "claims" / f"{entry.candidate_id}.json"
            try:
                metadata = claim_path.lstat()
            except FileNotFoundError:
                raise PremiumConstructionCampaignError("construction claim failed") from exc
            if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                return None
            raise PremiumConstructionCampaignError("construction claim topology differs") from exc
        return MappingProxyType(document)

    def write_terminal(self, record: PremiumConstructionTerminalRecord) -> None:
        _write_new_relative(
            self.root,
            f"terminal/{record.candidate_id}.json",
            canonical_json_bytes(record.to_document()),
        )

    def write_failure_receipts(
        self, receipts: Sequence[ConstructionFailureReceipt]
    ) -> None:
        for receipt in receipts:
            verify_construction_failure_receipt(receipt)
            _write_new_relative(
                self.root,
                f"failures/{receipt.receipt_blake3}.json",
                canonical_json_bytes(receipt.to_document()),
            )

    def write_attempt_evidence(
        self, evidence: Sequence[ConstructionAttemptEvidence]
    ) -> None:
        for item in evidence:
            verify_construction_attempt_evidence(item)
            _write_new_relative(
                self.root,
                f"attempts/{item.evidence_blake3}.json",
                canonical_json_bytes(item.to_document()),
            )

    def recover_abandoned(
        self,
        *,
        session_id: str,
        id_factory: RuntimeIdFactory,
        clock: Callable[[], str],
    ) -> tuple[PremiumConstructionTerminalRecord, ...]:
        terminals = self.terminals()
        entries = {entry.candidate_id: entry for entry in self.queue.entries}
        recovered: list[PremiumConstructionTerminalRecord] = []
        for candidate_id, claim in self.claims().items():
            entry = entries.get(candidate_id)
            if entry is None:
                raise PremiumConstructionCampaignError("abandoned construction claim differs")
            _verify_claim(claim, queue=self.queue, entry=entry)
            if candidate_id in terminals:
                continue
            recovered.append(
                _terminal_record(
                    entry=entry,
                    queue=self.queue,
                    session_id=session_id,
                    claim=claim,
                    id_factory=id_factory,
                    clock=clock,
                    status="quarantined",
                    logical_calls=None,
                    error_code="abandoned_after_process_loss",
                )
            )
            self.write_terminal(recovered[-1])
        return tuple(recovered)

    def write_catalog(
        self,
        *,
        id_factory: RuntimeIdFactory,
        clock: Callable[[], str],
    ) -> PremiumConstructionProofCatalog:
        terminals = self.terminals()
        claims = self.claims()
        attempts = self.attempt_evidence()
        by_id = {entry.candidate_id: entry for entry in self.queue.entries}
        if set(terminals) - set(by_id) or set(claims) - set(by_id):
            raise PremiumConstructionCampaignError("construction state candidate differs")
        for candidate_id, claim in claims.items():
            _verify_claim(claim, queue=self.queue, entry=by_id[candidate_id])
        if any(
            candidate_id not in claims
            or record.claim_blake3 != claims[candidate_id]["claim_blake3"]
            for candidate_id, record in terminals.items()
        ):
            raise PremiumConstructionCampaignError("construction terminal claim binding differs")
        for record in terminals.values():
            # v1/v2 terminals predate per-lane attempt evidence.  Their
            # failure receipts are self-bound by the terminal contract and
            # cannot be reconstructed from an intentionally empty attempt
            # list.  The stronger cross-document equality is a v3 invariant.
            if record.schema != "eva.premium-construction-terminal.v3":
                continue
            try:
                bound_attempts = tuple(
                    attempts[digest]
                    for digest in record.attempt_evidence_blake3s
                )
            except KeyError:
                raise PremiumConstructionCampaignError(
                    "construction terminal attempt evidence is missing"
                ) from None
            failed = tuple(
                item.failure_receipt.receipt_blake3
                for item in bound_attempts
                if item.failure_receipt is not None
            )
            if failed != record.failure_receipt_blake3s:
                raise PremiumConstructionCampaignError(
                    "construction terminal failure evidence binding differs"
                )
            if (
                tuple(item.lane for item in bound_attempts)
                != tuple(
                    lane
                    for lane in V3_LANE_ORDER
                    if lane in {item.lane for item in bound_attempts}
                )
            ):
                raise PremiumConstructionCampaignError(
                    "construction terminal attempt lane order differs"
                )
        records = tuple(
            terminals[candidate_id]
            for candidate_id in sorted(
                terminals, key=lambda value: by_id[value].campaign_ordinal
            )
        )
        pending = len(set(claims) - set(terminals))
        unclaimed = len(set(by_id) - set(claims))
        catalog_id = id_factory.new("premium-construction-proof-catalog")
        core = {
            "schema": "eva.premium-construction-proof-catalog.v1",
            "catalog_id": catalog_id,
            "queue_id": self.queue.queue_id,
            "queue_blake3": self.queue.queue_blake3,
            "selection_blake3": self.queue.selection_blake3,
            "records": tuple(record.to_document() for record in records),
            "succeeded": sum(record.status == "succeeded" for record in records),
            "quarantined": sum(record.status == "quarantined" for record in records),
            "pending_claimed": pending,
            "unclaimed": unclaimed,
            "created_at_utc": _now(clock),
            "controls": {
                "catalog_is_supervisor_transition": False,
                "admission_eligibility_claimed": False,
                "publication_roots_recorded": False,
                "append_only_terminal_records": True,
            },
        }
        catalog = PremiumConstructionProofCatalog(
            schema=core["schema"],
            catalog_id=core["catalog_id"],
            queue_id=core["queue_id"],
            queue_blake3=core["queue_blake3"],
            selection_blake3=core["selection_blake3"],
            records=records,
            succeeded=core["succeeded"],
            quarantined=core["quarantined"],
            pending_claimed=pending,
            unclaimed=unclaimed,
            created_at_utc=core["created_at_utc"],
            catalog_blake3=blake3_hex(core),
        )
        _write_new_relative(
            self.root,
            f"catalogs/{catalog_id}.json",
            canonical_json_bytes(catalog.to_document()),
        )
        return catalog


def _terminal_record(
    *,
    entry: PremiumConstructionQueueEntry,
    queue: PremiumConstructionQueue,
    session_id: str,
    claim: Mapping[str, Any],
    id_factory: RuntimeIdFactory,
    clock: Callable[[], str],
    status: str,
    logical_calls: int | None,
    error_code: str | None,
    failure_receipt_blake3s: Sequence[str] = (),
    attempt_evidence_blake3s: Sequence[str] = (),
    published: PublishedConstructionResult | None = None,
) -> PremiumConstructionTerminalRecord:
    publication: Mapping[str, JsonValue] | None = None
    binding = authority = executable = None
    if published is not None:
        publication = freeze_json(canonical_value(published.publication))
        binding = published.publication.metadata.get("binding_blake3")
        authority = published.publication.metadata.get("validator_authority_blake3")
        executable = published.publication.metadata.get("executable_material_blake3")
    core = {
        "schema": "eva.premium-construction-terminal.v3",
        "record_id": id_factory.new("premium-construction-terminal"),
        "queue_id": queue.queue_id,
        "queue_blake3": queue.queue_blake3,
        "selection_blake3": queue.selection_blake3,
        "candidate_id": entry.candidate_id,
        "source_candidate_id": entry.source_candidate_id,
        "campaign_ordinal": entry.campaign_ordinal,
        "session_id": session_id,
        "claim_blake3": claim["claim_blake3"],
        "status": status,
        "logical_provider_calls_started": logical_calls,
        "semantic_retry_count": 0,
        "error_code": error_code,
        "failure_receipt_blake3s": tuple(failure_receipt_blake3s),
        "attempt_evidence_blake3s": tuple(attempt_evidence_blake3s),
        "publication_receipt": publication,
        "binding_blake3": binding,
        "validator_authority_blake3": authority,
        "executable_material_blake3": executable,
        "completed_at_utc": _now(clock),
        "exception_text_recorded": False,
        "supervisor_transition_claimed": False,
    }
    return PremiumConstructionTerminalRecord(**core, record_blake3=blake3_hex(core))


class _CountingRunner:
    def __init__(self, runner: _RunnerPort, on_call: Callable[[], None]) -> None:
        self._runner = runner
        self._on_call = on_call
        self.calls = 0

    def run_once(self, options: Any, turn_input: Any) -> Any:
        self.calls += 1
        self._on_call()
        return self._runner.run_once(options, turn_input)

    def run_construction_once(self, request: ConstructionTurnRequest) -> Any:
        """Preserve the exact constructor capability port through accounting."""

        self.calls += 1
        self._on_call()
        exact_run = getattr(self._runner, "run_construction_once", None)
        if callable(exact_run):
            return exact_run(request)
        return self._runner.run_once(request.options, request.turn_input)


class _LockedIdFactory:
    """Serialize caller-supplied UUID factories across outer workers."""

    def __init__(self, inner: RuntimeIdFactory) -> None:
        self._inner = inner
        self._lock = Lock()

    def new(self, purpose: str) -> str:
        with self._lock:
            return self._inner.new(purpose)


@dataclass(frozen=True, slots=True)
class PremiumConstructionProgress:
    schema: str
    queue_total: int
    claimed: int
    active: int
    succeeded: int
    quarantined: int
    unclaimed: int
    logical_provider_calls_started: int
    candidates_per_second: float
    eta_seconds: float | None
    worker_width: int
    app_server_shards: int


def render_premium_construction_progress(
    progress: PremiumConstructionProgress, *, width: int = 36
) -> str:
    """Render one overwrite-friendly, credential-free campaign progress line."""

    if not isinstance(progress, PremiumConstructionProgress):
        raise PremiumConstructionCampaignError("construction progress type differs")
    if type(width) is not int or not 10 <= width <= 120:
        raise PremiumConstructionCampaignError("construction progress width must be in [10,120]")
    counts = (
        progress.queue_total,
        progress.claimed,
        progress.active,
        progress.succeeded,
        progress.quarantined,
        progress.unclaimed,
        progress.logical_provider_calls_started,
    )
    if any(type(value) is not int or value < 0 for value in counts):
        raise PremiumConstructionCampaignError("construction progress count differs")
    completed = progress.succeeded + progress.quarantined
    if not (
        progress.schema == "eva.premium-construction-progress.v1"
        and progress.worker_width in _ALLOWED_WORKER_WIDTHS
        and progress.app_server_shards == _APP_SERVER_SHARDS
        and completed <= progress.claimed <= progress.queue_total
        and progress.unclaimed == progress.queue_total - progress.claimed
        and progress.active <= progress.claimed - completed
        and isinstance(progress.candidates_per_second, float)
        and math.isfinite(progress.candidates_per_second)
        and progress.candidates_per_second >= 0
        and (
            progress.eta_seconds is None
            or isinstance(progress.eta_seconds, float)
            and math.isfinite(progress.eta_seconds)
            and progress.eta_seconds >= 0
        )
    ):
        raise PremiumConstructionCampaignError("construction progress proof differs")
    fraction = 1.0 if progress.queue_total == 0 else completed / progress.queue_total
    filled = min(width, int(fraction * width))
    bar = "#" * filled + "-" * (width - filled)
    if progress.eta_seconds is None:
        eta = "--:--:--"
    else:
        seconds = int(progress.eta_seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        eta = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return (
        f"[{bar}] {completed}/{progress.queue_total} ({fraction * 100:6.2f}%) "
        f"ok={progress.succeeded} quarantine={progress.quarantined} "
        f"active={progress.active}/{progress.worker_width} "
        f"calls={progress.logical_provider_calls_started} "
        f"rate={progress.candidates_per_second:.2f}/s eta={eta} "
        f"shards={progress.app_server_shards}"
    )


@dataclass(frozen=True, slots=True)
class PremiumConstructionCampaignReport:
    schema: str
    session_id: str
    queue_id: str
    queue_blake3: str
    claimed_this_session: int
    succeeded_this_session: int
    quarantined_this_session: int
    recovered_without_retry: int
    logical_provider_calls_started: int
    elapsed_seconds: float
    proof_catalog: PremiumConstructionProofCatalog
    report_blake3: str

    def __post_init__(self) -> None:
        if self.schema not in {
            "eva.premium-construction-campaign-report.v1",
            "eva.premium-construction-campaign-report.v2",
        }:
            raise PremiumConstructionCampaignError("construction campaign report schema differs")
        calls_per_candidate = (
            5 if self.schema == "eva.premium-construction-campaign-report.v1" else 7
        )
        uuid_text(self.session_id, label="construction campaign session_id")
        uuid_text(self.queue_id, label="construction campaign queue_id")
        counts = (
            self.claimed_this_session,
            self.succeeded_this_session,
            self.quarantined_this_session,
            self.recovered_without_retry,
            self.logical_provider_calls_started,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise PremiumConstructionCampaignError("construction campaign report count differs")
        if not (
            is_blake3(self.queue_blake3)
            and self.claimed_this_session
            == self.succeeded_this_session + self.quarantined_this_session
            and self.logical_provider_calls_started
            <= calls_per_candidate * self.claimed_this_session
            and type(self.elapsed_seconds) is float
            and math.isfinite(self.elapsed_seconds)
            and self.elapsed_seconds >= 0
            and isinstance(self.proof_catalog, PremiumConstructionProofCatalog)
            and self.proof_catalog.queue_id == self.queue_id
            and self.proof_catalog.queue_blake3 == self.queue_blake3
            and self.report_blake3 == blake3_hex(self.core())
        ):
            raise PremiumConstructionCampaignError("construction campaign report differs")

    def core(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "session_id": self.session_id,
            "queue_id": self.queue_id,
            "queue_blake3": self.queue_blake3,
            "claimed_this_session": self.claimed_this_session,
            "succeeded_this_session": self.succeeded_this_session,
            "quarantined_this_session": self.quarantined_this_session,
            "recovered_without_retry": self.recovered_without_retry,
            "logical_provider_calls_started": self.logical_provider_calls_started,
            "elapsed_seconds": self.elapsed_seconds,
            "proof_catalog_blake3": self.proof_catalog.catalog_blake3,
        }

    def to_document(self) -> dict[str, Any]:
        return {**canonical_value(self.core()), "report_blake3": self.report_blake3}


class PremiumConstructionCampaign:
    """High-width execution over one immutable premium construction queue."""

    def __init__(
        self,
        *,
        queue: PremiumConstructionQueue,
        factory: _CompositionFactoryPort,
        runner: _RunnerPort,
        routes: PremiumConstructionRoutes,
        state_root: str | Path,
        config: PremiumConstructionCampaignConfig,
        id_factory: RuntimeIdFactory | None = None,
        clock: Callable[[], str] | None = None,
        progress_callback: Callable[[PremiumConstructionProgress], None] | None = None,
        result_schema: str = PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
    ) -> None:
        if not isinstance(queue, PremiumConstructionQueue):
            raise PremiumConstructionCampaignError("premium construction queue differs")
        if factory.selection.selection_blake3 != queue.selection_blake3:
            raise PremiumConstructionCampaignError("construction factory selection differs")
        if getattr(runner, "shard_count", None) != config.app_server_shards:
            raise PremiumConstructionCampaignError("Codex app-server shard count differs")
        if not callable(getattr(runner, "run_once", None)):
            raise PremiumConstructionCampaignError("Codex construction runner differs")
        self.queue = queue
        self.factory = factory
        self.runner = runner
        self.routes = routes
        self.config = config
        if result_schema not in {
            PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
            PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4,
        }:
            raise PremiumConstructionCampaignError("construction result schema differs")
        self.result_schema = result_schema
        self._store = _AppendOnlyCampaignStore(state_root, queue)
        self._ids = _LockedIdFactory(id_factory or RandomUUIDFactory())
        self._clock = clock or (lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z"))
        self._progress_callback = progress_callback
        self._metrics_lock = Lock()
        self._logical_calls = 0

    def preflight(
        self, candidate_ids: Iterable[str] | None = None
    ) -> PremiumConstructionPreflightReceipt:
        return preflight_premium_construction_campaign(
            queue=self.queue,
            factory=self.factory,
            routes=self.routes,
            config=self.config,
            candidate_ids=candidate_ids,
            result_schema=self.result_schema,
        )

    def pending_candidate_ids(self, *, limit: int | None = None) -> tuple[str, ...]:
        """Reopen append-only state and return the outcome-blind queue suffix."""

        if limit is not None and (type(limit) is not int or limit < 1):
            raise PremiumConstructionCampaignError("pending candidate limit must be positive")
        with self._store.exclusive():
            entries = {entry.candidate_id: entry for entry in self.queue.entries}
            claims = self._store.claims()
            terminals = self._store.terminals()
            if set(claims) - set(entries) or set(terminals) - set(entries):
                raise PremiumConstructionCampaignError(
                    "construction state escaped its frozen queue"
                )
            for candidate_id, claim in claims.items():
                _verify_claim(
                    claim,
                    queue=self.queue,
                    entry=entries[candidate_id],
                )
            for candidate_id, record in terminals.items():
                entry = entries[candidate_id]
                claim = claims.get(candidate_id)
                if not (
                    claim is not None
                    and record.queue_id == self.queue.queue_id
                    and record.queue_blake3 == self.queue.queue_blake3
                    and record.selection_blake3 == self.queue.selection_blake3
                    and record.source_candidate_id == entry.source_candidate_id
                    and record.campaign_ordinal == entry.campaign_ordinal
                    and record.claim_blake3 == claim["claim_blake3"]
                ):
                    raise PremiumConstructionCampaignError(
                        "construction terminal queue binding differs"
                    )
            pending = tuple(
                entry.candidate_id
                for entry in self.queue.entries
                if entry.candidate_id not in claims
                and entry.candidate_id not in terminals
            )
        return pending if limit is None else pending[:limit]

    def _on_call(self) -> None:
        with self._metrics_lock:
            self._logical_calls += 1

    def _execute(
        self,
        entry: PremiumConstructionQueueEntry,
        composition: LegacyConstructionComposition,
        *,
        session_id: str,
        claim: Mapping[str, Any],
    ) -> PremiumConstructionTerminalRecord:
        counting = _CountingRunner(self.runner, self._on_call)
        failure_receipts: tuple[ConstructionFailureReceipt, ...] = ()
        attempt_evidence: tuple[ConstructionAttemptEvidence, ...] = ()
        try:
            orchestrator = PremiumCodexConstructionOrchestrator(
                counting,
                self.routes,
                id_factory=self._ids,
                result_schema=self.result_schema,
            )
            published = composition.run(orchestrator)
        except Exception as exc:
            failure_receipts = (
                tuple(exc.failures) if isinstance(exc, ConstructionWaveError) else ()
            )
            attempt_evidence = (
                tuple(exc.attempt_evidence)
                if isinstance(exc, ConstructionWaveError)
                else ()
            )
            # Construction waves execute lanes concurrently.  The exception's
            # receipt collection may therefore reflect completion order while
            # attempt evidence is deliberately emitted in the signed lane
            # order.  Bind the terminal to failures in that same evidence
            # order so a standalone catalog reopen is deterministic.
            failures = tuple(
                evidence.failure_receipt.receipt_blake3
                for evidence in attempt_evidence
                if evidence.failure_receipt is not None
            )
            code = (
                f"{exc.wave}_wave_failure"
                if isinstance(exc, ConstructionWaveError)
                else "construction_or_publication_failure"
                if counting.calls
                else "source_preflight_failure"
            )
            record = _terminal_record(
                entry=entry,
                queue=self.queue,
                session_id=session_id,
                claim=claim,
                id_factory=self._ids,
                clock=self._clock,
                status="quarantined",
                logical_calls=counting.calls,
                error_code=code,
                failure_receipt_blake3s=failures,
                attempt_evidence_blake3s=tuple(
                    evidence.evidence_blake3 for evidence in attempt_evidence
                ),
            )
        else:
            # Supplemental author/critic failures are successful-quorum
            # evidence, not campaign quarantine.  Persist them alongside the
            # terminal proof and bind their digests into the v2 record.
            failure_receipts = tuple(published.result.supplemental_failures)
            attempt_evidence = tuple(published.result.attempt_evidence)
            record = _terminal_record(
                entry=entry,
                queue=self.queue,
                session_id=session_id,
                claim=claim,
                id_factory=self._ids,
                clock=self._clock,
                status="succeeded",
                logical_calls=counting.calls,
                error_code=None,
                failure_receipt_blake3s=tuple(
                    receipt.receipt_blake3 for receipt in failure_receipts
                ),
                attempt_evidence_blake3s=tuple(
                    evidence.evidence_blake3 for evidence in attempt_evidence
                ),
                published=published,
            )
        if attempt_evidence:
            self._store.write_attempt_evidence(attempt_evidence)
        if failure_receipts:
            self._store.write_failure_receipts(failure_receipts)
        self._store.write_terminal(record)
        return record

    def _progress(
        self,
        *,
        active: int,
        claimed: int,
        succeeded: int,
        quarantined: int,
        completed_this_session: int,
        started: float,
    ) -> PremiumConstructionProgress:
        elapsed = max(time.monotonic() - started, 1e-9)
        rate = completed_this_session / elapsed
        unclaimed = len(self.queue.entries) - claimed
        with self._metrics_lock:
            calls = self._logical_calls
        remaining = active + unclaimed
        return PremiumConstructionProgress(
            schema="eva.premium-construction-progress.v1",
            queue_total=len(self.queue.entries),
            claimed=claimed,
            active=active,
            succeeded=succeeded,
            quarantined=quarantined,
            unclaimed=unclaimed,
            logical_provider_calls_started=calls,
            candidates_per_second=rate,
            eta_seconds=None if rate == 0 else remaining / rate,
            worker_width=self.config.worker_width,
            app_server_shards=self.config.app_server_shards,
        )

    def run(self) -> PremiumConstructionCampaignReport:
        started = time.monotonic()
        with self._metrics_lock:
            self._logical_calls = 0
        session_id = self._ids.new("premium-construction-session")
        with self._store.exclusive():
            self._store.start_session(
                session_id,
                clock=self._clock,
                config=self.config,
                routes=self.routes,
                result_schema=self.result_schema,
            )
            recovered = self._store.recover_abandoned(
                session_id=session_id, id_factory=self._ids, clock=self._clock
            )
            claims = self._store.claims()
            terminals = self._store.terminals()
            claimed_total = len(claims)
            succeeded_total = sum(
                record.status == "succeeded" for record in terminals.values()
            )
            quarantined_total = sum(
                record.status == "quarantined" for record in terminals.values()
            )
            pending = [
                entry
                for entry in self.queue.entries
                if entry.candidate_id not in claims and entry.candidate_id not in terminals
            ]
            if self.config.max_candidates is not None:
                pending = pending[: self.config.max_candidates]
            cursor = 0
            futures: dict[Future[PremiumConstructionTerminalRecord], PremiumConstructionQueueEntry] = {}
            completed_records: list[PremiumConstructionTerminalRecord] = []
            last_progress = 0.0

            def emit_progress(*, active: int, force: bool = False) -> None:
                nonlocal last_progress
                now = time.monotonic()
                if self._progress_callback is None or (
                    not force and now - last_progress < self.config.progress_seconds
                ):
                    return
                self._progress_callback(
                    self._progress(
                        active=active,
                        claimed=claimed_total,
                        succeeded=succeeded_total,
                        quarantined=quarantined_total,
                        completed_this_session=len(completed_records),
                        started=started,
                    )
                )
                last_progress = now

            def fill(executor: ThreadPoolExecutor) -> None:
                nonlocal cursor, claimed_total, quarantined_total
                while len(futures) < self.config.worker_width and cursor < len(pending):
                    entry = pending[cursor]
                    cursor += 1
                    claim = self._store.claim(entry, session_id=session_id, clock=self._clock)
                    if claim is None:
                        continue
                    claimed_total += 1
                    try:
                        # The real legacy source snapshotter must run on the
                        # main thread.  Submit only after its provider-free
                        # authority reopening has completed.
                        composition = self.factory.compose(entry.candidate_id, self.routes)
                    except Exception:
                        record = _terminal_record(
                            entry=entry,
                            queue=self.queue,
                            session_id=session_id,
                            claim=claim,
                            id_factory=self._ids,
                            clock=self._clock,
                            status="quarantined",
                            logical_calls=0,
                            error_code="source_preflight_failure",
                        )
                        self._store.write_terminal(record)
                        completed_records.append(record)
                        quarantined_total += 1
                        emit_progress(active=len(futures))
                        continue
                    futures[executor.submit(
                        self._execute,
                        entry,
                        composition,
                        session_id=session_id,
                        claim=claim,
                    )] = entry
                    # Source authority reopening is intentionally serialized
                    # on the caller main thread. Keep long high-width fills
                    # observable even before the first completion is drained.
                    emit_progress(active=len(futures))

            with ThreadPoolExecutor(
                max_workers=self.config.worker_width,
                thread_name_prefix="eva-premium-campaign",
            ) as executor:
                emit_progress(active=0, force=True)
                fill(executor)
                while futures:
                    done, _ = wait(
                        tuple(futures),
                        timeout=self.config.progress_seconds,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        futures.pop(future)
                        record = future.result()
                        completed_records.append(record)
                        if record.status == "succeeded":
                            succeeded_total += 1
                        else:
                            quarantined_total += 1
                    fill(executor)
                    emit_progress(active=len(futures))

            catalog = self._store.write_catalog(id_factory=self._ids, clock=self._clock)
            verify_premium_construction_proof_catalog(catalog, queue=self.queue)
            elapsed = time.monotonic() - started
            with self._metrics_lock:
                calls = self._logical_calls
            report_core = {
                "schema": "eva.premium-construction-campaign-report.v2",
                "session_id": session_id,
                "queue_id": self.queue.queue_id,
                "queue_blake3": self.queue.queue_blake3,
                "claimed_this_session": len(completed_records),
                "succeeded_this_session": sum(
                    record.status == "succeeded" for record in completed_records
                ),
                "quarantined_this_session": sum(
                    record.status == "quarantined" for record in completed_records
                ),
                "recovered_without_retry": len(recovered),
                "logical_provider_calls_started": calls,
                "elapsed_seconds": elapsed,
                "proof_catalog_blake3": catalog.catalog_blake3,
            }
            report = PremiumConstructionCampaignReport(
                schema=report_core["schema"],
                session_id=session_id,
                queue_id=self.queue.queue_id,
                queue_blake3=self.queue.queue_blake3,
                claimed_this_session=len(completed_records),
                succeeded_this_session=report_core["succeeded_this_session"],
                quarantined_this_session=report_core["quarantined_this_session"],
                recovered_without_retry=len(recovered),
                logical_provider_calls_started=calls,
                elapsed_seconds=elapsed,
                proof_catalog=catalog,
                report_blake3=blake3_hex(report_core),
            )
            self._store.finish_session(session_id, report.to_document())
            emit_progress(active=0, force=True)
            return report


__all__ = [
    "PremiumConstructionCampaign",
    "PremiumConstructionCampaignConfig",
    "PremiumConstructionCampaignError",
    "PremiumConstructionCampaignReport",
    "PremiumConstructionPreflightReceipt",
    "PremiumConstructionProgress",
    "PremiumConstructionProofCatalog",
    "PremiumConstructionQueue",
    "PremiumConstructionQueueEntry",
    "PremiumConstructionTerminalRecord",
    "build_premium_construction_queue",
    "preflight_premium_construction_campaign",
    "render_premium_construction_progress",
    "verify_premium_construction_proof_catalog",
]
