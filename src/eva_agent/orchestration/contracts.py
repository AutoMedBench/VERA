"""Ports and immutable contracts for the high-width campaign scheduler.

The orchestration layer deliberately depends on narrow protocols.  Production
code can supply :class:`CampaignLedger` and :class:`VerifiableDataPipeline`,
while tests can prove scheduling invariants without making provider calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import UUID

from eva_agent.pipeline.contracts import (
    BenchmarkEpisode,
    CompiledRubricTable,
    ModelTarget,
    PipelineResult,
)
from eva_agent.pipeline.digests import is_blake3


class OrchestrationContractError(ValueError):
    """An orchestration input violates a fail-closed campaign contract."""


def canonical_uuid(value: str, *, label: str) -> str:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError):
        raise OrchestrationContractError(f"{label} must be a UUID") from None
    if str(parsed) != value:
        raise OrchestrationContractError(f"{label} must be canonical UUID text")
    return value


@dataclass(frozen=True)
class CandidateJob:
    """Everything needed for one and only one three-cohort pipeline run."""

    candidate_id: str
    episode: BenchmarkEpisode
    rubric: CompiledRubricTable
    targets: tuple[ModelTarget, ...]

    def __post_init__(self) -> None:
        canonical_uuid(self.candidate_id, label="candidate_id")
        if len(self.targets) != 3 or {target.cohort.value for target in self.targets} != {
            "weak",
            "middle",
            "strong",
        }:
            raise OrchestrationContractError(
                "candidate targets must contain weak, middle, and strong exactly once"
            )


@dataclass(frozen=True)
class OrchestrationConfig:
    """Bounded concurrency and lease policy.

    Eight sandbox workers is intentionally conservative.  There is no small
    artificial upper bound: operators with sufficient provider capacity can
    select 64, 128, or more workers explicitly.
    """

    worker_width: int = 8
    queue_capacity: int = 16
    claim_batch_size: int = 8
    lease_seconds: float = 900.0
    heartbeat_seconds: float = 30.0
    progress_seconds: float = 5.0
    max_candidates: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("worker_width", self.worker_width),
            ("queue_capacity", self.queue_capacity),
            ("claim_batch_size", self.claim_batch_size),
        ):
            if type(value) is not int or value < 1:
                raise OrchestrationContractError(f"{name} must be a positive integer")
        for name, value in (
            ("lease_seconds", self.lease_seconds),
            ("heartbeat_seconds", self.heartbeat_seconds),
            ("progress_seconds", self.progress_seconds),
        ):
            if type(value) not in {int, float} or float(value) <= 0:
                raise OrchestrationContractError(f"{name} must be positive")
        if self.heartbeat_seconds >= self.lease_seconds:
            raise OrchestrationContractError("heartbeat_seconds must be shorter than the lease")
        if self.max_candidates is not None and (
            type(self.max_candidates) is not int or self.max_candidates < 1
        ):
            raise OrchestrationContractError("max_candidates must be a positive integer")


@dataclass(frozen=True)
class SignedSupervisorEvidence:
    """Already-created external evidence; the orchestrator never signs it."""

    candidate_id: str
    sandbox_id: str
    pipeline_result_blake3: str
    manifest_blake3: str
    admission_receipt_blake3: str
    supervisor_transition_blake3: str
    signature_bundle_blake3: str

    def __post_init__(self) -> None:
        canonical_uuid(self.candidate_id, label="candidate_id")
        canonical_uuid(self.sandbox_id, label="sandbox_id")
        if not all(
            is_blake3(value)
            for value in (
                self.pipeline_result_blake3,
                self.manifest_blake3,
                self.admission_receipt_blake3,
                self.supervisor_transition_blake3,
                self.signature_bundle_blake3,
            )
        ):
            raise OrchestrationContractError("signed supervisor evidence BLAKE3 differs")


@dataclass(frozen=True)
class ProgressMetrics:
    run_id: str
    target: int
    admitted: int
    train: int
    development: int
    sealed_evaluation: int
    queued: int
    active: int
    rejected: int
    infrastructure_quarantine: int
    run_claimed: int
    run_completed: int
    run_reviewed: int
    run_rejected: int
    run_quarantined: int
    run_admitted: int
    recovered_expired: int
    worker_width: int
    running_workers: int
    queue_depth: int
    stop_requested: bool
    elapsed_seconds: float
    completions_per_minute: float


@dataclass(frozen=True)
class CampaignRunReport:
    run_id: str
    run_receipt_blake3: str
    claimed: int
    completed: int
    reviewed: int
    rejected: int
    infrastructure_quarantine: int
    admitted: int
    recovered_expired: int
    stop_requested: bool
    stop_reason: str | None
    elapsed_seconds: float
    final_progress: ProgressMetrics


class CampaignProgressPort(Protocol):
    target: int
    admitted: int
    train: int
    development: int
    sealed_evaluation: int
    queued: int
    active: int
    rejected: int
    infrastructure_quarantine: int


class CampaignLedgerPort(Protocol):
    """The transactional subset of :class:`CampaignLedger` used here."""

    def claim(
        self, *, worker_id: str, limit: int, lease_seconds: float
    ) -> tuple[str, ...]: ...

    def heartbeat(
        self, *, worker_id: str, candidate_ids: tuple[str, ...], lease_seconds: float
    ) -> None: ...

    def mark_provider_boundary(self, *, worker_id: str, candidate_id: str) -> None: ...

    def recover_expired(self) -> tuple[tuple[str, ...], tuple[str, ...]]: ...

    def quarantine_expired(self) -> tuple[str, ...]: ...

    def record_attempt(
        self,
        *,
        attempt_id: str,
        candidate_id: str,
        phase: str,
        outcome: str,
        receipt_blake3: str,
    ) -> None: ...

    def finish_candidate(self, candidate_id: str, *, status: str) -> None: ...

    def admit(
        self,
        *,
        candidate_id: str,
        sandbox_id: str,
        manifest_blake3: str,
        admission_receipt_blake3: str,
        supervisor_transition_blake3: str,
        signatures_verified: bool,
    ) -> None: ...

    def progress(self) -> CampaignProgressPort: ...


class EligibleCampaignLedgerPort(CampaignLedgerPort, Protocol):
    """Ledger port that gates claims through an exact executable allowlist."""

    def claim_eligible(
        self,
        *,
        worker_id: str,
        limit: int,
        lease_seconds: float,
        eligible_candidate_ids: tuple[str, ...],
    ) -> tuple[str, ...]: ...


class CandidateSource(Protocol):
    def load(self, candidate_id: str) -> CandidateJob: ...


class VerifiablePipelinePort(Protocol):
    """Structural port implemented by :class:`VerifiableDataPipeline`."""

    def run(
        self,
        *,
        episode: BenchmarkEpisode,
        rubric: CompiledRubricTable,
        targets: Sequence[ModelTarget],
    ) -> PipelineResult: ...


class SupervisorEvidenceSource(Protocol):
    """Read an external supervisor decision; never manufacture one here."""

    def evidence_for(
        self, *, candidate_id: str, result: PipelineResult
    ) -> SignedSupervisorEvidence | None: ...


class SupervisorSignatureVerifier(Protocol):
    def verify(
        self,
        *,
        candidate_id: str,
        result: PipelineResult,
        evidence: SignedSupervisorEvidence,
    ) -> bool: ...


PipelineFactory = Callable[[str], VerifiablePipelinePort]
CandidatePipelineFactory = Callable[[str, CandidateJob], VerifiablePipelinePort]
ProgressCallback = Callable[[ProgressMetrics], None]


@dataclass(frozen=True)
class CampaignComponents:
    """Components returned by a deployment-specific factory module."""

    candidate_source: CandidateSource
    pipeline_factory: PipelineFactory
    supervisor_evidence_source: SupervisorEvidenceSource | None = None
    supervisor_signature_verifier: SupervisorSignatureVerifier | None = None


__all__ = [name for name in globals() if not name.startswith("_")]
