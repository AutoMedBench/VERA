"""Durable bounded scheduler for wide benchmark-to-sandbox campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import queue
from pathlib import Path
from threading import Event, Lock, Thread
import time
from typing import Any
from uuid import uuid4

from eva_agent.pipeline import InfrastructureQuarantineError
from eva_agent.pipeline.digests import is_blake3

from .contracts import (
    CampaignLedgerPort,
    CampaignRunReport,
    CandidateJob,
    CandidatePipelineFactory,
    CandidateSource,
    OrchestrationConfig,
    OrchestrationContractError,
    PipelineFactory,
    ProgressCallback,
    ProgressMetrics,
    SignedSupervisorEvidence,
    SupervisorEvidenceSource,
    SupervisorSignatureVerifier,
    canonical_uuid,
)
from .receipts import ReceiptJournal



@dataclass
class _ClaimRequest:
    """One initialized idle worker waiting for one durable lease."""

    worker_id: str
    ready: Event = field(default_factory=Event)
    candidate_id: str | None = None


class CampaignOrchestrationError(OrchestrationContractError):
    """The campaign scheduler itself could not safely continue."""


class CampaignOrchestrator:
    """Claim queued candidates once, run them widely, and fail closed.

    Workers request one candidate only after their shared pipeline is ready.
    The coordinator may batch those simultaneous ready-worker requests into a
    single SQLite transaction, but it never leases work without a waiting
    worker.  Outstanding claims are therefore bounded by ``worker_width`` and
    are independent of ``queue_capacity``.  A durable provider-boundary marker
    distinguishes provably uncalled crash leases (safe to requeue) from crossed
    or legacy-ambiguous leases (quarantine without retry).
    """

    def __init__(
        self,
        *,
        ledger: CampaignLedgerPort,
        candidate_source: CandidateSource,
        pipeline_factory: PipelineFactory | None = None,
        candidate_pipeline_factory: CandidatePipelineFactory | None = None,
        eligible_candidate_ids: tuple[str, ...] | None = None,
        receipt_root: Path,
        config: OrchestrationConfig | None = None,
        supervisor_evidence_source: SupervisorEvidenceSource | None = None,
        supervisor_signature_verifier: SupervisorSignatureVerifier | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        if (supervisor_evidence_source is None) != (
            supervisor_signature_verifier is None
        ):
            raise CampaignOrchestrationError(
                "supervisor evidence source and signature verifier must be configured together"
            )
        if (pipeline_factory is None) == (candidate_pipeline_factory is None):
            raise CampaignOrchestrationError(
                "exactly one worker or candidate pipeline factory is required"
            )
        if eligible_candidate_ids is not None:
            if (
                not isinstance(eligible_candidate_ids, tuple)
                or not eligible_candidate_ids
                or len(set(eligible_candidate_ids)) != len(eligible_candidate_ids)
                or not callable(getattr(ledger, "claim_eligible", None))
            ):
                raise CampaignOrchestrationError(
                    "eligible candidate claims require a non-empty exact ledger allowlist"
                )
            for candidate_id in eligible_candidate_ids:
                canonical_uuid(candidate_id, label="eligible_candidate_id")
        self._ledger = ledger
        self._source = candidate_source
        self._pipeline_factory = pipeline_factory
        self._candidate_pipeline_factory = candidate_pipeline_factory
        self._eligible_candidate_ids = eligible_candidate_ids
        self._receipt_root = Path(receipt_root)
        self.config = config or OrchestrationConfig()
        self._supervisor = supervisor_evidence_source
        self._signature_verifier = supervisor_signature_verifier
        self._progress_callback = progress_callback
        self._stop = Event()
        self._heartbeat_stop = Event()
        self._stop_lock = Lock()
        self._stop_reason: str | None = None
        self._run_lock = Lock()
        self._has_run = False
        self._owned_lock = Lock()
        self._owned: set[str] = set()
        self._stats_lock = Lock()
        self._state_changed = Event()
        self._stats = {
            "claimed": 0,
            "completed": 0,
            "reviewed": 0,
            "rejected": 0,
            "quarantined": 0,
            "admitted": 0,
            "recovered_expired": 0,
            "running_workers": 0,
        }

    def request_stop(self, reason: str = "operator_requested") -> None:
        """Stop claiming and drain every candidate already leased by this run."""

        with self._stop_lock:
            if self._stop_reason is None:
                self._stop_reason = reason or "operator_requested"
        self._stop.set()

    def _owned_add(self, candidate_ids: tuple[str, ...]) -> None:
        with self._owned_lock:
            overlap = self._owned.intersection(candidate_ids)
            if overlap:
                raise CampaignOrchestrationError("candidate was claimed twice in one run")
            self._owned.update(candidate_ids)

    def _stats_add(self, **values: int) -> None:
        with self._stats_lock:
            for key, value in values.items():
                self._stats[key] += value

    def _work_in_flight(self) -> bool:
        """Return whether this run still has a claimed task crossing a boundary.

        ``_owned`` is deliberately cleared before synchronous signed-admission
        handling, so the running-worker view prevents an empty-claim decision
        from racing the final worker across that narrow gap.
        """

        with self._owned_lock:
            owned = bool(self._owned)
        with self._stats_lock:
            running = self._stats["running_workers"] > 0
        return owned or running

    def _progress(
        self,
        *,
        run_id: str,
        queue_depth: int,
        started: float,
    ) -> ProgressMetrics:
        durable = self._ledger.progress()
        with self._stats_lock:
            stats = dict(self._stats)
        elapsed = max(0.0, time.monotonic() - started)
        rate = 0.0 if elapsed == 0 else stats["completed"] * 60.0 / elapsed
        return ProgressMetrics(
            run_id=run_id,
            target=int(durable.target),
            admitted=int(durable.admitted),
            train=int(durable.train),
            development=int(durable.development),
            sealed_evaluation=int(durable.sealed_evaluation),
            queued=int(durable.queued),
            active=int(durable.active),
            rejected=int(durable.rejected),
            infrastructure_quarantine=int(durable.infrastructure_quarantine),
            run_claimed=stats["claimed"],
            run_completed=stats["completed"],
            run_reviewed=stats["reviewed"],
            run_rejected=stats["rejected"],
            run_quarantined=stats["quarantined"],
            run_admitted=stats["admitted"],
            recovered_expired=stats["recovered_expired"],
            worker_width=self.config.worker_width,
            running_workers=stats["running_workers"],
            queue_depth=queue_depth,
            stop_requested=self._stop.is_set(),
            elapsed_seconds=elapsed,
            completions_per_minute=rate,
        )

    def _heartbeat_loop(self, *, run_id: str, journal: ReceiptJournal) -> None:
        while not self._heartbeat_stop.wait(self.config.heartbeat_seconds):
            with self._owned_lock:
                candidate_ids = tuple(sorted(self._owned))
                if not candidate_ids:
                    continue
                try:
                    self._ledger.heartbeat(
                        worker_id=run_id,
                        candidate_ids=candidate_ids,
                        lease_seconds=self.config.lease_seconds,
                    )
                except Exception as exc:
                    journal.append(
                        event="heartbeat_failed",
                        payload={
                            "error_type": type(exc).__name__,
                            "candidate_count": len(candidate_ids),
                            "provider_retry_created": False,
                        },
                    )
                    self.request_stop("heartbeat_failed")

    @staticmethod
    def _validated_result(result: Any) -> tuple[str, str, str, bool]:
        result_blake3 = getattr(result, "result_blake3", None)
        pipeline_run_id = getattr(result, "run_id", None)
        manifests = getattr(result, "sandbox_manifests", ())
        recommendation = getattr(
            getattr(result, "recommendation", None), "recommendation", None
        )
        policy_passed = getattr(getattr(result, "separation", None), "policy_passed", None)
        admission_policy_schema = getattr(
            getattr(result, "separation", None), "admission_policy_schema", None
        )
        admission_policy_passed = getattr(
            getattr(result, "separation", None), "admission_policy_passed", None
        )
        if (
            not is_blake3(result_blake3)
            or not isinstance(pipeline_run_id, str)
            or len(manifests) != 1
            or recommendation
            not in {"eligible_for_signed_supervisor_admission", "not_eligible"}
            or type(policy_passed) is not bool
            or (
                admission_policy_schema is not None
                and type(admission_policy_passed) is not bool
            )
        ):
            raise CampaignOrchestrationError("pipeline result envelope differs")
        canonical_uuid(pipeline_run_id, label="pipeline run_id")
        manifest = manifests[0]
        sandbox_id = getattr(manifest, "sandbox_id", None)
        manifest_blake3 = getattr(manifest, "manifest_blake3", None)
        if not isinstance(sandbox_id, str) or not is_blake3(manifest_blake3):
            raise CampaignOrchestrationError("pipeline sandbox commitment differs")
        canonical_uuid(sandbox_id, label="sandbox_id")
        eligible = recommendation == "eligible_for_signed_supervisor_admission"
        expected_eligible = (
            policy_passed
            if admission_policy_schema is None
            else admission_policy_passed
        )
        if eligible is not expected_eligible:
            raise CampaignOrchestrationError("recommendation and admission policy differ")
        return pipeline_run_id, result_blake3, manifest_blake3, eligible

    def _commit_terminal(
        self,
        *,
        candidate_id: str,
        status: str,
        outcome: str,
        receipt_blake3: str,
        worker_id: str,
        journal: ReceiptJournal,
    ) -> str:
        """Commit attempt + terminal state while excluding heartbeat races."""

        committed_status = status
        with self._owned_lock:
            try:
                self._ledger.record_attempt(
                    attempt_id=str(uuid4()),
                    candidate_id=candidate_id,
                    phase="weak_middle_strong_cascade",
                    outcome=outcome,
                    receipt_blake3=receipt_blake3,
                )
                self._ledger.finish_candidate(candidate_id, status=status)
            except Exception as exc:
                committed_status = "infrastructure_quarantine"
                journal.append(
                    worker_id=worker_id,
                    event="ledger_commit_failed",
                    payload={
                        "candidate_id": candidate_id,
                        "error_type": type(exc).__name__,
                        "semantic_score_created": False,
                        "provider_retry_created": False,
                    },
                )
                try:
                    self._ledger.finish_candidate(
                        candidate_id, status="infrastructure_quarantine"
                    )
                except Exception as terminal_exc:
                    journal.append(
                        worker_id=worker_id,
                        event="terminal_quarantine_deferred",
                        payload={
                            "candidate_id": candidate_id,
                            "error_type": type(terminal_exc).__name__,
                            "lease_expiry_will_quarantine": True,
                            "provider_retry_created": False,
                        },
                    )
            finally:
                self._owned.discard(candidate_id)
        return committed_status

    def _maybe_admit(
        self,
        *,
        candidate_id: str,
        result: Any,
        worker_id: str,
        journal: ReceiptJournal,
    ) -> bool:
        if self._supervisor is None or self._signature_verifier is None:
            journal.append(
                worker_id=worker_id,
                event="awaiting_external_supervisor",
                payload={
                    "candidate_id": candidate_id,
                    "admission_created": False,
                },
            )
            return False
        try:
            evidence = self._supervisor.evidence_for(
                candidate_id=candidate_id, result=result
            )
        except Exception as exc:
            journal.append(
                worker_id=worker_id,
                event="supervisor_evidence_unavailable",
                payload={
                    "candidate_id": candidate_id,
                    "error_type": type(exc).__name__,
                    "admission_created": False,
                },
            )
            return False
        if evidence is None:
            journal.append(
                worker_id=worker_id,
                event="awaiting_external_supervisor",
                payload={"candidate_id": candidate_id, "admission_created": False},
            )
            return False
        if (
            not isinstance(evidence, SignedSupervisorEvidence)
            or evidence.candidate_id != candidate_id
        ):
            journal.append(
                worker_id=worker_id,
                event="supervisor_evidence_rejected",
                payload={
                    "candidate_id": candidate_id,
                    "reason": "evidence_contract_or_identity",
                    "admission_created": False,
                },
            )
            return False
        manifest = result.sandbox_manifests[0]
        if (
            evidence.pipeline_result_blake3 != result.result_blake3
            or evidence.sandbox_id != manifest.sandbox_id
            or evidence.manifest_blake3 != manifest.manifest_blake3
        ):
            journal.append(
                worker_id=worker_id,
                event="supervisor_evidence_rejected",
                payload={
                    "candidate_id": candidate_id,
                    "reason": "pipeline_or_sandbox_commitment",
                    "signature_bundle_blake3": evidence.signature_bundle_blake3,
                    "admission_created": False,
                },
            )
            return False
        try:
            verified = self._signature_verifier.verify(
                candidate_id=candidate_id,
                result=result,
                evidence=evidence,
            )
        except Exception as exc:
            journal.append(
                worker_id=worker_id,
                event="supervisor_signature_verification_failed",
                payload={
                    "candidate_id": candidate_id,
                    "error_type": type(exc).__name__,
                    "admission_created": False,
                },
            )
            return False
        if verified is not True:
            journal.append(
                worker_id=worker_id,
                event="supervisor_evidence_rejected",
                payload={
                    "candidate_id": candidate_id,
                    "reason": "external_signature_verifier_returned_false",
                    "signature_bundle_blake3": evidence.signature_bundle_blake3,
                    "admission_created": False,
                },
            )
            return False
        try:
            self._ledger.admit(
                candidate_id=candidate_id,
                sandbox_id=evidence.sandbox_id,
                manifest_blake3=evidence.manifest_blake3,
                admission_receipt_blake3=evidence.admission_receipt_blake3,
                supervisor_transition_blake3=evidence.supervisor_transition_blake3,
                signatures_verified=True,
            )
        except Exception as exc:
            journal.append(
                worker_id=worker_id,
                event="signed_admission_commit_failed",
                payload={
                    "candidate_id": candidate_id,
                    "error_type": type(exc).__name__,
                    "signature_bundle_blake3": evidence.signature_bundle_blake3,
                    "admission_created": False,
                },
            )
            return False
        journal.append(
            worker_id=worker_id,
            event="signed_admission_committed",
            payload={
                "candidate_id": candidate_id,
                "sandbox_id": evidence.sandbox_id,
                "pipeline_result_blake3": evidence.pipeline_result_blake3,
                "manifest_blake3": evidence.manifest_blake3,
                "admission_receipt_blake3": evidence.admission_receipt_blake3,
                "supervisor_transition_blake3": evidence.supervisor_transition_blake3,
                "signature_bundle_blake3": evidence.signature_bundle_blake3,
                "external_signatures_verified": True,
            },
        )
        return True

    def _quarantine_unsigned_review(
        self,
        *,
        candidate_id: str,
        worker_id: str,
        journal: ReceiptJournal,
    ) -> bool:
        """Release a quota slot when synchronous signed admission did not commit."""

        try:
            self._ledger.finish_candidate(
                candidate_id, status="infrastructure_quarantine"
            )
        except Exception as exc:
            journal.append(
                worker_id=worker_id,
                event="unsigned_review_quarantine_failed",
                payload={
                    "candidate_id": candidate_id,
                    "error_type": type(exc).__name__,
                    "semantic_outcome_retained": True,
                    "admission_created": False,
                    "provider_retry_created": False,
                },
            )
            self.request_stop("unsigned_review_quarantine_failed")
            return False
        journal.append(
            worker_id=worker_id,
            event="unsigned_review_quarantined",
            payload={
                "candidate_id": candidate_id,
                "reason": "signed_admission_not_committed",
                "semantic_outcome_retained": True,
                "admission_created": False,
                "provider_retry_created": False,
            },
        )
        return True

    def _worker(
        self,
        *,
        lease_owner: str,
        worker_id: str,
        claim_requests: queue.Queue[_ClaimRequest],
        journal: ReceiptJournal,
    ) -> None:
        pipeline: Any = None
        pipeline_factory_error: str | None = None
        if self._pipeline_factory is not None:
            try:
                pipeline = self._pipeline_factory(worker_id)
            except Exception as exc:
                pipeline_factory_error = type(exc).__name__
        journal.append(
            worker_id=worker_id,
            event="worker_started",
            payload={
                "pipeline_ready": pipeline_factory_error is None,
                "candidate_scoped_pipeline": self._candidate_pipeline_factory is not None,
                "pipeline_factory_error_type": pipeline_factory_error,
            },
        )
        if pipeline_factory_error is not None:
            journal.append(
                worker_id=worker_id,
                event="worker_finished",
                payload={
                    "drained": True,
                    "stop_requested": self._stop.is_set(),
                    "candidate_claimed": False,
                    "reason": "pipeline_factory_unavailable_before_claim",
                },
            )
            self._state_changed.set()
            return
        while not self._stop.is_set():
            request = _ClaimRequest(worker_id=worker_id)
            claim_requests.put(request)
            self._state_changed.set()
            request.ready.wait()
            candidate_id = request.candidate_id
            if candidate_id is None:
                break
            self._stats_add(running_workers=1)
            started_receipt = journal.append(
                worker_id=worker_id,
                event="candidate_execution_started",
                payload={
                    "candidate_id": candidate_id,
                    "cohort_rollouts_expected": ["weak", "middle", "strong"],
                    "rollouts_per_cohort": 1,
                    "retry_count": 0,
                    "lease_state": "pre_provider",
                },
            )
            result: Any = None
            terminal_status = "infrastructure_quarantine"
            provider_boundary_state = "pre_provider"
            try:
                job = self._source.load(candidate_id)
                if not isinstance(job, CandidateJob) or job.candidate_id != candidate_id:
                    raise CampaignOrchestrationError("candidate source identity differs")
                if self._candidate_pipeline_factory is not None:
                    pipeline = self._candidate_pipeline_factory(worker_id, job)
                if pipeline is None or not callable(getattr(pipeline, "run", None)):
                    raise CampaignOrchestrationError("candidate pipeline factory failed")
                provider_boundary_state = "ambiguous"
                self._ledger.mark_provider_boundary(
                    worker_id=lease_owner,
                    candidate_id=candidate_id,
                )
                provider_boundary_state = "crossed"
                boundary_receipt = journal.append(
                    worker_id=worker_id,
                    event="candidate_provider_boundary_crossed",
                    payload={
                        "candidate_id": candidate_id,
                        "lease_owner": lease_owner,
                        "durable_state": "crossed",
                        "retry_count": 0,
                        "start_receipt_blake3": started_receipt["receipt_blake3"],
                    },
                )
                result = pipeline.run(
                    episode=job.episode,
                    rubric=job.rubric,
                    targets=job.targets,
                )
                pipeline_run_id, result_blake3, manifest_blake3, eligible = (
                    self._validated_result(result)
                )
                terminal_status = "reviewed" if eligible else "rejected"
                outcome = "qualified" if eligible else "not_eligible"
                outcome_receipt = journal.append(
                    worker_id=worker_id,
                    event="candidate_pipeline_finished",
                    payload={
                        "candidate_id": candidate_id,
                        "pipeline_run_id": pipeline_run_id,
                        "pipeline_result_blake3": result_blake3,
                        "manifest_blake3": manifest_blake3,
                        "outcome": outcome,
                        "semantic_score_created": True,
                        "cohort_rollouts_completed": ["weak", "middle", "strong"],
                        "retry_count": 0,
                        "start_receipt_blake3": started_receipt["receipt_blake3"],
                        "provider_boundary_receipt_blake3": boundary_receipt[
                            "receipt_blake3"
                        ],
                    },
                )
                terminal_status = self._commit_terminal(
                    candidate_id=candidate_id,
                    status=terminal_status,
                    outcome=outcome,
                    receipt_blake3=str(outcome_receipt["receipt_blake3"]),
                    worker_id=worker_id,
                    journal=journal,
                )
            except InfrastructureQuarantineError as exc:
                outcome_receipt = journal.append(
                    worker_id=worker_id,
                    event="candidate_infrastructure_quarantined",
                    payload={
                        "candidate_id": candidate_id,
                        "pipeline_run_id": exc.run_id,
                        "failures": dict(exc.failures),
                        "semantic_score_created": False,
                        "provider_failure_retained": True,
                        "provider_boundary_state": provider_boundary_state,
                        "retry_count": 0,
                        "start_receipt_blake3": started_receipt["receipt_blake3"],
                    },
                )
                terminal_status = self._commit_terminal(
                    candidate_id=candidate_id,
                    status="infrastructure_quarantine",
                    outcome="infrastructure_quarantine",
                    receipt_blake3=str(outcome_receipt["receipt_blake3"]),
                    worker_id=worker_id,
                    journal=journal,
                )
            except Exception as exc:
                outcome_receipt = journal.append(
                    worker_id=worker_id,
                    event="candidate_infrastructure_quarantined",
                    payload={
                        "candidate_id": candidate_id,
                        "error_type": type(exc).__name__,
                        "semantic_score_created": False,
                        "provider_failure_retained": provider_boundary_state
                        != "pre_provider",
                        "provider_boundary_state": provider_boundary_state,
                        "retry_count": 0,
                        "start_receipt_blake3": started_receipt["receipt_blake3"],
                    },
                )
                terminal_status = self._commit_terminal(
                    candidate_id=candidate_id,
                    status="infrastructure_quarantine",
                    outcome="infrastructure_quarantine",
                    receipt_blake3=str(outcome_receipt["receipt_blake3"]),
                    worker_id=worker_id,
                    journal=journal,
                )
            finally:
                if terminal_status == "reviewed":
                    admitted = result is not None and self._maybe_admit(
                        candidate_id=candidate_id,
                        result=result,
                        worker_id=worker_id,
                        journal=journal,
                    )
                    if admitted:
                        self._stats_add(reviewed=1, admitted=1)
                    elif self._quarantine_unsigned_review(
                        candidate_id=candidate_id,
                        worker_id=worker_id,
                        journal=journal,
                    ):
                        terminal_status = "infrastructure_quarantine"
                        self._stats_add(quarantined=1)
                    else:
                        # The durable row remains reviewed and occupies its
                        # quota slot; the scheduler is stopped fail closed.
                        self._stats_add(reviewed=1)
                elif terminal_status == "rejected":
                    self._stats_add(rejected=1)
                else:
                    self._stats_add(quarantined=1)
                self._stats_add(running_workers=-1, completed=1)
                self._state_changed.set()
        journal.append(
            worker_id=worker_id,
            event="worker_finished",
            payload={"drained": True, "stop_requested": self._stop.is_set()},
        )

    def run_until_idle(self) -> CampaignRunReport:
        with self._run_lock:
            if self._has_run:
                raise CampaignOrchestrationError("orchestrator instances are single-use")
            self._has_run = True
        run_id = str(uuid4())
        started = time.monotonic()
        journal = ReceiptJournal(self._receipt_root, run_id)
        requeued, quarantined = self._ledger.recover_expired()
        self._stats_add(recovered_expired=len(requeued) + len(quarantined))
        journal.start(
            {
                "config": asdict(self.config),
                "scheduling_policy": "ready_worker_jit_claim",
                "maximum_outstanding_claims": self.config.worker_width,
                "leased_queue_capacity": 0,
                "resume_policy": "requeue_proven_pre_provider_only",
                "expired_attempt_policy": "crossed_or_ambiguous_quarantine_no_retry",
                "requeued_expired_candidate_ids": list(requeued),
                "quarantined_expired_candidate_ids": list(quarantined),
            }
        )
        # These are tiny, unleased readiness signals, never candidate jobs.
        # At most one exists per worker, so their natural bound is worker_width.
        claim_requests: queue.Queue[_ClaimRequest] = queue.Queue()
        workers: list[Thread] = []
        for _ in range(self.config.worker_width):
            worker_id = str(uuid4())
            thread = Thread(
                target=self._worker,
                kwargs={
                    "lease_owner": run_id,
                    "worker_id": worker_id,
                    "claim_requests": claim_requests,
                    "journal": journal,
                },
                name=f"eva-sandbox-{worker_id[:8]}",
            )
            thread.start()
            workers.append(thread)
        heartbeat = Thread(
            target=self._heartbeat_loop,
            kwargs={"run_id": run_id, "journal": journal},
            name=f"eva-heartbeat-{run_id[:8]}",
        )
        heartbeat.start()

        last_progress = 0.0
        pending: list[_ClaimRequest] = []
        no_more_work = False
        coordinator_error: BaseException | None = None

        def release_pending() -> None:
            while True:
                try:
                    pending.append(claim_requests.get_nowait())
                except queue.Empty:
                    break
            while pending:
                request = pending.pop(0)
                request.candidate_id = None
                request.ready.set()

        try:
            while any(worker.is_alive() for worker in workers) or pending:
                while len(pending) < self.config.worker_width:
                    try:
                        pending.append(claim_requests.get_nowait())
                    except queue.Empty:
                        break
                with self._stats_lock:
                    claimed_so_far = self._stats["claimed"]
                limit_reached = (
                    self.config.max_candidates is not None
                    and claimed_so_far >= self.config.max_candidates
                )
                if self._stop.is_set() or limit_reached or no_more_work:
                    release_pending()
                elif pending:
                    limit = min(len(pending), self.config.claim_batch_size)
                    if self.config.max_candidates is not None:
                        limit = min(
                            limit,
                            self.config.max_candidates - claimed_so_far,
                        )
                    work_before_claim = self._work_in_flight()
                    if self._eligible_candidate_ids is None:
                        claimed = self._ledger.claim(
                            worker_id=run_id,
                            limit=limit,
                            lease_seconds=self.config.lease_seconds,
                        )
                    else:
                        claimed = self._ledger.claim_eligible(
                            worker_id=run_id,
                            limit=limit,
                            lease_seconds=self.config.lease_seconds,
                            eligible_candidate_ids=self._eligible_candidate_ids,
                        )
                    if (
                        not isinstance(claimed, tuple)
                        or len(claimed) > limit
                        or len(set(claimed)) != len(claimed)
                    ):
                        raise CampaignOrchestrationError(
                            "ledger claim batch exceeds ready-worker demand"
                        )
                    for candidate_id in claimed:
                        canonical_uuid(candidate_id, label="claimed candidate_id")
                    if not claimed:
                        if not work_before_claim and not self._work_in_flight():
                            no_more_work = True
                            release_pending()
                        else:
                            self._state_changed.wait(
                                min(0.05, self.config.progress_seconds)
                            )
                            self._state_changed.clear()
                    else:
                        assigned = pending[: len(claimed)]
                        self._owned_add(claimed)
                        self._stats_add(claimed=len(claimed))
                        journal.append(
                            event="candidate_batch_claimed",
                            payload={
                                "candidate_ids": list(claimed),
                                "candidate_count": len(claimed),
                                "ready_worker_ids": [
                                    request.worker_id for request in assigned
                                ],
                                "lease_owner": run_id,
                                "lease_seconds": self.config.lease_seconds,
                                "provider_boundary_state": "pre_provider",
                                "leased_queue_depth": 0,
                            },
                        )
                        del pending[: len(claimed)]
                        for request, candidate_id in zip(
                            assigned, claimed, strict=True
                        ):
                            request.candidate_id = candidate_id
                            request.ready.set()
                else:
                    try:
                        pending.append(
                            claim_requests.get(
                                timeout=min(0.05, self.config.progress_seconds)
                            )
                        )
                    except queue.Empty:
                        pass
                now = time.monotonic()
                if now - last_progress >= self.config.progress_seconds:
                    metrics = self._progress(
                        run_id=run_id,
                        queue_depth=len(pending) + claim_requests.qsize(),
                        started=started,
                    )
                    journal.append(event="progress", payload=asdict(metrics))
                    if self._progress_callback is not None:
                        self._progress_callback(metrics)
                    last_progress = now
        except BaseException as exc:
            coordinator_error = exc
            self.request_stop("coordinator_failed")
        finally:
            release_pending()
            for worker in workers:
                worker.join()
            self._heartbeat_stop.set()
            heartbeat.join()

        if coordinator_error is not None:
            raise coordinator_error

        final_progress = self._progress(
            run_id=run_id,
            queue_depth=0,
            started=started,
        )
        if self._progress_callback is not None:
            self._progress_callback(final_progress)
        with self._stats_lock:
            stats = dict(self._stats)
        elapsed = max(0.0, time.monotonic() - started)
        final_document = journal.finalize(
            {
                "claimed": stats["claimed"],
                "completed": stats["completed"],
                "reviewed": stats["reviewed"],
                "rejected": stats["rejected"],
                "infrastructure_quarantine": stats["quarantined"],
                "admitted": stats["admitted"],
                "recovered_expired": stats["recovered_expired"],
                "requeued_expired": len(requeued),
                "quarantined_expired": len(quarantined),
                "maximum_outstanding_claims": self.config.worker_width,
                "leased_queue_capacity": 0,
                "stop_requested": self._stop.is_set(),
                "stop_reason": self._stop_reason,
                "elapsed_seconds": elapsed,
                "final_progress": asdict(final_progress),
            }
        )
        return CampaignRunReport(
            run_id=run_id,
            run_receipt_blake3=str(final_document["receipt_blake3"]),
            claimed=stats["claimed"],
            completed=stats["completed"],
            reviewed=stats["reviewed"],
            rejected=stats["rejected"],
            infrastructure_quarantine=stats["quarantined"],
            admitted=stats["admitted"],
            recovered_expired=stats["recovered_expired"],
            stop_requested=self._stop.is_set(),
            stop_reason=self._stop_reason,
            elapsed_seconds=elapsed,
            final_progress=final_progress,
        )


__all__ = ["CampaignOrchestrationError", "CampaignOrchestrator"]
