from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sqlite3
from threading import Barrier, Event, Lock, Thread
import time
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid4, uuid5

from eva_agent.campaign import CampaignLedger
from eva_agent.orchestration import (
    CampaignOrchestrator,
    CandidateJob,
    OrchestrationConfig,
    ReceiptJournal,
)
from eva_agent.pipeline import BenchmarkEpisode, BenchmarkSource, Cohort, ModelTarget, Stage
from eva_agent.pipeline.digests import blake3_hex


class _Rubric:
    rubric_id = "medxpertqa-e2e-v1"
    version = 1
    digest = "a" * 64
    domain = "medxpertqa"
    stage = "E2E"
    items = ({"item_id": "evidence"},)

    def score(self, item_scores_bps, *, evaluation_id=None):
        return item_scores_bps


TARGETS = tuple(
    ModelTarget(cohort, f"fake-{cohort.value}", "fake")
    for cohort in (Cohort.WEAK, Cohort.MIDDLE, Cohort.STRONG)
)


def _job(candidate_id: str) -> CandidateJob:
    return CandidateJob(
        candidate_id=candidate_id,
        episode=BenchmarkEpisode(
            episode_id=candidate_id,
            source=BenchmarkSource("MedXpertQA", "fixture.json", "fixture-v1"),
            domain="medxpertqa",
            stage=Stage.E2E,
            instruction="Use tools to produce verified medical evidence.",
            policy_context={"question": candidate_id},
            initial_files={"seed.txt": b"evidence\n"},
        ),
        rubric=_Rubric(),
        targets=TARGETS,
    )


class _Source:
    def __init__(self, candidate_ids: list[str]) -> None:
        self.jobs = {candidate_id: _job(candidate_id) for candidate_id in candidate_ids}
        self.loads: Counter[str] = Counter()
        self.lock = Lock()

    def load(self, candidate_id: str) -> CandidateJob:
        with self.lock:
            self.loads[candidate_id] += 1
        return self.jobs[candidate_id]


class _ConcurrencyTracker:
    def __init__(self, width: int) -> None:
        self.barrier = Barrier(width)
        self.width = width
        self.started = 0
        self.active = 0
        self.maximum = 0
        self.calls: Counter[str] = Counter()
        self.lock = Lock()

    def enter(self, episode_id: str) -> bool:
        with self.lock:
            self.started += 1
            first_wave = self.started <= self.width
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            self.calls[episode_id] += 1
        return first_wave

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


class _TransientGapLedger:
    """Expose the reserve-ledger state where all open slots are active."""

    def __init__(self, candidate_ids: list[str]) -> None:
        self.remaining = list(candidate_ids)
        self.active: set[str] = set()
        self.reviewed: set[str] = set()
        self.finished: set[str] = set()
        self.lock = Lock()

    def claim(self, *, worker_id: str, limit: int, lease_seconds: float):
        del worker_id, lease_seconds
        with self.lock:
            if self.active or self.reviewed or not self.remaining:
                return ()
            candidate_id = self.remaining.pop(0)
            self.active.add(candidate_id)
            return (candidate_id,)

    def heartbeat(self, *, worker_id: str, candidate_ids, lease_seconds: float) -> None:
        del worker_id, candidate_ids, lease_seconds

    def quarantine_expired(self):
        return ()

    def recover_expired(self):
        return (), ()

    def mark_provider_boundary(self, *, worker_id: str, candidate_id: str) -> None:
        del worker_id
        with self.lock:
            if candidate_id not in self.active:
                raise AssertionError("provider boundary candidate is not active")

    def record_attempt(self, **kwargs) -> None:
        del kwargs

    def finish_candidate(self, candidate_id: str, *, status: str) -> None:
        with self.lock:
            if candidate_id in self.active:
                self.active.remove(candidate_id)
                if status == "reviewed":
                    self.reviewed.add(candidate_id)
                else:
                    self.finished.add(candidate_id)
                return
            if status == "infrastructure_quarantine" and candidate_id in self.reviewed:
                self.reviewed.remove(candidate_id)
                self.finished.add(candidate_id)
                return
            raise AssertionError("candidate terminal transition differs")

    def progress(self):
        with self.lock:
            return SimpleNamespace(
                target=6_000,
                admitted=0,
                train=0,
                development=0,
                sealed_evaluation=0,
                queued=len(self.remaining),
                active=len(self.active),
                rejected=0,
                infrastructure_quarantine=0,
            )


class _BoundaryRaceLedger(_TransientGapLedger):
    """Return one stale empty claim after the last owned task commits."""

    def __init__(self, candidate_ids: list[str]) -> None:
        super().__init__(candidate_ids)
        self.orchestrator = None
        self.forced_boundary = False

    def claim(self, *, worker_id: str, limit: int, lease_seconds: float):
        with self.lock:
            saw_active = bool(self.active)
        if saw_active and not self.forced_boundary:
            self.forced_boundary = True
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                orchestrator = self.orchestrator
                if orchestrator is not None:
                    with orchestrator._owned_lock:
                        if not orchestrator._owned:
                            return ()
                time.sleep(0.0005)
            raise AssertionError("worker did not cross the owned-task boundary")
        return super().claim(
            worker_id=worker_id,
            limit=limit,
            lease_seconds=lease_seconds,
        )


class _FakePipeline:
    def __init__(self, tracker: _ConcurrencyTracker) -> None:
        self.tracker = tracker

    def run(self, *, episode, rubric, targets):
        first_wave = self.tracker.enter(episode.episode_id)
        try:
            if first_wave:
                self.tracker.barrier.wait(timeout=15)
            time.sleep(0.002)
            sandbox_id = str(uuid5(NAMESPACE_URL, f"sandbox:{episode.episode_id}"))
            run_id = str(uuid5(NAMESPACE_URL, f"run:{episode.episode_id}"))
            return SimpleNamespace(
                run_id=run_id,
                result_blake3=blake3_hex({"episode": episode.episode_id}),
                sandbox_manifests=(
                    SimpleNamespace(
                        sandbox_id=sandbox_id,
                        manifest_blake3=blake3_hex({"sandbox": sandbox_id}),
                    ),
                ),
                separation=SimpleNamespace(policy_passed=True),
                recommendation=SimpleNamespace(
                    recommendation="eligible_for_signed_supervisor_admission"
                ),
            )
        finally:
            self.tracker.leave()


class _RejectedPipeline(_FakePipeline):
    def run(self, *, episode, rubric, targets):
        result = super().run(episode=episode, rubric=rubric, targets=targets)
        result.separation.policy_passed = False
        result.recommendation.recommendation = "not_eligible"
        return result


class _HeldRejectedPipeline:
    def __init__(
        self,
        tracker: _ConcurrencyTracker,
        *,
        expected_width: int,
        all_started: Event,
        release: Event,
    ) -> None:
        self.tracker = tracker
        self.expected_width = expected_width
        self.all_started = all_started
        self.release = release

    def run(self, *, episode, rubric, targets):
        del rubric, targets
        self.tracker.enter(episode.episode_id)
        try:
            with self.tracker.lock:
                if self.tracker.started == self.expected_width:
                    self.all_started.set()
            assert self.release.wait(timeout=15)
            sandbox_id = str(uuid5(NAMESPACE_URL, f"sandbox:{episode.episode_id}"))
            return SimpleNamespace(
                run_id=str(uuid5(NAMESPACE_URL, f"run:{episode.episode_id}")),
                result_blake3=blake3_hex({"episode": episode.episode_id}),
                sandbox_manifests=(
                    SimpleNamespace(
                        sandbox_id=sandbox_id,
                        manifest_blake3=blake3_hex({"sandbox": sandbox_id}),
                    ),
                ),
                separation=SimpleNamespace(policy_passed=False),
                recommendation=SimpleNamespace(recommendation="not_eligible"),
            )
        finally:
            self.tracker.leave()


def _enqueue(ledger: CampaignLedger, candidate_ids: list[str]) -> None:
    for index, candidate_id in enumerate(candidate_ids):
        ledger.enqueue_candidate(
            candidate_id=candidate_id,
            source_episode_id=f"source-{index}",
            domain="medxpertqa",
            stage="E2E",
            split="train",
            rubric_blake3=_Rubric.digest,
        )


def test_width_64_is_bounded_parallel_and_candidates_execute_exactly_once(
    tmp_path: Path,
) -> None:
    width = 64
    candidate_ids = [str(uuid4()) for _ in range(width)]
    ledger = CampaignLedger(tmp_path / "campaign.sqlite3")
    _enqueue(ledger, candidate_ids)
    source = _Source(candidate_ids)
    tracker = _ConcurrencyTracker(width)
    metrics = []
    orchestrator = CampaignOrchestrator(
        ledger=ledger,
        candidate_source=source,
        pipeline_factory=lambda _worker_id: _FakePipeline(tracker),
        receipt_root=tmp_path / "receipts",
        config=OrchestrationConfig(
            worker_width=width,
            queue_capacity=width,
            claim_batch_size=width,
            lease_seconds=30,
            heartbeat_seconds=1,
            progress_seconds=0.01,
        ),
        progress_callback=metrics.append,
    )
    report = orchestrator.run_until_idle()

    assert report.claimed == report.completed == report.infrastructure_quarantine == width
    assert report.reviewed == 0
    assert report.admitted == 0
    assert tracker.maximum == width
    assert set(tracker.calls.values()) == {1}
    assert set(source.loads.values()) == {1}
    assert metrics[-1].queue_depth == 0
    assert metrics[-1].running_workers == 0
    assert metrics[-1].run_completed == width
    assert metrics[-1].admitted == 0

    # Terminal rows can never be selected as replacement candidates.
    second = CampaignOrchestrator(
        ledger=CampaignLedger(tmp_path / "campaign.sqlite3"),
        candidate_source=source,
        pipeline_factory=lambda _worker_id: _FakePipeline(tracker),
        receipt_root=tmp_path / "receipts",
        config=OrchestrationConfig(
            worker_width=4,
            queue_capacity=4,
            claim_batch_size=4,
            lease_seconds=30,
            heartbeat_seconds=1,
            progress_seconds=1,
        ),
    ).run_until_idle()
    assert second.claimed == second.completed == 0
    assert set(tracker.calls.values()) == {1}

    receipt_files = list((tmp_path / "receipts" / report.run_id).rglob("*.json"))
    assert receipt_files
    assert all(ReceiptJournal.verify(path) for path in receipt_files)
    assert ReceiptJournal.verify_run(tmp_path / "receipts", report.run_id)


def test_ready_worker_jit_claims_never_fill_the_configured_candidate_queue(
    tmp_path: Path,
) -> None:
    """A large queue cannot add leases beyond the workers at a provider edge."""

    width = 32
    candidate_ids = [str(uuid4()) for _ in range(width * 3)]
    ledger = CampaignLedger(tmp_path / "campaign.sqlite3")
    _enqueue(ledger, candidate_ids)
    source = _Source(candidate_ids)
    tracker = _ConcurrencyTracker(width)
    all_started, release = Event(), Event()
    orchestrator = CampaignOrchestrator(
        ledger=ledger,
        candidate_source=source,
        pipeline_factory=lambda _worker_id: _HeldRejectedPipeline(
            tracker,
            expected_width=width,
            all_started=all_started,
            release=release,
        ),
        receipt_root=tmp_path / "receipts",
        config=OrchestrationConfig(
            worker_width=width,
            queue_capacity=width * 2,
            claim_batch_size=width,
            lease_seconds=30,
            heartbeat_seconds=1,
            progress_seconds=0.01,
        ),
    )
    reports = []
    thread = Thread(target=lambda: reports.append(orchestrator.run_until_idle()))
    thread.start()
    assert all_started.wait(timeout=15)
    with sqlite3.connect(ledger.path) as connection:
        active, crossed = connection.execute(
            """
            SELECT COUNT(*),SUM(provider_boundary_state='crossed')
            FROM candidates WHERE status='active'
            """
        ).fetchone()
    assert active == crossed == width
    assert ledger.progress().queued == width * 2

    orchestrator.request_stop("SIGHUP")
    release.set()
    thread.join(timeout=20)
    assert not thread.is_alive()
    report = reports[0]
    assert report.claimed == report.completed == report.rejected == width
    assert ledger.progress().active == 0
    assert ledger.progress().queued == width * 2


def test_profile_512_configuration_schedules_without_candidate_prelease_queue(
    tmp_path: Path,
) -> None:
    """The production width/queue topology remains valid with JIT claiming."""

    candidate_ids = [str(uuid4()) for _ in range(8)]
    ledger = CampaignLedger(tmp_path / "campaign.sqlite3")
    _enqueue(ledger, candidate_ids)
    source = _Source(candidate_ids)
    tracker = _ConcurrencyTracker(8)
    metrics = []
    report = CampaignOrchestrator(
        ledger=ledger,
        candidate_source=source,
        pipeline_factory=lambda _worker_id: _RejectedPipeline(tracker),
        receipt_root=tmp_path / "receipts",
        config=OrchestrationConfig(
            worker_width=512,
            queue_capacity=1_024,
            claim_batch_size=512,
            lease_seconds=30,
            heartbeat_seconds=1,
            progress_seconds=0.01,
        ),
        progress_callback=metrics.append,
    ).run_until_idle()

    assert report.claimed == report.completed == report.rejected == len(candidate_ids)
    assert tracker.maximum == len(candidate_ids)
    assert set(tracker.calls.values()) == {1}
    assert all(metric.queue_depth <= 512 for metric in metrics)
    documents = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "receipts" / report.run_id).rglob("*.json")
    ]
    start = next(row for row in documents if row["event"] == "run_started")
    assert start["payload"]["maximum_outstanding_claims"] == 512
    assert start["payload"]["leased_queue_capacity"] == 0


def test_scheduler_waits_through_active_slot_gap_before_claiming_reserve(
    tmp_path: Path,
) -> None:
    candidate_ids = [str(uuid4()), str(uuid4())]
    ledger = _TransientGapLedger(candidate_ids)
    source = _Source(candidate_ids)
    tracker = _ConcurrencyTracker(1)
    report = CampaignOrchestrator(
        ledger=ledger,
        candidate_source=source,
        pipeline_factory=lambda _worker_id: _FakePipeline(tracker),
        receipt_root=tmp_path / "receipts",
        config=OrchestrationConfig(
            worker_width=1,
            queue_capacity=1,
            claim_batch_size=1,
            lease_seconds=30,
            heartbeat_seconds=1,
            progress_seconds=0.01,
        ),
    ).run_until_idle()

    assert report.claimed == report.completed == 2
    assert ledger.finished == set(candidate_ids)
    assert set(source.loads.values()) == {1}


def test_jit_scheduler_does_not_claim_during_terminal_owned_task_boundary(
    tmp_path: Path,
) -> None:
    candidate_ids = [str(uuid4()), str(uuid4())]
    ledger = _BoundaryRaceLedger(candidate_ids)
    source = _Source(candidate_ids)
    tracker = _ConcurrencyTracker(1)
    orchestrator = CampaignOrchestrator(
        ledger=ledger,
        candidate_source=source,
        pipeline_factory=lambda _worker_id: _RejectedPipeline(tracker),
        receipt_root=tmp_path / "receipts",
        config=OrchestrationConfig(
            worker_width=1,
            queue_capacity=1,
            claim_batch_size=1,
            lease_seconds=30,
            heartbeat_seconds=1,
            progress_seconds=0.01,
        ),
    )
    ledger.orchestrator = orchestrator
    report = orchestrator.run_until_idle()

    # The ready worker asks for its replacement only after completing the
    # terminal transition, so the former speculative empty-claim race is gone.
    assert ledger.forced_boundary is False
    assert report.claimed == report.completed == report.rejected == 2
    assert ledger.finished == set(candidate_ids)
    assert set(source.loads.values()) == {1}
