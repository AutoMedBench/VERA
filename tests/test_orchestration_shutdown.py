from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
import time
from uuid import uuid4

from eva_agent.campaign import CampaignLedger
from eva_agent.orchestration import CampaignOrchestrator, OrchestrationConfig

from test_orchestration_parallel import _FakePipeline, _Source, _ConcurrencyTracker, _enqueue


class _GatePipeline(_FakePipeline):
    def __init__(self, tracker, started: Event, release: Event) -> None:
        super().__init__(tracker)
        self.started = started
        self.release = release

    def run(self, *, episode, rubric, targets):
        self.started.set()
        assert self.release.wait(timeout=10)
        return super().run(episode=episode, rubric=rubric, targets=targets)


def _config(width: int, max_candidates=None) -> OrchestrationConfig:
    return OrchestrationConfig(
        worker_width=width,
        queue_capacity=1,
        claim_batch_size=1,
        lease_seconds=10,
        heartbeat_seconds=0.05,
        progress_seconds=0.05,
        max_candidates=max_candidates,
    )


def test_stop_drains_claimed_work_and_restart_runs_only_remaining_queue(
    tmp_path: Path,
) -> None:
    candidate_ids = [str(uuid4()) for _ in range(5)]
    ledger = CampaignLedger(tmp_path / "campaign.sqlite3")
    _enqueue(ledger, candidate_ids)
    source = _Source(candidate_ids)
    first_tracker = _ConcurrencyTracker(1)
    started, release = Event(), Event()
    orchestrator = CampaignOrchestrator(
        ledger=ledger,
        candidate_source=source,
        pipeline_factory=lambda _worker_id: _GatePipeline(
            first_tracker, started, release
        ),
        receipt_root=tmp_path / "receipts",
        config=_config(1),
    )
    output = []
    thread = Thread(target=lambda: output.append(orchestrator.run_until_idle()))
    thread.start()
    assert started.wait(timeout=10)
    orchestrator.request_stop("SIGTERM")
    release.set()
    thread.join(timeout=15)
    assert not thread.is_alive()
    first = output[0]
    assert first.stop_requested is True
    assert first.stop_reason == "SIGTERM"
    assert first.claimed == first.completed
    assert 1 <= first.claimed < len(candidate_ids)
    assert ledger.progress().active == 0
    assert ledger.progress().queued == len(candidate_ids) - first.claimed

    remaining = ledger.progress().queued
    second_tracker = _ConcurrencyTracker(remaining)
    second = CampaignOrchestrator(
        ledger=CampaignLedger(tmp_path / "campaign.sqlite3"),
        candidate_source=source,
        pipeline_factory=lambda _worker_id: _FakePipeline(second_tracker),
        receipt_root=tmp_path / "receipts",
        config=OrchestrationConfig(
            worker_width=remaining,
            queue_capacity=remaining,
            claim_batch_size=remaining,
            lease_seconds=10,
            heartbeat_seconds=0.1,
            progress_seconds=0.1,
        ),
    ).run_until_idle()
    assert second.claimed == remaining
    assert ledger.progress().queued == ledger.progress().active == 0
    assert set(source.loads.values()) == {1}


def test_heartbeat_keeps_a_slow_provider_attempt_owned(tmp_path: Path) -> None:
    candidate_id = str(uuid4())
    ledger = CampaignLedger(tmp_path / "campaign.sqlite3")
    _enqueue(ledger, [candidate_id])
    source = _Source([candidate_id])
    tracker = _ConcurrencyTracker(1)
    started, release = Event(), Event()
    orchestrator = CampaignOrchestrator(
        ledger=ledger,
        candidate_source=source,
        pipeline_factory=lambda _worker_id: _GatePipeline(tracker, started, release),
        receipt_root=tmp_path / "receipts",
        config=OrchestrationConfig(
            worker_width=1,
            queue_capacity=1,
            claim_batch_size=1,
            lease_seconds=0.12,
            heartbeat_seconds=0.02,
            progress_seconds=0.05,
        ),
    )
    output = []
    thread = Thread(target=lambda: output.append(orchestrator.run_until_idle()))
    thread.start()
    assert started.wait(timeout=10)
    time.sleep(0.25)
    assert CampaignLedger(ledger.path).quarantine_expired() == ()
    release.set()
    thread.join(timeout=15)
    assert not thread.is_alive()
    assert output[0].infrastructure_quarantine == 1
    assert output[0].reviewed == 0
    assert ledger.progress().infrastructure_quarantine == 1
