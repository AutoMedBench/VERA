from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid4, uuid5

from eva_agent.campaign import CampaignLedger
from eva_agent.orchestration import CampaignOrchestrator, OrchestrationConfig
from eva_agent.pipeline import InfrastructureQuarantineError
from eva_agent.pipeline.digests import blake3_hex

from test_orchestration_parallel import _FakePipeline, _Source, _ConcurrencyTracker, _enqueue


def _config(*, width: int = 2) -> OrchestrationConfig:
    return OrchestrationConfig(
        worker_width=width,
        queue_capacity=width,
        claim_batch_size=width,
        lease_seconds=10,
        heartbeat_seconds=0.1,
        progress_seconds=0.05,
    )


def test_expired_pre_provider_crash_is_requeued_and_resumes_exactly_once(
    tmp_path: Path,
) -> None:
    crashed, queued = str(uuid4()), str(uuid4())
    ledger = CampaignLedger(tmp_path / "campaign.sqlite3")
    _enqueue(ledger, [crashed, queued])
    assert ledger.claim(worker_id="lost-process", limit=1, lease_seconds=0.02) == (
        crashed,
    )
    time.sleep(0.04)

    source = _Source([crashed, queued])
    tracker = _ConcurrencyTracker(1)
    report = CampaignOrchestrator(
        ledger=ledger,
        candidate_source=source,
        pipeline_factory=lambda _worker_id: _FakePipeline(tracker),
        receipt_root=tmp_path / "receipts",
        config=_config(),
    ).run_until_idle()

    assert report.recovered_expired == 1
    assert report.claimed == report.completed == report.infrastructure_quarantine == 2
    assert report.reviewed == 0
    assert source.loads == {crashed: 1, queued: 1}
    assert tracker.calls == {crashed: 1, queued: 1}
    progress = ledger.progress()
    assert progress.infrastructure_quarantine == 2
    assert progress.active == progress.queued == 0
    with sqlite3.connect(ledger.path) as connection:
        attempts = connection.execute(
            "SELECT candidate_id FROM attempts ORDER BY candidate_id"
        ).fetchall()
    assert attempts == sorted([(crashed,), (queued,)])


class _FailingPipeline:
    def run(self, *, episode, rubric, targets):
        run_id = str(uuid5(NAMESPACE_URL, f"failure:{episode.episode_id}"))
        raise InfrastructureQuarantineError(
            run_id,
            {"middle": "JudgeTimeout", "strong": "ProviderUnavailable"},
        )


def test_provider_and_judge_failures_are_retained_without_score_or_retry(
    tmp_path: Path,
) -> None:
    candidate_id = str(uuid4())
    ledger = CampaignLedger(tmp_path / "campaign.sqlite3")
    _enqueue(ledger, [candidate_id])
    source = _Source([candidate_id])
    report = CampaignOrchestrator(
        ledger=ledger,
        candidate_source=source,
        pipeline_factory=lambda _worker_id: _FailingPipeline(),
        receipt_root=tmp_path / "receipts",
        config=_config(width=1),
    ).run_until_idle()

    assert report.infrastructure_quarantine == 1
    assert report.reviewed == report.rejected == report.admitted == 0
    assert source.loads == {candidate_id: 1}
    events = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "receipts" / report.run_id).rglob("*.json")
    ]
    quarantine = next(
        row for row in events if row["event"] == "candidate_infrastructure_quarantined"
    )
    assert quarantine["payload"]["failures"] == {
        "middle": "JudgeTimeout",
        "strong": "ProviderUnavailable",
    }
    assert quarantine["payload"]["semantic_score_created"] is False
    assert quarantine["payload"]["retry_count"] == 0

    again = CampaignOrchestrator(
        ledger=CampaignLedger(tmp_path / "campaign.sqlite3"),
        candidate_source=source,
        pipeline_factory=lambda _worker_id: _FailingPipeline(),
        receipt_root=tmp_path / "receipts",
        config=_config(width=1),
    ).run_until_idle()
    assert again.claimed == 0
    assert source.loads == {candidate_id: 1}
