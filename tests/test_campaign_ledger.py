from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import os
import signal
import sqlite3
import subprocess
import sys
from threading import Barrier
import time
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest

from eva_agent.campaign import (
    CampaignLedger,
    CampaignLedgerError,
    CandidateQueueRecord,
    FROZEN_SCHEDULE_SIZE,
    exact_6000_plan,
)


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64
SELECTION_DIGEST = "e" * 64


def test_ledger_survives_reopen_and_counts_only_verified_admission(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    candidate_id = str(uuid4())
    ledger = CampaignLedger(path)
    ledger.enqueue_candidate(
        candidate_id=candidate_id,
        source_episode_id="source-1",
        domain="medxpertqa",
        stage="E2E",
        split="train",
        rubric_blake3=DIGEST_A,
    )
    assert ledger.claim(worker_id="worker-1", limit=8, lease_seconds=60) == (candidate_id,)
    ledger.heartbeat(
        worker_id="worker-1", candidate_ids=(candidate_id,), lease_seconds=60
    )
    ledger.mark_provider_boundary(worker_id="worker-1", candidate_id=candidate_id)
    ledger.record_attempt(
        attempt_id=str(uuid4()),
        candidate_id=candidate_id,
        phase="cascade",
        outcome="qualified",
        receipt_blake3=DIGEST_B,
    )
    ledger.finish_candidate(candidate_id, status="reviewed")
    with pytest.raises(CampaignLedgerError, match="signatures"):
        ledger.admit(
            candidate_id=candidate_id,
            sandbox_id=str(uuid4()),
            manifest_blake3=DIGEST_A,
            admission_receipt_blake3=DIGEST_C,
            supervisor_transition_blake3=DIGEST_D,
            signatures_verified=False,
        )
    sandbox_id = str(uuid4())
    ledger.admit(
        candidate_id=candidate_id,
        sandbox_id=sandbox_id,
        manifest_blake3=DIGEST_A,
        admission_receipt_blake3=DIGEST_C,
        supervisor_transition_blake3=DIGEST_D,
        signatures_verified=True,
    )
    reopened = CampaignLedger(path)
    progress = reopened.progress()
    assert progress.admitted == 1
    assert progress.train == 1
    assert progress.target == 6_000


def test_attempts_are_no_replace(tmp_path: Path) -> None:
    ledger = CampaignLedger(tmp_path / "campaign.sqlite3")
    candidate_id = str(uuid4())
    attempt_id = str(uuid4())
    ledger.enqueue_candidate(
        candidate_id=candidate_id,
        source_episode_id="source-2",
        domain="agentclinic",
        stage="S3",
        split="train",
        rubric_blake3=DIGEST_A,
    )
    ledger.record_attempt(
        attempt_id=attempt_id,
        candidate_id=candidate_id,
        phase="rollout",
        outcome="terminal",
        receipt_blake3=DIGEST_B,
    )
    with pytest.raises(CampaignLedgerError, match="already consumed"):
        ledger.record_attempt(
            attempt_id=attempt_id,
            candidate_id=candidate_id,
            phase="rollout",
            outcome="rewritten",
            receipt_blake3=DIGEST_C,
        )


def test_expired_pre_provider_lease_is_safely_requeued_after_reopen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.sqlite3"
    candidate_id = str(uuid4())
    ledger = CampaignLedger(path)
    ledger.enqueue_candidate(
        candidate_id=candidate_id,
        source_episode_id="pre-provider-crash",
        domain="agentclinic",
        stage="S3",
        split="train",
        rubric_blake3=DIGEST_A,
    )
    assert ledger.claim(worker_id="lost-tmux", limit=1, lease_seconds=0.01) == (
        candidate_id,
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE candidates SET lease_expires=? WHERE candidate_id=?",
            (time.time() - 1, candidate_id),
        )

    reopened = CampaignLedger(path)
    assert reopened.recover_expired() == ((candidate_id,), ())
    assert reopened.progress().queued == 1
    assert reopened.progress().infrastructure_quarantine == 0
    assert reopened.claim(
        worker_id="resumed-tmux", limit=1, lease_seconds=60
    ) == (candidate_id,)
    reopened.mark_provider_boundary(
        worker_id="resumed-tmux", candidate_id=candidate_id
    )
    reopened.finish_candidate(candidate_id, status="rejected")
    assert reopened.claim(worker_id="third-run", limit=1, lease_seconds=60) == ()


def test_sighup_process_loss_before_provider_boundary_resumes_from_sqlite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.sqlite3"
    candidate_id = str(uuid4())
    ledger = CampaignLedger(path)
    ledger.enqueue_candidate(
        candidate_id=candidate_id,
        source_episode_id="sighup-pre-provider",
        domain="agentclinic",
        stage="S3",
        split="train",
        rubric_blake3=DIGEST_A,
    )
    child = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys,time; "
                "from pathlib import Path; "
                "from eva_agent.campaign import CampaignLedger; "
                "c=CampaignLedger(Path(sys.argv[1])).claim("
                "worker_id='detached-run',limit=1,lease_seconds=60); "
                "print(c[0],flush=True); time.sleep(60)"
            ),
            str(path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == candidate_id
        child.send_signal(signal.SIGHUP)
        assert child.wait(timeout=10) != 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE candidates SET lease_expires=? WHERE candidate_id=?",
            (time.time() - 1, candidate_id),
        )

    assert CampaignLedger(path).recover_expired() == ((candidate_id,), ())
    assert CampaignLedger(path).progress().queued == 1


def test_expired_crossed_and_legacy_ambiguous_leases_are_quarantined(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.sqlite3"
    crossed, ambiguous, attempted = str(uuid4()), str(uuid4()), str(uuid4())
    ledger = CampaignLedger(path)
    for index, candidate_id in enumerate((crossed, ambiguous, attempted)):
        ledger.enqueue_candidate(
            candidate_id=candidate_id,
            source_episode_id=f"crash-boundary-{index}",
            domain="agentclinic",
            stage="S3",
            split="train",
            rubric_blake3=DIGEST_A,
        )
    assert set(
        ledger.claim(worker_id="lost-process", limit=3, lease_seconds=60)
    ) == {crossed, ambiguous, attempted}
    ledger.mark_provider_boundary(worker_id="lost-process", candidate_id=crossed)
    ledger.record_attempt(
        attempt_id=str(uuid4()),
        candidate_id=attempted,
        phase="incomplete-boundary",
        outcome="ambiguous",
        receipt_blake3=DIGEST_B,
    )
    with sqlite3.connect(path) as connection:
        # NULL is exactly what a pre-migration active lease presents as.  It
        # cannot prove that the older runner had not reached a provider.
        connection.execute(
            """
            UPDATE candidates SET lease_expires=?,provider_boundary_state=NULL
            WHERE candidate_id=?
            """,
            (time.time() - 1, ambiguous),
        )
        connection.execute(
            "UPDATE candidates SET lease_expires=? WHERE candidate_id IN (?,?)",
            (time.time() - 1, crossed, attempted),
        )

    requeued, quarantined = CampaignLedger(path).recover_expired()
    assert requeued == ()
    assert set(quarantined) == {crossed, ambiguous, attempted}
    progress = CampaignLedger(path).progress()
    assert progress.infrastructure_quarantine == 3
    assert progress.active == progress.queued == 0


def test_provider_boundary_marker_is_owned_single_transition(tmp_path: Path) -> None:
    ledger = CampaignLedger(tmp_path / "campaign.sqlite3")
    candidate_id = str(uuid4())
    ledger.enqueue_candidate(
        candidate_id=candidate_id,
        source_episode_id="boundary-owner",
        domain="agentclinic",
        stage="S3",
        split="train",
        rubric_blake3=DIGEST_A,
    )
    assert ledger.claim(worker_id="owner", limit=1, lease_seconds=60) == (
        candidate_id,
    )
    with pytest.raises(CampaignLedgerError, match="owned pre-provider"):
        ledger.mark_provider_boundary(worker_id="other", candidate_id=candidate_id)
    ledger.mark_provider_boundary(worker_id="owner", candidate_id=candidate_id)
    with pytest.raises(CampaignLedgerError, match="owned pre-provider"):
        ledger.mark_provider_boundary(worker_id="owner", candidate_id=candidate_id)


def test_all_training_plan_rejects_zero_target_splits_at_enqueue(tmp_path: Path) -> None:
    ledger = CampaignLedger(tmp_path / "campaign.sqlite3")
    for split in ("development", "sealed_evaluation"):
        with pytest.raises(CampaignLedgerError, match="zero-target"):
            ledger.enqueue_candidate(
                candidate_id=str(uuid4()),
                source_episode_id=f"source-zero-{split}",
                domain="agentclinic",
                stage="S3",
                split=split,
                rubric_blake3=DIGEST_A,
            )


@lru_cache(maxsize=1)
def _exact_bootstrap() -> tuple[CandidateQueueRecord, ...]:
    plan = exact_6000_plan()
    records: list[CandidateQueueRecord] = []
    ordinal = 0
    cell_ranks: dict[tuple[str, str], int] = defaultdict(int)

    def append(cell, tier: str) -> None:
        nonlocal ordinal
        ordinal += 1
        key = (cell.domain, cell.stage)
        cell_ranks[key] += 1
        records.append(
            CandidateQueueRecord(
                candidate_id=str(uuid5(NAMESPACE_URL, f"candidate-{ordinal}")),
                source_episode_id=str(uuid5(NAMESPACE_URL, f"episode-{ordinal}")),
                domain=cell.domain,
                stage=cell.stage,
                split="train",
                rubric_blake3=DIGEST_A,
                queue_ordinal=ordinal,
                cell_rank=cell_ranks[key],
                selection_tier=tier,
            )
        )

    for cell in plan.cells:
        for _ in range(cell.train):
            append(cell, "primary")
    reserve_count = FROZEN_SCHEDULE_SIZE - plan.total
    for index in range(reserve_count):
        append(plan.cells[index % len(plan.cells)], "reserve")
    assert ordinal == FROZEN_SCHEDULE_SIZE
    return tuple(records)


def test_exact_queue_bootstrap_is_atomic_and_idempotent(tmp_path: Path) -> None:
    ledger = CampaignLedger(tmp_path / "campaign.sqlite3")
    records = _exact_bootstrap()
    assert ledger.bootstrap_candidates(
        records, selection_blake3=SELECTION_DIGEST
    ) == FROZEN_SCHEDULE_SIZE
    assert ledger.progress().queued == FROZEN_SCHEDULE_SIZE
    assert ledger.progress().scheduled == FROZEN_SCHEDULE_SIZE
    assert ledger.bootstrap_candidates(records, selection_blake3=SELECTION_DIGEST) == 0

    changed = list(records)
    changed[0] = replace(
        changed[0], source_episode_id=str(uuid5(NAMESPACE_URL, "different-source"))
    )
    with pytest.raises(CampaignLedgerError, match="existing campaign queue differs"):
        ledger.bootstrap_candidates(tuple(changed), selection_blake3=SELECTION_DIGEST)
    assert ledger.progress().queued == FROZEN_SCHEDULE_SIZE


def test_invalid_exact_bootstrap_leaves_ledger_empty(tmp_path: Path) -> None:
    ledger = CampaignLedger(tmp_path / "campaign.sqlite3")
    records = list(_exact_bootstrap())
    records[0] = replace(records[0], split="development")
    with pytest.raises(CampaignLedgerError, match="positive quota"):
        ledger.bootstrap_candidates(tuple(records), selection_blake3=SELECTION_DIGEST)
    assert ledger.progress().queued == 0


def test_reserves_are_claimed_only_for_unfilled_quota_slots(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    ledger = CampaignLedger(path)
    records = _exact_bootstrap()
    ledger.bootstrap_candidates(records, selection_blake3=SELECTION_DIGEST)
    chosen = exact_6000_plan().cells[0]
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE quotas SET admitted=target")
        connection.execute(
            "UPDATE quotas SET admitted=target-2 WHERE domain=? AND stage=? AND split='train'",
            (chosen.domain, chosen.stage),
        )
        connection.execute(
            """
            UPDATE candidates SET status='rejected'
            WHERE domain=? AND stage=? AND split='train' AND selection_tier='primary'
            """,
            (chosen.domain, chosen.stage),
        )

    first = ledger.claim(worker_id="worker-1", limit=128, lease_seconds=60)
    assert len(first) == 2
    with sqlite3.connect(path) as connection:
        assert {
            row[0]
            for row in connection.execute(
                "SELECT selection_tier FROM candidates WHERE candidate_id IN (?,?)",
                first,
            )
        } == {"reserve"}
    assert ledger.claim(worker_id="worker-1", limit=128, lease_seconds=60) == ()
    ledger.mark_provider_boundary(worker_id="worker-1", candidate_id=first[0])
    ledger.finish_candidate(first[0], status="rejected")
    replacement = ledger.claim(worker_id="worker-1", limit=128, lease_seconds=60)
    assert len(replacement) == 1
    ledger.mark_provider_boundary(worker_id="worker-1", candidate_id=first[1])
    ledger.finish_candidate(first[1], status="reviewed")
    # The reviewed candidate is synchronously pending signed admission while
    # the replacement remains active; together they cover both open slots.
    assert ledger.claim(worker_id="worker-1", limit=128, lease_seconds=60) == ()
    ledger.finish_candidate(first[1], status="infrastructure_quarantine")
    assert len(ledger.claim(worker_id="worker-1", limit=128, lease_seconds=60)) == 1
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT DISTINCT domain,stage FROM candidates WHERE candidate_id IN (?,?,?)",
            (*first, *replacement),
        ).fetchall()
    assert rows == [(chosen.domain, chosen.stage)]


def test_bootstrap_requires_contiguous_global_order_and_exact_tiers(tmp_path: Path) -> None:
    ledger = CampaignLedger(tmp_path / "campaign.sqlite3")
    records = list(_exact_bootstrap())
    records[-1] = replace(records[-1], queue_ordinal=records[-2].queue_ordinal)
    with pytest.raises(CampaignLedgerError, match="ordinal is duplicated"):
        ledger.bootstrap_candidates(tuple(records), selection_blake3=SELECTION_DIGEST)

    records = list(_exact_bootstrap())
    records[-1] = replace(records[-1], selection_tier="primary")
    with pytest.raises(CampaignLedgerError, match="tier differs"):
        ledger.bootstrap_candidates(tuple(records), selection_blake3=SELECTION_DIGEST)


def test_reopen_preserves_queue_order_and_requires_complete_bootstrap_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.sqlite3"
    records = _exact_bootstrap()
    ledger = CampaignLedger(path)
    ledger.bootstrap_candidates(records, selection_blake3=SELECTION_DIGEST)
    claimed = ledger.claim(worker_id="worker-1", limit=1, lease_seconds=60)
    assert claimed == (records[0].candidate_id,)
    ledger.mark_provider_boundary(worker_id="worker-1", candidate_id=claimed[0])
    ledger.finish_candidate(claimed[0], status="rejected")

    reopened = CampaignLedger(path)
    assert reopened.bootstrap_candidates(
        records, selection_blake3=SELECTION_DIGEST
    ) == 0
    with pytest.raises(CampaignLedgerError, match="cannot be extended"):
        reopened.enqueue_candidate(
            candidate_id=str(uuid4()),
            source_episode_id="post-freeze-source",
            domain="medxpertqa",
            stage="E2E",
            split="train",
            rubric_blake3=DIGEST_A,
        )
    with sqlite3.connect(path) as connection:
        first = connection.execute(
            """
            SELECT queue_ordinal,cell_rank,selection_tier,status
            FROM candidates WHERE candidate_id=?
            """,
            (records[0].candidate_id,),
        ).fetchone()
    assert first == (1, records[0].cell_rank, "primary", "rejected")

    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM metadata WHERE key='selection_blake3'")
    with pytest.raises(CampaignLedgerError, match="missing frozen schedule metadata"):
        CampaignLedger(path).bootstrap_candidates(
            records, selection_blake3=SELECTION_DIGEST
        )


def test_bootstrap_database_failure_rolls_back_rows_and_commitments(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    ledger = CampaignLedger(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_middle_of_bootstrap
            BEFORE INSERT ON candidates WHEN NEW.queue_ordinal=4500
            BEGIN SELECT RAISE(ABORT, 'injected bootstrap failure'); END
            """
        )
    with pytest.raises(CampaignLedgerError, match="installed atomically"):
        ledger.bootstrap_candidates(
            _exact_bootstrap(), selection_blake3=SELECTION_DIGEST
        )
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM metadata WHERE key IN ('selection_blake3','scheduled_count')"
        ).fetchone()[0] == 0


def test_legacy_queue_schema_migrates_columns_but_nonempty_queue_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE candidates (
                candidate_id TEXT PRIMARY KEY,
                source_episode_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                stage TEXT NOT NULL,
                split TEXT NOT NULL,
                rubric_blake3 TEXT NOT NULL,
                status TEXT NOT NULL,
                lease_owner TEXT,
                lease_expires REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(source_episode_id, domain, stage, split)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(uuid4()),
                "legacy-source",
                "medxpertqa",
                "E2E",
                "train",
                DIGEST_A,
                "queued",
                None,
                None,
                1.0,
                1.0,
            ),
        )
    ledger = CampaignLedger(path)
    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(candidates)")
        }
    assert {"queue_ordinal", "cell_rank", "selection_tier"} <= columns
    assert ledger.progress().scheduled == 1
    with pytest.raises(CampaignLedgerError, match="frozen schedule metadata"):
        ledger.bootstrap_candidates(
            _exact_bootstrap(), selection_blake3=SELECTION_DIGEST
        )


def test_concurrent_admissions_cannot_exceed_exact_cell_target(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    ledger = CampaignLedger(path)
    candidate_ids = (str(uuid4()), str(uuid4()))
    for index, candidate_id in enumerate(candidate_ids):
        ledger.enqueue_candidate(
            candidate_id=candidate_id,
            source_episode_id=f"last-slot-{index}",
            domain="medxpertqa",
            stage="E2E",
            split="train",
            rubric_blake3=DIGEST_A,
        )
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            UPDATE quotas SET admitted=target-1
            WHERE domain='medxpertqa' AND stage='E2E' AND split='train'
            """
        )
        connection.execute(
            "UPDATE candidates SET status='reviewed' WHERE candidate_id IN (?,?)",
            candidate_ids,
        )

    contenders = (CampaignLedger(path), CampaignLedger(path))
    barrier = Barrier(2)

    def attempt(index: int) -> bool:
        barrier.wait(timeout=10)
        try:
            contenders[index].admit(
                candidate_id=candidate_ids[index],
                sandbox_id=str(uuid5(NAMESPACE_URL, f"sandbox-last-slot-{index}")),
                manifest_blake3=DIGEST_A,
                admission_receipt_blake3=DIGEST_B,
                supervisor_transition_blake3=DIGEST_C,
                signatures_verified=True,
            )
        except CampaignLedgerError as exc:
            assert "quota is already full" in str(exc)
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        accepted = list(pool.map(attempt, range(2)))
    assert sum(accepted) == 1
    with sqlite3.connect(path) as connection:
        target, admitted = connection.execute(
            """
            SELECT target,admitted FROM quotas
            WHERE domain='medxpertqa' AND stage='E2E' AND split='train'
            """
        ).fetchone()
        assert admitted == target == 250
        assert connection.execute("SELECT COUNT(*) FROM admissions").fetchone()[0] == 1


def test_execution_allowlist_claims_only_covered_rows_concurrently(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    records = _exact_bootstrap()
    CampaignLedger(path).bootstrap_candidates(
        records, selection_blake3=SELECTION_DIGEST
    )
    eligible = tuple(row.candidate_id for row in records[::37][:96])
    barrier = Barrier(2)

    def claim(index: int) -> tuple[str, ...]:
        ledger = CampaignLedger(path)
        barrier.wait(timeout=10)
        return ledger.claim_eligible(
            worker_id=f"eligible-worker-{index}",
            limit=64,
            lease_seconds=60,
            eligible_candidate_ids=eligible,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        batches = tuple(pool.map(claim, range(2)))
    claimed = tuple(candidate_id for batch in batches for candidate_id in batch)
    assert claimed
    assert len(claimed) == len(set(claimed))
    assert set(claimed) <= set(eligible)
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM candidates WHERE status='queued'"
        ).fetchone()[0] == FROZEN_SCHEDULE_SIZE - len(claimed)
        placeholders = ",".join("?" for _ in eligible)
        uncovered_active = connection.execute(
            f"SELECT COUNT(*) FROM candidates WHERE status='active' "
            f"AND candidate_id NOT IN ({placeholders})",
            eligible,
        ).fetchone()[0]
    assert uncovered_active == 0


def test_empty_execution_allowlist_fails_before_claim(tmp_path: Path) -> None:
    ledger = CampaignLedger(tmp_path / "campaign.sqlite3")
    ledger.bootstrap_candidates(_exact_bootstrap(), selection_blake3=SELECTION_DIGEST)
    with pytest.raises(CampaignLedgerError, match="non-empty"):
        ledger.claim_eligible(
            worker_id="worker",
            limit=1,
            lease_seconds=60,
            eligible_candidate_ids=(),
        )
    assert ledger.progress().queued == FROZEN_SCHEDULE_SIZE


def test_reopen_rejects_quota_rows_outside_the_signed_plan(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    CampaignLedger(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO quotas(domain,stage,split,target,admitted)
            VALUES('uncommitted-domain','S1','train',1,0)
            """
        )
    with pytest.raises(CampaignLedgerError, match="quota table differs"):
        CampaignLedger(path)


def test_different_plan_metadata_is_rejected_before_legacy_schema_migration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "different-plan.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO metadata(key,value) VALUES('plan_blake3',?)", (DIGEST_D,)
        )
        connection.execute(
            """
            CREATE TABLE candidates (
                candidate_id TEXT PRIMARY KEY,
                source_episode_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                stage TEXT NOT NULL,
                split TEXT NOT NULL,
                rubric_blake3 TEXT NOT NULL,
                status TEXT NOT NULL,
                lease_owner TEXT,
                lease_expires REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(source_episode_id, domain, stage, split)
            )
            """
        )
    with pytest.raises(CampaignLedgerError, match="different campaign plan"):
        CampaignLedger(path)
    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(candidates)")
        }
    assert "queue_ordinal" not in columns
