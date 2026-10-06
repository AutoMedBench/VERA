"""SQLite-backed campaign state that survives terminal and SSH interruption."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import stat
import time
from typing import Iterator
from uuid import UUID

from eva_agent.pipeline.digests import is_blake3

from .plan import CampaignPlan, exact_6000_plan, verify_plan


class CampaignLedgerError(ValueError):
    """A durable campaign transition failed closed."""


@dataclass(frozen=True)
class CampaignProgress:
    target: int
    scheduled: int
    admitted: int
    train: int
    development: int
    sealed_evaluation: int
    queued: int
    active: int
    rejected: int
    infrastructure_quarantine: int
    reserve_unused: int


@dataclass(frozen=True, slots=True)
class CandidateQueueRecord:
    """One source/rubric binding prepared for atomic campaign bootstrap."""

    candidate_id: str
    source_episode_id: str
    domain: str
    stage: str
    split: str
    rubric_blake3: str
    queue_ordinal: int
    cell_rank: int
    selection_tier: str


FROZEN_SCHEDULE_SIZE = 9_000


def _uuid(value: str, label: str) -> str:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError):
        raise CampaignLedgerError(f"{label} must be a UUID") from None
    if str(parsed) != value:
        raise CampaignLedgerError(f"{label} must be canonical UUID text")
    return value


class CampaignLedger:
    """No-replace attempts plus transactional quota accounting."""

    def __init__(self, path: Path, plan: CampaignPlan | None = None) -> None:
        requested_path = Path(path)
        if requested_path.is_symlink():
            raise CampaignLedgerError("campaign ledger cannot be a symlink")
        self.path = requested_path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.plan = plan or exact_6000_plan()
        verify_plan(self.plan)
        self._preflight_existing_plan()
        self._initialize()

    def _preflight_existing_plan(self) -> None:
        """Reject a different committed plan before running any schema migration."""

        if not self.path.exists():
            return
        try:
            uri = f"{self.path.as_uri()}?mode=ro"
            with sqlite3.connect(uri, uri=True) as connection:
                metadata_exists = connection.execute(
                    """
                    SELECT 1 FROM sqlite_master
                    WHERE type='table' AND name='metadata'
                    """
                ).fetchone()
                if metadata_exists is None:
                    return
                existing = connection.execute(
                    "SELECT value FROM metadata WHERE key='plan_blake3'"
                ).fetchone()
        except sqlite3.Error as exc:
            raise CampaignLedgerError(
                "existing campaign ledger cannot be reopened safely"
            ) from exc
        if existing is not None and existing[0] != self.plan.plan_blake3:
            raise CampaignLedgerError("existing ledger uses a different campaign plan")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            if connection.in_transaction:
                connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS quotas (
                    domain TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    split TEXT NOT NULL,
                    target INTEGER NOT NULL CHECK(target >= 0),
                    admitted INTEGER NOT NULL DEFAULT 0 CHECK(admitted >= 0),
                    PRIMARY KEY(domain, stage, split),
                    CHECK(admitted <= target)
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    source_episode_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    split TEXT NOT NULL,
                    rubric_blake3 TEXT NOT NULL,
                    queue_ordinal INTEGER,
                    cell_rank INTEGER,
                    selection_tier TEXT,
                    status TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_expires REAL,
                    provider_boundary_state TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(source_episode_id, domain, stage, split)
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
                    phase TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    receipt_blake3 TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(candidate_id, phase, receipt_blake3)
                );
                CREATE TABLE IF NOT EXISTS admissions (
                    sandbox_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL UNIQUE REFERENCES candidates(candidate_id),
                    manifest_blake3 TEXT NOT NULL,
                    admission_receipt_blake3 TEXT NOT NULL,
                    supervisor_transition_blake3 TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(candidates)").fetchall()
            }
            for name, declaration in (
                ("queue_ordinal", "INTEGER"),
                ("cell_rank", "INTEGER"),
                ("selection_tier", "TEXT"),
                ("provider_boundary_state", "TEXT"),
            ):
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE candidates ADD COLUMN {name} {declaration}"
                    )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS candidates_queue_ordinal_unique
                ON candidates(queue_ordinal) WHERE queue_ordinal IS NOT NULL
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS candidates_cell_rank_unique
                ON candidates(domain,stage,split,cell_rank)
                WHERE cell_rank IS NOT NULL
                """
            )
        finally:
            connection.close()
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT value FROM metadata WHERE key='plan_blake3'"
            ).fetchone()
            if existing is not None and existing["value"] != self.plan.plan_blake3:
                raise CampaignLedgerError("existing ledger uses a different campaign plan")
            connection.execute(
                "INSERT OR IGNORE INTO metadata(key,value) VALUES('plan_blake3',?)",
                (self.plan.plan_blake3,),
            )
            for cell in self.plan.cells:
                values = {
                    "train": cell.train,
                    "development": cell.development,
                    "sealed_evaluation": cell.sealed_evaluation,
                }
                for split, target in values.items():
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO quotas(domain,stage,split,target)
                        VALUES(?,?,?,?)
                        """,
                        (cell.domain, cell.stage, split, target),
                    )
                    observed = connection.execute(
                        "SELECT target FROM quotas WHERE domain=? AND stage=? AND split=?",
                        (cell.domain, cell.stage, split),
                    ).fetchone()
                    if observed is None or observed["target"] != target:
                        raise CampaignLedgerError("existing quota differs from signed plan")
            expected_quotas = {
                (cell.domain, cell.stage, split): target
                for cell in self.plan.cells
                for split, target in (
                    ("train", cell.train),
                    ("development", cell.development),
                    ("sealed_evaluation", cell.sealed_evaluation),
                )
            }
            observed_quotas = {
                (str(row["domain"]), str(row["stage"]), str(row["split"])): int(
                    row["target"]
                )
                for row in connection.execute(
                    "SELECT domain,stage,split,target FROM quotas"
                ).fetchall()
            }
            if observed_quotas != expected_quotas:
                raise CampaignLedgerError("existing quota table differs from signed plan")
        os_mode = stat.S_IMODE(self.path.stat().st_mode)
        if os_mode & 0o077:
            self.path.chmod(0o600)

    def enqueue_candidate(
        self,
        *,
        candidate_id: str,
        source_episode_id: str,
        domain: str,
        stage: str,
        split: str,
        rubric_blake3: str,
    ) -> None:
        _uuid(candidate_id, "candidate_id")
        if not source_episode_id or not is_blake3(rubric_blake3):
            raise CampaignLedgerError("candidate source or rubric commitment differs")
        now = time.time()
        with self._transaction() as connection:
            frozen = connection.execute(
                """
                SELECT COUNT(*) AS count FROM metadata
                WHERE key IN ('selection_blake3','scheduled_count')
                """
            ).fetchone()
            if frozen is not None and int(frozen["count"]) > 0:
                raise CampaignLedgerError("frozen campaign schedule cannot be extended")
            quota = connection.execute(
                "SELECT target FROM quotas WHERE domain=? AND stage=? AND split=?",
                (domain, stage, split),
            ).fetchone()
            if quota is None:
                raise CampaignLedgerError("candidate does not map to a campaign quota")
            if int(quota["target"]) == 0:
                raise CampaignLedgerError("candidate maps to a zero-target campaign split")
            try:
                connection.execute(
                    """
                    INSERT INTO candidates(
                        candidate_id,source_episode_id,domain,stage,split,
                        rubric_blake3,queue_ordinal,cell_rank,selection_tier,
                        status,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,NULL,NULL,'manual','queued',?,?)
                    """,
                    (
                        candidate_id,
                        source_episode_id,
                        domain,
                        stage,
                        split,
                        rubric_blake3,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise CampaignLedgerError("candidate identity or source cell is already consumed") from exc

    def bootstrap_candidates(
        self,
        records: tuple[CandidateQueueRecord, ...],
        *,
        selection_blake3: str,
    ) -> int:
        """Atomically install the full frozen schedule, or verify an existing one.

        A repeated bootstrap after a clean prior commit is idempotent.  Any
        partial or different 9,000-row primary+reserve schedule fails closed;
        the method never fills gaps in an already initialized campaign.
        """

        if not isinstance(records, tuple) or len(records) != FROZEN_SCHEDULE_SIZE:
            raise CampaignLedgerError("bootstrap must contain the exact frozen schedule")
        if not is_blake3(selection_blake3):
            raise CampaignLedgerError("bootstrap selection commitment differs")
        expected_quotas = {
            (cell.domain, cell.stage, split): target
            for cell in self.plan.cells
            for split, target in (
                ("train", cell.train),
                ("development", cell.development),
                ("sealed_evaluation", cell.sealed_evaluation),
            )
            if target > 0
        }
        observed_capacity: dict[tuple[str, str, str], int] = {}
        observed_primary: dict[tuple[str, str, str], int] = {}
        ranks_by_cell: dict[tuple[str, str, str], set[int]] = {}
        candidate_ids: set[str] = set()
        source_episode_ids: set[str] = set()
        queue_ordinals: set[int] = set()
        normalized: list[tuple[str, str, str, str, str, str, int, int, str]] = []
        for record in records:
            if not isinstance(record, CandidateQueueRecord):
                raise CampaignLedgerError("bootstrap row type differs")
            _uuid(record.candidate_id, "candidate_id")
            _uuid(record.source_episode_id, "source_episode_id")
            if not is_blake3(record.rubric_blake3):
                raise CampaignLedgerError("candidate source or rubric commitment differs")
            if (
                type(record.queue_ordinal) is not int
                or record.queue_ordinal < 1
                or type(record.cell_rank) is not int
                or record.cell_rank < 1
                or record.selection_tier not in {"primary", "reserve"}
            ):
                raise CampaignLedgerError("bootstrap queue rank or tier differs")
            if record.candidate_id in candidate_ids:
                raise CampaignLedgerError("bootstrap candidate identity is duplicated")
            if record.source_episode_id in source_episode_ids:
                raise CampaignLedgerError("bootstrap source episode is duplicated")
            if record.queue_ordinal in queue_ordinals:
                raise CampaignLedgerError("bootstrap queue ordinal is duplicated")
            candidate_ids.add(record.candidate_id)
            source_episode_ids.add(record.source_episode_id)
            queue_ordinals.add(record.queue_ordinal)
            key = (record.domain, record.stage, record.split)
            if key not in expected_quotas:
                raise CampaignLedgerError("bootstrap row does not map to a positive quota")
            observed_capacity[key] = observed_capacity.get(key, 0) + 1
            ranks_by_cell.setdefault(key, set()).add(record.cell_rank)
            target = expected_quotas[key]
            expected_tier = "primary" if record.cell_rank <= target else "reserve"
            if record.selection_tier != expected_tier:
                raise CampaignLedgerError("bootstrap primary/reserve tier differs")
            if record.selection_tier == "primary":
                observed_primary[key] = observed_primary.get(key, 0) + 1
            normalized.append(
                (
                    record.candidate_id,
                    record.source_episode_id,
                    record.domain,
                    record.stage,
                    record.split,
                    record.rubric_blake3,
                    record.queue_ordinal,
                    record.cell_rank,
                    record.selection_tier,
                )
            )
        if queue_ordinals != set(range(1, FROZEN_SCHEDULE_SIZE + 1)):
            raise CampaignLedgerError("bootstrap queue ordinals are not contiguous")
        if observed_primary != expected_quotas or set(observed_capacity) != set(expected_quotas):
            raise CampaignLedgerError("bootstrap primary quotas differ from the signed plan")
        for key, count in observed_capacity.items():
            if count < expected_quotas[key] or ranks_by_cell[key] != set(range(1, count + 1)):
                raise CampaignLedgerError("bootstrap cell ranks or reserve capacity differ")
        normalized.sort(key=lambda row: row[6])
        now = time.time()
        with self._transaction() as connection:
            metadata = connection.execute(
                "SELECT value FROM metadata WHERE key='selection_blake3'"
            ).fetchone()
            scheduled_metadata = connection.execute(
                "SELECT value FROM metadata WHERE key='scheduled_count'"
            ).fetchone()
            if metadata is not None and metadata["value"] != selection_blake3:
                raise CampaignLedgerError("existing campaign selection differs from bootstrap")
            existing = connection.execute(
                """
                SELECT candidate_id,source_episode_id,domain,stage,split,rubric_blake3,
                       queue_ordinal,cell_rank,selection_tier
                FROM candidates ORDER BY queue_ordinal
                """
            ).fetchall()
            if existing:
                if metadata is None or scheduled_metadata is None:
                    raise CampaignLedgerError(
                        "existing campaign queue is missing frozen schedule metadata"
                    )
                if scheduled_metadata["value"] != str(FROZEN_SCHEDULE_SIZE):
                    raise CampaignLedgerError(
                        "existing campaign scheduled count differs from bootstrap"
                    )
                if len(existing) != FROZEN_SCHEDULE_SIZE or any(
                    row["queue_ordinal"] is None
                    or row["cell_rank"] is None
                    or row["selection_tier"] is None
                    for row in existing
                ):
                    raise CampaignLedgerError(
                        "existing campaign queue predates the frozen schedule metadata"
                    )
                observed = [
                    (
                        str(row["candidate_id"]),
                        str(row["source_episode_id"]),
                        str(row["domain"]),
                        str(row["stage"]),
                        str(row["split"]),
                        str(row["rubric_blake3"]),
                        int(row["queue_ordinal"]),
                        int(row["cell_rank"]),
                        str(row["selection_tier"]),
                    )
                    for row in existing
                ]
                if observed != normalized:
                    raise CampaignLedgerError("existing campaign queue differs from bootstrap")
                return 0
            if metadata is not None or scheduled_metadata is not None:
                raise CampaignLedgerError(
                    "existing frozen schedule metadata has no campaign queue"
                )
            connection.execute(
                "INSERT OR IGNORE INTO metadata(key,value) VALUES('selection_blake3',?)",
                (selection_blake3,),
            )
            connection.execute(
                "INSERT OR IGNORE INTO metadata(key,value) VALUES('scheduled_count',?)",
                (str(FROZEN_SCHEDULE_SIZE),),
            )
            try:
                connection.executemany(
                    """
                    INSERT INTO candidates(
                        candidate_id,source_episode_id,domain,stage,split,
                        rubric_blake3,queue_ordinal,cell_rank,selection_tier,
                        status,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,'queued',?,?)
                    """,
                    (row + (now, now) for row in normalized),
                )
            except sqlite3.IntegrityError as exc:
                raise CampaignLedgerError(
                    "frozen campaign schedule could not be installed atomically"
                ) from exc
        return len(normalized)

    def claim(self, *, worker_id: str, limit: int, lease_seconds: float) -> tuple[str, ...]:
        """Claim from the complete frozen schedule."""

        return self._claim(
            worker_id=worker_id,
            limit=limit,
            lease_seconds=lease_seconds,
            eligible_candidate_ids=None,
        )

    def claim_eligible(
        self,
        *,
        worker_id: str,
        limit: int,
        lease_seconds: float,
        eligible_candidate_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Atomically claim only candidates in a verified execution catalog.

        The complete 9,000-row queue and its per-cell quota accounting remain
        authoritative.  This deployment-time allowlist merely prevents a
        candidate without a signed executable binding from crossing the claim
        boundary.  It never rewrites or quarantines an uncovered queued row.
        """

        if not isinstance(eligible_candidate_ids, tuple) or not eligible_candidate_ids:
            raise CampaignLedgerError("eligible candidate inventory must be non-empty")
        if len(set(eligible_candidate_ids)) != len(eligible_candidate_ids):
            raise CampaignLedgerError("eligible candidate inventory is duplicated")
        for candidate_id in eligible_candidate_ids:
            _uuid(candidate_id, "eligible_candidate_id")
        return self._claim(
            worker_id=worker_id,
            limit=limit,
            lease_seconds=lease_seconds,
            eligible_candidate_ids=eligible_candidate_ids,
        )

    def _claim(
        self,
        *,
        worker_id: str,
        limit: int,
        lease_seconds: float,
        eligible_candidate_ids: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        if not worker_id or limit < 1 or lease_seconds <= 0:
            raise CampaignLedgerError("worker claim parameters differ")
        eligible_json = (
            None
            if eligible_candidate_ids is None
            else json.dumps(eligible_candidate_ids, separators=(",", ":"))
        )
        now = time.time()
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE candidates AS candidate
                SET status='reserve_unused',updated_at=?
                WHERE candidate.status='queued' AND EXISTS (
                    SELECT 1 FROM quotas AS quota
                    WHERE quota.domain=candidate.domain
                      AND quota.stage=candidate.stage
                      AND quota.split=candidate.split
                      AND quota.admitted>=quota.target
                )
                """,
                (now,),
            )
            inconsistent = connection.execute(
                """
                SELECT candidate_id FROM candidates AS candidate
                WHERE candidate.status='queued' AND (
                    candidate.provider_boundary_state IS NOT NULL OR EXISTS (
                        SELECT 1 FROM attempts
                        WHERE attempts.candidate_id=candidate.candidate_id
                    )
                )
                ORDER BY candidate_id LIMIT 1
                """
            ).fetchone()
            if inconsistent is not None:
                raise CampaignLedgerError(
                    "queued candidate retains provider-boundary or attempt evidence"
                )
            rows = connection.execute(
                """
                WITH eligible AS (
                    SELECT candidate.candidate_id,
                           candidate.queue_ordinal,
                           candidate.created_at,
                           ROW_NUMBER() OVER (
                               PARTITION BY candidate.domain,candidate.stage,candidate.split
                               ORDER BY COALESCE(candidate.queue_ordinal,9223372036854775807),
                                        candidate.created_at,candidate.candidate_id
                           ) AS cell_queue_position,
                           quota.target-quota.admitted-(
                               SELECT COUNT(*) FROM candidates AS active
                               WHERE active.domain=candidate.domain
                                 AND active.stage=candidate.stage
                                 AND active.split=candidate.split
                                 AND active.status IN ('active','reviewed')
                           ) AS available_slots
                    FROM candidates AS candidate
                    JOIN quotas AS quota
                      ON quota.domain=candidate.domain
                     AND quota.stage=candidate.stage
                     AND quota.split=candidate.split
                    WHERE candidate.status='queued' AND quota.admitted<quota.target
                      AND (
                          ? IS NULL OR candidate.candidate_id IN (
                              SELECT value FROM json_each(?)
                          )
                      )
                )
                SELECT candidate_id FROM eligible
                WHERE cell_queue_position<=available_slots
                ORDER BY COALESCE(queue_ordinal,9223372036854775807),created_at,candidate_id
                LIMIT ?
                """,
                (eligible_json, eligible_json, limit),
            ).fetchall()
            identities = tuple(str(row["candidate_id"]) for row in rows)
            for candidate_id in identities:
                updated = connection.execute(
                    """
                    UPDATE candidates SET status='active',lease_owner=?,lease_expires=?,
                        provider_boundary_state='pre_provider',updated_at=?
                    WHERE candidate_id=? AND status='queued'
                    """,
                    (worker_id, now + lease_seconds, now, candidate_id),
                ).rowcount
                if updated != 1:
                    raise CampaignLedgerError("candidate claim transition differs")
            return identities

    def mark_provider_boundary(self, *, worker_id: str, candidate_id: str) -> None:
        """Durably make a lease non-replayable before its first provider call.

        The transaction commits before control returns to the caller.  A crash
        after this method begins is therefore treated conservatively: either
        the row remains ``pre_provider`` and is provably safe to requeue, or it
        is ``crossed`` and can never be replayed after lease expiry.
        """

        if not worker_id:
            raise CampaignLedgerError("provider boundary worker identity differs")
        _uuid(candidate_id, "candidate_id")
        with self._transaction() as connection:
            updated = connection.execute(
                """
                UPDATE candidates SET provider_boundary_state='crossed',updated_at=?
                WHERE candidate_id=? AND status='active' AND lease_owner=?
                  AND provider_boundary_state='pre_provider'
                """,
                (time.time(), candidate_id, worker_id),
            ).rowcount
            if updated != 1:
                raise CampaignLedgerError(
                    "provider boundary requires an owned pre-provider lease"
                )

    def heartbeat(
        self, *, worker_id: str, candidate_ids: tuple[str, ...], lease_seconds: float
    ) -> None:
        if not worker_id or not candidate_ids or lease_seconds <= 0:
            raise CampaignLedgerError("worker heartbeat parameters differ")
        for candidate_id in candidate_ids:
            _uuid(candidate_id, "candidate_id")
        now = time.time()
        with self._transaction() as connection:
            for candidate_id in candidate_ids:
                updated = connection.execute(
                    """
                    UPDATE candidates SET lease_expires=?,updated_at=?
                    WHERE candidate_id=? AND status='active' AND lease_owner=?
                    """,
                    (now + lease_seconds, now, candidate_id, worker_id),
                ).rowcount
                if updated != 1:
                    raise CampaignLedgerError("worker does not own the active lease")

    def recover_expired(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Recover expired leases according to their durable provider boundary.

        Returns ``(requeued, quarantined)``.  Only the exact ``pre_provider``
        state proves that no provider call could have started.  ``crossed``,
        NULL rows created by older runners, and every unknown/tampered value
        are ambiguous and therefore fail closed into quarantine.
        """

        now = time.time()
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT candidate_id,provider_boundary_state,
                       EXISTS(
                           SELECT 1 FROM attempts
                           WHERE attempts.candidate_id=candidates.candidate_id
                       ) AS has_attempt
                FROM candidates
                WHERE status='active' AND (lease_expires IS NULL OR lease_expires < ?)
                ORDER BY candidate_id
                """,
                (now,),
            ).fetchall()
            requeued = tuple(
                str(row["candidate_id"])
                for row in rows
                if row["provider_boundary_state"] == "pre_provider"
                and int(row["has_attempt"]) == 0
            )
            quarantined = tuple(
                str(row["candidate_id"])
                for row in rows
                if row["provider_boundary_state"] != "pre_provider"
                or int(row["has_attempt"]) != 0
            )
            for candidate_id in requeued:
                connection.execute(
                    """
                    UPDATE candidates SET status='queued',lease_owner=NULL,
                        lease_expires=NULL,provider_boundary_state=NULL,updated_at=?
                    WHERE candidate_id=? AND status='active'
                      AND provider_boundary_state='pre_provider'
                    """,
                    (now, candidate_id),
                )
            for candidate_id in quarantined:
                connection.execute(
                    """
                    UPDATE candidates SET status='infrastructure_quarantine',
                        lease_owner=NULL,lease_expires=NULL,updated_at=?
                    WHERE candidate_id=? AND status='active'
                    """,
                    (now, candidate_id),
                )
            return requeued, quarantined

    def quarantine_expired(self) -> tuple[str, ...]:
        """Compatibility wrapper returning only ambiguous quarantined leases."""

        _requeued, quarantined = self.recover_expired()
        return quarantined

    def record_attempt(
        self,
        *,
        attempt_id: str,
        candidate_id: str,
        phase: str,
        outcome: str,
        receipt_blake3: str,
    ) -> None:
        _uuid(attempt_id, "attempt_id")
        _uuid(candidate_id, "candidate_id")
        if not phase or not outcome or not is_blake3(receipt_blake3):
            raise CampaignLedgerError("attempt fields differ")
        with self._transaction() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO attempts(attempt_id,candidate_id,phase,outcome,receipt_blake3,created_at)
                    VALUES(?,?,?,?,?,?)
                    """,
                    (attempt_id, candidate_id, phase, outcome, receipt_blake3, time.time()),
                )
            except sqlite3.IntegrityError as exc:
                raise CampaignLedgerError("attempt identity or receipt is already consumed") from exc

    def finish_candidate(self, candidate_id: str, *, status: str) -> None:
        _uuid(candidate_id, "candidate_id")
        allowed = {"reviewed", "rejected", "infrastructure_quarantine"}
        if status not in allowed:
            raise CampaignLedgerError("terminal candidate status differs")
        with self._transaction() as connection:
            if status == "infrastructure_quarantine":
                updated = connection.execute(
                    """
                    UPDATE candidates
                    SET status=?,lease_owner=NULL,lease_expires=NULL,updated_at=?
                    WHERE candidate_id=? AND status IN ('active','reviewed')
                    """,
                    (status, time.time(), candidate_id),
                ).rowcount
            else:
                updated = connection.execute(
                    """
                    UPDATE candidates
                    SET status=?,lease_owner=NULL,lease_expires=NULL,updated_at=?
                    WHERE candidate_id=? AND status='active'
                      AND provider_boundary_state='crossed'
                    """,
                    (status, time.time(), candidate_id),
                ).rowcount
            if updated != 1:
                raise CampaignLedgerError("candidate cannot make the requested terminal transition")

    def admit(
        self,
        *,
        candidate_id: str,
        sandbox_id: str,
        manifest_blake3: str,
        admission_receipt_blake3: str,
        supervisor_transition_blake3: str,
        signatures_verified: bool,
    ) -> None:
        _uuid(candidate_id, "candidate_id")
        _uuid(sandbox_id, "sandbox_id")
        if signatures_verified is not True:
            raise CampaignLedgerError("admission signatures were not verified")
        if not all(
            is_blake3(value)
            for value in (
                manifest_blake3,
                admission_receipt_blake3,
                supervisor_transition_blake3,
            )
        ):
            raise CampaignLedgerError("admission content commitment differs")
        with self._transaction() as connection:
            candidate = connection.execute(
                "SELECT domain,stage,split,status FROM candidates WHERE candidate_id=?",
                (candidate_id,),
            ).fetchone()
            if candidate is None or candidate["status"] != "reviewed":
                raise CampaignLedgerError("only a reviewed candidate may be admitted")
            quota = connection.execute(
                "SELECT target,admitted FROM quotas WHERE domain=? AND stage=? AND split=?",
                (candidate["domain"], candidate["stage"], candidate["split"]),
            ).fetchone()
            if quota is None or quota["admitted"] >= quota["target"]:
                raise CampaignLedgerError("campaign quota is already full")
            try:
                connection.execute(
                    """
                    INSERT INTO admissions(
                        sandbox_id,candidate_id,manifest_blake3,admission_receipt_blake3,
                        supervisor_transition_blake3,created_at
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        sandbox_id,
                        candidate_id,
                        manifest_blake3,
                        admission_receipt_blake3,
                        supervisor_transition_blake3,
                        time.time(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise CampaignLedgerError("sandbox or candidate was already admitted") from exc
            connection.execute(
                "UPDATE quotas SET admitted=admitted+1 WHERE domain=? AND stage=? AND split=?",
                (candidate["domain"], candidate["stage"], candidate["split"]),
            )
            connection.execute(
                "UPDATE candidates SET status='admitted',updated_at=? WHERE candidate_id=?",
                (time.time(), candidate_id),
            )

    def progress(self) -> CampaignProgress:
        with self._connect() as connection:
            split_rows = connection.execute(
                "SELECT split,SUM(admitted) AS count FROM quotas GROUP BY split"
            ).fetchall()
            splits = {str(row["split"]): int(row["count"] or 0) for row in split_rows}
            status_rows = connection.execute(
                "SELECT status,COUNT(*) AS count FROM candidates GROUP BY status"
            ).fetchall()
            statuses = {str(row["status"]): int(row["count"]) for row in status_rows}
            scheduled = sum(statuses.values())
        admitted = sum(splits.values())
        return CampaignProgress(
            target=self.plan.total,
            scheduled=scheduled,
            admitted=admitted,
            train=splits.get("train", 0),
            development=splits.get("development", 0),
            sealed_evaluation=splits.get("sealed_evaluation", 0),
            queued=statuses.get("queued", 0),
            active=statuses.get("active", 0),
            rejected=statuses.get("rejected", 0),
            infrastructure_quarantine=statuses.get("infrastructure_quarantine", 0),
            reserve_unused=statuses.get("reserve_unused", 0),
        )


__all__ = [
    "CampaignLedger",
    "CampaignLedgerError",
    "CampaignProgress",
    "CandidateQueueRecord",
    "FROZEN_SCHEDULE_SIZE",
]
