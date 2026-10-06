from __future__ import annotations

from collections import defaultdict
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest

from eva_agent.campaign import (
    CandidateQueueRecord,
    FROZEN_SCHEDULE_SIZE,
    exact_6000_plan,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/bootstrap_campaign_queue.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_campaign_queue", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
bootstrap_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bootstrap_cli
SPEC.loader.exec_module(bootstrap_cli)


DIGEST = "a" * 64
SELECTION_DIGEST = "b" * 64


@pytest.fixture(scope="module")
def exact_records() -> tuple[CandidateQueueRecord, ...]:
    plan = exact_6000_plan()
    records: list[CandidateQueueRecord] = []
    ranks: dict[tuple[str, str], int] = defaultdict(int)

    def append(cell, tier: str) -> None:
        ordinal = len(records) + 1
        key = (cell.domain, cell.stage)
        ranks[key] += 1
        records.append(
            CandidateQueueRecord(
                candidate_id=str(uuid5(NAMESPACE_URL, f"bootstrap-candidate:{ordinal}")),
                source_episode_id=str(uuid5(NAMESPACE_URL, f"bootstrap-source:{ordinal}")),
                domain=cell.domain,
                stage=cell.stage,
                split="train",
                rubric_blake3=DIGEST,
                queue_ordinal=ordinal,
                cell_rank=ranks[key],
                selection_tier=tier,
            )
        )

    # Preserve the production invariant that all primaries precede reserves.
    for cell in plan.cells:
        for _ in range(cell.train):
            append(cell, "primary")
    for index in range(FROZEN_SCHEDULE_SIZE - plan.total):
        append(plan.cells[index % len(plan.cells)], "reserve")
    return tuple(records)


def _prepared(records: tuple[CandidateQueueRecord, ...]):
    return bootstrap_cli.VerifiedQueue(
        selection_id=str(uuid5(NAMESPACE_URL, "bootstrap-selection")),
        selection_blake3=SELECTION_DIGEST,
        plan_blake3="c" * 64,
        rubric_registry_blake3="d" * 64,
        source_registry_blake3="e" * 64,
        records=records,
        records_blake3=bootstrap_cli._validate_records(records),
    )


def test_queue_only_targets_are_exactly_three_provider_free_placeholders() -> None:
    assert {target.cohort.value for target in bootstrap_cli.QUEUE_ONLY_TARGETS} == {
        "weak",
        "middle",
        "strong",
    }
    assert {target.provider for target in bootstrap_cli.QUEUE_ONLY_TARGETS} == {
        "provider-free"
    }


def test_dry_run_materializes_without_inspecting_or_creating_a_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exact_records: tuple[CandidateQueueRecord, ...],
) -> None:
    ledger_path = tmp_path / "must-not-exist.sqlite3"
    prepared = _prepared(exact_records)
    monkeypatch.setattr(bootstrap_cli, "prepare_verified_queue", lambda _: prepared)
    args = SimpleNamespace(command="dry-run", ledger=ledger_path)

    result = bootstrap_cli.run(args)

    assert result["status"] == "verified"
    assert result["scheduled_count"] == 9_000
    assert result["primary_count"] == 6_000
    assert result["reserve_count"] == 3_000
    assert result["provider_calls"] == 0
    assert result["ledger"] is None
    assert not ledger_path.exists()


def test_bootstrap_is_atomic_and_idempotent(
    tmp_path: Path,
    exact_records: tuple[CandidateQueueRecord, ...],
) -> None:
    ledger_path = tmp_path / "campaign.sqlite3"
    prepared = _prepared(exact_records)

    assert (
        bootstrap_cli.bootstrap_verified_queue(ledger_path, prepared)
        == FROZEN_SCHEDULE_SIZE
    )
    before = ledger_path.read_bytes()
    assert bootstrap_cli.bootstrap_verified_queue(ledger_path, prepared) == 0
    assert ledger_path.read_bytes() == before


def test_ledger_target_rejects_final_and_parent_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    real_ledger = real / "campaign.sqlite3"
    real_ledger.touch()
    final_link = tmp_path / "ledger-link.sqlite3"
    final_link.symlink_to(real_ledger)
    with pytest.raises(
        bootstrap_cli.CampaignQueueBootstrapError, match="non-symlink"
    ):
        bootstrap_cli.validate_ledger_target(final_link)

    parent_link = tmp_path / "linked-parent"
    parent_link.symlink_to(real, target_is_directory=True)
    with pytest.raises(
        bootstrap_cli.CampaignQueueBootstrapError, match="real directory"
    ):
        bootstrap_cli.validate_ledger_target(parent_link / "new.sqlite3")


def test_ledger_parent_must_preexist_and_queue_tamper_fails_closed(
    tmp_path: Path,
    exact_records: tuple[CandidateQueueRecord, ...],
) -> None:
    with pytest.raises(
        bootstrap_cli.CampaignQueueBootstrapError, match="parent must already exist"
    ):
        bootstrap_cli.validate_ledger_target(tmp_path / "missing" / "campaign.sqlite3")

    with pytest.raises(
        bootstrap_cli.CampaignQueueBootstrapError, match="exactly 9,000"
    ):
        bootstrap_cli._validate_records(exact_records[:-1])
