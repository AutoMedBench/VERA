from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
import threading
import time
from uuid import NAMESPACE_URL, uuid5

import pytest

from eva_agent.campaign import (
    CampaignSelection,
    CampaignSelectionError,
    CampaignLedger,
    FrozenCampaignCandidateSource,
    SOURCE_FAMILY_BY_DOMAIN,
    build_campaign_selection,
    exact_6000_plan,
    load_campaign_selection,
    verify_campaign_selection,
    verify_campaign_selection_against_registry,
    verify_selected_campaign_sources,
    write_campaign_selection_new,
)
from eva_agent.campaign.selection import (
    _campaign_candidate_id,
    _queue_sort_key,
    _selection_rank,
)
from eva_agent.pipeline.contracts import Cohort, ModelTarget, Stage
from eva_agent.pipeline.digests import blake3_hex
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.sources import ImportedLegacyEpisode, LegacyCandidateRef, RubricAvailability


ROOT = Path(__file__).resolve().parents[1]


class _FrozenImporter:
    """In-memory port exposing only frozen registry/source data."""

    def __init__(self, registry_blake3: str | None = None) -> None:
        self.registry_file_blake3 = registry_blake3 or blake3_hex({"registry": "frozen-a"})
        self.upstream_registry_sha256 = hashlib.sha256(b"frozen-registry").hexdigest()
        self._domains: dict[str, str] = {}
        references: list[LegacyCandidateRef] = []
        auto_per_stage = {
            "automedbench-classification": 40,
            "automedbench-detection": 80,
            "automedbench-segmentation": 80,
            "automedbench-research": 175,
        }
        for stage in Stage:
            for domain, count in auto_per_stage.items():
                for index in range(count):
                    references.append(self._reference(domain, stage, index))
            for domain in ("agentclinic", "healthbench-professional", "medxpertqa"):
                for index in range(375):
                    references.append(self._reference(domain, stage, index))
        self._references = tuple(sorted(references, key=lambda row: row.candidate_id))
        self.candidate_count = len(self._references)
        self._lock = threading.Lock()
        self.import_counts: Counter[str] = Counter()
        self.active_imports = 0
        self.maximum_active_imports = 0

    def _reference(self, domain: str, stage: Stage, index: int) -> LegacyCandidateRef:
        family = SOURCE_FAMILY_BY_DOMAIN[domain]
        candidate_id = f"source-{domain}-{stage.value.lower()}-{index:04d}"
        self._domains[candidate_id] = domain
        return LegacyCandidateRef(
            candidate_id=candidate_id,
            family=family,
            stage=stage,
            priority=index,
            source_shard="frozen-test",
            artifact_relative_path=f"inputs/{candidate_id}.json",
            upstream_artifact_sha256=hashlib.sha256(candidate_id.encode()).hexdigest(),
            frozen_assertions=(),
        )

    def iter_candidates(self, *, candidate_ids=None, families=None, stages=None):
        ids = None if candidate_ids is None else frozenset(candidate_ids)
        family_set = None if families is None else frozenset(families)
        stage_set = None if stages is None else frozenset(
            value if isinstance(value, Stage) else Stage(value) for value in stages
        )
        for reference in self._references:
            if ids is not None and reference.candidate_id not in ids:
                continue
            if family_set is not None and reference.family not in family_set:
                continue
            if stage_set is not None and reference.stage not in stage_set:
                continue
            yield reference

    def import_candidate(self, reference: LegacyCandidateRef) -> ImportedLegacyEpisode:
        with self._lock:
            self.active_imports += 1
            self.maximum_active_imports = max(
                self.maximum_active_imports, self.active_imports
            )
        time.sleep(0.00005)
        try:
            domain = self._domains[reference.candidate_id]
            episode_id = str(uuid5(NAMESPACE_URL, f"episode:{reference.candidate_id}"))
            import_blake3 = blake3_hex(
                {"source_candidate_id": reference.candidate_id, "domain": domain}
            )
            manifest = SimpleNamespace(
                candidate_id=reference.candidate_id,
                episode_id=episode_id,
                source_family=reference.family,
                rubric_domain=domain,
                stage=reference.stage,
                upstream_artifact_sha256=reference.upstream_artifact_sha256,
                registry_file_blake3=self.registry_file_blake3,
                upstream_registry_sha256=self.upstream_registry_sha256,
                rubric_availability=RubricAvailability.READY,
                import_blake3=import_blake3,
            )
            episode = SimpleNamespace(
                episode_id=episode_id,
                domain=domain,
                stage=reference.stage,
            )
            with self._lock:
                self.import_counts[reference.family] += 1
            return ImportedLegacyEpisode(episode=episode, manifest=manifest)
        finally:
            with self._lock:
                self.active_imports -= 1


@pytest.fixture(scope="module")
def rubrics():
    return load_and_compile_registry(ROOT / "rubrics/source/domain-stage-tables.v1.json")


@pytest.fixture(scope="module")
def frozen_selection(rubrics):
    importer = _FrozenImporter()
    selection = build_campaign_selection(
        importer=importer, rubrics=rubrics, worker_width=32
    )
    return importer, selection


def _recommit(selection: CampaignSelection, *, entries=None, **changes) -> CampaignSelection:
    provisional = replace(
        selection,
        entries=selection.entries if entries is None else tuple(entries),
        selection_blake3="",
        **changes,
    )
    return replace(provisional, selection_blake3=blake3_hex(provisional.core_document()))


def test_build_is_exact_deterministic_and_uses_no_outcomes(
    rubrics, frozen_selection
) -> None:
    importer, selection = frozen_selection
    plan = exact_6000_plan()
    verify_campaign_selection(selection, plan=plan, rubrics=rubrics)
    assert len(selection.entries) == 9_000
    assert len(selection.primary_entries) == 6_000
    assert len(selection.reserve_entries) == 3_000
    assert len({row.candidate_id for row in selection.entries}) == 9_000
    assert len({row.source_candidate_id for row in selection.entries}) == 9_000
    assert {row.split for row in selection.entries} == {"train"}
    assert Counter((row.domain, row.stage) for row in selection.primary_entries) == Counter(
        {(cell.domain, cell.stage): cell.train for cell in plan.cells}
    )
    assert [row.queue_ordinal for row in selection.entries] == list(range(1, 9_001))
    assert all(row.selection_tier == "primary" for row in selection.entries[:6_000])
    assert all(row.selection_tier == "reserve" for row in selection.entries[6_000:])
    assert [row.cell_rank for row in selection.entries[:42]] == [1] * 42
    assert importer.import_counts == {"automedbench": 2_250}
    assert importer.maximum_active_imports > 1

    # A different whole-registry digest (which could include supervisor states)
    # changes the outer commitment but cannot alter rank or membership.
    other = _FrozenImporter(blake3_hex({"registry": "same-identities-new-states"}))
    rebuilt = build_campaign_selection(importer=other, rubrics=rubrics, worker_width=32)
    assert rebuilt.entries == selection.entries
    assert rebuilt.source_registry_blake3 != selection.source_registry_blake3


def test_strict_round_trip_write_once_and_symlink_rejection(
    tmp_path: Path, rubrics, frozen_selection
) -> None:
    _, selection = frozen_selection
    plan = exact_6000_plan()
    target = tmp_path / "selection.json"
    write_campaign_selection_new(target, selection, plan=plan, rubrics=rubrics)
    before = target.read_bytes()
    assert load_campaign_selection(target, plan=plan, rubrics=rubrics) == selection
    with pytest.raises(CampaignSelectionError, match="already exists"):
        write_campaign_selection_new(target, selection, plan=plan, rubrics=rubrics)
    assert target.read_bytes() == before

    link = tmp_path / "selection-link.json"
    link.symlink_to(target.name)
    with pytest.raises(CampaignSelectionError, match="opened safely"):
        load_campaign_selection(link, plan=plan, rubrics=rubrics)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(CampaignSelectionError, match="duplicate JSON key"):
        load_campaign_selection(duplicate, plan=plan, rubrics=rubrics)


def test_parse_and_commitment_tamper_fail_closed(rubrics, frozen_selection) -> None:
    _, selection = frozen_selection
    plan = exact_6000_plan()
    document = selection.to_document()
    document["entries"][0]["split"] = "development"
    with pytest.raises(CampaignSelectionError, match="content BLAKE3"):
        CampaignSelection.from_document(document)

    document = selection.to_document()
    document["unexpected"] = True
    with pytest.raises(CampaignSelectionError, match="keys differ"):
        CampaignSelection.from_document(document)

    wrong_identity = _recommit(
        selection, selection_id=str(uuid5(NAMESPACE_URL, "wrong-selection"))
    )
    with pytest.raises(CampaignSelectionError, match="deterministic identity"):
        verify_campaign_selection(wrong_identity, plan=plan, rubrics=rubrics)


def test_recomputed_non_top_n_substitution_fails_registry_reopen(
    rubrics, frozen_selection
) -> None:
    importer, selection = frozen_selection
    plan = exact_6000_plan()
    position = next(
        index
        for index, row in enumerate(selection.entries)
        if row.domain == "medxpertqa" and row.stage == "S1"
    )
    old = selection.entries[position]
    selected_ids = {row.source_candidate_id for row in selection.entries}
    assert len(selected_ids) == 9_000
    forged_source_id = old.source_candidate_id + "-forged"
    forged_sha256 = hashlib.sha256(forged_source_id.encode()).hexdigest()
    replacement = replace(
        old,
        candidate_id=_campaign_candidate_id(plan.plan_blake3, forged_source_id),
        source_candidate_id=forged_source_id,
        source_artifact_sha256=forged_sha256,
        selection_rank_blake3=_selection_rank(
            plan_blake3=plan.plan_blake3,
            rubric_registry_blake3=rubrics.digest,
            source_candidate_id=forged_source_id,
            source_family=old.source_family,
            source_stage=old.stage,
            source_artifact_sha256=forged_sha256,
        ),
    )
    entries = list(selection.entries)
    entries[position] = replacement
    cell_targets = {(cell.domain, cell.stage): cell.train for cell in plan.cells}
    normalized = []
    for key, target in cell_targets.items():
        cell = sorted(
            (row for row in entries if (row.domain, row.stage) == key),
            key=lambda row: (row.selection_rank_blake3, row.source_candidate_id),
        )
        normalized.extend(
            replace(
                row,
                cell_rank=rank,
                cell_target=target,
                selection_tier="primary" if rank <= target else "reserve",
            )
            for rank, row in enumerate(cell, start=1)
        )
    cell_order = {key: index for index, key in enumerate(cell_targets)}
    normalized.sort(key=lambda row: _queue_sort_key(row, cell_order=cell_order))
    entries = [
        replace(row, queue_ordinal=ordinal)
        for ordinal, row in enumerate(normalized, start=1)
    ]
    forged = _recommit(selection, entries=entries)
    verify_campaign_selection(forged, plan=plan, rubrics=rubrics)
    with pytest.raises(CampaignSelectionError, match="deterministic frozen top-N"):
        verify_campaign_selection_against_registry(
            forged,
            importer=importer,
            rubrics=rubrics,
            plan=plan,
            worker_width=32,
            verify_selected_sources=False,
        )


def test_parallel_full_source_verification_and_queue_rows(
    rubrics, frozen_selection
) -> None:
    importer, selection = frozen_selection
    receipts = verify_selected_campaign_sources(
        selection,
        importer=importer,
        rubrics=rubrics,
        worker_width=64,
    )
    assert len(receipts) == 9_000
    assert all(len(value) == 64 for value in receipts.values())
    assert importer.maximum_active_imports > 1

    source = FrozenCampaignCandidateSource(
        selection=selection,
        importer=importer,
        rubrics=rubrics,
        targets=(
            ModelTarget(Cohort.WEAK, "weak", "test"),
            ModelTarget(Cohort.MIDDLE, "middle", "test"),
            ModelTarget(Cohort.STRONG, "strong", "test"),
        ),
    )
    ids = [selection.entries[0].candidate_id, selection.entries[-1].candidate_id]
    rows = source.queue_rows(ids, worker_width=2)
    assert [row.candidate_id for row in rows] == ids
    assert all(row.split == "train" for row in rows)
    assert [row.queue_ordinal for row in rows] == [1, 9_000]
    records = tuple(row.to_candidate_queue_record() for row in rows)
    assert [record.queue_ordinal for record in records] == [1, 9_000]
    assert records[0].cell_rank == 1 and records[0].selection_tier == "primary"
    assert source.scheduled_count == source.candidate_count == 9_000
    assert source.primary_count == 6_000 and source.reserve_count == 3_000
    assert len(source.candidate_ids) == 9_000
    assert len(source.primary_candidate_ids) == 6_000
    assert len(source.reserve_candidate_ids) == 3_000
    assert source.selection_blake3 == selection.selection_blake3
    assert source.load(ids[0]).candidate_id == ids[0]
    with pytest.raises(CampaignSelectionError, match="duplicated"):
        source.verify_sources([ids[0], ids[0]])


def test_full_schedule_maps_exactly_to_atomic_ledger_bootstrap(
    tmp_path: Path, rubrics, frozen_selection
) -> None:
    importer, selection = frozen_selection
    source = FrozenCampaignCandidateSource(
        selection=selection,
        importer=importer,
        rubrics=rubrics,
        targets=(
            ModelTarget(Cohort.WEAK, "weak", "test"),
            ModelTarget(Cohort.MIDDLE, "middle", "test"),
            ModelTarget(Cohort.STRONG, "strong", "test"),
        ),
    )
    records = source.queue_records(worker_width=64)
    assert len(records) == 9_000
    assert [row.queue_ordinal for row in records] == list(range(1, 9_001))
    ledger = CampaignLedger(tmp_path / "campaign.sqlite3")
    assert ledger.bootstrap_candidates(
        records, selection_blake3=selection.selection_blake3
    ) == 9_000
    assert ledger.bootstrap_candidates(
        records, selection_blake3=selection.selection_blake3
    ) == 0
    assert ledger.progress().queued == 9_000
