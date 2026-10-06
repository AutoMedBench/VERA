from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
from pathlib import Path
from types import MappingProxyType
from uuid import NAMESPACE_URL, uuid5

import pytest

from eva_agent.campaign import (
    CampaignSelection,
    CampaignSelectionV2,
    CampaignSelectionV2Error,
    SelectionEntry,
    build_campaign_selection_v2,
    exact_6000_plan,
    load_campaign_selection,
    load_campaign_selection_v2,
    verify_campaign_selection,
    verify_campaign_selection_v2,
    write_campaign_selection_v2_new,
)
from eva_agent.campaign.selection import (
    _campaign_candidate_id,
    _queue_sort_key,
    _selection_id,
    _selection_rank,
)
from eva_agent.pipeline.digests import blake3_hex
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.sources import (
    ConstructionReadiness,
    SignedReadinessRow,
    SignedSupervisorV24Readiness,
    load_signed_supervisor_v24_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT.parent / "rlevo-med-research"


@pytest.fixture(scope="module")
def rubrics():
    return load_and_compile_registry(ROOT / "rubrics/source/domain-stage-tables.v1.json")


def _scheduled_count(domain: str, target: int) -> int:
    if domain in {"agentclinic", "healthbench-professional", "medxpertqa"}:
        return 375
    if domain == "automedbench-research":
        return 250
    return target


def _base_selection(rubrics) -> CampaignSelection:
    plan = exact_6000_plan()
    source_registry_blake3 = blake3_hex({"registry": "v4-test"})
    upstream_registry = hashlib.sha256(b"v4-test").hexdigest()
    by_cell: dict[tuple[str, str], list[SelectionEntry]] = defaultdict(list)
    for cell in plan.cells:
        rubric = rubrics.resolve(cell.domain, cell.stage)
        for index in range(_scheduled_count(cell.domain, cell.train)):
            source_id = f"source-{cell.domain}-{cell.stage.lower()}-{index:04d}"
            artifact = hashlib.sha256(source_id.encode()).hexdigest()
            rank = _selection_rank(
                plan_blake3=plan.plan_blake3,
                rubric_registry_blake3=rubrics.digest,
                source_candidate_id=source_id,
                source_family={
                    "agentclinic": "agentclinic",
                    "healthbench-professional": "healthbench-professional",
                    "medxpertqa": "medxpertqa",
                }.get(cell.domain, "automedbench"),
                source_stage=cell.stage,
                source_artifact_sha256=artifact,
            )
            by_cell[(cell.domain, cell.stage)].append(
                SelectionEntry(
                    candidate_id=_campaign_candidate_id(plan.plan_blake3, source_id),
                    source_candidate_id=source_id,
                    source_family={
                        "agentclinic": "agentclinic",
                        "healthbench-professional": "healthbench-professional",
                        "medxpertqa": "medxpertqa",
                    }.get(cell.domain, "automedbench"),
                    source_artifact_sha256=artifact,
                    domain=cell.domain,
                    stage=cell.stage,
                    split="train",
                    rubric_id=rubric.rubric_id,
                    rubric_blake3=rubric.digest,
                    selection_rank_blake3=rank,
                    cell_rank=0,
                    cell_target=cell.train,
                    selection_tier="",
                    queue_ordinal=0,
                )
            )
    entries: list[SelectionEntry] = []
    for cell in plan.cells:
        rows = sorted(
            by_cell[(cell.domain, cell.stage)],
            key=lambda row: (row.selection_rank_blake3, row.source_candidate_id),
        )
        entries.extend(
            replace(
                row,
                cell_rank=rank,
                selection_tier="primary" if rank <= cell.train else "reserve",
            )
            for rank, row in enumerate(rows, start=1)
        )
    cell_order = {
        (cell.domain, cell.stage): ordinal for ordinal, cell in enumerate(plan.cells)
    }
    entries.sort(key=lambda row: _queue_sort_key(row, cell_order=cell_order))
    entries = [
        replace(row, queue_ordinal=ordinal)
        for ordinal, row in enumerate(entries, start=1)
    ]
    provisional = CampaignSelection(
        selection_id=_selection_id(
            plan_blake3=plan.plan_blake3,
            source_registry_blake3=source_registry_blake3,
            rubric_registry_blake3=rubrics.digest,
        ),
        plan_id=plan.plan_id,
        plan_blake3=plan.plan_blake3,
        source_registry_blake3=source_registry_blake3,
        upstream_source_registry_sha256=upstream_registry,
        rubric_registry_blake3=rubrics.digest,
        entries=tuple(entries),
        selection_blake3="",
    )
    result = replace(
        provisional, selection_blake3=blake3_hex(provisional.core_document())
    )
    verify_campaign_selection(result, plan=plan, rubrics=rubrics)
    return result


def _readiness(base: CampaignSelection) -> SignedSupervisorV24Readiness:
    by_cell: dict[tuple[str, str], list[SelectionEntry]] = defaultdict(list)
    for entry in base.entries:
        by_cell[(entry.domain, entry.stage)].append(entry)
    state_by_source = {row.source_candidate_id: "frozen" for row in base.entries}
    remaining_primary = 1_338
    for key in sorted(by_cell):
        rows = sorted(by_cell[key], key=lambda row: (row.selection_rank_blake3, row.source_candidate_id))
        take = min(rows[0].cell_target, remaining_primary)
        for row in rows[:take]:
            state_by_source[row.source_candidate_id] = "promoted"
        remaining_primary -= take
    assert remaining_primary == 0
    # Put six additional promoted candidates in reserve cells.  Sorting still
    # maximizes promoted primary independently within each cell.
    for key in sorted(by_cell):
        rows = sorted(by_cell[key], key=lambda row: (row.selection_rank_blake3, row.source_candidate_id))
        target = rows[0].cell_target
        if all(state_by_source[row.source_candidate_id] == "promoted" for row in rows[:target]):
            reserve = rows[target : target + 6]
            if len(reserve) == 6:
                for row in reserve:
                    state_by_source[row.source_candidate_id] = "promoted"
                break
    reserve_rows = [
        row
        for row in reversed(base.entries)
        if row.selection_tier == "reserve" and state_by_source[row.source_candidate_id] == "frozen"
    ][:11]
    assert len(reserve_rows) == 11
    for row in reserve_rows:
        state_by_source[row.source_candidate_id] = "rejected"
    rows = tuple(
        SignedReadinessRow(
            candidate_id=entry.source_candidate_id,
            source_family=entry.source_family,
            stage=entry.stage,
            readiness=ConstructionReadiness(state_by_source[entry.source_candidate_id]),
            proof_root_blake3=blake3_hex(
                {"source": entry.source_candidate_id, "state": state_by_source[entry.source_candidate_id]}
            ),
        )
        for entry in sorted(base.entries, key=lambda row: row.source_candidate_id)
    )
    digest = lambda label: blake3_hex({"test": label})
    upstream = hashlib.sha256(b"legacy").hexdigest()
    provisional = SignedSupervisorV24Readiness(
        campaign_id="evamed-6000-campaign-supervisor-v1",
        registry_revision=24,
        candidate_set_upstream_sha256=upstream,
        registry_logical_upstream_sha256=upstream,
        registry_file_upstream_sha256=upstream,
        registry_file_blake3=digest("registry"),
        registry_attestation_file_upstream_sha256=upstream,
        registry_attestation_file_blake3=digest("registry-attestation"),
        status_file_upstream_sha256=upstream,
        status_file_blake3=digest("status"),
        checkpoint_head_file_upstream_sha256=upstream,
        checkpoint_head_file_blake3=digest("checkpoint-head"),
        checkpoint_chain_blake3=digest("checkpoint-chain"),
        trust_store_file_upstream_sha256=upstream,
        trust_store_file_blake3=digest("trust-store"),
        proof_roots_blake3=blake3_hex([row.to_document() for row in rows]),
        verified_construction_proofs_blake3=digest("verified-construction-proofs"),
        rows=rows,
        authority_blake3="",
    )
    return replace(
        provisional, authority_blake3=blake3_hex(provisional.core_document())
    )


@pytest.fixture(scope="module")
def frozen_v2(rubrics):
    base = _base_selection(rubrics)
    readiness = _readiness(base)
    selection = build_campaign_selection_v2(
        base_selection=base,
        readiness=readiness,
        plan=exact_6000_plan(),
        rubrics=rubrics,
    )
    return base, readiness, selection


def test_v2_exact_readiness_priorities_and_all_train(rubrics, frozen_v2) -> None:
    base, readiness, selection = frozen_v2
    verify_campaign_selection_v2(
        selection,
        base_selection=base,
        readiness=readiness,
        plan=exact_6000_plan(),
        rubrics=rubrics,
    )
    assert len(selection.entries) == 9_000
    assert len(selection.primary_entries) == 6_000
    assert len(selection.reserve_entries) == 3_000
    assert {row.split for row in selection.entries} == {"train"}
    assert Counter(row.construction_readiness for row in selection.primary_entries) == {
        "promoted": 1_338,
        "frozen": 4_662,
    }
    assert Counter(row.construction_readiness for row in selection.reserve_entries) == {
        "promoted": 6,
        "frozen": 2_983,
        "rejected": 11,
    }
    assert not any(row.construction_readiness == "rejected" for row in selection.primary_entries)
    assert {row.candidate_id for row in selection.entries} == {
        row.candidate_id for row in base.entries
    }


def test_v2_is_deterministic_and_keeps_every_v1_source_field(rubrics, frozen_v2) -> None:
    base, readiness, selection = frozen_v2
    rebuilt = build_campaign_selection_v2(
        base_selection=base,
        readiness=readiness,
        plan=exact_6000_plan(),
        rubrics=rubrics,
    )
    assert rebuilt == selection
    base_by_id = {row.candidate_id: row for row in base.entries}
    for row in selection.entries:
        old = base_by_id[row.candidate_id]
        assert row.base_fields == {
            "candidate_id": old.candidate_id,
            "source_candidate_id": old.source_candidate_id,
            "source_family": old.source_family,
            "source_artifact_sha256": old.source_artifact_sha256,
            "domain": old.domain,
            "stage": old.stage,
            "split": "train",
            "rubric_id": old.rubric_id,
            "rubric_blake3": old.rubric_blake3,
        }


def test_v2_tamper_and_rejected_primary_fail_closed(rubrics, frozen_v2) -> None:
    base, readiness, selection = frozen_v2
    document = selection.to_document()
    document["entries"][0]["construction_readiness"] = "rejected"
    with pytest.raises(CampaignSelectionV2Error, match="content"):
        CampaignSelectionV2.from_document(document)

    primary = next(row for row in selection.entries if row.selection_tier == "primary")
    forged_rows = list(readiness.rows)
    position = next(
        index for index, row in enumerate(forged_rows) if row.candidate_id == primary.source_candidate_id
    )
    forged_rows[position] = replace(
        forged_rows[position], readiness=ConstructionReadiness.REJECTED
    )
    forged_authority = replace(readiness, rows=tuple(forged_rows))
    with pytest.raises(CampaignSelectionV2Error, match="readiness binding"):
        verify_campaign_selection_v2(
            selection,
            base_selection=base,
            readiness=forged_authority,
            plan=exact_6000_plan(),
            rubrics=rubrics,
        )


def test_v2_write_once_round_trip_and_symlink_rejection(
    tmp_path: Path, rubrics, frozen_v2
) -> None:
    base, readiness, selection = frozen_v2
    target = tmp_path / "selection-v2.json"
    kwargs = {
        "base_selection": base,
        "readiness": readiness,
        "plan": exact_6000_plan(),
        "rubrics": rubrics,
    }
    write_campaign_selection_v2_new(target, selection, **kwargs)
    assert load_campaign_selection_v2(target, **kwargs) == selection
    before = target.read_bytes()
    with pytest.raises(CampaignSelectionV2Error, match="already exists"):
        write_campaign_selection_v2_new(target, selection, **kwargs)
    assert target.read_bytes() == before
    linked = tmp_path / "linked.json"
    linked.symlink_to(target.name)
    with pytest.raises(CampaignSelectionV2Error, match="opened safely"):
        load_campaign_selection_v2(linked, **kwargs)


def test_real_signed_v24_projects_expected_counts_without_outcome_files() -> None:
    root = LEGACY / "runs/evamed-campaign-supervisor-v24-attempt1"
    trust = LEGACY / "config/host-trust-store.v1.json"
    if not root.exists() or not trust.exists():
        pytest.skip("frozen external v24 authority is not installed")
    authority = load_signed_supervisor_v24_readiness(
        root,
        authority_root=LEGACY,
        trust_store_path=trust,
        worker_width=128,
    )
    assert Counter(row.readiness.value for row in authority.rows) == {
        "promoted": 1_344,
        "frozen": 7_645,
        "rejected": 11,
    }
    assert len(authority.verified_construction_proofs_blake3) == 64
    assert len(authority.authority_blake3) == 64


def test_real_v1_to_v24_v2_dry_run_matches_expected_primary() -> None:
    base_path = ROOT / "runs/campaign-selection.v1.json"
    root = LEGACY / "runs/evamed-campaign-supervisor-v24-attempt1"
    trust = LEGACY / "config/host-trust-store.v1.json"
    if not base_path.exists() or not root.exists() or not trust.exists():
        pytest.skip("frozen campaign artifacts are not installed")
    rubrics = load_and_compile_registry(ROOT / "rubrics/source/domain-stage-tables.v1.json")
    plan = exact_6000_plan()
    base = load_campaign_selection(base_path, plan=plan, rubrics=rubrics)
    authority = load_signed_supervisor_v24_readiness(
        root,
        authority_root=LEGACY,
        trust_store_path=trust,
        worker_width=128,
    )
    selection = build_campaign_selection_v2(
        base_selection=base,
        readiness=authority,
        plan=plan,
        rubrics=rubrics,
    )
    assert Counter(row.construction_readiness for row in selection.primary_entries) == {
        "promoted": 1_338,
        "frozen": 4_662,
    }
    assert Counter(row.construction_readiness for row in selection.reserve_entries) == {
        "promoted": 6,
        "frozen": 2_983,
        "rejected": 11,
    }
    assert selection.base_selection_blake3 == base.selection_blake3
