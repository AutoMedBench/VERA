from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

from eva_agent.campaign.selection_v3 import build_campaign_selection_v3
from eva_agent.campaign.split_plan_v3 import exact_5000_500_500_plan


def _base():
    plan = exact_5000_500_500_plan()
    reserve_by_cell = {i: cell.total // 2 for i, cell in enumerate(plan.cells)}
    for index in range(3000 - sum(reserve_by_cell.values())):
        reserve_by_cell[index] += 1
    rows = []
    ordinal = 0
    for index, cell in enumerate(plan.cells):
        count = cell.total + reserve_by_cell[index]
        for rank in range(1, count + 1):
            ordinal += 1
            source = f"source-{index:02d}-{rank:04d}"
            rows.append(SimpleNamespace(
                candidate_id=str(uuid5(NAMESPACE_URL, source)), source_candidate_id=source,
                source_family="automedbench" if cell.domain.startswith("automedbench-") else cell.domain,
                source_artifact_sha256="a" * 64,
                domain=cell.domain, stage=cell.stage, split="train",
                rubric_id=f"rubric-{index}", rubric_blake3="1" * 64,
                source_identity_rank_blake3=f"{ordinal:064x}",
                construction_readiness="frozen", readiness_proof_root_blake3="2" * 64,
                selection_rank_blake3=f"{9001 - ordinal:064x}",
                cell_rank=rank, cell_target=cell.total,
                selection_tier="primary" if rank <= cell.total else "reserve",
                queue_ordinal=ordinal,
            ))
    return SimpleNamespace(
        entries=tuple(rows), primary_entries=tuple(row for row in rows if row.selection_tier == "primary"),
        selection_blake3="3" * 64, source_registry_blake3="4" * 64,
        rubric_registry_blake3="5" * 64,
    )


def test_split_plan_is_exact_balanced_and_selection_preserves_all_identities() -> None:
    plan = exact_5000_500_500_plan()
    assert plan.target_by_split == {"train": 5000, "development": 500, "sealed_evaluation": 500}
    assert Counter(cell.stage for cell in plan.cells for _ in range(cell.total)) == {
        stage: 1000 for stage in ("S1", "S2", "S3", "S4", "S5", "E2E")
    }
    base = _base()
    selection = build_campaign_selection_v3(base, plan)
    assert {row.candidate_id for row in selection.entries} == {row.candidate_id for row in base.entries}
    assert Counter(row.split for row in selection.entries if row.selection_tier == "primary") == {
        "train": 5000, "development": 500, "sealed_evaluation": 500,
    }
    assert selection == build_campaign_selection_v3(base, plan)
