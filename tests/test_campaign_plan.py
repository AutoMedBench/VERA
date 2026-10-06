from pathlib import Path

from eva_agent.campaign import (
    DOMAINS,
    SOURCE_FAMILY_BY_DOMAIN,
    STAGES,
    TARGET_BY_DOMAIN,
    TARGET_BY_SOURCE_FAMILY,
    exact_6000_plan,
    verify_plan,
)
from eva_agent.rubrics import load_and_compile_registry


ROOT = Path(__file__).resolve().parents[1]


def test_exact_6000_plan_is_balanced_and_verifiable() -> None:
    plan = exact_6000_plan()
    verify_plan(plan)
    assert plan.total == 6_000
    assert plan.target_by_split == {
        "train": 6_000,
        "development": 0,
        "sealed_evaluation": 0,
    }
    assert len(plan.cells) == 42
    assert all(cell.total == cell.train and cell.train > 0 for cell in plan.cells)
    assert all(cell.development == cell.sealed_evaluation == 0 for cell in plan.cells)
    for domain in DOMAINS:
        cells = [cell for cell in plan.cells if cell.domain == domain]
        assert sum(cell.train for cell in cells) == TARGET_BY_DOMAIN[domain]
    for stage in STAGES:
        assert sum(cell.total for cell in plan.cells if cell.stage == stage) == 1_000
    assert {
        family: sum(
            cell.train
            for cell in plan.cells
            if SOURCE_FAMILY_BY_DOMAIN[cell.domain] == family
        )
        for family in TARGET_BY_SOURCE_FAMILY
    } == TARGET_BY_SOURCE_FAMILY


def test_every_campaign_cell_has_one_exact_compiled_rubric() -> None:
    """Production quotas and reward/judge bindings cannot drift independently."""

    plan = exact_6000_plan()
    registry = load_and_compile_registry(
        ROOT / "rubrics/source/domain-stage-tables.v1.json"
    )
    plan_keys = {(cell.domain, cell.stage) for cell in plan.cells}
    rubric_keys = {(rubric.domain, rubric.stage) for rubric in registry.rubrics}
    assert rubric_keys == plan_keys
    assert all(5 <= len(registry.resolve(*key).items) <= 10 for key in plan_keys)
