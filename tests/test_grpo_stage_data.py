from copy import deepcopy
import pytest
from eva_agent.training.grpo_stage_data import select_stage_rows


def row(identity, stage="S1", domain="classification", split="train"):
    return {"sandbox_id": identity, "stage": stage, "domain": domain, "split": split,
            "reward_contract": {"rubric_table": {"stage": stage, "domain": domain}}}


def test_select_stage_and_balance_domains_without_modifying_source():
    rows = [row("c"), row("a"), row("b", domain="detection"), row("d", "S2"), row("e", split="test")]
    before = deepcopy(rows)
    selected = select_stage_rows(rows, stage="S1", limit=2)
    assert [r["sandbox_id"] for r in selected] == ["a", "b"]
    assert rows == before


def test_unsupported_or_empty_target_not_silently_replaced():
    with pytest.raises(ValueError):
        select_stage_rows([row("a")], stage="S4", limit=1)
    with pytest.raises(ValueError):
        select_stage_rows([row("a")], stage="S2", limit=1)


def test_duplicates_and_mismatched_rubrics_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        select_stage_rows([row("a"), row("a")], stage="S1", limit=1)
    bad = row("a")
    bad["reward_contract"]["rubric_table"]["stage"] = "S3"
    with pytest.raises(ValueError, match="rubric"):
        select_stage_rows([bad], stage="S1", limit=1)
