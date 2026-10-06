"""Synthetic feedback only; no evaluation, provider, workspace, or GPU execution."""
from dataclasses import replace

import pytest

from eva_agent.rubrics import compile_registry
from eva_agent.training.coevolution import (
    EvaluationStatus, FeedbackError, RoundIdentity, VerifiedStageEvaluation, plan_stage_target,
)
from test_rubric_compiler import rubric, uid


@pytest.fixture
def tables():
    raw = []
    for number, (domain, stage) in enumerate((
        ("medical-research", "S1"), ("visual-vibe-coding", "S1"),
        ("medical-research", "S2"), ("medical-research", "S3"),
        ("medical-research", "E2E"),
    ), 1):
        table = rubric(number, domain=domain, stage=stage)
        for item in table["items"]:
            item["weight"] = 1
            item.pop("hard_gate", None)
        raw.append(table)
    return compile_registry({"schema": "eva.rubric-registry.v1", "registry_id": uid(1),
                             "registry_version": 1, "rubrics": raw})


IDENTITY = RoundIdentity("fixture-model", "fixture-checkpoint", "fixture-skill-catalog", "fixture-native-astra-judge-config")


def feedback(tables, case, stage="S1", domain="medical-research", positives=5,
             status=EvaluationStatus.SCORED):
    table = tables.resolve(domain, stage)
    score = table.score({item["item_id"]: 10000 if index < positives else 0
                         for index, item in enumerate(table.items)}) if status is EvaluationStatus.SCORED else None
    return VerifiedStageEvaluation(case, IDENTITY, table, status, "fixture-verified:" + case, score)


def test_domain_macro_not_sample_micro_and_item_deficits(tables):
    rows = [feedback(tables, f"many-{i}", positives=0) for i in range(9)]
    rows += [feedback(tables, "other-domain", domain="visual-vibe-coding"),
             feedback(tables, "stage2", stage="S2", positives=2)]
    plan = plan_stage_target(rows)
    assert plan["s_target"] == "S2"  # S1 macro=5000, not sample-micro=1000.
    assert plan["stages"]["S1"]["domain_macro_mean_reward_bps"] == 5000
    assert plan["stages"]["S1"]["sample_count"] == 10
    first = plan["stages"]["S1"]["domains"]["medical-research"]["item_deficits"][0]
    assert first["sample_count"] == 9 and first["mean_deficit_bps"] == 10000
    assert plan["attribution"] == "not_performed" and plan["skill_change_proposals"] == []
    assert plan["rl_materialization_performance_threshold"] is None
    assert not plan["execution_authorized"]


def test_failures_are_missing_e2e_is_separate_and_real_zero_is_measured(tables):
    plan = plan_stage_target([
        feedback(tables, "scored-zero", positives=0), feedback(tables, "e2e-zero", "E2E", positives=0),
        feedback(tables, "provider-error", "S3", status=EvaluationStatus.INFRASTRUCTURE_FAILURE),
    ])
    assert plan["s_target"] == "S1" and plan["e2e"]["sample_count"] == 1
    assert plan["stages"]["S3"]["status"] == "unknown"
    assert plan["stages"]["S3"]["domain_macro_mean_reward_bps"] is None
    assert plan["stages"]["S3"]["sample_count"] == 0 and plan["stages"]["S3"]["missing_count"] == 1
    assert plan["stages"]["S5"]["status"] == "unknown"
    assert plan_stage_target([feedback(tables, "e2e-only", "E2E")])["s_target"] is None
    assert plan_stage_target([])["status"] == "insufficient_stage_evidence"


def test_ties_use_stage_order_independent_of_input_order(tables):
    rows = [feedback(tables, "s2", "S2"), feedback(tables, "s1")]
    assert plan_stage_target(rows) == plan_stage_target(reversed(rows))
    assert plan_stage_target(rows)["s_target"] == "S1"


@pytest.mark.parametrize("field", ["model_id", "checkpoint_id", "skill_catalog_id", "judge_id"])
def test_mixed_round_identity_rejected_even_on_failed_rows(tables, field):
    row = feedback(tables, "failed", status=EvaluationStatus.INFRASTRUCTURE_FAILURE)
    row = replace(row, identity=replace(IDENTITY, **{field: "different"}))
    with pytest.raises(FeedbackError, match="cannot mix"):
        plan_stage_target([feedback(tables, "first"), row])


@pytest.mark.parametrize("field,value", [("reward_bps", 123), ("weighted_score_bps", 123),
    ("hard_gate_passed", False), ("rubric_digest", "wrong"), ("score_digest", "wrong")])
def test_numeric_or_commitment_tampering_rejected(tables, field, value):
    row = feedback(tables, "forged")
    with pytest.raises(FeedbackError, match="recomputation"):
        plan_stage_target([replace(row, score=replace(row.score, **{field: value}))])


def test_missing_score_and_duplicate_evidence_rejected(tables):
    row = feedback(tables, "case")
    with pytest.raises(FeedbackError, match="exact RubricScore"):
        plan_stage_target([replace(row, score=None)])
    with pytest.raises(FeedbackError, match="cannot carry"):
        plan_stage_target([replace(row, status=EvaluationStatus.INFRASTRUCTURE_FAILURE)])
    with pytest.raises(FeedbackError, match="duplicate case"):
        plan_stage_target([row, row])
    with pytest.raises(FeedbackError, match="duplicate scored evaluation"):
        plan_stage_target([row, replace(row, case_id="another")])
