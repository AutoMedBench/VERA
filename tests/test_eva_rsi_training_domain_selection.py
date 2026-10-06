"""Synthetic signed CPU fixtures only: no evaluation, actor, provider or GPU."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from training.eva_rsi import production
from eva_agent.training.grpo_stage_data import StageDataCoverageError
from test_eva_rsi_production import context, settings
from test_grpo_signed_stage_data import bulk, catalog, record

C = "automedbench-classification"
D = "automedbench-detection"
S = "automedbench-segmentation"


def fixture(tmp_path, count=2):
    rows = [record("c", domain=C), record("d", domain=D)]
    if count == 2:
        rows.append(record("s", domain=S))
    signed = catalog(tmp_path, rows, {"d": "source_only"})
    config = {**settings(), **{key: str(value) for key, value in signed.items()},
              "bulk_root": str(bulk(tmp_path, rows)), "training_sandbox_limit": 128}
    ctx = context(tmp_path / "attempt")
    ctx["previous_evaluation"]["proposal"]["stages"]["S1"]["domains"] = {
        row["domain"]: {"sample_count": 1, "mean_score": 0} for row in rows}
    return ctx, config


def test_all_observed_missing_executable_domain_fails_without_training_launch(tmp_path, monkeypatch):
    ctx, config = fixture(tmp_path)
    monkeypatch.setattr(production, "require_real_evaluation", lambda _: None)  # Fixture, not real eval evidence.
    monkeypatch.setattr(production.subprocess, "run", lambda *a, **k: pytest.fail("training launched"))
    with pytest.raises(StageDataCoverageError):
        production.execute_train(ctx, config)
    report = json.loads((tmp_path / "attempt/training-data-selection.json").read_text())
    assert report["status"] == "blocked"
    assert report["actual_distinct_sandbox_count"] == 0
    assert report["requested_domain_source_match_count"] == 3
    assert report["requested_domain_signed_executable_count"] == 2
    assert report["signed_selection"]["uncovered_requested_domains"] == [D]
    assert not (tmp_path / "attempt/data/grpo.jsonl").exists()


@pytest.mark.parametrize("count", [1, 2])
def test_explicit_subset_preserves_observed_coverage_and_actual_finite_counts(tmp_path, count):
    ctx, config = fixture(tmp_path, count)
    requested = [C, S] if count == 2 else [C]
    config["training_domains_by_stage"] = {"S1": requested}
    original = deepcopy(ctx)
    plan = production.train_plan(ctx, config)
    receipt = production.prepare_training_data(ctx, config, plan)
    assert ctx == original and receipt["sample_count"] == count
    selection = plan["domain_selection"]
    assert selection["omitted_observed_domains"] == [D]
    assert selection["observed_domain_coverage"][D]["sample_count"] == 1
    assert selection["requested_training_domains"] == requested
    assert selection["source_domains_relabeled"] is False
    report = json.loads((tmp_path / "attempt/training-data-selection.json").read_text())
    assert report["requested_distinct_limit"] == 128
    assert report["actual_distinct_sandbox_count"] == count
    assert report["requested_limit_shortfall"] == 128 - count
    assert report["requested_domain_source_match_count"] == count
    assert report["requested_domain_signed_executable_count"] == count
    assert report["batching_creates_new_distinct_sandboxes"] is False
    emitted = [json.loads(line) for line in (tmp_path / "attempt/data/grpo.jsonl").read_text().splitlines()]
    assert len({row["metadata"]["sandbox_id"] for row in emitted}) == count
    assert {row["metadata"]["domain"] for row in emitted} == set(requested)


def test_execute_train_forwards_signed_paths_and_subsets_before_mocked_launch(tmp_path, monkeypatch):
    ctx, config = fixture(tmp_path)
    config["training_domains_by_stage"] = {"S1": [C, S]}
    monkeypatch.setattr(production, "require_real_evaluation", lambda _: None)
    calls = []

    def mock_launch(argv, **kwargs):
        report = json.loads((tmp_path / "attempt/training-data-selection.json").read_text())
        assert report["signed_selection"]["catalog_signature_verified"] is True
        assert report["actual_distinct_sandbox_count"] == 2
        assert kwargs["env"]["EVA_MEDRESEARCH_DATA_ROOT"] == str(production.ROOT)
        calls.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(production.subprocess, "run", mock_launch)
    assert production.execute_train(ctx, config) == 0
    assert len(calls) == 1


def test_medical_data_root_is_forwarded_to_ray_workers():
    import runpy
    module = runpy.run_path(str(production.ROOT / 'training/slime/run_full_parameter.py'))
    assert 'EVA_MEDRESEARCH_DATA_ROOT' in module['RAY_ENVIRONMENT_KEYS']


@pytest.mark.parametrize("extra, code", [
    ({"execution_catalog_path": "/catalog"}, "catalog_and_trust"),
    ({"training_domains_by_stage": {"S2": [C]}}, "stage_missing"),
    ({"training_domains_by_stage": {"S1": ["agentclinic"]}}, "not_observed"),
    ({"training_domains_by_stage": {"S1": [C, C]}}, "invalid_explicit"),
    ({"training_domains_by_stage": {"S1": []}}, "invalid_explicit"),
])
def test_explicit_settings_never_relabel_or_fall_back(tmp_path, extra, code):
    with pytest.raises(ValueError, match=code):
        production.train_plan(context(tmp_path), {**settings(), **extra})


def test_default_plan_and_preparation_remain_historical_unfiltered(tmp_path):
    rows = [record("a"), record("b")]
    ctx = context(tmp_path / "attempt")
    config = {**settings(), "bulk_root": str(bulk(tmp_path, rows))}
    plan = production.train_plan(ctx, config)
    assert "domain_selection" not in plan
    receipt = production.prepare_training_data(ctx, config, plan)
    assert receipt["schema"] == "eva.grpo-stage-target-data.v1"
    assert receipt["sample_count"] == 2
    assert not (tmp_path / "attempt/training-data-selection.json").exists()


@pytest.mark.parametrize("limit, expected_count", [(1000, 4), (2, 2)])
def test_explicit_verified_stage_transfer_uses_original_broad_source_domains(tmp_path, limit, expected_count):
    rows = [record("c", domain=C), record("a1", domain="agentclinic"),
            record("a2", domain="agentclinic"), record("m", domain="medxpertqa"),
            record("d", domain=D), record("other-stage", domain="agentclinic", stage="S2")]
    paths = catalog(tmp_path, rows, {"d": "source_only"})
    ctx = context(tmp_path / "attempt")
    observed = ctx["previous_evaluation"]["proposal"]["stages"]["S1"]["domains"]
    observed["automedbench-report"] = {"sample_count": 1}
    before = deepcopy(ctx)
    config = {**settings(), **{key: str(value) for key, value in paths.items()},
              "bulk_root": str(bulk(tmp_path, rows)), "training_sandbox_limit": limit,
              "training_domain_scope": "verified_stage_transfer"}
    plan = production.train_plan(ctx, config)
    assert plan["stage"] == "S1" and plan["domains"] == []
    assert plan["domain_selection"]["mode"] == "verified_stage_transfer"
    assert plan["domain_selection"]["cross_domain_stage_transfer_explicitly_requested"] is True
    assert plan["domain_selection"]["observed_domain_coverage"] == observed and ctx == before
    production.prepare_training_data(ctx, config, plan)
    report = json.loads((tmp_path / "attempt/training-data-selection.json").read_text())
    assert report["actual_distinct_sandbox_count"] == expected_count
    assert report["requested_domain_source_match_count"] == 5
    assert report["requested_domain_signed_executable_count"] == 4
    assert report["eligible_signed_source_domains"] == ["agentclinic", C, "medxpertqa"]
    assert report["observed_domains_without_selected_training_sources"] == ["automedbench-report"]
    assert "agentclinic" in report["selected_training_domains_not_observed_in_eval"]
    assert report["domain_selection"]["source_domains_relabeled"] is False
    emitted = [json.loads(line) for line in (tmp_path / "attempt/data/grpo.jsonl").read_text().splitlines()]
    originals = {row["sandbox_id"]: row for row in rows}
    for item in emitted:
        metadata = item["metadata"]; original = originals[metadata["sandbox_id"]]
        assert metadata["stage"] == original["stage"] == "S1"
        assert metadata["domain"] == original["domain"]
        assert metadata["rubric_digest"] == original["reward_contract"]["rubric_table"]["rubric_digest"]
    if limit == 1000:
        assert report["selected_training_domains"] == ["agentclinic", C, "medxpertqa"]
    else:
        assert report["signed_selection"]["eligible_not_selected_due_to_limit"] == 2


@pytest.mark.parametrize("extra, code", [
    ({}, "requires_signed"),
    ({"execution_catalog_path": "/catalog", "trust_store_path": "/trust",
      "training_domains_by_stage": {"S1": [C]}}, "conflicts_with_explicit"),
])
def test_transfer_is_only_explicit_signed_scope_not_implicit_fallback(tmp_path, extra, code):
    with pytest.raises(ValueError, match=code):
        production.train_plan(context(tmp_path), {**settings(), **extra,
            "training_domain_scope": "verified_stage_transfer"})
