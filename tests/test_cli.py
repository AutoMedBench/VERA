from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import stat
from types import SimpleNamespace
from uuid import uuid4

import eva_agent.pipeline as pipeline_api
from eva_agent.campaign import CampaignLedger
from eva_agent.cli import main, read_progress, render_progress
from eva_agent.pipeline import Stage, VerificationReport
from eva_agent.pipeline.digests import is_blake3


ROOT = Path(__file__).resolve().parents[1]


def test_progress_is_honest_zero_when_verified_ledger_is_absent(tmp_path: Path) -> None:
    snapshot = read_progress(tmp_path / "absent.sqlite3")

    assert snapshot.total == 0
    assert snapshot.scheduled == 0
    assert snapshot.reserve_unused == 0
    assert snapshot.ledger_exists is False
    rendered = render_progress(snapshot)
    assert "0/6,000 signed+verified" in rendered
    assert "0/9,000 frozen candidates scheduled" in rendered
    assert "train 0/6,000" in rendered
    assert "development disabled (0 quota)" in rendered


def test_progress_reads_exact_signed_admissions_from_campaign_ledger(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    ledger = CampaignLedger(path)
    candidate_id = str(uuid4())
    ledger.enqueue_candidate(
        candidate_id=candidate_id,
        source_episode_id="cli-fixture-episode",
        domain="medxpertqa",
        stage="E2E",
        split="train",
        rubric_blake3="a" * 64,
    )
    assert ledger.claim(worker_id="cli-test", limit=1, lease_seconds=60) == (
        candidate_id,
    )
    ledger.mark_provider_boundary(worker_id="cli-test", candidate_id=candidate_id)
    ledger.finish_candidate(candidate_id, status="reviewed")
    ledger.admit(
        candidate_id=candidate_id,
        sandbox_id=str(uuid4()),
        manifest_blake3="b" * 64,
        admission_receipt_blake3="c" * 64,
        supervisor_transition_blake3="d" * 64,
        signatures_verified=True,
    )

    snapshot = read_progress(path)

    assert snapshot.total == 1
    assert snapshot.scheduled == 1
    assert snapshot.counts == {"train": 1, "development": 0, "sealed_evaluation": 0}
    assert snapshot.queued == snapshot.active == 0
    assert snapshot.reserve_unused == 0


def test_progress_reports_scheduled_reserve_state_without_advancing_admission(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.sqlite3"
    ledger = CampaignLedger(path)
    candidate_id = str(uuid4())
    ledger.enqueue_candidate(
        candidate_id=candidate_id,
        source_episode_id="unused-reserve-fixture",
        domain="medxpertqa",
        stage="E2E",
        split="train",
        rubric_blake3="a" * 64,
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE candidates SET status='reserve_unused' WHERE candidate_id=?",
            (candidate_id,),
        )

    snapshot = read_progress(path)
    rendered = render_progress(snapshot)

    assert snapshot.total == 0
    assert snapshot.scheduled == 1
    assert snapshot.reserve_unused == 1
    assert "0/6,000 signed+verified" in rendered
    assert "1/9,000 frozen candidates scheduled" in rendered
    assert "reserve_unused=1" in rendered


def test_no_provider_demo_exercises_parallel_tool_group(capsys) -> None:
    assert main(["demo-local"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["status"] == "passed"
    assert result["provider_calls"] == 0
    assert result["tool_calls"] == 2
    assert result["max_parallelism_observed"] == 2
    assert is_blake3(result["trajectory_blake3"])


def test_progress_json_command(capsys, tmp_path: Path) -> None:
    assert main(["progress", "--ledger", str(tmp_path / "missing"), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["admitted_verified"] == 0
    assert result["target"] == 6000
    assert result["scheduled"] == 0
    assert result["schedule_target"] == 9000
    assert result["reserve_unused"] == 0


def test_cli_validates_and_seals_the_canonical_rubric_registry(
    capsys, tmp_path: Path
) -> None:
    source = ROOT / "rubrics/source/domain-stage-tables.v1.json"
    output = tmp_path / "compiled.json"

    assert main(["rubric", "validate", str(source)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["status"] == "passed"
    assert validated["rubric_tables"] == 42

    assert main(["rubric", "compile", str(source), "--output", str(output)]) == 0
    compiled = json.loads(capsys.readouterr().out)
    assert compiled["status"] == "compiled"
    assert compiled["registry_blake3"] == validated["registry_blake3"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    document = json.loads(output.read_bytes())
    assert document["registry_digest"] == validated["registry_blake3"]


def test_verify_result_resolves_the_manifest_exact_rubric(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    result_path = tmp_path / "pipeline-result.json"
    result_path.write_text("fixture", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    result = SimpleNamespace(
        sandbox_manifests=(SimpleNamespace(domain="medxpertqa", stage=Stage.E2E),)
    )
    expected = VerificationReport(
        valid=True,
        checks=("one_exact_rubric_bound_sandbox",),
        errors=(),
        result_blake3="a" * 64,
        report_blake3="b" * 64,
    )
    monkeypatch.setattr(pipeline_api, "load_pipeline_result", lambda _: result)
    monkeypatch.setattr(
        pipeline_api,
        "verify_result_document",
        lambda path, *, artifact_store, rubric: expected,
    )

    status = main(
        [
            "verify-result",
            str(result_path),
            "--artifact-root",
            str(artifact_root),
            "--registry",
            str(ROOT / "rubrics/source/domain-stage-tables.v1.json"),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert status == 0
    assert output["valid"] is True
    assert output["checks"] == ["one_exact_rubric_bound_sandbox"]
