"""Opt-in pipeline plumbing around explicit no-provider fixtures."""
from pathlib import Path
import pytest

from test_eva_rsi_production_eval import integration
from training.automedbench_lite import evaluation_report
from training.eva_rsi import production, production_eval, task_score_audit
from training.eva_rsi.controller import write
from training.eva_rsi.evidence import read


def test_full_round_audit_then_committed_report(integration, monkeypatch):
    case = integration
    case["settings"].update(evaluation_mode="full_single_pass", task_agent_score_audit=True,
        seven_track_detailed_report=True, judge_material_view="historical-v1", evaluation_actor_profile="legacy")
    monkeypatch.setattr(production_eval, "native_scoring_options", lambda settings: {})
    def native(run, identity, root):
        write(root / "summary.json", {"fixture_only": True})
        case["events"].append("native_task_fixture")
        return root / "summary.json"
    monkeypatch.setattr(production_eval, "score_native_round", native)
    monkeypatch.setattr(production_eval, "evaluation_skill_index", lambda *args: {})
    monkeypatch.setattr(production, "require_complete_baseline", lambda *args: {})
    feedback = Path(case["context"]["attempt_root"]) / "fixture-feedback"
    monkeypatch.setattr(production_eval, "judge_full_round", lambda *a, **k: (feedback,))
    monkeypatch.setattr(production_eval, "verify_evaluation", lambda *args: {"verified_fixture_only": True})
    def audit(roots, native_summary, output, **options):
        assert roots == (feedback,) and native_summary.name == "summary.json"
        assert options["native_turn_timeout_seconds"] == 600
        case["events"].append("task_agent_fixture")
        assert not (output.parent / "evaluation-index.json").exists()
        return []
    monkeypatch.setattr(task_score_audit, "audit_native_scores", audit)
    def report(run, index):
        document = read(index)
        assert document["task_agent_audits"] == [] and document["native_task_scores"]
        case["events"].append("report_fixture")
        return {"schema": "fixture-only", "scores_are_real": False}
    monkeypatch.setattr(evaluation_report, "build_report", report)
    monkeypatch.setattr(evaluation_report, "render_markdown", lambda value: "Fixture only; not real scores.")
    result = production_eval.evaluate_round(case["context"], case["settings"])
    assert case["events"][-3:] == ["native_task_fixture", "task_agent_fixture", "report_fixture"]
    path = Path(result["seven_track_report"]["path"])
    assert read(path)["scores_are_real"] is False
    assert path.with_suffix(".md").read_text().startswith("Fixture only")


def test_detailed_scores_wrong_scope_fail_before_serving(integration):
    integration["settings"]["seven_track_detailed_report"] = True
    with pytest.raises(ValueError, match="detailed_scores_require_full"):
        production_eval.evaluate_round(integration["context"], integration["settings"])
    assert integration["events"] == []
