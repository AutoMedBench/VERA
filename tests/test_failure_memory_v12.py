"""CPU acceptance logic only; synthetic receipts are not model results."""
import json
from types import SimpleNamespace

import pytest

from training.context_memory import failure_probe_v12 as probe


def test_actual_parser_failure_and_note_absence(tmp_path):
    fixture = probe.FailureFixture(tmp_path)
    assert not (fixture.workspace.root / "notes/failures.md").exists()
    failure = fixture.execute_group([("analyze_table", {"name": "pilot.csv", "delimiter": ","})])[0]
    assert failure["status"] == "failed" and failure["result"]["error_type"] == "ValueError"
    good = fixture.execute_group([("analyze_table", {"name": "pilot.csv", "delimiter": ";"})])[0]
    assert good["result"]["total"] == fixture.expected["pilot.csv"]
    assert good["status"] == "completed"
    assert len(list((tmp_path / "events").glob("*.json"))) == 2
    missing = fixture.execute_group([("read_public_file", {"path": "notes/failures.md"})])[0]
    assert missing["status"] == "failed"


def test_correct_note_schema_and_public_only_paths(tmp_path):
    from jsonschema import ValidationError
    fixture = probe.FailureFixture(tmp_path)
    with pytest.raises(ValidationError):
        fixture.execute_group([("write_note", {"path": "notes/failures.md", "content": "Wrong schema"})])
    with pytest.raises(ValidationError):
        fixture.execute_group([("read_public_file", {"path": "/private/answer.json"})])
    assert fixture.calls == []


def completed_fixture(tmp_path, monkeypatch):
    fixture = probe.FailureFixture(tmp_path)
    failure = fixture.execute_group([("analyze_table", {"name": "pilot.csv", "delimiter": ","})])[0]
    fixture.execute_group([("write_note", {"name": "failures.md", "content":
        f"Synthetic test note links {failure['event_id']}; inspect delimiter before parsing."})])
    receipts = []
    for phase, name in enumerate(probe.NAMES, 1):
        fixture.phase = phase
        if phase > 1:
            fixture.execute_group([("read_public_file", {"path": "notes/failures.md"})])
        result = fixture.execute_group([("analyze_table", {"name": name, "delimiter": ";"})])[0]["result"]
        calls = [SimpleNamespace(mcp_server="memory_failure", mcp_tool=row["name"], arguments=row["arguments"],
            output={"result": {"structuredContent": row["response"]}}) for row in fixture.calls if row["phase"] == phase]
        receipts.append(SimpleNamespace(status="completed", thread_id="old" if phase < 3 else "new",
            thread_resumed=phase == 2, selected_skill_ids=("summary_failures",), tool_calls=calls,
            final_response=json.dumps({key: result[key] for key in ("input", "total", "artifact")})))
    monkeypatch.setattr(probe, "verify_codex_turn_receipt", lambda _: None)
    return fixture, receipts


def test_behavior_requires_actual_later_read_and_no_repeat(tmp_path, monkeypatch):
    fixture, receipts = completed_fixture(tmp_path, monkeypatch)
    assert probe.verify_behavior(fixture, receipts, [10, 11], True)["passed"]
    later_read = next(row for row in fixture.calls if row["phase"] == 3 and row["name"] == "read_public_file")
    fixture.calls.remove(later_read)
    receipts[2].tool_calls = receipts[2].tool_calls[1:]
    result = probe.verify_behavior(fixture, receipts, [10, 11], True)
    assert not result["passed"] and not result["checks"]["stage_3_reads_failure_note_before_analysis"]


def test_enabled_skill_and_resume_flags_are_not_success(tmp_path, monkeypatch):
    fixture, receipts = completed_fixture(tmp_path, monkeypatch)
    receipts[2].final_response = ""
    receipts[1].status = "interrupted"
    result = probe.verify_behavior(fixture, receipts, [10, 11], True)
    assert not result["passed"]
    assert not result["checks"]["three_completed_stages"]
    assert not result["checks"]["stage_3_correct_final_answer"]


def test_receipt_host_pairing_cannot_be_fabricated(tmp_path, monkeypatch):
    fixture, receipts = completed_fixture(tmp_path, monkeypatch)
    receipts[2].tool_calls[0].output = {"result": {"structuredContent": {"event_id": "not-an-actual-event"}}}
    with pytest.raises(ValueError, match="actual_host_receipt_event_binding_differs"):
        probe.verify_behavior(fixture, receipts, [10, 11], True)
