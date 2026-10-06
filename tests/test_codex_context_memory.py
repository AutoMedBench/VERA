import json

import pytest

from training.context_memory.probe import MemoryFixture


def test_transient_context_is_not_persistent_workspace_memory(tmp_path):
    fixture = MemoryFixture(tmp_path)
    record = fixture.execute_group([("read_record", {})])[0]
    assert record["observation_tag"] == fixture.observation_tag
    fixture.execute_group([("write_note", {"threshold_bps": 7300, "next_stage": "S3"})])
    assert fixture.observation_tag not in fixture.workspace.read_bytes("project-note.json").decode()
    fixture.phase = "context"
    assert fixture.execute_group([("read_record", {})]) == [{"available": False}]
    fixture.execute_group([("write_note", {"threshold_bps": 8100, "next_stage": "S4"})])
    fixture.phase = "memory"
    assert fixture.execute_group([("read_note", {})]) == [{"threshold_bps": 8100, "next_stage": "S4"}]
    assert len(fixture.calls) == 5


def test_note_cannot_copy_transient_observation_or_arbitrary_file_path(tmp_path):
    from jsonschema import ValidationError
    fixture = MemoryFixture(tmp_path)
    for extra in ({"observation_tag": fixture.observation_tag}, {"path": "somewhere"}):
        with pytest.raises(ValidationError):
            fixture.execute_group([("write_note", {"threshold_bps": 7300, "next_stage": "S3", **extra})])
    assert not (fixture.workspace.root / "project-note.json").exists()


def test_context_probe_tools_are_fixture_only_not_medical_tools(tmp_path):
    fixture = MemoryFixture(tmp_path)
    assert all(offer.server == "memory_fixture" for offer in fixture.offers())
    with pytest.raises(ValueError, match="unoffered"):
        fixture.execute_group([("execute_code", {"code": "anything"})])
    assert fixture.calls == []


def test_unknown_backend_is_not_a_fallback(tmp_path):
    from training.context_memory.probe import probe_backend
    with pytest.raises(ValueError, match="explicit"):
        with probe_backend(tmp_path, tmp_path / "absent-auth", "auto"):
            pytest.fail("unknown backend opened")


def test_local_probe_requires_exact_checkpoint_and_route(tmp_path):
    from training.context_memory.probe import checkpoint_binding
    with pytest.raises(ValueError, match="explicit"):
        checkpoint_binding(None)
    document = {"schema": "eva.qwen-final-serving-checkpoint-identity.v1",
                "receipt_id": "fixture", "exact_final_model_path": str(tmp_path),
                "final_checkpoint_iteration": 170,
                "settings": {"served_model_name": "Qwen/Qwen3.5-9B", "host": "127.0.0.1",
                             "port": 30910, "context_length": 32768}}
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(document))
    assert checkpoint_binding(path) == document
    document["settings"]["port"] = 30911
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="differs"):
        checkpoint_binding(path)


def test_resume_flags_alone_cannot_pass_behavioral_memory_check(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from training.context_memory import probe
    fixture = MemoryFixture(tmp_path)
    fixture.execute_group([("read_record", {}),
                           ("write_note", {"threshold_bps": 7300, "next_stage": "S3"})])
    fixture.phase = "context"
    fixture.execute_group([("write_note", {"threshold_bps": 8100, "next_stage": "S4"})])
    fixture.phase = "memory"
    fixture.execute_group([("read_note", {})])
    # This tests behavioral checking only; these are explicitly synthetic receipts.
    monkeypatch.setattr(probe, "verify_codex_turn_receipt", lambda _: None)
    receipts = [SimpleNamespace(status="completed", thread_id="old" if index < 3 else "fresh",
        thread_resumed=index == 2, tool_calls=(), final_response=json.dumps({
            "threshold_bps": 7300 if index == 0 else 8100,
            "next_stage": "S3" if index == 0 else "S4",
            "observation_tag": fixture.observation_tag if index < 3 else None})) for index in range(4)]
    assert probe.verify_probe(receipts, fixture, [10, 11], True)["passed"]
    receipts[2].final_response = json.dumps({"threshold_bps": 7300, "next_stage": "S3",
                                           "observation_tag": fixture.observation_tag})
    result = probe.verify_probe(receipts, fixture, [10, 11], True)
    assert not result["passed"] and not result["checks"]["phase_3_correct_state"]
    assert not probe.verify_probe(receipts, fixture, [10, 10], True)["checks"]["actual_app_server_process_replaced"]
