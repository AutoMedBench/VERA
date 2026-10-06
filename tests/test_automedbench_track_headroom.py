"""Future actor headroom/public guidance only; no provider, Docker or GPU."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import jsonschema
import pytest

from eva_agent.pipeline.digests import canonical_value
from training.automedbench_lite import track_actor as actor
from training.automedbench_lite.adapter import write_once
from training.automedbench_lite.track_tools import TOOLS


def test_real_run_setup_receives_earlier_compaction_and_unchanged_output_controls(tmp_path, monkeypatch):
    workspace = tmp_path / "actor"
    (workspace / "notes").mkdir(parents=True)
    (workspace / "outputs/agents_outputs").mkdir(parents=True)
    write_once(tmp_path / "track-run-manifest.json", {"run_id": "fixture", "tracks": [
        {"track": "vqa", "workspace_relative": "actor", "task_file_blake3": "fixture",
         "input_manifest_file_blake3": "fixture"}]})
    monkeypatch.setattr(actor, "resolve_image", lambda _: "unused-fixture-image")
    monkeypatch.setattr(actor, "serving_binding", lambda *_: {})
    monkeypatch.setattr(actor, "file_digest", lambda _: "fixture")
    monkeypatch.setattr(actor, "VerifiedEvaluationSkills", lambda _: SimpleNamespace(catalog_blake3="fixture", inventory=[]))
    monkeypatch.setattr(actor, "MutableInventory", lambda *_: None)
    captured = {}
    class StopBeforeAnyBackend(Exception):
        pass
    def capture(**kwargs):
        captured.update(kwargs)
        raise StopBeforeAnyBackend
    monkeypatch.setattr(actor, "local_qwen_setup", capture)
    args = SimpleNamespace(run_root=tmp_path, tracks=["all"], image="fixture", server_canary=None,
        server_identity=None, runtime_manifest=tmp_path / "runtime.json", codex_bin=tmp_path / "codex")
    with pytest.raises(StopBeforeAnyBackend):
        asyncio.run(actor.run_tracks(args))
    assert captured["auto_compact_token_limit"] == 12288
    assert captured["max_output_tokens"] == 4096 and captured["upstream_timeout_seconds"] == 600
    assert captured["token_budget"] and captured["thinking"] and captured["normalize_priority_messages"]


def test_public_argument_examples_match_exact_offered_schemas():
    before = deepcopy(TOOLS)
    schemas = {row["name"]: row["inputSchema"] for row in TOOLS}
    for name, arguments in actor.PUBLIC_ARGUMENT_EXAMPLES:
        jsonschema.Draft202012Validator(schemas[name]).validate(arguments)
    read = dict(actor.PUBLIC_ARGUMENT_EXAMPLES)["automed_read_file"]
    note = dict(actor.PUBLIC_ARGUMENT_EXAMPLES)["automed_write_note"]
    assert read["limit"] == 2048 and read["offset"] == 0
    assert set(note) == {"name", "content"} and note["name"] == "progress.md"
    assert "not calls to execute automatically" in actor.PUBLIC_CONTEXT_GUIDANCE
    assert TOOLS == before


def test_guidance_is_actually_delivered_without_private_material(tmp_path):
    setup = SimpleNamespace(thread_config={}, model="Qwen/Qwen3.5-9B", provider="eva_local_qwen")
    args = SimpleNamespace(public_python=tmp_path / "python", runtime_manifest=tmp_path / "models.json")
    options = actor.thread_options(setup, tmp_path, tmp_path / "audit", args, "unused-image")
    text = options.developer_instructions
    assert actor.PUBLIC_CONTEXT_GUIDANCE in text
    for phrase in ("limit <= 2048", "Do not issue parallel large text reads", "complete=false",
                   "selected case IDs", "durable notes", "not path", "Omit sigma, hu_min, hu_max and confidence"):
        assert phrase in text
    for private_marker in ("/scorer-only", "auth.json", "PRIVATE_KEY", "token_budget_failed", "41497", "59231f39"):
        assert private_marker not in actor.PUBLIC_CONTEXT_GUIDANCE
    assert [offer.fully_qualified_name for offer in options.offered_tools] == ["automed_eval/" + row["name"] for row in TOOLS]
    for offered, original in zip(options.offered_tools, TOOLS):
        assert offered.description == original["description"]
        assert canonical_value(offered.input_schema) == original["inputSchema"]


def test_extended_job_example_shape_omits_other_track_controls():
    schema = next(row["inputSchema"] for row in TOOLS if row["name"] == "automed_submit_extended_model_job")
    jsonschema.Draft202012Validator(schema).validate({"case_ids": ["public-fixture-case"]})
    assert "Only enhancement uses explicit sigma, hu_min and hu_max" in actor.PUBLIC_CONTEXT_GUIDANCE
    assert "do not send null or guessed defaults" in actor.PUBLIC_CONTEXT_GUIDANCE
    assert "bounded reads or public Python summaries" in actor.PHASES[0][2]
