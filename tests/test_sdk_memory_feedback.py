import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eva_agent.codex_runtime.research_memory import SKILL_PATH, summary_failures_skill
from eva_agent.codex_runtime import runtime as runtime_source
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value
from training.automedbench_lite.adapter import write_once
from training.automedbench_lite.sdk_memory_feedback import selected_sdk_memory_context


@pytest.fixture(params=["historical", "external_v2"])
def bound(tmp_path, request):
    run = tmp_path / "evaluation-attempt/benchmark/run"
    directory = run / "track-rollouts/extra-memory-skill"
    directory.mkdir(parents=True)
    path = directory / "summary-failures.md"
    path.write_bytes(SKILL_PATH.read_bytes())
    path.chmod(0o400)
    selected = [canonical_value(summary_failures_skill(path).catalog_entry())]
    runtime = Path(runtime_source.__file__)
    source = {"path": str(runtime.resolve()), "blake3": blake3_bytes(runtime.read_bytes())}
    version = ({"schema": "eva.automedbench-extra-memory-skill-binding.v2",
                "profile": "evamed-codex-v1.3-supra",
                "selected_memory_sources": {"runtime_formatter": source}}
               if request.param == "external_v2" else {
                "schema": "eva.automedbench-v1.2-extra-memory-skill-binding.v1",
                "harness_commit": "a361f9c74e7df0047f9d5c82a7bdfcb46599f6e1"})
    binding = write_once(directory / "profile.json", {
        **version,
        "canonical_mcp_catalog_blake3": "a" * 64, "canonical_mcp_payload_count": 25,
        "effective_global_skill_payload_union_count": 26,
        "canonical_mcp_catalog_and_tool_schemas_changed": False,
        "extra_sdk_skills": selected, "extra_sdk_source": {"blake3": selected[0]["content_blake3"]}})
    write_once(run / "baseline-launch.json", {"selected_memory_sources": {"runtime_formatter": source}, "runtime_source_blake3": {
        "src/eva_agent/codex_runtime/runtime.py": blake3_bytes(runtime.read_bytes())}})
    attempt = {"extra_memory_skill_binding": binding, "verified_skill_catalog_blake3": "a" * 64}
    request = {"logical_input": {"skills": selected, "public_text": "Public task"}, "extra_memory_skill_binding_blake3": binding["document_blake3"]}
    receipt = SimpleNamespace(selected_skill_ids=("summary_failures",), selected_skill_catalog_blake3=blake3_hex(selected))
    return run, attempt, request, receipt


def test_actual_skill_body_reconstructed_without_changing_canonical_identity(bound):
    import json
    text, proof = selected_sdk_memory_context(*bound)
    assert json.loads(text)["skills"][0]["content"] == SKILL_PATH.read_text()
    assert proof["canonical_mcp_payload_count"] == 25
    assert proof["effective_global_skill_payload_union_count"] == 26
    assert proof["counts_as_canonical_mcp_skill_load"] is False


def test_changed_selected_skill_bytes_fail_closed(bound):
    path = bound[0] / "track-rollouts/extra-memory-skill/summary-failures.md"
    path.chmod(0o600)
    path.write_text("changed public content")
    with pytest.raises(ValueError, match="body_differs"):
        selected_sdk_memory_context(*bound)


def test_logical_selected_catalog_must_equal_bound_actual_sdk_catalog(bound):
    bound[2]["logical_input"]["skills"] = []
    with pytest.raises(ValueError, match="catalog_differs"):
        selected_sdk_memory_context(*bound)


def test_historical_no_selected_sdk_skill_is_unchanged(tmp_path):
    assert selected_sdk_memory_context(tmp_path, {}, {"logical_input": {"skills": []}}, None) == (None, None)


def modern_source_binding(bound):
    run, attempt, _, _ = bound
    runtime = Path(runtime_source.__file__).resolve()
    identity_path = run.parent.parent / "serving/checkpoint-identity.json"
    identity_path.parent.mkdir()
    identity = {"schema": "eva.qwen-final-serving-checkpoint-identity.v1",
        "exact_final_model_path": "/fixture/trained-hf"}
    identity_path.write_text(json.dumps(identity))
    identity_ref = {"path": str(identity_path), "blake3": blake3_bytes(identity_path.read_bytes())}
    attempt["server_binding"] = {"identity": identity, "identity_file_blake3": identity_ref["blake3"],
        "canary": {"checkpoint_identity_blake3": identity_ref["blake3"],
                   "exact_final_model_path": identity["exact_final_model_path"]}}
    profile = ("supra-v1.3", False, True) if attempt["extra_memory_skill_binding"]["profile"] == "evamed-codex-v1.3-supra" else ("memory-v1.2", True, False)
    harness = runtime.parents[3]
    return {"schema": "eva.rsi-evaluation-actor-binding.v1",
        "actual_actor_function": "training.automedbench_lite.track_actor.run_tracks",
        "canonical_skill_ids_replaced": False, "model_path": identity["exact_final_model_path"],
        "checkpoint_identity": identity_ref, "selected_harness_root": str(harness),
        "equivalent_actor_cli": ["python", "track_entry.py", "run", "--run-root", str(run)],
        "actual_imported_sources": [],
        "profile_requested": {"name": profile[0], "memory_profile": profile[1], "supra_profile": profile[2]}}


def test_modern_in_process_actor_binding_replaces_missing_baseline_launch(bound):
    if bound[1]["extra_memory_skill_binding"]["schema"] != "eva.automedbench-extra-memory-skill-binding.v2":
        pytest.skip("modern source binding applies only to v2 profiles")
    (bound[0] / "baseline-launch.json").unlink()
    source = modern_source_binding(bound)
    text, proof = selected_sdk_memory_context(*bound, source_binding=source)
    assert text and proof["actual_formatter_source_blake3"] == bound[1]["extra_memory_skill_binding"][
        "selected_memory_sources"]["runtime_formatter"]["blake3"]


@pytest.mark.parametrize("identical", [True, False])
def test_separate_judge_checkout_requires_identical_actor_formatter(bound, tmp_path, monkeypatch, identical):
    if bound[1]["extra_memory_skill_binding"]["schema"] != "eva.automedbench-extra-memory-skill-binding.v2":
        pytest.skip("modern source binding applies only to v2 profiles")
    source = modern_source_binding(bound)
    recorded = dict(bound[1]["extra_memory_skill_binding"]["selected_memory_sources"]["runtime_formatter"])
    imported = tmp_path / "new-judge-runtime.py"
    imported.write_bytes(Path(runtime_source.__file__).read_bytes() + (b"" if identical else b"\n# changed\n"))
    monkeypatch.setattr(runtime_source, "__file__", str(imported))
    if identical:
        text, proof = selected_sdk_memory_context(*bound, source_binding=source)
        assert text and proof["actual_formatter_source_blake3"] == recorded["blake3"]
        assert bound[1]["extra_memory_skill_binding"]["selected_memory_sources"]["runtime_formatter"] == recorded
    else:
        with pytest.raises(ValueError, match="actor_skill_input_formatter_source_differs"):
            selected_sdk_memory_context(*bound, source_binding=source)


@pytest.mark.parametrize("changed", ["run", "checkpoint", "harness", "profile"])
def test_modern_actor_source_binding_fails_closed(bound, changed):
    if bound[1]["extra_memory_skill_binding"]["schema"] != "eva.automedbench-extra-memory-skill-binding.v2":
        pytest.skip("modern source binding applies only to v2 profiles")
    source = modern_source_binding(bound)
    if changed == "run":
        source["equivalent_actor_cli"][-1] = str(bound[0].parent)
    elif changed == "checkpoint":
        source["checkpoint_identity"]["blake3"] = "0" * 64
    elif changed == "harness":
        source["selected_harness_root"] = str(bound[0])
    else:
        source["profile_requested"]["name"] = "memory-v1.2"
    with pytest.raises((ValueError, FileNotFoundError), match="extra_sdk|formatter"):
        selected_sdk_memory_context(*bound, source_binding=source)
