"""Reopen the actual v1.2 SDK skill input separately from canonical MCP skills."""
import json
from pathlib import Path

from eva_agent.codex_runtime.contracts import CodexSkill, CodexTurnInput
from eva_agent.codex_runtime.runtime import _skill_text
from eva_agent.codex_runtime import runtime as runtime_source
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value
from .adapter import read_document, safe_file
from training.benchmark_feedback.automed_codex import require


def _modern_actor_runtime_source(run_root, attempt, binding, source_binding, runtime):
    """Bind the in-process evaluation actor without inventing a baseline launch."""
    require(isinstance(source_binding, dict), "extra_sdk_memory_binding_differs")
    run = Path(run_root).resolve(strict=True)
    source_attempt = run.parent.parent
    identity_ref = source_binding.get("checkpoint_identity", {})
    require(isinstance(identity_ref, dict) and isinstance(identity_ref.get("path"), str)
            and isinstance(source_binding.get("actual_imported_sources"), list)
            and all(isinstance(row, dict) for row in source_binding["actual_imported_sources"]),
            "extra_sdk_memory_binding_differs")
    identity_path = Path(identity_ref.get("path", "")).resolve(strict=True)
    identity_bytes = identity_path.read_bytes()
    identity = json.loads(identity_bytes)
    require(isinstance(identity, dict), "extra_sdk_memory_binding_differs")
    argv = source_binding.get("equivalent_actor_cli", ())
    positions = [index for index, value in enumerate(argv) if value == "--run-root"]
    profile = source_binding.get("profile_requested", {})
    expected_profile = {
        "evamed-codex-v1.2-research": ("memory-v1.2", True, False),
        "evamed-codex-v1.3-supra": ("supra-v1.3", False, True),
    }.get(binding.get("profile"))
    server = attempt.get("server_binding", {})
    require(source_binding.get("schema") == "eva.rsi-evaluation-actor-binding.v1"
            and source_binding.get("actual_actor_function") == "training.automedbench_lite.track_actor.run_tracks"
            and source_binding.get("canonical_skill_ids_replaced") is False
            and len(positions) == 1 and positions[0] + 1 < len(argv)
            and Path(argv[positions[0] + 1]).resolve(strict=True) == run
            and identity_path == source_attempt / "serving/checkpoint-identity.json"
            and set(identity_ref) == {"path", "blake3"}
            and identity_ref["blake3"] == blake3_bytes(identity_bytes)
            and source_binding.get("model_path") == identity.get("exact_final_model_path")
            and server.get("identity") == identity
            and server.get("identity_file_blake3") == identity_ref["blake3"]
            and server.get("canary", {}).get("checkpoint_identity_blake3") == identity_ref["blake3"]
            and server.get("canary", {}).get("exact_final_model_path") == identity.get("exact_final_model_path")
            and expected_profile is not None
            and (profile.get("name"), profile.get("memory_profile"), profile.get("supra_profile")) == expected_profile,
            "extra_sdk_memory_binding_differs")
    harness = Path(source_binding.get("selected_harness_root", "")).resolve(strict=True)
    recorded = binding["selected_memory_sources"]["runtime_formatter"]
    expected_runtime = harness / "src/eva_agent/codex_runtime/runtime.py"
    # A later Judge checkout may import the identical formatter at a new path.
    # Reopen the actor's exact recorded source AND require identical imported
    # bytes; keep the original actor binding, not the Judge's checkout path.
    require(expected_runtime.is_file() and not expected_runtime.is_symlink()
            and recorded == {"path": str(expected_runtime.resolve()),
                             "blake3": blake3_bytes(expected_runtime.read_bytes())}
            and blake3_bytes(runtime.read_bytes()) == recorded["blake3"],
            "actor_skill_input_formatter_source_differs")
    return recorded


def selected_sdk_memory_context(run_root, attempt, request, receipt, *, source_binding=None):
    binding = attempt.get("extra_memory_skill_binding")
    selected = request["logical_input"]["skills"]
    if binding is None:
        require(not selected, "unbound_selected_sdk_skill_content")
        return None, None
    directory = Path(run_root) / "track-rollouts/extra-memory-skill"
    modern = binding.get("schema") == "eva.automedbench-extra-memory-skill-binding.v2"
    require(binding == read_document(directory / "profile.json")
            and (modern or (binding["schema"] == "eva.automedbench-v1.2-extra-memory-skill-binding.v1"
                 and binding["harness_commit"] == "a361f9c74e7df0047f9d5c82a7bdfcb46599f6e1"))
            and binding["canonical_mcp_catalog_blake3"] == attempt["verified_skill_catalog_blake3"]
            and binding["canonical_mcp_payload_count"] == 25
            and binding["effective_global_skill_payload_union_count"] == 26
            and binding["canonical_mcp_catalog_and_tool_schemas_changed"] is False,
            "extra_sdk_memory_binding_differs")
    require(request["extra_memory_skill_binding_blake3"] == binding["document_blake3"]
            and selected == binding["extra_sdk_skills"] and len(selected) == 1
            and selected[0]["skill_id"] == "summary_failures"
            and receipt.selected_skill_ids == ("summary_failures",)
            and receipt.selected_skill_catalog_blake3 == blake3_hex(selected), "actual_selected_sdk_catalog_differs")
    path = safe_file(directory, "summary-failures.md", maximum=16384)
    require(Path(selected[0]["path"]) == path
            and blake3_bytes(path.read_bytes()) == selected[0]["content_blake3"] == binding["extra_sdk_source"]["blake3"],
            "actual_selected_sdk_body_differs")
    # This is the same exact public input formatter used by the recorded actor,
    # not a new helper prompt or a skill body inferred from its name.
    text = _skill_text(CodexTurnInput(public_text=request["logical_input"]["public_text"],
                                    skills=tuple(CodexSkill(**row) for row in selected)))
    runtime = Path(runtime_source.__file__).resolve()
    if modern:
        recorded = binding["selected_memory_sources"]["runtime_formatter"]
        if source_binding is not None:
            _modern_actor_runtime_source(run_root, attempt, binding, source_binding, runtime)
        else:
            launch = read_document(Path(run_root) / "baseline-launch.json")
            require(recorded == launch["selected_memory_sources"]["runtime_formatter"]
                    and recorded["blake3"] == blake3_bytes(runtime.read_bytes()),
                    "actor_skill_input_formatter_source_differs")
    else:
        launch = read_document(Path(run_root) / "baseline-launch.json")
        require(launch["runtime_source_blake3"]["src/eva_agent/codex_runtime/runtime.py"] == blake3_bytes(runtime.read_bytes()),
                "actor_skill_input_formatter_source_differs")
    return text, {"extra_memory_skill_binding_blake3": binding["document_blake3"],
        "canonical_mcp_catalog_blake3": attempt["verified_skill_catalog_blake3"],
        "canonical_mcp_payload_count": 25, "extra_sdk_payload_count": 1,
        "effective_global_skill_payload_union_count": 26,
        "actual_selected_sdk_catalog_blake3": receipt.selected_skill_catalog_blake3,
        "exact_runtime_skill_input_blake3": blake3_bytes(text.encode()),
        "exact_runtime_skill_input_bytes": len(text.encode()),
        "actual_formatter_source_blake3": blake3_bytes(runtime.read_bytes()),
        "original_logical_input_unchanged": True,
        "counts_as_canonical_mcp_skill_load": False}
