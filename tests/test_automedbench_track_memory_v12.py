from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from eva_agent.codex_runtime import CodexRole, CodexSandbox, CodexThreadOptions, CodexTurnInput
from eva_agent.codex_runtime.research_memory import MEMORY_INSTRUCTIONS, SKILL_PATH
from eva_agent.pipeline.digests import blake3_bytes, canonical_value
from training.automedbench_lite import track_actor
from training.automedbench_lite.track_memory_v12 import prepare_memory_profile, apply_thread, apply_turn
from training.automedbench_lite.skill_surface import VerifiedEvaluationSkills


def test_actual_catalog_and_extra_sdk_memory_are_separately_bound(tmp_path):
    skills = VerifiedEvaluationSkills(tmp_path / "skills")
    original = canonical_value(skills.inventory)
    mounted, binding = prepare_memory_profile(tmp_path / "extra", skills)
    assert binding["canonical_mcp_payload_count"] == 25
    assert binding["effective_global_skill_payload_union_count"] == 26
    assert canonical_value(skills.inventory) == original
    assert binding["canonical_mcp_catalog_blake3"] == skills.catalog_blake3
    assert mounted.read_bytes() == SKILL_PATH.read_bytes()
    assert mounted.stat().st_mode & 0o777 == 0o400
    value = apply_turn(CodexTurnInput(public_text="Actual public task"), mounted)
    assert len(value.skills) == 1 and value.skills[0].skill_id == "summary_failures"
    assert value.skills[0].content_blake3 == blake3_bytes(SKILL_PATH.read_bytes())
    for phase, stage, _, _ in [track_actor.phase_prompt(skills, i, "classification") for i in range(5)]:
        result = skills.call("search_skills", {"query": " ", "stage": stage}, stage=stage)
        assert "summary_failures" not in {row["skill_id"] for row in result["matches"]}


def test_memory_options_keep_model_security_tools_and_actual_instructions(tmp_path):
    base = CodexThreadOptions(role=CodexRole.STRONG_ACTOR, provider="test-provider", model="test-model",
        cwd=str(tmp_path), sandbox=CodexSandbox.READ_ONLY, ephemeral=False,
        config={"model_context_window": 32768, "model_auto_compact_token_limit": 12288},
        developer_instructions=track_actor.DEVELOPER, base_instructions=track_actor.BASE)
    actual = apply_thread(base)
    assert actual.model == base.model and actual.provider == base.provider
    assert actual.sandbox == base.sandbox and not actual.ephemeral
    assert actual.offered_tools == base.offered_tools
    assert actual.base_instructions == base.base_instructions
    assert actual.developer_instructions == track_actor.DEVELOPER + "\n\n" + MEMORY_INSTRUCTIONS
    assert dict(actual.config) == dict(base.config)


def test_track_runner_keeps_restart_resume_and_owned_budget_contract():
    # Source-level integration guard complements the existing actual mock-runtime
    # restart/receipt tests and owned-interrupt tests; this does not claim recall.
    import inspect
    source = inspect.getsource(track_actor.run_tracks)
    assert "runtime.resume_thread(state[\"thread_id\"], state[\"options\"])" in source
    assert source.count("async with CodexRuntime(backend)") == 2
    assert "capture_turn(runtime" in source
    assert "thinking=True" in source and "token_budget=True" in source
    assert "memory_profile=False" in source


def test_explicit_three_worker_budget_is_shared_by_semaphore_gateway_and_receipt():
    import inspect
    source = inspect.getsource(track_actor.run_tracks)
    assert 'slots = asyncio.Semaphore(workers)' in source
    assert 'workers=min(workers, len(states))' in source
    assert '"max_parallel_tracks": workers' in source
    assert 'getattr(args, "workers", 4)' in source
    import asyncio
    import pytest
    with pytest.raises(ValueError):
        asyncio.run(track_actor.run_tracks(SimpleNamespace(workers=5)))


def test_memory_selection_survives_actual_runtime_close_and_resume_api(tmp_path):
    import asyncio
    from test_codex_runtime import _FakeBackend
    from eva_agent.codex_runtime import CodexRuntime
    skills = VerifiedEvaluationSkills(tmp_path / "skills")
    mounted, _ = prepare_memory_profile(tmp_path / "extra", skills)
    options = apply_thread(CodexThreadOptions(role=CodexRole.STRONG_ACTOR,
        provider="fixture", model="fixture", cwd=str(tmp_path), sandbox=CodexSandbox.READ_ONLY,
        ephemeral=False, developer_instructions=track_actor.DEVELOPER))
    value = apply_turn(CodexTurnInput(public_text="Continue actual public work"), mounted)
    async def scenario():
        first, second = _FakeBackend(), _FakeBackend()
        async with CodexRuntime(first) as runtime:
            handle = await runtime.start_thread(options)
            one = await runtime.run_turn(handle, value)
        assert first.closed
        async with CodexRuntime(second) as runtime:
            resumed = await runtime.resume_thread(handle.thread_id, options)
            two = await runtime.run_turn(resumed, value)
        return one, two, second
    one, two, second = asyncio.run(scenario())
    assert one.thread_id == two.thread_id and two.thread_resumed
    assert one.selected_skill_ids == two.selected_skill_ids == ("summary_failures",)
    assert second.resumed[0][1].developer_instructions.endswith(MEMORY_INSTRUCTIONS)
    import json
    sidecars = [json.loads(item.text) for _, items, _ in second.turn_inputs for item in items
                if item.text and item.text.startswith("{")]
    assert any(value.get("schema") == "eva.codex-skill-mount-sidecar.v1"
               and value["skills"][0]["content"] == SKILL_PATH.read_text() for value in sidecars)


def test_actual_supra_composes_once_with_measured_context_and_exact_sdk_body(tmp_path):
    from eva_agent.codex_runtime.supra import WORKFLOW_INSTRUCTIONS
    from training.automedbench_lite.local_qwen import local_qwen_setup
    from training.automedbench_lite.track_memory_v12 import make_supra_profile
    skills = VerifiedEvaluationSkills(tmp_path / "skills")
    context = {"context_tokens": 32768, "compact_at_tokens": 20480,
               "output_tokens": 4096, "reserve_tokens": 2048,
               "compaction_headroom_mode": "exact_request_guard"}
    mounted, binding = prepare_memory_profile(tmp_path / "extra", skills,
                                              effective_context=context, supra=True)
    with local_qwen_setup(run_root=tmp_path / "runtime", workers=3, thinking=True,
            context_length=32768, auto_compact_token_limit=20480, max_output_tokens=4096,
            capacity_wait_seconds=60, token_budget=True) as setup:
        profile = make_supra_profile(setup, context_tokens=32768, output_tokens=4096,
            compact_tokens=20480, endpoint="http://127.0.0.1:30910/v1")
        base = CodexThreadOptions(role=CodexRole.STRONG_ACTOR, provider=setup.provider,
            model=setup.model, cwd=str(tmp_path), sandbox=CodexSandbox.READ_ONLY,
            ephemeral=False, config=setup.thread_config,
            base_instructions=track_actor.BASE, developer_instructions=track_actor.DEVELOPER)
        options = apply_thread(base, supra_profile=profile)
        value = apply_turn(CodexTurnInput(public_text="Actual public workflow"), mounted,
                           supra_profile=profile)
        assert options.developer_instructions.count(MEMORY_INSTRUCTIONS) == 1
        assert options.developer_instructions.count(WORKFLOW_INSTRUCTIONS) == 1
        assert options.config["model_auto_compact_token_limit"] == 20480
        assert options.config["model_context_window"] == 32768
        assert len(value.skills) == 1 and value.skills[0].content_blake3 == blake3_bytes(SKILL_PATH.read_bytes())
        assert value.skills[0].path == str(mounted)
        assert value.effort == "xhigh" and profile.inspection()["qwen_thinking_requested"] is True
        assert setup.safe_metadata["text_token_budget_enabled"] is True
        assert setup.safe_metadata["provider_calls_on_setup"] == 0
    assert binding["profile"] == "evamed-codex-v1.3-supra"
    assert binding["effective_context_policy"] == context
    assert binding["memory_default_context_policy"]["compact_at_tokens"] == 12288
    assert binding["effective_global_skill_payload_union_count"] == 26
    assert not list((tmp_path / "runtime").glob("**/adapter-receipt.json"))


def test_native_tool_context_bound_preserves_canonical_offer_and_original_options(tmp_path):
    setup = SimpleNamespace(model="Qwen/Qwen3.5-9B", provider="eva_local_qwen",
                            thread_config={"model_context_window": 32768})
    args = SimpleNamespace(public_python=Path("/usr/bin/python3"), runtime_manifest=tmp_path / "models.json")
    original = track_actor.thread_options(setup, tmp_path, tmp_path / "audit", args, "fixture-image")
    args.tool_output_token_limit = 2048
    bounded = track_actor.thread_options(setup, tmp_path, tmp_path / "audit", args, "fixture-image")
    assert bounded.config["tool_output_token_limit"] == 2048
    assert "tool_output_token_limit" not in original.config
    assert bounded.offered_tools == original.offered_tools
    assert bounded.developer_instructions == original.developer_instructions
    from training.automedbench_lite.track_memory_v12 import make_supra_profile
    profile = make_supra_profile(setup, context_tokens=32768, output_tokens=4096,
        compact_tokens=20480, endpoint="http://127.0.0.1:30910/v1")
    assert apply_thread(bounded, supra_profile=profile).config["tool_output_token_limit"] == 2048
