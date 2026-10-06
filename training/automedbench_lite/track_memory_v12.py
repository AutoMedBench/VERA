"""Explicit v1.2 SDK-memory addition beside the unchanged canonical catalog."""
from dataclasses import replace
from pathlib import Path

from eva_agent.codex_runtime import research_memory as memory_source, runtime as runtime_source
from eva_agent.codex_runtime import supra as supra_source
from eva_agent.codex_runtime.research_memory import (
    SKILL_PATH, MEMORY_INSTRUCTIONS, ResearchContextPolicy, memory_launch_options,
    memory_thread_options, memory_turn_input, summary_failures_skill,
)
from eva_agent.pipeline.digests import blake3_bytes, canonical_value
from .adapter import write_once

HARNESS_COMMIT = "a361f9c74e7df0047f9d5c82a7bdfcb46599f6e1"  # Historical memory behavior origin, not current runtime HEAD.


def selected_memory_sources():
    return {name: {"path": str(path.resolve()), "blake3": blake3_bytes(path.read_bytes())}
            for name, path in {"runtime_formatter": Path(runtime_source.__file__),
                               "memory_protocol": Path(memory_source.__file__),
                               "supra_profile": Path(supra_source.__file__),
                               "summary_skill": SKILL_PATH}.items()}


def make_supra_profile(setup, *, context_tokens, output_tokens, compact_tokens,
                       reserve_tokens=2048, measured_floor=13100, endpoint):
    return supra_source.SupraProfile(model=setup.model, provider=setup.provider,
        mode=supra_source.SupraMode.THINK, protocol="qwen_template", local_qwen_endpoint=endpoint,
        context=ResearchContextPolicy(context_tokens=context_tokens, output_tokens=output_tokens,
            reserve_tokens=reserve_tokens, compact_at_tokens=compact_tokens,
            compaction_headroom_mode="exact_request_guard"),
        measured_compacted_input_tokens=measured_floor)


def prepare_memory_profile(root: Path, canonical_skills, *, effective_context=None, supra=False):
    root.mkdir(mode=0o700)
    body = SKILL_PATH.read_bytes()
    mounted = root / "summary-failures.md"
    with mounted.open("xb") as stream:
        stream.write(body)
    mounted.chmod(0o400)
    skill = summary_failures_skill(mounted)
    original_count = len(canonical_skills.inventory) + 1  # native bootstrap + legacy24
    if original_count != 25:
        raise ValueError("original canonical skill inventory differs")
    policy = ResearchContextPolicy()
    document = write_once(root / "profile.json", {
        "schema": "eva.automedbench-extra-memory-skill-binding.v2",
        "memory_behavior_origin_commit": HARNESS_COMMIT,
        "profile": "evamed-codex-v1.3-supra" if supra else "evamed-codex-v1.2-research",
        "selected_memory_sources": selected_memory_sources(),
        "canonical_mcp_catalog_blake3": canonical_skills.catalog_blake3,
        "canonical_mcp_payload_count": original_count,
        "canonical_mcp_catalog_and_tool_schemas_changed": False,
        "extra_sdk_skills": [canonical_value(skill.catalog_entry())],
        "extra_sdk_source": {"path": str(SKILL_PATH), "bytes": len(body), "blake3": blake3_bytes(body)},
        "effective_global_skill_payload_union_count": original_count + 1,
        "stage_visible_mcp_subset_is_not_global_union": True,
        "delivery": "explicit_CodexTurnInput.skills_not_canonical_load_skill",
        "memory_instructions": MEMORY_INSTRUCTIONS,
        "supra_workflow_instructions": supra_source.WORKFLOW_INSTRUCTIONS if supra else None,
        "memory_default_context_policy": canonical_value(policy),
        "effective_context_policy": effective_context,
        "model_authored_notes_prefilled": False, "memory_behavior_claimed": False,
        "source_files": [{"path": str(path), "blake3": blake3_bytes(path.read_bytes())}
            for path in (Path(__file__), SKILL_PATH, Path(memory_source.__file__),
                         Path(runtime_source.__file__), Path(supra_source.__file__))],
    })
    return mounted, document


def apply_setup(setup):
    return replace(setup, launch=memory_launch_options(setup.launch))


def apply_thread(options, *, supra_profile=None):
    return supra_profile.thread_options(options) if supra_profile is not None else memory_thread_options(options)


def apply_turn(value, mounted, *, supra_profile=None):
    if supra_profile is None:
        return memory_turn_input(value, skill_path=mounted)
    actual = supra_profile.turn_input(value)  # Actual v1.3 behavior, including effort and one memory composition.
    retained = summary_failures_skill(mounted)
    source = [skill for skill in actual.skills if skill.skill_id == retained.skill_id]
    if len(source) != 1 or source[0].content_blake3 != retained.content_blake3:
        raise ValueError("supra selected skill differs from retained exact public body")
    # Only remount the identical body to its immutable run-local evidence path.
    return replace(actual, skills=tuple(retained if skill.skill_id == retained.skill_id else skill
                                        for skill in actual.skills))
