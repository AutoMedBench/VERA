from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from eva_agent.codex_pipeline import VerifiedActorSkillCatalog
from eva_agent.harness.skills import SkillCatalog, SkillDocument
from eva_agent.pipeline import (
    BenchmarkEpisode,
    BenchmarkSource,
    Cohort,
    FilesystemSandbox,
    ModelTarget,
    ParallelToolRuntime,
    RandomUUIDFactory,
    RolloutRequest,
    SandboxManifest,
    Stage,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
)
from eva_agent.pipeline.digests import canonical_value
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.sources.legacy_execution import EXPECTED_TOOL_NAMES
from eva_agent.training import ProgressiveTeacherSkillSurface, TeacherBatchError


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "evamed-codex"
LEGACY_SOURCE = (
    ROOT.parent
    / "rlevo-med-research"
    / "harness"
    / "source"
    / "rlevo-Med-RL-data"
    / "rev-79dd2a31f5f"
)


def _surface(tmp_path: Path) -> ProgressiveTeacherSkillSurface:
    runtime = tmp_path / "skill-runtime"
    runtime.mkdir(mode=0o700)
    runtime.chmod(0o700)
    verified = VerifiedActorSkillCatalog(
        manifest_path=(PLUGIN / "references/legacy-skill-manifest.v1.json").resolve(),
        legacy_source_root=LEGACY_SOURCE.resolve(),
        native_stage_skill_path=(PLUGIN / "skills/stage-rollout/SKILL.md").resolve(),
        runtime_root=runtime.resolve(),
    )
    return ProgressiveTeacherSkillSurface(verified)


def _request(stage: Stage, tools: ToolRegistry) -> RolloutRequest:
    rubric = load_and_compile_registry(
        ROOT / "rubrics/source/domain-stage-tables.v1.json"
    ).resolve("medxpertqa", stage.value)
    episode = BenchmarkEpisode(
        episode_id="teacher-progressive-fixture",
        source=BenchmarkSource("MedXpertQA", "fixture.json", "fixture-v1"),
        domain="medxpertqa",
        stage=stage,
        instruction="Plan, discover skills, and use candidate tools.",
        policy_context={"question": "fixture"},
        initial_files={},
    )
    manifest = SandboxManifest.create(
        sandbox_id=str(uuid4()), episode=episode, rubric=rubric
    )
    return RolloutRequest(
        rollout_id=str(uuid4()),
        sandbox=manifest,
        model=ModelTarget(Cohort.STRONG, "teacher", "fixture"),
        policy_visible_context=episode.policy_context,
        available_tools=tools.public_schemas(),
    )


def test_teacher_progressive_surface_preserves_candidate_schemas_and_mounts_nothing(
    tmp_path: Path,
) -> None:
    candidate_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    candidate = ToolDefinition(
        name="retrieve_frozen_evidence",
        description="Exact candidate-scoped source tool.",
        input_schema=candidate_schema,
        handler=lambda _workspace, arguments: {"query": arguments["query"]},
        parallel_safe=True,
        read_only=True,
    )
    base = ToolRegistry((candidate,))
    surface = _surface(tmp_path)
    augmented = surface.augment_registry(base, Stage.S3)
    names = tuple(
        schema["function"]["name"] for schema in augmented.public_schemas()
    )
    assert names == ("load_skill", "retrieve_frozen_evidence", "search_skills")
    assert augmented.definition("retrieve_frozen_evidence") is candidate
    assert canonical_value(augmented.definition("retrieve_frozen_evidence").input_schema) == canonical_value(
        candidate_schema
    )

    canonical_skill_tools = {
        item.name: item
        for item in SkillCatalog(
            (SkillDocument("fixture", "fixture", "fixture"),)
        ).tool_definitions()
    }
    for name in ("search_skills", "load_skill"):
        added = augmented.definition(name)
        canonical = canonical_skill_tools[name]
        assert added.description == canonical.description
        assert canonical_value(added.input_schema) == canonical_value(canonical.parameters)
        assert added.kind == "skill"
        assert added.parallel_safe is True
        assert added.read_only is True

    request = _request(Stage.S3, augmented)
    assert surface(request) == ()
    metadata = surface.public_metadata(Stage.S3)
    assert metadata["mode"] == "search-then-load"
    assert metadata["initial_skill_mount_count"] == 0
    assert metadata["visible_skill_count"] == 20
    assert metadata["discovery_tools"] == ["search_skills", "load_skill"]


def test_teacher_skill_search_and_load_are_stage_bound_and_parallel(
    tmp_path: Path,
) -> None:
    surface = _surface(tmp_path)
    registry = surface.augment_registry(ToolRegistry(()), Stage.S3)
    workspace = FilesystemSandbox(tmp_path / "workspaces", str(uuid4()), {})

    search = registry.definition("search_skills").handler(
        workspace, {"query": "pilot", "stage": "S3"}
    )
    skill_ids = {row["skill_id"] for row in search["matches"]}
    assert "pilot-recovery-validation-med" in skill_ids
    loaded = registry.definition("load_skill").handler(
        workspace,
        {"skill_id": "pilot-recovery-validation-med", "stage": "S3"},
    )
    assert loaded["skill_id"] == "pilot-recovery-validation-med"
    assert loaded["content"]
    assert loaded["delivery"] == "policy-visible-tool-observation"

    for name, arguments in (
        ("search_skills", {"query": "pilot", "stage": "S2"}),
        (
            "load_skill",
            {"skill_id": "pilot-recovery-validation-med", "stage": "S2"},
        ),
    ):
        with pytest.raises(TeacherBatchError, match="stage differs"):
            registry.definition(name).handler(workspace, arguments)

    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=registry,
        id_factory=RandomUUIDFactory(),
        maximum_parallel_calls=64,
    )
    calls = tuple(
        ToolCall(
            call_id=str(uuid4()),
            name="load_skill",
            arguments={
                "skill_id": "pilot-recovery-validation-med",
                "stage": "S3",
            },
        )
        for _ in range(2)
    )
    results = runtime.execute(calls)
    assert all(result.status == "completed" for result in results)
    assert runtime.trace().frontier_count == 1
    assert runtime.trace().max_parallelism_observed == 2
    assert workspace.snapshot("after-parallel-load").file_count == 0


def test_stateful_source_tools_stay_serial_while_discovery_is_parallel(
    tmp_path: Path,
) -> None:
    stateful = ToolRegistry(
        tuple(
            ToolDefinition(
                name=name,
                description=f"Exact stateful source tool {name}.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=lambda _workspace, _arguments: {},
                parallel_safe=False,
                read_only=False,
            )
            for name in EXPECTED_TOOL_NAMES
        )
    )
    augmented = _surface(tmp_path).augment_registry(stateful, Stage.E2E)
    for name in EXPECTED_TOOL_NAMES:
        definition = augmented.definition(name)
        assert definition is stateful.definition(name)
        assert definition.parallel_safe is False
        assert definition.read_only is False
    for name in ("search_skills", "load_skill"):
        definition = augmented.definition(name)
        assert definition.parallel_safe is True
        assert definition.read_only is True
