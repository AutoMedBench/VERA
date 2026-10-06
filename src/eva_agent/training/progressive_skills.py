"""Teacher-only progressive skill discovery over verified local skill bytes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from eva_agent.codex_runtime import CodexSkill
from eva_agent.harness.skills import (
    LOAD_SKILL_DESCRIPTION,
    LOAD_SKILL_INPUT_SCHEMA,
    SEARCH_SKILLS_DESCRIPTION,
    SEARCH_SKILLS_INPUT_SCHEMA,
)
from eva_agent.pipeline import RolloutRequest, Stage, ToolDefinition, ToolRegistry
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value

from .teacher_batch import TeacherBatchError


_NATIVE_SKILL_ID = "eva-workflow/stage-rollout"


@dataclass(frozen=True)
class _SkillDescriptor:
    skill: CodexSkill
    description: str
    allowed_stages: frozenset[str]


def _frontmatter_description(payload: bytes) -> str:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise TeacherBatchError("teacher skill is not UTF-8") from exc
    if not lines or lines[0] != "---":
        raise TeacherBatchError("teacher bootstrap skill frontmatter differs")
    try:
        end = lines.index("---", 1)
    except ValueError:
        raise TeacherBatchError("teacher bootstrap skill frontmatter differs") from None
    values: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    description = values.get("description")
    if not description:
        raise TeacherBatchError("teacher bootstrap skill description differs")
    return description


def _verified_skill_bytes(skill: CodexSkill) -> bytes:
    path = Path(skill.path)
    try:
        resolved = path.resolve(strict=True)
        payload = path.read_bytes()
    except (OSError, RuntimeError) as exc:
        raise TeacherBatchError("teacher skill bytes are unavailable") from exc
    if resolved != path or not path.is_file() or blake3_bytes(payload) != skill.content_blake3:
        raise TeacherBatchError("teacher skill bytes differ")
    return payload


class ProgressiveTeacherSkillSurface:
    """Expose verified skills through two stage-bound read-only MCP tools.

    No skill body is mounted in ``CodexTurnInput``.  The model sees the small
    search/load planning surface first, loads only relevant instructions, and
    then invokes the unchanged candidate-bound source tools already present in
    the base registry.  This object is used only by teacher rollout assembly;
    benchmark policies, reward rubrics, and bulk-RL records remain untouched.
    """

    def __init__(self, verified_catalog: Any) -> None:
        inventory = verified_catalog.inventory()
        if not isinstance(inventory, tuple):
            raise TeacherBatchError("verified teacher skill inventory differs")
        metadata: dict[str, Mapping[str, Any]] = {}
        for row in inventory:
            if not isinstance(row, Mapping):
                raise TeacherBatchError("verified teacher skill row differs")
            skill_id = row.get("skill_id")
            description = row.get("description")
            stages = row.get("allowed_stages")
            if (
                not isinstance(skill_id, str)
                or not isinstance(description, str)
                or not isinstance(stages, tuple)
                or skill_id in metadata
            ):
                raise TeacherBatchError("verified teacher skill row differs")
            metadata[skill_id] = row

        by_id: dict[str, _SkillDescriptor] = {}
        native: CodexSkill | None = None
        native_description: str | None = None
        stage_members: dict[str, set[str]] = {}
        for stage in Stage:
            selected = tuple(verified_catalog.for_stage(stage))
            if not selected or selected[0].skill_id != _NATIVE_SKILL_ID:
                raise TeacherBatchError("teacher bootstrap skill differs")
            current_native = selected[0]
            if native is None:
                native = current_native
                native_description = _frontmatter_description(
                    _verified_skill_bytes(current_native)
                )
            elif current_native != native:
                raise TeacherBatchError("teacher bootstrap skill differs by stage")
            stage_members[stage.value] = {skill.skill_id for skill in selected}
            for skill in selected[1:]:
                row = metadata.get(skill.skill_id)
                if row is None or stage.value not in row["allowed_stages"]:
                    raise TeacherBatchError("teacher stage skill grant differs")
                prior = by_id.get(skill.skill_id)
                descriptor = _SkillDescriptor(
                    skill=skill,
                    description=str(row["description"]),
                    allowed_stages=frozenset(str(item) for item in row["allowed_stages"]),
                )
                if prior is not None and prior != descriptor:
                    raise TeacherBatchError("teacher skill identity differs by stage")
                by_id[skill.skill_id] = descriptor
        if native is None or native_description is None:
            raise TeacherBatchError("teacher bootstrap skill is absent")
        by_id[_NATIVE_SKILL_ID] = _SkillDescriptor(
            skill=native,
            description=native_description,
            allowed_stages=frozenset(stage.value for stage in Stage),
        )
        expected_ids = set(metadata) | {_NATIVE_SKILL_ID}
        if set(by_id) != expected_ids or any(
            stage_members[stage.value]
            != {skill_id for skill_id, row in by_id.items() if stage.value in row.allowed_stages}
            for stage in Stage
        ):
            raise TeacherBatchError("teacher progressive skill grants differ")
        self._by_id = MappingProxyType(by_id)
        self.catalog_blake3 = str(verified_catalog.catalog_blake3)

    def __call__(self, request: RolloutRequest) -> tuple[CodexSkill, ...]:
        if not isinstance(request, RolloutRequest):
            raise TeacherBatchError("teacher skill factory requires a RolloutRequest")
        # Discovery is deliberately progressive: no complete skill body is
        # injected into the initial Codex context.
        return ()

    def augment_registry(
        self, registry: ToolRegistry, stage: Stage | str
    ) -> ToolRegistry:
        value = stage.value if isinstance(stage, Stage) else stage
        if value not in {item.value for item in Stage}:
            raise TeacherBatchError("teacher progressive skill stage differs")
        original = registry.definitions()
        original_names = {definition.name for definition in original}
        if original_names & {"search_skills", "load_skill"}:
            raise TeacherBatchError("candidate tools collide with teacher skill discovery")

        def require_stage(arguments: Mapping[str, Any]) -> None:
            if arguments.get("stage") != value:
                raise TeacherBatchError("teacher skill request stage differs")

        def search(_workspace: Any, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
            require_stage(arguments)
            query = arguments.get("query")
            if not isinstance(query, str) or not query:
                raise TeacherBatchError("teacher skill search query differs")
            needle = query.casefold()
            matches = [
                {"skill_id": skill_id, "description": descriptor.description}
                for skill_id, descriptor in self._by_id.items()
                if value in descriptor.allowed_stages
                and needle in f"{skill_id} {descriptor.description}".casefold()
            ]
            return {"matches": sorted(matches, key=lambda row: row["skill_id"])}

        def load(_workspace: Any, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
            require_stage(arguments)
            skill_id = arguments.get("skill_id")
            descriptor = self._by_id.get(skill_id) if isinstance(skill_id, str) else None
            if descriptor is None or value not in descriptor.allowed_stages:
                raise TeacherBatchError("skill is unavailable for this stage")
            payload = _verified_skill_bytes(descriptor.skill)
            try:
                content = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise TeacherBatchError("teacher skill is not UTF-8") from exc
            return {
                "skill_id": skill_id,
                "content": content,
                "content_blake3": blake3_hex(content),
                "delivery": "policy-visible-tool-observation",
            }

        search_tool = ToolDefinition(
            name="search_skills",
            description=SEARCH_SKILLS_DESCRIPTION,
            input_schema=SEARCH_SKILLS_INPUT_SCHEMA,
            handler=search,
            kind="skill",
            parallel_safe=True,
            read_only=True,
        )
        load_tool = ToolDefinition(
            name="load_skill",
            description=LOAD_SKILL_DESCRIPTION,
            input_schema=LOAD_SKILL_INPUT_SCHEMA,
            handler=load,
            kind="skill",
            parallel_safe=True,
            read_only=True,
        )
        augmented = ToolRegistry((*original, search_tool, load_tool))
        if tuple(
            schema
            for schema in augmented.public_schemas()
            if schema["function"]["name"] in original_names
        ) != tuple(sorted(registry.public_schemas(), key=lambda row: row["function"]["name"])):
            raise TeacherBatchError("candidate tool schemas changed during teacher augmentation")
        return augmented

    def public_metadata(self, stage: Stage | str) -> Mapping[str, Any]:
        value = stage.value if isinstance(stage, Stage) else stage
        visible = tuple(
            sorted(
                skill_id
                for skill_id, descriptor in self._by_id.items()
                if value in descriptor.allowed_stages
            )
        )
        if not visible:
            raise TeacherBatchError("teacher progressive skill stage differs")
        return MappingProxyType(
            canonical_value(
                {
                    "schema": "eva.teacher-progressive-skill-surface.v1",
                    "mode": "search-then-load",
                    "catalog_blake3": self.catalog_blake3,
                    "discovery_tools": ("search_skills", "load_skill"),
                    "initial_skill_mount_count": 0,
                    "visible_skill_count": len(visible),
                    "visible_skill_ids": visible,
                }
            )
        )


__all__ = ["ProgressiveTeacherSkillSurface"]
