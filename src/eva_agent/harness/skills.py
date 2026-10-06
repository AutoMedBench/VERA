"""Authenticated, policy-visible, on-demand skill delivery for EvaMed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from eva_agent.pipeline.digests import blake3_hex

from .contracts import HarnessContractError
from .tools import ToolDefinition, VALID_STAGES


SEARCH_SKILLS_DESCRIPTION = "Search stage-appropriate medical-research skills."
LOAD_SKILL_DESCRIPTION = "Load one stage-appropriate skill document on demand."
SKILL_STAGE_SCHEMA = {"type": "string", "enum": sorted(VALID_STAGES)}
SEARCH_SKILLS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1},
        "stage": SKILL_STAGE_SCHEMA,
    },
    "required": ["query", "stage"],
    "additionalProperties": False,
}
LOAD_SKILL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "skill_id": {"type": "string", "minLength": 1},
        "stage": SKILL_STAGE_SCHEMA,
    },
    "required": ["skill_id", "stage"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class SkillDocument:
    skill_id: str
    description: str
    content: str
    allowed_stages: frozenset[str] = VALID_STAGES

    def __post_init__(self) -> None:
        if not self.skill_id or not self.description or not self.content:
            raise HarnessContractError("skill identity, description, and content are required")
        if not self.allowed_stages or not self.allowed_stages <= VALID_STAGES:
            raise HarnessContractError("skill stages must be a non-empty S1-S5/E2E subset")


class SkillCatalog:
    def __init__(self, documents: Sequence[SkillDocument]) -> None:
        self._documents: dict[str, SkillDocument] = {}
        for document in documents:
            if document.skill_id in self._documents:
                raise HarnessContractError(f"duplicate skill: {document.skill_id}")
            self._documents[document.skill_id] = document

    def tool_definitions(self) -> tuple[ToolDefinition, ToolDefinition]:
        async def search(arguments: Mapping[str, Any]) -> dict[str, Any]:
            query = str(arguments["query"]).casefold()
            stage = str(arguments["stage"])
            matches = [
                {"skill_id": item.skill_id, "description": item.description}
                for item in self._documents.values()
                if stage in item.allowed_stages
                and query in f"{item.skill_id} {item.description}".casefold()
            ]
            return {"matches": sorted(matches, key=lambda item: item["skill_id"])}

        async def load(arguments: Mapping[str, Any]) -> dict[str, Any]:
            skill_id = str(arguments["skill_id"])
            stage = str(arguments["stage"])
            document = self._documents.get(skill_id)
            if document is None or stage not in document.allowed_stages:
                raise HarnessContractError("skill is unavailable for this stage")
            return {
                "skill_id": document.skill_id,
                "content": document.content,
                "content_blake3": blake3_hex(document.content),
                "delivery": "policy-visible-tool-observation",
            }

        search_tool = ToolDefinition(
            name="search_skills",
            description=SEARCH_SKILLS_DESCRIPTION,
            parameters=SEARCH_SKILLS_INPUT_SCHEMA,
            handler=search,
            parallel_safe=True,
        )
        load_tool = ToolDefinition(
            name="load_skill",
            description=LOAD_SKILL_DESCRIPTION,
            parameters=LOAD_SKILL_INPUT_SCHEMA,
            handler=load,
            parallel_safe=True,
        )
        return search_tool, load_tool


__all__ = [
    "LOAD_SKILL_DESCRIPTION",
    "LOAD_SKILL_INPUT_SCHEMA",
    "SEARCH_SKILLS_DESCRIPTION",
    "SEARCH_SKILLS_INPUT_SCHEMA",
    "SKILL_STAGE_SCHEMA",
    "SkillCatalog",
    "SkillDocument",
]
