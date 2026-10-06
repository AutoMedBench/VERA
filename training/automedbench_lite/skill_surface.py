"""Unmodified canonical discovery/load over the existing verified skill catalog."""
from pathlib import Path

from eva_agent.codex_pipeline import VerifiedActorSkillCatalog
from eva_agent.harness.skills import (LOAD_SKILL_DESCRIPTION, LOAD_SKILL_INPUT_SCHEMA,
    SEARCH_SKILLS_DESCRIPTION, SEARCH_SKILLS_INPUT_SCHEMA)
from eva_agent.pipeline import Stage, ToolRegistry
from eva_agent.pipeline.digests import canonical_value
from eva_agent.training.progressive_skills import ProgressiveTeacherSkillSurface

ROOT = Path(__file__).resolve().parents[2]
SKILL_TOOLS = (
    {"name": "search_skills", "description": SEARCH_SKILLS_DESCRIPTION,
     "inputSchema": canonical_value(SEARCH_SKILLS_INPUT_SCHEMA)},
    {"name": "load_skill", "description": LOAD_SKILL_DESCRIPTION,
     "inputSchema": canonical_value(LOAD_SKILL_INPUT_SCHEMA)},
)


class VerifiedEvaluationSkills:
    def __init__(self, runtime_root: Path):
        runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        plugin = ROOT / "plugins/evamed-codex"
        catalog = VerifiedActorSkillCatalog(
            manifest_path=plugin / "references/legacy-skill-manifest.v1.json",
            legacy_source_root=ROOT.parent / "rlevo-med-research/harness/source/rlevo-Med-RL-data/rev-79dd2a31f5f",
            native_stage_skill_path=plugin / "skills/stage-rollout/SKILL.md",
            runtime_root=runtime_root.resolve())
        surface = ProgressiveTeacherSkillSurface(catalog)
        self.catalog_blake3 = catalog.catalog_blake3
        self.inventory = canonical_value(catalog.inventory())
        self.definitions = {stage.value: {definition.name: definition for definition in
            surface.augment_registry(ToolRegistry(()), stage).definitions()} for stage in Stage}
        for definitions in self.definitions.values():
            for offer in SKILL_TOOLS:
                definition = definitions[offer["name"]]
                if (definition.description != offer["description"] or
                        canonical_value(definition.input_schema) != offer["inputSchema"]):
                    raise ValueError("canonical skill transport schema differs")

    def call(self, name: str, arguments: dict, *, stage: str):
        # Reuse the original handlers, including exact stage enforcement and
        # byte re-verification on every load; no body is synthesized or edited.
        return self.definitions[stage][name].handler(None, arguments)
