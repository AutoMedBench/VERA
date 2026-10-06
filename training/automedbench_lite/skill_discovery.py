"""Prospective public discovery metadata; never skill-body injection or usage.

Call with the already verified ``VerifiedEvaluationSkills`` used by the host.
Only the existing read-only search handler runs here. The actor must still
perform a real canonical load_skill call to receive a verified skill body.
No existing catalog, schema, handler, body or phase policy is changed.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

from eva_agent.pipeline import Stage
from eva_agent.pipeline.digests import blake3_hex, canonical_value

from .skill_surface import SKILL_TOOLS


SCHEMA = "eva.automedbench-public-skill-discovery.v1"
NATIVE_BOOTSTRAP = "eva-workflow/stage-rollout"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_public_skill_discovery(skills: Any, stage: Stage | str) -> dict[str, Any]:
    """Project the verified current-stage surface into public metadata only.

    ``inventory`` excludes the native bootstrap, which the canonical search
    surface adds for all stages. A single space is present in every canonical
    ``skill_id + ' ' + description`` search string, so this valid nonempty
    query enumerates exactly the stage-visible public descriptors. This is
    host preflight, NOT an actor tool event and NOT proof of loaded skill use.
    """
    value = stage.value if isinstance(stage, Stage) else stage
    _require(value in {s.value for s in Stage}, "discovery stage differs")
    digest = skills.catalog_blake3
    _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
             "verified catalog digest differs")
    definitions = skills.definitions[value]
    for offer in SKILL_TOOLS:
        definition = definitions[offer["name"]]
        _require(definition.name == offer["name"] and definition.description == offer["description"]
                 and canonical_value(definition.input_schema) == offer["inputSchema"],
                 "canonical skill tool differs")
    inventory: dict[str, str] = {}
    for row in skills.inventory:
        if value not in row["allowed_stages"]:
            continue
        identifier, description = row["skill_id"], row["description"]
        _require(isinstance(identifier, str) and identifier and identifier not in inventory
                 and isinstance(description, str) and bool(description), "verified inventory differs")
        inventory[identifier] = description
    result = skills.call("search_skills", {"query": " ", "stage": value}, stage=value)
    _require(isinstance(result, Mapping) and set(result) == {"matches"}, "search result differs")
    matches = result["matches"]
    _require(isinstance(matches, (list, tuple)), "search matches differ")
    public = []
    seen = set()
    for row in matches:
        _require(isinstance(row, Mapping) and set(row) == {"skill_id", "description"},
                 "search descriptor differs")
        identifier, description = row["skill_id"], row["description"]
        _require(isinstance(identifier, str) and identifier and identifier not in seen
                 and isinstance(description, str) and bool(description), "search identity differs")
        _require(identifier == NATIVE_BOOTSTRAP or inventory.get(identifier) == description,
                 "search and verified inventory differ")
        seen.add(identifier)
        public.append({"skill_id": identifier, "description": description,
                       "search_arguments": {"query": identifier, "stage": value},
                       "load_arguments": {"skill_id": identifier, "stage": value}})
    _require(seen == set(inventory) | {NATIVE_BOOTSTRAP}, "stage-visible inventory differs")
    document = {"schema": SCHEMA, "stage": value, "catalog_blake3": digest,
                "origin": "verified-host-stage-search-preflight",
                "contains_skill_bodies": False, "counts_as_skill_use": False,
                "search_semantics": "casefolded-contiguous-substring-of-id-space-description",
                "visible_skill_count": len(public), "skills": sorted(public, key=lambda r: r["skill_id"])}
    return {**document, "discovery_blake3": blake3_hex(document)}


def render_public_skill_discovery(document: Mapping[str, Any]) -> str:
    """Render allowlisted metadata for a prospective per-phase prompt appendix."""
    core = dict(document)
    commitment = core.pop("discovery_blake3", None)
    _require(core.get("schema") == SCHEMA and commitment == blake3_hex(core),
             "public discovery commitment differs")
    stage = core["stage"]
    _require(stage in {s.value for s in Stage} and core["contains_skill_bodies"] is False
             and core["counts_as_skill_use"] is False, "public discovery scope differs")
    # No JSON dump of the complete source inventory: it contains local paths
    # and content identities. Only exact search-public ID/description pairs.
    descriptors = [{"skill_id": row["skill_id"], "description": row["description"]}
                   for row in core["skills"]]
    return (
        f"Public skill discovery for the CURRENT phase {stage} (metadata only; no skill is loaded).\n"
        "search_skills uses a case-insensitive contiguous substring, not semantic or multi-keyword search. "
        "Spaces and hyphens are not interchangeable. Choose a relevant exact ID below; use that entire ID "
        f"as query with stage='{stage}', then call load_skill with the returned skill_id and stage='{stage}'. "
        "Copy skill IDs exactly: do not add a tool/server namespace, shorten IDs or invent one. "
        "Changing stage to E2E does not broaden the current phase's permissions. "
        "The list is discovery metadata, not evidence of skill use; only a real successful load_skill "
        "returns the verified body. A loaded skill may mention tools absent here; the actually offered "
        "tool catalog remains authoritative.\n"
        + json.dumps(descriptors, ensure_ascii=False, separators=(",", ":"))
    )


__all__ = ["build_public_skill_discovery", "render_public_skill_discovery"]
