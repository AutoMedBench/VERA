"""Versioned, signed-ready execution-binding inventory for all scheduled rows.

This catalog deliberately distinguishes a source/rubric binding from an
executable sandbox.  The frozen campaign contains 9,000 source identities,
but only candidates whose complete signed construction proof is present may
be placed in the executable allowlist.  Later construction waves create a new
catalog revision; they never rewrite an earlier catalog or upgrade a row from
an unsigned input alone.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from eva_agent.campaign.selection_v3 import CampaignSelectionV3
from eva_agent.pipeline.digests import blake3_hex, is_blake3
from eva_agent.rubrics.registry import CompiledRubricRegistry

from .legacy_execution import (
    CandidateBindingReference,
    LegacyExecutionBindingCatalog,
    ProductionSourceBlocker,
)


CATALOG_SCHEMA = "eva.prospective-execution-binding-catalog.v2"
CATALOG_METHOD = "frozen-source-plus-materialized-execution-proof-additive-v2"
EXPECTED_SCHEDULED = 9_000
EXPECTED_RUBRIC_TABLES = 42
EXPECTED_RUBRIC_ITEMS = 252
_STAGES = ("S1", "S2", "S3", "S4", "S5", "E2E")
_SOURCE_POLICY_TOOLS = (
    "execute_code",
    "materialize_evidence_selection",
    "materialize_plan",
    "retrieve_frozen_evidence",
    "submit_results",
)
_PRIMARY_INTERFACES = {
    "S1": (("mcp", "materialize_plan"),),
    "S2": (("mcp", "retrieve_frozen_evidence"), ("mcp", "materialize_evidence_selection")),
    "S3": (("mcp", "execute_code"), ("codex_native", "fileChange")),
    "S4": (("mcp", "execute_code"), ("codex_native", "fileChange")),
    "S5": (("mcp", "submit_results"),),
    "E2E": tuple(("mcp", name) for name in _SOURCE_POLICY_TOOLS)
    + (("codex_native", "fileChange"),),
}


class ProspectiveExecutionCatalogError(ValueError):
    """A catalog would weaken or misstate the frozen execution boundary."""


class LegacyBindingMaterializer(Protocol):
    """Runtime-equivalent validation required before executable classification."""

    def resolve(self, eva_candidate_id: str, *, source_candidate_id: str) -> Any: ...


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    return value


def _tool_profiles() -> tuple[Mapping[str, Any], ...]:
    profiles = []
    for stage in _STAGES:
        core = {
            "schema": "eva.stage-execution-tool-profile.v1",
            "stage": stage,
            "source_policy_required_tools": list(_SOURCE_POLICY_TOOLS),
            "primary_interfaces": [
                {"kind": kind, "name": name} for kind, name in _PRIMARY_INTERFACES[stage]
            ],
            "mcp_schema_rule": "canonical_input_schema_bytes_unchanged",
            "parallel_rule": "candidate_policy_and_handler_metadata_authoritative",
            "network_access": False,
        }
        profiles.append({**core, "profile_blake3": blake3_hex(core)})
    return tuple(profiles)


def _rubric_profiles(rubrics: CompiledRubricRegistry) -> tuple[Mapping[str, Any], ...]:
    profiles = tuple(
        rubric.to_document()
        for rubric in sorted(rubrics.rubrics, key=lambda row: (row.domain, row.stage))
    )
    if len(profiles) != EXPECTED_RUBRIC_TABLES or sum(len(row["items"]) for row in profiles) != EXPECTED_RUBRIC_ITEMS:
        raise ProspectiveExecutionCatalogError("authoritative 42-table/252-item rubric inventory differs")
    return profiles


def _legacy_proof(reference: CandidateBindingReference, catalog_blake3: str) -> Mapping[str, Any]:
    core = {
        "kind": "signed_v24_legacy_promoted",
        "resolver_catalog_blake3": catalog_blake3,
        "proof_root_blake3": reference.proof_root_blake3,
        "promoted_subject_blake3": reference.promoted_subject_blake3,
        "source_policy_upstream_sha256": reference.source_policy_upstream_sha256,
        "source_policy_blake3": reference.source_policy_blake3,
        "construction_manifest_upstream_sha256": reference.construction_manifest_upstream_sha256,
        "construction_manifest_blake3": reference.construction_manifest_blake3,
        "construction_attestation_blake3": reference.construction_attestation_blake3,
        "legacy_runtime_source_blake3": reference.to_document().get(
            "legacy_runtime_source_blake3"
        ),
    }
    # The runtime digest belongs to the catalog, not the row reference.
    core.pop("legacy_runtime_source_blake3")
    return {**core, "execution_proof_blake3": blake3_hex(core)}


def _execution_blocker() -> Mapping[str, Any]:
    """Describe a failed materialization without retaining exception payloads."""

    core = {
        "category": "full_binding_validation_failed",
        "validator": "LegacyExecutionBindingResolver.resolve",
        "raw_exception_recorded": False,
    }
    return {**core, "execution_blocker_blake3": blake3_hex(core)}


@dataclass(frozen=True, slots=True)
class ProspectiveExecutionBindingCatalog:
    document: Mapping[str, Any]

    @property
    def catalog_blake3(self) -> str:
        return str(self.document["catalog_blake3"])

    @property
    def executable_candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            row["candidate_id"]
            for row in self.document["rows"]
            if row["execution_status"] == "executable_legacy"
        )

    def to_document(self) -> dict[str, Any]:
        return _plain(self.document)


def build_prospective_execution_catalog(
    *,
    selection: CampaignSelectionV3,
    rubrics: CompiledRubricRegistry,
    legacy_catalog: LegacyExecutionBindingCatalog,
    legacy_binding_materializer: LegacyBindingMaterializer,
    source_import_blake3s: Mapping[str, str],
    catalog_revision: int = 1,
) -> ProspectiveExecutionBindingCatalog:
    """Build all 9,000 plan bindings without creating an executable claim."""

    if type(catalog_revision) is not int or catalog_revision < 1:
        raise ProspectiveExecutionCatalogError("catalog revision must be positive")
    if len(selection.entries) != EXPECTED_SCHEDULED:
        raise ProspectiveExecutionCatalogError("selection must contain exactly 9,000 rows")
    if selection.rubric_registry_blake3 != rubrics.digest:
        raise ProspectiveExecutionCatalogError("selection rubric registry commitment differs")
    if set(source_import_blake3s) != {row.candidate_id for row in selection.entries}:
        raise ProspectiveExecutionCatalogError("source import receipt inventory differs")
    if any(not is_blake3(value) for value in source_import_blake3s.values()):
        raise ProspectiveExecutionCatalogError("source import receipt BLAKE3 differs")
    if legacy_catalog.candidate_count < 1:
        raise ProspectiveExecutionCatalogError("legacy promoted catalog is empty")
    if not callable(getattr(legacy_binding_materializer, "resolve", None)):
        raise ProspectiveExecutionCatalogError("legacy binding materializer is unavailable")

    profiles = _tool_profiles()
    profile_by_stage = {row["stage"]: row for row in profiles}
    rubric_profiles = _rubric_profiles(rubrics)
    legacy_by_source = {row.source_candidate_id: row for row in legacy_catalog.rows}
    if len(legacy_by_source) != legacy_catalog.candidate_count:
        raise ProspectiveExecutionCatalogError("legacy promoted identity inventory differs")

    rows: list[Mapping[str, Any]] = []
    for entry in sorted(selection.entries, key=lambda row: row.queue_ordinal):
        rubric = rubrics.resolve(entry.domain, entry.stage)
        if rubric.rubric_id != entry.rubric_id or rubric.digest != entry.rubric_blake3:
            raise ProspectiveExecutionCatalogError("selection row rubric binding differs")
        reference = legacy_by_source.get(entry.source_candidate_id)
        if entry.construction_readiness == "promoted":
            if reference is None or reference.source_family != entry.source_family or reference.stage.value != entry.stage:
                raise ProspectiveExecutionCatalogError("promoted row lacks its signed legacy execution proof")
            blocker: Mapping[str, Any] | None = None
            try:
                binding = legacy_binding_materializer.resolve(
                    entry.candidate_id,
                    source_candidate_id=entry.source_candidate_id,
                )
            except ProductionSourceBlocker:
                status = "source_only"
                blocker = _execution_blocker()
            else:
                if not (
                    getattr(binding, "candidate_id", None) == entry.candidate_id
                    and getattr(binding, "source_candidate_id", None)
                    == entry.source_candidate_id
                ):
                    raise ProspectiveExecutionCatalogError(
                        "materialized legacy execution binding identity differs"
                    )
                status = "executable_legacy"
            proof: Mapping[str, Any] | None = _legacy_proof(reference, legacy_catalog.catalog_blake3)
        elif entry.construction_readiness == "frozen":
            if reference is not None:
                raise ProspectiveExecutionCatalogError("frozen row shadows an executable proof")
            status, proof, blocker = "source_only", None, None
        elif entry.construction_readiness == "rejected":
            if reference is not None:
                raise ProspectiveExecutionCatalogError("rejected row shadows an executable proof")
            status, proof, blocker = "rejected", None, None
        else:
            raise ProspectiveExecutionCatalogError("unsupported readiness state")
        tool_profile = profile_by_stage[entry.stage]
        row_core = {
            "candidate_id": entry.candidate_id,
            "source_candidate_id": entry.source_candidate_id,
            "queue_ordinal": entry.queue_ordinal,
            "selection_tier": entry.selection_tier,
            "split": entry.split,
            "source_family": entry.source_family,
            "domain": entry.domain,
            "stage": entry.stage,
            "source_artifact_sha256": entry.source_artifact_sha256,
            "source_import_blake3": source_import_blake3s[entry.candidate_id],
            "source_registry_blake3": selection.source_registry_blake3,
            "selection_entry_blake3": blake3_hex(entry.to_document()),
            "rubric_id": entry.rubric_id,
            "rubric_blake3": entry.rubric_blake3,
            "rubric_item_count": len(rubric.items),
            "tool_profile_blake3": tool_profile["profile_blake3"],
            "construction_readiness": entry.construction_readiness,
            "readiness_proof_root_blake3": entry.readiness_proof_root_blake3,
            "execution_status": status,
            "execution_proof": proof,
            "execution_blocker": blocker,
        }
        rows.append({**row_core, "row_blake3": blake3_hex(row_core)})

    status_counts = Counter(row["execution_status"] for row in rows)
    split_counts = Counter(row["split"] for row in rows if row["selection_tier"] == "primary")
    if sum(status_counts.values()) != EXPECTED_SCHEDULED:
        raise ProspectiveExecutionCatalogError("execution status counts differ")
    if split_counts != Counter(train=5_000, development=500, sealed_evaluation=500):
        raise ProspectiveExecutionCatalogError("primary split counts differ")
    core = {
        "schema": CATALOG_SCHEMA,
        "catalog_revision": catalog_revision,
        "catalog_method": CATALOG_METHOD,
        "selection_blake3": selection.selection_blake3,
        "base_selection_blake3": selection.base_selection_blake3,
        "plan_blake3": selection.plan_blake3,
        "source_registry_blake3": selection.source_registry_blake3,
        "rubric_registry_blake3": rubrics.digest,
        "ancestor_legacy_catalog_blake3": legacy_catalog.catalog_blake3,
        "legacy_runtime_source_blake3": legacy_catalog.legacy_runtime_source_blake3,
        "scheduled_count": len(rows),
        "primary_count": 6_000,
        "reserve_count": 3_000,
        "executable_count": status_counts["executable_legacy"],
        "source_only_count": status_counts["source_only"],
        "rejected_count": status_counts["rejected"],
        "primary_split_counts": dict(sorted(split_counts.items())),
        "tool_profiles": list(profiles),
        "rubric_profiles": list(rubric_profiles),
        "rows": rows,
        "controls": {
            "provider_calls": 0,
            "ledger_reads": 0,
            "ledger_writes": 0,
            "launch_reads": 0,
            "launch_writes": 0,
            "source_only_is_executable": False,
            "rejected_is_executable": False,
            "future_upgrades_require_new_signed_revision": True,
            "full_binding_validation_required": True,
            "binding_failure_demotes_to_source_only": True,
        },
    }
    document = {**core, "catalog_blake3": blake3_hex(core)}
    return ProspectiveExecutionBindingCatalog(MappingProxyType(document))


def verify_prospective_execution_catalog(
    value: Mapping[str, Any],
    *,
    selection: CampaignSelectionV3,
    rubrics: CompiledRubricRegistry,
    legacy_catalog: LegacyExecutionBindingCatalog,
    legacy_binding_materializer: LegacyBindingMaterializer,
    source_import_blake3s: Mapping[str, str],
) -> ProspectiveExecutionBindingCatalog:
    """Rebuild independently; byte-equivalent canonical content is required."""

    if not isinstance(value, Mapping) or value.get("schema") != CATALOG_SCHEMA:
        raise ProspectiveExecutionCatalogError("prospective catalog schema differs")
    revision = value.get("catalog_revision")
    expected = build_prospective_execution_catalog(
        selection=selection,
        rubrics=rubrics,
        legacy_catalog=legacy_catalog,
        legacy_binding_materializer=legacy_binding_materializer,
        source_import_blake3s=source_import_blake3s,
        catalog_revision=revision,
    )
    if _plain(value) != expected.to_document():
        raise ProspectiveExecutionCatalogError("prospective catalog derivation differs")
    return expected


__all__ = [
    "CATALOG_SCHEMA",
    "ProspectiveExecutionBindingCatalog",
    "ProspectiveExecutionCatalogError",
    "build_prospective_execution_catalog",
    "verify_prospective_execution_catalog",
]
