"""Provider-free bulk RL sandbox materialization from frozen commitments.

This is an additive training-data lane.  It intentionally does not change or
reuse the production-admission ledger: a benchmark-grounded starting state is
enough to make an RL sandbox training-ready.  Rollouts and Agent Judge results
are downstream selection signals, not materialization prerequisites.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from uuid import NAMESPACE_URL, uuid5

from eva_agent.campaign.selection_v2 import CampaignSelectionV2
from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes, canonical_value
from eva_agent.rubrics.registry import CompiledRubricRegistry


BULK_RL_SANDBOX_SCHEMA = "eva.bulk-rl-sandbox.v1"
BULK_RL_CATALOG_SCHEMA = "eva.bulk-rl-sandbox-catalog.v1"
FACTORY_METHOD = "frozen-primary-all-train-benchmark-start-rubric-bound-v1"
EXPECTED_SCHEDULED = 9_000
EXPECTED_SANDBOXES = 6_000
STAGES = ("S1", "S2", "S3", "S4", "S5", "E2E")

_INSTRUCTIONS = {
    "S1": "Create a bounded, benchmark-grounded research plan and its required S1 workspace evidence.",
    "S2": "Acquire, inspect, and select benchmark-grounded evidence, preserving provenance in the workspace.",
    "S3": "Analyze the available evidence with reproducible tool use and materialize the required S3 artifacts.",
    "S4": "Synthesize the analysis into the required domain artifact with explicit evidence links and uncertainty.",
    "S5": "Verify the staged work, resolve supported defects, and submit the benchmark-native result contract.",
    "E2E": "Complete the benchmark-grounded medical-research workflow end to end, using tools and workspace evidence.",
}
_SOURCE_POLICY_TOOLS = (
    "execute_code",
    "materialize_evidence_selection",
    "materialize_plan",
    "retrieve_frozen_evidence",
    "submit_results",
)
_PRIMARY_INTERFACES = {
    "S1": ("materialize_plan",),
    "S2": ("retrieve_frozen_evidence", "materialize_evidence_selection"),
    "S3": ("execute_code", "codex_native:fileChange"),
    "S4": ("execute_code", "codex_native:fileChange"),
    "S5": ("submit_results",),
    "E2E": (*_SOURCE_POLICY_TOOLS, "codex_native:fileChange"),
}


class BulkRLSandboxFactoryError(ValueError):
    """A bulk package differs from its frozen source or reward contract."""


def _plain(value: Any) -> Any:
    return canonical_value(value)


def _upstream_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verified_frozen_sources(
    document: Mapping[str, Any],
    *,
    selection: CampaignSelectionV2,
    registry_file_blake3: str,
) -> Mapping[str, Mapping[str, Any]]:
    authority = selection.readiness_authority
    if (
        registry_file_blake3 != authority.get("registry_file_blake3")
        or document.get("schema") != "rlevo.med-research-campaign-supervisor-registry.v1"
        or document.get("campaign_id") != authority.get("campaign_id")
        or document.get("registry_revision") != authority.get("registry_revision")
        or document.get("registry_sha256") != authority.get("registry_logical_upstream_sha256")
    ):
        raise BulkRLSandboxFactoryError("frozen source registry authority differs")
    rows = document.get("candidates")
    if not isinstance(rows, list) or len(rows) != EXPECTED_SCHEDULED:
        raise BulkRLSandboxFactoryError("frozen source registry inventory differs")
    by_id: dict[str, Mapping[str, Any]] = {}
    expected_keys = {
        "candidate_id", "family", "focus", "frozen_artifact", "frozen_assertions",
        "harness", "priority", "proofs", "source_shard",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise BulkRLSandboxFactoryError("frozen source row shape differs")
        candidate_id = row.get("candidate_id")
        artifact = row.get("frozen_artifact")
        harness = row.get("harness")
        if (
            not isinstance(candidate_id, str)
            or candidate_id in by_id
            or not isinstance(artifact, dict)
            or set(artifact) != {"path", "sha256"}
            or not isinstance(artifact.get("path"), str)
            or artifact["path"].startswith("/")
            or ".." in Path(artifact["path"]).parts
            or not _upstream_sha256(artifact.get("sha256"))
            or not isinstance(row.get("frozen_assertions"), list)
            or not isinstance(harness, dict)
            or not _upstream_sha256(harness.get("contract_sha256"))
        ):
            raise BulkRLSandboxFactoryError("frozen source row content differs")
        by_id[candidate_id] = MappingProxyType(row)
    return MappingProxyType(by_id)


def _workspace_file(path: str, content: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = canonical_json_bytes(content)
    return MappingProxyType(
        {
            "path": path,
            "media_type": "application/json",
            "content": _plain(content),
            "byte_count": len(payload),
            "content_blake3": blake3_hex(content),
        }
    )


def _record(
    selected: Any,
    *,
    selection: CampaignSelectionV2,
    rubrics: CompiledRubricRegistry,
    frozen_source: Mapping[str, Any],
    frozen_source_registry_file_blake3: str,
) -> Mapping[str, Any]:
    rubric = rubrics.resolve(selected.domain, selected.stage)
    if rubric.rubric_id != selected.rubric_id or rubric.digest != selected.rubric_blake3:
        raise BulkRLSandboxFactoryError("selection/rubric binding differs")
    sandbox_id = str(
        uuid5(
            NAMESPACE_URL,
            f"eva:bulk-rl:v1:{selection.selection_blake3}:{selected.candidate_id}",
        )
    )
    episode_id = str(uuid5(NAMESPACE_URL, f"eva:bulk-rl:episode:v1:{selected.candidate_id}"))
    binding_id = str(uuid5(NAMESPACE_URL, f"eva:bulk-rl:rubric:v1:{sandbox_id}:{rubric.digest}"))
    source_binding = {
        "schema": "eva.bulk-rl-source-binding.v1",
        "candidate_id": selected.candidate_id,
        "source_candidate_id": selected.source_candidate_id,
        "source_family": selected.source_family,
        "domain": selected.domain,
        "stage": selected.stage,
        "upstream_source_artifact_sha256": selected.source_artifact_sha256,
        "authority_relative_path": frozen_source["frozen_artifact"]["path"],
        "frozen_assertions": _plain(frozen_source["frozen_assertions"]),
        "harness_contract": _plain(frozen_source["harness"]),
        "frozen_source_registry_file_blake3": frozen_source_registry_file_blake3,
        "source_registry_blake3": selection.source_registry_blake3,
        "evidence_mode": "verified_benchmark_commitment",
    }
    task_contract = {
        "schema": "eva.bulk-rl-task-contract.v1",
        "episode_id": episode_id,
        "domain": selected.domain,
        "stage": selected.stage,
        "instruction": _INSTRUCTIONS[selected.stage],
        "completion_boundary": "stage" if selected.stage != "E2E" else "end_to_end",
        "reward_rubric_id": rubric.rubric_id,
        "reward_rubric_blake3": rubric.digest,
        "private_reference_visible": False,
        "tool_profile": {
            "schema": "eva.bulk-rl-stage-tool-profile.v1",
            "source_policy_tools": list(_SOURCE_POLICY_TOOLS),
            "primary_interfaces": list(_PRIMARY_INTERFACES[selected.stage]),
            "parallel_tool_calls_allowed": True,
        },
    }
    files = (
        _workspace_file("input/source-binding.json", source_binding),
        _workspace_file("input/task-contract.json", task_contract),
    )
    workspace_core = {
        "schema": "eva.bulk-rl-workspace-initial-state.v1",
        "materialization": "inline_json_files",
        "files": [_plain(row) for row in files],
        "file_count": len(files),
        "byte_count": sum(int(row["byte_count"]) for row in files),
    }
    reward_contract = {
        "schema": "eva.bulk-rl-reward-contract.v1",
        "binding_id": binding_id,
        "sandbox_id": sandbox_id,
        "registry_id": rubrics.registry_id,
        "registry_version": rubrics.registry_version,
        "registry_digest": rubrics.digest,
        "rubric_id": rubric.rubric_id,
        "rubric_digest": rubric.digest,
        "rubric_table": rubric.to_document(),
        "benchmark_judging_and_rollout_reward_same_object": True,
    }
    record_core = {
        "schema": BULK_RL_SANDBOX_SCHEMA,
        "sandbox_id": sandbox_id,
        "episode_id": episode_id,
        "candidate_id": selected.candidate_id,
        "queue_ordinal": selected.queue_ordinal,
        "source_family": selected.source_family,
        "domain": selected.domain,
        "stage": selected.stage,
        "split": "train",
        "source_binding": source_binding,
        "workspace_initial_state": {
            **workspace_core,
            "tree_blake3": blake3_hex(workspace_core),
        },
        "reward_contract": reward_contract,
        "lineage": {
            "selection_id": selection.selection_id,
            "selection_blake3": selection.selection_blake3,
            "construction_readiness": selected.construction_readiness,
            "readiness_proof_root_blake3": selected.readiness_proof_root_blake3,
        },
        "training_policy": {
            "status": "training_ready",
            "rollout_required_for_materialization": False,
            "cascade_required_for_materialization": False,
            "agent_judge_required_for_materialization": False,
            "semantic_score_required_for_materialization": False,
            "downstream_selection_after_rollout": True,
            "production_admission_ledger_mutated": False,
        },
    }
    return MappingProxyType({**record_core, "record_blake3": blake3_hex(record_core)})


@dataclass(frozen=True, slots=True)
class BulkRLSandboxCatalog:
    document: Mapping[str, Any]

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.document["records"])

    @property
    def catalog_blake3(self) -> str:
        return str(self.document["catalog_blake3"])

    def to_document(self) -> dict[str, Any]:
        return _plain(self.document)


def build_bulk_rl_sandbox_catalog(
    *,
    selection: CampaignSelectionV2,
    rubrics: CompiledRubricRegistry,
    frozen_source_registry: Mapping[str, Any],
    frozen_source_registry_file_blake3: str,
) -> BulkRLSandboxCatalog:
    """Derive exactly 6,000 all-train sandboxes without provider execution."""

    try:
        CampaignSelectionV2.from_document(selection.to_document())
    except ValueError as exc:
        raise BulkRLSandboxFactoryError("selection content differs") from exc
    if (
        len(selection.entries) != EXPECTED_SCHEDULED
        or len(selection.primary_entries) != EXPECTED_SANDBOXES
        or len(selection.reserve_entries) != EXPECTED_SCHEDULED - EXPECTED_SANDBOXES
        or selection.rubric_registry_blake3 != rubrics.digest
        or any(row.split != "train" for row in selection.entries)
    ):
        raise BulkRLSandboxFactoryError("selection population differs")
    frozen_by_id = _verified_frozen_sources(
        frozen_source_registry,
        selection=selection,
        registry_file_blake3=frozen_source_registry_file_blake3,
    )
    records: list[Mapping[str, Any]] = []
    for selected in sorted(selection.primary_entries, key=lambda row: row.queue_ordinal):
        frozen = frozen_by_id.get(selected.source_candidate_id)
        if frozen is None:
            raise BulkRLSandboxFactoryError("selected source is absent from frozen registry")
        if (
            not _upstream_sha256(selected.source_artifact_sha256)
            or selected.selection_tier != "primary"
            or selected.split != "train"
            or selected.stage not in STAGES
            or frozen["family"] != selected.source_family
            or frozen["focus"] != selected.stage
            or frozen["frozen_artifact"]["sha256"] != selected.source_artifact_sha256
        ):
            raise BulkRLSandboxFactoryError("selected source commitment differs")
        records.append(
            _record(
                selected,
                selection=selection,
                rubrics=rubrics,
                frozen_source=frozen,
                frozen_source_registry_file_blake3=frozen_source_registry_file_blake3,
            )
        )
    if len(records) != EXPECTED_SANDBOXES or len({row["sandbox_id"] for row in records}) != EXPECTED_SANDBOXES:
        raise BulkRLSandboxFactoryError("bulk sandbox identity inventory differs")
    if len({row["candidate_id"] for row in records}) != EXPECTED_SANDBOXES:
        raise BulkRLSandboxFactoryError("bulk source identity inventory differs")

    core = {
        "schema": BULK_RL_CATALOG_SCHEMA,
        "factory_method": FACTORY_METHOD,
        "selection_id": selection.selection_id,
        "selection_blake3": selection.selection_blake3,
        "source_registry_blake3": selection.source_registry_blake3,
        "frozen_source_registry_file_blake3": frozen_source_registry_file_blake3,
        "rubric_registry_blake3": rubrics.digest,
        "scheduled_source_count": EXPECTED_SCHEDULED,
        "sandbox_count": EXPECTED_SANDBOXES,
        "split_counts": {"train": EXPECTED_SANDBOXES},
        "stage_counts": dict(sorted(Counter(row["stage"] for row in records).items())),
        "domain_counts": dict(sorted(Counter(row["domain"] for row in records).items())),
        "source_family_counts": dict(sorted(Counter(row["source_family"] for row in records).items())),
        "records": [_plain(row) for row in records],
        "controls": {
            "provider_calls": 0,
            "all_training": True,
            "source_only_fallback_allowed": True,
            "rollout_required": False,
            "cascade_required": False,
            "agent_judge_required": False,
            "semantic_score_required": False,
            "production_admission_ledger_writes": 0,
            "legacy_artifacts_mutated": False,
            "post_materialization_selection_supported": True,
        },
    }
    return BulkRLSandboxCatalog(
        MappingProxyType({**core, "catalog_blake3": blake3_hex(core)})
    )


def verify_bulk_rl_sandbox_catalog(
    value: Mapping[str, Any],
    *,
    selection: CampaignSelectionV2,
    rubrics: CompiledRubricRegistry,
    frozen_source_registry: Mapping[str, Any],
    frozen_source_registry_file_blake3: str,
) -> BulkRLSandboxCatalog:
    """Independently rebuild the complete package and require exact bytes."""

    if not isinstance(value, Mapping) or value.get("schema") != BULK_RL_CATALOG_SCHEMA:
        raise BulkRLSandboxFactoryError("bulk RL catalog schema differs")
    expected = build_bulk_rl_sandbox_catalog(
        selection=selection,
        rubrics=rubrics,
        frozen_source_registry=frozen_source_registry,
        frozen_source_registry_file_blake3=frozen_source_registry_file_blake3,
    )
    if canonical_value(value) != canonical_value(expected.to_document()):
        raise BulkRLSandboxFactoryError("bulk RL catalog derivation differs")
    return expected


__all__ = [
    "BULK_RL_CATALOG_SCHEMA",
    "BULK_RL_SANDBOX_SCHEMA",
    "BulkRLSandboxCatalog",
    "BulkRLSandboxFactoryError",
    "build_bulk_rl_sandbox_catalog",
    "verify_bulk_rl_sandbox_catalog",
]
