"""Versioned construction-readiness schedule over the immutable v1 roster.

Selection v2 never changes the 9,000-candidate set, candidate UUIDs, rubric
objects, source artifacts, all-train split, or per-cell quotas.  Within each
cell it places signed v24 ``promoted`` construction states first, ``frozen``
next, and terminal ``rejected`` last.  Ties retain the v1 outcome-blind source
identity order.  No rollout, panel, judge, review, replay, or admission outcome
is an input to this module.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from eva_agent.orchestration.contracts import CandidateJob
from eva_agent.pipeline.contracts import ModelTarget, Stage
from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes, is_blake3
from eva_agent.rubrics.registry import CompiledRubricRegistry
from eva_agent.sources import (
    ConstructionReadiness,
    LegacySupervisorV4Importer,
    SignedSupervisorV24Readiness,
)

from .plan import CampaignPlan, SOURCE_FAMILY_BY_DOMAIN, exact_6000_plan, verify_plan
from .selection import (
    CampaignSelection,
    FrozenCampaignCandidateSource,
    SelectionEntry,
    SelectionQueueRow,
    verify_campaign_selection,
)


SELECTION_V2_SCHEMA = "eva.medresearch-campaign-selection.v2"
SELECTION_V2_METHOD = "construction_readiness_then_source_identity_blake3_rank_v2"
SELECTION_V2_OPERATION_SCHEMA = "eva.medresearch-campaign-selection-operation.v2"
MAXIMUM_SELECTION_V2_BYTES = 48 * 1024 * 1024
EXPECTED_SCHEDULED_COUNT = 9_000
EXPECTED_PRIMARY_COUNT = 6_000
EXPECTED_RESERVE_COUNT = 3_000
EXPECTED_PROMOTED_PRIMARY = 1_338
EXPECTED_PROMOTED_RESERVE = 6
EXPECTED_REJECTED_RESERVE = 11
_READINESS_ORDER = {
    ConstructionReadiness.PROMOTED.value: 0,
    ConstructionReadiness.FROZEN.value: 1,
    ConstructionReadiness.REJECTED.value: 2,
}
_ENTRY_KEYS = {
    "candidate_id",
    "source_candidate_id",
    "source_family",
    "source_artifact_sha256",
    "domain",
    "stage",
    "split",
    "rubric_id",
    "rubric_blake3",
    "source_identity_rank_blake3",
    "construction_readiness",
    "readiness_proof_root_blake3",
    "selection_rank_blake3",
    "cell_rank",
    "cell_target",
    "selection_tier",
    "queue_ordinal",
}
_SELECTION_KEYS = {
    "schema",
    "selection_id",
    "plan_id",
    "plan_blake3",
    "source_registry_blake3",
    "upstream_source_registry_sha256",
    "rubric_registry_blake3",
    "base_selection_id",
    "base_selection_blake3",
    "readiness_authority",
    "selection_method",
    "selected_count",
    "scheduled_count",
    "reserve_count",
    "entries",
    "selection_blake3",
}


class CampaignSelectionV2Error(ValueError):
    """A readiness-prioritized selection failed closed."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CampaignSelectionV2Error(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CampaignSelectionV2Error(f"non-finite JSON constant is forbidden: {value}")


def _required_text(value: Any, *, label: str, maximum: int = 1024) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise CampaignSelectionV2Error(f"{label} must be bounded non-empty text")
    return value


def _positive_integer(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 1:
        raise CampaignSelectionV2Error(f"{label} must be a positive integer")
    return value


def _canonical_uuid(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise CampaignSelectionV2Error(f"{label} must be a UUID")
    try:
        parsed = UUID(value)
    except ValueError:
        raise CampaignSelectionV2Error(f"{label} must be a UUID") from None
    if str(parsed) != value:
        raise CampaignSelectionV2Error(f"{label} must use canonical UUID text")
    return value


def _exact_mapping(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise CampaignSelectionV2Error(f"{label} keys differ")
    return value


def _selection_rank(
    *,
    base_selection_rank_blake3: str,
    construction_readiness: str,
    readiness_proof_root_blake3: str,
    readiness_authority_blake3: str,
) -> str:
    return blake3_hex(
        {
            "schema": "eva.construction-readiness-selection-rank.v2",
            "source_identity_rank_blake3": base_selection_rank_blake3,
            "construction_readiness": construction_readiness,
            "readiness_proof_root_blake3": readiness_proof_root_blake3,
            "readiness_authority_blake3": readiness_authority_blake3,
        }
    )


def _selection_id(
    *, base_selection_blake3: str, readiness_authority_blake3: str
) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            (
                "eva-agent:campaign-selection-v2:"
                f"{base_selection_blake3}:{readiness_authority_blake3}"
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class SelectionEntryV2:
    candidate_id: str
    source_candidate_id: str
    source_family: str
    source_artifact_sha256: str
    domain: str
    stage: str
    split: str
    rubric_id: str
    rubric_blake3: str
    source_identity_rank_blake3: str
    construction_readiness: str
    readiness_proof_root_blake3: str
    selection_rank_blake3: str
    cell_rank: int
    cell_target: int
    selection_tier: str
    queue_ordinal: int

    @property
    def base_fields(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_candidate_id": self.source_candidate_id,
            "source_family": self.source_family,
            "source_artifact_sha256": self.source_artifact_sha256,
            "domain": self.domain,
            "stage": self.stage,
            "split": self.split,
            "rubric_id": self.rubric_id,
            "rubric_blake3": self.rubric_blake3,
        }

    def to_document(self) -> dict[str, Any]:
        return {
            **self.base_fields,
            "source_identity_rank_blake3": self.source_identity_rank_blake3,
            "construction_readiness": self.construction_readiness,
            "readiness_proof_root_blake3": self.readiness_proof_root_blake3,
            "selection_rank_blake3": self.selection_rank_blake3,
            "cell_rank": self.cell_rank,
            "cell_target": self.cell_target,
            "selection_tier": self.selection_tier,
            "queue_ordinal": self.queue_ordinal,
        }

    @classmethod
    def from_document(cls, value: Any) -> "SelectionEntryV2":
        row = _exact_mapping(value, _ENTRY_KEYS, label="selection-v2 entry")
        text_keys = _ENTRY_KEYS - {"cell_rank", "cell_target", "queue_ordinal"}
        fields: dict[str, Any] = {
            key: _required_text(row[key], label=key) for key in text_keys
        }
        fields["cell_rank"] = _positive_integer(row["cell_rank"], label="cell_rank")
        fields["cell_target"] = _positive_integer(row["cell_target"], label="cell_target")
        fields["queue_ordinal"] = _positive_integer(
            row["queue_ordinal"], label="queue_ordinal"
        )
        _canonical_uuid(fields["candidate_id"], label="candidate_id")
        _canonical_uuid(fields["rubric_id"], label="rubric_id")
        for digest in (
            fields["rubric_blake3"],
            fields["source_identity_rank_blake3"],
            fields["readiness_proof_root_blake3"],
            fields["selection_rank_blake3"],
        ):
            if not is_blake3(digest):
                raise CampaignSelectionV2Error("selection-v2 entry BLAKE3 differs")
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class CampaignSelectionV2:
    selection_id: str
    plan_id: str
    plan_blake3: str
    source_registry_blake3: str
    upstream_source_registry_sha256: str
    rubric_registry_blake3: str
    base_selection_id: str
    base_selection_blake3: str
    readiness_authority: Mapping[str, Any]
    entries: tuple[SelectionEntryV2, ...]
    selection_blake3: str

    @property
    def primary_entries(self) -> tuple[SelectionEntryV2, ...]:
        return tuple(row for row in self.entries if row.selection_tier == "primary")

    @property
    def reserve_entries(self) -> tuple[SelectionEntryV2, ...]:
        return tuple(row for row in self.entries if row.selection_tier == "reserve")

    def core_document(self) -> dict[str, Any]:
        return {
            "schema": SELECTION_V2_SCHEMA,
            "selection_id": self.selection_id,
            "plan_id": self.plan_id,
            "plan_blake3": self.plan_blake3,
            "source_registry_blake3": self.source_registry_blake3,
            "upstream_source_registry_sha256": self.upstream_source_registry_sha256,
            "rubric_registry_blake3": self.rubric_registry_blake3,
            "base_selection_id": self.base_selection_id,
            "base_selection_blake3": self.base_selection_blake3,
            "readiness_authority": dict(self.readiness_authority),
            "selection_method": SELECTION_V2_METHOD,
            "selected_count": len(self.primary_entries),
            "scheduled_count": len(self.entries),
            "reserve_count": len(self.reserve_entries),
            "entries": [entry.to_document() for entry in self.entries],
        }

    def to_document(self) -> dict[str, Any]:
        return {**self.core_document(), "selection_blake3": self.selection_blake3}

    @classmethod
    def from_document(cls, value: Any) -> "CampaignSelectionV2":
        document = _exact_mapping(value, _SELECTION_KEYS, label="campaign selection-v2")
        if document["schema"] != SELECTION_V2_SCHEMA:
            raise CampaignSelectionV2Error("campaign selection-v2 schema differs")
        if document["selection_method"] != SELECTION_V2_METHOD:
            raise CampaignSelectionV2Error("campaign selection-v2 method differs")
        raw_entries = document["entries"]
        if not isinstance(raw_entries, list):
            raise CampaignSelectionV2Error("campaign selection-v2 entries must be a list")
        entries = tuple(SelectionEntryV2.from_document(row) for row in raw_entries)
        for name, expected in (
            ("selected_count", len([row for row in entries if row.selection_tier == "primary"])),
            ("scheduled_count", len(entries)),
            ("reserve_count", len([row for row in entries if row.selection_tier == "reserve"])),
        ):
            if type(document[name]) is not int or document[name] != expected:
                raise CampaignSelectionV2Error(f"campaign selection-v2 {name} differs")
        authority = document["readiness_authority"]
        if not isinstance(authority, Mapping):
            raise CampaignSelectionV2Error("readiness authority must be an object")
        selection = cls(
            selection_id=_required_text(document["selection_id"], label="selection_id"),
            plan_id=_required_text(document["plan_id"], label="plan_id"),
            plan_blake3=_required_text(document["plan_blake3"], label="plan_blake3"),
            source_registry_blake3=_required_text(
                document["source_registry_blake3"], label="source_registry_blake3"
            ),
            upstream_source_registry_sha256=_required_text(
                document["upstream_source_registry_sha256"],
                label="upstream_source_registry_sha256",
            ),
            rubric_registry_blake3=_required_text(
                document["rubric_registry_blake3"], label="rubric_registry_blake3"
            ),
            base_selection_id=_required_text(
                document["base_selection_id"], label="base_selection_id"
            ),
            base_selection_blake3=_required_text(
                document["base_selection_blake3"], label="base_selection_blake3"
            ),
            readiness_authority=MappingProxyType(dict(authority)),
            entries=entries,
            selection_blake3=_required_text(
                document["selection_blake3"], label="selection_blake3"
            ),
        )
        _canonical_uuid(selection.selection_id, label="selection_id")
        if not all(
            is_blake3(value)
            for value in (
                selection.plan_blake3,
                selection.source_registry_blake3,
                selection.rubric_registry_blake3,
                selection.base_selection_blake3,
                selection.selection_blake3,
            )
        ):
            raise CampaignSelectionV2Error("campaign selection-v2 BLAKE3 differs")
        if blake3_hex(selection.core_document()) != selection.selection_blake3:
            raise CampaignSelectionV2Error("campaign selection-v2 content differs")
        return selection


def _entry_sort_key(
    entry: SelectionEntryV2,
) -> tuple[int, str, str]:
    return (
        _READINESS_ORDER[entry.construction_readiness],
        entry.source_identity_rank_blake3,
        entry.source_candidate_id,
    )


def _queue_sort_key(
    entry: SelectionEntryV2, *, cell_order: Mapping[tuple[str, str], int]
) -> tuple[int, int, int, str]:
    try:
        cell_index = cell_order[(entry.domain, entry.stage)]
    except KeyError:
        raise CampaignSelectionV2Error("selection-v2 queue cell differs") from None
    if entry.selection_tier == "primary":
        return (0, entry.cell_rank, cell_index, entry.source_candidate_id)
    if entry.selection_tier == "reserve":
        return (
            1,
            entry.cell_rank - entry.cell_target,
            cell_index,
            entry.source_candidate_id,
        )
    raise CampaignSelectionV2Error("selection-v2 tier differs")


def build_campaign_selection_v2(
    *,
    base_selection: CampaignSelection,
    readiness: SignedSupervisorV24Readiness,
    plan: CampaignPlan,
    rubrics: CompiledRubricRegistry,
) -> CampaignSelectionV2:
    """Reorder the same roster solely from signed construction readiness."""

    verify_campaign_selection(base_selection, plan=plan, rubrics=rubrics)
    if readiness.candidate_count != EXPECTED_SCHEDULED_COUNT:
        raise CampaignSelectionV2Error("readiness authority must contain exactly 9,000 rows")
    authority_rows = readiness.readiness_by_candidate
    base_by_source = {row.source_candidate_id: row for row in base_selection.entries}
    if set(authority_rows) != set(base_by_source):
        raise CampaignSelectionV2Error("v1 and v24 candidate sets differ")
    by_cell: dict[tuple[str, str], list[SelectionEntryV2]] = defaultdict(list)
    authority_document = readiness.to_document()
    for base in base_selection.entries:
        authority_row = authority_rows[base.source_candidate_id]
        if authority_row.source_family != base.source_family or authority_row.stage != base.stage:
            raise CampaignSelectionV2Error("v1 and v24 source identity binding differs")
        state = authority_row.readiness.value
        by_cell[(base.domain, base.stage)].append(
            SelectionEntryV2(
                candidate_id=base.candidate_id,
                source_candidate_id=base.source_candidate_id,
                source_family=base.source_family,
                source_artifact_sha256=base.source_artifact_sha256,
                domain=base.domain,
                stage=base.stage,
                split="train",
                rubric_id=base.rubric_id,
                rubric_blake3=base.rubric_blake3,
                source_identity_rank_blake3=base.selection_rank_blake3,
                construction_readiness=state,
                readiness_proof_root_blake3=authority_row.proof_root_blake3,
                selection_rank_blake3=_selection_rank(
                    base_selection_rank_blake3=base.selection_rank_blake3,
                    construction_readiness=state,
                    readiness_proof_root_blake3=authority_row.proof_root_blake3,
                    readiness_authority_blake3=readiness.authority_blake3,
                ),
                cell_rank=0,
                cell_target=base.cell_target,
                selection_tier="",
                queue_ordinal=0,
            )
        )
    entries: list[SelectionEntryV2] = []
    for cell in plan.cells:
        rows = sorted(by_cell[(cell.domain, cell.stage)], key=_entry_sort_key)
        if len(rows) < cell.train:
            raise CampaignSelectionV2Error("selection-v2 cell cannot meet its quota")
        for cell_rank, row in enumerate(rows, start=1):
            entries.append(
                replace(
                    row,
                    cell_rank=cell_rank,
                    cell_target=cell.train,
                    selection_tier="primary" if cell_rank <= cell.train else "reserve",
                )
            )
    cell_order = {
        (cell.domain, cell.stage): ordinal for ordinal, cell in enumerate(plan.cells)
    }
    entries.sort(key=lambda row: _queue_sort_key(row, cell_order=cell_order))
    entries = [
        replace(row, queue_ordinal=ordinal)
        for ordinal, row in enumerate(entries, start=1)
    ]
    provisional = CampaignSelectionV2(
        selection_id=_selection_id(
            base_selection_blake3=base_selection.selection_blake3,
            readiness_authority_blake3=readiness.authority_blake3,
        ),
        plan_id=base_selection.plan_id,
        plan_blake3=base_selection.plan_blake3,
        source_registry_blake3=base_selection.source_registry_blake3,
        upstream_source_registry_sha256=base_selection.upstream_source_registry_sha256,
        rubric_registry_blake3=base_selection.rubric_registry_blake3,
        base_selection_id=base_selection.selection_id,
        base_selection_blake3=base_selection.selection_blake3,
        readiness_authority=MappingProxyType(authority_document),
        entries=tuple(entries),
        selection_blake3="",
    )
    selection = replace(
        provisional, selection_blake3=blake3_hex(provisional.core_document())
    )
    verify_campaign_selection_v2(
        selection,
        base_selection=base_selection,
        readiness=readiness,
        plan=plan,
        rubrics=rubrics,
    )
    return selection


def verify_campaign_selection_v2(
    selection: CampaignSelectionV2,
    *,
    base_selection: CampaignSelection,
    readiness: SignedSupervisorV24Readiness,
    plan: CampaignPlan,
    rubrics: CompiledRubricRegistry,
) -> None:
    """Reopen every source, readiness, ordering, quota, and digest binding."""

    if not isinstance(selection, CampaignSelectionV2):
        raise CampaignSelectionV2Error("selection must be CampaignSelectionV2")
    verify_plan(plan)
    verify_campaign_selection(base_selection, plan=plan, rubrics=rubrics)
    if not (
        selection.plan_id == base_selection.plan_id == plan.plan_id
        and selection.plan_blake3 == base_selection.plan_blake3 == plan.plan_blake3
        and selection.source_registry_blake3 == base_selection.source_registry_blake3
        and selection.upstream_source_registry_sha256
        == base_selection.upstream_source_registry_sha256
        and selection.rubric_registry_blake3 == base_selection.rubric_registry_blake3 == rubrics.digest
        and selection.base_selection_id == base_selection.selection_id
        and selection.base_selection_blake3 == base_selection.selection_blake3
        and dict(selection.readiness_authority) == readiness.to_document()
    ):
        raise CampaignSelectionV2Error("selection-v2 outer commitments differ")
    expected_id = _selection_id(
        base_selection_blake3=base_selection.selection_blake3,
        readiness_authority_blake3=readiness.authority_blake3,
    )
    if selection.selection_id != expected_id:
        raise CampaignSelectionV2Error("selection-v2 identity differs")
    if blake3_hex(selection.core_document()) != selection.selection_blake3:
        raise CampaignSelectionV2Error("selection-v2 content BLAKE3 differs")
    if len(selection.entries) != EXPECTED_SCHEDULED_COUNT:
        raise CampaignSelectionV2Error("selection-v2 must schedule exactly 9,000")
    if len(selection.primary_entries) != EXPECTED_PRIMARY_COUNT or len(selection.reserve_entries) != EXPECTED_RESERVE_COUNT:
        raise CampaignSelectionV2Error("selection-v2 must contain 6,000 primary and 3,000 reserve")
    if len({row.candidate_id for row in selection.entries}) != EXPECTED_SCHEDULED_COUNT:
        raise CampaignSelectionV2Error("selection-v2 candidate UUIDs differ")
    if len({row.source_candidate_id for row in selection.entries}) != EXPECTED_SCHEDULED_COUNT:
        raise CampaignSelectionV2Error("selection-v2 source candidates differ")
    base_by_id = {row.candidate_id: row for row in base_selection.entries}
    readiness_by_id = readiness.readiness_by_candidate
    targets = {(cell.domain, cell.stage): cell.train for cell in plan.cells}
    by_cell: dict[tuple[str, str], list[SelectionEntryV2]] = defaultdict(list)
    for row in selection.entries:
        base = base_by_id.get(row.candidate_id)
        authority_row = readiness_by_id.get(row.source_candidate_id)
        if base is None or authority_row is None:
            raise CampaignSelectionV2Error("selection-v2 source is absent")
        expected_base_fields = {
            "candidate_id": base.candidate_id,
            "source_candidate_id": base.source_candidate_id,
            "source_family": base.source_family,
            "source_artifact_sha256": base.source_artifact_sha256,
            "domain": base.domain,
            "stage": base.stage,
            "split": "train",
            "rubric_id": base.rubric_id,
            "rubric_blake3": base.rubric_blake3,
        }
        if row.base_fields != expected_base_fields:
            raise CampaignSelectionV2Error("selection-v2 changed an immutable v1 field")
        if SOURCE_FAMILY_BY_DOMAIN.get(row.domain) != row.source_family:
            raise CampaignSelectionV2Error("selection-v2 source family differs")
        if row.stage not in {stage.value for stage in Stage} or row.split != "train":
            raise CampaignSelectionV2Error("selection-v2 is not all-train")
        if row.source_identity_rank_blake3 != base.selection_rank_blake3:
            raise CampaignSelectionV2Error("selection-v2 source identity rank differs")
        if not (
            row.construction_readiness == authority_row.readiness.value
            and row.readiness_proof_root_blake3 == authority_row.proof_root_blake3
            and row.selection_rank_blake3
            == _selection_rank(
                base_selection_rank_blake3=base.selection_rank_blake3,
                construction_readiness=authority_row.readiness.value,
                readiness_proof_root_blake3=authority_row.proof_root_blake3,
                readiness_authority_blake3=readiness.authority_blake3,
            )
        ):
            raise CampaignSelectionV2Error("selection-v2 readiness binding differs")
        target = targets.get((row.domain, row.stage))
        if target is None or row.cell_target != target:
            raise CampaignSelectionV2Error("selection-v2 cell target differs")
        if (row.cell_rank <= target) != (row.selection_tier == "primary"):
            raise CampaignSelectionV2Error("selection-v2 tier differs from cell rank")
        by_cell[(row.domain, row.stage)].append(row)
    for cell in plan.cells:
        rows = sorted(by_cell[(cell.domain, cell.stage)], key=_entry_sort_key)
        observed = sorted(by_cell[(cell.domain, cell.stage)], key=lambda row: row.cell_rank)
        if rows != observed or [row.cell_rank for row in observed] != list(range(1, len(rows) + 1)):
            raise CampaignSelectionV2Error("selection-v2 does not maximize readiness per cell")
        if sum(row.selection_tier == "primary" for row in rows) != cell.train:
            raise CampaignSelectionV2Error("selection-v2 primary cell quota differs")
    primary_states = Counter(row.construction_readiness for row in selection.primary_entries)
    reserve_states = Counter(row.construction_readiness for row in selection.reserve_entries)
    if primary_states != Counter({"promoted": EXPECTED_PROMOTED_PRIMARY, "frozen": 4_662}):
        raise CampaignSelectionV2Error("selection-v2 primary readiness counts differ")
    if reserve_states != Counter(
        {"promoted": EXPECTED_PROMOTED_RESERVE, "frozen": 2_983, "rejected": EXPECTED_REJECTED_RESERVE}
    ):
        raise CampaignSelectionV2Error("selection-v2 reserve readiness counts differ")
    if any(row.construction_readiness == "rejected" for row in selection.primary_entries):
        raise CampaignSelectionV2Error("rejected candidate entered the primary schedule")
    primary_cells = Counter((row.domain, row.stage, row.split) for row in selection.primary_entries)
    expected_cells = Counter(
        {(cell.domain, cell.stage, "train"): cell.train for cell in plan.cells}
    )
    if primary_cells != expected_cells:
        raise CampaignSelectionV2Error("selection-v2 42-cell quotas differ")
    cell_order = {
        (cell.domain, cell.stage): ordinal for ordinal, cell in enumerate(plan.cells)
    }
    canonical = sorted(
        selection.entries, key=lambda row: _queue_sort_key(row, cell_order=cell_order)
    )
    if list(selection.entries) != canonical:
        raise CampaignSelectionV2Error("selection-v2 queue order differs")
    if [row.queue_ordinal for row in selection.entries] != list(
        range(1, EXPECTED_SCHEDULED_COUNT + 1)
    ):
        raise CampaignSelectionV2Error("selection-v2 queue ordinals differ")


def campaign_selection_v2_from_document(
    document: Any,
    *,
    base_selection: CampaignSelection,
    readiness: SignedSupervisorV24Readiness,
    plan: CampaignPlan,
    rubrics: CompiledRubricRegistry,
) -> CampaignSelectionV2:
    selection = CampaignSelectionV2.from_document(document)
    verify_campaign_selection_v2(
        selection,
        base_selection=base_selection,
        readiness=readiness,
        plan=plan,
        rubrics=rubrics,
    )
    return selection


def load_campaign_selection_v2(
    path: str | Path,
    *,
    base_selection: CampaignSelection,
    readiness: SignedSupervisorV24Readiness,
    plan: CampaignPlan,
    rubrics: CompiledRubricRegistry,
    maximum_bytes: int = MAXIMUM_SELECTION_V2_BYTES,
) -> CampaignSelectionV2:
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise CampaignSelectionV2Error("maximum_bytes must be positive")
    target = Path(path)
    try:
        descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise CampaignSelectionV2Error("selection-v2 cannot be opened safely") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= maximum_bytes:
            raise CampaignSelectionV2Error("selection-v2 file size differs")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(maximum_bytes + 1)
        if len(payload) > maximum_bytes:
            raise CampaignSelectionV2Error("selection-v2 exceeds bounded size")
    finally:
        os.close(descriptor)
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except CampaignSelectionV2Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CampaignSelectionV2Error("selection-v2 is not strict UTF-8 JSON") from exc
    return campaign_selection_v2_from_document(
        document,
        base_selection=base_selection,
        readiness=readiness,
        plan=plan,
        rubrics=rubrics,
    )


def write_campaign_selection_v2_new(
    path: str | Path,
    selection: CampaignSelectionV2,
    *,
    base_selection: CampaignSelection,
    readiness: SignedSupervisorV24Readiness,
    plan: CampaignPlan,
    rubrics: CompiledRubricRegistry,
) -> None:
    verify_campaign_selection_v2(
        selection,
        base_selection=base_selection,
        readiness=readiness,
        plan=plan,
        rubrics=rubrics,
    )
    target = Path(path)
    target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    payload = canonical_json_bytes(selection.to_document())
    if len(payload) > MAXIMUM_SELECTION_V2_BYTES:
        raise CampaignSelectionV2Error("selection-v2 exceeds bounded size")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError:
        raise CampaignSelectionV2Error("selection-v2 already exists") from None
    except OSError as exc:
        raise CampaignSelectionV2Error("selection-v2 cannot be created safely") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


class FrozenCampaignCandidateSourceV2:
    """v2 queue order with the exact same source loader and runtime jobs as v1."""

    def __init__(
        self,
        *,
        selection: CampaignSelectionV2,
        base_selection: CampaignSelection,
        readiness: SignedSupervisorV24Readiness,
        importer: LegacySupervisorV4Importer,
        rubrics: CompiledRubricRegistry,
        targets: Sequence[ModelTarget],
        plan: CampaignPlan | None = None,
    ) -> None:
        selected_plan = plan or exact_6000_plan()
        verify_campaign_selection_v2(
            selection,
            base_selection=base_selection,
            readiness=readiness,
            plan=selected_plan,
            rubrics=rubrics,
        )
        self._selection = selection
        self._base = FrozenCampaignCandidateSource(
            selection=base_selection,
            importer=importer,
            rubrics=rubrics,
            targets=targets,
            plan=selected_plan,
        )
        self._entries = MappingProxyType({row.candidate_id: row for row in selection.entries})

    @property
    def selection(self) -> CampaignSelectionV2:
        """Expose the already-verified immutable selection for additive releases."""

        return self._selection

    @property
    def base_source(self) -> FrozenCampaignCandidateSource:
        """Expose the verified source loader without weakening its checks."""

        return self._base

    @property
    def candidate_count(self) -> int:
        return len(self._entries)

    scheduled_count = candidate_count

    @property
    def primary_count(self) -> int:
        return len(self._selection.primary_entries)

    @property
    def reserve_count(self) -> int:
        return len(self._selection.reserve_entries)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(row.candidate_id for row in self._selection.entries)

    @property
    def primary_candidate_ids(self) -> tuple[str, ...]:
        return tuple(row.candidate_id for row in self._selection.primary_entries)

    @property
    def reserve_candidate_ids(self) -> tuple[str, ...]:
        return tuple(row.candidate_id for row in self._selection.reserve_entries)

    @property
    def plan_blake3(self) -> str:
        return self._selection.plan_blake3

    @property
    def rubric_registry_blake3(self) -> str:
        return self._selection.rubric_registry_blake3

    @property
    def source_registry_blake3(self) -> str:
        return self._selection.source_registry_blake3

    @property
    def selection_blake3(self) -> str:
        return self._selection.selection_blake3

    def load(self, candidate_id: str) -> CandidateJob:
        if candidate_id not in self._entries:
            raise CampaignSelectionV2Error("campaign candidate is not in selection-v2")
        return self._base.load(candidate_id)

    def entry(self, candidate_id: str) -> SelectionEntryV2:
        try:
            return self._entries[candidate_id]
        except KeyError:
            raise CampaignSelectionV2Error(
                "campaign candidate is not in selection-v2"
            ) from None

    def source_candidate_id(self, candidate_id: str) -> str:
        """Return the exact v1-preserved source identity for execution binding."""

        return self.entry(candidate_id).source_candidate_id

    def _selected_entries(
        self, candidate_ids: Iterable[str] | None
    ) -> tuple[SelectionEntryV2, ...]:
        identities = self.candidate_ids if candidate_ids is None else tuple(candidate_ids)
        if len(identities) != len(set(identities)):
            raise CampaignSelectionV2Error("selection-v2 candidate request is duplicated")
        try:
            return tuple(self._entries[candidate_id] for candidate_id in identities)
        except KeyError:
            raise CampaignSelectionV2Error("campaign candidate is not in selection-v2") from None

    def queue_rows(
        self,
        candidate_ids: Iterable[str] | None = None,
        *,
        worker_width: int = 64,
    ) -> tuple[SelectionQueueRow, ...]:
        entries = self._selected_entries(candidate_ids)
        if type(worker_width) is not int or worker_width < 1:
            raise CampaignSelectionV2Error("worker_width must be positive")
        with ThreadPoolExecutor(max_workers=worker_width, thread_name_prefix="eva-v2-source") as pool:
            jobs = tuple(pool.map(lambda row: self.load(row.candidate_id), entries))
        return tuple(
            SelectionQueueRow(
                candidate_id=entry.candidate_id,
                source_episode_id=job.episode.episode_id,
                domain=entry.domain,
                stage=entry.stage,
                split=entry.split,
                rubric_blake3=entry.rubric_blake3,
                queue_ordinal=entry.queue_ordinal,
                cell_rank=entry.cell_rank,
                selection_tier=entry.selection_tier,
            )
            for entry, job in zip(entries, jobs, strict=True)
        )

    def queue_records(
        self,
        candidate_ids: Iterable[str] | None = None,
        *,
        worker_width: int = 64,
    ) -> tuple[Any, ...]:
        return tuple(
            row.to_candidate_queue_record()
            for row in self.queue_rows(candidate_ids, worker_width=worker_width)
        )


__all__ = [
    "CampaignSelectionV2",
    "CampaignSelectionV2Error",
    "EXPECTED_PRIMARY_COUNT",
    "EXPECTED_PROMOTED_PRIMARY",
    "EXPECTED_PROMOTED_RESERVE",
    "EXPECTED_REJECTED_RESERVE",
    "EXPECTED_RESERVE_COUNT",
    "EXPECTED_SCHEDULED_COUNT",
    "FrozenCampaignCandidateSourceV2",
    "MAXIMUM_SELECTION_V2_BYTES",
    "SELECTION_V2_METHOD",
    "SELECTION_V2_OPERATION_SCHEMA",
    "SELECTION_V2_SCHEMA",
    "SelectionEntryV2",
    "build_campaign_selection_v2",
    "campaign_selection_v2_from_document",
    "load_campaign_selection_v2",
    "verify_campaign_selection_v2",
    "write_campaign_selection_v2_new",
]
