"""Outcome-blind 9,000-candidate schedule for exactly 6,000 admissions.

The selector deliberately ranks source identities, never construction or model
outcomes.  AutoMedBench's native track is the only value that must be opened
from a frozen artifact before quota allocation; the existing legacy importer
verifies that artifact and its evidence before returning the mapping.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from eva_agent.orchestration.contracts import CandidateJob
from eva_agent.pipeline.contracts import Cohort, ModelTarget, Stage
from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes, is_blake3
from eva_agent.rubrics.registry import CompiledRubricRegistry
from eva_agent.sources import (
    ImportedLegacyEpisode,
    LegacyCandidateRef,
    LegacySupervisorV4Importer,
    RubricAvailability,
)

from .ledger import FROZEN_SCHEDULE_SIZE
from .plan import (
    CampaignPlan,
    SOURCE_FAMILY_BY_DOMAIN,
    exact_6000_plan,
    verify_plan,
)

if TYPE_CHECKING:
    from .ledger import CandidateQueueRecord


SELECTION_SCHEMA = "eva.medresearch-campaign-selection.v1"
SELECTION_METHOD = "outcome_blind_source_identity_blake3_rank_v1"
SCHEDULED_CANDIDATE_COUNT = FROZEN_SCHEDULE_SIZE
MAXIMUM_SELECTION_BYTES = 32 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
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
    "selection_method",
    "selected_count",
    "scheduled_count",
    "reserve_count",
    "entries",
    "selection_blake3",
}


class CampaignSelectionError(ValueError):
    """A frozen selection or selected source failed closed."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CampaignSelectionError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CampaignSelectionError(f"non-finite JSON constant is forbidden: {value}")


def _exact_mapping(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise CampaignSelectionError(f"{label} keys differ")
    return value


def _required_text(value: Any, *, label: str, maximum: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise CampaignSelectionError(f"{label} must be bounded non-empty text")
    return value


def _source_id(value: Any) -> str:
    if not isinstance(value, str) or _SOURCE_ID.fullmatch(value) is None:
        raise CampaignSelectionError("source_candidate_id differs")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CampaignSelectionError(f"{label} must be lowercase SHA-256")
    return value


def _positive_integer(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 1:
        raise CampaignSelectionError(f"{label} must be a positive integer")
    return value


def _canonical_uuid(value: str, *, label: str) -> str:
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError):
        raise CampaignSelectionError(f"{label} must be a UUID") from None
    if str(parsed) != value:
        raise CampaignSelectionError(f"{label} must use canonical UUID text")
    return value


def _campaign_candidate_id(plan_blake3: str, source_candidate_id: str) -> str:
    namespace = uuid5(NAMESPACE_URL, f"eva-agent:selection:{plan_blake3}")
    return str(uuid5(namespace, source_candidate_id))


def _selection_rank(
    *,
    plan_blake3: str,
    rubric_registry_blake3: str,
    source_candidate_id: str,
    source_family: str,
    source_stage: str,
    source_artifact_sha256: str,
) -> str:
    """Rank only immutable source identity fields, never supervisor outcomes.

    The registry content digest is committed by the outer manifest but is
    intentionally absent here: supervisor state/proof fields can encode old
    outcomes, and even a common outcome-derived hash salt could change the
    relative ordering.  The fields below are registry-issued source identity
    commitments and are sufficient to make the ordering stable and auditable.
    """

    return blake3_hex(
        {
            "schema": "eva.outcome-blind-source-identity-rank.v1",
            "plan_blake3": plan_blake3,
            "rubric_registry_blake3": rubric_registry_blake3,
            "source_candidate_id": source_candidate_id,
            "source_family": source_family,
            "source_stage": source_stage,
            "source_artifact_sha256": source_artifact_sha256,
        }
    )


def _reference_rank(
    reference: LegacyCandidateRef,
    *,
    plan_blake3: str,
    rubric_registry_blake3: str,
) -> str:
    return _selection_rank(
        plan_blake3=plan_blake3,
        rubric_registry_blake3=rubric_registry_blake3,
        source_candidate_id=reference.candidate_id,
        source_family=reference.family,
        source_stage=reference.stage.value,
        source_artifact_sha256=reference.upstream_artifact_sha256,
    )


def _entry_rank(
    entry: "SelectionEntry",
    *,
    plan_blake3: str,
    rubric_registry_blake3: str,
) -> str:
    return _selection_rank(
        plan_blake3=plan_blake3,
        rubric_registry_blake3=rubric_registry_blake3,
        source_candidate_id=entry.source_candidate_id,
        source_family=entry.source_family,
        source_stage=entry.stage,
        source_artifact_sha256=entry.source_artifact_sha256,
    )


def _selection_id(
    *, plan_blake3: str, source_registry_blake3: str, rubric_registry_blake3: str
) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            (
                "eva-agent:campaign-selection:"
                f"{plan_blake3}:{source_registry_blake3}:{rubric_registry_blake3}"
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class SelectionEntry:
    candidate_id: str
    source_candidate_id: str
    source_family: str
    source_artifact_sha256: str
    domain: str
    stage: str
    split: str
    rubric_id: str
    rubric_blake3: str
    selection_rank_blake3: str
    cell_rank: int
    cell_target: int
    selection_tier: str
    queue_ordinal: int

    def to_document(self) -> dict[str, Any]:
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
            "selection_rank_blake3": self.selection_rank_blake3,
            "cell_rank": self.cell_rank,
            "cell_target": self.cell_target,
            "selection_tier": self.selection_tier,
            "queue_ordinal": self.queue_ordinal,
        }

    @classmethod
    def from_document(cls, value: Any) -> "SelectionEntry":
        row = _exact_mapping(value, _ENTRY_KEYS, label="selection entry")
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
        _source_id(fields["source_candidate_id"])
        _sha256(fields["source_artifact_sha256"], label="source_artifact_sha256")
        if not is_blake3(fields["rubric_blake3"]):
            raise CampaignSelectionError("rubric_blake3 differs")
        if not is_blake3(fields["selection_rank_blake3"]):
            raise CampaignSelectionError("selection_rank_blake3 differs")
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class CampaignSelection:
    selection_id: str
    plan_id: str
    plan_blake3: str
    source_registry_blake3: str
    upstream_source_registry_sha256: str
    rubric_registry_blake3: str
    entries: tuple[SelectionEntry, ...]
    selection_blake3: str

    @property
    def primary_entries(self) -> tuple[SelectionEntry, ...]:
        return tuple(row for row in self.entries if row.selection_tier == "primary")

    @property
    def reserve_entries(self) -> tuple[SelectionEntry, ...]:
        return tuple(row for row in self.entries if row.selection_tier == "reserve")

    def core_document(self) -> dict[str, Any]:
        return {
            "schema": SELECTION_SCHEMA,
            "selection_id": self.selection_id,
            "plan_id": self.plan_id,
            "plan_blake3": self.plan_blake3,
            "source_registry_blake3": self.source_registry_blake3,
            "upstream_source_registry_sha256": self.upstream_source_registry_sha256,
            "rubric_registry_blake3": self.rubric_registry_blake3,
            "selection_method": SELECTION_METHOD,
            "selected_count": len(self.primary_entries),
            "scheduled_count": len(self.entries),
            "reserve_count": len(self.reserve_entries),
            "entries": [entry.to_document() for entry in self.entries],
        }

    def to_document(self) -> dict[str, Any]:
        return {**self.core_document(), "selection_blake3": self.selection_blake3}

    @classmethod
    def from_document(cls, value: Any) -> "CampaignSelection":
        document = _exact_mapping(value, _SELECTION_KEYS, label="campaign selection")
        if document["schema"] != SELECTION_SCHEMA:
            raise CampaignSelectionError("campaign selection schema differs")
        if document["selection_method"] != SELECTION_METHOD:
            raise CampaignSelectionError("campaign selection method differs")
        raw_entries = document["entries"]
        if not isinstance(raw_entries, list):
            raise CampaignSelectionError("campaign selection entries must be a list")
        entries = tuple(SelectionEntry.from_document(row) for row in raw_entries)
        selected_count = document["selected_count"]
        scheduled_count = document["scheduled_count"]
        reserve_count = document["reserve_count"]
        if (
            type(selected_count) is not int
            or type(scheduled_count) is not int
            or type(reserve_count) is not int
            or selected_count != sum(row.selection_tier == "primary" for row in entries)
            or reserve_count != sum(row.selection_tier == "reserve" for row in entries)
            or scheduled_count != len(entries)
            or selected_count + reserve_count != scheduled_count
        ):
            raise CampaignSelectionError("campaign selection counts differ")
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
            entries=entries,
            selection_blake3=_required_text(
                document["selection_blake3"], label="selection_blake3"
            ),
        )
        _canonical_uuid(selection.selection_id, label="selection_id")
        _sha256(
            selection.upstream_source_registry_sha256,
            label="upstream_source_registry_sha256",
        )
        if not all(
            is_blake3(value)
            for value in (
                selection.plan_blake3,
                selection.source_registry_blake3,
                selection.rubric_registry_blake3,
                selection.selection_blake3,
            )
        ):
            raise CampaignSelectionError("campaign selection BLAKE3 commitment differs")
        if blake3_hex(selection.core_document()) != selection.selection_blake3:
            raise CampaignSelectionError("selection content BLAKE3 verification failed")
        return selection


def _direct_domain(reference: LegacyCandidateRef) -> str:
    mapping = {
        "agentclinic": "agentclinic",
        "medxpertqa": "medxpertqa",
        "healthbench-professional": "healthbench-professional",
    }
    try:
        return mapping[reference.family]
    except KeyError:
        raise CampaignSelectionError(
            "AutoMedBench domain requires a verified source artifact inspection"
        ) from None


def _auto_domains(
    importer: LegacySupervisorV4Importer,
    references: Sequence[LegacyCandidateRef],
    *,
    worker_width: int,
) -> dict[str, ImportedLegacyEpisode]:
    if type(worker_width) is not int or worker_width < 1:
        raise CampaignSelectionError("worker_width must be a positive integer")
    with ThreadPoolExecutor(max_workers=worker_width, thread_name_prefix="eva-select") as pool:
        imported = tuple(pool.map(importer.import_candidate, references))
    result = {row.manifest.candidate_id: row for row in imported}
    expected_ids = {row.candidate_id for row in references}
    if len(result) != len(references) or set(result) != expected_ids:
        raise CampaignSelectionError("AutoMedBench source identities are duplicated")
    return result


def _queue_sort_key(
    entry: SelectionEntry,
    *,
    cell_order: Mapping[tuple[str, str], int],
) -> tuple[int, int, int, str]:
    try:
        cell_index = cell_order[(entry.domain, entry.stage)]
    except KeyError:
        raise CampaignSelectionError("selection queue contains an unknown cell") from None
    if entry.selection_tier == "primary":
        return (0, entry.cell_rank, cell_index, entry.source_candidate_id)
    if entry.selection_tier == "reserve":
        return (
            1,
            entry.cell_rank - entry.cell_target,
            cell_index,
            entry.source_candidate_id,
        )
    raise CampaignSelectionError("selection tier differs")


def build_campaign_selection(
    *,
    importer: LegacySupervisorV4Importer,
    rubrics: CompiledRubricRegistry,
    plan: CampaignPlan | None = None,
    worker_width: int = 64,
) -> CampaignSelection:
    """Rank all 9,000 sources; mark exactly 6,000 primary without using outcomes."""

    selected_plan = plan or exact_6000_plan()
    verify_plan(selected_plan)
    if (
        type(importer.candidate_count) is not int
        or importer.candidate_count != SCHEDULED_CANDIDATE_COUNT
    ):
        raise CampaignSelectionError("source registry must contain exactly 9,000 candidates")
    references = tuple(importer.iter_candidates())
    if len(references) != importer.candidate_count:
        raise CampaignSelectionError("source registry enumeration differs")
    auto_refs = tuple(row for row in references if row.family == "automedbench")
    auto_imports = _auto_domains(importer, auto_refs, worker_width=worker_width)

    by_cell: dict[tuple[str, str], list[LegacyCandidateRef]] = defaultdict(list)
    for reference in references:
        if reference.family == "automedbench":
            imported = auto_imports[reference.candidate_id]
            manifest = imported.manifest
            if (
                manifest.candidate_id != reference.candidate_id
                or manifest.source_family != reference.family
                or manifest.stage != reference.stage
                or manifest.upstream_artifact_sha256
                != reference.upstream_artifact_sha256
                or manifest.registry_file_blake3 != importer.registry_file_blake3
                or manifest.upstream_registry_sha256 != importer.upstream_registry_sha256
                or imported.episode.episode_id != manifest.episode_id
                or imported.episode.stage != reference.stage
                or manifest.rubric_availability is not RubricAvailability.READY
            ):
                raise CampaignSelectionError("AutoMedBench verified source mapping differs")
            domain = manifest.rubric_domain
            if imported.episode.domain != domain:
                raise CampaignSelectionError("AutoMedBench episode domain differs")
        else:
            domain = _direct_domain(reference)
        if SOURCE_FAMILY_BY_DOMAIN.get(domain) != reference.family:
            raise CampaignSelectionError("source family and rubric domain differ")
        by_cell[(domain, reference.stage.value)].append(reference)

    entries: list[SelectionEntry] = []
    for cell in selected_plan.cells:
        quota = cell.train
        candidates = by_cell.get((cell.domain, cell.stage), [])
        ranked = sorted(
            candidates,
            key=lambda row: (
                _reference_rank(
                    row,
                    plan_blake3=selected_plan.plan_blake3,
                    rubric_registry_blake3=rubrics.digest,
                ),
                row.candidate_id,
            ),
        )
        if len(ranked) < quota:
            raise CampaignSelectionError(
                f"insufficient frozen candidates for {cell.domain} × {cell.stage}"
            )
        rubric = rubrics.resolve(cell.domain, cell.stage)
        for cell_rank, reference in enumerate(ranked, start=1):
            rank = _reference_rank(
                reference,
                plan_blake3=selected_plan.plan_blake3,
                rubric_registry_blake3=rubrics.digest,
            )
            entries.append(
                SelectionEntry(
                    candidate_id=_campaign_candidate_id(
                        selected_plan.plan_blake3, reference.candidate_id
                    ),
                    source_candidate_id=reference.candidate_id,
                    source_family=reference.family,
                    source_artifact_sha256=reference.upstream_artifact_sha256,
                    domain=cell.domain,
                    stage=cell.stage,
                    split="train",
                    rubric_id=rubric.rubric_id,
                    rubric_blake3=rubric.digest,
                    selection_rank_blake3=rank,
                    cell_rank=cell_rank,
                    cell_target=quota,
                    selection_tier="primary" if cell_rank <= quota else "reserve",
                    queue_ordinal=0,
                )
            )
    if len(entries) != len(references):
        raise CampaignSelectionError("frozen registry candidates were not scheduled exactly once")
    cell_order = {
        (cell.domain, cell.stage): index
        for index, cell in enumerate(selected_plan.cells)
    }
    entries.sort(key=lambda row: _queue_sort_key(row, cell_order=cell_order))
    entries = [
        replace(entry, queue_ordinal=ordinal)
        for ordinal, entry in enumerate(entries, start=1)
    ]
    selection_id = _selection_id(
        plan_blake3=selected_plan.plan_blake3,
        source_registry_blake3=importer.registry_file_blake3,
        rubric_registry_blake3=rubrics.digest,
    )
    provisional = CampaignSelection(
        selection_id=selection_id,
        plan_id=selected_plan.plan_id,
        plan_blake3=selected_plan.plan_blake3,
        source_registry_blake3=importer.registry_file_blake3,
        upstream_source_registry_sha256=importer.upstream_registry_sha256,
        rubric_registry_blake3=rubrics.digest,
        entries=tuple(entries),
        selection_blake3="",
    )
    selection = replace(
        provisional, selection_blake3=blake3_hex(provisional.core_document())
    )
    verify_campaign_selection(selection, plan=selected_plan, rubrics=rubrics)
    return selection


def verify_campaign_selection(
    selection: CampaignSelection,
    *,
    plan: CampaignPlan,
    rubrics: CompiledRubricRegistry,
) -> None:
    """Reopen every quota and content commitment without source-side mutation."""

    if not isinstance(selection, CampaignSelection):
        raise CampaignSelectionError("selection must be a CampaignSelection")
    verify_plan(plan)
    _canonical_uuid(selection.selection_id, label="selection_id")
    if selection.plan_id != plan.plan_id or selection.plan_blake3 != plan.plan_blake3:
        raise CampaignSelectionError("selection campaign plan commitment differs")
    if selection.rubric_registry_blake3 != rubrics.digest:
        raise CampaignSelectionError("selection rubric registry commitment differs")
    _sha256(
        selection.upstream_source_registry_sha256,
        label="upstream_source_registry_sha256",
    )
    if not all(
        is_blake3(value)
        for value in (
            selection.plan_blake3,
            selection.source_registry_blake3,
            selection.rubric_registry_blake3,
            selection.selection_blake3,
        )
    ):
        raise CampaignSelectionError("selection BLAKE3 commitment differs")
    if blake3_hex(selection.core_document()) != selection.selection_blake3:
        raise CampaignSelectionError("selection content BLAKE3 verification failed")
    expected_selection_id = _selection_id(
        plan_blake3=plan.plan_blake3,
        source_registry_blake3=selection.source_registry_blake3,
        rubric_registry_blake3=rubrics.digest,
    )
    if selection.selection_id != expected_selection_id:
        raise CampaignSelectionError("selection deterministic identity differs")
    if len(selection.entries) != SCHEDULED_CANDIDATE_COUNT:
        raise CampaignSelectionError("selection must schedule exactly 9,000 entries")
    if len({row.candidate_id for row in selection.entries}) != len(selection.entries):
        raise CampaignSelectionError("campaign candidate UUID is duplicated")
    if len({row.source_candidate_id for row in selection.entries}) != len(selection.entries):
        raise CampaignSelectionError("frozen source candidate is duplicated")
    primary = selection.primary_entries
    reserves = selection.reserve_entries
    if len(primary) != plan.total or len(reserves) != SCHEDULED_CANDIDATE_COUNT - plan.total:
        raise CampaignSelectionError("selection must contain 6,000 primary and 3,000 reserve entries")
    if Counter(row.source_family for row in selection.entries) != Counter(
        {family: 2_250 for family in set(SOURCE_FAMILY_BY_DOMAIN.values())}
    ):
        raise CampaignSelectionError("scheduled source-family balance differs")
    observed = Counter((row.domain, row.stage, row.split) for row in primary)
    expected = Counter(
        {
            (cell.domain, cell.stage, "train"): cell.train
            for cell in plan.cells
        }
    )
    if observed != expected:
        raise CampaignSelectionError("selection cell quotas differ from the campaign plan")
    targets = {(cell.domain, cell.stage): cell.train for cell in plan.cells}
    by_cell: dict[tuple[str, str], list[SelectionEntry]] = defaultdict(list)
    for entry in selection.entries:
        _canonical_uuid(entry.candidate_id, label="candidate_id")
        _canonical_uuid(entry.rubric_id, label="rubric_id")
        _source_id(entry.source_candidate_id)
        _sha256(entry.source_artifact_sha256, label="source_artifact_sha256")
        if not is_blake3(entry.selection_rank_blake3) or not is_blake3(
            entry.rubric_blake3
        ):
            raise CampaignSelectionError("selection entry BLAKE3 commitment differs")
        if SOURCE_FAMILY_BY_DOMAIN.get(entry.domain) != entry.source_family:
            raise CampaignSelectionError("selection source family differs")
        if entry.stage not in {stage.value for stage in Stage} or entry.split != "train":
            raise CampaignSelectionError("selection stage or split differs")
        target = targets.get((entry.domain, entry.stage))
        if target is None or entry.cell_target != target:
            raise CampaignSelectionError("selection cell target differs")
        if entry.selection_tier not in {"primary", "reserve"}:
            raise CampaignSelectionError("selection tier differs")
        if (entry.cell_rank <= target) != (entry.selection_tier == "primary"):
            raise CampaignSelectionError("selection tier does not follow its cell rank")
        if type(entry.queue_ordinal) is not int or entry.queue_ordinal < 1:
            raise CampaignSelectionError("selection queue ordinal differs")
        rubric = rubrics.resolve(entry.domain, entry.stage)
        if rubric.rubric_id != entry.rubric_id or rubric.digest != entry.rubric_blake3:
            raise CampaignSelectionError("selection rubric binding differs")
        expected_id = _campaign_candidate_id(plan.plan_blake3, entry.source_candidate_id)
        if entry.candidate_id != expected_id:
            raise CampaignSelectionError("campaign candidate identity differs")
        expected_rank = _entry_rank(
            entry,
            plan_blake3=plan.plan_blake3,
            rubric_registry_blake3=rubrics.digest,
        )
        if entry.selection_rank_blake3 != expected_rank:
            raise CampaignSelectionError("outcome-blind selection rank differs")
        by_cell[(entry.domain, entry.stage)].append(entry)
    for cell in plan.cells:
        rows = sorted(
            by_cell[(cell.domain, cell.stage)],
            key=lambda row: (row.selection_rank_blake3, row.source_candidate_id),
        )
        if len(rows) < cell.train:
            raise CampaignSelectionError("scheduled cell cannot meet its admission target")
        if [row.cell_rank for row in rows] != list(range(1, len(rows) + 1)):
            raise CampaignSelectionError("selection cell ranks are not contiguous")
    cell_order = {
        (cell.domain, cell.stage): index for index, cell in enumerate(plan.cells)
    }
    expected_order = sorted(
        selection.entries,
        key=lambda row: _queue_sort_key(row, cell_order=cell_order),
    )
    if list(selection.entries) != expected_order:
        raise CampaignSelectionError("selection entries are not in canonical queue order")
    if [row.queue_ordinal for row in selection.entries] != list(
        range(1, SCHEDULED_CANDIDATE_COUNT + 1)
    ):
        raise CampaignSelectionError("selection queue ordinals are not contiguous")
    if any(row.selection_tier != "primary" for row in selection.entries[: plan.total]):
        raise CampaignSelectionError("a reserve precedes the complete primary schedule")
    if any(row.selection_tier != "reserve" for row in selection.entries[plan.total :]):
        raise CampaignSelectionError("primary schedule extends into reserve ordinals")


def campaign_selection_from_document(
    document: Any,
    *,
    plan: CampaignPlan,
    rubrics: CompiledRubricRegistry,
) -> CampaignSelection:
    """Strictly parse and reopen a selection document."""

    selection = CampaignSelection.from_document(document)
    verify_campaign_selection(selection, plan=plan, rubrics=rubrics)
    return selection


def load_campaign_selection(
    path: str | Path,
    *,
    plan: CampaignPlan,
    rubrics: CompiledRubricRegistry,
    maximum_bytes: int = MAXIMUM_SELECTION_BYTES,
) -> CampaignSelection:
    """Open a bounded regular JSON file without following a final symlink."""

    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise CampaignSelectionError("maximum_bytes must be a positive integer")
    target = Path(path)
    try:
        descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise CampaignSelectionError("selection manifest cannot be opened safely") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CampaignSelectionError("selection manifest must be a regular file")
        if metadata.st_size < 1 or metadata.st_size > maximum_bytes:
            raise CampaignSelectionError("selection manifest size differs")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(maximum_bytes + 1)
        if len(payload) > maximum_bytes:
            raise CampaignSelectionError("selection manifest exceeds the bounded size")
    finally:
        os.close(descriptor)
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except CampaignSelectionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CampaignSelectionError("selection manifest is not strict UTF-8 JSON") from exc
    return campaign_selection_from_document(document, plan=plan, rubrics=rubrics)


def write_campaign_selection_new(
    path: str | Path,
    selection: CampaignSelection,
    *,
    plan: CampaignPlan,
    rubrics: CompiledRubricRegistry,
) -> None:
    """Create one canonical manifest with O_EXCL; an existing path is immutable."""

    verify_campaign_selection(selection, plan=plan, rubrics=rubrics)
    target = Path(path)
    target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    payload = canonical_json_bytes(selection.to_document())
    if len(payload) > MAXIMUM_SELECTION_BYTES:
        raise CampaignSelectionError("selection manifest exceeds the bounded size")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError:
        raise CampaignSelectionError("selection manifest already exists") from None
    except OSError as exc:
        raise CampaignSelectionError("selection manifest cannot be created safely") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _selection_source_index(
    selection: CampaignSelection,
    importer: LegacySupervisorV4Importer,
) -> Mapping[str, LegacyCandidateRef]:
    wanted = tuple(entry.source_candidate_id for entry in selection.entries)
    references = tuple(importer.iter_candidates(candidate_ids=wanted))
    index = {reference.candidate_id: reference for reference in references}
    if len(index) != len(references) or set(index) != set(wanted):
        raise CampaignSelectionError("selected source candidate is absent or duplicated")
    return MappingProxyType(index)


def _verify_imported_entry(
    *,
    selection: CampaignSelection,
    entry: SelectionEntry,
    reference: LegacyCandidateRef,
    imported: ImportedLegacyEpisode,
    rubrics: CompiledRubricRegistry,
) -> None:
    if (
        reference.candidate_id != entry.source_candidate_id
        or reference.family != entry.source_family
        or reference.stage.value != entry.stage
        or reference.upstream_artifact_sha256 != entry.source_artifact_sha256
    ):
        raise CampaignSelectionError("selected registry reference differs")
    manifest = imported.manifest
    episode = imported.episode
    if (
        manifest.candidate_id != entry.source_candidate_id
        or manifest.source_family != entry.source_family
        or manifest.rubric_domain != entry.domain
        or manifest.stage.value != entry.stage
        or manifest.upstream_artifact_sha256 != entry.source_artifact_sha256
        or manifest.registry_file_blake3 != selection.source_registry_blake3
        or manifest.upstream_registry_sha256
        != selection.upstream_source_registry_sha256
        or manifest.rubric_availability is not RubricAvailability.READY
        or episode.episode_id != manifest.episode_id
        or episode.domain != entry.domain
        or episode.stage.value != entry.stage
        or not is_blake3(manifest.import_blake3)
    ):
        raise CampaignSelectionError("selected source import differs")
    rubric = rubrics.resolve(entry.domain, entry.stage)
    if rubric.rubric_id != entry.rubric_id or rubric.digest != entry.rubric_blake3:
        raise CampaignSelectionError("selected runtime rubric differs")


def _verified_imports(
    selection: CampaignSelection,
    *,
    importer: LegacySupervisorV4Importer,
    rubrics: CompiledRubricRegistry,
    worker_width: int,
) -> tuple[ImportedLegacyEpisode, ...]:
    if type(worker_width) is not int or worker_width < 1:
        raise CampaignSelectionError("worker_width must be a positive integer")
    if selection.source_registry_blake3 != importer.registry_file_blake3:
        raise CampaignSelectionError("selection source registry BLAKE3 differs")
    if selection.upstream_source_registry_sha256 != importer.upstream_registry_sha256:
        raise CampaignSelectionError("selection upstream registry SHA-256 differs")
    index = _selection_source_index(selection, importer)

    def load(entry: SelectionEntry) -> ImportedLegacyEpisode:
        reference = index[entry.source_candidate_id]
        imported = importer.import_candidate(reference)
        _verify_imported_entry(
            selection=selection,
            entry=entry,
            reference=reference,
            imported=imported,
            rubrics=rubrics,
        )
        return imported

    with ThreadPoolExecutor(max_workers=worker_width, thread_name_prefix="eva-source") as pool:
        return tuple(pool.map(load, selection.entries))


def verify_selected_campaign_sources(
    selection: CampaignSelection,
    *,
    importer: LegacySupervisorV4Importer,
    rubrics: CompiledRubricRegistry,
    plan: CampaignPlan | None = None,
    worker_width: int = 64,
) -> Mapping[str, str]:
    """Reopen all scheduled artifacts/evidence concurrently and return receipts."""

    selected_plan = plan or exact_6000_plan()
    verify_campaign_selection(selection, plan=selected_plan, rubrics=rubrics)
    imported = _verified_imports(
        selection,
        importer=importer,
        rubrics=rubrics,
        worker_width=worker_width,
    )
    return MappingProxyType(
        {
            entry.candidate_id: row.manifest.import_blake3
            for entry, row in zip(selection.entries, imported, strict=True)
        }
    )


def verify_campaign_selection_against_registry(
    selection: CampaignSelection,
    *,
    importer: LegacySupervisorV4Importer,
    rubrics: CompiledRubricRegistry,
    plan: CampaignPlan | None = None,
    worker_width: int = 64,
    verify_selected_sources: bool = True,
) -> Mapping[str, str]:
    """Rebuild the 9,000-row schedule and optionally reopen every source."""

    selected_plan = plan or exact_6000_plan()
    verify_campaign_selection(selection, plan=selected_plan, rubrics=rubrics)
    if type(verify_selected_sources) is not bool:
        raise CampaignSelectionError("verify_selected_sources must be boolean")
    if selection.source_registry_blake3 != importer.registry_file_blake3:
        raise CampaignSelectionError("selection source registry BLAKE3 differs")
    if selection.upstream_source_registry_sha256 != importer.upstream_registry_sha256:
        raise CampaignSelectionError("selection upstream registry SHA-256 differs")
    expected = build_campaign_selection(
        importer=importer,
        rubrics=rubrics,
        plan=selected_plan,
        worker_width=worker_width,
    )
    if selection != expected:
        raise CampaignSelectionError("selection is not the deterministic frozen top-N")
    if not verify_selected_sources:
        return MappingProxyType({})
    return verify_selected_campaign_sources(
        selection,
        importer=importer,
        rubrics=rubrics,
        plan=selected_plan,
        worker_width=worker_width,
    )


@dataclass(frozen=True, slots=True)
class SelectionQueueRow:
    """Exact fields consumed by CampaignLedger.bootstrap_candidates."""

    candidate_id: str
    source_episode_id: str
    domain: str
    stage: str
    split: str
    rubric_blake3: str
    queue_ordinal: int
    cell_rank: int
    selection_tier: str

    def to_document(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "source_episode_id": self.source_episode_id,
            "domain": self.domain,
            "stage": self.stage,
            "split": self.split,
            "rubric_blake3": self.rubric_blake3,
            "queue_ordinal": self.queue_ordinal,
            "cell_rank": self.cell_rank,
            "selection_tier": self.selection_tier,
        }

    def to_candidate_queue_record(self) -> "CandidateQueueRecord":
        """Project without changing values into the durable ledger contract."""

        from .ledger import CandidateQueueRecord

        return CandidateQueueRecord(**self.to_document())


class FrozenCampaignCandidateSource:
    """Resolve one scheduled UUID into a fully verified three-cohort job."""

    def __init__(
        self,
        *,
        selection: CampaignSelection,
        importer: LegacySupervisorV4Importer,
        rubrics: CompiledRubricRegistry,
        targets: Sequence[ModelTarget],
        plan: CampaignPlan | None = None,
    ) -> None:
        selected_plan = plan or exact_6000_plan()
        verify_campaign_selection(selection, plan=selected_plan, rubrics=rubrics)
        if selection.source_registry_blake3 != importer.registry_file_blake3:
            raise CampaignSelectionError("selection source registry BLAKE3 differs")
        if selection.upstream_source_registry_sha256 != importer.upstream_registry_sha256:
            raise CampaignSelectionError("selection upstream registry SHA-256 differs")
        ordered_targets = tuple(targets)
        if len(ordered_targets) != 3 or {row.cohort for row in ordered_targets} != set(Cohort):
            raise CampaignSelectionError(
                "targets must contain weak, middle, and strong exactly once"
            )
        self._selection = selection
        self._importer = importer
        self._rubrics = rubrics
        self._targets = ordered_targets
        self._entries = MappingProxyType({row.candidate_id: row for row in selection.entries})
        self._references = MappingProxyType(
            {
                row.candidate_id: row
                for row in importer.iter_candidates(
                    candidate_ids=[entry.source_candidate_id for entry in selection.entries]
                )
            }
        )
        if set(self._references) != {row.source_candidate_id for row in selection.entries}:
            raise CampaignSelectionError("selected source candidate is absent from registry")

    @property
    def candidate_count(self) -> int:
        return len(self._entries)

    @property
    def scheduled_count(self) -> int:
        return len(self._entries)

    @property
    def primary_count(self) -> int:
        return len(self._selection.primary_entries)

    @property
    def reserve_count(self) -> int:
        return len(self._selection.reserve_entries)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(self._entries)

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

    def entry(self, candidate_id: str) -> SelectionEntry:
        _canonical_uuid(candidate_id, label="candidate_id")
        try:
            return self._entries[candidate_id]
        except KeyError:
            raise CampaignSelectionError("campaign candidate is not selected") from None

    def source_candidate_id(self, candidate_id: str) -> str:
        """Return the exact frozen source identity for deployment binding."""

        return self.entry(candidate_id).source_candidate_id

    def _selected_entries(
        self, candidate_ids: Iterable[str] | None
    ) -> tuple[SelectionEntry, ...]:
        identities = tuple(self._entries) if candidate_ids is None else tuple(candidate_ids)
        if len(set(identities)) != len(identities):
            raise CampaignSelectionError("campaign candidate selection is duplicated")
        return tuple(self.entry(candidate_id) for candidate_id in identities)

    def _import_entry(self, entry: SelectionEntry) -> ImportedLegacyEpisode:
        reference = self._references[entry.source_candidate_id]
        imported = self._importer.import_candidate(reference)
        _verify_imported_entry(
            selection=self._selection,
            entry=entry,
            reference=reference,
            imported=imported,
            rubrics=self._rubrics,
        )
        return imported

    def load(self, candidate_id: str) -> CandidateJob:
        entry = self.entry(candidate_id)
        imported = self._import_entry(entry)
        rubric = self._rubrics.resolve(entry.domain, entry.stage)
        return CandidateJob(
            candidate_id=entry.candidate_id,
            episode=imported.episode,
            rubric=rubric,
            targets=self._targets,
        )

    def verify_sources(
        self,
        candidate_ids: Iterable[str] | None = None,
        *,
        worker_width: int = 64,
    ) -> Mapping[str, str]:
        """Verify selected imports in parallel and return import commitments."""

        entries = self._selected_entries(candidate_ids)
        if type(worker_width) is not int or worker_width < 1:
            raise CampaignSelectionError("worker_width must be a positive integer")
        with ThreadPoolExecutor(max_workers=worker_width, thread_name_prefix="eva-source") as pool:
            imported = tuple(pool.map(self._import_entry, entries))
        return MappingProxyType(
            {
                entry.candidate_id: row.manifest.import_blake3
                for entry, row in zip(entries, imported, strict=True)
            }
        )

    def queue_rows(
        self,
        candidate_ids: Iterable[str] | None = None,
        *,
        worker_width: int = 64,
    ) -> tuple[SelectionQueueRow, ...]:
        """Return source-verified rows ready for one atomic ledger bootstrap."""

        entries = self._selected_entries(candidate_ids)
        if type(worker_width) is not int or worker_width < 1:
            raise CampaignSelectionError("worker_width must be a positive integer")
        with ThreadPoolExecutor(max_workers=worker_width, thread_name_prefix="eva-queue") as pool:
            imported = tuple(pool.map(self._import_entry, entries))
        return tuple(
            SelectionQueueRow(
                candidate_id=entry.candidate_id,
                source_episode_id=row.episode.episode_id,
                domain=entry.domain,
                stage=entry.stage,
                split=entry.split,
                rubric_blake3=entry.rubric_blake3,
                queue_ordinal=entry.queue_ordinal,
                cell_rank=entry.cell_rank,
                selection_tier=entry.selection_tier,
            )
            for entry, row in zip(entries, imported, strict=True)
        )

    def queue_records(
        self,
        candidate_ids: Iterable[str] | None = None,
        *,
        worker_width: int = 64,
    ) -> tuple["CandidateQueueRecord", ...]:
        """Return exact CandidateQueueRecord values for ledger.bootstrap_candidates."""

        return tuple(
            row.to_candidate_queue_record()
            for row in self.queue_rows(candidate_ids, worker_width=worker_width)
        )


__all__ = [
    "CampaignSelection",
    "CampaignSelectionError",
    "FrozenCampaignCandidateSource",
    "MAXIMUM_SELECTION_BYTES",
    "SCHEDULED_CANDIDATE_COUNT",
    "SELECTION_METHOD",
    "SELECTION_SCHEMA",
    "SelectionEntry",
    "SelectionQueueRow",
    "build_campaign_selection",
    "campaign_selection_from_document",
    "load_campaign_selection",
    "verify_campaign_selection",
    "verify_campaign_selection_against_registry",
    "verify_selected_campaign_sources",
    "write_campaign_selection_new",
]
