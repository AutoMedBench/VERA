"""Outcome-blind split assignment over the immutable 9,000 v2 identities."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes
from .plan import CampaignPlan
from .selection import SelectionQueueRow
from .selection_v2 import CampaignSelectionV2, FrozenCampaignCandidateSourceV2, SelectionEntryV2
from .split_plan_v3 import verify_5000_500_500_plan


SCHEMA = "eva.medresearch-campaign-selection.v3"
METHOD = "v2-identities_outcome-blind-source-rank_split-allocation_v3"


class CampaignSelectionV3Error(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SelectionEntryV3:
    candidate_id: str
    source_candidate_id: str
    source_family: str
    source_artifact_sha256: str
    domain: str
    stage: str
    split: str
    rubric_blake3: str
    rubric_id: str
    source_identity_rank_blake3: str
    construction_readiness: str
    readiness_proof_root_blake3: str
    selection_rank_blake3: str
    base_cell_rank: int
    base_cell_target: int
    base_selection_tier: str
    cell_rank: int
    cell_target: int
    selection_tier: str
    queue_ordinal: int

    def to_document(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class CampaignSelectionV3:
    plan_id: str
    plan_blake3: str
    base_selection_blake3: str
    source_registry_blake3: str
    rubric_registry_blake3: str
    entries: tuple[SelectionEntryV3, ...]
    selection_blake3: str

    def core_document(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "selection_method": METHOD,
            "plan_id": self.plan_id, "plan_blake3": self.plan_blake3,
            "base_selection_blake3": self.base_selection_blake3,
            "source_registry_blake3": self.source_registry_blake3,
            "rubric_registry_blake3": self.rubric_registry_blake3,
            "selected_count": 6000, "scheduled_count": 9000, "reserve_count": 3000,
            "entries": [row.to_document() for row in self.entries],
        }

    def to_document(self) -> dict[str, Any]:
        return {**self.core_document(), "selection_blake3": self.selection_blake3}


def _quotas(plan: CampaignPlan, domain: str, stage: str) -> dict[str, int]:
    row = next(cell for cell in plan.cells if (cell.domain, cell.stage) == (domain, stage))
    return {"train": row.train, "development": row.development, "sealed_evaluation": row.sealed_evaluation}


def _assign(rows: list[SelectionEntryV2], targets: Mapping[str, int]) -> dict[str, str]:
    ordered = sorted(rows, key=lambda row: (row.source_identity_rank_blake3, row.source_candidate_id))
    labels = [split for split in ("train", "development", "sealed_evaluation") for _ in range(targets[split])]
    if len(labels) != len(ordered):
        raise CampaignSelectionV3Error("split allocation cardinality differs")
    return {row.candidate_id: split for row, split in zip(ordered, labels, strict=True)}


def _derive_entries(base: CampaignSelectionV2, plan: CampaignPlan) -> tuple[SelectionEntryV3, ...]:
    verify_5000_500_500_plan(plan)
    if len(base.entries) != 9000 or len(base.primary_entries) != 6000:
        raise CampaignSelectionV3Error("base v2 inventory differs")
    by_cell: dict[tuple[str, str], list[SelectionEntryV2]] = defaultdict(list)
    for row in base.entries:
        by_cell[(row.domain, row.stage)].append(row)
    split_by_id: dict[str, str] = {}
    for cell in plan.cells:
        rows = by_cell[(cell.domain, cell.stage)]
        primary = [row for row in rows if row.selection_tier == "primary"]
        reserve = [row for row in rows if row.selection_tier == "reserve"]
        quotas = _quotas(plan, cell.domain, cell.stage)
        split_by_id.update(_assign(primary, quotas))
        total = sum(quotas.values())
        raw = {key: len(reserve) * value / total for key, value in quotas.items()}
        reserve_targets = {key: int(value) for key, value in raw.items()}
        remaining = len(reserve) - sum(reserve_targets.values())
        for key in sorted(raw, key=lambda name: (-(raw[name] - int(raw[name])), name))[:remaining]:
            reserve_targets[key] += 1
        split_by_id.update(_assign(reserve, reserve_targets))
    local_rank: dict[str, int] = {}
    for cell in plan.cells:
        quotas = _quotas(plan, cell.domain, cell.stage)
        cell_rows = by_cell[(cell.domain, cell.stage)]
        for split in ("train", "development", "sealed_evaluation"):
            primary = sorted(
                (row for row in cell_rows if row.selection_tier == "primary" and split_by_id[row.candidate_id] == split),
                key=lambda row: row.queue_ordinal,
            )
            reserve = sorted(
                (row for row in cell_rows if row.selection_tier == "reserve" and split_by_id[row.candidate_id] == split),
                key=lambda row: row.queue_ordinal,
            )
            for rank, row in enumerate(primary, start=1):
                local_rank[row.candidate_id] = rank
            for rank, row in enumerate(reserve, start=quotas[split] + 1):
                local_rank[row.candidate_id] = rank
    return tuple(
        SelectionEntryV3(
            candidate_id=row.candidate_id, source_candidate_id=row.source_candidate_id,
            source_family=row.source_family, source_artifact_sha256=row.source_artifact_sha256,
            domain=row.domain, stage=row.stage,
            split=split_by_id[row.candidate_id], rubric_blake3=row.rubric_blake3,
            rubric_id=row.rubric_id,
            source_identity_rank_blake3=row.source_identity_rank_blake3,
            construction_readiness=row.construction_readiness,
            readiness_proof_root_blake3=row.readiness_proof_root_blake3,
            selection_rank_blake3=row.selection_rank_blake3,
            base_cell_rank=row.cell_rank, base_cell_target=row.cell_target,
            base_selection_tier=row.selection_tier,
            cell_rank=local_rank[row.candidate_id],
            cell_target=_quotas(plan, row.domain, row.stage)[split_by_id[row.candidate_id]],
            selection_tier=row.selection_tier, queue_ordinal=row.queue_ordinal,
        ) for row in base.entries
    )


def build_campaign_selection_v3(base: CampaignSelectionV2, plan: CampaignPlan) -> CampaignSelectionV3:
    entries = _derive_entries(base, plan)
    provisional = CampaignSelectionV3(
        plan.plan_id, plan.plan_blake3, base.selection_blake3,
        base.source_registry_blake3, base.rubric_registry_blake3, entries, "",
    )
    result = replace(provisional, selection_blake3=blake3_hex(provisional.core_document()))
    verify_campaign_selection_v3(result, base, plan)
    return result


def verify_campaign_selection_v3(value: CampaignSelectionV3, base: CampaignSelectionV2, plan: CampaignPlan) -> None:
    verify_5000_500_500_plan(plan)
    if not (
        value.plan_id == plan.plan_id and value.plan_blake3 == plan.plan_blake3
        and value.base_selection_blake3 == base.selection_blake3
        and value.source_registry_blake3 == base.source_registry_blake3
        and value.rubric_registry_blake3 == base.rubric_registry_blake3
        and len(value.entries) == len(base.entries) == 9000
        and blake3_hex(value.core_document()) == value.selection_blake3
    ):
        raise CampaignSelectionV3Error("selection-v3 outer commitment differs")
    base_by_id = {row.candidate_id: row for row in base.entries}
    if (
        len({row.candidate_id for row in value.entries}) != 9000
        or {row.candidate_id for row in value.entries} != set(base_by_id)
        or tuple(row.candidate_id for row in value.entries)
        != tuple(row.candidate_id for row in base.entries)
        or [row.queue_ordinal for row in value.entries] != list(range(1, 9001))
    ):
        raise CampaignSelectionV3Error("selection-v3 changed candidate identities")
    for row in value.entries:
        old = base_by_id[row.candidate_id]
        immutable = (
            row.source_candidate_id, row.source_family, row.source_artifact_sha256,
            row.domain, row.stage, row.rubric_id, row.rubric_blake3,
            row.source_identity_rank_blake3,
            row.construction_readiness, row.readiness_proof_root_blake3,
            row.selection_rank_blake3,
            row.base_cell_rank, row.base_cell_target, row.base_selection_tier,
            row.selection_tier, row.queue_ordinal,
        )
        expected = (
            old.source_candidate_id, old.source_family, old.source_artifact_sha256,
            old.domain, old.stage, old.rubric_id, old.rubric_blake3,
            old.source_identity_rank_blake3,
            old.construction_readiness, old.readiness_proof_root_blake3,
            old.selection_rank_blake3,
            old.cell_rank, old.cell_target, old.selection_tier,
            old.selection_tier, old.queue_ordinal,
        )
        if immutable != expected:
            raise CampaignSelectionV3Error("selection-v3 changed a non-split field")
        if row.split not in {"train", "development", "sealed_evaluation"}:
            raise CampaignSelectionV3Error("selection-v3 split differs")
    primary = [row for row in value.entries if row.selection_tier == "primary"]
    expected_counts = Counter({
        (cell.domain, cell.stage, split): getattr(cell, split)
        for cell in plan.cells
        for split in ("train", "development", "sealed_evaluation")
    })
    actual_counts = Counter((row.domain, row.stage, row.split) for row in primary)
    if actual_counts != expected_counts:
        raise CampaignSelectionV3Error("selection-v3 primary split quotas differ")
    by_split_cell: dict[tuple[str, str, str], list[SelectionEntryV3]] = defaultdict(list)
    for row in value.entries:
        by_split_cell[(row.domain, row.stage, row.split)].append(row)
    for key, target in expected_counts.items():
        rows = by_split_cell[key]
        ranks = {row.cell_rank for row in rows}
        if (
            not rows
            or any(row.cell_target != target for row in rows)
            or ranks != set(range(1, len(rows) + 1))
            or any((row.cell_rank <= target) != (row.selection_tier == "primary") for row in rows)
            or not any(row.selection_tier == "reserve" for row in rows)
        ):
            raise CampaignSelectionV3Error("selection-v3 split-local reserve topology differs")
    if value.entries != _derive_entries(base, plan):
        raise CampaignSelectionV3Error("selection-v3 split assignment is not deterministic")


def write_new(path: Path, value: CampaignSelectionV3, base: CampaignSelectionV2, plan: CampaignPlan) -> None:
    verify_campaign_selection_v3(value, base, plan)
    payload = canonical_json_bytes(value.to_document())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.write(fd, payload); os.fsync(fd)
    finally:
        os.close(fd)


def load(path: Path, base: CampaignSelectionV2, plan: CampaignPlan) -> CampaignSelectionV3:
    document = json.loads(path.read_text())
    entries = tuple(SelectionEntryV3(**row) for row in document.pop("entries"))
    document.pop("schema"); document.pop("selection_method"); document.pop("selected_count"); document.pop("scheduled_count"); document.pop("reserve_count")
    value = CampaignSelectionV3(entries=entries, **document)
    verify_campaign_selection_v3(value, base, plan)
    deterministic = build_campaign_selection_v3(base, plan)
    if value != deterministic:
        raise CampaignSelectionV3Error("selection-v3 is not deterministic")
    return value


class SplitCampaignCandidateSourceV3:
    def __init__(self, base: FrozenCampaignCandidateSourceV2, selection: CampaignSelectionV3, plan: CampaignPlan) -> None:
        self._base, self._selection, self._plan = base, selection, plan
        self._entries = MappingProxyType({row.candidate_id: row for row in selection.entries})
    candidate_count = property(lambda self: len(self._entries))
    scheduled_count = candidate_count
    primary_count = property(lambda self: 6000)
    reserve_count = property(lambda self: 3000)
    candidate_ids = property(lambda self: tuple(row.candidate_id for row in self._selection.entries))
    primary_candidate_ids = property(lambda self: tuple(row.candidate_id for row in self._selection.entries if row.selection_tier == "primary"))
    reserve_candidate_ids = property(lambda self: tuple(row.candidate_id for row in self._selection.entries if row.selection_tier == "reserve"))
    plan_blake3 = property(lambda self: self._plan.plan_blake3)
    rubric_registry_blake3 = property(lambda self: self._selection.rubric_registry_blake3)
    source_registry_blake3 = property(lambda self: self._selection.source_registry_blake3)
    selection_blake3 = property(lambda self: self._selection.selection_blake3)
    def load(self, candidate_id: str): return self._base.load(candidate_id)
    def source_candidate_id(self, candidate_id: str) -> str: return self._entries[candidate_id].source_candidate_id
    def queue_rows(self, candidate_ids: Iterable[str] | None = None, *, worker_width: int = 64):
        ids = self.candidate_ids if candidate_ids is None else tuple(candidate_ids)
        with ThreadPoolExecutor(max_workers=worker_width) as pool:
            jobs = tuple(pool.map(self.load, ids))
        return tuple(SelectionQueueRow(
            candidate_id=row.candidate_id, source_episode_id=job.episode.episode_id,
            domain=row.domain, stage=row.stage, split=row.split,
            rubric_blake3=row.rubric_blake3, queue_ordinal=row.queue_ordinal,
            cell_rank=row.cell_rank, selection_tier=row.selection_tier,
        ) for row, job in zip((self._entries[i] for i in ids), jobs, strict=True))
    def queue_records(self, *args, **kwargs): return tuple(row.to_candidate_queue_record() for row in self.queue_rows(*args, **kwargs))


__all__ = ["CampaignSelectionV3", "CampaignSelectionV3Error", "SelectionEntryV3", "SplitCampaignCandidateSourceV3", "build_campaign_selection_v3", "load", "verify_campaign_selection_v3", "write_new"]
