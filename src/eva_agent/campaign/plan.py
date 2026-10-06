"""Exact, balanced release quotas for the 6,000-sandbox MedResearch campaign."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eva_agent.pipeline.digests import blake3_hex


DOMAINS = (
    "automedbench-classification",
    "automedbench-detection",
    "automedbench-segmentation",
    "automedbench-research",
    "medxpertqa",
    "agentclinic",
    "healthbench-professional",
)
STAGES = ("S1", "S2", "S3", "S4", "S5", "E2E")
SPLITS = ("train", "development", "sealed_evaluation")
TARGET_BY_SPLIT = {"train": 6_000, "development": 0, "sealed_evaluation": 0}
TARGET_BY_DOMAIN = {
    "automedbench-classification": 150,
    "automedbench-detection": 300,
    "automedbench-segmentation": 300,
    "automedbench-research": 750,
    "medxpertqa": 1_500,
    "agentclinic": 1_500,
    "healthbench-professional": 1_500,
}
SOURCE_FAMILY_BY_DOMAIN = {
    "automedbench-classification": "automedbench",
    "automedbench-detection": "automedbench",
    "automedbench-segmentation": "automedbench",
    "automedbench-research": "automedbench",
    "medxpertqa": "medxpertqa",
    "agentclinic": "agentclinic",
    "healthbench-professional": "healthbench-professional",
}
TARGET_BY_SOURCE_FAMILY = {
    "automedbench": 1_500,
    "medxpertqa": 1_500,
    "agentclinic": 1_500,
    "healthbench-professional": 1_500,
}


class CampaignPlanError(ValueError):
    """An exact release quota invariant was not met."""


@dataclass(frozen=True)
class CellQuota:
    domain: str
    stage: str
    train: int
    development: int
    sealed_evaluation: int

    @property
    def total(self) -> int:
        return self.train + self.development + self.sealed_evaluation

    def to_document(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "stage": self.stage,
            "train": self.train,
            "development": self.development,
            "sealed_evaluation": self.sealed_evaluation,
            "total": self.total,
        }


@dataclass(frozen=True)
class CampaignPlan:
    plan_id: str
    cells: tuple[CellQuota, ...]
    target_by_split: dict[str, int]
    total: int
    plan_blake3: str

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": "eva.medresearch-campaign-plan.v1",
            "plan_id": self.plan_id,
            "cells": [cell.to_document() for cell in self.cells],
            "target_by_split": dict(self.target_by_split),
            "total": self.total,
            "plan_blake3": self.plan_blake3,
        }


def exact_6000_plan() -> CampaignPlan:
    """Return 42 source-balanced all-training cells totaling exactly 6,000.

    The four benchmark source families contribute 1,500 sandboxes each.
    AgentClinic, MedXpertQA, and HealthBench Professional contribute 250 per
    stage. AutoMedBench contributes 25 classification, 50 detection, 50
    segmentation, and 125 research sandboxes per stage, following the pinned
    source's native 1:2:2:5 track distribution. Every stage therefore totals
    exactly 1,000. Development and sealed-evaluation fields remain in the
    unchanged campaign schema with zero quotas so old readers can reopen the
    plan without a schema fork.
    """

    per_stage = {
        "automedbench-classification": 25,
        "automedbench-detection": 50,
        "automedbench-segmentation": 50,
        "automedbench-research": 125,
        "medxpertqa": 250,
        "agentclinic": 250,
        "healthbench-professional": 250,
    }
    cells: list[CellQuota] = []
    for domain in DOMAINS:
        for stage in STAGES:
            cells.append(
                CellQuota(
                    domain=domain,
                    stage=stage,
                    train=per_stage[domain],
                    development=0,
                    sealed_evaluation=0,
                )
            )
    core = {
        "schema": "eva.medresearch-campaign-plan.v1",
        "plan_id": "eva-medresearch-exact-6000-all-train-v2",
        "cells": [cell.to_document() for cell in cells],
        "target_by_split": TARGET_BY_SPLIT,
        "total": 6_000,
    }
    plan = CampaignPlan(
        plan_id=core["plan_id"],
        cells=tuple(cells),
        target_by_split=dict(TARGET_BY_SPLIT),
        total=6_000,
        plan_blake3=blake3_hex(core),
    )
    verify_plan(plan)
    return plan


def verify_plan(plan: CampaignPlan) -> None:
    if plan.plan_id == "eva-medresearch-exact-5000-500-500-v3":
        # Delayed import keeps the additive split-plan module free of an import
        # cycle while allowing ledger/deployment callers to verify either
        # immutable campaign generation through the unchanged public port.
        from .split_plan_v3 import verify_5000_500_500_plan

        verify_5000_500_500_plan(plan)
        return
    if len(plan.cells) != len(DOMAINS) * len(STAGES):
        raise CampaignPlanError("campaign must contain every configured domain × stage cell")
    keys = [(cell.domain, cell.stage) for cell in plan.cells]
    if len(set(keys)) != len(keys) or set(keys) != {
        (domain, stage) for domain in DOMAINS for stage in STAGES
    }:
        raise CampaignPlanError("campaign cell identities differ")
    if any(
        cell.train < 1
        or cell.development != 0
        or cell.sealed_evaluation != 0
        or cell.total != cell.train
        for cell in plan.cells
    ):
        raise CampaignPlanError("every campaign cell must be training-only and non-empty")
    observed = {
        "train": sum(cell.train for cell in plan.cells),
        "development": sum(cell.development for cell in plan.cells),
        "sealed_evaluation": sum(cell.sealed_evaluation for cell in plan.cells),
    }
    if observed != TARGET_BY_SPLIT or observed != plan.target_by_split:
        raise CampaignPlanError("release split totals differ from 6000/0/0")
    if sum(observed.values()) != plan.total or plan.total != 6_000:
        raise CampaignPlanError("campaign total differs from exactly 6,000")
    for domain in DOMAINS:
        group = [cell for cell in plan.cells if cell.domain == domain]
        if sum(cell.train for cell in group) != TARGET_BY_DOMAIN[domain]:
            raise CampaignPlanError("per-domain split balance differs")
    observed_families = {
        family: sum(
            cell.train
            for cell in plan.cells
            if SOURCE_FAMILY_BY_DOMAIN[cell.domain] == family
        )
        for family in TARGET_BY_SOURCE_FAMILY
    }
    if observed_families != TARGET_BY_SOURCE_FAMILY:
        raise CampaignPlanError("per-source-family balance differs")
    if any(sum(cell.total for cell in plan.cells if cell.stage == stage) != 1_000 for stage in STAGES):
        raise CampaignPlanError("per-stage total balance differs")
    core = plan.to_document()
    recorded = core.pop("plan_blake3")
    if blake3_hex(core) != recorded:
        raise CampaignPlanError("campaign plan BLAKE3 differs")


__all__ = [
    "CampaignPlan",
    "CampaignPlanError",
    "CellQuota",
    "DOMAINS",
    "SPLITS",
    "STAGES",
    "TARGET_BY_SPLIT",
    "TARGET_BY_DOMAIN",
    "SOURCE_FAMILY_BY_DOMAIN",
    "TARGET_BY_SOURCE_FAMILY",
    "exact_6000_plan",
    "verify_plan",
]
