"""Additive exact 5,000/500/500 release plan over the seven-domain roster."""

from __future__ import annotations

from .plan import (
    CampaignPlan,
    CampaignPlanError,
    CellQuota,
    DOMAINS,
    SOURCE_FAMILY_BY_DOMAIN,
    STAGES,
)
from eva_agent.pipeline.digests import blake3_hex


TARGET_BY_SPLIT_V3 = {"train": 5_000, "development": 500, "sealed_evaluation": 500}
TARGET_BY_FAMILY_V3 = {
    "automedbench": {"train": 1_250, "development": 125, "sealed_evaluation": 125},
    "medxpertqa": {"train": 1_250, "development": 125, "sealed_evaluation": 125},
    "agentclinic": {"train": 1_250, "development": 125, "sealed_evaluation": 125},
    "healthbench-professional": {"train": 1_250, "development": 125, "sealed_evaluation": 125},
}
TARGET_BY_DOMAIN_V3 = {
    "automedbench-classification": {"train": 126, "development": 12, "sealed_evaluation": 12},
    "automedbench-detection": {"train": 252, "development": 24, "sealed_evaluation": 24},
    "automedbench-segmentation": {"train": 252, "development": 24, "sealed_evaluation": 24},
    "automedbench-research": {"train": 620, "development": 65, "sealed_evaluation": 65},
    "medxpertqa": {"train": 1_250, "development": 125, "sealed_evaluation": 125},
    "agentclinic": {"train": 1_250, "development": 125, "sealed_evaluation": 125},
    "healthbench-professional": {"train": 1_250, "development": 125, "sealed_evaluation": 125},
}


def _standalone(stage: str) -> tuple[int, int, int]:
    if stage in {"S1", "S2", "S3", "S4"}:
        return (208, 21, 21)
    if stage == "S5":
        return (209, 20, 21)
    return (209, 21, 20)


def _automed(domain: str, stage: str) -> tuple[int, int, int]:
    fixed = {
        "automedbench-classification": (21, 2, 2),
        "automedbench-detection": (42, 4, 4),
        "automedbench-segmentation": (42, 4, 4),
    }
    if domain in fixed:
        return fixed[domain]
    if stage in {"S1", "S2", "S3", "S4"}:
        return (103, 11, 11)
    if stage == "S5":
        return (104, 10, 11)
    return (104, 11, 10)


def exact_5000_500_500_plan() -> CampaignPlan:
    cells = []
    for domain in DOMAINS:
        for stage in STAGES:
            train, development, sealed = (
                _automed(domain, stage)
                if SOURCE_FAMILY_BY_DOMAIN[domain] == "automedbench"
                else _standalone(stage)
            )
            cells.append(CellQuota(domain, stage, train, development, sealed))
    core = {
        "schema": "eva.medresearch-campaign-plan.v1",
        "plan_id": "eva-medresearch-exact-5000-500-500-v3",
        "cells": [cell.to_document() for cell in cells],
        "target_by_split": TARGET_BY_SPLIT_V3,
        "total": 6_000,
    }
    plan = CampaignPlan(
        plan_id=core["plan_id"], cells=tuple(cells),
        target_by_split=dict(TARGET_BY_SPLIT_V3), total=6_000,
        plan_blake3=blake3_hex(core),
    )
    verify_5000_500_500_plan(plan)
    return plan


def verify_5000_500_500_plan(plan: CampaignPlan) -> None:
    if plan.plan_id != "eva-medresearch-exact-5000-500-500-v3":
        raise CampaignPlanError("split-v3 plan identity differs")
    if plan.total != 6_000 or len(plan.cells) != len(DOMAINS) * len(STAGES):
        raise CampaignPlanError("split-v3 domain × stage coverage differs")
    if {(row.domain, row.stage) for row in plan.cells} != {
        (domain, stage) for domain in DOMAINS for stage in STAGES
    }:
        raise CampaignPlanError("split-v3 cells differ")
    observed = {
        split: sum(getattr(row, split) for row in plan.cells)
        for split in TARGET_BY_SPLIT_V3
    }
    if observed != TARGET_BY_SPLIT_V3 or plan.target_by_split != TARGET_BY_SPLIT_V3:
        raise CampaignPlanError("split-v3 totals differ from 5000/500/500")
    if any(row.total < 1 or min(row.train, row.development, row.sealed_evaluation) < 1 for row in plan.cells):
        raise CampaignPlanError("split-v3 cell is empty")
    expected_cells = {
        (domain, stage): (
            _automed(domain, stage)
            if SOURCE_FAMILY_BY_DOMAIN[domain] == "automedbench"
            else _standalone(stage)
        )
        for domain in DOMAINS for stage in STAGES
    }
    if {
        (row.domain, row.stage): (row.train, row.development, row.sealed_evaluation)
        for row in plan.cells
    } != expected_cells:
        raise CampaignPlanError("split-v3 exact cell matrix differs")
    if any(sum(row.total for row in plan.cells if row.stage == stage) != 1_000 for stage in STAGES):
        raise CampaignPlanError("split-v3 stage balance differs")
    for family, targets in TARGET_BY_FAMILY_V3.items():
        actual = {
            split: sum(
                getattr(row, split) for row in plan.cells
                if SOURCE_FAMILY_BY_DOMAIN[row.domain] == family
            )
            for split in targets
        }
        if actual != targets:
            raise CampaignPlanError("split-v3 source-family balance differs")
    for domain, targets in TARGET_BY_DOMAIN_V3.items():
        actual = {
            split: sum(getattr(row, split) for row in plan.cells if row.domain == domain)
            for split in targets
        }
        if actual != targets:
            raise CampaignPlanError("split-v3 domain balance differs")
    document = plan.to_document()
    digest = document.pop("plan_blake3")
    if blake3_hex(document) != digest:
        raise CampaignPlanError("split-v3 plan commitment differs")


__all__ = ["TARGET_BY_DOMAIN_V3", "TARGET_BY_FAMILY_V3", "TARGET_BY_SPLIT_V3", "exact_5000_500_500_plan", "verify_5000_500_500_plan"]
