"""Pure, descriptive evaluation-feedback -> stage-target proposals.

The caller must first verify the actual workspace Judge, citations, case/model/
checkpoint/catalog bindings, and exact compiled rubric against upstream receipts.
These types carry that already-verified result; a verification ID is a reference,
not proof by itself. This module only rechecks numeric scoring and comparability.
It performs no evaluation, causal attribution, skill edits, or RL admission.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, Iterable

from eva_agent.rubrics.integrity import canonical_json_bytes
from eva_agent.rubrics.models import CompiledRubric, RubricScore

STAGES = ("S1", "S2", "S3", "S4", "S5")


class FeedbackError(ValueError):
    """Feedback is invalid or cannot describe one comparable round."""


class EvaluationStatus(str, Enum):
    SCORED = "scored"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    UNMEASURED = "unmeasured"


@dataclass(frozen=True)
class RoundIdentity:
    model_id: str
    checkpoint_id: str
    skill_catalog_id: str
    # Exact upstream identity/configuration commitment, not merely a model family.
    # Native Astra and Opus are both allowed, but never mixed within this round.
    judge_id: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip() for value in asdict(self).values()):
            raise FeedbackError("round identities must be nonempty strings")


@dataclass(frozen=True)
class VerifiedStageEvaluation:
    case_id: str
    identity: RoundIdentity
    rubric: CompiledRubric
    status: EvaluationStatus
    upstream_verification_id: str
    score: RubricScore | None = None

    def validate(self) -> None:
        if not isinstance(self.identity, RoundIdentity) or not isinstance(self.rubric, CompiledRubric):
            raise FeedbackError("feedback requires typed round identity and compiled rubric")
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.case_id, self.upstream_verification_id)):
            raise FeedbackError("case and upstream verification identities are required")
        if not isinstance(self.status, EvaluationStatus) or self.rubric.stage not in (*STAGES, "E2E"):
            raise FeedbackError("unsupported evaluation status or stage")
        if self.status is not EvaluationStatus.SCORED:
            if self.score is not None:
                raise FeedbackError("missing/infrastructure evaluations cannot carry a zero or other score")
            return
        if not isinstance(self.score, RubricScore):
            raise FeedbackError("scored evaluation requires an exact RubricScore")
        expected = self.rubric.score(self.score.item_scores_bps, evaluation_id=self.score.evaluation_id)
        if canonical_json_bytes(expected.to_document()) != canonical_json_bytes(self.score.to_document()):
            raise FeedbackError("feedback score differs from exact compiled-rubric recomputation")


def _stage_metrics(rows: list[VerifiedStageEvaluation]) -> tuple[dict[str, Any], Fraction | None]:
    by_domain: dict[str, list[VerifiedStageEvaluation]] = defaultdict(list)
    for row in rows:
        by_domain[row.rubric.domain].append(row)
    domains, observed_means = {}, []
    for domain, entries in sorted(by_domain.items()):
        scored = [row for row in entries if row.status is EvaluationStatus.SCORED]
        mean = Fraction(sum(row.score.reward_bps for row in scored), len(scored)) if scored else None
        if mean is not None:
            observed_means.append(mean)
        items = []
        for item in entries[0].rubric.items:
            maximum = max(level["score_bps"] for level in item["partial_credit"]["levels"])
            values = [row.score.item_scores_bps[item["item_id"]] for row in scored]
            item_mean = Fraction(sum(values), len(values)) if values else None
            items.append({"item_id": item["item_id"], "title": item["title"],
                          "sample_count": len(values), "maximum_compiled_score_bps": maximum,
                          "mean_score_bps": None if item_mean is None else float(item_mean),
                          "mean_deficit_bps": None if item_mean is None else float(maximum - item_mean)})
        domains[domain] = {
            "sample_count": len(scored), "missing_count": len(entries) - len(scored),
            "missing_by_status": dict(Counter(row.status.value for row in entries
                                               if row.status is not EvaluationStatus.SCORED)),
            "mean_reward_bps": None if mean is None else float(mean),
            "rubric_id": entries[0].rubric.rubric_id, "rubric_digest": entries[0].rubric.digest,
            "item_deficits": items,
        }
    macro = sum(observed_means, Fraction()) / len(observed_means) if observed_means else None
    return {
        "status": "observed" if macro is not None else "unknown",
        "sample_count": sum(row["sample_count"] for row in domains.values()),
        "missing_count": sum(row["missing_count"] for row in domains.values()),
        "observed_domain_count": len(observed_means),
        "domain_macro_mean_reward_bps": None if macro is None else float(macro),
        "domains": domains,
    }, macro


def plan_stage_target(evaluations: Iterable[VerifiedStageEvaluation]) -> dict[str, Any]:
    """Propose the lowest measured S1-S5 domain-macro reward; exact ties use S1..S5.

    Each observed domain has equal weight, regardless of sample count. Unknown
    domains/stages are not zeros. Different observed domain sets are disclosed,
    not assumed to be a matched-case comparison or a causal skill diagnosis.
    """
    rows = tuple(evaluations)
    identity = None
    seen_cases, seen_evaluations, rubric_bindings = set(), set(), {}
    stages: dict[str, list[VerifiedStageEvaluation]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, VerifiedStageEvaluation):
            raise FeedbackError("feedback must be typed VerifiedStageEvaluation values")
        row.validate()
        if identity is None:
            identity = row.identity
        elif identity != row.identity:
            raise FeedbackError("cannot mix models/checkpoints/skill catalogs/Judges within one round")
        case_key = (row.case_id, row.rubric.domain, row.rubric.stage)
        if case_key in seen_cases:
            raise FeedbackError("duplicate case/domain/stage evaluation")
        seen_cases.add(case_key)
        rubric_key = (row.rubric.domain, row.rubric.stage)
        binding = canonical_json_bytes(row.rubric.to_document())
        if rubric_key in rubric_bindings and rubric_bindings[rubric_key] != binding:
            raise FeedbackError("cannot mix compiled rubrics for one domain/stage")
        rubric_bindings[rubric_key] = binding
        if row.score is not None:
            if row.score.evaluation_id in seen_evaluations:
                raise FeedbackError("duplicate scored evaluation identity")
            seen_evaluations.add(row.score.evaluation_id)
        stages[row.rubric.stage].append(row)
    metrics, candidates = {}, []
    for order, stage in enumerate(STAGES):
        metrics[stage], mean = _stage_metrics(stages[stage])
        if mean is not None:
            candidates.append((mean, order, stage))
    e2e, _ = _stage_metrics(stages["E2E"])
    target = min(candidates)[2] if candidates else None
    return {
        "schema": "eva.coevolution-stage-target-proposal.v1",
        "status": "target_proposed" if target else "insufficient_stage_evidence",
        "round_identity": None if identity is None else asdict(identity), "s_target": target,
        "selection_rule": "lowest_observed_domain_macro_mean_reward; ties=S1,S2,S3,S4,S5",
        "stages": metrics, "e2e": e2e, "input_count": len(rows),
        "upstream_verification_refs": sorted({row.upstream_verification_id for row in rows}),
        "attribution": "not_performed", "skill_change_proposals": [], "execution_authorized": False,
        "rl_materialization_performance_threshold": None,
        "limitations": ["Upstream workspace-Judge verification is the caller's responsibility.",
                        "Observed domain/case coverage may differ between stages; this is a descriptive training heuristic.",
                        "Low rewards do not establish a skill deficiency or authorize skill changes."],
    }
