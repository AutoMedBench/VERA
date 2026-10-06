"""Immutable compiled rubric, binding, and scoring value objects."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID, uuid4

from .integrity import blake3_document


class RubricScoreError(ValueError):
    """A judge supplied an incomplete or invalid atomic score table."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def canonical_uuid(value: str, *, label: str) -> str:
    """Validate and normalize the canonical hyphenated UUID spelling."""

    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a UUID") from exc
    if str(parsed) != value:
        raise ValueError(f"{label} must use canonical lowercase UUID spelling")
    return value


@dataclass(frozen=True, slots=True)
class RubricScore:
    """Deterministic reward calculation tied to one rubric commitment."""

    evaluation_id: str
    rubric_id: str
    rubric_digest: str
    item_scores_bps: Mapping[str, int]
    weighted_score_bps: int
    hard_gate_passed: bool
    failed_hard_gate_item_ids: tuple[str, ...]
    reward_bps: int
    score_digest: str

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": "eva.rubric-score.v1",
            "evaluation_id": self.evaluation_id,
            "rubric_id": self.rubric_id,
            "rubric_digest": self.rubric_digest,
            "item_scores_bps": dict(self.item_scores_bps),
            "weighted_score_bps": self.weighted_score_bps,
            "hard_gate_passed": self.hard_gate_passed,
            "failed_hard_gate_item_ids": list(self.failed_hard_gate_item_ids),
            "reward_bps": self.reward_bps,
            "score_digest": self.score_digest,
        }


@dataclass(frozen=True, slots=True)
class CompiledRubric:
    """One immutable domain × stage rubric table shared by all consumers."""

    _document: Mapping[str, Any]

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "CompiledRubric":
        return cls(_freeze(document))

    @property
    def rubric_id(self) -> str:
        return str(self._document["rubric_id"])

    @property
    def version(self) -> int:
        return int(self._document["version"])

    @property
    def domain(self) -> str:
        return str(self._document["domain"])

    @property
    def stage(self) -> str:
        return str(self._document["stage"])

    @property
    def digest(self) -> str:
        return str(self._document["rubric_digest"])

    @property
    def items(self) -> tuple[Mapping[str, Any], ...]:
        return self._document["items"]  # type: ignore[return-value]

    def to_document(self) -> dict[str, Any]:
        return _thaw(self._document)

    def score(
        self,
        item_scores_bps: Mapping[str, int],
        *,
        evaluation_id: str | None = None,
    ) -> RubricScore:
        """Score one judge result with the compiled table's exact semantics."""

        expected = {str(item["item_id"]): item for item in self.items}
        if set(item_scores_bps) != set(expected):
            raise RubricScoreError("item score keys must exactly match the compiled rubric")
        scores: dict[str, int] = {}
        failed: list[str] = []
        weighted_numerator = 0
        total_weight = int(self._document["total_weight"])
        for item_id, item in expected.items():
            score = item_scores_bps[item_id]
            if type(score) is not int:
                raise RubricScoreError(f"score for {item_id} must be integer basis points")
            allowed = {
                int(level["score_bps"])
                for level in item["partial_credit"]["levels"]
            }
            if score not in allowed:
                raise RubricScoreError(f"score for {item_id} is not a compiled partial-credit level")
            scores[item_id] = score
            weighted_numerator += int(item["weight"]) * score
            gate = item.get("hard_gate")
            if gate is not None and score < int(gate["minimum_score_bps"]):
                failed.append(item_id)
        weighted = (weighted_numerator + total_weight // 2) // total_weight
        runtime_id = canonical_uuid(evaluation_id or str(uuid4()), label="evaluation_id")
        core = {
            "schema": "eva.rubric-score.v1",
            "evaluation_id": runtime_id,
            "rubric_id": self.rubric_id,
            "rubric_digest": self.digest,
            "item_scores_bps": scores,
            "weighted_score_bps": weighted,
            "hard_gate_passed": not failed,
            "failed_hard_gate_item_ids": failed,
            "reward_bps": weighted if not failed else 0,
        }
        return RubricScore(
            evaluation_id=runtime_id,
            rubric_id=self.rubric_id,
            rubric_digest=self.digest,
            item_scores_bps=MappingProxyType(scores),
            weighted_score_bps=weighted,
            hard_gate_passed=not failed,
            failed_hard_gate_item_ids=tuple(failed),
            reward_bps=core["reward_bps"],
            score_digest=blake3_document(core),
        )


@dataclass(frozen=True, slots=True)
class SandboxRubricBinding:
    """Exactly one UUID-identified sandbox bound to one compiled table."""

    binding_id: str
    sandbox_id: str
    rubric: CompiledRubric
    registry_id: str
    registry_version: int
    registry_digest: str

    def for_benchmark_judging(self) -> CompiledRubric:
        return self.rubric

    def for_rollout_reward(self) -> CompiledRubric:
        return self.rubric

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": "eva.sandbox-rubric-binding.v1",
            "binding_id": self.binding_id,
            "sandbox_id": self.sandbox_id,
            "domain": self.rubric.domain,
            "stage": self.rubric.stage,
            "rubric_id": self.rubric.rubric_id,
            "rubric_version": self.rubric.version,
            "rubric_digest": self.rubric.digest,
            "registry_id": self.registry_id,
            "registry_version": self.registry_version,
            "registry_digest": self.registry_digest,
        }
