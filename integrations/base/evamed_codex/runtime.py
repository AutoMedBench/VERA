from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import re
from typing import Any, Callable, Iterable, Mapping


HASH = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_MEMORY_KEYS = {
    "credential",
    "credentials",
    "api_key",
    "token",
    "phi",
    "raw_chain_of_thought",
    "private_evaluator_reference",
}


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class ToolEnvelope:
    ok: bool
    summary: str
    provenance: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    artifact_ref: str | None = None
    next_cursor: str | None = None

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> "ToolEnvelope":
        if not isinstance(value, Mapping):
            raise ContractError("tool_envelope_not_object")
        unknown = set(value) - {
            "ok", "summary", "provenance", "diagnostics", "artifact_ref", "next_cursor"
        }
        if unknown:
            raise ContractError("tool_envelope_unknown_fields")
        if (
            type(value.get("ok")) is not bool
            or not isinstance(value.get("summary"), str)
            or not isinstance(value.get("provenance"), Mapping)
            or not isinstance(value.get("diagnostics"), Mapping)
        ):
            raise ContractError("tool_envelope_required_fields_invalid")
        for optional in ("artifact_ref", "next_cursor"):
            if value.get(optional) is not None and not isinstance(value[optional], str):
                raise ContractError("tool_envelope_optional_field_invalid")
        return cls(**dict(value))


@dataclass(frozen=True)
class CapabilityPolicy:
    allow: tuple[str, ...]
    deny: tuple[str, ...]

    def permits(self, capability: str) -> bool:
        if any(fnmatch.fnmatchcase(capability, pattern) for pattern in self.deny):
            return False
        return capability in self.allow

    def require(self, capability: str) -> None:
        if not self.permits(capability):
            raise ContractError(f"capability_denied:{capability}")


@dataclass(frozen=True)
class MemoryRecord:
    scope: str
    record_id: str
    text: str
    evidence_refs: tuple[str, ...] = ()
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scope not in {"run", "project", "procedure"}:
            raise ContractError("memory_scope_invalid")
        lowered = {key.lower() for key in self.artifact_hashes}
        if lowered & FORBIDDEN_MEMORY_KEYS:
            raise ContractError("memory_forbidden_field")
        if any(not HASH.fullmatch(value) for value in self.artifact_hashes.values()):
            raise ContractError("memory_artifact_hash_invalid")


class MemoryStore:
    def __init__(self, *, evaluation: bool = False) -> None:
        self.evaluation = evaluation
        self._records: dict[str, MemoryRecord] = {}

    def commit(self, record: MemoryRecord) -> None:
        if self.evaluation and record.scope == "procedure":
            raise ContractError("procedure_memory_read_only_in_evaluation")
        if any(marker in record.text.lower() for marker in ("bearer ", "github_pat_", "hf_", "raw chain of thought")):
            raise ContractError("memory_sensitive_text")
        self._records[record.record_id] = record

    def recall(self, scopes: Iterable[str], *, max_records: int = 8, max_tokens: int = 2048) -> tuple[MemoryRecord, ...]:
        allowed = set(scopes)
        selected: list[MemoryRecord] = []
        tokens = 0
        for record in reversed(tuple(self._records.values())):
            if record.scope not in allowed:
                continue
            estimate = max(1, len(record.text) // 4)
            if selected and tokens + estimate > max_tokens:
                break
            selected.append(record)
            tokens += estimate
            if len(selected) >= max_records:
                break
        return tuple(selected)

    def reset_evaluation_scopes(self) -> None:
        self._records = {key: value for key, value in self._records.items() if value.scope == "procedure"}


@dataclass(frozen=True)
class ContextItem:
    kind: str
    text: str
    token_count: int
    sampled_by_policy: bool


class ContextManager:
    PRESERVE = {
        "task_contract",
        "submission_contract",
        "evidence_registry",
        "artifact_references",
        "failed_attempts",
        "next_action",
    }

    def __init__(self, context_limit: int, *, compact_at: float = 0.72, emergency_at: float = 0.85) -> None:
        if context_limit <= 0 or not 0 < compact_at < emergency_at < 1:
            raise ContractError("context_policy_invalid")
        self.context_limit = context_limit
        self.compact_at = compact_at
        self.emergency_at = emergency_at

    def needs_compaction(self, used_tokens: int) -> str | None:
        fraction = used_tokens / self.context_limit
        if fraction >= self.emergency_at:
            return "emergency"
        if fraction >= self.compact_at:
            return "normal"
        return None

    def compact(self, items: Iterable[ContextItem], summarizer: Callable[[tuple[ContextItem, ...]], str]) -> tuple[ContextItem, ...]:
        values = tuple(items)
        preserved = tuple(item for item in values if item.kind in self.PRESERVE)
        compressible = tuple(item for item in values if item.kind not in self.PRESERVE)
        if not compressible:
            return preserved
        summary = summarizer(compressible)
        injected = ContextItem(
            kind="compaction_summary",
            text=summary,
            token_count=max(1, len(summary) // 4),
            sampled_by_policy=False,
        )
        return preserved + (injected,)

    @staticmethod
    def loss_mask(items: Iterable[ContextItem]) -> tuple[int, ...]:
        mask: list[int] = []
        for item in items:
            mask.extend([1 if item.sampled_by_policy else 0] * item.token_count)
        return tuple(mask)


def _require_hash(value: Any, code: str) -> None:
    if not isinstance(value, str) or HASH.fullmatch(value) is None:
        raise ContractError(code)


def validate_trajectory(row: Mapping[str, Any]) -> None:
    if row.get("schema") != "eva.codex-cotraining-trajectory.v1":
        raise ContractError("trajectory_schema_invalid")
    phase = row.get("phase")
    if phase not in {"sft", "rl", "evaluation"}:
        raise ContractError("trajectory_phase_invalid")
    versions = row.get("versions")
    required_versions = {
        "harness", "harness_hash", "runner", "codex_cli", "slime", "sglang",
        "megatron", "model", "tokenizer", "dataset", "skill_catalog",
    }
    if not isinstance(versions, Mapping) or not required_versions <= set(versions):
        raise ContractError("trajectory_versions_incomplete")
    if versions["harness"] != "evamed-codex-v1.1" or versions["skill_catalog"] != "evamed-skills-v1.1":
        raise ContractError("trajectory_harness_identity_invalid")
    if re.fullmatch(r"0\.(153|154)\.\d+", str(versions["codex_cli"])) is None:
        raise ContractError("trajectory_codex_version_invalid")
    _require_hash(versions["harness_hash"], "trajectory_harness_hash_invalid")

    loss = row.get("loss")
    if not isinstance(loss, Mapping):
        raise ContractError("trajectory_loss_invalid")
    token_ids, mask, logprobs = (loss.get(key) for key in ("token_ids", "loss_mask", "logprobs"))
    if not all(isinstance(value, list) for value in (token_ids, mask, logprobs)) or not (
        len(token_ids) == len(mask) == len(logprobs)
    ):
        raise ContractError("trajectory_loss_alignment_invalid")
    if any(bit not in {0, 1} for bit in mask):
        raise ContractError("trajectory_loss_mask_invalid")
    if phase in {"sft", "rl"} and not any(mask):
        raise ContractError("trajectory_has_no_policy_tokens")

    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ContractError("trajectory_provenance_invalid")
    _require_hash(provenance.get("task_hash"), "trajectory_task_hash_invalid")
    _require_hash(provenance.get("tool_trace_hash"), "trajectory_tool_hash_invalid")
    artifacts = provenance.get("artifact_hashes")
    if not isinstance(artifacts, Mapping) or any(not HASH.fullmatch(str(value)) for value in artifacts.values()):
        raise ContractError("trajectory_artifact_hash_invalid")

    outcome = row.get("outcome")
    if not isinstance(outcome, Mapping) or outcome.get("status") not in {
        "completed", "failed", "infrastructure-invalid"
    }:
        raise ContractError("trajectory_outcome_invalid")
    if phase == "rl":
        required = {
            "reward_components", "judge_receipt", "reward_verifier_receipt",
            "attribution_receipt", "infrastructure_disposition",
        }
        if not required <= set(outcome):
            raise ContractError("rl_receipts_incomplete")
        if outcome["status"] == "infrastructure-invalid":
            raise ContractError("rl_infrastructure_invalid_not_trainable")
        judge = outcome["judge_receipt"]
        verifier = outcome["reward_verifier_receipt"]
        attribution = outcome["attribution_receipt"]
        if (
            not isinstance(judge, Mapping)
            or judge.get("model") != "aws/anthropic/bedrock-claude-opus-5"
            or not isinstance(verifier, Mapping)
            or verifier.get("model") != "aws/anthropic/bedrock-claude-opus-5"
            or verifier.get("outcome") != "accept"
            or not isinstance(attribution, Mapping)
            or attribution.get("model") != "openai/openai/gpt-6-astra"
            or attribution.get("sets_training_reward") is not False
            or outcome.get("infrastructure_disposition") != "valid"
        ):
            raise ContractError("rl_judge_verifier_attribution_invalid")
