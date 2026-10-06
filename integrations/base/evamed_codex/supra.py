"""Executable proposal contracts for evamed-codex-v1.3-supra.

These primitives are dependency-light so the workflow, tool, budget, and
trajectory contracts can be tested before the private EVA and SGLang payloads
are available. The release name remains a proposal until the live adapters use
the same contracts and produce passing receipts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence

from .runtime import CapabilityPolicy, ContractError


HARNESS_PROPOSAL = "evamed-codex-v1.3-supra-proposal"
HARNESS_RELEASE = "evamed-codex-v1.3-supra"
CONTEXT_TOKENS = 262_144
OUTPUT_TOKENS = 32_768
THINK_REASONING_TOKENS = 24_576
THINK_ANSWER_TOKENS = 8_192
MAX_TURNS = 100
WALL_SECONDS = 3_600
FINALIZE_SECONDS = 120
NORMAL_COMPACT_TOKENS = math.ceil(CONTEXT_TOKENS * 0.72)
EMERGENCY_COMPACT_TOKENS = math.ceil(CONTEXT_TOKENS * 0.85)
HASH = re.compile(r"^[0-9a-f]{64}$")
ZERO_HASH = "0" * 64


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError("canonical_json_invalid") from exc


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _reject_sensitive(value: Any) -> None:
    forbidden_keys = {
        "api_key", "authorization", "credential", "credentials", "token",
        "phi", "raw_chain_of_thought", "provider_private_reasoning",
        "private_evaluator_material",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in forbidden_keys:
                raise ContractError("sensitive_field_forbidden")
            _reject_sensitive(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_sensitive(child)
    elif isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in ("bearer ", "github_pat_", "raw chain of thought")):
            raise ContractError("sensitive_text_forbidden")


class Mode(str, Enum):
    INSTANT = "instant"
    THINK = "think"


class RunState(str, Enum):
    ADMITTED = "ADMITTED"
    BOUND = "BOUND"
    RUNNING = "RUNNING"
    VALIDATING = "VALIDATING"
    SUBMITTING = "SUBMITTING"
    CLOSED = "CLOSED"
    FAILED = "FAILED"
    INFRA_INVALID = "INFRA_INVALID"


TRANSITIONS = {
    RunState.ADMITTED: {RunState.BOUND, RunState.FAILED},
    RunState.BOUND: {RunState.RUNNING, RunState.FAILED, RunState.INFRA_INVALID},
    RunState.RUNNING: {RunState.VALIDATING, RunState.FAILED, RunState.INFRA_INVALID},
    RunState.VALIDATING: {RunState.RUNNING, RunState.SUBMITTING, RunState.FAILED, RunState.INFRA_INVALID},
    RunState.SUBMITTING: {RunState.CLOSED, RunState.FAILED, RunState.INFRA_INVALID},
    RunState.CLOSED: set(),
    RunState.FAILED: set(),
    RunState.INFRA_INVALID: set(),
}


@dataclass(frozen=True)
class FrozenCell:
    descriptor: Mapping[str, Any]
    sha256: str

    @classmethod
    def create(cls, descriptor: Mapping[str, Any]) -> "FrozenCell":
        value = dict(descriptor)
        required = {
            "harness", "mode", "policy_model", "policy_revision", "tokenizer_revision",
            "task_sha256", "tools_sha256", "memory_sha256", "sampling_sha256",
            "max_turns", "wall_seconds", "finalize_seconds", "context_tokens",
            "output_tokens",
        }
        if set(value) != required:
            raise ContractError("cell_fields_invalid")
        if value["harness"] not in {HARNESS_PROPOSAL, HARNESS_RELEASE}:
            raise ContractError("cell_harness_invalid")
        try:
            Mode(value["mode"])
        except ValueError as exc:
            raise ContractError("cell_mode_invalid") from exc
        for name in ("task_sha256", "tools_sha256", "memory_sha256", "sampling_sha256"):
            if not isinstance(value[name], str) or HASH.fullmatch(value[name]) is None:
                raise ContractError(f"cell_{name}_invalid")
        if (
            value["max_turns"] != MAX_TURNS
            or value["wall_seconds"] != WALL_SECONDS
            or value["finalize_seconds"] != FINALIZE_SECONDS
            or value["context_tokens"] != CONTEXT_TOKENS
            or value["output_tokens"] != OUTPUT_TOKENS
        ):
            raise ContractError("cell_budget_invalid")
        if not all(isinstance(value[name], str) and value[name] for name in (
            "policy_model", "policy_revision", "tokenizer_revision"
        )):
            raise ContractError("cell_model_identity_invalid")
        _reject_sensitive(value)
        return cls(descriptor=value, sha256=_hash(value))


@dataclass
class ContextBudget:
    components: dict[str, int] = field(default_factory=dict)

    @property
    def used(self) -> int:
        return sum(self.components.values())

    def set(self, component: str, tokens: int) -> None:
        if not component or type(tokens) is not int or tokens < 0:
            raise ContractError("context_component_invalid")
        proposed = self.used - self.components.get(component, 0) + tokens
        if proposed > CONTEXT_TOKENS:
            raise ContractError("context_limit_exceeded")
        self.components[component] = tokens

    def admit_generation(self, returned_tokens: int = OUTPUT_TOKENS) -> None:
        if type(returned_tokens) is not int or not 0 <= returned_tokens <= OUTPUT_TOKENS:
            raise ContractError("generation_reserve_invalid")
        if self.used + returned_tokens > CONTEXT_TOKENS:
            raise ContractError("context_plus_generation_exceeded")

    def compaction_level(self) -> str | None:
        if self.used >= EMERGENCY_COMPACT_TOKENS:
            return "emergency"
        if self.used >= NORMAL_COMPACT_TOKENS:
            return "normal"
        return None


@dataclass
class ModeBudget:
    mode: Mode
    reasoning_tokens: int = 0
    answer_tokens: int = 0

    @property
    def limits(self) -> tuple[int, int]:
        return (0, OUTPUT_TOKENS) if self.mode is Mode.INSTANT else (
            THINK_REASONING_TOKENS,
            THINK_ANSWER_TOKENS,
        )

    def consume(self, channel: str, tokens: int) -> None:
        if channel not in {"reasoning", "answer"} or type(tokens) is not int or tokens < 0:
            raise ContractError("mode_budget_event_invalid")
        reasoning_limit, answer_limit = self.limits
        proposed_reasoning = self.reasoning_tokens + (tokens if channel == "reasoning" else 0)
        proposed_answer = self.answer_tokens + (tokens if channel == "answer" else 0)
        if proposed_reasoning > reasoning_limit:
            raise ContractError("reasoning_budget_exceeded")
        if proposed_answer > answer_limit:
            raise ContractError("answer_budget_exceeded")
        if proposed_reasoning + proposed_answer > OUTPUT_TOKENS:
            raise ContractError("output_budget_exceeded")
        self.reasoning_tokens = proposed_reasoning
        self.answer_tokens = proposed_answer


TOOL_STATUSES = {"success", "empty", "retryable", "fatal"}
TOOL_ERRORS = {
    "E_SCHEMA_INVALID", "E_ENTITY_UNKNOWN", "E_AMBIGUOUS_ENTITY",
    "E_UNIT_MISMATCH", "E_STALE_SOURCE", "E_RATE_LIMIT", "E_TIMEOUT",
    "E_CAPABILITY_DENIED", "E_STAGE_INVALID", "E_INTERNAL",
}


@dataclass(frozen=True)
class SupraToolEnvelope:
    call_id: str
    tool: str
    schema_version: str
    status: str
    ok: bool
    summary: str
    data: Any
    provenance: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    error_code: str | None = None
    retry_hint: str | None = None
    artifact_ref: str | None = None

    def __post_init__(self) -> None:
        if self.status not in TOOL_STATUSES:
            raise ContractError("tool_status_invalid")
        if type(self.ok) is not bool or self.ok != (self.status in {"success", "empty"}):
            raise ContractError("tool_ok_status_mismatch")
        if not isinstance(self.summary, str):
            raise ContractError("tool_summary_invalid")
        if self.status == "success" and self.error_code is not None:
            raise ContractError("successful_tool_has_error")
        if self.status != "success" and self.error_code not in TOOL_ERRORS:
            raise ContractError("tool_error_code_invalid")
        if not isinstance(self.provenance, Mapping) or not isinstance(self.diagnostics, Mapping):
            raise ContractError("tool_receipt_metadata_invalid")
        if self.provenance.get("deidentified") is not True:
            raise ContractError("tool_deidentification_unverified")
        _reject_sensitive(self.__dict__)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _validate_value(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    kind = schema.get("type")
    checks = {
        "object": lambda x: isinstance(x, Mapping),
        "array": lambda x: isinstance(x, list),
        "string": lambda x: isinstance(x, str),
        "integer": lambda x: type(x) is int,
        "number": lambda x: type(x) in {int, float},
        "boolean": lambda x: type(x) is bool,
        "null": lambda x: x is None,
    }
    if kind in checks and not checks[kind](value):
        raise ContractError(f"tool_argument_type_invalid:{path}")
    if "enum" in schema and value not in schema["enum"]:
        raise ContractError(f"tool_argument_enum_invalid:{path}")
    if kind == "object":
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        if not required <= set(value):
            raise ContractError(f"tool_argument_required_missing:{path}")
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ContractError(f"tool_argument_unknown:{path}")
        for key, child in value.items():
            if key in properties:
                _validate_value(child, properties[key], f"{path}.{key}")
    elif kind == "array" and "items" in schema:
        for index, child in enumerate(value):
            _validate_value(child, schema["items"], f"{path}[{index}]")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    schema_version: str
    stage: str
    capability: str
    input_schema: Mapping[str, Any]
    handler: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    parallel_safe: bool = False


class MedicalToolGateway:
    def __init__(self, capability_policy: CapabilityPolicy) -> None:
        self.capability_policy = capability_policy
        self._tools: dict[str, ToolSpec] = {}
        self._calls: dict[str, tuple[str, str, SupraToolEnvelope]] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools or spec.stage not in {"S1", "S2", "S3", "S4", "S5"}:
            raise ContractError("tool_registration_invalid")
        if spec.input_schema.get("type") != "object" or spec.input_schema.get("additionalProperties") is not False:
            raise ContractError("tool_schema_not_strict")
        self._tools[spec.name] = spec

    def offered_schemas(self, stage: str) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": f"{spec.stage} medical workflow tool",
                    "parameters": spec.input_schema,
                    "x-schema-version": spec.schema_version,
                },
            }
            for spec in sorted(self._tools.values(), key=lambda item: item.name)
            if spec.stage == stage and self.capability_policy.permits(spec.capability)
        )

    def execute(
        self,
        *,
        call_id: str,
        name: str,
        arguments: Mapping[str, Any],
        stage: str,
        remaining_seconds: float,
    ) -> SupraToolEnvelope:
        if not call_id or remaining_seconds <= 0:
            raise ContractError("tool_deadline_invalid")
        spec = self._tools.get(name)
        if spec is None:
            raise ContractError("tool_unknown")
        self.capability_policy.require(spec.capability)
        if stage != spec.stage:
            raise ContractError("tool_stage_invalid")
        _reject_sensitive(arguments)
        _validate_value(arguments, spec.input_schema)
        arguments_hash = _hash(arguments)
        prior = self._calls.get(call_id)
        if prior is not None:
            prior_name, prior_hash, prior_envelope = prior
            if prior_name != name or prior_hash != arguments_hash:
                raise ContractError("tool_call_id_reused")
            return prior_envelope
        result = spec.handler(dict(arguments))
        envelope = SupraToolEnvelope(
            call_id=call_id,
            tool=name,
            schema_version=spec.schema_version,
            status=str(result.get("status")),
            ok=str(result.get("status")) in {"success", "empty"},
            summary=str(result.get("summary", "")),
            data=result.get("data"),
            provenance=result.get("provenance", {}),
            diagnostics=result.get("diagnostics", {}),
            error_code=result.get("error_code"),
            retry_hint=result.get("retry_hint"),
            artifact_ref=result.get("artifact_ref"),
        )
        self._calls[call_id] = (name, arguments_hash, envelope)
        return envelope


def _merkle_root(leaves: Sequence[str]) -> str:
    if not leaves:
        return _hash([])
    level = list(leaves)
    if any(HASH.fullmatch(leaf) is None for leaf in level):
        raise ContractError("merkle_leaf_invalid")
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(bytes.fromhex(level[i]) + bytes.fromhex(level[i + 1])).hexdigest() for i in range(0, len(level), 2)]
    return level[0]


class TrajectorySidecar:
    def __init__(self, cell: FrozenCell, trajectory_id: str, session_id: str) -> None:
        if not trajectory_id or not session_id:
            raise ContractError("trajectory_identity_invalid")
        self.cell = cell
        self.trajectory_id = trajectory_id
        self.session_id = session_id
        self.events: list[dict[str, Any]] = []
        self.samples: list[dict[str, Any]] = []
        self._sealed: dict[str, Any] | None = None

    def append_event(self, kind: str, payload: Mapping[str, Any], monotonic_ms: int) -> str:
        if self._sealed is not None:
            raise ContractError("trajectory_already_sealed")
        if not kind or type(monotonic_ms) is not int or monotonic_ms < 0:
            raise ContractError("trajectory_event_invalid")
        if self.events and monotonic_ms <= self.events[-1]["monotonic_ms"]:
            raise ContractError("trajectory_event_time_not_increasing")
        _reject_sensitive(payload)
        event = {
            "sequence": len(self.events),
            "kind": kind,
            "monotonic_ms": monotonic_ms,
            "cell_sha256": self.cell.sha256,
            "previous_event_sha256": self.events[-1]["event_sha256"] if self.events else ZERO_HASH,
            "payload": dict(payload),
        }
        event["event_sha256"] = _hash(event)
        self.events.append(event)
        return event["event_sha256"]

    def append_sample(
        self,
        *,
        sample_segment: int,
        token_ids: Sequence[int],
        logprobs: Sequence[float],
        loss_mask: Sequence[int],
        token_sources: Sequence[str],
    ) -> str:
        if self._sealed is not None:
            raise ContractError("trajectory_already_sealed")
        lengths = {len(token_ids), len(logprobs), len(loss_mask), len(token_sources)}
        if len(lengths) != 1 or sample_segment != len(self.samples):
            raise ContractError("sample_alignment_invalid")
        if any(bit not in {0, 1} for bit in loss_mask):
            raise ContractError("sample_loss_mask_invalid")
        allowed_sources = {"prompt", "policy_return", "tool", "memory", "compaction", "foreign", "scorer"}
        if any(source not in allowed_sources for source in token_sources):
            raise ContractError("sample_token_source_invalid")
        if any(bit == 1 and source != "policy_return" for bit, source in zip(loss_mask, token_sources)):
            raise ContractError("sample_nonpolicy_token_trainable")
        sample = {
            "sample_segment": sample_segment,
            "token_ids": list(token_ids),
            "logprobs": [float(value) for value in logprobs],
            "loss_mask": list(loss_mask),
            "token_sources": list(token_sources),
        }
        sample["sample_sha256"] = _hash(sample)
        self.samples.append(sample)
        return sample["sample_sha256"]

    def seal(self, artifact_hashes: Mapping[str, str]) -> Mapping[str, Any]:
        if any(HASH.fullmatch(str(value)) is None for value in artifact_hashes.values()):
            raise ContractError("trajectory_artifact_hash_invalid")
        leaves = [event["event_sha256"] for event in self.events]
        leaves.extend(sample["sample_sha256"] for sample in self.samples)
        leaves.extend(str(artifact_hashes[key]) for key in sorted(artifact_hashes))
        self._sealed = {
            "schema": "eva.codex-cotraining-trajectory.v1.3",
            "trajectory_id": self.trajectory_id,
            "session_id": self.session_id,
            "cell_sha256": self.cell.sha256,
            "event_count": len(self.events),
            "sample_count": len(self.samples),
            "artifact_hashes": dict(artifact_hashes),
            "merkle_root": _merkle_root(leaves),
        }
        return dict(self._sealed)

    def document(self) -> Mapping[str, Any]:
        if self._sealed is None:
            raise ContractError("trajectory_not_sealed")
        return {
            **self._sealed,
            "cell": dict(self.cell.descriptor),
            "events": [dict(event) for event in self.events],
            "samples": [dict(sample) for sample in self.samples],
        }

    def verify(self) -> None:
        prior = ZERO_HASH
        for sequence, event in enumerate(self.events):
            if event.get("sequence") != sequence or event.get("previous_event_sha256") != prior:
                raise ContractError("trajectory_chain_invalid")
            stored = event.get("event_sha256")
            candidate = dict(event)
            candidate.pop("event_sha256", None)
            if stored != _hash(candidate):
                raise ContractError("trajectory_event_tampered")
            prior = stored
        for sample in self.samples:
            stored = sample.get("sample_sha256")
            candidate = dict(sample)
            candidate.pop("sample_sha256", None)
            if stored != _hash(candidate):
                raise ContractError("trajectory_sample_tampered")
        if self._sealed is not None:
            artifact_hashes = self._sealed["artifact_hashes"]
            leaves = [event["event_sha256"] for event in self.events]
            leaves.extend(sample["sample_sha256"] for sample in self.samples)
            leaves.extend(str(artifact_hashes[key]) for key in sorted(artifact_hashes))
            if self._sealed.get("merkle_root") != _merkle_root(leaves):
                raise ContractError("trajectory_manifest_tampered")


class WorkflowMachine:
    def __init__(self, sidecar: TrajectorySidecar) -> None:
        self.sidecar = sidecar
        self.state = RunState.ADMITTED

    def transition(self, target: RunState, *, reason: str, monotonic_ms: int) -> None:
        if target not in TRANSITIONS[self.state] or not reason:
            raise ContractError("workflow_transition_invalid")
        previous = self.state
        self.state = target
        self.sidecar.append_event(
            "workflow_transition",
            {"from": previous.value, "to": target.value, "reason": reason},
            monotonic_ms,
        )
