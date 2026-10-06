"""Versioned premium candidate construction over the Codex ``run_once`` port.

This module is deliberately a construction orchestrator, not a legacy
candidate compiler.  A caller-owned adapter reopens an existing frozen source
request, produces Codex turn requests carrying the source output schemas
unchanged, and invokes the existing validators.  A caller-owned sink may then
publish the completed evidence without this module claiming compatibility
with any historical manifest.

The prospective v3 semantic schedule is fixed and contains no retry path::

    Opus 5 draft       --+       Opus 4.8 critique --+-- role quorum --+
                         +----> Opus 5 critique ------+                 |
    Gemini alternate  --+       GPT-5.6 comparison -+-- role quorum --+--> Opus 5 revision
                                Gemini comparison ---+

The two-call author wave and four-call critic wave drain completely.  Failure
of one lane never cancels or hides an already-started peer.  The primary Opus
author, at least one exact-schema result in each critic role, and revision
remain hard gates.  Critic selection is deterministic and primary-first; every
successful and failed attempt remains first-class evidence.  Only a failed
supplemental Gemini author may be replaced by an explicit zero-call,
adapter-validated canonical primary alias.  Outer campaign concurrency is
independent: ``construct_many`` submits one complete schedule per job to a
caller-selected worker width.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import json
import re
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from eva_agent.codex_runtime import (
    CodexRole,
    CodexSandbox,
    CodexThreadOptions,
    CodexTurnInput,
    CodexTurnReceipt,
    codex_turn_receipt_from_document,
    verify_codex_turn_receipt,
)
from eva_agent.pipeline.contracts import JsonValue, RuntimeIdFactory, freeze_json, uuid_text
from eva_agent.pipeline.digests import (
    blake3_hex,
    canonical_json_bytes,
    canonical_value,
    is_blake3,
)
from eva_agent.pipeline.ids import RandomUUIDFactory


class PremiumConstructionError(ValueError):
    """A premium construction boundary failed closed."""


class ConstructionLane(str, Enum):
    OPUS5_DRAFT = "opus5_draft"
    GEMINI_ALTERNATE = "gemini_alternate"
    OPUS48_CRITIQUE = "opus48_critique"
    OPUS5_CRITIQUE_BACKUP = "opus5_critique_backup"
    GPT56_COMPARISON = "gpt56_comparison"
    GEMINI_COMPARISON_BACKUP = "gemini_comparison_backup"
    OPUS5_REVISION = "opus5_revision"


AUTHOR_LANES = (
    ConstructionLane.OPUS5_DRAFT,
    ConstructionLane.GEMINI_ALTERNATE,
)
CRITIC_LANES = (
    ConstructionLane.OPUS48_CRITIQUE,
    ConstructionLane.GPT56_COMPARISON,
)
LANE_ORDER = (*AUTHOR_LANES, *CRITIC_LANES, ConstructionLane.OPUS5_REVISION)
CRITIQUE_QUORUM_LANES = (
    ConstructionLane.OPUS48_CRITIQUE,
    ConstructionLane.OPUS5_CRITIQUE_BACKUP,
)
COMPARISON_QUORUM_LANES = (
    ConstructionLane.GPT56_COMPARISON,
    ConstructionLane.GEMINI_COMPARISON_BACKUP,
)
CRITIC_QUORUM_LANES = (
    *CRITIQUE_QUORUM_LANES,
    *COMPARISON_QUORUM_LANES,
)
V3_LANE_ORDER = (
    *AUTHOR_LANES,
    *CRITIC_QUORUM_LANES,
    ConstructionLane.OPUS5_REVISION,
)
V4_CRITIC_LANES = (
    ConstructionLane.OPUS48_CRITIQUE,
    ConstructionLane.OPUS5_CRITIQUE_BACKUP,
    ConstructionLane.GPT56_COMPARISON,
    ConstructionLane.GEMINI_COMPARISON_BACKUP,
)
V4_LANE_ORDER = (*AUTHOR_LANES, *V4_CRITIC_LANES, ConstructionLane.OPUS5_REVISION)

_LANE_ROLE = {
    ConstructionLane.OPUS5_DRAFT: CodexRole.STRONG_ACTOR,
    ConstructionLane.GEMINI_ALTERNATE: CodexRole.STRONG_ACTOR,
    ConstructionLane.OPUS48_CRITIQUE: CodexRole.MIDDLE_ACTOR,
    ConstructionLane.OPUS5_CRITIQUE_BACKUP: CodexRole.STRONG_ACTOR,
    ConstructionLane.GPT56_COMPARISON: CodexRole.STRONG_ACTOR,
    ConstructionLane.GEMINI_COMPARISON_BACKUP: CodexRole.STRONG_ACTOR,
    ConstructionLane.OPUS5_REVISION: CodexRole.STRONG_ACTOR,
}
_LANE_SANDBOX = {
    ConstructionLane.OPUS5_DRAFT: CodexSandbox.READ_ONLY,
    ConstructionLane.GEMINI_ALTERNATE: CodexSandbox.READ_ONLY,
    ConstructionLane.OPUS48_CRITIQUE: CodexSandbox.READ_ONLY,
    ConstructionLane.OPUS5_CRITIQUE_BACKUP: CodexSandbox.READ_ONLY,
    ConstructionLane.GPT56_COMPARISON: CodexSandbox.READ_ONLY,
    ConstructionLane.GEMINI_COMPARISON_BACKUP: CodexSandbox.READ_ONLY,
    ConstructionLane.OPUS5_REVISION: CodexSandbox.READ_ONLY,
}
_FAILURE_CODES = frozenset(
    {
        "runner_error",
        "invalid_turn_receipt",
        "turn_identity_mismatch",
        "terminal_status",
        "missing_structured_output",
        "strict_json",
        "output_schema",
        "adapter_validation",
    }
)
PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V1 = "eva.premium-codex-construction.v1"
PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V2 = "eva.premium-codex-construction.v2"
PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3 = "eva.premium-codex-construction.v3"
PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4 = "eva.premium-codex-construction.v4"
AUTHOR_QUORUM_POLICY_V1 = "opus5_primary_required_gemini_supplemental_1_of_2_v1"
SUPPLEMENTAL_FALLBACK_POLICY_V1 = "adapter_exact_canonical_primary_alias_v1"
CRITIC_QUORUM_POLICY_V1 = (
    "parallel_role_quorum_primary_first_critique_and_comparison_1_of_2_v1"
)
CRITIC_AUTHORITY_POLICY_V2 = (
    "opus5_schema_authority_lower_tier_one_shot_evidence_v2"
)

_RESULT_SCHEMA_V1 = PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V1
_RESULT_SCHEMA_V2 = PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V2
_RESULT_SCHEMA_V3 = PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3
_RESULT_SCHEMA_V4 = PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4
_AUTHOR_QUORUM_POLICY = AUTHOR_QUORUM_POLICY_V1
_FALLBACK_POLICY = SUPPLEMENTAL_FALLBACK_POLICY_V1
_CRITIC_QUORUM_POLICY = CRITIC_QUORUM_POLICY_V1
_CRITIC_AUTHORITY_POLICY = CRITIC_AUTHORITY_POLICY_V2
STRICT_RESPONSE_EXTRACTION_POLICY_V1 = "strict_entire_response_json_object_v1"
UNIQUE_SCHEMA_RESPONSE_EXTRACTION_POLICY_V2 = (
    "unique_schema_valid_json_object_from_raw_response_v2"
)


def _mapping(value: Any, *, label: str) -> Mapping[str, JsonValue]:
    frozen = freeze_json(value)
    if not isinstance(frozen, Mapping):
        raise PremiumConstructionError(f"{label} must be a JSON object")
    return frozen


def _nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PremiumConstructionError(f"{label} is required")
    return value


@dataclass(frozen=True)
class FrozenConstructionSource:
    """One immutable, pre-provider construction input commitment."""

    source_id: str
    request: Mapping[str, JsonValue]
    request_blake3: str

    @classmethod
    def create(
        cls, *, source_id: str, request: Mapping[str, Any]
    ) -> "FrozenConstructionSource":
        frozen = _mapping(request, label="frozen construction request")
        return cls(source_id=source_id, request=frozen, request_blake3=blake3_hex(frozen))

    def __post_init__(self) -> None:
        _nonempty(self.source_id, label="construction source_id")
        frozen = _mapping(self.request, label="frozen construction request")
        object.__setattr__(self, "request", frozen)
        if self.request_blake3 != blake3_hex(frozen):
            raise PremiumConstructionError("frozen construction request BLAKE3 differs")


@dataclass(frozen=True)
class ConstructionModelRoute:
    """Public model identity for one logical premium route."""

    route_id: str
    model: str
    provider: str

    def __post_init__(self) -> None:
        _nonempty(self.route_id, label="construction route_id")
        _nonempty(self.model, label="construction model")
        _nonempty(self.provider, label="construction provider")

    @classmethod
    def from_target(cls, value: Any) -> "ConstructionModelRoute":
        """Project a deployment route structurally without importing deployment.

        Both ``ProviderRouteTarget(route_id, target=ModelTarget(...))`` and a
        direct route carrying ``model_id``/``provider`` are accepted.  No
        config value or transport secret crosses this projection.
        """

        route_id = getattr(value, "route_id", None)
        target = getattr(value, "target", value)
        model = getattr(target, "model_id", getattr(target, "model", None))
        provider = getattr(
            target, "provider", getattr(target, "provider_id", None)
        )
        return cls(route_id=route_id, model=model, provider=provider)


@dataclass(frozen=True)
class PremiumConstructionRoutes:
    opus5: ConstructionModelRoute
    gemini31: ConstructionModelRoute
    opus48: ConstructionModelRoute
    gpt56: ConstructionModelRoute

    def __post_init__(self) -> None:
        expected = {
            "opus5": "opus_5",
            "gemini31": "gemini_3_1_pro",
            "opus48": "opus_4_8",
            "gpt56": "gpt_5_6_sol",
        }
        for field, route_id in expected.items():
            value = getattr(self, field)
            if not isinstance(value, ConstructionModelRoute) or value.route_id != route_id:
                raise PremiumConstructionError(f"{field} construction route differs")

    def for_lane(self, lane: ConstructionLane) -> ConstructionModelRoute:
        if lane in {
            ConstructionLane.OPUS5_DRAFT,
            ConstructionLane.OPUS5_CRITIQUE_BACKUP,
            ConstructionLane.OPUS5_REVISION,
        }:
            return self.opus5
        if lane in {
            ConstructionLane.GEMINI_ALTERNATE,
            ConstructionLane.GEMINI_COMPARISON_BACKUP,
        }:
            return self.gemini31
        if lane is ConstructionLane.OPUS48_CRITIQUE:
            return self.opus48
        if lane is ConstructionLane.GPT56_COMPARISON:
            return self.gpt56
        raise PremiumConstructionError("unknown construction lane")

    @classmethod
    def from_targets(cls, values: Sequence[Any]) -> "PremiumConstructionRoutes":
        """Build the exact four premium routes from deployment route targets."""

        routes: dict[str, ConstructionModelRoute] = {}
        for value in values:
            route = ConstructionModelRoute.from_target(value)
            if route.route_id in routes:
                raise PremiumConstructionError("premium construction route is duplicated")
            routes[route.route_id] = route
        expected = {"opus_5", "gemini_3_1_pro", "opus_4_8", "gpt_5_6_sol"}
        if set(routes) != expected:
            raise PremiumConstructionError("premium construction route set differs")
        return cls(
            opus5=routes["opus_5"],
            gemini31=routes["gemini_3_1_pro"],
            opus48=routes["opus_4_8"],
            gpt56=routes["gpt_5_6_sol"],
        )


def _request_core(
    *,
    lane: ConstructionLane,
    options: CodexThreadOptions,
    turn_input: CodexTurnInput,
    source_request_blake3: str,
    source_phase_request_blake3: str,
    source_output_schema_blake3: str,
    dependencies: Mapping[str, str],
) -> Mapping[str, Any]:
    return {
        "lane": lane,
        "source_request_blake3": source_request_blake3,
        "source_phase_request_blake3": source_phase_request_blake3,
        "source_output_schema_blake3": source_output_schema_blake3,
        "dependencies": dependencies,
        "thread": {
            "role": options.role,
            "model": options.model,
            "provider": options.provider,
            "cwd": options.cwd,
            "sandbox": options.sandbox,
            "config_keys": options.config_keys,
            "offered_tools": tuple(
                {
                    "canonical": tool.canonical_catalog_entry(),
                    "sidecar": tool.sidecar_metadata_entry(),
                }
                for tool in options.offered_tools
            ),
            "base_instructions_blake3": (
                None
                if options.base_instructions is None
                else blake3_hex(options.base_instructions)
            ),
            "developer_instructions_blake3": (
                None
                if options.developer_instructions is None
                else blake3_hex(options.developer_instructions)
            ),
            "ephemeral": options.ephemeral,
            "service_name": options.service_name,
            "service_tier": options.service_tier,
        },
        "turn": {
            "public_text": turn_input.public_text,
            "public_context": turn_input.public_context,
            "judge_only_context": turn_input.judge_only_context,
            "skills": tuple(skill.catalog_entry() for skill in turn_input.skills),
            "mentions": tuple(
                {"name": mention.name, "path": mention.path}
                for mention in turn_input.mentions
            ),
            "output_schema": turn_input.output_schema,
            "sandbox": turn_input.sandbox,
            "model": turn_input.model,
            "effort": turn_input.effort,
            "summary": turn_input.summary,
            "service_tier": turn_input.service_tier,
        },
    }


@dataclass(frozen=True)
class ConstructionTurnRequest:
    """Exact Codex projection of one already-frozen phase request.

    ``source_output_schema_blake3`` commits the schema copied from the source
    phase request.  Construction fails before ``run_once`` unless that digest
    exactly matches the schema carried by :class:`CodexTurnInput`.
    """

    lane: ConstructionLane
    options: CodexThreadOptions
    turn_input: CodexTurnInput
    source_request_blake3: str
    source_phase_request_blake3: str
    source_output_schema_blake3: str
    dependencies: Mapping[str, str]
    request_blake3: str

    @classmethod
    def create(
        cls,
        *,
        lane: ConstructionLane,
        options: CodexThreadOptions,
        turn_input: CodexTurnInput,
        source_request_blake3: str,
        source_phase_request: Mapping[str, Any],
        source_output_schema: Mapping[str, Any],
        dependencies: Mapping[str, str] | None = None,
    ) -> "ConstructionTurnRequest":
        dependency_map = MappingProxyType(dict(sorted((dependencies or {}).items())))
        phase_blake3 = blake3_hex(source_phase_request)
        schema_blake3 = blake3_hex(source_output_schema)
        if turn_input.output_schema is None or blake3_hex(turn_input.output_schema) != schema_blake3:
            raise PremiumConstructionError(
                "Codex output_schema differs from the frozen source phase request"
            )
        core = _request_core(
            lane=lane,
            options=options,
            turn_input=turn_input,
            source_request_blake3=source_request_blake3,
            source_phase_request_blake3=phase_blake3,
            source_output_schema_blake3=schema_blake3,
            dependencies=dependency_map,
        )
        return cls(
            lane=lane,
            options=options,
            turn_input=turn_input,
            source_request_blake3=source_request_blake3,
            source_phase_request_blake3=phase_blake3,
            source_output_schema_blake3=schema_blake3,
            dependencies=dependency_map,
            request_blake3=blake3_hex(core),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.lane, ConstructionLane):
            raise PremiumConstructionError("construction lane differs")
        if not isinstance(self.options, CodexThreadOptions) or not isinstance(
            self.turn_input, CodexTurnInput
        ):
            raise PremiumConstructionError("construction Codex request differs")
        for digest, label in (
            (self.source_request_blake3, "source request"),
            (self.source_phase_request_blake3, "source phase request"),
            (self.source_output_schema_blake3, "source output schema"),
            (self.request_blake3, "Codex construction request"),
        ):
            if not is_blake3(digest):
                raise PremiumConstructionError(f"{label} BLAKE3 differs")
        dependencies = dict(sorted(self.dependencies.items()))
        if any(not isinstance(key, str) or not key or not is_blake3(value) for key, value in dependencies.items()):
            raise PremiumConstructionError("construction dependency commitment differs")
        object.__setattr__(self, "dependencies", MappingProxyType(dependencies))
        expected_role = _LANE_ROLE[self.lane]
        expected_sandbox = _LANE_SANDBOX[self.lane]
        if self.options.role is not expected_role or self.options.sandbox is not expected_sandbox:
            raise PremiumConstructionError("construction lane role/sandbox boundary differs")
        if self.turn_input.judge_only_context is not None:
            raise PremiumConstructionError("construction actor cannot receive judge-only context")
        if self.turn_input.sandbox not in {None, expected_sandbox}:
            raise PremiumConstructionError("construction turn sandbox override differs")
        if self.turn_input.model not in {None, self.options.model}:
            raise PremiumConstructionError("construction turn model override differs")
        if self.turn_input.output_schema is None or blake3_hex(
            self.turn_input.output_schema
        ) != self.source_output_schema_blake3:
            raise PremiumConstructionError("construction output_schema commitment differs")
        core = _request_core(
            lane=self.lane,
            options=self.options,
            turn_input=self.turn_input,
            source_request_blake3=self.source_request_blake3,
            source_phase_request_blake3=self.source_phase_request_blake3,
            source_output_schema_blake3=self.source_output_schema_blake3,
            dependencies=self.dependencies,
        )
        if self.request_blake3 != blake3_hex(core):
            raise PremiumConstructionError("Codex construction request BLAKE3 differs")


class ConstructionContractAdapterPort(Protocol):
    """Boundary to existing request builders and legacy validators.

    Implementations must reopen the real frozen phase requests and call their
    existing validators.  Returning a request does not certify compatibility
    with any legacy construction manifest.
    """

    def prepare_authors(
        self, source: FrozenConstructionSource, routes: PremiumConstructionRoutes
    ) -> Sequence[ConstructionTurnRequest]: ...

    def prepare_critics(
        self,
        source: FrozenConstructionSource,
        routes: PremiumConstructionRoutes,
        *,
        primary_draft: Mapping[str, JsonValue],
        alternate_draft: Mapping[str, JsonValue],
    ) -> Sequence[ConstructionTurnRequest]: ...

    def prepare_critic_quorum(
        self,
        source: FrozenConstructionSource,
        routes: PremiumConstructionRoutes,
        *,
        primary_draft: Mapping[str, JsonValue],
        alternate_draft: Mapping[str, JsonValue],
    ) -> Sequence[ConstructionTurnRequest]: ...

    def prepare_critic_authority_v4(
        self,
        source: FrozenConstructionSource,
        routes: PremiumConstructionRoutes,
        *,
        primary_draft: Mapping[str, JsonValue],
        alternate_draft: Mapping[str, JsonValue],
    ) -> Sequence[ConstructionTurnRequest]: ...

    def prepare_revision(
        self,
        source: FrozenConstructionSource,
        routes: PremiumConstructionRoutes,
        *,
        primary_draft: Mapping[str, JsonValue],
        alternate_draft: Mapping[str, JsonValue],
        opus48_critique: Mapping[str, JsonValue],
        gpt56_comparison: Mapping[str, JsonValue],
        selected_critique_lane: ConstructionLane = ConstructionLane.OPUS48_CRITIQUE,
        selected_comparison_lane: ConstructionLane = ConstructionLane.GPT56_COMPARISON,
    ) -> ConstructionTurnRequest: ...

    def validate_output(
        self,
        source: FrozenConstructionSource,
        request: ConstructionTurnRequest,
        output: Mapping[str, JsonValue],
    ) -> None: ...

    def resolve_supplemental_author_fallback(
        self,
        source: FrozenConstructionSource,
        request: ConstructionTurnRequest,
        *,
        primary_draft: Mapping[str, JsonValue],
        failure: "ConstructionFailureReceipt",
    ) -> Mapping[str, JsonValue]: ...


class _CodexTurnRunnerPort(Protocol):
    def run_once(
        self, options: CodexThreadOptions, turn_input: CodexTurnInput
    ) -> CodexTurnReceipt: ...

    def run_construction_once(
        self, request: ConstructionTurnRequest
    ) -> CodexTurnReceipt: ...


@dataclass(frozen=True)
class ConstructionPhaseReceipt:
    schema: str
    attempt_id: str
    lane: ConstructionLane
    source_request_blake3: str
    source_phase_request_blake3: str
    request_blake3: str
    output_schema_blake3: str
    dependency_blake3s: Mapping[str, str]
    model: str
    provider: str
    role: CodexRole
    sandbox: CodexSandbox
    codex_turn_receipt_blake3: str
    output_blake3: str
    semantic_attempt_count: int
    retry_count: int
    output_schema_unchanged: bool
    receipt_blake3: str

    def __post_init__(self) -> None:
        if self.schema != "eva.premium-codex-construction-phase.v1":
            raise PremiumConstructionError("construction phase receipt schema differs")
        uuid_text(self.attempt_id, label="construction attempt_id")
        if not isinstance(self.lane, ConstructionLane):
            raise PremiumConstructionError("construction phase lane differs")
        if (
            not self.model
            or not self.provider
            or self.role is not _LANE_ROLE[self.lane]
            or self.sandbox is not _LANE_SANDBOX[self.lane]
        ):
            raise PremiumConstructionError("construction phase identity boundary differs")
        dependencies = dict(sorted(self.dependency_blake3s.items()))
        object.__setattr__(self, "dependency_blake3s", MappingProxyType(dependencies))
        digests = (
            self.source_request_blake3,
            self.source_phase_request_blake3,
            self.request_blake3,
            self.output_schema_blake3,
            self.codex_turn_receipt_blake3,
            self.output_blake3,
            self.receipt_blake3,
            *dependencies.values(),
        )
        if any(not is_blake3(value) for value in digests):
            raise PremiumConstructionError("construction phase BLAKE3 differs")
        if (
            self.semantic_attempt_count != 1
            or self.retry_count != 0
            or self.output_schema_unchanged is not True
        ):
            raise PremiumConstructionError("construction phase attempt controls differ")
        if self.receipt_blake3 != blake3_hex(self.core()):
            raise PremiumConstructionError("construction phase receipt BLAKE3 differs")

    def core(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "attempt_id": self.attempt_id,
            "lane": self.lane,
            "source_request_blake3": self.source_request_blake3,
            "source_phase_request_blake3": self.source_phase_request_blake3,
            "request_blake3": self.request_blake3,
            "output_schema_blake3": self.output_schema_blake3,
            "dependency_blake3s": self.dependency_blake3s,
            "model": self.model,
            "provider": self.provider,
            "role": self.role,
            "sandbox": self.sandbox,
            "codex_turn_receipt_blake3": self.codex_turn_receipt_blake3,
            "output_blake3": self.output_blake3,
            "semantic_attempt_count": self.semantic_attempt_count,
            "retry_count": self.retry_count,
            "output_schema_unchanged": self.output_schema_unchanged,
        }

    def to_document(self) -> dict[str, Any]:
        return {**canonical_value(self.core()), "receipt_blake3": self.receipt_blake3}

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "ConstructionPhaseReceipt":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise PremiumConstructionError("construction phase receipt keys differ")
        fields = {key: value[key] for key in cls.__dataclass_fields__}
        try:
            fields["lane"] = ConstructionLane(fields["lane"])
            fields["role"] = CodexRole(fields["role"])
            fields["sandbox"] = CodexSandbox(fields["sandbox"])
        except (TypeError, ValueError):
            raise PremiumConstructionError("construction phase enum differs") from None
        return cls(**fields)


@dataclass(frozen=True)
class ConstructionPhaseResult:
    receipt: ConstructionPhaseReceipt
    output: Mapping[str, JsonValue]
    codex_turn_receipt: CodexTurnReceipt

    def __post_init__(self) -> None:
        output = _mapping(self.output, label="construction phase output")
        object.__setattr__(self, "output", output)
        verify_codex_turn_receipt(self.codex_turn_receipt)
        if (
            self.receipt.output_blake3 != blake3_hex(output)
            or self.receipt.codex_turn_receipt_blake3
            != self.codex_turn_receipt.receipt_blake3
            or self.receipt.model != self.codex_turn_receipt.model
            or self.receipt.provider != self.codex_turn_receipt.provider
            or self.receipt.role is not self.codex_turn_receipt.role
            or self.receipt.sandbox is not self.codex_turn_receipt.sandbox
        ):
            raise PremiumConstructionError("construction phase result binding differs")

    def to_document(self) -> dict[str, Any]:
        return canonical_value(
            {
                "receipt": self.receipt.to_document(),
                "output": self.output,
                "codex_turn_receipt": self.codex_turn_receipt,
            }
        )

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "ConstructionPhaseResult":
        if not isinstance(value, Mapping) or set(value) != {
            "receipt",
            "output",
            "codex_turn_receipt",
        }:
            raise PremiumConstructionError("construction phase result keys differ")
        receipt = value["receipt"]
        output = value["output"]
        turn_receipt = value["codex_turn_receipt"]
        if not all(isinstance(item, Mapping) for item in (receipt, output, turn_receipt)):
            raise PremiumConstructionError("construction phase result document differs")
        return cls(
            ConstructionPhaseReceipt.from_document(receipt),
            output,
            codex_turn_receipt_from_document(turn_receipt),
        )


@dataclass(frozen=True)
class ConstructionFailureReceipt:
    schema: str
    failure_id: str
    lane: ConstructionLane
    source_request_blake3: str
    request_blake3: str
    output_schema_blake3: str
    error_code: str
    provider_call_completed: bool | None
    codex_turn_receipt_blake3: str | None
    semantic_attempt_count: int
    retry_count: int
    exception_text_recorded: bool
    receipt_blake3: str

    def __post_init__(self) -> None:
        if self.schema != "eva.premium-codex-construction-failure.v1":
            raise PremiumConstructionError("construction failure receipt schema differs")
        uuid_text(self.failure_id, label="construction failure_id")
        if not isinstance(self.lane, ConstructionLane) or self.error_code not in _FAILURE_CODES:
            raise PremiumConstructionError("construction failure classification differs")
        if self.provider_call_completed is not None and type(self.provider_call_completed) is not bool:
            raise PremiumConstructionError("provider completion marker differs")
        for digest in (
            self.source_request_blake3,
            self.request_blake3,
            self.output_schema_blake3,
            self.receipt_blake3,
        ):
            if not is_blake3(digest):
                raise PremiumConstructionError("construction failure BLAKE3 differs")
        if self.codex_turn_receipt_blake3 is not None and not is_blake3(
            self.codex_turn_receipt_blake3
        ):
            raise PremiumConstructionError("construction failure turn BLAKE3 differs")
        if (
            self.semantic_attempt_count != 1
            or self.retry_count != 0
            or self.exception_text_recorded is not False
        ):
            raise PremiumConstructionError("construction failure controls differ")
        if self.receipt_blake3 != blake3_hex(self.core()):
            raise PremiumConstructionError("construction failure receipt BLAKE3 differs")

    def core(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "failure_id": self.failure_id,
            "lane": self.lane,
            "source_request_blake3": self.source_request_blake3,
            "request_blake3": self.request_blake3,
            "output_schema_blake3": self.output_schema_blake3,
            "error_code": self.error_code,
            "provider_call_completed": self.provider_call_completed,
            "codex_turn_receipt_blake3": self.codex_turn_receipt_blake3,
            "semantic_attempt_count": self.semantic_attempt_count,
            "retry_count": self.retry_count,
            "exception_text_recorded": self.exception_text_recorded,
        }

    def to_document(self) -> dict[str, Any]:
        return {
            **canonical_value(self.core()),
            "receipt_blake3": self.receipt_blake3,
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "ConstructionFailureReceipt":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise PremiumConstructionError("construction failure receipt keys differ")
        fields = {key: value[key] for key in cls.__dataclass_fields__}
        try:
            fields["lane"] = ConstructionLane(fields["lane"])
        except (TypeError, ValueError):
            raise PremiumConstructionError("construction failure lane differs") from None
        return cls(**fields)


def verify_construction_failure_receipt(receipt: ConstructionFailureReceipt) -> None:
    if not isinstance(receipt, ConstructionFailureReceipt):
        raise PremiumConstructionError("value is not a construction failure receipt")
    if receipt.receipt_blake3 != blake3_hex(receipt.core()):
        raise PremiumConstructionError("construction failure receipt BLAKE3 differs")


@dataclass(frozen=True)
class ConstructionAttemptEvidence:
    """Immutable internal evidence for one started v3 semantic lane.

    Successful lanes retain the complete phase result, including the original
    valid Codex turn receipt and output.  Failed lanes retain the failure
    receipt plus the complete valid Codex turn receipt whenever the provider
    returned one.  Runner errors and invalid receipts deliberately carry no
    untrusted turn object.
    """

    schema: str
    lane: ConstructionLane
    status: str
    phase_result: ConstructionPhaseResult | None
    failure_receipt: ConstructionFailureReceipt | None
    codex_turn_receipt: CodexTurnReceipt | None
    evidence_blake3: str

    def __post_init__(self) -> None:
        if self.schema != "eva.premium-codex-construction-attempt-evidence.v1":
            raise PremiumConstructionError("construction attempt evidence schema differs")
        if not isinstance(self.lane, ConstructionLane):
            raise PremiumConstructionError("construction attempt evidence lane differs")
        if self.status == "succeeded":
            if not (
                isinstance(self.phase_result, ConstructionPhaseResult)
                and self.failure_receipt is None
                and self.codex_turn_receipt is None
                and self.phase_result.receipt.lane is self.lane
            ):
                raise PremiumConstructionError("successful attempt evidence differs")
        elif self.status == "failed":
            if not (
                self.phase_result is None
                and isinstance(self.failure_receipt, ConstructionFailureReceipt)
                and self.failure_receipt.lane is self.lane
            ):
                raise PremiumConstructionError("failed attempt evidence differs")
            verify_construction_failure_receipt(self.failure_receipt)
            if self.codex_turn_receipt is None:
                if self.failure_receipt.codex_turn_receipt_blake3 is not None:
                    raise PremiumConstructionError("failed attempt turn evidence is missing")
            else:
                verify_codex_turn_receipt(self.codex_turn_receipt)
                if (
                    self.failure_receipt.codex_turn_receipt_blake3
                    != self.codex_turn_receipt.receipt_blake3
                ):
                    raise PremiumConstructionError("failed attempt turn binding differs")
        else:
            raise PremiumConstructionError("construction attempt evidence status differs")
        if not is_blake3(self.evidence_blake3) or self.evidence_blake3 != blake3_hex(
            self.core()
        ):
            raise PremiumConstructionError("construction attempt evidence BLAKE3 differs")

    def core(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "lane": self.lane,
            "status": self.status,
            "phase_result": (
                None if self.phase_result is None else self.phase_result.to_document()
            ),
            "failure_receipt": (
                None
                if self.failure_receipt is None
                else self.failure_receipt.to_document()
            ),
            "codex_turn_receipt": self.codex_turn_receipt,
        }

    def to_document(self) -> dict[str, Any]:
        return {**canonical_value(self.core()), "evidence_blake3": self.evidence_blake3}

    @classmethod
    def from_success(cls, result: ConstructionPhaseResult) -> "ConstructionAttemptEvidence":
        core = {
            "schema": "eva.premium-codex-construction-attempt-evidence.v1",
            "lane": result.receipt.lane,
            "status": "succeeded",
            "phase_result": result.to_document(),
            "failure_receipt": None,
            "codex_turn_receipt": None,
        }
        return cls(
            schema=core["schema"],
            lane=core["lane"],
            status=core["status"],
            phase_result=result,
            failure_receipt=None,
            codex_turn_receipt=None,
            evidence_blake3=blake3_hex(core),
        )

    @classmethod
    def from_failure(
        cls,
        receipt: ConstructionFailureReceipt,
        codex_turn_receipt: CodexTurnReceipt | None,
    ) -> "ConstructionAttemptEvidence":
        core = {
            "schema": "eva.premium-codex-construction-attempt-evidence.v1",
            "lane": receipt.lane,
            "status": "failed",
            "phase_result": None,
            "failure_receipt": receipt.to_document(),
            "codex_turn_receipt": codex_turn_receipt,
        }
        return cls(
            schema=core["schema"],
            lane=core["lane"],
            status=core["status"],
            phase_result=None,
            failure_receipt=receipt,
            codex_turn_receipt=codex_turn_receipt,
            evidence_blake3=blake3_hex(core),
        )

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "ConstructionAttemptEvidence":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise PremiumConstructionError("construction attempt evidence keys differ")
        try:
            lane = ConstructionLane(value["lane"])
        except (TypeError, ValueError):
            raise PremiumConstructionError("construction attempt evidence lane differs") from None
        raw_phase = value["phase_result"]
        raw_failure = value["failure_receipt"]
        raw_turn = value["codex_turn_receipt"]
        if raw_phase is not None and not isinstance(raw_phase, Mapping):
            raise PremiumConstructionError("construction attempt phase evidence differs")
        if raw_failure is not None and not isinstance(raw_failure, Mapping):
            raise PremiumConstructionError("construction attempt failure evidence differs")
        if raw_turn is not None and not isinstance(raw_turn, Mapping):
            raise PremiumConstructionError("construction attempt turn evidence differs")
        return cls(
            schema=value["schema"],
            lane=lane,
            status=value["status"],
            phase_result=(
                None
                if raw_phase is None
                else ConstructionPhaseResult.from_document(raw_phase)
            ),
            failure_receipt=(
                None
                if raw_failure is None
                else ConstructionFailureReceipt.from_document(raw_failure)
            ),
            codex_turn_receipt=(
                None
                if raw_turn is None
                else codex_turn_receipt_from_document(raw_turn)
            ),
            evidence_blake3=value["evidence_blake3"],
        )


def verify_construction_attempt_evidence(
    evidence: ConstructionAttemptEvidence,
) -> None:
    if not isinstance(evidence, ConstructionAttemptEvidence):
        raise PremiumConstructionError("value is not construction attempt evidence")
    if evidence.evidence_blake3 != blake3_hex(evidence.core()):
        raise PremiumConstructionError("construction attempt evidence BLAKE3 differs")


@dataclass(frozen=True)
class ConstructionFallbackReceipt:
    """Explicit proof that a failed supplemental author was not impersonated.

    The fallback itself makes no provider call.  Its bytes are produced by the
    committed construction adapter from the already validated primary draft,
    then validated against the failed lane's *unchanged* output schema and
    legacy validator.  Publication re-derives those bytes independently.
    """

    schema: str
    fallback_id: str
    lane: ConstructionLane
    source_request_blake3: str
    source_phase_request_blake3: str
    request_blake3: str
    output_schema_blake3: str
    failed_attempt_receipt_blake3: str
    primary_phase_receipt_blake3: str
    primary_output_blake3: str
    output_blake3: str
    derivation_policy: str
    provider_call_count: int
    semantic_attempt_count: int
    retry_count: int
    legacy_validator_passed: bool
    supplemental_authoritative_for_admission: bool
    receipt_blake3: str

    def __post_init__(self) -> None:
        if self.schema != "eva.premium-codex-construction-fallback.v1":
            raise PremiumConstructionError("construction fallback schema differs")
        uuid_text(self.fallback_id, label="construction fallback_id")
        if (
            self.lane is not ConstructionLane.GEMINI_ALTERNATE
            or self.derivation_policy != _FALLBACK_POLICY
            or self.provider_call_count != 0
            or self.semantic_attempt_count != 0
            or self.retry_count != 0
            or self.legacy_validator_passed is not True
            or self.supplemental_authoritative_for_admission is not False
        ):
            raise PremiumConstructionError("construction fallback controls differ")
        if any(
            not is_blake3(value)
            for value in (
                self.source_request_blake3,
                self.source_phase_request_blake3,
                self.request_blake3,
                self.output_schema_blake3,
                self.failed_attempt_receipt_blake3,
                self.primary_phase_receipt_blake3,
                self.primary_output_blake3,
                self.output_blake3,
                self.receipt_blake3,
            )
        ):
            raise PremiumConstructionError("construction fallback BLAKE3 differs")
        if self.receipt_blake3 != blake3_hex(self.core()):
            raise PremiumConstructionError("construction fallback receipt BLAKE3 differs")

    def core(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "fallback_id": self.fallback_id,
            "lane": self.lane,
            "source_request_blake3": self.source_request_blake3,
            "source_phase_request_blake3": self.source_phase_request_blake3,
            "request_blake3": self.request_blake3,
            "output_schema_blake3": self.output_schema_blake3,
            "failed_attempt_receipt_blake3": self.failed_attempt_receipt_blake3,
            "primary_phase_receipt_blake3": self.primary_phase_receipt_blake3,
            "primary_output_blake3": self.primary_output_blake3,
            "output_blake3": self.output_blake3,
            "derivation_policy": self.derivation_policy,
            "provider_call_count": self.provider_call_count,
            "semantic_attempt_count": self.semantic_attempt_count,
            "retry_count": self.retry_count,
            "legacy_validator_passed": self.legacy_validator_passed,
            "supplemental_authoritative_for_admission": (
                self.supplemental_authoritative_for_admission
            ),
        }

    def to_document(self) -> dict[str, Any]:
        return {**canonical_value(self.core()), "receipt_blake3": self.receipt_blake3}

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "ConstructionFallbackReceipt":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise PremiumConstructionError("construction fallback receipt keys differ")
        fields = {key: value[key] for key in cls.__dataclass_fields__}
        try:
            fields["lane"] = ConstructionLane(fields["lane"])
        except (TypeError, ValueError):
            raise PremiumConstructionError("construction fallback lane differs") from None
        return cls(**fields)


@dataclass(frozen=True)
class ConstructionFallbackResult:
    receipt: ConstructionFallbackReceipt
    output: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, ConstructionFallbackReceipt):
            raise PremiumConstructionError("construction fallback receipt differs")
        output = _mapping(self.output, label="construction fallback output")
        object.__setattr__(self, "output", output)
        if self.receipt.output_blake3 != blake3_hex(output):
            raise PremiumConstructionError("construction fallback output BLAKE3 differs")


def verify_construction_fallback_result(result: ConstructionFallbackResult) -> None:
    if not isinstance(result, ConstructionFallbackResult):
        raise PremiumConstructionError("value is not a construction fallback result")
    if (
        result.receipt.receipt_blake3 != blake3_hex(result.receipt.core())
        or result.receipt.output_blake3 != blake3_hex(result.output)
    ):
        raise PremiumConstructionError("construction fallback result differs")


class ConstructionWaveError(PremiumConstructionError):
    """A fully drained parallel wave contained one or more failed attempts."""

    def __init__(
        self,
        wave: str,
        *,
        completed: Sequence[ConstructionPhaseResult],
        failures: Sequence[ConstructionFailureReceipt],
        attempt_evidence: Sequence[ConstructionAttemptEvidence],
    ) -> None:
        super().__init__(f"premium construction {wave} wave failed closed")
        self.wave = wave
        self.completed = tuple(completed)
        self.failures = tuple(failures)
        self.attempt_evidence = tuple(attempt_evidence)
        if (
            any(
                not isinstance(item, ConstructionPhaseResult)
                for item in self.completed
            )
            or any(
                not isinstance(item, ConstructionFailureReceipt)
                for item in self.failures
            )
            or any(
                not isinstance(item, ConstructionAttemptEvidence)
                for item in self.attempt_evidence
            )
        ):
            raise PremiumConstructionError("construction wave evidence type differs")
        completed_by_lane = {item.receipt.lane: item for item in self.completed}
        failures_by_lane = {item.lane: item for item in self.failures}
        evidence_by_lane = {item.lane: item for item in self.attempt_evidence}
        if not (
            len(completed_by_lane) == len(self.completed)
            and len(failures_by_lane) == len(self.failures)
            and len(evidence_by_lane) == len(self.attempt_evidence)
            and set(completed_by_lane).isdisjoint(failures_by_lane)
            and set(evidence_by_lane) == set(completed_by_lane) | set(failures_by_lane)
        ):
            raise PremiumConstructionError("construction wave attempt inventory differs")
        for lane, evidence in evidence_by_lane.items():
            expected_phase = completed_by_lane.get(lane)
            expected_failure = failures_by_lane.get(lane)
            if not (
                (
                    expected_phase is not None
                    and evidence.phase_result == expected_phase
                    and evidence.failure_receipt is None
                )
                or (
                    expected_failure is not None
                    and evidence.phase_result is None
                    and evidence.failure_receipt == expected_failure
                )
            ):
                raise PremiumConstructionError("construction wave attempt binding differs")


class _LaneAttemptError(Exception):
    def __init__(self, evidence: ConstructionAttemptEvidence) -> None:
        super().__init__("premium construction lane failed closed")
        self.evidence = evidence
        assert evidence.failure_receipt is not None
        self.receipt = evidence.failure_receipt


@dataclass(frozen=True)
class PremiumConstructionResult:
    schema: str
    construction_run_id: str
    source_id: str
    source_request_blake3: str
    phases: tuple[ConstructionPhaseResult, ...]
    canonical_output_blake3: str
    provider_call_count: int
    wave_widths: tuple[int, ...]
    semantic_retry_count: int
    legacy_manifest_emitted: bool
    legacy_manifest_compatibility_claimed: bool
    receipt_blake3: str
    supplemental_failures: tuple[ConstructionFailureReceipt, ...] = ()
    fallbacks: tuple[ConstructionFallbackResult, ...] = ()
    author_quorum_policy: str | None = None
    critic_quorum_policy: str | None = None
    selected_critic_receipt_blake3s: Mapping[str, str] | None = None
    attempt_evidence: tuple[ConstructionAttemptEvidence, ...] = ()

    def __post_init__(self) -> None:
        if self.schema not in {
            _RESULT_SCHEMA_V1,
            _RESULT_SCHEMA_V2,
            _RESULT_SCHEMA_V3,
            _RESULT_SCHEMA_V4,
        }:
            raise PremiumConstructionError("premium construction result schema differs")
        uuid_text(self.construction_run_id, label="construction_run_id")
        _nonempty(self.source_id, label="construction source_id")
        if not is_blake3(self.source_request_blake3):
            raise PremiumConstructionError("construction source BLAKE3 differs")
        phases = tuple(self.phases)
        object.__setattr__(self, "phases", phases)
        failures = tuple(self.supplemental_failures)
        fallbacks = tuple(self.fallbacks)
        object.__setattr__(self, "supplemental_failures", failures)
        object.__setattr__(self, "fallbacks", fallbacks)
        attempt_evidence = tuple(self.attempt_evidence)
        object.__setattr__(self, "attempt_evidence", attempt_evidence)
        selections = (
            None
            if self.selected_critic_receipt_blake3s is None
            else MappingProxyType(
                dict(sorted(self.selected_critic_receipt_blake3s.items()))
            )
        )
        object.__setattr__(self, "selected_critic_receipt_blake3s", selections)
        if (
            any(not isinstance(phase, ConstructionPhaseResult) for phase in phases)
            or any(
                not isinstance(failure, ConstructionFailureReceipt)
                for failure in failures
            )
            or any(
                not isinstance(fallback, ConstructionFallbackResult)
                for fallback in fallbacks
            )
            or any(
                not isinstance(evidence, ConstructionAttemptEvidence)
                for evidence in attempt_evidence
            )
        ):
            raise PremiumConstructionError("premium construction evidence type differs")
        phase_lanes = tuple(phase.receipt.lane for phase in phases)
        if self.schema == _RESULT_SCHEMA_V1:
            if (
                phase_lanes != LANE_ORDER
                or failures
                or fallbacks
                or self.author_quorum_policy is not None
                or self.critic_quorum_policy is not None
                or selections is not None
                or attempt_evidence
            ):
                raise PremiumConstructionError("premium construction v1 schedule differs")
        elif self.schema == _RESULT_SCHEMA_V2:
            healthy = phase_lanes == LANE_ORDER and not failures and not fallbacks
            degraded = (
                phase_lanes
                == (
                    ConstructionLane.OPUS5_DRAFT,
                    ConstructionLane.OPUS48_CRITIQUE,
                    ConstructionLane.GPT56_COMPARISON,
                    ConstructionLane.OPUS5_REVISION,
                )
                and len(failures) == 1
                and failures[0].lane is ConstructionLane.GEMINI_ALTERNATE
                and len(fallbacks) == 1
                and fallbacks[0].receipt.lane is ConstructionLane.GEMINI_ALTERNATE
            )
            if (
                not (healthy or degraded)
                or self.author_quorum_policy != _AUTHOR_QUORUM_POLICY
                or self.critic_quorum_policy is not None
                or selections is not None
                or attempt_evidence
            ):
                raise PremiumConstructionError("premium construction v2 quorum differs")
        elif self.schema == _RESULT_SCHEMA_V3:
            success_lanes = set(phase_lanes)
            failure_lanes = {failure.lane for failure in failures}
            if len(failure_lanes) != len(failures):
                raise PremiumConstructionError("premium construction v3 failure lane duplicated")
            attempted = success_lanes | failure_lanes
            selected = dict(selections or {})
            by_lane = {phase.receipt.lane: phase for phase in phases}
            gemini_failed = ConstructionLane.GEMINI_ALTERNATE in failure_lanes
            expected_selected = {
                "critique": (
                    by_lane.get(ConstructionLane.OPUS48_CRITIQUE)
                    or by_lane.get(ConstructionLane.OPUS5_CRITIQUE_BACKUP)
                ),
                "comparison": (
                    by_lane.get(ConstructionLane.GPT56_COMPARISON)
                    or by_lane.get(ConstructionLane.GEMINI_COMPARISON_BACKUP)
                ),
            }
            evidence_by_lane = {evidence.lane: evidence for evidence in attempt_evidence}
            if not (
                len(success_lanes) == len(phases)
                and success_lanes.isdisjoint(failure_lanes)
                and phase_lanes
                == tuple(lane for lane in V3_LANE_ORDER if lane in success_lanes)
                and attempted == set(V3_LANE_ORDER)
                and ConstructionLane.OPUS5_DRAFT in success_lanes
                and ConstructionLane.OPUS5_REVISION in success_lanes
                and all(value is not None for value in expected_selected.values())
                and (gemini_failed == (ConstructionLane.GEMINI_ALTERNATE not in success_lanes))
                and (len(fallbacks) == 1) == gemini_failed
                and (
                    not fallbacks
                    or fallbacks[0].receipt.lane is ConstructionLane.GEMINI_ALTERNATE
                )
                and self.author_quorum_policy == _AUTHOR_QUORUM_POLICY
                and self.critic_quorum_policy == _CRITIC_QUORUM_POLICY
                and set(selected) == {"critique", "comparison"}
                and all(is_blake3(value) for value in selected.values())
                and selected
                == {
                    role: phase.receipt.receipt_blake3
                    for role, phase in expected_selected.items()
                    if phase is not None
                }
                and len(evidence_by_lane) == len(attempt_evidence) == len(V3_LANE_ORDER)
                and tuple(evidence.lane for evidence in attempt_evidence) == V3_LANE_ORDER
                and all(
                    (
                        evidence.phase_result == by_lane.get(evidence.lane)
                        and evidence.failure_receipt is None
                    )
                    or (
                        evidence.phase_result is None
                        and evidence.failure_receipt
                        == next(
                            (
                                failure
                                for failure in failures
                                if failure.lane is evidence.lane
                            ),
                            None,
                        )
                    )
                    for evidence in attempt_evidence
                )
            ):
                raise PremiumConstructionError("premium construction v3 quorum differs")
        else:
            success_lanes = set(phase_lanes)
            failure_lanes = {failure.lane for failure in failures}
            attempted = success_lanes | failure_lanes
            by_lane = {phase.receipt.lane: phase for phase in phases}
            selected = dict(selections or {})
            required = {
                ConstructionLane.OPUS5_DRAFT,
                ConstructionLane.OPUS5_CRITIQUE_BACKUP,
                ConstructionLane.GEMINI_COMPARISON_BACKUP,
                ConstructionLane.OPUS5_REVISION,
            }
            gemini_failed = ConstructionLane.GEMINI_ALTERNATE in failure_lanes
            evidence_by_lane = {evidence.lane: evidence for evidence in attempt_evidence}
            expected_selected = {
                "critique": by_lane.get(ConstructionLane.OPUS5_CRITIQUE_BACKUP),
                "comparison": by_lane.get(
                    ConstructionLane.GEMINI_COMPARISON_BACKUP
                ),
            }
            if not (
                len(failure_lanes) == len(failures)
                and len(success_lanes) == len(phases)
                and success_lanes.isdisjoint(failure_lanes)
                and phase_lanes
                == tuple(lane for lane in V4_LANE_ORDER if lane in success_lanes)
                and attempted == set(V4_LANE_ORDER)
                and required <= success_lanes
                and failure_lanes
                <= {
                    ConstructionLane.GEMINI_ALTERNATE,
                    ConstructionLane.OPUS48_CRITIQUE,
                    ConstructionLane.GPT56_COMPARISON,
                }
                and (len(fallbacks) == 1) == gemini_failed
                and (
                    not fallbacks
                    or fallbacks[0].receipt.lane
                    is ConstructionLane.GEMINI_ALTERNATE
                )
                and self.author_quorum_policy == _AUTHOR_QUORUM_POLICY
                and self.critic_quorum_policy == _CRITIC_AUTHORITY_POLICY
                and set(selected) == {"critique", "comparison"}
                and selected
                == {
                    role: phase.receipt.receipt_blake3
                    for role, phase in expected_selected.items()
                    if phase is not None
                }
                and len(evidence_by_lane) == len(attempt_evidence) == len(V4_LANE_ORDER)
                and tuple(evidence.lane for evidence in attempt_evidence)
                == V4_LANE_ORDER
                and all(
                    (
                        evidence.phase_result == by_lane.get(evidence.lane)
                        and evidence.failure_receipt is None
                    )
                    or (
                        evidence.phase_result is None
                        and evidence.failure_receipt
                        == next(
                            (
                                failure
                                for failure in failures
                                if failure.lane is evidence.lane
                            ),
                            None,
                        )
                    )
                    for evidence in attempt_evidence
                )
            ):
                raise PremiumConstructionError("premium construction v4 authority differs")
        if any(
            phase.receipt.source_request_blake3 != self.source_request_blake3
            for phase in phases
        ):
            raise PremiumConstructionError("premium construction source binding differs")
        if any(
            failure.source_request_blake3 != self.source_request_blake3
            for failure in failures
        ) or any(
            fallback.receipt.source_request_blake3 != self.source_request_blake3
            for fallback in fallbacks
        ):
            raise PremiumConstructionError("premium construction supplemental source differs")
        if (
            self.canonical_output_blake3 != blake3_hex(phases[-1].output)
            or self.provider_call_count
            != (7 if self.schema in {_RESULT_SCHEMA_V3, _RESULT_SCHEMA_V4} else 5)
            or self.wave_widths
            != (
                (2, 4, 1)
                if self.schema in {_RESULT_SCHEMA_V3, _RESULT_SCHEMA_V4}
                else (2, 2, 1)
            )
            or self.semantic_retry_count != 0
            or self.legacy_manifest_emitted is not False
            or self.legacy_manifest_compatibility_claimed is not False
        ):
            raise PremiumConstructionError("premium construction controls differ")
        if self.receipt_blake3 != blake3_hex(self.core()):
            raise PremiumConstructionError("premium construction receipt BLAKE3 differs")

    @property
    def canonical_output(self) -> Mapping[str, JsonValue]:
        return self.phases[-1].output

    def core(self) -> Mapping[str, Any]:
        core = {
            "schema": self.schema,
            "construction_run_id": self.construction_run_id,
            "source_id": self.source_id,
            "source_request_blake3": self.source_request_blake3,
            "phase_receipt_blake3s": tuple(
                phase.receipt.receipt_blake3 for phase in self.phases
            ),
            "canonical_output_blake3": self.canonical_output_blake3,
            "provider_call_count": self.provider_call_count,
            "wave_widths": self.wave_widths,
            "semantic_retry_count": self.semantic_retry_count,
            "legacy_manifest_emitted": self.legacy_manifest_emitted,
            "legacy_manifest_compatibility_claimed": self.legacy_manifest_compatibility_claimed,
        }
        if self.schema in {_RESULT_SCHEMA_V2, _RESULT_SCHEMA_V3, _RESULT_SCHEMA_V4}:
            core.update(
                {
                    "supplemental_failure_receipt_blake3s": tuple(
                        failure.receipt_blake3 for failure in self.supplemental_failures
                    ),
                    "fallback_receipt_blake3s": tuple(
                        fallback.receipt.receipt_blake3 for fallback in self.fallbacks
                    ),
                    "author_quorum_policy": self.author_quorum_policy,
                    "successful_phase_count": len(self.phases),
                    "supplemental_failure_count": len(self.supplemental_failures),
                    "fallback_count": len(self.fallbacks),
                }
            )
        if self.schema in {_RESULT_SCHEMA_V3, _RESULT_SCHEMA_V4}:
            core.update(
                {
                    "critic_quorum_policy": self.critic_quorum_policy,
                    "selected_critic_receipt_blake3s": (
                        self.selected_critic_receipt_blake3s
                    ),
                    "attempt_evidence_blake3s": tuple(
                        evidence.evidence_blake3 for evidence in self.attempt_evidence
                    ),
                }
            )
        return core


def verify_premium_construction_result(result: PremiumConstructionResult) -> None:
    if not isinstance(result, PremiumConstructionResult):
        raise PremiumConstructionError("value is not a premium construction result")
    by_lane = {phase.receipt.lane: phase for phase in result.phases}
    if len(by_lane) != len(result.phases):
        raise PremiumConstructionError("construction phase lane is duplicated")
    expected_authors: Mapping[str, str] = MappingProxyType({})
    primary = by_lane.get(ConstructionLane.OPUS5_DRAFT)
    if primary is None:
        raise PremiumConstructionError("primary Opus author phase is required")
    for failure in result.supplemental_failures:
        verify_construction_failure_receipt(failure)
    for evidence in result.attempt_evidence:
        verify_construction_attempt_evidence(evidence)
    if ConstructionLane.GEMINI_ALTERNATE in by_lane:
        alternate_blake3 = by_lane[
            ConstructionLane.GEMINI_ALTERNATE
        ].receipt.output_blake3
    else:
        gemini_failures = tuple(
            failure
            for failure in result.supplemental_failures
            if failure.lane is ConstructionLane.GEMINI_ALTERNATE
        )
        if len(gemini_failures) != 1 or len(result.fallbacks) != 1:
            raise PremiumConstructionError("supplemental author proof differs")
        failure = gemini_failures[0]
        fallback = result.fallbacks[0]
        verify_construction_failure_receipt(failure)
        verify_construction_fallback_result(fallback)
        receipt = fallback.receipt
        if not (
            failure.lane is ConstructionLane.GEMINI_ALTERNATE
            and receipt.failed_attempt_receipt_blake3 == failure.receipt_blake3
            and receipt.primary_phase_receipt_blake3
            == primary.receipt.receipt_blake3
            and receipt.primary_output_blake3 == primary.receipt.output_blake3
            and receipt.source_request_blake3 == failure.source_request_blake3
            and receipt.request_blake3 == failure.request_blake3
            and receipt.output_schema_blake3 == failure.output_schema_blake3
        ):
            raise PremiumConstructionError("supplemental fallback binding differs")
        alternate_blake3 = receipt.output_blake3
    expected_critics = MappingProxyType(
        {
            ConstructionLane.OPUS5_DRAFT.value: primary.receipt.output_blake3,
            ConstructionLane.GEMINI_ALTERNATE.value: alternate_blake3,
        }
    )
    if result.schema in {_RESULT_SCHEMA_V3, _RESULT_SCHEMA_V4}:
        selections = dict(result.selected_critic_receipt_blake3s or {})
        receipts = {
            phase.receipt.receipt_blake3: phase for phase in result.phases
        }
        try:
            selected_critique = receipts[selections["critique"]]
            selected_comparison = receipts[selections["comparison"]]
        except KeyError:
            raise PremiumConstructionError("selected critic receipt differs") from None
        expected_critique_lanes = (
            (ConstructionLane.OPUS5_CRITIQUE_BACKUP,)
            if result.schema == _RESULT_SCHEMA_V4
            else CRITIQUE_QUORUM_LANES
        )
        expected_comparison_lanes = (
            (ConstructionLane.GEMINI_COMPARISON_BACKUP,)
            if result.schema == _RESULT_SCHEMA_V4
            else COMPARISON_QUORUM_LANES
        )
        if not (
            selected_critique.receipt.lane in expected_critique_lanes
            and selected_comparison.receipt.lane in expected_comparison_lanes
        ):
            raise PremiumConstructionError("selected critic role differs")
    else:
        selected_critique = by_lane[ConstructionLane.OPUS48_CRITIQUE]
        selected_comparison = by_lane[ConstructionLane.GPT56_COMPARISON]
    expected_revision = MappingProxyType(
        {
            **dict(expected_critics),
            ConstructionLane.OPUS48_CRITIQUE.value: (
                selected_critique.receipt.output_blake3
            ),
            ConstructionLane.GPT56_COMPARISON.value: (
                selected_comparison.receipt.output_blake3
            ),
        }
    )
    for phase in result.phases:
        verify_codex_turn_receipt(phase.codex_turn_receipt)
        if phase.receipt.receipt_blake3 != blake3_hex(phase.receipt.core()):
            raise PremiumConstructionError("construction phase receipt BLAKE3 differs")
        expected = (
            expected_authors
            if phase.receipt.lane in AUTHOR_LANES
            else expected_critics
            if phase.receipt.lane in (
                (V4_CRITIC_LANES if result.schema == _RESULT_SCHEMA_V4 else CRITIC_QUORUM_LANES)
                if result.schema in {_RESULT_SCHEMA_V3, _RESULT_SCHEMA_V4}
                else CRITIC_LANES
            )
            else expected_revision
        )
        if canonical_value(phase.receipt.dependency_blake3s) != canonical_value(expected):
            raise PremiumConstructionError("construction phase dependency binding differs")
    if result.receipt_blake3 != blake3_hex(result.core()):
        raise PremiumConstructionError("premium construction receipt BLAKE3 differs")


@dataclass(frozen=True)
class ConstructionPublicationReceipt:
    """Receipt returned by an append-only caller-owned output sink."""

    schema: str
    publication_id: str
    sink_name: str
    construction_receipt_blake3: str
    artifact_blake3: str
    legacy_manifest_compatibility_verified: bool
    metadata: Mapping[str, JsonValue]
    receipt_blake3: str

    def __post_init__(self) -> None:
        if self.schema != "eva.premium-codex-construction-publication.v1":
            raise PremiumConstructionError("construction publication schema differs")
        uuid_text(self.publication_id, label="construction publication_id")
        _nonempty(self.sink_name, label="construction sink name")
        if any(
            not is_blake3(value)
            for value in (
                self.construction_receipt_blake3,
                self.artifact_blake3,
                self.receipt_blake3,
            )
        ):
            raise PremiumConstructionError("construction publication BLAKE3 differs")
        if type(self.legacy_manifest_compatibility_verified) is not bool:
            raise PremiumConstructionError("legacy compatibility marker differs")
        metadata = _mapping(self.metadata, label="construction publication metadata")
        object.__setattr__(self, "metadata", metadata)
        if self.receipt_blake3 != blake3_hex(self.core()):
            raise PremiumConstructionError("construction publication receipt BLAKE3 differs")

    def core(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "publication_id": self.publication_id,
            "sink_name": self.sink_name,
            "construction_receipt_blake3": self.construction_receipt_blake3,
            "artifact_blake3": self.artifact_blake3,
            "legacy_manifest_compatibility_verified": self.legacy_manifest_compatibility_verified,
            "metadata": self.metadata,
        }


class ConstructionResultSinkPort(Protocol):
    """Append-only publication boundary; implementations must not replace files."""

    def publish(
        self, result: PremiumConstructionResult
    ) -> ConstructionPublicationReceipt: ...


@dataclass(frozen=True)
class PublishedConstructionResult:
    result: PremiumConstructionResult
    publication: ConstructionPublicationReceipt

    def __post_init__(self) -> None:
        verify_premium_construction_result(self.result)
        if (
            not isinstance(self.publication, ConstructionPublicationReceipt)
            or self.publication.construction_receipt_blake3 != self.result.receipt_blake3
        ):
            raise PremiumConstructionError("construction publication binding differs")


@dataclass(frozen=True)
class ConstructionJob:
    source: FrozenConstructionSource
    adapter: ConstructionContractAdapterPort


def _strict_json_object(value: str) -> Mapping[str, JsonValue]:
    def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = child
        return result

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=strict_pairs,
            parse_constant=lambda _token: (_ for _ in ()).throw(
                ValueError("non-finite JSON constant")
            ),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        raise PremiumConstructionError("structured output is not strict JSON") from None
    return _mapping(parsed, label="structured construction output")


def _balanced_json_objects(value: str) -> tuple[str, ...]:
    """Return top-level brace spans without interpreting prose or strings."""

    spans: list[str] = []
    start: int | None = None
    depth = 0
    quoted = False
    escaped = False
    for index, character in enumerate(value):
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"' and depth:
            quoted = True
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(value[start : index + 1])
                start = None
    return tuple(spans)


def extract_unique_schema_valid_payload(
    raw_response: str,
    output_schema: Mapping[str, Any],
) -> Mapping[str, JsonValue]:
    """Extract exactly one distinct schema-valid object from retained raw text.

    Candidate discovery is deterministic and deliberately narrow: the whole
    response, fenced blocks, and top-level balanced object spans.  Parsed
    objects are deduplicated by canonical JSON bytes.  Zero or multiple
    distinct schema-valid objects fail closed.  The raw response is never
    rewritten and remains present in the Codex turn receipt.
    """

    if not isinstance(raw_response, str) or not raw_response:
        raise PremiumConstructionError("structured output is absent")
    schema = canonical_value(output_schema)
    Draft202012Validator.check_schema(schema)
    candidates = [raw_response.strip()]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json)?\s*\n?(.*?)```", raw_response, flags=re.IGNORECASE | re.DOTALL
        )
    )
    candidates.extend(_balanced_json_objects(raw_response))
    valid: dict[bytes, Mapping[str, JsonValue]] = {}
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = _strict_json_object(candidate)
            Draft202012Validator(schema).validate(canonical_value(parsed))
        except (PremiumConstructionError, ValidationError):
            continue
        valid[canonical_json_bytes(parsed)] = parsed
    if len(valid) != 1:
        raise PremiumConstructionError(
            "response must contain exactly one distinct schema-valid JSON object"
        )
    return next(iter(valid.values()))


class PremiumCodexConstructionOrchestrator:
    """Execute the prospective fixed seven-call premium schedule."""

    def __init__(
        self,
        runner: _CodexTurnRunnerPort,
        routes: PremiumConstructionRoutes,
        *,
        id_factory: RuntimeIdFactory | None = None,
        result_schema: str = PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
        response_extraction_policy: str = STRICT_RESPONSE_EXTRACTION_POLICY_V1,
    ) -> None:
        if not callable(getattr(runner, "run_once", None)):
            raise PremiumConstructionError("Codex construction runner must expose run_once")
        if not isinstance(routes, PremiumConstructionRoutes):
            raise PremiumConstructionError("premium construction routes differ")
        self._runner = runner
        self._routes = routes
        self._ids = id_factory or RandomUUIDFactory()
        if result_schema not in {_RESULT_SCHEMA_V3, _RESULT_SCHEMA_V4}:
            raise PremiumConstructionError("construction orchestrator result schema differs")
        self._result_schema = result_schema
        if response_extraction_policy not in {
            STRICT_RESPONSE_EXTRACTION_POLICY_V1,
            UNIQUE_SCHEMA_RESPONSE_EXTRACTION_POLICY_V2,
        }:
            raise PremiumConstructionError("construction response extraction policy differs")
        self._response_extraction_policy = response_extraction_policy

    def _failure(
        self,
        request: ConstructionTurnRequest,
        *,
        error_code: str,
        provider_call_completed: bool | None,
        turn_receipt: CodexTurnReceipt | None,
    ) -> _LaneAttemptError:
        turn_receipt_blake3 = (
            None if turn_receipt is None else turn_receipt.receipt_blake3
        )
        core = {
            "schema": "eva.premium-codex-construction-failure.v1",
            "failure_id": self._ids.new("premium-construction-failure"),
            "lane": request.lane,
            "source_request_blake3": request.source_request_blake3,
            "request_blake3": request.request_blake3,
            "output_schema_blake3": request.source_output_schema_blake3,
            "error_code": error_code,
            "provider_call_completed": provider_call_completed,
            "codex_turn_receipt_blake3": turn_receipt_blake3,
            "semantic_attempt_count": 1,
            "retry_count": 0,
            "exception_text_recorded": False,
        }
        receipt = ConstructionFailureReceipt(
            **core, receipt_blake3=blake3_hex(core)
        )
        return _LaneAttemptError(
            ConstructionAttemptEvidence.from_failure(receipt, turn_receipt)
        )

    def _validate_prepared(
        self,
        source: FrozenConstructionSource,
        request: ConstructionTurnRequest,
        *,
        lane: ConstructionLane,
        dependencies: Mapping[str, str],
    ) -> None:
        if not isinstance(request, ConstructionTurnRequest) or request.lane is not lane:
            raise PremiumConstructionError("construction adapter returned the wrong lane")
        if request.source_request_blake3 != source.request_blake3:
            raise PremiumConstructionError("construction adapter source binding differs")
        route = (
            self._routes.opus5
            if self._result_schema == _RESULT_SCHEMA_V4
            and lane is ConstructionLane.GEMINI_COMPARISON_BACKUP
            else self._routes.for_lane(lane)
        )
        if request.options.model != route.model or request.options.provider != route.provider:
            raise PremiumConstructionError("construction adapter model route differs")
        if canonical_value(request.dependencies) != canonical_value(dependencies):
            raise PremiumConstructionError("construction adapter dependency binding differs")
        if request.turn_input.output_schema is None or blake3_hex(
            request.turn_input.output_schema
        ) != request.source_output_schema_blake3:
            raise PremiumConstructionError("construction adapter mutated output_schema")

    def _run_lane(
        self,
        source: FrozenConstructionSource,
        adapter: ConstructionContractAdapterPort,
        request: ConstructionTurnRequest,
    ) -> ConstructionPhaseResult:
        schema_before = blake3_hex(request.turn_input.output_schema)
        try:
            exact_run = getattr(self._runner, "run_construction_once", None)
            turn_receipt = (
                exact_run(request)
                if callable(exact_run)
                else self._runner.run_once(request.options, request.turn_input)
            )
        except Exception:
            raise self._failure(
                request,
                error_code="runner_error",
                provider_call_completed=None,
                turn_receipt=None,
            ) from None
        try:
            verify_codex_turn_receipt(turn_receipt)
        except Exception:
            raise self._failure(
                request,
                error_code="invalid_turn_receipt",
                provider_call_completed=True,
                turn_receipt=None,
            ) from None
        turn_digest = turn_receipt.receipt_blake3
        if not (
            turn_receipt.role is request.options.role
            and turn_receipt.model == request.options.model
            and turn_receipt.provider == request.options.provider
            and turn_receipt.sandbox is request.options.sandbox
            and turn_receipt.thread_resumed is False
        ):
            raise self._failure(
                request,
                error_code="turn_identity_mismatch",
                provider_call_completed=True,
                turn_receipt=turn_receipt,
            )
        if turn_receipt.status != "completed":
            raise self._failure(
                request,
                error_code="terminal_status",
                provider_call_completed=True,
                turn_receipt=turn_receipt,
            )
        if not isinstance(turn_receipt.final_response, str) or not turn_receipt.final_response:
            raise self._failure(
                request,
                error_code="missing_structured_output",
                provider_call_completed=True,
                turn_receipt=turn_receipt,
            )
        try:
            assert request.turn_input.output_schema is not None
            output = (
                extract_unique_schema_valid_payload(
                    turn_receipt.final_response,
                    request.turn_input.output_schema,
                )
                if self._response_extraction_policy
                == UNIQUE_SCHEMA_RESPONSE_EXTRACTION_POLICY_V2
                else _strict_json_object(turn_receipt.final_response)
            )
        except PremiumConstructionError:
            raise self._failure(
                request,
                error_code="strict_json",
                provider_call_completed=True,
                turn_receipt=turn_receipt,
            ) from None
        try:
            assert request.turn_input.output_schema is not None
            # ``CodexTurnInput`` stores immutable mapping proxies and tuples.
            # jsonschema expects ordinary JSON containers, so thaw a canonical
            # validation view while retaining the committed source object.
            validation_schema = canonical_value(request.turn_input.output_schema)
            validation_output = canonical_value(output)
            Draft202012Validator.check_schema(validation_schema)
            Draft202012Validator(validation_schema).validate(validation_output)
        except (SchemaError, ValidationError):
            raise self._failure(
                request,
                error_code="output_schema",
                provider_call_completed=True,
                turn_receipt=turn_receipt,
            ) from None
        try:
            adapter.validate_output(source, request, output)
        except Exception:
            raise self._failure(
                request,
                error_code="adapter_validation",
                provider_call_completed=True,
                turn_receipt=turn_receipt,
            ) from None
        schema_after = blake3_hex(request.turn_input.output_schema)
        if schema_before != schema_after or schema_after != request.source_output_schema_blake3:
            raise self._failure(
                request,
                error_code="output_schema",
                provider_call_completed=True,
                turn_receipt=turn_receipt,
            )
        phase_core = {
            "schema": "eva.premium-codex-construction-phase.v1",
            "attempt_id": self._ids.new("premium-construction-attempt"),
            "lane": request.lane,
            "source_request_blake3": request.source_request_blake3,
            "source_phase_request_blake3": request.source_phase_request_blake3,
            "request_blake3": request.request_blake3,
            "output_schema_blake3": request.source_output_schema_blake3,
            "dependency_blake3s": request.dependencies,
            "model": request.options.model,
            "provider": request.options.provider,
            "role": request.options.role,
            "sandbox": request.options.sandbox,
            "codex_turn_receipt_blake3": turn_digest,
            "output_blake3": blake3_hex(output),
            "semantic_attempt_count": 1,
            "retry_count": 0,
            "output_schema_unchanged": True,
        }
        phase_receipt = ConstructionPhaseReceipt(
            **phase_core, receipt_blake3=blake3_hex(phase_core)
        )
        return ConstructionPhaseResult(phase_receipt, output, turn_receipt)

    def _run_wave(
        self,
        wave: str,
        source: FrozenConstructionSource,
        adapter: ConstructionContractAdapterPort,
        requests: Sequence[ConstructionTurnRequest],
    ) -> tuple[ConstructionPhaseResult, ...]:
        if len(requests) not in {2, 4}:
            raise PremiumConstructionError(
                f"construction {wave} wave must contain two or four lanes"
            )
        completed: list[ConstructionPhaseResult] = []
        failures: list[ConstructionFailureReceipt] = []
        attempt_evidence: list[ConstructionAttemptEvidence] = []
        with ThreadPoolExecutor(
            max_workers=len(requests), thread_name_prefix=f"eva-construction-{wave}"
        ) as executor:
            futures = tuple(
                executor.submit(self._run_lane, source, adapter, request)
                for request in requests
            )
            # All futures are already submitted.  Resolve in canonical lane
            # order and let the context manager drain the peer before raising.
            for future in futures:
                try:
                    result = future.result()
                    completed.append(result)
                    attempt_evidence.append(ConstructionAttemptEvidence.from_success(result))
                except _LaneAttemptError as exc:
                    failures.append(exc.receipt)
                    attempt_evidence.append(exc.evidence)
        if failures:
            raise ConstructionWaveError(
                wave,
                completed=completed,
                failures=failures,
                attempt_evidence=attempt_evidence,
            )
        return tuple(completed)

    def _resolve_supplemental_author(
        self,
        source: FrozenConstructionSource,
        adapter: ConstructionContractAdapterPort,
        request: ConstructionTurnRequest,
        *,
        primary: ConstructionPhaseResult,
        failure: ConstructionFailureReceipt,
    ) -> ConstructionFallbackResult:
        """Derive and validate a zero-call alias without inventing a phase."""

        if not (
            request.lane is ConstructionLane.GEMINI_ALTERNATE
            and primary.receipt.lane is ConstructionLane.OPUS5_DRAFT
            and failure.lane is ConstructionLane.GEMINI_ALTERNATE
            and failure.source_request_blake3 == request.source_request_blake3
            and failure.request_blake3 == request.request_blake3
            and failure.output_schema_blake3 == request.source_output_schema_blake3
        ):
            raise PremiumConstructionError("supplemental author fallback input differs")
        output = _mapping(
            adapter.resolve_supplemental_author_fallback(
                source,
                request,
                primary_draft=primary.output,
                failure=failure,
            ),
            label="supplemental author fallback output",
        )
        try:
            assert request.turn_input.output_schema is not None
            validation_schema = canonical_value(request.turn_input.output_schema)
            Draft202012Validator.check_schema(validation_schema)
            Draft202012Validator(validation_schema).validate(canonical_value(output))
            # This is deliberately the ordinary adapter validator.  The
            # legacy adapter therefore executes the same wrapped draft
            # compiler used for an actual Gemini success.
            adapter.validate_output(source, request, output)
        except Exception as exc:
            raise PremiumConstructionError(
                "supplemental author fallback failed the unchanged validator"
            ) from exc
        core = {
            "schema": "eva.premium-codex-construction-fallback.v1",
            "fallback_id": self._ids.new("premium-construction-fallback"),
            "lane": ConstructionLane.GEMINI_ALTERNATE,
            "source_request_blake3": request.source_request_blake3,
            "source_phase_request_blake3": request.source_phase_request_blake3,
            "request_blake3": request.request_blake3,
            "output_schema_blake3": request.source_output_schema_blake3,
            "failed_attempt_receipt_blake3": failure.receipt_blake3,
            "primary_phase_receipt_blake3": primary.receipt.receipt_blake3,
            "primary_output_blake3": primary.receipt.output_blake3,
            "output_blake3": blake3_hex(output),
            "derivation_policy": _FALLBACK_POLICY,
            "provider_call_count": 0,
            "semantic_attempt_count": 0,
            "retry_count": 0,
            "legacy_validator_passed": True,
            "supplemental_authoritative_for_admission": False,
        }
        return ConstructionFallbackResult(
            ConstructionFallbackReceipt(
                **core, receipt_blake3=blake3_hex(core)
            ),
            output,
        )

    @staticmethod
    def _require_pair(
        values: Sequence[ConstructionTurnRequest], expected: Sequence[ConstructionLane]
    ) -> tuple[ConstructionTurnRequest, ConstructionTurnRequest]:
        requests = tuple(values)
        if len(requests) != 2 or tuple(request.lane for request in requests) != tuple(expected):
            raise PremiumConstructionError("construction adapter fan-out order differs")
        return requests[0], requests[1]

    @staticmethod
    def _require_critic_quartet(
        values: Sequence[ConstructionTurnRequest],
    ) -> tuple[
        ConstructionTurnRequest,
        ConstructionTurnRequest,
        ConstructionTurnRequest,
        ConstructionTurnRequest,
    ]:
        requests = tuple(values)
        if (
            len(requests) != 4
            or tuple(request.lane for request in requests) != CRITIC_QUORUM_LANES
        ):
            raise PremiumConstructionError("construction critic quorum order differs")
        return requests  # type: ignore[return-value]

    @staticmethod
    def _require_isolated_roots(requests: Sequence[ConstructionTurnRequest]) -> None:
        roots = [request.options.cwd for request in requests]
        if len(roots) != len(set(roots)):
            raise PremiumConstructionError(
                "construction lanes require isolated workspace roots"
            )

    def _construct_v3(
        self,
        source: FrozenConstructionSource,
        adapter: ConstructionContractAdapterPort,
    ) -> PremiumConstructionResult:
        if not isinstance(source, FrozenConstructionSource):
            raise PremiumConstructionError("frozen construction source differs")
        for method in (
            "prepare_authors",
            "prepare_critics",
            "prepare_critic_quorum",
            "prepare_revision",
            "validate_output",
            "resolve_supplemental_author_fallback",
        ):
            if not callable(getattr(adapter, method, None)):
                raise PremiumConstructionError("construction adapter port is incomplete")

        authors = self._require_pair(
            adapter.prepare_authors(source, self._routes), AUTHOR_LANES
        )
        for request, lane in zip(authors, AUTHOR_LANES, strict=True):
            self._validate_prepared(source, request, lane=lane, dependencies={})
        self._require_isolated_roots(authors)
        supplemental_failures: tuple[ConstructionFailureReceipt, ...] = ()
        fallbacks: tuple[ConstructionFallbackResult, ...] = ()
        try:
            author_results = self._run_wave("author", source, adapter, authors)
            author_attempt_evidence = tuple(
                ConstructionAttemptEvidence.from_success(result)
                for result in author_results
            )
        except ConstructionWaveError as author_error:
            completed = tuple(author_error.completed)
            failures = tuple(author_error.failures)
            author_attempt_evidence = tuple(author_error.attempt_evidence)
            if not (
                len(completed) == 1
                and completed[0].receipt.lane is ConstructionLane.OPUS5_DRAFT
                and len(failures) == 1
                and failures[0].lane is ConstructionLane.GEMINI_ALTERNATE
            ):
                raise
            try:
                fallback = self._resolve_supplemental_author(
                    source,
                    adapter,
                    authors[1],
                    primary=completed[0],
                    failure=failures[0],
                )
            except Exception:
                # Preserve the original terminal attempt evidence as the
                # externally visible failure if deterministic recovery is not
                # independently valid.
                raise author_error from None
            author_results = completed
            supplemental_failures = failures
            fallbacks = (fallback,)
        author_by_lane = {result.receipt.lane: result for result in author_results}
        primary_result = author_by_lane[ConstructionLane.OPUS5_DRAFT]
        alternate_output = (
            author_by_lane[ConstructionLane.GEMINI_ALTERNATE].output
            if ConstructionLane.GEMINI_ALTERNATE in author_by_lane
            else fallbacks[0].output
        )
        author_dependencies = {
            ConstructionLane.OPUS5_DRAFT.value: primary_result.receipt.output_blake3,
            ConstructionLane.GEMINI_ALTERNATE.value: blake3_hex(alternate_output),
        }

        critics = self._require_critic_quartet(
            adapter.prepare_critic_quorum(
                source,
                self._routes,
                primary_draft=primary_result.output,
                alternate_draft=alternate_output,
            ),
        )
        for request, lane in zip(critics, CRITIC_QUORUM_LANES, strict=True):
            self._validate_prepared(
                source, request, lane=lane, dependencies=author_dependencies
            )
        self._require_isolated_roots((*authors, *critics))
        try:
            critic_results = self._run_wave("critic", source, adapter, critics)
            critic_failures: tuple[ConstructionFailureReceipt, ...] = ()
            critic_attempt_evidence = tuple(
                ConstructionAttemptEvidence.from_success(result)
                for result in critic_results
            )
        except ConstructionWaveError as critic_error:
            critic_results = tuple(critic_error.completed)
            critic_failures = tuple(critic_error.failures)
            critic_attempt_evidence = tuple(critic_error.attempt_evidence)
        critic_by_lane = {result.receipt.lane: result for result in critic_results}
        selected_critique = (
            critic_by_lane.get(ConstructionLane.OPUS48_CRITIQUE)
            or critic_by_lane.get(ConstructionLane.OPUS5_CRITIQUE_BACKUP)
        )
        selected_comparison = (
            critic_by_lane.get(ConstructionLane.GPT56_COMPARISON)
            or critic_by_lane.get(ConstructionLane.GEMINI_COMPARISON_BACKUP)
        )
        if selected_critique is None or selected_comparison is None:
            raise ConstructionWaveError(
                "critic",
                completed=(*author_results, *critic_results),
                failures=(*supplemental_failures, *critic_failures),
                attempt_evidence=(
                    *author_attempt_evidence,
                    *critic_attempt_evidence,
                ),
            ) from None
        supplemental_failures = (*supplemental_failures, *critic_failures)
        revision_dependencies = {
            **author_dependencies,
            ConstructionLane.OPUS48_CRITIQUE.value: (
                selected_critique.receipt.output_blake3
            ),
            ConstructionLane.GPT56_COMPARISON.value: (
                selected_comparison.receipt.output_blake3
            ),
        }
        revision = adapter.prepare_revision(
            source,
            self._routes,
            primary_draft=primary_result.output,
            alternate_draft=alternate_output,
            opus48_critique=selected_critique.output,
            gpt56_comparison=selected_comparison.output,
            selected_critique_lane=selected_critique.receipt.lane,
            selected_comparison_lane=selected_comparison.receipt.lane,
        )
        self._validate_prepared(
            source,
            revision,
            lane=ConstructionLane.OPUS5_REVISION,
            dependencies=revision_dependencies,
        )
        self._require_isolated_roots((*authors, *critics, revision))
        try:
            revision_result = self._run_lane(source, adapter, revision)
        except _LaneAttemptError as exc:
            raise ConstructionWaveError(
                "revision",
                completed=(*author_results, *critic_results),
                failures=(*supplemental_failures, exc.receipt),
                attempt_evidence=(
                    *author_attempt_evidence,
                    *critic_attempt_evidence,
                    exc.evidence,
                ),
            ) from None
        revision_attempt_evidence = ConstructionAttemptEvidence.from_success(
            revision_result
        )

        phases_by_lane = {
            result.receipt.lane: result
            for result in (*author_results, *critic_results, revision_result)
        }
        phases = tuple(
            phases_by_lane[lane]
            for lane in V3_LANE_ORDER
            if lane in phases_by_lane
        )
        selected_critic_receipt_blake3s = MappingProxyType(
            {
                "comparison": selected_comparison.receipt.receipt_blake3,
                "critique": selected_critique.receipt.receipt_blake3,
            }
        )
        attempt_evidence = (
            *author_attempt_evidence,
            *critic_attempt_evidence,
            revision_attempt_evidence,
        )
        core = {
            "schema": _RESULT_SCHEMA_V3,
            "construction_run_id": self._ids.new("premium-construction-run"),
            "source_id": source.source_id,
            "source_request_blake3": source.request_blake3,
            "phase_receipt_blake3s": tuple(
                phase.receipt.receipt_blake3 for phase in phases
            ),
            "canonical_output_blake3": phases[-1].receipt.output_blake3,
            "provider_call_count": 7,
            "wave_widths": (2, 4, 1),
            "semantic_retry_count": 0,
            "legacy_manifest_emitted": False,
            "legacy_manifest_compatibility_claimed": False,
            "supplemental_failure_receipt_blake3s": tuple(
                failure.receipt_blake3 for failure in supplemental_failures
            ),
            "fallback_receipt_blake3s": tuple(
                fallback.receipt.receipt_blake3 for fallback in fallbacks
            ),
            "author_quorum_policy": _AUTHOR_QUORUM_POLICY,
            "successful_phase_count": len(phases),
            "supplemental_failure_count": len(supplemental_failures),
            "fallback_count": len(fallbacks),
            "critic_quorum_policy": _CRITIC_QUORUM_POLICY,
            "selected_critic_receipt_blake3s": selected_critic_receipt_blake3s,
            "attempt_evidence_blake3s": tuple(
                evidence.evidence_blake3 for evidence in attempt_evidence
            ),
        }
        result = PremiumConstructionResult(
            schema=core["schema"],
            construction_run_id=core["construction_run_id"],
            source_id=core["source_id"],
            source_request_blake3=core["source_request_blake3"],
            phases=phases,
            canonical_output_blake3=core["canonical_output_blake3"],
            provider_call_count=core["provider_call_count"],
            wave_widths=core["wave_widths"],
            semantic_retry_count=core["semantic_retry_count"],
            legacy_manifest_emitted=core["legacy_manifest_emitted"],
            legacy_manifest_compatibility_claimed=core[
                "legacy_manifest_compatibility_claimed"
            ],
            receipt_blake3=blake3_hex(core),
            supplemental_failures=supplemental_failures,
            fallbacks=fallbacks,
            author_quorum_policy=_AUTHOR_QUORUM_POLICY,
            critic_quorum_policy=_CRITIC_QUORUM_POLICY,
            selected_critic_receipt_blake3s=selected_critic_receipt_blake3s,
            attempt_evidence=attempt_evidence,
        )
        verify_premium_construction_result(result)
        return result

    def _construct_v4(
        self,
        source: FrozenConstructionSource,
        adapter: ConstructionContractAdapterPort,
    ) -> PremiumConstructionResult:
        """Execute v4: lower-tier critics are evidence, Opus 5 is authority."""

        authors = self._require_pair(
            adapter.prepare_authors(source, self._routes), AUTHOR_LANES
        )
        for request, lane in zip(authors, AUTHOR_LANES, strict=True):
            self._validate_prepared(source, request, lane=lane, dependencies={})
        self._require_isolated_roots(authors)
        supplemental_failures: tuple[ConstructionFailureReceipt, ...] = ()
        fallbacks: tuple[ConstructionFallbackResult, ...] = ()
        try:
            author_results = self._run_wave("author", source, adapter, authors)
            author_evidence = tuple(
                ConstructionAttemptEvidence.from_success(value)
                for value in author_results
            )
        except ConstructionWaveError as error:
            completed, failures = tuple(error.completed), tuple(error.failures)
            author_evidence = tuple(error.attempt_evidence)
            if not (
                len(completed) == 1
                and completed[0].receipt.lane is ConstructionLane.OPUS5_DRAFT
                and len(failures) == 1
                and failures[0].lane is ConstructionLane.GEMINI_ALTERNATE
            ):
                raise
            fallback = self._resolve_supplemental_author(
                source,
                adapter,
                authors[1],
                primary=completed[0],
                failure=failures[0],
            )
            author_results = completed
            supplemental_failures = failures
            fallbacks = (fallback,)
        author_by_lane = {value.receipt.lane: value for value in author_results}
        primary = author_by_lane[ConstructionLane.OPUS5_DRAFT]
        alternate = (
            author_by_lane[ConstructionLane.GEMINI_ALTERNATE].output
            if ConstructionLane.GEMINI_ALTERNATE in author_by_lane
            else fallbacks[0].output
        )
        author_dependencies = {
            ConstructionLane.OPUS5_DRAFT.value: primary.receipt.output_blake3,
            ConstructionLane.GEMINI_ALTERNATE.value: blake3_hex(alternate),
        }
        prepare = getattr(adapter, "prepare_critic_authority_v4", None)
        if not callable(prepare):
            raise PremiumConstructionError("v4 construction adapter port is incomplete")
        critics = tuple(
            prepare(
                source,
                self._routes,
                primary_draft=primary.output,
                alternate_draft=alternate,
            )
        )
        if tuple(value.lane for value in critics) != V4_CRITIC_LANES:
            raise PremiumConstructionError("v4 critic authority order differs")
        for request, lane in zip(critics, V4_CRITIC_LANES, strict=True):
            self._validate_prepared(
                source, request, lane=lane, dependencies=author_dependencies
            )
        self._require_isolated_roots((*authors, *critics))
        try:
            critic_results = self._run_wave("critic", source, adapter, critics)
            critic_failures: tuple[ConstructionFailureReceipt, ...] = ()
            critic_evidence = tuple(
                ConstructionAttemptEvidence.from_success(value)
                for value in critic_results
            )
        except ConstructionWaveError as error:
            critic_results = tuple(error.completed)
            critic_failures = tuple(error.failures)
            critic_evidence = tuple(error.attempt_evidence)
        critic_by_lane = {value.receipt.lane: value for value in critic_results}
        try:
            critique = critic_by_lane[ConstructionLane.OPUS5_CRITIQUE_BACKUP]
            comparison = critic_by_lane[
                ConstructionLane.GEMINI_COMPARISON_BACKUP
            ]
        except KeyError:
            raise ConstructionWaveError(
                "critic",
                completed=(*author_results, *critic_results),
                failures=(*supplemental_failures, *critic_failures),
                attempt_evidence=(*author_evidence, *critic_evidence),
            ) from None
        supplemental_failures = (*supplemental_failures, *critic_failures)
        revision_dependencies = {
            **author_dependencies,
            ConstructionLane.OPUS48_CRITIQUE.value: critique.receipt.output_blake3,
            ConstructionLane.GPT56_COMPARISON.value: comparison.receipt.output_blake3,
        }
        revision = adapter.prepare_revision(
            source,
            self._routes,
            primary_draft=primary.output,
            alternate_draft=alternate,
            opus48_critique=critique.output,
            gpt56_comparison=comparison.output,
            selected_critique_lane=critique.receipt.lane,
            selected_comparison_lane=comparison.receipt.lane,
        )
        self._validate_prepared(
            source,
            revision,
            lane=ConstructionLane.OPUS5_REVISION,
            dependencies=revision_dependencies,
        )
        self._require_isolated_roots((*authors, *critics, revision))
        try:
            revision_result = self._run_lane(source, adapter, revision)
        except _LaneAttemptError as exc:
            raise ConstructionWaveError(
                "revision",
                completed=(*author_results, *critic_results),
                failures=(*supplemental_failures, exc.receipt),
                attempt_evidence=(*author_evidence, *critic_evidence, exc.evidence),
            ) from None
        revision_evidence = ConstructionAttemptEvidence.from_success(revision_result)
        phases_by_lane = {
            value.receipt.lane: value
            for value in (*author_results, *critic_results, revision_result)
        }
        phases = tuple(
            phases_by_lane[lane] for lane in V4_LANE_ORDER if lane in phases_by_lane
        )
        selected = MappingProxyType(
            {
                "critique": critique.receipt.receipt_blake3,
                "comparison": comparison.receipt.receipt_blake3,
            }
        )
        evidence = (*author_evidence, *critic_evidence, revision_evidence)
        core = {
            "schema": _RESULT_SCHEMA_V4,
            "construction_run_id": self._ids.new("premium-construction-run"),
            "source_id": source.source_id,
            "source_request_blake3": source.request_blake3,
            "phase_receipt_blake3s": tuple(
                phase.receipt.receipt_blake3 for phase in phases
            ),
            "canonical_output_blake3": phases[-1].receipt.output_blake3,
            "provider_call_count": 7,
            "wave_widths": (2, 4, 1),
            "semantic_retry_count": 0,
            "legacy_manifest_emitted": False,
            "legacy_manifest_compatibility_claimed": False,
            "supplemental_failure_receipt_blake3s": tuple(
                value.receipt_blake3 for value in supplemental_failures
            ),
            "fallback_receipt_blake3s": tuple(
                value.receipt.receipt_blake3 for value in fallbacks
            ),
            "author_quorum_policy": _AUTHOR_QUORUM_POLICY,
            "successful_phase_count": len(phases),
            "supplemental_failure_count": len(supplemental_failures),
            "fallback_count": len(fallbacks),
            "critic_quorum_policy": _CRITIC_AUTHORITY_POLICY,
            "selected_critic_receipt_blake3s": selected,
            "attempt_evidence_blake3s": tuple(
                value.evidence_blake3 for value in evidence
            ),
        }
        result = PremiumConstructionResult(
            schema=_RESULT_SCHEMA_V4,
            construction_run_id=core["construction_run_id"],
            source_id=source.source_id,
            source_request_blake3=source.request_blake3,
            phases=phases,
            canonical_output_blake3=core["canonical_output_blake3"],
            provider_call_count=7,
            wave_widths=(2, 4, 1),
            semantic_retry_count=0,
            legacy_manifest_emitted=False,
            legacy_manifest_compatibility_claimed=False,
            receipt_blake3=blake3_hex(core),
            supplemental_failures=supplemental_failures,
            fallbacks=fallbacks,
            author_quorum_policy=_AUTHOR_QUORUM_POLICY,
            critic_quorum_policy=_CRITIC_AUTHORITY_POLICY,
            selected_critic_receipt_blake3s=selected,
            attempt_evidence=evidence,
        )
        verify_premium_construction_result(result)
        return result

    def construct(
        self,
        source: FrozenConstructionSource,
        adapter: ConstructionContractAdapterPort,
    ) -> PremiumConstructionResult:
        if self._result_schema == _RESULT_SCHEMA_V4:
            return self._construct_v4(source, adapter)
        return self._construct_v3(source, adapter)

    def construct_and_publish(
        self,
        source: FrozenConstructionSource,
        adapter: ConstructionContractAdapterPort,
        sink: ConstructionResultSinkPort,
    ) -> PublishedConstructionResult:
        if not callable(getattr(sink, "publish", None)):
            raise PremiumConstructionError("construction sink must expose publish")
        result = self.construct(source, adapter)
        publication = sink.publish(result)
        return PublishedConstructionResult(result, publication)

    def construct_many(
        self,
        jobs: Sequence[ConstructionJob],
        *,
        max_workers: int,
    ) -> tuple[PremiumConstructionResult, ...]:
        """Run candidates concurrently with no implicit cap, queue, or retry.

        All jobs are submitted before any result is awaited.  ``max_workers``
        is an explicit deployment input so campaigns can scale independently
        from the fixed per-candidate 2/4/1 semantic schedule.
        """

        values = tuple(jobs)
        if not values:
            return ()
        if type(max_workers) is not int or max_workers < 1:
            raise PremiumConstructionError("construction max_workers must be positive")
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="eva-premium-construction"
        ) as executor:
            futures = tuple(
                executor.submit(self.construct, job.source, job.adapter) for job in values
            )
            return tuple(future.result() for future in futures)


__all__ = [
    "AUTHOR_QUORUM_POLICY_V1",
    "COMPARISON_QUORUM_LANES",
    "CRITIC_QUORUM_LANES",
    "CRITIC_QUORUM_POLICY_V1",
    "CRITIC_AUTHORITY_POLICY_V2",
    "CRITIQUE_QUORUM_LANES",
    "ConstructionContractAdapterPort",
    "ConstructionAttemptEvidence",
    "ConstructionFallbackReceipt",
    "ConstructionFallbackResult",
    "ConstructionFailureReceipt",
    "ConstructionJob",
    "ConstructionLane",
    "ConstructionModelRoute",
    "ConstructionPhaseReceipt",
    "ConstructionPhaseResult",
    "ConstructionPublicationReceipt",
    "ConstructionResultSinkPort",
    "ConstructionTurnRequest",
    "ConstructionWaveError",
    "FrozenConstructionSource",
    "PremiumCodexConstructionOrchestrator",
    "PremiumConstructionError",
    "PremiumConstructionResult",
    "PremiumConstructionRoutes",
    "PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V1",
    "PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V2",
    "PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3",
    "PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4",
    "PublishedConstructionResult",
    "SUPPLEMENTAL_FALLBACK_POLICY_V1",
    "STRICT_RESPONSE_EXTRACTION_POLICY_V1",
    "UNIQUE_SCHEMA_RESPONSE_EXTRACTION_POLICY_V2",
    "V3_LANE_ORDER",
    "V4_LANE_ORDER",
    "verify_construction_failure_receipt",
    "verify_construction_attempt_evidence",
    "verify_construction_fallback_result",
    "verify_premium_construction_result",
    "extract_unique_schema_valid_payload",
]
