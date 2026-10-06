"""Fail-closed production composition for the 6,000-training campaign.

The builder in this module is deliberately a *wiring* operation.  It opens no
Codex thread and makes no provider request.  At execution time a fixed pool of
persistent Codex app-server shards is shared while every actor and judge call
still starts a fresh, isolated Codex thread.

Existing EvaMed policy, tool, rubric, evidence, and admission contracts are
inputs to this module.  They are never translated or rewritten here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import resource
import shutil
import stat
import sys
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable
from uuid import UUID

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eva_agent.admission import (
    Ed25519AdmissionSupervisor,
    Ed25519SupervisorSignatureVerifier,
    SupervisorAdmissionContext,
    load_trust_store,
    SignedEnvelope,
    verify_signed_envelope,
)
from eva_agent.campaign import (
    CampaignLedger,
    CampaignPlan,
    CampaignSelection,
    FrozenCampaignCandidateSource,
    verify_campaign_selection,
    verify_plan,
)
from eva_agent.campaign.history import signed_historical_attempt_evidence
from eva_agent.codex_pipeline import (
    ActorOptionsFactory,
    CodexOpus5AgentJudge,
    CodexPipelineError,
    CodexRolloutAdapter,
    CodexToolExecutionBridge,
    JudgeOptionsFactory,
    TURN_MCP_DIRECTORY_PREFIX,
    TURN_MCP_PRIVATE_ROOT_MODE,
    TURN_MCP_PROTOCOL_VERSION,
    TURN_MCP_SERVER_VERSION,
    TURN_MCP_SOCKET_FILENAME,
    TURN_MCP_UNIX_SOCKET_PATH_CAPACITY_BYTES,
    TurnMCPBridgeFactoryPort,
)
from eva_agent.codex_providers import CodexProviderRoute
from eva_agent.codex_runtime import (
    CodexLaunchOptions,
    CodexRuntime,
    CodexSandbox,
    CodexSkill,
    CodexThreadOptions,
    CodexToolOffer,
    CodexTurnInput,
    CodexTurnReceipt,
    OpenAICodexBackend,
    ShardedPersistentCodexRuntimeRunner,
)
from eva_agent.orchestration import (
    CampaignOrchestrator,
    CandidateJob,
    CandidateSource,
    OrchestrationConfig,
    ProgressCallback,
)
from eva_agent.pipeline import (
    Cohort,
    ImmutableArtifactStore,
    ModelTarget,
    PipelineVerifier,
    RandomUUIDFactory,
    RuntimeIdFactory,
    SeparationPolicy,
    Stage,
    VerifiableDataPipeline,
    WeightedRubricRewarder,
)
from eva_agent.pipeline.digests import (
    blake3_bytes,
    blake3_hex,
    canonical_json_bytes,
    is_blake3,
)
from eva_agent.rubrics import CompiledRubricRegistry
from eva_agent.sources.prospective_execution_catalog import CATALOG_SCHEMA as PROSPECTIVE_CATALOG_SCHEMA

from .codex_child_exec import build_sanitized_codex_exec_plan


CAMPAIGN_DEPLOYMENT_SCHEMA = "eva.codex-campaign-deployment.v2"
ROUTE_ASSIGNMENT_SCHEMA = "eva.codex-campaign-route-assignment.v1"
CODEX_CHILD_STARTUP_SCHEMA = "eva.codex-child-startup.v1"
FROZEN_SCHEDULE_TOTAL = 9_000
PRIMARY_SELECTION_TOTAL = 6_000
RESERVE_SELECTION_TOTAL = 3_000
EXPECTED_EXECUTABLE_BINDINGS = 1_344
DEFAULT_CHILD_SOFT_NOFILE = 65_536
ACTOR_SKILL_MATERIALIZATION_PATH_POLICY = (
    "blake3-root/ordinal-content-blake3/SKILL.md"
)
TURN_MCP_CHILD_ENVIRONMENT_NAMES = (
    "EVA_TURN_MCP_MAXIMUM",
    "EVA_TURN_MCP_NONCE",
    "EVA_TURN_MCP_SOCKET",
    "LANG",
    "LC_ALL",
    "PATH",
    "TZ",
)
TURN_MCP_LAUNCH_METADATA_KEYS = frozenset(
    {
        "schema",
        "protocol_version",
        "server_version",
        "proxy_python",
        "proxy_python_blake3",
        "proxy_exec_script",
        "proxy_exec_script_blake3",
        "proxy_script",
        "proxy_script_blake3",
        "maximum_parallel_calls",
        "startup_timeout_seconds",
        "tool_timeout_seconds",
        "final_child_environment_names",
        "inherited_parent_environment",
        "environment_values_recorded",
        "turn_nonce_recorded",
        "unix_socket_preflight",
        "launch_blake3",
    }
)
TURN_MCP_UNIX_SOCKET_PREFLIGHT_KEYS = frozenset(
    {
        "schema",
        "temp_root",
        "temp_root_uid",
        "temp_root_mode",
        "temp_root_owned_by_process",
        "temp_root_is_symlink",
        "directory_prefix",
        "random_component_bytes",
        "socket_filename",
        "probed_socket_path_bytes",
        "sockaddr_un_path_capacity_bytes",
        "terminator_bytes",
        "name_length_source",
        "preflight_passed",
    }
)

# First-release Codex is deliberately a transport for exact candidate MCP
# tools and explicit SkillInput mounts.  These settings prevent uncommitted
# repository instructions and not-yet-projected native tools from entering a
# trajectory.  Enabling a native surface later requires a new versioned
# deployment recipe and mixed native/MCP projection evidence.
CODEX_FIRST_RELEASE_THREAD_CONFIG = MappingProxyType(
    {
        "project_doc_max_bytes": 0,
        "web_search": "disabled",
        "features": MappingProxyType(
            {
                "apps": False,
                "browser_use": False,
                "code_mode_host": False,
                "computer_use": False,
                "goals": False,
                "hooks": False,
                "image_generation": False,
                "multi_agent": False,
                "plugins": False,
                "shell_tool": False,
                "skill_search": False,
                "standalone_web_search": False,
                "tool_suggest": False,
                "tool_search_always_defer_mcp_tools": False,
                "unified_exec": False,
                "view_image": False,
            }
        ),
    }
)
CODEX_FIRST_RELEASE_CONFIG_OVERRIDES = (
    "project_doc_max_bytes=0",
    'web_search="disabled"',
    *(f"features.{name}=false" for name in CODEX_FIRST_RELEASE_THREAD_CONFIG["features"]),
)
CODEX_FIRST_RELEASE_NATIVE_SURFACE = MappingProxyType(
    {
        "shell_tool": False,
        "unified_exec": False,
        "multi_agent": False,
        "image_generation": False,
        "view_image": False,
        "apps": False,
        "plugins": False,
        "skill_search": False,
        "goals": False,
        "hooks": False,
        "tool_suggest": False,
        "web_search": False,
        "browser_use": False,
        "code_mode_host": False,
        "computer_use": False,
        "tool_search_always_defer_mcp_tools": False,
    }
)


@runtime_checkable
class PersistentCodexRunnerPort(Protocol):
    """Structural lifecycle/turn port implemented by one or many app-servers."""

    @property
    def shard_count(self) -> int: ...

    def start(self) -> None: ...

    def run_once(
        self, options: CodexThreadOptions, turn_input: CodexTurnInput
    ) -> CodexTurnReceipt: ...

    def close(self) -> None: ...


@runtime_checkable
class CandidateExecutionBindingPort(Protocol):
    candidate_id: str
    source_candidate_id: str
    initial_workspace_files: Mapping[str, bytes]
    public_runtime_context: Mapping[str, Any]
    tool_registry: Any
    binding_blake3: str


@runtime_checkable
class ExecutionBindingResolverPort(Protocol):
    """Signed candidate-specific policy/tool binding catalog."""

    @property
    def catalog_inventory_blake3(self) -> str: ...

    @property
    def executable_candidate_count(self) -> int: ...

    def inventory(self) -> tuple[Any, ...]: ...

    def resolve(
        self, eva_candidate_id: str, *, source_candidate_id: str
    ) -> CandidateExecutionBindingPort: ...


@runtime_checkable
class ActorSkillCatalogPort(Protocol):
    """Verified stage-aware Codex SkillInput catalog."""

    catalog_blake3: str
    materialization_root: Path
    materialization_blake3: str
    materialization_path_policy: str

    def inventory(self) -> tuple[Any, ...]: ...

    def for_stage(self, stage: Stage | str) -> tuple[CodexSkill, ...]: ...

    def __call__(self, request: Any) -> Sequence[Any]: ...


@runtime_checkable
class Opus5AdapterGatewayPort(Protocol):
    """Prepared, sharded Opus-5 Responses adapter kept alive for the run.

    A single legacy gateway is capped too low for the production ramp.  The
    application-owned implementation may shard gateways, but it must expose
    one aggregate capacity and one stable public binding commitment.
    """

    aggregate_max_concurrency: int
    shard_count: int
    binding_blake3: str

    def start(self) -> None: ...

    def close(self) -> None: ...


def _campaign_separation_policy() -> SeparationPolicy:
    """Retain cascade telemetry while admitting one valid judged trajectory."""

    return SeparationPolicy(
        minimum_strong_reward=0.60,
        minimum_strong_minus_weak=0.05,
        material_item_delta=0.10,
        require_strong_hard_gates=True,
        early_continuation_after_first_valid_trajectory=True,
    )


class CampaignDeploymentError(ValueError):
    """The production campaign could not be composed without ambiguity."""


@runtime_checkable
class FrozenCampaignSourcePort(CandidateSource, Protocol):
    """A preverified frozen selection exposed through the scheduler's port.

    Tests may inject a structural implementation and therefore do not need to
    construct or open all 6,000 sources.  Production should use
    :class:`FrozenSelectionCandidateSource`, which independently verifies the
    complete :class:`CampaignSelection` before exposing these commitments.
    """

    candidate_count: int
    scheduled_count: int
    primary_count: int
    reserve_count: int
    candidate_ids: tuple[str, ...]
    primary_candidate_ids: tuple[str, ...]
    reserve_candidate_ids: tuple[str, ...]
    plan_blake3: str
    rubric_registry_blake3: str
    source_registry_blake3: str
    selection_blake3: str

    def load(self, candidate_id: str) -> CandidateJob: ...

    def source_candidate_id(self, candidate_id: str) -> str: ...

    def queue_records(self, *, worker_width: int = 64) -> tuple[Any, ...]: ...


class FrozenSelectionCandidateSource:
    """Bind the existing lazy source loader to an exact verified selection."""

    def __init__(
        self,
        *,
        source: FrozenCampaignCandidateSource,
        selection: CampaignSelection,
        plan: CampaignPlan,
        rubrics: CompiledRubricRegistry,
    ) -> None:
        if not callable(getattr(source, "load", None)):
            raise CampaignDeploymentError("frozen candidate source lacks load")
        try:
            verify_campaign_selection(selection, plan=plan, rubrics=rubrics)
        except Exception as exc:
            raise CampaignDeploymentError("frozen campaign selection verification failed") from exc
        self._source = source
        self._source_ids = MappingProxyType(
            {row.candidate_id: row.source_candidate_id for row in selection.entries}
        )
        self.candidate_count = len(selection.entries)
        self.candidate_ids = tuple(row.candidate_id for row in selection.entries)
        self.primary_candidate_ids = tuple(
            row.candidate_id
            for row in selection.entries
            if getattr(row, "selection_tier", None) == "primary"
        )
        self.reserve_candidate_ids = tuple(
            row.candidate_id
            for row in selection.entries
            if getattr(row, "selection_tier", None) == "reserve"
        )
        self.scheduled_count = self.candidate_count
        self.primary_count = len(self.primary_candidate_ids)
        self.reserve_count = len(self.reserve_candidate_ids)
        self.plan_blake3 = selection.plan_blake3
        self.rubric_registry_blake3 = selection.rubric_registry_blake3
        self.source_registry_blake3 = selection.source_registry_blake3
        self.selection_blake3 = selection.selection_blake3

    def load(self, candidate_id: str) -> CandidateJob:
        return self._source.load(candidate_id)

    def source_candidate_id(self, candidate_id: str) -> str:
        try:
            return self._source_ids[candidate_id]
        except KeyError:
            raise CampaignDeploymentError("candidate is outside frozen selection") from None

    def queue_rows(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate source-verifying queue materialization without translation."""

        return self._source.queue_rows(*args, **kwargs)

    def queue_records(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate exact 9,000-row ledger records without schema translation."""

        return self._source.queue_records(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class ProviderRouteTarget:
    """Receipt-safe identity of one already-configured Codex provider route."""

    route_id: str
    target: ModelTarget
    config_blake3: str

    def __post_init__(self) -> None:
        if not self.route_id or not isinstance(self.target, ModelTarget):
            raise CampaignDeploymentError("provider route target is incomplete")
        if not is_blake3(self.config_blake3):
            raise CampaignDeploymentError("provider route config commitment differs")

    def to_document(self) -> dict[str, str]:
        return {
            "route_id": self.route_id,
            "cohort": self.target.cohort.value,
            "model_id": self.target.model_id,
            "provider": self.target.provider,
            "config_blake3": self.config_blake3,
        }


@dataclass(frozen=True, slots=True)
class ProviderRouteHealth:
    """Verified one-attempt canary outcome consumed by deployment preflight."""

    route_id: str
    model_id: str
    status: str
    canary_receipt_blake3: str
    verified: bool

    def __post_init__(self) -> None:
        if (
            not self.route_id
            or not self.model_id
            or self.status not in {"direct-pass", "adapter-pass", "unavailable"}
            or not is_blake3(self.canary_receipt_blake3)
            or type(self.verified) is not bool
        ):
            raise CampaignDeploymentError("provider route health proof differs")

    def to_document(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "model_id": self.model_id,
            "status": self.status,
            "canary_receipt_blake3": self.canary_receipt_blake3,
            "verified": self.verified,
        }


@dataclass(frozen=True, slots=True)
class CodexProviderTiers:
    """Non-empty actor route pools plus a fixed Opus 5 judge route."""

    weak: tuple[ProviderRouteTarget, ...]
    middle: tuple[ProviderRouteTarget, ...]
    strong: tuple[ProviderRouteTarget, ...]
    judge: ProviderRouteTarget

    def __post_init__(self) -> None:
        expected = (
            ("weak", self.weak, Cohort.WEAK),
            ("middle", self.middle, Cohort.MIDDLE),
            ("strong", self.strong, Cohort.STRONG),
        )
        seen_routes: set[str] = set()
        for label, routes, cohort in expected:
            if not isinstance(routes, tuple) or not routes:
                raise CampaignDeploymentError(f"{label} tier must contain a non-empty route pool")
            for route in routes:
                if not isinstance(route, ProviderRouteTarget) or route.target.cohort is not cohort:
                    raise CampaignDeploymentError(f"{label} provider target cohort differs")
                if route.route_id in seen_routes:
                    raise CampaignDeploymentError("actor provider route is assigned twice")
                seen_routes.add(route.route_id)
        if not isinstance(self.judge, ProviderRouteTarget):
            raise CampaignDeploymentError("judge provider route differs")
        normalized = self.judge.target.model_id.casefold().replace("_", "-").replace(" ", "-")
        if self.judge.route_id != "opus_5" or "opus-5" not in normalized:
            raise CampaignDeploymentError("workspace Agent Judge must use the Opus 5 route")
        if len({route.target.model_id for route in self.actor_routes}) != len(
            self.actor_routes
        ):
            raise CampaignDeploymentError("selected actor routes require distinct model IDs")

    @property
    def actor_routes(self) -> tuple[ProviderRouteTarget, ...]:
        return (*self.weak, *self.middle, *self.strong)

    @property
    def judge_model_id(self) -> str:
        return self.judge.target.model_id

    @property
    def judge_provider(self) -> str:
        return self.judge.target.provider

    def routes_for(self, cohort: Cohort) -> tuple[ProviderRouteTarget, ...]:
        return {
            Cohort.WEAK: self.weak,
            Cohort.MIDDLE: self.middle,
            Cohort.STRONG: self.strong,
        }[cohort]

    @classmethod
    def from_routes(
        cls,
        routes: Mapping[str, CodexProviderRoute],
        *,
        weak_routes: tuple[str, ...] = ("deepseek_v4_flash",),
        middle_routes: tuple[str, ...] = ("gemini_3_1_pro", "opus_4_8"),
        strong_routes: tuple[str, ...] = ("gpt_5_6_sol", "opus_5"),
        judge_route: str = "opus_5",
    ) -> "CodexProviderTiers":
        """Resolve all required routes or fail; no missing route is substituted."""

        pools = (
            (Cohort.WEAK, weak_routes),
            (Cohort.MIDDLE, middle_routes),
            (Cohort.STRONG, strong_routes),
        )
        actor_route_ids = tuple(route_id for _cohort, pool in pools for route_id in pool)
        if any(not pool for _cohort, pool in pools) or len(set(actor_route_ids)) != len(
            actor_route_ids
        ):
            raise CampaignDeploymentError("actor tier routes must be non-empty and unique")

        def resolve(route_id: str, cohort: Cohort) -> ProviderRouteTarget:
            route = routes.get(route_id)
            if not isinstance(route, CodexProviderRoute) or route.route_id != route_id:
                raise CampaignDeploymentError(
                    f"required Codex provider route is missing: {route_id}"
                )
            return ProviderRouteTarget(
                route_id=route_id,
                target=ModelTarget(cohort, route.model_id, route.config.provider_id),
                config_blake3=route.config.safe_blake3,
            )

        resolved = {
            cohort: tuple(resolve(route_id, cohort) for route_id in pool)
            for cohort, pool in pools
        }
        judge_source = routes.get(judge_route)
        if (
            not isinstance(judge_source, CodexProviderRoute)
            or judge_source.route_id != judge_route
        ):
            raise CampaignDeploymentError(
                f"required Codex provider route is missing: {judge_route}"
            )
        return cls(
            weak=resolved[Cohort.WEAK],
            middle=resolved[Cohort.MIDDLE],
            strong=resolved[Cohort.STRONG],
            judge=ProviderRouteTarget(
                route_id=judge_route,
                target=ModelTarget(
                    Cohort.STRONG,
                    judge_source.model_id,
                    judge_source.config.provider_id,
                ),
                config_blake3=judge_source.config.safe_blake3,
            ),
        )


def _verify_provider_health(
    health: Mapping[str, ProviderRouteHealth], *, tiers: CodexProviderTiers
) -> str:
    required = {route.route_id: route for route in tiers.actor_routes}
    # The Opus 5 judge intentionally reuses the already-canary-proven Opus 5
    # transport rather than consuming a duplicate semantic canary.
    required[tiers.judge.route_id] = tiers.judge
    if not isinstance(health, Mapping) or set(health) != set(required):
        raise CampaignDeploymentError("verified provider health inventory differs")
    rows: list[dict[str, Any]] = []
    for route_id in sorted(required):
        proof = health[route_id]
        route = required[route_id]
        if not isinstance(proof, ProviderRouteHealth):
            raise CampaignDeploymentError("provider health proof type differs")
        if proof.route_id != route_id or proof.model_id != route.target.model_id:
            raise CampaignDeploymentError("provider health route/model identity differs")
        if proof.verified is not True or proof.status not in {"direct-pass", "adapter-pass"}:
            raise CampaignDeploymentError(
                f"required provider route is unavailable: {route_id}"
            )
        rows.append(proof.to_document())
    return blake3_hex(rows)


@dataclass(frozen=True, slots=True)
class CandidateRouteAssignment:
    """One outcome-blind weak/middle/strong route assignment."""

    candidate_id: str
    weak_route_id: str
    middle_route_id: str
    strong_route_id: str
    assignment_blake3: str

    def to_document(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "weak_route_id": self.weak_route_id,
            "middle_route_id": self.middle_route_id,
            "strong_route_id": self.strong_route_id,
            "assignment_blake3": self.assignment_blake3,
        }


@dataclass(frozen=True, slots=True)
class RouteAssignmentReceipt:
    """Exact auditable route allocation for the full primary+reserve schedule."""

    selection_blake3: str
    route_catalog_blake3: str
    assignments: tuple[CandidateRouteAssignment, ...]
    counts_by_route: Mapping[str, int]
    receipt_blake3: str
    schema: str = ROUTE_ASSIGNMENT_SCHEMA

    def core_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "selection_blake3": self.selection_blake3,
            "route_catalog_blake3": self.route_catalog_blake3,
            "assignment_method": "per_cohort_blake3_rank_round_robin_v1",
            "candidate_count": len(self.assignments),
            "rollouts_per_candidate": {
                "weak": 1,
                "middle": 1,
                "strong": 1,
            },
            "assignments": [row.to_document() for row in self.assignments],
            "counts_by_route": dict(self.counts_by_route),
        }

    def to_document(self) -> dict[str, Any]:
        return {**self.core_document(), "receipt_blake3": self.receipt_blake3}


class DeterministicRouteSelector:
    """Balance every selected actor route without consulting model outcomes."""

    def __init__(
        self,
        *,
        candidate_ids: Sequence[str],
        selection_blake3: str,
        tiers: CodexProviderTiers,
    ) -> None:
        identities = tuple(candidate_ids)
        if len(identities) != FROZEN_SCHEDULE_TOTAL or len(set(identities)) != len(
            identities
        ):
            raise CampaignDeploymentError("route selector requires 9,000 unique candidates")
        for candidate_id in identities:
            try:
                if str(UUID(candidate_id)) != candidate_id:
                    raise ValueError
            except (TypeError, ValueError, AttributeError):
                raise CampaignDeploymentError(
                    "route selector candidate identity differs"
                ) from None
        if not is_blake3(selection_blake3):
            raise CampaignDeploymentError("route selector selection commitment differs")
        catalog = [route.to_document() for route in tiers.actor_routes]
        route_catalog_blake3 = blake3_hex(catalog)
        by_candidate: dict[str, dict[Cohort, ProviderRouteTarget]] = {
            candidate_id: {} for candidate_id in identities
        }
        counts: dict[str, int] = {}
        for cohort in (Cohort.WEAK, Cohort.MIDDLE, Cohort.STRONG):
            pool = tiers.routes_for(cohort)
            ranked = sorted(
                identities,
                key=lambda candidate_id: (
                    blake3_hex(
                        {
                            "schema": "eva.candidate-route-rank.v1",
                            "selection_blake3": selection_blake3,
                            "route_catalog_blake3": route_catalog_blake3,
                            "candidate_id": candidate_id,
                            "cohort": cohort.value,
                        }
                    ),
                    candidate_id,
                ),
            )
            for index, candidate_id in enumerate(ranked):
                route = pool[index % len(pool)]
                by_candidate[candidate_id][cohort] = route
                key = f"{cohort.value}:{route.route_id}"
                counts[key] = counts.get(key, 0) + 1
        assignments: list[CandidateRouteAssignment] = []
        targets: dict[str, tuple[ModelTarget, ModelTarget, ModelTarget]] = {}
        for candidate_id in sorted(identities):
            selected = by_candidate[candidate_id]
            core = {
                "schema": "eva.candidate-route-assignment.v1",
                "selection_blake3": selection_blake3,
                "route_catalog_blake3": route_catalog_blake3,
                "candidate_id": candidate_id,
                "weak_route_id": selected[Cohort.WEAK].route_id,
                "middle_route_id": selected[Cohort.MIDDLE].route_id,
                "strong_route_id": selected[Cohort.STRONG].route_id,
                "rollouts_per_cohort": 1,
            }
            assignments.append(
                CandidateRouteAssignment(
                    candidate_id=candidate_id,
                    weak_route_id=selected[Cohort.WEAK].route_id,
                    middle_route_id=selected[Cohort.MIDDLE].route_id,
                    strong_route_id=selected[Cohort.STRONG].route_id,
                    assignment_blake3=blake3_hex(core),
                )
            )
            targets[candidate_id] = (
                selected[Cohort.WEAK].target,
                selected[Cohort.MIDDLE].target,
                selected[Cohort.STRONG].target,
            )
        provisional = RouteAssignmentReceipt(
            selection_blake3=selection_blake3,
            route_catalog_blake3=route_catalog_blake3,
            assignments=tuple(assignments),
            counts_by_route=MappingProxyType(dict(sorted(counts.items()))),
            receipt_blake3="",
        )
        receipt = replace(
            provisional,
            receipt_blake3=blake3_hex(provisional.core_document()),
        )
        _verify_route_assignment_receipt(receipt, tiers=tiers)
        self._targets = MappingProxyType(targets)
        self.receipt = receipt

    def targets_for(
        self, candidate_id: str
    ) -> tuple[ModelTarget, ModelTarget, ModelTarget]:
        try:
            return self._targets[candidate_id]
        except KeyError:
            raise CampaignDeploymentError("candidate has no frozen route assignment") from None


def _verify_route_assignment_receipt(
    receipt: RouteAssignmentReceipt, *, tiers: CodexProviderTiers
) -> None:
    if (
        receipt.schema != ROUTE_ASSIGNMENT_SCHEMA
        or len(receipt.assignments) != FROZEN_SCHEDULE_TOTAL
        or not is_blake3(receipt.route_catalog_blake3)
        or not is_blake3(receipt.receipt_blake3)
        or blake3_hex(receipt.core_document()) != receipt.receipt_blake3
    ):
        raise CampaignDeploymentError("route assignment receipt differs")
    expected_keys = {
        f"{route.target.cohort.value}:{route.route_id}" for route in tiers.actor_routes
    }
    if set(receipt.counts_by_route) != expected_keys:
        raise CampaignDeploymentError("route assignment count inventory differs")
    for cohort in (Cohort.WEAK, Cohort.MIDDLE, Cohort.STRONG):
        pool = tiers.routes_for(cohort)
        base, remainder = divmod(FROZEN_SCHEDULE_TOTAL, len(pool))
        for index, route in enumerate(pool):
            expected = base + (1 if index < remainder else 0)
            if receipt.counts_by_route.get(f"{cohort.value}:{route.route_id}") != expected:
                raise CampaignDeploymentError("route assignment is not exactly balanced")
    route_sets = {
        Cohort.WEAK: {route.route_id for route in tiers.weak},
        Cohort.MIDDLE: {route.route_id for route in tiers.middle},
        Cohort.STRONG: {route.route_id for route in tiers.strong},
    }
    if (
        tuple(row.candidate_id for row in receipt.assignments)
        != tuple(sorted(row.candidate_id for row in receipt.assignments))
        or len({row.candidate_id for row in receipt.assignments}) != FROZEN_SCHEDULE_TOTAL
    ):
        raise CampaignDeploymentError("candidate route assignment commitment differs")
    for row in receipt.assignments:
        if (
            row.weak_route_id not in route_sets[Cohort.WEAK]
            or row.middle_route_id not in route_sets[Cohort.MIDDLE]
            or row.strong_route_id not in route_sets[Cohort.STRONG]
        ):
            raise CampaignDeploymentError("candidate route assignment tier differs")
        core = {
            "schema": "eva.candidate-route-assignment.v1",
            "selection_blake3": receipt.selection_blake3,
            "route_catalog_blake3": receipt.route_catalog_blake3,
            "candidate_id": row.candidate_id,
            "weak_route_id": row.weak_route_id,
            "middle_route_id": row.middle_route_id,
            "strong_route_id": row.strong_route_id,
            "rollouts_per_cohort": 1,
        }
        if row.assignment_blake3 != blake3_hex(core):
            raise CampaignDeploymentError("candidate route assignment BLAKE3 differs")


@dataclass(frozen=True, slots=True)
class PreparedPersistentCodexRuntime:
    """A provider-free, capacity-verified persistent app-server pool recipe."""

    runner: PersistentCodexRunnerPort
    launch_options: CodexLaunchOptions
    startup_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not all(
            callable(getattr(self.runner, name, None))
            for name in ("start", "run_once", "close")
        ):
            raise CampaignDeploymentError("prepared persistent Codex runner port differs")
        if not isinstance(self.launch_options, CodexLaunchOptions):
            raise CampaignDeploymentError("prepared Codex launch options differ")
        metadata = self.startup_metadata
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("schema") != CODEX_CHILD_STARTUP_SCHEMA
            or metadata.get("parent_limit_mutated") is not False
            or metadata.get("strategy") != "external_campaign_prlimit_prevalidated"
            or metadata.get("process_nofile_sufficient") is not (
                metadata.get("process_soft_nofile_observed", 0)
                >= metadata.get("process_soft_nofile_required", 1)
            )
            or metadata.get("app_server_shards") != getattr(self.runner, "shard_count", None)
            or metadata.get("strict_config") is not True
            or metadata.get("project_doc_max_bytes") != 0
            or metadata.get("uncommitted_ancestor_project_docs_enabled") is not False
            or dict(metadata.get("native_codex_surface", {}))
            != dict(CODEX_FIRST_RELEASE_NATIVE_SURFACE)
            or tuple(metadata.get("config_overrides", ()))
            != CODEX_FIRST_RELEASE_CONFIG_OVERRIDES
            or not is_blake3(metadata.get("codex_executable_blake3"))
            or not is_blake3(metadata.get("sanitized_child_exec_wrapper_blake3"))
            or not isinstance(metadata.get("sanitized_child_exec"), Mapping)
            or metadata["sanitized_child_exec"].get("schema")
            != "eva.codex-sanitized-child-exec.v1"
            or metadata["sanitized_child_exec"].get("inherited_parent_environment")
            is not False
            or metadata["sanitized_child_exec"].get("credential_values_recorded")
            is not False
            or not is_blake3(metadata.get("launch_blake3"))
        ):
            raise CampaignDeploymentError("Codex startup capacity metadata differs")
        metadata_core = {
            key: value for key, value in metadata.items() if key != "launch_blake3"
        }
        if blake3_hex(metadata_core) != metadata["launch_blake3"]:
            raise CampaignDeploymentError("Codex startup metadata BLAKE3 differs")


def prepare_persistent_codex_runtime(
    *,
    codex_bin: str | Path,
    cwd: str | Path,
    config_overrides: Sequence[str] = CODEX_FIRST_RELEASE_CONFIG_OVERRIDES,
    child_env: Mapping[str, str] | None = None,
    prlimit_bin: str | Path = "prlimit",
    required_process_soft_nofile: int = DEFAULT_CHILD_SOFT_NOFILE,
    app_server_shards: int = 1,
    worker_width: int = 128,
    codex_distribution_version: str | None = None,
    isolation_root: str | Path | None = None,
) -> PreparedPersistentCodexRuntime:
    """Prepare a structural app-server pool after external ``prlimit`` preflight.

    This function performs local path/limit preflight and creates Python
    objects only.  It neither starts app-server nor makes a provider request.
    The campaign process must itself have been launched through the documented
    ``prlimit --nofile=... --`` prefix.  This function never calls
    ``setrlimit`` and therefore cannot mutate its parent shell or the global
    host limit.
    """

    if (
        type(required_process_soft_nofile) is not int
        or required_process_soft_nofile < 1_024
    ):
        raise CampaignDeploymentError("required campaign soft nofile differs")
    if type(app_server_shards) is not int or not 1 <= app_server_shards <= 256:
        raise CampaignDeploymentError("Codex app-server shards must be in [1,256]")
    if type(worker_width) is not int or worker_width < 1:
        raise CampaignDeploymentError("configured campaign worker width differs")
    workspace = _absolute_path(Path(cwd), label="Codex launch cwd")

    def executable(value: str | Path, *, label: str) -> Path:
        raw = str(value)
        resolved = shutil.which(raw) if "/" not in raw else raw
        if not resolved:
            raise CampaignDeploymentError(f"{label} executable is missing")
        path = Path(resolved).resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise CampaignDeploymentError(f"{label} executable is unavailable")
        return path

    codex = executable(codex_bin, label="Codex")
    limiter = executable(prlimit_bin, label="prlimit")
    host_soft, host_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if host_hard != resource.RLIM_INFINITY and host_hard < required_process_soft_nofile:
        raise CampaignDeploymentError("host hard nofile cannot support campaign profile")
    hard_text = "unlimited" if host_hard == resource.RLIM_INFINITY else str(host_hard)
    overrides = tuple(config_overrides)
    if any(not isinstance(value, str) or not value for value in overrides):
        raise CampaignDeploymentError("Codex app-server config override differs")
    if overrides != CODEX_FIRST_RELEASE_CONFIG_OVERRIDES:
        raise CampaignDeploymentError(
            "Codex first-release isolation overrides differ"
        )
    if codex_distribution_version is not None and (
        not isinstance(codex_distribution_version, str)
        or not codex_distribution_version.strip()
    ):
        raise CampaignDeploymentError("Codex distribution version differs")
    codex_args: list[str] = []
    for value in overrides:
        codex_args.extend(("--config", value))
    codex_args.extend(("app-server", "--strict-config", "--listen", "stdio://"))
    isolated = (
        workspace / ".eva-codex-child"
        if isolation_root is None
        else _absolute_path(Path(isolation_root), label="Codex isolation root")
    )
    if not isolated.exists():
        isolated.mkdir(mode=0o700, parents=True)
    wrapper = Path(__file__).with_name("codex_child_exec.py").resolve()
    child_values = {} if child_env is None else dict(child_env)
    try:
        exec_plan = build_sanitized_codex_exec_plan(
            python_bin=Path(sys.executable).resolve(),
            wrapper_script=wrapper,
            codex_bin=codex,
            isolation_root=isolated,
            credential_env_names=tuple(child_values),
            codex_args=tuple(codex_args),
        )
    except Exception as exc:
        raise CampaignDeploymentError(
            "sanitized Codex child exec plan verification failed"
        ) from exc
    launch = CodexLaunchOptions(
        launch_args_override=exec_plan.launch_args,
        cwd=str(workspace),
        env=child_values,
    )
    metadata_core = {
        "schema": CODEX_CHILD_STARTUP_SCHEMA,
        "strategy": "external_campaign_prlimit_prevalidated",
        "process_soft_nofile_observed": host_soft,
        "process_hard_nofile_observed": hard_text,
        "process_soft_nofile_required": required_process_soft_nofile,
        "process_nofile_sufficient": host_soft >= required_process_soft_nofile,
        "parent_limit_mutated": False,
        "app_server_shards": app_server_shards,
        "worker_width_configured": worker_width,
        "codex_executable": str(codex),
        "codex_executable_blake3": blake3_bytes(codex.read_bytes()),
        "codex_distribution_version": codex_distribution_version or "not_supplied",
        "external_prlimit_executable": str(limiter),
        "external_prlimit_prefix": [
            str(limiter),
            f"--nofile={required_process_soft_nofile}:{hard_text}",
            "--",
        ],
        "config_override_count": len(overrides),
        "config_overrides": list(overrides),
        "strict_config": True,
        "project_doc_max_bytes": 0,
        "uncommitted_ancestor_project_docs_enabled": False,
        "native_codex_surface": dict(CODEX_FIRST_RELEASE_NATIVE_SURFACE),
        "child_env_names": sorted((child_env or {}).keys()),
        "child_env_values_recorded": False,
        "sanitized_child_exec": dict(exec_plan.public_metadata()),
        "sanitized_child_exec_wrapper_blake3": blake3_bytes(wrapper.read_bytes()),
    }
    metadata = MappingProxyType(
        {**metadata_core, "launch_blake3": blake3_hex(metadata_core)}
    )
    runner = ShardedPersistentCodexRuntimeRunner(
        lambda: CodexRuntime(OpenAICodexBackend(launch)),
        shard_count=app_server_shards,
    )
    return PreparedPersistentCodexRuntime(
        runner=runner,
        launch_options=launch,
        startup_metadata=metadata,
    )


@dataclass(frozen=True, slots=True)
class CampaignConcurrency:
    """Explicit high-width controls; no implicit semaphore or retry is added."""

    worker_width: int = 128
    queue_capacity: int = 256
    claim_batch_size: int = 128
    app_server_shards: int = 1
    required_process_soft_nofile: int = DEFAULT_CHILD_SOFT_NOFILE
    lease_seconds: float = 1_800.0
    heartbeat_seconds: float = 30.0
    progress_seconds: float = 2.0
    maximum_parallel_models: int = 3
    maximum_parallel_tools: int = 64
    maximum_parallel_judge_tools: int = 64
    max_candidates: int | None = None

    def __post_init__(self) -> None:
        if self.maximum_parallel_models != 3:
            raise CampaignDeploymentError(
                "production cascade must run weak, middle, and strong concurrently"
            )
        for label, value, upper in (
            ("maximum_parallel_tools", self.maximum_parallel_tools, 64),
            ("maximum_parallel_judge_tools", self.maximum_parallel_judge_tools, 64),
        ):
            if type(value) is not int or not 1 <= value <= upper:
                raise CampaignDeploymentError(f"{label} must be in [1,{upper}]")
        if type(self.app_server_shards) is not int or not 1 <= self.app_server_shards <= 256:
            raise CampaignDeploymentError("app_server_shards must be in [1,256]")
        if self.app_server_shards > self.worker_width:
            raise CampaignDeploymentError("app-server shards cannot exceed worker width")
        if (
            type(self.required_process_soft_nofile) is not int
            or self.required_process_soft_nofile < 1_024
        ):
            raise CampaignDeploymentError("required process soft nofile differs")
        try:
            self.orchestration_config()
        except Exception as exc:
            raise CampaignDeploymentError("campaign concurrency configuration differs") from exc

    def orchestration_config(self) -> OrchestrationConfig:
        return OrchestrationConfig(
            worker_width=self.worker_width,
            queue_capacity=self.queue_capacity,
            claim_batch_size=self.claim_batch_size,
            lease_seconds=self.lease_seconds,
            heartbeat_seconds=self.heartbeat_seconds,
            progress_seconds=self.progress_seconds,
            max_candidates=self.max_candidates,
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "worker_width": self.worker_width,
            "queue_capacity": self.queue_capacity,
            "claim_batch_size": self.claim_batch_size,
            "app_server_shards": self.app_server_shards,
            "required_process_soft_nofile": self.required_process_soft_nofile,
            "lease_seconds": self.lease_seconds,
            "heartbeat_seconds": self.heartbeat_seconds,
            "progress_seconds": self.progress_seconds,
            "maximum_parallel_models": self.maximum_parallel_models,
            "maximum_parallel_tools": self.maximum_parallel_tools,
            "maximum_parallel_judge_tools": self.maximum_parallel_judge_tools,
            "max_candidates": self.max_candidates,
        }

    @classmethod
    def maximum_profile(cls) -> "CampaignConcurrency":
        """Opt-in 72-CPU host profile, enabled only after staged canaries."""

        return cls(
            worker_width=512,
            queue_capacity=1_024,
            claim_batch_size=512,
            app_server_shards=64,
            required_process_soft_nofile=DEFAULT_CHILD_SOFT_NOFILE,
            maximum_parallel_models=3,
            maximum_parallel_tools=64,
            maximum_parallel_judge_tools=64,
        )


def _absolute_path(value: Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or "\x00" in str(path):
        raise CampaignDeploymentError(f"{label} must be an absolute normalized path")
    return path


@dataclass(frozen=True, slots=True)
class CampaignDeploymentConfig:
    """Local durable paths, signer identity, and explicit campaign width."""

    workspace_root: Path
    artifact_root: Path
    ledger_path: Path
    receipt_root: Path
    admission_bundle_root: Path
    private_key_path: Path
    trust_store_path: Path
    signing_key_id: str
    concurrency: CampaignConcurrency = field(default_factory=CampaignConcurrency)
    separation_policy: SeparationPolicy = field(default_factory=_campaign_separation_policy)

    def __post_init__(self) -> None:
        for name in (
            "workspace_root",
            "artifact_root",
            "ledger_path",
            "receipt_root",
            "admission_bundle_root",
            "private_key_path",
            "trust_store_path",
        ):
            object.__setattr__(
                self,
                name,
                _absolute_path(getattr(self, name), label=name),
            )
        output_paths = (
            self.workspace_root,
            self.artifact_root,
            self.ledger_path,
            self.receipt_root,
            self.admission_bundle_root,
        )
        if len(set(output_paths)) != len(output_paths):
            raise CampaignDeploymentError("campaign output paths must be distinct")
        if not isinstance(self.signing_key_id, str) or not self.signing_key_id.strip():
            raise CampaignDeploymentError("signing key identity is required")
        if not isinstance(self.concurrency, CampaignConcurrency):
            raise CampaignDeploymentError("campaign concurrency contract differs")
        if not isinstance(self.separation_policy, SeparationPolicy):
            raise CampaignDeploymentError("separation policy contract differs")


def _separation_policy_document(value: SeparationPolicy) -> dict[str, Any]:
    return {
        "schema": value.admission_policy_schema,
        "minimum_valid_judged_trajectories": value.minimum_valid_judged_trajectories,
        "ability_separation_required_for_admission": (
            value.ability_separation_required_for_admission
        ),
        "early_continuation_after_first_valid_trajectory": (
            value.early_continuation_after_first_valid_trajectory
        ),
        "minimum_strong_reward": value.minimum_strong_reward,
        "minimum_strong_minus_weak": value.minimum_strong_minus_weak,
        "material_item_delta": value.material_item_delta,
        "require_strong_hard_gates": value.require_strong_hard_gates,
        "perfect_monotonic_staircase_required": False,
        "raw_scores_retained": True,
        "provenance_gate_lowered": False,
        "workspace_agent_judge_gate_lowered": False,
        "native_metric_gate_lowered": False,
        "signature_gate_lowered": False,
    }


IdFactoryFactory = Callable[[str], RuntimeIdFactory]


def _random_ids(_worker_id: str) -> RuntimeIdFactory:
    return RandomUUIDFactory()


@dataclass(frozen=True, slots=True)
class CampaignDeploymentPorts:
    """Injected I/O ports.  Construction performs no provider interaction."""

    plan: CampaignPlan | None
    candidate_source: FrozenCampaignSourcePort | None
    rubrics: CompiledRubricRegistry | None
    provider_tiers: CodexProviderTiers | None
    provider_health: Mapping[str, ProviderRouteHealth] | None
    runtime: PreparedPersistentCodexRuntime | None
    actor_options_factory: ActorOptionsFactory | None
    judge_options_factory: JudgeOptionsFactory | None
    turn_mcp_factory: TurnMCPBridgeFactoryPort | None
    execution_binding_resolver: ExecutionBindingResolverPort | None
    actor_skills_factory: ActorSkillCatalogPort | None
    opus5_adapter_gateway: Opus5AdapterGatewayPort | None
    prospective_execution_catalog_path: Path | None = None
    native_actor_stages: tuple[Stage, ...] = ()
    actor_tool_runtime_guard: Callable[[Any, Any], Any] | None = None
    id_factory_factory: IdFactoryFactory = _random_ids
    progress_callback: ProgressCallback | None = None


@dataclass(frozen=True, slots=True)
class CampaignDeploymentRecipe:
    """Safe, digest-committed proof of the zero-provider wiring operation."""

    plan_blake3: str
    rubric_registry_blake3: str
    source_registry_blake3: str
    selection_blake3: str
    actor_routes: tuple[tuple[str, str, str, str, str], ...]
    judge_target: tuple[str, str, str]
    provider_health_blake3: str
    route_assignment_receipt_blake3: str
    route_counts: Mapping[str, int]
    execution_binding_catalog_blake3: str
    executable_candidate_ids_blake3: str
    executable_binding_count: int
    prospective_execution_catalog_envelope_blake3: str | None
    prospective_execution_catalog_blake3: str | None
    prospective_executable_candidate_ids_blake3: str | None
    prospective_executable_count: int | None
    historical_attempt_evidence: Mapping[str, Any]
    claimable_candidate_ids_blake3: str
    historical_exclusion_count: int
    skill_mount_catalog_blake3: str
    skill_materialization_root: str
    skill_materialization_blake3: str
    skill_materialization_path_policy: str
    skill_materialization_count: int
    turn_mcp_launch_blake3: str
    turn_mcp_launch_metadata: Mapping[str, Any]
    opus5_adapter_binding_blake3: str
    opus5_adapter_shards: int
    opus5_adapter_aggregate_capacity: int
    concurrency: Mapping[str, Any]
    separation_policy: Mapping[str, Any]
    runtime_startup_metadata: Mapping[str, Any]
    build_provider_calls_made: int
    semantic_retry_count: int
    recipe_blake3: str
    schema: str = CAMPAIGN_DEPLOYMENT_SCHEMA

    def core_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_blake3": self.plan_blake3,
            "rubric_registry_blake3": self.rubric_registry_blake3,
            "source_registry_blake3": self.source_registry_blake3,
            "selection_blake3": self.selection_blake3,
            "frozen_schedule_count": FROZEN_SCHEDULE_TOTAL,
            "primary_selection_count": PRIMARY_SELECTION_TOTAL,
            "reserve_selection_count": RESERVE_SELECTION_TOTAL,
            "actor_routes": self.actor_routes,
            "judge_target": self.judge_target,
            "provider_health_blake3": self.provider_health_blake3,
            "route_assignment_receipt_blake3": self.route_assignment_receipt_blake3,
            "route_counts": self.route_counts,
            "execution_binding_catalog_blake3": self.execution_binding_catalog_blake3,
            "executable_candidate_ids_blake3": self.executable_candidate_ids_blake3,
            "executable_binding_count": self.executable_binding_count,
            "prospective_execution_catalog_envelope_blake3": self.prospective_execution_catalog_envelope_blake3,
            "prospective_execution_catalog_blake3": self.prospective_execution_catalog_blake3,
            "prospective_executable_candidate_ids_blake3": self.prospective_executable_candidate_ids_blake3,
            "prospective_executable_count": self.prospective_executable_count,
            "historical_attempt_evidence": self.historical_attempt_evidence,
            "claimable_candidate_ids_blake3": self.claimable_candidate_ids_blake3,
            "historical_exclusion_count": self.historical_exclusion_count,
            "execution_binding_coverage": (
                "signed_v24_promoted_intersect_signed_prospective_v2"
                if self.prospective_execution_catalog_blake3 is not None
                else "signed_v24_promoted_only"
            ),
            "skill_mount_catalog_blake3": self.skill_mount_catalog_blake3,
            "skill_materialization_root": self.skill_materialization_root,
            "skill_materialization_blake3": self.skill_materialization_blake3,
            "skill_materialization_path_policy": (
                self.skill_materialization_path_policy
            ),
            "skill_materialization_count": self.skill_materialization_count,
            "turn_mcp_launch_blake3": self.turn_mcp_launch_blake3,
            "turn_mcp_launch_metadata": self.turn_mcp_launch_metadata,
            "opus5_adapter_binding_blake3": self.opus5_adapter_binding_blake3,
            "opus5_adapter_shards": self.opus5_adapter_shards,
            "opus5_adapter_aggregate_capacity": self.opus5_adapter_aggregate_capacity,
            "opus5_adapter_required_capacity": 4 * self.concurrency["worker_width"],
            "actor_runtime_surface": {
                "tools": "exact_candidate_policy_registry",
                "skills": "verified_stage_skill_mounts",
                "dynamic_skill_discovery_tools_exposed": False,
                "codex_sandbox": "read-only",
                "candidate_mutations": "exact_candidate_policy_mcp_only",
                "native_codex_surface": self.runtime_startup_metadata[
                    "native_codex_surface"
                ],
                "project_doc_max_bytes": 0,
                "ancestor_project_docs_injected": False,
            },
            "concurrency": self.concurrency,
            "separation_policy": self.separation_policy,
            "runtime_startup_metadata": self.runtime_startup_metadata,
            "runtime_strategy": "sharded_persistent_app_servers_fresh_ephemeral_threads",
            "actor_tool_bridge": "candidate_scoped_exactly_once_mcp",
            "judge_tool_bridge": "read_only_workspace_replay_mcp",
            "build_provider_calls_made": self.build_provider_calls_made,
            "semantic_retry_count": self.semantic_retry_count,
        }

    def to_document(self) -> dict[str, Any]:
        return {**self.core_document(), "recipe_blake3": self.recipe_blake3}


def _verify_turn_mcp_launch_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_blake3: str,
    expected_width: int,
) -> None:
    if (
        set(metadata) != TURN_MCP_LAUNCH_METADATA_KEYS
        or metadata.get("schema") != "eva.turn-mcp-bridge-launch.v1"
        or metadata.get("protocol_version") != TURN_MCP_PROTOCOL_VERSION
        or metadata.get("server_version") != TURN_MCP_SERVER_VERSION
        or metadata.get("launch_blake3") != expected_blake3
        or metadata.get("maximum_parallel_calls") != expected_width
        or type(metadata.get("startup_timeout_seconds")) not in {int, float}
        or not 0 < metadata["startup_timeout_seconds"] <= 120
        or type(metadata.get("tool_timeout_seconds")) is not int
        or not 1 <= metadata["tool_timeout_seconds"] <= 86_400
        or tuple(metadata.get("final_child_environment_names", ()))
        != TURN_MCP_CHILD_ENVIRONMENT_NAMES
        or metadata.get("inherited_parent_environment") is not False
        or metadata.get("environment_values_recorded") is not False
        or metadata.get("turn_nonce_recorded") is not False
        or not is_blake3(expected_blake3)
        or blake3_hex(
            {key: value for key, value in metadata.items() if key != "launch_blake3"}
        )
        != expected_blake3
    ):
        raise CampaignDeploymentError("turn MCP launch commitment differs")
    socket_preflight = metadata.get("unix_socket_preflight")
    if not isinstance(socket_preflight, Mapping) or set(
        socket_preflight
    ) != TURN_MCP_UNIX_SOCKET_PREFLIGHT_KEYS:
        raise CampaignDeploymentError("turn MCP Unix socket preflight differs")
    root_value = socket_preflight.get("temp_root")
    random_component_bytes = socket_preflight.get("random_component_bytes")
    if not isinstance(root_value, str) or type(random_component_bytes) is not int:
        raise CampaignDeploymentError("turn MCP Unix socket preflight differs")
    root = Path(root_value)
    try:
        root_metadata = root.lstat()
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise CampaignDeploymentError("turn MCP private runtime root is unavailable") from exc
    if (
        socket_preflight.get("schema")
        != "eva.turn-mcp-unix-socket-preflight.v1"
        or not root.is_absolute()
        or resolved_root != root
        or root.is_symlink()
        or not root.is_dir()
        or root_metadata.st_uid != os.getuid()
        or stat.S_IMODE(root_metadata.st_mode) != TURN_MCP_PRIVATE_ROOT_MODE
        or socket_preflight.get("temp_root_uid") != os.getuid()
        or socket_preflight.get("temp_root_mode") != TURN_MCP_PRIVATE_ROOT_MODE
        or socket_preflight.get("temp_root_owned_by_process") is not True
        or socket_preflight.get("temp_root_is_symlink") is not False
        or socket_preflight.get("directory_prefix") != TURN_MCP_DIRECTORY_PREFIX
        or not 1 <= random_component_bytes <= 64
        or socket_preflight.get("socket_filename") != TURN_MCP_SOCKET_FILENAME
        or socket_preflight.get("sockaddr_un_path_capacity_bytes")
        != TURN_MCP_UNIX_SOCKET_PATH_CAPACITY_BYTES
        or socket_preflight.get("terminator_bytes") != 1
        or socket_preflight.get("name_length_source")
        != "tempfile.mkdtemp_probe_removed"
        or socket_preflight.get("preflight_passed") is not True
    ):
        raise CampaignDeploymentError("turn MCP private runtime root differs")
    expected_socket_path = root / (
        TURN_MCP_DIRECTORY_PREFIX + ("X" * random_component_bytes)
    ) / TURN_MCP_SOCKET_FILENAME
    expected_path_bytes = len(os.fsencode(str(expected_socket_path)))
    if (
        socket_preflight.get("probed_socket_path_bytes") != expected_path_bytes
        or expected_path_bytes >= TURN_MCP_UNIX_SOCKET_PATH_CAPACITY_BYTES
    ):
        raise CampaignDeploymentError("turn MCP Unix socket path exceeds platform capacity")
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise CampaignDeploymentError(
            "turn MCP private runtime root cannot be reopened safely"
        ) from exc
    try:
        reopened_root = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        reopened_root.st_dev != root_metadata.st_dev
        or reopened_root.st_ino != root_metadata.st_ino
        or reopened_root.st_uid != root_metadata.st_uid
        or stat.S_IMODE(reopened_root.st_mode) != TURN_MCP_PRIVATE_ROOT_MODE
        or not stat.S_ISDIR(reopened_root.st_mode)
    ):
        raise CampaignDeploymentError("turn MCP private runtime root changed")
    for path_key, digest_key, executable in (
        ("proxy_python", "proxy_python_blake3", True),
        ("proxy_exec_script", "proxy_exec_script_blake3", False),
        ("proxy_script", "proxy_script_blake3", False),
    ):
        value = metadata.get(path_key)
        digest = metadata.get(digest_key)
        if not isinstance(value, str):
            raise CampaignDeploymentError("turn MCP launch path differs")
        path = Path(value)
        if (
            not path.is_absolute()
            or path != path.resolve()
            or path.is_symlink()
            or not path.is_file()
            or (executable and not os.access(path, os.X_OK))
            or not is_blake3(digest)
            or blake3_bytes(path.read_bytes()) != digest
        ):
            raise CampaignDeploymentError("turn MCP launch executable bytes differ")


def _verify_recipe(recipe: CampaignDeploymentRecipe) -> None:
    if recipe.schema != CAMPAIGN_DEPLOYMENT_SCHEMA:
        raise CampaignDeploymentError("campaign deployment recipe schema differs")
    if not all(
        is_blake3(value)
        for value in (
            recipe.plan_blake3,
            recipe.rubric_registry_blake3,
            recipe.source_registry_blake3,
            recipe.selection_blake3,
            recipe.provider_health_blake3,
            recipe.route_assignment_receipt_blake3,
            recipe.execution_binding_catalog_blake3,
            recipe.executable_candidate_ids_blake3,
            recipe.claimable_candidate_ids_blake3,
            recipe.skill_mount_catalog_blake3,
            recipe.skill_materialization_blake3,
            recipe.turn_mcp_launch_blake3,
            recipe.opus5_adapter_binding_blake3,
            recipe.recipe_blake3,
        )
    ):
        raise CampaignDeploymentError("campaign deployment recipe commitment differs")
    if recipe.build_provider_calls_made != 0 or recipe.semantic_retry_count != 0:
        raise CampaignDeploymentError("campaign deployment may not hide calls or retries")
    materialization_root = Path(recipe.skill_materialization_root)
    if (
        not materialization_root.is_absolute()
        or materialization_root != materialization_root.resolve()
        or recipe.skill_materialization_path_policy
        != ACTOR_SKILL_MATERIALIZATION_PATH_POLICY
        or type(recipe.skill_materialization_count) is not int
        or recipe.skill_materialization_count < 1
    ):
        raise CampaignDeploymentError("actor skill materialization receipt differs")
    turn_mcp = recipe.turn_mcp_launch_metadata
    if not isinstance(turn_mcp, Mapping):
        raise CampaignDeploymentError("turn MCP launch commitment differs")
    _verify_turn_mcp_launch_metadata(
        turn_mcp,
        expected_blake3=recipe.turn_mcp_launch_blake3,
        expected_width=recipe.concurrency["maximum_parallel_tools"],
    )
    if (
        recipe.runtime_startup_metadata.get("parent_limit_mutated") is not False
        or type(recipe.executable_binding_count) is not int
        or recipe.executable_binding_count < 1
        or type(recipe.opus5_adapter_shards) is not int
        or recipe.opus5_adapter_shards < 1
        or type(recipe.opus5_adapter_aggregate_capacity) is not int
        or recipe.opus5_adapter_aggregate_capacity
        < 4 * recipe.concurrency.get("worker_width", 0)
    ):
        raise CampaignDeploymentError("campaign runtime/binding capacity differs")
    prospective_values = (
        recipe.prospective_execution_catalog_envelope_blake3,
        recipe.prospective_execution_catalog_blake3,
        recipe.prospective_executable_candidate_ids_blake3,
        recipe.prospective_executable_count,
    )
    if any(value is not None for value in prospective_values):
        if (
            not all(
                is_blake3(value)
                for value in prospective_values[:3]
            )
            or type(recipe.prospective_executable_count) is not int
            or recipe.prospective_executable_count != recipe.executable_binding_count
        ):
            raise CampaignDeploymentError("prospective execution commitment differs")
    if _separation_policy_document(_campaign_separation_policy()) != dict(
        recipe.separation_policy
    ):
        # An explicit policy override is permitted only through config, but the
        # production default receipt must never accidentally drift.  The
        # builder performs this check for the default campaign below.
        required_keys = set(_separation_policy_document(_campaign_separation_policy()))
        if set(recipe.separation_policy) != required_keys:
            raise CampaignDeploymentError("campaign separation policy receipt differs")
    if blake3_hex(recipe.core_document()) != recipe.recipe_blake3:
        raise CampaignDeploymentError("campaign deployment recipe BLAKE3 differs")


def _verify_rubrics(plan: CampaignPlan, rubrics: CompiledRubricRegistry) -> None:
    def stage_text(value: Any) -> str:
        return value.value if hasattr(value, "value") else str(value)

    expected = {(cell.domain, cell.stage) for cell in plan.cells}
    actual = {(rubric.domain, stage_text(rubric.stage)) for rubric in rubrics.rubrics}
    if actual != expected or len(rubrics.rubrics) != 42:
        raise CampaignDeploymentError("compiled rubric registry differs from all 42 plan cells")
    for domain, stage in sorted(expected):
        rubric = rubrics.resolve(domain, stage)
        if rubric is not next(
            row
            for row in rubrics.rubrics
            if row.domain == domain and stage_text(row.stage) == stage
        ):
            raise CampaignDeploymentError("rubric registry lookup did not preserve exact object")
        if not 5 <= len(rubric.items) <= 10:
            raise CampaignDeploymentError("compiled rubric item count differs")


def _verify_frozen_source(
    source: FrozenCampaignSourcePort,
    *,
    plan: CampaignPlan,
    rubrics: CompiledRubricRegistry,
) -> None:
    if not callable(getattr(source, "load", None)):
        raise CampaignDeploymentError("frozen candidate source lacks load")
    if not callable(getattr(source, "queue_records", None)):
        raise CampaignDeploymentError("frozen candidate source lacks 9,000-row queue records")
    if not callable(getattr(source, "source_candidate_id", None)):
        raise CampaignDeploymentError(
            "frozen candidate source lacks execution-binding identity mapping"
        )
    if (
        getattr(source, "candidate_count", None) != FROZEN_SCHEDULE_TOTAL
        or getattr(source, "scheduled_count", None) != FROZEN_SCHEDULE_TOTAL
        or getattr(source, "primary_count", None) != plan.total
        or getattr(source, "reserve_count", None) != RESERVE_SELECTION_TOTAL
    ):
        raise CampaignDeploymentError("frozen candidate source count differs from 9,000")
    if getattr(source, "plan_blake3", None) != plan.plan_blake3:
        raise CampaignDeploymentError("frozen candidate source plan commitment differs")
    if getattr(source, "rubric_registry_blake3", None) != rubrics.digest:
        raise CampaignDeploymentError("frozen candidate source rubric commitment differs")
    for name in ("source_registry_blake3", "selection_blake3"):
        if not is_blake3(getattr(source, name, None)):
            raise CampaignDeploymentError(f"frozen candidate source {name} differs")
    identities = getattr(source, "candidate_ids", None)
    primary = getattr(source, "primary_candidate_ids", None)
    reserve = getattr(source, "reserve_candidate_ids", None)
    if (
        not isinstance(identities, tuple)
        or not isinstance(primary, tuple)
        or not isinstance(reserve, tuple)
        or len(identities) != FROZEN_SCHEDULE_TOTAL
        or len(primary) != plan.total
        or len(reserve) != RESERVE_SELECTION_TOTAL
        or len(set(identities)) != FROZEN_SCHEDULE_TOTAL
        or set(primary).intersection(reserve)
        or set(primary).union(reserve) != set(identities)
    ):
        raise CampaignDeploymentError("frozen primary/reserve candidate inventory differs")
    for candidate_id in identities:
        source_id = source.source_candidate_id(candidate_id)
        if not isinstance(source_id, str) or not source_id:
            raise CampaignDeploymentError("frozen source execution identity differs")


def _verify_signer(config: CampaignDeploymentConfig, *, challenge: bytes) -> None:
    """Prove the configured private key matches the active public trust entry."""

    path = config.private_key_path
    if path.is_symlink() or not path.is_file():
        raise CampaignDeploymentError("admission private key topology differs")
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise CampaignDeploymentError("admission private key ownership or permissions differ")
    try:
        private = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except (OSError, TypeError, ValueError) as exc:
        raise CampaignDeploymentError("admission private key could not be loaded") from exc
    if not isinstance(private, Ed25519PrivateKey):
        raise CampaignDeploymentError("admission private key is not Ed25519")
    try:
        trusted = load_trust_store(config.trust_store_path)
    except Exception as exc:
        raise CampaignDeploymentError("admission trust store verification failed") from exc
    public = trusted.get(config.signing_key_id)
    if public is None:
        raise CampaignDeploymentError("admission signing key is not active in trust store")
    try:
        public.verify(private.sign(challenge), challenge)
    except InvalidSignature as exc:
        raise CampaignDeploymentError("admission private key does not match trust store") from exc


def _verify_execution_binding_inventory(
    resolver: ExecutionBindingResolverPort,
    *,
    source: FrozenCampaignSourcePort,
) -> tuple[tuple[str, ...], str, str]:
    """Bind the signed v24-promoted inventory to exact EVA candidate UUIDs."""

    if not all(
        callable(getattr(resolver, name, None))
        for name in ("inventory", "resolve")
    ):
        raise CampaignDeploymentError("candidate execution binding resolver is missing")
    catalog_blake3 = getattr(resolver, "catalog_inventory_blake3", None)
    count = getattr(resolver, "executable_candidate_count", None)
    if not is_blake3(catalog_blake3) or count != EXPECTED_EXECUTABLE_BINDINGS:
        raise CampaignDeploymentError("signed execution binding catalog differs")
    inventory = resolver.inventory()
    if not isinstance(inventory, tuple) or len(inventory) != count:
        raise CampaignDeploymentError("execution binding inventory count differs")
    source_to_eva = {
        source.source_candidate_id(candidate_id): candidate_id
        for candidate_id in source.candidate_ids
    }
    if len(source_to_eva) != FROZEN_SCHEDULE_TOTAL:
        raise CampaignDeploymentError("frozen source identity mapping differs")
    source_ids: list[str] = []
    for row in inventory:
        source_id = getattr(row, "source_candidate_id", None)
        if not isinstance(source_id, str) or source_id not in source_to_eva:
            raise CampaignDeploymentError(
                "execution binding is outside the frozen schedule"
            )
        source_ids.append(source_id)
    if len(set(source_ids)) != count:
        raise CampaignDeploymentError("execution binding inventory is duplicated")
    candidate_ids = tuple(sorted(source_to_eva[source_id] for source_id in source_ids))
    allowlist_blake3 = blake3_hex(
        {
            "schema": "eva.execution-claim-allowlist.v1",
            "selection_blake3": source.selection_blake3,
            "execution_binding_catalog_blake3": catalog_blake3,
            "candidate_ids": candidate_ids,
        }
    )
    return candidate_ids, catalog_blake3, allowlist_blake3


def _verify_prospective_execution_allowlist(
    path: Path,
    *,
    trust_store_path: Path,
    source: FrozenCampaignSourcePort,
    legacy_catalog_blake3: str,
    legacy_candidate_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], str, str, str]:
    """Reopen a signed prospective catalog and derive its executable subset.

    This intentionally does not trust the summary counters.  Every frozen row,
    row digest, source identity, status and the catalog/envelope commitments are
    reopened before the allowlist is used.  The independently-built catalog is
    an additional restriction: it can demote a legacy binding, never add one.
    """

    target = Path(path)
    try:
        info = target.lstat()
        if (
            target.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise CampaignDeploymentError("prospective execution catalog topology differs")
        document = json.loads(target.read_text(encoding="utf-8"))
        envelope = SignedEnvelope.from_document(document)
        verify_signed_envelope(envelope, trust_store_path=trust_store_path)
    except CampaignDeploymentError:
        raise
    except Exception as exc:
        raise CampaignDeploymentError("prospective execution catalog signature differs") from exc
    payload = envelope.payload
    catalog_blake3 = payload.get("catalog_blake3")
    rows = payload.get("rows")
    if (
        payload.get("schema") != PROSPECTIVE_CATALOG_SCHEMA
        or type(payload.get("catalog_revision")) is not int
        or payload.get("catalog_revision", 0) < 1
        or payload.get("selection_blake3") != source.selection_blake3
        or payload.get("source_registry_blake3") != source.source_registry_blake3
        or payload.get("ancestor_legacy_catalog_blake3") != legacy_catalog_blake3
        or payload.get("scheduled_count") != FROZEN_SCHEDULE_TOTAL
        or not is_blake3(catalog_blake3)
        or not isinstance(rows, (tuple, list))
        or len(rows) != FROZEN_SCHEDULE_TOTAL
    ):
        raise CampaignDeploymentError("prospective execution catalog commitments differ")
    controls = payload.get("controls")
    if (
        not isinstance(controls, Mapping)
        or any(
            controls.get(key) != 0
            for key in (
                "provider_calls",
                "ledger_reads",
                "ledger_writes",
                "launch_reads",
                "launch_writes",
            )
        )
        or controls.get("source_only_is_executable") is not False
        or controls.get("rejected_is_executable") is not False
        or controls.get("full_binding_validation_required") is not True
    ):
        raise CampaignDeploymentError("prospective execution catalog controls differ")
    catalog_core = {key: value for key, value in payload.items() if key != "catalog_blake3"}
    if blake3_hex(catalog_core) != catalog_blake3:
        raise CampaignDeploymentError("prospective execution catalog BLAKE3 differs")
    expected_ids = set(source.candidate_ids)
    seen: set[str] = set()
    executable: list[str] = []
    allowed_statuses = {"executable_legacy", "source_only", "rejected"}
    status_counts = {status: 0 for status in allowed_statuses}
    for row in rows:
        if not isinstance(row, Mapping):
            raise CampaignDeploymentError("prospective execution catalog row differs")
        candidate_id = row.get("candidate_id")
        source_candidate_id = row.get("source_candidate_id")
        status = row.get("execution_status")
        row_blake3 = row.get("row_blake3")
        row_core = {key: value for key, value in row.items() if key != "row_blake3"}
        if (
            not isinstance(candidate_id, str)
            or candidate_id not in expected_ids
            or candidate_id in seen
            or source_candidate_id != source.source_candidate_id(candidate_id)
            or status not in allowed_statuses
            or not is_blake3(row_blake3)
            or blake3_hex(row_core) != row_blake3
        ):
            raise CampaignDeploymentError("prospective execution catalog row differs")
        seen.add(candidate_id)
        status_counts[status] += 1
        if status == "executable_legacy":
            proof = row.get("execution_proof")
            if (
                not isinstance(proof, Mapping)
                or proof.get("resolver_catalog_blake3") != legacy_catalog_blake3
                or row.get("execution_blocker") is not None
            ):
                raise CampaignDeploymentError("prospective executable proof differs")
            executable.append(candidate_id)
    if (
        seen != expected_ids
        or len(executable) != payload.get("executable_count")
        or status_counts["source_only"] != payload.get("source_only_count")
        or status_counts["rejected"] != payload.get("rejected_count")
    ):
        raise CampaignDeploymentError("prospective execution inventory differs")
    legacy_set = frozenset(legacy_candidate_ids)
    if any(candidate_id not in legacy_set for candidate_id in executable):
        raise CampaignDeploymentError("prospective catalog expands legacy execution coverage")
    executable_ids = tuple(sorted(executable))
    ids_blake3 = blake3_hex(
        {
            "schema": "eva.prospective-execution-claim-allowlist.v1",
            "selection_blake3": source.selection_blake3,
            "legacy_execution_catalog_blake3": legacy_catalog_blake3,
            "prospective_execution_catalog_blake3": catalog_blake3,
            "signed_envelope_blake3": envelope.envelope_blake3,
            "candidate_ids": executable_ids,
        }
    )
    return executable_ids, catalog_blake3, envelope.envelope_blake3, ids_blake3


class _ExactCandidateSource:
    """Validate each lazy job against the exact registry and tier objects."""

    def __init__(
        self,
        source: FrozenCampaignSourcePort,
        *,
        plan: CampaignPlan,
        rubrics: CompiledRubricRegistry,
        route_selector: DeterministicRouteSelector,
        binding_resolver: ExecutionBindingResolverPort,
        executable_candidate_ids: tuple[str, ...],
    ) -> None:
        self._source = source
        self._plan_cells = {(cell.domain, cell.stage) for cell in plan.cells}
        self._rubrics = rubrics
        self._route_selector = route_selector
        self._binding_resolver = binding_resolver
        self._eligible = frozenset(executable_candidate_ids)
        self._bindings: dict[str, CandidateExecutionBindingPort] = {}
        self._binding_lock = RLock()

    def _binding_for_id(self, candidate_id: str) -> CandidateExecutionBindingPort:
        if candidate_id not in self._eligible:
            raise CampaignDeploymentError(
                "candidate lacks a signed executable v24-promoted binding"
            )
        with self._binding_lock:
            cached = self._bindings.get(candidate_id)
            if cached is not None:
                return cached
        source_id = self._source.source_candidate_id(candidate_id)
        binding = self._binding_resolver.resolve(
            candidate_id, source_candidate_id=source_id
        )
        schemas = getattr(binding.tool_registry, "public_schemas", None)
        if (
            binding.candidate_id != candidate_id
            or binding.source_candidate_id != source_id
            or not is_blake3(binding.binding_blake3)
            or not isinstance(binding.initial_workspace_files, Mapping)
            or not isinstance(binding.public_runtime_context, Mapping)
            or not callable(schemas)
            or not schemas()
        ):
            raise CampaignDeploymentError("candidate execution binding differs")
        with self._binding_lock:
            return self._bindings.setdefault(candidate_id, binding)

    def binding_for(self, job: CandidateJob) -> CandidateExecutionBindingPort:
        if not isinstance(job, CandidateJob):
            raise CampaignDeploymentError("candidate binding job differs")
        return self._binding_for_id(job.candidate_id)

    def load(self, candidate_id: str) -> CandidateJob:
        job = self._source.load(candidate_id)
        if not isinstance(job, CandidateJob) or job.candidate_id != candidate_id:
            raise CampaignDeploymentError("frozen candidate job identity differs")
        key = (job.episode.domain, job.episode.stage.value)
        if key not in self._plan_cells:
            raise CampaignDeploymentError("frozen candidate job is outside the final plan")
        expected_rubric = self._rubrics.resolve(*key)
        if job.rubric is not expected_rubric:
            raise CampaignDeploymentError(
                "candidate must use the exact compiled rubric registry object"
            )
        if len(job.targets) != 3 or {target.cohort for target in job.targets} != set(Cohort):
            raise CampaignDeploymentError("candidate provider cohort inventory differs")
        binding = self._binding_for_id(candidate_id)
        initial_files = dict(job.episode.initial_files)
        for path, payload in binding.initial_workspace_files.items():
            if path in initial_files and initial_files[path] != payload:
                raise CampaignDeploymentError(
                    "candidate binding collides with frozen initial workspace"
                )
            initial_files[path] = payload
        policy_context = dict(job.episode.policy_context)
        if "execution_binding" in policy_context:
            raise CampaignDeploymentError(
                "frozen policy context shadows candidate execution binding"
            )
        policy_context["execution_binding"] = binding.public_runtime_context
        episode = replace(
            job.episode,
            initial_files=initial_files,
            policy_context=policy_context,
        )
        # FrozenCampaignCandidateSource historically accepts one static trio.
        # Replace only that deployment-time transport binding; the episode,
        # rubric object, manifest schemas, and one-rollout-per-cohort contract
        # remain byte-for-byte unchanged.
        return replace(
            job,
            episode=episode,
            targets=self._route_selector.targets_for(candidate_id),
        )


def _verified_actor_skill_materialization(
    catalog: ActorSkillCatalogPort,
    *,
    workspace_root: Path,
) -> tuple[str, str, str, int]:
    """Reopen every stage-mounted SkillInput from its immutable runtime tree."""

    root_value = getattr(catalog, "materialization_root", None)
    digest = getattr(catalog, "materialization_blake3", None)
    policy = getattr(catalog, "materialization_path_policy", None)
    if not isinstance(root_value, Path):
        raise CampaignDeploymentError("actor skill materialization root is required")
    root = root_value
    if (
        not root.is_absolute()
        or root != root.resolve()
        or root.name != f"blake3-{digest}"
        or not is_blake3(digest)
        or policy != ACTOR_SKILL_MATERIALIZATION_PATH_POLICY
        or root == workspace_root
        or root.is_relative_to(workspace_root)
        or workspace_root.is_relative_to(root)
    ):
        raise CampaignDeploymentError("actor skill materialization binding differs")
    try:
        root_stat = root.lstat()
        runtime_stat = root.parent.lstat()
    except OSError as exc:
        raise CampaignDeploymentError(
            "actor skill materialization tree is unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or stat.S_ISLNK(root_stat.st_mode)
        or stat.S_IMODE(root_stat.st_mode) != 0o500
        or root_stat.st_uid != os.geteuid()
        or not stat.S_ISDIR(runtime_stat.st_mode)
        or stat.S_ISLNK(runtime_stat.st_mode)
        or stat.S_IMODE(runtime_stat.st_mode) != 0o700
        or runtime_stat.st_uid != os.geteuid()
    ):
        raise CampaignDeploymentError("actor skill materialization permissions differ")
    if not callable(getattr(catalog, "for_stage", None)):
        raise CampaignDeploymentError("actor skill stage materialization is unavailable")

    skills: dict[str, CodexSkill] = {}
    for stage in Stage:
        mounted = catalog.for_stage(stage)
        if not isinstance(mounted, tuple) or not mounted:
            raise CampaignDeploymentError("actor skill stage mount inventory differs")
        for skill in mounted:
            if not isinstance(skill, CodexSkill):
                raise CampaignDeploymentError("actor skill mount type differs")
            prior = skills.get(skill.skill_id)
            if prior is not None and prior != skill:
                raise CampaignDeploymentError("actor skill identity is not stable across stages")
            skills[skill.skill_id] = skill

    inventory = catalog.inventory()
    inventory_paths = {
        row.get("path")
        for row in inventory
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    if len(inventory_paths) != len(inventory) or not inventory_paths:
        raise CampaignDeploymentError("actor skill source inventory paths differ")
    mounted_paths = {skill.path for skill in skills.values()}
    if not inventory_paths.issubset(mounted_paths):
        raise CampaignDeploymentError("actor skill source inventory is not stage-mounted")

    expected_directories: set[str] = set()
    for skill in skills.values():
        path = Path(skill.path)
        try:
            relative = path.relative_to(root)
            file_stat = path.lstat()
            directory_stat = path.parent.lstat()
        except (OSError, ValueError) as exc:
            raise CampaignDeploymentError("actor skill path escapes materialization") from exc
        expected_directory = path.parent.name
        expected_directories.add(expected_directory)
        if (
            len(relative.parts) != 2
            or relative.parts[1] != "SKILL.md"
            or path.resolve(strict=True) != path
            or path.parent.resolve(strict=True) != path.parent
            or not expected_directory.endswith(f"-{skill.content_blake3}")
            or not stat.S_ISREG(file_stat.st_mode)
            or stat.S_ISLNK(file_stat.st_mode)
            or stat.S_IMODE(file_stat.st_mode) != 0o400
            or file_stat.st_uid != os.geteuid()
            or file_stat.st_nlink != 1
            or not stat.S_ISDIR(directory_stat.st_mode)
            or stat.S_ISLNK(directory_stat.st_mode)
            or stat.S_IMODE(directory_stat.st_mode) != 0o500
            or directory_stat.st_uid != os.geteuid()
            or set(os.listdir(path.parent)) != {"SKILL.md"}
            or blake3_bytes(path.read_bytes()) != skill.content_blake3
        ):
            raise CampaignDeploymentError("actor skill materialized bytes differ")
    if set(os.listdir(root)) != expected_directories:
        raise CampaignDeploymentError("actor skill materialization inventory differs")
    ordinals = sorted(name.split("-", 1)[0] for name in expected_directories)
    if ordinals != [f"{index:03d}" for index in range(len(ordinals))]:
        raise CampaignDeploymentError("actor skill materialization ordinals differ")
    reopened_root_stat = root.lstat()
    stable_root_identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if stable_root_identity(reopened_root_stat) != stable_root_identity(root_stat):
        raise CampaignDeploymentError("actor skill materialization changed while reopening")
    return str(root), digest, policy, len(skills)


def _require_base_options(options: Any, *, label: str) -> CodexThreadOptions:
    if not isinstance(options, CodexThreadOptions):
        raise CodexPipelineError(f"{label} transport did not return CodexThreadOptions")
    if not isinstance(options.config, Mapping) or "mcp_servers" in options.config:
        raise CodexPipelineError(
            f"{label} base options pre-mount MCP; authenticated turn bridge owns it"
        )
    if options.sandbox is not CodexSandbox.READ_ONLY:
        raise CodexPipelineError(
            f"{label} base options must use read-only Codex sandbox; "
            "candidate mutations belong to the authenticated MCP runtime"
        )
    features = options.config.get("features")
    expected_features = CODEX_FIRST_RELEASE_THREAD_CONFIG["features"]
    if (
        options.config.get("project_doc_max_bytes") != 0
        or options.config.get("web_search") != "disabled"
        or not isinstance(features, Mapping)
        or any(features.get(name) is not False for name in expected_features)
    ):
        raise CodexPipelineError(
            f"{label} base options enable an uncommitted Codex surface"
        )
    return options


@dataclass(slots=True)
class CodexCampaignDeployment:
    """Fully composed, single-use production campaign with explicit lifecycle."""

    recipe: CampaignDeploymentRecipe
    plan: CampaignPlan
    rubrics: CompiledRubricRegistry
    provider_tiers: CodexProviderTiers
    route_selector: DeterministicRouteSelector
    route_assignment_receipt: RouteAssignmentReceipt
    frozen_source: FrozenCampaignSourcePort
    candidate_source: CandidateSource
    execution_binding_resolver: ExecutionBindingResolverPort
    opus5_adapter_gateway: Opus5AdapterGatewayPort
    executable_candidate_ids: tuple[str, ...]
    prospective_execution_catalog_path: Path | None
    prepared_runtime: PreparedPersistentCodexRuntime
    runtime_runner: PersistentCodexRunnerPort
    ledger: CampaignLedger
    artifact_store: ImmutableArtifactStore
    pipeline_verifier: PipelineVerifier
    supervisor: Ed25519AdmissionSupervisor
    signature_verifier: Ed25519SupervisorSignatureVerifier
    pipeline_factory: Callable[[str, CandidateJob], VerifiableDataPipeline]
    orchestrator: CampaignOrchestrator

    def install_frozen_schedule(self, *, worker_width: int = 128) -> int:
        """Source-verify and atomically install all 9,000 primary+reserve rows."""

        if type(worker_width) is not int or worker_width < 1:
            raise CampaignDeploymentError("queue verification worker width differs")
        records = self.frozen_source.queue_records(worker_width=worker_width)
        if not isinstance(records, tuple) or len(records) != FROZEN_SCHEDULE_TOTAL:
            raise CampaignDeploymentError("frozen queue record inventory differs")
        return self.ledger.bootstrap_candidates(
            records,
            selection_blake3=self.recipe.selection_blake3,
        )

    def run_until_idle(self):
        """Start app-server before any claim, then drain and close it exactly once."""

        candidate_ids, catalog_blake3, allowlist_blake3 = (
            _verify_execution_binding_inventory(
                self.execution_binding_resolver,
                source=self.frozen_source,
            )
        )
        if self.prospective_execution_catalog_path is not None:
            (
                candidate_ids,
                prospective_catalog_blake3,
                prospective_envelope_blake3,
                allowlist_blake3,
            ) = _verify_prospective_execution_allowlist(
                self.prospective_execution_catalog_path,
                trust_store_path=self.supervisor._trust_store_path,
                source=self.frozen_source,
                legacy_catalog_blake3=catalog_blake3,
                legacy_candidate_ids=candidate_ids,
            )
            if (
                prospective_catalog_blake3
                != self.recipe.prospective_execution_catalog_blake3
                or prospective_envelope_blake3
                != self.recipe.prospective_execution_catalog_envelope_blake3
                or allowlist_blake3
                != self.recipe.prospective_executable_candidate_ids_blake3
            ):
                raise CampaignDeploymentError(
                    "prospective execution coverage changed before claim"
                )
        if (
            candidate_ids != self.executable_candidate_ids
            or catalog_blake3 != self.recipe.execution_binding_catalog_blake3
            or allowlist_blake3 != self.recipe.executable_candidate_ids_blake3
        ):
            raise CampaignDeploymentError(
                "execution binding coverage changed before claim"
            )
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        required = self.prepared_runtime.startup_metadata[
            "process_soft_nofile_required"
        ]
        if soft < required:
            raise CampaignDeploymentError(
                "campaign process nofile dropped below committed startup capacity"
            )
        self.opus5_adapter_gateway.start()
        runtime_started = False
        try:
            self.runtime_runner.start()
            runtime_started = True
            return self.orchestrator.run_until_idle()
        finally:
            try:
                if runtime_started:
                    self.runtime_runner.close()
            finally:
                self.opus5_adapter_gateway.close()

    def request_stop(self, reason: str = "operator_requested") -> None:
        self.orchestrator.request_stop(reason)


def build_codex_campaign_deployment(
    *,
    config: CampaignDeploymentConfig,
    ports: CampaignDeploymentPorts,
) -> CodexCampaignDeployment:
    """Compose the exact 6,000-training run without starting Codex or a provider."""

    if not isinstance(config, CampaignDeploymentConfig) or not isinstance(
        ports, CampaignDeploymentPorts
    ):
        raise CampaignDeploymentError("campaign deployment config or ports differ")
    plan = ports.plan
    rubrics = ports.rubrics
    tiers = ports.provider_tiers
    provider_health = ports.provider_health
    source = ports.candidate_source
    prepared_runtime = ports.runtime
    runner = None if prepared_runtime is None else prepared_runtime.runner
    binding_resolver = ports.execution_binding_resolver
    turn_mcp_factory = ports.turn_mcp_factory
    actor_skills_factory = ports.actor_skills_factory
    opus5_adapter_gateway = ports.opus5_adapter_gateway
    if not isinstance(plan, CampaignPlan):
        raise CampaignDeploymentError("final CampaignPlan is required")
    try:
        verify_plan(plan)
    except Exception as exc:
        raise CampaignDeploymentError("final CampaignPlan verification failed") from exc
    if not isinstance(rubrics, CompiledRubricRegistry):
        raise CampaignDeploymentError("exact compiled rubric registry is required")
    if not isinstance(tiers, CodexProviderTiers):
        raise CampaignDeploymentError("provider tier targets are required")
    if provider_health is None:
        raise CampaignDeploymentError("verified provider route canaries are required")
    if source is None:
        raise CampaignDeploymentError("frozen campaign candidate source is required")
    if not isinstance(prepared_runtime, PreparedPersistentCodexRuntime) or not all(
        callable(getattr(runner, name, None)) for name in ("start", "run_once", "close")
    ):
        raise CampaignDeploymentError(
            "capacity-verified persistent Codex runner port is required"
        )
    startup = prepared_runtime.startup_metadata
    if (
        startup.get("app_server_shards") != config.concurrency.app_server_shards
        or startup.get("worker_width_configured") != config.concurrency.worker_width
        or startup.get("process_soft_nofile_required")
        != config.concurrency.required_process_soft_nofile
    ):
        raise CampaignDeploymentError("runtime startup capacity differs from campaign profile")
    if not callable(ports.actor_options_factory):
        raise CampaignDeploymentError("candidate-scoped actor MCP bridge transport is required")
    if not callable(ports.judge_options_factory):
        raise CampaignDeploymentError("read-only judge MCP transport is required")
    if binding_resolver is None:
        raise CampaignDeploymentError("candidate execution binding resolver is required")
    if turn_mcp_factory is None or not all(
        callable(getattr(turn_mcp_factory, name, None))
        for name in ("open_actor", "open_judge", "public_metadata")
    ):
        raise CampaignDeploymentError("authenticated actor/judge turn MCP bridge is required")
    required_turn_width = max(
        config.concurrency.maximum_parallel_tools,
        config.concurrency.maximum_parallel_judge_tools,
    )
    if getattr(turn_mcp_factory, "maximum", None) != required_turn_width:
        raise CampaignDeploymentError("turn MCP bridge width differs from campaign profile")
    turn_mcp_metadata = turn_mcp_factory.public_metadata()
    turn_mcp_blake3 = getattr(turn_mcp_factory, "launch_blake3", None)
    if not isinstance(turn_mcp_metadata, Mapping) or not isinstance(
        turn_mcp_blake3, str
    ):
        raise CampaignDeploymentError("turn MCP bridge launch isolation differs")
    _verify_turn_mcp_launch_metadata(
        turn_mcp_metadata,
        expected_blake3=turn_mcp_blake3,
        expected_width=required_turn_width,
    )
    if (
        actor_skills_factory is None
        or not callable(actor_skills_factory)
        or not callable(getattr(actor_skills_factory, "inventory", None))
        or not callable(getattr(actor_skills_factory, "for_stage", None))
        or not is_blake3(getattr(actor_skills_factory, "catalog_blake3", None))
        or not isinstance(actor_skills_factory.inventory(), tuple)
        or not actor_skills_factory.inventory()
    ):
        raise CampaignDeploymentError("verified actor skill mount catalog is required")
    (
        skill_materialization_root,
        skill_materialization_blake3,
        skill_materialization_path_policy,
        skill_materialization_count,
    ) = _verified_actor_skill_materialization(
        actor_skills_factory,
        workspace_root=config.workspace_root,
    )
    required_adapter_capacity = 4 * config.concurrency.worker_width
    if (
        opus5_adapter_gateway is None
        or not all(
            callable(getattr(opus5_adapter_gateway, name, None))
            for name in ("start", "close")
        )
        or type(getattr(opus5_adapter_gateway, "aggregate_max_concurrency", None))
        is not int
        or opus5_adapter_gateway.aggregate_max_concurrency
        < required_adapter_capacity
        or type(getattr(opus5_adapter_gateway, "shard_count", None)) is not int
        or opus5_adapter_gateway.shard_count < 1
        or not is_blake3(getattr(opus5_adapter_gateway, "binding_blake3", None))
    ):
        raise CampaignDeploymentError(
            "Opus 5 adapter capacity cannot cover four concurrent turns per worker"
        )
    if not callable(ports.id_factory_factory):
        raise CampaignDeploymentError("runtime identity factory port is required")
    _verify_rubrics(plan, rubrics)
    _verify_frozen_source(source, plan=plan, rubrics=rubrics)
    (
        executable_candidate_ids,
        execution_binding_catalog_blake3,
        executable_candidate_ids_blake3,
    ) = _verify_execution_binding_inventory(binding_resolver, source=source)
    prospective_catalog_blake3: str | None = None
    prospective_envelope_blake3: str | None = None
    prospective_ids_blake3: str | None = None
    prospective_count: int | None = None
    if ports.prospective_execution_catalog_path is not None:
        (
            executable_candidate_ids,
            prospective_catalog_blake3,
            prospective_envelope_blake3,
            prospective_ids_blake3,
        ) = _verify_prospective_execution_allowlist(
            ports.prospective_execution_catalog_path,
            trust_store_path=config.trust_store_path,
            source=source,
            legacy_catalog_blake3=execution_binding_catalog_blake3,
            legacy_candidate_ids=executable_candidate_ids,
        )
        prospective_count = len(executable_candidate_ids)
        executable_candidate_ids_blake3 = prospective_ids_blake3
    history_parent = config.receipt_root.parent
    receipt_roots = tuple(
        dict.fromkeys(
            (
                config.receipt_root,
                history_parent / "campaign-receipts",
                history_parent / "campaign-v2-receipts",
                history_parent / "campaign-v4-receipts",
            )
        )
    )
    ledger_paths = tuple(
        history_parent / name
        for name in ("campaign.sqlite3", "campaign-v2.sqlite3", "campaign-v4.sqlite3")
    )
    historical_evidence = signed_historical_attempt_evidence(
        receipt_roots=receipt_roots,
        ledger_paths=ledger_paths,
        private_key_path=config.private_key_path,
        key_id=config.signing_key_id,
    )
    attempted_candidate_ids = frozenset(
        historical_evidence["attempted_candidate_ids"]
    )
    claimable_candidate_ids = tuple(
        candidate_id
        for candidate_id in executable_candidate_ids
        if candidate_id not in attempted_candidate_ids
    )
    if not claimable_candidate_ids:
        raise CampaignDeploymentError("historical attempts exhaust executable coverage")
    claimable_candidate_ids_blake3 = blake3_hex(
        {
            "schema": "eva.historical-attempt-claim-allowlist.v1",
            "execution_allowlist_blake3": executable_candidate_ids_blake3,
            "historical_evidence_blake3": historical_evidence["evidence_blake3"],
            "candidate_ids": claimable_candidate_ids,
        }
    )
    if (
        config.concurrency.max_candidates is not None
        and config.concurrency.max_candidates > len(claimable_candidate_ids)
    ):
        raise CampaignDeploymentError(
            "max_candidates exceeds signed claimable execution coverage"
        )
    provider_health_blake3 = _verify_provider_health(provider_health, tiers=tiers)
    route_selector = DeterministicRouteSelector(
        candidate_ids=source.candidate_ids,
        selection_blake3=source.selection_blake3,
        tiers=tiers,
    )
    route_receipt = route_selector.receipt

    core = {
        "schema": CAMPAIGN_DEPLOYMENT_SCHEMA,
        "plan_blake3": plan.plan_blake3,
        "rubric_registry_blake3": rubrics.digest,
        "source_registry_blake3": source.source_registry_blake3,
        "selection_blake3": source.selection_blake3,
        "frozen_schedule_count": FROZEN_SCHEDULE_TOTAL,
        "primary_selection_count": PRIMARY_SELECTION_TOTAL,
        "reserve_selection_count": RESERVE_SELECTION_TOTAL,
        "actor_routes": tuple(
            (
                route.target.cohort.value,
                route.route_id,
                route.target.model_id,
                route.target.provider,
                route.config_blake3,
            )
            for route in tiers.actor_routes
        ),
        "judge_target": (
            tiers.judge.route_id,
            tiers.judge_model_id,
            tiers.judge_provider,
        ),
        "provider_health_blake3": provider_health_blake3,
        "route_assignment_receipt_blake3": route_receipt.receipt_blake3,
        "route_counts": route_receipt.counts_by_route,
        "execution_binding_catalog_blake3": execution_binding_catalog_blake3,
        "executable_candidate_ids_blake3": executable_candidate_ids_blake3,
        "executable_binding_count": len(executable_candidate_ids),
        "prospective_execution_catalog_envelope_blake3": prospective_envelope_blake3,
        "prospective_execution_catalog_blake3": prospective_catalog_blake3,
        "prospective_executable_candidate_ids_blake3": prospective_ids_blake3,
        "prospective_executable_count": prospective_count,
        "historical_attempt_evidence": historical_evidence,
        "claimable_candidate_ids_blake3": claimable_candidate_ids_blake3,
        "historical_exclusion_count": len(
            attempted_candidate_ids.intersection(executable_candidate_ids)
        ),
        "execution_binding_coverage": (
            "signed_v24_promoted_intersect_signed_prospective_v2"
            if prospective_catalog_blake3 is not None
            else "signed_v24_promoted_only"
        ),
        "skill_mount_catalog_blake3": actor_skills_factory.catalog_blake3,
        "skill_materialization_root": skill_materialization_root,
        "skill_materialization_blake3": skill_materialization_blake3,
        "skill_materialization_path_policy": skill_materialization_path_policy,
        "skill_materialization_count": skill_materialization_count,
        "turn_mcp_launch_blake3": turn_mcp_blake3,
        "turn_mcp_launch_metadata": turn_mcp_metadata,
        "opus5_adapter_binding_blake3": opus5_adapter_gateway.binding_blake3,
        "opus5_adapter_shards": opus5_adapter_gateway.shard_count,
        "opus5_adapter_aggregate_capacity": (
            opus5_adapter_gateway.aggregate_max_concurrency
        ),
        "opus5_adapter_required_capacity": required_adapter_capacity,
        "actor_runtime_surface": {
            "tools": "exact_candidate_policy_registry",
            "skills": "verified_stage_skill_mounts",
            "dynamic_skill_discovery_tools_exposed": False,
            "codex_sandbox": "read-only",
            "candidate_mutations": "exact_candidate_policy_mcp_only",
            "native_codex_surface": prepared_runtime.startup_metadata[
                "native_codex_surface"
            ],
            "project_doc_max_bytes": 0,
            "ancestor_project_docs_injected": False,
        },
        "concurrency": config.concurrency.to_document(),
        "separation_policy": _separation_policy_document(config.separation_policy),
        "runtime_startup_metadata": prepared_runtime.startup_metadata,
        "runtime_strategy": "sharded_persistent_app_servers_fresh_ephemeral_threads",
        "actor_tool_bridge": "candidate_scoped_exactly_once_mcp",
        "judge_tool_bridge": "read_only_workspace_replay_mcp",
        "build_provider_calls_made": 0,
        "semantic_retry_count": 0,
    }
    _verify_signer(config, challenge=canonical_json_bytes(core))
    recipe = CampaignDeploymentRecipe(
        plan_blake3=plan.plan_blake3,
        rubric_registry_blake3=rubrics.digest,
        source_registry_blake3=source.source_registry_blake3,
        selection_blake3=source.selection_blake3,
        actor_routes=core["actor_routes"],
        judge_target=core["judge_target"],
        provider_health_blake3=provider_health_blake3,
        route_assignment_receipt_blake3=route_receipt.receipt_blake3,
        route_counts=MappingProxyType(dict(route_receipt.counts_by_route)),
        execution_binding_catalog_blake3=execution_binding_catalog_blake3,
        executable_candidate_ids_blake3=executable_candidate_ids_blake3,
        executable_binding_count=len(executable_candidate_ids),
        prospective_execution_catalog_envelope_blake3=prospective_envelope_blake3,
        prospective_execution_catalog_blake3=prospective_catalog_blake3,
        prospective_executable_candidate_ids_blake3=prospective_ids_blake3,
        prospective_executable_count=prospective_count,
        historical_attempt_evidence=MappingProxyType(dict(historical_evidence)),
        claimable_candidate_ids_blake3=claimable_candidate_ids_blake3,
        historical_exclusion_count=len(
            attempted_candidate_ids.intersection(executable_candidate_ids)
        ),
        skill_mount_catalog_blake3=actor_skills_factory.catalog_blake3,
        skill_materialization_root=skill_materialization_root,
        skill_materialization_blake3=skill_materialization_blake3,
        skill_materialization_path_policy=skill_materialization_path_policy,
        skill_materialization_count=skill_materialization_count,
        turn_mcp_launch_blake3=turn_mcp_blake3,
        turn_mcp_launch_metadata=MappingProxyType(dict(turn_mcp_metadata)),
        opus5_adapter_binding_blake3=opus5_adapter_gateway.binding_blake3,
        opus5_adapter_shards=opus5_adapter_gateway.shard_count,
        opus5_adapter_aggregate_capacity=(
            opus5_adapter_gateway.aggregate_max_concurrency
        ),
        concurrency=MappingProxyType(dict(config.concurrency.to_document())),
        separation_policy=MappingProxyType(
            _separation_policy_document(config.separation_policy)
        ),
        runtime_startup_metadata=MappingProxyType(
            dict(prepared_runtime.startup_metadata)
        ),
        build_provider_calls_made=0,
        semantic_retry_count=0,
        recipe_blake3=blake3_hex(core),
    )
    _verify_recipe(recipe)

    artifact_store = ImmutableArtifactStore(config.artifact_root)
    ledger = CampaignLedger(config.ledger_path, plan=plan)
    exact_source = _ExactCandidateSource(
        source,
        plan=plan,
        rubrics=rubrics,
        route_selector=route_selector,
        binding_resolver=binding_resolver,
        executable_candidate_ids=executable_candidate_ids,
    )
    pipeline_verifier = PipelineVerifier(
        artifact_store,
        separation_policy=config.separation_policy,
    )

    def admission_context(candidate_id: str) -> SupervisorAdmissionContext:
        job = exact_source.load(candidate_id)
        return SupervisorAdmissionContext(
            candidate_id=candidate_id,
            split="train",
            rubric=job.rubric,
            plan_blake3=plan.plan_blake3,
        )

    supervisor = Ed25519AdmissionSupervisor(
        output_root=config.admission_bundle_root,
        pipeline_verifier=pipeline_verifier,
        context_resolver=admission_context,
        key_id=config.signing_key_id,
        private_key_path=config.private_key_path,
        trust_store_path=config.trust_store_path,
    )
    signature_verifier = Ed25519SupervisorSignatureVerifier(
        bundle_root=config.admission_bundle_root,
        pipeline_verifier=pipeline_verifier,
        context_resolver=admission_context,
        trust_store_path=config.trust_store_path,
    )

    actor_transport = ports.actor_options_factory
    judge_transport = ports.judge_options_factory
    assert actor_transport is not None and judge_transport is not None

    def checked_actor_options(
        request,
        cwd: str,
        offers: tuple[CodexToolOffer, ...],
        bridge: CodexToolExecutionBridge,
    ) -> CodexThreadOptions:
        if not isinstance(bridge, CodexToolExecutionBridge):
            raise CodexPipelineError("candidate-scoped actor tool bridge is missing")
        return _require_base_options(
            actor_transport(request, cwd, offers, bridge), label="actor"
        )

    def checked_judge_options(request, offers: tuple[CodexToolOffer, ...]) -> CodexThreadOptions:
        options = _require_base_options(
            judge_transport(request, offers), label="judge"
        )
        if options.provider != tiers.judge_provider:
            raise CodexPipelineError("judge provider route differs from deployment tier")
        return options

    def pipeline_factory(worker_id: str, job: CandidateJob) -> VerifiableDataPipeline:
        if not isinstance(worker_id, str) or not worker_id:
            raise CampaignDeploymentError("pipeline worker identity differs")
        if not isinstance(job, CandidateJob):
            raise CampaignDeploymentError("candidate pipeline job differs")
        binding = exact_source.binding_for(job)
        ids = ports.id_factory_factory(f"{worker_id}:{job.candidate_id}")
        if not callable(getattr(ids, "new", None)):
            raise CampaignDeploymentError("runtime identity factory result differs")
        rollout = CodexRolloutAdapter(
            options_factory=checked_actor_options,
            runner=runner,
            id_factory=ids,
            turn_mcp_factory=turn_mcp_factory,
            skills_factory=actor_skills_factory,
            enable_native_actions=job.episode.stage in ports.native_actor_stages,
            tool_runtime_guard=ports.actor_tool_runtime_guard,
        )
        judge = CodexOpus5AgentJudge(
            options_factory=checked_judge_options,
            model_id=tiers.judge_model_id,
            runner=runner,
            id_factory=ids,
            maximum_parallel_tools=config.concurrency.maximum_parallel_judge_tools,
            turn_mcp_factory=turn_mcp_factory,
        )
        return VerifiableDataPipeline(
            workspace_root=config.workspace_root,
            artifact_store=artifact_store,
            tool_registry=binding.tool_registry,
            rollout_provider=rollout,
            judge=judge,
            rewarder=WeightedRubricRewarder(),
            id_factory=ids,
            judge_model_id=tiers.judge_model_id,
            separation_policy=config.separation_policy,
            maximum_parallel_models=config.concurrency.maximum_parallel_models,
            maximum_parallel_tools=config.concurrency.maximum_parallel_tools,
        )

    orchestrator = CampaignOrchestrator(
        ledger=ledger,
        candidate_source=exact_source,
        candidate_pipeline_factory=pipeline_factory,
        eligible_candidate_ids=claimable_candidate_ids,
        receipt_root=config.receipt_root,
        config=config.concurrency.orchestration_config(),
        supervisor_evidence_source=supervisor,
        supervisor_signature_verifier=signature_verifier,
        progress_callback=ports.progress_callback,
    )
    return CodexCampaignDeployment(
        recipe=recipe,
        plan=plan,
        rubrics=rubrics,
        provider_tiers=tiers,
        route_selector=route_selector,
        route_assignment_receipt=route_receipt,
        frozen_source=source,
        candidate_source=exact_source,
        execution_binding_resolver=binding_resolver,
        opus5_adapter_gateway=opus5_adapter_gateway,
        executable_candidate_ids=executable_candidate_ids,
        prospective_execution_catalog_path=ports.prospective_execution_catalog_path,
        prepared_runtime=prepared_runtime,
        runtime_runner=runner,
        ledger=ledger,
        artifact_store=artifact_store,
        pipeline_verifier=pipeline_verifier,
        supervisor=supervisor,
        signature_verifier=signature_verifier,
        pipeline_factory=pipeline_factory,
        orchestrator=orchestrator,
    )


def build_one_candidate_canary_deployment(
    *,
    config: CampaignDeploymentConfig,
    ports: CampaignDeploymentPorts,
) -> CodexCampaignDeployment:
    """Build the real one-candidate entry point without making its provider calls.

    Calling ``run_until_idle()`` on the returned deployment claims exactly one
    frozen candidate.  Inside that candidate the weak, middle, and strong
    actors still run concurrently, once each, followed by their workspace
    Agent Judge passes.  The same signer, MCP bridges, route assignment,
    artifact store, and ledger used in production remain active.
    """

    if not isinstance(config, CampaignDeploymentConfig):
        raise CampaignDeploymentError("canary campaign config differs")
    prepared = ports.runtime
    if (
        not isinstance(prepared, PreparedPersistentCodexRuntime)
        or getattr(prepared.runner, "shard_count", None) != 1
    ):
        raise CampaignDeploymentError("canary requires exactly one app-server shard")
    canary_concurrency = replace(
        config.concurrency,
        worker_width=1,
        queue_capacity=1,
        claim_batch_size=1,
        app_server_shards=1,
        max_candidates=1,
    )
    startup_core = {
        key: value
        for key, value in prepared.startup_metadata.items()
        if key != "launch_blake3"
    }
    startup_core["worker_width_configured"] = 1
    canary_runtime = PreparedPersistentCodexRuntime(
        runner=prepared.runner,
        launch_options=prepared.launch_options,
        startup_metadata=MappingProxyType(
            {**startup_core, "launch_blake3": blake3_hex(startup_core)}
        ),
    )
    return build_codex_campaign_deployment(
        config=replace(config, concurrency=canary_concurrency),
        ports=replace(ports, runtime=canary_runtime),
    )


__all__ = [
    "CAMPAIGN_DEPLOYMENT_SCHEMA",
    "CODEX_CHILD_STARTUP_SCHEMA",
    "CODEX_FIRST_RELEASE_CONFIG_OVERRIDES",
    "CODEX_FIRST_RELEASE_NATIVE_SURFACE",
    "CODEX_FIRST_RELEASE_THREAD_CONFIG",
    "DEFAULT_CHILD_SOFT_NOFILE",
    "EXPECTED_EXECUTABLE_BINDINGS",
    "FROZEN_SCHEDULE_TOTAL",
    "PRIMARY_SELECTION_TOTAL",
    "RESERVE_SELECTION_TOTAL",
    "ROUTE_ASSIGNMENT_SCHEMA",
    "CandidateRouteAssignment",
    "CandidateExecutionBindingPort",
    "ExecutionBindingResolverPort",
    "ActorSkillCatalogPort",
    "Opus5AdapterGatewayPort",
    "CampaignConcurrency",
    "CampaignDeploymentConfig",
    "CampaignDeploymentError",
    "CampaignDeploymentPorts",
    "CampaignDeploymentRecipe",
    "CodexCampaignDeployment",
    "CodexProviderTiers",
    "DeterministicRouteSelector",
    "FrozenCampaignSourcePort",
    "FrozenSelectionCandidateSource",
    "PreparedPersistentCodexRuntime",
    "PersistentCodexRunnerPort",
    "ProviderRouteHealth",
    "ProviderRouteTarget",
    "RouteAssignmentReceipt",
    "build_codex_campaign_deployment",
    "build_one_candidate_canary_deployment",
    "prepare_persistent_codex_runtime",
]
