"""Immutable contracts shared by pipeline compilers and harness adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
from uuid import UUID

from .digests import blake3_hex, is_blake3


JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | Mapping[str, "JsonValue"] | tuple["JsonValue", ...]


class ContractError(ValueError):
    """A pipeline boundary failed closed."""


def freeze_json(value: Any) -> JsonValue:
    """Recursively copy JSON-shaped input into immutable containers."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ContractError("JSON mapping keys must be strings")
        return MappingProxyType({key: freeze_json(value[key]) for key in sorted(value)})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    raise ContractError(f"value is not JSON-shaped: {type(value).__name__}")


def uuid_text(value: str, *, label: str) -> str:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError):
        raise ContractError(f"{label} must be a UUID") from None
    if str(parsed) != value:
        raise ContractError(f"{label} must use canonical UUID text")
    return value


class Stage(str, Enum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    S5 = "S5"
    E2E = "E2E"


class Cohort(str, Enum):
    WEAK = "weak"
    MIDDLE = "middle"
    STRONG = "strong"


COHORT_ORDER = (Cohort.WEAK, Cohort.MIDDLE, Cohort.STRONG)


@dataclass(frozen=True)
class BenchmarkSource:
    benchmark: str
    source_file: str
    source_revision: str

    def __post_init__(self) -> None:
        if not all((self.benchmark, self.source_file, self.source_revision)):
            raise ContractError("benchmark source metadata must be complete")


@dataclass(frozen=True)
class BenchmarkEpisode:
    episode_id: str
    source: BenchmarkSource
    domain: str
    stage: Stage
    instruction: str
    policy_context: Mapping[str, JsonValue]
    initial_files: Mapping[str, bytes]
    judge_only_reference: JsonValue | None = None

    def __post_init__(self) -> None:
        if not self.episode_id or not self.domain or not self.instruction:
            raise ContractError("episode identity, domain, and instruction are required")
        if not isinstance(self.stage, Stage):
            raise ContractError("episode stage must be S1-S5 or E2E")
        context = freeze_json(self.policy_context)
        if not isinstance(context, Mapping):
            raise ContractError("policy context must be an object")
        files: dict[str, bytes] = {}
        for path, payload in self.initial_files.items():
            if not isinstance(path, str) or not isinstance(payload, bytes):
                raise ContractError("initial workspace files require string paths and bytes")
            files[path] = bytes(payload)
        object.__setattr__(self, "policy_context", context)
        object.__setattr__(self, "initial_files", MappingProxyType(dict(sorted(files.items()))))
        object.__setattr__(self, "judge_only_reference", freeze_json(self.judge_only_reference))


@runtime_checkable
class CompiledRubricTable(Protocol):
    """Defensive port for the separately owned versioned rubric registry."""

    rubric_id: str
    version: int | str
    digest: str
    domain: str
    stage: str
    items: Sequence[Mapping[str, Any]]

    def score(
        self,
        item_scores_bps: Mapping[str, int],
        *,
        evaluation_id: str | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class RubricBinding:
    rubric_id: str
    version: int | str
    digest: str
    domain: str
    stage: Stage

    @classmethod
    def from_compiled(cls, rubric: CompiledRubricTable) -> "RubricBinding":
        stage_value = rubric.stage.value if isinstance(rubric.stage, Stage) else rubric.stage
        try:
            stage = Stage(stage_value)
        except (ValueError, TypeError):
            raise ContractError("compiled rubric stage differs") from None
        value = cls(
            rubric_id=rubric.rubric_id,
            version=rubric.version,
            digest=rubric.digest,
            domain=rubric.domain,
            stage=stage,
        )
        if (
            not value.rubric_id
            or type(value.version) not in {int, str}
            or not value.version
            or not is_blake3(value.digest)
        ):
            raise ContractError("compiled rubric identity/version/BLAKE3 digest differs")
        return value


@dataclass(frozen=True)
class SandboxManifest:
    sandbox_id: str
    episode_id: str
    source: BenchmarkSource
    domain: str
    stage: Stage
    instruction: str
    policy_context: Mapping[str, JsonValue]
    initial_files: Mapping[str, bytes]
    rubric: RubricBinding
    manifest_blake3: str

    @classmethod
    def create(
        cls,
        *,
        sandbox_id: str,
        episode: BenchmarkEpisode,
        rubric: CompiledRubricTable,
    ) -> "SandboxManifest":
        uuid_text(sandbox_id, label="sandbox_id")
        binding = RubricBinding.from_compiled(rubric)
        if binding.domain != episode.domain or binding.stage is not episode.stage:
            raise ContractError("sandbox must bind exactly its domain x stage rubric table")
        core = {
            "sandbox_id": sandbox_id,
            "episode_id": episode.episode_id,
            "source": episode.source,
            "domain": episode.domain,
            "stage": episode.stage,
            "instruction": episode.instruction,
            "policy_context": episode.policy_context,
            "initial_files": episode.initial_files,
            "rubric": binding,
            "rubric_table_count": 1,
        }
        return cls(
            sandbox_id=sandbox_id,
            episode_id=episode.episode_id,
            source=episode.source,
            domain=episode.domain,
            stage=episode.stage,
            instruction=episode.instruction,
            policy_context=episode.policy_context,
            initial_files=episode.initial_files,
            rubric=binding,
            manifest_blake3=blake3_hex(core),
        )


@dataclass(frozen=True)
class ModelTarget:
    cohort: Cohort
    model_id: str
    provider: str

    def __post_init__(self) -> None:
        if not isinstance(self.cohort, Cohort) or not self.model_id or not self.provider:
            raise ContractError("model target is incomplete")


@dataclass(frozen=True)
class FileSnapshot:
    path: str
    content: bytes
    byte_count: int
    mode: str
    content_blake3: str


@dataclass(frozen=True)
class WorkspaceSnapshot:
    label: str
    files: tuple[FileSnapshot, ...]
    file_count: int
    byte_count: int
    tree_blake3: str


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, JsonValue]
    depends_on: tuple[str, ...] = ()
    resource_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        uuid_text(self.call_id, label="tool call_id")
        if not self.name:
            raise ContractError("tool call name is required")
        arguments = freeze_json(self.arguments)
        if not isinstance(arguments, Mapping):
            raise ContractError("tool arguments must be an object")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ContractError("tool dependencies must be unique")
        for dependency in self.depends_on:
            uuid_text(dependency, label="tool dependency")
        if len(set(self.resource_keys)) != len(self.resource_keys):
            raise ContractError("tool resources must be unique")
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        object.__setattr__(self, "resource_keys", tuple(sorted(self.resource_keys)))


@dataclass(frozen=True)
class ToolResult:
    result_id: str
    call_id: str
    name: str
    frontier: int
    parallel_group_id: str
    status: str
    output: JsonValue | None
    error_code: str | None
    workspace_before_blake3: str
    workspace_after_blake3: str
    receipt_blake3: str


@dataclass(frozen=True)
class ToolTrace:
    results: tuple[ToolResult, ...]
    declared_call_ids: tuple[str, ...]
    joined_call_ids: tuple[str, ...]
    frontier_count: int
    max_parallelism_observed: int
    retry_count: int
    trace_blake3: str


@dataclass(frozen=True)
class RolloutRequest:
    rollout_id: str
    sandbox: SandboxManifest
    model: ModelTarget
    policy_visible_context: Mapping[str, JsonValue]
    available_tools: tuple[Mapping[str, JsonValue], ...]


@dataclass(frozen=True)
class TrajectoryEvent:
    """One policy-visible event; hidden reasoning has no representation here."""

    event_id: str
    role: str
    content: JsonValue
    tool_call_ids: tuple[str, ...]
    event_blake3: str

    def __post_init__(self) -> None:
        uuid_text(self.event_id, label="trajectory event_id")
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ContractError("trajectory event role differs")
        frozen = freeze_json(self.content)
        for call_id in self.tool_call_ids:
            uuid_text(call_id, label="trajectory tool call_id")
        if len(set(self.tool_call_ids)) != len(self.tool_call_ids):
            raise ContractError("trajectory tool-call group contains duplicates")
        if self.role != "assistant" and len(self.tool_call_ids) > 1:
            raise ContractError("only one assistant decision may own a multi-call group")
        object.__setattr__(self, "content", frozen)
        object.__setattr__(self, "tool_call_ids", tuple(self.tool_call_ids))
        core = {
            "event_id": self.event_id,
            "role": self.role,
            "content": self.content,
            "tool_call_ids": self.tool_call_ids,
        }
        if self.event_blake3 != blake3_hex(core):
            raise ContractError("trajectory event BLAKE3 differs")


@dataclass(frozen=True)
class ProviderRollout:
    assistant_output: str
    provider_receipt_blake3: str
    policy_events: tuple[TrajectoryEvent, ...]
    safe_metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.assistant_output, str) or not is_blake3(self.provider_receipt_blake3):
            raise ContractError("provider rollout output or receipt differs")
        if not self.policy_events or not any(event.role == "assistant" for event in self.policy_events):
            raise ContractError("provider rollout must retain a policy-visible assistant decision")
        object.__setattr__(self, "policy_events", tuple(self.policy_events))
        metadata = freeze_json(self.safe_metadata)
        if not isinstance(metadata, Mapping):
            raise ContractError("provider safe metadata must be an object")
        object.__setattr__(self, "safe_metadata", metadata)


@dataclass(frozen=True)
class EvidenceBundle:
    bundle_id: str
    rollout_id: str
    sandbox_manifest: SandboxManifest
    model: ModelTarget
    policy_visible_context: Mapping[str, JsonValue]
    context_blake3: str
    workspace_before: WorkspaceSnapshot
    workspace_after: WorkspaceSnapshot
    tool_trace: ToolTrace
    policy_events: tuple[TrajectoryEvent, ...]
    assistant_output: str
    provider_receipt_blake3: str
    safe_provider_metadata: Mapping[str, JsonValue]
    bundle_blake3: str


@dataclass(frozen=True)
class JudgeRequest:
    judgment_id: str
    judge_model_id: str
    policy_visible_context: Mapping[str, JsonValue]
    workspace_evidence: EvidenceBundle
    judge_only_reference: JsonValue | None


@dataclass(frozen=True)
class RubricItemScore:
    item_id: str
    score: float
    evidence_refs: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not self.item_id or not 0.0 <= float(self.score) <= 1.0:
            raise ContractError("rubric item score differs")
        if not self.evidence_refs or any(
            not isinstance(reference, str) or not reference
            for reference in self.evidence_refs
        ):
            raise ContractError("rubric item evidence references differ")
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ContractError("rubric item evidence references contain duplicates")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ContractError("rubric item rationale differs")
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))


@dataclass(frozen=True)
class JudgeToolResult:
    """One immutable observation made by the read-only evidence judge."""

    result_id: str
    call_id: str
    name: str
    arguments: Mapping[str, JsonValue]
    frontier: int
    parallel_group_id: str
    status: str
    output: JsonValue | None
    error_code: str | None
    inspected_evidence_refs: tuple[str, ...]
    content_inspection: bool
    evidence_bundle_blake3: str
    receipt_blake3: str

    def __post_init__(self) -> None:
        uuid_text(self.result_id, label="judge tool result_id")
        uuid_text(self.call_id, label="judge tool call_id")
        uuid_text(self.parallel_group_id, label="judge tool parallel_group_id")
        if self.name not in {
            "workspace_list",
            "workspace_read",
            "workspace_search",
            "workspace_diff",
        }:
            raise ContractError("judge tool name differs")
        arguments = freeze_json(self.arguments)
        if not isinstance(arguments, Mapping):
            raise ContractError("judge tool arguments must be an object")
        if type(self.frontier) is not int or self.frontier < 0:
            raise ContractError("judge tool frontier differs")
        if self.status not in {"completed", "immutable_failure"}:
            raise ContractError("judge tool status differs")
        output = freeze_json(self.output)
        if self.status == "completed" and self.error_code is not None:
            raise ContractError("completed judge tool cannot retain an error")
        if self.status == "immutable_failure" and (
            not isinstance(self.error_code, str) or not self.error_code
        ):
            raise ContractError("failed judge tool must retain an error code")
        if any(
            not isinstance(reference, str) or not reference
            for reference in self.inspected_evidence_refs
        ):
            raise ContractError("judge tool inspected evidence references differ")
        if len(set(self.inspected_evidence_refs)) != len(self.inspected_evidence_refs):
            raise ContractError("judge tool inspected evidence references contain duplicates")
        if type(self.content_inspection) is not bool:
            raise ContractError("judge tool content-inspection marker differs")
        if self.content_inspection and not self.inspected_evidence_refs:
            raise ContractError("content inspection must bind at least one evidence reference")
        if not is_blake3(self.evidence_bundle_blake3):
            raise ContractError("judge tool evidence-bundle commitment differs")
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "output", output)
        object.__setattr__(
            self,
            "inspected_evidence_refs",
            tuple(self.inspected_evidence_refs),
        )
        core = {
            "result_id": self.result_id,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
            "frontier": self.frontier,
            "parallel_group_id": self.parallel_group_id,
            "status": self.status,
            "output": self.output,
            "error_code": self.error_code,
            "inspected_evidence_refs": self.inspected_evidence_refs,
            "content_inspection": self.content_inspection,
            "evidence_bundle_blake3": self.evidence_bundle_blake3,
        }
        if self.receipt_blake3 != blake3_hex(core):
            raise ContractError("judge tool result BLAKE3 differs")


@dataclass(frozen=True)
class JudgeAgentTrace:
    """Reopenable tool-using judge trajectory with no hidden reasoning."""

    policy_events: tuple[TrajectoryEvent, ...]
    results: tuple[JudgeToolResult, ...]
    declared_call_ids: tuple[str, ...]
    joined_call_ids: tuple[str, ...]
    inspected_evidence_refs: tuple[str, ...]
    content_inspection_count: int
    frontier_count: int
    max_parallelism_observed: int
    provider_turn_count: int
    retry_count: int
    evidence_bundle_blake3: str
    trace_blake3: str

    def __post_init__(self) -> None:
        events = tuple(self.policy_events)
        results = tuple(self.results)
        declared = tuple(self.declared_call_ids)
        joined = tuple(self.joined_call_ids)
        inspected = tuple(self.inspected_evidence_refs)
        if len(events) < 4 or events[0].role != "system" or events[1].role != "user":
            raise ContractError("judge agent trajectory framing differs")
        for call_id in declared + joined:
            uuid_text(call_id, label="judge trace tool call_id")
        if len(set(declared)) != len(declared) or len(set(joined)) != len(joined):
            raise ContractError("judge trace tool call identities differ")
        if joined != tuple(result.call_id for result in results) or set(declared) != set(joined):
            raise ContractError("judge trace declared/joined calls differ")
        expected_inspected = tuple(
            sorted(
                {
                    reference
                    for result in results
                    if result.status == "completed"
                    for reference in result.inspected_evidence_refs
                }
            )
        )
        if inspected != expected_inspected:
            raise ContractError("judge trace inspected evidence inventory differs")
        expected_content_count = sum(
            result.status == "completed" and result.content_inspection
            for result in results
        )
        if (
            type(self.content_inspection_count) is not int
            or self.content_inspection_count != expected_content_count
            or self.content_inspection_count < 0
        ):
            raise ContractError("judge trace content-inspection count differs")
        if (
            type(self.frontier_count) is not int
            or self.frontier_count < 1
            or type(self.max_parallelism_observed) is not int
            or not 1 <= self.max_parallelism_observed <= max(1, len(results))
            or type(self.provider_turn_count) is not int
            or self.provider_turn_count < 2
            or self.retry_count != 0
        ):
            raise ContractError("judge trace execution bounds differ")
        if not is_blake3(self.evidence_bundle_blake3) or any(
            result.evidence_bundle_blake3 != self.evidence_bundle_blake3
            for result in results
        ):
            raise ContractError("judge trace evidence-bundle binding differs")
        object.__setattr__(self, "policy_events", events)
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "declared_call_ids", declared)
        object.__setattr__(self, "joined_call_ids", joined)
        object.__setattr__(self, "inspected_evidence_refs", inspected)
        core = {
            "policy_events": events,
            "results": results,
            "declared_call_ids": declared,
            "joined_call_ids": joined,
            "inspected_evidence_refs": inspected,
            "content_inspection_count": self.content_inspection_count,
            "frontier_count": self.frontier_count,
            "max_parallelism_observed": self.max_parallelism_observed,
            "provider_turn_count": self.provider_turn_count,
            "retry_count": self.retry_count,
            "evidence_bundle_blake3": self.evidence_bundle_blake3,
        }
        if self.trace_blake3 != blake3_hex(core):
            raise ContractError("judge agent trace BLAKE3 differs")


@dataclass(frozen=True)
class JudgeAssessment:
    judgment_id: str
    judge_model_id: str
    rubric_digest: str
    agent_trace: JudgeAgentTrace
    item_scores: tuple[RubricItemScore, ...]
    hard_gates_passed: bool
    summary: str
    assessment_blake3: str


@dataclass(frozen=True)
class RewardRecord:
    reward_id: str
    rubric_digest: str
    item_scores: tuple[RubricItemScore, ...]
    total_reward: float
    hard_gates_passed: bool
    reward_blake3: str


@dataclass(frozen=True)
class ArtifactRef:
    relative_path: str
    byte_count: int
    mode: str
    content_blake3: str


@dataclass(frozen=True)
class ModelEvaluation:
    evaluation_id: str
    cohort: Cohort
    model: ModelTarget
    evidence: EvidenceBundle
    judgment: JudgeAssessment
    reward: RewardRecord
    manifest_artifact: ArtifactRef
    evidence_artifact: ArtifactRef
    evaluation_blake3: str


@dataclass(frozen=True)
class AbilitySeparationReport:
    rewards_by_cohort: Mapping[str, float]
    raw_item_scores_by_cohort: Mapping[str, Mapping[str, float]]
    strong_minus_weak: float | None
    strong_minus_middle: float | None
    middle_minus_weak: float | None
    perfect_monotonic_staircase_required: bool
    policy_passed: bool
    report_blake3: str
    # V2 keeps the cascade as diagnostic evidence while allowing one fully
    # verified actor+workspace-judge trajectory to qualify the sandbox.  None
    # identifies legacy v1 reports whose admission gate was separation itself.
    admission_policy_schema: str | None = None
    ability_separation_required_for_admission: bool | None = None
    minimum_valid_judged_trajectories: int | None = None
    valid_judged_trajectory_count: int | None = None
    qualifying_evaluation_blake3s: tuple[str, ...] = ()
    selected_evaluation_blake3: str | None = None
    selected_evaluation_cohort: str | None = None
    cohort_outcomes: Mapping[str, str] | None = None
    failure_types_by_cohort: Mapping[str, str] | None = None
    admission_policy_passed: bool | None = None


@dataclass(frozen=True)
class AdmissionRecommendation:
    recommendation: str
    reasons: tuple[str, ...]
    admission_authorized: bool
    signed_decision_created: bool
    recommendation_blake3: str


@dataclass(frozen=True)
class OutcomeTelemetry:
    cohort: Cohort
    model_id: str
    rubric_digest: str
    reward: float
    item_scores: Mapping[str, float]
    tool_call_count: int
    tool_failure_count: int
    parallelism_observed: int
    workspace_changed: bool
    hard_gates_passed: bool
    outcome: str
    telemetry_blake3: str


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    sandbox_manifests: tuple[SandboxManifest, ...]
    evaluations: tuple[ModelEvaluation, ...]
    separation: AbilitySeparationReport
    recommendation: AdmissionRecommendation
    telemetry: tuple[OutcomeTelemetry, ...]
    result_blake3: str


@dataclass(frozen=True)
class AdmissionEvidence:
    pipeline_result_blake3: str
    evaluation_blake3: str
    signed_admission_receipt_blake3: str
    signed_supervisor_transition_blake3: str

    def __post_init__(self) -> None:
        for value in (
            self.pipeline_result_blake3,
            self.evaluation_blake3,
            self.signed_admission_receipt_blake3,
            self.signed_supervisor_transition_blake3,
        ):
            if not is_blake3(value):
                raise ContractError("admission evidence BLAKE3 differs")


@dataclass(frozen=True)
class SFTExample:
    example_id: str
    sandbox_id: str
    source_rollout_id: str
    source_bundle_blake3: str
    source_evaluation_blake3: str
    source_pipeline_result_blake3: str
    rubric: RubricBinding
    prefix: tuple[TrajectoryEvent, ...]
    supervised_assistant_decision: TrajectoryEvent
    atomic_tool_call_group: bool
    signed_admission_receipt_blake3: str
    signed_supervisor_transition_blake3: str
    private_reference_included: bool
    hidden_reasoning_included: bool
    lineage_blake3: str


class AdmissionEvidenceVerifier(Protocol):
    def verify(
        self,
        *,
        result: PipelineResult,
        evaluation: ModelEvaluation,
        evidence: AdmissionEvidence,
    ) -> bool: ...


class RuntimeIdFactory(Protocol):
    def new(self, purpose: str) -> str: ...


class ToolRuntimePort(Protocol):
    def execute(self, calls: Sequence[ToolCall]) -> tuple[ToolResult, ...]: ...

    def trace(self) -> ToolTrace: ...


class RolloutProvider(Protocol):
    def run(self, request: RolloutRequest, tools: ToolRuntimePort) -> ProviderRollout: ...


class AgentJudge(Protocol):
    def judge(
        self,
        request: JudgeRequest,
        rubric: CompiledRubricTable,
    ) -> JudgeAssessment: ...


class RewardComputer(Protocol):
    def compute(
        self,
        *,
        reward_id: str,
        rubric: CompiledRubricTable,
        assessment: JudgeAssessment,
        evidence: EvidenceBundle,
    ) -> RewardRecord: ...


__all__ = [name for name in globals() if not name.startswith("_")]
