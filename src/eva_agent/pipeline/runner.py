"""Executable benchmark-to-evaluation pipeline.

One benchmark episode becomes one rubric-bound logical sandbox.  Exactly one
fresh trajectory is collected for each ability cohort.  The three physical
workspaces are byte-identical resets of that sandbox and may run concurrently.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import re
from threading import Event, RLock
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .artifacts import ImmutableArtifactStore
from .contracts import (
    COHORT_ORDER,
    AdmissionRecommendation,
    AgentJudge,
    AbilitySeparationReport,
    BenchmarkEpisode,
    Cohort,
    CompiledRubricTable,
    ContractError,
    EvidenceBundle,
    JudgeRequest,
    ModelEvaluation,
    ModelTarget,
    OutcomeTelemetry,
    PipelineResult,
    RewardComputer,
    RolloutProvider,
    RolloutRequest,
    RuntimeIdFactory,
    SandboxManifest,
    freeze_json,
    uuid_text,
)
from .digests import blake3_hex, canonical_value
from .tools import MAXIMUM_PARALLEL_TOOL_CALLS, ParallelToolRuntime, ToolRegistry
from .workspace import FilesystemSandbox


class PipelineError(ContractError):
    """The evaluation pipeline failed a deterministic contract."""


class InfrastructureQuarantineError(PipelineError):
    """A provider boundary failed; the attempt is quarantined, not scored zero."""

    def __init__(self, run_id: str, failures: Mapping[str, str]) -> None:
        self.run_id = run_id
        self.failures = MappingProxyType(dict(sorted(failures.items())))
        super().__init__(f"pipeline run {run_id} was infrastructure-quarantined")


class _EarlyCohortCancellation(PipelineError):
    """A non-selected cohort stopped after one evaluation verified in full."""


_SAFE_FAILURE_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")


def _safe_failure_detail(exc: Exception) -> dict[str, str] | None:
    category = getattr(exc, "safe_failure_category", None)
    message = getattr(exc, "safe_failure_message", None)
    provenance = getattr(exc, "safe_failure_provenance", None)
    if (
        not isinstance(category, str)
        or _SAFE_FAILURE_TOKEN.fullmatch(category) is None
        or not isinstance(provenance, str)
        or _SAFE_FAILURE_TOKEN.fullmatch(provenance) is None
        or not isinstance(message, str)
        or not message
        or len(message) > 160
        or any(ord(character) < 32 or ord(character) > 126 for character in message)
        or any(marker in message for marker in ("/", "\\", "=", "@", ":"))
    ):
        return None
    return {"category": category, "message": message, "provenance": provenance}


@dataclass(frozen=True)
class SeparationPolicy:
    """Versioned separation telemetry and trajectory admission policy.

    V2 still schedules the fixed three-cohort cascade and retains every score,
    but admission no longer depends on a complete or monotonic cascade.  One
    independently valid actor trajectory, workspace-aware Opus 5 judgment,
    and exact compiled-rubric reward is sufficient.
    """

    minimum_strong_reward: float = 0.70
    minimum_strong_minus_weak: float = 0.10
    material_item_delta: float = 0.25
    require_strong_hard_gates: bool = True
    admission_policy_schema: str = "eva.trajectory-admission-policy.v2"
    minimum_valid_judged_trajectories: int = 1
    ability_separation_required_for_admission: bool = False
    early_continuation_after_first_valid_trajectory: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_strong_reward", self.minimum_strong_reward),
            ("minimum_strong_minus_weak", self.minimum_strong_minus_weak),
            ("material_item_delta", self.material_item_delta),
        ):
            if type(value) not in {int, float} or not 0.0 <= float(value) <= 1.0:
                raise PipelineError(f"{name} must be in [0,1]")
        if self.admission_policy_schema != "eva.trajectory-admission-policy.v2":
            raise PipelineError("admission policy schema differs")
        if self.minimum_valid_judged_trajectories != 1:
            raise PipelineError("v2 admission requires exactly one valid judged trajectory")
        if type(self.ability_separation_required_for_admission) is not bool:
            raise PipelineError("ability separation admission flag differs")
        if type(self.early_continuation_after_first_valid_trajectory) is not bool:
            raise PipelineError("early trajectory continuation flag differs")
        if (
            self.early_continuation_after_first_valid_trajectory
            and self.ability_separation_required_for_admission
        ):
            raise PipelineError(
                "early trajectory continuation cannot require complete ability separation"
            )


def _cohort_outcomes(
    evaluations: Sequence[ModelEvaluation], failures: Mapping[str, str]
) -> dict[str, str]:
    completed = {row.cohort for row in evaluations}
    return {
        cohort.value: (
            "completed"
            if cohort in completed
            else (
                "cancelled_after_valid_evaluation"
                if failures.get(cohort.value) == _EarlyCohortCancellation.__name__
                else "infrastructure_failure"
            )
        )
        for cohort in COHORT_ORDER
    }


def partial_cohort_outcome_document(
    *,
    run_id: str,
    manifest_blake3: str,
    evaluations: Sequence[ModelEvaluation],
    failures: Mapping[str, str],
    early_continuation_enabled: bool,
) -> dict[str, Any]:
    """Canonical retained evidence for every cohort absent from a v2 result."""

    outcomes = _cohort_outcomes(evaluations, failures)
    core = {
        "schema": "eva.pipeline-partial-cohort-outcomes.v2",
        "run_id": run_id,
        "sandbox_manifest_blake3": manifest_blake3,
        "failures": dict(sorted(failures.items())),
        "cohort_outcomes": outcomes,
        "completed_cohorts": tuple(
            cohort.value
            for cohort in COHORT_ORDER
            if outcomes[cohort.value] == "completed"
        ),
        "cancelled_cohorts": tuple(
            cohort.value
            for cohort in COHORT_ORDER
            if outcomes[cohort.value] == "cancelled_after_valid_evaluation"
        ),
        "early_continuation_after_first_valid_trajectory": (
            early_continuation_enabled
        ),
        "absent_cohort_semantic_scores_created": False,
        "retry_count": 0,
    }
    return {**core, "failure_evidence_blake3": blake3_hex(core)}


def _forbidden_policy_key(value: Any) -> str | None:
    forbidden = {
        "answer_key",
        "chain_of_thought",
        "gold",
        "gold_answer",
        "hidden_reasoning",
        "judge_only_reference",
        "private_reference",
        "reference_answer",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold().replace("-", "_").replace(" ", "_")
            if normalized in forbidden:
                return str(key)
            found = _forbidden_policy_key(child)
            if found is not None:
                return found
    elif isinstance(value, (tuple, list)):
        for child in value:
            found = _forbidden_policy_key(child)
            if found is not None:
                return found
    return None


def evidence_core(evidence: EvidenceBundle) -> dict[str, Any]:
    return {
        "bundle_id": evidence.bundle_id,
        "rollout_id": evidence.rollout_id,
        "sandbox_manifest": evidence.sandbox_manifest,
        "model": evidence.model,
        "policy_visible_context": evidence.policy_visible_context,
        "context_blake3": evidence.context_blake3,
        "workspace_before": evidence.workspace_before,
        "workspace_after": evidence.workspace_after,
        "tool_trace": evidence.tool_trace,
        "policy_events": evidence.policy_events,
        "assistant_output": evidence.assistant_output,
        "provider_receipt_blake3": evidence.provider_receipt_blake3,
        "safe_provider_metadata": evidence.safe_provider_metadata,
    }


def evaluation_core(evaluation: ModelEvaluation) -> dict[str, Any]:
    """Non-circular evaluation commitment used by manifest and verifier."""

    return {
        "evaluation_id": evaluation.evaluation_id,
        "cohort": evaluation.cohort,
        "model": evaluation.model,
        "evidence_blake3": evaluation.evidence.bundle_blake3,
        "judgment_blake3": evaluation.judgment.assessment_blake3,
        "reward_blake3": evaluation.reward.reward_blake3,
        "evidence_artifact": evaluation.evidence_artifact,
    }


def evaluation_manifest_document(evaluation: ModelEvaluation) -> dict[str, Any]:
    return {
        "schema": "eva.model-evaluation-manifest.v1",
        **evaluation_core(evaluation),
        "evaluation_blake3": evaluation.evaluation_blake3,
    }


def separation_core(report: AbilitySeparationReport) -> dict[str, Any]:
    core = {
        "rewards_by_cohort": report.rewards_by_cohort,
        "raw_item_scores_by_cohort": report.raw_item_scores_by_cohort,
        "strong_minus_weak": report.strong_minus_weak,
        "strong_minus_middle": report.strong_minus_middle,
        "middle_minus_weak": report.middle_minus_weak,
        "perfect_monotonic_staircase_required": report.perfect_monotonic_staircase_required,
        "policy_passed": report.policy_passed,
    }
    if report.admission_policy_schema is not None:
        core.update(
            {
                "admission_policy_schema": report.admission_policy_schema,
                "ability_separation_required_for_admission": (
                    report.ability_separation_required_for_admission
                ),
                "minimum_valid_judged_trajectories": (
                    report.minimum_valid_judged_trajectories
                ),
                "valid_judged_trajectory_count": (
                    report.valid_judged_trajectory_count
                ),
                "qualifying_evaluation_blake3s": (
                    report.qualifying_evaluation_blake3s
                ),
                "selected_evaluation_blake3": report.selected_evaluation_blake3,
                "selected_evaluation_cohort": report.selected_evaluation_cohort,
                "cohort_outcomes": report.cohort_outcomes,
                "failure_types_by_cohort": report.failure_types_by_cohort,
                "admission_policy_passed": report.admission_policy_passed,
            }
        )
    return core


def recommendation_core(recommendation: AdmissionRecommendation) -> dict[str, Any]:
    return {
        "recommendation": recommendation.recommendation,
        "reasons": recommendation.reasons,
        "admission_authorized": recommendation.admission_authorized,
        "signed_decision_created": recommendation.signed_decision_created,
    }


def telemetry_core(value: OutcomeTelemetry) -> dict[str, Any]:
    return {
        "cohort": value.cohort,
        "model_id": value.model_id,
        "rubric_digest": value.rubric_digest,
        "reward": value.reward,
        "item_scores": value.item_scores,
        "tool_call_count": value.tool_call_count,
        "tool_failure_count": value.tool_failure_count,
        "parallelism_observed": value.parallelism_observed,
        "workspace_changed": value.workspace_changed,
        "hard_gates_passed": value.hard_gates_passed,
        "outcome": value.outcome,
    }


def result_core(result: PipelineResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "sandbox_manifests": result.sandbox_manifests,
        "evaluations": result.evaluations,
        "separation": {
            **separation_core(result.separation),
            "report_blake3": result.separation.report_blake3,
        },
        "recommendation": result.recommendation,
        "telemetry": result.telemetry,
    }


def pipeline_result_document(result: PipelineResult) -> dict[str, Any]:
    """Canonical v1/v2 projection without rewriting legacy result bytes."""

    return {
        **result_core(result),
        "result_blake3": result.result_blake3,
    }


class VerifiableDataPipeline:
    """Run the three-cohort evidence and judge slice at maximum safe width."""

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        artifact_store: ImmutableArtifactStore,
        tool_registry: ToolRegistry,
        rollout_provider: RolloutProvider | Mapping[Cohort | str, RolloutProvider],
        judge: AgentJudge,
        rewarder: RewardComputer,
        id_factory: RuntimeIdFactory,
        judge_model_id: str = "claude-opus-5",
        separation_policy: SeparationPolicy | None = None,
        maximum_parallel_models: int = 3,
        maximum_parallel_tools: int = MAXIMUM_PARALLEL_TOOL_CALLS,
    ) -> None:
        if not 1 <= maximum_parallel_models <= 3:
            raise PipelineError("model rollout width must be in [1,3]")
        if not 1 <= maximum_parallel_tools <= MAXIMUM_PARALLEL_TOOL_CALLS:
            raise PipelineError(
                f"tool width must be in [1,{MAXIMUM_PARALLEL_TOOL_CALLS}]"
            )
        if "opus-5" not in judge_model_id.casefold().replace("_", "-"):
            raise PipelineError("the evidence judge must be Opus 5")
        self._workspace_root = Path(workspace_root)
        self._artifacts = artifact_store
        self._tools = tool_registry
        self._providers = rollout_provider
        self._judge = judge
        self._rewarder = rewarder
        self._ids = id_factory
        self._judge_model_id = judge_model_id
        self._policy = separation_policy or SeparationPolicy()
        self._model_width = maximum_parallel_models
        self._tool_width = maximum_parallel_tools

    def _provider_for(self, target: ModelTarget) -> RolloutProvider:
        if not isinstance(self._providers, Mapping):
            return self._providers
        for key in (target.cohort, target.cohort.value, target.model_id):
            if key in self._providers:
                return self._providers[key]
        raise PipelineError(f"no rollout provider for {target.cohort.value}")

    @staticmethod
    def _validate_targets(targets: Sequence[ModelTarget]) -> tuple[ModelTarget, ...]:
        values = tuple(targets)
        by_cohort = {target.cohort: target for target in values}
        if len(values) != 3 or set(by_cohort) != set(COHORT_ORDER):
            raise PipelineError("targets must contain weak, middle, and strong exactly once")
        if len({target.model_id for target in values}) != 3:
            raise PipelineError("ability cohorts must use three distinct model identities")
        return tuple(by_cohort[cohort] for cohort in COHORT_ORDER)

    def _policy_context(
        self, episode: BenchmarkEpisode, manifest: SandboxManifest
    ) -> Mapping[str, Any]:
        value = freeze_json(
            {
                "sandbox_id": manifest.sandbox_id,
                "episode_id": episode.episode_id,
                "domain": episode.domain,
                "stage": episode.stage.value,
                "instruction": episode.instruction,
                "episode_context": episode.policy_context,
                "rubric_binding": {
                    "rubric_id": manifest.rubric.rubric_id,
                    "version": manifest.rubric.version,
                    "digest": manifest.rubric.digest,
                },
            }
        )
        if not isinstance(value, Mapping):  # defensive type narrowing
            raise PipelineError("policy context projection differs")
        forbidden = _forbidden_policy_key(value)
        if forbidden is not None:
            raise PipelineError(f"private judge material leaked into policy context key {forbidden!r}")
        return value

    def _evaluate(
        self,
        *,
        run_id: str,
        target: ModelTarget,
        manifest: SandboxManifest,
        episode: BenchmarkEpisode,
        rubric: CompiledRubricTable,
        context: Mapping[str, Any],
        early_stop: Event,
        publication_lock: RLock,
    ) -> ModelEvaluation:
        def require_current() -> None:
            if early_stop.is_set():
                raise _EarlyCohortCancellation(
                    "cohort cancelled after one fully verified evaluation"
                )

        require_current()
        evaluation_id = self._ids.new("model-evaluation")
        rollout_id = self._ids.new("model-rollout")
        workspace_id = self._ids.new("cohort-workspace")
        for value, label in (
            (evaluation_id, "evaluation_id"),
            (rollout_id, "rollout_id"),
            (workspace_id, "workspace_id"),
        ):
            uuid_text(value, label=label)
        workspace = FilesystemSandbox(
            self._workspace_root / run_id,
            workspace_id,
            manifest.initial_files,
        )
        before = workspace.snapshot("before-rollout")
        runtime = ParallelToolRuntime(
            workspace=workspace,
            registry=self._tools,
            id_factory=self._ids,
            maximum_parallel_calls=self._tool_width,
        )
        request = RolloutRequest(
            rollout_id=rollout_id,
            sandbox=manifest,
            model=target,
            policy_visible_context=context,
            available_tools=self._tools.public_schemas(),
        )
        try:
            provider_result = self._provider_for(target).run(request, runtime)
        except Exception as exc:
            # Codex compatibility failures may carry a fully verified, raw-input-
            # free turn receipt.  Retain that evidence prospectively instead of
            # reducing it to only the Python exception class in the aggregate
            # quarantine.  No exception text or configuration value is stored.
            with publication_lock:
                require_current()
                receipt = getattr(exc, "receipt", None)
                receipt_value = canonical_value(receipt) if receipt is not None else None
                safe_detail = _safe_failure_detail(exc)
                if isinstance(receipt_value, dict) or safe_detail is not None:
                    safe_detail = _safe_failure_detail(exc) or {
                        "category": "provider_boundary",
                        "message": "Provider boundary failed closed",
                        "provenance": "eva_agent.pipeline.rollout_provider",
                    }
                    detail_core = {
                        "schema": "eva.provider-failure-evidence.v2",
                        "run_id": run_id,
                        "cohort": target.cohort.value,
                        "model_id": target.model_id,
                        "failure_type": type(exc).__name__,
                        "failure_category": safe_detail["category"],
                        "failure_message": safe_detail["message"],
                        "failure_provenance": safe_detail["provenance"],
                        "failure_message_blake3": blake3_hex(str(exc)),
                        "provider_turn_receipt": receipt_value,
                        "provider_turn_receipt_blake3": (
                            receipt_value.get("receipt_blake3")
                            if isinstance(receipt_value, dict)
                            else None
                        ),
                        "raw_exception_message_recorded": False,
                        "retry_count": 0,
                    }
                    detail = {
                        **detail_core,
                        "failure_evidence_blake3": blake3_hex(detail_core),
                    }
                    self._artifacts.publish(
                        run_id,
                        f"{target.cohort.value}/provider-failure-evidence.json",
                        detail,
                    )
            raise
        require_current()
        after = workspace.snapshot("after-rollout")
        trace = runtime.trace()
        if not provider_result.assistant_output.strip():
            raise PipelineError("rollout lacks a runnable terminal assistant output")
        bundle_id = self._ids.new("evidence-bundle")
        uuid_text(bundle_id, label="bundle_id")
        partial = EvidenceBundle(
            bundle_id=bundle_id,
            rollout_id=rollout_id,
            sandbox_manifest=manifest,
            model=target,
            policy_visible_context=context,
            context_blake3=blake3_hex(context),
            workspace_before=before,
            workspace_after=after,
            tool_trace=trace,
            policy_events=provider_result.policy_events,
            assistant_output=provider_result.assistant_output,
            provider_receipt_blake3=provider_result.provider_receipt_blake3,
            safe_provider_metadata=provider_result.safe_metadata,
            bundle_blake3="",
        )
        evidence = EvidenceBundle(
            **{**partial.__dict__, "bundle_blake3": blake3_hex(evidence_core(partial))}
        )
        judgment_id = self._ids.new("opus5-judgment")
        uuid_text(judgment_id, label="judgment_id")
        assessment = self._judge.judge(
            JudgeRequest(
                judgment_id=judgment_id,
                judge_model_id=self._judge_model_id,
                policy_visible_context=context,
                workspace_evidence=evidence,
                judge_only_reference=episode.judge_only_reference,
            ),
            rubric,
        )
        require_current()
        reward = self._rewarder.compute(
            reward_id=self._ids.new("rubric-reward"),
            rubric=rubric,
            assessment=assessment,
            evidence=evidence,
        )
        with publication_lock:
            require_current()
            prefix = target.cohort.value
            evidence_ref = self._artifacts.publish(
                run_id, f"{prefix}/evidence.json", evidence
            )
            provisional = ModelEvaluation(
                evaluation_id=evaluation_id,
                cohort=target.cohort,
                model=target,
                evidence=evidence,
                judgment=assessment,
                reward=reward,
                manifest_artifact=evidence_ref,
                evidence_artifact=evidence_ref,
                evaluation_blake3="",
            )
            provisional = ModelEvaluation(
                **{
                    **provisional.__dict__,
                    "evaluation_blake3": blake3_hex(evaluation_core(provisional)),
                }
            )
            manifest_ref = self._artifacts.publish(
                run_id,
                f"{prefix}/evaluation-manifest.json",
                evaluation_manifest_document(provisional),
            )
            evaluation = ModelEvaluation(
                **{**provisional.__dict__, "manifest_artifact": manifest_ref}
            )
            from .verify import PipelineVerifier

            PipelineVerifier(
                self._artifacts,
                separation_policy=self._policy,
                rewarder=self._rewarder,
            ).verify_evaluation_or_raise(
                evaluation=evaluation, rubric=rubric, run_id=run_id
            )
            if self._policy.early_continuation_after_first_valid_trajectory:
                early_stop.set()
            return evaluation

    def _separation(
        self,
        evaluations: Sequence[ModelEvaluation],
        *,
        integrity_passed: bool,
        failures: Mapping[str, str],
    ) -> AbilitySeparationReport:
        by_cohort = {evaluation.cohort: evaluation for evaluation in evaluations}
        rewards = {
            cohort.value: float(by_cohort[cohort].reward.total_reward)
            for cohort in COHORT_ORDER
            if cohort in by_cohort
        }
        raw = {
            cohort.value: {
                row.item_id: float(row.score)
                for row in by_cohort[cohort].reward.item_scores
            }
            for cohort in COHORT_ORDER
            if cohort in by_cohort
        }
        def delta(left: str, right: str) -> float | None:
            if left not in rewards or right not in rewards:
                return None
            return rewards[left] - rewards[right]

        strong_minus_weak = delta("strong", "weak")
        strong_minus_middle = delta("strong", "middle")
        middle_minus_weak = delta("middle", "weak")
        complete = set(by_cohort) == set(COHORT_ORDER)
        material = complete and any(
            abs(raw["strong"][item_id] - raw["weak"][item_id])
            >= self._policy.material_item_delta
            for item_id in raw["strong"]
        )
        separation_passed = bool(
            integrity_passed
            and complete
            and rewards["strong"] >= self._policy.minimum_strong_reward
            and strong_minus_weak is not None
            and strong_minus_weak >= self._policy.minimum_strong_minus_weak
            and material
            and (
                not self._policy.require_strong_hard_gates
                or by_cohort[Cohort.STRONG].reward.hard_gates_passed
            )
        )
        qualifying = tuple(evaluation.evaluation_blake3 for evaluation in evaluations) if integrity_passed else ()
        selected = (
            max(
                evaluations,
                key=lambda row: (
                    float(row.reward.total_reward),
                    COHORT_ORDER.index(row.cohort),
                ),
            )
            if qualifying
            else None
        )
        admission_passed = bool(
            len(qualifying) >= self._policy.minimum_valid_judged_trajectories
            and (
                not self._policy.ability_separation_required_for_admission
                or separation_passed
            )
        )
        outcomes = _cohort_outcomes(evaluations, failures)
        provisional = AbilitySeparationReport(
            rewards_by_cohort=MappingProxyType(rewards),
            raw_item_scores_by_cohort=MappingProxyType(
                {key: MappingProxyType(value) for key, value in raw.items()}
            ),
            strong_minus_weak=strong_minus_weak,
            strong_minus_middle=strong_minus_middle,
            middle_minus_weak=middle_minus_weak,
            perfect_monotonic_staircase_required=False,
            policy_passed=separation_passed,
            report_blake3="",
            admission_policy_schema=self._policy.admission_policy_schema,
            ability_separation_required_for_admission=(
                self._policy.ability_separation_required_for_admission
            ),
            minimum_valid_judged_trajectories=(
                self._policy.minimum_valid_judged_trajectories
            ),
            valid_judged_trajectory_count=len(qualifying),
            qualifying_evaluation_blake3s=qualifying,
            selected_evaluation_blake3=(
                selected.evaluation_blake3 if selected is not None else None
            ),
            selected_evaluation_cohort=(
                selected.cohort.value if selected is not None else None
            ),
            cohort_outcomes=MappingProxyType(outcomes),
            failure_types_by_cohort=MappingProxyType(dict(sorted(failures.items()))),
            admission_policy_passed=admission_passed,
        )
        return AbilitySeparationReport(
            **{**provisional.__dict__, "report_blake3": blake3_hex(separation_core(provisional))}
        )

    def run(
        self,
        *,
        episode: BenchmarkEpisode,
        rubric: CompiledRubricTable,
        targets: Sequence[ModelTarget],
    ) -> PipelineResult:
        ordered_targets = self._validate_targets(targets)
        run_id = self._ids.new("pipeline-run")
        sandbox_id = self._ids.new("rubric-bound-sandbox")
        uuid_text(run_id, label="run_id")
        manifest = SandboxManifest.create(
            sandbox_id=sandbox_id,
            episode=episode,
            rubric=rubric,
        )
        context = self._policy_context(episode, manifest)
        self._artifacts.begin_run(run_id)

        evaluations: list[ModelEvaluation] = []
        failures: dict[str, str] = {}
        early_stop = Event()
        publication_lock = RLock()
        pool = ThreadPoolExecutor(max_workers=self._model_width)
        futures: dict[Future[ModelEvaluation], ModelTarget] = {}
        selected_early = False
        try:
            futures = {
                pool.submit(
                    self._evaluate,
                    run_id=run_id,
                    target=target,
                    manifest=manifest,
                    episode=episode,
                    rubric=rubric,
                    context=context,
                    early_stop=early_stop,
                    publication_lock=publication_lock,
                ): target
                for target in ordered_targets
            }
            for future in as_completed(futures):
                target = futures[future]
                try:
                    evaluations.append(future.result())
                    if self._policy.early_continuation_after_first_valid_trajectory:
                        selected_early = True
                        break
                except Exception as exc:
                    # Provider/judge transport failures are retained as
                    # infrastructure outcomes and never converted into a zero.
                    failures[target.cohort.value] = type(exc).__name__
        finally:
            if selected_early:
                for future, target in futures.items():
                    if any(row.cohort is target.cohort for row in evaluations):
                        continue
                    if future.done() and not future.cancelled():
                        try:
                            future.result()
                        except Exception as exc:
                            failures[target.cohort.value] = type(exc).__name__
                        else:  # pragma: no cover - publication lock admits one
                            raise PipelineError("multiple early evaluations were accepted")
                    else:
                        future.cancel()
                        failures[target.cohort.value] = _EarlyCohortCancellation.__name__
                pool.shutdown(wait=False, cancel_futures=True)
            else:
                pool.shutdown(wait=True, cancel_futures=False)
        if failures and not evaluations:
            quarantine = {
                "schema": "eva.pipeline-infrastructure-quarantine.v1",
                "run_id": run_id,
                "sandbox_manifest_blake3": manifest.manifest_blake3,
                "failures": failures,
                "semantic_score_created": False,
                "retry_count": 0,
            }
            quarantine["quarantine_blake3"] = blake3_hex(quarantine)
            self._artifacts.publish(run_id, "infrastructure-quarantine.json", quarantine)
            self._artifacts.seal_run(run_id)
            raise InfrastructureQuarantineError(run_id, failures)

        if failures:
            partial_failure = partial_cohort_outcome_document(
                run_id=run_id,
                manifest_blake3=manifest.manifest_blake3,
                evaluations=evaluations,
                failures=failures,
                early_continuation_enabled=(
                    self._policy.early_continuation_after_first_valid_trajectory
                ),
            )
            self._artifacts.publish(
                run_id, "partial-cohort-failures.json", partial_failure
            )

        order = {cohort: index for index, cohort in enumerate(COHORT_ORDER)}
        evaluations.sort(key=lambda value: order[value.cohort])
        before_roots = {value.evidence.workspace_before.tree_blake3 for value in evaluations}
        exact_rubric = all(
            value.evidence.sandbox_manifest.rubric.digest == rubric.digest
            and value.reward.rubric_digest == rubric.digest
            and value.judgment.rubric_digest == rubric.digest
            for value in evaluations
        )
        leak_free = _forbidden_policy_key(context) is None
        artifacts_valid = all(
            self._artifacts.verify(value.evidence_artifact)
            and self._artifacts.verify(value.manifest_artifact)
            for value in evaluations
        )
        integrity_passed = (
            len(before_roots) == 1
            and exact_rubric
            and leak_free
            and artifacts_valid
            and all(value.evidence.assistant_output.strip() for value in evaluations)
        )
        separation = self._separation(
            evaluations,
            integrity_passed=integrity_passed,
            failures=failures,
        )
        admission_passed = separation.admission_policy_passed is True
        reasons = (
            (
                "integrity gates passed",
                "at least one trajectory passed workspace-aware Opus 5 judging and exact rubric verification",
                "ability-separation and incomplete cohort outcomes are retained as non-gating metadata",
                "requires signed supervisor admission receipt",
            )
            if admission_passed
            else (
                "no valid workspace-judged exact-rubric trajectory completed",
                "ability separation remains diagnostic and is not an admission prerequisite",
            )
        )
        recommendation = AdmissionRecommendation(
            recommendation=(
                "eligible_for_signed_supervisor_admission"
                if admission_passed
                else "not_eligible"
            ),
            reasons=reasons,
            admission_authorized=False,
            signed_decision_created=False,
            recommendation_blake3="",
        )
        recommendation = AdmissionRecommendation(
            **{
                **recommendation.__dict__,
                "recommendation_blake3": blake3_hex(recommendation_core(recommendation)),
            }
        )
        telemetry: list[OutcomeTelemetry] = []
        for evaluation in evaluations:
            trace = evaluation.evidence.tool_trace
            row = OutcomeTelemetry(
                cohort=evaluation.cohort,
                model_id=evaluation.model.model_id,
                rubric_digest=evaluation.reward.rubric_digest,
                reward=evaluation.reward.total_reward,
                item_scores=MappingProxyType(
                    {score.item_id: score.score for score in evaluation.reward.item_scores}
                ),
                tool_call_count=len(trace.results),
                tool_failure_count=sum(
                    result.status != "completed" for result in trace.results
                ),
                parallelism_observed=trace.max_parallelism_observed,
                workspace_changed=(
                    evaluation.evidence.workspace_before.tree_blake3
                    != evaluation.evidence.workspace_after.tree_blake3
                ),
                hard_gates_passed=evaluation.reward.hard_gates_passed,
                outcome="completed",
                telemetry_blake3="",
            )
            telemetry.append(
                OutcomeTelemetry(
                    **{**row.__dict__, "telemetry_blake3": blake3_hex(telemetry_core(row))}
                )
            )
        provisional = PipelineResult(
            run_id=run_id,
            sandbox_manifests=(manifest,),
            evaluations=tuple(evaluations),
            separation=separation,
            recommendation=recommendation,
            telemetry=tuple(telemetry),
            result_blake3="",
        )
        result = PipelineResult(
            **{**provisional.__dict__, "result_blake3": blake3_hex(result_core(provisional))}
        )
        self._artifacts.publish(
            run_id, "pipeline-result.json", pipeline_result_document(result)
        )
        self._artifacts.seal_run(run_id)
        return result


__all__ = [
    "InfrastructureQuarantineError",
    "PipelineError",
    "SeparationPolicy",
    "VerifiableDataPipeline",
    "evaluation_core",
    "evaluation_manifest_document",
    "evidence_core",
    "recommendation_core",
    "pipeline_result_document",
    "partial_cohort_outcome_document",
    "result_core",
    "separation_core",
    "telemetry_core",
]
