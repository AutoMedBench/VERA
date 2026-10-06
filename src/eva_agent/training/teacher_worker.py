"""One real Codex/MCP teacher rollout with no judge, cascade, or admission."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Mapping

from eva_agent.codex_pipeline import CodexRolloutAdapter
from eva_agent.codex_pipeline.native_policy_v2 import (
    StageToolGuidanceV1,
    advance_guided_stage_frontier_v1,
    build_stage_tool_guidance_v1,
    materialize_s2_selection_arguments_v1,
    start_guided_stage_frontier_v1,
    verify_guided_stage_call_v1,
)
from eva_agent.pipeline import (
    BenchmarkEpisode,
    Cohort,
    CompiledRubricTable,
    FilesystemSandbox,
    ModelTarget,
    ParallelToolRuntime,
    RandomUUIDFactory,
    RolloutRequest,
    SandboxManifest,
    Stage,
    ToolCall,
    ToolRegistry,
)
from eva_agent.pipeline.contracts import RolloutProvider
from eva_agent.pipeline.digests import blake3_bytes, canonical_json_bytes, canonical_value

from .teacher_batch import TEACHER_ROUTES, TeacherBatchError


_TEACHER_PRIVATE_LEXEME = re.compile(
    r"\b(?:private[ _-]?rubric|ground[ _-]?truth|gold[ _-]?answer|"
    r"answer[ _-]?key|judge[ _-]?only[ _-]?(?:context|reference)|"
    r"reference[ _-]?answer)\b",
    re.IGNORECASE,
)
_TEACHER_PUBLIC_REPLACEMENTS = {
    "privaterubric": "withheld scoring criteria",
    "groundtruth": "benchmark target",
    "goldanswer": "benchmark label",
    "answerkey": "benchmark label",
    "judgeonlycontext": "withheld evaluation context",
    "judgeonlyreference": "withheld evaluation reference",
    "referenceanswer": "benchmark reference",
}

TEACHER_TERMINAL_INSTRUCTION = (
    "Keep between-tool commentary brief and prioritize the next required tool "
    "call over long prose. "
    "After the guided tool frontiers finish, emit one non-empty final answer "
    "that names the completed stage gates and exact workspace artifact paths. "
    "Do not end the turn with an empty final_answer."
)


def teacher_safe_actor_instruction(instruction: str) -> tuple[str, bool]:
    """Rephrase privacy-reserved lexemes without hiding public reward criteria."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise TeacherBatchError("teacher actor instruction differs")

    def replace_match(match: re.Match[str]) -> str:
        normalized = re.sub(r"[ _-]", "", match.group(0).casefold())
        return _TEACHER_PUBLIC_REPLACEMENTS[normalized]

    projected = _TEACHER_PRIVATE_LEXEME.sub(replace_match, instruction)
    if _TEACHER_PRIVATE_LEXEME.search(projected) is not None:
        raise TeacherBatchError("teacher actor instruction projection differs")
    return projected, projected != instruction


@dataclass(frozen=True)
class TeacherCandidateContext:
    episode: BenchmarkEpisode
    rubric: CompiledRubricTable
    tool_registry: ToolRegistry
    turn_mcp_factory: Any
    skills_factory: Any
    skill_delivery: Mapping[str, Any] = MappingProxyType({})
    stage_tool_guidance: StageToolGuidanceV1 | None = None


def teacher_actor_developer_instructions(
    context: TeacherCandidateContext,
) -> str | None:
    """Deliver exact focus-aware guidance plus the terminal-turn contract."""

    guidance = context.stage_tool_guidance
    if guidance is None:
        return None
    if guidance.focus not in {Stage.S1, Stage.S2, Stage.S3}:
        raise TeacherBatchError("teacher stage-tool guidance focus differs")
    return f"{guidance.prompt_text}\n\n{TEACHER_TERMINAL_INSTRUCTION}"


def _stage_artifact_path(
    context: TeacherCandidateContext, *, stage: Stage
) -> str:
    execution = context.episode.policy_context.get("execution_binding")
    s1 = execution.get("s1_plan_contract") if isinstance(execution, Mapping) else None
    artifacts = s1.get("stage_artifacts") if isinstance(s1, Mapping) else None
    path = artifacts.get(stage.value) if isinstance(artifacts, Mapping) else None
    if not isinstance(path, str) or not path:
        raise TeacherBatchError("teacher prerequisite artifact binding differs")
    return path


def _teacher_s2_selection_arguments(
    guidance: StageToolGuidanceV1, *, state: Any
) -> Mapping[str, Any]:
    selection_frontiers = tuple(
        frontier
        for frontier in guidance.frontiers
        if frontier.get("tool_name") == "materialize_evidence_selection"
    )
    if len(selection_frontiers) != 1:
        raise TeacherBatchError("teacher prerequisite S2 selection frontier differs")
    blueprint = selection_frontiers[0].get("argument_blueprint")
    inferences = blueprint.get("inferences") if isinstance(blueprint, Mapping) else None
    if not isinstance(inferences, (tuple, list)) or not inferences:
        raise TeacherBatchError("teacher prerequisite S2 inference inventory differs")
    claim_ids = tuple(
        row.get("inference_id") if isinstance(row, Mapping) else None
        for row in inferences
    )
    if any(not isinstance(claim_id, str) or not claim_id for claim_id in claim_ids):
        raise TeacherBatchError("teacher prerequisite S2 inference identity differs")
    try:
        return materialize_s2_selection_arguments_v1(
            guidance,
            state=state,
            inference_text_by_claim_id={
                claim_id: (
                    "The selected frozen evidence provides the declared support "
                    "for this bound research inference."
                )
                for claim_id in claim_ids
            },
            unresolved_gaps=(
                "The frozen evidence does not resolve every research uncertainty.",
            ),
            limitations=("Only the declared frozen evidence was available.",),
        )
    except ValueError as exc:
        raise TeacherBatchError(
            "teacher prerequisite S2 selection materialization differs"
        ) from exc


def _hydrate_teacher_s3_prerequisites(
    *,
    context: TeacherCandidateContext,
    workspace: FilesystemSandbox,
    id_factory: Any,
) -> Mapping[str, Any]:
    """Replay and receipt-reopen the complete S1/S2 frontier before S3."""

    guidance = context.stage_tool_guidance
    if guidance is None or guidance.focus is not Stage.S3:
        raise TeacherBatchError("teacher S3 prerequisite guidance differs")
    if guidance.to_document().get("first_frontier_index") != len(
        guidance.frontiers
    ):
        raise TeacherBatchError("teacher S3 actor frontier binding differs")
    before = workspace.snapshot("before-prerequisite-hydration")
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=context.tool_registry,
        id_factory=id_factory,
        maximum_parallel_calls=1,
    )
    state = start_guided_stage_frontier_v1(guidance)
    results = []
    workspace_cursor = before.tree_blake3
    selection_arguments: Mapping[str, Any] | None = None
    for frontier_index, frontier in enumerate(guidance.frontiers):
        tool_name = frontier.get("tool_name")
        if tool_name in {"materialize_plan", "retrieve_frozen_evidence"}:
            arguments = frontier.get("arguments")
            if not isinstance(arguments, Mapping):
                raise TeacherBatchError(
                    "teacher prerequisite dispatchable arguments differ"
                )
        elif tool_name == "materialize_evidence_selection":
            arguments = _teacher_s2_selection_arguments(guidance, state=state)
            selection_arguments = arguments
        else:
            raise TeacherBatchError("teacher prerequisite tool frontier differs")
        call_id = id_factory.new(f"teacher-prerequisite-{tool_name}-call")
        try:
            proof = verify_guided_stage_call_v1(
                guidance,
                state=state,
                frontier_index=frontier_index,
                tool_call_id=call_id,
                tool_name=str(tool_name),
                arguments=arguments,
            )
        except ValueError as exc:
            raise TeacherBatchError(
                "teacher prerequisite call proof differs"
            ) from exc
        observed = runtime.execute(
            (
                ToolCall(
                    call_id=call_id,
                    name=str(tool_name),
                    arguments=arguments,
                ),
            )
        )
        if len(observed) != 1:
            raise TeacherBatchError(
                "teacher prerequisite result inventory differs"
            )
        result = observed[0]
        output = result.output
        after_call = workspace.snapshot(
            f"after-prerequisite-frontier-{frontier_index}"
        )
        if (
            result.workspace_before_blake3 != workspace_cursor
            or result.workspace_after_blake3 != after_call.tree_blake3
            or not isinstance(output, Mapping)
            or output.get("gate_passed") is not True
            or output.get("failed_check_ids") != ()
            or not isinstance(output.get("receipt_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", output["receipt_sha256"]) is None
        ):
            raise TeacherBatchError(
                "teacher prerequisite signed workspace gate differs"
            )
        mutates_workspace = tool_name in {
            "materialize_plan",
            "materialize_evidence_selection",
        }
        if (
            result.workspace_before_blake3 != result.workspace_after_blake3
        ) != mutates_workspace:
            raise TeacherBatchError("teacher prerequisite workspace delta differs")
        try:
            state = advance_guided_stage_frontier_v1(
                guidance,
                state=state,
                frontier_index=frontier_index,
                tool_call_id=call_id,
                tool_name=str(tool_name),
                arguments=arguments,
                call_proof=proof,
                tool_result=result,
            )
        except ValueError as exc:
            raise TeacherBatchError(
                "teacher prerequisite result receipt did not reopen"
            ) from exc
        results.append(result)
        workspace_cursor = after_call.tree_blake3
    if state.next_frontier_index != len(guidance.frontiers):
        raise TeacherBatchError("teacher prerequisite frontier is incomplete")
    execution = context.episode.policy_context.get("execution_binding")
    s2 = execution.get("s2_evidence_contract") if isinstance(execution, Mapping) else None
    selection_path = s2.get("selection_relative_path") if isinstance(s2, Mapping) else None
    s1_path = _stage_artifact_path(context, stage=Stage.S1)
    s2_path = _stage_artifact_path(context, stage=Stage.S2)
    if not isinstance(selection_path, str) or s2_path != selection_path:
        raise TeacherBatchError("teacher prerequisite S2 artifact binding differs")
    try:
        plan_artifact = json.loads(workspace.read_bytes(s1_path))
        selection_artifact = json.loads(workspace.read_bytes(s2_path))
    except Exception as exc:
        raise TeacherBatchError("teacher prerequisite artifact differs") from exc
    if (
        canonical_value(plan_artifact)
        != canonical_value(guidance.frontiers[0].get("arguments"))
        or selection_arguments is None
        or canonical_value(selection_artifact)
        != canonical_value(selection_arguments)
    ):
        raise TeacherBatchError("teacher prerequisite artifact content differs")
    after = workspace.snapshot("after-prerequisite-hydration")
    if after.tree_blake3 != workspace_cursor or before.tree_blake3 == after.tree_blake3:
        raise TeacherBatchError("teacher prerequisite final workspace delta differs")
    return MappingProxyType(
        {
            "schema": "eva.codex-teacher-prerequisite-hydration.v1",
            "focus": Stage.S3.value,
            "source_candidate_id": guidance.source_candidate_id,
            "guidance_blake3": guidance.guidance_blake3,
            "hydrated_frontier_indices": list(range(len(guidance.frontiers))),
            "next_actor_frontier_index": len(guidance.frontiers),
            "artifact_relative_paths": [s1_path, s2_path],
            "tool_results": canonical_value(results),
            "tool_result_receipt_blake3s": list(
                state.tool_result_receipt_blake3s
            ),
            "frontier_state_blake3": state.state_blake3,
            "workspace_before_blake3": before.tree_blake3,
            "workspace_after_blake3": after.tree_blake3,
            "provider_calls": 0,
            "canonical_tool_schemas_changed": False,
            "mcp_wire_schema_changed": False,
        }
    )


def hydrate_teacher_stage_prerequisites(
    *,
    context: TeacherCandidateContext,
    workspace: FilesystemSandbox,
    id_factory: Any,
) -> Mapping[str, Any] | None:
    """Replay exact host-owned predecessors before a later-stage actor turn.

    S2 receives its complete dispatchable S1 example.  S3 receives that same
    S1 plan plus every exact S2 retrieval and one deterministically completed
    selection, all through the real signed host handlers in one workspace.
    """

    guidance = context.stage_tool_guidance
    if guidance is None or guidance.focus is Stage.S1:
        return None
    if guidance.focus is Stage.S3:
        return _hydrate_teacher_s3_prerequisites(
            context=context,
            workspace=workspace,
            id_factory=id_factory,
        )
    if guidance.focus is not Stage.S2:
        raise TeacherBatchError("teacher prerequisite hydration focus differs")
    if len(guidance.frontiers) < 2:
        raise TeacherBatchError("teacher prerequisite frontier inventory differs")
    frontier = guidance.frontiers[0]
    if (
        frontier.get("frontier_index") != 0
        or frontier.get("stage") != Stage.S1.value
        or frontier.get("tool_name") != "materialize_plan"
        or frontier.get("arguments_kind") != "complete-dispatchable-example"
        or frontier.get("parallel_allowed") is not False
        or frontier.get("single_call_required") is not True
        or frontier.get("must_observe_gate_passed_before_next") is not True
        or not isinstance(frontier.get("arguments"), Mapping)
    ):
        raise TeacherBatchError("teacher prerequisite S1 frontier differs")

    before = workspace.snapshot("before-prerequisite-hydration")
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=context.tool_registry,
        id_factory=id_factory,
        maximum_parallel_calls=1,
    )
    observed = runtime.execute(
        (
            ToolCall(
                call_id=id_factory.new("teacher-prerequisite-s1-call"),
                name="materialize_plan",
                arguments=frontier["arguments"],
            ),
        )
    )
    if len(observed) != 1:
        raise TeacherBatchError("teacher prerequisite S1 result inventory differs")
    result = observed[0]
    output = result.output
    if (
        result.name != "materialize_plan"
        or result.status != "completed"
        or result.error_code is not None
        or not isinstance(output, Mapping)
        or output.get("stage") != Stage.S1.value
        or output.get("effect") != "plan_materialization"
        or output.get("gate_passed") is not True
        or output.get("failed_check_ids") != ()
        or output.get("next_stage") != Stage.S2.value
    ):
        raise TeacherBatchError("teacher prerequisite S1 host gate did not pass")
    after = workspace.snapshot("after-prerequisite-hydration")
    if (
        result.workspace_before_blake3 != before.tree_blake3
        or result.workspace_after_blake3 != after.tree_blake3
        or before.tree_blake3 == after.tree_blake3
    ):
        raise TeacherBatchError("teacher prerequisite workspace delta differs")
    artifact_path = _stage_artifact_path(context, stage=Stage.S1)
    try:
        artifact = json.loads(workspace.read_bytes(artifact_path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TeacherBatchError("teacher prerequisite artifact differs") from exc
    if canonical_value(artifact) != canonical_value(frontier["arguments"]):
        raise TeacherBatchError("teacher prerequisite artifact content differs")
    return MappingProxyType(
        {
            "schema": "eva.codex-teacher-prerequisite-hydration.v1",
            "focus": Stage.S2.value,
            "source_candidate_id": guidance.source_candidate_id,
            "guidance_blake3": guidance.guidance_blake3,
            "hydrated_frontier_indices": [0],
            "next_actor_frontier_index": 1,
            "artifact_relative_paths": [artifact_path],
            "tool_result": canonical_value(result),
            "workspace_before_blake3": before.tree_blake3,
            "workspace_after_blake3": after.tree_blake3,
            "provider_calls": 0,
            "canonical_tool_schemas_changed": False,
            "mcp_wire_schema_changed": False,
        }
    )


def load_bulk_record(root: str | Path, sandbox_id: str) -> Mapping[str, Any]:
    bulk = Path(root).resolve(strict=True)
    manifest = json.loads((bulk / "manifest.json").read_text(encoding="utf-8"))
    payload = manifest.get("payload") if isinstance(manifest, dict) else None
    shards = payload.get("shards") if isinstance(payload, dict) else None
    if not isinstance(shards, list):
        raise TeacherBatchError("bulk manifest differs")
    for descriptor in shards:
        path = bulk / descriptor["path"]
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if isinstance(row, dict) and row.get("sandbox_id") == sandbox_id:
                return MappingProxyType(row)
    raise TeacherBatchError("teacher sandbox is absent from the bulk release")


def bulk_initial_files(record: Mapping[str, Any]) -> Mapping[str, bytes]:
    state = record.get("workspace_initial_state")
    rows = state.get("files") if isinstance(state, Mapping) else None
    if not isinstance(rows, list):
        raise TeacherBatchError("bulk workspace inventory differs")
    files: dict[str, bytes] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise TeacherBatchError("bulk workspace file differs")
        payload = canonical_json_bytes(row.get("content"))
        if len(payload) != row.get("byte_count") or blake3_bytes(payload) != row.get("content_blake3"):
            raise TeacherBatchError("bulk workspace bytes differ")
        files[row["path"]] = payload
    if len(files) != state.get("file_count") or sum(map(len, files.values())) != state.get("byte_count"):
        raise TeacherBatchError("bulk workspace totals differ")
    return MappingProxyType(files)


def execute_single_rollout(
    *,
    record: Mapping[str, Any],
    context: TeacherCandidateContext,
    provider: RolloutProvider,
    target: ModelTarget,
    workspace_root: str | Path,
    task_id: str | None = None,
    route_id: str | None = None,
) -> Mapping[str, Any]:
    """Execute exactly one provider turn and emit the complete training receipt."""

    if target.cohort is not Cohort.STRONG:
        raise TeacherBatchError("teacher rollout target must use the non-cascade actor role")
    episode = context.episode
    if (
        record.get("candidate_id") is None
        or record.get("domain") != episode.domain
        or record.get("stage") != episode.stage.value
    ):
        raise TeacherBatchError("bulk sandbox and executable candidate differ")
    reward = record.get("reward_contract")
    rubric_document = reward.get("rubric_table") if isinstance(reward, Mapping) else None
    if canonical_value(rubric_document) != canonical_value(context.rubric.to_document()):
        raise TeacherBatchError("bulk sandbox rubric differs from compiled executable rubric")
    initial = dict(episode.initial_files)
    for path, payload in bulk_initial_files(record).items():
        if path in initial and initial[path] != payload:
            raise TeacherBatchError("bulk workspace collides with executable binding")
        initial[path] = payload
    episode = replace(episode, initial_files=initial)
    sandbox_id = str(record["sandbox_id"])
    ids = RandomUUIDFactory()
    workspace = FilesystemSandbox(Path(workspace_root), sandbox_id, episode.initial_files)
    hydration = hydrate_teacher_stage_prerequisites(
        context=context,
        workspace=workspace,
        id_factory=ids,
    )
    if hydration is not None:
        hydrated = workspace.snapshot("actor-initial-state")
        policy = dict(episode.policy_context)
        policy["teacher_prerequisite_hydration"] = hydration
        episode = replace(
            episode,
            initial_files={row.path: row.content for row in hydrated.files},
            policy_context=policy,
        )
    manifest = SandboxManifest.create(
        sandbox_id=sandbox_id, episode=episode, rubric=context.rubric
    )
    before = workspace.snapshot("before-rollout")
    tools = ParallelToolRuntime(
        workspace=workspace,
        registry=context.tool_registry,
        id_factory=ids,
        maximum_parallel_calls=64,
    )
    request = RolloutRequest(
        rollout_id=ids.new("teacher-rollout"),
        sandbox=manifest,
        model=target,
        policy_visible_context=episode.policy_context,
        available_tools=context.tool_registry.public_schemas(),
    )
    result = provider.run(request, tools)
    after = workspace.snapshot("after-rollout")
    trace = tools.trace()
    return {
        "schema": "eva.codex-teacher-full-trajectory.v1",
        "task_id": task_id or os.environ.get("EVA_TEACHER_TASK_ID"),
        "sandbox_id": sandbox_id,
        "candidate_id": record["candidate_id"],
        "episode_id": record["episode_id"],
        "executable_episode_id": episode.episode_id,
        "route_id": route_id or os.environ.get("EVA_TEACHER_ROUTE_ID"),
        "model_id": target.model_id,
        "provider": target.provider,
        "score": 0.0,
        "score_kind": "selection_pending",
        "selection_pending": True,
        "messages": canonical_value(result.policy_events),
        "assistant_output": result.assistant_output,
        "tool_trace": canonical_value(trace),
        "skill_delivery": canonical_value(context.skill_delivery),
        "workspace_before": canonical_value(before),
        "workspace_after": canonical_value(after),
        "rubric_table": rubric_document,
        "provider_receipt_blake3": result.provider_receipt_blake3,
        "provider_metadata": canonical_value(result.safe_metadata),
        "semantic_retry_count": 0,
        "judge_calls": 0,
        "admission_writes": 0,
        "cascade_required": False,
    }


def _campaign_v2_context_resources() -> tuple[Any, Any, Any, Any]:
    """Build immutable campaign resources once for a persistent teacher process."""

    import sys
    from eva_agent.campaign import (
        FrozenCampaignCandidateSourceV2,
        exact_6000_plan,
        load_campaign_selection,
        load_campaign_selection_v2,
    )
    from eva_agent.codex_pipeline import TurnMCPBridgeFactory, VerifiedActorSkillCatalog
    from eva_agent.deployment.medresearch_v2 import (
        IMAGE_REFS,
        LEGACY_ROOT,
        LEGACY_SKILL_ROOT,
        PLUGIN_ROOT,
        PROJECT_ROOT,
        RUBRIC_SOURCE,
        SELECTION_V1,
        SELECTION_V2,
        V24_ROOT,
        V4_ROOT,
        _prepare_turn_mcp_runtime_root,
    )
    from eva_agent.pipeline import Stage
    from eva_agent.rubrics import load_and_compile_registry
    from eva_agent.sources import (
        LegacyExecutionBindingResolver,
        LegacySupervisorV4Importer,
        load_signed_supervisor_v24_readiness,
    )

    trust_store = LEGACY_ROOT / "config/host-trust-store.v1.json"
    private_key = Path(
        os.environ.get(
            "EVA_TEACHER_HOST_PRIVATE_KEY",
            "/localhome/local-operator/.config/rlevo-med-research/host-signing-ed25519-v1.pem",
        )
    )
    key_id = os.environ.get("EVA_TEACHER_HOST_KEY_ID", "rlevo-host-20260902-v1")
    plan = exact_6000_plan()
    rubrics = load_and_compile_registry(RUBRIC_SOURCE)
    importer = LegacySupervisorV4Importer(V4_ROOT, authority_root=LEGACY_ROOT)
    base = load_campaign_selection(SELECTION_V1, plan=plan, rubrics=rubrics)
    readiness = load_signed_supervisor_v24_readiness(
        V24_ROOT,
        authority_root=LEGACY_ROOT,
        trust_store_path=trust_store,
        worker_width=64,
    )
    selection = load_campaign_selection_v2(
        SELECTION_V2,
        base_selection=base,
        readiness=readiness,
        plan=plan,
        rubrics=rubrics,
    )
    targets = tuple(
        ModelTarget(cohort, f"teacher-binding-{cohort.value}", "local-binding")
        for cohort in (Cohort.WEAK, Cohort.MIDDLE, Cohort.STRONG)
    )
    source = FrozenCampaignCandidateSourceV2(
        selection=selection,
        base_selection=base,
        readiness=readiness,
        importer=importer,
        rubrics=rubrics,
        targets=targets,
        plan=plan,
    )
    state_root = PROJECT_ROOT / "runs/teacher-rollout-runtime-state"
    state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolver = LegacyExecutionBindingResolver(
        authority_root=LEGACY_ROOT,
        supervisor_root=V24_ROOT,
        trust_store_path=trust_store,
        legacy_python_root=LEGACY_ROOT / "src",
        runtime_state_root=state_root,
        host_private_key_path=private_key,
        host_key_id=key_id,
        image_refs_path=IMAGE_REFS,
        worker_width=64,
    )
    skill_root = PROJECT_ROOT / "runs/teacher-rollout-skill-materialization"
    skill_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    skill_root.chmod(0o700)
    verified_skills = VerifiedActorSkillCatalog(
        manifest_path=PLUGIN_ROOT / "references/legacy-skill-manifest.v1.json",
        legacy_source_root=LEGACY_SKILL_ROOT,
        native_stage_skill_path=PLUGIN_ROOT / "skills/stage-rollout/SKILL.md",
        runtime_root=skill_root.resolve(),
    )
    from .progressive_skills import ProgressiveTeacherSkillSurface

    skills = ProgressiveTeacherSkillSurface(verified_skills)
    turn_mcp = TurnMCPBridgeFactory(
        proxy_python=Path(sys.executable).resolve(),
        proxy_script=PROJECT_ROOT / "src/eva_agent/codex_pipeline/turn_mcp_proxy.py",
        temp_root=_prepare_turn_mcp_runtime_root(),
        maximum_parallel_calls=64,
    )
    return source, resolver, skills, turn_mcp


class CampaignV2TeacherContextPool:
    """Single-flight cache over one verified source/resolver/skill composition.

    The expensive frozen-selection, v24 readiness, execution-binding catalog,
    and skill materialization are opened once per long-lived process.  Different
    candidates may materialize concurrently; concurrent model routes for the
    same candidate share exactly one immutable context result.
    """

    def __init__(
        self,
        *,
        source: Any | None = None,
        resolver: Any | None = None,
        skills: Any | None = None,
        turn_mcp_factory: Any | None = None,
    ) -> None:
        injected = (source, resolver, skills, turn_mcp_factory)
        if all(value is None for value in injected):
            source, resolver, skills, turn_mcp_factory = (
                _campaign_v2_context_resources()
            )
        elif any(value is None for value in injected):
            raise TeacherBatchError("teacher context resource injection differs")
        self._source = source
        self._resolver = resolver
        self._skills = skills
        self._turn_mcp = turn_mcp_factory
        self._cache: dict[str, tuple[TeacherCandidateContext, Any]] = {}
        self._inflight: dict[
            str, Future[tuple[TeacherCandidateContext, Any]]
        ] = {}
        self._lock = RLock()
        self._build_count = 0

    @property
    def turn_mcp_factory(self) -> Any:
        return self._turn_mcp

    @property
    def skills_factory(self) -> Any:
        return self._skills

    @property
    def context_build_count(self) -> int:
        with self._lock:
            return self._build_count

    def _build(
        self, record: Mapping[str, Any]
    ) -> tuple[TeacherCandidateContext, Any]:
        candidate_id = str(record["candidate_id"])
        job = self._source.load(candidate_id)
        source_id = self._source.source_candidate_id(candidate_id)
        if source_id != record.get("source_binding", {}).get(
            "source_candidate_id"
        ):
            raise TeacherBatchError("bulk source binding differs")
        binding = self._resolver.resolve(
            candidate_id, source_candidate_id=source_id
        )
        initial = dict(job.episode.initial_files)
        for path, payload in binding.initial_workspace_files.items():
            if path in initial and initial[path] != payload:
                raise TeacherBatchError(
                    "executable binding workspace collides with source episode"
                )
            initial[path] = payload
        policy = dict(job.episode.policy_context)
        policy["execution_binding"] = binding.public_runtime_context
        reward = record.get("reward_contract")
        rubric_document = (
            reward.get("rubric_table") if isinstance(reward, Mapping) else None
        )
        if canonical_value(rubric_document) != canonical_value(
            job.rubric.to_document()
        ):
            raise TeacherBatchError(
                "bulk sandbox rubric differs from executable context"
            )
        policy["public_reward_contract"] = {
            "rubric_table": canonical_value(rubric_document)
        }
        instruction, instruction_rephrased = teacher_safe_actor_instruction(
            job.episode.instruction
        )
        policy["teacher_actor_projection"] = {
            "judge_only_material_included": False,
            "public_reward_contract_included": True,
            "privacy_reserved_instruction_lexemes_rephrased": instruction_rephrased,
        }
        stage_tool_guidance = None
        if job.episode.stage in {Stage.S1, Stage.S2, Stage.S3}:
            stage_tool_guidance = build_stage_tool_guidance_v1(
                public_runtime_context=binding.public_runtime_context,
                source_tool_catalog=binding.tool_registry.public_schemas(),
            )
            if (
                stage_tool_guidance.focus is not job.episode.stage
                or stage_tool_guidance.source_candidate_id != source_id
            ):
                raise TeacherBatchError(
                    "teacher stage-tool guidance binding differs"
                )
            policy["stage_tool_guidance_binding"] = {
                "schema": "eva.codex-stage-tool-guidance.v1",
                "source_candidate_id": stage_tool_guidance.source_candidate_id,
                "guidance_blake3": stage_tool_guidance.guidance_blake3,
                "prompt_blake3": stage_tool_guidance.prompt_blake3,
                "public_runtime_context_blake3": (
                    stage_tool_guidance.public_runtime_context_blake3
                ),
                "source_tool_catalog_blake3": (
                    stage_tool_guidance.source_tool_catalog_blake3
                ),
                "delivery": "codex-developer-instruction-sidecar",
                "canonical_tool_schemas_changed": False,
            }
        episode = replace(
            job.episode,
            instruction=instruction,
            initial_files=initial,
            policy_context=policy,
        )
        teacher_tools = self._skills.augment_registry(
            binding.tool_registry, episode.stage
        )
        return (
            TeacherCandidateContext(
                episode=episode,
                rubric=job.rubric,
                tool_registry=teacher_tools,
                turn_mcp_factory=self._turn_mcp,
                skills_factory=self._skills,
                skill_delivery=self._skills.public_metadata(episode.stage),
                stage_tool_guidance=stage_tool_guidance,
            ),
            binding,
        )

    def load(
        self, record: Mapping[str, Any]
    ) -> tuple[TeacherCandidateContext, Any]:
        candidate_id = record.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise TeacherBatchError("teacher candidate identity differs")
        leader = False
        with self._lock:
            cached = self._cache.get(candidate_id)
            if cached is not None:
                return cached
            pending = self._inflight.get(candidate_id)
            if pending is None:
                pending = Future()
                self._inflight[candidate_id] = pending
                leader = True
        if not leader:
            return pending.result()
        try:
            result = self._build(record)
            with self._lock:
                self._cache[candidate_id] = result
                self._build_count += 1
            pending.set_result(result)
            return result
        except BaseException as exc:
            pending.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._inflight.pop(candidate_id, None)


def campaign_v2_candidate_context(
    record: Mapping[str, Any]
) -> tuple[TeacherCandidateContext, Any]:
    """Compatibility cold path; persistent batches use one shared pool."""

    return CampaignV2TeacherContextPool().load(record)


__all__ = [
    "CampaignV2TeacherContextPool",
    "TeacherCandidateContext",
    "bulk_initial_files",
    "campaign_v2_candidate_context",
    "execute_single_rollout",
    "hydrate_teacher_stage_prerequisites",
    "load_bulk_record",
    "teacher_actor_developer_instructions",
    "teacher_safe_actor_instruction",
]
