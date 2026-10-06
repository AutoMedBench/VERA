from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from threading import Barrier, Event, Lock
from time import monotonic
from types import SimpleNamespace

import pytest

from eva_agent.pipeline import (
    AdapterError,
    BenchmarkEpisode,
    BenchmarkSource,
    Cohort,
    DeterministicUUIDFactory,
    ImmutableArtifactStore,
    JudgeRequest,
    ModelTarget,
    OpenAIStyleOpus5Judge,
    OpenAIStyleRolloutAdapter,
    PipelineVerifier,
    SeparationPolicy,
    Stage,
    ToolDefinition,
    ToolRegistry,
    VerifiableDataPipeline,
    WeightedRubricRewarder,
    load_pipeline_result,
    verify_result_document,
)
from eva_agent.rubrics import load_and_compile_registry


ROOT = Path(__file__).resolve().parents[1]


def _actual_compiled_rubric():
    source_path = ROOT / "rubrics/source/domain-stage-tables.v1.json"
    inventory = json.loads(source_path.read_text(encoding="utf-8"))
    assert inventory["schema"] == "eva.rubric-registry.v1"
    assert len(inventory["rubrics"]) == 42
    table = next(
        row for row in inventory["rubrics"]
        if row["domain"] == "medxpertqa" and row["stage"] == "E2E"
    )
    assert len(table["items"]) == 6
    registry = load_and_compile_registry(source_path)
    return registry.resolve(table["domain"], table["stage"])


class _RolloutCompletions:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.calls = 0

    def create(self, **request):
        assert request["model"] == self.model_id
        assert request["parallel_tool_calls"] is True
        self.calls += 1
        if self.calls == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Inspect both independent evidence sources.",
                            "tool_calls": [
                                {
                                    "id": "provider-a",
                                    "function": {"name": "inspect_a", "arguments": '{"path":"seed.txt"}'},
                                },
                                {
                                    "id": "provider-b",
                                    "function": {"name": "inspect_b", "arguments": '{"path":"seed.txt"}'},
                                },
                            ],
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"content": f"verified output from {self.model_id}"}}]}


class _JudgeCompletions:
    def __init__(self, item_ids: list[str]) -> None:
        self.item_ids = item_ids
        self.requests: list[dict] = []
        self._lock = Lock()

    def create(self, **request):
        with self._lock:
            self.requests.append(request)
        payload = json.loads(request["messages"][1]["content"])
        cohort = payload["workspace_evidence"]["model"]["cohort"]
        if request["messages"][-1]["role"] != "tool":
            assert request["tool_choice"] == "required"
            assert request["parallel_tool_calls"] is True
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Inspect the output and search its evidence in parallel.",
                            "tool_calls": [
                                {
                                    "id": "judge-read",
                                    "function": {
                                        "name": "workspace_read",
                                        "arguments": json.dumps(
                                            {
                                                "snapshot": "after",
                                                "path": "seed.txt",
                                                "offset": 0,
                                                "max_bytes": 65536,
                                            }
                                        ),
                                    },
                                },
                                {
                                    "id": "judge-search",
                                    "function": {
                                        "name": "workspace_search",
                                        "arguments": json.dumps(
                                            {
                                                "snapshot": "after",
                                                "query": "medical evidence",
                                                "path_prefix": None,
                                                "case_sensitive": False,
                                                "max_matches": 200,
                                            }
                                        ),
                                    },
                                },
                            ],
                        }
                    }
                ]
            }
        scores = {"weak": 0, "middle": 3, "strong": 6}
        rows = []
        for index, item_id in enumerate(self.item_ids):
            score = 1.0 if index < scores[cohort] else 0.0
            rows.append(
                {
                    "item_id": item_id,
                    "score": score,
                    "evidence_refs": ["workspace:after:seed.txt"],
                    "rationale": "Observable fixture evidence.",
                }
            )
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "item_scores": rows,
                                "hard_gates_passed": True,
                                "summary": f"{cohort} fixture judgment",
                            }
                        )
                    }
                }
            ]
        }


class _PrematureJudgeCompletions:
    def __init__(self, item_ids: list[str]) -> None:
        self.item_ids = item_ids

    def create(self, **request):
        del request
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "item_scores": [
                                    {
                                        "item_id": item_id,
                                        "score": 0.0,
                                        "evidence_refs": ["context:policy-visible"],
                                        "rationale": "Attempted score without workspace inspection.",
                                    }
                                    for item_id in self.item_ids
                                ],
                                "hard_gates_passed": False,
                                "summary": "premature",
                            }
                        )
                    }
                }
            ]
        }


class _UnreadCitationJudgeCompletions(_JudgeCompletions):
    def create(self, **request):
        response = super().create(**request)
        if request["messages"][-1]["role"] == "tool":
            value = json.loads(response["choices"][0]["message"]["content"])
            for row in value["item_scores"]:
                row["evidence_refs"] = ["workspace:after:not-inspected.txt"]
            response["choices"][0]["message"]["content"] = json.dumps(value)
        return response


def _client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _build_pipeline(tmp_path: Path):
    rubric = _actual_compiled_rubric()
    targets = tuple(
        ModelTarget(cohort, f"fixture-{cohort.value}", "fake-openai-sdk")
        for cohort in (Cohort.WEAK, Cohort.MIDDLE, Cohort.STRONG)
    )
    ids = DeterministicUUIDFactory("pipeline-e2e")
    providers = {}
    rollout_clients = {}
    for target in targets:
        completions = _RolloutCompletions(target.model_id)
        rollout_clients[target.cohort] = completions
        providers[target.cohort] = OpenAIStyleRolloutAdapter(
            _client(completions), id_factory=ids
        )

    tool_barrier = Barrier(6)

    def inspect(workspace, arguments):
        # Three model workers each dispatch two independent reads.  A barrier
        # proves real overlap without relying on scheduler-sensitive sleeps.
        tool_barrier.wait(timeout=2)
        return {"content": workspace.read_bytes(arguments["path"]).decode("utf-8")}

    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    registry = ToolRegistry(
        [
            ToolDefinition("inspect_a", "Inspect source A.", schema, inspect, parallel_safe=True, read_only=True),
            ToolDefinition("inspect_b", "Inspect source B.", schema, inspect, parallel_safe=True, read_only=True),
        ]
    )
    judge_completions = _JudgeCompletions([item["item_id"] for item in rubric.items])
    artifacts = ImmutableArtifactStore(tmp_path / "artifacts")
    pipeline = VerifiableDataPipeline(
        workspace_root=tmp_path / "workspaces",
        artifact_store=artifacts,
        tool_registry=registry,
        rollout_provider=providers,
        judge=OpenAIStyleOpus5Judge(_client(judge_completions), model_id="claude-opus-5"),
        rewarder=WeightedRubricRewarder(),
        id_factory=ids,
        maximum_parallel_models=3,
        maximum_parallel_tools=8,
    )
    episode = BenchmarkEpisode(
        episode_id="medxpertqa-fixture-1",
        source=BenchmarkSource(
            benchmark="MedXpertQA",
            source_file="benchmarks/medxpertqa/fixture.json",
            source_revision="fixture-v1",
        ),
        domain="medxpertqa",
        stage=Stage.E2E,
        instruction="Research and verify the medical question using tools.",
        policy_context={"question": "What evidence supports the fixture conclusion?"},
        initial_files={"seed.txt": b"public medical evidence\n"},
        judge_only_reference={"reference_answer": "private fixture answer"},
    )
    return pipeline, episode, rubric, targets, artifacts, rollout_clients, judge_completions


def test_full_fake_provider_pipeline_reopens_and_verifies(tmp_path: Path) -> None:
    pipeline, episode, rubric, targets, artifacts, clients, judge = _build_pipeline(tmp_path)
    result = pipeline.run(episode=episode, rubric=rubric, targets=targets)

    assert len(result.sandbox_manifests) == 1
    assert [row.cohort for row in result.evaluations] == [
        Cohort.WEAK, Cohort.MIDDLE, Cohort.STRONG
    ]
    assert [row.reward.total_reward for row in result.evaluations] == [0.0, 0.5, 1.0]
    assert result.separation.policy_passed is True
    assert result.separation.perfect_monotonic_staircase_required is False
    assert result.recommendation.recommendation == "eligible_for_signed_supervisor_admission"
    assert result.recommendation.admission_authorized is False
    assert result.recommendation.signed_decision_created is False
    assert all(client.calls == 2 for client in clients.values())
    assert len(judge.requests) == 6
    for request in judge.requests:
        payload = json.loads(request["messages"][1]["content"])
        assert "policy_visible_context" in payload
        assert "workspace_evidence" in payload
        assert "workspace_before" in payload["workspace_evidence"]
        assert "workspace_after" in payload["workspace_evidence"]
        assert "judge_only_reference" in payload
        compiled_item = payload["compiled_rubric"]["items"][0]
        assert "evidence_selectors" in compiled_item
        assert "partial_credit" in compiled_item
        assert "provenance" in compiled_item
        assert "labels" in compiled_item
        assert request["parallel_tool_calls"] is True
        assert all(tool["function"]["strict"] is True for tool in request["tools"])
        # Snapshot bytes are available only through judge tools, not eagerly
        # duplicated in either workspace manifest. Actor tool observations are
        # still retained separately as part of the trajectory evidence.
        assert "content" not in payload["workspace_evidence"]["workspace_before"]["files"][0]
        assert "content" not in payload["workspace_evidence"]["workspace_after"]["files"][0]
    for evaluation in result.evaluations:
        events = evaluation.evidence.policy_events
        grouped = next(event for event in events if len(event.tool_call_ids) == 2)
        assert grouped.role == "assistant"
        assert evaluation.evidence.tool_trace.retry_count == 0
        assert evaluation.evidence.tool_trace.max_parallelism_observed == 2
        assert evaluation.evidence.workspace_before.tree_blake3 == evaluation.evidence.workspace_after.tree_blake3
        judge_trace = evaluation.judgment.agent_trace
        assert judge_trace.content_inspection_count == 2
        assert judge_trace.max_parallelism_observed == 2
        assert judge_trace.provider_turn_count == 2
        assert judge_trace.retry_count == 0
        assert judge_trace.inspected_evidence_refs == ("workspace:after:seed.txt",)

    report = PipelineVerifier(artifacts).verify_or_raise(result=result, rubric=rubric)
    assert report.valid is True
    result_path = artifacts.root / result.run_id / "pipeline-result.json"
    reopened = load_pipeline_result(result_path)
    assert reopened.result_blake3 == result.result_blake3
    assert verify_result_document(result_path, artifact_store=artifacts, rubric=rubric).valid is True

    tampered = replace(result, result_blake3="0" * 64)
    assert PipelineVerifier(artifacts).verify(result=tampered, rubric=rubric).valid is False

    object.__setattr__(result.evaluations[0].judgment.agent_trace, "trace_blake3", "0" * 64)
    assert PipelineVerifier(artifacts).verify(result=result, rubric=rubric).valid is False


def test_one_valid_workspace_judged_trajectory_is_admission_eligible(
    tmp_path: Path,
) -> None:
    pipeline, episode, rubric, targets, artifacts, _clients, _judge = _build_pipeline(
        tmp_path
    )

    class FailedCohort:
        def run(self, request, runtime):
            del request, runtime
            raise RuntimeError("fixture cohort unavailable")

    def inspect(workspace, arguments):
        return {"content": workspace.read_bytes(arguments["path"]).decode("utf-8")}

    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    pipeline._tools = ToolRegistry(
        [
            ToolDefinition("inspect_a", "Inspect source A.", schema, inspect, parallel_safe=True, read_only=True),
            ToolDefinition("inspect_b", "Inspect source B.", schema, inspect, parallel_safe=True, read_only=True),
        ]
    )
    pipeline._providers = {
        Cohort.WEAK: FailedCohort(),
        Cohort.MIDDLE: FailedCohort(),
        Cohort.STRONG: pipeline._providers[Cohort.STRONG],
    }

    result = pipeline.run(episode=episode, rubric=rubric, targets=targets)

    assert tuple(row.cohort for row in result.evaluations) == (Cohort.STRONG,)
    assert result.separation.policy_passed is False
    assert result.separation.admission_policy_schema == "eva.trajectory-admission-policy.v2"
    assert result.separation.admission_policy_passed is True
    assert result.separation.valid_judged_trajectory_count == 1
    assert result.separation.failure_types_by_cohort == {
        "middle": "RuntimeError",
        "weak": "RuntimeError",
    }
    assert result.recommendation.recommendation == "eligible_for_signed_supervisor_admission"
    assert (artifacts.root / result.run_id / "partial-cohort-failures.json").is_file()
    PipelineVerifier(artifacts).verify_or_raise(result=result, rubric=rubric)
    reopened = load_pipeline_result(artifacts.root / result.run_id / "pipeline-result.json")
    assert reopened.result_blake3 == result.result_blake3


def test_early_valid_evaluation_does_not_wait_for_slow_cohorts(
    tmp_path: Path,
) -> None:
    pipeline, episode, rubric, targets, artifacts, _clients, _judge = _build_pipeline(
        tmp_path
    )
    pipeline._policy = SeparationPolicy(
        early_continuation_after_first_valid_trajectory=True
    )

    release = Event()
    slow_started = {cohort: Event() for cohort in (Cohort.WEAK, Cohort.MIDDLE)}
    slow_finished = {cohort: Event() for cohort in (Cohort.WEAK, Cohort.MIDDLE)}

    class SlowCohort:
        def __init__(self, cohort: Cohort) -> None:
            self._cohort = cohort

        def run(self, request, runtime):
            del request, runtime
            slow_started[self._cohort].set()
            release.wait(timeout=5)
            slow_finished[self._cohort].set()
            raise RuntimeError("slow fixture released")

    def inspect(workspace, arguments):
        return {"content": workspace.read_bytes(arguments["path"]).decode("utf-8")}

    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    pipeline._tools = ToolRegistry(
        [
            ToolDefinition(
                "inspect_a", "Inspect A.", schema, inspect,
                parallel_safe=True, read_only=True,
            ),
            ToolDefinition(
                "inspect_b", "Inspect B.", schema, inspect,
                parallel_safe=True, read_only=True,
            ),
        ]
    )
    pipeline._providers = {
        Cohort.WEAK: SlowCohort(Cohort.WEAK),
        Cohort.MIDDLE: SlowCohort(Cohort.MIDDLE),
        Cohort.STRONG: pipeline._providers[Cohort.STRONG],
    }

    try:
        started = monotonic()
        result = pipeline.run(episode=episode, rubric=rubric, targets=targets)
        elapsed = monotonic() - started
        assert all(event.is_set() for event in slow_started.values())
        assert not any(event.is_set() for event in slow_finished.values())
        assert elapsed < 2
        assert tuple(row.cohort for row in result.evaluations) == (Cohort.STRONG,)
        assert result.separation.admission_policy_passed is True
        assert result.separation.cohort_outcomes == {
            "weak": "cancelled_after_valid_evaluation",
            "middle": "cancelled_after_valid_evaluation",
            "strong": "completed",
        }
        assert result.separation.failure_types_by_cohort == {
            "middle": "_EarlyCohortCancellation",
            "weak": "_EarlyCohortCancellation",
        }
        partial_path = artifacts.root / result.run_id / "partial-cohort-failures.json"
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        assert partial["cancelled_cohorts"] == ["weak", "middle"]
        assert partial["retry_count"] == 0
        assert partial["absent_cohort_semantic_scores_created"] is False
        PipelineVerifier(
            artifacts, separation_policy=pipeline._policy
        ).verify_or_raise(result=result, rubric=rubric)
    finally:
        release.set()
        assert all(event.wait(timeout=2) for event in slow_finished.values())


def test_opus_agent_judge_rejects_one_shot_score_without_workspace_inspection(
    tmp_path: Path,
) -> None:
    pipeline, episode, rubric, targets, _artifacts, _clients, _judge = _build_pipeline(tmp_path)
    result = pipeline.run(episode=episode, rubric=rubric, targets=targets)
    evidence = result.evaluations[0].evidence
    premature = OpenAIStyleOpus5Judge(
        _client(_PrematureJudgeCompletions([item["item_id"] for item in rubric.items])),
        model_id="claude-opus-5",
        id_factory=DeterministicUUIDFactory("premature-judge"),
    )
    ids = DeterministicUUIDFactory("premature-request")
    with pytest.raises(AdapterError, match="without inspecting the workspace"):
        premature.judge(
            JudgeRequest(
                judgment_id=ids.new("judgment"),
                judge_model_id="claude-opus-5",
                policy_visible_context=evidence.policy_visible_context,
                workspace_evidence=evidence,
                judge_only_reference={"reference_answer": "private fixture answer"},
            ),
            rubric,
        )


def test_opus_agent_judge_rejects_uninspected_workspace_citation(tmp_path: Path) -> None:
    pipeline, episode, rubric, targets, _artifacts, _clients, _judge = _build_pipeline(tmp_path)
    result = pipeline.run(episode=episode, rubric=rubric, targets=targets)
    evidence = result.evaluations[0].evidence
    bad_citations = OpenAIStyleOpus5Judge(
        _client(_UnreadCitationJudgeCompletions([item["item_id"] for item in rubric.items])),
        model_id="claude-opus-5",
        id_factory=DeterministicUUIDFactory("unread-citation-judge"),
    )
    ids = DeterministicUUIDFactory("unread-citation-request")
    with pytest.raises(AdapterError, match="did not inspect"):
        bad_citations.judge(
            JudgeRequest(
                judgment_id=ids.new("judgment"),
                judge_model_id="claude-opus-5",
                policy_visible_context=evidence.policy_visible_context,
                workspace_evidence=evidence,
                judge_only_reference={"reference_answer": "private fixture answer"},
            ),
            rubric,
        )
