from __future__ import annotations

import base64
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from eva_agent.codex_pipeline import (
    CODEX_JUDGE_REQUIRED_MATERIAL_PATHS,
    CodexOpus5AgentJudge,
)
from eva_agent.codex_pipeline.adapter import (
    CodexPipelineError,
    _agent_judge_terminal_object,
    _live_judge_trace_matches,
)
from eva_agent.codex_runtime import (
    CodexRole,
    CodexRuntimeError,
    CodexSandbox,
    CodexThreadOptions,
)
from eva_agent.pipeline import (
    JudgeAssessment,
    JudgeRequest,
    RandomUUIDFactory,
    RubricItemScore,
    ToolCall,
    TrajectoryEvent,
)
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes
from eva_agent.pipeline.judge_workspace import JudgeWorkspaceTools
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.training.agent_judge import AgentJudgeSelectionError, AgentJudgeTask
from eva_agent.training.agent_judge_worker import (
    execute_agent_judge_task,
    prepare_agent_judge_task,
    provider_free_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
RUBRICS = ROOT / "rubrics/source/domain-stage-tables.v1.json"
WORKER = ROOT / "scripts/run_one_codex_agent_judge_v1.py"


def _snapshot(label: str, files: dict[str, bytes]) -> dict:
    rows = [
        {
            "path": path,
            "content": {"$bytes_base64": base64.b64encode(payload).decode("ascii")},
            "byte_count": len(payload),
            "mode": "0600",
            "content_blake3": blake3_bytes(payload),
        }
        for path, payload in sorted(files.items())
    ]
    core = {
        "files": rows,
        "file_count": len(rows),
        "byte_count": sum(len(payload) for payload in files.values()),
    }
    return {"label": label, **core, "tree_blake3": blake3_hex(core)}


def _event(role: str, content, tool_call_ids=()) -> dict:
    core = {
        "event_id": str(uuid4()),
        "role": role,
        "content": content,
        "tool_call_ids": list(tool_call_ids),
    }
    return {**core, "event_blake3": blake3_hex(core)}


def _fixture(
    tmp_path: Path,
    *,
    extra_before: dict[str, bytes] | None = None,
    extra_after: dict[str, bytes] | None = None,
):
    registry = load_and_compile_registry(RUBRICS)
    rubric = registry.resolve("medxpertqa", "E2E")
    sandbox_id = str(uuid4())
    candidate_id = str(uuid4())
    route_id = "gpt_5_6_sol"
    task_id = f"{sandbox_id}--{route_id}"
    trace_core = {
        "results": [],
        "declared_call_ids": [],
        "joined_call_ids": [],
        "frontier_count": 0,
        "max_parallelism_observed": 0,
        "retry_count": 0,
    }
    before_files = {"input.txt": b"seed\n", **(extra_before or {})}
    after_files = {
        "input.txt": b"seed\n",
        "result.json": b'{"verified":true}\n',
        **(extra_after or {}),
    }
    value = {
        "schema": "eva.codex-teacher-full-trajectory.v1",
        "task_id": task_id,
        "sandbox_id": sandbox_id,
        "candidate_id": candidate_id,
        "episode_id": "source-episode",
        "executable_episode_id": "executable-episode",
        "route_id": route_id,
        "model_id": "openai/gpt-5.6-sol",
        "provider": "eva_gpt_5_6_sol",
        "score": 0.0,
        "score_kind": "selection_pending",
        "selection_pending": True,
        "messages": [
            _event("system", "Use tools."),
            _event("user", {"question": "Evaluate retained medical evidence."}),
            _event("assistant", "Completed with workspace evidence."),
        ],
        "assistant_output": "Completed with workspace evidence.",
        "tool_trace": {**trace_core, "trace_blake3": blake3_hex(trace_core)},
        "skill_delivery": {},
        "workspace_before": _snapshot("before-rollout", before_files),
        "workspace_after": _snapshot("after-rollout", after_files),
        "rubric_table": rubric.to_document(),
        "provider_receipt_blake3": blake3_hex("provider"),
        "provider_metadata": {},
        "semantic_retry_count": 0,
        "judge_calls": 0,
        "admission_writes": 0,
        "cascade_required": False,
    }
    path = tmp_path / "result.json"
    path.write_bytes(canonical_json_bytes(value))
    task = AgentJudgeTask(
        judge_task_id=str(uuid4()),
        source_task_id=task_id,
        sandbox_id=sandbox_id,
        candidate_id=candidate_id,
        actor_route_id=route_id,
        stage="E2E",
        domain="medxpertqa",
        trajectory_path=str(path),
        judge_route_id="opus_5",
        judge_model_id="aws/anthropic/bedrock-claude-opus-5",
    )
    return registry, path, task


def _trace_event(ids: RandomUUIDFactory, role: str, content, call_ids=()):
    core = {
        "event_id": ids.new("judge-event"),
        "role": role,
        "content": content,
        "tool_call_ids": tuple(call_ids),
    }
    return TrajectoryEvent(**core, event_blake3=blake3_hex(core))


class _ProviderFreeWorkspaceJudge:
    def __init__(self, *, complete: bool = True) -> None:
        self.complete = complete
        self.calls = 0

    def judge(self, request: JudgeRequest, rubric) -> JudgeAssessment:
        self.calls += 1
        ids = RandomUUIDFactory()
        tools = JudgeWorkspaceTools(request.workspace_evidence, id_factory=ids)
        calls = []
        snapshots = (("before", request.workspace_evidence.workspace_before),)
        if self.complete:
            snapshots += (("after", request.workspace_evidence.workspace_after),)
        for name, snapshot in snapshots:
            for row in snapshot.files:
                for offset in range(0, max(1, row.byte_count), 65536):
                    calls.append(
                        ToolCall(
                            call_id=ids.new("judge-read"),
                            name="workspace_read",
                            arguments={
                                "snapshot": name,
                                "path": row.path,
                                "offset": offset,
                                "max_bytes": 65536,
                            },
                        )
                    )
        results = []
        for offset in range(0, len(calls), 64):
            results.extend(tools.execute_group(tuple(calls[offset : offset + 64])))
        events = [
            _trace_event(ids, "system", "Inspect workspace."),
            _trace_event(ids, "user", {"evidence": request.workspace_evidence.bundle_blake3}),
            _trace_event(
                ids,
                "assistant",
                {"tool_calls": len(calls)},
                tuple(call.call_id for call in calls),
            ),
        ]
        events.extend(
            _trace_event(ids, "tool", {"receipt": result.receipt_blake3}, (result.call_id,))
            for result in results
        )
        events.append(_trace_event(ids, "assistant", "Scored exact rubric."))
        trace = tools.trace(
            policy_events=events,
            provider_turn_count=1 + trace_frontiers(tools),
        )
        cited = "workspace:before:input.txt"
        scores = []
        hard_passed = True
        for raw in rubric.items:
            maximum = max(level["score_bps"] for level in raw["partial_credit"]["levels"])
            gate = raw.get("hard_gate")
            hard_passed = hard_passed and not (
                gate is not None and maximum < gate["minimum_score_bps"]
            )
            scores.append(
                RubricItemScore(
                    item_id=raw["item_id"],
                    score=maximum / 10_000,
                    evidence_refs=(cited,),
                    rationale="The cited immutable workspace bytes satisfy this rubric level.",
                )
            )
        core = {
            "judgment_id": request.judgment_id,
            "judge_model_id": request.judge_model_id,
            "rubric_digest": rubric.digest,
            "agent_trace": trace,
            "item_scores": tuple(scores),
            "hard_gates_passed": hard_passed,
            "summary": "All retained evidence sections and workspace bytes were inspected.",
        }
        return JudgeAssessment(**core, assessment_blake3=blake3_hex(core))


class _CaptureCompactTurnRunner:
    def __init__(self) -> None:
        self.turn_input = None
        self.timeout_seconds = None

    def run_once(self, options, turn_input):  # pragma: no cover - bounded path required
        raise AssertionError("unbounded runner path used")

    def run_once_with_timeout(self, options, turn_input, *, timeout_seconds):
        del options
        self.turn_input = turn_input
        self.timeout_seconds = timeout_seconds
        raise CodexRuntimeError("provider-free capture")


def trace_frontiers(tools: JudgeWorkspaceTools) -> int:
    # The test judge intentionally uses the public trace contract to read the
    # current frontier count without reaching into runtime internals.
    ids = RandomUUIDFactory()
    events = (
        _trace_event(ids, "system", "snapshot"),
        _trace_event(ids, "user", {}),
        _trace_event(ids, "assistant", "snapshot"),
        _trace_event(ids, "assistant", "snapshot"),
    )
    return tools.trace(policy_events=events, provider_turn_count=2).frontier_count


def test_real_worker_contract_maps_one_workspace_agent_judgment(tmp_path: Path) -> None:
    registry, _, task = _fixture(tmp_path)
    prepared = prepare_agent_judge_task(task, registry=registry)
    judge = _ProviderFreeWorkspaceJudge()
    result = execute_agent_judge_task(prepared, judge=judge)

    assert judge.calls == 1
    assert result["schema"] == "eva.codex-agent-judge-result.v1"
    assert result["judge_task_id"] == task.judge_task_id
    assert result["judge_tool_trace"]["retry_count"] == 0
    assert result["judge_tool_trace"]["workspace_read_count"] == 8
    assert len(result["item_scores"]) == len(prepared.trajectory.rubric.items) == 6
    assert all(row["evidence_refs"] for row in result["item_scores"])
    assert "agent_trace" not in result and "provider_receipt" not in result


def test_worker_rejects_partial_workspace_inspection(tmp_path: Path) -> None:
    registry, _, task = _fixture(tmp_path)
    prepared = prepare_agent_judge_task(task, registry=registry)
    with pytest.raises(AgentJudgeSelectionError, match="completely read"):
        execute_agent_judge_task(
            prepared, judge=_ProviderFreeWorkspaceJudge(complete=False)
        )


def test_one_task_cli_preflight_is_provider_free(tmp_path: Path) -> None:
    _, trajectory, _ = _fixture(tmp_path)
    result = subprocess.run(
        (
            sys.executable,
            str(WORKER),
            "preflight",
            "--trajectory",
            str(trajectory),
            "--judge-model-id",
            "aws/anthropic/bedrock-claude-opus-5",
        ),
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["provider_calls"] == 0
    assert receipt["judge_attempts"] == 0
    assert receipt["retry_count"] == 0
    assert receipt["rubric_item_count"] == 6
    assert receipt["workspace_file_count"] == 8
    assert receipt["turn_mcp_required"] is True
    assert receipt["large_bytes_embedded_in_initial_turn"] is False
    assert receipt["actor_evidence_snapshot"] == "after"
    assert receipt["required_actor_evidence_paths"] == list(
        CODEX_JUDGE_REQUIRED_MATERIAL_PATHS
    )
    assert receipt["initial_evidence_index_bytes"] < 65536
    assert receipt["complete_before_after_read_required"] is True
    assert receipt["private_reasoning_recorded"] is False
    assert receipt["auto_compact_token_limit"] == 49_152


def test_workspace_over_one_mib_stays_behind_chunked_read_tools(tmp_path: Path) -> None:
    payload = (b"verifiable-medical-evidence\n" * 48_000)[: 1_200_000]
    registry, trajectory, task = _fixture(
        tmp_path,
        extra_before={"large-evidence.txt": payload},
        extra_after={"large-evidence.txt": payload + b"reviewed\n"},
    )
    assert trajectory.stat().st_size > 1_048_576

    prepared = prepare_agent_judge_task(task, registry=registry)
    preflight = provider_free_preflight(prepared)
    schemas = JudgeWorkspaceTools(prepared.evidence).response_api_schemas()
    snapshot_enums = {
        tuple(schema["parameters"]["properties"]["snapshot"]["enum"])
        for schema in schemas
        if "snapshot" in schema["parameters"]["properties"]
    }
    assert snapshot_enums == {("before", "after")}
    assert preflight["workspace_byte_count"] > 2_000_000
    assert preflight["initial_evidence_index_bytes"] < 65536
    assert preflight["large_bytes_embedded_in_initial_turn"] is False

    capture = _CaptureCompactTurnRunner()

    def options(request, offers):
        return CodexThreadOptions(
            role=CodexRole.JUDGE,
            model=request.judge_model_id,
            provider="provider-free",
            cwd=str(tmp_path.resolve()),
            sandbox=CodexSandbox.READ_ONLY,
            offered_tools=offers,
        )

    judge = CodexOpus5AgentJudge(
        options_factory=options,
        model_id=task.judge_model_id,
        runner=capture,
        compact_evidence_index=True,
    )
    with pytest.raises(CodexRuntimeError, match="provider-free capture"):
        execute_agent_judge_task(prepared, judge=judge)
    assert capture.turn_input is not None
    assert len(canonical_json_bytes(capture.turn_input)) < 65_536
    assert "policy_visible_context" not in capture.turn_input.public_context
    projected = capture.turn_input.public_context["workspace_evidence"]
    assert "actor_policy_events" not in projected
    assert "actor_tool_trace" not in projected

    result = execute_agent_judge_task(
        prepared,
        judge=_ProviderFreeWorkspaceJudge(),
    )
    assert result["judge_tool_trace"]["retry_count"] == 0
    assert result["judge_tool_trace"]["workspace_read_count"] > 32


def test_fast_workspace_judge_reads_full_sidecars_with_bounded_source_sampling(
    tmp_path: Path,
) -> None:
    registry, _, task = _fixture(tmp_path)
    prepared = prepare_agent_judge_task(
        task,
        registry=registry,
        fast_workspace_judge=True,
    )
    result = execute_agent_judge_task(
        prepared,
        judge=_ProviderFreeWorkspaceJudge(),
    )

    assert result["schema"] == "eva.codex-agent-judge-result.v1"
    assert result["judge_tool_trace"] == {
        "workspace_read_count": 8,
        "workspace_search_count": 0,
        "provider_turn_count": 2,
        "retry_count": 0,
    }


def test_fast_workspace_judge_forces_short_turn_and_structured_verdict(
    tmp_path: Path,
) -> None:
    registry, _, task = _fixture(tmp_path)
    prepared = prepare_agent_judge_task(
        task,
        registry=registry,
        fast_workspace_judge=True,
    )
    capture = _CaptureCompactTurnRunner()

    def options(request, offers):
        return CodexThreadOptions(
            role=CodexRole.JUDGE,
            model=request.judge_model_id,
            provider="provider-free",
            cwd=str(tmp_path.resolve()),
            sandbox=CodexSandbox.READ_ONLY,
            offered_tools=offers,
        )

    judge = CodexOpus5AgentJudge(
        options_factory=options,
        model_id=task.judge_model_id,
        runner=capture,
        compact_evidence_index=True,
        fast_workspace_judge=True,
        maximum_workspace_tool_calls=64,
        maximum_workspace_tool_frontiers=4,
        turn_timeout_seconds=240,
    )
    with pytest.raises(CodexRuntimeError, match="provider-free capture"):
        execute_agent_judge_task(prepared, judge=judge)

    assert capture.timeout_seconds == 240
    assert capture.turn_input is not None
    assert "no more than 4 parallel tool rounds" in capture.turn_input.public_text
    assert "64 total workspace calls" in capture.turn_input.public_text
    assert "missing submission" in capture.turn_input.public_text
    assert capture.turn_input.output_schema is not None


def test_fast_worker_preflight_publishes_exact_resource_bounds(tmp_path: Path) -> None:
    _, trajectory, _ = _fixture(tmp_path)
    result = subprocess.run(
        (
            sys.executable,
            str(WORKER),
            "preflight",
            "--trajectory",
            str(trajectory),
            "--judge-model-id",
            "aws/anthropic/bedrock-claude-opus-5",
            "--fast-workspace-judge",
        ),
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["provider_calls"] == 0
    assert receipt["fast_workspace_judge"] is True
    assert receipt["auto_compact_token_limit"] == 114_688
    assert receipt["turn_timeout_seconds"] == 240
    assert receipt["maximum_workspace_tool_calls"] == 64
    assert receipt["maximum_workspace_tool_frontiers"] == 4
    assert receipt["upstream_max_output_tokens"] == 16_384


def test_live_judge_trace_accepts_parallel_arrival_order_without_losing_identity() -> None:
    calls = (
        SimpleNamespace(tool_call_id="provider-a"),
        SimpleNamespace(tool_call_id="provider-b"),
    )
    live = {
        "provider-a": SimpleNamespace(call_id="execution-a"),
        "provider-b": SimpleNamespace(call_id="execution-b"),
    }
    results = (
        SimpleNamespace(call_id="execution-b"),
        SimpleNamespace(call_id="execution-a"),
    )
    trace = SimpleNamespace(
        frontier_count=1,
        max_parallelism_observed=2,
        declared_call_ids=("execution-b", "execution-a"),
        joined_call_ids=("execution-b", "execution-a"),
        results=results,
        retry_count=0,
    )

    assert _live_judge_trace_matches(trace, (calls,), live) is True
    trace.declared_call_ids = ("execution-a", "execution-a")
    assert _live_judge_trace_matches(trace, (calls,), live) is False


def test_agent_judge_terminal_accepts_only_exact_or_singly_fenced_json() -> None:
    verdict = {"item_scores": [], "hard_gates_passed": False, "summary": "blocked"}
    encoded = json.dumps(verdict)

    assert _agent_judge_terminal_object(encoded) == verdict
    assert _agent_judge_terminal_object(f"```json\n{encoded}\n```") == verdict
    with pytest.raises(CodexPipelineError, match="not JSON"):
        _agent_judge_terminal_object(f"Verdict:\n```json\n{encoded}\n```")
    with pytest.raises(CodexPipelineError, match="not JSON"):
        _agent_judge_terminal_object(f"```json\n{encoded}\n```\nextra")
