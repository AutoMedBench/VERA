from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
from uuid import uuid4

import pytest

from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.training import TeacherCheckpoint
from eva_agent.training.agent_judge import (
    AgentJudgeCheckpoint,
    AgentJudgeSelectionError,
    REQUIRED_INSPECTION_SECTIONS,
    completed_teacher_tasks,
    run_agent_judge_batch,
    validate_full_trajectory,
    verify_selection_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
RUBRICS = ROOT / "rubrics/source/domain-stage-tables.v1.json"


def _snapshot(label: str, files: dict[str, bytes]) -> dict:
    rows = [
        {
            "path": path,
            "content": {
                "$bytes_base64": __import__("base64").b64encode(payload).decode("ascii")
            },
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


def _teacher_fixture(tmp_path: Path):
    registry = load_and_compile_registry(RUBRICS)
    rubric = registry.resolve("medxpertqa", "E2E")
    sandbox_id = str(uuid4())
    candidate_id = str(uuid4())
    route_id = "gpt_5_6_sol"
    source_task_id = f"{sandbox_id}--{route_id}"
    teacher_root = tmp_path / "teacher"
    result_path = teacher_root / "trajectories" / sandbox_id / route_id / "result.json"
    result_path.parent.mkdir(parents=True)
    trace_core = {
        "results": [],
        "declared_call_ids": [],
        "joined_call_ids": [],
        "frontier_count": 0,
        "max_parallelism_observed": 0,
        "retry_count": 0,
    }
    trajectory = {
        "schema": "eva.codex-teacher-full-trajectory.v1",
        "task_id": source_task_id,
        "sandbox_id": sandbox_id,
        "candidate_id": candidate_id,
        "episode_id": "fixture-episode",
        "executable_episode_id": "fixture-episode",
        "route_id": route_id,
        "model_id": "openai/gpt-5.6-sol",
        "provider": "openai",
        "score": 0.0,
        "score_kind": "selection_pending",
        "selection_pending": True,
        "messages": [
            _event("system", "Use tools."),
            _event("user", {"question": "Evaluate the evidence."}),
            _event("assistant", "Completed with cited evidence."),
        ],
        "assistant_output": "Completed with cited evidence.",
        "tool_trace": {**trace_core, "trace_blake3": blake3_hex(trace_core)},
        "workspace_before": _snapshot("before-rollout", {"input.txt": b"seed\n"}),
        "workspace_after": _snapshot(
            "after-rollout",
            {"input.txt": b"seed\n", "result.json": b'{"verified":true}\n'},
        ),
        "rubric_table": rubric.to_document(),
        "provider_receipt_blake3": blake3_hex("provider"),
        "provider_metadata": {},
        "semantic_retry_count": 0,
        "judge_calls": 0,
        "admission_writes": 0,
        "cascade_required": False,
    }
    result_path.write_bytes(canonical_json_bytes(trajectory))
    teacher = TeacherCheckpoint(teacher_root / "checkpoint.sqlite3")
    row = {
        "sandbox_id": sandbox_id,
        "candidate_id": candidate_id,
        "stage": "E2E",
        "domain": "medxpertqa",
    }
    teacher.schedule((row,), (route_id,))
    source_task = teacher.queued()[0]
    assert teacher.claim(source_task)
    teacher.finish(source_task, succeeded=True, path=result_path, error=None)
    return registry, teacher_root, result_path, trajectory


def _judge_worker(path: Path, *, score_bps: int = 10000, bad_ref: bool = False) -> Path:
    script = path / "judge_worker.py"
    script.write_text(
        "import json,os\n"
        "from pathlib import Path\n"
        "source=json.loads(Path(os.environ['EVA_AGENT_JUDGE_TRAJECTORY_PATH']).read_text())\n"
        f"ref={'\'workspace:after:absent.txt\'' if bad_ref else '\'workspace:after:result.json\''}\n"
        "rows=[{'item_id':i['item_id'],'score_bps':"
        f"{score_bps},'evidence_refs':[ref],'rationale':'Inspected committed result evidence.'}} "
        "for i in source['rubric_table']['items']]\n"
        "print(json.dumps({'schema':'eva.codex-agent-judge-result.v1',"
        "'judge_task_id':os.environ['EVA_AGENT_JUDGE_TASK_ID'],"
        "'source_task_id':os.environ['EVA_AGENT_JUDGE_SOURCE_TASK_ID'],"
        "'source_trajectory_blake3':os.environ['EVA_AGENT_JUDGE_SOURCE_BLAKE3'],"
        "'judge_model_id':os.environ['EVA_AGENT_JUDGE_MODEL_ID'],"
        f"'inspected_sections':{list(REQUIRED_INSPECTION_SECTIONS)!r},"
        "'inspected_evidence_refs':[ref],"
        "'judge_tool_trace':{'workspace_read_count':1,'workspace_search_count':0,"
        "'provider_turn_count':1,'retry_count':0},'item_scores':rows,"
        "'summary':'Workspace-aware rubric evaluation complete.'}))\n",
        encoding="utf-8",
    )
    return script


def test_async_agent_judge_selects_only_recomputed_high_score(tmp_path: Path) -> None:
    registry, teacher_root, trajectory_path, _ = _teacher_fixture(tmp_path)
    tasks = completed_teacher_tasks(
        teacher_root,
        judge_route_id="opus_5",
        judge_model_id="aws/anthropic/bedrock-claude-opus-5",
    )
    assert len(tasks) == 1
    checkpoint = AgentJudgeCheckpoint(tmp_path / "selection/judge-checkpoint.sqlite3")
    assert checkpoint.schedule(tasks) == 1
    assert checkpoint.schedule(tasks) == 0
    result = run_agent_judge_batch(
        checkpoint=checkpoint,
        output_root=tmp_path / "selection",
        registry=registry,
        worker_command=(sys.executable, str(_judge_worker(tmp_path))),
        workers=8,
        minimum_sft_score_bps=8000,
    )
    assert result["counts"] == {
        "queued": 0,
        "running": 0,
        "succeeded": 1,
        "failed": 0,
        "selected_for_sft": 1,
    }
    selection = (
        tmp_path
        / "selection/selections"
        / tasks[0].sandbox_id
        / tasks[0].actor_route_id
        / "selection.json"
    )
    sft = (
        tmp_path
        / "selection/sft-slices"
        / tasks[0].sandbox_id
        / f"{tasks[0].actor_route_id}.json"
    )
    assert selection.is_file() and sft.is_file()
    assert verify_selection_artifact(
        selection, trajectory_path=trajectory_path, registry=registry
    )
    selected = json.loads(selection.read_text())
    assert selected["rubric_score"]["reward_bps"] == 10000
    assert selected["selected_for_sft"] is True
    sliced = json.loads(sft.read_text())
    assert sliced["messages"] and "workspace_after" not in sliced
    assert sliced["selection_blake3"] == selected["selection_blake3"]


def test_low_score_is_retained_but_not_sliced(tmp_path: Path) -> None:
    registry, teacher_root, _, _ = _teacher_fixture(tmp_path)
    task = completed_teacher_tasks(
        teacher_root, judge_route_id="opus_5", judge_model_id="opus-5-agent-judge"
    )[0]
    checkpoint = AgentJudgeCheckpoint(tmp_path / "selection/judge-checkpoint.sqlite3")
    checkpoint.schedule((task,))
    result = run_agent_judge_batch(
        checkpoint=checkpoint,
        output_root=tmp_path / "selection",
        registry=registry,
        worker_command=(sys.executable, str(_judge_worker(tmp_path, score_bps=0))),
        workers=1,
        minimum_sft_score_bps=8000,
    )
    assert result["counts"]["succeeded"] == 1
    assert result["counts"]["selected_for_sft"] == 0
    assert not (tmp_path / "selection/sft-slices").exists()


def test_uninspected_workspace_citation_fails_once_and_never_requeues(
    tmp_path: Path,
) -> None:
    registry, teacher_root, _, _ = _teacher_fixture(tmp_path)
    task = completed_teacher_tasks(
        teacher_root, judge_route_id="opus_5", judge_model_id="opus-5-agent-judge"
    )[0]
    checkpoint = AgentJudgeCheckpoint(tmp_path / "selection/judge-checkpoint.sqlite3")
    checkpoint.schedule((task,))
    result = run_agent_judge_batch(
        checkpoint=checkpoint,
        output_root=tmp_path / "selection",
        registry=registry,
        worker_command=(sys.executable, str(_judge_worker(tmp_path, bad_ref=True))),
        workers=1,
        minimum_sft_score_bps=8000,
    )
    assert result["counts"]["failed"] == 1
    assert result["counts"]["queued"] == 0
    with sqlite3.connect(checkpoint.path) as db:
        attempt_count, error = db.execute(
            "SELECT attempt_count,error_type FROM judge_tasks"
        ).fetchone()
    assert attempt_count == 1
    assert error == "AgentJudgeSelectionError"


def test_interrupted_attempt_is_terminal_and_trajectory_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    registry, teacher_root, trajectory_path, trajectory = _teacher_fixture(tmp_path)
    task = completed_teacher_tasks(
        teacher_root, judge_route_id="opus_5", judge_model_id="opus-5-agent-judge"
    )[0]
    checkpoint = AgentJudgeCheckpoint(tmp_path / "selection/judge-checkpoint.sqlite3")
    checkpoint.schedule((task,))
    assert checkpoint.claim(task)
    assert checkpoint.fail_interrupted() == 1
    assert checkpoint.counts()["failed"] == 1
    assert checkpoint.queued() == []

    trajectory["workspace_after"]["files"][1]["content"]["$bytes_base64"] = "dGFtcGVy"
    trajectory_path.write_bytes(canonical_json_bytes(trajectory))
    with pytest.raises(AgentJudgeSelectionError, match="workspace commitment"):
        validate_full_trajectory(trajectory_path, registry=registry)


def test_completed_teacher_accepts_root_prefixed_relative_result_path(
    tmp_path: Path,
) -> None:
    _, teacher_root, trajectory_path, _ = _teacher_fixture(tmp_path)
    relative = trajectory_path.relative_to(teacher_root.parent.parent)
    with sqlite3.connect(teacher_root / "checkpoint.sqlite3") as db:
        db.execute("UPDATE tasks SET result_path=?", (str(relative),))

    tasks = completed_teacher_tasks(
        teacher_root,
        judge_route_id="opus_5",
        judge_model_id="aws/anthropic/bedrock-claude-opus-5",
    )
    assert len(tasks) == 1
    assert Path(tasks[0].trajectory_path) == trajectory_path.resolve()
