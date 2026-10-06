from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from threading import Barrier, Lock
import time

from eva_agent.codex_providers import ROUTE_DEFINITIONS
from eva_agent.codex_pipeline import CodexPipelineError
from eva_agent.training import TEACHER_ROUTES, TeacherCheckpoint, run_batch
from eva_agent.training.teacher_batch import run_persistent_batch


def _rows() -> list[dict[str, str]]:
    return [
        {
            "sandbox_id": "sandbox-0001",
            "candidate_id": "candidate-0001",
            "stage": "S3",
            "domain": "clinical-research",
        }
    ]


def test_checkpoint_is_one_rollout_per_model_and_interruption_is_not_retried(
    tmp_path: Path,
) -> None:
    checkpoint = TeacherCheckpoint(tmp_path / "checkpoint.sqlite3")
    assert checkpoint.schedule(_rows(), ("opus_5", "gpt_5_6_sol")) == 2
    assert checkpoint.schedule(_rows(), ("opus_5", "gpt_5_6_sol")) == 0
    task = checkpoint.queued(limit=1)[0]
    assert checkpoint.claim(task)
    assert checkpoint.fail_interrupted() == 1
    assert checkpoint.counts() == {
        "queued": 1,
        "running": 0,
        "succeeded": 0,
        "failed": 1,
    }


def test_every_optional_provider_route_is_scheduled_once_per_sandbox(
    tmp_path: Path,
) -> None:
    assert len(TEACHER_ROUTES) == len(ROUTE_DEFINITIONS)
    assert set(TEACHER_ROUTES) == set(ROUTE_DEFINITIONS)
    checkpoint = TeacherCheckpoint(tmp_path / "checkpoint.sqlite3")
    assert checkpoint.schedule(_rows(), TEACHER_ROUTES) == len(TEACHER_ROUTES)
    assert checkpoint.schedule(_rows(), TEACHER_ROUTES) == 0
    tasks = checkpoint.queued()
    assert len(tasks) == len(TEACHER_ROUTES)
    assert {task.route_id for task in tasks} == set(TEACHER_ROUTES)
    assert all(task.task_id == f"sandbox-0001--{task.route_id}" for task in tasks)


def test_checkpoint_supports_256_local_workers_without_sqlite_lock_loss(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "sandbox_id": f"sandbox-{index:04d}",
            "candidate_id": f"candidate-{index:04d}",
            "stage": "S2",
            "domain": "clinical-research",
        }
        for index in range(512)
    ]
    checkpoint = TeacherCheckpoint(tmp_path / "checkpoint.sqlite3")
    assert checkpoint.schedule(rows, ("qwen_3_5_122b_a10b",)) == len(rows)

    def complete(task) -> None:
        assert checkpoint.claim(task)
        checkpoint.finish(task, succeeded=True, path=None, error=None)

    with ThreadPoolExecutor(max_workers=256) as executor:
        tuple(executor.map(complete, checkpoint.queued()))

    assert checkpoint.counts() == {
        "queued": 0,
        "running": 0,
        "succeeded": 512,
        "failed": 0,
    }


def test_successful_full_receipt_is_saved_and_high_score_is_sliced(
    tmp_path: Path,
) -> None:
    checkpoint = TeacherCheckpoint(tmp_path / "checkpoint.sqlite3")
    checkpoint.schedule(_rows(), ("glm_5_1",))
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json\n"
        "print(json.dumps({'score':.9,'messages':[{'role':'assistant','content':'done'}],"
        "'tool_trace':[{'name':'read','result':'ok'}],'workspace_before':{},"
        "'workspace_after':{'answer.txt':'done'}}))\n",
        encoding="utf-8",
    )
    result = run_batch(
        checkpoint=checkpoint,
        output_root=tmp_path / "out",
        worker_command=(sys.executable, str(worker)),
        workers=4,
        minimum_sft_score=0.8,
        bulk_root=tmp_path,
    )
    assert result["counts"]["succeeded"] == 1
    receipt = tmp_path / "out/trajectories/sandbox-0001/glm_5_1/result.json"
    sliced = tmp_path / "out/sft-slices/sandbox-0001/glm_5_1.json"
    assert receipt.is_file() and sliced.is_file()
    assert json.loads(sliced.read_text())["score"] == 0.9


def test_incomplete_worker_receipt_fails_closed_once(tmp_path: Path) -> None:
    checkpoint = TeacherCheckpoint(tmp_path / "checkpoint.sqlite3")
    checkpoint.schedule(_rows(), ("gemini_3_1_pro",))
    result = run_batch(
        checkpoint=checkpoint,
        output_root=tmp_path / "out",
        worker_command=(sys.executable, "-c", "print('{}')"),
        workers=1,
        minimum_sft_score=0.8,
        bulk_root=tmp_path,
    )
    assert result["counts"]["failed"] == 1
    assert result["counts"]["queued"] == 0


def test_worker_failure_retains_bounded_stderr_for_diagnosis(tmp_path: Path) -> None:
    checkpoint = TeacherCheckpoint(tmp_path / "checkpoint.sqlite3")
    checkpoint.schedule(_rows(), ("qwen_3_6_27b",))
    result = run_batch(
        checkpoint=checkpoint,
        output_root=tmp_path / "out",
        worker_command=(
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('route protocol mismatch'); raise SystemExit(7)",
        ),
        workers=1,
        minimum_sft_score=0.8,
        bulk_root=tmp_path,
    )
    assert result["counts"]["failed"] == 1
    errors = list((tmp_path / "out/errors").glob("*.stderr.txt"))
    assert len(errors) == 1
    assert errors[0].read_text(encoding="utf-8") == "route protocol mismatch"


def test_persistent_worker_runs_concurrently_once_and_keeps_results_isolated(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "sandbox_id": f"sandbox-{index:04d}",
            "candidate_id": f"candidate-{index:04d}",
            "stage": "S3",
            "domain": "clinical-research",
        }
        for index in range(4)
    ]
    checkpoint = TeacherCheckpoint(tmp_path / "checkpoint.sqlite3")
    assert checkpoint.schedule(rows, ("qwen_3_5_397b_a17b",)) == 4
    calls: list[str] = []
    active = 0
    maximum = 0
    lock = Lock()

    def worker(task):
        nonlocal active, maximum
        with lock:
            calls.append(task.task_id)
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.03)
            return {
                "task_id": task.task_id,
                "score": 0.9,
                "messages": [{"role": "assistant", "content": task.sandbox_id}],
                "tool_trace": [{"name": "read", "result": task.candidate_id}],
                "workspace_before": {"sandbox_id": task.sandbox_id},
                "workspace_after": {"sandbox_id": task.sandbox_id},
            }
        finally:
            with lock:
                active -= 1

    result = run_persistent_batch(
        checkpoint=checkpoint,
        output_root=tmp_path / "out",
        worker=worker,
        workers=4,
        minimum_sft_score=0.8,
    )
    assert result["counts"] == {
        "queued": 0,
        "running": 0,
        "succeeded": 4,
        "failed": 0,
    }
    assert maximum == 4
    assert len(calls) == len(set(calls)) == 4
    for row in rows:
        result_path = (
            tmp_path
            / "out/trajectories"
            / row["sandbox_id"]
            / "qwen_3_5_397b_a17b/result.json"
        )
        receipt = json.loads(result_path.read_text())
        assert receipt["workspace_after"]["sandbox_id"] == row["sandbox_id"]

    second = run_persistent_batch(
        checkpoint=TeacherCheckpoint(tmp_path / "checkpoint.sqlite3"),
        output_root=tmp_path / "out",
        worker=worker,
        workers=4,
        minimum_sft_score=0.8,
    )
    assert second["counts"]["succeeded"] == 4
    assert len(calls) == 4


def test_persistent_worker_exception_is_terminal_and_not_retried(tmp_path: Path) -> None:
    checkpoint = TeacherCheckpoint(tmp_path / "checkpoint.sqlite3")
    checkpoint.schedule(_rows(), ("opus_5",))
    calls = 0

    def worker(_task):
        nonlocal calls
        calls += 1
        raise RuntimeError("fixture failure")

    first = run_persistent_batch(
        checkpoint=checkpoint,
        output_root=tmp_path / "out",
        worker=worker,
        workers=8,
        minimum_sft_score=0.8,
    )
    second = run_persistent_batch(
        checkpoint=checkpoint,
        output_root=tmp_path / "out",
        worker=worker,
        workers=8,
        minimum_sft_score=0.8,
    )
    assert first["counts"]["failed"] == second["counts"]["failed"] == 1
    assert calls == 1


def test_persistent_codex_failure_retains_only_safe_diagnostic(
    tmp_path: Path,
) -> None:
    checkpoint = TeacherCheckpoint(tmp_path / "checkpoint.sqlite3")
    checkpoint.schedule(_rows(), ("qwen_3_5_397b_a17b",))

    def worker(_task):
        raise CodexPipelineError(
            "Codex turn status is not completed: sk-private-upstream-detail"
        )

    result = run_persistent_batch(
        checkpoint=checkpoint,
        output_root=tmp_path / "out",
        worker=worker,
        workers=1,
        minimum_sft_score=0.8,
    )
    assert result["counts"]["failed"] == 1
    paths = list((tmp_path / "out/errors").glob("*.provider-failure.json"))
    assert len(paths) == 1
    raw = paths[0].read_text(encoding="utf-8")
    assert "sk-private-upstream-detail" not in raw
    document = json.loads(raw)
    assert document["failure_message"] == "Codex turn status is not completed"
    assert document["provider_turn_receipt"] is None
    assert document["raw_exception_message_recorded"] is False
    assert document["raw_provider_material_recorded"] is False
    assert document["retry_count"] == 0


def test_two_persistent_launchers_share_one_atomic_claim(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.sqlite3"
    TeacherCheckpoint(path).schedule(_rows(), ("gpt_5_6_sol",))
    barrier = Barrier(2)

    class SynchronizedCheckpoint(TeacherCheckpoint):
        def queued(self, *, limit=None):
            tasks = super().queued(limit=limit)
            barrier.wait(timeout=2)
            return tasks

    calls = 0
    lock = Lock()

    def worker(_task):
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.03)
        return {
            "score": 0.0,
            "messages": [],
            "tool_trace": [],
            "workspace_before": {},
            "workspace_after": {},
        }

    def launch(_index):
        return run_persistent_batch(
            checkpoint=SynchronizedCheckpoint(path),
            output_root=tmp_path / "out",
            worker=worker,
            workers=1,
            minimum_sft_score=0.8,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        tuple(executor.map(launch, range(2)))

    assert calls == 1
    assert TeacherCheckpoint(path).counts() == {
        "queued": 0,
        "running": 0,
        "succeeded": 1,
        "failed": 0,
    }
