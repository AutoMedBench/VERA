from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from eva_agent.training import TeacherCheckpoint


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_codex_teacher_batch_v1.py"
SPEC = importlib.util.spec_from_file_location("run_codex_teacher_batch_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)


def _row(index: int) -> dict[str, str]:
    return {
        "sandbox_id": f"sandbox-{index:04d}",
        "candidate_id": f"candidate-{index:04d}",
        "stage": "S1",
        "domain": "medical-research",
    }


def test_scattered_attempts_are_excluded_per_sandbox_and_route(tmp_path: Path) -> None:
    rows = [_row(1), _row(2)]
    first = TeacherCheckpoint(tmp_path / "first.sqlite3")
    first.schedule(rows, ("opus_5", "opus_4_8"))
    opus5_first = next(task for task in first.queued() if task.route_id == "opus_5")
    assert first.claim(opus5_first)

    second = TeacherCheckpoint(tmp_path / "second.sqlite3")
    second.schedule((rows[1],), ("opus_4_8",))
    assert second.claim(second.queued()[0])

    attempted = BATCH._load_attempted_pairs((first.path, second.path))
    assert attempted == {
        ("sandbox-0001", "opus_5"),
        ("sandbox-0002", "opus_4_8"),
    }
    selected = BATCH._select_unused_task_pairs(
        rows, ("opus_5", "opus_4_8"), attempted
    )
    assert [(row["sandbox_id"], route) for row, route in selected] == [
        ("sandbox-0001", "opus_4_8"),
        ("sandbox-0002", "opus_5"),
    ]

    output = TeacherCheckpoint(tmp_path / "output.sqlite3")
    assert BATCH._schedule_task_pairs(output, selected) == 2
    assert BATCH._schedule_task_pairs(output, selected) == 0


def test_duplicate_exclusion_checkpoint_fails_closed(tmp_path: Path) -> None:
    checkpoint = TeacherCheckpoint(tmp_path / "checkpoint.sqlite3")
    with pytest.raises(SystemExit, match="checkpoint is duplicated"):
        BATCH._load_attempted_pairs((checkpoint.path, checkpoint.path))
