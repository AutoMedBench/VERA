import asyncio
import json
from pathlib import Path

from training.automedbench_lite.job_wait import await_model_jobs


def test_wait_uses_actual_exit_and_never_resubmits(tmp_path):
    job = tmp_path / "model-jobs/job"
    job.mkdir(parents=True)
    (job / "submission.json").write_text('{}')
    target = tmp_path / "model-jobs/authoritative/job"
    target.mkdir(parents=True)
    (target / "process-exit.json").write_text(json.dumps({"schema": "eva.prescribed-model-process-exit.v1",
        "job_id": "job", "returncode": 1, "os_process_exit_observed": True}))
    asyncio.run(await_model_jobs(tmp_path, "03-smoke"))
    result = json.loads((tmp_path / "waiting-before-03-smoke.json").read_text())
    assert result["terminal_jobs"] == [{"job_id": "job", "returncode": 1}]
    assert result["pending_jobs"] == [] and result["automatic_resubmissions"] == 0


def test_no_jobs_does_not_create_or_delay_work(tmp_path):
    asyncio.run(await_model_jobs(tmp_path, "03-smoke"))
    assert not list(tmp_path.iterdir())
