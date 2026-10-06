"""Fake-child CPU supervision tests; never invoke the actual GPU worker."""
import json
from pathlib import Path
import signal
import sys

import pytest

from training.benchmark_models import job_supervisor as supervisor

JOB = "d57449c7-620f-43a5-a3c5-3a185c60e2df"


def arguments(tmp_path):
    workspace = tmp_path / "actor"
    workspace.mkdir()
    return ["--workspace", str(workspace), "--audit-root", str(tmp_path / "authoritative"), "--job-id", JOB,
            "--track", "report", "--prompt", "Generate the findings section.", "--max-new-tokens", "64", "--execute-gpu"]


def receipts(tmp_path):
    audit = tmp_path / "authoritative" / JOB / "process-exit.json"
    public = tmp_path / "actor/outputs/agents_outputs/prescribed-model-jobs" / JOB / "process-exit.json"
    assert audit.read_bytes() == public.read_bytes()
    assert audit.stat().st_mode & 0o777 == 0o600
    return json.loads(audit.read_text())


@pytest.mark.parametrize("returncode", [0, 2, -15])
def test_actual_wait_status_and_fixed_unchanged_child_argv(tmp_path, monkeypatch, returncode):
    args = arguments(tmp_path)
    captured = {}
    class Child:
        pid = 12345
        def wait(self):
            captured["waited"] = True
            return returncode
        def send_signal(self, signum): pytest.fail("No signal expected")
    def launch(command, **kwargs):
        captured.update(command=command, options=kwargs)
        return Child()
    monkeypatch.setattr(supervisor.subprocess, "Popen", launch)
    assert supervisor.main(args) == returncode
    assert captured["command"] == [sys.executable, "-B", str(supervisor.WORKER), *args]
    assert "env" not in captured["options"] and "shell" not in captured["options"]
    assert captured["waited"] and captured["options"]["start_new_session"] is True
    row = receipts(tmp_path)
    assert row["child_pid"] == 12345 and row["returncode"] == returncode and row["os_process_exit_observed"] is True
    assert row["worker_receipt_blake3"] is None and row["schema"] == "eva.prescribed-model-process-exit.v1"


def test_sigterm_only_forwards_to_owned_child_then_waits(tmp_path, monkeypatch):
    calls = []
    previous = signal.getsignal(signal.SIGTERM)
    class Child:
        pid = 54321
        def wait(self):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            calls.append("wait-returned")
            return -signal.SIGTERM
        def send_signal(self, signum): calls.append((self.pid, signum))
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: Child())
    assert supervisor.main(arguments(tmp_path)) == -signal.SIGTERM
    assert calls == [(54321, signal.SIGTERM), "wait-returned"]
    assert receipts(tmp_path)["forwarded_signals"] == [signal.SIGTERM]
    assert signal.getsignal(signal.SIGTERM) == previous


def test_spawn_failure_is_not_fabricated_exit(tmp_path, monkeypatch):
    def launch(*a, **k): raise OSError("synthetic non-secret failure")
    monkeypatch.setattr(supervisor.subprocess, "Popen", launch)
    with pytest.raises(OSError): supervisor.main(arguments(tmp_path))
    directory = tmp_path / "authoritative" / JOB
    assert not (directory / "process-exit.json").exists()
    assert json.loads((directory / "supervisor-error.json").read_text())["os_process_exit_observed"] is False


def test_old_attempt_and_ambiguous_binding_never_launch(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: pytest.fail("Must not launch"))
    for tail in (["--workspace", str(tmp_path)], ["--work", str(tmp_path)]):
        with pytest.raises(ValueError): supervisor.main(args + tail)
    (tmp_path / "authoritative" / JOB).mkdir(parents=True)
    with pytest.raises(FileExistsError): supervisor.main(args)


def test_audit_cannot_be_actor_writable(tmp_path):
    args = arguments(tmp_path)
    args[3] = str(tmp_path / "actor/audit")
    with pytest.raises(ValueError, match="outside"):
        supervisor.main(args)


def test_real_cpu_child_exit_is_reaped_and_recorded(tmp_path, monkeypatch):
    # Test-only replacement of the fixed constant, not a production CLI option.
    worker = tmp_path / "cpu_exit_fixture.py"
    worker.write_text("import sys\nsys.exit(7)\n")
    monkeypatch.setattr(supervisor, "WORKER", worker)
    assert supervisor.main(arguments(tmp_path)) == 7
    row = receipts(tmp_path)
    assert row["returncode"] == 7 and row["child_pid"] > 0
    assert row["os_process_exit_observed"] is True
