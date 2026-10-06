"""CPU-only controller sequence; never launches workers, providers, or GPUs."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys

import pytest

from training.eva_rsi import next_boundary_activation as boundary
from training.eva_rsi.controller import write
from training.eva_rsi.evidence import EvidenceError, commitment


ACTIVE = "11111111-1111-4111-8111-111111111111"
LOOP = "22222222-2222-4222-8222-222222222222"


class FakeController:
    def __init__(self, root, state, config):
        self.root, self.state, self.config = Path(root), state, config
        self.calls = []

    def load(self):
        return self.state, self.config

    def resume(self):
        self.calls.append(("resume",))
        stop = self.root / "stop-request.json"
        assert stop.is_file()
        stop.rename(self.root / "retained-stop-fixture.json")
        self.state["status"] = "ready"

    def run(self, *, execute, max_transitions=None, poll_seconds=2):
        self.calls.append(("run", execute, max_transitions, poll_seconds))
        if max_transitions == 2:
            self.state.update(status="ready", active_attempt=None,
                              pending_training=None, phase="evaluation")
            self.state["attempts"][0]["status"] = "verified"
            return {"status": "ready", "active_attempt": None}
        return {"status": "complete", "fixture_continuous_run": True}

    def reconfigure(self, path, *, reason, execute):
        self.calls.append(("reconfigure", Path(path), reason, execute))
        self.state["config"] = commitment(path)
        self.config = boundary.read(path)
        return {"schema": "eva.rsi-config-transition.v1", "attempts_launched": 0}


def setup(tmp_path):
    commands = {name: [sys.executable, f"/{name}.py", "{context}"]
                for name in ("baseline_eval", "train", "evaluation")}
    old = {"schema": "eva.rsi-loop-config.v1", "rounds": 10, "updates_per_round": 50,
        "cwd": str(tmp_path), "architecture_model_path": "/architecture",
        "initial_model_path": "/model", "initial_checkpoint_root": "/checkpoint",
        "skill_catalog_id": "catalog", "commands": commands}
    new = deepcopy(old)
    new["commands"]["evaluation"] = [sys.executable, "/new-evaluation.py", "{context}"]
    old_path, new_path = tmp_path / "old-config.json", tmp_path / "new-config.json"
    write(old_path, old); write(new_path, new)
    root = tmp_path / "loop"
    request = root / "attempts" / ACTIVE / "request.json"
    write(request, {"attempt_id": ACTIVE, "phase": "train"}, exclusive=True)
    worker_pid, supervisor_pid = os.getpid(), os.getppid()
    worker_birth, supervisor_birth = "12345", "67890"
    write(request.parent / "worker.json", {"pid": worker_pid, "start_identity": worker_birth,
        "request": commitment(request)}, exclusive=True)
    state = {"loop_id": LOOP, "config": commitment(old_path), "phase": "train",
        "status": "running", "active_attempt": ACTIVE, "pending_training": None,
        "attempts": [{"attempt_id": ACTIVE, "root": str(request.parent),
                      "phase": "train", "status": "running"}]}
    expected = {"loop_id": LOOP, "active_attempt": ACTIVE, "config": commitment(old_path),
        "supervisor_pid": supervisor_pid, "supervisor_birth": supervisor_birth,
        "worker_pid": worker_pid, "worker_birth": worker_birth}
    return FakeController(root, state, old), new_path, expected, request


def bind_fake_processes(monkeypatch, expected, request):
    monkeypatch.setattr(boundary, "start_identity", lambda pid: {
        expected["supervisor_pid"]: expected["supervisor_birth"],
        expected["worker_pid"]: expected["worker_birth"]}[pid])
    monkeypatch.setattr(boundary, "_process_argv", lambda pid: (
        (sys.executable, "/fixture/run_eva_rsi_loop_v1.py", "run", "--run-root",
         str(request.parents[2])) if pid == expected["supervisor_pid"] else
        (sys.executable, "/fixture/run_eva_rsi_loop_v1.py", "_worker", "--request", str(request))))


def test_dry_plan_then_exact_two_transition_activation(tmp_path, monkeypatch):
    controller, new_path, expected, request = setup(tmp_path)
    bind_fake_processes(monkeypatch, expected, request)
    proposed = boundary.activate(controller, new_path, expected)
    assert proposed["mode"] == "dry-run" and proposed["execution_armed"] is False
    assert proposed["baseline_train_commands_unchanged"] is True
    assert not (controller.root / "stop-request.json").exists() and controller.calls == []

    def drained(pid, birth, timeout):
        assert (pid, birth, timeout) == (expected["supervisor_pid"],
                                        expected["supervisor_birth"], 120)
        assert (controller.root / "stop-request.json").is_file()

    result = boundary.activate(controller, new_path, expected, execute=True,
                               wait_for_supervisor_exit=drained)
    assert result == {"status": "complete", "fixture_continuous_run": True}
    assert [call[0] for call in controller.calls] == ["resume", "run", "reconfigure", "run"]
    assert controller.calls[1][2] == 2 and controller.calls[-1][2] is None
    assert (controller.root / "retained-stop-fixture.json").is_file()


def test_stale_active_attempt_refuses_before_stop(tmp_path, monkeypatch):
    controller, new_path, expected, request = setup(tmp_path)
    bind_fake_processes(monkeypatch, expected, request)
    expected = {**expected, "active_attempt": "33333333-3333-4333-8333-333333333333"}
    with pytest.raises(EvidenceError, match="active_train_changed"):
        boundary.activate(controller, new_path, expected, execute=True)
    assert not (controller.root / "stop-request.json").exists() and controller.calls == []


@pytest.mark.parametrize(("change", "accepted"), [
    (None, True),
    ("wrong-root", False),
    ("wrong-module", False),
    ("unexecuted", False),
])
def test_exact_executed_module_supervisor_binding(tmp_path, monkeypatch, change, accepted):
    pid, birth = 918061, "127885203"
    run_root = (tmp_path / "loop").resolve()
    run_root.mkdir()
    module = "training.eva_rsi.next_boundary_activation"
    argv = [sys.executable, "-B", "-m", module, "--execute", "--run-root", str(run_root),
        "--new-config", str(tmp_path / "new.json")]
    if change == "wrong-root":
        argv[argv.index("--run-root") + 1] = str(tmp_path / "other-loop")
    elif change == "wrong-module":
        argv[argv.index("-m") + 1] = "training.eva_rsi.some_other_module"
    elif change == "unexecuted":
        argv.remove("--execute")
    monkeypatch.setattr(boundary, "start_identity", lambda actual: birth if actual == pid else None)
    monkeypatch.setattr(boundary, "_process_argv", lambda actual: tuple(argv) if actual == pid else None)
    if accepted:
        boundary._bound_process(pid, birth, mode="supervisor", run_root=run_root)
    else:
        with pytest.raises(EvidenceError, match="activation_supervisor_argv_differs"):
            boundary._bound_process(pid, birth, mode="supervisor", run_root=run_root)
