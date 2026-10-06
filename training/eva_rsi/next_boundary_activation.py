"""Operator-owned, next-boundary-only RSI config activation.

The default is a read-only plan.  Explicit execution asks the old supervisor to
stop scheduling without signaling its detached worker, waits for that exact
supervisor birth to exit, reconciles exactly the active train plus its durable
checkpoint, reconfigures only at the resulting ready boundary, then continues
the same controller process indefinitely.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from uuid import UUID

from .controller import CONFIG_IDENTITIES, Controller, start_identity, validate_config, write
from .evidence import EvidenceError, commitment, read, require


def _uuid(value, code):
    try:
        require(str(UUID(value)) == value, code)
    except (TypeError, ValueError):
        raise EvidenceError(code) from None
    return value


def _process_argv(pid):
    try:
        payload = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    require(len(payload) <= 128 * 1024, "activation_process_argv_too_large")
    return tuple(part.decode("utf-8") for part in payload.rstrip(b"\0").split(b"\0"))


def _bound_process(pid, birth, *, mode, run_root, request_path=None):
    require(type(pid) is int and pid > 1 and isinstance(birth, str) and birth.isdigit(),
            "activation_process_identity_invalid")
    require(start_identity(pid) == birth, f"activation_{mode}_not_live")
    argv = _process_argv(pid)
    require(argv is not None, f"activation_{mode}_argv_differs")
    standalone = any(Path(value).name == "run_eva_rsi_loop_v1.py" for value in argv)
    if mode == "supervisor":
        module = "training.eva_rsi.next_boundary_activation"
        module_form = (bool(argv) and re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(argv[0]).name)
            and argv.count("-m") == 1 and argv.index("-m") + 1 < len(argv)
            and argv[argv.index("-m") + 1] == module
            and argv.count("--execute") == 1 and argv.count("--run-root") == 1)
        require((standalone and "run" in argv and "--run-root" in argv) or module_form,
                "activation_supervisor_argv_differs")
        index = argv.index("--run-root")
        require(index + 1 < len(argv) and Path(argv[index + 1]).resolve() == run_root,
                "activation_supervisor_argv_differs")
    else:
        require(standalone and "_worker" in argv and "--request" in argv and request_path is not None,
                "activation_worker_argv_differs")
        index = argv.index("--request")
        require(index + 1 < len(argv) and Path(argv[index + 1]).resolve() == request_path,
                "activation_worker_argv_differs")


def _validate_new_config(old_config, new_config_path):
    new_config_path = Path(new_config_path)
    require(new_config_path.is_absolute() and not new_config_path.is_symlink(),
            "activation_new_config_topology")
    new_config_path = new_config_path.resolve(strict=True)
    require(new_config_path.is_file(), "activation_new_config_topology")
    new_config = read(new_config_path)
    validate_config(new_config)
    require(set(new_config) == set(old_config), "activation_new_config_shape_changed")
    require(all(new_config.get(key) == old_config.get(key) for key in CONFIG_IDENTITIES),
            "activation_initial_identity_changed")
    require(all(new_config[key] == old_config[key] for key in old_config
                if key not in {"commands", "cwd"}), "activation_noncommand_setting_changed")
    require(new_config["commands"]["baseline_eval"] == old_config["commands"]["baseline_eval"]
            and new_config["commands"]["train"] == old_config["commands"]["train"],
            "activation_baseline_or_train_command_changed")
    require(new_config != old_config, "activation_new_config_is_unchanged")
    return new_config_path, new_config


def _active_binding(controller, expected, *, require_live_worker):
    state, config = controller.load()
    require(state.get("loop_id") == expected["loop_id"], "activation_loop_identity_changed")
    require(state.get("config") == expected["config"], "activation_config_reference_changed")
    require(state.get("phase") == "train" and state.get("status") == "running"
            and state.get("active_attempt") == expected["active_attempt"]
            and state.get("pending_training") is None, "activation_active_train_changed")
    matches = [row for row in state.get("attempts", [])
               if row.get("attempt_id") == expected["active_attempt"]]
    require(len(matches) == 1 and matches[0].get("phase") == "train"
            and matches[0].get("status") == "running", "activation_attempt_ledger_changed")
    attempt_root = controller.root / "attempts" / expected["active_attempt"]
    require(Path(matches[0].get("root", "")) == attempt_root and not attempt_root.is_symlink()
            and attempt_root.resolve(strict=True) == attempt_root,
            "activation_attempt_root_changed")
    request_path = attempt_root / "request.json"
    request, owner = read(request_path), read(attempt_root / "worker.json")
    require(request.get("attempt_id") == expected["active_attempt"] and request.get("phase") == "train"
            and owner.get("request") == commitment(request_path)
            and owner.get("pid") == expected["worker_pid"]
            and owner.get("start_identity") == expected["worker_birth"],
            "activation_worker_binding_changed")
    if require_live_worker:
        _bound_process(expected["worker_pid"], expected["worker_birth"], mode="worker",
                       run_root=controller.root, request_path=request_path)
    else:
        observed = start_identity(expected["worker_pid"])
        require(observed == expected["worker_birth"] or (
            observed != expected["worker_birth"] and (attempt_root / "exit.json").is_file()),
            "activation_worker_exit_unbound")
    return state, config


def plan(controller, new_config_path, expected):
    state, old_config = _active_binding(controller, expected, require_live_worker=True)
    _bound_process(expected["supervisor_pid"], expected["supervisor_birth"],
                   mode="supervisor", run_root=controller.root)
    require(not (controller.root / "stop-request.json").exists(), "activation_stop_request_already_exists")
    selected, _ = _validate_new_config(old_config, new_config_path)
    return {"schema": "eva.rsi-next-boundary-activation-plan.v1", "mode": "dry-run",
        "loop_id": state["loop_id"], "active_attempt": state["active_attempt"],
        "current_config": state["config"], "new_config": commitment(selected),
        "old_supervisor": {"pid": expected["supervisor_pid"],
                           "start_identity": expected["supervisor_birth"]},
        "active_worker": {"pid": expected["worker_pid"],
                          "start_identity": expected["worker_birth"]},
        "stop_mode": "stop_scheduling_only_no_signal", "reconciliation_transition_limit": 2,
        "baseline_train_commands_unchanged": True, "providers_or_gpus_called": False,
        "execution_armed": False}


def _wait_for_supervisor_exit(pid, birth, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while start_identity(pid) == birth:
        require(time.monotonic() < deadline, "activation_old_supervisor_did_not_exit")
        time.sleep(0.25)


def activate(controller, new_config_path, expected, *, execute=False,
             supervisor_drain_seconds=120, poll_seconds=2,
             reason="activate-reviewed-next-evaluation-boundary",
             wait_for_supervisor_exit=_wait_for_supervisor_exit):
    proposed = plan(controller, new_config_path, expected)
    if not execute:
        return proposed
    require(type(supervisor_drain_seconds) is int and 1 <= supervisor_drain_seconds <= 600,
            "activation_supervisor_drain_limit")
    require(type(poll_seconds) is int and 1 <= poll_seconds <= 30, "activation_poll_limit")
    stop_path = controller.root / "stop-request.json"
    write(stop_path, {"schema": "eva.rsi-next-boundary-stop.v1",
        "mode": "stop_scheduling; active detached worker is not killed",
        "loop_id": expected["loop_id"], "active_attempt": expected["active_attempt"],
        "current_config": expected["config"],
        "old_supervisor": {"pid": expected["supervisor_pid"],
                           "start_identity": expected["supervisor_birth"]}}, exclusive=True)
    wait_for_supervisor_exit(expected["supervisor_pid"], expected["supervisor_birth"],
                             supervisor_drain_seconds)
    # Fail closed if the old supervisor won the completion race. Calling run(2)
    # from durable_checkpoint could otherwise launch an old-config evaluation.
    _, old_config = _active_binding(controller, expected, require_live_worker=False)
    selected, _ = _validate_new_config(old_config, new_config_path)
    controller.resume()
    settled = controller.run(execute=True, max_transitions=2, poll_seconds=poll_seconds)
    require(settled.get("status") != "blocked", "activation_boundary_reconciliation_blocked")
    state, unchanged = controller.load()
    require(state.get("status") == "ready" and state.get("active_attempt") is None
            and state.get("pending_training") is None and state.get("phase") != "durable_checkpoint"
            and not any(row.get("status") == "running" for row in state.get("attempts", [])),
            "activation_boundary_not_ready")
    require(state.get("config") == expected["config"] and unchanged == old_config,
            "activation_config_changed_during_reconciliation")
    require(commitment(selected) == proposed["new_config"],
            "activation_replacement_config_changed_while_waiting")
    transition = controller.reconfigure(selected, reason=reason, execute=True)
    print(json.dumps({"schema": "eva.rsi-next-boundary-activation.v1", "status": "activated",
        "loop_id": state["loop_id"], "settled_phase": state["phase"],
        "config_transition": transition, "continuous_supervision_started": True},
        sort_keys=True), flush=True)
    return controller.run(execute=True, poll_seconds=poll_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--new-config", required=True, type=Path)
    parser.add_argument("--expected-loop-id", required=True)
    parser.add_argument("--expected-active-attempt", required=True)
    parser.add_argument("--expected-config-path", required=True, type=Path)
    parser.add_argument("--expected-config-blake3", required=True)
    parser.add_argument("--expected-supervisor-pid", required=True, type=int)
    parser.add_argument("--expected-supervisor-birth", required=True)
    parser.add_argument("--expected-worker-pid", required=True, type=int)
    parser.add_argument("--expected-worker-birth", required=True)
    parser.add_argument("--supervisor-drain-seconds", type=int, default=120)
    parser.add_argument("--poll-seconds", type=int, default=2)
    parser.add_argument("--reason", default="activate-reviewed-next-evaluation-boundary")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    expected = {"loop_id": _uuid(args.expected_loop_id, "activation_loop_identity_invalid"),
        "active_attempt": _uuid(args.expected_active_attempt, "activation_attempt_identity_invalid"),
        "config": {"path": str(args.expected_config_path.resolve()),
                   "blake3": args.expected_config_blake3},
        "supervisor_pid": args.expected_supervisor_pid,
        "supervisor_birth": args.expected_supervisor_birth,
        "worker_pid": args.expected_worker_pid, "worker_birth": args.expected_worker_birth}
    try:
        result = activate(Controller(args.run_root), args.new_config, expected,
            execute=args.execute, supervisor_drain_seconds=args.supervisor_drain_seconds,
            poll_seconds=args.poll_seconds, reason=args.reason)
        print(json.dumps(result, sort_keys=True))
        return 2 if result.get("status") == "blocked" else 0
    except Exception as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__,
            "error_code": str(error) if isinstance(error, EvidenceError)
            else "next_boundary_activation_failed"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
