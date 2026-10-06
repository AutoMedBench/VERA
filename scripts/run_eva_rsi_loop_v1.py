#!/usr/bin/env python3
"""Initialize, inspect, stop/resume or detach the ten-round RSI supervisor."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.harness_source import activate_harness
activate_harness(ROOT)
from training.eva_rsi.controller import Controller, worker, write
from training.eva_rsi.evidence import EvidenceError, require


DETACHED_ENVIRONMENT_KEYS = (
    "EVA_HARNESS_ROOT", "EVA_MEDRESEARCH_DATA_ROOT", "EVA_GRPO_CONTEXT_PROFILE",
    "EVA_GRPO_CODEX_BIN", "EVA_GRPO_ENABLE_THINKING", "EVA_GRPO_MAX_TOOL_FRONTIERS",
    "EVA_SLIME_NATIVE_AUTH_PATH", "EVA_SLIME_JUDGE_CONCURRENCY", "EVA_SLIME_LOSS_MEMORY",
    "PYTHONDONTWRITEBYTECODE",
)


def detached_command(argv, log_path, environment=None):
    """A pre-existing tmux server does not inherit arbitrary caller variables.

    Forward only explicit runtime selectors and an auth-file path, never API
    keys or an environment dump. Child workers inherit these from the actual
    detached supervisor, and training records its selected source paths.
    """
    environment = os.environ if environment is None else environment
    assignments = [f"{key}={environment[key]}" for key in DETACHED_ENVIRONMENT_KEYS
                   if environment.get(key)]
    return ("umask 077; exec " + shlex.join(["env", *assignments, *argv])
            + " >> " + shlex.quote(str(log_path)) + " 2>&1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "status", "run", "launch", "stop", "resume", "reconfigure", "retire-prelaunch",
                 "retire-failed-zero-update", "retire-failed-nondurable", "attach-evaluation-supplement"):
        command = sub.add_parser(name)
        command.add_argument("--run-root", required=True, type=Path)
        if name in ("init", "reconfigure"):
            command.add_argument("--config", required=True, type=Path)
        if name in ("run", "launch", "resume", "reconfigure", "retire-prelaunch", "retire-failed-zero-update", "retire-failed-nondurable", "attach-evaluation-supplement"):
            command.add_argument("--execute", action="store_true")
        if name in ("run", "resume"):
            command.add_argument("--max-transitions", type=int)
        if name == "launch":
            command.add_argument("--session", required=True)
        if name in ("reconfigure", "retire-prelaunch", "retire-failed-zero-update", "retire-failed-nondurable", "attach-evaluation-supplement"):
            command.add_argument("--reason", required=True)
        if name == "attach-evaluation-supplement":
            command.add_argument("--index", required=True, type=Path)
    internal = sub.add_parser("_worker")
    internal.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "_worker":
            worker(args.request)
            return 0
        controller = Controller(args.run_root)
        if args.command == "init":
            controller = Controller.initialize(args.run_root, args.config)
        elif args.command == "reconfigure":
            result = controller.reconfigure(args.config, reason=args.reason, execute=args.execute)
            print(json.dumps(result, sort_keys=True))
            return 0
        elif args.command == "attach-evaluation-supplement":
            result = controller.attach_evaluation_supplement(args.index, reason=args.reason, execute=args.execute)
            print(json.dumps(result, sort_keys=True))
            return 0
        elif args.command == "retire-prelaunch":
            result = controller.retire_prelaunch(reason=args.reason, execute=args.execute)
            print(json.dumps(result, sort_keys=True))
            return 0
        elif args.command == "retire-failed-zero-update":
            result = controller.retire_failed_zero_update(reason=args.reason, execute=args.execute)
            print(json.dumps(result, sort_keys=True))
            return 0
        elif args.command == "retire-failed-nondurable":
            result = controller.retire_failed_nondurable(reason=args.reason, execute=args.execute)
            print(json.dumps(result, sort_keys=True))
            return 0
        elif args.command in ("run", "resume"):
            require(args.execute, "execution_requires_explicit_flag")
            if args.command == "resume":
                controller.resume()
            result = controller.run(execute=True, max_transitions=args.max_transitions)
            print(json.dumps(result, sort_keys=True))
            return 2 if result["status"] == "blocked" else 0
        elif args.command == "stop":
            controller.status()
            write(controller.root / "stop-request.json", {"request_id": str(uuid4()),
                "mode": "stop_scheduling; active detached worker is not killed"})
        elif args.command == "launch":
            require(args.execute, "execution_requires_explicit_flag")
            controller.status()
            require(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.session), "invalid_tmux_session")
            argv = [sys.executable, str(Path(__file__).resolve()), "run", "--run-root", str(controller.root), "--execute"]
            command = detached_command(argv, controller.root / "supervisor.log")
            subprocess.run(["tmux", "new-session", "-d", "-s", args.session, command], check=True)
        print(json.dumps(controller.status(), sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__,
                          "error_code": str(exc) if isinstance(exc, EvidenceError) else "supervisor_operation_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
