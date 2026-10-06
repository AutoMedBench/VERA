#!/usr/bin/env python3
"""Initialize, inspect, stop/resume or detach the ten-round RSI supervisor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from training.eva_rsi.controller import Controller, worker, write
from training.eva_rsi.evidence import EvidenceError, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "status", "run", "launch", "stop", "resume"):
        command = sub.add_parser(name)
        command.add_argument("--run-root", required=True, type=Path)
        if name == "init":
            command.add_argument("--config", required=True, type=Path)
        if name in ("run", "launch", "resume"):
            command.add_argument("--execute", action="store_true")
        if name in ("run", "resume"):
            command.add_argument("--max-transitions", type=int)
        if name == "launch":
            command.add_argument("--session", required=True)
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
            command = "umask 077; exec " + shlex.join(argv) + " >> " + shlex.quote(str(controller.root / "supervisor.log")) + " 2>&1"
            subprocess.run(["tmux", "new-session", "-d", "-s", args.session, command], check=True)
        print(json.dumps(controller.status(), sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__,
                          "error_code": str(exc) if isinstance(exc, EvidenceError) else "supervisor_operation_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
