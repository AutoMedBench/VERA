"""Small single-host durable state machine. No GPU/provider work at import/status.

Commands are explicit, operator-owned wrappers, not inferred from launch files.
Workers outlive a disconnected supervisor and leave an exclusive exit receipt.
Failed attempts remain in the ledger; incomplete evaluations are never replayed
automatically. A checkpoint-backed training prefix can resume in a new attempt.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from uuid import uuid4

from blake3 import blake3

from .evidence import (EvidenceError, commitment, optimizer_events, read, require, verify_checkpoint_evidence,
                       verify_evaluation, verify_training)

SCHEMA = "eva.rsi-loop-state.v1"


def write(path: Path, value, *, exclusive=False):
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    temporary = path.with_name(path.name + "." + str(uuid4()) + ".tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    if exclusive:
        os.link(temporary, path)  # Exclusive and atomic visibility of a complete JSON document.
        temporary.unlink()
    else:
        os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def start_identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def retain_evidence(path, value):
    if path.exists():
        require(read(path) == value, "retained_verification_evidence_changed")
    else:
        write(path, value, exclusive=True)


def worker(request_path: Path):
    """Private child runner; records exit even when the main supervisor disappears."""
    os.umask(0o077)
    request = read(request_path)
    root = request_path.parent
    write(root / "worker.json", {"pid": os.getpid(), "start_identity": start_identity(os.getpid()),
                                "request": commitment(request_path)}, exclusive=True)
    result = {"attempt_id": request["attempt_id"], "exit_code": None, "error_type": None}
    try:
        with (root / "command.log").open("xb") as log:
            process = subprocess.Popen(request["argv"], cwd=request["cwd"], stdout=log, stderr=log)
            result["exit_code"] = process.wait()
    except Exception as exc:
        result["error_type"] = type(exc).__name__  # Never store exception text or env.
    write(root / "exit.json", result, exclusive=True)


class Controller:
    def __init__(self, root, *, evaluation_verifier=verify_evaluation,
                 training_verifier=verify_training, checkpoint_verifier=verify_checkpoint_evidence):
        self.root = Path(root).resolve()
        self.evaluation_verifier = evaluation_verifier
        self.training_verifier = training_verifier
        self.checkpoint_verifier = checkpoint_verifier

    @classmethod
    def initialize(cls, root: Path, config_path: Path):
        config = read(config_path)
        require(config.get("schema") == "eva.rsi-loop-config.v1", "loop_config_schema")
        require(config.get("rounds") == 10 and config.get("updates_per_round") == 50, "requires_ten_rounds_fifty_updates")
        for phase in ("baseline_eval", "train", "evaluation"):
            argv = config["commands"][phase]
            require(isinstance(argv, list) and argv and all(isinstance(value, str) for value in argv), "command_argv_required")
            require(not any(re.search(r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|Bearer\s+\S+", value) for value in argv),
                    "credentials_forbidden_in_command_config")
        for name in ("architecture_model_path", "initial_model_path", "initial_checkpoint_root", "skill_catalog_id", "cwd"):
            require(isinstance(config.get(name), str) and bool(config[name]), "initial_identity_or_path_missing")
        root = Path(root).resolve()
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        write(root / "config.json", config, exclusive=True)
        state = {"schema": SCHEMA, "loop_id": str(uuid4()), "config": commitment(root / "config.json"),
                 "phase": "baseline_eval", "status": "ready", "round": 1, "attempts": [], "rounds": [],
                 "model_path": config["initial_model_path"], "checkpoint_root": config["initial_checkpoint_root"],
                 "skill_catalog_id": config["skill_catalog_id"], "evaluation": None, "active_attempt": None,
                 "pending_training": None, "errors": [], "zero_learning_streak": 0, "warnings": []}
        write(root / "state.json", state, exclusive=True)
        return cls(root)

    def load(self):
        state = read(self.root / "state.json")
        require(state["schema"] == SCHEMA and state["config"] == commitment(self.root / "config.json"), "loop_state_config_binding")
        return state, read(self.root / "config.json")

    def status(self):
        """Strictly read-only: no scheduling, lock creation, output writes or imports of GPU stacks."""
        state, _ = self.load()
        live = None
        if state["active_attempt"]:
            attempt = next(row for row in state["attempts"] if row["attempt_id"] == state["active_attempt"])
            if attempt["phase"] == "train":
                try:
                    observed = optimizer_events(Path(attempt["root"]) / "training/optimizer-steps.jsonl")
                    live = {key: observed[key] for key in ("executions", "learning_updates", "duplicate_events")}
                except (ValueError, OSError):
                    live = {"status": "partial_or_invalid_event_stream_not_credited"}
        return {"schema": "eva.rsi-loop-progress.v1", "loop_id": state["loop_id"], "status": state["status"],
                "phase": state["phase"], "round": state["round"],
                "completed_rounds": sum(row.get("complete", False) for row in state["rounds"]),
                "target_rounds": 10, "target_updates_per_round": 50,
                "checkpoint_backed_updates": sum(row["durable_updates"] for row in state["rounds"]),
                "checkpoint_backed_learning_updates": sum(row["durable_learning_updates"] for row in state["rounds"]),
                "observed_optimizer_executions": sum(row.get("optimizer_executions", 0) for row in state["attempts"]),
                "active_attempt": state["active_attempt"], "retained_attempts": len(state["attempts"]),
                "active_training_observations_not_yet_checkpoint_credited": live,
                "retained_errors": len(state["errors"]), "warnings": state["warnings"],
                "exact_optimizer_resume": False, "memory_context_pass": None}

    def save(self, state):
        write(self.root / "state.json", state)
        write(self.root / "progress.json", self.status())

    @contextmanager
    def lock(self):
        with (self.root / "supervisor.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def context(self, state, config, attempt_root):
        current = state["rounds"][-1] if state["rounds"] else {}
        return {"loop_id": state["loop_id"], "round": state["round"], "phase": state["phase"],
                "attempt_root": str(attempt_root), "context": str(attempt_root / "context.json"),
                "model_path": state["model_path"], "architecture_model_path": config["architecture_model_path"],
                "checkpoint_root": state["checkpoint_root"], "skill_catalog_id": state["skill_catalog_id"],
                "remaining_updates": 50 - current.get("durable_updates", 0), "s_target": current.get("s_target"),
                "previous_evaluation": state["evaluation"], "judge_backend": "native_astra",
                "exact_optimizer_resume": False}

    def launch(self, state, config):
        attempt_id = str(uuid4())
        root = self.root / "attempts" / attempt_id
        root.mkdir(parents=True, mode=0o700)
        context = self.context(state, config, root)
        write(root / "context.json", context, exclusive=True)
        argv = [part.format_map(context) for part in config["commands"][state["phase"]]]
        request = {"attempt_id": attempt_id, "argv": argv, "cwd": config["cwd"], "phase": state["phase"]}
        write(root / "request.json", request, exclusive=True)
        state["attempts"].append({"attempt_id": attempt_id, "root": str(root), "phase": state["phase"],
                                  "round": state["round"], "status": "running", "exit": None})
        state["active_attempt"] = attempt_id
        state["status"] = "running"
        self.save(state)  # Intent is durable before launch; ambiguous interruption never silently relaunches.
        script = Path(__file__).resolve().parents[2] / "scripts/run_eva_rsi_loop_v1.py"
        with (root / "worker.log").open("xb") as log:
            subprocess.Popen([sys.executable, str(script), "_worker", "--request", str(root / "request.json")],
                             cwd=config["cwd"], stdout=log, stderr=log, start_new_session=True)

    def finish_attempt(self, state):
        attempt = next(row for row in state["attempts"] if row["attempt_id"] == state["active_attempt"])
        root = Path(attempt["root"])
        exit_path = root / "exit.json"
        if not exit_path.exists():
            live = read(root / "worker.json") if (root / "worker.json").exists() else None
            if live and start_identity(live["pid"]) != live["start_identity"]:
                raise EvidenceError("worker_disappeared_without_exit_receipt")
            if not live and time.time() - (root / "request.json").stat().st_mtime > 30:
                raise EvidenceError("ambiguous_launch_without_worker_receipt")
            return False
        result = read(exit_path)
        require(result["attempt_id"] == attempt["attempt_id"], "worker_exit_identity")
        attempt["exit"] = result
        context = read(root / "context.json")
        failed = result["exit_code"] != 0 or result["error_type"] is not None
        if failed:
            failure = {"attempt_id": attempt["attempt_id"], "code": "command_failed", "exit": result}
            if failure not in state["errors"]:
                state["errors"].append(failure)
        if attempt["phase"] == "train":
            training = self.training_verifier(root / "training", context)
            attempt["optimizer_executions"] = training["executions"]
            attempt["learning_updates_observed"] = training["learning_updates"]
            retain_evidence(root / "training-verification.json", training)
            state["pending_training"] = {"attempt_root": str(root), "training": training, "context": context}
            state["phase"] = "durable_checkpoint"
        else:
            # A failing evaluation may still retain fully verified evidence; no
            # exit-zero or launch manifest is accepted in lieu of actual Judge validation.
            evaluation = self.evaluation_verifier(root / "evaluation-index.json", context)
            retain_evidence(root / "evaluation-verification.json", evaluation)
            state["evaluation"] = str(root / "evaluation-verification.json")
            if attempt["phase"] == "evaluation":
                state["rounds"][-1]["post_evaluation"] = state["evaluation"]
                state["phase"] = "skill_decision"
            else:
                state["phase"] = "target_selection"
        attempt["status"] = "verified_with_retained_command_failure" if failed else "verified"
        state["active_attempt"] = None
        state["status"] = "ready"
        self.save(state)
        return True

    def advance_internal(self, state):
        phase = state["phase"]
        if phase == "target_selection":
            proposal = read(state["evaluation"])["proposal"]
            require(proposal["status"] == "target_proposed" and proposal["s_target"] in ("S1", "S2", "S3", "S4", "S5"),
                    "no_observed_stage_for_target_selection")
            state["rounds"].append({"round": state["round"], "s_target": proposal["s_target"],
                                     "source_evaluation": state["evaluation"], "durable_updates": 0,
                                     "durable_learning_updates": 0, "segments": [], "complete": False})
            state["phase"] = "train"
        elif phase == "durable_checkpoint":
            pending = state["pending_training"]
            checkpoint = self.checkpoint_verifier(pending["training"], pending["context"])
            require(0 < checkpoint["durable_updates"] <= pending["context"]["remaining_updates"], "durable_update_count_invalid")
            root = Path(pending["attempt_root"])
            retain_evidence(root / "checkpoint-verification.json", checkpoint)
            current = state["rounds"][-1]
            current["segments"].append(str(root / "checkpoint-verification.json"))
            current["durable_updates"] += checkpoint["durable_updates"]
            current["durable_learning_updates"] += checkpoint["durable_learning_updates"]
            for event in pending["training"]["events"][:checkpoint["durable_updates"]]:
                learns = event["gradient_norm"] > 0 and event["sampled_changed_values"] > 0
                state["zero_learning_streak"] = 0 if learns else state["zero_learning_streak"] + 1
                if state["zero_learning_streak"] >= 3 and "three_consecutive_no_learning_updates" not in state["warnings"]:
                    state["warnings"].append("three_consecutive_no_learning_updates")
            state["model_path"], state["checkpoint_root"] = checkpoint["model_path"], checkpoint["checkpoint_root"]
            state["pending_training"] = None
            state["phase"] = "evaluation" if current["durable_updates"] == 50 else "train"
        elif phase == "skill_decision":
            current = state["rounds"][-1]
            require(current["durable_updates"] == 50 and current.get("post_evaluation"), "round_not_durable_and_evaluated")
            current["skill_decision"] = {"action": "retain_catalog", "catalog_id": state["skill_catalog_id"],
                "opus_attribution": "unavailable_or_not_supplied", "skill_change_performed": False,
                "attribution_claimed": False}
            current["learning_outcome"] = "validated" if current["durable_learning_updates"] == 50 else "inconclusive_learning_signal"
            current["complete"] = True
            if state["round"] == 10:
                state["phase"], state["status"] = "complete", "complete"
            else:
                state["round"] += 1
                state["phase"] = "target_selection"
        else:
            raise EvidenceError("unknown_internal_phase")
        self.save(state)

    def run(self, *, execute=False, max_transitions=None, poll_seconds=2):
        self.status()  # Read-only reconciliation always precedes any mutation.
        require(execute, "execution_requires_explicit_flag")
        with self.lock():
            count = 0
            while True:
                state, config = self.load()
                if state["status"] in ("complete", "blocked") or (self.root / "stop-request.json").exists():
                    return self.status()

                if max_transitions is not None and count >= max_transitions:
                    return self.status()
                try:
                    if state["active_attempt"]:
                        if not self.finish_attempt(state):
                            time.sleep(poll_seconds)
                            continue
                    elif state["phase"] in ("baseline_eval", "train", "evaluation"):
                        self.launch(state, config)
                    else:
                        self.advance_internal(state)
                    count += 1
                except Exception as exc:
                    state["status"] = "blocked"
                    state["errors"].append({"attempt_id": state["active_attempt"],
                        "code": str(exc) if isinstance(exc, EvidenceError) else "evidence_or_controller_error",
                        "error_type": type(exc).__name__})
                    self.save(state)
                    return self.status()

    def resume(self):
        """Explicitly retry only evidence reconciliation; never erase failures or relaunch attempts."""
        self.status()
        with self.lock():
            state, _ = self.load()
            require(state["status"] != "complete", "completed_loop_cannot_resume")
            state["status"] = "ready"
            stop = self.root / "stop-request.json"
            if stop.exists():
                stop.rename(self.root / ("retained-stop-" + str(uuid4()) + ".json"))
            self.save(state)
