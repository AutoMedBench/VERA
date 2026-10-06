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
from uuid import UUID

from blake3 import blake3

from .evidence import (EvidenceError, commitment, optimizer_events, read, require, verify_checkpoint_evidence,
                       verify_evaluation, verify_training)

SCHEMA = "eva.rsi-loop-state.v1"

CONFIG_IDENTITIES = ("schema", "rounds", "updates_per_round", "architecture_model_path",
                     "initial_model_path", "initial_checkpoint_root", "skill_catalog_id")


def validate_config(config):
    require(config.get("schema") == "eva.rsi-loop-config.v1", "loop_config_schema")
    require(config.get("rounds") == 10 and config.get("updates_per_round") == 50, "requires_ten_rounds_fifty_updates")
    commands = config.get("commands")
    require(isinstance(commands, dict) and {"baseline_eval", "train", "evaluation"} <= commands.keys()
            and commands.keys() <= {"baseline_eval", "train", "evaluation", "skill_attribution"}, "command_phases_invalid")
    for argv in commands.values():
        require(isinstance(argv, list) and argv and all(isinstance(value, str) and value and "\x00" not in value for value in argv), "command_argv_required")
        require(not any(re.search(r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|Bearer\s+\S+", value) for value in argv),
                "credentials_forbidden_in_command_config")
    for name in (*CONFIG_IDENTITIES[3:], "cwd"):
        require(isinstance(config.get(name), str) and bool(config[name]), "initial_identity_or_path_missing")


def operation_reason(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._:/-]{0,239}", value),
            "operation_reason_required")
    require(not re.search(r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|Bearer\s+\S+", value),
            "credentials_forbidden_in_reason")
    return value


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
                 training_verifier=verify_training, checkpoint_verifier=verify_checkpoint_evidence,
                 attribution_verifier=None):
        self.root = Path(root).resolve()
        self.evaluation_verifier = evaluation_verifier
        self.training_verifier = training_verifier
        self.checkpoint_verifier = checkpoint_verifier
        self.attribution_verifier = attribution_verifier

    @classmethod
    def initialize(cls, root: Path, config_path: Path):
        config = read(config_path)
        validate_config(config)
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
        path = self.owned_config_path(state["config"]["path"])
        require(state["schema"] == SCHEMA and state["config"] == commitment(path), "loop_state_config_binding")
        config = read(path)
        validate_config(config)
        if path != self.root / "config.json":
            transition_path = path.parent / "transition.json"
            require(state.get("config_transition") == commitment(transition_path), "config_transition_binding")
            transition = read(transition_path)
            require(transition.get("loop_id") == state["loop_id"] and transition.get("new_config") == state["config"],
                    "config_transition_identity")
        return state, config

    def owned_config_path(self, value):
        path = Path(value)
        require(path.is_absolute(), "config_path_not_owned")
        try:
            parts = path.relative_to(self.root).parts
        except ValueError:
            raise EvidenceError("config_path_not_owned") from None
        valid = parts == ("config.json",)
        if len(parts) == 3 and parts[0] == "config-revisions" and parts[2] == "config.json":
            try:
                valid = str(UUID(parts[1])) == parts[1]
            except ValueError:
                valid = False
        require(valid, "config_path_not_owned")
        cursor = self.root
        for part in parts:
            cursor /= part
            require(not cursor.is_symlink(), "config_path_symlink")
        require(path.is_file() and path.resolve(strict=True) == path, "config_path_not_owned")
        return path

    def reconfigure(self, config_path, *, reason, execute=False):
        require(execute, "execution_requires_explicit_flag")
        reason = operation_reason(reason)
        new_config = read(config_path)
        validate_config(new_config)
        with self.lock():
            state, old_config = self.load()
            require(state["status"] == "ready" and state["active_attempt"] is None
                    and state["pending_training"] is None and state["phase"] != "durable_checkpoint"
                    and not any(a.get("status") == "running" for a in state["attempts"]),
                    "reconfigure_requires_ready_between_attempts")
            require(all(new_config.get(k) == old_config.get(k) for k in CONFIG_IDENTITIES),
                    "reconfigure_initial_identity_changed")
            require(set(new_config) == set(old_config), "reconfigure_config_shape_changed")
            require(all(new_config[k] == old_config[k] for k in old_config if k not in {"commands", "cwd"}),
                    "reconfigure_only_commands_and_cwd")
            require(new_config != old_config, "reconfigure_no_change")
            revision_id = str(uuid4())
            directory = self.root / "config-revisions" / revision_id
            require(not directory.parent.is_symlink(), "config_path_symlink")
            write(directory / "config.json", new_config, exclusive=True)
            transition = {"schema": "eva.rsi-config-transition.v1", "transition_id": revision_id,
                "loop_id": state["loop_id"], "reason": reason, "previous_config": state["config"],
                "new_config": commitment(directory / "config.json"),
                "previous_state": commitment(self.root / "state.json"), "phase": state["phase"],
                "round": state["round"], "checkpoint_backed_updates": sum(r["durable_updates"] for r in state["rounds"]),
                "initial_identities_preserved": True, "attempts_launched": 0}
            write(directory / "transition.json", transition, exclusive=True)
            state["config"] = transition["new_config"]
            state["config_transition"] = commitment(directory / "transition.json")
            self.save(state)
            return transition

    def attach_evaluation_supplement(self, index_path, *, reason, execute=False):
        """Explicitly attach a new source composition to a terminal eval only.

        The failed command and all original outputs stay in place. This neither
        launches an actor nor advances the scheduler; explicit resume reopens the
        evidence. Recovery files live outside both immutable source attempts.
        """
        require(execute, "execution_requires_explicit_flag")
        reason = operation_reason(reason)
        with self.lock():
            state, _ = self.load()
            require(state["status"] == "blocked" and state["active_attempt"] is not None
                    and state["phase"] in ("baseline_eval", "evaluation")
                    and state["pending_training"] is None, "supplement_requires_blocked_terminal_evaluation")
            attempt = next(row for row in state["attempts"] if row["attempt_id"] == state["active_attempt"])
            root = self.root / "attempts" / attempt["attempt_id"]
            require(Path(attempt["root"]) == root and root.resolve(strict=True) == root
                    and not root.is_symlink() and "evaluation_supplement" not in attempt,
                    "supplement_attempt_identity_or_already_attached")
            result, owner = read(root / "exit.json"), read(root / "worker.json")
            require(result.get("attempt_id") == attempt["attempt_id"]
                    and (result.get("exit_code") != 0 or result.get("error_type") is not None),
                    "supplement_requires_retained_failed_exit")
            require(type(owner.get("pid")) is int and isinstance(owner.get("start_identity"), str)
                    and start_identity(owner["pid"]) != owner["start_identity"],
                    "supplement_original_worker_still_live")
            index_path = Path(index_path).resolve(strict=True)
            require(read(index_path).get("schema") == "eva.rsi-composite-evaluation-index.v1",
                    "supplement_requires_explicit_composite_index")
            context = read(root / "context.json")
            verification = self.evaluation_verifier(index_path, context)
            proof = verification["composite_provenance"]
            original = read(proof["sources"]["original"]["index"]["path"])
            actor = read(root / "evaluation-actor-binding.json")
            argv = actor["equivalent_actor_cli"]
            actual_run = Path(argv[argv.index("--run-root") + 1]).resolve(strict=True)
            require(actual_run.is_relative_to(root / "benchmark")
                    and Path(original["benchmark_run_root"]).resolve(strict=True) == actual_run
                    and original.get("actor_runtime_binding") == commitment(root / "evaluation-actor-binding.json")
                    and proof["sources"]["original"]["checkpoint_identity"] == actor["checkpoint_identity"],
                    "supplement_original_source_differs_from_active_attempt")
            recovery_id = str(uuid4())
            recovery_root = self.root / "recovery-records" / recovery_id
            record = {"schema": "eva.rsi-evaluation-supplement-attachment.v1",
                "recovery_id": recovery_id, "loop_id": state["loop_id"], "round": state["round"],
                "attempt_id": attempt["attempt_id"], "reason": reason,
                "original_exit": commitment(root / "exit.json"),
                "original_context": commitment(root / "context.json"),
                "composite_index": commitment(index_path), "original_outputs_mutated": False,
                "actors_launched": 0, "updates_credited": 0}
            write(recovery_root / "attachment.json", record, exclusive=True)
            write(recovery_root / "evaluation-verification.json", verification, exclusive=True)
            attempt["evaluation_supplement"] = commitment(recovery_root / "attachment.json")
            self.save(state)
            return record

    def retire_prelaunch(self, *, reason, execute=False):
        require(execute, "execution_requires_explicit_flag")
        reason = operation_reason(reason)
        with self.lock():
            state, _ = self.load()
            require(state["active_attempt"] is not None and state["phase"] == "train"
                    and state["pending_training"] is None, "retire_requires_active_prelaunch_train")
            matches = [a for a in state["attempts"] if a["attempt_id"] == state["active_attempt"]]
            require(len(matches) == 1 and matches[0]["phase"] == "train", "retire_train_attempt_identity")
            attempt = matches[0]
            root = self.root / "attempts" / attempt["attempt_id"]
            require(Path(attempt["root"]) == root and not root.is_symlink()
                    and root.resolve(strict=True) == root, "retire_attempt_path_not_owned")
            training = root / "training"
            require(not training.exists() and not training.is_symlink(), "retire_training_path_exists")
            exit_value = read(root / "exit.json")
            require(exit_value.get("attempt_id") == attempt["attempt_id"]
                    and type(exit_value.get("exit_code")) is int and exit_value["exit_code"] != 0,
                    "retire_requires_terminal_nonzero_exit")
            owner = read(root / "worker.json")
            require(type(owner.get("pid")) is int and isinstance(owner.get("start_identity"), str)
                    and start_identity(owner["pid"]) != owner["start_identity"], "retire_worker_still_live")
            request = read(root / "request.json")
            require(request.get("attempt_id") == attempt["attempt_id"] and request.get("phase") == "train"
                    and owner.get("request") == commitment(root / "request.json"), "retire_worker_request_binding")
            recovery_id = str(uuid4())
            receipt = {"schema": "eva.rsi-prelaunch-retirement.v1", "recovery_id": recovery_id,
                "loop_id": state["loop_id"], "attempt_id": attempt["attempt_id"], "reason": reason,
                "previous_state": commitment(self.root / "state.json"),
                "exit": commitment(root / "exit.json"), "request": commitment(root / "request.json"),
                "context": commitment(root / "context.json"), "worker": commitment(root / "worker.json"),
                "owned_worker_not_live": True, "training_path_absent": True,
                "credited_updates": 0, "retry_launched": False}
            path = self.root / "recovery-records" / recovery_id / "retirement.json"
            write(path, receipt, exclusive=True)
            attempt["status"] = "failed_prelaunch"
            attempt["exit"] = exit_value
            state.setdefault("recovery_records", []).append(commitment(path))
            state["active_attempt"] = None
            state["status"] = "ready"
            self.save(state)
            return receipt

    def _failed_zero_update_evidence(self, state):
        """Read-only proof for a failed training attempt, never a partial update."""
        require(state["active_attempt"] is not None and state["phase"] == "train"
                and state["pending_training"] is None, "retire_requires_active_zero_update_train")
        matches = [a for a in state["attempts"] if a["attempt_id"] == state["active_attempt"]]
        require(len(matches) == 1 and matches[0]["phase"] == "train", "retire_train_attempt_identity")
        attempt = matches[0]
        root = self.root / "attempts" / attempt["attempt_id"]
        require(Path(attempt["root"]) == root and not root.is_symlink()
                and root.resolve(strict=True) == root, "retire_attempt_path_not_owned")
        training = root / "training"
        require(training.is_dir() and not training.is_symlink()
                and training.resolve(strict=True) == training, "retire_training_path_not_owned")
        paths = {name: root / name for name in ("exit.json", "worker.json", "request.json", "context.json")}
        paths["training_receipt"] = training / "run-receipt.json"
        require(all(p.is_file() and not p.is_symlink() for p in paths.values()), "retire_evidence_file_invalid")
        exit_value, owner = read(paths["exit.json"]), read(paths["worker.json"])
        require(exit_value.get("attempt_id") == attempt["attempt_id"]
                and type(exit_value.get("exit_code")) is int and exit_value["exit_code"] != 0,
                "retire_requires_terminal_nonzero_exit")
        require(type(owner.get("pid")) is int and owner["pid"] > 0
                and isinstance(owner.get("start_identity"), str) and owner["start_identity"].isdigit(),
                "retire_worker_identity_invalid")
        birth = start_identity(owner["pid"])
        require(birth is not None or not Path(f"/proc/{owner['pid']}").exists(), "retire_worker_identity_unavailable")
        require(birth != owner["start_identity"], "retire_worker_still_live")
        request, context = read(paths["request.json"]), read(paths["context.json"])
        require(request.get("attempt_id") == attempt["attempt_id"] and request.get("phase") == "train"
                and owner.get("request") == commitment(paths["request.json"]), "retire_worker_request_binding")
        current = state["rounds"][-1] if state["rounds"] else {}
        remaining = 50 - current.get("durable_updates", 0)
        expected = {"loop_id": state["loop_id"], "phase": "train", "round": state["round"],
            "attempt_root": str(root), "context": str(root / "context.json"),
            "model_path": state["model_path"], "checkpoint_root": state["checkpoint_root"],
            "skill_catalog_id": state["skill_catalog_id"], "remaining_updates": remaining,
            "s_target": current.get("s_target"), "previous_evaluation": state["evaluation"]}
        require(0 < remaining <= 50 and all(context.get(k) == v for k, v in expected.items()),
                "retire_training_context_changed")
        receipt = read(paths["training_receipt"])
        require(receipt.get("stage") == "grpo" and receipt.get("status") == "failed",
                "retire_training_receipt_not_failed")
        argv = receipt.get("argv")
        require(isinstance(argv, list) and all(isinstance(x, str) for x in argv), "retire_training_argv_invalid")
        expected_args = {"--load": context["checkpoint_root"], "--save": str(training / "checkpoints"),
                         "--save-hf": str(training / "hf/iter_{rollout_id:07d}")}
        for key, value in expected_args.items():
            require(argv.count(key) == 1 and argv.index(key) + 1 < len(argv)
                    and argv[argv.index(key) + 1] == value, "retire_training_output_binding")
        absent = [training / "checkpoints", training / "hf", root / "checkpoint-verification.json"]
        require(all(not p.exists() and not p.is_symlink() for p in absent), "retire_output_checkpoint_exists")
        events = training / "optimizer-steps.jsonl"
        require(not events.is_symlink() and (not events.exists() or events.is_file()), "retire_optimizer_log_invalid")
        if events.exists():
            for line in events.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                require(isinstance(row, dict) and isinstance(row.get("event"), str), "retire_optimizer_log_invalid")
                require(row["event"] != "optimizer_step", "retire_optimizer_step_exists")
            paths["optimizer_events"] = events
        require(attempt.get("optimizer_executions", 0) == 0
                and attempt.get("learning_updates_observed", 0) == 0, "retire_attempt_update_already_observed")
        return {"evidence": {name: commitment(path) for name, path in paths.items()},
                "preserved_training_context": expected, "absent_output_paths": [str(p) for p in absent],
                "optimizer_event_file_present": events.exists(), "optimizer_steps_observed": 0,
                "owned_worker_not_live": True, "pending_checkpoint_absent": True}

    def retire_failed_zero_update(self, *, reason, execute=False):
        """Retain failed artifacts, then permit a fresh attempt from the same prefix."""
        require(execute, "execution_requires_explicit_flag")
        reason = operation_reason(reason)
        with self.lock():
            state, _ = self.load()
            proof = self._failed_zero_update_evidence(state)
            attempt = next(a for a in state["attempts"] if a["attempt_id"] == state["active_attempt"])
            recovery_id = str(uuid4())
            receipt = {"schema": "eva.rsi-failed-zero-update-retirement.v1", "recovery_id": recovery_id,
                "loop_id": state["loop_id"], "attempt_id": attempt["attempt_id"], "reason": reason,
                "previous_state": commitment(self.root / "state.json"), **proof,
                "credited_updates": 0, "training_artifacts_preserved": True, "retry_launched": False}
            path = self.root / "recovery-records" / recovery_id / "retirement.json"
            write(path, receipt, exclusive=True)
            attempt["status"] = "failed_zero_update"
            attempt["exit"] = read(Path(attempt["root"]) / "exit.json")
            state.setdefault("recovery_records", []).append(commitment(path))
            state["active_attempt"] = None
            state["status"] = "ready"
            self.save(state)
            return receipt

    def _failed_nondurable_evidence(self, state):
        """Prove an observed failed prefix has no saved output to recover."""
        pending = state.get("pending_training")
        require(state["phase"] == "durable_checkpoint" and state["status"] == "blocked"
                and state["active_attempt"] is None and isinstance(pending, dict),
                "retire_requires_blocked_nondurable_prefix")
        root = Path(pending["attempt_root"])
        matches = [a for a in state["attempts"] if a["root"] == str(root)]
        require(len(matches) == 1 and matches[0]["phase"] == "train"
                and matches[0]["round"] == state["round"], "retire_train_attempt_identity")
        attempt = matches[0]
        require(root == self.root / "attempts" / attempt["attempt_id"] and not root.is_symlink()
                and root.resolve(strict=True) == root, "retire_attempt_path_not_owned")
        training = root / "training"
        require(training.is_dir() and not training.is_symlink(), "retire_training_path_not_owned")
        paths = {name: root / name for name in
                 ("exit.json", "worker.json", "request.json", "context.json", "training-verification.json")}
        paths.update(training_receipt=training / "run-receipt.json",
                     optimizer_events=training / "optimizer-steps.jsonl")
        require(all(p.is_file() and not p.is_symlink() for p in paths.values()), "retire_evidence_file_invalid")
        context, verified = read(paths["context.json"]), pending["training"]
        require(context == pending["context"] and read(paths["training-verification.json"]) == verified,
                "retire_pending_evidence_changed")
        require(verified["run_receipt"] == commitment(paths["training_receipt"])
                and verified["event_source"] == commitment(paths["optimizer_events"]),
                "retire_training_commitment_changed")
        events = optimizer_events(paths["optimizer_events"])
        require(all(events[k] == verified[k] for k in ("events", "executions", "learning_updates", "duplicate_events"))
                and events["executions"] > 0
                and attempt.get("optimizer_executions") == events["executions"]
                and attempt.get("learning_updates_observed") == events["learning_updates"],
                "retire_observed_counts_inconsistent")
        current = state["rounds"][-1]
        remaining = 50 - current["durable_updates"]
        expected = {"loop_id": state["loop_id"], "phase": "train", "round": state["round"],
            "attempt_root": str(root), "context": str(root / "context.json"),
            "model_path": state["model_path"], "checkpoint_root": state["checkpoint_root"],
            "skill_catalog_id": state["skill_catalog_id"], "remaining_updates": remaining,
            "s_target": current["s_target"], "previous_evaluation": state["evaluation"]}
        require(0 < events["executions"] <= remaining <= 50
                and all(context.get(k) == v for k, v in expected.items()), "retire_training_context_changed")
        exit_value, owner = read(paths["exit.json"]), read(paths["worker.json"])
        require(exit_value.get("attempt_id") == attempt["attempt_id"] and attempt.get("exit") == exit_value
                and type(exit_value.get("exit_code")) is int and exit_value["exit_code"] != 0,
                "retire_requires_terminal_nonzero_exit")
        require(type(owner.get("pid")) is int and owner["pid"] > 0
                and isinstance(owner.get("start_identity"), str) and owner["start_identity"].isdigit(),
                "retire_worker_identity_invalid")
        birth = start_identity(owner["pid"])
        require(birth is not None or not Path(f"/proc/{owner['pid']}").exists(), "retire_worker_identity_unavailable")
        require(birth != owner["start_identity"], "retire_worker_still_live")
        request = read(paths["request.json"])
        require(request.get("attempt_id") == attempt["attempt_id"] and request.get("phase") == "train"
                and owner.get("request") == commitment(paths["request.json"]), "retire_worker_request_binding")
        receipt = read(paths["training_receipt"])
        require(receipt.get("stage") == "grpo" and receipt.get("status") == verified.get("launch_status") == "failed",
                "retire_training_receipt_not_failed")
        require(verified.get("training_root") == str(training)
                and verified.get("checkpoint_root") == str(training / "checkpoints"), "retire_training_output_binding")
        argv = receipt.get("argv")
        require(isinstance(argv, list) and all(isinstance(x, str) for x in argv), "retire_training_argv_invalid")
        for flag, value in {"--load": context["checkpoint_root"], "--save": str(training / "checkpoints"),
                            "--save-hf": str(training / "hf/iter_{rollout_id:07d}")}.items():
            require(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv)
                    and argv[argv.index(flag) + 1] == value, "retire_training_output_binding")
        absent = [training / "checkpoints", training / "hf", root / "checkpoint-verification.json"]
        require(all(not p.exists() and not p.is_symlink() for p in absent), "retire_saved_checkpoint_requires_verifier")
        for path in training.rglob("*"):
            require(path.name not in {"latest_checkpointed_iteration.txt", ".metadata", "pytorch_model.bin"}
                    and path.suffix.lower() not in {".distcp", ".safetensors", ".pt", ".pth", ".ckpt"},
                    "retire_saved_checkpoint_requires_verifier")
        return {"attempt_id": attempt["attempt_id"],
                "evidence": {name: commitment(path) for name, path in paths.items()},
                "preserved_pending_training": pending, "preserved_training_context": expected,
                "absent_output_paths": [str(p) for p in absent], "saved_checkpoint_absent": True,
                "observed_discarded_optimizer_executions": events["executions"],
                "observed_discarded_nonzero_learning_updates": events["learning_updates"],
                "owned_worker_not_live": True}

    def retire_failed_nondurable(self, *, reason, execute=False):
        """Explicitly abandon only an unsaved failed prefix; never resume or retry it."""
        require(execute, "execution_requires_explicit_flag")
        reason = operation_reason(reason)
        with self.lock():
            state, _ = self.load()
            proof = self._failed_nondurable_evidence(state)
            recovery_id = str(uuid4())
            receipt = {"schema": "eva.rsi-failed-nondurable-retirement.v1", "recovery_id": recovery_id,
                "loop_id": state["loop_id"], "reason": reason, "previous_state": commitment(self.root / "state.json"),
                **proof, "credited_updates": 0, "credited_learning_updates": 0,
                "training_artifacts_preserved": True, "retry_launched": False}
            path = self.root / "recovery-records" / recovery_id / "retirement.json"
            write(path, receipt, exclusive=True)
            attempt = next(a for a in state["attempts"] if a["attempt_id"] == proof["attempt_id"])
            attempt["status"] = "failed_nondurable"
            state.setdefault("recovery_records", []).append(commitment(path))
            state["pending_training"] = None
            state["phase"], state["status"] = "train", "ready"
            self.save(state)
            return receipt

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
        pending = None
        if state.get("pending_training"):
            verified = state["pending_training"].get("training", {})
            pending = {key: verified.get(key) for key in ("executions", "learning_updates", "duplicate_events")}
        return {"schema": "eva.rsi-loop-progress.v1", "loop_id": state["loop_id"], "status": state["status"],
                "phase": state["phase"], "round": state["round"],
                "completed_rounds": sum(row.get("complete", False) for row in state["rounds"]),
                "target_rounds": 10, "target_updates_per_round": 50,
                "checkpoint_backed_updates": sum(row["durable_updates"] for row in state["rounds"]),
                "checkpoint_backed_learning_updates": sum(row["durable_learning_updates"] for row in state["rounds"]),
                "observed_optimizer_executions": sum(row.get("optimizer_executions", 0) for row in state["attempts"]),
                "active_attempt": state["active_attempt"], "retained_attempts": len(state["attempts"]),
                "active_training_observations_not_yet_checkpoint_credited": live,
                "pending_training_observations_not_yet_checkpoint_credited": pending,
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
                "exact_optimizer_resume": False,
                **({"skill_selection": state["skill_selection"]} if state.get("skill_selection") else {})}

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
        if attempt["phase"] == "skill_attribution":
            from .skill_attribution import retain_decision, verify_result
            if (root / "skill-attribution/result.json").is_file():
                decision = (self.attribution_verifier or verify_result)(root, context)
            else:
                # A real detached command exited without a usable review. It is
                # not proof that Opus was contacted or that the service is down.
                decision = retain_decision(state["skill_catalog_id"], "command_failed_no_result",
                    command_exit=result, provider_dispatch_started=None)
            state["rounds"][-1]["skill_decision"] = decision
            retain_evidence(root / "attribution-verification.json", decision)
            state["phase"] = "skill_decision"
        elif attempt["phase"] == "train":
            training = self.training_verifier(root / "training", context)
            attempt["optimizer_executions"] = training["executions"]
            attempt["learning_updates_observed"] = training["learning_updates"]
            retain_evidence(root / "training-verification.json", training)
            state["pending_training"] = {"attempt_root": str(root), "training": training, "context": context}
            state["phase"] = "durable_checkpoint"
        else:
            # A failing evaluation may still retain fully verified evidence; no
            # exit-zero or launch manifest is accepted in lieu of actual Judge validation.
            supplement = attempt.get("evaluation_supplement")
            if supplement:
                require(commitment(supplement["path"]) == supplement,
                        "evaluation_supplement_attachment_changed")
                attachment = read(supplement["path"])
                require(attachment["loop_id"] == state["loop_id"]
                        and attachment["round"] == state["round"]
                        and attachment["attempt_id"] == attempt["attempt_id"]
                        and attachment["original_context"] == commitment(root / "context.json")
                        and attachment["original_exit"] == commitment(exit_path)
                        and commitment(attachment["composite_index"]["path"]) == attachment["composite_index"],
                        "evaluation_supplement_source_changed")
                index_path = Path(attachment["composite_index"]["path"])
                verification_path = Path(supplement["path"]).parent / "evaluation-verification.json"
            else:
                index_path, verification_path = root / "evaluation-index.json", root / "evaluation-verification.json"
            evaluation = self.evaluation_verifier(index_path, context)
            retain_evidence(verification_path, evaluation)
            state["evaluation"] = str(verification_path)
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
            if "skill_decision" not in current:
                _, config = self.load()
                if "skill_attribution" in config["commands"]:
                    state["phase"] = "skill_attribution"
                    self.save(state)
                    return
                from .skill_attribution import retain_decision
                current["skill_decision"] = retain_decision(state["skill_catalog_id"], "not_requested")
            if "selection_decision" not in current:
                from .skill_selection import apply_verified_attribution
                current["selection_decision"] = apply_verified_attribution(
                    self.root / "skill-selections", current["skill_decision"],
                    previous=state.get("skill_selection"), catalog_id=state["skill_catalog_id"])
                state["skill_selection"] = current["selection_decision"]["selection"]
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
                    elif state["phase"] in ("baseline_eval", "train", "evaluation", "skill_attribution"):
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
