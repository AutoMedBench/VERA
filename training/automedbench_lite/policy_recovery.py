"""Current-time recovery evidence for an interrupted, already executed turn.

This module never resumes a thread, calls a provider, or writes into the
original run.  A recovery bundle is admitted only after separately captured
process-birth identities for the original actor and its MCP server are gone.
The resulting snapshot describes the workspace at recovery time, not at the
original interruption time.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from eva_agent.codex_runtime import codex_turn_receipt_from_document, verify_codex_turn_receipt
from eva_agent.pipeline.digests import blake3_hex

from .adapter import EvaluationError, file_digest, read_document, write_once
from .policy_capture import require_joined_host_results
from .track_feedback import track_snapshot
from .track_tools import MutableInventory


OWNERSHIP_SCHEMA = "eva.automedbench-policy-recovery-ownership.v1"
POST_EXIT_OWNERSHIP_SCHEMA = "eva.automedbench-policy-recovery-post-exit-ownership.v1"
RECOVERY_SCHEMA = "eva.automedbench-policy-recovery.v1"
STABILITY_SCHEMA = "eva.automedbench-policy-recovery-stability.v1"
PHASES = ("01-planning", "02-setup", "03-smoke", "04-full-subset", "05-review")


def _process(pid: int) -> dict:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise EvaluationError("recovery_process_pid_invalid")
    root = Path(f"/proc/{pid}")
    try:
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        argv = (root / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    except FileNotFoundError as exc:
        raise EvaluationError("recovery_owned_process_not_live_at_capture") from exc
    if len(fields) < 20 or not argv or any(not item for item in argv):
        raise EvaluationError("recovery_process_identity_invalid")
    return {"pid": pid, "parent_pid": int(fields[1]), "start_ticks": fields[19],
            "argv_blake3": blake3_hex(tuple(item.decode("utf-8", "strict") for item in argv))}


def _argv(pid: int) -> tuple[str, ...]:
    try:
        values = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    except FileNotFoundError as exc:
        raise EvaluationError("recovery_owned_process_not_live_at_capture") from exc
    try:
        return tuple(item.decode("utf-8", "strict") for item in values if item)
    except UnicodeDecodeError as exc:
        raise EvaluationError("recovery_process_identity_invalid") from exc


def _option(argv: tuple[str, ...], name: str) -> str | None:
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        return None
    return argv[positions[0] + 1]


def _run_paths(run_root: Path, track: str, phase: str) -> dict:
    run = run_root.resolve(strict=True)
    manifest = read_document(run / "track-run-manifest.json", maximum=16 * 1024**2)
    matches = [row for row in manifest.get("tracks", ()) if row.get("track") == track]
    if len(matches) != 1 or phase not in PHASES:
        raise EvaluationError("recovery_track_or_phase_not_admitted")
    workspace = (run / matches[0]["workspace_relative"]).resolve(strict=True)
    audit = (run / "track-rollouts" / track).resolve(strict=True)
    turn = (audit / "turns" / phase).resolve(strict=True)
    return {"run": run, "workspace": workspace, "audit": audit, "turn": turn,
            "manifest": manifest}


def _outside_original(path: Path, run: Path) -> Path:
    target = path.resolve(strict=False)
    if target == run or target.is_relative_to(run):
        raise EvaluationError("recovery_output_must_be_outside_original_run")
    return target


def capture_ownership(*, run_root: Path, track: str, phase: str, output: Path) -> dict:
    """Bind live actor/app-server/MCP births before any recovery is possible."""
    paths = _run_paths(run_root, track, phase)
    target = _outside_original(output, paths["run"])
    baseline = read_document(paths["run"] / "baseline-process.json")
    thread = read_document(paths["audit"] / "thread.json")
    actor = _process(baseline.get("actor_pid"))
    actor_argv = _argv(actor["pid"])
    if actor_argv != tuple(baseline.get("command", ())):
        raise EvaluationError("recovery_actor_command_binding_invalid")
    app_server = _process(thread.get("app_server_pid"))
    app_argv = _argv(app_server["pid"])
    if app_server["parent_pid"] != actor["pid"] or "app-server" not in app_argv:
        raise EvaluationError("recovery_app_server_ownership_invalid")
    candidates = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            identity = _process(int(item.name))
            if identity["parent_pid"] != app_server["pid"]:
                continue
            argv = _argv(identity["pid"])
        except PermissionError as exc:
            raise EvaluationError("recovery_process_inventory_unreadable") from exc
        except (EvaluationError, ProcessLookupError):
            continue
        if (_option(argv, "--audit-root") is not None
                and Path(_option(argv, "--audit-root")).resolve() == paths["audit"]
                and _option(argv, "--workspace") is not None
                and Path(_option(argv, "--workspace")).resolve() == paths["workspace"]):
            candidates.append((identity, argv))
    if len(candidates) != 1:
        raise EvaluationError("recovery_owned_mcp_process_not_unique")
    mcp, mcp_argv = candidates[0]
    if not any(value.endswith("track_entry.py") for value in mcp_argv) or "serve" not in mcp_argv:
        raise EvaluationError("recovery_owned_mcp_command_invalid")
    value = {"schema": OWNERSHIP_SCHEMA, "run_root": str(paths["run"]), "track": track,
        "phase_intent": phase, "captured_while_processes_live_ns": time.time_ns(),
        "baseline_process_document_blake3": baseline["document_blake3"],
        "thread_document_blake3": thread["document_blake3"],
        "actor": actor, "app_server": app_server, "owned_mcp": mcp,
        "raw_argv_retained": False, "provider_invoked": False}
    target.parent.mkdir(parents=True, exist_ok=True)
    return write_once(target, value)


def _matching_track_mcp_pids(paths: dict) -> list[int]:
    matches = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            argv = _argv(int(item.name))
            audit_value, workspace_value = _option(argv, "--audit-root"), _option(argv, "--workspace")
            if (audit_value is not None and workspace_value is not None
                    and Path(audit_value).resolve() == paths["audit"]
                    and Path(workspace_value).resolve() == paths["workspace"]
                    and any(value.endswith("track_entry.py") for value in argv) and "serve" in argv):
                matches.append(int(item.name))
        except PermissionError as exc:
            raise EvaluationError("recovery_process_inventory_unreadable") from exc
        except (EvaluationError, ProcessLookupError):
            continue
    return sorted(matches)


def _pid_absent(pid: int) -> bool:
    return isinstance(pid, int) and not isinstance(pid, bool) and pid > 1 and not Path(f"/proc/{pid}").exists()


def observe_post_exit_ownership(*, run_root: Path, track: str, phase: str, output: Path,
                                stability_seconds: float = 2.0) -> dict:
    """Record current process absence without inventing an unavailable old birth."""
    paths = _run_paths(run_root, track, phase)
    _validate_stability_seconds(stability_seconds)
    target = _outside_original(output, paths["run"])
    if target.exists():
        raise EvaluationError("recovery_ownership_output_already_exists")
    baseline = read_document(paths["run"] / "baseline-process.json")
    thread = read_document(paths["audit"] / "thread.json")
    exit_path = paths["run"] / "baseline-process-exit.json"
    if not exit_path.exists():
        raise EvaluationError("recovery_baseline_process_exit_receipt_missing")
    exit_receipt = read_document(exit_path)
    if (exit_receipt.get("actor_pid") != baseline.get("actor_pid")
            or exit_receipt.get("os_process_exit_observed") is not True
            or exit_receipt.get("returncode") != 0
            or not _pid_absent(baseline.get("actor_pid"))
            or not _pid_absent(thread.get("app_server_pid"))):
        raise EvaluationError("recovery_post_exit_actor_or_app_server_not_absent")
    matches = _matching_track_mcp_pids(paths)
    if matches:
        raise EvaluationError("recovery_post_exit_matching_mcp_still_present")
    source = _source_evidence(paths)
    with tempfile.TemporaryDirectory(prefix="eva-policy-post-exit-observe-") as temporary:
        inventory = MutableInventory(paths["workspace"], Path(temporary))
        first = inventory.capture()
        time.sleep(stability_seconds)
        second = inventory.capture()
    if (_stable_core(first) != _stable_core(second)
            or _stable_core(second) != source["last_host_workspace"]):
        raise EvaluationError("recovery_post_exit_workspace_not_stable_at_last_host_result")
    value = {"schema": POST_EXIT_OWNERSHIP_SCHEMA, "run_root": str(paths["run"]),
        "track": track, "phase_intent": phase, "observed_after_process_exit_ns": time.time_ns(),
        "baseline_process_document_blake3": baseline["document_blake3"],
        "baseline_process_exit_document_blake3": exit_receipt["document_blake3"],
        "thread_document_blake3": thread["document_blake3"],
        "actor_pid": baseline["actor_pid"], "original_app_server_pid": thread["app_server_pid"],
        "actor_pid_absent": True, "original_app_server_pid_absent": True,
        "matching_track_mcp_process_count": 0,
        "source_receipt_blake3": source["receipt"].receipt_blake3,
        "observed_stable_inventory_blake3": blake3_hex(_stable_core(second)),
        "historical_pid_birth_capture_available": False,
        "historical_process_quiescence_claimed": False, "raw_argv_retained": False,
        "provider_invoked": False}
    target.parent.mkdir(parents=True, exist_ok=True)
    return write_once(target, value)


def _birth_is_gone(identity: dict) -> bool:
    try:
        fields = Path(f'/proc/{identity["pid"]}/stat').read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return True
    return len(fields) < 20 or fields[19] != identity["start_ticks"]


def _require_births_gone(ownership: dict) -> None:
    for role in ("actor", "app_server", "owned_mcp"):
        identity = ownership.get(role)
        if not isinstance(identity, dict) or not _birth_is_gone(identity):
            raise EvaluationError("recovery_owned_process_birth_still_present")


def _read_receipt(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024**2:
        raise EvaluationError("recovery_receipt_topology_or_size_invalid")
    value = json.loads(path.read_bytes())
    receipt = codex_turn_receipt_from_document(value)
    verify_codex_turn_receipt(receipt)
    return receipt


def _source_evidence(paths: dict) -> dict:
    turn, audit = paths["turn"], paths["audit"]
    if (turn / "after").exists():
        raise EvaluationError("recovery_original_after_snapshot_present")
    phase_index = PHASES.index(turn.name)
    turn_names = sorted(path.name for path in (audit / "turns").iterdir() if path.is_dir())
    if turn_names != sorted(PHASES[:phase_index + 1]):
        raise EvaluationError("recovery_later_turn_directory_present")
    receipt_path = turn / "receipt.json"
    receipt = _read_receipt(receipt_path)
    request = read_document(turn / "request.json", maximum=16 * 1024**2)
    terminal = read_document(turn / "policy-budget-terminal.json")
    failure = read_document(turn / "failure.json")
    if (receipt.status != "interrupted" or any(event.method == "error" for event in receipt.events)
            or receipt.visibility != "actor-public"):
        raise EvaluationError("recovery_original_turn_not_cleanly_interrupted")
    logical = request.get("logical_input")
    if (not isinstance(logical, dict) or logical.get("judge_only_context") is not None
            or receipt.input_blake3 != blake3_hex(logical)
            or receipt.offered_tool_schema_blake3 != blake3_hex(logical.get("offered_tools"))
            or receipt.selected_skill_catalog_blake3 != blake3_hex(logical.get("skills"))):
        raise EvaluationError("recovery_original_request_binding_invalid")
    offered = tuple("automed_eval/" + row["name"] for row in request.get("public_tool_catalog", ()))
    if receipt.offered_mcp_tool_names != offered:
        raise EvaluationError("recovery_original_tool_catalog_invalid")
    joined = require_joined_host_results(receipt, audit)
    host_path = audit / "mcp-events.jsonl"
    hosts = [json.loads(line) for line in host_path.read_bytes().splitlines()]
    all_host_ids = [row.get("event_id") for row in hosts]
    if len(set(all_host_ids)) != len(all_host_ids):
        raise EvaluationError("recovery_host_event_inventory_invalid")
    current_ids = set(joined["joined_host_event_ids"])
    turn_hosts = [row for row in hosts if row.get("event_id") in current_ids]
    control_count = joined.get("native_control_call_count", 0)
    if (joined["joined_host_event_ids"] != sorted(row["event_id"] for row in turn_hosts)
            or joined["joined_host_call_count"] != len(turn_hosts)
            or joined["joined_host_call_count"] + control_count != len(receipt.tool_calls)):
        raise EvaluationError("recovery_native_control_data_partition_inexact")
    if not turn_hosts:
        raise EvaluationError("recovery_host_workspace_chain_absent")
    if any(call.tool_type != "mcpToolCall" for call in receipt.tool_calls):
        raise EvaluationError("recovery_non_mcp_action_present")
    budget = terminal.get("policy_budget")
    archival_failure = (failure.get("error"), terminal.get("infrastructure_error"))
    times = () if not isinstance(budget, dict) else tuple(
        budget.get(key) for key in ("interrupt_requested_ns", "interrupt_acknowledged_ns", "terminal_observed_ns"))
    if (terminal.get("schema") != "eva.automedbench-policy-budget-terminal.v1"
            or terminal.get("receipt_blake3") != receipt.receipt_blake3
            or terminal.get("actual_terminal_status") != "interrupted"
            or terminal.get("workspace_quiescence_verified") is not False
            or terminal.get("later_stages_evaluated") is not False
            or terminal.get("reward") is not None
            or archival_failure not in {("AttributeError", "AttributeError"),
                ("EvaluationError", "policy_terminal_host_worker_quiescence_unproved")}
            or not isinstance(budget, dict) or budget.get("schema") != "eva.codex-policy-turn-budget.v1"
            or budget.get("receipt_blake3") != receipt.receipt_blake3
            or budget.get("turn_id") != receipt.turn_id or budget.get("budget_exhausted") is not True
            or budget.get("actual_terminal_status") != "interrupted"
            or budget.get("infrastructure_error") is not None or budget.get("reward") is not None
            or any(isinstance(value, bool) or not isinstance(value, int) for value in times)
            or not times[0] <= times[1] <= times[2]
            or failure.get("phase") != turn.name
            or failure.get("reward") is not None):
        raise EvaluationError("recovery_budget_control_evidence_invalid")
    prior_receipts = {}
    used_host_ids = set(current_ids)
    for prior_phase in PHASES[:phase_index]:
        prior_path = audit / "turns" / prior_phase / "receipt.json"
        prior = _read_receipt(prior_path)
        if prior.status != "completed" or any(event.method == "error" for event in prior.events):
            raise EvaluationError("recovery_prior_turn_not_completed")
        prior_joined = require_joined_host_results(prior, audit)
        prior_ids = set(prior_joined["joined_host_event_ids"])
        if (used_host_ids & prior_ids
                or prior_joined["joined_host_call_count"]
                    + prior_joined.get("native_control_call_count", 0) != len(prior.tool_calls)):
            raise EvaluationError("recovery_prior_turn_partition_invalid")
        used_host_ids.update(prior_ids)
        prior_receipts[prior_phase] = {"receipt_blake3": prior.receipt_blake3,
                                      "receipt_file_blake3": file_digest(prior_path)}
    if used_host_ids != set(all_host_ids):
        raise EvaluationError("recovery_track_host_events_not_exactly_partitioned")
    model_job_root = audit / "model-jobs"
    public_job_root = paths["workspace"] / "outputs/agents_outputs/prescribed-model-jobs"
    audit_job_files = sorted(path for path in model_job_root.rglob("*") if path.is_file()) if model_job_root.exists() else []
    public_job_files = sorted(path for path in public_job_root.rglob("*") if path.is_file()) if public_job_root.exists() else []
    ledger = paths["workspace"] / "notes/model-jobs.jsonl"
    submitted = [call for call in receipt.tool_calls if call.mcp_tool in {
        "automed_submit_model_job", "automed_submit_extended_model_job",
        "automed_submit_generative_model_job"}]
    if audit_job_files or public_job_files or submitted or (ledger.exists() and ledger.stat().st_size):
        raise EvaluationError("recovery_async_model_job_evidence_present")
    before, before_manifest = track_snapshot(turn / "before", "original-before")
    try:
        before_inventory = {key: before_manifest[key] for key in turn_hosts[0]["workspace_before"]}
        chained = (_stable_core(turn_hosts[0]["workspace_before"]) == _stable_core(before_inventory)
            and all(_stable_core(left["workspace_after"]) == _stable_core(right["workspace_before"])
                    for left, right in zip(turn_hosts, turn_hosts[1:])))
        last_host_workspace = _stable_core(turn_hosts[-1]["workspace_after"])
    except (KeyError, TypeError) as exc:
        raise EvaluationError("recovery_host_workspace_chain_invalid") from exc
    if not chained:
        raise EvaluationError("recovery_host_workspace_chain_invalid")
    return {"receipt": receipt, "request": request, "terminal": terminal, "failure": failure,
        "joined": joined, "before": before, "before_manifest": before_manifest,
        "last_host_workspace": last_host_workspace,
        "archival_failure": {"failure_error": archival_failure[0],
                             "terminal_infrastructure_error": archival_failure[1]},
        "source_commitments": {"receipt_file_blake3": file_digest(receipt_path),
            "receipt_blake3": receipt.receipt_blake3,
            "request_document_blake3": request["document_blake3"],
            "before_manifest_document_blake3": before_manifest["document_blake3"],
            "policy_budget_terminal_document_blake3": terminal["document_blake3"],
            "failure_document_blake3": failure["document_blake3"],
            "mcp_events_file_blake3": file_digest(host_path),
            "prior_receipts": prior_receipts}}


def _stable_core(value: dict) -> dict:
    return {key: item for key, item in value.items()
            if key not in {"observation_started_ns", "observation_completed_ns"}}


def _validate_stability_seconds(seconds: float) -> None:
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not 0 <= seconds <= 30):
        raise EvaluationError("recovery_stability_interval_invalid")


def _materialize_snapshot(inventory: MutableInventory, root: Path, observed: dict) -> dict:
    target = root / "after"
    target.mkdir(mode=0o700)
    for row in observed["files"]:
        destination = target / "files" / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.link(inventory.blobs / row["blake3"], destination)
    return write_once(target / "manifest.json",
                      {"schema": "eva.automedbench-track-snapshot.v1", **observed})


def _stable_snapshot(workspace: Path, root: Path, seconds: float) -> tuple[dict, dict]:
    _validate_stability_seconds(seconds)
    inventory = MutableInventory(workspace, root)
    first = inventory.capture()
    time.sleep(seconds)
    second = inventory.capture()
    first_core, second_core = _stable_core(first), _stable_core(second)
    if first_core != second_core:
        raise EvaluationError("recovery_workspace_not_stable")
    _materialize_snapshot(inventory, root, second)
    return write_once(root / "stability-observations.json", {"schema": STABILITY_SCHEMA,
        "minimum_interval_seconds": seconds, "first": first, "second": second,
        "stable_inventory_blake3": blake3_hex(first_core)}), second_core


def _ownership(paths: dict, path: Path) -> dict:
    value = read_document(_outside_original(path, paths["run"]))
    baseline = read_document(paths["run"] / "baseline-process.json")
    thread = read_document(paths["audit"] / "thread.json")
    if value.get("schema") == POST_EXIT_OWNERSHIP_SCHEMA:
        exit_receipt = read_document(paths["run"] / "baseline-process-exit.json")
        if (value.get("run_root") != str(paths["run"])
                or value.get("track") != paths["audit"].name
                or value.get("phase_intent") != paths["turn"].name
                or value.get("baseline_process_document_blake3") != baseline["document_blake3"]
                or value.get("baseline_process_exit_document_blake3") != exit_receipt["document_blake3"]
                or value.get("thread_document_blake3") != thread["document_blake3"]
                or exit_receipt.get("actor_pid") != baseline.get("actor_pid")
                or exit_receipt.get("os_process_exit_observed") is not True
                or exit_receipt.get("returncode") != 0
                or value.get("actor_pid") != baseline.get("actor_pid")
                or value.get("original_app_server_pid") != thread.get("app_server_pid")
                or not _pid_absent(value.get("actor_pid"))
                or not _pid_absent(value.get("original_app_server_pid"))
                or _matching_track_mcp_pids(paths)
                or value.get("actor_pid_absent") is not True
                or value.get("original_app_server_pid_absent") is not True
                or value.get("matching_track_mcp_process_count") != 0
                or value.get("historical_pid_birth_capture_available") is not False
                or value.get("historical_process_quiescence_claimed") is not False
                or value.get("raw_argv_retained") is not False
                or value.get("provider_invoked") is not False):
            raise EvaluationError("recovery_post_exit_ownership_binding_invalid")
        return value
    identities = tuple(value.get(role) for role in ("actor", "app_server", "owned_mcp"))
    valid_identities = all(isinstance(item, dict)
        and isinstance(item.get("pid"), int) and not isinstance(item.get("pid"), bool)
        and item["pid"] > 1 and isinstance(item.get("parent_pid"), int)
        and isinstance(item.get("start_ticks"), str) and item["start_ticks"].isdigit()
        and isinstance(item.get("argv_blake3"), str) and len(item["argv_blake3"]) == 64
        for item in identities)
    if (value.get("schema") != OWNERSHIP_SCHEMA or value.get("run_root") != str(paths["run"])
            or value.get("track") != paths["audit"].name or value.get("phase_intent") != paths["turn"].name
            or value.get("baseline_process_document_blake3") != baseline["document_blake3"]
            or value.get("thread_document_blake3") != thread["document_blake3"]
            or value.get("actor", {}).get("pid") != baseline.get("actor_pid")
            or value.get("app_server", {}).get("pid") != thread.get("app_server_pid")
            or not valid_identities
            or identities[1]["parent_pid"] != identities[0]["pid"]
            or identities[2]["parent_pid"] != identities[1]["pid"]
            or identities[0]["argv_blake3"] != blake3_hex(tuple(baseline.get("command", ())))
            or value.get("raw_argv_retained") is not False or value.get("provider_invoked") is not False):
        raise EvaluationError("recovery_ownership_binding_invalid")
    _require_births_gone(value)
    return value


def _ownership_claims(owner: dict) -> dict:
    post_exit = owner["schema"] == POST_EXIT_OWNERSHIP_SCHEMA
    return {"ownership_observation_mode":
                "post_exit_process_absence" if post_exit else "pre_exit_pid_births",
            "historical_pid_birth_capture_available": not post_exit,
            "pre_exit_pid_births_verified_gone": not post_exit,
            "post_exit_actor_pid_absent": post_exit,
            "post_exit_original_app_server_pid_absent": post_exit,
            "matching_track_mcp_processes_absent": True,
            **({"all_owned_process_births_gone": True} if not post_exit else {})}


def build_recovery(*, run_root: Path, track: str, phase: str, ownership_path: Path,
                   output: Path, stability_seconds: float = 2.0) -> dict:
    paths = _run_paths(run_root, track, phase)
    target = _outside_original(output, paths["run"])
    if target.exists():
        raise EvaluationError("recovery_output_already_exists")
    owner = _ownership(paths, ownership_path)
    source = _source_evidence(paths)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".policy-recovery-", dir=target.parent))
    try:
        retained_owner = temporary / "ownership.json"
        shutil.copyfile(_outside_original(ownership_path, paths["run"]), retained_owner)
        retained_owner.chmod(0o600)
        if read_document(retained_owner) != owner:
            raise EvaluationError("recovery_retained_ownership_binding_invalid")
        observations, stable = _stable_snapshot(paths["workspace"], temporary, stability_seconds)
        if stable != source["last_host_workspace"]:
            raise EvaluationError("recovery_workspace_changed_since_last_host_result")
        if (owner["schema"] == POST_EXIT_OWNERSHIP_SCHEMA
                and (owner.get("source_receipt_blake3") != source["receipt"].receipt_blake3
                     or owner.get("observed_stable_inventory_blake3") != blake3_hex(stable))):
            raise EvaluationError("recovery_post_exit_observation_changed")
        after, after_manifest = track_snapshot(temporary / "after", "recovery-current-after")
        if (after_manifest["immutable_inputs_manifest_blake3"]
                != source["before_manifest"]["immutable_inputs_manifest_blake3"]):
            raise EvaluationError("recovery_input_binding_changed")
        value = {"schema": RECOVERY_SCHEMA, "run_root": str(paths["run"]), "track": track,
            "phase_intent": phase, "ownership_document_blake3": owner["document_blake3"],
            "source_commitments": source["source_commitments"],
            "original_archival_failure": source["archival_failure"],
            "actual_terminal_status": "interrupted", "receipt_error_event_count": 0,
            "actual_tool_call_count": len(source["receipt"].tool_calls),
            "joined_host_call_count": source["joined"]["joined_host_call_count"],
            "native_control_call_count": source["joined"].get("native_control_call_count", 0),
            "native_control_operations": source["joined"].get("native_control_operations", []),
            "original_before_tree_blake3": source["before"].tree_blake3,
            "last_host_workspace_inventory_blake3": blake3_hex(source["last_host_workspace"]),
            "recovery_current_after_tree_blake3": after.tree_blake3,
            "recovery_after_manifest_document_blake3": after_manifest["document_blake3"],
            "stability_observations_document_blake3": observations["document_blake3"],
            "stable_inventory_blake3": blake3_hex(stable),
            "recovery_observed_ns": time.time_ns(), **_ownership_claims(owner),
            "async_model_jobs_present": False, "original_after_snapshot_missing": True,
            "recovery_snapshot_is_original_terminal_snapshot": False,
            "workspace_quiescence_at_original_terminal_claimed": False,
            "current_workspace_stability_observed": True, "provider_replayed": False,
            "turn_retried": False, "later_stages_evaluated": False, "reward": None,
            "judge_use_requires_explicit_current_after_semantics": True}
        result = write_once(temporary / "recovery.json", value)
        verify_recovery(run_root=paths["run"], track=track, phase=phase,
                        ownership_path=retained_owner, recovery_root=temporary)
        os.replace(temporary, target)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_recovery(*, run_root: Path, track: str, phase: str, ownership_path: Path,
                    recovery_root: Path) -> dict:
    paths = _run_paths(run_root, track, phase)
    root = _outside_original(recovery_root, paths["run"])
    owner = _ownership(paths, ownership_path)
    source = _source_evidence(paths)
    value = read_document(root / "recovery.json", maximum=16 * 1024**2)
    observations = read_document(root / "stability-observations.json", maximum=16 * 1024**2)
    after, after_manifest = track_snapshot(root / "after", "recovery-current-after")
    first_core, second_core = _stable_core(observations.get("first", {})), _stable_core(observations.get("second", {}))
    with tempfile.TemporaryDirectory(prefix="eva-policy-recovery-verify-") as temporary:
        current = MutableInventory(paths["workspace"], Path(temporary)).capture()
    manifest_inventory = {key: after_manifest[key] for key in observations.get("second", {})}
    if (observations.get("schema") != STABILITY_SCHEMA or first_core != second_core
            or observations.get("stable_inventory_blake3") != blake3_hex(first_core)
            or second_core != _stable_core(manifest_inventory)
            or second_core != _stable_core(current)):
        raise EvaluationError("recovery_stability_binding_invalid")
    expected = {"schema": RECOVERY_SCHEMA, "run_root": str(paths["run"]), "track": track,
        "phase_intent": phase, "ownership_document_blake3": owner["document_blake3"],
        "source_commitments": source["source_commitments"],
        "original_archival_failure": source["archival_failure"],
        "actual_terminal_status": "interrupted",
        "receipt_error_event_count": 0, "actual_tool_call_count": len(source["receipt"].tool_calls),
        "joined_host_call_count": source["joined"]["joined_host_call_count"],
        "native_control_call_count": source["joined"].get("native_control_call_count", 0),
        "native_control_operations": source["joined"].get("native_control_operations", []),
        "original_before_tree_blake3": source["before"].tree_blake3,
        "last_host_workspace_inventory_blake3": blake3_hex(source["last_host_workspace"]),
        "recovery_current_after_tree_blake3": after.tree_blake3,
        "recovery_after_manifest_document_blake3": after_manifest["document_blake3"],
        "stability_observations_document_blake3": observations["document_blake3"],
        "stable_inventory_blake3": observations["stable_inventory_blake3"]}
    for key, item in expected.items():
        if value.get(key) != item:
            raise EvaluationError("recovery_document_binding_invalid")
    if second_core != source["last_host_workspace"]:
        raise EvaluationError("recovery_workspace_changed_since_last_host_result")
    ownership_claims = _ownership_claims(owner)
    if any(value.get(key) != item for key, item in ownership_claims.items()):
        raise EvaluationError("recovery_ownership_semantics_invalid")
    if (owner["schema"] == POST_EXIT_OWNERSHIP_SCHEMA
            and (owner.get("source_receipt_blake3") != source["receipt"].receipt_blake3
                 or owner.get("observed_stable_inventory_blake3") != blake3_hex(second_core))):
        raise EvaluationError("recovery_post_exit_observation_changed")
    fixed = {"async_model_jobs_present": False,
        "original_after_snapshot_missing": True, "recovery_snapshot_is_original_terminal_snapshot": False,
        "workspace_quiescence_at_original_terminal_claimed": False,
        "current_workspace_stability_observed": True, "provider_replayed": False,
        "turn_retried": False, "later_stages_evaluated": False, "reward": None,
        "judge_use_requires_explicit_current_after_semantics": True}
    if any(value.get(key) != item for key, item in fixed.items()):
        raise EvaluationError("recovery_semantics_invalid")
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture-ownership")
    observe = subparsers.add_parser("observe-post-exit-ownership")
    build = subparsers.add_parser("build")
    verify = subparsers.add_parser("verify")
    for command in (capture, observe, build, verify):
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--track", required=True)
        command.add_argument("--phase", default="01-planning")
    capture.add_argument("--output", type=Path, required=True)
    observe.add_argument("--output", type=Path, required=True)
    observe.add_argument("--stability-seconds", type=float, default=2.0)
    for command in (build, verify):
        command.add_argument("--ownership", type=Path, required=True)
        command.add_argument("--recovery-root", type=Path, required=True)
    build.add_argument("--stability-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.command == "capture-ownership":
        result = capture_ownership(run_root=args.run_root, track=args.track, phase=args.phase,
                                   output=args.output)
    elif args.command == "observe-post-exit-ownership":
        result = observe_post_exit_ownership(run_root=args.run_root, track=args.track,
            phase=args.phase, output=args.output, stability_seconds=args.stability_seconds)
    elif args.command == "build":
        result = build_recovery(run_root=args.run_root, track=args.track, phase=args.phase,
            ownership_path=args.ownership, output=args.recovery_root,
            stability_seconds=args.stability_seconds)
    else:
        result = verify_recovery(run_root=args.run_root, track=args.track, phase=args.phase,
            ownership_path=args.ownership, recovery_root=args.recovery_root)
    print(json.dumps({"schema": result["schema"], "document_blake3": result["document_blake3"]}))


if __name__ == "__main__":
    main()
