"""Observe one live post-round server and run the v1.3 note probe once.

This module never starts or stops a model server.  It waits for the durable RSI
loop to expose a full seven-track evaluation using at most three actor lanes,
then reuses that evaluation's live checkpoint identity and canary for one
diagnostic invocation.  The diagnostic is supplementary: failure is retained
as unavailable/model failure and is never converted into a benchmark score.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from uuid import UUID

from blake3 import blake3


class ObservationUnavailable(ValueError):
    """A fixed-code readiness failure; never include provider text."""


@dataclass(frozen=True)
class ReadyEvaluation:
    attempt_id: str
    attempt_root: Path
    identity_path: Path
    canary_path: Path
    codex_bin: Path
    turn_timeout: int
    evidence: dict


def _read(path: Path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ObservationUnavailable("observer_document_topology_or_size_invalid")
    body = path.read_bytes()
    return json.loads(body), {"path": str(path.absolute()), "blake3": blake3(body).hexdigest()}


def _require(value, code):
    if not value:
        raise ObservationUnavailable(code)


def bound_benchmark_root(attempt: Path, runtime: dict) -> Path:
    """Use the producer's actual run argument, never a latest-directory guess."""
    argv = runtime.get("equivalent_actor_cli")
    _require(isinstance(argv, list) and all(isinstance(value, str) for value in argv)
             and argv.count("--run-root") == 1
             and not any(value.startswith("--run-root=") for value in argv),
             "observer_benchmark_run_argument_missing_or_ambiguous")
    offset = argv.index("--run-root") + 1
    _require(offset < len(argv), "observer_benchmark_run_argument_missing_or_ambiguous")
    run = Path(argv[offset])
    try:
        canonical_uuid = str(UUID(run.name)) == run.name
    except ValueError:
        canonical_uuid = False
    _require(run.is_absolute() and str(run) == argv[offset] and canonical_uuid
             and run.parent == attempt / "benchmark" and not run.parent.is_symlink()
             and not run.is_symlink() and run.is_dir(),
             "observer_benchmark_run_not_owned_uuid")
    return run


def inspect_ready(loop_root: Path, settings_path: Path) -> ReadyEvaluation | None:
    """Return a fully bound live evaluation, or None while it is not ready."""
    from training.automedbench_lite.actor import serving_binding
    from training.automedbench_lite.track_adapter import BY_TRACK
    from training.eva_rsi.production_eval import actor_profile, evaluation_scope

    loop = Path(loop_root).absolute()
    settings_path = Path(settings_path).absolute()
    settings, settings_ref = _read(settings_path)
    scope = evaluation_scope(settings)
    profile = actor_profile(settings, scope)
    _require(scope["mode"] == "full_single_pass" and set(scope["tracks"]) == set(BY_TRACK),
             "observer_requires_full_seven_track_evaluation")
    _require(profile["name"] == "supra-v1.3" and profile["args"]["workers"] <= 3,
             "observer_requires_spare_evaluation_lane")
    state_path = loop / "state.json"
    state, state_ref = _read(state_path)
    _require(state.get("schema") == "eva.rsi-loop-state.v1", "observer_loop_state_schema_invalid")
    if state.get("phase") != "evaluation" or not state.get("active_attempt"):
        return None
    attempt_id = state["active_attempt"]
    matches = [row for row in state.get("attempts", ()) if row.get("attempt_id") == attempt_id]
    _require(len(matches) == 1 and matches[0].get("phase") == "evaluation"
             and matches[0].get("status") == "running", "observer_active_evaluation_identity_invalid")
    attempt = Path(matches[0]["root"]).absolute()
    _require(attempt.parent == (loop / "attempts").absolute() and attempt.name == attempt_id,
             "observer_attempt_not_owned_by_loop")
    context, context_ref = _read(attempt / "context.json")
    _require(context.get("phase") == "evaluation" and context.get("attempt_root") == str(attempt)
             and context.get("loop_id") == state.get("loop_id"),
             "observer_evaluation_context_differs")
    identity_path = attempt / "serving/checkpoint-identity.json"
    canary_path = attempt / "serving/image-canary.json"
    request_path = attempt / "request.json"
    runtime_path = attempt / "evaluation-actor-binding.json"
    if not all(path.is_file() for path in (identity_path, canary_path, request_path, runtime_path)):
        return None
    request, request_ref = _read(request_path)
    argv = request.get("argv", ())
    _require(request.get("phase") == "evaluation" and isinstance(argv, list)
             and argv.count("--settings") == 1
             and argv.index("--settings") + 1 < len(argv)
             and argv[argv.index("--settings") + 1] == str(settings_path),
             "observer_evaluation_settings_not_command_bound")
    runtime, runtime_ref = _read(runtime_path)
    selected_harness = os.environ.get("EVA_HARNESS_ROOT")
    _require(selected_harness and runtime.get("selected_harness_root")
             and Path(selected_harness).resolve(strict=True)
             == Path(runtime["selected_harness_root"]).resolve(strict=True)
             and runtime.get("profile_requested") == profile,
             "observer_selected_harness_or_profile_differs")
    benchmark_run = bound_benchmark_root(attempt, runtime)
    actor_path = benchmark_run / "track-rollouts/attempt.json"
    if not actor_path.is_file():
        return None
    actor, actor_ref = _read(actor_path)
    _require(actor.get("schema") == "eva.automedbench-track-attempt.v1"
             and actor.get("full_seven_track_evaluation") is True
             and set(actor.get("tracks", ())) == set(BY_TRACK)
             and actor.get("max_parallel_tracks") == profile["args"]["workers"],
             "observer_actor_coverage_or_lane_binding_differs")
    binding = serving_binding(canary_path, identity_path)
    actor_server = actor.get("server_binding", {})
    _require(actor_server.get("identity_file_blake3") == binding["identity_file_blake3"]
             and actor_server.get("canary_file_blake3") == binding["canary_file_blake3"],
             "observer_actor_server_binding_differs")
    identity = binding["identity"]
    _require(Path(identity.get("exact_final_model_path", "")).absolute()
             == Path(context.get("model_path", "")).absolute()
             and Path(identity.get("checkpoint_root", "")).absolute()
             == Path(context.get("checkpoint_root", "")).absolute(),
             "observer_live_checkpoint_context_differs")
    _require(identity.get("settings", {}).get("port") == settings.get("evaluation_port", 30911)
             and identity.get("settings", {}).get("context_length") == profile["args"]["context_length"],
             "observer_live_server_profile_differs")
    codex_bin = Path(settings["codex_bin"]).absolute()
    timeout = settings.get("evaluation_turn_timeout_seconds", 900)
    evidence = {"schema": "eva.supra-note-retention-postround-observer.v1",
        "attempt_id": attempt_id, "loop_state": state_ref, "settings": settings_ref,
        "evaluation_context": context_ref, "evaluation_request": request_ref,
        "benchmark_run_root": str(benchmark_run),
        "actor_runtime_binding": runtime_ref, "track_attempt": actor_ref,
        "checkpoint_identity": {"path": str(identity_path),
            "blake3": binding["identity_file_blake3"]},
        "server_canary": {"path": str(canary_path), "blake3": binding["canary_file_blake3"]},
        "evaluation_workers": profile["args"]["workers"], "diagnostic_workers": 1,
        "same_static_checkpoint": True, "benchmark_workspace_modified": False,
        "medical_evaluation": False, "automatic_retry": False}
    return ReadyEvaluation(attempt_id, attempt, identity_path, canary_path, codex_bin, timeout, evidence)


def wait_ready(loop_root: Path, settings_path: Path, *, wait_seconds: int = 86400,
               poll_seconds: float = 1.0) -> ReadyEvaluation:
    if type(wait_seconds) is not int or not 1 <= wait_seconds <= 604800:
        raise ObservationUnavailable("observer_wait_budget_invalid")
    if not isinstance(poll_seconds, (int, float)) or not 0.1 <= poll_seconds <= 30:
        raise ObservationUnavailable("observer_poll_interval_invalid")
    deadline = time.monotonic() + wait_seconds
    while True:
        ready = inspect_ready(loop_root, settings_path)
        if ready is not None:
            # Close the state/canary race immediately before the sole invocation.
            confirmation = inspect_ready(loop_root, settings_path)
            if confirmation is not None and confirmation.attempt_id == ready.attempt_id:
                return confirmation
        if time.monotonic() >= deadline:
            raise ObservationUnavailable("observer_wait_expired_without_live_evaluation")
        time.sleep(poll_seconds)


async def run_once(ready: ReadyEvaluation, output: Path):
    """Invoke the existing two-turn probe once; no retry and no score promotion."""
    from training.context_memory import note_retention_v13 as probe

    output = Path(output).absolute()
    require_fresh_output(output)
    if output.is_relative_to(ready.attempt_root):
        raise ObservationUnavailable("observer_output_must_be_outside_evaluation_attempt")
    try:
        result = await probe.run(output, ready.identity_path, ready.canary_path,
                                 ready.codex_bin, timeout=ready.turn_timeout)
    except Exception as exc:
        result = {**probe._empty_result(), "reason": "runner_or_transport_failure",
                  "error_type": type(exc).__name__, "automatic_retry": False}
        if output.is_dir() and not (output / "failure.json").exists():
            probe.write_once(output / "failure.json", result)
    if output.is_dir():
        probe.write_once(output / "postround-observer.json", ready.evidence)
    return result


def require_fresh_output(output: Path):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ObservationUnavailable("observer_output_must_be_fresh")


def run_observer(loop_root: Path, settings_path: Path, output: Path, *,
                 wait_seconds: int = 86400, poll_seconds: float = 1.0):
    require_fresh_output(output)  # Fail before an otherwise long readiness wait.
    ready = wait_ready(loop_root, settings_path, wait_seconds=wait_seconds, poll_seconds=poll_seconds)
    return asyncio.run(run_once(ready, output))
