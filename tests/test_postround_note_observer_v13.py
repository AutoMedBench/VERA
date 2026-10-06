import asyncio
import json
from pathlib import Path
import sys
from uuid import uuid4

import pytest

from training.context_memory import postround_note_observer_v13 as observer


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, monkeypatch, *, workers=3):
    loop, attempt = tmp_path / "loop", tmp_path / "loop/attempts/eval-id"
    settings = tmp_path / "settings.json"
    configured = {"evaluation_mode": "full_single_pass", "evaluation_actor_profile": "supra-v1.3",
        "evaluation_workers": workers, "evaluation_context_length": 32768,
        "evaluation_compact_tokens": 20480, "evaluation_output_tokens": 4096,
        "evaluation_capacity_wait_seconds": 60, "evaluation_turn_timeout_seconds": 900,
        "evaluation_port": 30911, "codex_bin": "/opt/codex-0.153.4",
        "public_model_runtime": "/public/runtime.json", "training_python": "/public/python",
        "cpu_image": "fixture-image"}
    dump(settings, configured)
    dump(loop / "state.json", {"schema": "eva.rsi-loop-state.v1", "loop_id": "loop-id", "phase": "evaluation",
        "active_attempt": "eval-id", "attempts": [{"attempt_id": "eval-id", "phase": "evaluation",
        "status": "running", "root": str(attempt)}]})
    dump(attempt / "context.json", {"loop_id": "loop-id", "phase": "evaluation", "attempt_root": str(attempt),
        "model_path": "/models/iter_50", "checkpoint_root": "/checkpoints"})
    dump(attempt / "request.json", {"phase": "evaluation",
        "argv": ["python", "supervisor.py", "context.json", "--settings", str(settings)]})
    harness = tmp_path / "harness"
    harness.mkdir()
    monkeypatch.setenv("EVA_HARNESS_ROOT", str(harness))
    profile = {"name": "supra-v1.3", "memory_profile": False, "supra_profile": True,
        "thinking": True, "reserve_tokens": 2048, "tool_output_token_limit": None,
        "args": {"workers": workers, "context_length": 32768,
            "auto_compact_token_limit": 20480, "output_tokens": 4096,
            "adapter_capacity_wait_seconds": 60, "turn_timeout": 900}}
    benchmark = attempt / "benchmark" / str(uuid4())
    from training.eva_rsi.production_eval import actor_arguments, evaluation_scope
    _, actual_producer_argv = actor_arguments(benchmark, attempt, configured,
        evaluation_scope(configured), profile, identity_path=attempt / "serving/checkpoint-identity.json",
        canary_path=attempt / "serving/image-canary.json")
    dump(attempt / "evaluation-actor-binding.json", {"selected_harness_root": str(harness),
        "profile_requested": profile, "equivalent_actor_cli": actual_producer_argv})
    dump(attempt / "serving/checkpoint-identity.json", {"exact_final_model_path": "/models/iter_50",
        "checkpoint_root": "/checkpoints", "settings": {"port": 30911, "context_length": 32768}})
    dump(attempt / "serving/image-canary.json", {"server_pid": 123})
    from training.automedbench_lite.track_adapter import BY_TRACK
    identity_digest = observer.blake3((attempt / "serving/checkpoint-identity.json").read_bytes()).hexdigest()
    canary_digest = observer.blake3((attempt / "serving/image-canary.json").read_bytes()).hexdigest()
    dump(benchmark / "track-rollouts/attempt.json", {
        "schema": "eva.automedbench-track-attempt.v1", "full_seven_track_evaluation": True,
        "tracks": list(BY_TRACK), "max_parallel_tracks": workers,
        "server_binding": {"identity_file_blake3": identity_digest, "canary_file_blake3": canary_digest}})
    import training.automedbench_lite.actor as actor
    monkeypatch.setattr(actor, "serving_binding", lambda canary, identity: {
        "identity": json.loads(identity.read_text()), "canary": json.loads(canary.read_text()),
        "identity_file_blake3": observer.blake3(identity.read_bytes()).hexdigest(),
        "canary_file_blake3": observer.blake3(canary.read_bytes()).hexdigest()})
    return loop, settings, attempt


def test_selects_live_full_round_with_one_spare_lane(tmp_path, monkeypatch):
    loop, settings, attempt = fixture(tmp_path, monkeypatch)
    ready = observer.inspect_ready(loop, settings)
    assert ready.attempt_id == "eval-id" and ready.attempt_root == attempt
    assert ready.evidence["evaluation_workers"] == 3
    assert ready.evidence["diagnostic_workers"] == 1
    assert ready.evidence["benchmark_workspace_modified"] is False
    benchmark = Path(ready.evidence["benchmark_run_root"])
    assert benchmark.parent == attempt / "benchmark"
    assert ready.evidence["track_attempt"]["path"] == str(benchmark / "track-rollouts/attempt.json")
    assert not (attempt / "benchmark/track-rollouts/attempt.json").exists()


@pytest.mark.parametrize("kind", ["missing", "duplicate", "dangling", "non_uuid", "outside"])
def test_rejects_missing_ambiguous_or_unowned_producer_run_argument(tmp_path, monkeypatch, kind):
    loop, settings, attempt = fixture(tmp_path, monkeypatch)
    path = attempt / "evaluation-actor-binding.json"
    binding = json.loads(path.read_text())
    argv = binding["equivalent_actor_cli"]
    offset = argv.index("--run-root")
    if kind == "missing":
        del argv[offset:offset + 2]
    elif kind == "duplicate":
        argv.extend(argv[offset:offset + 2])
    elif kind == "dangling":
        del argv[offset:offset + 2]
        argv.append("--run-root")
    else:
        target = attempt / "benchmark/not-a-uuid" if kind == "non_uuid" else tmp_path / str(uuid4())
        target.mkdir(parents=True)
        argv[offset + 1] = str(target)
    dump(path, binding)
    with pytest.raises(observer.ObservationUnavailable, match="observer_benchmark_run"):
        observer.inspect_ready(loop, settings)


def test_waits_for_bound_uuid_actor_never_selects_other_directory(tmp_path, monkeypatch):
    loop, settings, attempt = fixture(tmp_path, monkeypatch)
    ready = observer.inspect_ready(loop, settings)
    actor = Path(ready.evidence["track_attempt"]["path"])
    other = attempt / "benchmark" / str(uuid4()) / "track-rollouts/attempt.json"
    dump(other, json.loads(actor.read_text()))
    actor.unlink()
    assert observer.inspect_ready(loop, settings) is None


def test_waits_before_evaluation_and_rejects_four_actor_lanes(tmp_path, monkeypatch):
    loop, settings, _ = fixture(tmp_path, monkeypatch)
    state = json.loads((loop / "state.json").read_text())
    state.update(phase="train", active_attempt=None)
    dump(loop / "state.json", state)
    assert observer.inspect_ready(loop, settings) is None
    loop, settings, _ = fixture(tmp_path / "other", monkeypatch, workers=4)
    with pytest.raises(observer.ObservationUnavailable, match="spare_evaluation_lane"):
        observer.inspect_ready(loop, settings)


def test_runs_existing_probe_once_and_retains_observer_binding(tmp_path, monkeypatch):
    loop, settings, _ = fixture(tmp_path, monkeypatch)
    ready = observer.inspect_ready(loop, settings)
    calls = []

    async def run(output, identity, canary, codex, *, timeout):
        calls.append((identity, canary, codex, timeout))
        output.mkdir()
        return {"status": "complete", "passed": True, "medical_evaluation": False}

    from training.context_memory import note_retention_v13 as probe
    monkeypatch.setattr(probe, "run", run)
    result = asyncio.run(observer.run_once(ready, tmp_path / "observation"))
    assert result["passed"] is True and len(calls) == 1
    retained = json.loads((tmp_path / "observation/postround-observer.json").read_text())
    assert retained["attempt_id"] == "eval-id" and retained["automatic_retry"] is False


def test_probe_exception_is_unavailable_without_retry(tmp_path, monkeypatch):
    loop, settings, _ = fixture(tmp_path, monkeypatch)
    ready = observer.inspect_ready(loop, settings)
    calls = []

    async def fail(output, *args, **kwargs):
        calls.append(True)
        output.mkdir()
        raise RuntimeError("private provider detail")

    from training.context_memory import note_retention_v13 as probe
    monkeypatch.setattr(probe, "run", fail)
    result = asyncio.run(observer.run_once(ready, tmp_path / "failed"))
    assert len(calls) == 1 and result["status"] == "unavailable" and result["passed"] is None
    assert "private provider detail" not in json.dumps(result)
    assert json.loads((tmp_path / "failed/failure.json").read_text())["automatic_retry"] is False


def test_refuses_to_write_inside_evaluation_attempt(tmp_path, monkeypatch):
    loop, settings, attempt = fixture(tmp_path, monkeypatch)
    ready = observer.inspect_ready(loop, settings)
    with pytest.raises(observer.ObservationUnavailable, match="outside_evaluation_attempt"):
        asyncio.run(observer.run_once(ready, attempt / "note-retention"))


def test_existing_output_is_rejected_before_wait(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(observer, "wait_ready", lambda *args, **kwargs: pytest.fail("waited"))
    with pytest.raises(observer.ObservationUnavailable, match="output_must_be_fresh"):
        observer.run_observer(tmp_path / "loop", tmp_path / "settings", output)


def test_cli_execute_invokes_observer_once(monkeypatch, tmp_path, capsys):
    import scripts.watch_note_retention_v13 as cli
    calls = []
    monkeypatch.setattr(observer, "run_observer", lambda *args, **kwargs:
        calls.append((args, kwargs)) or {"status": "complete", "passed": True, "medical_evaluation": False})
    monkeypatch.setattr(sys, "argv", ["watch_note_retention_v13.py", "--loop-root", str(tmp_path / "loop"),
        "--settings", str(tmp_path / "settings"), "--output", str(tmp_path / "output"), "--execute"])
    assert cli.main() == 0 and len(calls) == 1
    assert json.loads(capsys.readouterr().out)["passed"] is True
