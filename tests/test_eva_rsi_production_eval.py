"""CPU integration fixtures: no providers, subprocesses, serving, or GPU calls.

Codex source events are simulated; mock Judge grades are schema fixtures, not
medical scores. The real track projection, missing-stage handling, index writer
and RSI evidence consumer are exercised around those explicit boundaries.
"""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from eva_agent.pipeline.digests import blake3_bytes
from eva_agent.rubrics import load_and_compile_registry
from training.automedbench_lite import track_actor, track_adapter, track_feedback
from training.benchmark_feedback import automed_codex
from training.eva_rsi import production_eval, serving
from test_automedbench_track_feedback import fixture as retained_fixture


@pytest.fixture
def integration(tmp_path, monkeypatch):
    source = retained_fixture(tmp_path / "simulated-codex-source")
    run = source["run_root"]
    identity_bytes = source["checkpoint_identity"].read_bytes()
    checkpoint_id = blake3_bytes(identity_bytes)
    context = {"attempt_root": str(tmp_path / "attempt"), "model_path": "/fixture/model",
        "checkpoint_root": "/fixture/checkpoints", "architecture_model_path": "/fixture/architecture",
        "skill_catalog_id": "b" * 64}
    settings = {"cds_receipt": "/fixture/cds-receipt.json", "missing4_receipt": "/fixture/missing4-receipt.json",
        "public_image": "/fixture/public.jpg", "public_model_runtime": "/fixture/models.json",
        "training_python": "/fixture/scientific-venv/bin/python", "cpu_image": "fixture-immutable-image",
        "codex_bin": "/fixture/codex-0.153.4/bin/codex"}
    events, captures = [], {}
    state = {"serving": False, "actor_error": None, "cleanup_error": None, "judge_error": None}
    monkeypatch.setattr(track_adapter, "TrackRelease", lambda path: SimpleNamespace(receipt_path=path))
    def prepare(releases, output_root):
        assert set(releases) == set(track_adapter.BY_TRACK)
        assert all(releases[name].receipt_path == Path(settings["cds_receipt"])
                   for name in ("classification", "detection", "segmentation"))
        assert all(releases[name].receipt_path == Path(settings["missing4_receipt"])
                   for name in ("vqa", "report", "synthesis", "enhancement"))
        captures["prepare_root"] = output_root
        events.append("prepare_public_inputs")
        return run
    monkeypatch.setattr(track_adapter, "prepare_track_run", prepare)

    @contextmanager
    def session(model_path, checkpoint_root, architecture_model_path, output_root, public_image, port=30910,
                python_executable=None, context_length=32768):
        captures["server"] = dict(model_path=model_path, checkpoint_root=checkpoint_root,
            architecture_model_path=architecture_model_path, output_root=output_root, public_image=public_image, port=port,
            python_executable=python_executable,context_length=context_length)
        output_root.mkdir(parents=True)
        identity = output_root / "checkpoint-identity.json"
        identity.write_bytes(identity_bytes)
        canary = output_root / "image-canary.json"
        canary.write_text(json.dumps({"fixture_only": True, "checkpoint_identity_blake3": checkpoint_id}))
        events.append("start_owned_server")
        state["serving"] = True
        try:
            yield serving.ServingSession(identity, canary, output_root, 424242)
        finally:
            events.append("stop_owned_server")
            state["serving"] = False
            (output_root / "session-exit.json").write_text(json.dumps({"fixture_only": True,
                "cleanup": {"verified_members_remaining": 0, "unrelated_processes_signalled": False}}))
            if state["cleanup_error"]: raise state["cleanup_error"]
    monkeypatch.setattr(serving, "serving_session", session)

    async def actor(args):
        assert state["serving"]
        captures["actor"] = args
        events.append("actual_actor_boundary_mock")
        if state["actor_error"]: raise state["actor_error"]
        (run / "track-rollouts/summary.json").write_text(json.dumps({"fixture_only": True,
            "completed_requested_workflows": 0, "retained_complete_stage": "S1", "missing_stages": ["S2", "S3"]}))
    monkeypatch.setattr(track_actor, "run_tracks", actor)

    rubric = load_and_compile_registry(Path(__file__).resolve().parents[1] / "rubrics/source/domain-stage-tables.v1.json").resolve("automedbench-classification", "S1")
    score = rubric.score({item["item_id"]: 0 for item in rubric.items}, evaluation_id=str(uuid4())).to_document()
    def verified(root):
        assert not state["serving"]
        assert root.name == "S1"
        preflight = json.loads((root / "preflight.json").read_text())
        assert preflight["judge_eligible"] and preflight["actual_host_call_count"] == 1
        return {"valid": True, "status": "scored", "case_id": "classification", "domain": rubric.domain,
            "stage": "S1", "score": score, "source_rollout_blake3": preflight["source_rollout_blake3"],
            "round_identity": {"model_id": "fixture-Qwen", "checkpoint_id": checkpoint_id,
                "skill_catalog_id": "b" * 64, "judge_id": "c" * 64}, "judge_codex_receipt_blake3": "d" * 64}
    def judge(root):
        assert not state["serving"]
        events.append("judge_complete_s1_once_mock")
        if state["judge_error"]: raise state["judge_error"]
        return verified(root)
    monkeypatch.setattr(track_feedback, "judge_track_once", judge)
    monkeypatch.setattr(track_feedback, "verify_feedback", verified)
    monkeypatch.setattr(automed_codex, "verify_feedback", verified)
    return dict(context=context, settings=settings, captures=captures, events=events, state=state,
                run=run, identity_bytes=identity_bytes, checkpoint_id=checkpoint_id)


@pytest.mark.parametrize("port", [None, 31234])
def test_completed_s1_missing_later_stages_identity_endpoint_and_stop_order(integration, port):
    value = integration
    if port is not None: value["settings"]["evaluation_port"] = port
    report = production_eval.evaluate_round(value["context"], value["settings"])
    selected_port = 30911 if port is None else port
    args = value["captures"]["actor"]
    server = value["captures"]["server"]
    assert server["port"] == selected_port and args.endpoint == f"http://127.0.0.1:{selected_port}/v1"
    assert server["model_path"] == Path(value["context"]["model_path"])
    assert server["checkpoint_root"] == Path(value["context"]["checkpoint_root"])
    assert server["architecture_model_path"] == Path(value["context"]["architecture_model_path"])
    assert server["python_executable"] == Path(value["settings"]["training_python"])
    assert server["context_length"] == 32768
    assert args.server_identity.read_bytes() == value["identity_bytes"]
    assert args.server_canary == server["output_root"] / "image-canary.json"
    assert args.codex_bin == Path(value["settings"]["codex_bin"]) and args.tracks == ["classification"]
    assert args.public_python == Path(value["settings"]["training_python"])
    assert value["events"] == ["prepare_public_inputs", "start_owned_server", "actual_actor_boundary_mock",
                                "stop_owned_server", "judge_complete_s1_once_mock"]
    attempt = Path(value["context"]["attempt_root"])
    index = json.loads((attempt / "evaluation-index.json").read_text())
    assert index["feedback_roots"] == [str(attempt / "workspace-feedback/S1")]
    assert index["checkpoint_identity"] == str(args.server_identity)
    coverage = json.loads((attempt / "workspace-feedback/coverage.json").read_text())
    assert [row["status"] for row in coverage["stages"]] == ["independently_verified", "unknown", "unknown"]
    assert all(row["rubric_score"] is None for row in coverage["stages"][1:])
    assert len(index["diagnostic_artifacts"]) == len(report["diagnostic_artifacts"]) == 3
    assert str(attempt / "serving/session-exit.json") in index["diagnostic_artifacts"]
    assert str(attempt / "workspace-feedback/coverage.json") in index["diagnostic_artifacts"]
    assert report["checkpoint_identity"]["blake3"] == value["checkpoint_id"]
    assert len(report["feedback"]) == 1 and report["evaluation_mode"] == "diagnostic_subset"
    assert report["memory_context_pass"] is None and not index["full_seven_track_evaluation"]


@pytest.mark.parametrize("boundary", ["actor_error", "cleanup_error"])
def test_actor_or_server_cleanup_failure_never_starts_judge(integration, boundary):
    value = integration
    value["state"][boundary] = RuntimeError("explicit fixture boundary failure")
    with pytest.raises(RuntimeError, match="fixture"):
        production_eval.evaluate_round(value["context"], value["settings"])
    assert value["events"][-1] == "stop_owned_server"
    assert "judge_complete_s1_once_mock" not in value["events"]
    assert not (Path(value["context"]["attempt_root"]) / "evaluation-index.json").exists()


def test_judge_failure_stays_unknown_and_server_is_already_stopped(integration):
    value = integration
    value["state"]["judge_error"] = RuntimeError("fixture judge unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        production_eval.evaluate_round(value["context"], value["settings"])
    assert value["events"][-2:] == ["stop_owned_server", "judge_complete_s1_once_mock"]
    attempt = Path(value["context"]["attempt_root"])
    assert not (attempt / "evaluation-index.json").exists()
    assert not (attempt / "workspace-feedback/S1/feedback.json").exists()


def test_changed_checkpoint_identity_is_rejected_before_judge(integration, monkeypatch):
    value = integration
    original_actor = track_actor.run_tracks
    async def change_identity(args):
        await original_actor(args)
        args.server_identity.write_text('{"exact_final_model_path":"/wrong-fixture-model"}')
    monkeypatch.setattr(track_actor, "run_tracks", change_identity)
    with pytest.raises(ValueError, match="checkpoint_binding"):
        production_eval.evaluate_round(value["context"], value["settings"])
    assert value["events"][-1] == "stop_owned_server"
    assert "judge_complete_s1_once_mock" not in value["events"]
    assert not (Path(value["context"]["attempt_root"]) / "evaluation-index.json").exists()
