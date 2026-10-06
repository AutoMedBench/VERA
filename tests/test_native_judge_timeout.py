"""CPU-only explicit native Judge deadline plumbing; never opens a provider."""
import asyncio
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from eva_agent.pipeline.digests import blake3_hex
from eva_agent.training import slime_agent_judge as engine
from eva_agent.training.agent_judge import AgentJudgeSelectionError
from training.benchmark_feedback import automed_codex as bridge
from training.automedbench_lite import track_feedback
from test_slime_agent_judge import _generic_rollout, _ProviderFreeWorkspaceJudge
from test_automedbench_track_feedback import fixture


@pytest.mark.parametrize("seconds", [None, True, False, 0, -1, 901, 600.0, "600"])
def test_invalid_native_timeout_fails_before_dispatch_or_artifact(tmp_path, monkeypatch, seconds):
    monkeypatch.setattr(engine, "_grade_rollout_sync", lambda *a, **k: pytest.fail("dispatch occurred"))
    output = tmp_path / "uncreated"
    with pytest.raises(AgentJudgeSelectionError, match="integer from 1 to 900"):
        asyncio.run(engine.grade_rollout({}, rubric=None, output_root=output, sample_id="sample",
                                        backend="native_astra", native_turn_timeout_seconds=seconds))
    assert not output.exists()


def test_timeout_override_is_not_an_opus_option(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "_grade_rollout_sync", lambda *a, **k: pytest.fail("Opus dispatch occurred"))
    with pytest.raises(AgentJudgeSelectionError, match="requires native_astra"):
        asyncio.run(engine.grade_rollout({}, rubric=None, output_root=tmp_path, sample_id="sample",
                                        backend="opus_5", native_turn_timeout_seconds=600))
    assert engine.validate_native_judge_timeout(240, backend="opus_5") == 240


@pytest.mark.parametrize("seconds", [240, 600, 900])
def test_native_budget_reaches_actual_constructor_without_provider(tmp_path, monkeypatch, seconds):
    from eva_agent import codex_runtime
    from eva_agent.training import native_astra_teacher
    rollout, rubric, path, _ = _generic_rollout(tmp_path)
    prepared = engine.prepare_workspace_rollout(rollout, rubric=rubric, source_path=path,
        judge_model_id="gpt-6-astra", judge_route_id="gpt_6_astra")
    auth = tmp_path / "fixture-auth.json"
    auth.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "fixture-only"}}))
    runners = []
    class Runner:
        def __init__(self, factory):
            self.closed = False
            runners.append(self)
        def start(self): pass
        def close(self): self.closed = True
        def run_once(self, *_): pytest.fail("provider call")
    monkeypatch.setattr(codex_runtime, "PersistentCodexRuntimeRunner", Runner)
    monkeypatch.setattr(engine.shutil, "which", lambda _: "/bin/true")
    with engine._open_native_astra_judge(prepared, tmp_path / "artifacts", auth,
                                        native_turn_timeout_seconds=seconds) as judge:
        assert judge._turn_timeout_seconds == seconds
        assert judge._maximum_workspace_tool_calls == 64
        assert judge._maximum_workspace_tool_frontiers == 32
    assert runners[0].closed


def test_native_600_dispatch_and_provenance_bound_without_changing_score(tmp_path, monkeypatch):
    rollout, rubric, path, _ = _generic_rollout(tmp_path)
    seen = []
    @contextmanager
    def native(prepared, root, *, native_turn_timeout_seconds):
        seen.append(native_turn_timeout_seconds)
        yield _ProviderFreeWorkspaceJudge()
    monkeypatch.setattr(engine, "_open_native_astra_judge", native)
    grade = asyncio.run(engine.grade_rollout(rollout, rubric=rubric, output_root=tmp_path / "judge",
        sample_id="one", backend="native_astra", native_turn_timeout_seconds=600))
    assert seen == [600] and grade["reward"] == 1.0
    prepared = engine.prepare_workspace_rollout(rollout, rubric=rubric, source_path=path,
        judge_model_id="gpt-6-astra", judge_route_id="gpt_6_astra")
    default = engine._backend_provenance(prepared, "native_astra")
    changed = engine._backend_provenance(prepared, "native_astra", native_turn_timeout_seconds=600)
    assert default["turn_timeout_seconds"] == 240 and changed["turn_timeout_seconds"] == 600
    assert default["provenance_blake3"] != changed["provenance_blake3"]
    assert grade["judge_provenance"]["turn_timeout_seconds"] == 600
    a = {k: v for k, v in default.items() if k not in {"turn_timeout_seconds", "provenance_blake3"}}
    b = {k: v for k, v in changed.items() if k not in {"turn_timeout_seconds", "provenance_blake3"}}
    assert a == b
    assert changed["provenance_blake3"] == blake3_hex({k: v for k, v in changed.items() if k != "provenance_blake3"})


def test_track_to_benchmark_to_engine_threads_600_and_keeps_one_attempt(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    track_feedback.prepare_track_feedback(**args)
    seen = []
    async def unavailable(*args, **kwargs):
        seen.append(kwargs)
        raise TimeoutError("fixture deadline")
    monkeypatch.setattr(bridge, "grade_rollout", unavailable)
    with pytest.raises(TimeoutError):
        track_feedback.judge_track_once(args["output_root"], native_turn_timeout_seconds=600)
    attempt = json.loads((args["output_root"] / "judge-attempt.json").read_bytes())
    assert attempt["native_turn_timeout_seconds"] == 600
    assert attempt["retry_count"] == 0 and attempt["automatic_fallback"] is False
    assert seen[0]["native_turn_timeout_seconds"] == 600 and seen[0]["backend"] == "native_astra"
    with pytest.raises(FileExistsError):
        track_feedback.judge_track_once(args["output_root"], native_turn_timeout_seconds=600)
    assert len(seen) == 1 and not (args["output_root"] / "feedback.json").exists()


def test_600_coverage_grades_only_actual_complete_s1(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    seen = []
    def judge(root, *, native_turn_timeout_seconds):
        seen.append((root.name, native_turn_timeout_seconds))
        return {"valid": True, "status": "scored"}
    monkeypatch.setattr(track_feedback, "judge_track_once", judge)
    monkeypatch.setattr(track_feedback, "verify_feedback", lambda root: {
        "valid": True, "score": {"reward_bps": 0}, "source_rollout_blake3": "f" * 64})
    roots = track_feedback.evaluate_track_feedback(args["run_root"], args["checkpoint_identity"],
        args["output_root"], native_turn_timeout_seconds=600)
    assert seen == [("S1", 600)] and len(roots) == 1
    coverage = json.loads((args["output_root"] / "coverage.json").read_bytes())
    assert [r["status"] for r in coverage["stages"]] == ["independently_verified", "unknown", "unknown"]
    assert all(r["rubric_score"] is None for r in coverage["stages"][1:])


@pytest.mark.parametrize("filename,symbol", [("judge_automedbench_track_v1.py", "judge_track_once"),
                                            ("judge_automedbench_codex_v1.py", "judge_once")])
def test_both_benchmark_clis_forward_explicit_timeout(tmp_path, monkeypatch, filename, symbol):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location("timeout_cli_fixture_" + symbol, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    def judge(root, **kwargs):
        calls.append(kwargs)
        return {"valid": True, "stage": "S1", "status": "scored", "score": {"reward_bps": 0}}
    monkeypatch.setattr(module, symbol, judge)
    monkeypatch.setattr(module, "write_private", lambda *args: None)
    monkeypatch.setattr(sys, "argv", [filename, "judge", "--output-root", str(tmp_path),
                                     "--native-turn-timeout-seconds", "600"])
    module.main()
    assert calls == [{"native_turn_timeout_seconds": 600}]
