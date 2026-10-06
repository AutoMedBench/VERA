"""Real CPU threading and workspace grading; no provider or GPU calls."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from threading import Event, Lock

import pytest

from eva_agent.pipeline.digests import blake3_hex
from eva_agent.training import slime_agent_judge as engine
from eva_agent.training.agent_judge import AgentJudgeSelectionError
from test_slime_agent_judge import _generic_rollout, _ProviderFreeWorkspaceJudge


@pytest.fixture(autouse=True)
def independent_process_pool(monkeypatch):
    monkeypatch.setattr(engine, "_JUDGE_POOL", None)
    monkeypatch.delenv("EVA_SLIME_JUDGE_CONCURRENCY", raising=False)


@pytest.mark.parametrize("value", [True, False, 0, -1, 5, 4.0, "4", [], {}])
def test_invalid_explicit_limit_fails_before_dispatch_or_artifacts(tmp_path, monkeypatch, value):
    monkeypatch.setattr(engine, "_grade_rollout_sync", lambda *a, **k: pytest.fail("dispatch"))
    with pytest.raises(AgentJudgeSelectionError, match="integer from 1 to 4"):
        asyncio.run(engine.grade_rollout({}, rubric=None, output_root=tmp_path / "absent",
            sample_id="one", maximum_concurrent_judges=value))
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("value", ["", "0", "5", "04", "4.0", " 4", "true"])
def test_environment_limit_is_strict(monkeypatch, value):
    monkeypatch.setenv("EVA_SLIME_JUDGE_CONCURRENCY", value)
    with pytest.raises(AgentJudgeSelectionError, match="integer from 1 to 4"):
        engine.resolve_judge_concurrency()


def test_default_and_explicit_launch_selection(monkeypatch):
    assert engine.resolve_judge_concurrency() == 2
    monkeypatch.setenv("EVA_SLIME_JUDGE_CONCURRENCY", "4")
    assert engine.resolve_judge_concurrency() == 4
    assert engine.resolve_judge_concurrency(2) == 2
    with engine._judge_slot(4):
        with pytest.raises(AgentJudgeSelectionError, match="first dispatch"):
            with engine._judge_slot(2):
                pytest.fail("conflicting pool admitted")
    # Even after draining, changing a launch profile requires a new process.
    with pytest.raises(AgentJudgeSelectionError, match="first dispatch"):
        with engine._judge_slot(2):
            pytest.fail("second independent pool admitted")


@pytest.mark.parametrize("limit", [2, 4])
def test_real_threads_respect_limit_and_release_after_exception(limit):
    lock, at_limit, release = Lock(), Event(), Event()
    active = peak = entered = 0
    waits = []

    def work(index):
        nonlocal active, peak, entered
        with engine._judge_slot(limit) as wait_seconds:
            with lock:
                active += 1
                entered += 1
                peak = max(peak, active)
                waits.append(wait_seconds)
                if active == limit:
                    at_limit.set()
            try:
                assert release.wait(5), "CPU fixture release deadline"
                if index == 0:
                    raise RuntimeError("fixture Judge unavailable")
                return index
            finally:
                with lock:
                    active -= 1

    with ThreadPoolExecutor(max_workers=limit + 2) as executor:
        futures = [executor.submit(work, index) for index in range(limit + 2)]
        try:
            assert at_limit.wait(5)
            with lock:
                assert active == peak == entered == limit
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="Judge unavailable"):
            futures[0].result(timeout=5)
        assert [future.result(timeout=5) for future in futures[1:]] == list(range(1, limit + 2))
    assert active == 0 and peak == limit and entered == limit + 2
    assert all(value >= 0 for value in waits)
    with engine._judge_slot(limit):
        pass  # All slots, including the exceptional call's slot, were released.


def test_four_slots_preserve_real_workspace_grade_and_record_only_outer_diagnostics(tmp_path, monkeypatch):
    rollout, rubric, path, _ = _generic_rollout(tmp_path)

    @contextmanager
    def native(*args, **kwargs):
        yield _ProviderFreeWorkspaceJudge()

    monkeypatch.setattr(engine, "_open_native_astra_judge", native)
    grades = []
    for limit in (2, 4):
        monkeypatch.setattr(engine, "_JUDGE_POOL", None)  # Independent launch profiles.
        monkeypatch.setenv("EVA_SLIME_JUDGE_CONCURRENCY", str(limit))
        root = tmp_path / f"limit-{limit}"
        grade = asyncio.run(engine.grade_rollout(rollout, rubric=rubric, output_root=root,
            sample_id="one", backend="native_astra"))
        grades.append(grade)
        assert json.loads((root / "one/rollout.json").read_bytes()) == rollout
        diagnostic = json.loads((root / "one/judge-scheduling.json").read_bytes())
        assert diagnostic["maximum_concurrent_judges"] == limit
        assert diagnostic["slot_wait_seconds"] >= 0
        assert diagnostic["retry_count"] == 0
        assert diagnostic["diagnostics_blake3"] == blake3_hex({
            k: v for k, v in diagnostic.items() if k != "diagnostics_blake3"})
        assert grade["judge_provenance"]["maximum_concurrent_judges"] == limit
        assert "slot_wait_seconds" not in json.dumps(grade)
    for key in ("reward", "item_scores_bps", "rubric_digest", "source_actor_provider",
                "source_actor_schema", "judge_model_id", "workspace_inspected"):
        assert grades[0][key] == grades[1][key]
    assert grades[0]["reward"] == 1.0
    prepared = engine.prepare_workspace_rollout(rollout, rubric=rubric, source_path=path,
        judge_model_id="gpt-6-astra", judge_route_id="gpt_6_astra")
    original = engine._backend_provenance(prepared, "native_astra", maximum_concurrent_judges=2)
    selected = engine._backend_provenance(prepared, "native_astra", maximum_concurrent_judges=4)
    assert {k: v for k, v in original.items() if k not in {"maximum_concurrent_judges", "provenance_blake3"}} == {
        k: v for k, v in selected.items() if k not in {"maximum_concurrent_judges", "provenance_blake3"}}


def test_failed_judge_releases_slot_without_emitting_reward(tmp_path, monkeypatch):
    rollout, rubric, _, _ = _generic_rollout(tmp_path)

    @contextmanager
    def unavailable(*args, **kwargs):
        raise RuntimeError("fixture provider failure")
        yield  # pragma: no cover

    monkeypatch.setattr(engine, "_open_native_astra_judge", unavailable)
    with pytest.raises(RuntimeError, match="provider failure"):
        asyncio.run(engine.grade_rollout(rollout, rubric=rubric, output_root=tmp_path / "failed",
            sample_id="one", backend="native_astra", maximum_concurrent_judges=4))
    root = tmp_path / "failed/one"
    assert not (root / "grade.json").exists()
    assert json.loads((root / "failure.json").read_bytes())["reward_emitted"] is False
    assert json.loads((root / "judge-scheduling.json").read_bytes())["maximum_concurrent_judges"] == 4
    assert engine._JUDGE_POOL[1].acquire(blocking=False)
    engine._JUDGE_POOL[1].release()
