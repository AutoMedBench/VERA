"""CPU-only watcher orchestration: actor/provider/report execution is mocked."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def watcher(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/watch_eva_rsi_supplement_v1.py"
    spec = importlib.util.spec_from_file_location("supplement_watcher_fixture", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context, settings = tmp_path / "context.json", tmp_path / "settings.json"
    context.write_text(json.dumps({"attempt_root": str(tmp_path / "actor-attempt"), "phase": "evaluation"}))
    settings.write_text(json.dumps({"fixture_only": True}))
    packet = tmp_path / "packet"
    argv = [str(script), "--context", str(context), "--settings", str(settings),
        "--packet", str(packet), "--poll-seconds", "1", "--max-wait-seconds", "3"]
    monkeypatch.setattr(module.sys, "argv", argv)
    clock = {"now": 0.0, "sleeps": []}
    def sleep(seconds):
        clock["sleeps"].append(seconds)
        clock["now"] += seconds
    monkeypatch.setattr(module, "time", SimpleNamespace(
        monotonic=lambda: clock["now"], time=lambda: 1000 + clock["now"], sleep=sleep))
    monkeypatch.setattr(module, "start_identity", lambda pid: "fixture-birth")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("unexpected report subprocess"))
    calls = []
    result = {"index": {"path": str(tmp_path / "actor-attempt/supplemental-index.json"), "blake3": "a" * 64},
        "status": "collected", "standalone_training_admission": False}
    def plans(sequence):
        states = iter(sequence)
        last = {"ready": False}
        def run(context_value, settings_value, *, execute=False):
            nonlocal last
            calls.append(execute)
            assert context_value["attempt_root"] == str(tmp_path / "actor-attempt")
            assert settings_value == {"fixture_only": True}
            if execute:
                return result
            last = next(states, last)
            return last
        monkeypatch.setattr(module, "run_supplemental", run)
    return SimpleNamespace(module=module, packet=packet, context=context, settings=settings,
        argv=argv, clock=clock, calls=calls, plans=plans, result=result, root=tmp_path)


def read(path):
    return json.loads(path.read_bytes())


def test_default_preflight_does_not_write_wait_or_execute(watcher, capsys):
    watcher.plans([{"ready": False, "status": "WAITING_SOURCE", "provider_calls": 0, "gpu_launches": 0}])
    before = {path: path.read_bytes() for path in watcher.root.rglob("*") if path.is_file()}
    assert watcher.module.main() == 0
    assert watcher.calls == [False] and watcher.clock["sleeps"] == []
    assert not watcher.packet.exists()
    assert before == {path: path.read_bytes() for path in watcher.root.rglob("*") if path.is_file()}
    assert json.loads(capsys.readouterr().out)["status"] == "WAITING_SOURCE"


def test_waiting_to_ready_executes_once_and_retains_ssh_claim(watcher, capsys):
    watcher.argv.append("--execute")
    watcher.plans([{"ready": False}, {"ready": False}, {"ready": True}])
    assert watcher.module.main() == 0
    assert watcher.calls == [False, False, False, True]
    assert watcher.clock["sleeps"] == [1, 1]
    launch = read(watcher.packet / "watcher-launch.json")
    assert launch["start_identity"] == "fixture-birth"
    assert launch["inputs"] == [watcher.module.commitment(watcher.context), watcher.module.commitment(watcher.settings)]
    assert launch["automatic_actor_retries"] is False and launch["controller_state_modified"] is False
    result = read(watcher.packet / "watcher-result.json")
    assert result["status"] == "complete" and result["supplement"] == watcher.result
    statuses = [json.loads(line)["status"] for line in capsys.readouterr().out.splitlines()]
    assert statuses == ["waiting_for_original_cleanup", "waiting_for_original_cleanup",
        "running_two_fresh_tracks", "supplement_collected", "complete"]


def test_existing_one_shot_claim_prevents_actor_reuse(watcher):
    watcher.argv.append("--execute")
    watcher.plans([{"ready": True}])
    assert watcher.module.main() == 0
    original = (watcher.packet / "watcher-launch.json").read_bytes()
    with pytest.raises(FileExistsError):
        watcher.module.main()
    assert watcher.calls.count(True) == 1
    assert (watcher.packet / "watcher-launch.json").read_bytes() == original
    assert read(watcher.packet / "watcher-result.json")["status"] == "complete"


def test_source_wait_timeout_never_launches_actor(watcher):
    watcher.argv.append("--execute")
    watcher.plans([{"ready": False}])
    assert watcher.module.main() == 1
    assert watcher.calls == [False] * 4 and watcher.clock["sleeps"] == [1, 1, 1]
    assert read(watcher.packet / "watcher-progress.json")["status"] == "unavailable"
    assert read(watcher.packet / "watcher-failure.json")["automatic_actor_retries"] is False
    assert not (watcher.root / "actor-attempt").exists()
    assert not (watcher.packet / "watcher-result.json").exists()


def test_reporting_failure_does_not_rerun_collected_supplement(watcher, monkeypatch, capsys):
    original = watcher.root / "original-index.json"
    original.write_text(json.dumps({"fixture_only": True}))
    watcher.argv.extend(["--execute", "--original-index", str(original)])
    watcher.plans([{"ready": True}])
    reports = []
    def report(command, *, check):
        reports.append(command)
        assert check is False
        assert command[command.index("--original-index") + 1] == str(original)
        assert command[command.index("--supplement-index") + 1] == watcher.result["index"]["path"]
        assert command[command.index("--output-root") + 1] == str(watcher.packet / "report")
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(watcher.module.subprocess, "run", report)
    assert watcher.module.main() == 1
    assert watcher.calls == [False, True] and len(reports) == 1
    statuses = [json.loads(line)["status"] for line in capsys.readouterr().out.splitlines()]
    assert statuses == ["running_two_fresh_tracks", "supplement_collected", "unavailable"]
    assert not (watcher.packet / "watcher-result.json").exists()
    assert read(watcher.packet / "watcher-failure.json")["automatic_actor_retries"] is False
    with pytest.raises(FileExistsError):
        watcher.module.main()
    assert watcher.calls.count(True) == 1 and len(reports) == 1


def test_changed_input_during_wait_blocks_execution(watcher, monkeypatch):
    watcher.argv.append("--execute")
    watcher.plans([{"ready": False}, {"ready": True}])
    def tamper(seconds):
        watcher.settings.write_text(json.dumps({"changed_fixture": True}))
    monkeypatch.setattr(watcher.module.time, "sleep", tamper)
    assert watcher.module.main() == 1
    assert watcher.calls == [False]
    assert read(watcher.packet / "watcher-progress.json")["status"] == "unavailable"
    assert not (watcher.root / "actor-attempt").exists()
