"""CPU-only boundaries for per-evaluation Codex initialization serialization."""
import asyncio
from types import SimpleNamespace

import pytest

from training.automedbench_lite import track_actor as actor
from training.automedbench_lite.adapter import read_document


def deadline(seconds=1.0):
    return SimpleNamespace(require_remaining=lambda: seconds,
        document={"document_blake3": "0" * 64})


def test_shared_initialization_lock_serializes_both_opens_while_turns_overlap(tmp_path, monkeypatch):
    opening = 0
    peak_opening = 0
    turning = 0
    peak_turning = 0
    open_count = 0
    first_turns = 0
    all_first_turns = asyncio.Event()

    class Runtime:
        def __init__(self, backend):
            self.backend = backend

        async def __aenter__(self):
            nonlocal opening, peak_opening, open_count
            opening += 1
            peak_opening = max(peak_opening, opening)
            open_count += 1
            await asyncio.sleep(.002)
            opening -= 1
            return self

        async def __aexit__(self, *_):
            self.backend.closed = True

        async def start_thread(self, options):
            return SimpleNamespace(thread_id=options.cwd)

        async def resume_thread(self, thread_id, _options):
            return SimpleNamespace(thread_id=thread_id)

    monkeypatch.setattr(actor, "CodexRuntime", Runtime)
    monkeypatch.setattr(actor, "app_process", lambda backend: backend)
    processes = []

    def backend():
        process = SimpleNamespace(pid=1000 + len(processes), closed=False)
        process.poll = lambda: 0 if process.closed else None
        processes.append(process)
        return process

    states = []
    for index in range(3):
        audit = tmp_path / str(index)
        audit.mkdir()
        states.append({"audit": audit, "workspace": tmp_path / f"workspace-{index}",
            "options": SimpleNamespace(cwd=str(audit)), "receipts": [], "errors": []})

    async def turn(_runtime, state, index):
        nonlocal turning, peak_turning, first_turns
        turning += 1
        peak_turning = max(peak_turning, turning)
        if index == 0:
            first_turns += 1
            if first_turns == len(states):
                all_first_turns.set()
            await all_first_turns.wait()
        await asyncio.sleep(0)
        state["receipts"].append(SimpleNamespace())
        turning -= 1

    async def scenario():
        lock = asyncio.Lock()
        setup = SimpleNamespace(backend=backend)
        await asyncio.gather(*(actor.run_whole_track(setup, state, turn, seconds=3600,
            initialization_lock=lock) for state in states))

    asyncio.run(scenario())
    assert peak_opening == 1
    assert peak_turning == 3
    assert open_count == 6  # first open and same-home restart for all three tracks
    assert all(process.closed for process in processes)


def test_queued_initialization_expires_inside_same_track_deadline(tmp_path, monkeypatch):
    calls = []

    class Runtime:
        def __init__(self, _backend):
            pass

        async def __aenter__(self):
            calls.append("enter")

        async def __aexit__(self, *_):
            calls.append("cleanup")

    monkeypatch.setattr(actor, "CodexRuntime", Runtime)

    async def scenario():
        lock = asyncio.Lock()
        await lock.acquire()
        try:
            with pytest.raises(asyncio.TimeoutError):
                async with actor.deadline_runtime(None, deadline(.005), lock, audit=tmp_path,
                        runtime_sequence=1, phase_intent="01-planning"):
                    pytest.fail("queued initialization entered")
        finally:
            lock.release()

    asyncio.run(scenario())
    assert calls == ["cleanup"]
    evidence = read_document(tmp_path / "app-server-initialization-error.json")
    assert evidence["error_category"] == "TimeoutError"
    assert evidence["thread_start_or_resume_called_on_this_runtime"] is False


def test_partial_open_is_cleaned_before_original_failure_propagates(tmp_path, monkeypatch):
    calls = []

    class TransportClosedError(RuntimeError):
        pass

    class Runtime:
        def __init__(self, _backend):
            pass

        async def __aenter__(self):
            calls.append("partially-open")
            raise TransportClosedError("private transport detail")

        async def __aexit__(self, *_):
            calls.append("cleanup")

    monkeypatch.setattr(actor, "CodexRuntime", Runtime)

    async def scenario():
        with pytest.raises(TransportClosedError):
            async with actor.deadline_runtime(None, deadline(), asyncio.Lock(), audit=tmp_path,
                    runtime_sequence=1, phase_intent="01-planning"):
                pytest.fail("failed initialization yielded a runtime")

    asyncio.run(scenario())
    assert calls == ["partially-open", "cleanup"]
    evidence = read_document(tmp_path / "app-server-initialization-error.json")
    assert evidence["partial_runtime_cleanup_attempted"] is True
    assert evidence["partial_runtime_cleanup_error_category"] is None


def test_initialization_sidecar_retains_only_sanitized_failure_category(tmp_path, monkeypatch):
    class FirstOpenFailure(RuntimeError):
        pass

    class Runtime:
        def __init__(self, _backend):
            pass

        async def __aenter__(self):
            raise FirstOpenFailure("credential-like private detail")

        async def __aexit__(self, *_):
            pass

    monkeypatch.setattr(actor, "CodexRuntime", Runtime)

    async def scenario():
        with pytest.raises(FirstOpenFailure):
            async with actor.deadline_runtime(None, deadline(), asyncio.Lock(), audit=tmp_path,
                    runtime_sequence=2, phase_intent="04-full-subset"):
                pytest.fail("failed initialization yielded a runtime")

    asyncio.run(scenario())
    raw = (tmp_path / "app-server-initialization-error.json").read_text()
    evidence = read_document(tmp_path / "app-server-initialization-error.json")
    assert evidence["schema"] == "eva.automedbench-app-server-initialization-error.v1"
    assert evidence["error_category"] == "FirstOpenFailure"
    assert evidence["runtime_sequence"] == 2
    assert evidence["phase_intent"] == "04-full-subset"
    assert "credential-like" not in raw
