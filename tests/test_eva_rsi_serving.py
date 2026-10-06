"""Serving ownership tests: no subprocess, network, torch, or GPU execution."""
import json
import os
from pathlib import Path
import signal
import stat
import struct
import subprocess
import urllib.error

import pytest

from training.eva_rsi import serving as module


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    checkpoint = tmp_path / "training/checkpoints"
    checkpoint.mkdir(parents=True)
    model = checkpoint.parent / "hf/iter_0000049"
    model.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (model / name).write_text('{}')
    header = json.dumps({"tensor": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    (model / "one.safetensors").write_bytes(struct.pack('<Q', len(header)) + header + b'\x00\x00')
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"tensor": "one.safetensors"}}))
    architecture = tmp_path / "original"
    architecture.mkdir()
    image = tmp_path / "public.jpg"
    image.write_bytes(b'public image fixture, not a clinical reference')
    output = tmp_path / "serving"
    alive = {"value": True}
    owner = {"pid": 424242, "pgid": 424242, "session": 424242, "start_ticks": 123456, "state": "S"}
    captures = {"launch": [], "signals": [], "requests": [], "ports": []}

    class Process:
        pid = owner["pid"]
        def poll(self):
            return None if alive["value"] else 0
        def wait(self, timeout):
            if alive["value"]:
                raise subprocess.TimeoutExpired('fake', timeout)
            return 0
    process = Process()

    def launch(argv, **kwargs):
        captures["launch"].append((argv, kwargs))
        return process

    def request(url, *, body=None, timeout=10):
        captures["requests"].append((url, body))
        if body is None:
            return {"model_path": str(model), "served_model_name": module.MODEL_ALIAS}
        return {"model": module.MODEL_ALIAS, "choices": [{"message": {"role": "assistant",
                "content": "A round brown shape.", "reasoning_content": "PRIVATE THINKING SENTINEL"},
                "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 12}}

    def killpg(pgid, signum):
        assert pgid == owner["pid"]
        captures["signals"].append((pgid, signum))
        alive["value"] = False

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    monkeypatch.setattr(module, "_free_port", lambda port: captures["ports"].append(port))
    monkeypatch.setattr(module, "_checkpoint", lambda *args: {"valid": True, "selection": {"iteration": 49}})
    monkeypatch.setattr(module, "_process", lambda pid: owner.copy() if alive["value"] else None)
    monkeypatch.setattr(module, "_members", lambda saved: [owner.copy()] if alive["value"] else [])
    monkeypatch.setattr(module, "_listener_owned", lambda port, members: bool(members))
    monkeypatch.setattr(module, "_argv", lambda pid: ["python", "-m", "sglang.launch_server", "--model-path", str(model)])
    monkeypatch.setattr(module, "_request", request)
    monkeypatch.setattr(module.os, "killpg", killpg)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return {"args": (model, checkpoint, architecture, output, image), "captures": captures,
            "owner": owner, "alive": alive, "process": process, "request": request}


def test_context_yields_compatible_private_binding_and_stops_owned_group(prepared, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "SECRET SENTINEL")
    with module.serving_session(*prepared["args"]) as serving:
        assert serving.pid == prepared["owner"]["pid"]
        identity = json.loads(serving.identity_path.read_bytes())
        canary = json.loads(serving.canary_path.read_bytes())
        assert identity["schema"] == "eva.qwen-final-serving-checkpoint-identity.v1"
        assert identity["final_checkpoint_iteration"] == 49
        assert identity["settings"]["context_length"] == 32768
        assert canary["schema"] == "eva.qwen-final-serving-image-canary.v1"
        assert canary["status"] == "complete" and canary["multimodal_request_accepted"]
        assert canary["thinking_requested"] and canary["image_requests_attempted"] == 1
        assert canary["checkpoint_identity_blake3"] == module.blake3(serving.identity_path.read_bytes()).hexdigest()
        assert canary["actual_server_argv"][-1] == identity["exact_final_model_path"]
        assert prepared["captures"]["signals"] == []
        payloads = [json.loads(body) for _, body in prepared["captures"]["requests"] if body]
        assert len(payloads) == 1 and payloads[0]["chat_template_kwargs"] == {"enable_thinking": True}
    argv, kwargs = prepared["captures"]["launch"][0]
    assert argv == ["bash", str(module.LAUNCH_SCRIPT)] and kwargs["start_new_session"]
    assert "OPENAI_API_KEY" not in kwargs["env"]
    assert prepared["captures"]["ports"] == [30910, 30910]
    assert prepared["captures"]["signals"] == [(424242, signal.SIGTERM)]
    output = prepared["args"][3]
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for path in output.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        if path.suffix == '.json':
            assert "SENTINEL" not in path.read_text()
    exit_receipt = json.loads((output / "session-exit.json").read_bytes())
    assert exit_receipt["exception_type"] is None
    assert exit_receipt["cleanup"]["verified_members_remaining"] == 0


def test_occupied_port_never_launches_or_signals(prepared, monkeypatch):
    def occupied(port):
        raise module.ServingError('evaluation_port_is_occupied')
    monkeypatch.setattr(module, "_free_port", occupied)
    with pytest.raises(module.ServingError, match="occupied"):
        with module.serving_session(*prepared["args"]):
            pytest.fail("occupied server accepted")
    assert prepared["captures"]["launch"] == prepared["captures"]["signals"] == []
    assert not prepared["args"][3].exists()


@pytest.mark.parametrize("failure", ["wrong_model", "unowned_port", "wrong_argv", "exited"])
def test_readiness_identity_failure_never_makes_image_request(prepared, monkeypatch, failure):
    if failure == "wrong_model":
        monkeypatch.setattr(module, "_request", lambda *a, **k: {"model_path": "/wrong"})
    elif failure == "unowned_port":
        monkeypatch.setattr(module, "_listener_owned", lambda *a: False)
    elif failure == "wrong_argv":
        monkeypatch.setattr(module, "_argv", lambda pid: ["unrelated"])
    else:
        monkeypatch.setattr(prepared["process"], "poll", lambda: 7)
    with pytest.raises(module.ServingError):
        with module.serving_session(*prepared["args"]):
            pytest.fail("invalid readiness yielded")
    assert not any(body for _, body in prepared["captures"]["requests"])
    assert prepared["captures"]["signals"] == [(424242, signal.SIGTERM)]


def test_image_failure_is_retained_once_without_private_error_text(prepared, monkeypatch):
    def request(url, *, body=None, timeout=10):
        if body:
            prepared["captures"]["requests"].append((url, body))
            raise RuntimeError("PRIVATE ERROR SENTINEL")
        return prepared["request"](url, timeout=timeout)
    monkeypatch.setattr(module, "_request", request)
    with pytest.raises(module.ServingError, match="no_retry"):
        with module.serving_session(*prepared["args"]):
            pytest.fail("failed image accepted")
    assert sum(body is not None for _, body in prepared["captures"]["requests"]) == 1
    canary = json.loads((prepared["args"][3] / "image-canary.json").read_bytes())
    assert canary["status"] == "failed" and not canary["multimodal_request_accepted"]
    assert "SENTINEL" not in json.dumps(canary)
    assert prepared["captures"]["signals"] == [(424242, signal.SIGTERM)]


def test_caller_exception_still_cleans_only_owned_server(prepared):
    with pytest.raises(ValueError, match="caller"):
        with module.serving_session(*prepared["args"]):
            raise ValueError("caller")
    assert prepared["captures"]["signals"] == [(424242, signal.SIGTERM)]
    receipt = json.loads((prepared["args"][3] / "session-exit.json").read_bytes())
    assert receipt["exception_type"] == "ValueError"


def test_reused_pid_is_never_signalled(prepared, monkeypatch):
    owner = prepared["owner"].copy()
    monkeypatch.setattr(module, "_process", lambda pid: {**owner, "start_ticks": owner["start_ticks"] + 1})
    monkeypatch.setattr(module, "_members", lambda owner: [])
    result = module._stop_owned(prepared["process"], owner, [owner], timeout=.1)
    assert result["signals"] == [] and prepared["captures"]["signals"] == []


def test_nonowned_launch_identity_is_never_signalled(prepared, monkeypatch):
    monkeypatch.setattr(module, "_process", lambda pid: {**prepared["owner"], "pgid": 42})
    with pytest.raises(module.ServingError, match="session_leader"):
        with module.serving_session(*prepared["args"]):
            pytest.fail("not owner")
    assert prepared["captures"]["signals"] == []


def test_hf_path_must_match_checkpoint_iteration(prepared, monkeypatch):
    monkeypatch.setattr(module, "_checkpoint", lambda *args: {"valid": True, "selection": {"iteration": 48}})
    with pytest.raises(module.ServingError, match="lineage"):
        with module.serving_session(*prepared["args"]):
            pytest.fail("wrong lineage")
    assert prepared["captures"]["launch"] == []


def test_hf_missing_indexed_payload_rejected_before_launch(prepared):
    (prepared["args"][0] / "one.safetensors").unlink()
    with pytest.raises(module.ServingError, match="hf_shard_missing"):
        with module.serving_session(*prepared["args"]):
            pytest.fail("missing weights")
    assert prepared["captures"]["launch"] == []


def test_environment_keeps_valid_libpaths_without_credentials(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    cuda = tmp_path / ".venv/lib/python3.12/site-packages/nvidia/cu13/lib"
    cuda.mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/valid/a:/valid/b")
    monkeypatch.setenv("OPENAI_API_KEY", "SECRET")
    monkeypatch.setenv("HTTPS_PROXY", "https://service.example.invalid")
    env = module._environment(tmp_path / "model", 30910)
    assert env["LD_LIBRARY_PATH"] == str(cuda) + ":/valid/a:/valid/b"
    assert "OPENAI_API_KEY" not in env and "HTTPS_PROXY" not in env


def test_isolated_med_uses_explicit_venv_spelling_and_runtime(monkeypatch,tmp_path):
    monkeypatch.setattr(module,"ROOT",tmp_path/"detached-med-worktree")
    binary=tmp_path/"shared/.venv/bin/python"
    binary.parent.mkdir(parents=True)
    binary.symlink_to("/usr/bin/python3")
    cuda=tmp_path/"shared/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
    cuda.mkdir(parents=True)
    monkeypatch.setenv("PATH","/usr/bin:/bin")
    env=module._environment(tmp_path/"model",30911,python_executable=binary,context_length=32768)
    assert env["EVA_TRAINING_PYTHON"]==str(binary)  # No resolution to system Python.
    assert env["PATH"].split(":")[0]==str(binary.parent)
    assert env["LD_LIBRARY_PATH"].split(":")[0]==str(cuda)
    assert env["EVA_QWEN_CONTEXT_LENGTH"]=="32768"


def test_unavailable_explicit_python_rejected_before_gpu(prepared):
    with pytest.raises(module.ServingError,match="selected_python"):
        with module.serving_session(*prepared["args"],python_executable="/absent/venv/bin/python"):
            pytest.fail("invalid Python launched")
    assert prepared["captures"]["launch"]==[]


def test_unverified_context_rejected_before_gpu(prepared):
    with pytest.raises(module.ServingError,match="verified_32768"):
        with module.serving_session(*prepared["args"],context_length=24576):
            pytest.fail("wrong context launched")
    assert prepared["captures"]["launch"]==[]
