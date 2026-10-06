"""Owned, temporary local Qwen serving for verified post-round evaluation.

Entering this context launches a GPU server and makes exactly one public-image
canary request; importing it does neither. The caller must authorize that work and
choose the public image (never a scorer/reference input). DCP metadata is trusted
local pickle, audited by the existing checkpoint preflight. HF/DCP lineage and
headers are checked, not full tensor payload equality or benchmark performance.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
from uuid import uuid4

from blake3 import blake3


ROOT = Path(__file__).resolve().parents[2]
LAUNCH_SCRIPT = ROOT / "training/slime/serve_qwen35_validation.sh"
MODEL_ALIAS = "Qwen/Qwen3.5-9B"


class ServingError(RuntimeError):
    """Fixed error codes only, never raw server output or environment values."""


@dataclass(frozen=True)
class ServingSession:
    identity_path: Path
    canary_path: Path
    output_root: Path
    pid: int


def _require(value, code):
    if not value:
        raise ServingError(code)


def _write(path: Path, document: dict) -> None:
    data = json.dumps(document, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _process(pid: int) -> dict | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        tail = raw.rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": int(tail[19]), "pgid": int(tail[2]),
                "session": int(tail[3]), "state": tail[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def _same(saved: dict) -> bool:
    actual = _process(saved["pid"])
    return bool(actual and actual["state"] != "Z" and all(
        actual[k] == saved[k] for k in ("pid", "start_ticks", "pgid", "session")))


def _members(owner: dict) -> list[dict]:
    """Only take a fresh group snapshot while the exact session leader is alive."""
    if not _same(owner):
        return []
    members = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            item = _process(int(entry.name))
            if item and item["pgid"] == owner["pid"] == item["session"] and item["state"] != "Z":
                members.append(item)
    return members


def _argv(pid: int) -> list[str]:
    return [x.decode() for x in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if x]


def _listener_owned(port: int, members: list[dict]) -> bool:
    inodes = set()
    for name in ("tcp", "tcp6"):
        for line in Path(f"/proc/net/{name}").read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(f"socket:[{fields[9]}]")
    for member in members:
        if not _same(member):
            continue
        try:
            for fd in Path(f"/proc/{member['pid']}/fd").iterdir():
                try:
                    if os.readlink(fd) in inodes:
                        return True
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            pass
    return False


def _free_port(port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            raise ServingError("evaluation_port_is_occupied; existing_server_not_touched") from None


def _environment(model: Path, port: int, *, python_executable=None, context_length=32768) -> dict:
    # No provider credentials, proxy vars, or arbitrary inherited overrides.
    env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH")
           if k in os.environ}
    # Preserve the selected venv executable spelling: resolving its symlink
    # would bind system Python and lose the installed GPU environment.
    python = Path(python_executable) if python_executable is not None else ROOT.parent / ".venv/bin/python"
    _require(python.is_absolute(), "serving_python_must_be_absolute")
    cuda = python.parent.parent / "lib/python3.12/site-packages/nvidia/cu13"
    if (cuda / "lib").is_dir():
        existing = env.get("LD_LIBRARY_PATH", "")
        parts = [str(cuda / "lib"), *(p for p in existing.split(":") if p and p != str(cuda / "lib"))]
        env["LD_LIBRARY_PATH"] = ":".join(parts)
        env["CUDA_HOME"] = str(cuda)
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "/usr/bin:/bin")
    env.update(EVA_TRAINING_PYTHON=str(python),
               EVA_QWEN_MODEL=str(model), EVA_QWEN_PORT=str(port),
               EVA_QWEN_CONTEXT_LENGTH=str(context_length), EVA_CUDA_DEVICE="0")
    return env


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _request(url: str, *, body: bytes | None = None, timeout: float = 10):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    request = urllib.request.Request(url, data=body, method="POST" if body else "GET",
                                    headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=timeout) as response:
        _require(response.status == 200, "serving_http_status")
        raw = response.read(1024 * 1024 + 1)
        _require(len(raw) <= 1024 * 1024, "serving_response_too_large")
        value = json.loads(raw)
        _require(isinstance(value, dict), "serving_response_not_object")
        return value


def _checkpoint(checkpoint_root: Path, architecture_model_path: Path) -> dict:
    # Production-only lazy import/unpickle; focused tests replace this boundary.
    from training.slime.checkpoint_preflight import verify_checkpoint
    return verify_checkpoint(checkpoint_root, architecture_model_path)


def _hf_assets(model: Path) -> dict:
    """Cheap header/size commitments; no full weight scan or dtype assumptions."""
    assets = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        path = model / name
        _require(path.is_file(), "hf_required_asset_missing")
        assets[name] = {"blake3": blake3(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    index = json.loads((model / "model.safetensors.index.json").read_bytes())
    weight_map = index.get("weight_map")
    _require(isinstance(weight_map, dict) and bool(weight_map), "hf_weight_index_missing")
    headers = {}
    for name in sorted(set(weight_map.values())):
        _require(isinstance(name, str) and Path(name).name == name, "hf_shard_path_unsafe")
        path = model / name
        _require(path.is_file() and not path.is_symlink(), "hf_shard_missing")
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            _require(len(prefix) == 8, "hf_header_truncated")
            length = struct.unpack("<Q", prefix)[0]
            _require(0 < length <= 16 * 1024 * 1024, "hf_header_size")
            raw = stream.read(length)
        _require(len(raw) == length, "hf_header_truncated")
        header = json.loads(raw)
        for key, shard in weight_map.items():
            if shard == name:
                entry = header.get(key, {})
                offsets = entry.get("data_offsets", [])
                _require(len(offsets) == 2 and all(type(x) is int for x in offsets)
                         and 0 <= offsets[0] < offsets[1] <= size - 8 - length, "hf_index_tensor_extent_missing")
        headers[name] = {"header_blake3": blake3(prefix + raw).hexdigest(),
                         "header_bytes": 8 + length, "file_bytes": size}
    return {"assets": assets, "weight_headers": headers, "full_weight_files_rehashed": False,
            "hf_dcp_tensor_payload_equality_verified": False}


def _stop_owned(process, owner: dict, known_members: list[dict], timeout: float) -> dict:
    _require(owner["pgid"] == owner["pid"] == owner["session"], "cleanup_owner_not_session_leader")
    known = {item["pid"]: item for item in [owner, *known_members, *_members(owner)]}

    def survivors():
        return [item for item in known.values() if _same(item)]

    sent = []
    for signum in (signal.SIGTERM, signal.SIGKILL):
        live = survivors()
        if not live:
            break
        # A matching PID/starttime/session member pins the original process group,
        # even if its leader exited after TERM. Never signal a reused PID/group.
        try:
            os.killpg(owner["pgid"], signum)
            sent.append(signal.Signals(signum).name)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + timeout
        while survivors() and time.monotonic() < deadline:
            process.poll()  # Reap an exited child; zombies do not prove liveness.
            time.sleep(0.1)
    try:
        process.wait(timeout=0)
    except subprocess.TimeoutExpired:
        pass
    return {"signals": sent, "verified_members_remaining": len(survivors()),
            "only_identity_matched_owned_group_signalled": True,
            "unrelated_processes_signalled": False}


@contextmanager
def serving_session(model_path, checkpoint_root, architecture_model_path, output_root,
                    public_image, port=30910, *, startup_timeout=900, stop_timeout=10,
                    python_executable=None, context_length=32768):
    """Yield identity/canary paths while the exact owned server is alive.

    A occupied port is a blocker, never authority to stop an existing Qwen server.
    Model-info polling is read-only; the image POST is single-attempt/no retry.
    This synchronous context cleans up on ordinary exit/exception, not SIGKILL or
    power loss. Its durable PID/starttime receipt supports manual reconciliation;
    it never treats an abandoned server as an automatic reusable completed stage.
    """
    _require(type(port) is int and 1024 <= port <= 65535, "serving_port_invalid")
    _require(type(context_length) is int and context_length == 32768, "serving_requires_verified_32768_context")
    if python_executable is not None:
        python = Path(python_executable)
        _require(python.is_absolute() and python.is_file() and os.access(python, os.X_OK),
                 "serving_selected_python_unavailable")
    _require(0 < startup_timeout <= 3600 and 0 < stop_timeout <= 60, "serving_timeout_invalid")
    _free_port(port)
    model = Path(model_path).resolve(strict=True)
    checkpoint = Path(checkpoint_root).resolve(strict=True)
    architecture = Path(architecture_model_path).resolve(strict=True)
    image_path = Path(public_image).resolve(strict=True)
    _require(image_path.suffix.lower() in (".jpg", ".jpeg", ".png")
             and 0 < image_path.stat().st_size <= 16 * 1024 * 1024, "public_image_format_or_size")
    image = image_path.read_bytes()
    check = _checkpoint(checkpoint, architecture)
    _require(check.get("valid") is True, "checkpoint_preflight_failed")
    iteration = check.get("selection", {}).get("iteration")
    _require(type(iteration) is int and iteration >= 0, "checkpoint_iteration_missing")
    _require(model == checkpoint.parent / "hf" / f"iter_{iteration:07d}", "hf_checkpoint_lineage_path_differs")
    assets = _hf_assets(model)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    identity_path, canary_path = output / "checkpoint-identity.json", output / "image-canary.json"
    identity = {"schema": "eva.qwen-final-serving-checkpoint-identity.v1", "receipt_id": str(uuid4()),
                "created_at": _now(), "status": "launch_prepared", "exact_final_model_path": str(model),
                "final_checkpoint_iteration": iteration, "checkpoint_root": str(checkpoint),
                "architecture_model_path": str(architecture), "checkpoint_preflight": check,
                "hf_asset_commitments": assets, "full_weight_files_rehashed": False,
                "launch_script": str(LAUNCH_SCRIPT), "launch_script_blake3": blake3(LAUNCH_SCRIPT.read_bytes()).hexdigest(),
                "settings": {"served_model_name": MODEL_ALIAS, "host": "127.0.0.1", "port": port,
                    "context_length": context_length, "thinking_canary_requested": True,
                    "tool_call_parser": "qwen3_coder", "reasoning_parser": "qwen3"}}
    if python_executable is not None:
        identity["settings"]["python_executable"] = str(python_executable)
    _write(identity_path, identity)
    _free_port(port)  # Recheck after preflight; never send a canary to a known occupied port.
    log_fd = os.open(output / "server.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    owner, known, process, failure = None, [], None, None
    try:
        with os.fdopen(log_fd, "wb") as log:
            process = subprocess.Popen(["bash", str(LAUNCH_SCRIPT)], cwd=ROOT,
                env=_environment(model, port, python_executable=python_executable, context_length=context_length),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        observed = _process(process.pid)
        _require(observed and observed["pgid"] == observed["pid"] == observed["session"], "server_not_owned_session_leader")
        owner = observed
        _write(output / "process.json", {"schema": "eva.rsi-owned-serving-process.v1", **owner,
                "launch_argv": ["bash", str(LAUNCH_SCRIPT)], "created_at": _now(),
                "model_path": str(model), "port": port, "private_log": "server.log"})
        deadline = time.monotonic() + startup_timeout
        while True:
            _require(process.poll() is None and _same(owner), "owned_server_exited_before_ready")
            try:
                info = _request(f"http://127.0.0.1:{port}/model_info")
            except (OSError, urllib.error.URLError):
                info = None
            if info is not None:
                _require(info.get("model_path") == str(model), "server_returned_different_model_path")
                known = _members(owner)
                _require(_listener_owned(port, known), "ready_port_not_owned_by_launched_group")
                argv = _argv(process.pid)
                _require("--model-path" in argv and argv[argv.index("--model-path") + 1] == str(model),
                         "owned_process_model_argv_differs")
                break
            _require(time.monotonic() < deadline, "server_readiness_timeout")
            time.sleep(0.25)
        known = _members(owner)
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        payload = {"model": MODEL_ALIAS, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe only visible colors and general shape in one short sentence. Do not name a medical condition, diagnose, or give advice."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(image).decode()}}]}],
            "max_tokens": 1024, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": True}}
        body = json.dumps(payload).encode()
        canary = {"schema": "eva.qwen-final-serving-image-canary.v1", "receipt_id": str(uuid4()),
                  "started_at": _now(), "checkpoint_identity_blake3": blake3(identity_path.read_bytes()).hexdigest(),
                  "exact_final_model_path": str(model), "server_pid": process.pid,
                  "server_start_ticks": owner["start_ticks"], "actual_server_argv": argv, "model_info": info,
                  "public_image_path": str(image_path), "image_bytes": len(image), "image_blake3": blake3(image).hexdigest(),
                  "request_body_blake3": blake3(body).hexdigest(), "image_requests_attempted": 1,
                  "clinical_reference_supplied": False, "clinical_score_emitted": False,
                  "max_output_tokens": 1024, "thinking_requested": True,
                  "private_reasoning_recorded": False, "automatic_retry": False}
        try:
            result = _request(f"http://127.0.0.1:{port}/v1/chat/completions", body=body, timeout=180)
            choices = result.get("choices")
            _require(result.get("model") == MODEL_ALIAS and isinstance(choices, list) and len(choices) == 1
                     and isinstance(choices[0].get("message"), dict), "image_canary_response_shape")
            _require(_same(owner) and _argv(process.pid) == argv, "server_identity_changed_during_canary")
            canary.update(status="complete", http_status=200, response_model=result["model"],
                          finish_reason=choices[0].get("finish_reason"), usage=result.get("usage"),
                          multimodal_request_accepted=True,
                          response_body_blake3=blake3(json.dumps(result, sort_keys=True).encode()).hexdigest())
        except Exception as exc:
            canary.update(status="failed", error_type=type(exc).__name__, multimodal_request_accepted=False)
            raise ServingError("public_image_canary_failed; no_retry") from None
        finally:
            canary["completed_at"] = _now()
            _write(canary_path, canary)
        yield ServingSession(identity_path, canary_path, output, process.pid)
    except BaseException as exc:
        failure = type(exc).__name__
        raise
    finally:
        cleanup = _stop_owned(process, owner, known, stop_timeout) if process is not None and owner else {
            "signals": [], "ownership_unavailable_no_signal_sent": True}
        _write(output / "session-exit.json", {"schema": "eva.rsi-serving-session-exit.v1",
               "completed_at": _now(), "exception_type": failure, "cleanup": cleanup})
        if cleanup.get("verified_members_remaining", 0):
            raise ServingError("owned_server_members_remain_after_cleanup")
