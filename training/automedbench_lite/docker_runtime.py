"""Bounded public-workspace Python in the already-supported Docker sandbox.

Isolation pattern: rlevo_med_research.execution.execute_python_contract; this is
a separate evaluation tool, not a replacement for its signed canonical schema.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import tempfile
import time
from uuid import uuid4

from .adapter import EvaluationError, canonical, file_digest, write_once

MAX_STREAM = 128 * 1024
MAX_FILES = 256
MAX_WORKSPACE = 128 * 1024 * 1024


def resolve_image(reference: str) -> str:
    result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", reference],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15, check=False)
    image = result.stdout.decode().strip()
    if result.returncode or re.fullmatch(r"sha256:[a-f0-9]{64}", image) is None:
        raise EvaluationError("sandbox_image_not_present")
    return image


def workspace_inventory(workspace: Path) -> dict:
    rows = []
    for path in sorted(workspace.rglob("*")):
        if path.is_symlink():
            raise EvaluationError("sandbox_workspace_symlink")
        if path.is_file():
            info = path.stat()
            if info.st_nlink != 1 or info.st_size > MAX_WORKSPACE:
                raise EvaluationError("sandbox_workspace_file_invalid")
            rows.append({"path": path.relative_to(workspace).as_posix(), "bytes": info.st_size, "blake3": file_digest(path)})
            if len(rows) > MAX_FILES or sum(row["bytes"] for row in rows) > MAX_WORKSPACE:
                raise EvaluationError("sandbox_workspace_limit")
        elif not path.is_dir():
            raise EvaluationError("sandbox_workspace_special_file")
    return {"files": rows, "file_count": len(rows), "bytes": sum(row["bytes"] for row in rows)}


def create_command(*, image: str, name: str, workspace: Path, submission: Path) -> list[str]:
    if re.fullmatch(r"sha256:[a-f0-9]{64}", image) is None:
        raise EvaluationError("sandbox_requires_immutable_image")
    return ["docker", "create", "--pull", "never", "--name", name, "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--pids-limit", "64",
            "--memory", "2048m", "--cpus", "2", "--ulimit", "nofile=64:64", "--ulimit",
            f"fsize={MAX_WORKSPACE}:{MAX_WORKSPACE}", "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=256m",
            "--user", f"{os.getuid()}:{os.getgid()}", "--env", "HOME=/tmp", "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "PYTHONHASHSEED=0", "--env", "OPENBLAS_NUM_THREADS=2", "--env", "OMP_NUM_THREADS=2",
            "--mount", f"type=bind,source={workspace},target=/workspace",
            "--mount", f"type=bind,source={workspace / 'inputs'},target=/workspace/inputs,readonly",
            "--mount", f"type=bind,source={workspace / 'task.json'},target=/workspace/task.json,readonly",
            "--mount", f"type=bind,source={submission},target=/submission/solution.py,readonly",
            "--workdir", "/workspace", "--entrypoint", "python", image, "-I", "-B", "/submission/solution.py"]


def verify_container(document: dict, *, image: str, workspace: Path, submission: Path) -> dict:
    host, config = document["HostConfig"], document["Config"]
    mounts = {(row["Source"], row["Destination"], row["RW"]) for row in document["Mounts"] if row["Type"] == "bind"}
    expected = {(str(workspace), "/workspace", True), (str(workspace / "inputs"), "/workspace/inputs", False),
                (str(workspace / "task.json"), "/workspace/task.json", False), (str(submission), "/submission/solution.py", False)}
    if (document["Image"] != image or host["NetworkMode"] != "none" or not host["ReadonlyRootfs"]
            or host["Privileged"] or set(host["CapDrop"] or []) != {"ALL"}
            or not any("no-new-privileges" in item for item in host["SecurityOpt"] or [])
            or host.get("DeviceRequests") or host.get("Devices") or host.get("PidMode") == "host"
            or host["Memory"] != 2048 * 1024 * 1024 or host["NanoCpus"] != 2_000_000_000
            or host["PidsLimit"] != 64 or config["User"] != f"{os.getuid()}:{os.getgid()}"
            or mounts != expected or config["Entrypoint"] != ["python"]):
        raise EvaluationError("docker_isolation_configuration_mismatch")
    return {"image": image, "network": "none", "readonly_rootfs": True, "cap_drop": ["ALL"],
            "no_new_privileges": True, "devices": [], "public_workspace_bind_only": True,
            "input_mount_readonly": True, "scorer_or_socket_mount": False,
            "memory_bytes": host["Memory"], "cpus": 2, "pids_limit": 64, "verified_before_start": True}


def bounded_start(name: str, timeout: float) -> dict:
    process = subprocess.Popen(["docker", "start", "--attach", name], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True, start_new_session=True)
    selector = selectors.DefaultSelector()
    streams = {}
    for label, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, label)
        streams[label] = bytearray()
    stopped = False
    timed_out = False
    overflow = False
    deadline = time.monotonic() + timeout
    while selector.get_map():
        if time.monotonic() >= deadline or overflow:
            if not stopped:
                timed_out = time.monotonic() >= deadline
                subprocess.run(["docker", "kill", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stopped = True
        for key, _ in selector.select(0.1):
            chunk = os.read(key.fd, 65536)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            buffer = streams[key.data]
            available = MAX_STREAM - len(buffer)
            buffer.extend(chunk[:available])
            overflow |= len(chunk) > available
    process.wait(timeout=15)
    selector.close()
    return {"stdout": bytes(streams["stdout"]).decode("utf-8", errors="replace"),
            "stderr": bytes(streams["stderr"]).decode("utf-8", errors="replace"),
            "timed_out": timed_out, "stream_limit_exceeded": overflow, "cli_exit_code": process.returncode}


def execute_python(*, workspace: Path, code: str, image: str, audit_root: Path, timeout: float = 120) -> dict:
    if not isinstance(code, str) or not code.strip() or len(code.encode()) > 65536 or not 1 <= timeout <= 180:
        raise EvaluationError("public_python_request_invalid")
    workspace = Path(workspace).resolve(strict=True)
    audit_root = Path(audit_root).resolve()
    if workspace.is_relative_to(audit_root) or audit_root.is_relative_to(workspace):
        raise EvaluationError("code_audit_must_be_outside_actor_workspace")
    audit_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    execution_id = str(uuid4())
    audit = audit_root / execution_id
    audit.mkdir(mode=0o700)
    before = workspace_inventory(workspace)
    submission = audit / "solution.py"
    descriptor = os.open(submission, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(code)
    name = "eva-automed-" + execution_id
    started = False
    try:
        create = subprocess.run(create_command(image=image, name=name, workspace=workspace, submission=submission),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30)
        if create.returncode:
            raise EvaluationError("docker_container_create_failed")
        inspected = subprocess.run(["docker", "inspect", name], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15, check=True)
        isolation = verify_container(json.loads(inspected.stdout)[0], image=image, workspace=workspace, submission=submission)
        started = True
        outcome = bounded_start(name, timeout)
        inspected = subprocess.run(["docker", "inspect", name], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15, check=True)
        state = json.loads(inspected.stdout)[0]["State"]
        after = workspace_inventory(workspace)
        result = {"schema": "eva.automedbench-public-python.v1", "execution_id": execution_id,
            "code_blake3": file_digest(submission), "isolation": isolation, "exit_code": state["ExitCode"],
            "oom_killed": state["OOMKilled"], "process_started": state["StartedAt"] != "0001-01-01T00:00:00Z",
            "before": before, "after": after, **outcome}
        write_once(audit / "execution.json", result)
        return result
    finally:
        # Only this invocation's exact UUID container; never remove images/jobs.
        subprocess.run(["docker", "rm", "--force", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
