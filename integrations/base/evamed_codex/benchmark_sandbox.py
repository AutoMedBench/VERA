"""CPU scientific code execution with only the public actor tree mounted."""
from __future__ import annotations

import os
import ctypes
import ctypes.util
import errno
from pathlib import Path
import selectors
import signal
import subprocess
import time
from uuid import uuid4


MAX_STREAM = 128 * 1024


def seccomp_filter(path: Path) -> tuple[str, ...]:
    """Keep threads/processes available, but bind CPU affinity and namespace policy."""
    denied = ("sched_setaffinity", "unshare", "setns", "mount", "umount2", "ptrace", "bpf", "keyctl")
    library = ctypes.util.find_library("seccomp")
    if not library:
        raise RuntimeError("libseccomp_required_for_benchmark_cpu_boundary")
    lib = ctypes.CDLL(library, use_errno=True)
    lib.seccomp_init.argtypes, lib.seccomp_init.restype = [ctypes.c_uint32], ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes, lib.seccomp_syscall_resolve_name.restype = [ctypes.c_char_p], ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_export_bpf.argtypes, lib.seccomp_export_bpf.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x7FFF0000)
    if not context:
        raise RuntimeError("seccomp_initialization_failed")
    try:
        for name in denied:
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number < 0 or lib.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0):
                raise RuntimeError("required_seccomp_rule_failed")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            if lib.seccomp_export_bpf(context, descriptor):
                raise RuntimeError("seccomp_export_failed")
        finally:
            os.close(descriptor)
    finally:
        lib.seccomp_release(context)
    return denied


def command(*, bwrap: Path, runtime: Path, workspace: Path, source: Path, temporary: Path,
            python: str = "/usr/bin/python3.12", seccomp_fd: int | None = None,
            public_workspace: Path | None = None) -> list[str]:
    if not bwrap.is_file() or not runtime.is_dir() or not (runtime / python.lstrip("/")).exists():
        raise ValueError("benchmark_scientific_runtime_unavailable")
    args = [str(bwrap), "--die-with-parent", "--new-session", "--unshare-all", "--cap-drop", "ALL", "--clearenv"]
    # An empty namespace root gets only scientific runtime directories. No host
    # workspace, home, secrets, scorer tree, Docker socket, or GPU is mounted.
    for name in ("usr", "opt", "etc", "bin", "sbin", "lib", "lib64"):
        path = runtime / name
        if path.is_symlink():
            target = os.readlink(path)
            if target.startswith("/") or ".." in Path(target).parts:
                raise ValueError("unexpected_runtime_root_symlink")
            args += ["--symlink", target, "/" + name]
        elif path.is_dir():
            args += ["--ro-bind", str(path), "/" + name]
    args += ["--dir", "/proc", "--dev", "/dev", "--dir", "/submission",
             "--bind", str(temporary), "/tmp", "--bind", str(workspace), "/workspace"]
    from .benchmark_cpu_publication import READ_ONLY
    public_workspace = public_workspace or workspace
    for name in READ_ONLY:
        if (public_workspace / name).exists():
            args += ["--ro-bind", str(public_workspace / name), "/workspace/" + name]
    args += ["--ro-bind", str(source), "/submission/solution.py",
             "--remount-ro", "/", "--remount-ro", "/dev",
             "--chdir", "/workspace", "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
             "--setenv", "LANG", "C.UTF-8", "--setenv", "TMPDIR", "/tmp",
             "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
             "--setenv", "OPENBLAS_NUM_THREADS", "2", "--setenv", "OMP_NUM_THREADS", "2",
             "--setenv", "CUDA_VISIBLE_DEVICES", "", "--setenv", "HIP_VISIBLE_DEVICES", "",
             python, "-I", "-B", "/submission/solution.py"]
    if seccomp_fd is not None:
        args[1:1] = ["--seccomp", str(seccomp_fd)]
    return args


def execute(*, bwrap: Path, runtime: Path, workspace: Path, audit_root: Path, code: str,
            timeout: float, inventory, python: str = "/usr/bin/python3.12",
            track_deadline: float | None = None) -> dict:
    from training.automedbench_lite.adapter import file_digest, write_once

    if not isinstance(code, str) or not code.strip() or len(code.encode()) > 65536 or not 1 <= timeout <= 180:
        raise ValueError("scientific_code_request_invalid")
    execution_id = str(uuid4())
    audit = audit_root / execution_id
    audit.mkdir(parents=True, mode=0o700)
    temporary = audit / "tmp"
    temporary.mkdir(mode=0o700)
    source = audit / "solution.py"
    source.write_text(code)
    source.chmod(0o400)
    before = inventory.capture()
    from .benchmark_cpu_publication import prepare_private, publish_private
    private_workspace = prepare_private(workspace, audit, before)
    filter_path = audit / "benchmark.bpf"
    denied = seccomp_filter(filter_path)
    filter_fd = os.open(filter_path, os.O_RDONLY)
    argv = command(bwrap=bwrap, runtime=runtime, workspace=private_workspace, source=source,
                   temporary=temporary, python=python, seccomp_fd=filter_fd, public_workspace=workspace)

    # No preexec_fn: MCP is multithreaded, so fork-time Python callbacks are
    # unsafe. The exec wrappers apply limits before any actor code starts.
    allowed = sorted(os.sched_getaffinity(0))[:2]
    argv = ["/usr/bin/prlimit", f"--as={8 * 1024**3}:{8 * 1024**3}",
            f"--fsize={512 * 1024**2}:{512 * 1024**2}", "--nofile=128:128", "--core=0:0", "--",
            "/usr/bin/taskset", "--cpu-list", ",".join(map(str, allowed)), *argv]
    start = time.monotonic()
    effective_deadline = min(start + timeout, track_deadline) if track_deadline is not None else start + timeout
    if effective_deadline <= start:
        os.close(filter_fd)
        raise ValueError("track_wall_clock_budget_exhausted")
    try:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env={}, start_new_session=True,
                                   close_fds=True, pass_fds=(filter_fd,))
    finally:
        os.close(filter_fd)
    try:
        identity = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
        write_once(audit / "process.json", {"schema": "eva.evamed-bwrap-process.v1",
            "execution_id": execution_id, "pid": process.pid, "process_group": process.pid,
            "start_ticks": identity[19], "code_blake3": file_digest(source),
            "started_monotonic": start, "effective_deadline_monotonic": effective_deadline,
            "track_deadline_monotonic": track_deadline, "parent_death_terminates_namespace": True})
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
        raise
    selector = selectors.DefaultSelector()
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)
    overflow = timed_out = stopped = False
    kill_requested = None
    try:
        while selector.get_map() or process.poll() is None:
            timed_out |= time.monotonic() >= effective_deadline
            if (timed_out or overflow) and not stopped:
                kill_requested = time.monotonic()
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stopped = True
            wait_seconds = .1 if stopped else min(.1, max(0., effective_deadline - time.monotonic()))
            for key, _ in selector.select(wait_seconds):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                available = MAX_STREAM - len(streams[key.data])
                streams[key.data].extend(chunk[:available])
                overflow |= len(chunk) > available
        process.wait(timeout=10)
    finally:
        selector.close()
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    exited = time.monotonic()
    publication, publication_failure = None, None
    try:
        publication = publish_private(workspace, audit, before, track_deadline,
                                      execution_succeeded=process.returncode == 0 and not timed_out and not overflow)
    except Exception as exc:
        publication_failure = type(exc).__name__
        write_once(audit / 'cpu-publication-failure.json', {'error_type': publication_failure,
                   'native_score_admissible': False, 'track_deadline_monotonic': track_deadline})
    result = {"schema": "eva.evamed-bwrap-python.v1", "execution_id": execution_id,
              "code_blake3": file_digest(source), "exit_code": process.returncode,
              "process_started": True, "timed_out": timed_out, "stream_limit_exceeded": overflow,
              "wall_seconds": time.monotonic() - start,
              "kill_requested_monotonic": kill_requested,
              "process_exit_observed_monotonic": exited,
              "requested_timeout_seconds": timeout, "effective_deadline_monotonic": effective_deadline,
              "track_deadline_monotonic": track_deadline,
              "cpu_publication_document_blake3": publication["document_blake3"] if publication else None,
              "publication_infrastructure_error": publication_failure,
              "publication_policy_error": publication.get('policy_artifact_rejection') if publication else None,
              **{name: bytes(data).decode("utf-8", errors="replace") for name, data in streams.items()},
              "before": before, "after": inventory.capture(),
              "isolation": {"backend": "bubblewrap", "network_namespace": "isolated",
                            "pid_namespace": "isolated", "gpu_devices": False,
                            "public_workspace_only": True, "runtime_read_only": True,
                            "actor_direct_writable_mount": False,
                            "private_edits_publish_only_before_track_deadline": True,
                            "inputs_read_only": True, "namespace_root_read_only": True, "dev_read_only": True,
                            "seccomp_denied_syscalls": denied, "seccomp_blake3": file_digest(filter_path),
                            "cpu_affinity_expansion_denied": True, "credential_environment_inherited": False,
                            "address_space_limit_bytes": 8 * 1024**3,
                            "file_size_limit_bytes": 512 * 1024**2, "cpu_affinity_count": 2,
                            "physical_memory_cgroup_limit": None,
                            "outside_workspace_persistent_artifacts": False}}
    write_once(audit / "execution.json", result)
    return result
