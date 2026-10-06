"""Network-free, single-process CPU execution with an explicit stricter profile."""
from __future__ import annotations
import ctypes
import ctypes.util
import errno
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import time
from .integrity import byte_digest, file_digest

DENIED_SYSCALLS = ("fork", "vfork", "clone", "clone3", "unshare", "setns", "mount", "umount2",
                   "ptrace", "bpf", "keyctl", "symlink", "symlinkat", "link", "linkat", "sched_setaffinity")


def seccomp_filter(path):
    library = ctypes.util.find_library("seccomp")
    if not library:
        raise RuntimeError("libseccomp_required_for_single_process_admission")
    seccomp = ctypes.CDLL(library, use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_rule_add.restype = ctypes.c_int
    seccomp.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    context = seccomp.seccomp_init(0x7FFF0000)
    if not context:
        raise RuntimeError("seccomp_initialization_failed")
    installed = []
    try:
        for name in DENIED_SYSCALLS:
            number = seccomp.seccomp_syscall_resolve_name(name.encode())
            if number < 0:
                if name in {"fork", "vfork", "clone3"}:
                    continue
                raise RuntimeError("seccomp_required_syscall_unresolved")
            if seccomp.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0) != 0:
                raise RuntimeError("seccomp_rule_install_failed")
            installed.append(name)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            if seccomp.seccomp_export_bpf(context, descriptor) != 0:
                raise RuntimeError("seccomp_export_failed")
        finally:
            os.close(descriptor)
    finally:
        seccomp.seccomp_release(context)
    return installed


def footprint(roots, max_files, max_bytes):
    files = size = 0
    invalid = False
    for root in roots:
        for directory, dirs, names in os.walk(root, followlinks=False):
            for name in [*dirs, *names]:
                path = Path(directory) / name
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    continue
                if stat.S_ISLNK(info.st_mode) or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                    invalid = True
                if stat.S_ISREG(info.st_mode):
                    files += 1
                    size += info.st_size
                elif stat.S_ISDIR(info.st_mode):
                    files += 1
                if invalid or files > max_files or size > max_bytes:
                    return {"files": files, "bytes": size, "invalid_files": invalid, "early_limit_stop": True}
    return {"files": files, "bytes": size, "invalid_files": invalid}


def execute(*, bwrap, runtime_root, stage_root, code, limits, python="/usr/bin/python3.12"):
    runtime_root, stage_root, bwrap = Path(runtime_root).resolve(), Path(stage_root).resolve(), Path(bwrap).resolve()
    if not bwrap.is_file() or not (runtime_root / python.lstrip("/")).is_file():
        raise RuntimeError("scientific_runtime_unavailable")
    if not isinstance(code, str) or not code.strip() or len(code.encode()) > 1024 * 1024:
        raise ValueError("bounded_source_code_required")
    work, inputs, temporary = (stage_root / n for n in ("work", "input", "tmp"))
    for path in (work, inputs, temporary):
        path.mkdir(parents=True, exist_ok=True)
    source = stage_root / "source.py"
    source.write_text(code)
    source.chmod(0o400)
    filter_path = stage_root / "single-process.bpf"
    denied = seccomp_filter(filter_path)
    descriptor = os.open(filter_path, os.O_RDONLY)
    allowed_cpus = sorted(os.sched_getaffinity(0))[:max(1, int(limits["cpus"]))]
    memory = int(limits["memory_mebibytes"]) * 1024 * 1024
    max_bytes, max_files = int(limits["max_workspace_bytes"]), int(limits["max_workspace_files"])
    timeout = float(limits["wall_time_seconds"])
    args = [str(bwrap), "--die-with-parent", "--new-session", "--unshare-all", "--cap-drop", "ALL", "--clearenv"]
    for name in ("usr", "opt", "etc", "bin", "sbin", "lib", "lib64"):
        path = runtime_root / name
        if path.is_symlink():
            target = os.readlink(path)
            if target.startswith("/") or ".." in Path(target).parts:
                raise RuntimeError("runtime_root_symlink_unsafe")
            args += ["--symlink", target, "/" + name]
        elif path.is_dir():
            args += ["--ro-bind", str(path), "/" + name]
    # This node forbids mounting a new procfs. Empty /proc never exposes host PID data.
    args += ["--dir", "/proc", "--dev", "/dev", "--dir", "/workspace", "--dir", "/submission",
             "--ro-bind", str(inputs), "/workspace/input", "--bind", str(work), "/workspace/work",
             "--bind", str(temporary), "/tmp", "--ro-bind", str(source), "/submission/source.py",
             "--remount-ro", "/", "--remount-ro", "/dev",
             "--chdir", "/workspace", "--seccomp", str(descriptor)]
    environment = {"PATH": "/usr/bin:/usr/local/bin:/bin", "LANG": "C.UTF-8", "HOME": "/tmp",
                   "TMPDIR": "/tmp", "XDG_CACHE_HOME": "/tmp/cache", "HF_HOME": "/tmp/hf",
                   "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "NUMEXPR_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    for key, value in environment.items():
        args += ["--setenv", key, value]
    args += [python, "-I", "-B", "/submission/source.py"]
    args = ["/usr/bin/prlimit", f"--as={memory}:{memory}", f"--fsize={max_bytes}:{max_bytes}",
            "--nofile=64:64", "--core=0:0", "--", "/usr/bin/taskset", "--cpu-list",
            ",".join(map(str, allowed_cpus)), *args]
    process = None
    selector = selectors.DefaultSelector()
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    caps = {"stdout": int(limits["max_stdout_bytes"]), "stderr": int(limits["max_stderr_bytes"])}
    started = time.monotonic()
    violation = None
    peak = {"files": 0, "bytes": 0}
    try:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env={}, close_fds=True, pass_fds=(descriptor,), start_new_session=True)
        os.close(descriptor)
        descriptor = -1
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        stopped = False
        while selector.get_map() or process.poll() is None:
            usage = footprint((work, temporary), max_files, max_bytes)
            peak = {k: max(peak[k], usage[k]) for k in peak}
            if time.monotonic() - started >= timeout:
                violation = violation or "wall_time_limit"
            if usage["files"] > max_files or usage["bytes"] > max_bytes or usage["invalid_files"]:
                violation = violation or "workspace_limit_or_invalid_file"
            if violation and not stopped:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stopped = True
            for key, _ in selector.select(0.02):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                room = caps[key.data] - len(streams[key.data])
                streams[key.data].extend(chunk[:room])
                if len(chunk) > room:
                    violation = violation or "stream_limit"
        process.wait(timeout=10)
    finally:
        selector.close()
        if descriptor >= 0:
            os.close(descriptor)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
        if process is not None:
            process.stdout.close()
            process.stderr.close()
    final = footprint((work, temporary), max_files, max_bytes)
    if final["bytes"] > max_bytes or final["files"] > max_files or final["invalid_files"]:
        violation = violation or "workspace_limit_or_invalid_file"
    return {"backend": "bubblewrap-seccomp-single-process-v1", "exit_code": process.returncode,
            "wall_seconds": round(time.monotonic() - started, 6), "violation": violation,
            "successful_exit": process.returncode == 0 and violation is None,
            "source_blake3": file_digest(source), "bwrap_binary_blake3": file_digest(bwrap),
            "seccomp_blake3": file_digest(filter_path), "denied_syscalls": denied,
            "cpu_affinity_count": len(allowed_cpus), "address_space_limit_bytes": memory,
            "actor_process_limit": 1, "actor_threads": 1, "network_namespace_isolated": True,
            "actor_affinity_expansion_denied": True, "unmonitored_root_and_dev_readonly": True,
            "host_proc_exposed": False, "proc_mode": "empty", "gpu_devices_exposed": False,
            "credential_environment_inherited": False, "source_policy_limits": dict(limits),
            "workspace_limits": {"max_bytes": max_bytes, "max_files": max_files,
                "enforcement": "20ms monitor plus final verification; per-file hard RLIMIT_FSIZE; directories count toward max_files",
                "kernel_aggregate_filesystem_quota": False, "observed_peak": peak, "final": final},
            **{name: bytes(value).decode("utf-8", errors="replace") for name, value in streams.items()}}
