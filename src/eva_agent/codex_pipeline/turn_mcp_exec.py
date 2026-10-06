"""Exec the turn MCP proxy with an exact, secret-minimal environment.

Codex app-server needs provider credentials, but the candidate-bound MCP
process does not.  Codex may merge an MCP server's configured environment
with its own process environment, so clearing ``os.environ`` inside the proxy
is too late: Linux keeps the original process environment visible through
``/proc/<pid>/environ``.  This tiny stdlib-only boundary reads the three
turn-scoped values and then *execs* the proxy with a newly constructed
environment.  Provider and repository credentials therefore never cross the
final proxy exec boundary.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys


_SOCKET_ENV = "EVA_TURN_MCP_SOCKET"
_NONCE_ENV = "EVA_TURN_MCP_NONCE"
_MAXIMUM_ENV = "EVA_TURN_MCP_MAXIMUM"
_MAXIMUM_PARALLEL_CALLS = 256


def _real_file(raw: str, *, executable: bool, label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise SystemExit(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SystemExit(f"{label} cannot be reopened") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(f"{label} must be a real regular file")
    resolved = path.resolve(strict=True)
    if executable and not os.access(resolved, os.X_OK):
        raise SystemExit(f"{label} must be executable")
    return resolved


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("turn MCP exec boundary expects Python and proxy paths")

    socket_path = os.environ.get(_SOCKET_ENV, "")
    nonce = os.environ.get(_NONCE_ENV, "")
    raw_maximum = os.environ.get(_MAXIMUM_ENV, "")
    try:
        maximum = int(raw_maximum)
    except ValueError:
        maximum = 0
    if (
        not socket_path
        or not Path(socket_path).is_absolute()
        or "\x00" in socket_path
        or not nonce
        or "\x00" in nonce
        or not 1 <= maximum <= _MAXIMUM_PARALLEL_CALLS
    ):
        raise SystemExit("authenticated turn MCP environment is incomplete")

    python = _real_file(sys.argv[1], executable=True, label="proxy Python")
    proxy = _real_file(sys.argv[2], executable=False, label="proxy script")
    os.umask(0o077)
    environment = {
        _SOCKET_ENV: socket_path,
        _NONCE_ENV: nonce,
        _MAXIMUM_ENV: str(maximum),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "TZ": "UTC",
    }
    argv = (str(python), "-I", "-S", "-B", str(proxy))
    os.execve(str(python), argv, environment)
    raise AssertionError("os.execve unexpectedly returned")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
