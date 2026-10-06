"""Fail-closed exec boundary for a pinned Codex app-server child.

The published Python SDK intentionally overlays ``CodexConfig.env`` on a
copy of the parent environment.  That behaviour is convenient for interactive
use, but a production evaluation worker must not expose unrelated host
credentials to Codex or to an MCP subprocess that Codex starts.  This module
is therefore used as a tiny, static process boundary:

1. the SDK starts an isolated Python interpreter with this file;
2. this file validates a narrowly shaped app-server argv;
3. it replaces the environment with fixed locale/path/isolation settings and
   the explicitly named route credentials; and
4. it calls ``execve`` for the content-pinned Codex binary.

Credential *names* may be present in argv and receipts.  Credential values
never are.  The module deliberately has no project imports so it can run with
``python -I -S`` and cannot be redirected through ``PYTHONPATH`` or user-site
packages.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat
import sys
from types import MappingProxyType
from typing import Mapping, Sequence


class CodexChildExecError(ValueError):
    """The isolated Codex child specification is unsafe or incomplete."""


_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_CREDENTIAL_NAME = re.compile(
    r"^(?:EVA_CODEX_[A-Z0-9_]*TOKEN|[A-Z][A-Z0-9_]*(?:API_KEY|ACCESS_TOKEN))$"
)
_DENIED_ENV_NAMES = frozenset(
    {
        "BASH_ENV",
        "CDPATH",
        "CODEX_HOME",
        "ENV",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "HOME",
        "IFS",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "RUSTC_WRAPPER",
        "RUST_LOG",
        "SHELL",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    }
)
_FIXED_PATH = "/usr/bin:/bin"
_BASE_ENV_KEYS = (
    "CODEX_HOME",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_NOSYSTEM",
    "LANG",
    "LC_ALL",
    "PATH",
    "TMPDIR",
    "TZ",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
)


def _absolute_normalized(path: str | Path, *, label: str) -> Path:
    if not isinstance(path, (str, Path)):
        raise CodexChildExecError(f"{label} path differs")
    raw = str(path)
    if not raw or "\x00" in raw:
        raise CodexChildExecError(f"{label} path differs")
    candidate = Path(raw)
    if not candidate.is_absolute() or candidate != Path(os.path.normpath(raw)):
        raise CodexChildExecError(f"{label} path must be absolute and normalized")
    return candidate


def _regular_executable(path: str | Path, *, label: str) -> Path:
    candidate = _absolute_normalized(path, label=label)
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise CodexChildExecError(f"{label} executable is unavailable") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or not os.access(candidate, os.X_OK)
    ):
        raise CodexChildExecError(f"{label} executable topology differs")
    return candidate


def _private_directory(path: str | Path, *, label: str) -> Path:
    candidate = _absolute_normalized(path, label=label)
    try:
        info = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CodexChildExecError(f"{label} directory is unavailable") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or resolved != candidate
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
    ):
        raise CodexChildExecError(f"{label} directory topology differs")
    return candidate


def _credential_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(values)
    if not names or len(names) != len(set(names)):
        raise CodexChildExecError("credential environment names differ")
    for name in names:
        if (
            not isinstance(name, str)
            or _ENV_NAME.fullmatch(name) is None
            or _CREDENTIAL_NAME.fullmatch(name) is None
            or name in _DENIED_ENV_NAMES
        ):
            raise CodexChildExecError("credential environment name is unsafe")
    return tuple(sorted(names))


def _codex_app_server_args(values: Sequence[str]) -> tuple[str, ...]:
    args = tuple(values)
    if not args or any(not isinstance(value, str) or not value or "\x00" in value for value in args):
        raise CodexChildExecError("Codex app-server argv differs")
    try:
        app_server_index = args.index("app-server")
    except ValueError:
        raise CodexChildExecError("Codex app-server subcommand is missing") from None
    if args.count("app-server") != 1:
        raise CodexChildExecError("Codex app-server subcommand is duplicated")

    prefix = args[:app_server_index]
    index = 0
    while index < len(prefix):
        if prefix[index] != "--config" or index + 1 >= len(prefix):
            raise CodexChildExecError("Codex global argv is not an exact config pair")
        override = prefix[index + 1]
        if "\n" in override or "\r" in override:
            raise CodexChildExecError("Codex config override contains a line break")
        index += 2

    suffix = args[app_server_index + 1 :]
    if suffix not in {
        ("--strict-config", "--listen", "stdio://"),
        ("--listen", "stdio://", "--strict-config"),
    }:
        raise CodexChildExecError("Codex app-server transport must be strict stdio")
    return args


@dataclass(frozen=True, repr=False)
class SanitizedCodexExecPlan:
    """Public launch argv plus secret-safe child-environment commitments."""

    launch_args: tuple[str, ...]
    credential_env_names: tuple[str, ...]
    codex_bin: str
    isolation_root: str

    def __repr__(self) -> str:
        return (
            "SanitizedCodexExecPlan(launch_args=<redacted>, "
            f"credential_env_names={self.credential_env_names!r}, "
            f"codex_bin={self.codex_bin!r}, isolation_root={self.isolation_root!r})"
        )

    def public_metadata(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "schema": "eva.codex-sanitized-child-exec.v1",
                "credential_env_names": list(self.credential_env_names),
                "credential_values_recorded": False,
                "inherited_parent_environment": False,
                "codex_bin": self.codex_bin,
                "isolation_root": self.isolation_root,
                "base_environment_keys": list(_BASE_ENV_KEYS),
                "strict_stdio_app_server": True,
            }
        )


def build_sanitized_codex_exec_plan(
    *,
    python_bin: str | Path,
    wrapper_script: str | Path,
    codex_bin: str | Path,
    isolation_root: str | Path,
    credential_env_names: Sequence[str],
    codex_args: Sequence[str],
) -> SanitizedCodexExecPlan:
    """Build an SDK ``launch_args_override`` without embedding secret values."""

    python = _regular_executable(python_bin, label="isolated Python")
    wrapper = _absolute_normalized(wrapper_script, label="Codex exec wrapper")
    try:
        wrapper_info = wrapper.lstat()
    except OSError as exc:
        raise CodexChildExecError("Codex exec wrapper is unavailable") from exc
    if (
        stat.S_ISLNK(wrapper_info.st_mode)
        or not stat.S_ISREG(wrapper_info.st_mode)
        or wrapper_info.st_nlink != 1
    ):
        raise CodexChildExecError("Codex exec wrapper topology differs")
    codex = _regular_executable(codex_bin, label="pinned Codex")
    isolated = _private_directory(isolation_root, label="Codex isolation root")
    names = _credential_names(credential_env_names)
    args = _codex_app_server_args(codex_args)
    launch = (
        str(python),
        "-I",
        "-S",
        str(wrapper),
        str(codex),
        str(isolated),
        str(len(names)),
        *names,
        "--",
        *args,
    )
    return SanitizedCodexExecPlan(
        launch_args=launch,
        credential_env_names=names,
        codex_bin=str(codex),
        isolation_root=str(isolated),
    )


def sanitized_child_environment(
    *,
    isolation_root: str | Path,
    credential_env_names: Sequence[str],
    source_environment: Mapping[str, str],
) -> Mapping[str, str]:
    """Construct the exact post-wrapper environment without exposing values."""

    isolated = _private_directory(isolation_root, label="Codex isolation root")
    names = _credential_names(credential_env_names)
    values: dict[str, str] = {
        "PATH": _FIXED_PATH,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "CODEX_HOME": str(isolated / "codex"),
        "TMPDIR": str(isolated / "tmp"),
        "XDG_CACHE_HOME": str(isolated / "cache"),
        "XDG_CONFIG_HOME": str(isolated / "config"),
        "XDG_DATA_HOME": str(isolated / "data"),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    for child in ("codex", "tmp", "cache", "config", "data"):
        directory = isolated / child
        directory.mkdir(mode=0o700, exist_ok=True)
        _private_directory(directory, label=f"Codex isolated {child}")
    for name in names:
        value = source_environment.get(name)
        if not isinstance(value, str) or not value or "\x00" in value:
            raise CodexChildExecError(
                f"required credential environment value is unavailable: {name}"
            )
        values[name] = value
    return MappingProxyType(values)


def _parse_wrapper_argv(argv: Sequence[str]) -> tuple[Path, Path, tuple[str, ...], tuple[str, ...]]:
    if len(argv) < 5:
        raise CodexChildExecError("Codex exec wrapper argv differs")
    codex = _regular_executable(argv[0], label="pinned Codex")
    isolated = _private_directory(argv[1], label="Codex isolation root")
    try:
        count = int(argv[2], 10)
    except ValueError:
        raise CodexChildExecError("credential environment count differs") from None
    if count < 1 or len(argv) <= 3 + count or argv[3 + count] != "--":
        raise CodexChildExecError("credential environment count differs")
    names = _credential_names(tuple(argv[3 : 3 + count]))
    args = _codex_app_server_args(tuple(argv[4 + count :]))
    return codex, isolated, names, args


def main(argv: Sequence[str] | None = None) -> int:
    """Validate, sanitize, and replace this process with pinned Codex."""

    try:
        codex, isolated, names, args = _parse_wrapper_argv(
            tuple(sys.argv[1:] if argv is None else argv)
        )
        environment = sanitized_child_environment(
            isolation_root=isolated,
            credential_env_names=names,
            source_environment=os.environ,
        )
        os.execve(str(codex), (str(codex), *args), dict(environment))
    except CodexChildExecError as exc:
        print(f"error=CodexChildExecError:{exc}", file=sys.stderr)
        return 126
    except OSError as exc:
        print(f"error=CodexChildExecError:exec failed ({exc.errno})", file=sys.stderr)
        return 126
    return 127  # pragma: no cover - successful execve never returns


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CodexChildExecError",
    "SanitizedCodexExecPlan",
    "build_sanitized_codex_exec_plan",
    "sanitized_child_environment",
]
