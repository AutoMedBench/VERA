"""Isolated Codex app-server launch for teacher rollouts.

The published SDK overlays its child environment on the invoking process.
Teacher workers must instead start from an empty Codex home so interactive
user MCP servers and skills cannot inflate or alter the frozen teacher input.
The existing child-exec boundary replaces the environment before ``execve``;
this module only assembles its exact, provider-free launch plan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import stat
import sys

from eva_agent.codex_runtime import CodexLaunchOptions
from eva_agent.deployment.codex_child_exec import build_sanitized_codex_exec_plan

from .teacher_batch import TeacherBatchError


def _private_isolation_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_absolute():
        raise TeacherBatchError("teacher Codex isolation root must be absolute")
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = root.lstat()
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TeacherBatchError("teacher Codex isolation root is unavailable") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or resolved != root
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
    ):
        raise TeacherBatchError("teacher Codex isolation root topology differs")
    return root


def isolated_teacher_launch_options(
    *,
    codex_bin: str | Path,
    cwd: str | Path,
    isolation_root: str | Path,
    config_overrides: Sequence[str],
    child_env: Mapping[str, str],
) -> CodexLaunchOptions:
    """Return an SDK launch that cannot inherit user config, MCPs, or skills."""

    workspace = Path(cwd).resolve(strict=True)
    isolated = _private_isolation_root(isolation_root)
    overrides = tuple(config_overrides)
    if not overrides or any(not isinstance(value, str) or not value for value in overrides):
        raise TeacherBatchError("teacher Codex config override inventory differs")
    environment = dict(child_env)
    if not environment or any(
        not isinstance(name, str)
        or not isinstance(value, str)
        or not value
        for name, value in environment.items()
    ):
        raise TeacherBatchError("teacher Codex child credential binding differs")
    codex_args: list[str] = []
    for value in overrides:
        codex_args.extend(("--config", value))
    codex_args.extend(("app-server", "--strict-config", "--listen", "stdio://"))
    try:
        plan = build_sanitized_codex_exec_plan(
            python_bin=Path(sys.executable).resolve(strict=True),
            wrapper_script=(
                Path(__file__).resolve().parents[1]
                / "deployment"
                / "codex_child_exec.py"
            ),
            codex_bin=Path(codex_bin).resolve(strict=True),
            isolation_root=isolated,
            credential_env_names=tuple(sorted(environment)),
            codex_args=tuple(codex_args),
        )
    except Exception as exc:
        raise TeacherBatchError("teacher Codex isolated launch plan differs") from exc
    return CodexLaunchOptions(
        launch_args_override=plan.launch_args,
        cwd=str(workspace),
        env=environment,
    )


__all__ = ["isolated_teacher_launch_options"]
