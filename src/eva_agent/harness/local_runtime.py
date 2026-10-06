"""Direct local workspace tools for hosts without a separate HTTP tool service.

Commands use argv arrays (never a shell) and run inside a dedicated workspace.
Provider credentials are not inherited by child processes. A cached venv can be
mounted per environment profile so coding-heavy tasks do not pay setup cost for
every sandbox.
"""

from __future__ import annotations

import asyncio
from pathlib import Path, PurePosixPath
import os
import sys
from typing import Any, Mapping, Sequence

from .contracts import HarnessContractError
from .tools import ToolDefinition


class LocalVenvWorkspace:
    def __init__(
        self,
        *,
        workspace_root: Path,
        venv_root: Path | None = None,
        max_output_bytes: int = 2_000_000,
    ) -> None:
        root = workspace_root.resolve()
        if not root.is_dir() or root.is_symlink():
            raise HarnessContractError("workspace root must be a real directory")
        if max_output_bytes < 1:
            raise HarnessContractError("max output bytes must be positive")
        self.root = root
        self.venv_root = venv_root.resolve() if venv_root is not None else None
        if self.venv_root is not None:
            interpreter = self.venv_root / "bin" / "python"
            if not interpreter.is_file() or interpreter.is_symlink():
                raise HarnessContractError("venv Python interpreter is unavailable")
        self.max_output_bytes = max_output_bytes

    def _path(self, raw: str, *, must_exist: bool = False) -> Path:
        relative = PurePosixPath(raw)
        if not raw or relative.is_absolute() or ".." in relative.parts:
            raise HarnessContractError("workspace path must be relative and contained")
        candidate = self.root.joinpath(*relative.parts)
        resolved = candidate.resolve(strict=False)
        if self.root not in (resolved, *resolved.parents):
            raise HarnessContractError("workspace path escapes the sandbox")
        cursor = self.root
        for part in candidate.relative_to(self.root).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise HarnessContractError("workspace symlinks are not accepted")
        if must_exist and not candidate.exists():
            raise HarnessContractError("workspace path does not exist")
        return candidate

    def _environment(self) -> dict[str, str]:
        path_parts: list[str] = []
        if self.venv_root is not None:
            path_parts.append(str(self.venv_root / "bin"))
        path_parts.extend(("/usr/local/bin", "/usr/bin", "/bin"))
        return {
            "PATH": os.pathsep.join(path_parts),
            "LANG": os.getenv("LANG", "C.UTF-8"),
            "LC_ALL": os.getenv("LC_ALL", "C.UTF-8"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "VIRTUAL_ENV": str(self.venv_root) if self.venv_root else "",
        }

    async def _run(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        if not argv or any(not isinstance(part, str) or not part for part in argv):
            raise HarnessContractError("command argv must contain non-empty strings")
        if timeout_seconds <= 0:
            raise HarnessContractError("command timeout must be positive")
        working_directory = self._path(cwd, must_exist=True)
        if not working_directory.is_dir():
            raise HarnessContractError("command cwd must be a directory")
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=working_directory,
            env=self._environment(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_seconds
            )
            timed_out = False
        except asyncio.TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            timed_out = True
        truncated = len(stdout) + len(stderr) > self.max_output_bytes
        remaining = self.max_output_bytes
        stdout = stdout[:remaining]
        remaining -= len(stdout)
        stderr = stderr[:remaining]
        return {
            "argv": list(argv),
            "cwd": cwd,
            "exit_code": process.returncode,
            "timed_out": timed_out,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "output_truncated": truncated,
        }

    def tool_definitions(self) -> tuple[ToolDefinition, ...]:
        async def list_files(arguments: Mapping[str, Any]) -> dict[str, Any]:
            relative = str(arguments.get("path", "."))
            directory = self._path(relative, must_exist=True)
            if not directory.is_dir():
                raise HarnessContractError("list_files path must be a directory")
            limit = int(arguments.get("limit", 500))
            paths: list[str] = []
            for candidate in sorted(directory.rglob("*")):
                if candidate.is_file() and not candidate.is_symlink():
                    paths.append(candidate.relative_to(self.root).as_posix())
                    if len(paths) >= limit:
                        break
            return {"files": paths, "limit": limit, "truncated": len(paths) >= limit}

        async def read_file(arguments: Mapping[str, Any]) -> dict[str, Any]:
            path = self._path(str(arguments["path"]), must_exist=True)
            if not path.is_file():
                raise HarnessContractError("read_file path must be a file")
            maximum = int(arguments.get("max_bytes", 1_000_000))
            payload = path.read_bytes()
            return {
                "path": path.relative_to(self.root).as_posix(),
                "content": payload[:maximum].decode("utf-8", errors="replace"),
                "byte_count": len(payload),
                "truncated": len(payload) > maximum,
            }

        async def write_file(arguments: Mapping[str, Any]) -> dict[str, Any]:
            path = self._path(str(arguments["path"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            content = str(arguments["content"])
            path.write_text(content, encoding="utf-8")
            return {
                "path": path.relative_to(self.root).as_posix(),
                "byte_count": len(content.encode("utf-8")),
            }

        async def search_files(arguments: Mapping[str, Any]) -> dict[str, Any]:
            query = str(arguments["query"])
            path = str(arguments.get("path", "."))
            return await self._run(
                ("rg", "-n", "--fixed-strings", "--", query, path),
                cwd=".",
                timeout_seconds=float(arguments.get("timeout_seconds", 60)),
            )

        async def execute_code(arguments: Mapping[str, Any]) -> dict[str, Any]:
            return await self._run(
                tuple(str(item) for item in arguments["argv"]),
                cwd=str(arguments.get("cwd", ".")),
                timeout_seconds=float(arguments.get("timeout_seconds", 300)),
            )

        object_schema = {"type": "object", "additionalProperties": False}
        return (
            ToolDefinition(
                name="list_files",
                description="List files recursively inside the sandbox workspace.",
                parameters={
                    **object_schema,
                    "properties": {
                        "path": {"type": "string", "default": "."},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 5000},
                    },
                },
                handler=list_files,
                parallel_safe=True,
            ),
            ToolDefinition(
                name="read_file",
                description="Read one UTF-8 projected file from the sandbox workspace.",
                parameters={
                    **object_schema,
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "max_bytes": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 2000000,
                        },
                    },
                    "required": ["path"],
                },
                handler=read_file,
                parallel_safe=True,
            ),
            ToolDefinition(
                name="search_files",
                description="Search literal text with ripgrep inside the workspace.",
                parameters={
                    **object_schema,
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "path": {"type": "string", "default": "."},
                        "timeout_seconds": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": 300,
                        },
                    },
                    "required": ["query"],
                },
                handler=search_files,
                parallel_safe=True,
            ),
            ToolDefinition(
                name="write_file",
                description="Write complete UTF-8 content to a contained workspace file.",
                parameters={
                    **object_schema,
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
                handler=write_file,
                parallel_safe=False,
            ),
            ToolDefinition(
                name="execute_code",
                description=(
                    "Run an argv command without a shell in the local sandbox venv; "
                    "use for coding, tests, and verification."
                ),
                parameters={
                    **object_schema,
                    "properties": {
                        "argv": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 100,
                            "items": {"type": "string", "minLength": 1},
                        },
                        "cwd": {"type": "string", "default": "."},
                        "timeout_seconds": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": 3600,
                        },
                    },
                    "required": ["argv"],
                },
                handler=execute_code,
                parallel_safe=False,
            ),
        )


def bootstrap_venv(venv_root: Path, *, python: Path | None = None) -> None:
    """Create a local venv without installing unpinned network dependencies."""

    target = venv_root.resolve()
    if target.exists():
        interpreter = target / "bin" / "python"
        if not interpreter.is_file():
            raise HarnessContractError("existing venv root is incomplete")
        return
    import subprocess

    subprocess.run(
        [str(python or Path(sys.executable)), "-m", "venv", str(target)],
        check=True,
    )


__all__ = ["LocalVenvWorkspace", "bootstrap_venv"]
