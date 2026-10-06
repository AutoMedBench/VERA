"""Explicit, bounded filesystem sandboxes and immutable snapshots."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import stat
from threading import RLock
from typing import Mapping

from .contracts import ContractError, FileSnapshot, WorkspaceSnapshot, uuid_text
from .digests import blake3_bytes, blake3_hex


class SandboxError(ContractError):
    """A workspace path, topology, or resource bound is invalid."""


class FilesystemSandbox:
    """A fresh local root exposed to tools only through safe relative paths."""

    def __init__(
        self,
        base_directory: Path,
        sandbox_id: str,
        initial_files: Mapping[str, bytes],
        *,
        maximum_file_bytes: int = 16 * 1024 * 1024,
        maximum_total_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        uuid_text(sandbox_id, label="filesystem sandbox_id")
        if maximum_file_bytes < 1 or maximum_total_bytes < maximum_file_bytes:
            raise SandboxError("workspace byte bounds differ")
        self.sandbox_id = sandbox_id
        self._base = Path(base_directory)
        self._root = self._base / sandbox_id
        self._maximum_file_bytes = maximum_file_bytes
        self._maximum_total_bytes = maximum_total_bytes
        self._lock = RLock()
        self._create_root()
        for relative_path, payload in sorted(initial_files.items()):
            self.write_bytes(relative_path, payload, create_only=True)

    @property
    def root(self) -> Path:
        """Host-only root; adapters should pass the sandbox object to tools."""

        return self._root

    def _create_root(self) -> None:
        self._base.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self._base.is_symlink() or not self._base.is_dir():
            raise SandboxError("workspace base topology differs")
        try:
            self._root.mkdir(mode=0o700)
        except FileExistsError:
            raise SandboxError("filesystem sandbox identity is already consumed") from None

    def _relative(self, value: str) -> PurePosixPath:
        if not isinstance(value, str):
            raise SandboxError("workspace path must be a string")
        pure = PurePosixPath(value)
        if (
            pure.is_absolute()
            or pure.as_posix() != value
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise SandboxError("workspace path is unsafe")
        return pure

    def _path(self, value: str) -> Path:
        pure = self._relative(value)
        current = self._root
        for part in pure.parts[:-1]:
            current = current / part
            if current.exists() or current.is_symlink():
                if current.is_symlink() or not current.is_dir():
                    raise SandboxError("workspace parent topology differs")
            else:
                current.mkdir(mode=0o700)
        return self._root.joinpath(*pure.parts)

    def read_bytes(self, relative_path: str) -> bytes:
        with self._lock:
            path = self._path(relative_path)
            try:
                info = path.lstat()
            except FileNotFoundError:
                raise SandboxError(f"workspace file is absent: {relative_path}") from None
            if (
                path.is_symlink()
                or not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_size > self._maximum_file_bytes
            ):
                raise SandboxError("workspace file topology or size differs")
            payload = path.read_bytes()
            after = path.lstat()
            if (
                (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or len(payload) != info.st_size
            ):
                raise SandboxError("workspace file changed while reading")
            return payload

    def write_bytes(self, relative_path: str, payload: bytes, *, create_only: bool = False) -> None:
        if not isinstance(payload, bytes) or len(payload) > self._maximum_file_bytes:
            raise SandboxError("workspace write payload differs")
        with self._lock:
            path = self._path(relative_path)
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
            if create_only:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(path, flags, 0o600)
            except FileExistsError:
                raise SandboxError("workspace initial file already exists") from None
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short workspace write")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.chmod(path, 0o600)
            if self._current_total() > self._maximum_total_bytes:
                raise SandboxError("workspace total byte bound exceeded")

    def _current_total(self) -> int:
        total = 0
        for path in self._root.rglob("*"):
            info = path.lstat()
            if path.is_symlink():
                raise SandboxError("workspace contains a symlink")
            if stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    raise SandboxError("workspace contains a hard-linked file")
                total += info.st_size
            elif not stat.S_ISDIR(info.st_mode):
                raise SandboxError("workspace contains an unsupported inode")
        return total

    def snapshot(self, label: str) -> WorkspaceSnapshot:
        """Capture every file byte explicitly; root names and mtimes are excluded."""

        if not label:
            raise SandboxError("workspace snapshot label is required")
        with self._lock:
            files: list[FileSnapshot] = []
            for path in sorted(self._root.rglob("*"), key=lambda item: item.relative_to(self._root).as_posix()):
                info = path.lstat()
                if path.is_symlink():
                    raise SandboxError("workspace snapshot encountered a symlink")
                if stat.S_ISDIR(info.st_mode):
                    continue
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise SandboxError("workspace snapshot topology differs")
                relative = path.relative_to(self._root).as_posix()
                payload = self.read_bytes(relative)
                files.append(
                    FileSnapshot(
                        path=relative,
                        content=payload,
                        byte_count=len(payload),
                        mode=f"{stat.S_IMODE(path.stat().st_mode):04o}",
                        content_blake3=blake3_bytes(payload),
                    )
                )
            total = sum(item.byte_count for item in files)
            if total > self._maximum_total_bytes:
                raise SandboxError("workspace snapshot exceeds total byte bound")
            # ``label`` is provenance metadata, not workspace content.  Keeping
            # it outside the tree commitment lets a before/after comparison
            # detect real mutations and lets independently reset cohort
            # workspaces prove byte identity.
            core = {
                "files": files,
                "file_count": len(files),
                "byte_count": total,
            }
            return WorkspaceSnapshot(
                label=label,
                files=tuple(files),
                file_count=len(files),
                byte_count=total,
                tree_blake3=blake3_hex(core),
            )


__all__ = ["FilesystemSandbox", "SandboxError"]
