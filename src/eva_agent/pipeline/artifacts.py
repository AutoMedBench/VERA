"""No-replace publication for generated evidence (kept outside source control)."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any

from .contracts import ArtifactRef, ContractError, uuid_text
from .digests import blake3_bytes, canonical_json_bytes


class ArtifactStoreError(ContractError):
    """An immutable artifact could not be published or reopened."""


class ImmutableArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ArtifactStoreError("artifact store root topology differs")

    def begin_run(self, run_id: str) -> None:
        uuid_text(run_id, label="artifact run_id")
        target = self.root / run_id
        try:
            target.mkdir(mode=0o700)
        except FileExistsError:
            raise ArtifactStoreError("artifact run identity is already consumed") from None

    def _target(self, run_id: str, relative_path: str) -> Path:
        uuid_text(run_id, label="artifact run_id")
        pure = PurePosixPath(relative_path)
        if (
            pure.is_absolute()
            or pure.as_posix() != relative_path
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise ArtifactStoreError("artifact relative path is unsafe")
        run_root = self.root / run_id
        if run_root.is_symlink() or not run_root.is_dir():
            raise ArtifactStoreError("artifact run root is unavailable")
        parent = run_root
        for part in pure.parts[:-1]:
            parent = parent / part
            if parent.exists() or parent.is_symlink():
                if parent.is_symlink() or not parent.is_dir():
                    raise ArtifactStoreError("artifact parent topology differs")
            else:
                parent.mkdir(mode=0o700)
        return run_root.joinpath(*pure.parts)

    def publish(self, run_id: str, relative_path: str, value: Any) -> ArtifactRef:
        payload = canonical_json_bytes(value)
        target = self._target(run_id, relative_path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(target, flags, 0o400)
        except FileExistsError:
            raise ArtifactStoreError("artifact identity is already consumed") from None
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short artifact write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(target, 0o444)
        parent_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return ArtifactRef(
            relative_path=f"{run_id}/{relative_path}",
            byte_count=len(payload),
            mode="0444",
            content_blake3=blake3_bytes(payload),
        )

    def seal_run(self, run_id: str) -> None:
        run_root = self.root / uuid_text(run_id, label="artifact run_id")
        if run_root.is_symlink() or not run_root.is_dir():
            raise ArtifactStoreError("artifact run root is unavailable")
        for directory in sorted(
            (path for path in run_root.rglob("*") if path.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            os.chmod(directory, 0o555)
        os.chmod(run_root, 0o555)
        root_fd = os.open(run_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)

    def verify(self, reference: ArtifactRef) -> bool:
        pure = PurePosixPath(reference.relative_path)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise ArtifactStoreError("artifact reference path is unsafe")
        path = self.root.joinpath(*pure.parts)
        try:
            info = path.lstat()
        except FileNotFoundError:
            raise ArtifactStoreError("artifact reference is absent") from None
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ArtifactStoreError("artifact reference topology differs")
        payload = path.read_bytes()
        return (
            len(payload) == reference.byte_count
            and f"{stat.S_IMODE(info.st_mode):04o}" == reference.mode
            and blake3_bytes(payload) == reference.content_blake3
        )


__all__ = ["ArtifactStoreError", "ImmutableArtifactStore"]
