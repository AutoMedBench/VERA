"""Bounded snapshots of asynchronously published public analysis artifacts."""
from __future__ import annotations

import time
import os
import stat
from blake3 import blake3
from training.automedbench_lite.adapter import EvaluationError
from training.automedbench_lite.track_tools import MutableInventory


class PublishingInventory(MutableInventory):
    """Retry transient publication races; all successful bytes keep original checks."""
    def capture(self):
        deadline = time.monotonic() + 1.0
        attempts = 0
        while True:
            attempts += 1
            try:
                return self._capture_descriptors()
            except (EvaluationError, FileNotFoundError) as exc:
                transient = isinstance(exc, FileNotFoundError) or str(exc) == "track_file_changed_during_snapshot"
                if not transient or attempts >= 16 or time.monotonic() >= deadline:
                    raise
                # Never change or ignore a file, substitute a digest, or return a
                # partial inventory. Restart the checked observation after the
                # publisher gets an opportunity to complete its atomic update.
                time.sleep(min(.02 * attempts, max(0, deadline - time.monotonic())))

    def _capture_descriptors(self):
        rows, total_bytes = [], 0
        started = time.time_ns()
        for directory, dirs, files in os.walk(self.workspace, followlinks=False):
            from pathlib import Path
            base = Path(directory)
            if base == self.workspace:
                dirs[:] = [item for item in dirs if item != "inputs"]
            if any((base / name).is_symlink() for name in dirs):
                raise EvaluationError("track_workspace_symlink")
            for name in files:
                path = base / name
                # Read a single opened inode, rather than comparing its bytes
                # with a pathname that an atomic publisher may have replaced.
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, "rb") as stream:
                    info = os.fstat(stream.fileno())
                    if info.st_nlink == 0:
                        # An atomic replacement may unlink the opened old inode
                        # before fstat. Restart against the current pathname.
                        raise EvaluationError("track_file_changed_during_snapshot")
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 512 * 1024**2:
                        raise EvaluationError("track_mutable_file_invalid")
                    key = (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                    relative = path.relative_to(self.workspace).as_posix()
                    cached = self.cache.get(relative)
                    if cached is None or cached[0] != key:
                        data = stream.read(512 * 1024**2 + 1)
                        final = os.fstat(stream.fileno())
                        if len(data) > 512 * 1024**2:
                            raise EvaluationError("track_mutable_file_invalid")
                        if key != (final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns):
                            raise EvaluationError("track_file_changed_during_snapshot")
                        digest = blake3(data).hexdigest()
                        blob = self.blobs / digest
                        if not blob.exists():
                            with blob.open("xb") as target:
                                target.write(data)
                            blob.chmod(0o400)
                        cached = (key, {"path": relative, "bytes": len(data), "blake3": digest,
                                        "mode": stat.S_IMODE(info.st_mode)})
                        self.cache[relative] = cached
                rows.append(cached[1])
                total_bytes += cached[1]["bytes"]
                if len(rows) > 16000 or total_bytes > 8 * 1024**3:
                    raise EvaluationError("track_mutable_workspace_limit")
        rows.sort(key=lambda row: row["path"])
        return {"files": rows, "file_count": len(rows), "bytes": total_bytes,
            "immutable_inputs_manifest_blake3": self.binding_digest,
            "immutable_input_files": len(self.binding["files"]), "inputs_inlined": False,
            "observation_started_ns": started, "observation_completed_ns": time.time_ns(),
            "background_model_jobs_may_progress": True}
