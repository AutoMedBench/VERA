"""Keep in-progress analysis bytes private and publish complete case directories."""
from __future__ import annotations

import os
import ctypes
from pathlib import Path
import re
import stat
import json
import math
import time
from uuid import uuid4

from blake3 import blake3


class PolicyDeadlineExceeded(RuntimeError):
    """The trusted track clock denied further work; this is not a model error."""


class PublicationBoundaryViolation(RuntimeError):
    """A publication syscall crossed the cutoff; retain evidence and reject scoring."""


class DeadlineGuard:
    def __init__(self, audit: Path, deadline: float | None, *, component='worker'):
        if deadline is not None and (not math.isfinite(deadline) or deadline < 0):
            raise ValueError('invalid_trusted_track_deadline')
        self.audit, self.deadline = Path(audit), deadline
        if component not in {'worker', 'supervisor'}:
            raise ValueError('unrecognized_deadline_evidence_component')
        self.events = self.audit / (('supervisor-' if component == 'supervisor' else '') +
                                   'deadline-publication-events.jsonl')

    def record(self, event, **fields):
        self.audit.mkdir(parents=True, exist_ok=True, mode=0o700)
        row = {'schema': 'eva.prescribed-model-deadline-event.v1', 'event_id': str(uuid4()),
               'event': event, 'observed_monotonic': time.monotonic(),
               'track_deadline_monotonic': self.deadline, 'host_only': True, **fields}
        with self.events.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
        return row

    def check(self, operation, *, record_success=False):
        observed = time.monotonic()
        if self.deadline is not None and observed >= self.deadline:
            self.record('policy_cancellation', operation=operation,
                        policy_cancellation=True, cancellation_reason='track_wall_clock_budget_exhausted')
            raise PolicyDeadlineExceeded('track_wall_clock_budget_exhausted')
        if record_success:
            self.record('work_admission', operation=operation, admitted_monotonic=observed)
        return observed

    def publish(self, operation, destination, action):
        self.check(operation)
        publication_id = str(uuid4())
        self.record('public_publication_intent', publication_id=publication_id,
                    operation=operation, destination=str(destination))
        try:
            # The write-ahead log itself can take time. Recheck immediately
            # before the action, and terminally close a denied intent.
            started = self.check(operation)
        except PolicyDeadlineExceeded:
            self.record('public_publication_cancelled', publication_id=publication_id,
                        operation=operation, destination=str(destination), action_started=False,
                        policy_cancellation=True)
            raise
        action()
        completed = time.monotonic()
        within = self.deadline is None or completed < self.deadline
        self.record('public_publication', publication_id=publication_id,
                    operation=operation, destination=str(destination),
                    publication_started_monotonic=started, publication_completed_monotonic=completed,
                    complete_interval_before_deadline=within)
        if not within:
            raise PublicationBoundaryViolation('publication_interval_crossed_track_deadline')

    def evidence(self):
        return {'path': str(self.events), 'blake3': blake3(self.events.read_bytes()).hexdigest()
                if self.events.is_file() else None, 'track_deadline_monotonic': self.deadline}


def atomic_new(source: Path, destination: Path):
    """Linux atomic rename without overwrite or a transient hard-link count."""
    library = ctypes.CDLL(None, use_errno=True)
    rename = library.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(destination))


class CasePublisher:
    def __init__(self, workspace: Path, output: Path, audit: Path, *, track_deadline_monotonic=None):
        self.workspace = workspace.resolve(strict=True)
        self.output = output.resolve(strict=True)
        self.audit = audit.resolve(strict=True)
        self.deadline = DeadlineGuard(self.audit, track_deadline_monotonic)
        if self.audit.is_relative_to(self.workspace) or not self.output.is_relative_to(self.workspace / "outputs"):
            raise ValueError("analysis_publication_boundary_invalid")
        self.private = self.audit / "private-publication"
        self.cases, self.auxiliary, self.metadata = (self.private / name for name in ("cases", "auxiliary", "metadata"))
        for path in (self.cases, self.auxiliary, self.metadata):
            path.mkdir(parents=True, mode=0o700)
        if self.private.stat().st_dev != self.output.stat().st_dev:
            raise ValueError("atomic_publication_requires_same_filesystem")

    def case_path(self, case: str) -> Path:
        self.deadline.check('case_work_admission')
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", case):
            raise ValueError("invalid_publication_case")
        return self.cases / case

    def auxiliary_path(self, name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name):
            raise ValueError("invalid_auxiliary_publication_name")
        return self.auxiliary / name

    def public_path(self, path: Path) -> Path:
        path = path.absolute()
        for root in (self.cases, self.auxiliary):
            if path.is_relative_to(root):
                return self.output / path.relative_to(root)
        if path.is_relative_to(self.output):
            return path
        if path.is_relative_to(self.workspace / "inputs"):
            return path
        raise ValueError("artifact_outside_host_publication_roots")

    def artifact(self, path: Path) -> dict:
        path = path.absolute()
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("publication_artifact_not_regular_file")
        digest = blake3()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
        return {"path": self.public_path(path).relative_to(self.workspace).as_posix(),
                "bytes": info.st_size, "blake3": digest.hexdigest()}

    def publish_case(self, case: str):
        source = self.case_path(case)
        destination = self.output / case
        if source.exists():
            if destination.exists() or destination.is_symlink():
                raise ValueError("analysis_case_destination_already_exists")
            # Every file is complete before the directory becomes actor-visible.
            self.deadline.publish('publish_case', destination, lambda: atomic_new(source, destination))
        for path in tuple(self.auxiliary.iterdir()):
            if not path.is_file() or path.is_symlink():
                raise ValueError("invalid_auxiliary_publication")
            destination = self.public_path(path)
            if destination.exists():
                raise ValueError("auxiliary_artifact_already_published")
            self.deadline.publish('publish_auxiliary', destination, lambda: atomic_new(path, destination))

    def publish_metadata(self, name: str, payload: bytes):
        if name not in {"receipt.json", "case-outputs.jsonl", "progress.jsonl", "raw-output.json"}:
            raise ValueError("unrecognized_publication_metadata")
        temporary = self.metadata / str(uuid4())
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        self.deadline.publish('publish_metadata', self.output / name,
                              lambda: os.replace(temporary, self.output / name))
