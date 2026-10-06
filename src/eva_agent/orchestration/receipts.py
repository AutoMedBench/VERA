"""Crash-tolerant, immutable BLAKE3 receipts for orchestration decisions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from threading import Lock
from typing import Any, Mapping
from uuid import uuid4

from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes

from .contracts import OrchestrationContractError, canonical_uuid


class ReceiptJournalError(OrchestrationContractError):
    """An immutable orchestration receipt could not be written or reopened."""


def _safe_component(value: str, *, label: str) -> str:
    canonical_uuid(value, label=label)
    if "/" in value or value in {".", ".."}:
        raise ReceiptJournalError(f"{label} is not a safe path component")
    return value


class ReceiptJournal:
    """One no-replace receipt tree per scheduler invocation.

    A payload is fsynced to a private pending inode and hard-linked into its
    final unique name.  A sudden process loss can therefore leave a harmless
    ``.pending`` file, but never a partially published receipt.
    """

    def __init__(self, root: Path, run_id: str) -> None:
        requested_root = Path(root)
        if requested_root.is_symlink():
            raise ReceiptJournalError("receipt root cannot be a symlink")
        self.root = requested_root.resolve()
        self.run_id = _safe_component(run_id, label="run_id")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ReceiptJournalError("receipt root topology differs")
        self.run_root = self.root / self.run_id
        try:
            self.run_root.mkdir(mode=0o700)
        except FileExistsError:
            raise ReceiptJournalError("orchestration run identity is already consumed") from None
        (self.run_root / "workers").mkdir(mode=0o700)
        self._lock = Lock()
        self._published: list[str] = []
        self._finalized = False

    @staticmethod
    def _document(
        *,
        run_id: str,
        worker_id: str | None,
        event: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not event or not isinstance(payload, Mapping):
            raise ReceiptJournalError("receipt event and payload are required")
        receipt_id = str(uuid4())
        core: dict[str, Any] = {
            "schema": "eva.orchestration-receipt.v1",
            "receipt_id": receipt_id,
            "run_id": run_id,
            "worker_id": worker_id,
            "event": event,
            "payload": dict(payload),
        }
        return {**core, "receipt_blake3": blake3_hex(core)}

    @staticmethod
    def _publish(path: Path, document: Mapping[str, Any]) -> None:
        payload = canonical_json_bytes(document)
        pending = path.parent / f".{path.name}.{uuid4()}.pending"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(pending, flags, 0o400)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short receipt write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(pending, path, follow_symlinks=False)
            os.chmod(path, 0o444)
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            pending.unlink(missing_ok=True)

    def append(
        self,
        *,
        event: str,
        payload: Mapping[str, Any],
        worker_id: str | None = None,
        fixed_name: str | None = None,
    ) -> Mapping[str, Any]:
        with self._lock:
            if self._finalized:
                raise ReceiptJournalError("receipt journal is already finalized")
            if worker_id is not None:
                worker_id = _safe_component(worker_id, label="worker_id")
            document = self._document(
                run_id=self.run_id,
                worker_id=worker_id,
                event=event,
                payload=payload,
            )
            if fixed_name is not None:
                if fixed_name not in {"run-start.json", "run-final.json"}:
                    raise ReceiptJournalError("fixed receipt name differs")
                target = self.run_root / fixed_name
            elif worker_id is None:
                target = self.run_root / f"{document['receipt_id']}.json"
            else:
                worker_root = self.run_root / "workers" / worker_id
                worker_root.mkdir(mode=0o700, exist_ok=True)
                if worker_root.is_symlink() or not worker_root.is_dir():
                    raise ReceiptJournalError("worker receipt topology differs")
                target = worker_root / f"{document['receipt_id']}.json"
            self._publish(target, document)
            self._published.append(str(document["receipt_blake3"]))
            return document

    def start(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.append(event="run_started", payload=payload, fixed_name="run-start.json")

    def finalize(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        with self._lock:
            prior = tuple(sorted(self._published))
        final_payload = {
            **dict(payload),
            "prior_receipt_count": len(prior),
            "prior_receipts_blake3": blake3_hex(prior),
        }
        document = self.append(
            event="run_finished", payload=final_payload, fixed_name="run-final.json"
        )
        with self._lock:
            self._finalized = True
        for directory in sorted(
            (path for path in self.run_root.rglob("*") if path.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        self.run_root.chmod(0o555)
        return document

    @staticmethod
    def verify(path: Path) -> bool:
        path = Path(path)
        info = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o444
        ):
            return False
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or "receipt_blake3" not in document:
            return False
        recorded = document.pop("receipt_blake3")
        return recorded == blake3_hex(document)

    @classmethod
    def verify_run(cls, root: Path, run_id: str) -> bool:
        """Independently reopen a sealed run and its aggregate commitment."""

        try:
            requested_root = Path(root)
            if requested_root.is_symlink():
                return False
            run_root = requested_root.resolve() / _safe_component(run_id, label="run_id")
            if run_root.is_symlink() or not run_root.is_dir():
                return False
            if stat.S_IMODE(run_root.stat().st_mode) != 0o555:
                return False
            if any(path.is_symlink() for path in run_root.rglob("*")):
                return False
            if any(path.name.endswith(".pending") for path in run_root.rglob("*")):
                return False
            paths = sorted(run_root.rglob("*.json"))
            if not paths or not all(cls.verify(path) for path in paths):
                return False
            documents = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            if any(
                document.get("schema") != "eva.orchestration-receipt.v1"
                or document.get("run_id") != run_id
                for document in documents
            ):
                return False
            receipt_ids = [document.get("receipt_id") for document in documents]
            if len(set(receipt_ids)) != len(receipt_ids):
                return False
            starts = [document for document in documents if document.get("event") == "run_started"]
            finals = [document for document in documents if document.get("event") == "run_finished"]
            if len(starts) != 1 or len(finals) != 1:
                return False
            final = finals[0]
            prior = sorted(
                str(document["receipt_blake3"])
                for document in documents
                if document is not final
            )
            payload = final.get("payload")
            return (
                isinstance(payload, dict)
                and payload.get("prior_receipt_count") == len(prior)
                and payload.get("prior_receipts_blake3") == blake3_hex(tuple(prior))
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return False


__all__ = ["ReceiptJournal", "ReceiptJournalError"]
