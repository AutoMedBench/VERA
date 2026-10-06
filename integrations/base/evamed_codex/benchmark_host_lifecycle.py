"""Host dispatch evidence, distinct from whether Codex observed a tool result."""
from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
import time
from uuid import uuid4

from training.automedbench_lite.adapter import blake3, canonical, file_digest, write_once


def read_dispatch_events(audit: Path) -> list[dict]:
    path = audit / "host-dispatch.jsonl"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024**2:
        raise ValueError("host_dispatch_ledger_topology_or_size_invalid")
    previous = None
    rows = []
    for sequence, line in enumerate(path.read_bytes().splitlines()):
        row = json.loads(line)
        core = {key: value for key, value in row.items() if key != "document_blake3"}
        if (row.get("schema") != "eva.benchmark-host-dispatch-event.v1"
                or row.get("sequence") != sequence or row.get("previous_blake3") != previous
                or row.get("document_blake3") != blake3(canonical(core)).hexdigest()):
            raise ValueError("host_dispatch_hash_chain_invalid")
        if row.get("phase") not in {"started", "completed", "dispatch_failed", "delivery"}:
            raise ValueError("host_dispatch_phase_invalid")
        rows.append(row)
        previous = row["document_blake3"]
    return rows


class HostDispatchLedger:
    def __init__(self, audit: Path, deadline: float | None):
        self.path = audit / "host-dispatch.jsonl"
        self.deadline = deadline
        self.lock = Lock()
        self.server_id = str(uuid4())
        rows = read_dispatch_events(audit) if self.path.exists() else []
        self.sequence = len(rows)
        self.previous = rows[-1]["document_blake3"] if rows else None
        process = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
        write_once(audit / ("mcp-server-" + self.server_id + ".json"), {
            "schema": "eva.benchmark-host-mcp-process.v1", "server_id": self.server_id,
            "pid": os.getpid(), "start_ticks": process[19], "deadline_monotonic": deadline,
            "created_monotonic": time.monotonic(), "host_lifecycle_source_blake3": file_digest(Path(__file__))})

    def append(self, phase: str, dispatch_id: str, **data) -> dict:
        with self.lock:
            observed = time.monotonic()
            core = {"schema": "eva.benchmark-host-dispatch-event.v1", "sequence": self.sequence,
                    "previous_blake3": self.previous, "server_id": self.server_id,
                    "phase": phase, "dispatch_id": dispatch_id, "observed_ns": time.time_ns(),
                    "observed_monotonic": observed, "deadline_monotonic": self.deadline,
                    "deadline_elapsed": self.deadline is not None and observed >= self.deadline, **data}
            row = {**core, "document_blake3": blake3(canonical(core)).hexdigest()}
            with self.path.open("ab") as stream:
                stream.write(canonical(row) + b"\n")
            self.sequence += 1
            self.previous = row["document_blake3"]
            return row

    def start(self, request: dict) -> str:
        dispatch = str(uuid4())
        params = request.get("params", {})
        self.append("started", dispatch, request_id=request.get("id"), tool=params.get("name"),
                    arguments_blake3=blake3(canonical(params.get("arguments", {}))).hexdigest())
        return dispatch

    def complete(self, dispatch: str, response: dict) -> dict:
        structured = response.get("structuredContent", {})
        result = structured.get("result", {})
        return self.append("completed", dispatch, event_id=structured.get("event_id"),
                           response_blake3=blake3(canonical(response)).hexdigest(),
                           tool_is_error=response.get("isError", False),
                           error_code=result.get("error_code") if isinstance(result, dict) else None,
                           model_observed_result=False)

    def delivery(self, dispatch: str, *, written: bool) -> dict:
        return self.append("delivery", dispatch, state="written_to_pipe" if written else "peer_closed",
                           model_observed_result=False)
