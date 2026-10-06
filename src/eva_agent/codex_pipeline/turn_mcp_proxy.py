"""Tiny stdio-to-Unix-socket proxy for one authenticated Codex turn.

The executable intentionally contains no EvaMed handlers.  A parent process
owns the candidate-bound runtime and sends this child only a private socket
path plus an unrecorded nonce.  Every stdin request gets a separate socket
connection, which lets Codex issue independent ``tools/call`` requests in
parallel while keeping stdout writes atomic.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import sys
from threading import Lock
from typing import Any


_SOCKET_ENV = "EVA_TURN_MCP_SOCKET"
_NONCE_ENV = "EVA_TURN_MCP_NONCE"
_MAXIMUM_ENV = "EVA_TURN_MCP_MAXIMUM"
_MAXIMUM_PARALLEL_CALLS = 256


def _round_trip(line: str, *, socket_path: str, nonce: str) -> dict[str, Any] | None:
    try:
        message = json.loads(line)
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            raise ValueError("request must be a JSON-RPC object")
        envelope = json.dumps(
            {"nonce": nonce, "request": message},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(socket_path)
            client.sendall(envelope)
            chunks = bytearray()
            while not chunks.endswith(b"\n"):
                chunk = client.recv(65536)
                if not chunk:
                    raise RuntimeError("turn broker closed without a response")
                chunks.extend(chunk)
                if len(chunks) > 16 * 1024 * 1024:
                    raise RuntimeError("turn broker response exceeded the bound")
        wrapped = json.loads(bytes(chunks))
        if not isinstance(wrapped, dict) or set(wrapped) != {"response"}:
            raise RuntimeError("turn broker response envelope differs")
        response = wrapped["response"]
        if response is not None and not isinstance(response, dict):
            raise RuntimeError("turn broker JSON-RPC response differs")
        return response
    except Exception as exc:
        request_id = None
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                request_id = parsed.get("id")
        except Exception:
            pass
        if request_id is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32603, "message": type(exc).__name__},
        }


def main() -> int:
    socket_path = os.environ.get(_SOCKET_ENV, "")
    nonce = os.environ.get(_NONCE_ENV, "")
    try:
        maximum = int(os.environ.get(_MAXIMUM_ENV, "64"))
    except ValueError:
        maximum = 0
    if (
        not socket_path
        or not Path(socket_path).is_absolute()
        or not nonce
        or not 1 <= maximum <= _MAXIMUM_PARALLEL_CALLS
    ):
        raise SystemExit("authenticated turn MCP environment is incomplete")

    write_lock = Lock()

    def emit(response: dict[str, Any] | None) -> None:
        if response is None:
            return
        payload = json.dumps(
            response,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with write_lock:
            sys.stdout.write(payload + "\n")
            sys.stdout.flush()

    with ThreadPoolExecutor(max_workers=maximum, thread_name_prefix="eva-turn-mcp") as pool:
        futures = []
        for line in sys.stdin:
            if line.strip():
                future = pool.submit(
                    _round_trip,
                    line,
                    socket_path=socket_path,
                    nonce=nonce,
                )
                future.add_done_callback(lambda item: emit(item.result()))
                futures.append(future)
        for future in futures:
            future.result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
