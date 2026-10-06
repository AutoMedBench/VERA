"""Versioned evaluation-only MCP tools; no canonical EvaMed name is replaced."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
from threading import Lock, RLock
from uuid import uuid4

import jsonschema

from .adapter import EvaluationError, blake3, canonical, file_digest, read_document, safe_file
from .docker_runtime import execute_python, workspace_inventory
from .skill_surface import SKILL_TOOLS, VerifiedEvaluationSkills

CATALOG_VERSION = "eva.automedbench-public-coding-tools.v1"
TOOLS = (
    {"name": "automed_read_file", "description": "Read a bounded public task, notes, or output text file from this case workspace.",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string", "maxLength": 160},
        "offset": {"type": "integer", "minimum": 0, "maximum": 1048576},
        "limit": {"type": "integer", "minimum": 1, "maximum": 32768}}, "required": ["path"], "additionalProperties": False}},
    {"name": "automed_write_note", "description": "Persist a public progress/decision note in notes/ for subsequent turns and resumed work.",
     "inputSchema": {"type": "object", "properties": {"name": {"type": "string", "pattern": "^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$"},
        "content": {"type": "string", "maxLength": 16384}}, "required": ["name", "content"], "additionalProperties": False}},
    {"name": "automed_view_input", "description": "Inspect the actual public image, or one original-index CT slice with explicit geometry. Returns image pixels, not a diagnosis.",
     "inputSchema": {"type": "object", "properties": {"axis": {"type": "integer", "minimum": 0, "maximum": 2},
        "slice_index": {"type": "integer", "minimum": -1, "maximum": 8192}}, "additionalProperties": False}},
    {"name": "automed_execute_python", "description": "Run real bounded Python in a CPU-only Docker sandbox. Available: NumPy, SciPy, Pillow, Nibabel, Pandas, scikit-image. /workspace/inputs and task.json are read-only; notes/ and outputs/ persist. No network, GPU, host shell, package installation or external models. Returns actual stdout, stderr and exit status. Use code to implement and validate native prediction files.",
     "inputSchema": {"type": "object", "properties": {"code": {"type": "string", "maxLength": 65536},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 180}}, "required": ["code"], "additionalProperties": False}},
) + SKILL_TOOLS


class PublicTools:
    def __init__(self, *, workspace: Path, audit_root: Path, image: str, skills=None):
        self.workspace = workspace.resolve(strict=True)
        self.audit_root = audit_root.resolve()
        if self.audit_root.is_relative_to(self.workspace):
            raise EvaluationError("audit_directory_must_not_be_actor_visible")
        self.audit_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.contract = read_document(safe_file(self.workspace, "task.json"))
        self.image = image
        self.skills = skills
        self.lock = RLock()
        self.audit_lock = Lock()
        self.call_count = 0
        self.code_count = 0
        events = self.audit_root / "mcp-events.jsonl"
        if events.exists():
            for line in events.read_bytes().splitlines():
                row = json.loads(line)
                core = {key: value for key, value in row.items() if key != "event_blake3"}
                if row.get("event_blake3") != blake3(canonical(core)).hexdigest():
                    raise EvaluationError("retained_tool_event_digest_invalid")
                self.call_count += 1
                self.code_count += row["name"] == "automed_execute_python"

    def _read(self, arguments: dict) -> dict:
        relative = arguments["path"]
        if relative != "task.json" and not relative.startswith(("notes/", "outputs/")):
            raise EvaluationError("public_text_path_not_allowed")
        path = safe_file(self.workspace, relative, maximum=1048576)
        data = path.read_bytes()
        offset, limit = arguments.get("offset", 0), arguments.get("limit", 16384)
        return {"path": relative, "offset": offset, "total_bytes": len(data), "file_blake3": file_digest(path),
                "content": data[offset:offset + limit].decode("utf-8", errors="replace"),
                "complete": offset == 0 and limit >= len(data)}

    def _note(self, arguments: dict) -> dict:
        relative = "notes/" + arguments["name"]
        directory = self.workspace / "notes"
        directory.mkdir(exist_ok=True, mode=0o700)
        if directory.is_symlink():
            raise EvaluationError("note_directory_invalid")
        path = directory / arguments["name"]
        before = file_digest(safe_file(self.workspace, relative)) if path.exists() else None
        data = arguments["content"].encode()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        return {"path": relative, "before_blake3": before, "file_blake3": file_digest(path), "bytes": len(data)}

    def _view(self, arguments: dict) -> tuple[dict, dict]:
        from PIL import Image
        source = safe_file(self.workspace, self.contract["input_path"])
        if file_digest(source) != self.contract["input_blake3"]:
            raise EvaluationError("public_input_commitment_changed")
        facts = {"input_path": self.contract["input_path"], "input_blake3": self.contract["input_blake3"]}
        if self.contract["track"] != "segmentation":
            with Image.open(source) as original:
                facts.update({"original_size": list(original.size), "mode": original.mode})
                rendered = original.convert("RGB")
        else:
            import nibabel as nib
            import numpy as np
            volume = nib.load(source)
            axis = arguments.get("axis", 2)
            index = arguments.get("slice_index", -1)
            if index == -1:
                index = volume.shape[axis] // 2
            if not 0 <= index < volume.shape[axis]:
                raise EvaluationError("volume_slice_out_of_bounds")
            selector = [slice(None)] * 3
            selector[axis] = index
            pixels = np.asarray(volume.dataobj[tuple(selector)], dtype=np.float32)
            # Deterministic soft-tissue window; no learned interpretation.
            display = np.clip((np.nan_to_num(pixels) + 160.0) / 400.0, 0.0, 1.0)
            rendered = Image.fromarray((display * 255).astype(np.uint8)).convert("RGB")
            facts.update({"shape": list(volume.shape), "affine": volume.affine.tolist(), "axis": axis,
                          "slice_index": index, "display_transform": "original array axes, no rotation; HU window [-160,240]"})
        rendered.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        rendered.save(buffer, format="PNG")
        data = buffer.getvalue()
        facts.update({"rendered_size": list(rendered.size), "image_blake3": blake3(data).hexdigest(), "image_bytes": len(data)})
        return facts, {"type": "image", "mimeType": "image/png", "data": base64.b64encode(data).decode()}

    def call(self, name: str, arguments: dict, request_id=None) -> dict:
        with self.lock:
            return self._call_locked(name, arguments, request_id)

    def _capture_inventory(self):
        inventory = workspace_inventory(self.workspace)
        blobs = self.audit_root / "workspace-blobs"
        blobs.mkdir(exist_ok=True, mode=0o700)
        for row in inventory["files"]:
            target = blobs / row["blake3"]
            if not target.exists():
                shutil.copyfile(self.workspace / row["path"], target)
                target.chmod(0o400)
            if file_digest(target) != row["blake3"]:
                raise EvaluationError("retained_workspace_blob_differs")
        return inventory

    def _call_locked(self, name: str, arguments: dict, request_id=None) -> dict:
        event_id = str(uuid4())
        before = self._capture_inventory()
        definition = next((item for item in TOOLS if item["name"] == name), None)
        image = None
        error = None
        try:
            if definition is None:
                raise EvaluationError("unknown_public_tool")
            jsonschema.Draft202012Validator(definition["inputSchema"]).validate(arguments)
            with self.lock:
                self.call_count += 1
                if self.call_count > 64:
                    raise EvaluationError("public_tool_call_budget_exhausted")
                if name == "automed_read_file":
                    result = self._read(arguments)
                elif name == "automed_write_note":
                    result = self._note(arguments)
                elif name == "automed_view_input":
                    result, image = self._view(arguments)
                elif name in {"search_skills", "load_skill"}:
                    if self.skills is None:
                        raise EvaluationError("verified_skill_catalog_unavailable")
                    policy = read_document(self.audit_root / "phase-policy.json")
                    stage = policy.get("skill_stage")
                    if stage not in {"S1", "S3", "S4"}:
                        raise EvaluationError("evaluation_skill_phase_not_allowed")
                    result = self.skills.call(name, arguments, stage=stage)
                else:
                    self.code_count += 1
                    if self.code_count > 16:
                        raise EvaluationError("python_execution_budget_exhausted")
                    execution = execute_python(workspace=self.workspace, code=arguments["code"], image=self.image,
                        audit_root=self.audit_root / "code-executions", timeout=arguments.get("timeout_seconds", 120))
                    result = {key: execution[key] for key in ("execution_id", "exit_code", "stdout", "stderr", "timed_out",
                                                              "stream_limit_exceeded", "process_started", "oom_killed")}
                    result.update({"isolation_verified": execution["isolation"]["verified_before_start"],
                                   "workspace_after": execution["after"]})
        except Exception as exc:
            error = str(exc) if isinstance(exc, EvaluationError) else "public_tool_failed"
            result = {"error_code": error}
        after = self._capture_inventory()
        structured = {"event_id": event_id, "name": name, "result": result}
        content = [{"type": "text", "text": canonical(structured).decode()}]
        if image is not None:
            content.append(image)
        response = {"content": content, "structuredContent": structured, "isError": error is not None}
        row = {"schema": "eva.automedbench-public-tool-event.v1", "event_id": event_id,
               "request_id": request_id, "name": name, "arguments": arguments, "result": result,
               "workspace_before": before, "workspace_after": after,
               "is_error": error is not None, "response_blake3": blake3(canonical(response)).hexdigest()}
        row["event_blake3"] = blake3(canonical(row)).hexdigest()
        with self.audit_lock:
            descriptor = os.open(self.audit_root / "mcp-events.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(canonical(row) + b"\n")
        return response


def serve(tools: PublicTools):
    """Small stdio MCP server; no optional user config, network listener or shell."""
    output_lock = Lock()
    def send(value):
        with output_lock:
            sys.stdout.buffer.write(canonical(value) + b"\n")
            sys.stdout.buffer.flush()
    def dispatch(request):
        identifier = request.get("id")
        method = request.get("method")
        if identifier is None:
            return
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "automed_eval", "version": "1.0.0"}}
        elif method == "tools/list":
            result = {"tools": list(TOOLS)}
        elif method == "tools/call":
            params = request.get("params", {})
            result = tools.call(params.get("name"), params.get("arguments", {}), identifier)
        elif method == "ping":
            result = {}
        elif method in {"resources/list", "resources/templates/list", "prompts/list"}:
            result = {{"resources/list": "resources", "resources/templates/list": "resourceTemplates", "prompts/list": "prompts"}[method]: []}
        else:
            send({"jsonrpc": "2.0", "id": identifier, "error": {"code": -32601, "message": "method not exposed"}})
            return
        send({"jsonrpc": "2.0", "id": identifier, "result": result})
    with ThreadPoolExecutor(max_workers=4) as executor:
        for line in sys.stdin.buffer:
            if len(line) > 1024 * 1024:
                raise EvaluationError("mcp_request_too_large")
            executor.submit(dispatch, json.loads(line))
