"""Public-only track coding tools and durable fixed-model jobs.

The actor authors Python only inside the CPU Docker sandbox. Model processes
execute a fixed trusted program; neither actor code nor scorer paths are passed.
Large immutable input trees are bound once, not rehashed on every tool call.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from threading import Lock
from uuid import UUID, uuid4

import jsonschema

from .adapter import EvaluationError, blake3, canonical, file_digest, read_document, safe_file, write_once
from .docker_runtime import create_command, verify_container, bounded_start
from .public_tools import PublicTools, TOOLS as CASE_TOOLS
from .skill_surface import SKILL_TOOLS

ROOT = Path(__file__).resolve().parents[2]
CATALOG_VERSION = "eva.automedbench-track-coding-tools.v3"
MODEL_TRACKS = {"classification", "detection", "segmentation", "synthesis", "enhancement", "vqa", "report"}
TOOLS_V1 = (CASE_TOOLS[0], CASE_TOOLS[1],
    {"name": "automed_view_input", "description": "View one declared public case image or original-index volume slice; no diagnosis or private reference.",
     "inputSchema": {"type": "object", "properties": {"case_id": {"type": "string", "maxLength": 100},
        "image_index": {"type": "integer", "minimum": 0, "maximum": 32},
        "slice_index": {"type": "integer", "minimum": -1, "maximum": 8192}}, "required": ["case_id"], "additionalProperties": False}},
    CASE_TOOLS[3],
    {"name": "automed_submit_model_job", "description": "Submit a durable asynchronous job using this track's exact prescribed public pretrained model (currently classification/detection). This is an analysis tool, not the Qwen coding agent. Select 1..32 public case IDs. Returns job ID, not predictions; poll status later. No arbitrary program, path, endpoint or model selection.",
     "inputSchema": {"type": "object", "properties": {"case_ids": {"type": "array", "items": {"type": "string", "maxLength": 100}, "minItems": 1, "maxItems": 32, "uniqueItems": True},
        "confidence": {"type": "number", "minimum": 0.001, "maximum": 0.99}}, "required": ["case_ids"], "additionalProperties": False}},
    {"name": "automed_model_job_status", "description": "Read authoritative state/progress for a previously submitted job in this track; return bounded summary and actor artifact paths, never all raw predictions. A running job is not success. Do useful code/notes work while it runs; do not busy-poll.",
     "inputSchema": {"type": "object", "properties": {"job_id": {"type": "string", "format": "uuid"}}, "required": ["job_id"], "additionalProperties": False}},
) + SKILL_TOOLS
EXTENDED_MODEL_TOOL = {
    "name": "automed_submit_extended_model_job",
    "description": "Submit one asynchronous prescribed-model job for segmentation, synthesis or enhancement using this workspace's bound public cases. No arbitrary model, program or path. Enhancement requires explicit sigma and input-normalization HU bounds from its public guidance; its output HU limits are separately fixed by the task. Retain the job ID and poll automed_model_job_status later. Raw model artifacts are not final submissions or clinical scores.",
    "inputSchema": {"type": "object", "properties": {
        "case_ids": {"type": "array", "items": {"type": "string", "maxLength": 100},
                     "minItems": 1, "maxItems": 32, "uniqueItems": True},
        "sigma": {"type": "number", "minimum": 0, "maximum": 1},
        "hu_min": {"type": "number", "minimum": -4096, "maximum": 4096},
        "hu_max": {"type": "number", "minimum": -4096, "maximum": 8192}},
        "required": ["case_ids"], "additionalProperties": False}}
# Add a new capability without changing any historical tool or skill definition.
TOOLS_V2 = TOOLS_V1[:-len(SKILL_TOOLS)] + (EXTENDED_MODEL_TOOL,) + SKILL_TOOLS
GENERATIVE_MODEL_TOOL = {
    "name": "automed_submit_generative_model_job",
    "description": "Submit an asynchronous fixed public LLaVA-Med VQA or CheXagent report model job, not a replacement coding agent. Select 1..32 bound public cases and an explicit output token limit (VQA at most256, report at most512). VQA requires multi_image_mode=montage, including ALL declared images; report requires an explicit prompt from public guidance. Retain job ID and poll existing status tool later. Raw text, actual helper code and artifacts are retained as provided_analysis_tool evidence; you must still author the reusable pipeline and final submission. No private scoring or reference access.",
    "inputSchema": {"type": "object", "properties": {
        "case_ids": {"type": "array", "items": {"type": "string", "maxLength": 100},
                     "minItems": 1, "maxItems": 32, "uniqueItems": True},
        "max_new_tokens": {"type": "integer", "minimum": 1, "maximum": 512},
        "prompt": {"type": "string", "minLength": 10, "maxLength": 1024},
        "multi_image_mode": {"type": "string", "enum": ["montage"]}},
        "required": ["case_ids", "max_new_tokens"], "additionalProperties": False}}
TOOLS = TOOLS_V2[:-len(SKILL_TOOLS)] + (GENERATIVE_MODEL_TOOL,) + SKILL_TOOLS


class MutableInventory:
    """Cache unchanged files, retain exact observed bytes, bind public inputs apart."""
    def __init__(self, workspace: Path, audit: Path):
        self.workspace, self.audit = workspace, audit
        self.cache = {}
        self.binding = read_document(workspace / "inputs-manifest.json")
        self.binding_digest = file_digest(workspace / "inputs-manifest.json")
        self.blobs = audit / "workspace-blobs"
        self.blobs.mkdir(parents=True, exist_ok=True, mode=0o700)

    def capture(self):
        rows = []
        started = time.time_ns()
        for directory, dirs, files in os.walk(self.workspace, followlinks=False):
            base = Path(directory)
            if base == self.workspace:
                dirs[:] = [item for item in dirs if item != "inputs"]
            for name in dirs:
                if (base / name).is_symlink():
                    raise EvaluationError("track_workspace_symlink")
            for name in files:
                path = base / name
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 512 * 1024 * 1024:
                    raise EvaluationError("track_mutable_file_invalid")
                key = (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                relative = path.relative_to(self.workspace).as_posix()
                cached = self.cache.get(relative)
                if cached is None or cached[0] != key:
                    data = path.read_bytes()
                    final = path.stat()
                    if key != (final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns):
                        raise EvaluationError("track_file_changed_during_snapshot")
                    digest = blake3(data).hexdigest()
                    blob = self.blobs / digest
                    if not blob.exists():
                        with blob.open("xb") as stream:
                            stream.write(data)
                        blob.chmod(0o400)
                    cached = (key, {"path": relative, "bytes": len(data), "blake3": digest,
                                    "mode": stat.S_IMODE(info.st_mode)})
                    self.cache[relative] = cached
                rows.append(cached[1])
                if len(rows) > 16000 or sum(row["bytes"] for row in rows) > 8 * 1024**3:
                    raise EvaluationError("track_mutable_workspace_limit")
        rows.sort(key=lambda row: row["path"])
        return {"files": rows, "file_count": len(rows), "bytes": sum(row["bytes"] for row in rows),
            "immutable_inputs_manifest_blake3": self.binding_digest,
            "immutable_input_files": len(self.binding["files"]), "inputs_inlined": False,
            "observation_started_ns": started, "observation_completed_ns": time.time_ns(),
            "background_model_jobs_may_progress": True}


def execute_track_python(*, workspace, audit_root, image, code, timeout, inventory):
    execution_id = str(uuid4())
    audit = audit_root / execution_id
    audit.mkdir(parents=True, mode=0o700)
    submission = audit / "solution.py"
    with submission.open("x") as stream:
        stream.write(code)
    before = inventory.capture()
    name = "eva-automed-track-" + execution_id
    command = create_command(image=image, name=name, workspace=workspace, submission=submission)
    extras = [(workspace / "inputs-manifest.json", "/workspace/inputs-manifest.json"),
              (workspace / "public-guidance", "/workspace/public-guidance")]
    index = command.index("--workdir")
    for source, destination in extras:
        command[index:index] = ["--mount", f"type=bind,source={source},target={destination},readonly"]
    try:
        created = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30)
        if created.returncode:
            raise EvaluationError("track_docker_create_failed")
        document = json.loads(subprocess.check_output(["docker", "inspect", name], timeout=15))[0]
        extra_mounts = [row for row in document["Mounts"] if row["Destination"] in {dest for _, dest in extras}]
        if {(row["Source"], row["Destination"], row["RW"], row["Type"]) for row in extra_mounts} != {
                (str(source), destination, False, "bind") for source, destination in extras}:
            raise EvaluationError("track_extra_readonly_mount_invalid")
        base_document = {**document, "Mounts": [row for row in document["Mounts"] if row not in extra_mounts]}
        isolation = verify_container(base_document, image=image, workspace=workspace, submission=submission)
        isolation["additional_verified_readonly_mounts"] = [destination for _, destination in extras]
        outcome = bounded_start(name, timeout)
        state = json.loads(subprocess.check_output(["docker", "inspect", name], timeout=15))[0]["State"]
        result = {"schema": "eva.automedbench-track-python.v1", "execution_id": execution_id,
            "code_blake3": file_digest(submission), "exit_code": state["ExitCode"], "oom_killed": state["OOMKilled"],
            "process_started": state["StartedAt"] != "0001-01-01T00:00:00Z", "isolation": isolation,
            "before": before, "after": inventory.capture(), **outcome}
        write_once(audit / "execution.json", result)
        return result
    finally:
        subprocess.run(["docker", "rm", "--force", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)


class TrackTools(PublicTools):
    def __init__(self, *, workspace, audit_root, image, skills, model_manifest, public_python):
        super().__init__(workspace=workspace, audit_root=audit_root, image=image, skills=skills)
        self.inventory = MutableInventory(self.workspace, self.audit_root)
        self.inputs = self.inventory.binding["files"]
        self.model_manifest = model_manifest.absolute()
        self.public_python = public_python.absolute()
        self.job_root = self.audit_root / "model-jobs"
        self.job_root.mkdir(exist_ok=True, mode=0o700)

    def _read(self, arguments):
        relative = arguments["path"]
        if relative not in {"task.json", "inputs-manifest.json"} and not relative.startswith(("notes/", "outputs/", "code/", "public-guidance/", "inputs/")):
            raise EvaluationError("public_text_path_not_allowed")
        if relative.startswith("inputs/") and relative not in {row["path"] for row in self.inputs}:
            raise EvaluationError("public_input_not_declared")
        path = safe_file(self.workspace, relative, maximum=2 * 1024 * 1024)
        data = path.read_bytes()
        offset, limit = arguments.get("offset", 0), arguments.get("limit", 16384)
        return {"path": relative, "content": data[offset:offset+limit].decode("utf-8", errors="replace"),
            "offset": offset, "total_bytes": len(data), "file_blake3": blake3(data).hexdigest(),
            "complete": offset == 0 and limit >= len(data)}

    def _view(self, arguments):
        from PIL import Image
        rows = [row for row in self.inputs if Path(row["path"]).parts[1] == arguments["case_id"]
                and row["path"].lower().endswith((".jpg", ".jpeg", ".png", ".nii.gz", ".npy"))]
        index = arguments.get("image_index", 0)
        if index >= len(rows):
            raise EvaluationError("public_image_not_declared")
        row = rows[index]
        path = safe_file(self.workspace, row["path"])
        if file_digest(path) != row["blake3"]:
            raise EvaluationError("public_input_commitment_changed")
        facts = {"input_path": row["path"], "input_blake3": row["blake3"]}
        if path.name.endswith((".nii.gz", ".npy")):
            import numpy as np
            if path.name.endswith(".nii.gz"):
                import nibabel as nib
                volume = nib.load(path)
                slice_index = arguments.get("slice_index", -1)
                slice_index = volume.shape[2] // 2 if slice_index == -1 else slice_index
                if not 0 <= slice_index < volume.shape[2]:
                    raise EvaluationError("volume_slice_out_of_bounds")
                pixels = np.asarray(volume.dataobj[:, :, slice_index], dtype=np.float32)
                facts.update(shape=list(volume.shape), affine=volume.affine.tolist(), slice_index=slice_index)
            else:
                pixels = np.load(path, allow_pickle=False)
                facts["shape"] = list(pixels.shape)
            rendered = Image.fromarray((np.clip((np.nan_to_num(pixels) + 160) / 400, 0, 1) * 255).astype(np.uint8)).convert("RGB")
            facts["display_transform"] = "original axes; HU window [-160,240]"
        else:
            with Image.open(path) as original:
                facts["original_size"] = list(original.size)
                rendered = original.convert("RGB")
        rendered.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        rendered.save(buffer, format="PNG")
        data = buffer.getvalue()
        facts["image_blake3"] = blake3(data).hexdigest()
        return facts, {"type": "image", "mimeType": "image/png", "data": base64.b64encode(data).decode()}

    def _submit(self, arguments, *, extended=False, generative=False):
        track = self.contract["track"]
        supported = ({"vqa", "report"} if generative else {"segmentation", "synthesis", "enhancement"}
                     if extended else {"classification", "detection"})
        if track not in supported:
            raise EvaluationError("model_job_tool_does_not_support_this_track")
        controls = {key: arguments[key] for key in ("sigma", "hu_min", "hu_max") if key in arguments}
        if track == "enhancement":
            if set(controls) != {"sigma", "hu_min", "hu_max"} or controls["hu_min"] >= controls["hu_max"]:
                raise EvaluationError("explicit_enhancement_normalization_controls_required")
        elif controls:
            raise EvaluationError("enhancement_controls_for_wrong_track")
        if generative:
            allowed = {"max_new_tokens", "multi_image_mode"} if track == "vqa" else {"max_new_tokens", "prompt"}
            controls = {key: value for key, value in arguments.items() if key != "case_ids"}
            if set(controls) != allowed:
                raise EvaluationError("explicit_track_generation_controls_required")
            limit = controls["max_new_tokens"]
            if type(limit) is not int or not 1 <= limit <= (256 if track == "vqa" else 512):
                raise EvaluationError("track_generation_token_limit_invalid")
            if track == "vqa" and controls["multi_image_mode"] != "montage":
                raise EvaluationError("vqa_requires_explicit_all_image_montage")
            if track == "report":
                prompt = controls["prompt"]
                if not isinstance(prompt, str) or not 10 <= len(prompt) <= 1024 or any(
                    token in prompt for token in ("<|", "|>", "<img>", "</img>")
                ):
                    raise EvaluationError("report_prompt_invalid")
        if track not in MODEL_TRACKS or not set(arguments["case_ids"]) <= set(self.contract["case_ids"]):
            raise EvaluationError("prescribed_model_or_cases_not_admitted")
        previous = list(self.job_root.glob("*/submission.json"))
        if len(previous) >= 80:
            raise EvaluationError("track_model_job_budget_exhausted")
        job_id = str(uuid4())
        job = self.job_root / job_id
        job.mkdir(mode=0o700)
        write_once(job / "submission.json", {"job_id": job_id, "track": track, "case_ids": arguments["case_ids"],
            "parameters": controls if extended or generative else {"confidence": arguments.get("confidence", .25)}, "workspace": str(self.workspace),
            "model_manifest_blake3": file_digest(self.model_manifest), "actor_python_executed_on_host": False})
        command = [str(self.public_python), "-B", str(ROOT / "training/benchmark_models/job_supervisor.py"),
            "--workspace", str(self.workspace), "--model-manifest", str(self.model_manifest), "--track", track,
            "--case-ids", *arguments["case_ids"], "--job-id", job_id, "--audit-root", str(self.job_root / "authoritative"),
            "--execute-gpu", "--confidence", str(arguments.get("confidence", .25))]
        for key, value in controls.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        environment = {"PATH": str(self.public_python.parent) + ":/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "0", "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "4",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
        with (job / "process.log").open("xb") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                env=environment, cwd=str(ROOT), close_fds=True, start_new_session=True)
        write_once(job / "process.json", {"pid": process.pid, "job_id": job_id, "command": command,
            "start_ticks": Path(f"/proc/{process.pid}/stat").read_text().split()[21]})
        return self._status({"job_id": job_id})

    def _status(self, arguments):
        job_id = str(UUID(arguments["job_id"]))
        submission = read_document(self.job_root / job_id / "submission.json")
        if submission["track"] != self.contract["track"]:
            raise EvaluationError("job_track_binding_invalid")
        authoritative = self.job_root / "authoritative" / job_id
        receipt_file = authoritative / "receipt.json"
        receipt = json.loads(receipt_file.read_text()) if receipt_file.exists() else {"status": "submitted"}
        if not receipt_file.exists():
            process_file = self.job_root / job_id / "process.json"
            if process_file.exists():
                identity = read_document(process_file)
                proc = Path(f"/proc/{identity['pid']}/stat")
                if not proc.exists() or proc.read_text().split()[21] != identity["start_ticks"]:
                    receipt = {"status": "failed_before_authoritative_receipt"}
        progress = authoritative / "progress.jsonl"
        completed = len(progress.read_bytes().splitlines()) if progress.exists() else 0
        summary = {"job_id": job_id, "status": receipt.get("status"), "requested_cases": len(submission["case_ids"]),
            "observed_case_progress_rows": completed, "case_ids": submission["case_ids"],
            "actor_output_directory": "outputs/agents_outputs/prescribed-model-jobs/" + job_id,
            "analysis_model_is_not_coding_orchestrator": True, "authoritative_receipt_available": receipt_file.exists()}
        ledger = self.workspace / "notes" / "model-jobs.jsonl"
        with ledger.open("ab") as stream:
            stream.write(canonical(summary) + b"\n")
        return summary

    def _capture_inventory(self):
        return self.inventory.capture()

    def _call_locked(self, name, arguments, request_id=None):
        event_id = str(uuid4())
        before = self.inventory.capture()
        image = None
        error = None
        try:
            definition = next(row for row in TOOLS if row["name"] == name)
            jsonschema.Draft202012Validator(definition["inputSchema"]).validate(arguments)
            self.call_count += 1
            if self.call_count > 160:
                raise EvaluationError("track_tool_budget_exhausted")
            if name == "automed_read_file": result = self._read(arguments)
            elif name == "automed_write_note": result = self._note(arguments)
            elif name == "automed_view_input": result, image = self._view(arguments)
            elif name == "automed_submit_model_job": result = self._submit(arguments)
            elif name == "automed_submit_extended_model_job": result = self._submit(arguments, extended=True)
            elif name == "automed_submit_generative_model_job": result = self._submit(arguments, generative=True)
            elif name == "automed_model_job_status": result = self._status(arguments)
            elif name in {"search_skills", "load_skill"}:
                policy = read_document(self.audit_root / "phase-policy.json")
                result = self.skills.call(name, arguments, stage=policy["skill_stage"])
            else:
                self.code_count += 1
                if self.code_count > 32:
                    raise EvaluationError("track_code_budget_exhausted")
                execution = execute_track_python(workspace=self.workspace, audit_root=self.audit_root / "code-executions",
                    image=self.image, code=arguments["code"], timeout=arguments.get("timeout_seconds", 120), inventory=self.inventory)
                result = {key: execution[key] for key in ("execution_id", "exit_code", "stdout", "stderr", "timed_out", "process_started", "oom_killed")}
                result["isolation_verified"] = execution["isolation"]["verified_before_start"]
        except Exception as exc:
            error = str(exc) if isinstance(exc, EvaluationError) else type(exc).__name__
            result = {"error_code": error}
        after = self.inventory.capture()
        structured = {"event_id": event_id, "name": name, "result": result}
        response = {"content": [{"type": "text", "text": canonical(structured).decode()}],
                    "structuredContent": structured, "isError": error is not None}
        if image is not None: response["content"].append(image)
        row = {"schema": "eva.automedbench-track-tool-event.v1", "event_id": event_id, "request_id": request_id,
            "name": name, "arguments": arguments, "result": result, "workspace_before": before, "workspace_after": after,
            "is_error": error is not None, "response_blake3": blake3(canonical(response)).hexdigest()}
        row["event_blake3"] = blake3(canonical(row)).hexdigest()
        with (self.audit_root / "mcp-events.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")
        return response


def serve(tools):
    lock = Lock()
    def dispatch(request):
        identifier = request.get("id")
        if identifier is None: return
        method = request.get("method")
        if method == "initialize": result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "automed_eval", "version": "2.0.0"}}
        elif method == "tools/list": result = {"tools": list(TOOLS)}
        elif method == "tools/call":
            params = request.get("params", {})
            result = tools.call(params.get("name"), params.get("arguments", {}), identifier)
        elif method == "ping": result = {}
        elif method in {"resources/list", "resources/templates/list", "prompts/list"}:
            result = {{"resources/list": "resources", "resources/templates/list": "resourceTemplates", "prompts/list": "prompts"}[method]: []}
        else: result = None
        response = {"jsonrpc": "2.0", "id": identifier, **({"result": result} if result is not None else {"error": {"code": -32601, "message": "method not exposed"}})}
        with lock:
            sys.stdout.buffer.write(canonical(response) + b"\n")
            sys.stdout.buffer.flush()
    with ThreadPoolExecutor(max_workers=4) as executor:
        for line in sys.stdin.buffer:
            if len(line) > 1024 * 1024: raise EvaluationError("mcp_request_too_large")
            executor.submit(dispatch, json.loads(line))
