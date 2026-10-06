"""Unchanged public benchmark tools plus an explicitly named bwrap executor."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
from threading import Lock, Thread, Event
from uuid import uuid4, UUID

import jsonschema

from training.automedbench_lite.adapter import canonical, blake3, EvaluationError
from training.automedbench_lite.track_tools import TrackTools, TOOLS as ORIGINAL_TOOLS

from .benchmark_sandbox import execute
from .benchmark_inventory import PublishingInventory
from .benchmark_host_lifecycle import HostDispatchLedger


PYTHON_TOOL = {
    "name": "evamed_execute_python",
    "description": (
        "Run bounded scientific Python in an isolated CPU Linux namespace. NumPy, SciPy, Pillow, "
        "Nibabel, Pandas and scikit-image are available. The working directory is /workspace; "
        "inputs/, task.json, inputs-manifest.json and public-guidance/ are read-only. "
        "Write reusable programs under code/ and submissions under outputs/agents_outputs/. "
        "Edits publish after a successful call before the track deadline; failed or timed-out calls keep no public edits. "
        "No network, GPU, host files, credentials or package installation are available. "
        "Returns actual stdout, stderr, exit status and execution receipt identity."
    ),
    "inputSchema": {
        "type": "object", "properties": {
            "code": {"type": "string", "maxLength": 65536},
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 180}},
        "required": ["code"], "additionalProperties": False,
    },
}


def catalog(candidate: bool) -> tuple[dict, ...]:
    omitted = {"automed_execute_python"}
    if not candidate:
        omitted |= {"search_skills", "load_skill"}
    return tuple(row for row in ORIGINAL_TOOLS if row["name"] not in omitted) + (PYTHON_TOOL,)


class BenchmarkTools(TrackTools):
    def __init__(self, *, config: dict, skills):
        self.configuration = config
        self.job_launch_lock = Lock()
        super().__init__(workspace=Path(config["workspace"]), audit_root=Path(config["audit_root"]),
                         image="unused-bwrap-executor", skills=skills,
                         model_manifest=Path(config["runtime_manifest"]),
                         public_python=Path(config["task_python"]))
        self.inventory = PublishingInventory(self.workspace, self.audit_root)
        self.definitions = {row["name"]: row for row in catalog(config["candidate"])}
        retained = self.audit_root / "mcp-events.jsonl"
        if retained.exists():
            self.code_count += sum(json.loads(line)["name"] == PYTHON_TOOL["name"]
                                   for line in retained.read_bytes().splitlines())

    def _require_operation_budget(self):
        deadline = self.configuration.get("track_deadline_monotonic")
        if deadline is not None and time.monotonic() >= deadline:
            raise EvaluationError("track_wall_clock_budget_exhausted")

    def _read(self, arguments):
        self._require_operation_budget()
        return super()._read(arguments)

    def _note(self, arguments):
        self._require_operation_budget()
        from training.benchmark_models._publication import DeadlineGuard
        guard = DeadlineGuard(self.audit_root / 'host-publication', self.configuration.get('track_deadline_monotonic'))
        result = []
        guard.publish('automed_write_note', self.workspace / 'notes' / arguments['name'],
                      lambda: result.append(TrackTools._note(self, arguments)))
        return result[0]

    def _view(self, arguments):
        self._require_operation_budget()
        return super()._view(arguments)

    def _status(self, arguments):
        own_submission = getattr(self, '_observing_own_new_submission', False)
        if not own_submission:
            self._require_operation_budget()
        from training.automedbench_lite.adapter import read_document
        from training.benchmark_models._publication import DeadlineGuard, PolicyDeadlineExceeded
        job_id = str(UUID(arguments['job_id']))
        submission = read_document(self.job_root / job_id / 'submission.json')
        if submission['track'] != self.contract['track']:
            raise EvaluationError('job_track_binding_invalid')
        authoritative = self.job_root / 'authoritative' / job_id
        receipt_file = authoritative / 'receipt.json'
        receipt = json.loads(receipt_file.read_text()) if receipt_file.exists() else {'status': 'submitted'}
        if not receipt_file.exists():
            identity_path = self.job_root / job_id / 'process.json'
            if identity_path.exists():
                identity = read_document(identity_path)
                try:
                    fields = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(')', 1)[1].split()
                    live = fields[19] == str(identity['start_ticks']) and fields[0] != 'Z'
                except FileNotFoundError:
                    live = False
                if not live:
                    receipt = {'status': 'failed_before_authoritative_receipt'}
        progress = authoritative / 'progress.jsonl'
        summary = {'job_id': job_id, 'status': receipt.get('status'), 'requested_cases': len(submission['case_ids']),
            'observed_case_progress_rows': len(progress.read_bytes().splitlines()) if progress.exists() else 0,
            'case_ids': submission['case_ids'], 'actor_output_directory': 'outputs/agents_outputs/prescribed-model-jobs/' + job_id,
            'analysis_model_is_not_coding_orchestrator': True, 'authoritative_receipt_available': receipt_file.exists()}
        ledger = self.workspace / 'notes/model-jobs.jsonl'
        guard = DeadlineGuard(self.audit_root / 'host-publication', self.configuration.get('track_deadline_monotonic'))
        def publish():
            with ledger.open('ab') as stream:
                stream.write(canonical(summary) + b'\n')
        try:
            guard.publish('model_job_status_ledger', ledger, publish)
        except PolicyDeadlineExceeded:
            if not own_submission:
                raise EvaluationError('track_wall_clock_budget_exhausted')
            # Registration already occurred before cutoff. Return its actual ID
            # as host observation, with no additional public ledger mutation.
        return summary

    def _submit(self, arguments, *, extended=False, generative=False):
        # Only process launch/registration shares the deadline cleanup lock.
        # Slow workspace snapshots must never prevent stopping owned GPU jobs.
        with self.job_launch_lock:
            self._require_operation_budget()
            self._observing_own_new_submission = True
            try:
                return super()._submit(arguments, extended=extended, generative=generative)
            finally:
                self._observing_own_new_submission = False

    def _call_locked(self, name, arguments, request_id=None):
        deadline = self.configuration.get("track_deadline_monotonic")
        expired = deadline is not None and time.monotonic() >= deadline
        if not expired and name in self.definitions and name not in {PYTHON_TOOL["name"], "search_skills", "load_skill"}:
            return super()._call_locked(name, arguments, request_id)
        event_id = str(uuid4())
        before = self.inventory.capture()
        error = None
        try:
            self._require_operation_budget()
            if name not in self.definitions:
                raise EvaluationError("tool_not_offered_by_selected_profile")
            jsonschema.Draft202012Validator(self.definitions[name]["inputSchema"]).validate(arguments)
            self.call_count += 1
            if self.call_count > 160:
                raise EvaluationError("track_tool_budget_exhausted")
            if name == PYTHON_TOOL["name"]:
                self.code_count += 1
                if self.code_count > 32:
                    raise EvaluationError("track_code_budget_exhausted")
                result = execute(bwrap=Path(self.configuration["bwrap"]),
                                 runtime=Path(self.configuration["scientific_root"]),
                                 workspace=self.workspace, audit_root=self.audit_root / "code-executions",
                                 code=arguments["code"], timeout=arguments.get("timeout_seconds", 120),
                                 inventory=self.inventory, python=self.configuration["scientific_python"],
                                 track_deadline=deadline)
                error = result.get('publication_policy_error')
                result = {key: result[key] for key in ("execution_id", "exit_code", "stdout", "stderr",
                          "timed_out", "stream_limit_exceeded", "process_started", "wall_seconds", "isolation",
                          "publication_policy_error")}
            else:
                # E2E workflows may select any S1-S5/E2E skill stage. The exact
                # canonical stage-bound handler still enforces each skill grant.
                stage = arguments["stage"]
                if stage not in self.skills.definitions:
                    raise EvaluationError("skill_stage_not_available")
                result = self.skills.call(name, arguments, stage=stage)
        except Exception as exc:
            error = str(exc) if isinstance(exc, EvaluationError) else type(exc).__name__
            result = {"error_code": error}
        after = self.inventory.capture()
        structured = {"event_id": event_id, "name": name, "result": result}
        response = {"content": [{"type": "text", "text": canonical(structured).decode()}],
                    "structuredContent": structured, "isError": error is not None}
        row = {"schema": "eva.automedbench-track-tool-event.v1", "event_id": event_id,
               "request_id": request_id, "name": name, "arguments": arguments, "result": result,
               "workspace_before": before, "workspace_after": after, "is_error": error is not None,
               "response_blake3": blake3(canonical(response)).hexdigest()}
        row["event_blake3"] = blake3(canonical(row)).hexdigest()
        with (self.audit_root / "mcp-events.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")
        return response


def serve(config: dict) -> None:
    from training.automedbench_lite.skill_surface import VerifiedEvaluationSkills

    skill_root = Path(config["audit_root"]) / "skill-materialization"
    skill_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    skill_root.chmod(0o700)
    skills = VerifiedEvaluationSkills(skill_root) if config["candidate"] else None
    tools = BenchmarkTools(config=config, skills=skills)
    output_lock = Lock()
    stopping = Event()
    lifecycle = HostDispatchLedger(Path(config["audit_root"]), config.get("track_deadline_monotonic"))

    def deadline_cleanup():
        """Stop owned GPU workers at the same deadline as Codex and Python."""
        import asyncio
        from training.automedbench_lite.adapter import write_once
        from .benchmark_cleanup import cleanup_owned_jobs
        deadline = config.get("track_deadline_monotonic")
        if deadline is None or stopping.wait(max(0, deadline - time.monotonic())):
            return
        try:
            # CPU calls enforce the same monotonic deadline independently.
            # Coordinate only GPU process launch/registration, not snapshots.
            with tools.job_launch_lock:
                asyncio.run(cleanup_owned_jobs(tools.audit_root, tools.workspace,
                    trigger="track_wall_clock_deadline", filename="deadline-job-cleanup.json"))
        except Exception as exc:
            write_once(tools.audit_root / "deadline-cleanup-failure.json", {"error_type": type(exc).__name__})

    watchdog = Thread(target=deadline_cleanup, daemon=True)
    watchdog.start()

    def _dispatch(request):
        identifier = request.get("id")
        if identifier is None:
            return
        method = request.get("method")
        dispatch_id = None
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "automed_eval", "version": "evamed-codex-benchmark-v1"}}
        elif method == "tools/list":
            result = {"tools": list(tools.definitions.values())}
        elif method == "tools/call":
            params = request.get("params", {})
            dispatch_id = lifecycle.start(request)
            try:
                result = tools.call(params.get("name"), params.get("arguments", {}), identifier)
            except Exception as exc:
                lifecycle.append("dispatch_failed", dispatch_id, error_type=type(exc).__name__)
                raise
            lifecycle.complete(dispatch_id, result)
        elif method == "ping":
            result = {}
        elif method in {"resources/list", "resources/templates/list", "prompts/list"}:
            key = {"resources/list": "resources", "resources/templates/list": "resourceTemplates",
                   "prompts/list": "prompts"}[method]
            result = {key: []}
        else:
            result = None
        response = {"jsonrpc": "2.0", "id": identifier,
                    **({"result": result} if result is not None else
                       {"error": {"code": -32601, "message": "method not exposed"}})}
        try:
            with output_lock:
                sys.stdout.buffer.write(canonical(response) + b"\n")
                sys.stdout.buffer.flush()
        except BrokenPipeError:
            closure = lifecycle.delivery(dispatch_id, written=False) if dispatch_id else None
            if closure is not None and closure["deadline_elapsed"]:
                # This is delivery evidence, not a fabricated tool response.
                # Only the independently captured native policy terminal can
                # establish that this closure is a controlled interruption.
                return
            raise
        if dispatch_id is not None:
            lifecycle.delivery(dispatch_id, written=True)

    def dispatch(request):
        try:
            _dispatch(request)
        except Exception as exc:
            # Executor futures must not silently swallow an inventory/host
            # failure and leave Codex waiting for its200-second MCP timeout.
            import traceback
            diagnostic = {"schema": "eva.benchmark-mcp-infrastructure-error.v1",
                "error_id": str(uuid4()), "request_id": request.get("id"),
                "method": request.get("method"), "error_type": type(exc).__name__,
                "error_code": str(exc) if isinstance(exc, EvaluationError) else type(exc).__name__,
                "frames": [{"file": Path(frame.filename).name, "line": frame.lineno,
                            "function": frame.name} for frame in traceback.extract_tb(exc.__traceback__)],
                "host_result_synthesized": False}
            diagnostic["document_blake3"] = blake3(canonical(diagnostic)).hexdigest()
            response = {"jsonrpc": "2.0", "id": request.get("id"), "error": {
                "code": -32603, "message": "Host infrastructure failure; no tool result is available.",
                "data": {"error_id": diagnostic["error_id"], "error_code": diagnostic["error_code"]}}}
            with output_lock:
                with (Path(config["audit_root"]) / "mcp-infrastructure-errors.jsonl").open("ab") as stream:
                    stream.write(canonical(diagnostic) + b"\n")
                if not isinstance(exc, BrokenPipeError):
                    sys.stdout.buffer.write(canonical(response) + b"\n")
                    sys.stdout.buffer.flush()

    with ThreadPoolExecutor(max_workers=4) as executor:
        while line := sys.stdin.buffer.readline(1024 * 1024 + 1):
            if len(line) > 1024 * 1024:
                raise EvaluationError("mcp_request_too_large")
            executor.submit(dispatch, json.loads(line))
    stopping.set()
    watchdog.join(timeout=25)
