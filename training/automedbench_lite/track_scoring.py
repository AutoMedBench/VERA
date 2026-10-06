"""Host-side whole-track scorer handoff; private reference bytes stay in worker."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from uuid import UUID, uuid4

from blake3 import blake3

from .adapter import EvaluationError, REVISION, canonical, file_digest, read_document, safe_file, write_once
from .track_adapter import BY_TRACK, TrackRelease


def _require(value, code):
    if not value:
        raise EvaluationError(code)


def _json_file(root, relative, *, digest=None):
    path = safe_file(root, relative, maximum=32 * 1024**2)
    payload = path.read_bytes()
    _require(digest is None or blake3(payload).hexdigest() == digest, "model_evidence_digest_changed")
    return json.loads(payload), blake3(payload).hexdigest()


def completed_model_evidence(*, audit, workspace, track, input_binding, hosts, visible_files=None):
    """Verify only actually observed terminal jobs, not a later audit-directory scan.

    `visible_files` is the retained phase-after inventory, never live actor state.
    Module bytes are supplemental host evidence; no claim every source line ran
    or that the coding actor authored the provided analysis implementation.
    """
    jobs, seen = [], set()
    submissions = {}
    for host in hosts:
        _require(host["event_blake3"] == blake3(canonical({k: v for k, v in host.items()
            if k != "event_blake3"})).hexdigest(), "model_host_event_digest_changed")
        result = host.get("result", {})
        job_id = result.get("job_id")
        if host.get("is_error") or not job_id:
            continue
        _require(str(UUID(job_id)) == job_id, "model_job_uuid_invalid")
        if host["name"] in {"automed_submit_model_job", "automed_submit_extended_model_job", "automed_submit_generative_model_job"}:
            _require(job_id not in submissions, "model_job_submitted_twice")
            submissions[job_id] = host
        if host["name"] not in {"automed_submit_model_job", "automed_submit_extended_model_job",
                "automed_submit_generative_model_job", "automed_model_job_status"} or job_id in seen:
            continue
        relative_root = "outputs/agents_outputs/prescribed-model-jobs/" + job_id
        observed = visible_files.get(relative_root + "/receipt.json") if visible_files is not None else None
        if result.get("status") != "complete" and observed is None:
            continue
        _require(job_id in submissions, "model_job_missing_actual_submission")
        origin = submissions[job_id]
        if host["name"] == "automed_model_job_status":
            _require(host["arguments"].get("job_id") == job_id, "model_status_job_changed")
        job_root = audit / "model-jobs" / "authoritative" / job_id
        submission = read_document(audit / "model-jobs" / job_id / "submission.json")
        if not (job_root / "receipt.json").exists() and result.get("status") != "complete":
            continue
        receipt, receipt_digest = _json_file(job_root, "receipt.json")
        if result.get("status") != "complete" and (receipt.get("status") != "complete"
                or observed is None or observed["blake3"] != receipt_digest):
            continue  # Pending at the selected phase, even if it completed later.
        if visible_files is not None:
            _require(observed is not None and observed["blake3"] == receipt_digest,
                     "model_terminal_receipt_not_in_stage_snapshot")
        cases = submission["case_ids"]
        _require(submission["job_id"] == receipt["job_id"] == job_id and submission["track"] == receipt["track"] == track
            and Path(submission["workspace"]).resolve() == workspace.resolve()
            and cases == receipt["case_ids"] == origin["arguments"]["case_ids"] == result["case_ids"]
            and len(cases) == len(set(cases)) and receipt["input_manifest_blake3"] == input_binding
            and receipt["model_manifest_blake3"] == submission["model_manifest_blake3"]
            and receipt["schema"] == "eva.prescribed-public-model-job.v1"
            and receipt["status"] == "complete" and receipt.get("success") is True
            and receipt["authored_by"] == "provided_analysis_tool"
            and receipt.get("private_reference_access") is False and receipt.get("clinical_score") is None,
            "model_job_receipt_binding_changed")
        # Capture exact retained sources, not whatever is installed at scoring time.
        sources = []
        for row in [{"path": "executed-helper.py", "blake3": receipt["helper_source_blake3"]},
                    *receipt["executed_module_sources"], *receipt.get("executed_model_sources", [])]:
            path = safe_file(job_root, row["path"], maximum=2 * 1024**2)
            _require(file_digest(path) == row["blake3"], "model_executed_source_changed")
            _require(row["path"] == "executed-helper.py" or row["path"].startswith(("executed-modules/", "executed-model-source/")),
                     "model_executed_source_scope")
            sources.append({**row, "bytes": path.stat().st_size,
                # Custom model source inventories are retained and verified, not
                # stuffed into the Judge context. The actual invoking helpers are.
                **({"text": path.read_text()} if not row["path"].startswith("executed-model-source/") else {})})
        raw, raw_digest = _json_file(job_root, "raw-output.json", digest=receipt["raw_output_blake3"])
        _require(isinstance(raw, list) and [item["case_id"] for item in raw] == cases,
                 "model_actual_case_output_coverage")
        case_rows = [json.loads(line) for line in safe_file(job_root, "case-outputs.jsonl").read_bytes().splitlines() if line]
        progress = [json.loads(line) for line in safe_file(job_root, "progress.jsonl").read_bytes().splitlines() if line]
        _require(case_rows == raw and [item["case_id"] for item in progress] == cases
            and all(item["event"] == "case_inference_complete" for item in progress), "model_actual_progress_changed")
        artifacts, details = [], []
        expected = []
        for item in raw:
            case_artifacts = {}
            for kind in ("artifact", "raw_output", "candidate_output", "request", "model_input", "smoke_output"):
                value = item.get(kind)
                if value is None:
                    continue
                relative = PurePosixPath(value["path"])
                _require(relative.is_relative_to(relative_root), "model_artifact_wrong_job")
                audit_path = "artifacts/" + relative.relative_to(relative_root).as_posix()
                record = {"kind": kind, "case_id": item["case_id"], "path": audit_path,
                    "bytes": value["bytes"], "blake3": value["blake3"]}
                expected.append(record)
                actual = safe_file(job_root, audit_path, maximum=2 * 1024**3)
                _require(actual.stat().st_size == record["bytes"] and file_digest(actual) == record["blake3"],
                         "model_retained_artifact_changed")
                artifacts.append(record)
                if kind in {"raw_output", "candidate_output", "smoke_output"} and actual.suffix == ".json":
                    case_artifacts[kind] = _json_file(job_root, audit_path, digest=value["blake3"])[0]
            details.append({"case_id": item["case_id"], **case_artifacts})
        _require(expected == receipt.get("host_retained_artifacts", []), "model_retained_artifact_inventory_changed")
        process_exit = None
        exit_path = job_root / "process-exit.json"
        if exit_path.exists():
            process_exit, exit_digest = _json_file(job_root, "process-exit.json")
            _require(process_exit.get("schema") == "eva.prescribed-model-process-exit.v1"
                and process_exit.get("job_id") == job_id and process_exit.get("os_process_exit_observed") is True
                and type(process_exit.get("returncode")) is int
                and process_exit.get("worker_receipt_blake3") == receipt_digest
                and process_exit.get("worker_source_blake3_at_launch") == receipt["helper_source_blake3"],
                "model_process_exit_binding_changed")
            if visible_files is not None:
                visible_exit = visible_files.get(relative_root + "/process-exit.json")
                if visible_exit is None:
                    process_exit = None  # Actual exit occurred after this phase; do not import it.
                else:
                    _require(visible_exit["blake3"] == exit_digest, "model_process_exit_snapshot_changed")
        jobs.append({"job_id": job_id, "track": track, "host_event_id": host["event_id"],
            "completion_evidence_scope": "retained_phase_after_snapshot" if visible_files is not None else "actual_complete_status_call",
            "completion_observed_in_this_tool_response": result.get("status") == "complete",
            "submission_host_event_id": origin["event_id"], "receipt_blake3": receipt_digest,
            "submission_document_blake3": submission["document_blake3"], "authored_by": "provided_analysis_tool",
            "actor_authored": False, "source_lines_all_executed": False,
            "inference_completed": receipt.get("inference_completed", False), "execution_outcome": receipt.get("execution_outcome"),
            "process_exit": process_exit, "analysis_model": {key: receipt["analysis_model"].get(key)
                for key in ("repository", "revision")}, "inference_controls": receipt.get("inference_controls", {}),
            "case_ids": cases, "sources": sources, "artifacts": artifacts, "case_details": details,
            "raw_output_blake3": raw_digest, "clinical_score": None})
        seen.add(job_id)
    return jobs


def inference_conversation(jobs):
    """Native verifier projection: exit zero ONLY from an observed OS exit."""
    executions = []
    for job in jobs:
        observed = job["process_exit"]
        if observed is None:
            continue
        executions.append({"job_id": job["job_id"], "authored_by": "provided_analysis_tool",
            "actor_authored": False, "exit_code": observed["returncode"], "os_process_exit_observed": True,
            "code": "\n\n".join("# Retained provided source: " + row["path"] + "\n" + row["text"]
                for row in job["sources"] if "text" in row), "stdout_preview": "",
            "actual_job_receipt_blake3": job["receipt_blake3"], "case_ids": job["case_ids"]})
    return {"schema": "eva.automedbench-native-inference-conversation.v1", "authored_by": "host_actual_execution_projection",
        "code_executions": executions, "provided_analysis_tool_is_not_actor_authored": True,
        "function_return_is_not_os_exit": True}


def vqa_submission_evidence(stage, jobs):
    """Bind submitted smoke/calibration to real decodes; never fill missing work."""
    smoke = stage / "smoke_forward.json"
    smoke_bound = False
    if smoke.exists():
        digest = file_digest(smoke)
        smoke_bound = any(row["kind"] == "smoke_output" and row["blake3"] == digest
            and job["process_exit"] is not None and job["process_exit"]["returncode"] == 0
            for job in jobs for row in job["artifacts"])
        _require(smoke_bound, "vqa_submitted_smoke_not_actual_successful_job")
    calibration = stage / "s3_calibration.json"
    facts = {"smoke_submitted_and_bound": smoke_bound, "calibration_expected_public_records": 15,
             "calibration_present": calibration.exists(), "calibration_records": 0,
             "calibration_actual_raw_decodes_bound": False, "stage_completion_claimed": False}
    if calibration.exists():
        value = json.loads(calibration.read_bytes())
        records = value  # Pinned Lite requires a list, not a host-invented wrapper.
        if isinstance(records, list):
            facts["calibration_records"] = len(records)
            actual = {(item["case_id"], item["raw_output"].get("raw_model_output"))
                for job in jobs for item in job["case_details"] if "raw_output" in item}
            facts["calibration_actual_raw_decodes_bound"] = (len(records) == 15
                and all(isinstance(row, dict) and isinstance(row.get("question_id"), str)
                    and isinstance(row.get("raw_model_output"), str) and bool(row["raw_model_output"].strip())
                    and (row["question_id"], row["raw_model_output"]) in actual for row in records)
                and len({row["question_id"] for row in records if isinstance(row, dict)}) == 15)
    return facts


def output_paths(track, selected):
    if track == "classification":
        return ["predictions.csv"] + [case + "/prediction.json" for case in selected]
    filename = {"detection": "prediction.json", "segmentation": "dseg.nii.gz", "synthesis": "sct.nii.gz",
                "vqa": "answer.json", "report": "report.txt", "enhancement": "enhanced.npy"}[track]
    return [case + "/" + filename for case in selected]


def score_track(run: Path, track_name: str, evaluator_python: Path, *, timeout=600):
    """Score once, including native missing-output rules; never make up a reward."""
    run = run.resolve(strict=True)
    manifest = read_document(run / "track-run-manifest.json")
    track = BY_TRACK[track_name]
    rows = [row for row in manifest["tracks"] if row["track"] == track_name]
    if len(rows) != 1:
        raise EvaluationError("score_track_not_in_run")
    row = rows[0]
    source = TrackRelease(Path(row["source_receipt_path"]))
    if source.document["document_blake3"] != row["source_acquisition_document_blake3"]:
        raise EvaluationError("score_source_binding_changed")
    workspace = run / row["workspace_relative"]
    if file_digest(safe_file(workspace, "task.json")) != row["task_file_blake3"]:
        raise EvaluationError("score_public_task_changed")
    task = read_document(workspace / "task.json")
    selected = task["case_ids"]
    if len(selected) != track.count or selected != sorted(set(selected)):
        raise EvaluationError("score_full_selection_invalid")
    # A terminal actor receipt is mandatory. A failed actor remains failed even
    # if a native scorer can assess its partial submission.
    actor = read_document(run / "track-rollouts" / track_name / "rollout.json")
    if not actor.get("completed_requested_turns") or actor.get("errors"):
        raise EvaluationError("actor_workflow_incomplete_native_score_unknown")
    score_root = run / "native-scores" / track_name
    score_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_once(score_root / "attempt.json", {"score_id": str(uuid4()), "track": track_name,
        "one_attempt": True, "selected_case_count": len(selected), "actor_document_blake3": actor["document_blake3"]})
    commitments = {relative: value["blake3"] for relative, value in source.inventory.items()
        if relative.startswith((track.prefix + "/", "automedbench_release/"))}
    files = []
    jobs = []
    vqa_facts = None
    try:
        audit = run / "track-rollouts" / track_name
        host_path = audit / "mcp-events.jsonl"
        hosts = [json.loads(line) for line in host_path.read_bytes().splitlines() if line] if host_path.exists() else []
        after_manifests = sorted((audit / "turns").glob("*/after/manifest.json"))
        _require(bool(after_manifests), "native_score_actual_final_snapshot_missing")
        final_inventory = read_document(after_manifests[-1], maximum=16 * 1024**2)
        jobs = completed_model_evidence(audit=audit, workspace=workspace, track=track_name,
            input_binding=row["input_manifest_file_blake3"], hosts=hosts,
            visible_files={item["path"]: item for item in final_inventory["files"]})
        with tempfile.TemporaryDirectory(prefix="eva-automed-track-score-", dir="/tmp") as temporary:
            stage = Path(temporary)
            auxiliary = ["smoke_forward.json", "answer_postprocess.py", "s3_calibration.json"] if track_name == "vqa" else []
            for relative in output_paths(track_name, selected) + auxiliary:
                original = workspace / "outputs/agents_outputs" / relative
                if not original.exists():
                    continue
                original = safe_file(workspace / "outputs/agents_outputs", relative)
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copyfile(original, target)
                digest = file_digest(target)
                if digest != file_digest(original):
                    raise EvaluationError("track_submission_changed_during_copy")
                files.append({"path": relative, "bytes": target.stat().st_size, "blake3": digest})
            conversation_path = None
            if track_name == "vqa":
                # Neither actor declarations nor installed source can prove a
                # decode. The exact host audit and actual OS wait provide it.
                conversation = score_root / "native-inference-conversation.json"
                write_once(conversation, inference_conversation(jobs))
                vqa_facts = vqa_submission_evidence(stage, jobs)
                conversation_path = str(conversation)
            request = {"release_root": str(source.scorer), "submission_root": str(stage), "track": track_name,
                "selected_case_ids": selected, "conversation_path": conversation_path,
                "source_commitments": commitments, "full_public_subset": True}
            completed = subprocess.run([str(evaluator_python.absolute()), "-I", "-B",
                str(Path(__file__).with_name("track_native_worker.py"))], input=canonical(request),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=stage,
                env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONNOUSERSITE": "1",
                    "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "TMPDIR": str(stage),
                    "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"}, timeout=timeout)
            if completed.returncode or len(completed.stdout) > 8192:
                raise EvaluationError("native_track_worker_failed")
            native = json.loads(completed.stdout)
            value = native.get("task_score_0_1")
            if (native.get("schema") != "eva.automedbench-native-track-result.v1" or native.get("track") != track_name
                    or native.get("selected_case_count") != len(selected) or native.get("full_public_subset") is not True
                    or native.get("private_values_exported") is not False or type(value) not in (int, float)
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise EvaluationError("native_track_result_invalid")
    except Exception as exc:
        write_once(score_root / "failure.json", {"track": track_name, "status": "failed", "reward": None,
            "error_code": str(exc) if isinstance(exc, EvaluationError) else type(exc).__name__})
        raise EvaluationError("native_track_scoring_failed_no_fallback") from None
    return write_once(score_root / "score.json", {"schema": "eva.automedbench-native-whole-track-score.v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "track": track_name, "run_id": manifest["run_id"],
        "release_revision": REVISION, "source_acquisition_document_blake3": source.document["document_blake3"],
        "actor_document_blake3": actor["document_blake3"], "actor_completed_requested_turns": actor["completed_requested_turns"],
        "submitted_files": files, "native_result": native, "one_coding_workflow": True,
        "provided_analysis_tools": [{"job_id": job["job_id"], "authored_by": job["authored_by"],
            "actor_authored": False, "receipt_blake3": job["receipt_blake3"], "analysis_model": job["analysis_model"],
            "case_ids": job["case_ids"], "process_exit": job["process_exit"],
            "retained_artifact_count": len(job["artifacts"]), "raw_output_blake3": job["raw_output_blake3"]} for job in jobs],
        "vqa_public_execution_facts": vqa_facts,
        "official_five_repeat_leaderboard": False, "agent_judged": False,
        "private_reference_contents_read_by_host": False, "score_exposed_to_actor": False})
