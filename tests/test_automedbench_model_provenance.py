"""CPU-only synthetic audit fixtures; no claimed model inference or scores."""
import json
from uuid import uuid4

from blake3 import blake3
import pytest

from training.automedbench_lite.adapter import EvaluationError, canonical, file_digest, write_once
from training.automedbench_lite.track_scoring import (
    completed_model_evidence, inference_conversation, vqa_submission_evidence,
)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) if not isinstance(value, bytes) else value)
    return file_digest(path)


def event(name, arguments, result):
    row = {"event_id": str(uuid4()), "name": name, "arguments": arguments, "result": result, "is_error": False}
    return {**row, "event_blake3": blake3(canonical(row)).hexdigest()}


def job_fixture(tmp_path, track="vqa", *, os_exit=True, status="complete"):
    workspace = tmp_path / "actor"
    workspace.mkdir()
    audit = tmp_path / "audit"
    job_id = str(uuid4())
    root = audit / "model-jobs/authoritative" / job_id
    relative = "outputs/agents_outputs/prescribed-model-jobs/" + job_id
    root.mkdir(parents=True)
    source = b"# Synthetic fixture, never executed\nmodel = load_pretrained_model()\nmodel.generate()\n"
    helper_digest = put(root / "executed-helper.py", source)
    module_digest = put(root / "executed-modules/runtime.py", b"# Provided fixture module\n")
    model_digest = put(root / "executed-model-source/vqa/config.json", {"fixture": True})
    artifacts, output = [], {"case_id": "CASE", "authored_by": "provided_analysis_tool"}
    values = {"raw_output": ("CASE/model-response.json", {"raw_model_output": "Answer: A", "actual_generate_returned": True}),
        "candidate_output": ("CASE/answer.json", {"question_id": "CASE", "raw_model_output": "Answer: A"}),
        "smoke_output": ("smoke_forward.json", {"model_name": "fixture", "device": "cuda:0", "wall_s": .5,
            "raw_output_sample": "Answer: A", "success": True})} if track in {"vqa", "report"} else {
        "artifact": ("CASE/artifact.bin", b"actual synthetic fixture bytes")}
    for kind, (path, value) in values.items():
        target = root / "artifacts" / path
        digest = put(target, value)
        output[kind] = {"path": relative + "/" + path, "bytes": target.stat().st_size, "blake3": digest}
        artifacts.append({"kind": kind, "case_id": "CASE", "path": "artifacts/" + path,
            "bytes": target.stat().st_size, "blake3": digest})
    raw_digest = put(root / "raw-output.json", [output])
    put(root / "case-outputs.jsonl", canonical(output) + b"\n")
    put(root / "progress.jsonl", canonical({"case_id": "CASE", "event": "case_inference_complete"}) + b"\n")
    receipt = {"schema": "eva.prescribed-public-model-job.v1", "job_id": job_id, "track": track,
        "case_ids": ["CASE"], "input_manifest_blake3": "a" * 64, "model_manifest_blake3": "b" * 64,
        "status": "complete", "success": True, "authored_by": "provided_analysis_tool", "private_reference_access": False,
        "clinical_score": None, "helper_source_blake3": helper_digest,
        "executed_module_sources": [{"path": "executed-modules/runtime.py", "blake3": module_digest}],
        "executed_model_sources": [{"asset": "vqa", "path": "executed-model-source/vqa/config.json", "blake3": model_digest}],
        "raw_output_blake3": raw_digest, "host_retained_artifacts": artifacts,
        "analysis_model": {"repository": "fixture/model", "revision": "fixture"},
        "inference_completed": True, "execution_outcome": {"kind": "trusted_python_function_return", "success": True,
            "os_process_exit_observed": False}}
    receipt_digest = put(root / "receipt.json", receipt)
    visible = {relative + "/receipt.json": {"blake3": receipt_digest}}
    if os_exit:
        digest = put(root / "process-exit.json", {"schema": "eva.prescribed-model-process-exit.v1", "job_id": job_id,
            "os_process_exit_observed": True, "returncode": 0, "worker_receipt_blake3": receipt_digest,
            "worker_source_blake3_at_launch": helper_digest})
        visible[relative + "/process-exit.json"] = {"blake3": digest}
    submission_root = audit / "model-jobs" / job_id
    submission_root.mkdir()
    write_once(submission_root / "submission.json", {"job_id": job_id, "track": track, "case_ids": ["CASE"],
        "workspace": str(workspace), "model_manifest_blake3": "b" * 64})
    host = event("automed_submit_generative_model_job" if track in {"vqa", "report"} else "automed_submit_extended_model_job",
        {"case_ids": ["CASE"]}, {"job_id": job_id, "status": status, "case_ids": ["CASE"]})
    return {"audit": audit, "workspace": workspace, "track": track, "input_binding": "a" * 64,
            "hosts": [host], "visible_files": visible}, root


@pytest.mark.parametrize("track", ["vqa", "report", "segmentation", "synthesis", "enhancement"])
def test_actual_job_sources_and_artifacts_are_provided_not_actor_code(tmp_path, track):
    args, root = job_fixture(tmp_path, track)
    jobs = completed_model_evidence(**args)
    assert len(jobs) == 1 and jobs[0]["authored_by"] == "provided_analysis_tool"
    assert not jobs[0]["actor_authored"] and not jobs[0]["source_lines_all_executed"]
    assert jobs[0]["sources"][0]["text"].encode() == (root / "executed-helper.py").read_bytes()
    assert jobs[0]["artifacts"] and jobs[0]["clinical_score"] is None
    execution = inference_conversation(jobs)["code_executions"][0]
    assert execution["exit_code"] == 0 and execution["os_process_exit_observed"]
    assert not execution["actor_authored"] and ".generate()" in execution["code"]


def test_function_return_never_fabricates_os_exit(tmp_path):
    args, _ = job_fixture(tmp_path, os_exit=False)
    jobs = completed_model_evidence(**args)
    assert jobs[0]["inference_completed"] and jobs[0]["process_exit"] is None
    assert inference_conversation(jobs)["code_executions"] == []


def test_snapshot_proves_later_completion_without_rewriting_queued_response(tmp_path):
    args, _ = job_fixture(tmp_path, status="queued")
    original = canonical(args["hosts"])
    jobs = completed_model_evidence(**args)
    assert jobs[0]["completion_evidence_scope"] == "retained_phase_after_snapshot"
    assert not jobs[0]["completion_observed_in_this_tool_response"]
    assert canonical(args["hosts"]) == original


def test_completion_after_snapshot_does_not_leak_into_earlier_feedback(tmp_path):
    args, _ = job_fixture(tmp_path, status="queued")
    receipt = next(key for key in args["visible_files"] if key.endswith("/receipt.json"))
    args["visible_files"][receipt]["blake3"] = "f" * 64  # Earlier running receipt.
    assert completed_model_evidence(**args) == []


def test_os_exit_after_snapshot_is_not_imported(tmp_path):
    args, _ = job_fixture(tmp_path)
    args["visible_files"] = {key: value for key, value in args["visible_files"].items() if not key.endswith("/process-exit.json")}
    assert inference_conversation(completed_model_evidence(**args))["code_executions"] == []


@pytest.mark.parametrize("target", ["executed-helper.py", "artifacts/CASE/model-response.json", "case-outputs.jsonl"])
def test_changed_actual_evidence_is_error_not_missing_or_zero(tmp_path, target):
    args, root = job_fixture(tmp_path)
    (root / target).write_text("{}")
    with pytest.raises((EvaluationError, KeyError)):
        completed_model_evidence(**args)


def test_process_exit_must_bind_exact_executed_worker_and_receipt(tmp_path):
    args, root = job_fixture(tmp_path)
    value = json.loads((root / "process-exit.json").read_text())
    value["worker_receipt_blake3"] = "f" * 64
    put(root / "process-exit.json", value)
    with pytest.raises(EvaluationError, match="process_exit_binding"):
        completed_model_evidence(**args)


def test_smoke_requires_exact_actual_bytes_and_missing_calibration_is_not_complete(tmp_path):
    args, root = job_fixture(tmp_path)
    jobs = completed_model_evidence(**args)
    stage = tmp_path / "submission"
    stage.mkdir()
    facts = vqa_submission_evidence(stage, jobs)
    assert not facts["smoke_submitted_and_bound"] and facts["calibration_records"] == 0
    assert not facts["stage_completion_claimed"]
    put(stage / "smoke_forward.json", (root / "artifacts/smoke_forward.json").read_bytes())
    assert vqa_submission_evidence(stage, jobs)["smoke_submitted_and_bound"]
    put(stage / "smoke_forward.json", {"success": True})
    with pytest.raises(EvaluationError, match="not_actual_successful_job"):
        vqa_submission_evidence(stage, jobs)


def test_fifteen_calibration_records_need_fifteen_distinct_actual_decodes(tmp_path):
    stage = tmp_path / "submission"
    stage.mkdir()
    details = [{"case_id": f"CASE-{index}", "raw_output": {"raw_model_output": "Answer: A"}} for index in range(15)]
    jobs = [{"case_details": details}]
    records = [{"question_id": row["case_id"], "raw_model_output": row["raw_output"]["raw_model_output"]} for row in details]
    put(stage / "s3_calibration.json", records)
    facts = vqa_submission_evidence(stage, jobs)
    assert facts["calibration_actual_raw_decodes_bound"] and not facts["stage_completion_claimed"]
    put(stage / "s3_calibration.json", records[:1] * 15)
    assert not vqa_submission_evidence(stage, jobs)["calibration_actual_raw_decodes_bound"]
    put(stage / "s3_calibration.json", {"records": records})
    assert not vqa_submission_evidence(stage, jobs)["calibration_actual_raw_decodes_bound"]
