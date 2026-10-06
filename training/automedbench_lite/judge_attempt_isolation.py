"""Bounded fresh Judge attempts for a fixed post-round actor projection.

This is an opt-in MED evidence policy.  It never regenerates an actor rollout,
reuses a failed Judge output directory, filters a valid low score, or turns an
unavailable Judge into a zero.  Each additional Judge attempt reads byte-for-
byte copies of the same prepared rollout and preflight documents.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
from uuid import UUID, uuid4

from blake3 import blake3

from eva_agent.codex_runtime import codex_turn_receipt_from_document
from training.benchmark_feedback.automed_codex import read_json, write_private
from training.slime.judge_group_isolation import VERDICT_ERRORS
from .adapter import read_document, write_once


POLICY = "completed-invalid-judge-fresh-stage-attempt-v1"
MAX_REPLACEMENTS = 2
BINDING_NAME = "postround-judge-attempt-binding.json"


def replacement_limit(value) -> int:
    if isinstance(value, str) and value.isdecimal():
        value = int(value)
    if type(value) is not int or not 0 <= value <= MAX_REPLACEMENTS:
        raise ValueError("Post-round Judge verdict replacements must be an integer from 0 to 2")
    return value


def _commit(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    digest = blake3()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "blake3": digest.hexdigest()}


def _copy_exact(source: Path, destination: Path) -> None:
    source = Path(source)
    if source.is_symlink() or not source.is_file() or destination.exists():
        raise ValueError("Judge prepared source topology differs")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with source.open("rb") as incoming, os.fdopen(descriptor, "wb") as outgoing:
            descriptor = -1
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if _commit(source)["blake3"] != _commit(destination)["blake3"]:
        raise ValueError("Judge prepared source copy differs")


def _attempt_root(stage_root: Path, value: str) -> Path:
    if str(UUID(value)) != value:
        raise ValueError("Post-round Judge attempt identity differs")
    root = stage_root / "attempts" / value
    if root.is_symlink() or root.parent != stage_root / "attempts":
        raise ValueError("Post-round Judge attempt topology differs")
    return root


def completed_invalid_verdict(attempt_root: Path, error=None) -> dict:
    """Reopen only the exact completed-invalid-verdict class used in training."""
    from eva_agent.codex_pipeline import CodexPipelineError

    attempt_root = Path(attempt_root)
    preflight = read_json(attempt_root / "preflight.json")
    sample_id = preflight["sample_id"]
    judge_root = attempt_root / "judge" / sample_id
    failure_path = judge_root / "failure.json"
    receipt_path = judge_root / "judge-codex-failure-receipt.json"
    failure = read_json(failure_path)
    category = failure.get("pipeline_error")
    if (error is not None and (not isinstance(error, CodexPipelineError) or str(error) != category)):
        raise ValueError("not_a_completed_invalid_judge_verdict")
    if (failure.get("error_type") != "CodexPipelineError" or category not in VERDICT_ERRORS
            or failure.get("reward_emitted") is not False or failure.get("retry_count") != 0
            or failure.get("automatic_fallback") is not False or (judge_root / "grade.json").exists()
            or not receipt_path.is_file()):
        raise ValueError("not_a_completed_invalid_judge_verdict")
    receipt = codex_turn_receipt_from_document(read_json(receipt_path))
    if (receipt.status != "completed" or receipt.role.value != "judge"
            or receipt.visibility != "judge-only"
            or failure.get("codex_receipt_blake3") != receipt.receipt_blake3):
        raise ValueError("invalid_verdict_receipt_not_completed_or_bound")
    native_attempt = read_json(attempt_root / "judge-attempt.json")
    if (native_attempt.get("backend") != "native_astra"
            or native_attempt.get("semantic_attempt_count") != 1
            or native_attempt.get("retry_count") != 0):
        raise ValueError("invalid_verdict_native_attempt_binding_differs")
    return {"category": category, "failure": _commit(failure_path),
            "judge_receipt": _commit(receipt_path),
            "native_attempt": _commit(attempt_root / "judge-attempt.json")}


def run_stage_judge_attempts(stage_root: Path, prepared_root: Path, *, replacements: int,
                             judge, verify, native_turn_timeout_seconds: int = 240):
    """Accept the first valid grade, or re-raise the original terminal failure."""
    replacements = replacement_limit(replacements)
    if replacements == 0:
        raise ValueError("Post-round Judge attempt isolation requires a positive opt-in")
    stage_root, prepared_root = Path(stage_root), Path(prepared_root)
    source = {name: _commit(prepared_root / name) for name in ("rollout.json", "preflight.json")}
    preflight = read_json(prepared_root / "preflight.json")
    attempts_root = stage_root / "attempts"
    attempts_root.mkdir(mode=0o700, exist_ok=False)
    results = []
    for index in range(replacements + 1):
        attempt_id = str(uuid4())
        attempt_root = attempts_root / attempt_id
        attempt_root.mkdir(mode=0o700, exist_ok=False)
        for name in source:
            _copy_exact(prepared_root / name, attempt_root / name)
        request = write_once(attempt_root / "request.json", {
            "schema": "eva.postround-judge-attempt-request.v1", "policy": POLICY,
            "attempt_id": attempt_id, "attempt_index": index,
            "maximum_additional_judge_attempts": replacements,
            "track": preflight["case_id"], "stage": preflight["stage"],
            "prepared_source": source, "actor_rerun_by_policy": False,
            "additional_judge_attempt": index > 0})
        try:
            timeout = ({"native_turn_timeout_seconds": native_turn_timeout_seconds}
                       if native_turn_timeout_seconds != 240 else {})
            feedback = judge(attempt_root, **timeout)
            write_private(attempt_root / "feedback.json", feedback)
            verification = verify(attempt_root)
            if verification.get("valid") is not True or verification.get("status") != "scored":
                raise ValueError("Post-round Judge returned no independently verified grade")
            write_private(attempt_root / "verification.json", verification)
            result = write_once(attempt_root / "result.json", {
                "schema": "eva.postround-judge-attempt-result.v1", "policy": POLICY,
                "status": "accepted", "attempt_id": attempt_id, "attempt_index": index,
                "request": _commit(attempt_root / "request.json"),
                "feedback": _commit(attempt_root / "feedback.json"),
                "verification": _commit(attempt_root / "verification.json"),
                "valid_score_filtering": False, "actor_rerun_by_policy": False})
            results.append(_commit(attempt_root / "result.json"))
            policy_path = stage_root / "judge-attempt-isolation.json"
            write_once(policy_path, {"schema": "eva.postround-judge-attempt-isolation.v1",
                "policy": POLICY, "status": "accepted", "track": preflight["case_id"],
                "stage": preflight["stage"], "prepared_source": source,
                "maximum_additional_judge_attempts": replacements,
                "additional_judge_attempts_made": index, "judge_attempts_made": index + 1,
                "attempt_results": results, "accepted_attempt_root": str(attempt_root.resolve()),
                "accepted_result": _commit(attempt_root / "result.json"),
                "first_valid_grade_accepted": True, "valid_score_filtering": False,
                "actor_rerun_by_policy": False, "failed_attempts_rewritten": False})
            binding_path = attempt_root / BINDING_NAME
            write_once(binding_path, {"schema": "eva.postround-judge-attempt-binding.v1",
                "policy": POLICY, "track": preflight["case_id"], "stage": preflight["stage"],
                "accepted_attempt_root": str(attempt_root.resolve()),
                "isolation": _commit(policy_path)})
            return attempt_root, verification, binding_path
        except Exception as error:
            try:
                invalid = completed_invalid_verdict(attempt_root, error)
                eligible = True
            except (ValueError, OSError, KeyError, TypeError):
                invalid, eligible = None, False
            result = write_once(attempt_root / "result.json", {
                "schema": "eva.postround-judge-attempt-result.v1", "policy": POLICY,
                "status": "unavailable", "attempt_id": attempt_id, "attempt_index": index,
                "request": _commit(attempt_root / "request.json"),
                "error_type": type(error).__name__, "replacement_eligible": eligible,
                "completed_invalid_verdict": invalid, "score": None,
                "actor_rerun_by_policy": False})
            results.append(_commit(attempt_root / "result.json"))
            if eligible and index < replacements:
                continue
            write_once(stage_root / "judge-attempt-isolation.json", {
                "schema": "eva.postround-judge-attempt-isolation.v1", "policy": POLICY,
                "status": "unavailable", "track": preflight["case_id"], "stage": preflight["stage"],
                "prepared_source": source, "maximum_additional_judge_attempts": replacements,
                "additional_judge_attempts_made": index,
                "judge_attempts_made": index + 1, "attempt_results": results,
                "accepted_attempt_root": None, "accepted_result": None,
                "first_valid_grade_accepted": False, "valid_score_filtering": False,
                "actor_rerun_by_policy": False, "failed_attempts_rewritten": False,
                "replacement_limit_exhausted": eligible and index == replacements})
            raise


def verify_stage_judge_attempts(binding_path: Path, *, verify) -> dict:
    """Reopen the complete failure prefix and the first valid accepted grade."""
    binding_path = Path(binding_path)
    if binding_path.is_symlink():
        raise ValueError("Post-round Judge binding topology differs")
    binding_path = binding_path.resolve(strict=True)
    binding = read_document(binding_path)
    if binding.get("schema") != "eva.postround-judge-attempt-binding.v1" or binding.get("policy") != POLICY:
        raise ValueError("Post-round Judge binding differs")
    isolation_ref = binding["isolation"]
    if _commit(Path(isolation_ref["path"])) != isolation_ref:
        raise ValueError("Post-round Judge isolation commitment differs")
    policy = read_document(Path(isolation_ref["path"]))
    limit = replacement_limit(policy.get("maximum_additional_judge_attempts"))
    stage_root = Path(isolation_ref["path"]).resolve().parent
    prepared_root = stage_root / "prepared"
    source = {name: _commit(prepared_root / name) for name in ("rollout.json", "preflight.json")}
    preflight = read_json(prepared_root / "preflight.json")
    if (policy.get("schema") != "eva.postround-judge-attempt-isolation.v1"
            or policy.get("policy") != POLICY or policy.get("status") != "accepted"
            or not policy.get("track") == binding.get("track") == preflight.get("case_id")
            or not policy.get("stage") == binding.get("stage") == preflight.get("stage")
            or policy.get("prepared_source") != source
            or policy.get("actor_rerun_by_policy") is not False
            or policy.get("failed_attempts_rewritten") is not False
            or policy.get("valid_score_filtering") is not False
            or policy.get("first_valid_grade_accepted") is not True):
        raise ValueError("Post-round Judge isolation policy differs")
    refs = policy.get("attempt_results", [])
    if (not 1 <= len(refs) <= limit + 1 or policy.get("judge_attempts_made") != len(refs)
            or policy.get("additional_judge_attempts_made") != len(refs) - 1):
        raise ValueError("Post-round Judge attempt count differs")
    attempt_roots, results = [], []
    for index, ref in enumerate(refs):
        if _commit(Path(ref["path"])) != ref:
            raise ValueError("Post-round Judge result commitment differs")
        result_path = Path(ref["path"]).resolve()
        attempt_root = _attempt_root(stage_root, result_path.parent.name)
        if result_path != attempt_root / "result.json":
            raise ValueError("Post-round Judge result topology differs")
        result = read_document(result_path)
        request = read_document(attempt_root / "request.json")
        if (result.get("schema") != "eva.postround-judge-attempt-result.v1"
                or result.get("policy") != POLICY or result.get("attempt_id") != attempt_root.name
                or result.get("attempt_index") != index or result.get("actor_rerun_by_policy") is not False
                or result.get("request") != _commit(attempt_root / "request.json")
                or request.get("schema") != "eva.postround-judge-attempt-request.v1"
                or request.get("policy") != POLICY or request.get("attempt_id") != attempt_root.name
                or request.get("attempt_index") != index
                or request.get("maximum_additional_judge_attempts") != limit
                or request.get("track") != preflight.get("case_id")
                or request.get("stage") != preflight.get("stage")
                or request.get("additional_judge_attempt") is not (index > 0)
                or request.get("prepared_source") != source
                or request.get("actor_rerun_by_policy") is not False):
            raise ValueError("Post-round Judge request/result binding differs")
        for name in source:
            if (_commit(attempt_root / name)["blake3"] != source[name]["blake3"]
                    or (attempt_root / name).is_symlink()):
                raise ValueError("Post-round Judge immutable prepared bytes differ")
        attempt_roots.append(attempt_root)
        results.append(result)
    actual_roots = sorted(path.resolve() for path in (stage_root / "attempts").iterdir() if path.is_dir())
    if sorted(path.resolve() for path in attempt_roots) != actual_roots:
        raise ValueError("Post-round Judge unaccounted attempt directory")
    for root, result in zip(attempt_roots[:-1], results[:-1], strict=True):
        if (result.get("status") != "unavailable" or result.get("replacement_eligible") is not True
                or result.get("score") is not None
                or result.get("completed_invalid_verdict") != completed_invalid_verdict(root)):
            raise ValueError("Post-round Judge unavailable prefix differs")
    accepted_root, accepted = attempt_roots[-1], results[-1]
    if (accepted.get("status") != "accepted" or accepted.get("valid_score_filtering") is not False
            or policy.get("accepted_attempt_root") != str(accepted_root)
            or policy.get("accepted_result") != refs[-1]
            or binding.get("accepted_attempt_root") != str(accepted_root)
            or binding_path != accepted_root / BINDING_NAME):
        raise ValueError("Post-round Judge accepted attempt binding differs")
    for key in ("feedback", "verification"):
        if _commit(Path(accepted[key]["path"])) != accepted[key]:
            raise ValueError("Post-round Judge accepted evidence commitment differs")
    stored_feedback = read_json(accepted_root / "feedback.json")
    stored_verification = read_json(accepted_root / "verification.json")
    reopened = verify(accepted_root)
    for stored in (stored_feedback, stored_verification):
        if ({key: value for key, value in stored.items() if key != "verification_id"}
                != {key: value for key, value in reopened.items() if key != "verification_id"}):
            raise ValueError("Post-round Judge accepted verification differs")
    if reopened.get("valid") is not True or reopened.get("status") != "scored":
        raise ValueError("Post-round Judge accepted grade is not valid")
    return {"schema": "eva.verified-postround-judge-attempt-isolation.v1", "valid": True,
        "track": binding["track"], "stage": binding["stage"],
        "accepted_attempt_root": str(accepted_root), "judge_attempts_made": len(results),
        "additional_judge_attempts_made": len(results) - 1,
        "maximum_additional_judge_attempts": limit, "isolation": isolation_ref,
        "binding": _commit(binding_path), "score": reopened["score"]}
