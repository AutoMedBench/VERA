"""Host-only, verified construction prefixes for original S4/S5 sandboxes.

Archived receipts are evidence, never installed as current execution receipts.
Only the unchanged successful predecessor submission is replayed by the original
bounded host tool. No private final grade, reference answer, or transcript is read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any
from collections.abc import Mapping

from eva_agent.pipeline.digests import blake3_bytes, canonical_value
from eva_agent.pipeline.tools import ParallelToolRuntime, ToolCall
from .teacher_batch import TeacherBatchError


def _require(condition, message):
    if not condition:
        raise TeacherBatchError(message)


def _bytes(root: Path, relative: str, *, maximum=2 * 1024 * 1024):
    _require(isinstance(relative, str), "predecessor path differs")
    parts = PurePosixPath(relative)
    _require(not parts.is_absolute() and parts.parts and not {".", ".."} & set(parts.parts),
             "predecessor path differs")
    path = root
    for part in parts.parts:
        path = path / part
        _require(not path.is_symlink(), "predecessor path is a symlink")
    _require(path.is_file() and path.stat().st_size <= maximum, "predecessor file unavailable or oversized")
    return path, path.read_bytes()


def _referenced(root, ref):
    _require(isinstance(ref, dict), "predecessor reference differs")
    path, payload = _bytes(root, ref.get("path"))
    _require(hashlib.sha256(payload).hexdigest() == ref.get("sha256"),
             "predecessor upstream bytes differ")
    _require("byte_count" not in ref or ref["byte_count"] == len(payload),
             "predecessor upstream length differs")
    return path, payload


@dataclass(frozen=True)
class ArchivedPredecessor:
    stage: str
    code: str = field(repr=False)
    provenance: dict[str, Any]


def load_archived_predecessors(binding, *, verify_host_receipt, trust_store):
    """Reopen the exact resolver-verified archive; return only earlier stages."""
    focus = binding.stage.value
    _require(focus in {"S4", "S5"}, "archived predecessor target differs")
    root = Path(binding.construction_root).resolve(strict=True)
    path = Path(binding.construction_manifest_path)
    _require(path.parent.resolve(strict=True) == root and not path.is_symlink(),
             "predecessor manifest location differs")
    payload = path.read_bytes()
    _require(blake3_bytes(payload) == binding.construction_manifest_blake3,
             "predecessor manifest differs from verified resolver")
    manifest = json.loads(payload)
    _require(manifest.get("sandbox_id") == binding.source_candidate_id
             and manifest.get("focus") == focus
             and manifest.get("production_construction_eligible") is True,
             "predecessor source identity differs")
    rollout_path, rollout_bytes = _referenced(root, manifest["artifacts"]["construction_solver_rollout_receipt"])
    rollout = json.loads(rollout_bytes)
    _require(rollout.get("sandbox_id") == binding.source_candidate_id
             and rollout.get("status") == "completed"
             and rollout.get("infrastructure_healthy") is True,
             "predecessor construction rollout differs")
    result = []
    for stage in (("S3",) if focus == "S4" else ("S3", "S4")):
        refs = rollout["stage_evidence"].get(f"{stage.lower()}_execution_attempts")
        _require(isinstance(refs, list) and 1 <= len(refs) <= 2,
                 "predecessor execution attempt inventory differs")
        successes = []
        for ref in refs:
            receipt_path, receipt_bytes = _referenced(rollout_path.parent, ref)
            receipt = json.loads(receipt_bytes)
            verified = verify_host_receipt(receipt, trust_store)
            _require(verified.get("sandbox_id") == binding.source_candidate_id
                     and verified.get("episode_id") == rollout.get("episode_id"),
                     "predecessor signed episode differs")
            observed = verified.get("observations", {})
            execution = observed.get("execution", {})
            _require(execution.get("stage") == stage, "predecessor execution stage differs")
            if execution.get("gate_passed") is not True:
                continue
            _require(execution.get("exit_code") == 0 and execution.get("process_started") is True,
                     "predecessor successful process evidence differs")
            code_path, code_bytes = _bytes(receipt_path.parent, "submission/solution.py")
            _require(hashlib.sha256(code_bytes).hexdigest()
                     == observed.get("contract", {}).get("submission_sha256"),
                     "predecessor code differs from signed execution")
            artifact = observed.get("artifact", {})
            _require(artifact.get("host_reopened") is True
                     and artifact.get("reopen_sha256_match") is True,
                     "predecessor artifact was not reopened")
            artifact_path, artifact_bytes = _bytes(receipt_path.parent / "workspace", artifact.get("expected_path"))
            _require(hashlib.sha256(artifact_bytes).hexdigest() == artifact.get("sha256")
                     and len(artifact_bytes) == artifact.get("byte_count"),
                     "predecessor artifact bytes differ")
            successes.append(ArchivedPredecessor(stage, code_bytes.decode("utf-8"), {
                "schema": "eva.archived-predecessor-code.v1", "stage": stage,
                "source_candidate_id": binding.source_candidate_id,
                "construction_manifest_blake3": binding.construction_manifest_blake3,
                "original_episode_id": rollout["episode_id"],
                "rollout_receipt_blake3": blake3_bytes(rollout_bytes),
                "signed_receipt_blake3": blake3_bytes(receipt_bytes),
                "code_blake3": blake3_bytes(code_bytes),
                "artifact_blake3": blake3_bytes(artifact_bytes),
                "code_path": str(code_path), "artifact_path": str(artifact_path),
                "historical_attempt_count": len(refs), "signature_verified": True,
                "new_execution_verified": False,
            }))
        _require(len(successes) == 1, "predecessor successful attempt is not unique")
        result.extend(successes)
    return tuple(result)


def replay_archived_predecessors(*, context, workspace, id_factory):
    """Call the original signed CPU execution handlers once per predecessor."""
    rows = context.archived_predecessors
    expected = ("S3",) if context.episode.stage.value == "S4" else ("S3", "S4")
    _require(tuple(row.stage for row in rows) == expected,
             "predecessor replay inventory differs")
    runtime = ParallelToolRuntime(workspace=workspace, registry=context.tool_registry,
                                  id_factory=id_factory, maximum_parallel_calls=1)
    before = workspace.snapshot("before-archived-predecessor-execution")
    results = []
    for row in rows:
        _require(blake3_bytes(row.code.encode()) == row.provenance["code_blake3"],
                 "predecessor in-memory code differs")
        observed = runtime.execute((ToolCall(call_id=id_factory.new("archived-predecessor-execute"),
            name="execute_code", arguments={"stage": row.stage, "code": row.code}),))
        _require(len(observed) == 1, "predecessor replay result count differs")
        result = observed[0]
        output = result.output
        # The canonical host gate covers the required checks. Advisory failures
        # remain in the retained output; replay must not promote them to gates.
        passed = (result.status == "completed" and result.error_code is None
                 and isinstance(output, Mapping)
                 and output.get("stage") == row.stage
                 and output.get("active_episode_id") == context.episode.policy_context["execution_binding"]["active_episode_id"]
                 and output.get("gate_passed") is True
                 and output.get("exit_code") == 0)
        if not passed:
            error = TeacherBatchError("predecessor replay host gate failed")
            selected = os.environ.get("EVA_GRPO_PREDECESSOR_MISMATCH_POLICY")
            if selected == "archived-predecessor-episode-mismatch-fresh-group-v1":
                # Private failure evidence only. This does not classify the
                # failure, change the host gate, or admit an actor/reward.
                error.predecessor_replay_failure = {
                    "schema": "eva.archived-predecessor-replay-failure.v1",
                    "predecessor_mismatch_policy": selected,
                    "focus": context.episode.stage.value,
                    "active_episode_id": context.episode.policy_context["execution_binding"]["active_episode_id"],
                    "workspace_root": str(workspace.root),
                    "predecessor": canonical_value(row.provenance),
                    "tool_result": canonical_value(result),
                }
            raise error
        results.append(result)
    after = workspace.snapshot("after-archived-predecessor-execution")
    public_artifacts = []
    if context.episode.stage.value == "S5":
        # S5 needs the actual S4 result without adding a new canonical read tool.
        # Preserve the existing S4 blind-execution boundary: never expose S3 here.
        relative = context.episode.policy_context["execution_binding"]["execution_stages"]["S4"]["artifact_relative_path"]
        payload = workspace.read_bytes(relative)
        _require(len(payload) <= 65536, "S5 public predecessor exceeds bounded context view")
        public_artifacts.append({"path": relative, "byte_count": len(payload),
            "content_blake3": blake3_bytes(payload), "content": payload.decode("utf-8"),
            "source": "current_workspace_after_real_host_S4_execution"})
    return {"schema": "eva.archived-predecessor-replay.v1",
        "focus": context.episode.stage.value,
        "current_episode_id": context.episode.policy_context["execution_binding"]["active_episode_id"],
        "provenance": [{k: v for k, v in row.provenance.items()
                        if k not in {"code_path", "artifact_path"}} for row in rows],
        "tool_results": canonical_value(results),
        "public_artifacts": public_artifacts,
        "workspace_before_blake3": before.tree_blake3,
        "workspace_after_blake3": after.tree_blake3,
        "unchanged_code_reexecuted": True, "archived_receipts_imported": False,
        "prefix_author": "host_replay_of_verified_historical_predecessor",
        "actor_tokens_generated": 0, "actor_reward_emitted": False, "provider_calls": 0}
