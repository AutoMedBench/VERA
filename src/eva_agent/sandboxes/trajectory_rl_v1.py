"""Lean, deterministic rubric/evidence wrappers for RL sandbox starts.

Construction is deliberately broader than SFT selection or admission. A row
needs an independently reopenable starting workspace, evidence bytes,
benchmark provenance, and one exact compiled domain-stage rubric. Rollout
success, cascade completion, ability separation, judge output, and minimum
reward are not construction gates.

Only one *new* digest is created: ``wrapper_blake3`` streams over the contract
and every referenced byte. Existing signed trajectory, run, registry, rubric,
item, and selector digests are retained as authority references; no redundant
per-file or nested receipt hashes are minted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from uuid import UUID

from blake3 import blake3

from eva_agent.pipeline.digests import (
    blake3_hex,
    canonical_json_bytes,
    canonical_value,
    is_blake3,
)
from eva_agent.rubrics.registry import CompiledRubricRegistry


SCHEMA = "eva.trajectory-derived-rl-sandbox.v1"
METHOD = "trajectory-or-benchmark-stage-start-rubric-evidence-wrap-v1"
COMMITMENT_METHOD = "blake3-framed-wrapper-root-v1"
STAGES = frozenset({"S1", "S2", "S3", "S4", "S5", "E2E"})
FALLBACK_REASONS = frozenset(
    {
        "benchmark_source_selected",
        "launcher_defect",
        "trajectory_unverifiable",
        "trajectory_state_missing",
    }
)
_BASE_PROOFS = (
    "benchmark_provenance_committed",
    "starting_workspace_reopenable",
    "evidence_manifest_reopenable",
    "domain_stage_rubric_bound",
    "reward_calculation_bound",
)
_NON_GATES = (
    "source_rollout_success",
    "cascade_completion",
    "ability_separation",
    "agent_judge_assessment",
    "minimum_reward",
)


class TrajectoryRLSandboxError(ValueError):
    """A trajectory-derived sandbox cannot be independently reopened."""


def _plain(value: Any) -> Any:
    try:
        return canonical_value(value)
    except (TypeError, ValueError) as exc:
        raise TrajectoryRLSandboxError("value is not canonical JSON") from exc


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrajectoryRLSandboxError(f"{label} must be non-empty text")
    return value


def _uuid(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise TrajectoryRLSandboxError(f"{label} must be a canonical UUID")
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError):
        raise TrajectoryRLSandboxError(f"{label} must be a canonical UUID") from None
    if str(parsed) != value:
        raise TrajectoryRLSandboxError(f"{label} must be a canonical UUID")
    return value


def _relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise TrajectoryRLSandboxError(f"{label} must be a safe POSIX-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise TrajectoryRLSandboxError(f"{label} must be a safe POSIX-relative path")
    return value


def _file_rows(
    files: Mapping[str, bytes],
    *,
    label: str,
    require_nonempty: bool,
) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(files, Mapping) or (require_nonempty and not files):
        qualifier = "non-empty " if require_nonempty else ""
        raise TrajectoryRLSandboxError(f"{label} must be a {qualifier}file mapping")
    rows: list[dict[str, Any]] = []
    total = 0
    for raw_path, payload in files.items():
        path = _relative_path(raw_path, label=f"{label} path")
        if not isinstance(payload, bytes):
            raise TrajectoryRLSandboxError(f"{label} values must be literal bytes")
        rows.append({"path": path, "byte_count": len(payload)})
        total += len(payload)
    rows.sort(key=lambda row: row["path"])
    return rows, total


def _material_document(
    files: Mapping[str, bytes],
    *,
    schema: str,
    label: str,
    noun: str,
    count_name: str,
    require_nonempty: bool,
) -> dict[str, Any]:
    rows, byte_count = _file_rows(files, label=label, require_nonempty=require_nonempty)
    return {
        "schema": schema,
        noun: rows,
        count_name: len(rows),
        "byte_count": byte_count,
        "commitment": {
            "method": COMMITMENT_METHOD,
            "root_field": "wrapper_blake3",
            "namespace": label.replace(" ", "_"),
        },
    }


def _trajectory_events(document: Mapping[str, Any]) -> list[Any]:
    plain = _plain(document)
    if not isinstance(plain, dict):
        raise TrajectoryRLSandboxError("trajectory document must be an object")
    candidates = [plain[key] for key in ("policy_events", "events") if key in plain]
    if len(candidates) != 1 or not isinstance(candidates[0], list) or not candidates[0]:
        raise TrajectoryRLSandboxError(
            "trajectory must expose exactly one non-empty events or policy_events array"
        )
    return candidates[0]


def _verify_existing_document_reference(
    document: Mapping[str, Any], reference: str, *, label: str
) -> dict[str, Any]:
    plain = _plain(document)
    if not isinstance(plain, dict) or not is_blake3(reference):
        raise TrajectoryRLSandboxError(f"{label} document or existing BLAKE3 differs")
    # Existing EVA documents use either a whole-document digest or one
    # self-commitment field excluded from its own digest domain. Reopen only
    # these already-defined source conventions; never mint a replacement.
    if blake3_hex(plain) != reference:
        candidates = [
            field
            for field in (
                "document_blake3",
                "result_blake3",
                "receipt_blake3",
                "trace_blake3",
                "bundle_blake3",
            )
            if plain.get(field) == reference
        ]
        if len(candidates) != 1:
            raise TrajectoryRLSandboxError(f"{label} existing BLAKE3 reference differs")
        field = candidates[0]
        if blake3_hex({key: value for key, value in plain.items() if key != field}) != reference:
            raise TrajectoryRLSandboxError(f"{label} existing BLAKE3 reference differs")
    return plain


@dataclass(frozen=True, slots=True)
class TrajectoryStepStart:
    """A start immediately before one independently reopenable rollout step."""

    trajectory_id: str
    trajectory_blake3: str
    run_id: str
    run_receipt_blake3: str
    step_id: str
    event_ordinal: int
    producer_model_id: str
    trajectory_document: Mapping[str, Any]
    run_receipt_document: Mapping[str, Any]
    verification_status: str = "independently_reopenable"


@dataclass(frozen=True, slots=True)
class BenchmarkSourceStart:
    """The benchmark-native initial state, optionally excluding a bad run."""

    fallback_reason: str = "benchmark_source_selected"
    excluded_trajectory_id: str | None = None
    excluded_run_id: str | None = None
    excluded_run_receipt_blake3: str | None = None


@dataclass(frozen=True, slots=True)
class TrajectoryRLSandboxVerification:
    valid: bool
    sandbox_id: str
    wrapper_blake3: str
    checks: tuple[str, ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": "eva.trajectory-derived-rl-sandbox-verification.v1",
            "valid": self.valid,
            "sandbox_id": self.sandbox_id,
            "wrapper_blake3": self.wrapper_blake3,
            "checks": list(self.checks),
        }


def _trajectory_start(start: TrajectoryStepStart) -> dict[str, Any]:
    if start.verification_status != "independently_reopenable":
        raise TrajectoryRLSandboxError(
            "trajectory step is not independently reopenable; use benchmark-source fallback"
        )
    trajectory = _verify_existing_document_reference(
        start.trajectory_document,
        start.trajectory_blake3,
        label="trajectory",
    )
    receipt = _verify_existing_document_reference(
        start.run_receipt_document,
        start.run_receipt_blake3,
        label="run receipt",
    )
    trajectory_id = _uuid(start.trajectory_id, label="trajectory_id")
    run_id = _uuid(start.run_id, label="run_id")
    step_id = _uuid(start.step_id, label="step_id")
    if type(start.event_ordinal) is not int or start.event_ordinal < 0:
        raise TrajectoryRLSandboxError("event_ordinal must be a non-negative integer")
    events = _trajectory_events(trajectory)
    if start.event_ordinal >= len(events):
        raise TrajectoryRLSandboxError("event_ordinal lies outside the full trajectory")
    event = events[start.event_ordinal]
    if not isinstance(event, dict) or event.get("event_id") != step_id:
        raise TrajectoryRLSandboxError("selected trajectory step identity differs")
    return {
        "origin": "trajectory_step",
        "position": "before_step",
        "trajectory": {
            "trajectory_id": trajectory_id,
            "trajectory_blake3": start.trajectory_blake3,
            "run_id": run_id,
            "run_receipt_blake3": start.run_receipt_blake3,
            "step_id": step_id,
            "event_ordinal": start.event_ordinal,
            "event_count": len(events),
            "producer_model_id": _text(
                start.producer_model_id, label="producer_model_id"
            ),
            "verification_status": start.verification_status,
        },
        "fallback": None,
    }


def _benchmark_start(start: BenchmarkSourceStart) -> dict[str, Any]:
    if start.fallback_reason not in FALLBACK_REASONS:
        raise TrajectoryRLSandboxError("benchmark fallback reason differs")
    excluded = (
        start.excluded_trajectory_id,
        start.excluded_run_id,
        start.excluded_run_receipt_blake3,
    )
    if start.fallback_reason == "benchmark_source_selected":
        if any(value is not None for value in excluded):
            raise TrajectoryRLSandboxError("direct benchmark start cannot claim an excluded run")
        exclusion = None
    else:
        if any(value is None for value in excluded):
            raise TrajectoryRLSandboxError("trajectory fallback must identify the excluded run")
        if not is_blake3(start.excluded_run_receipt_blake3):
            raise TrajectoryRLSandboxError("excluded run receipt BLAKE3 differs")
        exclusion = {
            "trajectory_id": _uuid(
                start.excluded_trajectory_id, label="excluded_trajectory_id"
            ),
            "run_id": _uuid(start.excluded_run_id, label="excluded_run_id"),
            "run_receipt_blake3": start.excluded_run_receipt_blake3,
            "disposition": "excluded_from_training_evidence",
        }
    return {
        "origin": "benchmark_source",
        "position": "benchmark_initial",
        "trajectory": None,
        "fallback": {"reason": start.fallback_reason, "excluded_run": exclusion},
    }


def _reward_contract(registry: CompiledRubricRegistry, rubric: Any) -> dict[str, Any]:
    return {
        "schema": "eva.rubric-reward-calculation-contract.v1",
        "method": "compiled-rubric-score-v1",
        "registry_digest": registry.digest,
        "rubric_id": rubric.rubric_id,
        "rubric_digest": rubric.digest,
        "parameters_from": "rubric_table.items",
        "item_score_domain": "exact_compiled_partial_credit_levels_bps",
        "weighted_score": "nearest_integer_basis_points_using_compiled_weights",
        "hard_gate_behavior": "reward_bps_zero_when_any_compiled_hard_gate_fails",
        "score_receipt_schema": "eva.rubric-score.v1",
    }


def _framed_update(hasher: Any, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    hasher.update(len(label_bytes).to_bytes(8, "big"))
    hasher.update(label_bytes)
    hasher.update(len(payload).to_bytes(16, "big"))
    hasher.update(payload)


def _wrapper_root(
    core: Mapping[str, Any],
    *,
    initial_workspace_files: Mapping[str, bytes],
    evidence_files: Mapping[str, bytes],
) -> str:
    hasher = blake3()
    _framed_update(hasher, "contract", canonical_json_bytes(core))
    for namespace, files in (
        ("workspace_initial", initial_workspace_files),
        ("evidence", evidence_files),
    ):
        for path in sorted(files):
            _framed_update(hasher, f"{namespace}/{path}", files[path])
    return hasher.hexdigest()


def build_trajectory_rl_sandbox(
    *,
    sandbox_id: str,
    binding_id: str,
    benchmark: str,
    source_file: str,
    source_revision: str,
    source_record_id: str,
    domain: str,
    stage: str,
    instruction: str,
    start: TrajectoryStepStart | BenchmarkSourceStart,
    initial_workspace_files: Mapping[str, bytes],
    evidence_files: Mapping[str, bytes],
    rubric_registry: CompiledRubricRegistry,
) -> dict[str, Any]:
    """Build one deterministic train sandbox without quality/admission gates."""

    sandbox = _uuid(sandbox_id, label="sandbox_id")
    binding = _uuid(binding_id, label="binding_id")
    domain_text = _text(domain, label="domain")
    if stage not in STAGES:
        raise TrajectoryRLSandboxError("stage must be S1-S5 or E2E")
    if not isinstance(rubric_registry, CompiledRubricRegistry):
        raise TrajectoryRLSandboxError("compiled rubric registry type differs")
    rubric = rubric_registry.resolve(domain_text, stage)
    if not 5 <= len(rubric.items) <= 10:
        raise TrajectoryRLSandboxError("rubric must contain five to ten atomic items")
    workspace = _material_document(
        initial_workspace_files,
        schema="eva.reopenable-workspace-state.v1",
        label="workspace initial",
        noun="files",
        count_name="file_count",
        require_nonempty=False,
    )
    evidence = _material_document(
        evidence_files,
        schema="eva.rl-sandbox-evidence-manifest.v1",
        label="evidence",
        noun="artifacts",
        count_name="artifact_count",
        require_nonempty=True,
    )
    if isinstance(start, TrajectoryStepStart):
        start_document = _trajectory_start(start)
        required = (*_BASE_PROOFS, "trajectory_run_step_reopenable")
    elif isinstance(start, BenchmarkSourceStart):
        start_document = _benchmark_start(start)
        required = (*_BASE_PROOFS, "benchmark_initial_state_reopenable")
    else:
        raise TrajectoryRLSandboxError("starting point type differs")
    rubric_binding = rubric_registry.bind_sandbox(
        sandbox, domain_text, stage, binding_id=binding
    ).to_document()
    core = {
        "schema": SCHEMA,
        "construction_method": METHOD,
        "commitment_method": COMMITMENT_METHOD,
        "sandbox_id": sandbox,
        "split": "train",
        "domain": domain_text,
        "stage": stage,
        "instruction": _text(instruction, label="instruction"),
        "benchmark_provenance": {
            "benchmark": _text(benchmark, label="benchmark"),
            "source_file": _relative_path(source_file, label="benchmark source_file"),
            "source_revision": _text(source_revision, label="source_revision"),
            "source_record_id": _text(source_record_id, label="source_record_id"),
        },
        "starting_point": {**start_document, "workspace_initial_state": workspace},
        "evidence_manifest": evidence,
        "rubric_binding": rubric_binding,
        "rubric_table": rubric.to_document(),
        "reward_calculation_contract": _reward_contract(rubric_registry, rubric),
        "construction_policy": {
            "required_proofs": list(required),
            "explicitly_not_required": list(_NON_GATES),
            "quality_selection_deferred": True,
            "admission_claimed": False,
        },
    }
    return {
        **core,
        "wrapper_blake3": _wrapper_root(
            core,
            initial_workspace_files=initial_workspace_files,
            evidence_files=evidence_files,
        ),
    }


def _start_from_document(
    document: Mapping[str, Any],
    *,
    trajectory_document: Mapping[str, Any] | None,
    run_receipt_document: Mapping[str, Any] | None,
) -> TrajectoryStepStart | BenchmarkSourceStart:
    point = document.get("starting_point")
    if not isinstance(point, Mapping):
        raise TrajectoryRLSandboxError("starting point differs")
    if point.get("origin") == "trajectory_step":
        trajectory = point.get("trajectory")
        if (
            not isinstance(trajectory, Mapping)
            or trajectory_document is None
            or run_receipt_document is None
        ):
            raise TrajectoryRLSandboxError("trajectory start materials are incomplete")
        return TrajectoryStepStart(
            trajectory_id=trajectory.get("trajectory_id"),
            trajectory_blake3=trajectory.get("trajectory_blake3"),
            run_id=trajectory.get("run_id"),
            run_receipt_blake3=trajectory.get("run_receipt_blake3"),
            step_id=trajectory.get("step_id"),
            event_ordinal=trajectory.get("event_ordinal"),
            producer_model_id=trajectory.get("producer_model_id"),
            trajectory_document=trajectory_document,
            run_receipt_document=run_receipt_document,
            verification_status=trajectory.get("verification_status"),
        )
    if point.get("origin") == "benchmark_source":
        if trajectory_document is not None or run_receipt_document is not None:
            raise TrajectoryRLSandboxError("benchmark start must not consume trajectory materials")
        fallback = point.get("fallback")
        if not isinstance(fallback, Mapping):
            raise TrajectoryRLSandboxError("benchmark fallback differs")
        excluded = fallback.get("excluded_run")
        if excluded is not None and not isinstance(excluded, Mapping):
            raise TrajectoryRLSandboxError("excluded run provenance differs")
        return BenchmarkSourceStart(
            fallback_reason=fallback.get("reason"),
            excluded_trajectory_id=None if excluded is None else excluded.get("trajectory_id"),
            excluded_run_id=None if excluded is None else excluded.get("run_id"),
            excluded_run_receipt_blake3=(
                None if excluded is None else excluded.get("run_receipt_blake3")
            ),
        )
    raise TrajectoryRLSandboxError("starting point origin differs")


def verify_trajectory_rl_sandbox(
    document: Mapping[str, Any],
    *,
    initial_workspace_files: Mapping[str, bytes],
    evidence_files: Mapping[str, bytes],
    rubric_registry: CompiledRubricRegistry,
    trajectory_document: Mapping[str, Any] | None = None,
    run_receipt_document: Mapping[str, Any] | None = None,
) -> TrajectoryRLSandboxVerification:
    """Independently rebuild and compare the contract and its one root."""

    plain = _plain(document)
    if not isinstance(plain, dict) or plain.get("schema") != SCHEMA:
        raise TrajectoryRLSandboxError("RL sandbox document schema differs")
    provenance = plain.get("benchmark_provenance")
    binding = plain.get("rubric_binding")
    if not isinstance(provenance, dict) or not isinstance(binding, dict):
        raise TrajectoryRLSandboxError("benchmark or rubric binding differs")
    start = _start_from_document(
        plain,
        trajectory_document=trajectory_document,
        run_receipt_document=run_receipt_document,
    )
    expected = build_trajectory_rl_sandbox(
        sandbox_id=plain.get("sandbox_id"),
        binding_id=binding.get("binding_id"),
        benchmark=provenance.get("benchmark"),
        source_file=provenance.get("source_file"),
        source_revision=provenance.get("source_revision"),
        source_record_id=provenance.get("source_record_id"),
        domain=plain.get("domain"),
        stage=plain.get("stage"),
        instruction=plain.get("instruction"),
        start=start,
        initial_workspace_files=initial_workspace_files,
        evidence_files=evidence_files,
        rubric_registry=rubric_registry,
    )
    if plain != expected:
        raise TrajectoryRLSandboxError("RL sandbox bytes or contract differ")
    checks = (
        "single_wrapper_blake3",
        "benchmark_provenance",
        "starting_point_provenance",
        "workspace_initial_state",
        "evidence_manifest",
        "rubric_registry_binding",
        "rubric_table_5_to_10_items",
        "reward_calculation_contract",
        "construction_non_gates",
    )
    return TrajectoryRLSandboxVerification(
        valid=True,
        sandbox_id=plain["sandbox_id"],
        wrapper_blake3=plain["wrapper_blake3"],
        checks=checks,
    )


def read_regular_file_tree(root: Path) -> dict[str, bytes]:
    """Read one tree without following symlinks for offline wrapping."""

    if not isinstance(root, Path) or not root.exists() or root.is_symlink() or not root.is_dir():
        raise TrajectoryRLSandboxError("file tree root must be a real directory")
    values: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise TrajectoryRLSandboxError("file tree must not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise TrajectoryRLSandboxError("file tree contains a non-regular entry")
        relative = path.relative_to(root).as_posix()
        _relative_path(relative, label="file tree path")
        values[relative] = path.read_bytes()
    return values


__all__ = [
    "BenchmarkSourceStart",
    "COMMITMENT_METHOD",
    "FALLBACK_REASONS",
    "METHOD",
    "SCHEMA",
    "TrajectoryRLSandboxError",
    "TrajectoryRLSandboxVerification",
    "TrajectoryStepStart",
    "build_trajectory_rl_sandbox",
    "read_regular_file_tree",
    "verify_trajectory_rl_sandbox",
]
