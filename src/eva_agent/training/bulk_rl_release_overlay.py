"""Signed split overlay for the immutable 6,000-record bulk RL corpus.

The underlying sandbox shards remain byte-for-byte immutable.  This module
derives the release split from the outcome-blind v3 campaign selection and
binds the fixed five-model ability cascade used by release evaluation.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from eva_agent.admission.receipts import (
    AdmissionReceiptError,
    SignedEnvelope,
    verify_signed_envelope,
)
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value


OVERLAY_ROW_SCHEMA = "eva.bulk-rl-release-overlay-row.v1"
OVERLAY_DATASET_SCHEMA = "eva.bulk-rl-release-overlay-dataset.v1"
CASCADE_SCHEMA = "eva.fixed-ability-separating-evaluator-cascade.v1"
SPLITS = ("train", "development", "sealed_evaluation")
STAGES = ("S1", "S2", "S3", "S4", "S5", "E2E")
EXPECTED_SANDBOXES = 6_000
EXPECTED_SCHEDULED = 9_000
EXPECTED_SPLITS = {"train": 5_000, "development": 500, "sealed_evaluation": 500}
EXPECTED_STAGE_COUNTS = {stage: 1_000 for stage in STAGES}


class BulkRLReleaseOverlayError(ValueError):
    """A release overlay differs from its immutable authorities."""


def _plain(value: Any) -> Any:
    return canonical_value(value)


def _hex_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _blake3_digest(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("blake3:") and _hex_digest(value[7:])


def fixed_evaluator_recipe(model_registry: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve the exact signed-registry cascade without provider access."""

    if model_registry.get("schema") != "rlevo.med-research-model-registry.v1":
        raise BulkRLReleaseOverlayError("model registry schema differs")
    models = model_registry.get("models")
    order = model_registry.get("cascade_order")
    if not isinstance(models, list) or not isinstance(order, list):
        raise BulkRLReleaseOverlayError("model registry cascade shape differs")
    by_role: dict[str, Mapping[str, Any]] = {}
    for row in models:
        if not isinstance(row, Mapping):
            raise BulkRLReleaseOverlayError("model registry row differs")
        role = row.get("role_id")
        if not isinstance(role, str) or role in by_role:
            raise BulkRLReleaseOverlayError("model registry role differs")
        by_role[role] = row
    if (
        len(order) != 5
        or len(set(order)) != 5
        or any(not isinstance(role, str) or role not in by_role for role in order)
    ):
        raise BulkRLReleaseOverlayError("fixed cascade inventory differs")
    judge = by_role.get("architect_opus_5")
    if not isinstance(judge, Mapping) or "opus-5" not in str(judge.get("api_model_id", "")):
        raise BulkRLReleaseOverlayError("fixed Agent Judge identity differs")
    cascade_models: list[dict[str, str]] = []
    for ordinal, role in enumerate(order):
        row = by_role[role]
        model_id = row.get("api_model_id")
        family = row.get("provider_family")
        if not isinstance(model_id, str) or not model_id or not isinstance(family, str):
            raise BulkRLReleaseOverlayError("fixed cascade model identity differs")
        cascade_models.append(
            {
                "ordinal": ordinal,
                "role_id": role,
                "model_id": model_id,
                "provider_family": family,
            }
        )
    core = {
        "schema": CASCADE_SCHEMA,
        "cohort_order": list(order),
        "models": cascade_models,
        "rollouts_per_sandbox": len(cascade_models),
        "all_model_scores_retained": True,
        "ability_separation_computed_from_fixed_scores": True,
        "cascade_definition_is_release_fixed": True,
        "agent_judge": {
            "role_id": str(judge["role_id"]),
            "model_id": str(judge["api_model_id"]),
            "provider_family": str(judge["provider_family"]),
            "workspace_inspection_required": True,
        },
    }
    return MappingProxyType({**core, "recipe_blake3": blake3_hex(core)})


def _primary_selection(selection: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if (
        selection.get("schema") != "eva.medresearch-campaign-selection.v3"
        or selection.get("selected_count") != EXPECTED_SANDBOXES
        or selection.get("scheduled_count") != EXPECTED_SCHEDULED
        or not _hex_digest(selection.get("selection_blake3"))
        or not _hex_digest(selection.get("plan_blake3"))
        or not _hex_digest(selection.get("base_selection_blake3"))
    ):
        raise BulkRLReleaseOverlayError("split selection authority differs")
    entries = selection.get("entries")
    if not isinstance(entries, list) or len(entries) != EXPECTED_SCHEDULED:
        raise BulkRLReleaseOverlayError("split selection inventory differs")
    primary = tuple(
        row for row in entries
        if isinstance(row, Mapping) and row.get("selection_tier") == "primary"
    )
    if len(primary) != EXPECTED_SANDBOXES:
        raise BulkRLReleaseOverlayError("split selection primary inventory differs")
    counts = Counter(row.get("split") for row in primary)
    if dict(counts) != EXPECTED_SPLITS:
        raise BulkRLReleaseOverlayError("split selection quotas differ")
    if Counter(row.get("stage") for row in primary) != Counter(EXPECTED_STAGE_COUNTS):
        raise BulkRLReleaseOverlayError("split selection stage balance differs")
    return primary


def derive_overlay_rows(
    base_records: Sequence[Mapping[str, Any]],
    *,
    base_manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    execution_catalog: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Join immutable bulk records to the verified split and execution rows."""

    primary = _primary_selection(selection)
    if (
        base_manifest.get("schema") != "eva.bulk-rl-sandbox-sharded-dataset.v1"
        or base_manifest.get("sandbox_count") != EXPECTED_SANDBOXES
        or base_manifest.get("split_counts") != {"train": EXPECTED_SANDBOXES}
        or base_manifest.get("selection_blake3") != selection.get("base_selection_blake3")
        or not _hex_digest(base_manifest.get("dataset_blake3"))
        or not _hex_digest(base_manifest.get("catalog_blake3"))
    ):
        raise BulkRLReleaseOverlayError("base bulk authority differs")
    if (
        execution_catalog.get("schema") != "eva.prospective-execution-binding-catalog.v2"
        or execution_catalog.get("selection_blake3") != selection.get("selection_blake3")
        or execution_catalog.get("primary_split_counts") != EXPECTED_SPLITS
        or execution_catalog.get("primary_count") != EXPECTED_SANDBOXES
        or execution_catalog.get("scheduled_count") != EXPECTED_SCHEDULED
        or not _hex_digest(execution_catalog.get("catalog_blake3"))
    ):
        raise BulkRLReleaseOverlayError("execution catalog authority differs")
    execution_rows = execution_catalog.get("rows")
    if (
        not isinstance(execution_rows, Sequence)
        or isinstance(execution_rows, (str, bytes))
        or len(execution_rows) != EXPECTED_SCHEDULED
    ):
        raise BulkRLReleaseOverlayError("execution catalog inventory differs")
    selected_by_id = {str(row["candidate_id"]): row for row in primary}
    execution_by_id = {
        str(row["candidate_id"]): row
        for row in execution_rows
        if isinstance(row, Mapping)
    }
    if (
        len(selected_by_id) != EXPECTED_SANDBOXES
        or len(execution_by_id) != EXPECTED_SCHEDULED
    ):
        raise BulkRLReleaseOverlayError("release authority identity is duplicated")
    if len(base_records) != EXPECTED_SANDBOXES:
        raise BulkRLReleaseOverlayError("base bulk record count differs")
    rows: list[Mapping[str, Any]] = []
    observed_candidates: set[str] = set()
    observed_sandboxes: set[str] = set()
    for release_ordinal, base in enumerate(base_records, start=1):
        if not isinstance(base, Mapping) or base.get("schema") != "eva.bulk-rl-sandbox.v1":
            raise BulkRLReleaseOverlayError("base sandbox record differs")
        candidate_id = base.get("candidate_id")
        sandbox_id = base.get("sandbox_id")
        record_blake3 = base.get("record_blake3")
        if (
            not isinstance(candidate_id, str)
            or not isinstance(sandbox_id, str)
            or not _hex_digest(record_blake3)
            or candidate_id in observed_candidates
            or sandbox_id in observed_sandboxes
        ):
            raise BulkRLReleaseOverlayError("base sandbox identity differs")
        selected = selected_by_id.get(candidate_id)
        executed = execution_by_id.get(candidate_id)
        reward = base.get("reward_contract")
        source = base.get("source_binding")
        if not isinstance(selected, Mapping) or not isinstance(executed, Mapping):
            raise BulkRLReleaseOverlayError("base sandbox is absent from release authority")
        if not isinstance(reward, Mapping) or not isinstance(source, Mapping):
            raise BulkRLReleaseOverlayError("base sandbox binding differs")
        immutable = (
            base.get("queue_ordinal"), base.get("source_family"), base.get("domain"),
            base.get("stage"), reward.get("rubric_id"), reward.get("rubric_digest"),
            source.get("source_candidate_id"),
        )
        selected_values = (
            selected.get("queue_ordinal"), selected.get("source_family"), selected.get("domain"),
            selected.get("stage"), selected.get("rubric_id"), selected.get("rubric_blake3"),
            selected.get("source_candidate_id"),
        )
        executed_values = (
            executed.get("queue_ordinal"), executed.get("source_family"), executed.get("domain"),
            executed.get("stage"), executed.get("rubric_id"), executed.get("rubric_blake3"),
            executed.get("source_candidate_id"),
        )
        if immutable != selected_values or immutable != executed_values:
            raise BulkRLReleaseOverlayError("release authority changed an immutable sandbox field")
        if not _blake3_digest(reward.get("rubric_digest")):
            raise BulkRLReleaseOverlayError("base sandbox rubric commitment differs")
        split = selected.get("split")
        if split not in SPLITS or executed.get("split") != split:
            raise BulkRLReleaseOverlayError("release split authorities disagree")
        status = executed.get("execution_status")
        if status not in {"executable_legacy", "source_only", "rejected"}:
            raise BulkRLReleaseOverlayError("release execution status differs")
        proof = executed.get("execution_proof")
        proof_blake3 = proof.get("execution_proof_blake3") if isinstance(proof, Mapping) else None
        row = {
            "schema": OVERLAY_ROW_SCHEMA,
            "release_ordinal": release_ordinal,
            "sandbox_id": sandbox_id,
            "candidate_id": candidate_id,
            "base_record_blake3": record_blake3,
            "base_queue_ordinal": base.get("queue_ordinal"),
            "release_split": split,
            "source_family": base.get("source_family"),
            "domain": base.get("domain"),
            "stage": base.get("stage"),
            "rubric_id": reward.get("rubric_id"),
            "rubric_blake3": reward.get("rubric_digest"),
            "execution_status": status,
            "execution_row_blake3": executed.get("row_blake3"),
            "execution_proof_blake3": proof_blake3,
            "immutable_failure": status == "rejected",
        }
        if not _hex_digest(row["execution_row_blake3"]):
            raise BulkRLReleaseOverlayError("execution row commitment differs")
        if proof_blake3 is not None and not _hex_digest(proof_blake3):
            raise BulkRLReleaseOverlayError("execution proof commitment differs")
        rows.append(MappingProxyType(row))
        observed_candidates.add(candidate_id)
        observed_sandboxes.add(sandbox_id)
    if observed_candidates != set(selected_by_id):
        raise BulkRLReleaseOverlayError("release overlay changed the primary candidate set")
    return tuple(rows)


def _nested_counts(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, dict[str, int]]:
    counts = Counter((str(row[key]), str(row["release_split"])) for row in rows)
    names = sorted({name for name, _split in counts})
    return {
        name: {split: counts[(name, split)] for split in SPLITS}
        for name in names
    }


def release_manifest_core(
    rows: Sequence[Mapping[str, Any]],
    *,
    overlay_path: str,
    overlay_byte_count: int,
    overlay_blake3: str,
    base_manifest: Mapping[str, Any],
    base_manifest_envelope_blake3: str,
    selection: Mapping[str, Any],
    execution_catalog: Mapping[str, Any],
    execution_catalog_envelope_blake3: str,
    model_registry_file_blake3: str,
    evaluator_recipe: Mapping[str, Any],
) -> dict[str, Any]:
    if len(rows) != EXPECTED_SANDBOXES or not _hex_digest(overlay_blake3):
        raise BulkRLReleaseOverlayError("release overlay payload differs")
    split_counts = dict(Counter(str(row["release_split"]) for row in rows))
    stage_counts = dict(Counter(str(row["stage"]) for row in rows))
    if split_counts != EXPECTED_SPLITS or stage_counts != EXPECTED_STAGE_COUNTS:
        raise BulkRLReleaseOverlayError("release overlay balance differs")
    if not all(
        _hex_digest(value)
        for value in (
            base_manifest_envelope_blake3,
            execution_catalog_envelope_blake3,
            model_registry_file_blake3,
        )
    ):
        raise BulkRLReleaseOverlayError("release authority commitment differs")
    core = {
        "schema": OVERLAY_DATASET_SCHEMA,
        "release_method": "immutable-bulk-v1_plus-outcome-blind-v3-split-overlay-v1",
        "sandbox_count": EXPECTED_SANDBOXES,
        "split_counts": EXPECTED_SPLITS,
        "stage_counts": EXPECTED_STAGE_COUNTS,
        "stage_split_counts": _nested_counts(rows, "stage"),
        "source_family_split_counts": _nested_counts(rows, "source_family"),
        "domain_split_counts": _nested_counts(rows, "domain"),
        "immutable_failure_count": sum(bool(row["immutable_failure"]) for row in rows),
        "overlay": {
            "path": overlay_path,
            "record_count": EXPECTED_SANDBOXES,
            "byte_count": overlay_byte_count,
            "file_blake3": overlay_blake3,
        },
        "base_dataset": {
            "manifest_envelope_blake3": base_manifest_envelope_blake3,
            "dataset_blake3": base_manifest["dataset_blake3"],
            "catalog_blake3": base_manifest["catalog_blake3"],
            "selection_blake3": base_manifest["selection_blake3"],
            "shard_count": base_manifest["shard_count"],
        },
        "split_authority": {
            "selection_blake3": selection["selection_blake3"],
            "base_selection_blake3": selection["base_selection_blake3"],
            "plan_blake3": selection["plan_blake3"],
        },
        "execution_authority": {
            "manifest_envelope_blake3": execution_catalog_envelope_blake3,
            "catalog_blake3": execution_catalog["catalog_blake3"],
            "selection_blake3": execution_catalog["selection_blake3"],
        },
        "evaluator": _plain(evaluator_recipe),
        "model_registry_file_blake3": model_registry_file_blake3,
        "controls": {
            "provider_calls": 0,
            "base_shards_rewritten": False,
            "base_shards_rehashed": False,
            "base_sandbox_ids_changed": False,
            "base_candidate_ids_changed": False,
            "source_outcomes_reclassified": False,
            "immutable_failures_preserved": True,
            "release_consumers_must_join_overlay": True,
            "fixed_evaluator_cascade_required": True,
            "all_model_scores_retained": True,
        },
    }
    return {**core, "release_blake3": blake3_hex(core)}


def release_view_records(
    base_records: Sequence[Mapping[str, Any]],
    overlay_rows: Sequence[Mapping[str, Any]],
    *,
    split: str | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Expose split-aware records without changing the committed base bytes."""

    if split is not None and split not in SPLITS:
        raise BulkRLReleaseOverlayError("requested release split differs")
    if (
        len(base_records) != EXPECTED_SANDBOXES
        or len(overlay_rows) != EXPECTED_SANDBOXES
    ):
        raise BulkRLReleaseOverlayError("release view inventory differs")
    result: list[Mapping[str, Any]] = []
    for base, overlay in zip(base_records, overlay_rows, strict=True):
        if (
            overlay.get("schema") != OVERLAY_ROW_SCHEMA
            or overlay.get("sandbox_id") != base.get("sandbox_id")
            or overlay.get("candidate_id") != base.get("candidate_id")
            or overlay.get("base_record_blake3") != base.get("record_blake3")
        ):
            raise BulkRLReleaseOverlayError("release view join differs")
        if split is None or overlay.get("release_split") == split:
            result.append(
                MappingProxyType(
                    {
                        "schema": "eva.bulk-rl-release-view-record.v1",
                        "release_split": overlay["release_split"],
                        "sandbox": base,
                    }
                )
            )
    return tuple(result)


def _readonly_root(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if path.is_symlink():
        raise BulkRLReleaseOverlayError(f"{label} topology differs")
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError:
        raise BulkRLReleaseOverlayError(f"{label} topology differs") from None
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o222:
        raise BulkRLReleaseOverlayError(f"{label} must be a read-only directory")
    return resolved


def _read_readonly_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
    expected_bytes: int | None = None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise BulkRLReleaseOverlayError(f"{label} topology differs") from None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o222
        ):
            raise BulkRLReleaseOverlayError(
                f"{label} must be a read-only regular non-symlink single-link file"
            )
        if metadata.st_size > maximum_bytes or (
            expected_bytes is not None and metadata.st_size != expected_bytes
        ):
            raise BulkRLReleaseOverlayError(f"{label} byte count differs")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise BulkRLReleaseOverlayError(f"{label} byte count differs")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise BulkRLReleaseOverlayError(f"{label} byte count differs")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BulkRLReleaseOverlayError("authenticated JSON contains a duplicate key")
        result[key] = value
    return result


def _decode_json(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except BulkRLReleaseOverlayError:
        raise
    except (UnicodeError, ValueError):
        raise BulkRLReleaseOverlayError(f"{label} JSON differs") from None


def _load_signed_payload(
    path: Path,
    *,
    trust_store: Path,
    label: str,
) -> tuple[SignedEnvelope, Mapping[str, Any]]:
    document = _decode_json(
        _read_readonly_file(path, label=label, maximum_bytes=16 * 1024 * 1024),
        label=label,
    )
    try:
        envelope = SignedEnvelope.from_document(document)
        verify_signed_envelope(envelope, trust_store_path=trust_store)
    except (AdmissionReceiptError, TypeError, ValueError):
        raise BulkRLReleaseOverlayError(f"{label} signature differs") from None
    return envelope, envelope.payload


def _load_authenticated_base(
    root: Path,
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    if (
        manifest.get("schema") != "eva.bulk-rl-sandbox-sharded-dataset.v1"
        or manifest.get("sandbox_count") != EXPECTED_SANDBOXES
        or manifest.get("split_counts") != {"train": EXPECTED_SANDBOXES}
        or manifest.get("stage_counts") != EXPECTED_STAGE_COUNTS
        or not _hex_digest(manifest.get("dataset_blake3"))
        or not _hex_digest(manifest.get("catalog_blake3"))
        or not _hex_digest(manifest.get("selection_blake3"))
    ):
        raise BulkRLReleaseOverlayError("base bulk manifest differs")
    controls = manifest.get("controls")
    if not isinstance(controls, Mapping) or controls.get("records_are_canonical_jsonl") is not True:
        raise BulkRLReleaseOverlayError("base bulk canonical-record control differs")
    shards = manifest.get("shards")
    if (
        not isinstance(shards, Sequence)
        or isinstance(shards, (str, bytes))
        or manifest.get("shard_count") != len(shards)
        or not shards
    ):
        raise BulkRLReleaseOverlayError("base shard inventory differs")

    records: list[Mapping[str, Any]] = []
    observed_sandboxes: set[str] = set()
    observed_candidates: set[str] = set()
    for ordinal, descriptor in enumerate(shards):
        if not isinstance(descriptor, Mapping) or descriptor.get("ordinal") != ordinal:
            raise BulkRLReleaseOverlayError("base shard descriptor differs")
        relative = descriptor.get("path")
        record_count = descriptor.get("record_count")
        byte_count = descriptor.get("byte_count")
        if (
            not isinstance(relative, str)
            or relative != f"sandboxes-{ordinal:05d}.jsonl"
            or type(record_count) is not int
            or record_count <= 0
            or type(byte_count) is not int
            or byte_count <= 0
            or not _hex_digest(descriptor.get("file_blake3"))
            or not _hex_digest(descriptor.get("shard_blake3"))
        ):
            raise BulkRLReleaseOverlayError("base shard descriptor differs")
        payload = _read_readonly_file(
            root / relative,
            label=f"base shard {ordinal}",
            maximum_bytes=512 * 1024 * 1024,
            expected_bytes=byte_count,
        )
        lines = payload.splitlines()
        if len(lines) != record_count:
            raise BulkRLReleaseOverlayError("base shard record count differs")
        shard_rows: list[Mapping[str, Any]] = []
        for line in lines:
            row = _decode_json(line, label="base shard row")
            if not isinstance(row, Mapping) or row.get("schema") != "eva.bulk-rl-sandbox.v1":
                raise BulkRLReleaseOverlayError("base sandbox record schema differs")
            sandbox_id = row.get("sandbox_id")
            candidate_id = row.get("candidate_id")
            if (
                not isinstance(sandbox_id, str)
                or not sandbox_id
                or not isinstance(candidate_id, str)
                or not candidate_id
                or sandbox_id in observed_sandboxes
                or candidate_id in observed_candidates
                or not _hex_digest(row.get("record_blake3"))
            ):
                raise BulkRLReleaseOverlayError("base sandbox identity differs")
            observed_sandboxes.add(sandbox_id)
            observed_candidates.add(candidate_id)
            shard_rows.append(row)
        if (
            descriptor.get("first_sandbox_id") != shard_rows[0].get("sandbox_id")
            or descriptor.get("last_sandbox_id") != shard_rows[-1].get("sandbox_id")
        ):
            raise BulkRLReleaseOverlayError("base shard endpoint binding differs")
        records.extend(shard_rows)
    if len(records) != EXPECTED_SANDBOXES:
        raise BulkRLReleaseOverlayError("base sandbox inventory differs")
    return tuple(records)


def _verify_evaluator(evaluator: Any) -> None:
    if not isinstance(evaluator, Mapping) or evaluator.get("schema") != CASCADE_SCHEMA:
        raise BulkRLReleaseOverlayError("release evaluator schema differs")
    order = evaluator.get("cohort_order")
    models = evaluator.get("models")
    if (
        not isinstance(order, Sequence)
        or isinstance(order, (str, bytes))
        or not isinstance(models, Sequence)
        or isinstance(models, (str, bytes))
        or len(order) != 5
        or len(models) != 5
        or any(not isinstance(role, str) or not role for role in order)
        or len(set(order)) != 5
        or evaluator.get("rollouts_per_sandbox") != 5
        or evaluator.get("all_model_scores_retained") is not True
        or evaluator.get("ability_separation_computed_from_fixed_scores") is not True
        or evaluator.get("cascade_definition_is_release_fixed") is not True
    ):
        raise BulkRLReleaseOverlayError("release fixed evaluator cascade differs")
    for ordinal, (role, model) in enumerate(zip(order, models, strict=True)):
        if (
            not isinstance(role, str)
            or not isinstance(model, Mapping)
            or model.get("ordinal") != ordinal
            or model.get("role_id") != role
            or not isinstance(model.get("model_id"), str)
            or not model.get("model_id")
            or not isinstance(model.get("provider_family"), str)
            or not model.get("provider_family")
        ):
            raise BulkRLReleaseOverlayError("release evaluator model binding differs")
    judge = evaluator.get("agent_judge")
    if (
        not isinstance(judge, Mapping)
        or judge.get("role_id") != "architect_opus_5"
        or "opus-5" not in str(judge.get("model_id", ""))
        or not isinstance(judge.get("provider_family"), str)
        or judge.get("workspace_inspection_required") is not True
    ):
        raise BulkRLReleaseOverlayError("release Opus workspace Agent Judge binding differs")
    recipe = evaluator.get("recipe_blake3")
    recipe_core = {key: _plain(value) for key, value in evaluator.items() if key != "recipe_blake3"}
    if not _hex_digest(recipe) or recipe != blake3_hex(recipe_core):
        raise BulkRLReleaseOverlayError("release evaluator recipe commitment differs")


def _verify_release_manifest(
    manifest: Mapping[str, Any],
    *,
    base_manifest: Mapping[str, Any],
    base_envelope_blake3: str,
) -> Mapping[str, Any]:
    release_blake3 = manifest.get("release_blake3")
    release_core = {
        key: _plain(value) for key, value in manifest.items() if key != "release_blake3"
    }
    if (
        manifest.get("schema") != OVERLAY_DATASET_SCHEMA
        or manifest.get("release_method")
        != "immutable-bulk-v1_plus-outcome-blind-v3-split-overlay-v1"
        or manifest.get("sandbox_count") != EXPECTED_SANDBOXES
        or manifest.get("split_counts") != EXPECTED_SPLITS
        or manifest.get("stage_counts") != EXPECTED_STAGE_COUNTS
        or not _hex_digest(release_blake3)
        or release_blake3 != blake3_hex(release_core)
        or not _hex_digest(manifest.get("model_registry_file_blake3"))
        or type(manifest.get("immutable_failure_count")) is not int
    ):
        raise BulkRLReleaseOverlayError("release manifest contract differs")
    controls = manifest.get("controls")
    expected_controls = {
        "provider_calls": 0,
        "base_shards_rewritten": False,
        "base_shards_rehashed": False,
        "base_sandbox_ids_changed": False,
        "base_candidate_ids_changed": False,
        "source_outcomes_reclassified": False,
        "immutable_failures_preserved": True,
        "release_consumers_must_join_overlay": True,
        "fixed_evaluator_cascade_required": True,
        "all_model_scores_retained": True,
    }
    if not isinstance(controls, Mapping) or any(
        controls.get(key) != value for key, value in expected_controls.items()
    ):
        raise BulkRLReleaseOverlayError("release immutability controls differ")
    base = manifest.get("base_dataset")
    if (
        not isinstance(base, Mapping)
        or base.get("manifest_envelope_blake3") != base_envelope_blake3
        or base.get("dataset_blake3") != base_manifest.get("dataset_blake3")
        or base.get("catalog_blake3") != base_manifest.get("catalog_blake3")
        or base.get("selection_blake3") != base_manifest.get("selection_blake3")
        or base.get("shard_count") != base_manifest.get("shard_count")
    ):
        raise BulkRLReleaseOverlayError("release base authority binding differs")
    split_authority = manifest.get("split_authority")
    execution_authority = manifest.get("execution_authority")
    if (
        not isinstance(split_authority, Mapping)
        or not isinstance(execution_authority, Mapping)
        or split_authority.get("base_selection_blake3") != base_manifest.get("selection_blake3")
        or execution_authority.get("selection_blake3")
        != split_authority.get("selection_blake3")
        or not all(
            _hex_digest(value)
            for value in (
                split_authority.get("selection_blake3"),
                split_authority.get("base_selection_blake3"),
                split_authority.get("plan_blake3"),
                execution_authority.get("manifest_envelope_blake3"),
                execution_authority.get("catalog_blake3"),
            )
        )
    ):
        raise BulkRLReleaseOverlayError("release split/execution authority binding differs")
    _verify_evaluator(manifest.get("evaluator"))
    overlay = manifest.get("overlay")
    if (
        not isinstance(overlay, Mapping)
        or overlay.get("path") != "split-overlay.jsonl"
        or overlay.get("record_count") != EXPECTED_SANDBOXES
        or type(overlay.get("byte_count")) is not int
        or overlay.get("byte_count") <= 0
        or not _hex_digest(overlay.get("file_blake3"))
    ):
        raise BulkRLReleaseOverlayError("release overlay descriptor differs")
    return overlay


def _load_authenticated_overlay(
    root: Path,
    descriptor: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    payload = _read_readonly_file(
        root / "split-overlay.jsonl",
        label="release overlay",
        maximum_bytes=64 * 1024 * 1024,
        expected_bytes=descriptor["byte_count"],
    )
    if blake3_bytes(payload) != descriptor["file_blake3"]:
        raise BulkRLReleaseOverlayError("release overlay file BLAKE3 differs")
    lines = payload.splitlines()
    if len(lines) != descriptor["record_count"]:
        raise BulkRLReleaseOverlayError("release overlay record count differs")
    expected_keys = {
        "schema", "release_ordinal", "sandbox_id", "candidate_id",
        "base_record_blake3", "base_queue_ordinal", "release_split",
        "source_family", "domain", "stage", "rubric_id", "rubric_blake3",
        "execution_status", "execution_row_blake3", "execution_proof_blake3",
        "immutable_failure",
    }
    rows: list[Mapping[str, Any]] = []
    sandboxes: set[str] = set()
    candidates: set[str] = set()
    for ordinal, line in enumerate(lines, start=1):
        row = _decode_json(line, label="release overlay row")
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_keys
            or row.get("schema") != OVERLAY_ROW_SCHEMA
            or type(row.get("release_ordinal")) is not int
            or row.get("release_ordinal") != ordinal
            or type(row.get("base_queue_ordinal")) is not int
            or row.get("release_split") not in SPLITS
            or row.get("stage") not in STAGES
            or not isinstance(row.get("source_family"), str)
            or not row.get("source_family")
            or not isinstance(row.get("domain"), str)
            or not row.get("domain")
            or not isinstance(row.get("rubric_id"), str)
            or not row.get("rubric_id")
            or not _blake3_digest(row.get("rubric_blake3"))
            or row.get("execution_status") not in {"executable_legacy", "source_only", "rejected"}
            or not _hex_digest(row.get("execution_row_blake3"))
            or (
                row.get("execution_proof_blake3") is not None
                and not _hex_digest(row.get("execution_proof_blake3"))
            )
            or type(row.get("immutable_failure")) is not bool
            or row.get("immutable_failure") != (row.get("execution_status") == "rejected")
        ):
            raise BulkRLReleaseOverlayError("release overlay row contract differs")
        sandbox_id = row.get("sandbox_id")
        candidate_id = row.get("candidate_id")
        if (
            not isinstance(sandbox_id, str)
            or not sandbox_id
            or not isinstance(candidate_id, str)
            or not candidate_id
            or sandbox_id in sandboxes
            or candidate_id in candidates
            or not _hex_digest(row.get("base_record_blake3"))
        ):
            raise BulkRLReleaseOverlayError("release overlay identity differs")
        sandboxes.add(sandbox_id)
        candidates.add(candidate_id)
        rows.append(row)
    if (
        dict(Counter(str(row["release_split"]) for row in rows)) != manifest.get("split_counts")
        or dict(Counter(str(row["stage"]) for row in rows)) != manifest.get("stage_counts")
        or _nested_counts(rows, "stage") != manifest.get("stage_split_counts")
        or _nested_counts(rows, "source_family") != manifest.get("source_family_split_counts")
        or _nested_counts(rows, "domain") != manifest.get("domain_split_counts")
        or sum(bool(row["immutable_failure"]) for row in rows)
        != manifest.get("immutable_failure_count")
    ):
        raise BulkRLReleaseOverlayError("release overlay balance differs")
    return tuple(rows)


def _verify_exact_join(
    base_records: Sequence[Mapping[str, Any]],
    overlay_rows: Sequence[Mapping[str, Any]],
) -> None:
    for base, overlay in zip(base_records, overlay_rows, strict=True):
        reward = base.get("reward_contract")
        if not isinstance(reward, Mapping) or (
            overlay.get("sandbox_id"),
            overlay.get("candidate_id"),
            overlay.get("base_record_blake3"),
            overlay.get("base_queue_ordinal"),
            overlay.get("source_family"),
            overlay.get("domain"),
            overlay.get("stage"),
            overlay.get("rubric_id"),
            overlay.get("rubric_blake3"),
        ) != (
            base.get("sandbox_id"),
            base.get("candidate_id"),
            base.get("record_blake3"),
            base.get("queue_ordinal"),
            base.get("source_family"),
            base.get("domain"),
            base.get("stage"),
            reward.get("rubric_id"),
            reward.get("rubric_digest"),
        ):
            raise BulkRLReleaseOverlayError("authenticated release exact join differs")


def load_verified_release_view(
    base_root: str | Path,
    release_root: str | Path,
    trust_store: str | Path,
    split: str | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Load the authenticated split view without rehashing immutable base shards.

    The signed base manifest authenticates the already sealed base.  To keep the
    hot-path bounded, shard bytes are parsed but never digested again; this is
    allowed only while every authority and data file remains read-only,
    non-symlinked, and single-linked.  The much smaller release overlay is
    always BLAKE3-verified before its exact positional join is returned.
    """

    if split is not None and split not in SPLITS:
        raise BulkRLReleaseOverlayError("requested release split differs")
    base = _readonly_root(base_root, label="base root")
    release = _readonly_root(release_root, label="release root")
    trust = Path(trust_store)
    if trust.is_symlink():
        raise BulkRLReleaseOverlayError("trust store topology differs")
    try:
        trust = trust.resolve(strict=True)
    except OSError:
        raise BulkRLReleaseOverlayError("trust store topology differs") from None
    base_envelope, base_manifest = _load_signed_payload(
        base / "manifest.json", trust_store=trust, label="base manifest"
    )
    _release_envelope, release_manifest = _load_signed_payload(
        release / "manifest.json", trust_store=trust, label="release manifest"
    )
    descriptor = _verify_release_manifest(
        release_manifest,
        base_manifest=base_manifest,
        base_envelope_blake3=base_envelope.envelope_blake3,
    )
    base_records = _load_authenticated_base(base, base_manifest)
    overlay_rows = _load_authenticated_overlay(release, descriptor, release_manifest)
    _verify_exact_join(base_records, overlay_rows)
    return release_view_records(base_records, overlay_rows, split=split)


__all__ = [
    "BulkRLReleaseOverlayError",
    "CASCADE_SCHEMA",
    "EXPECTED_SANDBOXES",
    "EXPECTED_SCHEDULED",
    "EXPECTED_SPLITS",
    "EXPECTED_STAGE_COUNTS",
    "OVERLAY_DATASET_SCHEMA",
    "OVERLAY_ROW_SCHEMA",
    "derive_overlay_rows",
    "fixed_evaluator_recipe",
    "load_verified_release_view",
    "release_manifest_core",
    "release_view_records",
]
