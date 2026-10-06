"""Signed, provider-free promotion of premium construction publications.

The premium construction proof catalog is intentionally not a supervisor
transition.  This module is the narrow bridge that may turn its successful
rows into ``frozen -> executable_ready`` candidates.  It reopens the exact
selection UUIDs, rubric rows, construction publications, unchanged legacy
schemas, and real handler prerequisites *before* signing or materializing an
append-only transition bundle.

This is not an admission transition.  No rollout, judge, admission ledger, or
provider interface is imported or invoked here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from types import MappingProxyType
from threading import RLock
from typing import Any, Callable, Mapping

from eva_agent.admission.receipts import (
    SignedEnvelope,
    issue_signed_envelope,
    verify_signed_envelope,
)
from eva_agent.campaign.selection_v2 import CampaignSelectionV2, SelectionEntryV2
from eva_agent.pipeline.contracts import JsonValue, RuntimeIdFactory, Stage, freeze_json, uuid_text
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes, canonical_value, is_blake3
from eva_agent.pipeline.ids import RandomUUIDFactory
from eva_agent.pipeline.tools import ToolDefinition, ToolRegistry
from eva_agent.rubrics.registry import CompiledRubricRegistry
from eva_agent.sources.legacy_execution import (
    CandidateToolCatalogEntry,
    EXPECTED_TOOL_NAMES,
    ProductionSourceBlocker,
    _LegacyCandidateToolBackend,
    _decode_json,
    _load_legacy_modules,
    _read_bytes,
    _real_directory,
    _safe_file,
)

from .legacy_exact import (
    LegacyConstructionAdapterError,
    _LegacyAuthority,
    _decode_object,
    _ensure_directory,
    _existing_directory,
    _mkdir_new_child,
    _read_relative,
    _safe_relative,
    _seal_tree,
    _write_new_relative,
    verify_legacy_construction_publication,
)
from .premium_campaign import (
    PremiumConstructionProofCatalog,
    PremiumConstructionQueue,
    PremiumConstructionTerminalRecord,
    _receipt_from_document,
    verify_premium_construction_proof_catalog,
)


TRANSITION_SCHEMA = "eva.premium-construction-supervisor-transition.v1"
CLAIM_SCHEMA = "eva.premium-construction-supervisor-transition-claim.v1"
TRANSITION_SCHEMA_V2 = "eva.premium-construction-supervisor-transition.v2"
CLAIM_SCHEMA_V2 = "eva.premium-construction-supervisor-transition-claim.v2"
SELECTED_IDS_SCHEMA_V2 = "eva.premium-construction-selected-candidate-ids.v2"
CLAIM_DOMAIN_SCHEMA_V2 = "eva.premium-construction-supervisor-claim-domain.v2"
CATALOG_SCHEMA = "eva.premium-candidate-execution-binding-catalog.v1"
BINDING_SCHEMA = "eva.premium-candidate-execution-binding.v1"
_MAXIMUM_DOCUMENT_BYTES = 64 * 1024 * 1024
_SOURCE_FILES = MappingProxyType(
    {
        "selection": "campaign-selection.v2.json",
        "queue": "premium-construction-queue.v1.json",
        "proof_catalog": "premium-construction-proof-catalog.v1.json",
        "supervisor_input": "premium-construction-supervisor-input.v1.json",
    }
)


class PremiumConstructionSupervisorError(ProductionSourceBlocker):
    """A replenishment transition or executable binding failed closed."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    return value


def _utc(value: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PremiumConstructionSupervisorError("transition clock must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise PremiumConstructionSupervisorError("transition clock differs") from None
    if parsed.tzinfo != UTC or parsed.isoformat().replace("+00:00", "Z") != value:
        raise PremiumConstructionSupervisorError("transition clock must be canonical UTC")
    return value


def _read_document(root: Path, relative: str, *, label: str) -> tuple[Mapping[str, Any], bytes]:
    payload = _read_relative(
        root,
        _safe_relative(relative, label=label),
        label=label,
        maximum_bytes=_MAXIMUM_DOCUMENT_BYTES,
    )
    return _decode_object(payload, label=label), payload


def _file_reference(relative: str, payload: bytes) -> Mapping[str, JsonValue]:
    return freeze_json(
        {
            "path": relative,
            "byte_count": len(payload),
            "blake3": blake3_bytes(payload),
            "mode": "0400",
        }
    )


def _verify_file_reference(
    root: Path,
    value: Any,
    *,
    expected_path: str,
    label: str,
) -> tuple[Mapping[str, Any], bytes]:
    if not isinstance(value, Mapping) or set(value) != {"path", "byte_count", "blake3", "mode"}:
        raise PremiumConstructionSupervisorError(f"{label} reference shape differs")
    if value.get("path") != expected_path or value.get("mode") != "0400":
        raise PremiumConstructionSupervisorError(f"{label} reference path/mode differs")
    document, payload = _read_document(root, expected_path, label=label)
    path = root / expected_path
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or value.get("byte_count") != len(payload)
        or value.get("blake3") != blake3_bytes(payload)
    ):
        raise PremiumConstructionSupervisorError(f"{label} reference bytes differ")
    return document, payload


def _entry_binding(
    *,
    selection: CampaignSelectionV2,
    queue: PremiumConstructionQueue,
    rubrics: CompiledRubricRegistry,
) -> tuple[Mapping[str, SelectionEntryV2], Mapping[str, Any]]:
    CampaignSelectionV2.from_document(selection.to_document())
    PremiumConstructionQueue.from_document(queue.to_document())
    if not (
        queue.selection_id == selection.selection_id
        and queue.selection_blake3 == selection.selection_blake3
        and queue.readiness_authority_blake3
        == selection.readiness_authority.get("authority_blake3")
        and selection.rubric_registry_blake3 == rubrics.digest
    ):
        raise PremiumConstructionSupervisorError("selection/queue/rubric authority differs")
    selection_by_id = {row.candidate_id: row for row in selection.entries}
    if len(selection_by_id) != len(selection.entries):
        raise PremiumConstructionSupervisorError("selection UUID inventory differs")
    queue_by_id: dict[str, Any] = {}
    for queued in queue.entries:
        selected = selection_by_id.get(queued.candidate_id)
        if selected is None:
            raise PremiumConstructionSupervisorError("queue candidate UUID is outside selection")
        expected = {
            "candidate_id": selected.candidate_id,
            "source_candidate_id": selected.source_candidate_id,
            "source_family": selected.source_family,
            "source_artifact_sha256": selected.source_artifact_sha256,
            "domain": selected.domain,
            "stage": selected.stage,
            "selection_tier": selected.selection_tier,
            "selection_queue_ordinal": selected.queue_ordinal,
            "campaign_ordinal": queued.campaign_ordinal,
            "readiness_proof_root_blake3": selected.readiness_proof_root_blake3,
        }
        if (
            selected.construction_readiness != "frozen"
            or queued.core() != expected
            or queued.entry_blake3 != blake3_hex(expected)
        ):
            raise PremiumConstructionSupervisorError("queue changed one frozen selection field")
        rubric = rubrics.resolve(selected.domain, selected.stage)
        if rubric.rubric_id != selected.rubric_id or rubric.digest != selected.rubric_blake3:
            raise PremiumConstructionSupervisorError("candidate rubric/stage binding differs")
        queue_by_id[queued.candidate_id] = queued
    return MappingProxyType(selection_by_id), MappingProxyType(queue_by_id)


@dataclass(frozen=True, slots=True)
class _RuntimeAuthority:
    modules: Any
    validator: Any
    trust_store: Mapping[str, str]
    image_refs: Mapping[str, str]
    private_key: Any
    key_id: str
    docker_binary: str


def _runtime_authority(
    *,
    legacy_python_root: str | Path,
    trust_store_path: str | Path,
    image_refs_path: str | Path,
    host_private_key_path: str | Path,
    host_key_id: str,
    docker_binary: str,
) -> _RuntimeAuthority:
    python_root = _real_directory(legacy_python_root, label="legacy Python root")
    modules = _load_legacy_modules(python_root)
    validator = _LegacyAuthority(python_root)
    trust_path = Path(trust_store_path)
    image_path = Path(image_refs_path)
    if trust_path.is_symlink() or image_path.is_symlink():
        raise PremiumConstructionSupervisorError("runtime control path cannot be a symlink")
    trust_root = _real_directory(trust_path.resolve(strict=True).parent, label="trust-store parent")
    image_root = _real_directory(image_path.resolve(strict=True).parent, label="image-map parent")
    trust_file = _safe_file(trust_path.resolve(strict=True), root=trust_root, label="host trust store")
    image_file = _safe_file(image_path.resolve(strict=True), root=image_root, label="image reference map")
    try:
        trust_document = _decode_json(
            _read_bytes(trust_file, root=trust_root, label="host trust store"),
            label="host trust store",
        )
        image_document = _decode_json(
            _read_bytes(image_file, root=image_root, label="image reference map"),
            label="image reference map",
        )
    except ProductionSourceBlocker:
        raise PremiumConstructionSupervisorError("runtime control JSON differs") from None
    keys = trust_document.get("keys") if isinstance(trust_document, Mapping) else None
    images = image_document.get("images") if isinstance(image_document, Mapping) else None
    if not (
        trust_document.get("schema") == "rlevo.med-research-host-trust-store.v1"
        and trust_document.get("status") == "active"
        and isinstance(keys, Mapping)
        and all(isinstance(key, str) and isinstance(value, str) for key, value in keys.items())
        and image_document.get("schema") == "rlevo.med-research-image-refs.v1"
        and isinstance(images, Mapping)
        and all(isinstance(key, str) and isinstance(value, str) for key, value in images.items())
    ):
        raise PremiumConstructionSupervisorError("runtime trust/image control differs")
    key_path = Path(host_private_key_path)
    if not key_path.is_absolute() or key_path.is_symlink():
        raise PremiumConstructionSupervisorError("host private key must be absolute and non-symlink")
    key_file = _safe_file(
        key_path.resolve(strict=True),
        root=key_path.parent.resolve(strict=True),
        label="host private key",
    )
    if not isinstance(host_key_id, str) or host_key_id not in keys:
        raise PremiumConstructionSupervisorError("host signing key ID is not trusted")
    private_key = modules.key_management.load_host_private_key(key_file)
    if modules.host_receipts.public_key_base64(private_key) != keys[host_key_id]:
        raise PremiumConstructionSupervisorError("host private key differs from trust store")
    if not isinstance(docker_binary, str) or not docker_binary or shutil.which(docker_binary) is None:
        raise PremiumConstructionSupervisorError("Docker handler prerequisite is unavailable")
    return _RuntimeAuthority(
        modules=modules,
        validator=validator,
        trust_store=MappingProxyType(dict(keys)),
        image_refs=MappingProxyType(dict(images)),
        private_key=private_key,
        key_id=host_key_id,
        docker_binary=docker_binary,
    )


def _publication_artifact(
    publication_root: Path,
    reference: Any,
    *,
    label: str,
) -> tuple[Path, Mapping[str, Any], bytes]:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "byte_count", "blake3", "mode"}:
        raise PremiumConstructionSupervisorError(f"{label} artifact reference differs")
    if reference.get("mode") != "0400" or not is_blake3(reference.get("blake3")):
        raise PremiumConstructionSupervisorError(f"{label} artifact integrity differs")
    relative = _safe_relative(reference.get("path"), label=label)
    payload = _read_relative(
        publication_root,
        relative,
        label=label,
        maximum_bytes=_MAXIMUM_DOCUMENT_BYTES,
    )
    path = publication_root / relative.as_posix()
    metadata = path.lstat()
    if not (
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_nlink == 1
        and stat.S_IMODE(metadata.st_mode) == 0o400
        and len(payload) == reference.get("byte_count")
        and blake3_bytes(payload) == reference.get("blake3")
    ):
        raise PremiumConstructionSupervisorError(f"{label} artifact bytes differ")
    return path, _decode_object(payload, label=label), payload


@dataclass(frozen=True, slots=True)
class _ExecutableMaterial:
    selected: SelectionEntryV2
    queued: Any
    record: PremiumConstructionTerminalRecord
    catalog_binding: Mapping[str, JsonValue]
    publication_root: Path
    executable_binding_path: Path
    executable_binding_blake3: str
    source_policy_path: Path
    source_policy_blake3: str
    source_policy: Mapping[str, Any]
    runtime_policy: Mapping[str, Any]
    runtime_policy_blake3: str
    templates: Mapping[str, Mapping[str, Any]]
    tool_catalog: tuple[CandidateToolCatalogEntry, ...]
    tool_catalog_blake3: str
    source_request_blake3: str


def _open_executable_material(
    *,
    selected: SelectionEntryV2,
    queued: Any,
    record: PremiumConstructionTerminalRecord,
    catalog_binding: Mapping[str, JsonValue],
    selection_id: str,
    publication_output_root: Path,
    authority: _RuntimeAuthority,
) -> _ExecutableMaterial:
    if record.publication_receipt is None:
        raise PremiumConstructionSupervisorError("successful record lacks publication receipt")
    receipt = _receipt_from_document(record.publication_receipt)
    try:
        verify_legacy_construction_publication(publication_output_root, receipt)
    except LegacyConstructionAdapterError as exc:
        raise PremiumConstructionSupervisorError("construction publication failed reopen") from exc
    publication_root = _existing_directory(
        publication_output_root / receipt.publication_id,
        label="premium construction publication",
    )
    binding_path = _safe_file(
        publication_root / "executable-binding.v1.json",
        root=publication_root,
        label="premium executable binding",
    )
    binding_bytes = _read_relative(
        publication_root,
        _safe_relative(
            "executable-binding.v1.json", label="premium executable binding"
        ),
        label="premium executable binding",
        maximum_bytes=_MAXIMUM_DOCUMENT_BYTES,
    )
    binding = _decode_object(binding_bytes, label="premium executable binding")
    if not (
        binding.get("schema") == "eva.premium-codex-executable-binding.v1"
        and binding.get("source_id") == selected.candidate_id
        and binding.get("publication_id") == receipt.publication_id
        and binding.get("binding_blake3") == record.binding_blake3
        and blake3_bytes(binding_bytes) == receipt.artifact_blake3
        and binding.get("validator_authority_blake3") == record.validator_authority_blake3
        and binding.get("executable_material_blake3") == record.executable_material_blake3
        and _plain(catalog_binding)
        == {
            "candidate_id": record.candidate_id,
            "source_candidate_id": record.source_candidate_id,
            "selection_blake3": record.selection_blake3,
            "record_blake3": record.record_blake3,
            "publication_id": receipt.publication_id,
            "publication_receipt_blake3": receipt.receipt_blake3,
            "publication_artifact_blake3": receipt.artifact_blake3,
            "binding_blake3": record.binding_blake3,
            "validator_authority_blake3": record.validator_authority_blake3,
            "executable_material_blake3": record.executable_material_blake3,
        }
    ):
        raise PremiumConstructionSupervisorError("catalog/publication executable identity differs")
    files = binding.get("files")
    executable = binding.get("executable_artifacts")
    if not isinstance(files, Mapping) or not isinstance(executable, Mapping):
        raise PremiumConstructionSupervisorError("executable artifact inventory differs")
    required = {
        "policy",
        "private_evaluation",
        "evidence_manifest",
        "reference_answer",
        "s1_plan_contract",
        "s2_evidence_contract",
        "s3_execution_contract",
        "s4_execution_contract",
        "s5_submission_contract",
    }
    if set(executable) != required or any(executable[name] != files.get(name) for name in required):
        raise PremiumConstructionSupervisorError("executable artifact allowlist differs")
    loaded: dict[str, Mapping[str, Any]] = {}
    loaded_bytes: dict[str, bytes] = {}
    loaded_paths: dict[str, Path] = {}
    for name in sorted(required):
        path, document, payload = _publication_artifact(
            publication_root, executable[name], label=f"premium {name}"
        )
        loaded[name] = document
        loaded_bytes[name] = payload
        loaded_paths[name] = path
    _validator_path, validator_document, _validator_bytes = _publication_artifact(
        publication_root, files.get("validator_authority"), label="legacy validator authority"
    )
    if _plain(validator_document) != _plain(authority.validator.identity.to_document()):
        raise PremiumConstructionSupervisorError("legacy validator authority changed")
    _frozen_path, frozen_source, _frozen_bytes = _publication_artifact(
        publication_root, files.get("frozen_source"), label="frozen construction source"
    )
    if not (
        frozen_source.get("schema") == "eva.selection-v2-frozen-legacy-construction-source.v1"
        and frozen_source.get("selection_id") == selection_id
        and frozen_source.get("selection_blake3") == record.selection_blake3
        and frozen_source.get("selection_entry") == selected.to_document()
        and frozen_source.get("construction_input_upstream_sha256")
        == selected.source_artifact_sha256
        and frozen_source.get("controls", {}).get("selection_readiness_required") == "frozen"
        and frozen_source.get("controls", {}).get("provider_calls_during_source_resolution") == 0
    ):
        raise PremiumConstructionSupervisorError("frozen source UUID/selection binding differs")
    source_request_blake3 = blake3_hex(frozen_source)
    if binding.get("source_request_blake3") != source_request_blake3:
        raise PremiumConstructionSupervisorError("frozen source request commitment differs")
    _construction_path, construction_input, construction_bytes = _publication_artifact(
        publication_root, files.get("construction_input"), label="construction input"
    )
    if not (
        frozen_source.get("construction_input") == construction_input
        and frozen_source.get("construction_input_blake3") == blake3_bytes(construction_bytes)
    ):
        raise PremiumConstructionSupervisorError("construction input source binding differs")
    try:
        authority.validator.candidate.validate_construction_input(construction_input)
    except Exception as exc:
        raise PremiumConstructionSupervisorError("construction input legacy validation failed") from exc
    sandbox = construction_input.get("sandbox")
    if not isinstance(sandbox, Mapping) or not (
        sandbox.get("sandbox_id") == selected.source_candidate_id
        and sandbox.get("focus") == selected.stage
    ):
        raise PremiumConstructionSupervisorError("construction sandbox identity/stage differs")

    policy = loaded["policy"]
    private_evaluation = loaded["private_evaluation"]
    evidence_manifest = loaded["evidence_manifest"]
    reference_answer = loaded["reference_answer"]
    templates = {
        "s1": loaded["s1_plan_contract"],
        "s2": loaded["s2_evidence_contract"],
        "s3": loaded["s3_execution_contract"],
        "s4": loaded["s4_execution_contract"],
        "s5": loaded["s5_submission_contract"],
    }
    if not (
        policy.get("sandbox_id") == selected.source_candidate_id
        and policy.get("focus") == selected.stage
        and policy.get("target_split") == "candidate"
        and private_evaluation.get("sandbox_id") == selected.source_candidate_id
        and private_evaluation.get("focus") == selected.stage
        and evidence_manifest.get("sandbox_id") == selected.source_candidate_id
    ):
        raise PremiumConstructionSupervisorError("constructed policy/stage identity differs")
    candidate = authority.validator.candidate
    try:
        candidate.validate_policy(policy)
        candidate.validate_private_evaluation(private_evaluation)
        candidate.validate_evidence_manifest(evidence_manifest, private_evaluation)
        candidate.validate_reference_answer(
            reference_answer,
            private_evaluation=private_evaluation,
            evidence_manifest=evidence_manifest,
            private_evaluation_file_sha256=candidate.sha256_value(private_evaluation),
            evidence_manifest_file_sha256=candidate.sha256_value(evidence_manifest),
        )
        candidate.validate_stage_plan_contract(templates["s1"])
        candidate.validate_evidence_service_contract(templates["s2"])
        candidate.validate_execution_contract(templates["s3"])
        candidate.validate_execution_contract(templates["s4"])
        candidate.validate_submission_contract(templates["s5"])
    except Exception as exc:
        raise PremiumConstructionSupervisorError("constructed legacy artifacts failed validation") from exc
    runtime_policy = deepcopy(dict(policy))
    runtime_policy["target_split"] = selected.split
    try:
        candidate.validate_policy(runtime_policy)
        projected = authority.modules.panel_dispatch._panelize_constructor_templates(
            base_policy=runtime_policy,
            templates=templates,
        )
        authority.modules.panel_dispatch._validate_contract_templates(
            base_policy=runtime_policy,
            private_evaluation=private_evaluation,
            s1_template=projected["s1"],
            s2_template=projected["s2"],
            s3_template=projected["s3"],
            s4_template=projected["s4"],
            s5_template=projected["s5"],
        )
    except Exception as exc:
        raise PremiumConstructionSupervisorError("promoted runtime contracts are not executable") from exc
    input_rows: dict[str, Mapping[str, Any]] = {}
    for stage in ("s3", "s4"):
        inputs = projected[stage].get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise PremiumConstructionSupervisorError("runtime input inventory differs")
        for item in inputs:
            if not isinstance(item, Mapping) or set(item) != {
                "input_id",
                "relative_path",
                "byte_count",
                "sha256",
            }:
                raise PremiumConstructionSupervisorError("runtime input row differs")
            prior = input_rows.setdefault(item["relative_path"], item)
            if prior != item:
                raise PremiumConstructionSupervisorError(
                    "S3/S4 input inventories conflict"
                )
    evidence_refs = {
        value["path"]: value
        for name, value in files.items()
        if isinstance(name, str)
        and name.startswith("evidence:")
        and isinstance(value, Mapping)
        and isinstance(value.get("path"), str)
    }
    if set(input_rows) != set(evidence_refs):
        raise PremiumConstructionSupervisorError(
            "runtime inputs differ from published evidence"
        )
    for relative, item in input_rows.items():
        _path, _document, evidence_bytes = _publication_artifact(
            publication_root,
            evidence_refs[relative],
            label=f"runtime input {relative}",
        )
        if not (
            len(evidence_bytes) == item["byte_count"]
            and hashlib.sha256(evidence_bytes).hexdigest() == item["sha256"]
        ):
            raise PremiumConstructionSupervisorError(
                "runtime input contract bytes differ"
            )
    image_digest = projected["s3"].get("image_digest")
    if image_digest not in authority.image_refs or projected["s4"].get("image_digest") != image_digest:
        raise PremiumConstructionSupervisorError("runtime image digest is not allowlisted")
    tools = policy.get("tools")
    if (
        not isinstance(tools, list)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"name", "description", "input_schema"}
            or not isinstance(row["name"], str)
            or not isinstance(row["description"], str)
            or not isinstance(row["input_schema"], Mapping)
            for row in tools
        )
        or tuple(sorted(row["name"] for row in tools)) != EXPECTED_TOOL_NAMES
    ):
        raise PremiumConstructionSupervisorError("constructed public tool policy differs")
    tool_catalog = tuple(
        CandidateToolCatalogEntry(
            name=row["name"],
            description=row["description"],
            input_schema=row["input_schema"],
        )
        for row in sorted(tools, key=lambda item: item["name"])
    )
    tool_catalog_blake3 = blake3_hex([row.to_document() for row in tool_catalog])
    return _ExecutableMaterial(
        selected=selected,
        queued=queued,
        record=record,
        catalog_binding=freeze_json(catalog_binding),
        publication_root=publication_root,
        executable_binding_path=binding_path,
        executable_binding_blake3=receipt.artifact_blake3,
        source_policy_path=loaded_paths["policy"],
        source_policy_blake3=blake3_bytes(loaded_bytes["policy"]),
        source_policy=freeze_json(policy),
        runtime_policy=freeze_json(runtime_policy),
        runtime_policy_blake3=blake3_hex(runtime_policy),
        templates=MappingProxyType({key: freeze_json(value) for key, value in projected.items()}),
        tool_catalog=tool_catalog,
        tool_catalog_blake3=tool_catalog_blake3,
        source_request_blake3=source_request_blake3,
    )


def _transition_binding_row(material: _ExecutableMaterial) -> Mapping[str, JsonValue]:
    selected = material.selected
    return freeze_json(
        {
            "candidate_id": selected.candidate_id,
            "source_candidate_id": selected.source_candidate_id,
            "source_family": selected.source_family,
            "domain": selected.domain,
            "stage": selected.stage,
            "split": selected.split,
            "rubric_id": selected.rubric_id,
            "rubric_blake3": selected.rubric_blake3,
            "selection_entry_blake3": blake3_hex(selected.to_document()),
            "queue_entry_blake3": material.queued.entry_blake3,
            "readiness_proof_root_blake3": selected.readiness_proof_root_blake3,
            "catalog_binding": material.catalog_binding,
            "source_request_blake3": material.source_request_blake3,
            "source_policy_path": (
                f"{material.record.publication_receipt['publication_id']}/"
                + material.source_policy_path.relative_to(material.publication_root).as_posix()
            ),
            "source_policy_blake3": material.source_policy_blake3,
            "runtime_policy_blake3": material.runtime_policy_blake3,
            "executable_binding_path": (
                f"{material.record.publication_receipt['publication_id']}/executable-binding.v1.json"
            ),
            "executable_binding_file_blake3": material.executable_binding_blake3,
            "tool_catalog_blake3": material.tool_catalog_blake3,
        }
    )


def _materialize_runtime_inputs(
    material: _ExecutableMaterial,
    *,
    runtime_state_root: Path,
    authority: _RuntimeAuthority,
) -> Path:
    """Copy only contract-declared input bytes into one sealed runtime tree."""

    rows_by_path: dict[str, Mapping[str, Any]] = {}
    for stage in ("s3", "s4"):
        for row in material.templates[stage]["inputs"]:
            prior = rows_by_path.setdefault(row["relative_path"], row)
            if prior != row:
                raise PremiumConstructionSupervisorError(
                    "runtime input inventories changed after transition verification"
                )
    materialization_blake3 = blake3_hex(
        {
            "schema": "eva.premium-runtime-input-materialization.v1",
            "candidate_id": material.selected.candidate_id,
            "executable_binding_file_blake3": material.executable_binding_blake3,
            "inputs": [rows_by_path[path] for path in sorted(rows_by_path)],
        }
    )
    name = f"inputs-{materialization_blake3}"
    created = False
    try:
        root = _mkdir_new_child(runtime_state_root, name)
        created = True
    except LegacyConstructionAdapterError as exc:
        if "already exists" not in str(exc):
            raise PremiumConstructionSupervisorError(
                "runtime input root cannot be created safely"
            ) from exc
        root = _existing_directory(
            runtime_state_root / name, label="existing runtime input root"
        )
    if created:
        for relative, row in sorted(rows_by_path.items()):
            payload = _read_relative(
                material.publication_root,
                _safe_relative(relative, label="published runtime input"),
                label="published runtime input",
                maximum_bytes=_MAXIMUM_DOCUMENT_BYTES,
            )
            if not (
                len(payload) == row["byte_count"]
                and hashlib.sha256(payload).hexdigest() == row["sha256"]
            ):
                raise PremiumConstructionSupervisorError(
                    "published runtime input changed before materialization"
                )
            _write_new_relative(root, relative, payload)
        _seal_tree(root)
    try:
        authority.modules.execution._verify_input_inventory(
            material.templates["s3"], root
        )
        authority.modules.execution._verify_input_inventory(
            material.templates["s4"], root
        )
    except Exception as exc:
        raise PremiumConstructionSupervisorError(
            "sealed runtime input materialization differs"
        ) from exc
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PremiumConstructionSupervisorError(
                "sealed runtime input materialization contains a symlink"
            )
        expected_mode = 0o500 if stat.S_ISDIR(metadata.st_mode) else 0o400
        if stat.S_IMODE(metadata.st_mode) != expected_mode:
            raise PremiumConstructionSupervisorError(
                "sealed runtime input materialization mode differs"
            )
    return root


def _source_payloads(
    selection: CampaignSelectionV2,
    queue: PremiumConstructionQueue,
    catalog: PremiumConstructionProofCatalog,
    supervisor_input: Mapping[str, JsonValue],
) -> Mapping[str, bytes]:
    return MappingProxyType(
        {
            "selection": canonical_json_bytes(selection.to_document()),
            "queue": canonical_json_bytes(queue.to_document()),
            "proof_catalog": canonical_json_bytes(catalog.to_document()),
            "supervisor_input": canonical_json_bytes(_plain(supervisor_input)),
        }
    )


def _transition_payload(
    *,
    transition_id: str,
    issued_at_utc: str,
    selection: CampaignSelectionV2,
    queue: PremiumConstructionQueue,
    catalog: PremiumConstructionProofCatalog,
    supervisor_input: Mapping[str, JsonValue],
    source_payloads: Mapping[str, bytes],
    bindings: tuple[Mapping[str, JsonValue], ...],
    authority: _RuntimeAuthority,
) -> Mapping[str, JsonValue]:
    return freeze_json(
        {
            "schema": TRANSITION_SCHEMA,
            "transition_id": transition_id,
            "issued_at_utc": issued_at_utc,
            "from_state": "frozen",
            "to_state": "executable_ready",
            "selection_id": selection.selection_id,
            "selection_blake3": selection.selection_blake3,
            "queue_id": queue.queue_id,
            "queue_blake3": queue.queue_blake3,
            "catalog_id": catalog.catalog_id,
            "catalog_blake3": catalog.catalog_blake3,
            "supervisor_input_blake3": blake3_hex(supervisor_input),
            "readiness_authority_blake3": queue.readiness_authority_blake3,
            "rubric_registry_blake3": selection.rubric_registry_blake3,
            "legacy_runtime_source_blake3": authority.modules.source_blake3,
            "legacy_validator_authority_blake3": authority.validator.identity.authority_blake3,
            "source_documents": {
                name: _file_reference(_SOURCE_FILES[name], source_payloads[name])
                for name in sorted(_SOURCE_FILES)
            },
            "bindings": bindings,
            "binding_count": len(bindings),
            "controls": {
                "supervisor_transition_authorized": True,
                "admission_eligibility_claimed": False,
                "provider_calls": 0,
                "ledger_writes": 0,
                "append_only": True,
                "source_schema_bytes_mutated": False,
                "runtime_policy_projection": "target_split_only",
                "real_handler_prerequisites_verified": True,
            },
        }
    )


def _canonical_selected_candidate_ids_v2(
    value: Any,
    *,
    successful_records: Mapping[str, PremiumConstructionTerminalRecord],
    selected_by_id: Mapping[str, SelectionEntryV2],
    queued_by_id: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate one explicit set and return its queue-canonical ordering."""

    if not isinstance(value, (tuple, list)) or not value:
        raise PremiumConstructionSupervisorError(
            "v2 selected candidate IDs must be an explicit nonempty sequence"
        )
    candidate_ids = tuple(
        uuid_text(candidate_id, label="v2 selected candidate_id")
        for candidate_id in value
    )
    if len(set(candidate_ids)) != len(candidate_ids):
        raise PremiumConstructionSupervisorError(
            "v2 selected candidate IDs are duplicated"
        )
    if any(
        candidate_id not in successful_records
        or candidate_id not in selected_by_id
        or candidate_id not in queued_by_id
        for candidate_id in candidate_ids
    ):
        raise PremiumConstructionSupervisorError(
            "v2 selected candidate ID is not an exact successful catalog row"
        )
    return tuple(
        sorted(
            candidate_ids,
            key=lambda candidate_id: queued_by_id[candidate_id].campaign_ordinal,
        )
    )


def _selected_candidate_ids_blake3_v2(candidate_ids: tuple[str, ...]) -> str:
    return blake3_hex(
        {
            "schema": SELECTED_IDS_SCHEMA_V2,
            "candidate_ids": candidate_ids,
        }
    )


def _claim_domain_blake3_v2(
    *, catalog_blake3: str, selected_candidate_ids_blake3: str
) -> str:
    return blake3_hex(
        {
            "schema": CLAIM_DOMAIN_SCHEMA_V2,
            "catalog_blake3": catalog_blake3,
            "selected_candidate_ids_blake3": selected_candidate_ids_blake3,
        }
    )


def _transition_payload_v2(
    *,
    transition_id: str,
    issued_at_utc: str,
    selection: CampaignSelectionV2,
    queue: PremiumConstructionQueue,
    catalog: PremiumConstructionProofCatalog,
    supervisor_input: Mapping[str, JsonValue],
    source_payloads: Mapping[str, bytes],
    selected_candidate_ids: tuple[str, ...],
    bindings: tuple[Mapping[str, JsonValue], ...],
    authority: _RuntimeAuthority,
) -> Mapping[str, JsonValue]:
    selected_ids_blake3 = _selected_candidate_ids_blake3_v2(
        selected_candidate_ids
    )
    claim_domain_blake3 = _claim_domain_blake3_v2(
        catalog_blake3=catalog.catalog_blake3,
        selected_candidate_ids_blake3=selected_ids_blake3,
    )
    return freeze_json(
        {
            "schema": TRANSITION_SCHEMA_V2,
            "transition_id": transition_id,
            "issued_at_utc": issued_at_utc,
            "from_state": "frozen",
            "to_state": "executable_ready",
            "selection_id": selection.selection_id,
            "selection_blake3": selection.selection_blake3,
            "queue_id": queue.queue_id,
            "queue_blake3": queue.queue_blake3,
            "catalog_id": catalog.catalog_id,
            "catalog_blake3": catalog.catalog_blake3,
            "supervisor_input_blake3": blake3_hex(supervisor_input),
            "selected_candidate_ids": selected_candidate_ids,
            "selected_candidate_ids_blake3": selected_ids_blake3,
            "claim_domain_blake3": claim_domain_blake3,
            "readiness_authority_blake3": queue.readiness_authority_blake3,
            "rubric_registry_blake3": selection.rubric_registry_blake3,
            "legacy_runtime_source_blake3": authority.modules.source_blake3,
            "legacy_validator_authority_blake3": authority.validator.identity.authority_blake3,
            "source_documents": {
                name: _file_reference(_SOURCE_FILES[name], source_payloads[name])
                for name in sorted(_SOURCE_FILES)
            },
            "bindings": bindings,
            "binding_count": len(bindings),
            "controls": {
                "supervisor_transition_authorized": True,
                "admission_eligibility_claimed": False,
                "provider_calls": 0,
                "ledger_writes": 0,
                "append_only": True,
                "source_schema_bytes_mutated": False,
                "runtime_policy_projection": "target_split_only",
                "real_handler_prerequisites_verified": True,
                "proof_catalog_coverage": "complete_cumulative",
                "transition_selection": "explicit_successful_candidate_set",
            },
        }
    )


def _read_exact_claim_v2(root: Path, relative: str, *, label: str) -> Mapping[str, Any]:
    payload = _read_relative(
        root,
        _safe_relative(relative, label=label),
        label=label,
        maximum_bytes=_MAXIMUM_DOCUMENT_BYTES,
    )
    path = root / relative
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o400
    ):
        raise PremiumConstructionSupervisorError(f"{label} boundary differs")
    return _decode_object(payload, label=label)


@dataclass(frozen=True, slots=True)
class PremiumConstructionSupervisorTransition:
    root: Path
    envelope: SignedEnvelope
    binding_count: int

    @property
    def transition_id(self) -> str:
        return str(self.envelope.payload["transition_id"])

    @property
    def transition_blake3(self) -> str:
        return self.envelope.envelope_blake3


def issue_premium_construction_supervisor_transition(
    catalog: PremiumConstructionProofCatalog,
    *,
    queue: PremiumConstructionQueue,
    selection: CampaignSelectionV2,
    rubrics: CompiledRubricRegistry,
    publication_output_root: str | Path,
    transition_output_root: str | Path,
    legacy_python_root: str | Path,
    trust_store_path: str | Path,
    image_refs_path: str | Path,
    host_private_key_path: str | Path,
    host_key_id: str,
    docker_binary: str = "docker",
    id_factory: RuntimeIdFactory | None = None,
    clock: Callable[[], str] | None = None,
) -> PremiumConstructionSupervisorTransition:
    """Validate, sign, and O_EXCL-publish one replenishment transition."""

    if not isinstance(catalog, PremiumConstructionProofCatalog):
        raise PremiumConstructionSupervisorError("premium proof catalog type differs")
    selected_by_id, queued_by_id = _entry_binding(
        selection=selection, queue=queue, rubrics=rubrics
    )
    publication_root = _existing_directory(
        publication_output_root, label="premium publication output root"
    )
    supervisor_input = verify_premium_construction_proof_catalog(
        catalog,
        queue=queue,
        publication_output_root=publication_root,
    )
    authority = _runtime_authority(
        legacy_python_root=legacy_python_root,
        trust_store_path=trust_store_path,
        image_refs_path=image_refs_path,
        host_private_key_path=host_private_key_path,
        host_key_id=host_key_id,
        docker_binary=docker_binary,
    )
    catalog_by_id = {
        row["candidate_id"]: row for row in supervisor_input["successful_bindings"]
    }
    records = {
        record.candidate_id: record
        for record in catalog.records
        if record.status == "succeeded"
    }
    if not records or set(catalog_by_id) != set(records):
        raise PremiumConstructionSupervisorError("successful catalog identity inventory differs")
    materials = tuple(
        _open_executable_material(
            selected=selected_by_id[candidate_id],
            queued=queued_by_id[candidate_id],
            record=records[candidate_id],
            catalog_binding=catalog_by_id[candidate_id],
            selection_id=selection.selection_id,
            publication_output_root=publication_root,
            authority=authority,
        )
        for candidate_id in sorted(
            records, key=lambda value: queued_by_id[value].campaign_ordinal
        )
    )
    bindings = tuple(_transition_binding_row(material) for material in materials)
    factory = id_factory or RandomUUIDFactory()
    transition_id = uuid_text(
        factory.new("premium-construction-supervisor-transition"),
        label="premium transition_id",
    )
    issued_at = _utc(
        (clock or (lambda: datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")))()
    )
    payloads = _source_payloads(selection, queue, catalog, supervisor_input)
    payload = _transition_payload(
        transition_id=transition_id,
        issued_at_utc=issued_at,
        selection=selection,
        queue=queue,
        catalog=catalog,
        supervisor_input=supervisor_input,
        source_payloads=payloads,
        bindings=bindings,
        authority=authority,
    )
    envelope = issue_signed_envelope(
        payload,
        key_id=host_key_id,
        private_key_path=Path(host_private_key_path),
    )
    output_root = _ensure_directory(
        transition_output_root, label="premium transition output root"
    )
    _ensure_directory(output_root / ".claims", label="premium transition claim root")
    _ensure_directory(
        output_root / ".candidate-claims", label="premium candidate claim root"
    )
    claim = {
        "schema": CLAIM_SCHEMA,
        "transition_id": transition_id,
        "catalog_blake3": catalog.catalog_blake3,
        "candidate_ids": [material.selected.candidate_id for material in materials],
        "signed_envelope_blake3": envelope.envelope_blake3,
        "append_only": True,
    }
    _write_new_relative(
        output_root,
        f".claims/{catalog.catalog_blake3}.json",
        canonical_json_bytes(claim),
    )
    for material in materials:
        _write_new_relative(
            output_root,
            f".candidate-claims/{material.selected.candidate_id}.json",
            canonical_json_bytes(
                {
                    "schema": CLAIM_SCHEMA,
                    "transition_id": transition_id,
                    "candidate_id": material.selected.candidate_id,
                    "source_candidate_id": material.selected.source_candidate_id,
                    "catalog_blake3": catalog.catalog_blake3,
                    "append_only": True,
                }
            ),
        )
    transition_root = _mkdir_new_child(output_root, transition_id)
    for name, relative in _SOURCE_FILES.items():
        _write_new_relative(transition_root, relative, payloads[name])
    _write_new_relative(
        transition_root,
        "signed-transition.v1.json",
        canonical_json_bytes(envelope.to_document()),
    )
    _seal_tree(transition_root)
    return verify_premium_construction_supervisor_transition(
        transition_root,
        publication_output_root=publication_root,
        rubrics=rubrics,
        legacy_python_root=legacy_python_root,
        trust_store_path=trust_store_path,
        image_refs_path=image_refs_path,
        host_private_key_path=host_private_key_path,
        host_key_id=host_key_id,
        docker_binary=docker_binary,
    )


def verify_premium_construction_supervisor_transition(
    transition_root: str | Path,
    *,
    publication_output_root: str | Path,
    rubrics: CompiledRubricRegistry,
    legacy_python_root: str | Path,
    trust_store_path: str | Path,
    image_refs_path: str | Path,
    host_private_key_path: str | Path,
    host_key_id: str,
    docker_binary: str = "docker",
) -> PremiumConstructionSupervisorTransition:
    """Independently reopen signatures, source proofs, artifacts, and handlers."""

    root = _existing_directory(transition_root, label="premium transition bundle")
    entries = tuple(root.rglob("*"))
    observed_files = sorted(
        path.relative_to(root).as_posix()
        for path in entries
        if path.is_file()
    )
    expected_files = sorted([*_SOURCE_FILES.values(), "signed-transition.v1.json"])
    if (
        stat.S_IMODE(root.lstat().st_mode) != 0o500
        or observed_files != expected_files
        or any(
            path.is_symlink()
            or not path.is_file()
            or path.lstat().st_nlink != 1
            or stat.S_IMODE(path.lstat().st_mode) != 0o400
            for path in entries
        )
    ):
        raise PremiumConstructionSupervisorError("premium transition bundle inventory differs")
    envelope_document, _ = _read_document(
        root, "signed-transition.v1.json", label="signed premium transition"
    )
    envelope = SignedEnvelope.from_document(envelope_document)
    verify_signed_envelope(envelope, trust_store_path=Path(trust_store_path))
    payload = envelope.payload
    if payload.get("schema") != TRANSITION_SCHEMA:
        raise PremiumConstructionSupervisorError("premium transition payload schema differs")
    uuid_text(payload.get("transition_id"), label="premium transition_id")
    _utc(payload.get("issued_at_utc"))
    if root.name != payload["transition_id"]:
        raise PremiumConstructionSupervisorError("transition directory identity differs")
    references = payload.get("source_documents")
    if not isinstance(references, Mapping) or set(references) != set(_SOURCE_FILES):
        raise PremiumConstructionSupervisorError("transition source-document inventory differs")
    reopened: dict[str, Mapping[str, Any]] = {}
    source_payloads: dict[str, bytes] = {}
    for name, relative in _SOURCE_FILES.items():
        reopened[name], source_payloads[name] = _verify_file_reference(
            root,
            references[name],
            expected_path=relative,
            label=f"transition {name}",
        )
    selection = CampaignSelectionV2.from_document(reopened["selection"])
    queue = PremiumConstructionQueue.from_document(reopened["queue"])
    catalog = PremiumConstructionProofCatalog.from_document(reopened["proof_catalog"])
    selected_by_id, queued_by_id = _entry_binding(
        selection=selection, queue=queue, rubrics=rubrics
    )
    publication_root = _existing_directory(
        publication_output_root, label="premium publication output root"
    )
    expected_supervisor_input = verify_premium_construction_proof_catalog(
        catalog,
        queue=queue,
        publication_output_root=publication_root,
    )
    if _plain(reopened["supervisor_input"]) != _plain(expected_supervisor_input):
        raise PremiumConstructionSupervisorError("supervisor input projection differs")
    authority = _runtime_authority(
        legacy_python_root=legacy_python_root,
        trust_store_path=trust_store_path,
        image_refs_path=image_refs_path,
        host_private_key_path=host_private_key_path,
        host_key_id=host_key_id,
        docker_binary=docker_binary,
    )
    catalog_by_id = {
        row["candidate_id"]: row
        for row in expected_supervisor_input["successful_bindings"]
    }
    records = {
        record.candidate_id: record
        for record in catalog.records
        if record.status == "succeeded"
    }
    if not records or set(catalog_by_id) != set(records):
        raise PremiumConstructionSupervisorError(
            "successful catalog identity inventory differs"
        )
    materials = tuple(
        _open_executable_material(
            selected=selected_by_id[candidate_id],
            queued=queued_by_id[candidate_id],
            record=records[candidate_id],
            catalog_binding=catalog_by_id[candidate_id],
            selection_id=selection.selection_id,
            publication_output_root=publication_root,
            authority=authority,
        )
        for candidate_id in sorted(
            records, key=lambda value: queued_by_id[value].campaign_ordinal
        )
    )
    bindings = tuple(_transition_binding_row(material) for material in materials)
    expected_payload = _transition_payload(
        transition_id=payload["transition_id"],
        issued_at_utc=payload["issued_at_utc"],
        selection=selection,
        queue=queue,
        catalog=catalog,
        supervisor_input=expected_supervisor_input,
        source_payloads=source_payloads,
        bindings=bindings,
        authority=authority,
    )
    if canonical_value(payload) != canonical_value(expected_payload):
        raise PremiumConstructionSupervisorError("premium supervisor transition claims differ")
    return PremiumConstructionSupervisorTransition(
        root=root,
        envelope=envelope,
        binding_count=len(materials),
    )


def issue_premium_construction_supervisor_transition_v2(
    catalog: PremiumConstructionProofCatalog,
    *,
    selected_candidate_ids: tuple[str, ...],
    queue: PremiumConstructionQueue,
    selection: CampaignSelectionV2,
    rubrics: CompiledRubricRegistry,
    publication_output_root: str | Path,
    transition_output_root: str | Path,
    legacy_python_root: str | Path,
    trust_store_path: str | Path,
    image_refs_path: str | Path,
    host_private_key_path: str | Path,
    host_key_id: str,
    docker_binary: str = "docker",
    id_factory: RuntimeIdFactory | None = None,
    clock: Callable[[], str] | None = None,
) -> PremiumConstructionSupervisorTransition:
    """Publish an incremental transition over a cumulative proof catalog.

    The complete catalog and its complete successful-binding projection are
    embedded as source documents.  Only the explicit selected set becomes a
    transition binding, and global candidate claims prevent a successful row
    from being transitioned again through a later cumulative catalog.
    """

    if not isinstance(catalog, PremiumConstructionProofCatalog):
        raise PremiumConstructionSupervisorError("premium proof catalog type differs")
    selected_by_id, queued_by_id = _entry_binding(
        selection=selection, queue=queue, rubrics=rubrics
    )
    publication_root = _existing_directory(
        publication_output_root, label="premium publication output root"
    )
    supervisor_input = verify_premium_construction_proof_catalog(
        catalog,
        queue=queue,
        publication_output_root=publication_root,
    )
    catalog_by_id = {
        row["candidate_id"]: row
        for row in supervisor_input["successful_bindings"]
    }
    records = {
        record.candidate_id: record
        for record in catalog.records
        if record.status == "succeeded"
    }
    if not records or set(catalog_by_id) != set(records):
        raise PremiumConstructionSupervisorError(
            "successful catalog identity inventory differs"
        )
    selected_ids = _canonical_selected_candidate_ids_v2(
        selected_candidate_ids,
        successful_records=records,
        selected_by_id=selected_by_id,
        queued_by_id=queued_by_id,
    )
    authority = _runtime_authority(
        legacy_python_root=legacy_python_root,
        trust_store_path=trust_store_path,
        image_refs_path=image_refs_path,
        host_private_key_path=host_private_key_path,
        host_key_id=host_key_id,
        docker_binary=docker_binary,
    )
    materials = tuple(
        _open_executable_material(
            selected=selected_by_id[candidate_id],
            queued=queued_by_id[candidate_id],
            record=records[candidate_id],
            catalog_binding=catalog_by_id[candidate_id],
            selection_id=selection.selection_id,
            publication_output_root=publication_root,
            authority=authority,
        )
        for candidate_id in selected_ids
    )
    bindings = tuple(_transition_binding_row(material) for material in materials)
    factory = id_factory or RandomUUIDFactory()
    transition_id = uuid_text(
        factory.new("premium-construction-supervisor-transition-v2"),
        label="premium transition_id",
    )
    issued_at = _utc(
        (clock or (lambda: datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")))()
    )
    payloads = _source_payloads(selection, queue, catalog, supervisor_input)
    payload = _transition_payload_v2(
        transition_id=transition_id,
        issued_at_utc=issued_at,
        selection=selection,
        queue=queue,
        catalog=catalog,
        supervisor_input=supervisor_input,
        source_payloads=payloads,
        selected_candidate_ids=selected_ids,
        bindings=bindings,
        authority=authority,
    )
    envelope = issue_signed_envelope(
        payload,
        key_id=host_key_id,
        private_key_path=Path(host_private_key_path),
    )
    output_root = _ensure_directory(
        transition_output_root, label="premium transition output root"
    )
    _ensure_directory(output_root / ".claims", label="premium transition claim root")
    _ensure_directory(
        output_root / ".candidate-claims", label="premium candidate claim root"
    )
    claim_domain_blake3 = str(payload["claim_domain_blake3"])
    domain_relative = f".claims/{claim_domain_blake3}.json"
    candidate_relatives = tuple(
        f".candidate-claims/{candidate_id}.json" for candidate_id in selected_ids
    )
    for relative in (*candidate_relatives, domain_relative, transition_id):
        target = output_root / _safe_relative(relative, label="v2 transition publication")
        try:
            target.lstat()
        except FileNotFoundError:
            continue
        raise PremiumConstructionSupervisorError(
            "v2 transition or candidate has already been claimed"
        )
    domain_claim = {
        "schema": CLAIM_SCHEMA_V2,
        "claim_domain_blake3": claim_domain_blake3,
        "transition_id": transition_id,
        "catalog_id": catalog.catalog_id,
        "catalog_blake3": catalog.catalog_blake3,
        "selected_candidate_ids": selected_ids,
        "selected_candidate_ids_blake3": payload[
            "selected_candidate_ids_blake3"
        ],
        "signed_envelope_blake3": envelope.envelope_blake3,
        "append_only": True,
    }
    try:
        for material, relative in zip(materials, candidate_relatives, strict=True):
            _write_new_relative(
                output_root,
                relative,
                canonical_json_bytes(
                    {
                        "schema": CLAIM_SCHEMA_V2,
                        "claim_domain_blake3": claim_domain_blake3,
                        "transition_id": transition_id,
                        "candidate_id": material.selected.candidate_id,
                        "source_candidate_id": material.selected.source_candidate_id,
                        "catalog_blake3": catalog.catalog_blake3,
                        "selected_candidate_ids_blake3": payload[
                            "selected_candidate_ids_blake3"
                        ],
                        "signed_envelope_blake3": envelope.envelope_blake3,
                        "append_only": True,
                    }
                ),
            )
        _write_new_relative(
            output_root,
            domain_relative,
            canonical_json_bytes(domain_claim),
        )
        transition_root = _mkdir_new_child(output_root, transition_id)
        for name, relative in _SOURCE_FILES.items():
            _write_new_relative(transition_root, relative, payloads[name])
        _write_new_relative(
            transition_root,
            "signed-transition.v2.json",
            canonical_json_bytes(envelope.to_document()),
        )
        _seal_tree(transition_root)
    except LegacyConstructionAdapterError as exc:
        raise PremiumConstructionSupervisorError(
            "v2 append-only transition publication failed closed"
        ) from exc
    return verify_premium_construction_supervisor_transition_v2(
        transition_root,
        publication_output_root=publication_root,
        rubrics=rubrics,
        legacy_python_root=legacy_python_root,
        trust_store_path=trust_store_path,
        image_refs_path=image_refs_path,
        host_private_key_path=host_private_key_path,
        host_key_id=host_key_id,
        docker_binary=docker_binary,
    )


def verify_premium_construction_supervisor_transition_v2(
    transition_root: str | Path,
    *,
    publication_output_root: str | Path,
    rubrics: CompiledRubricRegistry,
    legacy_python_root: str | Path,
    trust_store_path: str | Path,
    image_refs_path: str | Path,
    host_private_key_path: str | Path,
    host_key_id: str,
    docker_binary: str = "docker",
) -> PremiumConstructionSupervisorTransition:
    """Reopen a v2 incremental transition and its append-only claims."""

    root = _existing_directory(transition_root, label="premium transition bundle")
    entries = tuple(root.rglob("*"))
    observed_files = sorted(
        path.relative_to(root).as_posix() for path in entries if path.is_file()
    )
    expected_files = sorted([*_SOURCE_FILES.values(), "signed-transition.v2.json"])
    if (
        stat.S_IMODE(root.lstat().st_mode) != 0o500
        or observed_files != expected_files
        or any(
            path.is_symlink()
            or not path.is_file()
            or path.lstat().st_nlink != 1
            or stat.S_IMODE(path.lstat().st_mode) != 0o400
            for path in entries
        )
    ):
        raise PremiumConstructionSupervisorError(
            "premium v2 transition bundle inventory differs"
        )
    envelope_document, _ = _read_document(
        root, "signed-transition.v2.json", label="signed premium v2 transition"
    )
    envelope = SignedEnvelope.from_document(envelope_document)
    verify_signed_envelope(envelope, trust_store_path=Path(trust_store_path))
    payload = envelope.payload
    if payload.get("schema") != TRANSITION_SCHEMA_V2:
        raise PremiumConstructionSupervisorError(
            "premium v2 transition payload schema differs"
        )
    uuid_text(payload.get("transition_id"), label="premium transition_id")
    _utc(payload.get("issued_at_utc"))
    if root.name != payload["transition_id"]:
        raise PremiumConstructionSupervisorError(
            "transition directory identity differs"
        )
    references = payload.get("source_documents")
    if not isinstance(references, Mapping) or set(references) != set(_SOURCE_FILES):
        raise PremiumConstructionSupervisorError(
            "transition source-document inventory differs"
        )
    reopened: dict[str, Mapping[str, Any]] = {}
    source_payloads: dict[str, bytes] = {}
    for name, relative in _SOURCE_FILES.items():
        reopened[name], source_payloads[name] = _verify_file_reference(
            root,
            references[name],
            expected_path=relative,
            label=f"transition {name}",
        )
    selection = CampaignSelectionV2.from_document(reopened["selection"])
    queue = PremiumConstructionQueue.from_document(reopened["queue"])
    catalog = PremiumConstructionProofCatalog.from_document(
        reopened["proof_catalog"]
    )
    selected_by_id, queued_by_id = _entry_binding(
        selection=selection, queue=queue, rubrics=rubrics
    )
    publication_root = _existing_directory(
        publication_output_root, label="premium publication output root"
    )
    expected_supervisor_input = verify_premium_construction_proof_catalog(
        catalog,
        queue=queue,
        publication_output_root=publication_root,
    )
    if _plain(reopened["supervisor_input"]) != _plain(expected_supervisor_input):
        raise PremiumConstructionSupervisorError("supervisor input projection differs")
    catalog_by_id = {
        row["candidate_id"]: row
        for row in expected_supervisor_input["successful_bindings"]
    }
    records = {
        record.candidate_id: record
        for record in catalog.records
        if record.status == "succeeded"
    }
    if not records or set(catalog_by_id) != set(records):
        raise PremiumConstructionSupervisorError(
            "successful catalog identity inventory differs"
        )
    selected_ids = _canonical_selected_candidate_ids_v2(
        payload.get("selected_candidate_ids"),
        successful_records=records,
        selected_by_id=selected_by_id,
        queued_by_id=queued_by_id,
    )
    authority = _runtime_authority(
        legacy_python_root=legacy_python_root,
        trust_store_path=trust_store_path,
        image_refs_path=image_refs_path,
        host_private_key_path=host_private_key_path,
        host_key_id=host_key_id,
        docker_binary=docker_binary,
    )
    materials = tuple(
        _open_executable_material(
            selected=selected_by_id[candidate_id],
            queued=queued_by_id[candidate_id],
            record=records[candidate_id],
            catalog_binding=catalog_by_id[candidate_id],
            selection_id=selection.selection_id,
            publication_output_root=publication_root,
            authority=authority,
        )
        for candidate_id in selected_ids
    )
    bindings = tuple(_transition_binding_row(material) for material in materials)
    expected_payload = _transition_payload_v2(
        transition_id=payload["transition_id"],
        issued_at_utc=payload["issued_at_utc"],
        selection=selection,
        queue=queue,
        catalog=catalog,
        supervisor_input=expected_supervisor_input,
        source_payloads=source_payloads,
        selected_candidate_ids=selected_ids,
        bindings=bindings,
        authority=authority,
    )
    if canonical_value(payload) != canonical_value(expected_payload):
        raise PremiumConstructionSupervisorError(
            "premium v2 supervisor transition claims differ"
        )
    output_root = _existing_directory(
        root.parent, label="premium transition output root"
    )
    claim_domain_blake3 = str(payload["claim_domain_blake3"])
    observed_domain_claim = _read_exact_claim_v2(
        output_root,
        f".claims/{claim_domain_blake3}.json",
        label="premium v2 claim domain",
    )
    expected_domain_claim = {
        "schema": CLAIM_SCHEMA_V2,
        "claim_domain_blake3": claim_domain_blake3,
        "transition_id": payload["transition_id"],
        "catalog_id": catalog.catalog_id,
        "catalog_blake3": catalog.catalog_blake3,
        "selected_candidate_ids": selected_ids,
        "selected_candidate_ids_blake3": payload[
            "selected_candidate_ids_blake3"
        ],
        "signed_envelope_blake3": envelope.envelope_blake3,
        "append_only": True,
    }
    if canonical_value(observed_domain_claim) != canonical_value(
        expected_domain_claim
    ):
        raise PremiumConstructionSupervisorError(
            "premium v2 claim domain differs"
        )
    for material in materials:
        observed_candidate_claim = _read_exact_claim_v2(
            output_root,
            f".candidate-claims/{material.selected.candidate_id}.json",
            label="premium v2 candidate claim",
        )
        expected_candidate_claim = {
            "schema": CLAIM_SCHEMA_V2,
            "claim_domain_blake3": claim_domain_blake3,
            "transition_id": payload["transition_id"],
            "candidate_id": material.selected.candidate_id,
            "source_candidate_id": material.selected.source_candidate_id,
            "catalog_blake3": catalog.catalog_blake3,
            "selected_candidate_ids_blake3": payload[
                "selected_candidate_ids_blake3"
            ],
            "signed_envelope_blake3": envelope.envelope_blake3,
            "append_only": True,
        }
        if canonical_value(observed_candidate_claim) != canonical_value(
            expected_candidate_claim
        ):
            raise PremiumConstructionSupervisorError(
                "premium v2 candidate claim differs"
            )
    return PremiumConstructionSupervisorTransition(
        root=root,
        envelope=envelope,
        binding_count=len(materials),
    )


@dataclass(frozen=True, slots=True)
class PremiumCandidateBindingReference:
    candidate_id: str
    source_candidate_id: str
    source_family: str
    domain: str
    stage: Stage
    split: str
    rubric_id: str
    rubric_blake3: str
    publication_id: str
    binding_blake3: str
    transition_binding_blake3: str

    def to_document(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_candidate_id": self.source_candidate_id,
            "source_family": self.source_family,
            "domain": self.domain,
            "stage": self.stage.value,
            "split": self.split,
            "rubric_id": self.rubric_id,
            "rubric_blake3": self.rubric_blake3,
            "publication_id": self.publication_id,
            "binding_blake3": self.binding_blake3,
            "transition_binding_blake3": self.transition_binding_blake3,
        }


@dataclass(frozen=True, slots=True)
class PremiumCandidateExecutionBinding:
    candidate_id: str
    source_candidate_id: str
    source_family: str
    domain: str
    stage: Stage
    split: str
    rubric_id: str
    rubric_blake3: str
    source_policy_path: Path
    source_policy_blake3: str
    runtime_policy_blake3: str
    construction_root: Path
    executable_binding_path: Path
    executable_binding_blake3: str
    legacy_runtime_source_blake3: str
    resolver_catalog_blake3: str
    tool_catalog: tuple[CandidateToolCatalogEntry, ...]
    tool_catalog_blake3: str
    public_runtime_context: Mapping[str, JsonValue]
    initial_workspace_files: Mapping[str, bytes]
    tool_registry: ToolRegistry
    binding_blake3: str

    def core_document(self) -> dict[str, Any]:
        return {
            "schema": BINDING_SCHEMA,
            "candidate_id": self.candidate_id,
            "source_candidate_id": self.source_candidate_id,
            "source_family": self.source_family,
            "domain": self.domain,
            "stage": self.stage.value,
            "split": self.split,
            "rubric_id": self.rubric_id,
            "rubric_blake3": self.rubric_blake3,
            "source_policy_path": str(self.source_policy_path),
            "source_policy_blake3": self.source_policy_blake3,
            "runtime_policy_blake3": self.runtime_policy_blake3,
            "construction_root": str(self.construction_root),
            "executable_binding_path": str(self.executable_binding_path),
            "executable_binding_blake3": self.executable_binding_blake3,
            "legacy_runtime_source_blake3": self.legacy_runtime_source_blake3,
            "resolver_catalog_blake3": self.resolver_catalog_blake3,
            "tool_catalog": [row.to_document() for row in self.tool_catalog],
            "tool_catalog_blake3": self.tool_catalog_blake3,
            "public_runtime_context": _plain(self.public_runtime_context),
            "initial_workspace_files": {
                path: blake3_bytes(payload)
                for path, payload in sorted(self.initial_workspace_files.items())
            },
        }


class PremiumConstructionExecutionBindingResolver:
    """Resolve only candidates authorized by one signed premium transition."""

    def __init__(
        self,
        *,
        transition_root: str | Path,
        publication_output_root: str | Path,
        rubrics: CompiledRubricRegistry,
        legacy_python_root: str | Path,
        runtime_state_root: str | Path,
        trust_store_path: str | Path,
        image_refs_path: str | Path,
        host_private_key_path: str | Path,
        host_key_id: str,
        docker_binary: str = "docker",
    ) -> None:
        self.publication_output_root = _existing_directory(
            publication_output_root, label="premium publication output root"
        )
        self.runtime_state_root = _real_directory(
            runtime_state_root, label="premium runtime state root", create=True
        )
        self.rubrics = rubrics
        self.authority = _runtime_authority(
            legacy_python_root=legacy_python_root,
            trust_store_path=trust_store_path,
            image_refs_path=image_refs_path,
            host_private_key_path=host_private_key_path,
            host_key_id=host_key_id,
            docker_binary=docker_binary,
        )
        transition_path = Path(transition_root)
        verifier = (
            verify_premium_construction_supervisor_transition_v2
            if os.path.lexists(transition_path / "signed-transition.v2.json")
            else verify_premium_construction_supervisor_transition
        )
        self.transition = verifier(
            transition_path,
            publication_output_root=self.publication_output_root,
            rubrics=rubrics,
            legacy_python_root=legacy_python_root,
            trust_store_path=trust_store_path,
            image_refs_path=image_refs_path,
            host_private_key_path=host_private_key_path,
            host_key_id=host_key_id,
            docker_binary=docker_binary,
        )
        selection_document, _ = _read_document(
            self.transition.root, _SOURCE_FILES["selection"], label="resolver selection"
        )
        queue_document, _ = _read_document(
            self.transition.root, _SOURCE_FILES["queue"], label="resolver queue"
        )
        catalog_document, _ = _read_document(
            self.transition.root, _SOURCE_FILES["proof_catalog"], label="resolver proof catalog"
        )
        supervisor_document, _ = _read_document(
            self.transition.root, _SOURCE_FILES["supervisor_input"], label="resolver supervisor input"
        )
        self.selection = CampaignSelectionV2.from_document(selection_document)
        self.queue = PremiumConstructionQueue.from_document(queue_document)
        self.proof_catalog = PremiumConstructionProofCatalog.from_document(catalog_document)
        self.supervisor_input = freeze_json(supervisor_document)
        selected_by_id, queued_by_id = _entry_binding(
            selection=self.selection, queue=self.queue, rubrics=rubrics
        )
        self._selected_by_id = selected_by_id
        self._queued_by_id = queued_by_id
        self._records = MappingProxyType(
            {
                record.candidate_id: record
                for record in self.proof_catalog.records
                if record.status == "succeeded"
            }
        )
        self._catalog_bindings = MappingProxyType(
            {
                row["candidate_id"]: row
                for row in self.supervisor_input["successful_bindings"]
            }
        )
        signed_rows = self.transition.envelope.payload["bindings"]
        references = tuple(
            PremiumCandidateBindingReference(
                candidate_id=row["candidate_id"],
                source_candidate_id=row["source_candidate_id"],
                source_family=row["source_family"],
                domain=row["domain"],
                stage=Stage(row["stage"]),
                split=row["split"],
                rubric_id=row["rubric_id"],
                rubric_blake3=row["rubric_blake3"],
                publication_id=row["catalog_binding"]["publication_id"],
                binding_blake3=row["catalog_binding"]["binding_blake3"],
                transition_binding_blake3=blake3_hex(row),
            )
            for row in signed_rows
        )
        self._references = MappingProxyType(
            {reference.source_candidate_id: reference for reference in references}
        )
        if len(self._references) != len(references):
            raise PremiumConstructionSupervisorError("signed binding source inventory is duplicated")
        catalog_core = {
            "schema": CATALOG_SCHEMA,
            "signed_transition_blake3": self.transition.transition_blake3,
            "selection_blake3": self.selection.selection_blake3,
            "coverage": "signed_premium_executable_ready_only",
            "candidate_count": len(references),
            "rows": [reference.to_document() for reference in references],
        }
        self.catalog_blake3 = blake3_hex(catalog_core)
        self._cache: dict[tuple[str, str], PremiumCandidateExecutionBinding] = {}
        self._resolve_locks: dict[tuple[str, str], RLock] = {}
        self._lock = RLock()

    @property
    def catalog_inventory_blake3(self) -> str:
        return self.catalog_blake3

    @property
    def executable_candidate_count(self) -> int:
        return len(self._references)

    def inventory(self) -> tuple[PremiumCandidateBindingReference, ...]:
        return tuple(self._references[key] for key in sorted(self._references))

    def resolve(
        self,
        eva_candidate_id: str,
        *,
        source_candidate_id: str,
    ) -> PremiumCandidateExecutionBinding:
        candidate_id = uuid_text(eva_candidate_id, label="premium EVA candidate_id")
        reference = self._references.get(source_candidate_id)
        if reference is None or reference.candidate_id != candidate_id:
            raise PremiumConstructionSupervisorError(
                "candidate is not authorized by the signed premium transition"
            )
        cache_key = (candidate_id, source_candidate_id)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
            resolve_lock = self._resolve_locks.setdefault(cache_key, RLock())
        with resolve_lock:
            with self._lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    return cached
            return self._resolve_uncached(
                candidate_id=candidate_id,
                source_candidate_id=source_candidate_id,
                reference=reference,
                cache_key=cache_key,
            )

    def _resolve_uncached(
        self,
        *,
        candidate_id: str,
        source_candidate_id: str,
        reference: PremiumCandidateBindingReference,
        cache_key: tuple[str, str],
    ) -> PremiumCandidateExecutionBinding:
        material = _open_executable_material(
            selected=self._selected_by_id[candidate_id],
            queued=self._queued_by_id[candidate_id],
            record=self._records[candidate_id],
            catalog_binding=self._catalog_bindings[candidate_id],
            selection_id=self.selection.selection_id,
            publication_output_root=self.publication_output_root,
            authority=self.authority,
        )
        signed_row = next(
            row
            for row in self.transition.envelope.payload["bindings"]
            if row["candidate_id"] == candidate_id
        )
        if _plain(_transition_binding_row(material)) != _plain(signed_row):
            raise PremiumConstructionSupervisorError("executable material changed after signing")
        public_context = freeze_json(
            {
                "schema": "eva.premium-candidate-runtime-context.v1",
                "candidate_id": candidate_id,
                "source_candidate_id": source_candidate_id,
                "source_family": reference.source_family,
                "domain": reference.domain,
                "focus": reference.stage.value,
                "target_split": reference.split,
                "rubric_id": reference.rubric_id,
                "rubric_blake3": reference.rubric_blake3,
                "signed_transition_blake3": self.transition.transition_blake3,
                "source_policy_blake3": material.source_policy_blake3,
                "runtime_policy_blake3": material.runtime_policy_blake3,
                "tool_catalog_blake3": material.tool_catalog_blake3,
                "s1_plan_contract": material.templates["s1"],
                "s2_evidence_contract": material.templates["s2"],
                "execution_stages": {
                    stage: {
                        "stage": template["stage"],
                        "artifact_relative_path": template["artifact"]["relative_path"],
                        "artifact_json_schema": template["artifact"]["json_schema"],
                        "limits": template["limits"],
                    }
                    for stage, template in (
                        ("S3", material.templates["s3"]),
                        ("S4", material.templates["s4"]),
                    )
                },
                "s5_terminal_json_schema": material.templates["s5"]["terminal"]["json_schema"],
            }
        )
        assert isinstance(public_context, Mapping)
        runtime_policy_bytes = canonical_json_bytes(_plain(material.runtime_policy))
        initial_files = MappingProxyType(
            {
                ".eva/runtime-context.json": canonical_json_bytes(_plain(public_context)),
                ".eva/source-policy.json": runtime_policy_bytes,
            }
        )
        evidence_root = _materialize_runtime_inputs(
            material,
            runtime_state_root=self.runtime_state_root,
            authority=self.authority,
        )
        backend = _LegacyCandidateToolBackend(
            modules=self.authority.modules,
            binding_identity={
                "candidate_id": candidate_id,
                "source_candidate_id": source_candidate_id,
                "signed_transition_blake3": self.transition.transition_blake3,
                "runtime_policy_blake3": material.runtime_policy_blake3,
                "tool_catalog_blake3": material.tool_catalog_blake3,
            },
            runtime_state_root=self.runtime_state_root,
            contracts=_plain(material.templates),
            evidence_root=evidence_root,
            image_refs=self.authority.image_refs,
            trust_store=self.authority.trust_store,
            key_id=self.authority.key_id,
            private_key=self.authority.private_key,
            docker_binary=self.authority.docker_binary,
        )
        registry = ToolRegistry(
            [
                ToolDefinition(
                    name=row.name,
                    description=row.description,
                    input_schema=row.input_schema,
                    handler=backend.handler(row.name),
                    parallel_safe=False,
                    read_only=False,
                )
                for row in material.tool_catalog
            ]
        )
        observed = {
            row["function"]["name"]: row["function"]
            for row in registry.public_schemas()
        }
        for row in material.tool_catalog:
            schema = observed.get(row.name)
            if not isinstance(schema, Mapping) or not (
                schema.get("description") == row.description
                and _plain(schema.get("parameters")) == _plain(row.input_schema)
                and schema.get("x-eva-kind") == "tool"
                and schema.get("x-eva-parallel-safe") is False
            ):
                raise PremiumConstructionSupervisorError(
                    f"candidate registry schema/visibility differs for {row.name}"
                )
        provisional = PremiumCandidateExecutionBinding(
            candidate_id=candidate_id,
            source_candidate_id=source_candidate_id,
            source_family=reference.source_family,
            domain=reference.domain,
            stage=reference.stage,
            split=reference.split,
            rubric_id=reference.rubric_id,
            rubric_blake3=reference.rubric_blake3,
            source_policy_path=material.source_policy_path,
            source_policy_blake3=material.source_policy_blake3,
            runtime_policy_blake3=material.runtime_policy_blake3,
            construction_root=material.publication_root,
            executable_binding_path=material.executable_binding_path,
            executable_binding_blake3=material.executable_binding_blake3,
            legacy_runtime_source_blake3=self.authority.modules.source_blake3,
            resolver_catalog_blake3=self.catalog_blake3,
            tool_catalog=material.tool_catalog,
            tool_catalog_blake3=material.tool_catalog_blake3,
            public_runtime_context=public_context,
            initial_workspace_files=initial_files,
            tool_registry=registry,
            binding_blake3="",
        )
        fields = {
            name: getattr(provisional, name)
            for name in provisional.__dataclass_fields__
            if name != "binding_blake3"
        }
        binding = PremiumCandidateExecutionBinding(
            **fields,
            binding_blake3=blake3_hex(provisional.core_document()),
        )
        with self._lock:
            return self._cache.setdefault(cache_key, binding)


__all__ = [
    "PremiumCandidateBindingReference",
    "PremiumCandidateExecutionBinding",
    "PremiumConstructionExecutionBindingResolver",
    "PremiumConstructionSupervisorError",
    "PremiumConstructionSupervisorTransition",
    "issue_premium_construction_supervisor_transition",
    "issue_premium_construction_supervisor_transition_v2",
    "verify_premium_construction_supervisor_transition",
    "verify_premium_construction_supervisor_transition_v2",
]
