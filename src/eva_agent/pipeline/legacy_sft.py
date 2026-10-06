"""Fast, strict import of signed legacy teacher rollouts into SFT shards.

Legacy rlevo rollouts use SHA-256 references.  They are verified here only as
upstream provenance; every new EVA identity and content commitment is UUID +
BLAKE3.  No new SHA-based identity, reward, receipt, or transition is created.
"""

from __future__ import annotations

import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .contracts import RuntimeIdFactory, freeze_json, uuid_text
from .digests import blake3_bytes, blake3_hex, canonical_json_bytes


LEGACY_SFT_DATASET_SCHEMA = "eva.legacy-teacher-sft-dataset.v1"
LEGACY_SFT_SLICE_SCHEMA = "eva.legacy-teacher-sft-slice.v1"
LEGACY_SFT_VERIFY_SCHEMA = "eva.legacy-teacher-sft-verification.v1"
LEGACY_SFT_INVENTORY_SCHEMA = "eva.legacy-teacher-sft-inventory.v1"
LEGACY_ROLLOUT_SCHEMAS = {
    "rlevo.med-research-agent-rollout-receipt.v1",
    "rlevo.med-research-agent-rollout-receipt.v2",
    "rlevo.med-research-agent-rollout-receipt.v3",
    "rlevo.med-research-agent-rollout-receipt.v4",
}
MAXIMUM_SOURCE_BYTES = 64 * 1024 * 1024
MAXIMUM_SHARD_BYTES = 512 * 1024 * 1024


class LegacySFTError(ValueError):
    """A legacy teacher source or new SFT shard failed closed."""


@dataclass(frozen=True, slots=True)
class LegacyTeacherSource:
    relative_receipt_path: str
    receipt: Mapping[str, Any]
    transcript: Mapping[str, Any]
    grade: Mapping[str, Any]
    final_host_receipt: Mapping[str, Any]
    rollout_attestation: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LegacySFTBuild:
    dataset_id: str
    dataset_root: Path
    source_count: int
    slice_count: int
    shard_count: int
    manifest_blake3: str


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LegacySFTError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, *, maximum_bytes: int = MAXIMUM_SOURCE_BYTES) -> tuple[Any, bytes]:
    try:
        before = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > maximum_bytes
        ):
            raise LegacySFTError("source topology or size differs")
        payload = path.read_bytes()
        after = path.lstat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise LegacySFTError("source changed while reopening")
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                LegacySFTError(f"non-finite JSON constant: {value}")
            ),
        )
    except LegacySFTError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise LegacySFTError("source JSON cannot be reopened") from exc
    return value, payload


def _legacy_canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    """Verify an upstream legacy reference; never use as a new EVA identity."""

    return hashlib.sha256(value).hexdigest()


def _sha256_value(value: Any) -> str:
    return _sha256_bytes(_legacy_canonical_bytes(value))


def _safe_child(root: Path, reference: Any, *, label: str) -> tuple[Path, str]:
    if (
        not isinstance(reference, Mapping)
        or not {"path", "sha256"}.issubset(reference)
        or not set(reference).issubset(
            {"path", "sha256", "grade_sha256", "normalized_score"}
        )
    ):
        raise LegacySFTError(f"{label} reference differs")
    raw = reference["path"]
    digest = reference["sha256"]
    pure = PurePosixPath(raw) if isinstance(raw, str) else PurePosixPath("/")
    if (
        pure.is_absolute()
        or not pure.parts
        or pure.as_posix() != raw
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise LegacySFTError(f"{label} path differs")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise LegacySFTError(f"{label} upstream digest differs")
    path = root.joinpath(*pure.parts)
    return path, digest


def _read_sha_reference(root: Path, reference: Any, *, label: str) -> tuple[Any, bytes]:
    path, expected = _safe_child(root, reference, label=label)
    value, payload = _read_json(path)
    if _sha256_bytes(payload) != expected:
        raise LegacySFTError(f"{label} upstream file digest differs")
    return value, payload


def _trust_keys(path: Path) -> Mapping[str, str]:
    value, _ = _read_json(path, maximum_bytes=1024 * 1024)
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "rlevo.med-research-host-trust-store.v1"
        or value.get("status") != "active"
        or value.get("algorithm") != "Ed25519"
        or not isinstance(value.get("keys"), Mapping)
        or not value["keys"]
    ):
        raise LegacySFTError("legacy trust store differs")
    return value["keys"]


def _verify_host_receipt(receipt: Any, keys: Mapping[str, str]) -> Mapping[str, Any]:
    expected = {"schema", "key_id", "payload", "payload_sha256", "signature_base64"}
    if not isinstance(receipt, Mapping) or set(receipt) != expected:
        raise LegacySFTError("legacy host receipt fields differ")
    if receipt["schema"] != "rlevo.med-research-signed-host-receipt.v1":
        raise LegacySFTError("legacy host receipt schema differs")
    encoded_key = keys.get(receipt["key_id"])
    try:
        public = base64.b64decode(encoded_key, validate=True)
        signature = base64.b64decode(receipt["signature_base64"], validate=True)
    except (binascii.Error, TypeError, ValueError):
        raise LegacySFTError("legacy host receipt key or signature differs") from None
    if len(public) != 32 or len(signature) != 64:
        raise LegacySFTError("legacy host receipt key or signature length differs")
    payload = receipt["payload"]
    payload_bytes = _legacy_canonical_bytes(payload)
    if _sha256_bytes(payload_bytes) != receipt["payload_sha256"]:
        raise LegacySFTError("legacy host receipt payload digest differs")
    try:
        Ed25519PublicKey.from_public_bytes(public).verify(signature, payload_bytes)
    except (InvalidSignature, ValueError):
        raise LegacySFTError("legacy host receipt signature differs") from None
    if not isinstance(payload, Mapping):
        raise LegacySFTError("legacy host receipt payload differs")
    events = payload.get("events")
    if not isinstance(events, list) or [row.get("ordinal") for row in events] != list(
        range(len(events))
    ):
        raise LegacySFTError("legacy host event ordinals differ")
    return payload


def _walk_sha_refs(value: Any) -> tuple[Mapping[str, str], ...]:
    rows: list[Mapping[str, str]] = []
    if isinstance(value, Mapping):
        if set(value) == {"path", "sha256"}:
            rows.append(value)
        else:
            for child in value.values():
                rows.extend(_walk_sha_refs(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_walk_sha_refs(child))
    return tuple(rows)


def _verify_stage_references(root: Path, receipt: Mapping[str, Any]) -> None:
    seen: set[str] = set()
    for reference in _walk_sha_refs(receipt.get("stage_evidence", {})):
        path, expected = _safe_child(root, reference, label="stage evidence")
        if path.as_posix() in seen:
            continue
        seen.add(path.as_posix())
        _, payload = _read_json(path)
        if _sha256_bytes(payload) != expected:
            raise LegacySFTError("stage evidence upstream file digest differs")


def verify_legacy_teacher_source(
    receipt_path: Path,
    *,
    source_root: Path,
    trust_store_path: Path,
    eligible_model_ids: Sequence[str],
    minimum_score: float = 0.8,
) -> LegacyTeacherSource:
    """Reopen one signed completed teacher rollout and its visible transcript."""

    source_root = Path(source_root).resolve()
    receipt_path = Path(receipt_path).resolve()
    try:
        relative_receipt = receipt_path.relative_to(source_root).as_posix()
    except ValueError:
        raise LegacySFTError("rollout receipt is outside source root") from None
    receipt, _receipt_bytes = _read_json(receipt_path)
    if not isinstance(receipt, Mapping) or receipt.get("schema") not in LEGACY_ROLLOUT_SCHEMAS:
        raise LegacySFTError("legacy rollout receipt schema differs")
    if (
        receipt.get("status") != "completed"
        or receipt.get("termination") not in {"submitted", "target_completed"}
        or receipt.get("exact_identity_all_turns") is not True
        or receipt.get("hidden_reasoning_recorded") is not False
        or receipt.get("raw_provider_response_recorded") is not False
    ):
        raise LegacySFTError("legacy rollout is not a completed privacy-safe teacher")
    if "infrastructure_healthy" in receipt and receipt["infrastructure_healthy"] is not True:
        raise LegacySFTError("legacy rollout infrastructure was not healthy")
    model_id = receipt.get("model_id")
    if model_id not in set(eligible_model_ids):
        raise LegacySFTError("legacy rollout model is not SFT eligible")
    turns = receipt.get("turns")
    if (
        not isinstance(turns, list)
        or len(turns) != receipt.get("turn_count")
        or any(row.get("identity_match") is not True for row in turns)
    ):
        raise LegacySFTError("legacy rollout turn identity differs")
    root = receipt_path.parent
    transcript, _ = _read_sha_reference(root, receipt.get("transcript"), label="transcript")
    grade, _ = _read_sha_reference(root, receipt.get("final_grade"), label="final grade")
    final_receipt, _ = _read_sha_reference(
        root, receipt.get("final_host_receipt"), label="final host receipt"
    )
    attestation, _ = _read_sha_reference(
        root, receipt.get("host_attestation"), label="rollout attestation"
    )
    keys = _trust_keys(trust_store_path)
    final_payload = _verify_host_receipt(final_receipt, keys)
    attestation_payload = _verify_host_receipt(attestation, keys)
    identity = (receipt.get("sandbox_id"), receipt.get("episode_id"))
    if (
        (final_payload.get("sandbox_id"), final_payload.get("episode_id")) != identity
        or (attestation_payload.get("sandbox_id"), attestation_payload.get("episode_id"))
        != identity
    ):
        raise LegacySFTError("legacy signed receipt identity differs")
    if not isinstance(grade, Mapping):
        raise LegacySFTError("legacy final grade differs")
    grade_core = {key: value for key, value in grade.items() if key != "grade_sha256"}
    if (
        grade.get("grade_sha256") != _sha256_value(grade_core)
        or receipt["final_grade"].get("grade_sha256") != grade.get("grade_sha256")
        or float(receipt["final_grade"].get("normalized_score", -1))
        != float(grade.get("normalized_score", -2))
        or float(grade.get("normalized_score", -1)) < float(minimum_score)
        or grade.get("execution_gate_passed") is not True
        or grade.get("critical_items_passed") is not True
        or grade.get("score_eligible") is not True
    ):
        raise LegacySFTError("legacy final grade is not high-performance eligible")
    if (
        grade.get("host_receipt_payload_sha256") != final_receipt.get("payload_sha256")
        or grade.get("sandbox_id") != receipt.get("sandbox_id")
        or grade.get("episode_id") != receipt.get("episode_id")
    ):
        raise LegacySFTError("legacy final grade lineage differs")
    if not isinstance(transcript, Mapping) or (
        transcript.get("schema") != "rlevo.med-research-agent-transcript.v1"
        or transcript.get("hidden_reasoning_recorded") is not False
        or transcript.get("raw_provider_response_recorded") is not False
        or transcript.get("rollout_id") != receipt.get("rollout_id")
        or transcript.get("sandbox_id") != receipt.get("sandbox_id")
        or transcript.get("episode_id") != receipt.get("episode_id")
        or transcript.get("model_id") != model_id
    ):
        raise LegacySFTError("legacy visible transcript lineage differs")
    messages = transcript.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) < 3
        or messages[0].get("role") != "system"
        or messages[1].get("role") != "user"
    ):
        raise LegacySFTError("legacy visible transcript framing differs")
    rollout_claim = dict(receipt)
    rollout_claim.pop("host_attestation", None)
    observations = attestation_payload.get("observations")
    if (
        not isinstance(observations, Mapping)
        or observations.get("rollout_claim_sha256") != _sha256_value(rollout_claim)
        or observations.get("transcript_sha256") != receipt["transcript"]["sha256"]
        or observations.get("final_host_receipt_sha256")
        != receipt["final_host_receipt"]["sha256"]
        or observations.get("final_grade_sha256") != receipt["final_grade"]["sha256"]
    ):
        raise LegacySFTError("legacy rollout attestation lineage differs")
    _verify_stage_references(root, receipt)
    return LegacyTeacherSource(
        relative_receipt_path=relative_receipt,
        receipt=freeze_json(receipt),
        transcript=freeze_json(transcript),
        grade=freeze_json(grade),
        final_host_receipt=freeze_json(final_receipt),
        rollout_attestation=freeze_json(attestation),
    )


def _clean_message(message: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"role", "content", "tool_calls", "tool_call_id", "source"}
    result = {key: message[key] for key in allowed if key in message}
    if result.get("source") == "harness":
        raise LegacySFTError("harness-generated assistant output is not an SFT target")
    if result.get("role") not in {"system", "user", "assistant", "tool"}:
        raise LegacySFTError("legacy transcript message role differs")
    if "content" not in result:
        result["content"] = ""
    return result


def _tool_ids(message: Mapping[str, Any]) -> tuple[str, ...]:
    calls = message.get("tool_calls", ())
    if not isinstance(calls, (list, tuple)):
        raise LegacySFTError("legacy assistant tool-call group differs")
    ids = tuple(row.get("id") for row in calls if isinstance(row, Mapping))
    if len(ids) != len(calls) or any(not isinstance(value, str) or not value for value in ids):
        raise LegacySFTError("legacy assistant tool-call identity differs")
    if len(set(ids)) != len(ids):
        raise LegacySFTError("legacy assistant tool-call identity is duplicated")
    return ids


def _target_stage(message: Mapping[str, Any], previous: str) -> str:
    calls = message.get("tool_calls")
    if not calls:
        return previous
    function = calls[0].get("function") if isinstance(calls[0], Mapping) else None
    name = function.get("name") if isinstance(function, Mapping) else None
    if name == "materialize_plan":
        return "S1"
    if name in {"retrieve_frozen_evidence", "materialize_evidence_selection"}:
        return "S2"
    if name == "submit_results":
        return "S5"
    if name == "execute_code":
        try:
            arguments = json.loads(function.get("arguments", ""))
        except (TypeError, ValueError):
            raise LegacySFTError("legacy execute_code arguments are not JSON") from None
        stage = arguments.get("stage") if isinstance(arguments, Mapping) else None
        if stage in {"S3", "S4"}:
            return stage
        return "S3" if previous in {"S1", "S2"} else "S4"
    return "E2E"


def _slice_source(source: LegacyTeacherSource, dataset_id: str, ids: RuntimeIdFactory) -> tuple[dict[str, Any], ...]:
    raw_messages = source.transcript["messages"]
    messages: list[dict[str, Any]] = []
    for raw in raw_messages:
        if not isinstance(raw, Mapping):
            raise LegacySFTError("legacy transcript message differs")
        if raw.get("role") == "assistant" and raw.get("source") == "harness":
            if not messages or messages[-1].get("role") != "assistant":
                raise LegacySFTError("legacy harness terminal result has no assistant call")
            call_ids = _tool_ids(messages[-1])
            if len(call_ids) != 1:
                raise LegacySFTError("legacy harness terminal result call identity differs")
            messages.append(
                {
                    "role": "tool",
                    "content": raw.get("content", ""),
                    "tool_call_id": call_ids[0],
                    "source": "harness",
                }
            )
            continue
        messages.append(_clean_message(raw))
    slices: list[dict[str, Any]] = []
    current_stage = "E2E"
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        current_stage = _target_stage(message, current_stage)
        call_ids = _tool_ids(message)
        observations: list[Mapping[str, Any]] = []
        if call_ids:
            by_id: dict[str, Mapping[str, Any]] = {}
            for later in messages[index + 1 :]:
                if later["role"] == "assistant":
                    break
                if later["role"] == "tool" and later.get("tool_call_id") in call_ids:
                    by_id[later["tool_call_id"]] = later
            if set(by_id) != set(call_ids):
                raise LegacySFTError("legacy SFT target omits a tool observation")
            observations = [by_id[call_id] for call_id in call_ids]
        example_id = uuid_text(ids.new("legacy-sft-slice"), label="legacy SFT example_id")
        core = {
            "schema": LEGACY_SFT_SLICE_SCHEMA,
            "dataset_id": dataset_id,
            "example_id": example_id,
            "sandbox_id": source.receipt["sandbox_id"],
            "rollout_id": source.receipt["rollout_id"],
            "model_id": source.receipt["model_id"],
            "stage": current_stage,
            "prefix": tuple(messages[:index]),
            "supervised_assistant_decision": message,
            "target_tool_observations": tuple(observations),
            "atomic_multi_tool_target": len(call_ids) > 1,
            "rubric_reward": {
                "table_id": source.grade["table_id"],
                "focus": source.grade["focus"],
                "normalized_score": source.grade["normalized_score"],
                "execution_gate_passed": source.grade["execution_gate_passed"],
                "critical_items_passed": source.grade["critical_items_passed"],
                "rubric_verdicts": source.grade["rubric_verdicts"],
                "upstream_grade_sha256": source.grade["grade_sha256"],
            },
            "provenance": {
                "source_receipt_path": source.relative_receipt_path,
                "transcript_ref": source.receipt["transcript"],
                "final_grade_ref": source.receipt["final_grade"],
                "final_host_receipt_ref": source.receipt["final_host_receipt"],
                "rollout_host_attestation_ref": source.receipt["host_attestation"],
                "stage_evidence_refs": source.receipt.get("stage_evidence", {}),
                "workspace_evidence": source.final_host_receipt["payload"].get(
                    "observations", {}
                ),
            },
            "hidden_reasoning_included": False,
            "private_reference_included": False,
        }
        slices.append({**core, "slice_blake3": blake3_hex(core)})
    if not slices:
        raise LegacySFTError("legacy teacher transcript has no assistant decision")
    return tuple(slices)


def _write_once(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short SFT artifact write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, mode)


def _source_row(source: LegacyTeacherSource) -> dict[str, Any]:
    return {
        "source_receipt_path": source.relative_receipt_path,
        "sandbox_id": source.receipt["sandbox_id"],
        "rollout_id": source.receipt["rollout_id"],
        "model_id": source.receipt["model_id"],
        "score": source.grade["normalized_score"],
        "focus": source.grade["focus"],
        "table_id": source.grade["table_id"],
        "transcript_ref": source.receipt["transcript"],
        "final_grade_ref": source.receipt["final_grade"],
        "final_host_receipt_ref": source.receipt["final_host_receipt"],
        "rollout_host_attestation_ref": source.receipt["host_attestation"],
    }


def build_legacy_teacher_sft_dataset(
    *,
    source_root: Path,
    output_root: Path,
    trust_store_path: Path,
    eligible_model_ids: Sequence[str],
    id_factory: RuntimeIdFactory,
    minimum_score: float = 0.8,
    shard_size: int = 1024,
    verification_workers: int = 64,
) -> LegacySFTBuild:
    """Verify all existing rollouts in parallel and stream eligible slices."""

    if not 0.0 <= float(minimum_score) <= 1.0:
        raise LegacySFTError("legacy SFT minimum score differs")
    if type(shard_size) is not int or not 1 <= shard_size <= 100_000:
        raise LegacySFTError("legacy SFT shard size differs")
    if type(verification_workers) is not int or not 1 <= verification_workers <= 256:
        raise LegacySFTError("legacy SFT verification width differs")
    models = tuple(sorted(set(eligible_model_ids)))
    if not models:
        raise LegacySFTError("legacy SFT eligible model set is empty")
    source_root = Path(source_root).resolve()
    if source_root.is_symlink() or not source_root.is_dir():
        raise LegacySFTError("legacy SFT source root topology differs")
    receipts = tuple(sorted(source_root.rglob("rollout-receipt.json")))
    if not receipts:
        raise LegacySFTError("legacy SFT source root has no rollout receipts")

    def verify(path: Path) -> tuple[Path, LegacyTeacherSource | None, str | None]:
        try:
            return (
                path,
                verify_legacy_teacher_source(
                    path,
                    source_root=source_root,
                    trust_store_path=trust_store_path,
                    eligible_model_ids=models,
                    minimum_score=minimum_score,
                ),
                None,
            )
        except Exception as exc:
            return path, None, f"{type(exc).__name__}:{exc}"

    with ThreadPoolExecutor(max_workers=min(verification_workers, len(receipts))) as pool:
        inspected = tuple(pool.map(verify, receipts))
    sources = tuple(row[1] for row in inspected if row[1] is not None)
    if not sources:
        raise LegacySFTError("legacy SFT inventory contains no eligible teacher")

    dataset_id = uuid_text(id_factory.new("legacy-sft-dataset"), label="legacy SFT dataset_id")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output_root.is_symlink() or not output_root.is_dir():
        raise LegacySFTError("legacy SFT output root topology differs")
    dataset_root = output_root / dataset_id
    dataset_root.mkdir(mode=0o700)
    shard_root = dataset_root / "shards"
    shard_root.mkdir(mode=0o700)

    source_rows: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    counts = {stage: 0 for stage in ("S1", "S2", "S3", "S4", "S5", "E2E")}
    pending: list[dict[str, Any]] = []
    slice_count = 0

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        ordinal = len(shard_rows)
        name = f"part-{ordinal:05d}.jsonl"
        payload = b"".join(canonical_json_bytes(row) for row in pending)
        _write_once(shard_root / name, payload)
        shard_rows.append(
            {
                "path": f"shards/{name}",
                "slice_count": len(pending),
                "first_example_id": pending[0]["example_id"],
                "last_example_id": pending[-1]["example_id"],
                "content_blake3": blake3_bytes(payload),
                "byte_count": len(payload),
            }
        )
        pending = []

    for source in sources:
        source_rows.append(_source_row(source))
        for row in _slice_source(source, dataset_id, id_factory):
            pending.append(row)
            counts[row["stage"]] += 1
            slice_count += 1
            if len(pending) >= shard_size:
                flush()
    flush()
    rejected = tuple(
        {
            "source_receipt_path": path.relative_to(source_root).as_posix(),
            "sft_eligible": False,
            "rl_start_candidate": True,
            "reason": error,
        }
        for path, source, error in inspected
        if source is None
    )
    manifest_core = {
        "schema": LEGACY_SFT_DATASET_SCHEMA,
        "dataset_id": dataset_id,
        "source_root_label": source_root.name,
        "selection": {
            "eligible_model_ids": models,
            "minimum_score": minimum_score,
            "require_completed_or_target_completed": True,
            "require_exact_model_identity_all_turns": True,
            "require_execution_and_critical_gates": True,
            "require_signed_host_evidence": True,
            "hidden_reasoning_allowed": False,
        },
        "inventory": {
            "scanned_rollouts": len(receipts),
            "eligible_sources": len(sources),
            "rl_start_candidates": len(rejected),
            "provider_calls": 0,
        },
        "source_count": len(source_rows),
        "slice_count": slice_count,
        "shard_count": len(shard_rows),
        "counts_by_stage": counts,
        "sources": tuple(source_rows),
        "shards": tuple(shard_rows),
        "rl_start_candidates": rejected,
        "private_reference_included": False,
        "hidden_reasoning_included": False,
    }
    manifest = {**manifest_core, "manifest_blake3": blake3_hex(manifest_core)}
    _write_once(dataset_root / "manifest.json", canonical_json_bytes(manifest))
    for directory in (shard_root, dataset_root):
        os.chmod(directory, 0o555)
    return LegacySFTBuild(
        dataset_id=dataset_id,
        dataset_root=dataset_root,
        source_count=len(sources),
        slice_count=slice_count,
        shard_count=len(shard_rows),
        manifest_blake3=manifest["manifest_blake3"],
    )


def verify_legacy_sft_dataset(root: Path, dataset_id: str) -> Mapping[str, Any]:
    """Reopen all JSONL shards and recompute the compact BLAKE3 chain."""

    checks: list[str] = []
    slice_count = 0
    try:
        uuid_text(dataset_id, label="legacy SFT dataset_id")
        dataset_root = Path(root) / dataset_id
        if dataset_root.is_symlink() or not dataset_root.is_dir():
            raise LegacySFTError("legacy SFT dataset root topology differs")
        manifest, manifest_bytes = _read_json(dataset_root / "manifest.json")
        if canonical_json_bytes(manifest) != manifest_bytes:
            raise LegacySFTError("legacy SFT manifest is not canonical JSON")
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("schema") != LEGACY_SFT_DATASET_SCHEMA
            or manifest.get("dataset_id") != dataset_id
        ):
            raise LegacySFTError("legacy SFT manifest identity differs")
        manifest_core = {key: value for key, value in manifest.items() if key != "manifest_blake3"}
        if manifest.get("manifest_blake3") != blake3_hex(manifest_core):
            raise LegacySFTError("legacy SFT manifest commitment differs")
        expected_files = {"manifest.json"}
        seen_ids: set[str] = set()
        counts = {stage: 0 for stage in ("S1", "S2", "S3", "S4", "S5", "E2E")}
        for shard in manifest.get("shards", ()):
            path = shard.get("path") if isinstance(shard, Mapping) else None
            pure = PurePosixPath(path) if isinstance(path, str) else PurePosixPath("/")
            if pure.is_absolute() or len(pure.parts) != 2 or pure.parts[0] != "shards":
                raise LegacySFTError("legacy SFT shard path differs")
            expected_files.add(path)
            shard_path = dataset_root.joinpath(*pure.parts)
            info = shard_path.lstat()
            if shard_path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise LegacySFTError("legacy SFT shard topology differs")
            payload = shard_path.read_bytes()
            if (
                len(payload) != shard.get("byte_count")
                or len(payload) > MAXIMUM_SHARD_BYTES
                or blake3_bytes(payload) != shard.get("content_blake3")
            ):
                raise LegacySFTError("legacy SFT shard commitment differs")
            rows = []
            for line in payload.splitlines():
                row = json.loads(line, object_pairs_hook=_strict_object)
                if not isinstance(row, Mapping) or row.get("schema") != LEGACY_SFT_SLICE_SCHEMA:
                    raise LegacySFTError("legacy SFT slice schema differs")
                if row.get("dataset_id") != dataset_id:
                    raise LegacySFTError("legacy SFT slice dataset binding differs")
                example_id = uuid_text(row.get("example_id"), label="legacy SFT example_id")
                if example_id in seen_ids:
                    raise LegacySFTError("legacy SFT example identity is duplicated")
                seen_ids.add(example_id)
                core = {key: value for key, value in row.items() if key != "slice_blake3"}
                if row.get("slice_blake3") != blake3_hex(core):
                    raise LegacySFTError("legacy SFT slice commitment differs")
                if row.get("hidden_reasoning_included") is not False or row.get("private_reference_included") is not False:
                    raise LegacySFTError("legacy SFT privacy marker differs")
                target = row.get("supervised_assistant_decision")
                if not isinstance(target, Mapping) or target.get("role") != "assistant" or target.get("source") == "harness":
                    raise LegacySFTError("legacy SFT supervised target differs")
                call_ids = _tool_ids(target)
                observations = row.get("target_tool_observations")
                if not isinstance(observations, list) or tuple(
                    item.get("tool_call_id") for item in observations
                ) != call_ids:
                    raise LegacySFTError("legacy SFT tool-result join differs")
                stage = row.get("stage")
                if stage not in counts:
                    raise LegacySFTError("legacy SFT stage differs")
                counts[stage] += 1
                rows.append(row)
            if (
                len(rows) != shard.get("slice_count")
                or rows[0]["example_id"] != shard.get("first_example_id")
                or rows[-1]["example_id"] != shard.get("last_example_id")
            ):
                raise LegacySFTError("legacy SFT shard index differs")
            slice_count += len(rows)
        actual_files = {
            path.relative_to(dataset_root).as_posix()
            for path in dataset_root.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise LegacySFTError("legacy SFT file inventory differs")
        if (
            slice_count != manifest.get("slice_count")
            or len(manifest.get("shards", ())) != manifest.get("shard_count")
            or counts != manifest.get("counts_by_stage")
        ):
            raise LegacySFTError("legacy SFT aggregate counts differ")
        checks.extend(
            (
                "manifest_commitment",
                "exact_shard_inventory",
                "all_slice_commitments",
                "tool_result_joins",
                "stage_counts",
                "privacy_markers",
            )
        )
        core = {
            "schema": LEGACY_SFT_VERIFY_SCHEMA,
            "valid": True,
            "dataset_id": dataset_id,
            "source_count": manifest["source_count"],
            "slice_count": slice_count,
            "shard_count": manifest["shard_count"],
            "manifest_blake3": manifest["manifest_blake3"],
            "checks": tuple(checks),
            "errors": (),
        }
    except Exception as exc:
        core = {
            "schema": LEGACY_SFT_VERIFY_SCHEMA,
            "valid": False,
            "dataset_id": dataset_id,
            "source_count": 0,
            "slice_count": slice_count,
            "shard_count": 0,
            "manifest_blake3": None,
            "checks": tuple(checks),
            "errors": (f"{type(exc).__name__}:{exc}",),
        }
    return freeze_json({**core, "report_blake3": blake3_hex(core)})


__all__ = [
    "LegacySFTBuild",
    "LegacySFTError",
    "LegacyTeacherSource",
    "build_legacy_teacher_sft_dataset",
    "verify_legacy_sft_dataset",
    "verify_legacy_teacher_source",
]
