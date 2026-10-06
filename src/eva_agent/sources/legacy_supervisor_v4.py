"""Read-only bridge from the frozen EvaMed supervisor-v4 source registry.

The old campaign used SHA-256 commitments.  This adapter verifies those
commitments *only* as upstream provenance, then creates UUID runtime identities
and BLAKE3 import receipts in the EVA-Agent integrity domain.  Candidate
artifacts and their evidence are opened lazily, after registry-only selection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from blake3 import blake3

from eva_agent.pipeline.contracts import BenchmarkEpisode, BenchmarkSource, Stage, freeze_json
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex


REGISTRY_FILENAME = "candidate-registry.v4.json"
REGISTRY_SCHEMA = "rlevo.med-research-campaign-supervisor-registry.v1"
CAMPAIGN_ID = "evamed-6000-campaign-supervisor-v1"
ARTIFACT_SCHEMA = "rlevo.med-research-candidate-construction-input.v3"
STAGES = frozenset(stage.value for stage in Stage)
FAMILIES = frozenset(
    {"agentclinic", "automedbench", "healthbench-professional", "medxpertqa"}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_PRIVATE_TOP_LEVEL_KEYS = frozenset(
    {
        "answer_key",
        "expected_answer",
        "gold",
        "judge_only",
        "judge_only_reference",
        "private",
        "private_reference",
        "reference",
        "reference_answer",
    }
)
_FORBIDDEN_POLICY_KEY_PARTS = (
    "answer_key",
    "chain_of_thought",
    "gold",
    "hidden_reasoning",
    "judge_only",
    "oracle",
    "private",
    "reference_answer",
)


class LegacyImportError(ValueError):
    """The legacy source failed a read-only integrity or projection gate."""


class RubricAvailability(str, Enum):
    READY = "ready"
    PENDING = "pending"


@dataclass(frozen=True, slots=True)
class LegacyCandidateRef:
    """Registry metadata only; constructing this does not open the payload."""

    candidate_id: str
    family: str
    stage: Stage
    priority: int
    source_shard: str
    artifact_relative_path: str
    upstream_artifact_sha256: str
    frozen_assertions: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class LegacyEvidenceReceipt:
    evidence_id: str
    relative_path: str
    byte_count: int
    upstream_sha256: str
    content_blake3: str

    def to_document(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "relative_path": self.relative_path,
            "byte_count": self.byte_count,
            "upstream_sha256": self.upstream_sha256,
            "content_blake3": self.content_blake3,
        }


@dataclass(frozen=True, slots=True)
class LegacyImportManifest:
    import_id: str
    episode_id: str
    candidate_id: str
    campaign_id: str
    registry_revision: int
    registry_file_blake3: str
    upstream_registry_sha256: str
    artifact_relative_path: str
    artifact_byte_count: int
    upstream_artifact_sha256: str
    artifact_blake3: str
    source_family: str
    source_track: str
    stage: Stage
    rubric_domain: str
    rubric_availability: RubricAvailability
    rubric_resolution_reason: str
    evidence_receipts: tuple[LegacyEvidenceReceipt, ...]
    policy_projection_blake3: str
    judge_projection_blake3: str
    import_blake3: str

    def core_document(self) -> dict[str, Any]:
        return {
            "schema": "eva.legacy-supervisor-v4-import.v1",
            "import_id": self.import_id,
            "episode_id": self.episode_id,
            "candidate_id": self.candidate_id,
            "campaign_id": self.campaign_id,
            "registry_revision": self.registry_revision,
            "registry_file_blake3": self.registry_file_blake3,
            # These names explicitly keep the old digest domain upstream-only.
            "upstream_registry_sha256": self.upstream_registry_sha256,
            "artifact_relative_path": self.artifact_relative_path,
            "artifact_byte_count": self.artifact_byte_count,
            "upstream_artifact_sha256": self.upstream_artifact_sha256,
            "artifact_blake3": self.artifact_blake3,
            "source_family": self.source_family,
            "source_track": self.source_track,
            "stage": self.stage.value,
            "rubric_domain": self.rubric_domain,
            "rubric_availability": self.rubric_availability.value,
            "rubric_resolution_reason": self.rubric_resolution_reason,
            "evidence_receipts": [row.to_document() for row in self.evidence_receipts],
            "policy_projection_blake3": self.policy_projection_blake3,
            "judge_projection_blake3": self.judge_projection_blake3,
        }

    def to_document(self) -> dict[str, Any]:
        return {**self.core_document(), "import_blake3": self.import_blake3}


@dataclass(frozen=True, slots=True)
class ImportedLegacyEpisode:
    episode: BenchmarkEpisode
    manifest: LegacyImportManifest


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LegacyImportError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise LegacyImportError(f"non-finite JSON constant is forbidden: {value}")


def _decode_json(payload: bytes, *, label: str) -> Any:
    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise LegacyImportError(f"{label} is not strict UTF-8 JSON") from exc


def _canonical_legacy_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise LegacyImportError("legacy registry is not canonical JSON-shaped data") from exc


def _legacy_sha256(payload: bytes) -> str:
    """Upstream-only compatibility verifier; never used as an EVA runtime ID."""

    return hashlib.sha256(payload).hexdigest()


def _require_exact_keys(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise LegacyImportError(f"{label} keys differ")
    return value


def _canonical_uuid(value: str, *, label: str) -> str:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError):
        raise LegacyImportError(f"{label} must be a UUID") from None
    if str(parsed) != value:
        raise LegacyImportError(f"{label} must use canonical UUID text")
    return value


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Deeply immutable assertion copy retained by a registry reference."""

    return MappingProxyType(
        {"path": tuple(value["path"]), "equals": freeze_json(value["equals"])}
    )


def _forbidden_policy_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold().replace("-", "_").replace(" ", "_")
            if any(part in normalized for part in _FORBIDDEN_POLICY_KEY_PARTS):
                return str(key)
            found = _forbidden_policy_key(child)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _forbidden_policy_key(child)
            if found is not None:
                return found
    return None


class LegacySupervisorV4Importer:
    """Select registry rows cheaply, then import selected payloads lazily."""

    def __init__(
        self,
        supervisor_root: str | Path,
        *,
        authority_root: str | Path | None = None,
    ) -> None:
        self.supervisor_root = Path(supervisor_root)
        self.authority_root = (
            Path(authority_root)
            if authority_root is not None
            else self.supervisor_root.parent.parent
        )
        self._authority_real = self._validate_root(self.authority_root, "authority root")
        self._supervisor_real = self._validate_root(self.supervisor_root, "supervisor root")
        try:
            self._supervisor_real.relative_to(self._authority_real)
        except ValueError:
            raise LegacyImportError("supervisor root escapes the authority root") from None
        registry_path = self.supervisor_root / REGISTRY_FILENAME
        self._registry_path = self._safe_file(
            registry_path,
            relative_to=self._authority_real,
            label="supervisor registry",
        )
        registry_bytes = self._read_regular_file(
            self._registry_path, label="supervisor registry", maximum_bytes=32 * 1024 * 1024
        )
        self._registry_file_blake3 = blake3_bytes(registry_bytes)
        self._registry = self._validate_registry(
            _decode_json(registry_bytes, label="supervisor registry")
        )
        self._candidate_index = {
            raw["candidate_id"]: raw for raw in self._registry["candidates"]
        }

    @staticmethod
    def _validate_root(path: Path, label: str) -> Path:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise LegacyImportError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise LegacyImportError(f"{label} must be a real directory, not a symlink")
        return path.resolve(strict=True)

    @staticmethod
    def _read_regular_file(path: Path, *, label: str, maximum_bytes: int) -> bytes:
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            raise LegacyImportError(f"{label} cannot be opened without following links") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise LegacyImportError(f"{label} must be a regular file")
            if metadata.st_size > maximum_bytes:
                raise LegacyImportError(f"{label} exceeds the bounded import size")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                payload = handle.read(maximum_bytes + 1)
            if len(payload) > maximum_bytes:
                raise LegacyImportError(f"{label} exceeds the bounded import size")
            return payload
        finally:
            os.close(descriptor)

    @staticmethod
    def _stream_regular_file_hashes(
        path: Path, *, label: str, maximum_bytes: int
    ) -> tuple[int, str, str]:
        """Hash a payload in bounded chunks without retaining its bytes."""

        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            raise LegacyImportError(f"{label} cannot be opened without following links") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise LegacyImportError(f"{label} must be a regular file")
            if metadata.st_size > maximum_bytes:
                raise LegacyImportError(f"{label} exceeds the bounded import size")
            legacy_hasher = hashlib.sha256()
            eva_hasher = blake3()
            byte_count = 0
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                while chunk := handle.read(1024 * 1024):
                    byte_count += len(chunk)
                    if byte_count > maximum_bytes:
                        raise LegacyImportError(f"{label} exceeds the bounded import size")
                    legacy_hasher.update(chunk)
                    eva_hasher.update(chunk)
            return byte_count, legacy_hasher.hexdigest(), eva_hasher.hexdigest()
        finally:
            os.close(descriptor)

    def _safe_relative_path(self, value: Any, *, label: str) -> PurePosixPath:
        if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
            raise LegacyImportError(f"{label} must be a non-empty POSIX relative path")
        relative = PurePosixPath(value)
        if (
            relative.is_absolute()
            or relative.as_posix() != value
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise LegacyImportError(f"{label} escapes or is not normalized")
        return relative

    def _safe_file(self, path: Path, *, relative_to: Path, label: str) -> Path:
        """Reject link traversal even when a link would resolve back inside the root."""

        try:
            relative = path.relative_to(relative_to)
        except ValueError:
            # A caller may supply the unresolved spelling under the same root.
            try:
                relative = path.resolve(strict=False).relative_to(relative_to)
            except ValueError:
                raise LegacyImportError(f"{label} escapes the authority root") from None
        cursor = relative_to
        for component in relative.parts:
            cursor = cursor / component
            try:
                metadata = cursor.lstat()
            except OSError as exc:
                raise LegacyImportError(f"{label} is unavailable") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise LegacyImportError(f"{label} traverses a symlink")
        if not stat.S_ISREG(cursor.lstat().st_mode):
            raise LegacyImportError(f"{label} must be a regular file")
        try:
            cursor.resolve(strict=True).relative_to(relative_to)
        except ValueError:
            raise LegacyImportError(f"{label} escapes the authority root") from None
        return cursor

    def _artifact_file(self, relative_path: Any, *, label: str) -> Path:
        relative = self._safe_relative_path(relative_path, label=label)
        return self._safe_file(
            self._authority_real.joinpath(*relative.parts),
            relative_to=self._authority_real,
            label=label,
        )

    def _validate_registry(self, value: Any) -> Mapping[str, Any]:
        registry = _require_exact_keys(
            value,
            {
                "schema",
                "campaign_id",
                "created_at_utc",
                "registry_revision",
                "slot_limits",
                "candidates",
                "registry_sha256",
            },
            label="supervisor registry",
        )
        if registry["schema"] != REGISTRY_SCHEMA:
            raise LegacyImportError("supervisor registry schema differs")
        if registry["campaign_id"] != CAMPAIGN_ID or registry["registry_revision"] != 4:
            raise LegacyImportError("source is not the frozen EvaMed supervisor-v4 registry")
        if not isinstance(registry["created_at_utc"], str) or not registry["created_at_utc"].endswith("Z"):
            raise LegacyImportError("supervisor registry creation time differs")
        limits = registry["slot_limits"]
        if not isinstance(limits, Mapping) or not limits:
            raise LegacyImportError("supervisor slot limits differ")
        if any(type(value) is not int or value < 0 for value in limits.values()):
            raise LegacyImportError("supervisor slot limit differs")
        candidates = registry["candidates"]
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 20_000:
            raise LegacyImportError("supervisor candidates must be a bounded non-empty list")
        previous = ""
        seen: set[str] = set()
        for index, raw in enumerate(candidates):
            candidate = self._validate_candidate(raw, index=index)
            candidate_id = candidate["candidate_id"]
            if candidate_id in seen or candidate_id < previous:
                raise LegacyImportError("supervisor candidate IDs must be unique and sorted")
            seen.add(candidate_id)
            previous = candidate_id
        claim = registry["registry_sha256"]
        if not isinstance(claim, str) or _SHA256.fullmatch(claim) is None:
            raise LegacyImportError("upstream registry SHA-256 claim differs")
        unsigned = dict(registry)
        unsigned.pop("registry_sha256")
        actual = _legacy_sha256(_canonical_legacy_bytes(unsigned))
        if actual != claim:
            raise LegacyImportError("upstream registry logical SHA-256 verification failed")
        return registry

    def _validate_candidate(self, raw: Any, *, index: int) -> Mapping[str, Any]:
        candidate = _require_exact_keys(
            raw,
            {
                "candidate_id",
                "family",
                "focus",
                "source_shard",
                "priority",
                "harness",
                "frozen_artifact",
                "frozen_assertions",
                "proofs",
            },
            label=f"candidate[{index}]",
        )
        candidate_id = candidate["candidate_id"]
        if not isinstance(candidate_id, str) or _SAFE_LABEL.fullmatch(candidate_id) is None:
            raise LegacyImportError(f"candidate[{index}] ID differs")
        if candidate["family"] not in FAMILIES or candidate["focus"] not in STAGES:
            raise LegacyImportError(f"candidate[{index}] family or focus differs")
        if not isinstance(candidate["source_shard"], str) or not candidate["source_shard"]:
            raise LegacyImportError(f"candidate[{index}] source shard differs")
        if type(candidate["priority"]) is not int or not 0 <= candidate["priority"] <= 1_000_000:
            raise LegacyImportError(f"candidate[{index}] priority differs")
        frozen = _require_exact_keys(
            candidate["frozen_artifact"], {"path", "sha256"}, label=f"candidate[{index}] artifact"
        )
        self._safe_relative_path(frozen["path"], label=f"candidate[{index}] artifact path")
        if not isinstance(frozen["sha256"], str) or _SHA256.fullmatch(frozen["sha256"]) is None:
            raise LegacyImportError(f"candidate[{index}] upstream artifact SHA-256 differs")
        assertions = candidate["frozen_assertions"]
        if not isinstance(assertions, list) or not 1 <= len(assertions) <= 64:
            raise LegacyImportError(f"candidate[{index}] frozen assertions differ")
        for assertion in assertions:
            row = _require_exact_keys(assertion, {"path", "equals"}, label="frozen assertion")
            if (
                not isinstance(row["path"], list)
                or not row["path"]
                or len(row["path"]) > 32
                or any(
                    isinstance(part, bool)
                    or not isinstance(part, (str, int))
                    or (isinstance(part, str) and not part)
                    or (isinstance(part, int) and part < 0)
                    for part in row["path"]
                )
            ):
                raise LegacyImportError("frozen assertion path differs")
        harness = candidate["harness"]
        if not isinstance(harness, Mapping) or harness.get("owner") != "rlevo-med-research":
            raise LegacyImportError(f"candidate[{index}] harness ownership differs")
        if harness.get("stage_chain") != ["S1", "S2", "S3", "S4", "S5"]:
            raise LegacyImportError(f"candidate[{index}] harness chain differs")
        if harness.get("s1_s5_harness_owned") is not True:
            raise LegacyImportError(f"candidate[{index}] harness control differs")
        contract_hash = harness.get("contract_sha256")
        if not isinstance(contract_hash, str) or _SHA256.fullmatch(contract_hash) is None:
            raise LegacyImportError(f"candidate[{index}] harness commitment differs")
        if not isinstance(candidate["proofs"], Mapping):
            raise LegacyImportError(f"candidate[{index}] proofs differ")
        return candidate

    @property
    def candidate_count(self) -> int:
        return len(self._registry["candidates"])

    @property
    def registry_file_blake3(self) -> str:
        return self._registry_file_blake3

    @property
    def upstream_registry_sha256(self) -> str:
        return str(self._registry["registry_sha256"])

    def iter_candidates(
        self,
        *,
        candidate_ids: Iterable[str] | None = None,
        families: Iterable[str] | None = None,
        stages: Iterable[str | Stage] | None = None,
    ) -> Iterator[LegacyCandidateRef]:
        """Yield registry-only references without opening any candidate payload."""

        selected_ids = None if candidate_ids is None else frozenset(candidate_ids)
        selected_families = None if families is None else frozenset(families)
        selected_stages = None
        if stages is not None:
            try:
                selected_stages = frozenset(
                    stage if isinstance(stage, Stage) else Stage(stage) for stage in stages
                )
            except ValueError:
                raise LegacyImportError("stage selector differs") from None
        if selected_families is not None and not selected_families.issubset(FAMILIES):
            raise LegacyImportError("family selector differs")
        for raw in self._registry["candidates"]:
            if selected_ids is not None and raw["candidate_id"] not in selected_ids:
                continue
            if selected_families is not None and raw["family"] not in selected_families:
                continue
            stage = Stage(raw["focus"])
            if selected_stages is not None and stage not in selected_stages:
                continue
            yield self._reference_from_raw(raw)

    @staticmethod
    def _reference_from_raw(raw: Mapping[str, Any]) -> LegacyCandidateRef:
        artifact = raw["frozen_artifact"]
        return LegacyCandidateRef(
            candidate_id=raw["candidate_id"],
            family=raw["family"],
            stage=Stage(raw["focus"]),
            priority=raw["priority"],
            source_shard=raw["source_shard"],
            artifact_relative_path=artifact["path"],
            upstream_artifact_sha256=artifact["sha256"],
            frozen_assertions=tuple(_frozen_mapping(row) for row in raw["frozen_assertions"]),
        )

    @staticmethod
    def _lookup_assertion(document: Any, path: Sequence[str | int]) -> Any:
        current = document
        for part in path:
            if isinstance(part, str) and isinstance(current, Mapping) and part in current:
                current = current[part]
            elif (
                isinstance(part, int)
                and not isinstance(part, bool)
                and isinstance(current, list)
                and part < len(current)
            ):
                current = current[part]
            else:
                raise LegacyImportError("frozen assertion does not resolve in source artifact")
        return current

    @staticmethod
    def _mapping_for(family: str, track: str) -> tuple[str, RubricAvailability, str]:
        if family == "agentclinic":
            if track != "AgentClinic":
                raise LegacyImportError("AgentClinic source track differs")
            return "agentclinic", RubricAvailability.READY, "direct family mapping"
        if family == "medxpertqa":
            if track != "MedX":
                raise LegacyImportError("MedXpertQA source track differs")
            return "medxpertqa", RubricAvailability.READY, "direct family mapping"
        if family == "healthbench-professional":
            return (
                "healthbench-professional",
                RubricAvailability.READY,
                "HealthBench Professional S1-S5/E2E tables are compiled",
            )
        if family != "automedbench":
            raise LegacyImportError("source family has no rubric mapping")
        mapping = {
            "Cls": "automedbench-classification",
            "Classification": "automedbench-classification",
            "Det": "automedbench-detection",
            "Detection": "automedbench-detection",
            "Seg": "automedbench-segmentation",
            "Segmentation": "automedbench-segmentation",
        }
        if track in mapping:
            return mapping[track], RubricAvailability.READY, "direct AutoMedBench track mapping"
        if track == "Research":
            return (
                "automedbench-research",
                RubricAvailability.READY,
                "AutoMedBench Research/report-generation S1-S5/E2E tables are compiled",
            )
        raise LegacyImportError("AutoMedBench source track has no explicit mapping")

    @staticmethod
    def _task_instruction(task_brief: Mapping[str, Any]) -> str:
        required = ("title", "objective", "research_question", "evidence_need")
        if any(not isinstance(task_brief.get(key), str) or not task_brief[key].strip() for key in required):
            raise LegacyImportError("candidate task brief differs")
        return (
            f"{task_brief['title']}\n\n"
            f"Objective: {task_brief['objective']}\n\n"
            f"Research question: {task_brief['research_question']}\n\n"
            f"Evidence requirement: {task_brief['evidence_need']}"
        )

    @staticmethod
    def _lineage(value: Any) -> tuple[dict[str, str], ...]:
        if not isinstance(value, list) or not value:
            raise LegacyImportError("candidate source lineage differs")
        result: list[dict[str, str]] = []
        keys = {"repository", "revision", "source_id", "source_split"}
        for raw in value:
            if not isinstance(raw, Mapping) or not keys.issubset(raw):
                raise LegacyImportError("candidate source lineage row differs")
            row = {key: raw[key] for key in sorted(keys)}
            if any(not isinstance(item, str) or not item for item in row.values()):
                raise LegacyImportError("candidate source lineage value differs")
            result.append(row)
        return tuple(result)

    def _evidence_receipts(self, value: Any) -> tuple[LegacyEvidenceReceipt, ...]:
        if not isinstance(value, list) or not value:
            raise LegacyImportError("candidate evidence catalog differs")
        receipts: list[LegacyEvidenceReceipt] = []
        seen: set[str] = set()
        for raw in value:
            if not isinstance(raw, Mapping):
                raise LegacyImportError("candidate evidence row differs")
            evidence_id = raw.get("evidence_id")
            relative_path = raw.get("source_relative_path")
            upstream_sha256 = raw.get("sha256")
            byte_count = raw.get("byte_count")
            content_kind = raw.get("content_kind")
            if (
                not isinstance(evidence_id, str)
                or not evidence_id
                or evidence_id in seen
                or not isinstance(relative_path, str)
                or not isinstance(upstream_sha256, str)
                or _SHA256.fullmatch(upstream_sha256) is None
                or type(byte_count) is not int
                or byte_count < 0
                or not isinstance(content_kind, str)
                or not content_kind
            ):
                raise LegacyImportError("candidate evidence commitment differs")
            seen.add(evidence_id)
            path = self._artifact_file(relative_path, label=f"evidence object {evidence_id}")
            actual_bytes, actual_sha256, actual_blake3 = self._stream_regular_file_hashes(
                path,
                label=f"evidence object {evidence_id}",
                maximum_bytes=64 * 1024 * 1024,
            )
            if actual_bytes != byte_count or actual_sha256 != upstream_sha256:
                raise LegacyImportError(f"upstream evidence verification failed: {evidence_id}")
            receipts.append(
                LegacyEvidenceReceipt(
                    evidence_id=evidence_id,
                    relative_path=relative_path,
                    byte_count=byte_count,
                    upstream_sha256=upstream_sha256,
                    content_blake3=actual_blake3,
                )
            )
        return tuple(receipts)

    def import_candidate(self, reference: LegacyCandidateRef) -> ImportedLegacyEpisode:
        """Verify and import exactly one selected candidate without source writes."""

        if not isinstance(reference, LegacyCandidateRef):
            raise LegacyImportError("import requires a registry-issued candidate reference")
        registered = self._candidate_index.get(reference.candidate_id)
        if registered is None or self._reference_from_raw(registered) != reference:
            raise LegacyImportError("candidate reference is not the frozen registry commitment")
        path = self._artifact_file(reference.artifact_relative_path, label="candidate artifact")
        payload = self._read_regular_file(
            path, label="candidate artifact", maximum_bytes=16 * 1024 * 1024
        )
        if _legacy_sha256(payload) != reference.upstream_artifact_sha256:
            raise LegacyImportError("upstream artifact SHA-256 verification failed")
        document = _decode_json(payload, label="candidate artifact")
        if not isinstance(document, Mapping) or document.get("schema") != ARTIFACT_SCHEMA:
            raise LegacyImportError("candidate artifact schema differs")
        for assertion in reference.frozen_assertions:
            path_parts = assertion["path"]
            if freeze_json(self._lookup_assertion(document, path_parts)) != assertion["equals"]:
                raise LegacyImportError("candidate frozen assertion failed")
        sandbox = document.get("sandbox")
        if not isinstance(sandbox, Mapping):
            raise LegacyImportError("candidate sandbox metadata differs")
        if sandbox.get("sandbox_id") != reference.candidate_id:
            raise LegacyImportError("registry candidate and artifact sandbox differ")
        if sandbox.get("focus") != reference.stage.value:
            raise LegacyImportError("registry candidate and artifact stage differ")
        track = sandbox.get("track")
        if not isinstance(track, str) or not track:
            raise LegacyImportError("candidate source track differs")
        domain, rubric_availability, rubric_reason = self._mapping_for(reference.family, track)
        task_brief = document.get("task_brief")
        if not isinstance(task_brief, Mapping):
            raise LegacyImportError("candidate task brief differs")
        instruction = self._task_instruction(task_brief)
        task_ids = task_brief.get("task_ids")
        if (
            not isinstance(task_ids, list)
            or not task_ids
            or any(not isinstance(value, str) or not value for value in task_ids)
        ):
            raise LegacyImportError("candidate task IDs differ")
        lineage = self._lineage(document.get("source_lineage"))
        evidence_rows = document.get("evidence_objects")
        evidence_receipts = self._evidence_receipts(evidence_rows)
        evidence_catalog = []
        assert isinstance(evidence_rows, list)  # narrowed by _evidence_receipts
        for row in evidence_rows:
            evidence_catalog.append(
                {
                    "evidence_id": row["evidence_id"],
                    "content_kind": row.get("content_kind", "unknown"),
                    "byte_count": row["byte_count"],
                    "access": "typed_host_tool_only",
                }
            )
        runtime = document.get("runtime")
        if not isinstance(runtime, Mapping):
            raise LegacyImportError("candidate runtime contract differs")
        policy_context = {
            "source_family": reference.family,
            "source_track": track,
            "source_lineage": list(lineage),
            "task_brief": {
                key: task_brief[key]
                for key in ("title", "objective", "research_question", "evidence_need", "task_ids")
                if key in task_brief
            },
            "content_scope": document.get("content_scope"),
            "safety": {
                key: document.get("safety", {})[key]
                for key in (
                    "care_facing",
                    "clinical_review_status",
                    "individual_level_data",
                    "phi_scan_findings",
                    "source_data_synthetic",
                )
                if isinstance(document.get("safety"), Mapping)
                and key in document.get("safety", {})
            },
            "runtime_limits": {
                key: runtime[key]
                for key in ("clock_utc", "execution_limits", "policy_budgets")
                if key in runtime
            },
            "evidence_catalog": evidence_catalog,
            "rubric_availability": rubric_availability.value,
        }
        forbidden = _forbidden_policy_key(policy_context)
        if forbidden is not None:
            raise LegacyImportError(
                f"candidate policy projection contains private key {forbidden!r}"
            )
        private_material = {
            key: document[key] for key in sorted(_PRIVATE_TOP_LEVEL_KEYS) if key in document
        }
        judge_only_reference = {
            "schema": "eva.legacy-supervisor-v4-judge-reference.v1",
            "candidate_id": reference.candidate_id,
            "frozen_assertions": [dict(row) for row in reference.frozen_assertions],
            "verifier_contract": runtime,
            "rubric_noncritical_families": document.get("rubric_noncritical_families", {}),
            "evidence_objects": evidence_rows,
            "evidence_source_verification": document.get("evidence_source_verification"),
            "evidence_source_verification_attestation": document.get(
                "evidence_source_verification_attestation"
            ),
            "private_material": private_material,
        }
        namespace = uuid5(NAMESPACE_URL, f"eva-agent:{CAMPAIGN_ID}:legacy-v4")
        episode_id = str(uuid5(namespace, f"episode:{reference.candidate_id}"))
        artifact_blake3 = blake3_bytes(payload)
        import_id = str(uuid5(namespace, f"import:{reference.candidate_id}:{artifact_blake3}"))
        _canonical_uuid(episode_id, label="episode_id")
        _canonical_uuid(import_id, label="import_id")
        benchmark_names = {
            "agentclinic": "AgentClinic",
            "automedbench": "AutoMedBench",
            "healthbench-professional": "HealthBench Professional",
            "medxpertqa": "MedXpertQA",
        }
        episode = BenchmarkEpisode(
            episode_id=episode_id,
            source=BenchmarkSource(
                benchmark=benchmark_names[reference.family],
                source_file=reference.artifact_relative_path,
                source_revision=lineage[0]["revision"],
            ),
            domain=domain,
            stage=reference.stage,
            instruction=instruction,
            policy_context=policy_context,
            initial_files={"TASK.md": (instruction + "\n").encode("utf-8")},
            judge_only_reference=judge_only_reference,
        )
        provisional = LegacyImportManifest(
            import_id=import_id,
            episode_id=episode_id,
            candidate_id=reference.candidate_id,
            campaign_id=CAMPAIGN_ID,
            registry_revision=4,
            registry_file_blake3=self._registry_file_blake3,
            upstream_registry_sha256=self.upstream_registry_sha256,
            artifact_relative_path=reference.artifact_relative_path,
            artifact_byte_count=len(payload),
            upstream_artifact_sha256=reference.upstream_artifact_sha256,
            artifact_blake3=artifact_blake3,
            source_family=reference.family,
            source_track=track,
            stage=reference.stage,
            rubric_domain=domain,
            rubric_availability=rubric_availability,
            rubric_resolution_reason=rubric_reason,
            evidence_receipts=evidence_receipts,
            policy_projection_blake3=blake3_hex(episode.policy_context),
            judge_projection_blake3=blake3_hex(episode.judge_only_reference),
            import_blake3="",
        )
        manifest = replace(
            provisional,
            import_blake3=blake3_hex(provisional.core_document()),
        )
        return ImportedLegacyEpisode(episode=episode, manifest=manifest)

    def iter_imports(
        self,
        *,
        candidate_ids: Iterable[str] | None = None,
        families: Iterable[str] | None = None,
        stages: Iterable[str | Stage] | None = None,
        tracks: Iterable[str] | None = None,
        rubric_ready_only: bool = False,
        limit: int | None = None,
    ) -> Iterator[ImportedLegacyEpisode]:
        """Lazily verify/import selected rows; no unselected artifact is opened."""

        if limit is not None and (type(limit) is not int or limit < 0):
            raise LegacyImportError("import limit differs")
        selected_tracks = None if tracks is None else frozenset(tracks)
        emitted = 0
        for reference in self.iter_candidates(
            candidate_ids=candidate_ids, families=families, stages=stages
        ):
            if limit is not None and emitted >= limit:
                break
            imported = self.import_candidate(reference)
            if selected_tracks is not None and imported.manifest.source_track not in selected_tracks:
                continue
            if (
                rubric_ready_only
                and imported.manifest.rubric_availability is not RubricAvailability.READY
            ):
                continue
            emitted += 1
            yield imported


__all__ = [
    "ImportedLegacyEpisode",
    "LegacyCandidateRef",
    "LegacyEvidenceReceipt",
    "LegacyImportError",
    "LegacyImportManifest",
    "LegacySupervisorV4Importer",
    "RubricAvailability",
]
