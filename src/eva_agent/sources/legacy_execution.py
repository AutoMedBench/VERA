"""Candidate-scoped execution bindings for signed EvaMed legacy sandboxes.

The v24 supervisor is an authority over *which* candidates were promoted.  A
promotion is not, by itself, an executable sandbox.  This module follows the
latest signed promoted proof to its exact policy and construction manifest,
reopens the candidate's signed construction artifacts, and binds the five
public policy tools to the original EvaMed handlers.

There is intentionally no merged/global medical tool registry here.  A
``CandidateExecutionBinding`` owns one exact policy catalog and one stateful
handler backend.  Missing or altered policy, construction, runtime code,
signing material, evidence, or Docker prerequisites fail before a provider can
be called.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
from threading import Lock, RLock
from types import MappingProxyType, ModuleType
from typing import Any, Mapping, Sequence
from uuid import UUID

from eva_agent.pipeline.contracts import JsonValue, Stage, freeze_json
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes
from eva_agent.pipeline.tools import ToolDefinition, ToolRegistry
from eva_agent.pipeline.workspace import FilesystemSandbox

from .supervisor_v24_readiness import (
    ConstructionReadiness,
    SignedSupervisorV24Readiness,
    load_signed_supervisor_v24_readiness,
)


POLICY_SCHEMA = "rlevo.med-research-sandbox-policy.v2"
CONSTRUCTION_SCHEMA = "rlevo.med-research-candidate-construction-manifest.v2"
EXPECTED_TOOL_NAMES = (
    "execute_code",
    "materialize_evidence_selection",
    "materialize_plan",
    "retrieve_frozen_evidence",
    "submit_results",
)
_POLICY_ROLES = frozenset(
    {"promoted_policy", "attempt2_promoted_policy", "promotion_v5_promoted_policy"}
)
_MANIFEST_ROLES = frozenset(
    {"construction_manifest", "candidate_construction_manifest"}
)
_ATTESTATION_ROLES = frozenset(
    {"construction_attestation", "candidate_construction_attestation"}
)
_MANDATORY_ARTIFACTS = frozenset(
    {
        "construction_input",
        "construction_solver_rollout",
        "construction_solver_rollout_attestation",
        "construction_solver_rollout_receipt",
        "evidence_manifest",
        "evidence_source_verification",
        "evidence_source_verification_attestation",
        "host_trust_store",
        "policy",
        "private_evaluation",
        "s1_plan_contract",
        "s2_evidence_contract",
        "s3_execution_contract",
        "s4_execution_contract",
        "s5_submission_contract",
    }
)
_AUTOMEDBENCH_SANDBOX = re.compile(
    r"rlevo-medres-automedbench-(s[1-5]|e2e)-([0-9a-f]{16})"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,255}$")
_MAX_JSON_BYTES = 40 * 1024 * 1024
_LEGACY_IMPORT_LOCK = Lock()


class ProductionSourceBlocker(ValueError):
    """A production execution prerequisite is absent or differs."""


def _legacy_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProductionSourceBlocker("legacy value is not canonical JSON") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_value(value: Any) -> str:
    return _sha256_bytes(_legacy_json_bytes(value))


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProductionSourceBlocker(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ProductionSourceBlocker(f"non-finite JSON constant is forbidden: {value}")


def _real_directory(value: str | Path, *, label: str, create: bool = False) -> Path:
    path = Path(value)
    if create and not path.exists() and not path.is_symlink():
        path.mkdir(parents=True, mode=0o700)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductionSourceBlocker(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ProductionSourceBlocker(f"{label} must be a real directory")
    return path.resolve(strict=True)


def _safe_file(path: Path, *, root: Path, label: str) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ProductionSourceBlocker(f"{label} escapes its authority root") from None
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        try:
            metadata = cursor.lstat()
        except OSError as exc:
            raise ProductionSourceBlocker(f"{label} is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ProductionSourceBlocker(f"{label} traverses a symlink")
    metadata = cursor.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ProductionSourceBlocker(f"{label} must be one regular file")
    return cursor


def _relative_file(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ProductionSourceBlocker(f"{label} path differs")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ProductionSourceBlocker(f"{label} path is unsafe")
    return _safe_file(root.joinpath(*pure.parts), root=root, label=label)


def _read_bytes(path: Path, *, root: Path, label: str, maximum: int = _MAX_JSON_BYTES) -> bytes:
    target = _safe_file(path, root=root, label=label)
    descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ProductionSourceBlocker(f"{label} byte bound differs")
        payload = b""
        while len(payload) <= maximum:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - len(payload)))
            if not chunk:
                break
            payload += chunk
        after = os.fstat(descriptor)
        if (
            len(payload) != before.st_size
            or len(payload) > maximum
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise ProductionSourceBlocker(f"{label} changed while opening")
        return payload
    finally:
        os.close(descriptor)


def _decode_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except ProductionSourceBlocker:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ProductionSourceBlocker(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ProductionSourceBlocker(f"{label} must be a JSON object")
    return value


def _read_json(path: Path, *, root: Path, label: str) -> tuple[Mapping[str, Any], bytes]:
    payload = _read_bytes(path, root=root, label=label)
    return _decode_json(payload, label=label), payload


def _artifact_ref(
    value: Any,
    *,
    root: Path,
    label: str,
) -> tuple[Path, Mapping[str, Any], bytes]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "byte_count"}:
        raise ProductionSourceBlocker(f"{label} reference keys differ")
    digest = value["sha256"]
    byte_count = value["byte_count"]
    if (
        not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or type(byte_count) is not int
        or byte_count < 1
    ):
        raise ProductionSourceBlocker(f"{label} reference integrity differs")
    path = _relative_file(root, value["path"], label=label)
    payload = _read_bytes(path, root=root, label=label)
    if len(payload) != byte_count or _sha256_bytes(payload) != digest:
        raise ProductionSourceBlocker(f"{label} bytes differ from the signed pointer")
    return path, value, payload


def _proof_ref(
    value: Any,
    *,
    root: Path,
    label: str,
) -> tuple[Path, bytes]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ProductionSourceBlocker(f"{label} proof reference keys differ")
    digest = value["sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ProductionSourceBlocker(f"{label} proof reference hash differs")
    path = _relative_file(root, value["path"], label=label)
    payload = _read_bytes(path, root=root, label=label)
    if _sha256_bytes(payload) != digest:
        raise ProductionSourceBlocker(f"{label} proof bytes differ")
    return path, payload


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(child) for child in value]
    return value


def _uuid(value: str, *, label: str) -> str:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError):
        raise ProductionSourceBlocker(f"{label} must be a UUID") from None
    if str(parsed) != value:
        raise ProductionSourceBlocker(f"{label} must use canonical UUID text")
    return value


def _project_automedbench_historical_episode_alias(
    *,
    base_policy: Mapping[str, Any],
    templates: Mapping[str, Mapping[str, Any]],
    sha256_value: Any,
) -> dict[str, dict[str, Any]]:
    """Project one signed historical AutoMedBench episode naming alias.

    Early AutoMedBench construction used ``evamed-automedb`` for the episode
    prefix while its signed sandbox identity used ``automedbench``.  The legacy
    panel projector deliberately accepts only literal suffix identities, so
    these otherwise exact promoted bindings stop before MCP construction.  Keep
    this compatibility rule narrow: both identities must encode the same stage
    and 16-hex task suffix, and the exact template episode must occur publicly.
    """

    copied = {name: deepcopy(dict(value)) for name, value in templates.items()}
    match = _AUTOMEDBENCH_SANDBOX.fullmatch(str(base_policy.get("sandbox_id", "")))
    episodes = {
        copied[name].get("episode_id")
        for name in ("s1", "s2", "s3", "s4", "s5")
    }
    if match is None or len(episodes) != 1:
        return copied
    template_episode = next(iter(episodes))
    expected_episode = f"evamed-automedb-{match.group(1)}-{match.group(2)}"
    messages = base_policy.get("messages")
    if template_episode != expected_episode or not isinstance(messages, Sequence) or not any(
        isinstance(message, Mapping)
        and isinstance(message.get("content"), str)
        and template_episode in message["content"]
        for message in messages
    ):
        return copied
    identity = {
        "domain": "rlevo.med-research-panel-template-projection.v1",
        "policy_sha256": sha256_value(base_policy),
        "source_template_sha256": {
            name: sha256_value(copied[name])
            for name in ("s1", "s2", "s3", "s4", "s5")
        },
    }
    panel_episode = "panel-template-" + sha256_value(identity)[:24]

    def replace_episode(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: replace_episode(child) for key, child in value.items()}
        if isinstance(value, list):
            return [replace_episode(child) for child in value]
        return panel_episode if value == template_episode else value

    projected = {name: replace_episode(value) for name, value in copied.items()}
    projected["s2"]["s1_plan_contract_sha256"] = sha256_value(projected["s1"])
    projected["s5"]["execution_contract_sha256"] = sha256_value(projected["s4"])
    return projected


@dataclass(frozen=True, slots=True)
class CandidateToolCatalogEntry:
    name: str
    description: str
    input_schema: Mapping[str, JsonValue]
    visibility: str = "actor_public"
    handler_origin: str = "signed_legacy_evamed"
    parallel_safe: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.description or self.visibility != "actor_public":
            raise ProductionSourceBlocker("candidate tool catalog entry differs")
        schema = freeze_json(self.input_schema)
        if not isinstance(schema, Mapping):
            raise ProductionSourceBlocker("candidate tool input schema is not an object")
        object.__setattr__(self, "input_schema", schema)

    def to_document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": _thaw(self.input_schema),
            "visibility": self.visibility,
            "handler_origin": self.handler_origin,
            "parallel_safe": self.parallel_safe,
        }


@dataclass(frozen=True, slots=True)
class CandidateBindingReference:
    source_candidate_id: str
    source_family: str
    stage: Stage
    promoted_subject_path: Path
    promoted_subject_blake3: str
    source_policy_path: Path
    source_policy_upstream_sha256: str
    source_policy_blake3: str
    construction_manifest_path: Path
    construction_manifest_upstream_sha256: str
    construction_manifest_blake3: str
    construction_attestation_path: Path
    construction_attestation_blake3: str
    proof_root_blake3: str

    def to_document(self) -> dict[str, Any]:
        return {
            "source_candidate_id": self.source_candidate_id,
            "source_family": self.source_family,
            "stage": self.stage.value,
            "promoted_subject_path": str(self.promoted_subject_path),
            "promoted_subject_blake3": self.promoted_subject_blake3,
            "source_policy_path": str(self.source_policy_path),
            "source_policy_upstream_sha256": self.source_policy_upstream_sha256,
            "source_policy_blake3": self.source_policy_blake3,
            "construction_manifest_path": str(self.construction_manifest_path),
            "construction_manifest_upstream_sha256": (
                self.construction_manifest_upstream_sha256
            ),
            "construction_manifest_blake3": self.construction_manifest_blake3,
            "construction_attestation_path": str(self.construction_attestation_path),
            "construction_attestation_blake3": self.construction_attestation_blake3,
            "proof_root_blake3": self.proof_root_blake3,
        }


@dataclass(frozen=True, slots=True)
class LegacyExecutionBindingCatalog:
    authority_blake3: str
    legacy_runtime_source_blake3: str
    rows: tuple[CandidateBindingReference, ...]
    catalog_blake3: str

    @property
    def candidate_count(self) -> int:
        return len(self.rows)

    def core_document(self) -> dict[str, Any]:
        return {
            "schema": "eva.legacy-candidate-execution-binding-catalog.v1",
            "authority_blake3": self.authority_blake3,
            "legacy_runtime_source_blake3": self.legacy_runtime_source_blake3,
            "coverage": "signed_v24_promoted_only",
            "candidate_count": len(self.rows),
            "rows": [row.to_document() for row in self.rows],
        }


@dataclass(frozen=True, slots=True)
class CandidateExecutionBinding:
    """One exact signed policy plus real, stateful candidate handlers."""

    candidate_id: str
    source_candidate_id: str
    source_family: str
    stage: Stage
    source_policy_path: Path
    source_policy_upstream_sha256: str
    source_policy_blake3: str
    construction_root: Path
    construction_manifest_path: Path
    construction_manifest_blake3: str
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
            "schema": "eva.candidate-execution-binding.v1",
            "candidate_id": self.candidate_id,
            "source_candidate_id": self.source_candidate_id,
            "source_family": self.source_family,
            "stage": self.stage.value,
            "source_policy_path": str(self.source_policy_path),
            "source_policy_upstream_sha256": self.source_policy_upstream_sha256,
            "source_policy_blake3": self.source_policy_blake3,
            "construction_root": str(self.construction_root),
            "construction_manifest_path": str(self.construction_manifest_path),
            "construction_manifest_blake3": self.construction_manifest_blake3,
            "legacy_runtime_source_blake3": self.legacy_runtime_source_blake3,
            "resolver_catalog_blake3": self.resolver_catalog_blake3,
            "tool_catalog": [row.to_document() for row in self.tool_catalog],
            "tool_catalog_blake3": self.tool_catalog_blake3,
            "public_runtime_context": _thaw(self.public_runtime_context),
            "initial_workspace_files": {
                path: blake3_bytes(payload)
                for path, payload in sorted(self.initial_workspace_files.items())
            },
        }


@dataclass(frozen=True, slots=True)
class _LegacyModules:
    package: ModuleType
    canonical: ModuleType
    execution: ModuleType
    host_receipts: ModuleType
    key_management: ModuleType
    panel_dispatch: ModuleType
    agent_rollout: ModuleType
    stage_services: ModuleType
    validation: ModuleType
    source_blake3: str


def _load_legacy_modules(python_root: Path) -> _LegacyModules:
    package_dir = _real_directory(
        python_root / "rlevo_med_research", label="legacy EvaMed Python package"
    )
    sources: list[dict[str, str]] = []
    for path in sorted(
        (
            item
            for item in package_dir.rglob("*")
            if item.suffix in {".py", ".json"}
        ),
        key=lambda item: item.relative_to(package_dir).as_posix(),
    ):
        target = _safe_file(path, root=package_dir, label="legacy runtime module")
        sources.append(
            {
                "path": target.relative_to(package_dir).as_posix(),
                "blake3": blake3_bytes(target.read_bytes()),
            }
        )
    source_blake3 = blake3_hex(
        {"schema": "eva.legacy-runtime-python-source.v1", "files": sources}
    )
    # The pinned package's schema loader deliberately uses the canonical
    # ``rlevo_med_research.schemas`` resource name.  Load the verified tree
    # under that exact name; silently coexisting with another installation
    # would make schema/handler provenance ambiguous, so that case is blocked.
    alias = "rlevo_med_research"
    with _LEGACY_IMPORT_LOCK:
        package = sys.modules.get(alias)
        if package is not None:
            existing = Path(package.__file__ or "").resolve(strict=True).parent
            if existing != package_dir:
                raise ProductionSourceBlocker(
                    "another rlevo_med_research package is already imported"
                )
        if package is None:
            init_path = _safe_file(
                package_dir / "__init__.py", root=package_dir, label="legacy package init"
            )
            spec = importlib.util.spec_from_file_location(
                alias,
                init_path,
                submodule_search_locations=[str(package_dir)],
            )
            if spec is None or spec.loader is None:
                raise ProductionSourceBlocker("legacy EvaMed package loader is unavailable")
            package = importlib.util.module_from_spec(spec)
            sys.modules[alias] = package
            try:
                spec.loader.exec_module(package)
            except BaseException:
                sys.modules.pop(alias, None)
                raise
        modules = {
            name: importlib.import_module(f"{alias}.{name}")
            for name in (
                "canonical",
                "execution",
                "host_receipts",
                "key_management",
                "panel_dispatch",
                "agent_rollout",
                "stage_services",
                "validation",
            )
        }
    for name, module in modules.items():
        module_path = Path(module.__file__ or "")
        try:
            module_path.resolve(strict=True).relative_to(package_dir)
        except (OSError, ValueError):
            raise ProductionSourceBlocker(f"legacy {name} module escaped pinned source") from None
    required = {
        "stage_services": (
            "materialize_s1_plan_attempt",
            "retrieve_frozen_evidence",
            "materialize_s2_selection_attempt",
        ),
        "validation": ("validate_stage_plan_artifact",),
        "execution": ("execute_python_contract", "finalize_submission"),
        "panel_dispatch": ("_panelize_constructor_templates", "_validate_contract_templates"),
        "agent_rollout": (
            "_stage_tool_result",
            "_tool_result_from_execution",
            "_instantiate_submission_contract",
            "_verified_source_binding",
        ),
    }
    for module_name, symbols in required.items():
        if any(not callable(getattr(modules[module_name], symbol, None)) for symbol in symbols):
            raise ProductionSourceBlocker(
                f"legacy {module_name} executable handler surface differs"
            )
    return _LegacyModules(
        package=package,
        canonical=modules["canonical"],
        execution=modules["execution"],
        host_receipts=modules["host_receipts"],
        key_management=modules["key_management"],
        panel_dispatch=modules["panel_dispatch"],
        agent_rollout=modules["agent_rollout"],
        stage_services=modules["stage_services"],
        validation=modules["validation"],
        source_blake3=source_blake3,
    )


@dataclass(slots=True)
class _WorkspaceSession:
    root: Path
    lock: RLock
    s1: dict[str, Any]
    s2_template: dict[str, Any]
    s3_template: dict[str, Any]
    s4_template: dict[str, Any]
    s5_template: dict[str, Any]
    s1_receipt: dict[str, Any] | None = None
    s2_contract: dict[str, Any] | None = None
    retrieval_receipts: list[dict[str, Any]] | None = None
    retrieval_ids: set[str] | None = None
    s2_receipt: dict[str, Any] | None = None
    s3_success: tuple[dict[str, Any], Path] | None = None
    s4_success: tuple[dict[str, Any], dict[str, Any], Path] | None = None
    s1_attempted: bool = False
    selection_attempts: int = 0
    s3_attempts: int = 0
    s4_attempts: int = 0
    submission_attempted: bool = False

    def __post_init__(self) -> None:
        if self.retrieval_receipts is None:
            self.retrieval_receipts = []
        if self.retrieval_ids is None:
            self.retrieval_ids = set()


class _LegacyCandidateToolBackend:
    def __init__(
        self,
        *,
        modules: _LegacyModules,
        binding_identity: Mapping[str, Any],
        runtime_state_root: Path,
        contracts: Mapping[str, Mapping[str, Any]],
        evidence_root: Path,
        image_refs: Mapping[str, str],
        trust_store: Mapping[str, str],
        key_id: str,
        private_key: Any,
        docker_binary: str,
    ) -> None:
        self.modules = modules
        self.binding_seed = blake3_hex(binding_identity)
        self.runtime_state_root = runtime_state_root
        self.contracts = {name: deepcopy(dict(value)) for name, value in contracts.items()}
        self.evidence_root = evidence_root
        self.image_refs = dict(image_refs)
        self.trust_store = dict(trust_store)
        self.key_id = key_id
        self.private_key = private_key
        self.docker_binary = docker_binary
        self._sessions: dict[Path, _WorkspaceSession] = {}
        self._lock = RLock()

    def _session(self, workspace: FilesystemSandbox) -> _WorkspaceSession:
        workspace_root = workspace.root.resolve(strict=True)
        with self._lock:
            existing = self._sessions.get(workspace_root)
            if existing is not None:
                return existing
            identity = blake3_bytes(str(workspace_root).encode("utf-8"))[:32]
            candidate_root = self.runtime_state_root / self.binding_seed
            candidate_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if candidate_root.is_symlink() or not candidate_root.is_dir():
                raise ProductionSourceBlocker("candidate runtime state root is unsafe")
            session_root = candidate_root / f"workspace-{identity}"
            try:
                session_root.mkdir(mode=0o700)
            except FileExistsError:
                raise ProductionSourceBlocker(
                    "candidate workspace runtime identity was already consumed"
                ) from None
            session = _WorkspaceSession(
                root=session_root,
                lock=RLock(),
                s1=deepcopy(self.contracts["s1"]),
                s2_template=deepcopy(self.contracts["s2"]),
                s3_template=deepcopy(self.contracts["s3"]),
                s4_template=deepcopy(self.contracts["s4"]),
                s5_template=deepcopy(self.contracts["s5"]),
            )
            self._sessions[workspace_root] = session
            return session

    def handler(self, name: str):
        if name not in EXPECTED_TOOL_NAMES:
            raise ProductionSourceBlocker(f"no exact legacy handler exists for {name}")

        def invoke(workspace: FilesystemSandbox, arguments: Mapping[str, JsonValue]) -> Any:
            session = self._session(workspace)
            with session.lock:
                return getattr(self, f"_handle_{name}")(
                    workspace, session, _thaw(arguments)
                )

        return invoke

    def _mirror(
        self,
        workspace: FilesystemSandbox,
        *,
        host_path: Path,
        relative_path: str,
    ) -> None:
        payload = host_path.read_bytes()
        workspace.write_bytes(relative_path, payload, create_only=True)

    def _handle_materialize_plan(
        self,
        workspace: FilesystemSandbox,
        session: _WorkspaceSession,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if session.s1_attempted:
            return {
                "stage": "S1",
                "effect": "plan_materialization",
                "gate_passed": False,
                "error": "single_s1_attempt_already_consumed",
            }
        # Schema validation is a pre-effect protocol check, not the candidate's
        # one semantic S1 attempt.  Keep the exact validator diagnostic (for
        # example ``$.budgets.stage_turns``) visible to the actor and allow one
        # clean corrected call.  Only a schema-valid submission crosses the
        # immutable signed-attempt boundary below.
        try:
            self.modules.validation.validate_stage_plan_artifact(arguments)
        except self.modules.validation.ContractError as exc:
            # The pinned validator owns the wording, including the exact JSON
            # path. Returning it as a non-effect observation keeps it visible
            # to the actor; ParallelToolRuntime intentionally reduces raised
            # exceptions to their type name in immutable receipts.
            return {
                "stage": "S1",
                "effect": "plan_materialization",
                "gate_passed": False,
                "attempt_consumed": False,
                "error": "schema_invalid_pre_effect",
                "diagnostic": str(exc),
            }
        session.s1_attempted = True
        attempt_dir = session.root / "stages" / "s1-attempt-1"
        receipt = self.modules.stage_services.materialize_s1_plan_attempt(
            session.s1,
            arguments,
            attempt_dir=attempt_dir,
            key_id=self.key_id,
            private_key=self.private_key,
        )
        result = self.modules.agent_rollout._stage_tool_result(receipt, self.trust_store)
        if result["gate_passed"]:
            contract = deepcopy(session.s2_template)
            contract["s1_receipt_sha256"] = self.modules.canonical.sha256_value(receipt)
            self.modules.stage_services.validate_evidence_service_contract(contract)
            artifact_relative = session.s1["stage_artifacts"]["S1"]
            self._mirror(
                workspace,
                host_path=attempt_dir / "workspace" / artifact_relative,
                relative_path=artifact_relative,
            )
            session.s1_receipt = receipt
            session.s2_contract = contract
            result["next_stage"] = "S2"
            result["evidence_contract_sha256"] = self.modules.canonical.sha256_value(
                contract
            )
        return result

    def _handle_retrieve_frozen_evidence(
        self,
        workspace: FilesystemSandbox,
        session: _WorkspaceSession,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del workspace
        if set(arguments) != {"evidence_id"} or not isinstance(
            arguments["evidence_id"], str
        ):
            raise ValueError("retrieval arguments differ from exact source policy")
        if session.s1_receipt is None or session.s2_contract is None:
            return {
                "stage": "S2",
                "effect": "frozen_evidence_retrieval",
                "gate_passed": False,
                "error": "verified_s1_prerequisite_required",
            }
        evidence_id = arguments["evidence_id"]
        assert session.retrieval_ids is not None
        assert session.retrieval_receipts is not None
        if evidence_id in session.retrieval_ids:
            return {
                "stage": "S2",
                "effect": "frozen_evidence_retrieval",
                "gate_passed": False,
                "error": "evidence_id_already_retrieved",
            }
        maximum = session.s2_contract["limits"]["max_retrievals"]
        if len(session.retrieval_receipts) >= maximum:
            return {
                "stage": "S2",
                "effect": "frozen_evidence_retrieval",
                "gate_passed": False,
                "error": "retrieval_budget_exhausted",
            }
        index = len(session.retrieval_receipts) + 1
        episode_dir = session.root / "stages" / "s2"
        observed = self.modules.stage_services.retrieve_frozen_evidence(
            session.s2_contract,
            evidence_id,
            retrieval_index=index,
            s1_receipt=session.s1_receipt,
            evidence_root=self.evidence_root,
            episode_dir=episode_dir,
            trust_store=self.trust_store,
            key_id=self.key_id,
            private_key=self.private_key,
        )
        receipt = observed["host_receipt"]
        result = self.modules.agent_rollout._stage_tool_result(receipt, self.trust_store)
        result["evidence_id"] = evidence_id
        if result["gate_passed"]:
            session.retrieval_receipts.append(receipt)
            session.retrieval_ids.add(evidence_id)
            result["retrieval_receipt_sha256"] = self.modules.canonical.sha256_value(
                receipt
            )
            payload = self.modules.host_receipts.verify_host_receipt(
                receipt, self.trust_store
            )
            digest = payload["observations"]["evidence"]["content_sha256"]
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ProductionSourceBlocker(
                    "signed retrieval lacks one verified evidence digest"
                )
            result["verified_evidence_content_sha256"] = digest
            result["evidence"] = observed["evidence"]
        return result

    def _handle_materialize_evidence_selection(
        self,
        workspace: FilesystemSandbox,
        session: _WorkspaceSession,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if session.s1_receipt is None or session.s2_contract is None:
            return {
                "stage": "S2",
                "effect": "evidence_selection_materialization",
                "gate_passed": False,
                "error": "verified_s1_prerequisite_required",
            }
        assert session.retrieval_receipts is not None
        if not session.retrieval_receipts:
            return {
                "stage": "S2",
                "effect": "evidence_selection_materialization",
                "gate_passed": False,
                "error": "successful_frozen_evidence_retrieval_required",
            }
        if session.s2_receipt is not None:
            return {
                "stage": "S2",
                "effect": "evidence_selection_materialization",
                "gate_passed": False,
                "error": "selection_already_materialized",
            }
        maximum = session.s2_contract["limits"]["max_selection_attempts"]
        if session.selection_attempts >= maximum:
            return {
                "stage": "S2",
                "effect": "evidence_selection_materialization",
                "gate_passed": False,
                "error": "selection_budget_exhausted",
            }
        session.selection_attempts += 1
        episode_dir = session.root / "stages" / "s2"
        receipt = self.modules.stage_services.materialize_s2_selection_attempt(
            session.s2_contract,
            arguments,
            session.retrieval_receipts,
            selection_attempt_index=session.selection_attempts,
            s1_receipt=session.s1_receipt,
            evidence_root=self.evidence_root,
            episode_dir=episode_dir,
            trust_store=self.trust_store,
            key_id=self.key_id,
            private_key=self.private_key,
        )
        result = self.modules.agent_rollout._stage_tool_result(receipt, self.trust_store)
        if result["gate_passed"]:
            relative = session.s2_contract["selection_relative_path"]
            self._mirror(
                workspace,
                host_path=episode_dir / "workspace" / relative,
                relative_path=relative,
            )
            session.s2_receipt = receipt
            result["next_stage"] = "S3"
            source_binding = self.modules.agent_rollout._verified_source_binding(
                session.s3_template
            )
            if source_binding is not None:
                result["next_stage_source_binding"] = source_binding
        return result

    def _handle_execute_code(
        self,
        workspace: FilesystemSandbox,
        session: _WorkspaceSession,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if set(arguments) != {"stage", "code"}:
            raise ValueError("execution arguments differ from exact source policy")
        stage = arguments["stage"]
        code = arguments["code"]
        if stage not in {"S3", "S4"} or not isinstance(code, str) or not code:
            raise ValueError("execution stage or code differs")
        if stage == "S3" and session.s2_receipt is None:
            return {
                "stage": "S3",
                "gate_passed": False,
                "error": "verified_s2_prerequisite_required",
            }
        if stage == "S4" and session.s3_success is None:
            return {
                "stage": "S4",
                "gate_passed": False,
                "error": "verified_s3_prerequisite_required",
            }
        if stage == "S3":
            session.s3_attempts += 1
            attempt = session.s3_attempts
            prerequisite = session.s2_receipt
            contract = deepcopy(session.s3_template)
        else:
            session.s4_attempts += 1
            attempt = session.s4_attempts
            prerequisite = session.s3_success[0] if session.s3_success else None
            contract = deepcopy(session.s4_template)
        if attempt > 2:
            return {
                "stage": stage,
                "gate_passed": False,
                "error": "execution_budget_exhausted",
            }
        assert prerequisite is not None
        contract["prerequisite_receipt_sha256"] = self.modules.canonical.sha256_value(
            prerequisite
        )
        episode_dir = session.root / "executions" / f"{stage.lower()}-attempt-{attempt}"
        from inspect import signature
        from eva_agent.pipeline.execution_diagnostics import (
            diagnostics_requested, record_execution_diagnostics,
        )
        capture = {}
        capture_options = {}
        if (diagnostics_requested()
                and "diagnostics_sink" in signature(self.modules.execution.execute_python_contract).parameters):
            capture_options["diagnostics_sink"] = capture
        receipt = self.modules.execution.execute_python_contract(
            contract,
            code,
            input_dir=self.evidence_root,
            episode_dir=episode_dir,
            image_refs=self.image_refs,
            key_id=self.key_id,
            private_key=self.private_key,
            trust_store=self.trust_store,
            prerequisite_receipt=prerequisite,
            docker_binary=self.docker_binary,
            **capture_options,
        )
        payload = self.modules.host_receipts.verify_host_receipt(
            receipt, self.trust_store
        )
        result = {
            **self.modules.agent_rollout._tool_result_from_execution(receipt),
            "attempt": attempt,
            "host_receipt_sha256": self.modules.canonical.sha256_file(
                episode_dir / "host-receipt.json"
            ),
        }
        if capture:
            record_execution_diagnostics(capture, episode_dir=episode_dir, stage=stage, attempt=attempt)
        if payload["observations"]["execution"]["gate_passed"] is True:
            if stage == "S3":
                session.s3_success = (receipt, episode_dir)
            else:
                relative = contract["artifact"]["relative_path"]
                self._mirror(
                    workspace,
                    host_path=episode_dir / "workspace" / relative,
                    relative_path=relative,
                )
                session.s4_success = (contract, receipt, episode_dir)
        return result

    def _handle_submit_results(
        self,
        workspace: FilesystemSandbox,
        session: _WorkspaceSession,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if set(arguments) != {"terminal"} or not isinstance(arguments["terminal"], Mapping):
            raise ValueError("submission arguments differ from exact source policy")
        if session.submission_attempted:
            return {"stage": "S5", "gate_passed": False, "error": "single_submission_consumed"}
        session.submission_attempted = True
        if session.s4_success is None:
            return {
                "stage": "S5",
                "gate_passed": False,
                "error": "verified_s4_prerequisite_required",
            }
        contract, receipt, episode_dir = session.s4_success
        relative = contract["artifact"]["relative_path"]
        actor_bytes = workspace.read_bytes(relative)
        host_bytes = (episode_dir / "workspace" / relative).read_bytes()
        if actor_bytes != host_bytes:
            raise ProductionSourceBlocker("actor S4 artifact differs at independent host reopen")
        submission_contract = self.modules.agent_rollout._instantiate_submission_contract(
            session.s5_template,
            contract,
            receipt,
        )
        terminal_bytes = self.modules.canonical.canonical_json_bytes(arguments["terminal"])
        final_receipt = self.modules.execution.finalize_submission(
            submission_contract,
            terminal_bytes,
            execution_contract=contract,
            execution_receipt=receipt,
            episode_dir=episode_dir,
            trust_store=self.trust_store,
            key_id=self.key_id,
            private_key=self.private_key,
        )
        payload = self.modules.host_receipts.verify_host_receipt(
            final_receipt, self.trust_store
        )
        workspace.write_bytes(
            ".eva/final-host-receipt.json",
            self.modules.canonical.canonical_json_bytes(final_receipt),
            create_only=True,
        )
        observations = payload["observations"]
        return {
            "stage": "S5",
            "gate_passed": observations["submission"]["accepted"],
            "terminal_valid": observations["terminal"]["valid"],
            "final_host_receipt_sha256": self.modules.canonical.sha256_value(
                final_receipt
            ),
        }


class LegacyExecutionBindingResolver:
    """Resolve v24-promoted source identities to exact real execution ports."""

    def __init__(
        self,
        *,
        authority_root: str | Path,
        supervisor_root: str | Path,
        trust_store_path: str | Path,
        legacy_python_root: str | Path,
        runtime_state_root: str | Path,
        host_private_key_path: str | Path,
        host_key_id: str,
        image_refs_path: str | Path,
        docker_binary: str = "docker",
        worker_width: int = 64,
    ) -> None:
        self.authority_root = _real_directory(authority_root, label="legacy authority root")
        self.supervisor_root = _real_directory(supervisor_root, label="v24 supervisor root")
        self.runtime_state_root = _real_directory(
            runtime_state_root, label="legacy execution state root", create=True
        )
        raw_trust_path = Path(trust_store_path)
        raw_image_path = Path(image_refs_path)
        if raw_trust_path.is_symlink() or raw_image_path.is_symlink():
            raise ProductionSourceBlocker("legacy control paths cannot be symlinks")
        self.trust_store_path = _safe_file(
            raw_trust_path.resolve(strict=True),
            root=self.authority_root,
            label="legacy host trust store",
        )
        self.image_refs_path = _safe_file(
            raw_image_path.resolve(strict=True),
            root=self.authority_root,
            label="legacy image reference map",
        )
        key_path = Path(host_private_key_path)
        if not key_path.is_absolute() or key_path.is_symlink():
            raise ProductionSourceBlocker("host signing key path must be absolute")
        self.host_private_key_path = _safe_file(
            key_path.resolve(strict=True), root=key_path.parent.resolve(strict=True), label="host signing key"
        )
        if not isinstance(host_key_id, str) or not host_key_id:
            raise ProductionSourceBlocker("host signing key ID is required")
        self.host_key_id = host_key_id
        if not isinstance(docker_binary, str) or not docker_binary or shutil.which(docker_binary) is None:
            raise ProductionSourceBlocker("Docker execution runtime is unavailable")
        self.docker_binary = docker_binary
        self.modules = _load_legacy_modules(Path(legacy_python_root).resolve(strict=True))
        self.readiness = load_signed_supervisor_v24_readiness(
            self.supervisor_root,
            authority_root=self.authority_root,
            trust_store_path=self.trust_store_path,
            worker_width=worker_width,
        )
        trust_document, _ = _read_json(
            self.trust_store_path, root=self.authority_root, label="legacy trust store"
        )
        keys = trust_document.get("keys")
        if not isinstance(keys, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in keys.items()
        ):
            raise ProductionSourceBlocker("legacy trust-store key map differs")
        self.trust_store = MappingProxyType(dict(keys))
        if self.host_key_id not in self.trust_store:
            raise ProductionSourceBlocker("host signing key ID is not trusted")
        self.private_key = self.modules.key_management.load_host_private_key(
            self.host_private_key_path
        )
        public = self.modules.host_receipts.public_key_base64(self.private_key)
        if public != self.trust_store[self.host_key_id]:
            raise ProductionSourceBlocker("host signing private key differs from trust store")
        image_document, _ = _read_json(
            self.image_refs_path, root=self.authority_root, label="legacy image references"
        )
        images = image_document.get("images")
        if image_document.get("schema") != "rlevo.med-research-image-refs.v1" or not isinstance(images, Mapping):
            raise ProductionSourceBlocker("legacy image-reference map differs")
        self.image_refs = MappingProxyType(dict(images))
        references = self._build_reference_index()
        self._references = MappingProxyType({row.source_candidate_id: row for row in references})
        provisional_catalog = LegacyExecutionBindingCatalog(
            authority_blake3=self.readiness.authority_blake3,
            legacy_runtime_source_blake3=self.modules.source_blake3,
            rows=references,
            catalog_blake3="",
        )
        self.catalog = LegacyExecutionBindingCatalog(
            authority_blake3=provisional_catalog.authority_blake3,
            legacy_runtime_source_blake3=provisional_catalog.legacy_runtime_source_blake3,
            rows=provisional_catalog.rows,
            catalog_blake3=blake3_hex(provisional_catalog.core_document()),
        )
        self._cache: dict[tuple[str, str], CandidateExecutionBinding] = {}
        self._cache_lock = RLock()

    @property
    def catalog_blake3(self) -> str:
        return self.catalog.catalog_blake3

    @property
    def catalog_inventory_blake3(self) -> str:
        """Stable v24-promoted binding inventory commitment for recipes."""

        return self.catalog.catalog_blake3

    @property
    def executable_candidate_count(self) -> int:
        """Exact current coverage: v24-promoted candidates only."""

        return self.catalog.candidate_count

    def inventory(self) -> tuple[CandidateBindingReference, ...]:
        return self.catalog.rows

    def _build_reference_index(self) -> tuple[CandidateBindingReference, ...]:
        registry_path = self.supervisor_root / "candidate-registry.v24.json"
        registry, _ = _read_json(
            registry_path, root=self.authority_root, label="v24 execution registry"
        )
        candidates = registry.get("candidates")
        if not isinstance(candidates, list):
            raise ProductionSourceBlocker("v24 execution registry candidates differ")
        readiness = self.readiness.readiness_by_candidate
        rows: list[CandidateBindingReference] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ProductionSourceBlocker("v24 execution registry row differs")
            source_id = candidate.get("candidate_id")
            signed_row = readiness.get(source_id)
            if signed_row is None or signed_row.readiness is not ConstructionReadiness.PROMOTED:
                continue
            proofs = candidate.get("proofs")
            promoted = proofs.get("promoted") if isinstance(proofs, Mapping) else None
            if not isinstance(promoted, Mapping):
                raise ProductionSourceBlocker(f"{source_id} promoted proof is unavailable")
            subject_path, subject_bytes = _proof_ref(
                promoted.get("subject"), root=self.authority_root, label=f"{source_id} promoted subject"
            )
            subject = _decode_json(subject_bytes, label=f"{source_id} promoted subject")
            source_evidence = subject.get("source_evidence")
            if not (
                subject.get("candidate_id") == source_id
                and subject.get("focus") == signed_row.stage
                and subject.get("proof_name") == "promoted"
                and subject.get("derived_state") == "promoted"
                and isinstance(source_evidence, list)
                and _sha256_value(source_evidence) == subject.get("source_evidence_sha256")
            ):
                raise ProductionSourceBlocker(f"{source_id} promoted subject binding differs")
            by_role: dict[str, list[Mapping[str, Any]]] = {}
            for evidence in source_evidence:
                if not isinstance(evidence, Mapping) or not isinstance(evidence.get("role"), str):
                    raise ProductionSourceBlocker(f"{source_id} promoted evidence row differs")
                by_role.setdefault(evidence["role"], []).append(evidence)

            def one(roles: frozenset[str], label: str) -> Mapping[str, Any]:
                matched = [row for role in roles for row in by_role.get(role, [])]
                if len(matched) != 1:
                    raise ProductionSourceBlocker(f"{source_id} exact {label} pointer differs")
                return matched[0]

            policy_row = one(_POLICY_ROLES, "promoted policy")
            manifest_row = one(_MANIFEST_ROLES, "construction manifest")
            attestation_row = one(_ATTESTATION_ROLES, "construction attestation")
            if policy_row.get("signed") is not True or attestation_row.get("signed") is not True:
                raise ProductionSourceBlocker(f"{source_id} signed execution evidence differs")
            policy_path, policy_ref, policy_bytes = _artifact_ref(
                policy_row.get("artifact"), root=self.authority_root, label=f"{source_id} policy"
            )
            manifest_path, manifest_ref, manifest_bytes = _artifact_ref(
                manifest_row.get("artifact"), root=self.authority_root, label=f"{source_id} manifest"
            )
            attestation_path, _attestation_ref, attestation_bytes = _artifact_ref(
                attestation_row.get("artifact"), root=self.authority_root, label=f"{source_id} construction attestation"
            )
            policy = _decode_json(policy_bytes, label=f"{source_id} policy")
            manifest = _decode_json(manifest_bytes, label=f"{source_id} manifest")
            if not (
                policy.get("schema") == POLICY_SCHEMA
                and policy.get("sandbox_id") == source_id
                and policy.get("focus") == signed_row.stage
                and manifest.get("schema") == CONSTRUCTION_SCHEMA
                and manifest.get("sandbox_id") == source_id
                and manifest.get("focus") == signed_row.stage
                and manifest.get("construction_complete") is True
                and manifest.get("production_construction_eligible") is True
                and manifest.get("conformance_only") is False
            ):
                raise ProductionSourceBlocker(f"{source_id} policy/construction identity differs")
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
                raise ProductionSourceBlocker(f"{source_id} exact policy handler set differs")
            attestation = _decode_json(
                attestation_bytes, label=f"{source_id} construction attestation"
            )
            try:
                attested = self.modules.host_receipts.verify_host_receipt(
                    attestation, self.trust_store
                )
            except Exception as exc:
                raise ProductionSourceBlocker(
                    f"{source_id} construction attestation signature differs"
                ) from exc
            observations = attested.get("observations")
            if not isinstance(observations, Mapping) or not (
                attested.get("sandbox_id") == source_id
                and observations.get("attestation_kind") == "candidate_construction"
                and observations.get("manifest_file_sha256") == manifest_ref["sha256"]
                and observations.get("manifest_content_sha256") == _sha256_value(manifest)
                and observations.get("construction_complete") is True
                and observations.get("production_construction_eligible") is True
            ):
                raise ProductionSourceBlocker(f"{source_id} signed construction facts differ")
            rows.append(
                CandidateBindingReference(
                    source_candidate_id=source_id,
                    source_family=signed_row.source_family,
                    stage=Stage(signed_row.stage),
                    promoted_subject_path=subject_path,
                    promoted_subject_blake3=blake3_bytes(subject_bytes),
                    source_policy_path=policy_path,
                    source_policy_upstream_sha256=policy_ref["sha256"],
                    source_policy_blake3=blake3_bytes(policy_bytes),
                    construction_manifest_path=manifest_path,
                    construction_manifest_upstream_sha256=manifest_ref["sha256"],
                    construction_manifest_blake3=blake3_bytes(manifest_bytes),
                    construction_attestation_path=attestation_path,
                    construction_attestation_blake3=blake3_bytes(attestation_bytes),
                    proof_root_blake3=signed_row.proof_root_blake3,
                )
            )
        rows.sort(key=lambda row: row.source_candidate_id)
        if len(rows) != 1_344:
            raise ProductionSourceBlocker("v24 promoted execution binding count differs")
        return tuple(rows)

    def resolve(
        self,
        eva_candidate_id: str,
        *,
        source_candidate_id: str,
    ) -> CandidateExecutionBinding:
        candidate_id = _uuid(eva_candidate_id, label="EVA candidate_id")
        if not isinstance(source_candidate_id, str) or _SOURCE_ID.fullmatch(source_candidate_id) is None:
            raise ProductionSourceBlocker("legacy source_candidate_id differs")
        cache_key = (candidate_id, source_candidate_id)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        try:
            reference = self._references[source_candidate_id]
        except KeyError:
            raise ProductionSourceBlocker(
                f"legacy source candidate is not v24-promoted: {source_candidate_id}"
            ) from None
        binding = self._materialize_binding(candidate_id, reference)
        with self._cache_lock:
            prior = self._cache.setdefault(cache_key, binding)
        return prior

    def _materialize_binding(
        self,
        candidate_id: str,
        reference: CandidateBindingReference,
    ) -> CandidateExecutionBinding:
        policy, policy_bytes = _read_json(
            reference.source_policy_path,
            root=self.authority_root,
            label=f"{reference.source_candidate_id} exact promoted policy",
        )
        manifest, manifest_bytes = _read_json(
            reference.construction_manifest_path,
            root=self.authority_root,
            label=f"{reference.source_candidate_id} exact construction manifest",
        )
        subject_bytes = _read_bytes(
            reference.promoted_subject_path,
            root=self.authority_root,
            label=f"{reference.source_candidate_id} promoted subject reopen",
        )
        attestation_bytes = _read_bytes(
            reference.construction_attestation_path,
            root=self.authority_root,
            label=f"{reference.source_candidate_id} construction attestation reopen",
        )
        if (
            blake3_bytes(policy_bytes) != reference.source_policy_blake3
            or _sha256_bytes(policy_bytes) != reference.source_policy_upstream_sha256
            or blake3_bytes(manifest_bytes) != reference.construction_manifest_blake3
            or _sha256_bytes(manifest_bytes)
            != reference.construction_manifest_upstream_sha256
            or blake3_bytes(subject_bytes) != reference.promoted_subject_blake3
            or blake3_bytes(attestation_bytes)
            != reference.construction_attestation_blake3
        ):
            raise ProductionSourceBlocker(
                "signed execution proof, policy, or construction changed after indexing"
            )
        construction_root = reference.construction_manifest_path.parent
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping) or not _MANDATORY_ARTIFACTS <= set(artifacts):
            raise ProductionSourceBlocker("construction executable artifact graph is incomplete")
        loaded: dict[str, Mapping[str, Any]] = {}
        loaded_bytes: dict[str, bytes] = {}
        loaded_paths: dict[str, Path] = {}
        for name in sorted(_MANDATORY_ARTIFACTS):
            path, _ref, payload = _artifact_ref(
                artifacts[name], root=construction_root, label=f"construction {name}"
            )
            loaded_paths[name] = path
            loaded_bytes[name] = payload
            loaded[name] = _decode_json(payload, label=f"construction {name}")
        candidate_policy = deepcopy(dict(policy))
        split = candidate_policy.get("target_split")
        if split not in {"train", "development", "sealed_evaluation"}:
            raise ProductionSourceBlocker("promoted policy target split differs")
        candidate_policy["target_split"] = "candidate"
        if loaded["policy"] != candidate_policy:
            raise ProductionSourceBlocker(
                "promoted and constructed policies differ beyond target_split"
            )
        templates = self.modules.panel_dispatch._panelize_constructor_templates(
            base_policy=policy,
            templates={
                "s1": loaded["s1_plan_contract"],
                "s2": loaded["s2_evidence_contract"],
                "s3": loaded["s3_execution_contract"],
                "s4": loaded["s4_execution_contract"],
                "s5": loaded["s5_submission_contract"],
            },
        )
        templates = _project_automedbench_historical_episode_alias(
            base_policy=policy,
            templates=templates,
            sha256_value=self.modules.canonical.sha256_value,
        )
        try:
            self.modules.panel_dispatch._validate_contract_templates(
                base_policy=policy,
                private_evaluation=loaded["private_evaluation"],
                s1_template=templates["s1"],
                s2_template=templates["s2"],
                s3_template=templates["s3"],
                s4_template=templates["s4"],
                s5_template=templates["s5"],
            )
        except Exception as exc:
            raise ProductionSourceBlocker(
                "candidate policy and real stage contracts do not execute together"
            ) from exc
        input_root = _real_directory(
            construction_root / "construction-private" / "04-solver-input",
            label="signed construction solver input tree",
        )
        try:
            self.modules.execution._verify_input_inventory(templates["s3"], input_root)
            self.modules.execution._verify_input_inventory(templates["s4"], input_root)
        except Exception as exc:
            raise ProductionSourceBlocker(
                "candidate signed input tree differs from execution contracts"
            ) from exc
        image_digest = templates["s3"].get("image_digest")
        if image_digest not in self.image_refs or templates["s4"].get("image_digest") != image_digest:
            raise ProductionSourceBlocker("candidate execution image is not allowlisted")

        policy_tools = policy.get("tools")
        assert isinstance(policy_tools, list)
        catalog = tuple(
            CandidateToolCatalogEntry(
                name=row["name"],
                description=row["description"],
                input_schema=row["input_schema"],
            )
            for row in sorted(policy_tools, key=lambda item: item["name"])
        )
        tool_catalog_blake3 = blake3_hex([row.to_document() for row in catalog])
        public_context = freeze_json(
            {
                "schema": "eva.legacy-candidate-runtime-context.v1",
                "source_candidate_id": reference.source_candidate_id,
                "source_family": reference.source_family,
                "focus": reference.stage.value,
                "target_split": split,
                "active_episode_id": templates["s1"]["episode_id"],
                "policy_blake3": reference.source_policy_blake3,
                "tool_catalog_blake3": tool_catalog_blake3,
                "s1_plan_contract": templates["s1"],
                "s2_evidence_contract": templates["s2"],
                "execution_stages": {
                    stage: {
                        "stage": template["stage"],
                        "artifact_relative_path": template["artifact"]["relative_path"],
                        "artifact_json_schema": template["artifact"]["json_schema"],
                        "limits": template["limits"],
                    }
                    for stage, template in (
                        ("S3", templates["s3"]),
                        ("S4", templates["s4"]),
                    )
                },
                "s5_terminal_json_schema": templates["s5"]["terminal"]["json_schema"],
            }
        )
        assert isinstance(public_context, Mapping)
        initial_files = MappingProxyType(
            {
                ".eva/runtime-context.json": canonical_json_bytes(_thaw(public_context)),
                ".eva/source-policy.json": policy_bytes,
            }
        )
        backend_identity = {
            "candidate_id": candidate_id,
            "source_candidate_id": reference.source_candidate_id,
            "source_policy_blake3": reference.source_policy_blake3,
            "construction_manifest_blake3": reference.construction_manifest_blake3,
            "tool_catalog_blake3": tool_catalog_blake3,
        }
        backend = _LegacyCandidateToolBackend(
            modules=self.modules,
            binding_identity=backend_identity,
            runtime_state_root=self.runtime_state_root,
            contracts=templates,
            evidence_root=input_root,
            image_refs=self.image_refs,
            trust_store=self.trust_store,
            key_id=self.host_key_id,
            private_key=self.private_key,
            docker_binary=self.docker_binary,
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
                for row in catalog
            ]
        )
        public_by_name = {
            row["function"]["name"]: row["function"]
            for row in registry.public_schemas()
        }
        for row in catalog:
            observed = public_by_name.get(row.name)
            if not isinstance(observed, Mapping) or not (
                observed.get("description") == row.description
                and _thaw(observed.get("parameters")) == _thaw(row.input_schema)
                and observed.get("x-eva-kind") == "tool"
                and observed.get("x-eva-parallel-safe") is False
            ):
                raise ProductionSourceBlocker(
                    f"candidate registry schema/visibility differs for {row.name}"
                )
        provisional = CandidateExecutionBinding(
            candidate_id=candidate_id,
            source_candidate_id=reference.source_candidate_id,
            source_family=reference.source_family,
            stage=reference.stage,
            source_policy_path=reference.source_policy_path,
            source_policy_upstream_sha256=reference.source_policy_upstream_sha256,
            source_policy_blake3=reference.source_policy_blake3,
            construction_root=construction_root,
            construction_manifest_path=reference.construction_manifest_path,
            construction_manifest_blake3=reference.construction_manifest_blake3,
            legacy_runtime_source_blake3=self.modules.source_blake3,
            resolver_catalog_blake3=self.catalog.catalog_blake3,
            tool_catalog=catalog,
            tool_catalog_blake3=tool_catalog_blake3,
            public_runtime_context=public_context,
            initial_workspace_files=initial_files,
            tool_registry=registry,
            binding_blake3="",
        )
        return CandidateExecutionBinding(
            **{
                **{
                    field: getattr(provisional, field)
                    for field in provisional.__dataclass_fields__
                    if field != "binding_blake3"
                },
                "binding_blake3": blake3_hex(provisional.core_document()),
            }
        )


__all__ = [
    "CandidateBindingReference",
    "CandidateExecutionBinding",
    "CandidateToolCatalogEntry",
    "LegacyExecutionBindingCatalog",
    "LegacyExecutionBindingResolver",
    "ProductionSourceBlocker",
]
