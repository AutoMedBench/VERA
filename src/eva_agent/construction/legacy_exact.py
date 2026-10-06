"""Exact legacy construction authority adapter and append-only EVA sink.

The sibling ``rlevo-med-research`` implementation remains the sole authority
for construction schemas, phase requests, wire compilation, safety checks,
evidence checks, and executable artifact compilation.  This module imports
that implementation from an explicitly pinned source tree and calls it; no
legacy schema or validator is copied into EVA-Agent.

The adapter is intentionally additive.  It produces a new EVA executable
binding and never emits, edits, or claims compatibility with a historical
candidate-construction manifest.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from threading import Lock, RLock
from types import MappingProxyType, ModuleType
from typing import Any, Callable, Mapping, Sequence

from eva_agent.campaign.selection_v2 import CampaignSelectionV2, SelectionEntryV2
from eva_agent.codex_runtime import (
    CodexRole,
    CodexSandbox,
    CodexThreadOptions,
    CodexTurnInput,
)
from eva_agent.pipeline.contracts import JsonValue, RuntimeIdFactory, freeze_json
from eva_agent.pipeline.digests import (
    blake3_bytes,
    blake3_hex,
    canonical_json_bytes,
    canonical_value,
    is_blake3,
)
from eva_agent.pipeline.ids import RandomUUIDFactory
from eva_agent.sources.supervisor_v24_readiness import SignedSupervisorV24Readiness

from .premium_codex import (
    COMPARISON_QUORUM_LANES,
    CRITIC_QUORUM_LANES,
    CRITIC_QUORUM_POLICY_V1,
    CRITIC_AUTHORITY_POLICY_V2,
    CRITIQUE_QUORUM_LANES,
    ConstructionFailureReceipt,
    ConstructionLane,
    LANE_ORDER,
    ConstructionPublicationReceipt,
    ConstructionTurnRequest,
    FrozenConstructionSource,
    PremiumCodexConstructionOrchestrator,
    PremiumConstructionError,
    PremiumConstructionResult,
    PremiumConstructionRoutes,
    PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
    PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4,
    PublishedConstructionResult,
    V3_LANE_ORDER,
    verify_premium_construction_result,
    extract_unique_schema_valid_payload,
)


_REGISTRY_NAME = "candidate-registry.v24.json"
_MAX_REGISTRY_BYTES = 40 * 1024 * 1024
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_CONTROL_BYTES = 8 * 1024 * 1024
_SAFE_COMPONENT = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
)
_IMPORT_LOCK = Lock()


class LegacyConstructionAdapterError(PremiumConstructionError):
    """The exact legacy adapter or append-only publication failed closed."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LegacyConstructionAdapterError("JSON object contains a duplicate key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise LegacyConstructionAdapterError("JSON contains a non-finite number")


def _decode_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except LegacyConstructionAdapterError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise LegacyConstructionAdapterError(f"{label} is not strict UTF-8 JSON") from None
    if not isinstance(value, dict):
        raise LegacyConstructionAdapterError(f"{label} must be a JSON object")
    return value


def _absolute(path: str | Path, *, label: str) -> Path:
    text = os.fspath(path)
    if not text or "\x00" in text:
        raise LegacyConstructionAdapterError(f"{label} path differs")
    value = Path(os.path.abspath(text))
    if not value.is_absolute() or ".." in value.parts:
        raise LegacyConstructionAdapterError(f"{label} path is not normalized")
    return value


def _open_directory_chain(path: Path, *, create: bool, mode: int = 0o700) -> int:
    """Open an absolute directory through no-follow descriptor-relative steps."""

    if not path.is_absolute():
        raise LegacyConstructionAdapterError("directory root must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            if not component or any(character not in _SAFE_COMPONENT for character in component):
                raise LegacyConstructionAdapterError("directory component is unsafe")
            if create:
                try:
                    os.mkdir(component, mode=mode, dir_fd=descriptor)
                except FileExistsError:
                    pass
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise LegacyConstructionAdapterError(
                    "directory path cannot be opened without following links"
                ) from exc
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise LegacyConstructionAdapterError("directory component is not a directory")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _existing_directory(path: str | Path, *, label: str) -> Path:
    value = _absolute(path, label=label)
    descriptor = _open_directory_chain(value, create=False)
    os.close(descriptor)
    return value


def _ensure_directory(path: str | Path, *, label: str) -> Path:
    value = _absolute(path, label=label)
    descriptor = _open_directory_chain(value, create=True)
    os.close(descriptor)
    return value


def _safe_relative(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise LegacyConstructionAdapterError(f"{label} is not a POSIX relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
        or any(any(character not in _SAFE_COMPONENT for character in part) for part in relative.parts)
    ):
        raise LegacyConstructionAdapterError(f"{label} escapes its authority root")
    return relative


def _open_parent(root: Path, relative: PurePosixPath, *, create: bool) -> tuple[int, str]:
    descriptor = _open_directory_chain(root, create=False)
    try:
        for component in relative.parts[:-1]:
            if create:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            child = os.open(component, flags, dir_fd=descriptor)
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise LegacyConstructionAdapterError("artifact parent is not a directory")
            os.close(descriptor)
            descriptor = child
        return descriptor, relative.parts[-1]
    except Exception:
        os.close(descriptor)
        raise


def _read_relative(
    root: Path,
    relative: PurePosixPath,
    *,
    label: str,
    maximum_bytes: int,
) -> bytes:
    parent, name = _open_parent(root, relative, create=False)
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=parent)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or not 0 < metadata.st_size <= maximum_bytes
            ):
                raise LegacyConstructionAdapterError(f"{label} file boundary differs")
            chunks: list[bytes] = []
            remaining = maximum_bytes + 1
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) != metadata.st_size or len(payload) > maximum_bytes:
                raise LegacyConstructionAdapterError(f"{label} bounded read differs")
            return payload
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise LegacyConstructionAdapterError(
            f"{label} cannot be opened without following links"
        ) from exc
    finally:
        os.close(parent)


def _write_new_relative(
    root: Path,
    relative: str,
    payload: bytes,
    *,
    mode: int = 0o400,
) -> Mapping[str, JsonValue]:
    pure = _safe_relative(relative, label="publication artifact path")
    parent, name = _open_parent(root, pure, create=True)
    descriptor = -1
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=parent)
        except FileExistsError:
            raise LegacyConstructionAdapterError("append-only artifact already exists") from None
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise LegacyConstructionAdapterError("append-only artifact is not a single regular file")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise LegacyConstructionAdapterError("append-only artifact write stalled")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    except OSError as exc:
        raise LegacyConstructionAdapterError("append-only artifact write failed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)
    return MappingProxyType(
        {
            "path": pure.as_posix(),
            "byte_count": len(payload),
            "blake3": blake3_bytes(payload),
            "mode": format(mode, "04o"),
        }
    )


def _mkdir_new_child(root: Path, name: str) -> Path:
    relative = _safe_relative(name, label="append-only directory")
    if len(relative.parts) != 1:
        raise LegacyConstructionAdapterError("append-only directory must be one component")
    descriptor = _open_directory_chain(root, create=False)
    try:
        try:
            os.mkdir(relative.name, mode=0o700, dir_fd=descriptor)
        except FileExistsError:
            raise LegacyConstructionAdapterError("append-only directory already exists") from None
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        child = os.open(relative.name, flags, dir_fd=descriptor)
        os.close(child)
    finally:
        os.close(descriptor)
    return root / relative.name


def _seal_tree(root: Path) -> None:
    for current, directories, files in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *files):
            candidate = current_path / name
            if candidate.is_symlink():
                raise LegacyConstructionAdapterError("publication tree contains a symlink")
        os.chmod(current_path, 0o500, follow_symlinks=False)


@dataclass(frozen=True)
class LegacyValidatorAuthorityIdentity:
    schema: str
    package_source_blake3: str
    candidate_construction_module_blake3: str
    premium_exact_module_blake3: str
    construction_input_schema: str
    validator_names: tuple[str, ...]
    authority_blake3: str

    def core(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "package_source_blake3": self.package_source_blake3,
            "candidate_construction_module_blake3": self.candidate_construction_module_blake3,
            "premium_exact_module_blake3": self.premium_exact_module_blake3,
            "construction_input_schema": self.construction_input_schema,
            "validator_names": self.validator_names,
        }

    def __post_init__(self) -> None:
        if self.schema != "eva.legacy-construction-validator-authority.v1":
            raise LegacyConstructionAdapterError("legacy validator authority schema differs")
        if not all(
            is_blake3(value)
            for value in (
                self.package_source_blake3,
                self.candidate_construction_module_blake3,
                self.premium_exact_module_blake3,
                self.authority_blake3,
            )
        ):
            raise LegacyConstructionAdapterError("legacy validator authority digest differs")
        if self.authority_blake3 != blake3_hex(self.core()):
            raise LegacyConstructionAdapterError("legacy validator authority commitment differs")

    def to_document(self) -> Mapping[str, Any]:
        return {**self.core(), "authority_blake3": self.authority_blake3}


class _LegacyAuthority:
    _REQUIRED_CANDIDATE = (
        "validate_construction_input",
        "validate_host_trust_store",
        "_load_source_verification",
        "_verify_source_attestation",
        "_load_evidence",
        "_request_body",
        "_compile_provider_wire_output",
        "_validate_blueprint",
        "_validate_critique",
        "_validate_revision",
        "_scan_model_output",
        "_compile_candidate",
        "validate_policy",
        "validate_private_evaluation",
        "validate_evidence_manifest",
        "validate_reference_answer",
        "validate_stage_plan_contract",
        "validate_evidence_service_contract",
        "validate_execution_contract",
        "validate_submission_contract",
        "sha256_value",
    )
    _REQUIRED_PREMIUM = (
        "gemini_alternate_request",
        "sol_comparison_request",
        "validate_sol_comparison",
        "premium_context",
        "alternate_author_context",
        "_strict_object_bytes",
    )

    def __init__(self, package_src: str | Path) -> None:
        self.package_src = _existing_directory(package_src, label="legacy package source")
        package_dir = self.package_src / "rlevo_med_research"
        _existing_directory(package_dir, label="legacy Python package")
        candidate_relative = _safe_relative(
            "rlevo_med_research/candidate_construction.py",
            label="candidate-construction module",
        )
        premium_relative = _safe_relative(
            "rlevo_med_research/premium_exact_construction.py",
            label="premium-exact module",
        )
        candidate_bytes = _read_relative(
            self.package_src,
            candidate_relative,
            label="candidate-construction module",
            maximum_bytes=4 * 1024 * 1024,
        )
        premium_bytes = _read_relative(
            self.package_src,
            premium_relative,
            label="premium-exact module",
            maximum_bytes=2 * 1024 * 1024,
        )
        with _IMPORT_LOCK:
            inserted = False
            if os.fspath(self.package_src) not in sys.path:
                sys.path.insert(0, os.fspath(self.package_src))
                inserted = True
            try:
                candidate = importlib.import_module(
                    "rlevo_med_research.candidate_construction"
                )
                premium = importlib.import_module(
                    "rlevo_med_research.premium_exact_construction"
                )
            finally:
                if inserted:
                    sys.path.remove(os.fspath(self.package_src))
        self._assert_module_path(candidate, candidate_relative)
        self._assert_module_path(premium, premium_relative)
        for module, names in (
            (candidate, self._REQUIRED_CANDIDATE),
            (premium, self._REQUIRED_PREMIUM),
        ):
            if any(not callable(getattr(module, name, None)) for name in names):
                raise LegacyConstructionAdapterError(
                    "legacy construction authority API is incomplete"
                )
        self.candidate = candidate
        self.premium = premium
        core = {
            "schema": "eva.legacy-construction-validator-authority.v1",
            "package_source_blake3": blake3_hex(
                {
                    "candidate_construction": blake3_bytes(candidate_bytes),
                    "premium_exact_construction": blake3_bytes(premium_bytes),
                }
            ),
            "candidate_construction_module_blake3": blake3_bytes(candidate_bytes),
            "premium_exact_module_blake3": blake3_bytes(premium_bytes),
            "construction_input_schema": candidate.CONSTRUCTION_INPUT_SCHEMA,
            "validator_names": (*self._REQUIRED_CANDIDATE, *self._REQUIRED_PREMIUM),
        }
        self.identity = LegacyValidatorAuthorityIdentity(
            **core, authority_blake3=blake3_hex(core)
        )

    def _assert_module_path(
        self, module: ModuleType, relative: PurePosixPath
    ) -> None:
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            raise LegacyConstructionAdapterError("legacy authority module has no file")
        expected = self.package_src.joinpath(*relative.parts)
        if Path(module_path).resolve(strict=True) != expected.resolve(strict=True):
            raise LegacyConstructionAdapterError(
                "a different legacy construction package is already loaded"
            )


@dataclass(frozen=True)
class _LegacySourceState:
    source: FrozenConstructionSource
    entry: SelectionEntryV2
    authority: _LegacyAuthority
    construction_input: Mapping[str, Any]
    construction_input_bytes: bytes
    evidence_documents: tuple[Mapping[str, Any], ...]
    evidence_payloads: Mapping[str, bytes]
    claim_support: Mapping[str, frozenset[str]]
    source_verification: Mapping[str, Any]
    source_verification_bytes: bytes
    source_attestation: Mapping[str, Any]
    source_attestation_bytes: bytes
    workspace_roots: Mapping[ConstructionLane, Path]


@dataclass(frozen=True)
class _ValidationPlan:
    lane: ConstructionLane
    phase_request: Mapping[str, Any]
    primary_draft: Mapping[str, Any] | None = None
    alternate_draft: Mapping[str, Any] | None = None
    critique: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LegacyReopenedConstruction:
    """Versioned outputs reopened by the sibling authority and compiled for execution."""

    phase_requests: Mapping[str, JsonValue]
    compiled_phase_outputs: Mapping[str, JsonValue]
    executable_artifacts: Mapping[str, JsonValue]
    validator_authority_blake3: str
    executable_material_blake3: str


_ROLE_BY_LANE = MappingProxyType(
    {
        ConstructionLane.OPUS5_DRAFT: CodexRole.STRONG_ACTOR,
        ConstructionLane.GEMINI_ALTERNATE: CodexRole.STRONG_ACTOR,
        ConstructionLane.OPUS48_CRITIQUE: CodexRole.MIDDLE_ACTOR,
        ConstructionLane.OPUS5_CRITIQUE_BACKUP: CodexRole.STRONG_ACTOR,
        ConstructionLane.GPT56_COMPARISON: CodexRole.STRONG_ACTOR,
        ConstructionLane.GEMINI_COMPARISON_BACKUP: CodexRole.STRONG_ACTOR,
        ConstructionLane.OPUS5_REVISION: CodexRole.STRONG_ACTOR,
    }
)
_SANDBOX_BY_LANE = MappingProxyType(
    {
        ConstructionLane.OPUS5_DRAFT: CodexSandbox.READ_ONLY,
        ConstructionLane.GEMINI_ALTERNATE: CodexSandbox.READ_ONLY,
        ConstructionLane.OPUS48_CRITIQUE: CodexSandbox.READ_ONLY,
        ConstructionLane.OPUS5_CRITIQUE_BACKUP: CodexSandbox.READ_ONLY,
        ConstructionLane.GPT56_COMPARISON: CodexSandbox.READ_ONLY,
        ConstructionLane.GEMINI_COMPARISON_BACKUP: CodexSandbox.READ_ONLY,
        ConstructionLane.OPUS5_REVISION: CodexSandbox.READ_ONLY,
    }
)


class LegacyExactConstructionAdapter:
    """Project exact sibling phase requests through the structural Codex port."""

    def __init__(self, state: _LegacySourceState) -> None:
        self._state = state
        self._plans: dict[str, _ValidationPlan] = {}
        self._plan_lock = RLock()

    @property
    def validator_authority(self) -> LegacyValidatorAuthorityIdentity:
        return self._state.authority.identity

    @property
    def source(self) -> FrozenConstructionSource:
        return self._state.source

    def _require_source(self, source: FrozenConstructionSource) -> None:
        if (
            source.source_id != self._state.source.source_id
            or source.request_blake3 != self._state.source.request_blake3
            or canonical_value(source.request)
            != canonical_value(self._state.source.request)
        ):
            raise LegacyConstructionAdapterError("legacy adapter source binding differs")

    def _require_routes(self, routes: PremiumConstructionRoutes) -> None:
        candidate = self._state.authority.candidate
        premium = self._state.authority.premium
        expected = {
            "opus5": candidate.OPUS_5_MODEL_ID,
            "opus48": candidate._CONSTRUCTION_MODEL_IDENTITIES["critic_opus_4_8"],
            "gpt56": premium.SOL_MODEL_ID,
        }
        for field, model in expected.items():
            if getattr(routes, field).model != model:
                raise LegacyConstructionAdapterError(
                    f"legacy exact {field} model identity differs"
                )

    @staticmethod
    def _output_schema(request: Mapping[str, Any]) -> Mapping[str, Any]:
        tools = request.get("tools")
        if not isinstance(tools, list) or len(tools) != 1:
            raise LegacyConstructionAdapterError("legacy phase request tool boundary differs")
        try:
            schema = tools[0]["function"]["parameters"]
        except (KeyError, TypeError):
            raise LegacyConstructionAdapterError(
                "legacy phase request output schema is unavailable"
            ) from None
        if not isinstance(schema, Mapping):
            raise LegacyConstructionAdapterError("legacy phase output schema differs")
        return schema

    @staticmethod
    def _model_only_projection(
        request: Mapping[str, Any], *, model_id: str
    ) -> dict[str, Any]:
        """Clone an exact phase request while changing only public model identity.

        The sibling GPT comparison builder intentionally accepts only its
        canonical GPT-5.6 identity.  Supplemental v3 attempts therefore start
        from that fully validated request and project only ``/model``.  This
        keeps system/user prompt bytes, tool declaration, tool choice, output
        schema, sampling controls, and seed byte-for-byte identical.
        """

        if not isinstance(model_id, str) or not model_id:
            raise LegacyConstructionAdapterError("supplemental model identity differs")
        primary = deepcopy(dict(request))
        if not isinstance(primary.get("model"), str) or primary["model"] == model_id:
            raise LegacyConstructionAdapterError(
                "supplemental request requires a distinct public model identity"
            )
        projected = deepcopy(primary)
        projected["model"] = model_id
        primary_without_model = deepcopy(primary)
        projected_without_model = deepcopy(projected)
        del primary_without_model["model"]
        del projected_without_model["model"]
        if canonical_json_bytes(primary_without_model) != canonical_json_bytes(
            projected_without_model
        ):
            raise LegacyConstructionAdapterError(
                "supplemental request changes bytes outside public model identity"
            )
        if blake3_hex(LegacyExactConstructionAdapter._output_schema(primary)) != blake3_hex(
            LegacyExactConstructionAdapter._output_schema(projected)
        ):
            raise LegacyConstructionAdapterError(
                "supplemental request output schema projection differs"
            )
        return projected

    def _turn_request(
        self,
        source: FrozenConstructionSource,
        routes: PremiumConstructionRoutes,
        lane: ConstructionLane,
        phase_request: Mapping[str, Any],
        dependencies: Mapping[str, str],
        plan: _ValidationPlan,
        *,
        route_override: ConstructionModelRoute | None = None,
    ) -> ConstructionTurnRequest:
        messages = phase_request.get("messages")
        if not (
            isinstance(messages, list)
            and len(messages) == 2
            and messages[0].get("role") == "system"
            and messages[1].get("role") == "user"
            and isinstance(messages[0].get("content"), str)
            and isinstance(messages[1].get("content"), str)
        ):
            raise LegacyConstructionAdapterError("legacy phase message envelope differs")
        route = route_override or routes.for_lane(lane)
        sandbox = _SANDBOX_BY_LANE[lane]
        options = CodexThreadOptions(
            role=_ROLE_BY_LANE[lane],
            model=route.model,
            provider=route.provider,
            cwd=os.fspath(self._state.workspace_roots[lane]),
            sandbox=sandbox,
            base_instructions=messages[0]["content"],
            ephemeral=True,
            service_name="evamed-codex-premium-construction",
        )
        output_schema = self._output_schema(phase_request)
        turn_input = CodexTurnInput(
            public_text=messages[1]["content"],
            output_schema=output_schema,
            sandbox=sandbox,
            model=route.model,
        )
        request = ConstructionTurnRequest.create(
            lane=lane,
            options=options,
            turn_input=turn_input,
            source_request_blake3=source.request_blake3,
            source_phase_request=phase_request,
            source_output_schema=output_schema,
            dependencies=dependencies,
        )
        with self._plan_lock:
            prior = self._plans.get(request.request_blake3)
            if prior is not None and canonical_value(prior) != canonical_value(plan):
                raise LegacyConstructionAdapterError("legacy validation plan collision")
            self._plans[request.request_blake3] = plan
        return request

    def _input(self) -> dict[str, Any]:
        return deepcopy(dict(self._state.construction_input))

    def _evidence(self) -> list[dict[str, Any]]:
        return [deepcopy(dict(value)) for value in self._state.evidence_documents]

    def _compile_draft(self, output: Mapping[str, Any], *, wrapped: bool) -> dict[str, Any]:
        candidate = self._state.authority.candidate
        premium = self._state.authority.premium
        wire: Mapping[str, Any]
        if wrapped:
            try:
                payload = output["payload"][premium.GEMINI_OUTPUT_KEY]
            except (KeyError, TypeError):
                raise LegacyConstructionAdapterError("Gemini wrapped output differs") from None
            if not isinstance(payload, str):
                raise LegacyConstructionAdapterError("Gemini wrapped payload differs")
            wire = premium._strict_object_bytes(
                payload.encode("utf-8"), label="Gemini alternate candidate"
            )
        else:
            wire = canonical_value(output)
        candidate._scan_model_output(wire)
        compiled = candidate._compile_provider_wire_output(
            stage="draft", wire=wire, construction_input=self._input()
        )
        candidate._scan_model_output(compiled)
        candidate._validate_blueprint(
            compiled,
            construction_input=self._input(),
            claim_support={key: set(value) for key, value in self._state.claim_support.items()},
        )
        return compiled

    def _compile_critique(
        self, output: Mapping[str, Any], *, primary: Mapping[str, Any]
    ) -> dict[str, Any]:
        candidate = self._state.authority.candidate
        wire = canonical_value(output)
        candidate._scan_model_output(wire)
        compiled = candidate._compile_provider_wire_output(
            stage="critique", wire=wire, construction_input=self._input()
        )
        candidate._scan_model_output(compiled)
        candidate._validate_critique(
            compiled,
            draft=primary,
            construction_input=self._input(),
            evidence_ids={item["evidence_id"] for item in self._evidence()},
        )
        return compiled

    def _compile_comparison(
        self,
        output: Mapping[str, Any],
        *,
        primary: Mapping[str, Any],
        alternate: Mapping[str, Any],
    ) -> dict[str, Any]:
        candidate = self._state.authority.candidate
        premium = self._state.authority.premium
        try:
            payload = output["payload"][premium.SOL_OUTPUT_KEY]
        except (KeyError, TypeError):
            raise LegacyConstructionAdapterError("GPT-5.6 wrapped output differs") from None
        if not isinstance(payload, str):
            raise LegacyConstructionAdapterError("GPT-5.6 wrapped payload differs")
        value = premium._strict_object_bytes(
            payload.encode("utf-8"), label="GPT-5.6 comparison"
        )
        candidate._scan_model_output(value)
        return premium.validate_sol_comparison(
            value,
            construction_input=self._input(),
            primary_draft=primary,
            alternate_draft=alternate,
        )

    def _compile_revision(
        self,
        output: Mapping[str, Any],
        *,
        primary: Mapping[str, Any],
        critique: Mapping[str, Any],
    ) -> dict[str, Any]:
        candidate = self._state.authority.candidate
        wire = canonical_value(output)
        candidate._scan_model_output(wire)
        compiled = candidate._compile_provider_wire_output(
            stage="revision", wire=wire, construction_input=self._input()
        )
        candidate._scan_model_output(compiled)
        candidate._validate_revision(
            compiled,
            draft=primary,
            critique=critique,
            construction_input=self._input(),
            claim_support={key: set(value) for key, value in self._state.claim_support.items()},
        )
        return compiled

    def prepare_authors(
        self, source: FrozenConstructionSource, routes: PremiumConstructionRoutes
    ) -> Sequence[ConstructionTurnRequest]:
        self._require_source(source)
        self._require_routes(routes)
        candidate = self._state.authority.candidate
        premium = self._state.authority.premium
        primary = candidate._request_body(
            stage="draft",
            model_id=routes.opus5.model,
            construction_input=self._input(),
            evidence_documents=self._evidence(),
        )
        alternate = premium.gemini_alternate_request(
            canonical_draft_request=primary,
            model_id=routes.gemini31.model,
            seed=self._input()["runtime"]["seed"],
        )
        return (
            self._turn_request(
                source,
                routes,
                ConstructionLane.OPUS5_DRAFT,
                primary,
                {},
                _ValidationPlan(ConstructionLane.OPUS5_DRAFT, primary),
            ),
            self._turn_request(
                source,
                routes,
                ConstructionLane.GEMINI_ALTERNATE,
                alternate,
                {},
                _ValidationPlan(ConstructionLane.GEMINI_ALTERNATE, alternate),
            ),
        )

    def prepare_critics(
        self,
        source: FrozenConstructionSource,
        routes: PremiumConstructionRoutes,
        *,
        primary_draft: Mapping[str, JsonValue],
        alternate_draft: Mapping[str, JsonValue],
    ) -> Sequence[ConstructionTurnRequest]:
        self._require_source(source)
        self._require_routes(routes)
        candidate = self._state.authority.candidate
        premium = self._state.authority.premium
        primary = self._compile_draft(primary_draft, wrapped=False)
        alternate = self._compile_draft(alternate_draft, wrapped=True)
        dependencies = {
            ConstructionLane.OPUS5_DRAFT.value: blake3_hex(primary_draft),
            ConstructionLane.GEMINI_ALTERNATE.value: blake3_hex(alternate_draft),
        }
        critique = candidate._request_body(
            stage="critique",
            model_id=routes.opus48.model,
            construction_input=self._input(),
            evidence_documents=self._evidence(),
            draft=primary,
            premium_context=premium.alternate_author_context(
                alternate_draft=alternate
            ),
        )
        comparison = premium.sol_comparison_request(
            model_id=routes.gpt56.model,
            construction_input=self._input(),
            evidence_documents=self._evidence(),
            primary_draft=primary,
            alternate_draft=alternate,
            seed=self._input()["runtime"]["seed"],
        )
        return (
            self._turn_request(
                source,
                routes,
                ConstructionLane.OPUS48_CRITIQUE,
                critique,
                dependencies,
                _ValidationPlan(
                    ConstructionLane.OPUS48_CRITIQUE,
                    critique,
                    primary_draft=primary,
                    alternate_draft=alternate,
                ),
            ),
            self._turn_request(
                source,
                routes,
                ConstructionLane.GPT56_COMPARISON,
                comparison,
                dependencies,
                _ValidationPlan(
                    ConstructionLane.GPT56_COMPARISON,
                    comparison,
                    primary_draft=primary,
                    alternate_draft=alternate,
                ),
            ),
        )

    def prepare_critic_quorum(
        self,
        source: FrozenConstructionSource,
        routes: PremiumConstructionRoutes,
        *,
        primary_draft: Mapping[str, JsonValue],
        alternate_draft: Mapping[str, JsonValue],
    ) -> Sequence[ConstructionTurnRequest]:
        """Build four independent critic attempts from two exact requests.

        Each backup is a prospective supplemental attempt, not a retry.  Its
        source request is the primary role's exact request with only ``model``
        replaced.  Both attempts still pass the same sibling output compiler
        and validator for that role.
        """

        critique_request, comparison_request = self.prepare_critics(
            source,
            routes,
            primary_draft=primary_draft,
            alternate_draft=alternate_draft,
        )
        with self._plan_lock:
            critique_plan = self._plans.get(critique_request.request_blake3)
            comparison_plan = self._plans.get(comparison_request.request_blake3)
        if critique_plan is None or comparison_plan is None:
            raise LegacyConstructionAdapterError(
                "primary critic validation plan is unavailable"
            )
        critique_backup_phase = self._model_only_projection(
            critique_plan.phase_request,
            model_id=routes.opus5.model,
        )
        comparison_backup_phase = self._model_only_projection(
            comparison_plan.phase_request,
            model_id=routes.gemini31.model,
        )
        critique_backup = self._turn_request(
            source,
            routes,
            ConstructionLane.OPUS5_CRITIQUE_BACKUP,
            critique_backup_phase,
            critique_request.dependencies,
            _ValidationPlan(
                ConstructionLane.OPUS5_CRITIQUE_BACKUP,
                critique_backup_phase,
                primary_draft=critique_plan.primary_draft,
                alternate_draft=critique_plan.alternate_draft,
            ),
        )
        comparison_backup = self._turn_request(
            source,
            routes,
            ConstructionLane.GEMINI_COMPARISON_BACKUP,
            comparison_backup_phase,
            comparison_request.dependencies,
            _ValidationPlan(
                ConstructionLane.GEMINI_COMPARISON_BACKUP,
                comparison_backup_phase,
                primary_draft=comparison_plan.primary_draft,
                alternate_draft=comparison_plan.alternate_draft,
            ),
        )
        if not (
            critique_request.turn_input.public_text
            == critique_backup.turn_input.public_text
            and canonical_json_bytes(critique_request.turn_input.output_schema)
            == canonical_json_bytes(critique_backup.turn_input.output_schema)
            and comparison_request.turn_input.public_text
            == comparison_backup.turn_input.public_text
            and canonical_json_bytes(comparison_request.turn_input.output_schema)
            == canonical_json_bytes(comparison_backup.turn_input.output_schema)
        ):
            raise LegacyConstructionAdapterError(
                "supplemental critic prompt/schema projection differs"
            )
        return (
            critique_request,
            critique_backup,
            comparison_request,
            comparison_backup,
        )

    def prepare_critic_authority_v4(
        self,
        source: FrozenConstructionSource,
        routes: PremiumConstructionRoutes,
        *,
        primary_draft: Mapping[str, JsonValue],
        alternate_draft: Mapping[str, JsonValue],
    ) -> Sequence[ConstructionTurnRequest]:
        """Build the prospective v4 evidence/authority critic frontier.

        Opus 4.8 and GPT-5.6 each keep one exact, independently validated
        semantic attempt.  Opus 5 receives the *same bytes and schemas* for
        both roles and is the only downstream authority.  These are distinct
        lanes, never retries or substitutions for a failed lower-tier turn.
        """

        critique_request, comparison_request = self.prepare_critics(
            source,
            routes,
            primary_draft=primary_draft,
            alternate_draft=alternate_draft,
        )
        with self._plan_lock:
            critique_plan = self._plans.get(critique_request.request_blake3)
            comparison_plan = self._plans.get(comparison_request.request_blake3)
        if critique_plan is None or comparison_plan is None:
            raise LegacyConstructionAdapterError(
                "v4 critic validation plan is unavailable"
            )

        def opus_projection(
            lane: ConstructionLane,
            original: ConstructionTurnRequest,
            plan: _ValidationPlan,
        ) -> ConstructionTurnRequest:
            phase = self._model_only_projection(
                plan.phase_request, model_id=routes.opus5.model
            )
            return self._turn_request(
                source,
                routes,
                lane,
                phase,
                original.dependencies,
                _ValidationPlan(
                    lane,
                    phase,
                    primary_draft=plan.primary_draft,
                    alternate_draft=plan.alternate_draft,
                ),
                route_override=routes.opus5,
            )

        opus_critique = opus_projection(
            ConstructionLane.OPUS5_CRITIQUE_BACKUP,
            critique_request,
            critique_plan,
        )
        opus_comparison = opus_projection(
            ConstructionLane.GEMINI_COMPARISON_BACKUP,
            comparison_request,
            comparison_plan,
        )
        if not (
            critique_request.turn_input.public_text
            == opus_critique.turn_input.public_text
            and canonical_json_bytes(critique_request.turn_input.output_schema)
            == canonical_json_bytes(opus_critique.turn_input.output_schema)
            and comparison_request.turn_input.public_text
            == opus_comparison.turn_input.public_text
            and canonical_json_bytes(comparison_request.turn_input.output_schema)
            == canonical_json_bytes(opus_comparison.turn_input.output_schema)
        ):
            raise LegacyConstructionAdapterError(
                "v4 Opus authority prompt/schema projection differs"
            )
        return (
            critique_request,
            opus_critique,
            comparison_request,
            opus_comparison,
        )

    def prepare_revision(
        self,
        source: FrozenConstructionSource,
        routes: PremiumConstructionRoutes,
        *,
        primary_draft: Mapping[str, JsonValue],
        alternate_draft: Mapping[str, JsonValue],
        opus48_critique: Mapping[str, JsonValue],
        gpt56_comparison: Mapping[str, JsonValue],
        selected_critique_lane: ConstructionLane = ConstructionLane.OPUS48_CRITIQUE,
        selected_comparison_lane: ConstructionLane = ConstructionLane.GPT56_COMPARISON,
    ) -> ConstructionTurnRequest:
        self._require_source(source)
        self._require_routes(routes)
        candidate = self._state.authority.candidate
        premium = self._state.authority.premium
        primary = self._compile_draft(primary_draft, wrapped=False)
        alternate = self._compile_draft(alternate_draft, wrapped=True)
        critique = self._compile_critique(opus48_critique, primary=primary)
        comparison = self._compile_comparison(
            gpt56_comparison, primary=primary, alternate=alternate
        )
        if (
            selected_critique_lane not in CRITIQUE_QUORUM_LANES
            or selected_comparison_lane not in COMPARISON_QUORUM_LANES
        ):
            raise LegacyConstructionAdapterError("selected critic role differs")
        if (
            selected_critique_lane is ConstructionLane.OPUS48_CRITIQUE
            and selected_comparison_lane is ConstructionLane.GPT56_COMPARISON
        ):
            # Keep historical v1/v2 and the v3 primary-selected request byte
            # compatible with the exact sibling context builder.
            premium_context = premium.premium_context(
                alternate_draft=alternate, sol_comparison=comparison
            )
        else:
            critique_route = routes.for_lane(selected_critique_lane)
            comparison_route = routes.for_lane(selected_comparison_lane)
            premium_context = {
                "schema": "eva.premium-codex-selected-critic-context.v1",
                "alternate_candidate": deepcopy(dict(alternate)),
                "alternate_candidate_sha256": candidate.sha256_value(alternate),
                "selected_critique_identity": {
                    "lane": selected_critique_lane.value,
                    "model": critique_route.model,
                    "provider": critique_route.provider,
                    "compiled_output_sha256": candidate.sha256_value(critique),
                },
                "selected_comparison_identity": {
                    "lane": selected_comparison_lane.value,
                    "model": comparison_route.model,
                    "provider": comparison_route.provider,
                    "compiled_output_sha256": candidate.sha256_value(comparison),
                },
                "selected_independent_comparison": deepcopy(dict(comparison)),
                "selected_independent_comparison_sha256": candidate.sha256_value(
                    comparison
                ),
                "selection_policy": CRITIC_QUORUM_POLICY_V1,
                "selection_protocol": (
                    "each critic role ran two independent fixed attempts; the "
                    "primary is selected when valid, otherwise the exact-schema "
                    "supplemental result is selected; every attempt remains retained"
                ),
                "public_calibration": deepcopy(premium._CALIBRATION),
                "private_answers_in_context": False,
            }
        phase_request = candidate._request_body(
            stage="revision",
            model_id=routes.opus5.model,
            construction_input=self._input(),
            evidence_documents=self._evidence(),
            draft=primary,
            critique=critique,
            premium_context=premium_context,
        )
        dependencies = {
            ConstructionLane.OPUS5_DRAFT.value: blake3_hex(primary_draft),
            ConstructionLane.GEMINI_ALTERNATE.value: blake3_hex(alternate_draft),
            ConstructionLane.OPUS48_CRITIQUE.value: blake3_hex(opus48_critique),
            ConstructionLane.GPT56_COMPARISON.value: blake3_hex(gpt56_comparison),
        }
        return self._turn_request(
            source,
            routes,
            ConstructionLane.OPUS5_REVISION,
            phase_request,
            dependencies,
            _ValidationPlan(
                ConstructionLane.OPUS5_REVISION,
                phase_request,
                primary_draft=primary,
                alternate_draft=alternate,
                critique=critique,
            ),
        )

    def validate_output(
        self,
        source: FrozenConstructionSource,
        request: ConstructionTurnRequest,
        output: Mapping[str, JsonValue],
    ) -> None:
        self._require_source(source)
        with self._plan_lock:
            plan = self._plans.get(request.request_blake3)
        if plan is None or plan.lane is not request.lane:
            raise LegacyConstructionAdapterError("legacy phase validation plan is unavailable")
        if request.source_phase_request_blake3 != blake3_hex(plan.phase_request):
            raise LegacyConstructionAdapterError("legacy phase request changed before validation")
        if request.lane is ConstructionLane.OPUS5_DRAFT:
            self._compile_draft(output, wrapped=False)
        elif request.lane is ConstructionLane.GEMINI_ALTERNATE:
            self._compile_draft(output, wrapped=True)
        elif request.lane in CRITIQUE_QUORUM_LANES:
            assert plan.primary_draft is not None
            self._compile_critique(output, primary=plan.primary_draft)
        elif request.lane in COMPARISON_QUORUM_LANES:
            assert plan.primary_draft is not None and plan.alternate_draft is not None
            self._compile_comparison(
                output,
                primary=plan.primary_draft,
                alternate=plan.alternate_draft,
            )
        elif request.lane is ConstructionLane.OPUS5_REVISION:
            assert plan.primary_draft is not None and plan.critique is not None
            self._compile_revision(
                output,
                primary=plan.primary_draft,
                critique=plan.critique,
            )
        else:  # pragma: no cover - enum exhaustiveness
            raise LegacyConstructionAdapterError("legacy construction lane differs")

    def resolve_supplemental_author_fallback(
        self,
        source: FrozenConstructionSource,
        request: ConstructionTurnRequest,
        *,
        primary_draft: Mapping[str, JsonValue],
        failure: ConstructionFailureReceipt,
    ) -> Mapping[str, JsonValue]:
        """Canonically wrap the validated primary without claiming Gemini authored it."""

        self._require_source(source)
        with self._plan_lock:
            plan = self._plans.get(request.request_blake3)
        if not (
            request.lane is ConstructionLane.GEMINI_ALTERNATE
            and plan is not None
            and plan.lane is ConstructionLane.GEMINI_ALTERNATE
            and request.source_phase_request_blake3 == blake3_hex(plan.phase_request)
            and failure.lane is ConstructionLane.GEMINI_ALTERNATE
            and failure.source_request_blake3 == source.request_blake3
            and failure.request_blake3 == request.request_blake3
            and failure.output_schema_blake3 == request.source_output_schema_blake3
        ):
            raise LegacyConstructionAdapterError(
                "supplemental Gemini failure does not bind its exact request"
            )
        # Reassert the primary hard gate, then feed its exact canonical wire
        # through the ordinary wrapped Gemini compiler.  No legacy validator
        # is skipped or weakened.
        self._compile_draft(primary_draft, wrapped=False)
        premium = self._state.authority.premium
        fallback = freeze_json(
            {
                "payload": {
                    premium.GEMINI_OUTPUT_KEY: canonical_json_bytes(primary_draft)
                    .decode("utf-8")
                    .rstrip("\n")
                }
            }
        )
        self._compile_draft(fallback, wrapped=True)
        return fallback

    @staticmethod
    def _assert_request_receipt(
        request: ConstructionTurnRequest, result_phase: Any
    ) -> None:
        receipt = result_phase.receipt
        if not (
            receipt.lane is request.lane
            and receipt.source_phase_request_blake3
            == request.source_phase_request_blake3
            and receipt.request_blake3 == request.request_blake3
            and receipt.output_schema_blake3 == request.source_output_schema_blake3
            and canonical_value(receipt.dependency_blake3s)
            == canonical_value(request.dependencies)
        ):
            raise LegacyConstructionAdapterError(
                "published phase does not reopen against its exact legacy request"
            )
        raw = result_phase.codex_turn_receipt.final_response
        if raw is None or request.turn_input.output_schema is None:
            raise LegacyConstructionAdapterError(
                "published phase raw response evidence is unavailable"
            )
        extracted = extract_unique_schema_valid_payload(
            raw, request.turn_input.output_schema
        )
        if canonical_value(extracted) != canonical_value(result_phase.output):
            raise LegacyConstructionAdapterError(
                "published phase raw response extraction differs"
            )

    @staticmethod
    def _assert_failure_request(
        request: ConstructionTurnRequest, failure: ConstructionFailureReceipt
    ) -> None:
        if not (
            failure.lane is request.lane
            and failure.source_request_blake3 == request.source_request_blake3
            and failure.request_blake3 == request.request_blake3
            and failure.output_schema_blake3 == request.source_output_schema_blake3
        ):
            raise LegacyConstructionAdapterError(
                "published failure does not reopen against its exact request"
            )

    def reopen_result(
        self,
        source: FrozenConstructionSource,
        result: PremiumConstructionResult,
        routes: PremiumConstructionRoutes,
        *,
        completed_at_utc: str,
    ) -> LegacyReopenedConstruction:
        """Rebuild every request, rerun every authority validator, then compile."""

        self._require_source(source)
        self._require_routes(routes)
        verify_premium_construction_result(result)
        if (
            result.source_id != source.source_id
            or result.source_request_blake3 != source.request_blake3
        ):
            raise LegacyConstructionAdapterError("construction result source differs")
        by_lane = {phase.receipt.lane: phase for phase in result.phases}
        author_requests = self.prepare_authors(source, routes)
        primary_request, alternate_request = author_requests
        primary_phase = by_lane[ConstructionLane.OPUS5_DRAFT]
        self._assert_request_receipt(primary_request, primary_phase)
        self.validate_output(source, primary_request, primary_phase.output)
        primary_output = by_lane[ConstructionLane.OPUS5_DRAFT].output
        alternate_phase = by_lane.get(ConstructionLane.GEMINI_ALTERNATE)
        if alternate_phase is not None:
            self._assert_request_receipt(alternate_request, alternate_phase)
            self.validate_output(source, alternate_request, alternate_phase.output)
            alternate_output = alternate_phase.output
        else:
            gemini_failures = tuple(
                failure
                for failure in result.supplemental_failures
                if failure.lane is ConstructionLane.GEMINI_ALTERNATE
            )
            if len(gemini_failures) != 1 or len(result.fallbacks) != 1:
                raise LegacyConstructionAdapterError(
                    "supplemental author publication proof differs"
                )
            failure = gemini_failures[0]
            fallback = result.fallbacks[0]
            fallback_receipt = fallback.receipt
            if not (
                failure.request_blake3 == alternate_request.request_blake3
                and failure.output_schema_blake3
                == alternate_request.source_output_schema_blake3
                and fallback_receipt.source_phase_request_blake3
                == alternate_request.source_phase_request_blake3
                and fallback_receipt.request_blake3
                == alternate_request.request_blake3
                and fallback_receipt.output_schema_blake3
                == alternate_request.source_output_schema_blake3
            ):
                raise LegacyConstructionAdapterError(
                    "supplemental author does not reopen against its exact request"
                )
            derived = self.resolve_supplemental_author_fallback(
                source,
                alternate_request,
                primary_draft=primary_output,
                failure=failure,
            )
            if canonical_value(derived) != canonical_value(fallback.output):
                raise LegacyConstructionAdapterError(
                    "supplemental author fallback is not deterministically reproducible"
                )
            # Explicitly rerun the same registered wrapped validator used for
            # an actual alternate-author response.
            self.validate_output(source, alternate_request, fallback.output)
            alternate_output = fallback.output
        critic_requests = (
            self.prepare_critic_authority_v4(
                source,
                routes,
                primary_draft=primary_output,
                alternate_draft=alternate_output,
            )
            if result.schema == PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4
            else self.prepare_critic_quorum(
                source,
                routes,
                primary_draft=primary_output,
                alternate_draft=alternate_output,
            )
            if result.schema in {
                PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
                PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4,
            }
            else self.prepare_critics(
                source,
                routes,
                primary_draft=primary_output,
                alternate_draft=alternate_output,
            )
        )
        for request in critic_requests:
            phase = by_lane.get(request.lane)
            if phase is not None:
                self._assert_request_receipt(request, phase)
                self.validate_output(source, request, phase.output)
                continue
            failures = tuple(
                failure
                for failure in result.supplemental_failures
                if failure.lane is request.lane
            )
            if len(failures) != 1:
                raise LegacyConstructionAdapterError(
                    "critic attempt evidence is incomplete"
                )
            self._assert_failure_request(request, failures[0])
        if result.schema in {
            PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
            PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4,
        }:
            selected = dict(result.selected_critic_receipt_blake3s or {})
            phases_by_receipt = {
                phase.receipt.receipt_blake3: phase for phase in result.phases
            }
            try:
                critique_phase = phases_by_receipt[selected["critique"]]
                comparison_phase = phases_by_receipt[selected["comparison"]]
            except KeyError:
                raise LegacyConstructionAdapterError(
                    "selected critic receipt cannot be reopened"
                ) from None
            expected_critique = (
                (ConstructionLane.OPUS5_CRITIQUE_BACKUP,)
                if result.schema == PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4
                else CRITIQUE_QUORUM_LANES
            )
            expected_comparison = (
                (ConstructionLane.GEMINI_COMPARISON_BACKUP,)
                if result.schema == PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4
                else COMPARISON_QUORUM_LANES
            )
            if not (
                critique_phase.receipt.lane in expected_critique
                and comparison_phase.receipt.lane in expected_comparison
            ):
                raise LegacyConstructionAdapterError("selected critic role differs")
        else:
            critique_phase = by_lane[ConstructionLane.OPUS48_CRITIQUE]
            comparison_phase = by_lane[ConstructionLane.GPT56_COMPARISON]
        critique_output = critique_phase.output
        comparison_output = comparison_phase.output
        revision_request = self.prepare_revision(
            source,
            routes,
            primary_draft=primary_output,
            alternate_draft=alternate_output,
            opus48_critique=critique_output,
            gpt56_comparison=comparison_output,
            selected_critique_lane=critique_phase.receipt.lane,
            selected_comparison_lane=comparison_phase.receipt.lane,
        )
        revision_phase = by_lane[ConstructionLane.OPUS5_REVISION]
        self._assert_request_receipt(revision_request, revision_phase)
        self.validate_output(source, revision_request, revision_phase.output)

        primary = self._compile_draft(primary_output, wrapped=False)
        alternate = self._compile_draft(alternate_output, wrapped=True)
        compiled_critics: dict[ConstructionLane, Mapping[str, Any]] = {}
        for request in critic_requests:
            phase = by_lane.get(request.lane)
            if phase is None:
                continue
            if request.lane in CRITIQUE_QUORUM_LANES:
                compiled_critics[request.lane] = self._compile_critique(
                    phase.output, primary=primary
                )
            elif request.lane in COMPARISON_QUORUM_LANES:
                compiled_critics[request.lane] = self._compile_comparison(
                    phase.output, primary=primary, alternate=alternate
                )
        critique = compiled_critics[critique_phase.receipt.lane]
        comparison = compiled_critics[comparison_phase.receipt.lane]
        revision = self._compile_revision(
            revision_phase.output, primary=primary, critique=critique
        )
        candidate = self._state.authority.candidate
        copied_evidence = [
            {
                "evidence_id": item["evidence_id"],
                "output_relative_path": f"evidence/{item['evidence_id']}.json",
                "byte_count": len(self._state.evidence_payloads[item["evidence_id"]]),
                "sha256": hashlib.sha256(
                    self._state.evidence_payloads[item["evidence_id"]]
                ).hexdigest(),
            }
            for item in self._input()["evidence_objects"]
        ]
        revision_plan = self._plans[revision_request.request_blake3]
        revision_call = {
            "completed_at_utc": completed_at_utc,
            "request": {
                "sha256": candidate.sha256_value(revision_plan.phase_request)
            },
            "visible_output": {"sha256": candidate.sha256_value(revision)},
        }
        compiled = candidate._compile_candidate(
            construction_input=self._input(),
            evidence_documents=self._evidence(),
            copied_evidence=copied_evidence,
            source_verification=deepcopy(dict(self._state.source_verification)),
            blueprint=revision["blueprint"],
            revision_call=revision_call,
            host_bind_template_episode=True,
        )
        private_sha = candidate.sha256_value(compiled["private_evaluation"])
        evidence_sha = candidate.sha256_value(compiled["evidence_manifest"])
        reference = deepcopy(compiled["reference_answer"])
        reference["private_evaluation_sha256"] = private_sha
        reference["evidence_manifest_sha256"] = evidence_sha
        compiled["reference_answer"] = reference
        candidate.validate_policy(compiled["policy"])
        candidate.validate_private_evaluation(compiled["private_evaluation"])
        candidate.validate_evidence_manifest(
            compiled["evidence_manifest"], compiled["private_evaluation"]
        )
        candidate.validate_reference_answer(
            reference,
            private_evaluation=compiled["private_evaluation"],
            evidence_manifest=compiled["evidence_manifest"],
            private_evaluation_file_sha256=private_sha,
            evidence_manifest_file_sha256=evidence_sha,
        )
        candidate.validate_stage_plan_contract(compiled["s1_plan_contract"])
        candidate.validate_evidence_service_contract(compiled["s2_evidence_contract"])
        candidate.validate_execution_contract(compiled["s3_execution_contract"])
        candidate.validate_execution_contract(compiled["s4_execution_contract"])
        candidate.validate_submission_contract(compiled["s5_submission_contract"])

        phase_requests = {
            request.lane.value: self._plans[request.request_blake3].phase_request
            for request in (*author_requests, *critic_requests, revision_request)
        }
        phase_outputs: dict[str, Any] = {
            ConstructionLane.OPUS5_DRAFT.value: primary,
            ConstructionLane.GEMINI_ALTERNATE.value: alternate,
            ConstructionLane.OPUS5_REVISION.value: revision,
        }
        phase_outputs.update(
            {
                lane.value: compiled
                for lane, compiled in compiled_critics.items()
            }
        )
        executable_material_blake3 = blake3_hex(compiled)
        return LegacyReopenedConstruction(
            phase_requests=freeze_json(phase_requests),
            compiled_phase_outputs=freeze_json(phase_outputs),
            executable_artifacts=freeze_json(compiled),
            validator_authority_blake3=self.validator_authority.authority_blake3,
            executable_material_blake3=executable_material_blake3,
        )


@dataclass(frozen=True)
class LegacyConstructionComposition:
    """Minimal production-callable source + adapter + sink composition."""

    source: FrozenConstructionSource
    adapter: LegacyExactConstructionAdapter
    sink: "LegacyAppendOnlyConstructionSink"

    def run(
        self, orchestrator: PremiumCodexConstructionOrchestrator
    ) -> PublishedConstructionResult:
        return orchestrator.construct_and_publish(
            self.source, self.adapter, self.sink
        )


class LegacySelectionV2ConstructionFactory:
    """Resolve frozen selection-v2 identities without opening outcome evidence."""

    def __init__(
        self,
        *,
        selection: CampaignSelectionV2,
        readiness: SignedSupervisorV24Readiness,
        authority_root: str | Path,
        supervisor_root: str | Path,
        trust_store_path: str | Path,
        legacy_package_src: str | Path,
        workspace_root: str | Path,
        output_root: str | Path,
        id_factory: RuntimeIdFactory | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(selection, CampaignSelectionV2):
            raise LegacyConstructionAdapterError("verified selection-v2 object is required")
        if not isinstance(readiness, SignedSupervisorV24Readiness):
            raise LegacyConstructionAdapterError("verified v24 readiness object is required")
        # Reopen the self-committed selection and bind it to the independently
        # verified signed-readiness projection supplied by deployment.
        CampaignSelectionV2.from_document(selection.to_document())
        if readiness.authority_blake3 != blake3_hex(readiness.core_document()):
            raise LegacyConstructionAdapterError("v24 readiness authority differs")
        if canonical_value(selection.readiness_authority) != canonical_value(
            readiness.to_document()
        ):
            raise LegacyConstructionAdapterError("selection-v2 readiness authority differs")
        self.selection = selection
        self.readiness = readiness
        self.authority_root = _existing_directory(authority_root, label="legacy authority root")
        self.supervisor_root = _existing_directory(supervisor_root, label="v24 supervisor root")
        try:
            supervisor_relative = self.supervisor_root.relative_to(self.authority_root)
        except ValueError:
            raise LegacyConstructionAdapterError("v24 supervisor escapes authority root") from None
        registry_relative = PurePosixPath(supervisor_relative.as_posix()) / _REGISTRY_NAME
        registry_bytes = _read_relative(
            self.authority_root,
            registry_relative,
            label="v24 registry",
            maximum_bytes=_MAX_REGISTRY_BYTES,
        )
        if not (
            blake3_bytes(registry_bytes) == readiness.registry_file_blake3
            and hashlib.sha256(registry_bytes).hexdigest()
            == readiness.registry_file_upstream_sha256
        ):
            raise LegacyConstructionAdapterError("v24 registry file commitment differs")
        registry = _decode_object(registry_bytes, label="v24 registry")
        candidates = registry.get("candidates")
        if not (
            registry.get("campaign_id") == readiness.campaign_id
            and registry.get("registry_revision") == readiness.registry_revision
            and registry.get("registry_sha256")
            == readiness.registry_logical_upstream_sha256
            and isinstance(candidates, list)
            and len(candidates) == readiness.candidate_count
        ):
            raise LegacyConstructionAdapterError("v24 registry projection differs")
        self._registry = MappingProxyType(
            {
                row["candidate_id"]: row
                for row in candidates
                if isinstance(row, dict) and isinstance(row.get("candidate_id"), str)
            }
        )
        if len(self._registry) != readiness.candidate_count:
            raise LegacyConstructionAdapterError("v24 registry identities differ")
        self._entries = MappingProxyType(
            {entry.candidate_id: entry for entry in selection.entries}
        )
        self._readiness = readiness.readiness_by_candidate
        self._authority = _LegacyAuthority(legacy_package_src)
        trust_path = _absolute(trust_store_path, label="legacy trust store")
        try:
            trust_relative = PurePosixPath(trust_path.relative_to(self.authority_root).as_posix())
        except ValueError:
            raise LegacyConstructionAdapterError("legacy trust store escapes authority root") from None
        trust_bytes = _read_relative(
            self.authority_root,
            trust_relative,
            label="legacy trust store",
            maximum_bytes=_MAX_CONTROL_BYTES,
        )
        self._trust_store = _decode_object(trust_bytes, label="legacy trust store")
        self._authority.candidate.validate_host_trust_store(self._trust_store)
        if self._trust_store.get("status") != "active":
            raise LegacyConstructionAdapterError("legacy trust store is not active")
        self.workspace_root = _ensure_directory(workspace_root, label="construction workspace root")
        self.output_root = _ensure_directory(output_root, label="construction output root")
        _ensure_directory(self.output_root / ".claims", label="construction claim root")
        self._ids = id_factory or RandomUUIDFactory()
        self._clock = clock or (lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z"))

    def _workspace_roots(self, candidate_id: str) -> Mapping[ConstructionLane, Path]:
        try:
            candidate_root = _mkdir_new_child(self.workspace_root, candidate_id)
        except LegacyConstructionAdapterError:
            # A provider-free preflight reserves the same deterministic roots
            # that execution later reopens.  Reuse is permitted only while the
            # topology is exact and every lane is still completely empty.  A
            # provider/tool write, extra file, partial topology, or symlink
            # therefore makes a second semantic attempt fail closed.
            candidate_root = _existing_directory(
                self.workspace_root / candidate_id,
                label="existing construction candidate workspace",
            )
            expected_v3 = {lane.value for lane in V3_LANE_ORDER}
            expected_v1_v2 = {lane.value for lane in LANE_ORDER}
            descriptor = _open_directory_chain(candidate_root, create=False)
            try:
                observed = set(os.listdir(descriptor))
            finally:
                os.close(descriptor)
            if frozenset(observed) not in {
                frozenset(expected_v3),
                frozenset(expected_v1_v2),
            }:
                raise LegacyConstructionAdapterError(
                    "existing construction workspace topology differs"
                ) from None
            roots: dict[ConstructionLane, Path] = {}
            lanes = (
                V3_LANE_ORDER
                if observed == expected_v3
                else LANE_ORDER
            )
            for lane in lanes:
                root = _existing_directory(
                    candidate_root / lane.value,
                    label="existing construction lane workspace",
                )
                descriptor = _open_directory_chain(root, create=False)
                try:
                    if os.listdir(descriptor):
                        raise LegacyConstructionAdapterError(
                            "existing construction lane workspace is not fresh"
                        )
                finally:
                    os.close(descriptor)
                roots[lane] = root
            return MappingProxyType(roots)
        roots = {
            lane: _mkdir_new_child(candidate_root, lane.value)
            for lane in V3_LANE_ORDER
        }
        return MappingProxyType(roots)

    def compose(
        self, candidate_id: str, routes: PremiumConstructionRoutes
    ) -> LegacyConstructionComposition:
        try:
            entry = self._entries[candidate_id]
        except KeyError:
            raise LegacyConstructionAdapterError(
                "candidate is outside selection-v2"
            ) from None
        if entry.construction_readiness != "frozen":
            raise LegacyConstructionAdapterError(
                "premium construction accepts only selection-v2 frozen identities"
            )
        readiness = self._readiness.get(entry.source_candidate_id)
        if not (
            readiness is not None
            and readiness.readiness.value == "frozen"
            and readiness.source_family == entry.source_family
            and readiness.stage == entry.stage
            and readiness.proof_root_blake3 == entry.readiness_proof_root_blake3
        ):
            raise LegacyConstructionAdapterError("selection-v2 frozen readiness differs")
        registry = self._registry.get(entry.source_candidate_id)
        if not isinstance(registry, Mapping):
            raise LegacyConstructionAdapterError("frozen source is absent from v24 registry")
        frozen = registry.get("frozen_artifact")
        if not (
            registry.get("family") == entry.source_family
            and registry.get("focus") == entry.stage
            and isinstance(frozen, Mapping)
            and frozen.get("sha256") == entry.source_artifact_sha256
        ):
            raise LegacyConstructionAdapterError("selection-v2 source identity differs")
        input_relative = _safe_relative(
            frozen.get("path"), label="frozen construction input"
        )
        input_bytes = _read_relative(
            self.authority_root,
            input_relative,
            label="frozen construction input",
            maximum_bytes=_MAX_SOURCE_BYTES,
        )
        if hashlib.sha256(input_bytes).hexdigest() != entry.source_artifact_sha256:
            raise LegacyConstructionAdapterError("frozen construction input SHA-256 differs")
        construction_input = _decode_object(
            input_bytes, label="frozen construction input"
        )
        candidate = self._authority.candidate
        candidate.validate_construction_input(construction_input)
        if not (
            construction_input["sandbox"]["sandbox_id"] == entry.source_candidate_id
            and construction_input["sandbox"]["focus"] == entry.stage
        ):
            raise LegacyConstructionAdapterError("frozen construction sandbox differs")
        (
            _verification_path,
            source_verification,
            verification_snapshot,
        ) = candidate._load_source_verification(
            construction_input, self.authority_root
        )
        attestation_ref = construction_input[
            "evidence_source_verification_attestation"
        ]
        attestation_relative = _safe_relative(
            attestation_ref["source_relative_path"],
            label="source-verification attestation",
        )
        attestation_bytes = _read_relative(
            self.authority_root,
            attestation_relative,
            label="source-verification attestation",
            maximum_bytes=attestation_ref["byte_count"],
        )
        if not (
            len(attestation_bytes) == attestation_ref["byte_count"]
            and hashlib.sha256(attestation_bytes).hexdigest()
            == attestation_ref["sha256"]
        ):
            raise LegacyConstructionAdapterError(
                "source-verification attestation bytes differ"
            )
        source_attestation = _decode_object(
            attestation_bytes, label="source-verification attestation"
        )
        candidate._verify_source_attestation(
            report_file_sha256=verification_snapshot.sha256,
            attestation=source_attestation,
            verification=source_verification,
            trust_store=self._trust_store["keys"],
        )
        evidence, support, evidence_snapshots = candidate._load_evidence(
            construction_input, self.authority_root, source_verification
        )
        evidence_payloads = MappingProxyType(
            {
                evidence_id: snapshot.payload
                for evidence_id, snapshot in evidence_snapshots.items()
            }
        )
        source_document = {
            "schema": "eva.selection-v2-frozen-legacy-construction-source.v1",
            "selection_id": self.selection.selection_id,
            "selection_blake3": self.selection.selection_blake3,
            "selection_entry": entry.to_document(),
            "readiness_authority_blake3": self.readiness.authority_blake3,
            "legacy_validator_authority": self._authority.identity.to_document(),
            "construction_input_path": input_relative.as_posix(),
            "construction_input_upstream_sha256": entry.source_artifact_sha256,
            "construction_input_blake3": blake3_bytes(input_bytes),
            "construction_input": construction_input,
            "source_verification": source_verification,
            "source_verification_blake3": blake3_bytes(
                verification_snapshot.payload
            ),
            "source_attestation_blake3": blake3_bytes(attestation_bytes),
            "frozen_evidence": evidence,
            "frozen_evidence_file_blake3s": {
                evidence_id: blake3_bytes(payload)
                for evidence_id, payload in evidence_payloads.items()
            },
            "controls": {
                "selection_readiness_required": "frozen",
                "legacy_schemas_copied": False,
                "legacy_validators_reused_exactly": True,
                "provider_calls_during_source_resolution": 0,
                "historical_manifest_mutated": False,
            },
        }
        source = FrozenConstructionSource.create(
            source_id=entry.candidate_id, request=source_document
        )
        state = _LegacySourceState(
            source=source,
            entry=entry,
            authority=self._authority,
            construction_input=construction_input,
            construction_input_bytes=input_bytes,
            evidence_documents=tuple(evidence),
            evidence_payloads=evidence_payloads,
            claim_support=MappingProxyType(
                {key: frozenset(value) for key, value in support.items()}
            ),
            source_verification=source_verification,
            source_verification_bytes=verification_snapshot.payload,
            source_attestation=source_attestation,
            source_attestation_bytes=attestation_bytes,
            workspace_roots=self._workspace_roots(candidate_id),
        )
        adapter = LegacyExactConstructionAdapter(state)
        adapter._require_routes(routes)
        sink = LegacyAppendOnlyConstructionSink(
            output_root=self.output_root,
            source=source,
            adapter=adapter,
            routes=routes,
            id_factory=self._ids,
            clock=self._clock,
        )
        return LegacyConstructionComposition(source, adapter, sink)


_EXECUTABLE_PATHS = MappingProxyType(
    {
        "policy": "policy.json",
        "private_evaluation": "private-evaluation.json",
        "evidence_manifest": "evidence-manifest.json",
        "reference_answer": "private/reference-answer.json",
        "s1_plan_contract": "contracts/s1-plan-contract-template.json",
        "s2_evidence_contract": "contracts/s2-evidence-contract-template.json",
        "s3_execution_contract": "contracts/s3-execution-contract-template.json",
        "s4_execution_contract": "contracts/s4-execution-contract-template.json",
        "s5_submission_contract": "contracts/s5-submission-contract-template.json",
    }
)


class LegacyAppendOnlyConstructionSink:
    """O_EXCL, no-follow materialization of a newly validated EVA binding."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        source: FrozenConstructionSource,
        adapter: LegacyExactConstructionAdapter,
        routes: PremiumConstructionRoutes,
        id_factory: RuntimeIdFactory | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.output_root = _existing_directory(
            output_root, label="construction output root"
        )
        self.source = source
        self.adapter = adapter
        self.routes = routes
        self._ids = id_factory or RandomUUIDFactory()
        self._clock = clock or (lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z"))

    def publish(
        self, result: PremiumConstructionResult
    ) -> ConstructionPublicationReceipt:
        completed_at = self._clock()
        try:
            datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            raise LegacyConstructionAdapterError("publication clock differs") from None
        reopened = self.adapter.reopen_result(
            self.source,
            result,
            self.routes,
            completed_at_utc=completed_at,
        )
        publication_id = self._ids.new("legacy-construction-publication")
        binding_id = self._ids.new("legacy-executable-binding")
        claim = {
            "schema": "eva.premium-construction-publication-claim.v1",
            "publication_id": publication_id,
            "construction_receipt_blake3": result.receipt_blake3,
            "source_request_blake3": self.source.request_blake3,
            "retry_count": 0,
        }
        _write_new_relative(
            self.output_root,
            f".claims/{result.receipt_blake3}.json",
            canonical_json_bytes(claim),
        )
        publication_root = _mkdir_new_child(self.output_root, publication_id)
        refs: dict[str, Mapping[str, JsonValue]] = {}

        def write(name: str, relative: str, value: Any) -> None:
            refs[name] = _write_new_relative(
                publication_root, relative, canonical_json_bytes(value)
            )

        write("frozen_source", "controls/frozen-source.json", self.source.request)
        write(
            "validator_authority",
            "controls/legacy-validator-authority.json",
            self.adapter.validator_authority.to_document(),
        )
        state = self.adapter._state
        refs["construction_input"] = _write_new_relative(
            publication_root,
            "controls/construction-input.json",
            state.construction_input_bytes,
        )
        refs["source_verification"] = _write_new_relative(
            publication_root,
            "controls/evidence-source-verification.json",
            state.source_verification_bytes,
        )
        refs["source_attestation"] = _write_new_relative(
            publication_root,
            "controls/evidence-source-verification-host-attestation.json",
            state.source_attestation_bytes,
        )
        for evidence_id, payload in sorted(state.evidence_payloads.items()):
            refs[f"evidence:{evidence_id}"] = _write_new_relative(
                publication_root, f"evidence/{evidence_id}.json", payload
            )
        by_lane = {phase.receipt.lane: phase for phase in result.phases}
        failures_by_lane = {
            failure.lane: failure for failure in result.supplemental_failures
        }
        fallbacks_by_lane = {
            fallback.receipt.lane: fallback for fallback in result.fallbacks
        }
        attempt_lanes = (
            V3_LANE_ORDER
            if result.schema in {
                PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
                PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4,
            }
            else LANE_ORDER
        )
        for lane in attempt_lanes:
            write(
                f"request:{lane.value}",
                f"construction-private/{lane.value}/request.json",
                reopened.phase_requests[lane.value],
            )
            phase = by_lane.get(lane)
            if phase is not None:
                write(
                    f"wire_output:{lane.value}",
                    f"construction-private/{lane.value}/wire-output.json",
                    phase.output,
                )
                write(
                    f"compiled_output:{lane.value}",
                    f"construction-private/{lane.value}/compiled-output.json",
                    reopened.compiled_phase_outputs[lane.value],
                )
                write(
                    f"phase_receipt:{lane.value}",
                    f"construction-private/{lane.value}/phase-receipt.json",
                    phase.receipt,
                )
                write(
                    f"codex_receipt:{lane.value}",
                    f"construction-private/{lane.value}/codex-turn-receipt.json",
                    phase.codex_turn_receipt,
                )
                continue
            failure = failures_by_lane.get(lane)
            if failure is None:
                raise LegacyConstructionAdapterError(
                    "missing attempted-lane failure receipt"
                )
            write(
                f"supplemental_failure:{lane.value}",
                f"construction-private/{lane.value}/"
                "supplemental-failure-receipt.json",
                failure,
            )
            fallback = fallbacks_by_lane.get(lane)
            if lane is ConstructionLane.GEMINI_ALTERNATE and fallback is not None:
                write(
                    f"wire_output:{lane.value}",
                    f"construction-private/{lane.value}/wire-output.json",
                    fallback.output,
                )
                write(
                    f"compiled_output:{lane.value}",
                    f"construction-private/{lane.value}/compiled-output.json",
                    reopened.compiled_phase_outputs[lane.value],
                )
                write(
                    f"deterministic_fallback:{lane.value}",
                    "construction-private/gemini_alternate/"
                    "deterministic-fallback-receipt.json",
                    fallback.receipt,
                )
            elif fallback is not None:
                raise LegacyConstructionAdapterError(
                    "critic failure cannot use deterministic fallback"
                )
        for name, relative in _EXECUTABLE_PATHS.items():
            write(name, relative, reopened.executable_artifacts[name])
        write(
            "construction_result_receipt",
            "construction-private/construction-result-receipt.json",
            result,
        )
        executable_refs = {
            name: refs[name] for name in sorted(_EXECUTABLE_PATHS)
        }
        binding_core = {
            "schema": "eva.premium-codex-executable-binding.v1",
            "binding_id": binding_id,
            "publication_id": publication_id,
            "construction_run_id": result.construction_run_id,
            "source_id": self.source.source_id,
            "source_request_blake3": self.source.request_blake3,
            "construction_receipt_blake3": result.receipt_blake3,
            "validator_authority_blake3": reopened.validator_authority_blake3,
            "executable_material_blake3": reopened.executable_material_blake3,
            "executable_artifacts": executable_refs,
            "parallel_frontiers": (
                [
                    [
                        ConstructionLane.OPUS5_DRAFT.value,
                        ConstructionLane.GEMINI_ALTERNATE.value,
                    ],
                    [lane.value for lane in CRITIC_QUORUM_LANES],
                    [ConstructionLane.OPUS5_REVISION.value],
                ]
                if result.schema in {
                    PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
                    PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4,
                }
                else [
                    [
                        ConstructionLane.OPUS5_DRAFT.value,
                        ConstructionLane.GEMINI_ALTERNATE.value,
                    ],
                    [
                        ConstructionLane.OPUS48_CRITIQUE.value,
                        ConstructionLane.GPT56_COMPARISON.value,
                    ],
                    [ConstructionLane.OPUS5_REVISION.value],
                ]
            ),
            "phase_routes": [
                (
                    {
                        "lane": lane.value,
                        "model": by_lane[lane].receipt.model,
                        "provider": by_lane[lane].receipt.provider,
                        "workspace_mode": by_lane[lane].receipt.sandbox.value,
                        "semantic_attempt_count": 1,
                        "retry_count": 0,
                        "source_output_schema_blake3": by_lane[
                            lane
                        ].receipt.output_schema_blake3,
                        "phase_receipt_blake3": by_lane[
                            lane
                        ].receipt.receipt_blake3,
                        "attempt_status": "succeeded",
                    }
                    if lane in by_lane
                    else {
                        "lane": lane.value,
                        "model": self.routes.for_lane(lane).model,
                        "provider": self.routes.for_lane(lane).provider,
                        "workspace_mode": CodexSandbox.READ_ONLY.value,
                        "semantic_attempt_count": 1,
                        "retry_count": 0,
                        "source_output_schema_blake3": failures_by_lane[
                            lane
                        ].output_schema_blake3,
                        "attempt_status": (
                            "failed_with_deterministic_fallback"
                            if lane is ConstructionLane.GEMINI_ALTERNATE
                            else "failed_supplemental"
                        ),
                        "failure_receipt_blake3": failures_by_lane[
                            lane
                        ].receipt_blake3,
                        **(
                            {
                                "fallback_receipt_blake3": fallbacks_by_lane[
                                    lane
                                ].receipt.receipt_blake3
                            }
                            if lane in fallbacks_by_lane
                            else {}
                        ),
                    }
                )
                for lane in attempt_lanes
            ],
            "controls": {
                "provider_call_count": result.provider_call_count,
                "one_call_per_lane": True,
                "successful_phase_count": len(result.phases),
                "supplemental_failure_count": len(result.supplemental_failures),
                "deterministic_fallback_count": len(result.fallbacks),
                "author_quorum_policy": result.author_quorum_policy,
                "critic_quorum_policy": result.critic_quorum_policy,
                "selected_critic_receipt_blake3s": (
                    result.selected_critic_receipt_blake3s
                ),
                "supplemental_fallback_provider_call_count": 0,
                "semantic_retry_count": 0,
                "source_output_schemas_unchanged": True,
                "legacy_validators_reopened": True,
                "final_legacy_validators_unchanged": True,
                "host_bind_template_episode": True,
                "historical_manifest_emitted": False,
                "historical_manifest_compatibility_claimed": False,
                "admission_eligible": False,
                "append_only": True,
            },
            "files": {key: refs[key] for key in sorted(refs)},
        }
        binding = {
            **binding_core,
            "binding_blake3": blake3_hex(binding_core),
        }
        binding_ref = _write_new_relative(
            publication_root,
            "executable-binding.v1.json",
            canonical_json_bytes(binding),
        )
        receipt_core = {
            "schema": "eva.premium-codex-construction-publication.v1",
            "publication_id": publication_id,
            "sink_name": "eva-legacy-validated-o-excl-v1",
            "construction_receipt_blake3": result.receipt_blake3,
            "artifact_blake3": binding_ref["blake3"],
            "legacy_manifest_compatibility_verified": False,
            "metadata": {
                "publication_directory": publication_id,
                "binding_id": binding_id,
                "binding_blake3": binding["binding_blake3"],
                "validator_authority_blake3": reopened.validator_authority_blake3,
                "executable_material_blake3": reopened.executable_material_blake3,
                "file_count": len(refs) + 2,
                "append_only": True,
                "retry_count": 0,
            },
        }
        receipt = ConstructionPublicationReceipt(
            **receipt_core, receipt_blake3=blake3_hex(receipt_core)
        )
        _write_new_relative(
            publication_root,
            "publication-receipt.json",
            canonical_json_bytes(receipt),
        )
        _seal_tree(publication_root)
        verify_legacy_construction_publication(self.output_root, receipt)
        return receipt


def verify_legacy_construction_publication(
    output_root: str | Path, receipt: ConstructionPublicationReceipt
) -> Mapping[str, Any]:
    """Reopen the O_EXCL bundle and verify all new EVA BLAKE3 commitments."""

    if not isinstance(receipt, ConstructionPublicationReceipt):
        raise LegacyConstructionAdapterError("publication receipt type differs")
    root = _existing_directory(output_root, label="construction output root")
    publication = _existing_directory(
        root / receipt.publication_id, label="construction publication"
    )
    binding_bytes = _read_relative(
        publication,
        _safe_relative("executable-binding.v1.json", label="binding path"),
        label="executable binding",
        maximum_bytes=_MAX_CONTROL_BYTES,
    )
    if blake3_bytes(binding_bytes) != receipt.artifact_blake3:
        raise LegacyConstructionAdapterError("executable binding file differs")
    binding = _decode_object(binding_bytes, label="executable binding")
    claimed = binding.pop("binding_blake3", None)
    if not is_blake3(claimed) or claimed != blake3_hex(binding):
        raise LegacyConstructionAdapterError("executable binding commitment differs")
    if not (
        binding.get("publication_id") == receipt.publication_id
        and binding.get("construction_receipt_blake3")
        == receipt.construction_receipt_blake3
        and binding.get("controls", {}).get("historical_manifest_emitted") is False
        and binding.get("controls", {}).get(
            "historical_manifest_compatibility_claimed"
        )
        is False
        and binding.get("controls", {}).get("semantic_retry_count") == 0
    ):
        raise LegacyConstructionAdapterError("executable binding controls differ")
    controls = binding.get("controls", {})
    phase_routes = binding.get("phase_routes")
    if controls.get("critic_quorum_policy") is not None:
        if not (
            controls.get("author_quorum_policy")
            == "opus5_primary_required_gemini_supplemental_1_of_2_v1"
            and controls.get("critic_quorum_policy")
            in {CRITIC_QUORUM_POLICY_V1, CRITIC_AUTHORITY_POLICY_V2}
            and controls.get("provider_call_count") == 7
            and controls.get("one_call_per_lane") is True
            and controls.get("supplemental_fallback_provider_call_count") == 0
            and controls.get("source_output_schemas_unchanged") is True
            and controls.get("legacy_validators_reopened") is True
            and controls.get("final_legacy_validators_unchanged") is True
            and binding.get("parallel_frontiers")
            == [
                [lane.value for lane in V3_LANE_ORDER[:2]],
                [lane.value for lane in CRITIC_QUORUM_LANES],
                [ConstructionLane.OPUS5_REVISION.value],
            ]
            and isinstance(phase_routes, list)
            and len(phase_routes) == len(V3_LANE_ORDER)
            and all(isinstance(row, dict) for row in phase_routes)
            and {row.get("lane") for row in phase_routes}
            == {lane.value for lane in V3_LANE_ORDER}
        ):
            raise LegacyConstructionAdapterError(
                "executable binding critic quorum controls differ"
            )
        routes_by_lane = {row["lane"]: row for row in phase_routes}
        succeeded = {
            lane: row
            for lane, row in routes_by_lane.items()
            if row.get("attempt_status") == "succeeded"
        }
        failed = {
            lane: row
            for lane, row in routes_by_lane.items()
            if row.get("attempt_status")
            in {"failed_supplemental", "failed_with_deterministic_fallback"}
        }
        if len(succeeded) + len(failed) != len(V3_LANE_ORDER):
            raise LegacyConstructionAdapterError(
                "critic quorum attempt status differs"
            )
        authority_v4 = (
            controls.get("critic_quorum_policy") == CRITIC_AUTHORITY_POLICY_V2
        )
        allowed_failures = (
            {
                ConstructionLane.GEMINI_ALTERNATE.value,
                ConstructionLane.OPUS48_CRITIQUE.value,
                ConstructionLane.GPT56_COMPARISON.value,
            }
            if authority_v4
            else {
                ConstructionLane.GEMINI_ALTERNATE.value,
                *(lane.value for lane in CRITIC_QUORUM_LANES),
            }
        )
        gemini_failed = ConstructionLane.GEMINI_ALTERNATE.value in failed
        if not (
            set(failed).issubset(allowed_failures)
            and all(is_blake3(row.get("failure_receipt_blake3")) for row in failed.values())
            and controls.get("successful_phase_count") == len(succeeded)
            and controls.get("supplemental_failure_count") == len(failed)
            and controls.get("deterministic_fallback_count")
            == (1 if gemini_failed else 0)
            and (
                not gemini_failed
                or (
                    failed[ConstructionLane.GEMINI_ALTERNATE.value].get(
                        "attempt_status"
                    )
                    == "failed_with_deterministic_fallback"
                    and is_blake3(
                        failed[ConstructionLane.GEMINI_ALTERNATE.value].get(
                            "fallback_receipt_blake3"
                        )
                    )
                )
            )
            and all(
                row.get("attempt_status") == "failed_supplemental"
                for lane, row in failed.items()
                if lane != ConstructionLane.GEMINI_ALTERNATE.value
            )
        ):
            raise LegacyConstructionAdapterError(
                "critic quorum failure evidence differs"
            )
        selected = controls.get("selected_critic_receipt_blake3s")
        if not isinstance(selected, dict) or set(selected) != {"critique", "comparison"}:
            raise LegacyConstructionAdapterError("selected critic controls differ")

        def selected_primary_first(lanes: Sequence[ConstructionLane]) -> Mapping[str, Any]:
            for lane in lanes:
                row = succeeded.get(lane.value)
                if row is not None:
                    return row
            raise LegacyConstructionAdapterError("critic role quorum is unsatisfied")

        selected_critique = selected_primary_first(
            (ConstructionLane.OPUS5_CRITIQUE_BACKUP,)
            if authority_v4
            else CRITIQUE_QUORUM_LANES
        )
        selected_comparison = selected_primary_first(
            (ConstructionLane.GEMINI_COMPARISON_BACKUP,)
            if authority_v4
            else COMPARISON_QUORUM_LANES
        )
        if not (
            is_blake3(selected_critique.get("phase_receipt_blake3"))
            and is_blake3(selected_comparison.get("phase_receipt_blake3"))
            and selected["critique"] == selected_critique["phase_receipt_blake3"]
            and selected["comparison"] == selected_comparison["phase_receipt_blake3"]
        ):
            raise LegacyConstructionAdapterError("selected critic evidence differs")
    elif controls.get("author_quorum_policy") is not None:
        if not (
            controls.get("author_quorum_policy")
            == "opus5_primary_required_gemini_supplemental_1_of_2_v1"
            and controls.get("provider_call_count") == 5
            and controls.get("critic_quorum_policy") is None
            and controls.get("selected_critic_receipt_blake3s") is None
            and controls.get("one_call_per_lane") is True
            and controls.get("supplemental_fallback_provider_call_count") == 0
            and controls.get("source_output_schemas_unchanged") is True
            and controls.get("legacy_validators_reopened") is True
            and controls.get("final_legacy_validators_unchanged") is True
            and isinstance(phase_routes, list)
            and len(phase_routes) == len(LANE_ORDER)
            and {row.get("lane") for row in phase_routes if isinstance(row, dict)}
            == {lane.value for lane in LANE_ORDER}
        ):
            raise LegacyConstructionAdapterError(
                "executable binding author quorum controls differ"
            )
        failed_routes = [
            row
            for row in phase_routes
            if row.get("attempt_status") == "failed_with_deterministic_fallback"
        ]
        if controls.get("supplemental_failure_count") == 0:
            if failed_routes or controls.get("deterministic_fallback_count") != 0:
                raise LegacyConstructionAdapterError(
                    "healthy author quorum evidence differs"
                )
        elif not (
            controls.get("supplemental_failure_count") == 1
            and controls.get("deterministic_fallback_count") == 1
            and controls.get("successful_phase_count") == 4
            and len(failed_routes) == 1
            and failed_routes[0].get("lane")
            == ConstructionLane.GEMINI_ALTERNATE.value
            and is_blake3(failed_routes[0].get("failure_receipt_blake3"))
            and is_blake3(failed_routes[0].get("fallback_receipt_blake3"))
        ):
            raise LegacyConstructionAdapterError(
                "degraded author quorum evidence differs"
            )
    files = binding.get("files")
    if not isinstance(files, dict) or not files:
        raise LegacyConstructionAdapterError("executable binding file inventory differs")
    for label, reference in files.items():
        if not isinstance(label, str) or not isinstance(reference, dict):
            raise LegacyConstructionAdapterError("executable binding reference differs")
        relative = _safe_relative(reference.get("path"), label="bound artifact")
        payload = _read_relative(
            publication,
            relative,
            label="bound artifact",
            maximum_bytes=max(_MAX_SOURCE_BYTES, reference.get("byte_count", 0)),
        )
        if not (
            len(payload) == reference.get("byte_count")
            and blake3_bytes(payload) == reference.get("blake3")
            and reference.get("mode") == "0400"
        ):
            raise LegacyConstructionAdapterError("bound artifact bytes differ")
        metadata = (publication / relative.as_posix()).lstat()
        if stat.S_ISLNK(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o400:
            raise LegacyConstructionAdapterError("bound artifact mode differs")
    receipt_bytes = _read_relative(
        publication,
        _safe_relative("publication-receipt.json", label="publication receipt path"),
        label="publication receipt",
        maximum_bytes=_MAX_CONTROL_BYTES,
    )
    if _decode_object(receipt_bytes, label="publication receipt") != canonical_value(
        receipt
    ):
        raise LegacyConstructionAdapterError("publication receipt bytes differ")
    return MappingProxyType(
        {
            "publication_id": receipt.publication_id,
            "binding_blake3": claimed,
            "artifact_count": len(files),
            "verified": True,
        }
    )


__all__ = [
    "LegacyAppendOnlyConstructionSink",
    "LegacyConstructionAdapterError",
    "LegacyConstructionComposition",
    "LegacyExactConstructionAdapter",
    "LegacyReopenedConstruction",
    "LegacySelectionV2ConstructionFactory",
    "LegacyValidatorAuthorityIdentity",
    "verify_legacy_construction_publication",
]
