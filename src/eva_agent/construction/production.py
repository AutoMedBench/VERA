"""Concrete, provider-gated production composition for premium construction.

Composition is provider-free: it reopens the signed selection/readiness,
resolves only allowlisted provider metadata, builds 64 sanitized Codex
app-server launch plans, and freezes a recipe.  Network/process lifecycles
start only through :class:`PreparedPremiumConstruction.start`.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, replace
from importlib.metadata import version as distribution_version
import json
import math
import os
from pathlib import Path
import stat
import sys
from threading import RLock
from types import MappingProxyType
from typing import Any, Mapping

from codex_cli_bin import bundled_codex_path
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from eva_agent.campaign import CampaignSelectionV2
from eva_agent.codex_providers import (
    AdapterReceiptSigner,
    CodexCanaryReceipt,
    CodexProviderRoute,
    SignedAdapterReceipt,
    load_codex_provider_routes,
    verify_adapter_receipt,
    verify_canary_receipt,
)
from eva_agent.codex_runtime import (
    CodexRuntime,
    CodexSandbox,
    CodexThreadOptions,
    OpenAICodexBackend,
)
from eva_agent.codex_runtime.runtime import (
    _ExactConstructionInputPolicy,
    _new_exact_construction_input_policy,
)
from eva_agent.deployment import (
    CODEX_FIRST_RELEASE_CONFIG_OVERRIDES,
    CODEX_FIRST_RELEASE_THREAD_CONFIG,
    DEFAULT_CHILD_SOFT_NOFILE,
    CampaignDeploymentError,
    PreparedPersistentCodexRuntime,
    prepare_persistent_codex_runtime,
)
from eva_agent.deployment.codex_child_exec import build_sanitized_codex_exec_plan
from eva_agent.deployment.medresearch_v2 import (
    GatewayAffinePersistentCodexRunner,
    ShardedOpus5AdapterGateway,
)
from eva_agent.pipeline.digests import (
    blake3_bytes,
    blake3_hex,
    canonical_json_bytes,
    is_blake3,
)
from eva_agent.pipeline.contracts import freeze_json
from eva_agent.sources import load_signed_supervisor_v24_readiness

from .legacy_exact import LegacySelectionV2ConstructionFactory
from .adapter_receipts import PremiumConstructionAdapterReceiptStore
from .premium_campaign import (
    PremiumConstructionCampaign,
    PremiumConstructionCampaignConfig,
    PremiumConstructionCampaignReport,
    PremiumConstructionPreflightReceipt,
    PremiumConstructionQueue,
    build_premium_construction_queue,
    render_premium_construction_progress,
)
from .premium_codex import (
    AUTHOR_QUORUM_POLICY_V1,
    CRITIC_QUORUM_POLICY_V1,
    CRITIC_AUTHORITY_POLICY_V2,
    ConstructionLane,
    ConstructionTurnRequest,
    ConstructionModelRoute,
    PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
    PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4,
    PremiumConstructionRoutes,
    SUPPLEMENTAL_FALLBACK_POLICY_V1,
    V3_LANE_ORDER,
    V4_LANE_ORDER,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_ROOT = (PROJECT_ROOT.parent / "rlevo-med-research").resolve()
V24_ROOT = LEGACY_ROOT / "runs" / "evamed-campaign-supervisor-v24-attempt1"
SELECTION_V2 = PROJECT_ROOT / "runs" / "campaign-selection.v2.json"


_AUTHOR_QUORUM_RECIPE = MappingProxyType(
    {
        "schema": "eva.premium-construction-author-quorum-policy.v1",
        "result_schema": PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
        "policy_id": AUTHOR_QUORUM_POLICY_V1,
        "primary_required_lane": "opus5_draft",
        "supplemental_lane": "gemini_alternate",
        "supplemental_semantic_attempt_count": 1,
        "supplemental_failure_receipt_required": True,
        "fallback_policy_id": SUPPLEMENTAL_FALLBACK_POLICY_V1,
        "fallback_provider_call_count": 0,
        "fallback_authoritative_for_admission": False,
        "author_wave_provider_call_count": 2,
        "author_wave_fully_drained": True,
        "final_legacy_validators_unchanged": True,
    }
)
_CRITIC_QUORUM_RECIPE = MappingProxyType(
    {
        "schema": "eva.premium-construction-critic-quorum-policy.v1",
        "result_schema": PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
        "policy_id": CRITIC_QUORUM_POLICY_V1,
        "critic_wave_provider_call_count": 4,
        "critic_wave_fully_drained": True,
        "role_success_requirement": "at_least_one_exact_schema_result",
        "selection": "primary_first",
        "critique_primary_lane": "opus48_critique",
        "critique_backup_lane": "opus5_critique_backup",
        "comparison_primary_lane": "gpt56_comparison",
        "comparison_backup_lane": "gemini_comparison_backup",
        "failed_attempt_receipts_required": True,
        "final_legacy_validators_unchanged": True,
    }
)
_CRITIC_AUTHORITY_RECIPE_V4 = MappingProxyType(
    {
        "schema": "eva.premium-construction-critic-authority-policy.v2",
        "result_schema": PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4,
        "policy_id": CRITIC_AUTHORITY_POLICY_V2,
        "critic_wave_provider_call_count": 4,
        "critic_wave_fully_drained": True,
        "lower_tier_evidence_lanes": ["opus48_critique", "gpt56_comparison"],
        "lower_tier_semantic_attempt_count_each": 1,
        "lower_tier_failure_receipts_required": True,
        "lower_tier_authoritative_for_revision": False,
        "critique_authority_lane": "opus5_critique_backup",
        "comparison_authority_lane": "gemini_comparison_backup",
        "comparison_authority_route_id": "opus_5",
        "authority_success_requirement": "both_opus5_exact_schema_results",
        "final_legacy_validators_unchanged": True,
    }
)
_V3_WAVE_WIDTHS = (2, 4, 1)
_V3_PROVIDER_CALL_COUNT = 7
_PHASE_PLAN_KEYS = frozenset(
    {
        "lane",
        "frontier",
        "route_id",
        "model",
        "provider",
        "role",
        "workspace_mode",
        "offered_tool_count",
        "semantic_attempt_count",
        "semantic_retry_count",
        "request_max_retries",
        "stream_max_retries",
    }
)
MODEL_REGISTRY = LEGACY_ROOT / "config" / "model-registry.20260903-v2.json"
DIRECT_CANARY_PATH = PROJECT_ROOT / ".eva" / "codex-route-canaries-direct-v1.json"
OPUS5_CANARY_PATH = PROJECT_ROOT / ".eva" / "codex-route-canary-opus5-adapter-v1.json"
DIRECT_CANARY_DOCUMENT_BLAKE3 = (
    "b6d73f3ed663cf1c757afb2aef0d6fa93413955ad62b57c784a66ef6043a6693"
)
OPUS5_CANARY_DOCUMENT_BLAKE3 = (
    "aa12f6ea27be63bf940030693da88ee7df1ce211bfd6cf59c6959fd4e8511c55"
)
EXACT_MODELS = MappingProxyType(
    {
        "opus_5": "aws/anthropic/bedrock-claude-opus-5",
        "gemini_3_1_pro": "gcp/google/gemini-3.1-pro-preview",
        "opus_4_8": "azure/anthropic/claude-opus-4-8",
        "gpt_5_6_sol": "azure/openai/gpt-5.6-sol",
    }
)
EXACT_GPT56_CANARY_DOCUMENT_BLAKE3 = (
    "d631f80c7d3ba667f65e89ce7e56512fc510fd845d37f66a0def6a8b6b300aa1"
)
EXACT_GPT56_CANARY_RECEIPT_BLAKE3 = (
    "056d78425c8352f5227978dd4372af19ae1359915a115185a4210b2d5a64415c"
)
REQUIRED_ROUTE_IDS = tuple(EXACT_MODELS)
APP_SERVER_SHARDS = 64
CONSTRUCTION_OPUS_MAX_OUTPUT_TOKENS = 32_768
ROUTE_CANARY_SIGNATURE_DOMAIN = b"eva.premium-construction-route-canary.v1\x00"
_MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
_MAX_PRIVATE_KEY_BYTES = 64 * 1024


class PremiumConstructionProductionError(ValueError):
    """A production construction authority or launch boundary differs."""


def _open_absolute_nofollow(path: Path) -> int:
    """Open an absolute file through real directories, never through links."""

    parts = path.parts
    if not parts or parts[0] != path.anchor or len(parts) < 2:
        raise OSError("unsafe path topology")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    directory = os.open(path.anchor, directory_flags)
    try:
        for component in parts[1:-1]:
            if component in {"", ".", ".."}:
                raise OSError("unsafe path topology")
            child = os.open(component, directory_flags, dir_fd=directory)
            os.close(directory)
            directory = child
        if parts[-1] in {"", ".", ".."}:
            raise OSError("unsafe path topology")
        return os.open(parts[-1], file_flags, dir_fd=directory)
    finally:
        os.close(directory)


def _stable_regular_bytes(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
    required_mode: int | None = None,
    require_current_owner: bool = False,
) -> bytes:
    path = Path(path)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PremiumConstructionProductionError(f"{label} is unavailable") from exc
    if not path.is_absolute() or path != resolved or path.is_symlink():
        raise PremiumConstructionProductionError(f"{label} path is unsafe")
    try:
        before = path.lstat()
    except OSError as exc:
        raise PremiumConstructionProductionError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or path.is_symlink()
        or not 0 < before.st_size <= maximum_bytes
        or (required_mode is not None and stat.S_IMODE(before.st_mode) != required_mode)
        or (require_current_owner and before.st_uid != os.getuid())
    ):
        raise PremiumConstructionProductionError(f"{label} topology or size differs")
    try:
        descriptor = _open_absolute_nofollow(path)
        try:
            chunks: list[bytes] = []
            remaining = maximum_bytes + 1
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after_fd = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after = path.lstat()
    except OSError as exc:
        raise PremiumConstructionProductionError(f"{label} could not be reopened") from exc
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if len(raw) != before.st_size or identity(before) != identity(after_fd) or identity(before) != identity(after):
        raise PremiumConstructionProductionError(f"{label} changed while read")
    if len(raw) > maximum_bytes:
        raise PremiumConstructionProductionError(f"{label} size differs")
    return raw


def _strict_json(
    path: Path, *, label: str, maximum_bytes: int = _MAX_DOCUMENT_BYTES
) -> Mapping[str, Any]:
    raw = _stable_regular_bytes(
        path,
        label=label,
        maximum_bytes=maximum_bytes,
    )
    def object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PremiumConstructionProductionError(
                    f"{label} contains duplicate JSON keys"
                )
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=object_without_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PremiumConstructionProductionError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise PremiumConstructionProductionError(f"{label} must be an object")
    return MappingProxyType(value)


def _document_commitment(value: Mapping[str, Any], *, label: str) -> str:
    digest = value.get("document_blake3")
    core = {key: item for key, item in value.items() if key != "document_blake3"}
    if not is_blake3(digest) or blake3_hex(core) != digest:
        raise PremiumConstructionProductionError(f"{label} BLAKE3 differs")
    return digest


def _canary(value: Any, *, label: str) -> CodexCanaryReceipt:
    if not isinstance(value, Mapping):
        raise PremiumConstructionProductionError(f"{label} receipt differs")
    try:
        receipt = CodexCanaryReceipt(**dict(value))
        verify_canary_receipt(receipt)
    except Exception as exc:
        raise PremiumConstructionProductionError(f"{label} receipt failed verification") from exc
    return receipt


def _signed_adapter(value: Any) -> SignedAdapterReceipt:
    if not isinstance(value, Mapping) or not isinstance(value.get("payload"), Mapping):
        raise PremiumConstructionProductionError("Opus 5 adapter receipt differs")
    try:
        receipt = SignedAdapterReceipt(
            payload=MappingProxyType(dict(value["payload"])),
            payload_blake3=value["payload_blake3"],
            key_id=value["key_id"],
            public_key_base64=value["public_key_base64"],
            public_key_blake3=value["public_key_blake3"],
            algorithm=value["algorithm"],
            signature_base64=value["signature_base64"],
            envelope_blake3=value["envelope_blake3"],
        )
        verify_adapter_receipt(receipt)
    except Exception as exc:
        raise PremiumConstructionProductionError("Opus 5 adapter receipt failed verification") from exc
    return receipt


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    raw = _stable_regular_bytes(
        path,
        label="host signing key",
        maximum_bytes=_MAX_PRIVATE_KEY_BYTES,
        required_mode=0o600,
        require_current_owner=True,
    )
    try:
        key = serialization.load_pem_private_key(raw, password=None)
    except Exception as exc:
        raise PremiumConstructionProductionError("route canary signing key is invalid") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise PremiumConstructionProductionError("route canary signing key is not Ed25519")
    return key


def _trusted_key(
    path: Path,
    key_id: str,
    *,
    expected_trust_store_blake3: str | None = None,
) -> Ed25519PublicKey:
    raw = _stable_regular_bytes(
        path,
        label="host trust store",
        maximum_bytes=1024 * 1024,
    )
    if (
        expected_trust_store_blake3 is not None
        and blake3_bytes(raw) != expected_trust_store_blake3
    ):
        raise PremiumConstructionProductionError(
            "host trust store commitment differs"
        )

    def object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PremiumConstructionProductionError(
                    "host trust store contains duplicate JSON keys"
                )
            result[key] = value
        return result

    try:
        document = json.loads(raw, object_pairs_hook=object_without_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PremiumConstructionProductionError(
            "host trust store is not valid JSON"
        ) from exc
    keys = document.get("keys") if isinstance(document, dict) else None
    if not (
        isinstance(document, dict)
        and set(document)
        == {"schema", "algorithm", "status", "created_at_utc", "keys"}
        and document.get("schema") == "rlevo.med-research-host-trust-store.v1"
        and document.get("algorithm") == "Ed25519"
        and document.get("status") == "active"
        and isinstance(document.get("created_at_utc"), str)
        and bool(document["created_at_utc"])
        and isinstance(keys, Mapping)
        and isinstance(keys.get(key_id), str)
    ):
        raise PremiumConstructionProductionError("host trust store contract differs")
    try:
        raw = base64.b64decode(keys[key_id], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PremiumConstructionProductionError("trusted route canary key differs") from exc
    if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != keys[key_id]:
        raise PremiumConstructionProductionError("trusted route canary key length differs")
    return Ed25519PublicKey.from_public_bytes(raw)


def _load_trusted_host_signing_material(
    *,
    private_key_path: Path,
    key_id: str,
    trust_store_path: Path,
    expected_public_key_blake3: str,
    expected_trust_store_blake3: str,
) -> tuple[Ed25519PrivateKey, AdapterReceiptSigner]:
    """Load one explicitly injected signer and bind it to the active trust entry."""

    if not is_blake3(expected_public_key_blake3) or not is_blake3(
        expected_trust_store_blake3
    ):
        raise PremiumConstructionProductionError(
            "host signing authority commitments differ"
        )
    trusted = _trusted_key(
        trust_store_path,
        key_id,
        expected_trust_store_blake3=expected_trust_store_blake3,
    )
    private = _load_private_key(private_key_path)
    public_bytes = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    if blake3_bytes(public_bytes) != expected_public_key_blake3:
        raise PremiumConstructionProductionError("host signing public key commitment differs")
    trusted_bytes = trusted.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    if trusted_bytes != public_bytes:
        raise PremiumConstructionProductionError(
            "host signing private key differs from trust store"
        )
    signer = AdapterReceiptSigner(private, key_id=key_id)
    if signer.public_key_blake3 != expected_public_key_blake3:
        raise PremiumConstructionProductionError("host signer identity differs")
    return private, signer


def verify_host_signing_authority(
    *,
    private_key_path: Path,
    key_id: str,
    trust_store_path: Path,
    expected_public_key_blake3: str,
    expected_trust_store_blake3: str,
) -> Mapping[str, str]:
    """Fail closed unless one injected private signer matches its trust anchor."""

    _private, signer = _load_trusted_host_signing_material(
        private_key_path=private_key_path,
        key_id=key_id,
        trust_store_path=trust_store_path,
        expected_public_key_blake3=expected_public_key_blake3,
        expected_trust_store_blake3=expected_trust_store_blake3,
    )
    return MappingProxyType(
        {
            "key_id": signer.key_id,
            "public_key_blake3": signer.public_key_blake3,
            "trust_store_blake3": expected_trust_store_blake3,
        }
    )


def sign_exact_gpt56_canary(
    receipt: CodexCanaryReceipt,
    *,
    private_key_path: Path,
    key_id: str,
    trust_store_path: Path,
    expected_public_key_blake3: str,
    expected_trust_store_blake3: str,
) -> Mapping[str, Any]:
    """Sign one already-verified exact Azure GPT-5.6 direct canary."""

    verify_canary_receipt(receipt)
    if not (
        receipt.route_id == "gpt_5_6_sol"
        and receipt.model_id == EXACT_MODELS["gpt_5_6_sol"]
        and receipt.route_status == "direct-pass"
        and receipt.status == "passed"
        and receipt.semantic_exact_ok
    ):
        raise PremiumConstructionProductionError("exact GPT-5.6 route canary did not pass")
    payload = {
        "schema": "eva.premium-construction-route-canary-payload.v1",
        "route_id": receipt.route_id,
        "model_id": receipt.model_id,
        "provider_id": receipt.provider_id,
        "canary_receipt": receipt.to_dict(),
        "canary_receipt_blake3": receipt.receipt_blake3,
        "semantic_attempt_count": 1,
        "semantic_retry_count": 0,
        "credential_values_recorded": False,
        "endpoint_values_recorded": False,
    }
    payload_bytes = canonical_json_bytes(payload)
    private, _signer = _load_trusted_host_signing_material(
        private_key_path=private_key_path,
        key_id=key_id,
        trust_store_path=trust_store_path,
        expected_public_key_blake3=expected_public_key_blake3,
        expected_trust_store_blake3=expected_trust_store_blake3,
    )
    signature = private.sign(
        ROUTE_CANARY_SIGNATURE_DOMAIN + payload_bytes
    )
    core = {
        "schema": "eva.premium-construction-route-canary-envelope.v1",
        "signature_domain": ROUTE_CANARY_SIGNATURE_DOMAIN[:-1].decode("ascii"),
        "payload": payload,
        "payload_blake3": blake3_bytes(payload_bytes),
        "key_id": key_id,
        "algorithm": "Ed25519",
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }
    document = {**core, "document_blake3": blake3_hex(core)}
    verify_exact_gpt56_canary(document, trust_store_path=trust_store_path)
    return MappingProxyType(document)


def verify_exact_gpt56_canary(
    document: Mapping[str, Any],
    *,
    trust_store_path: Path,
) -> CodexCanaryReceipt:
    expected = {
        "schema", "signature_domain", "payload", "payload_blake3", "key_id",
        "algorithm", "signature_base64", "document_blake3",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise PremiumConstructionProductionError("exact GPT-5.6 canary envelope differs")
    payload = document.get("payload")
    if not isinstance(payload, Mapping):
        raise PremiumConstructionProductionError("exact GPT-5.6 canary payload differs")
    payload_bytes = canonical_json_bytes(payload)
    core = {key: document[key] for key in expected if key != "document_blake3"}
    if not (
        document["schema"] == "eva.premium-construction-route-canary-envelope.v1"
        and document["signature_domain"] == ROUTE_CANARY_SIGNATURE_DOMAIN[:-1].decode("ascii")
        and document["algorithm"] == "Ed25519"
        and blake3_bytes(payload_bytes) == document["payload_blake3"]
        and blake3_hex(core) == document["document_blake3"]
    ):
        raise PremiumConstructionProductionError("exact GPT-5.6 canary commitment differs")
    try:
        signature = base64.b64decode(document["signature_base64"], validate=True)
        _trusted_key(trust_store_path, document["key_id"]).verify(
            signature, ROUTE_CANARY_SIGNATURE_DOMAIN + payload_bytes
        )
    except (InvalidSignature, binascii.Error, ValueError, TypeError) as exc:
        raise PremiumConstructionProductionError("exact GPT-5.6 canary signature differs") from exc
    expected_payload = {
        "schema", "route_id", "model_id", "provider_id", "canary_receipt",
        "canary_receipt_blake3", "semantic_attempt_count", "semantic_retry_count",
        "credential_values_recorded", "endpoint_values_recorded",
    }
    if set(payload) != expected_payload:
        raise PremiumConstructionProductionError("exact GPT-5.6 canary payload keys differ")
    receipt = _canary(payload["canary_receipt"], label="exact GPT-5.6")
    if not (
        payload["schema"] == "eva.premium-construction-route-canary-payload.v1"
        and payload["route_id"] == receipt.route_id == "gpt_5_6_sol"
        and payload["model_id"] == receipt.model_id == EXACT_MODELS["gpt_5_6_sol"]
        and payload["provider_id"] == receipt.provider_id == "eva_gpt_5_6_sol"
        and payload["canary_receipt_blake3"] == receipt.receipt_blake3
        and payload["semantic_attempt_count"] == 1
        and payload["semantic_retry_count"] == 0
        and payload["credential_values_recorded"] is False
        and payload["endpoint_values_recorded"] is False
        and receipt.route_status == "direct-pass"
        and receipt.status == "passed"
        and receipt.semantic_exact_ok
    ):
        raise PremiumConstructionProductionError("exact GPT-5.6 canary identity differs")
    return receipt


def detect_exact_construction_routes(
    *, exact_gpt56_model_override: str | None = None
) -> Mapping[str, Any]:
    """Resolve safe route metadata only and report exact missing assignments."""

    missing: list[str] = []
    safe: dict[str, Mapping[str, Any]] = {}
    if exact_gpt56_model_override is not None and exact_gpt56_model_override != EXACT_MODELS["gpt_5_6_sol"]:
        missing.append(
            f"MODEL_GPT_5_6_SOL={EXACT_MODELS['gpt_5_6_sol']}"
        )
    environment = (
        None
        if exact_gpt56_model_override is None
        else {"MODEL_GPT_5_6_SOL": exact_gpt56_model_override}
    )
    try:
        routes = load_codex_provider_routes(
            env_files=(PROJECT_ROOT.parent / ".env", PROJECT_ROOT.parent / "keys.env"),
            registry_path=MODEL_REGISTRY,
            route_ids=REQUIRED_ROUTE_IDS,
            environment=environment,
        )
    except Exception:
        routes = {}
        missing.extend(("one allowlisted provider endpoint variable", "one allowlisted provider key variable"))
    for route_id, expected_model in EXACT_MODELS.items():
        route = routes.get(route_id)
        if route is None:
            missing.append(f"{route_id} route configuration")
            continue
        safe[route_id] = route.safe_metadata
        if route.model_id != expected_model:
            missing.append(f"{route.model_env_name}={expected_model}")
        if route.config.provider_id != f"eva_{route_id}":
            missing.append(f"{route_id} Codex provider identity")
    core = {
        "schema": "eva.premium-construction-route-preflight.v1",
        "status": "ready" if not missing else "blocked",
        "exact_models": dict(EXACT_MODELS),
        "resolved_safe_routes": safe,
        "exact_gpt56_model_source": (
            "ambient_allowlisted_configuration"
            if exact_gpt56_model_override is None
            else "explicit_model_id_only_override"
        ),
        "missing_requirements": sorted(set(missing)),
        "credential_values_recorded": False,
        "endpoint_values_recorded": False,
        "provider_call_count": 0,
    }
    return MappingProxyType({**core, "document_blake3": blake3_hex(core)})


def load_exact_construction_routes(
    *, exact_gpt56_model_override: str | None = None
) -> Mapping[str, CodexProviderRoute]:
    detector = detect_exact_construction_routes(
        exact_gpt56_model_override=exact_gpt56_model_override
    )
    if detector["status"] != "ready":
        raise PremiumConstructionProductionError(
            "exact construction routes are blocked: "
            + ", ".join(detector["missing_requirements"])
        )
    routes = load_codex_provider_routes(
        env_files=(PROJECT_ROOT.parent / ".env", PROJECT_ROOT.parent / "keys.env"),
        registry_path=MODEL_REGISTRY,
        route_ids=REQUIRED_ROUTE_IDS,
        environment=(
            None
            if exact_gpt56_model_override is None
            else {"MODEL_GPT_5_6_SOL": exact_gpt56_model_override}
        ),
    )
    return routes


def _verify_existing_canaries(
    routes: Mapping[str, CodexProviderRoute], *, gpt56_canary_path: Path,
    trust_store_path: Path,
) -> Mapping[str, str]:
    direct = _strict_json(DIRECT_CANARY_PATH, label="direct route canary")
    if _document_commitment(direct, label="direct route canary") != DIRECT_CANARY_DOCUMENT_BLAKE3:
        raise PremiumConstructionProductionError("direct route canary authority differs")
    rows = direct.get("receipts")
    if not isinstance(rows, list):
        raise PremiumConstructionProductionError("direct route canary inventory differs")
    receipts = {receipt.route_id: receipt for receipt in (_canary(row, label="direct route") for row in rows)}
    digests: dict[str, str] = {}
    for route_id in ("gemini_3_1_pro", "opus_4_8"):
        receipt = receipts.get(route_id)
        if not (
            receipt is not None
            and receipt.model_id == routes[route_id].model_id
            and receipt.provider_id == routes[route_id].config.provider_id
            and receipt.safe_config_blake3 == routes[route_id].config.safe_blake3
            and receipt.route_status == "direct-pass"
            and receipt.status == "passed"
        ):
            raise PremiumConstructionProductionError(f"{route_id} canary identity differs")
        digests[route_id] = receipt.receipt_blake3

    opus = _strict_json(OPUS5_CANARY_PATH, label="Opus 5 adapter canary")
    if _document_commitment(opus, label="Opus 5 adapter canary") != OPUS5_CANARY_DOCUMENT_BLAKE3:
        raise PremiumConstructionProductionError("Opus 5 canary authority differs")
    opus_rows = opus.get("receipts")
    adapter_rows = opus.get("adapter_receipts", {}).get("receipts") if isinstance(opus.get("adapter_receipts"), Mapping) else None
    if not isinstance(opus_rows, list) or len(opus_rows) != 1 or not isinstance(adapter_rows, list) or len(adapter_rows) != 1:
        raise PremiumConstructionProductionError("Opus 5 canary inventory differs")
    opus_receipt = _canary(opus_rows[0], label="Opus 5")
    signed = _signed_adapter(adapter_rows[0])
    if not (
        opus_receipt.model_id == routes["opus_5"].model_id
        and signed.payload.get("model_id") == opus_receipt.model_id
        and signed.payload.get("route_id") == "opus_5"
        and signed.payload.get("status") == "passed"
    ):
        raise PremiumConstructionProductionError("Opus 5 adapter canary identity differs")
    digests["opus_5"] = signed.envelope_blake3

    gpt_document = _strict_json(gpt56_canary_path, label="exact GPT-5.6 canary")
    if gpt_document.get("schema") == "eva.codex-responses-canary-receipts.v1":
        if _document_commitment(gpt_document, label="exact GPT-5.6 canary") != EXACT_GPT56_CANARY_DOCUMENT_BLAKE3:
            raise PremiumConstructionProductionError("exact GPT-5.6 canary authority differs")
        gpt_rows = gpt_document.get("receipts")
        if (
            gpt_document.get("configured_routes") != ["gpt_5_6_sol"]
            or gpt_document.get("requested_routes") != ["gpt_5_6_sol"]
            or gpt_document.get("unconfigured_routes") != []
            or gpt_document.get("receipt_count") != 1
            or gpt_document.get("effective_route_status") != {"gpt_5_6_sol": "direct-pass"}
            or not isinstance(gpt_rows, list)
            or len(gpt_rows) != 1
        ):
            raise PremiumConstructionProductionError("exact GPT-5.6 canary inventory differs")
        gpt = _canary(gpt_rows[0], label="exact GPT-5.6")
        if gpt.receipt_blake3 != EXACT_GPT56_CANARY_RECEIPT_BLAKE3:
            raise PremiumConstructionProductionError("exact GPT-5.6 receipt commitment differs")
    else:
        gpt = verify_exact_gpt56_canary(
            gpt_document, trust_store_path=trust_store_path
        )
    if (
        gpt.model_id != routes["gpt_5_6_sol"].model_id
        or gpt.safe_config_blake3 != routes["gpt_5_6_sol"].config.safe_blake3
    ):
        raise PremiumConstructionProductionError("exact GPT-5.6 configured route changed after canary")
    digests["gpt_5_6_sol"] = gpt_document["document_blake3"]
    return MappingProxyType(digests)


def _provider_config(route: CodexProviderRoute) -> Mapping[str, Any]:
    _overrides, _environment, private = route.config.for_subprocess()
    endpoint, _credential = private
    provider = route.config.provider_id
    return {
        "project_doc_max_bytes": CODEX_FIRST_RELEASE_THREAD_CONFIG["project_doc_max_bytes"],
        "web_search": CODEX_FIRST_RELEASE_THREAD_CONFIG["web_search"],
        "features": dict(CODEX_FIRST_RELEASE_THREAD_CONFIG["features"]),
        "model_providers": {
            provider: {
                "name": "EVA Responses Gateway",
                "base_url": endpoint,
                "env_key": route.config.credential_env_name,
                "requires_openai_auth": False,
                "wire_api": "responses",
                "request_max_retries": 0,
                "stream_max_retries": 0,
            }
        },
    }


class ExactConstructionRouteRunner:
    """Inject the verified route config on every one-shot construction turn."""

    def __init__(
        self,
        inner: GatewayAffinePersistentCodexRunner,
        *,
        routes: Mapping[str, CodexProviderRoute],
        opus_gateway: ShardedOpus5AdapterGateway,
        input_policy: _ExactConstructionInputPolicy,
    ) -> None:
        if not isinstance(input_policy, _ExactConstructionInputPolicy):
            raise PremiumConstructionProductionError(
                "construction input policy authority differs"
            )
        self._inner = inner
        self._input_policy = input_policy
        self._by_provider = {
            route.config.provider_id: route for route_id, route in routes.items()
            if route_id != "opus_5"
        }
        self._by_provider[opus_gateway.provider_id] = opus_gateway.planned_route
        self.shard_count = inner.shard_count

    def start(self) -> None:
        self._inner.start()

    def close(self) -> None:
        self._inner.close()

    def _configured(self, options: CodexThreadOptions) -> CodexThreadOptions:
        route = self._by_provider.get(options.provider)
        if route is None or route.model_id != options.model:
            raise PremiumConstructionProductionError("construction turn route identity differs")
        if options.sandbox is not CodexSandbox.READ_ONLY or options.offered_tools:
            raise PremiumConstructionProductionError("construction turn must be read-only with no tools")
        config = dict(options.config)
        config.update(_provider_config(route))
        configured = replace(options, provider=route.config.provider_id, config=config)
        provider = configured.config["model_providers"][route.config.provider_id]
        if provider.get("request_max_retries") != 0 or provider.get("stream_max_retries") != 0:
            raise PremiumConstructionProductionError("construction transport retry policy differs")
        return configured

    def run_once(self, options: CodexThreadOptions, turn_input: Any) -> Any:
        """Run without a private-text capability (legacy structural port)."""

        configured = self._configured(options)
        return self._inner.run_once(configured, turn_input)

    def run_construction_once(self, request: ConstructionTurnRequest) -> Any:
        """Run one exact frozen construction binding under a one-shot capability."""

        if type(request) is not ConstructionTurnRequest:
            raise PremiumConstructionProductionError(
                "exact construction turn request is required"
            )
        # Re-open the frozen dataclass so object-level tampering is rejected
        # before a runtime thread or provider turn exists.
        request.__post_init__()
        configured = self._configured(request.options)
        with self._input_policy.authorize(
            options=configured,
            turn_input=request.turn_input,
            request_blake3=request.request_blake3,
            source_request_blake3=request.source_request_blake3,
            source_phase_request_blake3=request.source_phase_request_blake3,
            source_output_schema_blake3=request.source_output_schema_blake3,
        ):
            return self._inner.run_once(configured, request.turn_input)


def _child_environment(
    routes: Mapping[str, CodexProviderRoute], gateway: ShardedOpus5AdapterGateway
) -> Mapping[str, str]:
    values: dict[str, str] = {}
    for route in routes.values():
        _overrides, environment, _private = route.config.for_subprocess()
        for name, value in environment.items():
            if name in values and values[name] != value:
                raise PremiumConstructionProductionError("construction credential bindings conflict")
            values[name] = value
    for name, value in gateway.child_environment.items():
        if name in values:
            raise PremiumConstructionProductionError("construction adapter credential conflicts")
        values[name] = value
    return MappingProxyType(values)


def _shard_launches(
    *,
    base: PreparedPersistentCodexRuntime,
    codex_bin: Path,
    child_environment: Mapping[str, str],
    isolation_parent: Path,
) -> tuple[Any, Mapping[str, Any]]:
    isolation_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if isolation_parent.is_symlink() or not isolation_parent.is_dir():
        raise PremiumConstructionProductionError("construction Codex isolation root differs")
    codex_args: list[str] = []
    for override in CODEX_FIRST_RELEASE_CONFIG_OVERRIDES:
        codex_args.extend(("--config", override))
    codex_args.extend(("app-server", "--strict-config", "--listen", "stdio://"))
    plans = []
    wrapper = Path(__file__).parents[1] / "deployment" / "codex_child_exec.py"
    for index in range(APP_SERVER_SHARDS):
        root = isolation_parent / f"shard-{index:03d}"
        root.mkdir(mode=0o700, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise PremiumConstructionProductionError("construction Codex shard root differs")
        plans.append(build_sanitized_codex_exec_plan(
            python_bin=Path(sys.executable).resolve(),
            wrapper_script=wrapper.resolve(),
            codex_bin=codex_bin,
            isolation_root=root,
            credential_env_names=tuple(child_environment),
            codex_args=tuple(codex_args),
        ))
    launches = tuple(replace(base.launch_options, launch_args_override=plan.launch_args) for plan in plans)
    core = {
        "schema": "eva.premium-construction-codex-startup.v1",
        "base_launch_blake3": base.startup_metadata["launch_blake3"],
        "app_server_shards": APP_SERVER_SHARDS,
        "isolation_catalog_blake3": blake3_hex([dict(plan.public_metadata()) for plan in plans]),
        "child_env_names": sorted(child_environment),
        "child_env_values_recorded": False,
        "request_max_retries": 0,
        "stream_max_retries": 0,
        "construction_input_policy_blake3": _new_exact_construction_input_policy().policy_blake3,
        "construction_capability_port_blake3": blake3_hex(
            {
                "schema": "eva.premium-construction-capability-port.v1",
                "orchestrator_method": "run_construction_once",
                "accounting_wrapper_preserves_method": True,
                "generic_run_once_mints_capability": False,
            }
        ),
    }
    return launches, MappingProxyType({**core, "startup_blake3": blake3_hex(core)})


def _phase_plan_matches_v3(phase_plan: tuple[Mapping[str, Any], ...]) -> bool:
    frontiers = (1, 1, 2, 2, 2, 2, 3)
    roles = (
        "strong_actor",
        "strong_actor",
        "middle_actor",
        "strong_actor",
        "strong_actor",
        "strong_actor",
        "strong_actor",
    )
    route_ids = (
        "opus_5",
        "gemini_3_1_pro",
        "opus_4_8",
        "opus_5",
        "gpt_5_6_sol",
        "gemini_3_1_pro",
        "opus_5",
    )
    providers = (
        "eva_adapter_opus_5",
        "eva_gemini_3_1_pro",
        "eva_opus_4_8",
        "eva_adapter_opus_5",
        "eva_gpt_5_6_sol",
        "eva_gemini_3_1_pro",
        "eva_adapter_opus_5",
    )
    if len(phase_plan) != len(V3_LANE_ORDER):
        return False
    for row, lane, frontier, role, route_id, provider in zip(
        phase_plan,
        V3_LANE_ORDER,
        frontiers,
        roles,
        route_ids,
        providers,
        strict=True,
    ):
        if not isinstance(row, Mapping) or set(row) != _PHASE_PLAN_KEYS:
            return False
        if not (
            row["lane"] == lane.value
            and type(row["frontier"]) is int
            and row["frontier"] == frontier
            and row["route_id"] == route_id
            and row["model"] == EXACT_MODELS[route_id]
            and row["provider"] == provider
            and row["role"] == role
            and row["workspace_mode"] == "read-only"
            and type(row["offered_tool_count"]) is int
            and row["offered_tool_count"] == 0
            and type(row["semantic_attempt_count"]) is int
            and row["semantic_attempt_count"] == 1
            and type(row["semantic_retry_count"]) is int
            and row["semantic_retry_count"] == 0
            and type(row["request_max_retries"]) is int
            and row["request_max_retries"] == 0
            and type(row["stream_max_retries"]) is int
            and row["stream_max_retries"] == 0
        ):
            return False
    return True


def _phase_plan_matches_v4(phase_plan: tuple[Mapping[str, Any], ...]) -> bool:
    if len(phase_plan) != len(V4_LANE_ORDER):
        return False
    if tuple(row.get("lane") for row in phase_plan) != tuple(
        lane.value for lane in V4_LANE_ORDER
    ):
        return False
    if tuple(row.get("frontier") for row in phase_plan) != (1, 1, 2, 2, 2, 2, 3):
        return False
    expected_routes = (
        "opus_5",
        "gemini_3_1_pro",
        "opus_4_8",
        "opus_5",
        "gpt_5_6_sol",
        "opus_5",
        "opus_5",
    )
    if tuple(row.get("route_id") for row in phase_plan) != expected_routes:
        return False
    if tuple(row.get("provider") for row in phase_plan) != (
        "eva_adapter_opus_5",
        "eva_gemini_3_1_pro",
        "eva_opus_4_8",
        "eva_adapter_opus_5",
        "eva_gpt_5_6_sol",
        "eva_adapter_opus_5",
        "eva_adapter_opus_5",
    ) or tuple(row.get("role") for row in phase_plan) != (
        "strong_actor",
        "strong_actor",
        "middle_actor",
        "strong_actor",
        "strong_actor",
        "strong_actor",
        "strong_actor",
    ):
        return False
    return all(
        isinstance(row, Mapping)
        and set(row) == _PHASE_PLAN_KEYS
        and row.get("model") == EXACT_MODELS[row["route_id"]]
        and row.get("workspace_mode") == "read-only"
        and row.get("offered_tool_count") == 0
        and row.get("semantic_attempt_count") == 1
        and row.get("semantic_retry_count") == 0
        and row.get("request_max_retries") == 0
        and row.get("stream_max_retries") == 0
        for row in phase_plan
    )


@dataclass(frozen=True, slots=True)
class PremiumConstructionProductionRecipe:
    schema: str
    result_schema: str
    profile: int
    selection_blake3: str
    readiness_authority_blake3: str
    queue_blake3: str
    route_catalog_blake3: str
    provider_canary_blake3s: Mapping[str, str]
    codex_startup_blake3: str
    opus_adapter_binding_blake3: str
    opus_adapter_receipt_policy_blake3: str
    opus_upstream_max_output_tokens: int
    gpt56_model_source: str
    author_quorum: Mapping[str, Any]
    critic_quorum: Mapping[str, Any]
    provider_call_count: int
    wave_widths: tuple[int, ...]
    phase_plan: tuple[Mapping[str, Any], ...]
    phase_plan_blake3: str
    state_root: str
    workspace_root: str
    publication_root: str
    recipe_blake3: str

    def core(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "result_schema": self.result_schema,
            "profile": self.profile,
            "selection_blake3": self.selection_blake3,
            "readiness_authority_blake3": self.readiness_authority_blake3,
            "queue_blake3": self.queue_blake3,
            "route_catalog_blake3": self.route_catalog_blake3,
            "provider_canary_blake3s": self.provider_canary_blake3s,
            "codex_startup_blake3": self.codex_startup_blake3,
            "opus_adapter_binding_blake3": self.opus_adapter_binding_blake3,
            "opus_adapter_receipt_policy_blake3": (
                self.opus_adapter_receipt_policy_blake3
            ),
            "opus_upstream_max_output_tokens": self.opus_upstream_max_output_tokens,
            "gpt56_model_source": self.gpt56_model_source,
            "author_quorum": self.author_quorum,
            "critic_quorum": self.critic_quorum,
            "provider_call_count": self.provider_call_count,
            "wave_widths": self.wave_widths,
            "phase_plan": self.phase_plan,
            "phase_plan_blake3": self.phase_plan_blake3,
            "state_root": self.state_root,
            "workspace_root": self.workspace_root,
            "publication_root": self.publication_root,
        }

    def __post_init__(self) -> None:
        try:
            canaries = freeze_json(self.provider_canary_blake3s)
            author_quorum = freeze_json(self.author_quorum)
            critic_quorum = freeze_json(self.critic_quorum)
            phase_plan = tuple(freeze_json(row) for row in self.phase_plan)
            wave_widths = tuple(self.wave_widths)
        except Exception as exc:
            raise PremiumConstructionProductionError(
                "premium construction recipe is not immutable JSON"
            ) from exc
        object.__setattr__(self, "provider_canary_blake3s", canaries)
        object.__setattr__(self, "author_quorum", author_quorum)
        object.__setattr__(self, "critic_quorum", critic_quorum)
        object.__setattr__(self, "phase_plan", phase_plan)
        object.__setattr__(self, "wave_widths", wave_widths)
        if not all(
            isinstance(value, Mapping)
            for value in (canaries, author_quorum, critic_quorum)
        ):
            raise PremiumConstructionProductionError(
                "premium construction recipe object differs"
            )
        if not (
            (
                self.schema == "eva.premium-construction-production-recipe.v3"
                and self.result_schema == PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3
                or self.schema == "eva.premium-construction-production-recipe.v4"
                and self.result_schema == PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4
            )
            and type(self.profile) is int
            and self.profile in {1, 128, 256, 512}
            and all(is_blake3(value) for value in (
                self.selection_blake3, self.readiness_authority_blake3,
                self.queue_blake3, self.route_catalog_blake3,
                self.codex_startup_blake3, self.opus_adapter_binding_blake3,
                self.opus_adapter_receipt_policy_blake3,
                self.phase_plan_blake3,
                self.recipe_blake3,
            ))
            and self.opus_upstream_max_output_tokens
            == CONSTRUCTION_OPUS_MAX_OUTPUT_TOKENS
            and set(self.provider_canary_blake3s) == set(REQUIRED_ROUTE_IDS)
            and all(is_blake3(value) for value in self.provider_canary_blake3s.values())
            and type(self.provider_call_count) is int
            and self.provider_call_count == _V3_PROVIDER_CALL_COUNT
            and self.wave_widths == _V3_WAVE_WIDTHS
            and (
                _phase_plan_matches_v4(self.phase_plan)
                if self.result_schema == PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4
                else _phase_plan_matches_v3(self.phase_plan)
            )
            and self.phase_plan_blake3 == blake3_hex(self.phase_plan)
            and self.gpt56_model_source in {
                "ambient_allowlisted_configuration",
                "explicit_model_id_only_override",
            }
            and canonical_json_bytes(self.author_quorum)
            == canonical_json_bytes(_AUTHOR_QUORUM_RECIPE)
            and canonical_json_bytes(self.critic_quorum)
            == canonical_json_bytes(
                _CRITIC_AUTHORITY_RECIPE_V4
                if self.result_schema == PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4
                else _CRITIC_QUORUM_RECIPE
            )
            and self.recipe_blake3 == blake3_hex(self.core())
        ):
            raise PremiumConstructionProductionError("premium construction recipe differs")

    def to_document(self) -> dict[str, Any]:
        self.__post_init__()
        return {**self.core(), "recipe_blake3": self.recipe_blake3}


class PreparedPremiumConstruction:
    """Provider-free prepared composition with explicit lifecycle."""

    def __init__(
        self,
        *,
        recipe: PremiumConstructionProductionRecipe,
        campaign: PremiumConstructionCampaign,
        runner: ExactConstructionRouteRunner,
        gateway: ShardedOpus5AdapterGateway,
        receipt_store: PremiumConstructionAdapterReceiptStore,
    ) -> None:
        self.recipe = recipe
        self.campaign = campaign
        self.runner = runner
        self.gateway = gateway
        self.receipt_store = receipt_store
        self._state = "new"
        self._lock = RLock()

    def preflight(self, candidate_id: str | None = None) -> PremiumConstructionPreflightReceipt:
        identity = candidate_id or self.campaign.queue.entries[0].candidate_id
        return self.campaign.preflight((identity,))

    def pending_candidate_ids(self, *, limit: int | None = None) -> tuple[str, ...]:
        return self.campaign.pending_candidate_ids(limit=limit)

    def adapter_receipts_document(self) -> Mapping[str, Any]:
        """Expose only redacted, signed Opus adapter evidence."""

        return self.gateway.receipts_document()

    def adapter_receipt_catalog_document(self) -> Mapping[str, Any]:
        """Reopen and verify the crash-durable provider receipt catalog."""

        return self.receipt_store.catalog_document()

    def start(self) -> None:
        with self._lock:
            if self._state != "new":
                raise PremiumConstructionProductionError("construction lifecycle is not fresh")
            self.recipe.__post_init__()
            self._state = "starting"
        try:
            self.gateway.start()
            self.runner.start()
        except BaseException:
            try:
                self.runner.close()
            finally:
                self.gateway.close()
            with self._lock:
                self._state = "closed"
            raise
        with self._lock:
            self._state = "open"

    def run(self) -> PremiumConstructionCampaignReport:
        with self._lock:
            if self._state != "open":
                raise PremiumConstructionProductionError("construction runtime is not open")
        report = self.campaign.run()
        self.receipt_store.write_session_reference(report.session_id)
        return report

    def close(self) -> None:
        with self._lock:
            if self._state == "closed":
                return
            state = self._state
            self._state = "closing"
        failures: list[BaseException] = []
        if state in {"open", "starting"}:
            try:
                self.runner.close()
            except BaseException as exc:
                failures.append(exc)
        try:
            self.gateway.close()
        except BaseException as exc:
            failures.append(exc)
        with self._lock:
            self._state = "closed"
        if failures:
            raise PremiumConstructionProductionError("construction runtime close failed") from failures[0]


def _phase_plan(
    routes: PremiumConstructionRoutes,
    *,
    result_schema: str = PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
) -> tuple[Mapping[str, Any], ...]:
    lanes = (
        (ConstructionLane.OPUS5_DRAFT, 1, "strong_actor"),
        (ConstructionLane.GEMINI_ALTERNATE, 1, "strong_actor"),
        (ConstructionLane.OPUS48_CRITIQUE, 2, "middle_actor"),
        (ConstructionLane.OPUS5_CRITIQUE_BACKUP, 2, "strong_actor"),
        (ConstructionLane.GPT56_COMPARISON, 2, "strong_actor"),
        (ConstructionLane.GEMINI_COMPARISON_BACKUP, 2, "strong_actor"),
        (ConstructionLane.OPUS5_REVISION, 3, "strong_actor"),
    )
    return tuple(MappingProxyType({
        "lane": lane.value,
        "frontier": frontier,
        "route_id": (
            routes.opus5
            if result_schema == PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4
            and lane is ConstructionLane.GEMINI_COMPARISON_BACKUP
            else routes.for_lane(lane)
        ).route_id,
        "model": (
            routes.opus5
            if result_schema == PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4
            and lane is ConstructionLane.GEMINI_COMPARISON_BACKUP
            else routes.for_lane(lane)
        ).model,
        "provider": (
            routes.opus5
            if result_schema == PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4
            and lane is ConstructionLane.GEMINI_COMPARISON_BACKUP
            else routes.for_lane(lane)
        ).provider,
        "role": role,
        "workspace_mode": "read-only",
        "offered_tool_count": 0,
        "semantic_attempt_count": 1,
        "semantic_retry_count": 0,
        "request_max_retries": 0,
        "stream_max_retries": 0,
    }) for lane, frontier, role in lanes)


def _compose_premium_construction(
    *,
    profile: int,
    gpt56_canary_path: Path,
    state_root: Path,
    workspace_root: Path,
    publication_root: Path,
    host_private_key_path: Path,
    host_key_id: str,
    host_trust_store_path: Path,
    expected_host_public_key_blake3: str,
    expected_host_trust_store_blake3: str,
    max_candidates: int | None = None,
    progress: bool = True,
    exact_gpt56_model_override: str | None = None,
    result_schema: str = PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
) -> PreparedPremiumConstruction:
    """Build the real 64-shard premium campaign without starting any process."""

    if profile not in {1, 128, 256, 512}:
        raise PremiumConstructionProductionError("construction profile must be 1/128/256/512")
    routes_by_id = load_exact_construction_routes(
        exact_gpt56_model_override=exact_gpt56_model_override
    )
    canary_blake3s = _verify_existing_canaries(
        routes_by_id,
        gpt56_canary_path=gpt56_canary_path,
        trust_store_path=host_trust_store_path,
    )
    _private, signer = _load_trusted_host_signing_material(
        private_key_path=host_private_key_path,
        key_id=host_key_id,
        trust_store_path=host_trust_store_path,
        expected_public_key_blake3=expected_host_public_key_blake3,
        expected_trust_store_blake3=expected_host_trust_store_blake3,
    )
    gateway = ShardedOpus5AdapterGateway(
        route=routes_by_id["opus_5"],
        app_server_shards=APP_SERVER_SHARDS,
        required_capacity=profile,
        signer=signer,
        canary_receipt_blake3=canary_blake3s["opus_5"],
        upstream_max_tokens_override=CONSTRUCTION_OPUS_MAX_OUTPUT_TOKENS,
    )
    receipt_store = PremiumConstructionAdapterReceiptStore(
        state_root,
        binding_blake3=gateway.binding_blake3,
        route_id="opus_5",
        model_id=routes_by_id["opus_5"].model_id,
        provider_family=routes_by_id["opus_5"].provider_family,
        signing_key_id=signer.key_id,
        signing_public_key_blake3=signer.public_key_blake3,
        effective_upstream_max_tokens=CONSTRUCTION_OPUS_MAX_OUTPUT_TOKENS,
    )
    gateway.install_receipt_sink(receipt_store.commit)
    effective = dict(routes_by_id)
    effective["opus_5"] = gateway.planned_route
    routes = PremiumConstructionRoutes(
        opus5=ConstructionModelRoute("opus_5", effective["opus_5"].model_id, effective["opus_5"].config.provider_id),
        gemini31=ConstructionModelRoute("gemini_3_1_pro", effective["gemini_3_1_pro"].model_id, effective["gemini_3_1_pro"].config.provider_id),
        opus48=ConstructionModelRoute("opus_4_8", effective["opus_4_8"].model_id, effective["opus_4_8"].config.provider_id),
        gpt56=ConstructionModelRoute("gpt_5_6_sol", effective["gpt_5_6_sol"].model_id, effective["gpt_5_6_sol"].config.provider_id),
    )

    selection = CampaignSelectionV2.from_document(_strict_json(SELECTION_V2, label="selection-v2"))
    readiness = load_signed_supervisor_v24_readiness(
        V24_ROOT,
        authority_root=LEGACY_ROOT,
        trust_store_path=host_trust_store_path,
        worker_width=64,
    )
    factory = LegacySelectionV2ConstructionFactory(
        selection=selection,
        readiness=readiness,
        authority_root=LEGACY_ROOT,
        supervisor_root=V24_ROOT,
        trust_store_path=host_trust_store_path,
        legacy_package_src=LEGACY_ROOT / "src",
        workspace_root=workspace_root,
        output_root=publication_root,
    )
    queue = build_premium_construction_queue(selection)
    child_environment = _child_environment(routes_by_id, gateway)
    codex_bin = bundled_codex_path().resolve()
    isolation_parent = (state_root / "codex-isolation").resolve()
    base = prepare_persistent_codex_runtime(
        codex_bin=codex_bin,
        cwd=PROJECT_ROOT,
        child_env=child_environment,
        app_server_shards=APP_SERVER_SHARDS,
        worker_width=profile,
        required_process_soft_nofile=DEFAULT_CHILD_SOFT_NOFILE,
        codex_distribution_version=distribution_version("openai-codex-cli-bin"),
        isolation_root=isolation_parent / "shard-000",
    )
    launches, startup = _shard_launches(
        base=base,
        codex_bin=codex_bin,
        child_environment=child_environment,
        isolation_parent=isolation_parent,
    )
    input_policy = _new_exact_construction_input_policy()
    inner = GatewayAffinePersistentCodexRunner(
        launches,
        opus_gateway=gateway,
        runtime_factory=lambda launch: CodexRuntime(
            OpenAICodexBackend(launch),
            _construction_input_policy=input_policy,
        ),
    )
    runner = ExactConstructionRouteRunner(
        inner,
        routes=routes_by_id,
        opus_gateway=gateway,
        input_policy=input_policy,
    )
    config = PremiumConstructionCampaignConfig(
        worker_width=profile,
        app_server_shards=APP_SERVER_SHARDS,
        max_candidates=max_candidates,
    )
    campaign = PremiumConstructionCampaign(
        queue=queue,
        factory=factory,
        runner=runner,
        routes=routes,
        state_root=state_root,
        config=config,
        progress_callback=(lambda value: print(render_premium_construction_progress(value), file=sys.stderr, flush=True)) if progress else None,
        result_schema=result_schema,
    )
    plan = _phase_plan(routes, result_schema=result_schema)
    recipe_fields = {
        "schema": (
            "eva.premium-construction-production-recipe.v4"
            if result_schema == PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4
            else "eva.premium-construction-production-recipe.v3"
        ),
        "result_schema": result_schema,
        "profile": profile,
        "selection_blake3": selection.selection_blake3,
        "readiness_authority_blake3": readiness.authority_blake3,
        "queue_blake3": queue.queue_blake3,
        "route_catalog_blake3": blake3_hex([dict(route.safe_metadata) for route in routes_by_id.values()]),
        "provider_canary_blake3s": canary_blake3s,
        "codex_startup_blake3": startup["startup_blake3"],
        "opus_adapter_binding_blake3": gateway.binding_blake3,
        "opus_adapter_receipt_policy_blake3": receipt_store.policy[
            "policy_blake3"
        ],
        "opus_upstream_max_output_tokens": CONSTRUCTION_OPUS_MAX_OUTPUT_TOKENS,
        "gpt56_model_source": (
            "ambient_allowlisted_configuration"
            if exact_gpt56_model_override is None
            else "explicit_model_id_only_override"
        ),
        "author_quorum": _AUTHOR_QUORUM_RECIPE,
        "critic_quorum": (
            _CRITIC_AUTHORITY_RECIPE_V4
            if result_schema == PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4
            else _CRITIC_QUORUM_RECIPE
        ),
        "provider_call_count": _V3_PROVIDER_CALL_COUNT,
        "wave_widths": _V3_WAVE_WIDTHS,
        "phase_plan": plan,
        "phase_plan_blake3": blake3_hex(plan),
        "state_root": str(state_root.resolve()),
        "workspace_root": str(workspace_root.resolve()),
        "publication_root": str(publication_root.resolve()),
    }
    recipe = PremiumConstructionProductionRecipe(
        **recipe_fields, recipe_blake3=blake3_hex(recipe_fields)
    )
    return PreparedPremiumConstruction(
        recipe=recipe,
        campaign=campaign,
        runner=runner,
        gateway=gateway,
        receipt_store=receipt_store,
    )


def compose_premium_construction(**kwargs: Any) -> PreparedPremiumConstruction:
    """Compose the immutable v3 recipe (legacy default)."""

    return _compose_premium_construction(
        **kwargs, result_schema=PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3
    )


def compose_premium_construction_v4(**kwargs: Any) -> PreparedPremiumConstruction:
    """Compose prospective Opus-authoritative v4 without touching v3 state."""

    return _compose_premium_construction(
        **kwargs, result_schema=PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V4
    )


__all__ = [
    "EXACT_MODELS",
    "PremiumConstructionProductionError",
    "PremiumConstructionProductionRecipe",
    "PreparedPremiumConstruction",
    "ExactConstructionRouteRunner",
    "compose_premium_construction",
    "compose_premium_construction_v4",
    "detect_exact_construction_routes",
    "load_exact_construction_routes",
    "sign_exact_gpt56_canary",
    "verify_exact_gpt56_canary",
]
