"""Concrete, provider-free composition for the medical campaign-v2 runtime.

This module binds the immutable local authorities to the generic deployment
factory.  Calling :func:`compose_campaign_v2` may prepare private local run
directories, but starts no server, Codex process, or model request.  The
returned lifecycle objects start only when
``CodexCampaignDeployment.run_until_idle`` is explicitly called.
"""

from __future__ import annotations

import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import version as distribution_version
import math
import os
from pathlib import Path
import json
import secrets
import stat
import sys
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from codex_cli_bin import bundled_codex_path

from eva_agent.campaign import (
    FrozenCampaignCandidateSourceV2,
    exact_6000_plan,
    load_campaign_selection,
    load_campaign_selection_v2,
)
from eva_agent.codex_pipeline import TurnMCPBridgeFactory, VerifiedActorSkillCatalog
from eva_agent.codex_providers import (
    AdapterLimits,
    AdapterReceiptSigner,
    CodexProviderRoute,
    ResponsesAdapterGateway,
    SignedAdapterReceipt,
    load_codex_provider_routes,
)
from eva_agent.codex_runtime import (
    CodexLaunchOptions,
    CodexRole,
    CodexRuntime,
    CodexSandbox,
    CodexThreadOptions,
    CodexTurnInput,
    CodexTurnReceipt,
    OpenAICodexBackend,
    PersistentCodexRuntimeRunner,
)
from eva_agent.pipeline import Cohort
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, is_blake3
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.sources import (
    LegacyExecutionBindingResolver,
    LegacySupervisorV4Importer,
    load_signed_supervisor_v24_readiness,
)
from eva_agent.sources.legacy_python_source import legacy_python_root

from .campaign import (
    CODEX_FIRST_RELEASE_CONFIG_OVERRIDES,
    CODEX_FIRST_RELEASE_THREAD_CONFIG,
    CampaignConcurrency,
    CampaignDeploymentConfig,
    CampaignDeploymentError,
    CampaignDeploymentPorts,
    CodexProviderTiers,
    PreparedPersistentCodexRuntime,
    ProviderRouteHealth,
    prepare_persistent_codex_runtime,
)
from .codex_child_exec import build_sanitized_codex_exec_plan
from .rollout_adapters import (
    MultiGatewayAffinePersistentCodexRunner,
    ROLLOUT_ADAPTED_ROUTE_IDS,
    ROLLOUT_DIRECT_ROUTE_IDS,
    RolloutModelCatalog,
    ShardedRolloutAdapterGateway,
    materialize_rollout_model_catalog,
)


CODE_ROOT = Path(__file__).resolve().parents[3]


def medical_data_root(environment=None):
    """Bind data explicitly when runtime code lives in a separate checkout."""
    environment = os.environ if environment is None else environment
    selected = environment.get("EVA_MEDRESEARCH_DATA_ROOT")
    if selected is None:
        return CODE_ROOT
    if not isinstance(selected, str) or not Path(selected).is_absolute():
        raise ValueError("medical data root must be an explicit absolute directory")
    root = Path(selected).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("medical data root must be a directory")
    return root


PROJECT_ROOT = medical_data_root()
LEGACY_ROOT = (PROJECT_ROOT.parent / "rlevo-med-research").resolve()
V4_ROOT = LEGACY_ROOT / "runs" / "evamed-campaign-supervisor-v4"
V24_ROOT = LEGACY_ROOT / "runs" / "evamed-campaign-supervisor-v24-attempt1"
MODEL_REGISTRY = LEGACY_ROOT / "config" / "model-registry.20260903-v2.json"
IMAGE_REFS = LEGACY_ROOT / "config" / "image-refs.v1.json"
RUBRIC_SOURCE = PROJECT_ROOT / "rubrics" / "source" / "domain-stage-tables.v1.json"
SELECTION_V1 = PROJECT_ROOT / "runs" / "campaign-selection.v1.json"
SELECTION_V2 = PROJECT_ROOT / "runs" / "campaign-selection.v2.json"
PLUGIN_ROOT = PROJECT_ROOT / "plugins" / "evamed-codex"
LEGACY_SKILL_ROOT = (
    LEGACY_ROOT / "harness" / "source" / "rlevo-Med-RL-data" / "rev-79dd2a31f5f"
)
LOCAL_ADAPTER_TOKEN_ENV = "EVA_CODEX_OPUS5_ADAPTER_TOKEN"
REQUIRED_ROUTE_IDS = (
    "deepseek_v4_flash",
    "gemini_3_1_pro",
    "opus_4_8",
    "gpt_5_6_sol",
    "opus_5",
)
_MAX_PRIVATE_KEY_BYTES = 64 * 1024
_MAX_TRUST_STORE_BYTES = 1024 * 1024


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
    """Read one unchanged regular file without following its final component."""

    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise CampaignDeploymentError(f"{label} is unavailable") from exc
    if not candidate.is_absolute() or candidate != resolved or candidate.is_symlink():
        raise CampaignDeploymentError(f"{label} path is unsafe")
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise CampaignDeploymentError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= maximum_bytes
        or (required_mode is not None and stat.S_IMODE(before.st_mode) != required_mode)
        or (require_current_owner and before.st_uid != os.getuid())
    ):
        raise CampaignDeploymentError(f"{label} topology, ownership, or mode differs")
    try:
        descriptor = _open_absolute_nofollow(candidate)
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
        after_path = candidate.lstat()
    except OSError as exc:
        raise CampaignDeploymentError(f"{label} could not be reopened") from exc
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if (
        len(raw) != before.st_size
        or len(raw) > maximum_bytes
        or identity(before) != identity(after_fd)
        or identity(before) != identity(after_path)
    ):
        raise CampaignDeploymentError(f"{label} changed while read")
    return raw


def _load_host_signer(
    *,
    private_key_path: Path,
    key_id: str,
    trust_store_path: Path,
    expected_public_key_blake3: str,
    expected_trust_store_blake3: str,
) -> AdapterReceiptSigner:
    """Prove an injected Ed25519 signer matches one active trust-store entry."""

    if not is_blake3(expected_public_key_blake3) or not is_blake3(
        expected_trust_store_blake3
    ):
        raise CampaignDeploymentError("host signing authority commitments differ")
    trust_bytes = _stable_regular_bytes(
        trust_store_path,
        label="host trust store",
        maximum_bytes=_MAX_TRUST_STORE_BYTES,
    )
    if blake3_bytes(trust_bytes) != expected_trust_store_blake3:
        raise CampaignDeploymentError("host trust store commitment differs")

    def object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CampaignDeploymentError(
                    "host trust store contains duplicate JSON keys"
                )
            result[key] = value
        return result

    try:
        trust = json.loads(
            trust_bytes,
            object_pairs_hook=object_without_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignDeploymentError("host trust store is not valid JSON") from exc
    keys = trust.get("keys") if isinstance(trust, dict) else None
    if not (
        isinstance(trust, dict)
        and set(trust)
        == {"schema", "algorithm", "status", "created_at_utc", "keys"}
        and trust.get("schema") == "rlevo.med-research-host-trust-store.v1"
        and trust.get("algorithm") == "Ed25519"
        and trust.get("status") == "active"
        and isinstance(trust.get("created_at_utc"), str)
        and bool(trust["created_at_utc"])
        and isinstance(keys, dict)
        and isinstance(keys.get(key_id), str)
    ):
        raise CampaignDeploymentError("host trust store contract or key ID differs")
    try:
        trusted_public = base64.b64decode(keys[key_id], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CampaignDeploymentError("host trust-store public key differs") from exc
    if (
        len(trusted_public) != 32
        or base64.b64encode(trusted_public).decode("ascii") != keys[key_id]
    ):
        raise CampaignDeploymentError("host trust-store public key differs")
    private_bytes = _stable_regular_bytes(
        private_key_path,
        label="host signing key",
        maximum_bytes=_MAX_PRIVATE_KEY_BYTES,
        required_mode=0o600,
        require_current_owner=True,
    )
    try:
        private = serialization.load_pem_private_key(private_bytes, password=None)
    except (TypeError, ValueError) as exc:
        raise CampaignDeploymentError("host adapter signing key is invalid") from exc
    if not isinstance(private, Ed25519PrivateKey):
        raise CampaignDeploymentError("host adapter signing key is not Ed25519")
    actual_public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    if actual_public != trusted_public:
        raise CampaignDeploymentError("host signing private key differs from trust store")
    if blake3_bytes(actual_public) != expected_public_key_blake3:
        raise CampaignDeploymentError("host signing public key commitment differs")
    signer = AdapterReceiptSigner(private, key_id=key_id)
    if signer.public_key_blake3 != expected_public_key_blake3:
        raise CampaignDeploymentError("host adapter signer identity differs")
    return signer


def _prepare_turn_mcp_runtime_root() -> Path:
    """Create or reopen the short private AF_UNIX parent without following links."""

    temporary_parent = Path("/tmp")
    root = temporary_parent / f"eva-mcp-v2-{os.getuid()}"
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    try:
        metadata = root.lstat()
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise CampaignDeploymentError("turn MCP private runtime root is unavailable") from exc
    if (
        resolved != root
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise CampaignDeploymentError(
            "turn MCP private runtime root must be real, uid-owned, and mode 0700"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise CampaignDeploymentError(
            "turn MCP private runtime root cannot be reopened safely"
        ) from exc
    try:
        reopened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        reopened.st_dev != metadata.st_dev
        or reopened.st_ino != metadata.st_ino
        or reopened.st_uid != metadata.st_uid
        or stat.S_IMODE(reopened.st_mode) != 0o700
        or not stat.S_ISDIR(reopened.st_mode)
    ):
        raise CampaignDeploymentError("turn MCP private runtime root changed")
    return resolved


def _provider_config(route: CodexProviderRoute) -> Mapping[str, Any]:
    """Build one SDK config without retaining a credential value in it."""

    _overrides, _environment, private = route.config.for_subprocess()
    endpoint, _credential = private
    provider = route.config.provider_id
    return {
        "project_doc_max_bytes": CODEX_FIRST_RELEASE_THREAD_CONFIG[
            "project_doc_max_bytes"
        ],
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
        }
    }


def _replace_provider_config(
    options: CodexThreadOptions, route: CodexProviderRoute
) -> CodexThreadOptions:
    config = dict(options.config)
    route_config = _provider_config(route)
    config["model_providers"] = route_config["model_providers"]
    return replace(options, config=config, provider=route.config.provider_id)


class ShardedOpus5AdapterGateway:
    """One adapter shard per app-server shard with balanced reservations."""

    def __init__(
        self,
        *,
        route: CodexProviderRoute,
        app_server_shards: int,
        required_capacity: int,
        signer: AdapterReceiptSigner,
        canary_receipt_blake3: str,
        upstream_max_tokens_override: int | None = None,
    ) -> None:
        if route.route_id != "opus_5":
            raise CampaignDeploymentError("Opus adapter source route differs")
        if type(app_server_shards) is not int or app_server_shards < 1:
            raise CampaignDeploymentError("Opus adapter shard count differs")
        per_shard = math.ceil(required_capacity / app_server_shards)
        if not 1 <= per_shard <= 256:
            raise CampaignDeploymentError("Opus adapter shard capacity exceeds safe bound")
        self._token = secrets.token_urlsafe(32)
        self._gateways = tuple(
            ResponsesAdapterGateway(
                {"opus_5": route},
                limits=AdapterLimits(max_concurrency=per_shard),
                signer=signer,
                local_bearer_token=self._token,
                local_credential_env_name=LOCAL_ADAPTER_TOKEN_ENV,
                upstream_max_tokens_override=upstream_max_tokens_override,
            )
            for _ in range(app_server_shards)
        )
        planned = self._gateways[0].planned_adapted_routes()["opus_5"]
        if any(
            gateway.planned_adapted_routes()["opus_5"].config.safe_blake3
            != planned.config.safe_blake3
            for gateway in self._gateways
        ):
            raise CampaignDeploymentError("Opus adapter shard commitments differ")
        self.planned_route = planned
        self.shard_count = app_server_shards
        self.per_shard_capacity = per_shard
        self.aggregate_max_concurrency = per_shard * app_server_shards
        self.upstream_max_tokens_override = upstream_max_tokens_override
        self._routes: tuple[CodexProviderRoute, ...] = ()
        self._loads = [0] * app_server_shards
        self._cursor = 0
        self._lock = RLock()
        self._state = "new"
        core = {
            "schema": "eva.opus5-sharded-adapter-binding.v1",
            "route_id": route.route_id,
            "model_id": route.model_id,
            "adapted_provider_id": planned.config.provider_id,
            "adapted_config_blake3": planned.config.safe_blake3,
            "local_credential_env_name": LOCAL_ADAPTER_TOKEN_ENV,
            "shard_count": self.shard_count,
            "per_shard_capacity": self.per_shard_capacity,
            "aggregate_max_concurrency": self.aggregate_max_concurrency,
            "request_max_retries": 0,
            "stream_max_retries": 0,
            "upstream_request_max_retries": 0,
            "canary_receipt_blake3": canary_receipt_blake3,
            "signing_key_id": signer.key_id,
            "signing_public_key_blake3": signer.public_key_blake3,
            "loopback_token_recorded": False,
        }
        if upstream_max_tokens_override is not None:
            core["upstream_max_tokens_override"] = upstream_max_tokens_override
        self.binding_blake3 = blake3_hex(core)

    @property
    def child_environment(self) -> Mapping[str, str]:
        return MappingProxyType(
            {} if not self._token else {LOCAL_ADAPTER_TOKEN_ENV: self._token}
        )

    @property
    def provider_id(self) -> str:
        return self.planned_route.config.provider_id

    def install_receipt_sink(
        self, sink: Callable[[SignedAdapterReceipt], None]
    ) -> None:
        """Install one shared pre-response sink on every unopened shard."""

        if not callable(sink):
            raise CampaignDeploymentError("Opus adapter receipt sink differs")
        with self._lock:
            if self._state != "new":
                raise CampaignDeploymentError(
                    "Opus adapter receipt sink installation is not fresh"
                )
            try:
                for gateway in self._gateways:
                    gateway.install_receipt_sink(sink)
            except Exception as exc:
                # No shard may start after a partially installed sink set.
                self._state = "closed"
                self._token = ""
                raise CampaignDeploymentError(
                    "Opus adapter receipt sink installation failed"
                ) from exc

    def start(self) -> None:
        with self._lock:
            if self._state == "open":
                return
            if self._state != "new":
                raise CampaignDeploymentError("Opus adapter gateway is not restartable")
            self._state = "starting"
        opened: list[ResponsesAdapterGateway] = []
        try:
            for gateway in self._gateways:
                gateway.__enter__()
                opened.append(gateway)
            routes = tuple(
                gateway.adapted_routes()["opus_5"] for gateway in self._gateways
            )
            if any(
                route.config.safe_blake3 != self.planned_route.config.safe_blake3
                for route in routes
            ):
                raise CampaignDeploymentError("live Opus adapter route commitment differs")
        except BaseException:
            for gateway in reversed(opened):
                gateway.__exit__(None, None, None)
            with self._lock:
                self._routes = ()
                self._token = ""
                self._state = "closed"
            raise
        with self._lock:
            self._routes = routes
            self._state = "open"

    def close(self) -> None:
        with self._lock:
            if self._state == "closed":
                self._token = ""
                return
            if self._state == "new":
                self._state = "closed"
                self._token = ""
                return
            if self._state != "open":
                raise CampaignDeploymentError("Opus adapter gateway state differs")
            self._state = "closing"
        failures: list[BaseException] = []
        for gateway in reversed(self._gateways):
            try:
                gateway.__exit__(None, None, None)
            except BaseException as exc:
                failures.append(exc)
        with self._lock:
            self._routes = ()
            self._token = ""
            self._state = "closed"
        if failures:
            raise CampaignDeploymentError("Opus adapter gateway close failed") from failures[0]

    def receipts_document(self) -> Mapping[str, Any]:
        """Return every redacted signed shard receipt after or during execution."""

        shards = tuple(
            {
                "shard_index": index,
                "adapter_receipts": gateway.receipts_document(),
            }
            for index, gateway in enumerate(self._gateways)
        )
        receipt_count = sum(
            shard["adapter_receipts"]["receipt_count"] for shard in shards
        )
        core = {
            "schema": "eva.opus5-sharded-adapter-receipts.v1",
            "binding_blake3": self.binding_blake3,
            "shard_count": self.shard_count,
            "receipt_count": receipt_count,
            "shards": shards,
            "raw_requests_recorded": False,
            "raw_upstream_outputs_recorded": False,
            "credential_values_recorded": False,
            "endpoint_values_recorded": False,
        }
        return MappingProxyType({**core, "document_blake3": blake3_hex(core)})

    @contextmanager
    def reserve_route(self) -> Iterator[tuple[int, CodexProviderRoute]]:
        with self._lock:
            if self._state != "open" or len(self._routes) != self.shard_count:
                raise CampaignDeploymentError("Opus adapter gateway is not running")
            minimum = min(self._loads)
            selected = -1
            for offset in range(self.shard_count):
                index = (self._cursor + offset) % self.shard_count
                if self._loads[index] == minimum:
                    selected = index
                    break
            if selected < 0:  # pragma: no cover - exhaustive invariant
                raise CampaignDeploymentError("Opus adapter route selection differs")
            if self._loads[selected] >= self.per_shard_capacity:
                raise CampaignDeploymentError(
                    "Opus adapter aggregate capacity invariant was exceeded"
                )
            self._loads[selected] += 1
            self._cursor = (selected + 1) % self.shard_count
            route = self._routes[selected]
        try:
            yield selected, route
        finally:
            with self._lock:
                self._loads[selected] -= 1


class GatewayAffinePersistentCodexRunner:
    """No-retry app-server pool whose Opus shard is gateway-affine."""

    def __init__(
        self,
        launches: tuple[CodexLaunchOptions, ...],
        *,
        opus_gateway: ShardedOpus5AdapterGateway,
        runtime_factory: Callable[[CodexLaunchOptions], CodexRuntime] | None = None,
    ) -> None:
        if not launches:
            raise CampaignDeploymentError("Codex runner launch inventory is empty")
        make_runtime = runtime_factory or (
            lambda launch: CodexRuntime(OpenAICodexBackend(launch))
        )
        self._runners = tuple(
            PersistentCodexRuntimeRunner(
                lambda launch=launch: make_runtime(launch)
            )
            for launch in launches
        )
        self._opus = opus_gateway
        if len(self._runners) != opus_gateway.shard_count:
            raise CampaignDeploymentError("Codex/Opus adapter shard topology differs")
        self.shard_count = len(launches)
        self._loads = [0] * self.shard_count
        self._cursor = 0
        self._lock = RLock()
        self._state = "new"

    def start(self) -> None:
        with self._lock:
            if self._state == "open":
                return
            if self._state != "new":
                raise CampaignDeploymentError("Codex runner is not restartable")
            self._state = "starting"
        failures: list[BaseException] = []
        with ThreadPoolExecutor(max_workers=self.shard_count) as pool:
            futures = tuple(pool.submit(runner.start) for runner in self._runners)
            for future in futures:
                try:
                    future.result()
                except BaseException as exc:
                    failures.append(exc)
        if failures:
            for runner in self._runners:
                try:
                    runner.close()
                except BaseException:
                    pass
            with self._lock:
                self._state = "closed"
            raise CampaignDeploymentError("Codex runner shard startup failed") from failures[0]
        with self._lock:
            self._state = "open"

    @contextmanager
    def _reserve_direct(self) -> Iterator[int]:
        with self._lock:
            if self._state != "open":
                raise CampaignDeploymentError("Codex runner is not open")
            minimum = min(self._loads)
            for offset in range(self.shard_count):
                index = (self._cursor + offset) % self.shard_count
                if self._loads[index] == minimum:
                    self._loads[index] += 1
                    self._cursor = (index + 1) % self.shard_count
                    break
            else:  # pragma: no cover - exhaustive invariant
                raise CampaignDeploymentError("Codex runner selection differs")
        try:
            yield index
        finally:
            with self._lock:
                self._loads[index] -= 1

    def run_once(
        self, options: CodexThreadOptions, turn_input: CodexTurnInput
    ) -> CodexTurnReceipt:
        if options.provider == self._opus.provider_id:
            with self._opus.reserve_route() as (index, route):
                with self._lock:
                    if self._state != "open":
                        raise CampaignDeploymentError("Codex runner is not open")
                    self._loads[index] += 1
                try:
                    return self._runners[index].run_once(
                        _replace_provider_config(options, route), turn_input
                    )
                finally:
                    with self._lock:
                        self._loads[index] -= 1
        with self._reserve_direct() as index:
            return self._runners[index].run_once(options, turn_input)

    def run_once_with_timeout(self, options, turn_input, *, timeout_seconds):
        if options.provider == self._opus.provider_id:
            with self._opus.reserve_route() as (index, route):
                with self._lock:
                    if self._state != "open":
                        raise CampaignDeploymentError("Codex runner is not open")
                    self._loads[index] += 1
                try:
                    return self._runners[index].run_once_with_timeout(
                        _replace_provider_config(options, route),
                        turn_input,
                        timeout_seconds=timeout_seconds,
                    )
                finally:
                    with self._lock:
                        self._loads[index] -= 1
        with self._reserve_direct() as index:
            return self._runners[index].run_once_with_timeout(
                options, turn_input, timeout_seconds=timeout_seconds
            )

    def close(self) -> None:
        with self._lock:
            if self._state == "closed":
                return
            if self._state == "new":
                self._state = "closed"
                return
            self._state = "closing"
        failures: list[BaseException] = []
        with ThreadPoolExecutor(max_workers=self.shard_count) as pool:
            futures = tuple(pool.submit(runner.close) for runner in self._runners)
            for future in futures:
                try:
                    future.result()
                except BaseException as exc:
                    failures.append(exc)
        with self._lock:
            self._state = "closed"
        if failures:
            raise CampaignDeploymentError("Codex runner shard close failed") from failures[0]


def _provider_routes() -> Mapping[str, CodexProviderRoute]:
    routes = load_codex_provider_routes(
        env_files=(PROJECT_ROOT.parent / ".env", PROJECT_ROOT.parent / "keys.env"),
        registry_path=MODEL_REGISTRY,
        route_ids=REQUIRED_ROUTE_IDS,
    )
    if set(routes) != set(REQUIRED_ROUTE_IDS):
        missing = sorted(set(REQUIRED_ROUTE_IDS) - set(routes))
        raise CampaignDeploymentError(f"production provider routes are missing: {missing}")
    return routes


def _child_environment(
    routes: Mapping[str, CodexProviderRoute],
    gateway: ShardedRolloutAdapterGateway,
) -> Mapping[str, str]:
    values: dict[str, str] = {}
    for route_id in ROLLOUT_DIRECT_ROUTE_IDS:
        _overrides, environment, _private = routes[route_id].config.for_subprocess()
        for name, value in environment.items():
            if name in values and values[name] != value:
                raise CampaignDeploymentError("provider credential bindings conflict")
            values[name] = value
    for name, value in gateway.child_environment.items():
        if name in values:
            raise CampaignDeploymentError("adapter token conflicts with direct provider key")
        values[name] = value
    return MappingProxyType(values)


def _shard_launches(
    *,
    base: PreparedPersistentCodexRuntime,
    codex_bin: Path,
    child_environment: Mapping[str, str],
    shard_count: int,
    model_catalog: RolloutModelCatalog,
) -> tuple[tuple[CodexLaunchOptions, ...], Mapping[str, Any]]:
    """Give every persistent app-server its own private Codex state root."""

    isolation_parent = (PROJECT_ROOT / "runs" / "campaign-v2-codex-isolation").resolve()
    if not isolation_parent.exists():
        isolation_parent.mkdir(mode=0o700, parents=True)
    model_catalog.verify()
    codex_args: list[str] = []
    for override in CODEX_FIRST_RELEASE_CONFIG_OVERRIDES:
        codex_args.extend(("--config", override))
    codex_args.extend(("--config", model_catalog.config_override))
    codex_args.extend(("app-server", "--strict-config", "--listen", "stdio://"))
    plans = []
    for index in range(shard_count):
        root = isolation_parent / f"shard-{index:03d}"
        if not root.exists():
            root.mkdir(mode=0o700)
        plans.append(
            build_sanitized_codex_exec_plan(
                python_bin=Path(sys.executable).resolve(),
                wrapper_script=Path(__file__).with_name("codex_child_exec.py").resolve(),
                codex_bin=codex_bin,
                isolation_root=root,
                credential_env_names=tuple(child_environment),
                codex_args=tuple(codex_args),
            )
        )
    launches = tuple(
        replace(base.launch_options, launch_args_override=plan.launch_args)
        for plan in plans
    )
    metadata_core = {
        key: value for key, value in base.startup_metadata.items() if key != "launch_blake3"
    }
    metadata_core.update(
        {
            "app_server_isolation": "one_private_root_per_shard",
            "app_server_isolation_count": len(plans),
            "app_server_isolation_catalog_blake3": blake3_hex(
                [dict(plan.public_metadata()) for plan in plans]
            ),
            "rollout_model_catalog": dict(model_catalog.safe_metadata),
            "rollout_model_catalog_configured_at_app_server_start": True,
            "rollout_effective_config_overrides": [
                *CODEX_FIRST_RELEASE_CONFIG_OVERRIDES,
                model_catalog.config_override,
            ],
            "rollout_effective_config_override_count": (
                len(CODEX_FIRST_RELEASE_CONFIG_OVERRIDES) + 1
            ),
        }
    )
    model_catalog.verify()
    return launches, MappingProxyType(
        {**metadata_core, "launch_blake3": blake3_hex(metadata_core)}
    )


def compose_campaign_v2(
    *,
    concurrency: CampaignConcurrency,
    provider_health: Mapping[str, ProviderRouteHealth],
    host_private_key_path: Path,
    host_key_id: str,
    host_trust_store_path: Path,
    expected_host_public_key_blake3: str,
    expected_host_trust_store_blake3: str,
) -> tuple[CampaignDeploymentConfig, CampaignDeploymentPorts]:
    """Compose the real v2 campaign without starting any I/O lifecycle."""

    if not isinstance(concurrency, CampaignConcurrency):
        raise CampaignDeploymentError("campaign-v2 concurrency differs")
    routes = _provider_routes()
    if set(provider_health) != set(REQUIRED_ROUTE_IDS):
        raise CampaignDeploymentError("production provider health inventory differs")
    for route_id, route in routes.items():
        proof = provider_health[route_id]
        if proof.route_id != route_id or proof.model_id != route.model_id:
            raise CampaignDeploymentError("production provider canary identity differs")
        expected_status = (
            "adapter-pass"
            if route_id in ROLLOUT_ADAPTED_ROUTE_IDS
            else "direct-pass"
        )
        if proof.verified is not True or proof.status != expected_status:
            raise CampaignDeploymentError(
                f"production provider route lacks a verified {expected_status}: {route_id}"
            )

    signer = _load_host_signer(
        private_key_path=host_private_key_path,
        key_id=host_key_id,
        trust_store_path=host_trust_store_path,
        expected_public_key_blake3=expected_host_public_key_blake3,
        expected_trust_store_blake3=expected_host_trust_store_blake3,
    )
    gateway = ShardedRolloutAdapterGateway(
        routes={route_id: routes[route_id] for route_id in ROLLOUT_ADAPTED_ROUTE_IDS},
        app_server_shards=concurrency.app_server_shards,
        required_capacity=4 * concurrency.worker_width,
        signer=signer,
        canary_receipt_blake3s={
            route_id: provider_health[route_id].canary_receipt_blake3
            for route_id in ROLLOUT_ADAPTED_ROUTE_IDS
        },
    )
    effective_routes = dict(routes)
    effective_routes.update(gateway.planned_routes)
    tiers = CodexProviderTiers.from_routes(effective_routes)
    model_catalog = materialize_rollout_model_catalog(
        effective_routes,
        root=(PROJECT_ROOT / "runs" / "campaign-v2-codex-model-catalog").resolve(),
    )

    plan = exact_6000_plan()
    rubrics = load_and_compile_registry(RUBRIC_SOURCE)
    importer = LegacySupervisorV4Importer(V4_ROOT, authority_root=LEGACY_ROOT)
    base = load_campaign_selection(SELECTION_V1, plan=plan, rubrics=rubrics)
    readiness = load_signed_supervisor_v24_readiness(
        V24_ROOT,
        authority_root=LEGACY_ROOT,
        trust_store_path=host_trust_store_path,
        worker_width=min(64, concurrency.worker_width),
    )
    selection = load_campaign_selection_v2(
        SELECTION_V2,
        base_selection=base,
        readiness=readiness,
        plan=plan,
        rubrics=rubrics,
    )
    targets = tuple(pool[0].target for pool in (tiers.weak, tiers.middle, tiers.strong))
    source = FrozenCampaignCandidateSourceV2(
        selection=selection,
        base_selection=base,
        readiness=readiness,
        importer=importer,
        rubrics=rubrics,
        targets=targets,
        plan=plan,
    )

    state_root = PROJECT_ROOT / "runs" / "campaign-v2-runtime-state"
    state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolver = LegacyExecutionBindingResolver(
        authority_root=LEGACY_ROOT,
        supervisor_root=V24_ROOT,
        trust_store_path=host_trust_store_path,
        legacy_python_root=legacy_python_root(LEGACY_ROOT),
        runtime_state_root=state_root,
        host_private_key_path=host_private_key_path,
        host_key_id=host_key_id,
        image_refs_path=IMAGE_REFS,
        worker_width=min(64, concurrency.worker_width),
    )
    skill_runtime_root = PROJECT_ROOT / "runs" / "campaign-v2-skill-materialization"
    skill_runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    skill_runtime_root.chmod(0o700)
    skills = VerifiedActorSkillCatalog(
        manifest_path=PLUGIN_ROOT / "references" / "legacy-skill-manifest.v1.json",
        legacy_source_root=LEGACY_SKILL_ROOT,
        native_stage_skill_path=PLUGIN_ROOT / "skills" / "stage-rollout" / "SKILL.md",
        runtime_root=skill_runtime_root.resolve(),
    )
    mcp_root = _prepare_turn_mcp_runtime_root()
    turn_mcp = TurnMCPBridgeFactory(
        proxy_python=Path(sys.executable).resolve(),
        proxy_script=CODE_ROOT / "src" / "eva_agent" / "codex_pipeline" / "turn_mcp_proxy.py",
        temp_root=mcp_root,
        maximum_parallel_calls=concurrency.maximum_parallel_tools,
    )

    child_environment = _child_environment(routes, gateway)
    prepared_base = prepare_persistent_codex_runtime(
        codex_bin=bundled_codex_path().resolve(),
        cwd=PROJECT_ROOT,
        child_env=child_environment,
        app_server_shards=concurrency.app_server_shards,
        worker_width=concurrency.worker_width,
        required_process_soft_nofile=concurrency.required_process_soft_nofile,
        codex_distribution_version=distribution_version("openai-codex-cli-bin"),
        isolation_root=(
            PROJECT_ROOT
            / "runs"
            / "campaign-v2-codex-isolation"
            / "shard-000"
        ).resolve(),
    )
    launches, startup_metadata = _shard_launches(
        base=prepared_base,
        codex_bin=bundled_codex_path().resolve(),
        child_environment=child_environment,
        shard_count=concurrency.app_server_shards,
        model_catalog=model_catalog,
    )
    runner = MultiGatewayAffinePersistentCodexRunner(
        launches,
        adapter_gateway=gateway,
        provider_config_factory=_replace_provider_config,
        model_catalog=model_catalog,
    )
    prepared = PreparedPersistentCodexRuntime(
        runner=runner,
        launch_options=launches[0],
        startup_metadata=startup_metadata,
    )

    effective_by_provider = {
        route.config.provider_id: route
        for route in effective_routes.values()
    }
    planned_opus = gateway.planned_routes["opus_5"]

    def actor_options(request, cwd, offers, _bridge):
        cohort_roles = {
            Cohort.WEAK: CodexRole.WEAK_ACTOR,
            Cohort.MIDDLE: CodexRole.MIDDLE_ACTOR,
            Cohort.STRONG: CodexRole.STRONG_ACTOR,
        }
        try:
            route = effective_by_provider[request.model.provider]
        except KeyError:
            raise CampaignDeploymentError("actor provider is outside verified routes") from None
        if route.model_id != request.model.model_id:
            raise CampaignDeploymentError("actor provider model identity differs")
        return CodexThreadOptions(
            role=cohort_roles[request.model.cohort],
            model=route.model_id,
            provider=route.config.provider_id,
            cwd=cwd,
            sandbox=CodexSandbox.READ_ONLY,
            config=_provider_config(route),
            offered_tools=offers,
            ephemeral=True,
        )

    def judge_options(request, offers):
        if request.judge_model_id != planned_opus.model_id:
            raise CampaignDeploymentError("judge model differs from verified Opus 5")
        return CodexThreadOptions(
            role=CodexRole.JUDGE,
            model=planned_opus.model_id,
            provider=planned_opus.config.provider_id,
            cwd=str(PROJECT_ROOT),
            sandbox=CodexSandbox.READ_ONLY,
            config=_provider_config(planned_opus),
            offered_tools=offers,
            ephemeral=True,
        )

    config = CampaignDeploymentConfig(
        workspace_root=(PROJECT_ROOT / "runs" / "campaign-v2-workspaces").resolve(),
        artifact_root=(PROJECT_ROOT / "runs" / "campaign-v2-artifacts").resolve(),
        ledger_path=(PROJECT_ROOT / "runs" / "campaign-v2.sqlite3").resolve(),
        receipt_root=(PROJECT_ROOT / "runs" / "campaign-v2-receipts").resolve(),
        admission_bundle_root=(PROJECT_ROOT / "runs" / "campaign-v2-admission").resolve(),
        private_key_path=Path(host_private_key_path),
        trust_store_path=Path(host_trust_store_path),
        signing_key_id=host_key_id,
        concurrency=concurrency,
    )
    return config, CampaignDeploymentPorts(
        plan=plan,
        candidate_source=source,
        rubrics=rubrics,
        provider_tiers=tiers,
        provider_health=provider_health,
        runtime=prepared,
        actor_options_factory=actor_options,
        judge_options_factory=judge_options,
        turn_mcp_factory=turn_mcp,
        execution_binding_resolver=resolver,
        actor_skills_factory=skills,
        opus5_adapter_gateway=gateway,
    )


__all__ = [
    "GatewayAffinePersistentCodexRunner",
    "ShardedOpus5AdapterGateway",
    "compose_campaign_v2",
]
