"""Sharded Responses-to-Chat adapters for the production rollout cascade.

The public EvaMed contracts remain authoritative.  This module only selects a
route-affine loopback transport for provider endpoints that implement Chat
Completions but not the full Responses tool surface emitted by Codex.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import secrets
import stat
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Sequence

from eva_agent.codex_providers import (
    AdapterLimits,
    AdapterReceiptSigner,
    CodexProviderRoute,
    ResponsesAdapterGateway,
    SignedAdapterReceipt,
)
from eva_agent.codex_runtime import (
    CodexLaunchOptions,
    CodexRuntime,
    CodexThreadOptions,
    CodexTurnInput,
    CodexTurnReceipt,
    OpenAICodexBackend,
    PersistentCodexRuntimeRunner,
)
from eva_agent.pipeline.digests import (
    blake3_bytes,
    blake3_hex,
    canonical_json_bytes,
    is_blake3,
)

from .campaign import CampaignDeploymentError


ROLLOUT_ADAPTED_ROUTE_IDS = (
    "deepseek_v4_flash",
    "gemini_3_1_pro",
    "opus_5",
    "opus_4_8",
)
ROLLOUT_DIRECT_ROUTE_IDS = ("gpt_5_6_sol",)
ROLLOUT_ROUTE_IDS = (*ROLLOUT_ADAPTED_ROUTE_IDS, *ROLLOUT_DIRECT_ROUTE_IDS)
ROLLOUT_ADAPTER_TOKEN_ENV = "EVA_CODEX_OPUS5_ADAPTER_TOKEN"
ROLLOUT_MODEL_CATALOG_FILENAME = "models.rollout-v1.json"
ROLLOUT_MODEL_INSTRUCTIONS = (
    "You are an EVA medical-research coding agent. Complete the supplied task "
    "from the supplied context and workspace evidence. Use only the offered "
    "candidate-specific MCP tools and their exact schemas. When independent "
    "tool calls are ready, call them in parallel. Never claim evidence that "
    "you did not inspect."
)


def _catalog_model(
    route: CodexProviderRoute,
    *,
    priority: int,
    auto_compact_token_limit: int | None,
) -> dict[str, Any]:
    """Return an explicit Codex model descriptor with the full tool surface."""

    return {
        "slug": route.model_id,
        "display_name": route.route_id,
        "description": "EVA verified rollout route",
        "default_reasoning_level": "low" if route.model_id.split("/")[-1] == "gpt-6-astra" else None,
        "supported_reasoning_levels": (
            [{"effort": "low", "description": "Bounded Astra rollout reasoning"}]
            if route.model_id.split("/")[-1] == "gpt-6-astra" else []
        ),
        "shell_type": "disabled",
        "visibility": "none",
        "supported_in_api": True,
        "priority": priority,
        "additional_speed_tiers": [],
        "service_tiers": [],
        "default_service_tier": None,
        "availability_nux": None,
        "upgrade": None,
        "model_messages": {
            "instructions_template": ROLLOUT_MODEL_INSTRUCTIONS,
            "instructions_variables": None,
            "approvals": None,
            "collaboration_modes": None,
            "auto_review": None,
            "permissions": None,
            "token_budget": None,
        },
        "include_skills_usage_instructions": False,
        "include_plugin_usage_instructions": False,
        "include_apps_usage_instructions": False,
        "supports_reasoning_summary_parameter": False,
        "default_reasoning_summary": "none",
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": None,
        "web_search_tool_type": "text",
        "truncation_policy": {"mode": "tokens", "limit": 10_000},
        "supports_parallel_tool_calls": True,
        "supports_image_detail_original": False,
        "context_window": 131_072,
        "max_context_window": 131_072,
        "auto_compact_token_limit": auto_compact_token_limit,
        "comp_hash": None,
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
        "supports_search_tool": False,
        "use_responses_lite": False,
        "auto_review_model_override": None,
        "model_specialty": None,
        "tool_mode": "direct",
        "multi_agent_version": None,
    }


def rollout_model_catalog_document(
    routes: Mapping[str, CodexProviderRoute],
    *,
    route_order: Sequence[str] | None = None,
    auto_compact_token_limit: int | None = None,
) -> dict[str, Any]:
    """Build the exact static Codex catalog for all five frozen routes."""

    if (
        auto_compact_token_limit is not None
        and (
            isinstance(auto_compact_token_limit, bool)
            or not isinstance(auto_compact_token_limit, int)
            or not 0 < auto_compact_token_limit < 131_072
        )
    ):
        raise CampaignDeploymentError("rollout auto-compact token limit is invalid")

    order = ROLLOUT_ROUTE_IDS if route_order is None else tuple(route_order)
    if not order or len(set(order)) != len(order) or set(routes) != set(order):
        raise CampaignDeploymentError("rollout model catalog route inventory differs")
    ordered: list[CodexProviderRoute] = []
    for route_id in order:
        route = routes.get(route_id)
        if not isinstance(route, CodexProviderRoute) or route.route_id != route_id:
            raise CampaignDeploymentError("rollout model catalog route identity differs")
        ordered.append(route)
    if len({route.model_id for route in ordered}) != len(ordered):
        raise CampaignDeploymentError("rollout model catalog model identity is ambiguous")
    return {
        "models": [
            _catalog_model(
                route,
                priority=index + 1,
                auto_compact_token_limit=auto_compact_token_limit,
            )
            for index, route in enumerate(ordered)
        ]
    }


def _private_directory(path: Path) -> Path:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CampaignDeploymentError("rollout model catalog directory is unavailable") from exc
    if (
        not path.is_absolute()
        or path != resolved
        or stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise CampaignDeploymentError("rollout model catalog directory is not private")
    return resolved


def _read_regular_file(path: Path) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CampaignDeploymentError("rollout model catalog cannot be reopened") from exc
    try:
        info = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks), info


@dataclass(frozen=True, repr=False, slots=True)
class RolloutModelCatalog:
    """Immutable local catalog forcing full Responses multi-tool semantics."""

    path: Path
    payload_blake3: str
    document_blake3: str
    model_ids: tuple[str, ...]
    _payload: bytes

    def __repr__(self) -> str:
        return (
            "RolloutModelCatalog("
            f"path={str(self.path)!r}, payload_blake3={self.payload_blake3!r}, "
            f"document_blake3={self.document_blake3!r}, model_ids={self.model_ids!r})"
        )

    @property
    def config_override(self) -> str:
        return f"model_catalog_json={json.dumps(str(self.path), ensure_ascii=False)}"

    @property
    def safe_metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": "eva.codex-rollout-model-catalog.v1",
                "path": str(self.path),
                "payload_blake3": self.payload_blake3,
                "document_blake3": self.document_blake3,
                "model_ids": list(self.model_ids),
                "model_count": len(self.model_ids),
                "use_responses_lite": False,
                "supports_parallel_tool_calls": True,
                "tool_mode": "direct",
                "shell_type": "disabled",
                "canonical_evamed_schema_bytes_changed": False,
            }
        )

    def verify(self) -> None:
        payload, info = _read_regular_file(self.path)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o400
            or payload != self._payload
            or blake3_bytes(payload) != self.payload_blake3
        ):
            raise CampaignDeploymentError("rollout model catalog bytes differ")
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CampaignDeploymentError("rollout model catalog JSON differs") from exc
        models = document.get("models") if isinstance(document, dict) else None
        if (
            blake3_hex(document) != self.document_blake3
            or not isinstance(models, list)
            or tuple(row.get("slug") for row in models if isinstance(row, dict))
            != self.model_ids
            or any(
                not isinstance(row, dict)
                or row.get("use_responses_lite") is not False
                or row.get("supports_parallel_tool_calls") is not True
                or row.get("tool_mode") != "direct"
                or row.get("shell_type") != "disabled"
                for row in models
            )
        ):
            raise CampaignDeploymentError("rollout model catalog semantics differ")


def materialize_rollout_model_catalog(
    routes: Mapping[str, CodexProviderRoute], *, root: Path,
    route_order: Sequence[str] | None = None,
    auto_compact_token_limit: int | None = None,
) -> RolloutModelCatalog:
    """O_EXCL-publish or exactly reopen one provider-free model catalog."""

    directory = _private_directory(Path(root))
    document = rollout_model_catalog_document(
        routes,
        route_order=route_order,
        auto_compact_token_limit=auto_compact_token_limit,
    )
    payload = canonical_json_bytes(document)
    path = directory / ROLLOUT_MODEL_CATALOG_FILENAME
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o400)
    except FileExistsError:
        existing, _info = _read_regular_file(path)
        if existing != payload:
            raise CampaignDeploymentError("existing rollout model catalog bytes differ")
    except OSError as exc:
        raise CampaignDeploymentError("rollout model catalog cannot be created") from exc
    else:
        try:
            os.fchmod(descriptor, 0o400)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written < 1:
                    raise CampaignDeploymentError("rollout model catalog write failed")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        parent = os.open(directory, parent_flags)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    catalog = RolloutModelCatalog(
        path=path,
        payload_blake3=blake3_bytes(payload),
        document_blake3=blake3_hex(document),
        model_ids=tuple(row["slug"] for row in document["models"]),
        _payload=payload,
    )
    catalog.verify()
    return catalog


class ShardedRolloutAdapterGateway:
    """One multi-route adapter per Codex shard with route-affine reservations."""

    def __init__(
        self,
        *,
        routes: Mapping[str, CodexProviderRoute],
        app_server_shards: int,
        required_capacity: int,
        signer: AdapterReceiptSigner,
        canary_receipt_blake3s: Mapping[str, str],
    ) -> None:
        if set(routes) != set(ROLLOUT_ADAPTED_ROUTE_IDS):
            raise CampaignDeploymentError("rollout adapter route inventory differs")
        ordered = {route_id: routes[route_id] for route_id in ROLLOUT_ADAPTED_ROUTE_IDS}
        expected_families = {
            "deepseek_v4_flash": "deepseek",
            "gemini_3_1_pro": "google",
            "opus_5": "anthropic",
            "opus_4_8": "anthropic",
        }
        if any(
            route.route_id != route_id
            or route.provider_family != expected_families[route_id]
            for route_id, route in ordered.items()
        ):
            raise CampaignDeploymentError("rollout adapter source route differs")
        if set(canary_receipt_blake3s) != set(ordered) or any(
            not is_blake3(value)
            for value in canary_receipt_blake3s.values()
        ):
            raise CampaignDeploymentError("rollout adapter canary inventory differs")
        if type(app_server_shards) is not int or app_server_shards < 1:
            raise CampaignDeploymentError("rollout adapter shard count differs")
        if type(required_capacity) is not int or required_capacity < 1:
            raise CampaignDeploymentError("rollout adapter capacity differs")
        per_shard = math.ceil(required_capacity / app_server_shards)
        if not 1 <= per_shard <= 256:
            raise CampaignDeploymentError("rollout adapter shard capacity exceeds safe bound")
        self._token = secrets.token_urlsafe(32)
        self._gateways = tuple(
            ResponsesAdapterGateway(
                ordered,
                limits=AdapterLimits(max_concurrency=per_shard),
                signer=signer,
                local_bearer_token=self._token,
                local_credential_env_name=ROLLOUT_ADAPTER_TOKEN_ENV,
            )
            for _ in range(app_server_shards)
        )
        planned = dict(self._gateways[0].planned_adapted_routes())
        if any(
            {
                route_id: route.config.safe_blake3
                for route_id, route in gateway.planned_adapted_routes().items()
            }
            != {
                route_id: route.config.safe_blake3
                for route_id, route in planned.items()
            }
            for gateway in self._gateways
        ):
            raise CampaignDeploymentError("rollout adapter shard commitments differ")
        self.planned_routes = MappingProxyType(planned)
        self.shard_count = app_server_shards
        self.per_shard_capacity = per_shard
        self.aggregate_max_concurrency = per_shard * app_server_shards
        self._routes: tuple[Mapping[str, CodexProviderRoute], ...] = ()
        self._loads = [0] * app_server_shards
        self._cursor = 0
        self._lock = RLock()
        self._state = "new"
        core = {
            "schema": "eva.rollout-sharded-adapter-binding.v1",
            "route_ids": list(ROLLOUT_ADAPTED_ROUTE_IDS),
            "routes": [
                {
                    "route_id": route_id,
                    "model_id": ordered[route_id].model_id,
                    "provider_family": ordered[route_id].provider_family,
                    "adapted_provider_id": planned[route_id].config.provider_id,
                    "adapted_config_blake3": planned[route_id].config.safe_blake3,
                    "canary_receipt_blake3": canary_receipt_blake3s[route_id],
                }
                for route_id in ROLLOUT_ADAPTED_ROUTE_IDS
            ],
            "direct_route_ids": list(ROLLOUT_DIRECT_ROUTE_IDS),
            "local_credential_env_name": ROLLOUT_ADAPTER_TOKEN_ENV,
            "shard_count": self.shard_count,
            "per_shard_capacity": self.per_shard_capacity,
            "aggregate_max_concurrency": self.aggregate_max_concurrency,
            "capacity_basis": "three_actor_calls_plus_one_judge_call_per_worker",
            "request_max_retries": 0,
            "stream_max_retries": 0,
            "upstream_request_max_retries": 0,
            "signing_key_id": signer.key_id,
            "signing_public_key_blake3": signer.public_key_blake3,
            "loopback_token_recorded": False,
            "canonical_evamed_schema_bytes_changed": False,
        }
        self.binding_blake3 = blake3_hex(core)
        self._provider_to_route_id = MappingProxyType(
            {
                route.config.provider_id: route_id
                for route_id, route in self.planned_routes.items()
            }
        )

    @property
    def child_environment(self) -> Mapping[str, str]:
        return MappingProxyType(
            {} if not self._token else {ROLLOUT_ADAPTER_TOKEN_ENV: self._token}
        )

    @property
    def provider_ids(self) -> tuple[str, ...]:
        return tuple(self._provider_to_route_id)

    def handles_provider(self, provider_id: str) -> bool:
        return provider_id in self._provider_to_route_id

    def install_receipt_sink(
        self, sink: Callable[[SignedAdapterReceipt], None]
    ) -> None:
        """Install one shared pre-response sink on every unopened shard."""

        if not callable(sink):
            raise CampaignDeploymentError("rollout adapter receipt sink differs")
        with self._lock:
            if self._state != "new":
                raise CampaignDeploymentError(
                    "rollout adapter receipt sink installation is not fresh"
                )
            try:
                for gateway in self._gateways:
                    gateway.install_receipt_sink(sink)
            except Exception as exc:
                self._state = "closed"
                self._token = ""
                raise CampaignDeploymentError(
                    "rollout adapter receipt sink installation failed"
                ) from exc

    def start(self) -> None:
        with self._lock:
            if self._state == "open":
                return
            if self._state != "new":
                raise CampaignDeploymentError("rollout adapter gateway is not restartable")
            self._state = "starting"
        opened: list[ResponsesAdapterGateway] = []
        try:
            for gateway in self._gateways:
                gateway.__enter__()
                opened.append(gateway)
            routes = tuple(gateway.adapted_routes() for gateway in self._gateways)
            expected = {
                route_id: route.config.safe_blake3
                for route_id, route in self.planned_routes.items()
            }
            if any(
                {
                    route_id: route.config.safe_blake3
                    for route_id, route in shard.items()
                }
                != expected
                for shard in routes
            ):
                raise CampaignDeploymentError("live rollout adapter commitment differs")
        except BaseException:
            for gateway in reversed(opened):
                gateway.__exit__(None, None, None)
            with self._lock:
                self._routes = ()
                self._token = ""
                self._state = "closed"
            raise
        with self._lock:
            self._routes = tuple(MappingProxyType(dict(row)) for row in routes)
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
                raise CampaignDeploymentError("rollout adapter gateway state differs")
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
            raise CampaignDeploymentError("rollout adapter gateway close failed") from failures[0]

    def receipts_document(self) -> Mapping[str, Any]:
        shards = tuple(
            {
                "shard_index": index,
                "adapter_receipts": gateway.receipts_document(),
            }
            for index, gateway in enumerate(self._gateways)
        )
        core = {
            "schema": "eva.rollout-sharded-adapter-receipts.v1",
            "binding_blake3": self.binding_blake3,
            "route_ids": list(ROLLOUT_ADAPTED_ROUTE_IDS),
            "shard_count": self.shard_count,
            "receipt_count": sum(
                shard["adapter_receipts"]["receipt_count"] for shard in shards
            ),
            "shards": shards,
            "raw_requests_recorded": False,
            "raw_upstream_outputs_recorded": False,
            "credential_values_recorded": False,
            "endpoint_values_recorded": False,
        }
        return MappingProxyType({**core, "document_blake3": blake3_hex(core)})

    @contextmanager
    def reserve_route(
        self, provider_id: str
    ) -> Iterator[tuple[int, CodexProviderRoute]]:
        try:
            route_id = self._provider_to_route_id[provider_id]
        except KeyError:
            raise CampaignDeploymentError("provider is outside rollout adapters") from None
        with self._lock:
            if self._state != "open" or len(self._routes) != self.shard_count:
                raise CampaignDeploymentError("rollout adapter gateway is not running")
            minimum = min(self._loads)
            selected = -1
            for offset in range(self.shard_count):
                index = (self._cursor + offset) % self.shard_count
                if self._loads[index] == minimum:
                    selected = index
                    break
            if selected < 0:  # pragma: no cover - exhaustive invariant
                raise CampaignDeploymentError("rollout adapter route selection differs")
            if self._loads[selected] >= self.per_shard_capacity:
                raise CampaignDeploymentError("rollout adapter capacity invariant was exceeded")
            self._loads[selected] += 1
            self._cursor = (selected + 1) % self.shard_count
            route = self._routes[selected][route_id]
        try:
            yield selected, route
        finally:
            with self._lock:
                self._loads[selected] -= 1


class MultiGatewayAffinePersistentCodexRunner:
    """Persistent Codex shards affine to a multi-route adapter fleet."""

    def __init__(
        self,
        launches: Sequence[CodexLaunchOptions],
        *,
        adapter_gateway: ShardedRolloutAdapterGateway,
        provider_config_factory: Callable[[CodexThreadOptions, CodexProviderRoute], CodexThreadOptions],
        model_catalog: RolloutModelCatalog,
        runtime_factory: Callable[[CodexLaunchOptions], CodexRuntime] | None = None,
    ) -> None:
        launch_rows = tuple(launches)
        if not launch_rows:
            raise CampaignDeploymentError("Codex runner launch inventory is empty")
        make_runtime = runtime_factory or (
            lambda launch: CodexRuntime(OpenAICodexBackend(launch))
        )
        self._runners = tuple(
            PersistentCodexRuntimeRunner(lambda launch=launch: make_runtime(launch))
            for launch in launch_rows
        )
        if len(self._runners) != adapter_gateway.shard_count:
            raise CampaignDeploymentError("Codex/rollout adapter shard topology differs")
        self._adapters = adapter_gateway
        self._replace_provider_config = provider_config_factory
        self._model_catalog = model_catalog
        self.shard_count = len(self._runners)
        self._loads = [0] * self.shard_count
        self._cursor = 0
        self._lock = RLock()
        self._state = "new"

    def start(self) -> None:
        self._model_catalog.verify()
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
        self._model_catalog.verify()
        with self._lock:
            self._state = "open"

    @contextmanager
    def _reserve_direct(self) -> Iterator[int]:
        with self._lock:
            if self._state != "open":
                raise CampaignDeploymentError("Codex runner is not open")
            minimum = min(self._loads)
            selected = -1
            for offset in range(self.shard_count):
                index = (self._cursor + offset) % self.shard_count
                if self._loads[index] == minimum:
                    selected = index
                    self._loads[index] += 1
                    self._cursor = (index + 1) % self.shard_count
                    break
            if selected < 0:  # pragma: no cover - exhaustive invariant
                raise CampaignDeploymentError("Codex runner selection differs")
        try:
            yield selected
        finally:
            with self._lock:
                self._loads[selected] -= 1

    def run_once(
        self, options: CodexThreadOptions, turn_input: CodexTurnInput
    ) -> CodexTurnReceipt:
        if self._adapters.handles_provider(options.provider):
            with self._adapters.reserve_route(options.provider) as (index, route):
                with self._lock:
                    if self._state != "open":
                        raise CampaignDeploymentError("Codex runner is not open")
                    self._loads[index] += 1
                try:
                    return self._runners[index].run_once(
                        self._replace_provider_config(options, route), turn_input
                    )
                finally:
                    with self._lock:
                        self._loads[index] -= 1
        with self._reserve_direct() as index:
            return self._runners[index].run_once(options, turn_input)

    def run_once_with_timeout(self, options, turn_input, *, timeout_seconds):
        if self._adapters.handles_provider(options.provider):
            with self._adapters.reserve_route(options.provider) as (index, route):
                with self._lock:
                    if self._state != "open":
                        raise CampaignDeploymentError("Codex runner is not open")
                    self._loads[index] += 1
                try:
                    return self._runners[index].run_once_with_timeout(
                        self._replace_provider_config(options, route),
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


__all__ = [
    "MultiGatewayAffinePersistentCodexRunner",
    "ROLLOUT_ADAPTED_ROUTE_IDS",
    "ROLLOUT_ADAPTER_TOKEN_ENV",
    "ROLLOUT_DIRECT_ROUTE_IDS",
    "ROLLOUT_MODEL_CATALOG_FILENAME",
    "ROLLOUT_ROUTE_IDS",
    "RolloutModelCatalog",
    "ShardedRolloutAdapterGateway",
    "materialize_rollout_model_catalog",
    "rollout_model_catalog_document",
]
