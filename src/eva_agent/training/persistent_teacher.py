"""High-throughput, one-process Codex teacher rollout worker.

The worker keeps campaign verification resources, one multi-route adapter, and
a sharded Codex app-server runtime alive for an entire SQLite batch.  Each task
still opens a fresh Codex thread and a private filesystem workspace; no task is
retried or resumed after crossing its durable claim boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Lock, RLock
from types import MappingProxyType
from typing import Any

from codex_cli_bin import bundled_codex_path

from eva_agent.codex_pipeline import CodexRolloutAdapter
from eva_agent.codex_providers import (
    AdapterLimits,
    CodexProviderRoute,
    ResponsesAdapterGateway,
)
from eva_agent.codex_runtime import (
    CodexRole,
    CodexRuntime,
    CodexSandbox,
    CodexThreadOptions,
    OpenAICodexBackend,
    ShardedPersistentCodexRuntimeRunner,
)
from eva_agent.deployment.campaign import (
    CODEX_FIRST_RELEASE_CONFIG_OVERRIDES,
    CODEX_FIRST_RELEASE_THREAD_CONFIG,
)
from eva_agent.deployment.rollout_adapters import materialize_rollout_model_catalog
from eva_agent.pipeline import Cohort, ModelTarget, RandomUUIDFactory

from .teacher_batch import (
    TEACHER_ROUTES,
    TeacherBatchError,
    TeacherTask,
    persist_teacher_adapter_receipt,
)
from .teacher_launch import isolated_teacher_launch_options
from .teacher_worker import (
    CampaignV2TeacherContextPool,
    execute_single_rollout,
    teacher_actor_developer_instructions,
)


_ADAPTED_PROVIDER_FAMILIES = frozenset(
    {"anthropic", "deepseek", "google", "qwen"}
)
TEACHER_ADAPTER_UPSTREAM_MAX_TOKENS = 32_768


def _isolated_shard_runtime_factory(
    *,
    codex_bin: Path,
    launch_cwd: Path,
    isolation_parent: Path,
    config_overrides: Sequence[str],
    child_env: Mapping[str, str],
    shard_count: int,
):
    """Build each app-server with a distinct Codex state database root."""

    lock = Lock()
    next_index = 0

    def factory() -> CodexRuntime:
        nonlocal next_index
        with lock:
            index = next_index
            next_index += 1
        if index >= shard_count:
            raise TeacherBatchError("teacher Codex shard factory was over-consumed")
        launch = isolated_teacher_launch_options(
            codex_bin=codex_bin,
            cwd=launch_cwd,
            isolation_root=isolation_parent / f"shard-{index:03d}",
            config_overrides=config_overrides,
            child_env=child_env,
        )
        return CodexRuntime(OpenAICodexBackend(launch))

    return factory


def _thread_config(route: CodexProviderRoute) -> dict[str, Any]:
    _overrides, _child, private = route.config.for_subprocess()
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
        },
    }


class TeacherRecordIndex:
    """Immutable scheduled-row index; the 6,000-record corpus is not rescanned."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        records: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            sandbox_id = row.get("sandbox_id")
            candidate_id = row.get("candidate_id")
            if (
                not isinstance(sandbox_id, str)
                or not sandbox_id
                or not isinstance(candidate_id, str)
                or not candidate_id
                or sandbox_id in records
            ):
                raise TeacherBatchError("scheduled teacher record identity differs")
            records[sandbox_id] = MappingProxyType(dict(row))
        if not records:
            raise TeacherBatchError("persistent teacher record index is empty")
        self._records = MappingProxyType(records)

    @property
    def count(self) -> int:
        return len(self._records)

    def for_task(self, task: TeacherTask) -> Mapping[str, Any]:
        if (
            task.route_id not in TEACHER_ROUTES
            or task.task_id != f"{task.sandbox_id}--{task.route_id}"
        ):
            raise TeacherBatchError("persistent teacher task identity differs")
        try:
            record = self._records[task.sandbox_id]
        except KeyError:
            raise TeacherBatchError(
                "persistent teacher task is absent from the scheduled index"
            ) from None
        if (
            record.get("candidate_id") != task.candidate_id
            or record.get("stage") != task.stage
            or record.get("domain") != task.domain
        ):
            raise TeacherBatchError("persistent teacher task binding differs")
        return record


class PersistentCodexTeacherPool:
    """Route-shared persistent app-server pool with candidate isolation."""

    def __init__(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        routes: Mapping[str, CodexProviderRoute],
        output_root: str | Path,
        worker_width: int,
        app_server_shards: int,
        context_pool: CampaignV2TeacherContextPool | None = None,
        codex_bin: str | Path | None = None,
        launch_cwd: str | Path | None = None,
        upstream_max_tokens: int = TEACHER_ADAPTER_UPSTREAM_MAX_TOKENS,
        upstream_transport: Any = None,
    ) -> None:
        if (
            not routes
            or any(route_id not in TEACHER_ROUTES for route_id in routes)
            or any(route.route_id != route_id for route_id, route in routes.items())
        ):
            raise TeacherBatchError("persistent teacher route inventory differs")
        if (
            type(worker_width) is not int
            or not 1 <= worker_width <= 256
            or type(app_server_shards) is not int
            or not 1 <= app_server_shards <= worker_width
        ):
            raise TeacherBatchError("persistent teacher concurrency differs")
        if type(upstream_max_tokens) is not int or not 1 <= upstream_max_tokens <= TEACHER_ADAPTER_UPSTREAM_MAX_TOKENS:
            raise TeacherBatchError("persistent teacher output token budget differs")
        self._records = TeacherRecordIndex(rows)
        self._source_routes = MappingProxyType(dict(routes))
        self._route_order = tuple(
            route_id for route_id in TEACHER_ROUTES if route_id in routes
        )
        self._output_root = Path(output_root).resolve()
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._worker_width = worker_width
        self._shard_count = app_server_shards
        self._contexts = context_pool or CampaignV2TeacherContextPool()
        self._codex_bin = Path(codex_bin or bundled_codex_path()).resolve(
            strict=True
        )
        self._launch_cwd = Path(launch_cwd or Path.cwd()).resolve(strict=True)
        adapted = {
            route_id: route
            for route_id, route in self._source_routes.items()
            if route.provider_family in _ADAPTED_PROVIDER_FAMILIES
        }
        self._gateway = (
            ResponsesAdapterGateway(
                adapted,
                limits=AdapterLimits(max_concurrency=worker_width),
                local_credential_env_name="EVA_CODEX_OPUS5_ADAPTER_TOKEN",
                upstream_max_tokens_override=upstream_max_tokens,
                upstream_transport=upstream_transport,
                receipt_sink=lambda receipt: persist_teacher_adapter_receipt(
                    self._output_root, receipt
                ),
            )
            if adapted
            else None
        )
        self._runner: ShardedPersistentCodexRuntimeRunner | None = None
        self._live_routes: Mapping[str, CodexProviderRoute] = MappingProxyType({})
        self._state = "new"
        self._lock = RLock()

    def __enter__(self) -> "PersistentCodexTeacherPool":
        self.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    @property
    def safe_metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": "eva.codex-teacher-persistent-pool.v1",
                "record_count": self._records.count,
                "route_ids": list(self._route_order),
                "route_count": len(self._route_order),
                "worker_width": self._worker_width,
                "app_server_shards": self._shard_count,
                "campaign_resource_opens": 1,
                "model_catalog_materializations": 1,
                "app_server_opens": self._shard_count,
                "fresh_thread_per_task": True,
                "workspace_per_task": True,
                "retry_count": 0,
                "provider_calls_during_construction": 0,
                "isolated_codex_user_config": True,
                "durable_adapter_receipts": self._gateway is not None,
            }
        )

    def _child_environment(
        self, routes: Mapping[str, CodexProviderRoute]
    ) -> dict[str, str]:
        environment: dict[str, str] = {}
        for route in routes.values():
            _overrides, child, _private = route.config.for_subprocess()
            for name, value in child.items():
                existing = environment.get(name)
                if existing is not None and existing != value:
                    raise TeacherBatchError(
                        "persistent teacher child credential binding differs"
                    )
                environment[name] = value
        return environment

    def start(self) -> None:
        with self._lock:
            if self._state == "open":
                return
            if self._state != "new":
                raise TeacherBatchError("persistent teacher pool is not restartable")
            self._state = "starting"
        gateway_open = False
        runner: ShardedPersistentCodexRuntimeRunner | None = None
        try:
            live = dict(self._source_routes)
            if self._gateway is not None:
                self._gateway.__enter__()
                gateway_open = True
                live.update(self._gateway.adapted_routes())
            ordered = {route_id: live[route_id] for route_id in self._route_order}
            catalog = materialize_rollout_model_catalog(
                ordered,
                root=(self._output_root / "model-catalogs/persistent").resolve(),
                route_order=self._route_order,
            )
            runner = ShardedPersistentCodexRuntimeRunner(
                _isolated_shard_runtime_factory(
                    codex_bin=self._codex_bin,
                    launch_cwd=self._launch_cwd,
                    isolation_parent=(
                        self._output_root
                        / "codex-runtime-isolation"
                        / "persistent"
                    ).resolve(),
                    config_overrides=(
                        *CODEX_FIRST_RELEASE_CONFIG_OVERRIDES,
                        catalog.config_override,
                    ),
                    child_env=self._child_environment(ordered),
                    shard_count=self._shard_count,
                ),
                shard_count=self._shard_count,
            )
            runner.start()
        except BaseException:
            if runner is not None:
                runner.close()
            if gateway_open and self._gateway is not None:
                self._gateway.__exit__(None, None, None)
            with self._lock:
                self._state = "closed"
            raise
        with self._lock:
            self._runner = runner
            self._live_routes = MappingProxyType(ordered)
            self._state = "open"

    @staticmethod
    def _options_factory(route: CodexProviderRoute, context: Any):
        def options(request, cwd, offers, _bridge):
            return CodexThreadOptions(
                role=CodexRole.STRONG_ACTOR,
                model=request.model.model_id,
                provider=request.model.provider,
                cwd=cwd,
                sandbox=CodexSandbox.READ_ONLY,
                config=_thread_config(route),
                offered_tools=offers,
                ephemeral=True,
                developer_instructions=teacher_actor_developer_instructions(
                    context
                ),
            )

        return options

    def run(self, task: TeacherTask) -> Mapping[str, Any]:
        with self._lock:
            if self._state != "open" or self._runner is None:
                raise TeacherBatchError("persistent teacher pool is not open")
            runner = self._runner
            try:
                route = self._live_routes[task.route_id]
            except KeyError:
                raise TeacherBatchError(
                    "persistent teacher task route is unavailable"
                ) from None
        record = self._records.for_task(task)
        context, _binding = self._contexts.load(record)
        rollout = CodexRolloutAdapter(
            options_factory=self._options_factory(route, context),
            runner=runner,
            id_factory=RandomUUIDFactory(),
            turn_mcp_factory=context.turn_mcp_factory,
            skills_factory=context.skills_factory,
        )
        target = ModelTarget(
            Cohort.STRONG, route.model_id, route.config.provider_id
        )
        return execute_single_rollout(
            record=record,
            context=context,
            provider=rollout,
            target=target,
            workspace_root=self._output_root / "workspaces" / task.task_id,
            task_id=task.task_id,
            route_id=task.route_id,
        )

    def close(self) -> None:
        with self._lock:
            if self._state == "closed":
                return
            if self._state == "new":
                self._state = "closed"
                return
            if self._state != "open":
                raise TeacherBatchError("persistent teacher pool close state differs")
            self._state = "closing"
            runner = self._runner
        failures: list[BaseException] = []
        if runner is not None:
            try:
                runner.close()
            except BaseException as exc:
                failures.append(exc)
        if self._gateway is not None:
            try:
                self._gateway.__exit__(None, None, None)
            except BaseException as exc:
                failures.append(exc)
        with self._lock:
            self._runner = None
            self._live_routes = MappingProxyType({})
            self._state = "closed"
        if failures:
            raise TeacherBatchError("persistent teacher pool close failed") from failures[0]


__all__ = [
    "PersistentCodexTeacherPool",
    "TEACHER_ADAPTER_UPSTREAM_MAX_TOKENS",
    "TeacherRecordIndex",
]
