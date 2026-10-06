"""Native ChatGPT-authenticated Astra teachers over unchanged EvaMed MCP tools.

Only the login transport differs from the existing teacher pool. Auth copies
live in temporary private directories, and the ordinary rollout adapter owns
every observable tool, skill, workspace, and reasoning-redaction commitment.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from threading import Lock
from typing import Any, Iterator

from eva_agent.codex_pipeline import CodexRolloutAdapter
from eva_agent.codex_providers import ROUTE_DEFINITIONS
from eva_agent.codex_runtime import (
    CodexLaunchOptions, CodexRole, CodexRuntime, CodexSandbox,
    CodexThreadOptions, OpenAICodexBackend, ShardedPersistentCodexRuntimeRunner,
)
from eva_agent.deployment.campaign import (
    CODEX_FIRST_RELEASE_CONFIG_OVERRIDES, CODEX_FIRST_RELEASE_THREAD_CONFIG,
)
from eva_agent.deployment.rollout_adapters import _catalog_model
from eva_agent.pipeline import Cohort, ModelTarget, RandomUUIDFactory, Stage
from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes

from .persistent_teacher import TeacherRecordIndex
from .teacher_focus import focus_only_teacher_instructions
from .teacher_batch import TeacherBatchError, TeacherTask, _write_new_or_verify
from .teacher_worker import (
    CampaignV2TeacherContextPool, TeacherCandidateContext,
    execute_single_rollout,
)


NATIVE_ASTRA_MODEL = "gpt-6-astra"
NATIVE_ASTRA_ROUTE = "gpt_6_astra"
NATIVE_ASTRA_PROVIDER = "eva_native_astra"


@dataclass(frozen=True)
class NativeAstraConfigAuthority:
    """Public identity binding; deliberately contains no SDK credential material."""

    model_id: str = NATIVE_ASTRA_MODEL
    provider_id: str = NATIVE_ASTRA_PROVIDER


@dataclass(frozen=True)
class NativeAstraRouteAuthority:
    route_id: str = NATIVE_ASTRA_ROUTE
    model_id: str = NATIVE_ASTRA_MODEL
    model_env_name: str = "MODEL_GPT_6_ASTRA"
    registry_role_id: None = None
    provider_family: str = "openai"
    config: NativeAstraConfigAuthority = NativeAstraConfigAuthority()


def native_astra_route_authorities() -> Mapping[str, NativeAstraRouteAuthority]:
    """Use the registry's exact route identity with native login provenance."""

    authority = NativeAstraRouteAuthority()
    definition = ROUTE_DEFINITIONS[NATIVE_ASTRA_ROUTE]
    if any(getattr(authority, field) != getattr(definition, field) for field in (
        "route_id", "model_env_name", "registry_role_id", "provider_family"
    )):
        raise TeacherBatchError("native Astra registry identity differs")
    return {NATIVE_ASTRA_ROUTE: authority}


def native_astra_provider_config() -> dict[str, Any]:
    return {
        "name": "Native OpenAI ChatGPT auth", "wire_api": "responses",
        "requires_openai_auth": True, "request_max_retries": 0,
        "stream_max_retries": 0,
    }


def native_astra_model_catalog() -> Mapping[str, Any]:
    authority = native_astra_route_authorities()[NATIVE_ASTRA_ROUTE]
    model = _catalog_model(authority, priority=1, auto_compact_token_limit=None)
    model["default_reasoning_level"] = "low"
    model["supported_reasoning_levels"] = [
        {"effort": "low", "description": "Bounded native Astra medical teacher"}
    ]
    return {"models": [model]}


def _read_native_auth(path: Path) -> bytes:
    """Read an existing login without exposing tokens in errors or representations."""

    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise TeacherBatchError("native Codex auth file ownership differs")
            payload = stream.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            raise TeacherBatchError("native Codex auth file size differs")
        auth = json.loads(payload)
        tokens = auth.get("tokens") or {}
        if auth.get("auth_mode") != "chatgpt" or not tokens.get("access_token"):
            raise TeacherBatchError("existing native ChatGPT login is unavailable")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        if isinstance(exc, TeacherBatchError):
            raise
        raise TeacherBatchError("native Codex auth file is unavailable") from None
    return payload


@contextmanager
def private_native_auth_copy(auth_path: str | Path) -> Iterator[Path]:
    """Make a private runtime root outside published artifacts; remove it on exit."""

    payload = _read_native_auth(Path(auth_path))
    # This Linux native launcher must never honor an artifact-local TMPDIR.
    with tempfile.TemporaryDirectory(prefix="eva-native-astra-auth-", dir="/tmp") as directory:
        root = Path(directory).resolve()
        for child in ("codex", "tmp", "cache", "config", "data"):
            (root / child).mkdir(mode=0o700)
        destination = root / "codex" / "auth.json"
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        yield root


def native_astra_launch_options(
    *, codex_bin: Path, script_path: Path, isolation_root: Path,
    cwd: Path, catalog_path: Path,
) -> CodexLaunchOptions:
    provider = native_astra_provider_config()
    overrides = (
        *CODEX_FIRST_RELEASE_CONFIG_OVERRIDES,
        f'model_catalog_json={json.dumps(str(catalog_path))}',
        f'model_provider="{NATIVE_ASTRA_PROVIDER}"',
        'model_reasoning_effort="low"',
        'model_reasoning_summary="none"',
        'cli_auth_credentials_store="file"',
        'check_for_update_on_startup=false',
        'history.persistence="none"',
        *(f"model_providers.{NATIVE_ASTRA_PROVIDER}.{key}={json.dumps(value)}" for key, value in provider.items()),
    )
    args = [str(Path(sys.executable).resolve()), "-I", "-S", str(script_path),
            "--native-child", str(codex_bin), str(isolation_root)]
    for value in overrides:
        args.extend(("--config", value))
    args.extend(("app-server", "--strict-config", "--listen", "stdio://"))
    return CodexLaunchOptions(launch_args_override=tuple(args), cwd=str(cwd), env={})


def native_astra_thread_options(context: TeacherCandidateContext):
    def options(request, cwd, offers, _bridge):
        return CodexThreadOptions(
            role=CodexRole.STRONG_ACTOR, model=request.model.model_id,
            provider=request.model.provider, cwd=cwd, sandbox=CodexSandbox.READ_ONLY,
            config={
                "project_doc_max_bytes": 0, "web_search": "disabled",
                "features": dict(CODEX_FIRST_RELEASE_THREAD_CONFIG["features"]),
                "model_reasoning_effort": "low", "model_reasoning_summary": "none",
                "model_providers": {NATIVE_ASTRA_PROVIDER: native_astra_provider_config()},
            },
            offered_tools=offers, ephemeral=True,
            developer_instructions=focus_only_teacher_instructions(context),
        )
    return options


class NativeAstraTeacherPool:
    """Bounded native app-server shards; one fresh thread and workspace per task."""

    def __init__(
        self, *, rows: Sequence[Mapping[str, Any]], output_root: Path,
        workers: int = 4, stage: Stage = Stage.S1, auth_path: Path | None = None,
        codex_bin: Path | None = None, context_pool: Any = None,
    ) -> None:
        if not 1 <= workers <= 32 or not 1 <= len(rows) <= 64:
            raise TeacherBatchError("native Astra batch must have at most 64 tasks and 32 workers")
        if stage not in {Stage.S1, Stage.S2, Stage.S3} or any(row.get("stage") != stage.value for row in rows):
            raise TeacherBatchError("native Astra rollout stage binding differs")
        self._records = TeacherRecordIndex(rows)
        self._output_root = Path(output_root).resolve()
        self._workers = workers
        self._auth_path = auth_path or Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
        selected = codex_bin or shutil.which("codex")
        if selected is None:
            raise TeacherBatchError("native Codex executable is unavailable")
        self._codex_bin = Path(selected).resolve(strict=True)
        self._contexts = context_pool
        self._runner = None
        self._auth_contexts: list[Any] = []
        self._lock = Lock()
        self._state = "new"

    def __enter__(self) -> "NativeAstraTeacherPool":
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def start(self) -> None:
        if self._state != "new":
            raise TeacherBatchError("native Astra pool is not restartable")
        self._state = "starting"
        if self._contexts is None:
            self._contexts = CampaignV2TeacherContextPool()
        root = Path(__file__).resolve().parents[3]
        catalog_path = self._output_root / "model-catalogs" / "native-astra.json"
        _write_new_or_verify(catalog_path, canonical_json_bytes(native_astra_model_catalog()))

        def factory():
            manager = private_native_auth_copy(self._auth_path)
            isolation = manager.__enter__()
            with self._lock:
                self._auth_contexts.append(manager)
            launch = native_astra_launch_options(
                codex_bin=self._codex_bin,
                script_path=root / "scripts" / "run_native_astra_teacher_v1.py",
                isolation_root=isolation, cwd=root, catalog_path=catalog_path,
            )
            return CodexRuntime(OpenAICodexBackend(launch))

        self._runner = ShardedPersistentCodexRuntimeRunner(factory, shard_count=self._workers)
        try:
            self._runner.start()
        except BaseException:
            self.close()
            raise
        self._state = "open"

    def run(self, task: TeacherTask) -> Mapping[str, Any]:
        if self._state != "open" or self._runner is None:
            raise TeacherBatchError("native Astra pool is not open")
        if task.route_id != NATIVE_ASTRA_ROUTE:
            raise TeacherBatchError("native Astra task route differs")
        record = self._records.for_task(task)
        context, _binding = self._contexts.load(record)
        rollout = CodexRolloutAdapter(
            options_factory=native_astra_thread_options(context), runner=self._runner,
            id_factory=RandomUUIDFactory(), turn_mcp_factory=context.turn_mcp_factory,
            skills_factory=context.skills_factory,
        )
        receipt = execute_single_rollout(
            record=record, context=context, provider=rollout,
            target=ModelTarget(Cohort.STRONG, NATIVE_ASTRA_MODEL, NATIVE_ASTRA_PROVIDER),
            workspace_root=self._output_root / "workspaces" / task.task_id,
            task_id=task.task_id, route_id=task.route_id,
        )
        sidecar = {
            "schema": "eva.native-astra-teacher-provenance.v1", "task_id": task.task_id,
            "requested_model": NATIVE_ASTRA_MODEL, "returned_model": None,
            "returned_model_status": "not-exposed-by-codex-app-server",
            "auth_mode": "existing-chatgpt-login", "provider": NATIVE_ASTRA_PROVIDER,
            "custom_endpoint_or_api_key": False, "request_max_retries": 0,
            "stream_max_retries": 0, "semantic_retry_count": 0,
            "auth_copy_persisted_in_artifacts": False, "hidden_reasoning_retained": False,
            "instruction_scope": f"focus-only-{task.stage}", "skill_stage_binding": task.stage,
            "trajectory_blake3": blake3_hex(receipt),
        }
        sidecar["document_blake3"] = blake3_hex(sidecar)
        _write_new_or_verify(self._output_root / "native-route-receipts" / f"{task.task_id}.json", canonical_json_bytes(sidecar))
        return receipt

    def close(self) -> None:
        if self._state == "closed":
            return
        try:
            if self._runner is not None:
                self._runner.close()
        finally:
            for manager in reversed(self._auth_contexts):
                manager.__exit__(None, None, None)
            self._auth_contexts.clear()
            self._state = "closed"


__all__ = [
    "NATIVE_ASTRA_MODEL", "NATIVE_ASTRA_ROUTE", "NATIVE_ASTRA_PROVIDER",
    "NativeAstraTeacherPool", "native_astra_route_authorities",
    "native_astra_model_catalog", "native_astra_provider_config",
    "native_astra_launch_options", "native_astra_thread_options",
    "private_native_auth_copy",
]
