"""Prospective Native Codex v4 composition for the medical campaign.

This module is additive: it reuses the frozen v2 candidate/rubric/MCP source
and replaces only the actor execution boundary.  S1/S2 remain read-only MCP
turns; S3/S4/E2E actors use the pinned native SDK with one offline writable
candidate root.  Every judge remains a fresh read-only Opus 5 turn.
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from codex_cli_bin import bundled_codex_path

from eva_agent.codex_pipeline.native_policy_v2 import (
    build_native_policy_v2,
    build_stage_tool_guidance_v1,
    guard_guided_stage_tool_runtime_v1,
    native_policy_v2_config_overrides,
)
from eva_agent.campaign.selection_v2 import FrozenCampaignCandidateSourceV2
from eva_agent.campaign.selection_v3 import SplitCampaignCandidateSourceV3, load as load_selection_v3
from eva_agent.campaign.split_plan_v3 import exact_5000_500_500_plan
from eva_agent.codex_runtime import (
    CodexLaunchOptions,
    CodexRuntime,
    CodexSandbox,
    CodexThreadOptions,
    CodexTurnInput,
    CodexTurnReceipt,
    NativeCodexBackendV2,
    PersistentCodexRuntimeRunner,
    installed_native_runtime_versions,
)
from eva_agent.pipeline.contracts import RolloutRequest, Stage, ToolRuntimePort
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex

from .campaign import (
    CampaignConcurrency,
    CampaignDeploymentConfig,
    CampaignDeploymentError,
    CampaignDeploymentPorts,
    ProviderRouteHealth,
)
from .medresearch_v2 import compose_campaign_v2
from .native_fastpath_v2 import (
    bind_native_actor_options_v2,
    bind_native_judge_options_v2,
)


NATIVE_CAMPAIGN_V4_SCHEMA = "eva.medresearch-native-campaign-composition.v4"
NATIVE_ACTOR_STAGES = (Stage.S3, Stage.S4, Stage.E2E)
NATIVE_SDK_PROTOCOL = "app-server/experimental"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SELECTION_V3 = PROJECT_ROOT / "runs/campaign-selection.v3.4.json"
PROSPECTIVE_EXECUTION_CATALOG_V2 = (
    PROJECT_ROOT / "runs/prospective-execution-binding-catalog.v2.r2.json"
)


def _native_workspace_write_supported() -> bool:
    """Return whether this Linux host can create Codex's sandbox user namespace.

    A real probe is required here: permissive sysctls can still be overridden by
    a container seccomp profile.  The probe is local, bounded, and never opens a
    provider, Codex, MCP, or campaign artifact.
    """

    try:
        completed = subprocess.run(
            ("unshare", "--user", "--map-root-user", "true"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _append_developer_instruction(existing: str | None, required: str) -> str:
    if existing is None:
        return required
    if required in existing:
        return existing
    return f"{existing.rstrip()}\n\n{required}"


class StageAwareNativeRunnerV4:
    """Route write actors to pinned native SDK shards and all reads to v3.

    A native backend is not candidate-pinned: every fresh thread carries and
    reopens its own exact candidate root.  A fixed shard pool therefore gives
    bounded process count while preserving high concurrency.
    """

    def __init__(self, delegate: Any, launches: tuple[CodexLaunchOptions, ...]) -> None:
        if not launches or not all(isinstance(row, CodexLaunchOptions) for row in launches):
            raise CampaignDeploymentError("native v4 launch inventory differs")
        self._delegate = delegate
        self._native = tuple(
            PersistentCodexRuntimeRunner(
                lambda launch=launch: CodexRuntime(NativeCodexBackendV2(launch))
            )
            for launch in launches
        )
        self._lock = RLock()
        self._started = False

    @property
    def shard_count(self) -> int:
        return len(self._native)

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._delegate.start()
            try:
                with ThreadPoolExecutor(max_workers=len(self._native)) as pool:
                    futures = tuple(pool.submit(row.start) for row in self._native)
                    for future in futures:
                        future.result()
            except BaseException:
                with ThreadPoolExecutor(max_workers=len(self._native)) as pool:
                    close_futures = tuple(pool.submit(row.close) for row in self._native)
                    for future in close_futures:
                        try:
                            future.result()
                        except BaseException:
                            pass
                self._delegate.close()
                raise
            self._started = True

    def run_once(
        self, options: CodexThreadOptions, turn_input: CodexTurnInput
    ) -> CodexTurnReceipt:
        if options.sandbox is CodexSandbox.READ_ONLY:
            return self._delegate.run_once(options, turn_input)
        if options.sandbox is not CodexSandbox.WORKSPACE_WRITE:
            raise CampaignDeploymentError("native v4 received an unsupported sandbox")
        self.start()
        index = int(blake3_bytes(options.cwd.encode("utf-8"))[:16], 16) % len(self._native)
        return self._native[index].run_once(options, turn_input)

    def run_once_with_timeout(self, options, turn_input, *, timeout_seconds):
        if options.sandbox is CodexSandbox.READ_ONLY:
            return self._delegate.run_once_with_timeout(
                options, turn_input, timeout_seconds=timeout_seconds
            )
        if options.sandbox is not CodexSandbox.WORKSPACE_WRITE:
            raise CampaignDeploymentError("native v4 received an unsupported sandbox")
        self.start()
        index = int(blake3_bytes(options.cwd.encode("utf-8"))[:16], 16) % len(self._native)
        return self._native[index].run_once_with_timeout(
            options, turn_input, timeout_seconds=timeout_seconds
        )

    def close(self) -> None:
        with self._lock:
            if not self._started:
                self._delegate.close()
                return
            failures: list[BaseException] = []
            with ThreadPoolExecutor(max_workers=len(self._native) + 1) as pool:
                futures = [pool.submit(row.close) for row in self._native]
                futures.append(pool.submit(self._delegate.close))
                for future in futures:
                    try:
                        future.result()
                    except BaseException as exc:
                        failures.append(exc)
            self._started = False
            if failures:
                raise CampaignDeploymentError("native v4 runtime close failed") from failures[0]


def _source_catalog(request: RolloutRequest) -> tuple[Mapping[str, Any], ...]:
    return tuple(request.available_tools)


def _legacy_episode_context(
    public_context: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Select the exact signed legacy context from the pipeline projection."""

    if not isinstance(public_context, Mapping):
        raise CampaignDeploymentError(
            "native v4 legacy runtime context projection differs"
        )
    legacy_context = public_context.get("episode_context")
    if not isinstance(legacy_context, Mapping):
        raise CampaignDeploymentError(
            "native v4 legacy runtime context projection differs"
        )
    return legacy_context


def compose_campaign_v4(
    *,
    concurrency: CampaignConcurrency,
    provider_health: Mapping[str, ProviderRouteHealth],
    host_private_key_path: Path,
    host_key_id: str,
    host_trust_store_path: Path,
    expected_host_public_key_blake3: str,
    expected_host_trust_store_blake3: str,
) -> tuple[CampaignDeploymentConfig, CampaignDeploymentPorts]:
    """Compose v4 without opening Codex, MCP, adapters, or provider routes."""

    config, base = compose_campaign_v2(
        concurrency=concurrency,
        provider_health=provider_health,
        host_private_key_path=host_private_key_path,
        host_key_id=host_key_id,
        host_trust_store_path=host_trust_store_path,
        expected_host_public_key_blake3=expected_host_public_key_blake3,
        expected_host_trust_store_blake3=expected_host_trust_store_blake3,
    )
    if base.runtime is None or base.turn_mcp_factory is None or base.actor_skills_factory is None:
        raise CampaignDeploymentError("native v4 base composition is incomplete")
    if not isinstance(base.candidate_source, FrozenCampaignCandidateSourceV2):
        raise CampaignDeploymentError("native v4 base candidate source differs")
    split_plan = exact_5000_500_500_plan()
    split_selection = load_selection_v3(
        SELECTION_V3, base.candidate_source.selection, split_plan
    )
    split_source = SplitCampaignCandidateSourceV3(
        base.candidate_source, split_selection, split_plan
    )
    config = replace(
        config,
        workspace_root=PROJECT_ROOT / "runs/campaign-v4-workspaces",
        artifact_root=PROJECT_ROOT / "runs/campaign-v4-artifacts",
        ledger_path=PROJECT_ROOT / "runs/campaign-v4.sqlite3",
        receipt_root=PROJECT_ROOT / "runs/campaign-v4-receipts",
        admission_bundle_root=PROJECT_ROOT / "runs/campaign-v4-admissions",
    )
    base = replace(
        base,
        plan=split_plan,
        candidate_source=split_source,
        prospective_execution_catalog_path=PROSPECTIVE_EXECUTION_CATALOG_V2,
    )
    versions = installed_native_runtime_versions()
    cli_blake3 = blake3_bytes(bundled_codex_path().resolve().read_bytes())
    turn_mcp_metadata = base.turn_mcp_factory.public_metadata()
    base_actor = base.actor_options_factory
    base_judge = base.judge_options_factory
    if not callable(base_actor) or not callable(base_judge):
        raise CampaignDeploymentError("native v4 base option factories are absent")
    native_workspace_write = _native_workspace_write_supported()
    active_native_actor_stages = NATIVE_ACTOR_STAGES if native_workspace_write else ()

    def actor_options(request, cwd, offers, bridge):
        options = base_actor(request, cwd, offers, bridge)
        if request.sandbox.stage not in NATIVE_ACTOR_STAGES:
            return options
        guidance = (
            build_stage_tool_guidance_v1(
                public_runtime_context=_legacy_episode_context(
                    request.policy_visible_context
                ),
                source_tool_catalog=_source_catalog(request),
            )
            if request.sandbox.stage is Stage.E2E
            else None
        )
        if not native_workspace_write:
            if guidance is None:
                return options
            return replace(
                options,
                developer_instructions=_append_developer_instruction(
                    options.developer_instructions, guidance.prompt_text
                ),
            )
        thread_config = dict(options.config)
        features = dict(thread_config.get("features", {}))
        features.update({"shell_tool": False, "unified_exec": False})
        thread_config["features"] = features
        thread_config["sandbox_workspace_write"] = {
            "network_access": False,
            "writable_roots": [cwd],
            "exclude_slash_tmp": True,
            "exclude_tmpdir_env_var": True,
        }
        writable = replace(options, sandbox=CodexSandbox.WORKSPACE_WRITE, config=thread_config)
        skills = tuple(base.actor_skills_factory(request))
        policy = build_native_policy_v2(
            stage=request.sandbox.stage,
            sdk_version=versions.sdk,
            sdk_protocol=NATIVE_SDK_PROTOCOL,
            cli_version=versions.cli,
            cli_executable_blake3=cli_blake3,
            skills=skills,
            turn_mcp_metadata=turn_mcp_metadata,
        )
        return bind_native_actor_options_v2(writable, policy, guidance)

    def judge_options(request, offers):
        return bind_native_judge_options_v2(base_judge(request, offers))

    def guard(request: RolloutRequest, tools: ToolRuntimePort) -> ToolRuntimePort:
        if request.sandbox.stage is not Stage.E2E:
            return tools
        guidance = build_stage_tool_guidance_v1(
            public_runtime_context=_legacy_episode_context(
                request.policy_visible_context
            ),
            source_tool_catalog=_source_catalog(request),
        )
        return guard_guided_stage_tool_runtime_v1(tools, guidance)

    launch_env = base.runtime.launch_options.env
    native_launches = tuple(
        CodexLaunchOptions(cwd=str(Path(config.workspace_root).parent), env=launch_env)
        for _ in range(concurrency.app_server_shards)
    )
    runner = StageAwareNativeRunnerV4(base.runtime.runner, native_launches)
    startup_core = {
        key: value
        for key, value in base.runtime.startup_metadata.items()
        if key != "launch_blake3"
    }
    startup_core.update(
        {
            "native_campaign_schema": NATIVE_CAMPAIGN_V4_SCHEMA,
            "native_actor_stages": [stage.value for stage in active_native_actor_stages],
            "native_sdk_version": versions.sdk,
            "native_cli_version": versions.cli,
            "native_cli_executable_blake3": cli_blake3,
            "native_app_server_shards": concurrency.app_server_shards,
            "native_provider_calls_during_composition": 0,
            "native_activation_gate": (
                "thread-start must return the exact requested workspaceWrite "
                "policy with cwd as its implicit root, no additional writableRoots, "
                "and approvalPolicy=never; mismatch terminates "
                "before turn/start and before provider"
            ),
        }
    )
    runtime = replace(
        base.runtime,
        runner=runner,
        startup_metadata={**startup_core, "launch_blake3": blake3_hex(startup_core)},
    )
    return config, replace(
        base,
        runtime=runtime,
        actor_options_factory=actor_options,
        judge_options_factory=judge_options,
        native_actor_stages=active_native_actor_stages,
        actor_tool_runtime_guard=guard,
    )


__all__ = [
    "NATIVE_ACTOR_STAGES",
    "NATIVE_CAMPAIGN_V4_SCHEMA",
    "StageAwareNativeRunnerV4",
    "compose_campaign_v4",
]
