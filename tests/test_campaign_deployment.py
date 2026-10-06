from __future__ import annotations

import base64
from dataclasses import replace
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
from types import MappingProxyType, SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eva_agent.deployment import (
    CODEX_FIRST_RELEASE_CONFIG_OVERRIDES,
    DEFAULT_CHILD_SOFT_NOFILE,
    CampaignConcurrency,
    CampaignDeploymentConfig,
    CampaignDeploymentError,
    CampaignDeploymentPorts,
    CodexProviderTiers,
    DeterministicRouteSelector,
    ProviderRouteHealth,
    ProviderRouteTarget,
    build_codex_campaign_deployment,
    build_one_candidate_canary_deployment,
    prepare_persistent_codex_runtime,
)
from eva_agent.campaign import exact_6000_plan
from eva_agent.orchestration import CandidateJob
from eva_agent.orchestration.receipts import ReceiptJournal
from eva_agent.pipeline import (
    BenchmarkEpisode,
    BenchmarkSource,
    Cohort,
    ModelTarget,
    Stage,
    ToolDefinition,
    ToolRegistry,
)
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.codex_runtime import (
    CodexRole,
    CodexSandbox,
    CodexSkill,
    CodexThreadOptions,
)
from eva_agent.codex_pipeline import TurnMCPBridgeFactory
from eva_agent.admission import issue_signed_envelope
from eva_agent.pipeline.digests import canonical_json_bytes
from eva_agent.deployment.campaign import _verify_prospective_execution_allowlist


ROOT = Path(__file__).resolve().parents[1]


def _prospective_catalog(
    tmp_path: Path, source, private_path: Path, legacy_catalog_blake3: str
) -> tuple[Path, tuple[str, ...]]:
    executable = tuple(sorted(source.candidate_ids[:2]))
    executable_set = set(executable)
    rows = []
    for candidate_id in source.candidate_ids:
        status = "executable_legacy" if candidate_id in executable_set else "source_only"
        core = {
            "candidate_id": candidate_id,
            "source_candidate_id": source.source_candidate_id(candidate_id),
            "execution_status": status,
            "execution_proof": (
                {"resolver_catalog_blake3": legacy_catalog_blake3}
                if status == "executable_legacy"
                else None
            ),
            "execution_blocker": None,
        }
        rows.append({**core, "row_blake3": blake3_hex(core)})
    core = {
        "schema": "eva.prospective-execution-binding-catalog.v2",
        "catalog_revision": 1,
        "selection_blake3": source.selection_blake3,
        "source_registry_blake3": source.source_registry_blake3,
        "ancestor_legacy_catalog_blake3": legacy_catalog_blake3,
        "scheduled_count": 9_000,
        "executable_count": len(executable),
        "source_only_count": 9_000 - len(executable),
        "rejected_count": 0,
        "controls": {
            "provider_calls": 0,
            "ledger_reads": 0,
            "ledger_writes": 0,
            "launch_reads": 0,
            "launch_writes": 0,
            "source_only_is_executable": False,
            "rejected_is_executable": False,
            "full_binding_validation_required": True,
        },
        "rows": rows,
    }
    envelope = issue_signed_envelope(
        {**core, "catalog_blake3": blake3_hex(core)},
        key_id="campaign-key",
        private_key_path=private_path,
    )
    path = tmp_path / "prospective.json"
    path.write_bytes(canonical_json_bytes(envelope.to_document()))
    path.chmod(0o600)
    return path, executable


def test_prospective_catalog_is_signed_restrictive_allowlist(tmp_path: Path, rubrics) -> None:
    source = _FrozenSource(rubrics, _tiers())
    private_path, trust_path = _key_material(tmp_path)
    legacy_digest = blake3_hex("legacy-catalog")
    path, expected = _prospective_catalog(tmp_path, source, private_path, legacy_digest)
    candidate_ids, catalog_digest, envelope_digest, ids_digest = (
        _verify_prospective_execution_allowlist(
            path,
            trust_store_path=trust_path,
            source=source,
            legacy_catalog_blake3=legacy_digest,
            legacy_candidate_ids=tuple(sorted(source.candidate_ids[:10])),
        )
    )
    assert candidate_ids == expected
    assert all(isinstance(value, str) for value in (catalog_digest, envelope_digest, ids_digest))
    assert len({catalog_digest, envelope_digest, ids_digest}) == 3


def test_prospective_catalog_cannot_expand_legacy_inventory(tmp_path: Path, rubrics) -> None:
    source = _FrozenSource(rubrics, _tiers())
    private_path, trust_path = _key_material(tmp_path)
    legacy_digest = blake3_hex("legacy-catalog")
    path, _expected = _prospective_catalog(tmp_path, source, private_path, legacy_digest)
    with pytest.raises(CampaignDeploymentError, match="expands legacy"):
        _verify_prospective_execution_allowlist(
            path,
            trust_store_path=trust_path,
            source=source,
            legacy_catalog_blake3=legacy_digest,
            legacy_candidate_ids=(),
        )


def test_deployment_recipe_uses_dynamic_prospective_intersection(
    deployment_inputs,
) -> None:
    config, ports, _calls = deployment_inputs
    assert ports.candidate_source is not None
    assert ports.execution_binding_resolver is not None
    path, expected = _prospective_catalog(
        config.workspace_root.parent,
        ports.candidate_source,
        config.private_key_path,
        ports.execution_binding_resolver.catalog_inventory_blake3,
    )
    deployment = build_codex_campaign_deployment(
        config=config,
        ports=replace(ports, prospective_execution_catalog_path=path),
    )
    assert deployment.executable_candidate_ids == expected
    assert deployment.recipe.executable_binding_count == 2
    assert deployment.recipe.prospective_executable_count == 2
    assert deployment.recipe.prospective_execution_catalog_blake3 is not None
    assert deployment.recipe.core_document()["execution_binding_coverage"] == (
        "signed_v24_promoted_intersect_signed_prospective_v2"
    )


def _candidate_ids() -> tuple[str, ...]:
    return tuple(
        str(uuid5(NAMESPACE_URL, f"eva-deployment-test:{index}"))
        for index in range(9_000)
    )


def _route(route_id: str, cohort: Cohort, model_id: str) -> ProviderRouteTarget:
    return ProviderRouteTarget(
        route_id=route_id,
        target=ModelTarget(cohort, model_id, f"eva_{route_id}"),
        config_blake3=blake3_hex({"route": route_id, "retry_count": 0}),
    )


def _tiers() -> CodexProviderTiers:
    opus = _route("opus_5", Cohort.STRONG, "anthropic/claude-opus-5")
    return CodexProviderTiers(
        weak=(_route("deepseek_v4_flash", Cohort.WEAK, "deepseek/v4-flash"),),
        middle=(
            _route("gemini_3_1_pro", Cohort.MIDDLE, "google/gemini-3.1-pro-preview"),
            _route("opus_4_8", Cohort.MIDDLE, "anthropic/claude-opus-4.8"),
        ),
        strong=(
            _route("gpt_5_6_sol", Cohort.STRONG, "openai/gpt-5.6-sol"),
            opus,
        ),
        judge=opus,
    )


class _FrozenSource:
    def __init__(self, rubrics, tiers: CodexProviderTiers) -> None:
        self.candidate_ids = _candidate_ids()
        self.primary_candidate_ids = self.candidate_ids[:6_000]
        self.reserve_candidate_ids = self.candidate_ids[6_000:]
        self.candidate_count = 9_000
        self.scheduled_count = 9_000
        self.primary_count = 6_000
        self.reserve_count = 3_000
        plan = exact_6000_plan()
        self.plan_blake3 = plan.plan_blake3
        self.rubric_registry_blake3 = rubrics.digest
        self.source_registry_blake3 = blake3_hex("source-registry")
        self.selection_blake3 = blake3_hex("selection")
        self._source_ids = {
            candidate_id: f"source-candidate-{index:04d}"
            for index, candidate_id in enumerate(self.candidate_ids)
        }
        cell = plan.cells[0]
        rubric = rubrics.resolve(cell.domain, cell.stage)
        episode = BenchmarkEpisode(
            episode_id="deployment-test-episode",
            source=BenchmarkSource("test", "episode.json", "frozen-v1"),
            domain=cell.domain,
            stage=Stage(cell.stage),
            instruction="Use the offered tools and leave verifiable workspace evidence.",
            policy_context={"case": "synthetic"},
            initial_files={"README.md": b"frozen\n"},
        )
        self._job = CandidateJob(
            candidate_id=self.candidate_ids[0],
            episode=episode,
            rubric=rubric,
            targets=tuple(pool[0].target for pool in (tiers.weak, tiers.middle, tiers.strong)),
        )

    def load(self, candidate_id: str) -> CandidateJob:
        if candidate_id != self._job.candidate_id:
            raise KeyError(candidate_id)
        return self._job

    def source_candidate_id(self, candidate_id: str) -> str:
        return self._source_ids[candidate_id]

    def queue_records(self, *, worker_width: int = 64) -> tuple[object, ...]:
        del worker_width
        return ()


class _BindingResolver:
    def __init__(self, source: _FrozenSource, tools: ToolRegistry) -> None:
        self._source = source
        self._tools = tools
        self._inventory = tuple(
            SimpleNamespace(source_candidate_id=source.source_candidate_id(candidate_id))
            for candidate_id in source.candidate_ids[:1_344]
        )
        self.catalog_inventory_blake3 = blake3_hex(
            [row.source_candidate_id for row in self._inventory]
        )
        self.executable_candidate_count = 1_344

    def inventory(self):
        return self._inventory

    def resolve(self, eva_candidate_id: str, *, source_candidate_id: str):
        if self._source.source_candidate_id(eva_candidate_id) != source_candidate_id:
            raise KeyError(source_candidate_id)
        return SimpleNamespace(
            candidate_id=eva_candidate_id,
            source_candidate_id=source_candidate_id,
            initial_workspace_files={".eva/runtime.json": b"{}\n"},
            public_runtime_context={"source_candidate_id": source_candidate_id},
            tool_registry=self._tools,
            binding_blake3=blake3_hex(
                {"candidate_id": eva_candidate_id, "source_candidate_id": source_candidate_id}
            ),
        )


class _TurnMCPFactory:
    maximum = 64

    _proxy_python = Path(sys.executable).resolve()
    _proxy_exec = (
        ROOT / "src" / "eva_agent" / "codex_pipeline" / "turn_mcp_exec.py"
    ).resolve()
    _proxy = (
        ROOT / "src" / "eva_agent" / "codex_pipeline" / "turn_mcp_proxy.py"
    ).resolve()

    def __init__(self, temp_root: Path) -> None:
        actual = TurnMCPBridgeFactory(
            proxy_python=self._proxy_python,
            proxy_script=self._proxy,
            temp_root=temp_root,
            maximum_parallel_calls=self.maximum,
        )
        metadata = dict(actual.public_metadata())
        metadata.pop("launch_blake3")
        self.temp_root = actual.temp_root
        self._metadata_core = metadata
        self.launch_blake3 = blake3_hex(self._metadata_core)

    def public_metadata(self):
        return MappingProxyType(
            {**self._metadata_core, "launch_blake3": self.launch_blake3}
        )

    def open_actor(self, _options, _bridge):  # pragma: no cover - build-only port
        raise AssertionError("zero-provider build cannot open actor MCP")

    def open_judge(self, _options, _bridge):  # pragma: no cover - build-only port
        raise AssertionError("zero-provider build cannot open judge MCP")


class _Skills:
    materialization_path_policy = "blake3-root/ordinal-content-blake3/SKILL.md"

    def __init__(self, root: Path) -> None:
        runtime_root = root / "skill-runtime"
        runtime_root.mkdir(mode=0o700)
        runtime_root.chmod(0o700)
        payload = b"# Test stage skill\n"
        content_blake3 = blake3_bytes(payload)
        self.materialization_blake3 = blake3_hex(
            {"schema": "test-skill-materialization", "content": content_blake3}
        )
        self.materialization_root = (
            runtime_root / f"blake3-{self.materialization_blake3}"
        )
        self.materialization_root.mkdir(mode=0o700)
        skill_directory = self.materialization_root / f"000-{content_blake3}"
        skill_directory.mkdir(mode=0o700)
        skill_path = skill_directory / "SKILL.md"
        skill_path.write_bytes(payload)
        skill_path.chmod(0o400)
        skill_directory.chmod(0o500)
        self.materialization_root.chmod(0o500)
        self._skill = CodexSkill(
            skill_id="test-stage-skill",
            name="test-stage-skill",
            path=str(skill_path.resolve()),
            content_blake3=content_blake3,
        )
        self.catalog_blake3 = blake3_hex(
            {
                "materialization_blake3": self.materialization_blake3,
                "path": self._skill.path,
            }
        )

    def inventory(self):
        return ({"skill_id": "test-stage-skill", "path": self._skill.path},)

    def for_stage(self, _stage):
        return (self._skill,)

    def __call__(self, _request):
        return ()


class _OpusAdapterGateway:
    shard_count = 2
    binding_blake3 = blake3_hex("verified-opus5-adapter-binding")

    def __init__(self, capacity: int = 512) -> None:
        self.aggregate_max_concurrency = capacity
        self.started = 0
        self.closed = 0

    def start(self):  # pragma: no cover - provider-free build only
        self.started += 1

    def close(self):  # pragma: no cover - provider-free build only
        self.closed += 1


def _key_material(tmp_path: Path) -> tuple[Path, Path]:
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "signer.pem"
    private_path.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    private_path.chmod(0o600)
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trust_path = tmp_path / "trust.json"
    trust_path.write_text(
        json.dumps(
            {
                "schema": "eva.ed25519-trust-store.v1",
                "status": "active",
                "algorithm": "Ed25519",
                "keys": {"campaign-key": base64.b64encode(public).decode("ascii")},
            }
        ),
        encoding="utf-8",
    )
    return private_path, trust_path


@pytest.fixture(scope="module")
def rubrics():
    return load_and_compile_registry(ROOT / "rubrics/source/domain-stage-tables.v1.json")


@pytest.fixture
def deployment_inputs(tmp_path: Path, rubrics):
    tiers = _tiers()
    source = _FrozenSource(rubrics, tiers)
    private_path, trust_path = _key_material(tmp_path)
    calls = {"actor_options": 0, "judge_options": 0}

    def actor_options(*_args, **_kwargs):
        calls["actor_options"] += 1
        raise AssertionError("zero-provider build must not ask for actor options")

    def judge_options(*_args, **_kwargs):
        calls["judge_options"] += 1
        raise AssertionError("zero-provider build must not ask for judge options")

    before_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    runtime = prepare_persistent_codex_runtime(
        codex_bin="true",
        cwd=tmp_path,
        child_env={"EVA_CODEX_TEST_TOKEN": "never-record-this-value"},
        required_process_soft_nofile=before_limit[0],
        app_server_shards=1,
        worker_width=128,
    )
    assert resource.getrlimit(resource.RLIMIT_NOFILE) == before_limit
    config = CampaignDeploymentConfig(
        workspace_root=tmp_path / "workspaces",
        artifact_root=tmp_path / "artifacts",
        ledger_path=tmp_path / "campaign.sqlite3",
        receipt_root=tmp_path / "receipts",
        admission_bundle_root=tmp_path / "admission",
        private_key_path=private_path,
        trust_store_path=trust_path,
        signing_key_id="campaign-key",
        concurrency=CampaignConcurrency(
            required_process_soft_nofile=before_limit[0],
        ),
    )
    tools = ToolRegistry(
        (
            ToolDefinition(
                name="workspace_note",
                description="Write one evidence note.",
                input_schema={"type": "object", "additionalProperties": False},
                handler=lambda _workspace, _arguments: {"ok": True},
                parallel_safe=False,
            ),
        )
    )
    mcp_root = Path(tempfile.mkdtemp(prefix="eva-campaign-mcp-test-", dir="/tmp"))
    ports = CampaignDeploymentPorts(
        plan=exact_6000_plan(),
        candidate_source=source,
        rubrics=rubrics,
        provider_tiers=tiers,
        provider_health=MappingProxyType(
            {
                route.route_id: ProviderRouteHealth(
                    route_id=route.route_id,
                    model_id=route.target.model_id,
                    status=(
                        "adapter-pass"
                        if route.route_id == "deepseek_v4_flash"
                        else "direct-pass"
                    ),
                    canary_receipt_blake3=blake3_hex(
                        {"canary": route.route_id, "attempts": 1}
                    ),
                    verified=True,
                )
                for route in tiers.actor_routes
            }
        ),
        runtime=runtime,
        actor_options_factory=actor_options,
        judge_options_factory=judge_options,
        turn_mcp_factory=_TurnMCPFactory(mcp_root),
        execution_binding_resolver=_BindingResolver(source, tools),
        actor_skills_factory=_Skills(tmp_path),
        opus5_adapter_gateway=_OpusAdapterGateway(),
    )
    try:
        yield config, ports, calls
    finally:
        assert list(mcp_root.iterdir()) == []
        mcp_root.rmdir()


def test_zero_provider_factory_wires_shared_runner_exact_rubrics_and_lower_gate(
    deployment_inputs,
) -> None:
    config, ports, calls = deployment_inputs
    deployment = build_codex_campaign_deployment(config=config, ports=ports)
    assert calls == {"actor_options": 0, "judge_options": 0}
    assert deployment.runtime_runner is ports.runtime.runner
    assert deployment.prepared_runtime is ports.runtime
    assert deployment.rubrics is ports.rubrics
    assert deployment.plan is ports.plan
    assert deployment.plan.total == 6_000
    assert deployment.frozen_source.candidate_count == 9_000
    assert deployment.orchestrator.config.worker_width == 128
    assert deployment.orchestrator.config.max_candidates is None
    assert deployment.recipe.build_provider_calls_made == 0
    assert deployment.recipe.semantic_retry_count == 0
    assert dict(deployment.recipe.separation_policy) == {
        "schema": "eva.trajectory-admission-policy.v2",
        "minimum_valid_judged_trajectories": 1,
        "ability_separation_required_for_admission": False,
        "early_continuation_after_first_valid_trajectory": True,
        "minimum_strong_reward": 0.60,
        "minimum_strong_minus_weak": 0.05,
        "material_item_delta": 0.10,
        "require_strong_hard_gates": True,
        "perfect_monotonic_staircase_required": False,
        "raw_scores_retained": True,
        "provenance_gate_lowered": False,
        "workspace_agent_judge_gate_lowered": False,
        "native_metric_gate_lowered": False,
        "signature_gate_lowered": False,
    }


    job = deployment.candidate_source.load(
        ports.candidate_source.candidate_ids[0]
    )
    pipeline = deployment.pipeline_factory("worker-001", job)
    assert pipeline._providers._runner is deployment.runtime_runner
    assert pipeline._judge._runner is deployment.runtime_runner
    assert pipeline._model_width == 3
    assert pipeline._tool_width == 64
    assert pipeline._tools is ports.execution_binding_resolver._tools
    assert pipeline._providers._turn_mcp is ports.turn_mcp_factory
    assert pipeline._judge._turn_mcp is ports.turn_mcp_factory
    assert deployment.recipe.executable_binding_count == 1_344
    assert deployment.recipe.execution_binding_catalog_blake3 == (
        ports.execution_binding_resolver.catalog_inventory_blake3
    )
    assert deployment.recipe.skill_mount_catalog_blake3 == (
        ports.actor_skills_factory.catalog_blake3
    )
    assert deployment.recipe.skill_materialization_root == str(
        ports.actor_skills_factory.materialization_root
    )
    assert deployment.recipe.skill_materialization_blake3 == (
        ports.actor_skills_factory.materialization_blake3
    )
    assert deployment.recipe.skill_materialization_path_policy == (
        "blake3-root/ordinal-content-blake3/SKILL.md"
    )
    assert deployment.recipe.skill_materialization_count == 1
    assert deployment.recipe.turn_mcp_launch_blake3 == (
        ports.turn_mcp_factory.launch_blake3
    )
    assert (
        deployment.recipe.turn_mcp_launch_metadata[
            "inherited_parent_environment"
        ]
        is False
    )
    assert deployment.recipe.core_document()["actor_runtime_surface"] == {
        "tools": "exact_candidate_policy_registry",
        "skills": "verified_stage_skill_mounts",
        "dynamic_skill_discovery_tools_exposed": False,
        "codex_sandbox": "read-only",
        "candidate_mutations": "exact_candidate_policy_mcp_only",
        "native_codex_surface": deployment.recipe.runtime_startup_metadata[
            "native_codex_surface"
        ],
        "project_doc_max_bytes": 0,
        "ancestor_project_docs_injected": False,
    }
    assert pipeline._policy.minimum_strong_reward == 0.60
    assert calls == {"actor_options": 0, "judge_options": 0}


def test_route_assignment_is_deterministic_exactly_balanced_and_one_per_cohort(
    deployment_inputs,
) -> None:
    _config, ports, _calls = deployment_inputs
    first = DeterministicRouteSelector(
        candidate_ids=ports.candidate_source.candidate_ids,
        selection_blake3=ports.candidate_source.selection_blake3,
        tiers=ports.provider_tiers,
    )
    second = DeterministicRouteSelector(
        candidate_ids=tuple(reversed(ports.candidate_source.candidate_ids)),
        selection_blake3=ports.candidate_source.selection_blake3,
        tiers=ports.provider_tiers,
    )
    assert first.receipt == second.receipt
    assert first.receipt.counts_by_route["weak:deepseek_v4_flash"] == 9_000
    assert set(
        count
        for key, count in first.receipt.counts_by_route.items()
        if not key.startswith("weak:")
    ) == {4_500}
    assert first.receipt.core_document()["rollouts_per_candidate"] == {
        "weak": 1,
        "middle": 1,
        "strong": 1,
    }
    targets = first.targets_for(ports.candidate_source.candidate_ids[0])
    assert tuple(target.cohort for target in targets) == (
        Cohort.WEAK,
        Cohort.MIDDLE,
        Cohort.STRONG,
    )


def test_exact_candidate_source_rebinds_only_routes_and_preserves_rubric_identity(
    deployment_inputs,
) -> None:
    config, ports, _calls = deployment_inputs
    deployment = build_codex_campaign_deployment(config=config, ports=ports)
    candidate_id = ports.candidate_source.candidate_ids[0]
    source_job = ports.candidate_source.load(candidate_id)
    routed_job = deployment.candidate_source.load(candidate_id)
    assert routed_job.episode is not source_job.episode
    assert routed_job.episode.initial_files["README.md"] == b"frozen\n"
    assert routed_job.episode.initial_files[".eva/runtime.json"] == b"{}\n"
    assert "execution_binding" in routed_job.episode.policy_context
    assert routed_job.rubric is source_job.rubric
    assert routed_job.targets == deployment.route_selector.targets_for(candidate_id)
    assert len(routed_job.targets) == 3


def test_external_nofile_preflight_is_safe_and_receipt_contains_no_secret(
    tmp_path: Path,
) -> None:
    before = resource.getrlimit(resource.RLIMIT_NOFILE)
    prepared = prepare_persistent_codex_runtime(
        codex_bin="true",
        cwd=tmp_path,
        child_env={"EVA_CODEX_TEST_TOKEN": "top-secret-value"},
    )
    after = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert after == before
    argv = prepared.launch_options.launch_args_override
    assert argv is not None
    assert str(Path("/usr/bin/true").resolve()) in argv
    assert argv[-4:] == ("app-server", "--strict-config", "--listen", "stdio://")
    assert "project_doc_max_bytes=0" in argv
    assert "features.shell_tool=false" in argv
    assert "features.unified_exec=false" in argv
    assert "top-secret-value" not in "\x00".join(argv)
    metadata = dict(prepared.startup_metadata)
    assert metadata["process_soft_nofile_required"] == DEFAULT_CHILD_SOFT_NOFILE
    assert metadata["process_soft_nofile_observed"] == before[0]
    assert metadata["process_nofile_sufficient"] is (before[0] >= 65_536)
    assert metadata["app_server_shards"] == 1
    assert metadata["worker_width_configured"] == 128
    assert metadata["parent_limit_mutated"] is False
    assert metadata["child_env_values_recorded"] is False
    assert metadata["strict_config"] is True
    assert metadata["project_doc_max_bytes"] == 0
    assert metadata["uncommitted_ancestor_project_docs_enabled"] is False
    assert metadata["sanitized_child_exec"]["inherited_parent_environment"] is False
    assert metadata["native_codex_surface"]["shell_tool"] is False
    assert metadata["native_codex_surface"]["code_mode_host"] is False
    assert (
        metadata["native_codex_surface"]["tool_search_always_defer_mcp_tools"]
        is False
    )
    assert "top-secret-value" not in json.dumps(metadata)


def test_first_release_codex_config_excludes_root_agents_and_unknown_config_fails(
    tmp_path: Path,
) -> None:
    from codex_cli_bin import bundled_codex_path

    codex = bundled_codex_path().resolve()
    debug_argv = [str(codex)]
    for override in CODEX_FIRST_RELEASE_CONFIG_OVERRIDES:
        debug_argv.extend(("--config", override))
    debug_argv.extend(("debug", "prompt-input", "provider-free-isolation-probe"))
    rendered = subprocess.run(
        debug_argv,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout
    assert "# EVA-Agent contributor rules" not in rendered
    assert "The data pipeline is the primary product" not in rendered

    unknown = subprocess.run(
        (
            str(codex),
            "--strict-config",
            "--config",
            "eva_unknown_release_field=true",
            "app-server",
            "--listen",
            "stdio://",
        ),
        cwd=tmp_path,
        input="",
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert unknown.returncode != 0
    assert "unknown configuration field" in unknown.stderr


def test_candidate_transport_rejects_uncommitted_native_codex_surface(
    tmp_path: Path,
) -> None:
    from eva_agent.deployment.campaign import (
        CODEX_FIRST_RELEASE_THREAD_CONFIG,
        _require_base_options,
    )
    from eva_agent.codex_pipeline import CodexPipelineError

    def options(config):
        return CodexThreadOptions(
            role=CodexRole.STRONG_ACTOR,
            model="model",
            provider="provider",
            cwd=str(tmp_path.resolve()),
            sandbox=CodexSandbox.READ_ONLY,
            config=config,
        )

    safe = {
        "project_doc_max_bytes": 0,
        "web_search": "disabled",
        "features": dict(CODEX_FIRST_RELEASE_THREAD_CONFIG["features"]),
    }
    assert _require_base_options(options(safe), label="actor")
    unsafe = {**safe, "features": {**safe["features"], "shell_tool": True}}
    with pytest.raises(CodexPipelineError, match="uncommitted Codex surface"):
        _require_base_options(options(unsafe), label="actor")
    with pytest.raises(CodexPipelineError, match="read-only Codex sandbox"):
        _require_base_options(
            replace(options(safe), sandbox=CodexSandbox.WORKSPACE_WRITE),
            label="actor",
        )


def test_real_medresearch_route_source_uses_workspace_env_without_exposing_values() -> None:
    from eva_agent.deployment import medresearch_v2

    if not all(
        path.is_file()
        for path in (
            medresearch_v2.PROJECT_ROOT.parent / ".env",
            medresearch_v2.PROJECT_ROOT.parent / "keys.env",
            medresearch_v2.MODEL_REGISTRY,
        )
    ):
        pytest.skip("external signed medresearch route sources are not installed")
    routes = medresearch_v2._provider_routes()
    assert set(routes) == set(medresearch_v2.REQUIRED_ROUTE_IDS)
    assert {route.config.credential_env_name for route in routes.values()} == {
        "NVIDIA_INFERENCE_API_KEY"
    }
    rendered = repr(routes)
    assert "credential=<redacted>" in rendered


def test_real_opus_gateway_shards_cover_maximum_profile_without_token_receipt() -> None:
    from eva_agent.codex_providers import AdapterReceiptSigner
    from eva_agent.deployment import medresearch_v2

    if not all(
        path.is_file()
        for path in (
            medresearch_v2.PROJECT_ROOT.parent / ".env",
            medresearch_v2.PROJECT_ROOT.parent / "keys.env",
            medresearch_v2.MODEL_REGISTRY,
        )
    ):
        pytest.skip("external signed medresearch route sources are not installed")
    route = medresearch_v2._provider_routes()["opus_5"]
    gateway = medresearch_v2.ShardedOpus5AdapterGateway(
        route=route,
        app_server_shards=64,
        required_capacity=2_048,
        signer=AdapterReceiptSigner.ephemeral(),
        canary_receipt_blake3=blake3_hex("test-opus-canary"),
    )
    token = gateway.child_environment[medresearch_v2.LOCAL_ADAPTER_TOKEN_ENV]
    assert gateway.shard_count == 64
    assert gateway.per_shard_capacity == 32
    assert gateway.aggregate_max_concurrency == 2_048
    assert gateway.planned_route.config.credential_env_name == (
        medresearch_v2.LOCAL_ADAPTER_TOKEN_ENV
    )
    assert token not in repr(gateway)
    assert token not in gateway.binding_blake3
    gateway.close()
    assert gateway.child_environment == {}


def test_one_candidate_real_canary_entry_has_one_claim_and_three_parallel_models(
    deployment_inputs,
) -> None:
    config, ports, calls = deployment_inputs
    canary = build_one_candidate_canary_deployment(config=config, ports=ports)
    assert canary.orchestrator.config.worker_width == 1
    assert canary.orchestrator.config.queue_capacity == 1
    assert canary.orchestrator.config.claim_batch_size == 1
    assert canary.orchestrator.config.max_candidates == 1
    job = canary.candidate_source.load(ports.candidate_source.candidate_ids[0])
    pipeline = canary.pipeline_factory("canary-worker", job)
    assert pipeline._model_width == 3
    assert calls == {"actor_options": 0, "judge_options": 0}


def test_schedule_install_passes_all_rows_and_exact_selection_commitment(
    deployment_inputs, monkeypatch
) -> None:
    config, ports, _calls = deployment_inputs
    deployment = build_codex_campaign_deployment(config=config, ports=ports)
    records = tuple(object() for _ in range(9_000))
    observed = {}

    def queue_records(*, worker_width: int):
        observed["worker_width"] = worker_width
        return records

    def bootstrap(candidate_records, *, selection_blake3: str):
        observed["records"] = candidate_records
        observed["selection_blake3"] = selection_blake3
        return 9_000

    monkeypatch.setattr(ports.candidate_source, "queue_records", queue_records)
    monkeypatch.setattr(deployment.ledger, "bootstrap_candidates", bootstrap)
    assert deployment.install_frozen_schedule(worker_width=128) == 9_000
    assert observed == {
        "worker_width": 128,
        "records": records,
        "selection_blake3": ports.candidate_source.selection_blake3,
    }


@pytest.mark.parametrize(
    ("field_name", "message"),
    (
        ("plan", "CampaignPlan"),
        ("candidate_source", "candidate source"),
        ("rubrics", "rubric"),
        ("provider_tiers", "provider tier"),
        ("provider_health", "route canaries"),
        ("runtime", "persistent Codex runner"),
        ("actor_options_factory", "actor MCP"),
        ("judge_options_factory", "judge MCP"),
        ("turn_mcp_factory", "turn MCP"),
        ("execution_binding_resolver", "binding resolver"),
        ("actor_skills_factory", "skill mount"),
        ("opus5_adapter_gateway", "adapter capacity"),
    ),
)
def test_missing_production_port_fails_closed(deployment_inputs, field_name, message) -> None:
    config, ports, _calls = deployment_inputs
    with pytest.raises(CampaignDeploymentError, match=message):
        build_codex_campaign_deployment(
            config=config,
            ports=replace(ports, **{field_name: None}),
        )


def test_skill_materialization_permission_drift_fails_before_provider(
    deployment_inputs,
) -> None:
    config, ports, _calls = deployment_inputs
    skill_path = Path(ports.actor_skills_factory.inventory()[0]["path"])
    skill_path.chmod(0o600)
    try:
        with pytest.raises(CampaignDeploymentError, match="materialized bytes"):
            build_codex_campaign_deployment(config=config, ports=ports)
    finally:
        skill_path.chmod(0o400)


def test_turn_mcp_environment_inventory_drift_fails_before_provider(
    deployment_inputs,
) -> None:
    config, ports, _calls = deployment_inputs
    drifted = _TurnMCPFactory(ports.turn_mcp_factory.temp_root)
    drifted._metadata_core = {
        **drifted._metadata_core,
        "final_child_environment_names": (
            *drifted._metadata_core["final_child_environment_names"],
            "NVIDIA_INFERENCE_API_KEY",
        ),
    }
    drifted.launch_blake3 = blake3_hex(drifted._metadata_core)
    with pytest.raises(CampaignDeploymentError, match="turn MCP launch"):
        build_codex_campaign_deployment(
            config=config,
            ports=replace(ports, turn_mcp_factory=drifted),
        )


def test_turn_mcp_socket_length_commitment_drift_fails_before_provider(
    deployment_inputs,
) -> None:
    config, ports, _calls = deployment_inputs
    drifted = _TurnMCPFactory(ports.turn_mcp_factory.temp_root)
    socket_preflight = dict(drifted._metadata_core["unix_socket_preflight"])
    socket_preflight["probed_socket_path_bytes"] += 1
    drifted._metadata_core = {
        **drifted._metadata_core,
        "unix_socket_preflight": socket_preflight,
    }
    drifted.launch_blake3 = blake3_hex(drifted._metadata_core)
    with pytest.raises(CampaignDeploymentError, match="Unix socket"):
        build_codex_campaign_deployment(
            config=config,
            ports=replace(ports, turn_mcp_factory=drifted),
        )


def test_turn_mcp_private_root_mode_drift_fails_before_provider(
    deployment_inputs,
) -> None:
    config, ports, _calls = deployment_inputs
    root = ports.turn_mcp_factory.temp_root
    root.chmod(0o755)
    try:
        with pytest.raises(CampaignDeploymentError, match="private runtime root"):
            build_codex_campaign_deployment(config=config, ports=ports)
    finally:
        root.chmod(0o700)


def test_source_must_include_primary_and_ranked_reserves(deployment_inputs) -> None:
    config, ports, _calls = deployment_inputs
    source = ports.candidate_source
    source.candidate_count = 6_000
    try:
        with pytest.raises(CampaignDeploymentError, match="9,000"):
            build_codex_campaign_deployment(config=config, ports=ports)
    finally:
        source.candidate_count = 9_000


def test_provider_pool_rejects_missing_or_duplicate_routes() -> None:
    tiers = _tiers()
    with pytest.raises(CampaignDeploymentError, match="non-empty"):
        CodexProviderTiers(
            weak=(),
            middle=tiers.middle,
            strong=tiers.strong,
            judge=tiers.judge,
        )
    with pytest.raises(CampaignDeploymentError, match="assigned twice"):
        CodexProviderTiers(
            weak=tiers.weak,
            middle=tiers.middle,
            strong=(tiers.strong[0], tiers.strong[0]),
            judge=tiers.judge,
        )


def test_unavailable_canary_fails_closed_without_route_substitution(
    deployment_inputs,
) -> None:
    config, ports, _calls = deployment_inputs
    health = dict(ports.provider_health)
    prior = health["deepseek_v4_flash"]
    health["deepseek_v4_flash"] = replace(prior, status="unavailable")
    with pytest.raises(CampaignDeploymentError, match="deepseek_v4_flash"):
        build_codex_campaign_deployment(
            config=config,
            ports=replace(ports, provider_health=health),
        )


def test_selector_balances_arbitrary_pool_sizes(deployment_inputs) -> None:
    _config, ports, _calls = deployment_inputs
    tiers = _tiers()
    three_middle = replace(
        tiers,
        middle=(
            *tiers.middle,
            _route("middle_extra", Cohort.MIDDLE, "test/middle-extra"),
        ),
    )
    selector = DeterministicRouteSelector(
        candidate_ids=ports.candidate_source.candidate_ids,
        selection_blake3=ports.candidate_source.selection_blake3,
        tiers=three_middle,
    )
    assert selector.receipt.counts_by_route["weak:deepseek_v4_flash"] == 9_000
    assert "weak:qwen_3_6_27b" not in selector.receipt.counts_by_route
    assert {
        count
        for key, count in selector.receipt.counts_by_route.items()
        if key.startswith("middle:")
    } == {3_000}


def test_concurrency_is_explicit_and_has_no_hidden_small_cap() -> None:
    assert CampaignConcurrency().worker_width == 128
    assert CampaignConcurrency(worker_width=512).orchestration_config().worker_width == 512
    maximum = CampaignConcurrency.maximum_profile()
    assert (maximum.worker_width, maximum.queue_capacity, maximum.claim_batch_size) == (
        512,
        1_024,
        512,
    )
    assert maximum.app_server_shards == 64
    assert maximum.maximum_parallel_tools == 64
    assert maximum.maximum_parallel_judge_tools == 64
    with pytest.raises(CampaignDeploymentError, match="weak, middle, and strong"):
        CampaignConcurrency(maximum_parallel_models=2)


def test_structural_persistent_runner_is_accepted_without_concrete_type(
    deployment_inputs,
) -> None:
    config, ports, _calls = deployment_inputs

    class Runner:
        shard_count = 1

        def start(self):
            raise AssertionError("zero-provider build cannot start runtime")

        def run_once(self, _options, _turn):
            raise AssertionError("zero-provider build cannot run a turn")

        def close(self):
            raise AssertionError("zero-provider build cannot close runtime")

    prepared = replace(ports.runtime, runner=Runner())
    deployment = build_codex_campaign_deployment(
        config=config,
        ports=replace(ports, runtime=prepared),
    )
    assert deployment.runtime_runner is prepared.runner


def test_opus5_adapter_capacity_covers_four_turns_per_worker_without_local_429(
    deployment_inputs,
) -> None:
    config, ports, _calls = deployment_inputs
    assert ports.opus5_adapter_gateway.aggregate_max_concurrency == 4 * 128
    deployment = build_codex_campaign_deployment(config=config, ports=ports)
    assert deployment.recipe.opus5_adapter_aggregate_capacity == 512
    assert deployment.recipe.core_document()["opus5_adapter_required_capacity"] == 512
    with pytest.raises(CampaignDeploymentError, match="four concurrent turns"):
        build_codex_campaign_deployment(
            config=config,
            ports=replace(
                ports,
                opus5_adapter_gateway=_OpusAdapterGateway(capacity=511),
            ),
        )


def test_adapter_gateway_lives_around_shared_codex_runtime(
    deployment_inputs, monkeypatch
) -> None:
    config, ports, _calls = deployment_inputs
    events = []

    class Gateway(_OpusAdapterGateway):
        def start(self):
            events.append("gateway-start")

        def close(self):
            events.append("gateway-close")

    class Runner:
        shard_count = 1

        def start(self):
            events.append("runtime-start")

        def run_once(self, _options, _turn):  # pragma: no cover - no turn here
            raise AssertionError

        def close(self):
            events.append("runtime-close")

    prepared = replace(ports.runtime, runner=Runner())
    gateway = Gateway()
    deployment = build_codex_campaign_deployment(
        config=config,
        ports=replace(
            ports,
            runtime=prepared,
            opus5_adapter_gateway=gateway,
        ),
    )
    monkeypatch.setattr(
        deployment.orchestrator,
        "run_until_idle",
        lambda: events.append("orchestrator") or "complete",
    )
    required = deployment.prepared_runtime.startup_metadata[
        "process_soft_nofile_required"
    ]
    monkeypatch.setattr(
        "eva_agent.deployment.campaign.resource.getrlimit",
        lambda _kind: (required, max(required, 1_048_576)),
    )
    assert deployment.run_until_idle() == "complete"
    assert events == [
        "gateway-start",
        "runtime-start",
        "orchestrator",
        "runtime-close",
        "gateway-close",
    ]


def test_runtime_start_fails_before_claim_when_external_nofile_is_missing(
    deployment_inputs, monkeypatch
) -> None:
    config, ports, _calls = deployment_inputs
    required = DEFAULT_CHILD_SOFT_NOFILE
    metadata_core = {
        key: value
        for key, value in ports.runtime.startup_metadata.items()
        if key != "launch_blake3"
    }
    metadata_core.update(
        {
            "process_soft_nofile_observed": 1_024,
            "process_soft_nofile_required": required,
            "process_nofile_sufficient": False,
        }
    )
    prepared = replace(
        ports.runtime,
        startup_metadata=MappingProxyType(
            {**metadata_core, "launch_blake3": blake3_hex(metadata_core)}
        ),
    )
    deployment = build_codex_campaign_deployment(
        config=replace(
            config,
            concurrency=replace(
                config.concurrency,
                required_process_soft_nofile=required,
            ),
        ),
        ports=replace(ports, runtime=prepared),
    )
    monkeypatch.setattr(
        "eva_agent.deployment.campaign.resource.getrlimit",
        lambda _kind: (1_024, 1_048_576),
    )
    with pytest.raises(CampaignDeploymentError, match="external prlimit|nofile"):
        deployment.run_until_idle()


def test_prior_sealed_claim_is_removed_from_new_claim_allowlist(deployment_inputs) -> None:
    config, ports, _calls = deployment_inputs
    candidate_id = ports.candidate_source.candidate_ids[0]
    journal = ReceiptJournal(
        config.receipt_root, str(uuid5(NAMESPACE_URL, "historical-run"))
    )
    journal.start({"historical": True})
    journal.append(
        event="candidate_execution_started",
        payload={"candidate_id": candidate_id},
        worker_id=str(uuid5(NAMESPACE_URL, "historical-worker")),
    )
    journal.finalize({"complete": True})

    deployment = build_codex_campaign_deployment(config=config, ports=ports)
    assert candidate_id not in deployment.orchestrator._eligible_candidate_ids
    assert deployment.recipe.historical_exclusion_count == 1
    assert deployment.recipe.historical_attempt_evidence["attempted_candidate_ids"] == (
        candidate_id,
    )
