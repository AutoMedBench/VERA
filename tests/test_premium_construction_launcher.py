from __future__ import annotations

import base64
from dataclasses import replace
import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eva_agent.codex_runtime import (
    CodexRole,
    CodexSandbox,
    CodexThreadOptions,
    CodexToolOffer,
    CodexTurnInput,
)
from eva_agent.codex_runtime.runtime import _new_exact_construction_input_policy
from eva_agent.construction import (
    AUTHOR_QUORUM_POLICY_V1,
    CRITIC_QUORUM_POLICY_V1,
    ConstructionLane,
    ConstructionModelRoute,
    ConstructionTurnRequest,
    PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
    PremiumConstructionError,
    PremiumConstructionRoutes,
    SUPPLEMENTAL_FALLBACK_POLICY_V1,
)
from eva_agent.construction.production import (
    EXACT_MODELS,
    ExactConstructionRouteRunner,
    PremiumConstructionProductionError,
    PremiumConstructionProductionRecipe,
    _phase_plan,
    detect_exact_construction_routes,
    load_exact_construction_routes,
)
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes

import scripts.run_premium_codex_construction as launcher


class _Inner:
    shard_count = 64

    def __init__(self) -> None:
        self.options = None

    def run_once(self, options, turn_input):
        self.options = options
        return turn_input


class _Gateway:
    provider_id = "eva_opus_5"

    def __init__(self, route) -> None:
        self.planned_route = route


def test_production_recipe_binds_v3_quorums_and_exact_phase_plan() -> None:
    quorum = MappingProxyType(
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
    critic_quorum = MappingProxyType(
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
    phase_specs = (
        ("opus5_draft", 1, "opus_5", "eva_adapter_opus_5", "strong_actor"),
        ("gemini_alternate", 1, "gemini_3_1_pro", "eva_gemini_3_1_pro", "strong_actor"),
        ("opus48_critique", 2, "opus_4_8", "eva_opus_4_8", "middle_actor"),
        ("opus5_critique_backup", 2, "opus_5", "eva_adapter_opus_5", "strong_actor"),
        ("gpt56_comparison", 2, "gpt_5_6_sol", "eva_gpt_5_6_sol", "strong_actor"),
        ("gemini_comparison_backup", 2, "gemini_3_1_pro", "eva_gemini_3_1_pro", "strong_actor"),
        ("opus5_revision", 3, "opus_5", "eva_adapter_opus_5", "strong_actor"),
    )
    phase_plan = tuple(
        MappingProxyType(
            {
                "lane": lane,
                "frontier": frontier,
                "route_id": route_id,
                "model": EXACT_MODELS[route_id],
                "provider": provider,
                "role": role,
                "workspace_mode": "read-only",
                "offered_tool_count": 0,
                "semantic_attempt_count": 1,
                "semantic_retry_count": 0,
                "request_max_retries": 0,
                "stream_max_retries": 0,
            }
        )
        for lane, frontier, route_id, provider, role in phase_specs
    )
    core = {
        "schema": "eva.premium-construction-production-recipe.v3",
        "result_schema": PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3,
        "profile": 128,
        "selection_blake3": "1" * 64,
        "readiness_authority_blake3": "2" * 64,
        "queue_blake3": "3" * 64,
        "route_catalog_blake3": "4" * 64,
        "provider_canary_blake3s": MappingProxyType(
            {route_id: str(index) * 64 for index, route_id in enumerate(EXACT_MODELS, 5)}
        ),
        "codex_startup_blake3": "9" * 64,
        "opus_adapter_binding_blake3": "a" * 64,
        "opus_adapter_receipt_policy_blake3": "b" * 64,
        "opus_upstream_max_output_tokens": 32_768,
        "gpt56_model_source": "explicit_model_id_only_override",
        "author_quorum": quorum,
        "critic_quorum": critic_quorum,
        "provider_call_count": 7,
        "wave_widths": (2, 4, 1),
        "phase_plan": phase_plan,
        "phase_plan_blake3": blake3_hex(phase_plan),
        "state_root": "/tmp/eva-state",
        "workspace_root": "/tmp/eva-workspaces",
        "publication_root": "/tmp/eva-publications",
    }
    recipe = PremiumConstructionProductionRecipe(
        **core, recipe_blake3=blake3_hex(core)
    )
    assert recipe.author_quorum["primary_required_lane"] == "opus5_draft"
    assert recipe.critic_quorum["selection"] == "primary_first"
    assert recipe.provider_call_count == 7
    assert recipe.wave_widths == (2, 4, 1)
    assert tuple(row["lane"] for row in recipe.phase_plan) == tuple(
        row[0] for row in phase_specs
    )

    drifted = {**quorum, "fallback_provider_call_count": 1}
    drifted_core = {**core, "author_quorum": drifted}
    with pytest.raises(PremiumConstructionProductionError, match="recipe differs"):
        PremiumConstructionProductionRecipe(
            **drifted_core, recipe_blake3=blake3_hex(drifted_core)
        )

    provider_drift = tuple(
        {**row, "provider": "eva_wrong_provider"} if index == 3 else row
        for index, row in enumerate(phase_plan)
    )
    provider_drift_core = {
        **core,
        "phase_plan": provider_drift,
        "phase_plan_blake3": blake3_hex(provider_drift),
    }
    with pytest.raises(PremiumConstructionProductionError, match="recipe differs"):
        PremiumConstructionProductionRecipe(
            **provider_drift_core,
            recipe_blake3=blake3_hex(provider_drift_core),
        )

    old_version_core = {
        **core,
        "schema": "eva.premium-construction-production-recipe.v2",
    }
    with pytest.raises(PremiumConstructionProductionError, match="recipe differs"):
        PremiumConstructionProductionRecipe(
            **old_version_core,
            recipe_blake3=blake3_hex(old_version_core),
        )

    object.__setattr__(recipe, "schema", "eva.premium-construction-production-recipe.v2")
    with pytest.raises(PremiumConstructionProductionError, match="recipe differs"):
        recipe.to_document()


def test_production_phase_plan_is_exact_seven_lane_provider_plan() -> None:
    routes = PremiumConstructionRoutes(
        opus5=ConstructionModelRoute(
            "opus_5", EXACT_MODELS["opus_5"], "eva_adapter_opus_5"
        ),
        gemini31=ConstructionModelRoute(
            "gemini_3_1_pro",
            EXACT_MODELS["gemini_3_1_pro"],
            "eva_gemini_3_1_pro",
        ),
        opus48=ConstructionModelRoute(
            "opus_4_8", EXACT_MODELS["opus_4_8"], "eva_opus_4_8"
        ),
        gpt56=ConstructionModelRoute(
            "gpt_5_6_sol", EXACT_MODELS["gpt_5_6_sol"], "eva_gpt_5_6_sol"
        ),
    )
    plan = _phase_plan(routes)
    assert [row["lane"] for row in plan] == [
        "opus5_draft",
        "gemini_alternate",
        "opus48_critique",
        "opus5_critique_backup",
        "gpt56_comparison",
        "gemini_comparison_backup",
        "opus5_revision",
    ]
    assert [row["frontier"] for row in plan] == [1, 1, 2, 2, 2, 2, 3]
    assert [row["provider"] for row in plan] == [
        "eva_adapter_opus_5",
        "eva_gemini_3_1_pro",
        "eva_opus_4_8",
        "eva_adapter_opus_5",
        "eva_gpt_5_6_sol",
        "eva_gemini_3_1_pro",
        "eva_adapter_opus_5",
    ]
    assert all(row["offered_tool_count"] == 0 for row in plan)
    assert all(row["semantic_retry_count"] == 0 for row in plan)
    assert all(row["request_max_retries"] == 0 for row in plan)
    assert all(row["stream_max_retries"] == 0 for row in plan)


class _PolicyConsumingInner(_Inner):
    def __init__(self, policy) -> None:
        super().__init__()
        self.policy = policy

    def run_once(self, options, turn_input):
        token = self.policy.bind_fresh_thread(options)
        assert token is not None
        self.policy.consume(token, options=options, turn_input=turn_input)
        return super().run_once(options, turn_input)


def test_exact_route_override_is_narrow_and_direct_config_is_injected() -> None:
    blocked = detect_exact_construction_routes()
    assert blocked["status"] == "blocked"
    assert blocked["missing_requirements"] == [
        "MODEL_GPT_5_6_SOL=azure/openai/gpt-5.6-sol"
    ]
    ready = detect_exact_construction_routes(
        exact_gpt56_model_override=EXACT_MODELS["gpt_5_6_sol"]
    )
    assert ready["status"] == "ready"
    assert ready["provider_call_count"] == 0
    assert ready["exact_gpt56_model_source"] == "explicit_model_id_only_override"
    routes = load_exact_construction_routes(
        exact_gpt56_model_override=EXACT_MODELS["gpt_5_6_sol"]
    )
    assert {key: route.model_id for key, route in routes.items()} == dict(EXACT_MODELS)

    inner = _Inner()
    runner = ExactConstructionRouteRunner(
        inner,
        routes=routes,
        opus_gateway=_Gateway(routes["opus_5"]),
        input_policy=_new_exact_construction_input_policy(),
    )
    options = CodexThreadOptions(
        role=CodexRole.STRONG_ACTOR,
        model=EXACT_MODELS["gpt_5_6_sol"],
        provider="eva_gpt_5_6_sol",
        cwd=str(Path.cwd().resolve()),
        sandbox=CodexSandbox.READ_ONLY,
        offered_tools=(),
    )
    assert runner.run_once(options, "turn") == "turn"
    provider = inner.options.config["model_providers"]["eva_gpt_5_6_sol"]
    assert provider["request_max_retries"] == 0
    assert provider["stream_max_retries"] == 0
    assert inner.options.sandbox is CodexSandbox.READ_ONLY
    assert not inner.options.offered_tools


def test_exact_construction_runner_requires_untampered_frozen_binding() -> None:
    routes = load_exact_construction_routes(
        exact_gpt56_model_override=EXACT_MODELS["gpt_5_6_sol"]
    )
    policy = _new_exact_construction_input_policy()
    inner = _PolicyConsumingInner(policy)
    runner = ExactConstructionRouteRunner(
        inner,
        routes=routes,
        opus_gateway=_Gateway(routes["opus_5"]),
        input_policy=policy,
    )
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def request() -> ConstructionTurnRequest:
        options = CodexThreadOptions(
            role=CodexRole.STRONG_ACTOR,
            model=EXACT_MODELS["gpt_5_6_sol"],
            provider="eva_gpt_5_6_sol",
            cwd=str(Path.cwd().resolve()),
            sandbox=CodexSandbox.READ_ONLY,
            offered_tools=(),
            service_name="evamed-codex-premium-construction",
        )
        turn_input = CodexTurnInput(
            public_text="Keep the reference answer phrase as frozen source text.",
            output_schema=schema,
            sandbox=CodexSandbox.READ_ONLY,
            model=options.model,
        )
        return ConstructionTurnRequest.create(
            lane=ConstructionLane.GPT56_COMPARISON,
            options=options,
            turn_input=turn_input,
            source_request_blake3="a" * 64,
            source_phase_request={"schema": "fixture.phase.v1", "output_schema": schema},
            source_output_schema=schema,
        )

    exact = request()
    assert runner.run_construction_once(exact) is exact.turn_input

    digest_drift = request()
    object.__setattr__(digest_drift, "source_request_blake3", "b" * 64)
    with pytest.raises(PremiumConstructionError):
        runner.run_construction_once(digest_drift)

    tool_drift = request()
    tool = CodexToolOffer(
        fully_qualified_name="evamed/search",
        description="Search.",
        input_schema={"type": "object", "additionalProperties": False},
    )
    object.__setattr__(
        tool_drift,
        "options",
        replace(tool_drift.options, offered_tools=(tool,)),
    )
    with pytest.raises(PremiumConstructionError):
        runner.run_construction_once(tool_drift)

    sandbox_drift = request()
    object.__setattr__(
        sandbox_drift,
        "options",
        replace(sandbox_drift.options, sandbox=CodexSandbox.WORKSPACE_WRITE),
    )
    with pytest.raises(PremiumConstructionError):
        runner.run_construction_once(sandbox_drift)

    context_drift = request()
    object.__setattr__(
        context_drift,
        "turn_input",
        replace(context_drift.turn_input, judge_only_context={"rubric": "private"}),
    )
    with pytest.raises(PremiumConstructionError):
        runner.run_construction_once(context_drift)


class _Document:
    def __init__(self, value):
        self.value = value

    def to_document(self):
        return self.value


class _Prepared:
    def __init__(self) -> None:
        self.recipe = SimpleNamespace(
            recipe_blake3="a" * 64,
            to_document=lambda: {
                "schema": "fixture.recipe.v1",
                "recipe_blake3": "a" * 64,
            },
        )
        entry = SimpleNamespace(candidate_id="candidate-1", selection_tier="primary")
        self.campaign = SimpleNamespace(queue=SimpleNamespace(entries=(entry,)))
        self.started = 0
        self.closed = 0
        self.ran = 0

    def preflight(self, candidate_id):
        assert candidate_id == "candidate-1"
        return _Document({"schema": "fixture.preflight.v1", "provider_call_count": 0})

    def start(self):
        self.started += 1

    def run(self):
        self.ran += 1
        return _Document({"schema": "fixture.report.v1", "calls": 7})

    def close(self):
        self.closed += 1


def _fake_factory(holder, **kwargs):
    prepared = _Prepared()
    holder.append((prepared, kwargs))
    return prepared


def _host_signing_config(tmp_path: Path) -> tuple[Path, MappingProxyType]:
    key_id = "eva-test-host-v1"
    private = Ed25519PrivateKey.generate()
    private_path = (tmp_path / "host-signing.pem").resolve()
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
    trust_path = (tmp_path / "host-trust.json").resolve()
    trust_bytes = canonical_json_bytes(
        {
            "schema": "rlevo.med-research-host-trust-store.v1",
            "algorithm": "Ed25519",
            "status": "active",
            "created_at_utc": "2026-09-07T00:00:00Z",
            "keys": {key_id: base64.b64encode(public).decode("ascii")},
        }
    )
    trust_path.write_bytes(trust_bytes)
    config_path = (tmp_path / "host-signing-launch.json").resolve()
    config = MappingProxyType(
        {
            "schema": "eva.host-signing-launch.v1",
            "private_key_path": str(private_path),
            "key_id": key_id,
            "trust_store_path": str(trust_path),
            "expected_public_key_blake3": blake3_bytes(public),
            "expected_trust_store_blake3": blake3_bytes(trust_bytes),
        }
    )
    config_path.write_bytes(canonical_json_bytes(config))
    config_path.chmod(0o600)
    return config_path, config


def test_executable_preflight_and_canary_recipe_gate_without_provider(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(launcher, "RUNS_ROOT", tmp_path.resolve())
    state = tmp_path / "state"
    workspace = tmp_path / "workspace"
    publications = tmp_path / "publications"
    for path in (state, workspace, publications):
        path.mkdir()
    holder = []
    factory = lambda **kwargs: _fake_factory(holder, **kwargs)
    signing_config_path, signing_config = _host_signing_config(tmp_path)
    preflight_output = (tmp_path / "preflight.json").resolve()
    common = [
        "--profile", "128",
        "--exact-gpt56-model", EXACT_MODELS["gpt_5_6_sol"],
        "--host-signing-config", str(signing_config_path),
        "--state-root", str(state.resolve()),
        "--workspace-root", str(workspace.resolve()),
        "--publication-root", str(publications.resolve()),
    ]
    assert launcher.main(
        ["preflight", *common, "--receipt-output", str(preflight_output)],
        factory=factory,
    ) == 0
    prepared, values = holder[-1]
    assert prepared.started == prepared.ran == 0
    assert prepared.closed == 1
    assert values["max_candidates"] is None
    assert values["host_private_key_path"] == Path(signing_config["private_key_path"])
    assert values["host_key_id"] == signing_config["key_id"]
    assert values["host_trust_store_path"] == Path(signing_config["trust_store_path"])
    assert values["expected_host_public_key_blake3"] == signing_config[
        "expected_public_key_blake3"
    ]
    assert values["expected_host_trust_store_blake3"] == signing_config[
        "expected_trust_store_blake3"
    ]
    receipt = json.loads(preflight_output.read_text())
    assert receipt["schema"] == "eva.premium-construction-launch-receipt.v2"
    assert receipt["construction_result_schema"] == PREMIUM_CONSTRUCTION_RESULT_SCHEMA_V3
    assert receipt["provider_call_count_per_candidate"] == 7
    assert receipt["per_candidate_parallel_frontiers"] == [2, 4, 1]
    assert receipt["provider_call_count"] == 0
    assert receipt["all_lanes_read_only"] is True
    receipt_bytes = preflight_output.read_bytes()
    assert str(signing_config_path).encode() not in receipt_bytes
    assert signing_config["private_key_path"].encode() not in receipt_bytes
    assert signing_config["trust_store_path"].encode() not in receipt_bytes

    canary_output = (tmp_path / "canary.json").resolve()
    assert launcher.main(
        [
            "canary", *common,
            "--expected-recipe-blake3", "a" * 64,
            "--receipt-output", str(canary_output),
        ],
        factory=factory,
    ) == 0
    prepared, values = holder[-1]
    assert (prepared.started, prepared.ran, prepared.closed) == (1, 1, 1)
    assert values["max_candidates"] == 1
    canary_receipt = json.loads(canary_output.read_text())
    assert canary_receipt["campaign_report"]["calls"] == 7
    assert canary_receipt["provider_call_count_per_candidate"] == 7
    assert canary_receipt["per_candidate_parallel_frontiers"] == [2, 4, 1]

    calls_before_collision = len(holder)
    assert launcher.main(
        [
            "canary", *common,
            "--expected-recipe-blake3", "a" * 64,
            "--receipt-output", str(canary_output),
        ],
        factory=factory,
    ) == 2
    assert len(holder) == calls_before_collision
    assert json.loads(canary_output.read_text()) == canary_receipt

    next_output = (tmp_path / "next.json").resolve()
    def next_factory(**kwargs):
        prepared = _Prepared()
        prepared.campaign.queue.entries = (
            SimpleNamespace(candidate_id="candidate-1", selection_tier="primary"),
            SimpleNamespace(candidate_id="candidate-2", selection_tier="primary"),
        )
        prepared.pending_candidate_ids = lambda *, limit=None: ("candidate-2",)
        holder.append((prepared, kwargs))
        return prepared

    assert launcher.main(
        [
            "next-canary", *common,
            "--expected-recipe-blake3", "a" * 64,
            "--receipt-output", str(next_output),
        ],
        factory=next_factory,
    ) == 0
    prepared, values = holder[-1]
    assert (prepared.started, prepared.ran, prepared.closed) == (1, 1, 1)
    assert values["max_candidates"] == 1
    next_receipt = json.loads(next_output.read_text())
    assert next_receipt["max_candidates"] == 1
    assert next_receipt["scheduled_first_candidate_id"] == "candidate-2"

    production_output = (tmp_path / "production.json").resolve()
    assert launcher.main(
        [
            "production", *common,
            "--max-candidates", "17",
            "--expected-recipe-blake3", "a" * 64,
            "--receipt-output", str(production_output),
        ],
        factory=factory,
    ) == 0
    _, values = holder[-1]
    assert values["max_candidates"] == 17
    assert json.loads(production_output.read_text())["max_candidates"] == 17

    bad_output = (tmp_path / "bad.json").resolve()
    assert launcher.main(
        [
            "canary", *common,
            "--expected-recipe-blake3", "b" * 64,
            "--receipt-output", str(bad_output),
        ],
        factory=factory,
    ) == 2
    prepared, _ = holder[-1]
    assert prepared.started == prepared.ran == prepared.closed == 0
    assert not bad_output.exists()


def test_host_signing_launch_config_fails_closed_before_factory(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(launcher, "RUNS_ROOT", tmp_path.resolve())
    signing_config_path, config = _host_signing_config(tmp_path)
    state = (tmp_path / "state").resolve()
    workspace = (tmp_path / "workspace").resolve()
    publications = (tmp_path / "publications").resolve()
    for path in (state, workspace, publications):
        path.mkdir()
    common = [
        "preflight",
        "--profile",
        "128",
        "--exact-gpt56-model",
        EXACT_MODELS["gpt_5_6_sol"],
        "--host-signing-config",
        str(signing_config_path),
        "--state-root",
        str(state),
        "--workspace-root",
        str(workspace),
        "--publication-root",
        str(publications),
    ]
    holder: list[tuple[object, MappingProxyType]] = []

    signing_config_path.chmod(0o644)
    mode_output = (tmp_path / "bad-mode-receipt.json").resolve()
    assert launcher.main(
        [*common, "--receipt-output", str(mode_output)],
        factory=lambda **kwargs: _fake_factory(holder, **kwargs),
    ) == 2
    assert not holder and not mode_output.exists()

    signing_config_path.chmod(0o600)
    wrong = {**config, "expected_public_key_blake3": "f" * 64}
    signing_config_path.write_bytes(canonical_json_bytes(wrong))
    signing_config_path.chmod(0o600)
    digest_output = (tmp_path / "bad-digest-receipt.json").resolve()
    assert launcher.main(
        [*common, "--receipt-output", str(digest_output)],
        factory=lambda **kwargs: _fake_factory(holder, **kwargs),
    ) == 2
    assert not holder and not digest_output.exists()

    signing_config_path.write_bytes(canonical_json_bytes(config))
    signing_config_path.chmod(0o600)
    link_path = (tmp_path / "host-signing-link.json").resolve()
    os.symlink(signing_config_path, link_path)
    link_output = (tmp_path / "link-receipt.json").resolve()
    linked = [
        link_path.as_posix() if value == str(signing_config_path) else value
        for value in common
    ]
    assert launcher.main(
        [*linked, "--receipt-output", str(link_output)],
        factory=lambda **kwargs: _fake_factory(holder, **kwargs),
    ) == 2
    assert not holder and not link_output.exists()
