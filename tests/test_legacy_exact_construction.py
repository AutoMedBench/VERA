from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
from threading import Lock
from typing import Any
from uuid import uuid4

import pytest

from eva_agent.campaign.selection_v2 import CampaignSelectionV2
from eva_agent.codex_runtime import CodexEvent, CodexTurnReceipt
from eva_agent.construction import (
    ConstructionLane,
    ConstructionModelRoute,
    LegacyConstructionAdapterError,
    LegacySelectionV2ConstructionFactory,
    PremiumConstructionCampaign,
    PremiumCodexConstructionOrchestrator,
    PremiumConstructionCampaignConfig,
    PremiumConstructionExecutionBindingResolver,
    PremiumConstructionResult,
    PremiumConstructionSupervisorError,
    PremiumConstructionRoutes,
    build_premium_construction_queue,
    issue_premium_construction_supervisor_transition,
    preflight_premium_construction_campaign,
    verify_premium_construction_supervisor_transition,
    verify_legacy_construction_publication,
)
from eva_agent.pipeline import FilesystemSandbox, ParallelToolRuntime, ToolCall
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes
from eva_agent.pipeline.ids import DeterministicUUIDFactory
from eva_agent.rubrics import CompiledRubricRegistry, load_and_compile_registry
from eva_agent.sources.supervisor_v24_readiness import (
    SignedSupervisorV24Readiness,
    load_signed_supervisor_v24_readiness,
)


_REAL_CANDIDATE_ID = "d9f0a7d3-de40-50fa-988b-33e3200b5a6a"
_REAL_INPUT_DIRECTORY = (
    "curation/source-families-20260903/automedbench-medxpertqa/"
    "constructor-capacity-v2/inputs/"
    "evamed-automedbench-e2e-0181-1a28bbc9db-merged-cv2-467956ad"
)


def _legacy_root() -> Path:
    root = (Path(__file__).resolve().parents[2] / "rlevo-med-research").resolve()
    if not (root / "src/rlevo_med_research/candidate_construction.py").is_file():
        pytest.skip("real sibling rlevo-med-research authority is unavailable")
    return root


def _legacy_host_key_source() -> Path:
    configured = os.environ.get("EVA_TEST_HOST_SIGNING_KEY_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path.home()
        / ".config"
        / "rlevo-med-research"
        / "host-signing-ed25519-v1.pem"
    ).resolve()


@pytest.fixture(scope="module")
def signed_sources() -> tuple[CampaignSelectionV2, SignedSupervisorV24Readiness]:
    legacy = _legacy_root()
    selection_path = Path(__file__).resolve().parents[1] / "runs/campaign-selection.v2.json"
    if not selection_path.is_file():
        pytest.skip("real verified selection-v2 is unavailable")
    selection = CampaignSelectionV2.from_document(
        json.loads(selection_path.read_text(encoding="utf-8"))
    )
    readiness = load_signed_supervisor_v24_readiness(
        legacy / "runs/evamed-campaign-supervisor-v24-attempt1",
        authority_root=legacy,
        trust_store_path=legacy / "config/host-trust-store.v1.json",
        worker_width=64,
    )
    return selection, readiness


def _routes() -> PremiumConstructionRoutes:
    return PremiumConstructionRoutes(
        opus5=ConstructionModelRoute(
            "opus_5",
            "aws/anthropic/bedrock-claude-opus-5",
            "eva_adapter_opus_5",
        ),
        gemini31=ConstructionModelRoute(
            "gemini_3_1_pro",
            "gcp/google/gemini-3.1-pro-preview",
            "eva_gemini_3_1_pro",
        ),
        opus48=ConstructionModelRoute(
            "opus_4_8",
            "azure/anthropic/claude-opus-4-8",
            "eva_opus_4_8",
        ),
        gpt56=ConstructionModelRoute(
            "gpt_5_6_sol",
            "azure/openai/gpt-5.6-sol",
            "eva_gpt_5_6_sol",
        ),
    )


def _factory(
    tmp_path: Path,
    signed_sources: tuple[CampaignSelectionV2, SignedSupervisorV24Readiness],
) -> LegacySelectionV2ConstructionFactory:
    selection, readiness = signed_sources
    legacy = _legacy_root()
    return LegacySelectionV2ConstructionFactory(
        selection=selection,
        readiness=readiness,
        authority_root=legacy,
        supervisor_root=legacy / "runs/evamed-campaign-supervisor-v24-attempt1",
        trust_store_path=legacy / "config/host-trust-store.v1.json",
        legacy_package_src=legacy / "src",
        workspace_root=tmp_path / "workspaces",
        output_root=tmp_path / "published",
        id_factory=DeterministicUUIDFactory("legacy-publication-test"),
        clock=lambda: "2026-09-07T00:00:00Z",
    )


def test_real_frozen_selection_source_and_authority_request_dry_run(
    tmp_path: Path,
    signed_sources: tuple[CampaignSelectionV2, SignedSupervisorV24Readiness],
) -> None:
    factory = _factory(tmp_path, signed_sources)
    composition = factory.compose(_REAL_CANDIDATE_ID, _routes())

    source = composition.source
    assert source.source_id == _REAL_CANDIDATE_ID
    assert source.request["selection_entry"]["construction_readiness"] == "frozen"
    assert source.request["controls"]["provider_calls_during_source_resolution"] == 0
    assert source.request["controls"]["legacy_schemas_copied"] is False
    requests = composition.adapter.prepare_authors(source, _routes())
    assert tuple(request.lane for request in requests) == (
        ConstructionLane.OPUS5_DRAFT,
        ConstructionLane.GEMINI_ALTERNATE,
    )
    assert all(request.options.sandbox.value == "read-only" for request in requests)
    assert all(not request.options.offered_tools for request in requests)
    assert requests[1].options.cwd != requests[0].options.cwd
    assert all(
        request.source_output_schema_blake3
        == blake3_hex(request.turn_input.output_schema)
        for request in requests
    )
    assert list((tmp_path / "published/.claims").iterdir()) == []


def test_real_selection_candidate_campaign_preflight_is_provider_free(
    tmp_path: Path,
    signed_sources: tuple[CampaignSelectionV2, SignedSupervisorV24Readiness],
) -> None:
    selection, _readiness = signed_sources
    factory = _factory(tmp_path, signed_sources)
    queue = build_premium_construction_queue(selection)

    receipt = preflight_premium_construction_campaign(
        queue=queue,
        factory=factory,
        routes=_routes(),
        config=PremiumConstructionCampaignConfig(
            worker_width=1,
            app_server_shards=64,
        ),
        candidate_ids=(_REAL_CANDIDATE_ID,),
    )

    assert receipt.provider_call_count == 0
    assert len(receipt.candidate_proofs) == 1
    assert receipt.candidate_proofs[0]["candidate_id"] == _REAL_CANDIDATE_ID
    assert [row["model"] for row in receipt.phase_plan] == [
        "aws/anthropic/bedrock-claude-opus-5",
        "gcp/google/gemini-3.1-pro-preview",
        "azure/anthropic/claude-opus-4-8",
        "aws/anthropic/bedrock-claude-opus-5",
        "azure/openai/gpt-5.6-sol",
        "gcp/google/gemini-3.1-pro-preview",
        "aws/anthropic/bedrock-claude-opus-5",
    ]
    assert [row["provider"] for row in receipt.phase_plan] == [
        "eva_adapter_opus_5",
        "eva_gemini_3_1_pro",
        "eva_opus_4_8",
        "eva_adapter_opus_5",
        "eva_gpt_5_6_sol",
        "eva_gemini_3_1_pro",
        "eva_adapter_opus_5",
    ]
    composition = factory.compose(_REAL_CANDIDATE_ID, _routes())
    reopened = composition.adapter.prepare_authors(composition.source, _routes())
    assert [request.request_blake3 for request in reopened] == list(
        receipt.candidate_proofs[0]["author_request_blake3s"]
    )
    (Path(reopened[0].options.cwd) / "provider-write.txt").write_text("started")
    with pytest.raises(LegacyConstructionAdapterError, match="not fresh"):
        factory.compose(_REAL_CANDIDATE_ID, _routes())


def _turn_receipt(options: Any, final_response: str) -> CodexTurnReceipt:
    ids = DeterministicUUIDFactory(
        f"legacy-exact-turn:{options.model}:{blake3_hex(final_response)}"
    )
    thread_id = f"thread-{ids.new('thread')}"
    turn_id = f"turn-{ids.new('turn')}"
    event_core = {
        "event_id": ids.new("event"),
        "sequence": 0,
        "method": "turn/completed",
        "thread_id": thread_id,
        "turn_id": turn_id,
        "payload": {"status": "completed"},
        "content_redacted": False,
    }
    event = CodexEvent(**event_core, event_blake3=blake3_hex(event_core))
    core = {
        "schema": "eva.codex-turn-receipt.v1",
        "receipt_id": ids.new("receipt"),
        "runtime_thread_id": ids.new("runtime-thread"),
        "runtime_turn_id": ids.new("runtime-turn"),
        "thread_id": thread_id,
        "turn_id": turn_id,
        "role": options.role,
        "model": options.model,
        "provider": options.provider,
        "sandbox": options.sandbox,
        "thread_resumed": False,
        "visibility": "actor-public",
        "status": "completed",
        "final_response": final_response,
        "events": (event,),
        "tool_calls": (),
        "selected_skill_ids": (),
        "selected_skill_catalog_blake3": blake3_hex(()),
        "offered_mcp_tool_names": (),
        "offered_tool_schema_blake3": blake3_hex(()),
        "max_parallelism_observed": 0,
        "parallel_tool_calls_supported": True,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "input_blake3": blake3_hex({"fixture": "legacy-exact"}),
        "config_keys": options.config_keys,
        "config_values_recorded": False,
        "input_payload_recorded": False,
        "sdk_version": "fake-sdk",
        "server_version": "fake-app-server",
    }
    return CodexTurnReceipt(**core, receipt_blake3=blake3_hex(core))


class _RealOutputRunner:
    shard_count = 64

    def __init__(self, outputs: dict[str, Any]) -> None:
        self.outputs = outputs
        self.calls: Counter[str] = Counter()
        self._lock = Lock()

    def run_once(self, options: Any, _turn_input: Any) -> CodexTurnReceipt:
        with self._lock:
            self.calls[options.model] += 1
        lane = Path(options.cwd).name
        return _turn_receipt(
            options,
            canonical_json_bytes(self.outputs[lane]).decode("utf-8").rstrip("\n"),
        )


def test_fake_codex_seven_call_real_validator_and_o_excl_sink(
    tmp_path: Path,
    signed_sources: tuple[CampaignSelectionV2, SignedSupervisorV24Readiness],
) -> None:
    factory = _factory(tmp_path, signed_sources)
    routes = _routes()
    composition = factory.compose(_REAL_CANDIDATE_ID, routes)
    legacy_private = (
        _legacy_root() / _REAL_INPUT_DIRECTORY / "construction-v2/construction-private"
    )
    draft_wire = json.loads((legacy_private / "01-draft-wire.json").read_text())
    critique_wire = json.loads((legacy_private / "02-critique-wire.json").read_text())
    revision_wire = json.loads((legacy_private / "03-revision-wire.json").read_text())
    primary = composition.adapter._compile_draft(draft_wire, wrapped=False)
    alternate = primary
    candidate = composition.adapter._state.authority.candidate
    comparison = {
        "schema": "rlevo.med-research-premium-candidate-comparison.v1",
        "sandbox_id": composition.adapter._state.construction_input["sandbox"][
            "sandbox_id"
        ],
        "primary_draft_sha256": candidate.sha256_value(primary),
        "alternate_draft_sha256": candidate.sha256_value(alternate),
        "recommended_basis": "primary",
        "findings": ["The retained canonical candidate already passed exact validation."],
        "revision_directives": ["Preserve all exact bindings while addressing critique."],
        "ability_separation": {
            "lower_tier_resistance": True,
            "mid_tier_partial_credit": True,
            "upper_tier_tool_budget_robust": True,
        },
    }
    outputs = {
        "opus5_draft": draft_wire,
        "gemini_alternate": {
            "payload": {
                "candidate_json": canonical_json_bytes(draft_wire)
                .decode("utf-8")
                .rstrip("\n")
            }
        },
        "opus48_critique": critique_wire,
        "opus5_critique_backup": critique_wire,
        "gpt56_comparison": {
            "payload": {
                "comparison_json": canonical_json_bytes(comparison)
                .decode("utf-8")
                .rstrip("\n")
            }
        },
        "gemini_comparison_backup": {
            "payload": {
                "comparison_json": canonical_json_bytes(comparison)
                .decode("utf-8")
                .rstrip("\n")
            }
        },
        "opus5_revision": revision_wire,
    }
    runner = _RealOutputRunner(outputs)
    orchestrator = PremiumCodexConstructionOrchestrator(
        runner,
        routes,
        id_factory=DeterministicUUIDFactory("legacy-orchestrator-test"),
    )

    published = composition.run(orchestrator)

    assert sum(runner.calls.values()) == 7
    assert runner.calls["aws/anthropic/bedrock-claude-opus-5"] == 3
    assert runner.calls["gcp/google/gemini-3.1-pro-preview"] == 2
    assert published.result.provider_call_count == 7
    assert published.result.wave_widths == (2, 4, 1)
    assert published.publication.legacy_manifest_compatibility_verified is False
    verification = verify_legacy_construction_publication(
        tmp_path / "published", published.publication
    )
    assert verification["verified"] is True
    publication_root = (
        tmp_path / "published" / published.publication.publication_id
    )
    assert (publication_root / "policy.json").is_file()
    assert (publication_root / "contracts/s5-submission-contract-template.json").is_file()
    assert not any(path.name == "candidate-construction-manifest.json" for path in publication_root.rglob("*"))
    with pytest.raises(LegacyConstructionAdapterError, match="already exists"):
        composition.sink.publish(published.result)


def test_output_root_rejects_symlink_component(
    tmp_path: Path,
    signed_sources: tuple[CampaignSelectionV2, SignedSupervisorV24Readiness],
) -> None:
    target = tmp_path / "real-output"
    target.mkdir()
    link = tmp_path / "linked-output"
    link.symlink_to(target, target_is_directory=True)
    selection, readiness = signed_sources
    legacy = _legacy_root()
    with pytest.raises(LegacyConstructionAdapterError, match="without following links"):
        LegacySelectionV2ConstructionFactory(
            selection=selection,
            readiness=readiness,
            authority_root=legacy,
            supervisor_root=legacy / "runs/evamed-campaign-supervisor-v24-attempt1",
            trust_store_path=legacy / "config/host-trust-store.v1.json",
            legacy_package_src=legacy / "src",
            workspace_root=tmp_path / "workspaces",
            output_root=link,
        )


def _single_real_selection(selection: CampaignSelectionV2) -> CampaignSelectionV2:
    row = next(
        entry for entry in selection.entries if entry.candidate_id == _REAL_CANDIDATE_ID
    )
    core = selection.core_document()
    core.update(
        {
            "selection_id": DeterministicUUIDFactory(
                "premium-supervisor-single-selection"
            ).new("selection"),
            "selected_count": 1,
            "scheduled_count": 1,
            "reserve_count": 0,
            "entries": [
                {
                    **row.to_document(),
                    "cell_rank": 1,
                    "cell_target": 1,
                    "selection_tier": "primary",
                    "queue_ordinal": 1,
                }
            ],
        }
    )
    return CampaignSelectionV2.from_document(
        {**core, "selection_blake3": blake3_hex(core)}
    )


def _real_fake_outputs(composition: Any) -> dict[str, Any]:
    legacy_private = (
        _legacy_root() / _REAL_INPUT_DIRECTORY / "construction-v2/construction-private"
    )
    draft_wire = json.loads((legacy_private / "01-draft-wire.json").read_text())
    critique_wire = json.loads((legacy_private / "02-critique-wire.json").read_text())
    revision_wire = json.loads((legacy_private / "03-revision-wire.json").read_text())
    primary = composition.adapter._compile_draft(draft_wire, wrapped=False)
    candidate = composition.adapter._state.authority.candidate
    comparison = {
        "schema": "rlevo.med-research-premium-candidate-comparison.v1",
        "sandbox_id": composition.adapter._state.construction_input["sandbox"][
            "sandbox_id"
        ],
        "primary_draft_sha256": candidate.sha256_value(primary),
        "alternate_draft_sha256": candidate.sha256_value(primary),
        "recommended_basis": "primary",
        "findings": ["The retained canonical candidate already passed exact validation."],
        "revision_directives": ["Preserve all exact bindings while addressing critique."],
        "ability_separation": {
            "lower_tier_resistance": True,
            "mid_tier_partial_credit": True,
            "upper_tier_tool_budget_robust": True,
        },
    }
    return {
        "opus5_draft": draft_wire,
        "gemini_alternate": {
            "payload": {
                "candidate_json": canonical_json_bytes(draft_wire)
                .decode("utf-8")
                .rstrip("\n")
            }
        },
        "opus48_critique": critique_wire,
        "opus5_critique_backup": critique_wire,
        "gpt56_comparison": {
            "payload": {
                "comparison_json": canonical_json_bytes(comparison)
                .decode("utf-8")
                .rstrip("\n")
            }
        },
        "gemini_comparison_backup": {
            "payload": {
                "comparison_json": canonical_json_bytes(comparison)
                .decode("utf-8")
                .rstrip("\n")
            }
        },
        "opus5_revision": revision_wire,
    }


def test_critic_backups_are_exact_model_only_request_projections(
    tmp_path: Path,
    signed_sources: tuple[CampaignSelectionV2, SignedSupervisorV24Readiness],
) -> None:
    """Backups change identity, never the canonical role prompt or schema."""

    factory = _factory(tmp_path, signed_sources)
    routes = _routes()
    composition = factory.compose(_REAL_CANDIDATE_ID, routes)
    outputs = _real_fake_outputs(composition)
    critics = composition.adapter.prepare_critic_quorum(
        composition.source,
        routes,
        primary_draft=outputs["opus5_draft"],
        alternate_draft=outputs["gemini_alternate"],
    )
    assert tuple(request.lane for request in critics) == (
        ConstructionLane.OPUS48_CRITIQUE,
        ConstructionLane.OPUS5_CRITIQUE_BACKUP,
        ConstructionLane.GPT56_COMPARISON,
        ConstructionLane.GEMINI_COMPARISON_BACKUP,
    )

    by_lane = {request.lane: request for request in critics}
    for primary_lane, backup_lane in (
        (
            ConstructionLane.OPUS48_CRITIQUE,
            ConstructionLane.OPUS5_CRITIQUE_BACKUP,
        ),
        (
            ConstructionLane.GPT56_COMPARISON,
            ConstructionLane.GEMINI_COMPARISON_BACKUP,
        ),
    ):
        primary = by_lane[primary_lane]
        backup = by_lane[backup_lane]
        with composition.adapter._plan_lock:
            primary_phase = dict(
                composition.adapter._plans[primary.request_blake3].phase_request
            )
            backup_phase = dict(
                composition.adapter._plans[backup.request_blake3].phase_request
            )
        assert primary_phase.pop("model") == routes.for_lane(primary_lane).model
        assert backup_phase.pop("model") == routes.for_lane(backup_lane).model
        assert canonical_json_bytes(primary_phase) == canonical_json_bytes(backup_phase)
        assert primary.turn_input.public_text == backup.turn_input.public_text
        assert canonical_json_bytes(primary.turn_input.output_schema) == (
            canonical_json_bytes(backup.turn_input.output_schema)
        )
        assert primary.source_output_schema_blake3 == (
            backup.source_output_schema_blake3
        )
        assert primary.source_phase_request_blake3 != (
            backup.source_phase_request_blake3
        )
        assert primary.request_blake3 != backup.request_blake3
        assert primary.options.model != backup.options.model
        assert primary.options.provider != backup.options.provider
        assert primary.options.cwd != backup.options.cwd


def test_invalid_primary_critics_use_valid_backups_and_retain_all_evidence(
    tmp_path: Path,
    signed_sources: tuple[CampaignSelectionV2, SignedSupervisorV24Readiness],
) -> None:
    factory = _factory(tmp_path, signed_sources)
    routes = _routes()
    composition = factory.compose(_REAL_CANDIDATE_ID, routes)
    outputs = _real_fake_outputs(composition)
    # Both calls still return completed transport receipts.  The critique is
    # rejected by its exact output schema and the comparison by the unchanged
    # legacy comparison compiler, matching two independent role failures.
    outputs["opus48_critique"] = {}
    outputs["gpt56_comparison"] = {"payload": {"comparison_json": "{}"}}
    runner = _RealOutputRunner(outputs)
    published = composition.run(
        PremiumCodexConstructionOrchestrator(
            runner,
            routes,
            id_factory=DeterministicUUIDFactory("legacy-critic-quorum-test"),
        )
    )

    result = published.result
    assert sum(runner.calls.values()) == 7
    assert tuple(failure.lane for failure in result.supplemental_failures) == (
        ConstructionLane.OPUS48_CRITIQUE,
        ConstructionLane.GPT56_COMPARISON,
    )
    assert tuple(failure.error_code for failure in result.supplemental_failures) == (
        "output_schema",
        "adapter_validation",
    )
    phases = {phase.receipt.lane: phase for phase in result.phases}
    assert result.selected_critic_receipt_blake3s == {
        "critique": phases[
            ConstructionLane.OPUS5_CRITIQUE_BACKUP
        ].receipt.receipt_blake3,
        "comparison": phases[
            ConstructionLane.GEMINI_COMPARISON_BACKUP
        ].receipt.receipt_blake3,
    }
    assert ConstructionLane.OPUS5_REVISION in phases
    assert result.fallbacks == ()

    publication_root = tmp_path / "published" / published.publication.publication_id
    for failed_lane in (
        ConstructionLane.OPUS48_CRITIQUE,
        ConstructionLane.GPT56_COMPARISON,
    ):
        root = publication_root / "construction-private" / failed_lane.value
        assert (root / "supplemental-failure-receipt.json").is_file()
        assert not (root / "phase-receipt.json").exists()
        assert not (root / "wire-output.json").exists()
    for backup_lane in (
        ConstructionLane.OPUS5_CRITIQUE_BACKUP,
        ConstructionLane.GEMINI_COMPARISON_BACKUP,
    ):
        root = publication_root / "construction-private" / backup_lane.value
        assert (root / "phase-receipt.json").is_file()
        assert (root / "wire-output.json").is_file()
        assert (root / "compiled-output.json").is_file()
    binding = json.loads(
        (publication_root / "executable-binding.v1.json").read_text()
    )
    assert binding["controls"]["provider_call_count"] == 7
    assert binding["controls"]["supplemental_failure_count"] == 2
    assert binding["controls"]["deterministic_fallback_count"] == 0
    assert verify_legacy_construction_publication(
        tmp_path / "published", published.publication
    )["verified"] is True


@pytest.mark.parametrize(
    ("schema", "author_policy"),
    (
        ("eva.premium-codex-construction.v1", None),
        (
            "eva.premium-codex-construction.v2",
            "opus5_primary_required_gemini_supplemental_1_of_2_v1",
        ),
    ),
)
def test_legacy_v1_and_v2_results_still_reopen_and_publish_exactly(
    tmp_path: Path,
    signed_sources: tuple[CampaignSelectionV2, SignedSupervisorV24Readiness],
    schema: str,
    author_policy: str | None,
) -> None:
    factory = _factory(tmp_path, signed_sources)
    routes = _routes()
    composition = factory.compose(_REAL_CANDIDATE_ID, routes)
    current = PremiumCodexConstructionOrchestrator(
        _RealOutputRunner(_real_fake_outputs(composition)),
        routes,
        id_factory=DeterministicUUIDFactory(f"legacy-{schema}-reopen-source"),
    ).construct(composition.source, composition.adapter)
    old_lanes = {
        ConstructionLane.OPUS5_DRAFT,
        ConstructionLane.GEMINI_ALTERNATE,
        ConstructionLane.OPUS48_CRITIQUE,
        ConstructionLane.GPT56_COMPARISON,
        ConstructionLane.OPUS5_REVISION,
    }
    old_phases = tuple(
        phase for phase in current.phases if phase.receipt.lane in old_lanes
    )
    core: dict[str, Any] = {
        "schema": schema,
        "construction_run_id": current.construction_run_id,
        "source_id": current.source_id,
        "source_request_blake3": current.source_request_blake3,
        "phase_receipt_blake3s": tuple(
            phase.receipt.receipt_blake3 for phase in old_phases
        ),
        "canonical_output_blake3": current.canonical_output_blake3,
        "provider_call_count": 5,
        "wave_widths": (2, 2, 1),
        "semantic_retry_count": 0,
        "legacy_manifest_emitted": False,
        "legacy_manifest_compatibility_claimed": False,
    }
    if schema.endswith(".v2"):
        core.update(
            {
                "supplemental_failure_receipt_blake3s": (),
                "fallback_receipt_blake3s": (),
                "author_quorum_policy": author_policy,
                "successful_phase_count": 5,
                "supplemental_failure_count": 0,
                "fallback_count": 0,
            }
        )
    old = PremiumConstructionResult(
        schema=schema,
        construction_run_id=current.construction_run_id,
        source_id=current.source_id,
        source_request_blake3=current.source_request_blake3,
        phases=old_phases,
        canonical_output_blake3=current.canonical_output_blake3,
        provider_call_count=5,
        wave_widths=(2, 2, 1),
        semantic_retry_count=0,
        legacy_manifest_emitted=False,
        legacy_manifest_compatibility_claimed=False,
        receipt_blake3=blake3_hex(core),
        author_quorum_policy=author_policy,
    )

    reopened = composition.adapter.reopen_result(
        composition.source,
        old,
        routes,
        completed_at_utc="2026-09-07T00:00:00Z",
    )
    assert set(reopened.phase_requests) == {lane.value for lane in old_lanes}
    published = composition.sink.publish(old)
    assert verify_legacy_construction_publication(
        tmp_path / "published", published
    )["verified"] is True


def test_invalid_gemini_author_is_preserved_and_primary_fallback_reopens_exactly(
    tmp_path: Path,
    signed_sources: tuple[CampaignSelectionV2, SignedSupervisorV24Readiness],
) -> None:
    factory = _factory(tmp_path, signed_sources)
    routes = _routes()
    composition = factory.compose(_REAL_CANDIDATE_ID, routes)
    outputs = _real_fake_outputs(composition)
    # This satisfies the unchanged shallow Gemini transport schema, but is
    # not a valid legacy candidate draft.  It therefore exercises the exact
    # adapter-validation failure observed in campaign evidence.
    outputs["gemini_alternate"] = {"payload": {"candidate_json": "{}"}}
    runner = _RealOutputRunner(outputs)
    published = composition.run(
        PremiumCodexConstructionOrchestrator(
            runner,
            routes,
            id_factory=DeterministicUUIDFactory("legacy-fallback-test"),
        )
    )

    result = published.result
    assert sum(runner.calls.values()) == 7
    assert tuple(phase.receipt.lane for phase in result.phases) == (
        ConstructionLane.OPUS5_DRAFT,
        ConstructionLane.OPUS48_CRITIQUE,
        ConstructionLane.OPUS5_CRITIQUE_BACKUP,
        ConstructionLane.GPT56_COMPARISON,
        ConstructionLane.GEMINI_COMPARISON_BACKUP,
        ConstructionLane.OPUS5_REVISION,
    )
    assert result.supplemental_failures[0].error_code == "adapter_validation"
    assert result.fallbacks[0].receipt.provider_call_count == 0
    publication_root = tmp_path / "published" / published.publication.publication_id
    gemini_root = publication_root / "construction-private/gemini_alternate"
    assert (gemini_root / "supplemental-failure-receipt.json").is_file()
    assert (gemini_root / "deterministic-fallback-receipt.json").is_file()
    assert not (gemini_root / "phase-receipt.json").exists()
    assert not (gemini_root / "codex-turn-receipt.json").exists()
    binding = json.loads(
        (publication_root / "executable-binding.v1.json").read_text()
    )
    assert binding["controls"]["successful_phase_count"] == 6
    assert binding["controls"]["supplemental_failure_count"] == 1
    assert binding["controls"]["deterministic_fallback_count"] == 1
    assert verify_legacy_construction_publication(
        tmp_path / "published", published.publication
    )["verified"] is True


def test_fake_construction_catalog_becomes_signed_executable_ready_binding(
    tmp_path: Path,
    signed_sources: tuple[CampaignSelectionV2, SignedSupervisorV24Readiness],
) -> None:
    full_selection, readiness = signed_sources
    selection = _single_real_selection(full_selection)
    legacy = _legacy_root()
    host_key_source = _legacy_host_key_source()
    if not host_key_source.is_file() or host_key_source.is_symlink():
        pytest.skip("real host signing key is unavailable")
    host_key = (tmp_path / "host-signing-ed25519-v1.pem").resolve()
    shutil.copyfile(host_key_source, host_key)
    host_key.chmod(0o600)
    published_root = tmp_path / "published"
    factory = LegacySelectionV2ConstructionFactory(
        selection=selection,
        readiness=readiness,
        authority_root=legacy,
        supervisor_root=legacy / "runs/evamed-campaign-supervisor-v24-attempt1",
        trust_store_path=legacy / "config/host-trust-store.v1.json",
        legacy_package_src=legacy / "src",
        workspace_root=tmp_path / "construction-workspaces",
        output_root=published_root,
        id_factory=DeterministicUUIDFactory("premium-supervisor-publication"),
        clock=lambda: "2026-09-07T00:00:00Z",
    )
    composition = factory.compose(_REAL_CANDIDATE_ID, _routes())
    runner = _RealOutputRunner(_real_fake_outputs(composition))
    queue = build_premium_construction_queue(selection)
    campaign = PremiumConstructionCampaign(
        queue=queue,
        factory=factory,
        runner=runner,
        routes=_routes(),
        state_root=tmp_path / "construction-state",
        config=PremiumConstructionCampaignConfig(
            worker_width=1,
            app_server_shards=64,
        ),
        id_factory=DeterministicUUIDFactory("premium-supervisor-campaign"),
        clock=lambda: "2026-09-07T00:00:00Z",
    )
    report = campaign.run()
    assert report.succeeded_this_session == 1
    assert sum(runner.calls.values()) == 7
    fake_calls_before_bridge = dict(runner.calls)
    rubric_path = Path(__file__).resolve().parents[1] / (
        "rubrics/source/domain-stage-tables.v1.json"
    )
    rubrics = load_and_compile_registry(rubric_path)
    source_files = sorted(
        path for path in published_root.rglob("*") if path.is_file()
    )
    source_before = {path: path.read_bytes() for path in source_files}
    immutable_source_paths = sorted(
        path
        for root in (
            Path(__file__).resolve().parents[1] / "schemas",
            Path(__file__).resolve().parents[1] / "rubrics/source",
        )
        for path in root.rglob("*")
        if path.is_file()
    )
    immutable_source_before = {
        path: (path.stat().st_mode, path.read_bytes()) for path in immutable_source_paths
    }
    ledgers = [
        Path(__file__).resolve().parents[1] / "runs/campaign.sqlite3",
        Path(__file__).resolve().parents[1] / "runs/campaign-v2.sqlite3",
    ]
    ledger_before = {
        path: path.read_bytes() if path.exists() else None for path in ledgers
    }
    stale_rubrics = CompiledRubricRegistry(
        json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "data/generated/compiled-rubric-registry.v1.json"
            ).read_text(encoding="utf-8")
        )
    )
    with pytest.raises(PremiumConstructionSupervisorError, match="authority differs"):
        issue_premium_construction_supervisor_transition(
            report.proof_catalog,
            queue=queue,
            selection=selection,
            rubrics=stale_rubrics,
            publication_output_root=published_root,
            transition_output_root=tmp_path / "rubric-mismatch-transition",
            legacy_python_root=legacy / "src",
            trust_store_path=legacy / "config/host-trust-store.v1.json",
            image_refs_path=legacy / "config/image-refs.v1.json",
            host_private_key_path=host_key,
            host_key_id="rlevo-host-20260902-v1",
        )
    assert not (tmp_path / "rubric-mismatch-transition").exists()
    assert dict(runner.calls) == fake_calls_before_bridge

    transition = issue_premium_construction_supervisor_transition(
        report.proof_catalog,
        queue=queue,
        selection=selection,
        rubrics=rubrics,
        publication_output_root=published_root,
        transition_output_root=tmp_path / "transitions",
        legacy_python_root=legacy / "src",
        trust_store_path=legacy / "config/host-trust-store.v1.json",
        image_refs_path=legacy / "config/image-refs.v1.json",
        host_private_key_path=host_key,
        host_key_id="rlevo-host-20260902-v1",
        id_factory=DeterministicUUIDFactory("premium-supervisor-transition"),
        clock=lambda: "2026-09-07T00:00:00Z",
    )
    assert transition.binding_count == 1
    assert transition.envelope.payload["from_state"] == "frozen"
    assert transition.envelope.payload["to_state"] == "executable_ready"
    assert transition.envelope.payload["controls"]["admission_eligibility_claimed"] is False
    assert transition.envelope.payload["bindings"][0]["rubric_id"] == (
        selection.entries[0].rubric_id
    )
    assert transition.envelope.payload["bindings"][0]["stage"] == (
        selection.entries[0].stage
    )
    reopened = verify_premium_construction_supervisor_transition(
        transition.root,
        publication_output_root=published_root,
        rubrics=rubrics,
        legacy_python_root=legacy / "src",
        trust_store_path=legacy / "config/host-trust-store.v1.json",
        image_refs_path=legacy / "config/image-refs.v1.json",
        host_private_key_path=host_key,
        host_key_id="rlevo-host-20260902-v1",
    )
    assert reopened.transition_blake3 == transition.transition_blake3
    resolver = PremiumConstructionExecutionBindingResolver(
        transition_root=transition.root,
        publication_output_root=published_root,
        rubrics=rubrics,
        legacy_python_root=legacy / "src",
        runtime_state_root=tmp_path / "runtime-state",
        trust_store_path=legacy / "config/host-trust-store.v1.json",
        image_refs_path=legacy / "config/image-refs.v1.json",
        host_private_key_path=host_key,
        host_key_id="rlevo-host-20260902-v1",
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent_bindings = tuple(
            pool.map(
                lambda _index: resolver.resolve(
                    _REAL_CANDIDATE_ID,
                    source_candidate_id=selection.entries[0].source_candidate_id,
                ),
                range(8),
            )
        )
    binding = concurrent_bindings[0]
    assert all(candidate is binding for candidate in concurrent_bindings)
    assert resolver.executable_candidate_count == 1
    assert binding.rubric_blake3 == selection.entries[0].rubric_blake3
    assert binding.stage.value == selection.entries[0].stage
    source_policy = json.loads(binding.source_policy_path.read_text(encoding="utf-8"))
    expected_schemas = {
        row["name"]: (row["description"], row["input_schema"])
        for row in source_policy["tools"]
    }
    actual_schemas = {
        row["function"]["name"]: (
            row["function"]["description"],
            row["function"]["parameters"],
        )
        for row in binding.tool_registry.public_schemas()
    }
    assert canonical_json_bytes(actual_schemas) == canonical_json_bytes(expected_schemas)
    context = binding.public_runtime_context
    historical_plan = legacy / _REAL_INPUT_DIRECTORY / (
        "construction-v2/construction-private/04-solver-rollout/"
        "stages/s1-attempt-1/submitted-plan.json"
    )
    plan = json.loads(historical_plan.read_text(encoding="utf-8"))
    plan["episode_id"] = context["s1_plan_contract"]["episode_id"]
    plan_contract = json.loads(canonical_json_bytes(context["s1_plan_contract"]))
    plan["contract_sha256"] = composition.adapter._state.authority.candidate.sha256_value(
        plan_contract
    )
    ids = DeterministicUUIDFactory("premium-supervisor-real-handler")
    sandbox = FilesystemSandbox(
        tmp_path / "rollout-workspaces",
        ids.new("sandbox"),
        binding.initial_workspace_files,
    )
    runtime = ParallelToolRuntime(
        workspace=sandbox,
        registry=binding.tool_registry,
        id_factory=ids,
        maximum_parallel_calls=64,
    )
    materialized = runtime.execute(
        [ToolCall(ids.new("call"), "materialize_plan", plan)]
    )
    evidence_id = context["s2_evidence_contract"]["evidence_objects"][0]["evidence_id"]
    retrieved = runtime.execute(
        [
            ToolCall(
                ids.new("call"),
                "retrieve_frozen_evidence",
                {"evidence_id": evidence_id},
            )
        ]
    )
    assert materialized[0].status == retrieved[0].status == "completed"
    assert materialized[0].output["gate_passed"] is True
    assert retrieved[0].output["gate_passed"] is True
    assert dict(runner.calls) == fake_calls_before_bridge
    assert {path: path.read_bytes() for path in source_files} == source_before
    assert {
        path: (path.stat().st_mode, path.read_bytes()) for path in immutable_source_paths
    } == immutable_source_before
    assert {
        path: path.read_bytes() if path.exists() else None for path in ledgers
    } == ledger_before
    with pytest.raises((PremiumConstructionSupervisorError, LegacyConstructionAdapterError)):
        issue_premium_construction_supervisor_transition(
            report.proof_catalog,
            queue=queue,
            selection=selection,
            rubrics=rubrics,
            publication_output_root=published_root,
            transition_output_root=tmp_path / "transitions",
            legacy_python_root=legacy / "src",
            trust_store_path=legacy / "config/host-trust-store.v1.json",
            image_refs_path=legacy / "config/image-refs.v1.json",
            host_private_key_path=host_key,
            host_key_id="rlevo-host-20260902-v1",
            id_factory=DeterministicUUIDFactory("premium-supervisor-transition-retry"),
            clock=lambda: "2026-09-07T00:00:00Z",
        )
    with pytest.raises(PremiumConstructionSupervisorError):
        resolver.resolve(str(uuid4()), source_candidate_id=selection.entries[0].source_candidate_id)
    tampered_root = tmp_path / "tampered" / transition.root.name
    shutil.copytree(transition.root, tampered_root)
    signed_transition = tampered_root / "signed-transition.v1.json"
    tampered = bytearray(signed_transition.read_bytes())
    tampered[-2] ^= 1
    signed_transition.chmod(0o600)
    signed_transition.write_bytes(tampered)
    signed_transition.chmod(0o400)
    with pytest.raises(ValueError):
        verify_premium_construction_supervisor_transition(
            tampered_root,
            publication_output_root=published_root,
            rubrics=rubrics,
            legacy_python_root=legacy / "src",
            trust_store_path=legacy / "config/host-trust-store.v1.json",
            image_refs_path=legacy / "config/image-refs.v1.json",
            host_private_key_path=host_key,
            host_key_id="rlevo-host-20260902-v1",
        )
    source_policy_path = binding.source_policy_path
    source_policy_bytes = source_policy_path.read_bytes()
    source_policy_path.chmod(0o600)
    source_policy_path.write_bytes(source_policy_bytes + b" ")
    source_policy_path.chmod(0o400)
    with pytest.raises(ValueError):
        verify_premium_construction_supervisor_transition(
            transition.root,
            publication_output_root=published_root,
            rubrics=rubrics,
            legacy_python_root=legacy / "src",
            trust_store_path=legacy / "config/host-trust-store.v1.json",
            image_refs_path=legacy / "config/image-refs.v1.json",
            host_private_key_path=host_key,
            host_key_id="rlevo-host-20260902-v1",
        )
