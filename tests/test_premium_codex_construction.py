from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from threading import Barrier, Lock
import time
from types import SimpleNamespace
from typing import Any, Mapping
from uuid import uuid4

import pytest

from eva_agent.codex_runtime import (
    CodexEvent,
    CodexRole,
    CodexSandbox,
    CodexThreadOptions,
    CodexTurnInput,
    CodexTurnReceipt,
)
from eva_agent.construction import (
    ConstructionFallbackReceipt,
    ConstructionFallbackResult,
    ConstructionJob,
    ConstructionLane,
    ConstructionModelRoute,
    ConstructionPublicationReceipt,
    ConstructionTurnRequest,
    ConstructionWaveError,
    FrozenConstructionSource,
    PremiumCodexConstructionOrchestrator,
    PremiumConstructionError,
    PremiumConstructionResult,
    PremiumConstructionRoutes,
    PublishedConstructionResult,
    verify_construction_failure_receipt,
    verify_construction_fallback_result,
    verify_premium_construction_result,
)
from eva_agent.pipeline.digests import blake3_hex
from eva_agent.pipeline.ids import DeterministicUUIDFactory
from eva_agent.construction.premium_codex import ConstructionAttemptEvidence
from eva_agent.construction.premium_codex import (
    UNIQUE_SCHEMA_RESPONSE_EXTRACTION_POLICY_V2,
    extract_unique_schema_valid_payload,
)


def _routes() -> PremiumConstructionRoutes:
    return PremiumConstructionRoutes(
        opus5=ConstructionModelRoute("opus_5", "anthropic/claude-opus-5", "opus"),
        gemini31=ConstructionModelRoute(
            "gemini_3_1_pro", "google/gemini-3.1-pro-preview", "gemini"
        ),
        opus48=ConstructionModelRoute(
            "opus_4_8", "anthropic/claude-opus-4.8", "opus"
        ),
        gpt56=ConstructionModelRoute("gpt_5_6_sol", "openai/gpt-5.6-sol", "openai"),
    )


def test_routes_compose_structurally_from_deployment_targets() -> None:
    expected = _routes()
    targets = tuple(
        SimpleNamespace(
            route_id=route.route_id,
            target=SimpleNamespace(model_id=route.model, provider=route.provider),
        )
        for route in (expected.opus5, expected.gemini31, expected.opus48, expected.gpt56)
    )

    assert PremiumConstructionRoutes.from_targets(targets) == expected


def _schema(lane: ConstructionLane) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "lane", "value"],
        "properties": {
            "schema": {"const": "fixture.construction-output.v1"},
            "lane": {"const": lane.value},
            "value": {"type": "string", "minLength": 1},
        },
    }


_ROLES = {
    ConstructionLane.OPUS5_DRAFT: CodexRole.STRONG_ACTOR,
    ConstructionLane.GEMINI_ALTERNATE: CodexRole.STRONG_ACTOR,
    ConstructionLane.OPUS48_CRITIQUE: CodexRole.MIDDLE_ACTOR,
    ConstructionLane.OPUS5_CRITIQUE_BACKUP: CodexRole.STRONG_ACTOR,
    ConstructionLane.GPT56_COMPARISON: CodexRole.STRONG_ACTOR,
    ConstructionLane.GEMINI_COMPARISON_BACKUP: CodexRole.STRONG_ACTOR,
    ConstructionLane.OPUS5_REVISION: CodexRole.STRONG_ACTOR,
}


class _Adapter:
    def __init__(
        self,
        root: Path,
        *,
        bad_critic_dependencies: bool = False,
        shared_root: bool = False,
    ) -> None:
        self.root = root
        self.bad_critic_dependencies = bad_critic_dependencies
        self.shared_root = shared_root
        self.validated: list[ConstructionLane] = []
        self.source_schema_digests: dict[ConstructionLane, str] = {}

    def _request(
        self,
        source: FrozenConstructionSource,
        routes: PremiumConstructionRoutes,
        lane: ConstructionLane,
        dependencies: Mapping[str, str],
        *,
        route_override=None,
    ) -> ConstructionTurnRequest:
        schema = _schema(lane)
        self.source_schema_digests[lane] = blake3_hex(schema)
        route = route_override or routes.for_lane(lane)
        sandbox = CodexSandbox.READ_ONLY
        options = CodexThreadOptions(
            role=_ROLES[lane],
            model=route.model,
            provider=route.provider,
            cwd=str(
                (
                    self.root / "shared"
                    if self.shared_root
                    else self.root / lane.value
                ).resolve()
            ),
            sandbox=sandbox,
            base_instructions="Use the exact frozen medical construction contract.",
        )
        turn_input = CodexTurnInput(
            public_text=f"Execute the {lane.value} construction phase.",
            public_context={"dependencies": dict(dependencies)},
            output_schema=schema,
        )
        phase_request = {
            "schema": "fixture.frozen-phase-request.v1",
            "lane": lane.value,
            "output_schema": schema,
            "dependencies": dict(dependencies),
        }
        return ConstructionTurnRequest.create(
            lane=lane,
            options=options,
            turn_input=turn_input,
            source_request_blake3=source.request_blake3,
            source_phase_request=phase_request,
            source_output_schema=phase_request["output_schema"],
            dependencies=dependencies,
        )

    @staticmethod
    def _author_dependencies(primary: Mapping[str, Any], alternate: Mapping[str, Any]) -> dict[str, str]:
        return {
            ConstructionLane.OPUS5_DRAFT.value: blake3_hex(primary),
            ConstructionLane.GEMINI_ALTERNATE.value: blake3_hex(alternate),
        }

    def prepare_authors(self, source, routes):
        return tuple(self._request(source, routes, lane, {}) for lane in (
            ConstructionLane.OPUS5_DRAFT,
            ConstructionLane.GEMINI_ALTERNATE,
        ))

    def prepare_critics(
        self, source, routes, *, primary_draft, alternate_draft
    ):
        dependencies = self._author_dependencies(primary_draft, alternate_draft)
        if self.bad_critic_dependencies:
            dependencies = {**dependencies, "unexpected": "0" * 64}
        return tuple(self._request(source, routes, lane, dependencies) for lane in (
            ConstructionLane.OPUS48_CRITIQUE,
            ConstructionLane.GPT56_COMPARISON,
        ))

    def prepare_critic_quorum(
        self, source, routes, *, primary_draft, alternate_draft
    ):
        dependencies = self._author_dependencies(primary_draft, alternate_draft)
        if self.bad_critic_dependencies:
            dependencies = {**dependencies, "unexpected": "0" * 64}
        return tuple(self._request(source, routes, lane, dependencies) for lane in (
            ConstructionLane.OPUS48_CRITIQUE,
            ConstructionLane.OPUS5_CRITIQUE_BACKUP,
            ConstructionLane.GPT56_COMPARISON,
            ConstructionLane.GEMINI_COMPARISON_BACKUP,
        ))

    def prepare_critic_authority_v4(
        self, source, routes, *, primary_draft, alternate_draft
    ):
        dependencies = self._author_dependencies(primary_draft, alternate_draft)
        return (
            self._request(source, routes, ConstructionLane.OPUS48_CRITIQUE, dependencies),
            self._request(source, routes, ConstructionLane.OPUS5_CRITIQUE_BACKUP, dependencies),
            self._request(source, routes, ConstructionLane.GPT56_COMPARISON, dependencies),
            self._request(
                source,
                routes,
                ConstructionLane.GEMINI_COMPARISON_BACKUP,
                dependencies,
                route_override=routes.opus5,
            ),
        )

    def prepare_revision(
        self,
        source,
        routes,
        *,
        primary_draft,
        alternate_draft,
        opus48_critique,
        gpt56_comparison,
        selected_critique_lane=ConstructionLane.OPUS48_CRITIQUE,
        selected_comparison_lane=ConstructionLane.GPT56_COMPARISON,
    ):
        dependencies = {
            **self._author_dependencies(primary_draft, alternate_draft),
            ConstructionLane.OPUS48_CRITIQUE.value: blake3_hex(opus48_critique),
            ConstructionLane.GPT56_COMPARISON.value: blake3_hex(gpt56_comparison),
        }
        return self._request(
            source, routes, ConstructionLane.OPUS5_REVISION, dependencies
        )

    def validate_output(self, _source, request, output) -> None:
        assert output["lane"] == request.lane.value
        self.validated.append(request.lane)

    def resolve_supplemental_author_fallback(
        self, _source, request, *, primary_draft, failure
    ):
        assert request.lane is ConstructionLane.GEMINI_ALTERNATE
        assert failure.request_blake3 == request.request_blake3
        return {
            "schema": "fixture.construction-output.v1",
            "lane": ConstructionLane.GEMINI_ALTERNATE.value,
            "value": f"canonical-primary:{blake3_hex(primary_draft)}",
        }


def _turn_receipt(
    options: CodexThreadOptions,
    *,
    final_response: str,
    status: str = "completed",
) -> CodexTurnReceipt:
    thread_id = f"thread-{uuid4()}"
    turn_id = f"turn-{uuid4()}"
    event_core = {
        "event_id": str(uuid4()),
        "sequence": 0,
        "method": "turn/completed",
        "thread_id": thread_id,
        "turn_id": turn_id,
        "payload": {"status": status},
        "content_redacted": False,
    }
    event = CodexEvent(**event_core, event_blake3=blake3_hex(event_core))
    core = {
        "schema": "eva.codex-turn-receipt.v1",
        "receipt_id": str(uuid4()),
        "runtime_thread_id": str(uuid4()),
        "runtime_turn_id": str(uuid4()),
        "thread_id": thread_id,
        "turn_id": turn_id,
        "role": options.role,
        "model": options.model,
        "provider": options.provider,
        "sandbox": options.sandbox,
        "thread_resumed": False,
        "visibility": "actor-public",
        "status": status,
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
        "input_blake3": blake3_hex({"fixture": "input"}),
        "config_keys": options.config_keys,
        "config_values_recorded": False,
        "input_payload_recorded": False,
        "sdk_version": "fake-sdk",
        "server_version": "fake-app-server",
    }
    return CodexTurnReceipt(**core, receipt_blake3=blake3_hex(core))


class _Runner:
    def __init__(
        self,
        *,
        malformed_lane: ConstructionLane | None = None,
        malformed_lanes: set[ConstructionLane] | None = None,
        fenced_lanes: set[ConstructionLane] | None = None,
    ) -> None:
        self.malformed_lanes = set(malformed_lanes or ())
        self.fenced_lanes = set(fenced_lanes or ())
        if malformed_lane is not None:
            self.malformed_lanes.add(malformed_lane)
        self.calls: Counter[ConstructionLane] = Counter()
        self.active = 0
        self.maximum = 0
        self.lock = Lock()
        self.barriers = {
            "author": Barrier(2),
            "critic": Barrier(4),
        }

    def run_once(self, options, turn_input):
        lane = ConstructionLane(turn_input.output_schema["properties"]["lane"]["const"])
        with self.lock:
            self.calls[lane] += 1
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        try:
            if lane in {
                ConstructionLane.OPUS5_DRAFT,
                ConstructionLane.GEMINI_ALTERNATE,
            }:
                self.barriers["author"].wait(timeout=3)
            elif lane in {
                ConstructionLane.OPUS48_CRITIQUE,
                ConstructionLane.OPUS5_CRITIQUE_BACKUP,
                ConstructionLane.GPT56_COMPARISON,
                ConstructionLane.GEMINI_COMPARISON_BACKUP,
            }:
                self.barriers["critic"].wait(timeout=3)
            time.sleep(0.01)
            response = (
                "{broken"
                if lane in self.malformed_lanes
                else json.dumps(
                    {
                        "schema": "fixture.construction-output.v1",
                        "lane": lane.value,
                        "value": f"completed-{lane.value}",
                    }
                )
            )
            if lane in self.fenced_lanes:
                response = f"Here is the exact payload:\n```json\n{response}\n```\n"
            return _turn_receipt(options, final_response=response)
        finally:
            with self.lock:
                self.active -= 1


def _source(name: str = "candidate-1") -> FrozenConstructionSource:
    return FrozenConstructionSource.create(
        source_id=name,
        request={
            "schema": "rlevo.med-research-candidate-construction-input.v3",
            "construction_id": name,
            "frozen": True,
        },
    )


def test_exact_seven_lane_schedule_preserves_schemas_and_dependencies(tmp_path: Path) -> None:
    runner = _Runner()
    adapter = _Adapter(tmp_path)
    orchestrator = PremiumCodexConstructionOrchestrator(
        runner,
        _routes(),
        id_factory=DeterministicUUIDFactory("premium-construction"),
    )

    result = orchestrator.construct(_source(), adapter)

    verify_premium_construction_result(result)
    assert result.schema == "eva.premium-codex-construction.v3"
    assert result.provider_call_count == 7
    assert result.wave_widths == (2, 4, 1)
    assert result.semantic_retry_count == 0
    assert runner.maximum == 4
    assert runner.calls == Counter({lane: 1 for lane in ConstructionLane})
    assert tuple(adapter.validated)[:2] in {
        (ConstructionLane.OPUS5_DRAFT, ConstructionLane.GEMINI_ALTERNATE),
        (ConstructionLane.GEMINI_ALTERNATE, ConstructionLane.OPUS5_DRAFT),
    }
    assert set(adapter.validated) == set(ConstructionLane)
    assert all(
        phase.receipt.output_schema_blake3
        == adapter.source_schema_digests[phase.receipt.lane]
        for phase in result.phases
    )
    assert all(phase.receipt.sandbox is CodexSandbox.READ_ONLY for phase in result.phases)
    assert all(not phase.codex_turn_receipt.tool_calls for phase in result.phases)
    assert result.canonical_output["lane"] == ConstructionLane.OPUS5_REVISION.value
    by_lane = {phase.receipt.lane: phase for phase in result.phases}
    assert dict(result.selected_critic_receipt_blake3s or {}) == {
        "critique": by_lane[
            ConstructionLane.OPUS48_CRITIQUE
        ].receipt.receipt_blake3,
        "comparison": by_lane[
            ConstructionLane.GPT56_COMPARISON
        ].receipt.receipt_blake3,
    }
    assert result.legacy_manifest_emitted is False
    assert result.legacy_manifest_compatibility_claimed is False
    assert tuple(evidence.lane for evidence in result.attempt_evidence) == tuple(
        ConstructionLane
    )
    assert all(evidence.status == "succeeded" for evidence in result.attempt_evidence)
    assert all(
        ConstructionAttemptEvidence.from_document(evidence.to_document()) == evidence
        for evidence in result.attempt_evidence
    )


def test_v4_opus_authority_retains_lower_tier_failures_without_retry(
    tmp_path: Path,
) -> None:
    runner = _Runner(
        malformed_lanes={
            ConstructionLane.OPUS48_CRITIQUE,
            ConstructionLane.GPT56_COMPARISON,
        }
    )
    routes = _routes()
    result = PremiumCodexConstructionOrchestrator(
        runner,
        routes,
        id_factory=DeterministicUUIDFactory("premium-construction-v4"),
        result_schema="eva.premium-codex-construction.v4",
    ).construct(_source(), _Adapter(tmp_path))

    verify_premium_construction_result(result)
    phases = {phase.receipt.lane: phase for phase in result.phases}
    assert result.schema == "eva.premium-codex-construction.v4"
    assert result.provider_call_count == 7
    assert result.semantic_retry_count == 0
    assert tuple(failure.lane for failure in result.supplemental_failures) == (
        ConstructionLane.OPUS48_CRITIQUE,
        ConstructionLane.GPT56_COMPARISON,
    )
    assert phases[
        ConstructionLane.GEMINI_COMPARISON_BACKUP
    ].receipt.model == routes.opus5.model
    assert dict(result.selected_critic_receipt_blake3s or {}) == {
        "critique": phases[
            ConstructionLane.OPUS5_CRITIQUE_BACKUP
        ].receipt.receipt_blake3,
        "comparison": phases[
            ConstructionLane.GEMINI_COMPARISON_BACKUP
        ].receipt.receipt_blake3,
    }
    assert len(result.attempt_evidence) == 7
    assert runner.calls == Counter({lane: 1 for lane in ConstructionLane})


def test_v5_unique_schema_extraction_preserves_fenced_raw_response(
    tmp_path: Path,
) -> None:
    runner = _Runner(
        fenced_lanes={ConstructionLane.GEMINI_COMPARISON_BACKUP}
    )
    result = PremiumCodexConstructionOrchestrator(
        runner,
        _routes(),
        id_factory=DeterministicUUIDFactory("premium-construction-v5"),
        result_schema="eva.premium-codex-construction.v4",
        response_extraction_policy=UNIQUE_SCHEMA_RESPONSE_EXTRACTION_POLICY_V2,
    ).construct(_source(), _Adapter(tmp_path))

    verify_premium_construction_result(result)
    phase = next(
        value
        for value in result.phases
        if value.receipt.lane is ConstructionLane.GEMINI_COMPARISON_BACKUP
    )
    assert phase.codex_turn_receipt.final_response.startswith("Here is")
    assert phase.output["lane"] == ConstructionLane.GEMINI_COMPARISON_BACKUP.value
    assert runner.calls == Counter({lane: 1 for lane in ConstructionLane})


def test_v5_unique_schema_extraction_rejects_two_distinct_valid_payloads() -> None:
    schema = _schema(ConstructionLane.OPUS5_DRAFT)
    first = json.dumps(
        {"schema": "fixture.construction-output.v1", "lane": "opus5_draft", "value": "a"}
    )
    second = json.dumps(
        {"schema": "fixture.construction-output.v1", "lane": "opus5_draft", "value": "b"}
    )
    with pytest.raises(PremiumConstructionError, match="exactly one distinct"):
        extract_unique_schema_valid_payload(f"{first}\n{second}", schema)


def test_v1_result_receipt_remains_reopenable(tmp_path: Path) -> None:
    current = PremiumCodexConstructionOrchestrator(
        _Runner(),
        _routes(),
        id_factory=DeterministicUUIDFactory("v1-reopen-fixture"),
    ).construct(_source(), _Adapter(tmp_path))
    legacy_phases = tuple(
        phase
        for phase in current.phases
        if phase.receipt.lane
        in {
            ConstructionLane.OPUS5_DRAFT,
            ConstructionLane.GEMINI_ALTERNATE,
            ConstructionLane.OPUS48_CRITIQUE,
            ConstructionLane.GPT56_COMPARISON,
            ConstructionLane.OPUS5_REVISION,
        }
    )
    core = {
        "schema": "eva.premium-codex-construction.v1",
        "construction_run_id": current.construction_run_id,
        "source_id": current.source_id,
        "source_request_blake3": current.source_request_blake3,
        "phase_receipt_blake3s": tuple(
            phase.receipt.receipt_blake3 for phase in legacy_phases
        ),
        "canonical_output_blake3": current.canonical_output_blake3,
        "provider_call_count": 5,
        "wave_widths": (2, 2, 1),
        "semantic_retry_count": 0,
        "legacy_manifest_emitted": False,
        "legacy_manifest_compatibility_claimed": False,
    }
    legacy = PremiumConstructionResult(
        schema=core["schema"],
        construction_run_id=core["construction_run_id"],
        source_id=core["source_id"],
        source_request_blake3=core["source_request_blake3"],
        phases=legacy_phases,
        canonical_output_blake3=core["canonical_output_blake3"],
        provider_call_count=core["provider_call_count"],
        wave_widths=core["wave_widths"],
        semantic_retry_count=core["semantic_retry_count"],
        legacy_manifest_emitted=False,
        legacy_manifest_compatibility_claimed=False,
        receipt_blake3=blake3_hex(core),
    )

    verify_premium_construction_result(legacy)
    assert legacy.core() == core


def test_failed_peer_is_drained_once_and_blocks_revision(tmp_path: Path) -> None:
    runner = _Runner(
        malformed_lanes={
            ConstructionLane.OPUS48_CRITIQUE,
            ConstructionLane.OPUS5_CRITIQUE_BACKUP,
        }
    )
    orchestrator = PremiumCodexConstructionOrchestrator(runner, _routes())

    with pytest.raises(ConstructionWaveError) as caught:
        orchestrator.construct(_source(), _Adapter(tmp_path))

    error = caught.value
    assert error.wave == "critic"
    assert tuple(item.receipt.lane for item in error.completed) == (
        ConstructionLane.OPUS5_DRAFT,
        ConstructionLane.GEMINI_ALTERNATE,
        ConstructionLane.GPT56_COMPARISON,
        ConstructionLane.GEMINI_COMPARISON_BACKUP,
    )
    assert tuple(item.lane for item in error.failures) == (
        ConstructionLane.OPUS48_CRITIQUE,
        ConstructionLane.OPUS5_CRITIQUE_BACKUP,
    )
    assert error.failures[0].error_code == "strict_json"
    verify_construction_failure_receipt(error.failures[0])
    assert tuple(evidence.lane for evidence in error.attempt_evidence) == (
        ConstructionLane.OPUS5_DRAFT,
        ConstructionLane.GEMINI_ALTERNATE,
        ConstructionLane.OPUS48_CRITIQUE,
        ConstructionLane.OPUS5_CRITIQUE_BACKUP,
        ConstructionLane.GPT56_COMPARISON,
        ConstructionLane.GEMINI_COMPARISON_BACKUP,
    )
    failed_evidence = tuple(
        evidence for evidence in error.attempt_evidence
        if evidence.status == "failed"
    )
    assert len(failed_evidence) == 2
    assert all(evidence.codex_turn_receipt is not None for evidence in failed_evidence)
    assert all(
        evidence.codex_turn_receipt.final_response == "{broken"
        for evidence in failed_evidence
        if evidence.codex_turn_receipt is not None
    )
    assert all(
        ConstructionAttemptEvidence.from_document(evidence.to_document()) == evidence
        for evidence in error.attempt_evidence
    )
    assert runner.calls[ConstructionLane.OPUS48_CRITIQUE] == 1
    assert runner.calls[ConstructionLane.GPT56_COMPARISON] == 1
    assert runner.calls[ConstructionLane.GEMINI_COMPARISON_BACKUP] == 1
    assert runner.calls[ConstructionLane.OPUS5_REVISION] == 0
    assert sum(runner.calls.values()) == 6


def test_both_comparison_attempts_fail_closed_before_revision(tmp_path: Path) -> None:
    runner = _Runner(
        malformed_lanes={
            ConstructionLane.GPT56_COMPARISON,
            ConstructionLane.GEMINI_COMPARISON_BACKUP,
        }
    )

    with pytest.raises(ConstructionWaveError) as caught:
        PremiumCodexConstructionOrchestrator(runner, _routes()).construct(
            _source(), _Adapter(tmp_path)
        )

    assert caught.value.wave == "critic"
    assert tuple(item.receipt.lane for item in caught.value.completed) == (
        ConstructionLane.OPUS5_DRAFT,
        ConstructionLane.GEMINI_ALTERNATE,
        ConstructionLane.OPUS48_CRITIQUE,
        ConstructionLane.OPUS5_CRITIQUE_BACKUP,
    )
    assert tuple(item.lane for item in caught.value.failures) == (
        ConstructionLane.GPT56_COMPARISON,
        ConstructionLane.GEMINI_COMPARISON_BACKUP,
    )
    assert runner.calls[ConstructionLane.OPUS5_REVISION] == 0
    assert sum(runner.calls.values()) == 6


def test_failed_primary_critics_select_exact_backups_and_retain_all_evidence(
    tmp_path: Path,
) -> None:
    runner = _Runner(
        malformed_lanes={
            ConstructionLane.OPUS48_CRITIQUE,
            ConstructionLane.GPT56_COMPARISON,
        }
    )
    result = PremiumCodexConstructionOrchestrator(
        runner,
        _routes(),
        id_factory=DeterministicUUIDFactory("critic-backup-selection"),
    ).construct(_source(), _Adapter(tmp_path))

    verify_premium_construction_result(result)
    by_lane = {phase.receipt.lane: phase for phase in result.phases}
    assert tuple(failure.lane for failure in result.supplemental_failures) == (
        ConstructionLane.OPUS48_CRITIQUE,
        ConstructionLane.GPT56_COMPARISON,
    )
    assert dict(result.selected_critic_receipt_blake3s or {}) == {
        "critique": by_lane[
            ConstructionLane.OPUS5_CRITIQUE_BACKUP
        ].receipt.receipt_blake3,
        "comparison": by_lane[
            ConstructionLane.GEMINI_COMPARISON_BACKUP
        ].receipt.receipt_blake3,
    }
    assert result.provider_call_count == 7
    assert runner.calls == Counter({lane: 1 for lane in ConstructionLane})


def test_failed_supplemental_author_uses_proven_zero_call_fallback(
    tmp_path: Path,
) -> None:
    runner = _Runner(malformed_lane=ConstructionLane.GEMINI_ALTERNATE)
    adapter = _Adapter(tmp_path)
    result = PremiumCodexConstructionOrchestrator(
        runner,
        _routes(),
        id_factory=DeterministicUUIDFactory("supplemental-author-fallback"),
    ).construct(_source(), adapter)

    verify_premium_construction_result(result)
    assert result.schema == "eva.premium-codex-construction.v3"
    assert result.provider_call_count == 7
    assert tuple(phase.receipt.lane for phase in result.phases) == (
        ConstructionLane.OPUS5_DRAFT,
        ConstructionLane.OPUS48_CRITIQUE,
        ConstructionLane.OPUS5_CRITIQUE_BACKUP,
        ConstructionLane.GPT56_COMPARISON,
        ConstructionLane.GEMINI_COMPARISON_BACKUP,
        ConstructionLane.OPUS5_REVISION,
    )
    assert len(result.supplemental_failures) == 1
    assert result.supplemental_failures[0].error_code == "strict_json"
    assert len(result.fallbacks) == 1
    assert isinstance(result.fallbacks[0], ConstructionFallbackResult)
    verify_construction_fallback_result(result.fallbacks[0])
    assert ConstructionFallbackReceipt.from_document(
        result.fallbacks[0].receipt.to_document()
    ) == result.fallbacks[0].receipt
    assert result.fallbacks[0].receipt.provider_call_count == 0
    assert result.fallbacks[0].receipt.semantic_attempt_count == 0
    assert result.fallbacks[0].receipt.supplemental_authoritative_for_admission is False
    assert runner.calls == Counter({lane: 1 for lane in ConstructionLane})

    with pytest.raises(PremiumConstructionError, match="fallback output"):
        replace(
            result.fallbacks[0],
            output={
                "schema": "fixture.construction-output.v1",
                "lane": ConstructionLane.GEMINI_ALTERNATE.value,
                "value": "tampered",
            },
        )


def test_primary_author_failure_remains_terminal(tmp_path: Path) -> None:
    runner = _Runner(malformed_lane=ConstructionLane.OPUS5_DRAFT)

    with pytest.raises(ConstructionWaveError) as caught:
        PremiumCodexConstructionOrchestrator(runner, _routes()).construct(
            _source(), _Adapter(tmp_path)
        )

    assert caught.value.wave == "author"
    assert tuple(failure.lane for failure in caught.value.failures) == (
        ConstructionLane.OPUS5_DRAFT,
    )
    assert sum(runner.calls.values()) == 2


def test_dependency_mismatch_fails_before_critic_provider_calls(tmp_path: Path) -> None:
    runner = _Runner()
    adapter = _Adapter(tmp_path, bad_critic_dependencies=True)
    orchestrator = PremiumCodexConstructionOrchestrator(runner, _routes())

    with pytest.raises(PremiumConstructionError, match="dependency binding"):
        orchestrator.construct(_source(), adapter)

    assert sum(runner.calls.values()) == 2
    assert set(runner.calls) == {
        ConstructionLane.OPUS5_DRAFT,
        ConstructionLane.GEMINI_ALTERNATE,
    }


def test_source_output_schema_must_be_identical_before_run_once(tmp_path: Path) -> None:
    source = _source()
    routes = _routes()
    adapter = _Adapter(tmp_path)
    request = adapter._request(source, routes, ConstructionLane.OPUS5_DRAFT, {})
    different = _schema(ConstructionLane.GEMINI_ALTERNATE)

    with pytest.raises(PremiumConstructionError, match="output_schema differs"):
        ConstructionTurnRequest.create(
            lane=ConstructionLane.OPUS5_DRAFT,
            options=request.options,
            turn_input=request.turn_input,
            source_request_blake3=source.request_blake3,
            source_phase_request={"different": True},
            source_output_schema=different,
        )


def test_parallel_authors_require_isolated_roots_before_calls(tmp_path: Path) -> None:
    runner = _Runner()
    orchestrator = PremiumCodexConstructionOrchestrator(runner, _routes())

    with pytest.raises(PremiumConstructionError, match="isolated workspace"):
        orchestrator.construct(
            _source(), _Adapter(tmp_path, shared_root=True)
        )

    assert sum(runner.calls.values()) == 0


class _Sink:
    def __init__(self) -> None:
        self.calls = 0

    def publish(self, result):
        self.calls += 1
        core = {
            "schema": "eva.premium-codex-construction-publication.v1",
            "publication_id": str(uuid4()),
            "sink_name": "fixture-append-only",
            "construction_receipt_blake3": result.receipt_blake3,
            "artifact_blake3": blake3_hex({"result": result.receipt_blake3}),
            "legacy_manifest_compatibility_verified": False,
            "metadata": {"replace_existing": False},
        }
        return ConstructionPublicationReceipt(**core, receipt_blake3=blake3_hex(core))


def test_sink_publishes_once_after_complete_construction(tmp_path: Path) -> None:
    sink = _Sink()
    published = PremiumCodexConstructionOrchestrator(
        _Runner(), _routes()
    ).construct_and_publish(_source(), _Adapter(tmp_path), sink)

    assert isinstance(published, PublishedConstructionResult)
    assert sink.calls == 1
    assert published.publication.construction_receipt_blake3 == published.result.receipt_blake3
    assert published.publication.legacy_manifest_compatibility_verified is False


class _SleepRunner(_Runner):
    def __init__(self) -> None:
        super().__init__()
        self.barriers = {}

    def run_once(self, options, turn_input):
        lane = ConstructionLane(turn_input.output_schema["properties"]["lane"]["const"])
        with self.lock:
            self.calls[lane] += 1
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        try:
            time.sleep(0.025)
            return _turn_receipt(
                options,
                final_response=json.dumps(
                    {
                        "schema": "fixture.construction-output.v1",
                        "lane": lane.value,
                        "value": f"completed-{lane.value}",
                    }
                ),
            )
        finally:
            with self.lock:
                self.active -= 1


def test_outer_candidate_width_is_explicit_and_unthrottled(tmp_path: Path) -> None:
    runner = _SleepRunner()
    orchestrator = PremiumCodexConstructionOrchestrator(runner, _routes())
    jobs = tuple(
        ConstructionJob(_source(f"candidate-{index}"), _Adapter(tmp_path / str(index)))
        for index in range(4)
    )

    results = orchestrator.construct_many(jobs, max_workers=4)

    assert len(results) == 4
    assert runner.maximum >= 4
    assert sum(runner.calls.values()) == 28
    assert all(result.provider_call_count == 7 for result in results)
