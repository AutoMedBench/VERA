from __future__ import annotations

import base64
from collections import Counter
import json
from pathlib import Path
from threading import Lock
import time
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import eva_agent.construction.premium_supervisor as premium_supervisor
from eva_agent.campaign.selection_v2 import (
    CampaignSelectionV2,
    SELECTION_V2_METHOD,
)
from eva_agent.codex_runtime import (
    CodexEvent,
    CodexRole,
    CodexSandbox,
    CodexThreadOptions,
    CodexTurnInput,
    CodexTurnReceipt,
)
from eva_agent.construction import (
    ConstructionFailureReceipt,
    ConstructionLane,
    ConstructionModelRoute,
    ConstructionPublicationReceipt,
    ConstructionTurnRequest,
    FrozenConstructionSource,
    PremiumConstructionCampaign,
    PremiumConstructionCampaignConfig,
    PremiumConstructionCampaignError,
    PremiumConstructionExecutionBindingResolver,
    PremiumConstructionPreflightReceipt,
    PremiumConstructionProofCatalog,
    PremiumConstructionRoutes,
    PremiumConstructionSupervisorError,
    build_premium_construction_queue,
    issue_premium_construction_supervisor_transition_v2,
    render_premium_construction_progress,
    verify_premium_construction_proof_catalog,
    verify_premium_construction_supervisor_transition_v2,
    verify_construction_failure_receipt,
)
from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes
from eva_agent.pipeline.ids import DeterministicUUIDFactory
from eva_agent.construction.premium_codex import ConstructionAttemptEvidence


def _selection(count: int = 8) -> CampaignSelectionV2:
    entries = []
    for index in range(1, count + 1):
        candidate_id = str(uuid4())
        entry_core = {
            "candidate_id": candidate_id,
            "source_candidate_id": f"source-{index:04d}",
            "source_family": "automedbench",
            "source_artifact_sha256": f"{index:064x}",
            "domain": "general_medical_research",
            "stage": "E2E",
            "split": "train",
            "rubric_id": str(uuid4()),
            "rubric_blake3": f"{index + 100:064x}",
            "source_identity_rank_blake3": f"{index + 200:064x}",
            "construction_readiness": "frozen",
            "readiness_proof_root_blake3": f"{index + 300:064x}",
            "selection_rank_blake3": f"{index + 400:064x}",
            "cell_rank": index,
            "cell_target": count,
            "selection_tier": "primary",
            "queue_ordinal": index,
        }
        entries.append(entry_core)
    core = {
        "schema": "eva.medresearch-campaign-selection.v2",
        "selection_id": str(uuid4()),
        "plan_id": "fixture-plan",
        "plan_blake3": "1" * 64,
        "source_registry_blake3": "2" * 64,
        "upstream_source_registry_sha256": "3" * 64,
        "rubric_registry_blake3": "4" * 64,
        "base_selection_id": str(uuid4()),
        "base_selection_blake3": "5" * 64,
        "readiness_authority": {"authority_blake3": "6" * 64},
        "selection_method": SELECTION_V2_METHOD,
        "selected_count": count,
        "scheduled_count": count,
        "reserve_count": 0,
        "entries": entries,
    }
    return CampaignSelectionV2.from_document(
        {**core, "selection_blake3": blake3_hex(core)}
    )


def _routes() -> PremiumConstructionRoutes:
    return PremiumConstructionRoutes(
        opus5=ConstructionModelRoute("opus_5", "anthropic/opus-5", "opus"),
        gemini31=ConstructionModelRoute(
            "gemini_3_1_pro", "google/gemini-3.1-pro", "gemini"
        ),
        opus48=ConstructionModelRoute("opus_4_8", "anthropic/opus-4.8", "opus"),
        gpt56=ConstructionModelRoute("gpt_5_6_sol", "openai/gpt-5.6-sol", "openai"),
    )


_ROLES = {
    ConstructionLane.OPUS5_DRAFT: CodexRole.STRONG_ACTOR,
    ConstructionLane.GEMINI_ALTERNATE: CodexRole.STRONG_ACTOR,
    ConstructionLane.OPUS48_CRITIQUE: CodexRole.MIDDLE_ACTOR,
    ConstructionLane.OPUS5_CRITIQUE_BACKUP: CodexRole.STRONG_ACTOR,
    ConstructionLane.GPT56_COMPARISON: CodexRole.STRONG_ACTOR,
    ConstructionLane.GEMINI_COMPARISON_BACKUP: CodexRole.STRONG_ACTOR,
    ConstructionLane.OPUS5_REVISION: CodexRole.STRONG_ACTOR,
}


def _schema(lane: ConstructionLane) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "lane", "value"],
        "properties": {
            "schema": {"const": "fixture.campaign-output.v1"},
            "lane": {"const": lane.value},
            "value": {"type": "string"},
        },
    }


class _Adapter:
    def __init__(self, root: Path, candidate_id: str) -> None:
        self.root = root
        self.candidate_id = candidate_id

    def _request(self, source, routes, lane, dependencies):
        route = routes.for_lane(lane)
        sandbox = CodexSandbox.READ_ONLY
        schema = _schema(lane)
        options = CodexThreadOptions(
            role=_ROLES[lane],
            model=route.model,
            provider=route.provider,
            cwd=str((self.root / self.candidate_id / lane.value).resolve()),
            sandbox=sandbox,
        )
        turn = CodexTurnInput(
            public_text=f"Execute {lane.value}",
            public_context={"candidate_id": self.candidate_id},
            output_schema=schema,
        )
        phase = {
            "schema": "fixture.campaign-phase.v1",
            "lane": lane.value,
            "output_schema": schema,
            "dependencies": dependencies,
        }
        return ConstructionTurnRequest.create(
            lane=lane,
            options=options,
            turn_input=turn,
            source_request_blake3=source.request_blake3,
            source_phase_request=phase,
            source_output_schema=schema,
            dependencies=dependencies,
        )

    def prepare_authors(self, source, routes):
        return tuple(self._request(source, routes, lane, {}) for lane in (
            ConstructionLane.OPUS5_DRAFT,
            ConstructionLane.GEMINI_ALTERNATE,
        ))

    @staticmethod
    def _authors(primary, alternate):
        return {
            ConstructionLane.OPUS5_DRAFT.value: blake3_hex(primary),
            ConstructionLane.GEMINI_ALTERNATE.value: blake3_hex(alternate),
        }

    def prepare_critics(self, source, routes, *, primary_draft, alternate_draft):
        dependencies = self._authors(primary_draft, alternate_draft)
        return tuple(self._request(source, routes, lane, dependencies) for lane in (
            ConstructionLane.OPUS48_CRITIQUE,
            ConstructionLane.GPT56_COMPARISON,
        ))

    def prepare_critic_quorum(
        self, source, routes, *, primary_draft, alternate_draft
    ):
        dependencies = self._authors(primary_draft, alternate_draft)
        return tuple(self._request(source, routes, lane, dependencies) for lane in (
            ConstructionLane.OPUS48_CRITIQUE,
            ConstructionLane.OPUS5_CRITIQUE_BACKUP,
            ConstructionLane.GPT56_COMPARISON,
            ConstructionLane.GEMINI_COMPARISON_BACKUP,
        ))

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
            **self._authors(primary_draft, alternate_draft),
            ConstructionLane.OPUS48_CRITIQUE.value: blake3_hex(opus48_critique),
            ConstructionLane.GPT56_COMPARISON.value: blake3_hex(gpt56_comparison),
        }
        return self._request(
            source, routes, ConstructionLane.OPUS5_REVISION, dependencies
        )

    def validate_output(self, _source, request, output):
        if output["lane"] != request.lane.value:
            raise ValueError("wrong fixture lane")

    def resolve_supplemental_author_fallback(
        self, _source, request, *, primary_draft, failure
    ):
        if not (
            request.lane is ConstructionLane.GEMINI_ALTERNATE
            and failure.request_blake3 == request.request_blake3
        ):
            raise ValueError("wrong supplemental author request")
        return {
            "schema": "fixture.campaign-output.v1",
            "lane": ConstructionLane.GEMINI_ALTERNATE.value,
            "value": f"canonical-primary:{blake3_hex(primary_draft)}",
        }


class _Sink:
    def publish(self, result):
        metadata = {
            "binding_blake3": blake3_hex({"binding": result.source_id}),
            "validator_authority_blake3": blake3_hex({"authority": result.source_id}),
            "executable_material_blake3": blake3_hex({"material": result.source_id}),
            "append_only": True,
            "retry_count": 0,
        }
        core = {
            "schema": "eva.premium-codex-construction-publication.v1",
            "publication_id": str(uuid4()),
            "sink_name": "fixture-o-excl",
            "construction_receipt_blake3": result.receipt_blake3,
            "artifact_blake3": blake3_hex({"artifact": result.source_id}),
            "legacy_manifest_compatibility_verified": False,
            "metadata": metadata,
        }
        return ConstructionPublicationReceipt(**core, receipt_blake3=blake3_hex(core))


class _Factory:
    def __init__(self, selection: CampaignSelectionV2, root: Path) -> None:
        self.selection = selection
        self.root = root

    def compose(self, candidate_id, _routes):
        source = FrozenConstructionSource.create(
            source_id=candidate_id,
            request={"schema": "fixture.frozen-source.v1", "candidate_id": candidate_id},
        )
        return SimpleNamespace(
            source=source,
            adapter=_Adapter(self.root, candidate_id),
            sink=_Sink(),
            run=lambda orchestrator: orchestrator.construct_and_publish(
                source, _Adapter(self.root, candidate_id), _Sink()
            ),
        )


def _turn_receipt(options, final_response):
    thread_id = f"thread-{uuid4()}"
    turn_id = f"turn-{uuid4()}"
    event_core = {
        "event_id": str(uuid4()),
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
        "input_blake3": blake3_hex({"fixture": "campaign"}),
        "config_keys": options.config_keys,
        "config_values_recorded": False,
        "input_payload_recorded": False,
        "sdk_version": "fake",
        "server_version": "fake",
    }
    return CodexTurnReceipt(**core, receipt_blake3=blake3_hex(core))


class _Runner:
    shard_count = 64

    def __init__(
        self,
        *,
        fail_candidate: str | None = None,
        malformed_lanes: set[str] | None = None,
    ) -> None:
        self.fail_candidate = fail_candidate
        self.malformed_lanes = malformed_lanes or set()
        self.calls: Counter[str] = Counter()
        self.active = 0
        self.maximum_active = 0
        self.lock = Lock()
        self.exact_port_calls = 0

    def run_once(self, options, turn_input):
        candidate = turn_input.public_context["candidate_id"]
        lane = turn_input.output_schema["properties"]["lane"]["const"]
        with self.lock:
            self.calls[candidate] += 1
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            time.sleep(0.02)
            output = {
                "schema": "fixture.campaign-output.v1",
                "lane": lane,
                "value": "ok",
            }
            if candidate == self.fail_candidate and lane == "opus5_draft":
                output["lane"] = "invalid"
            if lane in self.malformed_lanes:
                output["lane"] = "invalid"
            return _turn_receipt(options, json.dumps(output))
        finally:
            with self.lock:
                self.active -= 1

    def run_construction_once(self, request):
        with self.lock:
            self.exact_port_calls += 1
        return self.run_once(request.options, request.turn_input)


def test_provider_free_preflight_and_128_worker_fake_seven_call_campaign(tmp_path: Path) -> None:
    selection = _selection(8)
    queue = build_premium_construction_queue(selection)
    runner = _Runner()
    progress = []
    campaign = PremiumConstructionCampaign(
        queue=queue,
        factory=_Factory(selection, tmp_path / "workspaces"),
        runner=runner,
        routes=_routes(),
        state_root=tmp_path / "state",
        config=PremiumConstructionCampaignConfig(
            worker_width=128, app_server_shards=64, progress_seconds=0.01
        ),
        id_factory=DeterministicUUIDFactory("premium-campaign"),
        clock=lambda: "2026-09-07T00:00:00Z",
        progress_callback=progress.append,
    )

    preflight = campaign.preflight()
    assert preflight.provider_call_count == 0
    assert sum(runner.calls.values()) == 0
    assert len(preflight.candidate_proofs) == 8
    assert [row["frontier"] for row in preflight.phase_plan] == [1, 1, 2, 2, 2, 2, 3]
    assert [row["workspace_mode"] for row in preflight.phase_plan] == [
        "read-only",
        "read-only",
        "read-only",
        "read-only",
        "read-only",
        "read-only",
        "read-only",
    ]
    assert PremiumConstructionPreflightReceipt.from_document(
        preflight.to_document()
    ) == preflight

    report = campaign.run()
    assert report.claimed_this_session == report.succeeded_this_session == 8
    assert report.quarantined_this_session == 0
    assert report.logical_provider_calls_started == 56
    assert runner.exact_port_calls == 56
    assert all(count == 7 for count in runner.calls.values())
    assert runner.maximum_active >= 4
    assert progress[0].claimed == 0
    assert progress[0].active == 0
    assert progress[0].unclaimed == 8
    assert progress[-1].active == 0
    assert progress[-1].unclaimed == 0
    line = render_premium_construction_progress(progress[-1], width=10)
    assert line.startswith("[##########] 8/8 (100.00%)")
    assert "calls=56" in line
    assert "shards=64" in line
    session = json.loads(
        (tmp_path / "state" / "sessions" / f"{report.session_id}.start.json").read_text()
    )
    assert [row["lane"] for row in session["phase_plan"]] == [
        "opus5_draft",
        "gemini_alternate",
        "opus48_critique",
        "opus5_critique_backup",
        "gpt56_comparison",
        "gemini_comparison_backup",
        "opus5_revision",
    ]
    assert [row["frontier"] for row in session["phase_plan"]] == [
        1, 1, 2, 2, 2, 2, 3
    ]
    assert [row["workspace_mode"] for row in session["phase_plan"]] == [
        "read-only",
        "read-only",
        "read-only",
        "read-only",
        "read-only",
        "read-only",
        "read-only",
    ]
    assert all(row["retry_count"] == 0 for row in session["phase_plan"])
    assert session["phase_plan_blake3"] == blake3_hex(session["phase_plan"])
    supervisor_input = verify_premium_construction_proof_catalog(
        report.proof_catalog, queue=queue
    )
    assert supervisor_input["successful_binding_count"] == 8
    assert supervisor_input["supervisor_transition_authorized"] is False
    reopened = PremiumConstructionProofCatalog.from_document(
        report.proof_catalog.to_document()
    )
    assert reopened == report.proof_catalog


def test_abandoned_claim_is_quarantined_without_retry_and_next_row_runs(tmp_path: Path) -> None:
    selection = _selection(2)
    queue = build_premium_construction_queue(selection)
    factory = _Factory(selection, tmp_path / "workspaces")
    runner = _Runner()
    campaign = PremiumConstructionCampaign(
        queue=queue,
        factory=factory,
        runner=runner,
        routes=_routes(),
        state_root=tmp_path / "state",
        config=PremiumConstructionCampaignConfig(
            worker_width=1,
            app_server_shards=64,
            max_candidates=1,
        ),
        id_factory=DeterministicUUIDFactory("recovery-campaign"),
        clock=lambda: "2026-09-07T00:00:00Z",
    )
    abandoned_session = str(uuid4())
    with campaign._store.exclusive():
        claim = campaign._store.claim(
            queue.entries[0],
            session_id=abandoned_session,
            clock=lambda: "2026-09-07T00:00:00Z",
        )
        assert claim is not None

    assert campaign.pending_candidate_ids(limit=1) == (
        queue.entries[1].candidate_id,
    )
    report = campaign.run()
    assert report.recovered_without_retry == 1
    assert report.succeeded_this_session == 1
    assert report.quarantined_this_session == 0
    assert runner.calls[queue.entries[0].candidate_id] == 0
    assert runner.calls[queue.entries[1].candidate_id] == 7
    records = report.proof_catalog.records
    assert [record.status for record in records] == ["quarantined", "succeeded"]
    assert records[0].error_code == "abandoned_after_process_loss"

    next_queue = build_premium_construction_queue(
        selection, consumed_records=records
    )
    assert next_queue.entries == ()
    assert next_queue.bound_excluded_count == 1
    assert next_queue.quarantined_excluded_count == 1


def test_failed_author_wave_is_immutable_quarantine_with_two_calls(tmp_path: Path) -> None:
    selection = _selection(2)
    queue = build_premium_construction_queue(selection)
    failed = queue.entries[0].candidate_id
    runner = _Runner(fail_candidate=failed)
    campaign = PremiumConstructionCampaign(
        queue=queue,
        factory=_Factory(selection, tmp_path / "workspaces"),
        runner=runner,
        routes=_routes(),
        state_root=tmp_path / "state",
        config=PremiumConstructionCampaignConfig(
            worker_width=128,
            app_server_shards=64,
        ),
        id_factory=DeterministicUUIDFactory("failure-campaign"),
        clock=lambda: "2026-09-07T00:00:00Z",
    )
    report = campaign.run()
    assert report.succeeded_this_session == 1
    assert report.quarantined_this_session == 1
    by_id = {record.candidate_id: record for record in report.proof_catalog.records}
    assert by_id[failed].error_code == "author_wave_failure"
    assert by_id[failed].logical_provider_calls_started == 2
    assert runner.calls[failed] == 2
    assert len(by_id[failed].failure_receipt_blake3s) == 1
    digest = by_id[failed].failure_receipt_blake3s[0]
    failure_path = tmp_path / "state" / "failures" / f"{digest}.json"
    assert failure_path.is_file() and not failure_path.is_symlink()
    reopened = ConstructionFailureReceipt.from_document(
        json.loads(failure_path.read_text(encoding="utf-8"))
    )
    verify_construction_failure_receipt(reopened)
    assert reopened.receipt_blake3 == digest
    assert reopened.exception_text_recorded is False
    attempt_digests = by_id[failed].attempt_evidence_blake3s
    assert len(attempt_digests) == 2
    attempts = tuple(
        ConstructionAttemptEvidence.from_document(
            json.loads(
                (tmp_path / "state" / "attempts" / f"{attempt_digest}.json")
                .read_text(encoding="utf-8")
            )
        )
        for attempt_digest in attempt_digests
    )
    assert tuple(attempt.lane for attempt in attempts) == (
        ConstructionLane.OPUS5_DRAFT,
        ConstructionLane.GEMINI_ALTERNATE,
    )
    failed_attempt = next(attempt for attempt in attempts if attempt.status == "failed")
    assert failed_attempt.codex_turn_receipt is not None
    assert '"lane": "invalid"' in failed_attempt.codex_turn_receipt.final_response


def test_primary_critic_failures_are_successful_quorum_evidence(tmp_path: Path) -> None:
    selection = _selection(1)
    queue = build_premium_construction_queue(selection)
    runner = _Runner(
        malformed_lanes={"opus48_critique", "gpt56_comparison"}
    )
    campaign = PremiumConstructionCampaign(
        queue=queue,
        factory=_Factory(selection, tmp_path / "workspaces"),
        runner=runner,
        routes=_routes(),
        state_root=tmp_path / "state",
        config=PremiumConstructionCampaignConfig(
            worker_width=1,
            app_server_shards=64,
        ),
        id_factory=DeterministicUUIDFactory("critic-quorum-campaign"),
        clock=lambda: "2026-09-07T00:00:00Z",
    )

    report = campaign.run()
    assert report.succeeded_this_session == 1
    assert report.quarantined_this_session == 0
    assert report.logical_provider_calls_started == 7
    record = report.proof_catalog.records[0]
    assert record.status == "succeeded"
    assert record.logical_provider_calls_started == 7
    assert len(record.failure_receipt_blake3s) == 2
    assert len(record.attempt_evidence_blake3s) == 7
    persisted_attempts = tuple(
        ConstructionAttemptEvidence.from_document(
            json.loads(
                (tmp_path / "state" / "attempts" / f"{digest}.json")
                .read_text(encoding="utf-8")
            )
        )
        for digest in record.attempt_evidence_blake3s
    )
    assert tuple(attempt.lane for attempt in persisted_attempts) == tuple(
        ConstructionLane
    )
    assert sum(attempt.status == "failed" for attempt in persisted_attempts) == 2
    assert all(
        attempt.codex_turn_receipt is not None
        for attempt in persisted_attempts
        if attempt.status == "failed"
    )
    for digest in record.failure_receipt_blake3s:
        failure_path = tmp_path / "state" / "failures" / f"{digest}.json"
        reopened = ConstructionFailureReceipt.from_document(
            json.loads(failure_path.read_text(encoding="utf-8"))
        )
        verify_construction_failure_receipt(reopened)
        assert reopened.receipt_blake3 == digest


def test_only_explicit_high_width_profiles_and_64_shards_are_accepted() -> None:
    for width in (1, 128, 256, 512):
        config = PremiumConstructionCampaignConfig(
            worker_width=width, app_server_shards=64
        )
        assert config.to_document()["maximum_simultaneous_logical_provider_calls"] == 4 * width
    for width in (0, 2, 64, 1024, True):
        with pytest.raises(PremiumConstructionCampaignError):
            PremiumConstructionCampaignConfig(
                worker_width=width, app_server_shards=64
            )
    with pytest.raises(PremiumConstructionCampaignError, match="64 app-server"):
        PremiumConstructionCampaignConfig(worker_width=128, app_server_shards=32)


def _bridge_signing_material(tmp_path: Path) -> tuple[Path, Path]:
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
    trust_path = tmp_path / "trust.json"
    trust_path.write_text(
        json.dumps(
            {
                "schema": "rlevo.med-research-host-trust-store.v1",
                "status": "active",
                "algorithm": "Ed25519",
                "keys": {
                    "eva-test-key": base64.b64encode(public).decode("ascii")
                },
            }
        ),
        encoding="utf-8",
    )
    return private_path, trust_path


def test_incremental_v2_transitions_bind_cumulative_catalog_and_disjoint_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection(2)
    queue = build_premium_construction_queue(selection)
    campaign = PremiumConstructionCampaign(
        queue=queue,
        factory=_Factory(selection, tmp_path / "workspaces"),
        runner=_Runner(),
        routes=_routes(),
        state_root=tmp_path / "state",
        config=PremiumConstructionCampaignConfig(
            worker_width=1,
            app_server_shards=64,
            max_candidates=1,
        ),
        id_factory=DeterministicUUIDFactory("incremental-v2-catalogs"),
        clock=lambda: "2026-09-08T00:00:00Z",
    )
    catalog_a = campaign.run().proof_catalog
    catalog_ab = campaign.run().proof_catalog
    candidate_a, candidate_b = (row.candidate_id for row in queue.entries)
    assert [record.candidate_id for record in catalog_a.records] == [candidate_a]
    assert [record.candidate_id for record in catalog_ab.records] == [
        candidate_a,
        candidate_b,
    ]

    real_catalog_verifier = verify_premium_construction_proof_catalog

    def provider_free_catalog_verifier(catalog, *, queue, publication_output_root):
        del publication_output_root
        return real_catalog_verifier(catalog, queue=queue)

    def entry_binding(*, selection, queue, rubrics):
        del rubrics
        return (
            {row.candidate_id: row for row in selection.entries},
            {row.candidate_id: row for row in queue.entries},
        )

    authority = SimpleNamespace(
        modules=SimpleNamespace(source_blake3=blake3_hex("legacy-runtime")),
        validator=SimpleNamespace(
            identity=SimpleNamespace(authority_blake3=blake3_hex("validator"))
        ),
    )

    def runtime_authority(**_kwargs):
        return authority

    def executable_material(
        *,
        selected,
        queued,
        record,
        catalog_binding,
        selection_id,
        publication_output_root,
        authority,
    ):
        del selection_id, authority
        publication_id = record.publication_receipt["publication_id"]
        publication_root = publication_output_root / publication_id
        return SimpleNamespace(
            selected=selected,
            queued=queued,
            record=record,
            catalog_binding=catalog_binding,
            publication_root=publication_root,
            source_policy_path=publication_root / "policy.json",
            source_request_blake3=blake3_hex(
                {"source_request": selected.candidate_id}
            ),
            source_policy_blake3=blake3_hex(
                {"source_policy": selected.candidate_id}
            ),
            runtime_policy_blake3=blake3_hex(
                {"runtime_policy": selected.candidate_id}
            ),
            executable_binding_path=publication_root
            / "executable-binding.v1.json",
            executable_binding_blake3=record.publication_receipt[
                "artifact_blake3"
            ],
            tool_catalog_blake3=blake3_hex(
                {"tool_catalog": selected.candidate_id}
            ),
        )

    monkeypatch.setattr(
        premium_supervisor,
        "verify_premium_construction_proof_catalog",
        provider_free_catalog_verifier,
    )
    monkeypatch.setattr(premium_supervisor, "_entry_binding", entry_binding)
    monkeypatch.setattr(
        premium_supervisor, "_runtime_authority", runtime_authority
    )
    monkeypatch.setattr(
        premium_supervisor, "_open_executable_material", executable_material
    )

    private_path, trust_path = _bridge_signing_material(tmp_path)
    publication_root = tmp_path / "published"
    publication_root.mkdir()
    transition_output_root = tmp_path / "transitions"
    common = {
        "queue": queue,
        "selection": selection,
        "rubrics": object(),
        "publication_output_root": publication_root,
        "transition_output_root": transition_output_root,
        "legacy_python_root": tmp_path / "unused-legacy-python",
        "trust_store_path": trust_path,
        "image_refs_path": tmp_path / "unused-image-refs.json",
        "host_private_key_path": private_path,
        "host_key_id": "eva-test-key",
        "clock": lambda: "2026-09-08T00:00:00Z",
    }
    transition_a = issue_premium_construction_supervisor_transition_v2(
        catalog_a,
        selected_candidate_ids=(candidate_a,),
        id_factory=DeterministicUUIDFactory("incremental-transition-a"),
        **common,
    )
    transition_b = issue_premium_construction_supervisor_transition_v2(
        catalog_ab,
        selected_candidate_ids=(candidate_b,),
        id_factory=DeterministicUUIDFactory("incremental-transition-b"),
        **common,
    )
    assert transition_a.binding_count == transition_b.binding_count == 1
    assert transition_a.envelope.payload["selected_candidate_ids"] == (
        candidate_a,
    )
    assert transition_b.envelope.payload["selected_candidate_ids"] == (
        candidate_b,
    )
    assert transition_a.envelope.payload["claim_domain_blake3"] != (
        transition_b.envelope.payload["claim_domain_blake3"]
    )
    assert transition_b.envelope.payload["claim_domain_blake3"] == blake3_hex(
        {
            "schema": "eva.premium-construction-supervisor-claim-domain.v2",
            "catalog_blake3": catalog_ab.catalog_blake3,
            "selected_candidate_ids_blake3": transition_b.envelope.payload[
                "selected_candidate_ids_blake3"
            ],
        }
    )
    assert transition_b.envelope.payload["catalog_blake3"] == (
        catalog_ab.catalog_blake3
    )
    assert tuple(
        row["candidate_id"]
        for row in transition_b.envelope.payload["bindings"]
    ) == (candidate_b,)
    cumulative_document = json.loads(
        (transition_b.root / "premium-construction-proof-catalog.v1.json").read_text()
    )
    cumulative_supervisor_input = json.loads(
        (transition_b.root / "premium-construction-supervisor-input.v1.json").read_text()
    )
    assert cumulative_document["succeeded"] == 2
    assert len(cumulative_document["records"]) == 2
    assert cumulative_supervisor_input["successful_binding_count"] == 2
    assert verify_premium_construction_supervisor_transition_v2(
        transition_a.root,
        publication_output_root=publication_root,
        rubrics=object(),
        legacy_python_root=tmp_path / "unused-legacy-python",
        trust_store_path=trust_path,
        image_refs_path=tmp_path / "unused-image-refs.json",
        host_private_key_path=private_path,
        host_key_id="eva-test-key",
    ).transition_blake3 == transition_a.transition_blake3
    resolver = PremiumConstructionExecutionBindingResolver(
        transition_root=transition_b.root,
        publication_output_root=publication_root,
        rubrics=object(),
        legacy_python_root=tmp_path / "unused-legacy-python",
        runtime_state_root=tmp_path / "runtime-state",
        trust_store_path=trust_path,
        image_refs_path=tmp_path / "unused-image-refs.json",
        host_private_key_path=private_path,
        host_key_id="eva-test-key",
    )
    assert resolver.executable_candidate_count == 1
    assert resolver.inventory()[0].candidate_id == candidate_b

    claims_before = sorted(
        path.relative_to(transition_output_root).as_posix()
        for path in transition_output_root.rglob("*.json")
        if ".claims" in path.parts or ".candidate-claims" in path.parts
    )
    with pytest.raises(
        PremiumConstructionSupervisorError, match="already been claimed"
    ):
        issue_premium_construction_supervisor_transition_v2(
            catalog_ab,
            selected_candidate_ids=(candidate_b,),
            id_factory=DeterministicUUIDFactory("repeat-transition-b"),
            **common,
        )
    with pytest.raises(
        PremiumConstructionSupervisorError, match="already been claimed"
    ):
        issue_premium_construction_supervisor_transition_v2(
            catalog_ab,
            selected_candidate_ids=(candidate_a,),
            id_factory=DeterministicUUIDFactory("overlap-transition-a"),
            **common,
        )
    claims_after = sorted(
        path.relative_to(transition_output_root).as_posix()
        for path in transition_output_root.rglob("*.json")
        if ".claims" in path.parts or ".candidate-claims" in path.parts
    )
    assert claims_after == claims_before

    with pytest.raises(PremiumConstructionSupervisorError, match="nonempty"):
        issue_premium_construction_supervisor_transition_v2(
            catalog_ab,
            selected_candidate_ids=(),
            id_factory=DeterministicUUIDFactory("empty-transition"),
            **common,
        )
    with pytest.raises(PremiumConstructionSupervisorError, match="duplicated"):
        issue_premium_construction_supervisor_transition_v2(
            catalog_ab,
            selected_candidate_ids=(candidate_a, candidate_a),
            id_factory=DeterministicUUIDFactory("duplicate-transition"),
            **common,
        )
    with pytest.raises(
        PremiumConstructionSupervisorError, match="exact successful catalog row"
    ):
        issue_premium_construction_supervisor_transition_v2(
            catalog_a,
            selected_candidate_ids=(candidate_b,),
            id_factory=DeterministicUUIDFactory("not-yet-successful-transition"),
            **common,
        )

    domain_claim_path = transition_output_root / (
        f".claims/{transition_b.envelope.payload['claim_domain_blake3']}.json"
    )
    tampered_claim = json.loads(domain_claim_path.read_text())
    tampered_claim["catalog_blake3"] = blake3_hex("tampered-catalog")
    domain_claim_path.chmod(0o600)
    domain_claim_path.write_bytes(canonical_json_bytes(tampered_claim))
    domain_claim_path.chmod(0o400)
    with pytest.raises(PremiumConstructionSupervisorError, match="claim domain"):
        verify_premium_construction_supervisor_transition_v2(
            transition_b.root,
            publication_output_root=publication_root,
            rubrics=object(),
            legacy_python_root=tmp_path / "unused-legacy-python",
            trust_store_path=trust_path,
            image_refs_path=tmp_path / "unused-image-refs.json",
            host_private_key_path=private_path,
            host_key_id="eva-test-key",
        )
