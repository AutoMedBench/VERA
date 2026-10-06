from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from eva_agent.campaign import CampaignLedger
from eva_agent.orchestration import (
    CampaignOrchestrator,
    OrchestrationConfig,
    SignedSupervisorEvidence,
)
from eva_agent.pipeline.digests import blake3_hex

from test_orchestration_parallel import _FakePipeline, _Source, _ConcurrencyTracker, _enqueue


def _config() -> OrchestrationConfig:
    return OrchestrationConfig(
        worker_width=1,
        queue_capacity=1,
        claim_batch_size=1,
        lease_seconds=10,
        heartbeat_seconds=0.1,
        progress_seconds=0.1,
    )


class _ExternalSupervisor:
    def evidence_for(self, *, candidate_id, result):
        manifest = result.sandbox_manifests[0]
        return SignedSupervisorEvidence(
            candidate_id=candidate_id,
            sandbox_id=manifest.sandbox_id,
            pipeline_result_blake3=result.result_blake3,
            manifest_blake3=manifest.manifest_blake3,
            admission_receipt_blake3=blake3_hex({"admission": candidate_id}),
            supervisor_transition_blake3=blake3_hex({"transition": candidate_id}),
            signature_bundle_blake3=blake3_hex({"signatures": candidate_id}),
        )


class _ExternalVerifier:
    def __init__(self, accepted: bool) -> None:
        self.accepted = accepted
        self.calls = 0

    def verify(self, *, candidate_id, result, evidence):
        self.calls += 1
        return self.accepted


def _run(
    *,
    tmp_path: Path,
    ledger: CampaignLedger,
    candidate_id: str,
    supervisor=None,
    verifier=None,
):
    source = _Source([candidate_id])
    tracker = _ConcurrencyTracker(1)
    return CampaignOrchestrator(
        ledger=ledger,
        candidate_source=source,
        pipeline_factory=lambda _worker_id: _FakePipeline(tracker),
        receipt_root=tmp_path / "receipts",
        config=_config(),
        supervisor_evidence_source=supervisor,
        supervisor_signature_verifier=verifier,
    ).run_until_idle()


def test_admission_requires_external_evidence_and_positive_signature_verifier(
    tmp_path: Path,
) -> None:
    ledger = CampaignLedger(tmp_path / "campaign.sqlite3")

    no_evidence = str(uuid4())
    _enqueue(ledger, [no_evidence])
    first = _run(tmp_path=tmp_path, ledger=ledger, candidate_id=no_evidence)
    assert first.reviewed == 0
    assert first.infrastructure_quarantine == 1
    assert first.admitted == ledger.progress().admitted == 0

    invalid_evidence = str(uuid4())
    ledger.enqueue_candidate(
        candidate_id=invalid_evidence,
        source_episode_id="invalid-evidence-source",
        domain="medxpertqa",
        stage="E2E",
        split="train",
        rubric_blake3="a" * 64,
    )
    rejected_verifier = _ExternalVerifier(False)
    second = _run(
        tmp_path=tmp_path,
        ledger=ledger,
        candidate_id=invalid_evidence,
        supervisor=_ExternalSupervisor(),
        verifier=rejected_verifier,
    )
    assert rejected_verifier.calls == 1
    assert second.reviewed == 0
    assert second.infrastructure_quarantine == 1
    assert second.admitted == ledger.progress().admitted == 0

    signed = str(uuid4())
    ledger.enqueue_candidate(
        candidate_id=signed,
        source_episode_id="signed-source",
        domain="medxpertqa",
        stage="E2E",
        split="train",
        rubric_blake3="a" * 64,
    )
    accepted_verifier = _ExternalVerifier(True)
    third = _run(
        tmp_path=tmp_path,
        ledger=ledger,
        candidate_id=signed,
        supervisor=_ExternalSupervisor(),
        verifier=accepted_verifier,
    )
    assert accepted_verifier.calls == 1
    assert third.reviewed == third.admitted == 1
    assert ledger.progress().admitted == 1
