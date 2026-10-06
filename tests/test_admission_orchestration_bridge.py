from __future__ import annotations

import base64
import json
from pathlib import Path
from uuid import uuid4

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eva_agent.admission import (
    Ed25519AdmissionSupervisor,
    Ed25519SupervisorSignatureVerifier,
    SupervisorAdmissionContext,
)
from eva_agent.campaign import CampaignLedger
from eva_agent.orchestration import (
    CampaignOrchestrator,
    CandidateJob,
    OrchestrationConfig,
)
from eva_agent.pipeline import DeterministicUUIDFactory, PipelineVerifier
from eva_agent.pipeline.digests import blake3_hex

from test_pipeline_e2e import _build_pipeline


def _keys(tmp_path: Path) -> tuple[Path, Path]:
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "signing.pem"
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
    trust = tmp_path / "trust.json"
    trust.write_text(
        json.dumps(
            {
                "schema": "eva.ed25519-trust-store.v1",
                "status": "active",
                "algorithm": "Ed25519",
                "keys": {"eva-test-key": base64.b64encode(public).decode("ascii")},
            }
        ),
        encoding="utf-8",
    )
    return private_path, trust


class _Source:
    def __init__(self, job: CandidateJob) -> None:
        self.job = job

    def load(self, candidate_id: str) -> CandidateJob:
        assert candidate_id == self.job.candidate_id
        return self.job


class _CompletedPipeline:
    def __init__(self, result) -> None:
        self.result = result

    def run(self, *, episode, rubric, targets):
        del episode, rubric, targets
        return self.result


def _fixture(tmp_path: Path):
    pipeline, episode, rubric, targets, artifacts, _clients, _judge = _build_pipeline(
        tmp_path / "pipeline"
    )
    result = pipeline.run(episode=episode, rubric=rubric, targets=targets)
    candidate_id = str(uuid4())
    context = SupervisorAdmissionContext(
        candidate_id=candidate_id,
        split="train",
        rubric=rubric,
        plan_blake3=blake3_hex("campaign-plan"),
    )
    private_path, trust_path = _keys(tmp_path)
    bundle_root = tmp_path / "signed-admissions"
    supervisor = Ed25519AdmissionSupervisor(
        output_root=bundle_root,
        pipeline_verifier=PipelineVerifier(artifacts),
        context_resolver=lambda requested: context if requested == candidate_id else None,
        key_id="eva-test-key",
        private_key_path=private_path,
        trust_store_path=trust_path,
        id_factory=DeterministicUUIDFactory("admission-supervisor"),
        issued_at_factory=lambda: "2026-09-07T13:00:00Z",
    )
    verifier = Ed25519SupervisorSignatureVerifier(
        bundle_root=bundle_root,
        pipeline_verifier=PipelineVerifier(artifacts),
        context_resolver=lambda requested: context if requested == candidate_id else None,
        trust_store_path=trust_path,
    )
    return (
        candidate_id,
        result,
        episode,
        rubric,
        targets,
        supervisor,
        verifier,
        bundle_root,
    )


def test_signed_supervisor_bundle_is_idempotent_and_publicly_reverified(
    tmp_path: Path,
) -> None:
    candidate_id, result, _episode, _rubric, _targets, supervisor, verifier, root = (
        _fixture(tmp_path)
    )

    first = supervisor.evidence_for(candidate_id=candidate_id, result=result)
    assert first is not None
    path = root / f"{candidate_id}.json"
    first_bytes = path.read_bytes()
    second = supervisor.evidence_for(candidate_id=candidate_id, result=result)
    assert second == first
    assert path.read_bytes() == first_bytes
    assert verifier.verify(candidate_id=candidate_id, result=result, evidence=first) is True

    document = json.loads(first_bytes)
    document["claims"]["split"] = "development"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert verifier.verify(candidate_id=candidate_id, result=result, evidence=first) is False


def test_real_signed_bridge_commits_one_ledger_admission(tmp_path: Path) -> None:
    candidate_id, result, episode, rubric, targets, supervisor, verifier, _root = _fixture(
        tmp_path
    )
    ledger = CampaignLedger(tmp_path / "campaign.sqlite3")
    ledger.enqueue_candidate(
        candidate_id=candidate_id,
        source_episode_id=episode.episode_id,
        domain=episode.domain,
        stage=episode.stage.value,
        split="train",
        rubric_blake3=rubric.digest,
    )
    job = CandidateJob(
        candidate_id=candidate_id,
        episode=episode,
        rubric=rubric,
        targets=targets,
    )
    report = CampaignOrchestrator(
        ledger=ledger,
        candidate_source=_Source(job),
        pipeline_factory=lambda _worker: _CompletedPipeline(result),
        receipt_root=tmp_path / "orchestration-receipts",
        config=OrchestrationConfig(
            worker_width=1,
            queue_capacity=1,
            claim_batch_size=1,
            lease_seconds=10,
            heartbeat_seconds=0.1,
            progress_seconds=0.1,
        ),
        supervisor_evidence_source=supervisor,
        supervisor_signature_verifier=verifier,
    ).run_until_idle()

    assert report.reviewed == 1
    assert report.admitted == 1
    assert ledger.progress().admitted == 1
