"""External Ed25519 admission components for the campaign orchestrator.

The signer and verifier deliberately meet through one immutable JSON bundle on
disk.  The signer first replays the complete pipeline verifier; the verifier
then repeats that replay and checks the public-key signatures and exact claims.
The orchestrator itself never receives the private key.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import stat
from threading import RLock
from typing import Callable, Mapping, Protocol

from eva_agent.orchestration.contracts import SignedSupervisorEvidence
from eva_agent.pipeline.contracts import (
    Cohort,
    CompiledRubricTable,
    PipelineResult,
    RuntimeIdFactory,
    uuid_text,
)
from eva_agent.pipeline.digests import (
    blake3_hex,
    canonical_json_bytes,
    canonical_value,
    is_blake3,
)
from eva_agent.pipeline.ids import RandomUUIDFactory
from eva_agent.pipeline.verify import PipelineVerifier, VerificationReport

from .receipts import (
    AdmissionClaims,
    TrajectoryAdmissionClaims,
    SignedAdmissionPair,
    SignedEnvelope,
    issue_admission_pair,
    verify_admission_pair,
)


BUNDLE_SCHEMA = "eva.signed-admission-bundle.v1"
MAXIMUM_BUNDLE_BYTES = 4 * 1024 * 1024


class SupervisorAdmissionError(ValueError):
    """An admission context, immutable bundle, or verification replay differed."""


@dataclass(frozen=True, slots=True)
class SupervisorAdmissionContext:
    """Private deployment metadata needed to decide one candidate admission."""

    candidate_id: str
    split: str
    rubric: CompiledRubricTable
    plan_blake3: str

    def __post_init__(self) -> None:
        uuid_text(self.candidate_id, label="candidate_id")
        if self.split not in {"train", "development", "sealed_evaluation"}:
            raise SupervisorAdmissionError("admission context split differs")
        if not isinstance(self.rubric, CompiledRubricTable):
            raise SupervisorAdmissionError("admission context rubric differs")
        if not is_blake3(self.plan_blake3):
            raise SupervisorAdmissionError("admission context plan commitment differs")


class AdmissionContextResolver(Protocol):
    def __call__(self, candidate_id: str) -> SupervisorAdmissionContext: ...


def _admission_policy_passed(result: PipelineResult) -> bool:
    if result.separation.admission_policy_schema is None:
        return result.separation.policy_passed is True
    return result.separation.admission_policy_passed is True


def _selected_evaluation(result: PipelineResult):
    if result.separation.admission_policy_schema is None:
        values = tuple(row for row in result.evaluations if row.cohort is Cohort.STRONG)
        if len(values) != 1:
            raise SupervisorAdmissionError(
                "pipeline result must contain exactly one strong evaluation"
            )
        return values[0]
    digest = result.separation.selected_evaluation_blake3
    values = tuple(row for row in result.evaluations if row.evaluation_blake3 == digest)
    if len(values) != 1 or values[0].cohort.value != result.separation.selected_evaluation_cohort:
        raise SupervisorAdmissionError("selected trajectory admission evidence differs")
    return values[0]


def _claims(
    *,
    candidate_id: str,
    result: PipelineResult,
    context: SupervisorAdmissionContext,
    report: VerificationReport,
) -> AdmissionClaims | TrajectoryAdmissionClaims:
    if context.candidate_id != candidate_id:
        raise SupervisorAdmissionError("admission context candidate identity differs")
    if not report.valid or report.result_blake3 != result.result_blake3:
        raise SupervisorAdmissionError("pipeline verification report is not valid for this result")
    if (
        result.recommendation.recommendation
        != "eligible_for_signed_supervisor_admission"
        or not _admission_policy_passed(result)
    ):
        raise SupervisorAdmissionError("pipeline result is not eligible for admission")
    if len(result.sandbox_manifests) != 1:
        raise SupervisorAdmissionError("pipeline result sandbox cardinality differs")
    manifest = result.sandbox_manifests[0]
    if manifest.rubric.digest != context.rubric.digest:
        raise SupervisorAdmissionError("admission context rubric binding differs")
    selected = _selected_evaluation(result)
    if result.separation.admission_policy_schema is not None:
        return TrajectoryAdmissionClaims(
            candidate_id=candidate_id,
            sandbox_id=manifest.sandbox_id,
            split=context.split,
            pipeline_result_blake3=result.result_blake3,
            selected_evaluation_blake3=selected.evaluation_blake3,
            selected_evaluation_cohort=selected.cohort.value,
            admission_policy_blake3=result.separation.report_blake3,
            ability_separation_passed=result.separation.policy_passed,
            manifest_blake3=manifest.manifest_blake3,
            rubric_blake3=context.rubric.digest,
            verification_report_blake3=report.report_blake3,
            recommendation_blake3=result.recommendation.recommendation_blake3,
            plan_blake3=context.plan_blake3,
        )
    return AdmissionClaims(
        candidate_id=candidate_id,
        sandbox_id=manifest.sandbox_id,
        split=context.split,
        pipeline_result_blake3=result.result_blake3,
        strong_evaluation_blake3=selected.evaluation_blake3,
        manifest_blake3=manifest.manifest_blake3,
        rubric_blake3=context.rubric.digest,
        verification_report_blake3=report.report_blake3,
        recommendation_blake3=result.recommendation.recommendation_blake3,
        plan_blake3=context.plan_blake3,
    )


def _bundle_core(
    *,
    candidate_id: str,
    claims: AdmissionClaims | TrajectoryAdmissionClaims,
    report: VerificationReport,
    pair: SignedAdmissionPair,
) -> dict[str, object]:
    return {
        "schema": BUNDLE_SCHEMA,
        "candidate_id": candidate_id,
        "claims": asdict(claims),
        "verification_report": asdict(report),
        "admission": pair.admission.to_document(),
        "transition": pair.transition.to_document(),
    }


def _bundle_document(
    *,
    candidate_id: str,
    claims: AdmissionClaims | TrajectoryAdmissionClaims,
    report: VerificationReport,
    pair: SignedAdmissionPair,
) -> dict[str, object]:
    core = _bundle_core(
        candidate_id=candidate_id,
        claims=claims,
        report=report,
        pair=pair,
    )
    return {**core, "bundle_blake3": blake3_hex(core)}


def _bundle_path(root: Path, candidate_id: str) -> Path:
    uuid_text(candidate_id, label="candidate_id")
    return root / f"{candidate_id}.json"


def _ensure_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink() or not root.is_dir():
        raise SupervisorAdmissionError("admission bundle root topology differs")


def _write_once(path: Path, document: Mapping[str, object]) -> None:
    payload = canonical_json_bytes(document)
    if len(payload) > MAXIMUM_BUNDLE_BYTES:
        raise SupervisorAdmissionError("admission bundle exceeds its byte bound")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        raise
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short admission-bundle write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600)


def _read_document(path: Path) -> Mapping[str, object]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise SupervisorAdmissionError("admission bundle is absent") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size < 1
        or before.st_size > MAXIMUM_BUNDLE_BYTES
    ):
        raise SupervisorAdmissionError("admission bundle topology or size differs")
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, ValueError) as exc:
        raise SupervisorAdmissionError("admission bundle could not be decoded") from exc
    after = path.lstat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(payload) != before.st_size
    ):
        raise SupervisorAdmissionError("admission bundle changed while reading")
    if not isinstance(value, dict):
        raise SupervisorAdmissionError("admission bundle must be an object")
    return value


def _open_pair(
    document: Mapping[str, object],
    *,
    candidate_id: str,
    claims: AdmissionClaims | TrajectoryAdmissionClaims,
    report: VerificationReport,
) -> tuple[SignedAdmissionPair, str]:
    expected_keys = {
        "schema",
        "candidate_id",
        "claims",
        "verification_report",
        "admission",
        "transition",
        "bundle_blake3",
    }
    if set(document) != expected_keys or document.get("schema") != BUNDLE_SCHEMA:
        raise SupervisorAdmissionError("admission bundle shape or schema differs")
    if document.get("candidate_id") != candidate_id:
        raise SupervisorAdmissionError("admission bundle candidate identity differs")
    if canonical_value(document.get("claims")) != canonical_value(asdict(claims)):
        raise SupervisorAdmissionError("admission bundle exact claims differ")
    if canonical_value(document.get("verification_report")) != canonical_value(asdict(report)):
        raise SupervisorAdmissionError("admission bundle verification report differs")
    admission = document.get("admission")
    transition = document.get("transition")
    if not isinstance(admission, Mapping) or not isinstance(transition, Mapping):
        raise SupervisorAdmissionError("admission bundle envelope shape differs")
    pair = SignedAdmissionPair(
        admission=SignedEnvelope.from_document(admission),
        transition=SignedEnvelope.from_document(transition),
    )
    core = _bundle_core(
        candidate_id=candidate_id,
        claims=claims,
        report=report,
        pair=pair,
    )
    digest = document.get("bundle_blake3")
    if not isinstance(digest, str) or digest != blake3_hex(core):
        raise SupervisorAdmissionError("admission bundle BLAKE3 differs")
    return pair, digest


def _evidence(
    *,
    claims: AdmissionClaims | TrajectoryAdmissionClaims,
    pair: SignedAdmissionPair,
    signature_bundle_blake3: str,
) -> SignedSupervisorEvidence:
    return SignedSupervisorEvidence(
        candidate_id=claims.candidate_id,
        sandbox_id=claims.sandbox_id,
        pipeline_result_blake3=claims.pipeline_result_blake3,
        manifest_blake3=claims.manifest_blake3,
        admission_receipt_blake3=pair.admission.envelope_blake3,
        supervisor_transition_blake3=pair.transition.envelope_blake3,
        signature_bundle_blake3=signature_bundle_blake3,
    )


class Ed25519AdmissionSupervisor:
    """Verify, sign, persist once, and return external supervisor evidence."""

    def __init__(
        self,
        *,
        output_root: Path,
        pipeline_verifier: PipelineVerifier,
        context_resolver: AdmissionContextResolver,
        key_id: str,
        private_key_path: Path,
        trust_store_path: Path,
        id_factory: RuntimeIdFactory | None = None,
        issued_at_factory: Callable[[], str] | None = None,
    ) -> None:
        self._root = Path(output_root)
        _ensure_root(self._root)
        self._pipeline_verifier = pipeline_verifier
        self._context_resolver = context_resolver
        self._key_id = key_id
        self._private_key_path = Path(private_key_path)
        self._trust_store_path = Path(trust_store_path)
        self._ids = id_factory or RandomUUIDFactory()
        self._issued_at_factory = issued_at_factory
        self._lock = RLock()

    def evidence_for(
        self, *, candidate_id: str, result: PipelineResult
    ) -> SignedSupervisorEvidence | None:
        if (
            result.recommendation.recommendation
            != "eligible_for_signed_supervisor_admission"
            or not _admission_policy_passed(result)
        ):
            return None
        context = self._context_resolver(candidate_id)
        report = self._pipeline_verifier.verify_or_raise(
            result=result,
            rubric=context.rubric,
        )
        claims = _claims(
            candidate_id=candidate_id,
            result=result,
            context=context,
            report=report,
        )
        path = _bundle_path(self._root, candidate_id)
        with self._lock:
            if not path.exists() and not path.is_symlink():
                pair = issue_admission_pair(
                    claims,
                    key_id=self._key_id,
                    private_key_path=self._private_key_path,
                    issued_at_utc=(
                        self._issued_at_factory() if self._issued_at_factory is not None else None
                    ),
                    receipt_id=self._ids.new("admission-receipt"),
                    transition_id=self._ids.new("admission-transition"),
                )
                document = _bundle_document(
                    candidate_id=candidate_id,
                    claims=claims,
                    report=report,
                    pair=pair,
                )
                try:
                    _write_once(path, document)
                except FileExistsError:
                    pass
            document = _read_document(path)
        pair, bundle_blake3 = _open_pair(
            document,
            candidate_id=candidate_id,
            claims=claims,
            report=report,
        )
        verify_admission_pair(
            pair,
            claims=claims,
            trust_store_path=self._trust_store_path,
        )
        return _evidence(
            claims=claims,
            pair=pair,
            signature_bundle_blake3=bundle_blake3,
        )


class Ed25519SupervisorSignatureVerifier:
    """Public-key-only verifier that independently replays the full result."""

    def __init__(
        self,
        *,
        bundle_root: Path,
        pipeline_verifier: PipelineVerifier,
        context_resolver: AdmissionContextResolver,
        trust_store_path: Path,
    ) -> None:
        self._root = Path(bundle_root)
        _ensure_root(self._root)
        self._pipeline_verifier = pipeline_verifier
        self._context_resolver = context_resolver
        self._trust_store_path = Path(trust_store_path)

    def verify(
        self,
        *,
        candidate_id: str,
        result: PipelineResult,
        evidence: SignedSupervisorEvidence,
    ) -> bool:
        try:
            if evidence.candidate_id != candidate_id:
                return False
            context = self._context_resolver(candidate_id)
            report = self._pipeline_verifier.verify_or_raise(
                result=result,
                rubric=context.rubric,
            )
            claims = _claims(
                candidate_id=candidate_id,
                result=result,
                context=context,
                report=report,
            )
            document = _read_document(_bundle_path(self._root, candidate_id))
            pair, bundle_blake3 = _open_pair(
                document,
                candidate_id=candidate_id,
                claims=claims,
                report=report,
            )
            verified = verify_admission_pair(
                pair,
                claims=claims,
                trust_store_path=self._trust_store_path,
            )
            expected = _evidence(
                claims=claims,
                pair=pair,
                signature_bundle_blake3=bundle_blake3,
            )
            return (
                verified.pipeline_result_blake3 == claims.pipeline_result_blake3
                and canonical_value(expected) == canonical_value(evidence)
            )
        except Exception:
            return False


__all__ = [
    "AdmissionContextResolver",
    "Ed25519AdmissionSupervisor",
    "Ed25519SupervisorSignatureVerifier",
    "SupervisorAdmissionContext",
    "SupervisorAdmissionError",
]
