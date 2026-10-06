from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eva_agent.admission.receipts import SignedEnvelope, issue_signed_envelope
from eva_agent.campaign.selection_v2 import (
    CampaignSelectionV2,
    SELECTION_V2_METHOD,
    SELECTION_V2_SCHEMA,
)
from eva_agent.pipeline.contracts import Stage
from eva_agent.pipeline.digests import blake3_hex
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.sources.legacy_execution import (
    CandidateBindingReference,
    LegacyExecutionBindingCatalog,
    ProductionSourceBlocker,
)
from eva_agent.sources.prospective_execution_catalog import (
    build_prospective_execution_catalog,
)
from eva_agent.sources.prospective_execution_catalog_v3 import (
    ProspectiveExecutionCatalogV3Error,
    build_prospective_execution_catalog_v3,
    verify_prospective_execution_catalog_v3,
)


ROOT = Path(__file__).resolve().parents[1]


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
                "keys": {
                    "catalog-test": base64.b64encode(public).decode("ascii")
                },
            }
        ),
        encoding="utf-8",
    )
    return private_path, trust_path


def _selection(rubrics) -> CampaignSelectionV2:
    rubric_rows = tuple(sorted(rubrics.rubrics, key=lambda row: (row.domain, row.stage)))
    entries = []
    for index in range(9_000):
        rubric = rubric_rows[index % len(rubric_rows)]
        candidate_id = str(uuid5(NAMESPACE_URL, f"eva:test:catalog-v3:{index}"))
        readiness = (
            "promoted" if index < 1_344 else "rejected" if index >= 8_989 else "frozen"
        )
        entries.append(
            {
                "candidate_id": candidate_id,
                "source_candidate_id": f"source-{index:04d}",
                "source_family": "fixture",
                "source_artifact_sha256": f"{index + 10:064x}",
                "domain": rubric.domain,
                "stage": rubric.stage,
                "split": "train",
                "rubric_id": rubric.rubric_id,
                "rubric_blake3": rubric.digest,
                "source_identity_rank_blake3": blake3_hex(f"identity:{index}"),
                "construction_readiness": readiness,
                "readiness_proof_root_blake3": blake3_hex(f"proof:{index}"),
                "selection_rank_blake3": blake3_hex(f"selection-rank:{index}"),
                "cell_rank": index + 1,
                "cell_target": 6_000,
                "selection_tier": "primary" if index < 6_000 else "reserve",
                "queue_ordinal": index + 1,
            }
        )
    core = {
        "schema": SELECTION_V2_SCHEMA,
        "selection_id": str(uuid5(NAMESPACE_URL, "eva:test:catalog-v3:selection")),
        "plan_id": "all-train-test-plan",
        "plan_blake3": blake3_hex("all-train-plan"),
        "source_registry_blake3": blake3_hex("source-registry"),
        "upstream_source_registry_sha256": "a" * 64,
        "rubric_registry_blake3": rubrics.digest,
        "base_selection_id": str(uuid5(NAMESPACE_URL, "eva:test:catalog-v3:base")),
        "base_selection_blake3": blake3_hex("base-selection"),
        "readiness_authority": {"authority_blake3": blake3_hex("readiness")},
        "selection_method": SELECTION_V2_METHOD,
        "selected_count": 6_000,
        "scheduled_count": 9_000,
        "reserve_count": 3_000,
        "entries": entries,
    }
    return CampaignSelectionV2.from_document(
        {**core, "selection_blake3": blake3_hex(core)}
    )


class _LegacyMaterializer:
    def __init__(self, blocked_sources: set[str]) -> None:
        self.blocked_sources = blocked_sources

    def resolve(self, candidate_id: str, *, source_candidate_id: str):
        if source_candidate_id in self.blocked_sources:
            raise ProductionSourceBlocker("fixture materialization blocker")
        return SimpleNamespace(
            candidate_id=candidate_id, source_candidate_id=source_candidate_id
        )


def _signed_legacy_r2(
    selection: CampaignSelectionV2,
    rubrics,
    *,
    private_path: Path,
) -> SignedEnvelope:
    promoted = tuple(
        row for row in selection.entries if row.construction_readiness == "promoted"
    )
    references = tuple(
        CandidateBindingReference(
            source_candidate_id=row.source_candidate_id,
            source_family=row.source_family,
            stage=Stage(row.stage),
            promoted_subject_path=ROOT,
            promoted_subject_blake3=blake3_hex(f"subject:{row.candidate_id}"),
            source_policy_path=ROOT,
            source_policy_upstream_sha256=blake3_hex(
                f"upstream-policy:{row.candidate_id}"
            ),
            source_policy_blake3=blake3_hex(f"policy:{row.candidate_id}"),
            construction_manifest_path=ROOT,
            construction_manifest_upstream_sha256=blake3_hex(
                f"upstream-manifest:{row.candidate_id}"
            ),
            construction_manifest_blake3=blake3_hex(
                f"manifest:{row.candidate_id}"
            ),
            construction_attestation_path=ROOT,
            construction_attestation_blake3=blake3_hex(
                f"attestation:{row.candidate_id}"
            ),
            proof_root_blake3=row.readiness_proof_root_blake3,
        )
        for row in promoted
    )
    legacy_catalog = LegacyExecutionBindingCatalog(
        authority_blake3=blake3_hex("legacy-authority"),
        legacy_runtime_source_blake3=blake3_hex("legacy-runtime"),
        rows=references,
        catalog_blake3=blake3_hex("legacy-binding-catalog"),
    )
    legacy_entries = []
    for index, row in enumerate(selection.entries):
        document = row.to_document()
        if row.selection_tier == "primary":
            document["split"] = (
                "train"
                if index < 5_000
                else "development"
                if index < 5_500
                else "sealed_evaluation"
            )
        legacy_entries.append(
            SimpleNamespace(
                **document,
                to_document=lambda value=document: dict(value),
            )
        )
    legacy_selection = SimpleNamespace(
        entries=tuple(legacy_entries),
        rubric_registry_blake3=rubrics.digest,
        selection_blake3=blake3_hex("split-selection"),
        base_selection_blake3=selection.selection_blake3,
        plan_blake3=blake3_hex("split-plan"),
        source_registry_blake3=selection.source_registry_blake3,
    )
    blocked = {row.source_candidate_id for row in promoted[-8:]}
    source_imports = {
        row.candidate_id: blake3_hex(f"source-import:{row.candidate_id}")
        for row in selection.entries
    }
    legacy_r2 = build_prospective_execution_catalog(
        selection=legacy_selection,
        rubrics=rubrics,
        legacy_catalog=legacy_catalog,
        legacy_binding_materializer=_LegacyMaterializer(blocked),
        source_import_blake3s=source_imports,
        catalog_revision=2,
    )
    assert legacy_r2.to_document()["executable_count"] == 1_336
    return issue_signed_envelope(
        legacy_r2.to_document(),
        key_id="catalog-test",
        private_key_path=private_path,
    )


@dataclass(frozen=True)
class _Reference:
    document: dict

    def to_document(self):
        return dict(self.document)


class _PremiumResolver:
    def __init__(
        self,
        selection: CampaignSelectionV2,
        rows,
        *,
        private_path: Path,
    ) -> None:
        bindings = []
        for index, selected in enumerate(rows):
            catalog_binding = {
                "publication_id": str(
                    uuid5(NAMESPACE_URL, f"eva:test:publication:{selected.candidate_id}")
                ),
                "binding_blake3": blake3_hex(
                    f"premium-binding:{selected.candidate_id}"
                ),
            }
            bindings.append(
                {
                    "candidate_id": selected.candidate_id,
                    "source_candidate_id": selected.source_candidate_id,
                    "source_family": selected.source_family,
                    "domain": selected.domain,
                    "stage": selected.stage,
                    "split": selected.split,
                    "rubric_id": selected.rubric_id,
                    "rubric_blake3": selected.rubric_blake3,
                    "catalog_binding": catalog_binding,
                    "fixture_ordinal": index,
                }
            )
        selected_ids = [row.candidate_id for row in rows]
        selected_ids_blake3 = blake3_hex(
            {
                "schema": "eva.premium-construction-selected-candidate-ids.v2",
                "candidate_ids": selected_ids,
            }
        )
        catalog_blake3 = blake3_hex("premium-proof-catalog")
        payload = {
            "schema": "eva.premium-construction-supervisor-transition.v2",
            "transition_id": str(uuid5(NAMESPACE_URL, "eva:test:premium-transition")),
            "selection_blake3": selection.selection_blake3,
            "rubric_registry_blake3": selection.rubric_registry_blake3,
            "from_state": "frozen",
            "to_state": "executable_ready",
            "catalog_blake3": catalog_blake3,
            "selected_candidate_ids": selected_ids,
            "selected_candidate_ids_blake3": selected_ids_blake3,
            "claim_domain_blake3": blake3_hex(
                {
                    "schema": "eva.premium-construction-supervisor-claim-domain.v2",
                    "catalog_blake3": catalog_blake3,
                    "selected_candidate_ids_blake3": selected_ids_blake3,
                }
            ),
            "bindings": bindings,
            "binding_count": len(bindings),
            "controls": {
                "provider_calls": 0,
                "ledger_writes": 0,
                "append_only": True,
                "admission_eligibility_claimed": False,
                "proof_catalog_coverage": "complete_cumulative",
                "transition_selection": "explicit_successful_candidate_set",
            },
        }
        envelope = issue_signed_envelope(
            payload, key_id="catalog-test", private_key_path=private_path
        )
        references = tuple(
            _Reference(
                {
                    "candidate_id": binding["candidate_id"],
                    "source_candidate_id": binding["source_candidate_id"],
                    "source_family": binding["source_family"],
                    "domain": binding["domain"],
                    "stage": binding["stage"],
                    "split": binding["split"],
                    "rubric_id": binding["rubric_id"],
                    "rubric_blake3": binding["rubric_blake3"],
                    "publication_id": binding["catalog_binding"]["publication_id"],
                    "binding_blake3": binding["catalog_binding"]["binding_blake3"],
                    "transition_binding_blake3": blake3_hex(binding),
                }
            )
            for binding in bindings
        )
        self.transition = SimpleNamespace(
            envelope=envelope,
            transition_blake3=envelope.envelope_blake3,
            binding_count=len(bindings),
        )
        self._references = references
        self.executable_candidate_count = len(references)
        self.catalog_inventory_blake3 = blake3_hex(
            {
                "schema": "eva.premium-candidate-execution-binding-catalog.v1",
                "signed_transition_blake3": envelope.envelope_blake3,
                "selection_blake3": selection.selection_blake3,
                "coverage": "signed_premium_executable_ready_only",
                "candidate_count": len(references),
                "rows": [row.to_document() for row in references],
            }
        )

    def inventory(self):
        return self._references


@pytest.fixture(scope="module")
def rubrics():
    return load_and_compile_registry(ROOT / "rubrics/source/domain-stage-tables.v1.json")


@pytest.fixture(scope="module")
def catalog_inputs(tmp_path_factory, rubrics):
    tmp_path = tmp_path_factory.mktemp("catalog-v3")
    private_path, trust_path = _key_material(tmp_path)
    selection = _selection(rubrics)
    legacy = _signed_legacy_r2(
        selection, rubrics, private_path=private_path
    )
    return selection, legacy, private_path, trust_path


def test_n0_reuses_exact_1336_legacy_and_all_6000_primaries_are_train(
    catalog_inputs, rubrics
) -> None:
    selection, legacy, _private, trust = catalog_inputs
    catalog = build_prospective_execution_catalog_v3(
        selection=selection,
        rubrics=rubrics,
        signed_legacy_catalog=legacy,
        trust_store_path=trust,
    )
    document = catalog.to_document()
    assert document["legacy_executable_count"] == 1_336
    assert document["premium_executable_count"] == 0
    assert document["executable_count"] == 1_336
    assert document["source_only_count"] == 7_653
    assert document["rejected_count"] == 11
    assert document["primary_split_counts"] == {"train": 6_000}
    assert {row["split"] for row in document["rows"]} == {"train"}
    assert document["premium_transitions"] == []
    assert len(catalog.executable_candidate_ids) == 1_336
    legacy_rows = {
        row["candidate_id"]: row for row in legacy.payload["rows"]
    }
    first_legacy = next(
        row for row in document["rows"] if row["execution_status"] == "executable_legacy"
    )
    assert first_legacy["execution_proof"] == legacy_rows[
        first_legacy["candidate_id"]
    ]["execution_proof"]
    assert verify_prospective_execution_catalog_v3(
        document,
        selection=selection,
        rubrics=rubrics,
        signed_legacy_catalog=legacy,
        trust_store_path=trust,
    ).catalog_blake3 == catalog.catalog_blake3


def test_fake_signed_transition_adds_only_its_exact_frozen_rows(
    catalog_inputs, rubrics
) -> None:
    selection, legacy, private, trust = catalog_inputs
    frozen = tuple(
        row for row in selection.entries if row.construction_readiness == "frozen"
    )
    resolver = _PremiumResolver(selection, frozen[:2], private_path=private)
    catalog = build_prospective_execution_catalog_v3(
        selection=selection,
        rubrics=rubrics,
        signed_legacy_catalog=legacy,
        premium_resolvers=(resolver,),
        trust_store_path=trust,
    )
    document = catalog.to_document()
    assert document["executable_count"] == 1_336 + 2
    assert document["premium_executable_count"] == 2
    assert document["source_only_count"] == 7_651
    rows = {row["candidate_id"]: row for row in document["rows"]}
    for selected in frozen[:2]:
        assert rows[selected.candidate_id]["construction_readiness"] == "frozen"
        assert rows[selected.candidate_id]["execution_status"] == "executable_premium"
        assert rows[selected.candidate_id]["execution_proof"]["kind"] == (
            "signed_premium_construction_transition_v2"
        )
    assert rows[frozen[2].candidate_id]["execution_status"] == "source_only"
    assert rows[frozen[2].candidate_id]["execution_proof"] is None


def test_unsigned_tamper_and_catalog_tamper_fail_closed(catalog_inputs, rubrics) -> None:
    selection, legacy, private, trust = catalog_inputs
    legacy_document = legacy.to_document()
    legacy_document["payload"]["rows"][0]["execution_status"] = "source_only"
    tampered_legacy = SignedEnvelope.from_document(legacy_document)
    with pytest.raises(ProspectiveExecutionCatalogV3Error, match="signature"):
        build_prospective_execution_catalog_v3(
            selection=selection,
            rubrics=rubrics,
            signed_legacy_catalog=tampered_legacy,
            trust_store_path=trust,
        )

    frozen = next(
        row for row in selection.entries if row.construction_readiness == "frozen"
    )
    resolver_mismatch = _PremiumResolver(selection, (frozen,), private_path=private)
    reference_document = resolver_mismatch._references[0].to_document()
    reference_document["binding_blake3"] = blake3_hex("different-binding")
    resolver_mismatch._references = (_Reference(reference_document),)
    with pytest.raises(
        ProspectiveExecutionCatalogV3Error, match="resolver/signed bindings"
    ):
        build_prospective_execution_catalog_v3(
            selection=selection,
            rubrics=rubrics,
            signed_legacy_catalog=legacy,
            premium_resolvers=(resolver_mismatch,),
            trust_store_path=trust,
        )

    resolver = _PremiumResolver(selection, (frozen,), private_path=private)
    transition_document = resolver.transition.envelope.to_document()
    transition_document["payload"]["bindings"][0]["domain"] = "tampered"
    resolver.transition = SimpleNamespace(
        envelope=SignedEnvelope.from_document(transition_document),
        transition_blake3=resolver.transition.transition_blake3,
        binding_count=1,
    )
    with pytest.raises(ProspectiveExecutionCatalogV3Error, match="signature"):
        build_prospective_execution_catalog_v3(
            selection=selection,
            rubrics=rubrics,
            signed_legacy_catalog=legacy,
            premium_resolvers=(resolver,),
            trust_store_path=trust,
        )

    catalog = build_prospective_execution_catalog_v3(
        selection=selection,
        rubrics=rubrics,
        signed_legacy_catalog=legacy,
        trust_store_path=trust,
    ).to_document()
    catalog["primary_split_counts"] = {"train": 5_999, "development": 1}
    with pytest.raises(ProspectiveExecutionCatalogV3Error, match="derivation"):
        verify_prospective_execution_catalog_v3(
            catalog,
            selection=selection,
            rubrics=rubrics,
            signed_legacy_catalog=legacy,
            trust_store_path=trust,
        )


def test_signed_transition_cannot_upgrade_promoted_or_change_6000_0_0_split(
    catalog_inputs, rubrics
) -> None:
    selection, legacy, private, trust = catalog_inputs
    blocked_promoted = tuple(
        row for row in selection.entries if row.construction_readiness == "promoted"
    )[-1]
    resolver = _PremiumResolver(selection, (blocked_promoted,), private_path=private)
    with pytest.raises(ProspectiveExecutionCatalogV3Error, match="only its exact frozen"):
        build_prospective_execution_catalog_v3(
            selection=selection,
            rubrics=rubrics,
            signed_legacy_catalog=legacy,
            premium_resolvers=(resolver,),
            trust_store_path=trust,
        )

    selection_document = selection.to_document()
    selection_document["entries"][0]["split"] = "development"
    core = {
        key: value
        for key, value in selection_document.items()
        if key != "selection_blake3"
    }
    selection_document["selection_blake3"] = blake3_hex(core)
    mixed = CampaignSelectionV2.from_document(selection_document)
    with pytest.raises(ProspectiveExecutionCatalogV3Error, match="6,000/0/0"):
        build_prospective_execution_catalog_v3(
            selection=mixed,
            rubrics=rubrics,
            signed_legacy_catalog=legacy,
            trust_store_path=trust,
        )
