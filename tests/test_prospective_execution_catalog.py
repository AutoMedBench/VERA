from types import SimpleNamespace
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from eva_agent.pipeline.contracts import Stage
from eva_agent.pipeline.digests import blake3_hex
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.sources.legacy_execution import (
    CandidateBindingReference,
    LegacyExecutionBindingCatalog,
    ProductionSourceBlocker,
)
from eva_agent.sources.prospective_execution_catalog import (
    ProspectiveExecutionCatalogError,
    build_prospective_execution_catalog,
    verify_prospective_execution_catalog,
)


def _fixture(root):
    rubrics = load_and_compile_registry(root / "rubrics/source/domain-stage-tables.v1.json")
    rubric_rows = sorted(rubrics.rubrics, key=lambda row: (row.domain, row.stage))
    entries, references, imports = [], [], {}
    for index in range(9_000):
        rubric = rubric_rows[index % len(rubric_rows)]
        candidate_id = str(uuid5(NAMESPACE_URL, f"eva:test:binding:{index}"))
        source_id = f"source-{index:04d}"
        readiness = "promoted" if index < 1_344 else "rejected" if index >= 8_989 else "frozen"
        split = "train" if index < 5_000 else "development" if index < 5_500 else "sealed_evaluation" if index < 6_000 else "train"
        entry_document = {
            "candidate_id": candidate_id, "source_candidate_id": source_id,
            "queue_ordinal": index + 1, "selection_tier": "primary" if index < 6_000 else "reserve",
            "split": split, "source_family": "fixture", "domain": rubric.domain,
            "stage": rubric.stage, "source_artifact_sha256": f"{index:064x}",
            "rubric_id": rubric.rubric_id, "rubric_blake3": rubric.digest,
            "construction_readiness": readiness,
            "readiness_proof_root_blake3": blake3_hex(f"proof:{index}"),
        }
        entries.append(SimpleNamespace(**entry_document, to_document=lambda value=entry_document: dict(value)))
        imports[candidate_id] = blake3_hex(f"import:{index}")
        if readiness == "promoted":
            references.append(CandidateBindingReference(
                source_candidate_id=source_id, source_family="fixture", stage=Stage(rubric.stage),
                promoted_subject_path=root, promoted_subject_blake3=blake3_hex(f"subject:{index}"),
                source_policy_path=root, source_policy_upstream_sha256=f"{index + 1:064x}",
                source_policy_blake3=blake3_hex(f"policy:{index}"), construction_manifest_path=root,
                construction_manifest_upstream_sha256=f"{index + 2:064x}",
                construction_manifest_blake3=blake3_hex(f"manifest:{index}"),
                construction_attestation_path=root,
                construction_attestation_blake3=blake3_hex(f"attestation:{index}"),
                proof_root_blake3=entry_document["readiness_proof_root_blake3"],
            ))
    selection = SimpleNamespace(
        entries=tuple(entries), rubric_registry_blake3=rubrics.digest,
        selection_blake3=blake3_hex("selection"), base_selection_blake3=blake3_hex("base"),
        plan_blake3=blake3_hex("plan"), source_registry_blake3=blake3_hex("sources"),
    )
    legacy = LegacyExecutionBindingCatalog(
        authority_blake3=blake3_hex("authority"), legacy_runtime_source_blake3=blake3_hex("runtime"),
        rows=tuple(references), catalog_blake3=blake3_hex("legacy-catalog"),
    )
    return selection, rubrics, legacy, imports


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Materializer:
    def __init__(self, *, embedded_episode_source=None):
        self.embedded_episode_source = embedded_episode_source
        self.calls = []

    def resolve(self, candidate_id, *, source_candidate_id):
        self.calls.append((candidate_id, source_candidate_id))
        if source_candidate_id == self.embedded_episode_source:
            raise ProductionSourceBlocker(
                "candidate policy and real stage contracts do not execute together"
            ) from ValueError(
                "public policy embeds the template episode ID instead of using host bindings"
            )
        return SimpleNamespace(
            candidate_id=candidate_id, source_candidate_id=source_candidate_id
        )


def test_all_9000_rows_are_bound_but_only_signed_promotions_are_executable():
    selection, rubrics, legacy, imports = _fixture(PROJECT_ROOT)
    materializer = _Materializer()
    catalog = build_prospective_execution_catalog(
        selection=selection, rubrics=rubrics, legacy_catalog=legacy,
        legacy_binding_materializer=materializer,
        source_import_blake3s=imports,
    )
    document = catalog.to_document()
    assert (document["scheduled_count"], document["executable_count"], document["source_only_count"], document["rejected_count"]) == (9_000, 1_344, 7_645, 11)
    assert document["primary_split_counts"] == {"development": 500, "sealed_evaluation": 500, "train": 5_000}
    assert len(document["rubric_profiles"]) == 42
    assert sum(len(row["items"]) for row in document["rubric_profiles"]) == 252
    assert verify_prospective_execution_catalog(
        document, selection=selection, rubrics=rubrics, legacy_catalog=legacy,
        legacy_binding_materializer=materializer,
        source_import_blake3s=imports,
    ).catalog_blake3 == catalog.catalog_blake3
    assert len(materializer.calls) == 2 * 1_344


def test_missing_source_receipt_fails_closed():
    selection, rubrics, legacy, imports = _fixture(PROJECT_ROOT)
    imports.pop(next(iter(imports)))
    with pytest.raises(ProspectiveExecutionCatalogError, match="source import receipt inventory"):
        build_prospective_execution_catalog(
            selection=selection, rubrics=rubrics, legacy_catalog=legacy,
            legacy_binding_materializer=_Materializer(),
            source_import_blake3s=imports,
        )


def test_embedded_template_episode_id_is_digest_bound_source_only():
    selection, rubrics, legacy, imports = _fixture(PROJECT_ROOT)
    blocked_source = selection.entries[0].source_candidate_id
    materializer = _Materializer(embedded_episode_source=blocked_source)

    catalog = build_prospective_execution_catalog(
        selection=selection,
        rubrics=rubrics,
        legacy_catalog=legacy,
        legacy_binding_materializer=materializer,
        source_import_blake3s=imports,
        catalog_revision=2,
    )
    document = catalog.to_document()
    blocked = next(
        row for row in document["rows"] if row["source_candidate_id"] == blocked_source
    )

    assert blocked["construction_readiness"] == "promoted"
    assert blocked["execution_status"] == "source_only"
    assert blocked["execution_proof"] is not None
    assert blocked["execution_blocker"] == {
        "category": "full_binding_validation_failed",
        "validator": "LegacyExecutionBindingResolver.resolve",
        "raw_exception_recorded": False,
        "execution_blocker_blake3": blake3_hex(
            {
                "category": "full_binding_validation_failed",
                "validator": "LegacyExecutionBindingResolver.resolve",
                "raw_exception_recorded": False,
            }
        ),
    }
    assert document["executable_count"] == 1_343
    assert document["source_only_count"] == 7_646
    assert "public policy embeds" not in str(blocked)

    rebuilt = verify_prospective_execution_catalog(
        document,
        selection=selection,
        rubrics=rubrics,
        legacy_catalog=legacy,
        legacy_binding_materializer=materializer,
        source_import_blake3s=imports,
    )
    assert rebuilt.catalog_blake3 == catalog.catalog_blake3

    tampered = catalog.to_document()
    tampered["rows"][0]["execution_blocker"]["category"] = "other"
    with pytest.raises(ProspectiveExecutionCatalogError, match="derivation differs"):
        verify_prospective_execution_catalog(
            tampered,
            selection=selection,
            rubrics=rubrics,
            legacy_catalog=legacy,
            legacy_binding_materializer=materializer,
            source_import_blake3s=imports,
        )
