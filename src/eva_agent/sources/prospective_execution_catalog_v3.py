"""All-train execution catalog over signed legacy and premium authorities.

Version 3 is deliberately additive.  It does not reconstruct or reinterpret
the legacy v2-r2 classification: that signed catalog remains the authority for
the exact 1,336 legacy-executable rows.  A row that was frozen there becomes
premium-executable only when an independently verified transition-v2 resolver
inventory names the same selection-v2 candidate and source binding.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence
from uuid import UUID

from eva_agent.admission.receipts import (
    AdmissionReceiptError,
    SignedEnvelope,
    verify_signed_envelope,
)
from eva_agent.campaign.selection_v2 import CampaignSelectionV2
from eva_agent.pipeline.digests import blake3_hex, canonical_value, is_blake3
from eva_agent.rubrics.registry import CompiledRubricRegistry

from .prospective_execution_catalog import (
    CATALOG_METHOD as LEGACY_CATALOG_METHOD,
    CATALOG_SCHEMA as LEGACY_CATALOG_SCHEMA,
    _rubric_profiles,
    _tool_profiles,
)


CATALOG_SCHEMA = "eva.prospective-execution-binding-catalog.v3"
CATALOG_METHOD = (
    "signed-v2-r2-legacy-plus-signed-premium-transition-v2-all-train-additive-v3"
)
PREMIUM_TRANSITION_SCHEMA = "eva.premium-construction-supervisor-transition.v2"
PREMIUM_RESOLVER_CATALOG_SCHEMA = (
    "eva.premium-candidate-execution-binding-catalog.v1"
)
EXPECTED_SCHEDULED = 9_000
EXPECTED_PRIMARY = 6_000
EXPECTED_RESERVE = 3_000
EXPECTED_LEGACY_EXECUTABLE = 1_336
EXPECTED_REJECTED = 11

_SELECTION_INVARIANT_FIELDS = (
    "candidate_id",
    "source_candidate_id",
    "source_family",
    "source_artifact_sha256",
    "domain",
    "stage",
    "rubric_id",
    "rubric_blake3",
    "construction_readiness",
    "readiness_proof_root_blake3",
    "selection_tier",
    "queue_ordinal",
)
_LEGACY_ROW_KEYS = {
    *_SELECTION_INVARIANT_FIELDS,
    "split",
    "source_import_blake3",
    "source_registry_blake3",
    "selection_entry_blake3",
    "rubric_item_count",
    "tool_profile_blake3",
    "execution_status",
    "execution_proof",
    "execution_blocker",
    "row_blake3",
}
_PREMIUM_REFERENCE_KEYS = {
    "candidate_id",
    "source_candidate_id",
    "source_family",
    "domain",
    "stage",
    "split",
    "rubric_id",
    "rubric_blake3",
    "publication_id",
    "binding_blake3",
    "transition_binding_blake3",
}
_LEGACY_PROOF_KEYS = {
    "kind",
    "resolver_catalog_blake3",
    "proof_root_blake3",
    "promoted_subject_blake3",
    "source_policy_upstream_sha256",
    "source_policy_blake3",
    "construction_manifest_upstream_sha256",
    "construction_manifest_blake3",
    "construction_attestation_blake3",
    "execution_proof_blake3",
}


class ProspectiveExecutionCatalogV3Error(ValueError):
    """A v3 catalog input would weaken the signed all-train boundary."""


class PremiumResolverInventoryPort(Protocol):
    transition: Any
    catalog_inventory_blake3: str
    executable_candidate_count: int

    def inventory(self) -> tuple[Any, ...]: ...


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    return value


def _uuid(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ProspectiveExecutionCatalogV3Error(f"{label} must be a canonical UUID")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError):
        raise ProspectiveExecutionCatalogV3Error(
            f"{label} must be a canonical UUID"
        ) from None
    if str(parsed) != value:
        raise ProspectiveExecutionCatalogV3Error(f"{label} must be a canonical UUID")
    return value


def _verify_envelope(envelope: Any, *, trust_store_path: Path, label: str) -> SignedEnvelope:
    if not isinstance(envelope, SignedEnvelope):
        raise ProspectiveExecutionCatalogV3Error(f"{label} envelope type differs")
    try:
        verify_signed_envelope(envelope, trust_store_path=trust_store_path)
    except AdmissionReceiptError as exc:
        raise ProspectiveExecutionCatalogV3Error(
            f"{label} signature verification failed"
        ) from exc
    return envelope


def _internal_catalog_digest(payload: Mapping[str, Any], *, label: str) -> str:
    digest = payload.get("catalog_blake3")
    if not is_blake3(digest):
        raise ProspectiveExecutionCatalogV3Error(f"{label} catalog BLAKE3 differs")
    core = {key: _plain(value) for key, value in payload.items() if key != "catalog_blake3"}
    if blake3_hex(core) != digest:
        raise ProspectiveExecutionCatalogV3Error(f"{label} catalog content differs")
    return str(digest)


def _selection_rows(
    selection: CampaignSelectionV2,
    rubrics: CompiledRubricRegistry,
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    if not isinstance(selection, CampaignSelectionV2):
        raise ProspectiveExecutionCatalogV3Error("selection-v2 type differs")
    # Reopen the complete canonical object before trusting convenient fields.
    try:
        CampaignSelectionV2.from_document(selection.to_document())
    except ValueError as exc:
        raise ProspectiveExecutionCatalogV3Error("selection-v2 content differs") from exc
    if len(selection.entries) != EXPECTED_SCHEDULED:
        raise ProspectiveExecutionCatalogV3Error("selection must contain exactly 9,000 rows")
    if selection.rubric_registry_blake3 != rubrics.digest:
        raise ProspectiveExecutionCatalogV3Error("selection rubric registry commitment differs")
    if len(selection.primary_entries) != EXPECTED_PRIMARY or len(selection.reserve_entries) != EXPECTED_RESERVE:
        raise ProspectiveExecutionCatalogV3Error("selection tier counts differ")
    split_counts = Counter(row.split for row in selection.primary_entries)
    if split_counts != Counter(train=EXPECTED_PRIMARY) or any(
        row.split != "train" for row in selection.entries
    ):
        raise ProspectiveExecutionCatalogV3Error(
            "selection must use the exact 6,000/0/0 all-train allocation"
        )
    rows = tuple(sorted(selection.entries, key=lambda row: row.queue_ordinal))
    if [row.queue_ordinal for row in rows] != list(range(1, EXPECTED_SCHEDULED + 1)):
        raise ProspectiveExecutionCatalogV3Error("selection queue ordinals differ")
    by_id = {row.candidate_id: row for row in rows}
    if len(by_id) != EXPECTED_SCHEDULED or len(
        {row.source_candidate_id for row in rows}
    ) != EXPECTED_SCHEDULED:
        raise ProspectiveExecutionCatalogV3Error("selection identity inventory differs")
    return rows, MappingProxyType(by_id)


def _verify_legacy_catalog(
    envelope: SignedEnvelope,
    *,
    selection: CampaignSelectionV2,
    selection_by_id: Mapping[str, Any],
    rubrics: CompiledRubricRegistry,
    trust_store_path: Path,
) -> tuple[Mapping[str, Mapping[str, Any]], str]:
    signed = _verify_envelope(
        envelope, trust_store_path=trust_store_path, label="legacy v2-r2 catalog"
    )
    payload = _plain(signed.payload)
    if not isinstance(payload, dict) or payload.get("schema") != LEGACY_CATALOG_SCHEMA:
        raise ProspectiveExecutionCatalogV3Error("legacy v2-r2 catalog schema differs")
    catalog_blake3 = _internal_catalog_digest(payload, label="legacy v2-r2")
    if not (
        type(payload.get("catalog_revision")) is int
        and payload["catalog_revision"] >= 1
        and payload.get("catalog_method") == LEGACY_CATALOG_METHOD
        and payload.get("base_selection_blake3") == selection.selection_blake3
        and payload.get("source_registry_blake3") == selection.source_registry_blake3
        and payload.get("rubric_registry_blake3") == rubrics.digest
        and is_blake3(payload.get("ancestor_legacy_catalog_blake3"))
        and is_blake3(payload.get("legacy_runtime_source_blake3"))
        and payload.get("scheduled_count") == EXPECTED_SCHEDULED
        and payload.get("primary_count") == EXPECTED_PRIMARY
        and payload.get("reserve_count") == EXPECTED_RESERVE
        and payload.get("executable_count") == EXPECTED_LEGACY_EXECUTABLE
        and payload.get("source_only_count")
        == EXPECTED_SCHEDULED - EXPECTED_LEGACY_EXECUTABLE - EXPECTED_REJECTED
        and payload.get("rejected_count") == EXPECTED_REJECTED
        and payload.get("controls", {}).get("provider_calls") == 0
        and payload.get("controls", {}).get("ledger_reads") == 0
        and payload.get("controls", {}).get("ledger_writes") == 0
        and payload.get("controls", {}).get("source_only_is_executable") is False
        and payload.get("controls", {}).get("rejected_is_executable") is False
        and payload.get("controls", {}).get("full_binding_validation_required") is True
    ):
        raise ProspectiveExecutionCatalogV3Error("legacy v2-r2 authority differs")
    if _plain(payload.get("tool_profiles")) != _plain(_tool_profiles()) or _plain(
        payload.get("rubric_profiles")
    ) != _plain(_rubric_profiles(rubrics)):
        raise ProspectiveExecutionCatalogV3Error("legacy tool or rubric profiles differ")

    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != EXPECTED_SCHEDULED:
        raise ProspectiveExecutionCatalogV3Error("legacy v2-r2 row inventory differs")
    by_id: dict[str, Mapping[str, Any]] = {}
    by_source: set[str] = set()
    status_counts: Counter[str] = Counter()
    for raw in raw_rows:
        if not isinstance(raw, dict) or set(raw) != _LEGACY_ROW_KEYS:
            raise ProspectiveExecutionCatalogV3Error("legacy v2-r2 row shape differs")
        candidate_id = _uuid(raw.get("candidate_id"), label="legacy row candidate_id")
        selected = selection_by_id.get(candidate_id)
        if selected is None or candidate_id in by_id:
            raise ProspectiveExecutionCatalogV3Error("legacy candidate inventory differs")
        if not isinstance(raw["source_candidate_id"], str):
            raise ProspectiveExecutionCatalogV3Error("legacy source inventory differs")
        if raw["source_candidate_id"] in by_source:
            raise ProspectiveExecutionCatalogV3Error("legacy source inventory differs")
        by_source.add(raw["source_candidate_id"])
        for field in _SELECTION_INVARIANT_FIELDS:
            if raw[field] != getattr(selected, field):
                raise ProspectiveExecutionCatalogV3Error(
                    f"legacy/selection invariant differs: {field}"
                )
        core = {key: _plain(value) for key, value in raw.items() if key != "row_blake3"}
        if not is_blake3(raw["row_blake3"]) or blake3_hex(core) != raw["row_blake3"]:
            raise ProspectiveExecutionCatalogV3Error("legacy row BLAKE3 differs")
        if not is_blake3(raw["source_import_blake3"]):
            raise ProspectiveExecutionCatalogV3Error("legacy source import BLAKE3 differs")
        if raw["source_registry_blake3"] != selection.source_registry_blake3:
            raise ProspectiveExecutionCatalogV3Error("legacy row source registry differs")
        rubric = rubrics.resolve(selected.domain, selected.stage)
        if raw["rubric_item_count"] != len(rubric.items):
            raise ProspectiveExecutionCatalogV3Error("legacy rubric item count differs")
        expected_profile = next(
            row for row in _tool_profiles() if row["stage"] == selected.stage
        )
        if raw["tool_profile_blake3"] != expected_profile["profile_blake3"]:
            raise ProspectiveExecutionCatalogV3Error("legacy tool profile differs")
        status = raw["execution_status"]
        proof = raw["execution_proof"]
        blocker = raw["execution_blocker"]
        if isinstance(proof, dict):
            proof_core = {
                key: _plain(value)
                for key, value in proof.items()
                if key != "execution_proof_blake3"
            }
            if not (
                set(proof) == _LEGACY_PROOF_KEYS
                and proof.get("kind") == "signed_v24_legacy_promoted"
                and proof.get("resolver_catalog_blake3")
                == payload.get("ancestor_legacy_catalog_blake3")
                and proof.get("proof_root_blake3")
                == selected.readiness_proof_root_blake3
                and all(
                    is_blake3(proof.get(field))
                    for field in _LEGACY_PROOF_KEYS - {"kind"}
                )
                and blake3_hex(proof_core) == proof["execution_proof_blake3"]
            ):
                raise ProspectiveExecutionCatalogV3Error(
                    "legacy signed execution proof differs"
                )
        if status == "executable_legacy":
            if selected.construction_readiness != "promoted" or not isinstance(proof, dict) or blocker is not None:
                raise ProspectiveExecutionCatalogV3Error("legacy executable proof state differs")
        elif status == "source_only":
            if selected.construction_readiness == "frozen":
                if proof is not None or blocker is not None:
                    raise ProspectiveExecutionCatalogV3Error("frozen legacy row gained a proof")
            elif selected.construction_readiness == "promoted":
                if not isinstance(proof, dict) or not isinstance(blocker, dict):
                    raise ProspectiveExecutionCatalogV3Error("blocked promoted legacy row differs")
                blocker_core = {
                    "category": "full_binding_validation_failed",
                    "validator": "LegacyExecutionBindingResolver.resolve",
                    "raw_exception_recorded": False,
                }
                if blocker != {
                    **blocker_core,
                    "execution_blocker_blake3": blake3_hex(blocker_core),
                }:
                    raise ProspectiveExecutionCatalogV3Error(
                        "blocked promoted legacy proof differs"
                    )
            else:
                raise ProspectiveExecutionCatalogV3Error("legacy source-only readiness differs")
        elif status == "rejected":
            if selected.construction_readiness != "rejected" or proof is not None or blocker is not None:
                raise ProspectiveExecutionCatalogV3Error("legacy rejected row differs")
        else:
            raise ProspectiveExecutionCatalogV3Error("legacy execution status differs")
        status_counts[status] += 1
        by_id[candidate_id] = MappingProxyType(raw)
    if set(by_id) != set(selection_by_id) or status_counts != Counter(
        executable_legacy=EXPECTED_LEGACY_EXECUTABLE,
        source_only=EXPECTED_SCHEDULED - EXPECTED_LEGACY_EXECUTABLE - EXPECTED_REJECTED,
        rejected=EXPECTED_REJECTED,
    ):
        raise ProspectiveExecutionCatalogV3Error("legacy v2-r2 status inventory differs")
    return MappingProxyType(by_id), catalog_blake3


def _premium_reference_from_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    catalog_binding = binding.get("catalog_binding")
    if not isinstance(catalog_binding, Mapping):
        raise ProspectiveExecutionCatalogV3Error("premium catalog binding differs")
    reference = {
        "candidate_id": binding.get("candidate_id"),
        "source_candidate_id": binding.get("source_candidate_id"),
        "source_family": binding.get("source_family"),
        "domain": binding.get("domain"),
        "stage": binding.get("stage"),
        "split": binding.get("split"),
        "rubric_id": binding.get("rubric_id"),
        "rubric_blake3": binding.get("rubric_blake3"),
        "publication_id": catalog_binding.get("publication_id"),
        "binding_blake3": catalog_binding.get("binding_blake3"),
        "transition_binding_blake3": blake3_hex(binding),
    }
    if set(reference) != _PREMIUM_REFERENCE_KEYS:
        raise AssertionError("premium reference projection changed")
    return reference


@dataclass(frozen=True, slots=True)
class _PremiumAuthorization:
    candidate_id: str
    source_candidate_id: str
    resolver_catalog_blake3: str
    signed_transition_blake3: str
    transition_payload_blake3: str
    transition_id: str
    reference: Mapping[str, Any]

    def proof(self) -> Mapping[str, Any]:
        core = {
            "kind": "signed_premium_construction_transition_v2",
            "resolver_catalog_blake3": self.resolver_catalog_blake3,
            "signed_transition_blake3": self.signed_transition_blake3,
            "transition_payload_blake3": self.transition_payload_blake3,
            "transition_id": self.transition_id,
            "transition_binding_blake3": self.reference[
                "transition_binding_blake3"
            ],
            "publication_id": self.reference["publication_id"],
            "binding_blake3": self.reference["binding_blake3"],
        }
        return MappingProxyType(
            {**core, "execution_proof_blake3": blake3_hex(core)}
        )


def _verify_premium_resolver(
    resolver: PremiumResolverInventoryPort,
    *,
    selection: CampaignSelectionV2,
    selection_by_id: Mapping[str, Any],
    legacy_rows: Mapping[str, Mapping[str, Any]],
    trust_store_path: Path,
) -> tuple[_PremiumAuthorization, ...]:
    transition = getattr(resolver, "transition", None)
    envelope = _verify_envelope(
        getattr(transition, "envelope", None),
        trust_store_path=trust_store_path,
        label="premium transition-v2",
    )
    if getattr(transition, "transition_blake3", None) != envelope.envelope_blake3:
        raise ProspectiveExecutionCatalogV3Error("premium transition identity differs")
    payload = _plain(envelope.payload)
    if not isinstance(payload, dict) or payload.get("schema") != PREMIUM_TRANSITION_SCHEMA:
        raise ProspectiveExecutionCatalogV3Error("premium transition-v2 schema differs")
    bindings = payload.get("bindings")
    selected_ids = payload.get("selected_candidate_ids")
    controls = payload.get("controls")
    if not (
        payload.get("selection_blake3") == selection.selection_blake3
        and payload.get("rubric_registry_blake3") == selection.rubric_registry_blake3
        and payload.get("from_state") == "frozen"
        and payload.get("to_state") == "executable_ready"
        and isinstance(bindings, list)
        and bool(bindings)
        and all(isinstance(row, dict) for row in bindings)
        and isinstance(selected_ids, list)
        and all(isinstance(candidate_id, str) for candidate_id in selected_ids)
        and selected_ids == [row.get("candidate_id") for row in bindings]
        and len(set(selected_ids)) == len(selected_ids)
        and payload.get("binding_count") == len(bindings)
        and getattr(transition, "binding_count", None) == len(bindings)
        and isinstance(controls, dict)
        and controls.get("provider_calls") == 0
        and controls.get("ledger_writes") == 0
        and controls.get("append_only") is True
        and controls.get("admission_eligibility_claimed") is False
        and controls.get("proof_catalog_coverage") == "complete_cumulative"
        and controls.get("transition_selection") == "explicit_successful_candidate_set"
    ):
        raise ProspectiveExecutionCatalogV3Error("premium transition-v2 claims differ")
    if not is_blake3(payload.get("catalog_blake3")):
        raise ProspectiveExecutionCatalogV3Error("premium proof catalog commitment differs")
    ids_core = {
        "schema": "eva.premium-construction-selected-candidate-ids.v2",
        "candidate_ids": selected_ids,
    }
    if payload.get("selected_candidate_ids_blake3") != blake3_hex(ids_core):
        raise ProspectiveExecutionCatalogV3Error("premium selected-ID commitment differs")
    claim_core = {
        "schema": "eva.premium-construction-supervisor-claim-domain.v2",
        "catalog_blake3": payload.get("catalog_blake3"),
        "selected_candidate_ids_blake3": payload.get(
            "selected_candidate_ids_blake3"
        ),
    }
    if payload.get("claim_domain_blake3") != blake3_hex(claim_core):
        raise ProspectiveExecutionCatalogV3Error("premium claim domain differs")

    inventory_method = getattr(resolver, "inventory", None)
    inventory = inventory_method() if callable(inventory_method) else None
    count = getattr(resolver, "executable_candidate_count", None)
    if not isinstance(inventory, tuple) or type(count) is not int or count != len(bindings) or len(inventory) != count:
        raise ProspectiveExecutionCatalogV3Error("premium resolver inventory count differs")
    expected_references = tuple(_premium_reference_from_binding(row) for row in bindings)
    observed_by_source: dict[str, Mapping[str, Any]] = {}
    for reference in inventory:
        to_document = getattr(reference, "to_document", None)
        try:
            document = canonical_value(to_document()) if callable(to_document) else None
        except (TypeError, ValueError) as exc:
            raise ProspectiveExecutionCatalogV3Error(
                "premium resolver reference differs"
            ) from exc
        if not isinstance(document, dict) or set(document) != _PREMIUM_REFERENCE_KEYS:
            raise ProspectiveExecutionCatalogV3Error("premium resolver reference differs")
        source_id = document.get("source_candidate_id")
        if not isinstance(source_id, str) or source_id in observed_by_source:
            raise ProspectiveExecutionCatalogV3Error("premium resolver source inventory differs")
        observed_by_source[source_id] = document
    if observed_by_source != {
        row["source_candidate_id"]: row for row in expected_references
    }:
        raise ProspectiveExecutionCatalogV3Error("premium resolver/signed bindings differ")
    resolver_catalog_core = {
        "schema": PREMIUM_RESOLVER_CATALOG_SCHEMA,
        "signed_transition_blake3": envelope.envelope_blake3,
        "selection_blake3": selection.selection_blake3,
        "coverage": "signed_premium_executable_ready_only",
        "candidate_count": len(expected_references),
        "rows": list(expected_references),
    }
    resolver_digest = getattr(resolver, "catalog_inventory_blake3", None)
    if not is_blake3(resolver_digest) or resolver_digest != blake3_hex(
        resolver_catalog_core
    ):
        raise ProspectiveExecutionCatalogV3Error("premium resolver catalog differs")

    authorizations: list[_PremiumAuthorization] = []
    for reference in expected_references:
        candidate_id = _uuid(
            reference["candidate_id"], label="premium candidate_id"
        )
        selected = selection_by_id.get(candidate_id)
        legacy = legacy_rows.get(candidate_id)
        if selected is None or legacy is None:
            raise ProspectiveExecutionCatalogV3Error("premium candidate is outside selection")
        if selected.construction_readiness != "frozen" or not (
            legacy["execution_status"] == "source_only"
            and legacy["execution_proof"] is None
            and legacy["execution_blocker"] is None
        ):
            raise ProspectiveExecutionCatalogV3Error(
                "premium transition may upgrade only its exact frozen source-only row"
            )
        for field in (
            "candidate_id",
            "source_candidate_id",
            "source_family",
            "domain",
            "stage",
            "split",
            "rubric_id",
            "rubric_blake3",
        ):
            if reference[field] != getattr(selected, field):
                raise ProspectiveExecutionCatalogV3Error(
                    f"premium/selection binding differs: {field}"
                )
        for field in ("binding_blake3", "transition_binding_blake3"):
            if not is_blake3(reference[field]):
                raise ProspectiveExecutionCatalogV3Error(
                    "premium binding commitment differs"
                )
        _uuid(reference["publication_id"], label="premium publication_id")
        authorizations.append(
            _PremiumAuthorization(
                candidate_id=candidate_id,
                source_candidate_id=reference["source_candidate_id"],
                resolver_catalog_blake3=resolver_digest,
                signed_transition_blake3=envelope.envelope_blake3,
                transition_payload_blake3=envelope.payload_blake3,
                transition_id=_uuid(
                    payload.get("transition_id"), label="premium transition_id"
                ),
                reference=MappingProxyType(reference),
            )
        )
    return tuple(authorizations)


@dataclass(frozen=True, slots=True)
class ProspectiveExecutionBindingCatalogV3:
    document: Mapping[str, Any]

    @property
    def catalog_blake3(self) -> str:
        return str(self.document["catalog_blake3"])

    @property
    def executable_candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            row["candidate_id"]
            for row in self.document["rows"]
            if row["execution_status"]
            in {"executable_legacy", "executable_premium"}
        )

    def to_document(self) -> dict[str, Any]:
        return _plain(self.document)


def build_prospective_execution_catalog_v3(
    *,
    selection: CampaignSelectionV2,
    rubrics: CompiledRubricRegistry,
    signed_legacy_catalog: SignedEnvelope,
    trust_store_path: str | Path,
    premium_resolvers: Sequence[PremiumResolverInventoryPort] = (),
    catalog_revision: int = 1,
) -> ProspectiveExecutionBindingCatalogV3:
    """Build one signed-ready, provider-free all-train execution inventory."""

    if type(catalog_revision) is not int or catalog_revision < 1:
        raise ProspectiveExecutionCatalogV3Error("catalog revision must be positive")
    if isinstance(premium_resolvers, (str, bytes)) or not isinstance(
        premium_resolvers, Sequence
    ):
        raise ProspectiveExecutionCatalogV3Error(
            "premium resolvers must be an ordered sequence"
        )
    trust_path = Path(trust_store_path)
    rows, selection_by_id = _selection_rows(selection, rubrics)
    legacy_rows, legacy_catalog_blake3 = _verify_legacy_catalog(
        signed_legacy_catalog,
        selection=selection,
        selection_by_id=selection_by_id,
        rubrics=rubrics,
        trust_store_path=trust_path,
    )
    premium_authorizations: dict[str, _PremiumAuthorization] = {}
    resolver_digests: set[str] = set()
    transition_digests: set[str] = set()
    for resolver in premium_resolvers:
        authorizations = _verify_premium_resolver(
            resolver,
            selection=selection,
            selection_by_id=selection_by_id,
            legacy_rows=legacy_rows,
            trust_store_path=trust_path,
        )
        resolver_digest = getattr(resolver, "catalog_inventory_blake3")
        transition_digest = authorizations[0].signed_transition_blake3
        if resolver_digest in resolver_digests or transition_digest in transition_digests:
            raise ProspectiveExecutionCatalogV3Error(
                "premium transition or resolver catalog is duplicated"
            )
        resolver_digests.add(resolver_digest)
        transition_digests.add(transition_digest)
        for authorization in authorizations:
            if authorization.candidate_id in premium_authorizations:
                raise ProspectiveExecutionCatalogV3Error(
                    "premium candidate is authorized more than once"
                )
            premium_authorizations[authorization.candidate_id] = authorization
    if len({row.source_candidate_id for row in premium_authorizations.values()}) != len(
        premium_authorizations
    ):
        raise ProspectiveExecutionCatalogV3Error(
            "premium source is authorized more than once"
        )

    tool_profiles = _tool_profiles()
    profile_by_stage = {row["stage"]: row for row in tool_profiles}
    rubric_profiles = _rubric_profiles(rubrics)
    output_rows: list[Mapping[str, Any]] = []
    for entry in rows:
        legacy = legacy_rows[entry.candidate_id]
        premium = premium_authorizations.get(entry.candidate_id)
        status = legacy["execution_status"]
        proof = legacy["execution_proof"]
        blocker = legacy["execution_blocker"]
        if premium is not None:
            status, proof, blocker = "executable_premium", premium.proof(), None
        rubric = rubrics.resolve(entry.domain, entry.stage)
        row_core = {
            "candidate_id": entry.candidate_id,
            "source_candidate_id": entry.source_candidate_id,
            "queue_ordinal": entry.queue_ordinal,
            "selection_tier": entry.selection_tier,
            "split": entry.split,
            "source_family": entry.source_family,
            "domain": entry.domain,
            "stage": entry.stage,
            "source_artifact_sha256": entry.source_artifact_sha256,
            "source_import_blake3": legacy["source_import_blake3"],
            "source_registry_blake3": selection.source_registry_blake3,
            "selection_entry_blake3": blake3_hex(entry.to_document()),
            "rubric_id": entry.rubric_id,
            "rubric_blake3": entry.rubric_blake3,
            "rubric_item_count": len(rubric.items),
            "tool_profile_blake3": profile_by_stage[entry.stage]["profile_blake3"],
            # The immutable selection readiness is preserved.  Premium proof
            # changes execution status, never historical selection state.
            "construction_readiness": entry.construction_readiness,
            "readiness_proof_root_blake3": entry.readiness_proof_root_blake3,
            "execution_status": status,
            "execution_proof": _plain(proof),
            "execution_blocker": _plain(blocker),
        }
        output_rows.append({**row_core, "row_blake3": blake3_hex(row_core)})

    status_counts = Counter(row["execution_status"] for row in output_rows)
    premium_count = len(premium_authorizations)
    expected_counts = Counter(
        executable_legacy=EXPECTED_LEGACY_EXECUTABLE,
        executable_premium=premium_count,
        source_only=(
            EXPECTED_SCHEDULED
            - EXPECTED_LEGACY_EXECUTABLE
            - premium_count
            - EXPECTED_REJECTED
        ),
        rejected=EXPECTED_REJECTED,
    )
    if status_counts != expected_counts:
        raise ProspectiveExecutionCatalogV3Error("v3 execution status counts differ")
    transition_rows = sorted(
        (
            {
                "transition_id": authorization.transition_id,
                "signed_transition_blake3": authorization.signed_transition_blake3,
                "transition_payload_blake3": authorization.transition_payload_blake3,
                "resolver_catalog_blake3": authorization.resolver_catalog_blake3,
            }
            for authorization in premium_authorizations.values()
        ),
        key=lambda row: (row["resolver_catalog_blake3"], row["transition_id"]),
    )
    # One transition is represented once even when it authorizes many rows.
    unique_transitions = []
    for row in transition_rows:
        if not unique_transitions or row != unique_transitions[-1]:
            unique_transitions.append(row)
    core = {
        "schema": CATALOG_SCHEMA,
        "catalog_revision": catalog_revision,
        "catalog_method": CATALOG_METHOD,
        "selection_blake3": selection.selection_blake3,
        "base_selection_blake3": selection.base_selection_blake3,
        "plan_blake3": selection.plan_blake3,
        "source_registry_blake3": selection.source_registry_blake3,
        "rubric_registry_blake3": rubrics.digest,
        "signed_legacy_catalog_blake3": legacy_catalog_blake3,
        "signed_legacy_envelope_blake3": signed_legacy_catalog.envelope_blake3,
        "legacy_executable_count": EXPECTED_LEGACY_EXECUTABLE,
        "premium_executable_count": premium_count,
        "premium_transition_count": len(unique_transitions),
        "premium_transitions": unique_transitions,
        "scheduled_count": EXPECTED_SCHEDULED,
        "primary_count": EXPECTED_PRIMARY,
        "reserve_count": EXPECTED_RESERVE,
        "executable_count": EXPECTED_LEGACY_EXECUTABLE + premium_count,
        "source_only_count": expected_counts["source_only"],
        "rejected_count": EXPECTED_REJECTED,
        "primary_split_counts": {"train": EXPECTED_PRIMARY},
        "tool_profiles": list(tool_profiles),
        "rubric_profiles": list(rubric_profiles),
        "rows": output_rows,
        "controls": {
            "provider_calls": 0,
            "ledger_reads": 0,
            "ledger_writes": 0,
            "launch_reads": 0,
            "launch_writes": 0,
            "all_training": True,
            "legacy_status_and_proof_reused_without_reclassification": True,
            "legacy_executable_count_fixed": EXPECTED_LEGACY_EXECUTABLE,
            "premium_requires_signed_transition_v2_and_exact_resolver_inventory": True,
            "unlisted_frozen_is_executable": False,
            "rejected_is_executable": False,
            "future_upgrades_require_new_signed_revision": True,
        },
    }
    document = {**core, "catalog_blake3": blake3_hex(core)}
    return ProspectiveExecutionBindingCatalogV3(MappingProxyType(document))


def verify_prospective_execution_catalog_v3(
    value: Mapping[str, Any],
    *,
    selection: CampaignSelectionV2,
    rubrics: CompiledRubricRegistry,
    signed_legacy_catalog: SignedEnvelope,
    trust_store_path: str | Path,
    premium_resolvers: Sequence[PremiumResolverInventoryPort] = (),
) -> ProspectiveExecutionBindingCatalogV3:
    """Independently rebuild v3; exact canonical content is required."""

    if not isinstance(value, Mapping) or value.get("schema") != CATALOG_SCHEMA:
        raise ProspectiveExecutionCatalogV3Error("prospective catalog-v3 schema differs")
    expected = build_prospective_execution_catalog_v3(
        selection=selection,
        rubrics=rubrics,
        signed_legacy_catalog=signed_legacy_catalog,
        premium_resolvers=premium_resolvers,
        trust_store_path=trust_store_path,
        catalog_revision=value.get("catalog_revision"),
    )
    if canonical_value(value) != canonical_value(expected.to_document()):
        raise ProspectiveExecutionCatalogV3Error(
            "prospective catalog-v3 derivation differs"
        )
    return expected


__all__ = [
    "CATALOG_SCHEMA",
    "ProspectiveExecutionBindingCatalogV3",
    "ProspectiveExecutionCatalogV3Error",
    "build_prospective_execution_catalog_v3",
    "verify_prospective_execution_catalog_v3",
]
