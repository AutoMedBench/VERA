"""Per-policy MCP catalog bindings and deterministic schema-variant inventory."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from eva_agent.pipeline.digests import blake3_hex, is_blake3

from .catalog import (
    MCPCompatibilityError,
    MCPToolCatalog,
    ToolAnnotations,
    _freeze,
    _json_copy,
    _mutable,
    build_mcp_catalog,
    verify_mcp_catalog,
)


POLICY_BINDING_SCHEMA = "eva.mcp-policy-catalog-binding.v1"
POLICY_INVENTORY_SCHEMA = "eva.mcp-policy-catalog-inventory.v1"


def _identity(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MCPCompatibilityError(f"{label} differs")
    return value


def _binding_core(
    *,
    policy_id: str,
    candidate_id: str,
    source_policy_blake3: str,
    catalog: MCPToolCatalog,
) -> dict[str, Any]:
    return {
        "schema": POLICY_BINDING_SCHEMA,
        "policy_id": policy_id,
        "candidate_id": candidate_id,
        "source_policy_blake3": source_policy_blake3,
        "schema_catalog_blake3": catalog.schema_catalog_blake3,
        "catalog_blake3": catalog.catalog_blake3,
        "tool_schema_commitments": _mutable(catalog.schema_commitments),
    }


@dataclass(frozen=True, slots=True)
class PolicyCatalogBinding:
    """Pin one exact MCP catalog to one source policy/candidate instance."""

    policy_id: str
    candidate_id: str
    source_policy_blake3: str
    catalog: MCPToolCatalog
    binding_blake3: str
    schema: str = POLICY_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != POLICY_BINDING_SCHEMA:
            raise MCPCompatibilityError("policy catalog binding schema differs")
        _identity(self.policy_id, label="policy_id")
        _identity(self.candidate_id, label="candidate_id")
        if not is_blake3(self.source_policy_blake3):
            raise MCPCompatibilityError("source policy BLAKE3 differs")
        if not isinstance(self.catalog, MCPToolCatalog):
            raise MCPCompatibilityError("policy catalog binding lacks a verified catalog")
        verify_mcp_catalog(self.catalog)
        expected = blake3_hex(
            _binding_core(
                policy_id=self.policy_id,
                candidate_id=self.candidate_id,
                source_policy_blake3=self.source_policy_blake3,
                catalog=self.catalog,
            )
        )
        if self.binding_blake3 != expected:
            raise MCPCompatibilityError("policy catalog binding BLAKE3 differs")

    def to_document(self, *, include_catalog: bool = True) -> dict[str, Any]:
        document = {
            **_binding_core(
                policy_id=self.policy_id,
                candidate_id=self.candidate_id,
                source_policy_blake3=self.source_policy_blake3,
                catalog=self.catalog,
            ),
            "binding_blake3": self.binding_blake3,
        }
        if include_catalog:
            document["catalog"] = self.catalog.to_document()
        return document

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, Any],
        *,
        expected_source_policy_blake3: str | None = None,
        canonical_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> "PolicyCatalogBinding":
        return verify_policy_catalog_binding(
            document,
            expected_source_policy_blake3=expected_source_policy_blake3,
            canonical_rows=canonical_rows,
        )

    def verify(
        self,
        *,
        expected_source_policy_blake3: str | None = None,
        canonical_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        # Re-run dataclass invariants even if an attacker used low-level object
        # mutation or supplied a catalog reconstructed from wire data.
        self.__post_init__()
        if (
            expected_source_policy_blake3 is not None
            and self.source_policy_blake3 != expected_source_policy_blake3
        ):
            raise MCPCompatibilityError("policy binding does not match required BLAKE3")
        verify_mcp_catalog(self.catalog, canonical_rows=canonical_rows)


def bind_policy_catalog(
    *,
    policy_id: str,
    candidate_id: str,
    source_policy_blake3: str,
    rows: Sequence[Mapping[str, Any]],
    annotations: Mapping[str, ToolAnnotations | Mapping[str, Any]] | None = None,
) -> PolicyCatalogBinding:
    """Build a catalog that cannot be reused for a different policy instance."""

    catalog = build_mcp_catalog(rows, annotations=annotations)
    core = _binding_core(
        policy_id=_identity(policy_id, label="policy_id"),
        candidate_id=_identity(candidate_id, label="candidate_id"),
        source_policy_blake3=source_policy_blake3,
        catalog=catalog,
    )
    if not is_blake3(source_policy_blake3):
        raise MCPCompatibilityError("source policy BLAKE3 differs")
    return PolicyCatalogBinding(
        policy_id=policy_id,
        candidate_id=candidate_id,
        source_policy_blake3=source_policy_blake3,
        catalog=catalog,
        binding_blake3=blake3_hex(core),
    )


_BINDING_DOCUMENT_KEYS = frozenset(
    {
        "schema",
        "policy_id",
        "candidate_id",
        "source_policy_blake3",
        "schema_catalog_blake3",
        "catalog_blake3",
        "tool_schema_commitments",
        "binding_blake3",
        "catalog",
    }
)


def verify_policy_catalog_binding(
    binding: PolicyCatalogBinding | Mapping[str, Any],
    *,
    expected_source_policy_blake3: str | None = None,
    canonical_rows: Sequence[Mapping[str, Any]] | None = None,
) -> PolicyCatalogBinding:
    """Reopen a binding and return only an exact, verified policy instance."""

    if isinstance(binding, PolicyCatalogBinding):
        verified = binding
    else:
        if not isinstance(binding, Mapping) or set(binding) != _BINDING_DOCUMENT_KEYS:
            raise MCPCompatibilityError("policy catalog binding document keys differ")
        if binding["schema"] != POLICY_BINDING_SCHEMA:
            raise MCPCompatibilityError("policy catalog binding schema differs")
        catalog = MCPToolCatalog.from_document(binding["catalog"])
        if (
            binding["schema_catalog_blake3"] != catalog.schema_catalog_blake3
            or binding["catalog_blake3"] != catalog.catalog_blake3
            or _json_copy(
                binding["tool_schema_commitments"], label="binding commitments"
            )
            != _json_copy(catalog.schema_commitments, label="catalog commitments")
        ):
            raise MCPCompatibilityError("policy binding catalog commitment differs")
        verified = PolicyCatalogBinding(
            schema=binding["schema"],
            policy_id=binding["policy_id"],
            candidate_id=binding["candidate_id"],
            source_policy_blake3=binding["source_policy_blake3"],
            catalog=catalog,
            binding_blake3=binding["binding_blake3"],
        )
    verified.verify(
        expected_source_policy_blake3=expected_source_policy_blake3,
        canonical_rows=canonical_rows,
    )
    return verified


def _inventory_core(
    entries: Sequence[Mapping[str, Any]], variants: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "schema": POLICY_INVENTORY_SCHEMA,
        "policy_count": len(entries),
        "entries": _mutable(entries),
        "schema_variants": _mutable(variants),
    }


@dataclass(frozen=True, slots=True)
class PolicyCatalogInventory:
    """Digest-only cross-policy inventory; never a merged serving catalog."""

    entries: tuple[Mapping[str, Any], ...]
    schema_variants: tuple[Mapping[str, Any], ...]
    inventory_blake3: str
    schema: str = POLICY_INVENTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != POLICY_INVENTORY_SCHEMA:
            raise MCPCompatibilityError("policy catalog inventory schema differs")
        _verify_inventory_parts(self.entries, self.schema_variants, self.inventory_blake3)
        object.__setattr__(
            self,
            "entries",
            tuple(_freeze(_json_copy(row, label="policy inventory entry")) for row in self.entries),
        )
        object.__setattr__(
            self,
            "schema_variants",
            tuple(_freeze(_json_copy(row, label="schema variant")) for row in self.schema_variants),
        )

    @property
    def policy_count(self) -> int:
        return len(self.entries)

    def to_document(self) -> dict[str, Any]:
        return {
            **_inventory_core(self.entries, self.schema_variants),
            "inventory_blake3": self.inventory_blake3,
        }

    def verify(self) -> None:
        _verify_inventory_parts(
            self.entries, self.schema_variants, self.inventory_blake3
        )


_ENTRY_KEYS = frozenset(
    {
        "policy_id",
        "candidate_id",
        "source_policy_blake3",
        "schema_catalog_blake3",
        "catalog_blake3",
        "tool_schema_commitments",
        "binding_blake3",
    }
)
_VARIANT_KEYS = frozenset({"name", "input_schema_blake3", "policy_count"})


def _verify_inventory_parts(
    entries: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    inventory_blake3: str,
) -> None:
    normalized_entries = _json_copy(entries, label="policy inventory entries")
    normalized_variants = _json_copy(variants, label="policy schema variants")
    if not isinstance(normalized_entries, list) or not isinstance(normalized_variants, list):
        raise MCPCompatibilityError("policy inventory arrays differ")
    keys: list[tuple[str, str]] = []
    observed: Counter[tuple[str, str]] = Counter()
    for entry in normalized_entries:
        if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
            raise MCPCompatibilityError("policy inventory entry keys differ")
        key = (
            _identity(entry["policy_id"], label="policy_id"),
            _identity(entry["candidate_id"], label="candidate_id"),
        )
        keys.append(key)
        for digest_key in (
            "source_policy_blake3",
            "schema_catalog_blake3",
            "catalog_blake3",
            "binding_blake3",
        ):
            if not is_blake3(entry[digest_key]):
                raise MCPCompatibilityError(f"inventory {digest_key} differs")
        commitments = entry["tool_schema_commitments"]
        if not isinstance(commitments, list):
            raise MCPCompatibilityError("inventory tool schema commitments differ")
        names: set[str] = set()
        for commitment in commitments:
            if not isinstance(commitment, dict) or set(commitment) != {
                "name",
                "input_schema_blake3",
            }:
                raise MCPCompatibilityError("inventory schema commitment keys differ")
            name = _identity(commitment["name"], label="tool name")
            if name in names:
                raise MCPCompatibilityError("duplicate tool name within one policy")
            names.add(name)
            digest = commitment["input_schema_blake3"]
            if not is_blake3(digest):
                raise MCPCompatibilityError("inventory input schema BLAKE3 differs")
            observed[(name, digest)] += 1
        binding_core = {
            "schema": POLICY_BINDING_SCHEMA,
            "policy_id": entry["policy_id"],
            "candidate_id": entry["candidate_id"],
            "source_policy_blake3": entry["source_policy_blake3"],
            "schema_catalog_blake3": entry["schema_catalog_blake3"],
            "catalog_blake3": entry["catalog_blake3"],
            "tool_schema_commitments": commitments,
        }
        if entry["binding_blake3"] != blake3_hex(binding_core):
            raise MCPCompatibilityError("inventory policy binding BLAKE3 differs")
    if len(keys) != len(set(keys)):
        raise MCPCompatibilityError("duplicate policy/candidate inventory binding")
    if keys != sorted(keys):
        raise MCPCompatibilityError("policy inventory order differs")

    expected_variants = [
        {
            "name": name,
            "input_schema_blake3": digest,
            "policy_count": count,
        }
        for (name, digest), count in sorted(observed.items())
    ]
    for variant in normalized_variants:
        if not isinstance(variant, dict) or set(variant) != _VARIANT_KEYS:
            raise MCPCompatibilityError("policy schema variant keys differ")
        if type(variant["policy_count"]) is not int or variant["policy_count"] < 1:
            raise MCPCompatibilityError("policy schema variant count differs")
    if normalized_variants != expected_variants:
        raise MCPCompatibilityError("policy schema variant inventory differs")
    expected = blake3_hex(_inventory_core(normalized_entries, normalized_variants))
    if inventory_blake3 != expected:
        raise MCPCompatibilityError("policy catalog inventory BLAKE3 differs")


def build_policy_catalog_inventory(
    bindings: Sequence[PolicyCatalogBinding],
) -> PolicyCatalogInventory:
    """Inventory variants deterministically without merging catalogs by name."""

    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for binding in bindings:
        if not isinstance(binding, PolicyCatalogBinding):
            raise MCPCompatibilityError("inventory requires policy catalog bindings")
        binding.verify()
        key = (binding.policy_id, binding.candidate_id)
        if key in seen:
            raise MCPCompatibilityError("duplicate policy/candidate inventory binding")
        seen.add(key)
        document = binding.to_document(include_catalog=False)
        document.pop("schema")
        entries.append(document)
    entries.sort(key=lambda row: (row["policy_id"], row["candidate_id"]))
    counts: Counter[tuple[str, str]] = Counter()
    for entry in entries:
        for commitment in entry["tool_schema_commitments"]:
            counts[(commitment["name"], commitment["input_schema_blake3"])] += 1
    variants = [
        {
            "name": name,
            "input_schema_blake3": digest,
            "policy_count": count,
        }
        for (name, digest), count in sorted(counts.items())
    ]
    core = _inventory_core(entries, variants)
    return PolicyCatalogInventory(
        entries=tuple(entries),
        schema_variants=tuple(variants),
        inventory_blake3=blake3_hex(core),
    )


def verify_policy_catalog_inventory(
    inventory: PolicyCatalogInventory | Mapping[str, Any],
    *,
    expected_inventory_blake3: str | None = None,
) -> None:
    if isinstance(inventory, PolicyCatalogInventory):
        document = inventory.to_document()
    else:
        document = inventory
    if not isinstance(document, Mapping) or set(document) != {
        "schema",
        "policy_count",
        "entries",
        "schema_variants",
        "inventory_blake3",
    }:
        raise MCPCompatibilityError("policy inventory document keys differ")
    if document["schema"] != POLICY_INVENTORY_SCHEMA:
        raise MCPCompatibilityError("policy catalog inventory schema differs")
    if document["policy_count"] != len(document["entries"]):
        raise MCPCompatibilityError("policy inventory count differs")
    _verify_inventory_parts(
        document["entries"], document["schema_variants"], document["inventory_blake3"]
    )
    if (
        expected_inventory_blake3 is not None
        and document["inventory_blake3"] != expected_inventory_blake3
    ):
        raise MCPCompatibilityError("policy inventory does not match expected BLAKE3")


__all__ = [
    "POLICY_BINDING_SCHEMA",
    "POLICY_INVENTORY_SCHEMA",
    "PolicyCatalogBinding",
    "PolicyCatalogInventory",
    "bind_policy_catalog",
    "build_policy_catalog_inventory",
    "verify_policy_catalog_inventory",
    "verify_policy_catalog_binding",
]
