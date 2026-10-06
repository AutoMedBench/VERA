"""Deterministic disjoint composition of candidate execution resolvers.

The legacy execution authority and each premium construction transition own
independent candidate-scoped tool registries.  This adapter combines only
their *routing inventories*: it never merges tools, policies, workspaces, or
runtime state.  A request is sent to exactly one resolver after the complete
union has been proven disjoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence
from uuid import UUID

from eva_agent.pipeline.digests import blake3_hex, canonical_value, is_blake3
from eva_agent.sources.legacy_execution import ProductionSourceBlocker

from .campaign import CandidateExecutionBindingPort, ExecutionBindingResolverPort

if TYPE_CHECKING:
    from eva_agent.construction.premium_supervisor import (
        PremiumConstructionExecutionBindingResolver,
    )
    from eva_agent.sources.legacy_execution import LegacyExecutionBindingResolver


COMPOSITE_EXECUTION_BINDING_CATALOG_SCHEMA = (
    "eva.composite-execution-binding-catalog.v1"
)
COMPOSITE_EXECUTION_BINDING_METADATA_SCHEMA = (
    "eva.composite-execution-binding-resolver-metadata.v1"
)


class CompositeExecutionBindingError(ProductionSourceBlocker):
    """The resolver union or an exact candidate/source route is ambiguous."""


@dataclass(frozen=True, slots=True)
class _InventoryRoute:
    role: str
    resolver: ExecutionBindingResolverPort
    resolver_catalog_blake3: str
    reference: Any
    source_candidate_id: str
    candidate_id: str | None
    reference_blake3: str

    def core_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "role": self.role,
            "resolver_catalog_blake3": self.resolver_catalog_blake3,
            "source_candidate_id": self.source_candidate_id,
            "reference_blake3": self.reference_blake3,
        }
        if self.candidate_id is not None:
            document["candidate_id"] = self.candidate_id
        return document


def _canonical_uuid(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise CompositeExecutionBindingError(f"{label} must be a canonical UUID")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError):
        raise CompositeExecutionBindingError(
            f"{label} must be a canonical UUID"
        ) from None
    if str(parsed) != value:
        raise CompositeExecutionBindingError(f"{label} must be a canonical UUID")
    return value


def _resolver_inventory(
    resolver: ExecutionBindingResolverPort,
    *,
    role: str,
    candidate_id_required: bool,
) -> tuple[_InventoryRoute, ...]:
    catalog_blake3 = getattr(resolver, "catalog_inventory_blake3", None)
    count = getattr(resolver, "executable_candidate_count", None)
    inventory_method = getattr(resolver, "inventory", None)
    resolve_method = getattr(resolver, "resolve", None)
    if (
        not is_blake3(catalog_blake3)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or not callable(inventory_method)
        or not callable(resolve_method)
    ):
        raise CompositeExecutionBindingError(
            f"{role} execution resolver port differs"
        )
    inventory = inventory_method()
    if not isinstance(inventory, tuple) or len(inventory) != count:
        raise CompositeExecutionBindingError(
            f"{role} execution resolver inventory count differs"
        )

    routes: list[_InventoryRoute] = []
    for reference in inventory:
        source_candidate_id = getattr(reference, "source_candidate_id", None)
        if not isinstance(source_candidate_id, str) or not source_candidate_id:
            raise CompositeExecutionBindingError(
                f"{role} execution inventory source identity differs"
            )
        raw_candidate_id = getattr(reference, "candidate_id", None)
        if candidate_id_required and raw_candidate_id is None:
            raise CompositeExecutionBindingError(
                f"{role} execution inventory candidate identity is missing"
            )
        candidate_id = (
            _canonical_uuid(raw_candidate_id, label=f"{role} inventory candidate_id")
            if raw_candidate_id is not None
            else None
        )
        to_document = getattr(reference, "to_document", None)
        if not callable(to_document):
            raise CompositeExecutionBindingError(
                f"{role} execution inventory reference is not documentable"
            )
        try:
            document = canonical_value(to_document())
        except (TypeError, ValueError) as exc:
            raise CompositeExecutionBindingError(
                f"{role} execution inventory reference is not canonical"
            ) from exc
        if not isinstance(document, dict) or document.get(
            "source_candidate_id"
        ) != source_candidate_id:
            raise CompositeExecutionBindingError(
                f"{role} execution inventory reference identity differs"
            )
        if candidate_id is not None and document.get("candidate_id") != candidate_id:
            raise CompositeExecutionBindingError(
                f"{role} execution inventory candidate document differs"
            )
        routes.append(
            _InventoryRoute(
                role=role,
                resolver=resolver,
                resolver_catalog_blake3=catalog_blake3,
                reference=reference,
                source_candidate_id=source_candidate_id,
                candidate_id=candidate_id,
                reference_blake3=blake3_hex(document),
            )
        )
    return tuple(routes)


class CompositeExecutionBindingResolver:
    """Strict union of one legacy and zero or more premium resolvers.

    Component order is not authoritative.  Premium components and the final
    inventory are sorted by their signed catalog and exact identities before
    the composite catalog is committed.  Resolution performs no probing: one
    source identity selects one prevalidated component, and any conflicting
    premium candidate identity fails before delegation.
    """

    def __init__(
        self,
        *,
        legacy_resolver: LegacyExecutionBindingResolver,
        premium_resolvers: Sequence[
            PremiumConstructionExecutionBindingResolver
        ] = (),
    ) -> None:
        if isinstance(premium_resolvers, (str, bytes)) or not isinstance(
            premium_resolvers, Sequence
        ):
            raise CompositeExecutionBindingError(
                "premium execution resolvers must be an ordered sequence"
            )
        premium = tuple(premium_resolvers)
        if any(resolver is legacy_resolver for resolver in premium) or len(
            {id(resolver) for resolver in premium}
        ) != len(premium):
            raise CompositeExecutionBindingError(
                "execution resolver instance is duplicated"
            )

        legacy_routes = _resolver_inventory(
            legacy_resolver,
            role="legacy",
            candidate_id_required=False,
        )
        premium_components: list[
            tuple[str, ExecutionBindingResolverPort, tuple[_InventoryRoute, ...]]
        ] = []
        for resolver in premium:
            routes = _resolver_inventory(
                resolver,
                role="premium",
                candidate_id_required=True,
            )
            digest = getattr(resolver, "catalog_inventory_blake3")
            premium_components.append((digest, resolver, routes))

        component_digests = [
            getattr(legacy_resolver, "catalog_inventory_blake3"),
            *(digest for digest, _resolver, _routes in premium_components),
        ]
        if len(component_digests) != len(set(component_digests)):
            raise CompositeExecutionBindingError(
                "execution resolver catalog is duplicated"
            )

        premium_components.sort(
            key=lambda component: (
                component[0],
                tuple(
                    sorted(
                        (route.candidate_id, route.source_candidate_id)
                        for route in component[2]
                    )
                ),
            )
        )
        routes = [*legacy_routes]
        for _digest, _resolver, component_routes in premium_components:
            routes.extend(component_routes)

        by_source: dict[str, _InventoryRoute] = {}
        by_candidate: dict[str, _InventoryRoute] = {}
        for route in routes:
            if route.source_candidate_id in by_source:
                raise CompositeExecutionBindingError(
                    "execution resolver source inventories overlap: "
                    f"{route.source_candidate_id}"
                )
            by_source[route.source_candidate_id] = route
            if route.candidate_id is not None:
                if route.candidate_id in by_candidate:
                    raise CompositeExecutionBindingError(
                        "execution resolver candidate inventories overlap: "
                        f"{route.candidate_id}"
                    )
                by_candidate[route.candidate_id] = route

        ordered_routes = tuple(
            sorted(
                routes,
                key=lambda route: (
                    route.source_candidate_id,
                    route.candidate_id or "",
                    route.resolver_catalog_blake3,
                ),
            )
        )
        component_documents = [
            {
                "role": "legacy",
                "catalog_inventory_blake3": getattr(
                    legacy_resolver, "catalog_inventory_blake3"
                ),
                "candidate_count": len(legacy_routes),
            },
            *(
                {
                    "role": "premium",
                    "catalog_inventory_blake3": digest,
                    "candidate_count": len(component_routes),
                }
                for digest, _resolver, component_routes in premium_components
            ),
        ]
        self._catalog_core = canonical_value(
            {
                "schema": COMPOSITE_EXECUTION_BINDING_CATALOG_SCHEMA,
                "coverage": "strict_disjoint_resolver_union",
                "component_count": len(component_documents),
                "candidate_count": len(ordered_routes),
                "components": component_documents,
                "rows": [route.core_document() for route in ordered_routes],
            }
        )
        self.catalog_blake3 = blake3_hex(self._catalog_core)
        self._inventory = tuple(route.reference for route in ordered_routes)
        self._by_source = MappingProxyType(by_source)
        self._by_candidate = MappingProxyType(by_candidate)
        self._components = tuple(component_documents)

    @property
    def catalog_inventory_blake3(self) -> str:
        return self.catalog_blake3

    @property
    def executable_candidate_count(self) -> int:
        return len(self._inventory)

    def inventory(self) -> tuple[Any, ...]:
        return self._inventory

    def public_metadata(self) -> Mapping[str, Any]:
        """Return path-free metadata suitable for deployment recipe binding."""

        metadata = canonical_value(
            {
                "schema": COMPOSITE_EXECUTION_BINDING_METADATA_SCHEMA,
                "catalog_schema": COMPOSITE_EXECUTION_BINDING_CATALOG_SCHEMA,
                "catalog_inventory_blake3": self.catalog_blake3,
                "coverage": "strict_disjoint_resolver_union",
                "component_count": len(self._components),
                "executable_candidate_count": len(self._inventory),
                "components": list(self._components),
            }
        )
        assert isinstance(metadata, dict)
        return MappingProxyType(metadata)

    def resolve(
        self,
        eva_candidate_id: str,
        *,
        source_candidate_id: str,
    ) -> CandidateExecutionBindingPort:
        candidate_id = _canonical_uuid(
            eva_candidate_id, label="composite EVA candidate_id"
        )
        if not isinstance(source_candidate_id, str) or not source_candidate_id:
            raise CompositeExecutionBindingError(
                "composite source_candidate_id differs"
            )
        source_route = self._by_source.get(source_candidate_id)
        candidate_route = self._by_candidate.get(candidate_id)
        if source_route is None:
            if candidate_route is not None:
                raise CompositeExecutionBindingError(
                    "candidate/source execution identities disagree"
                )
            raise CompositeExecutionBindingError(
                "candidate/source execution identity is not in the composite inventory"
            )
        if candidate_route is not None and candidate_route is not source_route:
            raise CompositeExecutionBindingError(
                "candidate/source execution resolution is ambiguous"
            )
        if (
            source_route.candidate_id is not None
            and source_route.candidate_id != candidate_id
        ):
            raise CompositeExecutionBindingError(
                "candidate/source execution identities disagree"
            )

        binding = source_route.resolver.resolve(
            candidate_id, source_candidate_id=source_candidate_id
        )
        if not (
            getattr(binding, "candidate_id", None) == candidate_id
            and getattr(binding, "source_candidate_id", None) == source_candidate_id
            and is_blake3(getattr(binding, "binding_blake3", None))
        ):
            raise CompositeExecutionBindingError(
                "resolved candidate execution binding identity differs"
            )
        return binding


__all__ = [
    "COMPOSITE_EXECUTION_BINDING_CATALOG_SCHEMA",
    "COMPOSITE_EXECUTION_BINDING_METADATA_SCHEMA",
    "CompositeExecutionBindingError",
    "CompositeExecutionBindingResolver",
]
