from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest

from eva_agent.deployment import (
    COMPOSITE_EXECUTION_BINDING_CATALOG_SCHEMA,
    CompositeExecutionBindingError,
    CompositeExecutionBindingResolver,
    ExecutionBindingResolverPort,
)
from eva_agent.pipeline.digests import blake3_hex


def _uuid(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"eva-composite-resolver:{label}"))


@dataclass(frozen=True)
class _Reference:
    source_candidate_id: str
    marker: str
    candidate_id: str | None = None

    def to_document(self):
        document = {
            "source_candidate_id": self.source_candidate_id,
            "marker": self.marker,
        }
        if self.candidate_id is not None:
            document["candidate_id"] = self.candidate_id
        return document


class _Resolver:
    def __init__(
        self,
        label: str,
        references: tuple[_Reference, ...],
        *,
        barrier: Barrier | None = None,
        bad_binding_identity: bool = False,
    ) -> None:
        self._references = references
        self._by_source = {row.source_candidate_id: row for row in references}
        self.catalog_inventory_blake3 = blake3_hex(
            {
                "resolver": label,
                "rows": [row.to_document() for row in references],
            }
        )
        self.executable_candidate_count = len(references)
        self.barrier = barrier
        self.bad_binding_identity = bad_binding_identity
        self.calls: list[tuple[str, str]] = []

    def inventory(self):
        return self._references

    def resolve(self, eva_candidate_id: str, *, source_candidate_id: str):
        if source_candidate_id not in self._by_source:
            raise KeyError(source_candidate_id)
        reference = self._by_source[source_candidate_id]
        if reference.candidate_id is not None and reference.candidate_id != eva_candidate_id:
            raise KeyError(eva_candidate_id)
        self.calls.append((eva_candidate_id, source_candidate_id))
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        resolved_source = (
            f"wrong-{source_candidate_id}"
            if self.bad_binding_identity
            else source_candidate_id
        )
        return SimpleNamespace(
            candidate_id=eva_candidate_id,
            source_candidate_id=resolved_source,
            initial_workspace_files={".eva/context.json": b"{}\n"},
            public_runtime_context={"marker": reference.marker},
            tool_registry=object(),
            binding_blake3=blake3_hex(
                {
                    "candidate_id": eva_candidate_id,
                    "source_candidate_id": resolved_source,
                }
            ),
        )


def _legacy(*sources: str) -> _Resolver:
    return _Resolver(
        "legacy",
        tuple(
            _Reference(source_candidate_id=source, marker=f"legacy:{source}")
            for source in sources
        ),
    )


def _premium(label: str, *pairs: tuple[str, str], barrier=None) -> _Resolver:
    return _Resolver(
        label,
        tuple(
            _Reference(
                source_candidate_id=source,
                marker=f"premium:{label}:{source}",
                candidate_id=candidate,
            )
            for candidate, source in pairs
        ),
        barrier=barrier,
    )


def test_union_routes_to_exact_owner_without_merging_bindings() -> None:
    legacy_candidate = _uuid("legacy-a")
    premium_candidate_a = _uuid("premium-a")
    premium_candidate_b = _uuid("premium-b")
    legacy = _legacy("legacy-source")
    premium_a = _premium("premium-a", (premium_candidate_a, "premium-source-a"))
    premium_b = _premium("premium-b", (premium_candidate_b, "premium-source-b"))
    resolver = CompositeExecutionBindingResolver(
        legacy_resolver=legacy,
        premium_resolvers=(premium_a, premium_b),
    )

    assert isinstance(resolver, ExecutionBindingResolverPort)
    assert resolver.executable_candidate_count == 3
    assert tuple(row.source_candidate_id for row in resolver.inventory()) == (
        "legacy-source",
        "premium-source-a",
        "premium-source-b",
    )
    legacy_binding = resolver.resolve(
        legacy_candidate, source_candidate_id="legacy-source"
    )
    premium_binding = resolver.resolve(
        premium_candidate_b, source_candidate_id="premium-source-b"
    )
    assert legacy_binding.public_runtime_context["marker"] == "legacy:legacy-source"
    assert premium_binding.public_runtime_context["marker"] == (
        "premium:premium-b:premium-source-b"
    )
    assert legacy.calls == [(legacy_candidate, "legacy-source")]
    assert premium_a.calls == []
    assert premium_b.calls == [(premium_candidate_b, "premium-source-b")]
    metadata = resolver.public_metadata()
    assert metadata["catalog_schema"] == COMPOSITE_EXECUTION_BINDING_CATALOG_SCHEMA
    assert metadata["catalog_inventory_blake3"] == resolver.catalog_blake3
    assert metadata["executable_candidate_count"] == 3


def test_overlap_and_ambiguous_identity_are_rejected_before_delegation() -> None:
    first_candidate = _uuid("overlap-a")
    second_candidate = _uuid("overlap-b")
    legacy = _legacy("legacy-source")
    with pytest.raises(CompositeExecutionBindingError, match="source inventories overlap"):
        CompositeExecutionBindingResolver(
            legacy_resolver=legacy,
            premium_resolvers=(
                _premium("same-source", (first_candidate, "legacy-source")),
            ),
        )

    with pytest.raises(CompositeExecutionBindingError, match="candidate inventories overlap"):
        CompositeExecutionBindingResolver(
            legacy_resolver=legacy,
            premium_resolvers=(
                _premium("candidate-a", (first_candidate, "premium-source-a")),
                _premium("candidate-b", (first_candidate, "premium-source-b")),
            ),
        )

    premium_a = _premium("premium-a", (first_candidate, "premium-source-a"))
    premium_b = _premium("premium-b", (second_candidate, "premium-source-b"))
    resolver = CompositeExecutionBindingResolver(
        legacy_resolver=legacy,
        premium_resolvers=(premium_a, premium_b),
    )
    with pytest.raises(CompositeExecutionBindingError, match="ambiguous"):
        resolver.resolve(first_candidate, source_candidate_id="premium-source-b")
    assert premium_a.calls == premium_b.calls == []


def test_catalog_digest_and_inventory_are_independent_of_premium_order() -> None:
    legacy = _legacy("legacy-z", "legacy-a")
    premium_a = _premium("premium-a", (_uuid("stable-a"), "premium-z"))
    premium_b = _premium("premium-b", (_uuid("stable-b"), "premium-a"))
    first = CompositeExecutionBindingResolver(
        legacy_resolver=legacy,
        premium_resolvers=(premium_a, premium_b),
    )
    second = CompositeExecutionBindingResolver(
        legacy_resolver=legacy,
        premium_resolvers=(premium_b, premium_a),
    )

    assert first.catalog_inventory_blake3 == second.catalog_inventory_blake3
    assert first.public_metadata() == second.public_metadata()
    assert tuple(row.source_candidate_id for row in first.inventory()) == tuple(
        row.source_candidate_id for row in second.inventory()
    ) == ("legacy-a", "legacy-z", "premium-a", "premium-z")


def test_distinct_resolutions_remain_concurrent() -> None:
    barrier = Barrier(2)
    candidate_a = _uuid("parallel-a")
    candidate_b = _uuid("parallel-b")
    premium = _premium(
        "parallel",
        (candidate_a, "parallel-source-a"),
        (candidate_b, "parallel-source-b"),
        barrier=barrier,
    )
    resolver = CompositeExecutionBindingResolver(
        legacy_resolver=_legacy("legacy-source"),
        premium_resolvers=(premium,),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            resolver.resolve,
            candidate_a,
            source_candidate_id="parallel-source-a",
        )
        second = pool.submit(
            resolver.resolve,
            candidate_b,
            source_candidate_id="parallel-source-b",
        )
        bindings = (first.result(timeout=3), second.result(timeout=3))
    assert {binding.candidate_id for binding in bindings} == {
        candidate_a,
        candidate_b,
    }


def test_unknown_or_wrong_candidate_and_bad_delegate_identity_fail_closed() -> None:
    candidate = _uuid("known")
    premium = _premium("known", (candidate, "known-source"))
    resolver = CompositeExecutionBindingResolver(
        legacy_resolver=_legacy("legacy-source"),
        premium_resolvers=(premium,),
    )
    with pytest.raises(CompositeExecutionBindingError, match="not in the composite"):
        resolver.resolve(_uuid("unknown"), source_candidate_id="unknown-source")
    with pytest.raises(CompositeExecutionBindingError, match="identities disagree"):
        resolver.resolve(_uuid("wrong"), source_candidate_id="known-source")
    assert premium.calls == []

    bad_legacy = _Resolver(
        "bad-legacy",
        (_Reference("bad-source", "bad"),),
        bad_binding_identity=True,
    )
    bad = CompositeExecutionBindingResolver(legacy_resolver=bad_legacy)
    with pytest.raises(CompositeExecutionBindingError, match="binding identity differs"):
        bad.resolve(_uuid("bad"), source_candidate_id="bad-source")
