"""In-memory registry for compiled domain × stage rubric tables."""

from __future__ import annotations

from typing import Any, Mapping
from uuid import uuid4

from .integrity import blake3_document, is_blake3_digest
from .models import CompiledRubric, SandboxRubricBinding, canonical_uuid


class RubricRegistryError(ValueError):
    """A compiled registry or lookup violates the registry contract."""


class CompiledRubricRegistry:
    """A versioned set of immutable rubrics, unique by domain and stage."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        if document.get("schema") != "eva.compiled-rubric-registry.v1":
            raise RubricRegistryError("compiled registry schema differs")
        shadow = dict(document)
        recorded_digest = shadow.pop("registry_digest", None)
        if not is_blake3_digest(recorded_digest) or blake3_document(shadow) != recorded_digest:
            raise RubricRegistryError("compiled registry BLAKE3 commitment differs")
        try:
            registry_id = canonical_uuid(str(document["registry_id"]), label="registry_id")
            registry_version = int(document["registry_version"])
            source = list(document["rubrics"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RubricRegistryError("compiled registry metadata differs") from exc
        if type(document["registry_version"]) is not int or registry_version < 1 or not source:
            raise RubricRegistryError("compiled registry version or rubric set differs")

        rubrics: list[CompiledRubric] = []
        by_key: dict[tuple[str, str], CompiledRubric] = {}
        by_id: dict[str, CompiledRubric] = {}
        for raw in source:
            if not isinstance(raw, Mapping):
                raise RubricRegistryError("compiled rubric entry differs")
            self._verify_rubric_document(raw)
            rubric = CompiledRubric.from_document(raw)
            key = (rubric.domain, rubric.stage)
            if key in by_key or rubric.rubric_id in by_id:
                raise RubricRegistryError("compiled rubric key or UUID is duplicated")
            by_key[key] = rubric
            by_id[rubric.rubric_id] = rubric
            rubrics.append(rubric)
        self._registry_id = registry_id
        self._registry_version = registry_version
        self._registry_digest = str(recorded_digest)
        self._compiler_version = str(document["compiler_version"])
        self._source_schema = str(document["source_schema"])
        self._rubrics = tuple(rubrics)
        self._by_key = by_key
        self._by_id = by_id

    @staticmethod
    def _verify_rubric_document(rubric: Mapping[str, Any]) -> None:
        shadow = dict(rubric)
        digest = shadow.pop("rubric_digest", None)
        if not is_blake3_digest(digest) or blake3_document(shadow) != digest:
            raise RubricRegistryError("compiled rubric BLAKE3 commitment differs")
        canonical_uuid(str(rubric.get("rubric_id")), label="rubric_id")
        items = rubric.get("items")
        if not isinstance(items, list) or not 5 <= len(items) <= 10:
            raise RubricRegistryError("compiled rubric item count differs")
        item_ids: set[str] = set()
        selector_ids: set[str] = set()
        for item in items:
            if not isinstance(item, Mapping):
                raise RubricRegistryError("compiled rubric item differs")
            item_shadow = dict(item)
            item_digest = item_shadow.pop("item_digest", None)
            if not is_blake3_digest(item_digest) or blake3_document(item_shadow) != item_digest:
                raise RubricRegistryError("compiled item BLAKE3 commitment differs")
            item_id = canonical_uuid(str(item.get("item_id")), label="item_id")
            if item_id in item_ids:
                raise RubricRegistryError("compiled item UUID is duplicated")
            item_ids.add(item_id)
            for selector in item.get("evidence_selectors", []):
                if not isinstance(selector, Mapping):
                    raise RubricRegistryError("compiled evidence selector differs")
                selector_shadow = dict(selector)
                selector_digest = selector_shadow.pop("selector_digest", None)
                if (
                    not is_blake3_digest(selector_digest)
                    or blake3_document(selector_shadow) != selector_digest
                ):
                    raise RubricRegistryError("compiled selector BLAKE3 commitment differs")
                selector_id = canonical_uuid(
                    str(selector.get("selector_id")), label="selector_id"
                )
                if selector_id in selector_ids:
                    raise RubricRegistryError("compiled selector UUID is duplicated")
                selector_ids.add(selector_id)

    @property
    def registry_id(self) -> str:
        return self._registry_id

    @property
    def registry_version(self) -> int:
        return self._registry_version

    @property
    def digest(self) -> str:
        return self._registry_digest

    @property
    def rubrics(self) -> tuple[CompiledRubric, ...]:
        return self._rubrics

    def resolve(self, domain: str, stage: str) -> CompiledRubric:
        try:
            return self._by_key[(domain, stage)]
        except KeyError:
            raise RubricRegistryError(f"no compiled rubric for {domain!r} × {stage!r}") from None

    def resolve_id(self, rubric_id: str) -> CompiledRubric:
        try:
            return self._by_id[canonical_uuid(rubric_id, label="rubric_id")]
        except KeyError:
            raise RubricRegistryError(f"unknown compiled rubric UUID: {rubric_id}") from None

    def bind_sandbox(
        self,
        sandbox_id: str,
        domain: str,
        stage: str,
        *,
        binding_id: str | None = None,
    ) -> SandboxRubricBinding:
        sandbox = canonical_uuid(sandbox_id, label="sandbox_id")
        binding = canonical_uuid(binding_id or str(uuid4()), label="binding_id")
        return SandboxRubricBinding(
            binding_id=binding,
            sandbox_id=sandbox,
            rubric=self.resolve(domain, stage),
            registry_id=self.registry_id,
            registry_version=self.registry_version,
            registry_digest=self.digest,
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": "eva.compiled-rubric-registry.v1",
            "compiler_version": self._compiler_version,
            "source_schema": self._source_schema,
            "registry_id": self.registry_id,
            "registry_version": self.registry_version,
            "rubrics": [rubric.to_document() for rubric in self.rubrics],
            "registry_digest": self.digest,
        }
