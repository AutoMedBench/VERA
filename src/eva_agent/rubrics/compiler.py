"""Compile source rubric definitions into one canonical reward contract."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .integrity import blake3_document, canonical_json_bytes
from .models import canonical_uuid
from .registry import CompiledRubricRegistry


SOURCE_SCHEMA = "eva.rubric-registry.v1"
COMPILED_SCHEMA = "eva.compiled-rubric-registry.v1"
COMPILER_VERSION = "eva.rubric-compiler.v1"
STAGES = ("S1", "S2", "S3", "S4", "S5", "E2E")
SCHEMA_DIR = Path(__file__).resolve().parents[3] / "schemas"
SOURCE_SCHEMA_PATH = SCHEMA_DIR / "rubric-registry.v1.schema.json"
COMPILED_SCHEMA_PATH = SCHEMA_DIR / "compiled-rubric-registry.v1.schema.json"


class RubricValidationError(ValueError):
    """A source or compiled rubric violates a declared invariant."""


def _load_schema(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise RubricValidationError(f"rubric schema cannot be loaded: {path.name}") from exc
    if not isinstance(value, dict):
        raise RubricValidationError(f"rubric schema is not an object: {path.name}")
    return value


def _validate_schema(value: Mapping[str, Any], schema_path: Path, *, label: str) -> None:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:  # pragma: no cover - dependency bootstrap owns this
        raise RubricValidationError("rubric compilation requires the 'jsonschema' package") from exc
    validator = Draft202012Validator(
        _load_schema(schema_path), format_checker=FormatChecker()
    )
    errors = sorted(
        validator.iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise RubricValidationError(f"{label} schema violation at {location}: {error.message}")


def _safe_relative_path(value: str, *, label: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise RubricValidationError(f"{label} must be a safe relative path")


def _validate_partial_credit(item: Mapping[str, Any]) -> None:
    partial = item["partial_credit"]
    levels = partial["levels"]
    scores = [level["score_bps"] for level in levels]
    if scores != sorted(set(scores)) or scores[0] != 0 or scores[-1] != 10000:
        raise RubricValidationError("partial-credit levels must be unique, ascending, and span 0..10000")
    if partial["mode"] == "binary" and scores != [0, 10000]:
        raise RubricValidationError("binary scoring must contain exactly 0 and 10000")
    if partial["mode"] == "levels" and len(scores) < 3:
        raise RubricValidationError("partial-credit level scoring requires an intermediate level")


def _validate_selector(selector: Mapping[str, Any], *, rubric_stage: str) -> None:
    source = selector["source"]
    kind = selector["kind"]
    query = selector["query"]
    context_kinds = {"json_pointer", "event_type", "text_query"}
    workspace_kinds = {
        "json_pointer", "text_query", "path", "glob", "file_content", "workspace_diff"
    }
    if kind not in (context_kinds if source == "context" else workspace_kinds):
        raise RubricValidationError(f"selector kind {kind!r} is incompatible with {source!r}")
    if kind == "json_pointer" and not query.startswith("/"):
        raise RubricValidationError("json_pointer selector queries must start with '/'")
    if source == "workspace" and kind in {"path", "glob", "file_content"}:
        _safe_relative_path(query, label="workspace selector query")
    evidence_stage = selector["evidence_stage"]
    if rubric_stage != "E2E" and evidence_stage != rubric_stage:
        raise RubricValidationError("stage rubric selectors may cite only their own stage")


def _validate_source_cross_fields(document: Mapping[str, Any]) -> None:
    rubric_ids: set[str] = set()
    item_ids: set[str] = set()
    selector_ids: set[str] = set()
    domain_stages: set[tuple[str, str]] = set()
    for rubric in document["rubrics"]:
        rubric_id = canonical_uuid(rubric["rubric_id"], label="rubric_id")
        key = (rubric["domain"], rubric["stage"])
        if rubric_id in rubric_ids or key in domain_stages:
            raise RubricValidationError("rubric UUID or domain × stage table is duplicated")
        rubric_ids.add(rubric_id)
        domain_stages.add(key)
        for item in rubric["items"]:
            item_id = canonical_uuid(item["item_id"], label="item_id")
            if item_id in item_ids:
                raise RubricValidationError("item UUID is duplicated across the registry")
            item_ids.add(item_id)
            _validate_partial_credit(item)
            labels = set(item["labels"])
            provenance = item["provenance"]
            if provenance["origin"] == "benchmark":
                _safe_relative_path(provenance["source_file"], label="benchmark source_file")
                if provenance["domain"] != rubric["domain"]:
                    raise RubricValidationError("benchmark provenance domain differs from rubric")
                if rubric["stage"] != "E2E" and provenance["stage"] != rubric["stage"]:
                    raise RubricValidationError("benchmark provenance stage differs from rubric")
                if "supplemental" in labels:
                    raise RubricValidationError("benchmark-derived items cannot carry supplemental label")
            else:
                required_labels = {"supplemental", provenance["supplemental_label"]}
                if not required_labels <= labels:
                    raise RubricValidationError(
                        "supplemental items must carry supplemental and supplemental_label labels"
                    )
            for selector in item["evidence_selectors"]:
                selector_id = canonical_uuid(selector["selector_id"], label="selector_id")
                if selector_id in selector_ids:
                    raise RubricValidationError("selector UUID is duplicated across the registry")
                selector_ids.add(selector_id)
                _validate_selector(selector, rubric_stage=rubric["stage"])


def _normalized_weights(items: Sequence[Mapping[str, Any]]) -> list[int]:
    total = sum(int(item["weight"]) for item in items)
    base = [int(item["weight"]) * 10000 // total for item in items]
    remainders = [int(item["weight"]) * 10000 % total for item in items]
    for index in sorted(range(len(items)), key=lambda position: (-remainders[position], position))[
        : 10000 - sum(base)
    ]:
        base[index] += 1
    if sum(base) != 10000:
        raise RubricValidationError("normalized weight allocation differs")
    return base


def _compile_selector(selector: Mapping[str, Any], *, ordinal: int) -> dict[str, Any]:
    value = {
        "schema": "eva.compiled-evidence-selector.v1",
        "ordinal": ordinal,
        **deepcopy(dict(selector)),
    }
    value["selector_digest"] = blake3_document(value)
    return value


def _compile_item(
    item: Mapping[str, Any], *, ordinal: int, normalized_weight_bps: int
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": "eva.compiled-rubric-item.v1",
        "ordinal": ordinal,
        "item_id": item["item_id"],
        "title": item["title"],
        "description": item["description"],
        "atomic": True,
        "observable": True,
        "weight": item["weight"],
        "normalized_weight_bps": normalized_weight_bps,
        "evidence_selectors": [
            _compile_selector(selector, ordinal=selector_ordinal)
            for selector_ordinal, selector in enumerate(item["evidence_selectors"])
        ],
        "partial_credit": deepcopy(item["partial_credit"]),
        "provenance": deepcopy(item["provenance"]),
        "labels": list(item["labels"]),
    }
    if "hard_gate" in item:
        value["hard_gate"] = deepcopy(item["hard_gate"])
    value["item_digest"] = blake3_document(value)
    return value


def _compile_rubric(rubric: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalized_weights(rubric["items"])
    items = [
        _compile_item(item, ordinal=index, normalized_weight_bps=normalized[index])
        for index, item in enumerate(rubric["items"])
    ]
    value = {
        "schema": "eva.compiled-rubric.v1",
        "rubric_id": rubric["rubric_id"],
        "version": rubric["version"],
        "domain": rubric["domain"],
        "stage": rubric["stage"],
        "title": rubric["title"],
        "description": rubric["description"],
        "items": items,
        "total_weight": sum(item["weight"] for item in rubric["items"]),
        "hard_gate_item_ids": [item["item_id"] for item in rubric["items"] if "hard_gate" in item],
    }
    value["rubric_digest"] = blake3_document(value)
    return value


def compile_registry(document: Mapping[str, Any]) -> CompiledRubricRegistry:
    """Validate and compile a source registry into immutable shared tables."""

    if not isinstance(document, Mapping):
        raise RubricValidationError("source rubric registry must be an object")
    _validate_schema(document, SOURCE_SCHEMA_PATH, label="source rubric registry")
    canonical_uuid(document["registry_id"], label="registry_id")
    _validate_source_cross_fields(document)
    rubrics = sorted(
        (_compile_rubric(rubric) for rubric in document["rubrics"]),
        key=lambda rubric: (rubric["domain"], STAGES.index(rubric["stage"])),
    )
    compiled = {
        "schema": COMPILED_SCHEMA,
        "compiler_version": COMPILER_VERSION,
        "source_schema": SOURCE_SCHEMA,
        "registry_id": document["registry_id"],
        "registry_version": document["registry_version"],
        "rubrics": rubrics,
    }
    compiled["registry_digest"] = blake3_document(compiled)
    _validate_schema(compiled, COMPILED_SCHEMA_PATH, label="compiled rubric registry")
    return CompiledRubricRegistry(compiled)


def load_and_compile_registry(path: str | Path) -> CompiledRubricRegistry:
    """Load a UTF-8 source registry and compile it with the v1 contract."""

    source_path = Path(path)
    try:
        raw = source_path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeError, ValueError) as exc:
        raise RubricValidationError(f"rubric registry cannot be loaded: {source_path}") from exc
    if canonical_json_bytes(document) != canonical_json_bytes(document):  # type/nan validation
        raise RubricValidationError("rubric registry canonicalization differs")
    return compile_registry(document)
