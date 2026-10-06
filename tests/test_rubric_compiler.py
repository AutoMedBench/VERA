from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from eva_agent.rubrics import (
    CompiledRubricRegistry,
    RubricRegistryError,
    RubricScoreError,
    RubricValidationError,
    blake3_document,
    compile_registry,
    load_and_compile_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def uid(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012d}"


def rubric_item(
    number: int,
    *,
    domain: str,
    stage: str,
    evidence_stage: str | None = None,
    supplemental: bool = False,
) -> dict:
    selector = {
        "selector_id": uid(10_000 + number),
        "source": "context" if number % 2 == 0 else "workspace",
        "kind": "json_pointer" if number % 2 == 0 else "file_content",
        "query": "/events/tool_calls" if number % 2 == 0 else f"artifacts/result-{number}.json",
        "evidence_stage": evidence_stage or stage,
        "required": True,
        "aggregation": "all",
    }
    item = {
        "item_id": uid(1_000 + number),
        "title": f"Observable outcome {number}",
        "description": f"The trajectory produces independently observable outcome {number}.",
        "atomic": True,
        "observable": True,
        "weight": number + 1,
        "evidence_selectors": [selector],
        "partial_credit": {
            "mode": "levels" if number % 10 == 0 else "binary",
            "levels": (
                [
                    {"score_bps": 0, "criterion": "No observable evidence."},
                    {"score_bps": 5000, "criterion": "Incomplete but valid evidence."},
                    {"score_bps": 10000, "criterion": "Complete observable evidence."},
                ]
                if number % 10 == 0
                else [
                    {"score_bps": 0, "criterion": "Criterion not met."},
                    {"score_bps": 10000, "criterion": "Criterion fully met."},
                ]
            ),
        },
        "provenance": (
            {
                "origin": "supplemental",
                "supplemental_label": "tool-integrity",
                "rationale": "Adds an explicit observable tool-integrity check.",
            }
            if supplemental
            else {
                "origin": "benchmark",
                "source_benchmark": "fixture-benchmark",
                "source_file": "benchmarks/fixture/examples.jsonl",
                "source_revision": "fixture-rev-1",
                "domain": domain,
                "stage": evidence_stage or stage,
                "extraction_status": "human_verified",
            }
        ),
        "labels": ["supplemental", "tool-integrity"] if supplemental else ["benchmark"],
    }
    if number % 10 == 0:
        item["hard_gate"] = {"category": "sandbox-validity", "minimum_score_bps": 10000}
    return item


def rubric(number: int, *, domain: str, stage: str) -> dict:
    items = [
        rubric_item(
            number * 10 + index,
            domain=domain,
            stage=stage,
            evidence_stage=(f"S{index + 1}" if stage == "E2E" else stage),
            supplemental=index == 4,
        )
        for index in range(5)
    ]
    # The scoring fixture expects the first item to have weight 1..5 rather
    # than an offset based on its globally unique fixture number.
    for index, item in enumerate(items):
        item["weight"] = index + 1
    return {
        "rubric_id": uid(100 + number),
        "version": 1,
        "domain": domain,
        "stage": stage,
        "title": f"{domain} {stage} rubric",
        "description": "Five atomic checks used by both judging and reward.",
        "items": items,
    }


def source_registry() -> dict:
    return {
        "schema": "eva.rubric-registry.v1",
        "registry_id": uid(1),
        "registry_version": 1,
        "rubrics": [
            rubric(2, domain="visual-vibe-coding", stage="E2E"),
            rubric(1, domain="medical-research", stage="S1"),
        ],
    }


def test_compile_registry_is_deterministic_versioned_and_blake3_committed() -> None:
    source = source_registry()
    first = compile_registry(source)
    second_source = deepcopy(source)
    second_source["rubrics"].reverse()
    second = compile_registry(second_source)

    assert first.digest == second.digest
    assert [
        (table.domain, table.stage) for table in first.rubrics
    ] == [("medical-research", "S1"), ("visual-vibe-coding", "E2E")]
    assert first.digest.startswith("blake3:") and len(first.digest) == 71
    document = first.to_document()
    shadow = dict(document)
    assert shadow.pop("registry_digest") == blake3_document(shadow)
    for table in first.rubrics:
        compiled = table.to_document()
        assert compiled["schema"] == "eva.compiled-rubric.v1"
        assert len(compiled["items"]) == 5
        assert sum(item["normalized_weight_bps"] for item in compiled["items"]) == 10000
        core = dict(compiled)
        assert core.pop("rubric_digest") == blake3_document(core)
        assert all(item["item_digest"].startswith("blake3:") for item in compiled["items"])


def test_each_sandbox_binds_one_table_and_both_consumers_share_object_identity() -> None:
    registry = compile_registry(source_registry())
    binding = registry.bind_sandbox(
        uid(80_001),
        "visual-vibe-coding",
        "E2E",
        binding_id=uid(80_002),
    )
    benchmark_table = binding.for_benchmark_judging()
    reward_table = binding.for_rollout_reward()

    assert benchmark_table is reward_table
    assert benchmark_table is registry.resolve("visual-vibe-coding", "E2E")
    assert binding.to_document()["rubric_digest"] == benchmark_table.digest
    schema = json.loads(
        (ROOT / "schemas/sandbox-rubric-binding.v1.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(binding.to_document())
    with pytest.raises(RubricRegistryError, match="no compiled rubric"):
        registry.bind_sandbox(uid(80_003), "visual-vibe-coding", "S3")


def test_e2e_table_may_cite_stage_evidence_but_stage_table_may_not() -> None:
    registry = compile_registry(source_registry())
    e2e = registry.resolve("visual-vibe-coding", "E2E")
    assert {selector["evidence_stage"] for item in e2e.items for selector in item["evidence_selectors"]} == {
        "S1", "S2", "S3", "S4", "S5"
    }

    invalid = source_registry()
    invalid["rubrics"][1]["items"][0]["evidence_selectors"][0]["evidence_stage"] = "S2"
    with pytest.raises(RubricValidationError, match="own stage"):
        compile_registry(invalid)


def test_partial_credit_and_hard_gate_reward_use_compiled_semantics() -> None:
    table = compile_registry(source_registry()).resolve("medical-research", "S1")
    item_ids = [item["item_id"] for item in table.items]
    scores = {item_id: 10000 for item_id in item_ids}
    scores[item_ids[0]] = 5000
    result = table.score(scores, evaluation_id=uid(90_001))

    assert result.weighted_score_bps == 9667
    assert result.hard_gate_passed is False
    assert result.failed_hard_gate_item_ids == (item_ids[0],)
    assert result.reward_bps == 0
    assert result.score_digest.startswith("blake3:")

    scores[item_ids[0]] = 10000
    passed = table.score(scores, evaluation_id=uid(90_002))
    assert passed.weighted_score_bps == passed.reward_bps == 10000
    assert passed.hard_gate_passed is True

    scores[item_ids[0]] = 7500
    with pytest.raises(RubricScoreError, match="partial-credit level"):
        table.score(scores)
    with pytest.raises(RubricScoreError, match="exactly match"):
        table.score(dict(list(scores.items())[:-1]))


@pytest.mark.parametrize("count", [4, 11])
def test_rubric_tables_require_five_to_ten_atomic_items(count: int) -> None:
    source = source_registry()
    source["rubrics"][1]["items"] = source["rubrics"][1]["items"][:count]
    if count == 11:
        base = source["rubrics"][1]
        base["items"] = [
            rubric_item(500 + index, domain=base["domain"], stage=base["stage"])
            for index in range(11)
        ]
    with pytest.raises(RubricValidationError, match="schema violation"):
        compile_registry(source)


def test_benchmark_and_supplemental_provenance_fail_closed() -> None:
    wrong_domain = source_registry()
    wrong_domain["rubrics"][1]["items"][0]["provenance"]["domain"] = "other-domain"
    with pytest.raises(RubricValidationError, match="provenance domain"):
        compile_registry(wrong_domain)

    traversal = source_registry()
    traversal["rubrics"][1]["items"][0]["provenance"]["source_file"] = "../private.json"
    with pytest.raises(RubricValidationError, match="safe relative path"):
        compile_registry(traversal)

    unlabeled = source_registry()
    unlabeled["rubrics"][1]["items"][4]["labels"] = ["tool-integrity"]
    with pytest.raises(RubricValidationError, match="supplemental items"):
        compile_registry(unlabeled)


def test_compiled_registry_is_immutable_and_detects_tampering() -> None:
    source = source_registry()
    registry = compile_registry(source)
    table = registry.resolve("medical-research", "S1")
    source["rubrics"][1]["items"][0]["title"] = "mutated after compilation"
    assert table.items[0]["title"] != "mutated after compilation"
    with pytest.raises(TypeError):
        table.items[0]["title"] = "cannot mutate"  # type: ignore[index]

    tampered = registry.to_document()
    tampered["rubrics"][0]["title"] = "tampered"
    with pytest.raises(RubricRegistryError, match="registry BLAKE3"):
        CompiledRubricRegistry(tampered)


def test_load_and_compile_registry_round_trip(tmp_path: Path) -> None:
    source_path = tmp_path / "rubrics.json"
    source_path.write_text(json.dumps(source_registry(), indent=2), encoding="utf-8")
    loaded = load_and_compile_registry(source_path)
    direct = compile_registry(source_registry())
    assert loaded.to_document() == direct.to_document()


def test_duplicate_domain_stage_and_uuid_are_rejected() -> None:
    duplicated_table = source_registry()
    duplicate = deepcopy(duplicated_table["rubrics"][1])
    duplicate["rubric_id"] = uid(999)
    duplicated_table["rubrics"].append(duplicate)
    with pytest.raises(RubricValidationError, match="domain × stage"):
        compile_registry(duplicated_table)

    duplicated_item = source_registry()
    duplicated_item["rubrics"][0]["items"][0]["item_id"] = (
        duplicated_item["rubrics"][1]["items"][0]["item_id"]
    )
    with pytest.raises(RubricValidationError, match="item UUID"):
        compile_registry(duplicated_item)
