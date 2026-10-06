from __future__ import annotations

from collections import Counter
import json
from itertools import product
from pathlib import Path
from typing import Any

from eva_agent.rubrics import load_and_compile_registry
from eva_agent.rubrics.integrity import blake3_document


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "rubrics" / "source"
REGISTRY_PATH = SOURCE_DIR / "domain-stage-tables.v1.json"
CATALOG_PATH = SOURCE_DIR / "catalog.v1.json"
NATIVE_RULES_PATH = SOURCE_DIR / "benchmark-native-rules.v1.json"

BAT_V2_REVISION = "27124a2a35fa9ee68ef890bd7e4a36ffb5802860"
AUTOMEDBENCH_REVISION = "5394fe7aa73e6b5891fe43942c99f4b0c2b50873"
HEALTHBENCH_REVISION = "349962fd46dd02343a0d8a606491baf59154ea1a"
HF_REVISION = "3ee184ab535a97af2fe3cb19f69bd0917b876dcc"
HF_REPOSITORY = "operator/rlevo-CoWork-RL-data"
STAGES = ("S1", "S2", "S3", "S4", "S5", "E2E")
DOMAIN_SOURCES = {
    "automedbench-classification": {
        "benchmark": "AutoMedBench",
        "revision": BAT_V2_REVISION,
        "stage_sources": {
            **{stage: {"medical-research/src/batmed/autocomed_bench/classification/rubric.py"} for stage in STAGES[:-1]},
            "E2E": {"medical-research/contracts/post_sft_rubrics_v3.json"},
        },
        "adapted_counts": (3, 3, 3, 3, 3, 5),
    },
    "automedbench-detection": {
        "benchmark": "AutoMedBench",
        "revision": BAT_V2_REVISION,
        "stage_sources": {
            **{stage: {"medical-research/src/batmed/autocomed_bench/detection/rubric.py"} for stage in STAGES[:-1]},
            "E2E": {"medical-research/contracts/post_sft_rubrics_v3.json"},
        },
        "adapted_counts": (3, 3, 3, 3, 3, 5),
    },
    "automedbench-segmentation": {
        "benchmark": "AutoMedBench",
        "revision": BAT_V2_REVISION,
        "stage_sources": {
            **{stage: {"medical-research/src/batmed/autocomed_bench/segmentation/rubric.py"} for stage in STAGES[:-1]},
            "E2E": {"medical-research/contracts/post_sft_rubrics_v3.json"},
        },
        "adapted_counts": (3, 3, 3, 3, 3, 5),
    },
    "medxpertqa": {
        "benchmark": "MedXpertQA",
        "revision": BAT_V2_REVISION,
        "stage_sources": {
            **{stage: {"medical-research/src/batmed/autocomed_bench/medxpertqa_text/rubric.py"} for stage in STAGES[:-1]},
            "E2E": {"medical-research/contracts/post_sft_rubrics_v3.json"},
        },
        "adapted_counts": (4, 4, 5, 4, 4, 5),
    },
    "agentclinic": {
        "benchmark": "AgentClinic",
        "revision": BAT_V2_REVISION,
        "stage_sources": {
            **{stage: {"medical-research/src/batmed/autocomed_bench/agentclinic/rubric.py"} for stage in STAGES[:-1]},
            "E2E": {"medical-research/contracts/post_sft_rubrics_v3.json"},
        },
        "adapted_counts": (3, 3, 3, 3, 3, 5),
    },
    "automedbench-research": {
        "benchmark": "AutoMedBench",
        "revision": AUTOMEDBENCH_REVISION,
        "stage_sources": {
            "S1": {"README.md", "docs/task-difficulty-tiers.md"},
            **{stage: {"README.md"} for stage in STAGES[1:]},
        },
        "adapted_counts": (3, 3, 3, 3, 3, 5),
    },
    "healthbench-professional": {
        "benchmark": "HealthBench Professional",
        "revision": HEALTHBENCH_REVISION,
        "stage_sources": {**{stage: set() for stage in STAGES[:-1]}, "E2E": {"README.md"}},
        "adapted_counts": (0, 0, 0, 0, 0, 6),
    },
}
NEW_DOMAINS = {"automedbench-research", "healthbench-professional"}
LEGACY_TABLES_BLAKE3 = "blake3:7e9b0e728d830c7e9702629a60afeafcec05bf62af62539f6e42c9da7d4399d0"
SOURCE_PINS = {
    "automedbench-research": {
        "repository": "AutoMedBench/AutoMedBench",
        "locator_prefix": "source-",
        "file_blake3": {
            "README.md": "b41fbf88511b60a44de80e01612863b828f2114aedd5e3bf0554b32819138e88",
            "docs/task-difficulty-tiers.md": "cec1266b85e1cdc9931ee9754f929e39ae692ea0192b2e79d8f61f5021a81954",
        },
        "locator_labels": {
            "README.md": {
                "source-readme-l26-l32-l74-l85",
                "source-readme-l26-l32-l150-l159",
                "source-readme-l74-l91",
                "source-readme-l76-l85",
                "source-readme-l76-l88",
                "source-readme-l76-l89",
                "source-readme-l87-l91",
                "source-readme-l87-l91-l141-l148",
                "source-readme-l150-l159",
            },
            "docs/task-difficulty-tiers.md": {
                "source-task-difficulty-tiers-l9-l15"
            },
        },
    },
    "healthbench-professional": {
        "repository": "openai/healthbench-professional",
        "locator_prefix": "source-",
        "file_blake3": {
            "README.md": "08610f961c3389720627343f1f48936069bc3be237dccdb90d0c86f9f6abd81f",
        },
        "locator_labels": {
            "README.md": {
                "source-readme-l11-l19",
                "source-readme-l21-l23",
            }
        },
    },
}
BAT_CONTRACT_BLAKE3 = "909cc8def38bfe47ddd6c1bc6d28449772ce3c62d61daa8ada6d75245ec913fb"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {
            *(str(key).casefold() for key in value),
            *(key for child in value.values() for key in nested_keys(child)),
        }
    if isinstance(value, list):
        return {key for child in value for key in nested_keys(child)}
    return set()


def test_canonical_domain_stage_source_compiles_and_covers_every_pair() -> None:
    source = load_json(REGISTRY_PATH)
    assert source["schema"] == "eva.rubric-registry.v1"
    assert source["registry_version"] == 2

    expected_pairs = set(product(DOMAIN_SOURCES, STAGES))
    source_pairs = {(table["domain"], table["stage"]) for table in source["rubrics"]}
    assert source_pairs == expected_pairs
    assert len(source["rubrics"]) == 42
    assert sum(len(table["items"]) for table in source["rubrics"]) == 252

    compiled = load_and_compile_registry(REGISTRY_PATH)
    compiled_again = load_and_compile_registry(REGISTRY_PATH)
    assert compiled.digest == compiled_again.digest
    assert compiled.digest.startswith("blake3:")
    assert len(compiled.rubrics) == 42

    for domain, stage in sorted(expected_pairs):
        table = compiled.resolve(domain, stage)
        assert len(table.items) == 6
        assert 5 <= len(table.items) <= 10
        assert sum(int(item["normalized_weight_bps"]) for item in table.items) == 10_000
        assert all(item["atomic"] is True and item["observable"] is True for item in table.items)
        assert all(item["partial_credit"]["mode"] == "binary" for item in table.items)


def test_each_table_has_direct_adapted_provenance_and_labeled_supplements() -> None:
    source = load_json(REGISTRY_PATH)
    relations: Counter[str] = Counter()

    for table in source["rubrics"]:
        domain = table["domain"]
        stage = table["stage"]
        source_contract = DOMAIN_SOURCES[domain]
        expected_adapted = source_contract["adapted_counts"][STAGES.index(stage)]
        origins = Counter(item["provenance"]["origin"] for item in table["items"])
        expected_origins = Counter(
            {
                origin: count
                for origin, count in {
                    "benchmark": expected_adapted,
                    "supplemental": 6 - expected_adapted,
                }.items()
                if count
            }
        )
        assert origins == expected_origins
        relations.update(origins)

        for item in table["items"]:
            assert item["atomic"] is True
            assert item["observable"] is True
            labels = set(item["labels"])
            provenance = item["provenance"]
            if provenance["origin"] == "benchmark":
                assert {"adapted", "benchmark-direct"} <= labels
                assert "supplemental" not in labels
                assert provenance["source_benchmark"] == source_contract["benchmark"]
                assert provenance["source_file"] in source_contract["stage_sources"][stage]
                assert provenance["source_revision"] == source_contract["revision"]
                assert provenance["domain"] == domain
                assert provenance["stage"] == stage
                assert provenance["extraction_status"] == "adapted"
            else:
                assert "adapted" not in labels
                assert {"supplemental", "verifier-control"} <= labels
                assert provenance["origin"] == "supplemental"
                assert provenance["supplemental_label"] in {"verifier-control", "stage-decomposition"}
                assert "upstream native benchmark metric" in provenance["rationale"]

    assert relations == {"benchmark": 132, "supplemental": 120}


def test_legacy_thirty_tables_are_semantically_byte_stable() -> None:
    source = load_json(REGISTRY_PATH)
    assert blake3_document(source["rubrics"][:30]) == LEGACY_TABLES_BLAKE3


def test_new_source_citations_are_exact_pinned_and_non_placeholder() -> None:
    source = load_json(REGISTRY_PATH)
    new_tables = [table for table in source["rubrics"] if table["domain"] in NEW_DOMAINS]
    assert len(new_tables) == 12

    for table in new_tables:
        domain = table["domain"]
        stage = table["stage"]
        pin = SOURCE_PINS[domain]
        for item in table["items"]:
            labels = set(item["labels"])
            provenance = item["provenance"]
            searchable = json.dumps(item, ensure_ascii=False, sort_keys=True).casefold()
            assert all(
                f'"{placeholder}"' not in searchable
                for placeholder in ("todo", "tbd", "fixme", "pending_review")
            )
            if provenance["origin"] == "benchmark":
                assert provenance["source_file"] in pin["file_blake3"]
                locators = [label for label in labels if label.startswith(pin["locator_prefix"])]
                assert len(locators) == 1
                assert locators[0] in pin["locator_labels"][provenance["source_file"]]
            else:
                rationale = provenance["rationale"]
                assert f"operator/benchmark-as-teacher-v2@{BAT_V2_REVISION}" in rationale
                assert "medical-research/contracts/post_sft_rubrics_v3.json#tracks." in rationale
                if stage != "E2E":
                    assert provenance["supplemental_label"] == "stage-decomposition"
                    assert "stage-decomposition" in labels

    hbp_stage_tables = [
        table
        for table in new_tables
        if table["domain"] == "healthbench-professional" and table["stage"] != "E2E"
    ]
    assert all(
        {item["provenance"]["origin"] for item in table["items"]} == {"supplemental"}
        for table in hbp_stage_tables
    )

    documentation = (ROOT / "docs" / "rubric-sources.md").read_text(encoding="utf-8")
    for domain, pin in SOURCE_PINS.items():
        assert DOMAIN_SOURCES[domain]["revision"] in documentation
        for relative_path, digest in pin["file_blake3"].items():
            assert relative_path in documentation
            assert digest in documentation
    assert BAT_CONTRACT_BLAKE3 in documentation


def test_judge_and_reward_share_each_new_compiled_table_identity() -> None:
    compiled = load_and_compile_registry(REGISTRY_PATH)
    sandbox_id = "6db0b9ee-7e6e-56f4-afce-6913920fc938"
    for domain, stage in product(NEW_DOMAINS, STAGES):
        table = compiled.resolve(domain, stage)
        binding = compiled.bind_sandbox(sandbox_id, domain, stage)
        assert binding.for_benchmark_judging() is table
        assert binding.for_rollout_reward() is table
        assert binding.for_benchmark_judging() is binding.for_rollout_reward()


def test_runtime_registry_has_no_private_payload_or_gold_fields() -> None:
    source = load_json(REGISTRY_PATH)
    forbidden_keys = {
        "answer_key",
        "correct_diagnosis",
        "gold",
        "gold_label",
        "gold_value",
        "judge_only_reference",
        "private_payload",
        "private_reference",
        "raw_private_result",
        "scorer_reference",
    }
    assert forbidden_keys.isdisjoint(nested_keys(source))

    serialized = json.dumps(source, ensure_ascii=False, sort_keys=True).casefold()
    assert HF_REPOSITORY.casefold() not in serialized
    assert HF_REVISION not in serialized
    assert "-----begin " not in serialized
    assert "bearer " not in serialized
    assert "token=" not in serialized
    assert "https://huggingface.co/" not in serialized


def test_catalog_records_only_nonsecret_hf_lineage() -> None:
    catalog = load_json(CATALOG_PATH)
    assert len(catalog["private_external_provenance"]) == 1
    lineage = catalog["private_external_provenance"][0]
    assert lineage["repository_type"] == "huggingface_dataset"
    assert lineage["repository"] == HF_REPOSITORY
    assert lineage["revision_type"] == "git_commit"
    assert lineage["revision"] == HF_REVISION
    assert lineage["provenance_use_only"] is True
    assert lineage["payload_accessed_for_this_inventory"] is False
    assert lineage["payload_copied_into_git"] is False
    assert lineage["authenticated_url_recorded"] is False
    assert "://" not in json.dumps(lineage)
    assert not ({"token", "credential", "authenticated_url", "payload"} & nested_keys(lineage))


def test_catalog_commits_exact_v2_inventory_and_public_source_pins() -> None:
    catalog = load_json(CATALOG_PATH)
    assert catalog["schema"] == "eva.rubric-source-catalog.v1"
    assert catalog["source_inventory_version"] == 2

    table_inventory = next(
        entry
        for entry in catalog["inventory_files"]
        if entry["path"] == "rubrics/source/domain-stage-tables.v1.json"
    )
    assert table_inventory == {
        "path": "rubrics/source/domain-stage-tables.v1.json",
        "purpose": "Canonical eva.rubric-registry.v1 source compiled directly for per-sandbox domain/stage bindings.",
        "domain_count": 7,
        "stage_count_per_domain": 6,
        "table_count": 42,
        "item_count": 252,
        "items_per_table": 6,
        "adapted_item_count": 132,
        "supplemental_item_count": 120,
        "compiler_schema": "eva.rubric-registry.v1",
        "compiled_registry_blake3": "blake3:f1b0e035a277a7b237f3ed804c3c2057fcca6165dfc3aeb2961ec6022f03aafc",
    }
    assert table_inventory["compiled_registry_blake3"] == load_and_compile_registry(
        REGISTRY_PATH
    ).digest
    assert {entry["domain"] for entry in catalog["domains"]} == set(DOMAIN_SOURCES)

    scopes = {entry["repository"]: entry for entry in catalog["additional_audit_scopes"]}
    direct_sources = {
        (entry.get("source_repository"), entry["source_relative_path"]): entry
        for entry in catalog["audited_sources"]
    }
    for domain, pin in SOURCE_PINS.items():
        repository = pin["repository"]
        scope = scopes[repository]
        assert scope["commit"] == DOMAIN_SOURCES[domain]["revision"]
        scope_files = {entry["path"]: entry for entry in scope["files"]}
        for relative_path, digest in pin["file_blake3"].items():
            assert scope_files[relative_path]["literal_file_blake3"] == digest
            direct = direct_sources[(repository, relative_path)]
            assert direct["source_commit"] == DOMAIN_SOURCES[domain]["revision"]
            assert direct["literal_file_blake3"] == digest
            assert direct["directly_used_in_inventory"] is True

    hbp_scope = scopes["openai/healthbench-professional"]
    assert hbp_scope["private_payload_accessed_for_this_inventory"] is False
    serialized = json.dumps(catalog, ensure_ascii=False, sort_keys=True).casefold()
    assert all(
        f'"{placeholder}"' not in serialized
        for placeholder in ("todo", "tbd", "fixme", "pending_review")
    )


def test_native_rule_audit_is_atomic_labeled_and_bat_v2_pinned() -> None:
    native = load_json(NATIVE_RULES_PATH)
    rules = [
        rule
        for domain in native["domains"]
        for rule in domain["native_outcome_rules"]
    ]
    assert len(native["domains"]) == 5
    assert len(rules) == 39
    assert Counter(rule["source_relation"] for rule in rules) == {
        "benchmark_derived": 26,
        "supplemental": 13,
    }
    assert all(rule["atomic"] is True and rule["observable"] is True for rule in rules)
    assert all(
        source["source_revision"] == BAT_V2_REVISION
        for rule in rules
        for source in rule["sources"]
    )
