from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_provider_usage", ROOT / "scripts/audit_provider_usage.py"
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _registry(path: Path) -> Path:
    return _write(
        path,
        {
            "schema": "rlevo.med-research-model-registry.v1",
            "transport": {
                "protocol": "openai-compatible-chat-completions",
                "credential_env_priority": ["PRIVATE_API_KEY"],
                "endpoint_env_priority": ["PRIVATE_BASE_URL"],
            },
            "models": [
                {
                    "role_id": "architect_opus_5",
                    "api_model_id": "aws/anthropic/bedrock-claude-opus-5",
                    "provider_family": "anthropic",
                },
                {
                    "role_id": "cascade_gpt_5_6_sol",
                    "api_model_id": "azure/openai/gpt-5.6-sol",
                    "provider_family": "openai",
                },
            ],
        },
    )


def _fixture_tree(root: Path) -> None:
    _write(
        root / "construction/candidate-construction-manifest.json",
        {
            "schema": "rlevo.med-research-candidate-construction-manifest.v2",
            "construction_id": "construction-1",
            "model_calls": [
                {
                    "ordinal": 0,
                    "role_id": "architect_opus_5",
                    "expected_model_id": "aws/anthropic/bedrock-claude-opus-5",
                    "returned_model_id": "aws/anthropic/bedrock-claude-opus-5",
                    "request_sha256": "a" * 64,
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                        "prompt_tokens_details": {"cached_tokens": 40},
                    },
                },
                {
                    "ordinal": 2,
                    "stage": "revision",
                    "role_id": "architect_opus_5",
                    "expected_model_id": "aws/anthropic/bedrock-claude-opus-5",
                    "returned_model_id": None,
                    "request_sha256": "b" * 64,
                },
            ],
        },
    )
    _write(
        root / "construction/construction-failure.json",
        {
            "schema": "rlevo.med-research-candidate-construction-failure.v2",
            "construction_id": "construction-1",
            "stage": "revision",
            "provider_call_completed": True,
        },
    )
    rollout = {
        "schema": "rlevo.med-research-agent-rollout-receipt.v3",
        "rollout_id": "rollout-1",
        "model_id": "azure/openai/gpt-5.6-sol",
        "model_role": "cascade_gpt_5_6_sol",
        "transcript": {"raw_response": "must-never-cache-this"},
        "turns": [
            {
                "turn": 0,
                "request_sha256": "c" * 64,
                "returned_model_id": "azure/openai/gpt-5.6-sol",
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 10,
                    "total_tokens": 90,
                },
            },
            {
                "turn": 1,
                "request_sha256": "d" * 64,
                "returned_model_id": "azure/openai/gpt-5.6-sol",
            },
        ],
    }
    _write(root / "panel-a/rollout-receipt.json", rollout)
    _write(root / "copied-panel/rollout-receipt.json", rollout)
    health = {
        "schema": "rlevo.med-research-model-health-receipt.v1",
        "created_at_utc": "2026-09-07T00:00:00Z",
        "registry_sha256": "e" * 64,
        "models": [
            {
                "role_id": "architect_opus_5",
                "expected_model_id": "aws/anthropic/bedrock-claude-opus-5",
                "returned_model_id": "aws/anthropic/bedrock-claude-opus-5",
                "usage": {
                    "input_tokens": 30,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 10,
                },
            }
        ],
    }
    _write(root / "receipts/model-health-1.json", health)
    _write(root / "copied/model-health.json", health)
    _write(
        root / "capacity/terminal.json",
        {
            "schema": "evamed.provider-capacity-one-use-sample-terminal.v1",
            "provider_call_count": 1,
            "expected_model_id": "azure/openai/gpt-5.6-sol",
            "returned_model_id": None,
            "call": {
                "call_id": "capacity-1",
                "attempt_ordinal": 0,
                "role_id": "cascade_gpt_5_6_sol",
            },
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        },
    )
    _write(
        root / "judge/blinded-judge-host-receipt.json",
        {
            "schema": "rlevo.med-research-signed-host-receipt.v1",
            "payload": {
                "sandbox_id": "sandbox-1",
                "observations": {
                    "request_sha256": "f" * 64,
                    "expected_model_id": "aws/anthropic/bedrock-claude-opus-5",
                    "returned_model_id": "aws/anthropic/bedrock-claude-opus-5",
                },
            },
        },
    )
    # The scanner never opens generic raw response/request filenames.
    _write(
        root / "raw/provider-response.json",
        {
            "model_id": "aws/anthropic/bedrock-claude-opus-5",
            "credential": "must-never-appear",
            "usage": {"prompt_tokens": 999999},
        },
    )


def test_usage_shapes_normalize_without_payload_material() -> None:
    openai = AUDIT.normalize_usage(
        {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 2},
        }
    )
    anthropic = AUDIT.normalize_usage(
        {
            "input_tokens": 20,
            "output_tokens": 5,
            "cache_read_input_tokens": 6,
            "cache_creation_input_tokens": 2,
        }
    )
    gemini = AUDIT.normalize_usage(
        {
            "prompt_token_count": 30,
            "candidates_token_count": 7,
            "total_token_count": 37,
            "cached_content_token_count": 8,
        }
    )
    assert openai.as_dict() == {
        "input_tokens": 12,
        "output_tokens": 3,
        "cache_read_tokens": 4,
        "cache_write_tokens": None,
        "reasoning_tokens": 2,
        "total_tokens": 15,
    }
    assert anthropic.input_tokens == 20 and anthropic.cache_write_tokens == 2
    assert gemini.output_tokens == 7 and gemini.cache_read_tokens == 8


def test_audit_deduplicates_copies_and_marks_lower_bounds(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _fixture_tree(root)
    registry = _registry(tmp_path / "model-registry.json")
    report = AUDIT.build_report([("fixture", root)], registry_paths=[registry])

    opus = report["models"]["opus-5"]
    assert opus["provable_request_count"] == 4
    assert opus["requests_with_usage"] == 2
    assert opus["missing_usage_request_count"] == 2
    assert opus["unproven_dispatch_like_record_count"] == 0
    assert opus["usage_token_lower_bound"] == {
        "input_tokens": 130,
        "output_tokens": 25,
        "cache_read_tokens": 50,
        "cache_write_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": 120,
    }
    sol = report["models"]["gpt-5.6-sol"]
    assert sol["provable_request_count"] == 3
    assert sol["requests_with_usage"] == 2
    assert sol["missing_usage_request_count"] == 1
    assert sol["usage_token_lower_bound"]["input_tokens"] == 87
    assert report["deduplication"]["conflicting_duplicate_records"] == 0
    assert report["by_root"]["fixture"]["models"]["opus-5"] == opus
    encoded = json.dumps(report, sort_keys=True)
    assert "must-never-appear" not in encoded
    assert "must-never-cache-this" not in encoded
    assert report["scope"]["raw_prompts_or_responses_recorded"] is False
    assert report["cost"]["actual_usd"] is None
    assert report["registry_routes"]["opus-5"][0]["credential_value_recorded"] is False


def test_sanitized_cache_makes_rerun_incremental(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _fixture_tree(root)
    cache = tmp_path / "cache/provider-usage.json"
    first = AUDIT.build_report([("fixture", root)], cache_path=cache)
    second = AUDIT.build_report([("fixture", root)], cache_path=cache)
    assert first["scan"]["files_parsed"] > 0
    assert second["scan"]["files_parsed"] == 0
    assert second["scan"]["cache_hits"] == second["scan"]["files_considered"]
    assert first["models"] == second["models"]
    assert first["evidence_blake3"] == second["evidence_blake3"]
    cached = cache.read_text(encoding="utf-8")
    assert "must-never-appear" not in cached
    assert "must-never-cache-this" not in cached
    assert cache.stat().st_mode & 0o777 == 0o600
