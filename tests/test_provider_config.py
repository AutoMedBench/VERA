from __future__ import annotations

import json
from pathlib import Path

import pytest

from eva_agent.pipeline import Cohort
from eva_agent.providers import ProviderConfigurationError, load_provider_plan


ROOT = Path(__file__).resolve().parents[1]
SIBLING_REGISTRY = ROOT.parent / "rlevo-med-research/config/model-registry.20260903-v2.json"


def _configuration() -> dict:
    return {
        "schema": "eva.provider-model-routing.v1",
        "protocol": "openai-compatible-chat-completions",
        "providers": {
            "gateway": {
                "max_concurrency": 8,
                "requests_per_minute": 600,
                "burst": 8,
                "queue_timeout_seconds": 5,
            }
        },
        "cohorts": {
            "weak": {
                "provider": "gateway",
                "model_env": "MODEL_DEEPSEEK_V4_FLASH",
                "registry_role": "cascade_deepseek_v4_flash",
            },
            "middle": {"provider": "gateway", "model_env": "MODEL_GEMINI_3_1_PRO"},
            "strong": {
                "provider": "gateway",
                "model_env": "MODEL_GPT_5_6_SOL",
                "registry_role": "cascade_gpt_5_6_sol",
            },
        },
        "judge": {
            "provider": "gateway",
            "model_env": "MODEL_OPUS_5",
            "registry_role": "architect_opus_5",
        },
        "auxiliary": {
            "critic": {
                "provider": "gateway",
                "model_env": "MODEL_OPUS_4_8",
                "registry_role": "critic_opus_4_8",
            },
            "deepseek-pro": {
                "provider": "gateway",
                "model_env": "MODEL_DEEPSEEK_V4_PRO",
            },
            "qwen": {"provider": "gateway", "registry_provider_family": "qwen"},
        },
    }


def _environment() -> dict[str, str]:
    return {
        "MODEL_DEEPSEEK_V4_FLASH": "deepseek-v4-flash-test",
        "MODEL_GEMINI_3_1_PRO": "gemini-3.1-pro-test",
        "MODEL_GPT_5_6_SOL": "gpt-5.6-sol-test",
        "MODEL_OPUS_5": "claude-opus-5-test",
        "MODEL_OPUS_4_8": "claude-opus-4-8-test",
        "MODEL_DEEPSEEK_V4_PRO": "deepseek-v4-pro-test",
        "NVIDIA_INFERENCE_API_KEY": "credential-super-secret-12345",
        "NVIDIA_INFERENCE_BASE_URL": "https://private-gateway.invalid/v1",
    }


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_sibling_registry_routes_all_tiers_and_redacts_values(tmp_path: Path) -> None:
    config = _write(tmp_path / "routing.json", _configuration())
    environment = _environment()
    plan = load_provider_plan(
        config, registry_path=SIBLING_REGISTRY, environment=environment
    )

    assert [target.cohort for target in plan.targets] == [
        Cohort.WEAK,
        Cohort.MIDDLE,
        Cohort.STRONG,
    ]
    assert [target.model_id for target in plan.targets] == [
        "deepseek-v4-flash-test",
        "gemini-3.1-pro-test",
        "gpt-5.6-sol-test",
    ]
    assert plan.judge.model_id == "claude-opus-5-test"
    assert plan.auxiliary["critic"].model_id == "claude-opus-4-8-test"
    assert plan.auxiliary["deepseek-pro"].model_id == "deepseek-v4-pro-test"
    # Qwen is discovered from the signed sibling registry even without a
    # model-specific environment variable.
    assert plan.auxiliary["qwen"].registry_provider_family == "qwen"
    assert "qwen" in plan.auxiliary["qwen"].model_id.casefold()

    serialized = json.dumps(plan.safe_metadata(), sort_keys=True)
    represented = repr(plan) + repr(plan.transports["gateway"])
    for forbidden in (
        environment["NVIDIA_INFERENCE_API_KEY"],
        environment["NVIDIA_INFERENCE_BASE_URL"],
    ):
        assert forbidden not in serialized
        assert forbidden not in represented
    assert plan.safe_metadata()["parallel_tool_calls"] is True
    assert plan.safe_metadata()["semantic_retry_count"] == 0


def test_environment_file_references_are_literal_allowlisted_and_process_wins(
    tmp_path: Path,
) -> None:
    config = _write(tmp_path / "routing.json", _configuration())
    env_file = tmp_path / "keys.env"
    env_file.write_text(
        "\n".join(
            [
                "NVIDIA_API_KEY_CAN=file-secret-value",
                "NVIDIA_INFERENCE_API_KEY=${NVIDIA_API_KEY_CAN}",
                "NVIDIA_INFERENCE_BASE_URL=https://file.invalid/v1",
                "MODEL_DEEPSEEK_V4_FLASH=deepseek-file",
                "MODEL_GEMINI_3_1_PRO=gemini-file",
                "MODEL_GPT_5_6_SOL=gpt-file",
                "MODEL_OPUS_5=claude-opus-5-file",
                "MODEL_OPUS_4_8=claude-opus-4-8-file",
                "MODEL_DEEPSEEK_V4_PRO=deepseek-pro-file",
                "UNRELATED_PRIVATE_TOKEN=must-not-be-read",
            ]
        ),
        encoding="utf-8",
    )
    environment = {
        "MODEL_GEMINI_3_1_PRO": "gemini-process-wins",
    }
    plan = load_provider_plan(
        config,
        registry_path=SIBLING_REGISTRY,
        environment=environment,
        env_files=[env_file],
    )
    assert plan.cohorts[Cohort.MIDDLE].model_id == "gemini-process-wins"
    assert plan.transports["gateway"].credential_env_name == "NVIDIA_INFERENCE_API_KEY"
    serialized = json.dumps(plan.safe_metadata(), sort_keys=True)
    assert "file-secret-value" not in serialized
    assert "https://file.invalid/v1" not in serialized
    assert "must-not-be-read" not in serialized


def test_missing_secret_fails_without_echoing_other_values(tmp_path: Path) -> None:
    config = _write(tmp_path / "routing.json", _configuration())
    environment = _environment()
    secret = environment.pop("NVIDIA_INFERENCE_API_KEY")
    with pytest.raises(ProviderConfigurationError) as raised:
        load_provider_plan(config, registry_path=SIBLING_REGISTRY, environment=environment)
    message = str(raised.value)
    assert "credential" in message
    assert secret not in message
    assert environment["NVIDIA_INFERENCE_BASE_URL"] not in message


def test_unapproved_environment_name_and_duplicate_primary_models_fail_closed(
    tmp_path: Path,
) -> None:
    raw = _configuration()
    raw["cohorts"]["middle"]["model_env"] = "AWS_SECRET_ACCESS_KEY"
    config = _write(tmp_path / "routing.json", raw)
    with pytest.raises(ProviderConfigurationError, match="not allowlisted"):
        load_provider_plan(config, registry_path=SIBLING_REGISTRY, environment=_environment())

    raw = _configuration()
    config = _write(tmp_path / "routing-duplicate.json", raw)
    environment = _environment()
    environment["MODEL_GEMINI_3_1_PRO"] = environment["MODEL_DEEPSEEK_V4_FLASH"]
    with pytest.raises(ProviderConfigurationError, match="distinct"):
        load_provider_plan(config, registry_path=SIBLING_REGISTRY, environment=environment)


def test_checked_in_example_resolves_against_sibling_registry_without_network() -> None:
    plan = load_provider_plan(
        ROOT / "config/model-tiers.example.json",
        registry_path=SIBLING_REGISTRY,
        environment=_environment(),
    )
    assert len(plan.targets) == 3
    assert set(plan.auxiliary) == {"critic-opus-4-8", "deepseek-v4-pro", "qwen"}
