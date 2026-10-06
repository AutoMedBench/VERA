"""Secret-safe custom-provider routes for the Codex Responses transport.

This compatibility layer does not import, translate, or mutate any EvaMed
tool, skill, rubric, policy, or evidence schema. It only binds exact model
identifiers and an existing API endpoint to Codex custom-provider settings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from eva_agent.pipeline.digests import blake3_hex


class CodexProviderConfigurationError(ValueError):
    """A provider route is unsafe, missing, or ambiguous."""


MODEL_ENV_NAMES = frozenset(
    {
        "MODEL_GPT_6_ASTRA",
        "MODEL_GPT_6_ASTRA_OPENAI",
        "MODEL_GPT_6_ASTRA_AZURE",
        "MODEL_GPT_5_6_SOL",
        "MODEL_GPT_5_6_TERRA",
        "MODEL_GPT_5_6_LUNA",
        "MODEL_OPUS_5",
        "MODEL_OPUS_4_8",
        "MODEL_GEMINI_3_1_PRO",
        "MODEL_GEMINI_3_5_FLASH",
        "MODEL_GEMINI_3_8_FLASH",
        "MODEL_GLM_5_1",
        "MODEL_GLM_5_2",
        "MODEL_GLM_5_3",
        "MODEL_GLM_5_3_FLASH",
        "MODEL_QWEN_3_5_0_8B",
        "MODEL_QWEN_3_5_9B",
        "MODEL_QWEN_3_5_35B_A3B",
        "MODEL_QWEN_3_5_122B_A10B",
        "MODEL_QWEN_3_5_397B_A17B",
        "MODEL_QWEN_3_6_27B",
        "MODEL_DEEPSEEK_V4_FLASH",
        "MODEL_DEEPSEEK_V4_PRO",
    }
)
CREDENTIAL_ENV_NAMES = frozenset(
    {
        "EVA_CODEX_OPUS5_ADAPTER_TOKEN",
        "NVIDIA_INFERENCE_API_KEY",
        "NVIDIA_API_KEY",
        "NVDA_API_KEY",
        "NVIDIA_API_KEY_CAN",
        "NVIDIA_API_KEY_DAGUANG",
        "NVIDIA_API_KEY_YUFAN",
        "OPENAI_API_KEY",
        "AUTOMEDBENCH_API_KEY",
        "AUTOMEDBENCH_JUDGE_API_KEY",
    }
)
ENDPOINT_ENV_NAMES = frozenset(
    {
        "NVIDIA_INFERENCE_BASE_URL",
        "NVIDIA_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "AUTOMEDBENCH_AGENT_BASE_URL",
        "AUTOMEDBENCH_JUDGE_BASE_URL",
    }
)
ALLOWED_ENV_NAMES = MODEL_ENV_NAMES | CREDENTIAL_ENV_NAMES | ENDPOINT_ENV_NAMES

_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,255}$")
_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_PROVIDER_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class RouteDefinition:
    route_id: str
    model_env_name: str
    registry_role_id: str | None
    provider_family: str


ROUTE_DEFINITIONS: Mapping[str, RouteDefinition] = MappingProxyType(
    {
        row.route_id: row
        for row in (
            RouteDefinition("gpt_6_astra", "MODEL_GPT_6_ASTRA", None, "openai"),
            RouteDefinition("gpt_6_astra_openai", "MODEL_GPT_6_ASTRA_OPENAI", None, "openai"),
            RouteDefinition("gpt_6_astra_azure", "MODEL_GPT_6_ASTRA_AZURE", None, "openai"),
            RouteDefinition(
                "gpt_5_6_sol",
                "MODEL_GPT_5_6_SOL",
                "cascade_gpt_5_6_sol",
                "openai",
            ),
            RouteDefinition("gpt_5_6_terra", "MODEL_GPT_5_6_TERRA", None, "openai"),
            RouteDefinition("gpt_5_6_luna", "MODEL_GPT_5_6_LUNA", None, "openai"),
            RouteDefinition("opus_5", "MODEL_OPUS_5", "architect_opus_5", "anthropic"),
            RouteDefinition(
                "opus_4_8", "MODEL_OPUS_4_8", "critic_opus_4_8", "anthropic"
            ),
            RouteDefinition("gemini_3_1_pro", "MODEL_GEMINI_3_1_PRO", None, "google"),
            RouteDefinition(
                "gemini_3_5_flash", "MODEL_GEMINI_3_5_FLASH", None, "google"
            ),
            RouteDefinition(
                "gemini_3_8_flash", "MODEL_GEMINI_3_8_FLASH", None, "google"
            ),
            RouteDefinition("glm_5_1", "MODEL_GLM_5_1", None, "glm"),
            RouteDefinition(
                "glm_5_2", "MODEL_GLM_5_2", "cascade_glm_5_2", "glm"
            ),
            RouteDefinition("glm_5_3", "MODEL_GLM_5_3", None, "glm"),
            RouteDefinition(
                "glm_5_3_flash", "MODEL_GLM_5_3_FLASH", None, "glm"
            ),
            RouteDefinition(
                "qwen_3_5_0_8b",
                "MODEL_QWEN_3_5_0_8B",
                None,
                "qwen",
            ),
            RouteDefinition(
                "qwen_3_5_9b",
                "MODEL_QWEN_3_5_9B",
                None,
                "qwen",
            ),
            RouteDefinition(
                "qwen_3_5_35b_a3b",
                "MODEL_QWEN_3_5_35B_A3B",
                None,
                "qwen",
            ),
            RouteDefinition(
                "qwen_3_5_122b_a10b",
                "MODEL_QWEN_3_5_122B_A10B",
                None,
                "qwen",
            ),
            RouteDefinition(
                "qwen_3_5_397b_a17b",
                "MODEL_QWEN_3_5_397B_A17B",
                "teacher_qwen_3_5_397b_a17b",
                "qwen",
            ),
            RouteDefinition(
                "qwen_3_6_27b",
                "MODEL_QWEN_3_6_27B",
                "cascade_qwen3_6_27b",
                "qwen",
            ),
            RouteDefinition(
                "deepseek_v4_flash",
                "MODEL_DEEPSEEK_V4_FLASH",
                "cascade_deepseek_v4_flash",
                "deepseek",
            ),
            RouteDefinition("deepseek_v4_pro", "MODEL_DEEPSEEK_V4_PRO", None, "deepseek"),
        )
    }
)

DEFAULT_CREDENTIAL_PRIORITY = (
    "NVIDIA_INFERENCE_API_KEY",
    "NVIDIA_API_KEY",
    "NVDA_API_KEY",
    "OPENAI_API_KEY",
    "AUTOMEDBENCH_API_KEY",
)
DEFAULT_ENDPOINT_PRIORITY = (
    "NVIDIA_INFERENCE_BASE_URL",
    "NVIDIA_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "AUTOMEDBENCH_AGENT_BASE_URL",
)


class _PrivateText:
    """A runtime value whose normal text and representation are redacted."""

    __slots__ = ("__value",)

    def __init__(self, value: str) -> None:
        self.__value = value

    def __repr__(self) -> str:
        return "<redacted>"

    def __str__(self) -> str:
        return "<redacted>"

    def reveal_at_process_boundary(self) -> str:
        return self.__value


class AllowlistedEnvironment:
    """Resolved dotenv material exposing only explicitly allowed names."""

    __slots__ = ("__values",)

    def __init__(self, values: Mapping[str, str]) -> None:
        if any(name not in ALLOWED_ENV_NAMES for name in values):
            raise CodexProviderConfigurationError("environment contains a non-allowlisted name")
        self.__values = dict(values)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.__values))

    def get(self, name: str) -> str | None:
        if name not in ALLOWED_ENV_NAMES:
            raise CodexProviderConfigurationError("environment name is not allowlisted")
        return self.__values.get(name)

    def __repr__(self) -> str:
        return f"AllowlistedEnvironment(names={self.names!r}, values=<redacted>)"


@dataclass(frozen=True, repr=False)
class CodexConfigOverrides:
    """Redaction-aware holder for Codex CLI configuration overrides."""

    provider_id: str
    model_id: str
    credential_env_name: str
    endpoint_env_name: str
    _credential: _PrivateText
    _endpoint: _PrivateText

    def __post_init__(self) -> None:
        if _PROVIDER_ID.fullmatch(self.provider_id) is None:
            raise CodexProviderConfigurationError("Codex provider ID is invalid")
        if _MODEL_ID.fullmatch(self.model_id) is None:
            raise CodexProviderConfigurationError("model ID is invalid")
        if self.credential_env_name not in CREDENTIAL_ENV_NAMES:
            raise CodexProviderConfigurationError("credential environment name is not allowlisted")
        if self.endpoint_env_name not in ENDPOINT_ENV_NAMES:
            raise CodexProviderConfigurationError("endpoint environment name is not allowlisted")

    def __repr__(self) -> str:
        return (
            "CodexConfigOverrides("
            f"provider_id={self.provider_id!r}, model_id={self.model_id!r}, "
            f"credential_env_name={self.credential_env_name!r}, "
            f"endpoint_env_name={self.endpoint_env_name!r}, "
            "credential=<redacted>, endpoint=<redacted>, wire_api='responses', "
            "request_max_retries=0, stream_max_retries=0)"
        )

    @property
    def safe_projection(self) -> Mapping[str, Any]:
        """Return receipt-safe metadata; no private value is projected."""

        return MappingProxyType(
            {
                "schema": "eva.codex-provider-config-sidecar.v1",
                "provider_id": self.provider_id,
                "model_id": self.model_id,
                "credential_env_name": self.credential_env_name,
                "endpoint_env_name": self.endpoint_env_name,
                "credential_value_recorded": False,
                "endpoint_value_recorded": False,
                "requires_openai_auth": False,
                "wire_api": "responses",
                "request_max_retries": 0,
                "stream_max_retries": 0,
            }
        )

    @property
    def safe_blake3(self) -> str:
        return blake3_hex(self.safe_projection)

    @staticmethod
    def _toml_text(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    def for_subprocess(self) -> tuple[tuple[str, ...], Mapping[str, str], tuple[str, ...]]:
        """Reveal private values only at the subprocess boundary; never log this."""

        endpoint = self._endpoint.reveal_at_process_boundary()
        credential = self._credential.reveal_at_process_boundary()
        prefix = f"model_providers.{self.provider_id}"
        overrides = (
            f"model_provider={self._toml_text(self.provider_id)}",
            f"{prefix}.name={self._toml_text('EVA Responses Gateway')}",
            f"{prefix}.base_url={self._toml_text(endpoint)}",
            f"{prefix}.env_key={self._toml_text(self.credential_env_name)}",
            f"{prefix}.requires_openai_auth=false",
            f"{prefix}.wire_api=\"responses\"",
            f"{prefix}.request_max_retries=0",
            f"{prefix}.stream_max_retries=0",
        )
        child_secret_env = MappingProxyType({self.credential_env_name: credential})
        return overrides, child_secret_env, (endpoint, credential)


@dataclass(frozen=True, repr=False)
class CodexProviderRoute:
    route_id: str
    model_id: str
    model_env_name: str
    registry_role_id: str | None
    provider_family: str
    config: CodexConfigOverrides

    def __repr__(self) -> str:
        return (
            "CodexProviderRoute("
            f"route_id={self.route_id!r}, model_id={self.model_id!r}, "
            f"model_env_name={self.model_env_name!r}, "
            f"registry_role_id={self.registry_role_id!r}, "
            f"provider_family={self.provider_family!r}, config={self.config!r})"
        )

    @property
    def safe_metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "route_id": self.route_id,
                "model_id": self.model_id,
                "model_env_name": self.model_env_name,
                "registry_role_id": self.registry_role_id,
                "provider_family": self.provider_family,
                "config_blake3": self.config.safe_blake3,
            }
        )


def _parse_dotenv(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError) as exc:
        raise CodexProviderConfigurationError("environment file is unreadable") from exc
    values: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name not in ALLOWED_ENV_NAMES:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if "\x00" in value or "\n" in value or "\r" in value or "`" in value:
            raise CodexProviderConfigurationError(
                f"environment value for {name} uses unsupported syntax"
            )
        values[name] = value
    return values


def load_allowlisted_environment(
    env_files: Sequence[str | Path],
    *,
    environment: Mapping[str, str] | None = None,
) -> AllowlistedEnvironment:
    """Parse, but never execute, allowlisted assignments from dotenv files."""

    raw: dict[str, str] = {}
    for path in env_files:
        raw.update({name: value for name, value in _parse_dotenv(Path(path)).items() if value})
    source = os.environ if environment is None else environment
    raw.update(
        {
            name: value
            for name, value in source.items()
            if name in ALLOWED_ENV_NAMES and isinstance(value, str) and value
        }
    )

    def resolve(name: str, stack: tuple[str, ...]) -> str | None:
        if name in stack:
            raise CodexProviderConfigurationError(f"environment reference cycle at {name}")
        value = raw.get(name)
        if not value:
            return None
        reference = _ENV_REFERENCE.fullmatch(value)
        if reference is not None:
            target = reference.group(1)
            if target not in ALLOWED_ENV_NAMES:
                raise CodexProviderConfigurationError(
                    f"environment reference from {name} is not allowlisted"
                )
            return resolve(target, (*stack, name))
        if "$" in value:
            raise CodexProviderConfigurationError(
                f"environment value for {name} uses unsupported expansion"
            )
        return value

    resolved = {
        name: value
        for name in sorted(ALLOWED_ENV_NAMES)
        if (value := resolve(name, ()))
    }
    return AllowlistedEnvironment(resolved)


def _safe_endpoint(value: str) -> str:
    endpoint = value.strip().rstrip("/")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise CodexProviderConfigurationError(
            "provider endpoint must be a credential-free HTTPS API base URL"
        )
    if parsed.path.rstrip("/").endswith(("/chat/completions", "/responses")):
        raise CodexProviderConfigurationError(
            "provider endpoint must be an API base, not a resource URL"
        )
    return endpoint


def _first(
    names: Sequence[str], environment: AllowlistedEnvironment, *, label: str
) -> tuple[str, str]:
    for name in names:
        value = environment.get(name)
        if value:
            return name, value
    raise CodexProviderConfigurationError(
        f"{label} is missing; configure one of: {', '.join(names)}"
    )


def _registry_models(path: str | Path | None) -> Mapping[str, str]:
    if path is None:
        return MappingProxyType({})
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodexProviderConfigurationError("model registry is missing or invalid") from exc
    if not isinstance(document, dict) or document.get("schema") != "rlevo.med-research-model-registry.v1":
        raise CodexProviderConfigurationError("model registry schema differs")
    models = document.get("models")
    if not isinstance(models, list):
        raise CodexProviderConfigurationError("model registry entries differ")
    by_role: dict[str, str] = {}
    for value in models:
        if not isinstance(value, dict):
            raise CodexProviderConfigurationError("model registry entry differs")
        role = value.get("role_id")
        model_id = value.get("api_model_id")
        if not isinstance(role, str) or not isinstance(model_id, str):
            raise CodexProviderConfigurationError("model registry identity differs")
        if role in by_role:
            raise CodexProviderConfigurationError("model registry role is duplicated")
        if _MODEL_ID.fullmatch(model_id) is None:
            raise CodexProviderConfigurationError("model registry model ID is invalid")
        by_role[role] = model_id
    return MappingProxyType(by_role)


def load_codex_provider_routes(
    *,
    env_files: Sequence[str | Path],
    registry_path: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    route_ids: Sequence[str] | None = None,
    credential_priority: Sequence[str] = DEFAULT_CREDENTIAL_PRIORITY,
    endpoint_priority: Sequence[str] = DEFAULT_ENDPOINT_PRIORITY,
) -> Mapping[str, CodexProviderRoute]:
    """Resolve exact model IDs and a shared secret-safe Responses transport."""

    requested = tuple(ROUTE_DEFINITIONS) if route_ids is None else tuple(route_ids)
    if len(set(requested)) != len(requested):
        raise CodexProviderConfigurationError("route request is duplicated")
    if any(route_id not in ROUTE_DEFINITIONS for route_id in requested):
        raise CodexProviderConfigurationError("route request contains an unknown route")
    if any(name not in CREDENTIAL_ENV_NAMES for name in credential_priority):
        raise CodexProviderConfigurationError("credential priority is not allowlisted")
    if any(name not in ENDPOINT_ENV_NAMES for name in endpoint_priority):
        raise CodexProviderConfigurationError("endpoint priority is not allowlisted")
    resolved = load_allowlisted_environment(env_files, environment=environment)
    registry = _registry_models(registry_path)
    model_ids: dict[str, str] = {}
    for route_id in requested:
        definition = ROUTE_DEFINITIONS[route_id]
        model_id = resolved.get(definition.model_env_name)
        if model_id is None and definition.registry_role_id is not None:
            model_id = registry.get(definition.registry_role_id)
        if model_id is None:
            continue
        if _MODEL_ID.fullmatch(model_id) is None:
            raise CodexProviderConfigurationError(f"model ID for {route_id} is invalid")
        model_ids[route_id] = model_id
    if not model_ids:
        return MappingProxyType({})
    credential_name, credential = _first(
        credential_priority, resolved, label="provider credential"
    )
    endpoint_name, endpoint_raw = _first(
        endpoint_priority, resolved, label="provider endpoint"
    )
    endpoint = _safe_endpoint(endpoint_raw)
    routes: dict[str, CodexProviderRoute] = {}
    for route_id in requested:
        model_id = model_ids.get(route_id)
        if model_id is None:
            continue
        definition = ROUTE_DEFINITIONS[route_id]
        config = CodexConfigOverrides(
            provider_id=f"eva_{route_id}",
            model_id=model_id,
            credential_env_name=credential_name,
            endpoint_env_name=endpoint_name,
            _credential=_PrivateText(credential),
            _endpoint=_PrivateText(endpoint),
        )
        routes[route_id] = CodexProviderRoute(
            route_id=route_id,
            model_id=model_id,
            model_env_name=definition.model_env_name,
            registry_role_id=definition.registry_role_id,
            provider_family=definition.provider_family,
            config=config,
        )
    return MappingProxyType(routes)


__all__ = [
    "ALLOWED_ENV_NAMES",
    "CREDENTIAL_ENV_NAMES",
    "DEFAULT_CREDENTIAL_PRIORITY",
    "DEFAULT_ENDPOINT_PRIORITY",
    "ENDPOINT_ENV_NAMES",
    "MODEL_ENV_NAMES",
    "ROUTE_DEFINITIONS",
    "AllowlistedEnvironment",
    "CodexConfigOverrides",
    "CodexProviderConfigurationError",
    "CodexProviderRoute",
    "RouteDefinition",
    "load_allowlisted_environment",
    "load_codex_provider_routes",
]
