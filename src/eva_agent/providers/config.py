"""Strict, secret-safe provider and model routing configuration.

Configuration files contain environment *names*. Secret and endpoint values
are only materialized in :class:`ResolvedTransport`, whose representation and
safe metadata deliberately omit them.
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

from eva_agent.pipeline import Cohort, ModelTarget


class ProviderConfigurationError(ValueError):
    """Provider configuration failed closed before a network call."""


_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,255}$")

MODEL_ENV_NAMES = frozenset(
    {
        "MODEL_OPUS_5",
        "MODEL_OPUS_4_8",
        "MODEL_GPT_5_6_SOL",
        "MODEL_GPT_5_6_TERRA",
        "MODEL_GPT_5_6_LUNA",
        "MODEL_GEMINI_3_1_PRO",
        "MODEL_GEMINI_3_5_FLASH",
        "MODEL_DEEPSEEK_V4_FLASH",
        "MODEL_DEEPSEEK_V4_PRO",
    }
)

# A registry cannot turn this loader into an arbitrary environment-variable
# reader. These names cover the current EvaMed gateway and compatible routes.
CREDENTIAL_ENV_NAMES = frozenset(
    {
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


class _SecretText:
    """A runtime-only string that cannot be exposed accidentally by repr/str."""

    __slots__ = ("__value",)

    def __init__(self, value: str) -> None:
        self.__value = value

    def __repr__(self) -> str:
        return "<redacted>"

    def __str__(self) -> str:
        return "<redacted>"

    def for_client(self) -> str:
        """Reveal only at the final SDK construction boundary."""

        return self.__value


@dataclass(frozen=True)
class ProviderLimits:
    max_concurrency: int
    requests_per_minute: int
    burst: int
    queue_timeout_seconds: float

    def __post_init__(self) -> None:
        if type(self.max_concurrency) is not int or not 1 <= self.max_concurrency <= 4096:
            raise ProviderConfigurationError("provider max_concurrency must be in [1,4096]")
        if type(self.requests_per_minute) is not int or not 1 <= self.requests_per_minute <= 1_000_000:
            raise ProviderConfigurationError("provider requests_per_minute must be in [1,1000000]")
        if type(self.burst) is not int or not 1 <= self.burst <= self.max_concurrency:
            raise ProviderConfigurationError("provider burst must be in [1,max_concurrency]")
        if type(self.queue_timeout_seconds) not in {int, float} or not 0.1 <= self.queue_timeout_seconds <= 3600:
            raise ProviderConfigurationError("provider queue timeout must be in [0.1,3600]")


class ResolvedTransport:
    """Resolved SDK material with an intentionally redacted representation."""

    __slots__ = (
        "provider_id",
        "credential_env_name",
        "endpoint_env_name",
        "limits",
        "_credential",
        "_endpoint",
    )

    def __init__(
        self,
        *,
        provider_id: str,
        credential_env_name: str,
        endpoint_env_name: str,
        credential: str,
        endpoint: str,
        limits: ProviderLimits,
    ) -> None:
        self.provider_id = provider_id
        self.credential_env_name = credential_env_name
        self.endpoint_env_name = endpoint_env_name
        self.limits = limits
        self._credential = _SecretText(credential)
        self._endpoint = _SecretText(endpoint)

    def __repr__(self) -> str:
        return (
            "ResolvedTransport(provider_id="
            f"{self.provider_id!r}, credential=<redacted>, endpoint=<redacted>, "
            f"credential_env_name={self.credential_env_name!r}, "
            f"endpoint_env_name={self.endpoint_env_name!r}, limits={self.limits!r})"
        )

    def client_material(self) -> tuple[str, str]:
        """Return ``(credential, base_url)`` only for injected SDK factories."""

        return self._credential.for_client(), self._endpoint.for_client()

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "credential_env_name": self.credential_env_name,
            "endpoint_env_name": self.endpoint_env_name,
            "credential_value_recorded": False,
            "endpoint_value_recorded": False,
            "max_concurrency": self.limits.max_concurrency,
            "requests_per_minute": self.limits.requests_per_minute,
            "burst": self.limits.burst,
            "queue_timeout_seconds": self.limits.queue_timeout_seconds,
        }


@dataclass(frozen=True)
class ResolvedModelRoute:
    route_name: str
    provider_id: str
    model_id: str
    model_env_name: str | None
    registry_role_id: str | None
    registry_provider_family: str | None

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "route_name": self.route_name,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_env_name": self.model_env_name,
            "registry_role_id": self.registry_role_id,
            "registry_provider_family": self.registry_provider_family,
        }


@dataclass(frozen=True, repr=False)
class ProviderPlan:
    protocol: str
    cohorts: Mapping[Cohort, ResolvedModelRoute]
    judge: ResolvedModelRoute
    auxiliary: Mapping[str, ResolvedModelRoute]
    transports: Mapping[str, ResolvedTransport]

    def __repr__(self) -> str:
        return f"ProviderPlan({self.safe_metadata()!r})"

    @property
    def targets(self) -> tuple[ModelTarget, ...]:
        return tuple(
            ModelTarget(
                cohort=cohort,
                model_id=self.cohorts[cohort].model_id,
                provider=self.cohorts[cohort].provider_id,
            )
            for cohort in (Cohort.WEAK, Cohort.MIDDLE, Cohort.STRONG)
        )

    def route(self, name: str) -> ResolvedModelRoute:
        if name in {cohort.value for cohort in Cohort}:
            return self.cohorts[Cohort(name)]
        if name == "judge":
            return self.judge
        try:
            return self.auxiliary[name]
        except KeyError:
            raise ProviderConfigurationError(f"unknown provider route: {name}") from None

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "schema": "eva.resolved-provider-plan.v1",
            "protocol": self.protocol,
            "cohorts": {
                cohort.value: self.cohorts[cohort].safe_metadata()
                for cohort in (Cohort.WEAK, Cohort.MIDDLE, Cohort.STRONG)
            },
            "judge": self.judge.safe_metadata(),
            "auxiliary": {
                name: route.safe_metadata() for name, route in sorted(self.auxiliary.items())
            },
            "transports": {
                name: transport.safe_metadata()
                for name, transport in sorted(self.transports.items())
            },
            "one_rollout_per_model_per_sandbox": True,
            "parallel_tool_calls": True,
            "semantic_retry_count": 0,
        }


def _object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProviderConfigurationError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ProviderConfigurationError(f"{label} keys differ")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderConfigurationError(f"{label} is missing or invalid") from exc
    return _object(value, label=label)


def _transport_spec(value: Any, *, label: str) -> dict[str, Any]:
    row = _object(value, label=label)
    protocol = row.get("protocol")
    credentials = row.get("credential_env_priority")
    endpoints = row.get("endpoint_env_priority")
    if protocol != "openai-compatible-chat-completions":
        raise ProviderConfigurationError(f"{label} protocol differs")
    if not isinstance(credentials, list) or not credentials:
        raise ProviderConfigurationError(f"{label} credential priority differs")
    if not isinstance(endpoints, list) or not endpoints:
        raise ProviderConfigurationError(f"{label} endpoint priority differs")
    if any(name not in CREDENTIAL_ENV_NAMES for name in credentials):
        raise ProviderConfigurationError(f"{label} requests an unapproved credential environment name")
    if any(name not in ENDPOINT_ENV_NAMES for name in endpoints):
        raise ProviderConfigurationError(f"{label} requests an unapproved endpoint environment name")
    return row


def _registry_models(registry: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if registry.get("schema") != "rlevo.med-research-model-registry.v1":
        raise ProviderConfigurationError("model registry schema differs")
    models = registry.get("models")
    if not isinstance(models, list) or not models:
        raise ProviderConfigurationError("model registry models differ")
    rows: list[dict[str, Any]] = []
    roles: set[str] = set()
    for raw in models:
        row = _object(raw, label="model registry entry")
        role = row.get("role_id")
        model_id = row.get("api_model_id")
        family = row.get("provider_family")
        if not all(isinstance(item, str) and item for item in (role, model_id, family)):
            raise ProviderConfigurationError("model registry identity differs")
        if role in roles:
            raise ProviderConfigurationError("model registry role is duplicated")
        roles.add(role)
        rows.append(row)
    _transport_spec(registry.get("transport"), label="model registry transport")
    return tuple(rows)


def _read_env_assignments(paths: Sequence[Path]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise ProviderConfigurationError("provider environment file is missing or invalid") from exc
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
            assignments[name] = value
    return assignments


def _resolved_environment(
    *, environment: Mapping[str, str] | None, env_files: Sequence[Path]
) -> dict[str, str]:
    file_values = _read_env_assignments(env_files)
    source = os.environ if environment is None else environment
    values = {name: value for name, value in file_values.items() if value}
    # The process environment wins over files, matching OpenAI SDK conventions.
    values.update(
        {
            name: value
            for name, value in source.items()
            if name in ALLOWED_ENV_NAMES and isinstance(value, str) and value
        }
    )
    resolved: dict[str, str] = {}

    def resolve(name: str, stack: tuple[str, ...]) -> str | None:
        if name in stack:
            raise ProviderConfigurationError(f"provider environment reference cycle at {name}")
        value = values.get(name)
        if not value:
            return None
        reference = _ENV_REFERENCE.fullmatch(value)
        if reference:
            target = reference.group(1)
            if target not in ALLOWED_ENV_NAMES:
                raise ProviderConfigurationError(
                    f"provider environment reference from {name} is not allowlisted"
                )
            return resolve(target, (*stack, name))
        if "$" in value or "`" in value or "\x00" in value or "\n" in value or "\r" in value:
            raise ProviderConfigurationError(
                f"provider environment value for {name} uses unsupported syntax"
            )
        return value

    for name in ALLOWED_ENV_NAMES:
        value = resolve(name, ())
        if value:
            resolved[name] = value
    return resolved


def _first(names: Sequence[str], environment: Mapping[str, str], *, kind: str) -> tuple[str, str]:
    for name in names:
        value = environment.get(name)
        if value:
            return name, value
    raise ProviderConfigurationError(
        f"provider {kind} is missing; configure one of: {', '.join(names)}"
    )


def _safe_endpoint(value: str) -> str:
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ProviderConfigurationError("provider endpoint must be a credential-free HTTPS base URL")
    return value.strip().rstrip("/")


def _route_spec(value: Any, *, label: str) -> dict[str, Any]:
    row = _object(value, label=label)
    allowed = {"provider", "model_env", "registry_role", "registry_provider_family"}
    if not set(row) <= allowed or "provider" not in row:
        raise ProviderConfigurationError(f"{label} keys differ")
    provider = row.get("provider")
    if not isinstance(provider, str) or not provider:
        raise ProviderConfigurationError(f"{label} provider differs")
    model_env = row.get("model_env")
    role = row.get("registry_role")
    family = row.get("registry_provider_family")
    if model_env is not None and model_env not in MODEL_ENV_NAMES:
        raise ProviderConfigurationError(f"{label} model environment name is not allowlisted")
    if role is not None and (not isinstance(role, str) or not role):
        raise ProviderConfigurationError(f"{label} registry role differs")
    if family is not None and (not isinstance(family, str) or not family):
        raise ProviderConfigurationError(f"{label} registry provider family differs")
    if role is not None and family is not None:
        raise ProviderConfigurationError(f"{label} must not select both a registry role and family")
    if model_env is None and role is None and family is None:
        raise ProviderConfigurationError(f"{label} has no model identity source")
    return row


def _registry_model(
    models: Sequence[Mapping[str, Any]], *, role: str | None, family: str | None, label: str
) -> Mapping[str, Any] | None:
    if role is not None:
        matches = [row for row in models if row["role_id"] == role]
    elif family is not None:
        matches = [row for row in models if row["provider_family"] == family]
    else:
        return None
    if len(matches) != 1:
        raise ProviderConfigurationError(f"{label} registry selector is missing or ambiguous")
    return matches[0]


def _resolve_route(
    name: str,
    raw: Any,
    *,
    providers: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    environment: Mapping[str, str],
) -> tuple[ResolvedModelRoute, Mapping[str, Any]]:
    spec = _route_spec(raw, label=f"route {name}")
    provider_id = spec["provider"]
    if provider_id not in providers:
        raise ProviderConfigurationError(f"route {name} refers to an unknown provider")
    registry_model = _registry_model(
        models,
        role=spec.get("registry_role"),
        family=spec.get("registry_provider_family"),
        label=f"route {name}",
    )
    model_env = spec.get("model_env")
    model_id = environment.get(model_env) if model_env is not None else None
    if model_id is None and registry_model is not None:
        model_id = registry_model["api_model_id"]
    if not isinstance(model_id, str) or _MODEL_ID.fullmatch(model_id) is None:
        source_name = model_env or "the selected registry entry"
        raise ProviderConfigurationError(f"route {name} has no valid model ID from {source_name}")
    transport = (
        registry_model.get("transport_override") if registry_model is not None else None
    ) or providers[provider_id]["registry_transport"]
    return (
        ResolvedModelRoute(
            route_name=name,
            provider_id=provider_id,
            model_id=model_id,
            model_env_name=model_env,
            registry_role_id=spec.get("registry_role"),
            registry_provider_family=(
                registry_model["provider_family"] if registry_model is not None else None
            ),
        ),
        _transport_spec(transport, label=f"route {name} transport"),
    )


def load_provider_plan(
    config_path: str | Path,
    *,
    registry_path: str | Path,
    environment: Mapping[str, str] | None = None,
    env_files: Sequence[str | Path] = (),
) -> ProviderPlan:
    """Resolve a production routing plan without making a provider call."""

    config = _load_json(Path(config_path), label="provider model-tier configuration")
    _exact_keys(
        config,
        {"schema", "protocol", "providers", "cohorts", "judge", "auxiliary"},
        label="provider model-tier configuration",
    )
    if config["schema"] != "eva.provider-model-routing.v1":
        raise ProviderConfigurationError("provider model-tier schema differs")
    if config["protocol"] != "openai-compatible-chat-completions":
        raise ProviderConfigurationError("provider protocol differs")
    registry = _load_json(Path(registry_path), label="rlevo model registry")
    models = _registry_models(registry)
    registry_transport = _transport_spec(registry["transport"], label="model registry transport")
    raw_providers = _object(config["providers"], label="providers")
    if not raw_providers:
        raise ProviderConfigurationError("at least one provider is required")
    providers: dict[str, dict[str, Any]] = {}
    for provider_id, raw in raw_providers.items():
        if not isinstance(provider_id, str) or not provider_id:
            raise ProviderConfigurationError("provider identity differs")
        row = _object(raw, label=f"provider {provider_id}")
        _exact_keys(
            row,
            {"max_concurrency", "requests_per_minute", "burst", "queue_timeout_seconds"},
            label=f"provider {provider_id}",
        )
        timeout = row["queue_timeout_seconds"]
        if type(timeout) not in {int, float}:
            raise ProviderConfigurationError(f"provider {provider_id} queue timeout differs")
        limits = ProviderLimits(
            max_concurrency=row["max_concurrency"],
            requests_per_minute=row["requests_per_minute"],
            burst=row["burst"],
            queue_timeout_seconds=float(timeout),
        )
        providers[provider_id] = {"limits": limits, "registry_transport": registry_transport}
    resolved_env = _resolved_environment(
        environment=environment, env_files=tuple(Path(path) for path in env_files)
    )
    raw_cohorts = _object(config["cohorts"], label="cohort routes")
    _exact_keys(raw_cohorts, {"weak", "middle", "strong"}, label="cohort routes")
    routes_by_name: dict[str, ResolvedModelRoute] = {}
    route_transports: dict[str, Mapping[str, Any]] = {}
    all_routes = list(raw_cohorts.items()) + [("judge", config["judge"])]
    all_routes += list(_object(config["auxiliary"], label="auxiliary routes").items())
    for name, raw in all_routes:
        if not isinstance(name, str) or not name:
            raise ProviderConfigurationError("provider route name differs")
        if name in routes_by_name:
            raise ProviderConfigurationError(f"provider route is duplicated: {name}")
        route, transport = _resolve_route(
            name, raw, providers=providers, models=models, environment=resolved_env
        )
        routes_by_name[name] = route
        route_transports[name] = transport

    primary = [routes_by_name[name].model_id for name in ("weak", "middle", "strong", "judge")]
    if len(set(primary[:3])) != 3:
        raise ProviderConfigurationError("weak, middle, and strong must use distinct model IDs")
    if primary[3] in primary[:3]:
        raise ProviderConfigurationError("Opus 5 judge must be distinct from rollout models")
    normalized_judge = primary[3].casefold().replace("_", "-").replace(" ", "-")
    if "opus-5" not in normalized_judge:
        raise ProviderConfigurationError("judge route must resolve to Opus 5")

    resolved_transports: dict[str, ResolvedTransport] = {}
    for provider_id, provider in providers.items():
        selected = [
            route_transports[name]
            for name, route in routes_by_name.items()
            if route.provider_id == provider_id
        ]
        if not selected:
            raise ProviderConfigurationError(f"provider {provider_id} has no routes")
        signatures = {
            (tuple(row["credential_env_priority"]), tuple(row["endpoint_env_priority"]))
            for row in selected
        }
        if len(signatures) != 1:
            raise ProviderConfigurationError(
                f"provider {provider_id} combines incompatible transport overrides"
            )
        transport = selected[0]
        credential_name, credential = _first(
            transport["credential_env_priority"], resolved_env, kind="credential"
        )
        endpoint_name, endpoint = _first(
            transport["endpoint_env_priority"], resolved_env, kind="endpoint"
        )
        if not 8 <= len(credential) <= 16_384:
            raise ProviderConfigurationError("provider credential length differs")
        resolved_transports[provider_id] = ResolvedTransport(
            provider_id=provider_id,
            credential_env_name=credential_name,
            endpoint_env_name=endpoint_name,
            credential=credential,
            endpoint=_safe_endpoint(endpoint),
            limits=provider["limits"],
        )

    cohorts = MappingProxyType({cohort: routes_by_name[cohort.value] for cohort in Cohort})
    auxiliary_names = set(routes_by_name) - {"weak", "middle", "strong", "judge"}
    return ProviderPlan(
        protocol=config["protocol"],
        cohorts=cohorts,
        judge=routes_by_name["judge"],
        auxiliary=MappingProxyType({name: routes_by_name[name] for name in sorted(auxiliary_names)}),
        transports=MappingProxyType(dict(sorted(resolved_transports.items()))),
    )


__all__ = [
    "ALLOWED_ENV_NAMES",
    "CREDENTIAL_ENV_NAMES",
    "ENDPOINT_ENV_NAMES",
    "MODEL_ENV_NAMES",
    "ProviderConfigurationError",
    "ProviderLimits",
    "ProviderPlan",
    "ResolvedModelRoute",
    "ResolvedTransport",
    "load_provider_plan",
]
