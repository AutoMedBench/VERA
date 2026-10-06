"""Bounded OpenAI-compatible clients adapted into the EVA data pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any

from eva_agent.pipeline import (
    Cohort,
    ModelTarget,
    OpenAIStyleOpus5Judge,
    OpenAIStyleRolloutAdapter,
    ProviderRollout,
    RolloutRequest,
    RuntimeIdFactory,
    ToolRuntimePort,
)

from .config import (
    ProviderConfigurationError,
    ProviderLimits,
    ProviderPlan,
    ResolvedModelRoute,
)


class ProviderCapacityError(RuntimeError):
    """A bounded provider queue/rate budget was exhausted."""


class DuplicateSemanticAttemptError(RuntimeError):
    """The same model was asked to roll out the same sandbox twice."""


class ProviderGate:
    """One shared concurrency semaphore and token bucket per provider."""

    def __init__(
        self,
        limits: ProviderLimits,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.limits = limits
        self._clock = clock
        self._sleep = sleep
        self._semaphore = threading.BoundedSemaphore(limits.max_concurrency)
        self._rate_lock = threading.Lock()
        self._tokens = float(limits.burst)
        self._updated = clock()

    def _take_rate_token(self, deadline: float) -> None:
        per_second = self.limits.requests_per_minute / 60.0
        while True:
            with self._rate_lock:
                now = self._clock()
                elapsed = max(0.0, now - self._updated)
                self._tokens = min(
                    float(self.limits.burst), self._tokens + elapsed * per_second
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                delay = (1.0 - self._tokens) / per_second
            if self._clock() + delay > deadline:
                raise ProviderCapacityError("provider rate queue deadline exceeded")
            self._sleep(delay)

    @contextmanager
    def slot(self):
        deadline = self._clock() + self.limits.queue_timeout_seconds
        if not self._semaphore.acquire(timeout=self.limits.queue_timeout_seconds):
            raise ProviderCapacityError("provider concurrency queue deadline exceeded")
        try:
            self._take_rate_token(deadline)
            yield
        finally:
            self._semaphore.release()


class _RoutedCompletions:
    def __init__(
        self,
        upstream: Any,
        *,
        route: ResolvedModelRoute,
        gate: ProviderGate,
    ) -> None:
        try:
            self._create = upstream.chat.completions.create
        except AttributeError:
            raise ProviderConfigurationError(
                "client factory result lacks chat.completions.create"
            ) from None
        self._route = route
        self._gate = gate

    def create(self, **request: Any) -> Any:
        if request.get("model") != self._route.model_id:
            raise ProviderConfigurationError(
                f"request model does not match route {self._route.route_name}"
            )
        if request.get("parallel_tool_calls") is False:
            raise ProviderConfigurationError("parallel_tool_calls cannot be disabled")
        outbound = dict(request)
        outbound.setdefault("parallel_tool_calls", True)
        with self._gate.slot():
            # Transport failures bubble untouched for infrastructure quarantine.
            return self._create(**outbound)


class RoutedChatClient:
    """Minimal chat-completions surface consumed by existing pipeline adapters."""

    def __init__(self, upstream: Any, *, route: ResolvedModelRoute, gate: ProviderGate) -> None:
        self.chat = SimpleNamespace(
            completions=_RoutedCompletions(upstream, route=route, gate=gate)
        )
        self.route = route

    def __repr__(self) -> str:
        return (
            f"RoutedChatClient(route={self.route.route_name!r}, "
            f"provider={self.route.provider_id!r}, credentials=<redacted>)"
        )


class SingleAttemptRolloutProvider:
    """Guard one immutable semantic rollout per ``(sandbox, model)``."""

    def __init__(
        self,
        delegate: OpenAIStyleRolloutAdapter,
        *,
        route: ResolvedModelRoute,
    ) -> None:
        self._delegate = delegate
        self._route = route
        self._claimed: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def run(self, request: RolloutRequest, tools: ToolRuntimePort) -> ProviderRollout:
        if (
            request.model.model_id != self._route.model_id
            or request.model.provider != self._route.provider_id
            or request.model.cohort.value != self._route.route_name
        ):
            raise ProviderConfigurationError(
                f"rollout request does not match route {self._route.route_name}"
            )
        attempt = (request.sandbox.manifest_blake3, request.model.model_id)
        with self._lock:
            if attempt in self._claimed:
                raise DuplicateSemanticAttemptError(
                    "semantic rollout already claimed for this sandbox and model"
                )
            # Claim before dispatch. A transport failure is terminal for this
            # attempt and is handled by infrastructure quarantine upstream.
            self._claimed.add(attempt)
        result = self._delegate.run(request, tools)
        metadata = dict(result.safe_metadata)
        metadata.update(self._route.safe_metadata())
        metadata.update(
            {
                "parallel_tool_calls": True,
                "semantic_retry_count": 0,
                "raw_provider_response_recorded": False,
                "credential_value_recorded": False,
                "endpoint_value_recorded": False,
            }
        )
        return ProviderRollout(
            assistant_output=result.assistant_output,
            provider_receipt_blake3=result.provider_receipt_blake3,
            policy_events=result.policy_events,
            safe_metadata=metadata,
        )


@dataclass(frozen=True)
class PipelineProviderRuntime:
    targets: tuple[ModelTarget, ...]
    rollout_providers: Mapping[Cohort, SingleAttemptRolloutProvider]
    judge: OpenAIStyleOpus5Judge
    judge_model_id: str
    auxiliary_clients: Mapping[str, RoutedChatClient]
    safe_metadata: Mapping[str, Any]


ClientFactory = Callable[..., Any]


def _openai_client_factory(*, api_key: str, base_url: str) -> Any:
    from openai import OpenAI

    # The campaign owns retry/quarantine policy; SDK hidden retries are off.
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=0)


def build_pipeline_provider_runtime(
    plan: ProviderPlan,
    *,
    id_factory: RuntimeIdFactory,
    client_factory: ClientFactory | None = None,
) -> PipelineProviderRuntime:
    """Adapt resolved routes into the existing pipeline rollout/judge ports."""

    factory = client_factory or _openai_client_factory
    gates = {
        provider_id: ProviderGate(transport.limits)
        for provider_id, transport in plan.transports.items()
    }
    upstreams: dict[str, Any] = {}
    for provider_id, transport in plan.transports.items():
        credential, endpoint = transport.client_material()
        upstreams[provider_id] = factory(api_key=credential, base_url=endpoint)

    def client_for(route: ResolvedModelRoute) -> RoutedChatClient:
        return RoutedChatClient(
            upstreams[route.provider_id], route=route, gate=gates[route.provider_id]
        )

    rollout_providers: dict[Cohort, SingleAttemptRolloutProvider] = {}
    for cohort in (Cohort.WEAK, Cohort.MIDDLE, Cohort.STRONG):
        route = plan.cohorts[cohort]
        adapter = OpenAIStyleRolloutAdapter(client_for(route), id_factory=id_factory)
        rollout_providers[cohort] = SingleAttemptRolloutProvider(adapter, route=route)
    judge_client = client_for(plan.judge)
    judge = OpenAIStyleOpus5Judge(
        judge_client,
        model_id=plan.judge.model_id,
        id_factory=id_factory,
    )
    auxiliary_clients = {name: client_for(route) for name, route in plan.auxiliary.items()}
    return PipelineProviderRuntime(
        targets=plan.targets,
        rollout_providers=MappingProxyType(rollout_providers),
        judge=judge,
        judge_model_id=plan.judge.model_id,
        auxiliary_clients=MappingProxyType(auxiliary_clients),
        safe_metadata=MappingProxyType(plan.safe_metadata()),
    )


__all__ = [
    "DuplicateSemanticAttemptError",
    "PipelineProviderRuntime",
    "ProviderCapacityError",
    "ProviderGate",
    "RoutedChatClient",
    "SingleAttemptRolloutProvider",
    "build_pipeline_provider_runtime",
]
