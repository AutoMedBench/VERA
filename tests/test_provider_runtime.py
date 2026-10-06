from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

from eva_agent.pipeline import Cohort, DeterministicUUIDFactory
from eva_agent.providers import (
    DuplicateSemanticAttemptError,
    ProviderConfigurationError,
    ProviderGate,
    ProviderLimits,
    ResolvedModelRoute,
    RoutedChatClient,
    SingleAttemptRolloutProvider,
    build_pipeline_provider_runtime,
    load_provider_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def _route(name: str = "weak", model_id: str = "model-weak") -> ResolvedModelRoute:
    return ResolvedModelRoute(
        route_name=name,
        provider_id="gateway",
        model_id=model_id,
        model_env_name="MODEL_DEEPSEEK_V4_FLASH",
        registry_role_id=None,
        registry_provider_family="deepseek",
    )


class _RecordingCompletions:
    def __init__(self, *, delay: float = 0) -> None:
        self.requests: list[dict] = []
        self.delay = delay
        self.active = 0
        self.maximum_active = 0
        self.lock = threading.Lock()

    def create(self, **request):
        with self.lock:
            self.requests.append(request)
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            return {"choices": [{"message": {"content": "done"}}]}
        finally:
            with self.lock:
                self.active -= 1


def _upstream(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_routed_client_forces_parallel_tools_and_bounds_provider_concurrency() -> None:
    completions = _RecordingCompletions(delay=0.02)
    gate = ProviderGate(
        ProviderLimits(
            max_concurrency=2,
            requests_per_minute=100_000,
            burst=2,
            queue_timeout_seconds=2,
        )
    )
    client = RoutedChatClient(_upstream(completions), route=_route(), gate=gate)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: client.chat.completions.create(model="model-weak", messages=[]),
                range(8),
            )
        )
    assert len(results) == 8
    assert completions.maximum_active == 2
    assert all(request["parallel_tool_calls"] is True for request in completions.requests)


def test_routed_client_rejects_model_drift_and_disabled_parallel_calls() -> None:
    completions = _RecordingCompletions()
    gate = ProviderGate(ProviderLimits(2, 1000, 2, 1))
    client = RoutedChatClient(_upstream(completions), route=_route(), gate=gate)
    with pytest.raises(ProviderConfigurationError, match="does not match"):
        client.chat.completions.create(model="wrong")
    with pytest.raises(ProviderConfigurationError, match="cannot be disabled"):
        client.chat.completions.create(model="model-weak", parallel_tool_calls=False)
    assert completions.requests == []


def test_rate_bucket_waits_without_retrying_request() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def clock() -> float:
        return now[0]

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    gate = ProviderGate(
        ProviderLimits(1, 60, 1, 2), clock=clock, sleep=sleep
    )
    with gate.slot():
        pass
    with gate.slot():
        pass
    assert sleeps == [pytest.approx(1.0)]


def test_transport_exception_bubbles_and_semantic_attempt_cannot_repeat() -> None:
    class TransportFailure(RuntimeError):
        pass

    failure = TransportFailure("transport unavailable")

    class Delegate:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, request, tools):
            del request, tools
            self.calls += 1
            raise failure

    delegate = Delegate()
    provider = SingleAttemptRolloutProvider(delegate, route=_route())  # type: ignore[arg-type]
    request = SimpleNamespace(
        model=SimpleNamespace(
            model_id="model-weak", provider="gateway", cohort=Cohort.WEAK
        ),
        sandbox=SimpleNamespace(manifest_blake3="a" * 64),
    )
    with pytest.raises(TransportFailure) as raised:
        provider.run(request, None)  # type: ignore[arg-type]
    assert raised.value is failure
    with pytest.raises(DuplicateSemanticAttemptError):
        provider.run(request, None)  # type: ignore[arg-type]
    assert delegate.calls == 1


def test_sdk_factory_is_not_constructed_by_import() -> None:
    # The production OpenAI client is lazy: importing and using limits/UUIDs
    # requires neither credentials nor a network-capable client.
    assert DeterministicUUIDFactory("provider-test").new("event")


def test_runtime_factory_builds_one_shared_upstream_and_reuses_pipeline_adapters() -> None:
    secret = "runtime-only-secret-value"
    endpoint = "https://runtime-private.invalid/v1"
    environment = {
        "MODEL_DEEPSEEK_V4_FLASH": "deepseek-runtime",
        "MODEL_GEMINI_3_1_PRO": "gemini-runtime",
        "MODEL_GPT_5_6_SOL": "gpt-5.6-sol-runtime",
        "MODEL_OPUS_5": "claude-opus-5-runtime",
        "MODEL_OPUS_4_8": "claude-opus-4-8-runtime",
        "MODEL_DEEPSEEK_V4_PRO": "deepseek-pro-runtime",
        "NVIDIA_INFERENCE_API_KEY": secret,
        "NVIDIA_INFERENCE_BASE_URL": endpoint,
    }
    plan = load_provider_plan(
        ROOT / "config/model-tiers.example.json",
        registry_path=(
            ROOT.parent / "rlevo-med-research/config/model-registry.20260903-v2.json"
        ),
        environment=environment,
    )
    calls: list[dict[str, str]] = []
    completions = _RecordingCompletions()

    def factory(**kwargs):
        calls.append(kwargs)
        return _upstream(completions)

    runtime = build_pipeline_provider_runtime(
        plan,
        id_factory=DeterministicUUIDFactory("runtime-factory"),
        client_factory=factory,
    )
    assert calls == [{"api_key": secret, "base_url": endpoint}]
    assert set(runtime.rollout_providers) == {
        Cohort.WEAK,
        Cohort.MIDDLE,
        Cohort.STRONG,
    }
    assert runtime.judge_model_id == "claude-opus-5-runtime"
    assert runtime.judge.__class__.__name__ == "OpenAIStyleOpus5Judge"
    runtime.auxiliary_clients["qwen"].chat.completions.create(
        model=plan.auxiliary["qwen"].model_id, messages=[]
    )
    assert completions.requests[-1]["parallel_tool_calls"] is True
    assert secret not in repr(runtime.safe_metadata)
    assert endpoint not in repr(runtime.safe_metadata)
