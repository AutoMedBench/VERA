from __future__ import annotations

import asyncio
from http import HTTPStatus
import json
from types import SimpleNamespace
import threading
import time
import urllib.error
import urllib.request

import pytest

from eva_agent.codex_providers import (
    AdapterLimits,
    ResponsesAdapterError,
    ResponsesAdapterGateway,
)
from eva_agent.codex_providers.routes import (
    CodexConfigOverrides,
    CodexProviderRoute,
    _PrivateText,
)
from eva_agent.pipeline.digests import blake3_bytes, canonical_json_bytes


MODEL = "Qwen/Qwen3.5-9B-capacity-fixture"
TOKEN = "local-capacity-fixture-token"


def _route() -> CodexProviderRoute:
    return CodexProviderRoute(
        route_id="qwen_capacity_fixture",
        model_id=MODEL,
        model_env_name="MODEL_QWEN_3_5_9B",
        registry_role_id=None,
        provider_family="qwen",
        config=CodexConfigOverrides(
            provider_id="eva_capacity_fixture",
            model_id=MODEL,
            credential_env_name="OPENAI_API_KEY",
            endpoint_env_name="OPENAI_BASE_URL",
            _credential=_PrivateText("local-nonsecret"),
            _endpoint=_PrivateText("http://127.0.0.1:1/v1"),
        ),
    )


def _response() -> bytes:
    return json.dumps({
        "id": "capacity-fixture",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }).encode()


def _request(endpoint: str) -> tuple[int, bytes]:
    request = urllib.request.Request(
        endpoint + "/responses",
        data=json.dumps({"model": MODEL, "input": "fixture", "stream": False,
                         "store": False}).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _endpoint(gateway: ResponsesAdapterGateway) -> str:
    adapted = gateway.adapted_routes()["qwen_capacity_fixture"]
    _, _, (endpoint, _) = adapted.config.for_subprocess()
    return endpoint


def test_bounded_wait_admits_after_delayed_orphan_slot_release() -> None:
    calls = 0

    def upstream(_binding, _body):
        nonlocal calls
        calls += 1
        return SimpleNamespace(status=HTTPStatus.OK, body=_response(), latency_ms=1)

    events = []
    limits = AdapterLimits(max_concurrency=1, capacity_wait_seconds=0.5)
    with ResponsesAdapterGateway(
        {"qwen_capacity_fixture": _route()}, limits=limits,
        upstream_transport=upstream, local_bearer_token=TOKEN,
        capacity_event_sink=lambda event: events.append(dict(event)),
    ) as gateway:
        assert gateway._semaphore.acquire(blocking=False)
        release = threading.Timer(0.15, gateway._semaphore.release)
        release.start()
        started = time.monotonic()
        status, _ = asyncio.run(asyncio.to_thread(_request, _endpoint(gateway)))
        elapsed = time.monotonic() - started
        release.join(timeout=1)
        assert status == 200 and elapsed >= 0.12
        assert calls == 1 and len(gateway.receipts) == 1
        assert len(events) == len(gateway.capacity_events) == 1
        assert events[0]["outcome"] == "acquired"
        assert events[0]["capacity_acquired"] is True
        assert events[0]["wait_elapsed_ms"] >= 80
        assert events[0]["upstream_dispatch_claimed"] is False


def test_expired_wait_is_actual_async_zero_provider_fixture() -> None:
    calls = 0

    def upstream(_binding, _body):
        nonlocal calls
        calls += 1
        raise AssertionError("expired admission must not cross provider boundary")

    limits = AdapterLimits(max_concurrency=1, capacity_wait_seconds=0.03)
    with ResponsesAdapterGateway(
        {"qwen_capacity_fixture": _route()}, limits=limits,
        upstream_transport=upstream, local_bearer_token=TOKEN,
    ) as gateway:
        assert gateway._semaphore.acquire(blocking=False)
        status, payload = asyncio.run(asyncio.to_thread(_request, _endpoint(gateway)))
        gateway._semaphore.release()
        assert status == 429 and json.loads(payload)["error"]["type"] == "capacity_error"
        assert calls == 0 and gateway.receipts == ()
        assert len(gateway.capacity_events) == 1
        event = dict(gateway.capacity_events[0])
        assert event["outcome"] == "wait_expired"
        assert event["capacity_acquired"] is False
        assert event["upstream_dispatch_claimed"] is False
        assert event["signed_adapter_receipt_claimed"] is False
        digest = event.pop("event_blake3")
        assert digest == blake3_bytes(canonical_json_bytes(event))


def test_wait_queue_never_exceeds_adapter_inflight_cap() -> None:
    lock = threading.Lock()
    active = maximum = calls = 0

    def upstream(_binding, _body):
        nonlocal active, maximum, calls
        with lock:
            active += 1
            calls += 1
            maximum = max(maximum, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return SimpleNamespace(status=HTTPStatus.OK, body=_response(), latency_ms=50)

    async def exercise(endpoint: str):
        return await asyncio.gather(*(
            asyncio.to_thread(_request, endpoint) for _ in range(6)
        ))

    limits = AdapterLimits(max_concurrency=2, capacity_wait_seconds=1)
    with ResponsesAdapterGateway(
        {"qwen_capacity_fixture": _route()}, limits=limits,
        upstream_transport=upstream, local_bearer_token=TOKEN,
    ) as gateway:
        results = asyncio.run(exercise(_endpoint(gateway)))
        assert [status for status, _ in results] == [200] * 6
        assert calls == 6 and maximum == 2 and active == 0
        assert len(gateway.receipts) == len(gateway.capacity_events) == 6
        assert all(event["capacity_acquired"] for event in gateway.capacity_events)


def test_default_remains_immediate_fail_fast() -> None:
    limits = AdapterLimits(max_concurrency=1)
    assert limits.capacity_wait_seconds == 0
    with ResponsesAdapterGateway(
        {"qwen_capacity_fixture": _route()}, limits=limits,
        upstream_transport=lambda *_: (_ for _ in ()).throw(AssertionError()),
        local_bearer_token=TOKEN,
    ) as gateway:
        assert gateway._semaphore.acquire(blocking=False)
        started = time.monotonic()
        status, _ = _request(_endpoint(gateway))
        gateway._semaphore.release()
        assert status == 429 and time.monotonic() - started < 0.2
        assert gateway.receipts == ()
        assert gateway.capacity_events[0]["outcome"] == "rejected_no_wait"


@pytest.mark.parametrize("value", [True, -0.1, 901, float("nan"), float("inf")])
def test_capacity_wait_rejects_ambiguous_or_unbounded_values(value) -> None:
    with pytest.raises(ResponsesAdapterError, match="capacity wait is invalid"):
        AdapterLimits(capacity_wait_seconds=value)
