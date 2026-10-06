"""Real local gateway + fake upstream; no native runtime or external provider."""
from copy import deepcopy
import json
from types import SimpleNamespace
import urllib.error

import pytest

from eva_agent.codex_pipeline.adapter import _judge_output_schema
from eva_agent.codex_providers import adapter
from eva_agent.codex_providers import judge_verdict_correction as correction
from eva_agent.pipeline.digests import blake3_hex, canonical_value
from test_codex_provider_routes import _routes, _post_json, _chat_response


VERDICT = {"item_scores": [{"item_id": "item-1", "score": 0,
    "evidence_refs": ["workspace:before:contract.txt"], "rationale": "Original zero."}],
    "hard_gates_passed": True, "summary": "Original assessment."}
SCHEMA = canonical_value(_judge_output_schema([{"item_id": "item-1"}]))


class FakeUpstream:
    def __init__(self, responses, clock=None):
        self.responses, self.calls, self.timeouts, self.clock = responses, [], [], clock

    def __call__(self, binding, body):
        return self.call_with_timeout(binding, body, timeout_seconds=240)

    def call_with_timeout(self, binding, body, *, timeout_seconds):
        self.calls.append(deepcopy(body))
        self.timeouts.append(timeout_seconds)
        response = self.responses[len(self.calls) - 1]
        if isinstance(response, Exception):
            raise response
        if self.clock is not None:
            self.clock[0] += 10
        return adapter._UpstreamOutcome(status=200, body=json.dumps(response).encode(), latency_ms=2)


def bad(extra="tool_calls"):
    value = deepcopy(VERDICT)
    value["item_scores"][0][extra] = [] if extra == "tool_calls" else "extra annotation"
    return value


def request(route):
    return {"model": route.model_id, "input": [{"role": "user", "content": "Exact compiled Judge task."}],
            "stream": False, "store": False,
            "text": {"format": {"type": "json_schema", "name": "judge", "strict": True, "schema": SCHEMA}}}


def send(gateway, route):
    selected = gateway.adapted_routes()[route.route_id]
    _, _, (endpoint, token) = selected.config.for_subprocess()
    return _post_json(endpoint + "/responses", token, request(route))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    route = _routes(tmp_path)["opus_5"]
    clock = [100.0]
    monkeypatch.setattr(correction, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    records, eligible_texts = [], []
    policy = correction.JudgeVerdictCorrectionPolicy(judgment_id="00000000-0000-4000-8000-000000000001",
        source_binding={"rubric_digest": blake3_hex("fixture"), "evidence_bundle_blake3": blake3_hex("read")}, sink=records.append)
    def eligible(text):
        eligible_texts.append(text)
        return {"actual_reads_checked": True, "required_fields_blake3": blake3_hex(VERDICT)}
    policy.bind(eligibility=eligible, deadline_monotonic=700)
    return route, clock, policy, records, eligible_texts


@pytest.mark.parametrize("extra", sorted(correction.EXTRA_KEYS))
def test_actual_gateway_preserves_original_history_and_model_values(fixture, extra):
    route, clock, policy, records, eligible = fixture
    original_text, final_text = json.dumps(bad(extra)), json.dumps(VERDICT, indent=2)
    upstream = FakeUpstream([_chat_response(route.model_id, content=original_text),
                             _chat_response(route.model_id, content=final_text)], clock)
    with adapter.ResponsesAdapterGateway({route.route_id: route}, upstream_transport=upstream,
                                        judge_verdict_correction=policy) as gateway:
        status, body, _ = send(gateway, route)
    assert status == 200 and json.loads(body)["output"][0]["content"][0]["text"] == final_text
    assert eligible == [original_text] and len(upstream.calls) == 2
    first, second = upstream.calls
    assert second["messages"][:-2] == first["messages"]
    assert second["messages"][-2] == _chat_response(route.model_id, content=original_text)["choices"][0]["message"]
    assert second["response_format"] == first["response_format"]
    assert len(records) == 2 and policy.summary()["outcome"] == "corrected"
    prepared, completed = [correction.verify_judge_verdict_correction(row) for row in records]
    assert prepared["event"] == "prepared" and completed["event"] == "completed"
    assert completed["original"]["final_text"] == original_text
    assert completed["corrected"]["final_text"] == final_text
    assert completed["original"]["usage"]["input_tokens"] == 3
    assert completed["corrected"]["usage"]["output_tokens"] == 2
    assert completed["original"]["usage"]["output_tokens_details"]["reasoning_tokens"] == 0
    assert completed["previous_receipt_blake3"] == records[0]["envelope_blake3"]
    receipt = gateway.receipts[0]
    adapter.verify_adapter_receipt(receipt)
    assert receipt.payload["upstream_body_blake3"] == completed["corrected"]["upstream_body_blake3"]
    assert receipt.payload["request_shape"]["judge_verdict_correction"]["upstream_attempt_count"] == 2
    assert receipt.payload["upstream_latency_ms"] == 4
    assert receipt.public_key_blake3 == records[0]["public_key_blake3"]
    # The same policy cannot dispatch another corrective call on a later native request.
    with pytest.raises(correction.JudgeVerdictCorrectionError, match="already bound"):
        policy.bind(eligibility=lambda _: {}, deadline_monotonic=700)


def test_default_off_keeps_invalid_verdict_unchanged(fixture):
    route, _, _, _, _ = fixture
    text = json.dumps(bad())
    upstream = FakeUpstream([_chat_response(route.model_id, content=text)])
    with adapter.ResponsesAdapterGateway({route.route_id: route}, upstream_transport=upstream) as gateway:
        _, body, _ = send(gateway, route)
    assert json.loads(body)["output"][0]["content"][0]["text"] == text
    assert len(upstream.calls) == 1
    assert "judge_verdict_correction" not in gateway.safe_metadata
    assert "judge_verdict_correction" not in gateway.receipts[0].payload["request_shape"]


@pytest.mark.parametrize("change", ["valid_zero", "unknown_extra", "missing_field", "wrong_type", "nonempty_row_tools",
                                    "refusal", "tool_call", "truncated", "malformed_json", "duplicate_key"])
def test_excluded_responses_never_dispatch_correction(fixture, change):
    route, _, policy, records, eligible = fixture
    value = bad()
    if change == "valid_zero": value = deepcopy(VERDICT)
    if change == "unknown_extra": value["item_scores"][0]["unrecognized"] = 1
    if change == "missing_field": del value["item_scores"][0]["rationale"]
    if change == "wrong_type": value["item_scores"][0]["score"] = False
    if change == "nonempty_row_tools": value["item_scores"][0]["tool_calls"] = ["do this"]
    text = json.dumps(value)
    if change == "malformed_json": text = "{" + text
    if change == "duplicate_key": text = text.replace('"score": 0', '"score": 0, "score": 0')
    response = _chat_response(route.model_id, content=text)
    if change == "refusal": response["choices"][0]["message"]["refusal"] = "Refused"
    if change == "truncated": response["choices"][0]["finish_reason"] = "length"
    if change == "tool_call":
        response["choices"][0]["finish_reason"] = "tool_calls"
        response["choices"][0]["message"]["tool_calls"] = [{"id": "call", "type": "function",
            "function": {"name": "missing", "arguments": "{}"}}]
    upstream = FakeUpstream([response])
    with adapter.ResponsesAdapterGateway({route.route_id: route}, upstream_transport=upstream,
                                        judge_verdict_correction=policy) as gateway:
        try: send(gateway, route)
        except urllib.error.HTTPError: pass  # Unknown tool is independently rejected by the normal adapter.
    assert len(upstream.calls) == 1 and not records and not eligible


def test_evidence_ineligibility_and_eligibility_failure_never_retry(fixture):
    route, _, policy, records, _ = fixture
    policy._eligibility = lambda _: None
    upstream = FakeUpstream([_chat_response(route.model_id, content=json.dumps(bad()))])
    with adapter.ResponsesAdapterGateway({route.route_id: route}, upstream_transport=upstream,
                                        judge_verdict_correction=policy) as gateway:
        send(gateway, route)
    assert len(upstream.calls) == 1 and not records and policy.outcome == "ineligible"


@pytest.mark.parametrize("change", ["score", "citation", "rationale", "summary", "gate", "still_extra", "timeout", "tools"])
def test_correction_cannot_change_any_required_value_or_add_tools(fixture, change):
    route, _, policy, records, _ = fixture
    final = deepcopy(VERDICT)
    if change == "score": final["item_scores"][0]["score"] = 1
    if change == "citation": final["item_scores"][0]["evidence_refs"] = ["workspace:after:unread"]
    if change == "rationale": final["item_scores"][0]["rationale"] = "Better reason"
    if change == "summary": final["summary"] = "Changed"
    if change == "gate": final["hard_gates_passed"] = False
    if change == "still_extra": final = bad()
    last = _chat_response(route.model_id, content=json.dumps(final))
    if change == "timeout": last = TimeoutError("fixture")
    if change == "tools":
        last["choices"][0]["message"]["tool_calls"] = [{"id": "must-not-run"}]
    upstream = FakeUpstream([_chat_response(route.model_id, content=json.dumps(bad())), last])
    with adapter.ResponsesAdapterGateway({route.route_id: route}, upstream_transport=upstream,
                                        judge_verdict_correction=policy) as gateway:
        with pytest.raises(urllib.error.HTTPError): send(gateway, route)
    assert len(upstream.calls) == 2 and policy.outcome == "failed"
    assert len(records) == 2
    assert correction.verify_judge_verdict_correction(records[-1])["event"] == "failed"


def test_one_total_deadline_not_two_full_budgets(fixture):
    route, clock, policy, records, _ = fixture
    clock[0] = 675
    upstream = FakeUpstream([_chat_response(route.model_id, content=json.dumps(bad())),
                             _chat_response(route.model_id, content=json.dumps(VERDICT))], clock)
    with adapter.ResponsesAdapterGateway({route.route_id: route}, upstream_transport=upstream,
                                        judge_verdict_correction=policy) as gateway:
        send(gateway, route)
    assert upstream.timeouts == [25, 15]
    assert correction.verify_judge_verdict_correction(records[-1])["budget"]["elapsed_seconds"] == 595


def test_expired_original_has_no_provider_call(fixture):
    route, clock, policy, _, _ = fixture
    clock[0] = 701
    upstream = FakeUpstream([])
    with adapter.ResponsesAdapterGateway({route.route_id: route}, upstream_transport=upstream,
                                        judge_verdict_correction=policy) as gateway:
        with pytest.raises(urllib.error.HTTPError): send(gateway, route)
    assert not upstream.calls


def test_late_original_response_cannot_trigger_a_new_correction(fixture):
    route, clock, policy, records, eligible = fixture
    clock[0] = 695
    upstream = FakeUpstream([_chat_response(route.model_id, content=json.dumps(bad()))], clock)
    with adapter.ResponsesAdapterGateway({route.route_id: route}, upstream_transport=upstream,
                                        judge_verdict_correction=policy) as gateway:
        with pytest.raises(urllib.error.HTTPError): send(gateway, route)
    assert upstream.timeouts == [5] and not records and not eligible


def test_signed_text_and_source_tampering_rejected(fixture):
    route, _, policy, records, _ = fixture
    upstream = FakeUpstream([_chat_response(route.model_id, content=json.dumps(bad())),
                             _chat_response(route.model_id, content=json.dumps(VERDICT))])
    with adapter.ResponsesAdapterGateway({route.route_id: route}, upstream_transport=upstream,
                                        judge_verdict_correction=policy) as gateway:
        send(gateway, route)
    for field in ("original", "corrected", "source_binding"):
        altered = deepcopy(records[-1])
        altered["payload"][field]["tampered"] = True
        with pytest.raises(correction.JudgeVerdictCorrectionError):
            correction.verify_judge_verdict_correction(altered)


def test_only_one_corrective_call_across_multiple_native_requests(fixture):
    route, _, policy, records, eligible = fixture
    first = _chat_response(route.model_id, content=json.dumps(bad()))
    upstream = FakeUpstream([first, _chat_response(route.model_id, content=json.dumps(VERDICT)), first])
    with adapter.ResponsesAdapterGateway({route.route_id: route}, upstream_transport=upstream,
                                        judge_verdict_correction=policy) as gateway:
        send(gateway, route)
        _, body, _ = send(gateway, route)
    assert json.loads(body)["output"][0]["content"][0]["text"] == json.dumps(bad())
    assert len(upstream.calls) == 3 and len(records) == 2 and len(eligible) == 1
