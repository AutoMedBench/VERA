from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from eva_agent.training.qwen_tool_types import HostArgumentProjector, HostTypedChatTransport, schema_types


def fixture():
    schema = {"type": "object", "$defs": {"sha": {"type": "string"}}, "properties": {
        "digest": {"$ref": "#/$defs/sha"}, "object": {"type": "object"},
        "array": {"anyOf": [{"type": "array"}, {"type": "null"}]},
        "flag": {"const": False}, "count": {"allOf": [{"type": "integer"}, {"minimum": 0}]},
        "ambiguous": {"oneOf": [{"type": "string"}, {"type": "array"}]}}}
    public = {"materialize_plan": {"name": "materialize_plan", "description": "canonical description",
                                   "input_schema": {"type": "object", "additionalProperties": True}}}
    projector = HostArgumentProjector(public_tools=public, host_schemas={"materialize_plan": schema},
                                      provenance={"fixture": True})
    tools = [{"type": "function", "function": {"name": "mcp__evamed__materialize_plan",
        "description": "canonical description", "parameters": {"type": "object", "properties": {},
                                                                      "additionalProperties": True}}}]
    return projector, tools


def message(arguments):
    return {"role": "assistant", "content": None, "reasoning_content": "PRIVATE",
        "tool_calls": [{"id": "call", "type": "function", "function": {
            "name": "mcp__evamed__materialize_plan", "arguments": json.dumps(arguments)}}]}


def test_authoritative_types_only_lossless_and_unknown_fields_unchanged():
    projector, tools = fixture()
    original = message({"digest": "123", "object": '{"nested":"[1]"}', "array": '[1,2]',
                        "flag": "False", "count": "12", "ambiguous": "[3]", "arguments": '{"object":{}}'})
    before, tool_before = deepcopy(original), deepcopy(tools)
    projected, audit = projector.project_message(original, tools)
    values = json.loads(projected["tool_calls"][0]["function"]["arguments"])
    assert values == {"digest": "123", "object": {"nested": "[1]"}, "array": [1, 2],
                      "flag": False, "count": 12, "ambiguous": "[3]", "arguments": '{"object":{}}'}
    assert original == before and tools == tool_before
    assert projected["reasoning_content"] == "PRIVATE" and "PRIVATE" not in json.dumps(audit)
    assert len(audit["calls"][0]["changes"]) == 4


@pytest.mark.parametrize("field,value", [("object", "{'x':1}"), ("object", '{"x":1,"x":2}'),
    ("object", '{"x":NaN}'), ("array", "[1,]"), ("count", "1e999"), ("flag", "0")])
def test_invalid_model_values_are_not_repaired(field, value):
    projector, tools = fixture()
    original = message({field: value})
    assert projector.project_message(original, tools)[0] == original


def test_boolean_value_is_not_changed_to_schema_const_and_no_required_fields_filled():
    projector, tools = fixture()
    projected, _ = projector.project_message(message({"flag": "True"}), tools)
    assert json.loads(projected["tool_calls"][0]["function"]["arguments"]) == {"flag": True}


def test_ref_cycles_fail_closed_and_ambiguous_types_stay_ambiguous():
    with pytest.raises(ValueError, match="cyclic"):
        schema_types({"$ref": "#/$defs/a", "$defs": {"a": {"$ref": "#/$defs/a"}}})
    assert schema_types({"anyOf": [{"type": "object"}, {}]}) is None


def test_schema_or_description_drift_rejected_and_unoffered_names_not_decoded():
    projector, tools = fixture()
    tools[0]["function"]["parameters"]["additionalProperties"] = False
    with pytest.raises(ValueError, match="differs"):
        projector.project_message(message({"flag": "False"}), tools)
    original = message({"flag": "False"})
    assert projector.project_message(original, [])[0] == original


def test_chat_wrapper_retains_raw_bytes_and_keeps_request_unchanged():
    projector, tools = fixture()
    raw = json.dumps({"choices": [{"message": message({"flag": "False"})}]}).encode()
    body = {"tools": tools, "messages": [{"role": "user", "content": "exact request"}]}
    before, retained = deepcopy(body), []
    def upstream(binding, value):
        assert value is body
        return SimpleNamespace(status=200, body=raw, latency_ms=7)
    transport = HostTypedChatTransport(upstream, projector, lambda data, audit: retained.append((data, audit)))
    outcome = transport(None, body)
    assert retained[0][0] == raw and body == before and outcome.latency_ms == 7
    args = json.loads(json.loads(outcome.body)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args == {"flag": False}
