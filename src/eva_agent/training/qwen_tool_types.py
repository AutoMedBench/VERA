"""Host-bound Qwen XML argument decoding, separate from public tool schemas.

The historical signed EvaMed policy offers two artifact tools with a generic
object schema. Their actual host validators have precise types. This module
decodes only those authoritative types; it never repairs model values, unwraps
arguments, adds required fields, or changes a prompt/tool offer. Raw provider
bytes and sampled tokens remain the caller's immutable private evidence.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

from eva_agent.pipeline.digests import blake3_bytes, blake3_hex


VERSION = "eva.qwen-host-argument-types.v1"
HOST_CONTRACTS = {
    "materialize_plan": ("stage-plan-artifact", "validate_stage_plan_artifact"),
    "materialize_evidence_selection": ("evidence-selection", "validate_evidence_selection"),
}


def _kind(value):
    return {str: "string", bool: "boolean", int: "integer", float: "number",
            dict: "object", list: "array", type(None): "null"}.get(type(value))


def schema_types(schema, root=None, seen=()):
    """Resolve local refs/combinators without fetching or inventing a type."""
    root = schema if root is None else root
    if not isinstance(schema, dict):
        return None
    restrictions = []
    if "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/") or ref in seen:
            raise ValueError("argument type reference is nonlocal or cyclic")
        target = root
        for part in ref[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        restrictions.append(schema_types(target, root, (*seen, ref)))
    declared = schema.get("type")
    if declared is not None:
        types = {declared} if isinstance(declared, str) else set(declared)
        if "number" in types:
            types.add("integer")
        restrictions.append(types)
    if "const" in schema:
        restrictions.append({_kind(schema["const"])})
    if "enum" in schema:
        restrictions.append({_kind(value) for value in schema["enum"]})
    for key in ("allOf", "anyOf", "oneOf"):
        if key not in schema:
            continue
        branches = [schema_types(value, root, seen) for value in schema[key]]
        if key == "allOf":
            restrictions.extend(branches)
        elif all(branch is not None for branch in branches):
            restrictions.append(set().union(*branches))
    known = [value for value in restrictions if value is not None]
    return set.intersection(*known) if known else None


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object field")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("non-finite JSON value")


def _decode(value, types):
    if not isinstance(value, str) or not types or "string" in types:
        return value
    stripped = value.strip()
    # Match the installed Qwen typed-boolean XML lexical convention, including
    # its True/False spelling; const equality remains the HOST validator's job.
    if types == {"boolean"} and stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    try:
        decoded = json.loads(stripped, object_pairs_hook=_unique_pairs,
                             parse_constant=_reject_constant)
    except (ValueError, TypeError):
        return value
    if isinstance(decoded, float) and not math.isfinite(decoded):
        return value
    return decoded if _kind(decoded) in types else value


class HostArgumentProjector:
    def __init__(self, *, public_tools, host_schemas, provenance):
        self.public_tools = deepcopy(public_tools)
        self.host_schemas = deepcopy(host_schemas)
        self.hints = {name: {key: sorted(types) for key, schema in document.get("properties", {}).items()
                            if (types := schema_types(schema, document))}
                      for name, document in self.host_schemas.items()}
        self.binding = {"schema": VERSION, "provenance": deepcopy(provenance),
            "canonical_tools_blake3": blake3_hex(self.public_tools),
            "host_schemas_blake3": blake3_hex(self.host_schemas), "field_types": self.hints,
            "public_schemas_mutated": False, "required_fields_filled": False,
            "unknown_fields_coerced": False, "host_validation_relaxed": False}
        self.binding_blake3 = blake3_hex(self.binding)

    @classmethod
    def from_context(cls, context):
        """Use the same already-verified legacy schema loader as the host."""
        policy_bytes = context.episode.initial_files[".eva/source-policy.json"]
        execution = context.episode.policy_context["execution_binding"]
        if execution["policy_blake3"] != blake3_bytes(policy_bytes):
            raise ValueError("parser host policy binding differs")
        policy = json.loads(policy_bytes)
        loader = sys.modules.get("rlevo_med_research.contract_validation")
        validation = sys.modules.get("rlevo_med_research.validation")
        if loader is None or validation is None or validation.validate_schema is not loader.validate_schema:
            raise ValueError("verified host schema loader is not bound")
        public = {tool["name"]: tool for tool in policy["tools"] if tool["name"] in HOST_CONTRACTS}
        schemas, source_files = {}, {}
        for name in public:
            schema_name, symbol = HOST_CONTRACTS[name]
            if not callable(getattr(validation, symbol, None)):
                raise ValueError("actual host artifact validator is absent")
            schemas[name] = deepcopy(loader.load_schema(schema_name))
            path = Path(loader.__file__).resolve().parent / "schemas" / loader.SCHEMA_FILES[schema_name]
            if json.loads(path.read_bytes()) != schemas[name]:
                raise ValueError("loaded host schema differs from retained source")
            source_files[str(path)] = blake3_bytes(path.read_bytes())
        for module in (loader, validation):
            path = Path(module.__file__).resolve()
            source_files[str(path)] = blake3_bytes(path.read_bytes())
        return cls(public_tools=public, host_schemas=schemas, provenance={
            "source_policy_blake3": blake3_bytes(policy_bytes),
            "execution_tool_catalog_blake3": execution["tool_catalog_blake3"],
            "actual_host_source_files": source_files,
            "host_validators": {name: HOST_CONTRACTS[name][1] for name in public}})

    def _name(self, wire_name):
        for name in self.public_tools:
            if wire_name in {name, f"mcp__evamed__{name}"}:
                return name
        return None

    def validate_tools(self, tools):
        offered = {}
        for tool in tools:
            function = tool.get("function", {})
            name = self._name(function.get("name"))
            if name is None:
                continue
            expected = deepcopy(self.public_tools[name]["input_schema"])
            actual = deepcopy(function.get("parameters"))
            # The native wire's empty properties object is a structural default,
            # not an extra field constraint. Its exact bytes remain committed.
            if isinstance(actual, dict) and actual.get("properties") == {} and "properties" not in expected:
                actual.pop("properties")
            if actual != expected or function.get("description") != self.public_tools[name]["description"]:
                raise ValueError("offered tool differs from parser's host-bound canonical policy")
            if function["name"] in offered:
                raise ValueError("duplicate host-bound tool offer")
            offered[function["name"]] = name
        return offered

    def project_message(self, message, tools):
        offered = self.validate_tools(tools)
        projected = deepcopy(message)
        audit = {"schema": "eva.qwen-host-argument-projection.v1", "binding_blake3": self.binding_blake3,
                 "tools_blake3": blake3_hex(tools), "calls": [], "raw_values_retained_by_caller": True}
        for call in projected.get("tool_calls") or []:
            function = call.get("function", {})
            name = offered.get(function.get("name"))
            if name is None:
                continue
            raw = function.get("arguments")
            try:
                values = json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
            except (ValueError, TypeError):
                continue  # Preserve malformed model arguments for actual host rejection.
            if not isinstance(values, dict):
                continue
            changes = []
            for field, value in list(values.items()):
                decoded = _decode(value, set(self.hints[name].get(field, [])))
                if type(decoded) is not type(value):
                    values[field] = decoded
                    changes.append({"field": field, "from": _kind(value), "to": _kind(decoded),
                                    "raw_value_blake3": blake3_hex(value)})
            if changes:
                function["arguments"] = json.dumps(values, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            audit["calls"].append({"tool": name, "call_id": call.get("id"), "changes": changes,
                "before_arguments_blake3": blake3_hex(raw),
                "after_arguments_blake3": blake3_hex(function["arguments"])})
        return projected, audit


class HostTypedChatTransport:
    """Optional Chat counterpart. Retain raw response BEFORE typed projection."""
    def __init__(self, upstream, projector, retain):
        self.upstream, self.projector, self.retain = upstream, projector, retain

    def __call__(self, binding, body):
        from eva_agent.codex_providers.adapter import _UpstreamOutcome
        tools = body.get("tools") or []
        self.projector.validate_tools(tools)
        outcome = self.upstream(binding, body)
        if outcome.status != 200:
            return outcome
        response = json.loads(outcome.body)
        audits = []
        for choice in response.get("choices") or []:
            choice["message"], audit = self.projector.project_message(choice["message"], tools)
            audits.append(audit)
        # Caller must durably retain private raw bytes; a failing retention sink
        # fails this request rather than returning an uncommitted transformation.
        self.retain(outcome.body, {"binding": self.projector.binding,
            "request_blake3": blake3_hex(body), "raw_response_blake3": blake3_bytes(outcome.body),
            "projections": audits})
        return _UpstreamOutcome(outcome.status, json.dumps(response, ensure_ascii=False,
            allow_nan=False).encode(), outcome.latency_ms)
