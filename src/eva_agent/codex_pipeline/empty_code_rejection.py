"""Recognize one legacy input rejection without changing its canonical outcome."""
import inspect
import json

from jsonschema import Draft202012Validator

from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value


BOOTSTRAP_PATHS = (".eva/source-policy.json", ".eva/runtime-context.json")


def legacy_execute_code_runtime(tools):
    """A same-named tool or arbitrary handler must never receive this exception."""
    from eva_agent.pipeline.tools import ParallelToolRuntime, ToolRegistry
    from eva_agent.sources.legacy_execution import _LegacyCandidateToolBackend

    if type(tools) is not ParallelToolRuntime or type(tools._registry) is not ToolRegistry:
        return False
    try:
        handler = tools._registry.definition("execute_code").handler
        closure = inspect.getclosurevars(handler).nonlocals
        backend = closure.get("self")
        return (type(backend) is _LegacyCandidateToolBackend
                and closure.get("name") == "execute_code"
                and handler.__code__ is _LegacyCandidateToolBackend.handler(backend, "execute_code").__code__
                and "_handle_execute_code" not in vars(backend)
                and "_session" not in vars(backend))
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def empty_code_rejection_ids(*, receipt, trace, mapping, catalog, context, bootstrap):
    """Reopen source policy, native offer/call and original bridge/result bindings.

    The caller separately proves the live handler identity before opting in.
    Offline reopening uses the retained source/execution binding, not tool names
    alone. The existing metadata carries only the original catalog and call IDs.
    """
    from eva_agent.sources.legacy_execution import CandidateToolCatalogEntry

    try:
        execution = context["execution_binding"]
        policy_bytes, runtime_bytes = (bootstrap[path] for path in BOOTSTRAP_PATHS)
        policy, runtime = json.loads(policy_bytes), json.loads(runtime_bytes)
        source_catalog = sorted(policy["tools"], key=lambda row: row["name"])
        source_binding = [CandidateToolCatalogEntry(**row).to_document() for row in source_catalog]
        catalog = canonical_value(catalog)
        if (execution.get("schema") != "eva.legacy-candidate-runtime-context.v1"
                or canonical_value(execution) != runtime
                or policy.get("schema") != "rlevo.med-research-sandbox-policy.v2"
                or blake3_bytes(policy_bytes) != execution["policy_blake3"]
                or blake3_hex(source_binding) != execution["tool_catalog_blake3"]
                or not isinstance(catalog, list)
                or len(catalog) != len(receipt.offered_mcp_tool_names)
                or blake3_hex(catalog) != receipt.offered_tool_schema_blake3):
            return ()
        offers = dict(zip(receipt.offered_mcp_tool_names, catalog, strict=True))
        offer = offers.get("evamed/execute_code")
        source = [row for row in source_catalog if row["name"] == "execute_code"]
        if (len(source) != 1 or offer != source[0]
                or any(set(row) != {"name", "description", "input_schema"}
                       or name.rsplit("/", 1)[-1] != row["name"]
                       for name, row in offers.items())):
            return ()
        results = {row.call_id: row for row in trace.results}
        matched = []
        for call in receipt.tool_calls:
            result = results.get(mapping.get(call.tool_call_id))
            arguments = canonical_value(call.arguments)
            if (result is None or call.tool_type != "mcpToolCall"
                    or call.fully_qualified_name != "evamed/execute_code"
                    or call.mcp_server != "evamed" or call.mcp_tool != "execute_code"
                    or call.status != "completed" or result.name != "execute_code"
                    or not isinstance(arguments, dict) or set(arguments) != {"stage", "code"}
                    or arguments["stage"] not in {"S3", "S4"} or arguments["code"] != ""
                    or not Draft202012Validator(offer["input_schema"]).is_valid(arguments)
                    or result.status != "immutable_failure" or result.output is not None
                    or result.error_code != "ValueError"
                    or result.workspace_before_blake3 != result.workspace_after_blake3):
                continue
            native = canonical_value(call.output)
            core = {"schema": "eva.codex-pipeline-tool-observation.v1",
                    "call_id": result.call_id, "name": result.name,
                    "arguments": arguments, "tool_result": canonical_value(result)}
            response = native.get("result") if isinstance(native, dict) else None
            if (not isinstance(response, dict) or native.get("error") is not None
                    or response.get("isError", False) is not False
                    or response.get("structuredContent") != {
                        **core, "bridge_receipt_blake3": blake3_hex(core)}):
                continue
            matched.append(result.call_id)
        return tuple(matched)
    except (KeyError, TypeError, ValueError, AttributeError):
        return ()


def semantic_tool_failure(result, empty_code_ids=()):
    return (result.status == "immutable_failure" and result.output is None
            and (result.error_code == "ContractError" or result.call_id in empty_code_ids))
