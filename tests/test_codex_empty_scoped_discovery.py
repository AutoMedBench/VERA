"""Empty built-in discovery is model behavior, not an uncommitted data read."""
import json
from types import SimpleNamespace

import pytest

from eva_agent.codex_pipeline.adapter import CodexPipelineError, _validate_codex_core_resource_call


@pytest.mark.parametrize("operation,key", [("list_mcp_resources", "resources"),
                                          ("list_mcp_resource_templates", "resourceTemplates")])
def test_actual_empty_scoped_listing_preserves_no_resource_access(operation, key):
    receipt = SimpleNamespace(offered_mcp_tool_names=("automed_eval/search_skills",))
    document = {"server": "automed_eval", key: []}
    output = {"error": None, "result": {"_meta": None, "structuredContent": None,
              "content": [{"type": "text", "text": json.dumps(document)}]}}
    call = SimpleNamespace(name=operation, lifecycle=("item/started", "item/completed"),
        status="completed", output=output, arguments={"server": "automed_eval"},
        mcp_server="automed_eval", mcp_tool=operation)
    _validate_codex_core_resource_call(receipt, call, operation=operation)
    for changed in ({"server": "other", key: []}, {"server": "automed_eval", key: [{"uri": "private://x"}]}):
        output["result"]["content"][0]["text"] = json.dumps(changed)
        with pytest.raises(CodexPipelineError, match="uncommitted resources"):
            _validate_codex_core_resource_call(receipt, call, operation=operation)
