"""One real Codex protocol fixture; all model replies are local mock replies.

The Rust Codex parser, not this test or the MCP server, produces the invalid
argument error. This tests continuation, not a model's ability to self-correct.
No benchmark, GPU, external provider, ambient configuration or API key is used.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


SCHEMA = {"type": "object", "additionalProperties": False, "required": ["sigma"],
          "properties": {"sigma": {"type": "number", "minimum": 0}}}
DESCRIPTION = "Exact test tool."
BAD_ARGUMENTS = '{"sigma":NaN}'
DEFAULT_BINARY = Path("/localhome/local-operator/operator_GB300-2/.tools/"
    "codex-0.153.4-v1/package/vendor/aarch64-unknown-linux-musl/bin/codex")


def _mcp_server(journal):
    """Minimal canonical MCP fixture; never manufactures a parser error."""
    for line in sys.stdin:
        message = json.loads(line)
        if "id" not in message:
            continue
        method = message.get("method")
        if method == "initialize":
            result = {"protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "evamed-native-history-fixture", "version": "1.0"}}
        elif method == "tools/list":
            result = {"tools": [{"name": "fixture_tool", "description": DESCRIPTION, "inputSchema": SCHEMA}]}
        elif method == "tools/call":
            # Only a real MCP invocation is recorded. The native parser must
            # reject NaN before this handler is ever called.
            with Path(journal).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(message["params"], allow_nan=False) + "\n")
            result = {"content": [{"type": "text", "text": "Fixture accepted sigma=0.1"}], "isError": False}
        else:
            result = {}
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)


def _transcripts(chat, prefix):
    rows = []
    for message in chat["messages"]:
        content = message.get("content")
        if isinstance(content, str) and prefix in content:
            for part in content.split(prefix)[1:]:
                rows.append(json.loads(part.strip()))
    return rows


def test_real_codex_parser_invalid_then_valid_in_one_conversation(tmp_path):
    from eva_agent.codex_providers import adapter as module
    from eva_agent.codex_providers.canary import CodexExecLaunch
    from eva_agent.codex_runtime import CodexToolOffer
    from eva_agent.deployment.campaign import CODEX_FIRST_RELEASE_CONFIG_OVERRIDES
    from eva_agent.pipeline.digests import canonical_json_bytes
    from test_codex_provider_routes import _chat_response, _routes

    binary = Path(os.environ.get("EVA_NATIVE_CODEX_BIN", str(DEFAULT_BINARY)))
    if not binary.is_file():
        pytest.skip("Pinned local Codex 0.153.4 is unavailable")
    private = tmp_path / "codex-private"
    workspace = tmp_path / "workspace"
    private.mkdir()
    workspace.mkdir()
    environment = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ}
    environment.update(CODEX_HOME=str(private), RUST_LOG="off", PYTHONDONTWRITEBYTECODE="1")
    version = subprocess.run([str(binary), "--version"], env=environment, cwd=workspace,
        capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    assert version == "codex-cli 0.153.4"
    source = _routes(tmp_path)["qwen_3_6_27b"]
    captured = []

    def upstream(binding, body):
        captured.append(deepcopy(body))
        ordinal = len(captured)
        if ordinal <= 2:
            tools = [row["function"] for row in body.get("tools", [])
                     if row["function"]["name"].endswith("fixture_tool")]
            assert len(tools) == 1
            assert canonical_json_bytes(tools[0]["parameters"]) == canonical_json_bytes(SCHEMA)
            assert tools[0]["description"] == DESCRIPTION
            call = {"id": "call-invalid" if ordinal == 1 else "call-valid", "type": "function",
                    "function": {"name": tools[0]["name"],
                                 "arguments": BAD_ARGUMENTS if ordinal == 1 else '{"sigma":0.1}'}}
            value = _chat_response(binding.model_id, content=None, tool_calls=[call])
        else:
            assert ordinal == 3
            value = _chat_response(binding.model_id, content="NATIVE_FIXTURE_COMPLETE")
        return module._UpstreamOutcome(200, json.dumps(value).encode(), 1)

    journal = tmp_path / "mcp-executions.jsonl"
    with module.ResponsesAdapterGateway({source.route_id: source}, upstream_transport=upstream,
        local_bearer_token="native-fixture-local-only", preserve_qwen_tool_schemas=True,
        recover_completed_invalid_tool_history=True) as gateway:
        binding = gateway.bind_canonical_mcp_tools((CodexToolOffer("evamed/fixture_tool", DESCRIPTION, SCHEMA),))
        adapted = gateway.adapted_routes()[source.route_id]
        launch = CodexExecLaunch(adapted, str(binary), str(workspace), 45)
        argv, _, _ = launch.for_subprocess()
        _, synthetic_auth, _ = adapted.config.for_subprocess()
        environment.update(synthetic_auth)
        config = {
            "features.shell_tool": "false", "features.apply_patch_freeform": "false",
            "features.code_mode": "false", "features.tool_search": "false",
            "mcp_servers.evamed.command": json.dumps(sys.executable),
            "mcp_servers.evamed.args": json.dumps(["-B", str(Path(__file__).resolve()), "--fixture-mcp", str(journal)]),
            "mcp_servers.evamed.startup_timeout_sec": "10",
            "mcp_servers.evamed.tool_timeout_sec": "10",
            "mcp_servers.evamed.omit_tools_from": '["deferred", "code_mode"]',
            "mcp_servers.evamed.tools.fixture_tool.approval_mode": '"approve"',
        }
        command = list(argv[:-1])
        for override in CODEX_FIRST_RELEASE_CONFIG_OVERRIDES:
            command += ["--config", override]
        for key, value in config.items():
            command += ["--config", f"{key}={value}"]
        command.append("Use the fixture tool, correct invalid arguments if rejected, then report completion.")
        child = subprocess.Popen(command, env=environment, cwd=workspace,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            stdout, stderr = child.communicate(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate(timeout=5)
            pytest.fail("Native fixture exceeded 45 seconds; owned fixture process group stopped")
        assert child.returncode == 0, (stderr.decode(errors="replace")[-1000:], stdout.decode(errors="replace")[-2000:])
        assert len(captured) == 3
        assert journal.is_file(), [row.get("content") for row in captured[-1]["messages"] if row.get("role") == "tool"]
        executed = [json.loads(line) for line in journal.read_text().splitlines()]
        assert len(executed) == 1
        assert executed[0]["name"] == "fixture_tool" and executed[0]["arguments"] == {"sigma": 0.1}
        assert executed[0]["_meta"]["callId"] == "call-valid"
        history = _transcripts(captured[1], module.TOOL_HISTORY_TRANSCRIPT_PREFIX)
        bad = next(row for row in history if row.get("type") == "function_call")
        failure = next(row for row in history if row.get("type") == "function_call_output")
        assert bad["arguments"] == BAD_ARGUMENTS
        assert bad["call_id"] == failure["call_id"] == "call-invalid"
        native_error = module._tool_output_text(failure["output"])
        assert "err: expected value at line 1 column 10" in native_error
        events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
        assert len([row for row in events if row.get("type") == "thread.started"]) == 1
        assert any(row.get("type") == "turn.completed" for row in events)
        assert any(row.get("item", {}).get("text") == "NATIVE_FIXTURE_COMPLETE" for row in events)
        assert len(gateway.receipts) == 3
        for receipt in gateway.receipts:
            module.verify_adapter_receipt(receipt)
            assert receipt.payload["status"] == "passed"
            assert receipt.payload["projection_version"] == binding["projection_version"]
            assert receipt.payload["upstream_request_max_retries"] == 0


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--fixture-mcp":
    _mcp_server(sys.argv[2])
