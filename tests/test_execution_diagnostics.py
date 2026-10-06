from __future__ import annotations

import json
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from eva_agent.codex_pipeline import CodexPipelineError, CodexToolExecutionBridge
from eva_agent.pipeline import DeterministicUUIDFactory, FilesystemSandbox, ParallelToolRuntime, ToolDefinition, ToolRegistry
from eva_agent.pipeline.digests import blake3_bytes, canonical_value
from eva_agent.pipeline.execution_diagnostics import (
    DIAGNOSTIC_PREFIX, MAXIMUM_CONTENT_BYTES, diagnostic_content,
    diagnostic_scope, diagnostics_requested, record_execution_diagnostics,
)


def _capture(stdout="", stderr="Traceback: ValueError: missing task artifact"):
    return {
        name: {"text": text, "byte_count": len(text.encode()), "truncated": False}
        for name, text in (("stdout", stdout), ("stderr", stderr))
    }


def _host(tmp_path: Path, name: str) -> Path:
    root = tmp_path / "host-executions" / name
    root.mkdir(parents=True)
    (root / "host-receipt.json").write_bytes(b'{"signed":"fixture"}\n')
    return root


def test_capture_is_sanitized_bounded_and_display_is_distinct(tmp_path):
    episode = _host(tmp_path, "failure")
    stderr = ('Authorization: Bearer dummy-value-A\n'
              'Authorization: Basic dummy-value-B\n'
              '"api_key": "dummy-value-C"\n'
              '/localhome/private/project/task.py\n' + '诊断' * 1700)
    capture = _capture("output " * 1200, stderr)
    with diagnostic_scope("call-fixture", "execute_code") as scope:
        assert record_execution_diagnostics(capture, episode_dir=episode, stage="S3", attempt=1)
    persisted = json.loads((episode / "execution-diagnostics.json").read_bytes())
    assert (episode / "execution-diagnostics.json").stat().st_mode & 0o777 == 0o400
    rendered = scope.content[0]["text"]
    assert len(rendered.encode()) <= MAXIMUM_CONTENT_BYTES
    shown = json.loads(rendered[len(DIAGNOSTIC_PREFIX):])
    assert shown["capture_blake3"] == persisted["diagnostic_blake3"]
    assert shown["host_receipt_blake3"] == blake3_bytes((episode / "host-receipt.json").read_bytes())
    assert shown["streams"]["stderr"]["text"]
    assert shown["streams"]["stdout"]["display_truncated"]
    assert shown["streams"]["stderr"]["display_truncated"]
    for name in ("stdout", "stderr"):
        assert len(persisted["streams"][name]["text"].encode()) <= 4096
        assert persisted["streams"][name]["byte_count"] == capture[name]["byte_count"]
    assert diagnostic_content({"result": {"content": scope.content}}, call_id="call-fixture") == scope.content
    with pytest.raises(ValueError, match="call binding"):
        diagnostic_content({"result": {"content": scope.content}}, call_id="another-call")


@pytest.mark.parametrize("secret_line", (
    "Authorization: Bearer dummy-value-A", "Authorization: Basic dummy-value-B",
    '"api_key": "dummy-value-C"', "access_token='dummy-value-D'",
))
def test_redaction_does_not_leave_authorization_value(tmp_path, secret_line):
    episode = _host(tmp_path, "redaction")
    with diagnostic_scope("call", "execute_code") as scope:
        assert record_execution_diagnostics(_capture(stderr=secret_line), episode_dir=episode, stage="S3", attempt=1)
    assert "dummy-value" not in scope.content[0]["text"]
    assert "dummy-value" not in (episode / "execution-diagnostics.json").read_text()
    assert "[redacted]" in scope.content[0]["text"]


def test_parallel_diagnostics_do_not_mix_or_change_canonical_result(tmp_path):
    barrier = Barrier(2)
    schema = {"type": "object", "properties": {"marker": {"type": "string"}},
              "required": ["marker"], "additionalProperties": False}

    def handler(workspace, arguments):
        assert diagnostics_requested()
        marker = arguments["marker"]
        episode = _host(tmp_path, marker)
        barrier.wait(timeout=3)
        assert record_execution_diagnostics(_capture(stderr="failure-" + marker), episode_dir=episode, stage="S3", attempt=1)
        return {"exit_code": 1, "gate_passed": False}

    registry = ToolRegistry((ToolDefinition("execute_code", "Execute.", schema, handler, parallel_safe=True, read_only=True),))
    workspace = FilesystemSandbox(tmp_path / "tasks", DeterministicUUIDFactory("diag-workspace").new("workspace"), {"seed.txt": b"seed"})
    runtime = ParallelToolRuntime(workspace=workspace, registry=registry, id_factory=DeterministicUUIDFactory("diag-tools"))
    bridge = CodexToolExecutionBridge(runtime, id_factory=DeterministicUUIDFactory("diag-bridge"))
    observations = bridge.execute_group((("execute_code", {"marker": "alpha"}), ("execute_code", {"marker": "beta"})))
    for observed, marker, other in zip(observations, ("alpha", "beta"), ("beta", "alpha"), strict=True):
        assert canonical_value(observed.result.output) == {"exit_code": 1, "gate_passed": False}
        assert set(observed.structured_content) == {"schema", "call_id", "name", "arguments", "tool_result", "bridge_receipt_blake3"}
        text = observed.supplemental_content[0]["text"]
        assert "failure-" + marker in text and "failure-" + other not in text
        assert diagnostic_content({"result": {"content": observed.supplemental_content}}, call_id=observed.call.call_id)
    assert runtime.trace().max_parallelism_observed == 2
    assert [row.path for row in workspace.snapshot("after").files] == ["seed.txt"]
    assert not diagnostics_requested()
    native_calls = tuple(SimpleNamespace(tool_call_id=f"native-{index}", mcp_tool="execute_code",
        arguments=observed.call.arguments, output={"result": {
            "structuredContent": observed.structured_content, "content": observed.supplemental_content}})
        for index, observed in enumerate(observations))
    mapping, trace = bridge.bind_receipt(None, (native_calls,))
    assert len(mapping) == 2 and trace == runtime.trace()
    native_calls[0].output["result"]["content"] = observations[1].supplemental_content
    with pytest.raises(CodexPipelineError, match="diagnostic differs"):
        bridge.bind_receipt(None, (native_calls,))
    native_calls[0].output["result"]["content"] = []
    with pytest.raises(CodexPipelineError, match="diagnostic differs"):
        bridge.bind_receipt(None, (native_calls,))


def test_absent_capture_never_fabricates_content(tmp_path):
    episode = _host(tmp_path, "absent")
    assert not record_execution_diagnostics(_capture(), episode_dir=episode, stage="S3", attempt=1)
    with diagnostic_scope("call", "execute_code") as scope:
        assert not record_execution_diagnostics({}, episode_dir=episode, stage="S3", attempt=1)
        assert scope.content == ()
    assert not (episode / "execution-diagnostics.json").exists()


@pytest.mark.parametrize("supports_capture", (False, True))
def test_legacy_backend_optional_capture_preserves_old_result(tmp_path, supports_capture):
    from eva_agent.sources.legacy_execution import _LegacyCandidateToolBackend

    calls = []

    def execute(contract, code, **kwargs):
        assert "diagnostics_sink" not in kwargs
        episode = kwargs["episode_dir"]
        episode.mkdir(parents=True)
        (episode / "host-receipt.json").write_bytes(b"signed fixture")
        calls.append((contract, code))
        return {"fixture": True}

    def execute_with_capture(contract, code, *, diagnostics_sink=None, **kwargs):
        receipt = execute(contract, code, **kwargs)
        assert diagnostics_sink is not None
        diagnostics_sink.update(_capture())
        return receipt

    backend = object.__new__(_LegacyCandidateToolBackend)
    backend.modules = SimpleNamespace(
        canonical=SimpleNamespace(sha256_value=lambda x: "prerequisite", sha256_file=lambda x: "receipt"),
        execution=SimpleNamespace(execute_python_contract=execute_with_capture if supports_capture else execute),
        host_receipts=SimpleNamespace(verify_host_receipt=lambda *a: {"observations": {"execution": {"gate_passed": False}}}),
        agent_rollout=SimpleNamespace(_tool_result_from_execution=lambda x: {"exit_code": 1, "gate_passed": False}),
    )
    for attr in ("evidence_root", "image_refs", "key_id", "private_key", "trust_store", "docker_binary"):
        setattr(backend, attr, None)
    session = SimpleNamespace(root=tmp_path / "host", s2_receipt={"signed": True}, s3_attempts=0, s3_template={})
    with diagnostic_scope("call", "execute_code") as scope:
        result = backend._handle_execute_code(None, session, {"stage": "S3", "code": "raise ValueError()"})
    assert result == {"exit_code": 1, "gate_passed": False, "attempt": 1, "host_receipt_sha256": "receipt"}
    assert len(calls) == 1
    assert bool(scope.content) is supports_capture
