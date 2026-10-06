from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def validation():
    script = Path(__file__).resolve().parents[1] / "scripts/validate_local_qwen_codex_v1.py"
    spec = importlib.util.spec_from_file_location("local_qwen_validation_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("nested", [False, True])
def test_local_diagnostics_retains_only_sanitized_error_metadata(validation, tmp_path, nested):
    diagnostic_path = tmp_path / "diagnostics.jsonl"
    transport = validation.LocalDiagnosticTransport(diagnostic_path, 30910)
    error = {"message": "input 13095 plus 4096 exceeds 16384 Bearer fixture-private-token", "private": "not retained"}
    payload = {"error": error} if nested else dict(error)
    payload["usage"] = {"prompt_tokens": 13095, "completion_tokens": 0, "other": "not retained"}
    payload["choices"] = [{"message": {"content": "not retained"}}]
    outcome = SimpleNamespace(status=400, body=json.dumps(payload).encode(), latency_ms=1)
    request = {"max_tokens": 4096, "messages": [{"content": "private input"}], "tools": []}
    seen = []
    transport.transport = lambda binding, body: seen.append(body) or outcome
    binding = SimpleNamespace(reveal_upstream_boundary=lambda: ("http://127.0.0.1:30910/v1", "local-nonsecret"))

    assert transport(binding, request) is outcome
    assert seen == [request]
    diagnostic = json.loads(diagnostic_path.read_text())
    assert diagnostic["error_message"] == "input 13095 plus 4096 exceeds 16384 [redacted]"
    assert diagnostic["usage"] == {"prompt_tokens": 13095, "completion_tokens": 0}
    assert diagnostic["request_or_response_body_retained"] is False
    assert "private" not in diagnostic_path.read_text()
    assert "not retained" not in diagnostic_path.read_text()


@pytest.mark.parametrize("endpoint,credential", [
    ("https://example.com/v1", "local-nonsecret"),
    ("http://127.0.0.1:30911/v1", "local-nonsecret"),
    ("http://127.0.0.1:30910/v1", "actual-private-credential"),
])
def test_local_diagnostics_rejects_other_provider_boundaries(validation, tmp_path, endpoint, credential):
    transport = validation.LocalDiagnosticTransport(tmp_path / "diagnostics.jsonl", 30910)
    transport.transport = lambda *args: pytest.fail("disallowed provider must not be called")
    binding = SimpleNamespace(reveal_upstream_boundary=lambda: (endpoint, credential))
    with pytest.raises(ValueError, match="loopback route"):
        transport(binding, {})
    assert not (tmp_path / "diagnostics.jsonl").exists()


def test_local_options_preserve_stage_guidance_and_tools(validation, monkeypatch, tmp_path):
    from eva_agent.codex_runtime import CodexRole, CodexSandbox, CodexThreadOptions

    original = CodexThreadOptions(
        role=CodexRole.STRONG_ACTOR, model="Qwen/Qwen3.5-9B", provider="local",
        cwd=str(tmp_path), sandbox=CodexSandbox.READ_ONLY,
        config={"project_doc_max_bytes": 0}, developer_instructions="exact canonical stage guidance",
    )
    monkeypatch.setattr(validation.PersistentCodexTeacherPool, "_options_factory",
                        staticmethod(lambda *_: lambda *_: original))
    validation.LocalQwenTeacherPool.compact_local_base = True
    validation.LocalQwenTeacherPool.local_context_length = 32768
    revised = validation.LocalQwenTeacherPool._options_factory(None, None)(None, None, (), None)
    assert revised.developer_instructions == original.developer_instructions
    assert revised.offered_tools == original.offered_tools
    assert revised.config == {"project_doc_max_bytes": 0, "model_context_window": 32768}
    assert "ONLY its arguments value" in revised.base_instructions


@pytest.mark.parametrize("tamper_artifact", [False, True])
def test_real_retained_canary_audit_reopens_receipts_and_artifact(validation, tmp_path, tamper_artifact):
    import shutil

    source_root = Path(__file__).resolve().parents[1] / "runs/qwen35-9b-codex-compatibility-20260910.v5"
    sandbox_id = "05381f58-7ed4-5314-8d43-3b098c2ee0d7"
    task_id = f"{sandbox_id}--qwen_3_5_9b"
    source_relative = Path("errors") / f"{task_id}.provider-failure.json"
    artifact_relative = Path("workspaces") / task_id / sandbox_id / "work/stage-plan.json"
    if not (source_root / source_relative).is_file():
        pytest.skip("optional local real-canary integration source is unavailable")
    for relative in (source_relative, artifact_relative):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, tmp_path / relative)
    if tamper_artifact:
        artifact = json.loads((tmp_path / artifact_relative).read_text())
        artifact["objective"] = "different observed artifact"
        (tmp_path / artifact_relative).write_text(json.dumps(artifact))
        with pytest.raises(ValueError, match="differs from actual passing tool arguments"):
            validation.audit_workspace_canary(tmp_path, sandbox_id)
    else:
        receipt = validation.audit_workspace_canary(tmp_path, sandbox_id)
        assert receipt["tool_workspace_compatibility_passed"] is True
        assert receipt["all_nested_codex_commitments_verified"] is True
        assert receipt["full_turn_completed"] is False
        assert receipt["training_admission"] is False
