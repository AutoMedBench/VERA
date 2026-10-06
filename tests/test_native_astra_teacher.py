from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from eva_agent.pipeline import Cohort, ModelTarget, Stage
from eva_agent.training.frontier_prefix_sft import _route_authority
from eva_agent.training.native_astra_teacher import (
    NATIVE_ASTRA_MODEL, NATIVE_ASTRA_PROVIDER, NativeAstraTeacherPool,
    native_astra_launch_options, native_astra_model_catalog,
    native_astra_route_authorities, native_astra_thread_options,
    private_native_auth_copy,
)
from eva_agent.training.teacher_batch import TeacherBatchError


def _auth(tmp_path: Path) -> Path:
    path = tmp_path / "source-auth.json"
    path.write_text(json.dumps({
        "auth_mode": "chatgpt", "OPENAI_API_KEY": None,
        "tokens": {"access_token": "fixture-private-token", "refresh_token": "fixture-refresh"},
    }))
    return path


def test_auth_copy_is_private_ephemeral_and_preserves_source(tmp_path: Path) -> None:
    source = _auth(tmp_path)
    original = source.read_bytes()
    with private_native_auth_copy(source) as root:
        copy = root / "codex/auth.json"
        assert copy.read_bytes() == original
        assert root.stat().st_mode & 0o777 == 0o700
        assert copy.stat().st_mode & 0o777 == 0o600
        assert tmp_path not in root.parents
    assert not root.exists()
    assert source.read_bytes() == original


def test_auth_copy_rejects_symlink_and_api_auth(tmp_path: Path) -> None:
    source = _auth(tmp_path)
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(TeacherBatchError):
        with private_native_auth_copy(link):
            pytest.fail("symlink auth accepted")
    source.write_text(json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": "fixture"}))
    with pytest.raises(TeacherBatchError, match="ChatGPT"):
        with private_native_auth_copy(source):
            pytest.fail("API auth accepted")


def test_native_authority_and_catalog_bind_exact_model_without_fake_key() -> None:
    authority, provider, scope = _route_authority("gpt_6_astra", routes=native_astra_route_authorities())
    assert authority.model_id == NATIVE_ASTRA_MODEL
    assert provider == NATIVE_ASTRA_PROVIDER
    assert scope == "native_codex_direct"
    catalog = native_astra_model_catalog()
    assert catalog["models"][0]["slug"] == NATIVE_ASTRA_MODEL
    assert catalog["models"][0]["default_reasoning_level"] == "low"
    assert catalog["models"][0]["shell_type"] == "disabled"
    assert catalog["models"][0]["supports_parallel_tool_calls"] is True
    assert not hasattr(authority.config, "_credential")


def test_native_launch_uses_private_wrapper_and_no_custom_endpoint(tmp_path: Path) -> None:
    launch = native_astra_launch_options(
        codex_bin=Path("/bin/true"), script_path=tmp_path / "runner.py",
        isolation_root=tmp_path / "isolated", cwd=tmp_path,
        catalog_path=tmp_path / "models.json",
    )
    assert launch.env == {}
    argv = launch.launch_args_override
    assert "--native-child" in argv
    assert argv[-4:] == ("app-server", "--strict-config", "--listen", "stdio://")
    joined = " ".join(argv)
    assert "requires_openai_auth=true" in joined
    assert "request_max_retries=0" in joined
    assert "stream_max_retries=0" in joined
    assert "base_url" not in joined and "env_key" not in joined
    assert "fixture-private-token" not in repr(launch)


def test_native_child_discards_host_api_environment(tmp_path: Path, monkeypatch) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/run_native_astra_teacher_v1.py"
    spec = importlib.util.spec_from_file_location("native_astra_script_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    isolated = tmp_path / "isolation"
    isolated.mkdir(mode=0o700)
    monkeypatch.setenv("OPENAI_API_KEY", "host-secret")
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "other-host-secret")
    captured = []
    monkeypatch.setattr(os, "execve", lambda *args: captured.append(args))
    assert module.native_child_main([
        str(Path("/bin/true").resolve()), str(isolated),
        "--config", "project_doc_max_bytes=0", "app-server", "--strict-config", "--listen", "stdio://",
    ]) == 127
    environment = captured[0][2]
    assert "OPENAI_API_KEY" not in environment
    assert "NVIDIA_INFERENCE_API_KEY" not in environment
    assert "HOME" not in environment
    assert environment["CODEX_HOME"] == str(isolated / "codex")


def test_native_options_preserve_real_guidance_and_mcp_inventory() -> None:
    context = SimpleNamespace(stage_tool_guidance=SimpleNamespace(focus=Stage.S1, prompt_text="original stage guidance"))
    request = SimpleNamespace(model=ModelTarget(Cohort.STRONG, NATIVE_ASTRA_MODEL, NATIVE_ASTRA_PROVIDER))
    options = native_astra_thread_options(context)(request, "/tmp/medical-workspace", (), None)
    assert options.model == NATIVE_ASTRA_MODEL
    assert options.sandbox.value == "read-only"
    assert options.developer_instructions.startswith("original stage guidance")
    assert "stop tool execution immediately after materialize_plan" in options.developer_instructions
    assert "stage='S1'" in options.developer_instructions
    assert options.config["model_reasoning_effort"] == "low"
    assert "base_url" not in options.config["model_providers"][NATIVE_ASTRA_PROVIDER]


def test_native_pool_construction_does_not_open_auth_or_provider(tmp_path: Path) -> None:
    rows = [{"sandbox_id": "s", "candidate_id": "c", "stage": "S1", "domain": "d"}]
    pool = NativeAstraTeacherPool(rows=rows, output_root=tmp_path, codex_bin=Path("/bin/true"), auth_path=tmp_path / "missing")
    pool.close()
    with pytest.raises(TeacherBatchError, match="at most"):
        NativeAstraTeacherPool(rows=rows, output_root=tmp_path, workers=33)


def test_native_pool_supports_bounded64_fresh_tasks_without_provider(tmp_path: Path) -> None:
    rows = [{"sandbox_id": str(index), "candidate_id": str(index), "stage": "S1", "domain": "d"} for index in range(64)]
    pool = NativeAstraTeacherPool(rows=rows, output_root=tmp_path, workers=32, codex_bin=Path("/bin/true"), auth_path=tmp_path / "missing")
    pool.close()


def test_focus_only_s2_keeps_original_guidance_and_stops_at_selection() -> None:
    from eva_agent.training.teacher_focus import focus_only_teacher_instructions
    context = SimpleNamespace(stage_tool_guidance=SimpleNamespace(focus=Stage.S2, prompt_text="immutable S2 guidance"))
    result = focus_only_teacher_instructions(context)
    assert result.startswith("immutable S2 guidance")
    assert "stop tool execution immediately after materialize_evidence_selection" in result
    assert "stage='S2'" in result


def test_native_s2_requires_explicit_stage_and_preserves_s1_default(tmp_path: Path) -> None:
    rows = [{"sandbox_id": "s", "candidate_id": "c", "stage": "S2", "domain": "d"}]
    with pytest.raises(TeacherBatchError, match="stage binding"):
        NativeAstraTeacherPool(rows=rows, output_root=tmp_path, codex_bin=Path("/bin/true"))
    pool = NativeAstraTeacherPool(rows=rows, output_root=tmp_path, stage=Stage.S2, codex_bin=Path("/bin/true"))
    pool.close()


@pytest.mark.parametrize("stage,expected", [
    (Stage.S1, "9b06815f2a93f257c39231746c20b5af31effbe8675bbfe80617aa77f60ad30a"),
    (Stage.S2, "c37ac8fae3d66e4e031260841a6dd6ee83ebff2fba1f360fb1e3f0cd473a7e27"),
])
def test_s3_addition_preserves_existing_s1_s2_focus_text_bytes(stage, expected):
    from eva_agent.pipeline.digests import blake3_bytes
    from eva_agent.training.teacher_focus import focus_only_teacher_instructions
    context = SimpleNamespace(stage_tool_guidance=SimpleNamespace(focus=stage, prompt_text="fixed guidance"))
    assert blake3_bytes(focus_only_teacher_instructions(context).encode()) == expected


def test_native_s3_is_explicit_readonly_and_only_completes_host_pilot(tmp_path: Path):
    rows = [{"sandbox_id": "s", "candidate_id": "c", "stage": "S3", "domain": "d"}]
    with pytest.raises(TeacherBatchError, match="stage binding"):
        NativeAstraTeacherPool(rows=rows, output_root=tmp_path, codex_bin=Path("/bin/true"))
    pool = NativeAstraTeacherPool(rows=rows, output_root=tmp_path, stage=Stage.S3, codex_bin=Path("/bin/true"))
    pool.close()
    context = SimpleNamespace(stage_tool_guidance=SimpleNamespace(focus=Stage.S3, prompt_text="immutable S3 guidance"))
    request = SimpleNamespace(model=ModelTarget(Cohort.STRONG, NATIVE_ASTRA_MODEL, NATIVE_ASTRA_PROVIDER))
    options = native_astra_thread_options(context)(request, str(tmp_path), (), None)
    assert options.sandbox.value == "read-only"
    assert options.developer_instructions.startswith("immutable S3 guidance")
    assert "execute_code MCP tool with stage='S3'" in options.developer_instructions
    assert "Stop tool execution immediately after execute_code for S3" in options.developer_instructions
    assert "call execute_code with stage='S4'" in options.developer_instructions
    assert "every search_skills or load_skill call must pass stage='S3'" in options.developer_instructions


def test_s3_prompt_distinguishes_hydrated_actor_files_from_empty_execution_mount():
    from eva_agent.training.teacher_focus import focus_only_teacher_instructions
    context = SimpleNamespace(stage_tool_guidance=SimpleNamespace(focus=Stage.S3, prompt_text="fixed guidance"))
    prompt = focus_only_teacher_instructions(context)
    assert "verified the S1 and S2 prerequisites host-side" in prompt
    assert "fresh empty writable /workspace" in prompt
    assert "Read each declared input at /inputs/<contract-relative-path>" in prompt
    assert "Do not read hydrated work/stage-plan.json or work/evidence-selection.json" in prompt
    assert "Create the parent directories" in prompt
    assert "prerequisites in this workspace" not in prompt


def test_native_s3_export_fails_before_context_or_provider(tmp_path: Path):
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts/run_native_astra_teacher_v1.py"),
        "export", "--stage", "S3", "--cohort", str(tmp_path / "absent.json"),
        "--output-root", str(tmp_path / "absent")], cwd=root, capture_output=True, text=True)
    assert result.returncode == 2
    assert "S3 export is not implemented" in result.stderr
    assert not (tmp_path / "absent").exists()


def test_s3_prompt_does_not_authorize_extra_diagnostic_files():
    from eva_agent.training.teacher_focus import focus_only_teacher_instructions
    context = SimpleNamespace(stage_tool_guidance=SimpleNamespace(focus=Stage.S3, prompt_text="fixed guidance"))
    prompt = focus_only_teacher_instructions(context)
    assert "exactly the output file or files declared by the bound contract" in prompt
    assert "no extra files in /workspace" in prompt
    assert "Do not write diagnostic, trace, scratch, or provenance sidecar files" in prompt
    assert "keep such diagnostics in memory or stdout" in prompt


def test_native_auth_copy_ignores_artifact_local_temp_override(tmp_path: Path, monkeypatch) -> None:
    import tempfile
    artifact_temp = tmp_path / "published-artifacts"
    artifact_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(artifact_temp))
    with private_native_auth_copy(_auth(tmp_path)) as isolated:
        assert artifact_temp not in isolated.parents
        assert isolated.parent == Path("/tmp").resolve()
