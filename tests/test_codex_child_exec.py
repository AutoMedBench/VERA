from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from eva_agent.deployment.codex_child_exec import (
    CodexChildExecError,
    build_sanitized_codex_exec_plan,
    sanitized_child_environment,
)


WRAPPER = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "eva_agent"
    / "deployment"
    / "codex_child_exec.py"
)


def _private_root(tmp_path: Path) -> Path:
    root = tmp_path / "isolated"
    root.mkdir(mode=0o700)
    return root


def _codex_bin() -> Path:
    from codex_cli_bin import bundled_codex_path

    return bundled_codex_path().resolve()


def _plan(tmp_path: Path, *, names: tuple[str, ...] = ("NVIDIA_INFERENCE_API_KEY",)):
    return build_sanitized_codex_exec_plan(
        python_bin=Path(sys.executable).resolve(),
        wrapper_script=WRAPPER,
        codex_bin=_codex_bin(),
        isolation_root=_private_root(tmp_path),
        credential_env_names=names,
        codex_args=(
            "--config",
            "project_doc_max_bytes=0",
            "--config",
            "features.shell_tool=false",
            "app-server",
            "--strict-config",
            "--listen",
            "stdio://",
        ),
    )


def test_sanitized_environment_is_exact_and_secret_values_never_enter_plan(
    tmp_path: Path,
) -> None:
    route_secret = "test-route-secret-6d5f"
    unrelated_secret = "must-not-survive"
    plan = _plan(tmp_path)
    environment = sanitized_child_environment(
        isolation_root=plan.isolation_root,
        credential_env_names=plan.credential_env_names,
        source_environment={
            "NVIDIA_INFERENCE_API_KEY": route_secret,
            "GITHUB_TOKEN": unrelated_secret,
            "PYTHONPATH": "/attacker",
            "LD_PRELOAD": "/attacker.so",
        },
    )

    assert environment["NVIDIA_INFERENCE_API_KEY"] == route_secret
    assert "GITHUB_TOKEN" not in environment
    assert "PYTHONPATH" not in environment
    assert "LD_PRELOAD" not in environment
    assert environment["CODEX_HOME"].startswith(plan.isolation_root)
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert route_secret not in repr(plan)
    assert route_secret not in "\x00".join(plan.launch_args)
    metadata = plan.public_metadata()
    assert metadata["credential_values_recorded"] is False
    assert metadata["inherited_parent_environment"] is False


@pytest.mark.parametrize(
    "name",
    (
        "HOME",
        "CODEX_HOME",
        "PATH",
        "PYTHONPATH",
        "LD_PRELOAD",
        "RUST_LOG",
        "lowercase_api_key",
        "UNRELATED_VALUE",
    ),
)
def test_unsafe_child_environment_names_fail_closed(tmp_path: Path, name: str) -> None:
    with pytest.raises(CodexChildExecError, match="environment name"):
        _plan(tmp_path, names=(name,))


def test_only_strict_stdio_app_server_argv_is_accepted(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    common = {
        "python_bin": Path(sys.executable).resolve(),
        "wrapper_script": WRAPPER,
        "codex_bin": _codex_bin(),
        "isolation_root": root,
        "credential_env_names": ("NVIDIA_INFERENCE_API_KEY",),
    }
    with pytest.raises(CodexChildExecError, match="strict stdio"):
        build_sanitized_codex_exec_plan(
            **common,
            codex_args=("app-server", "--listen", "unix:///tmp/not-allowed"),
        )
    with pytest.raises(CodexChildExecError, match="global argv"):
        build_sanitized_codex_exec_plan(
            **common,
            codex_args=("--disable", "shell_tool", "app-server", "--strict-config", "--listen", "stdio://"),
        )


def test_published_sdk_initializes_through_sanitized_exec_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openai_codex import Codex, CodexConfig

    plan = _plan(tmp_path)
    route_secret = "provider-free-initialize-only"
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-be-forwarded")
    config = CodexConfig(
        launch_args_override=plan.launch_args,
        cwd=str(tmp_path),
        env={"NVIDIA_INFERENCE_API_KEY": route_secret},
        client_name="eva_sanitized_exec_test",
        client_title="EVA Sanitized Exec Test",
        client_version="0.1.0",
    )
    client = Codex(config=config)
    try:
        client.__enter__()
        assert client.metadata is not None
        assert client.metadata.serverInfo.version.startswith("0.147.0 ")
        process = client._client._proc
        assert process is not None
        child_environment = {
            name: value
            for entry in Path(f"/proc/{process.pid}/environ").read_bytes().split(b"\0")
            if entry
            for name, value in (entry.decode("utf-8").split("=", 1),)
        }
        assert child_environment["NVIDIA_INFERENCE_API_KEY"] == route_secret
        assert "GITHUB_TOKEN" not in child_environment
        assert "PYTHONPATH" not in child_environment
        assert "LD_PRELOAD" not in child_environment
        assert set(child_environment) == {
            "CODEX_HOME",
            "GIT_CONFIG_GLOBAL",
            "GIT_CONFIG_NOSYSTEM",
            "LANG",
            "LC_ALL",
            "NVIDIA_INFERENCE_API_KEY",
            "PATH",
            "TMPDIR",
            "TZ",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
        }
    finally:
        client.__exit__(None, None, None)

    assert not any(
        part.startswith("provider-free-initialize-only") for part in plan.launch_args
    )
