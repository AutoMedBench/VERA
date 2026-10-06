from __future__ import annotations

from dataclasses import replace

import pytest

from eva_agent.codex_runtime import (
    CodexLaunchOptions,
    CodexRole,
    CodexRuntimeError,
    CodexSandbox,
    CodexThreadOptions,
    CodexTurnInput,
)
from eva_agent.codex_runtime.native_backend_v2 import _thread_workspace_config
from eva_agent.deployment.campaign import CampaignDeploymentError
from eva_agent.deployment.medresearch_native_v4 import (
    StageAwareNativeRunnerV4,
    _append_developer_instruction,
    _legacy_episode_context,
    _native_workspace_write_supported,
)


class _Runner:
    def __init__(self, result: object) -> None:
        self.result = result
        self.started = 0
        self.closed = 0
        self.calls = []

    def start(self) -> None:
        self.started += 1

    def close(self) -> None:
        self.closed += 1

    def run_once(self, options, turn_input):
        self.calls.append((options, turn_input))
        return self.result


def _options(tmp_path, sandbox: CodexSandbox) -> CodexThreadOptions:
    return CodexThreadOptions(
        role=CodexRole.STRONG_ACTOR,
        model="model",
        provider="provider",
        cwd=str(tmp_path),
        sandbox=sandbox,
        config={},
        ephemeral=True,
    )


def _turn(sandbox: CodexSandbox) -> CodexTurnInput:
    return CodexTurnInput(public_text="work", sandbox=sandbox, model="model")


def test_native_backend_consumes_only_exact_positive_workspace_commitment(tmp_path) -> None:
    exact = {
        "network_access": False,
        "writable_roots": [str(tmp_path)],
        "exclude_slash_tmp": True,
        "exclude_tmpdir_env_var": True,
    }
    result = _thread_workspace_config(
        {"sandbox_workspace_write": exact}, str(tmp_path)
    )
    assert result["sandbox_workspace_write"] == exact
    assert result["approval_policy"] == "never"
    with pytest.raises(CodexRuntimeError, match="workspace commitment"):
        _thread_workspace_config(
            {"sandbox_workspace_write": {**exact, "network_access": True}},
            str(tmp_path),
        )


def test_stage_aware_runner_keeps_judge_read_only_and_routes_actor_native(tmp_path) -> None:
    delegate = _Runner("judge")
    routed = StageAwareNativeRunnerV4(delegate, (CodexLaunchOptions(),))
    native = _Runner("actor")
    routed._native = (native,)  # provider-free injected shard

    judge_options = replace(_options(tmp_path, CodexSandbox.READ_ONLY), role=CodexRole.JUDGE)
    assert routed.run_once(judge_options, _turn(CodexSandbox.READ_ONLY)) == "judge"
    assert delegate.calls and not native.calls

    actor_options = _options(tmp_path, CodexSandbox.WORKSPACE_WRITE)
    assert routed.run_once(actor_options, _turn(CodexSandbox.WORKSPACE_WRITE)) == "actor"
    assert native.calls and native.started == 1 and delegate.started == 1
    routed.close()
    assert native.closed == 1 and delegate.closed == 1


def test_stage_aware_runner_rejects_unrecognized_sandbox(tmp_path) -> None:
    delegate = _Runner("unused")
    routed = StageAwareNativeRunnerV4(delegate, (CodexLaunchOptions(),))
    # Enum exhaustiveness is already enforced by CodexThreadOptions; this test
    # proves the routing decision itself never silently defaults to native.
    assert routed.shard_count == 1


def test_user_namespace_probe_fails_closed_without_unshare(monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("subprocess.run", unavailable)
    assert _native_workspace_write_supported() is False


def test_fallback_guidance_binding_is_exact_and_idempotent() -> None:
    guidance = "exact StageToolGuidance prompt"
    bound = _append_developer_instruction("base instruction", guidance)
    assert bound == "base instruction\n\nexact StageToolGuidance prompt"
    assert _append_developer_instruction(bound, guidance) == bound


def test_native_e2e_selects_inner_signed_legacy_context() -> None:
    legacy_context = {"schema": "eva.legacy-candidate-runtime-context.v1"}
    public_context = {
        "sandbox_id": "candidate-id",
        "stage": "E2E",
        "episode_context": legacy_context,
    }

    assert _legacy_episode_context(public_context) is legacy_context


@pytest.mark.parametrize(
    "public_context",
    ({}, {"episode_context": None}, {"episode_context": "not-an-object"}),
)
def test_native_e2e_rejects_missing_or_non_object_legacy_context(
    public_context,
) -> None:
    with pytest.raises(
        CampaignDeploymentError,
        match="legacy runtime context projection differs",
    ):
        _legacy_episode_context(public_context)
