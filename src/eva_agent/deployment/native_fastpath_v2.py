"""Provider-independent deployment binding for the Tier-A native fast path.

The module binds a :class:`~eva_agent.codex_pipeline.native_policy_v2.NativePolicyV2`
to exact actor/judge thread options without changing existing pipeline schemas.
It also reopens terminal Codex receipts so ``fileChange`` is the only accepted
native actor effect and the workspace-inspecting judge stays MCP/read-only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence

from eva_agent.codex_pipeline.native_policy_v2 import (
    NATIVE_POLICY_V2_ACTION_TYPES,
    NativePolicyV2,
    GuidedStageToolRuntimeV1,
    StageToolGuidanceV1,
    build_native_policy_v2,
    build_stage_tool_guidance_v1,
    guard_guided_stage_tool_runtime_v1,
    native_policy_v2_config_overrides,
    verify_native_policy_v2,
)
from eva_agent.codex_runtime import (
    CodexRole,
    CodexSandbox,
    CodexSkill,
    CodexThreadOptions,
    CodexToolCall,
    CodexTurnReceipt,
    codex_turn_receipt_from_document,
    verify_codex_turn_receipt,
)
from eva_agent.pipeline.contracts import Stage
from eva_agent.pipeline.digests import (
    blake3_bytes,
    blake3_hex,
    canonical_value,
    is_blake3,
)


NATIVE_FASTPATH_V2_SCHEMA = "eva.codex-native-fastpath-deployment.v2"
NATIVE_FASTPATH_V2_ACTOR_INSTRUCTION = (
    "This turn uses the verified Tier-A native fast path. Continue to use the exact "
    "offered EvaMed MCP tools and mounted SkillInput skills. The only Codex-native "
    "action permitted is a sequential, receipt-visible fileChange within the candidate "
    "workspace. Do not use commandExecution, shell or interpreter commands, network or "
    "web access, /tmp, extra writable roots, subagents, or thread resume."
)
NATIVE_FASTPATH_V2_JUDGE_INSTRUCTION = (
    "Inspect the committed context and workspace only through the offered read-only "
    "judge MCP tools. Do not emit any Codex-native action and do not reuse the actor thread."
)


class NativeFastpathV2Error(ValueError):
    """The Tier-A policy, thread binding, or terminal effect differed."""


def _clean_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise NativeFastpathV2Error(f"{label} differs")
    return value


def _real_directory(value: str, *, label: str) -> Path:
    _clean_text(value, label=label)
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise NativeFastpathV2Error(f"{label} must be an absolute normalized directory")
    try:
        entry = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise NativeFastpathV2Error(f"{label} is unavailable") from exc
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode) or resolved != path:
        raise NativeFastpathV2Error(f"{label} uses unsafe topology")
    return path


def _tool_catalog_blake3(options: CodexThreadOptions) -> str:
    return blake3_hex(
        tuple(tool.canonical_catalog_entry() for tool in options.offered_tools)
    )


def _reject_config_escape(
    value: Any,
    *,
    path: str = "config",
    actor_workspace_root: str | None = None,
) -> None:
    """Reject a caller's explicit attempt to contradict the sealed launch policy.

    The positive sandbox controls are committed separately by
    ``applied_config_overrides``.  This scan prevents a nested per-thread setting
    from silently widening them while leaving unrelated provider/MCP config intact.
    """

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold().replace("-", "_")
            compact = key.replace("_", "")
            location = f"{path}.{raw_key}"
            if compact in {"networkaccess", "networkenabled"} and child is not False:
                raise NativeFastpathV2Error(f"{location} attempts to enable network")
            if compact == "additionalwritableroots" and child not in ((), [], None):
                raise NativeFastpathV2Error(f"{location} attempts to add a writable root")
            if compact == "writableroots" and (
                actor_workspace_root is None
                or location != "config.sandbox_workspace_write.writable_roots"
                or not isinstance(child, (tuple, list))
                or tuple(child) != (actor_workspace_root,)
            ):
                raise NativeFastpathV2Error(f"{location} attempts to change writable roots")
            if compact in {"websearch", "standalonewebsearch"} and child not in (
                False,
                "disabled",
                None,
            ):
                raise NativeFastpathV2Error(f"{location} attempts to enable web search")
            if compact in {
                "shelltool",
                "unifiedexec",
                "commandexecution",
                "codemodehost",
                "browseruse",
                "computeruse",
            } and child not in (False, None):
                raise NativeFastpathV2Error(
                    f"{location} attempts to enable a forbidden native action"
                )
            if compact in {"resume", "resumethread", "threadresume"} and child not in (
                False,
                None,
            ):
                raise NativeFastpathV2Error(f"{location} attempts to enable thread resume")
            _reject_config_escape(
                child,
                path=location,
                actor_workspace_root=actor_workspace_root,
            )
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _reject_config_escape(
                child,
                path=f"{path}[{index}]",
                actor_workspace_root=actor_workspace_root,
            )


def _append_instruction(existing: str | None, required: str) -> str:
    if existing is None:
        return required
    if required in existing:
        return existing
    return f"{existing.rstrip()}\n\n{required}"


def _verify_actor_thread_config(options: CodexThreadOptions) -> None:
    sandbox = options.config.get("sandbox_workspace_write")
    features = options.config.get("features")
    if (
        not isinstance(sandbox, Mapping)
        or set(sandbox)
        != {
            "network_access",
            "writable_roots",
            "exclude_slash_tmp",
            "exclude_tmpdir_env_var",
        }
        or sandbox.get("network_access") is not False
        or tuple(sandbox.get("writable_roots", ())) != (options.cwd,)
        or sandbox.get("exclude_slash_tmp") is not True
        or sandbox.get("exclude_tmpdir_env_var") is not True
        or not isinstance(features, Mapping)
        or features.get("shell_tool") is not False
        or features.get("unified_exec") is not False
    ):
        raise NativeFastpathV2Error(
            "native actor thread config does not enforce its one-root offline sandbox"
        )


def _verify_judge_thread_config(options: CodexThreadOptions) -> None:
    features = options.config.get("features")
    if (
        not isinstance(features, Mapping)
        or features.get("shell_tool") is not False
        or features.get("unified_exec") is not False
        or "sandbox_workspace_write" in options.config
    ):
        raise NativeFastpathV2Error(
            "native judge config must disable command execution and every write sandbox"
        )


def bind_native_actor_options_v2(
    options: CodexThreadOptions,
    policy: NativePolicyV2,
    stage_tool_guidance: StageToolGuidanceV1 | None = None,
) -> CodexThreadOptions:
    """Return fresh workspace-write actor options carrying the Tier-A instruction."""

    verify_native_policy_v2(policy)
    if (
        not isinstance(options, CodexThreadOptions)
        or not options.role.is_actor
        or options.sandbox is not CodexSandbox.WORKSPACE_WRITE
        or options.ephemeral is not True
    ):
        raise NativeFastpathV2Error(
            "native actor requires fresh ephemeral workspace-write thread options"
        )
    _real_directory(options.cwd, label="native actor workspace")
    _reject_config_escape(options.config, actor_workspace_root=options.cwd)
    _verify_actor_thread_config(options)
    if (policy.stage is Stage.E2E) != (stage_tool_guidance is not None):
        raise NativeFastpathV2Error(
            "native E2E actor requires exactly one stage-tool guidance sidecar"
        )
    if stage_tool_guidance is not None and stage_tool_guidance.focus is not Stage.E2E:
        raise NativeFastpathV2Error("native E2E stage-tool guidance focus differs")
    required_instruction = NATIVE_FASTPATH_V2_ACTOR_INSTRUCTION
    if stage_tool_guidance is not None:
        required_instruction = (
            required_instruction + "\n\n" + stage_tool_guidance.prompt_text
        )
    return replace(
        options,
        developer_instructions=_append_instruction(
            options.developer_instructions, required_instruction
        ),
    )


def bind_native_judge_options_v2(options: CodexThreadOptions) -> CodexThreadOptions:
    """Return fresh read-only judge options with native actions explicitly forbidden."""

    if (
        not isinstance(options, CodexThreadOptions)
        or options.role is not CodexRole.JUDGE
        or options.sandbox is not CodexSandbox.READ_ONLY
        or options.ephemeral is not True
    ):
        raise NativeFastpathV2Error(
            "native fast-path judge requires fresh ephemeral read-only thread options"
        )
    _real_directory(options.cwd, label="native judge workspace")
    _reject_config_escape(options.config)
    _verify_judge_thread_config(options)
    if any(
        tool.visibility != "judge-only" or tool.read_only is not True
        for tool in options.offered_tools
    ):
        raise NativeFastpathV2Error(
            "native judge MCP inventory must be judge-only and read-only"
        )
    return replace(
        options,
        developer_instructions=_append_instruction(
            options.developer_instructions, NATIVE_FASTPATH_V2_JUDGE_INSTRUCTION
        ),
    )


def _deployment_core(
    *,
    candidate_id: str,
    policy: NativePolicyV2,
    actor: CodexThreadOptions,
    judge: CodexThreadOptions,
    overrides: tuple[str, ...],
    stage_tool_guidance: StageToolGuidanceV1 | None,
) -> dict[str, Any]:
    return {
        "schema": NATIVE_FASTPATH_V2_SCHEMA,
        "candidate_id": candidate_id,
        "stage": policy.stage.value,
        "policy_blake3": policy.policy_blake3,
        "actor": {
            "role": actor.role.value,
            "model": actor.model,
            "provider": actor.provider,
            "cwd": actor.cwd,
            "sandbox": actor.sandbox.value,
            "ephemeral": actor.ephemeral,
            "fresh_thread": True,
            "resume_allowed": False,
            "config_keys": list(actor.config_keys),
            "config_blake3": blake3_hex(actor.config),
            "offered_mcp_tool_names": [
                tool.fully_qualified_name for tool in actor.offered_tools
            ],
            "offered_tool_schema_blake3": _tool_catalog_blake3(actor),
            "base_instructions_blake3": blake3_hex(actor.base_instructions),
            "developer_instructions_blake3": blake3_hex(
                actor.developer_instructions
            ),
            "skill_input_ids": [skill.skill_id for skill in policy.skills],
            "skill_input_catalog_blake3": policy.skill_input_catalog_blake3,
            "native_action_types": list(NATIVE_POLICY_V2_ACTION_TYPES),
        },
        "judge": {
            "role": judge.role.value,
            "model": judge.model,
            "provider": judge.provider,
            "cwd": judge.cwd,
            "sandbox": judge.sandbox.value,
            "ephemeral": judge.ephemeral,
            "fresh_thread": True,
            "actor_thread_reuse_allowed": False,
            "native_actions_allowed": False,
            "config_keys": list(judge.config_keys),
            "config_blake3": blake3_hex(judge.config),
            "offered_mcp_tool_names": [
                tool.fully_qualified_name for tool in judge.offered_tools
            ],
            "offered_tool_schema_blake3": _tool_catalog_blake3(judge),
            "base_instructions_blake3": blake3_hex(judge.base_instructions),
            "developer_instructions_blake3": blake3_hex(
                judge.developer_instructions
            ),
        },
        "launch_binding": {
            "config_overrides": list(overrides),
            "sdk_version": policy.sdk_version,
            "sdk_protocol": policy.sdk_protocol,
            "cli_version": policy.cli_version,
            "cli_executable_blake3": policy.cli_executable_blake3,
            "turn_mcp_launch_blake3": policy.turn_mcp_launch_blake3,
        },
        "stage_tool_guidance_binding": (
            None
            if stage_tool_guidance is None
            else {
                "schema": "eva.codex-stage-tool-guidance.v1",
                "source_candidate_id": stage_tool_guidance.source_candidate_id,
                "guidance_blake3": stage_tool_guidance.guidance_blake3,
                "prompt_blake3": stage_tool_guidance.prompt_blake3,
                "public_runtime_context_blake3": (
                    stage_tool_guidance.public_runtime_context_blake3
                ),
                "source_tool_catalog_blake3": (
                    stage_tool_guidance.source_tool_catalog_blake3
                ),
                "host_pre_effect_guard": "GuidedStageToolRuntimeV1",
            }
        ),
        "existing_schemas_changed": False,
    }


@dataclass(frozen=True, slots=True)
class NativeFastpathV2Deployment:
    """Exact public deployment commitment for one candidate and stage."""

    candidate_id: str
    policy: NativePolicyV2
    actor_options: CodexThreadOptions
    judge_options: CodexThreadOptions
    stage_tool_guidance: StageToolGuidanceV1 | None
    applied_config_overrides: tuple[str, ...]
    deployment_blake3: str

    def __post_init__(self) -> None:
        _clean_text(self.candidate_id, label="native candidate ID")
        verify_native_policy_v2(self.policy)
        if self.stage_tool_guidance is not None and (
            self.candidate_id != self.stage_tool_guidance.source_candidate_id
        ):
            raise NativeFastpathV2Error(
                "native E2E candidate and stage-tool guidance identity differ"
            )
        if tuple(self.applied_config_overrides) != native_policy_v2_config_overrides(
            self.actor_options.cwd
        ):
            raise NativeFastpathV2Error("native launch config overrides differ")
        actor = bind_native_actor_options_v2(
            self.actor_options,
            self.policy,
            self.stage_tool_guidance,
        )
        judge = bind_native_judge_options_v2(self.judge_options)
        if actor != self.actor_options or judge != self.judge_options:
            raise NativeFastpathV2Error("native deployment lacks its required instruction")
        if self.deployment_blake3 != blake3_hex(self.core_document()):
            raise NativeFastpathV2Error("native deployment BLAKE3 differs")
        object.__setattr__(self, "applied_config_overrides", tuple(self.applied_config_overrides))

    def core_document(self) -> dict[str, Any]:
        return _deployment_core(
            candidate_id=self.candidate_id,
            policy=self.policy,
            actor=self.actor_options,
            judge=self.judge_options,
            overrides=self.applied_config_overrides,
            stage_tool_guidance=self.stage_tool_guidance,
        )

    def to_document(self) -> dict[str, Any]:
        return {**self.core_document(), "deployment_blake3": self.deployment_blake3}

    def guard_tool_runtime(self, tools: Any) -> GuidedStageToolRuntimeV1:
        """Bind E2E S1/S2 to the host-owned pre-effect frontier guard."""

        if self.stage_tool_guidance is None:
            raise NativeFastpathV2Error(
                "only native E2E deployments own an S1/S2 runtime guard"
            )
        return guard_guided_stage_tool_runtime_v1(
            tools,
            self.stage_tool_guidance,
        )


def prepare_native_fastpath_v2(
    *,
    candidate_id: str,
    stage: Stage | str,
    actor_options: CodexThreadOptions,
    judge_options: CodexThreadOptions,
    actor_skills: Sequence[CodexSkill],
    sdk_version: str,
    sdk_protocol: str,
    cli_version: str,
    cli_executable_blake3: str,
    turn_mcp_metadata: Mapping[str, Any],
    applied_config_overrides: Sequence[str],
    public_runtime_context: Mapping[str, Any] | None = None,
    source_tool_catalog: Sequence[Mapping[str, Any]] | None = None,
) -> NativeFastpathV2Deployment:
    """Prepare one zero-provider native deployment and reopen every commitment."""

    policy = build_native_policy_v2(
        stage=stage,
        sdk_version=sdk_version,
        sdk_protocol=sdk_protocol,
        cli_version=cli_version,
        cli_executable_blake3=cli_executable_blake3,
        skills=actor_skills,
        turn_mcp_metadata=turn_mcp_metadata,
    )
    if policy.stage is Stage.E2E:
        if public_runtime_context is None or source_tool_catalog is None:
            raise NativeFastpathV2Error(
                "native E2E deployment lacks public S1/S2 guidance inputs"
            )
        stage_tool_guidance = build_stage_tool_guidance_v1(
            public_runtime_context=public_runtime_context,
            source_tool_catalog=source_tool_catalog,
        )
        if candidate_id != stage_tool_guidance.source_candidate_id:
            raise NativeFastpathV2Error(
                "native E2E candidate and stage-tool guidance identity differ"
            )
    else:
        if public_runtime_context is not None or source_tool_catalog is not None:
            raise NativeFastpathV2Error(
                "stage-tool guidance inputs are reserved for native E2E"
            )
        stage_tool_guidance = None
    actor = bind_native_actor_options_v2(
        actor_options,
        policy,
        stage_tool_guidance,
    )
    judge = bind_native_judge_options_v2(judge_options)
    overrides = tuple(applied_config_overrides)
    deployment = NativeFastpathV2Deployment(
        candidate_id=candidate_id,
        policy=policy,
        actor_options=actor,
        judge_options=judge,
        stage_tool_guidance=stage_tool_guidance,
        applied_config_overrides=overrides,
        deployment_blake3=blake3_hex(
            _deployment_core(
                candidate_id=candidate_id,
                policy=policy,
                actor=actor,
                judge=judge,
                overrides=overrides,
                stage_tool_guidance=stage_tool_guidance,
            )
        ),
    )
    verify_native_fastpath_v2(deployment)
    return deployment


def verify_native_fastpath_v2(deployment: NativeFastpathV2Deployment) -> None:
    if not isinstance(deployment, NativeFastpathV2Deployment):
        raise NativeFastpathV2Error("value is not a native fast-path v2 deployment")
    verify_native_policy_v2(deployment.policy)
    if deployment.deployment_blake3 != blake3_hex(deployment.core_document()):
        raise NativeFastpathV2Error("native deployment commitment differs")
    if not is_blake3(deployment.deployment_blake3):
        raise NativeFastpathV2Error("native deployment BLAKE3 differs")


def _relative_effect_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise NativeFastpathV2Error(f"{label} differs")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise NativeFastpathV2Error(f"{label} must be workspace-relative")
    return path


def _verify_file_change(call: CodexToolCall) -> None:
    if (
        call.tool_type != "fileChange"
        or call.name != "fileChange"
        or call.status != "completed"
        or call.lifecycle != ("item/started", "item/completed")
        or call.mcp_server is not None
        or call.mcp_tool is not None
        or call.fully_qualified_name is not None
        or not isinstance(call.arguments, Mapping)
        or set(call.arguments) != {"changes"}
        or not isinstance(call.arguments["changes"], tuple)
        or not call.arguments["changes"]
        or not isinstance(call.output, Mapping)
        or set(call.output) != {"status"}
        or call.output.get("status") != "completed"
    ):
        raise NativeFastpathV2Error("native fileChange receipt differs")
    paths: set[PurePosixPath] = set()
    for change in call.arguments["changes"]:
        if (
            not isinstance(change, Mapping)
            or set(change) != {"diff", "kind", "path"}
            or not isinstance(change["diff"], str)
            or not change["diff"]
            or not isinstance(change["kind"], Mapping)
        ):
            raise NativeFastpathV2Error("native fileChange item differs")
        target = _relative_effect_path(change["path"], label="native fileChange path")
        if target in paths:
            raise NativeFastpathV2Error("native fileChange repeats an effect path")
        paths.add(target)
        kind = change["kind"]
        kind_type = kind.get("type")
        if kind_type in {"add", "delete"}:
            if set(kind) != {"type"}:
                raise NativeFastpathV2Error("native fileChange kind differs")
        elif kind_type == "update":
            if set(kind) not in (
                {"type"},
                {"type", "move_path"},
                {"type", "movePath"},
            ):
                raise NativeFastpathV2Error("native fileChange update differs")
            move = kind.get("move_path", kind.get("movePath"))
            if move is not None:
                destination = _relative_effect_path(
                    move, label="native fileChange move path"
                )
                if destination in paths:
                    raise NativeFastpathV2Error(
                        "native fileChange repeats an effect path"
                    )
                paths.add(destination)
        else:
            raise NativeFastpathV2Error("native fileChange kind differs")


def _completion_sequence(receipt: CodexTurnReceipt, call: CodexToolCall) -> int:
    sequences = []
    for event in receipt.events:
        if event.method != "item/completed":
            continue
        item = event.payload.get("item")
        if isinstance(item, Mapping) and item.get("id") == call.upstream_item_id:
            sequences.append(event.sequence)
    if len(sequences) != 1 or sequences[0] <= call.first_event_sequence:
        raise NativeFastpathV2Error("native effect lifecycle evidence differs")
    return sequences[0]


def _verify_native_effects_sequential(receipt: CodexTurnReceipt) -> None:
    intervals = tuple(
        (call.first_event_sequence, _completion_sequence(receipt, call), call)
        for call in receipt.tool_calls
    )
    for start, end, call in intervals:
        if call.tool_type != "fileChange":
            continue
        if any(
            other is not call and other_start < end and start < other_end
            for other_start, other_end, other in intervals
        ):
            raise NativeFastpathV2Error("native fileChange must execute sequentially")


def _verify_receipt_identity(
    receipt: CodexTurnReceipt,
    *,
    options: CodexThreadOptions,
    policy: NativePolicyV2,
) -> None:
    verify_codex_turn_receipt(receipt)
    if (
        receipt.role is not options.role
        or receipt.model != options.model
        or receipt.provider != options.provider
        or receipt.sandbox is not options.sandbox
        or receipt.thread_resumed is not False
        or receipt.sdk_version != policy.sdk_version
        or receipt.server_version != policy.cli_version
        or receipt.config_keys != options.config_keys
        or receipt.offered_mcp_tool_names
        != tuple(tool.fully_qualified_name for tool in options.offered_tools)
        or receipt.offered_tool_schema_blake3 != _tool_catalog_blake3(options)
        or receipt.status != "completed"
        or not isinstance(receipt.final_response, str)
        or not receipt.final_response.strip()
    ):
        raise NativeFastpathV2Error("native turn receipt binding differs")


def verify_native_actor_receipt_v2(
    receipt: CodexTurnReceipt, deployment: NativeFastpathV2Deployment
) -> None:
    """Accept MCP calls plus sequential ``fileChange``; reject every other native item."""

    verify_native_fastpath_v2(deployment)
    options = deployment.actor_options
    _verify_receipt_identity(receipt, options=options, policy=deployment.policy)
    if (
        not receipt.role.is_actor
        or receipt.sandbox is not CodexSandbox.WORKSPACE_WRITE
        or receipt.selected_skill_ids
        != tuple(skill.skill_id for skill in deployment.policy.skills)
        or receipt.selected_skill_catalog_blake3
        != deployment.policy.skill_input_catalog_blake3
    ):
        raise NativeFastpathV2Error("native actor receipt capability differs")
    offered = set(receipt.offered_mcp_tool_names)
    for call in receipt.tool_calls:
        if call.tool_type == "mcpToolCall":
            if call.fully_qualified_name not in offered:
                raise NativeFastpathV2Error("native actor used uncommitted MCP surface")
        elif call.tool_type == "fileChange":
            _verify_file_change(call)
        else:
            raise NativeFastpathV2Error(
                "native actor emitted a forbidden non-fileChange action"
            )
    _verify_native_effects_sequential(receipt)


def verify_native_judge_receipt_v2(
    receipt: CodexTurnReceipt, deployment: NativeFastpathV2Deployment
) -> None:
    """Require an independent, fresh, read-only MCP-only judge turn."""

    verify_native_fastpath_v2(deployment)
    options = deployment.judge_options
    _verify_receipt_identity(receipt, options=options, policy=deployment.policy)
    if (
        receipt.role is not CodexRole.JUDGE
        or receipt.sandbox is not CodexSandbox.READ_ONLY
        or receipt.selected_skill_ids
        or receipt.selected_skill_catalog_blake3 != blake3_hex(())
    ):
        raise NativeFastpathV2Error("native fast-path judge receipt capability differs")
    offered = set(receipt.offered_mcp_tool_names)
    if any(
        call.tool_type != "mcpToolCall" or call.fully_qualified_name not in offered
        for call in receipt.tool_calls
    ):
        raise NativeFastpathV2Error("native fast-path judge emitted a native action")


def verify_native_actor_judge_pair_v2(
    actor_receipt: CodexTurnReceipt,
    judge_receipt: CodexTurnReceipt,
    deployment: NativeFastpathV2Deployment,
) -> None:
    """Reopen both turns and prove that the actor thread was not reused by the judge."""

    verify_native_actor_receipt_v2(actor_receipt, deployment)
    verify_native_judge_receipt_v2(judge_receipt, deployment)
    if (
        actor_receipt.runtime_thread_id == judge_receipt.runtime_thread_id
        or actor_receipt.thread_id == judge_receipt.thread_id
        or actor_receipt.runtime_turn_id == judge_receipt.runtime_turn_id
        or actor_receipt.turn_id == judge_receipt.turn_id
    ):
        raise NativeFastpathV2Error("native actor thread was reused by the judge")


def _effect_file_state(value: Any, *, label: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if (
        not isinstance(value, Mapping)
        or set(value) != {"content_blake3", "byte_count", "mode"}
        or not is_blake3(value.get("content_blake3"))
        or type(value.get("byte_count")) is not int
        or value["byte_count"] < 0
        or not isinstance(value.get("mode"), str)
        or len(value["mode"]) != 4
        or any(character not in "01234567" for character in value["mode"])
    ):
        raise NativeFastpathV2Error(f"{label} differs")
    return value


def verify_native_rollout_effect_binding_v2(
    rollout_or_metadata: Any,
    actor_receipt: CodexTurnReceipt,
    deployment: NativeFastpathV2Deployment,
    *,
    workspace_before_tree_blake3: str | None = None,
    workspace_after_tree_blake3: str | None = None,
) -> str:
    """Cross-bind projection-v2 workspace effects to policy and receipt.

    ``rollout_or_metadata`` may be a ``ProviderRollout``/``EvidenceBundle``-like
    value or its already-frozen safe metadata mapping.  No pipeline contract is
    changed; this is an independent reopening seam for the additive projection
    sidecar.
    """

    verify_native_actor_receipt_v2(actor_receipt, deployment)
    if isinstance(rollout_or_metadata, Mapping):
        metadata = rollout_or_metadata
        if (
            workspace_before_tree_blake3 is None
            or workspace_after_tree_blake3 is None
        ):
            raise NativeFastpathV2Error(
                "standalone native metadata verification requires both workspace trees"
            )
    else:
        if hasattr(rollout_or_metadata, "safe_provider_metadata"):
            metadata = rollout_or_metadata.safe_provider_metadata
            before = getattr(rollout_or_metadata, "workspace_before", None)
            after = getattr(rollout_or_metadata, "workspace_after", None)
            if before is None or after is None:
                raise NativeFastpathV2Error(
                    "native evidence bundle lacks workspace snapshots"
                )
            observed_before = getattr(before, "tree_blake3", None)
            observed_after = getattr(after, "tree_blake3", None)
            if workspace_before_tree_blake3 is not None and (
                workspace_before_tree_blake3 != observed_before
            ):
                raise NativeFastpathV2Error("native before workspace tree differs")
            if workspace_after_tree_blake3 is not None and (
                workspace_after_tree_blake3 != observed_after
            ):
                raise NativeFastpathV2Error("native after workspace tree differs")
            workspace_before_tree_blake3 = observed_before
            workspace_after_tree_blake3 = observed_after
        else:
            metadata = getattr(rollout_or_metadata, "safe_metadata", None)
            if (
                workspace_before_tree_blake3 is None
                or workspace_after_tree_blake3 is None
            ):
                raise NativeFastpathV2Error(
                    "native rollout verification requires both workspace trees"
                )
    expected_metadata_keys = {
        "schema",
        "codex_turn_receipt",
        "tool_call_groups",
        "codex_to_pipeline_call_ids",
        "raw_input_recorded",
        "semantic_retry_count",
        "native_file_change_effect_binding",
    }
    if (
        not isinstance(metadata, Mapping)
        or set(metadata) != expected_metadata_keys
        or metadata.get("schema")
        != "eva.codex-provider-rollout-projection.v2-native-file-change"
        or metadata.get("raw_input_recorded") is not False
        or metadata.get("semantic_retry_count") != 0
    ):
        raise NativeFastpathV2Error("native rollout projection metadata differs")
    groups = metadata["tool_call_groups"]
    call_mapping = metadata["codex_to_pipeline_call_ids"]
    if not isinstance(groups, (tuple, list)) or not isinstance(call_mapping, Mapping):
        raise NativeFastpathV2Error("native rollout MCP call binding differs")
    flattened: list[str] = []
    for group in groups:
        if (
            not isinstance(group, (tuple, list))
            or not group
            or any(not isinstance(call_id, str) for call_id in group)
        ):
            raise NativeFastpathV2Error("native rollout MCP call group differs")
        flattened.extend(group)
    mcp_call_ids = {
        call.tool_call_id
        for call in actor_receipt.tool_calls
        if call.tool_type == "mcpToolCall"
    }
    if (
        set(call_mapping) != mcp_call_ids
        or any(not isinstance(value, str) for value in call_mapping.values())
        or len(flattened) != len(set(flattened))
        or len(flattened) != len(call_mapping)
        or set(flattened) != set(call_mapping.values())
    ):
        raise NativeFastpathV2Error("native rollout MCP call mapping differs")
    try:
        persisted_receipt = codex_turn_receipt_from_document(
            metadata["codex_turn_receipt"]
        )
    except Exception as exc:
        raise NativeFastpathV2Error(
            "native rollout projection receipt cannot be reopened"
        ) from exc
    if canonical_value(persisted_receipt) != canonical_value(actor_receipt):
        raise NativeFastpathV2Error("native rollout projection receipt differs")

    binding = metadata["native_file_change_effect_binding"]
    binding_keys = {
        "schema",
        "workspace_before_tree_blake3",
        "workspace_after_tree_blake3",
        "declared_changed_paths",
        "actual_changed_paths",
        "effects",
        "binding_blake3",
    }
    if not isinstance(binding, Mapping) or set(binding) != binding_keys:
        raise NativeFastpathV2Error("native fileChange effect binding fields differ")
    binding_core = {
        key: value for key, value in binding.items() if key != "binding_blake3"
    }
    before_tree = binding.get("workspace_before_tree_blake3")
    after_tree = binding.get("workspace_after_tree_blake3")
    if (
        binding.get("schema") != "eva.codex-native-file-change-effect-binding.v1"
        or not is_blake3(before_tree)
        or not is_blake3(after_tree)
        or not is_blake3(binding.get("binding_blake3"))
        or blake3_hex(binding_core) != binding["binding_blake3"]
        or workspace_before_tree_blake3 is not None
        and before_tree != workspace_before_tree_blake3
        or workspace_after_tree_blake3 is not None
        and after_tree != workspace_after_tree_blake3
    ):
        raise NativeFastpathV2Error("native fileChange effect binding commitment differs")
    declared = binding.get("declared_changed_paths")
    actual = binding.get("actual_changed_paths")
    effects = binding.get("effects")
    if (
        not isinstance(declared, (tuple, list))
        or not isinstance(actual, (tuple, list))
        or not isinstance(effects, (tuple, list))
        or any(not isinstance(path, str) for path in declared)
        or any(not isinstance(path, str) for path in actual)
        or tuple(declared) != tuple(actual)
        or tuple(declared) != tuple(sorted(set(declared)))
    ):
        raise NativeFastpathV2Error("native declared and actual changed paths differ")
    for path in declared:
        _relative_effect_path(path, label="native changed path")

    native_calls = tuple(
        call for call in actor_receipt.tool_calls if call.tool_type == "fileChange"
    )
    expected_changes = tuple(
        (call, index, canonical_value(change))
        for call in native_calls
        for index, change in enumerate(call.arguments["changes"])
    )
    if len(effects) != len(expected_changes):
        raise NativeFastpathV2Error("native effect count differs from fileChange calls")
    observed_paths: set[str] = set()
    for effect, (call, change_index, change) in zip(
        effects, expected_changes, strict=True
    ):
        if not isinstance(effect, Mapping):
            raise NativeFastpathV2Error("native effect row differs")
        effect_keys = {
            "tool_call_id",
            "upstream_item_id",
            "change_index",
            "kind",
            "source_path",
            "destination_path",
            "diff_blake3",
            "path_effects",
            "effect_blake3",
        }
        if set(effect) != effect_keys:
            raise NativeFastpathV2Error("native effect row fields differ")
        effect_core = {
            key: value for key, value in effect.items() if key != "effect_blake3"
        }
        kind = change["kind"]["type"]
        destination = change["kind"].get(
            "move_path", change["kind"].get("movePath")
        )
        expected_paths = (
            (change["path"],) if destination is None else (change["path"], destination)
        )
        if (
            effect.get("tool_call_id") != call.tool_call_id
            or effect.get("upstream_item_id") != call.upstream_item_id
            or effect.get("change_index") != change_index
            or effect.get("kind") != kind
            or effect.get("source_path") != change["path"]
            or effect.get("destination_path") != destination
            or effect.get("diff_blake3")
            != blake3_bytes(change["diff"].encode("utf-8"))
            or not is_blake3(effect.get("effect_blake3"))
            or blake3_hex(effect_core) != effect["effect_blake3"]
        ):
            raise NativeFastpathV2Error("native effect row commitment differs")
        path_effects = effect.get("path_effects")
        if not isinstance(path_effects, (tuple, list)) or tuple(
            row.get("path") if isinstance(row, Mapping) else None
            for row in path_effects
        ) != expected_paths:
            raise NativeFastpathV2Error("native effect path rows differ")
        states: list[tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]] = []
        for row in path_effects:
            if not isinstance(row, Mapping) or set(row) != {"path", "before", "after"}:
                raise NativeFastpathV2Error("native effect path row fields differ")
            path = _relative_effect_path(row["path"], label="native effect path")
            if path.as_posix() in observed_paths:
                raise NativeFastpathV2Error("native effect path appears more than once")
            observed_paths.add(path.as_posix())
            states.append(
                (
                    _effect_file_state(row["before"], label="native before file state"),
                    _effect_file_state(row["after"], label="native after file state"),
                )
            )
        source_before, source_after = states[0]
        if kind == "add":
            valid = len(states) == 1 and source_before is None and source_after is not None
        elif kind == "delete":
            valid = len(states) == 1 and source_before is not None and source_after is None
        elif kind == "update" and destination is None:
            valid = (
                len(states) == 1
                and source_before is not None
                and source_after is not None
                and source_before != source_after
            )
        elif kind == "update":
            destination_before, destination_after = states[1]
            valid = (
                len(states) == 2
                and source_before is not None
                and source_after is None
                and destination_before is None
                and destination_after is not None
            )
        else:  # already rejected by receipt verification, retained as defense in depth
            valid = False
        if not valid:
            raise NativeFastpathV2Error("native declared effect semantics differ")
    if tuple(sorted(observed_paths)) != tuple(declared):
        raise NativeFastpathV2Error("native effect paths differ from tree transition")
    return binding["binding_blake3"]


__all__ = [
    "NATIVE_FASTPATH_V2_ACTOR_INSTRUCTION",
    "NATIVE_FASTPATH_V2_JUDGE_INSTRUCTION",
    "NATIVE_FASTPATH_V2_SCHEMA",
    "NativeFastpathV2Deployment",
    "NativeFastpathV2Error",
    "bind_native_actor_options_v2",
    "bind_native_judge_options_v2",
    "prepare_native_fastpath_v2",
    "verify_native_actor_judge_pair_v2",
    "verify_native_actor_receipt_v2",
    "verify_native_fastpath_v2",
    "verify_native_judge_receipt_v2",
    "verify_native_rollout_effect_binding_v2",
]
