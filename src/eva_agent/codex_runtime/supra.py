"""Prospective v1.3 composition; canonical tools, skills and receipts stay intact."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
import json
from typing import Callable
from urllib.parse import urlsplit

from .contracts import CodexThreadOptions, CodexTurnInput
from .research_memory import (
    ResearchContextPolicy, memory_launch_options, memory_thread_options,
    memory_turn_input,
)

PROFILE_NAME = "evamed-codex-v1.3-supra"
WORKFLOW_INSTRUCTIONS = """Medical workflow execution:
Maintain a short current-stage/task objective in notes/task-state.md. After a
resume or compaction, compare that objective with the latest user request before
acting; do not advance to another stage just because its input is visible.
Keep task-state/status notes separate from the requested final answer artifact.
Before finishing, read back the final artifact and check it still contains the
answer, not a short status update. Preserve useful drafts as separate versions.
Discover and load relevant offered skills, use the exact available tool schemas,
and parallelize independent calls. Coordinate writes to each shared artifact.
Record consequential tool failures through summary_failures, then apply the
specific preventive check before a similar action. Reopen source artifacts when
memory conflicts with the workspace. Do not import hidden reference answers.
Follow the task's actual interaction and completion contract. A medical dialogue
or answer-only task need not invent S1-S5 research phases. Native task verifiers
and process/workspace rubrics remain distinct. Missing evidence stays unknown.
"""


class SupraMode(str, Enum):
    INSTANT = "instant"
    THINK = "think"


@dataclass(frozen=True)
class SupraProfile:
    """Caller supplies actual route/capacity, not a guessed universal model limit.

    Protocol is an explicit route contract. Merely constructing it does not
    establish endpoint support or a successful medical/tool-use rollout.
    """
    model: str
    provider: str
    mode: SupraMode
    context: ResearchContextPolicy
    protocol: str = "codex_effort"
    local_qwen_endpoint: str | None = None
    measured_compacted_input_tokens: int | None = None

    def __post_init__(self):
        if not self.model or not self.provider or not isinstance(self.mode, SupraMode):
            raise ValueError("explicit model, provider and supra mode required")
        if self.protocol not in {"codex_effort", "qwen_template", "chat_reasoning_effort"}:
            raise ValueError("unsupported supra route reasoning protocol")
        if self.local_qwen_endpoint is not None:
            endpoint = urlsplit(self.local_qwen_endpoint)
            if (self.protocol != "qwen_template" or "qwen" not in self.model.lower()
                    or endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1"
                    or not endpoint.port or endpoint.path != "/v1" or endpoint.username
                    or endpoint.password or endpoint.query or endpoint.fragment):
                raise ValueError("instant endpoint must be an explicit local Qwen route")
        if self.mode is SupraMode.INSTANT and self.local_qwen_endpoint is None:
            raise ValueError("instant mode is reserved for local Qwen")
        floor = self.measured_compacted_input_tokens
        if floor is not None and (type(floor) is not int or floor < 0
                or floor + self.context.reserve_tokens >= self.context.compact_at_tokens):
            raise ValueError("compaction threshold must exceed measured retained-context floor")

    def thread_options(self, base: CodexThreadOptions) -> CodexThreadOptions:
        if (base.model, base.provider) != (self.model, self.provider):
            raise ValueError("supra profile differs from selected model/provider")
        enhanced = memory_thread_options(base, policy=self.context)
        threshold = enhanced.config.get("model_auto_compact_token_limit")
        floor = self.measured_compacted_input_tokens
        if floor is not None and threshold <= floor + self.context.reserve_tokens:
            raise ValueError("caller compaction override recreates observed churn")
        return replace(enhanced, service_name=PROFILE_NAME,
            developer_instructions=enhanced.developer_instructions + "\n" + WORKFLOW_INSTRUCTIONS)

    def turn_input(self, base: CodexTurnInput) -> CodexTurnInput:
        if base.model is not None and base.model != self.model:
            raise ValueError("supra turn model differs")
        # xhigh is native Codex effort for direct routes. Qwen's real control is
        # the explicit Chat transport boolean, not a claim of xhigh equivalence.
        effort = "xhigh" if self.mode is SupraMode.THINK else "none"
        return replace(memory_turn_input(base), model=self.model, effort=effort)

    def inspection(self) -> dict:
        return {"schema": "eva.supra-profile.v1", "profile": PROFILE_NAME,
            "model_requested": self.model, "provider": self.provider, "mode": self.mode.value,
            "protocol": self.protocol, "codex_effort_requested": "xhigh" if self.mode is SupraMode.THINK else "none",
            "qwen_thinking_requested": (self.mode is SupraMode.THINK) if self.protocol == "qwen_template" else None,
            "local_qwen": self.local_qwen_endpoint is not None,
            "context_tokens": self.context.context_tokens,
            "compact_at_tokens": self.context.compact_at_tokens,
            "compaction_headroom_mode": self.context.compaction_headroom_mode,
            "measured_compacted_input_tokens": self.measured_compacted_input_tokens,
            "provider_support_verified": False, "medical_quality_verified": False,
            "canonical_tool_skill_schema_changes": False, "provider_calls": 0}


class SupraChatTransport:
    """Set the real Chat boundary controls without changing any tool schema.

    Supply as the existing ResponsesAdapterGateway upstream_transport. The
    adapter's generic reasoning field is otherwise not forwarded to Chat.
    Optional observations contain only request controls and numeric metadata;
    prompts, outputs, credentials and private reasoning are never emitted.
    """
    def __init__(self, transport, profile: SupraProfile, *, observer: Callable[[dict], None] | None = None):
        if profile.protocol == "codex_effort":
            raise ValueError("direct Codex reasoning does not use Chat transport")
        self.transport, self.profile, self.observer = transport, profile, observer

    def __call__(self, binding, body):
        if binding.model_id != self.profile.model or body.get("model") != self.profile.model:
            raise ValueError("supra transport model binding differs")
        forwarded = deepcopy(dict(body))
        thinking = self.profile.mode is SupraMode.THINK
        if self.profile.protocol == "qwen_template":
            kwargs = dict(forwarded.get("chat_template_kwargs") or {})
            kwargs["enable_thinking"] = thinking
            forwarded["chat_template_kwargs"] = kwargs
            controls = {"chat_template_kwargs": {"enable_thinking": thinking}}
        else:
            forwarded["reasoning_effort"] = "xhigh"
            controls = {"reasoning_effort": "xhigh"}
        if self.observer:
            self.observer({"schema": "eva.supra-mode-observation.v1", "event": "request",
                "model_requested": self.profile.model, "mode": self.profile.mode.value,
                "forwarded_controls": controls, "automatic_retry": False})
        response = self.transport(binding, forwarded)
        if self.observer:
            observed = {"schema": "eva.supra-mode-observation.v1", "event": "response",
                "http_status": response.status, "thinking_execution_verified": False,
                "reasoning_tokens": None, "visible_content_characters": None,
                "private_reasoning_recorded": False}
            try:
                value = json.loads(response.body)
                details = value.get("usage", {}).get("completion_tokens_details") or {}
                count = details.get("reasoning_tokens")
                if type(count) is int and count >= 0:
                    observed["reasoning_tokens"] = count
                choices = value.get("choices") or []
                content = (choices[0].get("message") or {}).get("content") if choices else None
                if isinstance(content, str):
                    observed["visible_content_characters"] = len(content)
            except (ValueError, TypeError, AttributeError, IndexError):
                pass
            self.observer(observed)
        return response


# Preserve the v1.1 lightweight defaults and all explicit caller security,
# endpoint, environment, and permissions. No new process is started here.
supra_launch_options = memory_launch_options
