"""Pluggable OpenAI-SDK-style rollout, judge, and reward adapters.

The adapters depend on injected clients and never discover credentials or open
network connections on their own.  A client only needs to implement
``client.chat.completions.create(**kwargs)``.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from .contracts import (
    AgentJudge,
    CompiledRubricTable,
    ContractError,
    EvidenceBundle,
    JudgeAssessment,
    JudgeRequest,
    ProviderRollout,
    RewardRecord,
    RolloutRequest,
    RubricItemScore,
    RuntimeIdFactory,
    ToolCall,
    ToolRuntimePort,
    TrajectoryEvent,
    freeze_json,
)
from .digests import blake3_hex, canonical_value
from .ids import RandomUUIDFactory
from .judge_workspace import JudgeWorkspaceTools
from .tools import MAXIMUM_PARALLEL_TOOL_CALLS


class AdapterError(ContractError):
    """An injected provider returned an invalid public response."""


def _member(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _message_from_response(response: Any) -> Any:
    choices = _member(response, "choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or len(choices) != 1:
        raise AdapterError("provider response must contain exactly one choice")
    message = _member(choices[0], "message")
    if message is None:
        raise AdapterError("provider choice lacks a message")
    return message


def _event(
    ids: RuntimeIdFactory,
    *,
    role: str,
    content: Any,
    tool_call_ids: Sequence[str] = (),
) -> TrajectoryEvent:
    event_id = ids.new("trajectory-event")
    core = {
        "event_id": event_id,
        "role": role,
        "content": freeze_json(content),
        "tool_call_ids": tuple(tool_call_ids),
    }
    return TrajectoryEvent(**core, event_blake3=blake3_hex(core))


class OpenAIStyleRolloutAdapter:
    """Drive one policy trajectory, including same-turn parallel tool calls."""

    def __init__(
        self,
        client: Any,
        *,
        id_factory: RuntimeIdFactory,
        system_instruction: str = "Use the available tools and leave verifiable workspace evidence.",
        maximum_model_turns: int = 12,
    ) -> None:
        if not system_instruction or not 1 <= maximum_model_turns <= 64:
            raise AdapterError("rollout adapter bounds differ")
        self._client = client
        self._ids = id_factory
        self._system = system_instruction
        self._maximum_turns = maximum_model_turns

    def _create(self, **kwargs: Any) -> Any:
        try:
            create = self._client.chat.completions.create
        except AttributeError:
            raise AdapterError("client lacks chat.completions.create") from None
        return create(**kwargs)

    def run(self, request: RolloutRequest, tools: ToolRuntimePort) -> ProviderRollout:
        context = canonical_value(request.policy_visible_context)
        events = [
            _event(self._ids, role="system", content=self._system),
            _event(self._ids, role="user", content=context),
        ]
        sdk_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": json.dumps(context, sort_keys=True, separators=(",", ":"))},
        ]
        tool_schemas = [canonical_value(value) for value in request.available_tools]
        final_output: str | None = None
        for turn in range(self._maximum_turns):
            kwargs: dict[str, Any] = {
                "model": request.model.model_id,
                "messages": sdk_messages,
                "temperature": 0,
            }
            if tool_schemas:
                kwargs.update(
                    {
                        "tools": tool_schemas,
                        "tool_choice": "auto",
                        "parallel_tool_calls": True,
                    }
                )
            response = self._create(**kwargs)
            message = _message_from_response(response)
            content = _member(message, "content", "")
            if content is None:
                content = ""
            if not isinstance(content, str):
                raise AdapterError("assistant content must be text")
            raw_calls = _member(message, "tool_calls", None) or []
            if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
                raise AdapterError("assistant tool_calls must be a sequence")
            if raw_calls:
                internal_calls: list[ToolCall] = []
                sdk_calls: list[dict[str, Any]] = []
                provider_ids: dict[str, str] = {}
                visible_calls: list[dict[str, Any]] = []
                for raw in raw_calls:
                    provider_id = _member(raw, "id")
                    function = _member(raw, "function")
                    name = _member(function, "name")
                    encoded_arguments = _member(function, "arguments")
                    if not isinstance(provider_id, str) or not provider_id or not isinstance(name, str) or not name:
                        raise AdapterError("provider tool-call identity differs")
                    if not isinstance(encoded_arguments, str):
                        raise AdapterError("provider tool-call arguments are not encoded JSON")
                    try:
                        arguments = json.loads(encoded_arguments)
                    except (ValueError, RecursionError):
                        raise AdapterError("provider tool-call arguments are invalid JSON") from None
                    if not isinstance(arguments, dict):
                        raise AdapterError("provider tool-call arguments must be an object")
                    internal_id = self._ids.new("tool-call")
                    provider_ids[internal_id] = provider_id
                    call = ToolCall(call_id=internal_id, name=name, arguments=arguments)
                    internal_calls.append(call)
                    sdk_calls.append(
                        {
                            "id": provider_id,
                            "type": "function",
                            "function": {"name": name, "arguments": encoded_arguments},
                        }
                    )
                    visible_calls.append(
                        {"call_id": internal_id, "name": name, "arguments": call.arguments}
                    )
                call_ids = tuple(call.call_id for call in internal_calls)
                events.append(
                    _event(
                        self._ids,
                        role="assistant",
                        content={"text": content, "tool_calls": visible_calls},
                        tool_call_ids=call_ids,
                    )
                )
                sdk_messages.append({"role": "assistant", "content": content, "tool_calls": sdk_calls})
                results = tools.execute(internal_calls)
                by_call = {result.call_id: result for result in results}
                if set(by_call) != set(call_ids):
                    raise AdapterError("tool runtime omitted or replaced a same-turn call")
                for call_id in call_ids:
                    result = by_call[call_id]
                    visible_result = {
                        "status": result.status,
                        "output": result.output,
                        "error_code": result.error_code,
                        "receipt_blake3": result.receipt_blake3,
                    }
                    events.append(
                        _event(
                            self._ids,
                            role="tool",
                            content=visible_result,
                            tool_call_ids=(call_id,),
                        )
                    )
                    sdk_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": provider_ids[call_id],
                            "content": json.dumps(
                                canonical_value(visible_result), sort_keys=True, separators=(",", ":")
                            ),
                        }
                    )
                continue
            final_output = content
            events.append(_event(self._ids, role="assistant", content=content))
            break
        if final_output is None:
            raise AdapterError("rollout exceeded the fixed model-turn bound")
        core = {
            "rollout_id": request.rollout_id,
            "model_id": request.model.model_id,
            "sandbox_manifest_blake3": request.sandbox.manifest_blake3,
            "policy_events": tuple(events),
            "tool_trace_blake3": tools.trace().trace_blake3,
            "assistant_output": final_output,
        }
        return ProviderRollout(
            assistant_output=final_output,
            provider_receipt_blake3=blake3_hex(core),
            policy_events=tuple(events),
            safe_metadata={"model_turns": turn + 1, "raw_provider_response_recorded": False},
        )


def _rubric_items(rubric: CompiledRubricTable) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in rubric.items:
        item_id = _member(item, "item_id")
        description = _member(item, "description")
        weight = _member(item, "weight")
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in seen
            or not isinstance(description, str)
            or not description
            or type(weight) not in {int, float}
            or float(weight) <= 0
        ):
            raise AdapterError("compiled rubric item differs")
        seen.add(item_id)
        document = canonical_value(item)
        if not isinstance(document, dict):
            raise AdapterError("compiled rubric item is not an object")
        partial_credit = document.get("partial_credit")
        levels = partial_credit.get("levels") if isinstance(partial_credit, dict) else None
        if (
            not isinstance(levels, list)
            or len(levels) < 2
            or any(
                not isinstance(level, dict)
                or type(level.get("score_bps")) is not int
                or not 0 <= level["score_bps"] <= 10_000
                or not isinstance(level.get("criterion"), str)
                or not level["criterion"]
                for level in levels
            )
        ):
            raise AdapterError("compiled rubric partial-credit levels differ")
        # The judge receives the complete compiled item, including observable
        # selectors, allowed score levels, hard gates, labels, and provenance.
        # Reward computation consumes this same object; no hand-translated
        # judge rubric is allowed to drift from it.
        items.append(document)
    if not items:
        raise AdapterError("compiled rubric table is empty")
    return items


def _workspace_manifest(snapshot: Any) -> dict[str, Any]:
    return {
        "label": _member(snapshot, "label"),
        "file_count": _member(snapshot, "file_count"),
        "byte_count": _member(snapshot, "byte_count"),
        "tree_blake3": _member(snapshot, "tree_blake3"),
        "files": [
            {
                "path": _member(row, "path"),
                "byte_count": _member(row, "byte_count"),
                "mode": _member(row, "mode"),
                "content_blake3": _member(row, "content_blake3"),
            }
            for row in _member(snapshot, "files", ())
        ],
    }


def _judge_evidence_projection(evidence: EvidenceBundle) -> dict[str, Any]:
    """Project committed evidence without eagerly copying workspace file bytes."""

    manifest = evidence.sandbox_manifest
    return {
        "bundle_id": evidence.bundle_id,
        "bundle_blake3": evidence.bundle_blake3,
        "rollout_id": evidence.rollout_id,
        "sandbox": {
            "sandbox_id": manifest.sandbox_id,
            "episode_id": manifest.episode_id,
            "source": manifest.source,
            "domain": manifest.domain,
            "stage": manifest.stage,
            "instruction": manifest.instruction,
            "manifest_blake3": manifest.manifest_blake3,
            "rubric": manifest.rubric,
        },
        "model": evidence.model,
        "context_blake3": evidence.context_blake3,
        "workspace_before": _workspace_manifest(evidence.workspace_before),
        "workspace_after": _workspace_manifest(evidence.workspace_after),
        "actor_policy_events": evidence.policy_events,
        "actor_tool_trace": evidence.tool_trace,
        "assistant_output": evidence.assistant_output,
        "provider_receipt_blake3": evidence.provider_receipt_blake3,
        "safe_provider_metadata": evidence.safe_provider_metadata,
    }


def _chat_completion_tool_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    """Translate one strict Responses-style function schema for Chat Completions."""

    document = canonical_value(value)
    if not isinstance(document, dict) or set(document) != {
        "type",
        "name",
        "description",
        "parameters",
        "strict",
    }:
        raise AdapterError("judge workspace tool schema differs")
    if document["type"] != "function" or document["strict"] is not True:
        raise AdapterError("judge workspace tool schema is not strict")
    return {
        "type": "function",
        "function": {
            "name": document["name"],
            "description": document["description"],
            "parameters": document["parameters"],
            "strict": True,
        },
    }


class OpenAIStyleOpus5Judge:
    """Run an Opus 5 evidence agent over immutable, read-only workspace snapshots."""

    def __init__(
        self,
        client: Any,
        *,
        model_id: str,
        id_factory: RuntimeIdFactory | None = None,
        maximum_model_turns: int = 12,
        maximum_parallel_tools: int = MAXIMUM_PARALLEL_TOOL_CALLS,
    ) -> None:
        normalized = model_id.casefold().replace("_", "-").replace(" ", "-")
        if "opus-5" not in normalized:
            raise AdapterError("agent judge model must be Opus 5")
        if not 2 <= maximum_model_turns <= 64:
            raise AdapterError("judge model-turn bound must be in [2,64]")
        if not 1 <= maximum_parallel_tools <= MAXIMUM_PARALLEL_TOOL_CALLS:
            raise AdapterError(
                f"judge parallel-tool width must be in [1,{MAXIMUM_PARALLEL_TOOL_CALLS}]"
            )
        self._client = client
        self.model_id = model_id
        self._ids = id_factory or RandomUUIDFactory()
        self._maximum_turns = maximum_model_turns
        self._tool_width = maximum_parallel_tools

    def _create(self, **kwargs: Any) -> Any:
        try:
            create = self._client.chat.completions.create
        except AttributeError:
            raise AdapterError("judge client lacks chat.completions.create") from None
        return create(**kwargs)

    def judge(self, request: JudgeRequest, rubric: CompiledRubricTable) -> JudgeAssessment:
        if request.judge_model_id != self.model_id:
            raise AdapterError("judge request model identity differs")
        items = _rubric_items(rubric)
        workspace_tools = JudgeWorkspaceTools(
            request.workspace_evidence,
            id_factory=self._ids,
            maximum_parallel_tools=self._tool_width,
        )
        judge_payload = {
            "policy_visible_context": request.policy_visible_context,
            "workspace_evidence": _judge_evidence_projection(request.workspace_evidence),
            "judge_only_reference": request.judge_only_reference,
            "compiled_rubric": {
                "rubric_id": rubric.rubric_id,
                "version": rubric.version,
                "digest": rubric.digest,
                "domain": rubric.domain,
                "stage": rubric.stage,
                "items": items,
            },
            "instruction": (
                "You are an evidence agent, not a one-shot text grader. Inspect the committed workspace "
                "with the supplied read-only tools before scoring. Workspace manifests contain metadata "
                "only; file content is available exclusively through workspace_read/workspace_search. "
                "Score every rubric item from observable context/workspace evidence. For each item, "
                "score must equal one of that item's partial_credit.levels score_bps divided by 10000. "
                "Use the exact item order and cite concrete evidence references. Workspace references must "
                "be `workspace:before:<path>` or `workspace:after:<path>` returned by a tool you actually "
                "used. Return JSON "
                "with item_scores, hard_gates_passed, and summary; do not emit hidden reasoning."
            ),
        }
        system_instruction = (
            "You are the Opus 5 EVA-Agent workspace evidence judge. Operate only on the immutable "
            "snapshot tools. Never claim to have inspected a file unless a tool observation proves it."
        )
        encoded_payload = json.dumps(
            canonical_value(judge_payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        events = [
            _event(self._ids, role="system", content=system_instruction),
            _event(self._ids, role="user", content=canonical_value(judge_payload)),
        ]
        sdk_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": encoded_payload},
        ]
        final_content: str | None = None
        provider_turn_count = 0
        content_inspected = False
        used_workspace_tool = False
        has_workspace_files = bool(
            request.workspace_evidence.workspace_before.files
            or request.workspace_evidence.workspace_after.files
        )
        for _turn in range(self._maximum_turns):
            provider_turn_count += 1
            response = self._create(
                model=self.model_id,
                messages=sdk_messages,
                tools=[
                    _chat_completion_tool_schema(value)
                    for value in workspace_tools.response_api_schemas()
                ],
                tool_choice=(
                    "required"
                    if not used_workspace_tool or (has_workspace_files and not content_inspected)
                    else "auto"
                ),
                parallel_tool_calls=True,
                temperature=0,
            )
            message = _message_from_response(response)
            content = _member(message, "content", "")
            if content is None:
                content = ""
            if not isinstance(content, str):
                raise AdapterError("judge response content is not text")
            raw_calls = _member(message, "tool_calls", None) or []
            if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
                raise AdapterError("judge tool_calls must be a sequence")
            if raw_calls:
                if len(raw_calls) > self._tool_width:
                    raise AdapterError("judge same-turn tool calls exceed the configured width")
                internal_calls: list[ToolCall] = []
                provider_ids: dict[str, str] = {}
                sdk_calls: list[dict[str, Any]] = []
                visible_calls: list[dict[str, Any]] = []
                for raw in raw_calls:
                    provider_id = _member(raw, "id")
                    function = _member(raw, "function")
                    name = _member(function, "name")
                    encoded_arguments = _member(function, "arguments")
                    if (
                        not isinstance(provider_id, str)
                        or not provider_id
                        or not isinstance(name, str)
                        or not name
                        or not isinstance(encoded_arguments, str)
                    ):
                        raise AdapterError("judge provider tool-call identity differs")
                    try:
                        arguments = json.loads(encoded_arguments)
                    except (ValueError, RecursionError):
                        raise AdapterError("judge tool-call arguments are invalid JSON") from None
                    if not isinstance(arguments, dict):
                        raise AdapterError("judge tool-call arguments must be an object")
                    internal_id = self._ids.new("judge-tool-call")
                    provider_ids[internal_id] = provider_id
                    call = ToolCall(call_id=internal_id, name=name, arguments=arguments)
                    internal_calls.append(call)
                    sdk_calls.append(
                        {
                            "id": provider_id,
                            "type": "function",
                            "function": {"name": name, "arguments": encoded_arguments},
                        }
                    )
                    visible_calls.append(
                        {"call_id": internal_id, "name": name, "arguments": call.arguments}
                    )
                if len(set(provider_ids.values())) != len(provider_ids):
                    raise AdapterError("judge provider tool-call identity was reused")
                call_ids = tuple(call.call_id for call in internal_calls)
                events.append(
                    _event(
                        self._ids,
                        role="assistant",
                        content={"text": content, "tool_calls": visible_calls},
                        tool_call_ids=call_ids,
                    )
                )
                sdk_messages.append({"role": "assistant", "content": content, "tool_calls": sdk_calls})
                results = workspace_tools.execute_group(internal_calls)
                by_call = {result.call_id: result for result in results}
                if set(by_call) != set(call_ids):
                    raise AdapterError("judge workspace runtime omitted or replaced a call")
                for call_id in call_ids:
                    result = by_call[call_id]
                    visible_result = {
                        "status": result.status,
                        "output": result.output,
                        "error_code": result.error_code,
                        "receipt_blake3": result.receipt_blake3,
                        "inspected_evidence_refs": result.inspected_evidence_refs,
                        "content_inspection": result.content_inspection,
                    }
                    events.append(
                        _event(
                            self._ids,
                            role="tool",
                            content=visible_result,
                            tool_call_ids=(call_id,),
                        )
                    )
                    sdk_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": provider_ids[call_id],
                            "content": json.dumps(
                                canonical_value(visible_result),
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        }
                    )
                    content_inspected = content_inspected or result.content_inspection
                    used_workspace_tool = True
                continue
            if not used_workspace_tool or (has_workspace_files and not content_inspected):
                raise AdapterError("agent judge returned a score without inspecting the workspace")
            if not content.strip():
                raise AdapterError("agent judge terminal response is empty")
            final_content = content
            events.append(_event(self._ids, role="assistant", content=content))
            break
        if final_content is None:
            raise AdapterError("agent judge exceeded the fixed model-turn bound")
        agent_trace = workspace_tools.trace(
            policy_events=tuple(events),
            provider_turn_count=provider_turn_count,
        )
        content = final_content
        try:
            value = json.loads(content)
        except (ValueError, RecursionError):
            raise AdapterError("judge response is not JSON") from None
        if not isinstance(value, dict) or set(value) != {"item_scores", "hard_gates_passed", "summary"}:
            raise AdapterError("judge response shape differs")
        rows = value["item_scores"]
        if not isinstance(rows, list):
            raise AdapterError("judge item scores must be a list")
        scores: list[RubricItemScore] = []
        hard_gates_passed = True
        for row, item in zip(rows, items, strict=False):
            if not isinstance(row, dict) or set(row) != {"item_id", "score", "evidence_refs", "rationale"}:
                raise AdapterError("judge item score shape differs")
            refs = row["evidence_refs"]
            if (
                not isinstance(refs, list)
                or not refs
                or any(not isinstance(ref, str) or not ref for ref in refs)
                or len(set(refs)) != len(refs)
            ):
                raise AdapterError("judge evidence references differ")
            for reference in refs:
                if reference.startswith("workspace:") and reference not in agent_trace.inspected_evidence_refs:
                    raise AdapterError("judge score cites workspace evidence it did not inspect")
            raw_score = row["score"]
            if type(raw_score) not in {int, float} or not math.isfinite(float(raw_score)):
                raise AdapterError("judge item score must be a finite number")
            score = float(raw_score)
            score_bps = round(score * 10_000)
            allowed_bps = {
                int(level["score_bps"])
                for level in item["partial_credit"]["levels"]
            }
            if not 0.0 <= score <= 1.0 or score_bps not in allowed_bps:
                raise AdapterError("judge item score is not an allowed compiled rubric level")
            if not isinstance(row["rationale"], str) or not row["rationale"].strip():
                raise AdapterError("judge item rationale differs")
            gate = item.get("hard_gate")
            if isinstance(gate, dict) and score_bps < int(gate["minimum_score_bps"]):
                hard_gates_passed = False
            scores.append(
                RubricItemScore(
                    item_id=row["item_id"],
                    score=score,
                    evidence_refs=tuple(refs),
                    rationale=row["rationale"],
                )
            )
        if [score.item_id for score in scores] != [item["item_id"] for item in items]:
            raise AdapterError("judge score rows do not match the compiled rubric table")
        if type(value["hard_gates_passed"]) is not bool or not isinstance(value["summary"], str):
            raise AdapterError("judge gate or summary differs")
        if value["hard_gates_passed"] is not hard_gates_passed:
            raise AdapterError("judge hard-gate claim differs from compiled rubric levels")
        core = {
            "judgment_id": request.judgment_id,
            "judge_model_id": self.model_id,
            "rubric_digest": rubric.digest,
            "agent_trace": agent_trace,
            "item_scores": tuple(scores),
            "hard_gates_passed": value["hard_gates_passed"],
            "summary": value["summary"],
        }
        return JudgeAssessment(**core, assessment_blake3=blake3_hex(core))


class WeightedRubricRewarder:
    """Compute reward directly from the exact compiled table passed to the judge."""

    def compute(
        self,
        *,
        reward_id: str,
        rubric: CompiledRubricTable,
        assessment: JudgeAssessment,
        evidence: EvidenceBundle,
    ) -> RewardRecord:
        del evidence  # assessment evidence refs remain bound in the retained judgment
        items = _rubric_items(rubric)
        if assessment.rubric_digest != rubric.digest:
            raise AdapterError("judge and reward rubric digests differ")
        by_id = {row.item_id: row for row in assessment.item_scores}
        if set(by_id) != {item["item_id"] for item in items}:
            raise AdapterError("reward item coverage differs from compiled rubric")
        score_method = getattr(rubric, "score", None)
        if not callable(score_method):
            raise AdapterError("compiled rubric lacks its authoritative score method")
        item_scores_bps = {
            item["item_id"]: round(by_id[item["item_id"]].score * 10_000)
            for item in items
        }
        try:
            compiled_score = score_method(item_scores_bps, evaluation_id=reward_id)
        except (TypeError, ValueError) as exc:
            raise AdapterError(f"judge scores violate compiled rubric semantics: {exc}") from exc
        if (
            _member(compiled_score, "rubric_digest") != rubric.digest
            or type(_member(compiled_score, "reward_bps")) is not int
            or type(_member(compiled_score, "hard_gate_passed")) is not bool
        ):
            raise AdapterError("compiled rubric score receipt differs")
        hard_gates_passed = bool(_member(compiled_score, "hard_gate_passed"))
        if assessment.hard_gates_passed != hard_gates_passed:
            raise AdapterError("judge hard-gate claim differs from compiled rubric semantics")
        reward = int(_member(compiled_score, "reward_bps")) / 10_000
        core = {
            "reward_id": reward_id,
            "rubric_digest": rubric.digest,
            "item_scores": tuple(assessment.item_scores),
            "total_reward": reward,
            "hard_gates_passed": hard_gates_passed,
        }
        return RewardRecord(**core, reward_blake3=blake3_hex(core))


__all__ = [
    "AdapterError",
    "OpenAIStyleOpus5Judge",
    "OpenAIStyleRolloutAdapter",
    "WeightedRubricRewarder",
]
