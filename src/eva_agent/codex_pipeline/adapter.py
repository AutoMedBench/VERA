"""Schema-preserving Codex adapters for the synchronous EVA data pipeline.

The official Codex runtime is asynchronous and owns its complete inner tool
loop.  This module gives the synchronous pipeline a one-turn lifecycle and
projects the resulting receipt into the already-existing pipeline contracts.
It deliberately contains no retry loop and no process-wide concurrency gate.
"""

from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import time
from threading import Lock
from typing import Any, Callable, Mapping, Protocol, Sequence

from jsonschema import Draft202012Validator

from eva_agent.codex_runtime import (
    CodexRole,
    CodexRuntime,
    CodexRuntimeError,
    CodexSandbox,
    CodexSkill,
    CodexThreadOptions,
    CodexToolCall,
    CodexToolOffer,
    CodexTurnInput,
    CodexTurnReceipt,
    codex_core_mcp_resource_call_error,
    codex_core_mcp_resource_operation,
    verify_codex_turn_receipt,
)
from eva_agent.pipeline.adapters import _judge_evidence_projection, _rubric_items
from eva_agent.pipeline.contracts import (
    Cohort,
    CompiledRubricTable,
    EvidenceBundle,
    JudgeAgentTrace,
    JudgeAssessment,
    JudgeToolResult,
    JudgeRequest,
    JsonValue,
    ProviderRollout,
    RolloutRequest,
    RubricItemScore,
    RuntimeIdFactory,
    Stage,
    ToolCall,
    ToolResult,
    ToolRuntimePort,
    TrajectoryEvent,
    WorkspaceSnapshot,
    freeze_json,
)
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value
from eva_agent.pipeline.ids import RandomUUIDFactory
from eva_agent.pipeline.judge_workspace import JudgeWorkspaceTools, TOOL_NAMES
from eva_agent.pipeline.tools import MAXIMUM_PARALLEL_TOOL_CALLS

from .empty_code_rejection import (
    BOOTSTRAP_PATHS, empty_code_rejection_ids, legacy_execute_code_runtime,
    semantic_tool_failure,
)


class CodexPipelineError(ValueError):
    """The Codex/pipeline compatibility boundary failed closed.

    ``receipt`` is retained when a terminal provider turn exists but is not
    admissible.  Campaign infrastructure can quarantine that immutable
    evidence without treating it as a semantic model score.
    """

    def __init__(
        self,
        message: str,
        *,
        receipt: CodexTurnReceipt | None = None,
        category: str = "compatibility_contract",
        provenance: str = "eva_agent.codex_pipeline",
    ) -> None:
        super().__init__(message)
        self.receipt = receipt
        # These fields are deliberately host-authored.  The pipeline persists
        # them instead of arbitrary exception text, which may contain provider
        # payloads, paths, credentials, or other configuration values.
        self.safe_failure_category = category
        self.safe_failure_message = _safe_codex_failure_message(message)
        self.safe_failure_provenance = provenance


def _safe_codex_failure_message(message: str) -> str:
    """Return a bounded diagnostic that cannot contain interpolated details."""

    summary = message.split(":", 1)[0].strip()
    if (
        not summary
        or len(summary) > 160
        or any(ord(character) < 32 for character in summary)
        or "/" in summary
        or "\\" in summary
        or "=" in summary
        or "@" in summary
    ):
        return "Codex compatibility boundary failed closed"
    return summary


DEFAULT_ACTOR_SYSTEM_INSTRUCTION = (
    "Solve this EvaMed task through the offered tools, skills, and workspace evidence. "
    "Use search_skills and load_skill when helpful. Batch independent parallel-safe "
    "tool calls in the same turn. Do not substitute memorized medical claims for frozen "
    "evidence; retain verifiable evidence for every conclusion. Use only the exact mounted "
    "MCP tools unless the run recipe explicitly enables constrained native coding actions."
)

_NATIVE_ACTOR_INSTRUCTION = (
    " This run explicitly permits sequential Codex-native fileChange actions at S3, S4, or "
    "E2E only. Use MCP for every read. Never invoke commandExecution, a shell, interpreter, "
    "script, network action, absolute path, path traversal, symlink, or hardlink. Every changed "
    "workspace path must be declared by exactly one receipt-visible fileChange."
)


_MAXIMUM_ACTOR_SKILL_BYTES = 4 * 1024 * 1024


def _stable_actor_skill_bytes(
    skill: CodexSkill,
    *,
    receipt: CodexTurnReceipt | None = None,
) -> bytes:
    """Reopen one selected SkillInput without a check-then-open race.

    The SDK accepts a path rather than bytes, so the adapter verifies the path
    immediately before and after the turn.  Each individual reopen uses
    ``O_NOFOLLOW`` and compares the descriptor with the live directory entry;
    symlink/hardlink substitution and same-content inode replacement therefore
    fail closed instead of being hidden by a matching content digest.
    """

    source = Path(skill.path)

    def fail() -> None:
        raise CodexPipelineError("actor selected skill bytes differ", receipt=receipt)

    if str(source) != skill.path:
        fail()
    try:
        resolved_before = source.resolve(strict=True)
    except (OSError, RuntimeError):
        fail()
    if resolved_before != source:
        fail()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError:
        fail()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 1 <= before.st_size <= _MAXIMUM_ACTOR_SKILL_BYTES
        ):
            fail()
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                fail()
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            fail()
        after = os.fstat(descriptor)
        entry = os.stat(source, follow_symlinks=False)
    except OSError:
        fail()
    finally:
        os.close(descriptor)
    def fingerprint(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
    try:
        resolved_after = source.resolve(strict=True)
    except (OSError, RuntimeError):
        fail()
    payload = b"".join(chunks)
    if (
        fingerprint(before) != fingerprint(after)
        or fingerprint(after) != fingerprint(entry)
        or resolved_after != source
        or blake3_bytes(payload) != skill.content_blake3
    ):
        fail()
    return payload


def _verify_actor_skills(
    skills: Sequence[CodexSkill],
    *,
    workspace_root: str,
    receipt: CodexTurnReceipt | None = None,
) -> None:
    if not skills:
        return
    candidate_root = Path(workspace_root)
    try:
        resolved_candidate_root = candidate_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CodexPipelineError(
            "actor candidate workspace is unavailable", receipt=receipt
        ) from exc
    if resolved_candidate_root != candidate_root:
        raise CodexPipelineError(
            "actor candidate workspace uses unsafe topology", receipt=receipt
        )
    for skill in skills:
        try:
            skill_path = Path(skill.path).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise CodexPipelineError(
                "actor selected skill path is unavailable", receipt=receipt
            ) from exc
        if skill_path.is_relative_to(resolved_candidate_root):
            raise CodexPipelineError(
                "actor selected skill is inside the candidate workspace",
                receipt=receipt,
            )
        _stable_actor_skill_bytes(skill, receipt=receipt)


RuntimeFactory = Callable[[], CodexRuntime]
ActorRuntimeFactory = Callable[["CodexToolExecutionBridge"], CodexRuntime]
ActorOptionsFactory = Callable[
    [RolloutRequest, str, tuple[CodexToolOffer, ...], "CodexToolExecutionBridge"],
    CodexThreadOptions,
]
ActorSkillsFactory = Callable[[RolloutRequest], Sequence[CodexSkill]]
JudgeOptionsFactory = Callable[
    [JudgeRequest, tuple[CodexToolOffer, ...]], CodexThreadOptions
]


class CodexTurnRunnerPort(Protocol):
    """Synchronous one-turn port shared by ephemeral and persistent runners."""

    def run_once(
        self, options: CodexThreadOptions, turn_input: CodexTurnInput
    ) -> CodexTurnReceipt: ...

    def run_once_with_timeout(
        self,
        options: CodexThreadOptions,
        turn_input: CodexTurnInput,
        *,
        timeout_seconds: int,
    ) -> CodexTurnReceipt: ...


class TurnMCPBridgeFactoryPort(Protocol):
    """Bind one exact in-memory executor to one ephemeral Codex thread.

    Production callers use :class:`TurnMCPBridgeFactory`; keeping this as a
    structural port avoids coupling the adapters to a particular local
    transport implementation and makes the fail-closed boundary testable.
    """

    def open_actor(
        self,
        options: CodexThreadOptions,
        bridge: "CodexToolExecutionBridge",
        *,
        allow_schema_validation_rejections: bool = False,
    ) -> AbstractContextManager[CodexThreadOptions]: ...

    def open_judge(
        self,
        options: CodexThreadOptions,
        bridge: "CodexJudgeToolExecutionBridge",
    ) -> AbstractContextManager[CodexThreadOptions]: ...


def _require_runner(runner: CodexTurnRunnerPort) -> CodexTurnRunnerPort:
    if not callable(getattr(runner, "run_once", None)):
        raise CodexPipelineError("Codex turn runner must expose callable run_once")
    return runner


class SyncCodexTurnRunner:
    """Open, use once, and close an async runtime from a synchronous caller.

    A fresh runtime is required for every invocation.  This keeps concurrent
    weak/middle/strong pipeline workers independent and leaves all throttling
    to the campaign and Codex app-server.  Calling from an already-running
    event loop is rejected rather than nesting or leaking a loop.
    """

    def __init__(self, runtime_factory: RuntimeFactory) -> None:
        if not callable(runtime_factory):
            raise CodexPipelineError("Codex runtime factory is required")
        self._factory = runtime_factory

    def run_once(
        self, options: CodexThreadOptions, turn_input: CodexTurnInput
    ) -> CodexTurnReceipt:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise CodexPipelineError(
                "synchronous Codex pipeline adapter cannot run inside an active event loop"
            )

        async def lifecycle() -> CodexTurnReceipt:
            runtime = self._factory()
            if not isinstance(runtime, CodexRuntime):
                raise CodexPipelineError("runtime factory did not return CodexRuntime")
            async with runtime:
                handle = await runtime.start_thread(options)
                if handle.resumed:
                    raise CodexPipelineError("pipeline turns must start a fresh Codex thread")
                return await runtime.run_turn(handle, turn_input)

        # asyncio.run creates and closes one local loop; no semaphore or global
        # executor is introduced here, so pipeline worker threads overlap.
        return asyncio.run(lifecycle())

    def run_once_with_timeout(
        self,
        options: CodexThreadOptions,
        turn_input: CodexTurnInput,
        *,
        timeout_seconds: int,
    ) -> CodexTurnReceipt:
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 900:
            raise CodexPipelineError("Codex turn wall-clock timeout differs")

        async def bounded() -> CodexTurnReceipt:
            runtime = self._factory()
            if not isinstance(runtime, CodexRuntime):
                raise CodexPipelineError("runtime factory did not return CodexRuntime")
            async with runtime:
                handle = await runtime.start_thread(options)
                if handle.resumed:
                    raise CodexPipelineError("pipeline turns must start a fresh Codex thread")
                try:
                    return await asyncio.wait_for(
                        runtime.run_turn(handle, turn_input), timeout=timeout_seconds
                    )
                except TimeoutError:
                    raise CodexPipelineError(
                        "Codex turn exceeded its wall-clock timeout",
                        category="turn_wall_clock_timeout",
                    ) from None

        return asyncio.run(bounded())


def _policy_wall_time(value: Any) -> int:
    if not isinstance(value, Mapping):
        return 900
    # Only the host-authored S1 plan contract owns this execution budget.
    # Public benchmark/evidence material may legitimately contain an unrelated
    # field with the same name and must not be allowed to shorten a turn.
    policy = value.get("policy_visible_context", value)
    if not isinstance(policy, Mapping):
        return 900
    episode_context = policy.get("episode_context")
    execution_binding = (
        episode_context.get("execution_binding")
        if isinstance(episode_context, Mapping)
        else None
    )
    s1 = (
        execution_binding.get("s1_plan_contract")
        if isinstance(execution_binding, Mapping)
        else policy.get("s1_plan_contract")
    )
    if not isinstance(s1, Mapping):
        return 900
    budgets = s1.get("budgets")
    if not isinstance(budgets, Mapping):
        return 900
    wall_time = budgets.get("wall_time_seconds")
    if type(wall_time) is not int or wall_time <= 0:
        return 900
    return min(900, wall_time)


def _bounded_run_once(
    runner: CodexTurnRunnerPort,
    options: CodexThreadOptions,
    turn_input: CodexTurnInput,
    *,
    policy_context: Any,
    timeout_seconds: int | None = None,
) -> CodexTurnReceipt:
    bounded = getattr(runner, "run_once_with_timeout", None)
    if not callable(bounded):
        raise CodexPipelineError("Codex turn runner lacks hard timeout support")
    effective_timeout = (
        _policy_wall_time(policy_context)
        if timeout_seconds is None
        else timeout_seconds
    )
    if type(effective_timeout) is not int or not 1 <= effective_timeout <= 900:
        raise CodexPipelineError("Codex turn wall-clock timeout differs")
    try:
        return bounded(
            options,
            turn_input,
            timeout_seconds=effective_timeout,
        )
    except CodexRuntimeError as exc:
        if str(exc) != "Codex turn exceeded its wall-clock timeout":
            raise
        raise CodexPipelineError(
            "Codex turn exceeded its wall-clock timeout",
            category="turn_wall_clock_timeout",
        ) from exc


def _tool_result_core(result: ToolResult) -> Mapping[str, Any]:
    return {
        "result_id": result.result_id,
        "call_id": result.call_id,
        "name": result.name,
        "frontier": result.frontier,
        "parallel_group_id": result.parallel_group_id,
        "status": result.status,
        "output": result.output,
        "error_code": result.error_code,
        "workspace_before_blake3": result.workspace_before_blake3,
        "workspace_after_blake3": result.workspace_after_blake3,
    }


def _judge_tool_result_core(result: JudgeToolResult) -> Mapping[str, Any]:
    return {
        "result_id": result.result_id,
        "call_id": result.call_id,
        "name": result.name,
        "arguments": result.arguments,
        "frontier": result.frontier,
        "parallel_group_id": result.parallel_group_id,
        "status": result.status,
        "output": result.output,
        "error_code": result.error_code,
        "inspected_evidence_refs": result.inspected_evidence_refs,
        "content_inspection": result.content_inspection,
        "evidence_bundle_blake3": result.evidence_bundle_blake3,
    }


@dataclass(frozen=True)
class CodexToolBridgeObservation:
    """One MCP-safe envelope around the existing pipeline ``ToolResult``."""

    call: ToolCall
    result: ToolResult
    structured_content: Mapping[str, JsonValue]
    supplemental_content: tuple = ()


@dataclass(frozen=True)
class CodexToolSchemaValidationRejectionObservation:
    """Bridge-issued proof that invalid model arguments never reached a tool."""

    offer: CodexToolOffer
    arguments_blake3: str
    structured_content: Mapping[str, JsonValue]


class CodexToolExecutionBridge:
    """Make MCP execution and the pipeline ToolTrace the very same operation.

    A transport adapter may expose ``execute_group`` to Codex over local MCP.
    It must preserve each returned ``structured_content`` object verbatim in
    the MCP result.  Calling handlers elsewhere and replaying them here is not
    accepted, because that would duplicate side effects.
    """

    def __init__(
        self,
        tools: ToolRuntimePort,
        *,
        id_factory: RuntimeIdFactory,
    ) -> None:
        self._tools = tools
        self._ids = id_factory
        self._issued: dict[str, CodexToolBridgeObservation] = {}
        self._schema_rejections: dict[
            str, CodexToolSchemaValidationRejectionObservation
        ] = {}
        self._lock = Lock()

    @staticmethod
    def _validation_issues(
        offer: CodexToolOffer, arguments: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], ...]:
        errors = Draft202012Validator(dict(offer.input_schema)).iter_errors(
            dict(arguments)
        )
        return tuple(
            sorted(
                (
                    {
                        "validator": str(error.validator),
                        "instance_path": tuple(error.absolute_path),
                        "schema_path": tuple(error.absolute_schema_path),
                        "required_properties": tuple(
                            str(value)
                            for value in (
                                error.validator_value
                                if error.validator == "required"
                                and isinstance(error.validator_value, (list, tuple))
                                and isinstance(error.instance, Mapping)
                                else ()
                            )
                            if isinstance(value, str)
                            and value not in error.instance
                        ),
                        "expected_types": tuple(
                            str(value)
                            for value in (
                                (error.validator_value,)
                                if error.validator == "type"
                                and isinstance(error.validator_value, str)
                                else error.validator_value
                                if error.validator == "type"
                                and isinstance(error.validator_value, (list, tuple))
                                else ()
                            )
                            if isinstance(value, str)
                        ),
                    }
                    for error in errors
                ),
                key=lambda row: json.dumps(
                    canonical_value(row),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )

    def record_schema_validation_rejection(
        self,
        *,
        offer: CodexToolOffer,
        arguments: Mapping[str, Any],
        validation_issues: Sequence[Mapping[str, Any]],
        offered_catalog_blake3: str,
    ) -> CodexToolSchemaValidationRejectionObservation:
        """Commit one exact schema rejection without dispatching a host tool."""

        if not isinstance(offer, CodexToolOffer) or not isinstance(arguments, Mapping):
            raise CodexPipelineError("schema-validation rejection input differs")
        frozen_arguments = canonical_value(arguments)
        frozen_issues = canonical_value(tuple(validation_issues))
        expected_issues = canonical_value(self._validation_issues(offer, arguments))
        if (
            not isinstance(frozen_arguments, dict)
            or not isinstance(frozen_issues, list)
            or not frozen_issues
            or frozen_issues != expected_issues
            or not isinstance(offered_catalog_blake3, str)
            or len(offered_catalog_blake3) != 64
        ):
            raise CodexPipelineError("schema-validation rejection evidence differs")
        trace = self._tools.trace()
        rejection_id = self._ids.new("codex-schema-validation-rejection")
        core = {
            "schema": "eva.codex-tool-schema-validation-rejection.v1",
            "rejection_id": rejection_id,
            "fully_qualified_name": offer.fully_qualified_name,
            "arguments_blake3": blake3_hex(frozen_arguments),
            "input_schema_blake3": blake3_hex(offer.input_schema),
            "offered_catalog_blake3": offered_catalog_blake3,
            "validation_issues": frozen_issues,
            "tool_trace_blake3_at_rejection": trace.trace_blake3,
            "host_execution_attempted": False,
            "host_result_claimed": False,
        }
        document = freeze_json(
            {**core, "rejection_blake3": blake3_hex(core)}
        )
        assert isinstance(document, Mapping)
        observation = CodexToolSchemaValidationRejectionObservation(
            offer=offer,
            arguments_blake3=core["arguments_blake3"],
            structured_content=document,
        )
        with self._lock:
            if rejection_id in self._schema_rejections:
                raise CodexPipelineError(
                    "schema-validation rejection identity was reused"
                )
            self._schema_rejections[rejection_id] = observation
        return observation

    def bind_schema_validation_rejections(
        self,
        receipt: CodexTurnReceipt,
        offers: tuple[CodexToolOffer, ...],
    ) -> Mapping[str, Mapping[str, JsonValue]]:
        """Bind model-visible rejected calls to in-turn bridge-issued proofs."""

        by_name = {offer.fully_qualified_name: offer for offer in offers}
        catalog_blake3 = blake3_hex(
            tuple(offer.canonical_catalog_entry() for offer in offers)
        )
        claimed: set[str] = set()
        bound: dict[str, Mapping[str, JsonValue]] = {}
        for call in receipt.tool_calls:
            document = _mcp_schema_validation_rejection_object(call)
            if document is None:
                continue
            expected_keys = {
                "schema",
                "rejection_id",
                "fully_qualified_name",
                "arguments_blake3",
                "input_schema_blake3",
                "offered_catalog_blake3",
                "validation_issues",
                "tool_trace_blake3_at_rejection",
                "host_execution_attempted",
                "host_result_claimed",
                "rejection_blake3",
            }
            rejection_id = document.get("rejection_id")
            with self._lock:
                issued = (
                    self._schema_rejections.get(rejection_id)
                    if isinstance(rejection_id, str)
                    else None
                )
            offer = by_name.get(call.fully_qualified_name or "")
            core = {
                key: document[key]
                for key in document
                if key != "rejection_blake3"
            }
            if (
                set(document) != expected_keys
                or document.get("schema")
                != "eva.codex-tool-schema-validation-rejection.v1"
                or issued is None
                or rejection_id in claimed
                or offer is None
                or document != canonical_value(issued.structured_content)
                or issued.offer != offer
                or document.get("fully_qualified_name")
                != call.fully_qualified_name
                or document.get("arguments_blake3")
                != blake3_hex(call.arguments)
                or document.get("arguments_blake3") != issued.arguments_blake3
                or document.get("input_schema_blake3")
                != blake3_hex(offer.input_schema)
                or document.get("offered_catalog_blake3") != catalog_blake3
                or document.get("validation_issues")
                != canonical_value(self._validation_issues(offer, call.arguments))
                or document.get("host_execution_attempted") is not False
                or document.get("host_result_claimed") is not False
                or document.get("rejection_blake3") != blake3_hex(core)
            ):
                raise CodexPipelineError(
                    "Codex schema-validation rejection receipt differs",
                    receipt=receipt,
                )
            claimed.add(rejection_id)
            bound[call.tool_call_id] = document
        with self._lock:
            issued_ids = set(self._schema_rejections)
        if claimed != issued_ids:
            raise CodexPipelineError(
                "Codex schema-validation rejection inventory differs",
                receipt=receipt,
            )
        return bound

    def execute_group(
        self, calls: Sequence[tuple[str, Mapping[str, Any]]]
    ) -> tuple[CodexToolBridgeObservation, ...]:
        values = tuple(calls)
        if not values:
            raise CodexPipelineError("Codex MCP bridge call group is empty")
        prepared = tuple(
            ToolCall(
                call_id=self._ids.new("codex-pipeline-tool-call"),
                name=name,
                arguments=arguments,
            )
            for name, arguments in values
        )
        # One call into the existing runtime is the one and only semantic tool
        # execution. Its results are therefore already present in ToolTrace.
        results = self._tools.execute(prepared)
        by_id = {result.call_id: result for result in results}
        if set(by_id) != {call.call_id for call in prepared}:
            raise CodexPipelineError("tool runtime omitted or replaced a bridged call")
        observations: list[CodexToolBridgeObservation] = []
        with self._lock:
            for call in prepared:
                result = by_id[call.call_id]
                if result.name != call.name or result.receipt_blake3 != blake3_hex(
                    _tool_result_core(result)
                ):
                    raise CodexPipelineError("bridged pipeline ToolResult differs")
                core = {
                    "schema": "eva.codex-pipeline-tool-observation.v1",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                    "tool_result": canonical_value(result),
                }
                document = freeze_json({**core, "bridge_receipt_blake3": blake3_hex(core)})
                assert isinstance(document, Mapping)
                observation = CodexToolBridgeObservation(
                    call, result, document, self._tools.supplemental_content(call.call_id)
                )
                if call.call_id in self._issued:
                    raise CodexPipelineError("Codex MCP bridge call identity was reused")
                self._issued[call.call_id] = observation
                observations.append(observation)
        return tuple(observations)

    def bind_receipt(
        self,
        receipt: CodexTurnReceipt,
        groups: tuple[tuple[CodexToolCall, ...], ...],
    ) -> tuple[Mapping[str, ToolResult], Any]:
        """Reopen Codex MCP outputs against the already-populated ToolTrace."""

        mapping: dict[str, ToolResult] = {}
        claimed: set[str] = set()
        execution_frontiers: list[tuple[int, str, int]] = []
        previous_execution_frontier = -1
        for group in groups:
            group_results: list[ToolResult] = []
            for codex_call in group:
                document = canonical_value(_mcp_result_object(codex_call))
                if not isinstance(document, dict) or set(document) != {
                    "schema",
                    "call_id",
                    "name",
                    "arguments",
                    "tool_result",
                    "bridge_receipt_blake3",
                } or document["schema"] != "eva.codex-pipeline-tool-observation.v1":
                    raise CodexPipelineError(
                        "Codex MCP call lacks a pipeline execution receipt", receipt=receipt
                    )
                core = {key: document[key] for key in document if key != "bridge_receipt_blake3"}
                if document["bridge_receipt_blake3"] != blake3_hex(core):
                    raise CodexPipelineError("Codex MCP bridge receipt differs", receipt=receipt)
                call_id = document["call_id"]
                with self._lock:
                    issued = self._issued.get(call_id) if isinstance(call_id, str) else None
                if (
                    issued is None
                    or call_id in claimed
                    or document != canonical_value(issued.structured_content)
                    or document["name"] != codex_call.mcp_tool
                    or canonical_value(document["arguments"])
                    != canonical_value(codex_call.arguments)
                ):
                    raise CodexPipelineError(
                        "Codex MCP bridge identity or arguments differ", receipt=receipt
                    )
                claimed.add(call_id)
                from eva_agent.pipeline.execution_diagnostics import diagnostic_content
                try:
                    supplemental = diagnostic_content(codex_call.output, call_id=call_id)
                except (ValueError, TypeError) as exc:
                    raise CodexPipelineError("Codex execution diagnostic differs", receipt=receipt) from exc
                if canonical_value(supplemental) != canonical_value(issued.supplemental_content):
                    raise CodexPipelineError("Codex execution diagnostic differs", receipt=receipt)
                if supplemental:
                    native_content = canonical_value(codex_call.output)["result"]["content"]
                    if native_content != canonical_value(issued.supplemental_content):
                        raise CodexPipelineError("Codex supplemental content differs", receipt=receipt)
                mapping[codex_call.tool_call_id] = issued.result
                group_results.append(issued.result)
            # One Codex transport frontier may contain a mixture of read-only
            # parallel-safe and mutating/serial calls.  ParallelToolRuntime
            # deliberately schedules that mixture as multiple execution
            # frontiers, and may run the read-only subset before an earlier
            # declared mutating call.  Preserve the provider-visible group in
            # the projected trajectory while verifying the exact execution
            # partition recorded by ToolTrace.
            local: dict[int, tuple[str, int]] = {}
            for result in group_results:
                observed = local.get(result.frontier)
                if observed is None:
                    local[result.frontier] = (result.parallel_group_id, 1)
                elif observed[0] == result.parallel_group_id:
                    local[result.frontier] = (observed[0], observed[1] + 1)
                else:
                    raise CodexPipelineError(
                        "Codex MCP transport frontier differs from ToolTrace",
                        receipt=receipt,
                    )
            frontier_numbers = tuple(sorted(local))
            if (
                not frontier_numbers
                or frontier_numbers
                != tuple(range(frontier_numbers[0], frontier_numbers[-1] + 1))
                or frontier_numbers[0] <= previous_execution_frontier
            ):
                raise CodexPipelineError(
                    "Codex MCP transport frontier differs from ToolTrace", receipt=receipt
                )
            for frontier in frontier_numbers:
                parallel_group_id, size = local[frontier]
                execution_frontiers.append((frontier, parallel_group_id, size))
            previous_execution_frontier = frontier_numbers[-1]
        trace = self._tools.trace()
        trace_ids = tuple(result.call_id for result in trace.results)
        trace_by_id = {result.call_id: result for result in trace.results}
        if (
            claimed != set(self._issued)
            or set(trace_ids) != claimed
            or any(
                canonical_value(trace_by_id[call_id])
                != canonical_value(observation.result)
                for call_id, observation in self._issued.items()
            )
            or trace.declared_call_ids != tuple(self._issued)
            or trace.joined_call_ids != trace_ids
            or tuple(row[0] for row in execution_frontiers)
            != tuple(range(trace.frontier_count))
            or len({row[1] for row in execution_frontiers})
            != len(execution_frontiers)
            or trace.frontier_count != len(execution_frontiers)
            or trace.max_parallelism_observed
            != (max((row[2] for row in execution_frontiers), default=0))
            or trace.retry_count != 0
        ):
            raise CodexPipelineError(
                "Codex bridge receipts and pipeline ToolTrace differ", receipt=receipt
            )
        return mapping, trace


@dataclass(frozen=True)
class CodexJudgeToolBridgeObservation:
    """One MCP-safe envelope around one immutable judge-workspace read."""

    call: ToolCall
    result: JudgeToolResult
    structured_content: Mapping[str, JsonValue]


@dataclass(frozen=True)
class JudgeFormatReadSnapshot:
    """Eligibility-only local observation; NOT a canonical/native Judge trace."""
    results: tuple[JudgeToolResult, ...]
    declared_call_ids: tuple[str, ...]
    joined_call_ids: tuple[str, ...]
    inspected_evidence_refs: tuple[str, ...]
    frontier_count: int
    provider_turn_count: int
    retry_count: int
    evidence_bundle_blake3: str
    snapshot_blake3: str


class CodexJudgeToolExecutionBridge:
    """Execute judge MCP calls once against the committed evidence bundle.

    The old compatibility path replayed tool results after the provider turn.
    A real MCP server must instead invoke ``JudgeWorkspaceTools`` while the
    turn is live.  This bridge commits those already-executed observations and
    later binds the Codex receipt to them without issuing a second read.
    """

    def __init__(
        self,
        tools: JudgeWorkspaceTools,
        *,
        id_factory: RuntimeIdFactory,
        maximum_total_calls: int | None = None,
        maximum_execution_groups: int | None = None,
    ) -> None:
        if not isinstance(tools, JudgeWorkspaceTools):
            raise CodexPipelineError("judge MCP bridge requires immutable workspace tools")
        self._tools = tools
        self._ids = id_factory
        for label, value in (
            ("total-call", maximum_total_calls),
            ("execution-group", maximum_execution_groups),
        ):
            if value is not None and (type(value) is not int or value < 1):
                raise CodexPipelineError(f"judge MCP {label} bound differs")
        self._maximum_total_calls = maximum_total_calls
        self._maximum_execution_groups = maximum_execution_groups
        self._calls_started = 0
        self._execution_groups_started = 0
        self._issued: dict[str, CodexJudgeToolBridgeObservation] = {}
        self._lock = Lock()

    def execute_group(
        self, calls: Sequence[tuple[str, Mapping[str, Any]]]
    ) -> tuple[CodexJudgeToolBridgeObservation, ...]:
        values = tuple(calls)
        if not values:
            raise CodexPipelineError("judge MCP bridge call group is empty")
        with self._lock:
            next_calls = self._calls_started + len(values)
            next_groups = self._execution_groups_started + 1
            if (
                self._maximum_total_calls is not None
                and next_calls > self._maximum_total_calls
            ):
                raise CodexPipelineError("judge MCP total-call bound exceeded")
            if (
                self._maximum_execution_groups is not None
                and next_groups > self._maximum_execution_groups
            ):
                raise CodexPipelineError("judge MCP execution-group bound exceeded")
            self._calls_started = next_calls
            self._execution_groups_started = next_groups
        prepared = tuple(
            ToolCall(
                call_id=self._ids.new("codex-judge-tool-call"),
                name=name,
                arguments=arguments,
            )
            for name, arguments in values
        )
        # This is the only invocation of JudgeWorkspaceTools for these calls.
        results = self._tools.execute_group(prepared)
        by_id = {result.call_id: result for result in results}
        if set(by_id) != {call.call_id for call in prepared}:
            raise CodexPipelineError("judge tool runtime omitted or replaced a bridged call")
        observations: list[CodexJudgeToolBridgeObservation] = []
        with self._lock:
            for call in prepared:
                result = by_id[call.call_id]
                if (
                    result.name != call.name
                    or canonical_value(result.arguments) != canonical_value(call.arguments)
                    or result.receipt_blake3 != blake3_hex(_judge_tool_result_core(result))
                ):
                    raise CodexPipelineError("bridged judge ToolResult differs")
                core = {
                    "schema": "eva.codex-judge-tool-observation.v1",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                    "judge_tool_result": canonical_value(result),
                }
                document = freeze_json({**core, "bridge_receipt_blake3": blake3_hex(core)})
                assert isinstance(document, Mapping)
                observation = CodexJudgeToolBridgeObservation(call, result, document)
                if call.call_id in self._issued:
                    raise CodexPipelineError("judge MCP bridge call identity was reused")
                self._issued[call.call_id] = observation
                observations.append(observation)
        return tuple(observations)

    def snapshot_for_verdict_correction(self):
        """Read completed, locally issued tool evidence without executing any tool."""
        with self._lock:
            if not self._calls_started or self._calls_started != len(self._issued):
                raise CodexPipelineError("judge correction requires quiescent completed tool calls")
            issued = {row.result.call_id: row.result for row in self._issued.values()}
            results = tuple(issued.values())
            identifiers = tuple(row.call_id for row in results)
            bundle = results[0].evidence_bundle_blake3
            core = {
                "results": results, "declared_call_ids": identifiers,
                "joined_call_ids": identifiers,
                "inspected_evidence_refs": tuple(sorted({ref for row in results
                    if row.status == "completed" for ref in row.inspected_evidence_refs})),
                "frontier_count": self._execution_groups_started,
                "provider_turn_count": self._execution_groups_started + 1,
                "retry_count": 0, "evidence_bundle_blake3": bundle,
            }
            trace = JudgeFormatReadSnapshot(**core, snapshot_blake3=blake3_hex(core))
            if (
                len(trace.results) != len(issued)
                or set(trace.declared_call_ids) != set(issued)
                or set(trace.joined_call_ids) != set(issued)
                or trace.frontier_count != self._execution_groups_started
                or any(row.status != "completed" or row != issued.get(row.call_id)
                       or row.evidence_bundle_blake3 != bundle
                       or row.receipt_blake3 != blake3_hex(_judge_tool_result_core(row))
                       for row in trace.results)
            ):
                raise CodexPipelineError("judge correction live tool evidence differs")
            return trace

    def execution_groups_for_receipt(
        self,
        receipt: CodexTurnReceipt,
        groups: tuple[tuple[CodexToolCall, ...], ...],
    ) -> tuple[tuple[CodexToolCall, ...], ...]:
        """Partition transport calls by exact locally issued read results.

        Opt-in callers retain the original transport groups separately. No
        observed call, result, UUID, or frontier is rewritten or re-executed.
        The ordinary strict binder must still validate the returned partition.
        """
        local: dict[int, tuple[str, list[CodexToolCall]]] = {}
        claimed: set[str] = set()
        parallel_ids: set[str] = set()
        for group in groups:
            if not group:
                raise CodexPipelineError("judge execution partition has an empty transport group", receipt=receipt)
            for call in group:
                document = canonical_value(_mcp_result_object(call))
                call_id = document.get("call_id")
                with self._lock:
                    issued = self._issued.get(call_id) if isinstance(call_id, str) else None
                if (
                    issued is None or call_id in claimed
                    or document != canonical_value(issued.structured_content)
                    or document.get("name") != call.mcp_tool
                    or canonical_value(call.arguments) != canonical_value(issued.call.arguments)
                    or issued.result.receipt_blake3 != blake3_hex(_judge_tool_result_core(issued.result))
                ):
                    raise CodexPipelineError("judge execution partition lacks an exact issued result", receipt=receipt)
                claimed.add(call_id)
                result = issued.result
                # Transport notifications are not local dependency evidence.
                # The unmodified issued result supplies the actual frontier.
                if result.frontier not in local:
                    local[result.frontier] = (result.parallel_group_id, [])
                parallel_id, calls = local[result.frontier]
                if parallel_id != result.parallel_group_id:
                    raise CodexPipelineError("judge execution partition group identity differs", receipt=receipt)
                calls.append(call)
        with self._lock:
            issued_ids = set(self._issued)
        if claimed != issued_ids:
            raise CodexPipelineError("judge execution partition omits issued calls", receipt=receipt)
        frontiers = tuple(sorted(local))
        if frontiers != tuple(range(len(frontiers))):
            raise CodexPipelineError("judge execution partition is not continuous and disjoint", receipt=receipt)
        partition = []
        for frontier in frontiers:
            parallel_id, calls = local[frontier]
            if parallel_id in parallel_ids:
                raise CodexPipelineError("judge execution partition reuses a parallel group", receipt=receipt)
            parallel_ids.add(parallel_id)
            partition.append(tuple(calls))
        return tuple(partition)

    def bind_receipt(
        self,
        receipt: CodexTurnReceipt,
        groups: tuple[tuple[CodexToolCall, ...], ...],
    ) -> Mapping[str, JudgeToolResult]:
        """Bind provider-visible results to calls already executed in-turn."""

        mapping: dict[str, JudgeToolResult] = {}
        claimed: set[str] = set()
        external_frontiers: list[tuple[int, str, int]] = []
        for group in groups:
            group_results: list[JudgeToolResult] = []
            for codex_call in group:
                document = canonical_value(_mcp_result_object(codex_call))
                if (
                    not isinstance(document, dict)
                    or set(document)
                    != {
                        "schema",
                        "call_id",
                        "name",
                        "arguments",
                        "judge_tool_result",
                        "bridge_receipt_blake3",
                    }
                    or document["schema"] != "eva.codex-judge-tool-observation.v1"
                ):
                    raise CodexPipelineError(
                        "judge MCP call lacks an immutable execution receipt", receipt=receipt
                    )
                core = {
                    key: document[key]
                    for key in document
                    if key != "bridge_receipt_blake3"
                }
                if document["bridge_receipt_blake3"] != blake3_hex(core):
                    raise CodexPipelineError(
                        "judge MCP bridge receipt differs", receipt=receipt
                    )
                call_id = document["call_id"]
                with self._lock:
                    issued = self._issued.get(call_id) if isinstance(call_id, str) else None
                if (
                    issued is None
                    or call_id in claimed
                    or document != canonical_value(issued.structured_content)
                    or document["name"] != codex_call.mcp_tool
                    or canonical_value(document["arguments"])
                    != canonical_value(codex_call.arguments)
                ):
                    raise CodexPipelineError(
                        "judge MCP bridge identity or arguments differ", receipt=receipt
                    )
                claimed.add(call_id)
                mapping[codex_call.tool_call_id] = issued.result
                group_results.append(issued.result)
            frontiers = {result.frontier for result in group_results}
            parallel_groups = {result.parallel_group_id for result in group_results}
            if len(frontiers) != 1 or len(parallel_groups) != 1:
                raise CodexPipelineError(
                    "judge MCP transport frontier differs from execution", receipt=receipt
                )
            external_frontiers.append(
                (next(iter(frontiers)), next(iter(parallel_groups)), len(group_results))
            )
        with self._lock:
            issued_ids = set(self._issued)
        if claimed != issued_ids or len({row[:2] for row in external_frontiers}) != len(groups):
            raise CodexPipelineError(
                "judge MCP receipts and immutable execution differ", receipt=receipt
            )
        return mapping


def _event(
    ids: RuntimeIdFactory,
    *,
    role: str,
    content: Any,
    tool_call_ids: Sequence[str] = (),
) -> TrajectoryEvent:
    core = {
        "event_id": ids.new("codex-pipeline-trajectory-event"),
        "role": role,
        "content": freeze_json(content),
        "tool_call_ids": tuple(tool_call_ids),
    }
    return TrajectoryEvent(**core, event_blake3=blake3_hex(core))


def _require_terminal(
    receipt: CodexTurnReceipt,
    *,
    options: CodexThreadOptions,
    expected_role: CodexRole,
) -> None:
    _verify_receipt_identity(receipt, options=options, expected_role=expected_role)
    if receipt.status != "completed":
        raise CodexPipelineError(
            f"Codex turn status is not completed: {receipt.status}", receipt=receipt
        )
    if not isinstance(receipt.final_response, str) or not receipt.final_response.strip():
        raise CodexPipelineError("Codex turn lacks a terminal response", receipt=receipt)
    final_messages = []
    for event in receipt.events:
        if event.method != "item/completed":
            continue
        payload = canonical_value(event.payload)
        item = payload.get("item") if isinstance(payload, dict) else None
        if (
            isinstance(item, dict)
            and item.get("type") == "agentMessage"
            and item.get("phase") in {"final_answer", "finalAnswer"}
            and isinstance(item.get("text"), str)
        ):
            final_messages.append(item["text"])
    if not final_messages or final_messages[-1] != receipt.final_response:
        raise CodexPipelineError("Codex terminal response provenance differs", receipt=receipt)


def _verify_receipt_identity(
    receipt: CodexTurnReceipt,
    *,
    options: CodexThreadOptions,
    expected_role: CodexRole,
) -> None:
    """Verify signed runtime identity independently of semantic turn status."""

    try:
        verify_codex_turn_receipt(receipt)
    except Exception as exc:
        raise CodexPipelineError("Codex receipt verification failed", receipt=receipt) from exc
    if (
        receipt.role is not expected_role
        or receipt.model != options.model
        or receipt.provider != options.provider
        or receipt.sandbox is not options.sandbox
        or receipt.thread_resumed
        or receipt.config_keys != options.config_keys
    ):
        raise CodexPipelineError("Codex receipt identity differs", receipt=receipt)


def _canonical_actor_tools(
    request: RolloutRequest, *, server: str
) -> tuple[CodexToolOffer, ...]:
    if not server or "/" in server:
        raise CodexPipelineError("actor MCP server name differs")
    offers: list[CodexToolOffer] = []
    names: set[str] = set()
    for raw in request.available_tools:
        value = canonical_value(raw)
        if not isinstance(value, dict) or set(value) != {"type", "function"}:
            raise CodexPipelineError("pipeline tool transport schema differs")
        function = value["function"]
        if not isinstance(function, dict) or set(function) != {
            "name",
            "description",
            "parameters",
            "x-eva-kind",
            "x-eva-parallel-safe",
        }:
            raise CodexPipelineError("pipeline tool definition fields differ")
        name = function["name"]
        description = function["description"]
        schema = function["parameters"]
        parallel_safe = function["x-eva-parallel-safe"]
        if (
            function["x-eva-kind"] not in {"tool", "skill"}
            or not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(description, str)
            or not description
            or not isinstance(schema, dict)
            or type(parallel_safe) is not bool
        ):
            raise CodexPipelineError("pipeline tool definition identity differs")
        names.add(name)
        offers.append(
            CodexToolOffer(
                fully_qualified_name=f"{server}/{name}",
                description=description,
                input_schema=schema,
                visibility="public",
                parallel_safe=parallel_safe,
            )
        )
    return tuple(offers)


def _workspace_root(tools: ToolRuntimePort) -> str:
    """Resolve the existing pipeline sandbox without inventing another root."""

    candidates = (
        getattr(tools, "workspace_root", None),
        getattr(getattr(tools, "workspace", None), "root", None),
        getattr(getattr(tools, "_workspace", None), "root", None),
    )
    for candidate in candidates:
        if candidate is not None:
            path = Path(candidate)
            if path.is_absolute() and ".." not in path.parts:
                return str(path)
    raise CodexPipelineError("tool runtime does not expose its bound workspace root")


def _bound_workspace_snapshot(
    tools: ToolRuntimePort,
    *,
    expected_root: str,
    label: str,
    receipt: CodexTurnReceipt | None = None,
) -> WorkspaceSnapshot:
    """Capture the exact pipeline workspace around a native-capable turn.

    Native effects cannot be inferred from a provider event alone.  The
    adapter therefore requires the already-bound pipeline sandbox to expose
    its immutable snapshot operation and commits both sides of the turn.  No
    new public port or persisted sandbox schema is introduced.
    """

    candidates = (
        getattr(tools, "workspace", None),
        getattr(tools, "_workspace", None),
    )
    workspace = next(
        (
            candidate
            for candidate in candidates
            if candidate is not None and callable(getattr(candidate, "snapshot", None))
        ),
        None,
    )
    root = Path(expected_root)
    try:
        root_info = root.lstat()
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError):
        root_info = None
        resolved_root = None
    if (
        workspace is None
        or Path(getattr(workspace, "root", "")) != root
        or root_info is None
        or resolved_root != root
        or stat.S_ISLNK(root_info.st_mode)
        or not stat.S_ISDIR(root_info.st_mode)
    ):
        raise CodexPipelineError(
            "native effect binding lacks the exact pipeline workspace",
            receipt=receipt,
        )
    try:
        snapshot = workspace.snapshot(label)
    except Exception as exc:
        raise CodexPipelineError(
            "native effect workspace snapshot failed closed",
            receipt=receipt,
        ) from exc
    if not isinstance(snapshot, WorkspaceSnapshot):
        raise CodexPipelineError(
            "native effect workspace snapshot differs",
            receipt=receipt,
        )
    paths: set[str] = set()
    total = 0
    for row in snapshot.files:
        pure = PurePosixPath(row.path)
        if (
            pure.is_absolute()
            or pure.as_posix() != row.path
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
            or row.path in paths
            or not isinstance(row.content, bytes)
            or row.byte_count != len(row.content)
            or row.content_blake3 != blake3_bytes(row.content)
            or not isinstance(row.mode, str)
            or len(row.mode) != 4
            or any(character not in "01234567" for character in row.mode)
        ):
            raise CodexPipelineError(
                "native effect workspace snapshot differs",
                receipt=receipt,
            )
        paths.add(row.path)
        total += row.byte_count
    if (
        snapshot.file_count != len(snapshot.files)
        or snapshot.byte_count != total
        or snapshot.tree_blake3
        != blake3_hex(
            {
                "files": snapshot.files,
                "file_count": snapshot.file_count,
                "byte_count": snapshot.byte_count,
            }
        )
    ):
        raise CodexPipelineError(
            "native effect workspace commitment differs",
            receipt=receipt,
        )
    return snapshot


def _event_item_id(receipt: CodexTurnReceipt, method: str) -> dict[str, int]:
    found: dict[str, int] = {}
    for event in receipt.events:
        if event.method != method:
            continue
        payload = canonical_value(event.payload)
        item = payload.get("item") if isinstance(payload, dict) else None
        item_id = item.get("id") if isinstance(item, dict) else None
        if isinstance(item_id, str):
            found[item_id] = event.sequence
    return found


def _tool_groups(
    receipt: CodexTurnReceipt, *, verify_event_peak: bool = False,
) -> tuple[tuple[CodexToolCall, ...], ...]:
    """Recover observable overlapping call frontiers from app-server events."""

    if not receipt.tool_calls:
        return ()
    starts = _event_item_id(receipt, "item/started")
    ends = _event_item_id(receipt, "item/completed")
    intervals: list[tuple[int, int, CodexToolCall]] = []
    for call in receipt.tool_calls:
        start = starts.get(call.upstream_item_id, call.first_event_sequence)
        end = ends.get(call.upstream_item_id)
        if end is None or end < start:
            raise CodexPipelineError("Codex tool lifecycle interval differs", receipt=receipt)
        intervals.append((start, end, call))
    intervals.sort(key=lambda row: (row[0], row[1], row[2].tool_call_id))
    groups: list[tuple[CodexToolCall, ...]] = []
    cursor = 0
    while cursor < len(intervals):
        first_start, first_end, first_call = intervals[cursor]
        del first_start
        calls = [first_call]
        cutoff = first_end
        cursor += 1
        # Calls beginning before the earliest completion were observably live
        # together and therefore form one atomic parallel decision projection.
        while cursor < len(intervals) and intervals[cursor][0] < cutoff:
            _start, end, call = intervals[cursor]
            calls.append(call)
            cutoff = min(cutoff, end)
            cursor += 1
        groups.append(tuple(calls))
    observed = max(len(group) for group in groups)
    if verify_event_peak:
        # A disjoint clique projection can have a smaller maximum width than
        # the true overlap peak when calls outlive their projected group.
        # Native-only users verify the independent event view exactly, while
        # retaining the original groups as a separate projection.
        boundaries = sorted(
            boundary for start, end, _ in intervals
            for boundary in ((start, 1), (end, -1))
        )
        active = peak = 0
        for _, change in boundaries:
            active += change
            if active < 0:
                raise CodexPipelineError("Codex tool overlap lifecycle differs", receipt=receipt)
            peak = max(peak, active)
        if active or observed > peak:
            raise CodexPipelineError("Codex tool overlap projection differs", receipt=receipt)
        observed = peak
    if observed != receipt.max_parallelism_observed:
        raise CodexPipelineError("Codex parallel call evidence differs", receipt=receipt)
    return tuple(groups)


_ACTOR_NATIVE_ACTION_STAGES = frozenset({Stage.S3, Stage.S4, Stage.E2E})
_ACTOR_NATIVE_ACTION_TYPES = frozenset({"fileChange"})
_NATIVE_PROJECTION_SCHEMA_V2 = (
    "eva.codex-provider-rollout-projection.v2-native-file-change"
)
_NATIVE_EFFECT_BINDING_SCHEMA_V1 = (
    "eva.codex-native-file-change-effect-binding.v1"
)


def _native_path(
    value: Any,
    *,
    workspace_root: Path,
    label: str,
) -> Path:
    """Validate one receipt-declared path without following a workspace escape."""

    if not isinstance(value, str) or not value or "\x00" in value:
        raise CodexPipelineError(f"{label} path differs")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise CodexPipelineError(f"{label} path is not workspace-relative")
    candidate = workspace_root.joinpath(*pure.parts)
    try:
        candidate.relative_to(workspace_root)
    except ValueError:
        raise CodexPipelineError(f"{label} path escapes the workspace") from None

    try:
        root_info = workspace_root.lstat()
        workspace_real = workspace_root.resolve(strict=True)
    except (OSError, RuntimeError):
        raise CodexPipelineError(f"{label} workspace is unavailable") from None
    if (
        workspace_real != workspace_root
        or stat.S_ISLNK(root_info.st_mode)
        or not stat.S_ISDIR(root_info.st_mode)
    ):
        raise CodexPipelineError(f"{label} workspace uses unsafe topology")

    # Existing ancestors and the final target may not be symlinks. The final
    # workspace snapshot independently commits every post-turn byte, but it
    # cannot make a receipt-declared symlink safe to dereference here.
    relative = candidate.relative_to(workspace_root)
    current = workspace_root
    for part in relative.parts[:-1]:
        current = current / part
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir():
                raise CodexPipelineError(f"{label} path uses unsafe topology")
        else:
            break
    target_exists = candidate.exists() or candidate.is_symlink()
    if target_exists:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            before = candidate.lstat()
            descriptor = os.open(candidate, flags)
        except (OSError, RuntimeError):
            raise CodexPipelineError(f"{label} path is unavailable") from None
        try:
            opened = os.fstat(descriptor)
            descriptor_path = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
            after = candidate.lstat()
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            raise CodexPipelineError(f"{label} path is unavailable") from None
        finally:
            os.close(descriptor)
        try:
            resolved.relative_to(workspace_real)
            descriptor_path.relative_to(workspace_real)
        except (OSError, RuntimeError, ValueError):
            raise CodexPipelineError(f"{label} path escapes the workspace") from None
        fingerprint = lambda info: (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        if (
            fingerprint(before) != fingerprint(opened)
            or fingerprint(opened) != fingerprint(after)
            or descriptor_path != resolved
            or stat.S_ISLNK(opened.st_mode)
            or stat.S_ISREG(opened.st_mode) and opened.st_nlink != 1
            or not (stat.S_ISREG(opened.st_mode) or stat.S_ISDIR(opened.st_mode))
        ):
            raise CodexPipelineError(f"{label} path uses unsafe topology")
    return candidate


def _validate_file_change(
    receipt: CodexTurnReceipt,
    call: CodexToolCall,
    *,
    workspace_root: Path,
) -> None:
    arguments = call.arguments
    output = call.output
    if (
        not isinstance(arguments, Mapping)
        or set(arguments) != {"changes"}
        or not isinstance(arguments["changes"], tuple)
        or not arguments["changes"]
        or not isinstance(output, Mapping)
        or set(output) != {"status"}
        or output["status"] != "completed"
    ):
        raise CodexPipelineError(
            "Codex native file-change outcome or shape differs", receipt=receipt
        )
    seen: set[Path] = set()
    for change in arguments["changes"]:
        if (
            not isinstance(change, Mapping)
            or set(change) != {"diff", "kind", "path"}
            or not isinstance(change["diff"], str)
            or not change["diff"]
            or not isinstance(change["kind"], Mapping)
        ):
            raise CodexPipelineError(
                "Codex native file-change schema differs", receipt=receipt
            )
        target = _native_path(
            change["path"], workspace_root=workspace_root, label="Codex native file-change"
        )
        if target in seen:
            raise CodexPipelineError(
                "Codex native file-change path is duplicated", receipt=receipt
            )
        seen.add(target)
        kind = change["kind"]
        kind_type = kind.get("type")
        if kind_type in {"add", "delete"}:
            if set(kind) != {"type"}:
                raise CodexPipelineError(
                    "Codex native file-change kind differs", receipt=receipt
                )
        elif kind_type == "update":
            if set(kind) not in (
                {"type"},
                {"type", "move_path"},
                {"type", "movePath"},
            ):
                raise CodexPipelineError(
                    "Codex native file-change update differs", receipt=receipt
                )
            move = kind.get("move_path", kind.get("movePath"))
            if move is not None:
                moved = _native_path(
                    move,
                    workspace_root=workspace_root,
                    label="Codex native file-change move",
                )
                if moved in seen:
                    raise CodexPipelineError(
                        "Codex native file-change path is duplicated", receipt=receipt
                    )
                seen.add(moved)
        else:
            raise CodexPipelineError(
                "Codex native file-change kind differs", receipt=receipt
            )


def _validate_actor_native_action(
    receipt: CodexTurnReceipt,
    call: CodexToolCall,
    *,
    stage: Stage,
    workspace_root: Path,
) -> None:
    if (
        stage not in _ACTOR_NATIVE_ACTION_STAGES
        or call.tool_type not in _ACTOR_NATIVE_ACTION_TYPES
        or call.name != call.tool_type
        or call.status != "completed"
        or call.lifecycle != ("item/started", "item/completed")
        or call.mcp_server is not None
        or call.mcp_tool is not None
        or call.fully_qualified_name is not None
    ):
        raise CodexPipelineError(
            "Codex native actor action is unauthorized or incomplete", receipt=receipt
        )
    try:
        _validate_file_change(receipt, call, workspace_root=workspace_root)
    except CodexPipelineError as exc:
        if exc.receipt is not None:
            raise
        raise CodexPipelineError(str(exc), receipt=receipt) from exc


def _native_file_state(row: Any | None) -> Mapping[str, Any] | None:
    if row is None:
        return None
    return {
        "content_blake3": row.content_blake3,
        "byte_count": row.byte_count,
        "mode": row.mode,
    }


def _native_relative_path_text(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CodexPipelineError("native effect path differs")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise CodexPipelineError("native effect path is not workspace-relative")
    return value


def _native_file_change_effect_binding(
    receipt: CodexTurnReceipt,
    calls: Sequence[CodexToolCall],
    *,
    before: WorkspaceSnapshot,
    after: WorkspaceSnapshot,
) -> Mapping[str, Any]:
    """Bind every declared fileChange to the exact observed tree transition."""

    before_by_path = {row.path: row for row in before.files}
    after_by_path = {row.path: row for row in after.files}
    actual_paths = tuple(
        sorted(
            path
            for path in set(before_by_path) | set(after_by_path)
            if _native_file_state(before_by_path.get(path))
            != _native_file_state(after_by_path.get(path))
        )
    )
    declared_paths: set[str] = set()
    effects: list[Mapping[str, Any]] = []

    def fail(message: str) -> None:
        raise CodexPipelineError(message, receipt=receipt)

    for call in calls:
        arguments = canonical_value(call.arguments)
        if (
            call.tool_type != "fileChange"
            or not isinstance(arguments, dict)
            or set(arguments) != {"changes"}
            or not isinstance(arguments["changes"], list)
            or not arguments["changes"]
        ):
            fail("native effect binding contains a non-fileChange action")
        for change_index, change in enumerate(arguments["changes"]):
            if not isinstance(change, dict) or not isinstance(change.get("kind"), dict):
                fail("native effect binding change shape differs")
            try:
                source_path = _native_relative_path_text(change.get("path"))
            except CodexPipelineError:
                fail("native effect path is not workspace-relative")
            kind = change["kind"].get("type")
            raw_destination = change["kind"].get(
                "move_path", change["kind"].get("movePath")
            )
            destination_path: str | None = None
            if raw_destination is not None:
                try:
                    destination_path = _native_relative_path_text(raw_destination)
                except CodexPipelineError:
                    fail("native effect move path is not workspace-relative")
            paths = (source_path,) if destination_path is None else (
                source_path,
                destination_path,
            )
            if any(path in declared_paths for path in paths):
                fail("native effect path was declared more than once")
            declared_paths.update(paths)

            source_before = _native_file_state(before_by_path.get(source_path))
            source_after = _native_file_state(after_by_path.get(source_path))
            destination_before = (
                None
                if destination_path is None
                else _native_file_state(before_by_path.get(destination_path))
            )
            destination_after = (
                None
                if destination_path is None
                else _native_file_state(after_by_path.get(destination_path))
            )
            if kind == "add":
                if (
                    destination_path is not None
                    or source_before is not None
                    or source_after is None
                ):
                    fail("native add effect differs from the workspace transition")
            elif kind == "delete":
                if (
                    destination_path is not None
                    or source_before is None
                    or source_after is not None
                ):
                    fail("native delete effect differs from the workspace transition")
            elif kind == "update" and destination_path is None:
                if (
                    source_before is None
                    or source_after is None
                    or source_before == source_after
                ):
                    fail("native update effect differs from the workspace transition")
            elif kind == "update":
                if (
                    source_before is None
                    or source_after is not None
                    or destination_before is not None
                    or destination_after is None
                ):
                    fail("native move effect differs from the workspace transition")
            else:
                fail("native effect kind differs")

            try:
                diff_blake3 = blake3_bytes(change["diff"].encode("utf-8"))
            except (AttributeError, UnicodeEncodeError):
                fail("native effect diff is not canonical UTF-8")
            path_effects = tuple(
                {
                    "path": path,
                    "before": _native_file_state(before_by_path.get(path)),
                    "after": _native_file_state(after_by_path.get(path)),
                }
                for path in paths
            )
            effect_core = {
                "tool_call_id": call.tool_call_id,
                "upstream_item_id": call.upstream_item_id,
                "change_index": change_index,
                "kind": kind,
                "source_path": source_path,
                "destination_path": destination_path,
                "diff_blake3": diff_blake3,
                "path_effects": path_effects,
            }
            effects.append(
                {**effect_core, "effect_blake3": blake3_hex(effect_core)}
            )

    declared = tuple(sorted(declared_paths))
    if declared != actual_paths:
        fail("native declared paths and actual workspace effects differ")
    binding_core = {
        "schema": _NATIVE_EFFECT_BINDING_SCHEMA_V1,
        "workspace_before_tree_blake3": before.tree_blake3,
        "workspace_after_tree_blake3": after.tree_blake3,
        "declared_changed_paths": declared,
        "actual_changed_paths": actual_paths,
        "effects": tuple(effects),
    }
    return {
        **binding_core,
        "binding_blake3": blake3_hex(binding_core),
    }


def _validate_codex_core_resource_call(
    receipt: CodexTurnReceipt,
    call: CodexToolCall,
    *,
    operation: str,
) -> None:
    """Accept only a shared, non-bypassing Codex resource-helper outcome."""

    error = codex_core_mcp_resource_call_error(
        call=call,
        operation=operation,
        offered_mcp_tool_names=receipt.offered_mcp_tool_names,
    )
    if error is not None:
        raise CodexPipelineError(error, receipt=receipt)


def _validate_tool_inventory(
    receipt: CodexTurnReceipt,
    offers: tuple[CodexToolOffer, ...],
    *,
    actor_stage: Stage | None = None,
    workspace_root: str | None = None,
    verify_event_peak: bool = False,
    schema_validation_rejection_ids: frozenset[str] = frozenset(),
) -> tuple[tuple[CodexToolCall, ...], ...]:
    expected_names = tuple(tool.fully_qualified_name for tool in offers)
    expected_schema = blake3_hex(tuple(tool.canonical_catalog_entry() for tool in offers))
    if (
        receipt.offered_mcp_tool_names != expected_names
        or receipt.offered_tool_schema_blake3 != expected_schema
    ):
        raise CodexPipelineError("Codex offered-tool schema commitment differs", receipt=receipt)
    allowed = set(expected_names)
    observed_rejection_ids: set[str] = set()
    for call in receipt.tool_calls:
        if call.tool_type == "mcpToolCall":
            if call.fully_qualified_name in allowed:
                if call.tool_call_id in schema_validation_rejection_ids:
                    observed_rejection_ids.add(call.tool_call_id)
                    if (
                        call.status not in {"completed", "failed"}
                        or not isinstance(call.arguments, Mapping)
                    ):
                        raise CodexPipelineError(
                            "Codex schema-validation rejection outcome differs",
                            receipt=receipt,
                        )
                    continue
                if call.status != "completed" or not isinstance(call.arguments, Mapping):
                    raise CodexPipelineError(
                        "Codex tool outcome or identity differs", receipt=receipt
                    )
                continue
            operation = codex_core_mcp_resource_operation(
                server=call.mcp_server,
                tool=call.mcp_tool,
                offered_mcp_tool_names=allowed,
            )
            if operation is None or not isinstance(call.arguments, Mapping):
                raise CodexPipelineError(
                    "Codex tool outcome or identity differs", receipt=receipt
                )
            _validate_codex_core_resource_call(
                receipt,
                call,
                operation=operation,
            )
            continue
        if actor_stage is None or workspace_root is None:
            raise CodexPipelineError(
                "Codex native actions are forbidden by this turn capability", receipt=receipt
            )
        _validate_actor_native_action(
            receipt,
            call,
            stage=actor_stage,
            workspace_root=Path(workspace_root),
        )
    if observed_rejection_ids != set(schema_validation_rejection_ids):
        raise CodexPipelineError(
            "Codex schema-validation rejection call inventory differs",
            receipt=receipt,
        )
    groups = _tool_groups(receipt, verify_event_peak=verify_event_peak)
    if any(
        len(group) > 1 and any(call.tool_type != "mcpToolCall" for call in group)
        for group in groups
    ):
        raise CodexPipelineError(
            "Codex native actions must be sequential", receipt=receipt
        )
    return groups


def _mcp_groups(
    groups: tuple[tuple[CodexToolCall, ...], ...],
    allowed_names: frozenset[str] | set[str],
    *,
    excluded_tool_call_ids: frozenset[str] = frozenset(),
) -> tuple[tuple[CodexToolCall, ...], ...]:
    return tuple(
        selected
        for group in groups
        if (
            selected := tuple(
                call
                for call in group
                if call.tool_type == "mcpToolCall"
                and call.fully_qualified_name in allowed_names
                and call.tool_call_id not in excluded_tool_call_ids
            )
        )
    )


def _project_events(
    ids: RuntimeIdFactory,
    *,
    system: str,
    user: Any,
    receipt: CodexTurnReceipt,
    groups: tuple[tuple[CodexToolCall, ...], ...],
    retain_full_receipt: bool,
    actor_results: Mapping[str, ToolResult] | None = None,
    schema_validation_rejections: (
        Mapping[str, Mapping[str, JsonValue]] | None
    ) = None,
    terminal_response: str | None = None,
) -> tuple[TrajectoryEvent, ...]:
    events = [_event(ids, role="system", content=system), _event(ids, role="user", content=user)]
    for group in groups:
        visible_ids = tuple(
            (
                actor_results[call.tool_call_id].call_id
                if actor_results is not None
                and call.tool_call_id in actor_results
                else call.tool_call_id
            )
            for call in group
        )
        decisions = []
        for call, visible_id in zip(group, visible_ids, strict=True):
            if actor_results is not None and call.tool_call_id in actor_results:
                decisions.append(
                    {
                        "call_id": visible_id,
                        "codex_tool_call_id": call.tool_call_id,
                        "upstream_item_id": call.upstream_item_id,
                        "fully_qualified_name": call.fully_qualified_name,
                        "arguments": call.arguments,
                        "receipt_blake3": call.receipt_blake3,
                    }
                )
            elif (
                schema_validation_rejections is not None
                and call.tool_call_id in schema_validation_rejections
            ):
                decisions.append(
                    {
                        "call_id": visible_id,
                        "upstream_item_id": call.upstream_item_id,
                        "schema_validation_rejection": {
                            "fully_qualified_name": call.fully_qualified_name,
                            "arguments": call.arguments,
                            "receipt_blake3": call.receipt_blake3,
                        },
                    }
                )
            elif call.tool_type == "mcpToolCall":
                decisions.append(
                    {
                        "call_id": visible_id,
                        "upstream_item_id": call.upstream_item_id,
                        "codex_core_mcp": {
                            "fully_qualified_name": call.fully_qualified_name,
                            "arguments": call.arguments,
                            "receipt_blake3": call.receipt_blake3,
                        },
                    }
                )
            else:
                decisions.append(
                    {
                        "call_id": visible_id,
                        "upstream_item_id": call.upstream_item_id,
                        "tool_type": call.tool_type,
                        "name": call.name,
                        "arguments": call.arguments,
                        "receipt_blake3": call.receipt_blake3,
                    }
                )
        events.append(
            _event(
                ids,
                role="assistant",
                content={
                    "codex_tool_calls": decisions
                },
                tool_call_ids=visible_ids,
            )
        )
        for call, visible_id in zip(group, visible_ids, strict=True):
            result = (
                actor_results[call.tool_call_id]
                if actor_results is not None and call.tool_call_id in actor_results
                else None
            )
            if result is not None:
                content = {
                    "status": result.status,
                    "output": result.output,
                    "error_code": result.error_code,
                    "receipt_blake3": result.receipt_blake3,
                }
                from eva_agent.pipeline.execution_diagnostics import diagnostic_content
                supplemental = diagnostic_content(call.output, call_id=result.call_id)
                if supplemental:
                    # Outer policy-event metadata preserves exactly what the
                    # actor saw; Judge messages.json already materializes it.
                    content["supplemental_content"] = supplemental
            elif (
                schema_validation_rejections is not None
                and call.tool_call_id in schema_validation_rejections
            ):
                content = {
                    "schema_validation_rejection": schema_validation_rejections[
                        call.tool_call_id
                    ],
                    "codex_tool_call_status": call.status,
                    "codex_tool_call_receipt_blake3": call.receipt_blake3,
                }
            elif call.tool_type == "mcpToolCall":
                content = {
                    "codex_core_mcp": {
                        "fully_qualified_name": call.fully_qualified_name,
                        "status": call.status,
                        "output": call.output,
                        "receipt_blake3": call.receipt_blake3,
                    }
                }
            else:
                content = {
                    "codex_native_action": {
                        "tool_type": call.tool_type,
                        "name": call.name,
                        "status": call.status,
                        "output": call.output,
                        "receipt_blake3": call.receipt_blake3,
                    }
                }
            events.append(
                _event(
                    ids,
                    role="tool",
                    content=content,
                    tool_call_ids=(visible_id,),
                )
            )
    terminal: dict[str, Any] = {
        "response": (
            receipt.final_response
            if terminal_response is None
            else terminal_response
        ),
        "codex_turn_receipt_blake3": receipt.receipt_blake3,
    }
    if retain_full_receipt:
        terminal["codex_turn_receipt"] = canonical_value(receipt)
    events.append(_event(ids, role="assistant", content=terminal))
    return tuple(events)


_ACTOR_ROLES = {
    Cohort.WEAK: CodexRole.WEAK_ACTOR,
    Cohort.MIDDLE: CodexRole.MIDDLE_ACTOR,
    Cohort.STRONG: CodexRole.STRONG_ACTOR,
}


class CodexRolloutAdapter:
    """Run exactly one fresh Codex actor trajectory for each cohort/candidate."""

    def __init__(
        self,
        runtime_factory: ActorRuntimeFactory | None = None,
        options_factory: ActorOptionsFactory | None = None,
        *,
        runner: CodexTurnRunnerPort | None = None,
        id_factory: RuntimeIdFactory | None = None,
        mcp_server: str = "evamed",
        turn_mcp_factory: TurnMCPBridgeFactoryPort | None = None,
        skills_factory: ActorSkillsFactory | None = None,
        tool_runtime_guard: (
            Callable[[RolloutRequest, ToolRuntimePort], ToolRuntimePort] | None
        ) = None,
        enable_native_actions: bool = False,
        controlled_budget_terminal: Callable[[], Mapping[str, Any] | None] | None = None,
        allow_empty_final_response: bool = False,
        allow_schema_validation_rejections: bool = False,
        allow_legacy_empty_code_rejections: bool = False,
        system_instruction: str = DEFAULT_ACTOR_SYSTEM_INSTRUCTION,
    ) -> None:
        if (
            not callable(options_factory)
            or not system_instruction.strip()
            or (runtime_factory is None and runner is None)
            or (runtime_factory is not None and not callable(runtime_factory))
            or (skills_factory is not None and not callable(skills_factory))
            or type(enable_native_actions) is not bool
            or type(allow_empty_final_response) is not bool
            or type(allow_schema_validation_rejections) is not bool
            or type(allow_legacy_empty_code_rejections) is not bool
            or (allow_legacy_empty_code_rejections and enable_native_actions)
            or (allow_empty_final_response and enable_native_actions)
            or (controlled_budget_terminal is not None and (
                not callable(controlled_budget_terminal) or enable_native_actions
            ))
            or (
                tool_runtime_guard is not None
                and not callable(tool_runtime_guard)
            )
            or (
                allow_schema_validation_rejections
                and turn_mcp_factory is None
            )
        ):
            raise CodexPipelineError("actor options factory and instruction are required")
        self._runtime_factory = runtime_factory
        self._runner = _require_runner(runner) if runner is not None else None
        self._options = options_factory
        self._ids = id_factory or RandomUUIDFactory()
        self._server = mcp_server
        if turn_mcp_factory is not None and not callable(
            getattr(turn_mcp_factory, "open_actor", None)
        ):
            raise CodexPipelineError("actor turn MCP factory is unavailable")
        self._turn_mcp = turn_mcp_factory
        self._skills = skills_factory
        self._tool_runtime_guard = tool_runtime_guard
        self._native_actions = enable_native_actions
        self._controlled_budget_terminal = controlled_budget_terminal
        self._allow_empty_final_response = allow_empty_final_response
        self._allow_legacy_empty_code_rejections = allow_legacy_empty_code_rejections
        self._allow_schema_validation_rejections = (
            allow_schema_validation_rejections
        )
        self._system = (
            system_instruction + _NATIVE_ACTOR_INSTRUCTION
            if enable_native_actions
            else system_instruction
        )
        self._consumed: set[tuple[str, Cohort]] = set()
        self._claim_lock = Lock()

    @property
    def native_actions_enabled(self) -> bool:
        """Expose the fail-closed recipe capability without changing receipts."""

        return self._native_actions

    def run(self, request: RolloutRequest, tools: ToolRuntimePort) -> ProviderRollout:
        claim = (request.sandbox.episode_id, request.model.cohort)
        with self._claim_lock:
            if claim in self._consumed:
                raise CodexPipelineError("actor cohort/candidate trajectory was already consumed")
            # Claims remain consumed after infrastructure/provider failure: no
            # semantic retry and no replacement trajectory in this process.
            self._consumed.add(claim)

        if (
            self._native_actions
            and request.sandbox.stage not in _ACTOR_NATIVE_ACTION_STAGES
        ):
            raise CodexPipelineError(
                "Codex native fileChange is unauthorized for this stage"
            )

        if self._tool_runtime_guard is not None:
            tools = self._tool_runtime_guard(request, tools)
        initial_trace = tools.trace()
        if initial_trace.results or initial_trace.declared_call_ids or initial_trace.retry_count:
            raise CodexPipelineError("actor tool runtime was not fresh")
        offers = _canonical_actor_tools(request, server=self._server)
        cwd = _workspace_root(tools)
        empty_code_bootstrap = {}
        if self._allow_legacy_empty_code_rejections and legacy_execute_code_runtime(tools):
            for path in BOOTSTRAP_PATHS:
                source = Path(cwd) / path
                if source.is_file() and not source.is_symlink() and source.stat().st_size <= 4 * 1024 * 1024:
                    empty_code_bootstrap[path] = source.read_bytes()
        bridge = CodexToolExecutionBridge(tools, id_factory=self._ids)
        schema_validation_rejections: Mapping[
            str, Mapping[str, JsonValue]
        ] = {}
        options = self._options(request, cwd, offers, bridge)
        expected_role = _ACTOR_ROLES[request.model.cohort]
        expected_sandbox = (
            CodexSandbox.WORKSPACE_WRITE
            if self._native_actions
            else CodexSandbox.READ_ONLY
        )
        if (
            options.role is not expected_role
            or options.model != request.model.model_id
            or options.provider != request.model.provider
            or options.cwd != cwd
            or options.sandbox is not expected_sandbox
            or options.offered_tools != offers
            or not options.ephemeral
        ):
            raise CodexPipelineError("actor Codex thread options differ from rollout request")
        selected_skills = (
            () if self._skills is None else tuple(self._skills(request))
        )
        if (
            any(not isinstance(skill, CodexSkill) for skill in selected_skills)
            or len({skill.skill_id for skill in selected_skills}) != len(selected_skills)
            or len({skill.name for skill in selected_skills}) != len(selected_skills)
        ):
            raise CodexPipelineError("actor selected-skill catalog differs")
        _verify_actor_skills(selected_skills, workspace_root=cwd)
        turn_input = CodexTurnInput(
            public_text=request.sandbox.instruction,
            public_context=request.policy_visible_context,
            skills=selected_skills,
            sandbox=expected_sandbox,
            model=request.model.model_id,
        )
        runner = self._runner
        if runner is None:
            runtime_factory = self._runtime_factory
            if runtime_factory is None:  # guarded in __init__, retained for typing
                raise CodexPipelineError("actor Codex runtime factory is unavailable")
            runner = SyncCodexTurnRunner(lambda: runtime_factory(bridge))
        native_before = (
            _bound_workspace_snapshot(
                tools,
                expected_root=cwd,
                label="codex-native-turn-before",
            )
            if self._native_actions
            else None
        )
        if self._turn_mcp is None:
            receipt = _bounded_run_once(
                runner, options, turn_input, policy_context=request.policy_visible_context
            )
            native_after = (
                _bound_workspace_snapshot(
                    tools,
                    expected_root=cwd,
                    label="codex-native-turn-after",
                    receipt=receipt,
                )
                if self._native_actions
                else None
            )
            _verify_actor_skills(
                selected_skills, workspace_root=cwd, receipt=receipt
            )
            _verify_receipt_identity(
                receipt, options=options, expected_role=expected_role
            )
            groups = _validate_tool_inventory(
                receipt,
                offers,
                actor_stage=request.sandbox.stage if self._native_actions else None,
                workspace_root=cwd if self._native_actions else None,
            )
            mcp_groups = _mcp_groups(groups, set(receipt.offered_mcp_tool_names))
            actor_results, final_trace = bridge.bind_receipt(receipt, mcp_groups)
        else:
            # The broker and its nonce exist for precisely this provider turn.
            # Binding is completed before teardown so no orphan process can
            # manufacture a post-turn observation.
            actor_session = (
                self._turn_mcp.open_actor(
                    options,
                    bridge,
                    allow_schema_validation_rejections=True,
                )
                if self._allow_schema_validation_rejections
                else self._turn_mcp.open_actor(options, bridge)
            )
            with actor_session as bound_options:
                if (
                    bound_options.role is not options.role
                    or bound_options.model != options.model
                    or bound_options.provider != options.provider
                    or bound_options.cwd != options.cwd
                    or bound_options.sandbox is not options.sandbox
                    or bound_options.offered_tools != offers
                    or not bound_options.ephemeral
                ):
                    raise CodexPipelineError("actor MCP-bound thread options differ")
                receipt = _bounded_run_once(
                    runner,
                    bound_options,
                    turn_input,
                    policy_context=request.policy_visible_context,
                )
                native_after = (
                    _bound_workspace_snapshot(
                        tools,
                        expected_root=cwd,
                        label="codex-native-turn-after",
                        receipt=receipt,
                    )
                    if self._native_actions
                    else None
                )
                _verify_actor_skills(
                    selected_skills, workspace_root=cwd, receipt=receipt
                )
                _verify_receipt_identity(
                    receipt, options=bound_options, expected_role=expected_role
                )
                if self._allow_schema_validation_rejections:
                    schema_validation_rejections = (
                        bridge.bind_schema_validation_rejections(receipt, offers)
                    )
                groups = _validate_tool_inventory(
                    receipt,
                    offers,
                    actor_stage=request.sandbox.stage if self._native_actions else None,
                    workspace_root=cwd if self._native_actions else None,
                    schema_validation_rejection_ids=frozenset(
                        schema_validation_rejections
                    ),
                )
                mcp_groups = _mcp_groups(
                    groups,
                    set(receipt.offered_mcp_tool_names),
                    excluded_tool_call_ids=frozenset(
                        schema_validation_rejections
                    ),
                )
                actor_results, final_trace = bridge.bind_receipt(receipt, mcp_groups)
        empty_code_ids = empty_code_rejection_ids(
            receipt=receipt, trace=final_trace,
            mapping={key: result.call_id for key, result in actor_results.items()},
            catalog=tuple(offer.canonical_catalog_entry() for offer in offers),
            context=request.policy_visible_context, bootstrap=empty_code_bootstrap,
        ) if empty_code_bootstrap else ()
        semantic_tool_gate_failure = (
            receipt.status == "failed"
            and bool(mcp_groups)
            and all(
                call.tool_type == "mcpToolCall"
                for group in groups
                for call in group
            )
            and bool(final_trace.results)
            and any(
                semantic_tool_failure(result, empty_code_ids)
                for result in final_trace.results
            )
            and all(
                result.status == "completed"
                or (
                    semantic_tool_failure(result, empty_code_ids)
                )
                for result in final_trace.results
            )
            and final_trace.retry_count == 0
        )
        budget_terminal = None
        if receipt.status == "failed" and self._controlled_budget_terminal is not None:
            raw_budget = self._controlled_budget_terminal()
            if raw_budget is not None:
                from .budget import budget_metadata
                try:
                    budget_terminal = budget_metadata(raw_budget)
                except (ValueError, TypeError) as exc:
                    raise CodexPipelineError("controlled rollout budget evidence differs", receipt=receipt) from exc
                if final_trace.retry_count or any(
                    result.status != "completed" and not (
                        semantic_tool_failure(result, empty_code_ids)
                    ) for result in final_trace.results
                ):
                    raise CodexPipelineError("rollout budget cannot mask a tool infrastructure failure", receipt=receipt)
                semantic_tool_gate_failure = False
        workspace_terminal = (self._allow_empty_final_response and receipt.status == "completed"
            and (receipt.final_response is None or not receipt.final_response.strip()))
        if workspace_terminal and any(
            result.status != "completed" and not (
                semantic_tool_failure(result, empty_code_ids)
            ) for result in final_trace.results
        ):
            raise CodexPipelineError("workspace terminal cannot mask a tool infrastructure failure", receipt=receipt)
        if receipt.status == "completed" and not workspace_terminal:
            _require_terminal(
                receipt,
                options=bound_options if self._turn_mcp is not None else options,
                expected_role=expected_role,
            )
        elif (
            receipt.status != "completed"
            and not semantic_tool_gate_failure
            and budget_terminal is None
        ):
            raise CodexPipelineError(
                f"Codex turn status is not completed: {receipt.status}",
                receipt=receipt,
            )
        native_calls = tuple(
            call
            for group in groups
            for call in group
            if call.tool_type != "mcpToolCall"
        )
        native_effect_binding = None
        if self._native_actions:
            if native_before is None or native_after is None:
                raise CodexPipelineError(
                    "native effect snapshots are absent", receipt=receipt
                )
            native_effect_binding = _native_file_change_effect_binding(
                receipt,
                native_calls,
                before=native_before,
                after=native_after,
            )
        if not mcp_groups and final_trace != initial_trace:
            raise CodexPipelineError("tool-free Codex turn changed ToolTrace", receipt=receipt)
        assistant_output = (
            receipt.final_response
            if isinstance(receipt.final_response, str)
            and receipt.final_response.strip()
            else "[trajectory terminated at a signed semantic tool gate]"
        )
        if budget_terminal is not None:
            from .budget import TERMINAL_MARKER
            assistant_output = TERMINAL_MARKER
        elif workspace_terminal:
            from .budget import WORKSPACE_TERMINAL_MARKER
            assistant_output = WORKSPACE_TERMINAL_MARKER
        events = _project_events(
            self._ids,
            system=self._system,
            user=request.policy_visible_context,
            receipt=receipt,
            groups=groups,
            actor_results=actor_results,
            schema_validation_rejections=schema_validation_rejections,
            retain_full_receipt=False,
            terminal_response=assistant_output,
        )
        metadata = {
            "schema": (
                _NATIVE_PROJECTION_SCHEMA_V2
                if self._native_actions
                else "eva.codex-provider-rollout-projection.v1"
            ),
            "codex_turn_receipt": canonical_value(receipt),
            "tool_call_groups": tuple(
                tuple(actor_results[call.tool_call_id].call_id for call in group)
                for group in mcp_groups
            ),
            "codex_to_pipeline_call_ids": {
                call.tool_call_id: actor_results[call.tool_call_id].call_id
                for group in mcp_groups for call in group
            },
            "raw_input_recorded": False,
            "semantic_retry_count": 0,
        }
        if empty_code_ids:
            metadata["legacy_empty_code_rejections"] = {
                "call_ids": empty_code_ids,
                "offered_catalog": tuple(offer.canonical_catalog_entry() for offer in offers),
            }
        if schema_validation_rejections:
            metadata["schema_validation_rejection_catalog"] = tuple(
                offer.canonical_catalog_entry() for offer in offers
            )
            metadata["schema_validation_rejections"] = tuple(
                {
                    "codex_tool_call_id": call.tool_call_id,
                    "codex_tool_call_receipt_blake3": call.receipt_blake3,
                    "rejection": schema_validation_rejections[call.tool_call_id],
                }
                for group in groups
                for call in group
                if call.tool_call_id in schema_validation_rejections
            )
        if budget_terminal is not None:
            from .budget import PROJECTION
            metadata["schema"] = PROJECTION
            metadata["controlled_budget_termination"] = budget_terminal
        elif workspace_terminal:
            from .budget import WORKSPACE_PROJECTION, WORKSPACE_TERMINAL
            metadata["schema"] = WORKSPACE_PROJECTION
            metadata["workspace_terminal"] = dict(WORKSPACE_TERMINAL)
        if semantic_tool_gate_failure:
            failed_results = tuple(
                result for result in final_trace.results
                if result.status == "immutable_failure"
            )
            metadata["semantic_tool_gate_failure"] = {
                "classification": "signed_tool_gate_failure",
                "codex_turn_status": receipt.status,
                "failed_call_ids": tuple(result.call_id for result in failed_results),
                "error_codes": tuple(result.error_code for result in failed_results),
                "tool_trace_blake3": final_trace.trace_blake3,
                "synthetic_terminal_marker": not (
                    isinstance(receipt.final_response, str)
                    and receipt.final_response.strip()
                ),
            }
        if native_effect_binding is not None:
            metadata["native_file_change_effect_binding"] = native_effect_binding
        core = {
            "rollout_id": request.rollout_id,
            "model_id": request.model.model_id,
            "sandbox_manifest_blake3": request.sandbox.manifest_blake3,
            "codex_turn_receipt_blake3": receipt.receipt_blake3,
            "policy_events": events,
            "tool_trace_blake3": final_trace.trace_blake3,
            "assistant_output": assistant_output,
        }
        return ProviderRollout(
            assistant_output=assistant_output,
            provider_receipt_blake3=blake3_hex(core),
            policy_events=events,
            safe_metadata=metadata,
        )


def _judge_offers(tools: JudgeWorkspaceTools, server: str) -> tuple[CodexToolOffer, ...]:
    offers: list[CodexToolOffer] = []
    for raw in tools.response_api_schemas():
        value = canonical_value(raw)
        if not isinstance(value, dict) or set(value) != {
            "type",
            "name",
            "description",
            "parameters",
            "strict",
        } or value["type"] != "function" or value["strict"] is not True:
            raise CodexPipelineError("judge workspace schema differs")
        offers.append(
            CodexToolOffer(
                fully_qualified_name=f"{server}/{value['name']}",
                description=value["description"],
                input_schema=value["parameters"],
                visibility="judge-only",
                parallel_safe=True,
                read_only=True,
            )
        )
    if tuple(offer.name for offer in offers) != TOOL_NAMES:
        raise CodexPipelineError("judge workspace tool inventory differs")
    return tuple(offers)


def _judge_output_schema(items: Sequence[Mapping[str, Any]]) -> Mapping[str, JsonValue]:
    return freeze_json(
        {
            "type": "object",
            "properties": {
                "item_scores": {
                    "type": "array",
                    "minItems": len(items),
                    "maxItems": len(items),
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string"},
                            "score": {"type": "number", "minimum": 0, "maximum": 1},
                            "evidence_refs": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string", "minLength": 1},
                            },
                            "rationale": {"type": "string", "minLength": 1},
                        },
                        "required": ["item_id", "score", "evidence_refs", "rationale"],
                        "additionalProperties": False,
                    },
                },
                "hard_gates_passed": {"type": "boolean"},
                "summary": {"type": "string"},
            },
            "required": ["item_scores", "hard_gates_passed", "summary"],
            "additionalProperties": False,
        }
    )  # type: ignore[return-value]


def _judge_observation(result: Any) -> Mapping[str, Any]:
    return {
        "status": result.status,
        "output": result.output,
        "error_code": result.error_code,
        "inspected_evidence_refs": result.inspected_evidence_refs,
        "content_inspection": result.content_inspection,
        "evidence_bundle_blake3": result.evidence_bundle_blake3,
    }


from eva_agent.pipeline.judge_material_view import (
    MATERIAL_PREFIX as CODEX_JUDGE_MATERIAL_PREFIX,
    REQUIRED_PATHS as CODEX_JUDGE_REQUIRED_MATERIAL_PATHS,
    HISTORICAL_VIEW, material_layout,
)


def codex_judge_evidence_index(evidence: EvidenceBundle) -> Mapping[str, Any]:
    """Return the bounded index sent in the initial Codex judge turn.

    Byte-bearing actor context, messages, tool trace, assistant output, and
    before/after workspace files remain behind the read-only TurnMCP tools.
    """

    manifest = evidence.sandbox_manifest
    materials = tuple(
        row
        for row in evidence.workspace_after.files
        if row.path.startswith(CODEX_JUDGE_MATERIAL_PREFIX)
    )
    try:
        material_view, required_paths, audit_paths = material_layout(evidence)
    except ValueError as error:
        raise CodexPipelineError("Codex judge evidence materials differ") from error

    def snapshot_index(snapshot: WorkspaceSnapshot) -> Mapping[str, Any]:
        return {
            "label": snapshot.label,
            "file_count": snapshot.file_count,
            "byte_count": snapshot.byte_count,
            "tree_blake3": snapshot.tree_blake3,
        }

    return {
        "schema": "eva.codex-agent-judge-evidence-index.v1",
        "bundle_id": evidence.bundle_id,
        "bundle_blake3": evidence.bundle_blake3,
        "rollout_id": evidence.rollout_id,
        "sandbox": {
            "sandbox_id": manifest.sandbox_id,
            "episode_id": manifest.episode_id,
            "domain": manifest.domain,
            "stage": manifest.stage,
            "manifest_blake3": manifest.manifest_blake3,
            "rubric_digest": manifest.rubric.digest,
        },
        "model": evidence.model,
        "context_blake3": evidence.context_blake3,
        "workspace_before": snapshot_index(evidence.workspace_before),
        "workspace_after": snapshot_index(evidence.workspace_after),
        "actor_evidence": {
            "snapshot": "after",
            "required_complete_reads": required_paths,
            **({"material_view": material_view, "available_targeted_audit_reads": audit_paths,
                "audit_files_are_not_actor_source_workspace": True,
                "audit_reads_do_not_replace_source_inspection_or_item_citations": True}
               if material_view != HISTORICAL_VIEW else {}),
            "file_count": len(materials),
            "byte_count": sum(row.byte_count for row in materials),
            "files": tuple(
                {
                    "path": row.path,
                    "byte_count": row.byte_count,
                    "mode": row.mode,
                    "content_blake3": row.content_blake3,
                }
                for row in materials
            ),
        },
        "provider_receipt_blake3": evidence.provider_receipt_blake3,
    }


def _mcp_result_object(call: CodexToolCall) -> Mapping[str, Any]:
    output = canonical_value(call.output)
    if not isinstance(output, dict) or "result" not in output:
        raise CodexPipelineError("MCP result envelope differs")
    if output.get("error") is not None:
        raise CodexPipelineError("MCP result envelope differs")
    result = output["result"]
    if not isinstance(result, dict):
        raise CodexPipelineError("MCP result is not an object")
    if "structuredContent" in result:
        # Codex app-server normalizes a successful MCP CallToolResult into its
        # McpToolCallResult protocol type.  That type deliberately omits the
        # MCP ``isError: false`` default while retaining ``structuredContent``
        # verbatim; older app-server projections retained the explicit flag.
        # Accept both success encodings, but continue to reject true, null, or
        # otherwise malformed flags.  The caller still binds the structured
        # receipt exactly to the in-turn execution result and its BLAKE3.
        is_error = result.get("isError", False)
        if is_error is not False or not isinstance(result["structuredContent"], dict):
            raise CodexPipelineError("MCP structured result differs")
        return result["structuredContent"]
    return result


def _mcp_schema_validation_rejection_object(
    call: CodexToolCall,
) -> Mapping[str, Any] | None:
    """Open only the explicit MCP error-result shape issued by TurnMCP."""

    output = canonical_value(call.output)
    if not isinstance(output, dict):
        return None
    result = output.get("result")
    if output.get("error") is not None or not isinstance(result, dict):
        return None
    is_error = result.get("isError")
    # Codex app-server's McpToolCallResult deliberately omits the MCP
    # ``isError`` member but maps true to the failed item status.  Test
    # transports may retain the original flag.  The bridge-issued structured
    # receipt below remains the authority in both representations.
    if is_error is not True and not (
        "isError" not in result and call.status == "failed"
    ):
        return None
    structured = result.get("structuredContent")
    if not isinstance(structured, dict):
        raise CodexPipelineError("MCP schema-validation rejection result differs")
    return structured


def _mcp_structured_output(call: CodexToolCall) -> Mapping[str, Any]:
    return _mcp_result_object(call)


def _live_judge_trace_matches(
    trace: JudgeAgentTrace,
    domain_groups: tuple[tuple[CodexToolCall, ...], ...],
    live_results: Mapping[str, JudgeToolResult],
) -> bool:
    """Bind an in-turn trace without assuming parallel arrival order.

    TurnMCP may receive calls from one provider-declared parallel frontier in
    a different order than the provider serialized them. ``bind_receipt`` has
    already proven every exact identity, argument, observation, and frontier;
    this final check therefore compares the unique call set while retaining
    all cardinality and execution-frontier commitments.
    """

    expected_ids = tuple(
        live_results[call.tool_call_id].call_id
        for group in domain_groups
        for call in group
    )
    declared_ids = trace.declared_call_ids
    return (
        trace.frontier_count == len(domain_groups)
        and trace.max_parallelism_observed
        == max(len(group) for group in domain_groups)
        and len(declared_ids) == len(expected_ids)
        and len(set(declared_ids)) == len(declared_ids)
        and len(set(expected_ids)) == len(expected_ids)
        and set(declared_ids) == set(expected_ids)
        and trace.joined_call_ids
        == tuple(result.call_id for result in trace.results)
        and trace.retry_count == 0
    )


def _agent_judge_terminal_object(value: str) -> Mapping[str, Any]:
    """Parse an exact JSON verdict, allowing only one Markdown JSON fence.

    Some OpenAI-compatible Chat projections return schema-conforming JSON in
    a single fenced block even when Codex requested structured output.  The
    fence carries no semantics; arbitrary prefixes, suffixes, or prose remain
    rejected before the exact verdict schema and rubric levels are validated.
    """

    text = value.strip()
    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError):
        lines = text.splitlines()
        if (
            len(lines) < 3
            or lines[0].casefold() not in {"```json", "```"}
            or lines[-1] != "```"
        ):
            raise CodexPipelineError("judge terminal response is not JSON") from None
        try:
            parsed = json.loads("\n".join(lines[1:-1]))
        except (ValueError, RecursionError):
            raise CodexPipelineError("judge terminal response is not JSON") from None
    if not isinstance(parsed, Mapping):
        raise CodexPipelineError("judge terminal response is not JSON")
    return parsed


def _validated_judge_verdict(value, items, trace, *, receipt=None):
    """One strict verdict validator shared by final acceptance and opt-in eligibility."""
    if not isinstance(value, dict) or set(value) != {
        "item_scores", "hard_gates_passed", "summary"
    } or not isinstance(value["item_scores"], list):
        raise CodexPipelineError("judge terminal response shape differs", receipt=receipt)
    scores: list[RubricItemScore] = []
    hard_gates_passed = True
    cited_inspected: set[str] = set()
    rows = value["item_scores"]
    if len(rows) != len(items):
        raise CodexPipelineError("judge score coverage differs", receipt=receipt)
    for row, item in zip(rows, items, strict=True):
        if not isinstance(row, dict) or set(row) != {
            "item_id", "score", "evidence_refs", "rationale"
        } or row["item_id"] != item["item_id"]:
            raise CodexPipelineError("judge score row identity differs", receipt=receipt)
        refs = row["evidence_refs"]
        if (
            not isinstance(refs, list)
            or not refs
            or any(not isinstance(ref, str) or not ref for ref in refs)
            or len(refs) != len(set(refs))
        ):
            raise CodexPipelineError("judge evidence references differ", receipt=receipt)
        for ref in refs:
            if ref.startswith("workspace:"):
                if ref not in trace.inspected_evidence_refs:
                    raise CodexPipelineError(
                        "judge cites workspace evidence it did not inspect", receipt=receipt
                    )
                cited_inspected.add(ref)
        raw_score = row["score"]
        if type(raw_score) not in {int, float} or not math.isfinite(float(raw_score)):
            raise CodexPipelineError("judge score is not finite", receipt=receipt)
        score = float(raw_score)
        score_bps = round(score * 10_000)
        allowed = {int(level["score_bps"]) for level in item["partial_credit"]["levels"]}
        if score_bps not in allowed:
            raise CodexPipelineError("judge score is not a compiled rubric level", receipt=receipt)
        if not isinstance(row["rationale"], str) or not row["rationale"].strip():
            raise CodexPipelineError("judge rationale differs", receipt=receipt)
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
    if not cited_inspected:
        raise CodexPipelineError("judge scores cite no inspected workspace reference", receipt=receipt)
    if (
        type(value["hard_gates_passed"]) is not bool
        or value["hard_gates_passed"] is not hard_gates_passed
        or not isinstance(value["summary"], str)
        or not value["summary"].strip()
    ):
        raise CodexPipelineError("judge hard-gate or summary claim differs", receipt=receipt)
    return tuple(scores), hard_gates_passed


class CodexWorkspaceAgentJudge:
    """Model-explicit Codex judge over replay-verified immutable workspace reads."""

    def __init__(
        self,
        runtime_factory: RuntimeFactory | None = None,
        options_factory: JudgeOptionsFactory | None = None,
        *,
        model_id: str,
        runner: CodexTurnRunnerPort | None = None,
        id_factory: RuntimeIdFactory | None = None,
        mcp_server: str = "evamed-judge",
        maximum_parallel_tools: int = MAXIMUM_PARALLEL_TOOL_CALLS,
        turn_mcp_factory: TurnMCPBridgeFactoryPort | None = None,
        compact_evidence_index: bool = False,
        fast_workspace_judge: bool = False,
        maximum_workspace_tool_calls: int = 64,
        maximum_workspace_tool_frontiers: int = 4,
        turn_timeout_seconds: int = 240,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise CodexPipelineError("Codex workspace judge model identity is required")
        if (
            not callable(options_factory)
            or not 1 <= maximum_parallel_tools <= MAXIMUM_PARALLEL_TOOL_CALLS
            or type(compact_evidence_index) is not bool
            or type(fast_workspace_judge) is not bool
            or not 1 <= maximum_workspace_tool_calls <= 256
            or not 1 <= maximum_workspace_tool_frontiers <= 32
            or not 1 <= turn_timeout_seconds <= 900
            or (runtime_factory is None and runner is None)
            or (runtime_factory is not None and not callable(runtime_factory))
        ):
            raise CodexPipelineError("judge options or parallel-tool bound differs")
        self.model_id = model_id
        if runner is not None:
            self._runner = _require_runner(runner)
        else:
            if runtime_factory is None:  # guarded above, retained for typing
                raise CodexPipelineError("judge Codex runtime factory is unavailable")
            self._runner = SyncCodexTurnRunner(runtime_factory)
        self._options = options_factory
        self._ids = id_factory or RandomUUIDFactory()
        self._server = mcp_server
        self._width = maximum_parallel_tools
        if turn_mcp_factory is not None and not callable(
            getattr(turn_mcp_factory, "open_judge", None)
        ):
            raise CodexPipelineError("judge turn MCP factory is unavailable")
        self._turn_mcp = turn_mcp_factory
        self._compact_evidence_index = compact_evidence_index
        self._fast_workspace_judge = fast_workspace_judge
        self._maximum_workspace_tool_calls = maximum_workspace_tool_calls
        self._maximum_workspace_tool_frontiers = maximum_workspace_tool_frontiers
        self._turn_timeout_seconds = turn_timeout_seconds
        self._consumed: set[str] = set()
        self._receipts: dict[str, CodexTurnReceipt] = {}
        self._claim_lock = Lock()

    def receipt_for(self, judgment_id: str) -> CodexTurnReceipt:
        """Return retained Codex evidence without altering JudgeAssessment."""

        with self._claim_lock:
            try:
                return self._receipts[judgment_id]
            except KeyError:
                raise CodexPipelineError("judge Codex receipt is unavailable") from None

    def _live_judge_groups(self, receipt, groups, bridge):
        """Production default retains strict one-transport/one-execution groups."""
        return groups

    def _judge_tool_inventory(self, receipt, offers):
        """Default keeps the original strict transport projection checks."""
        return _validate_tool_inventory(receipt, offers)

    def _bind_verdict_format_correction(self, request, rubric, workspace_tools, judge_bridge):
        """Historical/default judges have no final-format correction capability."""
        return None

    def judge(self, request: JudgeRequest, rubric: CompiledRubricTable) -> JudgeAssessment:
        with self._claim_lock:
            if request.judgment_id in self._consumed:
                raise CodexPipelineError("judge trajectory was already consumed")
            self._consumed.add(request.judgment_id)
        if request.judge_model_id != self.model_id:
            raise CodexPipelineError("judge request model identity differs")
        manifest = request.workspace_evidence.sandbox_manifest
        rubric_stage = rubric.stage.value if hasattr(rubric.stage, "value") else rubric.stage
        if (
            rubric.digest != manifest.rubric.digest
            or rubric.domain != manifest.rubric.domain
            or rubric_stage != manifest.rubric.stage.value
        ):
            raise CodexPipelineError("judge rubric does not match the sandbox binding")
        items = _rubric_items(rubric)
        workspace_tools = JudgeWorkspaceTools(
            request.workspace_evidence,
            id_factory=self._ids,
            maximum_parallel_tools=self._width,
        )
        judge_bridge = CodexJudgeToolExecutionBridge(
            workspace_tools,
            id_factory=self._ids,
            maximum_total_calls=(
                self._maximum_workspace_tool_calls
                if self._fast_workspace_judge
                else None
            ),
            maximum_execution_groups=(
                self._maximum_workspace_tool_frontiers
                if self._fast_workspace_judge
                else None
            ),
        )
        offers = _judge_offers(workspace_tools, self._server)
        options = self._options(request, offers)
        if (
            options.role is not CodexRole.JUDGE
            or options.model != self.model_id
            or options.sandbox is not CodexSandbox.READ_ONLY
            or options.offered_tools != offers
            or not options.ephemeral
        ):
            raise CodexPipelineError("judge Codex thread options differ")

        compiled = {
            "rubric_id": rubric.rubric_id,
            "version": rubric.version,
            "digest": rubric.digest,
            "domain": rubric.domain,
            "stage": rubric.stage,
            "items": items,
        }
        evidence_projection = (
            codex_judge_evidence_index(request.workspace_evidence)
            if self._compact_evidence_index
            else _judge_evidence_projection(request.workspace_evidence)
        )
        public_payload = {"workspace_evidence": evidence_projection}
        if not self._compact_evidence_index:
            public_payload["policy_visible_context"] = request.policy_visible_context
        public = canonical_value(public_payload)
        private = canonical_value({
            "judge_only_reference": request.judge_only_reference,
            "compiled_rubric": compiled,
        })
        if not isinstance(public, dict) or not isinstance(private, dict):  # defensive
            raise CodexPipelineError("judge context projection differs")
        instruction = (
            (
                "You are a bounded evidence agent, not a one-shot text grader. Use no more than "
                f"{self._maximum_workspace_tool_frontiers} parallel tool rounds and "
                f"{self._maximum_workspace_tool_calls} total workspace calls. In the first round, "
                "completely read every required actor-evidence sidecar listed in "
                "workspace_evidence.actor_evidence, including the full policy-visible context, "
                "messages, tool trace, assistant output, and workspace bindings. In the next "
                "round, inspect a small rubric-relevant sample of source files from both before "
                "and after, batching independent reads in parallel. If the tool trace already "
                "proves a blocked run, missing submission, or immutable failure, score it low "
                "without exhaustive source-file exploration. Otherwise inspect only the most "
                "decision-relevant changed outputs. Then immediately return the exact structured "
                "verdict, cite only tool-inspected source-workspace references, and return no "
                "private reasoning. Do not reread evidence or request compaction."
            )
            if self._fast_workspace_judge
            else
            (
                "You are an evidence agent, not a one-shot text grader. Before scoring, completely "
                "read every required actor-evidence file listed in workspace_evidence.actor_evidence "
                "using workspace_read with snapshot=after and bounded offsets. Also completely read "
                "every source file in both before and after snapshots; use workspace_list to discover "
                "paths and batch independent chunk reads in parallel. Apply the exact compiled rubric "
                "and cite only tool-returned source-workspace references. Return no private reasoning."
            )
            if self._compact_evidence_index
            else (
                "Inspect immutable workspace content with workspace_read or workspace_search before "
                "scoring. Apply the exact compiled rubric and cite tool-returned workspace references."
            )
        ) + (
            " Copy exact strings from tool-returned inspected_evidence_refs into verdict evidence_refs, "
            "preserving the before/after snapshot and full path, including actor/. "
            "Never shorten paths, drop actor/, or reconstruct references."
            " hard_gates_passed is the conjunction of only compiled items with a non-null hard_gate. "
            "If no items define hard_gate, hard_gates_passed MUST be true regardless of item scores. "
            "This flag is not overall task success; do not invent additional hard gates."
        )
        turn_input = CodexTurnInput(
            public_text=instruction,
            public_context=public,
            judge_only_context=private,
            output_schema=_judge_output_schema(items),
            sandbox=CodexSandbox.READ_ONLY,
            model=self.model_id,
        )
        correction_deadline = self._bind_verdict_format_correction(
            request, rubric, workspace_tools, judge_bridge
        )

        def remaining_turn_timeout():
            if correction_deadline is not None:
                # Start the native bound after binding, but never extend the
                # already-running total deadline shared with the gateway.
                remaining = math.floor(correction_deadline - time.monotonic())
                if remaining < 1:
                    raise CodexPipelineError("judge correction total deadline expired")
                return min(self._turn_timeout_seconds, remaining)
            return self._turn_timeout_seconds if self._fast_workspace_judge else None

        live_results: Mapping[str, JudgeToolResult] | None = None
        if self._turn_mcp is None:
            receipt = _bounded_run_once(
                self._runner,
                options,
                turn_input,
                policy_context=public,
                timeout_seconds=remaining_turn_timeout(),
            )
            _require_terminal(receipt, options=options, expected_role=CodexRole.JUDGE)
            groups = self._judge_tool_inventory(receipt, offers)
            domain_groups = _mcp_groups(
                groups, set(receipt.offered_mcp_tool_names)
            )
            transport_frontier_count = len(domain_groups)
        else:
            with self._turn_mcp.open_judge(options, judge_bridge) as bound_options:
                if (
                    bound_options.role is not options.role
                    or bound_options.model != options.model
                    or bound_options.provider != options.provider
                    or bound_options.cwd != options.cwd
                    or bound_options.sandbox is not options.sandbox
                    or bound_options.offered_tools != offers
                    or not bound_options.ephemeral
                ):
                    raise CodexPipelineError("judge MCP-bound thread options differ")
                receipt = _bounded_run_once(
                    self._runner,
                    bound_options,
                    turn_input,
                    policy_context=public,
                    timeout_seconds=remaining_turn_timeout(),
                )
                _require_terminal(
                    receipt, options=bound_options, expected_role=CodexRole.JUDGE
                )
                groups = self._judge_tool_inventory(receipt, offers)
                domain_groups = _mcp_groups(
                    groups, set(receipt.offered_mcp_tool_names)
                )
                transport_frontier_count = len(domain_groups)
                domain_groups = self._live_judge_groups(receipt, domain_groups, judge_bridge)
                live_results = judge_bridge.bind_receipt(receipt, domain_groups)
        with self._claim_lock:
            self._receipts[request.judgment_id] = receipt
        if not domain_groups or any(len(group) > self._width for group in domain_groups):
            raise CodexPipelineError("judge did not execute a bounded workspace frontier", receipt=receipt)
        if self._fast_workspace_judge and (
            len(domain_groups) > self._maximum_workspace_tool_frontiers
            or sum(len(group) for group in domain_groups)
            > self._maximum_workspace_tool_calls
        ):
            raise CodexPipelineError(
                "judge exceeded the fast workspace inspection bound", receipt=receipt
            )

        judge_payload = {
            "workspace_evidence": evidence_projection,
            "judge_only_reference": request.judge_only_reference,
            "compiled_rubric": compiled,
            "instruction": instruction,
        }
        if not self._compact_evidence_index:
            judge_payload["policy_visible_context"] = request.policy_visible_context
        events = [
            _event(self._ids, role="system", content=instruction),
            _event(self._ids, role="user", content=canonical_value(judge_payload)),
        ]
        inspected_content = False
        for group in domain_groups:
            if live_results is None:
                calls = tuple(
                    ToolCall(
                        call_id=call.tool_call_id,
                        name=call.mcp_tool or "",
                        arguments=(
                            call.arguments if isinstance(call.arguments, Mapping) else {}
                        ),
                    )
                    for call in group
                )
                results = workspace_tools.execute_group(calls)
            else:
                results = tuple(live_results[call.tool_call_id] for call in group)
                calls = tuple(
                    ToolCall(
                        call_id=result.call_id,
                        name=result.name,
                        arguments=result.arguments,
                    )
                    for result in results
                )
            events.append(
                _event(
                    self._ids,
                    role="assistant",
                    content={
                        "text": "",
                        "tool_calls": [
                            {"call_id": call.call_id, "name": call.name, "arguments": call.arguments}
                            for call in calls
                        ]
                    },
                    tool_call_ids=tuple(call.call_id for call in calls),
                )
            )
            for codex_call, result in zip(group, results, strict=True):
                if live_results is None:
                    actual = canonical_value(_mcp_structured_output(codex_call))
                    expected = canonical_value(_judge_observation(result))
                    if actual != expected:
                        raise CodexPipelineError(
                            "judge MCP observation differs from immutable local replay",
                            receipt=receipt,
                        )
                inspected_content = inspected_content or (
                    result.status == "completed"
                    and result.content_inspection
                    and result.name in {"workspace_read", "workspace_search"}
                )
                events.append(
                    _event(
                        self._ids,
                        role="tool",
                        content={
                            "status": result.status,
                            "output": result.output,
                            "error_code": result.error_code,
                            "receipt_blake3": result.receipt_blake3,
                            "inspected_evidence_refs": result.inspected_evidence_refs,
                            "content_inspection": result.content_inspection,
                        },
                        tool_call_ids=(result.call_id,),
                    )
                )
        if not inspected_content:
            raise CodexPipelineError(
                "agent judge must complete at least one content read/search", receipt=receipt
            )
        events.append(_event(self._ids, role="assistant", content=receipt.final_response))
        trace = workspace_tools.trace(
            policy_events=tuple(events), provider_turn_count=1 + transport_frontier_count
        )
        if live_results is not None and not _live_judge_trace_matches(
            trace, domain_groups, live_results
        ):
            raise CodexPipelineError(
                "judge MCP execution and immutable trace differ", receipt=receipt
            )

        try:
            value = _agent_judge_terminal_object(receipt.final_response or "")
        except CodexPipelineError as exc:
            raise CodexPipelineError(str(exc), receipt=receipt) from None
        scores, hard_gates_passed = _validated_judge_verdict(
            value, items, trace, receipt=receipt
        )
        core = {
            "judgment_id": request.judgment_id,
            "judge_model_id": self.model_id,
            "rubric_digest": rubric.digest,
            "agent_trace": trace,
            "item_scores": tuple(scores),
            "hard_gates_passed": hard_gates_passed,
            "summary": value["summary"],
        }
        return JudgeAssessment(**core, assessment_blake3=blake3_hex(core))


class CodexOpus5AgentJudge(CodexWorkspaceAgentJudge):
    """Production Opus 5 judge; the existing model constraint remains mandatory."""

    def __init__(self, *args: Any, model_id: str, **kwargs: Any) -> None:
        normalized = model_id.casefold().replace("_", "-").replace(" ", "-")
        if "opus-5" not in normalized:
            raise CodexPipelineError("Codex agent judge model must be Opus 5")
        super().__init__(*args, model_id=model_id, **kwargs)


__all__ = [
    "ActorSkillsFactory",
    "ActorOptionsFactory",
    "ActorRuntimeFactory",
    "CODEX_JUDGE_MATERIAL_PREFIX",
    "CODEX_JUDGE_REQUIRED_MATERIAL_PATHS",
    "CodexJudgeToolBridgeObservation",
    "CodexJudgeToolExecutionBridge",
    "CodexOpus5AgentJudge",
    "CodexWorkspaceAgentJudge",
    "CodexPipelineError",
    "CodexRolloutAdapter",
    "CodexToolBridgeObservation",
    "CodexToolExecutionBridge",
    "CodexTurnRunnerPort",
    "codex_judge_evidence_index",
    "DEFAULT_ACTOR_SYSTEM_INSTRUCTION",
    "JudgeOptionsFactory",
    "RuntimeFactory",
    "SyncCodexTurnRunner",
    "TurnMCPBridgeFactoryPort",
]
