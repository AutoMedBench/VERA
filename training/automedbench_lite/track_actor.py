"""One persistent, tool-using coding workflow per public benchmark track."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

from eva_agent.codex_runtime import (CodexRole, CodexRuntime, CodexSandbox, CodexThreadOptions,
    CodexToolOffer, CodexTurnInput)
from eva_agent.codex_runtime.policy_budget import CodexPolicyBudgetDrainError
from eva_agent.codex_runtime.runtime import _logical_input
from eva_agent.pipeline.digests import canonical_value

from .actor import serving_binding, app_process
from .adapter import EvaluationError, file_digest, read_document, write_once
from .docker_runtime import resolve_image
from .local_qwen import local_qwen_setup
from .skill_surface import VerifiedEvaluationSkills
from .skill_discovery import build_public_skill_discovery, render_public_skill_discovery
from .track_tools import TOOLS, CATALOG_VERSION, MutableInventory, MODEL_TRACKS
from .job_wait import await_model_jobs
from .policy_capture import capture_turn
from .track_budget import TrackDeadline, TrackBudgetExhausted, cleanup_owned_jobs

ENTRY = Path(__file__).with_name("track_entry.py").absolute()
AUTO_COMPACT_TOKEN_LIMIT = 12288
PUBLIC_ARGUMENT_EXAMPLES = (
    ("automed_read_file", {"path": "notes/progress.md", "offset": 0, "limit": 2048}),
    ("automed_write_note", {"name": "progress.md", "content": "REPLACE_WITH_OBSERVED_PROGRESS"}),
)
PUBLIC_CONTEXT_GUIDANCE = (
    "Keep context bounded: every automed_read_file call must explicitly set limit <= 2048. "
    "Do not issue parallel large text reads. A read with complete=false is only a chunk, not the whole file. "
    "Do not dump or repeatedly page through task.json or inputs-manifest.json. Use existing CPU Python in "
    "automed_execute_python to load these public JSON files and print only needed keys, counts, selected "
    "case IDs and relevant paths; never print entire documents or arrays. Persist concise observed facts, "
    "selected IDs, artifact/job paths and next action in durable notes so continuation does not reread large inputs. "
    "Keep each progress note concise enough for a bounded read. automed_write_note takes name (a basename, "
    "such as progress.md), not path; content is the actual note text. The following are argument-shape examples, "
    "not calls to execute automatically; replace the note placeholder with observed facts:\n"
    + "\n".join(name + " " + json.dumps(arguments, separators=(",", ":"))
                for name, arguments in PUBLIC_ARGUMENT_EXAMPLES)
    + "\nFor segmentation and synthesis, automed_submit_extended_model_job accepts only case_ids in this "
    "track: use a JSON array of actual IDs selected from task.json. Omit sigma, hu_min, hu_max and confidence; "
    "do not send null or guessed defaults. Only enhancement uses explicit sigma, hu_min and hu_max, chosen "
    "from its public guidance. These examples do not change any offered tool schema or handler."
)
BASE = ("You are the trained Qwen coding orchestrator for one complete public AutoMedBench-Lite track. "
    "Use only the offered automed_eval MCP tools. Native shell, resource URIs, unrelated files, network, "
    "private references and hidden evaluators are unavailable. This is research benchmarking, not clinical care. "
    "The prescribed public model is an analysis TOOL with its own identity; you remain responsible for "
    "implementing a reusable pipeline, inspecting evidence and creating correctly formatted submissions.")
DEVELOPER = ("Maintain notes/progress.md with current stage, concrete artifacts, completed steps, uncertainty "
    "and next action; do not record private reasoning. Read that actual note at the start of continued work. "
    "Canonical search_skills and load_skill expose real stage-permitted guidance. Load relevant skills on demand; "
    "do not assume every tool mentioned by a skill exists here. Public task/model_info/lite hints define the "
    "prescribed method. Model jobs are ASYNCHRONOUS: submit once, retain the job ID, inspect status later while "
    "doing useful code work, and do not call a queued/running job successful. The durable job ledger is "
    "notes/model-jobs.jsonl. Model job artifact paths are public raw predictions, not native scores. "
    "Use actual Python in automed_execute_python to author/save code under code/, transform actual model "
    "outputs into native submission artifacts, and verify the case inventory and format. This CPU Docker has "
    "NumPy/SciPy/Pillow/Nibabel/Pandas/scikit-image, no network or GPU; fixed model jobs run separately. "
    "For classification/detection use automed_submit_model_job. For segmentation/synthesis/enhancement use "
    "automed_submit_extended_model_job; enhancement requires explicit noise sigma and normalization HU bounds "
    "from public model guidance, distinct from required output HU clipping. "
    "For VQA/report use automed_submit_generative_model_job with explicit max_new_tokens; VQA additionally "
    "requires multi_image_mode=montage (all images), report requires an explicit public-guidance prompt. "
    "Keep inputs, task.json, inputs-manifest.json and public-guidance unchanged. There are 160 tool calls and "
    "32 code executions across this entire track. Avoid repeated successful calls and busy polling. "
    "After each requested phase, update progress once and STOP with concise observed results and next action. "
    "Never invent a prediction, decoded output, successful execution, clinical score or completion.\n\n"
    + PUBLIC_CONTEXT_GUIDANCE)
PHASES = (
    ("01-planning", "S1", "Search/load a relevant S1 bounded-plan skill. Inspect task.json and public-guidance/model_info.yaml "
     "through bounded reads or public Python summaries, "
     "inspect one real public input and the public output contract. Plan a reusable prescribed-model pipeline for "
     "the WHOLE selected track, not independent per-case chats. Save notes/plan.md and notes/progress.md. Do not run full inference yet."),
    ("02-setup", "S2", "Read notes/progress.md. Search/load a relevant S2 skill. Read public-guidance/lite_s1.md and lite_s2.md. "
     "Prepare reusable actual Python code for reading task/case manifests, consuming prescribed-model job outputs, "
     "writing the native submission format, and validating it. Save the program under code/ using the CPU Python tool. "
     "Inspect the available scientific libraries through real code. Record the exact prescribed model and setup limitations. "
     "You may submit ONE first-case model smoke job now; record its job ID. Update progress and finish."),
    ("03-smoke", "S3", "Read notes/progress.md and the job ledger if present. Search/load a relevant S3 medical-visual skill. "
     "Run one actual first-case prescribed model smoke (reuse an existing submitted job; do not duplicate it). "
     "Check its authoritative status. Implement/execute your saved pipeline to convert the actual raw prediction to "
     "the native submission format and validate it against public requirements. If the job is still running, "
     "finish with the truthful pending job ID instead of busy-polling. Save observed results and next action."),
    ("04-full-subset", "S4", "This is the SAME track/thread resumed after a real app-server restart. First actually read "
     "notes/progress.md and notes/model-jobs.jsonl to recover state. Search/load a relevant S4 long-horizon skill. "
     "Complete the reusable pipeline over EVERY case ID in task.json. Reuse previous actual completed jobs; submit "
     "remaining cases in batches of at most 32 with the prescribed model tool. Use job status and actual raw artifacts. "
     "Execute your saved code to assemble native outputs and validate coverage. Persist completed/pending job IDs, "
     "artifacts, uncertainties and next steps. If jobs are pending, report them honestly for continuation."),
    ("05-review", "S5", "Read progress and job ledger; search/load a relevant S5 review skill. Inspect actual completed "
     "model jobs, use the implemented pipeline to finish every available native output, and run a real completeness/format "
     "check against the full case manifest. If jobs are pending, report exact pending work, never claim complete. "
     "Write notes/final.md and update progress with evidence, limitations and output paths. No private native score is visible."),
)


def phase_prompt(skills, index, track, *, selection=None, catalog_id=None):
    from eva_agent.training.stage_execution_guidance import stage_execution_guidance
    phase, stage, prompt = PHASES[index]
    if track == "vqa" and stage == "S3":
        prompt += (" The VQA track-specific public protocol additionally requires calibration on at least "
            "15 distinct public cases; one-case smoke alone does not complete that phase. Use only actual "
            "retained predictions and public artifacts, never invented calibration results.")
    discovery = build_public_skill_discovery(skills, stage)
    prompt += "\n\n" + render_public_skill_discovery(discovery)
    execution_guidance = stage_execution_guidance(stage)
    if execution_guidance:
        prompt += "\n\n" + execution_guidance
    if selection:
        from training.eva_rsi.skill_selection import render_selection
        prompt += render_selection(selection, stage=stage,
            visible_ids=[row["skill_id"] for row in discovery["skills"]], catalog_id=catalog_id)
    return phase, stage, prompt, discovery


def snapshot(inventory, target):
    observed = inventory.capture()
    target.mkdir(parents=True, exist_ok=False, mode=0o700)
    for row in observed["files"]:
        destination = target / "files" / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Immutable content-addressed blobs are retained independently; snapshots
        # use links to those archival blobs, not to actor-writable source files.
        os.link(inventory.blobs / row["blake3"], destination)
    return write_once(target / "manifest.json", {"schema": "eva.automedbench-track-snapshot.v1", **observed})


def thread_options(setup, workspace, audit, args, image):
    names = [row["name"] for row in TOOLS]
    offers = tuple(CodexToolOffer(fully_qualified_name="automed_eval/" + row["name"], description=row["description"],
        input_schema=row["inputSchema"], parallel_safe=False,
        read_only=row["name"] in {"automed_read_file", "automed_view_input", "search_skills", "load_skill"},
        allowed_stages=("E2E",)) for row in TOOLS)
    config = {**setup.thread_config, "mcp_servers": {"automed_eval": {
        "command": str(args.public_python.absolute()), "args": ["-I", "-B", str(ENTRY), "serve",
            "--workspace", str(workspace), "--audit-root", str(audit), "--image", image,
            "--runtime-manifest", str(args.runtime_manifest.absolute()), "--public-python", str(args.public_python.absolute())],
        "cwd": str(workspace), "required": True, "startup_timeout_sec": 30, "tool_timeout_sec": 200,
        "env": {"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
                **({"EVA_HARNESS_ROOT": os.environ["EVA_HARNESS_ROOT"]} if os.environ.get("EVA_HARNESS_ROOT") else {})},
        "enabled_tools": names, "omit_tools_from": ["deferred", "code_mode"],
        "tools": {name: {"approval_mode": "approve"} for name in names}}}}
    tool_output_limit = getattr(args, "tool_output_token_limit", None)
    if tool_output_limit is not None:
        if type(tool_output_limit) is not int or not 256 <= tool_output_limit <= 4096:
            raise EvaluationError("native_tool_output_token_limit_invalid")
        # Native Codex bounds its context view; exact original MCP outputs
        # and canonical schemas remain unchanged in the host audit.
        config["tool_output_token_limit"] = tool_output_limit
    developer = DEVELOPER
    if getattr(args, "track_timeout", None) is not None:
        developer += ("\nThis entire S1-S5 track has ONE shared 3600-second budget, including model-job "
            "waits and runtime restart, not 3600 seconds per stage. Stage names express requested work, "
            "not successful completion. Inspect real prerequisite artifacts and recover missing work "
            "within the remaining budget; never invent predecessor outputs or later-stage success.")
    return CodexThreadOptions(role=CodexRole.STRONG_ACTOR, model=setup.model, provider=setup.provider,
        cwd=str(workspace), sandbox=CodexSandbox.READ_ONLY, ephemeral=False, config=config,
        offered_tools=offers, base_instructions=BASE, developer_instructions=developer)


async def settle_between_turn_deadline(state):
    """No invented turn/interruption when the shared clock ends between turns."""
    from .policy_capture import require_joined_host_results
    deadline = state["deadline"]
    document = {"schema": "eva.automedbench-between-turn-budget-terminal.v1",
        "track_budget": deadline.observation(), "workspace_quiescence_verified": False,
        "infrastructure_error": None, "reward": None, "later_stages_evaluated": False}
    try:
        cleanup = await cleanup_owned_jobs(state["audit"], state["workspace"])
        document["deadline_cleanup_document_blake3"] = cleanup["document_blake3"]
        if not state["receipts"]:
            raise EvaluationError("track_deadline_without_actual_turn")
        receipt = state["receipts"][-1]
        phase = PHASES[len(state["receipts"]) - 1][0]
        after = read_document(state["audit"] / "turns" / phase / "after/manifest.json")
        # Do not relabel a later asynchronous mutation as the original after.
        current = state["inventory"].capture()
        excluded = {"schema", "document_blake3", "observation_started_ns", "observation_completed_ns"}
        if {k:v for k,v in current.items() if k not in excluded} != {
                k:v for k,v in after.items() if k not in excluded}:
            cleanup_after = snapshot(state["inventory"], state["audit"] / "track-deadline-after")
            document.update(cleanup_after_snapshot_relative="track-deadline-after",
                cleanup_after_snapshot_document_blake3=cleanup_after["document_blake3"],
                original_phase_after_preserved=True,
                late_outputs_excluded_from_prior_stage_A=True)
        document.update(require_joined_host_results(receipt, state["audit"]))
        document.update(phase_intent=phase, receipt_blake3=receipt.receipt_blake3,
            actual_terminal_status=receipt.status, workspace_quiescence_verified=True,
            after_snapshot_document_blake3=after["document_blake3"],
            quiescence_observed_ns=current.get("observation_completed_ns"))
        state["errors"].append({"phase": phase, "error": "track_budget_exhausted",
            "category": "policy_budget", "infrastructure_error": None, "reward": None})
    except Exception as exc:
        document["infrastructure_error"] = str(exc) if isinstance(exc, EvaluationError) else type(exc).__name__
        state["errors"].append({"phase": "track_deadline_cleanup", "error": document["infrastructure_error"],
            "reward": None})
    finally:
        document["track_budget"] = deadline.observation()
        write_once(state["audit"] / "track-budget-terminal.json", document)


@asynccontextmanager
async def deadline_runtime(backend, deadline, initialization_lock, *, audit, runtime_sequence, phase_intent):
    """Open one app server at a time while leaving turns fully concurrent."""
    runtime = CodexRuntime(backend)
    initialization_failed = False
    try:
        # Backend.open already closes a partially started SDK client on
        # cancellation. The outer exit remains responsible for any partially
        # entered runtime, and lock waiting consumes this track's same deadline.
        async def initialize():
            async with initialization_lock:
                await runtime.__aenter__()
        try:
            await asyncio.wait_for(initialize(), deadline.require_remaining())
        except BaseException as exc:
            initialization_failed = True
            cleanup_error = None
            try:
                await runtime.__aexit__(None, None, None)
            except BaseException as cleanup_exc:
                cleanup_error = type(cleanup_exc).__name__
            write_once(audit / "app-server-initialization-error.json", {
                "schema": "eva.automedbench-app-server-initialization-error.v1",
                "operation": "codex_runtime_initialize", "status": "failed",
                "runtime_sequence": runtime_sequence, "phase_intent": phase_intent,
                "error_category": type(exc).__name__,
                "runtime_initialization_completed": False,
                "thread_start_or_resume_called_on_this_runtime": False,
                "partial_runtime_cleanup_attempted": True,
                "partial_runtime_cleanup_error_category": cleanup_error,
                "track_budget_document_blake3": deadline.document["document_blake3"],
            })
            raise
        yield runtime
    finally:
        if not initialization_failed:
            await runtime.__aexit__(None, None, None)


async def run_whole_track(setup, state, turn, *, seconds, initialization_lock):
    """Called only after acquiring a worker slot; each track owns both runtimes."""
    state["deadline"] = TrackDeadline(state["audit"], seconds=seconds)
    state["app_server_pids"] = []
    state["actual_restart"] = False
    first_stopped = False
    try:
        for runtime_sequence, block in enumerate(((0, 1, 2), (3, 4)), start=1):
            state["deadline"].require_remaining()
            backend = setup.backend()
            async with deadline_runtime(backend, state["deadline"], initialization_lock,
                    audit=state["audit"], runtime_sequence=runtime_sequence,
                    phase_intent=PHASES[block[0]][0]) as runtime:
                process = app_process(backend)
                state["app_server_pids"].append(process.pid)
                if block[0] == 0:
                    operation = runtime.start_thread(state["options"])
                else:
                    operation = runtime.resume_thread(state["thread_id"], state["options"])
                state["handle"] = await asyncio.wait_for(operation, state["deadline"].require_remaining())
                if block[0] == 0:
                    state["thread_id"] = state["handle"].thread_id
                    write_once(state["audit"] / "thread.json", {
                        "thread_id": state["thread_id"], "app_server_pid": process.pid})
                else:
                    state["actual_restart"] = first_stopped and state["app_server_pids"][0] != process.pid
                for index in block:
                    state["deadline"].require_remaining()
                    await turn(runtime, state, index)
                    if state["errors"]:
                        return
            first_stopped = process.poll() is not None
            if not first_stopped:
                raise EvaluationError("track_owned_app_server_not_stopped")
    except TrackBudgetExhausted:
        await settle_between_turn_deadline(state)
    except asyncio.TimeoutError:
        if state["deadline"].remaining() == 0:
            await settle_between_turn_deadline(state)
        else:
            state["errors"].append({"phase": "thread_start_or_resume", "error": "TimeoutError", "reward": None})
    except Exception as exc:
        state["errors"].append({"phase": "thread_start_or_resume", "error": type(exc).__name__, "reward": None})
    finally:
        if state["errors"] and not (state["audit"] / "track-deadline-cleanup.json").exists():
            try:
                await cleanup_owned_jobs(state["audit"], state["workspace"], trigger="track_error")
            except Exception as exc:
                state["errors"].append({"phase": "owned_model_job_cleanup", "error": type(exc).__name__, "reward": None})
        write_once(state["audit"] / "track-budget-outcome.json", {
            "schema": "eva.automedbench-track-budget-outcome.v1",
            **state["deadline"].observation(), "actual_turn_count": len(state["receipts"]),
            "app_server_pids": state["app_server_pids"], "phase_labels_assert_success": False})


async def run_tracks(args, *, memory_profile=False, supra_profile=False):
    workers = getattr(args, "workers", 4)
    if type(workers) is not int or not 1 <= workers <= 4 or memory_profile and supra_profile:
        raise EvaluationError("track_worker_or_memory_profile_configuration_invalid")
    context_tokens = getattr(args, "context_length", 32768)
    compact_tokens = getattr(args, "auto_compact_token_limit", AUTO_COMPACT_TOKEN_LIMIT)
    output_tokens = getattr(args, "output_tokens", 4096)
    track_timeout = getattr(args, "track_timeout", None)
    if track_timeout is not None and (type(track_timeout) is not int or track_timeout != 3600):
        raise EvaluationError("track_timeout_must_be_3600_seconds")
    tool_output_limit = getattr(args, "tool_output_token_limit", None)
    if tool_output_limit is not None and (type(tool_output_limit) is not int or not 256 <= tool_output_limit <= 4096):
        raise EvaluationError("native_tool_output_token_limit_invalid")
    memory_enabled = memory_profile or supra_profile
    run = args.run_root.resolve(strict=True)
    manifest = read_document(run / "track-run-manifest.json")
    selected = [row["track"] for row in manifest["tracks"]] if args.tracks == ["all"] else args.tracks
    if len(set(selected)) != len(selected) or not set(selected) <= MODEL_TRACKS:
        raise EvaluationError("selected_track_model_runtime_not_admitted")
    image = resolve_image(args.image)
    binding = serving_binding(args.server_canary, args.server_identity)
    root = run / "track-rollouts"
    root.mkdir(mode=0o700, exist_ok=False)
    skills = VerifiedEvaluationSkills(root / "skill-preflight")
    selection = None
    if getattr(args, "skill_selection", None) is not None:
        from training.eva_rsi.skill_selection import read_selection
        selection = {"path": str(args.skill_selection), "blake3": args.skill_selection_blake3}
        read_selection(selection, catalog_id=args.skill_catalog_id)
    memory_binding, memory_skill_path = None, None
    if memory_enabled:
        from .track_memory_v12 import prepare_memory_profile
        memory_skill_path, memory_binding = prepare_memory_profile(root / "extra-memory-skill", skills,
            supra=supra_profile, effective_context={"context_tokens": context_tokens,
                "compact_at_tokens": compact_tokens, "output_tokens": output_tokens,
                "reserve_tokens": 2048, "token_budget_margin": 256,
                "compaction_headroom_mode": "exact_request_guard" if supra_profile else "caller_override",
                "per_request_token_budget_wrapper": True,
                "native_tool_output_token_limit": tool_output_limit,
                "multimodal_token_count_exact": False,
                "measured_compacted_input_tokens": 13100 if supra_profile else None})
    write_once(root / "attempt.json", {"schema": "eva.automedbench-track-attempt.v1", "attempt_id": str(uuid4()),
        "tracks": selected, "planned_coding_rollouts": len(selected), "one_attempt_per_track": True,
        "max_parallel_tracks": workers,
        **({"track_budget_policy": "admitted-track-wallclock-3600-v1", "track_timeout_seconds": track_timeout,
            "phase_timeout_policy": "remaining_track_budget", "legacy_turn_timeout_not_applied": args.turn_timeout}
           if track_timeout is not None else {}),
        **({"skill_selection": selection} if selection else {}),
        "native_tool_output_token_limit": tool_output_limit,
        "original_mcp_responses_retained": True,
        **({"extra_memory_skill_binding": memory_binding} if memory_enabled else {}),
        "server_binding": binding, "docker_image": image, "tool_catalog_version": CATALOG_VERSION,
        "verified_skill_catalog_blake3": skills.catalog_blake3, "skill_inventory": skills.inventory,
        "runtime_manifest_file_blake3": file_digest(args.runtime_manifest), "phase_labels_are_intents_only": True,
        "full_seven_track_evaluation": set(selected) == set(row["track"] for row in manifest["tracks"])})
    states = []
    for row in manifest["tracks"]:
        if row["track"] not in selected: continue
        workspace = run / row["workspace_relative"]
        if file_digest(workspace / "task.json") != row["task_file_blake3"] or file_digest(workspace / "inputs-manifest.json") != row["input_manifest_file_blake3"]:
            raise EvaluationError("track_public_task_binding_changed")
        if any((workspace / "notes").iterdir()) or any((workspace / "outputs/agents_outputs").iterdir()):
            raise EvaluationError("track_workspace_already_used")
        audit = root / row["track"]
        audit.mkdir(mode=0o700)
        states.append({"track": row["track"], "workspace": workspace, "audit": audit,
            "inventory": MutableInventory(workspace, audit), "receipts": [], "errors": []})
    # Queue whole track phases before the gateway; its four-request cap is not a queue.
    slots = asyncio.Semaphore(workers)
    initialization_lock = asyncio.Lock()
    async def bounded(function, state):
        async with slots:
            await function(state)
    with local_qwen_setup(run_root=root / "local-runtime", image_inputs=True, workers=min(workers, len(states)),
            thinking=True, exact_tool_schemas=True, normalize_priority_messages=True,
            context_length=context_tokens, max_output_tokens=output_tokens,
            upstream_timeout_seconds=600, codex_bin=args.codex_bin.absolute(),
            endpoint=getattr(args, "endpoint", "http://127.0.0.1:30910/v1"),
            capacity_wait_seconds=getattr(args, "adapter_capacity_wait_seconds", 0.0),
            auto_compact_token_limit=compact_tokens, token_budget=True) as setup:
        profile = None
        if memory_enabled:
            from .track_memory_v12 import apply_setup, make_supra_profile
            setup = apply_setup(setup)
            if supra_profile:
                profile = make_supra_profile(setup, context_tokens=context_tokens, output_tokens=output_tokens,
                    compact_tokens=compact_tokens, endpoint=getattr(args, "endpoint", "http://127.0.0.1:30910/v1"))
                write_once(root / "supra-profile.json", profile.inspection())
        write_once(root / "backend.json", setup.safe_metadata)
        for state in states:
            state["options"] = thread_options(setup, state["workspace"], state["audit"], args, image)
            if memory_enabled:
                from .track_memory_v12 import apply_thread
                state["options"] = apply_thread(state["options"], supra_profile=profile)

        async def turn(runtime, state, index):
            phase, stage, prompt, discovery = phase_prompt(skills, index, state["track"],
                selection=selection, catalog_id=getattr(args, "skill_catalog_id", None))
            if index in (2, 4):
                if track_timeout is None:
                    await await_model_jobs(state["audit"], phase)
                else:
                    await await_model_jobs(state["audit"], phase,
                        timeout=state["deadline"].remaining(), track_deadline=state["deadline"])
            if track_timeout is not None:
                state["deadline"].require_remaining()
            target = state["audit"] / "turns" / phase
            target.mkdir(parents=True, mode=0o700)
            write_once(target / "public-skill-discovery.json", discovery)
            policy = state["audit"] / ("phase-policy-" + str(uuid4()) + ".json")
            write_once(policy, {"skill_stage": stage, "phase_intent": phase})
            os.replace(policy, state["audit"] / "phase-policy.json")
            value = CodexTurnInput(public_text=prompt, model=setup.model, effort="medium", summary="none")
            if memory_enabled:
                from .track_memory_v12 import apply_turn
                value = apply_turn(value, memory_skill_path, supra_profile=profile)
            write_once(target / "request.json", {"schema": "eva.automedbench-codex-track-request.v1", "phase_intent": phase,
                "logical_input": canonical_value(_logical_input(state["options"], value)),
                **({"skill_selection": selection} if selection else {}),
                "base_instructions": state["options"].base_instructions,
                "developer_instructions": state["options"].developer_instructions, "public_tool_catalog": list(TOOLS),
                **({"extra_memory_skill_binding_blake3": memory_binding["document_blake3"]} if memory_enabled else {}),
                "tool_catalog_version": CATALOG_VERSION, "verified_skill_catalog_blake3": skills.catalog_blake3})
            snapshot(state["inventory"], target / "before")
            try:
                timeout = args.turn_timeout
                deadline_options = {}
                if track_timeout is not None:
                    timeout = state["deadline"].require_remaining()
                    write_once(target / "track-budget.json", {
                        "schema": "eva.automedbench-track-turn-budget.v1",
                        "budget_document_blake3": state["deadline"].document["document_blake3"],
                        "timeout_seconds": timeout, "elapsed_before_turn_seconds": 3600 - timeout})
                    deadline_options["track_deadline"] = state["deadline"]
                receipt, budget_terminal = await capture_turn(runtime, state["handle"], value,
                    timeout=timeout, target=target, audit=state["audit"],
                    inventory=state["inventory"], snapshot=snapshot, **deadline_options)
                state["receipts"].append(receipt)
                print(json.dumps({"track": state["track"], "phase": phase, "status": receipt.status,
                                  "actual_tool_calls": len(receipt.tool_calls)}), flush=True)
                if budget_terminal is not None:
                    state["errors"].append({"phase": phase, "error": "policy_turn_budget_exhausted",
                        "category": "policy_budget", "infrastructure_error": None, "reward": None})
                elif receipt.status != "completed": state["errors"].append({"phase": phase, "error": "turn_not_completed"})
            except TrackBudgetExhausted:
                raise
            except Exception as exc:
                if track_timeout is not None and (target / "receipt.json").is_file():
                    # A failed cleanup does not erase an already validated
                    # actual Codex turn from usage/attempt counts.
                    from eva_agent.codex_runtime import codex_turn_receipt_from_document
                    retained = codex_turn_receipt_from_document(json.loads((target / "receipt.json").read_bytes()))
                    if not any(item.receipt_blake3 == retained.receipt_blake3 for item in state["receipts"]):
                        state["receipts"].append(retained)
                failure = {"phase": phase, "error": type(exc).__name__, "reward": None}
                if isinstance(exc, CodexPolicyBudgetDrainError):
                    failure["policy_budget"] = exc.outcome
                state["errors"].append(failure)
                write_once(target / "failure.json", failure)
                print(json.dumps({"track": state["track"], **failure}), flush=True)

        if track_timeout is not None:
            async def whole(state):
                await run_whole_track(setup, state, turn, seconds=track_timeout,
                    initialization_lock=initialization_lock)
            await asyncio.gather(*(bounded(whole, state) for state in states))
        else:
            backend = setup.backend()
            async with CodexRuntime(backend) as runtime:
                process = app_process(backend)
                first_pid = process.pid
                async def start(state):
                    try:
                        state["handle"] = await runtime.start_thread(state["options"])
                        state["thread_id"] = state["handle"].thread_id
                        write_once(state["audit"] / "thread.json", {"thread_id": state["thread_id"], "app_server_pid": first_pid})
                        for index in (0, 1, 2):
                            await turn(runtime, state, index)
                            if state["errors"]: break
                    except Exception as exc:
                        state["errors"].append({"phase": "thread_start_or_capture", "error": type(exc).__name__})
                await asyncio.gather(*(bounded(start, state) for state in states))
            stopped = process.poll() is not None
            backend = setup.backend()
            async with CodexRuntime(backend) as runtime:
                second_pid = app_process(backend).pid
                async def resume(state):
                    if state["errors"]: return
                    try:
                        state["handle"] = await runtime.resume_thread(state["thread_id"], state["options"])
                        for index in (3, 4):
                            await turn(runtime, state, index)
                            if state["errors"]: break
                    except Exception as exc:
                        state["errors"].append({"phase": "thread_resume_or_capture", "error": type(exc).__name__})
                await asyncio.gather(*(bounded(resume, state) for state in states))
    summaries = []
    for state in states:
        receipts = state["receipts"]
        summary = write_once(state["audit"] / "rollout.json", {"schema": "eva.automedbench-codex-track-rollout.v1",
            "track": state["track"], "run_id": manifest["run_id"], "thread_id": state.get("thread_id"),
            "actual_turn_count": len(receipts), "turn_receipt_blake3s": [r.receipt_blake3 for r in receipts],
            "completed_requested_turns": len(receipts) == 5 and not state["errors"], "errors": state["errors"],
            "requested_model": setup.model, "returned_model": None, "provider_backend": "local_qwen_responses_adapter",
            "source_checkpoint": binding["canary"]["exact_final_model_path"],
            "app_server_pids": state["app_server_pids"] if track_timeout is not None else [first_pid, second_pid],
            "actual_restart": state["actual_restart"] if track_timeout is not None else stopped and first_pid != second_pid,
            **({"track_budget": read_document(state["audit"] / "track-budget-outcome.json")}
               if track_timeout is not None else {}),
            "same_thread_resumed": any(r.thread_resumed for r in receipts) and len({r.thread_id for r in receipts}) == 1,
            "phase_labels_assert_rubric_success": False, "agent_judged": False, "native_score_available_to_actor": False,
            "native_score_completed": False, "canonical_evamed_tool_profile": False})
        summaries.append(summary)
    result = write_once(root / "summary.json", {"schema": "eva.automedbench-track-rollout-summary.v1", "tracks": summaries,
        "completed_requested_workflows": sum(row["completed_requested_turns"] for row in summaries),
        "full_seven_track_evaluation_complete": False, "no_missing_reward_fallback": True})
    print(json.dumps({"run_root": str(run), "completed_requested_workflows": result["completed_requested_workflows"]}), flush=True)
