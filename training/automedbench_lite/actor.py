"""Three real coding-agent cases, persistent context, restart/resume and native scores.

Only public task artifacts enter actor tools. Phase names express requested work,
not success under an EVA medical rubric. Canonical EvaMed schemas are unchanged.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

from eva_agent.codex_runtime import (CodexRole, CodexRuntime, CodexSandbox, CodexThreadOptions,
    CodexToolOffer, CodexTurnInput, verify_codex_turn_receipt)
from eva_agent.codex_runtime.runtime import _logical_input
from eva_agent.pipeline.digests import canonical_json_bytes, canonical_value

from .adapter import EvaluationError, blake3, file_digest, load_assets, read_document, score_case, write_once
from .docker_runtime import resolve_image, workspace_inventory
from .local_qwen import local_qwen_setup
from .public_tools import CATALOG_VERSION, TOOLS
from .skill_surface import VerifiedEvaluationSkills

ENTRY = Path(__file__).with_name("actor_entry.py").resolve()
BASE = ("You are a coding research agent performing one public AutoMedBench-Lite case. "
        "Use only the offered automed_eval MCP tools. No native shell, external network, private references, "
        "additional model endpoints, or hidden evaluator files are available. This is research evaluation, "
        "not patient care. Distinguish observations, hypotheses, implemented code, and verified results.")
DEVELOPER = ("Maintain notes/progress.md across turns, including concrete observations, decisions, "
             "uncertainties and next actions, not hidden chain of thought. Inspect actual public input before "
             "making predictions. Implement your chosen method as real Python through automed_execute_python. "
             "Available scientific libraries are documented by that tool; no pretrained task model is supplied. "
             "Be honest about approximations and do not claim an unavailable native score. Preserve input "
             "and task.json unchanged. Follow the exact public submission format. Tool and Python budgets "
             "are 64 and 16 respectively across this case, including resumed turns. "
             "Use the canonical search_skills and load_skill tools for actual stage-permitted guidance. "
             "Skill bodies describe broader EvaMed workflows; only tools actually offered here are available. "
             "Do not invent unavailable named tools or treat a skill load as task completion. "
             "After the requested phase work succeeds, STOP calling tools and send the concise final response. "
             "Do not repeat an already-successful read, note write, or execution without a concrete new reason.")
PHASES = (
    ("01-planning", "First search_skills with stage='S1' and query='bounded-plan'; load the matching relevant "
     "skill with stage='S1'. Read task.json using automed_read_file and inspect the actual input with automed_view_input. "
     "Understand the public taxonomy and native output contract. Plan a concrete feasible image/volume analysis "
     "method given the available CPU libraries. You may use bounded Python to inspect public dimensions or "
     "statistics, but do not create the final prediction yet. Save observations, uncertainty and next steps "
     "in notes/progress.md. Finish with a concise plan grounded in what you actually inspected."),
    ("02-implementation", "Continue our same case using the retained conversation. Search_skills with stage='S3' "
     "and query='medical-visual', then load the matching relevant skill with stage='S3'. First READ notes/progress.md "
     "with automed_read_file; it has not been reinserted into this prompt. Implement the planned method using "
     "real Python and the actual public input. Write the exact native prediction artifact specified in task.json, "
     "then validate its format and geometry using code. You may revise the method based on actual observations. "
     "Update notes/progress.md with actual code results, limitations and pending review. Do not claim a score."),
    ("03-review", "Resume our existing case after an actual app-server restart. Search_skills with stage='S4' "
     "and query='long-horizon', then load the matching relevant skill with stage='S4'. Use prior conversation and READ "
     "notes/progress.md to recover the latest state. Inspect the actual prediction and validate the submission "
     "format, public label range and, where applicable, shape/affine or box bounds through real Python. Correct "
     "any observed defect using available tools. Write notes/final.md summarizing actual evidence, method, "
     "limitations and output path. No private labels or native score are available in this actor phase."),
)


def snapshot(workspace: Path, destination: Path) -> dict:
    inventory = workspace_inventory(workspace)
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    rows = []
    for row in inventory["files"]:
        source = workspace / row["path"]
        target = destination / "files" / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(source, target)
        target.chmod(0o400)
        if file_digest(target) != row["blake3"]:
            raise EvaluationError("snapshot_changed_during_retention")
        rows.append({**row, "mode": source.stat().st_mode & 0o777})
    return write_once(destination / "manifest.json", {"schema": "eva.automedbench-workspace-snapshot.v1",
        "files": rows, "file_count": len(rows), "bytes": inventory["bytes"],
        "tree_blake3": blake3(canonical_json_bytes(rows)).hexdigest(), "retained_bytes_relative": "files"})


def serving_binding(canary_path: Path, identity_path: Path) -> dict:
    if max(canary_path.stat().st_size, identity_path.stat().st_size) > 1024 * 1024:
        raise EvaluationError("server_binding_too_large")
    canary, identity = json.loads(canary_path.read_bytes()), json.loads(identity_path.read_bytes())
    model = canary.get("exact_final_model_path")
    pid = canary.get("server_pid")
    if (canary.get("schema") != "eva.qwen-final-serving-image-canary.v1"
            or canary.get("status") != "complete" or canary.get("http_status") != 200
            or canary.get("multimodal_request_accepted") is not True
            or canary.get("checkpoint_identity_blake3") != file_digest(identity_path)
            or model != identity.get("exact_final_model_path")
            or canary.get("model_info", {}).get("model_path") != model
            or type(pid) is not int or pid <= 0):
        raise EvaluationError("server_checkpoint_canary_binding_invalid")
    argv = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    if argv != canary.get("actual_server_argv") or "--model-path" not in argv or argv[argv.index("--model-path") + 1] != model:
        raise EvaluationError("current_server_process_checkpoint_differs")
    return {"canary": canary, "identity": identity, "canary_file_blake3": file_digest(canary_path),
            "identity_file_blake3": file_digest(identity_path), "actual_process_argv_rechecked": True}


def thread_options(setup, workspace: Path, audit: Path, public_python: Path, image: str):
    offers = tuple(CodexToolOffer(fully_qualified_name="automed_eval/" + row["name"],
        description=row["description"], input_schema=row["inputSchema"], parallel_safe=False,
        read_only=row["name"] in {"automed_read_file", "automed_view_input", "search_skills", "load_skill"},
        allowed_stages=("E2E",)) for row in TOOLS)
    names = [row["name"] for row in TOOLS]
    config = {**setup.thread_config, "mcp_servers": {"automed_eval": {
        "command": str(public_python), "args": ["-I", "-B", str(ENTRY), "serve", "--workspace", str(workspace),
            "--audit-root", str(audit), "--image", image], "cwd": str(workspace),
        "env": {"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""},
        "required": True, "startup_timeout_sec": 30, "tool_timeout_sec": 200,
        "enabled_tools": names, "omit_tools_from": ["deferred", "code_mode"],
        "tools": {name: {"approval_mode": "approve"} for name in names}}}}
    return CodexThreadOptions(role=CodexRole.STRONG_ACTOR, model=setup.model, provider=setup.provider,
        cwd=str(workspace), sandbox=CodexSandbox.READ_ONLY, ephemeral=False, config=config,
        offered_tools=offers, base_instructions=BASE, developer_instructions=DEVELOPER)


def app_process(backend):
    return backend._client._client._sync._proc


async def run_evaluation(args):
    run = args.run_root.resolve(strict=True)
    manifest = read_document(run / "run-manifest.json")
    if manifest["diagnostic_only"] or len(manifest["cases"]) != 3:
        raise EvaluationError("requires_three_fresh_nondiagnostic_cases")
    image = resolve_image(args.image)
    binding = serving_binding(args.server_canary, args.server_identity)
    root = run / "codex-rollouts"
    root.mkdir(mode=0o700, exist_ok=False)
    skills = VerifiedEvaluationSkills(root / "skill-preflight")
    write_once(root / "attempt.json", {"attempt_id": str(uuid4()), "planned_unique_cases": 3,
        "planned_rollouts_per_case": 1, "phases_are_intents_not_rubric_success": True,
        "tool_catalog_version": CATALOG_VERSION, "docker_image": image, "server_binding": binding,
        "verified_skill_catalog_blake3": skills.catalog_blake3, "skill_inventory": skills.inventory,
        "skill_bodies_in_initial_prompt": False})
    states = []
    for case in manifest["cases"]:
        audit = root / case["case_id"]
        audit.mkdir(mode=0o700)
        workspace = run / case["workspace_relative"]
        if any((workspace / "outputs").iterdir()) or (workspace / "notes").exists():
            raise EvaluationError("actor_workspace_already_attempted")
        states.append({"case": case, "audit": audit, "workspace": workspace, "receipts": [], "errors": []})

    with local_qwen_setup(run_root=root / "local-runtime", image_inputs=True, workers=3,
                          thinking=True, max_output_tokens=8192) as setup:
        write_once(root / "backend.json", setup.safe_metadata)
        for state in states:
            state["options"] = thread_options(setup, state["workspace"], state["audit"], args.public_python.absolute(), image)

        async def one_turn(runtime, state, index):
            phase, prompt = PHASES[index]
            target = state["audit"] / "turns" / phase
            target.mkdir(parents=True, mode=0o700)
            phase_policy = state["audit"] / ("phase-policy-" + str(uuid4()) + ".json")
            write_once(phase_policy, {"skill_stage": ("S1", "S3", "S4")[index], "phase_intent": phase})
            os.replace(phase_policy, state["audit"] / "phase-policy.json")
            value = CodexTurnInput(public_text=prompt, model=setup.model, effort="medium", summary="none")
            write_once(target / "request.json", {"schema": "eva.automedbench-codex-request.v1",
                "phase_intent": phase, "logical_input": canonical_value(_logical_input(state["options"], value)),
                "base_instructions": BASE, "developer_instructions": DEVELOPER,
                "tool_catalog_version": CATALOG_VERSION, "public_tool_catalog": list(TOOLS)})
            snapshot(state["workspace"], target / "before")
            try:
                receipt = await asyncio.wait_for(runtime.run_turn(state["handle"], value), args.turn_timeout)
                verify_codex_turn_receipt(receipt)
                # Keep the exact receipt schema; do not add fields inside its commitment.
                with (target / "receipt.json").open("xb") as stream:
                    stream.write(canonical_json_bytes(receipt))
                (target / "receipt.json").chmod(0o600)
                state["receipts"].append(receipt)
                print(json.dumps({"case_id": state["case"]["case_id"], "phase": phase,
                    "status": receipt.status, "actual_tool_calls": len(receipt.tool_calls)}), flush=True)
                if receipt.status != "completed":
                    state["errors"].append({"phase": phase, "error": "codex_turn_not_completed"})
            except Exception as exc:
                error = {"phase": phase, "error": type(exc).__name__, "reward": None}
                state["errors"].append(error)
                write_once(target / "failure.json", error)
                print(json.dumps({"case_id": state["case"]["case_id"], **error}), flush=True)
            finally:
                snapshot(state["workspace"], target / "after")

        async def first_two(runtime, state):
            try:
                state["handle"] = await runtime.start_thread(state["options"])
                state["thread_id"] = state["handle"].thread_id
                for index in (0, 1):
                    await one_turn(runtime, state, index)
                    if state["errors"]:
                        break
            except Exception as exc:
                state["errors"].append({"phase": "thread_start", "error": type(exc).__name__})

        backend = setup.backend()
        async with CodexRuntime(backend) as runtime:
            process = app_process(backend)
            first_pid = process.pid
            await asyncio.gather(*(first_two(runtime, state) for state in states))
        stopped = process.poll() is not None
        backend = setup.backend()
        async with CodexRuntime(backend) as runtime:
            second_pid = app_process(backend).pid
            async def review(state):
                if state["errors"]:
                    return
                try:
                    state["handle"] = await runtime.resume_thread(state["thread_id"], state["options"])
                    await one_turn(runtime, state, 2)
                except Exception as exc:
                    state["errors"].append({"phase": "thread_resume", "error": type(exc).__name__})
            await asyncio.gather(*(review(state) for state in states))

    assets = load_assets(args.assets_root)
    summaries = []
    for state in states:
        receipts = state["receipts"]
        completed = len(receipts) == 3 and not state["errors"]
        summary = {"schema": "eva.automedbench-codex-case-rollout.v1", "case_id": state["case"]["case_id"],
            "run_id": manifest["run_id"], "thread_id": state.get("thread_id"), "actual_turn_count": len(receipts),
            "turn_receipt_blake3s": [r.receipt_blake3 for r in receipts], "completed": completed,
            "requested_model": setup.model, "returned_model": None, "provider_backend": "local_qwen_responses_adapter",
            "source_checkpoint": binding["canary"]["exact_final_model_path"], "errors": state["errors"],
            "app_server_pids": [first_pid, second_pid], "first_app_server_stopped": stopped,
            "actual_restart": stopped and first_pid != second_pid,
            "same_thread_resumed": len(receipts) == 3 and receipts[2].thread_resumed
                and len({r.thread_id for r in receipts}) == 1,
            "phase_names_assert_rubric_success": False, "agent_judged": False,
            "native_score_available_to_actor": False, "canonical_evamed_tool_profile": False}
        write_once(state["audit"] / "rollout.json", summary)
        if completed:
            try:
                native = await asyncio.to_thread(score_case, assets, run, state["case"]["case_id"], args.public_python.absolute())
                summary["native_score_status"] = native["native_result"]["status"]
            except Exception as exc:
                summary["native_score_error"] = type(exc).__name__
        summaries.append(summary)
    result = write_once(root / "summary.json", {"schema": "eva.automedbench-codex-evaluation-summary.v1",
        "unique_cases": 3, "completed_rollouts": sum(row["completed"] for row in summaries),
        "cases": summaries, "one_attempt_per_case": True, "no_missing_reward_fallback": True})
    print(json.dumps({"run_root": str(run), "completed_rollouts": result["completed_rollouts"]}), flush=True)
