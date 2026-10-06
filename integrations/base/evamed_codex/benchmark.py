"""Real Codex medical benchmark driver; native metrics remain a separate step."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from urllib.parse import urlsplit
from uuid import uuid4

from training.automedbench_lite.adapter import canonical, file_digest, read_document, write_once
from training.automedbench_lite.track_adapter import BY_TRACK, PUBLIC_CONFIG_KEYS, TrackRelease, public_output_contract

from .benchmark_assets import prepare_release
from .benchmark_provider import setup_provider
from .benchmark_tools import catalog, PYTHON_TOOL


ROOT = Path(__file__).resolve().parents[3]
HARNESS = "evamed-codex-v1.3-supra"
MAX_TURNS, MAX_SECONDS, CONTEXT = 100, 3600, 262144
CODEX = ROOT / "tools/codex-0.153.4/package/vendor/x86_64-unknown-linux-musl/bin/codex"


def defaults() -> dict:
    return {"purpose": "benchmark", "candidate": True, "harness": HARNESS, "runner": "codex", "mode": "think",
            "route": "local", "model": "Qwen3.8-27B", "endpoint": "http://127.0.0.1:30000/v1",
            "max_turns": MAX_TURNS, "max_seconds": MAX_SECONDS, "context_length": CONTEXT,
            "compact_at_tokens": int(CONTEXT * .72), "max_output_tokens": 32768,
            "thinking_budget_tokens": 24576, "final_answer_reserve_tokens": 8192,
            "thinking_logit_processor": None, "thinking_processor_receipt": str(ROOT / "evamed-codex/receipts/qwen35-thinking-processor.json"),
            "server_pid": None, "task_gpu": "7",
            "codex_bin": str(CODEX), "bwrap": str(ROOT / "tools/enroot/usr/bin/bwrap"),
            "scientific_root": str(ROOT / ".enroot/data/evamed-slime-v0.3.2"),
            "scientific_python": "/usr/bin/python3.12",
            "runtime_manifest": str(ROOT / "evamed-codex/receipts/automedbench-public-models/preparation.json"),
            "task_python": str(ROOT / "evamed-codex/scripts/training-python"),
            "mcp_python": str(ROOT / ".venvs/assets/bin/python")}


def config_for(args) -> dict:
    config = defaults()
    endpoint_env = os.environ.get("AUTOMEDBENCH_AGENT_BASE_URL")
    model_env = os.environ.get("AUTOMEDBENCH_AGENT_MODEL")
    if endpoint_env:
        config["endpoint"] = endpoint_env.rstrip("/")
        config["route"] = "local" if urlsplit(endpoint_env).hostname == "127.0.0.1" else "api"
    if model_env:
        config["model"] = model_env
    if getattr(args, "config", None):
        config.update(json.loads(args.config.read_text()))
    for key in ("model", "endpoint", "route", "mode", "server_pid", "task_gpu"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if getattr(args, "agent", None) == "codex-default":
        config.update(candidate=False, harness="codex-default", runner="codex-default")
    elif getattr(args, "agent", None) in {HARNESS, "evamed-codex-latest"}:
        config.update(candidate=True, harness=HARNESS, runner="codex")
    elif getattr(args, "agent", None) is not None:
        raise ValueError("unrecognized_harness_identity")
    if endpoint_env and config["endpoint"] != endpoint_env.rstrip("/"):
        raise ValueError("workflow_endpoint_binding_differs")
    if model_env and config["model"] != model_env:
        raise ValueError("workflow_model_binding_differs")
    if config["max_turns"] != MAX_TURNS or config["max_seconds"] != MAX_SECONDS or config["context_length"] != CONTEXT:
        raise ValueError("benchmark_requires_100_requests_3600_seconds_262144_context")
    if config["max_output_tokens"] != 32768 or config["mode"] not in {"instant", "think"}:
        raise ValueError("benchmark_mode_or_output_budget_differs")
    endpoint = urlsplit(config["endpoint"])
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment or endpoint.path != "/v1":
        raise ValueError("unsafe_provider_endpoint")
    if config["route"] == "local":
        if endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1" or not endpoint.port:
            raise ValueError("local_route_requires_explicit_loopback")
    elif config["route"] != "api" or endpoint.scheme != "https":
        raise ValueError("invalid_route_kind")
    if config["route"] == "local" and config["mode"] == "think" and not config.get("thinking_logit_processor"):
        processor_receipt = Path(config["thinking_processor_receipt"])
        if processor_receipt.is_file():
            processor = json.loads(processor_receipt.read_text())
            expected_source = ROOT / "sglang/python/sglang/srt/sampling/custom_logit_processor.py"
            if (processor.get("custom_params") != {"thinking_budget": 24575}
                    or processor.get("max_tokens") != 32768
                    or processor.get("reasoning_phase_limit_including_closing_delimiter") != 24576
                    or hashlib.sha256(expected_source.read_bytes()).hexdigest() != processor.get("source_sha256")):
                raise ValueError("thinking_processor_contract_or_source_differs")
            config["thinking_logit_processor"] = processor["custom_logit_processor"]
    if "qwen" not in config["model"].lower():
        raise ValueError("this_runner_route_is_qwen_specific")
    return config


def source_manifest(config: dict, output: Path) -> dict:
    from blake3 import blake3
    paths = []
    for tree in (ROOT / "evamed-codex/src/evamed_codex", ROOT / "EVA-Agent/training/automedbench_lite",
                 ROOT / "EVA-Agent/training/benchmark_models", ROOT / "EVA-Harness/src/eva_agent/codex_runtime",
                 ROOT / "EVA-Harness/src/eva_agent/codex_providers", ROOT / "EVA-Harness/plugins/evamed-codex"):
        paths.extend(path for path in tree.rglob("*") if path.is_file() and path.suffix in {".py", ".md", ".json", ".toml"}
                     and path.name not in {"benchmark_judge.py", "benchmark_partial_judge.py", "benchmark_report.py"}
                     and not (path.parent == ROOT / "evamed-codex/src/evamed_codex"
                              and path.name.startswith(("portable_", "slime_"))))
    paths.extend(ROOT / "evamed-codex/scripts" / name for name in (
        "automed-codex-track.py", "training-python", "run-training-container.sh", "python-container-rc.sh"))
    sources = [{"path": str(path.relative_to(ROOT)), "blake3": file_digest(path)} for path in sorted(set(paths))]
    for row in sources:
        retained = output.parent / "frozen-sources" / row["path"]
        retained.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / row["path"], retained)
        retained.chmod(0o400)
        if file_digest(retained) != row["blake3"]:
            raise ValueError("source_changed_during_freeze")
    model_name = "Qwen3.8-27B" if "27" in config["model"] else "Qwen3.5-9B"
    model_receipt_path = ROOT / "evamed-codex/receipts" / (model_name + "-download.json")
    model_receipt = json.loads(model_receipt_path.read_text())
    return write_once(output, {"schema": "eva.evamed-codex-benchmark-manifest.v1", "harness": config["harness"],
        "runner": config["runner"], "codex_version": "0.153.4", "codex_binary_blake3": file_digest(Path(config["codex_bin"])),
        "sources": sources, "config": config, "tool_catalog": catalog(config["candidate"]),
        "model_download_receipt_blake3": file_digest(model_receipt_path),
        "model_source": {key: model_receipt.get(key) for key in ("repository", "revision", "bytes", "file_count")},
        "canonical_skill_manifest_blake3": file_digest(ROOT / "EVA-Harness/plugins/evamed-codex/references/legacy-skill-manifest.v1.json"),
        "tool_catalog_blake3": blake3(canonical(catalog(config["candidate"]))).hexdigest(),
        "canonical_tool_changes": False, "removed_docker_executor": "automed_execute_python",
        "added_scientific_executor": PYTHON_TOOL["name"], "evaluation_only": True})


def preflight(config: dict, *, server: bool = True) -> dict:
    from training.automedbench_lite.skill_surface import VerifiedEvaluationSkills
    from training.automedbench_lite.adapter import REVISION

    version = subprocess.check_output([config["codex_bin"], "--version"], text=True, timeout=15).strip()
    if version != "codex-cli 0.153.4":
        raise ValueError("pinned_codex_0_153_4_required")
    receipt = ROOT / "datasets/automedbench-codex-8928073/download-all3.json"
    release = TrackRelease(receipt)
    counts = {name: len(release.case_inputs(track)[0]) for name, track in BY_TRACK.items()}
    manifest = json.loads(Path(config["runtime_manifest"]).read_text())
    if manifest.get("status") != "complete":
        raise ValueError("prescribed_analysis_models_incomplete")
    if config["candidate"]:
        skill_root = ROOT / "evamed-codex/receipts/benchmark-skill-preflight"
        skill_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        skill_root.chmod(0o700)
        skills = VerifiedEvaluationSkills(skill_root)
        skill_count = len(skills.inventory)
    else:
        skill_count = 0
    runtime = Path(config["scientific_root"])
    if not (runtime / ".evamed-rootfs-ready").is_file():
        raise ValueError("scientific_runtime_not_ready")
    if not Path(config["bwrap"]).is_file() or not (runtime / config["scientific_python"].lstrip("/")).exists():
        raise ValueError("scientific_sandbox_runtime_missing")
    serving = {"verified": False, "route": config["route"]}
    if server and config["route"] == "local":
        import httpx
        pid = config.get("server_pid")
        if type(pid) is not int or pid <= 0:
            raise ValueError("local_server_pid_binding_required")
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        def flag(name):
            return argv[argv.index(name) + 1] if name in argv else None
        model = ROOT / ("models/Qwen3.8-27B" if "27" in config["model"] else "models/Qwen3.5-9B")
        if flag("--model-path") != str(model) or flag("--context-length") != str(CONTEXT):
            raise ValueError("local_server_model_or_context_mismatch")
        with httpx.Client(timeout=30, trust_env=False) as client:
            result = client.get(config["endpoint"].removesuffix("/v1") + "/get_model_info")
        info = result.json()
        if result.status_code != 200 or info.get("model_path") != str(model):
            raise ValueError("local_server_model_probe_failed")
        serving = {"verified": True, "server_pid": pid, "model_path": str(model),
                   "context_length": CONTEXT, "route": "local"}
        if config["mode"] == "think":
            if not config.get("thinking_logit_processor"):
                raise ValueError("manual_thinking_processor_required")
            if "--enable-custom-logit-processor" not in argv or "--disable-overlap-schedule" not in argv:
                raise ValueError("manual_budget_requires_custom_processor_without_overlap")
    return {"schema": "eva.codex-benchmark-preflight.v1", "codex_version": version,
            "revision": REVISION, "case_counts": counts, "canonical_legacy_skills": skill_count,
            "serving": serving, "live_tool_roundtrip_proven_by_preflight": False}


def prepare(config: dict, output: Path, tracks: list[str]) -> Path:
    import yaml
    from training.automedbench_lite.adapter import REVISION

    if not output.resolve().is_relative_to(ROOT):
        raise ValueError("benchmark_output_must_remain_in_workspace")
    receipt = prepare_release(ROOT, ROOT / "datasets/automedbench-codex-8928073")
    release = TrackRelease(receipt)
    run = output.absolute() / str(uuid4())
    run.mkdir(parents=True, mode=0o700)
    rows = []
    for name in tracks:
        track = BY_TRACK[name]
        cases, inputs = release.case_inputs(track)
        workspace = run / "actors" / str(uuid4())
        workspace.mkdir(parents=True, mode=0o700)
        for name in ("outputs/agents_outputs", "notes", "code", "public-guidance"):
            (workspace / name).mkdir(parents=True, mode=0o700)
        for row in inputs:
            target = workspace / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(release.public_file(row["source_path"]), target)
            target.chmod(0o400)
        task_config = yaml.safe_load(release.public_file(track.task_prefix + "config.yaml").read_bytes())
        public = {key: task_config[key] for key in PUBLIC_CONFIG_KEYS if key in task_config}
        if "tissue_labels" in public:
            public["tissue_labels"] = {str(key): value for key, value in public["tissue_labels"].items()}
        guidance = []
        for name in ("model_info.yaml", "lite_s1.md", "lite_s2.md", "lite_s3.md"):
            source = release.public_file(track.task_prefix + name)
            target = workspace / "public-guidance" / name
            shutil.copyfile(source, target)
            target.chmod(0o400)
            guidance.append({"path": "public-guidance/" + name, "source_path": track.task_prefix + name,
                             "bytes": source.stat().st_size, "blake3": file_digest(source)})
        model_info = yaml.safe_load((workspace / "public-guidance/model_info.yaml").read_bytes())
        write_once(workspace / "task.json", {"schema": "eva.automedbench-public-track-task.v1", "track": track.name,
            "task_id": track.task, "release_revision": REVISION, "tier": "lite", "case_ids": cases,
            "expected_case_count": track.count, "one_coding_workflow_for_full_subset": True,
            "input_root": "inputs", "public_config": public, "prescribed_analysis_model": model_info.get("lite"),
            "coding_orchestrator": config["model"] + " through Codex", "analysis_model_is_not_orchestrator": True,
            "output_contract": public_output_contract(track), "source_guidance": guidance, "license": track.license,
            "no_training_or_finetuning": True, "no_private_reference_access": True})
        write_once(workspace / "inputs-manifest.json", {"schema": "eva.automedbench-public-track-inputs.v1",
            "track": track.name, "case_ids": cases, "files": inputs, "all_inputs_readonly_mount_required": True})
        rows.append({"track": track.name, "case_count": track.count, "workspace_relative": str(workspace.relative_to(run)),
            "source_acquisition_document_blake3": release.document["document_blake3"], "source_receipt_path": str(receipt),
            "task_file_blake3": file_digest(workspace / "task.json"),
            "input_manifest_file_blake3": file_digest(workspace / "inputs-manifest.json")})
    write_once(run / "track-run-manifest.json", {"schema": "eva.automedbench-seven-track-run.v1", "run_id": run.name,
        "release_revision": REVISION, "tracks": rows, "planned_coding_rollouts": len(rows),
        "planned_case_outputs": sum(row["case_count"] for row in rows), "repeats": 1,
        "official_five_repeat_leaderboard": False, "private_reference_contents_copied_to_actor": False})
    source_manifest(config, run / "harness-manifest.json")
    return run


async def run_track(config: dict, run: Path, track: str) -> dict:
    from eva_agent.codex_runtime import CodexRuntime, CodexThreadOptions, CodexTurnInput, CodexToolOffer, CodexRole, CodexSandbox
    from eva_agent.codex_runtime.runtime import _logical_input
    from eva_agent.codex_runtime.research_memory import ResearchContextPolicy
    from eva_agent.codex_runtime.supra import SupraProfile, SupraMode
    from eva_agent.pipeline.digests import canonical_value
    from training.automedbench_lite.track_tools import MutableInventory
    from training.automedbench_lite.track_actor import snapshot
    from training.automedbench_lite.track_budget import TrackDeadline
    from training.automedbench_lite.policy_capture import require_joined_host_results
    from eva_agent.codex_runtime import verify_codex_turn_receipt
    from eva_agent.codex_runtime.policy_budget import CodexPolicyBudgetExceeded
    from .benchmark_cleanup import cleanup_owned_jobs
    from .benchmark_deadline import CaptureSigner, DeadlineCaptureBackend, verify_capture
    from .benchmark_partial_admission import finalize_partial_host
    from eva_agent.pipeline.digests import blake3_hex

    if not run.resolve().is_relative_to(ROOT):
        raise ValueError("benchmark_run_must_remain_in_workspace")
    preflight(config)
    manifest = read_document(run / "track-run-manifest.json")
    frozen = read_document(run / "harness-manifest.json", maximum=16 * 1024**2)
    if frozen["config"] != config:
        raise ValueError("launch_configuration_differs_from_frozen_manifest")
    for source in frozen["sources"]:
        if file_digest(ROOT / source["path"]) != source["blake3"]:
            raise ValueError("source_changed_after_frozen_manifest:" + source["path"])
    row = next(row for row in manifest["tracks"] if row["track"] == track)
    workspace = run / row["workspace_relative"]
    if file_digest(workspace / "task.json") != row["task_file_blake3"]:
        raise ValueError("prepared_task_changed")
    audit = run / "track-rollouts" / track
    audit.mkdir(parents=True, mode=0o700)
    if list((workspace / "notes").iterdir()) or (audit / "rollout.json").exists():
        raise ValueError("fresh_track_attempt_required")
    diagnostic = config.get("purpose") == "infrastructure_smoke"
    deadline = TrackDeadline(audit, seconds=MAX_SECONDS)
    config = {**config, "track_deadline_monotonic": deadline.deadline,
              "task_gpu": config.get("task_gpu_by_track", {}).get(track, config["task_gpu"])}
    # Original task-tool workers default to GPU0. Bind their single Python shim
    # to the separately selected task GPU so local Qwen serving stays isolated.
    task_python = audit / "task-python"
    task_python.write_text("#!/bin/sh\nexport CUDA_VISIBLE_DEVICES=" + shlex.quote(config["task_gpu"]) +
                           "\nexec " + shlex.quote(config["task_python"]) + ' "$@" --track-deadline-monotonic ' +
                           shlex.quote(str(deadline.deadline)) + '\n')
    task_python.chmod(0o700)
    tool_config = {**config, "workspace": str(workspace), "audit_root": str(audit), "task_python": str(task_python)}
    tool_config_path = audit / "tool-config.json"
    tool_config_path.write_text(json.dumps(tool_config))
    tool_config_path.chmod(0o600)
    inventory = MutableInventory(workspace, audit)
    target = audit / "turns/01-e2e"
    target.mkdir(parents=True, mode=0o700)
    snapshot(inventory, target / "before")
    token = os.environ.get("AUTOMEDBENCH_AGENT_API_KEY", "local-nonsecret" if config["route"] == "local" else "")
    if not token:
        raise ValueError("api_route_credential_missing")
    errors, receipt, terminal, host_join = [], None, None, None
    capture = signer = capture_prelaunch = partial_rejection = partial_proof = None
    started = time.monotonic()
    try:
        with setup_provider(config, audit / "provider", token) as (backend, provider, thread, meter):
            offers = tuple(CodexToolOffer(fully_qualified_name="automed_eval/" + tool["name"],
                description=tool["description"], input_schema=tool["inputSchema"], parallel_safe=False,
                read_only=tool["name"] in {"automed_read_file", "automed_view_input", "search_skills", "load_skill"},
                allowed_stages=("E2E",)) for tool in catalog(config["candidate"]))
            names = [tool["name"] for tool in catalog(config["candidate"])]
            thread["mcp_servers"] = {"automed_eval": {"command": config["mcp_python"],
                "args": ["-I", "-B", str(ROOT / "evamed-codex/scripts/automed-codex-track.py"),
                         "serve", "--config", str(tool_config_path)], "cwd": str(workspace),
                "required": True, "startup_timeout_sec": 60, "tool_timeout_sec": 200,
                "enabled_tools": names, "omit_tools_from": ["deferred", "code_mode"],
                "tools": {name: {"approval_mode": "approve"} for name in names}}}
            base = "You are a coding agent completing one medical research benchmark. Use the offered tools to inspect public task files, write and execute code, and produce the required artifacts."
            developer = (
                "Complete every case in task.json using the prescribed analysis model. The Qwen backbone is the coding orchestrator. "
                "Author reusable Python using evamed_execute_python; GPU analysis uses the offered asynchronous model-job tools. "
                "Treat a queued or running job as pending and reuse its ID. Produce the exact native submission paths under "
                "outputs/agents_outputs. Validate coverage and format, then finish with a concise evidence-based status. "
                "This single workflow has 100 model requests including compaction and 3600 seconds, 160 tool calls and 32 Python executions. "
                "No private scores or references are visible. "
                "Canonical legacy skill discovery uses the current S1 planning, S2 setup, S3 pilot, S4 execution, or S5 validation stage; E2E has no legacy stage grants. "
            )
            options = CodexThreadOptions(role=CodexRole.STRONG_ACTOR, model=config["model"], provider=provider,
                cwd=str(workspace), sandbox=CodexSandbox.READ_ONLY, ephemeral=False, config=thread,
                offered_tools=offers, base_instructions=base, developer_instructions=developer)
            value = CodexTurnInput(public_text="Read task.json and the relevant public-guidance files. Complete this entire track, create and validate the required submission artifacts, and report the actual outcome.",
                                   model=config["model"], effort="xhigh" if config["mode"] == "think" else "none")
            if config["candidate"]:
                profile = SupraProfile(model=config["model"], provider=provider, mode=SupraMode(config["mode"]),
                    protocol="qwen_template", local_qwen_endpoint=config["endpoint"] if config["route"] == "local" else None,
                    context=ResearchContextPolicy(context_tokens=CONTEXT, output_tokens=32768, reserve_tokens=8192,
                        compact_at_tokens=config["compact_at_tokens"], compaction_headroom_mode="exact_request_guard"))
                options, value = profile.thread_options(options), profile.turn_input(value)
            if diagnostic:
                value = replace(value, public_text="Infrastructure smoke only: use automed_read_file to inspect task.json, if search_skills is offered, search query medical with stage S1 and load one returned skill_id with stage S1; the embedded summary_failures is a separate native skill. Then use evamed_execute_python to compute sum(range(10)) using NumPy and write exactly the three bytes 45 followed by a newline to notes/smoke.txt, without a label or explanation. Read the note back using automed_read_file and finish with SMOKE_OK. Do not launch any model jobs or solve the benchmark.")
            write_once(target / "request.json", {"schema": "eva.codex-benchmark-logical-request.v1",
                       "logical_input": canonical_value(_logical_input(options, value)), "task_manifest_blake3": row["task_file_blake3"],
                       "base_instructions": options.base_instructions,
                       "developer_instructions": options.developer_instructions,
                       "public_tool_catalog": catalog(config["candidate"])})
            capture_binding = {'run_id': manifest['run_id'], 'track': track,
                'source_revision': 'manifest:' + frozen['document_blake3'],
                'launch_file_blake3': file_digest(run / 'harness-manifest.json'),
                'codex_binary_blake3': frozen['codex_binary_blake3'],
                'logical_input_blake3': blake3_hex(_logical_input(options, value)),
                'tool_catalog_blake3': frozen['tool_catalog_blake3'],
                'deadline_monotonic': deadline.deadline, 'max_seconds': MAX_SECONDS}
            signer = CaptureSigner()
            capture_prelaunch = write_once(audit / 'native-capture-prelaunch.json', {
                'schema': 'eva.benchmark-native-capture-prelaunch.v1',
                'public_key_base64': signer.public_key_base64, 'binding': capture_binding,
                'capture_registered_before_actual_thread': True, 'private_signing_key_serialized': False})
            capture = DeadlineCaptureBackend(backend, audit_root=audit / 'native-capture',
                                             signer=signer, binding=capture_binding)
            async with CodexRuntime(capture) as runtime:
                handle = await runtime.start_thread(options)
                try:
                    receipt = await runtime.run_turn(handle, value, policy_timeout_seconds=min(deadline.remaining(), 180) if diagnostic else deadline.remaining())
                except CodexPolicyBudgetExceeded as exc:
                    receipt, terminal = exc.receipt, exc.outcome
                verify_codex_turn_receipt(receipt)
                (target / "receipt.json").write_bytes(canonical(canonical_value(receipt)))
                host_join = require_joined_host_results(receipt, audit)
            provider_count = meter.requests
            exhausted = meter.exhausted
    except Exception as exc:
        import traceback
        rejection = {"type": type(exc).__name__, "frames": [
            {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
            for frame in traceback.extract_tb(exc.__traceback__)]}
        if (type(exc).__name__ == 'CodexRuntimeError'
                and str(exc) == 'Codex turn completed with active tool calls'
                and capture is not None and capture.receipt_path is not None):
            # Preserve the ordinary receipt rejection. A separate signed native
            # partial proof, if fully admitted below, may establish a budget outcome.
            partial_rejection = {**rejection, 'reason': str(exc), 'ordinary_receipt_eligible': False}
        else:
            errors.append(rejection)
        provider_count = meter.requests if "meter" in locals() else 0
        exhausted = meter.exhausted if "meter" in locals() else None
    cleanup = await cleanup_owned_jobs(audit, workspace, trigger="track_finalization")
    after = snapshot(inventory, target / "after")
    final = snapshot(inventory, audit / "final")
    if (audit / "deadline-cleanup-failure.json").exists():
        errors.append({"type": "DeadlineCleanupFailed"})
    from .benchmark_admission import observed_infrastructure_failures
    infrastructure_failures = observed_infrastructure_failures(audit)
    if infrastructure_failures:
        errors.append({"type": "ObservedHostInfrastructureFailure", "count": len(infrastructure_failures)})
    hosts = (audit / "mcp-events.jsonl").read_bytes().splitlines() if (audit / "mcp-events.jsonl").exists() else []
    if hosts and not errors:
        from .benchmark_partial_admission import verify_host_publications
        try:
            verify_host_publications(audit, deadline.deadline, workspace)
        except Exception as exc:
            errors.append({'type': type(exc).__name__, 'scope': 'host_publication_admission', 'reason': str(exc)})
    if host_join is not None and host_join["joined_host_call_count"] != len(hosts):
        errors.append({"type": "UnjoinedHostEvents"})
    completed = receipt is not None and receipt.status == "completed" and not errors and terminal is None and exhausted is None
    disposition = "completed" if completed else "infrastructure-unknown"
    proof = None
    if not diagnostic and not errors and partial_rejection is not None:
        from .benchmark_admission import provider_evidence
        try:
            providers = provider_evidence(audit)
            sealed = finalize_partial_host(audit, capture_path=capture.receipt_path, signer=signer,
                prelaunch=capture_prelaunch, offered_names=tuple(tool.fully_qualified_name for tool in offers),
                cleanup=cleanup, after_snapshot=after, final_snapshot=final, workspace=workspace)
            partial_proof = write_once(audit / 'policy-partial-terminal.json', {
                'schema': 'eva.evamed-partial-policy-budget-terminal.v1', 'reason': 'wall_clock',
                'prelaunch_document_blake3': capture_prelaunch['document_blake3'],
                'host_seal_file_blake3': sealed['file_blake3'], 'provider_requests': providers,
                'ordinary_runtime_rejection': partial_rejection, 'ordinary_receipt_fabricated': False,
                'completed_requested_turns': False, 'native_scoring_only': True,
                'private_edits_not_published_after_deadline': True})
            disposition = 'policy-budget-exhausted'
        except Exception as exc:
            errors.append({'type': type(exc).__name__, 'scope': 'partial_terminal_evidence', 'reason': str(exc)})
    elif partial_rejection is not None:
        errors.append(partial_rejection)
    if not diagnostic and not errors and receipt is not None and (terminal is not None or exhausted == "model_requests"):
        from .benchmark_admission import provider_evidence
        try:
            providers = provider_evidence(audit)
            denial = audit / "provider/requests/budget-exhaustion.json"
            proof = write_once(audit / "policy-budget-terminal.json", {
                "schema": "eva.evamed-policy-budget-terminal.v1",
                "reason": "model_requests" if exhausted == "model_requests" else "wall_clock",
                "receipt_blake3": receipt.receipt_blake3, "actual_terminal_status": receipt.status,
                "policy_budget": terminal, "host_join": host_join,
                "provider_requests": providers,
                "provider_denial_blake3": read_document(denial)["document_blake3"] if denial.exists() else None,
                "track_budget": deadline.observation(), "workspace_quiescence_verified": True,
                "infrastructure_error": None, "reward": None,
                "cleanup_document_blake3": cleanup["document_blake3"],
                "after_snapshot_blake3": after["document_blake3"], "final_snapshot_blake3": final["document_blake3"]})
            disposition = "policy-budget-exhausted"
        except Exception as exc:
            errors.append({"type": type(exc).__name__, "scope": "policy_terminal_evidence"})
    result = write_once(audit / "rollout.json", {"schema": "eva.automedbench-codex-track-rollout.v1",
        "purpose": config.get("purpose", "benchmark"), "score_admissible": not diagnostic,
        "track": track, "run_id": manifest["run_id"], "harness": config["harness"], "runner": config["runner"],
        "requested_model": config["model"], "route": config["route"], "completed_requested_turns": completed,
        "terminal_disposition": disposition,
        "policy_budget_terminal_blake3": proof["document_blake3"] if proof else None,
        'partial_policy_terminal_blake3': partial_proof['document_blake3'] if partial_proof else None,
        'ordinary_runtime_rejection': partial_rejection,
        'actual_partial_native_turn_count': int(partial_proof is not None),
        "actual_turn_count": int(receipt is not None), "actual_model_request_count": provider_count,
        "max_model_requests": MAX_TURNS, "errors": errors, "budget_exhausted": exhausted,
        "infrastructure_failures": infrastructure_failures,
        "turn_receipt_blake3s": [receipt.receipt_blake3] if receipt else [],
        "wall_seconds": time.monotonic() - started, "max_seconds": MAX_SECONDS,
        "cleanup_document_blake3": cleanup["document_blake3"], "final_snapshot_blake3": final["document_blake3"],
        "native_score_completed": False, "agent_judged": False, "official_five_repeat_leaderboard": False})
    if disposition == "policy-budget-exhausted":
        from .benchmark_admission import require_scoring_admission
        require_scoring_admission(run, track, result)
    return result


def validate_smoke(run: Path, track: str, *, candidate: bool) -> dict:
    """Check actual tool results and bytes, independently of the model's final claim."""
    manifest = read_document(run / "track-run-manifest.json")
    row = next(row for row in manifest["tracks"] if row["track"] == track)
    workspace = run / row["workspace_relative"]
    audit = run / "track-rollouts" / track
    rollout = read_document(audit / "rollout.json")
    if rollout.get("purpose") != "infrastructure_smoke" or rollout.get("score_admissible") is not False:
        raise ValueError("smoke_scope_invalid")
    events = [json.loads(line) for line in (audit / "mcp-events.jsonl").read_text().splitlines()] if (audit / "mcp-events.jsonl").exists() else []
    receipt = json.loads((audit / "turns/01-e2e/receipt.json").read_text()) if (audit / "turns/01-e2e/receipt.json").exists() else {}
    names = {event["name"] for event in events if not event["is_error"]}
    output = workspace / "notes/smoke.txt"
    checks = {
        "actual_codex_completed": rollout["completed_requested_turns"],
        "actual_isolated_python": any(event["name"] == "evamed_execute_python" and event["result"].get("exit_code") == 0 for event in events),
        "exact_artifact_bytes": output.is_file() and output.read_bytes() == b"45\n",
        "artifact_read_back": any(event["name"] == "automed_read_file" and event["arguments"].get("path") == "notes/smoke.txt" and not event["is_error"] for event in events),
        "final_marker_present": "SMOKE_OK" in (receipt.get("final_response") or ""),
        "no_model_jobs": not list((audit / "model-jobs").glob("*/submission.json")),
    }
    if candidate:
        checks["canonical_skill_search_and_load"] = {"search_skills", "load_skill"} <= names
    return write_once(audit / "smoke-admission.json", {
        "schema": "eva.codex-benchmark-smoke-admission.v1", "checks": checks, "passed": all(checks.values()),
        "rollout_document_blake3": rollout["document_blake3"],
        "actual_turn_receipt_blake3": receipt.get("receipt_blake3"),
        "failed_host_tools": [event["name"] for event in events if event["is_error"]],
        "benchmark_score_admissible": False,
    })
