"""Actual three-stage Qwen/Codex diagnostic over public synthetic CSV files.

The initial comma parse really fails on a semicolon file. No host-authored
corrective memory is created. Only later actual note reads and correct operations
can establish recovery; this is neither a medical benchmark nor training data.
"""
from __future__ import annotations

import asyncio
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import RLock
from uuid import uuid4

from eva_agent.codex_runtime import (CodexRole, CodexRuntime, CodexSandbox,
    CodexThreadOptions, CodexToolOffer, CodexTurnInput, OpenAICodexBackend, verify_codex_turn_receipt)
from eva_agent.codex_runtime.policy_budget import CodexPolicyBudgetExceeded
from eva_agent.codex_runtime.research_memory import (memory_launch_options, memory_thread_options,
    memory_turn_input, inspect_memory_profile, SKILL_PATH)
from eva_agent.codex_pipeline.turn_mcp import TurnMCPBridgeFactory
from eva_agent.pipeline.digests import blake3_bytes, canonical_json_bytes, canonical_value
from eva_agent.pipeline.workspace import FilesystemSandbox, SandboxError
from training.automedbench_lite.local_qwen import local_qwen_setup
from .probe import write_once, _app_server_process, checkpoint_binding

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("pilot.csv", "batch-two.csv", "batch-three.csv")
ANSWER = {"type": "object", "properties": {"input": {"type": "string"}, "total": {"type": "integer"},
    "artifact": {"type": "string"}}, "required": ["input", "total", "artifact"], "additionalProperties": False}


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


class FailureFixture:
    def __init__(self, output: Path):
        self.output, self.phase, self.calls, self.lock = output, 1, [], RLock()
        seed = int(uuid4().hex[:5], 16)
        self.expected = {name: (seed + i * 11) + (13 + i) for i, name in enumerate(NAMES)}
        files = {"inputs/" + name: f"batch;count\na;{seed + i * 11}\nb;{13 + i}\n".encode()
                 for i, name in enumerate(NAMES)}
        self.workspace = FilesystemSandbox(output / "workspaces", str(uuid4()), files)
        self.offered = self.offers()

    def offers(self):
        definitions = (
            ("read_public_file", "Read one public input, saved note or output file. Paths are workspace-relative.",
             object_schema({"path": {"type": "string", "enum": [*("inputs/" + n for n in NAMES),
                "notes/task-state.md", "notes/failures.md", *("outputs/" + n + ".json" for n in NAMES)]}})),
            ("analyze_table", "Parse a public CSV using the explicit delimiter; sum its count column. "
             "Parsing failures are retained. Successful output is saved in outputs/NAME.json.",
             object_schema({"name": {"type": "string", "enum": list(NAMES)},
                            "delimiter": {"type": "string", "enum": [",", ";", "\\t"]}})),
            ("write_note", "Write a public task note using name, not path. Preserve evidence references and uncertainty.",
             object_schema({"name": {"type": "string", "enum": ["task-state.md", "failures.md"]},
                            "content": {"type": "string", "minLength": 1, "maxLength": 5000}})),
        )
        return tuple(CodexToolOffer("memory_failure/" + name, description, schema,
            read_only=name == "read_public_file", parallel_safe=False) for name, description, schema in definitions)

    def execute_group(self, calls):
        from jsonschema import Draft202012Validator
        schemas = {offer.name: canonical_value(offer.input_schema) for offer in self.offered}
        results = []
        with self.lock:
            for name, arguments in calls:
                name = name.rsplit("/", 1)[-1]
                Draft202012Validator(schemas[name]).validate(arguments)
                event_id, status = str(uuid4()), "completed"
                try:
                    if name == "read_public_file":
                        data = self.workspace.read_bytes(arguments["path"])
                        result = {"path": arguments["path"], "text": data.decode(), "bytes": len(data),
                                  "content_blake3": blake3_bytes(data)}
                    elif name == "write_note":
                        path = "notes/" + arguments["name"]
                        self.workspace.write_bytes(path, arguments["content"].encode())
                        result = {"path": path, "saved": True, "content_blake3": blake3_bytes(arguments["content"].encode())}
                    else:
                        text = self.workspace.read_bytes("inputs/" + arguments["name"]).decode()
                        reader = csv.DictReader(io.StringIO(text), delimiter=arguments["delimiter"].replace("\\t", "\t"))
                        if reader.fieldnames != ["batch", "count"]:
                            raise ValueError("csv_columns_do_not_match")
                        values = list(reader)
                        total = sum(int(row["count"]) for row in values)
                        artifact = "outputs/" + arguments["name"] + ".json"
                        result = {"input": arguments["name"], "total": total, "rows": len(values), "artifact": artifact}
                        self.workspace.write_bytes(artifact, canonical_json_bytes(result))
                except (ValueError, KeyError, FileNotFoundError, SandboxError) as error:
                    status = "failed"
                    result = {"error_type": type(error).__name__, "error": "public_file_or_csv_parse_failed",
                              "expected_columns": ["batch", "count"], "operation_succeeded": False}
                response = {"event_id": event_id, "status": status, "result": result}
                row = {"phase": self.phase, "name": name, "arguments": dict(arguments), "response": response}
                self.calls.append(row)
                write_once(self.output / "events" / f"{len(self.calls):04d}-{event_id}.json", row)
                results.append(response)
                print(json.dumps({"event": "actual_tool", "phase": self.phase, "name": name,
                                  "status": status, "event_id": event_id}), flush=True)
        return results


def verify_behavior(fixture, receipts, process_ids, old_stopped):
    checks = {}
    for receipt in receipts: verify_codex_turn_receipt(receipt)
    checks["three_completed_stages"] = len(receipts) == 3 and all(r.status == "completed" for r in receipts)
    checks["same_thread_actually_resumed_after_process_restart"] = len(receipts) >= 2 and (
        receipts[0].thread_id == receipts[1].thread_id and receipts[1].thread_resumed
        and len(process_ids) == 2 and process_ids[0] != process_ids[1] and old_stopped)
    checks["third_stage_has_fresh_thread"] = len(receipts) == 3 and receipts[2].thread_id != receipts[0].thread_id
    checks["actual_skill_selected_each_stage"] = len(receipts) == 3 and all(
        "summary_failures" in r.selected_skill_ids for r in receipts)
    host_by_id = {row["response"]["event_id"]: row for row in fixture.calls}
    joined = set()
    for receipt in receipts:
        for call in receipt.tool_calls:
            if call.mcp_server != "memory_failure":
                continue  # Native control-plane calls remain separately retained.
            structured = canonical_value(call.output).get("result", {}).get("structuredContent", {})
            event_id = structured.get("event_id")
            if event_id not in host_by_id or event_id in joined:
                raise ValueError("actual_host_receipt_event_binding_differs")
            host = host_by_id[event_id]
            if (host["response"] != structured or host["name"] != call.mcp_tool
                    or host["arguments"] != canonical_value(call.arguments)):
                raise ValueError("actual_host_receipt_payload_differs")
            joined.add(event_id)
    checks["actual_host_tool_results_bound_to_receipts"] = joined == set(host_by_id)
    failures = [r for r in fixture.calls if r["phase"] == 1 and r["name"] == "analyze_table"
                and r["arguments"]["delimiter"] == "," and r["response"]["status"] == "failed"]
    checks["actual_initial_parser_error"] = bool(failures)
    notes = [r for r in fixture.calls if r["phase"] == 1 and r["name"] == "write_note"
             and r["arguments"]["name"] == "failures.md"]
    checks["model_authored_failure_note_links_real_error"] = bool(failures) and any(
        failures[0]["response"]["event_id"] in row["arguments"]["content"] for row in notes)
    for phase, name in enumerate(NAMES, 1):
        rows = [r for r in fixture.calls if r["phase"] == phase]
        analyses = [r for r in rows if r["name"] == "analyze_table" and r["arguments"]["name"] == name]
        checks[f"stage_{phase}_actual_correct_analysis"] = any(r["response"]["status"] == "completed"
            and r["response"]["result"]["total"] == fixture.expected[name] for r in analyses)
        try: answer = json.loads(receipts[phase - 1].final_response or "null")
        except (ValueError, IndexError): answer = None
        checks[f"stage_{phase}_correct_final_answer"] = answer == {
            "input": name, "total": fixture.expected[name], "artifact": "outputs/" + name + ".json"}
        if phase > 1:
            first = next((i for i, row in enumerate(rows) if row in analyses), None)
            checks[f"stage_{phase}_reads_failure_note_before_analysis"] = first is not None and any(
                row["name"] == "read_public_file" and row["arguments"]["path"] == "notes/failures.md"
                and row["response"]["status"] == "completed" for row in rows[:first])
            checks[f"stage_{phase}_does_not_repeat_observed_error"] = bool(analyses) and all(
                row["arguments"]["delimiter"] != "," and row["response"]["status"] == "completed" for row in analyses)
    return {"passed": all(checks.values()), "checks": checks}


async def run(output: Path, identity_path: Path, server_pid: int, codex_bin: Path,
              *, timeout=180, drain=60):
    binding = checkpoint_binding(identity_path)
    argv = Path(f"/proc/{server_pid}/cmdline").read_bytes().split(b"\0")
    if binding["exact_final_model_path"].encode() not in argv:
        raise ValueError("serving_process_checkpoint_differs")
    version = subprocess.run([str(codex_bin), "--version"], check=True, capture_output=True, text=True).stdout.strip()
    if version != "codex-cli 0.153.4": raise ValueError("codex_version_differs")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    fixture = FailureFixture(output)
    prompts = [
        "Stage 1: run the public pilot.csv count-table analysis with the default comma delimiter first. "
        "If it fails, inspect the actual public input and correct the parser. Apply the summary_failures skill: "
        "write a compact failures.md with the observed event_id, confirmed/hypothesized cause, actual correction "
        "outcome and a preventive check. Save task-state.md. Never prewrite success. Once the analysis and notes "
        "are saved, stop calling tools and return only input,total,artifact JSON from the actual result.",
        "Stage 2: the app-server was restarted; continue this same task thread. Background unrelated log:\n" +
        "\n".join(f"batch-log-{i:03}: unrelated archive completed; no task instructions changed." for i in range(70)) +
        "\nRead the saved task-state and relevant failure note before acting. Analyze batch-two.csv. "
        "Use and verify the recorded preventive check rather than repeating an observed failure. Update task-state.md. "
        "After success stop calling tools and return only input,total,artifact JSON from the actual result.",
        "Stage 3: this is a NEW thread with no earlier conversation. Recover task progress from the saved "
        "task-state and failure notes using the offered tools; verify the relevant public input. Analyze "
        "batch-three.csv without repeating the recorded error. Update task-state.md, then stop calling tools "
        "and return only input,total,artifact JSON from the actual result.",
    ]
    write_once(output / "request-plan.json", {"schema": "eva.failure-memory-probe-plan.v1", "prompts": prompts,
        "server_pid": server_pid, "server_cmdline_blake3": blake3_bytes(b"\0".join(argv)),
        "checkpoint_identity": str(identity_path), "checkpoint_identity_blake3": blake3_bytes(identity_path.read_bytes()),
        "checkpoint_path": binding["exact_final_model_path"], "codex_binary": str(codex_bin), "codex_version": version,
        "skill": inspect_memory_profile(), "source_blake3": blake3_bytes(Path(__file__).read_bytes()),
        "policy_timeout_seconds": timeout, "interrupt_drain_seconds": drain, "concurrency": 1,
        "no_prefilled_memory": True, "medical_evaluation": False, "sft_rl_export": False})
    receipts, pids, outcomes = [], [], []
    with TemporaryDirectory(prefix="eva-failure-memory-mcp-", dir="/tmp") as temporary:
        with local_qwen_setup(run_root=output / "local-runtime", workers=1, thinking=True,
                exact_tool_schemas=True, normalize_priority_messages=True, max_output_tokens=4096,
                codex_bin=codex_bin, auto_compact_token_limit=12288, upstream_timeout_seconds=180) as setup:
            launch = memory_launch_options(setup.launch)
            options = memory_thread_options(CodexThreadOptions(role=CodexRole.WEAK_ACTOR,
                model=setup.model, provider=setup.provider, cwd=str(fixture.workspace.root),
                sandbox=CodexSandbox.READ_ONLY, ephemeral=False, config=setup.thread_config,
                offered_tools=fixture.offered, developer_instructions=
                "Use only the exact offered MCP function names and argument schemas. Public fixture only; "
                "no shell, subagents, resource URIs or web. Save public operational facts, never private reasoning. "
                "A successful operation need not be repeated. Use final JSON to finish each stage."))
            write_once(output / "effective-profile.json", {"setup": setup.safe_metadata,
                "launch_config_overrides": launch.config_overrides, "thread_config": canonical_value(options.config),
                "base_instructions": options.base_instructions, "developer_instructions": options.developer_instructions,
                "offered_tools": canonical_value(fixture.offered)})
            factory = TurnMCPBridgeFactory(proxy_python=Path(sys.executable).resolve(),
                proxy_script=ROOT / "src/eva_agent/codex_pipeline/turn_mcp_proxy.py",
                temp_root=Path(temporary), maximum_parallel_calls=1)

            async def stage(runtime, handle, index):
                fixture.phase = index + 1
                turn = memory_turn_input(CodexTurnInput(public_text=prompts[index], model=setup.model,
                    effort="low", summary="none", output_schema=ANSWER))
                write_once(output / f"stage-{index+1}-input.json", turn)
                try:
                    receipt = await runtime.run_turn(handle, turn, policy_timeout_seconds=timeout,
                                                     interruption_grace_seconds=drain)
                    outcome = {"controlled_budget_stop": False}
                except CodexPolicyBudgetExceeded as error:
                    receipt, outcome = error.receipt, error.outcome
                receipts.append(receipt)
                outcomes.append(outcome)
                write_once(output / f"stage-{index+1}-receipt.json", receipt)
                write_once(output / f"stage-{index+1}-outcome.json", outcome)
                write_once(output / f"stage-{index+1}-workspace.json", fixture.workspace.snapshot(f"phase-{index+1}"))
                print(json.dumps({"stage": index+1, "terminal_status": receipt.status,
                    "thread_id": receipt.thread_id, "resumed": receipt.thread_resumed,
                    "tool_calls": len(receipt.tool_calls)}), flush=True)

            with factory.open_actor(options, fixture) as bound:
                engine = OpenAICodexBackend(launch)
                async with CodexRuntime(engine) as runtime:
                    process = _app_server_process(engine)
                    pids.append(process.pid)
                    handle = await runtime.start_thread(bound)
                    thread_id = handle.thread_id
                    await stage(runtime, handle, 0)
                stopped = process.poll() is not None
                engine = OpenAICodexBackend(launch)
                async with CodexRuntime(engine) as runtime:
                    pids.append(_app_server_process(engine).pid)
                    handle = await runtime.resume_thread(thread_id, bound)
                    await stage(runtime, handle, 1)
                    handle = await runtime.start_thread(replace(bound, ephemeral=True))
                    await stage(runtime, handle, 2)
    from .compaction_observation import has_completed_compaction
    result = {"schema": "eva.failure-memory-probe-result.v1", **verify_behavior(fixture, receipts, pids, stopped),
        "app_server_pids": pids, "actual_tool_calls": len(fixture.calls), "stage_outcomes": outcomes,
        "requested_model": "Qwen/Qwen3.5-9B", "returned_model_identity": None,
        "checkpoint_identity_blake3": blake3_bytes(identity_path.read_bytes()), "model_weights_changed": False,
        "medical_evaluation": False, "sft_rl_export": False,
        "automatic_compaction_observed": has_completed_compaction(receipts),
        "bounded_distractor_is_not_compaction_proof": True, "skill_benefit_causally_established": False}
    write_once(output / "result.json", result)
    return result
