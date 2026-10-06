"""Real Codex multi-turn/restart probe using a small public synthetic workspace.

No existing medical tool, rubric, or policy is changed. These fixture-only MCP
tools are deliberately not a medical benchmark or an SFT/RL export source.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from threading import RLock
from types import SimpleNamespace
from uuid import uuid4

from eva_agent.codex_runtime import (
    CodexRole, CodexRuntime, CodexSandbox, CodexThreadOptions, CodexToolOffer,
    CodexTurnInput, OpenAICodexBackend, verify_codex_turn_receipt,
)
from eva_agent.codex_pipeline.turn_mcp import TurnMCPBridgeFactory
from eva_agent.deployment.campaign import CODEX_FIRST_RELEASE_THREAD_CONFIG
from eva_agent.pipeline.digests import canonical_json_bytes, canonical_value
from eva_agent.pipeline.workspace import FilesystemSandbox
from eva_agent.training.native_astra_teacher import (
    NATIVE_ASTRA_MODEL, NATIVE_ASTRA_PROVIDER, native_astra_launch_options,
    native_astra_model_catalog, native_astra_provider_config, private_native_auth_copy,
)

ROOT = Path(__file__).resolve().parents[2]
NOTE_SCHEMA = {"type": "object", "properties": {
    "threshold_bps": {"type": "integer", "minimum": 0, "maximum": 10000},
    "next_stage": {"type": "string", "enum": ["S3", "S4"]},
}, "required": ["threshold_bps", "next_stage"], "additionalProperties": False}
ANSWER_SCHEMA = {"type": "object", "properties": {
    **NOTE_SCHEMA["properties"], "observation_tag": {"type": ["string", "null"]},
}, "required": ["threshold_bps", "next_stage", "observation_tag"], "additionalProperties": False}
EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}


def write_once(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))
    path.chmod(0o600)


class MemoryFixture:
    """Only one note file is writable; the transient observation cannot enter it."""

    def __init__(self, output: Path):
        self.workspace = FilesystemSandbox(output / "workspaces", str(uuid4()), {})
        self.observation_tag = "observation-" + str(uuid4())
        self.phase = "learn"
        self.calls = []
        self.lock = RLock()

    def offers(self):
        return tuple(CodexToolOffer(
            fully_qualified_name="memory_fixture/" + name, description=description,
            input_schema=schema, read_only=name != "write_note", parallel_safe=False,
        ) for name, description, schema in (
            ("read_record", "Read the synthetic public pilot record, available only in the first phase.", EMPTY),
            ("read_note", "Read the persistent public project note from this workspace.", EMPTY),
            ("write_note", "Save only the public threshold and next stage in the persistent project note.", NOTE_SCHEMA),
        ))

    def execute_group(self, calls):
        from jsonschema import Draft202012Validator
        schemas = {offer.name: offer.input_schema for offer in self.offers()}
        results = []
        with self.lock:
            for name, arguments in calls:
                name = name.rsplit("/", 1)[-1]
                if name not in schemas:
                    raise ValueError("unoffered memory fixture tool")
                Draft202012Validator(canonical_value(schemas[name])).validate(arguments)
                if name == "read_record":
                    result = ({"threshold_bps": 7300, "next_stage": "S3",
                               "observation_tag": self.observation_tag}
                              if self.phase == "learn" else {"available": False})
                elif name == "read_note":
                    result = json.loads(self.workspace.read_bytes("project-note.json"))
                else:
                    self.workspace.write_bytes("project-note.json", canonical_json_bytes(arguments))
                    result = {"saved": True, "relative_path": "project-note.json"}
                row = {"phase": self.phase, "name": name, "arguments": dict(arguments), "result": result}
                self.calls.append(row)
                results.append(result)
            return results


def verify_probe(receipts, fixture: MemoryFixture, process_ids, stopped):
    """Behavioral checks: no passing claim from a thread/resume flag alone."""
    checks = {}
    answers = []
    for receipt in receipts:
        verify_codex_turn_receipt(receipt)
        try:
            answers.append(json.loads(receipt.final_response or "null"))
        except ValueError:
            answers.append(None)
    checks["four_completed_turns"] = len(receipts) == 4 and all(r.status == "completed" for r in receipts)
    if len(receipts) != 4:
        return {"passed": False, "checks": checks}
    checks["same_thread_context_then_restart_resume"] = (
        len({r.thread_id for r in receipts[:3]}) == 1
        and not receipts[0].thread_resumed and not receipts[1].thread_resumed
        and receipts[2].thread_resumed)
    checks["fresh_thread_for_workspace_memory"] = receipts[3].thread_id != receipts[0].thread_id
    checks["actual_app_server_process_replaced"] = (
        len(process_ids) == 2 and all(type(p) is int and p > 0 for p in process_ids)
        and process_ids[0] != process_ids[1] and stopped is True)
    for index, (threshold, stage, tag) in enumerate((
        (7300, "S3", fixture.observation_tag), (8100, "S4", fixture.observation_tag),
        (8100, "S4", fixture.observation_tag), (8100, "S4", None),
    )):
        checks[f"phase_{index+1}_correct_state"] = answers[index] == {
            "threshold_bps": threshold, "next_stage": stage, "observation_tag": tag}
    checks["first_phase_real_read_and_note_write"] = all(any(
        row["phase"] == "learn" and row["name"] == name for row in fixture.calls
    ) for name in ("read_record", "write_note"))
    checks["context_update_really_persisted"] = any(
        row["phase"] == "context" and row["name"] == "write_note"
        and row["arguments"] == {"threshold_bps": 8100, "next_stage": "S4"} for row in fixture.calls)
    checks["fresh_thread_really_reads_note"] = any(
        row["phase"] == "memory" and row["name"] == "read_note" for row in fixture.calls)
    checks["transient_tag_not_in_workspace"] = fixture.observation_tag.encode() not in fixture.workspace.read_bytes("project-note.json")
    checks["no_reexposed_record_after_first_phase"] = not any(
        row["phase"] != "learn" and fixture.observation_tag in json.dumps(row["result"])
        for row in fixture.calls)
    checks["only_fixture_mcp_calls"] = all(
        call.tool_type == "mcpToolCall" and call.mcp_server == "memory_fixture"
        for receipt in receipts for call in receipt.tool_calls)
    return {"passed": all(checks.values()), "checks": checks}


def _app_server_process(backend):
    # Exact installed SDK process object, not process-list command matching.
    return backend._client._client._sync._proc


@contextmanager
def probe_backend(output: Path, auth_path: Path, backend: str):
    if backend == "local_qwen":
        from training.automedbench_lite.local_qwen import local_qwen_setup
        with TemporaryDirectory(prefix="eva-context-mcp-", dir="/tmp") as temporary:
            with local_qwen_setup(run_root=output / "local-runtime", image_inputs=False, workers=1,
                                  thinking=True, exact_tool_schemas=True, max_output_tokens=8192) as setup:
                write_once(output / "local-backend.json", setup.safe_metadata)
                yield SimpleNamespace(backend=setup.backend, model=setup.model,
                    provider=setup.provider, config=setup.thread_config, mcp_root=Path(temporary))
        return
    if backend != "native_astra":
        raise ValueError("explicit native_astra or local_qwen backend required")
    catalog = output / "model-catalog.json"
    write_once(catalog, native_astra_model_catalog())
    with private_native_auth_copy(auth_path) as isolated:
        launch = native_astra_launch_options(codex_bin=Path(shutil.which("codex")).resolve(),
            script_path=ROOT / "scripts/run_native_astra_teacher_v1.py", isolation_root=isolated,
            cwd=output, catalog_path=catalog)
        config = {"project_doc_max_bytes": 0, "web_search": "disabled",
                  "features": dict(CODEX_FIRST_RELEASE_THREAD_CONFIG["features"]),
                  "model_reasoning_effort": "low", "model_reasoning_summary": "none",
                  "model_providers": {NATIVE_ASTRA_PROVIDER: native_astra_provider_config()}}
        yield SimpleNamespace(backend=lambda: OpenAICodexBackend(launch), model=NATIVE_ASTRA_MODEL,
            provider=NATIVE_ASTRA_PROVIDER, config=config, mcp_root=isolated / "tmp")


def checkpoint_binding(path: Path | None):
    if path is None:
        raise ValueError("local Qwen requires an explicit server checkpoint identity")
    document = json.loads(path.read_bytes())
    settings = document.get("settings", {})
    if (document.get("schema") != "eva.qwen-final-serving-checkpoint-identity.v1"
            or settings.get("served_model_name") != "Qwen/Qwen3.5-9B"
            or settings.get("host") != "127.0.0.1" or settings.get("port") != 30910
            or settings.get("context_length") != 32768
            or type(document.get("final_checkpoint_iteration")) is not int
            or not Path(document.get("exact_final_model_path", "")).is_dir()):
        raise ValueError("local server checkpoint binding differs")
    return document


async def run_probe(output: Path, auth_path: Path, timeout: int = 300, backend: str = "native_astra",
                    server_identity_path: Path | None = None):
    binding = checkpoint_binding(server_identity_path) if backend == "local_qwen" else None
    output = output.resolve()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    if binding is not None:
        write_once(output / "server-checkpoint-binding.json", binding)
    fixture = MemoryFixture(output)
    receipts, process_ids = [], []
    prompts = [
        "This is a synthetic research-workflow memory test, not a medical benchmark. "
        "Read the pilot record with read_record. Save its threshold_bps and next_stage using write_note. "
        "Remember observation_tag only in conversation; do not save it to any file. "
        "Return the three fields as JSON.",
        "The following irrelevant operational log is background only:\n" + "\n".join(
            f"log-{i:04d}: completed unrelated record batch; no change to the current project decision."
            for i in range(180)) + "\nNow update the project threshold to 8100 and next stage to S4, "
        "persist the updated note. Return the new fields and the ORIGINAL observation_tag from our "
        "earlier conversation. The pilot record is unavailable; do not try to reopen it.",
        "Continue this same research task after the app-server restart. Report our latest threshold "
        "and next stage, together with the original observation_tag, using the retained conversation. "
        "Do not invent or re-read the unavailable pilot record.",
        "You are continuing a project in a NEW conversation with no earlier chat history. Use read_note "
        "to recover the current threshold and next stage from the workspace. Return those fields. "
        "Return observation_tag=null because the persistent note does not contain it. Do not guess.",
    ]
    write_once(output / "prompts.json", prompts)
    with probe_backend(output, auth_path, backend) as setup:
        options = CodexThreadOptions(
            role=CodexRole.STRONG_ACTOR, model=setup.model, provider=setup.provider,
            cwd=str(fixture.workspace.root), sandbox=CodexSandbox.READ_ONLY, ephemeral=False,
            config=setup.config,
            offered_tools=fixture.offers(),
            developer_instructions="Use only the supplied fixture MCP tools. Record durable facts, not private reasoning. Follow each phase exactly.",
        )
        factory = TurnMCPBridgeFactory(proxy_python=Path(sys.executable).resolve(),
            proxy_script=ROOT / "src/eva_agent/codex_pipeline/turn_mcp_proxy.py",
            temp_root=setup.mcp_root, maximum_parallel_calls=1)

        async def turn(runtime, handle, index):
            receipt = await asyncio.wait_for(runtime.run_turn(handle, CodexTurnInput(
                public_text=prompts[index], model=setup.model, effort="low", summary="none",
                output_schema=ANSWER_SCHEMA)), timeout=timeout)
            receipts.append(receipt)
            write_once(output / f"turn-{index+1}.json", receipt)
            write_once(output / f"workspace-{index+1}.json", fixture.workspace.snapshot(f"phase-{index+1}"))
            write_once(output / f"tools-through-{index+1}.json", fixture.calls)
            print(json.dumps({"phase": index+1, "status": receipt.status, "tools": len(receipt.tool_calls)}), flush=True)

        with factory.open_actor(options, fixture) as bound:
            engine = setup.backend()
            async with CodexRuntime(engine) as runtime:
                process = _app_server_process(engine)
                process_ids.append(process.pid)
                handle = await runtime.start_thread(bound)
                thread_id = handle.thread_id
                await turn(runtime, handle, 0)
                fixture.phase = "context"
                await turn(runtime, handle, 1)
            stopped = process.poll() is not None
            fixture.phase = "resume"
            engine = setup.backend()
            async with CodexRuntime(engine) as runtime:
                process_ids.append(_app_server_process(engine).pid)
                handle = await runtime.resume_thread(thread_id, bound)
                await turn(runtime, handle, 2)
                fixture.phase = "memory"
                handle = await runtime.start_thread(replace(bound, ephemeral=True))
                await turn(runtime, handle, 3)
    result = {"schema": "eva.codex-context-memory-diagnostic.v1",
        **verify_probe(receipts, fixture, process_ids, stopped),
        "requested_model": setup.model, "returned_model": None, "backend": backend,
        "server_checkpoint_receipt_id": None if binding is None else binding["receipt_id"],
        "server_checkpoint_path": None if binding is None else binding["exact_final_model_path"],
        "app_server_pids": process_ids, "synthetic_fixture": True,
        "medical_performance_evaluated": False, "sft_rl_rows_exported": 0,
        "canonical_medical_schemas_changed": False, "automatic_compaction_tested": False,
        "context_prompt_bytes": len(prompts[1].encode()), "actual_tool_calls": len(fixture.calls),
        "limitations": ["Four controlled turns do not establish long-horizon clinical competence.",
                        "Workspace memory is explicit project notes, not changes to model weights.",
                        "Native returned model identity is not exposed by this runtime."]}
    write_once(output / "result.json", result)
    return result
