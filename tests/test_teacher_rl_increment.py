from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from eva_agent.codex_runtime import CodexToolCall
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes, canonical_value
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.training.teacher_rl_increment import (
    TeacherRLIncrementError, build_teacher_rl_increment, derive_teacher_rl_rows,
    verify_teacher_rl_increment, select_nonterminal_teacher_rl_increment,
)
from test_async_agent_judge import _event, _snapshot
from test_frontier_prefix_sft import _synthetic_receipt


ROOT = Path(__file__).resolve().parents[1]
REAL_SOURCE = (ROOT / "runs/codex-teacher-ladder-s2-wave16-canary.v2/trajectories"
               / "8201979a-7403-54b9-9916-8a2fcc8048fa/qwen_3_5_122b_a10b/result.json")


@pytest.fixture(scope="module")
def registry():
    return load_and_compile_registry(ROOT / "rubrics/source/domain-stage-tables.v1.json")


def _source(path, registry):
    rubric = registry.resolve("medxpertqa", "S1")
    candidate_id, sandbox_id = str(uuid4()), str(uuid4())
    source_binding = {"candidate_id": candidate_id, "domain": "medxpertqa", "stage": "S1"}
    task = {"instruction": "Use the loaded skill and produce a plan.", "domain": "medxpertqa",
            "stage": "S1", "private_reference_visible": False}
    policy = {"target_split": "train", "tools": [{"name": "materialize_plan", "description": "Materialize the plan.",
                         "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}}]}
    files = {"input/source-binding.json": canonical_json_bytes(source_binding),
             "input/task-contract.json": canonical_json_bytes(task),
             ".eva/source-policy.json": canonical_json_bytes(policy)}
    before = _snapshot("before-rollout", files)
    after = _snapshot("after-rollout", {**files, "work/plan.json": b'{"done":true}\n'})
    results, calls, messages, bindings = [], [], [_event("system", "Use tools."), _event("user", task)], {}
    for ordinal, (name, output, target) in enumerate((
        ("load_skill", {"skill_id": "plan", "content": "Read evidence before planning."}, before),
        ("materialize_plan", {"gate_passed": False, "details": "Plan retained for further work."}, after),
    )):
        call_id, codex_id = str(uuid4()), str(uuid4())
        arguments = {"skill_id": "plan", "stage": "S1"} if name == "load_skill" else {}
        result_core = {"call_id": call_id, "result_id": str(uuid4()), "name": name,
                       "frontier": ordinal, "parallel_group_id": str(uuid4()), "status": "completed",
                       "error_code": None, "output": output,
                       "workspace_before_blake3": before["tree_blake3"],
                       "workspace_after_blake3": target["tree_blake3"]}
        result = {**result_core, "receipt_blake3": blake3_hex(result_core)}
        observation_core = {"schema": "eva.codex-pipeline-tool-observation.v1", "call_id": call_id,
                            "name": name, "arguments": arguments, "tool_result": result}
        observation = {**observation_core, "bridge_receipt_blake3": blake3_hex(observation_core)}
        call_core = {"tool_call_id": codex_id, "upstream_item_id": f"tool-{ordinal}",
                     "tool_type": "mcpToolCall", "name": name, "mcp_server": "evamed", "mcp_tool": name,
                     "fully_qualified_name": f"evamed/{name}", "status": "completed", "arguments": arguments,
                     "output": {"result": {"structuredContent": observation}},
                     "lifecycle": ("item/started", "item/completed"), "first_event_sequence": 0}
        call = CodexToolCall(**call_core, receipt_blake3=blake3_hex(call_core))
        messages.extend([
            _event("assistant", {"codex_tool_calls": [{"call_id": call_id, "codex_tool_call_id": codex_id,
                    "fully_qualified_name": call.fully_qualified_name, "arguments": arguments,
                    "receipt_blake3": call.receipt_blake3}]}, (call_id,)),
            _event("tool", {key: result[key] for key in ("error_code", "output", "receipt_blake3", "status")}, (call_id,)),
        ])
        results.append(result)
        calls.append(call)
        bindings[codex_id] = call_id
    receipt_core = dict(_synthetic_receipt().core())
    receipt_core.update(tool_calls=tuple(calls), max_parallelism_observed=1,
                        offered_mcp_tool_names=("evamed/load_skill", "evamed/materialize_plan"))
    receipt = type(_synthetic_receipt())(**receipt_core, receipt_blake3=blake3_hex(receipt_core))
    messages.append(_event("assistant", {"response": receipt.final_response,
                          "codex_turn_receipt_blake3": receipt.receipt_blake3}))
    trace_core = {"results": results, "declared_call_ids": list(bindings.values()),
                  "joined_call_ids": list(bindings.values()), "frontier_count": 2,
                  "max_parallelism_observed": 1, "retry_count": 0}
    value = {"schema": "eva.codex-teacher-full-trajectory.v1", "task_id": f"{sandbox_id}--fixture",
             "sandbox_id": sandbox_id, "candidate_id": candidate_id, "route_id": "fixture",
             "model_id": receipt.model, "provider": receipt.provider, "assistant_output": receipt.final_response,
             "messages": messages, "tool_trace": {**trace_core, "trace_blake3": blake3_hex(trace_core)},
             "workspace_before": before, "workspace_after": after, "rubric_table": rubric.to_document(),
             "skill_delivery": {"mode": "search-then-load", "visible_skill_ids": ["plan"]},
             "provider_metadata": {"raw_input_recorded": False, "codex_to_pipeline_call_ids": bindings,
                                   "codex_turn_receipt": canonical_value(receipt)}}
    path.write_bytes(canonical_json_bytes(value))
    return value


def test_prefix_preserves_loaded_skill_and_hides_future_answer(tmp_path, registry):
    source = tmp_path / "source.json"
    original = _source(source, registry)
    output = tmp_path / "increment"
    report = build_teacher_rl_increment(source_paths=[source], output_root=output, registry=registry)
    assert report["valid"] and report["row_count"] == 1
    row = json.loads((output / "continuations.jsonl").read_text())
    assert row["actor_context"]["messages"] == original["messages"][:4]
    assert row["actor_context"]["messages"][-1]["content"]["output"]["skill_id"] == "plan"
    assert row["workspace_initial_state"] == original["workspace_before"]
    assert "work/plan.json" not in {item["path"] for item in row["workspace_initial_state"]["files"]}
    assert row["rubric_table"] == original["rubric_table"]
    assert row["runtime_contract"]["host_session_materialized"] is False
    assert (output / row["source"]["audit_path"]).read_bytes() == source.read_bytes()
    assert verify_teacher_rl_increment(output, registry=registry)["row_count"] == 1
    report = select_nonterminal_teacher_rl_increment(output, selection_output=tmp_path / 'selection.json')
    assert report['selected_candidate_count'] == 1
    assert report['usable_hosted_sandbox_count'] == 0


def test_already_completed_stage_is_not_a_new_rl_task(tmp_path, registry):
    source = tmp_path / 'source.json'
    original = _source(source, registry)
    output = tmp_path / 'raw'
    build_teacher_rl_increment(source_paths=[source], output_root=output, registry=registry)
    # Selection is a separate view of raw exports, never a repair to source data.
    row = json.loads((output / 'continuations.jsonl').read_text())
    source_path = output / row['source']['audit_path']
    prior = original['tool_trace']['results'][0]
    prior.update(name='materialize_plan', output={'stage': 'S1', 'gate_passed': True})
    raw = canonical_json_bytes(original)
    digest = blake3_bytes(raw)
    source_path = output / 'sources' / f'{digest}.json'
    source_path.write_bytes(raw)
    row['source'].update(file_blake3=digest, audit_path=f'sources/{digest}.json')
    rows = canonical_json_bytes(row)
    (output / 'continuations.jsonl').write_bytes(rows)
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest.update(rows_file_blake3=blake3_bytes(rows))
    manifest['manifest_blake3'] = blake3_hex({k: v for k, v in manifest.items() if k != 'manifest_blake3'})
    (output / 'manifest.json').write_bytes(canonical_json_bytes(manifest))
    report = select_nonterminal_teacher_rl_increment(output, selection_output=tmp_path / 'selection.json')
    assert report['selected_candidate_count'] == 0
    assert report['excluded'] == [{'sandbox_id': row['sandbox_id'], 'reason': 'target_stage_already_completed'}]
    assert (output / 'continuations.jsonl').read_bytes() == rows


def test_prior_increment_excludes_same_state_and_trace_ids_do_not_mint_new_state(tmp_path, registry):
    source = tmp_path / "source.json"
    original = _source(source, registry)
    first = tmp_path / "first"
    build_teacher_rl_increment(source_paths=[source], output_root=first, registry=registry)
    for event in original["messages"]:
        event["event_id"] = str(uuid4())
        event["event_blake3"] = blake3_hex({k: v for k, v in event.items() if k != "event_blake3"})
    second_source = tmp_path / "second-source.json"
    second_source.write_bytes(canonical_json_bytes(original))
    report = build_teacher_rl_increment(source_paths=[second_source], output_root=tmp_path / "second",
                                       registry=registry, exclude_roots=[first])
    assert report["valid"] and report["row_count"] == 0


def test_tampered_source_and_recommitted_future_context_are_rejected(tmp_path, registry):
    source = tmp_path / "source.json"
    original = _source(source, registry)
    output = tmp_path / "increment"
    build_teacher_rl_increment(source_paths=[source], output_root=output, registry=registry)
    row_path = output / "continuations.jsonl"
    row = json.loads(row_path.read_text())
    row["actor_context"]["messages"] = original["messages"]
    row["row_blake3"] = blake3_hex({k: v for k, v in row.items() if k != "row_blake3"})
    payload = canonical_json_bytes(row)
    row_path.write_bytes(payload)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest.update(rows_file_blake3=blake3_bytes(payload), rows_byte_count=len(payload))
    manifest["manifest_blake3"] = blake3_hex({k: v for k, v in manifest.items() if k != "manifest_blake3"})
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(TeacherRLIncrementError, match="rows differ"):
        verify_teacher_rl_increment(output, registry=registry)
    original["messages"][3]["content"]["output"]["content"] = "tampered"
    source.write_bytes(canonical_json_bytes(original))
    with pytest.raises(ValueError, match="commitment differs"):
        derive_teacher_rl_rows(source, registry=registry)


@pytest.mark.skipif(not REAL_SOURCE.exists(), reason="generated historical source is local only")
def test_retained_real_qwen_s2_source_keeps_historical_attribution(tmp_path, registry):
    report = build_teacher_rl_increment(source_paths=[REAL_SOURCE], output_root=tmp_path / "real", registry=registry)
    assert report["valid"] and report["row_count"] == 1
    row = json.loads((tmp_path / "real/continuations.jsonl").read_text())
    assert row["source"]["route_id"] == "qwen_3_5_122b_a10b"
    assert row["stage"] == "S2"
    assert row["starting_point"]["observed_tool_result_count"] == 1
