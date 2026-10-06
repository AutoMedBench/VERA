"""Authenticate interrupted native calls without inventing model observations."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

from eva_agent.codex_runtime import codex_core_mcp_resource_operation, codex_core_mcp_resource_call_error
from training.automedbench_lite.adapter import blake3, canonical, file_digest, read_document
from .benchmark_deadline import verify_capture, verify_signed_payload, write_signed_payload
from .benchmark_host_lifecycle import read_dispatch_events


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _frozen_sources(audit: Path, prelaunch: dict) -> tuple[dict, dict]:
    for root in audit.parents:
        path = root / 'harness-manifest.json'
        if path.is_file():
            manifest = read_document(path, maximum=16 * 1024**2)
            _require('manifest:' + manifest['document_blake3'] == prelaunch['binding']['source_revision'],
                     'partial_source_manifest_binding_changed')
            sources = {row['path']: row['blake3'] for row in manifest['sources']}
            names = ('_publication.py', 'run_prescribed_model.py', 'job_supervisor.py')
            paths = {name: root / 'frozen-sources/EVA-Agent/training/benchmark_models' / name for name in names}
            for name, path in paths.items():
                _require(file_digest(path) == sources['EVA-Agent/training/benchmark_models/' + name],
                         'partial_frozen_publisher_source_changed')
            return paths, sources
    raise ValueError('partial_frozen_manifest_missing')


def verify_host_publications(audit: Path, deadline: float, workspace: Path) -> dict:
    """Join every guarded note/ledger attempt to its actual host dispatch."""
    from uuid import UUID
    path = audit / 'host-publication/deadline-publication-events.jsonl'
    submit = {'automed_submit_model_job', 'automed_submit_extended_model_job',
              'automed_submit_generative_model_job'}
    mutators = submit | {'automed_write_note', 'automed_model_job_status'}
    host_file = audit / 'mcp-events.jsonl'
    _require(not host_file.is_symlink() and host_file.stat().st_size <= 256 * 1024**2,
             'host_publication_mcp_inventory_invalid')
    hosts = [json.loads(line) for line in host_file.read_bytes().splitlines()]
    hosts = [row for row in hosts if row['name'] in mutators]
    if not path.exists():
        _require(not any(row['is_error'] is False for row in hosts),
                 'successful_host_mutation_publication_journal_missing')
        return {'file_blake3': None, 'publication_count': 0}
    _require(not path.is_symlink() and path.stat().st_size <= 16 * 1024**2,
             'host_publication_journal_invalid')
    data = path.read_bytes()
    _require(not data or data.endswith(b'\n'), 'host_publication_journal_truncated')
    events = [json.loads(line) for line in data.splitlines()]
    intents, cancellations, ids, previous, count = {}, [], set(), 0, 0
    for row in events:
        _require(row.get('schema') == 'eva.prescribed-model-deadline-event.v1'
                 and row['track_deadline_monotonic'] == deadline and row.get('host_only') is True,
                 'host_publication_clock_changed')
        identifier = row['event_id']
        _require(str(UUID(identifier)) == identifier and identifier not in ids,
                 'host_publication_event_identity_invalid')
        ids.add(identifier)
        observed = row['observed_monotonic']
        _require(_finite(observed) and observed >= previous, 'host_publication_event_time_invalid')
        previous = observed
        _require(row.get('operation') in {'automed_write_note', 'model_job_status_ledger'},
                 'host_publication_operation_invalid')
        if row['event'] == 'policy_cancellation':
            _require(observed >= deadline and row.get('policy_cancellation') is True
                     and row.get('cancellation_reason') == 'track_wall_clock_budget_exhausted',
                     'host_publication_cancellation_unproved')
            cancellations.append(row)
            continue
        _require(row['event'] in {'public_publication_intent', 'public_publication', 'public_publication_cancelled'},
                 'host_publication_event_unrecognized')
        publication_id = row['publication_id']
        _require(str(UUID(publication_id)) == publication_id, 'host_publication_attempt_id_invalid')
        intents.setdefault(publication_id, []).append(row)
    attempts, used_cancellations = [], set()
    for rows in intents.values():
        _require(len(rows) == 2 and rows[0]['event'] == 'public_publication_intent'
                 and rows[1]['event'] in {'public_publication', 'public_publication_cancelled'}
                 and rows[0]['destination'] == rows[1]['destination']
                 and rows[0]['operation'] == rows[1]['operation'], 'host_publication_intent_unclosed')
        first, last = rows
        destination = Path(first['destination'])
        _require(destination.is_absolute() and destination.parent == workspace / 'notes'
                 and destination == destination.resolve(), 'host_publication_destination_invalid')
        published = last['event'] == 'public_publication'
        if published:
            start, end = last['publication_started_monotonic'], last['publication_completed_monotonic']
            _require(_finite(start) and _finite(end)
                     and first['observed_monotonic'] <= start <= end < deadline
                     and end <= last['observed_monotonic']
                     and last.get('complete_interval_before_deadline') is True,
                     'host_publication_interval_or_target_invalid')
            count += 1
        else:
            _require(last.get('action_started') is False and last.get('policy_cancellation') is True
                     and last['observed_monotonic'] >= deadline, 'host_publication_no_action_cancellation_invalid')
            matches = [row for row in cancellations if row['event_id'] not in used_cancellations
                       and row['operation'] == first['operation']
                       and first['observed_monotonic'] <= row['observed_monotonic'] <= last['observed_monotonic']]
            _require(len(matches) == 1, 'host_publication_intent_cancellation_missing_or_ambiguous')
            used_cancellations.add(matches[0]['event_id'])
        attempts.append({'operation': first['operation'], 'destination': str(destination),
            'started': first['observed_monotonic'], 'ended': last['observed_monotonic'], 'published': published})
    for row in cancellations:
        if row['event_id'] not in used_cancellations:
            attempts.append({'operation': row['operation'], 'destination': None,
                'started': row['observed_monotonic'], 'ended': row['observed_monotonic'], 'published': False})
    attempts.sort(key=lambda row: row['started'])
    dispatches = {}
    for row in read_dispatch_events(audit):
        dispatches.setdefault(row['dispatch_id'], []).append(row)
    by_host = {}
    for rows in dispatches.values():
        if rows[0].get('tool') not in mutators:
            continue
        _require([row['phase'] for row in rows] == ['started', 'completed', 'delivery'],
                 'host_publication_dispatch_incomplete')
        started, completed = rows[:2]
        _require(_finite(started['observed_monotonic']) and _finite(completed['observed_monotonic'])
                 and started['observed_monotonic'] <= completed['observed_monotonic']
                 and started['deadline_monotonic'] == completed['deadline_monotonic'] == deadline,
                 'host_publication_dispatch_clock_changed')
        _require(completed['event_id'] not in by_host, 'host_publication_dispatch_duplicate')
        by_host[completed['event_id']] = (started, completed)
    cursor = 0
    for host in hosts:
        _require(host['event_id'] in by_host, 'host_publication_mutation_dispatch_missing')
        start, end = by_host[host['event_id']]
        _require(start['tool'] == host['name'] and start['request_id'] == host['request_id']
                 and start['arguments_blake3'] == blake3(canonical(host['arguments'])).hexdigest()
                 and end['tool_is_error'] == host['is_error']
                 and end['response_blake3'] == host['response_blake3'], 'host_publication_dispatch_binding_changed')
        operation = 'automed_write_note' if host['name'] == 'automed_write_note' else 'model_job_status_ledger'
        destination = str(workspace / 'notes' / (host['arguments'].get('name', '')
                          if operation == 'automed_write_note' else 'model-jobs.jsonl'))
        attempt = attempts[cursor] if cursor < len(attempts) else None
        matches = (attempt is not None and attempt['operation'] == operation
                   and attempt['destination'] in {None, destination}
                   and start['observed_monotonic'] <= attempt['started'] <= attempt['ended'] <= end['observed_monotonic'])
        if not matches:
            _require(host['is_error'] is True, 'successful_host_mutation_publication_not_joined')
            continue
        cursor += 1
        if attempt['published']:
            _require(host['is_error'] is False, 'failed_host_call_has_completed_public_mutation')
            if operation == 'automed_write_note':
                content = host['arguments']['content'].encode()
                _require(host['result']['path'] == 'notes/' + host['arguments']['name']
                         and host['result']['bytes'] == len(content)
                         and host['result']['file_blake3'] == blake3(content).hexdigest(),
                         'host_note_published_result_changed')
        elif host['is_error'] is False:
            _require(host['name'] in submit and end['observed_monotonic'] >= deadline
                     and host['result'].get('job_id'), 'successful_host_mutation_cancellation_invalid')
        else:
            _require(host['result'].get('error_code') == 'track_wall_clock_budget_exhausted',
                     'host_publication_denial_error_changed')
    _require(cursor == len(attempts) and set(by_host) == {row['event_id'] for row in hosts},
             'host_publication_attempt_or_dispatch_unjoined')
    return {'file_blake3': file_digest(path), 'publication_count': count}


def _process_live(row):
    try:
        fields = Path(f"/proc/{row['pid']}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[19] == str(row["start_ticks"]) and fields[0] != "Z"


def host_joins(audit: Path, capture: dict, offered_names: tuple[str, ...]) -> dict:
    """Join only model-observed completed results; retain pending host results apart."""
    summary = capture["summary"]
    _require(summary["partial_deadline_terminal_admissible"] is True, "native_partial_terminal_not_proved")
    deadline = capture["binding"]["deadline_monotonic"]
    dispatches = {}
    events = read_dispatch_events(audit)
    for row in events:
        _require(row["deadline_monotonic"] == deadline, "host_dispatch_deadline_changed")
        _require(_finite(row["observed_monotonic"]), "host_dispatch_time_invalid")
        group = dispatches.setdefault(row["dispatch_id"], [])
        group.append(row)
    for group in dispatches.values():
        _require([row["phase"] for row in group] == ["started", "completed", "delivery"],
                 "host_dispatch_result_or_delivery_unobserved")
        _require(len({row["server_id"] for row in group}) == 1 and
                 all(group[i]["observed_monotonic"] <= group[i+1]["observed_monotonic"] for i in (0, 1)),
                 "host_dispatch_order_invalid")
        if group[2]["state"] == "peer_closed":
            _require(group[2]["observed_monotonic"] >= deadline, "predeadline_host_transport_failure")
        else:
            _require(group[2]["state"] == "written_to_pipe", "host_delivery_state_invalid")
    path = audit / "mcp-events.jsonl"
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 256 * 1024**2,
             "partial_host_events_missing_or_oversized")
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    hosts = {row["event_id"]: row for row in rows}
    _require(len(hosts) == len(rows), "partial_duplicate_host_event")
    by_host = {}
    for dispatch_id, (start, completed, delivered) in dispatches.items():
        host_id = completed["event_id"]
        _require(host_id in hosts and host_id not in by_host, "partial_dispatch_event_binding_invalid")
        host = hosts[host_id]
        _require(host["event_blake3"] == blake3(canonical({k: v for k, v in host.items() if k != "event_blake3"})).hexdigest()
                 and host["name"] == start["tool"] and host["request_id"] == start["request_id"]
                 and blake3(canonical(host["arguments"])).hexdigest() == start["arguments_blake3"]
                 and host["response_blake3"] == completed["response_blake3"]
                 and host["is_error"] == completed["tool_is_error"], "partial_host_event_commitment_invalid")
        if start["observed_monotonic"] >= deadline:
            _require(host["is_error"] and host["result"].get("error_code") == "track_wall_clock_budget_exhausted",
                     "postdeadline_host_operation_was_not_denied")
        by_host[host_id] = (dispatch_id, start, completed, delivered)
    used = set()
    completed_joins, pending_joins, controls = [], [], []
    for call in summary["complete_calls"]:
        control = codex_core_mcp_resource_operation(server=call["mcp_server"], tool=call["mcp_tool"],
                                                   offered_mcp_tool_names=offered_names)
        if control is not None:
            native = SimpleNamespace(name=call["name"], lifecycle=tuple(call["lifecycle"]),
                status=call["native_status"], arguments=call["arguments"],
                output=call["actual_native_output"], mcp_server=call["mcp_server"], mcp_tool=call["mcp_tool"])
            _require(call["tool_type"] == "mcpToolCall" and codex_core_mcp_resource_call_error(
                call=native, operation=control, offered_mcp_tool_names=offered_names) is None,
                "native_control_outcome_invalid")
            controls.append(control)
            continue
        output = call["actual_native_output"]
        result = output.get("result") if isinstance(output, dict) else None
        structured = result.get("structuredContent") if isinstance(result, dict) else None
        _require(call["tool_type"] == "mcpToolCall" and call["mcp_server"] == "automed_eval"
                 and call["native_status"] in {"completed", "failed"}
                 and tuple(call["lifecycle"]) == ("item/started", "item/completed")
                 and "automed_eval/" + call["mcp_tool"] in offered_names
                 and isinstance(structured, dict) and output.get("error") is None,
                 "completed_native_call_lacks_observed_host_result")
        host_id = structured.get("event_id")
        _require(host_id in hosts and host_id not in used, "completed_native_host_join_ambiguous")
        host = hosts[host_id]
        wire = dict(result)
        if wire.get("_meta", False) is None:
            wire.pop("_meta")
        wire.setdefault("isError", call["native_status"] == "failed")
        _require(host["name"] == call["mcp_tool"] == structured.get("name")
                 and host["arguments"] == call["arguments"] and host["result"] == structured.get("result")
                 and host["is_error"] == (call["native_status"] == "failed")
                 and host["response_blake3"] == blake3(canonical(wire)).hexdigest(),
                 "completed_native_host_result_changed")
        used.add(host_id)
        completed_joins.append({"upstream_item_id": call["upstream_item_id"], "host_event_id": host_id,
            "host_event_blake3": host["event_blake3"], "model_observed_result": True})
    for call in summary["pending_calls"]:
        _require(call["tool_type"] == "mcpToolCall" and call["mcp_server"] == "automed_eval"
                 and "automed_eval/" + call["mcp_tool"] in offered_names
                 and call["actual_native_output"] is None and call["model_observed_result"] is False,
                 "pending_native_call_type_or_visibility_invalid")
        matches = [host_id for host_id, host in hosts.items() if host_id not in used
                   and host["name"] == call["mcp_tool"] and host["arguments"] == call["arguments"]]
        _require(len(matches) == 1, "pending_host_dispatch_join_missing_or_ambiguous")
        host_id = matches[0]
        dispatch_id, started, completed, delivered = by_host[host_id]
        used.add(host_id)
        pending_joins.append({"upstream_item_id": call["upstream_item_id"], "dispatch_id": dispatch_id,
            "host_event_id": host_id, "host_event_blake3": hosts[host_id]["event_blake3"],
            "started_monotonic": started["observed_monotonic"],
            "host_completed_monotonic": completed["observed_monotonic"],
            "delivery_observed_monotonic": delivered["observed_monotonic"],
            "delivery_state": delivered["state"], "model_observed_result": False,
            "native_completion_synthesized": False})
    _require(used == set(hosts) and set(by_host) == set(hosts), "unjoined_host_work_at_partial_terminal")
    return {"completed_calls": completed_joins, "pending_calls": pending_joins,
            "native_control_operations": controls, "host_event_count": len(hosts),
            "dispatch_event_count": len(events), "last_dispatch_event_blake3": events[-1]["document_blake3"] if events else None}


def verify_host_stop(audit: Path, deadline: float, *, workspace: Path) -> dict:
    events = read_dispatch_events(audit)
    expected_servers = {row["server_id"] for row in events}
    servers = []
    for path in sorted(audit.glob("mcp-server-*.json")):
        process = read_document(path)
        _require(process["deadline_monotonic"] == deadline and not _process_live(process),
                 "host_mcp_process_not_quiescent")
        servers.append({"path": path.name, "document_blake3": process["document_blake3"]})
    _require({path["path"][11:-5] for path in servers} == expected_servers and servers,
             "host_mcp_process_inventory_incomplete")
    hosts = [json.loads(line) for line in (audit / "mcp-events.jsonl").read_bytes().splitlines()]
    expected_executions = {row["result"]["execution_id"] for row in hosts
        if row["name"] == "evamed_execute_python" and row["result"].get("process_started") is True}
    _require({path.parent.name for path in (audit / "code-executions").glob("*/process.json")}
             == expected_executions, "host_cpu_process_inventory_incomplete")
    executions = []
    for execution_id in sorted(expected_executions):
        _require('/' not in execution_id and execution_id not in {'.', '..'}, "cpu_execution_id_invalid")
        path = audit / "code-executions" / execution_id / "process.json"
        process = read_document(path)
        _require(process["track_deadline_monotonic"] == deadline
                 and process["effective_deadline_monotonic"] <= deadline
                 and process["started_monotonic"] < deadline and not _process_live(process),
                 "partial_cpu_process_not_bounded_or_quiescent")
        ended = read_document(path.with_name("execution.json"))
        _require(ended["execution_id"] == process["execution_id"] == execution_id
                 and ended["code_blake3"] == process["code_blake3"]
                 and ended["effective_deadline_monotonic"] == process["effective_deadline_monotonic"]
                 and file_digest(path.with_name("solution.py")) == process["code_blake3"],
                 "partial_cpu_execution_binding_changed")
        start, effective, exited, killed = (process["started_monotonic"], process["effective_deadline_monotonic"],
            ended["process_exit_observed_monotonic"], ended["kill_requested_monotonic"])
        _require(all(_finite(value) for value in (start, effective, exited)) and start < effective
                 and start <= exited and (killed is None or _finite(killed) and start <= killed <= exited),
                 "cpu_execution_time_order_invalid")
        if ended["process_exit_observed_monotonic"] > deadline:
            _require(ended["timed_out"] and ended["kill_requested_monotonic"] is not None
                     and ended["effective_deadline_monotonic"] == deadline
                     and deadline <= killed <= deadline + 1.0 and exited <= killed + 2.0,
                     "cpu_work_outlived_uncontrolled_deadline")
        from .benchmark_cpu_publication import verify_cpu_publication
        publication = verify_cpu_publication(path.parent, deadline, ended, workspace=workspace)
        executions.append({"execution_id": process["execution_id"],
            "process_document_blake3": process["document_blake3"],
            "execution_document_blake3": ended["document_blake3"],
            "kill_requested_monotonic": ended["kill_requested_monotonic"],
            "process_exit_observed_monotonic": ended["process_exit_observed_monotonic"],
            "publication": publication})
    return {"mcp_servers": servers, "cpu_executions": executions}


def finalize_partial_host(audit: Path, *, capture_path: Path, signer, prelaunch: dict,
                          offered_names: tuple[str, ...], cleanup: dict,
                          after_snapshot: dict, final_snapshot: dict, workspace: Path) -> dict:
    capture = verify_capture(capture_path, public_key=prelaunch["public_key_base64"],
                             expected_binding=prelaunch["binding"])
    joins = host_joins(audit, capture, offered_names)
    stopped = verify_host_stop(audit, capture["binding"]["deadline_monotonic"], workspace=workspace)
    host_publications = verify_host_publications(audit, capture['binding']['deadline_monotonic'], workspace)
    _require(cleanup["workspace_quiescent"] is True, "partial_gpu_cleanup_not_quiescent")
    from .benchmark_publication_admission import verify_publication_jobs
    source_paths, sources = _frozen_sources(audit, prelaunch)
    publishers = verify_publication_jobs(audit, workspace, capture['binding']['deadline_monotonic'],
                                         cleanup=cleanup, source_paths=source_paths)
    for path in audit.glob('mcp-server-*.json'):
        _require(read_document(path)['host_lifecycle_source_blake3'] ==
                 sources['evamed-codex/src/evamed_codex/benchmark_host_lifecycle.py'],
                 'partial_host_lifecycle_source_changed')
    payload = {"schema": "eva.benchmark-signed-partial-host-finalization.v1",
        "binding": prelaunch["binding"], "prelaunch_document_blake3": prelaunch["document_blake3"],
        "native_capture_relative_path": capture_path.relative_to(audit).as_posix(),
        "native_capture_file_blake3": capture["capture_file_blake3"],
        "host_dispatch_file_blake3": file_digest(audit / "host-dispatch.jsonl"),
        "mcp_events_file_blake3": file_digest(audit / "mcp-events.jsonl"),
        "host_joins": joins, "host_processes": stopped,
        'host_publications': host_publications,
        "model_publishers": publishers,
        "cleanup_document_blake3": cleanup["document_blake3"],
        "after_snapshot_blake3": after_snapshot["document_blake3"],
        "final_snapshot_blake3": final_snapshot["document_blake3"],
        "completed_requested_turns": False, "ordinary_CodexTurnReceipt_claimed": False,
        "model_observed_pending_host_results": False, "additional_policy_budget_granted": False,
        "sealed_monotonic": time.monotonic()}
    path = audit / "policy-partial-host-seal.json"
    write_signed_payload(path, payload, signer=signer)
    return {"path": str(path), "file_blake3": file_digest(path), "payload": payload}


def verify_partial_host(audit: Path, *, prelaunch: dict, offered_names: tuple[str, ...],
                        cleanup: dict, after_snapshot: dict, final_snapshot: dict,
                        workspace: Path, expected_file_blake3: str) -> dict:
    path = audit / 'policy-partial-host-seal.json'
    _require(file_digest(path) == expected_file_blake3, 'partial_host_seal_bytes_changed')
    payload = verify_signed_payload(json.loads(path.read_bytes()), prelaunch['public_key_base64'])
    _require(payload.get('schema') == 'eva.benchmark-signed-partial-host-finalization.v1'
             and payload['binding'] == prelaunch['binding']
             and payload['prelaunch_document_blake3'] == prelaunch['document_blake3']
             and payload['completed_requested_turns'] is False
             and payload['ordinary_CodexTurnReceipt_claimed'] is False
             and payload['model_observed_pending_host_results'] is False
             and payload['additional_policy_budget_granted'] is False, 'partial_host_seal_identity_invalid')
    relative = Path(payload['native_capture_relative_path'])
    _require(not relative.is_absolute() and '..' not in relative.parts, 'native_capture_path_invalid')
    capture = verify_capture(audit / relative, public_key=prelaunch['public_key_base64'],
                             expected_binding=prelaunch['binding'])
    _require(capture['capture_file_blake3'] == payload['native_capture_file_blake3']
             and file_digest(audit / 'host-dispatch.jsonl') == payload['host_dispatch_file_blake3']
             and file_digest(audit / 'mcp-events.jsonl') == payload['mcp_events_file_blake3'],
             'partial_host_evidence_bytes_changed')
    _require(host_joins(audit, capture, offered_names) == payload['host_joins']
             and verify_host_stop(audit, capture['binding']['deadline_monotonic'], workspace=workspace) == payload['host_processes'],
             'partial_host_join_or_stop_changed')
    _require(verify_host_publications(audit, capture['binding']['deadline_monotonic'], workspace)
             == payload['host_publications'], 'partial_host_publication_evidence_changed')
    from .benchmark_publication_admission import verify_publication_jobs
    source_paths, sources = _frozen_sources(audit, prelaunch)
    for path in audit.glob('mcp-server-*.json'):
        _require(read_document(path)['host_lifecycle_source_blake3'] ==
                 sources['evamed-codex/src/evamed_codex/benchmark_host_lifecycle.py'],
                 'partial_host_lifecycle_source_changed')
    _require(verify_publication_jobs(audit, workspace, capture['binding']['deadline_monotonic'],
                                    cleanup=cleanup, source_paths=source_paths)
             == payload['model_publishers'], 'partial_model_publication_evidence_changed')
    _require(cleanup['workspace_quiescent'] is True
             and cleanup['document_blake3'] == payload['cleanup_document_blake3']
             and after_snapshot['document_blake3'] == payload['after_snapshot_blake3']
             and final_snapshot['document_blake3'] == payload['final_snapshot_blake3'],
             'partial_snapshot_or_cleanup_changed')
    return {'capture': capture, 'host_seal': payload, 'host_seal_file_blake3': expected_file_blake3}


def require_partial_scoring_admission(run: Path, track: str, actor: dict) -> str:
    from eva_agent.pipeline.digests import blake3_hex
    from .benchmark_admission import provider_evidence
    audit = run / 'track-rollouts' / track
    frozen = read_document(run / 'harness-manifest.json', maximum=16 * 1024**2)
    manifest = read_document(run / 'track-run-manifest.json')
    budget = read_document(audit / 'track-budget.json')
    request = read_document(audit / 'turns/01-e2e/request.json')
    prelaunch = read_document(audit / 'native-capture-prelaunch.json')
    proof = read_document(audit / 'policy-partial-terminal.json')
    row = next(row for row in manifest['tracks'] if row['track'] == track)
    binding = {'run_id': manifest['run_id'], 'track': track,
        'source_revision': 'manifest:' + frozen['document_blake3'],
        'launch_file_blake3': file_digest(run / 'harness-manifest.json'),
        'codex_binary_blake3': frozen['codex_binary_blake3'],
        'logical_input_blake3': blake3_hex(request['logical_input']),
        'tool_catalog_blake3': frozen['tool_catalog_blake3'],
        'deadline_monotonic': budget['deadline_monotonic'], 'max_seconds': budget['timeout_seconds']}
    _require(prelaunch['binding'] == binding and proof['prelaunch_document_blake3'] == prelaunch['document_blake3']
             and proof['document_blake3'] == actor.get('partial_policy_terminal_blake3')
             and actor['completed_requested_turns'] is False and actor['turn_receipt_blake3s'] == []
             and actor['actual_turn_count'] == 0 and actor.get('actual_partial_native_turn_count') == 1
             and budget['timeout_seconds'] == actor['max_seconds'] == frozen['config']['max_seconds'] == 3600
             and actor['max_model_requests'] == frozen['config']['max_turns'] == 100,
             'partial_terminal_launch_or_budget_changed')
    _require(request['task_manifest_blake3'] == row['task_file_blake3']
             and request['public_tool_catalog'] == frozen['tool_catalog']
             and blake3(canonical(request['public_tool_catalog'])).hexdigest() == frozen['tool_catalog_blake3'],
             'partial_terminal_task_or_tools_changed')
    cleanup = read_document(audit / 'evamed-job-cleanup.json')
    after = read_document(audit / 'turns/01-e2e/after/manifest.json', maximum=16 * 1024**2)
    final = read_document(audit / 'final/manifest.json', maximum=16 * 1024**2)
    _require(cleanup['document_blake3'] == actor['cleanup_document_blake3']
             and final['document_blake3'] == actor['final_snapshot_blake3'], 'partial_actor_snapshot_changed')
    verified = verify_partial_host(audit, prelaunch=prelaunch,
        offered_names=tuple('automed_eval/' + tool['name'] for tool in frozen['tool_catalog']),
        cleanup=cleanup, after_snapshot=after, final_snapshot=final,
        workspace=run / row['workspace_relative'], expected_file_blake3=proof['host_seal_file_blake3'])
    start = verified['capture']['start']
    sources = {row['path']: row['blake3'] for row in frozen['sources']}
    for key, path in {
        'core_runtime_source_blake3': 'EVA-Harness/src/eva_agent/codex_runtime/runtime.py',
        'core_backend_source_blake3': 'EVA-Harness/src/eva_agent/codex_runtime/backend.py',
        'capture_source_blake3': 'evamed-codex/src/evamed_codex/benchmark_deadline.py'}.items():
        _require(start[key] == sources[path] == file_digest(run / 'frozen-sources' / path),
                 'partial_native_runtime_source_changed')
    providers = provider_evidence(audit)
    _require(providers == proof['provider_requests'] and len(providers) == actor['actual_model_request_count']
             and len(providers) <= 100, 'partial_provider_evidence_changed')
    return 'policy-budget-exhausted'
