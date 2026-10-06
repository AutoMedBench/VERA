from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eva_agent.admission import issue_signed_envelope
from eva_agent.campaign import CampaignLedger
from eva_agent.cli import main
from eva_agent.construction import PremiumConstructionQueue
from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes
from eva_agent.progress_dashboard import (
    FrontierSFTTrainingSnapshot,
    ProgressDashboardError,
    SFTTrainingSnapshot,
    read_bulk_rl_sandbox_progress,
    read_frontier_sft_training_progress,
    read_live_utilization,
    read_premium_construction_progress,
    read_sft_training_progress,
    read_teacher_batch_progress,
    render_bulk_rl_sandbox_progress,
    render_frontier_sft_training_progress,
    render_live_utilization,
    render_premium_construction_progress,
    render_sft_training_progress,
    render_sft_total_training_records,
    render_teacher_batch_progress,
)


def _signing_material(tmp_path: Path) -> tuple[Path, Path]:
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "host.pem"
    private_path.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    private_path.chmod(0o600)
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trust = tmp_path / "trust.json"
    trust.write_text(
        json.dumps(
            {
                "schema": "eva.ed25519-trust-store.v1",
                "status": "active",
                "algorithm": "Ed25519",
                "keys": {"dashboard-test": base64.b64encode(public).decode("ascii")},
            }
        )
    )
    return private_path, trust


def _install_bulk_rl(tmp_path: Path) -> tuple[Path, Path]:
    private, trust = _signing_material(tmp_path)
    root = tmp_path / "bulk"
    root.mkdir()
    payloads = (b"one\n", b"two-records\n")
    shards = []
    for ordinal, payload in enumerate(payloads):
        path = root / f"sandboxes-{ordinal:05d}.jsonl"
        path.write_bytes(payload)
        shards.append(
            {
                "ordinal": ordinal,
                "path": path.name,
                "record_count": 2,
                "byte_count": len(payload),
            }
        )
    manifest = {
        "schema": "eva.bulk-rl-sandbox-sharded-dataset.v1",
        "sandbox_count": 4,
        "split_counts": {"train": 4},
        "shard_count": 2,
        "shards": shards,
    }
    envelope = issue_signed_envelope(
        manifest,
        key_id="dashboard-test",
        private_key_path=private.resolve(),
    )
    (root / "manifest.json").write_bytes(canonical_json_bytes(envelope.to_document()))
    return root, trust


def _install_frontier_sft(
    tmp_path: Path,
    *,
    dataset_id: str,
    count: int,
    schema: str = "eva.execution-verified-frontier-prefix-sft-dataset.v2",
) -> Path:
    root = tmp_path / dataset_id
    shards = root / "shards"
    shards.mkdir(parents=True)
    payload = b"{}\n" * count
    (shards / "part-00000.jsonl").write_bytes(payload)
    selection = {
        "require_completed_frontier": True,
        "require_gate_passed_true": True,
        "require_empty_failed_check_ids": True,
        "require_exact_declared_artifact": True,
        "require_reopened_workspace_delta": True,
        "export_later_stages": False,
        "export_terminal_answer": False,
        "export_hidden_reasoning": False,
        "export_private_reference": False,
    }
    core = {
        "schema": schema,
        "dataset_id": dataset_id,
        "quality_tier": "execution_verified_frontier_prefix",
        "selection_status": "stage_prefix_verified_full_trajectory_not_claimed",
        "source_count": count,
        "slice_count": count,
        "counts_by_stage": {"S1": count},
        "selection": selection,
        "shards": [
            {
                "path": "shards/part-00000.jsonl",
                "slice_count": count,
                "byte_count": len(payload),
            }
        ],
        "agent_judged": False,
        "strict_full_trajectory_sft_eligible": False,
        "hidden_reasoning_included": False,
        "private_reference_included": False,
    }
    manifest = {**core, "manifest_blake3": blake3_hex(core)}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def _install_s2_frontier_sft(
    tmp_path: Path, *, dataset_id: str, count: int
) -> Path:
    root = tmp_path / dataset_id
    shards = root / "shards"
    shards.mkdir(parents=True)
    payload = b"{}\n" * count
    (shards / "part-00000.jsonl").write_bytes(payload)
    selection = {
        "require_completed_s2_selection_frontier": True,
        "require_exact_sequential_prerequisites": True,
        "require_authoritative_s1_s2_guidance": True,
        "require_gate_passed_true": True,
        "require_empty_failed_check_ids": True,
        "require_reopened_workspace_delta": True,
        "export_prerequisite_tool_observations": False,
        "export_private_reference_observations": False,
        "export_later_stages": False,
        "export_terminal_answer": False,
        "export_hidden_reasoning": False,
    }
    core = {
        "schema": "eva.execution-verified-s2-frontier-prefix-sft-dataset.v1",
        "dataset_id": dataset_id,
        "quality_tier": "execution_verified_s2_frontier_prefix",
        "selection_status": "s2_prefix_verified_full_trajectory_not_claimed",
        "source_count": count,
        "slice_count": count,
        "counts_by_stage": {"S2": count},
        "selection": selection,
        "shards": [
            {
                "path": "shards/part-00000.jsonl",
                "slice_count": count,
                "byte_count": len(payload),
            }
        ],
        "agent_judged": False,
        "strict_full_trajectory_sft_eligible": False,
        "hidden_reasoning_included": False,
        "private_reference_observations_included": False,
    }
    manifest = {**core, "manifest_blake3": blake3_hex(core)}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_bulk_rl_progress_reopens_only_manifest_and_shard_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, trust = _install_bulk_rl(tmp_path)
    original_read_bytes = Path.read_bytes

    def reject_jsonl_read(path: Path) -> bytes:
        if path.suffix == ".jsonl":
            raise AssertionError("dashboard must not read a dataset shard")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_jsonl_read)
    snapshot = read_bulk_rl_sandbox_progress(root, trust_store_path=trust)

    assert snapshot.target == 6_000
    assert snapshot.declared == snapshot.materialized_verified == 4
    assert snapshot.materialized_verified_shards == snapshot.shard_count == 2
    assert snapshot.manifest_signature_verified is True
    assert "4/6,000 materialized+verified" in render_bulk_rl_sandbox_progress(snapshot)
    assert snapshot.to_document()["verification_scope"].endswith("no_record_rehash")


def test_bulk_rl_progress_counts_only_present_descriptor_matched_shards(
    tmp_path: Path,
) -> None:
    root, trust = _install_bulk_rl(tmp_path)
    (root / "sandboxes-00001.jsonl").unlink()

    snapshot = read_bulk_rl_sandbox_progress(root, trust_store_path=trust)

    assert snapshot.declared == 4
    assert snapshot.materialized_verified == 2
    assert snapshot.materialized_verified_shards == 1


def test_sft_progress_reads_manifest_counts_without_opening_slice_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sft"
    shards = root / "shards"
    shards.mkdir(parents=True)
    payload = b"slice-a\nslice-b\n"
    (shards / "part-00000.jsonl").write_bytes(payload)
    manifest = {
        "schema": "eva.legacy-teacher-sft-dataset.v1",
        "source_count": 3,
        "slice_count": 2,
        "shard_count": 1,
        "shards": [
            {
                "path": "shards/part-00000.jsonl",
                "slice_count": 2,
                "byte_count": len(payload),
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    original_read_bytes = Path.read_bytes

    def reject_jsonl_read(path: Path) -> bytes:
        if path.suffix == ".jsonl":
            raise AssertionError("dashboard must not read an SFT shard")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_jsonl_read)
    snapshot = read_sft_training_progress(root)

    assert snapshot.full_trajectories == 3
    assert snapshot.declared_slices == snapshot.materialized_slices == 2
    assert "full_trajectories=3" in render_sft_training_progress(snapshot)
    assert "slices=2/2" in render_sft_training_progress(snapshot)


def test_frontier_sft_progress_aggregates_manifests_without_opening_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = (
        _install_frontier_sft(
            tmp_path,
            dataset_id="11111111-1111-4111-8111-111111111111",
            count=2,
        ),
        _install_frontier_sft(
            tmp_path,
            dataset_id="22222222-2222-4222-8222-222222222222",
            count=3,
        ),
    )
    original_open = os.open

    def reject_jsonl_open(path: str | bytes | os.PathLike[str], *args: object) -> int:
        if str(os.fspath(path)).endswith(".jsonl"):
            raise AssertionError("dashboard must not open a frontier SFT shard")
        return original_open(path, *args)

    monkeypatch.setattr(os, "open", reject_jsonl_open)
    snapshot = read_frontier_sft_training_progress(outputs)

    assert snapshot == FrontierSFTTrainingSnapshot(2, 5, 5, 5, 2, 2, True)
    rendered = render_frontier_sft_training_progress(snapshot)
    assert "datasets=2" in rendered
    assert "slices=5/5" in rendered
    assert "not Agent-Judged, not full-trajectory" in rendered
    with pytest.raises(ProgressDashboardError, match="duplicated"):
        read_frontier_sft_training_progress((outputs[0], outputs[0]))


def test_frontier_sft_progress_accepts_verified_native_route_v3_manifest(
    tmp_path: Path,
) -> None:
    output = _install_frontier_sft(
        tmp_path,
        dataset_id="55555555-5555-4555-8555-555555555555",
        count=4,
        schema="eva.execution-verified-frontier-prefix-sft-dataset.v3",
    )

    snapshot = read_frontier_sft_training_progress(output)

    assert snapshot.materialized_slices == 4


def test_frontier_sft_progress_accepts_s2_prefix_manifest(tmp_path: Path) -> None:
    output = _install_s2_frontier_sft(
        tmp_path,
        dataset_id="66666666-6666-4666-8666-666666666666",
        count=3,
    )

    snapshot = read_frontier_sft_training_progress(output)

    assert snapshot == FrontierSFTTrainingSnapshot(1, 3, 3, 3, 1, 1, True)


def test_sft_total_training_records_keeps_quality_categories_distinct() -> None:
    strict = SFTTrainingSnapshot(3, 246, 246, 2, 2, True)
    frontier = FrontierSFTTrainingSnapshot(1, 406, 406, 406, 1, 1, True)

    rendered = render_sft_total_training_records(strict, frontier)

    assert "total=652" in rendered
    assert "canonical_strict=246" in rendered
    assert "execution_verified_stage_prefix=406" in rendered


def test_teacher_progress_uses_checkpoint_aggregates_and_indexed_slice_count(
    tmp_path: Path,
) -> None:
    output = tmp_path / "teacher"
    output.mkdir()
    with sqlite3.connect(output / "checkpoint.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE tasks(state TEXT, result_path TEXT, score REAL, "
            "sft_slice_path TEXT)"
        )
        connection.executemany(
            "INSERT INTO tasks VALUES(?,?,?,?)",
            (
                ("queued", None, None, None),
                ("running", None, None, None),
                ("succeeded", "not-opened-result.json", 0.95, "not-opened-slice.json"),
                ("failed", None, None, None),
            ),
        )

    snapshot = read_teacher_batch_progress(output)

    assert (snapshot.queued, snapshot.running, snapshot.succeeded, snapshot.failed) == (
        1,
        1,
        1,
        1,
    )
    assert snapshot.high_score_sft_slices == 1
    assert snapshot.total == 4
    rendered = render_teacher_batch_progress(snapshot)
    assert "queued=1; running=1; succeeded=1; failed=1" in rendered
    assert "high_score_sft_slices=1" in rendered


def test_teacher_progress_reports_unavailable_for_legacy_unindexed_slices(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(checkpoint) as connection:
        connection.execute("CREATE TABLE tasks(state TEXT, result_path TEXT)")
        connection.execute("INSERT INTO tasks VALUES('succeeded','trajectory.json')")

    snapshot = read_teacher_batch_progress(checkpoint)

    assert snapshot.succeeded == 1
    assert snapshot.high_score_sft_slices is None
    assert "unavailable-in-checkpoint" in render_teacher_batch_progress(snapshot)


def test_teacher_progress_aggregates_multiple_checkpoints_without_opening_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = (tmp_path / "canary", tmp_path / "terra-wave", tmp_path / "top-canary")
    states = (
        (("running", None, None, None), ("succeeded", "never-read-a", 0.9, "slice-a")),
        (("queued", None, None, None), ("failed", None, None, None)),
        (("succeeded", "never-read-b", 0.95, "slice-b"),),
    )
    for output, rows in zip(outputs, states, strict=True):
        output.mkdir()
        with sqlite3.connect(output / "checkpoint.sqlite3") as connection:
            connection.execute(
                "CREATE TABLE tasks(state TEXT, result_path TEXT, score REAL, "
                "sft_slice_path TEXT)"
            )
            connection.executemany("INSERT INTO tasks VALUES(?,?,?,?)", rows)

    def reject_result_read(_path: Path) -> bytes:
        raise AssertionError("teacher progress must not open trajectory or slice content")

    monkeypatch.setattr(Path, "read_bytes", reject_result_read)
    snapshot = read_teacher_batch_progress(outputs)

    assert (snapshot.queued, snapshot.running, snapshot.succeeded, snapshot.failed) == (
        1,
        1,
        2,
        1,
    )
    assert snapshot.high_score_sft_slices == 2
    assert "queued=1; running=1; succeeded=2; failed=1" in render_teacher_batch_progress(
        snapshot
    )
    with pytest.raises(ProgressDashboardError, match="duplicated"):
        read_teacher_batch_progress((outputs[0], outputs[0]))


def test_cli_leads_with_rl_and_labels_rollout_admission_separately(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, trust = _install_bulk_rl(tmp_path)
    teacher_output = tmp_path / "teacher-cli"
    teacher_output.mkdir()
    with sqlite3.connect(teacher_output / "checkpoint.sqlite3") as connection:
        connection.execute("CREATE TABLE tasks(state TEXT, result_path TEXT)")

    assert main(
        [
            "progress",
            "--rl-root",
            str(root),
            "--rl-trust-store",
            str(trust),
            "--ledger",
            str(tmp_path / "missing.sqlite3"),
            "--teacher-output",
            str(teacher_output),
            "--no-premium",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert output.startswith("RL sandboxes [")
    assert "4/6,000 materialized+verified" in output
    assert "teacher batch: queued=0; running=0; succeeded=0; failed=0" in output
    assert "rollout admission (selection telemetry; not the RL count):" in output


def test_cli_repeated_teacher_outputs_render_one_aggregate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, trust = _install_bulk_rl(tmp_path)
    outputs = (tmp_path / "canary", tmp_path / "wave")
    for output, state in zip(outputs, ("running", "succeeded"), strict=True):
        output.mkdir()
        with sqlite3.connect(output / "checkpoint.sqlite3") as connection:
            connection.execute("CREATE TABLE tasks(state TEXT, result_path TEXT)")
            connection.execute("INSERT INTO tasks VALUES(?,NULL)", (state,))

    assert main(
        [
            "progress",
            "--rl-root",
            str(root),
            "--rl-trust-store",
            str(trust),
            "--teacher-output",
            str(outputs[0]),
            "--teacher-output",
            str(outputs[1]),
            "--no-premium",
            "--no-admission",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert output.count("teacher batch:") == 1
    assert "queued=0; running=1; succeeded=1; failed=0" in output


def test_cli_reports_strict_frontier_and_total_sft_without_scanning_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bulk_root, trust = _install_bulk_rl(tmp_path)
    strict = tmp_path / "strict"
    (strict / "shards").mkdir(parents=True)
    strict_payload = b"{}\n{}\n"
    (strict / "shards/part-00000.jsonl").write_bytes(strict_payload)
    (strict / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "eva.legacy-teacher-sft-dataset.v1",
                "source_count": 1,
                "slice_count": 2,
                "shard_count": 1,
                "shards": [
                    {
                        "path": "shards/part-00000.jsonl",
                        "slice_count": 2,
                        "byte_count": len(strict_payload),
                    }
                ],
            }
        )
    )
    frontier_a = _install_frontier_sft(
        tmp_path,
        dataset_id="33333333-3333-4333-8333-333333333333",
        count=2,
    )
    frontier_b = _install_frontier_sft(
        tmp_path,
        dataset_id="44444444-4444-4444-8444-444444444444",
        count=3,
    )

    assert main(
        [
            "progress",
            "--rl-root",
            str(bulk_root),
            "--rl-trust-store",
            str(trust),
            "--sft-output",
            str(strict),
            "--frontier-sft-output",
            str(frontier_a),
            "--frontier-sft-output",
            str(frontier_b),
            "--no-premium",
            "--no-admission",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "SFT canonical strict: full_trajectories=1; slices=2/2" in output
    assert "SFT execution-verified stage prefixes: datasets=2; sources=5" in output
    assert "SFT training records: total=7; canonical_strict=2; " in output
    assert "execution_verified_stage_prefix=5" in output


def test_cli_can_hide_legacy_admission_for_training_dashboard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, trust = _install_bulk_rl(tmp_path)
    assert main(
        [
            "progress",
            "--rl-root",
            str(root),
            "--rl-trust-store",
            str(trust),
            "--ledger",
            str(tmp_path / "missing.sqlite3"),
            "--no-premium",
            "--no-admission",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert output.startswith("RL sandboxes [")
    assert "rollout admission" not in output


def _queue_document(candidate_ids: list[str]) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for ordinal, candidate_id in enumerate(candidate_ids, start=1):
        core: dict[str, object] = {
            "candidate_id": candidate_id,
            "source_candidate_id": f"source-{ordinal}",
            "source_family": "automedbench",
            "source_artifact_sha256": "a" * 64,
            "domain": "automedbench",
            "stage": "E2E",
            "selection_tier": "primary",
            "selection_queue_ordinal": ordinal,
            "campaign_ordinal": ordinal,
            "readiness_proof_root_blake3": "b" * 64,
        }
        entries.append({**core, "entry_blake3": blake3_hex(core)})
    core = {
        "schema": "eva.premium-construction-queue.v1",
        "queue_id": str(uuid4()),
        "selection_id": str(uuid4()),
        "selection_blake3": "c" * 64,
        "readiness_authority_blake3": "d" * 64,
        "entries": entries,
        "consumed_record_blake3s": [],
        "frozen_candidate_count_before_exclusion": len(entries),
        "bound_excluded_count": 0,
        "quarantined_excluded_count": 0,
        "controls": {
            "selection_readiness": "frozen",
            "ordering": "selection_v2_queue_ordinal_after_consumed_exclusion",
            "outcome_fields_read_for_ordering": False,
            "semantic_retry_count": 0,
            "supervisor_transition_claimed": False,
        },
    }
    return {**core, "queue_blake3": blake3_hex(core)}


def _install_state(root: Path) -> tuple[PremiumConstructionQueue, list[str]]:
    candidate_ids = [str(uuid4()) for _ in range(3)]
    document = _queue_document(candidate_ids)
    queue = PremiumConstructionQueue.from_document(document)
    (root / "claims").mkdir(parents=True)
    (root / "terminal").mkdir()
    (root / "queue.v1.json").write_bytes(canonical_json_bytes(document))

    entry = queue.entries[0]
    claim_core = {
        "schema": "eva.premium-construction-claim.v1",
        "claim_id": str(uuid4()),
        "session_id": str(uuid4()),
        "queue_id": queue.queue_id,
        "queue_blake3": queue.queue_blake3,
        "selection_blake3": queue.selection_blake3,
        "candidate_id": entry.candidate_id,
        "source_candidate_id": entry.source_candidate_id,
        "campaign_ordinal": entry.campaign_ordinal,
        "entry_blake3": entry.entry_blake3,
        "started_at_utc": "2026-09-07T00:00:00Z",
        "semantic_retry_count": 0,
    }
    claim = {**claim_core, "claim_blake3": blake3_hex(claim_core)}
    (root / "claims" / f"{entry.candidate_id}.json").write_bytes(
        canonical_json_bytes(claim)
    )
    terminal_core = {
        "schema": "eva.premium-construction-terminal.v1",
        "record_id": str(uuid4()),
        "queue_id": queue.queue_id,
        "queue_blake3": queue.queue_blake3,
        "selection_blake3": queue.selection_blake3,
        "candidate_id": entry.candidate_id,
        "source_candidate_id": entry.source_candidate_id,
        "campaign_ordinal": entry.campaign_ordinal,
        "session_id": claim["session_id"],
        "claim_blake3": claim["claim_blake3"],
        "status": "quarantined",
        "logical_provider_calls_started": 2,
        "semantic_retry_count": 0,
        "error_code": "fixture_failure",
        "failure_receipt_blake3s": [],
        "publication_receipt": None,
        "binding_blake3": None,
        "validator_authority_blake3": None,
        "executable_material_blake3": None,
        "completed_at_utc": "2026-09-07T00:01:00Z",
        "exception_text_recorded": False,
        "supervisor_transition_claimed": False,
    }
    terminal = {**terminal_core, "record_blake3": blake3_hex(terminal_core)}
    (root / "terminal" / f"{entry.candidate_id}.json").write_bytes(
        canonical_json_bytes(terminal)
    )
    return queue, candidate_ids


def test_premium_dashboard_reopens_append_only_state_without_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "premium"
    _install_state(root)
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    snapshot = read_premium_construction_progress(root)

    assert snapshot.queue_total == 3
    assert snapshot.queued == 2
    assert snapshot.claimed == 1
    assert snapshot.active == snapshot.succeeded == 0
    assert snapshot.quarantined == 1
    assert snapshot.logical_provider_calls_started_known == 2
    assert "succeeded=0; quarantined=1" in render_premium_construction_progress(snapshot)
    assert before == {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_premium_dashboard_fails_closed_on_claim_tamper(tmp_path: Path) -> None:
    root = tmp_path / "premium"
    _, candidate_ids = _install_state(root)
    claim_path = root / "claims" / f"{candidate_ids[0]}.json"
    claim = json.loads(claim_path.read_bytes())
    claim["campaign_ordinal"] = 3
    claim_path.write_bytes(canonical_json_bytes(claim))

    with pytest.raises(ProgressDashboardError, match="claim verification"):
        read_premium_construction_progress(root)


def _fake_process(proc: Path, pid: int, cwd: Path, arguments: list[str]) -> None:
    process = proc / str(pid)
    process.mkdir(parents=True)
    (process / "cwd").symlink_to(cwd, target_is_directory=True)
    (process / "cmdline").write_bytes(
        b"\0".join(value.encode("utf-8") for value in arguments) + b"\0"
    )


def test_live_dashboard_reports_only_allowlisted_counts_and_stage_occupancy(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    ledger_path = project / "campaign.sqlite3"
    ledger = CampaignLedger(ledger_path)
    candidate_id = str(uuid4())
    ledger.enqueue_candidate(
        candidate_id=candidate_id,
        source_episode_id="progress-stage-fixture",
        domain="medxpertqa",
        stage="S2",
        split="train",
        rubric_blake3="a" * 64,
    )
    assert ledger.claim(worker_id="progress-test", limit=1, lease_seconds=60) == (
        candidate_id,
    )
    proc = tmp_path / "proc"
    proc.mkdir()
    _fake_process(
        proc,
        101,
        project,
        ["python", "scripts/run_campaign_v2.py", "production", "--profile", "128"],
    )
    _fake_process(
        proc,
        102,
        project,
        [
            "python",
            "scripts/run_premium_codex_construction.py",
            "production",
            "--profile",
            "256",
        ],
    )
    _fake_process(proc, 103, project, ["codex", "app-server"])
    outside = tmp_path / "outside"
    outside.mkdir()
    _fake_process(proc, 104, outside, ["codex", "app-server"])
    _fake_process(
        proc,
        105,
        project,
        ["python", "scripts/run_campaign_v2.py", "preflight", "--profile", "512"],
    )

    live = read_live_utilization(
        ledger=ledger_path,
        project_root=project,
        proc_root=proc,
    )

    assert live.rollout_launchers == live.construction_launchers == 1
    assert live.rollout_worker_lanes_configured == 128
    assert live.construction_worker_lanes_configured == 256
    assert live.persistent_codex_app_servers == 1
    assert dict(live.rollout_active_by_stage) == {"S2": 1}
    document = live.to_document()
    assert document["provider_requests_in_flight"] is None
    assert not {"pid", "argv", "environment", "credentials"} & set(document)
    rendered = render_live_utilization(
        live, rollout_active=1, construction_active=0
    )
    assert "rollout=1/128 candidate lanes" in rendered
    assert "active_stages=S2=1" in rendered
    assert "provider_requests_in_flight=unobserved" in rendered


def test_live_dashboard_counts_teacher_and_codex_in_explicit_runtime_worktree(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    runtime = tmp_path / "teacher-runtime-worktree"
    runtime.mkdir()
    proc = tmp_path / "proc"
    proc.mkdir()
    _fake_process(
        proc,
        201,
        runtime,
        ["python", "scripts/run_codex_teacher_batch_v1.py", "run", "--persistent"],
    )
    _fake_process(
        proc,
        202,
        runtime,
        ["python", "scripts/run_one_codex_teacher_rollout_v1.py"],
    )
    _fake_process(proc, 203, runtime, ["codex", "app-server", "--listen", "stdio://"])
    _fake_process(proc, 204, tmp_path, ["codex", "app-server"])

    live = read_live_utilization(
        ledger=project / "absent.sqlite3",
        project_root=project,
        proc_root=proc,
        runtime_worktrees=(runtime,),
    )

    assert live.teacher_batch_processes == 1
    assert live.teacher_task_processes == 1
    assert live.persistent_codex_app_servers == 1
    document = live.to_document()
    assert document["teacher_worker_processes"] == 2
    assert not {"pid", "argv", "environment", "credentials"} & set(document)
    rendered = render_live_utilization(live, rollout_active=0, construction_active=0)
    assert "teacher_workers=2 (batch=1,task=1)" in rendered
    assert "persistent_codex_app_servers=1" in rendered


def test_absent_premium_state_is_honest_zero(tmp_path: Path) -> None:
    snapshot = read_premium_construction_progress(tmp_path / "absent")

    assert snapshot.state_exists is False
    assert snapshot.queue_total == snapshot.completed == 0
    assert "state absent" in render_premium_construction_progress(snapshot)


def test_dangling_premium_state_symlink_is_not_reported_as_absent(
    tmp_path: Path,
) -> None:
    link = tmp_path / "premium"
    link.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(ProgressDashboardError, match="topology"):
        read_premium_construction_progress(link)
