from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from eva_agent.pipeline.digests import blake3_hex
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.sandboxes import (
    BenchmarkSourceStart,
    TrajectoryRLSandboxError,
    TrajectoryStepStart,
    build_trajectory_rl_sandbox,
    read_regular_file_tree,
    verify_trajectory_rl_sandbox,
)


ROOT = Path(__file__).resolve().parents[1]


def uid(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012d}"


@pytest.fixture(scope="module")
def registry():
    return load_and_compile_registry(ROOT / "rubrics/source/domain-stage-tables.v1.json")


def common(registry) -> dict:
    return {
        "sandbox_id": uid(1),
        "binding_id": uid(2),
        "benchmark": "AutoMedBench",
        "source_file": "benchmarks/automedbench/case-7.json",
        "source_revision": "5394fe7aa73e6b5891fe43942c99f4b0c2b50873",
        "source_record_id": "case-7",
        "domain": "automedbench-classification",
        "stage": "S2",
        "instruction": "Inspect the available public evidence and produce the S2 artifact.",
        "initial_workspace_files": {
            "case/input.json": b'{"case":7}\n',
            "README.md": b"public task\n",
        },
        "evidence_files": {"source/case.json": b'{"label_set":["a","b"]}\n'},
        "rubric_registry": registry,
    }


def test_benchmark_start_is_deterministic_and_has_no_quality_gate(registry) -> None:
    inputs = common(registry)
    first = build_trajectory_rl_sandbox(
        **inputs,
        start=BenchmarkSourceStart(),
    )
    second = build_trajectory_rl_sandbox(
        **inputs,
        start=BenchmarkSourceStart(),
    )

    assert first == second
    assert first["split"] == "train"
    assert first["starting_point"]["origin"] == "benchmark_source"
    assert len(first["rubric_table"]["items"]) in range(5, 11)
    assert first["rubric_binding"]["rubric_digest"] == first["rubric_table"]["rubric_digest"]
    assert first["reward_calculation_contract"]["rubric_digest"] == first["rubric_table"]["rubric_digest"]
    assert first["construction_policy"]["admission_claimed"] is False
    assert set(first["construction_policy"]["explicitly_not_required"]) == {
        "source_rollout_success",
        "cascade_completion",
        "ability_separation",
        "agent_judge_assessment",
        "minimum_reward",
    }
    assert "content_blake3" not in json.dumps(first)
    report = verify_trajectory_rl_sandbox(
        first,
        initial_workspace_files=inputs["initial_workspace_files"],
        evidence_files=inputs["evidence_files"],
        rubric_registry=registry,
    )
    assert report.valid is True
    assert report.wrapper_blake3 == first["wrapper_blake3"]
    schema = json.loads(
        (ROOT / "schemas/trajectory-derived-rl-sandbox.v1.schema.json").read_text()
    )
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(first)


def test_trajectory_step_reopens_full_run_and_exact_step(registry) -> None:
    inputs = common(registry)
    event_id = uid(12)
    trajectory = {
        "schema": "eva.fixture-full-trajectory.v1",
        "events": [
            {"event_id": uid(11), "role": "user", "content": "task"},
            {"event_id": event_id, "role": "assistant", "content": "call tools"},
        ],
    }
    receipt = {"schema": "eva.fixture-run-receipt.v1", "run_id": uid(10), "terminal": True}
    start = TrajectoryStepStart(
        trajectory_id=uid(9),
        trajectory_blake3=blake3_hex(trajectory),
        run_id=uid(10),
        run_receipt_blake3=blake3_hex(receipt),
        step_id=event_id,
        event_ordinal=1,
        producer_model_id="anthropic/claude-opus-5",
        trajectory_document=trajectory,
        run_receipt_document=receipt,
    )
    document = build_trajectory_rl_sandbox(**inputs, start=start)

    assert document["starting_point"]["trajectory"]["event_ordinal"] == 1
    report = verify_trajectory_rl_sandbox(
        document,
        initial_workspace_files=inputs["initial_workspace_files"],
        evidence_files=inputs["evidence_files"],
        rubric_registry=registry,
        trajectory_document=trajectory,
        run_receipt_document=receipt,
    )
    assert report.valid

    changed_evidence = dict(inputs["evidence_files"])
    changed_evidence["source/case.json"] = b"changed"
    with pytest.raises(TrajectoryRLSandboxError, match="bytes or contract"):
        verify_trajectory_rl_sandbox(
            document,
            initial_workspace_files=inputs["initial_workspace_files"],
            evidence_files=changed_evidence,
            rubric_registry=registry,
            trajectory_document=trajectory,
            run_receipt_document=receipt,
        )


def test_bad_or_unverifiable_trajectory_must_use_explicit_benchmark_fallback(registry) -> None:
    inputs = common(registry)
    event_id = uid(22)
    trajectory = {"events": [{"event_id": event_id, "role": "assistant"}]}
    receipt = {"run_id": uid(20), "error": "launcher defect"}
    invalid = TrajectoryStepStart(
        trajectory_id=uid(19),
        trajectory_blake3=blake3_hex(trajectory),
        run_id=uid(20),
        run_receipt_blake3=blake3_hex(receipt),
        step_id=event_id,
        event_ordinal=0,
        producer_model_id="openai/gpt-5.6-sol",
        trajectory_document=trajectory,
        run_receipt_document=receipt,
        verification_status="launcher_defect",
    )
    with pytest.raises(TrajectoryRLSandboxError, match="benchmark-source fallback"):
        build_trajectory_rl_sandbox(**inputs, start=invalid)

    fallback = BenchmarkSourceStart(
        fallback_reason="launcher_defect",
        excluded_trajectory_id=uid(19),
        excluded_run_id=uid(20),
        excluded_run_receipt_blake3=blake3_hex(receipt),
    )
    document = build_trajectory_rl_sandbox(**inputs, start=fallback)
    excluded = document["starting_point"]["fallback"]["excluded_run"]
    assert excluded["disposition"] == "excluded_from_training_evidence"
    assert verify_trajectory_rl_sandbox(
        document,
        initial_workspace_files=inputs["initial_workspace_files"],
        evidence_files=inputs["evidence_files"],
        rubric_registry=registry,
    ).valid


def test_tampered_rubric_or_workspace_fails_closed(registry) -> None:
    inputs = common(registry)
    document = build_trajectory_rl_sandbox(**inputs, start=BenchmarkSourceStart())
    tampered = deepcopy(document)
    tampered["rubric_table"]["items"][0]["title"] = "not the compiled table"
    with pytest.raises(TrajectoryRLSandboxError, match="bytes or contract"):
        verify_trajectory_rl_sandbox(
            tampered,
            initial_workspace_files=inputs["initial_workspace_files"],
            evidence_files=inputs["evidence_files"],
            rubric_registry=registry,
        )
    with pytest.raises(TrajectoryRLSandboxError, match="bytes or contract"):
        verify_trajectory_rl_sandbox(
            document,
            initial_workspace_files={"case/input.json": b"different"},
            evidence_files=inputs["evidence_files"],
            rubric_registry=registry,
        )


def test_file_tree_reader_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_bytes(b"a")
    assert read_regular_file_tree(root) == {"a.txt": b"a"}
    (root / "link").symlink_to(root / "a.txt")
    with pytest.raises(TrajectoryRLSandboxError, match="symlinks"):
        read_regular_file_tree(root)
