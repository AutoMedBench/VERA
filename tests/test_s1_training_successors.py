from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from eva_agent.codex_pipeline.native_policy_v2 import build_stage_tool_guidance_v1
from eva_agent.pipeline import BenchmarkEpisode, BenchmarkSource, Stage
from eva_agent.training.s1_training_successors import (
    SOURCE_ROOT,
    _runtime_materials,
    _strict_json,
    build_training_successor_bundle,
    provider_free_execute_context,
    verify_training_successor_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
BULK = ROOT / "runs/bulk-rl-sandboxes.v1"
CATALOG = ROOT / "runs/prospective-execution-binding-catalog.v3.r1.json"


def test_signed_source_only_successors_are_new_training_identities() -> None:
    bundle = build_training_successor_bundle(bulk_root=BULK, catalog_path=CATALOG)
    assert bundle["row_count"] == 754
    assert bundle["family_counts"] == {
        "agentclinic": 15,
        "automedbench": 247,
        "healthbench-professional": 249,
        "medxpertqa": 243,
    }
    rows = bundle["rows"]
    assert all(row["controls"]["training_only"] is True for row in rows)
    assert all(
        row["controls"]["production_promotion_claimed"] is False for row in rows
    )
    assert not {
        row[key] for row in rows for key in ("candidate_id", "sandbox_id", "episode_id")
    } & {
        row["predecessor"][key]
        for row in rows
        for key in ("candidate_id", "sandbox_id", "episode_id")
    }
    assert verify_training_successor_bundle(
        bundle, bulk_root=BULK, catalog_path=CATALOG
    ) == bundle["bundle_blake3"]


def test_training_successor_exact_s1_frontier_materializes_workspace(
    tmp_path: Path,
) -> None:
    descriptor = build_training_successor_bundle(
        bulk_root=BULK, catalog_path=CATALOG
    )["rows"][0]
    construction = _strict_json(
        SOURCE_ROOT / descriptor["public_materials"]["construction_input_relative_path"]
    )
    runtime_context, registry = _runtime_materials(descriptor, construction)
    guidance = build_stage_tool_guidance_v1(
        public_runtime_context=runtime_context,
        source_tool_catalog=registry.public_schemas(),
    )
    episode = BenchmarkEpisode(
        episode_id=descriptor["episode_id"],
        source=BenchmarkSource("fixture", "fixture.json", "v1"),
        domain=descriptor["domain"],
        stage=Stage.S1,
        instruction="Materialize the exact public S1 plan.",
        policy_context={"execution_binding": runtime_context},
        initial_files={"TASK.md": b"training-only S1 successor\n"},
    )
    checked = provider_free_execute_context(
        context=SimpleNamespace(
            episode=episode,
            tool_registry=registry,
            stage_tool_guidance=guidance,
        ),
        workspace_root=tmp_path,
        sandbox_id=descriptor["sandbox_id"],
    )
    assert checked["external_provider_calls"] == 0
    assert checked["gate_passed"] is True
    assert checked["workspace_before_blake3"] != checked["workspace_after_blake3"]
