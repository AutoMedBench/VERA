from __future__ import annotations

from pathlib import Path

from eva_agent.sandboxes.rollout_reuse_v1 import (
    ReusableStart,
    map_reusable_starts,
    overlay_root,
)


def _start(*, receipt: str, family: str, stage: str, disposition: str) -> ReusableStart:
    return ReusableStart(
        source_receipt_path=f"{receipt}/rollout-receipt.json",
        source_sandbox_id=f"source-{receipt}",
        source_rollout_id=f"rollout-{receipt}",
        source_family=family,
        stage=stage,
        producer_model_id="aws/anthropic/bedrock-claude-opus-5",
        disposition=disposition,
        score=1.0 if disposition == "high_score_success" else 0.0,
        transcript_path=f"{receipt}/transcript-private.json",
        prefix_message_count=4,
        next_assistant_message_index=4,
        workspace_paths=(f"{receipt}/workspace/work.json",),
        evidence_paths=(f"{receipt}/evidence.json",),
        qualification_paths=(f"{receipt}/rollout-receipt.json",),
    )


def test_sparse_mapping_preserves_base_count_and_uses_each_target_once() -> None:
    base = [
        {
            "sandbox_id": f"sandbox-{ordinal}",
            "candidate_id": f"candidate-{ordinal}",
            "queue_ordinal": ordinal,
            "source_family": "automedbench",
            "stage": "S2",
        }
        for ordinal in (9, 2, 5)
    ]
    starts = (
        _start(
            receipt="good", family="automedbench", stage="S2",
            disposition="high_score_success",
        ),
        _start(
            receipt="partial", family="automedbench", stage="S2",
            disposition="valid_intermediate_failure",
        ),
        _start(
            receipt="unmatched", family="medxpertqa", stage="S4",
            disposition="high_score_success",
        ),
    )
    rows = map_reusable_starts(starts, base)
    assert len(base) == 3
    assert [row["target_queue_ordinal"] for row in rows] == [2, 5]
    assert len({row["target_sandbox_id"] for row in rows}) == 2
    assert {row["disposition"] for row in rows} == {
        "high_score_success", "valid_intermediate_failure"
    }
    assert all(row["base_fallback_replaced"] is True for row in rows)


def test_single_overlay_root_covers_shards_and_referenced_source_bytes(tmp_path: Path) -> None:
    (tmp_path / "source").mkdir()
    (tmp_path / "source/a.json").write_bytes(b'{"a":1}\n')
    core = {"schema": "fixture", "new_blake3_commitments_created": 1}
    first = overlay_root(
        core,
        shard_payloads=(b'{"row":1}\n',),
        source_root=tmp_path,
        source_paths=("source/a.json",),
    )
    second = overlay_root(
        core,
        shard_payloads=(b'{"row":1}\n',),
        source_root=tmp_path,
        source_paths=("source/a.json",),
    )
    assert first == second
    (tmp_path / "source/a.json").write_bytes(b'{"a":2}\n')
    assert first != overlay_root(
        core,
        shard_payloads=(b'{"row":1}\n',),
        source_root=tmp_path,
        source_paths=("source/a.json",),
    )
