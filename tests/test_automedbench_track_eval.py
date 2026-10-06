from pathlib import Path

import pytest

from training.automedbench_lite.adapter import EvaluationError, REVISION, REPOSITORY, blake3, write_once
from training.automedbench_lite.track_adapter import TRACKS, TrackRelease, prepare_track_run, public_output_contract


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "source"
    public, private = root / "public-release", root / "scorer-only-release"
    public.mkdir(parents=True, mode=0o700)
    private.mkdir(mode=0o700)
    track = TRACKS[0]
    inventory = []
    for index in range(track.count):
        relative = track.public_prefix + f"ISIC_{index:08d}/image.jpg"
        path = public / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = f"synthetic public input {index}".encode()
        path.write_bytes(data)
        inventory.append({"path": relative, "bytes": len(data), "blake3": blake3(data).hexdigest(), "scorer_only": False})
    relative = track.public_prefix.replace("/public/", "/private/") + "private.json"
    inventory.append({"path": relative, "bytes": 6, "blake3": blake3(b"hidden").hexdigest(), "scorer_only": True})
    receipt = root / "download-all3.json"
    write_once(receipt, {"schema": "eva.automedbench-lite-pinned-assets.v1", "revision": REVISION,
                        "repository": REPOSITORY, "inventory": inventory})
    return TrackRelease(receipt)


def test_protocol_is_seven_track_workflows_not_2385_policy_rollouts():
    assert len(TRACKS) == 7
    assert sum(track.count for track in TRACKS) == 2385
    report = next(track for track in TRACKS if track.name == "report")
    assert "whole-track TaskScore to zero" in public_output_contract(report)["whole_track_rule"]
    vqa = next(track for track in TRACKS if track.name == "vqa")
    assert "raw_model_output" in public_output_contract(vqa)["required"]
    assert "LLaVA-Med" in public_output_contract(vqa)["prediction_rule"]


def test_full_public_subset_is_required_and_private_is_never_offered(release):
    cases, files = release.case_inputs(TRACKS[0])
    assert len(cases) == len(files) == 100
    assert all("private" not in row["source_path"] for row in files)
    private = next(relative for relative, row in release.inventory.items() if row["scorer_only"])
    with pytest.raises(EvaluationError, match="not_declared"):
        release.public_file(private)
    release.inventory.pop(files[0]["source_path"])
    with pytest.raises(EvaluationError, match="complete_public_subset"):
        release.case_inputs(TRACKS[0])


def test_public_commitment_tampering_fails(release):
    _, files = release.case_inputs(TRACKS[0])
    relative = files[0]["source_path"]
    (release.public / relative).write_bytes(b"changed")
    with pytest.raises(EvaluationError, match="commitment"):
        release.public_file(relative)


def test_incomplete_track_scope_cannot_be_prepared_as_full_run(release, tmp_path):
    with pytest.raises(EvaluationError, match="exact_seven"):
        prepare_track_run({"classification": release}, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()
