import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from training.automedbench_lite import adapter


@pytest.fixture
def assets(tmp_path):
    root = tmp_path / "assets"
    public, scorer = root / "public-release", root / "scorer-only-release"
    scorer.mkdir(parents=True, mode=0o700)
    public.mkdir(mode=0o700)
    contents = {"automedbench_release/" + name: b"# synthetic source\n" for name in ("__init__.py", "raw_track_worker.py", "tasks.py")}
    for case in adapter.CASES:
        input_path, config_path, private = adapter.case_paths(case)
        contents[input_path] = b"synthetic public image fixture"
        config = {"task_id": case["task_id"], "classes": ["actinic_keratoses"], "score_metric": case["metric"],
                  "tissue_labels": {1: "one", 10: "ten", 2: "two"}, "gt_subdir": "DO_NOT_COPY_PRIVATE_SETTING"}
        contents[config_path] = yaml.safe_dump(config).encode()
        prefix = f"benchmarks/AutoMedBench-{case['track']}/{case['eval']}/"
        scorer_name = {"classification": "acc_scorer.py", "detection": "det2d_scorer.py", "segmentation": "dice_scorer.py"}[case["track"]]
        for name in ("format_checker.py", "aggregate.py", scorer_name):
            contents[prefix + name] = b"# synthetic source\n"
        suffix = "masks/" + case["case_id"] + "/a.nii.gz" if case["track"] == "segmentation" else case["case_id"] + "/" + ("label.json" if case["track"] == "classification" else "boxes.json")
        contents[private + suffix] = b"SYNTHETIC_HIDDEN_SENTINEL"
    inventory = []
    for relative, data in contents.items():
        is_private = "/private/" in relative
        for target in ([scorer] if is_private else [scorer, public]):
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        inventory.append({"path": relative, "bytes": len(data), "blake3": adapter.blake3(data).hexdigest(), "scorer_only": is_private})
    adapter.write_once(root / "download-first3.json", {"schema": "eva.automedbench-lite-pinned-assets.v1",
        "revision": adapter.REVISION, "repository": adapter.REPOSITORY, "inventory": inventory})
    return adapter.load_assets(root)


def test_three_public_workspaces_are_unique_and_gold_free(assets, tmp_path):
    assert adapter.preflight(assets)["ready"]
    run = adapter.prepare_run(assets, tmp_path / "runs")
    manifest = adapter.read_document(run / "run-manifest.json")
    assert manifest["unique_case_count"] == 3 and manifest["planned_rollouts_per_case"] == 1
    assert len({row["workspace_id"] for row in manifest["cases"]}) == 3
    assert manifest["model_evaluation_complete"] is False
    for case in manifest["cases"]:
        workspace = run / case["workspace_relative"]
        contract = adapter.read_document(workspace / "task.json")
        assert contract["case_id"] == case["case_id"]
        for path in workspace.rglob("*"):
            if path.is_file():
                data = path.read_bytes()
                assert b"SYNTHETIC_HIDDEN_SENTINEL" not in data
                assert b"DO_NOT_COPY_PRIVATE_SETTING" not in data
                assert b"scorer-only-release" not in data
        assert set(p.name for p in workspace.iterdir()) == {"inputs", "outputs", "task.json"}


def test_rejects_asset_tampering_private_disclosure_and_symlinks(assets, tmp_path):
    private = next(path for path, row in assets.inventory.items() if row["scorer_only"])
    with pytest.raises(adapter.EvaluationError, match="allowlisted"):
        assets.public_file(private)
    path = adapter.case_paths(adapter.CASES[0])[0]
    (assets.public / path).write_bytes(b"tampered")
    with pytest.raises(adapter.EvaluationError, match="commitment"):
        adapter.preflight(assets)
    root = tmp_path / "unsafe"
    root.mkdir()
    (root / "link").symlink_to(assets.scorer)
    with pytest.raises(adapter.EvaluationError, match="symlink"):
        adapter.safe_file(root, "link/any")
    with pytest.raises(adapter.EvaluationError, match="unsafe"):
        adapter.safe_file(root, "../outside")


def native_result():
    return {"status": "scored", "track": "classification", "case_id": "ISIC_00000001", "metric": "balanced_accuracy",
            "task_score_0_1": 0.0, "present_outputs": 1, "valid_outputs": 1, "output_format_valid": True,
            "network_disabled": True, "private_values_exported": False}


@pytest.mark.parametrize("change", ({"task_score_0_1": float("nan")}, {"task_score_0_1": True},
                                  {"metric": "accuracy"}, {"hidden_gold": "sentinel"}, {"valid_outputs": 0}))
def test_native_result_rejects_nonfinite_wrong_metric_or_extra_gold(change):
    with pytest.raises(adapter.EvaluationError):
        adapter.validate_result({**native_result(), **change}, adapter.CASES[0])


def test_scoring_is_single_attempt_and_failure_has_no_reward(assets, tmp_path, monkeypatch):
    run = adapter.prepare_run(assets, tmp_path / "runs")
    manifest = adapter.read_document(run / "run-manifest.json")
    case = manifest["cases"][0]
    output = run / case["workspace_relative"] / "outputs/prediction.json"
    output.write_text('{"label":"actinic_keratoses"}')
    observed = {}
    def fail(*args, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(returncode=1, stdout=b"PRIVATE_SERVICE_FAILURE_MUST_NOT_ESCAPE")
    monkeypatch.setattr(adapter.subprocess, "run", fail)
    with pytest.raises(adapter.EvaluationError, match="no_fallback"):
        adapter.score_case(assets, run, case["case_id"], Path("/test/python"))
    assert observed["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert "PRIVATE_SERVICE_FAILURE" not in (run / "native-scores" / case["case_id"] / "failure.json").read_text()
    failure = adapter.read_document(run / "native-scores" / case["case_id"] / "failure.json")
    assert failure["reward"] is None
    with pytest.raises(FileExistsError):
        adapter.score_case(assets, run, case["case_id"], Path("/test/python"))


def test_score_is_exact_native_projection_and_not_agent_judgment(assets, tmp_path, monkeypatch):
    run = adapter.prepare_run(assets, tmp_path / "runs", diagnostic=True)
    case = adapter.read_document(run / "run-manifest.json")["cases"][0]
    (run / case["workspace_relative"] / "outputs/prediction.json").write_text('{"label":"actinic_keratoses"}')
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(native_result()).encode()))
    score = adapter.score_case(assets, run, case["case_id"], Path("/test/python"))
    assert score["native_result"] == native_result()
    assert score["agent_judged"] is False and score["policy_rollout_provenance_verified"] is False
    assert score["diagnostic_only"] is True
