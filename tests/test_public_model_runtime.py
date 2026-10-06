"""CPU-only public model boundary and archive regressions."""
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
from unittest.mock import patch

from blake3 import blake3
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/benchmark_models"))
from prepare_public_models import HF_MODELS, safe_zip_members
from run_prescribed_model import confined, preflight, main
import run_prescribed_model as runtime


def fixture_binding(tmp_path):
    workspace = tmp_path / "actor"
    workspace.mkdir()
    image = workspace / "inputs/case1/image.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"public fixture")
    (workspace / "task.json").write_text(json.dumps({"track": "classification", "no_private_reference_access": True, "case_ids": ["case1"], "coding_orchestrator": "fixture"}))
    (workspace / "inputs-manifest.json").write_text(json.dumps({"case_ids": ["case1"], "files": [{"path": "inputs/case1/image.jpg", "bytes": 14, "blake3": blake3(image.read_bytes()).hexdigest()}]}))
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_bytes(b"{}")
    repo, revision, _ = HF_MODELS["classification"]
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"models": {"classification": {"repository": repo, "revision": revision, "path": str(model_path), "files": [{"path": "config.json", "bytes": 2, "blake3": blake3(b"{}").hexdigest()}]}}}))
    return workspace, manifest


def test_exact_public_binding(tmp_path):
    workspace, manifest = fixture_binding(tmp_path)
    assert list(preflight(workspace, manifest, "classification", ["case1"])["selected"]) == ["case1"]


@pytest.mark.parametrize("cases", [[], ["case1", "case1"], ["private"], ["../private"]])
def test_case_ids_fail_closed(tmp_path, cases):
    workspace, manifest = fixture_binding(tmp_path)
    with pytest.raises(ValueError):
        preflight(workspace, manifest, "classification", cases)


def test_changed_input_or_model_rejected(tmp_path):
    workspace, manifest = fixture_binding(tmp_path)
    (workspace / "inputs/case1/image.jpg").write_bytes(b"changed bytes!")
    with pytest.raises(ValueError, match="commitment"):
        preflight(workspace, manifest, "classification", ["case1"])


def test_path_and_symlink_escape_rejected(tmp_path):
    actor = tmp_path / "actor"
    actor.mkdir()
    (actor / "escape").symlink_to(tmp_path, target_is_directory=True)
    for value in ("../private", "/private", "escape/private"):
        with pytest.raises(ValueError):
            confined(actor, value)


def test_archive_ignores_only_inert_macos_sidecars():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("Dataset291/weights.pth", b"public")
        archive.writestr("__MACOSX/._Dataset291", b"sidecar")
    with zipfile.ZipFile(buffer) as archive:
        assert [m.filename for m in safe_zip_members(archive, "Dataset291")] == ["Dataset291/weights.pth"]


@pytest.mark.parametrize("path", ["../escape", "/escape", "Other/weights.pth"])
def test_archive_escape_rejected(path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(path, b"x")
    with zipfile.ZipFile(buffer) as archive, pytest.raises(ValueError):
        safe_zip_members(archive, "Dataset291")


def test_durable_queued_state_precedes_gpu_lock_and_error_is_retained(tmp_path):
    workspace, manifest = fixture_binding(tmp_path)
    audit = tmp_path / "host-audit"
    job_id = "3c3a797f-5926-48fb-bc3b-2ff64036d98f"
    receipt = audit / job_id / "receipt.json"
    argv = ["run_prescribed_model.py", "--workspace", str(workspace), "--model-manifest", str(manifest),
            "--track", "classification", "--case-ids", "case1", "--execute-gpu", "--job-id", job_id,
            "--audit-root", str(audit)]
    def observe_queue(*args):
        assert json.loads(receipt.read_text())["status"] == "queued"
        raise RuntimeError("fixture stops before GPU initialization")
    with patch("sys.argv", argv), patch("run_prescribed_model.fcntl.flock", side_effect=observe_queue):
        with pytest.raises(RuntimeError, match="fixture"):
            main()
    assert json.loads(receipt.read_text())["status"] == "failed"
    assert (audit / job_id / "executed-helper.py").is_file()


@pytest.mark.parametrize("track", ["segmentation", "enhancement", "synthesis"])
def test_new_track_exact_asset_kind_preflight(tmp_path, monkeypatch, track):
    workspace, manifest = fixture_binding(tmp_path)
    task = json.loads((workspace / "task.json").read_text())
    task["track"] = track
    (workspace / "task.json").write_text(json.dumps(task))
    path = tmp_path / "model"
    if track == "segmentation":
        mod = runtime.segmentation
        model = {"repository": mod.REPOSITORY, "revision": mod.REVISION, "release_id": 121996387,
                 "task_ids": list(mod.PARTITIONS), "folds": [0], "resample_mm": 1.5, "fast": False}
        names = [f"{part}/{mod.PLAN_DIRECTORY}/{name}" for part in mod.PARTITIONS.values()
                 for name in ("dataset.json", "plans.json", "fold_0/checkpoint_final.pth")]
    elif track == "enhancement":
        mod = runtime.enhancement
        model = {"repository": mod.REPOSITORY, "revision": mod.REVISION}
        names = [mod.WEIGHT_FILE]
    else:
        mod = runtime.synthesis_runtime
        model = {"repository": mod.REPOSITORY, "revision": mod.REVISION}
        names = [mod.SOURCE_FILE, mod.WEIGHT_FILE]
        monkeypatch.setattr(mod, "PINNED_FILES", {name: (7, blake3(b"fixture").hexdigest()) for name in names})
    for name in names:
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"fixture")
    model.update({"path": str(path), "files": [{"path": name, "bytes": 7, "blake3": blake3(b"fixture").hexdigest()} for name in names]})
    manifest.write_text(json.dumps({"models": {track: model}}))
    assert preflight(workspace, manifest, track, ["case1"])["model"]["revision"] == mod.REVISION
    model["revision"] = "main"
    manifest.write_text(json.dumps({"models": {track: model}}))
    with pytest.raises(ValueError, match="identity"):
        preflight(workspace, manifest, track, ["case1"])


@pytest.mark.parametrize("track,module_name,function", [
    ("segmentation", "segmentation", "segment"), ("enhancement", "enhancement", "enhance"),
    ("synthesis", "synthesis_runtime", "run_synthesis"), ("report", "report_runtime", "run_report"),
    ("vqa", "vqa_runtime", "run_vqa")])
def test_dispatch_passes_bound_output_and_explicit_controls(monkeypatch, tmp_path, track, module_name, function):
    captured = {}
    def helper(binding, torch, **kwargs):
        captured.update(kwargs)
        return ["fixture"]
    monkeypatch.setattr(getattr(runtime, module_name), function, helper)
    callback = object()
    result = runtime.run_track({"task": {"track": track}, "assets": {}, "cache_root": tmp_path}, object(), output_root=tmp_path,
        confidence=.25, sigma=.04, hu_min=-1000, hu_max=3000, progress=callback,
        prompt="Generate the findings section.", max_new_tokens=64, multi_image_mode="montage")
    assert result == ["fixture"] and captured["output_root"] == tmp_path and captured["progress"] is callback
    if track == "enhancement":
        assert (captured["sigma"], captured["hu_min"], captured["hu_max"]) == (.04, -1000, 3000)
    if track in {"report", "vqa"}:
        assert captured["max_new_tokens"] == 64 and captured["cache_root"] == tmp_path
        assert captured["prompt"] == "Generate the findings section." if track == "report" else captured["multi_image_mode"] == "montage"


def test_cli_requires_enhancement_controls_and_rejects_them_elsewhere(tmp_path, monkeypatch):
    base = ["run_prescribed_model.py", "--workspace", str(tmp_path), "--model-manifest", str(tmp_path / "none.json"), "--case-ids", "case1"]
    for extra in (["--track", "enhancement"], ["--track", "classification", "--sigma", ".05"]):
        monkeypatch.setattr(sys, "argv", base + extra)
        with pytest.raises(ValueError):
            runtime.main()


def test_array_artifacts_copied_to_independent_host_audit(tmp_path):
    workspace = tmp_path / "actor"
    output = workspace / "outputs/agents_outputs/prescribed-model-jobs/job"
    output.mkdir(parents=True)
    source = output / "case1/enhanced.npy"
    source.parent.mkdir()
    source.write_bytes(b"actual-public-model-fixture")
    audit = tmp_path / "audit"
    result = {"case_id": "case1", "candidate_output": {"path": str(source.relative_to(workspace)),
        "bytes": source.stat().st_size, "blake3": blake3(source.read_bytes()).hexdigest()}}
    rows = runtime.retain_case_artifacts({"workspace": workspace, "task": {"track": "enhancement"}}, output, audit, result)
    source.write_bytes(b"actor-later-edit")
    assert (audit / rows[0]["path"]).read_bytes() == b"actual-public-model-fixture"
    assert (audit / rows[0]["path"]).stat().st_mode & 0o777 == 0o600


def test_complete_mocked_job_retains_helper_sources_and_actual_artifacts(tmp_path, monkeypatch):
    workspace, manifest = fixture_binding(tmp_path)
    binding = preflight(workspace, manifest, "classification", ["case1"])
    binding["task"].update({"track": "enhancement", "public_config": {"intensity_range": [-1024, 3000]}})
    monkeypatch.setattr(runtime, "preflight", lambda *a: binding)
    monkeypatch.setattr(runtime, "GPU_LOCK", tmp_path / "mock-gpu.lock")
    monkeypatch.setattr(runtime.fcntl, "flock", lambda *a: None)
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda *a: "cpu-fixture")
    fake_cuda = SimpleNamespace(mem_get_info=lambda: (100 * 1024**3, 200 * 1024**3),
        set_per_process_memory_fraction=lambda *a: None, get_device_name=lambda: "not-a-real-GPU", synchronize=lambda: None,
        max_memory_allocated=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda, set_num_threads=lambda *a: None))
    def run(binding, torch, **kwargs):
        assert kwargs["sigma"] == .05 and kwargs["hu_max"] == 3072
        path = kwargs["output_root"] / "case1/enhanced.npy"
        path.parent.mkdir()
        path.write_bytes(b"synthetic-model-output")
        row = {"case_id": "case1", "candidate_output": {"path": str(path.relative_to(workspace)),
            "bytes": path.stat().st_size, "blake3": blake3(path.read_bytes()).hexdigest()}}
        kwargs["progress"](row)
        return [row]
    monkeypatch.setattr(runtime, "run_track", run)
    job = "3c3a797f-5926-48fb-bc3b-2ff64036d98f"
    audit = tmp_path / "audit"
    monkeypatch.setattr(sys, "argv", ["runner", "--workspace", str(workspace), "--model-manifest", str(manifest),
        "--track", "enhancement", "--case-ids", "case1", "--sigma", ".05", "--hu-min", "-1024", "--hu-max", "3072",
        "--audit-root", str(audit), "--job-id", job, "--execute-gpu"])
    runtime.main()
    receipt = json.loads((audit / job / "receipt.json").read_text())
    assert receipt["status"] == "complete" and receipt["gpu_peak_allocated_bytes"] == 0
    assert receipt["inference_completed"] is True
    assert receipt["execution_outcome"] == {"kind": "trusted_python_function_return", "success": True, "os_process_exit_observed": False}
    assert {Path(row["path"]).name for row in receipt["executed_module_sources"]} == {"prepare_public_models.py", "enhancement.py", "_track_io.py"}
    assert len(receipt["host_retained_artifacts"]) == 1


@pytest.mark.parametrize("extra", [
    ["--track", "report", "--max-new-tokens", "64"],
    ["--track", "report", "--prompt", "Generate findings", "--max-new-tokens", "513"],
    ["--track", "report", "--prompt", "Generate findings", "--max-new-tokens", "64", "--multi-image-mode", "montage"],
    ["--track", "vqa", "--max-new-tokens", "64"],
    ["--track", "vqa", "--max-new-tokens", "257", "--multi-image-mode", "montage"],
    ["--track", "classification", "--max-new-tokens", "64"],
])
def test_cli_generative_controls_are_explicit_and_track_specific(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["runner", "--workspace", str(tmp_path), "--model-manifest", str(tmp_path / "none.json"),
        "--case-ids", "case1", *extra])
    with pytest.raises(ValueError): runtime.main()


@pytest.mark.parametrize("track,overlay", [("report", "chexagent-transformers440"), ("vqa", "llava-transformers436")])
def test_dedicated_worker_overlay_and_honest_memory_profiles(tmp_path, monkeypatch, track, overlay):
    path = tmp_path / "runtimes" / overlay / "transformers/__init__.py"
    path.parent.mkdir(parents=True)
    path.write_text("# inert CPU fixture")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(runtime.os, "environ", dict(runtime.os.environ))
    for name in ("transformers", "tokenizers", "huggingface_hub", "accelerate"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    binding = {"task": {"track": track}, "cache_root": tmp_path, "assets": {"vqa_source": {"path": str(tmp_path / "source")}}}
    result = runtime.configure_overlay(binding, tmp_path / "host-audit")
    assert result["offline"] and sys.path[0] == str(path.parent.parent)
    assert runtime.os.environ["HF_HUB_OFFLINE"] == "1" and runtime.os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert runtime.os.environ["HF_MODULES_CACHE"] == str(tmp_path / "host-audit/hf-modules")
    assert runtime.GPU_PROFILES_GIB[track] == ((12, 24) if track == "report" else (24, 48))
    monkeypatch.setitem(sys.modules, "transformers", object())
    with pytest.raises(RuntimeError, match="fresh isolated"):
        runtime.configure_overlay(binding, tmp_path / "another-audit")


def test_retains_actual_generative_request_montage_and_smoke_bytes(tmp_path):
    workspace = tmp_path / "actor"
    output = workspace / "outputs/job"
    output.mkdir(parents=True)
    result = {"case_id": "MM-0"}
    for key in ("raw_output", "candidate_output", "request", "model_input", "smoke_output"):
        path = output / (key + ".fixture")
        path.write_bytes(key.encode())
        result[key] = {"path": str(path.relative_to(workspace)), "bytes": len(key), "blake3": blake3(key.encode()).hexdigest()}
    rows = runtime.retain_case_artifacts({"workspace": workspace, "task": {"track": "vqa"}}, output, tmp_path / "audit", result)
    assert {row["kind"] for row in rows} == {"raw_output", "candidate_output", "request", "model_input", "smoke_output"}
    assert all(row["case_id"] == "MM-0" for row in rows)
