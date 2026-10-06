"""CPU mocks only: no real Torch/TotalSeg/deepinv import, checkpoint load, or GPU.

Numerical tests require numpy/nibabel in the public-model environment; the
repository's lean environment skips them explicitly, not as runtime success.
"""
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from training.benchmark_models import enhancement as enh, segmentation as seg
from training.benchmark_models._track_io import case_output, public_input


def model_fixture(tmp_path, track):
    root = tmp_path / "models" / track
    root.mkdir(parents=True)
    if track == "enhancement":
        model = {"repository": enh.REPOSITORY, "revision": enh.REVISION}
        names = [enh.WEIGHT_FILE]
    else:
        model = {"repository": seg.REPOSITORY, "revision": seg.REVISION, "release_id": 121996387,
                 "task_ids": list(seg.PARTITIONS), "folds": [0], "resample_mm": 1.5, "fast": False}
        names = [f"{part}/{seg.PLAN_DIRECTORY}/{name}" for part in seg.PARTITIONS.values()
                 for name in ("dataset.json", "plans.json", "fold_0/checkpoint_final.pth")]
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-not-model-weights")
    return {**model, "path": str(root), "files": [{"path": name} for name in names]}


def binding_fixture(tmp_path, track):
    workspace = tmp_path / "actor"
    (workspace / "inputs/case1").mkdir(parents=True)
    cfg = {"task_id": "ldct-denoising-task"} if track == "enhancement" else {
        "task_id": "tsg-multiorgan-seg-task", "num_foreground_classes": 117,
        "tissue_labels": {str(i): f"tissue_{i:03}" for i in range(1, 118)}}
    filename = "input.npy" if track == "enhancement" else "ct.nii.gz"
    return {"workspace": workspace, "task": {"track": track, "public_config": cfg, "no_private_reference_access": True},
            "selected": {"case1": [{"path": f"inputs/case1/{filename}"}]}, "model": model_fixture(tmp_path, track)}


def numerical():
    return pytest.importorskip("numpy"), pytest.importorskip("nibabel")


def test_import_is_inert_and_supports_direct_runner_style():
    root = Path(__file__).resolve().parents[1]
    command = "import sys; from training.benchmark_models import enhancement,segmentation; assert 'torch' not in sys.modules; assert 'deepinv' not in sys.modules; assert 'totalsegmentator' not in sys.modules"
    subprocess.run([sys.executable, "-c", command], cwd=root, check=True)
    command = "import sys; sys.path.insert(0,'training/benchmark_models'); import enhancement,segmentation; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", command], cwd=root, check=True)


@pytest.mark.parametrize("track,module", [("enhancement", enh), ("segmentation", seg)])
def test_fixed_model_identity_and_inventory(tmp_path, track, module):
    model = model_fixture(tmp_path, track)
    assert module.validate_model(model).exists()
    with pytest.raises(ValueError, match="identity"):
        module.validate_model({**model, "revision": "main"})
    with pytest.raises(ValueError):
        module.validate_model({**model, "files": []})


def test_segmentation_rejects_fast_and_alternate_folds(tmp_path):
    model = model_fixture(tmp_path, "segmentation")
    for changed in ({"fast": True}, {"folds": [0, 1, 2, 3, 4]}, {"task_ids": [297]}):
        with pytest.raises(ValueError, match="contract"):
            seg.validate_model({**model, **changed})


def test_output_and_input_paths_remain_public_and_no_overwrite(tmp_path):
    binding = binding_fixture(tmp_path, "enhancement")
    workspace = binding["workspace"]
    private = tmp_path / "private.npy"
    private.write_bytes(b"unread sentinel")
    (workspace / "inputs/case1/input.npy").symlink_to(private)
    with pytest.raises(ValueError, match="escapes"):
        public_input(binding, "case1", "input.npy")
    with pytest.raises(ValueError, match="outputs"):
        case_output(binding, tmp_path / "outside", "case1")
    case_output(binding, workspace / "outputs/job", "case1")
    with pytest.raises(FileExistsError):
        case_output(binding, workspace / "outputs/job", "case1")


def test_label_remap_uses_names_not_integer_identity():
    native = {i: f"tissue_{i:03}" for i in range(1, 118)}
    benchmark = {str(i): native[118 - i] for i in range(1, 118)}
    mapping = seg.label_remap(native, benchmark)
    assert mapping[0] == 0 and mapping[1] == 117 and mapping[117] == 1
    for invalid in ({**benchmark, "1": "unknown"}, {k: v for k, v in benchmark.items() if k != "117"}):
        with pytest.raises(ValueError):
            seg.label_remap(native, invalid)


def test_geometry_and_stored_label_values_are_not_repaired():
    np, nib = numerical()
    reference = nib.Nifti1Image(np.zeros((2, 3, 4), np.float32), np.diag([1.5, 2., 3., 1.]))
    mapping = {i: 0 if i == 0 else 118 - i for i in range(118)}
    prediction = nib.Nifti1Image(np.ones(reference.shape, np.uint8), reference.affine)
    output = seg.remap_prediction(prediction, reference, mapping)
    assert output.get_data_dtype() == np.dtype("uint8")
    assert np.all(np.asanyarray(output.dataobj) == 117)
    for data in (np.full(reference.shape, 1.5), np.full(reference.shape, np.nan), np.full(reference.shape, 118.)):
        with pytest.raises(ValueError, match="class IDs"):
            seg.remap_prediction(nib.Nifti1Image(data, reference.affine), reference, mapping)
    changed = reference.affine.copy()
    changed[0, 3] = 3.
    with pytest.raises(ValueError, match="affine"):
        seg.remap_prediction(nib.Nifti1Image(np.ones(reference.shape), changed), reference, mapping)
    prediction.set_qform(changed, 1)
    with pytest.raises(ValueError, match="Contradictory"):
        seg.validate_geometry(prediction, reference)


def test_offline_total_disables_downloads_and_telemetry(tmp_path, monkeypatch):
    model = model_fixture(tmp_path, "segmentation")
    weights = seg.validate_model(model)
    original_download = object()
    original_stats = object()
    api = SimpleNamespace(download_pretrained_weights=original_download, send_usage_stats=original_stats,
        get_task_config=lambda *a, **k: {"task_id": list(seg.PARTITIONS), "folds": [0], "resample": 1.5},
        totalsegmentator=object())
    package = ModuleType("totalsegmentator")
    package.python_api = api
    mapping_module = ModuleType("totalsegmentator.map_to_binary")
    mapping_module.class_map = {"total": {i: f"t{i}" for i in range(1, 118)}}
    nnunet = ModuleType("nnunetv2")
    nnunet.paths = SimpleNamespace(nnUNet_results=str(weights))
    for name, module in (("totalsegmentator", package), ("totalsegmentator.map_to_binary", mapping_module), ("nnunetv2", nnunet)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(seg, "version", lambda _: "2.17.0")
    monkeypatch.setenv("TOTALSEG_HOME_DIR", "original-fixture-path")
    cache = tmp_path / "job-cache"
    with seg._offline_total(weights, cache):
        import os
        assert os.environ["TOTALSEG_WEIGHTS_PATH"] == str(weights)
        assert json.loads((cache / "config.json").read_text())["send_usage_stats"] is False
        assert api.download_pretrained_weights(291) is None
        with pytest.raises(RuntimeError, match="downloads forbidden"):
            api.download_pretrained_weights(297)
        assert api.send_usage_stats(None) is None
    assert api.download_pretrained_weights is original_download and api.send_usage_stats is original_stats
    assert os.environ["TOTALSEG_HOME_DIR"] == "original-fixture-path"


@pytest.mark.parametrize("values", [(float("nan"), -1024, 3072), (0, -1024, 3072), (.3, -1024, 3072), (True, -1024, 3072), (.05, 1, 1)])
def test_enhancement_requires_bounded_explicit_controls(values):
    with pytest.raises(ValueError):
        enh.validate_controls(*values)


def test_enhancement_hu_conversion_obeys_native_range_not_example_range():
    np, _ = numerical()
    array = np.full((512, 512), 3072, dtype=np.float32)
    x = enh.normalize_hu(array, -1024, 3072)
    assert np.all(x == 1)
    candidate = enh.restore_hu(x, -1024, 3072)
    assert candidate.dtype == np.dtype("float32") and np.all(candidate == 3000)
    assert np.all(enh.restore_hu(np.full_like(x, -1), -1024, 3072) == -1024)
    with pytest.raises(ValueError):
        enh.normalize_hu(array.astype(np.float64), -1024, 3072)
    with pytest.raises(ValueError):
        enh.restore_hu(np.full_like(x, np.nan), -1024, 3072)


def test_drunet_strict_local_weights_only_load(tmp_path, monkeypatch):
    events = {}
    class Model:
        def __init__(self, **kwargs): events["constructor"] = kwargs
        def load_state_dict(self, state, strict): events["strict"] = strict
        def eval(self): return self
        def to(self, device): events["device"] = device; return self
    module = ModuleType("deepinv.models")
    module.DRUNet = Model
    monkeypatch.setitem(sys.modules, "deepinv", ModuleType("deepinv"))
    monkeypatch.setitem(sys.modules, "deepinv.models", module)
    monkeypatch.setattr(enh, "version", lambda _: "0.4.0")
    def load(path, **kwargs): events["load"] = kwargs; return {}
    enh._load_model(tmp_path / "synthetic.pth", SimpleNamespace(load=load))
    assert events == {"constructor": {"in_channels": 1, "out_channels": 1, "pretrained": None, "dim": 2},
                      "load": {"map_location": "cpu", "weights_only": True}, "strict": True, "device": "cuda"}


def test_real_enhancement_branch_with_fake_tensor_model(tmp_path, monkeypatch):
    np, _ = numerical()
    binding = binding_fixture(tmp_path, "enhancement")
    np.save(binding["workspace"] / "inputs/case1/input.npy", np.zeros((512, 512), np.float32))
    events = {}
    class Tensor:
        def __init__(self, value): self.value = value
        def unsqueeze(self, axis): return Tensor(np.expand_dims(self.value, axis))
        def to(self, device): assert device == "cuda"; return self
        def detach(self): return self
        def float(self): return self
        def cpu(self): return self
        def numpy(self): return self.value
    def fake_model(x, sigma): events["sigma"] = sigma; return Tensor(x.value + .1)
    monkeypatch.setattr(enh, "_load_model", lambda *a: fake_model)
    progress = []
    results = enh.enhance(binding, SimpleNamespace(from_numpy=Tensor, inference_mode=nullcontext),
        output_root=binding["workspace"] / "outputs/job", sigma=.05, hu_min=-1024, hu_max=3072, progress=progress.append)
    assert results == progress and events["sigma"] == .05
    row = results[0]
    assert not row["final_submission_written"] and not row["private_reference_access"]
    assert row["raw_output"]["blake3"] and row["candidate_output"]["bytes"]
    candidate = np.load(binding["workspace"] / row["candidate_output"]["path"], allow_pickle=False)
    assert np.allclose(candidate, 409.6, atol=.001)
    assert not (binding["workspace"] / "outputs/agents_outputs/case1").exists()


def test_real_segmentation_branch_with_fake_total_api(tmp_path, monkeypatch):
    np, nib = numerical()
    binding = binding_fixture(tmp_path, "segmentation")
    image = nib.Nifti1Image(np.zeros((2, 3, 4), np.float32), np.diag([1.5, 2., 3., 1.]))
    nib.save(image, binding["workspace"] / "inputs/case1/ct.nii.gz")
    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        return nib.Nifti1Image(np.ones(image.shape, np.uint8), image.affine)
    @contextmanager
    def offline(*args):
        yield run, {i: f"tissue_{118-i:03}" for i in range(1, 118)}
    monkeypatch.setattr(seg, "_offline_total", offline)
    progress = []
    rows = seg.segment(binding, SimpleNamespace(inference_mode=nullcontext),
        output_root=binding["workspace"] / "outputs/job", progress=progress.append)
    assert progress == rows
    assert calls[0]["fast"] is False and calls[0]["device"] == "gpu" and calls[0]["output"] is None
    assert calls[0]["higher_order_resampling"] is False and calls[0]["roi_subset"] is None
    row = rows[0]
    assert row["label_ids_present"] == [117] and row["native_to_benchmark_label_ids"]["1"] == 117
    reopened = nib.load(binding["workspace"] / row["candidate_output"]["path"])
    seg.validate_geometry(reopened, image)
    assert reopened.get_data_dtype() == np.dtype("uint8") and not row["final_submission_written"]
