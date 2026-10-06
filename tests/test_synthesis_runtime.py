"""CPU fixtures only: no downloaded model weights, CUDA, provider or gold reads."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import uuid

from blake3 import blake3
import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
from training.benchmark_models import synthesis_runtime as runtime


def public_image():
    data = (np.arange(7 * 9 * 11, dtype=np.float32).reshape(7, 9, 11) * 2.5 - 1000)
    affine = np.diag([-0.8, 0.9, 2.4, 1.0])
    affine[:3, 3] = [12.5, -23.5, 81.0]
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 2)
    image.header.set_xyzt_units("mm", "sec")
    return image


def test_public_normalization_uses_zyx_and_no_resampling():
    image = public_image()
    actual, meta = runtime.normalize_public_volume(image)
    source = np.asarray(image.dataobj)
    expected = ((source - source.min()) / (source.max() - source.min())).transpose(2, 1, 0)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (11, 9, 7)
    assert actual.flags.c_contiguous
    assert (meta["hu_min"], meta["hu_max"]) == (float(source.min()), float(source.max()))
    assert meta["checkpoint_training_window_recorded"] is False


def test_twelve_layer_cpu_stencil_matches_whole_volume_at_seams_and_edges():
    # An analytic fixed 12-layer spatial fixture, NOT the public neural model.
    # It moves information 12 voxels, so an insufficient halo would fail exactly.
    volume = np.random.default_rng(17).normal(size=(35, 29, 31)).astype(np.float32)
    calls = []
    def forward(patch):
        calls.append(patch.shape)
        residual = patch.copy()
        for _ in range(12):
            shifted = np.zeros_like(residual)
            shifted[1:, :, :] = residual[:-1, :, :]
            residual = np.where(shifted >= 0, shifted, shifted * 0.1)
        return patch + residual
    expected = forward(volume)
    calls.clear()
    actual, count = runtime.infer_tiled_zyx(volume, forward, core_shape=(8, 10, 11))
    np.testing.assert_array_equal(actual, expected)
    assert count == 5 * 3 * 3 == len(calls)
    assert all(all(got <= core + 24 for got, core in zip(shape, (8, 10, 11))) for shape in calls)


@pytest.mark.parametrize("qform_code,sform_code", [(1, 2), (0, 2), (1, 0)])
def test_hu_output_keeps_entire_shape_scaling_and_active_forms(tmp_path, qform_code, sform_code):
    source = public_image()
    source.set_qform(source.get_qform(), qform_code)
    source.set_sform(source.get_sform(), sform_code)
    normalized, meta = runtime.normalize_public_volume(source)
    destination = tmp_path / "MSDPSR_0001/sct.nii.gz"
    artifact = runtime.save_hu_prediction(source, normalized, meta, destination)
    actual = nib.load(destination)
    np.testing.assert_allclose(actual.get_fdata(), source.get_fdata(), atol=0.0002)
    np.testing.assert_allclose(actual.affine, source.affine, atol=1e-6)
    np.testing.assert_allclose(actual.get_qform(), source.get_qform(), atol=1e-6)
    np.testing.assert_allclose(actual.get_sform(), source.get_sform(), atol=1e-6)
    assert actual.shape == source.shape
    assert actual.get_data_dtype() == np.dtype("float32")
    assert actual.header.get_xyzt_units() == source.header.get_xyzt_units()
    assert (int(actual.header["qform_code"]), int(actual.header["sform_code"])) == (qform_code, sform_code)
    assert artifact["blake3"] == blake3(destination.read_bytes()).hexdigest()
    assert not (destination.parent / "sct.incomplete.nii.gz").exists()


def test_ct_scaling_is_applied_once_and_output_is_not_clamped(tmp_path):
    source_path = tmp_path / "ct.nii.gz"
    source = nib.Nifti1Image(np.arange(27, dtype=np.int16).reshape(3, 3, 3), np.eye(4))
    source.header.set_slope_inter(2, -1000)
    nib.save(source, source_path)
    reopened = nib.load(source_path)
    normalized, meta = runtime.normalize_public_volume(reopened)
    normalized += np.float32(0.5)  # A mock model may output outside [0,1].
    destination = tmp_path / "case/sct.nii.gz"
    runtime.save_hu_prediction(reopened, normalized, meta, destination)
    expected = reopened.get_fdata() + 0.5 * (meta["hu_max"] - meta["hu_min"])
    np.testing.assert_allclose(nib.load(destination).get_fdata(), expected, atol=0.0001)


@pytest.mark.parametrize("data", [np.zeros((2, 3, 4)), np.zeros((2, 3, 4, 1)),
                                  np.full((2, 3, 4), np.nan)])
def test_bad_public_volume_is_not_replaced_by_a_prediction(data):
    with pytest.raises(ValueError):
        runtime.normalize_public_volume(nib.Nifti1Image(data, np.eye(4)))


@pytest.mark.parametrize("forward", [lambda x: x[1:], lambda x: x * np.float32(np.nan)])
def test_bad_actual_prediction_fails_before_saving(forward):
    with pytest.raises(ValueError, match="output shape/finite"):
        runtime.infer_tiled_zyx(np.ones((3, 4, 5), np.float32), forward)


def binding_fixture(tmp_path, monkeypatch):
    workspace = tmp_path / "actor"
    case = "MSDPSR_0001"
    source = workspace / f"inputs/{case}/ct.nii.gz"
    source.parent.mkdir(parents=True)
    nib.save(public_image(), source)
    model_root = tmp_path / "public-model"
    payloads = {runtime.SOURCE_FILE: b"fixture source is not executable", runtime.WEIGHT_FILE: b"inert fixture weights"}
    rows = []
    for relative, data in payloads.items():
        path = model_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        rows.append({"path": relative, "bytes": len(data), "blake3": blake3(data).hexdigest()})
    monkeypatch.setattr(runtime, "PINNED_FILES", {row["path"]: (row["bytes"], row["blake3"]) for row in rows})
    output = workspace / "outputs/agents_outputs/prescribed-model-jobs" / str(uuid.uuid4())
    output.mkdir(parents=True)
    binding = {"workspace": workspace, "task": {"track": "synthesis", "no_private_reference_access": True,
               "case_ids": [case]}, "selected": {case: [{"path": str(source.relative_to(workspace)),
               "bytes": source.stat().st_size, "blake3": blake3(source.read_bytes()).hexdigest()}]},
               "model": {"repository": runtime.REPOSITORY, "revision": runtime.REVISION,
                         "path": str(model_root), "files": rows}}
    return binding, output


def test_reopens_public_commitments_and_rejects_changes(tmp_path, monkeypatch):
    binding, output = binding_fixture(tmp_path, monkeypatch)
    checked = runtime.validate_public_binding(binding, output)
    assert set(checked["inputs"]) == {"MSDPSR_0001"}
    source = checked["inputs"]["MSDPSR_0001"]
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Public CT commitment"):
        runtime.validate_public_binding(binding, output)


def test_wrong_source_output_or_selected_case_is_rejected(tmp_path, monkeypatch):
    binding, output = binding_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Output must"):
        runtime.validate_public_binding(binding, output.parent)
    binding["model"]["revision"] = "unreviewed"
    with pytest.raises(ValueError, match="source revision"):
        runtime.validate_public_binding(binding, output)
    binding["model"]["revision"] = runtime.REVISION
    binding["task"]["case_ids"] = []
    with pytest.raises(ValueError, match="case selection"):
        runtime.validate_public_binding(binding, output)


def test_public_input_symlink_outside_workspace_is_rejected(tmp_path, monkeypatch):
    binding, output = binding_fixture(tmp_path, monkeypatch)
    source = binding["workspace"] / "inputs/MSDPSR_0001/ct.nii.gz"
    external = tmp_path / "outside.nii.gz"
    source.rename(external)
    source.symlink_to(external)
    with pytest.raises(ValueError, match="escapes"):
        runtime.validate_public_binding(binding, output)


def test_loader_is_weights_only_exact_state_keys_and_strict(tmp_path, monkeypatch):
    source = b"class PlainCNN:\n def __new__(cls):\n  return spy_model\n"
    path = tmp_path / runtime.SOURCE_FILE
    path.parent.mkdir(parents=True)
    path.write_bytes(source)
    monkeypatch.setattr(runtime, "PINNED_FILES", {runtime.SOURCE_FILE: (len(source), blake3(source).hexdigest())})
    calls = []
    model = SimpleNamespace(load_state_dict=lambda state, **kw: calls.append(("state", set(state), kw)),
                            eval=lambda: model, to=lambda **kw: calls.append(("to", kw)) or model)
    real_module_type = runtime.types.ModuleType
    def module_type(name):
        module = real_module_type(name)
        module.spy_model = model
        return module
    monkeypatch.setattr(runtime.types, "ModuleType", module_type)
    def load(path, **kwargs):
        calls.append(("load", path, kwargs))
        return dict.fromkeys(runtime.EXPECTED_STATE_KEYS, "inert mock tensor")
    fake_torch = SimpleNamespace(load=load, float32="float32")
    assert runtime.load_plaincnn(tmp_path, fake_torch, device="cpu") is model
    assert calls[0][2] == {"map_location": "cpu", "weights_only": True}
    assert calls[1][2] == {"strict": True}
    assert calls[2] == ("to", {"device": "cpu", "dtype": "float32"})
    fake_torch.load = lambda *args, **kwargs: {"state_dict": {}}
    with pytest.raises(ValueError, match="exact expected"):
        runtime.load_plaincnn(tmp_path, fake_torch, device="cpu")


def test_complete_runtime_with_mock_forward_retains_only_real_artifact_record(tmp_path, monkeypatch):
    binding, output = binding_fixture(tmp_path, monkeypatch)
    class Tensor:
        def __init__(self, array): self.array = array
        def __getitem__(self, item): return Tensor(self.array[item])
        def to(self, **kwargs): return self
        def detach(self): return self
        def float(self): return self
        def cpu(self): return self
        def numpy(self): return self.array
    fake_torch = SimpleNamespace(inference_mode=nullcontext, from_numpy=Tensor, float32=np.float32)
    calls = []
    monkeypatch.setattr(runtime, "load_plaincnn", lambda *args, **kwargs: lambda tensor: calls.append(tensor.array.shape) or tensor)
    observed = []
    result = runtime.run_synthesis(binding, fake_torch, observed.append, output_root=output, device="cpu")
    assert len(result) == len(observed) == len(calls) == 1
    record = result[0]
    assert record is observed[0]
    assert record["clinical_score"] is None and record["private_reference_access"] is False
    assert record["analysis_model"]["revision"] == runtime.REVISION
    artifact = binding["workspace"] / record["artifact"]["path"]
    assert artifact == output / "MSDPSR_0001/sct.nii.gz"
    np.testing.assert_allclose(nib.load(artifact).get_fdata(), public_image().get_fdata(), atol=0.0002)
    # A restart cannot silently overwrite/re-count a completed case artifact.
    with pytest.raises(FileExistsError):
        runtime.run_synthesis(binding, fake_torch, output_root=output, device="cpu")
