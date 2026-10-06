"""Fixed public PlainCNN x4 inference for AutoMedBench-Lite CT synthesis.

This is a trusted analysis helper, not an actor-Python executor or scorer. The
host must admit/serialize GPU work and supply an immutable public-input binding
and a fresh, host-created job directory. Importing this module loads no model.

Architecture/weights: Roldbach/autoencoder_ct_3d_super_resolution, Apache-2.0,
commit 87ace8f44e77721fe43e2b249fbba211bbab13a3. The original architecture is
loaded from its digest-checked source; no copied or substituted network is used.
Upstream test.py defaults --window to (None, None), and io_utils reads Z,Y,X.
The checkpoint does not record its historical training-window arguments.

Benchmark adaptation: its public CT is ALREADY degraded and restored to the
target grid. Do not run upstream's degradation, XY downsampling, truncation or
uint8 metric conversion. Normalize the public volume once, infer with a 12-voxel
halo (12 stride-one 3x3x3 convolutions), and undo normalization into float32 HU.
Preserve public geometry. Native axial SSIM is computed ONLY by the separate
scorer; this helper never receives a reference path or produces a clinical score.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from itertools import product
from pathlib import Path, PurePosixPath
import re
import time
import types
import uuid

from blake3 import blake3

REPOSITORY = "Roldbach/autoencoder_ct_3d_super_resolution"
REVISION = "87ace8f44e77721fe43e2b249fbba211bbab13a3"
SOURCE_FILE = "model/PlainCNN.py"
WEIGHT_FILE = "weight/PlainCNN_trilinear_interpolation_x4.pth"
PINNED_FILES = {
    SOURCE_FILE: (1427, "69293a87b7e4681962591d37dee026f3daafe73846eb46a90c17b3f4114a75ee"),
    WEIGHT_FILE: (4449693, "4d9e4062492a0ce65865677e44265a32ac645b18160d1f74e09625dbe9ba7e1e"),
}
HALO = 12
CORE_SHAPE_ZYX = (32, 96, 96)
MAX_VOXELS = 256 * 1024**2
EXPECTED_STATE_KEYS = frozenset(
    f"{block}._layer.0.{kind}"
    for block in ("_block_in", *(f"_block_middle_iter.{i}" for i in range(10)), "_block_out")
    for kind in ("weight", "bias")
)


def _digest(path: Path) -> str:
    hasher = blake3()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _confined(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or not parts.parts:
        raise ValueError("Unconfined synthesis path")
    target = root.joinpath(*parts.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Synthesis path escapes its host binding")
    return target


def validate_public_binding(binding: Mapping, output_root: Path) -> dict:
    """Reopen fixed source/input commitments without importing torch or a model."""
    workspace = Path(binding["workspace"]).resolve(strict=True)
    task = binding["task"]
    if task.get("track") != "synthesis" or task.get("no_private_reference_access") is not True:
        raise ValueError("Synthesis requires an explicitly public-only task binding")
    model = binding["model"]
    if (model.get("repository"), model.get("revision")) != (REPOSITORY, REVISION):
        raise ValueError("Wrong prescribed PlainCNN source revision")
    model_root = Path(model["path"]).resolve(strict=True)
    if model_root.is_relative_to(workspace):
        raise ValueError("Trusted model source must be outside the actor workspace")
    rows = {row["path"]: row for row in model["files"]}
    if len(rows) != len(model["files"]):
        raise ValueError("Duplicate model manifest paths")
    for relative, (size, digest) in PINNED_FILES.items():
        row = rows.get(relative)
        path = _confined(model_root, relative)
        if row is None or (row["bytes"], row["blake3"]) != (size, digest):
            raise ValueError("Model manifest does not bind the fixed public asset")
        if path.stat().st_size != size or _digest(path) != digest:
            raise ValueError("Fixed public model commitment changed")

    # Only the host may choose the job UUID/path. Artifacts remain job-scoped;
    # actor-authored code installs them into the official final submission tree.
    output_root = Path(output_root).resolve(strict=True)
    jobs = _confined(workspace, "outputs/agents_outputs/prescribed-model-jobs")
    if output_root.parent != jobs or str(uuid.UUID(output_root.name)) != output_root.name:
        raise ValueError("Output must be the host-created, UUID-scoped public model job")
    selected = binding["selected"]
    if not 1 <= len(selected) <= 20 or not set(selected).issubset(task["case_ids"]):
        raise ValueError("Synthesis case selection is outside the public task")
    inputs = {}
    for case, input_rows in selected.items():
        if not re.fullmatch(r"MSDPSR_[0-9]{4}", case):
            raise ValueError("Invalid synthesis case ID")
        expected = f"inputs/{case}/ct.nii.gz"
        matching = [row for row in input_rows if row["path"] == expected]
        if len(matching) != 1:
            raise ValueError("Synthesis requires exactly one committed public ct.nii.gz")
        row = matching[0]
        path = _confined(workspace, expected)
        if path.stat().st_size != row["bytes"] or _digest(path) != row["blake3"]:
            raise ValueError("Public CT commitment changed")
        if (output_root / case).exists():
            raise FileExistsError("A selected synthesis job artifact already exists")
        inputs[case] = path
    return {"workspace": workspace, "model_root": model_root,
            "output_root": output_root, "inputs": inputs}


def load_plaincnn(model_root: Path, torch, *, device: str):
    """Called only after host GPU admission and validate_public_binding()."""
    source = (model_root / SOURCE_FILE).read_bytes()
    if (len(source), blake3(source).hexdigest()) != PINNED_FILES[SOURCE_FILE]:
        raise ValueError("PlainCNN source changed before import")
    module = types.ModuleType("eva_pinned_public_plaincnn_87ace8f4")
    module.__file__ = str(model_root / SOURCE_FILE)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    model = module.PlainCNN()
    # The pinned 1.12 checkpoint is a direct, unprefixed 24-key state_dict.
    # Never fall back to unsafe arbitrary-object deserialization or partial load.
    state = torch.load(model_root / WEIGHT_FILE, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or set(state) != EXPECTED_STATE_KEYS:
        raise ValueError("PlainCNN checkpoint is not the exact expected state_dict")
    model.load_state_dict(state, strict=True)
    return model.eval().to(device=device, dtype=torch.float32)


def normalize_public_volume(image):
    """Use public HU extrema, then transpose NIfTI X,Y,Z to upstream Z,Y,X."""
    import numpy as np

    if len(image.shape) != 3 or any(int(n) < 1 for n in image.shape):
        raise ValueError("Expected a nonempty 3D public CT")
    if int(np.prod(image.shape, dtype=np.int64)) > MAX_VOXELS:
        raise ValueError("Public CT exceeds the bounded host volume budget")
    xyz = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(xyz).all() or not np.isfinite(image.affine).all():
        raise ValueError("Public CT values and affine must be finite")
    low, high = float(xyz.min()), float(xyz.max())
    if not low < high or not np.isfinite(np.float32(high - low)):
        # Upstream normalization divides by zero for a constant volume. Do not
        # silently replace model inference by an identity/baseline prediction.
        raise ValueError("Public CT has a degenerate normalization range")
    zyx = np.array(xyz.transpose(2, 1, 0), dtype=np.float32, order="C", copy=True)
    zyx -= np.float32(low)
    zyx /= np.float32(high - low)
    return zyx, {"method": "public_volume_minmax_upstream_default_window_none",
                 "hu_min": low, "hu_max": high, "tensor_order": "ZYX",
                 "checkpoint_training_window_recorded": False}


def infer_tiled_zyx(volume, forward: Callable, *, core_shape=CORE_SHAPE_ZYX):
    """Cover every voxel once using real context halo, with no seam averaging.

    Only the *network* applies same-padding at true image edges. At internal tile
    edges the 12-voxel halo covers the exact receptive field. Equivalence is
    mathematical up to backend floating-point convolution differences, not a
    claim of already-tested full-checkpoint GPU equivalence.
    """
    import numpy as np

    if volume.ndim != 3 or volume.dtype != np.float32 or not np.isfinite(volume).all():
        raise ValueError("Expected a finite float32 ZYX volume")
    if len(core_shape) != 3 or any(type(n) is not int or not 1 <= n <= limit
                                   for n, limit in zip(core_shape, CORE_SHAPE_ZYX)):
        raise ValueError("Tile cores must fit the admitted inference bound")
    output = np.empty_like(volume)
    count = 0
    for start in product(*(range(0, n, step) for n, step in zip(volume.shape, core_shape))):
        stop = tuple(min(a + width, n) for a, width, n in zip(start, core_shape, volume.shape))
        lo = tuple(max(0, a - HALO) for a in start)
        hi = tuple(min(n, b + HALO) for b, n in zip(stop, volume.shape))
        outer = tuple(slice(a, b) for a, b in zip(lo, hi))
        core = tuple(slice(a, b) for a, b in zip(start, stop))
        inner = tuple(slice(a - h, b - h) for a, b, h in zip(start, stop, lo))
        patch = np.ascontiguousarray(volume[outer])
        prediction = np.asarray(forward(patch), dtype=np.float32)
        if prediction.shape != patch.shape or not np.isfinite(prediction).all():
            raise ValueError("Actual PlainCNN output shape/finite-value contract failed")
        output[core] = prediction[inner]
        count += 1
    return output, count


def save_hu_prediction(image, normalized_prediction, normalization: Mapping, destination: Path):
    """Restore CT units, all voxels, and public qform/sform without a reference."""
    import nibabel as nib
    import numpy as np

    expected = tuple(reversed(image.shape))
    if normalized_prediction.shape != expected or not np.isfinite(normalized_prediction).all():
        raise ValueError("Prediction does not cover the complete public image grid")
    # PlainCNN is residual and does not clamp its final activations. Preserve its
    # output, not a scorer-informed clipping/uint8 postprocessing shortcut.
    normalized_prediction *= np.float32(normalization["hu_max"] - normalization["hu_min"])
    normalized_prediction += np.float32(normalization["hu_min"])
    if not np.isfinite(normalized_prediction).all():
        raise ValueError("HU reconstruction overflowed")
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    result = image.__class__(normalized_prediction.transpose(2, 1, 0), image.affine.copy(), header)
    result.set_qform(image.get_qform(), int(image.header["qform_code"]))
    result.set_sform(image.get_sform(), int(image.header["sform_code"]))
    result.header.set_slope_inter(1.0, 0.0)
    result.header["cal_min"] = normalized_prediction.min()
    result.header["cal_max"] = normalized_prediction.max()
    destination.parent.mkdir(parents=False, exist_ok=False)
    # An interrupted save has a non-submission name and is never reported done.
    temporary = destination.with_name("sct.incomplete.nii.gz")
    nib.save(result, temporary)
    temporary.replace(destination)
    return {"path": str(destination), "bytes": destination.stat().st_size,
            "blake3": _digest(destination), "shape_xyz": list(image.shape),
            "dtype": "float32", "units": "HU", "affine": image.affine.tolist(),
            "qform_code": int(image.header["qform_code"]),
            "sform_code": int(image.header["sform_code"])}


def run_synthesis(binding: Mapping, torch, progress=None, *, output_root: Path, device="cuda:0"):
    """Infer real public cases once; return small artifact records, never GT/scores.

    Host integration must retain this executed source alongside the job receipt.
    This function does not acquire a GPU lock or allocate a CUDA context until
    called by the already-admitted host worker. No actor-controlled code runs.
    """
    import nibabel as nib

    checked = validate_public_binding(binding, output_root)
    model = load_plaincnn(checked["model_root"], torch, device=device)
    results = []
    with torch.inference_mode():
        for case, path in checked["inputs"].items():
            started = time.monotonic()
            image = nib.load(path)
            volume, normalization = normalize_public_volume(image)

            def forward(patch):
                tensor = torch.from_numpy(patch[None, None]).to(device=device, dtype=torch.float32)
                return model(tensor)[0, 0].detach().float().cpu().numpy()

            prediction, tiles = infer_tiled_zyx(volume, forward)
            artifact = save_hu_prediction(image, prediction, normalization,
                                          checked["output_root"] / case / "sct.nii.gz")
            artifact["path"] = str(Path(artifact["path"]).relative_to(checked["workspace"]))
            record = {"case_id": case, "artifact": artifact, "runtime_s": time.monotonic() - started,
                      "analysis_model": {"repository": REPOSITORY, "revision": REVISION,
                                         "checkpoint": WEIGHT_FILE, "checkpoint_blake3": PINNED_FILES[WEIGHT_FILE][1]},
                      "normalization": normalization, "tile_count": tiles,
                      "core_shape_zyx": list(CORE_SHAPE_ZYX), "halo_voxels": HALO,
                      "public_grid_preserved": True, "private_reference_access": False,
                      "clinical_score": None, "source_blake3": _digest(Path(__file__))}
            results.append(record)
            if progress is not None:
                progress(record)
    return results
