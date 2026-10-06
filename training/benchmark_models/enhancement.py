"""Pinned grayscale DRUNet inference on public LDCT slices, with explicit HU controls.

``enhance(binding, torch, output_root=..., sigma=..., hu_min=..., hu_max=...)``
is called ONLY after the host's manifest and GPU preflight. All three numeric
controls are required: the tool adapter must record the actor's choices (or
explicitly identify a supplied baseline), never invent actor authorship.

Raw normalized model output and a mechanically HU-converted/clipped candidate
are retained in the job directory, not installed as an actor's final submission.
No native score, clinical quality, or GPU validation is claimed by this helper.
Imports are lazy; importing this module does not import Torch/deepinv.
"""
from __future__ import annotations

from importlib.metadata import version
import math
from pathlib import Path

if __package__:
    from ._track_io import artifact, case_output, public_input
else:  # The owning runner is also executable directly by file path.
    from _track_io import artifact, case_output, public_input

REPOSITORY = "deepinv/drunet"
REVISION = "7e079a6800958ae777b48e41fc1202a2ee21ecbe"
WEIGHT_FILE = "drunet_deepinv_gray_finetune_26k.pth"
OUTPUT_HU_RANGE = (-1024.0, 3000.0)  # Pinned task config, not the example window.


def validate_model(model):
    if (model.get("repository"), model.get("revision")) != (REPOSITORY, REVISION):
        raise ValueError("Pinned grayscale DRUNet identity mismatch")
    if len([row for row in model["files"] if row["path"] == WEIGHT_FILE]) != 1:
        raise ValueError("Pinned grayscale checkpoint missing from verified inventory")
    path = Path(model["path"]) / WEIGHT_FILE
    if not path.is_file():
        raise ValueError("Pinned grayscale checkpoint is unavailable locally")
    return path


def validate_controls(sigma, hu_min, hu_max):
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in (sigma, hu_min, hu_max)):
        raise ValueError("DRUNet controls must be finite numbers")
    if not 1e-4 <= sigma <= 0.2 or not -4096 <= hu_min < hu_max <= 8192 or hu_max - hu_min < 1:
        raise ValueError("DRUNet controls exceed the admitted public inference range")


def normalize_hu(array, hu_min, hu_max):
    import numpy as np
    if array.shape != (512, 512) or array.dtype != np.dtype("float32") or not np.isfinite(array).all():
        raise ValueError("LDCT public input must be finite float32 with shape 512x512")
    return np.clip((array - hu_min) / (hu_max - hu_min), 0, 1).astype(np.float32)


def restore_hu(array, hu_min, hu_max):
    import numpy as np
    if array.shape != (512, 512) or not np.isfinite(array).all():
        raise ValueError("DRUNet prediction is nonfinite or has incompatible shape")
    return np.clip(array * (hu_max - hu_min) + hu_min, *OUTPUT_HU_RANGE).astype(np.float32)


def _load_model(checkpoint, torch):
    if version("deepinv") != "0.4.0":
        raise RuntimeError("This runtime was checked against deepinv 0.4.0 only")
    from deepinv.models import DRUNet
    # No download path and no custom pickle objects. Strictly load every weight.
    model = DRUNet(in_channels=1, out_channels=1, pretrained=None, dim=2)
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu", weights_only=True), strict=True)
    return model.eval().to("cuda")


def enhance(binding, torch, *, output_root, sigma, hu_min, hu_max, progress=None):
    import numpy as np
    if binding["task"].get("track") != "enhancement" or not binding["task"].get("no_private_reference_access"):
        raise ValueError("Expected public-only enhancement binding")
    cfg = binding["task"]["public_config"]
    if cfg.get("task_id") != "ldct-denoising-task":
        raise ValueError("Only the pinned LDCT task is supported")
    if tuple(cfg.get("intensity_range", OUTPUT_HU_RANGE)) != OUTPUT_HU_RANGE:
        raise ValueError("Pinned legal output HU range changed")
    validate_controls(sigma, hu_min, hu_max)
    checkpoint = validate_model(binding["model"])
    # Validate selected inputs before allocating the inference model.
    inputs = {case: public_input(binding, case, "input.npy") for case in binding["selected"]}
    for path in inputs.values():
        normalize_hu(np.load(path, allow_pickle=False, mmap_mode="r"), hu_min, hu_max)
    model = _load_model(checkpoint, torch)
    results = []
    with torch.inference_mode():
        for case, path in inputs.items():
            image = np.load(path, allow_pickle=False)
            x = torch.from_numpy(normalize_hu(image, hu_min, hu_max)).unsqueeze(0).unsqueeze(0).to("cuda")
            prediction = model(x, sigma=float(sigma)).detach().float().cpu().numpy()
            if prediction.shape != (1, 1, 512, 512):
                raise ValueError("DRUNet output tensor shape mismatch")
            raw = prediction[0, 0].astype(np.float32)
            candidate = restore_hu(raw, hu_min, hu_max)
            destination = case_output(binding, output_root, case)
            raw_path, candidate_path = destination / "denoised-unit.npy", destination / "enhanced.npy"
            np.save(raw_path, raw, allow_pickle=False)
            np.save(candidate_path, candidate, allow_pickle=False)
            result = {"case_id": case, "model": "DRUNet", "model_revision": REVISION,
                      "raw_output": artifact(binding, raw_path), "candidate_output": artifact(binding, candidate_path),
                      "sigma": float(sigma), "normalization_hu_window": [float(hu_min), float(hu_max)],
                      "output_hu_range": list(OUTPUT_HU_RANGE), "shape": [512, 512], "dtype": "float32",
                      "input_mean_absolute_change_hu": float(np.abs(candidate - image).mean()),
                      "postprocessing": "host_unit_to_HU_and_legal_range_clip; not_actor_authored",
                      "final_submission_written": False, "private_reference_access": False}
            results.append(result)
            if progress is not None:
                progress(result)
    return results
