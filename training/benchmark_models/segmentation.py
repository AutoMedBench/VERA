"""Pinned full-resolution TotalSegmentator v2, with public benchmark label remap.

Call ``segment(binding, torch, output_root=..., progress=None)`` only from the
host-admitted, single-job GPU subprocess after full manifest verification. The
installed 2.17.0 ``total`` task runs partitions 291..295, each fold 0, at 1.5mm;
this is NOT a five-fold ensemble. Fast/ROI/cropping alternatives are not used.

The upstream API already restores the original CT grid with nearest-neighbour
label interpolation. We validate that grid, fail on mismatch (no fake affine
repair), retain its native-label output, and derive a separately identified
benchmark-ID candidate. Neither artifact is installed as a final submission.
No scorer/reference inputs or network weight downloads are accepted.
"""
from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import version
import json
import os
from pathlib import Path
from unittest.mock import patch

if __package__:
    from ._track_io import artifact, case_output, public_input
else:
    from _track_io import artifact, case_output, public_input

REPOSITORY = "wasserth/TotalSegmentator"
REVISION = "v2.0.0-weights"
PARTITIONS = {
    291: "Dataset291_TotalSegmentator_part1_organs_1559subj",
    292: "Dataset292_TotalSegmentator_part2_vertebrae_1532subj",
    293: "Dataset293_TotalSegmentator_part3_cardiac_1559subj",
    294: "Dataset294_TotalSegmentator_part4_muscles_1559subj",
    295: "Dataset295_TotalSegmentator_part5_ribs_1559subj",
}
PLAN_DIRECTORY = "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"


def validate_model(model):
    if (model.get("repository"), model.get("revision"), model.get("release_id")) != (REPOSITORY, REVISION, 121996387):
        raise ValueError("Pinned TotalSegmentator release identity mismatch")
    if model.get("task_ids") != list(PARTITIONS) or model.get("folds") != [0] or model.get("resample_mm") != 1.5 or model.get("fast") is not False:
        raise ValueError("Full-resolution TotalSegmentator partition contract changed")
    root = Path(model["path"]).resolve(strict=True)
    inventory = {row["path"] for row in model["files"]}
    for directory in PARTITIONS.values():
        for name in ("dataset.json", "plans.json", "fold_0/checkpoint_final.pth"):
            relative = f"{directory}/{PLAN_DIRECTORY}/{name}"
            path = (root / relative).resolve(strict=True)
            if relative not in inventory or not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Pinned full-resolution partition is missing locally")
    return root


def label_remap(native_labels, benchmark_labels):
    native = {int(k): v for k, v in native_labels.items()}
    benchmark = {int(k): v for k, v in benchmark_labels.items()}
    expected_ids = set(range(1, 118))
    if set(native) != expected_ids or set(benchmark) != expected_ids:
        raise ValueError("Expected all 117 foreground label definitions")
    if len(set(native.values())) != 117 or set(native.values()) != set(benchmark.values()):
        raise ValueError("Native and benchmark tissue names differ")
    by_name = {name: label for label, name in benchmark.items()}
    return {0: 0, **{label: by_name[name] for label, name in native.items()}}


def validate_geometry(image, reference):
    import numpy as np
    if len(reference.shape) != 3 or image.shape != reference.shape:
        raise ValueError("Segmentation shape differs from public CT grid")
    affine = np.asarray(reference.affine)
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) <= np.finfo(float).eps:
        raise ValueError("Public CT geometry is invalid")
    tolerance = max(1e-5, float(np.abs(affine).max()) * 1e-6)
    for candidate in (reference, image):
        if not np.allclose(candidate.affine, affine, rtol=1e-6, atol=tolerance):
            raise ValueError("Segmentation affine differs from public CT geometry")
        qform, qcode = candidate.get_qform(coded=True)
        sform, scode = candidate.get_sform(coded=True)
        if any(code and not np.isfinite(form).all() for form, code in ((qform, qcode), (sform, scode))):
            raise ValueError("Nonfinite active NIfTI geometry")
        if qcode and scode and not np.allclose(qform, sform, rtol=1e-6, atol=tolerance):
            raise ValueError("Contradictory active NIfTI geometry")


def remap_prediction(image, reference, mapping):
    import nibabel as nib
    import numpy as np
    validate_geometry(image, reference)
    data = np.asanyarray(image.dataobj)
    values = np.unique(data)
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all() or not set(values.tolist()) <= set(mapping):
        raise ValueError("Native segmentation contains invalid class IDs")
    lookup = np.array([mapping[i] for i in range(118)], dtype=np.uint8)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    result = nib.Nifti1Image(lookup[data.astype(np.uint8)], reference.affine, header)
    for get_name, set_name in (("get_qform", "set_qform"), ("get_sform", "set_sform")):
        form, code = getattr(reference, get_name)(coded=True)
        getattr(result, set_name)(form, int(code))
    validate_geometry(result, reference)
    return result


@contextmanager
def _offline_total(weights, cache):
    """Process-local environment; upstream downloader is replaced with a guard."""
    if version("TotalSegmentator") != "2.17.0":
        raise RuntimeError("This runtime was checked against TotalSegmentator 2.17.0 only")
    cache.mkdir(parents=True, exist_ok=False)
    (cache / "config.json").write_text(json.dumps({"totalseg_id": "eva-local-public-inference",
        "send_usage_stats": False, "statistics_disclaimer_shown": True, "prediction_counter": 0}))
    environment = {"TOTALSEG_HOME_DIR": str(cache), "TOTALSEG_WEIGHTS_PATH": str(weights),
                   "nnUNet_raw": str(weights), "nnUNet_preprocessed": str(weights), "nnUNet_results": str(weights),
                   "TORCH_HOME": str(cache / "torch"), "HF_HOME": str(cache / "huggingface"),
                   "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
    with patch.dict(os.environ, environment):
        from totalsegmentator import python_api as api
        from totalsegmentator.map_to_binary import class_map
        from nnunetv2 import paths
        if Path(paths.nnUNet_results).resolve() != weights:
            raise RuntimeError("nnU-Net was imported with a different model cache; use a fresh subprocess")
        config = api.get_task_config("total", fast=False, fastest=False)
        if config["task_id"] != list(PARTITIONS) or config["folds"] != [0] or config["resample"] != 1.5:
            raise RuntimeError("Installed TotalSegmentator default task changed")

        def require_local_partition(task_id):
            if task_id not in PARTITIONS or not (weights / PARTITIONS[task_id] / PLAN_DIRECTORY / "fold_0/checkpoint_final.pth").is_file():
                raise RuntimeError("Unpinned/missing TotalSegmentator task; downloads forbidden")

        with patch.object(api, "download_pretrained_weights", require_local_partition), patch.object(api, "send_usage_stats", lambda *args, **kwargs: None):
            yield api.totalsegmentator, class_map["total"]


def segment(binding, torch, *, output_root, progress=None):
    import nibabel as nib
    import numpy as np
    if binding["task"].get("track") != "segmentation" or not binding["task"].get("no_private_reference_access"):
        raise ValueError("Expected public-only segmentation binding")
    cfg = binding["task"]["public_config"]
    if cfg.get("task_id") != "tsg-multiorgan-seg-task" or cfg.get("num_foreground_classes") != 117:
        raise ValueError("Only the pinned 117-class CT task is supported")
    weights = validate_model(binding["model"])
    inputs = {case: public_input(binding, case, "ct.nii.gz") for case in binding["selected"]}
    # Confine the writable cache via the same output boundary before importing TotalSeg.
    cache_parent = case_output(binding, output_root, "totalseg-runtime")
    results = []
    with _offline_total(weights, cache_parent / "config") as (run, native_labels):
        mapping = label_remap(native_labels, cfg["tissue_labels"])
        for case, input_path in inputs.items():
            reference = nib.load(input_path)
            validate_geometry(reference, reference)
            destination = case_output(binding, output_root, case)
            with torch.inference_mode():
                prediction = run(input=input_path, output=None, ml=True, task="total", fast=False,
                    fastest=False, device="gpu", roi_subset=None, body_seg=False, statistics=False,
                    radiomics=False, preview=False, skip_saving=True, nr_thr_resamp=1, nr_thr_saving=1,
                    quiet=True, v1_order=False, higher_order_resampling=False,
                    higher_order_resampling_LEGACY=False, save_lowres=False)
            candidate = remap_prediction(prediction, reference, mapping)
            raw_path, candidate_path = destination / "native-labels.nii.gz", destination / "dseg.nii.gz"
            nib.save(prediction, raw_path)
            nib.save(candidate, candidate_path)
            reopened = nib.load(candidate_path)
            validate_geometry(reopened, reference)
            result = {"case_id": case, "model": "TotalSegmentator", "model_revision": REVISION,
                "task_ids": list(PARTITIONS), "folds": [0], "resample_mm": 1.5, "fast": False,
                "raw_output": artifact(binding, raw_path), "candidate_output": artifact(binding, candidate_path),
                "native_to_benchmark_label_ids": {str(k): v for k, v in mapping.items()},
                "shape": list(candidate.shape), "dtype": "uint8", "label_ids_present": np.unique(np.asanyarray(reopened.dataobj)).tolist(),
                "postprocessing": "host_name_based_label_remap; not_actor_authored",
                "final_submission_written": False, "private_reference_access": False}
            results.append(result)
            if progress is not None:
                progress(result)
    return results
