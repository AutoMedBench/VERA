"""Pinned public preparation for seven track workflows, not per-case rollouts.

The existing three-case adapter remains a separate diagnostic. This module
never copies evaluator code/configuration or private references into an actor.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import shutil
from uuid import uuid4

import yaml

from .adapter import EvaluationError, REVISION, REPOSITORY, file_digest, read_document, safe_file, write_once


@dataclass(frozen=True)
class Track:
    name: str
    task: str
    data: str
    evaluation: str
    task_directory: str
    marker: str
    count: int
    output: str
    metric: str
    license: str

    @property
    def prefix(self):
        return f"benchmarks/AutoMedBench-{self.name}"

    @property
    def public_prefix(self):
        return f"{self.prefix}/data/{self.data}/public/"

    @property
    def task_prefix(self):
        return f"{self.prefix}/{self.evaluation}/{self.task_directory}/"


TRACKS = (
    Track("classification", "skin-lesion-cls-task", "SkinLesionISIC", "eval_cls", "skin-lesion-cls-task",
          "image.jpg", 100, "predictions.csv", "balanced_accuracy", "CC-BY-NC-4.0"),
    Track("synthesis", "msd-pancreas-ctsr-task", "MSD_Pancreas_CT_SR20", "eval_synthetic", "msd-pancreas-ctsr-task",
          "ct.nii.gz", 20, "CASE/sct.nii.gz", "completion_adjusted_mean_ssim", "CC-BY-SA-4.0"),
    Track("detection", "grazpedwri-det-task", "GRAZPEDWRI_Detection100", "eval_det2d", "grazpedwri-det-task",
          "image.png", 100, "CASE/prediction.json", "mAP", "CC-BY-4.0; no re-identification"),
    Track("segmentation", "tsg-multiorgan-seg-task", "TSG_multi-organ", "eval_seg", "tsg-multiorgan-seg-task",
          "ct.nii.gz", 40, "CASE/dseg.nii.gz", "macro_mean_dice", "CC-BY-4.0"),
    Track("vqa", "medxpertqa-mm-task", "MedXpertQA_MM", "eval_vqa", "tasks/medxpertqa-mm-task",
          "question.json", 2005, "CASE/answer.json", "accuracy", "MIT"),
    Track("report", "chexpert-plus-cxr-task", "CheXpert_Plus_Report/test_100_v1", "eval_report_gen", "tasks/chexpert-plus-cxr-task",
          "manifest.json", 100, "CASE/report.txt", "weighted_observation_f1_rouge_l_with_all_case_gate",
          "Unspecified mirror license; Stanford/CheXpert Plus terms; private local evaluation only; no redistribution"),
    Track("enhancement", "ldct-denoising-task", "LDCT_SimNICT", "eval_image_enhancement", "ldct-denoising-task",
          "input.npy", 20, "CASE/enhanced.npy", "completion_adjusted_mean_ssim", "CC-BY-ND-4.0"),
)
BY_TRACK = {track.name: track for track in TRACKS}
PUBLIC_CONFIG_KEYS = {"task_id", "task_type", "task_description", "organ", "modality", "classes", "class_names",
    "tissue_labels", "num_foreground_classes", "input_filename", "output_filename", "answer_mode", "split",
    "iou_threshold", "num_classes", "intensity_range"}


class TrackRelease:
    """One complete acquisition receipt and its two disjoint visibility roots."""
    def __init__(self, receipt_path: Path):
        document = read_document(receipt_path, maximum=16 * 1024 * 1024)
        if (document.get("schema") not in {"eva.automedbench-lite-pinned-assets.v1",
                "eva.automedbench-lite-pinned-missing4-assets.v1"}
                or document.get("revision") != REVISION or document.get("repository") != REPOSITORY):
            raise EvaluationError("track_release_identity_invalid")
        self.document = document
        self.receipt_path = receipt_path.absolute()
        self.public = Path(document.get("public_root", receipt_path.parent / "public-release")).absolute()
        self.scorer = Path(document.get("scorer_only_root", receipt_path.parent / "scorer-only-release")).absolute()
        if (self.public.resolve() == self.scorer.resolve() or self.public.is_relative_to(self.scorer)
                or self.scorer.is_relative_to(self.public) or self.scorer.stat().st_mode & 0o077):
            raise EvaluationError("track_visibility_roots_invalid")
        rows = document.get("inventory", [])
        self.inventory = {row["path"]: row for row in rows}
        if not rows or len(rows) != len(self.inventory):
            raise EvaluationError("track_inventory_invalid")
        for relative, row in self.inventory.items():
            if row.get("scorer_only") is not ("private" in PurePosixPath(relative).parts):
                raise EvaluationError("track_visibility_classification_invalid")

    def public_file(self, relative: str) -> Path:
        row = self.inventory.get(relative)
        if row is None or row["scorer_only"]:
            raise EvaluationError("track_public_file_not_declared")
        path = safe_file(self.public, relative)
        if path.stat().st_size != row["bytes"] or file_digest(path) != row["blake3"]:
            raise EvaluationError("track_public_file_commitment_invalid")
        return path

    def case_inputs(self, track: Track):
        rows = []
        case_ids = set()
        for relative, row in sorted(self.inventory.items()):
            if not relative.startswith(track.public_prefix) or row["scorer_only"]:
                continue
            local = relative.removeprefix(track.public_prefix)
            parts = PurePosixPath(local).parts
            if len(parts) < 2:
                raise EvaluationError("track_public_case_layout_invalid")
            if len(parts) == 2 and parts[-1] == track.marker:
                case_ids.add(parts[0])
            rows.append({"source_path": relative, "path": "inputs/" + local,
                         "bytes": row["bytes"], "blake3": row["blake3"]})
        if len(case_ids) != track.count or any(PurePosixPath(row["path"]).parts[1] not in case_ids for row in rows):
            raise EvaluationError("track_requires_exact_complete_public_subset")
        return sorted(case_ids), rows


def public_output_contract(track: Track):
    common = {"submission_root": "outputs/agents_outputs", "relative_output": track.output,
              "all_selected_cases_required": True, "native_metric": track.metric}
    if track.name == "classification":
        return {**common, "format": "CSV header patient_id,label; one exact public class label per selected case"}
    if track.name == "detection":
        return {**common, "format": "JSON boxes array with class,score,x1,y1,x2,y2; original-image pixel bounds"}
    if track.name == "segmentation":
        return {**common, "format": "Single integer NIfTI; input shape/affine; 0 or public tissue_labels integer keys"}
    if track.name == "synthesis":
        return {**common, "format": "Finite 3D NIfTI, same public-input shape/affine/spacing and consistent active qform/sform"}
    if track.name == "enhancement":
        return {**common, "format": "Finite float32 2D NPY, input shape, HU range [-1024,3000]"}
    if track.name == "report":
        return {**common, "format": "UTF-8 printable ASCII plus newline/tab; 40..8000chars, >=20alphabetic",
                "whole_track_rule": "Any missing/invalid selected report forces native whole-track TaskScore to zero"}
    return {**common, "format": "JSON", "required": ["question_id", "predicted_label", "predicted_answer",
            "raw_model_output", "model_name", "runtime_s"],
            "prediction_rule": "Use actual prescribed LLaVA-Med decode, never invented model output; public-option label/text match",
            "runtime_rule": "finite nonnegative runtime_s; retained actual decode/model identity and smoke evidence"}


def prepare_track_run(releases: dict[str, TrackRelease], output_root: Path) -> Path:
    if set(releases) != set(BY_TRACK):
        raise EvaluationError("exact_seven_track_sources_required")
    run = output_root.absolute() / str(uuid4())
    run.mkdir(parents=True, mode=0o700)
    prepared = []
    for track in TRACKS:
        source = releases[track.name]
        cases, inputs = source.case_inputs(track)
        workspace = run / "actors" / str(uuid4())
        if workspace.is_relative_to(source.public) or workspace.is_relative_to(source.scorer):
            raise EvaluationError("track_actor_source_overlap")
        workspace.mkdir(parents=True, mode=0o700)
        for directory in ("outputs/agents_outputs", "notes", "code", "public-guidance"):
            (workspace / directory).mkdir(parents=True, mode=0o700)
        for row in inputs:
            target = workspace / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(source.public_file(row["source_path"]), target)
            target.chmod(0o400)
        config = yaml.safe_load(source.public_file(track.task_prefix + "config.yaml").read_bytes())
        if config.get("task_id") != track.task:
            raise EvaluationError("track_task_identity_invalid")
        public_config = {key: config[key] for key in PUBLIC_CONFIG_KEYS if key in config}
        if "tissue_labels" in public_config:
            public_config["tissue_labels"] = {str(k): v for k, v in public_config["tissue_labels"].items()}
        guidance = []
        for name in ("model_info.yaml", "lite_s1.md", "lite_s2.md", "lite_s3.md"):
            relative = track.task_prefix + name
            original = source.public_file(relative)
            destination = workspace / "public-guidance" / name
            shutil.copyfile(original, destination)
            destination.chmod(0o400)
            guidance.append({"path": "public-guidance/" + name, "source_path": relative,
                             "bytes": original.stat().st_size, "blake3": file_digest(original)})
        model_info = yaml.safe_load((workspace / "public-guidance/model_info.yaml").read_bytes())
        write_once(workspace / "task.json", {"schema": "eva.automedbench-public-track-task.v1", "track": track.name,
            "task_id": track.task, "release_revision": REVISION, "tier": "lite", "case_ids": cases,
            "expected_case_count": track.count, "one_coding_workflow_for_full_subset": True,
            "input_root": "inputs", "public_config": public_config, "prescribed_analysis_model": model_info.get("lite"),
            "coding_orchestrator": "trained Qwen3.5-9B through Codex", "analysis_model_is_not_orchestrator": True,
            "output_contract": public_output_contract(track), "license": track.license,
            "no_training_or_finetuning": True, "no_private_reference_access": True,
            "source_priority": "Task-specific config/model_info/lite hints and exact native scorer; generic stale prompts do not override",
            "known_source_conflicts": ({"generic_prompt": "Old synthesis fragments mention Dice/masks", "authority": "CT super-resolution sct.nii.gz/SSIM"}
                if track.name == "synthesis" else {"generic_prompt": "Old enhancement preamble mentions PSNR", "authority": "completion-adjusted SSIM"}
                if track.name == "enhancement" else {}), "source_guidance": guidance})
        write_once(workspace / "inputs-manifest.json", {"schema": "eva.automedbench-public-track-inputs.v1",
            "track": track.name, "case_ids": cases, "files": inputs, "all_inputs_readonly_mount_required": True})
        prepared.append({"track": track.name, "case_count": track.count, "workspace_relative": workspace.relative_to(run).as_posix(),
            "source_acquisition_document_blake3": source.document["document_blake3"],
            "source_receipt_path": str(source.receipt_path), "task_file_blake3": file_digest(workspace / "task.json"),
            "input_manifest_file_blake3": file_digest(workspace / "inputs-manifest.json")})
    write_once(run / "track-run-manifest.json", {"schema": "eva.automedbench-seven-track-run.v1", "run_id": run.name,
        "release_revision": REVISION, "tracks": prepared, "planned_coding_rollouts": 7, "planned_case_outputs": 2385,
        "repeats": 1, "official_five_repeat_leaderboard": False, "provider_calls": 0,
        "runtime_dependencies_verified": False, "evaluation_complete": False,
        "private_reference_contents_copied_to_actor": False})
    return run
