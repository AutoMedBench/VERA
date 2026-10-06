"""Public-only workspaces and opaque native scores, independent of EVA schemas.

Native score semantics follow the pinned release and BAT's single-case bridge at
27124a2a35fa9ee68ef890bd7e4a36ffb5802860. No agent, GPU, or provider is launched.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
from typing import Any
from uuid import uuid4

from blake3 import blake3
import yaml

REVISION = "8928073d5c3f3b842a4a4278d9b44f6e8ceaa9c5"
REPOSITORY = "operator/AutoMedBench-Lite-release"
CASES = (
    {"track": "classification", "case_id": "ISIC_00000001", "task_id": "skin-lesion-cls-task",
     "data": "SkinLesionISIC", "eval": "eval_cls", "input": "image.jpg", "output": "prediction.json",
     "metric": "balanced_accuracy", "license": "CC-BY-NC-4.0", "source": "HAM10000 / ISIC 2018"},
    {"track": "detection", "case_id": "GRAZPEDWRI_00000001", "task_id": "grazpedwri-det-task",
     "data": "GRAZPEDWRI_Detection100", "eval": "eval_det2d", "input": "image.png", "output": "prediction.json",
     "metric": "mAP", "license": "CC-BY-4.0; no re-identification", "source": "GRAZPEDWRI-DX"},
    {"track": "segmentation", "case_id": "TSG_00000001", "task_id": "tsg-multiorgan-seg-task",
     "data": "TSG_multi-organ", "eval": "eval_seg", "input": "ct.nii.gz", "output": "dseg.nii.gz",
     "metric": "macro_mean_dice", "license": "CC-BY-4.0", "source": "TotalSegmentator CT-Lite"},
)


class EvaluationError(ValueError):
    """Safe local error: never include scorer stdout, stderr, or reference text."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_digest(path: Path) -> str:
    result = blake3()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_once(path: Path, value: dict) -> dict:
    document = {**value, "document_blake3": blake3(canonical(value)).hexdigest()}
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical(document))
    return document


def read_document(path: Path, maximum: int = 4 * 1024 * 1024) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise EvaluationError("document_topology_or_size_invalid")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise EvaluationError("document_shape_invalid")
    core = {key: item for key, item in value.items() if key != "document_blake3"}
    if value.get("document_blake3") != blake3(canonical(core)).hexdigest():
        raise EvaluationError("document_digest_invalid")
    return value


def safe_file(root: Path, relative: str, maximum: int = 1024 * 1024 * 1024) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or pure.as_posix() != relative or ".." in pure.parts:
        raise EvaluationError("unsafe_relative_path")
    current = root
    if root.is_symlink():
        raise EvaluationError("root_symlink_rejected")
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise EvaluationError("file_symlink_rejected")
    info = current.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > maximum:
        raise EvaluationError("file_topology_or_size_invalid")
    if not current.resolve().is_relative_to(root.resolve()):
        raise EvaluationError("file_outside_root")
    return current


def case_paths(case: dict) -> tuple[str, str, str]:
    base = f"benchmarks/AutoMedBench-{case['track']}"
    data = f"{base}/data/{case['data']}"
    return (f"{data}/public/{case['case_id']}/{case['input']}",
            f"{base}/{case['eval']}/{case['task_id']}/config.yaml", f"{data}/private/")


@dataclass(frozen=True)
class Assets:
    root: Path
    document: dict
    inventory: dict[str, dict]

    @property
    def public(self) -> Path:
        return self.root / "public-release"

    @property
    def scorer(self) -> Path:
        return self.root / "scorer-only-release"

    def public_file(self, relative: str) -> Path:
        row = self.inventory.get(relative)
        if row is None or row["scorer_only"] or "private" in PurePosixPath(relative).parts:
            raise EvaluationError("public_asset_not_allowlisted")
        path = safe_file(self.public, relative)
        if path.stat().st_size != row["bytes"] or file_digest(path) != row["blake3"]:
            raise EvaluationError("public_asset_commitment_invalid")
        return path


def load_assets(root: Path) -> Assets:
    root = Path(root).absolute()
    receipt = root / "download-all3.json"
    if not receipt.exists():
        receipt = root / "download-first3.json"
    document = read_document(receipt)
    if (document.get("schema") != "eva.automedbench-lite-pinned-assets.v1"
            or document.get("revision") != REVISION or document.get("repository") != REPOSITORY):
        raise EvaluationError("asset_release_binding_invalid")
    rows = document.get("inventory", [])
    inventory = {row["path"]: row for row in rows}
    if not rows or len(inventory) != len(rows):
        raise EvaluationError("asset_inventory_invalid")
    for path, row in inventory.items():
        if row.get("scorer_only") is not ("private" in PurePosixPath(path).parts):
            raise EvaluationError("asset_visibility_invalid")
    result = Assets(root, document, inventory)
    if stat.S_IMODE(result.scorer.stat().st_mode) & 0o077:
        raise EvaluationError("scorer_root_not_owner_only")
    return result


def preflight(assets: Assets) -> dict:
    cases = []
    required = ["automedbench_release/__init__.py", "automedbench_release/raw_track_worker.py", "automedbench_release/tasks.py"]
    for case in CASES:
        prefix = f"benchmarks/AutoMedBench-{case['track']}/{case['eval']}/"
        scorer = {"classification": "acc_scorer.py", "detection": "det2d_scorer.py", "segmentation": "dice_scorer.py"}[case["track"]]
        required.extend(prefix + name for name in ("format_checker.py", scorer, "aggregate.py"))
    if any(path not in assets.inventory for path in required):
        raise EvaluationError("required_native_source_missing")
    # Verify executable release source bytes; opaque gold is only stat'ed here.
    for relative, row in assets.inventory.items():
        if not row["scorer_only"] and "/public/" not in relative:
            assets.public_file(relative)
            scored = safe_file(assets.scorer, relative)
            if file_digest(scored) != row["blake3"]:
                raise EvaluationError("scorer_source_commitment_invalid")
    for case in CASES:
        public, config, private = case_paths(case)
        source = assets.public_file(public)
        if file_digest(safe_file(assets.scorer, public)) != assets.inventory[public]["blake3"]:
            raise EvaluationError("scorer_public_input_commitment_invalid")
        cfg = yaml.safe_load(assets.public_file(config).read_bytes())
        if cfg.get("task_id") != case["task_id"]:
            raise EvaluationError("task_config_binding_invalid")
        if case["track"] == "classification" and cfg.get("score_metric") != case["metric"]:
            raise EvaluationError("classification_metric_binding_invalid")
        if case["track"] == "segmentation":
            refs = [p for p in assets.inventory if p.startswith(private + "masks/" + case["case_id"] + "/")]
        else:
            refs = [private + case["case_id"] + "/" + ("label.json" if case["track"] == "classification" else "boxes.json")]
        if not refs:
            raise EvaluationError("native_reference_missing")
        for reference in refs:
            row = assets.inventory.get(reference)
            if row is None or not row["scorer_only"] or safe_file(assets.scorer, reference).stat().st_size != row["bytes"]:
                raise EvaluationError("native_reference_metadata_invalid")
        cases.append({"case_id": case["case_id"], "track": case["track"], "metric": case["metric"],
                      "public_bytes": source.stat().st_size, "native_reference_files": len(refs)})
    return {"ready": True, "cases": cases, "provider_calls": 0, "gpu_calls": 0,
            "private_reference_contents_parsed": False, "os_actor_isolation_verified": False}


def prepare_run(assets: Assets, output_root: Path, *, diagnostic: bool = False) -> Path:
    preflight(assets)
    output_root = Path(output_root).absolute()
    if output_root.is_relative_to(assets.root) or assets.root.is_relative_to(output_root):
        raise EvaluationError("actor_and_scorer_trees_must_be_disjoint")
    run_id = str(uuid4())
    run = output_root / run_id
    run.mkdir(parents=True, mode=0o700)
    rows = []
    for case in CASES:
        public, config_path, _ = case_paths(case)
        config = yaml.safe_load(assets.public_file(config_path).read_bytes())
        workspace_id = str(uuid4())
        workspace = run / "actors" / workspace_id
        (workspace / "inputs").mkdir(parents=True, mode=0o700)
        (workspace / "outputs").mkdir(mode=0o700)
        relative = "inputs/" + case["input"]
        shutil.copyfile(assets.public_file(public), workspace / relative)
        os.chmod(workspace / relative, 0o600)
        public_config = {key: config[key] for key in ("task_id", "task_type", "task_description", "organ", "modality",
            "classes", "class_names", "tissue_labels", "num_foreground_classes", "iou_threshold") if key in config}
        if "tissue_labels" in public_config:
            public_config["tissue_labels"] = {str(key): value for key, value in public_config["tissue_labels"].items()}
        contract = {"schema": "eva.automedbench-public-case.v1", "case_id": case["case_id"],
                    "track": case["track"], "release_revision": REVISION, "license": case["license"],
                    "upstream_source": case["source"], "public_config": public_config,
                    "input_path": relative, "input_blake3": assets.inventory[public]["blake3"],
                    "submission_path": "outputs/" + case["output"], "metric": case["metric"],
                    "submission_format": submission_format(case), "evaluation_only": True}
        write_once(workspace / "task.json", contract)
        rows.append({**case, "workspace_id": workspace_id, "workspace_relative": "actors/" + workspace_id,
                     "public_input_relative": relative, "public_input_blake3": contract["input_blake3"],
                     "task_contract_blake3": file_digest(workspace / "task.json")})
    write_once(run / "run-manifest.json", {"schema": "eva.automedbench-native-eval-run.v1", "run_id": run_id,
        "release_revision": REVISION, "asset_manifest_blake3": assets.document["document_blake3"], "cases": rows,
        "unique_case_count": 3, "planned_rollouts_per_case": 1, "diagnostic_only": diagnostic,
        "policy_rollouts_executed": 0, "model_evaluation_complete": False, "os_actor_isolation_verified": False,
        "public_workspaces_only": True, "private_references_in_actor_workspace": False})
    return run


def submission_format(case: dict) -> dict:
    if case["track"] == "classification":
        return {"type": "json", "required": ["label"], "label": "one exact public_config.classes string"}
    if case["track"] == "detection":
        return {"type": "json", "required": ["boxes"], "box_required": ["class", "x1", "y1", "x2", "y2"],
                "coordinates": "original image pixels, positive extents within image", "score": "optional [0,1]"}
    return {"type": "nifti", "dtype": "integer", "geometry": "same shape and affine as input",
            "labels": "0 for background or public_config.tissue_labels integer key"}


def validate_result(value: Any, case: dict) -> dict:
    expected = {"status", "track", "case_id", "metric", "task_score_0_1", "present_outputs", "valid_outputs",
                "output_format_valid", "network_disabled", "private_values_exported"}
    if not isinstance(value, dict) or set(value) != expected:
        raise EvaluationError("native_result_shape_invalid")
    if (value["status"] not in {"scored", "invalid_submission"} or value["track"] != case["track"]
            or value["case_id"] != case["case_id"] or value["metric"] != case["metric"]
            or value["network_disabled"] is not True or value["private_values_exported"] is not False):
        raise EvaluationError("native_result_binding_invalid")
    score = value["task_score_0_1"]
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise EvaluationError("native_score_invalid")
    if any(type(value[key]) is not int or value[key] not in (0, 1) for key in ("present_outputs", "valid_outputs")):
        raise EvaluationError("native_format_counts_invalid")
    if type(value["output_format_valid"]) is not bool:
        raise EvaluationError("native_format_flag_invalid")
    if value["status"] == "scored" and (value["valid_outputs"] != 1 or value["present_outputs"] != 1 or not value["output_format_valid"]):
        raise EvaluationError("native_status_invalid")
    return value


def score_case(assets: Assets, run: Path, case_id: str, evaluator_python: Path, *, timeout: float = 180) -> dict:
    manifest = read_document(run / "run-manifest.json")
    if manifest.get("asset_manifest_blake3") != assets.document["document_blake3"] or manifest.get("release_revision") != REVISION:
        raise EvaluationError("run_asset_binding_invalid")
    matches = [row for row in manifest["cases"] if row["case_id"] == case_id]
    if len(matches) != 1:
        raise EvaluationError("case_not_in_run")
    case = matches[0]
    if not any(all(case.get(key) == value for key, value in spec.items()) for spec in CASES):
        raise EvaluationError("case_identity_invalid")
    workspace = run / case["workspace_relative"]
    if workspace.resolve().parent != (run / "actors").resolve():
        raise EvaluationError("workspace_outside_run")
    if file_digest(safe_file(workspace, "task.json")) != case["task_contract_blake3"]:
        raise EvaluationError("task_contract_changed")
    if file_digest(safe_file(workspace, case["public_input_relative"])) != case["public_input_blake3"]:
        raise EvaluationError("public_input_changed")
    artifact = safe_file(workspace, "outputs/" + case["output"])
    score_root = run / "native-scores" / case_id
    score_root.mkdir(parents=True, mode=0o700)
    write_once(score_root / "attempt.json", {"score_id": str(uuid4()), "case_id": case_id, "one_attempt": True})
    preflight(assets)
    with tempfile.TemporaryDirectory(prefix="eva-automed-native-score-", dir="/tmp") as temporary:
        stage = Path(temporary)
        selected = stage / case_id
        selected.mkdir(mode=0o700)
        copied = selected / case["output"]
        shutil.copyfile(artifact, copied)
        commitment = file_digest(copied)
        if commitment != file_digest(artifact):
            raise EvaluationError("submission_changed_during_staging")
        request = {"release_root": str(assets.scorer), "submission_root": str(stage), "track": case["track"],
                   "case_id": case_id, "metric": case["metric"],
                   "reference_commitments": {path: row["blake3"] for path, row in assets.inventory.items()
                       if row["scorer_only"] and case_id in PurePosixPath(path).parts}}
        try:
            completed = subprocess.run([str(evaluator_python), "-I", str(Path(__file__).with_name("native_worker.py"))],
                input=canonical(request), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=stage,
                env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONNOUSERSITE": "1",
                     "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "TMPDIR": str(stage)},
                timeout=timeout, check=False)
            if completed.returncode or len(completed.stdout) > 4096:
                raise EvaluationError("native_worker_failed")
            native = validate_result(json.loads(completed.stdout), case)
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, EvaluationError) as error:
            write_once(score_root / "failure.json", {"case_id": case_id, "status": "failed", "reward": None,
                       "error_code": str(error) if isinstance(error, EvaluationError) else "native_worker_unavailable"})
            raise EvaluationError("native_scoring_failed_no_fallback") from None
    return write_once(score_root / "score.json", {"schema": "eva.automedbench-native-score.v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "case_id": case_id, "run_id": manifest["run_id"],
        "submission_blake3": commitment, "release_revision": REVISION, "native_result": native,
        "diagnostic_only": manifest["diagnostic_only"], "policy_rollout_provenance_verified": False,
        "agent_judged": False, "os_actor_isolation_verified": False})
