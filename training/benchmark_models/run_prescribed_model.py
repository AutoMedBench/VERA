"""Trusted, public-input-only analysis capability; never execute actor code.

Host binds --workspace and --model-manifest. The actor supplies only case IDs and
bounded numeric options through its separately implemented MCP transport. Actual
raw model outputs are retained for actor-authored containerized postprocessing.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time
import uuid

from blake3 import blake3

if __package__:
    from .prepare_public_models import GIT_SOURCES, HF_MODELS
    from . import enhancement, segmentation, synthesis_runtime, vqa_runtime, report_runtime
else:
    from prepare_public_models import GIT_SOURCES, HF_MODELS
    import enhancement
    import segmentation
    import synthesis_runtime
    import vqa_runtime
    import report_runtime

CANONICAL_LABELS = {"akiec": "actinic_keratoses", "bcc": "basal_cell_carcinoma", "bkl": "benign_keratosis_like_lesions", "df": "dermatofibroma", "mel": "melanoma", "nv": "melanocytic_nevi", "vasc": "vascular_lesions"}
SUPPORTED_TRACKS = ("classification", "detection", "segmentation", "synthesis", "enhancement", "vqa", "report")
GPU_LOCK = Path("/tmp/eva-automed-prescribed-model-gpu.lock")
TRACK_HELPERS = {"classification": (), "detection": (),
                 "segmentation": ("segmentation.py", "_track_io.py"),
                 "enhancement": ("enhancement.py", "_track_io.py"),
                 "synthesis": ("synthesis_runtime.py",), "vqa": ("vqa_runtime.py", "_track_io.py"),
                 "report": ("report_runtime.py", "_track_io.py")}
TRACK_PACKAGES = {"classification": (), "detection": ("ultralytics",),
                  "segmentation": ("TotalSegmentator", "nnunetv2", "nibabel"),
                  "enhancement": ("deepinv",), "synthesis": ("nibabel",),
                  "vqa": ("tokenizers", "accelerate"), "report": ("tokenizers", "accelerate")}
MODEL_DEPENDENCIES = {"vqa": ("vqa_vision", "vqa_source"), "report": ("report_vision",)}
# Prospective allowance/headroom, not hard total-VRAM caps or measured GPU fit.
GPU_PROFILES_GIB = {"vqa": (24, 48), "report": (12, 24)}


def digest_file(path):
    digest = blake3()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confined(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unconfined public input/output path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Public path escapes bound workspace")
    return path


def verify_model_binding(model, track):
    """Bind HF snapshots, Git source, or release partitions without model import."""
    if track in HF_MODELS:
        repo, revision, _ = HF_MODELS[track]
    elif track in GIT_SOURCES:
        repo, revision = GIT_SOURCES[track]
    elif track == "segmentation":
        repo, revision = segmentation.REPOSITORY, segmentation.REVISION
    else:
        raise ValueError("No prescribed model for this track")
    if model.get("repository") != repo or model.get("revision") != revision:
        raise ValueError("Prescribed analysis model identity mismatch")
    rows = model.get("files")
    if not isinstance(rows, list) or not rows or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Model requires a nonempty, unique extracted-file inventory")
    model_path = Path(model["path"]).resolve(strict=True)
    for row in rows:
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Invalid pinned model inventory path")
        # HF snapshot files legitimately link into their immutable blob cache.
        path = model_path / relative
        if path.stat().st_size != row["bytes"] or digest_file(path) != row["blake3"]:
            raise ValueError("Pinned public model content changed")
    if track == "enhancement":
        enhancement.validate_model(model)
    elif track == "segmentation":
        segmentation.validate_model(model)
    elif track == "synthesis":
        inventory = {row["path"]: row for row in rows}
        for name, expected in synthesis_runtime.PINNED_FILES.items():
            row = inventory.get(name, {})
            if (row.get("bytes"), row.get("blake3")) != expected:
                raise ValueError("Pinned PlainCNN source/weight binding mismatch")
    return model_path


def preflight(workspace, model_manifest, track, case_ids):
    workspace = workspace.resolve(strict=True)
    task = json.loads((workspace / "task.json").read_text())
    inputs_path = workspace / "inputs-manifest.json"
    inputs = json.loads(inputs_path.read_text())
    if task["track"] != track or not task.get("no_private_reference_access"):
        raise ValueError("Public track binding mismatch")
    if not 1 <= len(case_ids) <= 32 or len(set(case_ids)) != len(case_ids):
        raise ValueError("Request requires 1..32 distinct case IDs")
    allowed = set(inputs["case_ids"]) & set(task["case_ids"])
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", case) or case not in allowed for case in case_ids):
        raise ValueError("Case ID is outside the immutable public input manifest")
    selected = {}
    for case in case_ids:
        rows = [row for row in inputs["files"] if PurePosixPath(row["path"]).parts[:2] == ("inputs", case)]
        if not rows:
            raise ValueError("No retained public input for selected case")
        for row in rows:
            path = confined(workspace, row["path"])
            if path.stat().st_size != row["bytes"] or digest_file(path) != row["blake3"]:
                raise ValueError("Public input commitment changed")
        selected[case] = rows
    assets = json.loads(model_manifest.read_text())
    model = assets["models"][track]
    model_path = verify_model_binding(model, track)
    if model_path.is_relative_to(workspace):
        raise ValueError("Trusted model assets must be outside the actor workspace")
    for key in MODEL_DEPENDENCIES.get(track, ()):
        dependency = verify_model_binding(assets["models"][key], key)
        if dependency.is_relative_to(workspace):
            raise ValueError("Trusted model dependencies must be outside the actor workspace")
    return {"workspace": workspace, "task": task, "selected": selected, "model": model,
            "assets": assets["models"], "cache_root": Path(assets["cache_root"]).resolve(strict=True) if track in MODEL_DEPENDENCIES else None,
            "input_manifest_blake3": digest_file(inputs_path), "model_manifest_blake3": digest_file(model_manifest)}


def single_image(binding, case):
    rows = [row for row in binding["selected"][case] if Path(row["path"]).suffix.lower() in (".jpg", ".jpeg", ".png")]
    if len(rows) != 1:
        raise ValueError("This analysis backend requires exactly one public image per case")
    return confined(binding["workspace"], rows[0]["path"])


def classify(binding, torch, progress=None):
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    path = binding["model"]["path"]
    processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(path, local_files_only=True).eval().to("cuda")
    results = []
    with torch.inference_mode():
        for case in binding["selected"]:
            with Image.open(single_image(binding, case)) as source:
                inputs = processor(images=source.convert("RGB"), return_tensors="pt").to("cuda")
            logits = model(**inputs).logits[0].float()
            index = int(logits.argmax())
            abbreviation = model.config.id2label[index]
            results.append({"case_id": case, "logits": logits.cpu().tolist(), "probabilities": logits.softmax(-1).cpu().tolist(),
                            "class_index": index, "model_label": abbreviation, "canonical_label": CANONICAL_LABELS[abbreviation]})
            if progress:
                progress(results[-1])
    return results


def detect(binding, torch, confidence, progress=None):
    from ultralytics import YOLO
    model = YOLO(str(Path(binding["model"]["path"]) / "YOLOv8x-best.pt"))
    results = []
    for case in binding["selected"]:
        output = model.predict(source=str(single_image(binding, case)), device=0, conf=confidence, verbose=False)[0]
        boxes = output.boxes
        results.append({"case_id": case, "xyxy": boxes.xyxy.cpu().tolist(), "confidence": boxes.conf.cpu().tolist(),
                        "class_index": boxes.cls.long().cpu().tolist(), "names": output.names, "original_shape": list(output.orig_shape)})
        if progress:
            progress(results[-1])
    return results


def run_track(binding, torch, *, output_root, confidence, sigma, hu_min, hu_max, progress,
              prompt=None, max_new_tokens=None, multi_image_mode=None):
    track = binding["task"]["track"]
    if track == "classification":
        return classify(binding, torch, progress)
    if track == "detection":
        return detect(binding, torch, confidence, progress)
    if track == "segmentation":
        return segmentation.segment(binding, torch, output_root=output_root, progress=progress)
    if track == "synthesis":
        return synthesis_runtime.run_synthesis(binding, torch, output_root=output_root, progress=progress)
    if track == "enhancement":
        return enhancement.enhance(binding, torch, output_root=output_root, sigma=sigma,
                                   hu_min=hu_min, hu_max=hu_max, progress=progress)
    if track == "vqa":
        return vqa_runtime.run_vqa(binding, torch, output_root=output_root, assets=binding["assets"],
            cache_root=binding["cache_root"], multi_image_mode=multi_image_mode,
            max_new_tokens=max_new_tokens, progress=progress)
    if track == "report":
        return report_runtime.run_report(binding, torch, output_root=output_root, assets=binding["assets"],
            cache_root=binding["cache_root"], prompt=prompt, max_new_tokens=max_new_tokens, progress=progress)
    raise ValueError("Unsupported prescribed model track")


def retain_case_artifacts(binding, output, audit, result):
    """Copy actual model bytes to host-only audit before claiming case completion."""
    records = []
    for key in ("artifact", "raw_output", "candidate_output", "request", "model_input", "smoke_output"):
        row = result.get(key)
        if row is None:
            continue
        source = confined(binding["workspace"], row["path"])
        if not source.is_relative_to(output):
            raise ValueError("Model artifact escapes this host-generated job")
        destination = audit / "artifacts" / source.relative_to(output)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, 8 * 1024**2)
        destination.chmod(0o600)
        if destination.stat().st_size != row["bytes"] or digest_file(destination) != row["blake3"]:
            raise ValueError("Actual model artifact commitment changed during audit copy")
        records.append({"kind": key, "case_id": result["case_id"], "path": destination.relative_to(audit).as_posix(),
                        "bytes": row["bytes"], "blake3": row["blake3"]})
    if binding["task"]["track"] not in {"classification", "detection"} and not records:
        raise ValueError("Volumetric/array model runtime returned no retained artifact")
    return records


def configure_overlay(binding, audit):
    """Only this fresh CLI process: no shared package changes or model imports."""
    track = binding["task"]["track"]
    if track not in {"vqa", "report"}:
        return {}
    if any(name in sys.modules for name in ("transformers", "tokenizers", "huggingface_hub", "accelerate")):
        raise RuntimeError("Legacy analysis models require a fresh isolated worker")
    cache = Path(binding["cache_root"]).resolve(strict=True)
    overlay = cache / "runtimes" / ("llava-transformers436" if track == "vqa" else "chexagent-transformers440")
    if not (overlay / "transformers/__init__.py").is_file():
        raise RuntimeError("Prepared legacy Transformers overlay is missing")
    paths = [str(overlay)]
    if track == "vqa":
        paths.append(binding["assets"]["vqa_source"]["path"])
    environment = {"PYTHONPATH": os.pathsep.join(paths), "HF_HUB_CACHE": str(cache / "hf-cache"),
        "HF_HOME": str(cache / "runtime-cache"), "HF_MODULES_CACHE": str(audit / "hf-modules"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
    os.environ.update(environment)
    sys.path[:0] = paths
    return {"pythonpath": paths, "hf_modules_cache": environment["HF_MODULES_CACHE"], "offline": True}


def retain_model_sources(binding, audit):
    sources = []
    track = binding["task"]["track"]
    if track not in {"vqa", "report"}:
        return sources
    for key in (track, *MODEL_DEPENDENCIES[track]):
        model = binding["assets"][key]
        for row in model["files"]:
            if Path(row["path"]).suffix not in {".py", ".json", ".md"} and not Path(row["path"]).name.upper().startswith(("LICENSE", "NOTICE")):
                continue
            source = Path(model["path"]) / row["path"]
            copied = audit / "executed-model-source" / key / row["path"]
            copied.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            copied.write_bytes(source.read_bytes())
            if digest_file(copied) != row["blake3"]:
                raise ValueError("Pinned model source changed before audit retention")
            sources.append({"asset": key, "path": copied.relative_to(audit).as_posix(), "blake3": row["blake3"]})
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Host-bound, never an actor-controlled path")
    parser.add_argument("--model-manifest", type=Path, required=True, help="Host-bound prepared asset manifest")
    parser.add_argument("--track", choices=SUPPORTED_TRACKS, required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, help="Explicit enhancement noise level in [1e-4,0.2]")
    parser.add_argument("--hu-min", type=float, help="Explicit enhancement normalization window lower bound")
    parser.add_argument("--hu-max", type=float, help="Explicit enhancement normalization window upper bound")
    parser.add_argument("--prompt", help="Explicit report-generation instruction, 10..1024 characters")
    parser.add_argument("--max-new-tokens", type=int, help="Explicit report1..512 / VQA1..256 generation limit")
    parser.add_argument("--multi-image-mode", choices=["montage"], help="Explicit all-images VQA adaptation")
    parser.add_argument("--job-id", type=uuid.UUID, help="Optional host-generated UUID for asynchronous status binding")
    parser.add_argument("--audit-root", type=Path, help="Host-owned root outside every actor-writable mount; required for GPU execution")
    parser.add_argument("--execute-gpu", action="store_true", help="Explicit host admission; without this, CPU-only preflight")
    args = parser.parse_args()
    if not 0.001 <= args.confidence <= 0.99:
        raise ValueError("Detection confidence outside bounded range")
    if args.track == "enhancement":
        enhancement.validate_controls(args.sigma, args.hu_min, args.hu_max)
    elif any(value is not None for value in (args.sigma, args.hu_min, args.hu_max)):
        raise ValueError("Enhancement controls cannot be supplied to another track")
    if args.track == "report":
        report_runtime.validate_controls(args.prompt, args.max_new_tokens)
        if args.multi_image_mode is not None:
            raise ValueError("Report is single-frontal-image, not a montage task")
    elif args.track == "vqa":
        if args.prompt is not None or args.multi_image_mode != "montage" or type(args.max_new_tokens) is not int or not 1 <= args.max_new_tokens <= 256:
            raise ValueError("VQA requires explicit montage and 1..256 tokens; no report prompt")
    elif any(value is not None for value in (args.prompt, args.max_new_tokens, args.multi_image_mode)):
        raise ValueError("Generative model controls supplied to an incompatible track")
    binding = preflight(args.workspace, args.model_manifest, args.track, args.case_ids)
    if args.track == "enhancement" and tuple(binding["task"].get("public_config", {}).get("intensity_range", enhancement.OUTPUT_HU_RANGE)) != enhancement.OUTPUT_HU_RANGE:
        raise ValueError("Public enhancement output intensity range differs from pinned task")
    if not args.execute_gpu:
        print(json.dumps({"status": "preflight_passed", "track": args.track, "case_count": len(args.case_ids), "gpu_loaded": False}))
        return
    if args.audit_root is None or args.audit_root.resolve().is_relative_to(binding["workspace"]):
        raise ValueError("GPU jobs require a host-owned audit root outside the actor workspace")
    job_id = str(args.job_id or uuid.uuid4())
    output = confined(binding["workspace"], f"outputs/agents_outputs/prescribed-model-jobs/{job_id}")
    audit = args.audit_root.resolve() / job_id
    audit.mkdir(parents=True, exist_ok=False, mode=0o700)
    output.mkdir(parents=True, exist_ok=False)
    (audit / "executed-helper.py").write_bytes(Path(__file__).read_bytes())
    sources = []
    for name in ("prepare_public_models.py", *TRACK_HELPERS[args.track]):
        source = Path(__file__).parent / name
        copied = audit / "executed-modules" / name
        copied.parent.mkdir(exist_ok=True, mode=0o700)
        copied.write_bytes(source.read_bytes())
        sources.append({"path": copied.relative_to(audit).as_posix(), "blake3": digest_file(copied)})
    receipt = {"schema": "eva.prescribed-public-model-job.v1", "job_id": job_id, "track": args.track,
               "authored_by": "provided_analysis_tool", "coding_orchestrator": binding["task"]["coding_orchestrator"],
               "analysis_model": binding["model"], "input_manifest_blake3": binding["input_manifest_blake3"],
               "model_manifest_blake3": binding["model_manifest_blake3"], "selected_inputs": binding["selected"],
               "helper_source_blake3": digest_file(Path(__file__)), "case_ids": args.case_ids, "status": "queued",
               "private_reference_access": False, "clinical_score": None, "executed_module_sources": sources,
               "inference_controls": ({"sigma": args.sigma, "hu_min": args.hu_min, "hu_max": args.hu_max}
                                      if args.track == "enhancement" else {"confidence": args.confidence}
                                      if args.track == "detection" else {"prompt": args.prompt, "max_new_tokens": args.max_new_tokens}
                                      if args.track == "report" else {"multi_image_mode": args.multi_image_mode, "max_new_tokens": args.max_new_tokens}
                                      if args.track == "vqa" else {}), "host_retained_artifacts": [],
               "executed_model_sources": [],
               "analysis_dependencies": {key: binding["assets"][key] for key in MODEL_DEPENDENCIES.get(args.track, ())}}
    submitted = time.monotonic()
    started = None
    def write_receipt():
        payload = json.dumps(receipt, indent=2) + "\n"
        for directory in (audit, output):
            temporary = directory / "receipt.json.tmp"
            temporary.write_text(payload)
            temporary.replace(directory / "receipt.json")
    write_receipt()
    completed_cases = set()
    def progress(result):
        if result["case_id"] not in args.case_ids or result["case_id"] in completed_cases:
            raise ValueError("Model progress has an unexpected or repeated case")
        records = retain_case_artifacts(binding, output, audit, result)
        receipt["host_retained_artifacts"].extend(records)
        completed_cases.add(result["case_id"])
        write_receipt()
        for directory in (audit, output):
            with (directory / "case-outputs.jsonl").open("a") as stream:
                stream.write(json.dumps(result, sort_keys=True) + "\n")
            with (directory / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps({"event": "case_inference_complete", "case_id": result["case_id"], "elapsed_s": time.monotonic() - started}) + "\n")
    try:
        receipt["executed_model_sources"] = retain_model_sources(binding, audit)
        receipt["isolated_runtime"] = configure_overlay(binding, audit)
        write_receipt()
        descriptor = os.open(GPU_LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "a") as lock:
            # A host-supervised detached job may wait here. Waiting acquires no
            # CUDA context; the queued receipt is already externally readable.
            fcntl.flock(lock, fcntl.LOCK_EX)
            started = time.monotonic()
            receipt.update({"status": "running", "queue_seconds": started - submitted})
            write_receipt()
            os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
            import torch
            torch.set_num_threads(4)
            free, total = torch.cuda.mem_get_info()
            budget_gib, headroom_gib = GPU_PROFILES_GIB.get(args.track, (8, 16))
            budget = budget_gib * 1024**3
            if free < headroom_gib * 1024**3:
                raise RuntimeError("Insufficient free GPU headroom for bounded analysis admission")
            torch.cuda.set_per_process_memory_fraction(budget / total)
            receipt.update({"device": "cuda:0", "gpu_name": torch.cuda.get_device_name(),
                            "torch_allocator_limit_bytes": budget, "hard_total_vram_limit": False,
                            "required_free_headroom_bytes": headroom_gib * 1024**3,
                            "profile_is_measured_fit_guarantee": False,
                            "gpu_free_bytes_before": free,
                            "runtime": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "Pillow", *TRACK_PACKAGES[args.track])}})
            write_receipt()
            results = run_track(binding, torch, output_root=output, confidence=args.confidence,
                                sigma=args.sigma, hu_min=args.hu_min, hu_max=args.hu_max, progress=progress,
                                prompt=args.prompt, max_new_tokens=args.max_new_tokens, multi_image_mode=args.multi_image_mode)
            if len(results) != len(args.case_ids) or {row["case_id"] for row in results} != set(args.case_ids) or completed_cases != set(args.case_ids):
                raise ValueError("Model runtime did not complete each selected case exactly once")
            torch.cuda.synchronize()
            raw = output / "raw-output.json"
            raw.write_text(json.dumps(results, indent=2) + "\n")
            (audit / "raw-output.json").write_bytes(raw.read_bytes())
            receipt.update({"status": "complete", "success": True, "raw_output_path": str(raw), "raw_output_blake3": digest_file(raw),
                            "inference_completed": True,
                            "execution_outcome": {"kind": "trusted_python_function_return", "success": True, "os_process_exit_observed": False},
                            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "raw_output_sample": results[0]})
    except BaseException as error:
        receipt.update({"status": "failed", "success": False, "error_type": type(error).__name__})
        raise
    finally:
        receipt["wall_s"] = time.monotonic() - started if started is not None else None
        receipt["total_seconds_including_queue"] = time.monotonic() - submitted
        write_receipt()
        print(json.dumps({"status": receipt["status"], "job_id": job_id, "receipt": str(audit / "receipt.json"), "actor_output": str(output)}))


if __name__ == "__main__":
    main()
