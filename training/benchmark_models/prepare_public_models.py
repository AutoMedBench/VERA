"""Download only prescribed public AutoMedLite inference assets; never load a model.

Run with the existing GPU environment's Python (HF Hub, requests, blake3).
The cache may be mounted read-only into an isolated scientific runtime. No tokens,
private benchmark references, scorer files, or training data are copied here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import threading
import uuid
import zipfile

import blake3


HF_MODELS = {
    "classification": ("spycoder/vit-base-patch16-224-in21k-enhanced-ham10000", "13b148c76ec7a29663604b7278db1fe1ab6eb48c", ["README.md", "config.json", "preprocessor_config.json", "model.safetensors"]),
    "detection": ("AbdulManafSahito/YOLOv8-Pediatric-Wrist-Anomaly-Detection", "6075071a1214c9cc99c1619293514620a3e45a99", ["README.md", "YOLOv8x-best.pt"]),
    "enhancement": ("deepinv/drunet", "7e079a6800958ae777b48e41fc1202a2ee21ecbe", ["README.md", "drunet_deepinv_gray_finetune_26k.pth"]),
    "report": ("StanfordAIMI/CheXagent-2-3b", "8f19b53a2eceda4c33b0acec6c81fbc293ad80d0", ["*.json", "*.py", "*.safetensors", "merges.txt", "vocab.json", "README.md", "LICENSE*"]),
    "report_vision": ("StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli", "f0edbf5d90dba44edb7f4f96d8663537cb0749bf", ["*.json", "*.safetensors", "spiece.model", "README.md", "LICENSE*"]),
    "vqa": ("microsoft/llava-med-v1.5-mistral-7b", "91bb16c122001ddc9cf1fd36ce1dae09448943a2", ["*.json", "*.safetensors", "tokenizer.model", "README.md", "data_summary_card.md", "LICENSE*"]),
    "vqa_vision": ("openai/clip-vit-large-patch14-336", "ce19dc912ca5cd21c8a653c79e251e808ccabcd1", ["*.json", "README.md", "merges.txt", "vocab.json", "pytorch_model.bin", "LICENSE*"]),
}
GIT_SOURCES = {
    "synthesis": ("Roldbach/autoencoder_ct_3d_super_resolution", "87ace8f44e77721fe43e2b249fbba211bbab13a3"),
    "vqa_source": ("microsoft/LLaVA-Med", "30697ca50b5c29a8e955c99330b259776aef27b9"),
}
TOTAL_ASSETS = [
    (127050286, "Dataset291_TotalSegmentator_part1_organs_1559subj.zip", 233742255),
    (127050300, "Dataset292_TotalSegmentator_part2_vertebrae_1532subj.zip", 234050721),
    (127050328, "Dataset293_TotalSegmentator_part3_cardiac_1559subj.zip", 234190318),
    (127050360, "Dataset294_TotalSegmentator_part4_muscles_1559subj.zip", 233625081),
    (127050376, "Dataset295_TotalSegmentator_part5_ribs_1559subj.zip", 234016576),
]
LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def file_record(path: Path, root: Path):
    digest = blake3.blake3()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "blake3": digest.hexdigest()}


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def download(url: str, path: Path, expected_size=None, git_blob=None):
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        if git_blob is None or git_blob_hash(path) == git_blob:
            return
        raise ValueError(f"Existing source content does not match Git blob: {path.name}")
    temporary = path.with_name(path.name + ".incomplete")
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(8 * 1024 * 1024):
                stream.write(chunk)
    if expected_size is not None and temporary.stat().st_size != expected_size:
        raise ValueError(f"Wrong public asset size: {path.name}")
    if git_blob is not None and git_blob_hash(temporary) != git_blob:
        raise ValueError(f"Wrong Git source content: {path.name}")
    temporary.replace(path)


def git_blob_hash(path: Path):
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_zip_members(archive: zipfile.ZipFile, expected_root: str):
    members = archive.infolist()
    if sum(info.file_size for info in members) > 4 * 1024**3:
        raise ValueError("Public weight archive exceeds bounded extraction size")
    selected = []
    for info in members:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Unsafe public weight archive member")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError("Archive symlinks are not accepted")
        # Official TotalSegmentator archives contain inert macOS sidecars;
        # omit these explicitly rather than widening extraction destinations.
        if path.parts[0] == "__MACOSX" or path.name == ".DS_Store" or path.name.startswith("._"):
            continue
        if path.parts[0] != expected_root:
            raise ValueError("Unexpected public weight archive root")
        selected.append(info)
    return selected


def prepare_hf(key: str, root: Path):
    from huggingface_hub import HfApi, snapshot_download
    repo, revision, patterns = HF_MODELS[key]
    info = HfApi(token=False).model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision or info.gated:
        raise ValueError("Revision changed or manual access terms required")
    path = Path(snapshot_download(repo, revision=revision, allow_patterns=patterns,
                                  cache_dir=root / "hf-cache", token=False, max_workers=2))
    expected = {s.rfilename: s for s in info.siblings}
    records = []
    for file in sorted(path.rglob("*")):
        if not file.is_file():
            continue
        name = file.relative_to(path).as_posix()
        sibling = expected[name]
        if sibling.size is not None and file.stat().st_size != sibling.size:
            raise ValueError(f"Size mismatch: {repo}/{name}")
        record = file_record(file, path)
        if sibling.lfs:
            record["upstream_sha256"] = sibling.lfs.sha256
        records.append(record)
    return {"kind": "huggingface_snapshot", "repository": repo, "revision": revision,
            "path": str(path), "license": (info.card_data or {}).get("license"),
            "files": records, "total_bytes": sum(r["bytes"] for r in records),
            "source": f"https://huggingface.co/{repo}/tree/{revision}",
            "custom_code_executed": False, "gpu_loaded": False}


def prepare_git(key: str, root: Path):
    repo, revision = GIT_SOURCES[key]
    # Immutable commit archive avoids an unrelated GitHub metadata API quota.
    # Revisions were resolved from the official repository before preparation.
    prefix = f"{repo.rsplit('/', 1)[1]}-{revision}"
    archive_path = root / "archives" / f"{prefix}.zip"
    source_url = f"https://codeload.github.com/{repo}/zip/{revision}"
    download(source_url, archive_path)
    path = root / "sources" / prefix
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in safe_zip_members(archive, prefix):
            name = str(PurePosixPath(member.filename).relative_to(prefix))
            selected = name.endswith((".py", ".toml", ".txt", ".md", ".yaml", ".yml")) or PurePosixPath(name).name.startswith(("LICENSE", "NOTICE"))
            selected |= name == "weight/PlainCNN_trilinear_interpolation_x4.pth"
            if member.is_dir() or not selected:
                continue
            file = path / name
            file.parent.mkdir(parents=True, exist_ok=True)
            if not file.exists():
                with archive.open(member) as source, file.open("xb") as output:
                    shutil.copyfileobj(source, output, 8 * 1024**2)
            elif file.stat().st_size != member.file_size:
                raise ValueError("Existing pinned source size differs")
            records.append(file_record(file, path))
    return {"kind": "pinned_public_source", "repository": repo, "revision": revision,
            "path": str(path), "files": records, "total_bytes": sum(r["bytes"] for r in records),
            "archive": {**file_record(archive_path, root), "source": source_url},
            "source": f"https://github.com/{repo}/tree/{revision}", "custom_code_executed": False, "gpu_loaded": False}


def prepare_total(root: Path):
    path = root / "totalsegmentator" / "nnunet" / "results"
    records, archives = [], []
    for asset_id, name, size in TOTAL_ASSETS:
        source_url = f"https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/{name}"
        archive_path = root / "archives" / name
        download(source_url, archive_path, size)
        archive_record = file_record(archive_path, root)
        archive_record.update({"asset_id": asset_id, "source": source_url})
        archives.append(archive_record)
        with zipfile.ZipFile(archive_path) as archive:
            members = safe_zip_members(archive, name[:-4])
            for member in members:
                target = path / member.filename
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                elif not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                elif target.stat().st_size != member.file_size:
                    raise ValueError("Existing extracted public model differs")
        print(json.dumps({"event": "totalsegmentator_asset_ready", "name": name, "at": now()}), flush=True)
    for file in sorted(path.rglob("*")):
        if file.is_file():
            records.append(file_record(file, path))
    return {"kind": "public_release_weights", "repository": "wasserth/TotalSegmentator", "release_id": 121996387,
            "revision": "v2.0.0-weights", "path": str(path), "archives": archives, "files": records,
            "total_bytes": sum(r["bytes"] for r in records), "task": "total", "fast": False,
            "task_ids": [291, 292, 293, 294, 295], "folds": [0], "resample_mm": 1.5, "gpu_loaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--only", nargs="+", choices=[*HF_MODELS, *GIT_SOURCES, "segmentation"])
    args = parser.parse_args()
    root, run_root = args.cache_root.resolve(), args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 70 * 1024**3:
        raise RuntimeError("Require at least 70 GiB free for bounded public downloads")
    receipt = {"schema": "eva.automed-public-model-preparation.v1", "run_id": str(uuid.uuid4()),
               "started_at": now(), "cache_root": str(root), "status": "running", "models": {}, "errors": {},
               "gpu_jobs_started": 0, "credentials_copied": False, "private_reference_inputs": False}
    save_json(run_root / "preparation.json", receipt)
    def job(key):
        print(json.dumps({"event": "start", "model": key, "at": now()}), flush=True)
        result = prepare_hf(key, root) if key in HF_MODELS else prepare_git(key, root) if key in GIT_SOURCES else prepare_total(root)
        with LOCK:
            receipt["models"][key] = result
            save_json(run_root / "preparation.json", receipt)
        print(json.dumps({"event": "ready", "model": key, "bytes": result["total_bytes"], "path": result["path"], "at": now()}), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(job, key): key for key in (args.only or [*HF_MODELS, *GIT_SOURCES, "segmentation"])}
        for future in as_completed(futures):
            key = futures[future]
            try:
                future.result()
            except Exception as error:
                # Never print exception URLs/headers: they can contain signed redirects.
                with LOCK:
                    receipt["errors"][key] = {"error_type": type(error).__name__, "http_status": getattr(getattr(error, "response", None), "status_code", None), "message": "Public asset preparation failed; no model was loaded."}
                    save_json(run_root / "preparation.json", receipt)
                print(json.dumps({"event": "failed", "model": key, "error_type": type(error).__name__, "at": now()}), flush=True)
    receipt["status"] = "failed" if receipt["errors"] else "complete"
    receipt["completed_at"] = now()
    save_json(run_root / "preparation.json", receipt)
    return 1 if receipt["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
