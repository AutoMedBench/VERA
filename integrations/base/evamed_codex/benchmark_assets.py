"""Build the private runner's public/scorer split from verified HF downloads."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

from blake3 import blake3


REVISION = "8928073d5c3f3b842a4a4278d9b44f6e8ceaa9c5"
REPOSITORY = "operator/AutoMedBench-Lite-release"


def prepare_release(root: Path, output: Path) -> Path:
    """Copy exact pinned originals; local workflow overlays are not scorer code.

    No policy is launched. The returned acquisition document is understood by
    EVA-Agent's TrackRelease and prepare_track_run. Existing complete output is
    reopened, while a partial acquisition must be resumed in the same directory.
    """
    from training.automedbench_lite.adapter import read_document, write_once

    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("benchmark_assets_must_remain_in_workspace")
    document = json.loads((root / "evamed-codex/receipts/dataset-downloads.json").read_text())
    selected = [row for row in document["datasets"] if row["repo_id"] == REPOSITORY]
    if len(selected) != 1 or selected[0].get("status") != "complete":
        raise ValueError("complete_verified_benchmark_download_required")
    dataset = selected[0]
    if dataset["revision"] != REVISION or not dataset.get("local_overlays_restored"):
        raise ValueError("benchmark_revision_or_overlay_restore_invalid")
    source = root / dataset["path"]
    receipt = output / "download-all3.json"
    if receipt.exists():
        retained = read_document(receipt, maximum=16 * 1024**2)
        if retained.get("revision") != REVISION or len(retained.get("inventory", [])) != dataset["verified_files"]:
            raise ValueError("existing_split_receipt_mismatch")
        return receipt
    public, scorer = output / "public-release", output / "scorer-only-release"
    public.mkdir(parents=True, exist_ok=True, mode=0o700)
    scorer.mkdir(parents=True, exist_ok=True, mode=0o700)
    scorer.chmod(0o700)

    def copy(row: dict) -> dict:
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("benchmark_inventory_path_invalid")
        original = root / row["pristine_copy"] if row.get("pristine_copy") else source / str(relative)
        private = "private" in relative.parts
        destinations = [scorer / str(relative)] + ([] if private else [public / str(relative)])
        source_digest, native_digest = hashlib.sha256(), blake3()
        with original.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                source_digest.update(chunk)
                native_digest.update(chunk)
        if original.stat().st_size != row["bytes"] or source_digest.hexdigest() != row["sha256"]:
            raise ValueError("benchmark_pristine_source_changed")
        for target in destinations:
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if target.exists():
                observed = blake3()
                with target.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                        observed.update(chunk)
                if target.stat().st_size == row["bytes"] and observed.digest() == native_digest.digest():
                    continue
                raise ValueError("existing_split_file_changed")
            shutil.copyfile(original, target)
            target.chmod(0o400)
        return {"path": str(relative), "bytes": row["bytes"], "blake3": native_digest.hexdigest(),
                "scorer_only": private, "upstream_sha256": row["sha256"]}

    with ThreadPoolExecutor(max_workers=8) as executor:
        inventory = sorted(executor.map(copy, dataset["files"]), key=lambda row: row["path"])
    write_once(receipt, {"schema": "eva.automedbench-lite-pinned-assets.v1",
                        "repository": REPOSITORY, "revision": REVISION,
                        "public_root": str(public), "scorer_only_root": str(scorer),
                        "inventory": inventory, "source": "verified_workspace_dataset_snapshot",
                        "upstream_file_count": len(inventory), "local_overlays_excluded": True,
                        "policy_calls": 0, "private_references_exposed_to_actor": False})
    return receipt
