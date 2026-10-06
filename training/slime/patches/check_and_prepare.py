#!/usr/bin/env python3
"""Check scoped patches in fresh local clones; never mutate shared checkouts.

No model/package import, GPU work, provider call, network clone or training is
performed. --materialize-root retains new patched source clones for a later,
separately validated launcher; the default --check cleans its private clones.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from uuid import uuid4

from blake3 import blake3


PATCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PATCH_ROOT.parents[2]


def command(argv, *, cwd=None):
    if argv[0] == "git":
        argv = ["git", "-c", "core.hooksPath=/dev/null", *argv[1:]]
    return subprocess.run(argv, cwd=cwd, check=True, capture_output=True).stdout


def check_artifact(name, expected):
    path = PATCH_ROOT / name
    if path.resolve().parent not in (PATCH_ROOT, PATCH_ROOT / "licenses"):
        raise ValueError("Patch artifact path escaped its package")
    if blake3(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Patch artifact commitment differs: {name}")
    return path


def verify_and_apply(sources_root, destination_root, manifest):
    results = {}
    for name, spec in manifest["sources"].items():
        if name not in {"slime-upstream", "Megatron-LM"}:
            raise ValueError("Unexpected upstream source")
        source = (sources_root / name).resolve(strict=True)
        patch = check_artifact(spec["patch"], spec["patch_blake3"])
        check_artifact(spec["license"], spec["license_blake3"])
        commit = command(["git", "rev-parse", f'{spec["commit"]}^{{commit}}'], cwd=source).decode().strip()
        if commit != spec["commit"]:
            raise ValueError("Pinned source commit differs")
        target = destination_root / name
        # Explicitly local clone, no source index/worktree writes. Independent
        # copied objects avoid alternate-store dependencies in retained clones.
        command(["git", "clone", "--local", "--no-hardlinks", "--no-checkout", str(source), str(target)])
        command(["git", "checkout", "--detach", commit], cwd=target)
        if command(["git", "status", "--porcelain"], cwd=target):
            raise ValueError("Fresh checkout is not clean")
        command(["git", "apply", "--check", str(patch)], cwd=target)
        command(["git", "apply", str(patch)], cwd=target)
        changed = command(["git", "diff", "--name-only", "HEAD"], cwd=target).decode().splitlines()
        if set(changed) != set(spec["files"]):
            raise ValueError("Patched file inventory differs")
        files = {}
        for relative in changed:
            content = (target / relative).read_bytes()
            ast.parse(content.decode("utf-8"), filename=relative)
            base = command(["git", "show", f"{commit}:{relative}"], cwd=target)
            files[relative] = {"base_blake3": blake3(base).hexdigest(),
                               "patched_blake3": blake3(content).hexdigest()}
        command(["git", "diff", "--check"], cwd=target)
        results[name] = {"commit": commit, "patch_blake3": spec["patch_blake3"],
                         "application_check": "passed", "python_ast_check": "passed",
                         "files": files}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-root", type=Path, default=REPO_ROOT.parent)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check only in temporary clones (default)")
    mode.add_argument("--materialize-root", type=Path, help="Keep clones in a new exclusive directory")
    parser.add_argument("--receipt", type=Path, help="Optionally write a new exclusive verification receipt")
    args = parser.parse_args()
    manifest = json.loads((PATCH_ROOT / "manifest.json").read_text())
    if args.materialize_root is not None:
        target = args.materialize_root.resolve()
        target.mkdir(mode=0o700, parents=False, exist_ok=False)
        manager = nullcontext(str(target))
    else:
        manager = TemporaryDirectory(prefix="eva-qwen-runtime-patches-", dir="/tmp")
    with manager as temporary:
        results = verify_and_apply(args.sources_root.resolve(strict=True), Path(temporary), manifest)
    document = {
        "schema": "eva.qwen35-runtime-patch-check.v1", "receipt_id": str(uuid4()),
        "status": "application-and-ast-passed-not-gpu-validated",
        "sources": results, "shared_checkouts_modified": False,
        "network_calls": 0, "gpu_calls": 0, "provider_calls": 0,
        "retained_materialization_root": str(args.materialize_root.resolve()) if args.materialize_root else None,
    }
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    document["receipt_blake3"] = blake3(encoded).hexdigest()
    if args.receipt is not None:
        with args.receipt.open("x") as stream:
            json.dump(document, stream, sort_keys=True, indent=2)
            stream.write("\n")
    print(json.dumps(document, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
