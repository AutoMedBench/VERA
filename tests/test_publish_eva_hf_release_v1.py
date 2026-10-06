from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import pytest
from blake3 import blake3


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_eva_hf_release_v1.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("publish_eva_hf_release_v1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_dataset(root: Path, *, repo_id: str, prefix: str) -> None:
    (root / "data").mkdir(parents=True)
    (root / "verifier" / "scripts").mkdir(parents=True)
    for name, rows in (("train-00000.jsonl", 2), ("train-00001.jsonl", 3)):
        data = b"".join(
            json.dumps({"id": f"{prefix}-{name}-{index}"}, sort_keys=True).encode() + b"\n"
            for index in range(rows)
        )
        (root / "data" / name).write_bytes(data)
    for name, data in {
        "README.md": f"# {prefix}\n".encode(),
        "VERSION.json": b'{"version":"v1"}\n',
        "MIGRATION.md": b"# Migration\n",
        "migration-report.json": b'{"status":"verified"}\n',
    }.items():
        (root / name).write_bytes(data)
    verifier = root / "verifier" / "scripts" / "verify_eva_hf_dataset_v1.py"
    verifier.write_text(
        """#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-root", type=Path, required=True)
parser.add_argument("--kind", required=True)
parser.add_argument("--expected-rows", type=int, required=True)
args = parser.parse_args()
if "HF_TOKEN" in os.environ or os.environ.get("PYTHONPATH") != "":
    raise SystemExit(17)
if os.environ.get("HF_HUB_OFFLINE") != "1":
    raise SystemExit(18)
manifest = json.loads((args.dataset_root / "manifest.json").read_text())
bundle = json.loads((args.dataset_root / "verifier/manifest.json").read_text())
print(json.dumps({
    "schema": "test.offline-verifier-receipt.v1",
    "valid": True,
    "dataset_kind": args.kind,
    "rows": args.expected_rows,
    "manifest_blake3": manifest["manifest_blake3"],
    "verifier_bundle_blake3": bundle["bundle_blake3"],
}, sort_keys=True))
""",
        encoding="utf-8",
    )
    verifier_data = verifier.read_bytes()
    bundle_core = {
        "schema": "eva.dataset-verifier-source-bundle.v1",
        "dataset_kind": prefix,
        "python_requires": ">=3.11",
        "system_requirements": [],
        "entrypoint": "scripts/verify_eva_hf_dataset_v1.py",
        "file_count": 1,
        "byte_count": len(verifier_data),
        "files": [
            {
                "path": "scripts/verify_eva_hf_dataset_v1.py",
                "byte_count": len(verifier_data),
                "content_blake3": blake3(verifier_data).hexdigest(),
            }
        ],
    }
    bundle = {
        **bundle_core,
        "bundle_blake3": blake3(
            (
                json.dumps(bundle_core, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
        ).hexdigest(),
    }
    bundle_path = root / "verifier" / "manifest.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")
    shards = []
    for path in sorted((root / "data").iterdir()):
        data = path.read_bytes()
        shards.append(
            {
                "path": f"data/{path.name}",
                "row_count": len(data.splitlines()),
                "byte_count": len(data),
                "content_blake3": blake3(data).hexdigest(),
            }
        )
    manifest = {
        "schema": "test.dataset.v1",
        "target_dataset": repo_id,
        "visibility": "private",
        "row_count": sum(item["row_count"] for item in shards),
        "shards": shards,
        "manifest_blake3": f"{prefix}-manifest",
        "verifier_bundle": {
            "schema": "eva.dataset-verifier-source-bundle.v1",
            "path": "verifier/manifest.json",
            "dataset_kind": prefix,
            "file_count": 1,
            "byte_count": len(verifier_data),
            "bundle_blake3": bundle["bundle_blake3"],
            "manifest_content_blake3": blake3(bundle_path.read_bytes()).hexdigest(),
            "entrypoint": "verifier/scripts/verify_eva_hf_dataset_v1.py",
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


class _FakeHub:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.repos: dict[str, dict[str, bytes]] = {}
        self.revisions: dict[str, str | None] = {}
        self.uploads = 0
        self.snapshots: set[Path] = set()

    def ensure_private_dataset(self, repo_id: str) -> Mapping[str, Any]:
        self.events.append(f"ensure:{repo_id}")
        self.repos.setdefault(repo_id, {})
        self.revisions.setdefault(repo_id, None)
        return {"private": True, "revision": self.revisions[repo_id]}

    def upload_dataset(self, *, repo_id: str, folder: Path, message: str) -> str:
        self.events.append(f"upload:{repo_id}")
        self.uploads += 1
        self.repos[repo_id] = {
            str(path.relative_to(folder)): path.read_bytes()
            for path in sorted(folder.rglob("*"))
            if path.is_file()
        }
        revision = f"revision-{self.uploads}"
        self.revisions[repo_id] = revision
        return revision

    def repo_state(self, repo_id: str) -> Mapping[str, Any]:
        return {"private": True, "revision": self.revisions[repo_id]}

    def snapshot_dataset(self, *, repo_id: str, revision: str) -> Path:
        if not self.repos.get(repo_id):
            raise publisher.RemoteFileMissing("absent")
        snapshot = Path(tempfile.mkdtemp(prefix="fake-hf-snapshot-"))
        for relative, data in self.repos[repo_id].items():
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (snapshot / ".gitattributes").write_text("*.jsonl filter=lfs\n", encoding="utf-8")
        self.snapshots.add(snapshot.resolve())
        return snapshot

    def release_snapshot(self, snapshot: Path) -> None:
        resolved = snapshot.resolve()
        assert resolved in self.snapshots
        self.snapshots.remove(resolved)
        shutil.rmtree(resolved)


publisher = _module()


def test_publish_is_verified_private_read_back_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "must-not-cross-offline-verifier-boundary")
    release = tmp_path / "release"
    _write_dataset(
        release / "EVA-Med-SFT-data",
        repo_id="operator/EVA-Med-SFT-data",
        prefix="sft",
    )
    _write_dataset(
        release / "EVA-Med-RL-data",
        repo_id="operator/EVA-Med-RL-data",
        prefix="rl",
    )
    events: list[str] = []

    def verify(root: Path, *, expected_rl: int) -> Mapping[str, Any]:
        events.append("verify")
        assert root == release and expected_rl == 6_000
        return {
            "valid": True,
            "release_blake3": "release-blake3",
            "rl_rows": 6_000,
            "upload_performed": False,
            "repo_created": False,
        }

    hub = _FakeHub(events)
    receipt_path = tmp_path / "publish-receipt.json"
    first = publisher.publish_release(
        release_root=release,
        receipt_output=receipt_path,
        execute=True,
        verifier=verify,
        hub=hub,
    )
    assert events[0] == "verify"
    assert hub.uploads == 2
    assert first["upload_operation_count"] == 2
    assert first["remote_readback_verified"] is True
    assert all(repo["local_verifier_file_count"] == 1 for repo in first["repositories"])
    assert all(
        repo["remote_snapshot_verification"]["inventory_file_count"]
        == repo["local_file_count"]
        for repo in first["repositories"]
    )
    assert all(
        repo["remote_snapshot_verification"]["offline_verifier_receipt"]["valid"]
        is True
        for repo in first["repositories"]
    )
    assert all(
        repo["remote_snapshot_verification"]["remote_extra_paths"]
        == [".gitattributes"]
        for repo in first["repositories"]
    )
    assert "token" not in receipt_path.read_text(encoding="utf-8").lower()
    assert publisher.verify_receipt(receipt_path)["receipt_blake3"] == first["receipt_blake3"]

    second = publisher.publish_release(
        release_root=release,
        receipt_output=receipt_path,
        execute=True,
        verifier=verify,
        hub=hub,
    )
    assert hub.uploads == 2
    assert second["upload_operation_count"] == 0
    assert {repo["remote_revision"] for repo in second["repositories"]} == {
        "revision-1",
        "revision-2",
    }
    assert hub.snapshots == set()


def test_local_verifier_failure_blocks_all_hub_calls(tmp_path: Path) -> None:
    events: list[str] = []
    hub = _FakeHub(events)

    def reject(root: Path, *, expected_rl: int) -> Mapping[str, Any]:
        raise publisher.PublishError("local verification failed")

    with pytest.raises(publisher.PublishError, match="local verification failed"):
        publisher.publish_release(
            release_root=tmp_path / "absent",
            receipt_output=tmp_path / "receipt.json",
            execute=True,
            verifier=reject,
            hub=hub,
        )
    assert events == []
    assert hub.uploads == 0


def test_matching_manifest_with_tampered_unselected_file_fails_closed(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    _write_dataset(
        release / "EVA-Med-SFT-data",
        repo_id="operator/EVA-Med-SFT-data",
        prefix="sft",
    )
    _write_dataset(
        release / "EVA-Med-RL-data",
        repo_id="operator/EVA-Med-RL-data",
        prefix="rl",
    )

    def verify(root: Path, *, expected_rl: int) -> Mapping[str, Any]:
        return {"valid": True, "release_blake3": "release-blake3"}

    hub = _FakeHub([])
    publisher.publish_release(
        release_root=release,
        receipt_output=tmp_path / "receipt.json",
        execute=True,
        verifier=verify,
        hub=hub,
    )
    hub.repos["operator/EVA-Med-SFT-data"]["MIGRATION.md"] = b"tampered\n"
    with pytest.raises(publisher.PublishError, match="pinned snapshot byte commitment"):
        publisher.publish_release(
            release_root=release,
            receipt_output=tmp_path / "receipt.json",
            execute=True,
            verifier=verify,
            hub=hub,
        )
    assert hub.uploads == 2


def test_plan_never_resolves_token_or_calls_hub(tmp_path: Path) -> None:
    release = tmp_path / "release"
    _write_dataset(
        release / "EVA-Med-SFT-data",
        repo_id="operator/EVA-Med-SFT-data",
        prefix="sft",
    )
    _write_dataset(
        release / "EVA-Med-RL-data",
        repo_id="operator/EVA-Med-RL-data",
        prefix="rl",
    )

    def verify(root: Path, *, expected_rl: int) -> Mapping[str, Any]:
        return {"valid": True, "release_blake3": "release-blake3"}

    def forbidden_token() -> str:
        raise AssertionError("token resolver crossed in plan mode")

    result = publisher.publish_release(
        release_root=release,
        receipt_output=tmp_path / "unused.json",
        execute=False,
        verifier=verify,
        token_resolver=forbidden_token,
    )
    assert result["remote_calls"] == 0
    assert result["upload_performed"] is False


def test_missing_verifier_bundle_blocks_publish_plan(tmp_path: Path) -> None:
    release = tmp_path / "release"
    _write_dataset(
        release / "EVA-Med-SFT-data",
        repo_id="operator/EVA-Med-SFT-data",
        prefix="sft",
    )
    _write_dataset(
        release / "EVA-Med-RL-data",
        repo_id="operator/EVA-Med-RL-data",
        prefix="rl",
    )
    manifest_path = release / "EVA-Med-SFT-data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("verifier_bundle")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    def verify(root: Path, *, expected_rl: int) -> Mapping[str, Any]:
        return {"valid": True, "release_blake3": "release-blake3"}

    with pytest.raises(publisher.PublishError, match="verifier_bundle descriptor differs"):
        publisher.publish_release(
            release_root=release,
            receipt_output=tmp_path / "unused.json",
            execute=False,
            verifier=verify,
        )
