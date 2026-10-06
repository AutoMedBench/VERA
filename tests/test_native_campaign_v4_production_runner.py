from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_native_campaign_v4_production.py"


def _module():
    spec = importlib.util.spec_from_file_location("native_v4_production_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Deployment:
    def __init__(self, recipe: str) -> None:
        self.recipe = SimpleNamespace(
            recipe_blake3=recipe,
            concurrency={"worker_width": 1},
        )
        self.installs = 0
        self.runs = 0

    def install_frozen_schedule(self, *, worker_width: int) -> int:
        assert worker_width == 1
        self.installs += 1
        return 0

    def run_until_idle(self):
        self.runs += 1
        return SimpleNamespace(to_document=lambda: {"claimed": 1})

    def request_stop(self, _reason: str) -> None:
        pass


def test_run_reverifies_immediately_before_provider_boundary(monkeypatch, tmp_path):
    runner = _module()
    recipe = "a" * 64
    plan = {
        "document_blake3": "b" * 64,
        "launch_blake3": "c" * 64,
        "launch_path": str(tmp_path / "launch.json"),
        "preflight_path": str(tmp_path / "preflight.json"),
        "preflight_document_blake3": "d" * 64,
        "runner_source_blake3": runner.blake3_bytes(SCRIPT.read_bytes()),
        "profile": "canary",
        "max_candidates": 1,
        "recipe_blake3": recipe,
    }
    deployment = _Deployment(recipe)
    verifies = []
    monkeypatch.setattr(runner, "verify", lambda path: verifies.append(path) or plan)
    monkeypatch.setattr(runner, "_compose", lambda *_args: (object(), deployment))
    monkeypatch.setattr(
        runner,
        "_reopen_preflight",
        lambda *_args: (
            SimpleNamespace(launch_blake3="c" * 64),
            {"document_blake3": "d" * 64},
        ),
    )
    receipt = tmp_path / "receipt.json"

    result = runner.run(tmp_path / "plan.json", "canary", recipe, receipt)

    assert len(verifies) == 1
    assert deployment.installs == deployment.runs == 1
    assert result["max_candidates"] == 1
    assert receipt.exists() and receipt.stat().st_mode & 0o777 == 0o600


def test_run_rejects_uncommitted_recipe_before_composition(monkeypatch, tmp_path):
    runner = _module()
    plan = {
        "document_blake3": "b" * 64,
        "launch_path": str(tmp_path / "launch.json"),
        "profile": "canary",
        "max_candidates": 1,
        "recipe_blake3": "a" * 64,
    }
    monkeypatch.setattr(runner, "verify", lambda _path: plan)
    monkeypatch.setattr(
        runner, "_compose", lambda *_args: pytest.fail("composition must not occur")
    )
    with pytest.raises(RuntimeError, match="not authorized"):
        runner.run(tmp_path / "plan.json", "canary", "c" * 64)


def test_production_requires_explicit_bound(monkeypatch, tmp_path):
    runner = _module()
    monkeypatch.setattr(
        runner.launch_v4,
        "_reopen_launch",
        lambda _path: (
            SimpleNamespace(),
            {"host_private_key_path": "x", "host_key_id": "k", "host_trust_store_path": "y", "host_public_key_blake3": "a", "host_trust_store_blake3": "b"},
            object(),
        ),
    )
    monkeypatch.setattr(runner.v2, "load_verified_provider_health", lambda _p: {})
    with pytest.raises(RuntimeError, match="explicit candidate bound"):
        runner._compose(SCRIPT, "128", None)
