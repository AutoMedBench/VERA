"""CPU-only code-selection checks; no legacy import, authorities, or providers."""

import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from eva_agent.sources.legacy_python_source import legacy_python_root


ROOT = Path(__file__).resolve().parents[1]


def package_tree(tmp_path):
    source = tmp_path / "separate-code" / "src"
    package = source / "rlevo_med_research"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# fixture; never imported\n")
    return source


def test_default_keeps_original_authority_source_without_early_loading(tmp_path):
    authority = tmp_path / "unchanged-authorities"
    assert legacy_python_root(authority, {}) == authority / "src"
    assert not authority.exists()


def test_explicit_code_tree_does_not_change_environment_or_import_path(tmp_path, monkeypatch):
    selected = package_tree(tmp_path)
    authority = tmp_path / "unchanged-authorities"
    monkeypatch.setenv("EVA_LEGACY_PYTHON_ROOT", str(selected))
    monkeypatch.setenv("PYTHONPATH", "/unchanged/transport/path")
    before_path, before_environment = list(sys.path), dict(os.environ)
    assert legacy_python_root(authority) == selected
    assert dict(os.environ) == before_environment and sys.path == before_path
    assert not authority.exists()


@pytest.mark.parametrize("value", ["", "relative/src", "~/source", True, 17, Path("/fixture")])
def test_non_absolute_or_non_string_explicit_selector_is_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="absolute source directory"):
        legacy_python_root(tmp_path / "authorities", {"EVA_LEGACY_PYTHON_ROOT": value})


def test_missing_explicit_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unavailable"):
        legacy_python_root(tmp_path, {"EVA_LEGACY_PYTHON_ROOT": str(tmp_path / "absent")})


@pytest.mark.parametrize("kind", ["file", "empty-directory", "package-without-init"])
def test_explicit_selector_requires_a_python_source_package(tmp_path, kind):
    selected = tmp_path / "selected"
    if kind == "file":
        selected.write_text("not a directory")
    else:
        selected.mkdir()
        if kind == "package-without-init":
            (selected / "rlevo_med_research").mkdir()
    with pytest.raises(ValueError, match="contain the rlevo_med_research package"):
        legacy_python_root(tmp_path / "authorities", {"EVA_LEGACY_PYTHON_ROOT": str(selected)})


@pytest.mark.parametrize("relative", [
    "src/eva_agent/deployment/medresearch_v2.py",
    "src/eva_agent/training/teacher_worker.py",
])
def test_resolver_consumers_select_only_code_and_keep_authority_arguments(tmp_path, monkeypatch, relative):
    """Execute each actual resolver call expression with a capture-only factory."""
    selected = package_tree(tmp_path)
    authority = tmp_path / "immutable-authorities"
    monkeypatch.setenv("EVA_LEGACY_PYTHON_ROOT", str(selected))
    source = ast.parse((ROOT / relative).read_text())
    calls = [node for node in ast.walk(source) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "LegacyExecutionBindingResolver"]
    assert len(calls) == 1
    assert any(isinstance(node, ast.ImportFrom)
        and node.module == "eva_agent.sources.legacy_python_source"
        and any(name.name == "legacy_python_root" for name in node.names) for node in ast.walk(source))
    trust, image = authority / "config/trust.json", authority / "config/image.json"
    scope = {"LegacyExecutionBindingResolver": lambda **kwargs: kwargs,
        "legacy_python_root": legacy_python_root, "LEGACY_ROOT": authority,
        "V24_ROOT": authority / "runs/supervisor", "IMAGE_REFS": image,
        "host_trust_store_path": trust, "trust_store": trust,
        "host_private_key_path": tmp_path / "host.pem", "private_key": tmp_path / "host.pem",
        "host_key_id": "fixture", "key_id": "fixture", "state_root": tmp_path / "runtime-state",
        "concurrency": SimpleNamespace(worker_width=4)}
    captured = eval(compile(ast.Expression(calls[0]), relative, "eval"), scope)
    assert captured["legacy_python_root"] == selected
    assert captured["authority_root"] == authority
    assert captured["supervisor_root"] == authority / "runs/supervisor"
    assert captured["trust_store_path"] == trust and captured["image_refs_path"] == image
    assert captured["runtime_state_root"] == tmp_path / "runtime-state"


def test_data_and_skill_constants_do_not_reference_code_selector():
    tree = ast.parse((ROOT / "src/eva_agent/deployment/medresearch_v2.py").read_text())
    assignments = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)}
    for name in ("PROJECT_ROOT", "LEGACY_ROOT", "V4_ROOT", "V24_ROOT", "MODEL_REGISTRY",
        "IMAGE_REFS", "RUBRIC_SOURCE", "SELECTION_V1", "SELECTION_V2", "PLUGIN_ROOT", "LEGACY_SKILL_ROOT"):
        assert not any(isinstance(node, ast.Name) and node.id == "legacy_python_root"
            for node in ast.walk(assignments[name]))
