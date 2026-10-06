import os
from pathlib import Path
import subprocess
import sys

import pytest

from training.harness_source import runtime_paths


def test_default_and_explicit_checkout_selection(tmp_path):
    med = tmp_path / "med"
    med.mkdir()
    assert runtime_paths(med, {}) == [str(med), str(med / "src")]
    harness = tmp_path / "harness"
    runtime = harness / "src/eva_agent/codex_runtime"
    runtime.mkdir(parents=True)
    for name in ("runtime.py", "policy_budget.py", "research_memory.py"):
        (runtime / name).write_text("# fixture only")
    (runtime.parent / "__init__.py").write_text("MARKER = 'selected-harness'\n")
    selected = {"EVA_HARNESS_ROOT": str(harness)}
    assert runtime_paths(med, selected) == [str(med), str(harness / "src"), str(med / "src")]
    root = Path(__file__).resolve().parents[1]
    code = "from training.harness_source import activate_harness; import sys; activate_harness(sys.argv[1]); import eva_agent; print(eva_agent.MARKER)"
    result = subprocess.run([sys.executable, "-c", code, str(med)], cwd=root,
        env={**os.environ, **selected, "PYTHONPATH": str(root)}, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "selected-harness"


def test_incomplete_selected_checkout_cannot_silently_fallback(tmp_path):
    with pytest.raises(ValueError, match="lacks required"):
        runtime_paths(tmp_path, {"EVA_HARNESS_ROOT": str(tmp_path)})
