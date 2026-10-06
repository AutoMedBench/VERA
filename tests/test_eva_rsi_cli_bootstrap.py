"""Real CLI-help subprocesses; no controller state, provider, or GPU execution."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ('run_eva_rsi_loop_v1.py', 'configure_eva_rsi_v1.py')
PROBE = '''
import json, runpy, sys
def no_execution(event, arguments):
    if event in {"subprocess.Popen", "socket.connect"}:
        raise AssertionError("CLI help must not launch commands or contact providers")
sys.addaudithook(no_execution)
script = sys.argv[1]
sys.argv = [script, "--help"]
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("CLI help must exit before controller creation")
import eva_agent
from training.eva_rsi import controller
print(json.dumps({"eva_agent": eva_agent.__file__, "controller": controller.__file__,
                  "torch_imported": "torch" in sys.modules}))
'''


def environment(harness=None):
    # Deliberately seed the old wrong priority; the entrypoint must correct it.
    result = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
              'PYTHONPATH': str(ROOT / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'}
    if harness is not None:
        result['EVA_HARNESS_ROOT'] = str(harness)
    return result


@pytest.mark.parametrize('script', SCRIPTS)
@pytest.mark.parametrize('selected', (False, True))
def test_cli_help_uses_selected_harness_or_preserves_no_env_default(tmp_path, script, selected):
    harness = None
    if selected:
        harness = tmp_path / 'selected-harness'
        package = harness / 'src/eva_agent'
        shutil.copytree(ROOT / 'src/eva_agent', package, ignore=shutil.ignore_patterns('__pycache__'))
        # Source-selection fixture only, not a claim of these runtime features.
        for name in ('policy_budget.py', 'research_memory.py'):
            (package / 'codex_runtime' / name).write_text('# Provider-free selection fixture.\n')
    cwd = tmp_path / 'empty-working-directory';cwd.mkdir()
    result = subprocess.run([sys.executable, '-c', PROBE, str(ROOT / 'scripts' / script)],
        cwd=cwd, env=environment(harness), capture_output=True, text=True, timeout=30, check=True)
    observed = json.loads(result.stdout.splitlines()[-1])
    expected = (harness if selected else ROOT) / 'src/eva_agent/__init__.py'
    assert Path(observed['eva_agent']).resolve() == expected.resolve()
    assert Path(observed['controller']).resolve() == ROOT / 'training/eva_rsi/controller.py'
    assert observed['torch_imported'] is False
    assert list(cwd.iterdir()) == []


@pytest.mark.parametrize('script', SCRIPTS)
def test_missing_selected_harness_fails_without_silent_default(tmp_path, script):
    harness = tmp_path / 'incomplete';harness.mkdir()
    cwd = tmp_path / 'empty';cwd.mkdir()
    result = subprocess.run([sys.executable, '-c', PROBE, str(ROOT / 'scripts' / script)],
        cwd=cwd, env=environment(harness), capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'lacks required v1.2 runtime modules' in result.stderr
    assert list(cwd.iterdir()) == []
