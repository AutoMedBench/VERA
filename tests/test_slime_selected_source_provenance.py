"""Receipt path/hash selection only; no torch, Slime, GPU or provider execution."""
import importlib.util
from pathlib import Path

from blake3 import blake3
import eva_agent

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('selected_source_launcher_fixture', ROOT / 'training/slime/run_full_parameter.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_selected_harness_package_and_med_file_keep_exact_logical_keys(tmp_path, monkeypatch):
    repo, harness = tmp_path / 'MED', tmp_path / 'harness'
    logical = repo / 'src/eva_agent/training/codex_slime_rollout.py'
    actual = harness / 'src/eva_agent/training/codex_slime_rollout.py'
    local = repo / 'training/slime/runtime_hooks.py'
    for path, text in ((logical, 'old-unselected-source'), (actual, 'actual-selected-source'), (local, 'actual-med-hook')):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    init = harness / 'src/eva_agent/__init__.py'
    init.write_text('')
    monkeypatch.setattr(eva_agent, '__file__', str(init))
    hashes, paths = launcher.capture_eva_source_provenance([logical, local], repo=repo)
    assert set(hashes) == {'src/eva_agent/training/codex_slime_rollout.py', 'training/slime/runtime_hooks.py'}
    assert paths == {'src/eva_agent/training/codex_slime_rollout.py': str(actual),
                     'training/slime/runtime_hooks.py': str(local)}
    assert hashes['src/eva_agent/training/codex_slime_rollout.py'] == blake3(actual.read_bytes()).hexdigest()
    assert hashes['src/eva_agent/training/codex_slime_rollout.py'] != blake3(logical.read_bytes()).hexdigest()
    assert hashes['training/slime/runtime_hooks.py'] == blake3(local.read_bytes()).hexdigest()


def test_actual_imported_package_path_is_used_for_reward_normalizer():
    logical = ROOT / 'src/eva_agent/training/codex_segment_rewards.py'
    hashes, paths = launcher.capture_eva_source_provenance([logical])
    actual = Path(eva_agent.__file__).resolve().parent / 'training/codex_segment_rewards.py'
    key = 'src/eva_agent/training/codex_segment_rewards.py'
    assert paths[key] == str(actual)
    assert hashes[key] == blake3(actual.read_bytes()).hexdigest()
