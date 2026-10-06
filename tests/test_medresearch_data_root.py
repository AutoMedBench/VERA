from pathlib import Path

import pytest

from eva_agent.deployment.medresearch_v2 import CODE_ROOT, medical_data_root


def test_default_root_is_historical_code_root():
    assert medical_data_root({}) == CODE_ROOT


def test_separate_existing_data_root_does_not_relocate_runtime_code(tmp_path):
    assert medical_data_root({'EVA_MEDRESEARCH_DATA_ROOT': str(tmp_path)}) == tmp_path
    assert (CODE_ROOT / 'src/eva_agent/codex_pipeline/turn_mcp_proxy.py').is_file()
    assert not (tmp_path / 'src/eva_agent/codex_pipeline/turn_mcp_proxy.py').exists()


@pytest.mark.parametrize('value', ['', 'relative-path'])
def test_data_root_never_guesses_relative_paths(value):
    with pytest.raises(ValueError, match='explicit absolute'):
        medical_data_root({'EVA_MEDRESEARCH_DATA_ROOT': value})


def test_missing_data_root_fails_before_context_construction(tmp_path):
    with pytest.raises(FileNotFoundError):
        medical_data_root({'EVA_MEDRESEARCH_DATA_ROOT': str(tmp_path / 'absent')})
