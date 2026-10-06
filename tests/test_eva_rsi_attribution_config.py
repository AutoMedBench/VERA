"""Command forwarding only; no controller or provider is initialized."""
import pytest

from training.eva_rsi.evidence import EvidenceError
from training.eva_rsi.production import loop_config


def settings():
    return {"control_python": "/selected/python", "architecture_model_path": "/original",
            "initial_model_path": "/trained", "initial_checkpoint_root": "/checkpoint",
            "skill_catalog_id": "verified-current-content"}


def test_attribution_forwarded_exactly_without_aliasing_or_extra_phases(tmp_path):
    value = settings()
    value["skill_attribution"] = ["/selected/python", "/med/attribution.py", "--context", "{context}", "--execute"]
    actual = loop_config(value, tmp_path / "settings.json")
    assert actual["rounds"] == 10 and actual["updates_per_round"] == 50
    assert actual["commands"]["skill_attribution"] == value["skill_attribution"]
    assert set(actual["commands"]) == {"baseline_eval", "train", "evaluation", "skill_attribution"}
    actual["commands"]["skill_attribution"].append("not-an-input-mutation")
    assert value["skill_attribution"][-1] == "--execute"


def test_attribution_absent_preserves_default_commands(tmp_path):
    assert set(loop_config(settings(), tmp_path / "settings.json")["commands"]) == {
        "baseline_eval", "train", "evaluation"}


@pytest.mark.parametrize("invalid", [[], "shell command", {"backend": "opus_5"}, ["python", 3], [""]])
def test_invalid_attribution_command_rejected_before_controller(tmp_path, invalid):
    with pytest.raises(EvidenceError, match="explicit_command_argv"):
        loop_config({**settings(), "skill_attribution": invalid}, tmp_path / "settings.json")
