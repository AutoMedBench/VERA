import pytest

from eva_agent.pipeline.contracts import Stage
from eva_agent.training.stage_execution_guidance import (
    GUIDANCE_VERSION,
    stage_execution_guidance,
)


@pytest.mark.parametrize("stage", [Stage.S3, Stage.S4, Stage.S5])
def test_shared_public_guidance_is_deterministic_and_budget_aware(stage):
    text = stage_execution_guidance(stage)
    assert text == stage_execution_guidance(stage.value)
    assert GUIDANCE_VERSION in text
    assert "BEFORE the long code argument" in text
    assert "each execute_code call compact, syntactically complete, and fully closed" in text
    assert "parallel-safe reads, searches, and load_skill calls may run concurrently" in text
    assert "dependent calls, writes, and execution steps ordered" in text
    assert "Read the actual declared input files" in text
    assert "compute hashes from actual file bytes" in text
    assert "never hash placeholder text" in text
    assert "do not guarantee task success" in text
    assert len(text) < 2500


@pytest.mark.parametrize("stage", [Stage.S1, Stage.S2, Stage.E2E])
def test_unrelated_stage_context_is_unchanged(stage):
    assert stage_execution_guidance(stage) == ""


def test_advice_preserves_actual_execution_and_other_tool_contracts():
    text = stage_execution_guidance("S3")
    assert "stage='S3' argument BEFORE" in text
    assert "never add a stage argument to another tool" in text
    assert "Do not assume files persist" in text
    assert "self-contained program producing all of that call's declared outputs" in text
    assert "only where the offered tools permit them" in text
    assert "do not change tool schemas, stage permissions, output contracts, or scoring" in text
    assert "one compact, syntactically complete, fully closed tool call at a time" not in text


def test_s5_does_not_instruct_unsupported_s5_code_execution():
    text = stage_execution_guidance("S5")
    assert "use only the currently permitted review or terminal actions" in text
    assert "do not introduce a new execution stage" in text
    assert "stage='S5'" not in text


def test_unknown_stage_is_not_silently_reinterpreted():
    with pytest.raises(ValueError):
        stage_execution_guidance("S6")
