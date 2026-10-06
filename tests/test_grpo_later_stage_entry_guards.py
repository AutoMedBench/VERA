"""Later-stage labels/files do not create a live executable predecessor state."""
from types import SimpleNamespace
from threading import RLock

import pytest

from eva_agent.pipeline import Stage
from eva_agent.sources.legacy_execution import _LegacyCandidateToolBackend, _WorkspaceSession
from eva_agent.training.grpo_stage_data import select_stage_rows
from eva_agent.training.teacher_focus import focus_only_teacher_instructions
from eva_agent.training.teacher_worker import hydrate_teacher_stage_prerequisites


@pytest.mark.parametrize("stage", ("S4", "S5"))
def test_later_stage_source_record_is_not_admitted_by_label_alone(stage):
    row = {"sandbox_id": "unchanged-original", "stage": stage, "split": "train",
           "domain": "automedbench-classification",
           "reward_contract": {"rubric_table": {"stage": stage, "domain": "automedbench-classification"}}}
    with pytest.raises(ValueError, match="executable S1-S3"):
        select_stage_rows([row], stage=stage, limit=1)


@pytest.mark.parametrize("stage", (Stage.S4, Stage.S5))
def test_focus_and_hydration_do_not_claim_unimplemented_later_entry(stage):
    context = SimpleNamespace(stage_tool_guidance=SimpleNamespace(focus=stage))
    with pytest.raises(ValueError, match="S1, S2, or S3"):
        focus_only_teacher_instructions(context)
    with pytest.raises(ValueError, match="hydration focus"):
        hydrate_teacher_stage_prerequisites(context=context, workspace=None, id_factory=None)


def fresh_session(tmp_path):
    return _WorkspaceSession(root=tmp_path, lock=RLock(), s1={}, s2_template={},
                             s3_template={}, s4_template={}, s5_template={})


def test_real_s4_handler_requires_live_successful_s3_not_workspace_claims(tmp_path):
    backend = object.__new__(_LegacyCandidateToolBackend)
    session = fresh_session(tmp_path)
    # Even an object offering apparent predecessor bytes is not a trusted
    # execution receipt/session. The canonical handler must return before
    # using Docker, a signer, workspace payloads or execution modules.
    workspace = SimpleNamespace(read_bytes=lambda _: b'{"gate_passed":true}')
    result = backend._handle_execute_code(workspace, session, {"stage": "S4", "code": "pass"})
    assert result == {"stage": "S4", "gate_passed": False,
                      "error": "verified_s3_prerequisite_required"}
    assert session.s3_success is None and session.s4_success is None and session.s4_attempts == 0


def test_real_s5_handler_requires_live_s4_and_preserves_one_shot_semantics(tmp_path):
    backend = object.__new__(_LegacyCandidateToolBackend)
    session = fresh_session(tmp_path)
    workspace = SimpleNamespace(read_bytes=lambda _: b'{"gate_passed":true}')
    result = backend._handle_submit_results(workspace, session, {"terminal": {}})
    assert result == {"stage": "S5", "gate_passed": False,
                      "error": "verified_s4_prerequisite_required"}
    assert session.submission_attempted is True
    assert backend._handle_submit_results(workspace, session, {"terminal": {}}) == {
        "stage": "S5", "gate_passed": False, "error": "single_submission_consumed"}
