"""Small provider/GPU-free audit fixtures; no model data or tokens from real runs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import pytest

from eva_agent.pipeline.contracts import ToolResult
from eva_agent.pipeline.digests import blake3_hex, canonical_value
from eva_agent.training import slime_agent_judge
from eva_agent.training.slime_rollout import tool_observation_tail
from test_agent_judge_worker import _ProviderFreeWorkspaceJudge
from test_slime_agent_judge import _generic_rollout

SPEC = importlib.util.spec_from_file_location("offline_grpo_audit", Path(__file__).resolve().parents[1]
                                            / "scripts/verify_slime_grpo_group_v1.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class Tokenizer:
    def __len__(self):
        return 1024

    def convert_tokens_to_ids(self, value):
        return 999

    def encode(self, value, **kwargs):
        return list(value.encode())


@pytest.fixture
def token_fixture():
    core = dict(result_id=str(uuid4()), call_id=str(uuid4()), name="materialize_plan", frontier=0,
                parallel_group_id=str(uuid4()), status="completed", output={"gate_passed": False},
                error_code=None, workspace_before_blake3="a" * 64, workspace_after_blake3="a" * 64)
    observation = canonical_value(ToolResult(**core, receipt_blake3=blake3_hex(core)))
    env = Tokenizer().encode(tool_observation_tail([observation], has_end_token=True))
    tokens = dict(schema="eva.private-sampled-policy-tokens.v1", tokens=[10, 701, 999, *env, 702],
                  response_length=len(env) + 3, loss_mask=[1, 1, *([0] * len(env)), 1],
                  rollout_log_probs=[-0.3, -0.1, *([0.0] * len(env)), -0.2],
                  sampled_token_count=3, reward_emitted=False, judge_status="not_yet_completed",
                  sft_export_eligible=False)
    rollout = {"messages": [{"role": "assistant"}, {"role": "tool", "content": observation},
                            {"role": "assistant"}]}
    # Persistence sorts outer keys; reconstruction must recover ToolResult order.
    return json.loads(json.dumps(rollout, sort_keys=True)), tokens


def test_exact_tokens_masks_and_dataclass_order(token_fixture):
    rollout, tokens = token_fixture
    report = audit.verify_tokens(tokens, vocabulary_size=len(Tokenizer()))
    assert report["sampled_tokens"] == 3
    assert report["masked_tokens"] > 0
    assert audit.verify_masked_observations(rollout, tokens, Tokenizer()) == {
        "exact_masked_observation_blocks": 1, "terminal_unconditioned_tool_frontiers": 0}


@pytest.mark.parametrize("mutation,code", [
    (lambda x: x.update(response_length=0), "response_length"),
    (lambda x: x["rollout_log_probs"].pop(), "token_alignment"),
    (lambda x: x["tokens"].__setitem__(0, 1024), "token_vocabulary"),
    (lambda x: x["loss_mask"].__setitem__(0, 2), "loss_mask"),
    (lambda x: x["rollout_log_probs"].__setitem__(0, float("nan")), "logprob_domain"),
    (lambda x: x["rollout_log_probs"].__setitem__(2, -0.5), "masked_logprob_nonzero"),
    (lambda x: x.update(sampled_token_count=4), "sampled_token_count"),
    (lambda x: x.update(reward_emitted=True), "prejudge_receipt_boundary"),
])
def test_token_corruption_fails(token_fixture, mutation, code):
    _, tokens = token_fixture
    mutation(tokens)
    with pytest.raises(audit.AuditError, match=code):
        audit.verify_tokens(tokens, vocabulary_size=1024)


def test_masked_tail_corruption_and_disappearing_mask_fail(token_fixture):
    rollout, tokens = token_fixture
    bad = deepcopy(tokens)
    bad["tokens"][3] += 1
    with pytest.raises(audit.AuditError, match="masked_observation_tokens_differ"):
        audit.verify_masked_observations(rollout, bad, Tokenizer())
    bad["loss_mask"] = [1] * len(bad["loss_mask"])
    with pytest.raises(audit.AuditError, match="missing_nonterminal_observation_mask"):
        audit.verify_masked_observations(rollout, bad, Tokenizer())


def test_private_permissions_and_symlinks_fail_closed(tmp_path):
    path = tmp_path / "tokens.json"
    path.write_text("{}")
    path.chmod(0o600)
    assert audit.read_json(path, private=True) == {}
    path.chmod(0o644)
    with pytest.raises(audit.AuditError, match="token_file_not_0600"):
        audit.read_json(path, private=True)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(audit.AuditError, match="artifact_symlink"):
        audit.read_json(link, private=True)


@pytest.fixture
def judged_fixture(tmp_path):
    rollout, rubric, source, model = _generic_rollout(tmp_path)
    prepared = slime_agent_judge.prepare_workspace_rollout(
        rollout, rubric=rubric, source_path=source, judge_model_id=model)
    grade = slime_agent_judge.grade_prepared_rollout(
        prepared, judge=_ProviderFreeWorkspaceJudge(), output_root=tmp_path)
    (tmp_path / "grade.json").write_text(json.dumps(grade))
    return prepared, tmp_path


def test_real_workspace_fixture_reopens_and_shared_score_recomputes(judged_fixture):
    prepared, root = judged_fixture
    bound, assessment, reads = audit.reopen_assessment(prepared, root / "assessment.json")
    assert reads == {"original_before": 1, "original_after": 2, "sidecar": 5}
    assert audit.verify_scoring(bound, assessment, root)["reward"] == 1.0
    grade = json.loads((root / "grade.json").read_text())
    grade["reward"] = 0.25
    (root / "grade.json").write_text(json.dumps(grade))
    with pytest.raises(audit.AuditError, match="grade_differs"):
        audit.verify_scoring(bound, assessment, root)


def test_self_consistently_rehashed_forged_read_rejected(judged_fixture):
    prepared, root = judged_fixture
    path = root / "assessment.json"
    document = json.loads(path.read_text())
    trace = document["agent_trace"]
    result = trace["results"][0]
    result["output"]["content"] = "x" * result["output"]["returned_bytes"]
    result["receipt_blake3"] = blake3_hex({k: v for k, v in result.items() if k != "receipt_blake3"})
    trace["trace_blake3"] = blake3_hex({k: v for k, v in trace.items() if k != "trace_blake3"})
    document["assessment_blake3"] = blake3_hex({k: v for k, v in document.items() if k != "assessment_blake3"})
    path.write_text(json.dumps(document))
    with pytest.raises(audit.AuditError, match="workspace_read_bytes_differ"):
        audit.reopen_assessment(prepared, path)


def test_missing_original_source_citation_still_rejected(judged_fixture):
    from dataclasses import replace
    prepared, root = judged_fixture
    bound, assessment, _ = audit.reopen_assessment(prepared, root / "assessment.json")
    sidecar = next(ref for ref in assessment.agent_trace.inspected_evidence_refs if ".eva-agent-judge/" in ref)
    bad = replace(assessment, item_scores=tuple(replace(item, evidence_refs=(sidecar,))
                                              for item in assessment.item_scores))
    with pytest.raises(slime_agent_judge.AgentJudgeSelectionError, match="cite inspected workspace evidence"):
        audit.verify_scoring(bound, bad, root)


def test_cli_rejects_nonlocal_model_without_loading_transformers(tmp_path, capsys):
    assert audit.main(["--group-root", str(tmp_path), "--tokenizer-model", "not-a-local-model"]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "local_tokenizer_directory_required"


def test_source_drift_does_not_invalidate_retained_group_evidence(tmp_path, monkeypatch):
    from eva_agent.pipeline.digests import blake3_bytes
    checkout = tmp_path / "checkout"
    names = ("training/slime/run_full_parameter.py", "training/slime/runtime_hooks.py",
             "training/slime/checkpoint_preflight.py", "src/eva_agent/training/slime_data.py",
             "src/eva_agent/training/slime_rollout.py", "src/eva_agent/training/slime_agent_judge.py")
    for name in names:
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# launch-time fixture\n")
    group = tmp_path / "run/rollouts/000000"
    sample_id = str(uuid4())
    (group / sample_id).mkdir(parents=True)
    (group.parent.parent / "run-receipt.json").write_text(json.dumps({"eva_source_blake3": {
        name: blake3_bytes((checkout / name).read_bytes()) for name in names}}))
    (group / "summary.json").write_text(json.dumps({"samples": 1, "rewards": [0.0],
        "reward_spread": 0.0, "zero_variance_group": True, "rubric_agent_judged": True}))
    # Isolate only mutable-source comparison: artifact verification has its own
    # real workspace/rubric and corruption tests above, and must remain untouched.
    monkeypatch.setattr(audit, "ROOT", checkout)
    monkeypatch.setattr(audit, "verify_sample", lambda *_: {
        "sample_uuid": sample_id, "judge_status": "accepted", "reward": 0.0})
    before = audit.verify_group(group, Tokenizer())
    assert before["valid"] and before["current_source_matches_launch"]
    (checkout / names[0]).write_bytes(b"# prospective launcher edit\n")
    after = audit.verify_group(group, Tokenizer())
    assert after["valid"] and after["complete"] and after["group_summary_verified"]
    assert after["errors"] == [] and not after["current_source_matches_launch"]
    assert after["accepted_rewards"] == 1
    assert "exact historical runtime has not been reconstructed" in after["limitations"][-1]
