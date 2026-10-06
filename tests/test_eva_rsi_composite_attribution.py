"""Provider-free attribution routing fixtures; no real grade or Opus claim."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eva_agent.pipeline.digests import blake3_hex
from training.eva_rsi import composite_eval, skill_attribution, skill_identity
from training.eva_rsi.evidence import commitment
from training.benchmark_feedback import automed_codex


def fixture(tmp_path, monkeypatch):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return commitment(path)

    model = tmp_path / 'same-checkpoint'
    model.mkdir()
    sources, skills, feedback, rollouts = {}, {}, {}, {}
    for name in ('original', 'supplement'):
        root = tmp_path / name
        identity = write(root / 'identity.json', {'exact_final_model_path': str(model), 'receipt_id': name})
        sidecar = write(root / 'skills.json', {'source': name})
        sources[name] = {'index': write(root / 'index.json', {'source': name}),
            'benchmark_run_root': str(root / 'run'), 'checkpoint_identity': identity,
            'skill_content_binding': sidecar}
        skills[sidecar['path']] = {'mounted_catalog_blake3': name + '-mount',
            'verified_content': {'content': {'legacy_skill_metadata': [{'skill_id': 'same-skill'}],
                'mount_path_independent_catalog': {'native_skill': {'skill_id': 'stage-rollout'}}}}}
    rows = []
    for name, track, stage, reward in [('original', 'report', 'S2', 5000),
                                     ('original', 'vqa', 'S3', 0),
                                     ('supplement', 'classification', 'S1', 0)]:
        root = tmp_path / name / track / stage
        rollout = {'source_workspace': name + '/' + track, 'stage': stage}
        write(root / 'rollout.json', rollout)
        write(root / 'preflight.json', {'source': name})
        rollouts[str(root / 'rollout.json')] = rollout
        raw = {'checkpoint_id': sources[name]['checkpoint_identity']['blake3'],
               'skill_catalog_id': name + '-mount'}
        verdict = {'valid': True, 'status': 'scored', 'stage': stage, 'domain': 'automedbench-' + track,
            'judge_codex_receipt_blake3': name + track, 'round_identity': raw,
            'score': {'reward_bps': reward}, 'source_rollout_blake3': blake3_hex(rollout)}
        feedback[str(root)] = verdict
        rows.append({'root': str(root), 'source_name': name, 'track': track, 'stage': stage,
            'judge_receipt_blake3': verdict['judge_codex_receipt_blake3'],
            'raw_round_identity': deepcopy(raw), 'score_document': deepcopy(verdict['score'])})
    index_path = tmp_path / 'composite.json'
    index_ref = write(index_path, {'schema': composite_eval.SCHEMA})
    evaluation = {'index': index_ref, 'checkpoint_identity': sources['original']['checkpoint_identity'],
        'feedback': [{'root': row['root'], 'judge_receipt_blake3': row['judge_receipt_blake3']}
                     for row in reversed(rows)]}
    evaluation_path = tmp_path / 'evaluation.json'
    write(evaluation_path, evaluation)
    context = {'previous_evaluation': str(evaluation_path), 'skill_catalog_id': 'same-content',
               'model_path': str(model)}
    proof = {'valid': True, 'sources': sources, 'feedback': rows,
        'comparison_checkpoint_identity': sources['original']['checkpoint_identity'],
        'checkpoint_equivalence_blake3': 'same-headers-and-lineage', 'skill_content_id': 'same-content'}
    calls = {'proof': [], 'skills': [], 'rollouts': []}
    def verify_index(path):
        calls['proof'].append(path)
        assert path == index_path
        return deepcopy(proof)
    def verify_skills(path, *, expected_run_root, expected_content_id):
        calls['skills'].append((path, expected_run_root, expected_content_id))
        assert expected_content_id == 'same-content'
        assert expected_run_root == path.parent / 'run'
        return deepcopy(skills[str(path)])
    def read_rollout(path, preflight):
        calls['rollouts'].append(path)
        assert preflight['source'] in path.parts
        return deepcopy(rollouts[str(path)])
    monkeypatch.setattr(composite_eval, 'verify_composite_index', verify_index)
    monkeypatch.setattr(skill_identity, 'verify_skill_content_binding', verify_skills)
    monkeypatch.setattr(automed_codex, 'verify_feedback', lambda root, **_: deepcopy(feedback[str(root)]))
    monkeypatch.setattr(automed_codex, 'read_feedback_rollout', read_rollout)
    monkeypatch.setattr(automed_codex, '_bound_feedback_rubric', lambda *args: SimpleNamespace(digest='same-rubric'))
    return context, proof, evaluation, feedback, sources, calls, write


def test_composite_weakest_reads_selected_actual_source_and_keeps_raw_ids(tmp_path, monkeypatch):
    context, proof, _, _, sources, calls, _ = fixture(tmp_path, monkeypatch)
    before = deepcopy(proof)
    rollout, rubric, selected = skill_attribution.prepare_source(context)
    assert rollout['source_workspace'] == 'supplement/classification' and rubric.digest == 'same-rubric'
    assert selected['selected_stage'] == 'S1' and selected['verified_candidate_count'] == 3
    source = selected['composite_source']
    assert source['source_name'] == 'supplement'
    assert source['raw_checkpoint_identity'] == sources['supplement']['checkpoint_identity']
    assert source['raw_checkpoint_identity'] != proof['comparison_checkpoint_identity']
    assert source['raw_round_identity']['skill_catalog_id'] == 'supplement-mount'
    assert source['comparison_content_identity'] == 'same-content'
    assert source['canonical_source_evidence_normalized'] is False
    assert calls['rollouts'] == [Path(selected['source_rollout']['path'])]
    assert len(calls['skills']) == 2 and len(calls['proof']) == 1
    assert proof == before and not selected['all_round_trajectories_inspected_by_opus']


@pytest.mark.parametrize('field', ['checkpoint_id', 'skill_catalog_id'])
def test_composite_attribution_rejects_normalized_raw_source_identity(tmp_path, monkeypatch, field):
    context, proof, _, feedback, _, _, _ = fixture(tmp_path, monkeypatch)
    row = proof['feedback'][-1]
    wrong = proof['comparison_checkpoint_identity']['blake3'] if field == 'checkpoint_id' else 'original-mount'
    row['raw_round_identity'][field] = wrong
    feedback[row['root']]['round_identity'][field] = wrong
    with pytest.raises(ValueError, match='attribution_feedback_identity'):
        skill_attribution.prepare_source(context)


def test_composite_attribution_requires_entire_verified_union(tmp_path, monkeypatch):
    context, _, evaluation, _, _, _, write = fixture(tmp_path, monkeypatch)
    evaluation['feedback'].pop()
    write(Path(context['previous_evaluation']), evaluation)
    with pytest.raises(ValueError, match='attribution_composite_feedback_coverage'):
        skill_attribution.prepare_source(context)


def test_single_source_v2_keeps_historical_selection_shape(tmp_path, monkeypatch):
    context, _, evaluation, _, sources, calls, write = fixture(tmp_path, monkeypatch)
    source = sources['original']
    index_ref = write(Path(source['index']['path']), {'schema': 'eva.rsi-evaluation-index.v2',
        'skill_content_identity': source['skill_content_binding'],
        'benchmark_run_root': source['benchmark_run_root'],
        'checkpoint_identity': source['checkpoint_identity']['path']})
    evaluation.update(index=index_ref, feedback=[row for row in evaluation['feedback']
                                               if '/original/' in row['root']])
    write(Path(context['previous_evaluation']), evaluation)
    rollout, _, selected = skill_attribution.prepare_source(context)
    assert rollout['source_workspace'] == 'original/vqa'
    assert selected['selected_stage'] == 'S3' and selected['verified_candidate_count'] == 2
    assert 'composite_source' not in selected and calls['proof'] == []
