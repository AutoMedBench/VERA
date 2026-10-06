"""Synthetic, provider-free disjoint report assembly; no benchmark result claim."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.automedbench_lite.adapter import write_once
from training.automedbench_lite.evaluation_report import build_report as real_build_report
from training.automedbench_lite.track_adapter import BY_TRACK
from training.eva_rsi import composite_report as report
from training.eva_rsi.evidence import commitment


def fixture(tmp_path, monkeypatch):
    def put(path, value, committed=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        if committed:
            return write_once(path, value)
        path.write_text(json.dumps(value))
        return value
    model, checkpoint, architecture = (tmp_path / name for name in ('hf', 'dcp', 'architecture'))
    for path in (model, checkpoint, architecture):
        path.mkdir()
    indexes, documents, reports = {}, {}, {}
    for name in ('original', 'supplement'):
        root = tmp_path / name
        run = root / 'run'
        put(run / 'track-rollouts/attempt.json', {'tracks': list(BY_TRACK) if name == 'original'
            else list(report.SUPPLEMENTED), 'verified_skill_catalog_blake3': name + '-mount'}, committed=True)
        identity = root / 'identity.json'
        put(identity, {'schema': 'eva.qwen-final-serving-checkpoint-identity.v1', 'status': 'launch_prepared',
            'receipt_id': name, 'architecture_model_path': str(architecture), 'checkpoint_root': str(checkpoint),
            'exact_final_model_path': str(model), 'final_checkpoint_iteration': 10,
            'checkpoint_preflight': {'fixture': 'same-checkpoint'}, 'hf_asset_commitments': {'headers': 'same'}})
        path = root / 'index.json'
        value = {'schema': 'eva.rsi-evaluation-index.v1', 'benchmark_run_root': str(run),
            'checkpoint_identity': str(identity), 'feedback_roots': [], 'matched_codex_version_requested': '0.153.4'}
        put(path, value)
        indexes[name], documents[name] = path, value
        rows = []
        for track in BY_TRACK:
            A = 90 if name == 'original' else 10
            stages = {stage: {'A': A if stage == 'S1' else None, 'T': None}
                      for stage in ('S1', 'S2', 'S3', 'S4', 'S5')}
            rows.append({'track': track, 'A': A, 'T': 0 if name == 'supplement' else None,
                'A_plus_T': A if name == 'supplement' else None,
                'mean_A_T': A/2 if name == 'supplement' else None,
                'A_stage_coverage': 1, 'A_provisional': True, 'stages': stages,
                'task': {'task_judge_audit_status': 'not_performed'},
                'runtime': {'turns': 2 if name == 'supplement' else 3,
                    'reported_time_seconds': 3600, 'reported_time_source': 'admitted_track_elapsed',
                    'input_tokens': 100, 'output_tokens': 10}})
        reports[str(path)] = {'schema': 'eva.automedbench-seven-track-report.v1',
            'evaluation_index': commitment(path), 'checkpoint_identity': commitment(identity),
            'harness': {'codex_version_requested': '0.153.4', 'selected_harness_root': '/fixture/' + name},
            'tracks': rows, 'definitions': {'A': 'fixture process rubric', 'T': 'fixture native task'}}
    for track in report.SUPPLEMENTED:
        put(Path(documents['original']['benchmark_run_root']) / 'track-rollouts' / track / 'rollout.json',
            {'actual_turn_count': 0, 'turn_receipt_blake3s': [], 'completed_requested_turns': False,
             'errors': [{'phase': 'initialization', 'error': 'fixture'}]}, committed=True)
    calls = []
    def build(run, index):
        calls.append((run, index))
        assert run == Path(documents['original' if index == indexes['original'] else 'supplement']['benchmark_run_root'])
        return deepcopy(reports[str(index)])
    monkeypatch.setattr(report, 'build_report', build)
    return SimpleNamespace(indexes=indexes, documents=documents, reports=reports, calls=calls, put=put)


def test_fixed_cd_selection_keeps_missing_scores_and_each_raw_source(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    before = deepcopy(data.reports)
    value = report.build_composite_report(data.indexes['original'], data.indexes['supplement'])
    assert len(data.calls) == 2 and data.reports == before
    assert value['training_admission_verified'] is False and value['score_based_source_selection'] is False
    assert 'benchmark_run_root' not in value and value['synthetic_seven_track_run_created'] is False
    for row in value['tracks']:
        selected = 'supplement' if row['track'] in report.SUPPLEMENTED else 'original'
        assert row['source']['source_name'] == selected
        assert row['A'] == (10 if selected == 'supplement' else 90)
        assert row['source']['checkpoint_identity'] == data.reports[str(data.indexes[selected])]['checkpoint_identity']
        assert row['source']['harness']['selected_harness_root'] == '/fixture/' + selected
    assert value['numeric_task_count'] == 2 and value['unavailable_process_stage_count'] == 28
    report_row = next(row for row in value['tracks'] if row['track'] == 'report')
    assert report_row['stages']['S4']['A'] is None and report_row['stages']['S5']['A'] is None
    assert report_row['mean_A_T'] is None
    assert set(value['original_failed_tracks']) == set(report.SUPPLEMENTED)
    assert all(row['failure_preserved'] for row in value['original_failed_tracks'].values())
    text = report.render_markdown(value)
    assert 'not a single seven-track actor run' in text and 'N/A' in text
    assert '| classification | supplement | 10 |' in text
    assert 'Judge runtime versions are not inferred from actor versions' in text
    assert value['sources']['original']['judge_runtime']['inferred_from_actor_version'] is False


@pytest.mark.parametrize('mutation', ['checkpoint', 'scope', 'not_zero_turn'])
def test_invalid_pair_cannot_be_relabelled_as_composed_report(tmp_path, monkeypatch, mutation):
    data = fixture(tmp_path, monkeypatch)
    if mutation == 'checkpoint':
        path = Path(data.documents['supplement']['checkpoint_identity'])
        value = json.loads(path.read_text()); value['hf_asset_commitments'] = {'headers': 'different'}
        data.put(path, value)
        data.reports[str(data.indexes['supplement'])]['checkpoint_identity'] = commitment(path)
        error = 'checkpoint_lineage_or_assets_differ'
    elif mutation == 'scope':
        path = Path(data.documents['supplement']['benchmark_run_root']) / 'track-rollouts/attempt.json'
        path.unlink()
        data.put(path, {'tracks': ['classification', 'detection', 'vqa']}, committed=True)
        error = 'actor_scope_differs'
    else:
        path = Path(data.documents['original']['benchmark_run_root']) / 'track-rollouts/classification/rollout.json'
        path.unlink()
        data.put(path, {'actual_turn_count': 1}, committed=True)
        error = 'original_track_not_failed_zero_turn'
    with pytest.raises(ValueError, match=error):
        report.build_composite_report(data.indexes['original'], data.indexes['supplement'])


def test_actual_source_report_builder_can_render_all_unavailable_without_admission(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(report, 'build_report', real_build_report)
    value = report.build_composite_report(data.indexes['original'], data.indexes['supplement'])
    assert value['verified_process_stage_count'] == 0 and value['unavailable_process_stage_count'] == 35
    assert value['numeric_task_count'] == 0 and all(row['A_plus_T'] is None for row in value['tracks'])
    assert value['report_only'] and not value['training_admission_verified']


def test_cli_writes_fresh_report_only_and_rejects_source_overlap(tmp_path, monkeypatch, capsys):
    data = fixture(tmp_path, monkeypatch)
    value = report.build_composite_report(data.indexes['original'], data.indexes['supplement'])
    script = Path(__file__).resolve().parents[1] / 'scripts/build_eva_rsi_composite_report_v1.py'
    spec = importlib.util.spec_from_file_location('composite_report_cli', script)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, 'build_composite_report', lambda *args: deepcopy(value))
    output = tmp_path / 'new-report'
    argv = [str(script), '--original-index', str(data.indexes['original']),
        '--supplement-index', str(data.indexes['supplement']), '--output-root', str(output)]
    before = {path: path.read_bytes() for path in tmp_path.rglob('*.json')}
    monkeypatch.setattr(cli.sys, 'argv', argv)
    cli.main()
    assert json.loads((output / 'composite-report.json').read_text())['schema'] == report.SCHEMA
    assert (output / 'composite-report.md').stat().st_mode & 0o777 == 0o600
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert json.loads(capsys.readouterr().out)['training_admission_verified'] is False
    with pytest.raises(FileExistsError):
        cli.main()
    monkeypatch.setattr(cli.sys, 'argv', [*argv[:-1], str(data.indexes['original'].parent / 'forbidden')])
    with pytest.raises(ValueError, match='output_overlaps_source'):
        cli.main()
