"""CPU-only synthetic fixtures; no benchmark metric or model result is claimed."""
import json
from pathlib import Path
from types import SimpleNamespace

from blake3 import blake3
import pytest

from training.automedbench_lite import scorer_runtime, track_scoring
from training.automedbench_lite.adapter import EvaluationError, canonical, file_digest, write_once
from test_automedbench_model_provenance import job_fixture, put


def test_authentic_size_tokenizer_keeps_exact_digest_and_non_actor_attribution(tmp_path):
    args, root = job_fixture(tmp_path)
    relative = "executed-model-source/vqa_vision/tokenizer.json"
    body = b'{"fixture":"' + b'x' * (2_224_041 - 14) + b'"}'
    assert len(body) == 2_224_041
    digest = put(root / relative, body)
    receipt = json.loads((root / 'receipt.json').read_text())
    receipt['executed_model_sources'].append({'asset': 'vqa_vision', 'path': relative, 'blake3': digest})
    receipt_digest = put(root / 'receipt.json', receipt)
    exit_record = json.loads((root / 'process-exit.json').read_text())
    exit_record['worker_receipt_blake3'] = receipt_digest
    exit_digest = put(root / 'process-exit.json', exit_record)
    for path, row in args['visible_files'].items():
        row['blake3'] = exit_digest if path.endswith('/process-exit.json') else receipt_digest
    job = track_scoring.completed_model_evidence(**args)[0]
    source = next(row for row in job['sources'] if row['path'] == relative)
    assert source['bytes'] == 2_224_041 and source['blake3'] == digest and 'text' not in source
    assert not job['actor_authored'] and job['authored_by'] == 'provided_analysis_tool'
    (root / relative).write_bytes(b'y' * len(body))
    with pytest.raises(EvaluationError, match='model_executed_source_changed'):
        track_scoring.completed_model_evidence(**args)


@pytest.mark.parametrize('relative,size', [
    ('executed-modules/tokenizer.json', 2_224_041),
    ('executed-model-source/vqa_vision/config.json', 2_224_041),
    ('executed-model-source/vqa_vision/tokenizer.json', 4 * 1024**2 + 1),
])
def test_larger_bound_is_tokenizer_metadata_only_and_still_bounded(tmp_path, relative, size):
    put(tmp_path / relative, b'x' * size)
    with pytest.raises(EvaluationError, match='file_topology_or_size_invalid'):
        track_scoring.model_source_file(tmp_path, relative)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / 'public-cache'
    weight = root / scorer_runtime.ALEXNET_RELATIVE
    payload = b'synthetic fixture, never loaded by Torch'
    put(weight, payload)
    weight.chmod(0o400)
    monkeypatch.setattr(scorer_runtime, 'ALEXNET_BYTES', len(payload))
    monkeypatch.setattr(scorer_runtime, 'ALEXNET_BLAKE3', blake3(payload).hexdigest())
    return root, weight


def test_cache_is_exact_readonly_and_no_network_runtime(cache):
    root, weight = cache
    result = scorer_runtime.verify_lpips_cache(root)
    assert result['device'] == 'cpu' and result['network_downloads_allowed'] is False
    assert result['blake3'] == file_digest(weight)
    weight.chmod(0o600)
    with pytest.raises(ValueError, match='file_metadata_differs'):
        scorer_runtime.verify_lpips_cache(root)
    weight.write_bytes(b'x' * weight.stat().st_size)
    weight.chmod(0o400)
    with pytest.raises(ValueError, match='file_digest_differs'):
        scorer_runtime.verify_lpips_cache(root)


def test_cache_symlink_is_not_admitted(cache, tmp_path):
    root, _ = cache
    link = tmp_path / 'link'
    link.symlink_to(root)
    with pytest.raises(ValueError, match='directory_invalid'):
        scorer_runtime.verify_lpips_cache(link)


def test_new_score_root_and_verified_cache_are_forwarded_without_gpu_or_original_writes(
        tmp_path, monkeypatch, cache):
    torch_home, _ = cache
    run, public, private = (tmp_path / name for name in ('run', 'public', 'scorer'))
    for path in (run, public, private): path.mkdir()
    workspace = run / 'actor'
    workspace.mkdir()
    cases = [f'CT{index:03d}' for index in range(1, 21)]
    write_once(workspace / 'task.json', {'case_ids': cases})
    write_once(run / 'track-run-manifest.json', {'run_id': run.name, 'tracks': [{
        'track': 'enhancement', 'workspace_relative': 'actor', 'source_receipt_path': '/fixture/source',
        'source_acquisition_document_blake3': 'a' * 64, 'task_file_blake3': file_digest(workspace / 'task.json'),
        'input_manifest_file_blake3': 'b' * 64}]})
    audit = run / 'track-rollouts/enhancement'
    (audit / 'turns/05-review/after').mkdir(parents=True)
    write_once(audit / 'turns/05-review/after/manifest.json', {'files': []})
    write_once(audit / 'rollout.json', {'completed_requested_turns': True, 'errors': []})
    old = run / 'native-scores/enhancement/score.json'
    put(old, b'original score sentinel; never overwrite')
    monkeypatch.setattr(track_scoring, 'TrackRelease', lambda _: SimpleNamespace(
        public=public, scorer=private, document={'document_blake3': 'a' * 64},
        inventory={'automedbench_release/fixture.py': {'blake3': 'c' * 64}}))
    monkeypatch.setattr(track_scoring, 'completed_model_evidence', lambda **_: [])
    calls = []
    def fake_worker(argv, **kwargs):
        calls.append((argv, kwargs))
        environment = kwargs['env']
        assert environment['CUDA_VISIBLE_DEVICES'] == environment['HIP_VISIBLE_DEVICES'] == ''
        assert environment['TORCH_HOME'] == str(torch_home)
        assert 'OPENAI_API_KEY' not in environment and 'HOME' not in environment
        request = json.loads(kwargs['input'])
        assert request['lpips_cache'] == scorer_runtime.verify_lpips_cache(torch_home)
        assert request['selected_case_ids'] == cases
        return SimpleNamespace(returncode=0, stdout=canonical({
            'schema': 'eva.automedbench-native-track-result.v1', 'track': 'enhancement',
            'selected_case_count': 20, 'full_public_subset': True, 'private_values_exported': False,
            'task_score_0_1': 0.0, 'fixture_not_a_real_metric': True}))
    monkeypatch.setattr(track_scoring.subprocess, 'run', fake_worker)
    output = tmp_path / 'new-score'
    track_scoring.score_track(run, 'enhancement', Path('/fixture/python'),
        lpips_torch_home=torch_home, score_output_root=output)
    assert len(calls) == 1 and old.read_bytes() == b'original score sentinel; never overwrite'
    assert (output / 'enhancement/score.json').exists()
    with pytest.raises(FileExistsError):
        track_scoring.score_track(run, 'enhancement', Path('/fixture/python'),
            lpips_torch_home=torch_home, score_output_root=output)
    assert len(calls) == 1
