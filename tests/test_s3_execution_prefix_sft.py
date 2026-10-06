"""Small CPU fixtures; real signatures/artifacts are reopened by the CLI canary."""
import hashlib
import json
from types import SimpleNamespace
import pytest
from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes
from eva_agent.training.s2_skill_frontier_support import canonical_skill_tools
from eva_agent.training.s3_execution_prefix_sft import S3PilotSFTError, _execution, _jsonl, _public_input_root, _skills


def test_s3_jsonl_has_one_line_per_row_even_with_multiline_code():
    rows = [{'code': 'print(1)\nprint(2)\n'}, {'code': 'print(3)\n'}]
    lines = _jsonl(rows).splitlines()
    assert len(lines) == len(rows) and [json.loads(line) for line in lines] == rows


@pytest.mark.parametrize('tamper', (False, True))
def test_input_locator_uses_recorded_cross_family_paths(tmp_path, tamper):
    sha = lambda data: hashlib.sha256(data).hexdigest()
    manifest = tmp_path / 'family/construction/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b'{}')
    inputs = manifest.parent / 'construction-private/04-solver-input'
    inputs.mkdir(parents=True)
    subject = tmp_path / 'unusual-promotion-location.json'
    subject.write_bytes(canonical_json_bytes({'source_evidence': [{'role': 'candidate_construction_manifest',
        'artifact': {'path': manifest.relative_to(tmp_path).as_posix(), 'sha256': sha(manifest.read_bytes())}}]}))
    registry = tmp_path / 'runs/evamed-campaign-supervisor-v24-attempt1/candidate-registry.v24.json'
    registry.parent.mkdir(parents=True)
    registry.write_bytes(canonical_json_bytes({'candidates': [{'candidate_id': 'source', 'proofs': {'promoted': {
        'subject': {'path': subject.name, 'sha256': '0' * 64 if tamper else sha(subject.read_bytes())}}}}]}))
    modules = SimpleNamespace(canonical=SimpleNamespace(sha256_bytes=sha))
    if tamper:
        with pytest.raises(S3PilotSFTError, match='construction_subject_commitment'):
            _public_input_root(tmp_path, 'source', modules)
    else:
        assert _public_input_root(tmp_path, 'source', modules) == inputs


def test_s3_skill_context_uses_canonical_stage_and_exact_loaded_content():
    content = 'Only the declared output file.'
    item = SimpleNamespace(arguments={'stage': 'S3', 'skill_id': 'visible'},
        output={'skill_id': 'visible', 'content': content, 'content_blake3': blake3_hex(content),
                'delivery': 'policy-visible-tool-observation'},
        tool_result={'name': 'load_skill', 'workspace_before_blake3': 'a' * 64, 'workspace_after_blake3': 'a' * 64})
    _skills([item], canonical_skill_tools(), {'visible_skill_ids': ['visible']})
    item.arguments['stage'] = 'S2'
    with pytest.raises(S3PilotSFTError, match='actor_tool_stage'):
        _skills([item], canonical_skill_tools(), {'visible_skill_ids': ['visible']})
    item.arguments['stage'] = 'S3'
    item.output['content'] += ' changed'
    with pytest.raises(S3PilotSFTError, match='skill_load_binding'):
        _skills([item], canonical_skill_tools(), {'visible_skill_ids': ['visible']})


def _fixture():
    sha = lambda data: hashlib.sha256(data).hexdigest()
    value_sha = lambda value: sha(canonical_json_bytes(value))
    code, data, artifact = b'print(1)\n', b'public input', b'{"computed":1}'
    contract = {'stage': 'S3', 'focus': 'S3', 'episode_id': 'episode', 'prerequisite_receipt_sha256': 'prereq',
        'artifact': {'relative_path': 'work/pilot.json', 'json_schema': {'type': 'object'}}, 'limits': {},
        'inputs': [{'relative_path': 'evidence/input', 'byte_count': len(data), 'sha256': sha(data)}]}
    checks = {'computed': True}
    artifact_obs = {'byte_count': len(artifact), 'sha256': sha(artifact), 'parser_valid': True,
        'schema_valid': True, 'checks': checks, 'critical_checks_passed': True, 'newly_created': True,
        'host_reopened': True, 'reopen_sha256_match': True, 'no_unexpected_files': True}
    obs = {'prerequisites': {'s2_verified': True},
        'contract': {'submission_sha256': sha(code), 'contract_sha256': value_sha(contract)},
        'artifact': artifact_obs, 'workspace': {'changed': True, 'file_count': 1, 'total_bytes': len(artifact), 'within_limits': True},
        'isolation': {'verified': True}, 'execution': {'gate_passed': True}}
    envelope = {'payload': {'episode_id': 'episode', 'observations': obs}, 'signature': 'fixture-only'}
    bundle = {'host/execution-contract.json': canonical_json_bytes(contract), 'host/host-receipt.json': canonical_json_bytes(envelope),
        'inputs/evidence/input': data,
        'host/submission.py': code, 'host/workspace/work/pilot.json': artifact}
    def verify(e, _trust):
        assert e['signature'] == 'fixture-only'
        return e['payload']
    def tool(e):
        return {'gate_passed': e['payload']['observations']['execution']['gate_passed'], 'failed_check_ids': []}
    def evaluate(_contract, content):
        return {**artifact_obs, 'sha256': sha(content), 'all_checks_passed': json.loads(content) == {'computed': 1}}
    modules = SimpleNamespace(host_receipts=SimpleNamespace(verify_host_receipt=verify),
        validation=SimpleNamespace(validate_execution_contract=lambda _contract: None),
        canonical=SimpleNamespace(sha256_bytes=sha, sha256_value=value_sha),
        agent_rollout=SimpleNamespace(_tool_result_from_execution=tool),
        execution=SimpleNamespace(evaluate_json_artifact_content=evaluate))
    target = SimpleNamespace(arguments={'stage': 'S3', 'code': code.decode()},
        output={**tool(envelope), 'attempt': 1, 'host_receipt_sha256': sha(bundle['host/host-receipt.json'])})
    runtime = {'active_episode_id': 'episode', 'execution_stages': {'S3': {
        'artifact_relative_path': 'work/pilot.json', 'artifact_json_schema': {'type': 'object'}, 'limits': {}}}}
    return bundle, target, runtime, {'evidence/input': data}, modules


def test_s3_positive_fixture_distinguishes_execution_workspace_and_real_code():
    bundle, target, runtime, actor, modules = _fixture()
    row = _execution(bundle, target, runtime, actor, 'prereq', modules, {})
    assert row['host_gate_passed'] is True and row['execution_workspace_changed'] is True
    assert row['artifact_byte_count'] == len(bundle['host/workspace/work/pilot.json'])
    assert 'actor_workspace_changed' not in row  # Actor boundary is independently checked by _slice.


@pytest.mark.parametrize('mutation,reason', [
    ('code', 'actual_code_contract_binding'), ('artifact', 'independent_host_artifact_check'),
    ('extra-file', 'unexpected_execution_files'), ('input', 'execution_input_binding'),
    ('prerequisite', 'execution_prerequisite_binding'), ('gate', 'S3_host_gate_not_passed'),
    ('public-contract', 'public_execution_contract'),
])
def test_s3_rejects_actual_evidence_mismatch_and_failed_gate(mutation, reason):
    bundle, target, runtime, actor, modules = _fixture()
    if mutation == 'code': bundle['host/submission.py'] += b'# changed'
    elif mutation == 'artifact': bundle['host/workspace/work/pilot.json'] = b'{"computed":2}'
    elif mutation == 'extra-file': bundle['host/workspace/trace.json'] = b'{}'
    elif mutation == 'input': bundle['inputs/evidence/input'] += b'changed'
    elif mutation == 'prerequisite': runtime['active_episode_id'] = 'another-episode'
    elif mutation == 'public-contract': runtime['execution_stages']['S3']['artifact_relative_path'] = 'other.json'
    else:
        envelope = json.loads(bundle['host/host-receipt.json'])
        envelope['payload']['observations']['execution']['gate_passed'] = False
        bundle['host/host-receipt.json'] = canonical_json_bytes(envelope)
        target.output['gate_passed'] = False
        target.output['host_receipt_sha256'] = modules.canonical.sha256_bytes(bundle['host/host-receipt.json'])
    with pytest.raises(S3PilotSFTError, match=reason):
        _execution(bundle, target, runtime, actor, 'prereq', modules, {})
