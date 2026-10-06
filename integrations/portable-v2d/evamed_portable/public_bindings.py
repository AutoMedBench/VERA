"""Public metadata representation, derived only from declared evidence bytes.

This module defines a prospective task version. It contains no source expected
answers, numerical results, private assessment, or evaluator-only input.
"""
import hashlib
import json

POLICY = 'eva.healthbench-public-artifact-bindings.v1'


def canonical_text(value):
    """UTF-8 JSON representation: sorted keys, compact separators, no newline."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def specification():
    return {
        'schema': POLICY,
        'scope': 'Prospective HealthBench Professional E2E task metadata bindings; not a historical source contract.',
        'canonical_json': 'UTF-8 JSON, ensure_ascii=False, sort_keys=True, separators=(comma,colon), allow_nan=False, no trailing newline.',
        'source_bundle_sha256': (
            'Read the exact bytes of every evidence_id declared in task.evidence_inventory. '
            'For each file form an object with exactly evidence_id and sha256 (lowercase SHA-256 hex of its bytes). '
            'Sort the objects by evidence_id. Canonically serialize the resulting array using canonical_json above. '
            'The field is lowercase SHA-256 hex of those UTF-8 array bytes. '
            'This is a new public evidence-manifest digest, not a hidden upstream bundle digest.'),
        'unit_of_analysis_rule': (
            'Find the unique declared evidence object whose structured_payload.schema equals '
            'evamed.synthetic-professional-research-fixture.v1. Canonically serialize its entire '
            'structured_payload.unit_contract object using canonical_json. Use the resulting string in S3 and S4.'),
        'synthetic_identifier_use_policy': (
            'In S4, canonically serialize the public synthetic_identifier_policy object below using canonical_json. '
            'The identifiers are synthetic fixture labels only; do not enrich or use them for care.'),
        'synthetic_identifier_policy': {
            'care_facing': False, 'external_identifier_enrichment': False,
            'identifier_scope': 'synthetic_fixture_labels_only'},
        'unit_of_analysis': (
            'In S5, canonically serialize an object with exactly record_unit and component_unit, '
            'taking both values from the same public unit_contract. Use canonical_json.'),
        'unchanged_obligations': [
            'Compute all numerical results from the frozen evidence; do not infer results from metadata bindings.',
            'S4 independently recomputes and reconciles results from evidence in a fresh workspace.',
            'Complete all S1-S5 stages and obtain accepted submission and independent artifact reopen checks.',
            'Retain source confidentiality, isolation, no external enrichment, and non-clinical interpretation constraints.'],
    }


def derive(evidence_bytes):
    """Input is exactly {declared evidence_id: raw public file bytes}."""
    if not evidence_bytes or any(not isinstance(k, str) or not isinstance(v, bytes)
                                 for k, v in evidence_bytes.items()):
        raise ValueError('declared_public_evidence_bytes_required')
    manifest = [{'evidence_id': eid, 'sha256': hashlib.sha256(raw).hexdigest()}
                for eid, raw in sorted(evidence_bytes.items())]
    fixtures = []
    for raw in evidence_bytes.values():
        value = json.loads(raw)
        payload = value.get('structured_payload', {})
        if payload.get('schema') == 'evamed.synthetic-professional-research-fixture.v1':
            fixtures.append(payload)
    if len(fixtures) != 1:
        raise ValueError('unique_public_professional_fixture_required')
    unit = fixtures[0]['unit_contract']
    if not isinstance(unit, dict) or not {'record_unit', 'component_unit'} <= unit.keys():
        raise ValueError('public_unit_contract_required')
    return {
        'source_bundle_sha256': hashlib.sha256(canonical_text(manifest).encode('utf-8')).hexdigest(),
        'unit_of_analysis_rule': canonical_text(unit),
        'synthetic_identifier_use_policy': canonical_text(specification()['synthetic_identifier_policy']),
        'unit_of_analysis': canonical_text({k: unit[k] for k in ('record_unit', 'component_unit')}),
    }
