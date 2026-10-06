"""Host-only prospective overlay; never supplies numerical answers to actors."""
from copy import deepcopy
from .public_bindings import POLICY, derive, specification

REPLACEMENTS = {
    's3_artifact': {'source_hash_bound': 'source_bundle_sha256', 'unit_rule_exact': 'unit_of_analysis_rule'},
    's4_artifact': {'source_hash_bound': 'source_bundle_sha256',
                    'unit_rule_recomputed_exact': 'unit_of_analysis_rule',
                    'identifier_no_enrichment_exact': 'synthetic_identifier_use_policy'},
    'terminal': {'source_hash_exact': 'source_bundle_sha256', 'unit_scope_exact': 'unit_of_analysis'},
}


def revise(construction, evidence_bytes):
    result = deepcopy(construction)
    binding = derive(evidence_bytes)
    public = specification()
    delta = []
    for stage, replacements in REPLACEMENTS.items():
        spec = result['runtime'][stage]
        checks = {c['check_id']: c for c in spec['checks']}
        if len(checks) != len(spec['checks']):
            raise ValueError('duplicate_source_check_id')
        for check_id, field in replacements.items():
            check = checks[check_id]
            if check['op'] != 'eq' or check['pointer'] != '/' + field:
                raise ValueError('unexpected_source_metadata_check')
            # Old expected values are neither copied to public metadata nor used
            # to derive replacements. New values come exclusively from evidence.
            check['value'] = binding[field]
            check['prospective_public_binding'] = POLICY
            spec['json_schema']['properties'][field]['description'] = public[field]
            delta.append({'stage': stage, 'check_id': check_id, 'pointer': '/' + field,
                          'replacement_source': 'declared_public_evidence_and_public_representation_rule'})
        original = construction['runtime'][stage]
        for check in spec['checks']:
            if check['check_id'] not in replacements:
                old = next(c for c in original['checks'] if c['check_id'] == check['check_id'])
                if check != old:
                    raise ValueError('nonmetadata_source_check_changed')
        for key in set(spec) - {'checks', 'json_schema'}:
            if spec[key] != original[key]:
                raise ValueError('source_gate_or_artifact_limit_changed')
    result['prospective_public_artifact_bindings'] = public
    return result, delta
