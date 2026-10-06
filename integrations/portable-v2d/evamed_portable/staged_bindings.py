"""Extend public metadata derivation to HealthBench S3/S4/S5 task contracts."""
from evamed_portable.derive_contract import revise as revise_public_metadata, REPLACEMENTS
from evamed_portable.clarify_contract import clarify

POLICY = 'eva.healthbench-staged-public-artifact-bindings.v1'
STAGES = ('S3', 'S4', 'S5')


def revise(construction, evidence_bytes, *, focus_stage):
    if focus_stage not in STAGES:
        raise ValueError('staged_healthbench_focus_not_admitted')
    result, delta = revise_public_metadata(construction, evidence_bytes)
    result = clarify(result)
    public = result['prospective_public_artifact_bindings']
    public['schema'] = POLICY
    public['scope'] = ('Prospective HealthBench Professional S3/S4/S5 task metadata bindings; '
                       'all fresh S1-S5 execution prerequisites remain required. '
                       'Not a historical source contract.')
    for stage, replacements in REPLACEMENTS.items():
        for check in result['runtime'][stage]['checks']:
            if check['check_id'] in replacements:
                check['prospective_public_binding'] = POLICY
    return result, delta
