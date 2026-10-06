"""Prospective public definitions; contains no case answers or grader values."""
from copy import deepcopy

POLICY = 'eva.healthbench-public-missingness-definitions.v1'
DEFINITIONS = {
    'missing_measure_count': (
        'Count frozen structured_payload.records rows whose measure_numerator is JSON null. '
        'Use every row before any eligibility, status, severity, conflict, or component filtering. '
        'This is a row count of missing numerators, not the union of missing numerators and '
        'missing denominators and not a component count. A numeric zero is present, not missing. '
        'Count missing denominators separately in missing_denominator_count; a row with both '
        'fields null contributes once to each count. Do not impute missing values.'),
    'missing_denominator_count': (
        'Count frozen structured_payload.records rows whose measure_denominator is JSON null. '
        'Use every row before any eligibility, status, severity, conflict, or component filtering. '
        'This is a row count, not a component count. A numeric zero is present, not missing. '
        'Count missing numerators separately in missing_measure_count; a row with both fields '
        'null contributes once to each count. Do not impute missing values.'),
}


def clarify(construction):
    """Add four schema descriptions; preserve every validation rule and value."""
    result = deepcopy(construction)
    targets = [result['runtime']['s3_artifact']['json_schema']['properties'],
               result['runtime']['s4_artifact']['json_schema']['properties']['recomputed']['properties']]
    for properties in targets:
        for field, description in DEFINITIONS.items():
            schema = properties[field]
            if schema.get('type') != 'integer' or schema.get('minimum') != 0:
                raise ValueError('unexpected_missingness_field_schema')
            if 'description' in schema and schema['description'] != description:
                raise ValueError('preexisting_missingness_definition_requires_review')
            schema['description'] = description
    return result
