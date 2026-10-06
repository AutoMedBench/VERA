"""Enforce pinned split bytes on every call; reuse only their validated parse."""
from functools import lru_cache
import json
from common import ROOT, sha

PLAN = ROOT/'evamed-codex/runs/direct-rl-96h100-readiness-20260914/selection-plan.json'
PLAN_SHA256 = 'eb67c6e379f8bffa1a75b6a6a796bda03d9ce3f1d854916c54b7a6b7c17c69da'
NEW_PLAN = ROOT/'evamed-codex/datasets-derived/harder-e2e-v2f-20260920/new-task-splits.json'
NEW_PLAN_SHA256 = '81f3ec294265f959735bb6414868780e6b9ae29eb15ddc133198c47d8834ee00'


@lru_cache(maxsize=1)
def _validated_splits(raw):
    # Preserve common.read's duplicate-key and non-finite JSON refusals while
    # parsing the exact bytes already read and authenticated by this call.
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate_json_key')
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_json')))
    rows = value['cases']
    result = {row['id']: row['split'] for row in rows}
    if len(rows) != len(result) or len(result) != 6000:
        raise ValueError('reserved_split_inventory_changed')
    return result


def splits():
    raw = PLAN.read_bytes()
    if sha(raw) != PLAN_SHA256:
        raise ValueError('reserved_split_plan_changed')
    result = _validated_splits(raw)
    if len(result) != 6000:
        raise ValueError('reserved_split_inventory_changed')
    # Callers receive independent maps; cache contents cannot be mutated through
    # the public API. The cache key is immutable verified bytes, never file stat.
    combined = dict(result)
    raw_new = NEW_PLAN.read_bytes()
    if sha(raw_new) != NEW_PLAN_SHA256:
        raise ValueError('harder_reserved_split_plan_changed')
    from collections import Counter
    new = json.loads(raw_new)
    if (new['schema'] != 'eva.fresh-harder-task-splits.v1'
            or len(new['cases']) != 80
            or Counter(row['split'] for row in new['cases']) != {'train':64, 'validation':8, 'test':8}):
        raise ValueError('harder_reserved_split_inventory_changed')
    for row in new['cases']:
        if row['id'] in combined or row['id'] in new['development_cases_excluded']:
            raise ValueError('harder_case_identity_collision')
        combined[row['id']] = row['split']
    return combined


def training_ids():
    return {case_id for case_id, split in splits().items() if split == 'train'}


def require_training_case(case_id):
    if splits().get(case_id) != 'train':
        raise ValueError('reserved_case_forbidden_in_training_or_harness_search')
