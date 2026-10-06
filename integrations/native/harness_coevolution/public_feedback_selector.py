"""Prospective pure ordering only; production admission/authentication stays native.

Inputs are authenticated actor-public projections and already eligible metadata.
This module deliberately cannot open files, launch actors, or call providers.
"""
from dataclasses import dataclass
from typing import Mapping

DOMAINS = ('agentclinic', 'automedbench-classification', 'automedbench-detection',
           'automedbench-research', 'automedbench-segmentation',
           'healthbench-professional', 'medxpertqa', 'synthetic-annotation-migration')
STAGES = ['S1', 'S2', 'S3', 'S4', 'S5']
CATEGORIES = ('public_incomplete', 'public_recovery', 'public_clean_complete')


@dataclass(frozen=True)
class Candidate:
    domain: str
    optimizer_update: int
    original_index: int
    case_id: str
    actor: str
    category: str


def public_category(public):
    """Read only completion booleans/stages and explicit tool-error signals.

    In particular, public_result's clinical_or_ability_score is not accessed.
    Caller MUST authenticate this projection with native feedback.actor_public.
    """
    state = public['final_public_state']
    events = public['tool_events']
    if not isinstance(state, Mapping) or not isinstance(events, list):
        raise ValueError('public_projection_shape_invalid')
    if (not isinstance(state.get('completed_stages'), list) or
            not isinstance(state.get('next_stage'), str) or
            type(state.get('submission_attempted')) is not bool):
        raise ValueError('public_completion_shape_invalid')
    error_seen = False
    accepted_submission = False
    for event in events:
        result = event['public_result']
        if not isinstance(result, Mapping):
            raise ValueError('public_tool_result_shape_invalid')
        if 'gate_passed' in result and type(result['gate_passed']) is not bool:
            raise ValueError('public_gate_shape_invalid')
        error = result.get('error')
        if error is not None and not isinstance(error, (str, Mapping)):
            raise ValueError('public_error_shape_invalid')
        error_seen |= result.get('gate_passed') is False or bool(error)
        accepted_submission |= (event['name'] == 'submit_results' and
                                result.get('gate_passed') is True and
                                result.get('next_stage') == 'complete')
    complete = (state['next_stage'] == 'complete' and
                state['completed_stages'] == STAGES and
                state['submission_attempted'] is True)
    if complete != accepted_submission:
        raise ValueError('public_submission_state_inconsistent')
    if not complete:
        return CATEGORIES[0]
    return CATEGORIES[1] if error_seen else CATEGORIES[2]


def select(candidates, through_update, *, window=20):
    """Choose one eligible original/domain; no score, text, or error-count tie.

    Recency is bounded by the native window. Consecutive actual scheduled
    windows (interval == window == 20) cannot repeatedly select the same actor.
    Missing families remain missing; native probe policy is outside this helper.
    """
    if window != 20 or type(through_update) is not int or through_update < 1:
        raise ValueError('native_feedback_window_changed')
    chosen = {}
    identities = set()
    originals = set()
    for row in candidates:
        if (row.domain not in DOMAINS or row.category not in CATEGORIES or
                type(row.optimizer_update) is not int or row.optimizer_update < 1 or
                type(row.original_index) is not int or row.original_index < 0 or
                not row.actor or not row.case_id):
            raise ValueError('candidate_metadata_invalid')
        if row.actor in identities or row.original_index in originals:
            raise ValueError('duplicate_feedback_original')
        identities.add(row.actor)
        originals.add(row.original_index)
        if not through_update-window < row.optimizer_update <= through_update:
            continue
        key = (CATEGORIES.index(row.category), -row.optimizer_update, row.original_index)
        if row.domain not in chosen or key < chosen[row.domain][0]:
            chosen[row.domain] = (key, row)
    return [chosen[d][1] for d in DOMAINS if d in chosen]
