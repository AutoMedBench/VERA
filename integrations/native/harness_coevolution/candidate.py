"""Immutable candidate instructions/skill; prospective application before signing."""
from dataclasses import replace
from pathlib import Path
import re

from .feedback import ROOT, committed, digest, file_digest, require
from evamed_portable.integrity import write_json

SCHEMA = 'eva.portable-harness-candidate.v1'
FIELDS = {'name', 'workflow_instructions', 'skill_markdown', 'hypothesis', 'public_evidence_refs', 'risks'}


def validate_proposal(proposal, feedback):
    require(set(proposal) == FIELDS, 'candidate_fields_differ')
    for key in ('name', 'workflow_instructions', 'skill_markdown', 'hypothesis'):
        require(isinstance(proposal[key], str) and proposal[key].strip(), 'candidate_text_missing')
    require(len(proposal['workflow_instructions']) <= 6000 and len(proposal['skill_markdown']) <= 8000,
            'candidate_context_budget_exceeded')
    require(isinstance(proposal['risks'], list) and all(isinstance(x, str) for x in proposal['risks']), 'candidate_risks_invalid')
    refs = proposal['public_evidence_refs']
    allowed = {e['event_id'] for row in feedback['episodes'] for e in row['tool_events']}
    require(isinstance(refs, list) and refs and len(refs) == len(set(refs)) and set(refs) <= allowed,
            'candidate_evidence_not_public_feedback')
    instructions = proposal['workflow_instructions']+'\n'+proposal['skill_markdown']
    # IDs and digest literals are provenance in the proposal, never reusable
    # actor instructions. This check complements, not replaces, independent review.
    require(not re.search(r'\b(?:[0-9a-f]{8}-[0-9a-f-]{27,}|[0-9a-f]{32,})\b', instructions, re.I),
            'candidate_contains_task_identity_or_digest_literal')
    require('summary_failures' in proposal['skill_markdown'], 'candidate_skill_alias_changed')


def materialize(proposal, feedback, output, generation_receipt):
    validate_proposal(proposal, feedback)
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT), 'candidate_output_outside_workspace')
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    skill = output/'skills/summary-failures/SKILL.md'
    from .teacher_api import blob
    from .trial_runner import journal
    blob(skill,proposal['skill_markdown'].encode())
    document = {'schema':SCHEMA, 'name':proposal['name'], 'workflow_instructions':proposal['workflow_instructions'],
        'skill':{'skill_id':'summary_failures', 'name':'summary-failures', 'path':str(skill), 'content_blake3':file_digest(skill)},
        'hypothesis':proposal['hypothesis'], 'public_evidence_refs':proposal['public_evidence_refs'], 'risks':proposal['risks'],
        'feedback_blake3':feedback['document_blake3'], 'generation_receipt':str(generation_receipt),
        'state':'unpromoted_candidate', 'performance_verified':False}
    document['document_blake3'] = digest(document)
    journal(output/'candidate.json',document)
    return document


def apply_candidate(options, value, path):
    """For a prospective runner: call AFTER Supra composition, BEFORE authorization.

    This is not installed in the sealed v25 training source. The existing launch
    signatures then bind the replacement text and exact skill bytes normally.
    """
    from eva_agent.codex_runtime.research_memory import MEMORY_INSTRUCTIONS
    from eva_agent.codex_runtime.supra import WORKFLOW_INSTRUCTIONS
    from eva_agent.codex_runtime import CodexSkill
    candidate = committed(path)
    require(candidate['schema'] == SCHEMA, 'candidate_schema_invalid')
    skill = candidate['skill']; source = Path(skill['path']).resolve(strict=True)
    require(source.is_relative_to(ROOT) and file_digest(source) == skill['content_blake3'], 'candidate_skill_changed')
    text = options.developer_instructions
    require(text.count(MEMORY_INSTRUCTIONS) == text.count(WORKFLOW_INSTRUCTIONS) == 1,
            'candidate_baseline_profile_changed')
    require(len([s for s in value.skills if s.skill_id == 'summary_failures']) == 1,
            'candidate_baseline_skill_changed')
    revised = text.replace(MEMORY_INSTRUCTIONS, candidate['workflow_instructions']).replace(WORKFLOW_INSTRUCTIONS, '')
    skills = tuple(CodexSkill(**skill) if s.skill_id == 'summary_failures' else s for s in value.skills)
    return replace(options, developer_instructions=revised), replace(value, skills=skills)
