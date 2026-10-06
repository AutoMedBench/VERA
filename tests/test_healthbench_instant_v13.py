from pathlib import Path
from uuid import uuid4
import sys

import jsonschema
import pytest

from eva_agent.pipeline.contracts import ToolCall
from eva_agent.pipeline.digests import canonical_value
from training.healthbench_instant_v13 import (
    ACTOR_NAMES, CapturedTools, episode_from_public, judge_schema, successful_read_paths,
)
from training.automedbench_lite.public_tools import TOOLS
from eva_agent.codex_pipeline.turn_mcp import TurnMCPBridgeFactory


def public_episode():
    return {'episode_id': str(uuid4()), 'domain': 'healthbench-professional', 'stage': 'E2E',
            'source': {'benchmark': 'HealthBench Professional', 'source_file': 'fixture.jsonl', 'source_revision': 'fixture'},
            'instruction': 'Synthetic fixture question.',
            'policy_context': {'conversation': [{'role': 'user', 'content': 'Synthetic fixture question.'}]},
            'judge_only_reference': None}


def test_public_boundary():
    document = public_episode()
    episode = episode_from_public(document)
    assert episode.instruction == document['instruction']
    assert set(episode.initial_files) == {'task.json', 'inputs/conversation.json'}
    document['judge_only_reference'] = {'forbidden': 'private fixture'}
    with pytest.raises(ValueError):
        episode_from_public(document)


def test_actual_existing_read_note_and_skill_bindings(tmp_path):
    tools = CapturedTools(output=tmp_path, episode=episode_from_public(public_episode()), image='fixture-not-executed')
    offers = {offer.name.split('/')[-1]: offer for offer in tools.offers}
    for source in TOOLS:
        if source['name'] in ACTOR_NAMES:
            assert canonical_value(offers[source['name']].input_schema) == source['inputSchema']
            assert offers[source['name']].description == source['description']
    assert {'search_skills', 'load_skill'} <= offers.keys()
    calls = [ToolCall(str(uuid4()), 'automed_write_note', {'name': 'final-answer.md', 'content': 'Synthetic answer.'}),
             ToolCall(str(uuid4()), 'automed_read_file', {'path': 'notes/final-answer.md'}),
             ToolCall(str(uuid4()), 'automed_read_file', {'path': 'notes/missing.md'})]
    results = tools.execute(calls)
    assert results[0].output['isError'] is False
    assert results[1].output['isError'] is False
    assert results[2].output['isError'] is True
    assert successful_read_paths(tools) == {'notes/final-answer.md'}
    assert (tools.workspace.root / 'notes/final-answer.md').read_text() == 'Synthetic answer.'
    assert len(tools.after) == len(calls) == len(tools.trace().results)


def test_judge_read_only_toolset(tmp_path):
    tools = CapturedTools(output=tmp_path, episode=episode_from_public(public_episode()), image='fixture-not-executed', judge=True)
    assert [offer.name for offer in tools.offers] == ['automed_read_file']
    assert all(offer.server == 'automed_eval' for offer in tools.offers)
    assert all(offer.read_only for offer in tools.offers)


def test_exact_two_native_items_not_inflated():
    result = {'items': [{'index': i, 'satisfied': False, 'evidence_refs': ['task.json'], 'reason': 'Synthetic fixture.'} for i in range(2)]}
    jsonschema.validate(result, judge_schema(2))
    result['items'].append(result['items'][0])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(result, judge_schema(2))


def test_actual_stdlib_proxy_factory_is_provider_free(tmp_path):
    from training.healthbench_instant_v13 import ROOT
    factory = TurnMCPBridgeFactory(proxy_python=Path(sys.executable).resolve(),
        proxy_script=ROOT / 'src/eva_agent/codex_pipeline/turn_mcp_proxy.py',
        temp_root=tmp_path, maximum_parallel_calls=1)
    assert factory.proxy_python.is_file()


def test_instant_profile_is_explicit_local_mode():
    from training.healthbench_instant_v13 import instant_profile
    profile = instant_profile()
    assert profile.mode.value == 'instant'
    assert profile.protocol == 'qwen_template'
    assert profile.inspection()['qwen_thinking_requested'] is False
