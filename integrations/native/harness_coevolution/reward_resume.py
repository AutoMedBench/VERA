"""Reuse only independently reverified roles tied to the same actual actor."""
from pathlib import Path

from .feedback import read, require


def verified_prior_roles(current_source, prior_outputs):
    from evamed_codex.portable_stage_target import _comparable_projection
    from eva_agent.rubrics.models import CompiledRubric
    from eva_agent.rubrics.registry import CompiledRubricRegistry
    from reward_execution_policy import for_output
    execution = for_output(Path(current_source).parent)
    verify_assessment, MODELS = execution.verify_assessment, execution.models
    current = _comparable_projection(read(current_source))
    results = {}
    for root in map(Path, prior_outputs):
        require(for_output(root).name == execution.name, 'prior_reward_execution_policy_changed')
        if not (root/'source.json').is_file():continue
        require(_comparable_projection(read(root/'source.json')) == current,
                'prior_reward_belongs_to_other_actor_or_contract')
        for role, source_name in [('judge','process-source.json'), ('reward-verifier','hacking-source.json')]:
            if role in results:continue
            source = root/source_name
            if not source.is_file():continue
            table = read(source)['rubric_table']
            expected = read(Path(current_source).parent/source_name)['rubric_table']
            require(table['rubric_digest'] == expected['rubric_digest'], 'prior_reward_rubric_changed')
            CompiledRubricRegistry._verify_rubric_document(table)
            rubric = CompiledRubric.from_document(table)
            targets = [root/role, *sorted(root.glob(role+'-recovery-*'))]
            ordered_models = MODELS
            from reward_execution_policy import FAST
            if execution.name == FAST:
                from fast_judge_policy09 import assignment
                ordered_models = assignment(read(source)['executable_episode_id'], role)
            from reward_execution_policy import POLICY as CURRENT_POLICY
            if execution.name == CURRENT_POLICY:
                from terra_judge_policy import assignment
                ordered_models = assignment(read(source)['executable_episode_id'], role)
            for target in targets:
                selection_path = target/'fallback-selection.json'
                selection = read(selection_path) if selection_path.is_file() else None
                chosen = selection.get('selected_assessment_id') if selection else None
                for route, model, _ in ordered_models:
                    for attempt in sorted((target/route).glob('attempt-*')):
                        if not (attempt/'grade.json').is_file():continue
                        try:
                            verified = verify_assessment(source,rubric,attempt,expected_model=model,expected_route=route)
                        except Exception:
                            if chosen and (attempt/'attempt.json').is_file():
                                require(not chosen or chosen != read(attempt/'attempt.json')['assessment_id'],
                                        'previously_admitted_reward_integrity_failure')
                            # Interrupted/invalid assessments stay retained and
                            # unavailable; no zero or preferred-score selection.
                            continue
                        if chosen:
                            require(verified['assessment_id'] == chosen
                                and selection.get('selected_model') == model
                                and selection.get('selected_route') == route,
                                'previously_admitted_reward_selection_changed')
                        elif selection is not None:
                            require(False, 'withheld_selection_has_valid_assessment')
                        results[role] = {**verified, 'reused_verified_assessment':True,
                            'assessment_source':str(source), 'assessment_output':str(attempt),
                            'fallback_used':route != ordered_models[0][0]}
                        break
                    if role in results:break
                if role in results:break
    return results
