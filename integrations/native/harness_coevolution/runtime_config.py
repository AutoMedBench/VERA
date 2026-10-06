"""Propagate the same immutable harness configuration to Ray actors and rollouts."""
import os
from pathlib import Path
from .driver import validate_config
from .cycle import validate_spec
from .feedback import ROOT, require


def install(ray_environment):
    selected = {key:os.environ[key] for key in ('EVAMED_HARNESS_DRIVER_CONFIG','EVAMED_HARNESS_CYCLE_SPEC')
                if os.environ.get(key)}
    require(len(selected) <= 1,'choose_one_harness_driver_or_manual_cycle')
    if not selected:return {'enabled':False}
    key,path = next(iter(selected.items()))
    source = Path(path).resolve(strict=True)
    require(source.is_relative_to(ROOT),'harness_runtime_config_outside_workspace')
    value = validate_config(source) if key == 'EVAMED_HARNESS_DRIVER_CONFIG' else validate_spec(source)
    ray_environment[key] = str(source)
    return {'enabled':True,'environment_key':key,'config_path':str(source),'config_blake3':value['document_blake3'],
            'candidate_requires_native_comparison':True,'harness_teacher_and_static_review_reward_weight':0}
