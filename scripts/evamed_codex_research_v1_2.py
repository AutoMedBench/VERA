#!/usr/bin/env python3
"""Inspect the opt-in v1.2 SDK memory profile; actual actors use its composition helpers."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from eva_agent.codex_runtime.research_memory import inspect_memory_profile
from eva_agent.codex_runtime.research_profile import check_profile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codex-bin')
    args = parser.parse_args()
    result = inspect_memory_profile()
    if args.codex_bin:
        result['provider_free_binary_check'] = check_profile(args.codex_bin)
    print(json.dumps(result, sort_keys=True))
