#!/usr/bin/env python3
"""Run the opt-in repository-local research profile, never the benchmark profile."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from eva_agent.codex_runtime.research_profile import main

if __name__ == '__main__':
    raise SystemExit(main())
