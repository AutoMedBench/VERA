#!/usr/bin/env python3
"""One fresh C/D supplement; default read-only preflight, --execute launches it.

Settings require allow_classification_detection_supplement=true and
supplemental_source_attempt; supplemental_workers defaults to two. The context
uses a fresh operator attempt_root and the original loop/round/checkpoint. This
command does not call the controller or attach its component index to a loop.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.harness_source import activate_harness
activate_harness(ROOT)
from training.eva_rsi.evidence import read
from training.eva_rsi.supplemental_eval import run_supplemental


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("context", type=Path)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run_supplemental(read(args.context), read(args.settings), execute=args.execute)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "unavailable", "error_type": type(error).__name__,
            "automatic_retry": False, "controller_state_modified": False}), file=sys.stderr, flush=True)
        raise SystemExit(1) from None
