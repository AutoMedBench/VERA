"""Read bounded track progress or score a terminal actual workflow once."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from training.automedbench_lite.adapter import read_document


def status(run_root):
    run = Path(run_root).resolve(strict=True)
    manifest = read_document(run / "track-run-manifest.json")
    result = []
    for track in manifest["tracks"]:
        audit = run / "track-rollouts" / track["track"]
        if not audit.exists(): continue
        events = []
        if (audit / "mcp-events.jsonl").exists():
            lines = (audit / "mcp-events.jsonl").read_bytes().splitlines()
            for line in lines:
                try: events.append(json.loads(line))
                except json.JSONDecodeError: pass  # incomplete in-flight line; never count it
        terminal = read_document(audit / "rollout.json") if (audit / "rollout.json").exists() else None
        score = run / "native-scores" / track["track"] / "score.json"
        result.append({"track": track["track"], "expected_case_outputs": track["case_count"],
            "terminal": terminal is not None, "completed_requested_turns": terminal.get("completed_requested_turns", False) if terminal else False,
            "actual_receipts_retained": len(list((audit / "turns").glob("*/receipt.json"))),
            "actual_host_tool_events": len(events), "successful_host_tool_events": sum(not row["is_error"] for row in events),
            "tool_name_counts": dict(Counter(row["name"] for row in events)),
            "native_score": read_document(score)["native_result"] if score.exists() else None,
            "native_score_missing_is_not_zero": True})
    return {"schema": "eva.automedbench-track-progress.v1", "run_id": manifest["run_id"], "tracks": result,
        "full_seven_track_evaluation": len(result) == 7 and all(row["native_score"] is not None for row in result)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("status", "score"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--track", choices=("classification", "detection", "segmentation", "synthesis", "vqa", "report", "enhancement"))
    parser.add_argument("--evaluator-python", type=Path)
    parser.add_argument("--lpips-torch-home", type=Path,
                        help="Enhancement only: explicitly pinned offline AlexNet cache")
    parser.add_argument("--score-output-root", type=Path,
                        help="Separate new native-score root; original run scores remain untouched")
    args = parser.parse_args()
    if args.mode == "status": result = status(args.run_root)
    else:
        if args.track is None or args.evaluator_python is None: parser.error("score requires --track and --evaluator-python")
        from training.automedbench_lite.track_scoring import score_track
        result = score_track(args.run_root, args.track, args.evaluator_python,
            lpips_torch_home=args.lpips_torch_home, score_output_root=args.score_output_root)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__": main()
