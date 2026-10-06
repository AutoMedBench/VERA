"""Scorer-only subprocess: expose exact native metric and bounded format facts.

Adapted from BAT's automedbench_native_worker.py at
27124a2a35fa9ee68ef890bd7e4a36ffb5802860; native math remains in the pinned release.
This in-process network denial is not an operating-system security boundary.
"""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import io
import json
import math
from pathlib import Path
import socket
import sys


def deny(*args, **kwargs):
    raise RuntimeError("network_disabled")


class NoNetworkSocket(socket.socket):
    connect = connect_ex = bind = listen = accept = deny


def main() -> int:
    socket.socket = NoNetworkSocket
    socket.create_connection = socket.getaddrinfo = deny
    try:
        request = json.loads(sys.stdin.buffer.read(32769))
        if set(request) != {"release_root", "submission_root", "track", "case_id", "metric", "reference_commitments"}:
            raise ValueError("request")
        metrics = {"classification": "balanced_accuracy", "detection": "mAP", "segmentation": "macro_mean_dice"}
        track = request["track"]
        case_id = request["case_id"]
        if track not in metrics or request["metric"] != metrics[track] or not case_id or any(c in case_id for c in "/\\"):
            raise ValueError("identity")
        release = Path(request["release_root"]).resolve(strict=True)
        submission = Path(request["submission_root"]).resolve(strict=True)
        sys.path.insert(0, str(release))
        # Imports or native routines may print diagnostics; none cross the boundary.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            from blake3 import blake3
            commitments = request["reference_commitments"]
            if not commitments:
                raise ValueError("reference_commitment")
            for relative, expected in commitments.items():
                path = (release / relative).resolve(strict=True)
                if not path.is_relative_to(release) or not path.is_file():
                    raise ValueError("reference_path")
                digest = blake3()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError("reference_commitment")
            from automedbench_release.raw_track_worker import TRACK_LAYOUTS, _SCORERS, _track_modules
            from automedbench_release.tasks import TASK_BY_TRACK
            import yaml
            layout = TRACK_LAYOUTS[track]
            task = TASK_BY_TRACK[track]
            eval_dir = release / layout["eval_dir"]
            config = yaml.safe_load((eval_dir / layout["config"]).read_text())
            if config["task_id"] != task.task_id:
                raise ValueError("task")
            if track == "classification" and config.get("score_metric") != metrics[track]:
                raise ValueError("metric")
            data = task.data_path(release)
            if track == "classification":
                required = data / "private" / case_id / "label.json"
            elif track == "detection":
                required = data / "private" / case_id / "boxes.json"
            else:
                required = data / "private" / config.get("gt_subdir", "") / case_id
            if not required.exists() or not (data / "public" / case_id / task.public_marker).is_file():
                raise ValueError("native_reference")
            with _track_modules(eval_dir, tuple(layout["modules"])) as modules:
                score, facts, _raw_private = _SCORERS[track](modules=modules, output_dir=submission,
                    public_dir=data / "public", private_dir=data / "private", selected=[case_id], config=config)
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1 or facts.get("expected_outputs") != 1:
            raise ValueError("score")
        present, valid = facts.get("present_outputs"), facts.get("valid_outputs")
        if type(present) is not int or type(valid) is not int or present not in (0, 1) or valid not in (0, 1):
            raise ValueError("counts")
        result = {"status": "scored" if present == valid == 1 else "invalid_submission", "track": track,
                  "case_id": case_id, "metric": metrics[track], "task_score_0_1": score,
                  "present_outputs": present, "valid_outputs": valid,
                  "output_format_valid": facts.get("output_format_valid") is True,
                  "network_disabled": True, "private_values_exported": False}
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except Exception:
        print('{"status":"error","error_code":"native_worker_failed"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
