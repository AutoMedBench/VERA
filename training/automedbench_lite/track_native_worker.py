"""Opaque native whole-track scoring; never a policy or analysis-model process."""
from contextlib import redirect_stderr, redirect_stdout
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


COUNTS = {"classification": 100, "synthesis": 20, "detection": 100, "segmentation": 40,
          "vqa": 2005, "report": 100, "enhancement": 20}
FACTS = {"expected_outputs", "present_outputs", "parsed_outputs", "valid_outputs", "placeholder_outputs",
         "submission_format_valid", "output_format_valid", "model_call_detected", "scorer_metric_present",
         "scorer_metric_finite"}


def result_projection(track, selected, score, facts):
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("nonfinite_native_score")
    if facts.get("expected_outputs") != len(selected):
        raise ValueError("native_denominator_differs")
    kept = {}
    for name in FACTS & facts.keys():
        value = facts[name]
        if name.endswith("outputs"):
            if type(value) is not int or not 0 <= value <= len(selected):
                raise ValueError("native_count_differs")
        elif type(value) is not bool:
            raise ValueError("native_boolean_fact_differs")
        kept[name] = value
    return {"schema": "eva.automedbench-native-track-result.v1", "status": "scored", "track": track,
            "selected_case_count": len(selected), "full_public_subset": len(selected) == COUNTS[track],
            "task_score_0_1": float(score), "native_facts": kept,
            "private_values_exported": False, "network_disabled": True, "agent_judged": False}


def main():
    socket.socket = NoNetworkSocket
    socket.create_connection = socket.getaddrinfo = deny
    try:
        raw = sys.stdin.buffer.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("request_too_large")
        request = json.loads(raw)
        if set(request) != {"release_root", "submission_root", "track", "selected_case_ids",
                            "conversation_path", "source_commitments", "full_public_subset"}:
            raise ValueError("request_shape")
        track, selected = request["track"], request["selected_case_ids"]
        if (track not in COUNTS or not isinstance(selected, list) or not selected
                or selected != sorted(set(selected)) or len(selected) > COUNTS[track]
                or any(not isinstance(case, str) or not case or any(c in case for c in "/\\") for case in selected)
                or type(request["full_public_subset"]) is not bool
                or request["full_public_subset"] != (len(selected) == COUNTS[track])):
            raise ValueError("track_subset_identity")
        release = Path(request["release_root"]).resolve(strict=True)
        submission = Path(request["submission_root"]).resolve(strict=True)
        sys.path.insert(0, str(release))
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            from blake3 import blake3
            commitments = request["source_commitments"]
            if not isinstance(commitments, dict) or not commitments:
                raise ValueError("source_commitments_missing")
            for relative, expected in commitments.items():
                path = (release / relative).resolve(strict=True)
                if not path.is_relative_to(release) or not path.is_file():
                    raise ValueError("source_path_differs")
                digest = blake3()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise ValueError("source_commitment_differs")
            from automedbench_release.raw_track_worker import TRACK_LAYOUTS, _SCORERS, _track_modules
            from automedbench_release.tasks import TASK_BY_TRACK
            import yaml
            task, layout = TASK_BY_TRACK[track], TRACK_LAYOUTS[track]
            evaluation = release / layout["eval_dir"]
            config = yaml.safe_load((evaluation / layout["config"]).read_bytes())
            if config["task_id"] != task.task_id or task.expected_public_cases != COUNTS[track]:
                raise ValueError("native_task_contract_differs")
            public = task.data_path(release) / "public"
            all_cases = sorted(path.name for path in public.iterdir() if path.is_dir() and (path / task.public_marker).is_file())
            if len(all_cases) != COUNTS[track] or not set(selected) <= set(all_cases):
                raise ValueError("native_public_subset_differs")
            if request["full_public_subset"] and selected != all_cases:
                raise ValueError("native_full_selection_differs")
            kwargs = {"output_dir": submission, "public_dir": public,
                      "private_dir": task.data_path(release) / "private", "selected": selected, "config": config}
            if track == "vqa":
                conversation = Path(request["conversation_path"]).resolve(strict=True)
                if not conversation.is_file() or config.get("answer_mode") != "multiple_choice":
                    raise ValueError("native_vqa_evidence_missing")
                kwargs["conversation_path"] = conversation
            with _track_modules(evaluation, tuple(layout["modules"])) as modules:
                score, facts, _private_result = _SCORERS[track](modules=modules, **kwargs)
            result = result_projection(track, selected, score, facts)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except Exception:
        print('{"status":"error","error_code":"native_track_worker_failed","reward":null}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
