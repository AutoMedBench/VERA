"""Synthetic CPU-only subprocess for supervisor tests; NEVER a training receipt."""
import argparse
import json
from pathlib import Path
import time

from blake3 import blake3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("context", type=Path)
    parser.add_argument("--mode", default="normal")
    parser.add_argument("--sleep", type=float, default=0)
    args = parser.parse_args()
    time.sleep(args.sleep)
    context = json.loads(args.context.read_text())
    root = args.context.parent
    if context["phase"] != "train":
        value = {"synthetic_test_fixture": True, "model_path": context["model_path"],
                 "valid": not (args.mode == "bad_evaluation" and context["phase"] == "evaluation")}
        (root / "evaluation-index.json").write_text(json.dumps(value))
        return 0
    train = root / "training"
    train.mkdir()
    rows = [{"metadata": {"stage": context["s_target"], "judge_backend": "native_astra"}}]
    data = train / "synthetic-data.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    n = context["remaining_updates"]
    events = [{"event": "optimizer_step", "rollout_id": i, "step_id": 0, "successful_update": True,
               "gradient_norm": 0.0 if args.mode == "zero_learning" else 1.0,
               "sampled_changed_values": 0 if args.mode == "zero_learning" else 2,
               "trainable_parameters": 8953803264, "synthetic_test_fixture": True} for i in range(n)]
    if args.mode == "duplicates":
        events += [dict(events[-1]) for _ in range(20)]
    (train / "optimizer-steps.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")
    argv = ["SYNTHETIC-CPU-FIXTURE-NOT-TRAINING", "--use-rollout-logprobs", "--num-steps-per-rollout", "1",
            "--num-rollout", str(n), "--hf-checkpoint", context["architecture_model_path"],
            "--load", context["checkpoint_root"], "--save", str(train / "checkpoints"), "--prompt-data", str(data)]
    partial = args.mode == "partial_checkpoint" and n == 50
    (train / "run-receipt.json").write_text(json.dumps({"stage": "grpo", "judge_backend": "native_astra",
        "argv": argv, "training_data_blake3": blake3(data.read_bytes()).hexdigest(),
        "status": "failed" if partial else "complete", "synthetic_test_fixture": True}))
    if args.mode != "missing_checkpoint":
        (train / "checkpoint-fixture.json").write_text(json.dumps({"synthetic_test_fixture": True,
            "durable_updates": n - 1 if partial else n, "model_path": str(train / "fake-hf"),
            "checkpoint_root": str(train / "fake-dcp")}))
    return 7 if partial else 0


if __name__ == "__main__":
    raise SystemExit(main())
