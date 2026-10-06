"""Explicit evaluation-only CLI and scrubbed public MCP subprocess entrypoint."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if len(sys.argv) > 1 and sys.argv[1] == "serve":
    os.environ.clear()
    os.environ.update({"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
                       "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
                       "HIP_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"})
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    server = modes.add_parser("serve")
    server.add_argument("--workspace", type=Path, required=True)
    server.add_argument("--audit-root", type=Path, required=True)
    server.add_argument("--image", required=True)
    run = modes.add_parser("run")
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--assets-root", type=Path, required=True)
    run.add_argument("--server-canary", type=Path, required=True)
    run.add_argument("--server-identity", type=Path, required=True)
    run.add_argument("--public-python", type=Path, required=True)
    run.add_argument("--image", default="eva-automedbench-cpu:20260910-v1")
    run.add_argument("--turn-timeout", type=int, default=600)
    args = parser.parse_args()
    if args.mode == "serve":
        from training.automedbench_lite.public_tools import PublicTools, serve
        from training.automedbench_lite.skill_surface import VerifiedEvaluationSkills
        skills = VerifiedEvaluationSkills(args.audit_root / "skill-materialization")
        serve(PublicTools(workspace=args.workspace, audit_root=args.audit_root, image=args.image, skills=skills))
    else:
        import asyncio
        from training.automedbench_lite.actor import run_evaluation
        asyncio.run(run_evaluation(args))
