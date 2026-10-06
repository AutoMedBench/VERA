"""Seven-track preparation and persistent coding-workflow runner (version 1)."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if len(sys.argv) > 1 and sys.argv[1] == "serve":
    selected_harness = os.environ.get("EVA_HARNESS_ROOT")
    os.environ.clear()
    os.environ.update({"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"})
    if selected_harness:
        os.environ["EVA_HARNESS_ROOT"] = selected_harness
sys.path.insert(0, str(ROOT))
from training.harness_source import activate_harness
activate_harness(ROOT)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("prepare")
    prepare.add_argument("--cds-receipt", type=Path, required=True)
    prepare.add_argument("--missing4-receipt", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    server = modes.add_parser("serve")
    server.add_argument("--workspace", type=Path, required=True)
    server.add_argument("--audit-root", type=Path, required=True)
    for selected in (server, modes.add_parser("run")):
        selected.add_argument("--runtime-manifest", type=Path, required=True)
        selected.add_argument("--public-python", type=Path, required=True)
        selected.add_argument("--image", default="eva-automedbench-cpu:20260910-v1")
        if selected is not server:
            selected.add_argument("--run-root", type=Path, required=True)
            selected.add_argument("--server-canary", type=Path, required=True)
            selected.add_argument("--server-identity", type=Path, required=True)
            selected.add_argument("--codex-bin", type=Path, required=True)
            selected.add_argument("--tracks", nargs="+", default=["classification"])
            selected.add_argument("--turn-timeout", type=int, default=900)
            selected.add_argument("--track-timeout", type=int, choices=[3600], default=None,
                help="Total admitted-track wall clock across S1-S5; replaces per-phase turn timeout")
            selected.add_argument("--workers", type=int, choices=range(1, 5), default=4)
            selected.add_argument("--context-length", type=int, default=32768)
            selected.add_argument("--auto-compact-token-limit", type=int, default=12288)
            selected.add_argument("--output-tokens", type=int, default=4096)
            selected.add_argument("--tool-output-token-limit", type=int, default=None)
            selected.add_argument("--skill-selection", type=Path, default=None)
            selected.add_argument("--skill-selection-blake3", default=None)
            selected.add_argument("--skill-catalog-id", default=None)
            selected.add_argument("--adapter-capacity-wait-seconds", type=float, default=0.0)
            profiles = selected.add_mutually_exclusive_group()
            profiles.add_argument("--memory-profile-v12", action="store_true")
            profiles.add_argument("--supra-profile", action="store_true")
            selected.add_argument("--endpoint", default="http://127.0.0.1:30910/v1")
    args = parser.parse_args()
    if args.mode == "prepare":
        from training.automedbench_lite.track_adapter import TrackRelease, prepare_track_run, BY_TRACK
        cds, other = TrackRelease(args.cds_receipt), TrackRelease(args.missing4_receipt)
        print(prepare_track_run({name: cds if name in {"classification", "detection", "segmentation"} else other
                                for name in BY_TRACK}, args.output_root), flush=True)
    elif args.mode == "serve":
        from training.automedbench_lite.track_tools import TrackTools, serve
        from training.automedbench_lite.skill_surface import VerifiedEvaluationSkills
        serve(TrackTools(workspace=args.workspace, audit_root=args.audit_root, image=args.image,
            model_manifest=args.runtime_manifest, public_python=args.public_python,
            skills=VerifiedEvaluationSkills(args.audit_root / "skill-materialization")))
    else:
        import asyncio
        from training.automedbench_lite.track_actor import run_tracks
        asyncio.run(run_tracks(args, memory_profile=args.memory_profile_v12, supra_profile=args.supra_profile))
