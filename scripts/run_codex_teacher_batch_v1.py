#!/usr/bin/env python3
"""Plan or execute durable one-rollout/model/sandbox teacher batches."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import shlex
import sqlite3
import stat
from typing import Any

from eva_agent.admission.receipts import SignedEnvelope, verify_signed_envelope
from eva_agent.codex_providers import load_codex_provider_routes
from eva_agent.training import TEACHER_ROUTES, TeacherCheckpoint, iter_bulk_sandboxes, run_batch
from eva_agent.training.persistent_teacher import PersistentCodexTeacherPool
from eva_agent.training.s1_training_successors import (
    TrainingS1SuccessorContextPool,
    load_training_successor_records,
)
from eva_agent.training.teacher_batch import run_persistent_batch


ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run", "status"))
    parser.add_argument("--bulk-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, action="append", default=[])
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--max-sandboxes", type=int)
    parser.add_argument("--skip-sandboxes", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--minimum-sft-score", type=float, default=0.8)
    parser.add_argument(
        "--stage",
        action="append",
        choices=("S1", "S2", "S3", "S4", "S5", "E2E"),
        help=(
            "Schedule only this exact stage; repeat for multiple stages. "
            "Skip/max are applied after this filter."
        ),
    )
    parser.add_argument("--worker-command")
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--app-server-shards", type=int, default=8)
    parser.add_argument("--execution-catalog", type=Path)
    parser.add_argument("--trust-store", type=Path)
    parser.add_argument(
        "--training-successor-bundle",
        type=Path,
        help=(
            "Use a verified training-only S1 successor bundle. This requires "
            "persistent --stage S1 plus the signed source execution catalog."
        ),
    )
    parser.add_argument(
        "--exclude-checkpoint",
        type=Path,
        action="append",
        default=[],
        help=(
            "Exclude every (sandbox, route) pair whose attempt_count is 1 in "
            "this teacher checkpoint; repeat for scattered historical waves."
        ),
    )
    parser.add_argument(
        "--route",
        action="append",
        choices=TEACHER_ROUTES,
        help="Schedule only this exact teacher route; repeat for multiple routes.",
    )
    return parser


def _load_attempted_pairs(
    checkpoint_paths: Sequence[Path],
) -> frozenset[tuple[str, str]]:
    """Read route-scoped one-shot history without mutating source checkpoints."""

    resolved_paths: list[Path] = []
    for candidate in checkpoint_paths:
        path = Path(candidate)
        if path.is_symlink():
            raise SystemExit("excluded teacher checkpoint must not be a symlink")
        try:
            path = path.resolve(strict=True)
            metadata = path.stat()
        except OSError:
            raise SystemExit("excluded teacher checkpoint is missing") from None
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SystemExit("excluded teacher checkpoint topology differs")
        resolved_paths.append(path)
    if len(resolved_paths) != len(set(resolved_paths)):
        raise SystemExit("excluded teacher checkpoint is duplicated")

    attempted: set[tuple[str, str]] = set()
    for path in resolved_paths:
        try:
            with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as database:
                columns = {
                    str(row[1]) for row in database.execute("PRAGMA table_info(tasks)")
                }
                if not {"sandbox_id", "route_id", "attempt_count"} <= columns:
                    raise SystemExit("excluded teacher checkpoint schema differs")
                records = database.execute(
                    "SELECT sandbox_id,route_id,attempt_count FROM tasks"
                ).fetchall()
        except sqlite3.Error:
            raise SystemExit("excluded teacher checkpoint could not be read") from None
        for sandbox_id, route_id, attempt_count in records:
            if (
                not isinstance(sandbox_id, str)
                or not sandbox_id
                or route_id not in TEACHER_ROUTES
                or type(attempt_count) is not int
                or attempt_count not in (0, 1)
            ):
                raise SystemExit("excluded teacher checkpoint task differs")
            if attempt_count == 1:
                attempted.add((sandbox_id, route_id))
    return frozenset(attempted)


def _select_unused_task_pairs(
    rows: Sequence[Mapping[str, Any]],
    routes: Sequence[str],
    attempted: frozenset[tuple[str, str]],
) -> tuple[tuple[Mapping[str, Any], str], ...]:
    """Preserve bulk order while independently filtering every model route."""

    return tuple(
        (row, route)
        for row in rows
        for route in routes
        if (row["sandbox_id"], route) not in attempted
    )


def _schedule_task_pairs(
    checkpoint: TeacherCheckpoint,
    task_pairs: Sequence[tuple[Mapping[str, Any], str]],
) -> int:
    return sum(
        checkpoint.schedule((row,), (route,)) for row, route in task_pairs
    )


def main() -> int:
    args = _parser().parse_args()
    if args.persistent and not 1 <= args.app_server_shards <= args.workers <= 256:
        raise SystemExit(
            "persistent topology requires 1 <= app-server-shards <= workers <= 256"
        )
    checkpoint = TeacherCheckpoint(args.output_root / "checkpoint.sqlite3")
    if args.mode == "status":
        print(json.dumps({"schema": "eva.codex-teacher-batch-status.v1", "counts": checkpoint.counts()}))
        return 0
    requested_routes = tuple(args.route or TEACHER_ROUTES)
    if len(requested_routes) != len(set(requested_routes)):
        raise SystemExit("teacher route request is duplicated")
    routes = load_codex_provider_routes(
        env_files=args.env_file,
        registry_path=args.registry,
        route_ids=requested_routes,
    )
    missing = sorted(set(requested_routes) - set(routes))
    if not routes:
        raise SystemExit("no exact teacher model route is configured")
    if args.training_successor_bundle is not None and (
        not args.persistent
        or args.execution_catalog is None
        or tuple(args.stage or ()) != ("S1",)
    ):
        raise SystemExit(
            "training successor rollout requires persistent --stage S1 "
            "and --execution-catalog"
        )
    eligible = None
    if args.execution_catalog is not None:
        if args.trust_store is None:
            raise SystemExit("--trust-store is required with --execution-catalog")
        envelope = SignedEnvelope.from_document(
            json.loads(args.execution_catalog.read_text(encoding="utf-8"))
        )
        verify_signed_envelope(envelope, trust_store_path=args.trust_store)
        catalog_rows = envelope.payload.get("rows")
        if not isinstance(catalog_rows, (list, tuple)):
            raise SystemExit("execution catalog row inventory differs")
        eligible = {
            row["candidate_id"]
            for row in catalog_rows
            if isinstance(row, Mapping)
            and row.get("execution_status") == "executable_legacy"
            and row.get("selection_tier") == "primary"
        }
    rows = []
    eligible_seen = 0
    requested_stages = tuple(
        args.stage or ("S1", "S2", "S3", "S4", "S5", "E2E")
    )
    if len(requested_stages) != len(set(requested_stages)):
        raise SystemExit("teacher stage request is duplicated")
    if args.skip_sandboxes < 0:
        raise SystemExit("--skip-sandboxes must be non-negative")
    source_rows = (
        load_training_successor_records(
            args.training_successor_bundle,
            bulk_root=args.bulk_root,
            catalog_path=args.execution_catalog,
        )
        if args.training_successor_bundle is not None
        else iter_bulk_sandboxes(args.bulk_root)
    )
    for row in source_rows:
        if (
            eligible is not None
            and args.training_successor_bundle is None
            and row["candidate_id"] not in eligible
        ):
            continue
        if row["stage"] not in requested_stages:
            continue
        if eligible_seen < args.skip_sandboxes:
            eligible_seen += 1
            continue
        rows.append(row)
        if args.max_sandboxes is not None and len(rows) >= args.max_sandboxes:
            break
    configured = tuple(route for route in requested_routes if route in routes)
    attempted = _load_attempted_pairs(args.exclude_checkpoint)
    task_pairs = _select_unused_task_pairs(rows, configured, attempted)
    candidate_pairs = {
        (row["sandbox_id"], route) for row in rows for route in configured
    }
    excluded_pairs = candidate_pairs & attempted
    selected_sandboxes = {row["sandbox_id"] for row, _route in task_pairs}
    selected_by_route = Counter(route for _row, route in task_pairs)
    inserted = _schedule_task_pairs(checkpoint, task_pairs)
    if args.mode == "preflight":
        print(
            json.dumps(
                {
                    "schema": "eva.codex-teacher-batch-preflight.v1",
                    "provider_calls": 0,
                    "sandbox_count": len(rows),
                    "selected_sandbox_count": len(selected_sandboxes),
                    "skipped_sandboxes": args.skip_sandboxes,
                    "route_count": len(configured),
                    "candidate_task_count": len(candidate_pairs),
                    "excluded_checkpoint_count": len(args.exclude_checkpoint),
                    "excluded_attempted_task_count": len(excluded_pairs),
                    "selected_task_count": len(task_pairs),
                    "selected_task_counts_by_route": {
                        route: selected_by_route[route] for route in configured
                    },
                    "inserted_tasks": inserted,
                    "routes": list(configured),
                    "missing_optional_routes": missing,
                    "counts": checkpoint.counts(),
                    "retry_count": 0,
                    "execution_mode": (
                        "persistent_pool" if args.persistent else "subprocess_per_task"
                    ),
                    "app_server_shards": (
                        args.app_server_shards if args.persistent else None
                    ),
                    "campaign_resource_opens_per_process": (
                        1 if args.persistent else len(rows) * len(configured)
                    ),
                },
                separators=(",", ":"),
            )
        )
        return 0
    if args.persistent and args.worker_command:
        raise SystemExit("--persistent and --worker-command are mutually exclusive")
    if not args.persistent and not args.worker_command:
        raise SystemExit("run requires --worker-command; no implicit provider boundary exists")
    checkpoint.fail_interrupted()
    if not checkpoint.queued():
        print(
            json.dumps(
                {
                    "schema": "eva.codex-teacher-batch-result.v1",
                    "counts": checkpoint.counts(),
                    "retry_count": 0,
                },
                separators=(",", ":"),
            )
        )
        return 0
    if args.persistent:
        context_pool = (
            TrainingS1SuccessorContextPool()
            if args.training_successor_bundle is not None
            else None
        )
        with PersistentCodexTeacherPool(
            rows=rows,
            routes={route_id: routes[route_id] for route_id in configured},
            output_root=args.output_root,
            worker_width=args.workers,
            app_server_shards=args.app_server_shards,
            context_pool=context_pool,
            launch_cwd=ROOT,
        ) as pool:
            result = run_persistent_batch(
                checkpoint=checkpoint,
                output_root=args.output_root,
                worker=pool.run,
                workers=args.workers,
                minimum_sft_score=args.minimum_sft_score,
                limit=None,
            )
        print(json.dumps(result, separators=(",", ":")))
        return 0
    result = run_batch(
        checkpoint=checkpoint,
        output_root=args.output_root,
        worker_command=shlex.split(args.worker_command),
        workers=args.workers,
        minimum_sft_score=args.minimum_sft_score,
        bulk_root=args.bulk_root,
        limit=None,
    )
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
