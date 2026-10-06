"""Command-line entry points for the verifiable EVA-Agent data pipeline."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from eva_agent.rubrics import (
    RubricValidationError,
    canonical_json_bytes,
    load_and_compile_registry,
)


TARGETS = {
    "train": 6_000,
    "development": 0,
    "sealed_evaluation": 0,
}
TOTAL_TARGET = sum(TARGETS.values())
SCHEDULE_TARGET = 9_000
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = Path("runs/campaign.sqlite3")
DEFAULT_PREMIUM_CONSTRUCTION_STATE = Path("runs/premium-construction-campaign-v1")
DEFAULT_BULK_RL_ROOT = Path("runs/bulk-rl-sandboxes.v1")
DEFAULT_BULK_RL_TRUST_STORE = (
    PROJECT_ROOT.parent / "rlevo-med-research/config/host-trust-store.v1.json"
)


class CliError(ValueError):
    """A command could not establish the requested verified state."""


@dataclass(frozen=True)
class ProgressSnapshot:
    counts: Mapping[str, int]
    scheduled: int
    queued: int
    active: int
    rejected: int
    infrastructure_quarantine: int
    reserve_unused: int
    ledger_exists: bool
    targets: Mapping[str, int] = field(default_factory=lambda: dict(TARGETS))

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def target(self) -> int:
        return sum(self.targets.values())

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": "eva.progress-snapshot.v1",
            "admitted_verified": self.total,
            "target": self.target,
            "counts": dict(self.counts),
            "targets": dict(self.targets),
            "scheduled": self.scheduled,
            "schedule_target": SCHEDULE_TARGET,
            "queued": self.queued,
            "active": self.active,
            "rejected": self.rejected,
            "infrastructure_quarantine": self.infrastructure_quarantine,
            "reserve_unused": self.reserve_unused,
            "ledger_exists": self.ledger_exists,
        }


def read_progress(ledger: Path) -> ProgressSnapshot:
    """Read the transactional campaign ledger as the sole progress authority."""

    counts = {split: 0 for split in TARGETS}
    if not ledger.exists():
        return ProgressSnapshot(
            counts=counts,
            targets=TARGETS,
            scheduled=0,
            queued=0,
            active=0,
            rejected=0,
            infrastructure_quarantine=0,
            reserve_unused=0,
            ledger_exists=False,
        )
    if ledger.is_symlink() or not ledger.is_file():
        raise CliError("admission ledger must be a regular non-symlink file")
    try:
        plan, targets = _ledger_plan_and_targets(ledger)
        from eva_agent.campaign import CampaignLedger
        current = CampaignLedger(ledger, plan=plan).progress()
    except (OSError, ValueError) as exc:
        raise CliError(f"cannot verify campaign ledger: {ledger}") from exc
    return _snapshot_from_campaign_progress(current, targets)


def _ledger_plan_and_targets(ledger: Path) -> tuple[Any, Mapping[str, int]]:
    from eva_agent.campaign import exact_6000_plan
    from eva_agent.campaign.split_plan_v3 import exact_5000_500_500_plan

    uri = f"{ledger.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='plan_blake3'"
        ).fetchone()
    if row is None:
        raise CliError("campaign ledger lacks its plan commitment")
    plans = (exact_6000_plan(), exact_5000_500_500_plan())
    matches = tuple(plan for plan in plans if plan.plan_blake3 == row[0])
    if len(matches) != 1:
        raise CliError("campaign ledger plan commitment is unsupported")
    plan = matches[0]
    return plan, dict(plan.target_by_split)


def _snapshot_from_campaign_progress(
    current: Any, targets: Mapping[str, int] = TARGETS
) -> ProgressSnapshot:
    counts = {
        "train": current.train,
        "development": current.development,
        "sealed_evaluation": current.sealed_evaluation,
    }
    total_target = sum(targets.values())
    if current.target != total_target or current.admitted != sum(counts.values()):
        raise CliError("campaign ledger totals differ from the exact 6,000 plan")
    if any(not 0 <= counts[split] <= targets[split] for split in targets):
        raise CliError("campaign ledger split count exceeds its signed quota")
    if (
        type(current.scheduled) is not int
        or not 0 <= current.scheduled <= SCHEDULE_TARGET
        or type(current.reserve_unused) is not int
        or not 0 <= current.reserve_unused <= current.scheduled
    ):
        raise CliError("campaign scheduled queue totals differ from the frozen 9,000 plan")
    return ProgressSnapshot(
        counts=counts,
        targets=dict(targets),
        scheduled=current.scheduled,
        queued=current.queued,
        active=current.active,
        rejected=current.rejected,
        infrastructure_quarantine=current.infrastructure_quarantine,
        reserve_unused=current.reserve_unused,
        ledger_exists=True,
    )


def _bar(value: int, target: int, width: int) -> str:
    filled = min(width, value * width // target)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _eta_text(remaining: int, admissions_per_hour: float | None) -> str:
    if admissions_per_hour is None:
        return "ETA: awaiting a measured admission rate (run the pilot first)"
    hours = remaining / admissions_per_hour
    if hours < 48:
        return f"ETA at {admissions_per_hour:,.1f}/hour: {hours:.1f} hours"
    return f"ETA at {admissions_per_hour:,.1f}/hour: {hours / 24:.1f} days"


def render_progress(
    snapshot: ProgressSnapshot,
    *,
    width: int = 30,
    admissions_per_hour: float | None = None,
) -> str:
    percent = 100.0 * snapshot.total / snapshot.target
    schedule_percent = 100.0 * snapshot.scheduled / SCHEDULE_TARGET
    source = (
        "campaign SQLite ledger"
        if snapshot.ledger_exists
        else "ledger absent; fail-closed zero"
    )
    def split_text(label: str, display: str) -> str:
        target = snapshot.targets[label]
        if target == 0:
            return f"{display} disabled (0 quota)"
        return f"{display} {snapshot.counts[label]:,}/{target:,}"
    return "\n".join(
        (
            f"{_bar(snapshot.total, snapshot.target, width)} "
            f"{snapshot.total:,}/{snapshot.target:,} signed+verified ({percent:.2f}%)",
            (
                f"{_bar(snapshot.scheduled, SCHEDULE_TARGET, width)} "
                f"{snapshot.scheduled:,}/{SCHEDULE_TARGET:,} frozen candidates scheduled "
                f"({schedule_percent:.2f}%; 6,000 primary + 3,000 reserve)"
            ),
            " | ".join(
                (
                    split_text("train", "train"),
                    split_text("development", "development"),
                    split_text("sealed_evaluation", "sealed evaluation"),
                )
            ),
            (
                f"source: {source}; queued={snapshot.queued}; active={snapshot.active}; "
                f"rejected={snapshot.rejected}; quarantine={snapshot.infrastructure_quarantine}; "
                f"reserve_unused={snapshot.reserve_unused}"
            ),
            _eta_text(snapshot.target - snapshot.total, admissions_per_hour),
        )
    )


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise CliError(f"refusing to replace existing output: {path}") from None
    path.chmod(0o444)


def _cmd_rubric_validate(args: argparse.Namespace) -> int:
    registry = load_and_compile_registry(args.source)
    result = {
        "status": "passed",
        "registry_id": registry.registry_id,
        "registry_version": registry.registry_version,
        "registry_blake3": registry.digest,
        "rubric_tables": len(registry.rubrics),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def _cmd_rubric_compile(args: argparse.Namespace) -> int:
    registry = load_and_compile_registry(args.source)
    payload = canonical_json_bytes(registry.to_document())
    if args.output is None:
        sys.stdout.buffer.write(payload)
    else:
        _write_new(args.output, payload)
        print(
            json.dumps(
                {
                    "status": "compiled",
                    "output": args.output.as_posix(),
                    "registry_blake3": registry.digest,
                    "rubric_tables": len(registry.rubrics),
                },
                sort_keys=True,
            )
        )
    return 0


def _cmd_progress(args: argparse.Namespace) -> int:
    from eva_agent.progress_dashboard import (
        read_bulk_rl_sandbox_progress,
        read_frontier_sft_training_progress,
        read_live_utilization,
        read_premium_construction_progress,
        read_sft_training_progress,
        read_teacher_batch_progress,
        render_bulk_rl_sandbox_progress,
        render_frontier_sft_training_progress,
        render_live_utilization,
        render_premium_construction_progress,
        render_sft_training_progress,
        render_sft_total_training_records,
        render_teacher_batch_progress,
    )

    campaign_ledger: Any | None = None
    while True:
        if campaign_ledger is None and args.ledger.exists():
            if args.ledger.is_symlink() or not args.ledger.is_file():
                raise CliError("admission ledger must be a regular non-symlink file")
            try:
                from eva_agent.campaign import CampaignLedger

                plan, active_targets = _ledger_plan_and_targets(args.ledger)
                campaign_ledger = CampaignLedger(args.ledger, plan=plan)
            except (OSError, ValueError) as exc:
                raise CliError(f"cannot verify campaign ledger: {args.ledger}") from exc
        snapshot = (
            _snapshot_from_campaign_progress(campaign_ledger.progress(), active_targets)
            if campaign_ledger is not None
            else ProgressSnapshot(
                counts={split: 0 for split in TARGETS},
                targets=TARGETS,
                scheduled=0,
                queued=0,
                active=0,
                rejected=0,
                infrastructure_quarantine=0,
                reserve_unused=0,
                ledger_exists=False,
            )
        )
        rl_sandboxes = read_bulk_rl_sandbox_progress(
            args.rl_root,
            trust_store_path=args.rl_trust_store,
        )
        sft = read_sft_training_progress(args.sft_output)
        frontier_sft = read_frontier_sft_training_progress(args.frontier_sft_output)
        teacher = read_teacher_batch_progress(args.teacher_output)
        premium = (
            None
            if args.no_premium
            else read_premium_construction_progress(args.premium_state)
        )
        live = (
            read_live_utilization(
                ledger=args.ledger,
                project_root=PROJECT_ROOT,
                runtime_worktrees=args.runtime_worktree,
            )
            if args.live
            else None
        )
        if args.json:
            document = snapshot.to_document()
            document["admissions_per_hour"] = args.rate
            document["eta_hours"] = (
                (snapshot.target - snapshot.total) / args.rate
                if args.rate is not None
                else None
            )
            document["dashboard_schema"] = "eva.progress-dashboard.v3"
            document["rl_sandboxes"] = rl_sandboxes.to_document()
            document["sft"] = sft.to_document() if args.sft_output is not None else None
            document["frontier_sft"] = (
                frontier_sft.to_document()
                if args.frontier_sft_output is not None
                else None
            )
            document["sft_training_records_total"] = (
                sft.materialized_slices + frontier_sft.materialized_slices
                if args.sft_output is not None or args.frontier_sft_output is not None
                else None
            )
            document["teacher_batch"] = (
                teacher.to_document() if args.teacher_output is not None else None
            )
            document["rollout_admission"] = (
                None if args.no_admission else snapshot.to_document()
            )
            document["premium_construction"] = (
                premium.to_document() if premium is not None else None
            )
            document["live_local_utilization"] = (
                live.to_document() if live is not None else None
            )
            rendered = json.dumps(document, sort_keys=True)
        else:
            lines = [
                render_bulk_rl_sandbox_progress(rl_sandboxes, width=args.width)
            ]
            if args.sft_output is not None:
                lines.append(render_sft_training_progress(sft))
            if args.frontier_sft_output is not None:
                lines.append(render_frontier_sft_training_progress(frontier_sft))
            if args.sft_output is not None or args.frontier_sft_output is not None:
                lines.append(render_sft_total_training_records(sft, frontier_sft))
            if args.teacher_output is not None:
                lines.append(render_teacher_batch_progress(teacher))
            if not args.no_admission:
                lines.extend(
                    (
                        "rollout admission (selection telemetry; not the RL count):",
                        render_progress(
                            snapshot,
                            width=args.width,
                            admissions_per_hour=args.rate,
                        ),
                    )
                )
            if premium is not None:
                lines.append(
                    render_premium_construction_progress(premium, width=args.width)
                )
            if live is not None:
                lines.append(
                    render_live_utilization(
                        live,
                        rollout_active=snapshot.active,
                        construction_active=(premium.active if premium is not None else 0),
                    )
                )
            rendered = "\n".join(lines)
        if args.watch and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(rendered, flush=True)
        if not args.watch:
            return 0
        time.sleep(args.interval)


class _DemoResponses:
    """Deterministic SDK-shaped model used only by the no-provider demo."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> Any:
        self.requests.append(request)
        if len(self.requests) == 1:
            output = [
                {
                    "type": "function_call",
                    "call_id": "demo-call-a",
                    "name": "read_file",
                    "arguments": '{"path":"source-a.txt"}',
                },
                {
                    "type": "function_call",
                    "call_id": "demo-call-b",
                    "name": "read_file",
                    "arguments": '{"path":"source-b.txt"}',
                },
            ]
            return SimpleNamespace(
                id="demo-response-1",
                model="eva-no-provider-demo",
                output=output,
                output_text="",
                usage={"input_tokens": 0, "output_tokens": 0},
            )
        return SimpleNamespace(
            id="demo-response-2",
            model="eva-no-provider-demo",
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "demo complete"}],
                }
            ],
            output_text="demo complete",
            usage={"input_tokens": 0, "output_tokens": 0},
        )


async def _run_demo() -> dict[str, Any]:
    from eva_agent.harness import (
        EvaMedHarness,
        LocalVenvWorkspace,
        OpenAIResponsesModel,
        ToolDefinition,
        ToolRegistry,
    )

    with tempfile.TemporaryDirectory(prefix="eva-agent-demo-") as temporary:
        root = Path(temporary)
        (root / "source-a.txt").write_text("provenance A", encoding="utf-8")
        (root / "source-b.txt").write_text("provenance B", encoding="utf-8")
        runtime = LocalVenvWorkspace(workspace_root=root)
        local_read = next(
            definition
            for definition in runtime.tool_definitions()
            if definition.name == "read_file"
        )
        active = 0
        both_started = asyncio.Event()

        async def observable_parallel_read(arguments: Mapping[str, Any]) -> Any:
            nonlocal active
            active += 1
            if active == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=2)
            try:
                return await local_read.handler(arguments)
            finally:
                active -= 1

        tools = ToolRegistry(
            (
                ToolDefinition(
                    name=local_read.name,
                    description=local_read.description,
                    parameters=local_read.parameters,
                    handler=observable_parallel_read,
                    allowed_stages=local_read.allowed_stages,
                    parallel_safe=True,
                ),
            )
        )
        responses = _DemoResponses()
        model = OpenAIResponsesModel(
            model_id="eva-no-provider-demo",
            client=SimpleNamespace(responses=responses),
        )
        trajectory = await EvaMedHarness(
            model=model,
            tools=tools,
            max_parallel_tools=8,
        ).run(
            initial_input=[
                {
                    "role": "user",
                    "content": "Read both projected evidence files in one tool-call group.",
                }
            ],
            stage="S2",
            instructions="Exercise local evidence tools without an HTTP tool service.",
        )
        observations = sum(len(group.observations) for group in trajectory.tool_groups)
        return {
            "schema": "eva.no-provider-demo-result.v1",
            "status": "passed" if trajectory.terminal and observations == 2 else "failed",
            "provider_calls": 0,
            "sdk_requests_to_fake_client": len(responses.requests),
            "tool_calls": observations,
            "max_parallelism_observed": trajectory.max_parallelism_observed,
            "trajectory_blake3": trajectory.trajectory_blake3,
        }


def _cmd_demo_local(_: argparse.Namespace) -> int:
    result = asyncio.run(_run_demo())
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


def _cmd_bootstrap_venv(args: argparse.Namespace) -> int:
    from eva_agent.harness import bootstrap_venv

    bootstrap_venv(args.path, python=args.python)
    print(json.dumps({"status": "ready", "venv": args.path.resolve().as_posix()}, sort_keys=True))
    return 0


def _cmd_verify_result(args: argparse.Namespace) -> int:
    try:
        from eva_agent.pipeline import (
            ImmutableArtifactStore,
            load_pipeline_result,
            verify_result_document,
        )
        from eva_agent.pipeline.digests import canonical_value
    except ImportError as exc:
        raise CliError(
            "serialized pipeline-result verification is unavailable in this build"
        ) from exc
    if (
        not args.artifact_root.exists()
        or args.artifact_root.is_symlink()
        or not args.artifact_root.is_dir()
    ):
        raise CliError("artifact root must be an existing real directory")
    result = load_pipeline_result(args.result)
    if len(result.sandbox_manifests) != 1:
        raise CliError("pipeline result must contain exactly one sandbox manifest")
    manifest = result.sandbox_manifests[0]
    registry = load_and_compile_registry(args.registry)
    rubric = registry.resolve(manifest.domain, manifest.stage.value)
    report = verify_result_document(
        args.result,
        artifact_store=ImmutableArtifactStore(args.artifact_root),
        rubric=rubric,
    )
    document = canonical_value(report)
    print(json.dumps(document, ensure_ascii=False, sort_keys=True))
    passed = document.get("valid") if isinstance(document, Mapping) else False
    return 0 if passed else 1


def _cmd_sft_import_legacy(args: argparse.Namespace) -> int:
    from eva_agent.pipeline import (
        RandomUUIDFactory,
        build_legacy_teacher_sft_dataset,
        verify_legacy_sft_dataset,
    )
    from eva_agent.pipeline.digests import canonical_value

    build = build_legacy_teacher_sft_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        trust_store_path=args.trust_store,
        eligible_model_ids=tuple(args.eligible_model),
        id_factory=RandomUUIDFactory(),
        minimum_score=args.minimum_score,
        shard_size=args.shard_size,
        verification_workers=args.verification_workers,
    )
    report = verify_legacy_sft_dataset(args.output_root, build.dataset_id)
    document = {
        "schema": "eva.legacy-teacher-sft-import-result.v1",
        "dataset_id": build.dataset_id,
        "dataset_root": build.dataset_root.resolve().as_posix(),
        "source_count": build.source_count,
        "slice_count": build.slice_count,
        "shard_count": build.shard_count,
        "manifest_blake3": build.manifest_blake3,
        "verification": canonical_value(report),
    }
    print(json.dumps(document, ensure_ascii=False, sort_keys=True))
    return 0 if report["valid"] else 1


def _cmd_sft_verify_legacy(args: argparse.Namespace) -> int:
    from eva_agent.pipeline import verify_legacy_sft_dataset
    from eva_agent.pipeline.digests import canonical_value

    report = verify_legacy_sft_dataset(args.output_root, args.dataset_id)
    print(json.dumps(canonical_value(report), ensure_ascii=False, sort_keys=True))
    return 0 if report["valid"] else 1


def _path(value: str) -> Path:
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eva-agent",
        description="Verifiable medical-research data-pipeline and EvaMed harness utilities.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    rubric = commands.add_parser("rubric", help="validate or compile the shared rubric registry")
    rubric_commands = rubric.add_subparsers(dest="rubric_command", required=True)
    validate = rubric_commands.add_parser("validate", help="validate and compile without writing")
    validate.add_argument("source", type=_path)
    validate.set_defaults(handler=_cmd_rubric_validate)
    compile_command = rubric_commands.add_parser("compile", help="write canonical compiled JSON")
    compile_command.add_argument("source", type=_path)
    compile_command.add_argument("--output", "-o", type=_path)
    compile_command.set_defaults(handler=_cmd_rubric_compile)

    progress = commands.add_parser(
        "progress", help="show RL materialization, SFT, and rollout-selection progress"
    )
    progress.add_argument(
        "--rl-root",
        type=_path,
        default=DEFAULT_BULK_RL_ROOT,
        help="signed bulk-RL dataset root (manifest plus shards)",
    )
    progress.add_argument(
        "--rl-trust-store",
        type=_path,
        default=DEFAULT_BULK_RL_TRUST_STORE,
        help="Ed25519 public trust store for the compact RL manifest",
    )
    progress.add_argument(
        "--sft-output",
        type=_path,
        help="optional teacher-SFT dataset directory or its manifest.json",
    )
    progress.add_argument(
        "--frontier-sft-output",
        type=_path,
        action="append",
        help=(
            "optional execution-verified stage-prefix SFT dataset directory or "
            "manifest.json; repeat to aggregate distinct datasets"
        ),
    )
    progress.add_argument(
        "--teacher-output",
        type=_path,
        action="append",
        help=(
            "optional teacher-batch output root or checkpoint.sqlite3; repeat "
            "to aggregate canaries and waves"
        ),
    )
    progress.add_argument("--ledger", type=_path, default=DEFAULT_LEDGER)
    progress.add_argument(
        "--premium-state",
        type=_path,
        default=DEFAULT_PREMIUM_CONSTRUCTION_STATE,
        help="read-only append-only premium construction state root",
    )
    progress.add_argument(
        "--no-premium",
        action="store_true",
        help="omit the premium-construction queue line",
    )
    progress.add_argument(
        "--no-admission",
        action="store_true",
        help="omit legacy admission telemetry from the training-data dashboard",
    )
    progress.add_argument(
        "--live",
        action="store_true",
        help="add credential-free local launcher, Codex process, and active-stage counts",
    )
    progress.add_argument(
        "--runtime-worktree",
        type=_path,
        action="append",
        default=[],
        help="additional explicit EVA runtime worktree for local process counting",
    )
    progress.add_argument("--watch", action="store_true", help="refresh until interrupted")
    progress.add_argument("--interval", type=float, default=2.0)
    progress.add_argument("--width", type=int, default=30)
    progress.add_argument(
        "--rate",
        type=float,
        help="measured signed+verified admissions/hour for an evidence-based ETA",
    )
    progress.add_argument("--json", action="store_true")
    progress.set_defaults(handler=_cmd_progress)

    verify = commands.add_parser("verify-result", help="reopen a serialized pipeline result")
    verify.add_argument("result", type=_path)
    verify.add_argument("--artifact-root", type=_path, required=True)
    verify.add_argument("--registry", type=_path, required=True)
    verify.set_defaults(handler=_cmd_verify_result)

    sft = commands.add_parser(
        "sft", help="import and verify high-performance trajectory slices"
    )
    sft_commands = sft.add_subparsers(dest="sft_command", required=True)
    sft_import = sft_commands.add_parser(
        "import-legacy", help="verify legacy teacher rollouts and stream SFT shards"
    )
    sft_import.add_argument("--source-root", type=_path, required=True)
    sft_import.add_argument("--output-root", type=_path, required=True)
    sft_import.add_argument("--trust-store", type=_path, required=True)
    sft_import.add_argument(
        "--eligible-model",
        action="append",
        required=True,
        help="exact teacher model identity; repeat for Opus/GPT/Gemini/GLM",
    )
    sft_import.add_argument("--minimum-score", type=float, default=0.8)
    sft_import.add_argument("--shard-size", type=int, default=1024)
    sft_import.add_argument("--verification-workers", type=int, default=64)
    sft_import.set_defaults(handler=_cmd_sft_import_legacy)
    sft_verify = sft_commands.add_parser(
        "verify-legacy", help="reopen one immutable legacy-teacher SFT dataset"
    )
    sft_verify.add_argument("--output-root", type=_path, required=True)
    sft_verify.add_argument("dataset_id")
    sft_verify.set_defaults(handler=_cmd_sft_verify_legacy)

    demo = commands.add_parser("demo-local", help="run a zero-provider parallel-tool smoke test")
    demo.set_defaults(handler=_cmd_demo_local)

    venv = commands.add_parser("bootstrap-venv", help="prepare a cached local coding venv")
    venv.add_argument("path", type=_path)
    venv.add_argument("--python", type=_path)
    venv.set_defaults(handler=_cmd_bootstrap_venv)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "interval", 1.0) <= 0:
        parser.error("--interval must be positive")
    if getattr(args, "rate", None) is not None and args.rate <= 0:
        parser.error("--rate must be positive")
    if not 10 <= getattr(args, "width", 30) <= 100:
        parser.error("--width must be between 10 and 100")
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        return 130
    except (CliError, RubricValidationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
