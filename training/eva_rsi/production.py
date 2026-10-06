"""Explicit production wiring; planning/imports never start a model or provider."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import subprocess

from .controller import read, write
from .evidence import require, verify_evaluation

ROOT = Path(__file__).resolve().parents[2]
NATIVE_GENERATOR = "eva_agent.training.codex_slime_rollout.generate_grpo_rollout"
SEGMENT_REWARD = "eva_agent.training.codex_segment_rewards.normalize_segment_rewards"
MEASURED_GRPO_CONTEXT_PROFILE = "evamed-grpo-native-24576-v1"
EARLY_COMPACT_GRPO_CONTEXT_PROFILE = "evamed-grpo-native-24576-earlycompact-v2"
LOSSLESS_HEADROOM_GRPO_CONTEXT_PROFILE = "evamed-grpo-native-32768-lossless-headroom-v3"
SUPPORTED_GRPO_CONTEXT_PROFILES = {
    MEASURED_GRPO_CONTEXT_PROFILE,
    EARLY_COMPACT_GRPO_CONTEXT_PROFILE,
    LOSSLESS_HEADROOM_GRPO_CONTEXT_PROFILE,
}


def training_domain_selection(stage, observed, settings):
    """Explicit scopes preserve evaluated coverage and never relabel source domains."""
    observed_domains = sorted(name for name, row in observed.items() if row["sample_count"] > 0)
    require(bool(observed_domains), "selected_target_has_no_observed_domain")
    catalog, trust = (settings.get(name) for name in ("execution_catalog_path", "trust_store_path"))
    require((catalog is None) == (trust is None), "signed_selection_requires_catalog_and_trust_store")
    if catalog is not None:
        require(all(isinstance(value, str) and value.strip() for value in (catalog, trust)),
                "invalid_signed_selection_paths")
    configured = settings.get("training_domains_by_stage")
    scope = settings.get("training_domain_scope")
    require(scope in (None, "verified_stage_transfer"), "unsupported_training_domain_scope")
    transfer = scope == "verified_stage_transfer"
    if transfer:
        require(catalog is not None, "stage_transfer_requires_signed_executable_catalog")
        require(configured is None, "stage_transfer_conflicts_with_explicit_domain_subset")
    domains = observed_domains
    if configured is not None:
        require(isinstance(configured, dict) and stage in configured
                and all(key in ("S1", "S2", "S3", "S4", "S5") for key in configured),
                "explicit_training_domain_stage_missing_or_invalid")
        for values in configured.values():
            require(isinstance(values, list) and bool(values)
                    and all(isinstance(value, str) and value for value in values)
                    and len(set(values)) == len(values), "invalid_explicit_training_domains")
        domains = sorted(configured[stage])
        require(set(domains) <= set(observed_domains), "training_domain_not_observed_no_relabeling")
    if transfer:
        domains = []  # Existing signed selector: all original source domains at this stage.
    if configured is None and catalog is None:
        return domains, None  # Historical plan and unfiltered preparation remain unchanged.
    return domains, {"schema": "eva.rsi-training-domain-selection.v1",
        "mode": ("verified_stage_transfer" if transfer else
                 "explicit_observed_subset" if configured is not None else "all_observed_domains"),
        "stage": stage, "observed_domain_coverage": deepcopy(observed),
        "observed_domains": observed_domains, "requested_training_domains": domains,
        "requested_domain_semantics": ("all_signed_executable_source_domains_at_selected_stage"
                                       if transfer else "explicit_domain_names"),
        "omitted_observed_domains": None if transfer else sorted(set(observed_domains) - set(domains)),
        "cross_domain_stage_transfer_explicitly_requested": transfer,
        "source_domains_relabeled": False, "signed_executable_filter_requested": catalog is not None,
        "execution_catalog_path": catalog, "trust_store_path": trust,
        "counts_verified_by_plan": False}


def require_real_evaluation(context):
    """Never start training from the historical, explicitly provisional baseline."""
    from .evidence import commitment
    previous = context["previous_evaluation"]
    if isinstance(previous, str):
        previous = read(previous)
    reference = previous.get("index", {})
    require(isinstance(reference.get("path"), str), "real_evaluation_required_before_training")
    path = Path(reference["path"])
    require(commitment(path) == reference, "previous_evaluation_index_changed")
    index = read(path)
    if index.get("schema") == "eva.rsi-composite-evaluation-index.v1":
        from .composite_eval import verify_composite_index
        proof = verify_composite_index(path)
        require(previous.get("composite_provenance") == proof,
                "previous_composite_evaluation_proof_changed")
        return index
    require(index.get("new_rollouts_per_track") == 1
            and index.get("matched_codex_version_requested") == "0.153.4"
            and not index.get("baseline_semantics", "").startswith("existing diagnostic"),
            "real_evaluation_required_before_training")
    require(index.get("baseline_gate_mode", "complete-workflows-v1") in
            {"complete-workflows-v1", "terminal-attempts-v1"}, "unsupported_baseline_gate_mode")
    if index.get("baseline_gate_mode") == "terminal-attempts-v1":
        from .terminal_attempts import verify_terminal_attempts, require_judged_attempts
        require_judged_attempts(index, verify_terminal_attempts(index))
        return index
    run = Path(index["benchmark_run_root"])
    summary = read(run / "track-rollouts/summary.json")
    require(bool(summary.get("tracks")), "real_evaluation_has_no_track_attempts")
    require(all(not row.get("errors") for row in summary["tracks"]),
            "real_evaluation_has_runner_errors")
    return index


def train_plan(context, settings):
    require(context["phase"] == "train", "requires_training_phase")
    stage = context["s_target"]
    require(stage in ("S1", "S2", "S3", "S4", "S5"), "selected_stage_not_executable_yet")
    remaining = context["remaining_updates"]
    require(type(remaining) is int and 1 <= remaining <= 50, "invalid_round_remainder")
    # Test the first real GRPO checkpoint without risking 25 unsaved updates.
    # The continuation is explicitly a model-only warm start, not exact Adam resume.
    prefix = settings.get("first_update_checkpoint_probe", False) is True and context.get("round") == 1 and remaining == 50
    steps = 1 if prefix else remaining
    context_profile = settings.get("grpo_context_profile")
    require(context_profile is None or context_profile in SUPPORTED_GRPO_CONTEXT_PROFILES,
            "unsupported_grpo_context_profile")
    if context_profile is not None:
        require(settings.get("trajectory_output_budget", 8192) == 8192,
                "measured_grpo_profile_requires_8192_trajectory_budget")
    context_tokens = (
        32768
        if context_profile == LOSSLESS_HEADROOM_GRPO_CONTEXT_PROFILE
        else 24576
    )
    previous = context["previous_evaluation"]
    if isinstance(previous, str):
        previous = read(previous)
    observed = previous["proposal"]["stages"][stage]["domains"]
    domains, domain_selection = training_domain_selection(stage, observed, settings)
    attempt = Path(context["attempt_root"])
    argv = [settings["training_python"], str(ROOT / "training/slime/run_full_parameter.py"),
        "--stage", "grpo", "--data", str(attempt / "data/grpo.jsonl"),
        "--output", str(attempt / "training"), "--model", context["architecture_model_path"],
        "--load", context["checkpoint_root"], "--steps", str(steps),
        "--batch-size", "4", "--samples-per-prompt", "4", "--max-tokens", str(context_tokens),
        "--max-response-tokens", str(settings.get("trajectory_output_budget", 8192)),
        "--generation-function", NATIVE_GENERATOR,
        "--reward-post-process-function", SEGMENT_REWARD,
        "--judge-backend", "native_astra", "--checkpoint-model-only", "--export-hf",
        "--save-interval", str(min(25, steps))]
    full_graphs = settings.get("decode_full_graphs", True)
    require(type(full_graphs) is bool, "invalid_decode_graph_configuration")
    if full_graphs:
        argv.append("--decode-full-graphs")
    abort_save = settings.get("save_on_rollout_error", False)
    require(type(abort_save) is bool, "invalid_save_on_rollout_error_configuration")
    if abort_save:
        argv.append("--save-on-rollout-error")
    first_checkpoint = settings.get("checkpoint_first_update", False)
    require(type(first_checkpoint) is bool, "invalid_checkpoint_first_update_configuration")
    require(not first_checkpoint or abort_save, "first_checkpoint_requires_abort_save_driver")
    if first_checkpoint:
        argv.append("--checkpoint-first-update")
    replacements = settings.get("judge_group_replacements", 0)
    require(type(replacements) is int and 0 <= replacements <= 3,
            "invalid_judge_group_replacements_configuration")
    require(not replacements or abort_save, "judge_group_isolation_requires_abort_save_driver")
    if replacements:
        argv.extend(["--judge-group-replacements", str(replacements)])
    return {"schema": "eva.rsi-native-codex-training-plan.v1", "argv": argv,
        **({"judge_group_replacements": replacements} if replacements else {}),
        **({"domain_selection": domain_selection} if domain_selection is not None else {}),
        "stage": stage, "domains": domains, "remaining_updates": remaining,
        "planned_chunk_updates": steps, "first_update_checkpoint_probe": prefix,
        "actor_backend": "native_codex_sglang", "actor_execution_verified_by_plan": False,
        **({"grpo_context_profile": context_profile,
            "context_profile_environment": {"EVA_GRPO_CONTEXT_PROFILE": context_profile}}
           if context_profile is not None else {}),
        "codex_bin": settings["codex_bin"], "thinking": True,
        **({"skill_selection": context["skill_selection"]} if context.get("skill_selection") else {}),
        "output_tokens_include_private_reasoning": True,
        "checkpoint_policy": "DCP plus HF every25 updates and final; no deletion"
        + ("; also after the first acknowledged update while continuing the same optimizer"
           if first_checkpoint else "")
        + ("; on next-rollout failure save acknowledged finite prefix before propagating failure"
           if abort_save else ""),
        "exact_optimizer_resume": False, "provider_calls": 0, "gpu_launches": 0}


def prepare_training_data(context, settings, plan):
    """CPU-only source selection; the requested limit is never a distinct count."""
    from eva_agent.training.grpo_stage_data import StageDataCoverageError, prepare_stage_data
    from .evidence import commitment

    attempt = Path(context["attempt_root"])
    selection = plan.get("domain_selection")
    options = {}
    if selection and selection["signed_executable_filter_requested"]:
        options = {name: Path(selection[name]) for name in ("execution_catalog_path", "trust_store_path")}
    limit = settings.get("training_sandbox_limit", 128)

    def retain_counts(receipt):
        if selection is None:
            return
        coverage = receipt.get("selection")
        selected_domains = sorted(receipt.get("domains", []))
        write(attempt / "training-data-selection.json", {
            "schema": "eva.rsi-training-data-selection.v1", "domain_selection": selection,
            "status": receipt.get("status", "prepared"), "requested_distinct_limit": limit,
            "actual_distinct_sandbox_count": receipt["sample_count"],
            "requested_limit_shortfall": max(0, limit - receipt["sample_count"]),
            "requested_domain_source_match_count": coverage["target_match_count"] if coverage else None,
            "requested_domain_signed_executable_count": coverage["eligible_count"] if coverage else None,
            "selected_training_domains": selected_domains,
            "observed_domains_without_selected_training_sources": sorted(
                set(selection["observed_domains"]) - set(selected_domains)),
            "selected_training_domains_not_observed_in_eval": sorted(
                set(selected_domains) - set(selection["observed_domains"])),
            "eligible_signed_source_domains": sorted(domain for domain, row in coverage["domain_coverage"].items()
                                                      if row["executable_legacy_rows"] > 0) if coverage else None,
            "signed_selection": coverage,
            "preparation_receipt": commitment(attempt / "data/preparation-receipt.json"),
            "batching_creates_new_distinct_sandboxes": False,
            "online_generation_required": True, "provider_calls": 0, "gpu_launches": 0}, exclusive=True)

    try:
        receipt = prepare_stage_data(Path(settings["bulk_root"]), attempt / "data", stage=plan["stage"],
            domains=plan["domains"], limit=limit, actor_backend="native_codex_sglang",
            judge_backend="native_astra", **options)
    except StageDataCoverageError:
        retain_counts(read(attempt / "data/preparation-receipt.json"))
        raise
    retain_counts(receipt)
    return receipt


def execute_train(context, settings):
    from training.harness_source import runtime_paths

    require_real_evaluation(context)
    plan = train_plan(context, settings)
    attempt = Path(context["attempt_root"])
    write(attempt / "training-plan.json", plan, exclusive=True)
    prepare_training_data(context, settings, plan)
    # Do not serialize inherited credentials. The launcher propagates only its
    # explicit worker environment allowlist and requires an unused GPU.
    environment = dict(os.environ)
    environment.update({"EVA_GRPO_CODEX_BIN": settings["codex_bin"],
        "EVA_MEDRESEARCH_DATA_ROOT": str(Path(settings.get("medical_data_root", ROOT)).resolve(strict=True)),
        "EVA_GRPO_ENABLE_THINKING": "1", "EVA_GRPO_MAX_TOOL_FRONTIERS": "32",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.pathsep.join(runtime_paths(ROOT))})
    environment.update(plan.get("context_profile_environment", {}))
    for key in ("EVA_RSI_SKILL_SELECTION", "EVA_RSI_SKILL_SELECTION_BLAKE3", "EVA_RSI_SKILL_CATALOG_ID"):
        environment.pop(key, None)  # Selection belongs to this exact attempt, not inherited shell state.
    if context.get("skill_selection"):
        from .skill_selection import read_selection
        reference = context["skill_selection"]
        read_selection(reference, catalog_id=context["skill_catalog_id"])
        environment.update(EVA_RSI_SKILL_SELECTION=reference["path"],
            EVA_RSI_SKILL_SELECTION_BLAKE3=reference["blake3"],
            EVA_RSI_SKILL_CATALOG_ID=context["skill_catalog_id"])
    result = subprocess.run(plan["argv"], cwd=ROOT, env=environment, check=False)
    return result.returncode


def materialize_provisional_baseline(context, settings):
    """Reuse a real prior Judge as disclosed curriculum evidence, not a new eval."""
    require(context["phase"] == "baseline_eval", "requires_baseline_phase")
    source = Path(settings["provisional_baseline_index"])
    verification = verify_evaluation(source, context)
    document = read(source)
    document.update({"baseline_semantics": "existing diagnostic; provisional curriculum only",
        "matched_codex_01534_baseline": False, "new_rollouts_performed": 0,
        "prior_evidence_index": str(source.resolve())})
    destination = Path(context["attempt_root"]) / "evaluation-index.json"
    write(destination, document, exclusive=True)
    return verification


def materialize_verified_baseline(context, settings):
    """Bind the actual new evaluation; never regenerate or replace its outcomes."""
    require(context["phase"] == "baseline_eval", "requires_baseline_phase")
    source = Path(settings["verified_baseline_index"])
    verification = verify_evaluation(source, context)
    document = require_real_evaluation({**context, "previous_evaluation": verification})
    require_complete_baseline(document)
    document = {**document, "baseline_semantics": "actual current-checkpoint Codex evaluation",
                "prior_evidence_index": str(source.resolve())}
    write(Path(context["attempt_root"]) / "evaluation-index.json", document, exclusive=True)
    return verification


def require_complete_baseline(document):
    """The initial training gate requires the actual seven requested workflows.

    Completion is execution coverage, not a minimum model score.
    """
    if document.get("schema") == "eva.rsi-composite-evaluation-index.v1":
        from .composite_eval import verify_composite_index
        return verify_composite_index(document)
    if document.get("baseline_gate_mode") == "terminal-attempts-v1":
        from .terminal_attempts import verify_terminal_attempts
        return verify_terminal_attempts(document)
    require(document.get("baseline_gate_mode", "complete-workflows-v1") == "complete-workflows-v1",
            "unsupported_baseline_gate_mode")
    from training.automedbench_lite.track_adapter import BY_TRACK
    summary = read(Path(document["benchmark_run_root"]) / "track-rollouts/summary.json")
    rows = summary.get("tracks", [])
    require(document.get("evaluation_mode") == "full_single_pass"
            and len(rows) == len(BY_TRACK)
            and {row.get("track") for row in rows} == set(BY_TRACK)
            and all(row.get("completed_requested_turns") is True and not row.get("errors") for row in rows),
            "complete_seven_track_baseline_required")


def loop_config(settings, settings_path):
    python = settings["control_python"]
    common = [python, str(ROOT / "scripts/eva_rsi_production_v1.py"), "{context}",
              "--settings", str(Path(settings_path).resolve()), "--execute"]
    commands = {name: list(common) for name in ("baseline_eval", "train", "evaluation")}
    attribution = settings.get("skill_attribution")
    if attribution is not None:
        require(isinstance(attribution, list) and bool(attribution)
                and all(isinstance(part, str) and part for part in attribution),
                "skill_attribution_requires_explicit_command_argv")
        commands["skill_attribution"] = list(attribution)
    return {"schema": "eva.rsi-loop-config.v1", "rounds": 10, "updates_per_round": 50,
        "cwd": str(ROOT), "architecture_model_path": settings["architecture_model_path"],
        "initial_model_path": settings["initial_model_path"],
        "initial_checkpoint_root": settings["initial_checkpoint_root"],
        "skill_catalog_id": settings["skill_catalog_id"],
        "commands": commands}
