"""Native Codex actor + canonical EVA tools + exact-token Slime GRPO segments.

The on-policy actor is an actual pinned Codex app-server. Its Responses gateway
forwards each real provider request to SGLang /generate, retaining original IDs
and sampled log probabilities. Each request is an independent training segment;
overlapping/compacted history is conditioning, never duplicated supervision.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
from threading import Lock
from typing import Any
from uuid import uuid4

from eva_agent.pipeline.digests import canonical_value, blake3_bytes
from eva_agent.training.slime_rollout import _write_private_json, preserve_sampled_tokens


REWARD_HOOK = "eva_agent.training.codex_segment_rewards.normalize_segment_rewards"
MODEL_ALIAS = "Qwen/Qwen3.5-9B"
_CONTEXTS = None
_CONTEXT_LOCK = Lock()
_BINARY_IDENTITIES = {}
_BINARY_LOCK = Lock()


def native_compaction_limit(context_tokens, output_tokens=4096):
    # Actual retained 32k evaluation history expanded during tool-free
    # compaction. Reserve render headroom prospectively for native 24k
    # training; this does not bypass exact per-request tokenizer admission.
    return min(10240, context_tokens - output_tokens - 1024)


def _binary_identity(binary):
    stat = binary.stat()
    key = (str(binary), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    with _BINARY_LOCK:
        if key not in _BINARY_IDENTITIES:
            version = subprocess.check_output([str(binary), "--version"], text=True, timeout=15).strip()
            if version != "codex-cli 0.153.4":
                raise ValueError("Native Codex rollout requires the reviewed codex-cli 0.153.4")
            digest = blake3_bytes(binary.read_bytes())
            current = binary.stat()
            if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != key[1:]:
                raise ValueError("Pinned Codex executable changed during identity verification")
            _BINARY_IDENTITIES[key] = (version, digest)
        return _BINARY_IDENTITIES[key]


def native_settings(args):
    if os.environ.get("EVA_GRPO_ENABLE_THINKING", "1") != "1":
        raise ValueError("Native Codex GRPO requires explicit thinking enabled")
    maximum = int(os.environ.get("EVA_GRPO_MAX_TOOL_FRONTIERS", "12"))
    if not 1 <= maximum <= 64:
        raise ValueError("Native Codex provider request bound differs")
    context = int(args.seq_length)
    if not 8192 <= context <= 24576 or args.rollout_max_response_len < 4096:
        raise ValueError("Native Codex rollout requires measured <=24576 context and >=4096 output budget")
    binary = os.environ.get("EVA_GRPO_CODEX_BIN")
    if not binary or not Path(binary).is_file() or not os.access(binary, os.X_OK):
        raise ValueError("EVA_GRPO_CODEX_BIN must bind the pinned executable explicitly")
    binary = Path(binary).resolve(strict=True)
    version, binary_digest = _binary_identity(binary)
    return {"codex_bin": binary, "codex_version": version,
            "codex_binary_blake3": binary_digest,
            "max_requests": maximum, "max_context_tokens": context,
            "max_output_tokens": 4096, "total_output_budget": int(args.rollout_max_response_len),
            "auto_compact_token_limit": native_compaction_limit(context),
            # This is a provider-request cap, including compaction. The actual
            # canonical runtime separately retains its true execution frontiers.
            "provider_request_cap_derived_from_frontier_budget": True}


def samples_from_segments(args, original, segments, *, max_requests, training_iteration,
                          sample_id, trajectory_path, sample_type=None):
    """Construct exact response-only samples without re-tokenizing any output."""
    if sample_type is None:
        from slime.utils.types import Sample
        sample_type = Sample
    if (type(original.index) is not int or original.index < 0
            or type(original.group_index) is not int or original.group_index < 0
            or not 1 <= len(segments) <= max_requests):
        raise ValueError("Native trajectory/segment identities differ")
    result = []
    for position, segment in enumerate(segments):
        prompt, output = segment["prompt_token_ids"], segment["output_token_ids"]
        probabilities = segment["output_token_logprobs"]
        if (segment["segment_index"] != position or not prompt or not output
                or any(type(token) is not int or token < 0 for token in [*prompt, *output])
                or len(prompt) + len(output) > args.seq_length
                or len(probabilities) != len(output)
                or any(type(value) not in (int, float) or not math.isfinite(value) or value > 0 for value in probabilities)
                or segment["loss_mask"] != [1] * len(output)):
            raise ValueError("Native exact prompt/output token, logprob, or mask alignment differs")
        metadata = deepcopy(original.metadata or {})
        metadata.update({"actor_backend": "native_codex_sglang", "training_iteration": training_iteration,
                         "original_trajectory_id": sample_id, "original_sample_index": original.index,
                         "provider_request_id": segment["request_id"], "provider_segment_index": position,
                         "prompt_token_count": len(prompt), "generated_token_count": len(output),
                         "prompt_tokens_are_conditioning_only": True, "sampled_outputs_retokenized": False,
                         "response_text_projection_omitted": True, "trajectory_path": str(trajectory_path),
                         "requested_model_alias": segment["requested_model"],
                         "returned_model_identity": segment.get("returned_model")})
        sample = sample_type(index=original.index * max_requests + position,
                             rollout_id=original.index, group_index=original.group_index,
                             prompt="", tokens=list(prompt), response="", response_length=0,
                             loss_mask=[], rollout_log_probs=None, metadata=metadata)
        sample.append_response_tokens(args, tokens=list(output), log_probs=list(probabilities),
                                      trainable=True, meta_info=deepcopy(segment["meta_info"]),
                                      update_terminal_info=True)
        if (sample.tokens != [*prompt, *output] or sample.response_length != len(output)
                or sample.loss_mask != [1] * len(output) or sample.rollout_log_probs != probabilities):
            raise ValueError("Installed Slime Sample changed exact native segment data")
        result.append(sample)
    return result


def _persist_failure(sample_root, error, *, phase, transport):
    receipt = getattr(error, "receipt", None)
    _write_private_json(sample_root / "failure.json", {
        "schema": "eva.native-codex-grpo-failure.v1", "phase": phase,
        "error_type": type(error).__name__, "safe_failure_category": getattr(error, "safe_failure_category", None),
        "safe_failure_message": getattr(error, "safe_failure_message", None),
        "codex_turn_receipt": canonical_value(receipt) if receipt is not None else None,
        "transport": transport.safe_metadata, "reward_emitted": False,
        "automatic_retry": False, "partial_segments_are_not_training_admission": True,
    })


def run_native_trajectory(args, original, *, tokenizer, sampling_params, record, context,
                          sample_root: Path, sample_id: str, training_iteration: int,
                          settings=None, sample_type=None):
    """One actual Codex trajectory, one actual whole-workspace judge, no retries."""
    from eva_agent.codex_pipeline import CodexRolloutAdapter
    from eva_agent.codex_runtime import CodexRole, CodexRuntime, CodexSandbox, CodexThreadOptions
    from eva_agent.pipeline import Cohort, ModelTarget
    from eva_agent.training.codex_sglang_transport import CodexSGLangTransport
    from eva_agent.training.qwen_tool_types import HostArgumentProjector
    from eva_agent.training.slime_agent_judge import grade_rollout, resolve_judge_backend
    from eva_agent.training.teacher_focus import focus_only_teacher_instructions
    from eva_agent.training.teacher_worker import execute_single_rollout
    from training.automedbench_lite.local_qwen import local_qwen_setup

    settings = settings or native_settings(args)
    meta = original.metadata or {}
    backend = resolve_judge_backend()
    if meta.get("actor_backend") != "native_codex_sglang" or meta.get("judge_backend") != backend:
        raise ValueError("Native Codex actor/Judge differs from explicit data provenance")
    if record["stage"] not in {"S1", "S2", "S3"} or meta.get("stage") != record["stage"]:
        raise ValueError("Native Codex stage target differs")
    sample_root = Path(sample_root).resolve()
    sample_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    generate_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    transport = CodexSGLangTransport(tokenizer=tokenizer, generate_url=generate_url,
        sampling_params=sampling_params, max_context_tokens=settings["max_context_tokens"],
        max_output_tokens=settings["max_output_tokens"], total_output_budget=settings["total_output_budget"],
        max_requests=settings["max_requests"], evidence_root=sample_root / "private-provider-tokens",
        argument_projector=HostArgumentProjector.from_context(context))
    trajectory_path = sample_root / "trajectory.json"
    phase = "native_codex_actor"
    captured = []
    try:
        with local_qwen_setup(run_root=sample_root / "codex-provider", model=MODEL_ALIAS,
                endpoint=f"http://127.0.0.1:{args.sglang_router_port}/v1",
                context_length=settings["max_context_tokens"], max_output_tokens=settings["max_output_tokens"],
                workers=1, thinking=True, exact_tool_schemas=True, normalize_priority_messages=True,
                codex_bin=settings["codex_bin"],
                auto_compact_token_limit=settings.get("auto_compact_token_limit",
                    native_compaction_limit(settings["max_context_tokens"], settings["max_output_tokens"])),
                upstream_transport=transport) as setup:
            def options(request, cwd, offers, bridge):
                return CodexThreadOptions(role=CodexRole.STRONG_ACTOR, model=request.model.model_id,
                    provider=request.model.provider, cwd=cwd, sandbox=CodexSandbox.READ_ONLY,
                    config=setup.thread_config, offered_tools=offers, ephemeral=True,
                    developer_instructions=focus_only_teacher_instructions(context))
            actor = CodexRolloutAdapter(runtime_factory=lambda bridge: CodexRuntime(setup.backend()),
                options_factory=options, turn_mcp_factory=context.turn_mcp_factory,
                skills_factory=context.skills_factory,
                controlled_budget_terminal=lambda: transport.budget_termination,
                allow_empty_final_response=True)
            rollout = execute_single_rollout(record=record, context=context, provider=actor,
                target=ModelTarget(Cohort.STRONG, MODEL_ALIAS, setup.provider),
                workspace_root=sample_root / "workspace", task_id=sample_id, route_id="qwen_3_5_9b")
            rollout = canonical_value(rollout)
            rollout["provider_metadata"].update({
                "actor_backend": "native_codex_sglang", "native_codex_actor": True,
                "thinking_explicitly_enabled": True, "sampled_token_capture": transport.safe_metadata,
                "codex_version": settings["codex_version"], "codex_binary_blake3": settings["codex_binary_blake3"],
                "requested_model_alias": MODEL_ALIAS, "returned_model_identity": None,
                "provider_request_cap": settings["max_requests"], "initialization": setup.safe_metadata,
            })
        phase = "exact_segment_validation"
        captured = samples_from_segments(args, original, transport.segments,
            max_requests=settings["max_requests"], training_iteration=training_iteration,
            sample_id=sample_id, trajectory_path=trajectory_path, sample_type=sample_type)
        _write_private_json(trajectory_path, rollout)
        for index, sample in enumerate(captured):
            preserve_sampled_tokens(sample, sample_root / f"sampled-policy-tokens-{index:04d}.json")
        phase = "workspace_agent_judge"
        verdict = asyncio.run(grade_rollout(rollout, rubric=context.rubric,
                                          output_root=sample_root / "judge", sample_id=sample_id, backend=backend))
        for index, sample in enumerate(captured):
            sample.reward = verdict["reward"]
            sample.metadata.update({"judge_result": verdict, "reward_shared_across_original_trajectory": True})
            _write_private_json(sample_root / f"training-tokens-{index:04d}.json", {
                "tokens": sample.tokens, "response_length": sample.response_length,
                "loss_mask": sample.loss_mask, "rollout_log_probs": sample.rollout_log_probs,
                "reward": sample.reward, "rollout_id": sample.rollout_id,
                "sample_index": sample.index, "group_index": sample.group_index})
        _write_private_json(sample_root / "summary.json", {
            "schema": "eva.native-codex-grpo-trajectory.v1", "sample_id": sample_id,
            "original_sample_index": original.index, "group_index": original.group_index,
            "training_iteration": training_iteration, "status": "complete", "native_codex_actor": True,
            "segments": len(captured), "sampled_tokens": sum(s.response_length for s in captured),
            "reward": verdict["reward"], "workspace_agent_judged": True,
            "codex_turn_receipt_retained": "codex_turn_receipt" in rollout["provider_metadata"],
            "transport": transport.safe_metadata})
        return captured
    except BaseException as error:
        # The transport persists each real token response before projection, so
        # even terminal Codex/Judge failures cannot erase earlier sampled IDs.
        _persist_failure(sample_root, error, phase=phase, transport=transport)
        raise


def generate_grpo_rollout(args, rollout_id: int, data_buffer, evaluation=False):
    """Installed Slime hook: prompt × original trajectory × exact segments."""
    if evaluation:
        raise ValueError("Use the independent held-out Codex evaluation harness")
    if getattr(args, "custom_reward_post_process_path", None) != REWARD_HOOK:
        raise ValueError("Segmented native Codex requires original-trajectory reward normalization")
    from slime.rollout.sglang_rollout import GenerateState
    from eva_agent.training.teacher_worker import CampaignV2TeacherContextPool, load_bulk_record
    global _CONTEXTS
    settings = native_settings(args)
    with _CONTEXT_LOCK:
        if _CONTEXTS is None:
            _CONTEXTS = CampaignV2TeacherContextPool()
    state = GenerateState(args)
    groups = data_buffer.get_samples(args.rollout_batch_size)
    if len(groups) != args.rollout_batch_size or any(len(group) != args.n_samples_per_prompt for group in groups):
        raise ValueError("Native Codex original trajectory group cardinality differs")
    output = Path(args.save).parent / "rollouts" / f"{rollout_id:06d}"
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    originals = [sample for group in groups for sample in group]
    def one(sample):
        record = load_bulk_record(Path(sample.metadata["bulk_root"]), sample.metadata["sandbox_id"])
        context, _ = _CONTEXTS.load(record)
        sample_id = str(uuid4())
        return run_native_trajectory(args, sample, tokenizer=state.tokenizer, sampling_params=state.sampling_params,
            record=record, context=context, sample_root=output / sample_id, sample_id=sample_id,
            training_iteration=rollout_id, settings=settings)
    with ThreadPoolExecutor(max_workers=min(4, len(originals))) as executor:
        trajectories = list(executor.map(one, originals))
    nested, cursor = [], 0
    for group in groups:
        nested.append(trajectories[cursor:cursor + len(group)])
        cursor += len(group)
    summary_groups = []
    for group in nested:
        rewards = [trajectory[0].reward for trajectory in group]
        summary_groups.append({"rewards": rewards, "zero_variance_group": len(set(rewards)) == 1,
                               "original_trajectories": len(group), "segments": sum(len(t) for t in group)})
    _write_private_json(output / "summary.json", {"native_codex_actor": True, "groups": summary_groups,
                        "original_trajectories": len(originals), "segments": sum(len(t) for t in trajectories),
                        "rubric_agent_judged": True, "original_trajectory_reward_normalization_required": True})
    return nested
