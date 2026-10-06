"""Opt-in FP32 logits consume-on-backward path for the one-GPU GRPO loss.

The installed native chunk kernel still computes FP32 temperature-scaled
log-probabilities and metric-only entropy. Exact softmax chunks live on CPU.
Backward consumes otherwise-dead logits storage, eliminating the full-sized
SplitBackward and temperature-division gradient allocations. No parameters,
tokens, masks, rewards, PPO terms or reductions are removed or rewritten.

This is a one-shot, first-order backward only. Retained/repeated backward,
higher-order derivatives and additional consumers of logits are unsupported.
"""
from __future__ import annotations

from copy import copy
import inspect
import math
import os
from types import FunctionType

import torch

ENV = "EVA_SLIME_LOSS_MEMORY"
POLICY = "fp32-chunked-reuse-v1"
RESPONSE_POLICY = "fp32-response-chunked-reuse-v2"


class _ConsumeFP32Logits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, tokens, temperature, chunk_size, with_entropy, kernel, process_group):
        if logits.dtype != torch.float32 or logits.ndim != 2 or not logits.is_contiguous():
            raise ValueError("FP32 contiguous [tokens,vocab] logits required")
        if not math.isfinite(temperature) or temperature <= 0 or chunk_size < 1:
            raise ValueError("positive finite temperature and chunk size required")
        rows, vocab = logits.shape
        if rows < 1 or tokens.shape != (rows,) or tokens.dtype != torch.long:
            raise ValueError("nonempty aligned token targets required")
        backing = torch.empty((rows, vocab), dtype=torch.float32, device="cpu")
        staging = torch.empty((min(rows, chunk_size), vocab), dtype=torch.float32, device=logits.device)
        lp_parts, entropy_parts = [], []
        # Match installed torch.chunk(num_chunks), including its uneven final geometry.
        width = math.ceil(rows / math.ceil(rows / chunk_size))
        for start in range(0, rows, width):
            end = min(start + width, rows)
            work = staging[:end-start]
            work.copy_(logits[start:end])
            if temperature != 1.0:
                work.div_(temperature)  # Same FP32 operation as the original full tensor.
            with torch.enable_grad():
                proxy = work.detach().requires_grad_(True)
                lp, entropy = kernel.apply(proxy, tokens[start:end], None, process_group,
                                          with_entropy, False, None, 0)
            # The exact installed kernel's saved FP32 probability, not a recomputation.
            softmax = lp.grad_fn.saved_tensors[0]
            if softmax.shape != work.shape or softmax.dtype != torch.float32:
                raise RuntimeError("installed log-probability kernel softmax contract changed")
            lp_parts.append(lp.detach())
            if with_entropy:
                entropy_parts.append(entropy.detach())
            work.copy_(softmax)
            backing[start:end].copy_(work)
            del softmax, proxy, lp, entropy
        ctx.logits, ctx.backing, ctx.staging = logits, backing, staging
        ctx.temperature, ctx.width = temperature, width
        ctx.save_for_backward(tokens)
        entropy = torch.cat(entropy_parts) if with_entropy else logits.new_empty((0,))
        ctx.mark_non_differentiable(entropy)
        return torch.cat(lp_parts), entropy

    @staticmethod
    def backward(ctx, grad_log_prob, _grad_entropy):
        if grad_log_prob is None:
            raise RuntimeError("log-probability gradient required")
        tokens, = ctx.saved_tensors
        logits, backing, staging = ctx.logits, ctx.backing, ctx.staging
        for start in range(0, logits.size(0), ctx.width):
            end = min(start + ctx.width, logits.size(0))
            work = staging[:end-start]
            work.copy_(backing[start:end])
            work.neg_()
            row = torch.arange(end-start, device=work.device)
            work[row, tokens[start:end]] += 1.0
            work.mul_(grad_log_prob[start:end].reshape(-1, 1))
            if ctx.temperature != 1.0:
                work.div_(ctx.temperature)  # Before the unchanged model-output dtype boundary.
            logits[start:end].copy_(work)
        ctx.logits = ctx.backing = ctx.staging = None
        # Sole consumer contract: the original logits are no longer readable after backward.
        return logits, None, None, None, None, None, None


def calculate(logits, tokens, process_group, *, temperature, chunk_size, with_entropy,
              kernel, rank_size):
    if rank_size(process_group) != (0, 1):
        raise ValueError("loss-memory path supports tensor-parallel size one only")
    lp, entropy = _ConsumeFP32Logits.apply(logits, tokens, temperature, chunk_size,
                                          with_entropy, kernel, process_group)
    return lp, entropy if with_entropy else None


def response_chunks(rows, chunk_size, total_lengths, response_lengths):
    """Original torch.chunk geometry intersecting exact CP1 response-logit slices.

    Boundary chunks remain whole: changing their row geometry can change the
    native entropy reduction. Padding and prompt-only chunks are not consumers
    of the installed response loss. This does not shorten model conditioning.
    """
    if rows < 1 or chunk_size < 1 or len(total_lengths) != len(response_lengths):
        raise ValueError("aligned nonempty CP1 lengths and positive chunk size required")
    intervals, offset = [], 0
    for total, response in zip(total_lengths, response_lengths, strict=True):
        if (type(total) is not int or type(response) is not int or total < 1
                or response < 0 or response >= total):
            raise ValueError("each response requires at least one preceding prompt token")
        end = offset + total
        if response:
            intervals.append((end - response - 1, end - 1))
        offset = end
    if offset > rows or not intervals:
        raise ValueError("response lengths must fit padded logits and contain a response")
    width = math.ceil(rows / math.ceil(rows / chunk_size))
    return tuple((start, min(start + width, rows)) for start in range(0, rows, width)
                 if any(start < hi and min(start + width, rows) > lo for lo, hi in intervals))


class _ConsumeResponseFP32Logits(torch.autograd.Function):
    """One-shot v2: exact native chunks, compact CPU backing, full model gradient."""
    @staticmethod
    def forward(ctx, logits, tokens, temperature, chunks, with_entropy, kernel, process_group):
        if logits.dtype != torch.float32 or logits.ndim != 2 or not logits.is_contiguous():
            raise ValueError("FP32 contiguous [tokens,vocab] logits required")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("positive finite temperature required")
        rows, vocab = logits.shape
        if tokens.shape != (rows,) or tokens.dtype != torch.long:
            raise ValueError("aligned token targets required")
        kept_rows = sum(end - start for start, end in chunks)
        backing = torch.empty((kept_rows, vocab), dtype=torch.float32, device="cpu")
        staging = torch.empty((max(end - start for start, end in chunks), vocab),
                              dtype=torch.float32, device=logits.device)
        log_probs = logits.new_zeros((rows, 1))
        entropy_full = logits.new_zeros((rows,)) if with_entropy else logits.new_empty((0,))
        backing_offset = 0
        for start, end in chunks:
            work = staging[:end-start]
            work.copy_(logits[start:end])
            if temperature != 1.0:
                work.div_(temperature)
            with torch.enable_grad():
                proxy = work.detach().requires_grad_(True)
                lp, entropy = kernel.apply(proxy, tokens[start:end], None, process_group,
                                          with_entropy, False, None, 0)
            softmax = lp.grad_fn.saved_tensors[0]
            if softmax.shape != work.shape or softmax.dtype != torch.float32:
                raise RuntimeError("installed log-probability kernel softmax contract changed")
            log_probs[start:end].copy_(lp.detach())
            if with_entropy:
                entropy_full[start:end].copy_(entropy.detach())
            work.copy_(softmax)
            backing[backing_offset:backing_offset + end-start].copy_(work)
            backing_offset += end-start
            del softmax, proxy, lp, entropy
        ctx.logits, ctx.backing, ctx.staging = logits, backing, staging
        ctx.temperature, ctx.chunks = temperature, chunks
        ctx.save_for_backward(tokens)
        ctx.mark_non_differentiable(entropy_full)
        return log_probs, entropy_full

    @staticmethod
    def backward(ctx, grad_log_prob, _grad_entropy):
        if grad_log_prob is None:
            raise RuntimeError("log-probability gradient required")
        tokens, = ctx.saved_tensors
        logits, backing, staging = ctx.logits, ctx.backing, ctx.staging
        # Prompt-only/padded output logits have zero upstream loss derivative.
        # Context hidden states still receive gradients through full attention.
        logits.zero_()
        backing_offset = 0
        for start, end in ctx.chunks:
            work = staging[:end-start]
            work.copy_(backing[backing_offset:backing_offset + end-start])
            work.neg_()
            row = torch.arange(end-start, device=work.device)
            work[row, tokens[start:end]] += 1.0
            work.mul_(grad_log_prob[start:end].reshape(-1, 1))
            if ctx.temperature != 1.0:
                work.div_(ctx.temperature)
            logits[start:end].copy_(work)
            backing_offset += end-start
        ctx.logits = ctx.backing = ctx.staging = None
        return logits, None, None, None, None, None, None


def calculate_responses(logits, tokens, process_group, *, temperature, chunk_size, with_entropy,
                        kernel, rank_size, total_lengths, response_lengths):
    """Only response consumers may use returned values; other rows are placeholders."""
    if rank_size(process_group) != (0, 1):
        raise ValueError("loss-memory path supports tensor-parallel size one only")
    chunks = response_chunks(logits.size(0), chunk_size, total_lengths, response_lengths)
    lp, entropy = _ConsumeResponseFP32Logits.apply(
        logits, tokens, temperature, chunks, with_entropy, kernel, process_group)
    return lp, entropy if with_entropy else None


def make_get_log_probs(original, ppo, *, policy=POLICY):
    """Retain installed response slicing/offset logic verbatim, changing its kernel only."""
    if policy not in (POLICY, RESPONSE_POLICY):
        raise ValueError("unknown explicit EVA loss-memory policy")
    def wrapped(logits, *, args, **kwargs):
        if not torch.is_grad_enabled() or not logits.requires_grad:
            return original(logits, args=args, **kwargs)
        if (getattr(args, "entropy_coef", 0) != 0 or getattr(args, "rollout_top_p", 1) != 1
                or kwargs.get("top_p_token_ids") is not None or kwargs.get("top_p_token_offsets") is not None
                or not sum(kwargs["response_lengths"])):
            raise ValueError("loss-memory path requires nonempty full-vocabulary policy loss and metric-only entropy")
        temperature = float(args.rollout_temperature)
        if policy == RESPONSE_POLICY and getattr(args, "allgather_cp", False):
            raise ValueError("response-chunk policy requires the installed CP1 non-allgather layout")

        def compute(values, targets, group, *, with_entropy=False, chunk_size=-1,
                    log_prob_keep_mask=None, with_entropy_grad=True, reuse_logits_storage_for_backward=False):
            if log_prob_keep_mask is not None or (with_entropy and with_entropy_grad):
                raise ValueError("loss-memory path cannot change top-p or entropy-gradient semantics")
            if policy == RESPONSE_POLICY:
                return calculate_responses(values, targets, group, temperature=temperature,
                    chunk_size=chunk_size, with_entropy=with_entropy,
                    kernel=ppo._VocabParallelLogProbEntropy, rank_size=ppo._get_vocab_parallel_rank_size,
                    total_lengths=kwargs["total_lengths"], response_lengths=kwargs["response_lengths"])
            return calculate(values, targets, group, temperature=temperature, chunk_size=chunk_size,
                with_entropy=with_entropy, kernel=ppo._VocabParallelLogProbEntropy,
                rank_size=ppo._get_vocab_parallel_rank_size)

        namespace = {**original.__globals__, "calculate_log_probs_and_entropy": compute}
        native = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
        native.__kwdefaults__ = original.__kwdefaults__
        local_args = copy(args)
        local_args.rollout_temperature = 1.0  # Real temperature applied inside the FP32 chunk kernel above.
        return native(logits, args=local_args, **kwargs)

    wrapped._eva_loss_memory_policy = policy
    return wrapped


def install(args):
    selection = os.environ.get(ENV)
    if selection is None:
        return None
    if selection not in (POLICY, RESPONSE_POLICY):
        raise ValueError("unknown explicit EVA loss-memory policy")
    if (args.loss_type != "policy_loss" or args.entropy_coef != 0 or args.rollout_top_p != 1
            or args.tensor_model_parallel_size != 1 or args.context_parallel_size != 1
            or args.recompute_loss_function or args.log_probs_chunk_size <= 0):
        raise ValueError("loss-memory policy requires reviewed TP1/CP1 no-entropy-gradient GRPO geometry")
    from slime.backends.megatron_utils import loss
    from slime.utils import ppo_utils
    if selection == RESPONSE_POLICY and getattr(args, "allgather_cp", False):
        raise ValueError("response-chunk policy requires CP1 non-allgather layout")
    current = getattr(loss.get_log_probs_and_entropy, "_eva_loss_memory_policy", None)
    if current == selection:
        return None
    if current is not None:
        raise ValueError("loss-memory policy cannot change inside an initialized actor")
    original = loss.get_log_probs_and_entropy
    loss.get_log_probs_and_entropy = make_get_log_probs(original, ppo_utils, policy=selection)
    return {"event": "loss_memory_policy", "policy": selection,
        "temperature": args.rollout_temperature, "chunk_rows": args.log_probs_chunk_size,
        "installed_loss_source": inspect.getsourcefile(original),
        "installed_chunk_kernel_source": inspect.getsourcefile(ppo_utils._VocabParallelLogProbEntropy),
        "full_logits_dtype": "float32", "full_backward_gradient_storage": "consumed_logits",
        "saved_softmax_dtype": "float32", "saved_softmax_device": "cpu",
        "entropy_metric_preserved": True, "entropy_gradient_coefficient": 0,
        "sampled_tokens_masks_rewards_reduction_unchanged": True}
