"""Explicit no-model GPU fixture for the previously failing FP32 logits geometry."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--response-chunks', action='store_true',
                        help='Test the opt-in response-chunk policy; never changes a running trainer')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute or args.output.exists():
        parser.error('Requires --execute and fresh output')
    import torch
    from blake3 import blake3
    from loss_memory import calculate, POLICY
    if args.response_chunks:
        from loss_memory import calculate_responses, RESPONSE_POLICY
        selected_policy = RESPONSE_POLICY
    else:
        selected_policy = POLICY
    slime = Path(__file__).resolve().parents[3] / 'slime-upstream'
    sys.path.insert(0, str(slime))
    from slime.utils import ppo_utils as ppo
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('GPU compute process already active; no fixture allocation')
    args.output.mkdir(parents=True, mode=0o700)
    document = {'schema': 'eva.loss-memory-gpu-fixture.v1', 'receipt_id': str(uuid4()),
        'pid': os.getpid(), 'policy': selected_policy, 'status': 'started', 'model_loaded': False,
        'provider_calls': 0, 'optimizer_steps': 0, 'training_credit': 0,
        'rows': 32768, 'vocab': 248320, 'logits_dtype': 'float32', 'model_output_dtype': 'bfloat16',
        'temperature': .8, 'chunk_rows': 256,
        'source': {str(path): blake3(path.read_bytes()).hexdigest() for path in (
            Path(__file__), Path(__file__).with_name('loss_memory.py'),
            Path(ppo.__file__))}}
    def record(name):
        with (args.output / name).open('x') as stream:
            json.dump(document, stream, indent=2)
        (args.output / name).chmod(0o600)
    record('start.json')
    started = time.monotonic()
    try:
        torch.cuda.init()
        free, total = torch.cuda.mem_get_info()
        if free < 100 * 1024**3:
            raise RuntimeError('Insufficient GPU headroom for bounded fixture')
        document.update(free_bytes_before=free, total_bytes=total)
        torch.manual_seed(260910)
        # Small actual-GPU exact native-kernel parity precedes the full fixture.
        small = torch.randn(17, 113, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        other = small.detach().clone().requires_grad_()
        targets = torch.randint(113, (17,), device='cuda')
        weights = torch.linspace(-1, 1, 17, device='cuda').reshape(-1, 1)
        if args.response_chunks:
            small_mask = (torch.arange(17, device='cuda') >= 7) & (torch.arange(17, device='cuda') < 16)
            weights = weights * small_mask[:, None]
        ref, ref_ent = ppo.calculate_log_probs_and_entropy(small.float() / .8, targets, None,
            with_entropy=True, with_entropy_grad=False, chunk_size=7)
        small_kwargs = dict(temperature=.8, chunk_size=7, with_entropy=True,
            kernel=ppo._VocabParallelLogProbEntropy, rank_size=ppo._get_vocab_parallel_rank_size)
        if args.response_chunks:
            actual, ent = calculate_responses(other.float(), targets, None, **small_kwargs,
                total_lengths=[17], response_lengths=[9])
        else:
            actual, ent = calculate(other.float(), targets, None, **small_kwargs)
        a, = torch.autograd.grad(ref, small, weights)
        b, = torch.autograd.grad(actual, other, weights)
        compared_rows = small_mask if args.response_chunks else slice(None)
        torch.testing.assert_close(actual[compared_rows], ref[compared_rows], rtol=0, atol=0)
        torch.testing.assert_close(ent[compared_rows], ref_ent[compared_rows], rtol=0, atol=0)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        document['small_gpu_exact_value_entropy_gradient_parity'] = True
        print('Small GPU exact parity passed; starting full geometry.', flush=True)
        del small, other, ref, actual, ent, ref_ent, a, b, targets, weights
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rows, vocab = document['rows'], document['vocab']
        source = torch.empty((rows, vocab), device='cuda', dtype=torch.bfloat16).normal_().requires_grad_()
        targets = torch.arange(rows, device='cuda', dtype=torch.long) % vocab
        weights = torch.linspace(-1, 1, rows, device='cuda').reshape(-1, 1)
        if args.response_chunks:
            response_start, response_end = rows - 4096 - 1, rows - 1
            weights[:response_start].zero_()
            weights[response_end:].zero_()
        # Entropy's CUDA einsum reduction depends on the row geometry. Match
        # four real 256-row chunks instead of comparing a different 5-row GEMM.
        reference_starts = (28416, 28672, 30720, 32512) if args.response_chunks else (0, 256, 16384, 32512)
        chosen = torch.cat([torch.arange(start, start + 256, device='cuda')
                            for start in reference_starts])
        sample = source[chosen].detach().clone().requires_grad_()
        ref, ref_ent = ppo.calculate_log_probs_and_entropy(sample.float() / .8, targets[chosen], None,
            with_entropy=True, with_entropy_grad=False, chunk_size=256)
        expected_gradient, = torch.autograd.grad(ref, sample, weights[chosen])
        logits = source.float()
        pointer = logits.data_ptr()
        def observe_storage(grad):
            document['full_gradient_reuses_logits_storage'] = grad.data_ptr() == pointer
        logits.register_hook(observe_storage)
        loss_kwargs = dict(temperature=.8, chunk_size=256, with_entropy=True,
            kernel=ppo._VocabParallelLogProbEntropy, rank_size=ppo._get_vocab_parallel_rank_size)
        if args.response_chunks:
            lp, entropy = calculate_responses(logits, targets, None, **loss_kwargs,
                total_lengths=[rows], response_lengths=[4096])
        else:
            lp, entropy = calculate(logits, targets, None, **loss_kwargs)
        torch.testing.assert_close(lp[chosen], ref, rtol=0, atol=0)
        torch.testing.assert_close(entropy[chosen], ref_ent, rtol=0, atol=0)
        gradient, = torch.autograd.grad(lp, source, weights)
        torch.testing.assert_close(gradient[chosen], expected_gradient, rtol=0, atol=0)
        finite = all(bool(torch.isfinite(part).all()) for part in gradient.split(256))
        if args.response_chunks:
            unused_zero = all(bool((part == 0).all()) for span in
                (gradient[:response_start], gradient[response_end:]) for part in span.split(256))
            if not unused_zero:
                raise RuntimeError('Prompt/padding gradient is not zero')
            document.update(response_lengths=[4096], total_lengths=[rows],
                full_unused_gradient_zero=unused_zero, computed_softmax_rows=4352,
                response_row_selection_tested=True)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        if not finite or peak > 90 * 1024**3 or not document.get('full_gradient_reuses_logits_storage'):
            raise RuntimeError('Gradient finiteness or bounded fixture peak failed')
        document.update(status='passed', full_gradient_finite=finite, sampled_rows_exact_gradient_parity=True,
            full_shape_forward_backward_completed=True, peak_allocated_bytes=peak,
            full_shape_reference_rows=chosen.numel(), reference_chunk_geometry=256,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            avoided_full_fp32_gradient_allocation_bytes=rows * vocab * 4,
            saved_softmax_cpu_bytes=(4352 if args.response_chunks else rows) * vocab * 4,
            memory_saver_offload_cycles_tested=False, full_model_training_tested=False)
    except Exception as exc:
        document.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        document['elapsed_seconds'] = time.monotonic() - started
        record('receipt.json')
        print(json.dumps({k: v for k, v in document.items() if k != 'source'}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
