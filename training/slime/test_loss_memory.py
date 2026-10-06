"""Small CPU numerical fixtures execute actual installed kernel/loss source bodies."""
import ast
import __future__
from argparse import Namespace
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import loss_memory

SLIME = Path(__file__).resolve().parents[3] / 'slime-upstream'


def installed_functions():
    """Avoid initializing Megatron/TE; compile unchanged selected CPU-callable bodies."""
    ppo_path = SLIME / 'slime/utils/ppo_utils.py'
    loss_path = SLIME / 'slime/backends/megatron_utils/loss.py'
    names = {'_maybe_all_reduce', '_get_vocab_parallel_rank_size', '_LogProbSoftmaxWorkspace',
             '_VocabParallelLogProbEntropy', '_calculate_log_probs_and_entropy_chunk',
             'calculate_log_probs_and_entropy', 'compute_policy_loss'}
    ppo = {'torch': torch, 'dist': dist, 'logger': logging.getLogger(__name__)}
    tree = ast.parse(ppo_path.read_text())
    selected = ast.Module(body=[n for n in tree.body if getattr(n, 'name', None) in names], type_ignores=[])
    # Exercise the actual PPO body eagerly: no compiler/model/GPU initialization.
    for node in selected.body:
        if getattr(node, 'name', None) == 'compute_policy_loss':
            node.decorator_list = []
    exec(compile(selected, str(ppo_path), 'exec', flags=__future__.annotations.compiler_flag), ppo)
    ns = {'torch': torch, 'Namespace': Namespace,
          'mpu': SimpleNamespace(get_context_parallel_world_size=lambda: 1,
                                 get_tensor_model_parallel_group=lambda: None),
          'calculate_log_probs_and_entropy': ppo['calculate_log_probs_and_entropy']}
    names = {'_build_shifted_tokens', '_extract_per_sample', 'get_log_probs_and_entropy'}
    tree = ast.parse(loss_path.read_text())
    selected = ast.Module(body=[n for n in tree.body if getattr(n, 'name', None) in names], type_ignores=[])
    exec(compile(selected, str(loss_path), 'exec', flags=__future__.annotations.compiler_flag), ns)
    return ns['get_log_probs_and_entropy'], SimpleNamespace(**ppo)


@pytest.mark.parametrize('temperature', [0.8, 1.0])
@pytest.mark.parametrize('sizes', [(13, 67, 4), (32, 113, 7)])
def test_installed_response_loss_forward_and_full_parameter_gradient_parity(temperature, sizes):
    torch.manual_seed(100)
    rows, vocab, chunk = sizes
    original, ppo = installed_functions()
    fixed = loss_memory.make_get_log_probs(original, ppo)
    args = Namespace(rollout_temperature=temperature, entropy_coef=0, rollout_top_p=1,
                     log_probs_chunk_size=chunk, allgather_cp=False)
    hidden = torch.randn(rows, 5, requires_grad=True)
    weight = torch.randn(5, vocab, requires_grad=True)
    new_hidden, new_weight = hidden.detach().clone().requires_grad_(), weight.detach().clone().requires_grad_()
    before = (new_hidden.detach().clone(), new_weight.detach().clone())
    lengths = [rows // 2, rows - rows // 2]
    responses = [3, 4]
    tokens = [torch.randint(vocab, (n,), dtype=torch.long) for n in lengths]
    kwargs = dict(args=args, unconcat_tokens=tokens, total_lengths=lengths,
                  response_lengths=responses, with_entropy=True)
    # Model's BF16 projection -> eager FP32 output stays unchanged in the fix.
    ref_logits = (hidden @ weight).to(torch.bfloat16).float().unsqueeze(0)
    new_logits = (new_hidden @ new_weight).to(torch.bfloat16).float().unsqueeze(0)
    _, ref = original(ref_logits, **kwargs)
    _, actual = fixed(new_logits, **kwargs)
    for key in ('log_probs', 'entropy'):
        for a, b in zip(ref[key], actual[key], strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    factors = torch.tensor([0., 1., -2., 0., .7, -1., .3])
    # Arbitrary masked policy derivatives; each selected token retains its exact weight.
    ref_loss = (torch.cat(ref['log_probs']) * factors).sum()
    actual_loss = (torch.cat(actual['log_probs']) * factors).sum()
    ref_gradient = torch.autograd.grad(ref_loss, (hidden, weight))
    actual_gradient = torch.autograd.grad(actual_loss, (new_hidden, new_weight))
    for a, b in zip(ref_gradient, actual_gradient, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(new_hidden, before[0]) and torch.equal(new_weight, before[1])
    assert args.rollout_temperature == temperature


def test_gradient_alias_is_consumed_logits_not_a_full_new_allocation():
    _, ppo = installed_functions()
    original = torch.randn(11, 29, requires_grad=True)
    logits = original.clone()
    tokens = torch.randint(29, (11,))
    lp, _ = loss_memory.calculate(logits, tokens, None, temperature=.8, chunk_size=4, with_entropy=False,
        kernel=ppo._VocabParallelLogProbEntropy, rank_size=ppo._get_vocab_parallel_rank_size)
    gradient, = torch.autograd.grad(lp, logits, torch.ones_like(lp))
    assert gradient.data_ptr() == logits.data_ptr()
    assert torch.isfinite(gradient).all()


def test_unsupported_math_fails_closed_and_unselected_policy_is_noop(monkeypatch):
    monkeypatch.delenv(loss_memory.ENV, raising=False)
    assert loss_memory.install(Namespace()) is None
    original, ppo = installed_functions()
    fixed = loss_memory.make_get_log_probs(original, ppo)
    args = Namespace(rollout_temperature=.8, entropy_coef=.01, rollout_top_p=1)
    with pytest.raises(ValueError, match='metric-only'):
        fixed(torch.randn(1, 5, 11, requires_grad=True), args=args, response_lengths=[3])
    monkeypatch.setenv(loss_memory.ENV, 'not-a-policy')
    with pytest.raises(ValueError, match='unknown'):
        loss_memory.install(Namespace())


@pytest.mark.parametrize('temperature', [.8, 1.0])
@pytest.mark.parametrize('zero_advantages', [False, True])
@pytest.mark.parametrize('rows,chunk,lengths,responses', [
    (32, 8, [32], [8]),           # response starts inside a full-width boundary chunk
    (41, 8, [18, 17], [2, 6]),   # native width 7 (not 8), packed samples and padding
    (13, 4, [13], [1]),          # one token ending exactly on a chunk boundary
    (14, 4, [14], [1]),          # selected short final chunk keeps its original two rows
    (40, 8, [8, 25], [0, 5]),    # empty response in one packed sample
    (20, 7, [3, 5], [2, 1]),     # first row, shared chunk and mostly padding
])
def test_response_chunks_exact_installed_ppo_and_context_parameter_gradients(
        temperature, zero_advantages, rows, chunk, lengths, responses):
    torch.manual_seed(513)
    original, ppo = installed_functions()
    fixed = loss_memory.make_get_log_probs(original, ppo, policy=loss_memory.RESPONSE_POLICY)
    assert fixed._eva_loss_memory_policy == loss_memory.RESPONSE_POLICY
    args = Namespace(rollout_temperature=temperature, entropy_coef=0, rollout_top_p=1,
                     log_probs_chunk_size=chunk, allgather_cp=False)
    hidden = torch.randn(rows, 5, requires_grad=True)
    weight = torch.randn(5, 31, requires_grad=True)
    new_hidden = hidden.detach().clone().requires_grad_()
    new_weight = weight.detach().clone().requires_grad_()
    # Causal coupling ensures earlier prompt states also get response gradients.
    ref_logits = (hidden.cumsum(0) @ weight).to(torch.bfloat16).float().unsqueeze(0)
    actual_logits = (new_hidden.cumsum(0) @ new_weight).to(torch.bfloat16).float().unsqueeze(0)
    kwargs = dict(args=args, unconcat_tokens=[torch.randint(31, (n,)) for n in lengths],
                  total_lengths=lengths, response_lengths=responses, with_entropy=True)
    _, ref = original(ref_logits, **kwargs)
    _, actual = fixed(actual_logits, **kwargs)
    for key in ('log_probs', 'entropy'):
        for a, b in zip(ref[key], actual[key], strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    ref_lp, actual_lp = torch.cat(ref['log_probs']), torch.cat(actual['log_probs'])
    advantages = torch.zeros_like(ref_lp) if zero_advantages else torch.linspace(-1., 1., ref_lp.numel())
    if not zero_advantages and ref_lp.numel() == 1:
        advantages.fill_(1.)
    offsets = torch.zeros_like(ref_lp) if ref_lp.numel() == 1 else torch.linspace(-.3, .3, ref_lp.numel())
    old_lp = ref_lp.detach() + offsets
    ref_loss, ref_clip = ppo.compute_policy_loss(old_lp - ref_lp, advantages, .2, .2)
    actual_loss, actual_clip = ppo.compute_policy_loss(old_lp - actual_lp, advantages, .2, .2)
    torch.testing.assert_close(ref_loss, actual_loss, rtol=0, atol=0)
    torch.testing.assert_close(ref_clip, actual_clip, rtol=0, atol=0)
    # Include a real zero-weight loss-mask entry without dropping its entropy metric.
    mask = torch.ones_like(ref_loss)
    if mask.numel() > 1:
        mask[-1] = 0
    ref_grads = torch.autograd.grad((ref_loss * mask).sum(), (hidden, weight))
    new_grads = torch.autograd.grad((actual_loss * mask).sum(), (new_hidden, new_weight))
    for a, b in zip(ref_grads, new_grads, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    if zero_advantages:
        assert all(torch.count_nonzero(g) == 0 for g in new_grads)
    else:
        assert torch.count_nonzero(new_grads[0][0]) > 0  # prompt-state gradient retained


def test_response_storage_is_compact_and_gradient_reuses_full_logits():
    original, ppo = installed_functions()
    torch.manual_seed(514)
    logits = torch.randn(40, 31, requires_grad=True).clone()
    tokens = torch.randint(31, (40,))
    chunks = loss_memory.response_chunks(40, 8, [40], [5])
    assert chunks == ((32, 40),)
    lp, entropy = loss_memory.calculate_responses(logits, tokens, None,
        temperature=.8, chunk_size=8, with_entropy=False,
        kernel=ppo._VocabParallelLogProbEntropy, rank_size=ppo._get_vocab_parallel_rank_size,
        total_lengths=[40], response_lengths=[5])
    assert entropy is None
    assert lp.grad_fn.backing.shape == (8, 31)
    derivative = torch.zeros_like(lp)
    derivative[34:39] = .7
    grad, = torch.autograd.grad(lp, logits, derivative)
    assert grad.data_ptr() == logits.data_ptr()
    assert torch.count_nonzero(grad[:34]) == 0 and torch.count_nonzero(grad[39:]) == 0
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad[34:39]) > 0


def test_response_gpu_fixture_uses_original_full_size_chunk_geometry():
    chunks = loss_memory.response_chunks(32768, 256, [32768], [4096])
    assert chunks == tuple((start, start + 256) for start in range(28416, 32768, 256))
    assert sum(end - start for start, end in chunks) == 4352
    assert all((start, start + 256) in chunks for start in (28416, 28672, 30720, 32512))


@pytest.mark.parametrize('lengths,responses', [([7], [7]), ([7], [-1]), ([42], [1]),
                                              ([7], [0]), ([7, 5], [1])])
def test_response_layout_rejects_unsupported_or_unaligned_ranges(lengths, responses):
    with pytest.raises(ValueError):
        loss_memory.response_chunks(40, 8, lengths, responses)


def test_v2_allgather_rejected_and_v1_remains_default():
    original, ppo = installed_functions()
    assert loss_memory.make_get_log_probs(original, ppo)._eva_loss_memory_policy == loss_memory.POLICY
    fixed = loss_memory.make_get_log_probs(original, ppo, policy=loss_memory.RESPONSE_POLICY)
    args = Namespace(rollout_temperature=.8, entropy_coef=0, rollout_top_p=1, allgather_cp=True)
    with pytest.raises(ValueError, match='non-allgather'):
        fixed(torch.randn(1, 5, 11, requires_grad=True), args=args, response_lengths=[3])
