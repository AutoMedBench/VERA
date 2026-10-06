"""Record real full-parameter Megatron optimizer updates for the validation run."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

import torch


EXPECTED_LANGUAGE_PARAMETERS = 8_953_803_264


def _write(event: dict) -> None:
    event['created_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    path = Path(os.environ['EVA_OPTIMIZER_RECEIPT'])
    with path.open('a') as stream:
        stream.write(json.dumps(event, sort_keys=True) + '\n')
    print('EVA_FULL_PARAMETER ' + json.dumps(event, sort_keys=True), flush=True)


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    """Attach an observation hook; never change the loss or optimizer result."""
    from loss_memory import install
    memory_policy = install(args)
    if memory_policy is not None:
        _write(memory_policy)
    optimizer._eva_step_identity = (rollout_id, step_id)
    if getattr(optimizer, '_eva_observer_installed', False):
        return
    parameters = []
    seen = set()
    frozen_count = 0
    for part in model:
        for name, parameter in part.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if parameter.requires_grad:
                if 'lora' in name.lower() or 'adapter' in name.lower():
                    raise RuntimeError('Adapter parameter found in full-parameter validation')
                parameters.append((name, parameter))
            else:
                frozen_count += parameter.numel()
    trainable_count = sum(p.numel() for _, p in parameters)
    if trainable_count != EXPECTED_LANGUAGE_PARAMETERS:
        raise RuntimeError(f'Full language parameter count mismatch: {trainable_count}')
    _write({
        'event': 'optimizer_parameter_inventory',
        'trainable_parameters': trainable_count,
        'trainable_tensors': len(parameters),
        'frozen_auxiliary_parameters': frozen_count,
        'policy': 'all_language_model_parameters_no_lora_no_adapters',
        'optimizer_type': type(optimizer).__name__,
        'training_backend': 'Megatron',
        'slime_loss_type': args.loss_type,
    })
    original_step = optimizer.step

    def observed_step(*step_args, **step_kwargs):
        samples = []
        with torch.no_grad():
            for name, parameter in parameters:
                flat = parameter.detach().reshape(-1)
                stride = max(1, flat.numel() // 128)
                indices = torch.arange(0, flat.numel(), stride, device=flat.device)[:128]
                samples.append((name, parameter, indices, flat[indices].float().clone()))
        started = time.monotonic()
        result = original_step(*step_args, **step_kwargs)
        torch.cuda.synchronize()
        changed_tensors = 0
        changed_values = 0
        max_delta = 0.0
        with torch.no_grad():
            for name, parameter, indices, before in samples:
                after = parameter.detach().reshape(-1)[indices].float()
                difference = (after - before).abs()
                count = int(torch.count_nonzero(difference).item())
                changed_values += count
                changed_tensors += int(count > 0)
                max_delta = max(max_delta, float(difference.max().item()))
        success = bool(result[0])
        grad_norm = None if result[1] is None else float(result[1])
        finite_gradient = grad_norm is not None and math.isfinite(grad_norm) and grad_norm >= 0
        allow_zero_gradient = args.loss_type != 'sft_loss' and args.advantage_estimator == 'grpo'
        rollout, step = optimizer._eva_step_identity
        event = {
            'event': 'optimizer_step',
            'rollout_id': rollout,
            'step_id': step,
            'successful_update': success,
            'gradient_norm': grad_norm,
            'finite_nonzero_gradient': grad_norm is not None and math.isfinite(grad_norm) and grad_norm > 0,
            'finite_gradient': finite_gradient,
            'learning_update_observed': finite_gradient and grad_norm > 0 and changed_values > 0,
            'zero_gradient_no_learning_signal': finite_gradient and grad_norm == 0,
            'trainable_parameters': trainable_count,
            'sampled_changed_tensors': changed_tensors,
            'sampled_changed_values': changed_values,
            'sampled_max_abs_delta': max_delta,
            'optimizer_seconds': time.monotonic() - started,
            'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        }
        _write(event)
        if not success or not finite_gradient or (not allow_zero_gradient and not event['finite_nonzero_gradient']):
            raise RuntimeError('Full-parameter optimizer update was skipped or had invalid gradients')
        return result

    optimizer.step = observed_step
    optimizer._eva_observer_installed = True
