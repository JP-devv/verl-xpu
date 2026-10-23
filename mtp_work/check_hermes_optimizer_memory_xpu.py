#!/usr/bin/env python3
"""Measure AdamW vs Adafactor state/peak on the exact native-head shapes."""
import gc
import json

import torch
from transformers.optimization import Adafactor


SHAPES = (
    [(5120, 10240)]                 # fusion projection
    + [(5120, 5120)] * 4           # Q/K/V/O
    + [(17408, 5120)] * 2          # MLP gate/up
    + [(5120, 17408)]              # MLP down
    + [(5120,)] * 7                # RMSNorm weights
)
EXPECTED_PARAMETERS = 424_699_392


def state_bytes(optimizer):
    return sum(value.numel() * value.element_size()
               for state in optimizer.state.values()
               for value in state.values() if torch.is_tensor(value))


def measure(name, params):
    if name == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=2e-5, foreach=False)
    else:
        optimizer = Adafactor(params, lr=2e-5, scale_parameter=False,
                              relative_step=False, warmup_init=False, beta1=None,
                              weight_decay=0.)
    for param in params:
        param.grad = torch.zeros_like(param)
    torch.xpu.synchronize()
    before = torch.xpu.memory_allocated(0)
    torch.xpu.reset_peak_memory_stats(0)
    optimizer.step()
    torch.xpu.synchronize()
    return {'optimizer': name, 'state_bytes': state_bytes(optimizer),
            'allocated_before_step': before,
            'allocated_after_step': torch.xpu.memory_allocated(0),
            'peak_allocated': torch.xpu.max_memory_allocated(0)}


def main():
    if not torch.xpu.is_available():
        raise SystemExit('Intel XPU required')
    torch.xpu.set_device(0)
    n = sum(torch.tensor(shape).prod().item() for shape in SHAPES)
    assert n == EXPECTED_PARAMETERS, (n, EXPECTED_PARAMETERS)
    results = []
    for optimizer_name in ('AdamW', 'Adafactor'):
        params = [torch.nn.Parameter(torch.zeros(shape, device='xpu:0', dtype=torch.bfloat16))
                  for shape in SHAPES]
        torch.xpu.synchronize()
        result = measure(optimizer_name, params)
        results.append(result)
        del params
        gc.collect()
        torch.xpu.empty_cache()
    adamw, adafactor = results
    reduction = adamw['state_bytes'] - adafactor['state_bytes']
    assert reduction > 1_500_000_000, results
    print(json.dumps({'pass': True, 'device': torch.xpu.get_device_name(0),
                      'parameters': n, 'dtype': 'bfloat16', 'results': results,
                      'optimizer_state_reduction_bytes': reduction,
                      'scope': 'synthetic exact-shape optimizer allocation; not a training step'},
                     indent=2))


if __name__ == '__main__':
    main()
