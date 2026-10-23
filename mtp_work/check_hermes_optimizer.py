#!/usr/bin/env python3
"""CPU validation for the explicit AdamW-to-factored-Adafactor migration."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers.optimization import Adafactor

from train_hermes_rollout import (
    ADAFACTOR_NAME,
    ADAMW_NAME,
    validate_optimizer_resume,
)


assert validate_optimizer_resume('AdamW', ADAFACTOR_NAME, True)
assert validate_optimizer_resume(None, ADAFACTOR_NAME, True)
try:
    validate_optimizer_resume('AdamW', ADAFACTOR_NAME, False)
except ValueError:
    pass
else:
    raise AssertionError('optimizer mismatch without explicit reset was accepted')
try:
    validate_optimizer_resume(ADAFACTOR_NAME, ADAFACTOR_NAME, True)
except ValueError:
    pass
else:
    raise AssertionError('reset of unchanged optimizer was accepted')

# Exercise the same optimizer configuration on matrix and vector parameters.
matrix = torch.nn.Parameter(torch.ones(128, 64))
vector = torch.nn.Parameter(torch.ones(64))
optimizer = Adafactor([matrix, vector], lr=2e-5, scale_parameter=False,
                      relative_step=False, warmup_init=False, beta1=None,
                      weight_decay=0.)
(matrix.square().mean() + vector.square().mean()).backward()
optimizer.step()
matrix_state = optimizer.state[matrix]
vector_state = optimizer.state[vector]
assert {'exp_avg_sq_row', 'exp_avg_sq_col'} <= set(matrix_state)
assert 'exp_avg_sq' in vector_state
state_elements = sum(t.numel() for state in optimizer.state.values()
                     for t in state.values() if isinstance(t, torch.Tensor))
assert state_elements == 128 + 64 + 64 + 2  # factored moments plus per-parameter RMS scalars
assert all(torch.isfinite(t).all() for state in optimizer.state.values()
           for t in state.values() if isinstance(t, torch.Tensor))
print({'pass': True, 'migration': 'AdamW -> Adafactor explicit state reset',
       'matrix_state': 'factored row/column', 'vector_state': 'unfactored second moment',
       'state_elements': state_elements})
