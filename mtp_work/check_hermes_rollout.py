# SPDX-License-Identifier: Apache-2.0
"""CPU checks for exact recurrent cache use and the depth-6 acceptance loss."""
import os
from pathlib import Path
import sys
import tempfile

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
os.environ['HERMES_HELPERS'] = str(ROOT / 'scripts')
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / 'scripts'))

from hermes_rollout import (RolloutFeatureStore, distribution_overlap,
                            filtered_topk_probs, label_nll, verifier_rollout_loss)
from mtp_head import MTPHead
from train_hermes_capture import FrozenProjection
from hermes_feature_store import capture_key
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding


class ToyProjection:
    def __init__(self, weight, chunk=23):
        self.w = weight.half()
        self.n_out, self.n_in = self.w.shape
        self.chunk = chunk

    def blocks(self):
        for lo in range(0, self.n_out, self.chunk):
            yield lo, min(self.n_out, lo + self.chunk)

    def weight(self, lo, hi):
        return self.w[lo:hi]

    @torch.no_grad()
    def topk(self, hidden, k=1):
        best = torch.full((len(hidden), k), -torch.inf, device=hidden.device)
        ids = torch.full((len(hidden), k), -1, dtype=torch.long, device=hidden.device)
        for lo, hi in self.blocks():
            values, indices = F.linear(hidden.half(), self.weight(lo, hi)).float().topk(k, -1)
            merged, choice = torch.cat((best, values), -1).topk(k, -1)
            ids = torch.cat((ids, indices + lo), -1).gather(1, choice)
            best = merged
        return best, ids


def main():
    torch.manual_seed(29)
    cfg = Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        layer_types=['full_attention'], vocab_size=137,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                         'partial_rotary_factor': .25},
    )
    cfg._attn_implementation = 'sdpa'
    embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    frozen = FrozenProjection(torch.randn(cfg.vocab_size, cfg.hidden_size).half() * .02)
    head = MTPHead(cfg, embedding, frozen).bfloat16()
    projection = ToyProjection(frozen.weight)

    # Compare sequential KV-cache outputs against one causal full-sequence call.
    batch, depth = 2, 6
    initial = torch.randn(batch, cfg.hidden_size).bfloat16()
    tokens = torch.randint(0, cfg.vocab_size, (batch, depth))
    positions = torch.arange(110, 110 + depth).repeat(batch, 1)
    cache, recurrent, cached_states = DynamicCache(), initial, []
    fused_inputs = []
    with torch.autocast('cpu', dtype=torch.bfloat16):
        for step in range(depth):
            token = tokens[:, 0] if step == 0 else tokens[:, step - 1]
            emb = head.embed_tokens(token)
            x = head.fc(torch.cat((head.pre_fc_norm_embedding(emb),
                                   head.pre_fc_norm_hidden(recurrent)), -1)).unsqueeze(1)
            fused_inputs.append(x)
            state = head.forward_hidden(recurrent, token, positions[:, step], cache)
            cached_states.append(state)
            recurrent = state
        x_all = torch.cat(fused_inputs, dim=1)
        pos_all = positions
        cos, sin = head.rotary(x_all, pos_all)
        causal = torch.full((batch, 1, depth, depth), -torch.inf, dtype=x_all.dtype)
        causal = torch.triu(causal, diagonal=1)
        full = head.layer(x_all, (cos, sin), attention_mask=causal,
                          position_ids=pos_all)
        full = head.norm(full)
    cache_error = max(float((cached_states[i] - full[:, i]).detach().abs().max()) for i in range(depth))
    assert cache_error < 0.02, cache_error
    assert cache.get_seq_length() == depth

    # The streamed per-row NLL and its gradient match a dense vocabulary path.
    h = torch.randn(7, cfg.hidden_size, dtype=torch.bfloat16, requires_grad=True)
    href = h.detach().clone().requires_grad_(True)
    labels = torch.randint(0, cfg.vocab_size, (7,))
    nll = label_nll(h, labels, projection)
    dense = F.cross_entropy(F.linear(href.half(), frozen.weight).float(), labels, reduction='none')
    grad = torch.randn_like(nll)
    (nll * grad).sum().backward()
    (dense * grad).sum().backward()
    nll_error = float((nll - dense).detach().abs().max())
    grad_error = float((h.grad.float() - href.grad.float()).abs().max())
    assert nll_error < 2e-5 and grad_error < 0.002, (nll_error, grad_error)

    # A real six-position loss returns hard greedy acceptance separately from
    # its smooth expected-accepted-length surrogate and propagates finite grads.
    head.zero_grad(set_to_none=True)
    verifier = torch.randn(batch, depth, cfg.hidden_size).half()
    first = torch.randint(0, cfg.vocab_size, (batch,))
    absolute = torch.arange(20, 20 + depth).repeat(batch, 1)
    rollout_tokens = torch.randint(cfg.vocab_size, (batch, depth))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss, metrics = verifier_rollout_loss(
            head, initial, first, verifier, absolute, projection,
            rollout_tokens=rollout_tokens,
        )
    loss.backward()
    assert torch.isfinite(loss) and 0 <= float(metrics['hard_accepted_length']) <= depth
    assert metrics['per_depth_prefix_acceptance'].shape == (depth,)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert float(metrics['per_depth_prefix_acceptance'][1:][-1]) <= float(
        metrics['per_depth_prefix_acceptance'][0]
    )

    with tempfile.TemporaryDirectory() as directory:
        ids = list(range(20, 34))
        record = {'sid': 'rollout-fixture', 'w': 'capture', 'split': 'train',
                  'root_id': 'root-a', 'input_ids': ids, 'spans': [[3, 13]],
                  'n_tgt': 10, 'capture_crop_start': 100}
        hidden = torch.arange(len(ids) * cfg.hidden_size, dtype=torch.float16).reshape(
            len(ids), cfg.hidden_size
        )
        save_file({'input_ids': torch.tensor(ids), 'hidden': hidden},
                  str(Path(directory) / (capture_key(record) + '.safetensors')))
        store = RolloutFeatureStore([record], directory, cfg.hidden_size, depth=6)
        initial_fixture, first_fixture, teacher_fixture, position_fixture = store.batch([0])
        assert len(store) == 1 and store.covered_labels == 6 and store.uncovered_labels == 4
        assert first_fixture.tolist() == [ids[2]]
        assert torch.equal(initial_fixture[0], hidden[1])
        assert torch.equal(teacher_fixture[0], hidden[2:8])
        assert position_fixture[0].tolist() == list(range(102, 108))
        assert torch.equal(store.continuation_batch([0], 'cpu')[0], torch.tensor(ids[3:9]))

    # Exact rejection-sampling overlap is one for identical filtered policies.
    logits = torch.tensor([[5., 4., 3., 2.]])
    probs, keep = filtered_topk_probs(logits, temperature=1., top_p=.95)
    overlap = distribution_overlap(torch.tensor([[9, 8, 7, 6]]), probs, keep,
                                   torch.tensor([[9, 8, 7, 6]]), probs, keep)
    assert torch.allclose(probs.sum(-1), torch.ones(1)) and torch.allclose(overlap, torch.ones(1))
    assert torch.isfinite(metrics['expected_accepted'])
    assert 0 <= float(metrics['expected_accepted']) <= depth
    print({'pass': True, 'depth': depth, 'cache_max_abs_error': cache_error,
           'streamed_nll_max_error': nll_error, 'streamed_nll_grad_max_error': grad_error,
           'hard_accepted_length': float(metrics['hard_accepted_length']),
           'expected_accepted': float(metrics['expected_accepted']),
           'trainable_tensors': len(list(head.parameters())),
           'absolute_position_and_alignment': True})


if __name__ == '__main__':
    main()
